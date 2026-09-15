# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Click interface for authoring, reviewing, and running distillation recipes.

The group registers on the training entry point as ``nvalchemi-training
distill`` — and on the ``nvalchemi-distill`` alias — beside the ``train`` and
``finetune`` groups it mirrors. A recipe is one JSON file validated by
:class:`DistillationJobSpec`: where the teacher and the student come from, what
data they see, the strategy bundle, the on-policy segment loop when there is
one, and the acceptance bars an evaluation is read against.

The student tiers the scaffold offers are size templates and nothing more —
``small``, ``base``, and ``large`` name a width and a depth for whatever
architecture the recipe points ``student.spec`` at, because a distillation
recipe is about the size of the student rather than its family.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from contextlib import ExitStack
from pathlib import Path
from typing import Annotated, Any, Literal, Self, TypeAlias

import click
import torch
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator
from rich import box
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from torch import nn

from nvalchemi._serialization import _import_callable
from nvalchemi.hooks._context import TrainContext
from nvalchemi.training import _spec_utils as strategy_spec
from nvalchemi.training import load_checkpoint
from nvalchemi.training._checkpoint import _strategy_metadata_path
from nvalchemi.training._spec import create_model_spec
from nvalchemi.training._stages import TrainingStage
from nvalchemi.training.cli import (
    DatasetSpec,
    MaceSourceOptions,
    OutputSpec,
    RuntimeHookSpec,
    SourceSpec,
    ValidationSpec,
    _attach_validation_config,
    _build_checked_hook,
    _build_dataloader,
    _build_supported_source_model,
    _dataset_device,
    _path_exists,
    _primary_strategy_device,
    _resolve_distributed_enabled,
    _setup_distributed_manager,
    _write_or_print,
    console,
)
from nvalchemi.training.distillation.config import (
    OnPolicyConfig,
    OnPolicyKnobs,
    _on_policy_knobs,
)
from nvalchemi.training.distillation.evaluation import (
    AcceptanceThresholds,
    StudentEvaluation,
    build_acceptance_report,
    evaluate_accuracy,
    measured_bars,
)
from nvalchemi.training.distillation.evaluation.accuracy import AccuracyQuantity
from nvalchemi.training.distillation.replay import _batch_allocation, _same_device
from nvalchemi.training.distillation.scoring import (
    SUPPORTED_SIGNALS,
    signal_for_field,
)
from nvalchemi.training.distillation.seeding import _SeedSourceSpec
from nvalchemi.training.distillation.strategy import DistillationStrategy
from nvalchemi.training.distributed import get_rank, get_world_size
from nvalchemi.training.hooks.checkpoint import CheckpointHook
from nvalchemi.training.hooks.ddp import DDPHook
from nvalchemi.training.hooks.ema import EMAHook
from nvalchemi.training.losses.composition import (
    ComposedLossFunction,
    loss_component_to_spec,
)
from nvalchemi.training.losses.terms import EnergyMSELoss, ForceMSELoss
from nvalchemi.training.optimizers import OptimizerConfig

DistillationMode: TypeAlias = Literal["offline", "on-policy"]
StudentTier: TypeAlias = Literal["small", "base", "large"]

_STUDENT_TIERS: dict[str, dict[str, int]] = {
    "small": {"hidden_dim": 64, "num_layers": 2, "num_radial": 8},
    "base": {"hidden_dim": 128, "num_layers": 3, "num_radial": 8},
    "large": {"hidden_dim": 256, "num_layers": 4, "num_radial": 12},
}
"""Size templates a scaffold writes into the student spec, by tier name."""

_TIERS: tuple[StudentTier, ...] = ("small", "base", "large")

_CHECKPOINT_HOOK_PATH = f"{CheckpointHook.__module__}.{CheckpointHook.__qualname__}"
"""Hook class a recipe attaches for output.checkpoint_dir to be written at all."""

_EMA_HOOK_PATH = f"{EMAHook.__module__}.{EMAHook.__qualname__}"
"""Hook class a recipe attaches for the weights it trains to be an average."""

_SCAFFOLD_CHECKPOINTS = 10
"""Restart checkpoints a scaffolded run spreads over its step budget."""

_SCAFFOLD_BATCH_SIZE = 8
"""Samples per training batch a scaffolded recipe records."""

_DATASET_FORMATS = ("alchemi-zarr", "alchemi-zarr-multidataset")
"""Loader families a recipe's dataset.format may name."""

_RECIPE_SOURCES = ("mace", "aimnet2", "native-checkpoint")
"""Model families a recipe loads a teacher or a student from."""

_DISTILL_EPILOG = (
    "A recipe is one JSON file: teacher, student, data, strategy, and — for "
    "on-policy runs — the segment loop. Author it with `distill init`, review "
    "it with `distill spec report`, start it with `distill spec run`, pick an "
    "interrupted run up with `distill spec resume`, and gate the result with "
    "`distill evaluate`.\n\n"
    "Examples:\n\n"
    "Scaffold an offline recipe against a teacher-labeled store:\n\n"
    "  nvalchemi-training distill init --tier small --teacher-model mace --teacher-id small-0b --dataset data/labeled.zarr --output-dir runs/distill --out recipe.json\n\n"
    "Review and then run it:\n\n"
    "  nvalchemi-training distill spec report recipe.json\n\n"
    "  nvalchemi-training distill spec run recipe.json\n\n"
    "Score the trained student against a holdout:\n\n"
    "  nvalchemi-training distill evaluate recipe.json --student-checkpoint runs/distill/checkpoints\n"
)


class EvaluationSpec(BaseModel):
    """Holdout set and acceptance bars a recipe is gated on.

    ``EvaluationSpec`` is the optional ``evaluation`` member of
    :class:`DistillationJobSpec`, read by ``distill evaluate`` rather than by
    the run itself: a recipe therefore carries the bars it was meant to clear,
    and gating a trained student is one command against the same file.

    The bars it may carry are
    ``measured_bars("accuracy", accuracy_quantities=quantities)``, because
    scoring a student over a holdout is all ``distill evaluate`` does and an
    accuracy pass fills only the fields of the quantities it compared. A
    stability, throughput, extensivity, RDF, or from-scratch bar needs a
    propagator and a timestep, a supercell builder, or a second trained model,
    none of which a recipe names; a stress bar needs ``"stress"`` among the
    *quantities*. A bar with no measurement behind it fails the student rather
    than passing it, so the recipe is refused at parse time instead of running
    a gate nothing could clear.

    Raises
    ------
    ValueError
        If ``thresholds`` sets a bar the holdout pass over ``quantities`` does
        not measure.

    Examples
    --------
    ::

        EvaluationSpec(
            holdout_path="data/holdout.zarr",
            targets="teacher",
            thresholds={"max_forces_mae": 0.05},
        )
    """

    model_config = ConfigDict(extra="forbid")

    holdout_path: Annotated[
        str,
        Field(description="Held-out dataset the student is scored over."),
    ]
    targets: Annotated[
        Literal["reference", "teacher"],
        Field(
            default="teacher",
            description=(
                "Whether errors are measured against the holdout's own labels "
                "or against the teacher's."
            ),
        ),
    ] = "teacher"
    quantities: list[AccuracyQuantity] = Field(
        default_factory=lambda: ["energy", "forces"],
        description="Quantities the accuracy evaluation compares.",
    )
    batch_size: Annotated[
        int | None,
        Field(default=None, ge=1, description="Batch size of the holdout loader."),
    ] = None
    thresholds: AcceptanceThresholds = Field(
        default_factory=AcceptanceThresholds,
        description="Acceptance bars the verdict is formed against.",
    )

    @model_validator(mode="after")
    def _validate_measurable_thresholds(self) -> Self:
        """Refuse the bars `distill evaluate` has no measurement to fill."""
        measurable = measured_bars("accuracy", accuracy_quantities=self.quantities)
        unmeasurable = sorted(
            set(self.thresholds.model_dump(exclude_defaults=True)) - measurable
        )
        if unmeasurable:
            raise ValueError(
                f"evaluation.thresholds sets {unmeasurable}, which `distill "
                "evaluate` does not measure: it scores the student over the "
                f"holdout on quantities {list(self.quantities)} and fills the "
                f"accuracy bars {sorted(measurable)} only, so any other bar "
                "would fail the student on a number nobody took. A bar reading "
                "a quantity nothing compared is measured once that quantity is "
                "added to evaluation.quantities; the rest need a propagator and "
                "a timestep, a supercell builder, or a second trained model, "
                "none of which a recipe carries: measure them with "
                "StabilityMonitor, measure_throughput, and extensivity_error, "
                "then assemble one report from their to_dict() exports with "
                "build_acceptance_report."
            )
        return self


class StudentSpec(BaseModel):
    """Where the student comes from, and at what size.

    ``StudentSpec`` is the ``student`` member of :class:`DistillationJobSpec`.
    A student is normally constructed rather than loaded, so the common form is
    ``spec``: a ``{"cls_path": ..., "kwargs": {...}}`` reference naming the
    constructor and the arguments it is called with, the same shape the
    segment loop names its propagator by. ``tier`` records which size template
    those arguments came from, purely so a report and a sweep can say which
    tier a run belongs to; it selects a size, never an architecture.
    ``source`` loads a student from a checkpoint instead, for a run that
    continues from existing weights.

    Examples
    --------
    ::

        StudentSpec(
            tier="small",
            spec={"cls_path": "my_package.MyMLIP", "kwargs": {"hidden_dim": 64}},
        )
    """

    model_config = ConfigDict(extra="forbid")

    tier: Annotated[
        StudentTier | None,
        Field(description="Size template the student's arguments came from."),
    ] = None
    spec: Annotated[
        dict[str, Any] | None,
        Field(
            description=(
                "Constructor reference building the student, as cls_path plus kwargs."
            )
        ),
    ] = None
    source: Annotated[
        SourceSpec | None,
        Field(description="Checkpoint or supported wrapper to load the student from."),
    ] = None
    hooks: list[RuntimeHookSpec] = Field(
        default_factory=list,
        description=(
            "Runtime hooks attached to the training strategy, serialized as "
            "BaseSpec JSON objects. Attached at execution time rather than "
            "stored in the strategy bundle."
        ),
    )

    @model_validator(mode="after")
    def _validate_student_source(self) -> Self:
        """Require exactly one way to obtain the student, named the way it builds."""
        if (self.spec is None) == (self.source is None):
            raise ValueError(
                "student needs exactly one of spec or source: spec constructs a "
                "fresh student, source loads one from a checkpoint."
            )
        if self.spec is not None and "cls_path" not in self.spec:
            raise ValueError(
                "student.spec names the constructor by cls_path, with its "
                "arguments under kwargs."
            )
        return self


class DistillationJobSpec(BaseModel):
    """Top-level envelope describing one distillation recipe.

    The file ``distill spec report`` reads and ``distill spec run`` executes.
    ``mode`` selects offline distillation over a teacher-labeled store or the
    on-policy segment loop; ``teacher`` and ``student`` say where the two models
    come from; ``dataset`` names the training store — the labeled dataset
    offline, the anchor on-policy; ``strategy`` is the JSON-ready
    :meth:`~nvalchemi.training.distillation.DistillationStrategy.to_spec_dict`
    bundle carrying optimizers, loss, devices, and duration; ``on_policy`` is
    the segment-loop recipe an on-policy run needs; and ``evaluation`` records
    the bars ``distill evaluate`` gates on.

    Examples
    --------
    A minimal offline recipe:

    .. code-block:: json

        {
          "mode": "offline",
          "teacher": {"model": "mace", "model_id": "small-0b"},
          "student": {"tier": "small", "spec": {"cls_path": "my_package.MyMLIP"}},
          "dataset": {"path": "data/labeled.zarr"},
          "output": {"run_dir": "runs/distill"},
          "strategy": {"...": "DistillationStrategy.to_spec_dict()"}
        }

    Notes
    -----
    Validation is pre-flight in the strict sense: the strategy bundle is
    deserialized with the same helpers the runtime uses, and an ``on_policy``
    recipe goes through
    :class:`~nvalchemi.training.distillation.OnPolicyConfig`'s own field
    constraints rather than a second copy of them, so a knob out of range or a
    key the segment loop needs fails at ``spec report`` rather than after a
    teacher has been loaded onto a GPU. Its ``seeds`` block goes through the
    very description
    :meth:`~nvalchemi.training.distillation.SeedSource.from_spec_dict` rebuilds
    through, so a budget that is not a positive count, a budget spelled
    wrongly, and a block naming no store fail there too. What it cannot check
    without building
    models — that the loss's teacher targets are signals the teacher can
    produce, or that a propagator's ``cls_path`` imports — the strategy's own
    constructor checks at ``spec run``, and the CLI reports it as a clean error
    rather than a traceback.

    ``mode`` is the single source of truth for which loop runs. The strategy
    bundle a Python-side
    :meth:`~nvalchemi.training.distillation.DistillationStrategy.to_spec_dict`
    produces carries its own ``on_policy`` and ``reference_dataset`` entries;
    an offline recipe carrying either is rejected rather than quietly rebuilding
    the segment loop it says it is not running, and in on-policy mode the
    top-level ``on_policy`` block is the one that is built.
    """

    model_config = ConfigDict(extra="forbid")

    name: Annotated[str, Field(description="Human-readable recipe name.")] = (
        "distillation-job"
    )
    mode: Annotated[
        DistillationMode,
        Field(description="Offline distillation, or the on-policy segment loop."),
    ]
    teacher: Annotated[SourceSpec, Field(description="Where the teacher comes from.")]
    student: Annotated[StudentSpec, Field(description="Where the student comes from.")]
    dataset: Annotated[
        DatasetSpec,
        Field(description="Training store: the labeled dataset, or the anchor."),
    ]
    output: Annotated[OutputSpec, Field(description="Output path intent.")]
    validation: Annotated[
        ValidationSpec | None,
        Field(description="Optional validation cadence for CLI execution."),
    ] = None
    on_policy: Annotated[
        dict[str, Any] | None,
        Field(
            description=(
                "Segment-loop recipe produced by OnPolicyConfig.to_spec_dict(). "
                "Required by, and only read in, on-policy mode."
            )
        ),
    ] = None
    evaluation: Annotated[
        EvaluationSpec | None,
        Field(description="Holdout and acceptance bars for `distill evaluate`."),
    ] = None
    strategy: Annotated[
        dict[str, Any],
        Field(
            description=(
                "JSON-ready bundle produced by DistillationStrategy.to_spec_dict()."
            )
        ),
    ]
    notes: Annotated[
        str | None,
        Field(description="Optional notes rendered in the report."),
    ] = None

    @model_validator(mode="after")
    def _validate_mode(self) -> Self:
        """Require the segment-loop recipe exactly when the mode asks for one."""
        if self.mode == "on-policy" and self.on_policy is None:
            raise ValueError(
                "on-policy recipes need an on_policy block: the segment loop "
                "generates its own batches and has no dataloader to fall back "
                "on."
            )
        if self.mode == "offline" and self.on_policy is not None:
            raise ValueError(
                "offline recipes train on the dataset they name, so an "
                "on_policy block would never be read; set mode='on-policy' to "
                "use it."
            )
        if self.mode == "offline":
            bundled = [
                key
                for key in ("on_policy", "reference_dataset")
                if self.strategy.get(key) is not None
            ]
            if bundled:
                raise ValueError(
                    f"strategy carries {bundled} while mode='offline'. The "
                    "bundle a DistillationStrategy.to_spec_dict() of an "
                    "on-policy run produces rebuilds the segment loop, so an "
                    "offline run would either train the loop the mode says it "
                    "is not running or fail in the strategy's own constructor; "
                    "drop the entries, or set mode='on-policy' and lift the "
                    "recipe to the top-level on_policy block."
                )
        return self

    @model_validator(mode="after")
    def _validate_strategy(self) -> Self:
        """Deserialize the strategy bundle with the runtime's own helpers."""
        missing = [
            key
            for key in ("optimizer_configs", "devices", "loss_fn_spec")
            if key not in self.strategy
        ]
        if missing:
            raise ValueError(
                f"strategy is missing required DistillationStrategy spec key(s) "
                f"{missing}."
            )
        num_epochs = self.strategy.get("num_epochs")
        num_steps = self.strategy.get("num_steps")
        if (num_epochs is None) == (num_steps is None):
            raise ValueError(
                "strategy must set exactly one of num_epochs or num_steps."
            )
        budget = num_steps if num_epochs is None else num_epochs
        if budget < 1:
            raise ValueError(
                f"strategy.{'num_steps' if num_epochs is None else 'num_epochs'} "
                f"sizes the run and must be at least 1; got {budget!r}."
            )
        if self.mode == "on-policy" and num_steps is None:
            raise ValueError(
                "on-policy distillation is sized in optimizer steps, because "
                "every segment builds its own loader; set strategy.num_steps."
            )
        optimizers = strategy_spec._optimizer_configs_from_spec(
            self.strategy["optimizer_configs"]
        )
        if "teacher" in optimizers:
            raise ValueError(
                "the teacher is frozen by omission, so strategy."
                "optimizer_configs must not configure it."
            )
        if "student" not in optimizers:
            raise ValueError(
                "strategy.optimizer_configs must configure the student; got "
                f"{sorted(optimizers)!r}."
            )
        strategy_spec._devices_from_spec(self.strategy["devices"])
        strategy_spec._loss_fn_from_spec(self.strategy["loss_fn_spec"])
        strategy_spec._training_fn_from_spec(self.strategy, None)
        if self.validation is not None and self.dataset.validation_path is None:
            raise ValueError(
                "validation cadence requires dataset.validation_path to be set."
            )
        if self.dataset.format not in _DATASET_FORMATS:
            raise ValueError(
                f"dataset.format {self.dataset.format!r} is not a format the "
                f"loader builds; supported formats: {sorted(_DATASET_FORMATS)}."
            )
        return self

    @model_validator(mode="after")
    def _validate_sources(self) -> Self:
        """Require of each model source what the wrapper building it needs.

        The rules are the ones
        :meth:`~nvalchemi.training.cli.TrainingJobSpec._validate_workflow_source`
        applies to a fine-tune source, because a recipe obtains both of its
        models the same way: every source is pretrained, and none of them is
        the from-scratch case ``student.spec`` covers.
        """
        sources = [("teacher", self.teacher)]
        if self.student.source is not None:
            sources.append(("student.source", self.student.source))
        for field, source in sources:
            if source.model not in _RECIPE_SOURCES:
                raise ValueError(
                    f"{field}.model={source.model!r} is not a source a recipe "
                    f"builds from; name one of {sorted(_RECIPE_SOURCES)!r}, or "
                    "— for the student — construct it from student.spec."
                )
            if source.model != "mace" and (source.model_extra or {}).get("mace"):
                raise ValueError(
                    f"{field}.mace options are only valid when {field}.model='mace'."
                )
            if source.model == "mace":
                MaceSourceOptions.from_source(source)
            if source.model == "native-checkpoint" and not source.checkpoint_path:
                raise ValueError(
                    f"native-checkpoint sources require {field}.checkpoint_path."
                )
            if not (source.model_id or source.checkpoint_path):
                raise ValueError(
                    f"{source.model} sources require {field}.model_id or "
                    f"{field}.checkpoint_path."
                )
        return self

    @model_validator(mode="after")
    def _validate_on_policy_recipe(self) -> Self:
        """Check the segment-loop recipe the way the config it builds would."""
        if self.on_policy is None:
            return self
        required = ("dynamics", "teacher_scorer", "replay_ratio", "steps_per_segment")
        missing = [key for key in required if key not in self.on_policy]
        if missing:
            raise ValueError(f"on_policy is missing required key(s) {missing}.")
        if "cls_path" not in self.on_policy["dynamics"]:
            raise ValueError(
                "on_policy.dynamics names the propagator by cls_path, with its "
                "constructor arguments under kwargs; the student is bound at "
                "build time and must not be named."
            )
        signals = self.on_policy["teacher_scorer"].get("signals")
        unsupported = sorted(set(signals or ()) - SUPPORTED_SIGNALS)
        if not signals or unsupported:
            raise ValueError(
                "on_policy.teacher_scorer.signals must name teacher signals "
                f"from {sorted(SUPPORTED_SIGNALS)!r}; got {signals!r}."
            )
        try:
            seeds = _SeedSourceSpec.model_validate(self.on_policy.get("seeds") or {})
        except ValidationError as exc:
            raise ValueError(
                "on_policy.seeds names the store the first segment is seeded "
                "from, under a dataset entry giving its path, and the budgets "
                "the batch packed from it is held to. A run may hand the loop "
                "a SeedSource over an in-memory dataset instead, but no recipe "
                f"describes one. The block is invalid: {exc}"
            ) from exc
        try:
            knobs = _on_policy_knobs(self.on_policy)
        except ValidationError as exc:
            raise ValueError(f"on_policy knobs are invalid: {exc}") from exc
        if seeds.recycle and knobs.convergence is None:
            raise ValueError(
                "SeedSource.recycle restarts a backfill that has reached the "
                "end of the seed rows, and only a run managing a trajectory "
                "lifecycle ever backfills; got it set with convergence=None. "
                "Pass a convergence criterion, or drop the flag."
            )
        self._validate_mixture(knobs)
        devices = strategy_spec._devices_from_spec(self.strategy["devices"])
        if (
            knobs.replay_device is not None
            and devices
            and not _same_device(torch.device(knobs.replay_device), devices[0])
        ):
            raise ValueError(
                "the mixture is collated before training moves it, so the "
                "replay buffer and the anchor have to be staged on one device, "
                "and the CLI loads the anchor on the strategy's own; got "
                f"on_policy.replay_device={str(knobs.replay_device)!r} against "
                f"strategy.devices[0]={str(devices[0])!r}. Leave replay_device "
                "unset to stage the frames where the anchor is loaded."
            )
        return self

    def _validate_mixture(self, knobs: OnPolicyKnobs) -> None:
        """Refuse the top of the ratio, which only a recipe-driven run can refuse.

        Everything else about the mixture — the bottom of the ratio, and the
        rounding that leaves one source out of a batch — is
        :class:`~nvalchemi.training.distillation.OnPolicyKnobs`'s own refusal
        and has already fired by the time this runs, so the allocator is asked
        exactly once and the CLI carries no second copy of its rounding. What
        is left is the one refusal the knobs cannot make: a recipe always names
        an anchor for ``dataset`` to open, so the strategy builds a
        ``reference_dataset`` on every CLI path and ``replay_ratio=1`` is a
        certainty here where it is merely a possibility there.

        Parameters
        ----------
        knobs : OnPolicyKnobs
            Scalar knobs the segment-loop recipe sets, already validated.

        Raises
        ------
        ValueError
            If every sample of every batch is drawn from the replay buffer.
        """
        if knobs.replay_ratio == 1.0:
            raise ValueError(
                "replay_ratio=1 draws every sample of every batch from the "
                "replay buffer, and a recipe always names an anchor for "
                "dataset to open, so the anchor would be policed for schema "
                "and device and then never sampled; lower replay_ratio to mix "
                "it in."
            )

    @classmethod
    def template(
        cls,
        *,
        mode: DistillationMode,
        tier: StudentTier,
        dataset: str,
        output_dir: str,
        teacher_model: str,
        teacher_id: str | None = None,
        teacher_checkpoint: str | None = None,
        student_cls_path: str = "my_package.my_module.MyStudentModel",
        lr: float = 1e-4,
        num_steps: int = 1000,
        batch_size: int = _SCAFFOLD_BATCH_SIZE,
        device: str = "cuda",
        seed_dataset: str | None = None,
        validation_path: str | None = None,
        holdout_path: str | None = None,
    ) -> Self:
        """Build a validated scaffold for a distillation recipe.

        Parameters
        ----------
        mode : {"offline", "on-policy"}
            Which loop the recipe describes.
        tier : {"small", "base", "large"}
            Size template written into the student's constructor arguments.
        dataset : str
            Training store: the teacher-labeled dataset offline, the anchor
            on-policy.
        output_dir : str
            Run directory. ``output/checkpoint_dir`` and the CheckpointHook
            that writes it are scaffolded beneath it, since nothing else in the
            recipe would produce the weights ``distill evaluate`` scores.
        teacher_model : str
            Teacher source family, as in the training CLI.
        teacher_id : str | None, optional
            Teacher model id for a supported wrapper. Default ``None``.
        teacher_checkpoint : str | None, optional
            Teacher checkpoint path. Default ``None``.
        student_cls_path : str, optional
            Dotted path of the student constructor the tier sizes.
        lr : float, optional
            Student learning rate. Default ``1e-4``.
        num_steps : int, optional
            Optimizer steps to run. Default ``1000``.
        batch_size : int, optional
            Samples per training batch, recorded as ``dataset.batch_size``.
            Default ``8``. It sizes the offline training loader and, in either
            mode, the validation loader; left unset the loader falls back to a
            single graph per batch, which under a step budget is the whole run
            seeing eight times less data than the on-policy mixture does.
        device : str, optional
            Strategy device string. Default ``"cuda"``.
        seed_dataset : str | None, optional
            Store the on-policy loop seeds its trajectories from, written into
            the recipe as ``on_policy.seeds.dataset.path``. Required in
            on-policy mode.
        validation_path : str | None, optional
            Validation store. Default ``None``.
        holdout_path : str | None, optional
            Holdout store recorded in the evaluation section. Default ``None``.

        Returns
        -------
        DistillationJobSpec
            Validated scaffold ready to be edited and reported on.

        Raises
        ------
        ValueError
            If *mode* is ``"on-policy"`` and no *seed_dataset* is named. The
            anchor *dataset* names is the store the batch mixture draws its
            reference share from and carries no forces, so it cannot stand in
            for the store the propagator takes its first step from.
        """
        if mode == "on-policy" and seed_dataset is None:
            raise ValueError(
                "on-policy recipes name a seed store of their own under "
                "on_policy.seeds: the propagator reads energy and forces off "
                "the seed batch before the student's first forward, and the "
                "anchor named by dataset carries neither."
            )
        teacher: dict[str, Any] = {"model": teacher_model}
        if teacher_id is not None:
            teacher["model_id"] = teacher_id
        if teacher_checkpoint is not None:
            teacher["checkpoint_path"] = teacher_checkpoint
        dataset_payload: dict[str, Any] = {
            "path": dataset,
            "format": "alchemi-zarr",
            "batch_size": batch_size,
        }
        if validation_path is not None:
            dataset_payload["validation_path"] = validation_path
        checkpoint_dir = str(Path(output_dir) / "checkpoints")
        return cls(
            name=f"{tier}-student-{mode}-distillation",
            mode=mode,
            teacher=teacher,
            student={
                "tier": tier,
                "spec": {
                    "cls_path": student_cls_path,
                    "kwargs": dict(_STUDENT_TIERS[tier]),
                },
                "hooks": [_checkpoint_hook_template(checkpoint_dir, num_steps)],
            },
            dataset=dataset_payload,
            output={"run_dir": output_dir, "checkpoint_dir": checkpoint_dir},
            validation=(None if validation_path is None else {"every_n_epochs": 1}),
            on_policy=(
                None if mode == "offline" else _on_policy_template(seed_dataset, device)
            ),
            evaluation=(
                None
                if holdout_path is None
                else {"holdout_path": holdout_path, "targets": "teacher"}
            ),
            strategy=_default_distillation_strategy_spec(
                lr=lr, num_steps=num_steps, device=device
            ),
        )


def _checkpoint_hook_template(checkpoint_dir: str, num_steps: int) -> dict[str, Any]:
    """Return the runtime CheckpointHook entry a scaffold writes into the student.

    The hook is what makes ``output.checkpoint_dir`` more than a declaration:
    nothing else in a recipe writes weights, so a scaffold without one runs to
    completion and leaves ``distill evaluate`` no checkpoint to score.
    :class:`~nvalchemi.training.CheckpointHook` takes exactly one cadence, and
    the step budget is the one the scaffold already knows. A cadence that the
    budget is not a multiple of is fine here: ``distill spec run`` and ``distill
    spec resume`` close the gap with a terminal checkpoint of their own.
    """
    interval = max(1, num_steps // _SCAFFOLD_CHECKPOINTS)
    spec = create_model_spec(
        CheckpointHook, checkpoint_dir=checkpoint_dir, step_interval=interval
    )
    return {"spec": spec.model_dump(mode="json")}


def _on_policy_template(seed_dataset: str, device: str) -> dict[str, Any]:
    """Return a segment-loop recipe scaffold seeded from *seed_dataset*."""
    return {
        "dynamics": {
            "cls_path": "nvalchemi.dynamics.integrators.nvt_langevin.NVTLangevin",
            "kwargs": {
                "dt": 0.5,
                "temperature": 300.0,
                "friction": 0.01,
                "random_seed": 42,
            },
        },
        "teacher_scorer": {
            "teacher": "teacher",
            "signals": ["energy", "forces"],
            "cast_to": None,
        },
        "seeds": {
            "dataset": {"path": seed_dataset, "device": device},
            "max_atoms": None,
            "max_edges": None,
            "max_batch_size": None,
            "recycle": False,
        },
        "replay_ratio": 0.25,
        "steps_per_segment": 32,
        "batch_size": 8,
        "segment_steps": 50,
        "label_frequency": 10,
        "replay_capacity": 8192,
        "replay_eviction": "fifo",
        "replay_device": None,
        "seed": 0,
        "convergence": None,
        "weight_sync_frequency": 1,
    }


def _default_distillation_strategy_spec(
    *, lr: float, num_steps: int, device: str
) -> dict[str, Any]:
    """Return a strategy bundle matching energies and forces against the teacher."""
    loss_fn = ComposedLossFunction(
        [
            EnergyMSELoss(target_key="teacher_energy"),
            ForceMSELoss(target_key="teacher_forces", normalize_by_atom_count=True),
        ],
        weights=[1.0, 10.0],
        normalize_weights=False,
    )
    loss_fn_spec = create_model_spec(
        type(loss_fn),
        components=[loss_component_to_spec(comp) for comp in loss_fn.components],
        weights=list(loss_fn._weights),
        normalize_weights=loss_fn.normalize_weights,
        dtype_policy=loss_fn.dtype_policy,
    )
    optimizer_config = OptimizerConfig(
        optimizer_cls=torch.optim.AdamW,
        optimizer_kwargs={"lr": lr, "weight_decay": 1e-6},
    )
    return {
        "optimizer_configs": {"student": [optimizer_config.to_spec().model_dump()]},
        "num_epochs": None,
        "num_steps": num_steps,
        "epoch_step_modifier": 1.0,
        "devices": [device],
        "loss_fn_spec": loss_fn_spec.model_dump(),
        "model_specs": {},
        "single_model_input": False,
        "training_fn": (
            "nvalchemi.training.distillation.strategy.default_distillation_fn"
        ),
        "teacher_signals": None,
        "label_missing": True,
    }


def _load_recipe(path: Path) -> DistillationJobSpec:
    """Load and validate a distillation recipe from JSON."""
    try:
        raw = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise click.ClickException(f"Could not parse {path}: {exc}") from exc
    try:
        return DistillationJobSpec.model_validate(raw)
    except ValidationError as exc:
        raise click.ClickException(str(exc)) from exc


def _json_safe(value: Any) -> Any:
    """Return *value* with every non-finite float replaced by its name.

    ``json.dumps`` writes ``NaN``, ``Infinity``, and ``-Infinity`` as bare
    tokens, which are an extension to JSON rather than part of it, so an
    acceptance report carrying a metric that could not be measured would land
    as a file a strict reader rejects. The strings ``"nan"``, ``"inf"``, and
    ``"-inf"`` keep the reason a bar failed visible, where ``null`` would read
    as the measurement never having been taken.
    """
    if isinstance(value, Mapping):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        if math.isnan(value):
            return "nan"
        return "inf" if value > 0 else "-inf"
    return value


def _dataset_store_paths(job: DistillationJobSpec) -> list[str]:
    """Return the training stores a recipe names, by ``paths`` or by ``path``."""
    return list(job.dataset.paths) or ([job.dataset.path] if job.dataset.path else [])


def _recipe_paths(job: DistillationJobSpec) -> list[tuple[str, str]]:
    """Return the local paths a recipe references, keyed by field."""
    checks: list[tuple[str, str | None]] = [
        ("dataset.path", job.dataset.path),
        *(
            (f"dataset.paths[{index}]", value)
            for index, value in enumerate(job.dataset.paths)
        ),
        ("dataset.validation_path", job.dataset.validation_path),
        ("teacher.checkpoint_path", job.teacher.checkpoint_path),
    ]
    if job.student.source is not None:
        checks.append(
            ("student.source.checkpoint_path", job.student.source.checkpoint_path)
        )
    if job.on_policy is not None and job.on_policy.get("seeds"):
        checks.append(
            ("on_policy.seeds.dataset.path", job.on_policy["seeds"]["dataset"]["path"])
        )
    if job.evaluation is not None:
        checks.append(("evaluation.holdout_path", job.evaluation.holdout_path))
    return [(field, value) for field, value in checks if value is not None]


def _derived_teacher_signals(job: DistillationJobSpec) -> list[str]:
    """Return the teacher signals the recipe's loss targets imply."""
    loss_fn = strategy_spec._loss_fn_from_spec(job.strategy["loss_fn_spec"])
    signals = {
        signal
        for component in loss_fn.components
        if (signal := signal_for_field(getattr(component, "target_key", "")))
        is not None
    }
    return sorted(signals)


def _mixture_rows(job: DistillationJobSpec) -> list[tuple[str, str]]:
    """Return the composition of one training batch, as label/value rows."""
    if job.on_policy is None:
        return [("mixture", "every sample from the labeled dataset (offline)")]
    knobs = _on_policy_knobs(job.on_policy)
    anchor, replay = _batch_allocation(knobs.replay_ratio, knobs.batch_size)
    return [
        ("replay_ratio", f"{knobs.replay_ratio:g}"),
        ("batch composition", f"{anchor} anchor + {replay} generated"),
        ("segment", f"{knobs.segment_steps} generated steps"),
        ("label cadence", f"every {knobs.label_frequency} steps"),
        ("training per segment", f"{knobs.steps_per_segment} batches"),
    ]


def _intent_table(job: DistillationJobSpec) -> Table:
    """Build the Rich table summarizing recipe intent."""
    table = Table(title="Distillation intent", box=box.SIMPLE_HEAD, expand=True)
    table.add_column("Area", style="cyan", no_wrap=True)
    table.add_column("Value", overflow="fold")
    table.add_row("recipe", job.name)
    table.add_row("mode", job.mode)
    table.add_row(
        "teacher",
        f"{job.teacher.model} ({job.teacher.model_id or job.teacher.checkpoint_path})",
    )
    student = job.student
    table.add_row("student tier", student.tier or "not specified")
    table.add_row(
        "student",
        (student.spec or {}).get("cls_path", "")
        if student.spec is not None
        else f"{student.source.model} ({student.source.checkpoint_path})",
    )
    table.add_row("teacher signals", ", ".join(_derived_teacher_signals(job)))
    table.add_row(
        "dataset", f"{', '.join(_dataset_store_paths(job))} ({job.dataset.format})"
    )
    table.add_row("validation", job.dataset.validation_path or "none")
    table.add_row("batch size", str(job.dataset.batch_size))
    table.add_row("run dir", job.output.run_dir)
    table.add_row("num_steps", str(job.strategy.get("num_steps")))
    table.add_row("num_epochs", str(job.strategy.get("num_epochs")))
    table.add_row("devices", ", ".join(map(str, job.strategy.get("devices", []))))
    for label, value in _mixture_rows(job):
        table.add_row(label, value)
    return table


def _threshold_table(job: DistillationJobSpec) -> Table | None:
    """Build the acceptance-bar table, or ``None`` when the recipe sets none."""
    if job.evaluation is None:
        return None
    bars = job.evaluation.thresholds.model_dump(exclude_none=True)
    table = Table(title="Acceptance bars", box=box.SIMPLE_HEAD, expand=True)
    table.add_column("Bar", style="cyan", no_wrap=True)
    table.add_column("Value", overflow="fold")
    table.add_row("holdout", job.evaluation.holdout_path)
    table.add_row("targets", job.evaluation.targets)
    table.add_row("quantities", ", ".join(job.evaluation.quantities))
    for name, value in sorted(bars.items()):
        table.add_row(name, str(value))
    return table


def _has_checkpoint_hook(job: DistillationJobSpec) -> bool:
    """Return whether a runtime hook writes into ``output.checkpoint_dir``.

    Any other hook leaves ``output.checkpoint_dir`` unwritten, and so does a
    :class:`~nvalchemi.training.CheckpointHook` pointed at another directory,
    so the destination is matched alongside the class: a run whose hook writes
    elsewhere finishes cleanly, never creates the directory the recipe names,
    and leaves ``distill evaluate`` nothing to read there.
    """
    target = Path(job.output.checkpoint_dir or "")
    return any(
        hook.spec.cls_path == _CHECKPOINT_HOOK_PATH
        and Path(str((hook.spec.model_extra or {}).get("checkpoint_dir", ""))) == target
        for hook in job.student.hooks
    )


def _ema_hook_specs(job: DistillationJobSpec) -> list[RuntimeHookSpec]:
    """Return the recipe's runtime hooks that average the student's weights.

    Only an ``EMAHook`` publishes an average worth scoring in place of the
    trained weights, so the check matches the class rather than merely counting
    the hooks a recipe declares.
    """
    return [hook for hook in job.student.hooks if hook.spec.cls_path == _EMA_HOOK_PATH]


def _load_evaluated_student(
    job: DistillationJobSpec,
    checkpoint: Path,
    *,
    checkpoint_index: int,
    device: torch.device,
) -> tuple[Any, str]:
    """Load the student weights a recipe is gated on, and name which they are.

    Parameters
    ----------
    job : DistillationJobSpec
        Recipe whose ``student.hooks`` say what the run trained.
    checkpoint : Path
        Native checkpoint directory the trained student is read from.
    checkpoint_index : int
        Index within *checkpoint* to read, ``-1`` for the latest.
    device : torch.device
        The one device the evaluation runs on.

    Returns
    -------
    tuple[Any, str]
        The module to score and a phrase naming whose weights it holds.

    Raises
    ------
    click.ClickException
        If the checkpoint cannot be restored under the recipe's EMA hooks.

    Notes
    -----
    A recipe carrying an ``EMAHook`` trained an average, and the run's own
    validation reads that average rather than the live weights, so the gate
    reads it too. The averaged tensors live in the hook's own checkpoint file
    and are revived by restoring the whole strategy under that hook and
    dispatching :attr:`TrainingStage.SETUP`, which is where the hook rebuilds
    its averaged model and publishes it into ``inference_model``. Only the EMA
    hooks are rebuilt: the recipe's other hooks have no part in scoring, and a
    ``DDPHook`` among them would open a process group inside an evaluation.
    A recipe declaring no such hook loads the student alone, which is all a
    bare :func:`~nvalchemi.training.save_checkpoint` directory holds.
    """
    specs = _ema_hook_specs(job)
    if not specs:
        student = _build_role_model(
            SourceSpec(
                model="native-checkpoint",
                checkpoint_path=str(checkpoint),
                checkpoint_index=checkpoint_index,
            ),
            device=device,
            role="student",
            map_location=str(device),
        )
        return student, "raw"
    try:
        hooks = [_build_checked_hook(spec.spec) for spec in specs]
        strategy = DistillationStrategy.load_checkpoint(
            checkpoint,
            checkpoint_index=checkpoint_index,
            map_location=str(device),
            hooks=hooks,
        )
        ctx = TrainContext(batch=None, models=strategy.models, workflow=strategy)
        for hook in hooks:
            hook(ctx, TrainingStage.SETUP)
    except (ValueError, TypeError, KeyError, FileNotFoundError) as exc:
        raise click.ClickException(
            f"student checkpoint {str(checkpoint)!r} could not be restored "
            f"under the EMAHook the recipe declares at index "
            f"{checkpoint_index!r}: {exc} The averaged weights are the hook's "
            "own state, so the hook has to be the one the run trained with."
        ) from exc
    published = strategy.inference_model
    if isinstance(published, nn.ModuleDict):
        published = published["student"] if "student" in published else None
    if published is None:
        return strategy.models["student"], (
            "raw (the recipe's EMAHook published no averaged student)"
        )
    return published, "ema (student.hooks EMAHook)"


def _stores_a_teacher(checkpoint_dir: str | None) -> bool:
    """Return whether a checkpoint root already holds a teacher of its own.

    A teacher is stored once per checkpoint root, so a second run writing a
    different one into an occupied root is refused at its first checkpoint —
    after ``num_steps // 10`` optimizer steps for a scaffolded recipe. Reading
    the manifest is one JSON load and needs neither model, which is why the
    report can say it up front. A root that cannot be read is left unremarked
    rather than reported as occupied.
    """
    if not checkpoint_dir:
        return False
    try:
        manifest = json.loads((Path(checkpoint_dir) / "manifest.json").read_text())
    except (OSError, json.JSONDecodeError):
        return False
    return "teacher" in (manifest.get("model_references") or {})


def _warning_table(job: DistillationJobSpec) -> Table:
    """Build the table of pre-flight warnings a recipe earns."""
    table = Table(title="Pre-flight", box=box.SIMPLE_HEAD, expand=True)
    table.add_column("Check", style="cyan", no_wrap=True)
    table.add_column("Detail", overflow="fold")
    missing = [
        (field, value) for field, value in _recipe_paths(job) if not _path_exists(value)
    ]
    for field, value in missing:
        table.add_row(field, f"[yellow]missing on disk:[/] {value}")
    if job.output.checkpoint_dir and not _has_checkpoint_hook(job):
        table.add_row(
            "output.checkpoint_dir",
            "[yellow]set with no CheckpointHook writing into it; nothing will "
            "be written[/]",
        )
    if _stores_a_teacher(job.output.checkpoint_dir):
        table.add_row(
            "output.checkpoint_dir",
            "[yellow]already holds a teacher stored once per root; a different "
            "one is refused at the first checkpoint this run writes[/]",
        )
    if job.evaluation is None:
        table.add_row(
            "evaluation",
            "no acceptance bars recorded; `distill evaluate` "
            "will report numbers without a verdict",
        )
    if not table.rows:
        table.add_row("all", "[green]no issues found[/]")
    return table


def _render_report(job: DistillationJobSpec) -> None:
    """Render the Rich report card for a distillation recipe."""
    console.rule(f"[bold]Distillation report: {job.name}")
    console.print(_intent_table(job))
    console.print(_warning_table(job))
    thresholds = _threshold_table(job)
    if thresholds is not None:
        console.print(thresholds)
    if job.notes:
        console.print(Panel(Text(job.notes, overflow="fold"), title="Notes"))


def _build_role_model(
    source: SourceSpec, *, device: Any, role: str, map_location: str | None
) -> Any:
    """Build the model one role of the recipe names."""
    if source.model in {"mace", "aimnet2"}:
        return _build_supported_source_model(source, device=device)
    if source.model != "native-checkpoint":
        raise click.ClickException(
            f"{role} source model {source.model!r} cannot be built by the CLI; "
            "use a supported wrapper, a native checkpoint, or — for the "
            "student — a constructor spec."
        )
    if source.checkpoint_path is None:
        raise click.ClickException(f"{role} native-checkpoint needs checkpoint_path.")
    name = (source.model_extra or {}).get("model_name", role)
    advice = (
        "Check that the directory is one save_checkpoint wrote, and that the "
        "checkpoint index — --checkpoint-index for `distill evaluate` — names "
        "one of the indices it saved."
        if role == "student"
        else f"Set {role}.model_name to a model the checkpoint does hold, or "
        "point checkpoint_path at the run that wrote it."
    )
    try:
        loaded = load_checkpoint(
            source.checkpoint_path,
            checkpoint_index=source.checkpoint_index,
            map_location=map_location or str(device),
            model_names={name},
        )
    except (KeyError, ValueError, TypeError, FileNotFoundError) as exc:
        raise click.ClickException(
            f"{role} checkpoint {source.checkpoint_path!r} could not be read "
            f"for a model named {name!r} at index "
            f"{source.checkpoint_index!r}: {exc}. {advice}"
        ) from exc
    except RuntimeError as exc:
        raise click.ClickException(
            f"{role} checkpoint {source.checkpoint_path!r} could not be placed "
            f"on {map_location or str(device)!r}: {exc} Name a device this host "
            "has with --map-location, or point strategy.devices at one."
        ) from exc
    models = loaded["models"] if isinstance(loaded, Mapping) else loaded.models
    entry = models[name]
    return entry["model"] if isinstance(entry, Mapping) else entry[0]


def _build_student(
    job: DistillationJobSpec, *, device: Any, map_location: str | None
) -> Any:
    """Build the student a recipe constructs or loads."""
    if job.student.source is not None:
        return _build_role_model(
            job.student.source, device=device, role="student", map_location=map_location
        )
    spec = job.student.spec
    try:
        student = _import_callable(spec["cls_path"])(**dict(spec.get("kwargs", {})))
    except Exception as exc:
        raise click.ClickException(
            f"student.spec did not build a model from {spec['cls_path']!r}: {exc}"
        ) from exc
    return student.to(device)


def _reference_dataset(
    job: DistillationJobSpec, stack: ExitStack, *, device: Any
) -> Any:
    """Open the anchor store or stores an on-policy run mixes reference batches from."""
    from nvalchemi.data.datapipes import AtomicDataZarrReader, Dataset, MultiDataset

    paths = _dataset_store_paths(job)
    if not paths:
        raise click.ClickException(
            "dataset names no store for the on-policy anchor; set dataset.path "
            "or dataset.paths."
        )
    datasets = [
        Dataset(stack.enter_context(AtomicDataZarrReader(path)), device=device)
        for path in paths
    ]
    return datasets[0] if len(datasets) == 1 else MultiDataset(*datasets)


def _build_strategy(
    job: DistillationJobSpec,
    stack: ExitStack,
    *,
    hooks: list[Any],
    distributed_manager: Any | None,
    map_location: str | None,
) -> DistillationStrategy:
    """Build the strategy a recipe declares, reporting its own errors cleanly."""
    device = _dataset_device(job, distributed_manager)
    teacher = _build_role_model(
        job.teacher, device=device, role="teacher", map_location=map_location
    )
    student = _build_student(job, device=device, map_location=map_location)
    try:
        on_policy = None
        reference_dataset = None
        if job.on_policy is not None:
            on_policy = OnPolicyConfig.from_spec_dict(
                job.on_policy, student=student, teacher=teacher
            )
            reference_dataset = _reference_dataset(job, stack, device=device)
        strategy = DistillationStrategy.from_spec_dict(
            dict(job.strategy),
            models={"student": student, "teacher": teacher},
            hooks=hooks,
            on_policy=on_policy,
            reference_dataset=reference_dataset,
        )
    except (ValueError, TypeError, KeyError) as exc:
        raise click.ClickException(f"strategy could not be built: {exc}") from exc
    strategy.distributed_manager = distributed_manager
    return strategy


def _build_recipe_hooks(
    job: DistillationJobSpec,
    *,
    enable_ddp: bool = False,
    ddp_backend: str | None = None,
) -> list[Any]:
    """Build the runtime hooks a recipe declares, one per requested stage."""
    hooks: list[Any] = []
    if enable_ddp:
        hooks.append(DDPHook(backend=ddp_backend))
    for hook_spec in job.student.hooks:
        stages = hook_spec.stage_values()
        if not stages:
            hooks.append(_build_checked_hook(hook_spec.spec))
            continue
        for stage in stages:
            hook = _build_checked_hook(hook_spec.spec)
            hook.stage = stage
            hooks.append(hook)
    return hooks


def _execute_strategy(
    job: DistillationJobSpec,
    strategy: DistillationStrategy,
    stack: ExitStack,
    *,
    device: Any,
) -> None:
    """Attach the recipe's validation cadence and drive the loop its mode names."""
    _attach_validation_config(
        strategy,
        job,
        stack,
        device=device,
        batch_size=job.dataset.batch_size,
        prefetch_factor=2,
        num_streams=4,
        use_streams=True,
        pin_memory=False,
        validation_path=None,
        validation_every_epochs=None,
        validation_every_steps=None,
    )
    if job.mode == "on-policy":
        _run_strategy(strategy)
        return
    dataloader = _build_dataloader(
        job,
        stack,
        device=device,
        batch_size=job.dataset.batch_size,
        shuffle=True,
        drop_last=False,
        prefetch_factor=2,
        num_streams=4,
        use_streams=True,
        pin_memory=False,
    )
    _run_strategy(strategy, dataloader)


def _last_checkpointed_step(checkpoint_dir: Path | str) -> int | None:
    """Return the completed-step count the newest checkpoint under *dir* records.

    ``None`` when the directory holds no checkpoint yet, or one written before
    a strategy recorded its counters.
    """
    path = _strategy_metadata_path(Path(checkpoint_dir))
    if not path.is_file():
        return None
    runtime_state = json.loads(path.read_text()).get("runtime_state") or {}
    recorded = runtime_state.get("step_count")
    return None if recorded is None else int(recorded)


def _checkpoint_terminal_state(strategy: DistillationStrategy) -> None:
    """Save the final weights of a run that ended between two scheduled saves.

    :class:`~nvalchemi.training.CheckpointHook` saves on a completed-step
    cadence and never at training end, so a step budget that is not a multiple
    of the interval leaves the last updates in memory only: ``distill
    evaluate`` would score weights the run has already moved past, and
    ``distill spec resume`` would re-train the steps that were dropped. The
    hook writes this one itself, at the next index, so what lands is an
    ordinary latest checkpoint the restore and evaluation paths already read —
    the hook's own state, the EMA average among it, included.

    A hook whose newest checkpoint already records the step the strategy
    finished on writes nothing, which covers both the run that ended on the
    cadence and the resume that had nothing left to train.

    Parameters
    ----------
    strategy : DistillationStrategy
        Strategy that has just finished running.
    """
    for hook in strategy.hooks:
        if not isinstance(hook, CheckpointHook):
            continue
        if hook.rank_zero_only and get_rank(strategy.distributed_manager) != 0:
            continue
        if _last_checkpointed_step(hook.checkpoint_dir) == strategy.step_count:
            continue
        ctx = TrainContext(batch=None, models=strategy.models, workflow=strategy)
        # The run closed the hook's background writer on its way out.
        with hook:
            hook._save_checkpoint(ctx)


def _run_strategy(strategy: DistillationStrategy, *args: Any) -> None:
    """Drive the loop, reporting the strategy's own contract errors cleanly.

    The strategy decides which loop it runs from what it was built with, so a
    recipe whose ``mode`` disagrees with the checkpoint ``distill spec resume``
    restored is refused here rather than by a second copy of the rule. The
    wrapper spans the whole run rather than its opening, so what it reports is
    a run that failed rather than one that never started. A run that finishes
    leaves its terminal state checkpointed, which the cadence alone does not
    guarantee.
    """
    try:
        strategy.run(*args)
    except ValueError as exc:
        raise click.ClickException(f"the run failed: {exc}") from exc
    _checkpoint_terminal_state(strategy)


def _run_recipe(
    job: DistillationJobSpec,
    *,
    distributed: bool | None,
    ddp_backend: str | None,
    map_location: str | None,
) -> None:
    """Build the runtime components of a recipe and run it."""
    distributed_enabled = _resolve_distributed_enabled(distributed)
    distributed_manager = _setup_distributed_manager(distributed_enabled)
    hooks = _build_recipe_hooks(
        job, enable_ddp=distributed_enabled, ddp_backend=ddp_backend
    )
    with ExitStack() as stack:
        strategy = _build_strategy(
            job,
            stack,
            hooks=hooks,
            distributed_manager=distributed_manager,
            map_location=map_location,
        )
        _execute_strategy(
            job, strategy, stack, device=_dataset_device(job, distributed_manager)
        )


def _restart_map_location(
    distributed_manager: Any | None, map_location: str | None
) -> str | None:
    """Return the device a restarting rank loads its checkpoint onto.

    Parameters
    ----------
    distributed_manager : Any | None
        Manager attached to the resumed run, or ``None`` for a single process.
    map_location : str | None
        Device the caller asked for, or ``None`` to take the rank's own.

    Returns
    -------
    str | None
        This rank's device under a multi-rank launch, otherwise *map_location*
        unchanged.

    Raises
    ------
    click.ClickException
        If *map_location* names a device other than this rank's while more than
        one rank is running.

    Notes
    -----
    A checkpoint records the device rank zero was pinned to, and that recording
    is the load location every rank restores against when nothing overrides it,
    so the rank's device has to be named both as the load location and as the
    device the restored strategy is rebuilt on. Naming it here settles both:
    :func:`nvalchemi.training.load_checkpoint` overrides the recorded devices
    with ``map_location`` before it rebuilds the strategy, and then restores
    against the same device.
    """
    if distributed_manager is None or get_world_size(distributed_manager) <= 1:
        return map_location
    device = torch.device(distributed_manager.device)
    if map_location is None:
        return str(device)
    if torch.device(map_location) != device:
        raise click.ClickException(
            f"--map-location {map_location!r} is not this rank's device "
            f"{str(device)!r}. Restoring onto another rank's device leaves part "
            "of the optimizer state there, so the first step fails with "
            "'Tensors of the same index must be on the same device' and the "
            "launch hangs tearing the process group down. Drop --map-location "
            "to take the rank's device."
        )
    return map_location


def _resume_recipe(
    job: DistillationJobSpec,
    checkpoint_dir: Path,
    *,
    checkpoint_index: int,
    distributed: bool | None,
    ddp_backend: str | None,
    map_location: str | None,
) -> None:
    """Restore a checkpointed run and continue it under the recipe that started it."""
    distributed_enabled = _resolve_distributed_enabled(distributed)
    distributed_manager = _setup_distributed_manager(distributed_enabled)
    hooks = _build_recipe_hooks(
        job, enable_ddp=distributed_enabled, ddp_backend=ddp_backend
    )
    load_location = _restart_map_location(distributed_manager, map_location)
    try:
        strategy = DistillationStrategy.load_checkpoint(
            checkpoint_dir,
            checkpoint_index=checkpoint_index,
            map_location=load_location,
            hooks=hooks,
        )
    except (
        ValueError,
        TypeError,
        KeyError,
        FileNotFoundError,
        ImportError,
        AttributeError,
    ) as exc:
        raise click.ClickException(
            f"checkpoint {str(checkpoint_dir)!r} could not be restored: {exc}"
        ) from exc
    if not isinstance(strategy, DistillationStrategy):
        raise click.ClickException(
            f"checkpoint {str(checkpoint_dir)!r} holds a "
            f"{type(strategy).__name__} rather than a DistillationStrategy; "
            "resume it with the group that wrote it."
        )
    strategy.distributed_manager = distributed_manager
    device = (
        _dataset_device(job, distributed_manager)
        if load_location is None
        else torch.device(load_location)
    )
    with ExitStack() as stack:
        _execute_strategy(job, strategy, stack, device=device)


@click.group(name="distill", epilog=_DISTILL_EPILOG)
def distill() -> None:
    """Author, review, run, and gate distillation recipes."""


@distill.group(name="spec")
def distill_spec() -> None:
    """Validate, report on, and execute saved distillation recipes."""


@distill.command("init")
@click.option(
    "--mode",
    type=click.Choice(["offline", "on-policy"]),
    default="offline",
    show_default=True,
    help="Which distillation loop the recipe describes.",
)
@click.option(
    "--tier",
    type=click.Choice(_TIERS),
    default="small",
    show_default=True,
    help="Student size template: width and depth only, never an architecture.",
)
@click.option("--dataset", required=True, help="Teacher-labeled store, or the anchor.")
@click.option("--output-dir", required=True, help="Run output directory.")
@click.option(
    "--teacher-model",
    default="mace",
    show_default=True,
    help="Teacher source family.",
)
@click.option("--teacher-id", default=None, help="Teacher model id.")
@click.option("--teacher-checkpoint", default=None, help="Teacher checkpoint path.")
@click.option(
    "--student-cls-path",
    default="my_package.my_module.MyStudentModel",
    show_default=True,
    help="Dotted path of the student constructor the tier sizes.",
)
@click.option("--lr", type=float, default=1e-4, show_default=True, help="Student LR.")
@click.option(
    "--num-steps",
    type=click.IntRange(min=1),
    default=1000,
    show_default=True,
    help="Optimizer steps.",
)
@click.option(
    "--batch-size",
    type=click.IntRange(min=1),
    default=_SCAFFOLD_BATCH_SIZE,
    show_default=True,
    help="Samples per training batch, recorded as dataset.batch_size.",
)
@click.option("--device", default="cuda", show_default=True, help="Strategy device.")
@click.option(
    "--seed-dataset",
    default=None,
    help="Store the segment loop seeds from; required with --mode on-policy.",
)
@click.option(
    "--validation-dataset", "validation_path", default=None, help="Validation store."
)
@click.option(
    "--holdout-dataset", "holdout_path", default=None, help="Acceptance holdout store."
)
@click.option(
    "--out",
    "output",
    type=click.Path(path_type=Path),
    help="Write the recipe JSON to this file.",
)
def init_recipe(
    mode: DistillationMode,
    tier: StudentTier,
    dataset: str,
    output_dir: str,
    teacher_model: str,
    teacher_id: str | None,
    teacher_checkpoint: str | None,
    student_cls_path: str,
    lr: float,
    num_steps: int,
    batch_size: int,
    device: str,
    seed_dataset: str | None,
    validation_path: str | None,
    holdout_path: str | None,
    output: Path | None,
) -> None:
    """Create a distillation recipe scaffold at the requested student tier."""
    if mode == "on-policy" and seed_dataset is None:
        raise click.ClickException(
            "on-policy recipes need --seed-dataset. --dataset names the anchor "
            "the batch mixture draws its reference share from, and an anchor "
            "carries no energy or forces of its own: the propagator reads both "
            "off the seed batch before the student's first forward, and the "
            "strategy rejects an anchor that does carry them. Point "
            "--seed-dataset at a store a dynamics sink or a labeled relaxation "
            "wrote."
        )
    try:
        payload = DistillationJobSpec.template(
            mode=mode,
            tier=tier,
            dataset=dataset,
            output_dir=output_dir,
            teacher_model=teacher_model,
            teacher_id=teacher_id,
            teacher_checkpoint=teacher_checkpoint,
            student_cls_path=student_cls_path,
            lr=lr,
            num_steps=num_steps,
            batch_size=batch_size,
            device=device,
            seed_dataset=seed_dataset,
            validation_path=validation_path,
            holdout_path=holdout_path,
        )
    except ValidationError as exc:
        raise click.ClickException(str(exc)) from exc
    _write_or_print(payload, output)
    if output is not None:
        console.print(f"[green]Created {mode} distillation recipe[/] {output}")


@distill.command("schema")
@click.option(
    "--out",
    "output",
    type=click.Path(path_type=Path),
    help="Write the schema JSON to this file.",
)
def dump_schema(output: Path | None) -> None:
    """Dump the distillation recipe JSON schema."""
    _write_or_print(DistillationJobSpec.model_json_schema(), output)


@distill_spec.command("report")
@click.argument("path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--json", "show_json", is_flag=True, help="Print the normalized recipe.")
def report_recipe(path: Path, show_json: bool) -> None:
    """Validate a recipe and render what it intends to do."""
    job = _load_recipe(path)
    _render_report(job)
    if show_json:
        _write_or_print(job, None)


@distill_spec.command("run")
@click.argument("path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "--distributed/--no-distributed",
    default=None,
    help="Attach DistributedManager and DDPHook. Defaults to auto when WORLD_SIZE > 1.",
)
@click.option(
    "--ddp-backend",
    type=click.Choice(["nccl", "gloo"]),
    default=None,
    help="Process-group backend forwarded to DDPHook.",
)
@click.option("--map-location", default=None, help="Checkpoint map_location.")
@click.option(
    "--report/--no-report",
    "show_report",
    default=True,
    show_default=True,
    help="Render the report before execution.",
)
def run_recipe(
    path: Path,
    distributed: bool | None,
    ddp_backend: str | None,
    map_location: str | None,
    show_report: bool,
) -> None:
    """Build the models, data, and strategy of a recipe, then run it."""
    job = _load_recipe(path)
    if show_report:
        _render_report(job)
    _run_recipe(
        job,
        distributed=distributed,
        ddp_backend=ddp_backend,
        map_location=map_location,
    )


@distill_spec.command("resume")
@click.argument(
    "checkpoint_dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
)
@click.option(
    "--spec",
    "spec_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Recipe that started the run; supplies the data and hook intent.",
)
@click.option("--checkpoint-index", type=int, default=-1, show_default=True)
@click.option(
    "--distributed/--no-distributed",
    default=None,
    help="Attach DistributedManager and DDPHook. Defaults to auto when WORLD_SIZE > 1.",
)
@click.option(
    "--ddp-backend",
    type=click.Choice(["nccl", "gloo"]),
    default=None,
    help="Process-group backend forwarded to DDPHook.",
)
@click.option(
    "--map-location",
    default=None,
    help=(
        "Device the restart loads onto and continues on. Defaults to this "
        "rank's device when distributed."
    ),
)
def resume_recipe(
    checkpoint_dir: Path,
    spec_path: Path,
    checkpoint_index: int,
    distributed: bool | None,
    ddp_backend: str | None,
    map_location: str | None,
) -> None:
    """Continue an interrupted run from its checkpoint and its recipe.

    The checkpoint carries the models, the optimizer and scheduler state, the
    counters, and — for an on-policy run — the trajectory, the propagator's
    step count, and the replay frames. The recipe supplies what a checkpoint
    deliberately does not: the runtime hooks and, offline, the dataloader.

    Under a multi-rank launch the restart is pinned to this rank's device
    rather than to the device the checkpoint records, which is rank zero's.
    --map-location names the device the continued run takes, not only the one
    the checkpoint is read onto, because the two cannot disagree.
    """
    job = _load_recipe(spec_path)
    _resume_recipe(
        job,
        checkpoint_dir,
        checkpoint_index=checkpoint_index,
        distributed=distributed,
        ddp_backend=ddp_backend,
        map_location=map_location,
    )


@distill.command("evaluate")
@click.argument("path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "--student-checkpoint",
    required=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Native checkpoint directory holding the trained student.",
)
@click.option("--checkpoint-index", type=int, default=-1, show_default=True)
@click.option(
    "--holdout", "holdout_path", default=None, help="Override the holdout store."
)
@click.option("--batch-size", type=int, default=None, help="Holdout loader batch size.")
@click.option(
    "--map-location",
    default=None,
    help=(
        "Device the whole evaluation runs on. Defaults to the recipe's "
        "strategy.devices[0]."
    ),
)
@click.option(
    "--json-out",
    "json_out",
    type=click.Path(path_type=Path),
    default=None,
    help=(
        "Write the acceptance report as JSON to this file. A non-finite "
        'metric is written as the string "nan", "inf", or "-inf", so the '
        "file stays readable by a strict JSON parser."
    ),
)
def evaluate_student(
    path: Path,
    student_checkpoint: Path,
    checkpoint_index: int,
    holdout_path: str | None,
    batch_size: int | None,
    map_location: str | None,
    json_out: Path | None,
) -> None:
    """Score a trained student against the recipe's holdout and acceptance bars.

    A recipe whose student.hooks carry an EMAHook is gated on the averaged
    weights that hook trained, the way the run's own validation reads them
    rather than the live ones, and the line above the report names which
    weights were scored. --map-location names the one device the student, the
    teacher, the holdout, and the errors are all placed on.

    Exits non-zero when a bar is not cleared, so a sweep can gate on the
    command rather than on reading its output.
    """
    job = _load_recipe(path)
    evaluation = job.evaluation
    if evaluation is None and holdout_path is None:
        raise click.ClickException(
            "the recipe records no evaluation section, so there is no holdout "
            "to score against; add one, or pass --holdout."
        )
    resolved_holdout = holdout_path or evaluation.holdout_path
    holdout_field = (
        "--holdout" if holdout_path is not None else "evaluation.holdout_path"
    )
    device = (
        torch.device(map_location) if map_location else _primary_strategy_device(job)
    )
    student, weights = _load_evaluated_student(
        job, student_checkpoint, checkpoint_index=checkpoint_index, device=device
    )
    targets = "teacher" if evaluation is None else evaluation.targets
    quantities = None if evaluation is None else list(evaluation.quantities)
    scorer = None
    if targets == "teacher":
        scorer = _build_role_model(
            job.teacher, device=device, role="teacher", map_location=map_location
        )
    with ExitStack() as stack:
        try:
            holdout = _build_dataloader(
                job,
                stack,
                device=device,
                batch_size=batch_size
                or (None if evaluation is None else evaluation.batch_size),
                shuffle=False,
                drop_last=False,
                prefetch_factor=2,
                num_streams=4,
                use_streams=True,
                pin_memory=False,
                paths=[resolved_holdout],
            )
        except (FileNotFoundError, ValueError) as exc:
            raise click.ClickException(
                f"{holdout_field} names {resolved_holdout!r}, which could not "
                f"be opened as a holdout store: {exc}"
            ) from exc
        try:
            metrics = evaluate_accuracy(
                student,
                holdout,
                targets=targets,
                quantities=quantities,
                scorer=scorer,
                device=device,
                name=job.name,
            )
        except (AttributeError, ValueError) as exc:
            raise click.ClickException(
                f"the holdout {resolved_holdout!r} carries no target the "
                "evaluation asked for, or the teacher cannot produce one of "
                f"its quantities: {exc} The errors are measured against "
                f"targets={targets!r} over quantities {quantities!r}; label the "
                "store, narrow evaluation.quantities to what the store holds "
                "and the teacher predicts, or score against the teacher with "
                "evaluation.targets='teacher'."
            ) from exc
    try:
        report = build_acceptance_report(
            [
                StudentEvaluation(
                    name=job.student.tier or job.name,
                    accuracy=metrics,
                    num_parameters=sum(
                        parameter.numel() for parameter in student.parameters()
                    ),
                )
            ],
            None if evaluation is None else evaluation.thresholds,
        )
    except ValueError as exc:
        raise click.ClickException(
            f"the acceptance report could not be formed from the recipe's bars: {exc}"
        ) from exc
    console.print(f"weights: {weights}")
    console.print(report)
    if json_out is not None:
        _write_or_print(_json_safe(report.to_dict()), json_out)
    if not report.accepted:
        raise click.exceptions.Exit(1)
