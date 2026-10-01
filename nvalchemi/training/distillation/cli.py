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
distill``, beside the ``train`` and ``finetune`` groups it mirrors. Every
command reads or writes a recipe: one
JSON file, validated by :class:`DistillationJobSpec`, that describes a whole
distillation run.

The student tiers the scaffold offers are size templates and nothing more.
``small``, ``base``, and ``large`` name a width, a depth, and a radial basis
size (``hidden_dim``, ``num_layers``, and ``num_radial``) for whatever
architecture the recipe points ``student.spec`` at, because a distillation
recipe is about the size of the student rather than its family.
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Mapping
from contextlib import ExitStack
from pathlib import Path
from typing import Annotated, Any, Literal, Self, TypeAlias, get_args

import click
import torch
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator
from rich import box
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from torch import nn

from nvalchemi._serialization import _import_callable, json_safe
from nvalchemi.training import (
    CheckpointManifest,
    ValidationConfig,
    create_model_spec,
    load_checkpoint,
)
from nvalchemi.training import _spec_utils as strategy_spec
from nvalchemi.training.cli_common import (
    DatasetFormat,
    DatasetSpec,
    ModelSource,
    OutputSpec,
    ResumeBudget,
    RuntimeHookSpec,
    SourceSpec,
    ValidationSpec,
    apply_resume_budget,
    build_checked_hook,
    build_dataloader,
    build_runtime_hooks,
    build_supported_source_model,
    build_validation_config,
    common_loader_options,
    common_prefetch_options,
    common_validation_options,
    console,
    dataset_device,
    hook_spec_is,
    path_exists,
    primary_strategy_device,
    resolve_distributed_enabled,
    restart_map_location,
    setup_distributed_manager,
    validate_pretrained_source,
    write_or_print,
)
from nvalchemi.training.distillation.config import (
    _SCORER_CLS_KEY,
    _SOURCE_CLS_KEY,
    OnPolicyConfig,
    _on_policy_settings,
    _signal_from_spec,
)
from nvalchemi.training.distillation.evaluation import (
    AcceptanceThresholds,
    StudentEvaluation,
    build_acceptance_report,
    evaluate_accuracy,
    measured_bars,
)
from nvalchemi.training.distillation.evaluation.accuracy import AccuracyQuantity
from nvalchemi.training.distillation.replay import _batch_allocation
from nvalchemi.training.distillation.scoring import (
    _NEIGHBOR_LIST_POLICIES,
    SUPPORTED_SIGNALS,
)
from nvalchemi.training.distillation.seeding import _InitialStructuresSpec
from nvalchemi.training.distillation.strategy import DistillationStrategy
from nvalchemi.training.hooks.checkpoint import CheckpointHook
from nvalchemi.training.hooks.ema import EMAHook
from nvalchemi.training.losses.composition import ComposedLossFunction
from nvalchemi.training.losses.terms import EnergyMSELoss, ForceMSELoss
from nvalchemi.training.optimizers import OptimizerConfig

__all__ = [
    "DEFAULT_STUDENT_TIERS",
    "DistillationJobSpec",
    "EvaluationSpec",
    "StudentSpec",
    "StudentTier",
    "register_student_tier",
]

DistillationMode: TypeAlias = Literal["offline", "on-policy"]
EvaluatedWeights: TypeAlias = Literal["auto", "ema", "raw"]


@dataclasses.dataclass(frozen=True)
class StudentTier:
    """Size template ``distill init --tier`` writes into the student's constructor arguments.

    A *student tier* is a named size template: a width, a depth, and a
    radial-basis count. It never selects an architecture or a model family.

    Parameters
    ----------
    name : str
        Name the tier is registered and selected under.
    kwargs : dict[str, Any]
        Constructor arguments recorded as ``student.spec.kwargs``.
    """

    name: str
    kwargs: dict[str, Any]


DEFAULT_STUDENT_TIERS: dict[str, StudentTier] = {
    tier.name: tier
    for tier in (
        StudentTier("small", {"hidden_dim": 64, "num_layers": 2, "num_radial": 8}),
        StudentTier("base", {"hidden_dim": 128, "num_layers": 3, "num_radial": 8}),
        StudentTier("large", {"hidden_dim": 256, "num_layers": 4, "num_radial": 12}),
    )
}
"""Registry of the student tiers ``distill init --tier`` selects from, by name."""


def register_student_tier(name: str, **kwargs: Any) -> StudentTier:
    """Register a student size template under *name* for ``distill init --tier``.

    Parameters
    ----------
    name : str
        Tier name, which must not be registered yet.
    **kwargs : Any
        Constructor arguments the tier writes as ``student.spec.kwargs``.

    Returns
    -------
    StudentTier
        The registered tier.

    Raises
    ------
    ValueError
        If *name* is already registered.

    Examples
    --------
    >>> from nvalchemi.training.distillation.cli import register_student_tier
    >>> register_student_tier("xl", hidden_dim=512, num_layers=6).name  # doctest: +SKIP
    'xl'
    """
    if name in DEFAULT_STUDENT_TIERS:
        raise ValueError(
            f"A student tier named {name!r} is already registered; registered "
            f"tiers are {sorted(DEFAULT_STUDENT_TIERS)!r}. Pick another name."
        )
    tier = StudentTier(name, dict(kwargs))
    DEFAULT_STUDENT_TIERS[name] = tier
    return tier


def _tier_override(entry: str) -> tuple[str, Any]:
    """Return the ``(key, value)`` a ``--tier-kwargs KEY=VALUE`` entry sets.

    The value is read as JSON when it parses as one, so ``hidden_dim=96`` is
    an integer and ``activation=silu`` stays a string.
    """
    key, separator, value = entry.partition("=")
    if not separator or not key:
        raise click.BadParameter(
            f"expected KEY=VALUE; got {entry!r}.", param_hint="--tier-kwargs"
        )
    try:
        return key, json.loads(value)
    except json.JSONDecodeError:
        return key, value


def _is_autocast_spec(value: Any) -> bool:
    """Return whether *value* spells an ``InProcessTeacherScorer.autocast`` setting.

    The recipe form is the constructor's, with a floating-point dtype written
    by its ``torch`` name.
    """
    if value is None or isinstance(value, bool):
        return True
    dtype = getattr(torch, value, None) if isinstance(value, str) else None
    return isinstance(dtype, torch.dtype) and dtype.is_floating_point


@dataclasses.dataclass(frozen=True)
class _LoaderOptions:
    """Dataloader and validation settings a spec command forwards to the core builders.

    The defaults are the recipe's own batch size and validation cadence, a
    shuffled training loader that keeps its last batch, and the core CLI's
    prefetch settings.
    """

    batch_size: int | None = None
    shuffle: bool = True
    drop_last: bool = False
    prefetch_factor: int = 2
    num_streams: int = 4
    pin_memory: bool = False
    use_streams: bool = True
    validation_path: str | None = None
    validation_every_epochs: int | None = None
    validation_every_steps: int | None = None


_SCAFFOLD_CHECKPOINTS = 10
"""Restart checkpoints a scaffolded run spreads over its step budget."""

_SCAFFOLD_BATCH_SIZE = 8
"""Samples per training batch a scaffolded recipe records."""

_DATASET_FORMATS = frozenset(get_args(DatasetFormat))
"""Loader families a recipe's dataset.format may name: the ones the core CLI builds."""

_RECIPE_SOURCES = frozenset(get_args(ModelSource)) - {"custom"}
"""Model families a recipe loads a teacher or a student from: every core source but custom."""

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
    :class:`DistillationJobSpec`. ``distill evaluate`` reads it; the run itself
    does not. A recipe therefore carries the bars it was meant to clear, and
    gating a trained student is one command against the same file.

    The bars it may carry are
    ``measured_bars("accuracy", accuracy_quantities=quantities)``. Scoring a
    student over a holdout is all ``distill evaluate`` does, and a bar with no
    measurement behind it fails the student rather than passing it. A
    stability, throughput, extensivity, RDF, or from-scratch bar is therefore
    refused at parse time, and so is a stress bar without ``"stress"`` among
    the *quantities*, rather than run as a gate nothing could clear.

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
        """Refuse the bars ``distill evaluate`` has no measurement to fill."""
        measurable = measured_bars("accuracy", accuracy_quantities=self.quantities)
        configured = self.thresholds.model_dump(exclude_defaults=True)
        configured.update(configured.pop("extra", {}))
        unmeasurable = sorted(set(configured) - measurable)
        if unmeasurable:
            raise ValueError(
                f"evaluation.thresholds sets bars {unmeasurable!r} that `distill "
                "evaluate` does not measure; got quantities "
                f"{list(self.quantities)!r}, which fill the accuracy bars "
                f"{sorted(measurable)!r} only. A bar nothing measured would fail "
                "the student on a number nobody took. Add the quantity a bar "
                "reads to evaluation.quantities. The stability, throughput, and "
                "extensivity bars need a propagator and a timestep, a supercell "
                "builder, or a second trained model, which no recipe carries: "
                "measure them with StabilityMonitor, measure_throughput, and "
                "extensivity_error, and assemble one report from their "
                "to_dict() exports with build_acceptance_report."
            )
        return self


class StudentSpec(BaseModel):
    """Where the student comes from, and at what size.

    ``StudentSpec`` is the ``student`` member of :class:`DistillationJobSpec`.
    A student is normally constructed rather than loaded, so the common form is
    ``spec``. It is a ``{"cls_path": ..., "kwargs": {...}}`` reference that
    names the constructor and the arguments it is called with, the same shape
    the segment loop names its propagator by. ``tier`` records which student
    tier those arguments came from, only so that a report and a sweep can say
    which tier a run belongs to. A tier selects a size, never an architecture.
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
        str | None,
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
                "Exactly one of student.spec or student.source must be set; got "
                f"{'both' if self.spec is not None else 'neither'}. spec "
                "constructs a fresh student, and source loads one from a "
                "checkpoint."
            )
        if self.spec is not None and "cls_path" not in self.spec:
            raise ValueError(
                "student.spec names the constructor by cls_path, with its "
                "arguments under kwargs."
            )
        return self


class DistillationJobSpec(BaseModel):
    """Top-level envelope describing one distillation recipe.

    A *recipe* is one JSON file, validated by this class, that describes a
    whole distillation run: where the teacher and the student come from, the
    data, the output paths, the strategy bundle, the on-policy segment loop
    when there is one, and the acceptance bars ``distill evaluate`` gates on.
    It is the file ``distill spec report`` reads and ``distill spec run``
    executes.

    ``mode`` selects offline distillation over a teacher-labeled store, or the
    on-policy segment loop. ``teacher`` and ``student`` say where the two
    models come from. ``dataset`` names the training store: the labeled
    dataset offline, the reference dataset on-policy. ``strategy`` is the
    *strategy bundle*, the JSON-ready
    :meth:`~nvalchemi.training.distillation.DistillationStrategy.to_spec_dict`
    output carrying optimizers, loss, devices, and duration. ``on_policy`` is
    the block ``OnPolicyConfig.to_spec_dict()`` writes, which an on-policy run
    needs, and ``evaluation`` records the bars ``distill evaluate`` gates on.

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
    Validation is pre-flight. The strategy bundle is deserialized with the
    runtime's own helpers. An ``on_policy`` block is checked against
    :class:`~nvalchemi.training.distillation.OnPolicyConfig`'s field
    constraints, and its ``initial_structures`` block against the description
    :meth:`~nvalchemi.training.distillation.InitialStructures.from_spec_dict`
    rebuilds through. A setting out of range, a misspelled or non-positive
    budget, and a block naming no store therefore fail at ``spec report``.
    Checks that need the models built run in the strategy at ``spec run``:
    that the loss's teacher targets are signals the teacher can produce, or
    that a propagator's ``cls_path`` imports. Their failures are reported as a
    CLI error rather than a traceback.

    ``mode`` alone decides which loop runs. An offline recipe whose strategy
    bundle carries ``on_policy`` or ``reference_dataset`` is rejected, and in
    on-policy mode the top-level ``on_policy`` block is the one built.
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
        Field(
            description=(
                "Training store: the labeled dataset offline, the reference "
                "dataset on-policy."
            )
        ),
    ]
    output: Annotated[
        OutputSpec,
        Field(
            description=(
                "Where the run writes: its run directory and, when set, the "
                "checkpoint root a CheckpointHook in student.hooks fills."
            )
        ),
    ]
    validation: Annotated[
        ValidationSpec | None,
        Field(description="Optional validation cadence for CLI execution."),
    ] = None
    on_policy: Annotated[
        dict[str, Any] | None,
        Field(
            description=(
                "Segment-loop on_policy block, as OnPolicyConfig.to_spec_dict() "
                "writes it. Required by, and only read in, on-policy mode."
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
        """Require the ``on_policy`` block exactly when the mode asks for one."""
        if self.mode == "on-policy" and self.on_policy is None:
            raise ValueError(
                "On-policy recipes need an on_policy block; got mode='on-policy' "
                "with on_policy=None. The segment loop generates its own batches "
                "and has no dataloader to fall back on. Add the block, or set "
                "mode='offline'."
            )
        if self.mode == "offline" and self.on_policy is not None:
            raise ValueError(
                "Offline recipes train on the dataset they name, so an on_policy "
                "block would never be read; got mode='offline' with an on_policy "
                "block. Set mode='on-policy' to use it, or drop the block."
            )
        if self.mode == "offline":
            bundled = [
                key
                for key in ("on_policy", "reference_dataset")
                if self.strategy.get(key) is not None
            ]
            if bundled:
                raise ValueError(
                    f"strategy carries {bundled!r} while mode='offline'. Those "
                    "entries come from DistillationStrategy.to_spec_dict() of an "
                    "on-policy run, and they rebuild the segment loop, so an "
                    "offline run would either train that loop or fail in the "
                    "strategy's constructor. Drop the entries, or set "
                    "mode='on-policy' and lift strategy.on_policy to the "
                    "top-level on_policy block."
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
        strategy_spec._optimizer_configs_from_spec(self.strategy["optimizer_configs"])
        strategy_spec._devices_from_spec(self.strategy["devices"])
        DistillationStrategy.resolve_teacher_signals(
            strategy_spec._loss_fn_from_spec(self.strategy["loss_fn_spec"]),
            teacher_signals=self.strategy.get("teacher_signals"),
        )
        strategy_spec._training_fn_from_spec(self.strategy, None)
        if self.validation is not None and self.dataset.validation_path is None:
            raise ValueError(
                "A validation cadence requires dataset.validation_path; got a "
                "validation section with dataset.validation_path=None. Name the "
                "validation store, or drop the validation section."
            )
        if self.dataset.format not in _DATASET_FORMATS:
            raise ValueError(
                f"dataset.format {self.dataset.format!r} is not a format the "
                f"loader builds; supported formats: {sorted(_DATASET_FORMATS)}."
            )
        return self

    @model_validator(mode="after")
    def _validate_sources(self) -> Self:
        """Require each model source to carry what the wrapper that builds it needs.

        The rules match the ones
        :meth:`~nvalchemi.training.cli.TrainingJobSpec._validate_workflow_source`
        applies to a fine-tune source, because a recipe obtains both of its
        models the same way. Every source is pretrained; the from-scratch case
        is ``student.spec``, never a source.
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
            validate_pretrained_source(field, source)
        return self

    @staticmethod
    def _validate_scorer_block(scorer_block: Mapping[str, Any]) -> None:
        """Check an ``InProcessTeacherScorer`` block; a ``scorer_cls`` block is its class's."""
        signals = scorer_block.get("signals")
        try:
            resolved = [_signal_from_spec(entry) for entry in signals or ()]
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "on_policy.teacher_scorer.signals describes a custom signal as a "
                "TeacherSignal dict with name, model_output, field, and level; "
                f"got {signals!r}: {exc}"
            ) from exc
        unsupported = sorted(
            entry
            for entry in resolved
            if isinstance(entry, str) and entry not in SUPPORTED_SIGNALS
        )
        if not signals or unsupported:
            raise ValueError(
                "on_policy.teacher_scorer.signals must name teacher signals "
                f"from {sorted(SUPPORTED_SIGNALS)!r} or describe custom ones as "
                f"TeacherSignal dicts; got {signals!r}."
            )
        policy = scorer_block.get("neighbor_list", "rebuild")
        if policy not in _NEIGHBOR_LIST_POLICIES:
            raise ValueError(
                "on_policy.teacher_scorer.neighbor_list must be one of "
                f"{sorted(_NEIGHBOR_LIST_POLICIES)!r}; got {policy!r}."
            )
        autocast = scorer_block.get("autocast", False)
        if not _is_autocast_spec(autocast):
            raise ValueError(
                "on_policy.teacher_scorer.autocast must be false to disable "
                "autocast for the scoring pass, null to leave the caller's "
                "autocast state in force, or true or a floating-point dtype name "
                f"such as 'bfloat16' to enable it; got {autocast!r}."
            )

    @model_validator(mode="after")
    def _validate_on_policy_recipe(self) -> Self:
        """Check the ``on_policy`` block the way the config it builds would."""
        if self.on_policy is None:
            return self
        required = (
            "dynamics",
            "teacher_scorer",
            "replay_ratio",
            "training_steps_per_segment",
        )
        missing = [key for key in required if key not in self.on_policy]
        if missing:
            raise ValueError(f"on_policy is missing required key(s) {missing}.")
        if "cls_path" not in self.on_policy["dynamics"]:
            raise ValueError(
                "on_policy.dynamics names the propagator by cls_path, with its "
                "constructor arguments under kwargs; the student is bound at "
                "build time and must not be named."
            )
        scorer_block = self.on_policy["teacher_scorer"]
        if _SCORER_CLS_KEY not in scorer_block:
            self._validate_scorer_block(scorer_block)
        block = self.on_policy.get("initial_structures") or {}
        if _SOURCE_CLS_KEY not in block:
            try:
                _InitialStructuresSpec.model_validate(block)
            except ValidationError as exc:
                raise ValueError(
                    f"on_policy.initial_structures is invalid; got {exc}\n"
                    "The block names the store the first segment starts from, as "
                    "a dataset entry giving its path, and the budgets a batch "
                    "packed from it is held to. A custom source travels under "
                    "source_cls instead and is checked when it is rebuilt. An "
                    "InitialStructures over an in-memory dataset has no recipe "
                    "form: write its samples to a store and name that."
                ) from exc
        try:
            _on_policy_settings(self.on_policy)
        except ValidationError as exc:
            raise ValueError(f"on_policy settings are invalid: {exc}") from exc
        return self

    @classmethod
    def template(
        cls,
        *,
        mode: DistillationMode,
        tier: str,
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
        initial_structures: str | None = None,
        validation_path: str | None = None,
        holdout_path: str | None = None,
        tier_kwargs: Mapping[str, Any] | None = None,
    ) -> Self:
        """Build a validated scaffold for a distillation recipe.

        Parameters
        ----------
        mode : {"offline", "on-policy"}
            Which loop the recipe describes.
        tier : str
            Name of a tier in :data:`DEFAULT_STUDENT_TIERS`, whose template is
            written into the student's constructor arguments.
        dataset : str
            Training store: the teacher-labeled dataset offline, the reference
            dataset on-policy.
        output_dir : str
            Run directory. The scaffold sets ``output.checkpoint_dir`` beneath
            it and adds the ``CheckpointHook`` that writes there, because
            nothing else in the recipe would produce the weights
            ``distill evaluate`` scores.
        teacher_model : str
            Teacher source family, as in the training CLI.
        teacher_id : str | None, optional
            Teacher model id for a supported wrapper. Default ``None``.
        teacher_checkpoint : str | None, optional
            Teacher checkpoint path. Default ``None``.
        student_cls_path : str, optional
            Dotted path of the student constructor the tier sizes. Default
            ``"my_package.my_module.MyStudentModel"``, a placeholder to edit.
        lr : float, optional
            Student learning rate. Default ``1e-4``.
        num_steps : int, optional
            Optimizer steps to run. Default ``1000``.
        batch_size : int, optional
            Samples per training batch, recorded as ``dataset.batch_size``
            for the offline training loader and the validation loader.
            Default ``8``.
        device : str, optional
            Strategy device string. Default ``"cuda"``.
        initial_structures : str | None, optional
            Store the on-policy loop's initial structures are read from,
            written into the recipe as
            ``on_policy.initial_structures.dataset.path``. Required in
            on-policy mode. Default ``None``.
        validation_path : str | None, optional
            Validation store. Default ``None``.
        holdout_path : str | None, optional
            Holdout store recorded in the evaluation section. Default ``None``.
        tier_kwargs : Mapping[str, Any] | None, optional
            Constructor arguments overriding or extending the tier's template.
            Default ``None``.

        Returns
        -------
        DistillationJobSpec
            Validated scaffold ready to be edited and reported on.

        Raises
        ------
        ValueError
            If *tier* names no registered tier, or if *mode* is
            ``"on-policy"`` and no *initial_structures* is named. The
            reference dataset cannot stand in, because it carries no forces
            for the propagator's first step.
        """
        if tier not in DEFAULT_STUDENT_TIERS:
            raise ValueError(
                f"No student tier named {tier!r} is registered; registered tiers "
                f"are {sorted(DEFAULT_STUDENT_TIERS)!r}. Pick one of them, or "
                "register the tier with register_student_tier first."
            )
        if mode == "on-policy" and initial_structures is None:
            raise ValueError(
                "on-policy recipes name a store of initial structures under "
                "on_policy.initial_structures: the propagator reads energy and "
                "forces off the initial batch before the student's first "
                "forward, and the reference dataset named by dataset carries "
                "neither."
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
                    "kwargs": {
                        **DEFAULT_STUDENT_TIERS[tier].kwargs,
                        **(tier_kwargs or {}),
                    },
                },
                "hooks": [_checkpoint_hook_template(checkpoint_dir, num_steps)],
            },
            dataset=dataset_payload,
            output={"run_dir": output_dir, "checkpoint_dir": checkpoint_dir},
            validation=(None if validation_path is None else {"every_n_epochs": 1}),
            on_policy=(
                None
                if mode == "offline"
                else _on_policy_template(initial_structures, device)
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

    Nothing else in a recipe writes weights. Without the hook,
    ``output.checkpoint_dir`` is only a declaration: a scaffolded run completes
    and leaves ``distill evaluate`` no checkpoint to score.
    :class:`~nvalchemi.training.CheckpointHook` takes exactly one cadence, and
    the scaffold derives it from the step budget, which it already knows. The
    budget need not be a multiple of the interval, because the hook is
    scaffolded with ``save_at_end`` on: a run ending on a step the interval
    misses still leaves its final weights on disk, for ``evaluate`` to score
    and for ``spec resume`` to continue from without repeating steps.
    """
    interval = max(1, num_steps // _SCAFFOLD_CHECKPOINTS)
    spec = create_model_spec(
        CheckpointHook,
        checkpoint_dir=checkpoint_dir,
        step_interval=interval,
        save_at_end=True,
    )
    return {"spec": spec.model_dump(mode="json")}


def _on_policy_template(initial_structures: str, device: str) -> dict[str, Any]:
    """Return a scaffold ``on_policy`` block over the *initial_structures* store."""
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
            "dtype": None,
            "probe_seed": None,
            "neighbor_list": "rebuild",
            "autocast": False,
        },
        "initial_structures": {
            "dataset": {"path": initial_structures, "device": device},
            "max_atoms": None,
            "max_edges": None,
            "max_batch_size": None,
            "recycle": False,
        },
        "replay_ratio": 0.25,
        "training_steps_per_segment": 32,
        "batch_size": 8,
        "generation_steps": 50,
        "label_frequency": 10,
        "replay_capacity": 8192,
        "replay_eviction": "fifo",
        "replay_device": None,
        "seed": 0,
        "fmax": None,
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
        "loss_fn_spec": loss_fn.to_spec().model_dump(),
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
    store = (
        job.on_policy.get("initial_structures", {}).get("dataset", {}).get("path")
        if job.on_policy is not None
        else None
    )
    if store is not None:
        checks.append(("on_policy.initial_structures.dataset.path", store))
    if job.evaluation is not None:
        checks.append(("evaluation.holdout_path", job.evaluation.holdout_path))
    return [(field, value) for field, value in checks if value is not None]


def _mixture_rows(job: DistillationJobSpec) -> list[tuple[str, str]]:
    """Return the composition of one training batch, as label/value rows."""
    if job.on_policy is None:
        return [("mixture", "every sample from the labeled dataset (offline)")]
    settings = _on_policy_settings(job.on_policy)
    reference, replay = _batch_allocation(settings.replay_ratio, settings.batch_size)
    return [
        ("replay_ratio", f"{settings.replay_ratio:g}"),
        ("batch composition", f"{reference} reference + {replay} generated"),
        ("segment", f"{settings.generation_steps} generated steps"),
        ("label cadence", f"every {settings.label_frequency} steps"),
        ("training per segment", f"{settings.training_steps_per_segment} batches"),
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
    table.add_row(
        "teacher signals",
        ", ".join(
            DistillationStrategy.resolve_teacher_signals(
                strategy_spec._loss_fn_from_spec(job.strategy["loss_fn_spec"]),
                teacher_signals=job.strategy.get("teacher_signals"),
            )
        ),
    )
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
    :class:`~nvalchemi.training.CheckpointHook` pointed at another directory.
    The check therefore matches the destination as well as the class. A run
    whose hook writes elsewhere finishes cleanly, never creates the directory
    the recipe names, and leaves ``distill evaluate`` nothing to read there.
    """
    target = Path(job.output.checkpoint_dir or "")
    return any(
        hook_spec_is(hook, CheckpointHook)
        and Path(str((hook.spec.model_extra or {}).get("checkpoint_dir", ""))) == target
        for hook in job.student.hooks
    )


def _ema_hook_specs(job: DistillationJobSpec) -> list[RuntimeHookSpec]:
    """Return the recipe's runtime hooks that average the student's weights.

    Only an ``EMAHook`` publishes an average worth scoring in place of the
    trained weights. The check therefore matches the class, a subclass
    included, rather than counting the hooks a recipe declares.
    """
    return [hook for hook in job.student.hooks if hook_spec_is(hook, EMAHook)]


def _load_evaluated_student(
    job: DistillationJobSpec,
    checkpoint: Path,
    *,
    checkpoint_index: int,
    device: torch.device,
    weights: EvaluatedWeights = "auto",
) -> tuple[Any, Literal["ema", "raw"], str]:
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
    weights : {"auto", "ema", "raw"}, optional
        Which weights to score. ``"auto"`` reads the averaged weights when the
        recipe declares an ``EMAHook`` and the trained ones otherwise;
        ``"ema"`` insists on the average and fails when the checkpoint holds
        none; ``"raw"`` scores the trained weights whatever the recipe
        declares. Default ``"auto"``.

    Returns
    -------
    tuple[Any, Literal["ema", "raw"], str]
        The module to score, the marker
        :attr:`~nvalchemi.training.distillation.evaluation.StudentEvaluation.weights`
        records it under, and the phrase the report line names it with.

    Raises
    ------
    click.ClickException
        If the checkpoint cannot be restored under the recipe's EMA hooks, or
        if ``weights="ema"`` finds no averaged student to score.

    Notes
    -----
    A recipe carrying an ``EMAHook``, a subclass included, trained an
    average. The run's own validation reads that average, so by default the
    gate reads it too. The strategy is restored under that hook alone and
    :attr:`TrainingStage.SETUP` is dispatched, which rebuilds the averaged
    model into ``inference_model``. The recipe's other hooks are left out:
    they have no part in scoring, and a ``DDPHook`` would open a process
    group. Without an EMA hook, or under ``weights="raw"``, the student is
    loaded alone.
    """
    specs = _ema_hook_specs(job)
    if weights == "ema" and not specs:
        raise click.ClickException(
            "--weights ema asks for the averaged weights, but the recipe's "
            "student.hooks declare no EMAHook, so the checkpoint holds none. "
            "Score the trained weights with --weights raw, or add the EMAHook "
            "the run trained with to the recipe."
        )
    if weights == "raw" or not specs:
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
        detail = (
            "raw" if not specs else "raw (--weights raw; the EMA average is not scored)"
        )
        return student, "raw", detail
    try:
        hooks = [build_checked_hook(spec.spec) for spec in specs]
        strategy = DistillationStrategy.load_checkpoint(
            checkpoint,
            checkpoint_index=checkpoint_index,
            map_location=str(device),
            hooks=hooks,
        )
        strategy.run_setup_hooks()
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
        if weights == "ema":
            raise click.ClickException(
                "--weights ema asks for the averaged weights, but the recipe's "
                "EMAHook published no averaged student from checkpoint "
                f"{str(checkpoint)!r} at index {checkpoint_index!r}: its "
                "model_key names no model the checkpoint holds. Score the "
                "trained weights with --weights raw, or fix the hook's "
                "model_key."
            )
        return (
            strategy.models["student"],
            "raw",
            "raw (the recipe's EMAHook published no averaged student)",
        )
    return published, "ema", "ema (student.hooks EMAHook)"


def _stores_a_teacher(checkpoint_dir: str | None) -> bool:
    """Return whether a checkpoint root already holds a teacher of its own.

    A teacher is stored once per checkpoint root. A second run that writes a
    different teacher into an occupied root is refused at its first
    checkpoint, which a scaffolded recipe reaches after
    ``max(1, num_steps // 10)`` optimizer steps. Reading the manifest needs
    neither model, so the report can say it up front. A root whose manifest
    the reader rejects is left unremarked rather than reported as occupied.
    """
    if not checkpoint_dir:
        return False
    try:
        manifest = CheckpointManifest.read(Path(checkpoint_dir))
    except (OSError, ValueError):
        return False
    return "teacher" in manifest.model_references


def _warning_table(job: DistillationJobSpec) -> Table:
    """Build the table of pre-flight warnings for a recipe."""
    table = Table(title="Pre-flight", box=box.SIMPLE_HEAD, expand=True)
    table.add_column("Check", style="cyan", no_wrap=True)
    table.add_column("Detail", overflow="fold")
    missing = [
        (field, value) for field, value in _recipe_paths(job) if not path_exists(value)
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
        return build_supported_source_model(source, device=device)
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
    """Open the store or stores an on-policy run mixes reference batches from."""
    from nvalchemi.data.datapipes import AtomicDataZarrReader, Dataset, MultiDataset

    paths = _dataset_store_paths(job)
    if not paths:
        raise click.ClickException(
            "dataset names no store for the on-policy reference dataset; set "
            "dataset.path or dataset.paths."
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
    validation_config: ValidationConfig | None,
) -> DistillationStrategy:
    """Build the strategy a recipe declares, raising its build errors as CLI errors."""
    device = dataset_device(job, distributed_manager)
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
            validation_config=validation_config,
            on_policy=on_policy,
            reference_dataset=reference_dataset,
        )
    except (ValueError, TypeError, KeyError) as exc:
        raise click.ClickException(f"The strategy could not be built: {exc}") from exc
    strategy.distributed_manager = distributed_manager
    return strategy


def _recipe_validation_config(
    job: DistillationJobSpec,
    stack: ExitStack,
    *,
    device: Any,
    options: _LoaderOptions = _LoaderOptions(),
) -> ValidationConfig | None:
    """Build the validation configuration a recipe declares, before the strategy exists.

    A validation loss with a ``teacher_*`` target of its own widens the signals
    the teacher is scored for and the fields by which a batch counts as
    labeled. Neither check re-runs on assignment, so the config has to reach
    the constructor rather than the built strategy.
    """
    return build_validation_config(
        job,
        stack,
        device=device,
        batch_size=options.batch_size or job.dataset.batch_size,
        prefetch_factor=options.prefetch_factor,
        num_streams=options.num_streams,
        use_streams=options.use_streams,
        pin_memory=options.pin_memory,
        validation_path=options.validation_path,
        validation_every_epochs=options.validation_every_epochs,
        validation_every_steps=options.validation_every_steps,
    )


def _execute_strategy(
    job: DistillationJobSpec,
    strategy: DistillationStrategy,
    stack: ExitStack,
    *,
    device: Any,
    options: _LoaderOptions = _LoaderOptions(),
) -> None:
    """Drive the loop the recipe's mode names."""
    if job.mode == "on-policy":
        _run_strategy(strategy)
        return
    dataloader = build_dataloader(
        job,
        stack,
        device=device,
        batch_size=options.batch_size or job.dataset.batch_size,
        shuffle=options.shuffle,
        drop_last=options.drop_last,
        prefetch_factor=options.prefetch_factor,
        num_streams=options.num_streams,
        use_streams=options.use_streams,
        pin_memory=options.pin_memory,
    )
    _run_strategy(strategy, dataloader)


def _run_strategy(strategy: DistillationStrategy, *args: Any) -> None:
    """Drive the loop, reporting the strategy's own contract errors as CLI errors.

    The strategy picks its loop from what it was built with. A recipe whose
    ``mode`` disagrees with a restored checkpoint is therefore refused here,
    by the strategy, rather than by a second copy of the rule.
    """
    try:
        strategy.run(*args)
    except ValueError as exc:
        raise click.ClickException(f"The run failed: {exc}") from exc


def _run_recipe(
    job: DistillationJobSpec,
    *,
    distributed: bool | None,
    ddp_backend: str | None,
    map_location: str | None,
    options: _LoaderOptions = _LoaderOptions(),
) -> None:
    """Build the runtime components of a recipe and run it."""
    distributed_enabled = resolve_distributed_enabled(distributed)
    distributed_manager = setup_distributed_manager(distributed_enabled)
    hooks = build_runtime_hooks(
        job.student.hooks, enable_ddp=distributed_enabled, ddp_backend=ddp_backend
    )
    with ExitStack() as stack:
        device = dataset_device(job, distributed_manager)
        strategy = _build_strategy(
            job,
            stack,
            hooks=hooks,
            distributed_manager=distributed_manager,
            map_location=map_location,
            validation_config=_recipe_validation_config(
                job, stack, device=device, options=options
            ),
        )
        _execute_strategy(job, strategy, stack, device=device, options=options)


def _resume_recipe(
    job: DistillationJobSpec,
    checkpoint_dir: Path,
    *,
    checkpoint_index: int,
    distributed: bool | None,
    ddp_backend: str | None,
    map_location: str | None,
    options: _LoaderOptions = _LoaderOptions(),
    budget: ResumeBudget = "checkpoint",
) -> None:
    """Restore a checkpointed run and continue it under the recipe."""
    distributed_enabled = resolve_distributed_enabled(distributed)
    distributed_manager = setup_distributed_manager(distributed_enabled)
    hooks = build_runtime_hooks(
        job.student.hooks, enable_ddp=distributed_enabled, ddp_backend=ddp_backend
    )
    load_location = restart_map_location(distributed_manager, map_location)
    device = (
        dataset_device(job, distributed_manager)
        if load_location is None
        else torch.device(load_location)
    )
    with ExitStack() as stack:
        try:
            strategy = DistillationStrategy.load_checkpoint(
                checkpoint_dir,
                checkpoint_index=checkpoint_index,
                map_location=load_location,
                hooks=hooks,
                validation_config=_recipe_validation_config(
                    job, stack, device=device, options=options
                ),
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
        apply_resume_budget(job.strategy, strategy, budget=budget)
        _execute_strategy(job, strategy, stack, device=device, options=options)


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
    default="small",
    show_default=True,
    help=(
        "Student size template: a width, a depth, and a radial basis size, never "
        "an architecture. "
        "One of the registered tiers (built in: small, base, large)."
    ),
)
@click.option(
    "--tier-kwargs",
    "tier_overrides",
    multiple=True,
    metavar="KEY=VALUE",
    help=(
        "Constructor argument overriding or extending the tier's template; "
        "repeatable. A value that parses as JSON is written as that type."
    ),
)
@click.option(
    "--dataset",
    required=True,
    help="Training store: teacher-labeled offline, the reference dataset on-policy.",
)
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
@click.option(
    "--lr", type=float, default=1e-4, show_default=True, help="Student learning rate."
)
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
@click.option(
    "--device",
    default="cuda",
    show_default=True,
    help="Device written to strategy.devices.",
)
@click.option(
    "--initial-structures",
    default=None,
    help="Store of initial structures the segment loop starts from; required with --mode on-policy.",
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
    tier: str,
    tier_overrides: tuple[str, ...],
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
    initial_structures: str | None,
    validation_path: str | None,
    holdout_path: str | None,
    output: Path | None,
) -> None:
    """Create a distillation recipe scaffold at the requested student tier.

    --tier is checked against the tier registry when the command runs, so a
    tier registered by an imported plugin is selectable by name.
    """
    if tier not in DEFAULT_STUDENT_TIERS:
        raise click.BadParameter(
            f"{tier!r} is not a registered student tier; registered tiers are "
            f"{sorted(DEFAULT_STUDENT_TIERS)!r}.",
            param_hint="--tier",
        )
    if mode == "on-policy" and initial_structures is None:
        raise click.ClickException(
            "on-policy recipes need --initial-structures. --dataset names the "
            "reference dataset the batch mixture draws its reference share "
            "from, and it carries no energy or forces of its own: the "
            "propagator reads both off the initial batch before the student's "
            "first forward, and the strategy rejects a reference dataset that "
            "does carry them. Point --initial-structures at a store a dynamics sink "
            "or a labeled relaxation wrote."
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
            initial_structures=initial_structures,
            validation_path=validation_path,
            holdout_path=holdout_path,
            tier_kwargs=dict(_tier_override(entry) for entry in tier_overrides),
        )
    except ValidationError as exc:
        raise click.ClickException(str(exc)) from exc
    write_or_print(payload, output)
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
    write_or_print(DistillationJobSpec.model_json_schema(), output)


@distill_spec.command("report")
@click.argument("path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "--json",
    "show_json",
    is_flag=True,
    help="Also print the normalized recipe, with omitted fields set to their defaults.",
)
def report_recipe(path: Path, show_json: bool) -> None:
    """Validate a recipe and render what it intends to do."""
    job = _load_recipe(path)
    _render_report(job)
    if show_json:
        write_or_print(job, None)


@distill_spec.command("run")
@click.argument("path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@common_loader_options
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
    help="Device a teacher or student checkpoint loads onto.",
)
@common_validation_options
@click.option(
    "--report/--no-report",
    "show_report",
    default=True,
    show_default=True,
    help="Render the report before execution.",
)
def run_recipe(
    path: Path,
    batch_size: int | None,
    shuffle: bool,
    drop_last: bool,
    prefetch_factor: int,
    num_streams: int,
    pin_memory: bool,
    use_streams: bool,
    distributed: bool | None,
    ddp_backend: str | None,
    map_location: str | None,
    validation_path: str | None,
    validation_every_epochs: int | None,
    validation_every_steps: int | None,
    show_report: bool,
) -> None:
    """Build the models, data, and strategy of a recipe, then run it.

    The loader and validation options are the training CLI's own. They
    override the recipe's dataset.batch_size and validation cadence for this
    run, and shape the offline training loader. An on-policy run builds its
    own loaders from the segment loop and takes only the validation options.
    """
    job = _load_recipe(path)
    if show_report:
        _render_report(job)
    _run_recipe(
        job,
        distributed=distributed,
        ddp_backend=ddp_backend,
        map_location=map_location,
        options=_LoaderOptions(
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            prefetch_factor=prefetch_factor,
            num_streams=num_streams,
            pin_memory=pin_memory,
            use_streams=use_streams,
            validation_path=validation_path,
            validation_every_epochs=validation_every_epochs,
            validation_every_steps=validation_every_steps,
        ),
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
    help=(
        "Recipe the run started from. It supplies the data and the hooks a "
        "checkpoint does not carry and, with --budget recipe, the num_steps or "
        "num_epochs that size the continued run."
    ),
)
@click.option("--checkpoint-index", type=int, default=-1, show_default=True)
@click.option(
    "--budget",
    type=click.Choice(["checkpoint", "recipe"]),
    default="checkpoint",
    show_default=True,
    help=(
        "Whose num_steps/num_epochs size the continued run: the checkpoint's "
        "stored spec, or the recipe's, which lets an edited recipe extend or "
        "shorten the run. A recipe budget below what the checkpoint completed, "
        "or in the other unit, is refused either way."
    ),
)
@common_loader_options
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
        "Device the checkpoint is loaded onto and the restart continues on. "
        "Defaults to this rank's device when distributed, so no rank stages "
        "its weights through rank zero's."
    ),
)
@common_validation_options
def resume_recipe(
    checkpoint_dir: Path,
    spec_path: Path,
    checkpoint_index: int,
    budget: ResumeBudget,
    batch_size: int | None,
    shuffle: bool,
    drop_last: bool,
    prefetch_factor: int,
    num_streams: int,
    pin_memory: bool,
    use_streams: bool,
    distributed: bool | None,
    ddp_backend: str | None,
    map_location: str | None,
    validation_path: str | None,
    validation_every_epochs: int | None,
    validation_every_steps: int | None,
) -> None:
    """Continue an interrupted run from its checkpoint and its recipe.

    The checkpoint carries the models, the optimizer and scheduler state, the
    counters, and, for an on-policy run, the trajectory, the propagator's
    step count, the initial structures' position, and the replay frames. The
    recipe supplies what a checkpoint
    does not carry: the runtime hooks and, offline, the dataloader. An
    on-policy recipe's on_policy.restart says what the run does with a restart
    bundle it cannot consume.

    --budget says whose num_steps or num_epochs size the continued run: the
    checkpoint's by default, or the recipe's with --budget recipe, so that
    editing the recipe extends or shortens the run. A disagreement is reported
    with both values. A recipe budget the checkpoint has already passed, or
    one in the other unit, is refused.

    Under a multi-rank launch the checkpoint is loaded onto this rank's device
    rather than the one it records, which is rank zero's, so no rank stages its
    weights through another's memory. --map-location overrides that and names
    the device the continued run takes.
    """
    job = _load_recipe(spec_path)
    _resume_recipe(
        job,
        checkpoint_dir,
        checkpoint_index=checkpoint_index,
        distributed=distributed,
        ddp_backend=ddp_backend,
        map_location=map_location,
        options=_LoaderOptions(
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            prefetch_factor=prefetch_factor,
            num_streams=num_streams,
            pin_memory=pin_memory,
            use_streams=use_streams,
            validation_path=validation_path,
            validation_every_epochs=validation_every_epochs,
            validation_every_steps=validation_every_steps,
        ),
        budget=budget,
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
@common_prefetch_options
@click.option(
    "--weights",
    type=click.Choice(["auto", "ema", "raw"]),
    default="auto",
    show_default=True,
    help=(
        "Which student weights to score: auto reads the EMA average when the "
        "recipe declares an EMAHook and the trained weights otherwise; ema "
        "fails when the checkpoint holds no average; raw scores the trained "
        "weights regardless. The choice is recorded as the report's weights."
    ),
)
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
    prefetch_factor: int,
    num_streams: int,
    pin_memory: bool,
    use_streams: bool,
    weights: EvaluatedWeights,
    map_location: str | None,
    json_out: Path | None,
) -> None:
    """Score a trained student against the recipe's holdout and acceptance bars.

    By default, a recipe whose student.hooks carry an EMAHook is gated on the
    averaged weights that hook trained rather than on the live ones, the way
    the run's own validation reads them. --weights ema insists on the averaged
    weights, and --weights raw scores the trained weights instead. The line
    above the report names which weights were scored, and the report records
    the same "ema" or "raw" marker, so a --json-out export stays attributable
    once a sweep assembles several of them. --map-location names the one
    device the student, the teacher, the holdout, and the errors are all
    placed on.

    Exits non-zero when a bar is not cleared, so a sweep can gate on the
    command rather than on reading its output.
    """
    job = _load_recipe(path)
    evaluation = job.evaluation
    if evaluation is None and holdout_path is None:
        raise click.ClickException(
            "The recipe records no evaluation section, so there is no holdout "
            "to score against. Add one, or pass --holdout."
        )
    resolved_holdout = holdout_path or evaluation.holdout_path
    holdout_field = (
        "--holdout" if holdout_path is not None else "evaluation.holdout_path"
    )
    device = (
        torch.device(map_location) if map_location else primary_strategy_device(job)
    )
    student, scored_weights, weights_detail = _load_evaluated_student(
        job,
        student_checkpoint,
        checkpoint_index=checkpoint_index,
        device=device,
        weights=weights,
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
            holdout = build_dataloader(
                job,
                stack,
                device=device,
                batch_size=batch_size
                or (None if evaluation is None else evaluation.batch_size),
                shuffle=False,
                drop_last=False,
                prefetch_factor=prefetch_factor,
                num_streams=num_streams,
                use_streams=use_streams,
                pin_memory=pin_memory,
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
                    weights=scored_weights,
                )
            ],
            None if evaluation is None else evaluation.thresholds,
        )
    except ValueError as exc:
        raise click.ClickException(
            f"the acceptance report could not be formed from the recipe's bars: {exc}"
        ) from exc
    console.print(f"weights: {weights_detail}")
    console.print(report)
    if json_out is not None:
        write_or_print(json_safe(report.to_dict()), json_out)
    if not report.accepted:
        raise click.exceptions.Exit(1)
