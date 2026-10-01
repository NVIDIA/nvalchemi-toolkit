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
"""Knowledge-distillation strategy built on :class:`TrainingStrategy`."""

from __future__ import annotations

import dataclasses
import inspect
import warnings
from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager, nullcontext
from typing import TYPE_CHECKING, Annotated, Any

import torch
from pydantic import Field, PrivateAttr, model_validator

from nvalchemi._serialization import _dtype_deserialize, _import_cls
from nvalchemi._typing import ModelOutputs
from nvalchemi.data.datapipes.dataset import (
    BatchDatasetProtocol,
    dataset_device,
    same_device,
)
from nvalchemi.dynamics.sinks import HostMemory
from nvalchemi.dynamics.structure_sampler import WithinBudget
from nvalchemi.models.base import BaseModelMixin
from nvalchemi.training import TrainingStage
from nvalchemi.training import _spec_utils as strategy_spec
from nvalchemi.training import _strategy_validation as strategy_validation
from nvalchemi.training.distillation._attach import _attach_teacher_labels
from nvalchemi.training.distillation.config import (
    OnPolicyConfig,
    ResizableSink,
    _check_sole_migrator,
    _check_structure_status,
)
from nvalchemi.training.distillation.hooks import (
    TeacherLabelHook,
    _ConvergedFrameHook,
    _DivergenceHook,
    _run_local_keys,
    _score_and_attach,
    _strip_replay_frame,
    nonfinite_divergence,
)
from nvalchemi.training.distillation.replay import (
    _SCHEMA_REMEDY,
    ReplayBuffer,
    _frame_schema,
    build_mixed_loader,
)
from nvalchemi.training.distillation.scoring import (
    _EMBEDDING_KEYS,
    _TEACHER_FIELD_PREFIX,
    InProcessTeacherScorer,
    _reject_foreign_fields,
    scorer_fields,
    signal_fields,
    signal_for_field,
)
from nvalchemi.training.distillation.seeding import InitialStructures
from nvalchemi.training.distributed import get_rank, get_world_size
from nvalchemi.training.losses.composition import loss_target_keys
from nvalchemi.training.runtime import (
    eval_configured_models,
    evaluating,
    freeze_unconfigured_models,
    move_to_devices,
    train_configured_models,
)
from nvalchemi.training.strategy import TrainingStrategy

if TYPE_CHECKING:
    from torch.optim.lr_scheduler import LRScheduler

    from nvalchemi.data.batch import Batch
    from nvalchemi.dynamics.sinks import DataSink
    from nvalchemi.hooks import TrainContext
    from nvalchemi.training import ValidationConfig
    from nvalchemi.training.losses.composition import (
        BaseLossFunction,
        ComposedLossFunction,
    )

__all__ = ["DistillationStrategy", "default_distillation_fn"]

_REQUIRED_MODELS = frozenset({"student", "teacher"})
"""Model names every distillation strategy must be given."""

_PREDICTION_KEY_PREFIX = "predicted_"
"""Prefix the stock training function publishes every student output under."""


def default_distillation_fn(
    models: Mapping[str, BaseModelMixin], batch: Batch
) -> dict[str, torch.Tensor]:
    """Run the student forward pass and prefix output keys with ``predicted_``.

    The teacher is never called here: teacher knowledge reaches the loss as
    ``teacher_*`` batch fields, either written offline by
    :func:`~nvalchemi.training.distillation.label_dataset` or attached to the
    batch by :meth:`DistillationStrategy.attach_teacher_labels`.

    Parameters
    ----------
    models : Mapping[str, BaseModelMixin]
        Named models of the strategy; only ``"student"`` is read.
    batch : Batch
        Input batch of atomic graphs.

    Returns
    -------
    dict[str, torch.Tensor]
        Predictions keyed by ``predicted_<output_name>`` with ``None`` outputs
        omitted.
    """
    outputs: ModelOutputs = models["student"](batch)
    return {
        f"{_PREDICTION_KEY_PREFIX}{key}": value
        for key, value in outputs.items()
        if value is not None
    }


def _derived_teacher_signals(loss_fn: ComposedLossFunction) -> frozenset[str]:
    """Return the built-in teacher signals the loss composition's targets require.

    A ``teacher_*`` target no built-in signal populates is a custom teacher
    field — one :func:`~nvalchemi.training.distillation.label_dataset` persisted
    from a custom scorer — that the batch must already carry, so it is passed
    over here rather than refused.
    """
    signals: set[str] = set()
    for key in loss_target_keys(loss_fn):
        signal = (
            signal_for_field(key) if key.startswith(_TEACHER_FIELD_PREFIX) else None
        )
        if signal is not None:
            signals.add(signal)
    return frozenset(signals)


def _set_rebuild_overrides(
    strategy_cls: type[DistillationStrategy], overrides: Mapping[str, Any]
) -> dict[str, Any]:
    """Return the *overrides* that are set, checked against *strategy_cls*'s signature.

    A subclass overriding ``from_spec_dict`` with the base signature of an
    earlier release knows nothing of a later optional keyword, so an unset
    override is dropped instead of forwarded and a plain rebuild keeps working.
    A set one the subclass cannot take is refused here, where the keyword and
    the class can be named, rather than as a bare ``TypeError`` from the call.
    """
    parameters = inspect.signature(strategy_cls.from_spec_dict).parameters
    takes_var_keyword = any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )
    forwarded: dict[str, Any] = {}
    for name, value in overrides.items():
        if value is None:
            continue
        if not takes_var_keyword and name not in parameters:
            raise TypeError(
                f"from_spec_dict: {strategy_cls.__name__}.from_spec_dict does not "
                f"accept the {name!r} keyword, so the supplied {value!r} cannot be "
                f"applied; add {name} to its signature or drop the override."
            )
        forwarded[name] = value
    return forwarded


@dataclasses.dataclass(frozen=True)
class _RelaxationLifecycle:
    """Machinery a relaxation segment loop drives between its segments."""

    capture: _ConvergedFrameHook
    structures: InitialStructures
    divergence: _DivergenceHook


@contextmanager
def _relaxation_lifecycle(
    config: OnPolicyConfig, state: Batch, label_hook: TeacherLabelHook | None = None
) -> Iterator[_RelaxationLifecycle | None]:
    """Install the convergence machinery of a relaxation run on the propagator.

    The config's :attr:`~OnPolicyConfig.convergence_criterion` is installed on
    the propagator in two roles. As a registered ``AFTER_STEP`` hook, it
    migrates the status of converged graphs, which freezes them. The
    propagator reports that migration at ``ON_GRADUATE``, where the capture
    hook stores the frame, and the segment boundary reads the migrated
    status. As the propagator's ``convergence_hook``, it is the detector that
    ends a chunk early once every graph has converged. A detector the
    propagator was built with is restored on exit.

    The criterion must be the only status migrator. A looser migrator would
    graduate a structure before this criterion accepts it, and neither capture
    route would store it. The config rejects such a migrator at construction,
    and the check runs again here for a hook registered since then. The
    lifecycle must also be the only source of refills, because a mid-segment
    refill compacts the surviving graphs and invalidates the divergence hook's
    per-graph record.

    A graph diverges when the config's :attr:`~OnPolicyConfig.divergence`
    predicate flags it; by default, when its state is no longer finite. A
    divergence hook registered after the criterion calls the predicate once
    per step, freezes each flagged graph at ``exit_status`` without capturing
    it, and records the verdict. The capture hook and the segment boundary
    read that record and never call the predicate themselves. The segment
    boundary then retires a diverged graph like a converged one.

    Parameters
    ----------
    config : OnPolicyConfig
        Segment-loop configuration, holding the criterion.
    state : Batch
        Initial batch, already carrying the bookkeeping
        :meth:`~nvalchemi.training.distillation.InitialStructures.initial_batch`
        stamped on it.
    label_hook : TeacherLabelHook | None, optional
        Labeling hook of the run's path route. The capture hook asks it which
        graphs it stored and skips a graph whose final frame it stored on the
        step the graph graduated. Default ``None``.

    Yields
    ------
    _RelaxationLifecycle | None
        The machinery the segment loop drives, or ``None`` for a config that
        manages no lifecycle.

    Raises
    ------
    ValueError
        If the propagator already carries a status-migrating criterion, if it
        carries a sampler of its own, or if the configured criterion migrates
        off a status no initial structure carries.
    """
    criterion = config.convergence_criterion
    if criterion is None:
        yield None
        return
    dynamics = config.dynamics
    _check_sole_migrator(dynamics, criterion)
    if dynamics.sampler is not None:
        raise ValueError(
            "The relaxation lifecycle owns the refill as well as graduation, "
            "so the propagator must carry no sampler of its own; got "
            f"{type(dynamics.sampler).__name__!r}. A propagator that refills "
            "inside run reorders the batch mid-segment, so the capture hook "
            "would store the wrong structures and lose the minima. Give "
            "OnPolicyConfig.initial_structures the same budget instead, which "
            "backfills from the same dataset at the segment boundary, and "
            "leave the propagator's sampler unset."
        )
    _check_structure_status(state, criterion)
    divergence = _DivergenceHook(config.divergence or nonfinite_divergence)
    capture = _ConvergedFrameHook(
        sink=HostMemory(capacity=state.num_graphs),
        divergence=divergence,
        path=label_hook,
    )
    detector = dynamics.convergence_hook
    # Registered ahead of the labeling hook, so a graph that converges or
    # diverges on this step graduates before the path route reads its status:
    # no converged graph is stored twice, and no diverged one at all.
    dynamics.register_hook(criterion)
    dynamics.register_hook(divergence)
    dynamics.register_hook(capture)
    dynamics.convergence_hook = criterion
    try:
        yield _RelaxationLifecycle(
            capture=capture,
            structures=config.initial_structures,
            divergence=divergence,
        )
    finally:
        dynamics.convergence_hook = detector
        dynamics.hooks.remove(criterion)
        dynamics.hooks.remove(divergence)
        dynamics.hooks.remove(capture)


def _propagates_student(propagator_model: object, student: BaseModelMixin) -> bool:
    """Return whether *propagator_model* is *student* or a model composing it."""
    if propagator_model is student:
        return True
    modules = getattr(propagator_model, "modules", None)
    return callable(modules) and any(module is student for module in modules())


def _student_label_dtype(student: BaseModelMixin) -> torch.dtype | None:
    """Return the dtype teacher labels are cast to for *student*.

    The first floating-point parameter decides, floored at single precision: a
    store reads labels back at the dataset's ``positions`` dtype, float32 for
    essentially every dataset, so a narrower on-the-fly label would disagree
    with the persisted one. A student exposing no parameters gets ``None``,
    which keeps the teacher's own dtype.
    """
    parameters = getattr(student, "parameters", None)
    if not callable(parameters):
        return None
    for parameter in parameters():
        if parameter.is_floating_point():
            if parameter.dtype.itemsize < torch.float32.itemsize:
                return torch.float32
            return parameter.dtype
    return None


def _segment_sink(config: OnPolicyConfig, num_graphs: int) -> DataSink:
    """Return the sink that stages one segment's labeled frames.

    A segment captures at most one frame per trajectory per labeled step,
    including the forced last frame. The sink must therefore hold
    ``(generation_steps + 1) * num_graphs`` frames. Without a configured
    ``capture_sink``, a host-memory sink of that capacity is built. A
    configured sink is kept. When it is too small, it is grown through
    ``resize(capacity)`` if it satisfies
    :class:`~nvalchemi.training.distillation.ResizableSink`.

    Parameters
    ----------
    config : OnPolicyConfig
        Segment-loop configuration, holding the optional ``capture_sink``.
    num_graphs : int
        Number of trajectories the segment propagates.

    Returns
    -------
    DataSink
        Empty sink of at least the segment's capacity.

    Raises
    ------
    ValueError
        If the configured sink still holds frames, which the segment boundary
        would drain into the replay buffer as generated frames, or if it is too
        small and is not a ``ResizableSink``.
    """
    capacity = (config.generation_steps + 1) * num_graphs
    sink = config.capture_sink
    if sink is None:
        return HostMemory(capacity=capacity)
    if len(sink) > 0:
        raise ValueError(
            "OnPolicyConfig.capture_sink must be empty when a segment starts, "
            "because everything it holds is drained into the replay buffer at "
            f"the segment boundary as generated frames; got {len(sink)!r} frames "
            f"in a {type(sink).__name__}. Drain or zero it first."
        )
    if sink.capacity < capacity:
        if not isinstance(sink, ResizableSink):
            raise ValueError(
                "OnPolicyConfig.capture_sink must hold every frame one segment "
                "can capture, (generation_steps + 1) per trajectory: "
                f"{capacity!r} for {num_graphs!r} trajectories over "
                f"{config.generation_steps!r} steps; got a {type(sink).__name__} "
                f"of capacity {sink.capacity!r} without a resize method. Build it "
                "with at least that capacity, or give it resize(capacity)."
            )
        sink.resize(capacity)
    return sink


def _to_device(batch: Batch, device: torch.device) -> Batch:
    """Return *batch* on *device*, copying without blocking only into device memory.

    A non-blocking copy into host memory returns before the transfer
    completes. The moved batch's ``segment_lengths`` and ``batch_ptr`` are read
    on the host right away, so a copy into host memory blocks.
    """
    return batch.to(device, non_blocking=device.type != "cpu")


class _TeacherLabelHook:
    """Label the batch a forward pass is about to consume, training or validation.

    The first batch the seam labels raises one :class:`UserWarning` per strategy
    naming the missing fields, since from then on every such batch costs a
    teacher pass that a store written by
    :func:`~nvalchemi.training.distillation.label_dataset` would have spared.
    """

    frequency = 1
    stage = TrainingStage.BEFORE_FORWARD

    def __call__(self, ctx: TrainContext, stage: TrainingStage) -> None:  # noqa: ARG002
        """Attach the teacher fields the upcoming batch is missing."""
        strategy: DistillationStrategy = ctx.workflow
        if ctx.batch is None or not strategy.label_missing:
            return
        missing = strategy._missing_teacher_fields(ctx.batch)
        if not missing:
            return
        strategy.attach_teacher_labels(ctx.batch)
        if strategy._warned_label_seam:
            return
        strategy._warned_label_seam = True
        warnings.warn(
            "DistillationStrategy is labeling batches on the fly: a batch reached "
            f"the forward pass without the teacher fields {missing!r}, so a teacher "
            "pass now runs for every such batch, in training and validation alike. "
            "Label the dataset ahead of time with label_dataset to avoid the cost, "
            "or set label_missing=False to surface it as a missing target instead.",
            UserWarning,
            stacklevel=2,
        )


class DistillationStrategy(TrainingStrategy):
    """Train a student against a frozen teacher's signals.

    A :class:`~nvalchemi.training.TrainingStrategy` over the named models
    ``"student"`` and ``"teacher"``. The teacher is frozen by omission — it must
    not appear in ``optimizer_configs`` — while the student and any auxiliary
    model must be configured with one. Teacher knowledge reaches the loss as
    ``teacher_*`` batch fields, so a built-in term distills by pointing its
    ``target_key`` at one (``EnergyMSELoss(target_key="teacher_energy")``) and
    :class:`~nvalchemi.training.distillation.AtomicEnergyMatchingLoss` reads
    ``teacher_atomic_energies``; mixing teacher and reference targets is
    ordinary loss composition.

    The signals the teacher is asked for are derived from the ``teacher_*``
    targets the training loss and a ``validation_config`` loss read, or named
    in ``teacher_signals``, which must cover the derived set. They are checked
    against the teacher's outputs at construction, as are both losses'
    prediction keys against the outputs the student actually computes whenever
    the effective training or validation function is the stock
    :func:`default_distillation_fn`. Nothing re-runs on assignment, so pass
    ``validation_config`` to the constructor or, when rebuilding from a spec
    that excludes it, to :meth:`from_spec_dict`. Every resolved signal is required
    on every batch: a batch missing any resolved field is labeled on the fly by
    an internal ``BEFORE_FORWARD`` hook, in training and validation alike,
    unless ``label_missing=False`` leaves it to surface as a missing loss
    target, while a store written by
    :func:`~nvalchemi.training.distillation.label_dataset` with the same signal
    set trains with no teacher pass at all. A ``teacher_*`` target no built-in
    signal populates is a custom field such a store carries: it is neither
    derived nor attached, and a batch lacking it fails as a missing target.
    See :ref:`training-distillation-api` for the full contract.

    Setting ``on_policy`` switches :meth:`run` to the on-policy segment loop
    instead. In each segment, the student's own propagator generates frames
    and the teacher labels them. The labeled frames accumulate in a replay
    buffer. The segment then trains on a mixture of that buffer and
    ``reference_dataset`` at the configured ``replay_ratio``. The propagator
    holds the very module the optimizer updates, so every segment generates
    from a fresher policy than the last. That is what makes the data
    on-policy, and it is why the propagator's model is checked for object
    identity with ``models["student"]`` at construction. For a relaxation
    propagator, set ``OnPolicyConfig.fmax``. Converged structures are then
    stored once, graduate out of the batch at the segment boundary, and are
    replaced by fresh initial structures. The buffer therefore keeps filling
    with structures that are still moving.

    Raises
    ------
    ValueError
        If ``models`` is not a named mapping containing ``"student"`` and
        ``"teacher"``, if the teacher is given an optimizer config, if the
        student or an auxiliary model is not, if a loss component reads a
        prediction the student does not compute or names one outside the
        ``predicted_`` namespace under the stock ``training_fn``, if an explicit
        ``teacher_signals`` omits a signal a loss needs, if no built-in teacher
        signal is requested at all, if the teacher cannot produce a requested signal,
        if the teacher is a composition that plans more than one neighbor-list
        source, or if ``label_dtype`` is not a floating-point dtype. In
        on-policy mode, also if the run is sized in epochs, if the propagator
        holds neither the student nor a model composing it, if
        ``replay_ratio`` and ``reference_dataset`` disagree (a ratio below
        ``1`` requires a reference dataset and a ratio of ``1`` rejects one),
        if ``reference_dataset`` is empty or emits on an accelerator the run
        does not train on, if ``replay_device`` or the reference dataset's
        fields cannot be mixed with generated frames, or if the propagator's
        scorer and the reference dataset do not carry the same teacher fields.

    Examples
    --------
    Distill energies, forces, and the teacher's per-atom energy decomposition
    from a store written by :func:`label_dataset`:

    >>> import torch
    >>> from nvalchemi.training import EnergyMSELoss, ForceMSELoss, OptimizerConfig
    >>> from nvalchemi.training.distillation import (
    ...     DistillationStrategy,
    ...     AtomicEnergyMatchingLoss,
    ... )
    >>> loss_fn = (
    ...     EnergyMSELoss(target_key="teacher_energy")
    ...     + ForceMSELoss(target_key="teacher_forces", normalize_by_atom_count=True)
    ...     + 0.1 * AtomicEnergyMatchingLoss()
    ... )
    >>> strategy = DistillationStrategy(  # doctest: +SKIP
    ...     models={"student": student, "teacher": teacher},
    ...     optimizer_configs={
    ...         "student": [OptimizerConfig(optimizer_cls=torch.optim.Adam)]
    ...     },
    ...     loss_fn=loss_fn,
    ...     num_steps=1_000,
    ... )
    >>> strategy.run(labeled_loader)  # doctest: +SKIP

    Notes
    -----
    Label precision is the scorer's decision. The strategy's own scorer is an
    :class:`~nvalchemi.training.distillation.InProcessTeacherScorer`, which
    disables autocast. It casts labels to ``label_dtype`` when one is given.
    By default the label dtype is inferred as the student's first
    floating-point parameter dtype, never below single precision, so a
    ``bfloat16`` or ``float16`` student gets float32 labels and needs
    ``dtype_policy="prediction_to_target"`` on its loss terms; a float64 student
    reads float32 back from a store and needs a ``dtype_policy`` too. Set
    ``label_dtype`` explicitly when the inference guesses wrong, as for a
    mixed-precision student whose first parameter is not representative or a
    student exposing no parameters at all, which otherwise keeps the teacher's
    own dtype. Labels are attached to the device-placed copy the strategy trains
    on, not the caller's batch, so a loader replaying the same systems costs one
    teacher pass per epoch, and the first batch labeled this way raises one
    :class:`UserWarning` naming the fields it lacked.

    Teacher conservativeness is not validated: a teacher predicting forces from
    its own head is first class, since every signal is detached before the
    student sees it. A teacher composition planning more than one neighbor-list
    source is refused, as by
    :class:`~nvalchemi.training.distillation.InProcessTeacherScorer`.

    The labeling hook is never serialized; :meth:`from_spec_dict` and
    :meth:`load_checkpoint` re-register it ahead of the caller's hooks and
    replace any carried copy. ``ValidationConfig(use_ema="auto")`` validates
    the EMA student against the live teacher (``model_source="mixed"``), while
    ``use_ema="always"`` also demands an inference-slot entry for the teacher.
    Checkpoints serialize every model, teacher included, so size the checkpoint
    interval for a large teacher.

    ``on_policy`` and ``reference_dataset`` hold live runtime objects that no
    spec can describe. :meth:`to_spec_dict` therefore omits them and warns. A
    strategy rebuilt from such a spec runs offline until they are supplied
    again.
    """

    teacher_signals: Annotated[
        frozenset[str] | None,
        Field(
            description=(
                "Teacher signals produced for every scored batch. ``None`` "
                "derives them from the ``teacher_*`` targets the training and "
                "validation losses read; an explicit set must cover those and "
                "may request more, at the cost of re-scoring every batch a "
                "store labeled without the extra fields delivers."
            )
        ),
    ] = None
    label_missing: Annotated[
        bool,
        Field(
            description=(
                "Whether a batch lacking the required ``teacher_*`` fields is "
                "labeled on the fly by a teacher forward pass, in training and "
                "validation alike. ``False`` skips the teacher, so an unlabeled "
                "batch surfaces as a missing loss target."
            )
        ),
    ] = True
    label_dtype: Annotated[
        torch.dtype | None,
        Field(
            description=(
                "Floating-point dtype teacher labels are cast to when a batch is "
                "labeled on the fly. ``None`` infers it from the student's first "
                "floating-point parameter, never below float32; an explicit dtype "
                "is passed to the teacher scorer verbatim."
            )
        ),
    ] = None
    on_policy: Annotated[
        OnPolicyConfig | None,
        Field(
            default=None,
            exclude=True,
            description=(
                "Segment-loop configuration turning ``run`` into on-policy "
                "distillation. ``None`` keeps the offline loop over the "
                "dataloader the caller passes to ``run``."
            ),
        ),
    ] = None
    reference_dataset: Annotated[
        BatchDatasetProtocol | None,
        Field(
            default=None,
            exclude=True,
            description=(
                "Teacher-labeled dataset the on-policy mixture draws "
                "its ``1 - replay_ratio`` share from. Required whenever the "
                "ratio is below 1, and read only in on-policy mode."
            ),
        ),
    ] = None

    _scorer: InProcessTeacherScorer | None = PrivateAttr(default=None)
    _teacher_fields: tuple[str, ...] = PrivateAttr(default=())
    _warned_label_seam: bool = PrivateAttr(default=False)
    _replay_buffer: ReplayBuffer | None = PrivateAttr(default=None)
    _validated_step: int | None = PrivateAttr(default=None)

    @property
    def replay_buffer(self) -> ReplayBuffer | None:
        """Frames generated so far, or ``None`` before an on-policy run starts.

        One buffer serves every :meth:`run` call on a strategy. A run
        continued with a larger ``num_steps`` therefore keeps training on
        everything generated so far instead of discarding and regenerating it.
        The trajectory is still reseeded on each call.
        """
        return self._replay_buffer

    @property
    def teacher_scorer(self) -> InProcessTeacherScorer:
        """Scorer producing the resolved teacher signals for one batch."""
        if self._scorer is None:
            raise RuntimeError(
                "DistillationStrategy has no teacher scorer; it is built during "
                "validation and must not be cleared."
            )
        return self._scorer

    @model_validator(mode="before")
    @classmethod
    def _default_distillation_training_fn(cls, data: Any) -> Any:
        """Fall back to the stock student-forward training function."""
        if not isinstance(data, dict):
            return data
        normalized = dict(data)
        if normalized.get("training_fn") is None:
            normalized["training_fn"] = default_distillation_fn
        return normalized

    @model_validator(mode="before")
    @classmethod
    def _prepend_labeling_hook(cls, data: Any) -> Any:
        """Put the internal teacher-labeling hook ahead of the caller's hooks.

        A seam carried in the incoming hooks is replaced rather than kept, so
        rebuilding a strategy from a live one's ``hooks`` leaves exactly one
        labeling hook, still ahead of every caller hook.
        """
        if not isinstance(data, dict):
            return data
        normalized = dict(data)
        normalized["hooks"] = [
            _TeacherLabelHook(),
            *(
                hook
                for hook in (normalized.get("hooks") or [])
                if not isinstance(hook, _TeacherLabelHook)
            ),
        ]
        return normalized

    @model_validator(mode="after")
    def _validate_distillation(self) -> DistillationStrategy:
        """Enforce the student/teacher contract and resolve the teacher signals."""
        missing_models = _REQUIRED_MODELS - set(self.models)
        if self.single_model_input or missing_models:
            raise ValueError(
                "DistillationStrategy needs a named-model mapping holding "
                f"'student' and 'teacher'; got models={sorted(self.models)!r}."
            )
        if "teacher" in self.optimizer_configs:
            raise ValueError(
                "The teacher is frozen by omission, so it must not appear in "
                f"optimizer_configs; got {sorted(self.optimizer_configs)!r}."
            )
        unconfigured = set(self.models) - set(self.optimizer_configs) - {"teacher"}
        if unconfigured:
            raise ValueError(
                "Every model but the teacher must be given an optimizer config; "
                f"got unconfigured {sorted(unconfigured)!r}."
            )
        if self.label_dtype is not None and not self.label_dtype.is_floating_point:
            raise ValueError(
                "label_dtype must be a floating-point dtype or None; got "
                f"{self.label_dtype!r}."
            )
        self._validate_student_outputs()
        signals = self._resolve_teacher_signals()
        self._scorer = InProcessTeacherScorer(
            self.models["teacher"],
            signals,
            dtype=(
                _student_label_dtype(self.models["student"])
                if self.label_dtype is None
                else self.label_dtype
            ),
        )
        self._teacher_fields = signal_fields(signals)
        return self

    def _validate_student_outputs(self) -> None:
        """Check both losses' prediction keys against what the student computes.

        Only under the stock function: ``default_distillation_fn`` emits exactly
        ``active_outputs`` intersected with ``outputs``, so a narrowed student
        is caught here rather than on its first batch. The validation loss is
        checked whenever ``validation_fn`` falling back to ``training_fn`` is
        the stock one.
        """
        if self.training_fn is default_distillation_fn:
            self._validate_prediction_keys(self.loss_fn.components, "training")
        validation = self.validation_config
        if validation is None or validation.loss_fn is None:
            return
        if (validation.validation_fn or self.training_fn) is default_distillation_fn:
            self._validate_prediction_keys(validation.loss_fn.components, "validation")

    def _validate_prediction_keys(
        self, components: Sequence[BaseLossFunction], side: str
    ) -> None:
        """Check one composition's prediction keys, naming *side* in every error."""
        student = self.models["student"]
        declared = student.model_config.outputs
        active = student.output_data()
        for component in components:
            key = getattr(component, "prediction_key", None)
            if key is None:
                continue
            label = f"{side} loss component {type(component).__name__!r}"
            if not key.startswith(_PREDICTION_KEY_PREFIX):
                raise ValueError(
                    f"The {label} reads prediction_key={key!r}, which "
                    "default_distillation_fn never emits: it publishes every "
                    f"student output under {_PREDICTION_KEY_PREFIX}<output>. "
                    "Rename the key into that namespace, or pass a training_fn "
                    "that owns its own convention."
                )
            output = key.removeprefix(_PREDICTION_KEY_PREFIX)
            if output in active:
                continue
            if output in _EMBEDDING_KEYS:
                raise ValueError(
                    f"The {label} reads prediction_key={key!r}, which the stock "
                    "training_fn cannot produce: embeddings come from the "
                    "student's compute_embeddings(), not from its forward pass. "
                    "Pass a training_fn that calls compute_embeddings and returns "
                    f"the embedding under {key!r}."
                )
            if output in declared:
                raise ValueError(
                    "Student declares but does not compute the output required by "
                    f"the {label} reading prediction_key={key!r}; got "
                    f"active_outputs={sorted(active)!r}, missing {output!r}. Add "
                    "it to the student's model_config.active_outputs."
                )
            raise ValueError(
                f"Student cannot produce the output required by the {label} "
                f"reading prediction_key={key!r}; got outputs={sorted(declared)!r}, "
                f"missing {output!r}."
            )

    def _resolve_teacher_signals(self) -> frozenset[str]:
        """Return the signals both losses need, widened by an explicit request."""
        derived = {"training": _derived_teacher_signals(self.loss_fn)}
        validation = self.validation_config
        if validation is not None and validation.loss_fn is not None:
            derived["validation"] = _derived_teacher_signals(validation.loss_fn)
        required: frozenset[str] = frozenset().union(*derived.values())
        resolved = required if self.teacher_signals is None else self.teacher_signals
        uncovered = {
            side: sorted(signals - resolved)
            for side, signals in derived.items()
            if signals - resolved
        }
        if uncovered:
            raise ValueError(
                "teacher_signals must cover every teacher target the training and "
                f"validation losses read; got {sorted(resolved)!r}, missing "
                f"{uncovered!r}."
            )
        if not resolved:
            raise ValueError(
                "DistillationStrategy needs at least one teacher signal; got no "
                "built-in teacher_* target in the training or validation loss and "
                f"teacher_signals={self.teacher_signals!r}. A custom teacher_* field "
                "the batch already carries is not a signal; name one in "
                "teacher_signals or read a built-in teacher target."
            )
        return resolved

    @model_validator(mode="after")
    def _validate_on_policy(self) -> DistillationStrategy:
        """Enforce the segment loop's duration, ownership, and mixture contract."""
        if self.on_policy is None:
            if self.reference_dataset is not None:
                raise ValueError(
                    "reference_dataset is the on-policy mixture's reference share "
                    "and is read only by the segment loop; got it set alongside "
                    "on_policy=None. Offline distillation trains on the "
                    "dataloader passed to run()."
                )
            return self
        if self.num_steps is None:
            raise ValueError(
                "On-policy distillation is sized in optimizer steps: every "
                "segment builds its own loader, so there is no fixed epoch to "
                f"convert. Got num_epochs={self.num_epochs!r}; set num_steps "
                "instead."
            )
        propagator_model = getattr(self.on_policy.dynamics, "model", None)
        if not _propagates_student(propagator_model, self.models["student"]):
            held = (
                "no model at all"
                if propagator_model is None
                else f"a separate {type(propagator_model).__name__} instance"
            )
            raise ValueError(
                "OnPolicyConfig.dynamics must propagate the very module "
                "registered as models['student'], on its own or composed into "
                "a larger model: the data is on-policy only because each "
                "optimizer step is immediately visible to the propagator. Got "
                f"a propagator holding {held}; build the dynamics around the "
                "student object itself."
            )
        if self.on_policy.replay_ratio < 1.0 and self.reference_dataset is None:
            raise ValueError(
                "A replay_ratio below 1 mixes reference data into every batch, "
                f"so reference_dataset is required; got replay_ratio="
                f"{self.on_policy.replay_ratio!r} and reference_dataset=None."
            )
        if self.on_policy.replay_ratio == 1.0 and self.reference_dataset is not None:
            raise ValueError(
                "replay_ratio=1 draws every sample of every batch from the "
                "replay buffer, so a reference dataset would only be checked for "
                "schema and device and never sampled; got replay_ratio=1.0 "
                f"alongside a {type(self.reference_dataset).__name__} "
                "reference_dataset. Drop reference_dataset, or lower replay_ratio "
                "to mix it in."
            )
        if self.reference_dataset is not None and len(self.reference_dataset) == 0:
            raise ValueError(
                "A replay_ratio below 1 draws part of every batch from "
                "reference_dataset, so it has to hold at least one sample; got an "
                f"empty {type(self.reference_dataset).__name__}. Pass the labeled "
                "reference set the mixture draws from, or set replay_ratio=1 to "
                "train on generated frames alone."
            )
        # One probe batch serves the device and schema checks alike.
        probe = (
            None
            if self.reference_dataset is None
            else self.reference_dataset.load_batches([[0]])[0]
        )
        self._validate_reference_device(probe)
        self._validate_mixture_device(probe)
        self._validate_reference_schema(probe)
        self._validate_generation_signals()
        return self

    def _validate_reference_device(self, probe: Batch | None) -> None:
        """Reject a reference dataset emitting on an accelerator the run does not train on.

        Parameters
        ----------
        probe : Batch | None
            One batch already drawn from ``reference_dataset``. A composed
            dataset, or a store opened without a device, is measured by the
            device of this batch. ``None`` when there is no reference dataset.
        """
        if self.reference_dataset is None:
            return
        reference_device = dataset_device(self.reference_dataset, probe)
        primary = self.devices[0]
        if reference_device.type == "cpu" or same_device(reference_device, primary):
            return
        raise ValueError(
            "A segment's mixture is collated on the reference dataset's own "
            "device before the strategy moves it, so a reference dataset that "
            "emits on an accelerator has to emit on the device the run trains "
            f"on; got a reference dataset emitting on {reference_device!s} and "
            f"devices[0]={primary!s}. A Zarr-backed Dataset resolves an unset "
            "device to CUDA whenever one is visible — open it as "
            f"Dataset(..., device={str(primary)!r}) to follow the run, or leave "
            "it in host memory."
        )

    def _validate_mixture_device(self, probe: Batch | None) -> None:
        """Reject a staging device the reference dataset cannot be collated with.

        Parameters
        ----------
        probe : Batch | None
            One batch already drawn from ``reference_dataset``. The device of a
            composed dataset or of a store opened without a device is read from
            it. ``None`` when there is no reference dataset to measure.
        """
        if self.reference_dataset is None or self.on_policy.replay_device is None:
            return
        reference_device = dataset_device(self.reference_dataset, probe)
        replay_device = torch.device(self.on_policy.replay_device)
        if same_device(reference_device, replay_device):
            return
        raise ValueError(
            "A mixed batch is collated before the strategy moves it, so the "
            "replay buffer and reference_dataset have to live on one device; "
            f"got replay_device={replay_device!s} and a reference dataset "
            f"emitting on {reference_device!s}. Leave replay_device unset to "
            "stage generated frames wherever the reference dataset lives, or "
            f"load the reference dataset on {replay_device!s}."
        )

    def _validate_reference_schema(self, probe: Batch | None) -> None:
        """Reject a reference dataset holding fields no generated frame can carry.

        The full schema comparison runs inside the first segment's
        :func:`~nvalchemi.training.distillation.build_mixed_loader`, after a
        whole generation phase has been paid for. The part that depends on
        nothing the run produces is checked here: the predictions, neighbor
        tensors, and bookkeeping that the labeling hook strips from every
        frame. In particular, this catches a store that
        :func:`~nvalchemi.training.distillation.label_dataset` wrote over a
        reference set, which keeps that set's ``energy`` and ``forces``.
        """
        if probe is None:
            return
        dropped = _run_local_keys()
        unmixable = sorted(
            name for name in _frame_schema(probe) if name.partition(".")[2] in dropped
        )
        if not unmixable:
            return
        raise ValueError(
            "reference_dataset carries fields no generated frame can, so the "
            "mixture would be rejected on the first segment's loader; got "
            f"{unmixable!r} on reference_dataset, which the labeling hook strips "
            f"from every frame it stores. {_SCHEMA_REMEDY}"
        )

    def _validate_generation_signals(self) -> None:
        """Check the propagator's teacher fields against the reference dataset and loss.

        A scorer that declares neither ``label_fields`` nor a set of built-in
        signals writes fields that cannot be known before it scores a batch.
        Both checks are then skipped with a warning. Running them against an
        empty set would reject a custom scorer that in fact produces exactly
        what the reference dataset carries.
        """
        generated = scorer_fields(self.on_policy.teacher_scorer)
        if generated is None:
            warnings.warn(
                "The propagator's scorer declares neither label_fields nor "
                "built-in signals, so the teacher fields it writes are unknown "
                "until the first segment has generated them: neither their "
                "parity with reference_dataset nor whether every generated "
                "frame is scored twice can be checked at construction, and a "
                "mismatch surfaces as a rejected mixture once a whole "
                "generation phase has been paid for. Got signals="
                f"{sorted(self.on_policy.teacher_scorer.signals)!r}; declare "
                "label_fields on the scorer to restore both checks.",
                UserWarning,
                stacklevel=2,
            )
            return
        _reject_foreign_fields(generated, "A scorer's label_fields")
        if self.reference_dataset is not None:
            stored = frozenset(
                field
                for field in self.reference_dataset.field_names
                if field.startswith(_TEACHER_FIELD_PREFIX)
            )
            if frozenset(generated) != stored:
                raise ValueError(
                    "Generated frames and reference_dataset must carry the same "
                    "teacher fields, because mixing them into one batch keeps "
                    f"only the fields both hold; got generation "
                    f"{sorted(generated)!r} and reference {sorted(stored)!r}. "
                    "Request the same signals on OnPolicyConfig.teacher_scorer, "
                    "or relabel the reference dataset with label_dataset."
                )
        self._warn_on_partial_generation_signals(generated)

    def _warn_on_partial_generation_signals(self, generated: tuple[str, ...]) -> None:
        """Warn when generated frames will be relabeled on their way into training.

        The comparison is by field rather than by signal name. A custom scorer
        that writes the fields the loss reads, under signal names of its own,
        is therefore not reported as missing them.
        """
        missing = frozenset(self._teacher_fields) - frozenset(generated)
        if missing:
            warnings.warn(
                "The propagator's scorer does not produce every teacher field "
                "the loss reads, so each generated frame is scored twice: once "
                "during generation and again on its way into a training step; "
                f"missing {sorted(missing)!r}. Request the signals populating "
                "those fields on OnPolicyConfig.teacher_scorer to pay the "
                "teacher once.",
                UserWarning,
                stacklevel=2,
            )

    def attach_teacher_labels(self, batch: Batch) -> bool:
        """Attach the teacher fields *batch* is missing, and report whether it did.

        A batch already carrying every resolved ``teacher_*`` field is left
        untouched, so pre-labeling a batch that later reaches :meth:`run` costs
        one teacher pass; a batch carrying only some of them is re-scored in
        full, since a partial set was written for a different signal set. The
        scorer is called inside whatever autocast region the training loop
        holds open. The strategy's own scorer disables autocast, so the labels
        match what :func:`~nvalchemi.training.distillation.label_dataset`
        persisted wherever the store returns the label dtype.

        Parameters
        ----------
        batch : Batch
            Batch to label in place, already on the teacher's device.

        Returns
        -------
        bool
            ``True`` when the teacher ran, ``False`` when *batch* already
            carried every resolved field.
        """
        if not self._missing_teacher_fields(batch):
            return False
        _attach_teacher_labels(batch, self.teacher_scorer.label(batch))
        return True

    def _missing_teacher_fields(self, batch: Batch) -> list[str]:
        """Return the resolved teacher fields *batch* does not carry."""
        return [field for field in self._teacher_fields if field not in batch]

    def run(self, dataloader: Iterable[Batch] | None = None) -> None:
        """Execute the offline training loop or the on-policy segment loop.

        Without ``on_policy``, this is
        :meth:`~nvalchemi.training.TrainingStrategy.run` over *dataloader*.
        With ``on_policy``, the strategy owns the loop and repeats three phases
        until ``num_steps`` optimizer steps have run. In the *generate* phase,
        the propagator advances the live state batch by ``generation_steps``
        steps; on the first segment, the state is seeded from
        ``initial_structures``. In the *label and capture* phase, a
        :class:`~nvalchemi.training.distillation.TeacherLabelHook` registered
        on the propagator labels every ``label_frequency`` steps and the
        segment's last frame. It copies each labeled frame into the capture
        sink, which is drained into the replay buffer. The capture sink is host
        memory unless ``capture_sink`` names another. In the *train* phase, a
        freshly built mixed loader draws ``training_steps_per_segment`` batches
        at ``replay_ratio``, and each batch goes through the ordinary
        per-batch stages.

        For relaxation propagators, whose trajectories end, an
        ``OnPolicyConfig.fmax`` threshold adds a fourth phase between
        generation and training: *graduate and backfill*. Converged structures
        are stored once, as the minimum they reached. They then leave the
        batch and are replaced by fresh initial structures while the source
        still holds any. Generation stops once the source runs dry and the last
        trajectory finishes. The remaining steps train on the frames already
        in the buffer.

        Parameters
        ----------
        dataloader : Iterable[Batch] | None, optional
            Batches to train on in offline mode. Default ``None``, which is
            required in on-policy mode and rejected otherwise.

        Raises
        ------
        ValueError
            If *dataloader* is ``None`` in offline mode or supplied in on-policy
            mode, if the on-policy loop is entered on more than one rank, if a
            segment's loader produces no batches, if the propagator already
            carries a status-migrating criterion or a sampler of its own, or if
            the configured criterion migrates off a status no initial structure
            carries.

        Warns
        -----
        UserWarning
            If a lifecycle-managed run runs out of trajectories and initial
            structures before reaching ``num_steps``. The remaining steps then
            train on the frames already generated. Also once per segment
            boundary that retires trajectories the ``divergence`` predicate
            flagged, by default those whose positions or forces stopped
            being finite.

        Notes
        -----
        One segment is one epoch. ``AFTER_EPOCH`` and epoch-cadence validation
        fire at segment boundaries, and step-cadence validation fires inside
        segments. The run closes with one final validation unless a cadence
        already validated at the final step. The segment is also the restart
        granularity. The propagator state is not checkpointed. A resumed run
        reseeds its trajectory, and a segment that a checkpoint interrupted is
        counted as finished on resume. A second call keeps the replay buffer
        the first call filled and reseeds only the trajectory.

        The student generates in evaluation mode and trains in training mode.
        A propagator model that merely composes the student is held in
        evaluation mode for the whole loop. Every mode is restored on exit.
        Generated frames are staged on ``reference_dataset``'s device unless
        ``replay_device`` names another. The loop is single-process: it
        refuses to start on more than one rank rather than regenerate the same
        frames on every rank. Splitting a built-in propagator's run into
        segments is exact, because ``run`` never resets ``step_count`` and the
        Langevin thermostat's generator is keyed on it. The progress of a
        segment that exits early is read from ``dynamics.step_count``. A
        :class:`~nvalchemi.dynamics.FusedStage` pays its priming forward pass
        once per segment. See :ref:`training-distillation-api` for the mixture
        and schema contract.

        Relaxation runs are the reason a segment can exit early, and
        ``OnPolicyConfig.fmax`` turns such a run into a lifecycle. For the
        duration of the loop, the criterion is registered ahead of the labeling
        hook and installed as the propagator's convergence detector. A
        converged structure is captured once, on the step its ``status``
        reaches ``exit_status``, and left out of every later path capture. At
        the segment boundary, initial structures are drawn to fill the room the
        graduates freed. A budgeted
        :class:`~nvalchemi.training.distillation.InitialStructures` draws them
        from the rows it has not served yet. An unbudgeted one draws them only
        with ``recycle=True``; otherwise the batch narrows. Once no trajectory
        is left and no initial structure remains to start one, the loop warns
        and trains on the buffer it has until ``num_steps``.
        """
        if self.on_policy is None:
            if dataloader is None:
                raise ValueError(
                    "Offline distillation trains on the caller's batches; got "
                    "run(dataloader=None) with on_policy=None. Pass a "
                    "dataloader, or configure on_policy to generate one."
                )
            super().run(dataloader)
            return
        if dataloader is not None:
            raise ValueError(
                "On-policy distillation builds its own loader every segment "
                "from reference_dataset and the replay buffer; got a "
                f"{type(dataloader).__name__} passed to run(). Set it as "
                "reference_dataset instead."
            )
        self._run_on_policy(self.on_policy)

    def _run_on_policy(self, config: OnPolicyConfig) -> None:
        """Drive generate-label-train segments until ``num_steps`` is reached.

        The rank shard is installed on the initial structures here rather than
        at construction, for two reasons. The world size is set by the
        launcher. Installing the shard also resets the position, which a second
        ``run()`` on one strategy needs: that call keeps the replay buffer and
        reseeds only the trajectory, so the source must start again at the
        front of its shard. A propagator model that merely composes the
        student is moved whole to the generation device. Only the named models
        move with the strategy, so a correction held only by the composition
        would otherwise stay on the device it was built on.
        """
        training_started = False
        strategy_context = nullcontext(self) if self._context_depth > 0 else self
        with strategy_context:
            self._prepare_setup_hooks()
            self._validate_runtime_devices()
            self._validate_single_process()
            self.models = move_to_devices(self.models, self.devices)
            propagator_model = config.dynamics.model
            if propagator_model is not self.models["student"] and isinstance(
                propagator_model, torch.nn.Module
            ):
                propagator_model.to(self.devices[0])
            self._run_setup_hooks()
            target_step_count = self._resolve_target_step_count(None)
            if self.step_count >= target_step_count:
                return
            self._close_interrupted_segment()
            self._apply_requires_grad_filter()
            try:
                primary_device = self.devices[0]
                flat_opts, flat_scheds = self._setup_runtime_optimizers(
                    rebuild=not self._resume_optimizer_state
                )
                config.initial_structures.shard(
                    get_rank(self.distributed_manager),
                    get_world_size(self.distributed_manager),
                )
                state = _to_device(
                    config.initial_structures.initial_batch(), primary_device
                )
                if self._replay_buffer is None:
                    self._replay_buffer = ReplayBuffer(
                        capacity=config.replay_capacity,
                        eviction=config.replay_eviction,
                        admission=config.replay_admission,
                        device=self._resolve_replay_device(config),
                    )
                buffer = self._replay_buffer
                propagator_model = config.dynamics.model
                held_propagator = (
                    evaluating(propagator_model)
                    if isinstance(propagator_model, torch.nn.Module)
                    and propagator_model is not self.models["student"]
                    else nullcontext()
                )
                # Only a lifecycle stores graduated graphs by another route;
                # a propagator that manages its own keeps every frame here.
                label_hook = TeacherLabelHook(
                    config.teacher_scorer,
                    frequency=config.label_frequency,
                    exit_status=None
                    if config.convergence_criterion is None
                    else config.dynamics.exit_status,
                )
                with _relaxation_lifecycle(config, state, label_hook) as lifecycle:
                    config.dynamics.register_hook(label_hook)
                    try:
                        # Freeze the teacher for both phases and keep the student in
                        # eval mode outside the training phase. A composition holding
                        # the student is not in models, so it is held separately.
                        with (
                            freeze_unconfigured_models(
                                self.models, self.optimizer_configs
                            ),
                            eval_configured_models(self.models, self.optimizer_configs),
                            held_propagator,
                        ):
                            while self.step_count < target_step_count:
                                if state is not None:
                                    state = self._generate_segment(
                                        config,
                                        state,
                                        label_hook,
                                        lifecycle,
                                        buffer,
                                        target_step_count,
                                    )
                                training_steps = min(
                                    config.training_steps_per_segment,
                                    target_step_count - self.step_count,
                                )
                                with train_configured_models(
                                    self.models, self.optimizer_configs
                                ):
                                    training_started = self._train_segment(
                                        config,
                                        buffer,
                                        training_steps=training_steps,
                                        target_step_count=target_step_count,
                                        training_started=training_started,
                                        flat_opts=flat_opts,
                                        flat_scheds=flat_scheds,
                                    )
                    finally:
                        config.dynamics.hooks.remove(label_hook)

                if self._last_batch is not None:
                    self._update_hook_snapshot(loss_out=None)
                    self._run_hooks(TrainingStage.AFTER_TRAINING, self._last_batch)
                    if (
                        self.validation_config is not None
                        and self._validated_step != self.step_count
                    ):
                        self.validate()
                        self._step_metric_schedulers()
            finally:
                self._restore_requires_grad_filter()

    def _validate_single_process(self) -> None:
        """Reject a multi-rank launch, whose segment loaders the loop does not shard.

        The world size is read at run time rather than at construction, for
        two reasons. A launcher has initialized the process group by then. An
        offline strategy built by the same script is also free to be
        distributed.
        """
        world_size = get_world_size(self.distributed_manager)
        if world_size == 1:
            return
        raise ValueError(
            "On-policy distillation runs on one process: each rank would build "
            "its segment loader from its own replay buffer and an unsharded "
            "reference dataset, so the ranks would train on generated frames no "
            "rank shares and on the same reference samples; got "
            f"world_size={world_size!r}. Run the segment loop on one process, or "
            "distill offline: label the dataset with label_dataset and train the "
            "store with a DDPHook, which shards it as usual."
        )

    def _close_interrupted_segment(self) -> None:
        """Count a segment that a restored run stopped partway through as finished.

        Each segment builds its own loader, so the batches an interrupted
        segment had left are lost with it, and the trajectory is reseeded
        anyway. Closing the segment keeps ``BEFORE_EPOCH`` firing for the
        resumed segment and keeps ``epoch_step_count`` inside the segment
        budget. It also moves the mixture sampler past the epoch index the
        interrupted segment drew with. The parent's
        :meth:`_prepare_epoch_step_count` does not fit here: it reconciles
        against a fixed number of batches per epoch, and a run that continues
        from offline epochs of another size has none.
        """
        if self.epoch_step_count == 0:
            return
        self.epoch_count += 1
        self.epoch_step_count = 0
        self._refresh_hook_counters()

    def _validation_checkpoint(self, stage: TrainingStage) -> bool:
        """Run a scheduled validation and remember the step it fired at.

        The segment loop's closing validation is skipped when a cadence already
        validated at the final step, so a metric-driven scheduler is never stepped
        twice on one set of metrics.
        """
        fired = super()._validation_checkpoint(stage)
        if fired:
            self._validated_step = self.step_count
        return fired

    def _train_segment(
        self,
        config: OnPolicyConfig,
        buffer: ReplayBuffer,
        *,
        training_steps: int,
        target_step_count: int,
        training_started: bool,
        flat_opts: list[torch.optim.Optimizer],
        flat_scheds: list[LRScheduler | None],
    ) -> bool:
        """Train one segment's mixture and close it as an epoch.

        Returns
        -------
        bool
            Whether the ``BEFORE_TRAINING`` stage has fired by now.
        """
        loader = build_mixed_loader(
            self.reference_dataset,
            buffer,
            replay_ratio=config.replay_ratio,
            batch_size=config.batch_size,
            num_batches=training_steps,
            seed=config.seed,
        )
        self._set_sampler_epoch(loader)
        primary_device = self.devices[0]
        consumed = 0
        for batch in loader:
            if consumed >= training_steps or self.step_count >= target_step_count:
                break
            batch = _to_device(batch, primary_device)
            self._update_hook_snapshot(batch=batch, loss_out=None)
            if not training_started:
                self._run_hooks(TrainingStage.BEFORE_TRAINING, batch)
                training_started = True
            if self.epoch_step_count == 0:
                self._run_hooks(TrainingStage.BEFORE_EPOCH, batch)
            self._train_batch_with_optimizers(batch, flat_opts, flat_scheds)
            self._validation_checkpoint(TrainingStage.AFTER_OPTIMIZER_STEP)
            consumed += 1
        if consumed == 0:
            raise ValueError(
                "The segment's mixed loader produced no batches before the "
                "target step count was reached; ensure reference_dataset and "
                "the replay buffer together hold at least one batch of "
                f"batch_size={config.batch_size!r} samples."
            )
        self.epoch_count += 1
        self.epoch_step_count = 0
        self._refresh_hook_counters()
        self._run_hooks(TrainingStage.AFTER_EPOCH, self._last_batch)
        self._validation_checkpoint(TrainingStage.AFTER_EPOCH)
        return training_started

    def _generate_segment(
        self,
        config: OnPolicyConfig,
        state: Batch,
        label_hook: TeacherLabelHook,
        lifecycle: _RelaxationLifecycle | None,
        buffer: ReplayBuffer,
        target_step_count: int,
    ) -> Batch | None:
        """Propagate one segment, store what it produced, and refill the batch.

        Parameters
        ----------
        config : OnPolicyConfig
            Segment-loop configuration.
        state : Batch
            Batch this segment propagates from.
        label_hook : TeacherLabelHook
            Hook labeling and capturing the frames along the path.
        lifecycle : _RelaxationLifecycle | None
            Convergence machinery, or ``None`` when none is managed.
        buffer : ReplayBuffer
            Buffer the segment's frames are stored in.
        target_step_count : int
            Training step the run ends at, named by the exhaustion warning.

        Returns
        -------
        Batch | None
            The batch the next segment propagates from, or ``None`` once every
            trajectory has finished and no initial structure is left to start a
            fresh one from. That case warns once.
        """
        # Sized per segment because a refill changes the trajectory count. The
        # converged route keeps its own host sink, one frame per graph.
        label_hook.sink = _segment_sink(config, state.num_graphs)
        if lifecycle is not None:
            lifecycle.capture.sink = HostMemory(capacity=state.num_graphs)
        state = config.dynamics.run(state, n_steps=config.generation_steps)
        self._capture_segment(config, state, label_hook, buffer)
        if lifecycle is None:
            return state
        self._capture_converged(config, lifecycle, buffer)
        refilled = self._backfill_segment(config, lifecycle, state)
        if refilled is None:
            self._warn_generation_exhausted(config, target_step_count, state)
        return refilled

    def _capture_converged(
        self,
        config: OnPolicyConfig,
        lifecycle: _RelaxationLifecycle,
        buffer: ReplayBuffer,
    ) -> None:
        """Label the structures that converged this segment and store them.

        Converged frames are captured without teacher labels, at the step each
        structure reached its minimum. Here the teacher labels all of the
        segment's graduates in one pass, under the same guards the path route
        uses. The frames are then stripped to the replay-frame contract, so
        they match the schema the path frames froze the buffer with. Finally
        they are moved to the buffer's own device.
        """
        sink = lifecycle.capture.sink
        if len(sink) == 0:
            return
        frames = _to_device(sink.drain(), self.devices[0])
        _score_and_attach(config.teacher_scorer, frames)
        buffer.extend(_strip_replay_frame(frames).to(buffer.device or "cpu"))

    def _backfill_segment(
        self,
        config: OnPolicyConfig,
        lifecycle: _RelaxationLifecycle,
        state: Batch,
    ) -> Batch | None:
        """Graduate the finished structures and backfill fresh ones in their place.

        A trajectory finishes in one of two ways. It converges and the
        criterion freezes it, or it diverges and the lifecycle freezes it on
        the step the divergence predicate flagged it. This method counts the
        diverged ones from the divergence hook's record and warns about them.
        It then draws initial structures
        to fill the room the graduates freed: at most as many structures as
        graduated, within the atoms they held. The draw is also bounded by
        their edges, but only when the source declared ``max_edges``, because
        a dataset's stored edge count is not the neighbor list a propagator
        rebuilds. The propagator's per-structure state follows the membership
        change. The run restamps its bookkeeping over the appended rows and
        keeps only the ``system_id`` the source assigned. A structure stored
        with the ``status`` it once graduated on therefore still moves.

        Returns
        -------
        Batch | None
            The refilled batch, or ``None`` once nothing is left to propagate.
        """
        dynamics = config.dynamics
        status = state["status"].view(-1)[: state.num_graphs]
        graduated = status >= dynamics.exit_status
        if not bool(graduated.any()):
            return state
        flagged_graphs = lifecycle.divergence.diverged
        diverged = (
            0 if flagged_graphs is None else int((graduated & flagged_graphs).sum())
        )
        if diverged:
            predicate = lifecycle.divergence.divergence
            flagged = (
                "their positions or forces stopped being finite"
                if predicate is nonfinite_divergence
                else f"the divergence predicate {predicate!r} flagged them"
            )
            warnings.warn(
                f"{diverged} of {state.num_graphs} generated trajectories "
                f"diverged: {flagged}. The lifecycle froze them on that step "
                "without storing that frame. They are now dropped from the "
                "batch and replaced from the initial structures. A diverging "
                "student is extrapolating; shorten the propagator's step, or "
                "register a MaxForceClampHook on it.",
                UserWarning,
                stacklevel=2,
            )
        structures = lifecycle.structures
        edges_per_graph = state.num_edges_per_graph
        fresh = structures.draw(
            limit=int(graduated.sum()),
            fits=WithinBudget(
                atoms=int(state.num_nodes_per_graph[graduated].sum()),
                edges=int(edges_per_graph[graduated].sum())
                if getattr(structures, "max_edges", None) is not None
                and edges_per_graph.numel() > 0
                else None,
            ),
            on_miss="skip",
        )
        lifecycle.divergence.reset()
        survivors = torch.where(~graduated)[0]
        refilled = state.index_select(survivors) if survivors.numel() > 0 else None
        if fresh:
            appended = type(state).from_data_list(fresh, device=state.device)
            if refilled is None:
                refilled = appended
            else:
                refilled.append(appended)
        # As refill_check does: the state rows follow the membership change,
        # and stale converged indices must not reach the next step's context.
        dynamics._sync_state_to_batch(
            survivors, len(fresh), state if refilled is None else refilled
        )
        dynamics._last_converged = None
        if refilled is None:
            return None
        # Appending keeps only the keys both sides hold, so every bookkeeping
        # column except system_id is rebuilt, keeping the survivors' values.
        kept = survivors.numel()
        for key, default_fn in dynamics._bookkeeping_keys.items():
            if key == "system_id":
                continue
            column = default_fn(refilled.num_graphs, refilled.device)
            if kept > 0 and key in state:
                carried = state[key][survivors]
                column[:kept] = carried.unsqueeze(-1) if carried.dim() == 1 else carried
            refilled[key] = column
        return refilled

    def _warn_generation_exhausted(
        self, config: OnPolicyConfig, target_step_count: int, finished: Batch
    ) -> None:
        """Announce that the run trains on what it has already generated.

        *finished* is the batch whose last trajectories graduated. Every one of
        them did, so its atoms and edges are the room the final backfill had
        to fill.
        """
        structures = config.initial_structures
        if structures.exhausted:
            remedy = (
                "Pass initial_structures=InitialStructures(dataset, recycle=True) "
                "to keep generating from the front of the rows this rank owns, or "
                "start from more structures — an unbudgeted source is propagated "
                "whole, so more of them lengthen the run by widening the initial "
                "batch rather than by backfilling it."
            )
        else:
            freed = f"{int(finished.num_nodes_per_graph.sum())!r} atoms"
            if getattr(structures, "max_edges", None) is not None:
                freed += f" and {int(finished.num_edges_per_graph.sum())!r} edges"
            remedy = (
                f"The source still holds rows, but none of them fits the {freed} "
                "the finished trajectories freed. Widen the source's budget, "
                "which sets the initial batch and with it the room a graduation "
                "frees."
            )
        warnings.warn(
            "Every generated trajectory has finished and the initial structures "
            "have nothing left to start a fresh one from, so generation stopped "
            f"after {config.dynamics.step_count} propagator steps with "
            f"{len(self._replay_buffer)} frames in the replay buffer; the "
            f"remaining {target_step_count - self.step_count} training steps "
            f"draw from that buffer. {remedy}",
            UserWarning,
            stacklevel=2,
        )

    def _resolve_replay_device(
        self, config: OnPolicyConfig
    ) -> torch.device | str | None:
        """Return the device the segment loop stages generated frames on.

        Frames reach the buffer from a host-memory sink. An unset
        ``replay_device`` therefore means the reference dataset's device,
        because the two sources are collated before the strategy moves the
        batch. That device is measured from a batch when no declaration
        settles it. A run with no reference dataset leaves the frames in host
        memory.
        """
        if config.replay_device is not None:
            return config.replay_device
        if self.reference_dataset is None:
            return None
        return dataset_device(self.reference_dataset)

    def _capture_segment(
        self,
        config: OnPolicyConfig,
        state: Batch,
        label_hook: TeacherLabelHook,
        buffer: ReplayBuffer,
    ) -> None:
        """Label the frame the segment ended on and drain the sink into *buffer*.

        The cadence rarely lands on a segment's last step, yet that frame is the
        most on-policy one the segment produced. The hook's forced entry point
        is therefore called for the step just finished. A forced label is never
        skipped. Labeling is idempotent per step, so the forced call costs
        nothing when the cadence did land on that step.
        """
        label_hook._label_frame(
            state, max(config.dynamics.step_count - 1, 0), forced=True
        )
        if label_hook.sink is not None and len(label_hook.sink) > 0:
            buffer.extend(label_hook.sink.drain())

    def to_spec_dict(self) -> dict[str, Any]:
        """Serialize declarative distillation settings to a JSON-ready dict.

        The bundle names its own class under ``strategy_cls``, which
        :meth:`from_spec_dict` builds. ``on_policy`` and ``reference_dataset``
        hold a live propagator, scorer, and datasets that no spec can describe.
        They are therefore omitted, and a rebuilt strategy runs offline.

        Returns
        -------
        dict[str, Any]
            JSON-ready bundle suitable for :func:`json.dumps`.

        Warns
        -----
        UserWarning
            If ``on_policy`` is set, because the spec cannot carry it.
        """
        spec = super().to_spec_dict()
        spec["strategy_cls"] = f"{type(self).__module__}.{type(self).__qualname__}"
        spec["teacher_signals"] = (
            None if self.teacher_signals is None else sorted(self.teacher_signals)
        )
        spec["label_missing"] = self.label_missing
        spec["label_dtype"] = (
            None if self.label_dtype is None else str(self.label_dtype)
        )
        if self.on_policy is not None:
            warnings.warn(
                "on_policy and reference_dataset hold live runtime objects and "
                "are omitted from the spec, so a strategy rebuilt from it runs "
                "offline over the dataloader passed to run(). Re-supply them at "
                "construction to keep generating on-policy.",
                UserWarning,
                stacklevel=2,
            )
        return spec

    @classmethod
    def from_spec_dict(
        cls,
        spec: Mapping[str, Any],
        *,
        models: strategy_validation.ModelInput | None = None,
        hooks: Sequence[Any] | None = None,
        training_fn: Any = None,
        validation_config: ValidationConfig | None = None,
    ) -> DistillationStrategy:
        """Rebuild a :class:`DistillationStrategy` from ``to_spec_dict`` output.

        A ``strategy_cls`` naming a subclass dispatches to that class's own
        ``from_spec_dict`` with the spec and every runtime override, so the
        strategy a spec names is the one that runs; a subclass adding a runtime
        keyword must widen this call with it. An optional keyword is forwarded
        only when it is set, so a subclass overriding ``from_spec_dict`` without
        it still rebuilds from a plain spec.

        Parameters
        ----------
        spec : Mapping[str, Any]
            A dict produced by :meth:`to_spec_dict`, optionally after a JSON
            round-trip.
        models : BaseModelMixin | dict[str, BaseModelMixin] | None, optional
            Runtime model override(s). Distillation models are not serialized
            in full, so the student and teacher are normally re-supplied here.
        hooks : Sequence[Any] | None, optional
            Runtime hooks; defaults to an empty list.
        training_fn : Any, optional
            Runtime callable or dotted-path override.
        validation_config : ValidationConfig | None, optional
            Runtime validation configuration. Specs exclude it because it
            carries a live loader, so a validation-only ``teacher_*`` target is
            resolved by passing the config here rather than assigning it
            afterwards, which re-runs no validator.

        Returns
        -------
        DistillationStrategy
            A freshly validated strategy of the class *spec* names, ready to
            :meth:`run`.

        Raises
        ------
        ValueError
            If *spec* is missing a required key, if its ``strategy_cls`` entry
            is not a dotted class path string, or if that path resolves to a
            class that is not a :class:`DistillationStrategy` subclass.
        TypeError
            If a runtime keyword is supplied but the subclass *spec* names
            overrides ``from_spec_dict`` without accepting it.
        """
        required = ("optimizer_configs", "devices", "loss_fn_spec")
        missing = [key for key in required if key not in spec]
        if missing:
            raise ValueError(
                f"from_spec_dict: spec is missing required key(s) {missing}. "
                f"Expected keys: {list(required)}."
            )
        raw_strategy_cls = spec.get("strategy_cls")
        if raw_strategy_cls is not None:
            if not isinstance(raw_strategy_cls, str):
                raise ValueError(
                    "from_spec_dict: 'strategy_cls' must be a dotted class path "
                    f"string; got {type(raw_strategy_cls).__name__}."
                )
            imported = _import_cls(raw_strategy_cls)
            if not issubclass(imported, cls):
                raise ValueError(
                    f"from_spec_dict: {raw_strategy_cls!r} must resolve to a "
                    f"{cls.__name__} subclass."
                )
            if imported is not cls:
                return imported.from_spec_dict(
                    spec,
                    models=models,
                    hooks=hooks,
                    training_fn=training_fn,
                    **_set_rebuild_overrides(
                        imported, {"validation_config": validation_config}
                    ),
                )
        model_input = strategy_spec._models_from_spec_and_overrides(
            spec.get("model_specs", {}),
            models,
            single_model_input=strategy_spec._single_model_input_from_spec(
                spec.get("single_model_input")
            ),
        )
        return cls(
            models=model_input,
            optimizer_configs=strategy_spec._optimizer_configs_from_spec(
                spec["optimizer_configs"]
            ),
            num_epochs=spec.get("num_epochs"),
            num_steps=spec.get("num_steps"),
            epoch_step_modifier=spec.get("epoch_step_modifier", 1.0),
            hooks=list(hooks) if hooks is not None else [],
            training_fn=strategy_spec._training_fn_from_spec(spec, training_fn),
            loss_fn=strategy_spec._loss_fn_from_spec(spec["loss_fn_spec"]),
            devices=strategy_spec._devices_from_spec(spec["devices"]),
            validation_config=validation_config,
            teacher_signals=spec.get("teacher_signals"),
            label_missing=spec.get("label_missing", True),
            label_dtype=(
                None
                if spec.get("label_dtype") is None
                else _dtype_deserialize(spec["label_dtype"])
            ),
        )
