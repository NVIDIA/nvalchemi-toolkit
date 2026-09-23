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
from collections.abc import Iterable, Iterator, Mapping, Sequence, Sized
from contextlib import contextmanager, nullcontext
from typing import TYPE_CHECKING, Annotated, Any

import torch
from pydantic import Field, PrivateAttr, model_validator
from torch import distributed as dist

from nvalchemi._serialization import _dtype_deserialize, _import_cls
from nvalchemi._typing import ModelOutputs
from nvalchemi.data.datapipes.dataset import BatchDatasetProtocol
from nvalchemi.data.level_storage import resolve_device
from nvalchemi.distributed import collective_device
from nvalchemi.dynamics.base import BaseDynamics, ConvergenceHook, DynamicsStage
from nvalchemi.dynamics.sinks import HostMemory
from nvalchemi.hooks import DynamicsContext
from nvalchemi.models.base import BaseModelMixin
from nvalchemi.training import TrainingStage
from nvalchemi.training import _spec_utils as strategy_spec
from nvalchemi.training import _strategy_validation as strategy_validation
from nvalchemi.training.distillation._attach import _attach_teacher_labels
from nvalchemi.training.distillation.config import OnPolicyConfig, ResizableSink
from nvalchemi.training.distillation.hooks import (
    TeacherLabelHook,
    _checked_divergence,
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
    _emitted_device,
    _frame_schema,
    _same_device,
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
from nvalchemi.training.distillation.seeding import (
    InitialStructures,
    InitialStructuresSource,
    WithinBudget,
    _check_structure_status,
    _propagator_tree,
)
from nvalchemi.training.distributed import all_reduce, get_rank, get_world_size
from nvalchemi.training.losses.composition import loss_target_keys
from nvalchemi.training.runtime import (
    eval_configured_models,
    evaluating,
    freeze_unconfigured_models,
    move_to_devices,
    train_configured_models,
    unwrap_model,
)
from nvalchemi.training.strategy import TrainingStrategy

if TYPE_CHECKING:
    from torch.optim.lr_scheduler import LRScheduler

    from nvalchemi.data.batch import Batch
    from nvalchemi.dynamics.sinks import DataSink
    from nvalchemi.hooks import TrainContext
    from nvalchemi.training import ValidationConfig
    from nvalchemi.training.distillation.hooks import _DivergencePredicate
    from nvalchemi.training.losses.composition import (
        BaseLossFunction,
        ComposedLossFunction,
    )

__all__ = ["DistillationStrategy", "default_distillation_fn"]

_REQUIRED_MODELS = frozenset({"student", "teacher"})
"""Model names every distillation strategy must be given."""

_PREDICTION_KEY_PREFIX = "predicted_"
"""Prefix the stock training function publishes every student output under."""

_PROPAGATOR_SEED_ATTRS = ("random_seed", "_random_seed")
"""Attribute names a propagator may hold an integer RNG seed under."""


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
    divergence: _DivergencePredicate


def _competing_migrators(
    dynamics: BaseDynamics, criterion: ConvergenceHook
) -> list[ConvergenceHook]:
    """Return the status migrators already on *dynamics* that are not *criterion*.

    Every place a propagator can hold one is searched: its registered hooks,
    which on a :class:`~nvalchemi.dynamics.FusedStage` are the ones registered
    at the fused level, its ``convergence_hook``, which the lifecycle is about
    to replace, and the same on every sub-stage, which is where the stage puts
    the migrators it builds itself — one per non-last sub-stage, and one on the
    last whenever it declares a ``convergence_hook``.

    Parameters
    ----------
    dynamics : BaseDynamics
        Propagator the lifecycle is being installed on.
    criterion : ConvergenceHook
        The lifecycle's own criterion, which is not a competitor.

    Returns
    -------
    list[ConvergenceHook]
        The competing criteria, in the order they were found.
    """
    return [
        hook
        for propagator in _propagator_tree(dynamics)
        for hook in (*propagator.hooks, propagator.convergence_hook)
        if isinstance(hook, ConvergenceHook)
        and hook is not criterion
        and hook.source_status is not None
        and hook.target_status is not None
    ]


@contextmanager
def _relaxation_lifecycle(
    config: OnPolicyConfig, state: Batch
) -> Iterator[_RelaxationLifecycle | None]:
    """Install the convergence machinery of a relaxation run on the propagator.

    The config's :attr:`~OnPolicyConfig.convergence_criterion` goes on the
    propagator twice: as a registered ``AFTER_STEP`` hook it migrates the
    status of converged graphs, which freezes them and is what the capture hook
    behind it and the segment boundary read; as the propagator's
    ``convergence_hook`` it is the detector ending a chunk early once every
    graph has converged. A detector the propagator was built with is restored
    on the way out. The criterion has to be the sole migrator — a looser one
    would graduate a structure before this one accepts it, out of both capture
    routes — and the lifecycle the sole refill, since a mid-segment refill
    compacts the survivors under the capture hook's positional bookkeeping. A
    divergence hook behind the criterion freezes a graph the config's
    :attr:`~OnPolicyConfig.divergence` predicate flags — by default one whose
    state stopped being finite — at ``exit_status``, uncaptured, so the
    boundary retires it like a converged one.

    Parameters
    ----------
    config : OnPolicyConfig
        Segment-loop configuration, holding the criterion.
    state : Batch
        Initial batch, already carrying the bookkeeping
        :meth:`~nvalchemi.training.distillation.InitialStructures.initial_batch`
        stamped on it.

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
    competing = _competing_migrators(dynamics, criterion)
    if competing:
        migrations = [(hook.source_status, hook.target_status) for hook in competing]
        raise ValueError(
            "The relaxation lifecycle owns graduation for this run, so the "
            "propagator must carry no other status-migrating ConvergenceHook; "
            f"got {migrations!r} beside the configured "
            f"({criterion.source_status!r}, "
            f"{criterion.target_status!r}). A second migrator graduates "
            "structures at its own threshold, and one that graduates them "
            "before the configured criterion accepts them stores them by "
            "neither capture route. Remove it, or drop fmax and let the "
            "propagator manage its own lifecycle. On a FusedStage the migrator "
            "is one the stage built for a sub-stage: every non-last sub-stage "
            "carries one, and the last one does whenever it was given a "
            "convergence_hook, so only a single sub-stage without its own "
            "criterion is free of them."
        )
    if dynamics.sampler is not None:
        raise ValueError(
            "The relaxation lifecycle owns the refill as well as graduation, "
            "so the propagator must carry no sampler of its own; got "
            f"{type(dynamics.sampler).__name__!r}. A propagator that refills "
            "inside run compacts the survivors to the front of the batch mid "
            "segment, which leaves the capture hook's positional bookkeeping "
            "pointing at the wrong structures and drops the minima it was "
            "meant to store. Give OnPolicyConfig.initial_structures the same "
            "budget, which backfills from the same dataset at the segment "
            "boundary, and leave the propagator's own unset."
        )
    _check_structure_status(state, criterion)
    predicate = config.divergence or nonfinite_divergence
    capture = _ConvergedFrameHook(
        sink=HostMemory(capacity=state.num_graphs), divergence=predicate
    )
    divergence = _DivergenceHook(predicate)
    detector = dynamics.convergence_hook
    # Registered ahead of the capture and labeling hooks, so a graph that
    # converges or diverges on this step is graduated before either of them
    # reads its status and neither route stores it twice, or at all.
    dynamics.register_hook(criterion)
    dynamics.register_hook(divergence)
    dynamics.register_hook(capture)
    dynamics.convergence_hook = criterion
    try:
        yield _RelaxationLifecycle(
            capture=capture, structures=config.initial_structures, divergence=predicate
        )
    finally:
        dynamics.convergence_hook = detector
        dynamics.hooks.remove(criterion)
        dynamics.hooks.remove(divergence)
        dynamics.hooks.remove(capture)


def _movable_seed(node: BaseDynamics) -> tuple[BaseDynamics, str, int] | None:
    """Return the first integer seed of *node* a rank offset can write back.

    Each candidate is probed by writing back the value just read, so a
    getter-only ``random_seed`` property falls through to the writable name
    behind it instead of raising where the offsets are applied.
    """
    for name in _PROPAGATOR_SEED_ATTRS:
        seed = getattr(node, name, None)
        if not isinstance(seed, int):
            continue
        try:
            setattr(node, name, seed)
        except AttributeError:
            continue
        return node, name, seed
    return None


def _propagator_seed_plan(
    dynamics: BaseDynamics,
) -> tuple[list[tuple[BaseDynamics, str, int]], list[BaseDynamics]]:
    """Return the seeds of a composition a rank offset moves, and what it misses.

    Accounting is per node, so one seeded sub-stage does not pass for the
    stages beside it. Only a node holding a :class:`torch.Generator` and no
    integer seed is reported as missed; one exposing neither is passed over,
    since nothing tells hidden randomness from a deterministic stage.
    """
    seeds: list[tuple[BaseDynamics, str, int]] = []
    unmoved: list[BaseDynamics] = []
    for node in _propagator_tree(dynamics):
        movable = _movable_seed(node)
        if movable is not None:
            seeds.append(movable)
        elif any(
            isinstance(value, torch.Generator)
            for value in getattr(node, "__dict__", {}).values()
        ):
            unmoved.append(node)
    return seeds, unmoved


@contextmanager
def _rank_local_propagator_seed(dynamics: BaseDynamics, offset: int) -> Iterator[None]:
    """Temporarily move a stochastic propagator's RNG onto this rank's own stream.

    A counter-based thermostat draws its noise from ``seed + step_count`` and
    the atom index, so ranks stepping in lockstep would apply the same kicks to
    their different structures, and identical kicks to replicas of one
    geometry. Every propagator in the composition is moved, since a
    ``FIRE(...) + NVTLangevin(...)`` root exposes no seed of its own.

    Parameters
    ----------
    dynamics : BaseDynamics
        Propagator whose seed is offset along with every sub-stage's, under the
        names in ``_PROPAGATOR_SEED_ATTRS``, and restored on the way out.
        Randomness held anywhere else is left alone here and reported by
        :meth:`DistillationStrategy._warn_shared_propagator_streams`.
    offset : int
        Amount added to every seed found. Zero leaves the propagator untouched.

    Yields
    ------
    None
        Control while the propagator draws from this rank's stream.
    """
    if offset == 0:
        yield
        return
    seeds, _ = _propagator_seed_plan(dynamics)
    for node, name, seed in seeds:
        setattr(node, name, seed + offset)
    try:
        yield
    finally:
        for node, name, seed in seeds:
            setattr(node, name, seed)


def _structure_count(structures: InitialStructuresSource) -> int | None:
    """Return the rows *structures* holds in all, or ``None`` when it does not say.

    :class:`~nvalchemi.training.distillation.InitialStructures` publishes its
    dataset; another source is counted through ``len()`` when it has one, and
    a streaming source with neither is left uncounted, so the world-size checks
    that need the count are skipped for it.
    """
    dataset = getattr(structures, "dataset", None)
    if dataset is not None:
        return len(dataset)
    return len(structures) if isinstance(structures, Sized) else None


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
    """Return the sink one segment's labeled frames are staged in.

    A segment captures at most one frame per trajectory per labeled step, the
    forced last frame included, so the sink has to hold
    ``(generation_steps + 1) * num_graphs`` frames. Without a configured
    ``capture_sink`` a host-memory sink of that capacity is built; a configured
    one is kept and, when it is too small, resized through ``resize(capacity)``
    if it satisfies :class:`~nvalchemi.training.distillation.ResizableSink`.

    Parameters
    ----------
    config : OnPolicyConfig
        Segment-loop configuration, holding the optional ``capture_sink``.
    num_graphs : int
        Trajectories the segment propagates.

    Returns
    -------
    DataSink
        Empty sink of at least the segment's capacity.

    Raises
    ------
    ValueError
        If the configured sink still holds frames, which the segment boundary
        would drain into the replay buffer as generated ones, or if it is too
        small and is no ``ResizableSink``.
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
    """Return *batch* on *device*, overlapping the copy only into device memory.

    A copy into host memory returns before the transfer lands, and the moved
    batch's ``segment_lengths`` and ``batch_ptr`` are read on the host at once,
    so that direction blocks.
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

    Setting ``on_policy`` switches :meth:`run` to the segment loop instead:
    the student's own propagator generates frames, the teacher labels them,
    they accumulate in a replay buffer, and each segment trains on a mixture of
    that buffer and ``reference_dataset`` at the configured ``replay_ratio``.
    Because the propagator holds the very module the optimizer updates, every
    segment generates from a fresher policy than the last — which is what makes
    the data on-policy, and why the propagator's model is checked for object
    identity with ``models["student"]`` at construction. A relaxation
    propagator adds ``OnPolicyConfig.fmax``: converged structures are
    stored once, graduate out of the batch at the segment boundary, and are
    replaced by fresh initial structures, so the buffer keeps filling with
    structures that are still moving.

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
        on-policy mode, additionally if the run is sized in epochs, if the
        propagator holds neither the student nor a model composing it, if
        ``replay_ratio`` and ``reference_dataset`` disagree (a ratio below
        ``1`` needs one, a ratio of ``1`` refuses one), if ``reference_dataset`` emits on an accelerator the run
        does not train on, if ``replay_device`` or
        the reference dataset's fields cannot be mixed with generated frames,
        or if the propagator's scorer and the reference dataset do not carry
        the same teacher fields.

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
    Labeling runs with autocast disabled, and labels are cast to ``label_dtype``
    when one is given. By default it is inferred as the student's first
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

    ``on_policy`` and ``reference_dataset`` hold live runtime objects no spec
    can describe, so :meth:`to_spec_dict` omits them and warns; a strategy
    rebuilt from such a spec runs offline until they are re-supplied.
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
    _warned_unwrapped_student: bool = PrivateAttr(default=False)
    _replay_buffer: ReplayBuffer | None = PrivateAttr(default=None)
    _validated_step: int | None = PrivateAttr(default=None)

    @property
    def replay_buffer(self) -> ReplayBuffer | None:
        """Frames generated so far, or ``None`` before an on-policy run starts.

        One buffer serves every :meth:`run` call on a strategy, so a run
        continued with a raised ``num_steps`` keeps training on everything
        generated so far instead of throwing it away and regenerating it. The
        trajectory is still reseeded per call.
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

    @property
    def structure_shard(self) -> tuple[int, ...]:
        """Rows of the initial structures this rank propagates from.

        The rows are read off the source once :meth:`run` has installed this
        rank's shard on it, and dealt here from the launcher's world before
        that, so the property answers the same question either side of a run.

        Returns
        -------
        tuple[int, ...]
            Indices into ``on_policy.initial_structures.dataset``, in dataset
            order. Empty for an offline strategy, and for a source that
            publishes neither its rows nor a count.

        See Also
        --------
        nvalchemi.training.distillation.InitialStructures.shard :
            The deal itself, and the shard-local cursor it opens.
        """
        if self.on_policy is None:
            return ()
        structures = self.on_policy.initial_structures
        rank = get_rank(self.distributed_manager)
        world_size = get_world_size(self.distributed_manager)
        installed = structures.state_dict()
        if (installed.get("rank"), installed.get("world_size")) == (rank, world_size):
            return tuple(getattr(structures, "rows", ()))
        total = _structure_count(structures)
        return () if total is None else tuple(range(rank, total, world_size))

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
                "replay buffer, so the reference dataset is policed for schema and "
                "device and then never sampled; got replay_ratio=1.0 alongside a "
                f"{type(self.reference_dataset).__name__} reference_dataset. "
                "Drop reference_dataset, or lower replay_ratio to mix it in."
            )
        if self.reference_dataset is not None and len(self.reference_dataset) == 0:
            raise ValueError(
                "A replay_ratio below 1 draws part of every batch from "
                "reference_dataset, so it has to hold at least one sample; got an "
                f"empty {type(self.reference_dataset).__name__}. Pass the labeled "
                "reference set the mixture draws from, or set replay_ratio=1 to "
                "train on generated frames alone."
            )
        # One probe answers the device and the schema questions alike.
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
            One batch already drawn from ``reference_dataset``, whose device is
            what a composition or a device-less store is measured by. ``None``
            when there is no reference dataset to measure.
        """
        if self.reference_dataset is None:
            return
        reference_device = _emitted_device(self.reference_dataset, probe)
        primary = self.devices[0]
        if reference_device.type == "cpu" or _same_device(reference_device, primary):
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
            One batch already drawn from ``reference_dataset``, whose device is
            what a composition or a device-less store is measured by. ``None``
            when there is no reference dataset to measure.
        """
        if self.reference_dataset is None or self.on_policy.replay_device is None:
            return
        reference_device = _emitted_device(self.reference_dataset, probe)
        replay_device = torch.device(self.on_policy.replay_device)
        if _same_device(reference_device, replay_device):
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
        """Reject a reference dataset holding fields no generated frame can ever carry.

        The full schema comparison runs inside the first segment's
        :func:`~nvalchemi.training.distillation.build_mixed_loader`, after a whole
        generation phase has been paid for; the part that depends on nothing the
        run produces — the predictions, neighbor tensors, and bookkeeping the
        labeling hook strips from every frame — is checked here. A store
        :func:`~nvalchemi.training.distillation.label_dataset` wrote over a
        reference set keeps that set's ``energy`` and ``forces``, which is what
        this catches.
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
        """Check the propagator's teacher fields against the reference set and the loss.

        A scorer that declares neither ``label_fields`` nor a set of built-in
        signals writes fields nothing can know before it has scored a batch, so
        both checks below are skipped with a warning rather than run against an
        empty set — which would reject a custom scorer that in fact produces
        exactly what the reference dataset carries.
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

        Compared as fields rather than as signal names, so a custom scorer
        declaring the fields the loss reads under signal names of its own is
        not accused of leaving them out.
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
        teacher runs with autocast disabled, so the labels match what
        :func:`~nvalchemi.training.distillation.label_dataset` persisted
        wherever the store returns the label dtype.

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
        with torch.autocast(device_type=batch.device.type, enabled=False):
            labels = self.teacher_scorer.label(batch)
        _attach_teacher_labels(batch, labels)
        return True

    def _missing_teacher_fields(self, batch: Batch) -> list[str]:
        """Return the resolved teacher fields *batch* does not carry."""
        return [field for field in self._teacher_fields if field not in batch]

    def run(self, dataloader: Iterable[Batch] | None = None) -> None:
        """Execute the offline training loop or the on-policy segment loop.

        Without ``on_policy`` this is
        :meth:`~nvalchemi.training.TrainingStrategy.run` over *dataloader*. With
        it, the strategy owns the loop and repeats three phases until ``num_steps``
        optimizer steps have run: *generate* — the propagator advances the live
        state batch by ``generation_steps``, seeded on the first segment from
        ``initial_structures``; *label and capture* — a
        :class:`~nvalchemi.training.distillation.TeacherLabelHook` registered on
        the propagator scores every ``label_frequency`` steps and the segment's
        last frame, mirroring each labeled frame into the capture sink — host
        memory unless ``capture_sink`` names another — that is drained into the
        replay buffer; *train* — a freshly built mixed loader
        draws ``training_steps_per_segment`` batches at ``replay_ratio``, each
        through the ordinary per-batch stages.

        An ``OnPolicyConfig.fmax`` threshold adds a fourth phase between
        generation and training, for the relaxation propagators whose
        trajectories end: *graduate and backfill* — converged structures are
        stored once as the minimum they reached, then leave the batch and are
        replaced by fresh initial structures wherever the cursor still holds
        any. Generation stops when it runs dry and the last trajectory
        finishes, and the remaining steps train on the buffer already filled.

        Parameters
        ----------
        dataloader : Iterable[Batch] | None, optional
            Batches to train on in offline mode. Default ``None``, which is
            required in on-policy mode and rejected otherwise.

        Raises
        ------
        ValueError
            If *dataloader* is ``None`` in offline mode or supplied in on-policy
            mode, if a multi-rank launch holds fewer initial structures than
            ranks or leaves the student unsynchronized, if a segment's loader
            produces no batches, if the propagator already carries a
            status-migrating criterion or a sampler of its own, or if the
            configured criterion migrates off a status no initial structure
            carries.

        Warns
        -----
        UserWarning
            If a lifecycle-managed run runs out of trajectories and structures
            before reaching ``num_steps``, because the remaining steps then
            train on the frames already generated; once per segment boundary
            that retires trajectories whose state stopped being finite; if a
            multi-rank run cannot deal its initial structures out in equal
            shares; or if its propagator holds randomness the rank offsets
            cannot separate.

        Notes
        -----
        One segment is one epoch: ``AFTER_EPOCH`` and epoch-cadence validation
        fire at segment boundaries, step-cadence validation inside them, and the
        run closes with one terminal validation unless a cadence already validated
        at the final step. The segment is also the restart granularity: the
        propagator state is not checkpointed, a resumed run reseeds its
        trajectory, and a segment a checkpoint interrupted is counted as finished
        on the way in. A second call keeps the replay buffer the first filled and
        reseeds only the trajectory.

        The student generates in evaluation mode and trains in training mode; a
        propagator model that merely composes the student is held in evaluation
        mode for the whole loop, and every mode is restored on the way out.
        Generated frames are staged on ``reference_dataset``'s device unless
        ``replay_device`` names another. Chunking the built-in propagators
        across segments is exact, since ``run`` never resets ``step_count`` and
        the Langevin thermostat's generator is keyed on it; an early-exiting
        chunk is read from ``dynamics.step_count``, and a
        :class:`~nvalchemi.dynamics.FusedStage` pays its priming forward pass
        once per segment.

        Across ranks the loop is data-parallel and self-labeling: each rank
        propagates its own strided shard of ``initial_structures``, labels
        those frames with its own teacher replica, and fills its own replay
        buffer, while the reference dataset stays replicated and every rank
        draws from all of it under a rank-offset ``seed``. The same offset
        moves every integer seed the propagator and its sub-stages expose; a
        stage holding a :class:`torch.Generator` and no integer seed is named
        in a warning. The student's gradient all-reduce, installed by a
        :class:`~nvalchemi.training.hooks.DDPHook`, is the only cross-rank
        traffic, and every rank runs the same number of segments and batches,
        so the ranks reach it together. A launch whose student nothing wraps,
        or with fewer initial structures than ranks, is refused up front. See
        :ref:`training-distillation-api` for the mixture and schema contract
        and the scale-out runbook.

        A relaxation run is what that early exit exists for, and
        ``OnPolicyConfig.fmax`` turns it into a lifecycle: the criterion
        is registered ahead of the labeling hook and installed as the detector
        for the duration of the loop, a converged structure is captured once on
        the step its ``status`` reaches ``exit_status`` and left out of every
        later path capture, and at the boundary the initial structures are
        drawn for the room the graduates freed — a budgeted
        :class:`~nvalchemi.training.distillation.InitialStructures` from its
        remainder, an unbudgeted one only under ``recycle``, the batch narrowing
        otherwise. Once no trajectory is left and no structure remains to start
        one, the loop warns and trains on the buffer it has until ``num_steps``.
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

        The rank shard is installed on the initial structures here rather than at
        construction, because the world size is a launcher fact and because
        installing it rewinds the cursor: a second ``run()`` on one strategy
        keeps the replay buffer it filled and reseeds only the trajectory, so
        the source has to open at the front of its shard again. A propagator
        model that merely composes the student is moved whole to the generation
        device, since only the named models travel with the strategy and a
        correction the composition alone holds would otherwise stay where it
        was built.
        """
        training_started = False
        strategy_context = nullcontext(self) if self._context_depth > 0 else self
        with strategy_context:
            self._prepare_setup_hooks()
            self._validate_runtime_devices()
            self._validate_structure_shards(config)
            self._warn_unequal_structure_shards(config)
            self._warn_shared_propagator_streams(config)
            self.models = move_to_devices(self.models, self.devices)
            propagator_model = config.dynamics.model
            if propagator_model is not self.models["student"] and isinstance(
                propagator_model, torch.nn.Module
            ):
                propagator_model.to(self.devices[0])
            unsynchronized = self.models["student"]
            self._run_setup_hooks()
            self._validate_synchronized_student(config, unsynchronized)
            replay_device = self._resolve_replay_device(config)
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
                state = self._seed_initial_state(config, primary_device)
                if self._replay_buffer is None:
                    self._replay_buffer = ReplayBuffer(
                        capacity=config.replay_capacity,
                        eviction=config.replay_eviction,
                        admission=config.replay_admission,
                        device=replay_device,
                    )
                buffer = self._replay_buffer
                # The mode contexts want the module a DDPHook may have wrapped.
                student = unwrap_model(self.models["student"])
                propagator_model = config.dynamics.model
                held_propagator = (
                    evaluating(propagator_model)
                    if isinstance(propagator_model, torch.nn.Module)
                    and propagator_model is not student
                    else nullcontext()
                )
                with _relaxation_lifecycle(config, state) as lifecycle:
                    # Only a lifecycle stores a graduated graph elsewhere; a
                    # propagator managing its own keeps every frame here.
                    label_hook = TeacherLabelHook(
                        config.teacher_scorer,
                        frequency=config.label_frequency,
                        exit_status=None
                        if lifecycle is None
                        else config.dynamics.exit_status,
                    )
                    config.dynamics.register_hook(label_hook)
                    try:
                        # The teacher is frozen across both phases; the student
                        # sits in eval mode and is flipped to training mode by
                        # the inner context for the training phase only. A
                        # composition holding the student is no entry of models,
                        # so it is held on its own.
                        with (
                            freeze_unconfigured_models(
                                self.models, self.optimizer_configs
                            ),
                            eval_configured_models(self.models, self.optimizer_configs),
                            held_propagator,
                            _rank_local_propagator_seed(
                                config.dynamics, self._rank_seed_offset(config)
                            ),
                        ):
                            while self.step_count < target_step_count:
                                if state is not None:
                                    state = self._generate_segment(
                                        config, state, label_hook, lifecycle, buffer
                                    )
                                    if state is None:
                                        self._warn_generation_exhausted(
                                            config, target_step_count
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

    def _validate_structure_shards(self, config: OnPolicyConfig) -> None:
        """Reject initial structures a multi-rank generation phase cannot share out.

        The world size is read at run time, once a launcher has initialized the
        process group; an offline strategy the same script builds distributes
        freely.

        Parameters
        ----------
        config : OnPolicyConfig
            Configuration of the loop about to start.

        Raises
        ------
        ValueError
            If the initial structures hold fewer rows than there are ranks.
        """
        world_size = get_world_size(self.distributed_manager)
        if world_size == 1:
            return
        num_structures = _structure_count(config.initial_structures)
        if num_structures is None:
            return
        if num_structures < world_size:
            raise ValueError(
                "Every rank propagates its own share of the initial structures, "
                "so there has to be at least one for each; got "
                f"{num_structures!r} structures on {world_size!r} ranks. Start "
                "from more structures, or launch fewer ranks."
            )

    def _seed_initial_state(
        self, config: OnPolicyConfig, device: torch.device
    ) -> Batch:
        """Draw this rank's first batch, holding every rank in the world to one.

        :meth:`_validate_structure_shards` can only weigh a source that says how
        many rows it holds; one that deals its own is measured here instead, by
        what its ``shard()`` actually left this rank. The verdict is reduced
        across the world before any rank goes on to the first gradient
        collective, so a rank whose shard came up empty stops the run rather
        than failing alone while its peers block.

        Parameters
        ----------
        config : OnPolicyConfig
            Configuration of the loop about to start, already sharded.
        device : torch.device
            Device the run trains on, which the seeded batch is moved to.

        Returns
        -------
        Batch
            Initial batch of this rank's shard.

        Raises
        ------
        ValueError
            If any rank's shard seeded nothing.
        """
        world_size = get_world_size(self.distributed_manager)
        if world_size == 1:
            return _to_device(config.initial_structures.initial_batch(), device)
        failure: Exception | None = None
        try:
            state = _to_device(config.initial_structures.initial_batch(), device)
        except Exception as exc:
            state, failure = None, exc
        empty = all_reduce(
            torch.tensor(
                int(state is None or state.num_graphs == 0),
                device=collective_device(),
            ),
            self.distributed_manager,
            op=dist.ReduceOp.MAX,
        )
        if not bool(empty.item()):
            return state
        raise ValueError(
            "Every rank propagates its own shard of the initial structures, so "
            f"each one has to be dealt at least one; got a shard that seeded "
            f"nothing on {world_size!r} ranks. Start from more structures, deal "
            "them out evenly in a source that shards itself, or launch fewer "
            "ranks."
        ) from failure

    def _warn_unequal_structure_shards(self, config: OnPolicyConfig) -> None:
        """Report a structure set the world cannot deal out in equal shares.

        Every rank draws the same number of replay samples per batch from a
        buffer holding only its own trajectories, and DDP averages gradients
        evenly, so a frame on a shorter shard is drawn more often. The
        arithmetic is the world's, so every rank reaches the same verdict
        without a collective.

        Parameters
        ----------
        config : OnPolicyConfig
            Configuration of the loop about to start.

        Warns
        -----
        UserWarning
            If the initial structures do not divide evenly across the ranks.
        """
        world_size = get_world_size(self.distributed_manager)
        if world_size == 1:
            return
        num_structures = _structure_count(config.initial_structures)
        if num_structures is None:
            return
        smallest, remainder = divmod(num_structures, world_size)
        if remainder == 0:
            return
        warnings.warn(
            "The initial structures do not divide evenly across the world, so "
            f"the ranks propagate shards of different sizes: {num_structures!r} "
            f"structures on {world_size!r} ranks deals {smallest + 1!r} to "
            f"{remainder!r} of them and {smallest!r} to the rest. Every rank "
            "draws the same number of replay samples per batch from a buffer "
            "holding only its own trajectories, and the gradients are averaged "
            "rank by rank, so a frame generated on a shorter shard reaches the "
            f"optimizer with up to {(smallest + 1) / smallest:.2f}x the weight "
            "of one from a longer shard. Size the dataset as a whole multiple "
            f"of {world_size!r} to weight every generated frame alike.",
            UserWarning,
            stacklevel=2,
        )

    def _warn_shared_propagator_streams(self, config: OnPolicyConfig) -> None:
        """Report the propagator randomness the rank offsets cannot separate.

        Warns rather than raises, because a propagator deterministic in the
        stages the walk cannot reach is correct and nothing here can tell the
        two apart. The verdict is the world's rather than this rank's, so rank
        zero reports it too, before a segment is generated or a teacher pass
        paid for.

        Parameters
        ----------
        config : OnPolicyConfig
            Configuration of the loop about to start.

        Warns
        -----
        UserWarning
            If a stage holding a :class:`torch.Generator` exposes no integer
            seed for the offset to move, or if nothing in the composition
            exposes one at all.
        """
        if get_world_size(self.distributed_manager) == 1:
            return
        seeds, unmoved = _propagator_seed_plan(config.dynamics)
        if unmoved:
            warnings.warn(
                "Part of this run's propagator stays on the shared random "
                f"stream: {sorted({type(node).__name__ for node in unmoved})!r} "
                "draw from a torch.Generator and expose no integer seed under "
                f"{list(_PROPAGATOR_SEED_ATTRS)!r} for the per-rank offset to "
                "move, so every rank applies the same kicks in those stages, "
                "whatever the walk separated around them. Ranks seeded with "
                "replicas of one structure then generate identical frames for "
                "as long as such a stage owns the batch, and the teacher is "
                "billed once per copy. Seed those generators from the global "
                "rank yourself, or expose the seed as an integer attribute the "
                "loop can offset.",
                UserWarning,
                stacklevel=2,
            )
        elif not seeds:
            warnings.warn(
                "This run's propagator noise could not be moved onto per-rank "
                "streams: neither the propagator nor anything it composes "
                "exposes an integer seed under "
                f"{list(_PROPAGATOR_SEED_ATTRS)!r}; got a "
                f"{type(config.dynamics).__name__}. A deterministic "
                "propagator has no stream to separate and can ignore this; one "
                "keeping its randomness elsewhere has to be handed a "
                "rank-distinct seed by the caller, or every rank applies the "
                "same kicks to the structures it was dealt.",
                UserWarning,
                stacklevel=2,
            )

    def _validate_synchronized_student(
        self, config: OnPolicyConfig, unsynchronized: BaseModelMixin
    ) -> None:
        """Reject a multi-rank run whose student nothing keeps in step.

        Called after the ``SETUP`` stage, when a
        :class:`~nvalchemi.training.hooks.DDPHook` has replaced every
        optimizer-configured model with a wrapper publishing the module it
        owns. The check is that the stage put something else in the student's
        place that owns it, read the way
        :func:`~nvalchemi.training.runtime.unwrap_model` reads ownership, so a
        hand-rolled or FSDP wrapper clears it as a ``DDPHook`` does and the
        model the propagator happens to hold plays no part. Comparing against
        the module registered before the stage rather than only unwrapping
        what is there afterwards keeps a bare student that happens to hold a
        submodule named ``module`` from reading as a wrapped one. A wrapper
        working in place leaves nothing to read, so
        ``require_wrapped_student=False`` waives the check, once with a
        warning, and gradient synchronization becomes the caller's business.

        Parameters
        ----------
        config : OnPolicyConfig
            Configuration whose ``require_wrapped_student`` governs the check.
        unsynchronized : BaseModelMixin
            The module registered as ``models["student"]`` before the ``SETUP``
            stage ran.

        Raises
        ------
        ValueError
            If nothing has taken ownership of the student to synchronize its
            gradients.

        Warns
        -----
        UserWarning
            Once per strategy, when the check is waived on a multi-rank world.
        """
        world_size = get_world_size(self.distributed_manager)
        if world_size == 1:
            return
        if not config.require_wrapped_student:
            if not self._warned_unwrapped_student:
                self._warned_unwrapped_student = True
                warnings.warn(
                    "require_wrapped_student=False: the check that the SETUP stage "
                    "replaced models['student'] with a wrapper owning it is "
                    f"skipped on {world_size!r} ranks, so synchronizing the "
                    "student's gradients is the caller's responsibility; an "
                    "in-place wrapper (FSDP2 fully_shard, hook-based sync) has "
                    "to be installed, or every rank trains and generates from a "
                    "private student while only rank zero's is checkpointed.",
                    UserWarning,
                    stacklevel=2,
                )
            return
        student = self.models["student"]
        if student is unsynchronized or unwrap_model(student) is not unsynchronized:
            raise ValueError(
                "A multi-rank segment loop trains one student from every rank's "
                "own frames, so the gradients have to be synchronized: without "
                "that, each rank keeps a private student, generates from it, and "
                "the policies diverge segment by segment while only rank zero's "
                "is checkpointed. Got a models['student'] the SETUP stage left "
                "unreplaced, or replaced with something that does not own the "
                f"module the run was built around, on {world_size!r} ranks; add a "
                "DDPHook to "
                "hooks, which wraps every optimizer-configured model at setup "
                "and leaves the frozen teacher replicated and out of the "
                "all-reduce, or install a gradient-synchronizing wrapper of "
                "your own — the check is that the SETUP stage replaced "
                "models['student'] with something owning it, not that a DDPHook "
                "was what did so."
            )

    def _close_interrupted_segment(self) -> None:
        """Count a segment a restored run stopped part-way through as finished.

        Each segment builds its own loader, so the batches an interrupted segment
        had left are gone with it and the trajectory is reseeded anyway; closing
        it keeps ``BEFORE_EPOCH`` firing for the resumed segment,
        ``epoch_step_count`` inside the segment budget, and the mixture sampler
        past the epoch index the interrupted segment drew with. The parent's
        :meth:`_prepare_epoch_step_count` reconciles against a fixed number of
        batches per epoch, which a run graduating from offline epochs of another
        size does not have.
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
            seed=config.seed + self._rank_seed_offset(config),
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

        Returns
        -------
        Batch | None
            The batch the next segment propagates from, or ``None`` once every
            trajectory has finished and no initial structure is left to start a
            fresh one from.
        """
        # Sized per segment because a refill changes the trajectory count; the
        # converged route keeps a host sink of its own, one frame per graph.
        label_hook.sink = _segment_sink(config, state.num_graphs)
        if lifecycle is not None:
            lifecycle.capture.sink = HostMemory(capacity=state.num_graphs)
        state = config.dynamics.run(state, n_steps=config.generation_steps)
        if lifecycle is not None:
            self._capture_budget_graduates(config, state, label_hook, lifecycle)
        self._capture_segment(config, state, label_hook, buffer)
        if lifecycle is None:
            return state
        self._capture_converged(config, lifecycle, buffer)
        return self._backfill_segment(config, lifecycle, state)

    def _capture_budget_graduates(
        self,
        config: OnPolicyConfig,
        state: Batch,
        label_hook: TeacherLabelHook,
        lifecycle: _RelaxationLifecycle,
    ) -> None:
        """Store the structures a step budget graduated as the chunk ended.

        A :class:`~nvalchemi.dynamics.FusedStage` sub-stage graduating on an
        ``n_steps`` budget migrates status after the fused ``AFTER_STEP``
        dispatch, so the capture hook reads the moving status on the step the
        budget runs out, and a budget that graduates every remaining graph ends
        the chunk there. This runs before the segment's closing dispatch, which
        would mark the step as covered: the label hook's marker is the
        idempotence guard, since the path route stores a whole frame only while
        nothing has graduated yet, and the capture hook's own record keeps a
        criterion's graduates from being written twice.
        """
        last_step = max(config.dynamics.step_count - 1, 0)
        if label_hook.labeled_step == last_step:
            return
        lifecycle.capture(
            DynamicsContext(
                batch=state, step_count=last_step, workflow=config.dynamics
            ),
            DynamicsStage.AFTER_STEP,
        )

    def _capture_converged(
        self,
        config: OnPolicyConfig,
        lifecycle: _RelaxationLifecycle,
        buffer: ReplayBuffer,
    ) -> None:
        """Label the structures that converged this segment and store them.

        Converged frames are captured raw, at the step each structure reached
        its minimum, and the teacher sees them here in one pass over the whole
        segment's graduates, under the guards the path route labels with. They
        are stripped to the replay-frame contract afterwards, so they enter the
        buffer under the schema the path frames froze it with, and staged onto
        the buffer's own device.
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

        A trajectory finishes converged, frozen by the criterion, or diverged,
        frozen by the lifecycle where its divergence predicate flagged it; the
        diverged ones are counted and warned about here. The initial structures are drawn for the room the graduates
        freed — as many structures, within the atoms they held, and within
        their edges only when the source declared ``max_edges``, since a
        dataset's stored edge count is not the neighbor list a propagator
        rebuilds. The propagator's per-structure state follows the membership
        change, and the run restamps its bookkeeping over the appended rows,
        keeping only the ``system_id`` the source numbered, so a structure
        stored with the ``status`` it once graduated on still moves.

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
        divergence = lifecycle.divergence
        diverged = int((graduated & _checked_divergence(divergence, state)).sum())
        if diverged:
            flagged = (
                "their positions or forces stopped being finite"
                if divergence is nonfinite_divergence
                else f"the divergence predicate {divergence!r} flagged them"
            )
            warnings.warn(
                f"{diverged} of {state.num_graphs} generated trajectories "
                f"diverged: {flagged}, so the lifecycle froze them on that "
                "step, kept them out of both capture routes, and retires and "
                "backfills them here like converged ones. A diverging student "
                "is extrapolating; shorten the propagator's step, or register "
                "a MaxForceClampHook on it.",
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
        lifecycle.capture.reset()
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
        # column but the ids the source numbered is rebuilt over the survivors.
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
        self, config: OnPolicyConfig, target_step_count: int
    ) -> None:
        """Announce that the run trains on what it has already generated."""
        remedy = (
            "Pass initial_structures=InitialStructures(dataset, recycle=True) to "
            "keep generating from the front of the rows this rank owns, or start "
            "from more structures — an unbudgeted source is propagated whole, so "
            "more of them lengthen the run by widening the initial batch rather "
            "than by backfilling it."
            if config.initial_structures.exhausted
            else "The source still holds rows, so widen its budget: nothing a "
            "pass over it reached fits the room the graduates freed."
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

        Frames reach the buffer from a host-memory sink, so an unset
        ``replay_device`` means the reference dataset's device — the two sources
        are collated before the strategy moves the batch — measured from a batch
        when no declaration settles it. A run with no reference dataset leaves
        them in host memory. The measurement is taken here rather than reused
        from construction, because a launcher pins the process to its device
        only after the datasets are built and a reference dataset may be moved
        onto this rank's device by a ``SETUP`` hook.

        A ``replay_device`` spelled index-less is resolved through
        :func:`nvalchemi.data.resolve_device` to the device this process has
        made current — under a launcher, the one it pinned this rank to — so
        the caller's "this rank's GPU" becomes a concrete device the
        concentration check and the mixture's device comparison can reason
        about, the same way a storage records its own. An emitted device is
        concrete already and is left as measured.

        Warns
        -----
        UserWarning
            If a multi-rank world resolves an indexed accelerator that is not
            the device every rank trains on.
        """
        if config.replay_device is not None:
            device = resolve_device(config.replay_device)
        elif self.reference_dataset is None:
            return None
        else:
            device = _emitted_device(self.reference_dataset)
        self._warn_concentrated_replay_device(device)
        return device

    def _warn_concentrated_replay_device(self, device: torch.device) -> None:
        """Report a world staging every rank's replay frames on one accelerator.

        Datasets are built before a launcher pins the process, so a reference
        dataset on an indexed device emits there in every process and the
        buffer follows it, since a mixed batch is collated before the strategy
        moves it; the world's buffers then pile onto one GPU sized for a single
        rank's ``replay_capacity``. An index-less ``cuda`` is what a rank-local
        dataset looks like and is left alone. Each rank reduces the one bit it
        can see, whether it stages on its own device, inside the collective
        rather than behind a guard, so every rank past a single-process world
        joins it and every rank reports the world's verdict.

        Parameters
        ----------
        device : torch.device
            Device the buffer is about to stage its frames on.

        Warns
        -----
        UserWarning
            If a multi-rank world stages its replay frames on an indexed
            accelerator that is not every rank's own device.
        """
        if get_world_size(self.distributed_manager) == 1:
            return
        elsewhere = (
            device.type != "cpu"
            and device.index is not None
            and not _same_device(device, self.devices[0])
        )
        concentrated = all_reduce(
            torch.tensor(int(elsewhere), device=collective_device()),
            self.distributed_manager,
            op=dist.ReduceOp.MAX,
        )
        if not bool(concentrated.item()):
            return
        warnings.warn(
            "Every rank stages its replay buffer and collates its mixture on "
            f"{device!s}, which is not the device every rank trains on: a "
            "reference dataset pre-staged on an indexed device emits there in "
            "every process, and the generated frames have to follow it because "
            "a mixed batch is collated before it is moved. The whole "
            "world's replay frames then sit on one accelerator, sized as if "
            "each rank held its own, and only the rank that owns it is spared. "
            "Keep reference_dataset in host memory, or move it to this rank's "
            "device once the launcher has pinned the process, so every rank "
            "builds its mixture where it trains.",
            UserWarning,
            stacklevel=2,
        )

    def _rank_seed_offset(self, config: OnPolicyConfig) -> int:
        """Return the offset moving this rank's seeded streams off its neighbors'.

        Both seeded streams add a counter to their base seed, so ranks sit a
        whole ``rank_seed_stride`` apart rather than one, keyed on the global
        rank because node-local ranks repeat across nodes.
        """
        return get_rank(self.distributed_manager) * config.rank_seed_stride

    def _capture_segment(
        self,
        config: OnPolicyConfig,
        state: Batch,
        label_hook: TeacherLabelHook,
        buffer: ReplayBuffer,
    ) -> None:
        """Label the frame the segment ended on and drain the sink into *buffer*.

        The cadence rarely lands on a segment's last step, and that frame is the
        most on-policy one it produced, so the hook's forced entry point is called
        for the step just finished: a forced label is never passed over and, being
        idempotent per step, costs nothing where the cadence did land.
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
        hold a live propagator, scorer, and datasets no spec can describe, so they
        are omitted and a rebuilt strategy is offline-shaped.

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
