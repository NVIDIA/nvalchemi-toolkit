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

import contextvars
import dataclasses
import warnings
from collections.abc import Iterable, Iterator, Mapping, Sequence, Sized
from contextlib import contextmanager, nullcontext
from typing import TYPE_CHECKING, Annotated, Any

import torch
from pydantic import Field, PrivateAttr, model_validator
from torch import distributed as dist

from nvalchemi._serialization import _import_cls
from nvalchemi._typing import Forces, ModelOutputs, NodePositions
from nvalchemi.data.datapipes.dataset import BatchDatasetProtocol
from nvalchemi.distributed import collective_device
from nvalchemi.dynamics.base import BaseDynamics, ConvergenceHook, DynamicsStage
from nvalchemi.dynamics.sinks import HostMemory
from nvalchemi.hooks._context import DynamicsContext
from nvalchemi.models.base import BaseModelMixin
from nvalchemi.training import _spec_utils as strategy_spec
from nvalchemi.training import _strategy_validation as strategy_validation
from nvalchemi.training._stages import TrainingStage
from nvalchemi.training.distillation._attach import _attach_teacher_labels
from nvalchemi.training.distillation._restart import (
    _batch_from_state,
    _OnPolicyRestartHook,
)
from nvalchemi.training.distillation.config import OnPolicyConfig
from nvalchemi.training.distillation.hooks import (
    TeacherLabelHook,
    _ConvergedFrameHook,
    _DivergenceHook,
    _nonfinite_graphs,
    _run_local_keys,
    _score_and_attach,
    _strip_replay_frame,
)
from nvalchemi.training.distillation.losses.distribution import BoltzmannMatchingLoss
from nvalchemi.training.distillation.losses.embedding import (
    _PROJECTOR_REMEDY,
    EmbeddingMatchingLoss,
)
from nvalchemi.training.distillation.losses.hessian import HessianMatchingLoss
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
    _HVP_PROBE_FIELD,
    _SIGNAL_SPECS,
    _TEACHER_FIELD_PREFIX,
    InProcessTeacherScorer,
    _isolated_embeddings,
    _node_embedding_shapes,
    _reject_foreign_fields,
    _restore_grad_flags,
    _snapshot_grad_flags,
    hessian_vector_product,
    scorer_fields,
    signal_fields,
    signal_for_field,
)
from nvalchemi.training.distillation.seeding import (
    InitialStructures,
    InitialStructuresSource,
    WithinBudget,
    _check_structure_status,
    _dataset_from_spec_dict,
    _dataset_spec_dict,
    _propagator_tree,
)
from nvalchemi.training.distributed import all_reduce, get_rank, get_world_size
from nvalchemi.training.losses.composition import loss_target_keys
from nvalchemi.training.runtime import (
    freeze_unconfigured_models,
    move_to_devices,
    train_configured_models,
    unwrap_model,
)
from nvalchemi.training.strategy import TrainingStrategy

if TYPE_CHECKING:
    from pathlib import Path

    from torch.optim.lr_scheduler import LRScheduler

    from nvalchemi.data.batch import Batch
    from nvalchemi.dynamics.sinks import DataSink
    from nvalchemi.hooks._context import TrainContext
    from nvalchemi.training.losses.composition import (
        BaseLossFunction,
        ComposedLossFunction,
    )

__all__ = [
    "DistillationStrategy",
    "default_distillation_fn",
    "embedding_distillation_fn",
    "hessian_distillation_fn",
]

_REQUIRED_MODELS = frozenset({"student", "teacher"})
"""Model names every distillation strategy must be given."""

_PROJECTOR_MODEL = "projector"
"""Name of the auxiliary model an embedding objective projects the student with."""

_PREDICTION_KEY_PREFIX = "predicted_"
"""Prefix the stock training functions publish every student output under."""

_HVP_OUTPUT = "hvp"
"""Student output name a Hessian objective's prediction key resolves to."""

_RELAXATION_MODULE = "nvalchemi.dynamics.optimizers"
"""Module every built-in relaxation propagator is defined in."""

_SUPPLIED_RUNTIME_OBJECTS: contextvars.ContextVar[dict[str, Any]] = (
    contextvars.ContextVar("nvalchemi_distillation_runtime_objects", default={})
)
"""Runtime objects a checkpoint rebuild offers the constructor no spec can carry them to."""


@contextmanager
def _supplied_runtime_objects(**objects: Any) -> Iterator[None]:
    """Offer *objects* to the :meth:`DistillationStrategy.from_spec_dict` a rebuild reaches.

    :func:`nvalchemi.training.load_checkpoint` rebuilds a strategy through
    :meth:`~nvalchemi.training.TrainingStrategy.from_checkpoint_dict`, whose
    signature has no room for the live objects a spec cannot describe; they
    travel over a context variable instead, which keeps the parent's loader
    reusable.

    Yields
    ------
    None
    """
    token = _SUPPLIED_RUNTIME_OBJECTS.set(objects)
    try:
        yield
    finally:
        _SUPPLIED_RUNTIME_OBJECTS.reset(token)


_RANK_SEED_STRIDE = 1_000_003
"""Stride separating each rank's seed stream from the next rank's."""

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


def embedding_distillation_fn(
    models: Mapping[str, BaseModelMixin], batch: Batch
) -> dict[str, torch.Tensor]:
    """Run the student forward pass and add its node embeddings as a prediction.

    A representation comes from
    :meth:`~nvalchemi.models.base.BaseModelMixin.compute_embeddings`, a second
    pass over the batch, which is why
    :class:`~nvalchemi.training.distillation.EmbeddingMatchingLoss` needs this
    training function. The batch is left as it was found. A ``"projector"``
    model, when the strategy has one, is applied to the student's embeddings
    on the way out — never to the teacher's — and trains through its own
    ``optimizer_configs`` entry.

    Parameters
    ----------
    models : Mapping[str, BaseModelMixin]
        Named models of the strategy; ``"student"`` and, when present,
        ``"projector"`` are read.
    batch : Batch
        Input batch of atomic graphs.

    Returns
    -------
    dict[str, torch.Tensor]
        The stock ``predicted_*`` outputs plus ``predicted_node_embeddings``.

    Raises
    ------
    RuntimeError
        If the student's ``compute_embeddings`` writes no ``node_embeddings``,
        or writes embeddings detached from the student's trainable parameters
        while gradients are enabled — a wrapper that computes them under
        :func:`torch.no_grad` — which the objective could not train the
        student through.

    See Also
    --------
    nvalchemi.training.distillation.EmbeddingProjector : The width adapter.

    Notes
    -----
    The student is run twice per batch, which doubles its share of a training
    step. ``compute_embeddings`` is not part of the interface a
    :class:`~torch.nn.parallel.DistributedDataParallel` replica proxies, so the
    embedding pass is taken on the wrapped module; its gradients are still
    reduced, but a student submodule exercised *only* by ``compute_embeddings``
    is invisible to ``find_unused_parameters=True``, the one configuration to
    avoid.
    """
    predictions = default_distillation_fn(models, batch)
    student = unwrap_model(models["student"])
    with _isolated_embeddings(batch):
        student.compute_embeddings(batch)
        if "node_embeddings" not in batch:
            raise RuntimeError(
                "Student compute_embeddings() must write ``node_embeddings`` onto "
                "the batch for embedding matching; got a batch carrying "
                f"{sorted(key for key in _EMBEDDING_KEYS if key in batch)!r}."
            )
        embeddings = batch["node_embeddings"]
    if (
        torch.is_grad_enabled()
        and not embeddings.requires_grad
        and any(parameter.requires_grad for parameter in student.parameters())
    ):
        raise RuntimeError(
            "Student compute_embeddings() returned node embeddings detached from "
            "the student's trainable parameters, so the embedding objective would "
            "train the projector and nothing else; got a "
            f"{type(student).__name__!r} student whose embedding pass runs without "
            "gradients. Compute the student's embeddings with gradients enabled — "
            "override compute_embeddings on the wrapper — or drop the term."
        )
    if _PROJECTOR_MODEL in models:
        embeddings = models[_PROJECTOR_MODEL](embeddings)
    predictions["predicted_node_embeddings"] = embeddings
    return predictions


def hessian_distillation_fn(
    models: Mapping[str, BaseModelMixin], batch: Batch
) -> dict[str, torch.Tensor]:
    """Run the student forward pass and add its Hessian-vector product.

    The product is taken along ``teacher_hvp_probe``, the direction the
    teacher's own product was labeled with, on a second pass narrowed to the
    student's energy: a conservative model derives its forces from the graph
    the second derivative needs and frees it outside training mode, so the
    stock forward's energy cannot be differentiated again. Both derivatives are
    taken with ``create_graph=True``, which is what
    :class:`~nvalchemi.training.distillation.HessianMatchingLoss`
    backpropagates through. The batch's ``requires_grad`` flags are restored,
    and the narrowed pass reuses the neighbor list the stock forward just ran
    on.

    Parameters
    ----------
    models : Mapping[str, BaseModelMixin]
        Named models of the strategy; only ``"student"`` is read.
    batch : Batch
        Input batch, carrying the ``teacher_hvp_probe`` field the ``hessian``
        teacher signal writes.

    Returns
    -------
    dict[str, torch.Tensor]
        The stock ``predicted_*`` outputs plus ``predicted_hvp``.

    Raises
    ------
    AttributeError
        If the batch carries no probe, which means it was never labeled with
        the ``hessian`` signal.
    KeyError
        If the student computes no energy to differentiate.

    Notes
    -----
    The second pass adds two backward passes, one through a second-order graph
    held for the whole step, on every frame the student trains on. A stochastic
    student draws afresh in the narrowed pass, so its curvature is measured on
    a different realization than its energy.
    """
    probe = getattr(batch, _HVP_PROBE_FIELD, None)
    if probe is None:
        raise AttributeError(
            f"Batch is missing the {_HVP_PROBE_FIELD!r} field required to take "
            "the student's Hessian-vector product along the direction the "
            "teacher was labeled with. Request the 'hessian' teacher signal so "
            "the probe travels with the label."
        )
    student = unwrap_model(models["student"])
    grad_flags = _snapshot_grad_flags(batch, student.model_config)
    try:
        predictions = default_distillation_fn(models, batch)
        predictions["predicted_hvp"] = _student_hvp(student, batch, probe)
    finally:
        _restore_grad_flags(batch, grad_flags)
    return predictions


def _student_hvp(student: BaseModelMixin, batch: Batch, probe: NodePositions) -> Forces:
    """Return the student's Hessian-vector product from an energy-only pass.

    Whatever neighbor list the batch carries is used as it stands. The teacher
    side isolates a rebuild because the batch it is handed was built for a
    different model at a different cutoff, but this pass runs the model the
    stock forward just ran on this very batch, so a rebuild would only
    reproduce the list that forward already consumed — once per optimizer step,
    for every neighbor-list student.
    """
    config = student.model_config
    previous_active = set(config.active_outputs)
    try:
        student.set_config("active_outputs", {"energy"})
        positions = batch.positions
        with torch.enable_grad():
            positions.requires_grad_(True)
            energy = student(batch).get("energy")
            if energy is None:
                raise KeyError(
                    "Hessian matching differentiates the student's energy "
                    "twice, so the student must compute an energy; got a "
                    f"student declaring outputs {sorted(config.outputs)!r}."
                )
            return hessian_vector_product(energy, positions, probe, create_graph=True)
    finally:
        student.set_config("active_outputs", previous_active)


_STOCK_TRAINING_FNS = {
    default_distillation_fn: frozenset(),
    embedding_distillation_fn: frozenset({"node_embeddings"}),
    hessian_distillation_fn: frozenset({_HVP_OUTPUT}),
}
"""Predictions each stock training function adds beyond the student's outputs."""


def _derived_teacher_signals(loss_fn: ComposedLossFunction) -> frozenset[str]:
    """Return the built-in teacher signals the loss composition's targets require.

    A ``teacher_*`` target no built-in signal populates is a custom teacher
    field — one :func:`~nvalchemi.training.distillation.label_dataset` persisted
    from a custom scorer — that the batch must already carry, so it is passed
    over here rather than refused. A companion field a signal writes beside its
    own — the probe direction of ``hessian`` — is refused instead: it records
    how the label was produced and is nothing to supervise against.
    """
    signals: set[str] = set()
    for key in loss_target_keys(loss_fn):
        signal = (
            signal_for_field(key) if key.startswith(_TEACHER_FIELD_PREFIX) else None
        )
        if signal is None:
            continue
        produced = _SIGNAL_SPECS[signal].field
        if key != produced:
            raise ValueError(
                "Loss targets must name what a teacher signal produces; got "
                f"{key!r}, which the {signal!r} signal writes alongside "
                f"{produced!r} to record how that field was produced: a probe "
                "is the direction the product was taken along, not a quantity "
                f"the student is supervised against. Point the loss at "
                f"{produced!r}."
            )
        signals.add(signal)
    return frozenset(signals)


def _matching_components(
    loss_fn: ComposedLossFunction, kind: type[Any]
) -> tuple[str, ...]:
    """Return the class names of the loss components that are instances of *kind*."""
    return tuple(
        type(component).__name__
        for component in loss_fn.components
        if isinstance(component, kind)
    )


@contextmanager
def _eval_configured_models(
    models: Mapping[str, torch.nn.Module], optimizer_configs: Mapping[str, object]
) -> Iterator[None]:
    """Temporarily put the optimizer-configured models in evaluation mode.

    The mirror of :func:`~nvalchemi.training.runtime.train_configured_models`,
    which only ever sets training mode: a model never told otherwise generates
    with dropout live, batch-norm statistics moving, and a conservative model's
    forces building a second-order graph.

    Parameters
    ----------
    models : Mapping[str, torch.nn.Module]
        Named models participating in the run.
    optimizer_configs : Mapping[str, object]
        Optimizer configuration keyed by model name; the models it names are
        switched to evaluation mode while the context is active.

    Yields
    ------
    None
    """
    state = {
        name: model.training
        for name, model in models.items()
        if name in optimizer_configs
    }
    for name in state:
        models[name].eval()
    try:
        yield
    finally:
        for name, training in state.items():
            models[name].train(training)


@contextmanager
def _eval_propagator_model(
    propagator_model: object, student: BaseModelMixin
) -> Iterator[None]:
    """Temporarily put a propagator model that only *composes* the student in eval mode.

    A composition holding the student is no entry of ``models``, so nothing
    else takes it out of training mode, where a shared-autograd composition
    builds the second-order graph generation exists to avoid. Enter this
    context inside :func:`_eval_configured_models`, since restoring the
    composition's mode touches the student too; every submodule's own mode is
    snapshotted, so a correction head the caller froze alone comes back frozen.

    Parameters
    ----------
    propagator_model : object
        Model the propagator holds; *student* itself, or anything that is not
        a :class:`torch.nn.Module`, is left alone.
    student : BaseModelMixin
        Student the strategy trains.

    Yields
    ------
    None
    """
    if propagator_model is student or not isinstance(propagator_model, torch.nn.Module):
        yield
        return
    modes = {module: module.training for module in propagator_model.modules()}
    propagator_model.eval()
    try:
        yield
    finally:
        for module, training in modes.items():
            module.training = training


@dataclasses.dataclass(frozen=True)
class _RelaxationLifecycle:
    """Machinery a relaxation segment loop drives between its segments."""

    capture: _ConvergedFrameHook
    structures: InitialStructures


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
    divergence hook behind the criterion freezes a graph whose state stopped
    being finite at ``exit_status``, uncaptured, so the boundary retires it
    like a converged one.

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
    capture = _ConvergedFrameHook(sink=HostMemory(capacity=state.num_graphs))
    divergence = _DivergenceHook()
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
            capture=capture, structures=config.initial_structures
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


def _graduates_graphs_out(hook: object, exit_status: int) -> bool:
    """Return whether a registered *hook* migrates converged graphs past *exit_status*.

    A :class:`~nvalchemi.dynamics.base.ConvergenceHook` migrates only when it
    carries both a source and a target status, and a target the root propagator
    still steps hands the graph to another sub-stage rather than freezing it —
    which is what a :class:`~nvalchemi.dynamics.FusedStage` installs between its
    own sub-stages as it is constructed.
    """
    return (
        isinstance(hook, ConvergenceHook)
        and hook.source_status is not None
        and hook.target_status is not None
        and hook.target_status >= exit_status
    )


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
    if it offers one.

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
        small and offers no ``resize``.
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
        resize = getattr(sink, "resize", None)
        if not callable(resize):
            raise ValueError(
                "OnPolicyConfig.capture_sink must hold every frame one segment "
                "can capture, (generation_steps + 1) per trajectory: "
                f"{capacity!r} for {num_graphs!r} trajectories over "
                f"{config.generation_steps!r} steps; got a {type(sink).__name__} "
                f"of capacity {sink.capacity!r} without a resize method. Build it "
                "with at least that capacity, or give it resize(capacity)."
            )
        resize(capacity)
    return sink


def _to_device(batch: Batch, device: torch.device) -> Batch:
    """Return *batch* on *device*, overlapping the copy only into device memory.

    A copy into host memory returns before the transfer lands, and the moved
    batch's ``segment_lengths`` and ``batch_ptr`` are read on the host at once,
    so that direction blocks.
    """
    return batch.to(device, non_blocking=device.type != "cpu")


class _TeacherLabelHook:
    """Label the batch a forward pass is about to consume, training or validation."""

    frequency = 1
    stage = TrainingStage.BEFORE_FORWARD

    def __call__(self, ctx: TrainContext, stage: TrainingStage) -> None:  # noqa: ARG002
        """Attach the teacher fields the upcoming batch is missing."""
        strategy: DistillationStrategy = ctx.workflow
        if ctx.batch is not None and strategy.label_missing:
            strategy.attach_teacher_labels(ctx.batch)


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
    ``validation_config`` to the constructor. Every resolved signal is required
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

    Beyond the signals that have a supervised shape, three objectives need more
    from the run than a target field. Embedding matching needs a second pass
    over the batch on both sides and, across architectures, the learnable
    :class:`~nvalchemi.training.distillation.EmbeddingProjector` registered as a
    ``"projector"`` model with an optimizer of its own; Hessian matching needs
    the student's energy differentiated twice along the probe the teacher was
    labeled with; and both need the training function that produces those
    predictions —
    :func:`~nvalchemi.training.distillation.embedding_distillation_fn` and
    :func:`~nvalchemi.training.distillation.hessian_distillation_fn` — since
    neither is a forward-pass output. Boltzmann matching needs no new
    prediction but does need the on-policy loop, because it reads a batch as a
    sample of the student's own ensemble. All three are checked at
    construction.

    Raises
    ------
    ValueError
        If ``models`` is not a named mapping containing ``"student"`` and
        ``"teacher"``, if the teacher is given an optimizer config, if the
        student or an auxiliary model is not, if a loss component reads a
        prediction the student does not compute or names one outside the
        ``predicted_`` namespace under a stock ``training_fn``, if an explicit
        ``teacher_signals`` omits a signal a loss needs, if no built-in teacher
        signal is requested at all, if the teacher cannot produce a requested signal,
        or if the teacher is a composition that plans more than one
        neighbor-list source. With an embedding objective on the stock embedding
        training function, additionally if the student publishes no
        node-embedding shape or the student, projector, and teacher widths do
        not compose; with a Hessian objective, if the student computes no
        energy; with a Boltzmann objective, if the run is not on-policy, if it
        generates with a relaxation or converging propagator, or if the term
        sits in the validation loss. In on-policy mode, additionally if the run is
        sized in epochs, if the propagator holds neither the student nor a
        model composing it, if ``replay_ratio`` and ``reference_dataset``
        disagree (a ratio below ``1`` needs one, a ratio of ``1`` refuses
        one), if ``reference_dataset`` emits on an accelerator the run does
        not train on, if ``replay_device`` or the reference dataset's fields
        cannot be mixed with generated frames, or if the propagator's scorer
        and the reference dataset do not carry the same teacher fields.

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
    Labeling runs with autocast disabled, and labels are cast to the student's
    first floating-point parameter dtype, never below single precision, so a
    ``bfloat16`` or ``float16`` student gets float32 labels and needs
    ``dtype_policy="prediction_to_target"`` on its loss terms; a float64 student
    reads float32 back from a store and needs a ``dtype_policy`` too. Labels are
    attached to the device-placed copy the strategy trains on, not the caller's
    batch, so a loader replaying the same systems costs one teacher pass per
    epoch.

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
    Checkpoints store the frozen teacher once per checkpoint root and reference
    it from every later index, so a periodic write costs the student's weights
    alone; see :meth:`checkpoint_model_references`.

    ``on_policy`` and ``reference_dataset`` serialize as references — the
    propagator's constructor spec, the scorer's signals over
    ``models["teacher"]``, and the stores the datasets read — so
    :meth:`to_spec_dict` carries a whole on-policy recipe and
    :meth:`from_spec_dict` rebuilds it around the supplied ``models``. A piece
    no recipe describes, such as an in-memory dataset or a propagator hiding
    its constructor arguments, leaves ``on_policy`` out with a warning.
    :meth:`from_spec_dict`, :meth:`from_checkpoint_dict`, and
    :meth:`load_checkpoint` take ``on_policy`` and ``reference_dataset``
    keyword arguments, and a live object outranks the recipe; a Boltzmann term
    refuses to rebuild without the loop. An interrupted on-policy run carries
    its trajectory, the propagator's step count, the initial-structure cursor,
    and its replay frames through the checkpoint, so a resumed run continues
    rather than reseeding.
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
    _replay_buffer: ReplayBuffer | None = PrivateAttr(default=None)
    _on_policy_state: Any = PrivateAttr(default=None)
    _generation_exhausted: bool = PrivateAttr(default=False)
    _validation_probe_index: int | None = PrivateAttr(default=None)
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
        """Put the internal labeling and restart hooks ahead of the caller's hooks.

        A seam carried in the incoming hooks is replaced rather than kept, so
        rebuilding a strategy from a live one's ``hooks`` leaves exactly one of
        each, still ahead of every caller hook.
        """
        if not isinstance(data, dict):
            return data
        normalized = dict(data)
        internal: list[Any] = [_TeacherLabelHook()]
        if normalized.get("on_policy") is not None:
            internal.append(_OnPolicyRestartHook())
        normalized["hooks"] = [
            *internal,
            *(
                hook
                for hook in (normalized.get("hooks") or [])
                if not isinstance(hook, (_TeacherLabelHook, _OnPolicyRestartHook))
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
        self._validate_student_outputs()
        signals = self._resolve_teacher_signals()
        self._scorer = InProcessTeacherScorer(
            self.models["teacher"],
            signals,
            dtype=_student_label_dtype(self.models["student"]),
        )
        self._teacher_fields = signal_fields(signals)
        return self

    def _validate_student_outputs(self) -> None:
        """Check both losses' prediction keys against what the student computes.

        Only under a stock function: each emits ``active_outputs`` intersected
        with ``outputs`` plus whatever it derives on top, so a narrowed student
        is caught here rather than on its first batch. The validation loss is
        checked whenever ``validation_fn`` falling back to ``training_fn`` is
        a stock one.
        """
        derived = _STOCK_TRAINING_FNS.get(self.training_fn)
        if derived is not None:
            self._validate_prediction_keys(self.loss_fn.components, "training", derived)
        validation = self.validation_config
        if validation is None or validation.loss_fn is None:
            return
        validation_derived = _STOCK_TRAINING_FNS.get(
            validation.validation_fn or self.training_fn
        )
        if validation_derived is not None:
            self._validate_prediction_keys(
                validation.loss_fn.components, "validation", validation_derived
            )

    def _validate_prediction_keys(
        self,
        components: Sequence[BaseLossFunction],
        side: str,
        derived: frozenset[str],
    ) -> None:
        """Check one composition's prediction keys, naming *side* in every error.

        *derived* names the predictions the stock training function in play adds
        on top of the student's own outputs.
        """
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
                    f"The {label} reads prediction_key={key!r}, which the stock "
                    "training functions never emit: they publish every student "
                    f"output under {_PREDICTION_KEY_PREFIX}<output>. Rename the "
                    "key into that namespace, or pass a training_fn that owns "
                    "its own convention."
                )
            output = key.removeprefix(_PREDICTION_KEY_PREFIX)
            if output in active or output in derived:
                continue
            if output in _EMBEDDING_KEYS:
                raise ValueError(
                    f"The {label} reads prediction_key={key!r}, which this "
                    "training_fn cannot produce: embeddings come from the "
                    "student's compute_embeddings(), not from its forward pass. "
                    "Pass training_fn=embedding_distillation_fn, which calls "
                    "compute_embeddings, routes the result through a 'projector' "
                    f"model when one is registered, and returns it under {key!r}."
                )
            if output == _HVP_OUTPUT:
                raise ValueError(
                    f"The {label} reads prediction_key={key!r}, which this "
                    "training_fn cannot produce: a Hessian-vector product is a "
                    "second derivative of the student's energy, not a forward "
                    "output. Pass training_fn=hessian_distillation_fn, which "
                    "differentiates the energy twice along the probe the teacher "
                    "was labeled with."
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

    @model_validator(mode="after")
    def _validate_advanced_objectives(self) -> DistillationStrategy:
        """Enforce what the representation, curvature, and Boltzmann terms need."""
        self._validate_embedding_matching()
        self._validate_hessian_matching()
        self._validate_distribution_matching()
        return self

    def _matching_sides(
        self, kind: type[Any], training_fn: Any = None
    ) -> list[tuple[tuple[str, ...], str]]:
        """Return the components of *kind* each side's loss runs under *training_fn*.

        A ``validation_config`` carrying its own ``loss_fn`` reaches the student
        through its effective validation function — ``validation_fn`` falling
        back to ``training_fn`` — so a term only the validation loss holds is
        checked here too, and the side it came from names it in every message.
        ``training_fn=None`` matches a term whatever function its side runs.
        """
        sides = [(self.loss_fn, self.training_fn, "training")]
        validation = self.validation_config
        if validation is not None and validation.loss_fn is not None:
            sides.append(
                (
                    validation.loss_fn,
                    validation.validation_fn or self.training_fn,
                    "validation",
                )
            )
        return [
            (terms, side)
            for loss_fn, effective_fn, side in sides
            if (terms := _matching_components(loss_fn, kind))
            and (training_fn is None or effective_fn is training_fn)
        ]

    def _validate_embedding_matching(self) -> None:
        """Reconcile the student, projector, and teacher embedding widths.

        Only the stock embedding training function is checked, because it is the
        one whose routing this can reason about: it projects the student's
        embeddings with ``models['projector']`` when there is one, so the widths
        have to compose. A caller's own training function owns its own routing.
        """
        student = self.models["student"]
        for terms, side in self._matching_sides(
            EmbeddingMatchingLoss, embedding_distillation_fn
        ):
            label = f"{side} loss component(s) {list(terms)!r}"
            student_shape = _node_embedding_shapes(student).get("node_embeddings")
            if student_shape is None:
                raise ValueError(
                    f"The {label} match the student's node embeddings, so the "
                    "student must publish a 'node_embeddings' shape and write it "
                    "in compute_embeddings(); got embedding_shapes="
                    f"{sorted(_node_embedding_shapes(student))!r}."
                )
            width = student_shape[-1]
            projector = (
                self.models[_PROJECTOR_MODEL]
                if _PROJECTOR_MODEL in self.models
                else None
            )
            if projector is not None:
                in_features = getattr(projector, "in_features", None)
                if in_features is not None and in_features != width:
                    raise ValueError(
                        "The projector reads the student's embeddings, so its input "
                        f"width must be the student's; got in_features={in_features!r} "
                        f"against a student of width {width!r}."
                    )
                width = getattr(projector, "out_features", width)
            teacher_shape = _node_embedding_shapes(self.models["teacher"]).get(
                "node_embeddings"
            )
            if teacher_shape is not None and width != teacher_shape[-1]:
                raise ValueError(
                    f"The {label} compare representations component by component, "
                    "so what reaches the loss must have the teacher's width; got "
                    f"{width!r} against a teacher of width {teacher_shape[-1]!r}. "
                    f"{_PROJECTOR_REMEDY}"
                )

    def _validate_hessian_matching(self) -> None:
        """Check the student has the energy a curvature objective differentiates.

        A direct-force student is warned about rather than refused: the term is
        computable and its numbers are right, but the second derivative it
        drives down is the one its energy head implies, not the Jacobian of the
        force head a force loss trains, so the curvature that decides an
        integrator's stability for that student is left unsupervised.
        """
        sides = self._matching_sides(HessianMatchingLoss, hessian_distillation_fn)
        if not sides:
            return
        active = self.models["student"].output_data()
        for terms, side in sides:
            if "energy" not in active:
                raise ValueError(
                    f"The {side} loss component(s) {list(terms)!r} need the "
                    "student's Hessian-vector product, which "
                    "hessian_distillation_fn takes by differentiating the "
                    "student's energy twice, so the student must compute an "
                    f"energy; got active outputs {sorted(active)!r}."
                )
        config = self.models["student"].model_config
        if "forces" in config.outputs and "forces" not in config.autograd_outputs:
            named = sorted({name for terms, _ in sides for name in terms})
            warnings.warn(
                f"Loss component(s) {named!r} "
                "differentiate the student's energy twice, but the student "
                "predicts its forces with a head of its own rather than as that "
                "energy's gradient; got autograd_outputs="
                f"{sorted(config.autograd_outputs)!r}. The curvature term then "
                "supervises the energy head alone, and the force head a force "
                "loss trains gets no second-order signal at all. Distill a "
                "conservative student for the term to reach the forces, or read "
                "it as a constraint on the energy surface only.",
                UserWarning,
                stacklevel=2,
            )

    def _validate_distribution_matching(self) -> None:
        """Require an equilibrium on-policy sample for every Boltzmann term.

        The estimator reads the batch as a sample of the student's own
        canonical distribution, so it needs the segment loop and a propagator
        that keeps sampling: a relaxation propagator descends to a minimum and
        a converging one freezes each graph as it arrives, whether the
        criterion is the propagator's own, a hook registered on it, or the one
        the loop installs from ``fmax`` or ``convergence_hook``. What reaches
        the loss is a draw from the replay buffer, so an unbounded buffer and a
        mixed ``replay_ratio`` are warned about. Validation data is off-policy
        by construction, so a term on the validation side is refused, as is a
        validation config that would reuse the training loss.
        """
        sides = {
            side: terms for terms, side in self._matching_sides(BoltzmannMatchingLoss)
        }
        if "validation" in sides:
            raise ValueError(
                f"The validation loss component(s) {list(sides['validation'])!r} "
                "read a batch as a sample of the student's own Boltzmann "
                "distribution, and a validation set is off-policy by construction: "
                "a fixed sample of whatever produced it, on which the uniform "
                "weights the estimator assumes are wrong rather than noisy, so the "
                "metric would mislead checkpoint selection and the metric "
                "schedulers. Give the validation config a pointwise loss — "
                "EnergyMSELoss(target_key='teacher_energy') + "
                "ForceMSELoss(target_key='teacher_forces') — instead."
            )
        terms = sides.get("training")
        if terms is None:
            return
        if self.on_policy is None:
            raise ValueError(
                f"Loss component(s) {list(terms)!r} compare the teacher's and "
                "student's Boltzmann ensembles over configurations the student "
                "itself visited, and read a batch as a sample of the student's "
                "own distribution; an offline dataset is a sample of whatever "
                "produced it, which makes the objective's weights wrong rather "
                "than merely noisy. Got on_policy=None; configure the segment "
                "loop, or drop the term. A spec or checkpoint carries no "
                "segment loop, so a rebuild of one re-supplies it: pass "
                "on_policy (with the models its propagator holds) to "
                "load_checkpoint or from_spec_dict, or restore_checkpoint into "
                "a strategy already built with it. Reweighting an off-policy "
                "sample is not offered, because the importance weights the estimator "
                "folds away as uniform are not recoverable from the batch it "
                "is handed; an existing dataset reaches the term as "
                "reference_dataset instead, mixed into generated frames by "
                "replay_ratio and read as regularization."
            )
        stages = list(_propagator_tree(self.on_policy.dynamics))
        relaxing = [
            type(stage).__name__
            for stage in stages
            if type(stage).__module__.startswith(_RELAXATION_MODULE)
        ]
        if relaxing:
            raise ValueError(
                f"Loss component(s) {list(terms)!r} are defined on an equilibrium "
                "ensemble, and a relaxation propagator does not sample one: it "
                "descends to a minimum, so its frames are a path rather than a "
                f"distribution. Got a propagator driving {relaxing!r}; generate "
                "with a thermostatted integrator, or drop the term."
            )
        if (
            self.on_policy.fmax is not None
            or self.on_policy.convergence_hook is not None
        ):
            configured = (
                f"fmax={self.on_policy.fmax!r}"
                if self.on_policy.fmax is not None
                else f"convergence_hook={self.on_policy.convergence_hook!r}"
            )
            raise ValueError(
                f"Loss component(s) {list(terms)!r} are defined on an equilibrium "
                "ensemble, and a segment loop that converges graphs out stops "
                "sampling them: the criterion freezes each converged graph at the "
                "state it converged to and graduates it out of the batch the term "
                "is matching against. The propagator does not carry it until the "
                f"loop installs it, so it is refused here. Got {configured}; "
                "generate without a convergence criterion, or drop the term."
            )
        exit_status = self.on_policy.dynamics.exit_status
        converging = [
            type(stage).__name__
            for stage in stages
            if getattr(stage, "convergence_hook", None) is not None
            or any(
                _graduates_graphs_out(hook, exit_status)
                for hook in getattr(stage, "hooks", ())
            )
        ]
        if converging:
            raise ValueError(
                f"Loss component(s) {list(terms)!r} are defined on an equilibrium "
                "ensemble, and a propagator that converges graphs out stops "
                "sampling them: every converged graph is frozen at the state it "
                f"converged to. Got a convergence hook on {converging!r}; "
                "generate without one, or drop the term."
            )
        if self.on_policy.replay_ratio < 1.0:
            warnings.warn(
                f"Loss component(s) {list(terms)!r} read every batch as a sample "
                "of the student's own ensemble, but replay_ratio="
                f"{self.on_policy.replay_ratio!r} mixes reference frames into "
                "each one, which the estimator cannot tell apart from generated "
                "ones and weights as if the student had visited them. Set "
                "replay_ratio=1 to keep reference rows out of the batch — the "
                "estimate is then as current as the replay buffer is, which is "
                "what replay_capacity bounds — or keep the reference share small "
                "and read the term as regularization.",
                UserWarning,
                stacklevel=2,
            )
        if self.on_policy.replay_capacity is None:
            labelings = (
                self.on_policy.generation_steps // self.on_policy.label_frequency + 1
            )
            warnings.warn(
                f"Loss component(s) {list(terms)!r} read every batch as a sample "
                "of the student's own ensemble, but an unbounded replay buffer "
                "retires nothing and every segment's loader draws uniformly over "
                "all of it: after N segments only about one N-th of a batch came "
                "from the current student and the rest is the time-average of "
                "every policy the run has had, which is the off-policy sample an "
                "offline dataset is refused for. Got replay_capacity=None. Bound "
                "it to what one segment or a few segments yield — "
                f"{labelings} labeling(s) per segment here, one frame per walker "
                "each — and leave replay_eviction='fifo', which retires the "
                "stalest frames first. The size is a trade-off the run owns: a "
                "one-segment buffer is the most current and gives the softmax "
                "the fewest distinct configurations to weight.",
                UserWarning,
                stacklevel=2,
            )
        if (
            self.validation_config is not None
            and self.validation_config.loss_fn is None
        ):
            raise ValueError(
                f"Loss component(s) {list(terms)!r} are defined on the batches "
                "the student generated, and a validation set is off-policy by "
                "construction: it is a fixed sample of whatever produced it, its "
                "graphs need not be one system's configurations, and reducing "
                "energies by k_B T lets the term dominate the composite metric "
                "that checkpoint selection and the metric schedulers read. A "
                "ValidationConfig without a loss_fn of its own reuses this "
                "strategy's, Boltzmann term included, and the labeling seam "
                "scores validation batches for it, so the term would run there. "
                "Got validation_config.loss_fn=None; give the validation config "
                "a pointwise loss — EnergyMSELoss(target_key='teacher_energy') + "
                "ForceMSELoss(target_key='teacher_forces') — or drop the "
                "validation config."
            )

    def attach_teacher_labels(self, batch: Batch) -> bool:
        """Attach the teacher fields *batch* is missing, and report whether it did.

        A batch already carrying every resolved ``teacher_*`` field is left
        untouched, so pre-labeling a batch that later reaches :meth:`run` costs
        one teacher pass; a batch carrying only some of them is re-scored in
        full, since a partial set was written for a different signal set. The
        teacher runs with autocast disabled, so the labels match what
        :func:`~nvalchemi.training.distillation.label_dataset` persisted
        wherever the store returns the label dtype. Inside :meth:`validate` the
        Hessian probe is pinned per batch; see there.

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
        if all(field in batch for field in self._teacher_fields):
            return False
        with (
            torch.autocast(device_type=batch.device.type, enabled=False),
            self._pinned_validation_probe(),
        ):
            labels = self.teacher_scorer.label(batch)
        _attach_teacher_labels(batch, labels)
        return True

    @contextmanager
    def _pinned_validation_probe(self) -> Iterator[None]:
        """Key this batch's Hessian probe to its position in the validation pass.

        A no-op outside :meth:`validate`, where redrawing the probe is what
        covers the Hessian over a run.
        """
        index = self._validation_probe_index
        if index is None:
            yield
            return
        self._validation_probe_index = index + 1
        scorer = self.teacher_scorer
        previous = scorer.probe_seed
        scorer.probe_seed = index
        try:
            yield
        finally:
            scorer.probe_seed = previous

    def validate(self) -> dict[str, Any] | None:
        """Run a validation pass whose batches keep their probe directions.

        Validation batches are relabeled on the fly on every pass, and a
        curvature objective would then draw a fresh Hutchinson probe each time,
        moving the reported number for a student that had not changed. Each
        batch is instead scored along a direction keyed to its position in the
        pass, which makes the metric a function of the student alone as long
        as the validation data iterates in a stable order. Training keeps
        drawing fresh probes.

        Returns
        -------
        dict[str, Any] | None
            The validation summary, also stored on ``last_validation``.
        """
        self._validation_probe_index = 0
        try:
            return super().validate()
        finally:
            self._validation_probe_index = None

    def checkpoint_model_references(self) -> dict[str, dict[str, Any]]:
        """Return the models a checkpoint stores once per root, not at every index.

        The teacher is frozen for the whole run, so writing it into every
        periodic checkpoint duplicates a model that never changed. Declared
        here, it is stored at the first index under a root and referenced by
        every later checkpoint, so a periodic write costs the student's weights
        alone and a restart reads back the teacher the run actually trained
        against — stored rather than rebuilt from its ``checkpoint_spec()``,
        which names the factory that built the architecture but not the
        weights a fine-tune left in it. The reference carries a sampled
        fingerprint that identifies the stored copy without validating it.

        Returns
        -------
        dict[str, dict[str, Any]]
            ``{"teacher": {"rebuild": "stored"}}``; the checkpoint layer adds
            the index and the fingerprint.
        """
        return {"teacher": {"rebuild": "stored"}}

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
        at the final step. The segment is also the restart granularity: a run
        restored from a checkpoint picks its trajectory, the propagator's step
        count, the initial-structure cursor, and its replay frames back up, the
        restored frames replacing the buffer's contents, and a segment a
        checkpoint interrupted is counted as finished on the way in, so a
        checkpoint written part-way through a training phase costs the resumed
        run one extra generation phase. Once generation has run dry, the bundle
        carries the frames and the exhaustion itself, and the resumed run keeps
        training on the buffer. The bundle is rank-local, since the
        checkpoint it rides in is written on rank zero alone, so a multi-rank
        restart drops it with a warning and each rank reseeds with a cold
        buffer. A second call keeps the replay buffer the first filled and
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

        The labeling hook is rebuilt every call, so a resumed run hands it the
        step the interrupted one last labeled: the forced boundary label at
        ``generation_steps - 1``, which the restart bundle does not carry, and
        without which the cadence would fire on the adjacent step and store a
        frame one step from one already in the buffer.

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
            self._run_setup_hooks()
            self._validate_synchronized_student()
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
                if self._replay_buffer is None:
                    self._replay_buffer = ReplayBuffer(
                        capacity=config.replay_capacity,
                        eviction=config.replay_eviction,
                        admission=config.replay_admission,
                        device=replay_device,
                    )
                buffer = self._replay_buffer
                self._generation_exhausted = False
                state, labeled_step = self._resume_or_seed(config, buffer)
                if state is not None:
                    state = _to_device(state, primary_device)
                self._on_policy_state = state
                lifecycle_context = (
                    _relaxation_lifecycle(config, state)
                    if state is not None
                    else nullcontext(None)
                )
                with lifecycle_context as lifecycle:
                    # Only a lifecycle stores a graduated graph elsewhere; a
                    # propagator managing its own keeps every frame here.
                    label_hook = TeacherLabelHook(
                        config.teacher_scorer,
                        frequency=config.label_frequency,
                        exit_status=None
                        if lifecycle is None
                        else config.dynamics.exit_status,
                    )
                    label_hook._labeled_step = labeled_step
                    config.dynamics.register_hook(label_hook)
                    # The mode contexts want the module a DDPHook may have wrapped.
                    student = unwrap_model(self.models["student"])
                    try:
                        # The teacher is frozen across both phases; the student
                        # sits in eval mode and is flipped to training mode by
                        # the inner context for the training phase only.
                        with (
                            freeze_unconfigured_models(
                                self.models, self.optimizer_configs
                            ),
                            _eval_configured_models(
                                self.models, self.optimizer_configs
                            ),
                            _eval_propagator_model(config.dynamics.model, student),
                            _rank_local_propagator_seed(
                                config.dynamics, self._rank_seed_offset()
                            ),
                        ):
                            while self.step_count < target_step_count:
                                if state is not None:
                                    state = self._generate_segment(
                                        config, state, label_hook, lifecycle, buffer
                                    )
                                    self._on_policy_state = state
                                    if state is None:
                                        self._generation_exhausted = True
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

    def _validate_synchronized_student(self) -> None:
        """Reject a multi-rank run whose student nothing keeps in step.

        Called after the ``SETUP`` stage, when a
        :class:`~nvalchemi.training.hooks.DDPHook` has replaced every
        optimizer-configured model with a wrapper publishing the module it
        owns. The check is that something owns ``models["student"]``, read the
        way :func:`~nvalchemi.training.runtime.unwrap_model` reads it, so a
        hand-rolled or FSDP wrapper clears it as a ``DDPHook`` does and the
        model the propagator happens to hold plays no part.

        Raises
        ------
        ValueError
            If nothing has taken ownership of the student to synchronize its
            gradients.
        """
        world_size = get_world_size(self.distributed_manager)
        if world_size == 1:
            return
        student = self.models["student"]
        if unwrap_model(student) is student:
            raise ValueError(
                "A multi-rank segment loop trains one student from every rank's "
                "own frames, so the gradients have to be synchronized: without "
                "that, each rank keeps a private student, generates from it, and "
                "the policies diverge segment by segment while only rank zero's "
                "is checkpointed. Got the bare student still registered as "
                f"models['student'] on {world_size!r} ranks; add a DDPHook to "
                "hooks, which wraps every optimizer-configured model at setup "
                "and leaves the frozen teacher replicated and out of the "
                "all-reduce, or install a gradient-synchronizing wrapper of "
                "your own — the check is that something owns models['student'] "
                "by the end of the SETUP stage, not that a DDPHook put it there."
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
            seed=config.seed + self._rank_seed_offset(),
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
        frozen by the lifecycle; the diverged ones are counted and warned about
        here. The initial structures are drawn for the room the graduates
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
        diverged = int((graduated & _nonfinite_graphs(state)).sum())
        if diverged:
            warnings.warn(
                f"{diverged} of {state.num_graphs} generated trajectories "
                "diverged: their positions or forces stopped being finite, so "
                "the lifecycle froze them on that step, kept them out of both "
                "capture routes, and retires and backfills them here like "
                "converged ones. A diverging student is extrapolating; shorten "
                "the propagator's step, or register a MaxForceClampHook on it.",
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

        A ``replay_device`` spelled index-less is resolved to the device this
        process has made current — under a launcher, the one it pinned this
        rank to — so the caller's "this rank's GPU" becomes a concrete device
        the concentration check and the mixture's device comparison can reason
        about. An emitted device is concrete already and is left as measured.

        Warns
        -----
        UserWarning
            If a multi-rank world resolves an indexed accelerator that is not
            the device every rank trains on.
        """
        if config.replay_device is not None:
            device = torch.device(config.replay_device)
            if device.type == "cuda" and device.index is None:
                device = torch.device("cuda", torch.cuda.current_device())
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

    def _rank_seed_offset(self) -> int:
        """Return the offset moving this rank's seeded streams off its neighbors'.

        Both seeded streams add a counter to their base seed, so ranks sit a
        whole stride apart rather than one, keyed on the global rank because
        node-local ranks repeat across nodes.
        """
        return get_rank(self.distributed_manager) * _RANK_SEED_STRIDE

    def _resume_or_seed(
        self, config: OnPolicyConfig, buffer: ReplayBuffer
    ) -> tuple[Batch | None, int | None]:
        """Return the batch to propagate, resuming a checkpointed run when there is one.

        A restored checkpoint carries the trajectory the interrupted run had
        reached, the propagator's cumulative step count, the initial
        structures' cursor, and the frames already in its replay buffer, so
        the resumed run continues the same trajectory rather than starting a
        fresh one. The continuation is exact for a propagator whose whole
        state is the batch and that counter — the built-in integrators, whose
        Langevin noise is keyed on the step count; a relaxation optimizer's
        adaptive history lives outside the batch and restarts from its
        constructor arguments, so only the positions continue.

        The bundle's frames *are* the replay buffer as of the checkpoint, so
        they replace what the buffer holds rather than being appended. The
        bundle describes one rank's run, since
        :class:`~nvalchemi.training.hooks.CheckpointHook` writes on rank zero
        alone, so it is consumed only when that rank is the whole world at
        both ends of the restart and dropped with a warning otherwise, as is
        one whose cursor this rank's shard cannot take. A bundle written after
        generation ran dry carries the frames and the exhaustion rather than a
        trajectory, so the resumed run keeps training on the buffer.

        Returns
        -------
        tuple[Batch | None, int | None]
            The batch the next segment propagates from — ``None`` once
            generation is exhausted — and the step the interrupted run last
            labeled, which the segment loop hands to the labeling hook it
            rebuilds. The step is ``None`` when the run seeds, leaving a fresh
            hook's cadence untouched.
        """
        restored = self._take_restart_state()
        if restored is None:
            return config.initial_structures.initial_batch(), None
        reason = self._rank_local_restart_reason(restored)
        if reason is None:
            reason = self._restore_structure_cursor(config, restored)
        if reason is not None:
            warnings.warn(
                f"The on-policy restart bundle is dropped: {reason} It holds "
                "one rank's trajectory and one rank's replay frames, and "
                "replaying those onto every rank would have every rank "
                "propagate rank zero's structures and train on rank zero's "
                "frames. This rank seeds afresh from its own share of the "
                "initial structures with a cold replay buffer instead, so "
                "budget the first segments after the restart for refilling "
                "it: until they do, the mixture is drawn from the reference "
                "dataset alone.",
                UserWarning,
                stacklevel=2,
            )
            return config.initial_structures.initial_batch(), None
        config.dynamics.step_count = int(restored["dynamics_step_count"])
        self._warn_settings_drift(config, restored)
        frames = restored.get("replay_frames")
        if frames is not None:
            buffer.clear()
            buffer.extend(_batch_from_state(frames))
        labeled_step = max(config.dynamics.step_count - 1, 0)
        if restored.get("generation_exhausted", False):
            self._generation_exhausted = True
            return None, labeled_step
        return _batch_from_state(restored["trajectory"]), labeled_step

    def _restore_structure_cursor(
        self, config: OnPolicyConfig, restored: Mapping[str, Any]
    ) -> str | None:
        """Resume the initial structures at the bundle's cursor, or say why not.

        Parameters
        ----------
        config : OnPolicyConfig
            Segment loop whose initial structures are resumed, already
            narrowed to this rank's shard.
        restored : Mapping[str, Any]
            Bundle a restored checkpoint carried.

        Returns
        -------
        str | None
            A sentence naming a cursor this rank's shard cannot take, or
            ``None`` when the cursor was resumed.
        """
        cursor = restored["initial_structures"]
        try:
            config.initial_structures.load_state_dict(cursor)
        except ValueError:
            return (
                f"its cursor was written for rank {cursor['rank']!r} of "
                f"{cursor['world_size']!r}, and this rank's shard counts "
                "positions in a different set of rows."
            )
        return None

    @staticmethod
    def _warn_settings_drift(
        config: OnPolicyConfig, restored: Mapping[str, Any]
    ) -> None:
        """Report the settings the resumed loop sets differently from the interrupted one.

        Parameters
        ----------
        config : OnPolicyConfig
            Segment loop the run resumes under.
        restored : Mapping[str, Any]
            Bundle a restored checkpoint carried.

        Warns
        -----
        UserWarning
            If a setting the bundle recorded differs from the one in hand.
        """
        recorded = restored["settings"]
        current = config.settings.model_dump(mode="json")
        drifted = sorted(
            name
            for name, value in current.items()
            if name in recorded and recorded[name] != value
        )
        if not drifted:
            return
        now = {name: current[name] for name in drifted}
        then = {name: recorded[name] for name in drifted}
        warnings.warn(
            f"The resumed segment loop sets {drifted!r} differently from the "
            "run the restart bundle was written by, so the restored "
            "trajectory, replay frames, and cursor were produced under "
            f"settings the rest of this run will not use; got {now!r} against "
            f"{then!r}. Restore the settings to compare the halves, or start a "
            "fresh run to change them.",
            UserWarning,
            stacklevel=2,
        )

    def _rank_local_restart_reason(self, restored: Mapping[str, Any]) -> str | None:
        """Return why a rank-zero-only restart bundle cannot be consumed, or ``None``.

        Two worlds have to agree for the bundle to describe the whole run: the
        one it was written in, which the shard its cursor records names, and
        the one it is restored into.

        Parameters
        ----------
        restored : Mapping[str, Any]
            Bundle a restored checkpoint carried.

        Returns
        -------
        str | None
            A sentence naming the mismatch, or ``None`` when a single rank
            wrote the bundle and a single rank is restoring it.
        """
        world_size = get_world_size(self.distributed_manager)
        if world_size > 1:
            return (
                f"the segment loop is resuming on world_size={world_size!r}, "
                "and the checkpoint it rides in is written by rank zero alone."
            )
        saved_world_size = int(restored["initial_structures"]["world_size"])
        if saved_world_size > 1:
            return (
                f"it was written on world_size={saved_world_size!r} and is "
                "being restored on one rank, which would silently continue "
                "rank zero's trajectories and discard the rest."
            )
        return None

    def _take_restart_state(self) -> dict[str, Any] | None:
        """Return the on-policy bundle a restored checkpoint carried, once."""
        for hook in self.hooks:
            if isinstance(hook, _OnPolicyRestartHook):
                return hook.take()
        return None

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
        :meth:`from_spec_dict` builds. An on-policy run serializes as
        references: ``on_policy`` becomes the recipe
        :meth:`~nvalchemi.training.distillation.OnPolicyConfig.to_spec_dict`
        produces and ``reference_dataset`` the store it reads, so the rebuilt
        strategy needs only its models supplied. A piece no recipe describes —
        a dataset holding its samples in memory, a propagator hiding its
        constructor arguments — leaves the whole ``on_policy`` entry out with a
        warning naming it, rather than writing a recipe that would rebuild into
        a different run; such a strategy is offline-shaped unless the two are
        passed back to :meth:`from_spec_dict`, :meth:`from_checkpoint_dict`, or
        :meth:`load_checkpoint`.

        Returns
        -------
        dict[str, Any]
            JSON-ready bundle suitable for :func:`json.dumps`.

        Warns
        -----
        UserWarning
            If ``on_policy`` or ``reference_dataset`` holds something no recipe
            can describe, naming what it was.
        """
        spec = super().to_spec_dict()
        spec["strategy_cls"] = f"{type(self).__module__}.{type(self).__qualname__}"
        spec["teacher_signals"] = (
            None if self.teacher_signals is None else sorted(self.teacher_signals)
        )
        spec["label_missing"] = self.label_missing
        if self.on_policy is None:
            return spec
        try:
            spec["on_policy"] = self.on_policy.to_spec_dict(
                teacher=self.models["teacher"]
            )
            spec["reference_dataset"] = (
                None
                if self.reference_dataset is None
                else _dataset_spec_dict(
                    self.reference_dataset, "DistillationStrategy.reference_dataset"
                )
            )
        except ValueError as exc:
            warnings.warn(
                f"The on-policy recipe is omitted from the spec: {exc} A "
                "strategy rebuilt from this spec runs offline over the "
                "dataloader passed to run().",
                UserWarning,
                stacklevel=2,
            )
            spec.pop("on_policy", None)
        return spec

    @classmethod
    def from_spec_dict(
        cls,
        spec: Mapping[str, Any],
        *,
        models: strategy_validation.ModelInput | None = None,
        hooks: Sequence[Any] | None = None,
        training_fn: Any = None,
        on_policy: OnPolicyConfig | None = None,
        reference_dataset: BatchDatasetProtocol | None = None,
    ) -> DistillationStrategy:
        """Rebuild a :class:`DistillationStrategy` from ``to_spec_dict`` output.

        A ``strategy_cls`` naming a subclass dispatches to that class's own
        ``from_spec_dict`` with the spec and every runtime override, so the
        strategy a spec names is the one that runs; a subclass adding a runtime
        keyword must widen this call with it.

        A spec carrying an ``on_policy`` recipe rebuilds the segment loop around
        the supplied ``models``: the propagator around ``models['student']``,
        the scorer around ``models['teacher']``, and the initial structures and
        the reference dataset from the stores they name, opened on the spec's
        primary device rather than the one the recipe recorded, so a checkpoint
        restored under another ``map_location`` reads its data where the run
        now trains. ``on_policy`` and
        ``reference_dataset`` resolve in a fixed order — an explicit keyword
        here, then whatever :meth:`load_checkpoint` or
        :meth:`from_checkpoint_dict` offered over
        :func:`_supplied_runtime_objects`, then the spec — and a dispatched
        subclass reads the same offer.

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
        on_policy : OnPolicyConfig | None, optional
            Segment loop to use instead of the spec's recipe. Default ``None``
            (rebuild the recipe, when the spec carries one).
        reference_dataset : BatchDatasetProtocol | None, optional
            Reference dataset to use instead of the one the recipe names. Default
            ``None``.

        Returns
        -------
        DistillationStrategy
            A freshly validated strategy of the class *spec* names, ready to
            :meth:`run`.

        Raises
        ------
        ValueError
            If *spec* is missing a required key, if its ``strategy_cls`` entry
            is not a dotted class path string, if that path cannot be imported,
            or if it resolves to a class that is not a
            :class:`DistillationStrategy` subclass.

        Notes
        -----
        A recipe is the weakest source because it is the only one that cannot
        be complete: it names its initial-structure store by path and its
        propagator by constructor arguments. A run started from an
        :class:`~nvalchemi.data.datapipes.in_memory_dataset.InMemoryDataset`,
        or whose propagator carries hooks, therefore serializes without its
        ``on_policy`` block, and restoring it means passing *on_policy* here.
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
            try:
                imported = _import_cls(raw_strategy_cls)
            except (ImportError, AttributeError, TypeError) as exc:
                raise ValueError(
                    f"from_spec_dict: 'strategy_cls' {raw_strategy_cls!r} could "
                    f"not be imported: {exc}"
                ) from exc
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
                    on_policy=on_policy,
                    reference_dataset=reference_dataset,
                )
        supplied = _SUPPLIED_RUNTIME_OBJECTS.get()
        if on_policy is None:
            on_policy = supplied.get("on_policy")
        if reference_dataset is None:
            reference_dataset = supplied.get("reference_dataset")
        restored_models = supplied.get("models")
        model_input = strategy_spec._models_from_spec_and_overrides(
            spec.get("model_specs", {}),
            models if restored_models is None else restored_models,
            single_model_input=strategy_spec._single_model_input_from_spec(
                spec.get("single_model_input")
            ),
        )
        devices = strategy_spec._devices_from_spec(spec["devices"])
        recipe = spec.get("on_policy")
        rebuildable = isinstance(model_input, Mapping) and _REQUIRED_MODELS <= set(
            model_input
        )
        if on_policy is None and recipe is not None and rebuildable:
            on_policy = OnPolicyConfig.from_spec_dict(
                recipe,
                student=model_input["student"],
                teacher=model_input["teacher"],
                device=devices[0],
            )
        reference_spec = spec.get("reference_dataset")
        if reference_dataset is None and reference_spec is not None:
            reference_dataset = _dataset_from_spec_dict(
                {**reference_spec, "device": str(devices[0])}
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
            devices=devices,
            teacher_signals=spec.get("teacher_signals"),
            label_missing=spec.get("label_missing", True),
            on_policy=on_policy,
            reference_dataset=reference_dataset,
        )

    @classmethod
    def from_checkpoint_dict(
        cls,
        spec: Mapping[str, Any],
        *,
        models: strategy_validation.ModelInput | None = None,
        hooks: Sequence[Any] | None = None,
        training_fn: Any = None,
        on_policy: OnPolicyConfig | None = None,
        reference_dataset: BatchDatasetProtocol | None = None,
    ) -> DistillationStrategy:
        """Rebuild a strategy from checkpoint metadata, the segment loop included.

        :meth:`~nvalchemi.training.TrainingStrategy.from_checkpoint_dict`, with
        the two runtime objects :meth:`to_spec_dict` cannot carry threaded
        through to :meth:`from_spec_dict`.

        Parameters
        ----------
        spec : Mapping[str, Any]
            A dict produced by :meth:`to_checkpoint_dict`.
        models : BaseModelMixin | dict[str, BaseModelMixin] | None, optional
            Runtime model override(s), normally the models loaded from the
            checkpoint weight files.
        hooks : Sequence[Any] | None, optional
            Runtime hooks appended by the caller.
        training_fn : Any, optional
            Runtime callable or dotted-path override.
        on_policy : OnPolicyConfig | None, optional
            Segment loop to rebuild the run with, around the supplied student.
            Default ``None``, which is an offline-shaped rebuild.
        reference_dataset : BatchDatasetProtocol | None, optional
            Reference dataset the segment loop mixes into every batch. Default
            ``None``.

        Returns
        -------
        DistillationStrategy
            A strategy with declarative fields and restart counters restored.
        """
        with _supplied_runtime_objects(
            on_policy=on_policy, reference_dataset=reference_dataset
        ):
            return super().from_checkpoint_dict(
                spec, models=models, hooks=hooks, training_fn=training_fn
            )

    @classmethod
    def load_checkpoint(
        cls,
        root_folder: Path | str,
        checkpoint_index: int = -1,
        map_location: str | torch.device | None = None,
        *,
        models: strategy_validation.ModelInput | None = None,
        hooks: Sequence[Any] | None = None,
        training_fn: Any = None,
        validators: Sequence[Any] | None = None,
        on_policy: OnPolicyConfig | None = None,
        reference_dataset: BatchDatasetProtocol | None = None,
    ) -> DistillationStrategy:
        """Load a restartable checkpoint, re-supplying what the spec omits.

        :meth:`~nvalchemi.training.TrainingStrategy.load_checkpoint`, extended
        with the runtime objects a distillation spec cannot describe. The
        segment loop travels with the student it propagates, so *models* is
        re-supplied alongside *on_policy* and the checkpoint's weights are
        loaded into them; a Boltzmann term refuses to rebuild without the loop.

        Parameters
        ----------
        root_folder : Path | str
            Root directory containing checkpoint files.
        checkpoint_index : int, optional
            Checkpoint index to load. ``-1`` loads the latest manifest index.
        map_location : str | torch.device | None, optional
            Device override passed through to :func:`torch.load` and the
            restored strategy metadata.
        models : BaseModelMixin | dict[str, BaseModelMixin] | None, optional
            Models to restore the checkpoint's weights into, overriding the
            ones the loader builds from the saved specs. Default ``None``.
        hooks : Sequence[Any] | None, optional
            Runtime hooks to attach to the restored strategy.
        training_fn : Any, optional
            Runtime training function override.
        validators : Sequence[Any] | None, optional
            Loaded-checkpoint validators forwarded to the lower-level loader.
        on_policy : OnPolicyConfig | None, optional
            Segment loop to restore the run with. Default ``None``, which is an
            offline-shaped restore.
        reference_dataset : BatchDatasetProtocol | None, optional
            Reference dataset the segment loop mixes into every batch. Default
            ``None``.

        Returns
        -------
        DistillationStrategy
            Restored strategy with model, optimizer, scheduler, and runtime
            counters loaded.
        """
        with _supplied_runtime_objects(
            models=models, on_policy=on_policy, reference_dataset=reference_dataset
        ):
            return super().load_checkpoint(
                root_folder,
                checkpoint_index,
                map_location,
                hooks=hooks,
                training_fn=training_fn,
                validators=validators,
            )
