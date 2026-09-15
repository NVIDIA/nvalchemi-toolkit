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
from collections.abc import Collection, Iterable, Iterator, Mapping, Sequence
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
from nvalchemi.training.distillation._labels import (
    _TEACHER_FIELD_PREFIX,
    _attach_teacher_labels,
    _reject_foreign_fields,
)
from nvalchemi.training.distillation._restart import (
    _batch_from_state,
    _OnPolicyRestartHook,
)
from nvalchemi.training.distillation.config import OnPolicyConfig
from nvalchemi.training.distillation.hooks import (
    TeacherLabelHook,
    _ConvergedFrameHook,
    _run_local_keys,
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
    SUPPORTED_SIGNALS,
    InProcessTeacherScorer,
    _isolated_embeddings,
    _node_embedding_shapes,
    _restore_grad_flags,
    _snapshot_grad_flags,
    hessian_vector_product,
    scorer_fields,
    signal_fields,
    signal_for_field,
)
from nvalchemi.training.distillation.seeding import (
    SeedSource,
    _check_seed_status,
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
    signature has no room for the live objects a spec cannot describe, and that
    is the only call standing between a caller of
    :meth:`DistillationStrategy.load_checkpoint` and the constructor that needs
    them. They travel over this variable instead of through it, which is what
    keeps the parent's loader reusable rather than reimplemented here.

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

    A representation is not a forward-pass output: it comes from
    :meth:`~nvalchemi.models.base.BaseModelMixin.compute_embeddings`, a second
    pass over the batch, which is why
    :class:`~nvalchemi.training.distillation.EmbeddingMatchingLoss` needs this
    training function rather than the stock one. The embeddings are read back
    off the batch and the batch is left as it was found, so nothing downstream
    sees a field that only this objective wants.

    A ``"projector"`` model, when the strategy has one, is applied to the
    student's embeddings on the way out — never to the teacher's, which are
    fixed targets. It is an ordinary named model, so its parameters are trained
    by its own ``optimizer_configs`` entry.

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
        If the student's ``compute_embeddings`` writes no ``node_embeddings``.

    See Also
    --------
    nvalchemi.training.distillation.EmbeddingProjector : The width adapter.

    Notes
    -----
    The student is run twice per batch — once for its outputs and once for its
    embeddings — because the model contract exposes no way to get both from one
    pass. That doubles the student's share of a training step, which is the
    price of the objective and worth measuring before scaling a run up.

    ``compute_embeddings`` is not part of the
    :class:`~torch.nn.Module` interface a
    :class:`~torch.nn.parallel.DistributedDataParallel` replica proxies, so
    under a :class:`~nvalchemi.training.hooks.DDPHook` the embedding pass is
    taken on the module the hook wrapped. Its gradients are still reduced,
    since both passes accumulate into the same parameters, but they reach the
    reducer outside the replica's own forward: a student submodule exercised
    *only* by ``compute_embeddings`` is invisible to
    ``find_unused_parameters=True``, which is the one configuration to avoid.
    The projector is applied through its replica's ``__call__`` and needs no
    such care.
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
    if _PROJECTOR_MODEL in models:
        embeddings = models[_PROJECTOR_MODEL](embeddings)
    predictions["predicted_node_embeddings"] = embeddings
    return predictions


def hessian_distillation_fn(
    models: Mapping[str, BaseModelMixin], batch: Batch
) -> dict[str, torch.Tensor]:
    """Run the student forward pass and add its Hessian-vector product.

    The product is taken along ``teacher_hvp_probe``, the probe direction the
    teacher's own product was labeled with, so the two are comparable. It comes
    from a dedicated second pass narrowed to the student's energy, which is
    what :meth:`~nvalchemi.training.distillation.InProcessTeacherScorer.label_hvp`
    already does on the teacher side and for the same reason: a conservative
    model derives its forces from the very graph the second derivative needs,
    and frees that graph whenever it is not in training mode, so the stock
    forward's energy cannot be differentiated again. The narrowed pass computes
    no forces, and takes both of its derivatives with ``create_graph=True``,
    which is what
    :class:`~nvalchemi.training.distillation.HessianMatchingLoss` backpropagates
    into the student's parameters through.

    The batch is left exactly as it was found: whatever ``requires_grad`` flags
    the two passes enable are restored, and the narrowed pass reuses the
    neighbor list the stock forward just ran on rather than rebuilding one.

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
    The student is run twice per batch — once for its outputs and once, energy
    only, for the product — and the second pass adds two backward passes, one
    of them through a second-order graph whose memory is held for the whole
    step. The teacher paid the same on the labeling side once; the student pays
    it every time the frame is trained on. A stochastic student draws afresh in
    the narrowed pass, so its curvature is measured on a different realization
    than its energy.
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


def _derived_teacher_signals(
    loss_fn: ComposedLossFunction, *, supplied_fields: Collection[str] = ()
) -> frozenset[str]:
    """Return the teacher signals the loss composition's targets require.

    A ``teacher_*`` target listed in *supplied_fields* is skipped rather than
    resolved or refused: it is a generation-supplied target, written onto every
    captured frame by the on-policy propagator's own scorer, so no signal of
    the strategy's scorer stands behind it.

    Parameters
    ----------
    loss_fn : ComposedLossFunction
        Loss composition whose target keys are read.
    supplied_fields : Collection[str], optional
        Batch fields the on-policy propagator's scorer declares it writes.
        Default ``()``, which is offline distillation, where nothing but a
        built-in signal populates the namespace.

    Returns
    -------
    frozenset[str]
        Signal names the strategy's own scorer has to produce.

    Raises
    ------
    ValueError
        If a ``teacher_*`` target maps to no built-in signal and no scorer
        declares it, or names a companion field rather than the one its signal
        produces.
    """
    signals: set[str] = set()
    for key in loss_target_keys(loss_fn):
        if not key.startswith(_TEACHER_FIELD_PREFIX):
            continue
        signal = signal_for_field(key)
        if signal is None:
            if key in supplied_fields:
                continue
            raise ValueError(
                "Loss targets must name a supported teacher target from "
                f"{list(signal_fields(SUPPORTED_SIGNALS))!r}; got {key!r}. The "
                f"{_TEACHER_FIELD_PREFIX!r} prefix is reserved for those signals, so "
                "a field a custom scorer writes must be named outside it to reach "
                "the loss as an ordinary batch field — unless an on-policy "
                "propagator's scorer declares it in label_fields, which makes it a "
                "generation-supplied target every captured frame carries."
            )
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

    The mirror of
    :func:`~nvalchemi.training.runtime.train_configured_models`, which only
    ever sets training mode and restores the mode it found. A model that is
    never told otherwise therefore runs in training mode outside a training
    phase — with dropout live, batch-norm statistics moving, and a
    conservative model's forces building a second-order graph — which is what
    the on-policy loop's generation phase has to avoid.

    Parameters
    ----------
    models : Mapping[str, torch.nn.Module]
        Named models participating in the run.
    optimizer_configs : Mapping[str, object]
        Optimizer configuration keyed by model name. Models present in it are
        switched to evaluation mode while the context is active.

    Yields
    ------
    None
        Control while the configured models are in evaluation mode.
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

    :func:`_eval_configured_models` reaches the named models an optimizer
    updates, which a composition holding the student is not: it is no entry of
    ``models``, so nothing else ever takes it out of training mode. Left there,
    a shared-autograd composition differentiates its summed energy with
    ``create_graph=True`` — the second-order graph the generation phase exists
    to avoid — and every submodule the student does not own keeps moving its
    batch-norm statistics on generated frames. Enter this context *inside*
    :func:`_eval_configured_models`: restoring a composition's mode sets the
    mode of every module it holds, the student included, so it has to happen
    before the student's own mode is put back.

    Every submodule's own mode is snapshotted, not just the composition root's.
    :meth:`~torch.nn.Module.train` stamps one flag recursively, so restoring
    the root alone would hand back a frozen correction head — one the caller
    had put in evaluation mode individually, which a non-teacher entry of
    ``models`` cannot be because every one of those needs an optimizer config —
    in training mode, silently running its dropout afterwards.

    Parameters
    ----------
    propagator_model : object
        Model the propagator holds. A propagator holding *student* itself, or
        anything that is not a :class:`torch.nn.Module`, is left alone.
    student : BaseModelMixin
        Student the strategy trains, whose own mode
        :func:`_eval_configured_models` owns.

    Yields
    ------
    None
        Control while the composing model is in evaluation mode.
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
    sampler: SeedSource


def _competing_migrators(
    dynamics: BaseDynamics, criterion: ConvergenceHook
) -> list[ConvergenceHook]:
    """Return the status migrators already on *dynamics* that are not *criterion*.

    Both places a propagator can hold one are searched: its registered hooks,
    where a migrating :class:`~nvalchemi.dynamics.base.ConvergenceHook` fires
    every step, and its ``convergence_hook``, which the lifecycle is about to
    replace and whose migration would otherwise be dropped without a word.

    A :class:`~nvalchemi.dynamics.FusedStage` is searched sub-stage by
    sub-stage as well, because that is where its own migrators live:
    constructing one registers a migrating hook on every non-last sub-stage
    unconditionally, and on the last one whenever it declares a
    ``convergence_hook`` of its own. Those hooks fire ahead of the fused-level
    ones, so a scan of the fused propagator alone reports a clean propagator
    while the sub-stage migrator graduates the batch first. A sub-stage
    ``convergence_hook`` that only detects convergence is not a competitor
    itself — the migrator ``FusedStage`` derives from it is, and it is found
    among that sub-stage's hooks. The hooks registered at the fused level
    through ``register_fused_hook`` fire on the whole batch right behind the
    fused propagator's own, so they are read alongside them.

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
        for hook in (
            *propagator.hooks,
            *getattr(propagator, "fused_hooks", ()),
            propagator.convergence_hook,
        )
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

    The config's :attr:`~OnPolicyConfig.convergence_criterion` is put on the
    propagator twice, deliberately. As a registered ``AFTER_STEP`` hook it
    migrates the status of converged graphs, which is what freezes them in the
    propagator's step, what the capture hook behind it stores them on, and what
    :meth:`~nvalchemi.dynamics.base.BaseDynamics.refill_check` graduates them
    on; as the propagator's ``convergence_hook`` it is the detector ending a
    chunk early once every graph has converged. One criterion drives both,
    rather than a run whose graduation and detection disagree — a criterion the
    propagator was built with is restored on the way out, and so is ``done``,
    which :meth:`~nvalchemi.dynamics.base.BaseDynamics.refill_check` raises off
    the temporary refill sampler this context owns and would otherwise leave on
    a propagator the caller means to reuse.

    That is only true while it is the *sole* migrator, so a propagator already
    carrying one is refused rather than run: a looser criterion of its own
    graduates a structure before the configured one accepts it, which freezes
    it out of the path capture and leaves the converged route nothing to store,
    so the trajectory ends in neither. The criterion also has to migrate off
    the status the seed source stamped, or nothing ever freezes and nothing
    ever graduates while the run reports itself configured.

    The lifecycle is likewise the run's sole refill source, so a propagator
    carrying a sampler of its own is refused too. That sampler makes
    :meth:`~nvalchemi.dynamics.base.BaseDynamics.run` refill on its own
    cadence, mid segment, and the compaction that follows a graduation moves
    the survivors under the capture hook's positional bookkeeping, which then
    reads the wrong rows and stores neither the minima it is holding nor the
    ones still to come.

    Parameters
    ----------
    config : OnPolicyConfig
        Segment-loop configuration, holding the criterion.
    state : Batch
        Seed batch, already carrying the bookkeeping
        :meth:`~nvalchemi.training.distillation.SeedSource.initial_batch`
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
        off a status no seed carries.
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
            "neither capture route. Remove it, or drop convergence and let the "
            "propagator manage its own lifecycle. On a FusedStage the migrator "
            "is one the stage built for a sub-stage rather than one the caller "
            "registered: every non-last sub-stage carries one, and the last "
            "one does whenever it was given a convergence_hook, so only a "
            "single sub-stage without its own criterion is free of them."
        )
    if dynamics.sampler is not None:
        raise ValueError(
            "The relaxation lifecycle owns the refill as well as graduation, "
            "so the propagator must carry no sampler of its own; got "
            f"{type(dynamics.sampler).__name__!r}. A propagator that refills "
            "inside run compacts the survivors to the front of the batch mid "
            "segment, which leaves the capture hook's positional bookkeeping "
            "pointing at the wrong structures and drops the minima it was "
            "meant to store. Give OnPolicyConfig.seeds the same budget, which "
            "backfills from the same dataset at the segment boundary, and "
            "leave the propagator's own unset."
        )
    _check_seed_status(state, criterion)
    capture = _ConvergedFrameHook(sink=HostMemory(capacity=state.num_graphs))
    detector = dynamics.convergence_hook
    was_done = dynamics.done
    # Registered ahead of the capture and labeling hooks, so a graph that
    # converges on this step is graduated before either of them reads its
    # status and the two capture routes never store it twice.
    dynamics.register_hook(criterion)
    dynamics.register_hook(capture)
    dynamics.convergence_hook = criterion
    try:
        yield _RelaxationLifecycle(capture=capture, sampler=config.seeds)
    finally:
        dynamics.convergence_hook = detector
        dynamics.done = was_done
        dynamics.hooks.remove(criterion)
        dynamics.hooks.remove(capture)


def _movable_seed(node: BaseDynamics) -> tuple[BaseDynamics, str, int] | None:
    """Return the first integer seed of *node* a rank offset can write back.

    Each name is probed by writing back the value it just read. A getter-only
    ``random_seed`` property forwarding the private field — the natural next
    step for the built-ins, and the spelling ``_PROPAGATOR_SEED_ATTRS`` puts
    first — reads as an integer and cannot be assigned, so recording it would
    raise where the offsets are applied: on every rank but rank zero, which
    would run its whole generation segment and then wait at the first
    all-reduce for ranks that have already died. A name that cannot be written
    falls through to the next one instead.
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

    Accounting is per node rather than per tree, because a composition mixing
    the two is the case that reads as working: one seeded sub-stage is enough
    to make the walk look successful while the stages beside it draw the same
    numbers on every rank.

    The second list is deliberately narrow. A node is reported as left behind
    only when it holds a :class:`torch.Generator`, which is randomness the walk
    can see and cannot offset. A node exposing neither an integer seed nor a
    generator is passed over in silence, because nothing tells a stage hiding
    its randomness from a deterministic one — a fused stage, a minimizer, a
    velocity Verlet integrator — and naming those would bury the real report
    exactly where compositions are deep.
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

    Sharding the seed structures already gives every rank its own initial
    conditions, but a counter-based thermostat draws its noise from
    ``seed + step_count`` and the atom index alone, so ranks stepping in lockstep
    would otherwise apply the *same* random kicks to their different structures —
    and byte-identical kicks to structures that are replicas of one geometry,
    which is how a run asks for one trajectory per rank. The offset is a whole
    stride of the seed space per rank, which keeps the streams apart for as many
    propagator steps as the stride is wide.

    The whole composition is moved, not just its root. A relax-then-sample
    propagator built as ``FIRE(...) + NVTLangevin(...)`` exposes no seed of its
    own: the thermostat drawing the noise sits in a sub-stage, so probing the
    root alone would leave every rank on one stream and silently claim
    otherwise.

    Parameters
    ----------
    dynamics : BaseDynamics
        Propagator whose seed is offset, along with every propagator it
        composes. Each is probed for a writable integer under the names in
        ``_PROPAGATOR_SEED_ATTRS`` and restored on the way out; one holding its
        randomness anywhere else — a differently named attribute, a
        :class:`torch.Generator` — is left alone here, and named before the
        first segment by
        ``DistillationStrategy._warn_shared_propagator_streams``, which is
        where the world size is known and rank zero is listening.
    offset : int
        Amount added to every seed found, for the duration of the context. A
        zero offset — rank zero, and every single-process run — leaves the
        propagator untouched.

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

    The first floating-point parameter decides, but never below single
    precision: a ``bfloat16``, ``float16``, or narrower student gets
    ``float32`` labels, while ``float32`` and ``float64`` are kept as they are.
    Two things make reduced precision the wrong label dtype. A store round-trips
    every floating field to the dtype of the dataset's ``positions``, which is
    float32 for essentially every dataset, so a label below it would disagree
    with what :func:`~nvalchemi.training.distillation.label_dataset` persisted;
    and the graph-balanced reductions the loss terms use accumulate in the
    residual's dtype, where a ``bfloat16`` sum saturates at 256. A student that
    exposes no parameters at all gets ``None``, which leaves labels in the
    teacher's own dtype.
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


def _to_device(batch: Batch, device: torch.device) -> Batch:
    """Return *batch* on *device*, overlapping the copy only into device memory.

    A copy into device memory is queued asynchronously so it overlaps the work
    already on the stream, and stream ordering keeps every consumer behind it.
    A copy into host memory has no such ordering: ATen issues the transfer and
    returns without synchronizing, so a read that follows the call can observe
    a destination the transfer has not filled. That race is not tolerable here
    because the moved batch's index tensors are read on the host immediately —
    ``segment_lengths`` feeds the ``repeat_interleave`` behind ``batch_idx``,
    and ``batch_ptr`` slices the per-graph rows — where a half-written buffer
    surfaces as negative repeats, out-of-range indices, or a hang rather than
    as a wrong number.
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

    ``DistillationStrategy`` is a :class:`~nvalchemi.training.TrainingStrategy`
    whose named models are ``"student"`` and ``"teacher"``. The teacher is
    frozen by omission — it must not appear in ``optimizer_configs``, which is
    what puts it in eval mode with gradients disabled for the duration of
    :meth:`run` — while the student, and any auxiliary model such as a
    projection head, must be configured with an optimizer.

    Teacher knowledge reaches the loss as ordinary batch fields. Every signal
    an :class:`~nvalchemi.training.distillation.InProcessTeacherScorer`
    produces populates one ``teacher_*`` field, and a loss term consumes it by
    pointing its ``target_key`` there: ``EnergyMSELoss(target_key="teacher_energy")``,
    ``ForceMSELoss(target_key="teacher_forces")``,
    :class:`~nvalchemi.training.distillation.PerAtomEnergyMatchingLoss` for
    ``teacher_node_energies``. Mixing teacher targets with reference targets in
    one objective is therefore ordinary loss composition, and so is annealing
    between them with a
    :class:`~nvalchemi.training.losses.base.LossWeightSchedule` — offline,
    where every sample carries its own reference labels. An on-policy run
    cannot: generated frames have no reference labels, and both mixture sources
    are required to carry the same fields, so its anchor has to be
    teacher-labeled rather than reference-labeled.

    Those targets also decide what the teacher is asked for.
    ``teacher_signals=None`` (the default) derives the signal set from the
    ``teacher_*`` targets the loss reads — the validation loss's included,
    whenever ``validation_config`` carries its own ``loss_fn`` — so objective
    and teacher cannot drift apart; an explicit set must cover the derived one
    and may add more. The resolved set is checked against the teacher's declared
    outputs at construction, as is the model/optimizer contract above and — for
    the stock ``training_fn`` — every loss component's prediction key against
    the outputs the student actually computes, which is its ``active_outputs``
    intersected with its declared ``outputs``, so a misconfigured run fails
    before it starts rather than on its first batch. The validation loss's
    prediction keys go through the same check whenever the effective validation
    function, ``validation_config.validation_fn`` falling back to
    ``training_fn``, is the stock one. None of this re-runs on assignment, so a
    ``validation_config`` attached after construction keeps the signals already
    resolved: pass it to the constructor, or name the wider set in
    ``teacher_signals``.

    Every resolved signal is a request for its fields on every batch rather
    than a permission to carry them, whether it was derived or named in
    ``teacher_signals``. A batch counts as labeled only when it holds every
    resolved field, so adding a validation loss with a new ``teacher_*`` target
    puts a training store written before it back on the teacher batch after
    batch — the same values, at the price of a forward pass each time. A store
    meant to train with no teacher pass at all has to be labeled with the same
    signal set the strategy resolves.

    In offline distillation the labels travel with the sample. The intended
    path is :func:`~nvalchemi.training.distillation.label_dataset`: score the
    dataset once, persist the teacher fields into a Zarr store, and train from
    that store with no teacher forward pass at all. A batch that arrives
    without the required fields is labeled on the fly instead — training and
    validation alike — which keeps short runs and interactive sessions working
    without a labeling pass. ``label_missing=False`` turns that off and leaves
    an unlabeled batch to surface as a missing loss target. Either way
    ``training_fn`` stays a plain student forward —
    :func:`default_distillation_fn` unless the caller supplies one — so the
    recipe survives :meth:`to_spec_dict` and the teacher never enters the
    student's autograd graph.

    Setting ``on_policy`` switches :meth:`run` to the segment loop instead:
    the student's own propagator generates frames, the teacher labels them,
    they accumulate in a replay buffer, and each segment trains on a mixture of
    that buffer and ``reference_dataset`` at the configured ``replay_ratio``.
    Because the propagator holds the very module the optimizer updates, every
    segment generates from a fresher policy than the last — which is what makes
    the data on-policy, and why the propagator's model is checked for object
    identity with ``models["student"]`` at construction. A relaxation
    propagator adds ``OnPolicyConfig.convergence``: converged structures are
    stored once, graduate out of the batch at the segment boundary, and are
    replaced by fresh seeds, so the buffer keeps filling with structures that
    are still moving.

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
        ``predicted_`` namespace under a stock ``training_fn``, if a loss reads
        a ``teacher_*`` target that maps to no known signal and that no
        on-policy propagator's scorer declares, if an explicit
        ``teacher_signals`` omits a signal a loss needs, if no teacher signal
        is requested at all, if the teacher cannot produce a requested signal,
        or if the teacher is a composition that plans more than one
        neighbor-list source. With an embedding objective on the stock training
        function, additionally if the student publishes no node-embedding shape
        or if the student, projector, and teacher widths do not compose; with a
        Hessian objective, if the student computes no energy; with a
        distribution objective, if the run is not on-policy or generates with a
        relaxation or converging propagator. In on-policy mode, additionally if
        the run is sized in epochs rather than steps, if the propagator holds
        neither the student nor a model composing it, if ``replay_ratio`` is
        ``0``, if a ratio below ``1`` is paired with no ``reference_dataset``,
        if a ratio of ``1`` is paired with one, if the ratio and ``batch_size``
        together allocate no samples to one mixture source, if
        ``reference_dataset`` emits on an accelerator the run does not train
        on, if ``replay_device`` names a device the ``reference_dataset`` does
        not emit on, if the ``reference_dataset`` carries fields the labeling
        hook strips from every generated frame, if the propagator's scorer
        declares a field outside the ``teacher_*`` namespace, or if that
        scorer's known fields and ``reference_dataset`` do not carry the same
        teacher fields.

    Examples
    --------
    Distill energies, forces, and the teacher's per-atom energy decomposition
    from a store written by :func:`label_dataset`:

    >>> import torch
    >>> from nvalchemi.training import EnergyMSELoss, ForceMSELoss, OptimizerConfig
    >>> from nvalchemi.training.distillation import (
    ...     DistillationStrategy,
    ...     PerAtomEnergyMatchingLoss,
    ... )
    >>> loss_fn = (
    ...     EnergyMSELoss(target_key="teacher_energy")
    ...     + ForceMSELoss(target_key="teacher_forces", normalize_by_atom_count=True)
    ...     + 0.1 * PerAtomEnergyMatchingLoss()
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
    Teacher conservativeness is deliberately not validated. A teacher that
    predicts forces with its own head rather than as the negative gradient of
    its energy is a first-class teacher here: the scorer detaches every signal
    it returns, so how the teacher produced a force never reaches the student.

    A composed teacher whose stages plan more than one neighbor-list source is
    refused at construction, since the scorer builds exactly one list per
    batch. Compose it to plan a single list instead —
    ``neighbor_adaptation="always"``, or a ``max_cutoff_ratio`` of at least
    the ratio of its largest to its smallest cutoff — and the scorer adapts
    that one list per batch.

    One seam does the labeling: an internal hook the strategy registers ahead
    of the caller's own, on ``BEFORE_FORWARD``, a stage both the training loop
    and the validation loop dispatch on the device-placed batch before its
    forward pass. Unlabeled validation data therefore needs no preparation — a
    ``validation_config`` with its own ``loss_fn`` has its ``teacher_*`` targets
    derived and its prediction keys checked alongside the training loss's — and
    a caller-supplied ``training_fn`` is covered too. Hooks are never
    serialized, so the seam is simply re-registered when :meth:`from_spec_dict`
    or :meth:`load_checkpoint` rebuilds the strategy, and a seam carried in the
    ``hooks`` such a rebuild is handed is replaced rather than kept, so chained
    rebuilds never accumulate one.

    Validating an EMA-averaged student against the live teacher is what
    ``ValidationConfig(use_ema="auto")`` does: the student's averaged weights
    replace its live ones, the teacher stays live, and the pass reports
    ``model_source="mixed"``. ``use_ema="always"`` currently also demands an
    inference-slot entry for the frozen teacher and fails at the first
    validation pass without one.

    On-the-fly labels are attached to the device-placed batch the strategy
    trains on, which is a copy of the one the caller handed over, so they do not
    persist on the caller's object. A loader that replays the same systems every
    epoch therefore costs one teacher pass per epoch, which is the other reason
    a long run should label its dataset offline first.

    The ``teacher_`` prefix is reserved for the built-in signals, so a loss
    target under it that names none of them is refused rather than left to fail
    as a missing batch field. A custom scorer's own field — anything
    :func:`~nvalchemi.training.distillation.label_dataset` persisted outside
    that signal set — reaches the loss as an ordinary batch field by being
    named outside the prefix, and is then invisible to signal derivation, which
    is what an explicit ``teacher_signals`` is for.

    On-policy runs relax that rule in exactly one way, the generation-supplied
    target: a ``teacher_*`` target naming no built-in signal is accepted when
    the propagator's scorer declares it in ``label_fields``, because that
    scorer writes it onto every frame the labeling hook captures and the frame
    carries it into the replay buffer. Such a field derives no signal — this
    strategy's own scorer produces built-in signals only — so at least one
    built-in ``teacher_*`` target, or an explicit ``teacher_signals``, is still
    required alongside it, ``reference_dataset`` has to carry it too (which the
    generation/anchor parity check enforces), and any validation data has to
    arrive already carrying it, since nothing labels it on the fly: a
    validation batch without it surfaces as the loss's missing-target
    ``KeyError``. A scorer declaring no ``label_fields`` supplies nothing, its
    fields being unknowable until it has scored a batch, so a custom target
    read against it is refused exactly as offline.

    Labeling runs with autocast disabled, so the teacher computes at its own
    precision no matter what precision context the surrounding training or
    validation step establishes, and an on-the-fly label matches the offline one
    bit for bit wherever the store returns the label dtype: a store round-trips
    every floating field to the dtype of the dataset's ``positions``, so over
    the usual float32 dataset every student but a float64 one sees identical
    labels on both paths, while a float64 student reads float32 back from the
    store and needs a ``dtype_policy`` to train from it.

    Labels are cast to the student's first floating-point parameter dtype, but
    never below single precision: a ``bfloat16`` or ``float16`` student gets
    float32 labels, because a store round-trips every floating field to the
    dtype of the dataset's ``positions`` and graph-balanced reductions
    accumulate in the residual's dtype. Such a student therefore needs
    ``dtype_policy="prediction_to_target"`` on its loss terms, which computes
    the loss in float32; a float64 teacher feeds a float32 student with no
    dtype policy at all. The cast is resolved at construction, so a student
    whose dtype changes afterwards needs a ``dtype_policy`` too.

    :class:`~nvalchemi.training.ComposedLossFunction` renormalizes weights by
    default, so the ``0.1`` above is a ratio rather than a coefficient: the
    three terms run at ``1/2.1``, ``1/2.1``, and
    ``0.1/2.1``. Pass ``normalize_weights=False`` for literal weights, which
    also stops a :class:`~nvalchemi.training.losses.base.LossWeightSchedule` on
    one term from rescaling the others as it ramps.

    The teacher is stored once per checkpoint root rather than at every index,
    so a periodic write costs the student's weights rather than the student's
    plus a frozen foundation teacher's; see :meth:`checkpoint_model_references`
    for what that means for a restart.

    ``on_policy`` and ``reference_dataset`` serialize as references too — the
    propagator's spec, the scorer's signal set over the strategy's own teacher,
    and the stores the datasets read — so a whole on-policy recipe survives
    :meth:`to_spec_dict` and rebuilds around re-supplied models. A run whose
    datasets live in memory, or whose propagator hides its constructor
    arguments, leaves the recipe out of the spec with a warning naming the
    piece. :meth:`from_spec_dict`, :meth:`from_checkpoint_dict`, and
    :meth:`load_checkpoint` all take ``on_policy`` and ``reference_dataset``
    keyword arguments, alongside the ``models`` the propagator was built
    around, and a live object handed over that way outranks whatever recipe the
    spec carries. That matters most for an objective that is *only* defined on
    generated batches — an ensemble term refuses to rebuild without the loop —
    and :meth:`~nvalchemi.training.TrainingStrategy.restore_checkpoint` into a
    strategy the caller constructed is the other way back. An interrupted
    on-policy run additionally carries its trajectory, the propagator's step
    count, and its replay frames through the checkpoint, so a resumed run
    continues the same trajectory instead of seeding a fresh one.
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
                "Teacher-labeled anchor dataset the on-policy mixture draws "
                "its ``1 - replay_ratio`` share from. Required whenever the "
                "ratio is below 1, and read only in on-policy mode."
            ),
        ),
    ] = None

    _scorer: InProcessTeacherScorer | None = PrivateAttr(default=None)
    _teacher_fields: tuple[str, ...] = PrivateAttr(default=())
    _replay_buffer: ReplayBuffer | None = PrivateAttr(default=None)
    _on_policy_state: Any = PrivateAttr(default=None)
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
    def seed_shard(self) -> tuple[int, ...]:
        """Seed rows this rank propagates its own trajectories from.

        The rows are read off the seed source once :meth:`run` has installed
        this rank's shard on it, and dealt here from the launcher's world
        before that, so the property answers the same question either side of
        a run.

        Returns
        -------
        tuple[int, ...]
            Indices into ``on_policy.seeds.dataset``, in dataset order. Empty
            for an offline strategy.

        See Also
        --------
        nvalchemi.training.distillation.SeedSource.shard :
            The deal itself, and the shard-local cursor it opens.
        """
        if self.on_policy is None:
            return ()
        seeds = self.on_policy.seeds
        rank = get_rank(self.distributed_manager)
        world_size = get_world_size(self.distributed_manager)
        installed = seeds.state_dict()
        if (installed["rank"], installed["world_size"]) == (rank, world_size):
            return seeds.rows
        return tuple(range(rank, len(seeds.dataset), world_size))

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
            cast_to=_student_label_dtype(self.models["student"]),
        )
        self._teacher_fields = signal_fields(signals)
        return self

    def _validate_student_outputs(self) -> None:
        """Check both losses' prediction keys against the student's effective outputs.

        A stock ``training_fn`` returns what the student's forward emits — which
        is ``active_outputs`` intersected with ``outputs`` rather than the
        declared set, so a student whose active set is narrowed, the common
        default for a pretrained wrapper, is caught here instead of on its first
        batch — plus whatever that function derives on top. A caller's own
        training function owns the contract itself and is left alone. A
        ``validation_config`` carrying its own ``loss_fn`` goes through the same
        check whenever its effective validation function — ``validation_fn``
        falling back to ``training_fn`` — is a stock one, since the validation
        loop reads the same predictions.
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
        # Pydantic populates every field before the first mode="after"
        # validator, so the propagator's scorer is already readable here.
        supplied = (
            ()
            if self.on_policy is None
            else scorer_fields(self.on_policy.teacher_scorer) or ()
        )
        derived = {
            "training": _derived_teacher_signals(self.loss_fn, supplied_fields=supplied)
        }
        validation = self.validation_config
        if validation is not None and validation.loss_fn is not None:
            derived["validation"] = _derived_teacher_signals(
                validation.loss_fn, supplied_fields=supplied
            )
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
                "teacher_* target in the training or validation loss and "
                f"teacher_signals={self.teacher_signals!r}. A generation-supplied "
                "target resolves no signal here, because this strategy's own "
                "scorer produces built-in signals only; pair it with a built-in "
                "teacher_* target, or request teacher_signals explicitly."
            )
        return resolved

    @model_validator(mode="after")
    def _validate_on_policy(self) -> DistillationStrategy:
        """Enforce the segment loop's duration, ownership, and mixture contract."""
        if self.on_policy is None:
            if self.reference_dataset is not None:
                raise ValueError(
                    "reference_dataset anchors the on-policy mixture and is "
                    "read only by the segment loop; got it set alongside "
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
                "replay buffer, so the anchor is policed for schema and device "
                "and then never sampled; got replay_ratio=1.0 alongside a "
                f"{type(self.reference_dataset).__name__} reference_dataset. "
                "Drop the anchor, or lower replay_ratio to mix it in."
            )
        # One probe answers the device and the schema questions alike.
        probe = (
            None
            if self.reference_dataset is None
            else self.reference_dataset.load_batches([[0]])[0]
        )
        self._validate_anchor_device(probe)
        self._validate_mixture_device(probe)
        self._validate_anchor_schema(probe)
        self._validate_generation_signals()
        return self

    def _validate_anchor_device(self, probe: Batch | None) -> None:
        """Reject an anchor emitting on an accelerator the run does not train on.

        Parameters
        ----------
        probe : Batch | None
            One batch already drawn from ``reference_dataset``, whose device is
            what a composition or a device-less store is measured by. ``None``
            when there is no anchor to measure.
        """
        if self.reference_dataset is None:
            return
        reference_device = _emitted_device(self.reference_dataset, probe)
        primary = self.devices[0]
        if reference_device.type == "cpu" or _same_device(reference_device, primary):
            return
        raise ValueError(
            "A segment's mixture is collated on the reference dataset's own "
            "device before the strategy moves it, so an anchor that emits on "
            "an accelerator has to emit on the device the run trains on; got a "
            f"reference dataset emitting on {reference_device!s} and "
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
            when there is no anchor to measure.
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

    def _validate_anchor_schema(self, probe: Batch | None) -> None:
        """Reject an anchor holding fields no generated frame can ever carry.

        The full schema comparison needs frames to compare against and so runs
        inside the first segment's
        :func:`~nvalchemi.training.distillation.build_mixed_loader`, once a
        whole generation phase — propagator steps plus a teacher pass per
        labeled frame — has already been paid for. The part that depends on
        nothing the run produces is checked here instead: the labeling hook
        strips the propagator's own predictions, the ephemeral neighbor
        tensors, and the dynamics bookkeeping from every frame it stores, so an
        anchor carrying any of them can never be mixed. A store
        :func:`~nvalchemi.training.distillation.label_dataset` wrote over an
        existing reference set — the anchor a run graduating from offline
        distillation reaches for — keeps that set's own ``energy`` and
        ``forces``, which is the part rejected here; its neighbor tensors are
        dropped by default, and the sparse list ``keep_neighbors=True`` writes
        back is rejected here too.

        Parameters
        ----------
        probe : Batch | None
            One batch already drawn from ``reference_dataset``, read here for
            the schema its levels and fields report. ``None`` when there is no
            anchor to check.
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
            f"{unmixable!r} on the anchor, which the labeling hook strips from "
            f"every frame it stores. {_SCHEMA_REMEDY}"
        )

    def _validate_generation_signals(self) -> None:
        """Check the propagator's teacher fields against the anchor and the loss.

        A scorer that declares neither ``label_fields`` nor a set of built-in
        signals writes fields nothing can know before it has scored a batch, so
        both checks below are skipped with a warning rather than run against an
        empty set — which would reject a custom scorer that in fact produces
        exactly what the anchor carries.
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
        """Enforce what the representation, curvature, and ensemble terms need."""
        self._validate_embedding_matching()
        self._validate_hessian_matching()
        self._validate_distribution_matching()
        return self

    def _matching_sides(
        self, kind: type[Any], training_fn: Any
    ) -> list[tuple[tuple[str, ...], str]]:
        """Return the components of *kind* each side's loss runs under *training_fn*.

        A ``validation_config`` carrying its own ``loss_fn`` reaches the student
        through its effective validation function — ``validation_fn`` falling
        back to ``training_fn`` — so a term only the validation loss holds is
        checked here too, and the side it came from names it in every message.
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
            and effective_fn is training_fn
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
        """Require an equilibrium on-policy ensemble for every distribution term.

        The estimator reads the batch as a sample of the student's own canonical
        ensemble, so what it needs is generation that samples one: the loop
        itself, and a propagator that keeps sampling. A relaxation propagator
        descends to a minimum and a converging one freezes each graph as it
        arrives, and in both cases the frames pile up on states the ensemble
        gives a measure of zero. Neither is detectable in a propagator the
        caller wrote, so the check is on the ones this repository ships and on
        the convergence the run is configured with — the propagator's own hook
        or a :class:`~nvalchemi.dynamics.base.ConvergenceHook` registered on it
        that graduates graphs to the root's exit status, and the criterion the
        segment loop installs from the config's ``convergence`` threshold or
        its ``convergence_hook``, which the propagator does not carry until the
        loop is running and which the hook probes below would therefore miss.
        The temperature the term is set to is not checkable at all against a
        thermostat that has not run yet.

        Generating on-policy frames is necessary and not sufficient, because
        what reaches the loss is a draw from the replay buffer rather than the
        segment that filled it: an unbounded buffer keeps every frame every
        policy ever generated and hands the term a uniform draw over all of
        them, which is warned about here. A validation config is refused
        outright when it has no loss of its own, since it would reuse this one
        on held-out batches the student never visited.
        """
        terms = _matching_components(self.loss_fn, BoltzmannMatchingLoss)
        if not terms:
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
            self.on_policy.convergence is not None
            or self.on_policy.convergence_hook is not None
        ):
            configured = (
                f"convergence={self.on_policy.convergence!r}"
                if self.on_policy.convergence is not None
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
                for hook in (
                    *getattr(stage, "hooks", ()),
                    *getattr(stage, "fused_hooks", ()),
                )
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
                "replay_ratio=1 to keep anchor rows out of the batch — the "
                "estimate is then as current as the replay buffer is, which is "
                "what replay_capacity bounds — or keep the anchor share small "
                "and read the term as regularization.",
                UserWarning,
                stacklevel=2,
            )
        if self.on_policy.replay_capacity is None:
            labelings = (
                self.on_policy.segment_steps // self.on_policy.label_frequency + 1
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
                "strategy's, ensemble term included, and the labeling seam "
                "scores validation batches for it, so the term would run there. "
                "Got validation_config.loss_fn=None; give the validation config "
                "a pointwise loss — EnergyMSELoss(target_key='teacher_energy') + "
                "ForceMSELoss(target_key='teacher_forces') — or drop the "
                "validation config."
            )

    def attach_teacher_labels(self, batch: Batch) -> bool:
        """Attach the teacher fields *batch* is missing, and report whether it did.

        Labeling is idempotent: a batch that already carries every required
        ``teacher_*`` field is returned untouched, so re-training on a batch, or
        pre-labeling one that later reaches :meth:`run`, costs at most one
        teacher forward pass. A batch carrying only some of them is re-scored in
        full and its existing teacher fields are overwritten, since a partial
        set means the batch was labeled for a different signal set than this
        objective reads.

        The teacher runs with autocast disabled whatever the caller's precision
        context, so labels never depend on how the surrounding training step is
        configured and on-the-fly labels match what
        :func:`~nvalchemi.training.distillation.label_dataset` persisted exactly
        wherever the store returns the label dtype, which over the usual float32
        dataset is every student but a float64 one; a float64 student reads
        float32 back and needs a ``dtype_policy`` on its loss terms. Inside
        :meth:`validate` it also runs with a pinned Hessian probe, so the
        validation metric compares across passes; see :meth:`validate`.

        Parameters
        ----------
        batch : Batch
            Batch to label in place. It must already sit on the teacher's
            device, which is the case for batches the strategy itself moves.

        Returns
        -------
        bool
            ``True`` when the teacher ran and fields were attached, ``False``
            when *batch* already carried them all.
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
        """Run a validation pass whose relabeled batches keep their probe directions.

        :meth:`~nvalchemi.training.TrainingStrategy.validate`, with the labeling
        seam pinned. Validation batches are labeled on the fly like training
        ones, and the labels attach to the device-placed copy rather than to the
        caller's data, so an unlabeled validation loader is relabeled from
        scratch on every pass. For a curvature objective that means a fresh
        Hutchinson probe each time, whose single-sample variance is of the order
        of its mean: the reported number would then move between passes for a
        student that had not changed at all, and best-checkpoint selection and
        the metric-driven schedulers would follow the noise. Each batch is
        therefore scored along a direction keyed to its position in the pass,
        which makes the metric a function of the student alone — as long as the
        validation data is iterated in a stable order, which a fixed held-out
        set is. Training keeps drawing fresh probes, and so does a store labeled
        once by :func:`~nvalchemi.training.distillation.label_dataset`, whose
        probe travels with the label and never needs redrawing.

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

        The teacher is frozen for the whole run, so writing its weights into
        every periodic checkpoint duplicates a model that never changed — the
        dominant cost of checkpointing a foundation teacher. Declaring it here
        stores it exactly once instead: the first checkpoint written under a
        root holds the teacher's weights, and every later one records the index
        they sit at. A run's hundredth checkpoint therefore costs the student's
        weights alone, and the tree stays self-contained, so a restart reads
        back the teacher the run actually trained against.

        Storing rather than referencing an external source is what makes that
        last part true. A teacher's ``checkpoint_spec()`` names the factory call
        that built it, which is the right thing to rebuild its *architecture*
        from but not its weights: a teacher loaded from a fine-tune checkpoint,
        or given a state dict after construction, carries weights that call
        does not reproduce. The checkpoint holds those weights itself and
        fingerprints them on the way back in. The digest samples each tensor
        rather than reading it whole, so it identifies the stored copy without
        validating it: a wrong file, a re-trained teacher, or one written at
        another precision is caught before a student trains against it, while
        an edit confined to values between two samples is not.

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
        :meth:`~nvalchemi.training.TrainingStrategy.run` over *dataloader*,
        unchanged. With it, the strategy owns the loop and repeats three phases
        until ``num_steps`` optimizer steps have run:

        *Generate* — the propagator advances the live state batch by
        ``segment_steps``, seeded on the first segment from ``seeds``.
        *Label and capture* — a
        :class:`~nvalchemi.training.distillation.TeacherLabelHook` registered on
        the propagator scores every ``label_frequency`` steps and mirrors each
        labeled frame into a host-memory sink; the segment's final frame is
        labeled too, then the sink is drained into the replay buffer.
        *Train* — a freshly built mixed loader draws ``steps_per_segment``
        batches at the configured ``replay_ratio``, each of which goes through
        the ordinary per-batch stages.

        An ``OnPolicyConfig.convergence`` criterion adds a fourth phase between
        generation and training, for the relaxation propagators whose
        trajectories end: *graduate and backfill* — converged structures are
        stored once as the minimum they reached, then leave the batch through
        :meth:`~nvalchemi.dynamics.base.BaseDynamics.refill_check` and are
        replaced by fresh seeds wherever the seed source still holds any.
        Generation stops when it runs dry and the last trajectory finishes, and
        the remaining steps train on the buffer already filled.

        Parameters
        ----------
        dataloader : Iterable[Batch] | None, optional
            Batches to train on in offline mode; any iterable, not necessarily
            a :class:`~nvalchemi.data.datapipes.dataloader.DataLoader`. Default
            ``None``, which is required in on-policy mode and rejected
            otherwise.

        Raises
        ------
        ValueError
            If *dataloader* is ``None`` in offline mode or supplied in
            on-policy mode, if a multi-rank on-policy launch holds fewer seed
            structures than ranks or an unsynchronized student, if a segment's
            loader produces no batches, if the seed structures lack a field the
            propagator opens its step with, if the propagator already carries a
            status-migrating criterion or a sampler of its own, or if the
            configured criterion migrates off a status no seed carries.

        Warns
        -----
        UserWarning
            If a lifecycle-managed run runs out of trajectories and seeds before
            reaching ``num_steps``, because the remaining steps then train on
            the frames already generated, if a multi-rank run cannot deal its
            seed structures out in equal shares, or if its propagator holds
            randomness the rank offsets cannot separate.

        Notes
        -----
        One segment is one epoch: ``AFTER_EPOCH`` fires at each segment
        boundary and an epoch-cadence ``validation_config`` follows the
        segments, while a step-cadence one fires inside them, exactly as in the
        offline loop. The run then closes with one terminal validation, skipped
        when a cadence already validated at the final step, so a metric-driven
        scheduler is never stepped twice on one set of metrics. Validation data
        is labeled on the fly by the same ``BEFORE_FORWARD`` seam that labels
        training batches, and generated frames arrive pre-labeled, so that seam
        skips them. The buffer the
        segments fill stays reachable as :attr:`replay_buffer` afterwards.

        The student is held in evaluation mode for the whole loop and flipped
        to training mode for each training phase only, so its dropout and
        batch-norm statistics never see a generated frame and a conservative
        student's forces cost no second-order graph during generation. A
        propagator model that merely composes the student is held in evaluation
        mode for the whole loop instead, because the training phase forwards
        ``models["student"]`` rather than the composition. The teacher stays
        frozen and in evaluation mode across both phases. Every mode is
        restored on the way out.

        Generated frames reach the buffer as training samples rather than
        propagator states: the labeling hook strips the ``energy``, ``forces``,
        and ``stress`` the student wrote during
        :meth:`~nvalchemi.dynamics.base.BaseDynamics.compute` along with the
        neighbor tensors and dynamics bookkeeping, so a replay frame carries no
        self-label under a reference target's name. That shape is what
        ``reference_dataset`` has to match: each segment's loader compares the
        anchor's own batch schema against the buffer's and rejects any
        difference, because collation drops a field only one side holds and
        zero-fills a whole level only one side holds. An anchor carrying plain
        ``energy`` or ``forces`` is therefore an error rather than a batch that
        silently loses or fabricates them — label it with
        :func:`~nvalchemi.training.distillation.label_dataset` first. On-policy
        losses read ``teacher_*``, built-in fields and any generation-supplied
        field the propagator's scorer declares alike, and the teacher fields
        the two sources carry are checked against each other at construction
        whenever the scorer declares enough for them to be known.

        Both mixture sources are collated before the strategy moves the batch,
        so generated frames are staged on the reference dataset's device unless
        ``OnPolicyConfig.replay_device`` names another one; a run with no anchor
        keeps them in host memory, where the segment's sink drained them. That
        placement is the anchor's to get right on a multi-rank launch: an
        anchor pinned to an indexed device emits there in every process, which
        would stage the whole world's replay frames on one accelerator, and the
        loop warns rather than re-pinning them, because the buffer cannot leave
        the device its mixture partner is on.

        The loop leaves out two pieces of the offline loop's bookkeeping. It
        never seeks a dataloader to a restored intra-epoch position, because
        each segment's loader is built from scratch, and it passes no
        dataloader to the ``SETUP`` stage, so a hook that rewraps the caller's
        loader has nothing to rewrap. It does call ``set_epoch`` on each
        segment's sampler: a freshly built mixed sampler owns a generator keyed
        on ``OnPolicyConfig.seed`` that would otherwise restart at the same
        seed every segment and redraw the identical reference samples for the
        whole run. That knob, not the global ``torch`` seed, is what makes
        replicate runs draw independently.

        The segment is the restart granularity. A run restored from a
        checkpoint picks its trajectory, the propagator's step count, and its
        replay frames back up instead of seeding afresh, and a segment a
        checkpoint interrupted part-way is counted as finished on the way in:
        its ``AFTER_EPOCH`` hooks never fire, the batches it had left are not
        replayed, and the run opens a fresh segment at the next epoch index
        rather than redrawing the reference samples the interrupted one
        already trained on. The trajectory is continuous either way, while a
        checkpoint taken part-way through a training phase costs the resumed
        run one extra generation phase for the segment it re-enters. An
        offline run graduating to the segment loop from a partial epoch is
        closed the same way. The replay buffer is kept across calls: a second
        :meth:`run` on one strategy — continuing a finished run with a raised
        ``num_steps`` — appends to the frames the first filled instead of
        regenerating them, while still reseeding its own trajectory: installing
        the rank shard reopens ``seeds`` at the front of the rows this rank
        owns, so the second call generates from the same structures again
        rather than from whatever remainder the first left behind.

        Across ranks the loop is data-parallel and self-labeling. Each rank
        propagates its own strided shard of ``seeds``, scores those
        frames with its own teacher replica, and fills its own replay buffer, so
        no generated frame and no teacher pass is duplicated. The anchor is not
        sharded: every rank builds its mixed loader over the whole
        ``reference_dataset`` from a rank-offset seed, and the mixture sampler
        draws with replacement, so the ranks draw *independently* rather than
        disjointly and one anchor sample can reach two ranks' contributions to a
        single all-reduced gradient. The same rank offset moves a stochastic
        propagator's own seed — a composition's sub-stages included — so a
        counter-based thermostat does not kick every rank identically. What it
        cannot move it accounts for per stage, before the first segment and
        from every rank: a stage exposing a :class:`torch.Generator` and no
        integer seed is named in a warning, whether or not the stages beside it
        were moved, while randomness held anywhere the walk cannot see it — the
        global ``torch`` stream, a closure — is left on the shared stream
        silently. The only cross-rank traffic is the student's gradient
        all-reduce, which a ``DDPHook`` in ``hooks`` installs by wrapping every
        optimizer-configured model — the teacher is not one of them, so it
        stays replicated and out of the collective. Because every
        rank runs the same number of segments and the same number of batches per
        segment, the ranks reach each all-reduce together. A multi-rank launch
        with nothing owning the student, or with a seed dataset holding fewer
        structures than there are ranks, is refused up front. The restart
        bundle is rank-local for the same reason it is written at all — it
        rides in a strategy checkpoint, which ``CheckpointHook`` writes on rank
        zero alone — so a bundle whose world size differs from the resuming one
        at either end is dropped with a warning and the rank seeds afresh.

        Chunking a propagator across segments is exact for the built-in
        propagators: :meth:`~nvalchemi.dynamics.base.BaseDynamics.run` never
        resets ``step_count`` or the integrator state, and the Langevin
        thermostat draws from a counter-based generator keyed on the cumulative
        step count, so ``2 x K`` steps in one call and two ``K``-step calls
        produce identical trajectories. Three consequences are the loop's to
        own. Each chunk re-enters the propagator's hook context, which
        truncates the output of an open/close-sensitive hook such as
        :class:`~nvalchemi.dynamics.hooks.LoggingHook` once per segment — the
        loop registers no such hook itself, and a caller who does should expect
        per-segment files. And a chunk stops early once every graph has
        converged, so progress is read from ``dynamics.step_count`` rather than
        assumed to be ``segment_steps``. Prefer a bare propagator to a
        :class:`~nvalchemi.dynamics.FusedStage` here for the same reason:
        a fused stage fires a priming forward pass on every ``run``, so
        chunking one into segments pays that pass once per segment.

        A relaxation run is what that early exit exists for, and
        ``OnPolicyConfig.convergence`` is what turns it into a lifecycle. The
        criterion is registered on the propagator ahead of the labeling hook and
        installed as its detector for the duration of the loop, so a converged
        structure freezes in the propagator's step, is captured once at
        ``AFTER_STEP`` on the step its ``status`` reaches the propagator's
        ``exit_status`` — a transition rather than an ``ON_CONVERGE`` dispatch,
        which a :class:`~nvalchemi.dynamics.FusedStage` never makes on itself —
        and is left out of every later path capture of the segment instead of
        being stored again on each one. At the segment
        boundary those structures graduate through
        :meth:`~nvalchemi.dynamics.base.BaseDynamics.refill_check` and fresh
        seeds are appended in their place, where the seed source has any left.
        A budgeted :class:`~nvalchemi.training.distillation.SeedSource` packs
        the initial batch and leaves the remainder in cursor order for that
        backfill, while an unbudgeted one is propagated whole and therefore
        opens its cursor past the last row: a graduation narrows the batch
        instead, until ``SeedSource.recycle`` restarts it at the front of the
        rows this rank owns. The seed source is attached for that call alone,
        because the
        propagator's ``run`` only exits a chunk early while it holds none. Once
        no trajectory is left and no seed remains to start one, the loop warns
        and keeps training on the buffer it has until ``num_steps``. The two
        capture routes therefore partition a segment's frames rather than
        overlapping on any of them, and the converged ones are labeled in a
        single teacher pass as their sink is drained rather than one pass per
        convergence step.

        Note that generation and graduation move together only for a propagator
        whose trajectories end. A thermostat run never converges, which is
        exactly why ``convergence`` defaults to ``None`` and no lifecycle is
        managed unless it is set.
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

        The labeling hook is rebuilt every call, so a resumed run hands it back
        the step the interrupted one last labeled. That step is the forced
        boundary label :meth:`_capture_segment` makes at ``segment_steps - 1``,
        which the restart bundle does not carry: a hook starting out unaware of
        it lets the cadence fire on the adjacent step, paying for a second
        teacher pass and storing a frame one step from one already in the
        buffer — the pair the adjacency rule exists to avoid.

        The rank shard is installed on the seed source here rather than at
        construction, because the world size is a launcher fact and because
        installing it rewinds the cursor: a second ``run()`` on one strategy
        keeps the replay buffer it filled and reseeds only the trajectory, so
        the source has to open at the front of its shard again.
        """
        training_started = False
        strategy_context = nullcontext(self) if self._context_depth > 0 else self
        with strategy_context:
            self._prepare_setup_hooks()
            self._validate_runtime_devices()
            self._validate_distributed_generation(config)
            self._warn_unequal_seed_shards(config)
            self._warn_shared_propagator_streams(config)
            self.models = move_to_devices(self.models, self.devices)
            self._run_setup_hooks()
            self._validate_synchronized_student(config)
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
                config.seeds.shard(
                    get_rank(self.distributed_manager),
                    get_world_size(self.distributed_manager),
                )
                if self._replay_buffer is None:
                    self._replay_buffer = ReplayBuffer(
                        capacity=config.replay_capacity,
                        eviction=config.replay_eviction,
                        device=replay_device,
                    )
                buffer = self._replay_buffer
                state, labeled_step = self._resume_or_seed(config, buffer)
                state = _to_device(state, primary_device)
                self._on_policy_state = state
                label_hook = TeacherLabelHook(
                    config.teacher_scorer, frequency=config.label_frequency
                )
                label_hook._labeled_step = labeled_step
                with _relaxation_lifecycle(config, state) as lifecycle:
                    config.dynamics.register_hook(label_hook)
                    # A DDPHook has replaced models["student"] with a wrapper by
                    # now; the mode contexts are about the module the propagator
                    # holds.
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
                                        self._warn_generation_exhausted(
                                            config, target_step_count
                                        )
                                segment_steps = min(
                                    config.steps_per_segment,
                                    target_step_count - self.step_count,
                                )
                                with train_configured_models(
                                    self.models, self.optimizer_configs
                                ):
                                    training_started = self._train_segment(
                                        config,
                                        buffer,
                                        segment_steps=segment_steps,
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

    def _validate_distributed_generation(self, config: OnPolicyConfig) -> None:
        """Reject a seed source a multi-rank generation phase cannot share out.

        The world size is read at run time rather than at construction because
        that is when a launcher has initialized the process group, and because
        an offline strategy the same script builds is free to be distributed
        however it likes.

        Parameters
        ----------
        config : OnPolicyConfig
            Configuration of the loop about to start.

        Raises
        ------
        ValueError
            If the seed dataset holds fewer structures than there are ranks.
        """
        world_size = get_world_size(self.distributed_manager)
        if world_size == 1:
            return
        num_seeds = len(config.seeds.dataset)
        if num_seeds < world_size:
            raise ValueError(
                "Every rank propagates its own share of the seed structures, so "
                "there has to be at least one for each; got a seed dataset of "
                f"{num_seeds!r} structures on {world_size!r} ranks. Seed the run "
                "with more structures, or launch fewer ranks."
            )

    def _warn_unequal_seed_shards(self, config: OnPolicyConfig) -> None:
        """Report a seed set the world cannot deal out in equal shares.

        A shard shorter by one structure is not a rounding detail. Every rank
        draws the same number of replay samples per batch from a buffer holding
        only its own trajectories, so a frame on a shorter shard is drawn more
        often, and DDP averages the ranks' gradients evenly rather than by the
        frames behind them. The arithmetic is the world's rather than this
        rank's, so every rank reaches the same verdict without a collective.

        Parameters
        ----------
        config : OnPolicyConfig
            Configuration of the loop about to start.

        Warns
        -----
        UserWarning
            If the seed structures do not divide evenly across the ranks.
        """
        world_size = get_world_size(self.distributed_manager)
        if world_size == 1:
            return
        num_seeds = len(config.seeds.dataset)
        smallest, remainder = divmod(num_seeds, world_size)
        if remainder == 0:
            return
        warnings.warn(
            "The seed structures do not divide evenly across the world, so the "
            f"ranks propagate shards of different sizes: {num_seeds!r} "
            f"structures on {world_size!r} ranks deals {smallest + 1!r} to "
            f"{remainder!r} of them and {smallest!r} to the rest. Every rank "
            "draws the same number of replay samples per batch from a buffer "
            "holding only its own trajectories, and the gradients are averaged "
            "rank by rank, so a frame generated on a shorter shard reaches the "
            f"optimizer with up to {(smallest + 1) / smallest:.2f}x the weight "
            "of one from a longer shard. Size the seed dataset as a whole "
            f"multiple of {world_size!r} to weight every generated frame alike.",
            UserWarning,
            stacklevel=2,
        )

    def _warn_shared_propagator_streams(self, config: OnPolicyConfig) -> None:
        """Report the propagator randomness the rank offsets cannot separate.

        Warns rather than raises, because a run whose propagator is
        deterministic in the stages the walk cannot reach is perfectly correct,
        and nothing here can tell the two apart.

        The report is bound to the world rather than to this rank's offset. The
        offset is zero on rank zero, so a check hanging off it speaks only from
        the ranks whose stderr a launcher filters away — and never at all from
        the single-process run a user smoke-tests with before scaling out. The
        composition is identical on every rank, so every rank reaches the same
        verdict here, before a segment has been generated or a teacher pass
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

    def _validate_synchronized_student(self, config: OnPolicyConfig) -> None:
        """Reject a multi-rank run whose student nothing keeps in step.

        Called after the ``SETUP`` stage, which is when a
        :class:`~nvalchemi.training.hooks.DDPHook` has replaced every
        optimizer-configured model with a wrapper — leaving the propagator
        holding the bare student the wrapper now owns. What is tested is exactly
        that: whether ``models['student']`` is still the object the propagator
        drives. Anything that has taken ownership of the student clears the
        guard, a hand-rolled wrapper or an FSDP one as much as a ``DDPHook``,
        and nothing here can tell a synchronizing wrapper from one that only
        looks like one.

        Parameters
        ----------
        config : OnPolicyConfig
            Configuration of the loop about to start.

        Raises
        ------
        ValueError
            If nothing has taken ownership of the student to synchronize its
            gradients.
        """
        world_size = get_world_size(self.distributed_manager)
        if world_size == 1:
            return
        if _propagates_student(config.dynamics.model, self.models["student"]):
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

        A checkpoint taken mid-segment — and an offline run graduating to the
        segment loop from a partial epoch — restores a nonzero
        ``epoch_step_count``, which the loop has no way to honor: each segment
        builds its own loader, the batches the interrupted segment had already
        drawn are gone with it, and the trajectory that produced them is
        reseeded anyway. Closing it here is what keeps the rest of the loop
        coherent: ``BEFORE_EPOCH`` fires for the resumed segment,
        ``epoch_step_count`` stays inside ``steps_per_segment``, and the
        mixture sampler advances past the epoch index the interrupted segment
        already drew with instead of redrawing its reference samples.

        The parent's :meth:`_prepare_epoch_step_count` is deliberately not used
        for this: it reconciles the restored counters against a fixed number of
        batches per epoch, which the graduation path — where the offline
        epochs were a different size — does not have.
        """
        if self.epoch_step_count == 0:
            return
        self.epoch_count += 1
        self.epoch_step_count = 0
        self._refresh_hook_counters()

    def _validation_checkpoint(self, stage: TrainingStage) -> bool:
        """Run a scheduled validation and remember the step it fired at.

        The segment loop closes with a terminal validation, which would
        otherwise repeat the pass an epoch cadence has just run at the same
        ``step_count`` and step every metric-driven scheduler a second time on
        identical metrics. Recording the step is what lets the closing block
        tell a cadence that already landed there from one that did not.

        Parameters
        ----------
        stage : TrainingStage
            Lifecycle stage that triggered this checkpoint.

        Returns
        -------
        bool
            Whether a validation pass ran at this checkpoint.
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
        segment_steps: int,
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
            num_batches=segment_steps,
            seed=config.seed + self._rank_seed_offset(),
        )
        self._set_sampler_epoch(loader)
        primary_device = self.devices[0]
        consumed = 0
        for batch in loader:
            if consumed >= segment_steps or self.step_count >= target_step_count:
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
            trajectory has finished and the seed source has nothing left to
            start a fresh one from.
        """
        # Sized per segment because a refill changes the trajectory count.
        label_hook.sink = HostMemory(
            capacity=(config.segment_steps + 1) * state.num_graphs
        )
        if lifecycle is not None:
            lifecycle.capture.sink = HostMemory(capacity=state.num_graphs)
        state = config.dynamics.run(state, n_steps=config.segment_steps)
        if lifecycle is not None:
            self._capture_budget_graduates(config, state, label_hook, lifecycle)
        self._capture_segment(config, state, label_hook, buffer)
        if lifecycle is None:
            return state
        self._capture_converged(config, lifecycle, buffer)
        return self._refill_segment(config, lifecycle, state)

    def _capture_budget_graduates(
        self,
        config: OnPolicyConfig,
        state: Batch,
        label_hook: TeacherLabelHook,
        lifecycle: _RelaxationLifecycle,
    ) -> None:
        """Store the structures a step budget graduated as the chunk ended.

        A :class:`~nvalchemi.dynamics.FusedStage` sub-stage that graduates on
        an ``n_steps`` budget rather than on a criterion migrates status after
        the fused ``AFTER_STEP`` dispatch, so the capture hook reads ``0`` on
        the step the budget runs out and the status is consistent only once
        :meth:`~nvalchemi.dynamics.base.BaseDynamics.run` has returned. A
        budget that graduates every remaining graph ends the chunk on that step
        as well, so the segment behind it opens on a batch with nothing left
        moving and neither capture route ever reaches the frame.

        Which is why this runs before the segment's closing dispatch rather
        than after it: that dispatch labels a subset still moving and marks the
        step as covered, and this one has to read the marker as the propagator
        left it. The marker is the idempotence guard, because the path route
        stores a whole frame only while nothing has graduated yet — exactly the
        status the budget migration hid behind — so a step it already stored
        needs no second capture and a re-dispatch would only duplicate it. The
        capture hook's own record covers the other direction, keeping a
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

        This is the deferred half of on-policy labeling. Converged frames are
        captured raw, at the step each structure reached its minimum, and the
        teacher sees them here in one pass over the whole segment's graduates
        rather than one pass per convergence step — which is what decouples the
        teacher's batch size from the propagated one. They are stripped to the
        replay-frame contract afterwards, so they enter the buffer under the
        same schema the path frames froze it with, and staged back onto the
        buffer's own device, which the path route left in host memory when the
        run has no anchor to follow.
        """
        sink = lifecycle.capture.sink
        if len(sink) == 0:
            return
        frames = _to_device(sink.drain(), self.devices[0])
        _attach_teacher_labels(frames, config.teacher_scorer.label(frames))
        buffer.extend(_strip_replay_frame(frames).to(buffer.device or "cpu"))

    def _refill_segment(
        self,
        config: OnPolicyConfig,
        lifecycle: _RelaxationLifecycle,
        state: Batch,
    ) -> Batch | None:
        """Graduate the converged structures and backfill fresh seeds.

        The sampler is attached for this call alone.
        :meth:`~nvalchemi.dynamics.base.BaseDynamics.run` cuts a chunk short
        once every graph has converged, but only while no sampler is
        configured, and that early exit is exactly the signal that a refill is
        due — leaving the sampler attached for the whole loop would trade it
        for segments spent propagating frozen structures.

        A replacement arrives holding whatever its source stored it with, and
        ``refill_check`` deliberately preserves that, so the run installs its
        own bookkeeping over the rows the backfill appended — the same
        invariant the seed batch enters under, completed here. The one field
        kept is the ``system_id`` the sampler handed out, which is the sampler's
        to number. Anything else a source carried is the record of the run that
        wrote it: a seed store filled by a relaxation holds ``status`` at the
        code its structures graduated on, and a replacement arriving frozen is
        never propagated, stored raw as a minimum it never reached, and
        graduated again at the next boundary.

        Returns
        -------
        Batch | None
            The refilled batch, or ``None`` once nothing is left to propagate,
            which is what ``refill_check`` itself returns in that case — the
            ``done`` flag it raises alongside outlives the sampler it was
            derived from and is not read here.
        """
        dynamics = config.dynamics
        survivors = int((state["status"].view(-1) < dynamics.exit_status).sum())
        previous = dynamics.sampler
        dynamics.sampler = lifecycle.sampler
        try:
            refilled = dynamics.refill_check(state, dynamics.exit_status)
        finally:
            dynamics.sampler = previous
        if refilled is state:
            return refilled
        lifecycle.capture.reset()
        if refilled is not None:
            fresh = refilled.num_graphs - survivors
            for key, default_fn in dynamics._bookkeeping_keys.items():
                if key != "system_id":
                    refilled[key][survivors:] = default_fn(fresh, refilled.device)
        return refilled

    def _warn_generation_exhausted(
        self, config: OnPolicyConfig, target_step_count: int
    ) -> None:
        """Announce that the run trains on what it has already generated."""
        remedy = (
            "Pass seeds=SeedSource(dataset, recycle=True) to keep generating "
            "from the front of the rows this rank owns, or seed from more "
            "structures — an unbudgeted source is propagated whole, so more of "
            "them lengthen the run by widening the initial batch rather than "
            "by backfilling it."
            if config.seeds.exhausted
            else "The source still holds rows, so widen its budget: nothing a "
            "pass over it reached fits the envelope the seeded batch recorded."
        )
        warnings.warn(
            "Every generated trajectory has finished and the seed source has "
            "nothing left to start a fresh one from, so generation stopped "
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

        Frames reach the buffer from a host-memory sink rather than from the
        propagator, so an unset ``replay_device`` means the reference dataset's
        device: the two mixture sources are collated into one batch before the
        strategy moves it, and only the anchor decides where that happens. A
        run with no anchor leaves them in host memory.

        The anchor's device is the one it actually emits on, measured from a
        batch when no declaration settles it — a composition declares no device
        at all, and a store opened without one declares an index-less ``cuda``
        that names whichever device is current. Reading the declaration alone
        would stage the buffer in host memory beside a CUDA-resident anchor and
        fail only once the first segment's loader collated them. The anchor is
        measured here rather than at construction, where validation drew a probe
        of its own: a launcher pins the process to its device only after the
        datasets are built, and moving the anchor once it has is the documented
        remedy for a world staging every rank's frames on one accelerator.

        A ``replay_device`` the caller spells index-less is resolved to the
        device this process has made current, which under a launcher is the one
        it pinned this rank to. The spelling would otherwise survive into the
        staged frames: a batch moved by ``.to("cuda")`` records the spelling
        rather than the device its tensors landed on, and an index into those
        frames is resolved against the record, which need not name the same
        device. An emitted device is concrete already and is left as measured.

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

        Datasets are built before a launcher pins the process to its device, so
        an anchor loaded onto ``cuda:0`` — or declaring an indexed
        ``target_device`` — emits there in *every* process, and the buffer has
        to follow it because a mixed batch is collated before the strategy
        moves it. The whole world's buffers and mixture collation then land on
        one GPU while the ranks train on their own. Nothing is computed wrongly,
        which is the problem: it surfaces as an unexplained out-of-memory on a
        single device at a ``replay_capacity`` the run sized per rank. An
        index-less ``cuda`` names whichever device the process is on and is
        what a rank-local anchor looks like, so it is left alone.

        The report is bound to the world rather than to this rank's placement.
        Rank zero is the rank an anchor pinned to ``cuda:0`` concentrates onto,
        so its own placement says nothing about the world's — and a check
        hanging off it would speak only from the ranks whose stderr a launcher
        filters away. Each rank reduces the one bit it alone can see, whether
        the device it is about to stage on is its own, and every rank reports
        once the world agrees that some rank's is not. The placement
        conditions live inside that bit rather than in a guard above it, so
        every rank past a single-process world reduces exactly one verdict and
        none can return from a collective its peers are still waiting on.

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
            f"{device!s}, which is not the device every rank trains on: an "
            "anchor pre-staged on an indexed device emits there in every "
            "process, and the generated frames have to follow the anchor "
            "because a mixed batch is collated before it is moved. The whole "
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

        Both the segment's mixture sampler and a stochastic propagator seed
        themselves from a base seed plus a counter — the segment index and the
        propagator's cumulative step count — so ranks are separated by a whole
        stride of the seed space rather than by one, and their streams stay
        apart for as many segments and steps as the stride is wide. The stride
        is taken on the *global* rank, as the seed shard is: node-local indices
        repeat once the world spans more than one node, and every node's rank
        zero would then draw the one stream.
        """
        return get_rank(self.distributed_manager) * _RANK_SEED_STRIDE

    def _resume_or_seed(
        self, config: OnPolicyConfig, buffer: ReplayBuffer
    ) -> tuple[Batch, int | None]:
        """Return the batch to propagate, resuming a checkpointed run when there is one.

        A restored checkpoint carries the trajectory the interrupted run had
        reached, the propagator's cumulative step count, and the frames already
        in its replay buffer, so the resumed run continues the same trajectory
        rather than starting a fresh one from the seeds. That is what makes the
        continuation exact for a propagator whose whole state is the batch and
        that counter — the built-in integrators, whose Langevin noise is drawn
        from a counter-based generator keyed on the step count. A propagator
        carrying internal state of its own is not continued that far: a
        relaxation optimizer's adaptive history lives outside the batch, so
        :class:`~nvalchemi.dynamics.optimizers.FIRE` re-initializes its
        timestep, its mixing coefficient, and its uphill counter from the
        constructor arguments and only the positions continue.

        The bundle's frames *are* the replay buffer as of the checkpoint, so
        they replace what the buffer holds rather than being appended to it.
        The buffer outlives a :meth:`run` call, and a strategy restored while
        still holding the frames it generated would otherwise carry the
        pre-checkpoint half of them twice: not a diversity loss, since the
        mixed loader draws with replacement, but a weighting skew toward the
        stale states — exactly backwards for an on-policy loop — on top of
        double the buffer memory and an eviction horizon reached a restart
        early.

        The bundle describes one rank's run, because
        :class:`~nvalchemi.training.hooks.CheckpointHook` writes the strategy
        checkpoint it rides in on rank zero alone. It is consumed only when
        that rank is the whole world at both ends of the restart, and dropped
        with a warning otherwise — see :meth:`_rank_local_restart_reason`.

        The restore order follows what each piece is read from. The seed
        cursor is resumed first, because it is what the shard the bundle was
        written on has to agree with and the cheapest thing to refuse on — a
        cursor this rank's shard cannot agree with drops the bundle rather
        than raising out of :meth:`run` — and it carries the size envelope an
        unbudgeted source measured off its seeds; the trajectory is rebuilt
        next and handed to
        :meth:`~nvalchemi.training.distillation.SeedSource.record_envelope`,
        which covers a bundle written before that envelope was checkpointed
        and yields to one the source already holds, since a restored run never
        calls ``initial_batch``; and the replay frames land last, because
        nothing else reads them.

        Returns
        -------
        tuple[Batch, int | None]
            The batch the next segment propagates from, and the step the
            interrupted run last labeled, which the segment loop hands to the
            labeling hook it rebuilds. The step is ``None`` when the run seeds,
            leaving a fresh hook's cadence untouched.
        """
        restored = self._take_restart_state()
        if restored is None:
            return config.seeds.initial_batch(), None
        reason = self._rank_local_restart_reason(restored)
        if reason is None:
            reason = self._restore_seed_cursor(config, restored)
        if reason is not None:
            warnings.warn(
                f"The on-policy restart bundle is dropped: {reason} It holds "
                "one rank's trajectory and one rank's replay frames, and "
                "replaying those onto every rank would have every rank "
                "propagate rank zero's structures and train on rank zero's "
                "frames. This rank seeds afresh from its own share of the seed "
                "source with a cold replay buffer instead, so budget the first "
                "segments after the restart for refilling it: until they do, "
                "the mixture is drawn from the reference dataset alone.",
                UserWarning,
                stacklevel=2,
            )
            return config.seeds.initial_batch(), None
        config.dynamics.step_count = int(restored["dynamics_step_count"])
        self._warn_knob_drift(config, restored)
        state = _batch_from_state(restored["md_state"])
        config.seeds.record_envelope(state)
        frames = restored.get("replay_frames")
        if frames is not None:
            buffer.clear()
            buffer.extend(_batch_from_state(frames))
        return state, max(config.dynamics.step_count - 1, 0)

    def _restore_seed_cursor(
        self, config: OnPolicyConfig, restored: Mapping[str, Any]
    ) -> str | None:
        """Resume the seed cursor the bundle recorded, or say why it cannot be.

        The source is the authority on whether a cursor belongs to the shard
        it is being loaded onto, and refuses one that does not; the segment
        loop turns that refusal into the same drop a mismatched world size
        gets, because the alternative is a :class:`ValueError` out of
        :meth:`run` once the weights, the optimizers and the counters have
        already been restored.

        Parameters
        ----------
        config : OnPolicyConfig
            Segment loop whose source is resumed, already narrowed to this
            rank's shard.
        restored : Mapping[str, Any]
            Bundle a restored checkpoint carried.

        Returns
        -------
        str | None
            A sentence naming a cursor this rank's shard cannot take, or
            ``None`` when the cursor was resumed or the bundle carries none.

        Warns
        -----
        UserWarning
            If the bundle predates the seed cursor and carries no position.
        """
        cursor = restored.get("seeds")
        if cursor is None:
            warnings.warn(
                "The on-policy restart bundle carries no seed cursor, so the "
                "resumed run backfills from the front of its shard and serves "
                "structures the interrupted run already propagated. It was "
                "written before the cursor was checkpointed; take a fresh "
                "checkpoint to restore exactly.",
                UserWarning,
                stacklevel=2,
            )
            return None
        try:
            config.seeds.load_state_dict(cursor)
        except ValueError:
            return (
                f"its seed cursor was written for rank {cursor['rank']!r} of "
                f"{cursor['world_size']!r}, and this rank's shard counts "
                "positions in a different set of rows."
            )
        return None

    @staticmethod
    def _warn_knob_drift(config: OnPolicyConfig, restored: Mapping[str, Any]) -> None:
        """Report the knobs the resumed loop sets differently from the interrupted one.

        Parameters
        ----------
        config : OnPolicyConfig
            Segment loop the run resumes under.
        restored : Mapping[str, Any]
            Bundle a restored checkpoint carried.

        Warns
        -----
        UserWarning
            If a knob the bundle recorded differs from the one in hand.
        """
        recorded = restored.get("knobs")
        if recorded is None:
            return
        current = config.knobs.model_dump(mode="json")
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
            "trajectory, replay frames and seed cursor were produced under "
            f"knobs the rest of this run will not use; got {now!r} against "
            f"{then!r}. Restore the knobs to compare the halves, or start a "
            "fresh run to change them.",
            UserWarning,
            stacklevel=2,
        )

    def _rank_local_restart_reason(self, restored: Mapping[str, Any]) -> str | None:
        """Return why a rank-zero-only restart bundle cannot be consumed, or ``None``.

        Two worlds have to agree for the bundle to describe the whole run: the
        one it was written in and the one it is restored into. The bundle names
        the first itself, in the shard its seed cursor records. The two step
        counters are the fallback for a bundle written before the cursor was:
        ``step_count`` counts a rank's own optimizer steps while
        ``global_step_count`` advances by the world size, so their ratio
        recovers the world of the last leg alone, and a run whose history spans
        world sizes leaves counters whose ratio names neither.

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
        cursor = restored.get("seeds")
        recorded = None if cursor is None else cursor.get("world_size")
        if recorded is None:
            # Fallback: a bundle written before its cursor named the shard.
            recorded = (
                self.global_step_count // self.step_count if self.step_count > 0 else 1
            )
        saved_world_size = int(recorded)
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

        The propagator's cadence rarely lands on a segment's last step, and that
        frame is the most on-policy one the segment produced, so the hook is
        asked once more for the step it just finished. Labeling is idempotent
        per step, so a cadence that did land there costs nothing and stores
        nothing twice.

        The hook's private entry point is called rather than the hook itself,
        because this is a *forced* label rather than a cadence dispatch, and the
        two are treated differently: a cadence firing on the step right after a
        forced label is passed over, so a ``segment_steps`` that is a multiple
        of ``label_frequency`` pays for one teacher pass per segment boundary
        instead of two on adjacent frames. Going through ``__call__`` would
        build a :class:`~nvalchemi.hooks._context.DynamicsContext` the hook
        reads two fields of and lose that distinction.
        """
        label_hook._label_frame(
            state,
            max(config.dynamics.step_count - 1, 0),
            exit_status=config.dynamics.exit_status,
            forced=True,
        )
        if label_hook.sink is not None and len(label_hook.sink) > 0:
            buffer.extend(label_hook.sink.drain())

    def to_spec_dict(self) -> dict[str, Any]:
        """Serialize declarative distillation knobs to a JSON-ready dict.

        The bundle names its own strategy class under ``strategy_cls``, the key
        :meth:`to_checkpoint_dict` writes with the same value, so a spec that
        travels alone still says which strategy rebuilds it — and
        :meth:`from_spec_dict` builds the class it names.

        An on-policy run serializes too: ``on_policy`` becomes the recipe
        :meth:`~nvalchemi.training.distillation.OnPolicyConfig.to_spec_dict`
        produces — the propagator's spec, the scorer's signals, the seed
        store's path and budgets, and every scalar knob — and
        ``reference_dataset`` becomes
        the store it reads. Both are references rather than objects: the
        rebuilt strategy needs its models supplied, and a dataset that holds
        its samples in memory cannot be named at all.

        A piece the recipe cannot describe leaves the whole ``on_policy``
        entry out and says why, rather than writing a recipe that would rebuild
        into a different run. A strategy rebuilt from such a spec is
        offline-shaped unless the two are passed back to
        :meth:`from_spec_dict`, :meth:`from_checkpoint_dict`, or
        :meth:`load_checkpoint` as keyword arguments — which is also how a live
        loop replaces a recipe the spec does carry.

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

        A spec carrying an ``on_policy`` recipe rebuilds the segment loop too:
        the propagator around the supplied student, the scorer around the
        supplied teacher, and the seed and reference datasets from the stores
        they name. Both objects travel with the *models* they were built
        around: the propagator has to hold the very object supplied as
        ``models['student']``, which is what makes each segment generate from
        the weights the last one trained.

        A ``strategy_cls`` naming a subclass builds that subclass rather than
        this one: the spec and *every* runtime override are handed to the named
        class's own ``from_spec_dict``, so the strategy a spec says rebuilds it
        is the strategy that runs. A forward that drops an override would be
        worse than no dispatch at all — the subclass would rebuild that object
        from the recipe and quietly discard the live one the caller handed
        over — so a subclass adding a runtime keyword must widen this call
        with it.

        Runtime objects resolve in a fixed order — an explicit keyword here,
        then whatever :meth:`load_checkpoint` or :meth:`from_checkpoint_dict`
        offered over :func:`_supplied_runtime_objects`, then the recipe the
        spec carries — and a dispatched subclass resolves them the same way
        because the offer is still standing when its own ``from_spec_dict``
        reads it. A live loop therefore outranks a describable recipe, which is
        what restores a run whose datasets live in memory, or whose propagator
        carries hooks, around the objects the caller still holds.

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
            Anchor dataset to use instead of the one the recipe names. Default
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
        The segment loop is resolved by a fixed precedence: an explicitly
        supplied *on_policy* wins, then a loop the caller registered for the
        restore, then the spec's own recipe. A recipe is the weakest source
        because it is the only one that cannot be complete — it names its seed
        store by path and its propagator by constructor arguments, so a loop
        the caller is already holding is the more faithful description of the
        run. A loop handed to this method or to
        :func:`~nvalchemi.training.load_checkpoint` must therefore be the one
        that runs, never quietly replaced by a describable recipe the
        checkpoint happens to carry.

        The corollary is that a spec resume cannot re-supply an in-memory seed
        set. A recipe refuses to describe an
        :class:`~nvalchemi.data.datapipes.in_memory_dataset.InMemoryDataset` by
        design — there is no path to name it by — so a run seeded from one
        serializes without its ``on_policy`` block at all, and restoring it
        means passing *on_policy* here. A propagator carrying hooks takes the
        same route.
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
        recipe = spec.get("on_policy")
        rebuildable = isinstance(model_input, Mapping) and _REQUIRED_MODELS <= set(
            model_input
        )
        if on_policy is None and recipe is not None and rebuildable:
            on_policy = OnPolicyConfig.from_spec_dict(
                recipe,
                student=model_input["student"],
                teacher=model_input["teacher"],
            )
        anchor_spec = spec.get("reference_dataset")
        if reference_dataset is None and anchor_spec is not None:
            reference_dataset = _dataset_from_spec_dict(anchor_spec)
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
            Anchor dataset the segment loop mixes into every batch. Default
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
        with the runtime objects a distillation spec cannot describe. An
        objective defined on generated batches — an ensemble term is — refuses
        to be rebuilt without them, so restoring such a run means re-supplying
        the segment loop here.

        The segment loop travels with the student it propagates: the propagator
        has to hold the very object registered as ``models['student']``, which
        is why *models* is re-supplied alongside *on_policy* rather than left
        to the loader's own rebuild from the checkpoint's model specs. The
        checkpoint's weights are loaded into whatever models this call is
        given, so the restored run generates from where it left off.

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
            Anchor dataset the segment loop mixes into every batch. Default
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
