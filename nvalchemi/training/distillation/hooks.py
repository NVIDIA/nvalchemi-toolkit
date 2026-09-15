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
"""Dynamics hooks capturing on-policy frames as a propagator produces them."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from nvalchemi.dynamics.base import BaseDynamics, DynamicsStage, FusedStage
from nvalchemi.dynamics.hooks.snapshot import ConvergedSnapshotHook
from nvalchemi.training.distillation._labels import (
    _attach_teacher_labels,
    _prune_empty_edges,
    _reject_foreign_fields,
)
from nvalchemi.training.distillation.scoring import _NEIGHBOR_KEYS, scorer_fields

if TYPE_CHECKING:
    from enum import Enum

    from nvalchemi.data import Batch
    from nvalchemi.data.level_storage import BaseLevelStorage
    from nvalchemi.dynamics.sinks import DataSink
    from nvalchemi.hooks._context import DynamicsContext
    from nvalchemi.training.distillation.scoring import TeacherScorer

__all__ = ["TeacherLabelHook"]

_PREDICTION_KEYS = frozenset(BaseDynamics._OUTPUT_KEY_TO_BATCH_ATTR.values())
"""Batch fields a propagator overwrites with the propagated model's predictions."""


def _run_local_keys() -> frozenset[str]:
    """Return the fields of a live frame that mean nothing outside its run.

    Read at call time rather than at import time, because
    :meth:`~nvalchemi.dynamics.base.BaseDynamics.register_bookkeeping_key` grows
    the bookkeeping registry as stages are built — a fused stage registers one
    step counter per sub-stage.

    :class:`~nvalchemi.dynamics.FusedStage` declares bookkeeping of its own on
    the class instead of through that registry — the ``reprime_pending`` flag it
    raises on a graph that has just entered a sub-stage — so its registry is
    read alongside the base one rather than reached through it.
    """
    return (
        _NEIGHBOR_KEYS
        | _PREDICTION_KEYS
        | frozenset(BaseDynamics._bookkeeping_keys)
        | frozenset(FusedStage._bookkeeping_keys)
    )


def _strip_replay_frame(frames: Batch) -> Batch:
    """Reduce *frames* to the replay-frame contract, in place.

    A frame captured off the propagator carries the run with it: the ephemeral
    neighbor tensors, the dynamics bookkeeping, and the ``energy``, ``forces``,
    and ``stress`` the propagated model wrote over. A replay frame carries none
    of that — only the structure, the propagator state travelling with it, and
    the ``teacher_*`` labels — so a stored frame never offers a self-label under
    the name a reference target uses.

    Parameters
    ----------
    frames : Batch
        Frames to strip, mutated in place.

    Returns
    -------
    Batch
        The same object, holding nothing run-local.
    """
    dropped = _run_local_keys()
    for group in frames._storage.groups.values():
        for key in [name for name in group.keys() if name in dropped]:
            del group[key]
    if frames.keys is not None:
        for names in frames.keys.values():
            names -= dropped
    _prune_empty_edges(frames)
    return frames


def _graph_status(batch: Batch) -> torch.Tensor | None:
    """Return one status per graph of *batch*, or ``None`` when it carries none.

    Bookkeeping is stored as a column, and an inflight batch keeps rows past
    the graphs it currently holds, so neither the shape nor the length of the
    stored field can be compared against an exit status as it stands.

    Parameters
    ----------
    batch : Batch
        Frame the status is read off.

    Returns
    -------
    torch.Tensor | None
        Flat per-graph status, or ``None`` for a frame carrying no ``status``.
    """
    status = getattr(batch, "status", None)
    if status is None:
        return None
    flat = status.squeeze(-1) if status.dim() == 2 else status
    return flat[: batch.num_graphs]


def _active_graphs(batch: Batch, exit_status: int | None) -> torch.Tensor | None:
    """Return the graphs still being propagated, or ``None`` when all of them are.

    Parameters
    ----------
    batch : Batch
        Live frame, carrying ``status`` once a lifecycle is managed.
    exit_status : int | None
        Status at which a graph counts as graduated, or ``None`` when the
        propagator declares none.

    Returns
    -------
    torch.Tensor | None
        Indices of the graphs below *exit_status*, or ``None`` when every graph
        is below it — which is also the answer for a frame carrying no status
        at all.
    """
    status = _graph_status(batch)
    if status is None or exit_status is None:
        return None
    active = status < exit_status
    if bool(active.all()):
        return None
    return torch.where(active)[0]


class TeacherLabelHook:
    """Label the live propagator frame with teacher signals, inline.

    This is the inline half of on-policy labeling: the hook fires at
    :attr:`~nvalchemi.dynamics.base.DynamicsStage.AFTER_STEP`, once the
    propagator has fully resolved the step, and attaches every signal its
    scorer produces to the very :class:`~nvalchemi.data.Batch` being
    propagated. Each signal lands at the level it declares, through
    :meth:`~nvalchemi.data.Batch.add_key`: a per-atom field written as a plain
    attribute would be routed to the system group and left out of the level
    tracking that the Zarr writer and the loss target lookup both read.

    The propagator is left alone. Teacher signals populate ``teacher_*`` fields
    only, so the ``energy`` and ``forces`` the student wrote during
    :meth:`~nvalchemi.dynamics.base.BaseDynamics.compute` — the values driving
    the next step — are never overwritten, and a scorer that declares, or
    returns, a field outside that namespace is rejected — at construction and
    at labeling respectively — rather than allowed to clobber propagator
    state. Nothing here assumes molecular dynamics: a relaxation optimizer
    such as :class:`~nvalchemi.dynamics.optimizers.FIRE` is labeled the same
    way, at the same stage.

    A ``sink`` additionally mirrors every labeled frame into a
    :class:`~nvalchemi.dynamics.sinks.DataSink`, so a segment's trajectory can
    be drained into a replay buffer or a store after the run. What reaches the
    sink is a training sample rather than a propagator state: a copy stripped
    of the ephemeral neighbor tensors — whose neighbor dimension changes
    between adaptive rebuilds and would break concatenation — of the dynamics
    bookkeeping fields (``status``, ``system_id``, the per-status step
    counters) that are meaningless outside the run that wrote them, and of the
    ``energy``, ``forces``, and ``stress`` that
    :meth:`~nvalchemi.dynamics.base.BaseDynamics.compute` overwrote with the
    propagated model's own predictions. Keeping those last three would store
    the student's self-labels under the names a reference target uses, which a
    mixed training batch cannot tell apart from ground truth. The live batch
    keeps all of it.

    Graphs a lifecycle has graduated — ``status`` at or above the propagator's
    ``exit_status`` — are left out of that copy, and out of the teacher pass
    behind it. They are frozen, so every later capture of the segment would
    store the same structure again and score it again to do so, while
    :class:`~nvalchemi.dynamics.hooks.ConvergedSnapshotHook` stores each of them
    once at the step it converged. Narrowing before the teacher rather than
    after it is what keeps the expensive model off frozen structures in the
    heterogeneous case this lifecycle exists for, where most of a batch is
    finished for most of a segment. A frame carrying no ``status``, which is
    every frame of a run without status migration, is labeled and captured
    whole, and so is every frame of a run that keeps no sink.

    Parameters
    ----------
    teacher_scorer : TeacherScorer
        Scorer producing the teacher signals for each labeled frame. A scorer
        publishing ``label_fields`` makes the idempotency check below exact
        from the first dispatch.
    sink : DataSink | None, optional
        Sink each labeled frame is copied into. Default ``None`` (label the
        live batch and write nothing).
    frequency : int, optional
        Label every ``frequency`` steps; the dynamics registry does the
        gating. Default ``1`` (every step).

    Attributes
    ----------
    teacher_scorer : TeacherScorer
        The scorer signals are requested from.
    sink : DataSink | None
        The sink labeled frames are copied into, if any.
    frequency : int
        Labeling frequency in steps.
    stage : DynamicsStage
        Fixed to ``AFTER_STEP``.

    Raises
    ------
    ValueError
        If the scorer declares, or returns, a field outside the ``teacher_*``
        namespace.

    See Also
    --------
    nvalchemi.dynamics.hooks.SnapshotHook : Capture frames without labeling them.
    nvalchemi.training.distillation.label_dataset : Label a dataset offline.

    Examples
    --------
    >>> from nvalchemi.dynamics.sinks import HostMemory
    >>> from nvalchemi.training.distillation import (
    ...     InProcessTeacherScorer,
    ...     TeacherLabelHook,
    ... )
    >>> scorer = InProcessTeacherScorer(teacher, ["energy", "forces"])  # doctest: +SKIP
    >>> sink = HostMemory(capacity=10_000)  # doctest: +SKIP
    >>> hook = TeacherLabelHook(scorer, sink=sink, frequency=10)  # doctest: +SKIP
    >>> dynamics.register_hook(hook)  # doctest: +SKIP

    Notes
    -----
    Do not confuse this hook with the labeling seam inside
    :class:`~nvalchemi.training.distillation.DistillationStrategy`, which is a
    private ``TrainingStage.BEFORE_FORWARD`` hook: that one labels a batch the
    *trainer* is about to run a forward pass on, this one labels a frame the
    *propagator* just produced. They are different stage enums on different
    engines and both can be active in one on-policy run.

    A labeling cadence and a forced label are also kept from labeling the same
    stretch of trajectory twice. The dynamics registry gates on the step count
    before it is incremented, so a ``frequency`` of ``f`` fires at steps
    ``0, f, 2f, ...`` while a segment's forced last frame is step ``S - 1``;
    with ``S`` a multiple of ``f`` the two land one step apart at every segment
    boundary, which would pay for two teacher passes over what is effectively
    one frame. A registry dispatch on the step immediately after a labeled one
    is therefore passed over whenever ``frequency`` is above ``1``. The forced
    frame is the one that wins, because it is the frame the segment ends on and
    the one training sees; a forced call is never passed over itself, so an
    early-exiting segment and a run's final frame are always labeled.

    Labeling is idempotent per step: a frame already carrying every field the
    scorer writes, at the step it was labeled on, is passed over, so
    dispatching the hook twice on one state — re-entering a chunked
    :meth:`~nvalchemi.dynamics.base.BaseDynamics.run`, or force-labeling a
    segment's last frame — never pays for a second teacher pass and never
    duplicates it into the sink. The step count is what makes the check safe on
    a live batch, whose ``teacher_*`` fields stay attached while the positions
    underneath them move, and it is the whole check on a step narrowed to the
    graphs still moving, whose labels went to the stored copy rather than to
    the batch left behind. Which fields those are comes from
    :func:`~nvalchemi.training.distillation.scorer_fields` — a ``label_fields``
    declaration, or the built-in signals behind a scorer's signal names — and
    from the first pass's own labels for a scorer publishing neither, so a
    scorer with a signal name of its own is re-scored exactly once, on the
    dispatch that reveals what it writes, and skipped on every re-dispatch
    after that. The sink write is gated on the step count alone, so no frame is
    stored twice whatever the scorer declares.

    The teacher runs with autocast disabled whatever precision context the
    propagator establishes, so a frame labeled inside a mixed-precision
    generation phase carries exactly the labels a full-precision one would, and
    matches what :func:`~nvalchemi.training.distillation.label_dataset` would
    have written offline.

    ``requires_grad`` hygiene is the scorer's contract, not this hook's:
    :meth:`~nvalchemi.training.distillation.TeacherScorer.label` snapshots the
    flags of ``positions`` and the teacher's autograd inputs and restores them
    before returning, which leaves the batch exactly as
    :meth:`~nvalchemi.dynamics.base.BaseDynamics.compute` left it — flags
    cleared — so the next step's in-place updates stay legal.
    """

    def __init__(
        self,
        teacher_scorer: TeacherScorer,
        sink: DataSink | None = None,
        frequency: int = 1,
    ) -> None:
        """Resolve the fields the scorer populates, when they can be known."""
        self.teacher_scorer = teacher_scorer
        self.sink = sink
        self.frequency = frequency
        self.stage = DynamicsStage.AFTER_STEP
        self._teacher_fields: tuple[str, ...] | None = scorer_fields(teacher_scorer)
        if self._teacher_fields is not None:
            _reject_foreign_fields(self._teacher_fields, "A scorer's label_fields")
        self._labeled_step: int | None = None

    @property
    def labeled_step(self) -> int | None:
        """Propagator step this hook last labeled a frame on, or ``None``.

        A segment loop reads it to tell a step the cadence already covered from
        one it skipped, which is what lets a closing dispatch against the step
        the propagator finished on store the frame exactly once.

        It records the step a frame was actually *labeled* on, which is not
        every step the propagator took: a step whose graphs had all graduated
        leaves it unchanged, because nothing was labeled and the segment loop's
        budget-graduate capture keys its own idempotence off this value and
        still has to run for that frame. A consumer deriving "the last step of
        the previous segment" from a step count is therefore over-estimating
        whenever the segment ended with nothing still moving.
        """
        return self._labeled_step

    @torch.compiler.disable
    def _label_frame(
        self,
        batch: Batch,
        step_count: int,
        *,
        exit_status: int | None = None,
        forced: bool = False,
    ) -> None:
        """Label the graphs of *batch* still moving, once per step.

        The narrowed frame is cut before the teacher sees it, so neither the
        labels a graduated graph would get nor the copy they would ride into
        the sink is ever paid for. That copy is also what the run keeps, so the
        live batch is deliberately left unlabeled whenever one is cut —
        scattering per-graph labels back into it would cost the pass just
        avoided — and a re-dispatch at that step recognizes its own work from
        the step count rather than from fields the batch never received.

        *forced* marks the out-of-band call a caller makes to label a frame the
        cadence did not land on — the last frame of an on-policy segment. It is
        never passed over by the adjacency rule, and never made by the dynamics
        registry.
        """
        if (
            not forced
            and self.frequency > 1
            and self._labeled_step is not None
            and step_count == self._labeled_step + 1
        ):
            return
        active = _active_graphs(batch, exit_status) if self.sink is not None else None
        if active is not None and active.numel() == 0:
            return
        stored = step_count == self._labeled_step
        if stored and (
            active is not None
            or (
                self._teacher_fields is not None
                and all(field in batch for field in self._teacher_fields)
            )
        ):
            return
        frame = batch if active is None else self._captured_frame(batch, active)
        with torch.autocast(device_type=batch.device.type, enabled=False):
            labels = self.teacher_scorer.label(frame)
        _reject_foreign_fields(labels, "Teacher labels")
        _attach_teacher_labels(frame, labels)
        if self._teacher_fields is None:
            self._teacher_fields = tuple(sorted(labels))
        self._labeled_step = step_count
        if self.sink is None or stored:
            return
        self.sink.write(frame if active is not None else self._captured_frame(batch))

    def _captured_frame(
        self, batch: Batch, active: torch.Tensor | None = None
    ) -> Batch:
        """Return a copy of *batch* holding nothing run-local.

        The dropped fields leave the live batch only for the duration of the
        copy and are put back before returning, so the next step still finds
        the neighbor tensors and the predictions it reuses. Cloning first and
        deleting afterwards would be simpler, but it would allocate a full
        copy of the neighbor list — usually the largest tensor in a frame — on
        the propagation device just to throw it away.

        A whole-frame :meth:`~nvalchemi.data.Batch.clone` is the operation
        wanted while every graph is still being propagated. Once a lifecycle
        graduates graphs out of a relaxation batch, *active* names the ones
        still moving and the copy narrows to them: a graduated graph is frozen,
        so the cadence would otherwise store the same structure once per
        remaining step of the segment, and it has already been stored once by
        the converged route. An edge group emptied by dropping the neighbor
        list is removed as well, so a store does not record edges that no array
        backs.

        The copy is taken under :func:`torch.no_grad`. A fused propagator holds
        its autograd inputs tracking for the whole step, hooks included, so a
        plain copy taken at ``AFTER_STEP`` would carry the step's graph into the
        sink and the first training pass over a stored frame would try to
        backward through a graph the propagator has already freed.
        """
        dropped = _run_local_keys()
        detached: list[tuple[BaseLevelStorage, str, torch.Tensor]] = []
        try:
            for group in batch._storage.groups.values():
                for key in [name for name in group.keys() if name in dropped]:
                    detached.append((group, key, group[key]))
                    del group[key]
            with torch.no_grad():
                frame = batch.clone() if active is None else batch.index_select(active)
        finally:
            for group, key, tensor in detached:
                group[key] = tensor
        return _strip_replay_frame(frame)

    def __call__(self, ctx: DynamicsContext, stage: Enum) -> None:  # noqa: ARG002
        """Label the frame the propagator has just resolved."""
        self._label_frame(
            ctx.batch,
            ctx.step_count,
            exit_status=getattr(ctx.workflow, "exit_status", None),
        )


class _ConvergedFrameHook(ConvergedSnapshotHook):
    """Capture each graduating structure once, on the step it stopped moving.

    Graduation is a status transition, and every propagator publishes it at
    ``AFTER_STEP``: this hook is registered there, immediately behind the
    lifecycle's criterion, and writes the graphs whose ``status`` has just
    reached the propagator's ``exit_status``.
    :class:`~nvalchemi.dynamics.hooks.ConvergedSnapshotHook`'s own
    ``ON_CONVERGE`` stage cannot serve here:
    :class:`~nvalchemi.dynamics.FusedStage` dispatches it on its
    sub-stages alone, never on itself, so a hook the lifecycle registers on a
    fused propagator would never fire while its criterion kept graduating
    structures — every relaxed minimum lost, silently, to a capture route that
    was never called.

    Reading the transition rather than the criterion's mask is also what makes
    the capture exact. ``ON_CONVERGE`` fires with every graph the criterion
    currently accepts, not with the ones that just reached it, so a bare
    ``ConvergedSnapshotHook`` rewrites a converged structure on every remaining
    step of the segment: status migration freezes the graph, which keeps its
    forces exactly where the criterion found them. This subclass remembers what
    it has already written and passes only the newly graduated graphs down.

    The frames are captured raw, unlabeled, and the segment loop scores them
    when it drains the sink — one teacher pass over a segment's graduates
    rather than one per convergence step, which is what keeps the teacher's
    batch size independent of the propagated one.

    Parameters
    ----------
    sink : DataSink
        Sink converged frames are written to.

    Notes
    -----
    A fused sub-stage that graduates on its own ``n_steps`` budget rather than
    on a criterion migrates after the fused ``AFTER_STEP`` dispatch, so the
    status this hook reads on the step the budget runs out is still the moving
    one. A later step of the same chunk captures them instead, frozen and so on
    the same frame; a budget that graduates every remaining graph ends the
    chunk there and leaves no later step, which is why the segment loop
    dispatches this hook once more when the chunk returns.

    The write is taken under :func:`torch.no_grad` for the reason the path
    route's copy is: a fused propagator keeps its autograd inputs tracking
    across its hooks, so a frame captured there would otherwise reach the
    buffer still attached to the step's graph.
    """

    def __init__(self, sink: DataSink) -> None:
        """Start with nothing captured, listening for the status transition."""
        super().__init__(sink=sink, stage=DynamicsStage.AFTER_STEP)
        self._captured: torch.Tensor | None = None

    def reset(self) -> None:
        """Forget what was captured, after a refill changed the batch."""
        self._captured = None

    def __call__(self, ctx: DynamicsContext, stage: Enum) -> None:  # noqa: ARG002
        """Write the graphs that graduated on this step, and only those."""
        status = _graph_status(ctx.batch)
        exit_status = getattr(ctx.workflow, "exit_status", None)
        if status is None or exit_status is None:
            return
        graduated = status >= exit_status
        if self._captured is None or self._captured.numel() != graduated.numel():
            self._captured = torch.zeros_like(graduated)
        fresh = graduated & ~self._captured
        self._captured |= graduated
        with torch.no_grad():
            self._write_converged(ctx.batch, fresh)
