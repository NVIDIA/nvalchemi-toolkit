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
"""Dynamics hooks that capture on-policy frames as a propagator produces them."""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeAlias

import torch
from jaxtyping import Bool

from nvalchemi.dynamics.base import BaseDynamics, DynamicsStage
from nvalchemi.dynamics.hooks.safety import nonfinite_graph_mask
from nvalchemi.dynamics.hooks.snapshot import ConvergedSnapshotHook
from nvalchemi.training.distillation._attach import (
    _attach_teacher_labels,
    _prune_empty_edges,
)
from nvalchemi.training.distillation.scoring import (
    _NEIGHBOR_KEYS,
    _reject_foreign_fields,
    scorer_fields,
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from enum import Enum

    from nvalchemi.data import Batch
    from nvalchemi.dynamics.sinks import DataSink
    from nvalchemi.hooks import DynamicsContext
    from nvalchemi.training.distillation.scoring import TeacherLabels, TeacherScorer

    _DivergencePredicate: TypeAlias = Callable[[Batch], Bool[torch.Tensor, "G"]]

__all__ = ["TeacherLabelHook", "nonfinite_divergence"]

_PREDICTION_KEYS = frozenset(BaseDynamics._OUTPUT_KEY_TO_BATCH_ATTR.values())
"""Batch fields a propagator overwrites with the propagated model's predictions."""


def _run_local_keys(dynamics: BaseDynamics) -> frozenset[str]:
    """Return the fields of a live frame that mean nothing outside its run.

    The bookkeeping part is read from *dynamics* at call time through
    :meth:`~nvalchemi.dynamics.base.BaseDynamics.bookkeeping_keys`, which
    walks the composition, because
    :meth:`~nvalchemi.dynamics.base.BaseDynamics.register_bookkeeping_key`
    grows the registries as stages are built. For example, a fused stage
    registers one step counter per sub-stage, and keeps a ``reprime_pending``
    flag of its own. A hook therefore reads the propagator it is registered
    on, and belongs on the root of a composition so that root's own keys are
    included.
    """
    return _NEIGHBOR_KEYS | _PREDICTION_KEYS | dynamics.bookkeeping_keys()


def _score_and_attach(scorer: TeacherScorer, frame: Batch) -> TeacherLabels:
    """Label *frame* in place with *scorer*, refusing a label outside ``teacher_*``.

    The scorer is called inside the caller's autocast region, if any. Label
    precision is the scorer's decision, and
    :class:`~nvalchemi.training.distillation.InProcessTeacherScorer` disables
    autocast unless its ``autocast`` setting says otherwise. A label outside
    ``teacher_*`` is refused before it can overwrite propagator state.

    Returns
    -------
    TeacherLabels
        The attached labels, so a caller can resolve the fields the scorer
        writes.

    Raises
    ------
    ValueError
        If the scorer returns a field outside ``teacher_*``.
    """
    labels = scorer.label(frame)
    _attach_teacher_labels(frame, labels)
    return labels


def _strip_replay_frame(frames: Batch, dynamics: BaseDynamics) -> Batch:
    """Reduce *frames* to the replay-frame contract, in place.

    A frame captured from the propagator carries run-local fields: the
    ephemeral neighbor tensors, the dynamics bookkeeping, and the ``energy``,
    ``forces``, and ``stress`` the propagated model wrote. A replay frame keeps
    only the structure, the propagator state that travels with it, and the
    ``teacher_*`` labels. A stored frame therefore never offers the propagated
    model's own prediction under a reference target's name.

    Parameters
    ----------
    frames : Batch
        Frames to strip, mutated in place.
    dynamics : BaseDynamics
        Propagator that wrote *frames*, whose bookkeeping keys are dropped.

    Returns
    -------
    Batch
        The same object, holding nothing run-local.
    """
    dropped = _run_local_keys(dynamics)
    for key in dropped:
        if key in frames:
            del frames[key]
    if frames.keys is not None:
        for names in frames.keys.values():
            names -= dropped
    _prune_empty_edges(frames)
    return frames


nonfinite_divergence = nonfinite_graph_mask
"""Default :attr:`~nvalchemi.training.distillation.OnPolicyConfig.divergence`: :func:`~nvalchemi.dynamics.hooks.nonfinite_graph_mask` over positions and forces."""


class TeacherLabelHook:
    """Label the live propagator frame with teacher signals, inline.

    The hook runs at ``AFTER_STEP``. It attaches every signal its scorer
    produces to the batch being propagated, each at the level its signal
    declares. It can also copy the labeled frame into a
    :class:`~nvalchemi.dynamics.sinks.DataSink`. The live batch keeps the
    ``energy`` and ``forces`` the propagator wrote, because they drive the
    next step. The copy drops them, the ephemeral neighbor tensors, and the
    dynamics bookkeeping. A stored frame is therefore a training sample rather
    than a propagator state, and it never carries the propagated model's own
    prediction under a reference target's name. A scorer that declares or
    returns a field outside ``teacher_*`` is rejected, so it cannot overwrite
    propagator state.

    With ``exit_status`` set, graphs whose ``status`` has reached it are left
    out of that copy and out of the teacher pass that labels it. A lifecycle
    freezes those graphs at ``exit_status`` and stores each one once, through
    a separate converged-frame route. Capturing them here would store and
    score the same structure again on every later capture of the segment.
    The hook records which graphs it stored on which step. The
    converged-frame route asks :meth:`stored_graphs` before it writes, so a
    structure that a step budget graduates right after this hook stored its
    frame is not stored twice. Without ``exit_status``, every graph is
    captured, frozen or not, because nothing else keeps the final frame of a
    graduation the propagator manages itself. A frame that carries no
    ``status`` is labeled and captured whole either way, and a hook without a
    sink labels every frame whole.

    Labeling is idempotent per step. A cadence dispatch right after a forced
    label is skipped, so the teacher is not paid twice for a segment's last
    frame and the next cadence step. See :ref:`training-distillation-api`.

    Parameters
    ----------
    teacher_scorer : TeacherScorer
        Scorer that produces the teacher signals. A scorer that publishes
        ``label_fields`` makes the idempotency check exact from the first
        dispatch.
    sink : DataSink | None, optional
        Sink each labeled frame is copied into. Default ``None``.
    frequency : int, optional
        Label every ``frequency`` steps. Default ``1``.
    exit_status : int | None, optional
        Propagator status at which a graph counts as graduated. A graduated
        graph is stored by another route, so this hook leaves it out. Default
        ``None`` labels and stores every graph.
    labeled_step : int | None, optional
        Step a hook this one replaces last labeled on, so the adjacency rule
        treats the following step as covered from the first dispatch. A
        segment loop rebuilt mid-run passes the step its interrupted
        predecessor recorded. Default ``None`` starts with nothing labeled.

    Raises
    ------
    ValueError
        If the scorer declares, or returns, a field outside ``teacher_*``.

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
    >>> dynamics.register_hook(TeacherLabelHook(scorer, sink=sink, frequency=10))  # doctest: +SKIP

    Notes
    -----
    This hook is not the labeling seam inside
    :class:`~nvalchemi.training.distillation.DistillationStrategy`, which is a
    training hook that labels batches on their way into a forward pass. The
    two run on different engines, and both are active in an on-policy run.
    Label precision is the scorer's decision, and the hook opens no autocast
    region of its own. The built-in scorer disables autocast unless its
    ``autocast`` setting says otherwise, so a frame it labels during a
    mixed-precision generation phase matches what
    :func:`~nvalchemi.training.distillation.label_dataset` writes offline.
    ``requires_grad`` handling is the scorer's responsibility, and the scorer
    leaves the batch as :meth:`~nvalchemi.dynamics.base.BaseDynamics.compute`
    left it.
    """

    def __init__(
        self,
        teacher_scorer: TeacherScorer,
        sink: DataSink | None = None,
        frequency: int = 1,
        exit_status: int | None = None,
        labeled_step: int | None = None,
    ) -> None:
        """Resolve the fields the scorer populates, when they can be known."""
        self.teacher_scorer = teacher_scorer
        self.sink = sink
        self.frequency = frequency
        self.exit_status = exit_status
        self.stage = DynamicsStage.AFTER_STEP
        self._teacher_fields: tuple[str, ...] | None = scorer_fields(teacher_scorer)
        if self._teacher_fields is not None:
            _reject_foreign_fields(self._teacher_fields, "A scorer's label_fields")
        self._labeled_step: int | None = labeled_step
        self._stored: tuple[int, torch.Tensor | None] | None = None

    @property
    def labeled_step(self) -> int | None:
        """Propagator step this hook last labeled a frame on, or ``None``.

        A step on which every graph had already graduated leaves it
        unchanged, because nothing was labeled. A segment loop uses this to
        tell a step the cadence covered from one it skipped.
        """
        return self._labeled_step

    def stored_graphs(
        self, step_count: int, batch: Batch
    ) -> Bool[torch.Tensor, "G"] | None:
        """Return the graphs of *batch* whose frame this hook stored on *step_count*.

        ``None`` when the hook stored nothing on that step. The converged-frame
        route calls this at ``ON_GRADUATE`` and skips a graph whose final frame
        this hook already stored. That happens when a step budget graduates
        the graph after this hook ran on the same step.
        """
        if self._stored is None or self._stored[0] != step_count:
            return None
        stored = torch.ones(batch.num_graphs, dtype=torch.bool, device=batch.device)
        active = self._stored[1]
        if active is not None:
            stored.zero_()
            stored[active] = True
        return stored

    @torch.compiler.disable
    def _label_frame(
        self,
        batch: Batch,
        step_count: int,
        *,
        dynamics: BaseDynamics,
        forced: bool = False,
    ) -> None:
        """Label the graphs of *batch* that are still moving, once per step.

        The frame is narrowed to the graphs below ``exit_status`` before the
        teacher sees it. The status is read as it is now, not from the
        step-start mask in the hook context, so a graph the criterion froze
        earlier in this step is already left out. The run therefore pays
        neither for labeling a graduated graph nor for copying it into the
        sink. Whenever a graph is cut, the live batch itself is left
        unlabeled. A second dispatch at the same step then recognizes its own
        work from the step count, because the batch never received the label
        fields.

        *dynamics* is the propagator that wrote *batch*; the copy handed to
        the sink drops its bookkeeping. *forced* marks an out-of-band call
        that labels a frame the cadence did not land on, such as the last
        frame of an on-policy segment. The adjacency rule never skips a forced
        call, and the dynamics registry never makes one.
        """
        if (
            not forced
            and self.frequency > 1
            and self._labeled_step is not None
            and step_count == self._labeled_step + 1
        ):
            return
        active = None
        if self.sink is not None and self.exit_status is not None:
            moving = BaseDynamics.active_graph_mask(batch, self.exit_status)
            if moving is not None and not bool(moving.all()):
                active = torch.where(moving)[0]
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
        frame = (
            batch if active is None else self._captured_frame(batch, dynamics, active)
        )
        labels = _score_and_attach(self.teacher_scorer, frame)
        if self._teacher_fields is None:
            self._teacher_fields = tuple(sorted(labels))
        self._labeled_step = step_count
        if self.sink is None or stored:
            return
        self.sink.write(
            frame if active is not None else self._captured_frame(batch, dynamics)
        )
        self._stored = (step_count, active)

    def _captured_frame(
        self, batch: Batch, dynamics: BaseDynamics, active: torch.Tensor | None = None
    ) -> Batch:
        """Return a copy of *batch* holding nothing run-local.

        The copy leaves out every run-local field, read from *dynamics*, and
        the live batch keeps the neighbor tensors and predictions the next
        step reads. An edge
        group left empty is removed too, so a store never records edges that
        no array backs. *active*, when given, narrows the copy to the graphs
        still moving, for a lifecycle that graduates graphs out of the batch.
        The copy is taken under :func:`torch.no_grad`. A fused propagator
        keeps its autograd inputs tracking gradients across its hooks, so a
        stored frame would otherwise carry the step's autograd graph into the
        first training pass.
        """
        dropped = _run_local_keys(dynamics)
        with torch.no_grad():
            frame = (
                batch.clone(drop=dropped)
                if active is None
                else batch.index_select(active, drop=dropped)
            )
        _prune_empty_edges(frame)
        return frame

    def __call__(self, ctx: DynamicsContext, stage: Enum) -> None:  # noqa: ARG002
        """Label the frame the propagator has just resolved."""
        self._label_frame(ctx.batch, ctx.step_count, dynamics=ctx.workflow)


class _DivergenceHook:
    """Evaluate the divergence predicate once per step and freeze what it flags.

    A diverged relaxation never converges, because every comparison with NaN
    is false. Without this hook, nothing would migrate its status, the path
    route would keep storing and labeling it, and its labels would reach the
    loss. Freezing the graph at the propagator's ``exit_status`` takes it out
    of the step and out of both capture routes. The segment boundary then
    retires and backfills it like a converged graph. An
    :class:`~nvalchemi.training.distillation.AdmissionPolicy` that refuses
    non-finite frames at the buffer would achieve the exclusion alone. The
    lifecycle still freezes the graph itself, because freezing also stops
    propagating and labeling it.

    The lifecycle calls the predicate only here, once per step. Each verdict
    is ORed into :attr:`diverged`. The converged-frame hook and the segment
    boundary read that record and never call the predicate themselves. A
    graph that a stateful predicate flags on one step and not on the next
    therefore stays out of the converged route and still counts as diverged
    at the boundary. :meth:`reset` clears the record once a refill has
    changed the batch's rows.

    Parameters
    ----------
    divergence : Callable[[Batch], Bool[torch.Tensor, "G"]], optional
        Predicate flagging the diverged graphs of the live frame. Its verdict
        is moved onto the batch's device before it is recorded, so a predicate
        that judges on the host is accepted for a frame on an accelerator.
        Default :func:`nonfinite_divergence`.

    Raises
    ------
    TypeError
        If the predicate returns something other than a tensor.
    ValueError
        If the tensor is not boolean or does not carry one entry per graph.
    """

    frequency = 1
    stage = DynamicsStage.AFTER_STEP

    def __init__(self, divergence: _DivergencePredicate = nonfinite_divergence) -> None:
        """Freeze the graphs *divergence* flags, starting with none recorded."""
        self.divergence = divergence
        self._diverged: Bool[torch.Tensor, "G"] | None = None

    @property
    def diverged(self) -> Bool[torch.Tensor, "G"] | None:
        """Graphs the predicate has flagged since the last reset, or ``None`` before the first verdict."""
        return self._diverged

    def reset(self) -> None:
        """Forget which graphs diverged, after a refill changed the batch."""
        self._diverged = None

    def _flags(self, batch: Batch) -> Bool[torch.Tensor, "G"]:
        """Evaluate the predicate on *batch* and return its one-boolean-per-graph verdict on the batch's device."""
        flags = self.divergence(batch)
        if not isinstance(flags, torch.Tensor):
            raise TypeError(
                "The divergence predicate must return a boolean tensor with one flag "
                f"per graph; got {type(flags).__name__!r}."
            )
        if flags.dtype != torch.bool or flags.shape != (batch.num_graphs,):
            raise ValueError(
                "The divergence predicate must return one boolean per graph; got "
                f"shape={tuple(flags.shape)!r} of dtype {flags.dtype!r}, expected "
                f"({batch.num_graphs},) of torch.bool."
            )
        return flags.to(batch.device)

    def __call__(self, ctx: DynamicsContext, stage: Enum) -> None:  # noqa: ARG002
        """Record the predicate's verdict and migrate the flagged graphs to the exit status."""
        exit_status = getattr(ctx.workflow, "exit_status", None)
        if exit_status is None:
            return
        moving = BaseDynamics.active_graph_mask(ctx.batch, exit_status)
        if moving is None:
            return
        flags = self._flags(ctx.batch)
        if self._diverged is None or self._diverged.numel() != flags.numel():
            self._diverged = torch.zeros_like(flags)
        self._diverged |= flags
        status = ctx.batch.status
        column = status.view(-1) if status.dim() == 2 else status
        column[: ctx.batch.num_graphs].masked_fill_(flags & moving, exit_status)


class _ConvergedFrameHook(ConvergedSnapshotHook):
    """Capture each graduating structure once, on the step it stopped moving.

    The hook listens at ``ON_GRADUATE``, the stage every
    :class:`~nvalchemi.dynamics.base.BaseDynamics` and
    :class:`~nvalchemi.dynamics.FusedStage` propagator dispatches with
    ``ctx.graduated_mask`` marking the graphs that graduated during the step.
    A :class:`~nvalchemi.distributed.DomainParallel` propagator dispatches no
    such stage, so :class:`~nvalchemi.training.distillation.OnPolicyConfig`
    refuses one with a criterion. A graph graduates when its ``status``
    reaches ``exit_status``, whether the lifecycle's criterion, a fused
    sub-stage's step budget, or the divergence hook moved it. The parent's
    ``ON_CONVERGE`` stage does not work here, for two reasons.
    :class:`~nvalchemi.dynamics.FusedStage` dispatches it on its sub-stages
    only. It also fires with every graph the criterion currently accepts, not
    only the ones that just reached it. The frames are captured without
    teacher labels. The segment loop labels them in one teacher pass when it
    drains the sink, which keeps the teacher's batch size independent of the
    propagated batch size. A graph the divergence hook has
    recorded as diverged is never written, because it diverged rather than
    converged. A graph whose frame the path route stored on the same step is
    not written again.

    Parameters
    ----------
    sink : DataSink
        Sink converged frames are written to.
    divergence : _DivergenceHook
        Hook recording which graphs diverged, registered ahead of this one.
    path : TeacherLabelHook | None, optional
        Labeling hook of the path route, asked which graphs it stored on the
        step. Default ``None`` skips that check, so every graduated graph that
        did not diverge is written.

    Notes
    -----
    The write runs under :func:`torch.no_grad`, because a fused propagator
    keeps its autograd inputs tracking gradients across its hooks.
    """

    def __init__(
        self,
        sink: DataSink,
        divergence: _DivergenceHook,
        path: TeacherLabelHook | None = None,
    ) -> None:
        """Listen at ``ON_GRADUATE`` for the graphs that graduate."""
        super().__init__(sink=sink, stage=DynamicsStage.ON_GRADUATE)
        self.divergence = divergence
        self.path = path

    def __call__(self, ctx: DynamicsContext, stage: Enum) -> None:  # noqa: ARG002
        """Write the graphs that graduated on this step, and only those."""
        fresh = ctx.graduated_mask
        if fresh is None:
            return
        diverged = self.divergence.diverged
        if diverged is not None:
            fresh = fresh & ~diverged
        if self.path is not None:
            stored = self.path.stored_graphs(ctx.step_count, ctx.batch)
            if stored is not None:
                fresh = fresh & ~stored
        with torch.no_grad():
            self._write_converged(ctx.batch, fresh)
