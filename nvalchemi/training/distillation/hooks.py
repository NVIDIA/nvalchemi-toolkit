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

from typing import TYPE_CHECKING, TypeAlias

import torch
from jaxtyping import Bool

from nvalchemi.dynamics.base import BaseDynamics, DynamicsStage
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


def _run_local_keys() -> frozenset[str]:
    """Return the fields of a live frame that mean nothing outside its run.

    Read at call time rather than at import time, because
    :meth:`~nvalchemi.dynamics.base.BaseDynamics.register_bookkeeping_key` grows
    the bookkeeping registry as stages are built — a fused stage registers one
    step counter per sub-stage.
    """
    return _NEIGHBOR_KEYS | _PREDICTION_KEYS | frozenset(BaseDynamics._bookkeeping_keys)


def _score_and_attach(scorer: TeacherScorer, frame: Batch) -> TeacherLabels:
    """Label *frame* in place with *scorer*, under the guards every labeling route shares.

    The teacher runs with autocast disabled, so a frame labeled inside a
    mixed-precision generation phase matches what
    :func:`~nvalchemi.training.distillation.label_dataset` writes offline, and a
    label outside ``teacher_*`` is refused before it can overwrite propagator
    state.

    Returns
    -------
    TeacherLabels
        The labels attached, for a caller resolving the fields a scorer writes.

    Raises
    ------
    ValueError
        If the scorer returns a field outside ``teacher_*``.
    """
    with torch.autocast(device_type=frame.device.type, enabled=False):
        labels = scorer.label(frame)
    _attach_teacher_labels(frame, labels)
    return labels


def _strip_replay_frame(frames: Batch) -> Batch:
    """Reduce *frames* to the replay-frame contract, in place.

    A frame captured off the propagator carries the run with it: the ephemeral
    neighbor tensors, the dynamics bookkeeping, and the ``energy``, ``forces``,
    and ``stress`` the propagated model wrote. A replay frame keeps only the
    structure, the propagator state travelling with it, and the ``teacher_*``
    labels, so a stored frame never offers a self-label under a reference
    target's name.

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
    for key in dropped:
        if key in frames:
            del frames[key]
    if frames.keys is not None:
        for names in frames.keys.values():
            names -= dropped
    _prune_empty_edges(frames)
    return frames


def _graph_status(batch: Batch) -> torch.Tensor | None:
    """Return one status per graph of *batch*, or ``None`` when it carries none.

    Bookkeeping is stored as a column, and an inflight batch keeps rows past the
    graphs it currently holds, so the stored field is flattened and cut to the
    live graphs before it is compared against an exit status.
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
        is below it — which is also the answer for a frame carrying no status.
    """
    status = _graph_status(batch)
    if status is None or exit_status is None:
        return None
    active = status < exit_status
    if bool(active.all()):
        return None
    return torch.where(active)[0]


def nonfinite_divergence(batch: Batch) -> Bool[torch.Tensor, "G"]:
    """Flag the graphs of *batch* whose positions or forces are not finite.

    The default divergence predicate of the on-policy relaxation lifecycle,
    and the contract any predicate set as
    :attr:`~nvalchemi.training.distillation.OnPolicyConfig.divergence` meets:
    one boolean per graph, on the batch's device, ``True`` where the trajectory
    has left the region the student can be trusted in. A graph is flagged here
    when any of its positions, or any of its forces when the batch carries
    them, is NaN or infinite.

    Parameters
    ----------
    batch : Batch
        Live propagator frame.

    Returns
    -------
    Bool[torch.Tensor, "G"]
        One flag per graph of *batch*, set where it diverged.

    Examples
    --------
    >>> from nvalchemi.training.distillation import nonfinite_divergence
    >>> nonfinite_divergence(batch)  # doctest: +SKIP
    tensor([False,  True, False])
    """
    finite = torch.isfinite(batch.positions).all(dim=-1)
    forces = getattr(batch, "forces", None)
    if forces is not None:
        finite &= torch.isfinite(forces).all(dim=-1)
    diverged = torch.zeros(batch.num_graphs, dtype=torch.bool, device=batch.device)
    diverged[batch.batch_idx.long()[~finite]] = True
    return diverged


def _checked_divergence(
    divergence: _DivergencePredicate, batch: Batch
) -> Bool[torch.Tensor, "G"]:
    """Return *divergence* over *batch*, held to one boolean per graph.

    Raises
    ------
    TypeError
        If the predicate returns something other than a tensor.
    ValueError
        If the tensor is not boolean or does not carry one entry per graph.
    """
    flags = divergence(batch)
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
    return flags


class TeacherLabelHook:
    """Label the live propagator frame with teacher signals, inline.

    An ``AFTER_STEP`` dynamics hook that attaches every signal its scorer
    produces to the batch being propagated, each at the level its signal
    declares, and optionally mirrors a copy of the labeled frame into a
    :class:`~nvalchemi.dynamics.sinks.DataSink`. The live batch keeps the
    ``energy`` and ``forces`` the propagator wrote, which drive the next step;
    the copy is stripped of them, of the ephemeral neighbor tensors, and of the
    dynamics bookkeeping, so a stored frame is a training sample rather than a
    propagator state and never carries a self-label under a reference target's
    name. A scorer that declares, or returns, a field outside ``teacher_*`` is
    refused rather than allowed to overwrite propagator state.

    Given ``exit_status``, graphs whose ``status`` has reached it are left out
    of that copy, and out of the teacher pass behind it: a lifecycle freezes
    them there and stores each once through a converged-frame route, so every
    later capture of the segment would store the same structure again and
    score it again to do so. Without it every graph is captured, frozen or
    not, since nothing else keeps a propagator-managed graduation's final
    frame; a frame carrying no ``status``, and every frame of a run that keeps
    no sink, is labeled and captured whole either way.

    Labeling is idempotent per step, and the cadence dispatch immediately after
    a forced label is passed over, so a segment's last frame and the next
    cadence step are not both paid for; see :ref:`training-distillation-api`.

    Parameters
    ----------
    teacher_scorer : TeacherScorer
        Scorer producing the teacher signals. One publishing ``label_fields``
        makes the idempotency check exact from the first dispatch.
    sink : DataSink | None, optional
        Sink each labeled frame is copied into. Default ``None``.
    frequency : int, optional
        Label every ``frequency`` steps. Default ``1``.
    exit_status : int | None, optional
        Propagator status at which a graph has graduated and is stored by
        another route, so this hook leaves it out. Default ``None`` (every
        graph is labeled and stored).

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
    This is not the labeling seam inside
    :class:`~nvalchemi.training.distillation.DistillationStrategy`, a training
    hook labeling batches on their way into a forward pass; the two run on
    different engines and both are active in an on-policy run. The teacher
    runs with autocast disabled, so a frame labeled inside a mixed-precision
    generation phase matches what
    :func:`~nvalchemi.training.distillation.label_dataset` writes offline, and
    ``requires_grad`` hygiene is the scorer's contract, which leaves the batch
    as :meth:`~nvalchemi.dynamics.base.BaseDynamics.compute` left it.
    """

    def __init__(
        self,
        teacher_scorer: TeacherScorer,
        sink: DataSink | None = None,
        frequency: int = 1,
        exit_status: int | None = None,
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
        self._labeled_step: int | None = None

    @property
    def labeled_step(self) -> int | None:
        """Propagator step this hook last labeled a frame on, or ``None``.

        A step whose graphs had all graduated leaves it unchanged, since
        nothing was labeled, which is how a segment loop tells a step the
        cadence covered from one it skipped.
        """
        return self._labeled_step

    @torch.compiler.disable
    def _label_frame(
        self, batch: Batch, step_count: int, *, forced: bool = False
    ) -> None:
        """Label the graphs of *batch* still moving, once per step.

        The frame is narrowed to the graphs below ``exit_status`` before the
        teacher sees it, so neither the labels a graduated graph would get nor
        the copy they would ride into the sink is paid for; the live batch is
        left unlabeled whenever one is cut, and a re-dispatch at that step
        recognizes its own work from the step count rather than from fields the
        batch never received.

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
        active = (
            _active_graphs(batch, self.exit_status) if self.sink is not None else None
        )
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
        labels = _score_and_attach(self.teacher_scorer, frame)
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

        The copy is taken first and stripped afterwards, so the live batch is
        never left without the neighbor tensors and predictions the next step
        reads. An edge group the strip emptied is removed too, so a store
        records no edges no array backs. *active* narrows the copy to the
        graphs still moving once a lifecycle graduates graphs out of a batch.
        The copy is taken under :func:`torch.no_grad`, because a fused
        propagator keeps its autograd inputs tracking across its hooks and a
        stored frame would otherwise carry the step's graph into the first
        training pass.
        """
        with torch.no_grad():
            frame = batch.clone() if active is None else batch.index_select(active)
        return _strip_replay_frame(frame)

    def __call__(self, ctx: DynamicsContext, stage: Enum) -> None:  # noqa: ARG002
        """Label the frame the propagator has just resolved."""
        self._label_frame(ctx.batch, ctx.step_count)


class _DivergenceHook:
    """Freeze a graph the divergence predicate flags, on the step it happened.

    A diverged relaxation never converges — every comparison is false under
    NaN — so nothing would migrate its status, the path route would keep
    storing and labeling it, and its labels would reach the loss. Freezing it
    at the propagator's ``exit_status`` takes it out of the step and out of
    both capture routes, and the segment boundary retires and backfills it like
    a converged one. The exclusion alone is also one
    :class:`~nvalchemi.training.distillation.AdmissionPolicy` refusing
    non-finite frames at the buffer; the lifecycle keeps its own freeze because
    it also stops propagating and labeling the graph.

    Parameters
    ----------
    divergence : Callable[[Batch], Bool[torch.Tensor, "G"]], optional
        Predicate flagging the diverged graphs of the live frame. Default
        :func:`nonfinite_divergence`.
    """

    frequency = 1
    stage = DynamicsStage.AFTER_STEP

    def __init__(self, divergence: _DivergencePredicate = nonfinite_divergence) -> None:
        """Freeze the graphs *divergence* flags."""
        self.divergence = divergence

    def __call__(self, ctx: DynamicsContext, stage: Enum) -> None:  # noqa: ARG002
        """Migrate the graphs the predicate flags to the exit status."""
        status = _graph_status(ctx.batch)
        exit_status = getattr(ctx.workflow, "exit_status", None)
        if status is None or exit_status is None:
            return
        diverged = _checked_divergence(self.divergence, ctx.batch) & (
            status < exit_status
        )
        status.masked_fill_(diverged, exit_status)


class _ConvergedFrameHook(ConvergedSnapshotHook):
    """Capture each graduating structure once, on the step it stopped moving.

    Graduation is a status transition, and every propagator publishes it at
    ``AFTER_STEP``: this hook is registered there, right behind the lifecycle's
    criterion, and writes the graphs whose ``status`` has just reached the
    propagator's ``exit_status``. The parent's ``ON_CONVERGE`` stage cannot
    serve, because :class:`~nvalchemi.dynamics.FusedStage` dispatches it on
    its sub-stages alone, and it fires with every graph the criterion currently
    accepts rather than the ones that just reached it, so a bare snapshot hook
    would rewrite a frozen structure on every remaining step. The frames are
    captured raw, and the segment loop labels them in one teacher pass when it
    drains the sink, which keeps the teacher's batch size independent of the
    propagated one. A graph the divergence predicate flags as it graduates is
    never written: it diverged rather than converged.

    Parameters
    ----------
    sink : DataSink
        Sink converged frames are written to.
    divergence : Callable[[Batch], Bool[torch.Tensor, "G"]], optional
        Predicate flagging the diverged graphs of the live frame. Default
        :func:`nonfinite_divergence`.

    Notes
    -----
    A fused sub-stage graduating on an ``n_steps`` budget migrates after the
    fused ``AFTER_STEP`` dispatch, so the status read here on the step the
    budget runs out is still the moving one; the segment loop dispatches this
    hook once more when the chunk returns. The write is taken under
    :func:`torch.no_grad`, since a fused propagator keeps its autograd inputs
    tracking across its hooks.
    """

    def __init__(
        self, sink: DataSink, divergence: _DivergencePredicate = nonfinite_divergence
    ) -> None:
        """Start with nothing captured, listening for the status transition."""
        super().__init__(sink=sink, stage=DynamicsStage.AFTER_STEP)
        self.divergence = divergence
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
        fresh = (
            graduated
            & ~self._captured
            & ~_checked_divergence(self.divergence, ctx.batch)
        )
        self._captured |= graduated
        with torch.no_grad():
            self._write_converged(ctx.batch, fresh)
