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
"""Dynamics hook that labels on-policy frames as a propagator produces them."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from nvalchemi.dynamics.base import BaseDynamics, DynamicsStage
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
    from enum import Enum

    from nvalchemi.data import Batch
    from nvalchemi.dynamics.sinks import DataSink
    from nvalchemi.hooks import DynamicsContext
    from nvalchemi.training.distillation.scoring import TeacherScorer

__all__ = ["TeacherLabelHook"]

_PREDICTION_KEYS = frozenset(BaseDynamics._OUTPUT_KEY_TO_BATCH_ATTR.values())
"""Batch fields a propagator overwrites with the propagated model's predictions."""


def _run_local_keys() -> frozenset[str]:
    """Return the fields of a live frame that mean nothing outside its run.

    The set is read at call time rather than at import time, because
    :meth:`~nvalchemi.dynamics.base.BaseDynamics.register_bookkeeping_key` grows
    the bookkeeping registry as stages are built. For example, a fused stage
    registers one step counter per sub-stage.
    """
    return _NEIGHBOR_KEYS | _PREDICTION_KEYS | frozenset(BaseDynamics._bookkeeping_keys)


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
    The teacher runs with autocast disabled, so a frame labeled during a
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

    @torch.compiler.disable
    def _label_frame(
        self, batch: Batch, step_count: int, *, forced: bool = False
    ) -> None:
        """Label *batch* unless it was already labeled at or just before *step_count*.

        *forced* marks an out-of-band call that labels a frame the cadence did
        not land on, such as the last frame of an on-policy segment. The
        adjacency rule never skips a forced call, and the dynamics registry
        never makes one.
        """
        if (
            not forced
            and self.frequency > 1
            and self._labeled_step is not None
            and step_count == self._labeled_step + 1
        ):
            return
        stored = step_count == self._labeled_step
        if (
            stored
            and self._teacher_fields is not None
            and all(field in batch for field in self._teacher_fields)
        ):
            return
        with torch.autocast(device_type=batch.device.type, enabled=False):
            labels = self.teacher_scorer.label(batch)
        _attach_teacher_labels(batch, labels)
        if self._teacher_fields is None:
            self._teacher_fields = tuple(sorted(labels))
        self._labeled_step = step_count
        if self.sink is not None and not stored:
            self.sink.write(self._captured_frame(batch))

    def _captured_frame(self, batch: Batch) -> Batch:
        """Return a labeled copy of *batch* holding nothing run-local.

        The copy is taken first and stripped afterwards, so the live batch is
        never left without the neighbor tensors and predictions the next step
        reads. An edge group left empty by the drop is removed too, so a store
        never records edges that no array backs.
        """
        dropped = _run_local_keys()
        frame = batch.clone()
        for key in dropped:
            if key in frame:
                del frame[key]
        if frame.keys is not None:
            for names in frame.keys.values():
                names -= dropped
        _prune_empty_edges(frame)
        return frame

    def __call__(self, ctx: DynamicsContext, stage: Enum) -> None:  # noqa: ARG002
        """Label the frame the propagator has just resolved."""
        self._label_frame(ctx.batch, ctx.step_count)
