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
"""Training hook protocol and context dataclass.

Provides two public objects:

:class:`TrainingContext`
    A mutable dataclass that carries all per-step state through the hook
    pipeline.  Hooks read from *and* write to this context (e.g. appending
    metrics, overriding losses).

:class:`TrainingHook`
    A :func:`~typing.runtime_checkable` :class:`~typing.Protocol` that
    every hook must satisfy.  A hook declares **which** stage(s) it fires
    at and **how often** via the ``frequency`` / ``stage`` attributes.
    Multi-stage hooks expose a ``stages`` frozenset instead of a single
    ``stage``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import torch
from tensordict import TensorDict

from nvalchemi.training._stages import TrainingStageEnum

if TYPE_CHECKING:
    from nvalchemi._typing import ModelOutputs
    from nvalchemi.data import Batch
    from nvalchemi.models.base import BaseModelMixin


@dataclass
class TrainingContext:
    """Mutable state bag passed to every training hook.

    The trainer populates fields before dispatching hooks at each stage;
    hooks may read or mutate the context freely (e.g. to inject custom
    metrics or modify the loss dictionary).

    Attributes
    ----------
    epoch : int
        Current epoch index (0-based).
    global_step : int
        Cumulative training step across all epochs.
    batch : Batch | None
        The current mini-batch, or ``None`` at epoch/validation boundaries.
    model_outputs : ModelOutputs | None
        Raw model outputs from the forward pass, or ``None`` before forward.
    losses : TensorDict
        Named loss components stored as a :class:`TensorDict` with
        ``batch_size=[]`` (e.g. ``TensorDict(energy=..., forces=...)``).
        Provides device management, element-wise math, and reduction
        ops (``losses.sum()``) out of the box.
    total_loss : torch.Tensor | None
        Scalar loss used for backpropagation, or ``None`` before loss computation.
    metrics : dict[str, float]
        Arbitrary scalar metrics accumulated during the step/epoch.
    extra : dict[str, Any]
        Free-form storage for hook-to-hook or hook-to-trainer communication.
    stage_counts : dict[TrainingStageEnum, int]
        Per-stage fire counter, auto-incremented by the trainer each time a
        stage is dispatched.  Hooks like :class:`TerminateOnStepsHook` read
        this to decide when to stop training.
    """

    epoch: int = 0
    global_step: int = 0
    batch: Batch | None = None
    model_outputs: ModelOutputs | None = None
    losses: TensorDict = field(default_factory=lambda: TensorDict(batch_size=[]))
    total_loss: torch.Tensor | None = None
    metrics: dict[str, float] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)
    stage_counts: dict[TrainingStageEnum, int] = field(default_factory=dict)


@runtime_checkable
class TrainingHook(Protocol):
    """Protocol that all training hooks must satisfy.

    A conforming hook declares a ``frequency`` (execute every *n* steps)
    and either a single ``stage`` or a ``stages`` frozenset for multi-stage
    registration.

    Parameters
    ----------
    ctx : TrainingContext
        The mutable training context for the current step.
    model : BaseModelMixin
        The model being trained.
    trainer : Trainer
        The trainer instance driving the loop (used for accessing
        optimizer, scheduler, etc.).

    Attributes
    ----------
    frequency : int
        Execute the hook every ``frequency`` steps.  Must be >= 1.
    stage : TrainingStageEnum
        The single stage this hook fires at.  Ignored when ``stages`` is
        present.
    """

    frequency: int
    stage: TrainingStageEnum

    def __call__(
        self, ctx: TrainingContext, model: BaseModelMixin, trainer: Any
    ) -> None:
        """Execute the hook."""
        ...
