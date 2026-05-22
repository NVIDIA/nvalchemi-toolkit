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
"""Training-lifecycle stage enum."""

from __future__ import annotations

from enum import Enum, auto

__all__ = ["TrainingStage"]


class TrainingStage(Enum):
    """Stages of the training lifecycle at which hooks can fire.

    Parallel to :class:`nvalchemi.dynamics.base.DynamicsStage`, this enum
    marks the points before and after each operation in a training run.
    Members are paired ``BEFORE_*`` / ``AFTER_*`` around each lifecycle
    event, from the once-per-run ``BEFORE_TRAINING`` / ``AFTER_TRAINING``
    outer pair down to the per-batch forward, loss, backward, and
    optimizer-step phases.

    Attributes
    ----------
    BEFORE_TRAINING : TrainingStage
        Fires once before the epoch loop, after the model is on device
        and optimizers are constructed.
    BEFORE_EPOCH : TrainingStage
        Fires at the start of each epoch, before the first batch.
    BEFORE_BATCH : TrainingStage
        Fires at the start of each batch, before the default gradient
        zeroing path. A training-update orchestrator may claim this stage
        to decide whether zeroing should run for the batch.
    BEFORE_FORWARD : TrainingStage
        Fires before the model forward pass.
    AFTER_FORWARD : TrainingStage
        Fires after the model forward pass; predictions are available.
    BEFORE_LOSS : TrainingStage
        Fires before the loss computation.
    AFTER_LOSS : TrainingStage
        Fires after the loss computation; the loss tensor is populated.
    BEFORE_BACKWARD : TrainingStage
        Fires before the backward pass.
    DO_BACKWARD : TrainingStage
        Replacement slot for the backward pass. At most one hook may claim
        this stage; when claimed, ``TrainingStrategy`` skips its default
        ``loss.backward()`` and the claiming hook is responsible for
        performing (and scaling, if needed) the backward. Observers should
        use ``BEFORE_BACKWARD``/``AFTER_BACKWARD``.
    AFTER_BACKWARD : TrainingStage
        Fires after the backward pass and before the optimizer step;
        typical slot for gradient clipping or gradient-norm logging.
    BEFORE_OPTIMIZER_STEP : TrainingStage
        Fires immediately before the optimizer step; typical slot for
        observers that need to see unscaled gradients (see ``DO_BACKWARD``).
    DO_OPTIMIZER_STEP : TrainingStage
        Replacement slot for the optimizer and LR-scheduler step. At most
        one hook may claim this stage; when claimed, ``TrainingStrategy``
        skips its default optimizer and scheduler stepping and the claiming
        hook must step each optimizer in ``ctx.optimizers`` (and its
        corresponding scheduler if present). Observers should use
        ``BEFORE_OPTIMIZER_STEP``/``AFTER_OPTIMIZER_STEP``.
    AFTER_OPTIMIZER_STEP : TrainingStage
        Fires after the optimizer and scheduler step path completes;
        typical slot for EMA updates and post-step logging.
    AFTER_BATCH : TrainingStage
        Fires at the end of each batch.
    AFTER_EPOCH : TrainingStage
        Fires at the end of each epoch, after the last batch.
    AFTER_TRAINING : TrainingStage
        Fires once after the final epoch.
    """

    BEFORE_TRAINING = auto()
    BEFORE_EPOCH = auto()
    BEFORE_BATCH = auto()
    BEFORE_FORWARD = auto()
    AFTER_FORWARD = auto()
    BEFORE_LOSS = auto()
    AFTER_LOSS = auto()
    BEFORE_BACKWARD = auto()
    DO_BACKWARD = auto()
    AFTER_BACKWARD = auto()
    BEFORE_OPTIMIZER_STEP = auto()
    DO_OPTIMIZER_STEP = auto()
    AFTER_OPTIMIZER_STEP = auto()
    AFTER_BATCH = auto()
    AFTER_EPOCH = auto()
    AFTER_TRAINING = auto()
