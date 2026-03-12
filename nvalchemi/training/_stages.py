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
"""Training stage enumeration.

Defines :class:`TrainingStageEnum`, the set of hook-points available during a
training run.  Values are grouped by tens so that related stages sort together
and future stages can be inserted without renumbering.

Groups
------
0-1     epoch boundaries
10-20   within-step stages (data load through optimizer)
30-31   validation boundaries
40-41   checkpoint boundaries
50      end-of-training sentinel
"""

from __future__ import annotations

from enum import Enum


class TrainingStageEnum(Enum):
    """Hook-points available during a training run.

    Attributes
    ----------
    BEFORE_EPOCH : int
        Fires at the start of each epoch.
    AFTER_EPOCH : int
        Fires at the end of each epoch.
    BEFORE_STEP : int
        Fires before each training step.
    AFTER_DATA_LOAD : int
        Fires after the batch has been loaded (and moved to device).
    BEFORE_FORWARD : int
        Fires before the model forward pass.
    AFTER_FORWARD : int
        Fires after the model forward pass.
    BEFORE_LOSS : int
        Fires before loss computation.
    AFTER_LOSS : int
        Fires after loss computation.
    BEFORE_BACKWARD : int
        Fires before the backward pass.
    AFTER_BACKWARD : int
        Fires after the backward pass.
    BEFORE_OPTIMIZER_STEP : int
        Fires before the optimizer step.
    AFTER_OPTIMIZER_STEP : int
        Fires after the optimizer step.
    AFTER_STEP : int
        Fires after each training step completes.
    BEFORE_VALIDATION : int
        Fires before the validation loop.
    AFTER_VALIDATION : int
        Fires after the validation loop.
    BEFORE_CHECKPOINT : int
        Fires before a checkpoint is saved.
    AFTER_CHECKPOINT : int
        Fires after a checkpoint is saved.
    ON_TRAINING_END : int
        Fires once when training finishes.
    """

    BEFORE_EPOCH = 0
    AFTER_EPOCH = 1

    BEFORE_STEP = 10
    AFTER_DATA_LOAD = 11
    BEFORE_FORWARD = 12
    AFTER_FORWARD = 13
    BEFORE_LOSS = 14
    AFTER_LOSS = 15
    BEFORE_BACKWARD = 16
    AFTER_BACKWARD = 17
    BEFORE_OPTIMIZER_STEP = 18
    AFTER_OPTIMIZER_STEP = 19
    AFTER_STEP = 20

    BEFORE_VALIDATION = 30
    AFTER_VALIDATION = 31

    BEFORE_CHECKPOINT = 40
    AFTER_CHECKPOINT = 41

    ON_TRAINING_END = 50
