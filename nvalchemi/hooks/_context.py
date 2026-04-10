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
"""Hook context for passing state to hooks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from nvalchemi.data.batch import Batch
    from nvalchemi.models.base import BaseModelMixin


@dataclass
class HookContext:
    """Context object passed to hooks at each stage.

    Attributes
    ----------
    batch : Batch
        Current batch being processed.
    step_count : int
        Current step number in the workflow.
    model : BaseModelMixin | None
        Model being used (if applicable).
    loss : torch.Tensor | None
        Current loss value (training only).
    optimizer : torch.optim.Optimizer | None
        Optimizer being used (training only).
    lr_scheduler : object | None
        Learning rate scheduler (training only).
    gradients : dict[str, torch.Tensor] | None
        Parameter gradients (training only).
    converged_mask : torch.Tensor | None
        Boolean mask of converged samples (dynamics only).
    epoch : int | None
        Current epoch number (training only).
    global_rank : int
        Distributed rank of this process.
    workflow : Any
        Back-reference to the engine running the hooks (e.g. a
        ``BaseDynamics`` instance).  ``None`` when the workflow does
        not inject itself.
    """

    batch: Batch
    step_count: int
    model: BaseModelMixin | None = None
    loss: torch.Tensor | None = None
    optimizer: torch.optim.Optimizer | None = None
    lr_scheduler: object | None = None
    gradients: dict[str, torch.Tensor] | None = None
    converged_mask: torch.Tensor | None = None
    epoch: int | None = None
    global_rank: int = 0
    workflow: Any = None
