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
"""Hook context dataclasses for passing workflow state to hooks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
from torch.nn import ModuleDict

if TYPE_CHECKING:
    from nvalchemi.data.batch import Batch
    from nvalchemi.models.base import BaseModelMixin


@dataclass(kw_only=True)
class HookContext:
    """Common context object passed to hooks.

    ``HookContext`` contains fields shared by all hook-enabled workflows.
    Workflow-specific subclasses add state that is only meaningful in that
    domain, such as dynamics step counts or training losses.

    Attributes
    ----------
    batch : Batch
        Current batch being processed.
    model : BaseModelMixin | None
        Model being used (if applicable).
    global_rank : int
        Distributed rank of this process.
    workflow : Any
        Back-reference to the engine running the hooks. ``None`` when
        the workflow does not inject itself.
    """

    batch: Batch
    model: BaseModelMixin | None = None
    global_rank: int = 0
    workflow: Any = None


@dataclass(kw_only=True)
class DynamicsContext(HookContext):
    """Context object passed to dynamics hooks.

    Attributes
    ----------
    step_count : int
        Current dynamics step number.
    converged_mask : torch.Tensor | None
        Boolean mask of samples that converged at the current hook stage.
        ``None`` when convergence has not fired for this dispatch.
    """

    step_count: int = 0
    converged_mask: torch.Tensor | None = None


@dataclass(kw_only=True)
class TrainContext(HookContext):
    """Context object passed to training hooks.

    Attributes
    ----------
    step_count : int
        Current optimizer step number.
    epoch : int
        Current training epoch.
    loss : torch.Tensor | None
        Aggregate loss for the current step.
    losses : dict[str, torch.Tensor] | None
        Named loss components for the current step.
    models : dict[str, BaseModelMixin] | ModuleDict[str, BaseModelMixin] | None
        Models participating in the training step.
    optimizers : list[torch.optim.Optimizer] | None
        Optimizers participating in the training step.
    lr_schedulers : list[object] | None
        Learning rate schedulers participating in the training step.
    gradients : dict[str, torch.Tensor] | None
        Parameter gradients for the current step.
    """

    step_count: int = 0
    epoch: int = 0
    loss: torch.Tensor | None = None
    losses: dict[str, torch.Tensor] | None = None
    models: dict[str, BaseModelMixin] | ModuleDict[str, BaseModelMixin] | None = None
    optimizers: list[torch.optim.Optimizer] | None = None
    lr_schedulers: list[object] | None = None
    gradients: dict[str, torch.Tensor] | None = None
