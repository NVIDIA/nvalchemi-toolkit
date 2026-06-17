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

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch
from torch.nn import ModuleDict
from torch.optim.lr_scheduler import LRScheduler

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
    batch : Batch | None
        Current batch being processed. ``None`` is used for lifecycle stages
        that run before the first batch is available.
    model : BaseModelMixin | None
        Model being used (if applicable).
    global_rank : int
        Distributed rank of this process.
    workflow : Any
        Back-reference to the engine running the hooks. ``None`` when
        the workflow does not inject itself.
    """

    batch: Batch | None
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
        Current optimizer step number on this worker.
    global_step_count : int
        Current optimizer step number across all data-parallel workers.
    batch_count : int
        Number of training batches consumed, including batches whose
        optimizer step was skipped by update hooks.
    epoch_step_count : int
        Number of batches consumed within the current training epoch.
    epoch : int
        Current training epoch.
    loss : torch.Tensor | None
        Aggregate loss for the current step.
    losses : dict[str, torch.Tensor] | None
        Named loss components for the current step.
    models : dict[str, BaseModelMixin] | ModuleDict | None
        Models participating in the training step; this differs
        from the ``model`` attribute which is intended to
        represent a 'main' model in multi-model workflows. The
        key/model mapping should be semantic, e.g. 'student' and
        'teacher' in distillation workflows, with 'student' being
        the intended 'main' model.
    optimizers : list[torch.optim.Optimizer]
        Optimizers participating in the training step. Empty when no
        optimizer is attached (e.g. eval-only or manually-driven hook
        contexts); ``TrainingUpdateOrchestrator`` and similar consumers
        treat an empty list as a no-op.
    lr_schedulers : list[torch.optim.lr_scheduler.LRScheduler | None]
        Learning rate schedulers participating in the training step.
        Aligned positionally with ``optimizers`` when populated; entries
        may be ``None`` when an optimizer has no scheduler. Empty when no
        scheduler is attached.
    gradients : dict[str, torch.Tensor] | None
        Parameter gradients for the current step.
    grad_scaler : torch.amp.GradScaler | None
        AMP gradient scaler for mixed-precision training; ``None`` when
        AMP is not in use.
    validation : dict[str, Any] | None
        Latest validation summary produced by the training strategy's
        validation checkpoint (``TrainingStrategy.validate()``).
        ``None`` until validation has run or after the latest summary is
        consumed by metric-driven schedulers. In distributed runs, the reduced
        summary is available on every rank.
    """

    step_count: int = 0
    global_step_count: int = 0
    batch_count: int = 0
    epoch_step_count: int = 0
    epoch: int = 0
    loss: torch.Tensor | None = None
    losses: dict[str, torch.Tensor] | None = None
    models: dict[str, BaseModelMixin] | ModuleDict | None = None
    optimizers: list[torch.optim.Optimizer] = field(default_factory=list)
    lr_schedulers: list[LRScheduler | None] = field(default_factory=list)
    gradients: dict[str, torch.Tensor] | None = None
    grad_scaler: torch.amp.GradScaler | None = None
    validation: dict[str, Any] | None = None
