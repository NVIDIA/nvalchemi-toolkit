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

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from nvalchemi.data.batch import Batch
    from nvalchemi.models.base import BaseModelMixin
    from nvalchemi.training.losses.composition import ComposedLossOutput


@dataclass(init=False)
class HookContext:
    """Context object passed to hooks at each stage.

    Attributes
    ----------
    batch : Batch
        Current batch being processed.
    step_count : int
        Current step number in the workflow.
    model : BaseModelMixin | None
        Backwards-compatible alias for ``models["main"]`` when present.
        ``None`` when no primary model is registered.
    models : dict[str, BaseModelMixin]
        Named models visible to hooks. Training strategies populate this for
        multi-model workflows. The ``"main"`` key is the only primary model
        used by the ``model`` alias; single-model and dynamics code can pass
        ``model=...`` to populate that key.
    loss : torch.Tensor | None
        Total scalar loss value produced by the training step. When
        ``losses`` is populated, this equals ``losses["total_loss"]``; see
        ``losses`` for the per-component breakdown.
    losses : ComposedLossOutput | None
        Per-component loss contributions produced by a ComposedLossFunction.
        Populated by training strategies from AFTER_LOSS onward so hooks can
        log or monitor individual loss terms without re-running the loss
        forward pass. Tensors are autograd-connected through BEFORE_BACKWARD,
        then detached from AFTER_BACKWARD onward. ``None`` for non-training
        contexts (e.g. dynamics). Hooks must not retain live tensors across
        steps, and calling ``.item()`` on CUDA tensors forces a host-device
        sync.
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
    models: dict[str, BaseModelMixin] = field(default_factory=dict)
    loss: torch.Tensor | None = None
    losses: ComposedLossOutput | None = None
    optimizer: torch.optim.Optimizer | None = None
    lr_scheduler: object | None = None
    gradients: dict[str, torch.Tensor] | None = None
    converged_mask: torch.Tensor | None = None
    epoch: int | None = None
    global_rank: int = 0
    workflow: Any = None

    def __init__(
        self,
        batch: Batch,
        step_count: int,
        model: BaseModelMixin | None = None,
        models: dict[str, BaseModelMixin] | None = None,
        loss: torch.Tensor | None = None,
        losses: ComposedLossOutput | None = None,
        optimizer: torch.optim.Optimizer | None = None,
        lr_scheduler: object | None = None,
        gradients: dict[str, torch.Tensor] | None = None,
        converged_mask: torch.Tensor | None = None,
        epoch: int | None = None,
        global_rank: int = 0,
        workflow: Any = None,
    ) -> None:
        """Initialize context state while preserving legacy ``model=`` input."""
        self.batch = batch
        self.step_count = step_count
        self.models = models if models is not None else {}
        if model is not None:
            self.model = model
        if self.models and "main" not in self.models:
            raise ValueError(
                "HookContext models must include a 'main' entry when named "
                f"models are provided; available model keys: {sorted(self.models)}."
            )
        self.loss = loss
        self.losses = losses
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.gradients = gradients
        self.converged_mask = converged_mask
        self.epoch = epoch
        self.global_rank = global_rank
        self.workflow = workflow

    def _get_model(self) -> BaseModelMixin | None:
        """Return the primary model from the named model dictionary."""
        if self.models and "main" not in self.models:
            raise ValueError(
                "HookContext models must include a 'main' entry to use the "
                f"model alias; available model keys: {sorted(self.models)}."
            )
        return self.models.get("main")

    def _set_model(self, value: BaseModelMixin) -> None:
        """Assign the backwards-compatible primary model alias."""
        self.models["main"] = value

    # ``model`` is also a dataclass field above so existing runtime
    # introspection sees ``__dataclass_fields__["model"]`` while this property
    # keeps the alias live with ``models["main"]``.
    model = property(_get_model, _set_model)
