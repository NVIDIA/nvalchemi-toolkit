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
from __future__ import annotations

from unittest.mock import MagicMock

import torch

from nvalchemi.hooks import HookContext


class TestHookContext:
    def test_create_with_required_fields_only(self):
        mock_batch = MagicMock()
        ctx = HookContext(batch=mock_batch, step_count=10)

        assert ctx.batch is mock_batch
        assert ctx.step_count == 10

    def test_create_with_all_optional_fields(self):
        mock_batch = MagicMock()
        mock_model = MagicMock()
        mock_loss = torch.tensor(0.5)
        mock_optimizer = MagicMock()
        mock_scheduler = MagicMock()
        mock_gradients = {"param": torch.tensor([1.0, 2.0])}
        mock_converged = torch.tensor([True, False])

        ctx = HookContext(
            batch=mock_batch,
            step_count=42,
            model=mock_model,
            loss=mock_loss,
            optimizer=mock_optimizer,
            lr_scheduler=mock_scheduler,
            gradients=mock_gradients,
            converged_mask=mock_converged,
            epoch=5,
            global_rank=2,
        )

        assert ctx.batch is mock_batch
        assert ctx.step_count == 42
        assert ctx.model is mock_model
        assert ctx.loss is mock_loss
        assert ctx.optimizer is mock_optimizer
        assert ctx.lr_scheduler is mock_scheduler
        assert ctx.gradients is mock_gradients
        assert ctx.converged_mask is mock_converged
        assert ctx.epoch == 5
        assert ctx.global_rank == 2

    def test_default_values_for_optional_fields(self):
        mock_batch = MagicMock()
        ctx = HookContext(batch=mock_batch, step_count=0)

        assert ctx.model is None
        assert ctx.loss is None
        assert ctx.optimizer is None
        assert ctx.lr_scheduler is None
        assert ctx.gradients is None
        assert ctx.converged_mask is None
        assert ctx.epoch is None
        assert ctx.global_rank == 0

    def test_type_annotations_work_at_runtime(self):
        mock_batch = MagicMock()
        ctx = HookContext(batch=mock_batch, step_count=1)
        assert hasattr(ctx, "__dataclass_fields__")
        fields = ctx.__dataclass_fields__
        assert "batch" in fields
        assert "step_count" in fields
        assert "model" in fields
        assert "global_rank" in fields
