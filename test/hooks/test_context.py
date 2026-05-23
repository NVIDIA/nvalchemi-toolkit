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

from nvalchemi.hooks import DynamicsContext, HookContext, TrainContext


class TestHookContext:
    def test_create_with_required_fields_only(self):
        mock_batch = MagicMock()
        ctx = HookContext(batch=mock_batch)

        assert ctx.batch is mock_batch

    def test_create_with_common_optional_fields(self):
        mock_batch = MagicMock()
        mock_model = MagicMock()
        mock_workflow = MagicMock()

        ctx = HookContext(
            batch=mock_batch,
            model=mock_model,
            global_rank=2,
            workflow=mock_workflow,
        )

        assert ctx.batch is mock_batch
        assert ctx.model is mock_model
        assert ctx.global_rank == 2
        assert ctx.workflow is mock_workflow

    def test_default_values_for_optional_fields(self):
        mock_batch = MagicMock()
        ctx = HookContext(batch=mock_batch)

        assert ctx.model is None
        assert ctx.global_rank == 0
        assert ctx.workflow is None

    def test_type_annotations_work_at_runtime(self):
        mock_batch = MagicMock()
        ctx = HookContext(batch=mock_batch)
        assert hasattr(ctx, "__dataclass_fields__")
        fields = ctx.__dataclass_fields__
        assert "batch" in fields
        assert "model" in fields
        assert "global_rank" in fields
        assert "workflow" in fields


class TestDynamicsContext:
    def test_create_with_dynamics_fields(self):
        mock_batch = MagicMock()
        mock_model = MagicMock()
        mock_converged = torch.tensor([True, False])

        ctx = DynamicsContext(
            batch=mock_batch,
            step_count=42,
            model=mock_model,
            converged_mask=mock_converged,
            global_rank=2,
        )

        assert ctx.batch is mock_batch
        assert ctx.step_count == 42
        assert ctx.model is mock_model
        assert ctx.converged_mask is mock_converged
        assert ctx.global_rank == 2

    def test_default_values_for_dynamics_fields(self):
        mock_batch = MagicMock()
        ctx = DynamicsContext(batch=mock_batch)

        assert ctx.step_count == 0
        assert ctx.converged_mask is None
        assert ctx.model is None
        assert ctx.global_rank == 0


class TestTrainContext:
    def test_create_with_training_fields(self):
        mock_batch = MagicMock()
        mock_model = MagicMock()
        mock_loss = torch.tensor(0.5)
        mock_losses = {"energy": torch.tensor(0.4), "force": torch.tensor(0.1)}
        mock_optimizer = MagicMock()
        mock_scheduler = MagicMock()
        mock_gradients = {"param": torch.tensor([1.0, 2.0])}
        mock_scaler = MagicMock(spec=torch.amp.GradScaler)

        ctx = TrainContext(
            batch=mock_batch,
            step_count=42,
            batch_count=43,
            epoch_step_count=2,
            epoch=5,
            loss=mock_loss,
            losses=mock_losses,
            models={"main": mock_model},
            optimizers=[mock_optimizer],
            lr_schedulers=[mock_scheduler],
            gradients=mock_gradients,
            grad_scaler=mock_scaler,
            global_rank=2,
        )

        assert ctx.batch is mock_batch
        assert ctx.step_count == 42
        assert ctx.batch_count == 43
        assert ctx.epoch_step_count == 2
        assert ctx.epoch == 5
        assert ctx.loss is mock_loss
        assert ctx.losses is mock_losses
        assert ctx.models == {"main": mock_model}
        assert ctx.optimizers == [mock_optimizer]
        assert ctx.lr_schedulers == [mock_scheduler]
        assert ctx.gradients is mock_gradients
        assert ctx.grad_scaler is mock_scaler
        assert ctx.global_rank == 2

    def test_default_values_for_training_fields(self):
        mock_batch = MagicMock()
        ctx = TrainContext(batch=mock_batch)

        assert ctx.step_count == 0
        assert ctx.batch_count == 0
        assert ctx.epoch_step_count == 0
        assert ctx.epoch == 0
        assert ctx.loss is None
        assert ctx.losses is None
        assert ctx.models is None
        assert ctx.optimizers == []
        assert ctx.lr_schedulers == []
        assert ctx.gradients is None
        assert ctx.grad_scaler is None

    def test_optimizers_default_is_independent_per_instance(self):
        ctx_a = TrainContext(batch=MagicMock())
        ctx_b = TrainContext(batch=MagicMock())
        ctx_a.optimizers.append(MagicMock())
        assert ctx_b.optimizers == []
