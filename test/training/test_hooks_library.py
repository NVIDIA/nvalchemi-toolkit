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

"""Tests for built-in training hooks: EarlyStopping, SWA."""

from __future__ import annotations

from unittest.mock import MagicMock, call, patch

import pytest
from torch import nn
from torch.optim.swa_utils import AveragedModel

from nvalchemi.training._hooks import TrainingContext
from nvalchemi.training._hooks_library import (
    EarlyStoppingHook,
    SWAFinalizeHook,
    SWAHook,
)
from nvalchemi.training._stages import TrainingStageEnum
from nvalchemi.training._terminate import StopTraining

# ---------------------------------------------------------------------------
# EarlyStoppingHook
# ---------------------------------------------------------------------------


class TestEarlyStoppingHook:
    """Tests for EarlyStoppingHook."""

    def _make_ctx(self, **metrics: float) -> TrainingContext:
        """Build a TrainingContext with given metrics."""
        return TrainingContext(metrics=dict(metrics))

    def test_raises_after_patience_exhausted(self) -> None:
        """StopTraining is raised after ``patience`` validations with no improvement."""
        hook = EarlyStoppingHook(metric="val_loss", patience=3, mode="min")

        # First call sets the best value.
        ctx = self._make_ctx(val_loss=1.0)
        hook(ctx, MagicMock(), MagicMock())
        assert hook.best == 1.0
        assert hook.counter == 0

        # Three calls with no improvement.
        for i in range(2):
            ctx = self._make_ctx(val_loss=2.0)
            hook(ctx, MagicMock(), MagicMock())
            assert hook.counter == i + 1

        # Third non-improving call should raise.
        ctx = self._make_ctx(val_loss=2.0)
        with pytest.raises(StopTraining, match="No improvement"):
            hook(ctx, MagicMock(), MagicMock())

    def test_resets_counter_on_improvement(self) -> None:
        """Counter resets when metric improves."""
        hook = EarlyStoppingHook(metric="val_loss", patience=3, mode="min")

        hook(self._make_ctx(val_loss=1.0), MagicMock(), MagicMock())
        hook(self._make_ctx(val_loss=2.0), MagicMock(), MagicMock())
        assert hook.counter == 1

        # Improvement: counter resets.
        hook(self._make_ctx(val_loss=0.5), MagicMock(), MagicMock())
        assert hook.counter == 0
        assert hook.best == 0.5

    def test_mode_max(self) -> None:
        """mode='max' treats higher values as improvement."""
        hook = EarlyStoppingHook(metric="accuracy", patience=2, mode="max")

        hook(self._make_ctx(accuracy=0.8), MagicMock(), MagicMock())
        assert hook.best == 0.8

        # Higher is better — no counter increment.
        hook(self._make_ctx(accuracy=0.9), MagicMock(), MagicMock())
        assert hook.counter == 0
        assert hook.best == 0.9

        # Two non-improvements should raise.
        hook(self._make_ctx(accuracy=0.85), MagicMock(), MagicMock())
        assert hook.counter == 1
        with pytest.raises(StopTraining):
            hook(self._make_ctx(accuracy=0.85), MagicMock(), MagicMock())

    def test_missing_metric_is_noop(self) -> None:
        """If the metric key is absent from ctx.metrics, the hook does nothing."""
        hook = EarlyStoppingHook(metric="val_loss", patience=1, mode="min")

        # No val_loss key — should not raise.
        ctx = self._make_ctx(other_metric=1.0)
        hook(ctx, MagicMock(), MagicMock())
        assert hook.counter == 0

    def test_protocol_attributes(self) -> None:
        """Hook has required protocol attributes."""
        hook = EarlyStoppingHook()
        assert hook.frequency == 1
        assert hook.stage == TrainingStageEnum.AFTER_VALIDATION


# ---------------------------------------------------------------------------
# SWAHook
# ---------------------------------------------------------------------------


class TestSWAHook:
    """Tests for SWAHook."""

    def _make_model_and_swa(self) -> tuple[nn.Module, AveragedModel]:
        """Build a tiny model and its AveragedModel wrapper."""
        model = nn.Linear(3, 1)
        swa_model = AveragedModel(model)
        return model, swa_model

    def test_update_after_start_step(self) -> None:
        """update_parameters is called when global_step >= swa_start_step."""
        model, swa_model = self._make_model_and_swa()
        hook = SWAHook(swa_model=swa_model, swa_start_step=5)

        with patch.object(swa_model, "update_parameters") as mock_update:
            # Before start step — no call.
            ctx = TrainingContext(global_step=4)
            hook(ctx, model, MagicMock())
            mock_update.assert_not_called()

            # At start step — called.
            ctx = TrainingContext(global_step=5)
            hook(ctx, model, MagicMock())
            mock_update.assert_called_once_with(model)

    def test_update_called_every_step_after_start(self) -> None:
        """update_parameters is called on every dispatch after start step."""
        model, swa_model = self._make_model_and_swa()
        hook = SWAHook(swa_model=swa_model, swa_start_step=0)

        with patch.object(swa_model, "update_parameters") as mock_update:
            for step in range(3):
                ctx = TrainingContext(global_step=step)
                hook(ctx, model, MagicMock())
            assert mock_update.call_count == 3
            mock_update.assert_has_calls([call(model)] * 3)

    def test_protocol_attributes(self) -> None:
        """Hook has required protocol attributes."""
        _, swa_model = self._make_model_and_swa()
        hook = SWAHook(swa_model=swa_model)
        assert hook.frequency == 1
        assert hook.stage == TrainingStageEnum.AFTER_OPTIMIZER_STEP


# ---------------------------------------------------------------------------
# SWAFinalizeHook
# ---------------------------------------------------------------------------


class TestSWAFinalizeHook:
    """Tests for SWAFinalizeHook."""

    def test_calls_update_bn(self) -> None:
        """update_bn is called with the swa_model and train_loader."""
        model = nn.Linear(3, 1)
        swa_model = AveragedModel(model)
        loader = MagicMock()
        hook = SWAFinalizeHook(swa_model=swa_model, train_loader=loader, device="cpu")

        with patch("nvalchemi.training._hooks_library.update_bn") as mock_update_bn:
            ctx = TrainingContext()
            hook(ctx, MagicMock(), MagicMock())
            mock_update_bn.assert_called_once_with(loader, swa_model, device="cpu")

    def test_protocol_attributes(self) -> None:
        """Hook has required protocol attributes."""
        model = nn.Linear(3, 1)
        swa_model = AveragedModel(model)
        hook = SWAFinalizeHook(swa_model=swa_model, train_loader=MagicMock())
        assert hook.frequency == 1
        assert hook.stage == TrainingStageEnum.ON_TRAINING_END
