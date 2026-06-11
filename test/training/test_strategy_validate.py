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
"""Tests for :meth:`TrainingStrategy.validate` (Phase B)."""

from __future__ import annotations

from typing import Any

import pytest
import torch

from nvalchemi.data import Batch
from nvalchemi.models.base import BaseModelMixin
from nvalchemi.training import EnergyMSELoss
from nvalchemi.training._validation import ValidationConfig
from nvalchemi.training.strategy import TrainingStrategy, default_training_fn
from test.training.conftest import (
    _build_baseline_strategy_kwargs,
    _build_batch,
)


def _energy_only_training_fn(
    model: BaseModelMixin, batch: Batch
) -> dict[str, torch.Tensor]:
    """Run the demo model with only energy active."""
    active_outputs = set(model.model_config.active_outputs)
    model.set_config("active_outputs", {"energy"})
    try:
        return default_training_fn(model, batch)
    finally:
        model.set_config("active_outputs", active_outputs)


def _make_validation_strategy(**overrides: Any) -> TrainingStrategy:
    """Build a strategy with a ValidationConfig attached."""
    batch = _build_batch()
    vc_kwargs = overrides.pop("validation_config_kwargs", {})
    vc = ValidationConfig(validation_data=[batch], **vc_kwargs)
    kwargs = _build_baseline_strategy_kwargs()
    kwargs["validation_config"] = vc
    kwargs.update(overrides)
    return TrainingStrategy(**kwargs)


class TestStrategyValidateLiveWeights:
    """validate() with default (live) model weights."""

    def test_returns_summary_dict_with_expected_keys(self) -> None:
        """validate() returns a summary dict with the canonical key set."""
        strategy = _make_validation_strategy()
        summary = strategy.validate()

        assert summary is not None
        assert summary["name"] == "validation"
        assert summary["model_source"] == "live"
        assert summary["precision"] == "float32"
        assert "total_loss" in summary
        assert "per_component_unweighted" in summary
        assert "EnergyMSELoss" in summary["per_component_unweighted"]
        assert "ForceMSELoss" in summary["per_component_unweighted"]
        assert summary["num_batches"] == 1

    def test_summary_stored_on_last_validation(self) -> None:
        """validate() sets last_validation / validation property."""
        strategy = _make_validation_strategy()
        summary = strategy.validate()

        assert strategy.last_validation is summary


class TestStrategyValidateInferenceModel:
    """validate() with inference_model (EMA) slot populated."""

    def test_single_module_slot_reports_ema_source(self) -> None:
        """Setting inference_model (single module) -> model_source='ema'."""
        strategy = _make_validation_strategy(
            loss_fn=EnergyMSELoss(),
            training_fn=_energy_only_training_fn,
            validation_config_kwargs={"grad_mode": "disabled"},
        )
        # Populate the inference_model slot with a copy of the live model
        live = strategy.models["main"]
        import copy

        ema_model = copy.deepcopy(live)
        strategy.inference_model = ema_model

        summary = strategy.validate()

        assert summary is not None
        assert summary["model_source"] == "ema"
        assert summary["ema_model_keys"] == ["main"]


class TestStrategyValidateGradIsolation:
    """validate() with grad_mode='enabled' preserves training gradients."""

    def test_grad_enabled_restores_pre_existing_grads(self) -> None:
        """Pre-existing param.grad is identical after a grad-enabled validate()."""
        strategy = _make_validation_strategy(
            validation_config_kwargs={"grad_mode": "enabled"},
        )
        model = strategy.models["main"]
        # Set a fake gradient on every parameter
        original_grads: dict[str, torch.Tensor] = {}
        for name, param in model.named_parameters():
            fake_grad = torch.randn_like(param)
            param.grad = fake_grad.clone()
            original_grads[name] = fake_grad

        strategy.validate()

        for name, param in model.named_parameters():
            assert param.grad is not None, f"grad lost for {name}"
            assert torch.equal(param.grad, original_grads[name]), (
                f"grad changed for {name}"
            )


class TestStrategyValidateTrainingModeRestoration:
    """validate() restores module training modes when set_eval=True."""

    def test_train_mode_restored_after_validate(self) -> None:
        """Modules in train() mode before validate() are restored to train()."""
        strategy = _make_validation_strategy(
            validation_config_kwargs={"set_eval": True},
        )
        model = strategy.models["main"]
        model.train()

        strategy.validate()

        assert model.training is True


class TestStrategyValidateErrorHandling:
    """validate() error paths."""

    def test_raises_when_validation_config_is_none(self) -> None:
        """validate() raises RuntimeError when validation_config is not set."""
        kwargs = _build_baseline_strategy_kwargs()
        strategy = TrainingStrategy(**kwargs)
        assert strategy.validation_config is None

        with pytest.raises(RuntimeError, match="requires a validation_config"):
            strategy.validate()

    def test_raises_when_mixed_precision_always_without_hook(self) -> None:
        """use_mixed_precision='always' without MixedPrecisionHook raises RuntimeError."""
        strategy = _make_validation_strategy(
            validation_config_kwargs={"use_mixed_precision": "always"},
        )
        with pytest.raises(RuntimeError, match="MixedPrecisionHook"):
            strategy.validate()
