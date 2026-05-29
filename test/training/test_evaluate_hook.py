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
"""Tests for :class:`nvalchemi.training.hooks.EvaluateHook`."""

from __future__ import annotations

from typing import Any

import pytest
import torch
from pydantic import ValidationError

from nvalchemi.data import Batch
from nvalchemi.models.base import BaseModelMixin
from nvalchemi.training import EnergyLoss, TrainingStage
from nvalchemi.training.hooks import EMAHook, EvaluateHook, MixedPrecisionHook
from nvalchemi.training.optimizers import OptimizerConfig
from nvalchemi.training.strategy import TrainingStrategy, default_training_fn
from test.training.conftest import (
    _build_baseline_strategy_kwargs,
    _build_demo_model,
)


def energy_only_training_fn(
    model: BaseModelMixin, batch: Batch
) -> dict[str, torch.Tensor]:
    """Run the demo model with only energy active."""
    active_outputs = set(model.model_config.active_outputs)
    model.set_config("active_outputs", {"energy"})
    try:
        return default_training_fn(model, batch)
    finally:
        model.set_config("active_outputs", active_outputs)


def energy_only_cast_back_training_fn(
    model: BaseModelMixin, batch: Batch
) -> dict[str, torch.Tensor]:
    """Energy-only forward that restores fp32 predictions after autocast."""
    return {
        key: value.to(torch.float32)
        for key, value in energy_only_training_fn(model, batch).items()
    }


def named_energy_training_fn(
    models: dict[str, BaseModelMixin], batch: Batch
) -> dict[str, torch.Tensor]:
    """Named-model training function that uses the student model."""
    return energy_only_training_fn(models["student"], batch)


def _energy_strategy_kwargs(model: BaseModelMixin | None = None) -> dict[str, Any]:
    """Return a minimal energy-only strategy configuration."""
    return {
        **_build_baseline_strategy_kwargs(models=model or _build_demo_model()),
        "training_fn": energy_only_training_fn,
        "loss_fn": EnergyLoss(),
    }


class TestEvaluateHookConstruction:
    """Constructor validation and convenience scheduling."""

    def test_every_n_steps_maps_to_optimizer_step_stage(self, batch: Batch) -> None:
        hook = EvaluateHook(validation_data=[batch], every_n_steps=5)
        assert hook.stage is TrainingStage.AFTER_OPTIMIZER_STEP
        assert hook.frequency == 1

    def test_every_n_epochs_maps_to_epoch_stage(self, batch: Batch) -> None:
        hook = EvaluateHook(validation_data=[batch], every_n_epochs=2)
        assert hook.stage is TrainingStage.AFTER_EPOCH
        assert hook.frequency == 1

    def test_convenience_schedules_are_exclusive(self, batch: Batch) -> None:
        with pytest.raises(ValidationError, match="Only one"):
            EvaluateHook(
                validation_data=[batch],
                every_n_epochs=1,
                every_n_steps=1,
            )

    def test_every_n_steps_rejects_conflicting_stage(self, batch: Batch) -> None:
        with pytest.raises(ValidationError, match="AFTER_OPTIMIZER_STEP"):
            EvaluateHook(
                validation_data=[batch],
                every_n_steps=1,
                stage=TrainingStage.AFTER_EPOCH,
            )


class TestEvaluateHookValidationLoop:
    """Validation loop behavior through TrainingStrategy."""

    def test_default_strategy_functions_publish_summary(
        self, batch: Batch, dataset: list[Batch]
    ) -> None:
        hook = EvaluateHook(validation_data=[batch], grad_mode="auto")
        strategy = TrainingStrategy(
            **{**_build_baseline_strategy_kwargs(), "hooks": [hook]}
        )

        strategy.run(dataset[:1])

        assert strategy.validation is not None
        assert strategy.validation["name"] == "validation"
        assert strategy.validation["num_batches"] == 1
        assert strategy.validation["model_source"] == "live"
        assert "total_loss" in strategy.validation
        assert "EnergyLoss" in strategy.validation["per_component_total"]
        assert "ForceLoss" in strategy.validation["per_component_total"]
        assert all(param.grad is None for param in strategy.models["main"].parameters())

    def test_every_n_steps_uses_completed_optimizer_steps(self, batch: Batch) -> None:
        calls: list[int] = []

        def validation_fn(
            model: BaseModelMixin, validation_batch: Batch
        ) -> dict[str, torch.Tensor]:
            calls.append(len(calls))
            return energy_only_training_fn(model, validation_batch)

        hook = EvaluateHook(
            validation_data=[batch],
            validation_fn=validation_fn,
            loss_fn=EnergyLoss(),
            every_n_steps=2,
            grad_mode="disabled",
        )
        strategy = TrainingStrategy(
            **{
                **_energy_strategy_kwargs(),
                "num_epochs": None,
                "num_steps": 3,
                "hooks": [hook],
            }
        )

        strategy.run([batch, batch, batch])

        assert len(calls) == 1
        assert strategy.validation is not None
        assert strategy.validation["num_batches"] == 1

    def test_named_models_use_named_validation_call(self, batch: Batch) -> None:
        seen_keys: list[tuple[str, ...]] = []

        def validation_fn(
            models: dict[str, BaseModelMixin], validation_batch: Batch
        ) -> dict[str, torch.Tensor]:
            seen_keys.append(tuple(sorted(models)))
            return energy_only_training_fn(models["student"], validation_batch)

        hook = EvaluateHook(
            validation_data=[batch],
            validation_fn=validation_fn,
            loss_fn=EnergyLoss(),
            grad_mode="disabled",
        )
        strategy = TrainingStrategy(
            models={"student": _build_demo_model(), "teacher": _build_demo_model()},
            optimizer_configs={
                "student": [OptimizerConfig(optimizer_cls=torch.optim.Adam)]
            },
            num_epochs=1,
            training_fn=named_energy_training_fn,
            loss_fn=EnergyLoss(),
            hooks=[hook],
        )

        strategy.run([batch])

        assert seen_keys == [("student", "teacher")]
        assert strategy.validation is not None

    def test_eval_mode_restored(self, batch: Batch) -> None:
        model = _build_demo_model()
        model.train()
        hook = EvaluateHook(
            validation_data=[batch],
            validation_fn=energy_only_training_fn,
            loss_fn=EnergyLoss(),
            grad_mode="disabled",
        )
        strategy = TrainingStrategy(
            **{**_energy_strategy_kwargs(model), "hooks": [hook]}
        )

        strategy.run([batch])

        assert model.training is True

    def test_empty_validation_data_raises(self, batch: Batch) -> None:
        hook = EvaluateHook(
            validation_data=[],
            validation_fn=energy_only_training_fn,
            loss_fn=EnergyLoss(),
            grad_mode="disabled",
        )
        strategy = TrainingStrategy(**{**_energy_strategy_kwargs(), "hooks": [hook]})

        with pytest.raises(ValueError, match="produced no batches"):
            strategy.run([batch])


class TestEvaluateHookEMA:
    """EMA model selection."""

    def test_uses_initialized_ema_model_by_default(self, batch: Batch) -> None:
        seen_model: list[BaseModelMixin] = []

        def validation_fn(
            model: BaseModelMixin, validation_batch: Batch
        ) -> dict[str, torch.Tensor]:
            seen_model.append(model)
            return energy_only_training_fn(model, validation_batch)

        ema = EMAHook(decay=0.0)
        hook = EvaluateHook(
            validation_data=[batch],
            validation_fn=validation_fn,
            loss_fn=EnergyLoss(),
            every_n_steps=1,
            grad_mode="disabled",
        )
        strategy = TrainingStrategy(
            **{**_energy_strategy_kwargs(), "hooks": [ema, hook]}
        )

        strategy.run([batch])

        averaged = ema.get_averaged_model().module
        assert seen_model == [averaged]
        assert strategy.validation is not None
        assert strategy.validation["model_source"] == "ema"
        assert strategy.validation["ema_model_keys"] == ["main"]

    def test_use_ema_always_requires_initialized_weights(self, batch: Batch) -> None:
        ema = EMAHook()
        hook = EvaluateHook(
            validation_data=[batch],
            validation_fn=energy_only_training_fn,
            loss_fn=EnergyLoss(),
            stage=TrainingStage.BEFORE_TRAINING,
            grad_mode="disabled",
            use_ema="always",
        )
        strategy = TrainingStrategy(
            **{**_energy_strategy_kwargs(), "hooks": [ema, hook]}
        )

        with pytest.raises(RuntimeError, match="initialized averaged weights"):
            strategy.run([batch])


class TestEvaluateHookMixedPrecision:
    """Mixed-precision validation integration."""

    def test_auto_uses_registered_mixed_precision_autocast(self, batch: Batch) -> None:
        records: dict[str, Any] = {}

        def validation_fn(
            model: BaseModelMixin, validation_batch: Batch
        ) -> dict[str, torch.Tensor]:
            records["enabled"] = torch.is_autocast_enabled("cpu")
            records["dtype"] = torch.get_autocast_dtype("cpu")
            return energy_only_cast_back_training_fn(model, validation_batch)

        mp = MixedPrecisionHook(precision=torch.bfloat16)
        hook = EvaluateHook(
            validation_data=[batch],
            validation_fn=validation_fn,
            loss_fn=EnergyLoss(),
            every_n_steps=1,
            grad_mode="disabled",
        )
        strategy = TrainingStrategy(
            **{
                **_energy_strategy_kwargs(),
                "training_fn": energy_only_cast_back_training_fn,
                "hooks": [mp, hook],
            }
        )

        strategy.run([batch])

        assert records == {"enabled": True, "dtype": torch.bfloat16}
        assert strategy.validation is not None
        assert strategy.validation["precision"] == "bfloat16"

    def test_always_requires_mixed_precision_hook(self, batch: Batch) -> None:
        hook = EvaluateHook(
            validation_data=[batch],
            validation_fn=energy_only_training_fn,
            loss_fn=EnergyLoss(),
            use_mixed_precision="always",
            grad_mode="disabled",
        )
        strategy = TrainingStrategy(**{**_energy_strategy_kwargs(), "hooks": [hook]})

        with pytest.raises(RuntimeError, match="MixedPrecisionHook"):
            strategy.run([batch])


class TestEvaluateHookDistributedSummary:
    """Distributed summary publication behavior."""

    def test_nonzero_rank_does_not_publish_validation(
        self, monkeypatch: pytest.MonkeyPatch, batch: Batch
    ) -> None:
        hook = EvaluateHook(
            validation_data=[batch],
            validation_fn=energy_only_training_fn,
            loss_fn=EnergyLoss(),
            grad_mode="disabled",
        )
        strategy = TrainingStrategy(**{**_energy_strategy_kwargs(), "hooks": [hook]})
        monkeypatch.setattr(
            "nvalchemi.training.strategy.dist.is_available", lambda: True
        )
        monkeypatch.setattr(
            "nvalchemi.training.strategy.dist.is_initialized", lambda: True
        )
        monkeypatch.setattr("nvalchemi.training.strategy.dist.get_rank", lambda: 1)
        monkeypatch.setattr(
            "nvalchemi.training.hooks.evaluate.dist.all_reduce",
            lambda tensor, op=None: None,
        )

        strategy.run([batch])

        assert strategy.validation is None
