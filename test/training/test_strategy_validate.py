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
from unittest.mock import patch

import pytest
import torch

from nvalchemi.data import Batch
from nvalchemi.hooks._context import TrainContext
from nvalchemi.models.base import BaseModelMixin
from nvalchemi.training import EnergyMSELoss
from nvalchemi.training._stages import TrainingStage
from nvalchemi.training._validation import ValidationConfig
from nvalchemi.training.hooks import TrainingUpdateHook, TrainingUpdateOrchestrator
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


class _GradAccumulationVetoHook(TrainingUpdateHook):
    """Allow only every ``accumulate_every``-th optimizer step.

    Mirrors gradient accumulation: vetoed ``DO_OPTIMIZER_STEP`` batches keep
    ``step_count`` stalled at its last value while ``batch_count`` advances.
    """

    priority = 10

    def __init__(self, accumulate_every: int) -> None:
        self.accumulate_every = accumulate_every
        self.optimizer_step_calls = 0

    def __call__(
        self,
        ctx: TrainContext,
        stage: TrainingStage,
        will_skip: bool,
    ) -> tuple[bool, torch.Tensor | None]:
        if stage is TrainingStage.DO_OPTIMIZER_STEP:
            self.optimizer_step_calls += 1
            return self.optimizer_step_calls % self.accumulate_every == 0, ctx.loss
        return True, ctx.loss


class _ForwardStageRecorderHook:
    """Record forward-stage hook dispatch during validation."""

    frequency = 1

    def __init__(self, stage: TrainingStage, events: list[str]) -> None:
        self.stage = stage
        self.events = events

    def __call__(self, ctx: TrainContext, stage: TrainingStage) -> None:
        assert ctx.batch is not None
        self.events.append(stage.name)


class TestStrategyValidateLiveWeights:
    """validate() with default (live) model weights."""

    def test_forward_hooks_run_around_validation_forward(self) -> None:
        """Strategy-owned validation reuses forward hooks for input transforms."""
        events: list[str] = []

        def _validation_fn(
            model: BaseModelMixin, batch: Batch
        ) -> dict[str, torch.Tensor]:
            events.append("forward")
            return _energy_only_training_fn(model, batch)

        strategy = _make_validation_strategy(
            loss_fn=EnergyMSELoss(),
            training_fn=_validation_fn,
            validation_config_kwargs={"grad_mode": "disabled"},
            hooks=[
                _ForwardStageRecorderHook(TrainingStage.BEFORE_FORWARD, events),
                _ForwardStageRecorderHook(TrainingStage.AFTER_FORWARD, events),
            ],
        )

        strategy.validate()

        assert events == ["BEFORE_FORWARD", "forward", "AFTER_FORWARD"]

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


def _find_orchestrator(strategy: TrainingStrategy) -> TrainingUpdateOrchestrator:
    """Return the orchestrator the strategy coalesced its update hooks into."""
    return next(
        hook for hook in strategy.hooks if isinstance(hook, TrainingUpdateOrchestrator)
    )


class TestStrategyValidateStepCadenceGate:
    """Step cadence fires only on batches whose optimizer step ran."""

    def test_stalled_step_count_validates_once_per_multiple(self) -> None:
        """Batches stalled on an eval multiple must not re-fire validation.

        With 3-batch gradient accumulation, ``step_count`` only advances on
        every third batch and sits on the eval multiple for the two vetoed
        batches that follow it. Without the step-ran gate each stalled batch
        re-ran a full validation pass; with it, every multiple fires exactly
        once.
        """
        accum = _GradAccumulationVetoHook(accumulate_every=3)
        strategy = _make_validation_strategy(
            validation_config_kwargs={"every_n_steps": 2},
            num_epochs=None,
            num_steps=4,
            hooks=[accum],
        )
        validate_steps: list[int] = []
        orig_validate = TrainingStrategy.validate

        def _recording_validate(self_: Any) -> Any:
            validate_steps.append(self_.step_count)
            return orig_validate(self_)

        dataset = [_build_batch(seed=i * 10) for i in range(12)]
        with patch.object(TrainingStrategy, "validate", _recording_validate):
            strategy.run(dataset)

        assert strategy.batch_count == 12
        assert strategy.step_count == 4
        # Steps 2 and 4 each fire exactly once — the stalled batches after
        # step 2 (vetoed optimizer steps) do not re-fire — followed by the
        # unconditional end-of-run pass at the final step (4).
        assert validate_steps == [2, 4, 4]

    def test_vetoed_step_does_not_fire_on_parked_multiple(self) -> None:
        """The gate mirrors the orchestrator's per-batch step-skipped flag."""
        strategy = _make_validation_strategy(
            validation_config_kwargs={"every_n_steps": 2},
            hooks=[_GradAccumulationVetoHook(accumulate_every=3)],
        )
        orchestrator = _find_orchestrator(strategy)
        strategy.step_count = 4

        # The batch that advanced step_count onto the multiple fires.
        orchestrator._optimizer_step_skipped = False
        assert strategy._should_validate(TrainingStage.AFTER_OPTIMIZER_STEP) is True
        assert (
            strategy._validation_checkpoint(TrainingStage.AFTER_OPTIMIZER_STEP) is True
        )
        assert strategy.last_validation is not None

        # Vetoed batches leave step_count parked; must not re-fire.
        orchestrator._optimizer_step_skipped = True
        assert strategy._should_validate(TrainingStage.AFTER_OPTIMIZER_STEP) is False
        assert (
            strategy._validation_checkpoint(TrainingStage.AFTER_OPTIMIZER_STEP) is False
        )

    def test_plain_strategy_without_orchestrator_always_fires(self) -> None:
        """Without an update orchestrator every step counts as ran."""
        strategy = _make_validation_strategy(
            validation_config_kwargs={"every_n_steps": 2},
        )
        strategy.step_count = 4

        assert strategy._should_validate(TrainingStage.AFTER_OPTIMIZER_STEP) is True
        assert (
            strategy._validation_checkpoint(TrainingStage.AFTER_OPTIMIZER_STEP) is True
        )
        # Off-multiple steps stay gated by the cadence itself.
        strategy.step_count = 5
        assert strategy._should_validate(TrainingStage.AFTER_OPTIMIZER_STEP) is False

    def test_epoch_cadence_ignores_step_ran_signal(self) -> None:
        """every_n_epochs fires even when the last optimizer step was vetoed."""
        strategy = _make_validation_strategy(
            validation_config_kwargs={"every_n_epochs": 1},
            hooks=[_GradAccumulationVetoHook(accumulate_every=3)],
        )
        orchestrator = _find_orchestrator(strategy)
        orchestrator._optimizer_step_skipped = True
        strategy.step_count = 7
        strategy.epoch_count = 1

        # Epoch cadence ignores the step-ran signal.
        assert strategy._should_validate(TrainingStage.AFTER_EPOCH) is True
        assert strategy._validation_checkpoint(TrainingStage.AFTER_EPOCH) is True
