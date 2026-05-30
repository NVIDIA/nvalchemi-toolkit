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

from pathlib import Path
from typing import Any

import pytest
import torch
import zarr
from pydantic import ValidationError

from nvalchemi.data import Batch
from nvalchemi.hooks._context import TrainContext
from nvalchemi.models.base import BaseModelMixin
from nvalchemi.training import EnergyLoss, TrainingStage
from nvalchemi.training.hooks import (
    EMAHook,
    EvaluateHook,
    EvaluationZarrSink,
    MixedPrecisionHook,
    TrainingUpdateHook,
)
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


class _GradRecorderHook:
    """Hook that records gradient magnitudes before optimizer stepping."""

    stage = TrainingStage.BEFORE_OPTIMIZER_STEP
    frequency = 1

    def __init__(self) -> None:
        """Initialize the observed gradient-magnitude list."""
        self.grad_sums: list[float] = []

    def __call__(self, ctx: TrainContext, stage: TrainingStage) -> None:
        """Record the aggregate absolute gradient visible to the optimizer."""
        total = 0.0
        for parameter in ctx.workflow.models["main"].parameters():
            if parameter.grad is not None:
                total += float(parameter.grad.detach().abs().sum())
        self.grad_sums.append(total)


class _SkipFirstOptimizerStepHook(TrainingUpdateHook):
    """Training update hook that vetoes the first optimizer step attempt."""

    priority = 10

    def __init__(self) -> None:
        """Initialize the optimizer-step attempt counter."""
        self.attempts = 0

    def __call__(
        self,
        ctx: TrainContext,
        stage: TrainingStage,
        will_skip: bool,  # noqa: ARG002
    ) -> tuple[bool, torch.Tensor | None]:
        """Veto the first optimizer-step stage and allow subsequent attempts."""
        if stage is TrainingStage.DO_OPTIMIZER_STEP:
            self.attempts += 1
            return self.attempts > 1, ctx.loss
        return True, ctx.loss


class _RecordingEvaluationSink:
    """Evaluation sink test double that records every granular call."""

    def __init__(self) -> None:
        self.entered = False
        self.exited = False
        self.begin_calls: list[dict[str, int | str]] = []
        self.sample_batches: list[tuple[Batch, dict[str, int]]] = []
        self.batch_summaries: list[tuple[Batch, dict[str, int]]] = []
        self.epoch_summaries: list[tuple[Batch, dict[str, Any]]] = []
        self.end_calls: list[dict[str, int | str]] = []

    def __enter__(self) -> "_RecordingEvaluationSink":
        self.entered = True
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: Any,
    ) -> None:
        del exc_type, exc, tb
        self.exited = True

    def begin_evaluation(self, *, step_count: int, epoch: int, name: str) -> None:
        self.begin_calls.append(
            {"step_count": step_count, "epoch": epoch, "name": name}
        )

    def write_samples(
        self,
        batch: Batch,
        *,
        step_count: int,
        epoch: int,
        batch_count: int,
    ) -> None:
        self.sample_batches.append(
            (
                batch,
                {"step_count": step_count, "epoch": epoch, "batch_count": batch_count},
            )
        )

    def write_batch_summary(
        self,
        batch: Batch,
        *,
        step_count: int,
        epoch: int,
        batch_count: int,
    ) -> None:
        self.batch_summaries.append(
            (
                batch,
                {"step_count": step_count, "epoch": epoch, "batch_count": batch_count},
            )
        )

    def write_epoch_summary(
        self,
        batch: Batch,
        *,
        step_count: int,
        epoch: int,
        local_summary: dict[str, torch.Tensor],
        global_summary: dict[str, torch.Tensor],
    ) -> None:
        self.epoch_summaries.append(
            (
                batch,
                {
                    "step_count": step_count,
                    "epoch": epoch,
                    "local_summary": local_summary,
                    "global_summary": global_summary,
                },
            )
        )

    def end_evaluation(self, *, step_count: int, epoch: int, name: str) -> None:
        self.end_calls.append({"step_count": step_count, "epoch": epoch, "name": name})


class _WriteOnlySink:
    """Minimal DataSink-like sink for fallback write-path tests."""

    def __init__(self) -> None:
        self.batches: list[Batch] = []

    def write(self, batch: Batch) -> None:
        self.batches.append(batch)


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

    def test_default_epoch_schedule_runs_at_training_end_for_num_steps(
        self, batch: Batch
    ) -> None:
        hook = EvaluateHook(
            validation_data=[batch],
            validation_fn=energy_only_training_fn,
            loss_fn=EnergyLoss(),
            grad_mode="disabled",
        )
        strategy = TrainingStrategy(
            **{
                **_energy_strategy_kwargs(),
                "num_epochs": None,
                "num_steps": 1,
                "hooks": [hook],
            }
        )

        strategy.run([batch, batch])

        assert strategy.validation is not None
        assert strategy.validation["num_batches"] == 1

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

    def test_every_n_steps_skips_vetoed_optimizer_steps(self, batch: Batch) -> None:
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
            every_n_steps=1,
            grad_mode="disabled",
        )
        skip_first = _SkipFirstOptimizerStepHook()
        strategy = TrainingStrategy(
            **{
                **_energy_strategy_kwargs(),
                "num_epochs": None,
                "num_steps": 1,
                "hooks": [skip_first, hook],
            }
        )

        strategy.run([batch, batch])

        assert skip_first.attempts == 2
        assert len(calls) == 1

    def test_every_n_epochs_uses_completed_epoch_count(self, batch: Batch) -> None:
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
            every_n_epochs=2,
            grad_mode="disabled",
        )
        strategy = TrainingStrategy(
            **{
                **_energy_strategy_kwargs(),
                "num_epochs": 2,
                "hooks": [hook],
            }
        )

        strategy.run([batch])

        assert len(calls) == 1

    def test_gradient_validation_preserves_optimizer_gradients(
        self, batch: Batch
    ) -> None:
        recorder = _GradRecorderHook()
        hook = EvaluateHook(
            validation_data=[batch],
            stage=TrainingStage.BEFORE_OPTIMIZER_STEP,
            grad_mode="auto",
        )
        strategy = TrainingStrategy(
            **{**_build_baseline_strategy_kwargs(), "hooks": [hook, recorder]}
        )

        strategy.run([batch])

        assert recorder.grad_sums
        assert recorder.grad_sums[0] > 0.0

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

    def test_step_validation_uses_ema_when_registered_before_ema_hook(
        self, batch: Batch
    ) -> None:
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
            **{**_energy_strategy_kwargs(), "hooks": [hook, ema]}
        )

        strategy.run([batch])

        averaged = ema.get_averaged_model().module
        assert seen_model == [averaged]
        assert strategy.hooks[0].__class__.__name__ == "TrainingUpdateOrchestrator"

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

        with pytest.raises(ValidationError, match="MixedPrecisionHook"):
            TrainingStrategy(**{**_energy_strategy_kwargs(), "hooks": [hook]})


class TestEvaluateHookSinks:
    """Evaluation sink output behavior."""

    def test_sink_receives_augmented_batches_and_summaries(self, batch: Batch) -> None:
        sink = _RecordingEvaluationSink()
        hook = EvaluateHook(
            validation_data=[batch],
            validation_fn=energy_only_training_fn,
            loss_fn=EnergyLoss(),
            grad_mode="disabled",
            sink=sink,
            include_predictions=True,
            write_batch_summaries=True,
        )
        strategy = TrainingStrategy(**{**_energy_strategy_kwargs(), "hooks": [hook]})

        strategy.run([batch])

        assert sink.entered is True
        assert sink.exited is True
        assert len(sink.begin_calls) == 1
        assert len(sink.end_calls) == 1
        assert len(sink.sample_batches) == 1
        output_batch, sample_meta = sink.sample_batches[0]
        assert sample_meta["batch_count"] == 0
        assert output_batch is not batch
        assert "eval_total_loss" in output_batch
        assert "eval_loss_EnergyLoss" in output_batch
        assert "eval_component_total_EnergyLoss" in output_batch
        assert "eval_prediction_predicted_energy" in output_batch
        assert output_batch.eval_total_loss.shape[0] == batch.num_graphs
        assert (
            output_batch.eval_prediction_predicted_energy.shape[0] == batch.num_graphs
        )
        assert "eval_total_loss" not in batch
        assert len(sink.batch_summaries) == 1
        assert "eval_loss_mean_EnergyLoss" in sink.batch_summaries[0][0]
        assert len(sink.epoch_summaries) == 1
        _epoch_batch, epoch_meta = sink.epoch_summaries[0]
        assert set(epoch_meta["local_summary"]) == {"EnergyLoss", "total_loss"}
        assert set(epoch_meta["global_summary"]) == {"EnergyLoss", "total_loss"}

    def test_write_only_sink_gets_sample_batch_fallback(self, batch: Batch) -> None:
        sink = _WriteOnlySink()
        hook = EvaluateHook(
            validation_data=[batch],
            validation_fn=energy_only_training_fn,
            loss_fn=EnergyLoss(),
            grad_mode="disabled",
            sink=sink,
            write_epoch_summary=False,
        )
        strategy = TrainingStrategy(**{**_energy_strategy_kwargs(), "hooks": [hook]})

        strategy.run([batch])

        assert len(sink.batches) == 1
        assert "eval_total_loss" in sink.batches[0]
        assert "eval_loss_EnergyLoss" in sink.batches[0]

    def test_sample_writes_can_be_coalesced(self, batch: Batch) -> None:
        sink = _RecordingEvaluationSink()
        hook = EvaluateHook(
            validation_data=[batch, batch],
            validation_fn=energy_only_training_fn,
            loss_fn=EnergyLoss(),
            grad_mode="disabled",
            sink=sink,
            write_batch_size=2,
            write_epoch_summary=False,
        )
        strategy = TrainingStrategy(**{**_energy_strategy_kwargs(), "hooks": [hook]})

        strategy.run([batch])

        assert len(sink.sample_batches) == 1
        output_batch, sample_meta = sink.sample_batches[0]
        assert output_batch.num_graphs == 2 * batch.num_graphs
        assert sample_meta["batch_count"] == 0
        assert output_batch.eval_batch_index.tolist() == [0, 0, 1, 1]

    def test_zarr_sink_writes_single_store_hierarchy(
        self, tmp_path: Path, batch: Batch
    ) -> None:
        store = tmp_path / "eval.zarr"
        sink = EvaluationZarrSink(store)
        hook = EvaluateHook(
            validation_data=[batch],
            validation_fn=energy_only_training_fn,
            loss_fn=EnergyLoss(),
            grad_mode="disabled",
            sink=sink,
            include_predictions=True,
            write_batch_summaries=True,
        )
        strategy = TrainingStrategy(**{**_energy_strategy_kwargs(), "hooks": [hook]})

        strategy.run([batch])

        root = zarr.open(store, mode="r")
        step_key = next(iter(root.group_keys()))
        step_group = root[step_key]
        sample_group = step_group["0"]["0"]
        assert "eval_total_loss" in sample_group["core"]
        assert "eval_loss_EnergyLoss" in sample_group["core"]
        assert "eval_prediction_predicted_energy" in sample_group["core"]
        assert (
            "eval_loss_mean_EnergyLoss"
            in step_group["0"]["batch_summaries"]["0"]["core"]
        )
        assert step_group["rank_means"]["total_loss"].shape == (1,)
        assert step_group["rank_means"]["EnergyLoss"].shape == (1,)
        assert step_group["summary"]["total_loss"].shape == ()
        assert "summary_batch" in step_group

    def test_zarr_sink_creates_distributed_rank_mean_arrays(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, batch: Batch
    ) -> None:
        barriers = 0
        store = tmp_path / "eval.zarr"
        sink = EvaluationZarrSink(store)
        hook = EvaluateHook(
            validation_data=[batch],
            validation_fn=energy_only_training_fn,
            loss_fn=EnergyLoss(),
            grad_mode="disabled",
            sink=sink,
        )
        strategy = TrainingStrategy(**{**_energy_strategy_kwargs(), "hooks": [hook]})

        def all_reduce(tensor: torch.Tensor, op: Any = None) -> None:
            del op
            tensor.mul_(2.0)

        def barrier() -> None:
            nonlocal barriers
            barriers += 1

        monkeypatch.setattr(
            "nvalchemi.training.strategy.dist.is_available", lambda: True
        )
        monkeypatch.setattr(
            "nvalchemi.training.strategy.dist.is_initialized", lambda: True
        )
        monkeypatch.setattr("nvalchemi.training.strategy.dist.get_rank", lambda: 0)
        monkeypatch.setattr(
            "nvalchemi.training.hooks.evaluate.dist.all_reduce",
            all_reduce,
        )
        monkeypatch.setattr(
            "nvalchemi.training.hooks.evaluate.dist.is_available", lambda: True
        )
        monkeypatch.setattr(
            "nvalchemi.training.hooks.evaluate.dist.is_initialized", lambda: True
        )
        monkeypatch.setattr("nvalchemi.training.hooks.evaluate.dist.barrier", barrier)
        monkeypatch.setattr(
            "nvalchemi.training.hooks.evaluation_sinks.dist.is_available",
            lambda: True,
        )
        monkeypatch.setattr(
            "nvalchemi.training.hooks.evaluation_sinks.dist.is_initialized",
            lambda: True,
        )
        monkeypatch.setattr(
            "nvalchemi.training.hooks.evaluation_sinks.dist.get_rank", lambda: 0
        )
        monkeypatch.setattr(
            "nvalchemi.training.hooks.evaluation_sinks.dist.get_world_size", lambda: 2
        )
        monkeypatch.setattr(
            "nvalchemi.training.hooks.evaluation_sinks.dist.barrier", barrier
        )

        strategy.run([batch])

        root = zarr.open(store, mode="r")
        step_key = next(iter(root.group_keys()))
        rank_means = root[step_key]["rank_means"]
        assert rank_means["total_loss"].shape == (2,)
        assert not torch.isnan(torch.as_tensor(rank_means["total_loss"][0]))
        assert torch.isnan(torch.as_tensor(rank_means["total_loss"][1]))
        assert barriers == 2


class TestEvaluateHookDistributedSummary:
    """Distributed summary publication behavior."""

    def test_distributed_summary_uses_one_packed_all_reduce(
        self, monkeypatch: pytest.MonkeyPatch, batch: Batch
    ) -> None:
        all_reduce_shapes: list[tuple[int, ...]] = []
        hook = EvaluateHook(
            validation_data=[batch],
            validation_fn=energy_only_training_fn,
            loss_fn=EnergyLoss(),
            grad_mode="disabled",
        )
        strategy = TrainingStrategy(**{**_energy_strategy_kwargs(), "hooks": [hook]})

        def all_reduce(tensor: torch.Tensor, op: Any = None) -> None:
            all_reduce_shapes.append(tuple(tensor.shape))

        monkeypatch.setattr(
            "nvalchemi.training.strategy.dist.is_available", lambda: True
        )
        monkeypatch.setattr(
            "nvalchemi.training.strategy.dist.is_initialized", lambda: True
        )
        monkeypatch.setattr("nvalchemi.training.strategy.dist.get_rank", lambda: 0)
        monkeypatch.setattr(
            "nvalchemi.training.hooks.evaluate.dist.all_reduce",
            all_reduce,
        )

        strategy.run([batch])

        assert all_reduce_shapes == [(5,)]
        assert strategy.validation is not None
        assert strategy.validation["distributed_reduced"] is True

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
