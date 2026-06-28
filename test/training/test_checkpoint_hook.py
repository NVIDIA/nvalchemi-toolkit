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
"""Tests for periodic training checkpoint hooks."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import distributed as dist

from nvalchemi.training import (
    CheckpointHook,
    EMAHook,
    OptimizerConfig,
    TrainingStrategy,
    load_checkpoint,
)
from test.training.conftest import _build_baseline_strategy_kwargs


def _model_parameter_vector(strategy: TrainingStrategy) -> torch.Tensor:
    """Return a detached flat parameter vector for the strategy's main model."""
    return torch.cat(
        [
            param.detach().cpu().reshape(-1)
            for param in strategy.models["main"].parameters()
        ]
    )


def _ema_state_dict(hook: EMAHook) -> dict[str, Any]:
    """Return a detached CPU snapshot of an initialized EMA wrapper."""
    return {
        key: value.detach().cpu().clone() if isinstance(value, torch.Tensor) else value
        for key, value in hook.get_averaged_model().state_dict().items()
    }


def _assert_state_dict_close(
    actual: Mapping[str, Any],
    expected: Mapping[str, Any],
) -> None:
    """Assert two state dictionaries contain equal scalar and tensor values."""
    assert actual.keys() == expected.keys()
    for key, value in actual.items():
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(value, expected[key], msg=f"state {key!r}")
        else:
            assert value == expected[key]


def _ema_restart_strategy_kwargs() -> dict[str, Any]:
    """Return deterministic strategy kwargs for interrupted-run comparisons."""
    return {
        **_build_baseline_strategy_kwargs(),
        "optimizer_configs": OptimizerConfig(
            optimizer_cls=torch.optim.SGD,
            optimizer_kwargs={"lr": 1e-3},
        ),
    }


def _init_single_process_group(tmp_path: Path) -> None:
    """Initialize a single-rank process group for CPU DDP tests."""
    init_file = tmp_path / "ddp_init"
    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=0,
        world_size=1,
    )


class TestCheckpointHookConstruction:
    """Validate checkpoint hook configuration."""

    def test_without_interval_is_rejected(self, tmp_path: Path) -> None:
        """A checkpoint hook requires one explicit save cadence."""
        with pytest.raises(ValueError, match="exactly one"):
            CheckpointHook(tmp_path)

    def test_rejects_step_and_epoch_interval_together(self, tmp_path: Path) -> None:
        """A single checkpoint hook owns one cadence policy."""
        with pytest.raises(ValueError, match="exactly one"):
            CheckpointHook(tmp_path, step_interval=10, epoch_interval=1)

    @pytest.mark.parametrize("field", ["step_interval", "epoch_interval"])
    def test_interval_must_be_positive(self, tmp_path: Path, field: str) -> None:
        """Configured checkpoint cadences must be positive."""
        with pytest.raises(ValueError, match="greater than 0"):
            CheckpointHook(tmp_path, **{field: 0})


class TestCheckpointHookCadence:
    """Verify periodic checkpoint saves from a running strategy."""

    def test_step_interval_saves_restartable_checkpoints(
        self,
        tmp_path: Path,
        baseline_strategy_kwargs: dict[str, Any],
        dataset: list[Any],
    ) -> None:
        """Step cadence writes restart checkpoints at completed optimizer steps."""
        hook = CheckpointHook(tmp_path, step_interval=2, async_save=False)
        strategy = TrainingStrategy(
            **{
                **baseline_strategy_kwargs,
                "num_epochs": None,
                "num_steps": 4,
                "hooks": [hook],
            }
        )

        strategy.run(dataset)

        assert hook.last_checkpoint_index == 1
        assert (tmp_path / "models" / "main" / "checkpoints" / "0.pt").is_file()
        assert (tmp_path / "models" / "main" / "checkpoints" / "1.pt").is_file()
        first = load_checkpoint(tmp_path, checkpoint_index=0)["strategy"]
        second = load_checkpoint(tmp_path, checkpoint_index=1)["strategy"]
        assert first.step_count == 2
        assert second.step_count == 4

    def test_epoch_interval_saves_completed_epoch_state(
        self,
        tmp_path: Path,
        baseline_strategy_kwargs: dict[str, Any],
        dataset: list[Any],
    ) -> None:
        """Epoch cadence saves after epoch counters have advanced."""
        hook = CheckpointHook(tmp_path, epoch_interval=1, async_save=False)
        strategy = TrainingStrategy(
            **{
                **baseline_strategy_kwargs,
                "num_epochs": 2,
                "hooks": [hook],
            }
        )

        strategy.run(dataset)

        assert hook.last_checkpoint_index == 1
        first_metadata = json.loads(
            (tmp_path / "strategy" / "checkpoints" / "0.json").read_text()
        )
        second_metadata = json.loads(
            (tmp_path / "strategy" / "checkpoints" / "1.json").read_text()
        )
        assert first_metadata["runtime_state"]["epoch_count"] == 1
        assert first_metadata["runtime_state"]["epoch_step_count"] == 0
        assert second_metadata["runtime_state"]["epoch_count"] == 2
        assert second_metadata["runtime_state"]["epoch_step_count"] == 0

    def test_async_save_flushes_on_strategy_exit(
        self,
        tmp_path: Path,
        baseline_strategy_kwargs: dict[str, Any],
        dataset: list[Any],
    ) -> None:
        """Async checkpoint writes finish before ``TrainingStrategy.run`` returns."""
        hook = CheckpointHook(tmp_path, step_interval=1)
        strategy = TrainingStrategy(
            **{
                **baseline_strategy_kwargs,
                "num_epochs": None,
                "num_steps": 1,
                "hooks": [hook],
            }
        )

        strategy.run(dataset)

        assert hook.last_checkpoint_index == 0
        restored = load_checkpoint(tmp_path)["strategy"]
        assert restored.step_count == 1

    def test_restarted_strategy_continues_periodic_checkpoint_round_trip(
        self,
        tmp_path: Path,
        baseline_strategy_kwargs: dict[str, Any],
        dataset: list[Any],
    ) -> None:
        """Repeated save-load cycles preserve updated restart state."""
        strategy = TrainingStrategy(
            **{
                **baseline_strategy_kwargs,
                "num_epochs": None,
                "num_steps": 1,
                "hooks": [
                    CheckpointHook(tmp_path, step_interval=1, async_save=False),
                ],
            }
        )
        previous_params = _model_parameter_vector(strategy)

        for checkpoint_index in range(3):
            strategy.num_steps = strategy.step_count + 1
            strategy.run([dataset[checkpoint_index]])

            current_params = _model_parameter_vector(strategy)
            assert not torch.allclose(current_params, previous_params)

            loaded = load_checkpoint(
                tmp_path,
                checkpoint_index=checkpoint_index,
                hooks=[
                    CheckpointHook(tmp_path, step_interval=1, async_save=False),
                ],
            )
            restored = loaded["strategy"]

            assert loaded["checkpoint_index"] == checkpoint_index
            assert restored.step_count == checkpoint_index + 1
            assert restored.batch_count == checkpoint_index + 1
            assert restored._resume_optimizer_state is True
            assert restored._optimizers[0].state_dict()["state"]
            torch.testing.assert_close(
                _model_parameter_vector(restored),
                current_params,
            )

            strategy = restored
            previous_params = current_params

    def test_periodic_checkpoint_restores_ema_hook_state(
        self,
        tmp_path: Path,
        baseline_strategy_kwargs: dict[str, Any],
        dataset: list[Any],
    ) -> None:
        """Periodic checkpoints restore checkpointable EMA hook state."""
        ema = EMAHook(model_key="main", decay=0.5)
        strategy = TrainingStrategy(
            **{
                **baseline_strategy_kwargs,
                "num_epochs": None,
                "num_steps": 2,
                "hooks": [
                    ema,
                    CheckpointHook(tmp_path, step_interval=1, async_save=False),
                ],
            }
        )

        strategy.run(dataset)
        saved_state = ema.state_dict()

        restored_ema = EMAHook(model_key="main", decay=0.5)
        restored = TrainingStrategy(
            **{
                **baseline_strategy_kwargs,
                "num_epochs": None,
                "num_steps": 2,
                "hooks": [
                    restored_ema,
                    CheckpointHook(tmp_path, step_interval=1, async_save=False),
                ],
            }
        )
        restored.restore_checkpoint(tmp_path, checkpoint_index=1)

        assert restored.step_count == 2
        assert restored_ema.num_updates == ema.num_updates
        assert restored_ema._averaged_model is None
        assert restored_ema._pending_averaged_state is not None

        saved_average = saved_state["averaged_model_state"]
        for key, value in restored_ema._pending_averaged_state.items():
            torch.testing.assert_close(value, saved_average[key])

    def test_restarted_training_matches_uninterrupted_ema_average(
        self,
        tmp_path: Path,
        dataset: list[Any],
    ) -> None:
        """EMA average continues exactly across a strategy checkpoint restart."""
        full_dataset = dataset[:3]
        decay = 0.5

        reference_ema = EMAHook(model_key="main", decay=decay)
        reference = TrainingStrategy(
            **{
                **_ema_restart_strategy_kwargs(),
                "num_epochs": None,
                "num_steps": 3,
                "hooks": [reference_ema],
            }
        )
        reference.run(full_dataset)
        expected_params = _model_parameter_vector(reference)
        expected_ema_state = _ema_state_dict(reference_ema)

        checkpoint_ema = EMAHook(model_key="main", decay=decay)
        checkpointed = TrainingStrategy(
            **{
                **_ema_restart_strategy_kwargs(),
                "num_epochs": None,
                "num_steps": 2,
                "hooks": [
                    checkpoint_ema,
                    CheckpointHook(tmp_path, step_interval=2, async_save=False),
                ],
            }
        )
        checkpointed.run(full_dataset)

        assert checkpoint_ema.num_updates == 2
        assert (tmp_path / "hooks" / "checkpoints" / "0.pt").is_file()

        restored_ema = EMAHook(model_key="main", decay=decay)
        restored = TrainingStrategy.load_checkpoint(
            tmp_path,
            checkpoint_index=0,
            hooks=[restored_ema],
        )
        assert restored.step_count == 2
        assert restored_ema._averaged_model is None
        assert restored_ema._pending_averaged_state is not None

        restored.num_steps = 3
        restored.run(full_dataset)

        assert restored.step_count == reference.step_count
        assert restored_ema.num_updates == reference_ema.num_updates
        torch.testing.assert_close(_model_parameter_vector(restored), expected_params)
        _assert_state_dict_close(_ema_state_dict(restored_ema), expected_ema_state)

    @pytest.mark.skipif(not dist.is_gloo_available(), reason="gloo backend required")
    def test_ddp_wrapped_strategy_saves_unwrapped_model_state(
        self,
        tmp_path: Path,
        baseline_strategy_kwargs: dict[str, Any],
        dataset: list[Any],
    ) -> None:
        """DDP checkpoints save the underlying model, not ``module.`` keys."""
        if dist.is_initialized():
            pytest.skip("test requires ownership of the process group")
        checkpoint_dir = tmp_path / "checkpoints"
        _init_single_process_group(tmp_path)
        try:
            strategy = TrainingStrategy(
                **{
                    **baseline_strategy_kwargs,
                    "num_epochs": None,
                    "num_steps": 1,
                    "hooks": [
                        CheckpointHook(
                            checkpoint_dir,
                            step_interval=1,
                            async_save=False,
                        ),
                    ],
                }
            )
            strategy.models["main"] = torch.nn.parallel.DistributedDataParallel(
                strategy.models["main"]
            )

            strategy.run([dataset[0]])

            weights = torch.load(
                checkpoint_dir / "models" / "main" / "checkpoints" / "0.pt",
                weights_only=True,
            )
            assert all(not key.startswith("module.") for key in weights)
            restored = load_checkpoint(checkpoint_dir)["strategy"]
            torch.testing.assert_close(
                _model_parameter_vector(restored),
                _model_parameter_vector(strategy),
            )
        finally:
            if dist.is_initialized():
                dist.destroy_process_group()

    def test_native_checkpoint_rejects_fsdp_wrapped_model(
        self,
        monkeypatch: pytest.MonkeyPatch,
        demo_model: torch.nn.Module,
    ) -> None:
        """FSDP/FSDP2 models fail clearly until DCP support is implemented."""
        from nvalchemi.training import _checkpoint

        monkeypatch.setattr(_checkpoint, "_is_fsdp_wrapped", lambda module: True)

        with pytest.raises(
            NotImplementedError,
            match="torch.distributed.checkpoint",
        ):
            _checkpoint._checkpoint_model(demo_model)
