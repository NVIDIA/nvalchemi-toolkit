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
from pathlib import Path
from typing import Any

import pytest

from nvalchemi.training import CheckpointHook, TrainingStrategy, load_checkpoint


class TestCheckpointHookConstruction:
    """Validate checkpoint hook configuration."""

    def test_requires_step_or_epoch_interval(self, tmp_path: Path) -> None:
        """A checkpoint hook needs at least one cadence."""
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
