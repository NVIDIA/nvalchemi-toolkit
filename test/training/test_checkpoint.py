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
"""Tests for checkpoint save/load utilities."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import torch
from torch import nn

from nvalchemi.training._checkpoint import (
    load_training_checkpoint,
    save_training_checkpoint,
)


class _DemoModel(nn.Module):
    """Minimal model for checkpoint tests."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class TestSaveTrainingCheckpoint:
    """Tests for :func:`save_training_checkpoint`."""

    @patch("nvalchemi.training._checkpoint._save_checkpoint")
    def test_rank_zero_saves(self, mock_save: MagicMock, tmp_path: Path) -> None:
        model = _DemoModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        metrics = {"val_loss": 0.5}

        save_training_checkpoint(
            path=tmp_path / "ckpt",
            model=model,
            optimizer=optimizer,
            epoch=3,
            metrics=metrics,
        )

        mock_save.assert_called_once()
        call_kwargs = mock_save.call_args
        assert call_kwargs.kwargs["epoch"] == 3
        assert call_kwargs.kwargs["metadata"] == {"metrics": metrics}

    @patch("nvalchemi.training._checkpoint._save_checkpoint")
    def test_non_rank_zero_skips(self, mock_save: MagicMock, tmp_path: Path) -> None:
        model = _DemoModel()
        mgr = MagicMock()
        mgr.rank = 1

        save_training_checkpoint(
            path=tmp_path / "ckpt",
            model=model,
            epoch=5,
            dist_manager=mgr,
        )

        mock_save.assert_not_called()

    @patch("nvalchemi.training._checkpoint._save_checkpoint")
    def test_default_metrics_empty(self, mock_save: MagicMock, tmp_path: Path) -> None:
        model = _DemoModel()

        save_training_checkpoint(path=tmp_path / "ckpt", model=model)

        call_kwargs = mock_save.call_args
        assert call_kwargs.kwargs["metadata"] == {"metrics": {}}


class TestLoadTrainingCheckpoint:
    """Tests for :func:`load_training_checkpoint`."""

    def test_missing_path_returns_clean_start(self, tmp_path: Path) -> None:
        model = _DemoModel()
        result = load_training_checkpoint(
            path=tmp_path / "nonexistent",
            model=model,
        )
        assert result == {"epoch": 0, "metrics": {}}

    @patch("nvalchemi.training._checkpoint._load_checkpoint")
    def test_loads_epoch_and_metrics(
        self, mock_load: MagicMock, tmp_path: Path
    ) -> None:
        ckpt_dir = tmp_path / "ckpt"
        ckpt_dir.mkdir()

        def _fake_load(
            path: str,
            models: nn.Module | None = None,
            optimizer: torch.optim.Optimizer | None = None,
            scheduler: object = None,
            scaler: object = None,
            metadata_dict: dict | None = None,
            device: str = "cpu",
        ) -> int:
            if metadata_dict is not None:
                metadata_dict["metrics"] = {"val_loss": 0.42}
            return 7

        mock_load.side_effect = _fake_load

        model = _DemoModel()
        result = load_training_checkpoint(path=ckpt_dir, model=model, device="cpu")

        assert result["epoch"] == 7
        assert result["metrics"] == {"val_loss": 0.42}
        mock_load.assert_called_once()

    @patch("nvalchemi.training._checkpoint._load_checkpoint")
    def test_missing_metrics_key_defaults_empty(
        self, mock_load: MagicMock, tmp_path: Path
    ) -> None:
        ckpt_dir = tmp_path / "ckpt"
        ckpt_dir.mkdir()

        mock_load.return_value = 2

        model = _DemoModel()
        result = load_training_checkpoint(path=ckpt_dir, model=model)

        assert result["epoch"] == 2
        assert result["metrics"] == {}
