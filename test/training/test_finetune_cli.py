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
"""Tests for the fine-tuning Click/Rich CLI."""

from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner

from nvalchemi.training.cli.finetune import FineTuningJobSpec, _lr_series, main


def _combined_output(result) -> str:
    """Return stdout and stderr from a Click test result."""
    return result.output + getattr(result, "stderr", "")


def test_schema_dump_outputs_job_schema() -> None:
    """``schema dump`` prints the CLI fine-tuning job schema as JSON."""
    result = CliRunner().invoke(main, ["schema", "dump"])

    assert result.exit_code == 0, result.output
    schema = json.loads(result.output)
    assert schema["title"] == "FineTuningJobSpec"
    assert "source" in schema["properties"]
    assert "strategy" in schema["properties"]


def test_checkpoint_init_writes_valid_spec(tmp_path: Path) -> None:
    """``init checkpoint`` writes a native-checkpoint fine-tuning spec."""
    output = tmp_path / "finetune.json"
    result = CliRunner().invoke(
        main,
        [
            "init",
            "checkpoint",
            "runs/pretrain/checkpoints",
            "--dataset",
            "data/train.zarr",
            "--output-dir",
            "runs/ft",
            "--trainable-pattern",
            "main.model.readout.*",
            "--out",
            str(output),
        ],
    )

    assert result.exit_code == 0, _combined_output(result)
    spec = FineTuningJobSpec.model_validate(json.loads(output.read_text()))
    assert spec.source.endpoint == "native-checkpoint"
    assert spec.source.checkpoint_path == "runs/pretrain/checkpoints"
    assert spec.dataset.path == "data/train.zarr"
    assert spec.output.checkpoint_dir == "runs/ft/checkpoints"
    assert spec.strategy["trainable_patterns"] == ["main.model.readout.*"]


def test_report_renders_intent_and_lr_plot(tmp_path: Path) -> None:
    """``spec report`` validates a spec and renders source, data, output, and LR intent."""
    output = tmp_path / "finetune.json"
    runner = CliRunner()
    init_result = runner.invoke(
        main,
        [
            "init",
            "mace",
            "small-0b",
            "--dataset",
            "data/domain.zarr",
            "--output-dir",
            "runs/mace-ft",
            "--out",
            str(output),
        ],
    )
    assert init_result.exit_code == 0, _combined_output(init_result)

    result = runner.invoke(main, ["spec", "report", str(output)])

    assert result.exit_code == 0, _combined_output(result)
    rendered = _combined_output(result)
    assert "Fine-tuning Intent" in rendered
    assert "mace" in rendered
    assert "data/domain.zarr" in rendered
    assert "Learning-rate preview" in rendered


def test_validate_rejects_missing_native_checkpoint(tmp_path: Path) -> None:
    """``spec validate`` fails with endpoint-specific source validation errors."""
    path = tmp_path / "bad.json"
    payload = {
        "source": {"endpoint": "native-checkpoint"},
        "dataset": {"path": "data/train.zarr"},
        "output": {"run_dir": "runs/ft"},
        "strategy": {},
    }
    path.write_text(json.dumps(payload))

    result = CliRunner().invoke(main, ["spec", "validate", str(path)])

    assert result.exit_code != 0
    assert "source.checkpoint_path" in _combined_output(result)


def test_lr_series_approximates_step_lr_schedule() -> None:
    """Learning-rate previews reflect supported scheduler metadata."""
    strategy = {
        "num_steps": 4,
        "optimizer_configs": {
            "main": [
                {
                    "optimizer_kwargs": {"lr": 1.0},
                    "scheduler_cls": "torch.optim.lr_scheduler.StepLR",
                    "scheduler_kwargs": {"step_size": 2, "gamma": 0.5},
                }
            ]
        },
    }

    series = dict(_lr_series(strategy, samples=5))

    assert series[0] == 1.0
    assert series[2] == 0.5
    assert series[4] == 0.25
