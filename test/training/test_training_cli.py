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
"""Tests for the training Click/Rich CLI."""

from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner

from nvalchemi.training._spec import create_model_spec
from nvalchemi.training.cli import _load_job_spec, _lr_series, main
from nvalchemi.training.hooks import CheckpointHook


def _combined_output(result) -> str:
    """Return stdout and stderr from a Click test result."""
    return result.output + getattr(result, "stderr", "")


def test_schema_dump_outputs_job_schema() -> None:
    """``schema dump`` prints the CLI training job schema as JSON."""
    result = CliRunner().invoke(main, ["schema", "dump"])

    assert result.exit_code == 0, result.output
    schema = json.loads(result.output)
    assert schema["title"] == "TrainingJobSpec"
    assert "source" in schema["properties"]
    assert "strategy" in schema["properties"]
    source_ref = schema["properties"]["source"]["$ref"].split("/")[-1]
    source_schema = schema["$defs"][source_ref]
    assert "scratch" not in source_schema["properties"]["model"]["enum"]
    assert "hooks" in source_schema["properties"]
    assert "workflow" in schema["properties"]
    assert (
        "FineTuningStrategy.to_spec_dict"
        in schema["properties"]["strategy"]["description"]
    )


def test_checkpoint_init_writes_valid_spec(tmp_path: Path) -> None:
    """``finetune init checkpoint`` writes a native-checkpoint spec."""
    output = tmp_path / "finetune.json"
    result = CliRunner().invoke(
        main,
        [
            "finetune",
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
    spec = _load_job_spec(output)
    assert spec.workflow == "finetune"
    assert spec.source.model == "native-checkpoint"
    assert spec.source.checkpoint_path == "runs/pretrain/checkpoints"
    assert spec.dataset.path == "data/train.zarr"
    assert spec.output.checkpoint_dir == "runs/ft/checkpoints"
    assert spec.strategy["trainable_patterns"] == ["main.model.readout.*"]


def test_load_job_spec_accepts_deprecated_endpoint_key(tmp_path: Path) -> None:
    """Older CLI specs using ``source.endpoint`` normalize to ``source.model``."""
    output = tmp_path / "finetune.json"
    result = CliRunner().invoke(
        main,
        [
            "finetune",
            "init",
            "checkpoint",
            "runs/pretrain/checkpoints",
            "--dataset",
            "data/train.zarr",
            "--output-dir",
            "runs/ft",
            "--out",
            str(output),
        ],
    )
    assert result.exit_code == 0, _combined_output(result)
    payload = json.loads(output.read_text())
    payload["source"]["endpoint"] = payload["source"].pop("model")
    output.write_text(json.dumps(payload))

    spec = _load_job_spec(output)

    assert spec.source.model == "native-checkpoint"
    assert "endpoint" not in spec.source.model_dump()


def test_job_spec_accepts_serialized_hook_specs(tmp_path: Path) -> None:
    """Source specs can carry runtime hooks serialized with ``create_model_spec``."""
    output = tmp_path / "finetune.json"
    result = CliRunner().invoke(
        main,
        [
            "finetune",
            "init",
            "checkpoint",
            "runs/pretrain/checkpoints",
            "--dataset",
            "data/train.zarr",
            "--output-dir",
            "runs/ft",
            "--out",
            str(output),
        ],
    )
    assert result.exit_code == 0, _combined_output(result)
    payload = json.loads(output.read_text())
    hook_spec = create_model_spec(
        CheckpointHook,
        checkpoint_dir="runs/ft/checkpoints",
        step_interval=10,
        async_save=False,
    ).model_dump(mode="json")
    payload["source"]["hooks"] = [hook_spec]
    output.write_text(json.dumps(payload))

    spec = _load_job_spec(output)

    assert spec.source.hooks[0]["cls_path"].endswith("CheckpointHook")


def test_scratch_init_writes_training_from_scratch_spec(tmp_path: Path) -> None:
    """``train init`` writes a training-from-scratch spec using the job wrapper."""
    output = tmp_path / "scratch.json"
    result = CliRunner().invoke(
        main,
        [
            "train",
            "init",
            "--dataset",
            "data/train.zarr",
            "--output-dir",
            "runs/scratch",
            "--out",
            str(output),
        ],
    )

    assert result.exit_code == 0, _combined_output(result)
    spec = _load_job_spec(output)
    assert spec.workflow == "train"
    assert spec.source.model == "custom"
    assert spec.strategy["model_specs"] == {}
    assert spec.dataset.path == "data/train.zarr"


def test_repeated_dataset_options_write_multidataset_spec(tmp_path: Path) -> None:
    """Repeated ``--dataset`` options map to a MultiDataset intent."""
    output = tmp_path / "multidataset.json"
    result = CliRunner().invoke(
        main,
        [
            "finetune",
            "init",
            "mace",
            "small-0b",
            "--dataset",
            "data/domain-a.zarr",
            "--dataset",
            "data/domain-b.zarr",
            "--output-dir",
            "runs/mace-ft",
            "--out",
            str(output),
        ],
    )

    assert result.exit_code == 0, _combined_output(result)
    payload = json.loads(output.read_text())
    assert payload["dataset"]["format"] == "alchemi-zarr-multidataset"
    assert payload["dataset"]["paths"] == [
        "data/domain-a.zarr",
        "data/domain-b.zarr",
    ]
    assert "path" not in payload["dataset"]
    spec = _load_job_spec(output)
    assert spec.dataset.path is None
    assert spec.dataset.paths == ["data/domain-a.zarr", "data/domain-b.zarr"]


def test_help_includes_common_workflow_examples() -> None:
    """Top-level help shows practical workflow examples."""
    result = CliRunner().invoke(main, ["--help"])

    assert result.exit_code == 0, _combined_output(result)
    rendered = _combined_output(result)
    assert "Fine-tune MACE on an ALCHEMI dataset" in rendered
    assert "It does not execute" in rendered
    assert "training runs yet" in rendered
    assert "--model aimnet2" in rendered


def test_documented_init_examples_are_executable(tmp_path: Path) -> None:
    """Help examples for scaffold commands include required options."""
    runner = CliRunner()
    examples = [
        [
            "finetune",
            "init",
            "mace",
            "small-0b",
            "--dataset",
            "data/domain.zarr",
            "--output-dir",
            "runs/mace-ft",
            "--out",
            str(tmp_path / "mace-ft.json"),
        ],
        [
            "finetune",
            "init",
            "mace",
            "small-0b",
            "--dataset",
            "data/a.zarr",
            "--dataset",
            "data/b.zarr",
            "--output-dir",
            "runs/mace-ft",
            "--out",
            str(tmp_path / "multi-ft.json"),
        ],
        [
            "train",
            "init",
            "--dataset",
            "data/train.zarr",
            "--output-dir",
            "runs/train",
            "--out",
            str(tmp_path / "train.json"),
        ],
        [
            "finetune",
            "init",
            "checkpoint",
            "runs/pretrain/checkpoints",
            "--dataset",
            "data/domain.zarr",
            "--output-dir",
            "runs/domain-ft",
            "--out",
            str(tmp_path / "checkpoint-ft.json"),
        ],
    ]
    for args in examples:
        result = runner.invoke(main, args)
        assert result.exit_code == 0, _combined_output(result)


def test_schema_template_dumps_aimnet2_finetuning_config() -> None:
    """``schema template`` can dump an AIMNet2 fine-tuning template."""
    result = CliRunner().invoke(
        main, ["schema", "template", "--workflow", "finetune", "--model", "aimnet2"]
    )

    assert result.exit_code == 0, _combined_output(result)
    payload = json.loads(result.output)
    assert payload["workflow"] == "finetune"
    assert payload["source"]["model"] == "aimnet2"
    assert payload["source"]["model_id"] == "aimnet2-example"


def test_report_renders_intent_and_lr_plot(tmp_path: Path) -> None:
    """``spec report`` validates a spec and renders source, data, output, and LR intent."""
    output = tmp_path / "finetune.json"
    runner = CliRunner()
    init_result = runner.invoke(
        main,
        [
            "finetune",
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
    dataset_path = tmp_path / "domain.zarr"
    dataset_path.mkdir()
    payload = json.loads(output.read_text())
    payload["dataset"]["path"] = str(dataset_path)
    output.write_text(json.dumps(payload))

    result = runner.invoke(main, ["spec", "report", str(output)])

    assert result.exit_code == 0, _combined_output(result)
    rendered = _combined_output(result)
    assert "Training Intent" in rendered
    assert "mace" in rendered
    assert dataset_path.name in rendered
    assert "Learning-rate preview" in rendered
    assert "Warnings" in rendered


def test_report_warns_about_common_finetuning_mistakes(tmp_path: Path) -> None:
    """``spec report`` flags common MACE fine-tuning mistakes."""
    output = tmp_path / "finetune.json"
    runner = CliRunner()
    init_result = runner.invoke(
        main,
        [
            "finetune",
            "init",
            "mace",
            "small-0b",
            "--dataset",
            "data/domain.zarr",
            "--output-dir",
            "runs/mace-ft",
            "--lr",
            "0.001",
            "--out",
            str(output),
        ],
    )
    assert init_result.exit_code == 0, _combined_output(init_result)
    dataset_path = tmp_path / "domain.zarr"
    dataset_path.mkdir()
    payload = json.loads(output.read_text())
    payload["dataset"]["path"] = str(dataset_path)
    payload["source"]["compile_model"] = True
    output.write_text(json.dumps(payload))

    result = runner.invoke(main, ["spec", "report", str(output)])

    assert result.exit_code == 0, _combined_output(result)
    rendered = _combined_output(result)
    assert "Warnings" in rendered
    assert "MACE compile" in rendered
    assert "Learning rate" in rendered
    assert "Validation data" in rendered


def test_report_warns_about_missing_dataset_path(tmp_path: Path) -> None:
    """``spec report`` warns when local dataset paths are missing."""
    output = tmp_path / "finetune.json"
    init_result = CliRunner().invoke(
        main,
        [
            "finetune",
            "init",
            "mace",
            "small-0b",
            "--dataset",
            str(tmp_path / "missing.zarr"),
            "--output-dir",
            "runs/mace-ft",
            "--out",
            str(output),
        ],
    )
    assert init_result.exit_code == 0, _combined_output(init_result)

    result = CliRunner().invoke(main, ["spec", "report", str(output)])

    assert result.exit_code == 0, _combined_output(result)
    rendered = _combined_output(result)
    assert "Missing local path" in rendered
    assert "dataset.path" in rendered


def test_report_warns_about_missing_multidataset_path(tmp_path: Path) -> None:
    """``spec report`` warns about every missing local multidataset path."""
    output = tmp_path / "finetune.json"
    existing = tmp_path / "domain-a.zarr"
    existing.mkdir()
    missing = tmp_path / "domain-b.zarr"
    init_result = CliRunner().invoke(
        main,
        [
            "finetune",
            "init",
            "mace",
            "small-0b",
            "--dataset",
            str(existing),
            "--dataset",
            str(missing),
            "--output-dir",
            "runs/mace-ft",
            "--out",
            str(output),
        ],
    )
    assert init_result.exit_code == 0, _combined_output(init_result)

    result = CliRunner().invoke(main, ["spec", "report", str(output)])

    assert result.exit_code == 0, _combined_output(result)
    rendered = _combined_output(result)
    assert "Missing local path" in rendered
    assert "dataset.paths[1]" in rendered


def test_report_warns_about_missing_source_checkpoint_path(tmp_path: Path) -> None:
    """``spec report`` warns when local source checkpoint paths are missing."""
    output = tmp_path / "finetune.json"
    dataset_path = tmp_path / "domain.zarr"
    dataset_path.mkdir()
    init_result = CliRunner().invoke(
        main,
        [
            "finetune",
            "init",
            "checkpoint",
            str(tmp_path / "missing-checkpoint"),
            "--dataset",
            str(dataset_path),
            "--output-dir",
            "runs/domain-ft",
            "--out",
            str(output),
        ],
    )
    assert init_result.exit_code == 0, _combined_output(init_result)

    result = CliRunner().invoke(main, ["spec", "report", str(output)])

    assert result.exit_code == 0, _combined_output(result)
    rendered = _combined_output(result)
    assert "Missing local path" in rendered
    assert "source.checkpoint_path" in rendered


def test_report_rejects_missing_native_checkpoint(tmp_path: Path) -> None:
    """``spec report`` fails with model-specific source validation errors."""
    path = tmp_path / "bad.json"
    payload = {
        "workflow": "finetune",
        "source": {"model": "native-checkpoint"},
        "dataset": {"path": "data/train.zarr"},
        "output": {"run_dir": "runs/ft"},
        "strategy": {},
    }
    path.write_text(json.dumps(payload))

    result = CliRunner().invoke(main, ["spec", "report", str(path)])

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
