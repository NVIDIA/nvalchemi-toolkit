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
import sys
import types
from pathlib import Path
from typing import Any

import pytest
import torch
from click.testing import CliRunner
from pydantic import ValidationError

import nvalchemi.training.cli as training_cli
from nvalchemi.hooks import NeighborListHook
from nvalchemi.models.base import NeighborConfig
from nvalchemi.training import TrainingStage, TrainingStrategy
from nvalchemi.training._spec import create_model_spec
from nvalchemi.training.cli import TrainingJobSpec, _load_job_spec, _lr_series, main
from nvalchemi.training.hooks import CheckpointHook, MixedPrecisionHook


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
    hooks_schema = source_schema["properties"]["hooks"]
    hook_ref = hooks_schema["items"]["$ref"].split("/")[-1]
    hook_schema = schema["$defs"][hook_ref]
    assert {"spec", "stages"} <= set(hook_schema["properties"])
    spec_ref = hook_schema["properties"]["spec"]["$ref"].split("/")[-1]
    spec_schema = schema["$defs"][spec_ref]
    assert {"cls_path", "timestamp"} <= set(spec_schema["properties"])
    assert spec_schema["additionalProperties"] is True
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


def test_finetune_init_writes_loss_dtype_policy(tmp_path: Path) -> None:
    """Fine-tuning scaffolds persist the requested composed-loss dtype policy."""
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
            "--loss-dtype-policy",
            "prediction_to_target",
            "--out",
            str(output),
        ],
    )

    assert result.exit_code == 0, _combined_output(result)
    payload = json.loads(output.read_text())
    assert payload["strategy"]["loss_fn_spec"]["dtype_policy"] == "prediction_to_target"
    spec = _load_job_spec(output)
    assert spec.strategy["loss_fn_spec"]["dtype_policy"] == "prediction_to_target"


def test_report_renders_mace_atomic_energy_options(tmp_path: Path) -> None:
    """``source.mace`` carries MACE-only atomic energy overrides."""
    output = tmp_path / "mace-ft.json"
    result = CliRunner().invoke(
        main,
        [
            "finetune",
            "init",
            "mace",
            "small-0b",
            "--dataset",
            "data/train.zarr",
            "--output-dir",
            "runs/mace-ft",
            "--out",
            str(output),
        ],
    )
    assert result.exit_code == 0, _combined_output(result)
    payload = json.loads(output.read_text())
    payload["source"]["mace"] = {"atomic_energies": {"1": -1.0, "8": -2.0}}
    output.write_text(json.dumps(payload))

    report = CliRunner().invoke(main, ["spec", "report", str(output)])

    assert report.exit_code == 0, _combined_output(report)
    rendered = _combined_output(report)
    assert "MACE E0 override" in rendered
    assert "inline (2 elements)" in rendered


def test_source_mace_options_are_rejected_for_other_models(tmp_path: Path) -> None:
    """MACE-specific options are scoped to MACE source specs."""
    payload = TrainingJobSpec.template(
        workflow="finetune",
        model="aimnet2",
        dataset=("data/train.zarr",),
        output_dir="runs/aimnet2-ft",
        model_id="aimnet2-default",
    ).model_dump(mode="json")
    payload["source"]["mace"] = {"atomic_energies": {"1": -1.0}}

    with pytest.raises(ValidationError, match="source.mace options"):
        TrainingJobSpec.model_validate(payload)


def test_mace_source_options_are_passed_to_wrapper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only the MACE source builder consumes ``source.mace`` options."""
    calls: list[dict[str, Any]] = []

    class FakeMACEWrapper:
        @classmethod
        def from_checkpoint(cls, checkpoint: str, **kwargs: Any) -> object:
            calls.append({"checkpoint": checkpoint, **kwargs})
            return object()

    fake_module = types.ModuleType("nvalchemi.models.mace")
    fake_module.MACEWrapper = FakeMACEWrapper
    monkeypatch.setitem(sys.modules, "nvalchemi.models.mace", fake_module)
    payload = TrainingJobSpec.template(
        workflow="finetune",
        model="mace",
        dataset=("data/train.zarr",),
        output_dir="runs/mace-ft",
        model_id="small-0b",
    ).model_dump(mode="json")
    payload["source"]["mace"] = {"atomic_energies": {"1": -1.0, "8": -2.0}}
    job = TrainingJobSpec.model_validate(payload)

    model = training_cli._build_supported_source_model(job.source, device="cpu")

    assert model is not None
    assert calls == [
        {
            "checkpoint": "small-0b",
            "device": torch.device("cpu"),
            "compile_model": False,
            "atomic_energies": {1: -1.0, 8: -2.0},
            "atomic_energies_path": None,
        }
    ]


def test_train_init_defaults_loss_dtype_policy_to_strict(tmp_path: Path) -> None:
    """Training scaffolds default to strict loss dtype validation."""
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
    payload = json.loads(output.read_text())
    assert payload["strategy"]["loss_fn_spec"]["dtype_policy"] == "strict"


def test_loss_dtype_policy_rejects_unknown_values(tmp_path: Path) -> None:
    """Click rejects unsupported loss dtype policies before writing a spec."""
    output = tmp_path / "finetune.json"
    result = CliRunner().invoke(
        main,
        [
            "finetune",
            "init",
            "mace",
            "small-0b",
            "--dataset",
            "data/train.zarr",
            "--output-dir",
            "runs/ft",
            "--loss-dtype-policy",
            "match_labels",
            "--out",
            str(output),
        ],
    )

    assert result.exit_code != 0
    rendered = _combined_output(result)
    assert "Invalid value for '--loss-dtype-policy'" in rendered
    assert "prediction_to_target" in rendered
    assert not output.exists()


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

    assert spec.source.hooks[0].spec.cls_path.endswith("CheckpointHook")


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


def test_run_generated_scratch_spec_requires_model_specs(tmp_path: Path) -> None:
    """``spec run`` fails clearly when a scratch scaffold lacks model specs."""
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

    run_result = CliRunner().invoke(main, ["spec", "run", str(output), "--no-report"])

    assert run_result.exit_code != 0
    rendered = _combined_output(run_result)
    assert "strategy.model_specs" in rendered
    assert "training-from-scratch" in rendered


def test_run_builds_validation_config_from_validation_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``dataset.validation_path`` builds a validation dataloader for spec run."""
    output = tmp_path / "ft.json"
    result = CliRunner().invoke(
        main,
        [
            "finetune",
            "init",
            "mace",
            "small-0b",
            "--dataset",
            "data/train.zarr",
            "--output-dir",
            "runs/ft",
            "--device",
            "cpu",
            "--out",
            str(output),
        ],
    )
    assert result.exit_code == 0, _combined_output(result)
    payload = json.loads(output.read_text())
    payload["dataset"]["validation_path"] = "data/valid.zarr"
    output.write_text(json.dumps(payload))
    calls: dict[str, Any] = {"dataloaders": []}

    def setup_distributed(enabled: bool) -> None:
        calls["distributed_enabled"] = enabled
        return None

    def build_hooks(
        job: training_cli.TrainingJobSpec, *, enable_ddp: bool, ddp_backend: str | None
    ) -> list[str]:
        return ["hook"]

    def build_strategy(
        job: training_cli.TrainingJobSpec,
        *,
        hooks: list[Any],
        distributed_manager: Any | None,
        map_location: str | None,
    ) -> _FakeStrategy:
        strategy = _FakeStrategy(calls)
        calls["strategy"] = strategy
        return strategy

    def build_dataloader(
        job: training_cli.TrainingJobSpec,
        stack: Any,
        *,
        device: Any,
        batch_size: int | None,
        shuffle: bool,
        drop_last: bool,
        prefetch_factor: int,
        num_streams: int,
        use_streams: bool,
        pin_memory: bool,
        paths: list[str] | None = None,
    ) -> list[str]:
        loader = ["validation-batch"] if paths else ["training-batch"]
        calls["dataloaders"].append((paths, shuffle, drop_last, loader))
        return loader

    monkeypatch.setattr(training_cli, "_setup_distributed_manager", setup_distributed)
    monkeypatch.setattr(training_cli, "_build_runtime_hooks", build_hooks)
    monkeypatch.setattr(training_cli, "_build_strategy", build_strategy)
    monkeypatch.setattr(training_cli, "_build_dataloader", build_dataloader)

    run_result = CliRunner().invoke(main, ["spec", "run", str(output), "--no-report"])

    assert run_result.exit_code == 0, _combined_output(run_result)
    strategy = calls["strategy"]
    assert strategy.validation_config is not None
    assert strategy.validation_config.validation_data == ["validation-batch"]
    assert strategy.validation_config.every_n_epochs == 1
    assert calls["dataloaders"] == [
        (None, True, False, ["training-batch"]),
        (["data/valid.zarr"], False, False, ["validation-batch"]),
    ]
    assert calls["run_dataloader"] == ["training-batch"]


def test_init_writes_validation_path_and_step_cadence(tmp_path: Path) -> None:
    """Scaffold commands persist validation data and cadence intent."""
    output = tmp_path / "ft.json"
    result = CliRunner().invoke(
        main,
        [
            "finetune",
            "init",
            "mace",
            "small-0b",
            "--dataset",
            "data/train.zarr",
            "--validation-dataset",
            "data/valid.zarr",
            "--validation-every-steps",
            "25",
            "--output-dir",
            "runs/ft",
            "--out",
            str(output),
        ],
    )

    assert result.exit_code == 0, _combined_output(result)
    spec = _load_job_spec(output)
    assert spec.dataset.validation_path == "data/valid.zarr"
    assert spec.validation is not None
    assert spec.validation.every_n_steps == 25
    assert spec.validation.every_n_epochs is None


def test_run_validation_cadence_can_be_overridden(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``spec run`` can override validation path and cadence without JSON edits."""
    output = tmp_path / "train.json"
    init_result = CliRunner().invoke(
        main,
        [
            "finetune",
            "init",
            "mace",
            "small-0b",
            "--dataset",
            "data/train.zarr",
            "--output-dir",
            "runs/train",
            "--device",
            "cpu",
            "--out",
            str(output),
        ],
    )
    assert init_result.exit_code == 0, _combined_output(init_result)
    calls: dict[str, Any] = {"dataloaders": []}

    def setup_distributed(enabled: bool) -> None:
        calls["distributed_enabled"] = enabled
        return None

    def build_hooks(
        job: training_cli.TrainingJobSpec, *, enable_ddp: bool, ddp_backend: str | None
    ) -> list[Any]:
        return []

    def build_strategy(
        job: training_cli.TrainingJobSpec,
        *,
        hooks: list[Any],
        distributed_manager: Any | None,
        map_location: str | None,
    ) -> _FakeStrategy:
        strategy = _FakeStrategy(calls)
        calls["strategy"] = strategy
        return strategy

    def build_dataloader(
        job: training_cli.TrainingJobSpec,
        stack: Any,
        *,
        device: Any,
        batch_size: int | None,
        shuffle: bool,
        drop_last: bool,
        prefetch_factor: int,
        num_streams: int,
        use_streams: bool,
        pin_memory: bool,
        paths: list[str] | None = None,
    ) -> list[str]:
        loader = ["validation-batch"] if paths else ["training-batch"]
        calls["dataloaders"].append((paths, loader))
        return loader

    monkeypatch.setattr(training_cli, "_setup_distributed_manager", setup_distributed)
    monkeypatch.setattr(training_cli, "_build_runtime_hooks", build_hooks)
    monkeypatch.setattr(training_cli, "_build_strategy", build_strategy)
    monkeypatch.setattr(training_cli, "_build_dataloader", build_dataloader)

    result = CliRunner().invoke(
        main,
        [
            "spec",
            "run",
            str(output),
            "--no-report",
            "--validation-dataset",
            "data/valid.zarr",
            "--validation-every-steps",
            "5",
        ],
    )

    assert result.exit_code == 0, _combined_output(result)
    strategy = calls["strategy"]
    assert strategy.validation_config.validation_data == ["validation-batch"]
    assert strategy.validation_config.every_n_steps == 5
    assert strategy.validation_config.every_n_epochs is None
    assert calls["dataloaders"] == [
        (None, ["training-batch"]),
        (["data/valid.zarr"], ["validation-batch"]),
    ]


def test_resume_loads_strategy_checkpoint_with_spec_runtime_components(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``spec resume`` loads a restart checkpoint and reattaches runtime pieces."""
    spec_path = tmp_path / "train.json"
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    init_result = CliRunner().invoke(
        main,
        [
            "train",
            "init",
            "--dataset",
            "data/train.zarr",
            "--validation-dataset",
            "data/valid.zarr",
            "--validation-every-epochs",
            "2",
            "--output-dir",
            "runs/train",
            "--device",
            "cpu",
            "--out",
            str(spec_path),
        ],
    )
    assert init_result.exit_code == 0, _combined_output(init_result)
    calls: dict[str, Any] = {"dataloaders": []}

    class Manager:
        """Distributed manager test double."""

        device = "cpu"

    manager = Manager()

    def setup_distributed(enabled: bool) -> Manager | None:
        calls["distributed_enabled"] = enabled
        return manager if enabled else None

    def build_hooks(
        job: training_cli.TrainingJobSpec, *, enable_ddp: bool, ddp_backend: str | None
    ) -> list[str]:
        calls["hook_args"] = (job.name, enable_ddp, ddp_backend)
        return ["hook"]

    def load_checkpoint(
        cls: type[TrainingStrategy],
        root_folder: Path,
        checkpoint_index: int = -1,
        map_location: str | torch.device | None = None,
        **kwargs: Any,
    ) -> _FakeStrategy:
        calls["load_args"] = (cls, root_folder, checkpoint_index, map_location, kwargs)
        strategy = _FakeStrategy(calls)
        calls["strategy"] = strategy
        return strategy

    def build_dataloader(
        job: training_cli.TrainingJobSpec,
        stack: Any,
        *,
        device: Any,
        batch_size: int | None,
        shuffle: bool,
        drop_last: bool,
        prefetch_factor: int,
        num_streams: int,
        use_streams: bool,
        pin_memory: bool,
        paths: list[str] | None = None,
    ) -> list[str]:
        loader = ["validation-batch"] if paths else ["training-batch"]
        calls["dataloaders"].append((device, paths, loader))
        return loader

    monkeypatch.setattr(training_cli, "_setup_distributed_manager", setup_distributed)
    monkeypatch.setattr(training_cli, "_build_runtime_hooks", build_hooks)
    monkeypatch.setattr(
        training_cli.TrainingStrategy,
        "load_checkpoint",
        classmethod(load_checkpoint),
    )
    monkeypatch.setattr(training_cli, "_build_dataloader", build_dataloader)

    result = CliRunner().invoke(
        main,
        [
            "spec",
            "resume",
            str(checkpoint_dir),
            "--spec",
            str(spec_path),
            "--checkpoint-index",
            "3",
            "--distributed",
            "--ddp-backend",
            "gloo",
            "--map-location",
            "cpu",
        ],
    )

    assert result.exit_code == 0, _combined_output(result)
    assert calls["distributed_enabled"] is True
    assert calls["hook_args"] == ("train-from-scratch", True, "gloo")
    load_cls, root, index, map_location, kwargs = calls["load_args"]
    assert load_cls is TrainingStrategy
    assert root == checkpoint_dir
    assert index == 3
    assert map_location == "cpu"
    assert kwargs["hooks"] == ["hook"]
    assert calls["strategy"].distributed_manager is manager
    assert calls["strategy"].validation_config.every_n_epochs == 2
    assert calls["run_dataloader"] == ["training-batch"]


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
    assert "scaffolds, validates, and starts" in rendered
    assert "spec run" in rendered
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
    assert "loss dtype policy" in rendered
    assert "strict" in rendered


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
    assert "Checkpoint hook" in rendered


def test_report_does_not_warn_when_checkpoint_hook_declared(tmp_path: Path) -> None:
    """``spec report`` accepts output.checkpoint_dir when CheckpointHook is declared."""
    output = tmp_path / "finetune.json"
    dataset_path = tmp_path / "domain.zarr"
    dataset_path.mkdir()
    init_result = CliRunner().invoke(
        main,
        [
            "finetune",
            "init",
            "mace",
            "small-0b",
            "--dataset",
            str(dataset_path),
            "--output-dir",
            "runs/mace-ft",
            "--out",
            str(output),
        ],
    )
    assert init_result.exit_code == 0, _combined_output(init_result)
    payload = json.loads(output.read_text())
    payload["source"]["hooks"] = [
        create_model_spec(
            CheckpointHook,
            checkpoint_dir="runs/mace-ft/checkpoints",
            step_interval=10,
        ).model_dump(mode="json")
    ]
    output.write_text(json.dumps(payload))

    result = CliRunner().invoke(main, ["spec", "report", str(output)])

    assert result.exit_code == 0, _combined_output(result)
    rendered = _combined_output(result)
    assert "CheckpointHook" in rendered
    assert "Checkpoint hook" not in rendered


def test_report_renders_hook_firing_order(tmp_path: Path) -> None:
    """``spec report`` shows runtime hooks ordered by declared training stage."""
    output = tmp_path / "finetune.json"
    init_result = CliRunner().invoke(
        main,
        [
            "finetune",
            "init",
            "mace",
            "small-0b",
            "--dataset",
            "s3://bucket/domain.zarr",
            "--output-dir",
            "runs/mace-ft",
            "--device",
            "cpu",
            "--out",
            str(output),
        ],
    )
    assert init_result.exit_code == 0, _combined_output(init_result)
    payload = json.loads(output.read_text())
    hook_spec = create_model_spec(
        NeighborListHook,
        config=NeighborConfig(cutoff=5.0),
    ).model_dump(mode="json")
    payload["source"]["hooks"] = [
        {
            "spec": hook_spec,
            "stages": ["BEFORE_FORWARD", "AFTER_FORWARD"],
        }
    ]
    output.write_text(json.dumps(payload))

    result = CliRunner().invoke(main, ["spec", "report", str(output)])

    assert result.exit_code == 0, _combined_output(result)
    rendered = _combined_output(result)
    assert "Hook Firing Order" in rendered
    assert "BEFORE_FORWARD" in rendered
    assert "AFTER_FORWARD" in rendered
    assert "NeighborListHook" in rendered


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


class _FakeStrategy:
    """Minimal strategy test double for CLI execution tests."""

    def __init__(self, calls: dict[str, Any]) -> None:
        """Store call records for assertions."""
        self.calls = calls

    def run(self, dataloader: Any) -> None:
        """Record that the training run started with the provided dataloader."""
        self.calls["run_dataloader"] = dataloader


def test_run_executes_loaded_spec_with_runtime_components(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``spec run`` builds runtime components and calls strategy.run."""
    output = tmp_path / "train.json"
    init_result = CliRunner().invoke(
        main,
        [
            "train",
            "init",
            "--dataset",
            "data/train.zarr",
            "--output-dir",
            "runs/train",
            "--device",
            "cpu",
            "--out",
            str(output),
        ],
    )
    assert init_result.exit_code == 0, _combined_output(init_result)
    calls: dict[str, Any] = {}

    def setup_distributed(enabled: bool) -> None:
        calls["distributed_enabled"] = enabled
        return None

    def build_hooks(
        job: training_cli.TrainingJobSpec, *, enable_ddp: bool, ddp_backend: str | None
    ) -> list[str]:
        calls["hook_args"] = (job.name, enable_ddp, ddp_backend)
        return ["hook"]

    def build_strategy(
        job: training_cli.TrainingJobSpec,
        *,
        hooks: list[Any],
        distributed_manager: Any | None,
        map_location: str | None,
    ) -> _FakeStrategy:
        calls["strategy_args"] = (job.name, hooks, distributed_manager, map_location)
        return _FakeStrategy(calls)

    def build_dataloader(
        job: training_cli.TrainingJobSpec,
        stack: Any,
        *,
        device: Any,
        batch_size: int | None,
        shuffle: bool,
        drop_last: bool,
        prefetch_factor: int,
        num_streams: int,
        use_streams: bool,
        pin_memory: bool,
        paths: list[str] | None = None,
    ) -> list[str]:
        calls["dataloader_args"] = {
            "job": job.name,
            "device": str(device),
            "batch_size": batch_size,
            "shuffle": shuffle,
            "drop_last": drop_last,
            "prefetch_factor": prefetch_factor,
            "num_streams": num_streams,
            "use_streams": use_streams,
            "pin_memory": pin_memory,
        }
        return ["batch"]

    monkeypatch.setattr(training_cli, "_setup_distributed_manager", setup_distributed)
    monkeypatch.setattr(training_cli, "_build_runtime_hooks", build_hooks)
    monkeypatch.setattr(training_cli, "_build_strategy", build_strategy)
    monkeypatch.setattr(training_cli, "_build_dataloader", build_dataloader)

    result = CliRunner().invoke(
        main,
        [
            "spec",
            "run",
            str(output),
            "--no-report",
            "--batch-size",
            "7",
            "--no-shuffle",
            "--drop-last",
            "--prefetch-factor",
            "3",
            "--num-streams",
            "2",
            "--pin-memory",
            "--no-use-streams",
            "--map-location",
            "cpu",
        ],
    )

    assert result.exit_code == 0, _combined_output(result)
    assert calls["distributed_enabled"] is False
    assert calls["hook_args"] == ("train-from-scratch", False, None)
    assert calls["strategy_args"] == ("train-from-scratch", ["hook"], None, "cpu")
    assert calls["dataloader_args"] == {
        "job": "train-from-scratch",
        "device": "cpu",
        "batch_size": 7,
        "shuffle": False,
        "drop_last": True,
        "prefetch_factor": 3,
        "num_streams": 2,
        "use_streams": False,
        "pin_memory": True,
    }
    assert calls["run_dataloader"] == ["batch"]


def test_run_distributed_options_attach_manager_and_ddp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``spec run --distributed`` passes manager and DDP settings downstream."""
    output = tmp_path / "train.json"
    init_result = CliRunner().invoke(
        main,
        [
            "train",
            "init",
            "--dataset",
            "data/train.zarr",
            "--output-dir",
            "runs/train",
            "--device",
            "cpu",
            "--out",
            str(output),
        ],
    )
    assert init_result.exit_code == 0, _combined_output(init_result)
    calls: dict[str, Any] = {}

    class Manager:
        """Distributed manager test double."""

        device = "cpu"

    manager = Manager()

    def setup_distributed(enabled: bool) -> Manager | None:
        calls["distributed_enabled"] = enabled
        return manager if enabled else None

    def build_hooks(
        job: training_cli.TrainingJobSpec, *, enable_ddp: bool, ddp_backend: str | None
    ) -> list[str]:
        calls["hook_args"] = (enable_ddp, ddp_backend)
        return ["ddp"]

    def build_strategy(
        job: training_cli.TrainingJobSpec,
        *,
        hooks: list[Any],
        distributed_manager: Any | None,
        map_location: str | None,
    ) -> _FakeStrategy:
        calls["strategy_manager"] = distributed_manager
        calls["strategy_hooks"] = hooks
        return _FakeStrategy(calls)

    def build_dataloader(
        job: training_cli.TrainingJobSpec,
        stack: Any,
        *,
        device: Any,
        batch_size: int | None,
        shuffle: bool,
        drop_last: bool,
        prefetch_factor: int,
        num_streams: int,
        use_streams: bool,
        pin_memory: bool,
        paths: list[str] | None = None,
    ) -> list[str]:
        calls["dataloader_device"] = device
        return ["distributed-batch"]

    monkeypatch.setattr(training_cli, "_setup_distributed_manager", setup_distributed)
    monkeypatch.setattr(training_cli, "_build_runtime_hooks", build_hooks)
    monkeypatch.setattr(training_cli, "_build_strategy", build_strategy)
    monkeypatch.setattr(training_cli, "_build_dataloader", build_dataloader)

    result = CliRunner().invoke(
        main,
        [
            "spec",
            "run",
            str(output),
            "--no-report",
            "--distributed",
            "--ddp-backend",
            "gloo",
        ],
    )

    assert result.exit_code == 0, _combined_output(result)
    assert calls["distributed_enabled"] is True
    assert calls["hook_args"] == (True, "gloo")
    assert calls["strategy_manager"] is manager
    assert calls["strategy_hooks"] == ["ddp"]
    assert calls["dataloader_device"] == "cpu"
    assert calls["run_dataloader"] == ["distributed-batch"]


def test_runtime_hooks_accept_training_neighbor_list_hook() -> None:
    """Serialized neighbor-list hooks can target the training forward stage."""
    hook_spec = {
        "spec": create_model_spec(
            NeighborListHook,
            config=NeighborConfig(cutoff=5.0),
        ).model_dump(mode="json"),
        "stages": ["BEFORE_FORWARD"],
    }
    job = TrainingJobSpec.template(
        workflow="finetune",
        model="mace",
        dataset="s3://bucket/domain.zarr",
        output_dir="runs/ft",
        model_id="small-0b",
        device="cpu",
        hooks=(hook_spec,),
    )

    hooks = training_cli._build_runtime_hooks(job, enable_ddp=False, ddp_backend=None)

    assert isinstance(hooks[0], NeighborListHook)
    assert hooks[0].stage is TrainingStage.BEFORE_FORWARD


def test_runtime_hooks_build_one_hook_per_stage_override() -> None:
    """Multiple stage overrides build one runtime hook instance per stage."""
    hook_spec = {
        "spec": create_model_spec(
            NeighborListHook,
            config=NeighborConfig(cutoff=5.0),
        ).model_dump(mode="json"),
        "stages": ["BEFORE_FORWARD", "AFTER_FORWARD"],
    }
    job = TrainingJobSpec.template(
        workflow="finetune",
        model="mace",
        dataset="s3://bucket/domain.zarr",
        output_dir="runs/ft",
        model_id="small-0b",
        device="cpu",
        hooks=(hook_spec,),
    )

    hooks = training_cli._build_runtime_hooks(job, enable_ddp=False, ddp_backend=None)

    assert [hook.stage for hook in hooks] == [
        TrainingStage.BEFORE_FORWARD,
        TrainingStage.AFTER_FORWARD,
    ]
    assert all(isinstance(hook, NeighborListHook) for hook in hooks)


def test_runtime_hooks_accept_raw_spec_with_stage_override() -> None:
    """Raw serialized hook specs treat ``stages`` as CLI metadata."""
    hook_spec = create_model_spec(
        NeighborListHook,
        config=NeighborConfig(cutoff=5.0),
    ).model_dump(mode="json")
    hook_spec["stages"] = ["BEFORE_FORWARD"]
    job = TrainingJobSpec.template(
        workflow="finetune",
        model="mace",
        dataset="s3://bucket/domain.zarr",
        output_dir="runs/ft",
        model_id="small-0b",
        device="cpu",
        hooks=(hook_spec,),
    )

    hooks = training_cli._build_runtime_hooks(job, enable_ddp=False, ddp_backend=None)

    assert isinstance(hooks[0], NeighborListHook)
    assert hooks[0].stage is TrainingStage.BEFORE_FORWARD
    assert "stages" not in job.source.hooks[0].spec.model_extra


def test_runtime_hooks_accept_training_update_hooks() -> None:
    """Serialized training update hooks are valid CLI runtime hooks."""
    hook_spec = create_model_spec(
        MixedPrecisionHook,
        precision="bf16",
    ).model_dump(mode="json")
    job = TrainingJobSpec.template(
        workflow="finetune",
        model="mace",
        dataset="s3://bucket/domain.zarr",
        output_dir="runs/ft",
        model_id="small-0b",
        device="cpu",
        hooks=(hook_spec,),
    )

    hooks = training_cli._build_runtime_hooks(job, enable_ddp=False, ddp_backend=None)

    assert isinstance(hooks[0], MixedPrecisionHook)
    assert hooks[0].precision is torch.bfloat16


def test_runtime_hooks_reject_specs_that_do_not_build_hooks() -> None:
    """Runtime hook specs must instantiate Hook or CheckpointableHook objects."""
    config_spec = create_model_spec(
        NeighborConfig,
        cutoff=5.0,
    ).model_dump(mode="json")

    with pytest.raises(
        ValueError, match="Hook, CheckpointableHook, or TrainingUpdateHook"
    ):
        TrainingJobSpec.template(
            workflow="finetune",
            model="mace",
            dataset="s3://bucket/domain.zarr",
            output_dir="runs/ft",
            model_id="small-0b",
            device="cpu",
            hooks=(config_spec,),
        )


def test_runtime_hooks_add_ddp_before_source_hooks() -> None:
    """Runtime hook construction prepends DDP before serialized source hooks."""
    hook_spec = create_model_spec(
        CheckpointHook,
        checkpoint_dir="runs/ft/checkpoints",
        step_interval=10,
        async_save=False,
    ).model_dump(mode="json")
    job = TrainingJobSpec.template(
        workflow="finetune",
        model="mace",
        dataset="s3://bucket/domain.zarr",
        output_dir="runs/ft",
        model_id="small-0b",
        device="cpu",
        hooks=(hook_spec,),
    )

    hooks = training_cli._build_runtime_hooks(job, enable_ddp=True, ddp_backend="gloo")

    assert isinstance(hooks[0], training_cli.DDPHook)
    assert hooks[0].backend == "gloo"
    assert isinstance(hooks[1], CheckpointHook)


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
