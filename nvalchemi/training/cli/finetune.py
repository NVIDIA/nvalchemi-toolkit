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
"""Rich Click interface for reviewing fine-tuning strategy specifications."""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping, MutableMapping
from pathlib import Path
from typing import Any, Literal, TypeAlias

import click
import plotext as plt
import torch
from rich import box
from rich.ansi import AnsiDecoder
from rich.console import Console, ConsoleOptions, Group, RenderResult
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

from nvalchemi.training import FineTuningStrategy
from nvalchemi.training import _spec_utils as strategy_spec
from nvalchemi.training._spec import create_model_spec
from nvalchemi.training.losses.composition import (
    ComposedLossFunction,
    loss_component_to_spec,
)
from nvalchemi.training.losses.terms import EnergyMSELoss, ForceMSELoss
from nvalchemi.training.optimizers import OptimizerConfig

console = Console(stderr=True)

EndpointName: TypeAlias = Literal["native-checkpoint", "mace", "aimnet2", "custom"]
JobSpec: TypeAlias = dict[str, Any]
StrategySpec: TypeAlias = Mapping[str, Any]

_ENDPOINTS: tuple[EndpointName, ...] = (
    "native-checkpoint",
    "mace",
    "aimnet2",
    "custom",
)


class _LRSchedulePlot:
    """Rich renderable for a learning-rate schedule preview."""

    def __init__(
        self, series: Iterable[tuple[int, float]], *, height: int = 10
    ) -> None:
        self.series = tuple(series)
        self.height = height
        self.decoder = AnsiDecoder()

    def __rich_console__(
        self, console_: Console, options: ConsoleOptions
    ) -> RenderResult:
        """Render the learning-rate schedule as an ANSI plot."""
        width = max(28, options.max_width or console_.width)
        plt.clf()
        plt.plotsize(width, self.height)
        plt.theme("dark")
        plt.title("learning rate")
        plt.xlabel("step")
        if not self.series:
            yield Text("No optimizer learning-rate metadata found.")
            return
        steps = [step for step, _ in self.series]
        values = [value for _, value in self.series]
        if len(values) == 1:
            plt.scatter(steps, values)
        else:
            plt.plot(steps, values)
        yield Group(*self.decoder.decode(plt.build()))


def _job_schema() -> dict[str, Any]:
    """Return the CLI envelope schema around a ``FineTuningStrategy`` spec."""
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "FineTuningStrategySpecEnvelope",
        "type": "object",
        "additionalProperties": False,
        "required": ["source", "dataset", "output", "strategy"],
        "properties": {
            "name": {"type": "string", "default": "fine-tune"},
            "source": {
                "type": "object",
                "required": ["endpoint"],
                "additionalProperties": True,
                "properties": {
                    "endpoint": {"enum": list(_ENDPOINTS)},
                    "checkpoint_path": {"type": ["string", "null"]},
                    "model_id": {"type": ["string", "null"]},
                    "checkpoint_index": {"type": "integer", "default": -1},
                    "compile_model": {"type": ["boolean", "null"]},
                    "use_original_loss": {"type": "boolean", "default": False},
                    "use_original_opt_class": {"type": "boolean", "default": False},
                    "optimizer_lr": {"type": ["number", "null"], "default": 1e-5},
                },
            },
            "dataset": {
                "type": "object",
                "required": ["path"],
                "additionalProperties": True,
                "properties": {
                    "path": {"type": "string"},
                    "format": {"type": "string", "default": "alchemi-zarr"},
                    "validation_path": {"type": ["string", "null"]},
                    "batch_size": {"type": ["integer", "null"], "minimum": 1},
                },
            },
            "output": {
                "type": "object",
                "required": ["run_dir"],
                "additionalProperties": True,
                "properties": {
                    "run_dir": {"type": "string"},
                    "checkpoint_dir": {"type": ["string", "null"]},
                    "report_path": {"type": ["string", "null"]},
                },
            },
            "strategy": {
                "type": "object",
                "description": (
                    "A JSON-ready bundle produced by FineTuningStrategy.to_spec_dict()."
                ),
            },
            "notes": {"type": ["string", "null"]},
        },
    }


def _default_strategy_spec(
    *,
    lr: float,
    num_steps: int | None,
    num_epochs: int | None,
    device: str,
    trainable_patterns: tuple[str, ...],
) -> dict[str, Any]:
    """Return a conservative ``FineTuningStrategy.to_spec_dict`` scaffold."""
    return {
        "optimizer_configs": {"main": [_default_optimizer_config_spec(lr)]},
        "num_epochs": num_epochs,
        "num_steps": num_steps,
        "epoch_step_modifier": 1.0,
        "devices": [device],
        "loss_fn_spec": _default_loss_fn_spec(),
        "model_specs": {},
        "single_model_input": True,
        "training_fn": "nvalchemi.training.strategy.default_training_fn",
        "module_patches": {},
        "freeze_patterns": [],
        "trainable_patterns": list(trainable_patterns),
        "freeze_mode": "requires_grad",
    }


def _default_optimizer_config_spec(lr: float) -> dict[str, Any]:
    """Build the default optimizer spec through ``OptimizerConfig``."""
    config = OptimizerConfig(
        optimizer_cls=torch.optim.AdamW,
        optimizer_kwargs={"lr": lr, "weight_decay": 1e-6},
    )
    return config.to_spec().model_dump()


def _default_loss_fn_spec() -> dict[str, Any]:
    """Build the default loss spec with the training loss serializers."""
    loss_fn = ComposedLossFunction(
        [EnergyMSELoss(), ForceMSELoss(normalize_by_atom_count=True)],
        weights=[1.0, 10.0],
        normalize_weights=False,
    )
    return create_model_spec(
        type(loss_fn),
        components=[loss_component_to_spec(comp) for comp in loss_fn.components],
        weights=list(loss_fn._weights),
        normalize_weights=loss_fn.normalize_weights,
    ).model_dump()


def _job_template(
    *,
    endpoint: EndpointName,
    dataset: str,
    output_dir: str,
    source_path: str | None,
    model_id: str | None,
    lr: float,
    num_steps: int | None,
    num_epochs: int | None,
    device: str,
    trainable_patterns: tuple[str, ...],
    compile_model: bool | None,
) -> JobSpec:
    """Build a CLI envelope around a fine-tuning strategy spec."""
    source: dict[str, Any] = {"endpoint": endpoint}
    if source_path is not None:
        source["checkpoint_path"] = source_path
    if model_id is not None:
        source["model_id"] = model_id
    if compile_model is not None:
        source["compile_model"] = compile_model
    if endpoint == "native-checkpoint":
        source.update(
            {
                "checkpoint_index": -1,
                "use_original_loss": False,
                "use_original_opt_class": False,
                "optimizer_lr": 1e-5,
            }
        )
    return {
        "name": f"{endpoint}-fine-tune",
        "source": source,
        "dataset": {"path": dataset, "format": "alchemi-zarr"},
        "output": {
            "run_dir": output_dir,
            "checkpoint_dir": str(Path(output_dir) / "checkpoints"),
        },
        "strategy": _default_strategy_spec(
            lr=lr,
            num_steps=num_steps,
            num_epochs=num_epochs,
            device=device,
            trainable_patterns=trainable_patterns,
        ),
    }


def _load_job_spec(path: Path) -> JobSpec:
    """Load and validate a fine-tuning job specification from JSON."""
    try:
        raw = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise click.ClickException(f"Could not parse {path}: {exc}") from exc
    try:
        return _validate_job_spec(raw)
    except (TypeError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc


def _validate_job_spec(raw: Any) -> JobSpec:
    """Validate the CLI envelope and reusable fine-tuning strategy spec fields."""
    if not isinstance(raw, MutableMapping):
        raise TypeError(f"spec root must be a JSON object; got {type(raw).__name__}.")
    job = dict(raw)
    source = _require_mapping(job, "source")
    dataset = _require_mapping(job, "dataset")
    output = _require_mapping(job, "output")
    strategy = _require_mapping(job, "strategy")
    _validate_source(source)
    _require_string(dataset, "path", "dataset.path")
    _require_string(output, "run_dir", "output.run_dir")
    _validate_strategy_spec(strategy)
    job["source"] = dict(source)
    job["dataset"] = dict(dataset)
    job["output"] = dict(output)
    job["strategy"] = dict(strategy)
    return job


def _require_mapping(raw: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    """Return a required mapping field from the job spec."""
    value = raw.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"{key} must be a JSON object.")
    return value


def _require_string(raw: Mapping[str, Any], key: str, label: str) -> str:
    """Return a required string field from a JSON mapping."""
    value = raw.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty string.")
    return value


def _validate_source(source: Mapping[str, Any]) -> None:
    """Validate endpoint-specific source metadata."""
    endpoint = source.get("endpoint")
    if endpoint not in _ENDPOINTS:
        raise ValueError(
            f"source.endpoint must be one of {_ENDPOINTS}; got {endpoint!r}."
        )
    if endpoint == "native-checkpoint" and not source.get("checkpoint_path"):
        raise ValueError("native-checkpoint specs require source.checkpoint_path.")
    if endpoint in {"mace", "aimnet2"} and not (
        source.get("model_id") or source.get("checkpoint_path")
    ):
        raise ValueError(
            f"{endpoint} specs require source.model_id or source.checkpoint_path."
        )
    if endpoint == "custom" and not source.get("checkpoint_path"):
        raise ValueError("custom specs require source.checkpoint_path.")


def _validate_strategy_spec(strategy: Mapping[str, Any]) -> None:
    """Validate strategy spec fields with existing training serializers."""
    missing = [
        key
        for key in ("optimizer_configs", "devices", "loss_fn_spec")
        if key not in strategy
    ]
    if missing:
        raise ValueError(
            f"strategy is missing required FineTuningStrategy spec key(s) {missing}."
        )
    num_epochs = strategy.get("num_epochs")
    num_steps = strategy.get("num_steps")
    if num_epochs is not None and num_steps is not None:
        raise ValueError("strategy must set only one of num_epochs or num_steps.")
    if num_epochs is None and num_steps is None:
        raise ValueError("strategy must set one of num_epochs or num_steps.")
    strategy_spec._optimizer_configs_from_spec(strategy["optimizer_configs"])
    strategy_spec._devices_from_spec(strategy["devices"])
    strategy_spec._loss_fn_from_spec(strategy["loss_fn_spec"])
    strategy_spec._training_fn_from_spec(strategy, None)
    if strategy.get("model_specs"):
        FineTuningStrategy.from_spec_dict(dict(strategy), hooks=[])


def _write_or_print(payload: Mapping[str, Any], output: Path | None) -> None:
    """Write a JSON payload to a file or stdout."""
    text = json.dumps(payload, indent=2) + "\n"
    if output is None:
        click.echo(text, nl=False)
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text)
    console.print(f"[green]Wrote[/] {output}")


def _strategy_section(strategy: StrategySpec) -> Table:
    """Build a Rich table summarizing strategy intent."""
    table = Table(title="FineTuningStrategy", box=box.SIMPLE_HEAD, expand=True)
    table.add_column("Field", style="cyan", no_wrap=True)
    table.add_column("Value", overflow="fold")
    table.add_row("num_epochs", _format_optional(strategy.get("num_epochs")))
    table.add_row("num_steps", _format_optional(strategy.get("num_steps")))
    table.add_row(
        "devices", ", ".join(map(str, strategy.get("devices", []))) or "not specified"
    )
    table.add_row("training_fn", str(strategy.get("training_fn", "not specified")))
    table.add_row("freeze_mode", str(strategy.get("freeze_mode", "requires_grad")))
    table.add_row(
        "trainable_patterns", _format_sequence(strategy.get("trainable_patterns"))
    )
    table.add_row("freeze_patterns", _format_sequence(strategy.get("freeze_patterns")))
    table.add_row(
        "module_patches", _format_mapping_keys(strategy.get("module_patches"))
    )
    return table


def _intent_section(job: JobSpec) -> Table:
    """Build a Rich table summarizing source, data, and output intent."""
    source = job["source"]
    dataset = job["dataset"]
    output = job["output"]
    table = Table(title="Fine-tuning Intent", box=box.SIMPLE_HEAD, expand=True)
    table.add_column("Area", style="cyan", no_wrap=True)
    table.add_column("Value", overflow="fold")
    table.add_row("job", str(job.get("name", "fine-tune")))
    table.add_row("endpoint", str(source.get("endpoint")))
    table.add_row("source checkpoint", _format_optional(source.get("checkpoint_path")))
    table.add_row("model id", _format_optional(source.get("model_id")))
    table.add_row("checkpoint index", str(source.get("checkpoint_index", -1)))
    table.add_row("compile_model", _format_optional(source.get("compile_model")))
    table.add_row("reuse source loss", str(source.get("use_original_loss", False)))
    table.add_row(
        "reuse source optimizer", str(source.get("use_original_opt_class", False))
    )
    table.add_row("dataset", f"{dataset['path']} ({dataset.get('format', 'unknown')})")
    table.add_row("validation data", _format_optional(dataset.get("validation_path")))
    table.add_row("batch size", _format_optional(dataset.get("batch_size")))
    table.add_row("run dir", str(output["run_dir"]))
    table.add_row("checkpoint dir", _format_optional(output.get("checkpoint_dir")))
    return table


def _warning_section(job: JobSpec) -> Table:
    """Build a Rich table of fine-tuning heuristic warnings."""
    table = Table(title="Warnings", box=box.SIMPLE_HEAD, expand=True)
    table.add_column("Level", no_wrap=True)
    table.add_column("Check", style="cyan", no_wrap=True)
    table.add_column("Details", overflow="fold")
    warnings = _fine_tuning_warnings(job)
    if not warnings:
        table.add_row(
            "ok",
            "No common issues detected",
            "Review dataset units and model outputs before running.",
        )
        return table
    for level, check, details in warnings:
        style = "yellow" if level == "warning" else "red"
        table.add_row(f"[{style}]{level}[/]", check, details)
    return table


def _fine_tuning_warnings(job: JobSpec) -> list[tuple[str, str, str]]:
    """Return heuristic warnings for common fine-tuning mistakes."""
    source = job["source"]
    dataset = job["dataset"]
    output = job["output"]
    strategy = job["strategy"]
    warnings: list[tuple[str, str, str]] = []
    endpoint = source.get("endpoint")
    if endpoint == "mace" and source.get("compile_model") is True:
        warnings.append(
            (
                "warning",
                "MACE compile",
                "compile_model=true is inference-oriented for MACE; use false for fine-tuning.",
            )
        )
    if not strategy.get("trainable_patterns") and not strategy.get("freeze_patterns"):
        warnings.append(
            (
                "warning",
                "Full model update",
                "No trainable or freeze patterns are set, so every optimizer-configured parameter can update.",
            )
        )
    if not strategy.get("module_patches") and not strategy.get("trainable_patterns"):
        warnings.append(
            (
                "warning",
                "No adaptation boundary",
                "No module patches or trainable allow-list are declared; confirm full-model fine-tuning is intended.",
            )
        )
    max_lr = max((row[2] for row in _optimizer_rows(strategy)), default=None)
    if max_lr is not None and max_lr > 1e-4:
        warnings.append(
            (
                "warning",
                "Learning rate",
                f"The largest optimizer LR is {max_lr:.3g}; pretrained fine-tuning usually starts at 1e-5 to 1e-4.",
            )
        )
    if not dataset.get("validation_path"):
        warnings.append(
            (
                "warning",
                "Validation data",
                "No dataset.validation_path is recorded; make sure validation_config is supplied in the execution script.",
            )
        )
    if not output.get("checkpoint_dir"):
        warnings.append(
            (
                "warning",
                "Restart checkpoints",
                "No output.checkpoint_dir is recorded for restartable fine-tuning checkpoints.",
            )
        )
    if output.get("checkpoint_dir") and output.get("checkpoint_dir") == source.get(
        "checkpoint_path"
    ):
        warnings.append(
            (
                "danger",
                "Checkpoint overwrite",
                "Output checkpoint_dir matches the source checkpoint path.",
            )
        )
    if (
        endpoint == "native-checkpoint"
        and source.get("use_original_opt_class")
        and max_lr is None
    ):
        warnings.append(
            (
                "warning",
                "Optimizer reuse",
                "Optimizer class reuse is requested, but the strategy spec does not expose an LR preview.",
            )
        )
    if endpoint == "custom" and not strategy.get("model_specs"):
        warnings.append(
            (
                "warning",
                "Custom model reload",
                "No model_specs are present; the execution script must instantiate the model and load weights.",
            )
        )
    return warnings


def _format_optional(value: Any) -> str:
    """Format an optional value for Rich tables."""
    return "not specified" if value is None else str(value)


def _format_sequence(value: Any) -> str:
    """Format a JSON sequence for Rich tables."""
    if not value:
        return "none"
    if isinstance(value, list | tuple):
        return "\n".join(map(str, value))
    return str(value)


def _format_mapping_keys(value: Any) -> str:
    """Format mapping keys for Rich tables."""
    if not isinstance(value, Mapping) or not value:
        return "none"
    return "\n".join(map(str, value))


def _optimizer_rows(strategy: StrategySpec) -> list[tuple[str, str, float, str]]:
    """Extract optimizer rows as model key, class, LR, and scheduler."""
    rows: list[tuple[str, str, float, str]] = []
    raw_configs = strategy.get("optimizer_configs")
    if not isinstance(raw_configs, Mapping):
        return rows
    for model_key, configs in raw_configs.items():
        if not isinstance(configs, list):
            continue
        for config in configs:
            if not isinstance(config, Mapping):
                continue
            kwargs = config.get("optimizer_kwargs")
            lr = kwargs.get("lr") if isinstance(kwargs, Mapping) else None
            if isinstance(lr, int | float):
                rows.append(
                    (
                        str(model_key),
                        str(config.get("optimizer_cls", "not specified")),
                        float(lr),
                        str(config.get("scheduler_cls") or "none"),
                    )
                )
    return rows


def _optimizer_section(strategy: StrategySpec) -> Table:
    """Build a Rich table summarizing optimizer configuration."""
    table = Table(title="Optimizers", box=box.SIMPLE_HEAD, expand=True)
    table.add_column("Model", style="cyan", no_wrap=True)
    table.add_column("Optimizer", overflow="fold")
    table.add_column("LR", justify="right", no_wrap=True)
    table.add_column("Scheduler", overflow="fold")
    rows = _optimizer_rows(strategy)
    if not rows:
        table.add_row("main", "not specified", "", "")
        return table
    for model_key, optimizer_cls, lr, scheduler_cls in rows:
        table.add_row(model_key, optimizer_cls, f"{lr:.3g}", scheduler_cls)
    return table


def _lr_series(strategy: StrategySpec, *, samples: int = 80) -> list[tuple[int, float]]:
    """Build a representative learning-rate series from optimizer metadata."""
    raw_configs = strategy.get("optimizer_configs")
    if not isinstance(raw_configs, Mapping):
        return []
    first_config: Mapping[str, Any] | None = None
    for configs in raw_configs.values():
        if isinstance(configs, list) and configs and isinstance(configs[0], Mapping):
            first_config = configs[0]
            break
    if first_config is None:
        return []
    kwargs = first_config.get("optimizer_kwargs")
    lr = kwargs.get("lr") if isinstance(kwargs, Mapping) else None
    if not isinstance(lr, int | float):
        return []
    total_steps = int(strategy.get("num_steps") or 100)
    total_steps = max(total_steps, 1)
    stride = max(1, math.ceil(total_steps / max(samples - 1, 1)))
    steps = sorted({0, *range(stride, total_steps + 1, stride), total_steps})
    return [(step, _lr_at_step(float(lr), first_config, step)) for step in steps]


def _lr_at_step(base_lr: float, config: Mapping[str, Any], step: int) -> float:
    """Approximate scheduler learning rate for supported scheduler specs."""
    scheduler_cls = config.get("scheduler_cls")
    kwargs = config.get("scheduler_kwargs")
    scheduler_kwargs = kwargs if isinstance(kwargs, Mapping) else {}
    if not scheduler_cls:
        return base_lr
    scheduler_name = str(scheduler_cls)
    if scheduler_name.endswith("StepLR"):
        step_size = int(scheduler_kwargs.get("step_size", 1))
        gamma = float(scheduler_kwargs.get("gamma", 0.1))
        return base_lr * gamma ** (step // max(step_size, 1))
    if scheduler_name.endswith("ExponentialLR"):
        gamma = float(scheduler_kwargs.get("gamma", 1.0))
        return base_lr * gamma**step
    if scheduler_name.endswith("CosineAnnealingLR"):
        t_max = max(int(scheduler_kwargs.get("T_max", 1)), 1)
        eta_min = float(scheduler_kwargs.get("eta_min", 0.0))
        phase = min(step, t_max) / t_max
        return eta_min + 0.5 * (base_lr - eta_min) * (1.0 + math.cos(math.pi * phase))
    return base_lr


def _render_report(job: JobSpec) -> None:
    """Render a Rich report card for a fine-tuning job spec."""
    strategy = job["strategy"]
    console.rule(f"[bold]Fine-tuning report: {job.get('name', 'fine-tune')}")
    console.print(_intent_section(job))
    console.print(_warning_section(job))
    console.print(_strategy_section(strategy))
    console.print(_optimizer_section(strategy))
    if job.get("notes"):
        console.print(Panel(Text(str(job["notes"]), overflow="fold"), title="Notes"))
    console.print(
        Panel(_LRSchedulePlot(_lr_series(strategy)), title="Learning-rate preview")
    )


def _print_template_message(output: Path | None, endpoint: EndpointName) -> None:
    """Print a concise template-generation status message."""
    if output is not None:
        console.print(f"[green]Created {endpoint} fine-tuning spec[/] {output}")


def _common_template_options(function: Any) -> Any:
    """Attach common template options to an endpoint scaffold command."""
    options = [
        click.option("--dataset", required=True, help="Training dataset path or URI."),
        click.option("--output-dir", required=True, help="Run output directory."),
        click.option(
            "--out",
            "output",
            type=click.Path(path_type=Path),
            help="Write JSON spec to this file.",
        ),
        click.option(
            "--lr",
            type=float,
            default=1e-5,
            show_default=True,
            help="Initial fine-tuning learning rate.",
        ),
        click.option(
            "--num-steps",
            type=int,
            default=1000,
            show_default=True,
            help="Number of fine-tuning steps.",
        ),
        click.option(
            "--num-epochs", type=int, default=None, help="Use epochs instead of steps."
        ),
        click.option(
            "--device",
            default="cuda",
            show_default=True,
            help="Strategy device string.",
        ),
        click.option(
            "--trainable-pattern",
            "trainable_patterns",
            multiple=True,
            help="Glob pattern for trainable parameters.",
        ),
    ]
    for option in reversed(options):
        function = option(function)
    return function


@click.group()
def main() -> None:
    """Review and scaffold nvalchemi fine-tuning specifications."""


@main.group()
def init() -> None:
    """Create endpoint-specific fine-tuning specification scaffolds."""


@main.group()
def schema() -> None:
    """Dump JSON schema and templates for offline specification authoring."""


@main.group(name="spec")
def spec_group() -> None:
    """Validate and report on saved fine-tuning specifications."""


@schema.command("dump")
@click.option(
    "--out",
    "output",
    type=click.Path(path_type=Path),
    help="Write schema JSON to this file.",
)
def dump_schema(output: Path | None) -> None:
    """Dump the CLI fine-tuning job JSON schema."""
    _write_or_print(_job_schema(), output)


@schema.command("template")
@click.option(
    "--out",
    "output",
    type=click.Path(path_type=Path),
    help="Write template JSON to this file.",
)
def dump_template(output: Path | None) -> None:
    """Dump a native-checkpoint fine-tuning template."""
    payload = _job_template(
        endpoint="native-checkpoint",
        dataset="data/train.zarr",
        output_dir="runs/finetune",
        source_path="runs/pretrain/checkpoints",
        model_id=None,
        lr=1e-5,
        num_steps=1000,
        num_epochs=None,
        device="cuda",
        trainable_patterns=("main.model.readout.*",),
        compile_model=None,
    )
    _write_or_print(payload, output)


@init.command("checkpoint")
@_common_template_options
@click.argument("checkpoint_dir")
def init_checkpoint(
    checkpoint_dir: str,
    dataset: str,
    output_dir: str,
    output: Path | None,
    lr: float,
    num_steps: int | None,
    num_epochs: int | None,
    device: str,
    trainable_patterns: tuple[str, ...],
) -> None:
    """Create a spec for a native nvalchemi checkpoint source."""
    if num_epochs is not None:
        num_steps = None
    payload = _job_template(
        endpoint="native-checkpoint",
        dataset=dataset,
        output_dir=output_dir,
        source_path=checkpoint_dir,
        model_id=None,
        lr=lr,
        num_steps=num_steps,
        num_epochs=num_epochs,
        device=device,
        trainable_patterns=trainable_patterns,
        compile_model=None,
    )
    _write_or_print(payload, output)
    _print_template_message(output, "native-checkpoint")


@init.command("mace")
@_common_template_options
@click.argument("model_or_checkpoint")
def init_mace(
    model_or_checkpoint: str,
    dataset: str,
    output_dir: str,
    output: Path | None,
    lr: float,
    num_steps: int | None,
    num_epochs: int | None,
    device: str,
    trainable_patterns: tuple[str, ...],
) -> None:
    """Create a spec for a MACE wrapper fine-tuning source."""
    if num_epochs is not None:
        num_steps = None
    payload = _job_template(
        endpoint="mace",
        dataset=dataset,
        output_dir=output_dir,
        source_path=None
        if Path(model_or_checkpoint).suffix == ""
        else model_or_checkpoint,
        model_id=model_or_checkpoint
        if Path(model_or_checkpoint).suffix == ""
        else None,
        lr=lr,
        num_steps=num_steps,
        num_epochs=num_epochs,
        device=device,
        trainable_patterns=trainable_patterns or ("main.model.readouts.*",),
        compile_model=False,
    )
    _write_or_print(payload, output)
    _print_template_message(output, "mace")


@init.command("aimnet2")
@_common_template_options
@click.argument("model_or_checkpoint")
def init_aimnet2(
    model_or_checkpoint: str,
    dataset: str,
    output_dir: str,
    output: Path | None,
    lr: float,
    num_steps: int | None,
    num_epochs: int | None,
    device: str,
    trainable_patterns: tuple[str, ...],
) -> None:
    """Create a spec for an AIMNet2 wrapper fine-tuning source."""
    if num_epochs is not None:
        num_steps = None
    payload = _job_template(
        endpoint="aimnet2",
        dataset=dataset,
        output_dir=output_dir,
        source_path=None
        if Path(model_or_checkpoint).suffix == ""
        else model_or_checkpoint,
        model_id=model_or_checkpoint
        if Path(model_or_checkpoint).suffix == ""
        else None,
        lr=lr,
        num_steps=num_steps,
        num_epochs=num_epochs,
        device=device,
        trainable_patterns=trainable_patterns,
        compile_model=None,
    )
    _write_or_print(payload, output)
    _print_template_message(output, "aimnet2")


@init.command("custom")
@_common_template_options
@click.argument("checkpoint_path")
def init_custom(
    checkpoint_path: str,
    dataset: str,
    output_dir: str,
    output: Path | None,
    lr: float,
    num_steps: int | None,
    num_epochs: int | None,
    device: str,
    trainable_patterns: tuple[str, ...],
) -> None:
    """Create a spec for a custom user-managed model checkpoint."""
    if num_epochs is not None:
        num_steps = None
    payload = _job_template(
        endpoint="custom",
        dataset=dataset,
        output_dir=output_dir,
        source_path=checkpoint_path,
        model_id=None,
        lr=lr,
        num_steps=num_steps,
        num_epochs=num_epochs,
        device=device,
        trainable_patterns=trainable_patterns,
        compile_model=None,
    )
    _write_or_print(payload, output)
    _print_template_message(output, "custom")


@spec_group.command("validate")
@click.argument("path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
def validate_spec(path: Path) -> None:
    """Validate a fine-tuning job specification without rendering the report."""
    _load_job_spec(path)
    console.print(f"[green]Valid fine-tuning specification:[/] {path}")


@spec_group.command("report")
@click.argument("path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "--json",
    "show_json",
    is_flag=True,
    help="Print normalized JSON after the Rich report.",
)
def report_spec(path: Path, show_json: bool) -> None:
    """Validate and render a Rich report card for a fine-tuning spec."""
    job = _load_job_spec(path)
    _render_report(job)
    if show_json:
        console.print(Syntax(json.dumps(job, indent=2), "json"))


if __name__ == "__main__":
    main()
