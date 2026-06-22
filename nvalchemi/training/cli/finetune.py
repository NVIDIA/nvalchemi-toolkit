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
"""Rich Click interface for reviewing fine-tuning job specifications."""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, Literal

import click
import plotext as plt
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator
from rich import box
from rich.ansi import AnsiDecoder
from rich.console import Console, ConsoleOptions, Group, RenderResult
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

console = Console(stderr=True)

EndpointName = Literal["native-checkpoint", "mace", "aimnet2", "custom"]


class DatasetIntent(BaseModel):
    """Dataset location and loader intent recorded in a fine-tuning spec."""

    path: str = Field(description="Path or URI to the user-provided training dataset.")
    format: str = Field(
        default="alchemi-zarr",
        description="Dataset format or loader family, for example 'alchemi-zarr'.",
    )
    validation_path: str | None = Field(
        default=None,
        description="Optional path or URI to validation data.",
    )
    batch_size: int | None = Field(
        default=None,
        ge=1,
        description="Requested fine-tuning batch size.",
    )


class SourceIntent(BaseModel):
    """Pretrained model or checkpoint source recorded in a fine-tuning spec."""

    endpoint: EndpointName = Field(description="Fine-tuning source endpoint.")
    checkpoint_path: str | None = Field(
        default=None,
        description="Native nvalchemi checkpoint root or foreign model checkpoint path.",
    )
    model_id: str | None = Field(
        default=None,
        description="Supported pretrained model identifier, such as a MACE or AIMNet2 name.",
    )
    checkpoint_index: int = Field(
        default=-1,
        description="Native checkpoint index; -1 means latest.",
    )
    compile_model: bool | None = Field(
        default=None,
        description="Whether a supported wrapper should compile the model. MACE fine-tuning should normally use false.",
    )
    use_original_loss: bool = Field(
        default=False,
        description="For native checkpoints, reuse source loss metadata when strategy.loss_fn is omitted.",
    )
    use_original_opt_class: bool = Field(
        default=False,
        description="For native checkpoints, reuse source optimizer/scheduler classes when strategy.optimizer_configs is omitted.",
    )
    optimizer_lr: float | None = Field(
        default=1e-5,
        description="Learning rate applied to reused optimizer configs; null preserves serialized LR.",
    )

    @model_validator(mode="after")
    def _validate_source(self) -> SourceIntent:
        """Validate endpoint-specific source requirements."""
        if self.endpoint == "native-checkpoint" and not self.checkpoint_path:
            raise ValueError("native-checkpoint specs require source.checkpoint_path")
        if self.endpoint in {"mace", "aimnet2"} and not (
            self.model_id or self.checkpoint_path
        ):
            raise ValueError(
                f"{self.endpoint} specs require source.model_id or source.checkpoint_path"
            )
        if self.endpoint == "custom" and not self.checkpoint_path:
            raise ValueError("custom specs require source.checkpoint_path")
        return self


class OutputIntent(BaseModel):
    """Output paths recorded in a fine-tuning spec."""

    run_dir: str = Field(description="Run directory for logs and artifacts.")
    checkpoint_dir: str | None = Field(
        default=None,
        description="Directory for restartable fine-tuning checkpoints.",
    )
    report_path: str | None = Field(
        default=None,
        description="Optional path where generated intent reports may be saved.",
    )


class FineTuningJobSpec(BaseModel):
    """CLI-facing fine-tuning job specification envelope."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(default="fine-tune", description="Human-readable job name.")
    source: SourceIntent = Field(description="Pretrained source endpoint and options.")
    dataset: DatasetIntent = Field(
        description="Training and validation dataset intent."
    )
    output: OutputIntent = Field(description="Output path intent.")
    strategy: dict[str, Any] = Field(
        default_factory=dict,
        description="JSON-ready FineTuningStrategy.to_spec_dict() bundle or partial scaffold.",
    )
    notes: str | None = Field(
        default=None,
        description="Optional user notes shown in the Rich report.",
    )

    @model_validator(mode="after")
    def _validate_strategy_window(self) -> FineTuningJobSpec:
        """Validate the high-level strategy execution window."""
        num_epochs = self.strategy.get("num_epochs")
        num_steps = self.strategy.get("num_steps")
        if num_epochs is not None and num_steps is not None:
            raise ValueError("strategy must set only one of num_epochs or num_steps")
        return self


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


def _default_strategy_spec(
    *,
    lr: float,
    num_steps: int | None,
    num_epochs: int | None,
    device: str,
    trainable_patterns: tuple[str, ...],
) -> dict[str, Any]:
    """Return a conservative JSON-ready fine-tuning strategy scaffold."""
    return {
        "optimizer_configs": {
            "main": [
                {
                    "cls_path": "nvalchemi.training.optimizers.OptimizerConfig",
                    "optimizer_cls": "torch.optim.adamw.AdamW",
                    "optimizer_kwargs": {"lr": lr, "weight_decay": 1e-6},
                    "scheduler_cls": None,
                    "scheduler_kwargs": {},
                    "scheduler_metric_adapter": None,
                }
            ]
        },
        "num_epochs": num_epochs,
        "num_steps": num_steps,
        "epoch_step_modifier": 1.0,
        "devices": [device],
        "loss_fn_spec": {
            "cls_path": "nvalchemi.training.losses.composition.ComposedLossFunction",
            "components": [
                {
                    "cls_path": "nvalchemi.training.losses.terms.EnergyMSELoss",
                    "target_key": "energy",
                    "prediction_key": "predicted_energy",
                    "per_atom": False,
                    "ignore_nonfinite": False,
                },
                {
                    "cls_path": "nvalchemi.training.losses.terms.ForceMSELoss",
                    "target_key": "forces",
                    "prediction_key": "predicted_forces",
                    "normalize_by_atom_count": True,
                    "ignore_nonfinite": False,
                },
            ],
            "weights": [1.0, 10.0],
            "normalize_weights": False,
        },
        "model_specs": {},
        "single_model_input": True,
        "training_fn": "nvalchemi.training.strategy.default_training_fn",
        "module_patches": {},
        "freeze_patterns": [],
        "trainable_patterns": list(trainable_patterns),
        "freeze_mode": "requires_grad",
    }


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
) -> dict[str, Any]:
    """Build a CLI job-spec template for one fine-tuning endpoint."""
    return FineTuningJobSpec(
        name=f"{endpoint}-fine-tune",
        source=SourceIntent(
            endpoint=endpoint,
            checkpoint_path=source_path,
            model_id=model_id,
            compile_model=compile_model,
        ),
        dataset=DatasetIntent(path=dataset),
        output=OutputIntent(
            run_dir=output_dir,
            checkpoint_dir=str(Path(output_dir) / "checkpoints"),
        ),
        strategy=_default_strategy_spec(
            lr=lr,
            num_steps=num_steps,
            num_epochs=num_epochs,
            device=device,
            trainable_patterns=trainable_patterns,
        ),
    ).model_dump(mode="json", exclude_none=True)


def _load_job_spec(path: Path) -> FineTuningJobSpec:
    """Load and validate a fine-tuning job specification from JSON."""
    try:
        raw = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise click.ClickException(f"Could not parse {path}: {exc}") from exc
    try:
        return FineTuningJobSpec.model_validate(raw)
    except ValidationError as exc:
        raise click.ClickException(str(exc)) from exc


def _write_or_print(payload: Mapping[str, Any], output: Path | None) -> None:
    """Write a JSON payload to a file or stdout."""
    text = json.dumps(payload, indent=2) + "\n"
    if output is None:
        click.echo(text, nl=False)
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text)
    console.print(f"[green]Wrote[/] {output}")


def _strategy_section(strategy: Mapping[str, Any]) -> Table:
    """Build a Rich table summarizing strategy intent."""
    table = Table(title="Strategy", box=box.SIMPLE_HEAD, expand=True)
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


def _intent_section(job: FineTuningJobSpec) -> Table:
    """Build a Rich table summarizing source, data, and output intent."""
    table = Table(title="Fine-tuning Intent", box=box.SIMPLE_HEAD, expand=True)
    table.add_column("Area", style="cyan", no_wrap=True)
    table.add_column("Value", overflow="fold")
    table.add_row("job", job.name)
    table.add_row("endpoint", job.source.endpoint)
    table.add_row("source checkpoint", _format_optional(job.source.checkpoint_path))
    table.add_row("model id", _format_optional(job.source.model_id))
    table.add_row("checkpoint index", str(job.source.checkpoint_index))
    table.add_row("compile_model", _format_optional(job.source.compile_model))
    table.add_row("reuse source loss", str(job.source.use_original_loss))
    table.add_row("reuse source optimizer", str(job.source.use_original_opt_class))
    table.add_row("dataset", f"{job.dataset.path} ({job.dataset.format})")
    table.add_row("validation data", _format_optional(job.dataset.validation_path))
    table.add_row("batch size", _format_optional(job.dataset.batch_size))
    table.add_row("run dir", job.output.run_dir)
    table.add_row("checkpoint dir", _format_optional(job.output.checkpoint_dir))
    return table


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


def _optimizer_rows(strategy: Mapping[str, Any]) -> list[tuple[str, str, float, str]]:
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


def _optimizer_section(strategy: Mapping[str, Any]) -> Table:
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


def _lr_series(
    strategy: Mapping[str, Any], *, samples: int = 80
) -> list[tuple[int, float]]:
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


def _render_report(job: FineTuningJobSpec) -> None:
    """Render a Rich report card for a fine-tuning job spec."""
    console.rule(f"[bold]Fine-tuning report: {job.name}")
    console.print(_intent_section(job))
    console.print(_strategy_section(job.strategy))
    console.print(_optimizer_section(job.strategy))
    if job.notes:
        console.print(Panel(Text(job.notes, overflow="fold"), title="Notes"))
    console.print(
        Panel(_LRSchedulePlot(_lr_series(job.strategy)), title="Learning-rate preview")
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
    _write_or_print(FineTuningJobSpec.model_json_schema(), output)


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
        console.print(
            Syntax(
                json.dumps(job.model_dump(mode="json", exclude_none=True), indent=2),
                "json",
            )
        )


if __name__ == "__main__":
    main()
