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
"""Rich Click interface for reviewing training strategy specifications.

Supported source models share the flat :class:`SourceSpec` envelope for common
fields such as checkpoint path, model id, compile behavior, and runtime hooks.
Architecture-specific source knobs should live under a namespaced block in
``source`` and be parsed by a small options model owned by that architecture,
such as ``source.mace`` via :class:`MaceSourceOptions`. This keeps unrelated
model interfaces from accumulating MACE-, AIMNet2-, or custom-only fields while
still allowing the CLI JSON to expose source-specific behavior.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping
from contextlib import ExitStack
from pathlib import Path
from typing import Annotated, Any, Literal, Self, TypeAlias, get_args

import click
import plotext as plt
import torch
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator
from rich import box
from rich.ansi import AnsiDecoder
from rich.console import Console, ConsoleOptions, Group, RenderResult
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

from nvalchemi.training import (
    FineTuningStrategy,
    TrainingStage,
    TrainingStrategy,
)
from nvalchemi.training import _spec_utils as strategy_spec
from nvalchemi.training._spec import create_model_spec_from_json
from nvalchemi.training.cli_common import (
    DatasetSpec,
    MaceSourceOptions,
    ModelSource,
    OutputSpec,
    RuntimeHookSpec,
    SourceSpec,
    ValidationSpec,
    _training_stage_name,
    build_dataloader,
    build_runtime_hooks,
    build_supported_source_model,
    build_validation_config,
    common_loader_options,
    common_validation_options,
    console,
    dataset_device,
    path_exists,
    primary_strategy_device,
    resolve_distributed_enabled,
    setup_distributed_manager,
    write_or_print,
)
from nvalchemi.training.losses.composition import ComposedLossFunction, DTypePolicy
from nvalchemi.training.losses.terms import EnergyMSELoss, ForceMSELoss
from nvalchemi.training.optimizers import OptimizerConfig

_MAIN_EPILOG = (
    "This CLI scaffolds, validates, and starts training specifications. "
    "Use `spec report` to review intent, then `spec run` to build the "
    "dataset/model/strategy components and call `run(...)`.\n\n"
    "Examples:\n\n"
    "Fine-tune MACE on an ALCHEMI dataset:\n\n"
    "  nvalchemi-training finetune init mace small-0b --dataset data/domain.zarr --output-dir runs/mace-ft --out mace-ft.json\n\n"
    "Fine-tune MACE with a MultiDataset intent:\n\n"
    "  nvalchemi-training finetune init mace small-0b --dataset data/a.zarr --dataset data/b.zarr --output-dir runs/mace-ft --out ft.json\n\n"
    "Train from scratch with a user-provided model spec placeholder:\n\n"
    "  nvalchemi-training train init --dataset data/train.zarr --output-dir runs/train --out train.json\n\n"
    "Validate and review an existing config:\n\n"
    "  nvalchemi-training spec report train.json\n\n"
    "Dump an AIMNet2 fine-tuning template:\n\n"
    "  nvalchemi-training schema template --workflow finetune --model aimnet2 --out aimnet2-ft.json\n"
)

_TRAIN_EPILOG = (
    "Execution: review the generated JSON with `spec report`, then start "
    "the run with `spec run`. Specs with `strategy.model_specs` can execute "
    "directly; otherwise provide runtime model construction in the spec.\n\n"
    "Example:\n\n"
    "  nvalchemi-training train init --dataset data/a.zarr --dataset data/b.zarr --output-dir runs/train --out train.json\n"
)

_FINETUNE_EPILOG = (
    "Execution: use `spec report` to review fine-tuning intent, then "
    "`spec run` to build supported source models or native checkpoints and "
    "call `FineTuningStrategy.run(...)`.\n\n"
    "Examples:\n\n"
    "  nvalchemi-training finetune init checkpoint runs/pretrain/checkpoints --dataset data/domain.zarr --output-dir runs/domain-ft --out ft.json\n\n"
    "  nvalchemi-training finetune init mace small-0b --dataset data/domain.zarr --output-dir runs/mace-ft --out mace-ft.json\n"
)

_SCHEMA_EPILOG = (
    "Examples:\n\n"
    "  nvalchemi-training schema dump --out training.schema.json\n\n"
    "  nvalchemi-training schema template --workflow train --out train.json\n\n"
    "  nvalchemi-training schema template --workflow finetune --model aimnet2 --out aimnet2-ft.json\n"
)

_SPEC_EPILOG = (
    "`spec report` validates local paths, deserializes strategy components, "
    "and renders warnings without loading models or tensors. `spec run` "
    "constructs datasets, source models, hooks, and the strategy, then calls "
    "`run(...)`.\n\n"
    "Examples:\n\n"
    "Report and validate a saved config:\n\n"
    "  nvalchemi-training spec report train.json\n\n"
    "Report and print normalized JSON:\n\n"
    "  nvalchemi-training spec report train.json --json\n\n"
    "Execute a reviewed spec locally:\n\n"
    "  nvalchemi-training spec run train.json\n\n"
    "Execute under torchrun with DDP wiring:\n\n"
    "  torchrun --nproc_per_node=4 -m nvalchemi.training.cli spec run train.json --distributed\n"
)

TrainingWorkflow: TypeAlias = Literal["train", "finetune"]
StrategySpec: TypeAlias = Mapping[str, Any]

_DTYPE_POLICIES: tuple[DTypePolicy, ...] = get_args(DTypePolicy)

_MODEL_SOURCES: tuple[ModelSource, ...] = (
    "native-checkpoint",
    "mace",
    "aimnet2",
    "custom",
)


class TrainingJobSpec(BaseModel):
    """Top-level CLI envelope describing one training or fine-tuning job.

    ``TrainingJobSpec`` is the object the CLI reads from and writes to disk as a
    single JSON file (see ``_load_job_spec`` / ``schema template``). It bundles
    the intent needed to review and run a job: ``workflow`` (``train`` vs.
    ``finetune``), a :class:`SourceSpec` (``source``), a :class:`DatasetSpec`
    (``dataset``), an :class:`OutputSpec` (``output``), an optional
    :class:`ValidationSpec` (``validation``), and ``strategy``, the JSON-ready
    ``FineTuningStrategy.to_spec_dict()`` bundle that carries optimizers, loss,
    devices, and trainable/freeze patterns. ``spec report`` renders this intent
    and ``spec run`` builds the dataset, source model, hooks, and strategy from
    it, then calls ``run(...)``. Runtime hooks live under ``source.hooks`` rather
    than in ``strategy``, since they are attached at execution time.

    Examples
    --------
    Because jobs are loaded from a JSON config, a minimal fine-tuning envelope
    looks like:

    .. code-block:: json

        {
          "name": "mace-fine-tune",
          "workflow": "finetune",
          "source": {"model": "mace", "model_id": "small-0b"},
          "dataset": {"path": "data/domain.zarr", "format": "alchemi-zarr"},
          "output": {"run_dir": "runs/mace-ft"},
          "strategy": {
            "optimizer_configs": {"main": [{"optimizer_cls": "torch.optim.AdamW",
                                            "optimizer_kwargs": {"lr": 1e-5}}]},
            "num_steps": 1000,
            "devices": ["cuda"],
            "loss_fn_spec": {"...": "ComposedLossFunction spec"}
          }
        }

    Prefer :meth:`template` to construct a validated scaffold rather than hand-
    writing the ``strategy`` bundle.

    Notes
    -----
    Two ``after`` validators cross-check the pieces. ``_validate_workflow_source``
    enforces source rules per workflow: ``train`` requires
    ``source.model == "custom"`` and forbids ``source.checkpoint_path`` /
    ``source.model_id``; fine-tune sources require the appropriate checkpoint or
    model id, and a ``source.mace`` block is rejected unless the model is MACE.
    ``_validate_strategy`` requires ``strategy`` to contain ``optimizer_configs``,
    ``devices``, and ``loss_fn_spec`` plus exactly one of ``num_epochs`` or
    ``num_steps``, and requires ``dataset.validation_path`` whenever
    ``validation`` is set. ``model_config`` forbids unknown top-level keys.
    """

    model_config = ConfigDict(extra="forbid")

    name: Annotated[str, Field(description="Human-readable training job name.")] = (
        "training-job"
    )
    workflow: Annotated[
        TrainingWorkflow,
        Field(
            description="Whether this job trains from scratch or fine-tunes a source."
        ),
    ]
    source: Annotated[SourceSpec, Field(description="Model source intent.")]
    dataset: Annotated[DatasetSpec, Field(description="Training dataset intent.")]
    output: Annotated[OutputSpec, Field(description="Output path intent.")]
    validation: Annotated[
        ValidationSpec | None,
        Field(description="Optional validation cadence for CLI execution."),
    ] = None
    strategy: Annotated[
        dict[str, Any],
        Field(
            description="JSON-ready bundle produced by FineTuningStrategy.to_spec_dict()."
        ),
    ]
    notes: Annotated[
        str | None,
        Field(description="Optional notes rendered in the Rich report."),
    ] = None

    @model_validator(mode="after")
    def _validate_workflow_source(self) -> Self:
        """Validate source metadata required by the selected workflow."""
        source = self.source
        if self.workflow == "train":
            if source.checkpoint_path or source.model_id:
                raise ValueError(
                    "train workflow starts from model_specs and must not set "
                    "source.checkpoint_path or source.model_id."
                )
            if source.model != "custom":
                raise ValueError(
                    "train workflow currently expects source.model='custom' for "
                    "a user-provided model specification."
                )
            return self
        if (
            source.model != "mace"
            and (source.model_extra or {}).get("mace") is not None
        ):
            raise ValueError(
                "source.mace options are only valid when source.model='mace'."
            )
        if source.model == "native-checkpoint" and not source.checkpoint_path:
            raise ValueError("native-checkpoint specs require source.checkpoint_path.")
        if source.model == "mace":
            MaceSourceOptions.from_source(source)
        if source.model in {"mace", "aimnet2"} and not (
            source.model_id or source.checkpoint_path
        ):
            raise ValueError(
                f"{source.model} specs require source.model_id or source.checkpoint_path."
            )
        if source.model == "custom" and not source.checkpoint_path:
            raise ValueError("custom fine-tune specs require source.checkpoint_path.")
        return self

    @model_validator(mode="after")
    def _validate_strategy(self) -> Self:
        """Validate strategy payload with the existing training serializers."""
        missing = [
            key
            for key in ("optimizer_configs", "devices", "loss_fn_spec")
            if key not in self.strategy
        ]
        if missing:
            raise ValueError(
                "strategy is missing required FineTuningStrategy spec key(s) "
                f"{missing}."
            )
        num_epochs = self.strategy.get("num_epochs")
        num_steps = self.strategy.get("num_steps")
        if num_epochs is not None and num_steps is not None:
            raise ValueError("strategy must set only one of num_epochs or num_steps.")
        if num_epochs is None and num_steps is None:
            raise ValueError("strategy must set one of num_epochs or num_steps.")
        strategy_spec._optimizer_configs_from_spec(self.strategy["optimizer_configs"])
        strategy_spec._devices_from_spec(self.strategy["devices"])
        strategy_spec._loss_fn_from_spec(self.strategy["loss_fn_spec"])
        strategy_spec._training_fn_from_spec(self.strategy, None)
        if self.strategy.get("model_specs"):
            FineTuningStrategy.from_spec_dict(dict(self.strategy), hooks=[])
        if self.validation is not None and self.dataset.validation_path is None:
            raise ValueError(
                "validation cadence requires dataset.validation_path to be set."
            )
        return self

    @classmethod
    def template(
        cls,
        *,
        workflow: TrainingWorkflow,
        model: ModelSource,
        dataset: str | tuple[str, ...],
        output_dir: str,
        source_path: str | None = None,
        model_id: str | None = None,
        lr: float = 1e-5,
        num_steps: int | None = 1000,
        num_epochs: int | None = None,
        device: str = "cuda",
        trainable_patterns: tuple[str, ...] = (),
        compile_model: bool | None = None,
        hooks: tuple[dict[str, Any], ...] = (),
        loss_dtype_policy: DTypePolicy = "strict",
        validation_path: str | None = None,
        validation_every_epochs: int | None = None,
        validation_every_steps: int | None = None,
    ) -> Self:
        """Build a validated scaffold for a training or fine-tuning job."""
        source: dict[str, Any] = {"model": model}
        if source_path is not None:
            source["checkpoint_path"] = source_path
        if model_id is not None:
            source["model_id"] = model_id
        if compile_model is not None:
            source["compile_model"] = compile_model
        if hooks:
            source["hooks"] = list(hooks)
        if model == "native-checkpoint":
            source.update(
                {
                    "checkpoint_index": -1,
                    "use_original_loss": False,
                    "use_original_opt_class": False,
                    "optimizer_lr": 1e-5,
                }
            )
        if validation_every_epochs is not None and validation_every_steps is not None:
            raise click.ClickException(
                "Use only one of --validation-every-epochs or --validation-every-steps."
            )
        dataset_payload = _dataset_payload(dataset)
        validation_payload = None
        if validation_path is not None:
            dataset_payload["validation_path"] = validation_path
            validation_payload = {
                "every_n_epochs": validation_every_epochs,
                "every_n_steps": validation_every_steps,
            }
            if validation_every_epochs is None and validation_every_steps is None:
                validation_payload["every_n_epochs"] = 1
        name = f"{model}-fine-tune" if workflow == "finetune" else "train-from-scratch"
        return cls(
            name=name,
            workflow=workflow,
            source=source,
            dataset=dataset_payload,
            output={
                "run_dir": output_dir,
                "checkpoint_dir": str(Path(output_dir) / "checkpoints"),
            },
            validation=validation_payload,
            strategy=_default_strategy_spec(
                lr=lr,
                num_steps=num_steps,
                num_epochs=num_epochs,
                device=device,
                trainable_patterns=trainable_patterns,
                loss_dtype_policy=loss_dtype_policy,
            ),
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


def _dataset_payload(dataset: str | tuple[str, ...]) -> dict[str, Any]:
    """Build the JSON-ready dataset payload for one or more paths."""
    paths = (dataset,) if isinstance(dataset, str) else tuple(dataset)
    if not paths:
        raise click.ClickException("At least one --dataset path is required.")
    if len(paths) == 1:
        return {"path": paths[0], "format": "alchemi-zarr"}
    return {"paths": list(paths), "format": "alchemi-zarr-multidataset"}


def _format_dataset_spec(dataset: DatasetSpec) -> str:
    """Format single-dataset or multidataset intent for Rich tables."""
    if len(dataset.paths) > 1:
        return "MultiDataset:\n" + "\n".join(dataset.paths)
    path = dataset.path or (dataset.paths[0] if dataset.paths else None)
    return f"{_format_optional(path)} ({dataset.format})"


def _default_strategy_spec(
    *,
    lr: float,
    num_steps: int | None,
    num_epochs: int | None,
    device: str,
    trainable_patterns: tuple[str, ...],
    loss_dtype_policy: DTypePolicy = "strict",
) -> dict[str, Any]:
    """Return a conservative ``FineTuningStrategy.to_spec_dict`` scaffold."""
    optimizer_config = OptimizerConfig(
        optimizer_cls=torch.optim.AdamW,
        optimizer_kwargs={"lr": lr, "weight_decay": 1e-6},
    )
    loss_fn = ComposedLossFunction(
        [EnergyMSELoss(), ForceMSELoss(normalize_by_atom_count=True)],
        weights=[1.0, 10.0],
        normalize_weights=False,
        dtype_policy=loss_dtype_policy,
    )
    return {
        "optimizer_configs": {"main": [optimizer_config.to_spec().model_dump()]},
        "num_epochs": num_epochs,
        "num_steps": num_steps,
        "epoch_step_modifier": 1.0,
        "devices": [device],
        "loss_fn_spec": loss_fn.to_spec().model_dump(),
        "model_specs": {},
        "single_model_input": True,
        "training_fn": "nvalchemi.training.strategy.default_training_fn",
        "module_patches": {},
        "freeze_patterns": [],
        "trainable_patterns": list(trainable_patterns),
        "freeze_mode": "requires_grad",
    }


def _load_job_spec(path: Path) -> TrainingJobSpec:
    """Load and validate a training job specification from JSON."""
    try:
        raw = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise click.ClickException(f"Could not parse {path}: {exc}") from exc
    try:
        return TrainingJobSpec.model_validate(raw)
    except ValidationError as exc:
        raise click.ClickException(str(exc)) from exc


def _local_path_checks(job: TrainingJobSpec) -> list[tuple[str, str | None]]:
    """Return local filesystem paths that report validation should check."""
    checks: list[tuple[str, str | None]] = []
    dataset = job.dataset
    source = job.source
    if dataset.paths:
        checks.extend(
            (f"dataset.paths[{index}]", value)
            for index, value in enumerate(dataset.paths)
        )
    else:
        checks.append(("dataset.path", dataset.path))
    checks.append(("dataset.validation_path", dataset.validation_path))
    if source.checkpoint_path is not None:
        checks.append(("source.checkpoint_path", source.checkpoint_path))
    return checks


def _missing_local_paths(job: TrainingJobSpec) -> list[tuple[str, str]]:
    """Return local paths that are referenced by a job but missing on disk."""
    return [
        (field, value)
        for field, value in _local_path_checks(job)
        if value is not None and not path_exists(value)
    ]


def _build_runtime_hooks(
    job: TrainingJobSpec, *, enable_ddp: bool, ddp_backend: str | None
) -> list[Any]:
    """Build runtime hooks declared by the job and CLI execution options."""
    return build_runtime_hooks(
        job.source.hooks, enable_ddp=enable_ddp, ddp_backend=ddp_backend
    )


def _attach_validation_config(
    strategy: TrainingStrategy,
    job: TrainingJobSpec,
    stack: ExitStack,
    *,
    device: Any,
    batch_size: int | None,
    prefetch_factor: int,
    num_streams: int,
    use_streams: bool,
    pin_memory: bool,
    validation_path: str | None,
    validation_every_epochs: int | None,
    validation_every_steps: int | None,
) -> None:
    """Attach CLI validation data to a strategy when configured."""
    config = build_validation_config(
        job,
        stack,
        device=device,
        batch_size=batch_size,
        prefetch_factor=prefetch_factor,
        num_streams=num_streams,
        use_streams=use_streams,
        pin_memory=pin_memory,
        validation_path=validation_path,
        validation_every_epochs=validation_every_epochs,
        validation_every_steps=validation_every_steps,
    )
    if config is not None:
        strategy.validation_config = config


def _finetuning_kwargs_from_spec(
    spec: Mapping[str, Any],
    *,
    hooks: list[Any],
    training_fn: Any = None,
) -> dict[str, Any]:
    """Convert a serialized strategy payload into constructor kwargs."""
    return {
        "optimizer_configs": strategy_spec._optimizer_configs_from_spec(
            spec["optimizer_configs"]
        ),
        "num_epochs": spec.get("num_epochs"),
        "num_steps": spec.get("num_steps"),
        "epoch_step_modifier": spec.get("epoch_step_modifier", 1.0),
        "hooks": hooks,
        "training_fn": strategy_spec._training_fn_from_spec(spec, training_fn),
        "loss_fn": strategy_spec._loss_fn_from_spec(spec["loss_fn_spec"]),
        "devices": strategy_spec._devices_from_spec(spec["devices"]),
        "module_patches": {
            target: create_model_spec_from_json(raw_spec)
            for target, raw_spec in spec.get("module_patches", {}).items()
        },
        "freeze_patterns": tuple(spec.get("freeze_patterns", ())),
        "trainable_patterns": tuple(spec.get("trainable_patterns", ())),
        "freeze_mode": spec.get("freeze_mode", "requires_grad"),
    }


def _ensure_executable_job(job: TrainingJobSpec) -> None:
    """Raise actionable errors for spec fields that CLI run cannot execute."""
    if job.workflow == "train" and not job.strategy.get("model_specs"):
        raise click.ClickException(
            "spec run requires strategy.model_specs for training-from-scratch "
            "jobs. Add at least one serialized model spec or use the Python API "
            "to construct models before execution."
        )


def _build_strategy(
    job: TrainingJobSpec,
    *,
    hooks: list[Any],
    distributed_manager: Any | None,
    map_location: str | None,
) -> TrainingStrategy:
    """Build the concrete strategy declared by a CLI job spec."""
    _ensure_executable_job(job)
    if job.workflow == "train":
        strategy = TrainingStrategy.from_spec_dict(job.strategy, hooks=hooks)
    elif job.source.model == "native-checkpoint":
        if job.source.checkpoint_path is None:
            raise click.ClickException(
                "native-checkpoint run requires checkpoint_path."
            )
        strategy = FineTuningStrategy.from_pretrained_checkpoint(
            job.source.checkpoint_path,
            checkpoint_index=job.source.checkpoint_index,
            map_location=map_location,
            use_original_loss=job.source.use_original_loss,
            use_original_opt_class=job.source.use_original_opt_class,
            optimizer_lr=job.source.optimizer_lr,
            distributed_manager=distributed_manager,
            **_finetuning_kwargs_from_spec(job.strategy, hooks=hooks),
        )
    elif job.source.model in {"mace", "aimnet2"}:
        model = build_supported_source_model(
            job.source,
            device=distributed_manager.device
            if distributed_manager is not None
            else primary_strategy_device(job),
        )
        strategy = FineTuningStrategy.from_spec_dict(
            job.strategy, models=model, hooks=hooks
        )
    elif job.strategy.get("model_specs"):
        strategy = FineTuningStrategy.from_spec_dict(job.strategy, hooks=hooks)
    else:
        raise click.ClickException(
            "custom fine-tuning execution requires strategy.model_specs or a "
            "supported source model. Use native-checkpoint for nvalchemi "
            "checkpoint directories."
        )
    strategy.distributed_manager = distributed_manager
    return strategy


def _run_job(
    job: TrainingJobSpec,
    *,
    batch_size: int | None,
    shuffle: bool,
    drop_last: bool,
    prefetch_factor: int,
    num_streams: int,
    use_streams: bool,
    pin_memory: bool,
    distributed: bool | None,
    ddp_backend: str | None,
    map_location: str | None,
    validation_path: str | None = None,
    validation_every_epochs: int | None = None,
    validation_every_steps: int | None = None,
) -> None:
    """Construct runtime components and execute a CLI job."""
    distributed_enabled = resolve_distributed_enabled(distributed)
    distributed_manager = setup_distributed_manager(distributed_enabled)
    hooks = _build_runtime_hooks(
        job, enable_ddp=distributed_enabled, ddp_backend=ddp_backend
    )
    strategy = _build_strategy(
        job,
        hooks=hooks,
        distributed_manager=distributed_manager,
        map_location=map_location,
    )
    with ExitStack() as stack:
        device = dataset_device(job, distributed_manager)
        dataloader = build_dataloader(
            job,
            stack,
            device=device,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            prefetch_factor=prefetch_factor,
            num_streams=num_streams,
            use_streams=use_streams,
            pin_memory=pin_memory,
        )
        _attach_validation_config(
            strategy,
            job,
            stack,
            device=device,
            batch_size=batch_size,
            prefetch_factor=prefetch_factor,
            num_streams=num_streams,
            use_streams=use_streams,
            pin_memory=pin_memory,
            validation_path=validation_path,
            validation_every_epochs=validation_every_epochs,
            validation_every_steps=validation_every_steps,
        )
        strategy.run(dataloader)


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
    table.add_row("loss dtype policy", _loss_dtype_policy(strategy))
    table.add_row(
        "trainable_patterns", _format_sequence(strategy.get("trainable_patterns"))
    )
    table.add_row("freeze_patterns", _format_sequence(strategy.get("freeze_patterns")))
    table.add_row(
        "module_patches", _format_mapping_keys(strategy.get("module_patches"))
    )
    return table


def _intent_section(job: TrainingJobSpec) -> Table:
    """Build a Rich table summarizing source, data, and output intent."""
    source = job.source
    dataset = job.dataset
    output = job.output
    table = Table(title="Training Intent", box=box.SIMPLE_HEAD, expand=True)
    table.add_column("Area", style="cyan", no_wrap=True)
    table.add_column("Value", overflow="fold")
    table.add_row("job", job.name)
    table.add_row("model", source.model)
    table.add_row("source checkpoint", _format_optional(source.checkpoint_path))
    table.add_row("model id", _format_optional(source.model_id))
    table.add_row("checkpoint index", str(source.checkpoint_index))
    table.add_row("compile_model", _format_optional(source.compile_model))
    table.add_row("reuse source loss", str(source.use_original_loss))
    table.add_row("reuse source optimizer", str(source.use_original_opt_class))
    if source.model == "mace":
        mace_options = MaceSourceOptions.from_source(source)
        if mace_options.atomic_energies is not None:
            table.add_row(
                "MACE E0 override",
                f"inline ({len(mace_options.atomic_energies)} elements)",
            )
        elif mace_options.atomic_energies_path is not None:
            table.add_row("MACE E0 override", mace_options.atomic_energies_path)
    table.add_row("hooks", _format_hook_specs(source.hooks))
    table.add_row("dataset", _format_dataset_spec(dataset))
    table.add_row("validation data", _format_optional(dataset.validation_path))
    if job.validation is not None:
        cadence = (
            f"every {job.validation.every_n_steps} steps"
            if job.validation.every_n_steps is not None
            else f"every {job.validation.every_n_epochs or 1} epochs"
        )
        table.add_row("validation cadence", cadence)
    table.add_row("batch size", _format_optional(dataset.batch_size))
    table.add_row("run dir", output.run_dir)
    table.add_row("checkpoint dir", _format_optional(output.checkpoint_dir))
    return table


def _warning_section(job: TrainingJobSpec) -> Table:
    """Build a Rich table of training heuristic warnings."""
    table = Table(title="Warnings", box=box.SIMPLE_HEAD, expand=True)
    table.add_column("Level", no_wrap=True)
    table.add_column("Check", style="cyan", no_wrap=True)
    table.add_column("Details", overflow="fold")
    warnings = _training_warnings(job)
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


def _has_checkpoint_hook(job: TrainingJobSpec) -> bool:
    """Return whether runtime hooks include restart checkpoint writing."""
    return any(
        hook.spec.cls_path == "nvalchemi.training.hooks.checkpoint.CheckpointHook"
        for hook in job.source.hooks
    )


def _training_warnings(job: TrainingJobSpec) -> list[tuple[str, str, str]]:
    """Return heuristic warnings for common training mistakes."""
    source = job.source
    dataset = job.dataset
    output = job.output
    strategy = job.strategy
    warnings: list[tuple[str, str, str]] = []
    model = source.model
    for field, value in _missing_local_paths(job):
        warnings.append(
            (
                "warning",
                "Missing local path",
                f"{field} does not exist: {value}",
            )
        )
    if job.workflow == "finetune" and model == "mace" and source.compile_model is True:
        warnings.append(
            (
                "warning",
                "MACE compile",
                "compile_model=true is inference-oriented for MACE; use false for fine-tuning.",
            )
        )
    if (
        job.workflow == "finetune"
        and not strategy.get("trainable_patterns")
        and not strategy.get("freeze_patterns")
    ):
        warnings.append(
            (
                "warning",
                "Full model update",
                "No trainable or freeze patterns are set, so every optimizer-configured parameter can update.",
            )
        )
    if (
        job.workflow == "finetune"
        and not strategy.get("module_patches")
        and not strategy.get("trainable_patterns")
    ):
        warnings.append(
            (
                "warning",
                "No adaptation boundary",
                "No module patches or trainable allow-list are declared; confirm full-model fine-tuning is intended.",
            )
        )
    max_lr = max((row[2] for row in _optimizer_rows(strategy)), default=None)
    if job.workflow == "finetune" and max_lr is not None and max_lr > 1e-4:
        warnings.append(
            (
                "warning",
                "Learning rate",
                f"The largest optimizer LR is {max_lr:.3g}; pretrained fine-tuning usually starts at 1e-5 to 1e-4.",
            )
        )
    if not dataset.validation_path:
        warnings.append(
            (
                "warning",
                "Validation data",
                "No dataset.validation_path is recorded; make sure validation_config is supplied in the execution script.",
            )
        )
    if not output.checkpoint_dir:
        warnings.append(
            (
                "warning",
                "Restart checkpoints",
                "No output.checkpoint_dir is recorded for restartable training checkpoints.",
            )
        )
    elif not _has_checkpoint_hook(job):
        warnings.append(
            (
                "warning",
                "Checkpoint hook",
                "output.checkpoint_dir is recorded, but no CheckpointHook is declared in source.hooks; spec run will not write restart checkpoints.",
            )
        )
    if output.checkpoint_dir and output.checkpoint_dir == source.checkpoint_path:
        warnings.append(
            (
                "danger",
                "Checkpoint overwrite",
                "Output checkpoint_dir matches the source checkpoint path.",
            )
        )
    if (
        model == "native-checkpoint"
        and source.use_original_opt_class
        and max_lr is None
    ):
        warnings.append(
            (
                "warning",
                "Optimizer reuse",
                "Optimizer class reuse is requested, but the strategy spec does not expose an LR preview.",
            )
        )
    if job.workflow == "train" and not strategy.get("model_specs"):
        warnings.append(
            (
                "warning",
                "Model spec",
                "Training-from-scratch specs need strategy.model_specs before execution can build a model.",
            )
        )
    if (
        job.workflow == "finetune"
        and model == "custom"
        and not strategy.get("model_specs")
    ):
        warnings.append(
            (
                "warning",
                "Custom model reload",
                "No model_specs are present; the execution script must instantiate the model and load weights.",
            )
        )
    devices = strategy.get("devices", None)
    if isinstance(devices, str) and "cuda" not in devices:
        warnings.append(
            (
                "warning",
                "Device selection",
                "CUDA device not selected; specify 'cuda' in strategy for production runs.",
            )
        )
    if isinstance(devices, list):
        for index, device in enumerate(devices):
            if "cuda" not in device:
                warnings.append(
                    (
                        "warning",
                        "Device selection",
                        f"CUDA device not selected for device index {index}. Confirm this is intended.",
                    )
                )
            if "cuda" in device and not torch.cuda.is_available():
                warnings.append(
                    (
                        "warning",
                        "Device selection",
                        f"CUDA device selected at index {index}, but no CUDA devices are available.",
                    )
                )
    return warnings


def _format_hook_specs(value: list[RuntimeHookSpec]) -> str:
    """Format serialized hook specs for Rich tables."""
    if not value:
        return "none"
    names = [hook.spec.cls_path for hook in value]
    return "\n".join(names)


def _hook_stage_names(hook: RuntimeHookSpec) -> list[str]:
    """Return declared or spec-stored stage names for a runtime hook."""
    if hook.stages:
        return list(hook.stages)
    raw_stage = (hook.spec.model_extra or {}).get("stage")
    if raw_stage is None:
        return ["constructor default"]
    try:
        return [_training_stage_name(raw_stage)]
    except ValueError:
        return [str(raw_stage)]


def _hook_order_section(job: TrainingJobSpec) -> Table:
    """Build a Rich table showing hook firing order by training stage."""
    table = Table(title="Hook Firing Order", box=box.SIMPLE_HEAD, expand=True)
    table.add_column("Stage", style="cyan", no_wrap=True)
    table.add_column("Order", justify="right", no_wrap=True)
    table.add_column("Hook", overflow="fold")
    table.add_column("Source", overflow="fold")
    rows: list[tuple[int, int, str, str, str]] = []
    for index, hook in enumerate(job.source.hooks):
        name = hook.spec.cls_path
        for stage_name in _hook_stage_names(hook):
            stage_index = (
                list(TrainingStage).index(TrainingStage[stage_name])
                if stage_name in TrainingStage.__members__
                else len(TrainingStage)
            )
            rows.append((stage_index, index, stage_name, str(index + 1), name))
    if not rows:
        table.add_row("none", "", "No runtime hooks declared.", "source.hooks")
        return table
    for _, _, stage_name, order, name in sorted(rows):
        table.add_row(stage_name, order, name, "source.hooks")
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


def _loss_dtype_policy(strategy: StrategySpec) -> str:
    """Return the composed loss dtype policy from a strategy spec."""
    loss_fn_spec = strategy.get("loss_fn_spec")
    if not isinstance(loss_fn_spec, Mapping):
        return "not specified"
    value = loss_fn_spec.get("dtype_policy")
    return "not specified" if value is None else str(value)


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


def _render_report(job: TrainingJobSpec) -> None:
    """Render a Rich report card for a training job spec."""
    strategy = job.strategy
    console.rule(f"[bold]Training report: {job.name}")
    console.print(_intent_section(job))
    console.print(_hook_order_section(job))
    console.print(_warning_section(job))
    console.print(_strategy_section(strategy))
    console.print(_optimizer_section(strategy))
    if job.notes:
        console.print(Panel(Text(job.notes, overflow="fold"), title="Notes"))
    console.print(
        Panel(_LRSchedulePlot(_lr_series(strategy)), title="Learning-rate preview")
    )


def _print_template_message(
    output: Path | None, workflow: TrainingWorkflow, model: ModelSource
) -> None:
    """Print a concise scaffold-generation status message."""
    if output is not None:
        console.print(f"[green]Created {workflow} {model} training spec[/] {output}")


def _common_template_options(function: Any) -> Any:
    """Attach common template options to a model scaffold command."""
    options = [
        click.option(
            "--dataset",
            "dataset",
            required=True,
            multiple=True,
            help="Training dataset path or URI. Repeat to create a MultiDataset intent.",
        ),
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
            help="Initial training learning rate.",
        ),
        click.option(
            "--num-steps",
            type=int,
            default=1000,
            show_default=True,
            help="Number of training steps.",
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
        click.option(
            "--loss-dtype-policy",
            type=click.Choice(_DTYPE_POLICIES),
            default="strict",
            show_default=True,
            help="Loss prediction/target dtype alignment policy written to the spec.",
        ),
        click.option(
            "--validation-dataset",
            "validation_path",
            default=None,
            help="Validation dataset path or URI to record in the spec.",
        ),
        click.option(
            "--validation-every-epochs",
            "validation_every_epochs",
            type=int,
            default=None,
            help="Run validation every N completed epochs.",
        ),
        click.option(
            "--validation-every-steps",
            "validation_every_steps",
            type=int,
            default=None,
            help="Run validation every N optimizer steps.",
        ),
    ]
    for option in reversed(options):
        function = option(function)
    return function


class _TrainingGroup(click.Group):
    """Training CLI root that resolves the distillation group on demand.

    The distillation recipe CLI lives beside the strategy it drives and pulls
    in the dynamics and evaluation stack a recipe needs. Importing it only
    when ``distill`` is looked up, or when the command list is rendered, keeps
    that stack out of an ordinary training run while ``--help`` stays
    complete.
    """

    def _load_distillation(self) -> None:
        """Attach the distillation group unless it is already registered."""
        if "distill" in self.commands:
            return
        from nvalchemi.training.distillation.cli import distill

        self.add_command(distill)

    def get_command(self, ctx: click.Context, name: str) -> click.Command | None:
        """Return a registered command, resolving the distillation group by name."""
        if name == "distill":
            self._load_distillation()
        return super().get_command(ctx, name)

    def list_commands(self, ctx: click.Context) -> list[str]:
        """Return every command name, resolving the distillation group first."""
        self._load_distillation()
        return super().list_commands(ctx)


@click.group(
    cls=_TrainingGroup, context_settings={"max_content_width": 100}, epilog=_MAIN_EPILOG
)
def main() -> None:
    """Review and scaffold nvalchemi training specifications."""


@main.group(epilog=_TRAIN_EPILOG)
def train() -> None:
    """Create training-from-scratch specification scaffolds."""


@main.group(epilog=_FINETUNE_EPILOG)
def finetune() -> None:
    """Create fine-tuning specification scaffolds from pretrained sources."""


@finetune.group()
def init() -> None:
    """Create model-specific fine-tuning specification scaffolds."""


@main.group(epilog=_SCHEMA_EPILOG)
def schema() -> None:
    """Dump JSON schema and templates for offline specification authoring."""


@main.group(name="spec", epilog=_SPEC_EPILOG)
def spec_group() -> None:
    """Validate and report on saved training specifications."""


@train.command("init")
@_common_template_options
def init_train(
    dataset: tuple[str, ...],
    output_dir: str,
    output: Path | None,
    lr: float,
    num_steps: int | None,
    num_epochs: int | None,
    device: str,
    trainable_patterns: tuple[str, ...],
    loss_dtype_policy: DTypePolicy,
    validation_path: str | None,
    validation_every_epochs: int | None,
    validation_every_steps: int | None,
) -> None:
    """Create a spec for training a user-provided model from scratch."""
    if num_epochs is not None:
        num_steps = None
    payload = TrainingJobSpec.template(
        workflow="train",
        model="custom",
        dataset=dataset,
        output_dir=output_dir,
        lr=lr,
        num_steps=num_steps,
        num_epochs=num_epochs,
        device=device,
        trainable_patterns=trainable_patterns,
        loss_dtype_policy=loss_dtype_policy,
        validation_path=validation_path,
        validation_every_epochs=validation_every_epochs,
        validation_every_steps=validation_every_steps,
    )
    write_or_print(payload, output)
    _print_template_message(output, "train", "custom")


@schema.command("dump")
@click.option(
    "--out",
    "output",
    type=click.Path(path_type=Path),
    help="Write schema JSON to this file.",
)
def dump_schema(output: Path | None) -> None:
    """Dump the CLI training job JSON schema."""
    write_or_print(TrainingJobSpec.model_json_schema(), output)


@schema.command("template")
@click.option(
    "--workflow",
    type=click.Choice(["train", "finetune"]),
    default="finetune",
    show_default=True,
    help="Template workflow to generate.",
)
@click.option(
    "--model",
    "model_source",
    type=click.Choice(_MODEL_SOURCES),
    default="native-checkpoint",
    show_default=True,
    help="Fine-tuning model source to scaffold when --workflow=finetune.",
)
@click.option(
    "--out",
    "output",
    type=click.Path(path_type=Path),
    help="Write template JSON to this file.",
)
def dump_template(
    workflow: TrainingWorkflow, model_source: ModelSource, output: Path | None
) -> None:
    """Dump a training or fine-tuning template."""
    if workflow == "train":
        payload = TrainingJobSpec.template(
            workflow="train",
            model="custom",
            dataset="data/train.zarr",
            output_dir="runs/train",
            lr=1e-4,
            num_steps=1000,
            num_epochs=None,
            device="cuda",
            trainable_patterns=(),
            loss_dtype_policy="strict",
        )
    else:
        source_path = (
            "runs/pretrain/checkpoints"
            if model_source in {"native-checkpoint", "custom"}
            else None
        )
        model_id = "aimnet2-example" if model_source == "aimnet2" else None
        if model_source == "mace":
            model_id = "small-0b"
        payload = TrainingJobSpec.template(
            workflow="finetune",
            model=model_source,
            dataset="data/train.zarr",
            output_dir="runs/finetune",
            source_path=source_path,
            model_id=model_id,
            lr=1e-5,
            num_steps=1000,
            num_epochs=None,
            device="cuda",
            trainable_patterns=("main.model.readout.*",)
            if model_source == "native-checkpoint"
            else (),
            compile_model=False if model_source == "mace" else None,
            loss_dtype_policy="strict",
        )
    write_or_print(payload, output)


@init.command("checkpoint")
@_common_template_options
@click.argument("checkpoint_dir")
def init_checkpoint(
    checkpoint_dir: str,
    dataset: tuple[str, ...],
    output_dir: str,
    output: Path | None,
    lr: float,
    num_steps: int | None,
    num_epochs: int | None,
    device: str,
    trainable_patterns: tuple[str, ...],
    loss_dtype_policy: DTypePolicy,
    validation_path: str | None,
    validation_every_epochs: int | None,
    validation_every_steps: int | None,
) -> None:
    """Create a fine-tuning spec for a native nvalchemi checkpoint source."""
    if num_epochs is not None:
        num_steps = None
    payload = TrainingJobSpec.template(
        workflow="finetune",
        model="native-checkpoint",
        dataset=dataset,
        output_dir=output_dir,
        source_path=checkpoint_dir,
        lr=lr,
        num_steps=num_steps,
        num_epochs=num_epochs,
        device=device,
        trainable_patterns=trainable_patterns,
        loss_dtype_policy=loss_dtype_policy,
        validation_path=validation_path,
        validation_every_epochs=validation_every_epochs,
        validation_every_steps=validation_every_steps,
    )
    write_or_print(payload, output)
    _print_template_message(output, "finetune", "native-checkpoint")


@init.command("mace")
@_common_template_options
@click.argument("model_or_checkpoint")
def init_mace(
    model_or_checkpoint: str,
    dataset: tuple[str, ...],
    output_dir: str,
    output: Path | None,
    lr: float,
    num_steps: int | None,
    num_epochs: int | None,
    device: str,
    trainable_patterns: tuple[str, ...],
    loss_dtype_policy: DTypePolicy,
    validation_path: str | None,
    validation_every_epochs: int | None,
    validation_every_steps: int | None,
) -> None:
    """Create a fine-tuning spec for a MACE wrapper source."""
    if num_epochs is not None:
        num_steps = None
    source_path = model_or_checkpoint if Path(model_or_checkpoint).suffix else None
    model_id = None if source_path is not None else model_or_checkpoint
    payload = TrainingJobSpec.template(
        workflow="finetune",
        model="mace",
        dataset=dataset,
        output_dir=output_dir,
        source_path=source_path,
        model_id=model_id,
        lr=lr,
        num_steps=num_steps,
        num_epochs=num_epochs,
        device=device,
        trainable_patterns=trainable_patterns or ("main.model.readouts.*",),
        compile_model=False,
        loss_dtype_policy=loss_dtype_policy,
        validation_path=validation_path,
        validation_every_epochs=validation_every_epochs,
        validation_every_steps=validation_every_steps,
    )
    write_or_print(payload, output)
    _print_template_message(output, "finetune", "mace")


@init.command("aimnet2")
@_common_template_options
@click.argument("model_or_checkpoint")
def init_aimnet2(
    model_or_checkpoint: str,
    dataset: tuple[str, ...],
    output_dir: str,
    output: Path | None,
    lr: float,
    num_steps: int | None,
    num_epochs: int | None,
    device: str,
    trainable_patterns: tuple[str, ...],
    loss_dtype_policy: DTypePolicy,
    validation_path: str | None,
    validation_every_epochs: int | None,
    validation_every_steps: int | None,
) -> None:
    """Create a fine-tuning spec for an AIMNet2 wrapper source."""
    if num_epochs is not None:
        num_steps = None
    source_path = model_or_checkpoint if Path(model_or_checkpoint).suffix else None
    model_id = None if source_path is not None else model_or_checkpoint
    payload = TrainingJobSpec.template(
        workflow="finetune",
        model="aimnet2",
        dataset=dataset,
        output_dir=output_dir,
        source_path=source_path,
        model_id=model_id,
        lr=lr,
        num_steps=num_steps,
        num_epochs=num_epochs,
        device=device,
        trainable_patterns=trainable_patterns,
        loss_dtype_policy=loss_dtype_policy,
        validation_path=validation_path,
        validation_every_epochs=validation_every_epochs,
        validation_every_steps=validation_every_steps,
    )
    write_or_print(payload, output)
    _print_template_message(output, "finetune", "aimnet2")


@init.command("custom")
@_common_template_options
@click.argument("checkpoint_path")
def init_custom(
    checkpoint_path: str,
    dataset: tuple[str, ...],
    output_dir: str,
    output: Path | None,
    lr: float,
    num_steps: int | None,
    num_epochs: int | None,
    device: str,
    trainable_patterns: tuple[str, ...],
    loss_dtype_policy: DTypePolicy,
    validation_path: str | None,
    validation_every_epochs: int | None,
    validation_every_steps: int | None,
) -> None:
    """Create a fine-tuning spec for a custom user-managed checkpoint."""
    if num_epochs is not None:
        num_steps = None
    payload = TrainingJobSpec.template(
        workflow="finetune",
        model="custom",
        dataset=dataset,
        output_dir=output_dir,
        source_path=checkpoint_path,
        lr=lr,
        num_steps=num_steps,
        num_epochs=num_epochs,
        device=device,
        trainable_patterns=trainable_patterns,
        loss_dtype_policy=loss_dtype_policy,
        validation_path=validation_path,
        validation_every_epochs=validation_every_epochs,
        validation_every_steps=validation_every_steps,
    )
    write_or_print(payload, output)
    _print_template_message(output, "finetune", "custom")


@spec_group.command("run")
@click.argument("path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@common_loader_options
@click.option(
    "--distributed/--no-distributed",
    default=None,
    help="Attach DistributedManager and DDPHook. Defaults to auto when WORLD_SIZE > 1.",
)
@click.option(
    "--ddp-backend",
    type=click.Choice(["nccl", "gloo"]),
    default=None,
    help="Process-group backend forwarded to DDPHook.",
)
@click.option(
    "--map-location",
    default=None,
    help="Checkpoint map_location for native-checkpoint fine-tuning sources.",
)
@common_validation_options
@click.option(
    "--report/--no-report",
    "show_report",
    default=True,
    show_default=True,
    help="Render the Rich report before execution.",
)
def run_spec(
    path: Path,
    batch_size: int | None,
    shuffle: bool,
    drop_last: bool,
    prefetch_factor: int,
    num_streams: int,
    pin_memory: bool,
    use_streams: bool,
    distributed: bool | None,
    ddp_backend: str | None,
    map_location: str | None,
    validation_path: str | None,
    validation_every_epochs: int | None,
    validation_every_steps: int | None,
    show_report: bool,
) -> None:
    """Construct runtime components and execute a saved training spec."""
    job = _load_job_spec(path)
    if show_report:
        _render_report(job)
    _run_job(
        job,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        prefetch_factor=prefetch_factor,
        num_streams=num_streams,
        use_streams=use_streams,
        pin_memory=pin_memory,
        distributed=distributed,
        ddp_backend=ddp_backend,
        map_location=map_location,
        validation_path=validation_path,
        validation_every_epochs=validation_every_epochs,
        validation_every_steps=validation_every_steps,
    )


@spec_group.command("resume")
@click.argument(
    "checkpoint_dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
)
@click.option(
    "--spec",
    "spec_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Training job spec that supplies dataloader and runtime hook intent.",
)
@click.option("--checkpoint-index", type=int, default=-1, show_default=True)
@common_loader_options
@click.option(
    "--distributed/--no-distributed",
    default=None,
    help="Attach DistributedManager and DDPHook. Defaults to auto when WORLD_SIZE > 1.",
)
@click.option(
    "--ddp-backend",
    type=click.Choice(["nccl", "gloo"]),
    default=None,
    help="Process-group backend forwarded to DDPHook.",
)
@click.option("--map-location", default=None, help="Checkpoint map_location.")
@common_validation_options
def resume_spec(
    checkpoint_dir: Path,
    spec_path: Path,
    checkpoint_index: int,
    batch_size: int | None,
    shuffle: bool,
    drop_last: bool,
    prefetch_factor: int,
    num_streams: int,
    pin_memory: bool,
    use_streams: bool,
    distributed: bool | None,
    ddp_backend: str | None,
    map_location: str | None,
    validation_path: str | None,
    validation_every_epochs: int | None,
    validation_every_steps: int | None,
) -> None:
    """Resume a restartable strategy checkpoint with a saved job spec."""
    job = _load_job_spec(spec_path)
    distributed_enabled = resolve_distributed_enabled(distributed)
    distributed_manager = setup_distributed_manager(distributed_enabled)
    hooks = _build_runtime_hooks(
        job, enable_ddp=distributed_enabled, ddp_backend=ddp_backend
    )
    strategy = TrainingStrategy.load_checkpoint(
        checkpoint_dir,
        checkpoint_index=checkpoint_index,
        map_location=map_location,
        hooks=hooks,
    )
    strategy.distributed_manager = distributed_manager
    with ExitStack() as stack:
        device = dataset_device(job, distributed_manager)
        dataloader = build_dataloader(
            job,
            stack,
            device=device,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            prefetch_factor=prefetch_factor,
            num_streams=num_streams,
            use_streams=use_streams,
            pin_memory=pin_memory,
        )
        _attach_validation_config(
            strategy,
            job,
            stack,
            device=device,
            batch_size=batch_size,
            prefetch_factor=prefetch_factor,
            num_streams=num_streams,
            use_streams=use_streams,
            pin_memory=pin_memory,
            validation_path=validation_path,
            validation_every_epochs=validation_every_epochs,
            validation_every_steps=validation_every_steps,
        )
        strategy.run(dataloader)


@spec_group.command("report")
@click.argument("path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "--json",
    "show_json",
    is_flag=True,
    help="Print normalized JSON after the Rich report.",
)
def report_spec(path: Path, show_json: bool) -> None:
    """Validate and render a Rich report card for a training spec."""
    job = _load_job_spec(path)
    _render_report(job)
    if show_json:
        console.print(Syntax(job.model_dump_json(indent=2), "json"))


if __name__ == "__main__":
    main()
