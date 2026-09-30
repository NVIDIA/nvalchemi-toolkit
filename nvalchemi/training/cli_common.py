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
"""Spec models and helpers shared by the training and distillation CLIs.

The Click commands in :mod:`nvalchemi.training.cli` and
:mod:`nvalchemi.training.distillation.cli` read the same source, dataset,
output, validation, and hook envelopes. They also build models, hooks, and
files the same way. The pieces that do not read a whole job spec live here, so
a second CLI imports them by their public names.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from contextlib import ExitStack
from pathlib import Path
from typing import Annotated, Any, Literal, Protocol, Self, TypeAlias, get_args

import click
import torch
from pydantic import BaseModel, ConfigDict, Field, model_validator
from rich.console import Console

from nvalchemi._serialization import _import_cls
from nvalchemi.hooks import CheckpointableHook, Hook
from nvalchemi.training import (
    DDPHook,
    TrainingStage,
    TrainingStrategy,
    ValidationConfig,
)
from nvalchemi.training import _spec_utils as strategy_spec
from nvalchemi.training._spec import create_model_spec_from_json
from nvalchemi.training.distributed import get_rank, get_world_size
from nvalchemi.training.hooks.update import TrainingUpdateHook

ModelSource: TypeAlias = Literal["native-checkpoint", "mace", "aimnet2", "custom"]
DatasetFormat: TypeAlias = Literal["alchemi-zarr", "alchemi-zarr-multidataset"]
ResumeBudget: TypeAlias = Literal["checkpoint", "recipe"]

console = Console(stderr=True)

__all__ = [
    "DataJobSpec",
    "DatasetFormat",
    "DatasetSpec",
    "HookSpec",
    "MaceSourceOptions",
    "ModelSource",
    "OutputSpec",
    "ResumeBudget",
    "RuntimeHookSpec",
    "SourceSpec",
    "ValidationSpec",
    "apply_resume_budget",
    "build_checked_hook",
    "build_dataloader",
    "build_runtime_hooks",
    "build_supported_source_model",
    "build_validation_config",
    "common_loader_options",
    "common_prefetch_options",
    "common_validation_options",
    "console",
    "dataset_device",
    "hook_spec_is",
    "path_exists",
    "primary_strategy_device",
    "resolve_distributed_enabled",
    "restart_map_location",
    "setup_distributed_manager",
    "validate_pretrained_source",
    "write_or_print",
]


def _training_stage_name(value: Any) -> str:
    """Return a canonical ``TrainingStage`` name from JSON-friendly input."""
    if isinstance(value, TrainingStage):
        return value.name
    if isinstance(value, int):
        try:
            return TrainingStage(value).name
        except ValueError as exc:
            raise ValueError(f"unknown TrainingStage value {value!r}") from exc
    if isinstance(value, str):
        name = value.removeprefix("TrainingStage.")
        try:
            return TrainingStage[name].name
        except KeyError as exc:
            raise ValueError(f"unknown TrainingStage name {value!r}") from exc
    raise ValueError(
        "Training stage overrides must be TrainingStage names or integer values."
    )


def _training_stage(value: Any) -> TrainingStage:
    """Return a ``TrainingStage`` from a canonical name or JSON value."""
    return TrainingStage[_training_stage_name(value)]


class HookSpec(BaseModel):
    """Serialized constructor payload for a single runtime training hook.

    A ``HookSpec`` is the innermost hook layer in the CLI job envelope: it
    mirrors the ``BaseSpec`` JSON emitted for a hook, carrying the dotted
    ``cls_path`` to import and a ``timestamp``, while ``extra="allow"`` retains
    the remaining keyword fields that get unpacked into the hook constructor
    at build time. It is wrapped by :class:`RuntimeHookSpec`, which pairs it
    with optional stage overrides; execution code turns it into a live hook via
    ``create_model_spec_from_json(...).build()`` (see ``build_checked_hook``).

    Examples
    --------
    A checkpoint hook spec as it appears inside ``source.hooks``::

        HookSpec(
            cls_path="nvalchemi.training.hooks.checkpoint.CheckpointHook",
            timestamp="2026-01-01T00:00:00Z",
            checkpoint_dir="runs/mace-ft/checkpoints",
        )

    Notes
    -----
    Extra keys beyond ``cls_path`` and ``timestamp`` are preserved (not
    rejected) and are forwarded as constructor keyword arguments; the built
    object must satisfy :class:`Hook`, :class:`CheckpointableHook`, or
    :class:`TrainingUpdateHook` or validation of the enclosing spec fails.
    """

    model_config = ConfigDict(extra="allow")

    cls_path: Annotated[
        str,
        Field(description="Dotted import path for the hook class or factory."),
    ]
    timestamp: Annotated[
        str,
        Field(description="Timestamp recorded by the serialized BaseSpec."),
    ]


class RuntimeHookSpec(BaseModel):
    """Runtime hook entry with optional training-stage overrides.

    Each element of ``SourceSpec.hooks`` is a ``RuntimeHookSpec``: it wraps one
    :class:`HookSpec` (the ``spec`` constructor payload) and an optional list of
    :class:`TrainingStage` names in ``stages`` that override where the hook
    fires. When several stages are listed, execution builds one hook instance
    per stage. These hooks are attached at run time by CLI execution code and
    are deliberately not part of ``FineTuningStrategy.to_spec_dict()``.

    Examples
    --------
    Fire a logging hook before and after each forward pass::

        RuntimeHookSpec(
            spec={
                "cls_path": "nvalchemi.training.hooks.logging.LoggingHook",
                "timestamp": "2026-01-01T00:00:00Z",
            },
            stages=["BEFORE_FORWARD", "AFTER_FORWARD"],
        )

    Notes
    -----
    A ``before`` validator accepts two JSON shapes: the explicit
    ``{"spec": ..., "stages": ...}`` form above, or a bare ``BaseSpec`` object
    (one containing ``cls_path``) whose ``stages``/``stage`` keys are lifted out
    automatically. Stage names are normalized and the hook is trial-built during
    validation, so an unimportable ``cls_path`` or a wrong hook type fails fast.
    Omitting ``stages`` falls back to the stage stored in the spec or the hook
    constructor default.
    """

    model_config = ConfigDict(extra="forbid")

    spec: Annotated[
        HookSpec,
        Field(
            description=(
                "Serialized hook constructor spec: cls_path, timestamp, and "
                "the keyword arguments to unpack into the hook constructor."
            )
        ),
    ]
    stages: list[str] = Field(
        default_factory=list,
        description=(
            "Optional TrainingStage name overrides where this hook should fire, "
            "for example ['BEFORE_FORWARD']. Multiple stages build one hook "
            "instance per stage. Omit to use the stage stored in spec or the "
            "hook constructor default."
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def _accept_raw_spec(cls, data: Any) -> Any:
        """Accept source.hooks entries that are bare BaseSpec JSON objects."""
        if isinstance(data, Mapping) and "spec" in data:
            return data
        if isinstance(data, Mapping) and "cls_path" in data:
            spec = dict(data)
            stages = spec.pop("stages", None)
            if stages is None and "stage" in spec:
                stages = [spec["stage"]]
            return {"spec": spec, "stages": stages or []}
        return data

    @model_validator(mode="after")
    def _validate_runtime_hook(self) -> Self:
        """Validate hook construction and normalize stage override names."""
        payload = _normalize_runtime_hook_spec(self.spec.model_dump(mode="json"))
        self.spec = HookSpec.model_validate(payload)
        build_checked_hook(self.spec)
        self.stages = [_training_stage_name(stage) for stage in self.stages]
        return self

    def stage_values(self) -> list[TrainingStage]:
        """Return explicit stage overrides as enum values."""
        return [_training_stage(stage) for stage in self.stages]


class MaceSourceOptions(BaseModel):
    """MACE-only source knobs parsed from the ``source.mace`` block.

    This is the architecture-specific options model referenced in the module
    docstring: rather than adding MACE-only fields to the shared
    :class:`SourceSpec`, MACE options live under a namespaced ``source.mace``
    object and are read out with :meth:`from_source`. It currently exposes the
    atomic-energy (E0) override, either inline per-element values or a path to a
    JSON file of them, consumed by ``MACEWrapper.from_checkpoint`` during
    ``spec run``.

    Examples
    --------
    Override E0 values for two elements inline::

        MaceSourceOptions(atomic_energies={1: -13.6, 8: -2043.9})

    Notes
    -----
    ``atomic_energies`` and ``atomic_energies_path`` are mutually exclusive; a
    validator rejects setting both. :class:`TrainingJobSpec` only accepts a
    ``source.mace`` block when ``source.model == "mace"``.
    """

    model_config = ConfigDict(extra="forbid")

    atomic_energies: Annotated[
        dict[int, float] | None,
        Field(description="Per-element E0 overrides keyed by atomic number."),
    ] = None
    atomic_energies_path: Annotated[
        str | None,
        Field(description="JSON file containing per-element E0 overrides."),
    ] = None

    @model_validator(mode="after")
    def _validate_single_atomic_energy_source(self) -> Self:
        """Require at most one atomic-energy override source."""
        if self.atomic_energies is not None and self.atomic_energies_path is not None:
            raise ValueError(
                "source.mace accepts only one of atomic_energies or "
                "atomic_energies_path."
            )
        return self

    @classmethod
    def from_source(cls, source: "SourceSpec") -> "MaceSourceOptions":
        """Return validated MACE-specific options from a source spec."""
        raw = (source.model_extra or {}).get("mace", {})
        return cls.model_validate(raw)

    @property
    def has_atomic_energy_override(self) -> bool:
        """Return whether E0 replacement was requested."""
        return self.atomic_energies is not None or self.atomic_energies_path is not None


class SourceSpec(BaseModel):
    """Where the job's model comes from, plus runtime hooks to attach.

    ``SourceSpec`` is the ``source`` member of :class:`TrainingJobSpec` and the
    flat, model-agnostic envelope described in the module docstring. ``model``
    selects the family (``native-checkpoint``, ``mace``, ``aimnet2``, or
    ``custom``) and the shared fields cover checkpoint location, model id,
    compile behavior, and optimizer reuse. Architecture-specific knobs are not
    fields here: ``extra="allow"`` lets them ride along under a namespaced block
    such as ``source.mace``, parsed by :class:`MaceSourceOptions`. ``hooks`` is a
    list of :class:`RuntimeHookSpec` attached at execution time.

    Examples
    --------
    Fine-tune a supported MACE model by id::

        SourceSpec(model="mace", model_id="small-0b", compile_model=False)

    Notes
    -----
    A ``before`` validator accepts a deprecated ``endpoint`` alias for
    ``model`` and errors if the two disagree. Which fields are required is
    enforced by :class:`TrainingJobSpec`, not here: e.g. ``native-checkpoint``
    and ``custom`` require ``checkpoint_path``, ``mace``/``aimnet2`` require
    ``model_id`` or ``checkpoint_path``, and the ``train`` workflow forbids both
    ``checkpoint_path`` and ``model_id``.
    """

    model_config = ConfigDict(extra="allow")

    model: Annotated[
        ModelSource,
        Field(description="Model family or checkpoint source used to start training."),
    ]
    checkpoint_path: Annotated[
        str | None,
        Field(description="Native checkpoint root or model checkpoint file."),
    ] = None
    model_id: Annotated[
        str | None,
        Field(description="Model identifier for supported model wrappers."),
    ] = None
    checkpoint_index: Annotated[
        int,
        Field(description="Native checkpoint index; -1 means latest."),
    ] = -1
    compile_model: Annotated[
        bool | None,
        Field(description="Whether the model wrapper should compile the model."),
    ] = None
    use_original_loss: Annotated[
        bool,
        Field(description="Reuse source checkpoint loss metadata when available."),
    ] = False
    use_original_opt_class: Annotated[
        bool,
        Field(description="Reuse source checkpoint optimizer classes when available."),
    ] = False
    optimizer_lr: Annotated[
        float | None,
        Field(description="Learning rate applied to reused optimizer configs."),
    ] = 1e-5
    hooks: list[RuntimeHookSpec] = Field(
        default_factory=list,
        description=(
            "Runtime hooks serialized as BaseSpec JSON objects with cls_path, "
            "timestamp, and constructor keyword fields. These are attached by "
            "execution code, not stored in "
            "FineTuningStrategy.to_spec_dict()."
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def _accept_endpoint_alias(cls, data: Any) -> Any:
        """Accept older specs that used ``source.endpoint``."""
        if isinstance(data, Mapping) and "endpoint" in data:
            normalized = dict(data)
            endpoint = normalized.pop("endpoint")
            if "model" in normalized and normalized["model"] != endpoint:
                raise ValueError(
                    "source.model and deprecated source.endpoint disagree."
                )
            normalized.setdefault("model", endpoint)
            return normalized
        return data


def _normalize_runtime_hook_spec(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize CLI runtime hook spec values before spec loading."""
    spec = dict(raw)
    stage = spec.get("stage")
    if isinstance(stage, int):
        try:
            spec["stage"] = TrainingStage(stage)
        except ValueError as exc:
            raise ValueError(f"unknown TrainingStage value {stage!r}") from exc
    elif isinstance(stage, str):
        try:
            spec["stage"] = TrainingStage[stage]
        except KeyError as exc:
            raise ValueError(f"unknown TrainingStage name {stage!r}") from exc
    return spec


def build_checked_hook(spec: HookSpec) -> Any:
    """Build a hook spec and verify it satisfies the runtime hook protocols."""
    payload = spec.model_dump(mode="json")
    try:
        hook_spec = create_model_spec_from_json(payload)
    except ValueError as exc:
        raise ValueError(f"spec is not a valid BaseSpec JSON object: {exc}") from exc
    try:
        hook = hook_spec.build()
    except Exception as exc:
        raise ValueError(
            f"spec did not instantiate a hook from {spec.cls_path!r}: {exc}"
        ) from exc
    if not isinstance(hook, (Hook, CheckpointableHook, TrainingUpdateHook)):
        raise ValueError(
            f"spec built {type(hook).__name__}, which does not satisfy "
            "Hook, CheckpointableHook, or TrainingUpdateHook."
        )
    return hook


class DatasetSpec(BaseModel):
    """Training (and optional validation) dataset intent for a job.

    ``DatasetSpec`` is the ``dataset`` member of :class:`TrainingJobSpec`. It
    records one training source via ``path`` or several via ``paths``, the
    loader ``format``, an optional ``validation_path``, and a requested
    ``batch_size``. During ``spec run`` these become a :class:`Dataset` or
    :class:`MultiDataset` wrapped in a :class:`DataLoader`.

    Examples
    --------
    A single-dataset intent::

        DatasetSpec(path="data/domain.zarr", validation_path="data/val.zarr")

    A multi-dataset intent (normalizes to the MultiDataset format)::

        DatasetSpec(paths=["data/a.zarr", "data/b.zarr"])

    Notes
    -----
    An ``after`` validator requires at least one of ``path``/``paths`` and
    normalizes them: a single-element ``paths`` populates ``path``; more than
    one path clears ``path`` and upgrades the default ``alchemi-zarr`` format to
    ``alchemi-zarr-multidataset``. Setting ``path`` to a value absent from a
    populated ``paths`` list is rejected.
    """

    model_config = ConfigDict(extra="allow")

    path: Annotated[
        str | None,
        Field(description="Single training dataset path or URI."),
    ] = None
    paths: list[str] = Field(
        default_factory=list,
        description=(
            "Training dataset paths or URIs. More than one path indicates a "
            "MultiDataset-backed workflow."
        ),
    )
    format: Annotated[
        str,
        Field(
            description=(
                "Loader family the CLI opens the dataset with: 'alchemi-zarr' "
                "reads the one store at path, and 'alchemi-zarr-multidataset' "
                "concatenates the stores in paths into a MultiDataset."
            )
        ),
    ] = "alchemi-zarr"
    validation_path: Annotated[
        str | None,
        Field(description="Optional validation dataset path or URI."),
    ] = None
    batch_size: Annotated[
        int | None,
        Field(ge=1, description="Requested training batch size."),
    ] = None

    @model_validator(mode="after")
    def _validate_dataset_paths(self) -> Self:
        """Normalize single-dataset and multidataset path intent."""
        if self.path and self.paths and self.path not in self.paths:
            raise ValueError(
                "dataset.path must match one of dataset.paths when both are set."
            )
        if not self.path and not self.paths:
            raise ValueError("dataset requires path or paths.")
        if len(self.paths) == 1 and self.path is None:
            self.path = self.paths[0]
        if len(self.paths) > 1:
            self.path = None
            if self.format == "alchemi-zarr":
                self.format = "alchemi-zarr-multidataset"
        return self


class OutputSpec(BaseModel):
    """Filesystem destinations for a job's artifacts.

    ``OutputSpec`` is the ``output`` member of :class:`TrainingJobSpec`. The
    required ``run_dir`` holds logs and artifacts; ``checkpoint_dir`` is where
    restartable training checkpoints are written, and ``report_path`` optionally
    persists intent reports. These are recorded intent: writing restart
    checkpoints still requires a ``CheckpointHook`` in ``source.hooks`` (the
    report surfaces a warning when ``checkpoint_dir`` is set without one).

    Examples
    --------
    ::

        OutputSpec(
            run_dir="runs/mace-ft",
            checkpoint_dir="runs/mace-ft/checkpoints",
        )
    """

    model_config = ConfigDict(extra="allow")

    run_dir: Annotated[str, Field(description="Run directory for logs and artifacts.")]
    checkpoint_dir: Annotated[
        str | None,
        Field(description="Directory for restartable training checkpoints."),
    ] = None
    report_path: Annotated[
        str | None,
        Field(description="Optional path for saved intent reports."),
    ] = None


class ValidationSpec(BaseModel):
    """How often CLI-owned validation runs during a job.

    ``ValidationSpec`` is the optional ``validation`` member of
    :class:`TrainingJobSpec`. It records only the cadence, either
    ``every_n_epochs`` or ``every_n_steps``; the validation dataset itself comes
    from ``DatasetSpec.validation_path``. At run time this cadence and that path
    build the strategy's :class:`ValidationConfig`.

    Examples
    --------
    Validate once per epoch::

        ValidationSpec(every_n_epochs=1)

    Notes
    -----
    ``every_n_epochs`` and ``every_n_steps`` are mutually exclusive (a validator
    rejects both). Supplying a ``ValidationSpec`` requires
    ``dataset.validation_path`` to be set, enforced by :class:`TrainingJobSpec`.
    """

    model_config = ConfigDict(extra="forbid")

    every_n_epochs: Annotated[
        int | None,
        Field(default=None, ge=1, description="Epoch cadence for validation."),
    ] = None
    every_n_steps: Annotated[
        int | None,
        Field(default=None, ge=1, description="Step cadence for validation."),
    ] = None

    @model_validator(mode="after")
    def _validate_single_cadence(self) -> Self:
        """Require at most one validation cadence field."""
        if self.every_n_epochs is not None and self.every_n_steps is not None:
            raise ValueError(
                "validation accepts only one of every_n_epochs or every_n_steps."
            )
        return self


class DataJobSpec(Protocol):
    """Job spec shape the dataset and validation builders read.

    :class:`~nvalchemi.training.cli.TrainingJobSpec` and the distillation
    recipe both satisfy it, so one builder serves both CLIs.
    """

    @property
    def dataset(self) -> DatasetSpec:
        """Training and validation dataset intent."""
        ...

    @property
    def validation(self) -> ValidationSpec | None:
        """Validation cadence, or ``None`` for the default."""
        ...

    @property
    def strategy(self) -> Mapping[str, Any]:
        """Serialized strategy bundle whose ``devices`` places the datasets."""
        ...


def _attach_options(function: Any, options: list[Any]) -> Any:
    """Attach *options* to *function* in the order they are listed."""
    for option in reversed(options):
        function = option(function)
    return function


def common_prefetch_options(function: Any) -> Any:
    """Attach the dataloader prefetch options a command forwards to ``DataLoader``.

    The four options — ``--prefetch-factor``, ``--num-streams``,
    ``--pin-memory``, and ``--use-streams/--no-use-streams`` — are the ones a
    loader still takes when it scores rather than trains. A training loader
    takes :func:`common_loader_options`, which adds to them.
    """
    return _attach_options(
        function,
        [
            click.option(
                "--prefetch-factor",
                type=int,
                default=2,
                show_default=True,
                help="Number of emitted batches to fuse per backend read.",
            ),
            click.option(
                "--num-streams",
                type=int,
                default=4,
                show_default=True,
                help="CUDA stream count for dataloader prefetching.",
            ),
            click.option(
                "--pin-memory", is_flag=True, help="Request pinned-memory reads."
            ),
            click.option(
                "--use-streams/--no-use-streams",
                default=True,
                show_default=True,
                help="Enable CUDA stream prefetching when CUDA is available.",
            ),
        ],
    )


def common_loader_options(function: Any) -> Any:
    """Attach the training dataloader options ``spec run`` and ``spec resume`` share.

    The options are ``--batch-size``, ``--shuffle/--no-shuffle``, and
    ``--drop-last``, followed by those of :func:`common_prefetch_options`. The
    parameter names match the keyword arguments of the CLI's dataloader
    builder.
    """
    return _attach_options(
        common_prefetch_options(function),
        [
            click.option(
                "--batch-size",
                type=int,
                default=None,
                help="Override dataset.batch_size.",
            ),
            click.option(
                "--shuffle/--no-shuffle",
                default=True,
                show_default=True,
                help=(
                    "Shuffle the training dataloader when no distributed "
                    "sampler replaces it."
                ),
            ),
            click.option(
                "--drop-last", is_flag=True, help="Drop the final incomplete batch."
            ),
        ],
    )


def common_validation_options(function: Any) -> Any:
    """Attach the validation store and cadence options that override a run's spec."""
    return _attach_options(
        function,
        [
            click.option(
                "--validation-dataset",
                "validation_path",
                default=None,
                help="Validation dataset path or URI for this run.",
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
        ],
    )


def path_exists(value: str) -> bool:
    """Return whether a local path exists, skipping URI-like references."""
    if "://" in value:
        return True
    return Path(value).expanduser().exists()


def resolve_distributed_enabled(requested: bool | None) -> bool:
    """Resolve whether CLI execution should attach distributed runtime hooks."""
    if requested is not None:
        return requested
    return int(os.environ.get("WORLD_SIZE", "1")) > 1


def setup_distributed_manager(enabled: bool) -> Any | None:
    """Initialize and return the distributed manager when requested."""
    if not enabled:
        return None
    from nvalchemi.distributed import DistributedManager

    if not DistributedManager.is_initialized():
        DistributedManager.initialize()
    return DistributedManager()


def build_supported_source_model(source: SourceSpec, *, device: Any) -> Any:
    """Build a supported model wrapper from source intent."""
    checkpoint = source.checkpoint_path or source.model_id
    if checkpoint is None:
        raise click.ClickException(f"{source.model} execution requires a source model.")
    compile_model = bool(source.compile_model)
    if source.model == "mace":
        from nvalchemi.models.mace import MACEWrapper

        mace_options = MaceSourceOptions.from_source(source)
        return MACEWrapper.from_checkpoint(
            checkpoint,
            device=torch.device(device),
            compile_model=compile_model,
            atomic_energies=mace_options.atomic_energies,
            atomic_energies_path=mace_options.atomic_energies_path,
        )
    if source.model == "aimnet2":
        from nvalchemi.models.aimnet2 import AIMNet2Wrapper

        return AIMNet2Wrapper.from_checkpoint(
            checkpoint,
            device=torch.device(device),
            compile_model=compile_model,
        )
    raise click.ClickException(f"Unsupported source model {source.model!r}.")


def write_or_print(payload: BaseModel | Mapping[str, Any], output: Path | None) -> None:
    """Write a JSON payload to a file or stdout."""
    data = (
        payload.model_dump(mode="json", exclude_none=True)
        if isinstance(payload, BaseModel)
        else payload
    )
    text = json.dumps(data, indent=2) + "\n"
    if output is None:
        click.echo(text, nl=False)
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text)
    console.print(f"[green]Wrote[/] {output}")


def hook_spec_is(hook: RuntimeHookSpec, base: type) -> bool:
    """Return whether the hook class *hook* names is *base* or a subclass of it.

    The class path is known to import, because a runtime hook spec is
    trial-built when the enclosing spec is validated.
    """
    return issubclass(_import_cls(hook.spec.cls_path), base)


def build_runtime_hooks(
    hook_specs: Sequence[RuntimeHookSpec], *, enable_ddp: bool, ddp_backend: str | None
) -> list[Any]:
    """Build the runtime hooks a spec declares, one per requested stage.

    Parameters
    ----------
    hook_specs : Sequence[RuntimeHookSpec]
        Hook entries of the job's source, in declaration order.
    enable_ddp : bool
        Whether to put a :class:`~nvalchemi.training.DDPHook` ahead of them.
    ddp_backend : str | None
        Process-group backend forwarded to that hook.

    Returns
    -------
    list[Any]
        The hooks, with a stage override applied to each copy it names.
    """
    hooks: list[Any] = []
    if enable_ddp:
        hooks.append(DDPHook(backend=ddp_backend))
    for hook_spec in hook_specs:
        stages = hook_spec.stage_values()
        if not stages:
            hooks.append(build_checked_hook(hook_spec.spec))
            continue
        for stage in stages:
            hook = build_checked_hook(hook_spec.spec)
            hook.stage = stage
            hooks.append(hook)
    return hooks


def validate_pretrained_source(field: str, source: SourceSpec) -> None:
    """Check that *source* carries what the wrapper loading a pretrained model needs.

    Parameters
    ----------
    field : str
        Dotted name of the source inside the enclosing spec, for the message.
    source : SourceSpec
        Source to check.

    Raises
    ------
    ValueError
        If a ``mace`` block rides on a source of another family, if a
        native checkpoint names no path, or if the source names neither a
        model id nor a checkpoint path.
    """
    if source.model != "mace" and (source.model_extra or {}).get("mace") is not None:
        raise ValueError(
            f"{field}.mace options are only valid when {field}.model='mace'."
        )
    if source.model == "mace":
        MaceSourceOptions.from_source(source)
    if source.model == "native-checkpoint" and not source.checkpoint_path:
        raise ValueError(f"native-checkpoint sources require {field}.checkpoint_path.")
    if not (source.model_id or source.checkpoint_path):
        raise ValueError(
            f"{source.model} sources require {field}.model_id or "
            f"{field}.checkpoint_path."
        )


def primary_strategy_device(job: DataJobSpec) -> torch.device:
    """Return the first strategy device as a torch device."""
    devices = strategy_spec._devices_from_spec(job.strategy["devices"])
    if not devices:
        raise click.ClickException("strategy.devices must contain at least one device.")
    return devices[0]


def dataset_device(job: DataJobSpec, distributed_manager: Any | None) -> Any:
    """Return the device used for CLI-constructed datasets."""
    if distributed_manager is not None:
        return distributed_manager.device
    return primary_strategy_device(job)


def build_dataloader(
    job: DataJobSpec,
    stack: ExitStack,
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
) -> Any:
    """Build a DataLoader declared by a CLI job spec."""
    from nvalchemi.data.datapipes import (
        AtomicDataZarrReader,
        DataLoader,
        Dataset,
        MultiDataset,
    )

    resolved_paths = (
        paths
        if paths is not None
        else list(job.dataset.paths) or ([job.dataset.path] if job.dataset.path else [])
    )
    if not resolved_paths:
        raise click.ClickException("dataset requires at least one path before run.")
    if job.dataset.format not in get_args(DatasetFormat):
        raise click.ClickException(
            f"Unsupported dataset.format {job.dataset.format!r}; "
            f"supported formats: {', '.join(get_args(DatasetFormat))}."
        )
    datasets = [
        Dataset(stack.enter_context(AtomicDataZarrReader(path)), device=device)
        for path in resolved_paths
    ]
    dataset = datasets[0] if len(datasets) == 1 else MultiDataset(*datasets)
    return DataLoader(
        dataset,
        batch_size=batch_size or job.dataset.batch_size or 1,
        shuffle=shuffle,
        drop_last=drop_last,
        prefetch_factor=prefetch_factor,
        num_streams=num_streams,
        use_streams=use_streams,
        pin_memory=pin_memory,
    )


def _resolve_validation_cadence(
    job: DataJobSpec,
    *,
    every_n_epochs: int | None,
    every_n_steps: int | None,
) -> tuple[int | None, int | None]:
    """Resolve validation cadence from CLI overrides or the job spec."""
    if every_n_epochs is not None and every_n_steps is not None:
        raise click.ClickException(
            "Use only one of --validation-every-epochs or --validation-every-steps."
        )
    if every_n_epochs is not None or every_n_steps is not None:
        return every_n_epochs, every_n_steps
    if job.validation is not None:
        return job.validation.every_n_epochs, job.validation.every_n_steps
    return 1, None


def build_validation_config(
    job: DataJobSpec,
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
) -> ValidationConfig | None:
    """Build the CLI's validation configuration, or return ``None`` without a store."""
    resolved_path = validation_path or job.dataset.validation_path
    if resolved_path is None:
        return None
    every_n_epochs, every_n_steps = _resolve_validation_cadence(
        job,
        every_n_epochs=validation_every_epochs,
        every_n_steps=validation_every_steps,
    )
    validation_data = build_dataloader(
        job,
        stack,
        device=device,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        prefetch_factor=prefetch_factor,
        num_streams=num_streams,
        use_streams=use_streams,
        pin_memory=pin_memory,
        paths=[resolved_path],
    )
    return ValidationConfig(
        validation_data=validation_data,
        every_n_epochs=every_n_epochs,
        every_n_steps=every_n_steps,
    )


def restart_map_location(
    distributed_manager: Any | None, map_location: str | None
) -> str | None:
    """Return the device a restarting rank loads its checkpoint onto.

    Parameters
    ----------
    distributed_manager : Any | None
        Manager attached to the resumed run, or ``None`` for a single process.
    map_location : str | None
        Device the caller asked for, or ``None`` to take the rank's own under
        a multi-rank launch.

    Returns
    -------
    str | None
        *map_location* when given; otherwise this rank's device under a
        multi-rank launch, and ``None`` for a single process.

    Notes
    -----
    A checkpoint records the device rank zero was pinned to. Loading every
    rank's weights onto that device would stage the whole world's restore
    through one accelerator's memory, so the rank's own device is the default.
    Either way, the restored strategy's ``devices`` decide where the run
    continues.
    """
    if distributed_manager is None or get_world_size(distributed_manager) <= 1:
        return map_location
    return map_location if map_location is not None else str(distributed_manager.device)


def _budget_label(num_steps: int | None, num_epochs: int | None) -> str:
    """Return the budget *num_steps* or *num_epochs* names, for a message."""
    return f"{num_epochs!r} epochs" if num_steps is None else f"{num_steps!r} steps"


def apply_resume_budget(
    job_strategy: Mapping[str, Any],
    strategy: TrainingStrategy,
    *,
    budget: ResumeBudget = "checkpoint",
) -> None:
    """Size a restored run by the checkpoint's stored budget or by the spec's.

    A checkpoint's spec records the budget the run started with, while the
    job spec is what the user edits, so the two can disagree. With
    ``budget="checkpoint"`` the restored ``num_steps``/``num_epochs`` are
    kept, and rank zero reports a spec that says otherwise. With
    ``budget="recipe"`` the spec's budget is applied and the replacement
    reported. Either way, a spec is refused, with both numbers, when it would
    cut the run below the steps or epochs already completed, or when it sizes
    the run in epochs where the checkpoint counts steps, or the reverse. Such a
    spec describes a different run from the one being resumed.

    Parameters
    ----------
    job_strategy : Mapping[str, Any]
        The job spec's strategy bundle, read for ``num_steps`` and
        ``num_epochs``.
    strategy : TrainingStrategy
        The restored strategy, resized in place under ``budget="recipe"``.
    budget : {"checkpoint", "recipe"}, optional
        Whose budget sizes the continued run. Default ``"checkpoint"``.

    Raises
    ------
    click.UsageError
        If the spec switches between steps and epochs against the
        checkpoint, or its budget is below what the checkpoint has completed.
    """
    num_steps = job_strategy.get("num_steps")
    num_epochs = job_strategy.get("num_epochs")
    stored = (strategy.num_steps, strategy.num_epochs)
    recipe_label = _budget_label(num_steps, num_epochs)
    stored_label = _budget_label(*stored)
    if (num_steps is None) != (stored[0] is None):
        raise click.UsageError(
            f"the recipe sizes the run at {recipe_label} while the checkpoint "
            f"recorded {stored_label}; a resumed run keeps the unit it started "
            "in. Size the recipe in the checkpoint's unit, or start a fresh run."
        )
    completed = strategy.step_count if num_steps is not None else strategy.epoch_count
    unit = "steps" if num_steps is not None else "epochs"
    if (num_steps if num_steps is not None else num_epochs) < completed:
        raise click.UsageError(
            f"the recipe sizes the run at {recipe_label}, below the {completed!r} "
            f"{unit} the checkpoint has already completed, so it describes a "
            f"different run; the checkpoint recorded {stored_label}. Raise the "
            f"recipe's budget to at least {completed!r} {unit}, or resume from an "
            "earlier --checkpoint-index."
        )
    if (num_steps, num_epochs) == stored:
        return
    if get_rank(strategy.distributed_manager) == 0:
        click.echo(
            f"recipe sizes the run at {recipe_label}, replacing the {stored_label} "
            "the checkpoint recorded (--budget recipe)."
            if budget == "recipe"
            else f"recipe sizes the run at {recipe_label}, but the {stored_label} "
            "the checkpoint recorded is kept (--budget checkpoint); pass --budget "
            "recipe to apply the recipe's."
        )
    if budget == "recipe":
        strategy.num_steps = num_steps
        strategy.num_epochs = num_epochs
