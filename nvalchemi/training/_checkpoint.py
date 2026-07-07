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
"""Multi-component, manifest-based checkpoint layer.

This module saves and loads checkpoints for multiple named models,
optimizers, and schedulers without relying on :mod:`pickle`. A top-level
``manifest.json`` coordinates all components and their associations.

Layout
------
A single call to :func:`save_checkpoint` writes::

    {root_folder}/
      manifest.json
      models/{name}/
        spec.json
        checkpoints/{N}.pt
      optimizers/{name}/          # optional
        spec.json
        checkpoints/{N}.pt
      schedulers/{name}/          # optional
        spec.json
        checkpoints/{N}.pt

The ``manifest.json`` records which components are present, the latest
checkpoint index, and optional associations that wire optimizers to models
and schedulers to optimizers::

    {
      "checkpoint_index": 0,
      "models": ["student", "teacher"],
      "optimizers": ["student_opt"],
      "schedulers": ["student_sched"],
      "associations": {
        "student": {
          "optimizers": ["student_opt"],
          "schedulers": ["student_sched"]
        }
      }
    }

The ``associations`` key specifies connectivity between models and
their respective optimizer(s) and LR scheduler(s). This can be explicitly
provided by the user, or automatically inferred by matching parameters
with optimizers/LR schedulers.

Examples
--------
Single model::

    save_checkpoint("runs/exp1", models={"main": (model, spec)})
    result = load_checkpoint("runs/exp1")
    model, spec = result.models["main"]

Knowledge distillation (two models + optimizer + scheduler)::

    save_checkpoint(
        "runs/kd",
        models={"student": (student, s_spec), "teacher": (teacher, t_spec)},
        optimizers={"s_opt": (optimizer, opt_spec)},
        schedulers={"s_sched": (scheduler, sched_spec)},
        # associations can be inferred automatically from param_groups
    )
    result = load_checkpoint("runs/kd")
    student, _ = result.models["student"]
"""

from __future__ import annotations

import itertools
import json
import warnings
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Annotated, Any

import torch
import torch.nn as nn
from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, PlainSerializer

from nvalchemi.hooks._protocol import CheckpointableHook
from nvalchemi.training._spec import (
    BaseSpec,
    create_model_spec,
    create_model_spec_from_json,
)
from nvalchemi.training.distributed import get_world_size

CheckpointValidator = Callable[[str, Mapping[str, Any], Mapping[str, Any]], None]
"""Callable used to validate a loaded model entry.

Validators receive ``(model_name, model_entry, loaded_checkpoint)`` and should
raise an exception with an actionable message when compatibility checks fail.
"""

# ---------------------------------------------------------------------------
# Dual-mode field helpers
# ---------------------------------------------------------------------------


def _component_before(v: Any) -> dict[str, Any]:
    """Accept ``list[str]`` (from JSON) or ``dict`` (from code) for component fields."""
    if isinstance(v, list):
        # From disk: list of names → placeholder dict (values populated later)
        return {name: None for name in v}
    return v


def _component_serialize(d: dict[str, Any]) -> list[str]:
    """Serialize a component dict to a sorted list of its keys."""
    return sorted(d.keys())


def _is_fsdp_wrapped(module: nn.Module) -> bool:
    """Return whether ``module`` is wrapped by FSDP or FSDP2."""
    fsdp_types: list[type[nn.Module]] = []
    try:
        from torch.distributed.fsdp import FullyShardedDataParallel

        fsdp_types.append(FullyShardedDataParallel)
    except (ImportError, AttributeError):
        pass
    try:
        from torch.distributed._composable.fsdp import FSDPModule

        fsdp_types.append(FSDPModule)
    except (ImportError, AttributeError):
        pass
    return bool(fsdp_types) and isinstance(module, tuple(fsdp_types))


def _checkpoint_model(module: nn.Module) -> nn.Module:
    """Return a model suitable for native checkpoint state and spec extraction."""
    if isinstance(module, torch.nn.parallel.DistributedDataParallel):
        return module.module
    if _is_fsdp_wrapped(module):
        recipe_url = (
            "https://docs.pytorch.org/tutorials/recipes/"
            "distributed_checkpoint_recipe.html"
        )
        raise NotImplementedError(
            "Native nvalchemi checkpoints do not yet support FSDP/FSDP2-wrapped "
            "models. Use torch.distributed.checkpoint with PyTorch's distributed "
            f"checkpoint recipe instead: {recipe_url}"
        )
    return module


def _checkpoint_model_components(
    models: Mapping[str, tuple[nn.Module, BaseSpec]],
) -> dict[str, tuple[nn.Module, BaseSpec]]:
    """Unwrap supported distributed model wrappers before checkpointing."""
    return {
        name: (_checkpoint_model(module), spec)
        for name, (module, spec) in models.items()
    }


# ---------------------------------------------------------------------------
# Manifest schema + runtime container (unified)
# ---------------------------------------------------------------------------

_SCHEMA_VERSION = 1
"""Current manifest schema version.  Bump when manifest structure changes."""

_STRATEGY_FILENAME = "strategy.json"
"""File containing strategy recipe and runtime counters for native checkpoints."""

_STRATEGY_CHECKPOINT_DIR = Path("strategy") / "checkpoints"
"""Directory containing per-index strategy checkpoint metadata."""

_HOOK_CHECKPOINT_DIR = Path("hooks") / "checkpoints"
"""Directory containing per-index runtime hook state."""

_SCHEDULER_OPTIMIZERS_KEY = "scheduler_optimizers"
"""Association key mapping scheduler component names to optimizer names."""

_OPTIMIZER_PARAMETER_NAMES_KEY = "optimizer_parameter_names"
"""Association key mapping optimizer component names to parameter names."""

# Type aliases for the runtime dict shapes
_ModelDict = dict[str, tuple[nn.Module, BaseSpec] | None]
_OptimizerDict = dict[str, tuple[torch.optim.Optimizer, BaseSpec] | None]
_SchedulerDict = dict[str, tuple[torch.optim.lr_scheduler.LRScheduler, BaseSpec] | None]
_Associations = dict[str, dict[str, Any]]


class CheckpointManifest(BaseModel):
    """Unified checkpoint manifest and runtime container.

    This Pydantic model serves a dual role:

    1. **On-disk schema** — ``manifest.json`` stores component names as
       sorted string lists together with metadata and associations.
    2. **Runtime container** — after :func:`load_checkpoint` hydrates the
       components, the same instance carries live ``(object, spec)`` tuples.

    The ``models``, ``optimizers``, and ``schedulers`` fields accept
    either a ``list[str]`` (from JSON) or a ``dict[str, tuple]`` (from
    code).  Serialization always produces sorted name lists via
    :class:`~pydantic.PlainSerializer`.

    Attributes
    ----------
    schema_version
        Schema version. Defaults to the current ``_SCHEMA_VERSION``.
        When manifest structure changes, bump ``_SCHEMA_VERSION`` and
        add a migration step in :meth:`read`.
    checkpoint_index
        The latest checkpoint index written.
    models
        Component dict keyed by name. At runtime each value is a
        ``(nn.Module, BaseSpec)`` tuple; on disk, serialized as a
        sorted ``list[str]`` of names.
    optimizers
        Same dual-mode dict for optimizers (empty by default).
    schedulers
        Same dual-mode dict for schedulers (empty by default).
    associations
        Model-centric linkage: maps a model name to
        ``{"optimizers": [...], "schedulers": [...]}``.

    Examples
    --------
    >>> manifest = CheckpointManifest(
    ...     checkpoint_index=0, models={"main": None},
    ... )
    >>> manifest.model_dump()["models"]
    ['main']
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    schema_version: Annotated[
        int, Field(default=_SCHEMA_VERSION, description="Manifest schema version.")
    ]
    checkpoint_index: Annotated[
        int, Field(description="Latest checkpoint index written.")
    ]
    models: Annotated[
        _ModelDict,
        BeforeValidator(_component_before),
        PlainSerializer(_component_serialize, return_type=list[str]),
        Field(description="Model components keyed by name."),
    ]
    optimizers: Annotated[
        _OptimizerDict,
        BeforeValidator(_component_before),
        PlainSerializer(_component_serialize, return_type=list[str]),
        Field(default_factory=dict, description="Optimizer components keyed by name."),
    ]
    schedulers: Annotated[
        _SchedulerDict,
        BeforeValidator(_component_before),
        PlainSerializer(_component_serialize, return_type=list[str]),
        Field(default_factory=dict, description="Scheduler components keyed by name."),
    ]
    associations: Annotated[
        _Associations,
        Field(
            default_factory=dict,
            description="Model-centric linkage to optimizers/schedulers.",
        ),
    ]

    @staticmethod
    def _migrate(raw: dict[str, Any]) -> dict[str, Any]:
        """Migrate an older manifest dict to the current schema version.

        Parameters
        ----------
        raw
            Parsed ``manifest.json`` content.

        Returns
        -------
        dict[str, Any]
            Dict conforming to the current ``_SCHEMA_VERSION``, ready
            for :meth:`pydantic.BaseModel.model_validate`.

        Raises
        ------
        ValueError
            If the manifest's schema version is newer than supported.
        """
        version = raw.get("schema_version", 0)
        if version > _SCHEMA_VERSION:
            raise ValueError(
                f"Checkpoint schema version {version} is newer than supported "
                f"({_SCHEMA_VERSION}). Upgrade nvalchemi to load this checkpoint."
            )
        # Future migrations chain here:
        # if version < 1:
        #     raw = _migrate_v0_to_v1(raw)
        raw["schema_version"] = _SCHEMA_VERSION
        return raw

    @classmethod
    def read(cls, root: Path) -> CheckpointManifest:
        """Read, migrate, and validate ``manifest.json`` from *root*.

        Parameters
        ----------
        root
            Checkpoint root directory containing ``manifest.json``.

        Returns
        -------
        CheckpointManifest
            Validated manifest instance. Component dicts contain
            placeholder ``None`` values until hydrated by
            :func:`load_checkpoint`.

        Raises
        ------
        FileNotFoundError
            If ``manifest.json`` does not exist.
        ValueError
            If the manifest's schema version is newer than supported.
        pydantic.ValidationError
            If the manifest JSON does not conform to the schema.
        """
        manifest_path = root / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"No manifest.json found in {root}. Use save_checkpoint to "
                f"create a checkpoint first."
            )
        raw = json.loads(manifest_path.read_text())
        migrated = cls._migrate(raw)
        return cls.model_validate(migrated)

    def write(self, root: Path) -> None:
        """Write this manifest to ``{root}/manifest.json``.

        Parameters
        ----------
        root
            Checkpoint root directory.
        """
        (root / "manifest.json").write_text(self.model_dump_json(indent=2))


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _ckpt_indices(ckpt_dir: Path) -> list[int]:
    """Return sorted integer stems from ``*.pt`` files in *ckpt_dir*."""
    return sorted(int(p.stem) for p in ckpt_dir.glob("*.pt") if p.stem.isdigit())


def _without_spec_timestamps(value: Any) -> Any:
    """Return JSON-like *value* with BaseSpec timestamps removed recursively."""
    if isinstance(value, dict):
        return {
            key: _without_spec_timestamps(item)
            for key, item in value.items()
            if not (key == "timestamp" and "cls_path" in value)
        }
    if isinstance(value, list):
        return [_without_spec_timestamps(item) for item in value]
    return value


def _check_spec_consistency(spec_path: Path, spec: BaseSpec) -> None:
    """Write *spec* to *spec_path* on first call; raise on mismatch thereafter.

    Parameters
    ----------
    spec_path
        Path to the ``spec.json`` file.
    spec
        The spec to write or compare against the existing file.

    Raises
    ------
    ValueError
        If the existing ``spec.json`` disagrees with *spec* on any field
        other than ``timestamp``.
    """
    spec_json = spec.model_dump_json(indent=2)
    if spec_path.exists():
        existing = _without_spec_timestamps(json.loads(spec_path.read_text()))
        new_spec = _without_spec_timestamps(json.loads(spec_json))
        if existing != new_spec:
            diffs = sorted(
                k
                for k in set(existing) | set(new_spec)
                if existing.get(k) != new_spec.get(k)
            )
            preview = ", ".join(
                f"{k}: {existing.get(k)!r} -> {new_spec.get(k)!r}" for k in diffs[:3]
            )
            suffix = f" (+{len(diffs) - 3} more)" if len(diffs) > 3 else ""
            raise ValueError(
                f"spec.json at {spec_path} disagrees with the spec being "
                f"saved. Differing fields: {preview}{suffix}."
            )
    else:
        spec_path.write_text(spec_json)


def _save_component(
    root: Path,
    category: str,
    name: str,
    state_dict: dict[str, Any],
    spec: BaseSpec,
    checkpoint_index: int,
) -> None:
    """Write *spec* and *state_dict* under ``root/category/name/``."""
    comp_dir = root / category / name
    ckpt_dir = comp_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    _check_spec_consistency(comp_dir / "spec.json", spec)
    torch.save(state_dict, ckpt_dir / f"{checkpoint_index}.pt")


def _snapshot_state_value(value: Any) -> Any:
    """Return a CPU copy of tensors nested inside a state-dict value."""
    if isinstance(value, torch.Tensor):
        return value.detach().to(device="cpu", copy=True)
    if isinstance(value, Mapping):
        return {key: _snapshot_state_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_snapshot_state_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_snapshot_state_value(item) for item in value)
    return value


def _snapshot_state_dict(state_dict: Mapping[str, Any]) -> dict[str, Any]:
    """Return a CPU-only state dict detached from live training objects."""
    return {key: _snapshot_state_value(value) for key, value in state_dict.items()}


def _snapshot_components(
    components: Mapping[str, tuple[Any, BaseSpec]],
) -> dict[str, tuple[dict[str, Any], BaseSpec]]:
    """Capture component state dicts and specs for asynchronous writing."""
    return {
        name: (_snapshot_state_dict(component.state_dict()), spec)
        for name, (component, spec) in components.items()
    }


def _hook_state_key(hook: object, occurrence: int) -> str:
    """Return the stable class-occurrence key used for hook state matching."""
    return f"{type(hook).__module__}.{type(hook).__qualname__}:{occurrence}"


def _iter_checkpointable_hooks(hooks: Iterable[object]) -> Iterator[CheckpointableHook]:
    """Yield hooks that explicitly opt into checkpointed runtime state."""
    for hook in hooks:
        children = getattr(hook, "_hooks", None)
        if isinstance(children, Sequence) and not isinstance(children, (str, bytes)):
            yield from _iter_checkpointable_hooks(children)
        if isinstance(hook, CheckpointableHook):
            yield hook


def _snapshot_hook_states(strategy: Any) -> dict[str, dict[str, Any]]:
    """Capture checkpointable runtime hook state detached from live tensors."""
    states: dict[str, dict[str, Any]] = {}
    occurrences: dict[str, int] = {}
    for hook in _iter_checkpointable_hooks(strategy.hooks):
        class_name = f"{type(hook).__module__}.{type(hook).__qualname__}"
        occurrence = occurrences.get(class_name, 0)
        occurrences[class_name] = occurrence + 1
        states[_hook_state_key(hook, occurrence)] = _snapshot_state_dict(
            hook.state_dict()
        )
    return states


def _hook_state_path(root: Path, checkpoint_index: int) -> Path:
    """Return the hook-state checkpoint path for ``checkpoint_index``."""
    return root / _HOOK_CHECKPOINT_DIR / f"{checkpoint_index}.pt"


def _save_hook_states(
    root: Path,
    hook_states: Mapping[str, Mapping[str, Any]],
    checkpoint_index: int,
) -> None:
    """Write hook state for a checkpoint when checkpointable hooks are present."""
    if not hook_states:
        return
    path = _hook_state_path(root, checkpoint_index)
    path.parent.mkdir(parents=True, exist_ok=True)
    state_dict = dict(hook_states)
    torch.save(state_dict, path)


def _load_hook_states(
    root: Path,
    strategy: Any,
    checkpoint_index: int,
    *,
    map_location: str | torch.device | None,
) -> None:
    """Restore matching checkpointable hook state into a loaded strategy."""
    path = _hook_state_path(root, checkpoint_index)
    if not path.exists():
        return
    saved_states = torch.load(
        path,
        weights_only=True,
        map_location=map_location,
    )
    occurrences: dict[str, int] = {}
    for hook in _iter_checkpointable_hooks(strategy.hooks):
        class_name = f"{type(hook).__module__}.{type(hook).__qualname__}"
        occurrence = occurrences.get(class_name, 0)
        occurrences[class_name] = occurrence + 1
        state = saved_states.get(_hook_state_key(hook, occurrence))
        if state is not None:
            hook.load_state_dict(state)


def _resolve_checkpoint_index(root: Path, checkpoint_index: int) -> int:
    """Return an explicit checkpoint index, resolving ``-1`` by auto-increment."""
    if checkpoint_index != -1:
        return checkpoint_index
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        prev = CheckpointManifest.read(root)
        return prev.checkpoint_index + 1
    return 0


def _create_checkpoint_snapshot(
    root_folder: Path | str,
    *,
    checkpoint_index: int = -1,
    strategy: Any,
) -> dict[str, Any]:
    """Capture a strategy checkpoint payload detached from live tensors.

    The snapshot is intended for background filesystem writes. It still runs
    on the caller thread and copies tensors to CPU so later training updates
    cannot mutate data while :func:`torch.save` serializes it.
    """
    from nvalchemi.training.strategy import TrainingStrategy

    if not isinstance(strategy, TrainingStrategy):
        raise TypeError(
            "strategy must be a TrainingStrategy instance; got "
            f"{type(strategy).__name__}."
        )
    root = Path(root_folder)
    models, optimizers, schedulers, associations, strategy_metadata = (
        _strategy_components(strategy)
    )
    return {
        "checkpoint_index": _resolve_checkpoint_index(root, checkpoint_index),
        "models": _snapshot_components(models),
        "optimizers": _snapshot_components(optimizers),
        "schedulers": _snapshot_components(schedulers),
        "associations": _copy_associations(associations),
        "strategy_metadata": dict(strategy_metadata),
        "hook_states": _snapshot_hook_states(strategy),
    }


def _write_checkpoint_snapshot(
    root_folder: Path | str, snapshot: Mapping[str, Any]
) -> int:
    """Write a detached checkpoint snapshot to disk."""
    root = Path(root_folder)
    checkpoint_index = int(snapshot["checkpoint_index"])
    models = snapshot["models"]
    optimizers = snapshot["optimizers"]
    schedulers = snapshot["schedulers"]
    associations = snapshot["associations"]
    strategy_metadata = snapshot.get("strategy_metadata")
    hook_states = snapshot.get("hook_states", {})

    for name, (state_dict, spec) in models.items():
        _save_component(
            root,
            "models",
            name,
            state_dict,
            spec,
            checkpoint_index,
        )
    for name, (state_dict, spec) in optimizers.items():
        _save_component(
            root,
            "optimizers",
            name,
            state_dict,
            spec,
            checkpoint_index,
        )
    for name, (state_dict, spec) in schedulers.items():
        _save_component(
            root,
            "schedulers",
            name,
            state_dict,
            spec,
            checkpoint_index,
        )

    manifest = CheckpointManifest(
        checkpoint_index=checkpoint_index,
        models={name: None for name in models},
        optimizers={name: None for name in optimizers},
        schedulers={name: None for name in schedulers},
        associations=associations,
    )
    manifest.write(root)
    _save_hook_states(root, hook_states, checkpoint_index)
    if strategy_metadata is not None:
        _write_strategy_metadata(
            root, strategy_metadata, checkpoint_index=checkpoint_index
        )
    return checkpoint_index


def _assoc_names(assoc: Mapping[str, Any], key: str) -> list[str]:
    """Return an association list field, tolerating older or malformed entries."""
    raw = assoc.get(key, [])
    return list(raw) if isinstance(raw, list) else []


def _assoc_scheduler_optimizers(assoc: Mapping[str, Any]) -> dict[str, str]:
    """Return scheduler-to-optimizer association edges from *assoc*."""
    raw = assoc.get(_SCHEDULER_OPTIMIZERS_KEY, {})
    if not isinstance(raw, Mapping):
        return {}
    return {str(scheduler): str(optimizer) for scheduler, optimizer in raw.items()}


def _assoc_optimizer_parameter_names(
    assoc: Mapping[str, Any],
) -> dict[str, tuple[str, ...]]:
    """Return optimizer-to-parameter-name association edges from *assoc*."""
    raw = assoc.get(_OPTIMIZER_PARAMETER_NAMES_KEY, {})
    if not isinstance(raw, Mapping):
        return {}
    result: dict[str, tuple[str, ...]] = {}
    for optimizer, names in raw.items():
        if isinstance(names, str) or not isinstance(names, Sequence):
            continue
        result[str(optimizer)] = tuple(str(name) for name in names)
    return result


def _copy_associations(associations: Mapping[str, Mapping[str, Any]]) -> _Associations:
    """Return a shallow JSON-like copy of association entries."""
    copied: _Associations = {}
    for model_name, assoc in associations.items():
        entry: dict[str, Any] = {
            "optimizers": _assoc_names(assoc, "optimizers"),
            "schedulers": _assoc_names(assoc, "schedulers"),
        }
        scheduler_optimizers = _assoc_scheduler_optimizers(assoc)
        if scheduler_optimizers:
            entry[_SCHEDULER_OPTIMIZERS_KEY] = scheduler_optimizers
        optimizer_parameter_names = _assoc_optimizer_parameter_names(assoc)
        if optimizer_parameter_names:
            entry[_OPTIMIZER_PARAMETER_NAMES_KEY] = {
                optimizer: list(names)
                for optimizer, names in optimizer_parameter_names.items()
            }
        copied[model_name] = entry
    return copied


def _scheduler_optimizer_edges(
    optimizers: Mapping[str, tuple[torch.optim.Optimizer, BaseSpec]],
    schedulers: Mapping[str, tuple[torch.optim.lr_scheduler.LRScheduler, BaseSpec]],
) -> dict[str, str]:
    """Return scheduler component names keyed to their optimizer component names."""
    edges: dict[str, str] = {}
    for scheduler_name, (scheduler, _) in schedulers.items():
        for optimizer_name, (optimizer, _) in optimizers.items():
            if scheduler.optimizer is optimizer:  # type: ignore[attr-defined]
                edges[scheduler_name] = optimizer_name
                break
    return edges


def _with_scheduler_optimizer_edges(
    associations: Mapping[str, Mapping[str, Any]],
    optimizers: Mapping[str, tuple[torch.optim.Optimizer, BaseSpec]],
    schedulers: Mapping[str, tuple[torch.optim.lr_scheduler.LRScheduler, BaseSpec]],
) -> _Associations:
    """Attach explicit scheduler-to-optimizer edges to model associations."""
    enriched = _copy_associations(associations)
    edges = _scheduler_optimizer_edges(optimizers, schedulers)
    if not edges:
        return enriched

    for assoc in enriched.values():
        optimizer_names = set(_assoc_names(assoc, "optimizers"))
        scheduler_names = set(_assoc_names(assoc, "schedulers"))
        model_edges = {
            scheduler_name: optimizer_name
            for scheduler_name, optimizer_name in edges.items()
            if scheduler_name in scheduler_names and optimizer_name in optimizer_names
        }
        if model_edges:
            assoc[_SCHEDULER_OPTIMIZERS_KEY] = {
                **_assoc_scheduler_optimizers(assoc),
                **model_edges,
            }
    return enriched


def _infer_associations(
    models: dict[str, tuple[nn.Module, BaseSpec]],
    optimizers: dict[str, tuple[torch.optim.Optimizer, BaseSpec]],
    schedulers: dict[str, tuple[torch.optim.lr_scheduler.LRScheduler, BaseSpec]],
) -> _Associations:
    """Infer model-centric associations from optimizer ``param_groups``.

    For each optimizer, collect the ``data_ptr()`` values of every parameter
    in its ``param_groups`` and match against each model's ``parameters()``.
    The optimizer is associated with every model that owns at least one of
    those parameters.

    Schedulers are linked to their optimizer via
    ``scheduler.optimizer is optimizer`` identity checks.

    Parameters
    ----------
    models
        ``{name: (module, spec)}`` mapping.
    optimizers
        ``{name: (optimizer, spec)}`` mapping.
    schedulers
        ``{name: (scheduler, spec)}`` mapping.

    Returns
    -------
    dict[str, dict[str, list[str]]]
        Model-centric associations, e.g.
        ``{"student": {"optimizers": ["s_opt"], "schedulers": ["s_sched"]}}``.
    """
    # Build data_ptr → model_name index
    ptr_to_model: dict[int, str] = {}
    for model_name, (module, _) in models.items():
        for p in module.parameters():
            ptr_to_model[p.data_ptr()] = model_name

    # Map each optimizer to every model that owns at least one parameter
    opt_to_models: dict[str, list[str]] = {}
    for opt_name, (optimizer, _) in optimizers.items():
        matched: dict[str, bool] = {}
        for group in optimizer.param_groups:
            for p in group["params"]:
                owner = ptr_to_model.get(p.data_ptr())
                if owner is not None:
                    matched[owner] = True
        if matched:
            opt_to_models[opt_name] = list(matched)

    # Map each scheduler to its optimizer (identity check)
    sched_to_opt = _scheduler_optimizer_edges(optimizers, schedulers)

    # Build model-centric structure
    assoc: _Associations = {}
    for opt_name, model_names in opt_to_models.items():
        for model_name in model_names:
            assoc.setdefault(model_name, {"optimizers": [], "schedulers": []})
            assoc[model_name]["optimizers"].append(opt_name)
    for sched_name, opt_name in sched_to_opt.items():
        model_names = opt_to_models.get(opt_name, [])
        for model_name in model_names:
            assoc.setdefault(model_name, {"optimizers": [], "schedulers": []})
            assoc[model_name]["schedulers"].append(sched_name)
            scheduler_optimizers = assoc[model_name].setdefault(
                _SCHEDULER_OPTIMIZERS_KEY, {}
            )
            scheduler_optimizers[sched_name] = opt_name

    return assoc


def _find_associated_model_params(
    optimizer_name: str,
    associations: _Associations,
    models: dict[str, tuple[nn.Module, BaseSpec]],
) -> Iterator[torch.nn.Parameter]:
    """Return chained parameters from all models associated with *optimizer_name*."""
    matched: list[str] = []
    for model_name, assoc in associations.items():
        if optimizer_name in _assoc_names(assoc, "optimizers"):
            matched.append(model_name)
    if matched:
        named_params_by_model = {
            model_name: dict(models[model_name][0].named_parameters())
            for model_name in matched
        }
        parameter_names = []
        for model_name in matched:
            parameter_names.extend(
                _assoc_optimizer_parameter_names(associations[model_name]).get(
                    optimizer_name, ()
                )
            )
        if parameter_names:
            params: list[torch.nn.Parameter] = []
            missing: list[str] = []
            for qualified_name in parameter_names:
                model_name, _, parameter_name = qualified_name.partition(".")
                named_params = named_params_by_model.get(model_name)
                if named_params is None or parameter_name not in named_params:
                    missing.append(qualified_name)
                    continue
                params.append(named_params[parameter_name])
            if missing:
                raise ValueError(
                    f"Checkpoint optimizer {optimizer_name!r} references missing "
                    f"parameter(s): {missing!r}."
                )
            return iter(params)
        return itertools.chain.from_iterable(
            models[name][0].parameters() for name in matched
        )
    # Fallback: if exactly one model exists, use it
    if len(models) == 1:
        return next(iter(models.values()))[0].parameters()
    raise ValueError(
        f"Cannot determine which model's parameters to use for optimizer "
        f"{optimizer_name!r}. Provide associations or use a single model."
    )


def _find_associated_optimizer(
    scheduler_name: str,
    associations: _Associations,
    optimizers: dict[str, tuple[torch.optim.Optimizer, BaseSpec]],
) -> torch.optim.Optimizer:
    """Return the optimizer whose associations include *scheduler_name*."""
    for assoc in associations.values():
        edge = _assoc_scheduler_optimizers(assoc).get(scheduler_name)
        if edge is not None:
            if edge in optimizers:
                return optimizers[edge][0]
            raise ValueError(
                f"Scheduler {scheduler_name!r} is associated with optimizer "
                f"{edge!r}, but that optimizer was not loaded."
            )

        scheduler_names = _assoc_names(assoc, "schedulers")
        optimizer_names = _assoc_names(assoc, "optimizers")
        if scheduler_name in scheduler_names:
            scheduler_index = scheduler_names.index(scheduler_name)
            if scheduler_index < len(optimizer_names):
                optimizer_name = optimizer_names[scheduler_index]
                if optimizer_name in optimizers:
                    return optimizers[optimizer_name][0]
    # Fallback: if exactly one optimizer exists, use it
    if len(optimizers) == 1:
        return next(iter(optimizers.values()))[0]
    raise ValueError(
        f"Cannot determine which optimizer to use for scheduler "
        f"{scheduler_name!r}. Provide associations or use a single optimizer."
    )


def _strategy_metadata_path(root: Path) -> Path:
    """Return the checkpoint strategy metadata path under ``root``."""
    return root / _STRATEGY_FILENAME


def _indexed_strategy_metadata_path(root: Path, checkpoint_index: int) -> Path:
    """Return the per-index strategy metadata path under ``root``."""
    return root / _STRATEGY_CHECKPOINT_DIR / f"{checkpoint_index}.json"


def _read_strategy_metadata(
    root: Path,
    *,
    checkpoint_index: int,
    latest_checkpoint_index: int,
) -> dict[str, Any] | None:
    """Read strategy checkpoint metadata if the checkpoint contains it."""
    indexed_path = _indexed_strategy_metadata_path(root, checkpoint_index)
    if indexed_path.exists():
        return json.loads(indexed_path.read_text())

    path = _strategy_metadata_path(root)
    if not path.exists():
        return None
    if checkpoint_index != latest_checkpoint_index:
        raise FileNotFoundError(
            "This checkpoint has root-level strategy metadata only, so "
            f"checkpoint_index={checkpoint_index} cannot be loaded coherently. "
            f"Load the latest index ({latest_checkpoint_index}) or recreate the "
            "checkpoint with per-index strategy metadata."
        )
    return json.loads(path.read_text())


def _write_strategy_metadata(
    root: Path,
    metadata: Mapping[str, Any],
    *,
    checkpoint_index: int,
) -> None:
    """Write latest and per-index JSON strategy metadata."""
    root.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(metadata, indent=2)
    _strategy_metadata_path(root).write_text(payload)
    indexed_path = _indexed_strategy_metadata_path(root, checkpoint_index)
    indexed_path.parent.mkdir(parents=True, exist_ok=True)
    indexed_path.write_text(payload)


def _component_name(model_name: str, kind: str, index: int, count: int) -> str:
    """Return a stable optimizer/scheduler component name for a model config."""
    suffix = kind if count == 1 else f"{kind}_{index}"
    return f"{model_name}_{suffix}"


def _models_from_strategy_metadata(
    strategy: Any,
    metadata: Mapping[str, Any],
) -> dict[str, tuple[nn.Module, BaseSpec]]:
    """Collect model components and specs from a strategy checkpoint payload."""
    raw_specs = metadata.get("model_specs", {})
    if not isinstance(raw_specs, Mapping):
        raise ValueError("strategy checkpoint metadata has invalid 'model_specs'.")

    models: dict[str, tuple[nn.Module, BaseSpec]] = {}
    missing: list[str] = []
    for name, module in strategy.models.items():
        checkpoint_module = _checkpoint_model(module)
        raw = raw_specs.get(name)
        if raw is None:
            missing.append(name)
            continue
        models[name] = (checkpoint_module, create_model_spec_from_json(dict(raw)))
    if missing:
        raise ValueError(
            "Cannot save strategy checkpoint because model spec generation "
            f"failed for model(s) {missing!r}. Ensure these models can be "
            "reconstructed from BaseSpec before checkpointing."
        )
    return models


def _strategy_components(
    strategy: Any,
) -> tuple[
    dict[str, tuple[nn.Module, BaseSpec]],
    dict[str, tuple[torch.optim.Optimizer, BaseSpec]],
    dict[str, tuple[torch.optim.lr_scheduler.LRScheduler, BaseSpec]],
    _Associations,
    dict[str, Any],
]:
    """Extract manifest components from a :class:`TrainingStrategy` instance."""
    metadata = strategy.to_checkpoint_dict()
    models = _models_from_strategy_metadata(strategy, metadata)
    flat_opts, flat_scheds = strategy._setup_runtime_optimizers(rebuild=False)

    optimizers: dict[str, tuple[torch.optim.Optimizer, BaseSpec]] = {}
    schedulers: dict[str, tuple[torch.optim.lr_scheduler.LRScheduler, BaseSpec]] = {}
    associations: _Associations = {}

    cursor = 0
    for model_name, configs in strategy.optimizer_configs.items():
        assoc = associations.setdefault(
            model_name, {"optimizers": [], "schedulers": []}
        )
        for index, config in enumerate(configs):
            try:
                optimizer = flat_opts[cursor]
                scheduler = flat_scheds[cursor]
            except IndexError as exc:
                raise RuntimeError(
                    "Strategy optimizer state is inconsistent with optimizer_configs."
                ) from exc

            optimizer_name = _component_name(
                model_name, "optimizer", index, len(configs)
            )
            optimizers[optimizer_name] = (
                optimizer,
                create_model_spec(config.optimizer_cls, **config.optimizer_kwargs),
            )
            assoc["optimizers"].append(optimizer_name)
            parameter_names = assoc.setdefault(_OPTIMIZER_PARAMETER_NAMES_KEY, {})
            optimizer_param_ids = {
                id(parameter)
                for group in optimizer.param_groups
                for parameter in group["params"]
            }
            parameter_names[optimizer_name] = [
                f"{model_name}.{name}"
                for name, parameter in models[model_name][0].named_parameters()
                if id(parameter) in optimizer_param_ids
            ]

            if scheduler is not None:
                if config.scheduler_cls is None:
                    raise RuntimeError(
                        f"Strategy has scheduler state for {optimizer_name!r}, "
                        "but its OptimizerConfig has scheduler_cls=None."
                    )
                scheduler_name = _component_name(
                    model_name, "scheduler", index, len(configs)
                )
                schedulers[scheduler_name] = (
                    scheduler,
                    create_model_spec(config.scheduler_cls, **config.scheduler_kwargs),
                )
                assoc["schedulers"].append(scheduler_name)
                scheduler_optimizers = assoc.setdefault(_SCHEDULER_OPTIMIZERS_KEY, {})
                scheduler_optimizers[scheduler_name] = optimizer_name
            cursor += 1

    return models, optimizers, schedulers, associations, metadata


def _loaded_model_objects(
    manifest: CheckpointManifest,
) -> dict[str, nn.Module]:
    """Return loaded models from a hydrated manifest."""
    return {name: pair[0] for name, pair in manifest.models.items() if pair is not None}


def _install_strategy_optimizer_state(
    strategy: Any, manifest: CheckpointManifest
) -> None:
    """Attach loaded optimizer/scheduler objects to a strategy for restart."""
    flat_opts: list[torch.optim.Optimizer] = []
    flat_scheds: list[torch.optim.lr_scheduler.LRScheduler | None] = []
    for model_name, configs in strategy.optimizer_configs.items():
        for index, config in enumerate(configs):
            optimizer_name = _component_name(
                model_name, "optimizer", index, len(configs)
            )
            optimizer_pair = manifest.optimizers.get(optimizer_name)
            if optimizer_pair is None:
                raise ValueError(
                    f"Checkpoint strategy expects optimizer {optimizer_name!r}, "
                    "but it was not loaded from the manifest."
                )
            flat_opts.append(optimizer_pair[0])

            scheduler_name = _component_name(
                model_name, "scheduler", index, len(configs)
            )
            scheduler_pair = manifest.schedulers.get(scheduler_name)
            if config.scheduler_cls is not None and scheduler_pair is None:
                raise ValueError(
                    f"Checkpoint strategy expects scheduler {scheduler_name!r}, "
                    "but it was not loaded from the manifest."
                )
            flat_scheds.append(
                scheduler_pair[0] if scheduler_pair is not None else None
            )

    strategy._optimizers = flat_opts
    strategy._lr_schedulers = flat_scheds
    strategy._resume_optimizer_state = bool(flat_opts)
    if flat_opts:
        strategy._restore_runtime_optimizers_from_loaded_state()


def _restore_strategy_runtime_state(
    strategy: Any,
    metadata: Mapping[str, Any] | None,
) -> None:
    """Restore saved runtime counters into a live strategy."""
    if metadata is None:
        return
    runtime_state = metadata.get("runtime_state", {})
    if runtime_state is None:
        return
    if not isinstance(runtime_state, Mapping):
        raise ValueError(
            "strategy checkpoint metadata has invalid 'runtime_state'; "
            f"got {type(runtime_state).__name__}."
        )
    for key in (
        "step_count",
        "global_step_count",
        "batch_count",
        "epoch_count",
        "epoch_step_count",
    ):
        if key in runtime_state:
            value = int(runtime_state[key])
            if value < 0:
                raise ValueError(
                    f"strategy checkpoint runtime counter {key!r} must be "
                    f"non-negative; got {value}."
                )
            setattr(strategy, key, value)
    if "global_step_count" not in runtime_state and strategy.step_count > 0:
        strategy.global_step_count = strategy.step_count * get_world_size(
            getattr(strategy, "distributed_manager", None)
        )


def _optimizer_scheduler_maps_from_strategy(
    strategy: Any,
) -> tuple[
    dict[str, torch.optim.Optimizer],
    dict[str, torch.optim.lr_scheduler.LRScheduler],
]:
    """Return checkpoint component-name maps for a live strategy runtime."""
    flat_opts, flat_scheds = strategy._setup_runtime_optimizers(rebuild=False)
    optimizers: dict[str, torch.optim.Optimizer] = {}
    schedulers: dict[str, torch.optim.lr_scheduler.LRScheduler] = {}

    cursor = 0
    for model_name, configs in strategy.optimizer_configs.items():
        for index, config in enumerate(configs):
            try:
                optimizer = flat_opts[cursor]
                scheduler = flat_scheds[cursor]
            except IndexError as exc:
                raise RuntimeError(
                    "Strategy optimizer state is inconsistent with optimizer_configs."
                ) from exc

            optimizers[
                _component_name(model_name, "optimizer", index, len(configs))
            ] = optimizer
            if scheduler is not None:
                schedulers[
                    _component_name(model_name, "scheduler", index, len(configs))
                ] = scheduler
            cursor += 1

    return optimizers, schedulers


def _restore_checkpoint_into_strategy(
    root: Path,
    manifest: CheckpointManifest,
    *,
    checkpoint_index: int,
    strategy: Any,
    strategy_metadata: Mapping[str, Any] | None,
    map_location: str | torch.device | None,
) -> dict[str, Any]:
    """Load checkpoint state into an already-constructed strategy."""
    from nvalchemi.training.strategy import TrainingStrategy

    if not isinstance(strategy, TrainingStrategy):
        raise TypeError(
            "strategy must be a TrainingStrategy instance; got "
            f"{type(strategy).__name__}."
        )

    missing_models = sorted(set(manifest.models) - set(strategy.models))
    if missing_models:
        raise KeyError(
            "Checkpoint contains model(s) not present in the live strategy: "
            f"{missing_models!r}."
        )

    loaded_models: dict[str, tuple[nn.Module, BaseSpec | None]] = {}
    for name in manifest.models:
        model = _checkpoint_model(strategy.models[name])
        weights = torch.load(
            root / "models" / name / "checkpoints" / f"{checkpoint_index}.pt",
            weights_only=True,
            map_location=map_location,
        )
        model.load_state_dict(weights)
        spec_path = root / "models" / name / "spec.json"
        spec = _load_spec(spec_path) if spec_path.exists() else None
        loaded_models[name] = (model, spec)

    live_optimizers, live_schedulers = _optimizer_scheduler_maps_from_strategy(strategy)

    loaded_optimizers: dict[str, tuple[torch.optim.Optimizer, BaseSpec | None]] = {}
    missing_optimizers = sorted(set(manifest.optimizers) - set(live_optimizers))
    if missing_optimizers:
        raise KeyError(
            "Checkpoint contains optimizer(s) not present in the live strategy: "
            f"{missing_optimizers!r}."
        )
    for name in manifest.optimizers:
        optimizer = live_optimizers[name]
        state = torch.load(
            root / "optimizers" / name / "checkpoints" / f"{checkpoint_index}.pt",
            weights_only=True,
            map_location=map_location,
        )
        optimizer.load_state_dict(state)
        spec_path = root / "optimizers" / name / "spec.json"
        spec = _load_spec(spec_path) if spec_path.exists() else None
        loaded_optimizers[name] = (optimizer, spec)

    loaded_schedulers: dict[
        str, tuple[torch.optim.lr_scheduler.LRScheduler, BaseSpec | None]
    ] = {}
    missing_schedulers = sorted(set(manifest.schedulers) - set(live_schedulers))
    if missing_schedulers:
        raise KeyError(
            "Checkpoint contains scheduler(s) not present in the live strategy: "
            f"{missing_schedulers!r}."
        )
    for name in manifest.schedulers:
        scheduler = live_schedulers[name]
        state = torch.load(
            root / "schedulers" / name / "checkpoints" / f"{checkpoint_index}.pt",
            weights_only=True,
            map_location=map_location,
        )
        scheduler.load_state_dict(state)
        spec_path = root / "schedulers" / name / "spec.json"
        spec = _load_spec(spec_path) if spec_path.exists() else None
        loaded_schedulers[name] = (scheduler, spec)

    strategy._resume_optimizer_state = bool(loaded_optimizers)
    _restore_strategy_runtime_state(strategy, strategy_metadata)
    _load_hook_states(
        root,
        strategy,
        checkpoint_index,
        map_location=map_location,
    )

    manifest.models = loaded_models
    manifest.optimizers = loaded_optimizers
    manifest.schedulers = loaded_schedulers
    manifest.checkpoint_index = checkpoint_index
    return _manifest_to_loaded_checkpoint(manifest, root=root, strategy=strategy)


def _manifest_to_loaded_checkpoint(
    manifest: CheckpointManifest,
    *,
    root: Path,
    strategy: Any = None,
    source_format: str = "native",
) -> dict[str, Any]:
    """Convert a hydrated manifest into the high-level builtin dict shape."""
    models: dict[str, dict[str, Any]] = {}
    for model_name, pair in manifest.models.items():
        if pair is None:
            continue
        model, spec = pair
        assoc = manifest.associations.get(
            model_name, {"optimizers": [], "schedulers": []}
        )
        model_optimizers = {
            name: {"optimizer": opt_pair[0], "spec": opt_pair[1]}
            for name in _assoc_names(assoc, "optimizers")
            if (opt_pair := manifest.optimizers.get(name)) is not None
        }
        model_schedulers = {
            name: {"scheduler": sched_pair[0], "spec": sched_pair[1]}
            for name in _assoc_names(assoc, "schedulers")
            if (sched_pair := manifest.schedulers.get(name)) is not None
        }
        models[model_name] = {
            "model": model,
            "spec": spec,
            "optimizers": model_optimizers,
            "schedulers": model_schedulers,
            "metadata": {"associations": assoc},
        }

    return {
        "strategy": strategy,
        "models": models,
        "manifest": manifest,
        "checkpoint_index": manifest.checkpoint_index,
        "source": {"format": source_format, "path": str(root)},
    }


def _run_validators(
    loaded: Mapping[str, Any],
    validators: Sequence[CheckpointValidator] | None,
) -> None:
    """Run caller-supplied validators against each loaded model entry."""
    if not validators:
        return
    source = loaded.get("source", {})
    source_path = (
        source.get("path", "<unknown>") if isinstance(source, Mapping) else source
    )
    for model_name, entry in loaded.get("models", {}).items():
        for validator in validators:
            validator_name = getattr(validator, "__name__", type(validator).__name__)
            try:
                validator(model_name, entry, loaded)
            except Exception as exc:
                raise ValueError(
                    f"Checkpoint validator {validator_name!r} failed for model "
                    f"{model_name!r} loaded from {source_path}: {exc}"
                ) from exc


def _load_mace_checkpoint(
    checkpoint_path: Path,
    *,
    map_location: str | torch.device | None,
    adapter_kwargs: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Load a local MACE checkpoint through :class:`MACEWrapper`."""
    kwargs = dict(adapter_kwargs or {})
    allowed = {"model_name", "dtype", "enable_cueq", "compile_model", "compile_kwargs"}
    unknown = sorted(set(kwargs) - allowed)
    if unknown:
        raise ValueError(f"Unknown MACE adapter option(s): {unknown}.")

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            "The MACE checkpoint adapter only accepts local checkpoint files; "
            f"{checkpoint_path} does not exist."
        )

    model_name = kwargs.pop("model_name", "main")
    dtype = kwargs.pop("dtype", None)
    enable_cueq = kwargs.pop("enable_cueq", False)
    compile_model = kwargs.pop("compile_model", False)
    compile_kwargs = kwargs.pop("compile_kwargs", {})
    if not isinstance(compile_kwargs, Mapping):
        raise TypeError("MACE adapter option 'compile_kwargs' must be a mapping.")

    device = torch.device("cpu") if map_location is None else torch.device(map_location)
    warnings.warn(
        "Loading MACE .pt checkpoints requires the MACE full-model pickle "
        "loader under the hood. Only load local MACE checkpoints from trusted "
        "sources.",
        UserWarning,
        stacklevel=2,
    )
    from nvalchemi.models.mace import MACEWrapper

    model = MACEWrapper.from_checkpoint(
        checkpoint_path,
        device=device,
        dtype=dtype,
        enable_cueq=enable_cueq,
        compile_model=compile_model,
        **dict(compile_kwargs),
    )
    return {
        "strategy": None,
        "models": {
            model_name: {
                "model": model,
                "spec": None,
                "optimizers": {},
                "schedulers": {},
                "metadata": {"adapter": "mace"},
            }
        },
        "manifest": None,
        "checkpoint_index": None,
        "source": {"format": "mace", "path": str(checkpoint_path)},
    }


def _strategy_target_device(
    strategy_metadata: Mapping[str, Any] | None,
    map_location: str | torch.device | None,
) -> torch.device | None:
    """Return the model/optimizer load device for a strategy checkpoint."""
    if map_location is not None:
        return torch.device(map_location)
    if strategy_metadata is None:
        return None

    raw_devices = strategy_metadata.get("devices")
    if not isinstance(raw_devices, Sequence) or isinstance(raw_devices, str):
        return None
    if not raw_devices:
        return None
    return torch.device(raw_devices[0])


def _with_strategy_device_override(
    strategy_metadata: Mapping[str, Any],
    map_location: str | torch.device | None,
) -> dict[str, Any]:
    """Return strategy metadata with runtime devices overridden when requested."""
    metadata = dict(strategy_metadata)
    if map_location is not None:
        metadata["devices"] = [str(torch.device(map_location))]
    return metadata


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def save_checkpoint(
    root_folder: Path | str,
    models: dict[str, tuple[nn.Module, BaseSpec]] | Any | None = None,
    optimizers: dict[str, tuple[torch.optim.Optimizer, BaseSpec]] | None = None,
    schedulers: (
        dict[str, tuple[torch.optim.lr_scheduler.LRScheduler, BaseSpec]] | None
    ) = None,
    associations: _Associations | None = None,
    checkpoint_index: int = -1,
    strategy: Any | None = None,
) -> int:
    """Save a checkpoint with a manifest.

    The low-level component form accepts explicit ``models``, ``optimizers``,
    and ``schedulers`` mappings. The strategy-aware form accepts
    ``strategy=TrainingStrategy(...)`` (or the strategy as the second
    positional argument) and writes additional ``strategy.json`` metadata with
    the serializable recipe and restart counters.

    Parameters
    ----------
    root_folder
        Root directory for the checkpoint tree.
    models
        Mapping of model name to ``(module, spec)`` pairs, or a
        :class:`~nvalchemi.training.strategy.TrainingStrategy` instance.
    optimizers
        Optional mapping of optimizer name to ``(optimizer, spec)`` pairs.
    schedulers
        Optional mapping of scheduler name to ``(scheduler, spec)`` pairs.
    associations
        Optional model-centric linkage mapping a model name to
        ``{"optimizers": [...], "schedulers": [...]}``. When ``None``
        (default), associations are inferred automatically by matching
        optimizer ``param_groups`` to model parameters via ``data_ptr()``
        identity, and schedulers to optimizers via object identity.
    checkpoint_index
        Index for the checkpoint files. ``-1`` (default) auto-increments
        from the manifest's last index, or starts at ``0``.
    strategy
        Optional training strategy to save as a restartable checkpoint.

    Returns
    -------
    int
        The checkpoint index that was written.

    Raises
    ------
    ValueError
        If an existing ``spec.json`` disagrees with the spec being saved
        (ignoring ``timestamp``).

    Examples
    --------
    >>> import tempfile, torch.nn as nn
    >>> from nvalchemi.training._spec import create_model_spec
    >>> with tempfile.TemporaryDirectory() as tmp:
    ...     spec = create_model_spec(nn.Linear, in_features=4, out_features=2)
    ...     save_checkpoint(tmp, models={"main": (nn.Linear(4, 2), spec)})
    0
    """
    from nvalchemi.training.strategy import TrainingStrategy

    root = Path(root_folder)
    strategy_metadata: dict[str, Any] | None = None
    if strategy is None and isinstance(models, TrainingStrategy):
        strategy = models
        models = None
    if strategy is not None:
        if not isinstance(strategy, TrainingStrategy):
            raise TypeError(
                "strategy must be a TrainingStrategy instance; got "
                f"{type(strategy).__name__}."
            )
        (
            models,
            optimizers,
            schedulers,
            associations,
            strategy_metadata,
        ) = _strategy_components(strategy)
    if models is None:
        raise ValueError("save_checkpoint requires models=... or strategy=....")

    models = _checkpoint_model_components(models)
    optimizers = optimizers or {}
    schedulers = schedulers or {}
    if associations is None:
        associations = _infer_associations(models, optimizers, schedulers)
    else:
        associations = _with_scheduler_optimizer_edges(
            associations, optimizers, schedulers
        )

    checkpoint_index = _resolve_checkpoint_index(root, checkpoint_index)

    # Save each component category
    for name, (module, spec) in models.items():
        _save_component(
            root, "models", name, module.state_dict(), spec, checkpoint_index
        )

    for name, (opt, spec) in optimizers.items():
        _save_component(
            root, "optimizers", name, opt.state_dict(), spec, checkpoint_index
        )

    for name, (sched, spec) in schedulers.items():
        _save_component(
            root, "schedulers", name, sched.state_dict(), spec, checkpoint_index
        )

    # Write manifest — pass live dicts directly; PlainSerializer extracts keys
    manifest = CheckpointManifest(
        checkpoint_index=checkpoint_index,
        models=models,
        optimizers=optimizers,
        schedulers=schedulers,
        associations=associations,
    )
    manifest.write(root)
    if strategy_metadata is not None:
        _write_strategy_metadata(
            root, strategy_metadata, checkpoint_index=checkpoint_index
        )
    if strategy is not None:
        _save_hook_states(root, _snapshot_hook_states(strategy), checkpoint_index)
    return checkpoint_index


def load_checkpoint(
    root_folder: Path | str,
    checkpoint_index: int = -1,
    map_location: str | torch.device | None = None,
    model_names: Iterable[str] | None = None,
    *,
    adapter: str | None = None,
    adapter_kwargs: Mapping[str, Any] | None = None,
    validators: Sequence[CheckpointValidator] | None = None,
    hooks: Sequence[Any] | None = None,
    training_fn: Any = None,
    strategy: Any | None = None,
) -> CheckpointManifest | dict[str, Any]:
    """Load a multi-component checkpoint written by :func:`save_checkpoint`.

    Components are rebuilt in dependency order: models first, then
    optimizers (which need model parameters), then schedulers (which need
    an optimizer instance). Associations from the manifest wire each
    optimizer to the correct model and each scheduler to the correct
    optimizer.

    Parameters
    ----------
    root_folder
        Root directory containing ``manifest.json``.
    checkpoint_index
        Index of the checkpoint to load. ``-1`` (default) loads the
        latest index recorded in the manifest.
    map_location
        Forwarded to every :func:`torch.load` call. When not ``None``,
        each loaded model is additionally moved via
        ``model.to(map_location)``. Optimizers and schedulers have their
        state placed by ``torch.load`` alone (they lack a standard
        ``.to()`` API).
    model_names
        If given, load only the models with these names together with the
        optimizers and schedulers wired to them through
        ``manifest.associations``. Accepts any iterable of strings
        (typically a set). ``None`` (default) loads every component on
        disk. The returned manifest's ``associations`` still reflects the
        full on-disk mapping, so callers can inspect what was not loaded.
    adapter
        Optional foreign-checkpoint adapter name. V1 supports ``"mace"`` for
        trusted local MACE ``.pt`` files.
    adapter_kwargs
        Adapter-specific options. For ``adapter="mace"``, accepted keys are
        ``model_name``, ``dtype``, ``enable_cueq``, ``compile_model``, and
        ``compile_kwargs``.
    validators
        Optional callbacks invoked as ``validator(model_name, entry, loaded)``
        for each high-level loaded model entry. Use these for model-specific
        chemistry or topology compatibility checks.
    hooks
        Runtime hooks supplied when reconstructing a saved strategy.
    training_fn
        Runtime training function override supplied when reconstructing a
        saved strategy.
    strategy
        Optional already-constructed strategy to hydrate from the checkpoint.
        This mode restores model, optimizer, scheduler, runtime-counter, and
        checkpointable hook state into the live objects instead of rebuilding
        models from saved specs.

    Returns
    -------
    CheckpointManifest
        For legacy component-only checkpoints, a hydrated manifest is
        returned.
    dict[str, Any]
        For strategy checkpoints or adapter loads, a builtin dict containing
        ``strategy``, ``models``, ``manifest``, ``checkpoint_index``, and
        ``source`` is returned.

    Raises
    ------
    FileNotFoundError
        If ``manifest.json`` is missing or a checkpoint ``.pt`` file
        does not exist.
    KeyError
        If any name in ``model_names`` does not appear in
        ``manifest.models``.
    RuntimeError
        If a model spec does not build an :class:`~torch.nn.Module`.

    Examples
    --------
    >>> import tempfile, torch.nn as nn
    >>> from nvalchemi.training._spec import create_model_spec
    >>> with tempfile.TemporaryDirectory() as tmp:
    ...     spec = create_model_spec(nn.Linear, in_features=4, out_features=2)
    ...     _ = save_checkpoint(tmp, models={"main": (nn.Linear(4, 2), spec)})
    ...     result = load_checkpoint(tmp)
    ...     isinstance(result.models["main"][0], nn.Linear)
    True

    Loading onto CPU regardless of the original device::

        result = load_checkpoint("runs/exp1", map_location="cpu")

    Selecting a subset of models (e.g., teacher and student but not the
    third auxiliary model)::

        result = load_checkpoint("runs/kd", model_names={"teacher", "student"})
    """
    root = Path(root_folder)
    if adapter is not None:
        if strategy is not None:
            raise ValueError("load_checkpoint does not support strategy with adapter.")
        if adapter != "mace":
            raise ValueError(
                f"Unsupported checkpoint adapter {adapter!r}; supported: ['mace']."
            )
        loaded = _load_mace_checkpoint(
            root,
            map_location=map_location,
            adapter_kwargs=adapter_kwargs,
        )
        _run_validators(loaded, validators)
        return loaded

    manifest = CheckpointManifest.read(root)

    if checkpoint_index == -1:
        checkpoint_index = manifest.checkpoint_index

    associations = manifest.associations
    strategy_metadata = _read_strategy_metadata(
        root,
        checkpoint_index=checkpoint_index,
        latest_checkpoint_index=manifest.checkpoint_index,
    )
    load_location = _strategy_target_device(strategy_metadata, map_location)

    if strategy is not None:
        if model_names is not None:
            raise ValueError(
                "load_checkpoint(strategy=...) restores the complete live strategy; "
                "model_names is not supported in this mode."
            )
        loaded = _restore_checkpoint_into_strategy(
            root,
            manifest,
            checkpoint_index=checkpoint_index,
            strategy=strategy,
            strategy_metadata=strategy_metadata,
            map_location=load_location,
        )
        if strategy_metadata is not None:
            loaded["strategy_metadata"] = _with_strategy_device_override(
                strategy_metadata, map_location
            )
        _run_validators(loaded, validators)
        return loaded

    # determine what models to load
    selected_models = set(manifest.models) if model_names is None else set(model_names)
    unknown = selected_models - set(manifest.models)
    if unknown:
        raise KeyError(
            f"Unknown model(s) {sorted(unknown)!r}. "
            f"Available: {sorted(manifest.models)!r}"
        )

    # Build the load set as the union of each selected model's associations.
    # When ``model_names is None`` this is equivalent to loading every
    # component listed in the manifest.
    models_to_load = [n for n in manifest.models if n in selected_models]
    if model_names is None:
        optimizers_to_load = list(manifest.optimizers)
        schedulers_to_load = list(manifest.schedulers)
    else:
        wanted_optimizers: set[str] = set()
        wanted_schedulers: set[str] = set()
        for n in selected_models:
            assoc = associations.get(n, {})
            wanted_optimizers.update(_assoc_names(assoc, "optimizers"))
            wanted_schedulers.update(_assoc_names(assoc, "schedulers"))
        optimizers_to_load = [n for n in manifest.optimizers if n in wanted_optimizers]
        schedulers_to_load = [n for n in manifest.schedulers if n in wanted_schedulers]

    # --- Models ---
    loaded_models: dict[str, tuple[nn.Module, BaseSpec]] = {}
    for name in models_to_load:
        spec = _load_spec(root / "models" / name / "spec.json")
        build_kwargs = (
            {"device": load_location}
            if load_location is not None and spec.accepts_kwarg("device")
            else {}
        )
        model = spec.build(**build_kwargs)
        if not isinstance(model, nn.Module):
            raise RuntimeError(
                f"Model spec for {name!r} built {type(model)!r}, expected nn.Module."
            )
        # Move models whose factories do not accept device after construction.
        # Factory-loaded models such as MACE + cuEq need the device during
        # construction so conversion happens on the intended accelerator.
        if load_location is not None and not build_kwargs:
            model.to(load_location)
        weights = torch.load(
            root / "models" / name / "checkpoints" / f"{checkpoint_index}.pt",
            weights_only=True,
            map_location=load_location,
        )
        model.load_state_dict(weights)
        loaded_models[name] = (model, spec)

    # --- Optimizers ---
    loaded_optimizers: dict[str, tuple[torch.optim.Optimizer, BaseSpec]] = {}
    for name in optimizers_to_load:
        spec = _load_spec(root / "optimizers" / name / "spec.json")
        params = _find_associated_model_params(name, associations, loaded_models)
        optimizer = spec.build(params)
        state = torch.load(
            root / "optimizers" / name / "checkpoints" / f"{checkpoint_index}.pt",
            weights_only=True,
            map_location=load_location,
        )
        optimizer.load_state_dict(state)
        loaded_optimizers[name] = (optimizer, spec)

    # --- Schedulers ---
    loaded_schedulers: dict[
        str, tuple[torch.optim.lr_scheduler.LRScheduler, BaseSpec]
    ] = {}
    for name in schedulers_to_load:
        spec = _load_spec(root / "schedulers" / name / "spec.json")
        assoc_optimizer = _find_associated_optimizer(
            name, associations, loaded_optimizers
        )
        scheduler = spec.build(assoc_optimizer)
        state = torch.load(
            root / "schedulers" / name / "checkpoints" / f"{checkpoint_index}.pt",
            weights_only=True,
            map_location=load_location,
        )
        scheduler.load_state_dict(state)
        loaded_schedulers[name] = (scheduler, spec)

    # Hydrate manifest with live objects
    manifest.models = loaded_models
    manifest.optimizers = loaded_optimizers
    manifest.schedulers = loaded_schedulers
    manifest.checkpoint_index = checkpoint_index
    if strategy_metadata is None:
        if validators is not None:
            loaded = _manifest_to_loaded_checkpoint(manifest, root=root)
            _run_validators(loaded, validators)
        return manifest

    strategy = None
    if strategy_metadata is not None and model_names is None:
        from nvalchemi.training.strategy import TrainingStrategy

        loaded_strategy_models: Any = _loaded_model_objects(manifest)
        if strategy_metadata.get("single_model_input") is True and set(
            loaded_strategy_models
        ) == {"main"}:
            loaded_strategy_models = loaded_strategy_models["main"]

        runtime_strategy_metadata = _with_strategy_device_override(
            strategy_metadata, map_location
        )
        strategy = TrainingStrategy.from_checkpoint_dict(
            runtime_strategy_metadata,
            models=loaded_strategy_models,
            hooks=hooks,
            training_fn=training_fn,
        )
        _install_strategy_optimizer_state(strategy, manifest)
        _load_hook_states(
            root,
            strategy,
            checkpoint_index,
            map_location=load_location,
        )

    loaded = _manifest_to_loaded_checkpoint(
        manifest,
        root=root,
        strategy=strategy,
    )
    if strategy_metadata is not None:
        loaded["strategy_metadata"] = _with_strategy_device_override(
            strategy_metadata, map_location
        )
    _run_validators(loaded, validators)
    return loaded


def _load_spec(spec_path: Path) -> BaseSpec:
    """Read and rehydrate a :class:`BaseSpec` from *spec_path*."""
    if not spec_path.exists():
        raise FileNotFoundError(f"Expected spec at {spec_path} but file not found.")
    return create_model_spec_from_json(json.loads(spec_path.read_text()))
