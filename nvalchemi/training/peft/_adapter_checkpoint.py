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
"""Adapter checkpoint helpers."""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import torch
from torch import nn

from nvalchemi.training._checkpoint import _checkpoint_model
from nvalchemi.training._spec import BaseSpec, create_model_spec_from_json
from nvalchemi.training.hooks.finetune import _resolve_parent


def resolve_child_module(models: dict[str, nn.Module], target: str) -> nn.Module:
    """Resolve a fully-qualified module target from ``models``.

    Parameters
    ----------
    models
        Models from which to resolve the target module.
    target
        Fully-qualified module name, including the model prefix.
    """
    parent, child_name = _resolve_parent(models, target)
    module = getattr(parent, child_name)
    if not isinstance(module, nn.Module):
        raise TypeError(
            f"Module patch target {target!r} resolved to "
            f"{type(module).__name__}, expected nn.Module."
        )
    return module


def adapter_export_models(
    live_models: dict[str, nn.Module],
    inference_model: nn.Module | nn.ModuleDict | None,
    model_names: list[str],
    *,
    use_ema: bool,
) -> dict[str, nn.Module]:
    """Return the model objects whose adapter state should be exported.

    Parameters
    ----------
    live_models
        Models used by the training workflow.
    inference_model
        Inference model populated with EMA weights, when available.
    model_names
        Names of the models selected for export.
    use_ema
        Whether to export weights from ``inference_model`` instead of live models.
    """
    if not use_ema:
        return {name: _checkpoint_model(live_models[name]) for name in model_names}
    if inference_model is None:
        raise RuntimeError(
            "save_adapter(use_ema=True) requires a populated inference_model slot, "
            "for example from EMAHook."
        )
    if isinstance(inference_model, nn.ModuleDict):
        missing = [name for name in model_names if name not in inference_model]
        if missing:
            raise RuntimeError(
                "save_adapter(use_ema=True) requires inference_model entries for "
                f"model(s) {missing!r}; available entries: "
                f"{sorted(inference_model.keys())}."
            )
        return {name: _checkpoint_model(inference_model[name]) for name in model_names}
    if model_names != ["main"]:
        raise RuntimeError(
            "save_adapter(use_ema=True) with multiple named models requires "
            "inference_model to be an nn.ModuleDict."
        )
    return {"main": _checkpoint_model(inference_model)}


def adapter_dir(root_folder: Path | str, *, dirname: str = "adapter") -> Path:
    """Normalize a path to the saved adapter directory.

    Parameters
    ----------
    root_folder
        Adapter directory or its parent directory.
    dirname
        Directory name used for this adapter export.
    """
    normalized = Path(root_folder)
    if normalized.name != dirname:
        normalized = normalized / dirname
    return normalized


def create_dir(dir_path: Path) -> Path:
    """Create a directory, warning before overwriting known files.

    Parameters
    ----------
    dir_path
        Directory to create.
    """
    if dir_path.exists() and any(dir_path.iterdir()):
        warnings.warn(
            "Directory already exists and is non-empty; "
            f"overwriting known files in {dir_path}.",
            UserWarning,
            stacklevel=2,
        )
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


def read_adapter_metadata(
    adapter_dir: Path,
    *,
    adapter_kind: str,
    schema_version: int,
    target_patterns_key: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Read and validate adapter manifest and strategy metadata.

    Parameters
    ----------
    adapter_dir
        Directory containing the adapter manifest and strategy files.
    target_patterns_key
        Strategy key required to recreate the adapter target selection.
    """
    manifest_path = adapter_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError("Adapter is missing required manifest.json.")
    manifest = json.loads(manifest_path.read_text())
    if not isinstance(manifest, dict):
        raise ValueError("Adapter manifest.json must contain a JSON object.")

    kind = manifest.get("kind")
    if kind != adapter_kind:
        raise ValueError(f"Unsupported adapter kind {kind!r}; expected {adapter_kind!r}.")
    version = manifest.get("schema_version", 0)
    if not isinstance(version, int):
        raise ValueError("Adapter manifest schema_version must be an integer.")
    if version > schema_version:
        raise ValueError(
            f"Adapter schema version {version} is newer than supported "
            f"({schema_version}). Upgrade nvalchemi to load this adapter."
        )
    manifest["schema_version"] = schema_version
    models = manifest.get("models")
    if not isinstance(models, dict):
        raise ValueError("Adapter manifest must contain a models object.")

    strategy_path = adapter_dir / "strategy.json"
    if not strategy_path.is_file():
        raise FileNotFoundError("Adapter is missing required strategy.json.")
    strategy = json.loads(strategy_path.read_text())
    if not isinstance(strategy, dict):
        raise ValueError("Adapter strategy.json must contain a JSON object.")
    if target_patterns_key not in strategy:
        raise ValueError(f"Adapter strategy.json is missing {target_patterns_key}.")
    if "base_model_fingerprints" not in strategy:
        raise ValueError("Adapter strategy.json is missing base_model_fingerprints.")

    if not models:
        raise ValueError("Adapter manifest contains no models.")
    return manifest, strategy


def load_adapter_state(adapter_dir: Path, relative_path: str) -> dict[str, Any]:
    """Load an adapter tensor group from a manifest-relative path.

    Parameters
    ----------
    adapter_dir
        Directory containing the saved adapter.
    relative_path
        Path to the tensor group relative to ``adapter_dir``.
    """
    state_path = adapter_dir / relative_path
    if not state_path.is_file():
        raise FileNotFoundError(f"Adapter state file {relative_path!r} is missing.")
    state = torch.load(state_path, weights_only=True, map_location="cpu")
    if not isinstance(state, dict):
        raise ValueError(f"Adapter state file {relative_path!r} must load a dict.")
    return state


def restore_module_patch_specs(
    strategy: dict[str, Any],
    patch_targets: tuple[str, ...],
) -> dict[str, BaseSpec]:
    """Restore saved module patch specs for the requested adapter.

    Parameters
    ----------
    strategy
        Saved strategy metadata containing module patch specifications.
    patch_targets
        Fully-qualified module patch names to restore (i.e., including the
        model prefix).
    """
    raw_patches = strategy.get("module_patches", {})
    if patch_targets and not isinstance(raw_patches, dict):
        raise ValueError("strategy.json module_patches must be an object.")
    patches: dict[str, BaseSpec] = {}
    for target in patch_targets:
        patch_spec = raw_patches.get(target)
        if not isinstance(patch_spec, dict):
            raise ValueError(
                f"Adapter is missing BaseSpec metadata for module patch {target!r}."
            )
        if "cls_path" not in patch_spec:
            raise ValueError(
                f"Adapter module patch {target!r} must be saved as a "
                "BaseSpec, not direct module metadata."
            )
        patches[target] = create_model_spec_from_json(patch_spec)
    return patches


def stored_base_fingerprints(workflow: Any) -> dict[str, str]:
    """Return base fingerprints saved by an adapter setup hook.

    Parameters
    ----------
    workflow
        Workflow containing models and base fingerprint metadata.
    """
    fingerprints = workflow._base_fingerprints
    if not isinstance(fingerprints, dict) or not fingerprints:
        raise ValueError(
            "Adapter checkpointing requires base fingerprints generated."
        )
    normalized: dict[str, str] = {}
    for model_name in workflow.models:
        fingerprint = fingerprints.get(model_name)
        if not isinstance(fingerprint, str) or not fingerprint:
            raise ValueError(
                f"Adapter base fingerprints are missing an entry for model {model_name!r}."
            )
        normalized[model_name] = fingerprint
    return normalized
