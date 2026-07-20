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
"""LoRA adapter checkpoint helpers."""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import torch
from torch import nn

from nvalchemi.training._checkpoint import _checkpoint_model, _snapshot_state_dict
from nvalchemi.training._spec import BaseSpec, create_model_spec_from_json
from nvalchemi.training.hooks.finetune import _resolve_parent

ADAPTER_SCHEMA_VERSION = 1
ADAPTER_KIND = "nvalchemi_lora_adapter"
ADAPTER_DIR = "lora"


def model_local_parameter_state(
    model_name: str,
    parameter_names: set[str],
    named_parameters: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Return model-owned parameter tensors keyed by model-local names.

    Parameters
    ----------
    model_name
        Name used to qualify the model's parameters.
    parameter_names
        Fully-qualified parameter names selected for export (i.e., including
        the model prefix).
    named_parameters
        Fully-qualified parameter names mapped to their tensors.
    """
    prefix = f"{model_name}."
    return {
        name.removeprefix(prefix): named_parameters[name]
        for name in sorted(parameter_names)
        if name.startswith(prefix)
    }


def model_local_names(model_name: str, names: set[str]) -> set[str]:
    """Return parameter names for one model with the model prefix removed.

    Parameters
    ----------
    model_name
        Name used to qualify the model's parameters.
    names
        Fully-qualified parameter names to filter and normalize (i.e., including
        the model prefix).
    """
    prefix = f"{model_name}."
    return {name.removeprefix(prefix) for name in names if name.startswith(prefix)}


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


def adapter_dir(root_folder: Path | str) -> Path:
    """Normalize a path to the saved LoRA adapter directory.

    Parameters
    ----------
    root_folder
        Adapter directory or its parent directory.
    """
    normalized = Path(root_folder)
    if normalized.name != ADAPTER_DIR:
        normalized = normalized / ADAPTER_DIR
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


def read_adapter_metadata(adapter_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Read and validate LoRA adapter manifest and strategy metadata.

    Parameters
    ----------
    adapter_dir
        Directory containing the adapter manifest and strategy files.
    """
    manifest_path = adapter_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError("LoRA adapter is missing required manifest.json.")
    manifest = json.loads(manifest_path.read_text())
    if not isinstance(manifest, dict):
        raise ValueError("LoRA adapter manifest.json must contain a JSON object.")

    kind = manifest.get("kind")
    if kind != ADAPTER_KIND:
        raise ValueError(
            f"Unsupported LoRA adapter kind {kind!r}; expected {ADAPTER_KIND!r}."
        )
    version = manifest.get("schema_version", 0)
    if not isinstance(version, int):
        raise ValueError("LoRA adapter manifest schema_version must be an integer.")
    if version > ADAPTER_SCHEMA_VERSION:
        raise ValueError(
            f"LoRA adapter schema version {version} is newer than supported "
            f"({ADAPTER_SCHEMA_VERSION}). Upgrade nvalchemi to load this adapter."
        )
    manifest["schema_version"] = ADAPTER_SCHEMA_VERSION
    models = manifest.get("models")
    if not isinstance(models, dict):
        raise ValueError("LoRA adapter manifest must contain a models object.")

    strategy_path = adapter_dir / "strategy.json"
    if not strategy_path.is_file():
        raise FileNotFoundError("LoRA adapter is missing required strategy.json.")
    strategy = json.loads(strategy_path.read_text())
    if not isinstance(strategy, dict):
        raise ValueError("LoRA adapter strategy.json must contain a JSON object.")
    if "lora_target_patterns" not in strategy:
        raise ValueError("LoRA adapter strategy.json is missing lora_target_patterns.")
    if "base_model_fingerprints" not in strategy:
        raise ValueError(
            "LoRA adapter strategy.json is missing base_model_fingerprints."
        )

    if not models:
        raise ValueError("LoRA adapter manifest contains no models.")
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
        raise FileNotFoundError(
            f"LoRA adapter state file {relative_path!r} is missing."
        )
    state = torch.load(state_path, weights_only=True, map_location="cpu")
    if not isinstance(state, dict):
        raise ValueError(f"LoRA adapter state file {relative_path!r} must load a dict.")
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
                f"LoRA adapter is missing BaseSpec metadata for module patch "
                f"{target!r}."
            )
        if "cls_path" not in patch_spec:
            raise ValueError(
                f"LoRA adapter module patch {target!r} must be saved as a "
                "BaseSpec, not direct module metadata."
            )
        patches[target] = create_model_spec_from_json(patch_spec)
    return patches


def selected_state_dict(
    state_dict: dict[str, Any],
    names: set[str],
) -> dict[str, Any]:
    """Return selected state entries and fail if metadata names are stale.

    Parameters
    ----------
    state_dict
        Complete model state dictionary.
    names
        State entry names to select.
    """
    missing = sorted(names - set(state_dict))
    if missing:
        raise KeyError(f"Cannot checkpoint missing LoRA parameter(s): {missing!r}.")
    return {name: state_dict[name] for name in sorted(names)}


def stored_base_fingerprints(workflow: Any) -> dict[str, str]:
    """Return base fingerprints saved by ``LoRAApplyHook``.

    Parameters
    ----------
    workflow
        LoRA workflow containing models and adapter metadata.
    """
    metadata = getattr(workflow, "_lora_metadata", None)
    if not isinstance(metadata, dict) or not metadata:
        raise ValueError(
            "LoRACheckpointHook requires LoRA adapter metadata from LoRAApplyHook."
        )
    fingerprints: dict[str, str] = {}
    for model_name in workflow.models:
        model_metadata = metadata.get(model_name)
        if not isinstance(model_metadata, dict):
            raise ValueError(
                f"LoRA adapter metadata is missing for model {model_name!r}."
            )
        fingerprint = model_metadata.get("base_model_fingerprint")
        if not isinstance(fingerprint, str) or not fingerprint:
            raise ValueError(
                "LoRA adapter metadata is missing base_model_fingerprint "
                f"for model {model_name!r}."
            )
        fingerprints[model_name] = fingerprint
    return fingerprints


def filter_snapshot_to_trainable_state(
    snapshot: dict[str, Any],
    workflow: Any,
) -> None:
    """Mutate a checkpoint snapshot to keep only trainable model states 
    (e.g., LoRA adapter, module-patch, and extra trainable parameters).

    Parameters
    ----------
    snapshot
        Checkpoint snapshot to filter in place.
    workflow
        LoRA workflow containing trainable-parameter bookkeeping.
    """
    lora_names = set(getattr(workflow, "_lora_adapter_parameter_names", set()))
    extra_names = set(getattr(workflow, "_extra_trainable_parameter_names", set()))
    patch_names = set(getattr(workflow, "_patched_parameter_names", set()))
    if not (lora_names or extra_names or patch_names):
        raise ValueError(
            "LoRACheckpointHook selected no trainable parameters. Ensure "
            "LoRATrainableParameterHook has run before checkpointing."
        )
    warnings.warn(
        "Saving a LoRA checkpoint with include_base_parameters=False stores only "
        "adapter, module-patch, and extra trainable tensors. Restoring this "
        "checkpoint requires the saved model spec to reconstruct the exact base "
        "model weights; set include_base_parameters=True when the base weights "
        "are local, custom, or otherwise not reproducibly recoverable from the spec.",
        UserWarning,
        stacklevel=3,
    )

    filtered_models = {}
    for model_name, (state_dict, spec) in snapshot["models"].items():
        names = set()
        names.update(model_local_names(model_name, lora_names))
        names.update(model_local_names(model_name, extra_names))
        names.update(model_local_names(model_name, patch_names))
        filtered_models[model_name] = (
            _snapshot_state_dict(selected_state_dict(state_dict, names)),
            spec,
        )
    snapshot["models"] = filtered_models
    snapshot["strategy_metadata"]["model_state_load"] = "partial"
