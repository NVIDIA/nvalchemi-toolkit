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
"""Load PEFT weights from native training checkpoints."""

from __future__ import annotations

import warnings
from collections.abc import Collection, Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, TypeAlias

import torch
from torch import nn

from nvalchemi.training._checkpoint import (
    CheckpointManifest,
    _load_partial_model_state,
    _read_strategy_metadata,
)
from nvalchemi.training._spec import create_model_spec_from_json
from nvalchemi.training.hooks.finetune import ModulePatchHook
from nvalchemi.training.peft.registry import get_peft_registration_by_method

__all__ = ["load_peft_checkpoint_into_model"]

ImportPathValidator: TypeAlias = Callable[[str, str], None]

_DEFAULT_ALLOWED_IMPORT_PATHS: frozenset[str] = frozenset(
    {
        "cuequivariance_torch.operations.linear.Linear",
        "e3nn.nn._fc._Layer",
        "e3nn.o3._linear.Linear",
        "nvalchemi.training.peft.lora_wrappers.CuEquivarianceLoRALinear",
        "nvalchemi.training.peft.lora_wrappers.E3NNFullyConnectedLoRALayer",
        "nvalchemi.training.peft.lora_wrappers.EquivariantLoRALinear",
        "physicsnemo.experimental.peft.LoRALinear",
        "physicsnemo.experimental.peft.lora.LoRALinear",
        "torch.nn.Linear",
        "torch.nn.modules.linear.Linear",
    }
)


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------


def load_peft_checkpoint_into_model(
    model: nn.Module,
    checkpoint_dir: Path | str,
    *,
    checkpoint_index: int = -1,
    model_name: str = "main",
    map_location: str | torch.device | None = "cpu",
    merge: bool = True,
    strict: bool = True,
    allowed_import_paths: Collection[str] | None = None,
    trust_remote_code: bool = False,
) -> nn.Module:
    """Load PEFT weights from a native checkpoint into a compatible base model.

    This mirrors the registration-time order used by
    :class:`~nvalchemi.training.FineTuningStrategy`: validate the base model,
    apply the saved PEFT structure, apply saved module patches, then load the
    checkpointed model state.

    Parameters
    ----------
    model
        Base model instance compatible with the checkpoint metadata. The loader
        mutates this model in place by applying saved PEFT structure, module
        patches, and checkpointed weights.
    checkpoint_dir
        Root directory containing a checkpoint written by ``save_checkpoint`` or
        ``CheckpointHook``.
    checkpoint_index
        Checkpoint index to read. ``-1`` loads the latest manifest index.
    model_name
        Model entry to load from the checkpoint. Defaults to ``"main"``.
    map_location
        Device mapping used when reading checkpoint tensors.
    merge
        If ``True``, merge PEFT weights into the model when the backend supports it.
    strict
        If ``True``, raise on compatibility mismatches. If ``False``, warn and
        continue for supported soft mismatches.
    allowed_import_paths
        Additional dotted import paths trusted for custom PEFT metadata and saved
        module patch classes. Built-in PEFT and supported layer paths are always
        trusted. Entries may be exact paths or namespace patterns ending in
        ``.*``. Cannot be provided with ``trust_remote_code=True``.
    trust_remote_code
        If ``True``, trust checkpoint metadata to import arbitrary custom Python
        classes. Use only for locally trusted checkpoints. Defaults to ``False``.

    Returns
    -------
    torch.nn.Module
        The input model after PEFT structure and learned weights have been loaded.
    """
    root = Path(checkpoint_dir)
    manifest = CheckpointManifest.read(root)
    resolved_index = (
        manifest.checkpoint_index if checkpoint_index == -1 else checkpoint_index
    )
    if model_name not in manifest.models:
        raise KeyError(
            f"PEFT checkpoint does not contain model {model_name!r}; available "
            f"models: {sorted(manifest.models)}."
        )
    strategy_metadata = _read_strategy_metadata(
        root,
        checkpoint_index=resolved_index,
        latest_checkpoint_index=manifest.checkpoint_index,
    )
    if strategy_metadata is None:
        raise ValueError(
            "PEFT checkpoint loading requires strategy metadata. Use a checkpoint "
            "written from a FineTuningStrategy with peft_config."
        )

    # Select the appropriate PEFT method from the canonical tagged config.
    raw_peft_config = strategy_metadata.get("peft_config")
    if not isinstance(raw_peft_config, Mapping):
        raise ValueError("Checkpoint strategy metadata does not contain peft_config.")
    method = raw_peft_config.get("peft_method")
    registration = get_peft_registration_by_method(method)

    # Validate the base model and recreate the saved PEFT/module structure before loading weights.
    import_path_validator = _make_import_path_validator(
        allowed_import_paths=allowed_import_paths,
        trust_remote_code=trust_remote_code,
    )
    _validate_checkpoint_base_fingerprint(
        model,
        strategy_metadata,
        model_name=model_name,
        strict=strict,
    )
    registration.apply_peft_from_checkpoint_metadata(
        model,
        strategy_metadata,
        model_name=model_name,
        import_path_validator=import_path_validator,
    )
    _apply_checkpoint_module_patches(
        model,
        strategy_metadata,
        model_name=model_name,
        import_path_validator=import_path_validator,
    )
    _load_checkpoint_model_state(
        model,
        root,
        strategy_metadata=strategy_metadata,
        checkpoint_index=resolved_index,
        model_name=model_name,
        map_location=map_location,
    )

    # Merge PEFT weights into the model if supported.
    if merge:
        model = registration.merge_peft(model, strict=strict)

    return model


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _make_import_path_validator(
    *,
    allowed_import_paths: Collection[str] | None,
    trust_remote_code: bool,
) -> ImportPathValidator:
    """Return a checkpoint metadata import validator."""
    if trust_remote_code and allowed_import_paths is not None:
        raise ValueError(
            "allowed_import_paths must be None when trust_remote_code=True."
        )
    if allowed_import_paths is not None and not all(
        isinstance(path, str) for path in allowed_import_paths
    ):
        raise ValueError("allowed_import_paths entries must be strings.")
    allowed = set(_DEFAULT_ALLOWED_IMPORT_PATHS)
    allowed.update(allowed_import_paths or ())

    def validate(path: str, context: str) -> None:
        if trust_remote_code:
            return
        if any(_path_matches(path, pattern) for pattern in allowed):
            return
        raise ValueError(
            f"{context} import path {path!r} is not allowed. Pass "
            "allowed_import_paths to trust this path, or set "
            "trust_remote_code=True for locally trusted checkpoint metadata."
        )

    return validate


def _path_matches(path: str, pattern: str) -> bool:
    """Return whether ``path`` matches an exact path or ``namespace.*`` pattern."""
    if pattern.endswith(".*"):
        return path.startswith(pattern[:-1])
    return path == pattern


def _validate_base_spec_import_paths(
    spec: Mapping[str, Any],
    *,
    import_path_validator: ImportPathValidator,
    context: str,
) -> None:
    """Validate all nested ``BaseSpec`` import paths before loading."""
    cls_path = spec.get("cls_path")
    if isinstance(cls_path, str):
        import_path_validator(cls_path, context)
    for value in spec.values():
        if isinstance(value, Mapping):
            _validate_base_spec_import_paths(
                value,
                import_path_validator=import_path_validator,
                context=context,
            )
        elif isinstance(value, list | tuple):
            for item in value:
                if isinstance(item, Mapping):
                    _validate_base_spec_import_paths(
                        item,
                        import_path_validator=import_path_validator,
                        context=context,
                    )


def _load_checkpoint_model_state(
    model: nn.Module,
    checkpoint_dir: Path,
    *,
    strategy_metadata: Mapping[str, Any],
    checkpoint_index: int,
    model_name: str,
    map_location: str | torch.device | None,
) -> None:
    """Load full or partial checkpoint state into a prepared model."""
    weights = torch.load(
        checkpoint_dir
        / "models"
        / model_name
        / "checkpoints"
        / f"{checkpoint_index}.pt",
        weights_only=True,
        map_location=map_location,
    )
    if not isinstance(weights, dict):
        raise ValueError(
            f"PEFT checkpoint model state for {model_name!r} must load a dict."
        )
    if strategy_metadata.get("model_state_load") == "partial":
        _load_partial_model_state(model, weights, model_name=model_name)
    else:
        model.load_state_dict(weights)


def _validate_checkpoint_base_fingerprint(
    model: nn.Module,
    strategy_metadata: Mapping[str, Any],
    *,
    model_name: str,
    strict: bool,
) -> None:
    """Validate that ``model`` matches the base model saved in checkpoint metadata."""
    if not strategy_metadata.get("compute_base_fingerprints", False):
        return
    raw_fingerprints = strategy_metadata.get("base_model_fingerprints")
    if not isinstance(raw_fingerprints, Mapping):
        raise ValueError(
            "PEFT checkpoint strategy metadata is missing base_model_fingerprints."
        )
    saved_fingerprint = raw_fingerprints.get(model_name)
    if not isinstance(saved_fingerprint, str) or not saved_fingerprint:
        raise ValueError(
            f"PEFT checkpoint is missing a base fingerprint for model {model_name!r}."
        )
    from nvalchemi.training.peft import _peft

    current_fingerprint = _peft.compute_base_fingerprint(model)
    if current_fingerprint == saved_fingerprint:
        return
    message = (
        f"PEFT checkpoint base fingerprint mismatch for model {model_name!r}: "
        f"saved {saved_fingerprint!r}, current {current_fingerprint!r}."
    )
    if strict:
        raise ValueError(message)
    warnings.warn(message, UserWarning, stacklevel=2)


def _apply_checkpoint_module_patches(
    model: nn.Module,
    strategy_metadata: Mapping[str, Any],
    *,
    model_name: str,
    import_path_validator: ImportPathValidator,
) -> None:
    """Apply serializable module patches saved for ``model_name``."""
    raw_patches = strategy_metadata.get("module_patches", {})
    if not raw_patches:
        return
    if not isinstance(raw_patches, Mapping):
        raise ValueError(
            "PEFT checkpoint strategy metadata module_patches must be an object."
        )
    prefix = f"{model_name}."
    patch_targets = tuple(
        sorted(target for target in raw_patches if target.startswith(prefix))
    )
    patches = {}
    for target in patch_targets:
        raw_spec = raw_patches[target]
        if not isinstance(raw_spec, Mapping):
            raise ValueError(
                f"PEFT checkpoint module patch {target!r} must be a BaseSpec object."
            )
        _validate_base_spec_import_paths(
            raw_spec,
            import_path_validator=import_path_validator,
            context=f"PEFT checkpoint module patch {target!r}",
        )
        patches[target] = create_model_spec_from_json(dict(raw_spec))
    if patches:
        ModulePatchHook(patches=patches, register_parameters=False).on_register(
            SimpleNamespace(models={model_name: model})
        )
