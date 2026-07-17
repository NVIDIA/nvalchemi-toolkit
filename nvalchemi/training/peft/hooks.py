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
"""LoRA fine-tuning hooks."""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from enum import Enum
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict, Field
from torch import nn

import nvalchemi.training.peft._adapter_checkpoint as adapter_checkpoint
from nvalchemi.hooks._context import HookContext, TrainContext
from nvalchemi.training._checkpoint import (
    _create_checkpoint_snapshot,
    _write_checkpoint_snapshot,
)
from nvalchemi.training.hooks.checkpoint import CheckpointHook
from nvalchemi.training.hooks.finetune import _matched_names
from nvalchemi.training.optimizers import iter_qualified_named_parameters
from nvalchemi.training.peft import _peft
from nvalchemi.training.peft.wrappers import (
    LoRAWrappableLayer,
    LoRAWrapper,
    LoRAWrapperRegistration,
    LoRAWrapperRegistrations,
)

__all__ = [
    "LoRAWrapper",
    "LoRAApplyHook",
    "LoRACheckpointHook",
    "LoRATrainableParameterHook",
    "LoRAWrapperRegistration",
    "LoRAWrapperRegistrations",
    "LoRAWrappableLayer",
]


def _lora_metadata_trainable_parameter_names(
    models: Mapping[str, nn.Module],
    metadata: Mapping[str, Any],
    names: set[str],
) -> set[str]:
    """Return LoRA trainable names for LoRATrainableParameterHook from adapter metadata stored in LoRAApplyHook.
    
    Parameters
    ----------
    models : Mapping[str, nn.Module]
        Mapping of model names to models.
    metadata : Mapping[str, Mapping[str, Any]]
        Mapping of model names to LoRA metadata dictionaries.
    names : set[str]
        Set of all qualified parameter names.

    Returns
    -------
    set[str]
        Set of LoRA trainable names.
    """
    allowed: set[str] = set()
    for model_name in models:
        model_metadata = metadata.get(model_name)

        # LoRAApplyHook writes one metadata mapping per model.
        # Skip non-conforming or missing entries defensively so explicit trainable sources can still apply.
        if not isinstance(model_metadata, Mapping):
            continue
        trainable_names = model_metadata.get("trainable_names")
        if trainable_names is None:
            continue
        if not isinstance(trainable_names, (list, tuple, set)):
            raise TypeError(
                f"trainable_names must be a list, tuple, or set, got {type(trainable_names)}"
            )

        prefix = f"{model_name}."
        missing: list[str] = []
        for name in trainable_names:
            # LoRAApplyHook stores string parameter names. Ignore other types of entries defensively
            # if metadata was modified.
            if not isinstance(name, str):
                continue
            # Adapter application reports model-local names; nvalchemi filters
            # use model-qualified names.
            qualified_name = f"{prefix}{name}"
            if qualified_name in names:
                allowed.add(qualified_name)
            else:
                missing.append(qualified_name)
        if missing:
            raise RuntimeError(
                "LoRA metadata references trainable parameter(s) that are not "
                f"present on model {model_name!r}: {sorted(missing)!r}."
            )

    return allowed


def _filter_trainable_matches_by_lora_modules(
    candidate_names: set[str],
    models: Mapping[str, nn.Module],
    metadata: Mapping[str, Any],
) -> set[str]:
    """Return candidate names outside metadata-recorded LoRA modules."""
    # Build set of LoRA module prefixes from metadata.
    lora_prefixes: set[str] = set()
    for model_name in models:
        model_metadata = metadata.get(model_name)
        if not isinstance(model_metadata, Mapping):
            continue
        lora_modules = model_metadata.get("lora_modules", ())
        if not isinstance(lora_modules, (list, tuple, set)):
            raise TypeError(
                f"lora_modules must be a list, tuple, or set, got {type(lora_modules)}"
            )
        for module_name in lora_modules:
            if not isinstance(module_name, str):
                continue
            module_prefix = f"{module_name}." if module_name else ""
            lora_prefixes.add(f"{model_name}.{module_prefix}")

    # Only add names that are not owned by LoRA wrapper modules.
    filtered: set[str] = set()
    for name in candidate_names:
        if any(name.startswith(prefix) for prefix in lora_prefixes):
            continue
        filtered.add(name)
    return filtered


def _lora_config_metadata(lora_config: Any) -> dict[str, Any]:
    """Return JSON-compatible LoRA config metadata."""
    if isinstance(lora_config, dict):
        metadata = dict(lora_config)
    elif is_dataclass(lora_config):
        metadata = asdict(lora_config)
    else:
        raise TypeError(
            "LoRA config metadata must be built from a dict or dataclass instance."
        )
    metadata.pop("target_filter", None)
    if callable(metadata.get("init")):
        metadata.pop("init")
    return {key: value for key, value in metadata.items() if value is not None}


def _lora_apply_metadata(
    model: Any,
    result: Any,
    lora_config: Any | None = None,
) -> dict[str, Any]:
    """Build adapter metadata for a model after LoRA injection."""
    model_lora_config = getattr(model, "_lora_config", None)
    if isinstance(model_lora_config, dict):
        metadata = dict(model_lora_config)
    elif lora_config is not None:
        metadata = _lora_config_metadata(lora_config)
    else:
        raise ValueError("LoRA adapter metadata is missing from the wrapped model.")
    wrapped_module_names = [
        name for name, module in model.named_modules() if _peft.is_lora_layer(module)
    ]
    trainable_names = getattr(result, "trainable_names", [])
    if not isinstance(trainable_names, (list, tuple, set)):
        trainable_names = []
    if not all(isinstance(name, str) for name in trainable_names):
        raise TypeError("LoRA apply result trainable_names must contain only strings.")
    trainable_names = list(trainable_names)

    metadata.update(
        {
            "lora_modules": wrapped_module_names,
            "trainable_names": trainable_names,
            "base_model_fingerprint": getattr(result, "base_fingerprint", ""),
            "n_wrapped": getattr(result, "n_wrapped", len(wrapped_module_names)),
            "n_trainable": getattr(result, "n_trainable", None),
            "n_frozen": getattr(result, "n_frozen", None),
        }
    )
    return {key: value for key, value in metadata.items() if value is not None}


class LoRAApplyHook(BaseModel):
    """Apply LoRA adapters to models.
    This hook is automatically prepended by :class:`LoRAFineTuningStrategy`.

    Parameters
    ----------
    lora_config : LoRAConfig
        Expected LoRA configuration applied to each model in the workflow.
    wrapper_registrations : LoRAWrapperRegistrations
        Expected custom layer-to-wrapper registrations applied before adapter
        injection.

    Notes
    -----
    Although ``lora_config`` and ``wrapper_registrations`` use broad typing,
    their expected types are still ``LoRAConfig`` and
    ``LoRAWrapperRegistrations``. The public configuration is validated by
    ``LoRAFineTuningStrategy`` before creating this hook, and the broad typing
    avoids forcing Pydantic to rebuild schemas for PhysicsNeMo LoRA classes.

    Attributes
    ----------
     frequency : int
        Required by the hook protocol; always ``1``.
    stage : None
        This hook does not run at training stages.
    """

    lora_config: Any
    wrapper_registrations: tuple[tuple[type[Any], type[Any]], ...] = ()

    frequency: ClassVar[int] = 1
    stage: ClassVar[None] = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _runs_on_stage(self, stage: Enum) -> bool:
        """Return ``False`` because LoRA injection runs on registration."""
        return False

    def __call__(self, ctx: HookContext, stage: Enum) -> None:
        """No-op stage hook; LoRA injection is handled by :meth:`on_register`."""
        return

    def on_register(self, workflow: Any) -> None:
        """Register wrappers, inject adapters, and store apply results."""

        from nvalchemi.training.peft.wrappers import register_builtin_lora_wrappers

        # Validate workflow
        models = getattr(workflow, "models", None)
        if not isinstance(models, Mapping):
            raise TypeError("LoRAApplyHook requires a workflow with a models mapping.")
        if getattr(workflow, "_optimizers", None) or getattr(
            workflow, "_flat_opts", None
        ):
            raise RuntimeError(
                "LoRAApplyHook must be registered before optimizers are built."
            )

        # Register built-in lora and user-defined wrappers
        register_builtin_lora_wrappers()
        for layer_cls, wrapper_cls in self.wrapper_registrations:
            _peft.register_lora_wrapper(layer_cls, wrapper_cls)

        if getattr(self.lora_config, "extras_trainable", []):
            raise ValueError(
                "LoRAFineTuningStrategy does not support LoRAConfig's extras_trainable. "
                "Use trainable_patterns or module_patches to select extra trainable parameters."
            )

        # Apply LoRA adapters to models and store metadata
        metadata: dict[str, dict[str, Any]] = {}
        for model_name, model in models.items():
            result: _peft.ApplyResult = _peft.apply_lora(model, self.lora_config)
            metadata[model_name] = _lora_apply_metadata(
                model,
                result,
                self.lora_config,
            )

        setattr(workflow, "_lora_metadata", metadata)


class LoRATrainableParameterHook(BaseModel):
    """Identify trainable parameters for LoRA fine-tuning.
    In particular, this hook identifies LoRA adapters, patched modules, and explicitly opt-in parameters,
    and sets them as trainable parameters for optimization. All other parameters are set as frozen.
    This hook is automatically prepended by :class:`LoRAFineTuningStrategy` after adapter injection and
    optional module patching.

    Parameters
    ----------
    trainable_patterns : tuple[str, ...]
        Glob patterns for parameter names to include in training.
    patched_module_paths : tuple[str, ...]
        Module paths to include in training. This is used to identify modules that have been patched
        and should be included in training.

    Attributes
    ----------
    frequency : int
        Required by the hook protocol; always ``1``.
    stage : None
        This hook does not run at training stages.
    """

    trainable_patterns: tuple[str, ...] = ()
    patched_module_paths: tuple[str, ...] = ()

    frequency: ClassVar[int] = 1
    stage: ClassVar[None] = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _runs_on_stage(self, stage: Enum) -> bool:  # noqa: ARG002
        """Return ``False`` because everything runs on registration."""
        return False

    def __call__(self, ctx: HookContext, stage: Enum) -> None:  # noqa: ARG002
        """No-op stage hook; everything is handled by :meth:`on_register`."""
        return

    def on_register(self, workflow: Any) -> None:
        """Store LoRA adapter, patched, and other trainable parameters on ``workflow``."""
        models = getattr(workflow, "models", None)
        if not isinstance(models, Mapping):
            raise TypeError(
                "LoRATrainableParameterHook requires a workflow with a models mapping."
            )

        named_parameters = dict(iter_qualified_named_parameters(models))
        names = set(named_parameters)
        lora_metadata = getattr(workflow, "_lora_metadata", {}) or {}
        if not isinstance(lora_metadata, Mapping):
            lora_metadata = {}

        # Get trainable names from LoRA metadata
        lora_matches = _lora_metadata_trainable_parameter_names(
            models, lora_metadata, names
        )

        # Get trainable names from trainable_patterns, excluding LoRA module
        # parameters because they contain LoRA adapters and wrapped base parameters that
        # should stay frozen.
        trainable_matches = _matched_names(
            self.trainable_patterns,
            names,
            label="trainable_patterns",
        )
        extra_trainable_matches = _filter_trainable_matches_by_lora_modules(
            trainable_matches,
            models,
            lora_metadata,
        )

        # Get trainable names from patched_module_paths
        patched_matches = {
            name
            for name in names
            if any(name.startswith(f"{path}.") for path in self.patched_module_paths)
        }

        # Keep export categories mutually exclusive.
        # User-defined trainable_patterns may match patched modules by accident.
        extra_trainable_matches -= patched_matches

        # Update overall allowed trainable names
        allowed = set(lora_matches)
        allowed.update(extra_trainable_matches)
        allowed.update(patched_matches)

        # Validate that allowed trainable names are not empty
        if not allowed:
            raise ValueError(
                "LoRA trainable parameter filtering selected no parameters. "
                "Check LoRA adapter injection, module_patches, or trainable_patterns."
            )

        # Validate that optimizers were not built yet
        if getattr(workflow, "_optimizers", None) or getattr(
            workflow, "_flat_opts", None
        ):
            warnings.warn(
                "LoRATrainableParameterHook registered after optimizers were built; "
                "existing optimizer parameter groups are unchanged until the "
                "strategy builds optimizers again.",
                UserWarning,
                stacklevel=2,
            )

        # Set trainable parameter filters
        for method_name in (
            "set_optimizer_parameter_filter",
            "set_trainable_parameter_filter",
            "set_force_trainable_parameter_filter",
        ):
            method = getattr(workflow, method_name, None)
            if not callable(method):
                raise TypeError(
                    "LoRATrainableParameterHook requires a workflow with a "
                    f"{method_name}(names) method."
                )
        workflow.set_optimizer_parameter_filter(allowed)
        workflow.set_trainable_parameter_filter(allowed)
        workflow.set_force_trainable_parameter_filter(allowed)

        # Set trainable parameter names as immutable sets for export compatibility
        workflow._lora_adapter_parameter_names = frozenset(lora_matches)
        workflow._extra_trainable_parameter_names = frozenset(extra_trainable_matches)
        workflow._patched_parameter_names = frozenset(patched_matches)


class LoRACheckpointHook(CheckpointHook):
    """Save restartable LoRA checkpoints with trainable model state by default."""

    include_base_parameters: bool = Field(
        default=False,
        description="Save full base model parameters instead of trainable LoRA state.",
    )

    def _save_checkpoint(self, ctx: TrainContext) -> None:
        """Capture and write one full or trainable-only LoRA checkpoint."""
        if self.include_base_parameters:
            super()._save_checkpoint(ctx)
            return
        if ctx.workflow is None:
            raise RuntimeError(
                "LoRACheckpointHook requires TrainContext.workflow to reference "
                "the active LoRAFineTuningStrategy."
            )
        self._finish_pending(block=False)
        if self._future is not None:
            self._finish_pending(block=True)

        snapshot = _create_checkpoint_snapshot(
            self.checkpoint_dir,
            strategy=ctx.workflow,
        )
        # When include_base_parameters is False, the only change is that we only save trainable parameters
        # instead of all parameters.
        adapter_checkpoint.filter_lora_snapshot(snapshot, ctx.workflow)
        if not self.async_save:
            self.last_checkpoint_index = _write_checkpoint_snapshot(
                self.checkpoint_dir,
                snapshot,
            )
            return

        if self._executor is None:
            raise RuntimeError(
                "LoRACheckpointHook async writer is not initialized. Run it "
                "through TrainingStrategy or call __enter__() before invoking it."
            )
        self._future = self._executor.submit(
            _write_checkpoint_snapshot,
            self.checkpoint_dir,
            snapshot,
        )
