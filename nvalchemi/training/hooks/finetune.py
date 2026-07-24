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
"""Fine-tuning hooks for module patching and optimizer parameter filtering."""

from __future__ import annotations

import fnmatch
import warnings
from collections.abc import Mapping
from enum import Enum
from typing import Any, ClassVar, Literal, TypeAlias

import torch
from pydantic import BaseModel, ConfigDict, Field

from nvalchemi.hooks._context import HookContext
from nvalchemi.training._spec import BaseSpec
from nvalchemi.training.optimizers import iter_qualified_named_parameters

__all__ = ["ModulePatchHook", "TrainableParameterHook"]


PatchValue = BaseSpec | torch.nn.Module
"""Supported replacement value for :class:`ModulePatchHook`."""

FreezeMode: TypeAlias = Literal["requires_grad", "optimizer_only"]
"""Supported parameter-freezing modes for :class:`TrainableParameterHook`."""


def _matched_names(
    patterns: tuple[str, ...],
    names: set[str],
    *,
    label: str,
    target_type: Literal["parameter", "module"] = "parameter",
) -> set[str]:
    """Return names matched by glob patterns, raising on empty matches."""
    example = {
        "parameter": "main.model.projection.weight",
        "module": "main.model.projection",
    }[target_type]
    matched: set[str] = set()
    for pattern in patterns:
        pattern_matches = {name for name in names if fnmatch.fnmatchcase(name, pattern)}
        if not pattern_matches:
            raise ValueError(
                f"{label} pattern {pattern!r} did not match any {target_type}. "
                "Patterns are matched against fully-qualified names like "
                f"{example!r}."
            )
        matched.update(pattern_matches)
    return matched


def _resolve_parent(
    models: Mapping[str, torch.nn.Module],
    target: str,
) -> tuple[torch.nn.Module, str]:
    """Resolve a module patch target to ``(parent_module, child_name)``."""
    parts = target.split(".")
    if len(parts) < 2 or any(part == "" for part in parts):
        raise ValueError(
            f"Module patch target {target!r} must be '<model_key>.<path>.<child>'."
        )
    model_key, *module_parts = parts
    if model_key not in models:
        raise KeyError(
            f"Module patch target {target!r} references unknown model "
            f"{model_key!r}; available models: {sorted(models)}."
        )
    parent: torch.nn.Module = models[model_key]
    for part in module_parts[:-1]:
        try:
            next_parent = getattr(parent, part)
        except AttributeError as exc:
            raise AttributeError(
                f"Module patch target {target!r} has missing parent component {part!r}."
            ) from exc
        if not isinstance(next_parent, torch.nn.Module):
            raise TypeError(
                f"Module patch target {target!r} parent component {part!r} "
                f"resolved to {type(next_parent).__name__}, expected nn.Module."
            )
        parent = next_parent
    return parent, module_parts[-1]


def _build_patch_module(target: str, value: PatchValue) -> torch.nn.Module:
    """Build or validate a module patch value."""
    if isinstance(value, BaseSpec):
        value = value.build()
    if not isinstance(value, torch.nn.Module):
        raise TypeError(
            f"Module patch target {target!r} must build or provide an "
            f"nn.Module; got {type(value).__name__}."
        )
    return value


def _parameter_names_under_prefix(names: set[str], prefix: str) -> set[str]:
    """Return parameter names directly under a fully-qualified module prefix."""
    parameter_prefix = f"{prefix}."
    return {name for name in names if name.startswith(parameter_prefix)}


def _registered_parameter_names(workflow: Any, method_name: str) -> frozenset[str]:
    """Return registered parameter names when the workflow supports a registry."""
    method = getattr(workflow, method_name, None)
    if not callable(method):
        return frozenset()
    return method()


def _validate_registered_parameter_names(
    names: set[str],
    registered_names: set[str],
    *,
    label: str,
) -> None:
    """Raise when registry names no longer exist on the model."""
    missing = sorted(registered_names - names)
    if missing:
        raise RuntimeError(
            f"Registered {label} parameter(s) are not present on the final "
            f"model: {missing!r}."
        )


class ModulePatchHook(BaseModel):
    """Patch model submodules at registration time.

    Patches run when the hook is registered on a workflow. Each target path
    must include the model key followed by an existing parent path and a final
    child attribute, for example ``"main.model.projection"``. The parent module
    must exist; the final child is added when missing or replaced when present.
    Shape compatibility is intentionally user-owned and is validated naturally
    by the model's forward pass or downstream checkpoint loading.

    Parameters
    ----------
    patches : dict[str, BaseSpec | torch.nn.Module]
        Ordered mapping of target paths to replacement modules or specs that
        build modules.
    register_parameters : bool
        If ``True``, register the patched module parameters as both trainable
        and managed under the fixed ``"module_patch"`` source. Registered
        patch parameters are included in the final trainable allow-list by default 
        and are protected from being overridden by other sources modifying the models.
        Defaults to ``False`` for backward compatibility.

    Attributes
    ----------
    patches : dict[str, BaseSpec | torch.nn.Module]
        Module patches applied in insertion order.
    frequency : int
        Required by the hook protocol; always ``1``.
    stage : None
        This hook does not run at training stages.
    _hook_identifier : str
        Identifier for the hook used to register trainable and managed parameter names
        when ``register_parameters=True``.

    Warns
    -----
    UserWarning
        If the same direct module instance is assigned to multiple targets.

    Examples
    --------
    >>> import torch
    >>> from nvalchemi.training.hooks import ModulePatchHook
    >>> hook = ModulePatchHook(
    ...     patches={"main.model.projection": torch.nn.Linear(8, 1)}
    ... )
    >>> hook.frequency
    1
    """

    patches: dict[str, PatchValue] = Field(default_factory=dict)

    register_parameters: bool = False

    frequency: ClassVar[int] = 1
    stage: ClassVar[None] = None
    _hook_identifier: ClassVar[str] = "patch"

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _runs_on_stage(self, stage: Enum) -> bool:  # noqa: ARG002
        """Return ``False`` because module patches run only on registration."""
        return False

    def __call__(self, ctx: HookContext, stage: Enum) -> None:  # noqa: ARG002
        """No-op stage hook; patching is handled by :meth:`on_register`."""
        return

    def on_register(self, workflow: Any) -> None:
        """Apply all module patches to ``workflow.models``."""
        models = getattr(workflow, "models", None)
        if not isinstance(models, Mapping):
            raise TypeError(
                "ModulePatchHook requires a workflow with a models mapping."
            )
        if getattr(workflow, "_optimizers", None) or getattr(
            workflow, "_flat_opts", None
        ):
            raise RuntimeError(
                "ModulePatchHook must be registered before optimizers are built; "
                "create a new strategy or rebuild optimizer state before patching "
                "model modules."
            )

        direct_module_targets: dict[int, list[str]] = {}
        for target, value in self.patches.items():
            if isinstance(value, torch.nn.Module):
                direct_module_targets.setdefault(id(value), []).append(target)
        for targets in direct_module_targets.values():
            if len(targets) > 1:
                warnings.warn(
                    "The same nn.Module instance is patched into multiple "
                    f"targets {targets}; parameters will be shared.",
                    UserWarning,
                    stacklevel=2,
                )

        # Get the previously registered managed parameter names on the workflow.
        registered_managed_names: Mapping[str, frozenset[str]] = {}
        get_registered_managed = getattr(
            workflow,
            "get_registered_managed_parameter_names",
            None,
        )
        if callable(get_registered_managed):
            registered_managed_names = get_registered_managed()

        resolved: list[tuple[str, torch.nn.Module, str, torch.nn.Module]] = []
        for target, value in self.patches.items():
            # Check if the patch target overlaps with any other registered managed parameter names.
            overlaps: dict[str, list[str]] = {}
            for source, source_names in registered_managed_names.items():
                if source == self._hook_identifier:
                    continue
                matched_names = _parameter_names_under_prefix(set(source_names), target)
                if matched_names:
                    overlaps[source] = sorted(matched_names)
            if overlaps:
                raise RuntimeError(
                    f"Module patch target {target!r} overlaps parameter(s) "
                    f"already managed by another source: {overlaps!r}."
                )

            parent, child_name = _resolve_parent(models, target)
            if hasattr(parent, child_name):
                existing = getattr(parent, child_name)
                if not isinstance(existing, torch.nn.Module):
                    raise TypeError(
                        f"Module patch target {target!r} would replace "
                        f"{type(existing).__name__}, expected an existing "
                        "nn.Module or a new child name."
                    )
            resolved.append(
                (target, parent, child_name, _build_patch_module(target, value))
            )

        for _target, parent, child_name, module in resolved:
            setattr(parent, child_name, module)

        if self.register_parameters:
            names = {name for name, _ in iter_qualified_named_parameters(models)}
            patch_parameter_names = set().union(
                *(
                    _parameter_names_under_prefix(names, target)
                    for target, *_rest in resolved
                )
            )
            register_trainable = getattr(
                workflow,
                "register_trainable_parameter_names",
                None,
            )
            if not callable(register_trainable):
                raise TypeError(
                    "ModulePatchHook requires a workflow with a "
                    "register_trainable_parameter_names(names, source=...) method "
                    "when register_parameters=True."
                )
            register_trainable(
                tuple(sorted(patch_parameter_names)),
                source=self._hook_identifier,
            )
            register_managed = getattr(
                workflow,
                "register_managed_parameter_names",
                None,
            )
            if not callable(register_managed):
                raise TypeError(
                    "ModulePatchHook requires a workflow with a "
                    "register_managed_parameter_names(names, source=...) method "
                    "when register_parameters=True."
                )
            register_managed(
                tuple(sorted(patch_parameter_names)),
                source=self._hook_identifier,
            )
            patched_parameter_names = set(
                getattr(workflow, "_patched_parameter_names", set())
            )
            patched_parameter_names.update(patch_parameter_names)
            workflow._patched_parameter_names = frozenset(patched_parameter_names)


class TrainableParameterHook(BaseModel):
    """Select trainable parameters for fine-tuning.

    The hook computes a fully-qualified parameter allow-list when registered
    on a :class:`~nvalchemi.training.strategy.TrainingStrategy`. By default,
    parameters outside the allow-list are temporarily marked
    ``requires_grad=False`` during ``run`` and restored afterward. Set
    ``freeze_mode="optimizer_only"`` to preserve gradients for excluded
    parameters while keeping them out of optimizer parameter groups.

    Parameters
    ----------
    freeze_patterns : tuple[str, ...]
        Glob patterns to exclude from training. These exclusions are overridden
        by ``trainable_patterns``.
    trainable_patterns : tuple[str, ...]
        Glob patterns to include. When supplied without ``freeze_patterns``,
        these patterns form an allow-list, along with the trainable parameters
        registered by earlier hooks, e.g., ``ModulePatchHook``.
    freeze_mode : {"requires_grad", "optimizer_only"}
        Whether excluded parameters are temporarily frozen via
        ``requires_grad=False`` or only excluded from optimizer construction.

    Attributes
    ----------
    freeze_patterns : tuple[str, ...]
        Exclusion patterns matched against names such as
        ``"main.model.joint_mlp.0.weight"``.
    trainable_patterns : tuple[str, ...]
        Inclusion override patterns matched after exclusions.
    freeze_mode : {"requires_grad", "optimizer_only"}
        Parameter-freezing mode.
    frequency : int
        Required by the hook protocol; always ``1``.
    stage : None
        This hook does not run at training stages.

    Raises
    ------
    ValueError
        If no trainable parameters are selected, or if any pattern matches no
        parameter.

    Warns
    -----
    UserWarning
        If registered after optimizers already exist. The stored filter is
        updated, but existing optimizer parameter groups are not rebuilt.

    Examples
    --------
    >>> from nvalchemi.training.hooks import TrainableParameterHook
    >>> TrainableParameterHook(
    ...     freeze_patterns=("main.model.*",),
    ...     trainable_patterns=("main.model.projection.*",),
    ... ).frequency
    1
    """

    freeze_patterns: tuple[str, ...] = ()
    trainable_patterns: tuple[str, ...] = ()
    freeze_mode: FreezeMode = "requires_grad"

    frequency: ClassVar[int] = 1
    stage: ClassVar[None] = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _runs_on_stage(self, stage: Enum) -> bool:  # noqa: ARG002
        """Return ``False`` because optimizer filters run only on registration."""
        return False

    def __call__(self, ctx: HookContext, stage: Enum) -> None:  # noqa: ARG002
        """No-op stage hook; filtering is handled by :meth:`on_register`."""
        return

    def on_register(self, workflow: Any) -> None:
        """Store the computed optimizer parameter allow-list on ``workflow``."""
        models = getattr(workflow, "models", None)
        if not isinstance(models, Mapping):
            raise TypeError(
                "TrainableParameterHook requires a workflow with a models mapping."
            )

        names = {name for name, _ in iter_qualified_named_parameters(models)}
        freeze_matches: set[str] = set()
        trainable_matches: set[str] = set()
        if self.freeze_patterns:
            freeze_matches = _matched_names(
                self.freeze_patterns, names, label="freeze_patterns"
            )
        if self.trainable_patterns:
            trainable_matches = _matched_names(
                self.trainable_patterns, names, label="trainable_patterns"
            )
        registered_trainable_names = set(
            _registered_parameter_names(
                workflow,
                "get_flattened_registered_trainable_parameter_names",
            )
        )
        registered_managed_names = set(
            _registered_parameter_names(
                workflow,
                "get_flattened_registered_managed_parameter_names",
            )
        )
        _validate_registered_parameter_names(
            names,
            registered_trainable_names,
            label="trainable",
        )
        _validate_registered_parameter_names(
            names,
            registered_managed_names,
            label="managed",
        )
        trainable_matches -= registered_managed_names
        if self.trainable_patterns and not self.freeze_patterns:
            allowed = trainable_matches
        elif not self.trainable_patterns and not self.freeze_patterns:
            allowed = set()
        else:
            allowed = (names - freeze_matches) | trainable_matches
        allowed.update(registered_trainable_names)
        if not allowed:
            raise ValueError(
                "TrainableParameterHook selected no trainable parameters. "
                "Provide freeze_patterns or trainable_patterns, or register "
                "trainable parameters before this hook."
            )

        if getattr(workflow, "_optimizers", None) or getattr(
            workflow, "_flat_opts", None
        ):
            warnings.warn(
                "TrainableParameterHook registered after optimizers were built; "
                "existing optimizer parameter groups are unchanged until the "
                "strategy builds optimizers again.",
                UserWarning,
                stacklevel=2,
            )
        set_optimizer_parameter_filter = getattr(
            workflow, "set_optimizer_parameter_filter", None
        )
        if not callable(set_optimizer_parameter_filter):
            raise TypeError(
                "TrainableParameterHook requires a workflow with a "
                "set_optimizer_parameter_filter(names) method."
            )
        set_optimizer_parameter_filter(allowed)

        # For freeze_mode="requires_grad", preserve requires_grad for parameters in the
        # allowed list and temporarily set all other parameters to requires_grad=False.
        # For freeze_mode="optimizer_only", do not change requires_grad here
        set_trainable_parameter_filter = getattr(
            workflow, "set_trainable_parameter_filter", None
        )
        if not callable(set_trainable_parameter_filter):
            raise TypeError(
                "TrainableParameterHook requires a workflow with a "
                "set_trainable_parameter_filter(names) method."
            )
        if self.freeze_mode == "requires_grad":
            set_trainable_parameter_filter(allowed)
        else:
            set_trainable_parameter_filter(None)

        # Force explicitly selected or registered trainable parameters on to be trainable.
        set_force_trainable_parameter_filter = getattr(
            workflow, "set_force_trainable_parameter_filter", None
        )
        if not callable(set_force_trainable_parameter_filter):
            raise TypeError(
                "TrainableParameterHook requires a workflow with a "
                "set_force_trainable_parameter_filter(names) method."
            )
        set_force_trainable_parameter_filter(
            (trainable_matches | registered_trainable_names)
            if self.trainable_patterns or registered_trainable_names
            else None
        )
