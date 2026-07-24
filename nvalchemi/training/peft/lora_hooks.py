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

from collections.abc import Mapping
from enum import Enum
from typing import Any, ClassVar, Final

from pydantic import BaseModel, ConfigDict, Field

from nvalchemi.hooks._context import HookContext
from nvalchemi.training.hooks.finetune import _matched_names
from nvalchemi.training.peft import _peft
from nvalchemi.training.peft.wrappers import (
    LoRAWrappableLayer,
    LoRAWrapper,
    LoRAWrapperRegistrations,
)

__all__ = [
    "LoRAWrapper",
    "LoRAApplyHook",
    "LoRAWrapperRegistrations",
    "LoRAWrappableLayer",
]

_LORA_HOOK_IDENTIFIER: Final = "lora"


def _collate_lora_metadata(
    model_name: str,
    model: Any,
    result: _peft.ApplyResult,
) -> tuple[str, set[str], set[str]]:
    """Return base fingerprint, trainable names, and managed names for a model."""
    wrapped_module_names = [
        name for name, module in model.named_modules() if _peft.is_lora_layer(module)
    ]
    model_parameter_names = {name for name, _parameter in model.named_parameters()}
    trainable_names = getattr(result, "trainable_names", None)
    if trainable_names is None:
        trainable_names = [
            name
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        ]
    if not isinstance(trainable_names, (list, tuple, set)):
        raise TypeError(
            "LoRA apply result trainable_names must be a list, tuple, or set, "
            f"got {type(trainable_names).__name__}."
        )
    if not all(isinstance(name, str) for name in trainable_names):
        raise TypeError("LoRA apply result trainable_names must contain only strings.")
    trainable_names = sorted(set(trainable_names))
    missing = sorted(set(trainable_names) - model_parameter_names)
    if missing:
        raise RuntimeError(
            "LoRA apply result references trainable parameter(s) that are not "
            f"present on model {model_name!r}: "
            f"{[f'{model_name}.{name}' for name in missing]!r}."
        )
    qualified_trainable_names = {f"{model_name}.{name}" for name in trainable_names}
    qualified_managed_names: set[str] = set()
    for module_name in wrapped_module_names:
        prefix = f"{module_name}." if module_name else ""
        qualified_managed_names.update(
            f"{model_name}.{name}"
            for name in model_parameter_names
            if name.startswith(prefix)
        )
    return (
        getattr(result, "base_fingerprint", ""),
        qualified_trainable_names,
        qualified_managed_names,
    )


class LoRAApplyHook(BaseModel):
    """Apply LoRA adapters to models.

    This hook is automatically prepended by :class:`~nvalchemi.training.LoRAFineTuningStrategy`.

    Parameters
    ----------
    lora_target_patterns : tuple[str, ...]
        Shell-style glob patterns matched against model-prefixed module names,
        using the same ``*``, ``?``, and ``[...]`` syntax as
        ``trainable_patterns``. Dots are literal path separators. For example,
        ``"main.model.projection"`` selects exactly
        ``"main.model.projection"``, ``"student.model.*projection"`` selects
        projection-like modules under ``student.model``, and
        ``"main.model.readout*"`` selects modules whose final path component
        starts with ``readout``. Patterns without glob characters are exact
        matches.
    lora_rank : int
        Rank of the low-rank adapter factors.
    lora_alpha : float, optional
        Scaling numerator for adapter updates. Defaults to ``1.0``.
    lora_dropout : float, optional
        Dropout probability on the adapter input path. Defaults to ``0.0``.
    lora_wrap_mlp : bool, optional
        Also target supported feed-forward sub-blocks discovered by the adapter
        implementation. This is only supported for single-model strategies.
        Defaults to ``False``.
    wrapper_registrations : LoRAWrapperRegistrations, optional
        Custom layer-to-wrapper registrations installed before adapter
        injection. Each pair maps a base layer class to the adapter wrapper
        class that should handle it. Defaults to ``()``.

    Notes
    -----
    ``wrapper_registrations`` is typed broadly on the Pydantic field because it
    stores runtime class objects. The expected public shape is still
    ``LoRAWrapperRegistrations``, and ``LoRAFineTuningStrategy`` validates it
    before creating this hook.

    Attributes
    ----------
    frequency : int
        Required by the hook protocol; always ``1``.
    stage : None
        This hook does not run at training stages.
    Examples
    --------
    Exact module names can be written as patterns without glob characters:

    >>> hook = LoRAApplyHook(
    ...     lora_target_patterns=("main.model.projection",),
    ... )

    Glob patterns match module names from ``model.named_modules()``, prefixed with
    the model key such as ``"main"``:

    >>> hook = LoRAApplyHook(
    ...     lora_target_patterns=("main.model.*projection",),
    ...     lora_rank=4,
    ...     lora_alpha=1.0,
    ... )

    In multi-model workflows, the prefix helps to identify which model receives adapters:

    >>> hook = LoRAApplyHook(
    ...     lora_target_patterns=("student.model.projection",),
    ... )
    """

    lora_target_patterns: tuple[str, ...] = Field(min_length=1)
    lora_rank: int = 8
    lora_alpha: float = 1.0
    lora_dropout: float = 0.0
    lora_wrap_mlp: bool = False
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
        """Register wrappers, inject adapters, and store base fingerprints.

        The public configuration is validated by ``LoRAFineTuningStrategy``
        before creating this hook. During registration, model-prefixed targets
        are converted to model-local targets for each model to be compatible
        with the PhysicsNeMo ``apply_lora`` function. For example,
        ``"student.model.projection"`` becomes the PhysicsNeMo
        ``target_modules`` entry ``"model.projection"`` for the ``"student"``
        model. Adapters are then injected, and their trainable and managed
        parameter names are registered on the workflow.
        """

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

        # Register built-in and user-defined LoRA wrappers
        register_builtin_lora_wrappers()
        for layer_cls, wrapper_cls in self.wrapper_registrations:
            _peft.register_lora_wrapper(layer_cls, wrapper_cls)

        # Apply LoRA to models and collect registration data.
        # Adapter selectors operate on one model at a time. Convert
        # model-prefixed selectors such as "student.model.projection" to
        # model-local selectors before delegating to the PhysicsNeMo
        # ``apply_lora`` function.
        base_fingerprints: dict[str, str] = {}

        # Get all module names in the models.
        model_names = set(models)
        module_names = {
            f"{model_name}.{name}"
            for model_name, model in models.items()
            for name, _module in model.named_modules()
            if name
        }

        # Identify module names that match the LoRA target patterns.
        matched_module_names = tuple(
            sorted(
                _matched_names(
                    self.lora_target_patterns,
                    module_names,
                    label="LoRA target",
                    target_type="module",
                )
            )
        )

        # Apply LoRA to each model.
        lora_trainable_names: set[str] = set()
        lora_managed_names: set[str] = set()
        for model_name, model in models.items():
            # Identify module names that are local to the model and match the LoRA target patterns.
            local_targets: list[str] = []
            for target in matched_module_names:
                prefix, separator, module_name = target.partition(".")
                if not separator:
                    raise ValueError(
                        f"LoRA target module {target!r} must include a model prefix, "
                        "for example 'main.model.projection'."
                    )
                if prefix not in model_names:
                    raise KeyError(
                        f"LoRA target module {target!r} references unknown model "
                        f"{prefix!r}; available models: {sorted(model_names)}."
                    )
                if prefix == model_name:
                    local_targets.append(module_name)

            # Reconstruct the LoRA configuration for this model.
            model_lora_config = _peft.LoRAConfig(
                rank=self.lora_rank,
                alpha=self.lora_alpha,
                target_modules=local_targets or ["__nvalchemi_no_lora_target__"],
                lora_dropout=self.lora_dropout,
                extras_trainable=[],
                wrap_mlp=self.lora_wrap_mlp,
                init="default",
            )

            # If no target modules are found, record only the base fingerprint for this model.
            if not local_targets:
                base_fingerprints[model_name] = _peft.compute_base_fingerprint(model)
                continue

            # Apply LoRA to the model and collect names to register.
            result: _peft.ApplyResult = _peft.apply_lora(model, model_lora_config)
            (
                base_fingerprints[model_name],
                model_trainable_names,
                model_managed_names,
            ) = _collate_lora_metadata(
                model_name,
                model,
                result,
            )
            lora_trainable_names.update(model_trainable_names)
            lora_managed_names.update(model_managed_names)

        # Store only the base fingerprints needed for checkpoint/spec validation.
        workflow._base_fingerprints = base_fingerprints

        # Register LoRA trainable and managed parameter names on the workflow.
        # Trainable parameters refer to the trainable adapter parameters.
        # Managed parameters refer to the parameters of the underlying modules modified by LoRA.
        # These managed parameters are protected from being overridden by other hooks modifying the models.
        for method_name in (
            "register_trainable_parameter_names",
            "register_managed_parameter_names",
        ):
            method = getattr(workflow, method_name, None)
            if not callable(method):
                raise TypeError(
                    "LoRAApplyHook requires a workflow with a "
                    f"{method_name}(names, source=...) method."
                )
        workflow.register_trainable_parameter_names(
            tuple(sorted(lora_trainable_names)),
            source=_LORA_HOOK_IDENTIFIER,
        )
        workflow.register_managed_parameter_names(
            tuple(sorted(lora_managed_names)),
            source=_LORA_HOOK_IDENTIFIER,
        )
