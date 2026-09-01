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

from pydantic import BaseModel, ConfigDict

from nvalchemi.hooks._context import HookContext
from nvalchemi.training.hooks.finetune import _matched_names
from nvalchemi.training.peft import _peft
from nvalchemi.training.peft.lora import LORA_PEFT_METHOD, LoRAConfig
from nvalchemi.training.peft.lora_wrappers import (
    LoRAWrappableLayer,
    LoRAWrapper,
    LoRAWrapperRegistrations,
    _temporary_lora_wrapper_registrations,
)

__all__ = [
    "LoRAWrapper",
    "LoRAHook",
    "LoRAWrapperRegistrations",
    "LoRAWrappableLayer",
]

_LORA_PARAMETER_SOURCE: Final = LORA_PEFT_METHOD


def _collate_lora_metadata(
    model_name: str,
    model: Any,
    result: _peft.ApplyResult,
) -> tuple[set[str], set[str]]:
    """Return trainable and managed names for a model."""
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
    return qualified_trainable_names, qualified_managed_names


class LoRAHook(BaseModel):
    """Apply LoRA adapters to models.

    This hook is automatically prepended when
    :class:`~nvalchemi.training.FineTuningStrategy` receives a
    :class:`~nvalchemi.training.peft.lora.LoRAConfig`.

    Parameters
    ----------
    lora_config : LoRAConfig
        LoRA PEFT configuration describing adapter targets, rank, scaling,
        dropout, MLP wrapping, and custom wrapper registrations.
    register_parameters : bool, optional
        If ``True``, register LoRA adapter parameters as trainable and managed
        on the workflow. Set to ``False`` for standalone model loading where no
        training workflow registry exists. Defaults to ``True``.

    Attributes
    ----------
    frequency : int
        Required by the hook protocol; always ``1``.
    stage : None
        This hook does not run at training stages.
    """

    lora_config: LoRAConfig
    register_parameters: bool = True

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
        """Register wrappers, inject adapters, and register adapter parameters.

        The public configuration is validated before creating this hook. During registration, model-prefixed targets
        are converted to model-local targets for each model to be compatible
        with the PhysicsNeMo ``apply_lora`` function. For example,
        ``"student.model.projection"`` becomes the PhysicsNeMo
        ``target_modules`` entry ``"model.projection"`` for the ``"student"``
        model. Adapters are then injected, and their trainable and managed
        parameter names are registered on the workflow.
        """

        # Validate workflow
        models = getattr(workflow, "models", None)
        if not isinstance(models, Mapping):
            raise TypeError("LoRAHook requires a workflow with a models mapping.")
        if getattr(workflow, "_optimizers", None) or getattr(
            workflow, "_flat_opts", None
        ):
            raise RuntimeError(
                "LoRAHook must be registered before optimizers are built."
            )

        # Apply LoRA to models and collect registration data.
        # Adapter selectors operate on one model at a time. Convert
        # model-prefixed selectors such as "student.model.projection" to
        # model-local selectors before delegating to the PhysicsNeMo
        # ``apply_lora`` function.
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
                    self.lora_config.lora_target_patterns,
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
            model_lora_config = _peft.PhysicsNeMoLoRAConfig(
                rank=self.lora_config.rank,
                alpha=self.lora_config.alpha,
                target_modules=local_targets or ["__nvalchemi_no_lora_target__"],
                lora_dropout=self.lora_config.lora_dropout,
                extras_trainable=[],
                wrap_mlp=self.lora_config.wrap_mlp,
                init="default",  # hardcode adapter initialization for now
            )

            if not local_targets:
                continue

            # Apply LoRA to the model and collect names to register.
            with _temporary_lora_wrapper_registrations(
                self.lora_config.wrapper_registrations or ()
            ):
                result: _peft.ApplyResult = _peft.apply_lora(
                    model, model_lora_config, compute_fingerprint=False
                )
            (
                model_trainable_names,
                model_managed_names,
            ) = _collate_lora_metadata(
                model_name,
                model,
                result,
            )
            lora_trainable_names.update(model_trainable_names)
            lora_managed_names.update(model_managed_names)

        if not self.register_parameters:
            return

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
                    "LoRAHook requires a workflow with a "
                    f"{method_name}(names, source=...) method."
                )
        workflow.register_trainable_parameter_names(
            tuple(sorted(lora_trainable_names)),
            source=_LORA_PARAMETER_SOURCE,
        )
        workflow.register_managed_parameter_names(
            tuple(sorted(lora_managed_names)),
            source=_LORA_PARAMETER_SOURCE,
        )
