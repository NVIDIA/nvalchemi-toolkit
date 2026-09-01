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
"""LoRA integration for generic fine-tuning PEFT config."""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from types import SimpleNamespace
from typing import Any, Callable, Final, Literal, Self

from pydantic import ConfigDict, Field, ValidationInfo, field_validator, model_validator
from torch import nn

from nvalchemi._serialization import _cls_path_of, _import_cls
from nvalchemi.training import _strategy_validation as strategy_validation
from nvalchemi.training.peft import _peft
from nvalchemi.training.peft import lora_wrappers as _lora_wrappers
from nvalchemi.training.peft.config import PeftConfig
from nvalchemi.training.peft.lora_wrappers import (
    CuEquivarianceLoRALinear,
    E3NNFullyConnectedLoRALayer,
    EquivariantLoRALinear,
    LoRALayer,
    LoRALinear,
    LoRAWrappableLayer,
    LoRAWrapper,
    LoRAWrapperRegistrations,
    available_lora_wrappers,
)
from nvalchemi.training.peft.registry import register_peft_method

__all__ = [
    "CuEquivarianceLoRALinear",
    "E3NNFullyConnectedLoRALayer",
    "EquivariantLoRALinear",
    "LORA_PEFT_METHOD",
    "LoRAConfig",
    "LoRALayer",
    "LoRALinear",
    "LoRAWrappableLayer",
    "LoRAWrapper",
    "LoRAWrapperRegistrations",
    "available_lora_wrappers",
    "is_lora_layer",
    "merge_lora_into_model",
    "register_lora_method",
]

if "TransformerEngineLoRALinear" in _lora_wrappers.__all__:
    TransformerEngineLoRALinear = _lora_wrappers.TransformerEngineLoRALinear
    __all__.append("TransformerEngineLoRALinear")

LORA_PEFT_METHOD: Final = "lora"
is_lora_layer = _peft.is_lora_layer


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class LoRAConfig(PeftConfig):
    """ALCHEMI LoRA configuration for :class:`FineTuningStrategy`.

    This configuration is passed to
    :class:`~nvalchemi.training.FineTuningStrategy` to enable LoRA PEFT. The
    strategy prepends a :class:`~nvalchemi.training.peft.lora_hook.LoRAHook` that applies the
    adapters during workflow registration.

    Parameters
    ----------
    peft_method : Literal["lora"]
        Fixed PEFT method name as ``"lora"``.
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
    rank : int
        Rank of the low-rank adapter factors. Defaults to ``8``.
    alpha : float, optional
        Scaling numerator for adapter updates. Defaults to ``1.0``.
    lora_dropout : float, optional
        Dropout probability on the adapter input path. Defaults to ``0.0``.
    wrap_mlp : bool, optional
        Also target supported feed-forward sub-blocks discovered by the adapter
        implementation. This is only supported for single-model strategies.
        Defaults to ``False``.
    wrapper_registrations : LoRAWrapperRegistrations, optional
        Custom layer-to-wrapper registrations installed before adapter
        injection. Each pair maps a base layer class to the adapter wrapper
        class that should handle it. The registrations only apply during
        adapter injection, and the global wrapper registry is restored to its
        prior state as soon as injection finishes. Replacing an existing wrapper
        emits a ``UserWarning``; assigning two different wrappers to one layer class
        in the same configuration raises ``ValueError``. Defaults to ``()``.

    Examples
    --------
    Exact module names can be written as patterns without glob characters:

    >>> config = LoRAConfig(
    ...     lora_target_patterns=("main.model.projection",),
    ... )

    Glob patterns match module names from ``model.named_modules()``, prefixed with
    the model key such as ``"main"``:

    >>> config = LoRAConfig(
    ...     lora_target_patterns=("main.model.*projection",),
    ...     rank=4,
    ...     alpha=1.0,
    ... )

    In multi-model workflows, the prefix helps to identify which model receives adapters:

    >>> config = LoRAConfig(
    ...     lora_target_patterns=("student.model.projection",),
    ... )
    """

    peft_method: Literal["lora"] = Field(default=LORA_PEFT_METHOD, frozen=True)
    rank: int = 8
    alpha: float = 1.0
    lora_target_patterns: tuple[str, ...] = Field(min_length=1)
    lora_dropout: float = 0.0
    wrap_mlp: bool = False
    wrapper_registrations: LoRAWrapperRegistrations | None = None

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    @field_validator("wrapper_registrations", mode="before")
    @classmethod
    def _deserialize_wrapper_registrations(
        cls,
        value: Any,
        info: ValidationInfo,
    ) -> Any:
        """Import wrapper class paths when validating a serialized PEFT spec."""
        context = info.context or {}
        if "import_path_validator" not in context or value is None:
            return value
        if not isinstance(value, list):
            raise ValueError("LoRAConfig wrapper_registrations must be a list or null.")

        import_path_validator = context["import_path_validator"]
        wrapper_registrations = []
        for item in value:
            if (
                not isinstance(item, list)
                or len(item) != 2
                or not all(isinstance(path, str) for path in item)
            ):
                raise ValueError(
                    "LoRAConfig wrapper_registrations entries must be "
                    "[layer_cls_path, wrapper_cls_path] lists."
                )
            if import_path_validator is not None:
                import_path_validator(item[0], "LoRA wrapper layer class")
                import_path_validator(item[1], "LoRA wrapper class")
            wrapper_registrations.append((_import_cls(item[0]), _import_cls(item[1])))
        return tuple(wrapper_registrations)

    @model_validator(mode="after")
    def _validate_wrapper_registrations(self) -> Self:
        """Reject conflicting wrappers for one layer within this config."""
        selected: dict[type[nn.Module], type[nn.Module]] = {}
        for layer_cls, wrapper_cls in self.wrapper_registrations or ():
            existing = selected.get(layer_cls)
            if existing is not None and existing is not wrapper_cls:
                raise ValueError(
                    "Multiple LoRA wrappers configured for "
                    f"{layer_cls.__module__}.{layer_cls.__qualname__}: "
                    f"{existing.__module__}.{existing.__qualname__} and "
                    f"{wrapper_cls.__module__}.{wrapper_cls.__qualname__}."
                )
            selected[layer_cls] = wrapper_cls
        return self

    def to_spec_dict(self) -> dict[str, Any]:
        """Return a JSON-safe representation of this LoRA config.

        Returns
        -------
        dict[str, Any]
            JSON-safe LoRA configuration data suitable for serialization.
        """
        spec = super().to_spec_dict()
        spec["lora_target_patterns"] = list(self.lora_target_patterns)
        spec["wrapper_registrations"] = [
            [_cls_path_of(layer_cls), _cls_path_of(wrapper_cls)]
            for layer_cls, wrapper_cls in self.wrapper_registrations or ()
        ]
        return spec


# ---------------------------------------------------------------------------
# # PEFT hooks for strategy
# ---------------------------------------------------------------------------


def _lora_setup_hooks(
    config: PeftConfig,
    strategy_data: Mapping[str, Any],
) -> list[Any]:
    """Build registration-time hooks for a LoRA config.

    Parameters
    ----------
    config : PeftConfig
        PEFT config used to build LoRA registration hooks.
    strategy_data : Mapping[str, Any]
        Strategy data used to validate model-dependent LoRA options.

    Returns
    -------
    list[Any]
        Registration-time hooks needed to inject LoRA adapters.
    """
    from nvalchemi.training.peft.lora_hook import LoRAHook

    if not isinstance(config, LoRAConfig):
        raise ValueError(f"Unsupported PEFT config type: {type(config).__name__}.")
    normalized_models = strategy_validation._normalize_models(
        strategy_data.get("models")
    )
    if (
        config.wrap_mlp
        and isinstance(normalized_models, dict)
        and len(normalized_models) > 1
    ):
        raise ValueError(
            "LoRAConfig.wrap_mlp is not supported with multiple models. Use "
            "explicit lora_target_patterns to select target modules for each "
            "model instead."
        )
    return [LoRAHook(lora_config=config)]


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------


def merge_lora_into_model(model: nn.Module, *, strict: bool = True) -> nn.Module:
    """Merge LoRA weights into ``model`` in place.

    Parameters
    ----------
    model : torch.nn.Module
        Model containing LoRA adapter layers to merge.
    strict : bool, optional
        If ``True``, raise when non-mergeable LoRA layers remain after merging.
        If ``False``, warn and return the partially merged model. Defaults to
        ``True``.

    Returns
    -------
    torch.nn.Module
        The model after merge has been attempted.
    """
    merged = _peft.merge_lora(model)
    remaining = [
        name for name, module in merged.named_modules() if _peft.is_lora_layer(module)
    ]
    if remaining:
        message = (
            f"LoRA merge left non-mergeable adapter module(s) in model: {remaining}."
        )
        if strict:
            raise RuntimeError(message)
        warnings.warn(message, UserWarning, stacklevel=2)
    return merged


def _apply_lora_from_checkpoint_metadata(
    model: nn.Module,
    strategy_metadata: Mapping[str, Any],
    *,
    model_name: str,
    import_path_validator: Callable[[str, str], None] | None = None,
) -> None:
    """Inject LoRA adapters into ``model`` using checkpoint strategy metadata."""
    raw_config = strategy_metadata.get("peft_config")
    if not isinstance(raw_config, Mapping):
        raise ValueError("Checkpoint strategy metadata does not contain peft_config.")
    config = PeftConfig.from_spec_dict(
        raw_config,
        import_path_validator=import_path_validator,
    )
    if not isinstance(config, LoRAConfig):
        raise ValueError(f"Unsupported PEFT config type: {type(config).__name__}.")
    from nvalchemi.training.peft.lora_hook import LoRAHook

    LoRAHook(lora_config=config, register_parameters=False).on_register(
        SimpleNamespace(models={model_name: model})
    )


# ---------------------------------------------------------------------------
# PEFT Method Registration
# ---------------------------------------------------------------------------


def register_lora_method() -> None:
    """Register the built-in LoRA implementation with the PEFT registry."""
    register_peft_method(
        LORA_PEFT_METHOD,
        config_cls=LoRAConfig,
        build_peft_setup_hooks=_lora_setup_hooks,
        apply_peft_from_checkpoint_metadata=_apply_lora_from_checkpoint_metadata,
        merge_peft=merge_lora_into_model,
    )
