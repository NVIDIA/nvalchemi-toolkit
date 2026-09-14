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
"""Registration and lookup for parameter-efficient fine-tuning methods."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from torch import nn

    from nvalchemi.training.peft.config import PeftConfig

__all__ = [
    "PeftMethodRegistration",
    "available_peft_methods",
    "register_peft_method",
]


class _ApplyPeftFromCheckpointMetadata(Protocol):
    """Callable that recreates PEFT structure from checkpoint metadata."""

    def __call__(
        self,
        model: nn.Module,
        strategy_metadata: Mapping[str, Any],
        *,
        model_name: str,
        import_path_validator: Callable[[str, str], None] | None = None,
    ) -> None: ...


@dataclass(frozen=True)
class PeftMethodRegistration:
    """Operations and configuration type associated with one PEFT method.

    Attributes
    ----------
    peft_method
        Stable and unique PEFT method discriminator stored in serialized metadata.
    config_cls
        Configuration class associated with the method.
    build_peft_setup_hooks
        Callable that builds the hooks needed to configure PEFT before training.
    apply_peft_from_checkpoint_metadata
        Callable that recreates the PEFT structure described by checkpoint
        metadata on a model.
    merge_peft
        Callable that merges PEFT parameters into a model.
    """

    peft_method: str
    config_cls: type[PeftConfig]
    build_peft_setup_hooks: Callable[[PeftConfig, Mapping[str, Any]], list[Any]]
    apply_peft_from_checkpoint_metadata: _ApplyPeftFromCheckpointMetadata
    merge_peft: Callable[..., nn.Module]


_REGISTRATIONS_BY_METHOD: dict[str, PeftMethodRegistration] = {}
_REGISTRATIONS_BY_CONFIG: dict[type[PeftConfig], PeftMethodRegistration] = {}


def register_peft_method(
    method: str,
    *,
    config_cls: type[PeftConfig],
    build_peft_setup_hooks: Callable[[PeftConfig, Mapping[str, Any]], list[Any]],
    apply_peft_from_checkpoint_metadata: _ApplyPeftFromCheckpointMetadata,
    merge_peft: Callable[..., nn.Module],
) -> PeftMethodRegistration:
    """Register all behavior required to use one PEFT method.

    Parameters
    ----------
    method
        Stable method name used in serialized PEFT metadata.
    config_cls
        Configuration class associated with the method.
    build_peft_setup_hooks
        Callable that builds the hooks needed to configure PEFT before training.
    apply_peft_from_checkpoint_metadata
        Callable that recreates the PEFT structure described by checkpoint
        metadata on a model.
    merge_peft
        Callable that merges PEFT parameters into a model.

    Returns
    -------
    PeftMethodRegistration
        The installed registration.

    Raises
    ------
    ValueError
        If ``method`` is empty or the method or configuration class is already
        registered.
    TypeError
        If ``config_cls`` is not a :class:`PeftConfig` subclass.
    """
    from nvalchemi.training.peft.config import PeftConfig

    if not isinstance(method, str) or not method:
        raise ValueError("PEFT method name must be a non-empty string.")
    if not isinstance(config_cls, type) or not issubclass(config_cls, PeftConfig):
        raise TypeError("PEFT config_cls must be a PeftConfig subclass.")
    if method in _REGISTRATIONS_BY_METHOD:
        raise ValueError(f"PEFT method {method!r} is already registered.")
    if config_cls in _REGISTRATIONS_BY_CONFIG:
        raise ValueError(
            f"PEFT config class {config_cls.__module__}.{config_cls.__qualname__} "
            "is already registered."
        )
    registration = PeftMethodRegistration(
        peft_method=method,
        config_cls=config_cls,
        build_peft_setup_hooks=build_peft_setup_hooks,
        apply_peft_from_checkpoint_metadata=apply_peft_from_checkpoint_metadata,
        merge_peft=merge_peft,
    )
    _REGISTRATIONS_BY_METHOD[method] = registration
    _REGISTRATIONS_BY_CONFIG[config_cls] = registration
    return registration


# ---------------------------------------------------------------------------
# Built-in integrations
# ---------------------------------------------------------------------------


def _register_builtin_peft_methods() -> None:
    """Lazily register bundled PEFT methods.

    New built-in PEFT integrations must add their registration callback here.
    External integrations should call :func:`register_peft_method` directly.
    """
    from nvalchemi.training.peft.lora import (
        LORA_PEFT_METHOD,
        register_lora_method,
    )

    if LORA_PEFT_METHOD not in _REGISTRATIONS_BY_METHOD:
        register_lora_method()


def get_peft_registration_by_method(method: Any) -> PeftMethodRegistration:
    """Return the registration identified by a PEFT method name.

    Parameters
    ----------
    method
        Method name read from serialized PEFT metadata.

    Returns
    -------
    PeftMethodRegistration
        The matching registration.

    Raises
    ------
    ValueError
        If no PEFT method is registered under ``method``.
    """
    _register_builtin_peft_methods()
    try:
        return _REGISTRATIONS_BY_METHOD[method]
    except (KeyError, TypeError):
        pass
    raise ValueError(f"Unsupported PEFT method {method!r}.") from None


def get_peft_registration_by_config(config: PeftConfig) -> PeftMethodRegistration:
    """Return the registration associated with a PEFT config object.

    Parameters
    ----------
    config
        PEFT configuration whose method registration should be resolved.

    Returns
    -------
    PeftMethodRegistration
        The registration for the nearest registered class in the configuration's
        method resolution order.

    Raises
    ------
    ValueError
        If the configuration class has no registered PEFT method.
    """
    _register_builtin_peft_methods()
    for config_cls in type(config).__mro__:
        registration = _REGISTRATIONS_BY_CONFIG.get(config_cls)
        if registration is not None:
            return registration
    raise ValueError(f"Unsupported PEFT config type: {type(config).__name__}.")


def available_peft_methods() -> tuple[str, ...]:
    """Return registered and built-in PEFT method names.

    Returns
    -------
    tuple[str, ...]
        Available method names in sorted order.
    """
    _register_builtin_peft_methods()
    return tuple(sorted(_REGISTRATIONS_BY_METHOD))
