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

"""Registry of equations used to specialize batched NEB kernels."""

from __future__ import annotations

from dataclasses import dataclass

import warp as wp

from .equations import (
    climbing_image_effective_force,
    improved_tangent_weights,
    neb_effective_force,
)

__all__ = [
    "available_neb_methods",
    "get_neb_method",
    "register_neb_method",
]


@dataclass(frozen=True, slots=True)
class _StoredTangentMethod:
    """Device equations using the stored-tangent kernel contract."""

    tangent_fn: wp.Function
    force_fn: wp.Function
    climbing_force_fn: wp.Function


@dataclass(frozen=True, slots=True)
class _GramStatsMethod:
    """Device equations using the full Gram-statistics kernel contract."""

    tangent_fn: wp.Function
    force_fn: wp.Function
    climbing_force_fn: wp.Function


_NEB_METHOD_REGISTRY: dict[str, _StoredTangentMethod | _GramStatsMethod] = {
    "improved_tangent": _StoredTangentMethod(
        tangent_fn=improved_tangent_weights,
        force_fn=neb_effective_force,
        climbing_force_fn=climbing_image_effective_force,
    ),
}


def available_neb_methods() -> tuple[str, ...]:
    """Return the registered NEB method names in sorted order."""
    return tuple(sorted(_NEB_METHOD_REGISTRY))


def get_neb_method(name: str) -> _StoredTangentMethod | _GramStatsMethod:
    """Return the method specification registered under ``name``.

    Parameters
    ----------
    name : str
        Registered NEB method name.

    Raises
    ------
    ValueError
        If ``name`` is not registered.
    """
    try:
        return _NEB_METHOD_REGISTRY[name]
    except KeyError:
        available = ", ".join(available_neb_methods())
        raise ValueError(
            f"unknown NEB method {name!r}; available methods: {available}"
        ) from None


def register_neb_method(
    *,
    name: str,
    tangent_fn: wp.Function,
    force_fn: wp.Function,
    climbing_force_fn: wp.Function,
) -> None:
    """Register a user-defined Gram-statistics NEB method.

    Registration specializes kernels only when the registered method is first
    launched. It must therefore occur before ``torch.compile`` or CUDA graph
    capture begins.

    Parameters
    ----------
    name : str
        Stable method name used for kernel selection and caching.
    tangent_fn, force_fn, climbing_force_fn : wp.Function
        Warp device functions implementing the Gram-statistics kernel contracts.

    Raises
    ------
    TypeError
        If a name or equation is of the wrong type.
    ValueError
        If the name is empty or is already bound to different equations.
    """
    functions = {
        "tangent_fn": tangent_fn,
        "force_fn": force_fn,
        "climbing_force_fn": climbing_force_fn,
    }
    for field_name, function in functions.items():
        if not isinstance(function, wp.Function):
            raise TypeError(
                f"{field_name} must be a function decorated with warp.func; "
                f"got {type(function).__name__}"
            )

    if not isinstance(name, str):
        raise TypeError(f"name must be a string; got {type(name).__name__}")
    elif not name.strip():
        raise ValueError("name must not be empty or contain only whitespace")

    # Bind user-defined methods to the full Gram-matrix-statistics contract;
    # the stored-tangent alternative remains a private optimization for built-ins.
    method = _GramStatsMethod(
        tangent_fn=tangent_fn,
        force_fn=force_fn,
        climbing_force_fn=climbing_force_fn,
    )
    existing = _NEB_METHOD_REGISTRY.get(name)
    if existing is None:
        _NEB_METHOD_REGISTRY[name] = method
    elif existing != method:
        raise ValueError(f"NEB method {name!r} is already registered")
