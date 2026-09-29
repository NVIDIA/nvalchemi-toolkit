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

"""Prepare and resolve stored-tangent and Gram-statistics NEB methods."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from importlib import import_module

import warp as wp

from .equations import (
    climbing_image_effective_force,
    improved_tangent_weights,
    neb_effective_force,
    neb_effective_force_from_gram_stats,
)

_KEY_SEPARATOR = "|"
_STORED_PREFIX = "stored_tangent|"
_GRAM_PREFIX = "gram_stats|"


@dataclass(frozen=True, slots=True)
class _StoredTangentMethod:
    """Equations for the stored-tangent kernel."""

    tangent_fn: wp.Function
    force_fn: wp.Function
    climbing_force_fn: wp.Function


@dataclass(frozen=True, slots=True)
class _GramStatsMethod:
    """Equations for the Gram-statistics kernel."""

    tangent_fn: wp.Function
    force_fn: wp.Function
    climbing_force_fn: wp.Function


def _import_equation(path: str) -> wp.Function:
    """Import a module-level Warp device function by dotted path."""
    module_name, separator, function_name = path.rpartition(".")
    if not separator or not module_name or not function_name:
        raise ValueError(f"NEB equation {path!r} is not importable")
    try:
        module = import_module(module_name)
    except ModuleNotFoundError as exc:
        if exc.name != module_name and not module_name.startswith(f"{exc.name}."):
            raise
        raise ValueError(f"NEB equation {path!r} is not importable") from exc
    try:
        function = getattr(module, function_name)
    except AttributeError as exc:
        raise ValueError(f"NEB equation {path!r} is not importable") from exc
    if not isinstance(function, wp.Function):
        raise TypeError(f"NEB equation {path!r} is not a warp.func function")
    return function


def _equation_path(name: str, function: wp.Function) -> str:
    """Return an equation's importable path, validating its round trip."""
    if not isinstance(function, wp.Function):
        raise TypeError(f"{name} must be a warp.func function")
    module = getattr(function.func, "__module__", None)
    qualname = getattr(function.func, "__qualname__", None)
    if not module or not qualname or "." in qualname:
        raise ValueError(
            f"{name} must be an importable module-level warp.func function"
        )
    path = f"{module}.{qualname}"
    if _import_equation(path) is not function:
        raise ValueError(
            f"{name} must resolve to the same warp.func function at {path!r}"
        )
    return path


def _classify_neb_method(
    tangent_fn: wp.Function,
    force_fn: wp.Function,
    climbing_force_fn: wp.Function,
) -> type[_StoredTangentMethod] | type[_GramStatsMethod]:
    """Select the kernel from exact equation parameter names."""
    signatures = (
        ("tangent_fn", tangent_fn, improved_tangent_weights),
        ("climbing_force_fn", climbing_force_fn, climbing_image_effective_force),
    )
    for name, function, reference in signatures:
        actual = tuple(function.input_types)
        expected = tuple(reference.input_types)
        if actual != expected:
            raise ValueError(f"{name} parameters must be {expected}; got {actual}")
    params = tuple(force_fn.input_types)
    if params == tuple(neb_effective_force.input_types):
        return _StoredTangentMethod
    if params == tuple(neb_effective_force_from_gram_stats.input_types):
        return _GramStatsMethod
    raise ValueError(f"force_fn matches neither supported NEB signature; got {params}")


def prepare_neb_method_key(
    tangent_fn: wp.Function,
    force_fn: wp.Function,
    climbing_force_fn: wp.Function,
) -> str:
    """Prepare a stable method key to pass to neb_forces.

    The key identifies the stored-tangent or Gram-statistics kernel and includes
    dotted paths to the tangent, regular-force, and climbing-force equations.

    Parameters
    ----------
    tangent_fn : wp.Function
        Warp function that selects the tangent weights.
    force_fn : wp.Function
        Warp function for the regular-image effective force.
    climbing_force_fn : wp.Function
        Warp function for the climbing-image effective force.

    Returns
    -------
    str
        Stable kernel type and equation paths in the order described above.

    Raises
    ------
    TypeError
        If an equation is not a Warp function.
    ValueError
        If an equation is not importable or its parameter names are unsupported.
    """
    paths = (
        _equation_path("tangent_fn", tangent_fn),
        _equation_path("force_fn", force_fn),
        _equation_path("climbing_force_fn", climbing_force_fn),
    )
    method_type = _classify_neb_method(tangent_fn, force_fn, climbing_force_fn)
    prefix = _STORED_PREFIX if method_type is _StoredTangentMethod else _GRAM_PREFIX
    return prefix + _KEY_SEPARATOR.join(paths)


DEFAULT_NEB_METHOD_KEY = prepare_neb_method_key(
    improved_tangent_weights, neb_effective_force, climbing_image_effective_force
)


@cache
def resolve_neb_method(method: str) -> _StoredTangentMethod | _GramStatsMethod:
    """Resolve a key from prepare_neb_method_key to its Warp equations and kernel type.

    Parameters
    ----------
    method : str
        Key returned by prepare_neb_method_key.

    Returns
    -------
    _StoredTangentMethod or _GramStatsMethod
        The three equations in the record for the selected kernel.

    Raises
    ------
    TypeError
        If method is not a string or a path resolves to a non-Warp function.
    ValueError
        If the key or an equation signature is invalid.
    """
    if not isinstance(method, str):
        raise TypeError(f"method must be a string; got {type(method).__name__}")
    if method.startswith(_STORED_PREFIX):
        expected_type = _StoredTangentMethod
        encoded_paths = method[len(_STORED_PREFIX) :]
    elif method.startswith(_GRAM_PREFIX):
        expected_type = _GramStatsMethod
        encoded_paths = method[len(_GRAM_PREFIX) :]
    else:
        raise ValueError("method must be a prepared NEB method key")
    paths = encoded_paths.split(_KEY_SEPARATOR)
    if len(paths) != 3 or any(not path for path in paths):
        raise ValueError("method must be a prepared NEB method key")
    functions = tuple(_import_equation(path) for path in paths)
    method_type = _classify_neb_method(*functions)
    if method_type is not expected_type:
        raise ValueError("method key kernel kind does not match force_fn signature")
    return method_type(*functions)
