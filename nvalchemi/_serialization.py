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
"""Shared no-pickle serialization helpers."""

from __future__ import annotations

import importlib
import inspect
from collections.abc import Callable
from functools import lru_cache
from types import NoneType, UnionType
from typing import Annotated, Any, Union, get_args, get_origin

import torch
from pydantic import BeforeValidator, PlainSerializer

_TYPE_SERIALIZERS: dict[type, tuple[Callable[[Any], Any], Callable[[Any], Any]]] = {}
"""Registry mapping a type to its ``(serialize, deserialize)`` callable pair."""


def register_type_serializer(
    type_: type,
    serialize: Callable[[Any], Any],
    deserialize: Callable[[Any], Any],
) -> None:
    """Register JSON (de)serializers for a custom type.

    Parameters
    ----------
    type_
        The Python type to register, for example :class:`torch.dtype`.
    serialize
        Callable converting a ``type_`` instance to a JSON-safe value.
    deserialize
        Callable converting the JSON-safe value back into a ``type_`` instance.
    """
    _TYPE_SERIALIZERS[type_] = (serialize, deserialize)


def _wrap_custom_type(t: type) -> Any:
    """Wrap a registered type in an ``Annotated[...]`` with Pydantic hooks."""
    ser, deser = _TYPE_SERIALIZERS[t]

    def _before(v: Any) -> Any:
        return v if isinstance(v, t) else deser(v)

    return Annotated[t, BeforeValidator(_before), PlainSerializer(ser)]


def _dtype_deserialize(s: Any) -> torch.dtype:
    """Rehydrate a :class:`torch.dtype` from its string form with a type guard."""
    if isinstance(s, torch.dtype):
        return s
    if not isinstance(s, str):
        raise TypeError(
            f"torch.dtype deserializer expected str, got {type(s).__name__}"
        )
    result = getattr(torch, s.removeprefix("torch."), None)
    if not isinstance(result, torch.dtype):
        raise ValueError(
            f"{s!r} does not resolve to a torch.dtype "
            "(defense-in-depth against attacker-controlled JSON smuggling "
            "non-dtype torch.* attributes)."
        )
    return result


def _tensor_serialize(t: torch.Tensor) -> dict[str, Any]:
    """Serialize a :class:`torch.Tensor` as ``{data, dtype, shape}``."""
    return {
        "data": t.detach().cpu().tolist(),
        "dtype": str(t.dtype),
        "shape": list(t.shape),
    }


def _tensor_deserialize(v: Any) -> torch.Tensor:
    """Rehydrate a :class:`torch.Tensor` from its ``{data, dtype, shape}`` dict."""
    if isinstance(v, torch.Tensor):
        return v
    if not isinstance(v, dict):
        raise TypeError(f"Cannot deserialize torch.Tensor from {type(v).__name__}")
    dtype = _dtype_deserialize(v["dtype"])
    out = torch.tensor(v["data"], dtype=dtype)
    expected_shape = tuple(v["shape"])
    if tuple(out.shape) != expected_shape:
        out = out.reshape(expected_shape)
    return out


register_type_serializer(
    torch.dtype,
    serialize=str,
    deserialize=_dtype_deserialize,
)
register_type_serializer(
    torch.device,
    serialize=str,
    deserialize=lambda s: s if isinstance(s, torch.device) else torch.device(s),
)
register_type_serializer(torch.Tensor, _tensor_serialize, _tensor_deserialize)


@lru_cache(maxsize=None)
def _import_object(path: str) -> Any:
    """Import an object identified by a dotted module/attribute path."""
    parts = path.split(".")
    module: Any = None
    module_depth = 0
    for i in range(1, len(parts)):
        try:
            module = importlib.import_module(".".join(parts[:i]))
        except ModuleNotFoundError:
            break
        module_depth = i
    if module is None:
        raise ModuleNotFoundError(
            f"Could not import any module prefix of {path!r}. "
            "Expected a dotted path like 'pkg.mod.Object' or "
            "'pkg.mod.Outer.method'."
        )
    obj: Any = module
    for part in parts[module_depth:]:
        obj = getattr(obj, part)
    return obj


@lru_cache(maxsize=None)
def _import_cls(cls_path: str) -> type:
    """Import the class identified by a dotted path."""
    obj = _import_object(cls_path)
    if not isinstance(obj, type):
        raise TypeError(f"{cls_path!r} resolved to non-class {obj!r}")
    return obj


@lru_cache(maxsize=None)
def _import_callable(target_path: str) -> Callable[..., Any]:
    """Import the callable identified by a dotted path."""
    obj = _import_object(target_path)
    if not callable(obj):
        raise TypeError(f"{target_path!r} resolved to non-callable {obj!r}")
    return obj


def _callable_path_of(target: Callable[..., Any]) -> str:
    """Return the canonical dotted path (``module.QualName``) for ``target``."""
    module = getattr(target, "__module__", None)
    qualname = getattr(target, "__qualname__", None)
    if not module or not qualname or "<locals>" in qualname or "<lambda>" in qualname:
        raise TypeError(
            f"{target!r} is not an importable callable. Specs require a "
            "module-level class, function, staticmethod, or classmethod."
        )
    return f"{module}.{qualname}"


def _cls_path_of(cls_: type) -> str:
    """Return the canonical dotted path (``module.QualName``) for ``cls_``."""
    return _callable_path_of(cls_)


@lru_cache(maxsize=None)
def _callable_signature(target: Callable[..., Any]) -> inspect.Signature:
    """Return the string-annotation-resolved signature for ``target``."""
    return inspect.signature(target, eval_str=True)


@lru_cache(maxsize=None)
def _constructor_signature(cls_: type) -> inspect.Signature:
    """Return the string-annotation-resolved constructor signature for ``cls_``."""
    return _callable_signature(cls_)


def _extract_init_kwargs_from_attrs(instance: Any) -> dict[str, Any]:
    """Extract constructor kwargs from matching attributes on ``instance``."""
    sig = _constructor_signature(type(instance))
    kwargs: dict[str, Any] = {}
    for name, param in sig.parameters.items():
        if name == "self" or param.kind in {
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        }:
            continue
        try:
            kwargs[name] = getattr(instance, name)
        except AttributeError:
            continue
    return kwargs


def _serialize_type(value: type | None) -> str | None:
    """Serialize a class to its dotted path; pass ``None`` through."""
    if value is None:
        return None
    return _cls_path_of(value)


def _validate_type(value: Any) -> Any:
    """Accept a ``type`` or dotted-path string; convert strings to classes."""
    if value is None or isinstance(value, type):
        return value
    if _is_tagged_type(value):
        return _deserialize_tagged_type(value)
    if isinstance(value, str):
        try:
            return _import_cls(value)
        except (ImportError, AttributeError, TypeError) as exc:
            raise ValueError(f"{value!r} must resolve to an importable class.") from exc
    return value


def _is_class_type_annotation(annotation: Any) -> bool:
    """Return whether ``annotation`` accepts a class object."""
    if annotation is type:
        return True
    return get_origin(annotation) is type


def _is_optional_class_type_annotation(annotation: Any) -> bool:
    """Return whether ``annotation`` accepts a class object or ``None``."""
    origin = get_origin(annotation)
    if origin not in {Union, UnionType}:
        return False
    args = get_args(annotation)
    non_none_args = [arg for arg in args if arg is not NoneType]
    return len(non_none_args) == 1 and _is_class_type_annotation(non_none_args[0])


def _wrap_class_type_annotation(annotation: Any) -> Any:
    """Wrap class-object annotations with dotted-path Pydantic hooks."""
    return Annotated[
        annotation,
        BeforeValidator(_validate_type),
        PlainSerializer(_serialize_type),
    ]


def _serialize_tagged_type(value: type) -> dict[str, str]:
    """Serialize an inferred class value with an explicit type tag."""
    return {"__type__": _cls_path_of(value)}


def _is_tagged_type(value: Any) -> bool:
    """Return whether ``value`` is a tagged class serialization payload."""
    return isinstance(value, dict) and set(value) == {"__type__"}


def _deserialize_tagged_type(value: Any) -> type:
    """Deserialize a tagged class serialization payload."""
    if isinstance(value, type):
        return value
    if not _is_tagged_type(value):
        raise TypeError(
            f"tagged type deserializer expected {{'__type__': str}}, "
            f"got {type(value).__name__}"
        )
    cls_path = value["__type__"]
    if not isinstance(cls_path, str):
        raise TypeError(f"tagged type path must be str, got {type(cls_path).__name__}")
    return _deserialize_type(cls_path)


SerializableTaggedClass = Annotated[
    type,
    BeforeValidator(_deserialize_tagged_type),
    PlainSerializer(_serialize_tagged_type),
]
"""``type`` annotation for inferred class fields using tagged JSON."""


def _is_serializable_class_annotation(annotation: Any) -> bool:
    """Return whether ``annotation`` should use class dotted-path hooks."""
    return _is_class_type_annotation(annotation) or _is_optional_class_type_annotation(
        annotation
    )


def _deserialize_type(value: Any) -> type:
    """Deserialize a class object from a dotted path for the type registry."""
    if isinstance(value, type):
        return value
    if not isinstance(value, str):
        raise TypeError(
            f"type deserializer expected str or type, got {type(value).__name__}"
        )
    try:
        return _import_cls(value)
    except (ImportError, AttributeError, TypeError) as exc:
        raise ValueError(
            f"{value!r} is not a dotted path resolving to a class."
        ) from exc


register_type_serializer(
    type,
    serialize=_serialize_type,
    deserialize=_deserialize_type,
)


SerializableClass = Annotated[
    type,
    BeforeValidator(_validate_type),
    PlainSerializer(_serialize_type),
]
"""``type`` field annotation that round-trips via dotted-path strings."""

SerializableOptionalClass = Annotated[
    type | None,
    BeforeValidator(_validate_type),
    PlainSerializer(_serialize_type),
]
"""``type | None`` field annotation that round-trips via dotted-path strings."""
