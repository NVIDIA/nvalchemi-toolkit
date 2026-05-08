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
"""Reproducible, no-pickle serialization of MLIP hyperparameters.

This module provides :class:`BaseSpec`, a Pydantic model that captures the
``__init__`` arguments of any target class --- typically an MLIP, an optimizer,
or a learning-rate scheduler --- and serializes them to plain JSON. Spec
reconstruction imports the target class by its dotted path and instantiates
it with the stored kwargs. This approach ensures that ``pickle`` is not needed
to recreate objects at runtime:

- Hyperparameters are stored as plain JSON (strings, numbers, lists, dicts).
- :class:`torch.Tensor` is serialized as ``{dtype, shape, data}`` — a data
  structure, not a bytecode payload.
- :class:`torch.dtype` is serialized as its string name and rehydrated with
  an :func:`isinstance` guard so that an attacker-controlled string cannot
  smuggle arbitrary ``torch.*`` attributes through :func:`getattr`.
- Model weights (stored separately) must be loaded with
  ``torch.load(..., weights_only=True)`` — the only pickle-free code path
  that PyTorch offers for weight bundles.

Custom (de)serializers for additional types are registered via
:func:`register_type_serializer`. The module pre-registers handlers for
:class:`torch.dtype`, :class:`torch.device`, and :class:`torch.Tensor`.
"""

from __future__ import annotations

import importlib
import inspect
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Annotated, Any

import torch
from pydantic import (
    AfterValidator,
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    PlainSerializer,
    SerializeAsAny,
    create_model,
)

_META_FIELDS: frozenset[str] = frozenset({"cls_path", "timestamp"})
"""Field names reserved by :class:`BaseSpec` itself; never forwarded to ``build``."""


# ---------------------------------------------------------------------------
# Custom type serializer registry (torch.dtype, torch.device, torch.Tensor, ...)
# ---------------------------------------------------------------------------

_TYPE_SERIALIZERS: dict[type, tuple[Callable[[Any], Any], Callable[[Any], Any]]] = {}
"""Registry mapping a type to its ``(serialize, deserialize)`` callable pair."""


def register_type_serializer(
    type_: type,
    serialize: Callable[[Any], Any],
    deserialize: Callable[[Any], Any],
) -> None:
    """Register JSON (de)serializers for a custom type.

    Registered types can appear as field values on a :class:`BaseSpec`
    subclass; :func:`create_model_spec` wraps them in a Pydantic
    :class:`~pydantic.BeforeValidator` / :class:`~pydantic.PlainSerializer`
    pair so that :meth:`~pydantic.BaseModel.model_dump_json` and
    :meth:`~pydantic.BaseModel.model_validate` round-trip values through the
    provided hooks.

    Parameters
    ----------
    type_
        The Python type to register, for example :class:`torch.dtype`.
    serialize
        Callable converting a ``type_`` instance to a JSON-safe value
        (usually a :class:`str` or a plain :class:`dict`).
    deserialize
        Callable converting the JSON-safe value back into a ``type_``
        instance. Must be tolerant of the case where it is handed an
        already-rehydrated ``type_`` instance (the wrapper short-circuits in
        that case, but custom implementations should not crash on it).

    Notes
    -----
    Re-registering an already-registered ``type_`` silently replaces the
    previous ``(serialize, deserialize)`` pair; no warning is emitted. This
    matches the prototype and allows downstream code to override built-in
    handlers for :class:`torch.dtype`, :class:`torch.device`, and
    :class:`torch.Tensor`. Callers that need to detect collisions can test
    ``type_ in _TYPE_SERIALIZERS`` against the module-private registry
    before registering.

    Examples
    --------
    >>> import torch
    >>> register_type_serializer(
    ...     torch.device,
    ...     serialize=str,
    ...     deserialize=torch.device,
    ... )
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


# register some serializers that will definitely be used
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


# ---------------------------------------------------------------------------
# cls_path <-> class resolution
# ---------------------------------------------------------------------------


def _import_cls(cls_path: str) -> type:
    """Import the class identified by a dotted path.

    Resolves a dotted path such as ``"module.submodule.Class"`` or
    ``"module.Outer.Inner"`` into a class object. The module prefix is
    matched greedily: the longest importable prefix of ``cls_path`` is
    used as the module, and the remaining dotted components are resolved
    as attributes via :func:`getattr` (supporting nested classes).

    Parameters
    ----------
    cls_path : str
        Dotted path of the form ``"module.[submodule...].QualName"``.
        ``QualName`` may itself contain dots when the target is a nested
        class (e.g. ``"pkg.mod.Outer.Inner"``).

    Returns
    -------
    type
        The resolved class object.

    Raises
    ------
    ModuleNotFoundError
        No importable module prefix was found in ``cls_path``.
    AttributeError
        A component of the attribute chain after the module does not
        exist on its parent.
    TypeError
        The resolved object is not a class.
    """
    parts = cls_path.split(".")
    # Find the longest dotted prefix that is an importable module. Try
    # increasingly long prefixes left-to-right; stop at the first
    # ModuleNotFoundError and keep the last successful module. Only
    # catch ModuleNotFoundError so genuine import failures inside a
    # real module still propagate.
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
            f"Could not import any module prefix of {cls_path!r}. "
            "Expected a dotted path like 'pkg.mod.Class' or 'pkg.mod.Outer.Inner'."
        )
    obj: Any = module
    for part in parts[module_depth:]:
        obj = getattr(obj, part)
    if not isinstance(obj, type):
        raise TypeError(f"{cls_path!r} resolved to non-class {obj!r}")
    return obj


def _cls_path_of(cls_: type) -> str:
    """Return the canonical dotted path (``module.QualName``) for ``cls_``."""
    return f"{cls_.__module__}.{cls_.__qualname__}"


def _ensure_importable(cls_path: str) -> str:
    """Pydantic validator: ensure the class path is importable without modifying the string.

    We cannot use `_import_cls` directly as a validator because it returns the
    class object, but the `cls_path` field must store the raw string.
    """
    _import_cls(cls_path)
    return cls_path


# ---------------------------------------------------------------------------
# Signature introspection
# ---------------------------------------------------------------------------


def _signature(cls_: type) -> inspect.Signature:
    """Return the (string-annotation-resolved) signature of ``cls_.__init__``."""
    return inspect.signature(cls_, eval_str=True)


def _check_no_positional_only(cls_: type) -> None:
    """Raise :class:`TypeError` if ``cls_.__init__`` has positional-only params."""
    for name, p in _signature(cls_).parameters.items():
        if p.kind is inspect.Parameter.POSITIONAL_ONLY:
            raise TypeError(
                f"{_cls_path_of(cls_)} has positional-only param {name!r}; "
                "create_model_spec only supports kwargs."
            )


# ---------------------------------------------------------------------------
# BaseSpec
# ---------------------------------------------------------------------------


class BaseSpec(BaseModel):
    """Base class for JSON-serializable, no-pickle hyperparameter specs.

    Concrete spec classes are built dynamically by :func:`create_model_spec`
    via :func:`pydantic.create_model`; each carries one field per
    ``__init__`` kwarg of its target class plus the two metadata fields
    defined here.

    Attributes
    ----------
    cls_path
        Dotted path (``"module.submodule.QualName"``) identifying the target
        class. Validated at assignment time by :func:`_import_cls`.
    timestamp
        ISO-8601 UTC timestamp recording when the spec was created.

    Notes
    -----
    ``revalidate_instances="never"`` is deliberate: specs are immutable
    records of past state; revalidating on access would reject any
    already-typed field values (e.g. rehydrated :class:`torch.Tensor`
    objects) that were stored through a :class:`~pydantic.BeforeValidator`.
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        revalidate_instances="never",
    )

    cls_path: Annotated[
        str,
        AfterValidator(_ensure_importable),
        Field(description="Dotted import path of the target class."),
    ]
    timestamp: Annotated[
        str,
        Field(description="ISO-8601 UTC timestamp of spec creation."),
    ]

    def build(self, *args: Any, strict: bool = False, **extra_kwargs: Any) -> object:
        """Instantiate the target class from the stored hyperparameters.

        Positional ``*args`` and ``**extra_kwargs`` inject runtime-only
        values that cannot be serialized into the spec --- for example,
        ``model.parameters()`` for an optimizer or an ``optimizer`` instance
        for a learning-rate scheduler.

        Nested :class:`BaseSpec` field values are built recursively before
        forwarding to the target constructor. Non-empty homogeneous
        ``list``/``tuple`` fields whose items are all :class:`BaseSpec`
        are built item-wise, preserving the container type; mixed or
        empty collections are passed through unchanged. Nested
        collections (e.g. ``list[list[BaseSpec]]``) are not traversed;
        wrap them in a serializable spec object or flatten the
        collection. A JSON round-trip preserves the order of items in
        such sequences but not the ``list`` vs. ``tuple`` container
        type (tuple fields come back as lists); the annotated container
        type is rebuilt here when each item is ``.build()``-ed.

        Parameters
        ----------
        *args
            Positional arguments forwarded to the target class constructor
            (runtime-only, not stored in the spec).
        strict
            Reserved for future use; currently a no-op retained to preserve
            the public API. Accepts any value without effect.
        **extra_kwargs
            Extra keyword arguments forwarded to the target class
            constructor, overriding any spec-stored kwargs of the same name.

        Returns
        -------
        object
            A freshly constructed instance of the class at :attr:`cls_path`.

        Raises
        ------
        TypeError
            If the target class cannot be instantiated with the resolved
            kwargs.
        """
        del strict  # reserved for future use
        cls_ = _import_cls(self.cls_path)
        sig = _signature(cls_)
        resolved: dict[str, Any] = {}
        for name in type(self).model_fields:
            if name in _META_FIELDS:
                continue
            v = getattr(self, name)
            # Nested spec: build unless target expects the spec itself.
            if isinstance(v, BaseSpec):
                param = sig.parameters.get(name)
                ann = param.annotation if param is not None else None
                wants_spec = isinstance(ann, type) and issubclass(ann, BaseSpec)
                resolved[name] = v if wants_spec else v.build()
            elif _is_basespec_sequence(v):
                resolved[name] = _build_sequence_of_specs(v)
            else:
                resolved[name] = v
        resolved.update(extra_kwargs)
        try:
            return cls_(*args, **resolved)
        except TypeError as e:
            raise TypeError(
                f"Failed to build {self.cls_path} from spec "
                f"(saved at {self.timestamp}): {e}. The class signature "
                "may have changed since the spec was created."
            ) from e


# ---------------------------------------------------------------------------
# Type annotation resolution
# ---------------------------------------------------------------------------


def _try_deserialize(value: Any) -> Any:
    """Probe registered deserializers to rehydrate a raw JSON value.

    Returns the first successfully deserialized typed instance, or the
    original ``value`` unchanged if no deserializer accepts it. This covers
    the case where ``__init__`` has no annotation for a parameter whose
    stored value is a serialized custom type (e.g. ``torch.dtype`` as a
    str) — without this probe, :func:`_resolve_annotation` would infer the
    plain ``str`` / ``dict`` type and the registered BeforeValidator would
    never fire.

    Only primitives likely to match a custom serialized form are probed
    (``str`` and ``dict``); all other values pass through unchanged.
    """
    if not isinstance(value, (str, dict)):
        return value
    for _, deser in _TYPE_SERIALIZERS.values():
        try:
            return deser(value)
        except (TypeError, ValueError, KeyError, AttributeError, RuntimeError):
            continue
    return value


def _is_basespec_sequence(value: Any) -> bool:
    """Return whether value is a non-empty list/tuple of BaseSpec instances."""
    return (
        isinstance(value, (list, tuple))
        and len(value) > 0
        and all(isinstance(v, BaseSpec) for v in value)
    )


def _is_spec_dict(value: Any) -> bool:
    """Return whether value is a JSON-dict representation of a BaseSpec."""
    return isinstance(value, dict) and "cls_path" in value


def _is_spec_dict_sequence(value: Any) -> bool:
    """Return whether value is a non-empty list of spec-dicts (as in JSON)."""
    return (
        isinstance(value, list)
        and len(value) > 0
        and all(_is_spec_dict(v) for v in value)
    )


def _build_sequence_of_specs(value: Any) -> Any:
    """Rebuild each :class:`BaseSpec` item in a list/tuple, preserving container type."""
    return type(value)(item.build() for item in value)


def _resolve_annotation(name: str, value: Any, sig: inspect.Signature) -> Any:
    """Pick the Pydantic field annotation for ``(name, value)`` in ``sig``.

    Order of precedence:

    1. ``value`` is a :class:`BaseSpec` → ``SerializeAsAny[BaseSpec]``
       (preserves the concrete dynamic schema under
       :meth:`~pydantic.BaseModel.model_dump_json`).
    2. ``value`` is a non-empty ``list``/``tuple`` whose items are all
       :class:`BaseSpec` → ``SerializeAsAny[list[BaseSpec]]`` or
       ``SerializeAsAny[tuple[BaseSpec, ...]]``. This lets collection
       fields (e.g. ``ComposedLossFunction.components``) round-trip by
       preserving each item's dynamic spec schema.
    3. The ``__init__`` signature annotates this parameter with a registered
       custom type → wrap via :func:`_wrap_custom_type`.
    4. The ``__init__`` signature has any non-``Any`` annotation → use it.
    5. Otherwise infer from ``type(value)``; if the inferred type is in the
       registry, wrap it; ``None`` values fall back to :class:`typing.Any`.
    """
    if isinstance(value, BaseSpec):
        return SerializeAsAny[BaseSpec]

    if _is_basespec_sequence(value):
        return (
            SerializeAsAny[list[BaseSpec]]
            if isinstance(value, list)
            else SerializeAsAny[tuple[BaseSpec, ...]]
        )

    param = sig.parameters.get(name)
    sig_ann = param.annotation if param is not None else inspect.Parameter.empty
    has_sig_ann = sig_ann is not inspect.Parameter.empty and sig_ann is not Any

    if has_sig_ann and sig_ann in _TYPE_SERIALIZERS:
        return _wrap_custom_type(sig_ann)
    if has_sig_ann:
        return sig_ann

    vt = type(value)
    if vt in _TYPE_SERIALIZERS:
        return _wrap_custom_type(vt)
    return vt if value is not None else Any


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def create_model_spec(cls_: type, **kwargs: Any) -> BaseSpec:
    """Build a :class:`BaseSpec` instance for ``cls_`` with the given kwargs.

    A new Pydantic model class is dynamically created via
    :func:`pydantic.create_model`, one field per kwarg, each annotated by
    :func:`_resolve_annotation`. The resulting spec is JSON-serializable
    with :meth:`~pydantic.BaseModel.model_dump_json` and reconstructible
    with :func:`create_model_spec_from_json`.

    Non-empty homogeneous ``list``/``tuple`` kwargs whose items are all
    :class:`BaseSpec` are annotated so each item's dynamic spec schema
    survives JSON dump and rehydration, and :meth:`BaseSpec.build` then
    rebuilds each item. Mixed or empty collections are stored as-is and
    are not specially rehydrated. Nested collections (e.g.
    ``list[list[BaseSpec]]``) are not traversed; wrap them in a
    serializable spec object or flatten the collection. A JSON round-trip
    preserves the order of items in tuple-valued fields but not the
    container type (tuple fields come back as lists); the annotated
    container type is rebuilt by :meth:`BaseSpec.build`.

    Parameters
    ----------
    cls_
        The target class. Must accept all ``**kwargs`` as keyword arguments
        and must not declare any positional-only parameters.
    **kwargs
        Hyperparameters for ``cls_``. Registered types
        (:class:`torch.Tensor`, :class:`torch.dtype`, :class:`torch.device`,
        and any user-registered types) are handled via the type-serializer
        registry. Other values must themselves be JSON-serializable by
        Pydantic.

    Returns
    -------
    BaseSpec
        A dynamically subclassed :class:`BaseSpec` instance named
        ``"{cls_.__name__}Spec"`` with one field per kwarg plus the two
        metadata fields.

    Raises
    ------
    TypeError
        If ``cls_`` has positional-only parameters, or if ``**kwargs``
        contains names absent from the signature while the signature has no
        ``**kwargs`` parameter.

    Examples
    --------
    >>> import torch.nn as nn
    >>> spec = create_model_spec(nn.Linear, in_features=8, out_features=4)
    >>> module = spec.build()
    >>> (module.in_features, module.out_features)
    (8, 4)
    """
    _check_no_positional_only(cls_)
    sig = _signature(cls_)

    unknown = set(kwargs) - set(sig.parameters)
    if unknown:
        var_kw = any(
            p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
        )
        if not var_kw:
            raise TypeError(
                f"Unknown kwargs for {_cls_path_of(cls_)}: {sorted(unknown)}"
            )

    fields: dict[str, tuple[Any, Any]] = {}
    for name, value in kwargs.items():
        annotation = _resolve_annotation(name, value, sig)
        fields[name] = (annotation, value)

    model_cls = create_model(
        f"{cls_.__name__}Spec",
        __base__=BaseSpec,
        **fields,
    )
    return model_cls(
        cls_path=_cls_path_of(cls_),
        timestamp=datetime.now(timezone.utc).isoformat(),
        **kwargs,
    )


def create_model_spec_from_json(spec: dict[str, Any]) -> BaseSpec:
    """Rebuild a :class:`BaseSpec` from its JSON-dict form.

    Recursively rehydrates nested specs (detected as values that are
    :class:`dict` and contain a ``"cls_path"`` key). Lists of such dicts
    are rehydrated item-wise, preserving the collection order. Pydantic's
    :class:`~pydantic.BeforeValidator` hooks on registered types handle the
    str → :class:`torch.dtype` / :class:`torch.device` / dict →
    :class:`torch.Tensor` conversions transparently.

    The original ``timestamp`` is preserved via :func:`object.__setattr__`
    rather than stamped fresh, so that a round-tripped spec remains
    byte-identical (up to JSON-whitespace) with its source.

    Parameters
    ----------
    spec
        A :class:`dict` as produced by
        :meth:`~pydantic.BaseModel.model_dump` or by
        :func:`json.loads` on the output of
        :meth:`~pydantic.BaseModel.model_dump_json`.

    Returns
    -------
    BaseSpec
        A spec instance equivalent to the source, with the original
        ``timestamp`` preserved.

    Raises
    ------
    ValueError
        If ``spec`` is missing ``cls_path`` or ``timestamp``, or if
        ``cls_path`` cannot be imported / resolves to a non-class. The
        underlying exception is preserved as ``__cause__``.

    Examples
    --------
    >>> import json, torch.nn as nn
    >>> s = create_model_spec(nn.Linear, in_features=4, out_features=2)
    >>> dumped = json.loads(s.model_dump_json())
    >>> s2 = create_model_spec_from_json(dumped)
    >>> s2.timestamp == s.timestamp
    True
    """
    schema = dict(spec)
    try:
        cls_path = schema.pop("cls_path")
        stored_timestamp = schema.pop("timestamp")
    except KeyError as e:
        raise ValueError(
            f"Spec JSON missing required field {e.args[0]!r}; "
            f"present keys: {sorted(spec)}"
        ) from e

    try:
        cls_ = _import_cls(cls_path)
    except Exception as e:
        raise ValueError(
            f"Could not resolve cls_path={cls_path!r} while rehydrating spec JSON: {e}"
        ) from e

    kwargs: dict[str, Any] = {}
    for name, value in schema.items():
        if _is_spec_dict(value):
            kwargs[name] = create_model_spec_from_json(value)
        elif _is_spec_dict_sequence(value):
            kwargs[name] = [create_model_spec_from_json(v) for v in value]
        else:
            # Eagerly deserialize strings/dicts that match a registered
            # custom type's serialized form. This is required for classes
            # whose ``__init__`` has no annotation for the parameter
            # (e.g. ``nn.Linear(..., dtype=...)``), because
            # :func:`_resolve_annotation` would otherwise infer ``type(value)
            # == str`` and miss the registered ``torch.dtype`` serializer.
            kwargs[name] = _try_deserialize(value)

    rebuilt = create_model_spec(cls_, **kwargs)
    # Preserve original provenance rather than stamping a fresh timestamp.
    object.__setattr__(rebuilt, "timestamp", stored_timestamp)
    return rebuilt
