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
keyword arguments of any importable target callable --- typically an MLIP
constructor, model factory, optimizer, or learning-rate scheduler --- and
serializes them to plain JSON. Spec reconstruction imports the target callable
by its dotted path and invokes it with the stored kwargs. This approach ensures that ``pickle`` is not needed
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

import inspect
from datetime import datetime, timezone
from typing import Annotated, Any, get_args, get_origin

import torch
from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    SerializeAsAny,
    create_model,
)

from nvalchemi._serialization import (
    _TYPE_SERIALIZERS,
    SerializableTaggedClass,
    _callable_path_of,
    _callable_signature,
    _constructor_signature,
    _deserialize_tagged_type,
    _import_callable,
    _is_serializable_class_annotation,
    _is_tagged_type,
    _wrap_class_type_annotation,
    _wrap_custom_type,
)
from nvalchemi._serialization import (
    _dtype_deserialize as _dtype_deserialize,
)
from nvalchemi._serialization import (
    _import_cls as _import_cls,
)
from nvalchemi._serialization import (
    register_type_serializer as register_type_serializer,
)

_META_FIELDS: frozenset[str] = frozenset({"cls_path", "timestamp"})
"""Field names reserved by :class:`BaseSpec` itself; never forwarded to ``build``."""


def _ensure_importable(cls_path: str) -> str:
    """Pydantic validator: ensure the target path is importable and callable."""
    _import_callable(cls_path)
    return cls_path


# ---------------------------------------------------------------------------
# Signature introspection
# ---------------------------------------------------------------------------


def _signature(target: Any) -> inspect.Signature:
    """Return the string-annotation-resolved signature for ``target``."""
    if isinstance(target, type):
        return _constructor_signature(target)
    return _callable_signature(target)


def _check_no_positional_only(target: Any) -> None:
    """Raise :class:`TypeError` if ``target`` has positional-only params."""
    for name, p in _signature(target).parameters.items():
        if p.kind is inspect.Parameter.POSITIONAL_ONLY:
            raise TypeError(
                f"{_callable_path_of(target)} has positional-only param {name!r}; "
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
        callable. Validated at assignment time by :func:`_import_callable`.
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
        Field(description="Dotted import path of the target callable."),
    ]
    timestamp: Annotated[
        str,
        Field(description="ISO-8601 UTC timestamp of spec creation."),
    ]

    def accepts_kwarg(self, name: str) -> bool:
        """Return whether the target callable accepts ``name`` as a keyword."""
        target = _import_callable(self.cls_path)
        sig = _signature(target)
        return name in sig.parameters or any(
            p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
        )

    def build(self, *args: Any, strict: bool = False, **extra_kwargs: Any) -> object:
        """Invoke the target callable with the stored hyperparameters.

        Positional ``*args`` and ``**extra_kwargs`` inject runtime-only
        values that cannot be serialized into the spec --- for example,
        ``model.parameters()`` for an optimizer or an ``optimizer`` instance
        for a learning-rate scheduler.

        Nested :class:`BaseSpec` field values are built recursively before
        forwarding to the target constructor. Non-empty ``list``/``tuple``
        fields that contain :class:`BaseSpec` items are built item-wise,
        preserving non-spec items and the container type. Nested
        collections (e.g. ``list[list[BaseSpec]]``) are not traversed;
        wrap them in a serializable spec object or flatten the
        collection. A JSON round-trip preserves tuple-valued spec sequences
        when the target constructor annotates the parameter as a tuple;
        otherwise JSON arrays rehydrate as lists.

        Parameters
        ----------
        *args
            Positional arguments forwarded to the target class constructor
            (runtime-only, not stored in the spec).
        strict
            Reserved for future use; currently a no-op retained to preserve
            the public API. Accepts any value without effect.
        **extra_kwargs
            Extra keyword arguments forwarded to the target callable, overriding
            any spec-stored kwargs of the same name.

        Returns
        -------
        object
            A freshly constructed object from the callable at :attr:`cls_path`.

        Raises
        ------
        TypeError
            If the target callable cannot be invoked with the resolved
            kwargs.
        """
        del strict  # reserved for future use
        target = _import_callable(self.cls_path)
        sig = _signature(target)
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
            return target(*args, **resolved)
        except TypeError as e:
            raise TypeError(
                f"Failed to build {self.cls_path} from spec "
                f"(saved at {self.timestamp}): {e}. The callable signature "
                "may have changed since the spec was created."
            ) from e


# ---------------------------------------------------------------------------
# Type annotation resolution
# ---------------------------------------------------------------------------


def _try_deserialize(name: str, value: Any, sig: inspect.Signature) -> Any:
    """Probe registered deserializers to rehydrate a raw JSON value.

    Returns the first successfully deserialized typed instance, or the
    original ``value`` unchanged if no safe deserializer accepts it. This
    covers the case where ``__init__`` has no annotation for well-known
    parameters whose stored value is a serialized custom type (e.g.
    ``torch.dtype`` as a str for a ``dtype`` parameter).

    Only tagged class dictionaries, unannotated ``dtype`` / ``device`` strings,
    and tensor-shaped dicts are probed. Broad string deserializers such as raw
    class dotted-path resolution are deliberately skipped here so ordinary
    string fields remain strings.
    """
    if not isinstance(value, (str, dict)):
        return value

    param = sig.parameters.get(name)
    sig_ann = param.annotation if param is not None else inspect.Parameter.empty
    if sig_ann is not inspect.Parameter.empty and sig_ann is not Any:
        return value

    deserializer: Any | None = None
    if isinstance(value, str):
        if name == "dtype":
            deserializer = _TYPE_SERIALIZERS[torch.dtype][1]
        elif name == "device":
            deserializer = _TYPE_SERIALIZERS[torch.device][1]
    elif _is_tagged_type(value):
        deserializer = _deserialize_tagged_type
    elif set(value) == {"data", "dtype", "shape"}:
        deserializer = _TYPE_SERIALIZERS[torch.Tensor][1]

    if deserializer is None:
        return value

    try:
        return deserializer(value)
    except (TypeError, ValueError, KeyError, AttributeError, RuntimeError):
        return value


def _maybe_class_annotation(annotation: Any) -> Any | None:
    """Return a dotted-path serializer annotation for class types if applicable."""
    if not _is_serializable_class_annotation(annotation):
        return None
    return _wrap_class_type_annotation(annotation)


def _maybe_registered_type_annotation(annotation: Any) -> Any | None:
    """Return a serializer annotation for registered types and optional variants."""
    if annotation in _TYPE_SERIALIZERS:
        return _wrap_custom_type(annotation)
    args = get_args(annotation)
    if len(args) != 2 or type(None) not in args:
        return None
    registered = [arg for arg in args if arg in _TYPE_SERIALIZERS]
    if len(registered) != 1:
        return None
    return _wrap_custom_type(registered[0]) | None


def _expects_tuple_sequence(name: str, sig: inspect.Signature) -> bool:
    """Return whether ``name`` is annotated as a tuple-valued parameter."""
    param = sig.parameters.get(name)
    if param is None:
        return False
    annotation = param.annotation
    return annotation is tuple or get_origin(annotation) is tuple


def _is_basespec_sequence(value: Any) -> bool:
    """Return whether value is a non-empty list/tuple containing BaseSpec items."""
    return (
        isinstance(value, (list, tuple))
        and len(value) > 0
        and any(isinstance(v, BaseSpec) for v in value)
    )


def _is_spec_dict(value: Any) -> bool:
    """Return whether value is a JSON-dict representation of a BaseSpec."""
    return isinstance(value, dict) and "cls_path" in value


def _is_spec_dict_sequence(value: Any) -> bool:
    """Return whether value is a non-empty list containing spec-dicts."""
    return (
        isinstance(value, list)
        and len(value) > 0
        and any(_is_spec_dict(v) for v in value)
    )


def _build_sequence_of_specs(value: Any) -> Any:
    """Rebuild :class:`BaseSpec` items in a list/tuple, preserving other items."""
    return type(value)(
        item.build() if isinstance(item, BaseSpec) else item for item in value
    )


def _rehydrate_spec_sequence(
    name: str,
    value: list[Any],
    sig: inspect.Signature,
) -> list[Any] | tuple[Any, ...]:
    """Rehydrate spec-dict items in a JSON list, preserving other items."""
    spec_items = [
        create_model_spec_from_json(item)
        if _is_spec_dict(item)
        else _try_deserialize(name, item, sig)
        for item in value
    ]
    return tuple(spec_items) if _expects_tuple_sequence(name, sig) else spec_items


def _resolve_annotation(name: str, value: Any, sig: inspect.Signature) -> Any:
    """Pick the Pydantic field annotation for ``(name, value)`` in ``sig``.

    Order of precedence:

    1. ``value`` is a :class:`BaseSpec` → ``SerializeAsAny[BaseSpec]``
       (preserves the concrete dynamic schema under
       :meth:`~pydantic.BaseModel.model_dump_json`).
    2. ``value`` is a non-empty ``list``/``tuple`` containing
       :class:`BaseSpec` items → ``SerializeAsAny[list[Any]]`` or
       ``SerializeAsAny[tuple[Any, ...]]``. This lets collection fields
       (e.g. ``ComposedLossFunction.components`` and mixed scalar/spec
       weight lists) round-trip by preserving each item's dynamic schema.
    3. The ``__init__`` signature annotates this parameter as a class type
       (``type``, ``type[T]``, or optional variants) → wrap with dotted-path
       class serialization hooks.
    4. The ``__init__`` signature annotates this parameter with a registered
       custom type → wrap via :func:`_wrap_custom_type`.
    5. The ``__init__`` signature has any non-``Any`` annotation → use it.
    6. Otherwise infer from ``type(value)``; if the inferred type is in the
       registry, wrap it; ``None`` values fall back to :class:`typing.Any`.
    """
    if isinstance(value, BaseSpec):
        return SerializeAsAny[BaseSpec]

    if _is_basespec_sequence(value):
        return (
            SerializeAsAny[list[Any]]
            if isinstance(value, list)
            else SerializeAsAny[tuple[Any, ...]]
        )

    param = sig.parameters.get(name)
    sig_ann = param.annotation if param is not None else inspect.Parameter.empty
    if isinstance(sig_ann, str):
        sig_ann = inspect.Parameter.empty
    has_sig_ann = sig_ann is not inspect.Parameter.empty and sig_ann is not Any

    if has_sig_ann:
        class_annotation = _maybe_class_annotation(sig_ann)
        if class_annotation is not None:
            return class_annotation
        registered_annotation = _maybe_registered_type_annotation(sig_ann)
        if registered_annotation is not None:
            return registered_annotation
    if has_sig_ann:
        return sig_ann

    if isinstance(value, type):
        return SerializableTaggedClass

    vt = type(value)
    if vt in _TYPE_SERIALIZERS:
        return _wrap_custom_type(vt)
    return vt if value is not None else Any


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def create_model_spec(target: Any, **kwargs: Any) -> BaseSpec:
    """Build a :class:`BaseSpec` instance for ``target`` with the given kwargs.

    A new Pydantic model class is dynamically created via
    :func:`pydantic.create_model`, one field per kwarg, each annotated by
    :func:`_resolve_annotation`. The resulting spec is JSON-serializable
    with :meth:`~pydantic.BaseModel.model_dump_json` and reconstructible
    with :func:`create_model_spec_from_json`.

    Non-empty ``list``/``tuple`` kwargs containing :class:`BaseSpec`
    items are annotated so each dynamic spec schema survives JSON dump
    and rehydration, and :meth:`BaseSpec.build` then rebuilds each spec
    item while preserving non-spec items. Empty collections are stored
    as-is. Nested collections (e.g. ``list[list[BaseSpec]]``) are not
    traversed; wrap them in a serializable spec object or flatten the
    collection. A JSON round-trip preserves tuple-valued spec sequences
    when the target constructor annotates the parameter as a tuple;
    otherwise JSON arrays rehydrate as lists.

    Parameters
    ----------
    target
        The target importable callable. Must accept all ``**kwargs`` as keyword
        arguments and must not declare any positional-only parameters.
    **kwargs
        Hyperparameters for ``target``. Registered types
        (:class:`torch.Tensor`, :class:`torch.dtype`, :class:`torch.device`,
        and any user-registered types) are handled via the type-serializer
        registry. Other values must themselves be JSON-serializable by
        Pydantic.

    Returns
    -------
    BaseSpec
        A dynamically subclassed :class:`BaseSpec` instance named
        ``"{target.__name__}Spec"`` with one field per kwarg plus the two
        metadata fields.

    Raises
    ------
    TypeError
        If ``target`` has positional-only parameters, or if ``**kwargs``
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
    _check_no_positional_only(target)
    sig = _signature(target)

    unknown = set(kwargs) - set(sig.parameters)
    if unknown:
        var_kw = any(
            p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
        )
        if not var_kw:
            raise TypeError(
                f"Unknown kwargs for {_callable_path_of(target)}: {sorted(unknown)}"
            )

    fields: dict[str, tuple[Any, Any]] = {}
    for name, value in kwargs.items():
        annotation = _resolve_annotation(name, value, sig)
        fields[name] = (annotation, value)

    model_cls = create_model(
        f"{getattr(target, '__name__', type(target).__name__)}Spec",
        __base__=BaseSpec,
        **fields,
    )
    return model_cls(
        cls_path=_callable_path_of(target),
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
        ``cls_path`` cannot be imported / resolves to a non-callable. The
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
        target = _import_callable(cls_path)
    except Exception as e:
        raise ValueError(
            f"Could not resolve cls_path={cls_path!r} while rehydrating spec JSON: {e}"
        ) from e

    sig = _signature(target)
    kwargs: dict[str, Any] = {}
    for name, value in schema.items():
        if _is_spec_dict(value):
            kwargs[name] = create_model_spec_from_json(value)
        elif _is_spec_dict_sequence(value):
            kwargs[name] = _rehydrate_spec_sequence(name, value, sig)
        else:
            # Eagerly deserialize safe unannotated custom forms (tagged class
            # dicts, dtype/device strings, tensor dicts). This keeps raw
            # importable strings as strings while preserving known structured
            # serializer payloads.
            kwargs[name] = _try_deserialize(name, value, sig)

    rebuilt = create_model_spec(target, **kwargs)
    # Preserve original provenance rather than stamping a fresh timestamp.
    object.__setattr__(rebuilt, "timestamp", stored_timestamp)
    return rebuilt
