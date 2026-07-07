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
"""Validation helpers for :mod:`nvalchemi.training.strategy`."""

from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping
from typing import Any, TypeAlias, get_origin, get_type_hints

from torch.nn import ModuleDict

from nvalchemi.models.base import BaseModelMixin

ModelInput: TypeAlias = BaseModelMixin | dict[str, BaseModelMixin] | ModuleDict
_TRAINING_FN_REQUIRED_MESSAGE = (
    "training_fn must be provided explicitly. To opt into the stock "
    "single-model behavior, use `from nvalchemi.training import "
    "default_training_fn` or `from nvalchemi.training.strategy import "
    "default_training_fn`."
)


def _normalize_models(value: Any) -> Any:
    """Normalize model inputs to a plain named-model dict."""
    if isinstance(value, BaseModelMixin):
        return {"main": value}
    if isinstance(value, ModuleDict):
        value = dict(value.items())
    if isinstance(value, dict):
        invalid = {
            key: type(model).__name__
            for key, model in value.items()
            if not isinstance(model, BaseModelMixin)
        }
        if invalid:
            raise ValueError(
                "models must map names to BaseModelMixin instances; "
                f"invalid entries: {invalid}."
            )
        return dict(value)
    return value


def _callable_accepts_two_args(fn: Callable[..., Any]) -> bool:
    """Return whether ``fn`` can be called with exactly two positional args."""
    sig = inspect.signature(fn)
    try:
        sig.bind(object(), object())
    except TypeError:
        return False
    return True


def _first_parameter_annotation(fn: Callable[..., Any]) -> Any:
    """Return the first parameter annotation, resolving type hints when possible."""
    sig = inspect.signature(fn)
    try:
        first = next(iter(sig.parameters.values()))
    except StopIteration:
        return inspect.Parameter.empty
    try:
        hints = get_type_hints(fn)
    except (NameError, TypeError, AttributeError):
        hints = getattr(fn, "__annotations__", {})
    return hints.get(first.name, first.annotation)


def _is_mapping_model_annotation(annotation: Any) -> bool:
    """Return whether annotation clearly means named model mapping."""
    if annotation in (Any, inspect.Parameter.empty):
        return False
    origin = get_origin(annotation)
    if origin is dict or origin is Mapping:
        args = getattr(annotation, "__args__", ())
        return len(args) == 2 and args[0] is str and _is_model_annotation(args[1])
    try:
        return isinstance(annotation, type) and issubclass(annotation, ModuleDict)
    except TypeError:
        return False


def _is_model_annotation(annotation: Any) -> bool:
    """Return whether annotation clearly means ``BaseModelMixin`` or subclass."""
    if annotation in (Any, inspect.Parameter.empty):
        return False
    try:
        return isinstance(annotation, type) and issubclass(annotation, BaseModelMixin)
    except TypeError:
        return False


def _validate_training_fn_call_shape(
    fn: Callable[..., Any],
    *,
    single_model_input: bool,
) -> None:
    """Validate ``training_fn`` arity and obvious first-argument mismatches."""
    if not _callable_accepts_two_args(fn):
        raise ValueError(
            "training_fn must accept exactly the two arguments "
            "(model_or_models, batch) without requiring additional args."
        )
    annotation = _first_parameter_annotation(fn)
    if single_model_input and _is_mapping_model_annotation(annotation):
        raise ValueError(
            "single-model strategies call training_fn(model, batch), but the "
            "first parameter is annotated as a model mapping."
        )
    if not single_model_input and _is_model_annotation(annotation):
        raise ValueError(
            "named-model strategies call training_fn(models, batch), but the "
            "first parameter is annotated as a single BaseModelMixin. Pass "
            "models=model for single-model behavior, or define "
            "training_fn(models: dict[str, BaseModelMixin], batch)."
        )
