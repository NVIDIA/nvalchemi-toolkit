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
"""Serialization utilities for :class:`nvalchemi.training.strategy.TrainingStrategy`."""

from __future__ import annotations

import importlib
import warnings
from collections.abc import Callable, Mapping
from typing import Any

import torch

from nvalchemi._serialization import _extract_init_kwargs_from_attrs
from nvalchemi.models.base import BaseModelMixin
from nvalchemi.training._spec import (
    BaseSpec,
    create_model_spec,
    create_model_spec_from_json,
)
from nvalchemi.training._strategy_validation import ModelInput, _normalize_models
from nvalchemi.training.losses.composition import ComposedLossFunction
from nvalchemi.training.optimizers import OptimizerConfig


def _resolve_dotted_callable(path: str) -> Callable[..., Any]:
    """Resolve a dotted path ``"module.attribute"`` to a callable."""
    module_path, _, attr = path.rpartition(".")
    if not module_path:
        raise ValueError(
            f"Cannot resolve training_fn from dotted path {path!r}: "
            "expected 'module.attribute'."
        )
    try:
        module = importlib.import_module(module_path)
    except ModuleNotFoundError as exc:
        missing = exc.name or ""
        if missing == module_path or module_path.startswith(f"{missing}."):
            raise ValueError(
                f"Cannot resolve training_fn from dotted path {path!r}: "
                f"module {module_path!r} not found. Expected 'module.attribute'."
            ) from exc
        raise ValueError(
            f"Imported module {module_path!r} failed while resolving "
            f"training_fn {path!r}: missing transitive dependency "
            f"{missing!r}. Install it or fix the import inside "
            f"{module_path!r}."
        ) from exc
    except ImportError as exc:
        raise ValueError(
            f"Imported module {module_path!r} failed while resolving "
            f"training_fn {path!r}: {exc}. Check imports and dependencies "
            "inside that module."
        ) from exc
    try:
        obj = getattr(module, attr)
    except AttributeError as exc:
        raise ValueError(
            f"Cannot resolve training_fn from dotted path {path!r}: "
            f"module {module_path!r} has no attribute {attr!r}."
        ) from exc
    if not callable(obj):
        raise ValueError(
            f"{path!r} resolves to {type(obj).__name__}, which is not callable."
        )
    return obj


def _callable_dotted_path(fn: Callable[..., Any]) -> str:
    """Return ``"module.name"`` for a module-level callable or raise ``ValueError``."""
    module = getattr(fn, "__module__", None)
    qualname = getattr(fn, "__qualname__", None)
    name = getattr(fn, "__name__", None)
    if not module or not qualname:
        raise ValueError(
            f"training_fn is not serializable — {type(fn).__name__} "
            "lacks __module__ / __qualname__. Only importable "
            "module-level callables can be written to spec."
        )
    if "<lambda>" in qualname or "<locals>" in qualname:
        raise ValueError(
            f"training_fn is not serializable — {qualname!r} is a lambda "
            "or local function. Only importable module-level callables "
            "can be written to spec."
        )
    if name is None or qualname != name:
        raise ValueError(
            f"training_fn is not serializable — {qualname!r} is not a "
            "module-level callable (nested class/function or bound method). "
            "Only importable module-level callables can be written to spec."
        )
    return f"{module}.{qualname}"


def _model_specs_from_models(
    models: dict[str, BaseModelMixin],
) -> dict[str, dict[str, Any]]:
    """Best-effort ``BaseSpec`` dumps for importable model constructors."""
    specs: dict[str, dict[str, Any]] = {}
    for key, model in models.items():
        try:
            spec = _model_provided_checkpoint_spec(model)
            validate_rebuild = spec is None
            if spec is None:
                spec = _module_spec_from_attrs(model)
            if validate_rebuild:
                rebuilt = create_model_spec_from_json(spec.model_dump()).build()
                if not isinstance(rebuilt, BaseModelMixin):
                    raise TypeError(
                        f"rebuilt {type(rebuilt).__name__}, expected BaseModelMixin"
                    )
                rebuilt.to(torch.device("cpu"))
            specs[key] = spec.model_dump()
        except (TypeError, ValueError, AttributeError) as exc:
            warnings.warn(
                f"Omitting model spec for {key!r}: {exc}",
                UserWarning,
                stacklevel=2,
            )
    return specs


def _model_provided_checkpoint_spec(module: torch.nn.Module) -> BaseSpec | None:
    """Return an explicit model-provided checkpoint spec, if available."""
    if isinstance(module, torch.nn.parallel.DistributedDataParallel):
        module = module.module
    checkpoint_spec = getattr(module, "checkpoint_spec", None)
    if not callable(checkpoint_spec):
        return None
    spec = checkpoint_spec()
    if spec is None:
        return None
    if not isinstance(spec, BaseSpec):
        raise TypeError(
            "checkpoint_spec() must return a BaseSpec or None; got "
            f"{type(spec).__name__}."
        )
    return spec


def _module_spec_from_attrs(module: torch.nn.Module) -> BaseSpec:
    """Build a recursive spec from constructor-matching module attributes."""
    if isinstance(module, torch.nn.parallel.DistributedDataParallel):
        module = module.module
    kwargs = _extract_init_kwargs_from_attrs(module)
    for name, value in list(kwargs.items()):
        if isinstance(value, torch.nn.Module):
            kwargs[name] = _module_spec_from_attrs(value)
    return create_model_spec(type(module), **kwargs)


def _models_from_spec_dict(
    spec_models: Mapping[str, Any],
) -> dict[str, BaseModelMixin]:
    """Build serialized model specs, omitting entries that fail to rebuild."""
    models: dict[str, BaseModelMixin] = {}
    for key, raw in spec_models.items():
        if not isinstance(raw, Mapping):
            warnings.warn(
                f"Omitting model spec for {key!r}: expected BaseSpec dict, "
                f"got {type(raw).__name__}.",
                UserWarning,
                stacklevel=2,
            )
            continue
        try:
            model = create_model_spec_from_json(dict(raw)).build()
        except (TypeError, ValueError, AttributeError) as exc:
            warnings.warn(
                f"Omitting model spec for {key!r}: {exc}",
                UserWarning,
                stacklevel=2,
            )
            continue
        if not isinstance(model, BaseModelMixin):
            warnings.warn(
                f"Omitting model spec for {key!r}: built "
                f"{type(model).__name__}, expected BaseModelMixin.",
                UserWarning,
                stacklevel=2,
            )
            continue
        models[key] = model
    return models


def _optimizer_configs_from_spec(raw: Any) -> dict[str, list[OptimizerConfig]]:
    """Rebuild named optimizer configs from a serialized spec field."""
    if not isinstance(raw, Mapping):
        raise ValueError(
            "from_spec_dict: 'optimizer_configs' must be a mapping of "
            f"str -> list[dict]; got {type(raw).__name__}."
        )
    optimizer_configs: dict[str, list[OptimizerConfig]] = {}
    for raw_key, entries in raw.items():
        if not isinstance(raw_key, str):
            raise ValueError(
                "from_spec_dict: 'optimizer_configs' keys must be strings; "
                f"got key of type {type(raw_key).__name__}."
            )
        if not isinstance(entries, list) or not all(
            isinstance(entry, Mapping) for entry in entries
        ):
            raise ValueError(
                f"from_spec_dict: 'optimizer_configs[{raw_key!r}]' must "
                "be a list of OptimizerConfig spec dicts."
            )
        key = "main" if raw_key == "0" else raw_key
        optimizer_configs[key] = [
            OptimizerConfig.from_spec(create_model_spec_from_json(entry))
            for entry in entries
        ]
    return optimizer_configs


def _devices_from_spec(raw: Any) -> list[torch.device]:
    """Rebuild device strings from a serialized spec field."""
    if not isinstance(raw, list) or not all(isinstance(device, str) for device in raw):
        raise ValueError(
            "from_spec_dict: 'devices' must be a list of device strings; "
            f"got {type(raw).__name__}."
        )
    return [torch.device(device) for device in raw]


def _loss_fn_from_spec(raw: Any) -> ComposedLossFunction:
    """Rebuild the composed loss from a serialized spec field."""
    if not isinstance(raw, Mapping):
        raise ValueError(
            "from_spec_dict: 'loss_fn_spec' must be a BaseSpec dump dict; "
            f"got {type(raw).__name__}."
        )
    loss_fn = create_model_spec_from_json(raw).build()
    if not isinstance(loss_fn, ComposedLossFunction):
        raise ValueError(
            f"loss_fn_spec built {type(loss_fn).__name__}, expected "
            "ComposedLossFunction."
        )
    return loss_fn


def _training_fn_from_spec(
    spec: Mapping[str, Any],
    override: Callable[..., Mapping[str, torch.Tensor]] | str | None,
) -> Callable[..., Mapping[str, torch.Tensor]] | str:
    """Resolve runtime or serialized training function input."""
    if override is not None:
        return override
    raw = spec.get("training_fn")
    if raw is None:
        raise ValueError(
            "from_spec_dict: no training_fn was supplied and the spec does "
            "not contain one. Pass training_fn=... explicitly."
        )
    if not isinstance(raw, str):
        raise ValueError(
            "from_spec_dict: 'training_fn' must be a dotted-path string "
            f"('module.attribute'); got {type(raw).__name__}."
        )
    return _resolve_dotted_callable(raw)


def _models_from_spec_and_overrides(
    spec_models_raw: Any,
    runtime_models: ModelInput | None,
    *,
    single_model_input: bool | None = None,
) -> ModelInput:
    """Build spec models, apply runtime overrides, and preserve call mode."""
    if not isinstance(spec_models_raw, Mapping):
        raise ValueError(
            "from_spec_dict: 'model_specs' must be a mapping when present; "
            f"got {type(spec_models_raw).__name__}."
        )
    merged = _models_from_spec_dict(spec_models_raw)
    if runtime_models is not None:
        merged.update(_normalize_models(runtime_models))
    # Return shape intentionally preserves the public call-mode distinction:
    # ``models=model`` means ``training_fn(model, batch)``, while
    # ``models={"main": model}`` means ``training_fn(models, batch)``.
    if isinstance(runtime_models, BaseModelMixin) and set(merged) == {"main"}:
        return merged["main"]
    if runtime_models is not None:
        return merged
    if single_model_input is True and set(merged) == {"main"}:
        return merged["main"]
    if single_model_input is False:
        return merged
    if set(merged) == {"main"}:
        return merged["main"]
    return merged


def _single_model_input_from_spec(raw: Any) -> bool | None:
    """Return serialized call mode or ``None`` for legacy specs."""
    if raw is None:
        return None
    if not isinstance(raw, bool):
        raise ValueError(
            "from_spec_dict: 'single_model_input' must be a bool when present; "
            f"got {type(raw).__name__}."
        )
    return raw
