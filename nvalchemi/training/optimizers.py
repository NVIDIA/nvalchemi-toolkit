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
"""Optimizer configuration and stepping helpers for training strategies."""

from __future__ import annotations

import inspect
from collections.abc import Callable, Iterable, Mapping
from typing import Any, TypeAlias

import torch
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)
from torch.optim.lr_scheduler import LRScheduler, ReduceLROnPlateau

from nvalchemi._serialization import SerializableClass, SerializableOptionalClass
from nvalchemi.training._spec import (
    BaseSpec,
    create_model_spec,
)

OptSchedPair: TypeAlias = tuple[torch.optim.Optimizer, LRScheduler | None]
SchedulerMetricAdapter: TypeAlias = Callable[[dict[str, Any]], float] | str | None

_DEFAULT_METRIC_KEY = "total_loss"

__all__ = [
    "OptSchedPair",
    "OptimizerConfig",
    "SchedulerMetricAdapter",
    "iter_qualified_named_parameters",
    "setup_optimizers",
    "step_lr_schedulers",
    "step_metric_schedulers",
    "step_optimizers",
    "zero_gradients",
]


def _check_kwargs(cls: type, kwargs: Mapping[str, Any], label: str) -> None:
    """Raise ``ValueError`` if ``kwargs`` are not accepted by ``cls.__init__``."""
    try:
        sig = inspect.signature(cls.__init__)
    except (TypeError, ValueError):
        return
    try:
        sig.bind_partial(None, None, **kwargs)
    except TypeError as exc:
        accepted = {
            name
            for name, param in sig.parameters.items()
            if param.kind
            not in {
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            }
        }
        invalid = sorted(set(kwargs) - accepted)
        if not invalid:
            raise ValueError(
                f"Invalid {label} kwargs for {cls.__name__}: {exc}"
            ) from None
        raise ValueError(
            f"Invalid {label} kwargs for {cls.__name__}: {invalid}"
        ) from None


def _normalize_optimizer_configs(
    value: Any,
    *,
    single_model_input: bool,
) -> Any:
    """Normalize accepted optimizer config inputs to named lists."""
    if isinstance(value, OptimizerConfig):
        if not single_model_input and value is not None:
            raise ValueError(
                "Unkeyed optimizer_configs require single-model input; pass "
                "{'model_name': [OptimizerConfig(...)]} for named models."
            )
        return {"main": [value]}
    if isinstance(value, list):
        if not single_model_input:
            raise ValueError(
                "Unkeyed optimizer_configs require single-model input; pass "
                "{'model_name': [...]} for named models."
            )
        return {"main": value}
    if isinstance(value, dict):
        if set(value) == {0}:
            return {"main": value[0]}
        return value
    return value


class OptimizerConfig(BaseModel):
    """Declarative optimizer + optional LR-scheduler bundle.

    Kwargs are validated against each class's ``__init__`` at construction
    time so mistakes surface before training starts. Build the concrete
    ``(optimizer, scheduler)`` pair via :meth:`build`.

    Attributes
    ----------
    optimizer_cls : type[torch.optim.Optimizer]
        Optimizer class; ``optimizer_kwargs`` must match its signature.
    optimizer_kwargs : dict[str, Any]
    scheduler_cls : type | None
        Optional LR scheduler. Time-based schedulers (``StepLR``,
        ``CosineAnnealingLR``, etc.) step every optimizer step.
        Metric-driven schedulers (``ReduceLROnPlateau`` and subclasses)
        step only at validation checkpoints via
        :func:`step_metric_schedulers`.
    scheduler_kwargs : dict[str, Any]
        Must be empty unless ``scheduler_cls`` is set.
    scheduler_metric_adapter : Callable[[dict], float] | str | None
        How a metric-driven scheduler (``ReduceLROnPlateau``) extracts
        its scalar metric from the validation summary dict. A ``str``
        is treated as a key lookup into the summary; a callable
        receives the whole summary dict and returns a ``float``;
        ``None`` uses the default extractor (see
        :func:`_extract_scheduler_metric`).

    Examples
    --------
    >>> import torch
    >>> cfg = OptimizerConfig(
    ...     optimizer_cls=torch.optim.Adam,
    ...     optimizer_kwargs={"lr": 1e-3},
    ...     scheduler_cls=torch.optim.lr_scheduler.StepLR,
    ...     scheduler_kwargs={"step_size": 10, "gamma": 0.1},
    ... )
    """

    optimizer_cls: SerializableClass
    optimizer_kwargs: dict[str, Any] = Field(default_factory=dict)
    scheduler_cls: SerializableOptionalClass = None
    scheduler_kwargs: dict[str, Any] = Field(default_factory=dict)
    scheduler_metric_adapter: SchedulerMetricAdapter = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @model_validator(mode="after")
    def _validate_kwargs(self) -> OptimizerConfig:
        """Validate optimizer/scheduler kwargs against their __init__ signatures."""
        _check_kwargs(self.optimizer_cls, self.optimizer_kwargs, "optimizer")
        if self.scheduler_cls is None:
            if self.scheduler_kwargs:
                raise ValueError(
                    "scheduler_kwargs provided but scheduler_cls is None; "
                    "set scheduler_cls or remove scheduler_kwargs. "
                    f"Got: {sorted(self.scheduler_kwargs)}"
                )
            if self.scheduler_metric_adapter is not None:
                raise ValueError(
                    "scheduler_metric_adapter provided but scheduler_cls is None."
                )
        else:
            _check_kwargs(self.scheduler_cls, self.scheduler_kwargs, "scheduler")
        return self

    def build(self, params: Iterable[torch.nn.Parameter]) -> OptSchedPair:
        """Instantiate the optimizer and optional scheduler for ``params``.

        Parameters
        ----------
        params : Iterable[torch.nn.Parameter]

        Returns
        -------
        tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler | None]
        """
        optimizer = self.optimizer_cls(params, **self.optimizer_kwargs)
        scheduler = (
            self.scheduler_cls(optimizer, **self.scheduler_kwargs)
            if self.scheduler_cls is not None
            else None
        )
        return optimizer, scheduler

    def to_spec(self) -> BaseSpec:
        """Serialize to a :class:`BaseSpec` via :func:`create_model_spec`.

        Returns
        -------
        BaseSpec
            A spec instance that rebuilds the original :class:`OptimizerConfig`.
        """
        return create_model_spec(type(self), **self.model_dump())

    @classmethod
    def from_spec(cls, spec: BaseSpec) -> OptimizerConfig:
        """Rebuild an :class:`OptimizerConfig` from a :class:`BaseSpec`.

        Parameters
        ----------
        spec : BaseSpec
            A spec produced by :meth:`to_spec`.

        Returns
        -------
        OptimizerConfig
            A freshly validated instance equivalent to the original.

        Raises
        ------
        TypeError
            If ``spec`` does not build an :class:`OptimizerConfig`.
        """
        instance = spec.build()
        if not isinstance(instance, cls):
            raise TypeError(
                f"Spec at {spec.cls_path!r} built {type(instance).__name__}, "
                f"expected {cls.__name__}."
            )
        return instance


def iter_qualified_named_parameters(
    models: torch.nn.Module | Mapping[str, torch.nn.Module],
) -> Iterable[tuple[str, torch.nn.Parameter]]:
    """Yield parameters keyed by ``<model_key>.<parameter_name>``.

    Parameters
    ----------
    models : torch.nn.Module | Mapping[str, torch.nn.Module]
        Single model or named model mapping.

    Yields
    ------
    tuple[str, torch.nn.Parameter]
        Fully qualified parameter name and parameter object.
    """
    named_models = {"main": models} if not isinstance(models, Mapping) else models
    for key, model in named_models.items():
        for name, parameter in model.named_parameters():
            yield f"{key}.{name}", parameter


def setup_optimizers(
    models: torch.nn.Module | dict[str, torch.nn.Module] | torch.nn.ModuleDict,
    optimizer_configs: OptimizerConfig
    | list[OptimizerConfig]
    | dict[str, list[OptimizerConfig]],
    *,
    allowed_parameter_names: set[str] | None = None,
) -> dict[str, list[OptSchedPair]]:
    """Build optimizers and schedulers for configured model names.

    Parameters
    ----------
    models : torch.nn.Module | dict[str, torch.nn.Module] | torch.nn.ModuleDict
    optimizer_configs : OptimizerConfig | list[OptimizerConfig] | dict[str, list[OptimizerConfig]]
    allowed_parameter_names : set[str] | None, optional
        Fully qualified parameter names allowed in optimizers. Names use the
        format ``"<model_key>.<parameter_name>"``. When ``None`` all
        ``requires_grad=True`` parameters are eligible.

    Returns
    -------
    dict[str, list[tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler | None]]]

    Raises
    ------
    ValueError
        If a config key is not present in ``models`` or a configured model has
        no trainable parameters.
    """
    named_model_input = isinstance(models, (dict, torch.nn.ModuleDict))
    named_models = dict(models.items()) if named_model_input else {"main": models}
    configs = _normalize_optimizer_configs(
        optimizer_configs, single_model_input=not named_model_input
    )
    result: dict[str, list[OptSchedPair]] = {}
    for key, cfgs in configs.items():
        if key not in named_models:
            raise ValueError(
                f"optimizer_configs key {key!r} is not present in models; "
                f"available model keys: {sorted(named_models)}."
            )
        if allowed_parameter_names is None:
            trainable = [p for p in named_models[key].parameters() if p.requires_grad]
        else:
            trainable = [
                p
                for name, p in iter_qualified_named_parameters({key: named_models[key]})
                if p.requires_grad and name in allowed_parameter_names
            ]
        if not trainable:
            filter_suffix = (
                ""
                if allowed_parameter_names is None
                else " after optimizer parameter filtering"
            )
            raise ValueError(
                f"Configured model {key!r} has no trainable parameters "
                f"(requires_grad=True{filter_suffix})."
            )
        result[key] = [cfg.build(trainable) for cfg in cfgs]
    return result


def zero_gradients(opts: Iterable[torch.optim.Optimizer]) -> None:
    """Call ``zero_grad(set_to_none=True)`` on each optimizer.

    Parameters
    ----------
    opts : Iterable[torch.optim.Optimizer]
    """
    for opt in opts:
        opt.zero_grad(set_to_none=True)


def step_optimizers(opts: Iterable[torch.optim.Optimizer]) -> None:
    """Call ``step()`` on each optimizer.

    Parameters
    ----------
    opts : Iterable[torch.optim.Optimizer]
    """
    for opt in opts:
        opt.step()


def _is_metric_driven(
    scheduler: LRScheduler | ReduceLROnPlateau | None,
) -> bool:
    """Return whether ``scheduler`` is a metric-driven LR scheduler.

    Metric-driven schedulers (``ReduceLROnPlateau`` and subclasses)
    require a scalar metric argument for each ``step()`` call and
    are therefore stepped only at validation checkpoints, not on
    every optimizer step.

    Parameters
    ----------
    scheduler : LRScheduler | ReduceLROnPlateau | None
        Scheduler instance to check.

    Returns
    -------
    bool
        ``True`` when ``scheduler`` is an instance of
        ``ReduceLROnPlateau``.
    """
    return isinstance(scheduler, ReduceLROnPlateau)


def _extract_scheduler_metric(
    summary: dict[str, Any],
    adapter: SchedulerMetricAdapter,
) -> float:
    """Extract a scalar metric from a validation summary for a metric-driven scheduler.

    Parameters
    ----------
    summary : dict[str, Any]
        Validation summary dictionary produced by
        :meth:`~nvalchemi.training._validation._LossAccumulator.summary`.
    adapter : SchedulerMetricAdapter
        Extraction strategy. A callable receives the full summary and
        returns a float. A ``str`` is used as a direct key lookup. When
        ``None``, the default key ``"total_loss"`` is used (the
        aggregate/total validation loss).

    Returns
    -------
    float
        Scalar metric value.

    Raises
    ------
    KeyError
        When a string adapter (or the default key) is not present in
        ``summary``.
    """
    if callable(adapter):
        return float(adapter(summary))
    key = adapter if isinstance(adapter, str) else _DEFAULT_METRIC_KEY
    if key not in summary:
        available = sorted(summary.keys())
        raise KeyError(
            f"Scheduler metric key {key!r} not found in validation summary; "
            f"available keys: {available}"
        )
    return float(summary[key])


def step_lr_schedulers(
    schedulers: Iterable[LRScheduler | ReduceLROnPlateau | None],
) -> None:
    """Call ``step()`` on each non-``None`` time-based scheduler.

    Metric-driven schedulers (``ReduceLROnPlateau``) are skipped here;
    they step at validation checkpoints via :func:`step_metric_schedulers`.

    Parameters
    ----------
    schedulers : Iterable[LRScheduler | ReduceLROnPlateau | None]
    """
    for scheduler in schedulers:
        if scheduler is None or _is_metric_driven(scheduler):
            continue
        scheduler.step()


def step_metric_schedulers(
    schedulers: Iterable[LRScheduler | ReduceLROnPlateau | None],
    adapters: Iterable[SchedulerMetricAdapter],
    summary: dict[str, Any],
) -> None:
    """Step metric-driven schedulers using a validation summary.

    Zips ``schedulers`` with ``adapters`` (positional correspondence
    must match) and calls ``scheduler.step(metric)`` for each
    metric-driven scheduler. Non-metric-driven and ``None``
    schedulers are skipped.

    Parameters
    ----------
    schedulers : Iterable[LRScheduler | ReduceLROnPlateau | None]
        Flat list of schedulers in the same positional order as
        ``adapters``.
    adapters : Iterable[SchedulerMetricAdapter]
        Per-scheduler metric extraction adapters.
    summary : dict[str, Any]
        Validation summary dictionary.
    """
    for scheduler, adapter in zip(schedulers, adapters, strict=True):
        if scheduler is None or not _is_metric_driven(scheduler):
            continue
        scheduler.step(_extract_scheduler_metric(summary, adapter))
