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
"""Validation configuration, shared helpers, and the :class:`ValidationLoop` orchestrator.

This module contains :class:`ValidationConfig`, :class:`ValidationLoop`,
and the low-level utilities used by
:meth:`~nvalchemi.training.TrainingStrategy.validate` validation passes.
"""

from __future__ import annotations

import contextlib
import dataclasses
from collections.abc import Callable, Iterable, Mapping
from contextlib import AbstractContextManager
from types import TracebackType
from typing import TYPE_CHECKING, Annotated, Any, Literal, Protocol, runtime_checkable

import torch
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PlainValidator,
    field_validator,
    model_validator,
)
from torch import nn

from nvalchemi.data import Batch
from nvalchemi.training._stages import TrainingStage
from nvalchemi.training.distributed import (
    all_reduce as distributed_all_reduce,
)
from nvalchemi.training.distributed import (
    is_distributed_initialized,
)
from nvalchemi.training.losses.composition import (
    ComposedLossFunction,
    ComposedLossOutput,
    LossTargetAssemblyProtocol,
    as_composed_loss,
    assemble_loss_targets,
    compute_supervised_loss,
)

if TYPE_CHECKING:
    from nvalchemi.training.strategy import TrainingStrategy

__all__ = ["BatchValidationCallback", "ValidationConfig", "ValidationLoop"]


@runtime_checkable
class BatchValidationCallback(Protocol):
    """Protocol for an optional per-batch validation callback.

    A user-supplied object implementing this protocol is invoked once per
    validation batch inside :meth:`ValidationLoop.execute`, immediately
    after predictions and the per-batch loss are computed. It is the
    extension point for streaming per-batch outputs (e.g. predictions or
    diagnostics) to a custom logging or storage system.

    Summary-level logging does not require this callback: register a hook
    on :attr:`~nvalchemi.training.TrainingStage.AFTER_VALIDATION` and read
    the validation summary from ``ctx.validation``.

    Notes
    -----
    No concrete implementation is provided. Users supply their own.
    """

    def __call__(
        self,
        *,
        batch: Batch,
        predictions: Mapping[str, torch.Tensor],
        loss: ComposedLossOutput,
        batch_count: int,
        step_count: int,
        epoch: int,
    ) -> None:
        """Consume one validation batch's predictions and loss.

        Parameters
        ----------
        batch : Batch
            The validation batch that was evaluated.
        predictions : Mapping[str, torch.Tensor]
            The output of the validation function for this batch.
        loss : ComposedLossOutput
            The per-batch composed loss output.
        batch_count : int
            Zero-based index of this batch within the validation pass.
        step_count : int
            Training step count at which this validation pass runs.
        epoch : int
            Training epoch at which this validation pass runs.
        """
        ...


def _ensure_reiterable_validation_data(value: Any) -> Any:
    """Reject one-shot iterators so validation can restart each pass.

    Parameters
    ----------
    value : Any
        Candidate ``validation_data``. Must be a re-iterable container
        (e.g. ``list``, ``DataLoader``, ``Dataset``) whose ``__iter__``
        returns a fresh iterator each call.

    Returns
    -------
    Any
        The value unchanged when it is re-iterable.

    Raises
    ------
    ValueError
        When ``value`` is not iterable at all, or when it is a one-shot
        iterator (e.g. a generator) that cannot be re-iterated across
        repeated validation passes.
    """
    try:
        iterator = iter(value)
    except TypeError as exc:
        raise ValueError(
            "validation_data must be iterable (e.g. a list, DataLoader, or "
            f"Dataset of Batch); got {type(value).__name__}."
        ) from exc
    if iterator is value:
        raise ValueError(
            "validation_data must be a re-iterable container, not a one-shot "
            "iterator/generator. Validation runs multiple times and must "
            "restart from the beginning each pass; pass a list (or a "
            "re-iterable DataLoader/Dataset) instead of a generator."
        )
    return value


class ValidationConfig(BaseModel):
    """Configuration for strategy-owned validation passes.

    ``ValidationConfig`` is a plain data object consumed by
    ``TrainingStrategy.validate()`` via :class:`ValidationLoop`.
    It does NOT drive hook dispatch — the strategy reads it directly.

    Attributes
    ----------
    validation_data : Iterable[Batch]
        Re-iterable container (e.g. ``list``, ``DataLoader``, ``Dataset``)
        yielding :class:`~nvalchemi.data.Batch` instances. The strategy
        re-iterates this on every validation pass; one-shot generators
        and bare iterators are rejected at construction time.
    validation_fn : Callable | None
        Validation forward callable. ``None`` means use the strategy's
        ``training_fn`` with the same single-model or named-model call
        convention.
    loss_fn : ComposedLossFunction | None
        Validation loss function. ``None`` means use the strategy's
        ``loss_fn``. Leaf losses are auto-normalized to a
        :class:`ComposedLossFunction` via :func:`as_composed_loss`.
    every_n_epochs : int | None
        Run validation after every *n*-th completed epoch. Mutually
        exclusive with ``every_n_steps``.
    every_n_steps : int | None
        Run validation after every *n*-th completed optimizer step.
        Mutually exclusive with ``every_n_epochs``.
    grad_mode : {"auto", "enabled", "disabled"}
        Autograd policy during validation. ``"auto"`` enables gradients
        when any loss component has ``requires_eval_grad=True`` and
        disables them when all components report ``False``.
    set_eval : bool
        If ``True``, set validation modules to eval mode and restore
        their original training modes afterward.
    use_ema : {"auto", "always", "never"}
        Whether the strategy's ``inference_model`` slot (populated by
        EMA) should replace live training weights for validation.
    use_mixed_precision : {"auto", "always", "never"}
        Whether to reuse a registered :class:`MixedPrecisionHook`
        autocast context for validation inference.
    batch_callback : BatchValidationCallback | None
        Optional user-supplied callable invoked once per validation
        batch with the batch, predictions, and per-batch loss output.
        Use it to stream per-sample diagnostics to a custom logging or
        storage backend. ``None`` disables per-batch callbacks. For
        epoch-level (summary) logging, register a hook on the
        ``AFTER_VALIDATION`` stage and read ``ctx.validation`` instead.
    name : str
        Name stored in the validation summary dictionary.
    """

    validation_data: Annotated[
        Iterable[Batch], PlainValidator(_ensure_reiterable_validation_data)
    ]
    validation_fn: Callable[..., Any] | None = None
    loss_fn: ComposedLossFunction | None = None
    every_n_epochs: int | None = Field(default=None, ge=1)
    every_n_steps: int | None = Field(default=None, ge=1)
    grad_mode: Literal["auto", "enabled", "disabled"] = "auto"
    set_eval: bool = True
    use_ema: Literal["auto", "always", "never"] = "auto"
    use_mixed_precision: Literal["auto", "always", "never"] = "auto"
    batch_callback: BatchValidationCallback | None = None
    name: str = Field(default="validation", min_length=1)

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
    )

    @field_validator("loss_fn", mode="before")
    @classmethod
    def _normalize_loss_fn(cls, value: Any) -> ComposedLossFunction | None:
        """Normalize a leaf loss into a one-component composed loss."""
        return None if value is None else as_composed_loss(value)

    @model_validator(mode="after")
    def _validate_schedule(self) -> ValidationConfig:
        """Enforce mutual exclusion of ``every_n_epochs`` and ``every_n_steps``."""
        if self.every_n_epochs is not None and self.every_n_steps is not None:
            raise ValueError("Only one of every_n_epochs or every_n_steps may be set.")
        return self


# ------------------------------------------------------------------
# Shared validation utilities
# ------------------------------------------------------------------


def _unique_modules(modules: Iterable[nn.Module]) -> tuple[nn.Module, ...]:
    """Return unique modules while preserving first-seen order."""
    seen: set[int] = set()
    unique: list[nn.Module] = []
    for module in modules:
        if id(module) in seen:
            continue
        seen.add(id(module))
        unique.append(module)
    return tuple(unique)


def _module_training_modes(
    modules: Iterable[nn.Module],
) -> dict[int, tuple[nn.Module, bool]]:
    """Snapshot unique module training modes for later restoration."""
    modes: dict[int, tuple[nn.Module, bool]] = {}
    for module in modules:
        if id(module) not in modes:
            modes[id(module)] = (module, module.training)
    return modes


def _snapshot_parameter_grads(
    modules: Iterable[nn.Module],
) -> dict[int, tuple[nn.Parameter, torch.Tensor | None]]:
    """Clone current parameter gradients so validation can restore them."""
    snapshot: dict[int, tuple[nn.Parameter, torch.Tensor | None]] = {}
    for module in modules:
        for parameter in module.parameters():
            if id(parameter) in snapshot:
                continue
            grad = parameter.grad
            snapshot[id(parameter)] = (
                parameter,
                None if grad is None else grad.detach().clone(),
            )
    return snapshot


def _clear_parameter_grads(modules: Iterable[nn.Module]) -> None:
    """Clear parameter gradients on validation modules."""
    for module in modules:
        for parameter in module.parameters():
            parameter.grad = None


def _restore_parameter_grads(
    snapshot: Mapping[int, tuple[nn.Parameter, torch.Tensor | None]],
) -> None:
    """Restore parameter gradients captured by :func:`_snapshot_parameter_grads`."""
    for parameter, grad in snapshot.values():
        parameter.grad = grad


def _tensor_to_cpu(value: torch.Tensor) -> torch.Tensor:
    """Detach a scalar summary tensor and move it to CPU."""
    return value.detach().cpu()


def _as_float64_scalar(value: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Detach ``value`` and return a scalar float64 tensor on ``device``."""
    return value.detach().to(device=device, dtype=torch.float64).reshape(-1).sum()


class _LossAccumulator:
    """Accumulate composed-loss diagnostics over validation batches."""

    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.batch_count = 0
        self.total_sum: torch.Tensor | None = None
        self.per_component_unweighted_sum: dict[str, torch.Tensor] = {}
        self.per_component_sample_sum: dict[str, torch.Tensor] = {}
        self.per_component_sample_count: dict[str, int] = {}
        self.per_component_weight: dict[str, float] = {}
        self.per_component_raw_weight: dict[str, float] = {}

    def update(self, loss_out: ComposedLossOutput) -> None:
        """Add one batch's loss output to the running totals."""
        self.batch_count += 1
        total = loss_out["total_loss"].detach()
        self.total_sum = total if self.total_sum is None else self.total_sum + total
        for name, value in loss_out["per_component_unweighted"].items():
            detached = value.detach()
            previous = self.per_component_unweighted_sum.get(name)
            self.per_component_unweighted_sum[name] = (
                detached if previous is None else previous + detached
            )
        for name, sample in loss_out["per_component_sample"].items():
            detached_sum = sample.detach().sum()
            previous = self.per_component_sample_sum.get(name)
            self.per_component_sample_sum[name] = (
                detached_sum if previous is None else previous + detached_sum
            )
            self.per_component_sample_count[name] = (
                self.per_component_sample_count.get(name, 0) + sample.numel()
            )
        self.per_component_weight = dict(loss_out["per_component_weight"])
        self.per_component_raw_weight = dict(loss_out["per_component_raw_weight"])

    def summary(
        self,
        *,
        name: str,
        model_source: str,
        ema_model_keys: tuple[str, ...],
        precision: str,
        distributed_manager: Any | None = None,
    ) -> dict[str, Any]:
        """Return the local or distributed-reduced validation summary."""
        if self.batch_count == 0 or self.total_sum is None:
            raise ValueError("validation_data produced no batches.")

        component_keys = tuple(sorted(self.per_component_unweighted_sum))
        sample_keys = tuple(sorted(self.per_component_sample_sum))
        values = [
            _as_float64_scalar(self.total_sum, self.device),
            torch.tensor(
                float(self.batch_count), device=self.device, dtype=torch.float64
            ),
        ]
        values.extend(
            _as_float64_scalar(self.per_component_unweighted_sum[key], self.device)
            for key in component_keys
        )
        for key in sample_keys:
            values.append(
                _as_float64_scalar(self.per_component_sample_sum[key], self.device)
            )
            values.append(
                torch.tensor(
                    float(self.per_component_sample_count[key]),
                    device=self.device,
                    dtype=torch.float64,
                )
            )
        packed = torch.stack(values)
        distributed_reduced = _distributed_sum_in_place(packed, distributed_manager)

        index = 0
        total_sum = packed[index]
        index += 1
        batch_count = packed[index]
        index += 1
        reduced_batch_count = int(batch_count.item())

        per_component_unweighted: dict[str, torch.Tensor] = {}
        for key in component_keys:
            per_component_unweighted[key] = _tensor_to_cpu(packed[index] / batch_count)
            index += 1

        per_component_sample: dict[str, torch.Tensor] = {}
        sample_counts: dict[str, int] = {}
        for key in sample_keys:
            sample_sum = packed[index]
            index += 1
            sample_count = packed[index]
            index += 1
            sample_counts[key] = int(sample_count.item())
            per_component_sample[key] = _tensor_to_cpu(sample_sum / sample_count)

        return {
            "name": name,
            "total_loss": _tensor_to_cpu(total_sum / batch_count),
            "per_component_unweighted": per_component_unweighted,
            "per_component_weight": dict(self.per_component_weight),
            "per_component_raw_weight": dict(self.per_component_raw_weight),
            "per_component_sample": per_component_sample,
            "num_batches": reduced_batch_count,
            "per_component_sample_count": sample_counts,
            "model_source": model_source,
            "ema_model_keys": list(ema_model_keys),
            "precision": precision,
            "distributed_reduced": distributed_reduced,
        }


def _distributed_sum_in_place(
    value: torch.Tensor, distributed_manager: Any | None
) -> bool:
    """All-reduce ``value`` when distributed communication is active."""
    if not is_distributed_initialized(distributed_manager):
        return False
    distributed_all_reduce(value, distributed_manager)
    return True


# ------------------------------------------------------------------
# Internal context accessor for ValidationLoop
# ------------------------------------------------------------------


@dataclasses.dataclass
class _LoopContext:
    """Snapshot of counters and handles consumed by :class:`ValidationLoop`.

    Attributes
    ----------
    step_count : int
        Current optimizer step count.
    epoch : int
        Current epoch count.
    distributed_manager : Any | None
        Distributed manager handle.
    num_models : int
        Total number of models in the workflow.
    """

    step_count: int
    epoch: int
    distributed_manager: Any | None
    num_models: int


def _resolve_grad_from_config(
    config: ValidationConfig,
    loss_fn: ComposedLossFunction,
) -> bool:
    """Resolve the autograd policy from a :class:`ValidationConfig`.

    Parameters
    ----------
    config : ValidationConfig
        Validation configuration containing the ``grad_mode`` policy.
    loss_fn : ComposedLossFunction
        The resolved validation loss function used to infer gradient
        requirements when ``grad_mode='auto'``.

    Returns
    -------
    bool
        ``True`` when validation should run with gradients enabled.
    """
    if config.grad_mode == "enabled":
        return True
    if config.grad_mode == "disabled":
        return False
    return loss_fn.requires_eval_grad()


def _resolve_model_arg(
    strategy: TrainingStrategy,
    config: ValidationConfig,
) -> tuple[Any, tuple[nn.Module, ...], tuple[str, ...]]:
    """Resolve the model argument for a strategy-integrated validation pass.

    Reads the strategy-owned ``inference_model`` slot and falls back
    to live training models for keys not covered by the slot.

    Parameters
    ----------
    strategy : TrainingStrategy
        The training strategy owning the validation pass.
    config : ValidationConfig
        The resolved validation configuration.

    Returns
    -------
    tuple[Any, tuple[nn.Module, ...], tuple[str, ...]]
        A three-element tuple:

        * **model_arg** -- The value passed to the validation forward
          callable. A single :class:`nn.Module` for single-model
          strategies, or a ``dict[str, ...]`` for named-model
          strategies.
        * **modules** -- All unique :class:`nn.Module` instances
          participating in the forward pass (for training-mode
          management).
        * **ema_keys** -- Sorted tuple of model keys that were
          sourced from the ``inference_model`` slot rather than
          live training weights.

    Raises
    ------
    RuntimeError
        When ``use_ema='always'`` and the ``inference_model`` slot
        cannot satisfy the requirement (empty slot or missing keys).
    """
    use_ema = config.use_ema
    slot = strategy.inference_model

    if use_ema == "never":
        slot = None

    if use_ema == "always" and slot is None:
        raise RuntimeError(
            "ValidationConfig use_ema='always' requires a populated "
            "inference_model slot (e.g. via EMAHook)."
        )

    if strategy.single_model_input:
        live = strategy.models["main"]
        if isinstance(slot, nn.Module) and not isinstance(slot, nn.ModuleDict):
            model = slot
            ema_keys: tuple[str, ...] = ("main",)
        else:
            model = live
            ema_keys = ()
        return model, (model,), ema_keys

    # Named-model path
    resolved: dict[str, Any] = dict(strategy.models)
    used_ema_keys: list[str] = []

    if isinstance(slot, nn.ModuleDict):
        for key in list(slot.keys()):
            if key in resolved:
                resolved[key] = slot[key]
                used_ema_keys.append(key)
    elif isinstance(slot, nn.Module):
        if "main" in resolved:
            resolved["main"] = slot
            used_ema_keys.append("main")

    if use_ema == "always":
        missing = sorted(set(resolved) - set(used_ema_keys))
        if missing:
            raise RuntimeError(
                "ValidationConfig use_ema='always' requires the "
                "inference_model slot to cover every model key; "
                f"missing: {missing}."
            )

    modules = tuple(
        value for value in resolved.values() if isinstance(value, nn.Module)
    )
    return resolved, _unique_modules(modules), tuple(sorted(used_ema_keys))


# ------------------------------------------------------------------
# ValidationLoop — public context-manager orchestrator
# ------------------------------------------------------------------


class ValidationLoop:
    """Context-manager orchestrator for a single validation pass.

    ``ValidationLoop`` encapsulates the full validation lifecycle —
    setup, per-batch forward + loss accumulation, distributed summary
    reduction, sink writes, and teardown — in a single reusable object.

    Two construction paths are supported:

    * **Standalone** via :meth:`__init__`: caller provides all
      dependencies explicitly. No strategy or hook scanning.
    * **Strategy-integrated** via :meth:`from_training_strategy`:
      reads capabilities through strategy introspection and holds
      a live reference for counter/model access during ``execute()``.

    Usage::

        with ValidationLoop.from_training_strategy(strategy) as loop:
            summary = loop.execute()

    Parameters
    ----------
    validation_data : Iterable[Batch]
        Re-iterable object yielding validation batches.
    config : ValidationConfig
        Validation configuration.
    device : torch.device
        Primary device for the validation pass.
    model : nn.Module | None
        Single model for single-model validation. Mutually exclusive
        with ``models``.
    models : dict[str, nn.Module] | None
        Named models for named-model validation. Mutually exclusive
        with ``model``.
    loss_fn : ComposedLossFunction | None
        Validation loss function. Falls back to ``config.loss_fn``
        when ``None``.
    validation_fn : Callable[..., Any] | None
        Validation forward callable. Required in standalone mode.
    inference_model : nn.Module | nn.ModuleDict | None
        Optional EMA/inference model to swap in during validation.
    autocast : Callable[[], AbstractContextManager[None]] | None
        Precision context factory. ``None`` uses
        :func:`contextlib.nullcontext` and precision label ``"float32"``.
    grad_enabled : bool | None
        Autograd policy. ``None`` infers from ``config.grad_mode``
        and ``loss_fn.requires_eval_grad()``.
    distributed_manager : Any | None
        Optional distributed manager for all-reduce and barrier ops.
    step_count : int
        Optimizer step counter for sink metadata.
    epoch : int
        Epoch counter for sink metadata.

    Raises
    ------
    ValueError
        When both or neither of ``model``/``models`` are supplied,
        or when required arguments (``loss_fn``, ``validation_fn``)
        are missing.
    """

    def __init__(
        self,
        *,
        validation_data: Iterable[Batch],
        config: ValidationConfig,
        device: torch.device,
        model: nn.Module | None = None,
        models: dict[str, nn.Module] | None = None,
        loss_fn: ComposedLossFunction | None = None,
        loss_target_assembler: LossTargetAssemblyProtocol = assemble_loss_targets,
        validation_fn: Callable[..., Any] | None = None,
        inference_model: nn.Module | nn.ModuleDict | None = None,
        autocast: Callable[[], AbstractContextManager[None]] | None = None,
        grad_enabled: bool | None = None,
        distributed_manager: Any | None = None,
        step_count: int = 0,
        epoch: int = 0,
    ) -> None:
        have_model = model is not None
        have_models = models is not None
        if have_model == have_models:
            raise ValueError("Exactly one of 'model' or 'models' must be provided.")

        resolved_loss_fn = loss_fn if loss_fn is not None else config.loss_fn
        if resolved_loss_fn is None:
            raise ValueError(
                "loss_fn must be provided either directly or via "
                "config.loss_fn in standalone mode."
            )
        resolved_loss_fn = as_composed_loss(resolved_loss_fn)

        if validation_fn is None:
            raise ValueError("validation_fn is required in standalone mode.")

        if autocast is not None:
            self._precision_context = autocast
            self._precision = "mixed"
        else:
            self._precision_context: Callable[[], AbstractContextManager[None]] = (
                contextlib.nullcontext
            )
            self._precision = "float32"

        if grad_enabled is None:
            grad_enabled = _resolve_grad_from_config(config, resolved_loss_fn)

        self._validation_data = validation_data
        self._config = config
        self._device = device
        self._loss_fn = resolved_loss_fn
        self._loss_target_assembler = loss_target_assembler
        self._validation_fn = validation_fn
        self._grad_enabled = grad_enabled

        # Resolve model_arg, modules, ema_model_keys for standalone path
        if have_model:
            assert model is not None  # noqa: S101  # narrowing
            self._single_model_input = True
            ema_keys: tuple[str, ...] = ()
            if (
                inference_model is not None
                and isinstance(inference_model, nn.Module)
                and not isinstance(inference_model, nn.ModuleDict)
            ):
                effective_model = inference_model
                ema_keys = ("main",)
            else:
                effective_model = model
            self._model_arg: Any = effective_model
            self._modules = _unique_modules((effective_model,))
            self._ema_model_keys = ema_keys
            self._num_models = 1
        else:
            assert models is not None  # noqa: S101  # narrowing
            self._single_model_input = False
            resolved: dict[str, Any] = dict(models)
            used_ema_keys: list[str] = []
            if isinstance(inference_model, nn.ModuleDict):
                for key in list(inference_model.keys()):
                    if key in resolved:
                        resolved[key] = inference_model[key]
                        used_ema_keys.append(key)
            elif isinstance(inference_model, nn.Module):
                if "main" in resolved:
                    resolved["main"] = inference_model
                    used_ema_keys.append("main")
            mods = tuple(v for v in resolved.values() if isinstance(v, nn.Module))
            self._model_arg = resolved
            self._modules = _unique_modules(mods)
            self._ema_model_keys = tuple(sorted(used_ema_keys))
            self._num_models = len(models)

        # Standalone context: fixed values
        self._strategy: TrainingStrategy | None = None
        self._standalone_context = _LoopContext(
            step_count=step_count,
            epoch=epoch,
            distributed_manager=distributed_manager,
            num_models=self._num_models,
        )
        self._successful = False
        self._entered = False
        self._modes: dict[int, tuple[nn.Module, bool]] = {}
        self._grad_snapshot: dict[int, tuple[nn.Parameter, torch.Tensor | None]] = {}

    @classmethod
    def from_training_strategy(
        cls,
        strategy: TrainingStrategy,
        config: ValidationConfig | None = None,
    ) -> ValidationLoop:
        """Build a :class:`ValidationLoop` from a :class:`TrainingStrategy`.

        Reads capabilities through the strategy's introspection methods
        and holds a live reference for counter/model access during
        :meth:`execute`.

        Parameters
        ----------
        strategy : TrainingStrategy
            The training strategy owning the validation pass.
        config : ValidationConfig | None
            Override validation config. ``None`` uses
            ``strategy.validation_config``.

        Returns
        -------
        ValidationLoop
            A loop instance ready to be used as a context manager.

        Raises
        ------
        RuntimeError
            When ``strategy.validation_config`` is ``None`` and no
            ``config`` override is provided.
        """
        resolved_config = config if config is not None else strategy.validation_config
        if resolved_config is None:
            raise RuntimeError(
                "ValidationLoop.from_training_strategy() requires a "
                "validation_config on the strategy or as an argument."
            )

        device = strategy.devices[0]

        # -- loss resolution (was _resolve_validation_loss_fn) --
        if resolved_config.loss_fn is not None:
            loss_fn = resolved_config.loss_fn
        else:
            loss_fn = as_composed_loss(strategy.loss_fn)

        validation_fn = resolved_config.validation_fn or strategy.training_fn

        # -- grad resolution (was _resolve_validation_grad) --
        grad_enabled = _resolve_grad_from_config(resolved_config, loss_fn)

        # -- model resolution (was _validation_model_arg) --
        model_arg, modules, ema_model_keys = _resolve_model_arg(
            strategy, resolved_config
        )

        precision_context, precision = strategy._inference_autocast(device)

        loop = cls.__new__(cls)
        loop._validation_data = resolved_config.validation_data
        loop._config = resolved_config
        loop._device = device
        loop._loss_fn = loss_fn
        loop._loss_target_assembler = strategy.loss_target_assembler
        loop._validation_fn = validation_fn
        loop._grad_enabled = grad_enabled
        loop._precision_context = precision_context
        loop._precision = precision
        loop._model_arg = model_arg
        loop._modules = _unique_modules(modules)
        loop._ema_model_keys = ema_model_keys
        loop._single_model_input = strategy.single_model_input
        loop._num_models = len(strategy.models)
        loop._strategy = strategy
        loop._standalone_context = None
        loop._successful = False
        loop._entered = False
        loop._modes = {}
        loop._grad_snapshot = {}
        return loop

    def _context(self) -> _LoopContext:
        """Return live counters and handles for the current execution.

        Returns
        -------
        _LoopContext
            Context snapshot. Strategy-integrated loops read live
            values from the held strategy reference; standalone loops
            return stored values.
        """
        if self._strategy is not None:
            return _LoopContext(
                step_count=self._strategy.step_count,
                epoch=self._strategy.epoch_count,
                distributed_manager=self._strategy.distributed_manager,
                num_models=len(self._strategy.models),
            )
        assert self._standalone_context is not None  # noqa: S101  # narrowing
        return self._standalone_context

    def __enter__(self) -> ValidationLoop:
        """Set up the validation pass.

        Snapshots training modes, sets eval mode (if configured), and
        snapshots and clears parameter gradients (if grad-enabled).

        Returns
        -------
        ValidationLoop
            The loop handle.
        """
        # Snapshot + set eval
        self._modes = _module_training_modes(self._modules)
        if self._config.set_eval:
            for module, _training in self._modes.values():
                module.eval()

        # Snapshot + clear grads
        if self._grad_enabled:
            self._grad_snapshot = _snapshot_parameter_grads(self._modules)
            _clear_parameter_grads(self._modules)

        self._entered = True
        self._successful = False
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> bool:
        """Tear down the validation pass.

        Restores parameter gradients (if grad-enabled) and restores
        module training modes (if ``set_eval``).

        Returns ``False`` so exceptions are not suppressed.

        Parameters
        ----------
        exc_type : type[BaseException] | None
            Exception type, if any.
        exc_val : BaseException | None
            Exception instance, if any.
        exc_tb : TracebackType | None
            Exception traceback, if any.

        Returns
        -------
        bool
            Always ``False``.
        """
        try:
            # Grad restore
            if self._grad_enabled:
                _clear_parameter_grads(self._modules)
                _restore_parameter_grads(self._grad_snapshot)

            # Training mode restore
            if self._config.set_eval:
                for module, training in self._modes.values():
                    module.train(training)
        finally:
            self._entered = False
        return False

    def execute(self) -> dict[str, Any]:
        """Run the validation loop over all batches and return the summary.

        Iterates ``validation_data``, runs the forward pass and loss
        computation per batch, invokes the optional per-batch callback,
        accumulates results, computes the distributed-reduced summary,
        and returns the summary dictionary.

        Returns
        -------
        dict[str, Any]
            The local validation summary outside distributed execution, or
            the distributed-reduced summary on every distributed rank.

        Raises
        ------
        RuntimeError
            When called outside the context manager.
        ValueError
            When ``validation_data`` produces no batches.
        """
        if not self._entered:
            raise RuntimeError(
                "ValidationLoop.execute() must be called inside a 'with' block."
            )

        ctx = self._context()
        device = self._device
        accumulator = _LossAccumulator(device)

        # Per-batch loop
        for batch_count, batch in enumerate(self._validation_data):
            validation_batch = batch.to(device, non_blocking=True)
            previous_hook_context = None
            if self._strategy is not None and self._strategy.hooks:
                previous_hook_context = self._strategy._ctx
                self._strategy._ctx = self._strategy._new_train_context(
                    validation_batch
                )
            try:
                if self._grad_enabled:
                    _clear_parameter_grads(self._modules)
                grad_ctx = (
                    torch.enable_grad() if self._grad_enabled else torch.no_grad()
                )
                with grad_ctx, self._precision_context():
                    if self._strategy is not None:
                        self._strategy._run_hooks(
                            TrainingStage.BEFORE_FORWARD, validation_batch
                        )
                    predictions = self._validation_fn(self._model_arg, validation_batch)
                    if self._strategy is not None:
                        self._strategy._run_hooks(
                            TrainingStage.AFTER_FORWARD, validation_batch
                        )
                    loss_out = compute_supervised_loss(
                        self._loss_fn,
                        predictions,
                        validation_batch,
                        step=ctx.step_count,
                        epoch=ctx.epoch,
                        workflow=self._strategy if self._strategy is not None else self,
                        target_assembler=self._loss_target_assembler,
                        batch_label="Validation batch",
                    )
            finally:
                if self._strategy is not None and self._strategy.hooks:
                    self._strategy._ctx = previous_hook_context
            accumulator.update(loss_out)
            # call the per-batch callback; this allows for user-defined operations
            # on the scope, e.g. log as much as you'd like
            if self._config.batch_callback is not None:
                self._config.batch_callback(
                    batch=validation_batch,
                    predictions=predictions,
                    loss=loss_out,
                    batch_count=batch_count,
                    step_count=ctx.step_count,
                    epoch=ctx.epoch,
                )

        # Build summary
        num_models = ctx.num_models
        model_source = (
            "ema"
            if (self._ema_model_keys and len(self._ema_model_keys) == num_models)
            else "mixed"
            if self._ema_model_keys
            else "live"
        )
        summary = accumulator.summary(
            name=self._config.name,
            model_source=model_source,
            ema_model_keys=self._ema_model_keys,
            precision=self._precision,
            distributed_manager=ctx.distributed_manager,
        )

        self._successful = True
        return summary
