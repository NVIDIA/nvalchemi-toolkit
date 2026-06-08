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
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import AbstractContextManager
from types import TracebackType
from typing import TYPE_CHECKING, Annotated, Any, Literal

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

from nvalchemi.data import AtomicData, Batch
from nvalchemi.training.distributed import (
    all_reduce as distributed_all_reduce,
)
from nvalchemi.training.distributed import (
    barrier as distributed_barrier,
)
from nvalchemi.training.distributed import (
    get_rank as get_distributed_rank,
)
from nvalchemi.training.distributed import (
    is_distributed_initialized,
)
from nvalchemi.training.losses.composition import (
    ComposedLossFunction,
    ComposedLossOutput,
    as_composed_loss,
    compute_supervised_loss,
)

if TYPE_CHECKING:
    from nvalchemi.training.strategy import TrainingStrategy

__all__ = ["ValidationConfig", "ValidationLoop"]

BatchTensorLevel = Literal["node", "edge", "system"]


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
    sink : Any | None
        Optional evaluation sink receiving packed validation batches.
        Accepts any object following the :class:`EvaluationSink` protocol.
    include_predictions : bool
        If ``True``, attach model predictions to sample output batches.
    write_samples : bool
        If ``True``, write augmented validation batches to ``sink``.
    write_batch_summaries : bool
        If ``True``, write one compact summary batch per validation batch.
    write_epoch_summary : bool
        If ``True``, write validation-epoch scalar means to capable sinks.
    write_batch_size : int | None
        Number of validation batches to coalesce into each sample sink
        write. ``None`` writes each batch individually.
    distributed_barrier : bool
        If ``True``, synchronize distributed ranks after sink writes.
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
    sink: Any | None = None
    include_predictions: bool = False
    write_samples: bool = True
    write_batch_summaries: bool = False
    write_epoch_summary: bool = True
    write_batch_size: int | None = Field(default=None, ge=1)
    distributed_barrier: bool = True
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


def _safe_batch_key(prefix: str, name: str) -> str:
    """Return a storage-safe evaluation field name."""
    safe_name = re.sub(r"[^0-9A-Za-z_]+", "_", name).strip("_")
    return f"{prefix}_{safe_name}" if safe_name else prefix


def _expanded_scalar(
    value: torch.Tensor,
    *,
    length: int,
    device: torch.device,
) -> torch.Tensor:
    """Return ``value`` as a detached system-level tensor of length ``length``."""
    scalar = value.detach().to(device=device).reshape(-1).sum()
    return scalar.reshape(1).expand(length).clone()


def _set_batch_tensor(
    batch: Batch,
    key: str,
    value: torch.Tensor,
    *,
    level: BatchTensorLevel,
) -> None:
    """Attach ``value`` to ``batch`` without revalidating storage shapes."""
    group_name = {"node": "atoms", "edge": "edges", "system": "system"}[level]
    batch._storage.attr_map.set(
        key,
        group_name,
        is_segmented=level != "system",
    )
    value = value.detach().to(device=batch.device)
    if group_name not in batch._storage.groups:
        if level != "system":
            raise ValueError(
                f"Cannot add {level}-level evaluation tensor {key!r} to a batch "
                f"without a {group_name!r} storage group."
            )
        batch[key] = value
    else:
        batch._storage.groups[group_name]._data[key] = value
    if batch.keys is not None:
        batch.keys[level].add(key)


def _prediction_tensor_level(
    key: str,
    value: torch.Tensor,
    batch: Batch,
) -> tuple[BatchTensorLevel, torch.Tensor] | None:
    """Infer the storage level for a prediction tensor."""
    detached = value.detach()
    if detached.ndim == 0:
        return "system", detached.reshape(1).expand(batch.num_graphs).clone()
    leading = detached.shape[0]
    lowered = key.lower()
    if leading == batch.num_edges and any(
        fragment in lowered for fragment in ("edge", "neighbor", "shift")
    ):
        return "edge", detached
    if leading == batch.num_nodes and any(
        fragment in lowered
        for fragment in ("force", "position", "atomic", "charge", "mass", "node")
    ):
        return "node", detached
    if leading == batch.num_graphs:
        return "system", detached
    if leading == batch.num_nodes:
        return "node", detached
    if batch.num_edges > 0 and leading == batch.num_edges:
        return "edge", detached
    return None


def _minimal_summary_batch(
    fields: Mapping[str, torch.Tensor],
    *,
    device: torch.device,
) -> Batch:
    """Pack scalar summary fields into a one-graph :class:`Batch`."""
    data = AtomicData(
        positions=torch.zeros(1, 3, device=device),
        atomic_numbers=torch.ones(1, dtype=torch.long, device=device),
    )
    batch = Batch.from_data_list([data], device=device, skip_validation=True)
    for key, value in fields.items():
        _set_batch_tensor(batch, key, value.detach().reshape(1), level="system")
    return batch


def _combine_batches(batches: Sequence[Batch]) -> Batch:
    """Return one batch containing all graphs from ``batches``."""
    if not batches:
        raise ValueError("Cannot combine an empty batch sequence.")
    combined = batches[0].clone()
    for batch in batches[1:]:
        combined.append(batch)
    return combined


class _LossAccumulator:
    """Accumulate composed-loss diagnostics over validation batches."""

    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.batch_count = 0
        self.total_sum: torch.Tensor | None = None
        self.per_component_total_sum: dict[str, torch.Tensor] = {}
        self.per_component_sample_sum: dict[str, torch.Tensor] = {}
        self.per_component_sample_count: dict[str, int] = {}
        self.per_component_weight: dict[str, float] = {}
        self.per_component_raw_weight: dict[str, float] = {}

    def update(self, loss_out: ComposedLossOutput) -> None:
        """Add one batch's loss output to the running totals."""
        self.batch_count += 1
        total = loss_out["total_loss"].detach()
        self.total_sum = total if self.total_sum is None else self.total_sum + total
        for name, value in loss_out["per_component_total"].items():
            detached = value.detach()
            previous = self.per_component_total_sum.get(name)
            self.per_component_total_sum[name] = (
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

    def scalar_means(
        self,
        *,
        distributed: bool,
        distributed_manager: Any | None = None,
    ) -> dict[str, torch.Tensor]:
        """Return scalar loss means for sink summary output."""
        if self.batch_count == 0 or self.total_sum is None:
            raise ValueError("validation_data produced no batches.")

        entries: dict[str, tuple[torch.Tensor, int]] = {}
        entries["total_loss"] = (self.total_sum, self.batch_count)
        for name in sorted(self.per_component_total_sum):
            entries[name] = (self.per_component_total_sum[name], self.batch_count)

        values: list[torch.Tensor] = []
        for loss_sum, count in entries.values():
            values.append(_as_float64_scalar(loss_sum, self.device))
            values.append(
                torch.tensor(float(count), device=self.device, dtype=torch.float64)
            )
        packed = torch.stack(values)
        if distributed:
            _distributed_sum_in_place(packed, distributed_manager)

        means: dict[str, torch.Tensor] = {}
        index = 0
        for name in entries:
            loss_sum = packed[index]
            count = packed[index + 1]
            means[name] = _tensor_to_cpu(loss_sum / count)
            index += 2
        return means

    def summary(
        self,
        *,
        name: str,
        model_source: str,
        ema_model_keys: tuple[str, ...],
        precision: str,
        publish: bool,
        distributed_manager: Any | None = None,
    ) -> dict[str, Any] | None:
        """Return the local or distributed-reduced validation summary."""
        if self.batch_count == 0 or self.total_sum is None:
            raise ValueError("validation_data produced no batches.")

        component_keys = tuple(sorted(self.per_component_total_sum))
        sample_keys = tuple(sorted(self.per_component_sample_sum))
        values = [
            _as_float64_scalar(self.total_sum, self.device),
            torch.tensor(
                float(self.batch_count), device=self.device, dtype=torch.float64
            ),
        ]
        values.extend(
            _as_float64_scalar(self.per_component_total_sum[key], self.device)
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
        if not publish:
            return None

        index = 0
        total_sum = packed[index]
        index += 1
        batch_count = packed[index]
        index += 1
        reduced_batch_count = int(batch_count.item())

        per_component_total: dict[str, torch.Tensor] = {}
        for key in component_keys:
            per_component_total[key] = _tensor_to_cpu(packed[index] / batch_count)
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
            "per_component_total": per_component_total,
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


def _distributed_barrier_fn(distributed_manager: Any | None) -> None:
    """Synchronize ranks when distributed communication is active."""
    if is_distributed_initialized(distributed_manager):
        distributed_barrier(distributed_manager)


# ------------------------------------------------------------------
# Shared sink helpers
# ------------------------------------------------------------------


def _begin_sink(
    sink: Any | None,
    *,
    step_count: int,
    epoch: int,
    name: str,
    distributed_manager: Any | None,
) -> None:
    """Notify a sink that one validation run is starting.

    Parameters
    ----------
    sink : Any | None
        Evaluation sink, or ``None`` to skip.
    step_count : int
        Current optimizer step count.
    epoch : int
        Current epoch count.
    name : str
        Validation name string.
    distributed_manager : Any | None
        Optional distributed manager for the sink.
    """
    if sink is None:
        return
    _configure_sink_distributed_manager(sink, distributed_manager)
    method = getattr(sink, "begin_evaluation", None)
    if method is not None:
        method(step_count=step_count, epoch=epoch, name=name)


def _configure_sink_distributed_manager(
    sink: Any | None, distributed_manager: Any | None
) -> None:
    """Pass the distributed manager to sinks that accept one.

    Parameters
    ----------
    sink : Any | None
        Evaluation sink, or ``None`` to skip.
    distributed_manager : Any | None
        Optional distributed manager to pass.
    """
    if sink is None:
        return
    method = getattr(sink, "set_distributed_manager", None)
    if callable(method):
        method(distributed_manager)


def _end_sink(
    sink: Any | None,
    *,
    step_count: int,
    epoch: int,
    name: str,
) -> None:
    """Notify a sink that one validation run has finished.

    Parameters
    ----------
    sink : Any | None
        Evaluation sink, or ``None`` to skip.
    step_count : int
        Current optimizer step count.
    epoch : int
        Current epoch count.
    name : str
        Validation name string.
    """
    if sink is None:
        return
    method = getattr(sink, "end_evaluation", None)
    if method is not None:
        method(step_count=step_count, epoch=epoch, name=name)


def _sample_output_batch(
    batch: Batch,
    predictions: Mapping[str, torch.Tensor],
    loss_out: ComposedLossOutput,
    *,
    batch_count: int,
    step_count: int,
    epoch: int,
    include_predictions: bool,
) -> Batch:
    """Pack per-sample loss diagnostics into a new validation batch.

    Parameters
    ----------
    batch : Batch
        The validation batch to augment (cloned internally).
    predictions : Mapping[str, torch.Tensor]
        Model prediction tensors.
    loss_out : ComposedLossOutput
        Loss output from :func:`compute_supervised_loss`.
    batch_count : int
        Zero-based validation batch index.
    step_count : int
        Current optimizer step count.
    epoch : int
        Current epoch count.
    include_predictions : bool
        Whether to attach prediction tensors to the output batch.

    Returns
    -------
    Batch
        A cloned batch augmented with evaluation metadata.
    """
    output = batch.clone()
    num_graphs = output.num_graphs
    device = output.device
    _set_batch_tensor(
        output,
        "eval_step",
        torch.full((num_graphs,), step_count, dtype=torch.long, device=device),
        level="system",
    )
    _set_batch_tensor(
        output,
        "eval_epoch",
        torch.full((num_graphs,), epoch, dtype=torch.long, device=device),
        level="system",
    )
    _set_batch_tensor(
        output,
        "eval_batch_index",
        torch.full((num_graphs,), batch_count, dtype=torch.long, device=device),
        level="system",
    )
    _set_batch_tensor(
        output,
        "eval_total_loss",
        _expanded_scalar(loss_out["total_loss"], length=num_graphs, device=device),
        level="system",
    )

    total_sample: torch.Tensor | None = None
    for name, sample in loss_out["per_component_sample"].items():
        sample = sample.detach().to(device=device).reshape(num_graphs)
        _set_batch_tensor(
            output,
            _safe_batch_key("eval_loss", name),
            sample,
            level="system",
        )
        total_sample = sample if total_sample is None else total_sample + sample
    if total_sample is not None:
        _set_batch_tensor(
            output,
            "eval_sample_loss",
            total_sample,
            level="system",
        )

    for name, value in loss_out["per_component_total"].items():
        _set_batch_tensor(
            output,
            _safe_batch_key("eval_component_total", name),
            _expanded_scalar(value, length=num_graphs, device=device),
            level="system",
        )
    for name, value in loss_out["per_component_weight"].items():
        _set_batch_tensor(
            output,
            _safe_batch_key("eval_component_weight", name),
            torch.full((num_graphs,), value, dtype=torch.float64, device=device),
            level="system",
        )
    for name, value in loss_out["per_component_raw_weight"].items():
        _set_batch_tensor(
            output,
            _safe_batch_key("eval_component_raw_weight", name),
            torch.full((num_graphs,), value, dtype=torch.float64, device=device),
            level="system",
        )

    if include_predictions:
        for key, value in predictions.items():
            if not isinstance(value, torch.Tensor):
                continue
            inferred = _prediction_tensor_level(key, value, output)
            if inferred is None:
                continue
            level, tensor = inferred
            _set_batch_tensor(
                output,
                _safe_batch_key("eval_prediction", key),
                tensor,
                level=level,
            )
    return output


def _batch_summary_output_batch(
    loss_out: ComposedLossOutput,
    batch: Batch,
    *,
    batch_count: int,
    step_count: int,
    epoch: int,
) -> Batch:
    """Pack one validation batch's summary into a compact batch.

    Parameters
    ----------
    loss_out : ComposedLossOutput
        Loss output from :func:`compute_supervised_loss`.
    batch : Batch
        The validation batch (used for ``num_graphs`` and ``device``).
    batch_count : int
        Zero-based validation batch index.
    step_count : int
        Current optimizer step count.
    epoch : int
        Current epoch count.

    Returns
    -------
    Batch
        A minimal one-graph summary batch.
    """
    device = batch.device
    fields: dict[str, torch.Tensor] = {
        "eval_step": torch.tensor(step_count, device=device),
        "eval_epoch": torch.tensor(epoch, device=device),
        "eval_batch_index": torch.tensor(batch_count, device=device),
        "eval_num_samples": torch.tensor(batch.num_graphs, device=device),
        "eval_total_loss": loss_out["total_loss"].detach(),
    }
    for name, value in loss_out["per_component_total"].items():
        fields[_safe_batch_key("eval_component_total", name)] = value.detach()
    for name, sample in loss_out["per_component_sample"].items():
        fields[_safe_batch_key("eval_loss_mean", name)] = sample.detach().mean()
    return _minimal_summary_batch(fields, device=device)


def _epoch_summary_output_batch(
    local_summary: Mapping[str, torch.Tensor],
    global_summary: Mapping[str, torch.Tensor],
    *,
    step_count: int,
    epoch: int,
    device: torch.device,
) -> Batch:
    """Pack validation-epoch scalar means into a compact batch.

    Parameters
    ----------
    local_summary : Mapping[str, torch.Tensor]
        Per-rank scalar loss means.
    global_summary : Mapping[str, torch.Tensor]
        Globally reduced scalar loss means.
    step_count : int
        Current optimizer step count.
    epoch : int
        Current epoch count.
    device : torch.device
        Device for the output batch.

    Returns
    -------
    Batch
        A minimal one-graph epoch summary batch.
    """
    fields: dict[str, torch.Tensor] = {
        "eval_step": torch.tensor(step_count, device=device),
        "eval_epoch": torch.tensor(epoch, device=device),
    }
    for name, value in local_summary.items():
        fields[_safe_batch_key("eval_rank_mean", name)] = value
    for name, value in global_summary.items():
        fields[_safe_batch_key("eval_global_mean", name)] = value
    return _minimal_summary_batch(fields, device=device)


def _write_or_buffer_sample_batch(
    sink: Any | None,
    batch: Batch,
    *,
    batch_count: int,
    step_count: int,
    epoch: int,
    write_batch_size: int | None,
    buffer: list[Batch],
    buffer_start: int | None,
) -> int | None:
    """Write or buffer one sample output batch.

    Parameters
    ----------
    sink : Any | None
        Evaluation sink, or ``None`` to skip.
    batch : Batch
        Augmented sample batch to write or buffer.
    batch_count : int
        Zero-based validation batch index.
    step_count : int
        Current optimizer step count.
    epoch : int
        Current epoch count.
    write_batch_size : int | None
        Coalescing size, or ``None`` for immediate writes.
    buffer : list[Batch]
        Mutable buffer for coalesced writes.
    buffer_start : int | None
        Start index of the current buffer window.

    Returns
    -------
    int | None
        Updated ``buffer_start`` value.
    """
    if sink is None:
        return None
    if write_batch_size is None:
        _write_sink_samples(
            sink, batch, batch_count=batch_count, step_count=step_count, epoch=epoch
        )
        return None
    if buffer_start is None:
        buffer_start = batch_count
    buffer.append(batch)
    if len(buffer) >= write_batch_size:
        _flush_sample_buffer(
            sink,
            buffer,
            buffer_start=buffer_start,
            step_count=step_count,
            epoch=epoch,
        )
        return None
    return buffer_start


def _flush_sample_buffer(
    sink: Any | None,
    buffer: list[Batch],
    *,
    buffer_start: int | None,
    step_count: int,
    epoch: int,
) -> None:
    """Write and clear buffered sample output batches.

    Parameters
    ----------
    sink : Any | None
        Evaluation sink, or ``None`` to skip.
    buffer : list[Batch]
        Mutable buffer to flush.
    buffer_start : int | None
        Start index of the current buffer window.
    step_count : int
        Current optimizer step count.
    epoch : int
        Current epoch count.
    """
    if sink is None or not buffer:
        return
    if buffer_start is None:
        raise RuntimeError("Sample buffer is missing its start index.")
    _write_sink_samples(
        sink,
        _combine_batches(buffer),
        batch_count=buffer_start,
        step_count=step_count,
        epoch=epoch,
    )
    buffer.clear()


def _write_sink_samples(
    sink: Any | None,
    batch: Batch,
    *,
    batch_count: int,
    step_count: int,
    epoch: int,
) -> None:
    """Write one augmented sample batch to the configured sink.

    Parameters
    ----------
    sink : Any | None
        Evaluation sink, or ``None`` to skip.
    batch : Batch
        Augmented sample batch.
    batch_count : int
        Zero-based batch index.
    step_count : int
        Current optimizer step count.
    epoch : int
        Current epoch count.
    """
    if sink is None:
        return
    method = getattr(sink, "write_samples", None)
    if method is not None:
        method(
            batch,
            step_count=step_count,
            epoch=epoch,
            batch_count=batch_count,
        )
        return
    write = getattr(sink, "write", None)
    if write is not None:
        write(batch)


def _write_sink_batch_summary(
    sink: Any | None,
    batch: Batch,
    *,
    batch_count: int,
    step_count: int,
    epoch: int,
) -> None:
    """Write a per-validation-batch summary if the sink supports it.

    Parameters
    ----------
    sink : Any | None
        Evaluation sink, or ``None`` to skip.
    batch : Batch
        One-graph summary batch.
    batch_count : int
        Zero-based batch index.
    step_count : int
        Current optimizer step count.
    epoch : int
        Current epoch count.
    """
    if sink is None:
        return
    method = getattr(sink, "write_batch_summary", None)
    if method is not None:
        method(
            batch,
            step_count=step_count,
            epoch=epoch,
            batch_count=batch_count,
        )


def _write_sink_epoch_summary(
    sink: Any | None,
    batch: Batch,
    *,
    local_summary: Mapping[str, torch.Tensor],
    global_summary: Mapping[str, torch.Tensor],
    step_count: int,
    epoch: int,
) -> None:
    """Write a validation-epoch summary if the sink supports it.

    Parameters
    ----------
    sink : Any | None
        Evaluation sink, or ``None`` to skip.
    batch : Batch
        One-graph epoch summary batch.
    local_summary : Mapping[str, torch.Tensor]
        Per-rank scalar loss means.
    global_summary : Mapping[str, torch.Tensor]
        Globally reduced scalar loss means.
    step_count : int
        Current optimizer step count.
    epoch : int
        Current epoch count.
    """
    if sink is None:
        return
    method = getattr(sink, "write_epoch_summary", None)
    if method is not None:
        method(
            batch,
            step_count=step_count,
            epoch=epoch,
            local_summary=local_summary,
            global_summary=global_summary,
        )


# ------------------------------------------------------------------
# Orchestration helpers for TrainingStrategy.validate()
# ------------------------------------------------------------------


@dataclasses.dataclass
class _ValidationRun:
    """Resolved per-run state threaded between strategy validation helpers.

    Attributes
    ----------
    loss_fn : ComposedLossFunction
        Resolved validation loss function.
    validation_fn : Callable[..., Any]
        Forward callable for validation batches.
    grad_enabled : bool
        Whether autograd is enabled during the validation pass.
    model_arg : Any
        Model or model dict passed to ``validation_fn``.
    modules : tuple[nn.Module, ...]
        Unique modules participating in the validation forward pass.
    ema_model_keys : tuple[str, ...]
        Model keys sourced from the EMA inference slot.
    precision : str
        Precision label for the validation pass.
    precision_context : Callable[[], AbstractContextManager[None]]
        Zero-arg factory returning the autocast context manager.
    accumulator : _LossAccumulator
        Running loss accumulator for the validation pass.
    modes : dict[int, tuple[nn.Module, bool]]
        Snapshot of module training modes for restoration.
    grad_snapshot : dict[int, tuple[nn.Parameter, torch.Tensor | None]]
        Snapshot of parameter gradients for restoration.
    """

    loss_fn: ComposedLossFunction
    validation_fn: Callable[..., Any]
    grad_enabled: bool
    model_arg: Any
    modules: tuple[nn.Module, ...]
    ema_model_keys: tuple[str, ...]
    precision: str
    precision_context: Callable[[], AbstractContextManager[None]]
    accumulator: _LossAccumulator
    modes: dict[int, tuple[nn.Module, bool]]
    grad_snapshot: dict[int, tuple[nn.Parameter, torch.Tensor | None]] = (
        dataclasses.field(default_factory=dict)
    )


class _SinkWriter:
    """Encapsulate evaluation-sink lifecycle and per-batch write logic.

    Wraps the module-level sink helpers (``_begin_sink``,
    ``_sample_output_batch``, ``_write_or_buffer_sample_batch``, etc.)
    so callers do not need to thread buffer state or repeat guard
    conditionals.

    Parameters
    ----------
    sink : Any | None
        Evaluation sink, or ``None`` for a full no-op writer.
    step_count : int
        Current optimizer step count.
    epoch : int
        Current epoch count.
    name : str
        Validation name string.
    write_batch_size : int | None
        Coalescing size for sample writes, or ``None`` for immediate.
    write_samples : bool
        Whether per-sample output batches should be written.
    write_batch_summaries : bool
        Whether per-batch summary writes are enabled.
    write_epoch_summary : bool
        Whether epoch-level summary writes are enabled.
    include_predictions : bool
        Whether to attach model predictions to sample output batches.
    distributed_barrier : bool
        Whether to synchronize ranks after sink writes.
    distributed_manager : Any | None
        Distributed manager for barrier and sink configuration.
    """

    def __init__(
        self,
        sink: Any | None,
        *,
        step_count: int,
        epoch: int,
        name: str,
        write_batch_size: int | None,
        write_samples: bool,
        write_batch_summaries: bool,
        write_epoch_summary: bool,
        include_predictions: bool,
        distributed_barrier: bool,
        distributed_manager: Any | None,
    ) -> None:
        self._sink = sink
        self._step_count = step_count
        self._epoch = epoch
        self._name = name
        self._write_batch_size = write_batch_size
        self._write_samples = write_samples
        self._write_batch_summaries = write_batch_summaries
        self._write_epoch_summary = write_epoch_summary
        self._include_predictions = include_predictions
        self._distributed_barrier = distributed_barrier
        self._distributed_manager = distributed_manager
        self._started = False
        self._sample_buffer: list[Batch] = []
        self._sample_buffer_start: int | None = None

    def begin(self) -> None:
        """Notify the sink that a validation run is starting."""
        _begin_sink(
            self._sink,
            step_count=self._step_count,
            epoch=self._epoch,
            name=self._name,
            distributed_manager=self._distributed_manager,
        )
        self._started = self._sink is not None

    def record_batch(
        self,
        validation_batch: Batch,
        predictions: Mapping[str, torch.Tensor],
        loss_out: ComposedLossOutput,
        batch_count: int,
    ) -> None:
        """Write per-batch sample and summary data to the sink.

        Parameters
        ----------
        validation_batch : Batch
            The validation batch on the target device.
        predictions : Mapping[str, torch.Tensor]
            Model prediction tensors.
        loss_out : ComposedLossOutput
            Loss output from ``compute_supervised_loss``.
        batch_count : int
            Zero-based validation batch index.
        """
        if self._sink is not None and self._write_samples:
            output_batch: Batch | None = _sample_output_batch(
                validation_batch,
                predictions,
                loss_out,
                batch_count=batch_count,
                step_count=self._step_count,
                epoch=self._epoch,
                include_predictions=self._include_predictions,
            )
        else:
            output_batch = None
        if output_batch is not None and self._write_samples:
            self._sample_buffer_start = _write_or_buffer_sample_batch(
                self._sink,
                output_batch,
                batch_count=batch_count,
                step_count=self._step_count,
                epoch=self._epoch,
                write_batch_size=self._write_batch_size,
                buffer=self._sample_buffer,
                buffer_start=self._sample_buffer_start,
            )
        if self._sink is not None and self._write_batch_summaries:
            _write_sink_batch_summary(
                self._sink,
                _batch_summary_output_batch(
                    loss_out,
                    validation_batch,
                    batch_count=batch_count,
                    step_count=self._step_count,
                    epoch=self._epoch,
                ),
                batch_count=batch_count,
                step_count=self._step_count,
                epoch=self._epoch,
            )

    def flush(self) -> None:
        """Flush any remaining buffered sample output batches."""
        _flush_sample_buffer(
            self._sink,
            self._sample_buffer,
            buffer_start=self._sample_buffer_start,
            step_count=self._step_count,
            epoch=self._epoch,
        )

    def write_epoch_summary(
        self,
        accumulator: _LossAccumulator,
        device: torch.device,
    ) -> None:
        """Write epoch-level scalar summary to the sink.

        Parameters
        ----------
        accumulator : _LossAccumulator
            Accumulator holding loss totals for the validation pass.
        device : torch.device
            Device for summary tensor construction.
        """
        if self._sink is None or not self._write_epoch_summary:
            return
        local_scalar_summary = accumulator.scalar_means(
            distributed=False,
            distributed_manager=self._distributed_manager,
        )
        global_scalar_summary = accumulator.scalar_means(
            distributed=True,
            distributed_manager=self._distributed_manager,
        )
        _write_sink_epoch_summary(
            self._sink,
            _epoch_summary_output_batch(
                local_scalar_summary,
                global_scalar_summary,
                step_count=self._step_count,
                epoch=self._epoch,
                device=device,
            ),
            local_summary=local_scalar_summary,
            global_summary=global_scalar_summary,
            step_count=self._step_count,
            epoch=self._epoch,
        )

    def end(self) -> None:
        """Notify the sink that the validation run has finished."""
        if self._started:
            _end_sink(
                self._sink,
                step_count=self._step_count,
                epoch=self._epoch,
                name=self._name,
            )

    def barrier_if_needed(self, successful: bool) -> None:
        """Synchronize distributed ranks when appropriate.

        Parameters
        ----------
        successful : bool
            Whether the validation pass completed without error.
        """
        if successful and self._sink is not None and self._distributed_barrier:
            _distributed_barrier_fn(self._distributed_manager)


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
        self._writer: _SinkWriter | None = None

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
        loop._writer = None
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

        Snapshots training modes, sets eval mode (if configured),
        snapshots and clears parameter gradients (if grad-enabled),
        and begins the evaluation sink.

        Returns
        -------
        ValidationLoop
            The loop handle.
        """
        ctx = self._context()

        # Snapshot + set eval
        self._modes = _module_training_modes(self._modules)
        if self._config.set_eval:
            for module, _training in self._modes.values():
                module.eval()

        # Snapshot + clear grads
        if self._grad_enabled:
            self._grad_snapshot = _snapshot_parameter_grads(self._modules)
            _clear_parameter_grads(self._modules)

        # Begin sink
        self._writer = _SinkWriter(
            self._config.sink,
            step_count=ctx.step_count,
            epoch=ctx.epoch,
            name=self._config.name,
            write_batch_size=self._config.write_batch_size,
            write_samples=self._config.write_samples,
            write_batch_summaries=self._config.write_batch_summaries,
            write_epoch_summary=self._config.write_epoch_summary,
            include_predictions=self._config.include_predictions,
            distributed_barrier=self._config.distributed_barrier,
            distributed_manager=ctx.distributed_manager,
        )
        self._writer.begin()
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

        Restores parameter gradients (if grad-enabled), restores
        module training modes (if ``set_eval``), ends the sink,
        and runs the distributed barrier on success.

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

            # End sink
            if self._writer is not None:
                self._writer.end()

            # Barrier
            if self._writer is not None:
                self._writer.barrier_if_needed(self._successful)
        finally:
            self._entered = False
        return False

    def execute(self) -> dict[str, Any] | None:
        """Run the validation loop over all batches and return the summary.

        Iterates ``validation_data``, runs the forward pass and loss
        computation per batch, accumulates results, flushes the sink
        buffer, computes the distributed-reduced summary, writes the
        epoch summary to the sink, and returns the summary dictionary.

        Returns
        -------
        dict[str, Any] | None
            The validation summary on rank 0, ``None`` on
            non-publishing distributed ranks.

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
        assert self._writer is not None  # noqa: S101  # narrowing

        ctx = self._context()
        device = self._device
        accumulator = _LossAccumulator(device)

        # Per-batch loop
        for batch_count, batch in enumerate(self._validation_data):
            validation_batch = batch.to(device, non_blocking=True)
            if self._grad_enabled:
                _clear_parameter_grads(self._modules)
            grad_ctx = torch.enable_grad() if self._grad_enabled else torch.no_grad()
            with grad_ctx, self._precision_context():
                predictions = self._validation_fn(self._model_arg, validation_batch)
                loss_out = compute_supervised_loss(
                    self._loss_fn,
                    predictions,
                    validation_batch,
                    step=ctx.step_count,
                    epoch=ctx.epoch,
                    batch_label="Validation batch",
                )
            accumulator.update(loss_out)
            self._writer.record_batch(
                validation_batch, predictions, loss_out, batch_count
            )

        # Flush sample buffer
        self._writer.flush()

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
            publish=get_distributed_rank(ctx.distributed_manager) == 0,
            distributed_manager=ctx.distributed_manager,
        )

        # Epoch summary sink write
        self._writer.write_epoch_summary(accumulator, device)

        self._successful = True
        return summary
