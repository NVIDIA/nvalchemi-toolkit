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
"""Validation/evaluation hook for :class:`nvalchemi.training.TrainingStrategy`."""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import AbstractContextManager, nullcontext
from typing import Any, Literal

import torch
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    field_validator,
    model_validator,
)
from torch import nn

from nvalchemi.data import AtomicData, Batch
from nvalchemi.hooks._context import TrainContext
from nvalchemi.training._stages import TrainingStage
from nvalchemi.training.distributed import (
    all_reduce as distributed_all_reduce,
)
from nvalchemi.training.distributed import (
    barrier as distributed_barrier,
)
from nvalchemi.training.distributed import (
    get_rank as distributed_get_rank,
)
from nvalchemi.training.distributed import (
    is_distributed_initialized,
)
from nvalchemi.training.hooks.ema import EMAHook
from nvalchemi.training.hooks.evaluation_sinks import EvaluationSink
from nvalchemi.training.hooks.mixed_precision import MixedPrecisionHook
from nvalchemi.training.hooks.update import TrainingUpdateOrchestrator
from nvalchemi.training.losses.composition import (
    BaseLossFunction,
    ComposedLossFunction,
    ComposedLossOutput,
    as_composed_loss,
    compute_supervised_loss,
)

__all__ = ["EvaluateHook"]


GradMode = Literal["auto", "enabled", "disabled"]
HookPolicy = Literal["auto", "always", "never"]
BatchTensorLevel = Literal["node", "edge", "system"]


def _iter_registered_hooks(hooks: Iterable[Any]) -> Iterator[Any]:
    """Yield registered hooks and children nested in update orchestrators."""
    for hook in hooks:
        yield hook
        if isinstance(hook, TrainingUpdateOrchestrator):
            yield from _iter_registered_hooks(hook.iter_hooks())


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
            raise ValueError("EvaluateHook validation_data produced no batches.")

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
            raise ValueError("EvaluateHook validation_data produced no batches.")

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


def _distributed_manager(ctx: TrainContext) -> Any | None:
    """Return the workflow distributed manager when one is configured."""
    workflow = ctx.workflow
    return None if workflow is None else getattr(workflow, "distributed_manager", None)


def _distributed_sum_in_place(
    value: torch.Tensor, distributed_manager: Any | None
) -> bool:
    """All-reduce ``value`` when distributed communication is active."""
    if not is_distributed_initialized(distributed_manager):
        return False
    distributed_all_reduce(value, distributed_manager)
    return True


def _distributed_barrier(distributed_manager: Any | None) -> None:
    """Synchronize ranks when distributed communication is active."""
    if is_distributed_initialized(distributed_manager):
        distributed_barrier(distributed_manager)


class EvaluateHook(BaseModel):
    """Run validation from inside :class:`~nvalchemi.training.TrainingStrategy`.

    Parameters
    ----------
    validation_data : Any
        Re-iterable validation batches. The hook iterates this object
        directly and never constructs a DataLoader.
    validation_fn : Callable | None, optional
        Validation forward callable. Defaults to the strategy's
        ``training_fn`` and uses the same single-model or named-model call
        convention.
    loss_fn : BaseLossFunction | ComposedLossFunction | None, optional
        Validation loss. Defaults to the strategy's ``loss_fn``.
    stage : TrainingStage, optional
        Stage where validation should run. Default ``AFTER_EPOCH``.
    frequency : int, optional
        Standard hook frequency for explicit ``stage`` scheduling.
    every_n_epochs : int | None, optional
        Convenience schedule for ``AFTER_EPOCH`` based on completed epochs.
    every_n_steps : int | None, optional
        Convenience schedule for ``AFTER_OPTIMIZER_STEP`` based on completed
        optimizer steps.
    grad_mode : {"auto", "enabled", "disabled"}, optional
        Validation gradient policy. ``"auto"`` enables gradients for force
        or stress losses and disables them for scalar-only losses.
    set_eval : bool, optional
        If ``True``, run selected validation modules in eval mode and restore
        their original modes afterward.
    use_ema : {"auto", "always", "never"}, optional
        Whether initialized :class:`EMAHook` averaged weights should replace
        live model weights for validation.
    use_mixed_precision : {"auto", "always", "never"}, optional
        Whether to reuse the registered :class:`MixedPrecisionHook` autocast
        precision for validation inference.
    run_at_end : bool, optional
        For the default epoch schedule, run one final validation at
        ``AFTER_TRAINING`` when no epoch-level validation fired. This covers
        ``num_steps`` training that stops before an epoch boundary.
    sink : EvaluationSink | Any | None, optional
        Optional sink receiving packed evaluation batches. Sinks may implement
        granular evaluation methods; objects with only ``write(batch)`` receive
        augmented sample batches.
    include_predictions : bool, optional
        If ``True``, attach model predictions to sample output batches.
    write_samples : bool, optional
        If ``True``, write augmented validation batches to ``sink``.
    write_batch_summaries : bool, optional
        If ``True``, write one compact summary batch per validation batch.
    write_epoch_summary : bool, optional
        If ``True``, write validation-epoch scalar means to capable sinks.
    write_batch_size : int | None, optional
        Number of validation batches to coalesce into each sample sink write.
    distributed_barrier : bool, optional
        If ``True``, synchronize distributed ranks after sink writes finish.
    name : str, optional
        Name stored in the validation summary.
    """

    validation_data: Any
    validation_fn: Callable[..., Mapping[str, torch.Tensor]] | None = None
    loss_fn: BaseLossFunction | ComposedLossFunction | None = None
    stage: TrainingStage = TrainingStage.AFTER_EPOCH
    frequency: int = Field(default=1, ge=1)
    every_n_epochs: int | None = Field(default=None, ge=1)
    every_n_steps: int | None = Field(default=None, ge=1)
    grad_mode: GradMode = "auto"
    set_eval: bool = True
    use_ema: HookPolicy = "auto"
    use_mixed_precision: HookPolicy = "auto"
    run_at_end: bool = True
    sink: EvaluationSink | Any | None = None
    include_predictions: bool = False
    write_samples: bool = True
    write_batch_summaries: bool = False
    write_epoch_summary: bool = True
    write_batch_size: int | None = Field(default=None, ge=1)
    distributed_barrier: bool = True
    name: str = Field(default="validation", min_length=1)

    _has_run: bool = PrivateAttr(default=False)

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
        validate_assignment=False,
    )

    def _runs_on_stage(self, stage: TrainingStage) -> bool:
        """Return whether the hook should receive ``stage`` dispatches."""
        return stage is self.stage or self._is_end_fallback_stage(stage)

    def _requires_update_orchestrator_before_stage(self, stage: TrainingStage) -> bool:
        """Return whether this hook needs update hooks to run before ``stage``."""
        return stage is TrainingStage.AFTER_OPTIMIZER_STEP and self._runs_on_stage(
            stage
        )

    def _validate_registered_hooks(self, hooks: Iterable[Any]) -> None:
        """Validate dependencies that require seeing the full registered hook set."""
        registered_hooks = tuple(_iter_registered_hooks(hooks))
        if self.use_mixed_precision == "always" and not any(
            isinstance(hook, MixedPrecisionHook) for hook in registered_hooks
        ):
            raise ValueError(
                "EvaluateHook use_mixed_precision='always' requires a registered "
                "MixedPrecisionHook."
            )
        if self.use_ema == "always" and not any(
            isinstance(hook, EMAHook) for hook in registered_hooks
        ):
            raise ValueError(
                "EvaluateHook use_ema='always' requires a registered EMAHook."
            )

    def __enter__(self) -> EvaluateHook:
        """Reset per-run bookkeeping when the owning strategy starts."""
        self._has_run = False
        if self.sink is not None and hasattr(self.sink, "__enter__"):
            self.sink.__enter__()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: Any,
    ) -> None:
        """Close the evaluation sink when the owning strategy exits."""
        if self.sink is None:
            return
        if hasattr(self.sink, "__exit__"):
            self.sink.__exit__(exc_type, exc, tb)
        elif hasattr(self.sink, "close"):
            self.sink.close()

    @field_validator("loss_fn", mode="after")
    @classmethod
    def _normalize_loss_fn(
        cls, value: BaseLossFunction | ComposedLossFunction | None
    ) -> ComposedLossFunction | None:
        """Normalize validation leaf losses to a composed loss."""
        return None if value is None else as_composed_loss(value)

    @model_validator(mode="after")
    def _validate_schedule(self) -> EvaluateHook:
        """Validate convenience scheduling knobs."""
        if self.every_n_epochs is not None and self.every_n_steps is not None:
            raise ValueError("Only one of every_n_epochs or every_n_steps may be set.")
        fields_set = self.model_fields_set
        if self.every_n_epochs is not None:
            if "stage" in fields_set and self.stage is not TrainingStage.AFTER_EPOCH:
                raise ValueError("every_n_epochs requires stage=AFTER_EPOCH.")
            if "frequency" in fields_set and self.frequency != 1:
                raise ValueError("every_n_epochs cannot be combined with frequency.")
            self.stage = TrainingStage.AFTER_EPOCH
            self.frequency = 1
        if self.every_n_steps is not None:
            if (
                "stage" in fields_set
                and self.stage is not TrainingStage.AFTER_OPTIMIZER_STEP
            ):
                raise ValueError("every_n_steps requires stage=AFTER_OPTIMIZER_STEP.")
            if "frequency" in fields_set and self.frequency != 1:
                raise ValueError("every_n_steps cannot be combined with frequency.")
            self.stage = TrainingStage.AFTER_OPTIMIZER_STEP
            self.frequency = 1
        return self

    def __call__(self, ctx: TrainContext, stage: TrainingStage) -> None:
        """Run validation if the configured schedule matches this dispatch."""
        if not self._should_run(ctx, stage):
            return
        workflow = ctx.workflow
        if workflow is None:
            raise RuntimeError("EvaluateHook requires TrainContext.workflow.")
        distributed_manager = _distributed_manager(ctx)
        device = workflow.devices[0]
        loss_fn = self._resolve_loss_fn(workflow)
        validation_fn = self.validation_fn or workflow.training_fn
        grad_enabled = self._resolve_grad_enabled(loss_fn)
        model_arg, modules, ema_model_keys = self._validation_model_arg(ctx)
        precision_context, precision = self._mixed_precision_context(ctx, device)
        modules = _unique_modules(modules)
        modes = _module_training_modes(modules)
        if self.set_eval:
            for module, _training in modes.values():
                module.eval()

        accumulator = _LossAccumulator(device)
        sample_buffer: list[Batch] = []
        sample_buffer_start: int | None = None
        sink_started = False
        successful = False
        grad_snapshot = _snapshot_parameter_grads(modules) if grad_enabled else {}
        try:
            self._begin_sink(ctx)
            sink_started = self.sink is not None
            if grad_enabled:
                _clear_parameter_grads(modules)
            for validation_batch_count, batch in enumerate(self.validation_data):
                validation_batch = batch.to(device, non_blocking=True)
                if grad_enabled:
                    _clear_parameter_grads(modules)
                grad_ctx = torch.enable_grad() if grad_enabled else torch.no_grad()
                with grad_ctx, precision_context():
                    predictions = validation_fn(model_arg, validation_batch)
                    loss_out = compute_supervised_loss(
                        loss_fn,
                        predictions,
                        validation_batch,
                        step=ctx.step_count,
                        epoch=ctx.epoch,
                        batch_label="Validation batch",
                    )
                accumulator.update(loss_out)
                if self.sink is not None and self.write_samples:
                    output_batch = self._sample_output_batch(
                        validation_batch,
                        predictions,
                        loss_out,
                        batch_count=validation_batch_count,
                        ctx=ctx,
                    )
                else:
                    output_batch = None
                if output_batch is not None and self.write_samples:
                    sample_buffer_start = self._write_or_buffer_sample_batch(
                        output_batch,
                        batch_count=validation_batch_count,
                        ctx=ctx,
                        buffer=sample_buffer,
                        buffer_start=sample_buffer_start,
                    )
                if self.sink is not None and self.write_batch_summaries:
                    self._write_sink_batch_summary(
                        self._batch_summary_output_batch(
                            loss_out,
                            validation_batch,
                            batch_count=validation_batch_count,
                            ctx=ctx,
                        ),
                        batch_count=validation_batch_count,
                        ctx=ctx,
                    )
            self._flush_sample_buffer(
                sample_buffer,
                buffer_start=sample_buffer_start,
                ctx=ctx,
            )

            num_workflow_models = len(getattr(workflow, "models", {}) or {})
            model_source = (
                "ema"
                if ema_model_keys and len(ema_model_keys) == num_workflow_models
                else "mixed"
                if ema_model_keys
                else "live"
            )
            summary = accumulator.summary(
                name=self.name,
                model_source=model_source,
                ema_model_keys=ema_model_keys,
                precision=precision,
                publish=distributed_get_rank(distributed_manager) == 0,
                distributed_manager=distributed_manager,
            )
            if self.sink is not None and self.write_epoch_summary:
                local_scalar_summary = accumulator.scalar_means(
                    distributed=False,
                    distributed_manager=distributed_manager,
                )
                global_scalar_summary = accumulator.scalar_means(
                    distributed=True,
                    distributed_manager=distributed_manager,
                )
                self._write_sink_epoch_summary(
                    self._epoch_summary_output_batch(
                        local_scalar_summary,
                        global_scalar_summary,
                        ctx=ctx,
                    ),
                    local_summary=local_scalar_summary,
                    global_summary=global_scalar_summary,
                    ctx=ctx,
                )
            successful = True
        finally:
            if grad_enabled:
                _clear_parameter_grads(modules)
                _restore_parameter_grads(grad_snapshot)
            if self.set_eval:
                for module, training in modes.values():
                    module.train(training)
            if sink_started:
                self._end_sink(ctx)
            if successful and self.sink is not None and self.distributed_barrier:
                _distributed_barrier(distributed_manager)
        self._has_run = True
        workflow.validation = summary
        ctx.validation = summary

    def _begin_sink(self, ctx: TrainContext) -> None:
        """Notify a sink that one validation run is starting."""
        if self.sink is None:
            return
        self._configure_sink_distributed_manager(ctx)
        method = getattr(self.sink, "begin_evaluation", None)
        if method is not None:
            method(step_count=ctx.step_count, epoch=ctx.epoch, name=self.name)

    def _configure_sink_distributed_manager(self, ctx: TrainContext) -> None:
        """Pass the workflow distributed manager to sinks that accept one."""
        if self.sink is None:
            return
        method = getattr(self.sink, "set_distributed_manager", None)
        if callable(method):
            method(_distributed_manager(ctx))

    def _end_sink(self, ctx: TrainContext) -> None:
        """Notify a sink that one validation run has finished."""
        if self.sink is None:
            return
        method = getattr(self.sink, "end_evaluation", None)
        if method is not None:
            method(step_count=ctx.step_count, epoch=ctx.epoch, name=self.name)

    def _sample_output_batch(
        self,
        batch: Batch,
        predictions: Mapping[str, torch.Tensor],
        loss_out: ComposedLossOutput,
        *,
        batch_count: int,
        ctx: TrainContext,
    ) -> Batch:
        """Pack per-sample loss diagnostics into a new validation batch."""
        output = batch.clone()
        num_graphs = output.num_graphs
        device = output.device
        _set_batch_tensor(
            output,
            "eval_step",
            torch.full((num_graphs,), ctx.step_count, dtype=torch.long, device=device),
            level="system",
        )
        _set_batch_tensor(
            output,
            "eval_epoch",
            torch.full((num_graphs,), ctx.epoch, dtype=torch.long, device=device),
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

        if self.include_predictions:
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
        self,
        loss_out: ComposedLossOutput,
        batch: Batch,
        *,
        batch_count: int,
        ctx: TrainContext,
    ) -> Batch:
        """Pack one validation batch's summary into a compact batch."""
        device = batch.device
        fields: dict[str, torch.Tensor] = {
            "eval_step": torch.tensor(ctx.step_count, device=device),
            "eval_epoch": torch.tensor(ctx.epoch, device=device),
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
        self,
        local_summary: Mapping[str, torch.Tensor],
        global_summary: Mapping[str, torch.Tensor],
        *,
        ctx: TrainContext,
    ) -> Batch:
        """Pack validation-epoch scalar means into a compact batch."""
        device = ctx.workflow.devices[0]
        fields: dict[str, torch.Tensor] = {
            "eval_step": torch.tensor(ctx.step_count, device=device),
            "eval_epoch": torch.tensor(ctx.epoch, device=device),
        }
        for name, value in local_summary.items():
            fields[_safe_batch_key("eval_rank_mean", name)] = value
        for name, value in global_summary.items():
            fields[_safe_batch_key("eval_global_mean", name)] = value
        return _minimal_summary_batch(fields, device=device)

    def _write_or_buffer_sample_batch(
        self,
        batch: Batch,
        *,
        batch_count: int,
        ctx: TrainContext,
        buffer: list[Batch],
        buffer_start: int | None,
    ) -> int | None:
        """Write or buffer one sample output batch."""
        if self.sink is None:
            return None
        if self.write_batch_size is None:
            self._write_sink_samples(batch, batch_count=batch_count, ctx=ctx)
            return None
        if buffer_start is None:
            buffer_start = batch_count
        buffer.append(batch)
        if len(buffer) >= self.write_batch_size:
            self._flush_sample_buffer(buffer, buffer_start=buffer_start, ctx=ctx)
            return None
        return buffer_start

    def _flush_sample_buffer(
        self,
        buffer: list[Batch],
        *,
        buffer_start: int | None,
        ctx: TrainContext,
    ) -> None:
        """Write and clear buffered sample output batches."""
        if self.sink is None or not buffer:
            return
        if buffer_start is None:
            raise RuntimeError("EvaluateHook sample buffer is missing its start index.")
        self._write_sink_samples(
            _combine_batches(buffer),
            batch_count=buffer_start,
            ctx=ctx,
        )
        buffer.clear()

    def _write_sink_samples(
        self,
        batch: Batch,
        *,
        batch_count: int,
        ctx: TrainContext,
    ) -> None:
        """Write one augmented sample batch to the configured sink."""
        if self.sink is None:
            return
        method = getattr(self.sink, "write_samples", None)
        if method is not None:
            method(
                batch,
                step_count=ctx.step_count,
                epoch=ctx.epoch,
                batch_count=batch_count,
            )
            return
        write = getattr(self.sink, "write", None)
        if write is not None:
            write(batch)

    def _write_sink_batch_summary(
        self,
        batch: Batch,
        *,
        batch_count: int,
        ctx: TrainContext,
    ) -> None:
        """Write a per-validation-batch summary if the sink supports it."""
        if self.sink is None:
            return
        method = getattr(self.sink, "write_batch_summary", None)
        if method is not None:
            method(
                batch,
                step_count=ctx.step_count,
                epoch=ctx.epoch,
                batch_count=batch_count,
            )

    def _write_sink_epoch_summary(
        self,
        batch: Batch,
        *,
        local_summary: Mapping[str, torch.Tensor],
        global_summary: Mapping[str, torch.Tensor],
        ctx: TrainContext,
    ) -> None:
        """Write a validation-epoch summary if the sink supports it."""
        if self.sink is None:
            return
        method = getattr(self.sink, "write_epoch_summary", None)
        if method is not None:
            method(
                batch,
                step_count=ctx.step_count,
                epoch=ctx.epoch,
                local_summary=local_summary,
                global_summary=global_summary,
            )

    def _should_run(self, ctx: TrainContext, stage: TrainingStage) -> bool:
        """Return whether this dispatch satisfies the configured schedule."""
        if self._is_end_fallback_stage(stage):
            return not self._has_run
        if stage is not self.stage:
            return False
        if self.every_n_epochs is not None:
            return (ctx.epoch + 1) % self.every_n_epochs == 0
        if self.every_n_steps is not None:
            if self._optimizer_step_skipped(ctx):
                return False
            return (ctx.step_count + 1) % self.every_n_steps == 0
        return True

    def _is_end_fallback_stage(self, stage: TrainingStage) -> bool:
        """Return whether ``stage`` is the default end-of-training fallback."""
        return (
            self.run_at_end
            and stage is TrainingStage.AFTER_TRAINING
            and self.stage is TrainingStage.AFTER_EPOCH
            and self.every_n_epochs is None
            and self.every_n_steps is None
        )

    def _optimizer_step_skipped(self, ctx: TrainContext) -> bool:
        """Return whether the update orchestrator skipped the last optimizer step."""
        for hook in _iter_registered_hooks(ctx.workflow.hooks):
            if isinstance(hook, TrainingUpdateOrchestrator):
                return hook.optimizer_step_skipped
        return False

    def _resolve_loss_fn(self, workflow: Any) -> ComposedLossFunction:
        """Return the explicit validation loss or the workflow loss."""
        if self.loss_fn is not None:
            return self.loss_fn
        return as_composed_loss(workflow.loss_fn)

    def _resolve_grad_enabled(self, loss_fn: ComposedLossFunction) -> bool:
        """Resolve the validation autograd policy from ``grad_mode``."""
        if self.grad_mode == "enabled":
            return True
        if self.grad_mode == "disabled":
            return False
        return self._loss_requires_grad(loss_fn)

    def _loss_requires_grad(self, loss_fn: ComposedLossFunction) -> bool:
        """Infer whether the loss needs autograd-enabled validation."""
        unknown: list[str] = []
        for component in loss_fn.components:
            requires_eval_grad = getattr(component, "requires_eval_grad", None)
            if requires_eval_grad is True:
                return True
            if requires_eval_grad is None:
                unknown.append(type(component).__name__)
        if unknown:
            names = ", ".join(unknown)
            raise ValueError(
                "EvaluateHook grad_mode='auto' cannot infer whether validation "
                f"requires gradients for component(s): {names}. Set "
                "grad_mode='enabled' or grad_mode='disabled' explicitly."
            )
        return False

    def _validation_model_arg(
        self, ctx: TrainContext
    ) -> tuple[Any, tuple[nn.Module, ...], tuple[str, ...]]:
        """Return validation model argument, modules to manage, and EMA keys."""
        workflow = ctx.workflow
        live_models = workflow.models
        ema_models = self._initialized_ema_models(ctx)
        single_model_input = bool(getattr(workflow, "single_model_input", False))

        if single_model_input:
            live = live_models["main"]
            ema = ema_models.get("main")
            if self.use_ema == "always" and ema is None:
                raise RuntimeError(
                    "EvaluateHook use_ema='always' requires an initialized "
                    "EMAHook for model_key='main'."
                )
            model = ema if ema is not None and self.use_ema != "never" else live
            return (
                model,
                (model,),
                ("main",) if model is ema and ema is not None else (),
            )

        validation_models = dict(live_models)
        used_ema_keys: list[str] = []
        if self.use_ema != "never":
            for key, model in ema_models.items():
                if key in validation_models:
                    validation_models[key] = model
                    used_ema_keys.append(key)
        if self.use_ema == "always":
            missing = sorted(set(validation_models) - set(used_ema_keys))
            if missing:
                raise RuntimeError(
                    "EvaluateHook use_ema='always' requires initialized EMAHook "
                    "weights for every workflow model; missing model_key(s): "
                    f"{missing}."
                )
        modules = tuple(
            module
            for module in validation_models.values()
            if isinstance(module, nn.Module)
        )
        return validation_models, modules, tuple(sorted(used_ema_keys))

    def _initialized_ema_models(self, ctx: TrainContext) -> dict[str, nn.Module]:
        """Return initialized EMA modules keyed by their source model key."""
        if self.use_ema == "never":
            return {}
        ema_models: dict[str, nn.Module] = {}
        saw_matching_hook = False
        for hook in _iter_registered_hooks(ctx.workflow.hooks):
            if not isinstance(hook, EMAHook):
                continue
            saw_matching_hook = True
            try:
                module = hook.get_averaged_model().module
            except RuntimeError:
                continue
            if hook.model_key in ema_models:
                raise RuntimeError(
                    "EvaluateHook found multiple initialized EMAHook instances "
                    f"for model_key={hook.model_key!r}."
                )
            ema_models[hook.model_key] = module
        if self.use_ema == "always" and saw_matching_hook and not ema_models:
            raise RuntimeError(
                "EvaluateHook use_ema='always' found EMAHook instance(s), but none "
                "had initialized averaged weights."
            )
        return ema_models

    def _mixed_precision_context(
        self, ctx: TrainContext, device: torch.device
    ) -> tuple[Callable[[], AbstractContextManager[None]], str]:
        """Return validation autocast context factory and precision label."""
        if self.use_mixed_precision == "never":
            return nullcontext, "float32"
        for hook in _iter_registered_hooks(ctx.workflow.hooks):
            if isinstance(hook, MixedPrecisionHook):
                precision = str(hook.precision).removeprefix("torch.")
                return lambda: hook.inference_autocast(device), precision
        if self.use_mixed_precision == "always":
            raise RuntimeError(
                "EvaluateHook use_mixed_precision='always' requires a registered "
                "MixedPrecisionHook."
            )
        return nullcontext, "float32"
