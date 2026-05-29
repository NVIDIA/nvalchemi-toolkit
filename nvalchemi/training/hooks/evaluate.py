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

from collections.abc import Callable, Iterable, Iterator, Mapping
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
from torch import distributed as dist
from torch import nn

from nvalchemi.hooks._context import TrainContext
from nvalchemi.training._stages import TrainingStage
from nvalchemi.training.hooks.ema import EMAHook
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

    def summary(
        self,
        *,
        name: str,
        model_source: str,
        ema_model_keys: tuple[str, ...],
        precision: str,
        publish: bool,
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
        distributed_reduced = _distributed_sum_in_place(packed)
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


def _distributed_sum_in_place(value: torch.Tensor) -> bool:
    """All-reduce ``value`` when torch.distributed is active."""
    if not dist.is_available() or not dist.is_initialized():
        return False
    dist.all_reduce(value, op=dist.ReduceOp.SUM)
    return True


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
        return self

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
        grad_snapshot = _snapshot_parameter_grads(modules) if grad_enabled else {}
        try:
            if grad_enabled:
                _clear_parameter_grads(modules)
            for batch in self.validation_data:
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
        finally:
            if grad_enabled:
                _clear_parameter_grads(modules)
                _restore_parameter_grads(grad_snapshot)
            if self.set_eval:
                for module, training in modes.values():
                    module.train(training)

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
            publish=ctx.global_rank == 0,
        )
        self._has_run = True
        workflow.validation = summary
        ctx.validation = summary

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
