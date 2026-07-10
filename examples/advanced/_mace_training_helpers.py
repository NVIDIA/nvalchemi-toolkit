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
"""LR schedule, data transforms, and metrics helpers for the MACE training example.

Implements the two-stage cosine-then-constant LR schedule, validation metrics
logging, and shared dataloader utilities for ``examples/advanced/10_mace_training.py``.
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any, ClassVar, Literal

import torch
from omegaconf import DictConfig
from scipy.constants import angstrom, electron_volt, giga
from torch.utils.data.distributed import DistributedSampler

from nvalchemi.data.batch import Batch
from nvalchemi.data.datapipes import DataLoader, Dataset, InMemoryDataset
from nvalchemi.distributed import DistributedManager
from nvalchemi.training import TrainingStage, TrainingStrategy
from nvalchemi.training.distributed import all_reduce, get_world_size
from nvalchemi.training.hooks import TrainingUpdateHook

GPA_TO_EV_PER_ANGSTROM_CUBED = giga * angstrom**3 / electron_volt
KBAR_TO_EV_PER_ANGSTROM_CUBED = 0.1 * GPA_TO_EV_PER_ANGSTROM_CUBED


def get_dtype(name: str) -> torch.dtype:
    """Return the torch dtype named by a MACE config value."""
    return {
        "float32": torch.float32,
        "float64": torch.float64,
        "float": torch.float32,
        "double": torch.float64,
    }[str(name).lower()]


class ToDType:
    """Cast floating tensors on each batch to the configured dtype.

    Attributes
    ----------
    dtype : torch.dtype
        Target floating-point dtype.
    """

    def __init__(self, dtype: torch.dtype) -> None:
        """Store the target dtype."""
        self.dtype = dtype

    def __call__(
        self,
        batch: Batch,
    ) -> Batch:
        """Cast floating-point tensor attributes in place."""
        for group in batch._storage.groups.values():
            for key, value in list(group._data.items()):
                if isinstance(value, torch.Tensor) and value.is_floating_point():
                    group._data[key] = value.to(dtype=self.dtype)
        return batch


class ScaleField:
    """Scale one tensor field on a :class:`~nvalchemi.data.Batch`.

    Attributes
    ----------
    field : str
        Name of the tensor field to scale.
    scale : float
        Multiplicative scale factor.
    missing_ok : bool
        Whether missing fields should be ignored.
    """

    def __init__(
        self,
        field: str,
        scale: float,
        *,
        missing_ok: bool = False,
    ) -> None:
        """Configure the field scale transform."""
        self.field = field
        self.scale = float(scale)
        self.missing_ok = bool(missing_ok)

    def __call__(
        self,
        batch: Batch,
    ) -> Batch:
        """Scale ``field`` in place when present."""
        if not hasattr(batch, self.field):
            if self.missing_ok:
                return batch
            raise AttributeError(f"Batch has no field {self.field!r}.")
        value = getattr(batch, self.field)
        if not isinstance(value, torch.Tensor):
            raise TypeError(
                f"Batch field {self.field!r} must be a torch.Tensor, "
                f"got {type(value).__name__}."
            )
        setattr(batch, self.field, value * self.scale)
        return batch


def get_cfg(cfg: Any, key: str, default: Any = None) -> Any:
    """Return a config value from an OmegaConf node, namespace, or mapping."""
    if isinstance(cfg, Mapping):
        return cfg.get(key, default)
    return cfg.get(key, default) if hasattr(cfg, "get") else getattr(cfg, key, default)


def stress_target_scale(data_cfg: Any) -> float:
    """Return a scale that converts stress targets to eV/A^3."""
    units = str(get_cfg(data_cfg, "stress_unit", "kBar")).lower()
    opposite_sign = get_cfg(data_cfg, "stress_opposite_sign_convention", True)
    if not isinstance(opposite_sign, bool):
        raise TypeError("stress_opposite_sign_convention must be true or false.")
    sign = -1.0 if opposite_sign else 1.0

    if units in {"gpa", "gigapascal", "gigapascals"}:
        return sign * GPA_TO_EV_PER_ANGSTROM_CUBED
    if units in {"kbar", "kilobar", "kilobars"}:
        return sign * KBAR_TO_EV_PER_ANGSTROM_CUBED
    if units in {"ev/a^3", "ev/angstrom^3", "ev_per_a3"}:
        return sign
    raise ValueError(
        f"Unsupported stress_unit={units!r}; use 'GPa', 'kBar', or 'eV/A^3'."
    )


def count_model_parameters(model: torch.nn.Module) -> int:
    """Return the total number of parameters in a model.

    Parameters
    ----------
    model : torch.nn.Module
        Model whose parameters should be counted.

    Returns
    -------
    int
        Total number of scalar parameters across all model parameters.
    """
    return sum(parameter.numel() for parameter in model.parameters())


def make_validation_sampler(
    dataset: Dataset | InMemoryDataset,
    manager: DistributedManager,
) -> DistributedSampler | None:
    """Return the distributed validation sampler for multi-rank validation."""
    if manager.world_size <= 1:
        return None
    return DistributedSampler(
        dataset,
        num_replicas=manager.world_size,
        rank=manager.rank,
        shuffle=False,
        drop_last=True,
    )


def close_zarr_loaders(*loaders: DataLoader | None) -> None:
    """Close loader datasets, ignoring close failures."""
    for loader in loaders:
        if loader is None:
            continue
        try:
            loader.dataset.close()
        except Exception:  # noqa: S110
            pass


def save_final_checkpoint(
    cfg: DictConfig,
    strategy: TrainingStrategy,
    manager: DistributedManager,
) -> None:
    """Save a final checkpoint on rank zero when configured."""
    checkpoint_cfg = cfg.training.checkpoint
    if not bool(checkpoint_cfg.get("save_final", False)):
        return
    if manager.rank != 0:
        return
    step_interval = checkpoint_cfg.get("step_interval", None)
    if (
        step_interval is not None
        and strategy.step_count > 0
        and strategy.step_count % int(step_interval) == 0
    ):
        return
    strategy.save_checkpoint(checkpoint_cfg.dir)


class GradientClipHook(TrainingUpdateHook):
    """Clip gradients by global norm immediately before optimizer stepping.

    Parameters
    ----------
    max_norm : float
        Maximum total gradient norm passed to
        :func:`torch.nn.utils.clip_grad_norm_`.

    Attributes
    ----------
    max_norm : float
        Maximum gradient norm.
    last_total_norm : torch.Tensor | None
        Total gradient norm returned by the most recent clipping call.
    """

    priority: ClassVar[int] = 30
    _exclusive_update_key: ClassVar[str | None] = "GradientClipHook"

    def __init__(self, max_norm: float) -> None:
        """Configure the clipping threshold."""
        self.max_norm = float(max_norm)
        if self.max_norm <= 0.0:
            raise ValueError("max_norm must be positive.")
        self.last_total_norm: torch.Tensor | None = None

    def __call__(
        self,
        ctx: Any,
        stage: TrainingStage,
        will_skip: bool,
    ) -> tuple[bool, torch.Tensor | None]:
        """Clip optimizer gradients at the optimizer-step stage."""
        if stage is not TrainingStage.DO_OPTIMIZER_STEP or will_skip:
            return True, ctx.loss
        params: list[torch.nn.Parameter] = []
        seen: set[int] = set()
        for optimizer in ctx.optimizers:
            for group in optimizer.param_groups:
                for parameter in group["params"]:
                    if parameter.grad is None:
                        continue
                    parameter_id = id(parameter)
                    if parameter_id in seen:
                        continue
                    seen.add(parameter_id)
                    params.append(parameter)
        if params:
            self.last_total_norm = torch.nn.utils.clip_grad_norm_(
                params,
                max_norm=self.max_norm,
            )
        else:
            self.last_total_norm = None
        return True, ctx.loss


class TwoStageCosineConstantLR(torch.optim.lr_scheduler.SequentialLR):
    """Customized learning rate scheduler that starts with a cosine annealing schedule
    and then switches to a constant learning rate. It uses the SequentialLR scheduler to
    chain the two schedules for convenience. It is possible to directly inherit from
    ``torch.optim.lr_scheduler.LRScheduler`` as well.

    Attributes
    ----------
    first_stage_steps : int
        Optimizer steps assigned to the first training stage.
    second_stage_lr : float
        Learning rate used after ``first_stage_steps``.
    eta_min : float
        Minimum learning rate reached by the cosine schedule.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        *,
        first_stage_steps: int,
        second_stage_lr: float,
        eta_min: float = 1e-5,
        last_epoch: int = -1,
    ) -> None:
        """Configure the two-stage learning-rate schedule."""
        if first_stage_steps <= 0:
            raise ValueError("first_stage_steps must be positive.")
        if second_stage_lr < 0.0:
            raise ValueError("second_stage_lr must be non-negative.")
        if eta_min < 0.0:
            raise ValueError("eta_min must be non-negative.")
        self.first_stage_steps = int(first_stage_steps)
        self.second_stage_lr = float(second_stage_lr)
        self.eta_min = float(eta_min)
        base_lr = float(optimizer.param_groups[0]["lr"])
        if base_lr <= 0.0:
            raise ValueError("base learning rate must be positive.")
        if any(float(group["lr"]) != base_lr for group in optimizer.param_groups):
            raise ValueError("all optimizer parameter groups must share one LR.")
        second_stage_factor = self.second_stage_lr / base_lr

        # SequentialLR chains the two schedulers.
        # Stage 1: cosine decay from the optimizer's initial LR down to eta_min.
        schedulers = [
            torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.first_stage_steps,
                eta_min=self.eta_min,
            ),
            # Stage 2: constant LR via a fixed multiplicative factor on base_lr.
            torch.optim.lr_scheduler.LambdaLR(
                optimizer,
                lr_lambda=lambda _: second_stage_factor,
            ),
        ]
        super().__init__(
            optimizer,
            schedulers=schedulers,
            # Hand off to the second stage scheduler after first_stage_steps.
            milestones=[self.first_stage_steps],
            last_epoch=last_epoch,
        )


class JsonLinesLogger:
    """Append scalar metric snapshots to a JSON Lines file.

    Attributes
    ----------
    path : pathlib.Path
        JSON Lines file to be saved.
    """

    def __init__(self, path: str | Path) -> None:
        """Open ``path`` for append logging, creating parent directories."""
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("a", encoding="utf-8")

    def log_metrics(self, metrics: Mapping[str, Any], step: int | None = None) -> None:
        """Append one set of metrics to the JSON Lines file."""
        payload: dict[str, Any] = {
            "metrics": {
                key: float(value)
                for key, value in metrics.items()
                if key != "global_rank"
            },
            "step": step,
            "time_s": time.time(),
        }
        if (global_rank := metrics.get("global_rank")) is not None:
            payload["global_rank"] = int(global_rank)
        self._file.write(json.dumps(payload, sort_keys=True) + "\n")
        self._file.flush()

    def close(self) -> None:
        """Close the underlying file handle."""
        self._file.close()


class TrainingMetricsLogger:
    """Log training and validation metrics from hook context.

    Training metrics are logged at ``AFTER_BACKWARD`` every ``every`` optimizer
    steps; validation metrics are logged at every ``AFTER_VALIDATION``. Training
    metrics are all-reduced across ranks. When configured, the optional external logger
    receives metrics such as ``train/loss`` and ``validation/loss``.

    Parameters
    ----------
    every : int
        Optimizer-step interval for training metric logs. Validation
        metrics are still emitted after every validation pass.
    logger : Any | None, optional
        Optional external logger (e.g., MLflow, W&B, or the provided
        :class:`JsonLinesLogger`). Supported APIs are ``log_metrics``,
        ``log``, or ``log_metric``, each accepting a ``step=`` keyword.
    logger_axis : {"step", "epoch"}, optional
        X-axis value forwarded to ``logger`` as ``step``. Default
        ``"step"`` uses ``ctx.step_count``; ``"epoch"`` uses
        ``ctx.epoch``.
    """

    stage = None

    def __init__(
        self,
        *,
        every: int,
        logger: Any | None = None,
        logger_axis: Literal["step", "epoch"] = "step",
    ) -> None:
        """Configure logging cadence and optional external logger."""
        if logger_axis not in {"step", "epoch"}:
            raise ValueError("logger_axis must be 'step' or 'epoch'.")
        self.frequency = 1
        self.every = max(int(every), 1)
        self.logger = logger
        self.logger_axis = logger_axis

    def _runs_on_stage(self, stage: TrainingStage) -> bool:
        """Return whether this logger handles the specified ``stage``."""
        return stage in {TrainingStage.AFTER_BACKWARD, TrainingStage.AFTER_VALIDATION}

    def close(self) -> None:
        """Close the optional external experiment logger when supported."""
        close = getattr(self.logger, "close", None)
        if callable(close):
            close()

    def __call__(self, ctx: Any, stage: TrainingStage) -> None:
        """Log training metrics at ``AFTER_BACKWARD`` on the configured cadence.

        Metrics are all-reduced across ranks, then printed on rank zero and
        forwarded to the optional external logger.
        """
        if stage is TrainingStage.AFTER_VALIDATION:
            self._log_validation(ctx)
            return
        if ctx.loss is None:
            return
        if ctx.step_count % self.every != 0:
            return
        metrics = {"loss": float(ctx.loss.detach().cpu())}
        if ctx.losses is not None:
            metrics.update(
                {
                    self._metric_name(name): float(value.detach().cpu())
                    for name, value in ctx.losses.get(
                        "per_component_unweighted", {}
                    ).items()
                }
            )
        reduced = self._reduced_metrics(
            metrics,
            ctx.loss.device,
            ctx.workflow.distributed_manager,
        )
        self._require_finite(reduced, ctx)
        if ctx.global_rank != 0:
            return
        suffix = " ".join(
            f"{name}={value:.6g}" for name, value in reduced.items() if name != "loss"
        )
        message = f"step={ctx.step_count} epoch={ctx.epoch} loss={reduced['loss']:.6g}"
        if suffix:
            message = f"{message} {suffix}"
        self._log_external_metrics(
            ctx,
            {f"train/{name}": value for name, value in reduced.items()}
            | {
                "train/step": float(ctx.step_count),
                "train/epoch": float(ctx.epoch),
            },
        )
        print(
            message,
            flush=True,
        )

    def _log_validation(self, ctx: Any) -> None:
        """Log validation metrics at ``AFTER_VALIDATION`` on rank zero.

        Prints the latest validation summary and forwards the metrics
        to the optional external logger.
        """
        if ctx.global_rank != 0 or ctx.validation is None:
            return
        summary = ctx.validation
        components = summary.get("per_component_unweighted", {})
        self._require_finite(
            {"loss": float(summary["total_loss"])}
            | {
                self._metric_name(name): float(value)
                for name, value in components.items()
            },
            ctx,
        )

        # Prepare the message to be printed to the console.
        message = (
            f"validation step={ctx.step_count} epoch={ctx.epoch} "
            f"loss={float(summary['total_loss']):.6g} "
            f"model_source={summary.get('model_source', 'unknown')}"
        )
        lrs = self._learning_rates(ctx)
        if len(lrs) == 1:
            message = f"{message} lr={lrs[0]:.6g}"
        elif lrs:
            message = f"{message} lrs={','.join(f'{lr:.6g}' for lr in lrs)}"
        ema_keys = summary.get("ema_model_keys")
        if ema_keys:
            message = f"{message} ema_model_keys={ema_keys}"
        if components:
            suffix = " ".join(
                f"{self._metric_name(name)}={float(value):.6g}"
                for name, value in sorted(components.items())
            )
            message = f"{message} {suffix}"

        # Prepare the metrics to be logged to the external logger.
        external_metrics = {"validation/loss": float(summary["total_loss"])}
        external_metrics.update(
            {
                f"validation/{self._metric_name(name)}": float(value)
                for name, value in components.items()
            }
        )
        if len(lrs) == 1:
            external_metrics["lr"] = lrs[0]
        else:
            external_metrics.update(
                {f"lr/group_{idx}": lr for idx, lr in enumerate(lrs)}
            )
        external_metrics.update(
            {
                "validation/step": float(ctx.step_count),
                "validation/epoch": float(ctx.epoch),
            }
        )
        self._log_external_metrics(ctx, external_metrics)
        print(message, flush=True)

    @staticmethod
    def _require_finite(metrics: Mapping[str, float], ctx: Any) -> None:
        """Raise ``RuntimeError`` if any metric value is non-finite."""
        bad = {
            name: value for name, value in metrics.items() if not math.isfinite(value)
        }
        if bad:
            raise RuntimeError(
                f"Non-finite metrics at step={ctx.step_count} epoch={ctx.epoch}: {bad}"
            )

    def _log_external_metrics(self, ctx: Any, metrics: dict[str, float]) -> None:
        """Log metrics to an optional external experiment logger.

        Tries ``log_metrics``, then ``log``, then ``log_metric`` to accommodate
        different external logger APIs.
        """
        if self.logger is None or not metrics:
            return
        step = self._logger_step(ctx)
        log_metrics = getattr(self.logger, "log_metrics", None)
        if callable(log_metrics):
            log_metrics(metrics, step=step)
            return
        log = getattr(self.logger, "log", None)
        if callable(log):
            log(metrics, step=step)
            return
        log_metric = getattr(self.logger, "log_metric", None)
        if callable(log_metric):
            for name, value in metrics.items():
                log_metric(name, value, step=step)
            return
        raise TypeError(
            "logger must provide log_metrics(metrics, step=...), "
            "log(metrics, step=...), or log_metric(name, value, step=...)."
        )

    def _logger_step(self, ctx: Any) -> int:
        """Return the external logger x-axis value."""
        if self.logger_axis == "epoch":
            return int(ctx.epoch)
        return int(ctx.step_count)

    @staticmethod
    def _learning_rates(ctx: Any) -> list[float]:
        """Return current learning rates from all optimizer parameter groups."""
        return [
            float(group["lr"])
            for optimizer in getattr(ctx, "optimizers", ())
            for group in optimizer.param_groups
            if "lr" in group
        ]

    def _metric_name(self, name: str) -> str:
        """Return the concise display name used for component loss metrics."""
        return name.removesuffix("_unweighted")

    @staticmethod
    def _reduced_metrics(
        metrics: dict[str, float],
        device: torch.device,
        distributed_manager: Any | None,
    ) -> dict[str, float]:
        """Return metrics averaged across distributed ranks."""
        names = sorted(metrics)
        values = torch.tensor([metrics[name] for name in names], device=device)
        all_reduce(values, distributed_manager)
        world_size = get_world_size(distributed_manager)
        if world_size > 1:
            values /= float(world_size)
        return {
            name: float(value) for name, value in zip(names, values.cpu(), strict=True)
        }
