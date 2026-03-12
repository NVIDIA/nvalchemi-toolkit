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
"""Trainer class and TrainingResult dataclass."""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from loguru import logger

from nvalchemi.training._checkpoint import (
    load_training_checkpoint,
    save_training_checkpoint,
)
from nvalchemi.training._configs import TrainingConfig
from nvalchemi.training._distributed import reduce_loss_across_ranks, wrap_model
from nvalchemi.training._hooks import TrainingContext, TrainingHook
from nvalchemi.training._stages import TrainingStageEnum
from nvalchemi.training._terminate import StopTraining

if TYPE_CHECKING:
    from collections.abc import Iterable

    from nvalchemi.models.base import BaseModelMixin
    from nvalchemi.training.losses._base import CompositeLoss


@dataclasses.dataclass
class TrainingResult:
    """Result returned by :meth:`Trainer.fit`.

    Attributes
    ----------
    epochs_completed : int
        Number of full epochs completed.
    best_val_loss : float | None
        Best validation loss observed, or ``None`` if validation was not run.
    best_checkpoint : Path | None
        Path to the best checkpoint, or ``None`` if no checkpoint was saved.
    final_metrics : dict[str, float]
        Metrics from the final validation or training step.
    history : list[dict[str, float]]
        Per-epoch metrics history.
    """

    epochs_completed: int = 0
    best_val_loss: float | None = None
    best_checkpoint: Path | None = None
    final_metrics: dict[str, float] = dataclasses.field(default_factory=dict)
    history: list[dict[str, float]] = dataclasses.field(default_factory=list)


class Trainer:
    """Orchestrates the training loop for MLIP models.

    Parameters
    ----------
    model : BaseModelMixin
        The model to train.
    loss : CompositeLoss
        Composite loss module.
    optimizer : torch.optim.Optimizer or Sequence[torch.optim.Optimizer]
        One or more optimizer instances.
    scheduler : LRScheduler, Sequence[LRScheduler], or None
        Optional learning-rate scheduler(s).  When multiple optimizers are
        provided, a sequence of the same length is expected.
    train_loader : Iterable
        Training data loader.
    val_loader : Iterable | None
        Optional validation data loader.
    config : TrainingConfig
        Training hyper-parameters.
    dist_manager : Any | None
        Optional distributed manager (e.g. from physicsnemo).
    """

    def __init__(
        self,
        model: BaseModelMixin,
        loss: CompositeLoss,
        optimizer: (torch.optim.Optimizer | Sequence[torch.optim.Optimizer]),
        scheduler: (
            torch.optim.lr_scheduler.LRScheduler
            | Sequence[torch.optim.lr_scheduler.LRScheduler]
            | None
        ) = None,
        train_loader: Iterable | None = None,
        val_loader: Iterable | None = None,
        config: TrainingConfig | None = None,
        dist_manager: Any | None = None,
    ) -> None:
        self.config = config or TrainingConfig(max_epochs=1)
        self.dist_manager = dist_manager

        # Wrap model for distributed training.
        self.model = wrap_model(model, self.config.strategy, dist_manager)
        self.loss = loss

        # Normalise optimizers / schedulers to lists.
        if isinstance(optimizer, torch.optim.Optimizer):
            self.optimizers: list[torch.optim.Optimizer] = [optimizer]
        else:
            self.optimizers = list(optimizer)

        if scheduler is None:
            self.schedulers: list[torch.optim.lr_scheduler.LRScheduler] = []
        elif isinstance(scheduler, torch.optim.lr_scheduler.LRScheduler):
            self.schedulers = [scheduler]
        else:
            self.schedulers = list(scheduler)

        self.train_loader = train_loader
        self.val_loader = val_loader

        # Optional torch.compile.
        if self.config.torch_compile:
            self.model = torch.compile(self.model, **self.config.compile_kwargs)

        # Mixed precision scaler.
        mp = self.config.mixed_precision
        self.scaler: torch.amp.GradScaler | None = None
        if mp.enabled and mp.grad_scaler:
            self.scaler = torch.amp.GradScaler("cuda")

        # Hook registry.
        self.hooks: dict[TrainingStageEnum, list[TrainingHook]] = {
            stage: [] for stage in TrainingStageEnum
        }

        # CUDA stream bookkeeping (mirrors _CommunicationMixin).
        self._stream: torch.cuda.Stream | None = None
        self._stream_ctx: Any | None = None
        self._autocast_ctx: Any | None = None

        # Track the start epoch (set by _maybe_resume).
        self._start_epoch: int = 0

        # Resume from checkpoint if configured.
        if self.config.resume_from is not None:
            self._maybe_resume()

    # ------------------------------------------------------------------
    # Hook management
    # ------------------------------------------------------------------

    def register_hook(self, hook: TrainingHook) -> None:
        """Register a hook at its declared stage(s).

        Parameters
        ----------
        hook : TrainingHook
            A hook implementing the ``TrainingHook`` protocol.  May expose
            either a single ``stage`` or multiple ``stages``.
        """
        stages: list[TrainingStageEnum] = []
        if hasattr(hook, "stages"):
            stages = list(hook.stages)
        else:
            stages = [hook.stage]
        for stage in stages:
            self.hooks[stage].append(hook)

    def _call_hooks(self, stage: TrainingStageEnum, ctx: TrainingContext) -> None:
        """Dispatch hooks for *stage*, incrementing the stage counter first.

        Parameters
        ----------
        stage : TrainingStageEnum
            The current training stage.
        ctx : TrainingContext
            Mutable context shared across hooks.
        """
        ctx.stage_counts[stage] = ctx.stage_counts.get(stage, 0) + 1
        for hook in self.hooks[stage]:
            if ctx.global_step % hook.frequency == 0:
                hook(ctx, self.model, self)

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> Trainer:
        """Enter the training context: CUDA stream + optional autocast."""
        if self.dist_manager:
            device = self.dist_manager.device
        else:
            device_type = "cuda" if torch.cuda.is_available() else "cpu"
            rank = 0
            if torch.distributed.is_initialized():
                rank = torch.distributed.get_node_local_rank()
            device = torch.device(f"{device_type}:{rank}")
        if device.type == "cuda" and torch.cuda.is_available():
            self._stream = torch.cuda.Stream(device=device)
            self._stream_ctx = torch.cuda.stream(self._stream)
            self._stream_ctx.__enter__()

        mp = self.config.mixed_precision
        if mp.enabled:
            self._autocast_ctx = torch.amp.autocast(
                device_type=device.type,
                dtype=mp.torch_dtype,
            )
            self._autocast_ctx.__enter__()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any | None,
    ) -> None:
        """Exit the training context."""
        if self._autocast_ctx is not None:
            self._autocast_ctx.__exit__(exc_type, exc_val, exc_tb)
            self._autocast_ctx = None
        if self._stream_ctx is not None:
            self._stream_ctx.__exit__(exc_type, exc_val, exc_tb)
        self._stream = None
        self._stream_ctx = None

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def fit(self) -> TrainingResult:
        """Run the full training loop.

        Returns
        -------
        TrainingResult
            Summary of the training run.
        """
        ctx = TrainingContext(
            stage_counts={stage: 0 for stage in TrainingStageEnum},
        )
        result = TrainingResult()
        epochs_completed = 0

        try:
            for epoch in range(self._start_epoch, self.config.max_epochs):
                ctx.epoch = epoch
                self._call_hooks(TrainingStageEnum.BEFORE_EPOCH, ctx)

                self._train_one_epoch(ctx)
                epochs_completed = epoch + 1

                self._call_hooks(TrainingStageEnum.AFTER_EPOCH, ctx)

                # Validation.
                if (
                    self.val_loader is not None
                    and (epoch + 1) % self.config.val_every_n_epochs == 0
                ):
                    val_metrics = self.validate(ctx)
                    result.history.append(val_metrics)

                    val_loss = val_metrics.get("val_loss")
                    if val_loss is not None and (
                        result.best_val_loss is None or val_loss < result.best_val_loss
                    ):
                        result.best_val_loss = val_loss
                    result.final_metrics = val_metrics

                # Checkpointing.
                if (epoch + 1) % self.config.checkpoint_every_n_epochs == 0:
                    self._save_checkpoint(ctx)
                    ckpt_path = self.config.checkpoint_dir / f"epoch_{epoch}.pt"
                    if ckpt_path.exists():
                        result.best_checkpoint = ckpt_path

        except StopTraining:
            logger.info("Training stopped early via StopTraining.")
        finally:
            self._call_hooks(TrainingStageEnum.ON_TRAINING_END, ctx)

        result.epochs_completed = epochs_completed
        return result

    # ------------------------------------------------------------------
    # Single-epoch training
    # ------------------------------------------------------------------

    def _train_one_epoch(self, ctx: TrainingContext) -> None:
        """Train for one full epoch.

        Parameters
        ----------
        ctx : TrainingContext
            Mutable training context.
        """
        self.model.train()
        accum = self.config.grad_accumulation_steps

        for step_in_epoch, batch in enumerate(self.train_loader):
            self._call_hooks(TrainingStageEnum.BEFORE_STEP, ctx)

            ctx.batch = batch
            self._call_hooks(TrainingStageEnum.AFTER_DATA_LOAD, ctx)

            # Forward pass.
            self._call_hooks(TrainingStageEnum.BEFORE_FORWARD, ctx)
            model_outputs = self.model(batch)
            ctx.model_outputs = model_outputs
            self._call_hooks(TrainingStageEnum.AFTER_FORWARD, ctx)

            # Loss computation.
            self._call_hooks(TrainingStageEnum.BEFORE_LOSS, ctx)
            total_loss, loss_dict = self.loss(batch, model_outputs)
            total_loss = reduce_loss_across_ranks(total_loss, self.dist_manager)
            # Scale for gradient accumulation.
            scaled_loss = total_loss / accum
            ctx.total_loss = total_loss
            ctx.losses = loss_dict
            self._call_hooks(TrainingStageEnum.AFTER_LOSS, ctx)

            # Backward pass.
            self._call_hooks(TrainingStageEnum.BEFORE_BACKWARD, ctx)
            if self.scaler is not None:
                self.scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()
            self._call_hooks(TrainingStageEnum.AFTER_BACKWARD, ctx)

            # Optimizer step (every accum steps).
            if (step_in_epoch + 1) % accum == 0:
                self._call_hooks(TrainingStageEnum.BEFORE_OPTIMIZER_STEP, ctx)

                if self.scaler is not None:
                    if self.config.grad_clip is not None:
                        self.scaler.unscale_(*self.optimizers)
                    self._clip_gradients()
                    for opt in self.optimizers:
                        self.scaler.step(opt)
                    self.scaler.update()
                else:
                    self._clip_gradients()
                    for opt in self.optimizers:
                        opt.step()

                for sched in self.schedulers:
                    sched.step()

                # make sure model gradients are cleared
                self.model.zero_grad(set_to_none=True)
                self._call_hooks(TrainingStageEnum.AFTER_OPTIMIZER_STEP, ctx)

            self._call_hooks(TrainingStageEnum.AFTER_STEP, ctx)
            ctx.global_step += 1

    # ------------------------------------------------------------------
    # Gradient clipping
    # ------------------------------------------------------------------

    def _clip_gradients(self) -> None:
        """Apply gradient clipping according to :attr:`config.grad_clip`."""
        clip_cfg = self.config.grad_clip
        if clip_cfg is None:
            return
        params = self.model.parameters()
        if clip_cfg.method == "norm":
            torch.nn.utils.clip_grad_norm_(params, clip_cfg.max_value)
        else:
            torch.nn.utils.clip_grad_value_(params, clip_cfg.max_value)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self, ctx: TrainingContext) -> dict[str, float]:
        """Run one validation pass.

        Parameters
        ----------
        ctx : TrainingContext
            Mutable training context.

        Returns
        -------
        dict[str, float]
            Validation metrics including ``"val_loss"``.
        """
        self._call_hooks(TrainingStageEnum.BEFORE_VALIDATION, ctx)
        self.model.eval()

        total_loss = 0.0
        n_batches = 0

        with torch.no_grad():
            for batch in self.val_loader:
                outputs = self.model(batch)
                loss, _ = self.loss(batch, outputs)
                total_loss += loss.item()
                n_batches += 1
                self.model.zero_grad(set_to_none=True)

        avg_loss = total_loss / max(n_batches, 1)
        metrics = {"val_loss": avg_loss}
        ctx.metrics = metrics

        self._call_hooks(TrainingStageEnum.AFTER_VALIDATION, ctx)
        return metrics

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def _save_checkpoint(self, ctx: TrainingContext) -> None:
        """Save a training checkpoint.

        Parameters
        ----------
        ctx : TrainingContext
            Current training context.
        """
        self._call_hooks(TrainingStageEnum.BEFORE_CHECKPOINT, ctx)
        path = self.config.checkpoint_dir / f"epoch_{ctx.epoch}.pt"
        save_training_checkpoint(
            path=path,
            model=self.model,
            optimizer=self.optimizers,
            scheduler=self.schedulers or None,
            scaler=self.scaler,
            epoch=ctx.epoch,
            metrics=ctx.metrics,
            dist_manager=self.dist_manager,
        )
        self._call_hooks(TrainingStageEnum.AFTER_CHECKPOINT, ctx)

    def _maybe_resume(self) -> None:
        """Resume training from a checkpoint if configured."""
        resume_path = self.config.resume_from
        if resume_path is None:
            return

        device = next(self.model.parameters()).device
        state = load_training_checkpoint(
            path=resume_path,
            model=self.model,
            optimizer=self.optimizers,
            scheduler=self.schedulers or None,
            scaler=self.scaler,
            device=device,
        )
        self._start_epoch = state.get("epoch", 0)
        logger.info("Resumed training from epoch {}.", self._start_epoch)
