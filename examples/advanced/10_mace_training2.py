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
"""Train a MACE model with current ALCHEMI training utilities.

The example expects dataset metadata such as ``avg_num_neighbors`` and ``E0s``
to be precomputed and loaded into ``cfg.model`` by Hydra defaults. It uses the
toolkit data pipes, loss composition, DDP hook, and periodic checkpoint hook
directly instead of carrying recipe-local training helpers.

Single GPU:

.. code-block:: bash

   uv run torchrun --standalone --nproc_per_node=1 \
       examples/advanced/10_mace_training2.py

Multi-GPU:

.. code-block:: bash

   uv run torchrun --standalone --nproc_per_node=4 \
       examples/advanced/10_mace_training2.py

``training.batch_size`` is per process; effective global batch size scales with
``nproc_per_node``.
"""

from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data.distributed import DistributedSampler

from examples.advanced._mace_models import build_training_mace_model
from examples.advanced._mace_training_helpers import (
    GradientClipHook,
    JsonLinesLogger,
    ScaleField,
    ToDType,
    TrainingMetricsLogger,
    _dtype,
    build_mace_huber_loss,
    count_model_parameters,
    mark_charge_target_as_node,
    stress_target_scale,
)
from nvalchemi.data.datapipes import AtomicDataZarrReader, DataLoader, Dataset
from nvalchemi.distributed import DistributedManager
from nvalchemi.hooks import NeighborListHook
from nvalchemi.training import (
    CheckpointHook,
    DDPHook,
    EMAHook,
    OptimizerConfig,
    TrainingStage,
    TrainingStrategy,
    ValidationConfig,
    default_training_fn,
)
from nvalchemi.training.distributed import get_rank, get_world_size

_DATALOADER_NUM_STREAMS = 2
_DATALOADER_DEFAULT_PREFETCH_FACTOR = 2
_FULL_SHUFFLE_READ_WINDOW = 1024
_DATALOADER_USE_STREAMS = True


class PlateauThenConstantLR(torch.optim.lr_scheduler.ReduceLROnPlateau):
    """Reduce LR on validation plateaus, then hold a constant stage-two LR.

    Attributes
    ----------
    constant_start_epoch : int
        Epoch at which plateau updates stop and the constant LR begins.
    constant_lr : float
        Learning rate used after ``constant_start_epoch``.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        *,
        constant_start_epoch: int,
        constant_lr: float,
        mode: str = "min",
        factor: float = 0.8,
        patience: int = 50,
        threshold: float = 1e-4,
        threshold_mode: str = "rel",
        cooldown: int = 0,
        min_lr: float | list[float] = 0.0,
        eps: float = 1e-8,
    ) -> None:
        """Configure the plateau-to-constant learning-rate schedule."""
        if constant_start_epoch <= 0:
            raise ValueError("constant_start_epoch must be positive.")
        if constant_lr < 0.0:
            raise ValueError("constant_lr must be non-negative.")
        self.constant_start_epoch = int(constant_start_epoch)
        self.constant_lr = float(constant_lr)
        super().__init__(
            optimizer,
            mode=mode,
            factor=factor,
            patience=patience,
            threshold=threshold,
            threshold_mode=threshold_mode,
            cooldown=cooldown,
            min_lr=min_lr,
            eps=eps,
        )

    def step(self, metrics: float, epoch: int | None = None) -> None:
        """Step the plateau scheduler until Stage Two, then enforce constant LR."""
        next_epoch = self.last_epoch + 1 if epoch is None else epoch
        if next_epoch < self.constant_start_epoch:
            super().step(metrics, epoch=epoch)
            return
        self.last_epoch = int(next_epoch)
        for group in self.optimizer.param_groups:
            group["lr"] = self.constant_lr
        self._last_lr = [self.constant_lr for _ in self.optimizer.param_groups]


def _e0s(model_cfg: DictConfig) -> tuple[list[int], np.ndarray]:
    e0s = OmegaConf.to_container(model_cfg.E0s, resolve=True)
    if not isinstance(e0s, dict):
        raise ValueError("cfg.model.E0s must be a mapping from atomic number to E0.")
    e0_by_z = {int(z): float(e0) for z, e0 in e0s.items()}
    atomic_numbers = sorted(e0_by_z)
    atomic_energies = np.asarray([e0_by_z[z] for z in atomic_numbers])
    return atomic_numbers, atomic_energies


def _loader(
    path: str,
    cfg: DictConfig,
    device: torch.device,
    *,
    batch_size: int | None = None,
    shuffle: bool = True,
    skip_validation: bool = True,
) -> DataLoader:
    dtype = _dtype(cfg.model.get("dtype", "float32"))
    batch_transforms = [ToDType(dtype)]
    if float(cfg.training.loss.get("stress_weight", 0.0)) != 0.0:
        batch_transforms.insert(
            0,
            ScaleField(
                "stress",
                stress_target_scale(cfg.data),
                missing_ok=False,
            ),
        )
    if float(cfg.training.loss.get("charge_weight", 0.0)) != 0.0:
        batch_transforms.append(
            partial(
                mark_charge_target_as_node,
                charge_target_key=str(cfg.data.charge_target_key),
            )
        )
    dataset = Dataset(
        AtomicDataZarrReader(path),
        device=device,
        skip_validation=skip_validation,
    )
    loader_cfg = cfg.training.dataloader
    resolved_batch_size = int(
        cfg.training.batch_size if batch_size is None else batch_size
    )
    prefetch_factor = _DATALOADER_DEFAULT_PREFETCH_FACTOR
    if shuffle:
        prefetch_factor = max(
            _FULL_SHUFFLE_READ_WINDOW // resolved_batch_size,
            _DATALOADER_DEFAULT_PREFETCH_FACTOR,
        )
    return DataLoader(
        dataset,
        batch_size=resolved_batch_size,
        shuffle=shuffle,
        drop_last=bool(loader_cfg.get("drop_last", False)),
        prefetch_factor=prefetch_factor,
        num_streams=_DATALOADER_NUM_STREAMS,
        use_streams=_DATALOADER_USE_STREAMS,
        batch_transforms=batch_transforms,
    )


def _make_validation_sampler(
    dataset: Dataset,
    manager: DistributedManager,
) -> DistributedSampler | None:
    """Return the distributed validation sampler for multi-rank validation."""
    if get_world_size(manager) <= 1:
        return None
    return DistributedSampler(
        dataset,
        num_replicas=get_world_size(manager),
        rank=get_rank(manager),
        shuffle=False,
        drop_last=True,
    )


def _build_mace_validation_config(
    cfg: DictConfig,
    model: torch.nn.Module,
    manager: DistributedManager,
    device: torch.device,
    loss_fn: Any,
) -> tuple[DataLoader | None, ValidationConfig | None]:
    """Build validation loader and config when validation is enabled."""
    del model
    if not bool(cfg.training.validation.get("enabled", True)):
        return None, None
    validation_loader = _loader(
        str(cfg.data.validation_zarr_path),
        cfg,
        device,
        batch_size=int(cfg.training.validation.batch_size),
        shuffle=False,
        skip_validation=bool(cfg.training.dataloader.get("skip_validation", True)),
    )
    validation_sampler = _make_validation_sampler(validation_loader.dataset, manager)
    if validation_sampler is not None:
        validation_loader.sampler = validation_sampler
    return validation_loader, ValidationConfig(
        validation_data=validation_loader,
        validation_fn=default_training_fn,
        loss_fn=loss_fn,
        every_n_epochs=int(cfg.training.validation.get("every_epochs", 1)),
        grad_mode="auto",
        use_ema="auto",
        name="validation",
    )


def _optimizer(cfg: DictConfig) -> OptimizerConfig:
    scheduler_cfg = cfg.training.get("scheduler", {})
    scheduler_cls: type | None = None
    scheduler_kwargs: dict[str, Any] = {}
    if bool(scheduler_cfg.get("enabled", False)):
        scheduler_type = str(scheduler_cfg.get("type", "plateau_then_constant"))
        if scheduler_type not in {"plateau", "plateau_then_constant"}:
            raise ValueError(
                f"Unsupported scheduler type {scheduler_type!r}; "
                "the MACE example supports 'plateau' and 'plateau_then_constant'."
            )
        scheduler_cls = torch.optim.lr_scheduler.ReduceLROnPlateau
        scheduler_kwargs = {
            "factor": float(scheduler_cfg.get("factor", 0.8)),
            "patience": int(scheduler_cfg.get("patience", 50)),
        }
        if scheduler_type == "plateau_then_constant":
            stage_two_cfg = cfg.training.loss.get("stage_two", {})
            configured_start_epoch = scheduler_cfg.get("constant_start_epoch", None)
            constant_start_epoch = int(
                stage_two_cfg.get("start_epoch", cfg.training.epochs)
                if configured_start_epoch is None
                else configured_start_epoch
            )
            if constant_start_epoch >= int(cfg.training.epochs):
                raise ValueError(
                    "The plateau-then-constant LR schedule requires Stage Two "
                    "to start before training.epochs. Set "
                    "training.loss.stage_two.start_epoch or "
                    "training.scheduler.constant_start_epoch below "
                    "training.epochs."
                )
            scheduler_cls = PlateauThenConstantLR
            scheduler_kwargs.update(
                {
                    "constant_start_epoch": constant_start_epoch,
                    "constant_lr": float(cfg.training.optimizer.stage_two_lr),
                }
            )
        if "min_lr" in scheduler_cfg:
            scheduler_kwargs["min_lr"] = float(scheduler_cfg.min_lr)
        if "threshold" in scheduler_cfg:
            scheduler_kwargs["threshold"] = float(scheduler_cfg.threshold)
    if cfg.training.epochs is None:
        raise ValueError("The MACE example is epoch-based; set training.epochs.")
    if cfg.training.get("steps", None) is not None:
        raise ValueError(
            "The MACE example now uses epochs. Set training.steps=null and "
            "training.epochs to the desired epoch count."
        )
    if (
        bool(scheduler_cfg.get("enabled", False))
        and cfg.training.validation.get("every_epochs", None) is None
    ):
        raise ValueError(
            "ReduceLROnPlateau requires validation metrics; set "
            "training.validation.every_epochs."
        )
    if bool(scheduler_cfg.get("enabled", False)) and not bool(
        cfg.training.validation.get("enabled", True)
    ):
        raise ValueError("ReduceLROnPlateau requires training.validation.enabled=true.")
    return OptimizerConfig(
        optimizer_cls=torch.optim.AdamW,
        optimizer_kwargs={
            "lr": float(cfg.training.optimizer.lr),
            "weight_decay": float(cfg.training.optimizer.get("weight_decay", 5e-7)),
        },
        scheduler_cls=scheduler_cls,
        scheduler_kwargs=scheduler_kwargs,
    )


def _hooks(
    cfg: DictConfig,
    model: torch.nn.Module,
) -> list[Any]:
    hooks: list[Any] = []
    distributed_cfg = cfg.training.get("distributed", {})
    if bool(distributed_cfg.get("enabled", False)):
        backend = str(distributed_cfg.get("backend", "nccl"))
        hooks.append(
            DDPHook(backend=backend, sampler_kwargs={"seed": int(cfg.training.seed)})
        )
    ema_cfg = cfg.training.get("ema", {})
    if bool(ema_cfg.get("enabled", True)):
        hooks.append(
            EMAHook(
                model_key="main",
                decay=float(ema_cfg.get("decay", 0.999)),
                update_every=int(ema_cfg.get("update_every", 1)),
                start_step=int(ema_cfg.get("start_step", 0)),
            )
        )
    clip_grad = cfg.training.optimizer.get("clip_grad", 100.0)
    if clip_grad is not None and float(clip_grad) > 0.0:
        hooks.append(GradientClipHook(max_norm=float(clip_grad)))
    metrics_cfg = cfg.training.get("metrics", {})
    jsonl_path = metrics_cfg.get("jsonl_path", None)
    metrics_logger = JsonLinesLogger(jsonl_path) if jsonl_path is not None else None
    hooks.extend(
        [
            NeighborListHook(
                model.model_config.neighbor_config,
                max_neighbors=int(cfg.training.get("max_neighbors", 256)),
                stage=TrainingStage.BEFORE_FORWARD,
            ),
            TrainingMetricsLogger(
                every=int(cfg.training.log_every_steps),
                logger=metrics_logger,
                logger_axis=str(metrics_cfg.get("logger_axis", "epoch")),
            ),
        ]
    )
    checkpoint_cfg = cfg.training.checkpoint
    if bool(checkpoint_cfg.get("enabled", False)):
        hook_kwargs: dict[str, Any] = {"checkpoint_dir": Path(checkpoint_cfg.dir)}
        if checkpoint_cfg.get("step_interval") is not None:
            hook_kwargs["step_interval"] = int(checkpoint_cfg.step_interval)
        else:
            hook_kwargs["epoch_interval"] = int(checkpoint_cfg.get("epoch_interval", 1))
        hooks.append(CheckpointHook(**hook_kwargs))
    return hooks


def _validate_charge_correction_config(cfg: DictConfig) -> None:
    """Validate optional short-range Coulomb correction settings."""
    correction_cfg = cfg.model.get("charge_correction", {})
    if not bool(correction_cfg.get("enabled", False)):
        return
    if str(cfg.model.get("model_type", "mace")) != "charged_mace":
        raise ValueError(
            "model.charge_correction.enabled requires "
            "model.model_type='charged_mace'."
        )
    outer_radius = float(correction_cfg.get("outer_radius", cfg.model.r_max))
    if outer_radius <= 0.0:
        raise ValueError(
            "model.charge_correction.outer_radius must be positive Angstrom."
        )
    inner_radius = float(correction_cfg.get("inner_radius", 0.0))
    if inner_radius < 0.0:
        raise ValueError(
            "model.charge_correction.inner_radius must be non-negative Angstrom."
        )
    if inner_radius >= outer_radius:
        raise ValueError(
            "model.charge_correction.inner_radius must be smaller than "
            "outer_radius."
        )
    lambda_sub = float(correction_cfg.get("lambda_sub", 1.0))
    if lambda_sub < 0.0:
        raise ValueError("model.charge_correction.lambda_sub must be non-negative.")


def _build_model(cfg: DictConfig, device: torch.device) -> torch.nn.Module:
    _validate_charge_correction_config(cfg)
    atomic_numbers, atomic_energies = _e0s(cfg.model)
    active_outputs = {"energy"}
    if float(cfg.training.loss.force_weight) != 0.0:
        active_outputs.add("forces")
    if float(cfg.training.loss.get("stress_weight", 0.0)) != 0.0:
        active_outputs.add("stress")
    if float(cfg.training.loss.get("charge_weight", 0.0)) != 0.0:
        if str(cfg.model.get("model_type", "mace")) != "charged_mace":
            raise ValueError(
                "training.loss.charge_weight requires model.model_type='charged_mace'."
            )
        active_outputs.add("charges")
    return build_training_mace_model(
        model_type=str(cfg.model.get("model_type", "mace")),
        atomic_numbers=atomic_numbers,
        atomic_energies=atomic_energies.tolist(),
        r_max=float(cfg.model.r_max),
        avg_num_neighbors=float(cfg.model.avg_num_neighbors),
        model_config=OmegaConf.to_container(cfg.model, resolve=True),
        dtype=_dtype(cfg.model.get("dtype", "float32")),
        device=device,
        active_outputs=sorted(active_outputs),
    )


@hydra.main(version_base=None, config_path=".", config_name="10_mace_training2")
def main(cfg: DictConfig) -> None:
    """Run MACE training."""
    DistributedManager.initialize()
    manager = DistributedManager()
    if get_rank(manager) == 0:
        print(OmegaConf.to_yaml(cfg, resolve=True), flush=True)
    device = torch.device(manager.device)
    torch.manual_seed(int(cfg.training.seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(cfg.training.seed))

    model = _build_model(cfg, device)
    if get_rank(manager) == 0:
        print(f"Model parameters: {count_model_parameters(model):,}")
    loss_fn = build_mace_huber_loss(cfg.training.loss, cfg)
    skip_validation = bool(cfg.training.dataloader.get("skip_validation", True))
    validation_loader, validation_config = _build_mace_validation_config(
        cfg,
        model,
        manager,
        device,
        loss_fn,
    )
    hooks = _hooks(cfg, model)
    strategy = TrainingStrategy(
        models=model,
        optimizer_configs=_optimizer(cfg),
        num_epochs=cfg.training.epochs,
        num_steps=None,
        training_fn=default_training_fn,
        loss_fn=loss_fn,
        devices=[device],
        distributed_manager=manager,
        hooks=hooks,
        validation_config=validation_config,
    )
    if bool(cfg.training.restart.get("enabled", False)):
        strategy.restore_checkpoint(
            cfg.training.restart.get("dir", cfg.training.checkpoint.dir),
            map_location=device,
        )
        strategy.num_epochs = cfg.training.epochs

    train_loader = _loader(
        str(cfg.data.zarr_path),
        cfg,
        device,
        skip_validation=skip_validation,
    )
    try:
        strategy.run(train_loader)
    finally:
        train_loader.dataset.reader.close()
        if validation_loader is not None:
            validation_loader.dataset.reader.close()
        DistributedManager.cleanup()


if __name__ == "__main__":
    main()
