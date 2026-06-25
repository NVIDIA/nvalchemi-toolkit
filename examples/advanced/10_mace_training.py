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
"""Train a MACE model with ALCHEMI training utilities.

The example expects dataset metadata such as ``avg_num_neighbors`` and ``E0s``
to be precomputed and loaded into ``cfg.model``. Training
runs for a fixed optimizer-step budget via ``training.steps``, uses
:class:`~examples.advanced._mace_training_helpers.TwoStageCosineConstantLR`
for the learning rate, and step-function loss weights from
:func:`~examples.advanced._mace_training_helpers.build_mace_step_huber_loss`.
Stage-two ``training.loss.stage_two`` can override energy, force, and stress weights.
Validation and checkpointing run on step cadences.

``training.batch_size`` is per process; effective global batch size scales with
``nproc_per_node``.

Single GPU:

.. code-block:: bash

   uv run torchrun --standalone --nproc_per_node=1 \
       examples/advanced/10_mace_training.py

Multi-GPU:

.. code-block:: bash

   uv run torchrun --standalone --nproc_per_node=4 \
       examples/advanced/10_mace_training.py
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
    TwoStageCosineConstantLR,
    _dtype,
    build_mace_step_huber_loss,
    count_model_parameters,
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
    """Build step-cadence validation for fixed-step MACE training."""
    del model
    if not bool(cfg.training.validation.get("enabled", True)):
        return None, None
    if cfg.training.validation.get("every_epochs", None) is not None:
        raise ValueError(
            "Training is step-based; set "
            "training.validation.every_epochs=null and use every_steps."
        )
    every_steps = cfg.training.validation.get("every_steps", None)
    if every_steps is None:
        raise ValueError("Set training.validation.every_steps for step-based training.")
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
        every_n_steps=int(every_steps),
        grad_mode="auto",
        use_ema="auto",
        name="validation",
    )


def _optimizer(cfg: DictConfig) -> OptimizerConfig:
    """Build AdamW with an optional two-stage cosine LR schedule."""
    if cfg.training.get("epochs", None) is not None:
        raise ValueError("Set training.epochs=null and training.steps to an integer.")
    if cfg.training.get("steps", None) is None:
        raise ValueError("Set training.steps to the desired optimizer-step count.")

    total_steps = int(cfg.training.steps)
    scheduler_cfg = cfg.training.get("scheduler", {})
    scheduler_cls: type | None = None
    scheduler_kwargs: dict[str, Any] = {}
    if bool(scheduler_cfg.get("enabled", False)):
        scheduler_type = str(scheduler_cfg.get("type", "cosine"))
        if scheduler_type != "cosine":
            raise ValueError(
                f"Unsupported scheduler type {scheduler_type!r}; "
                "Supported scheduler type is 'cosine'."
            )
        stage_two_cfg = cfg.training.loss.get("stage_two", {})
        configured_first_stage = scheduler_cfg.get("first_stage_steps", None)
        first_stage_steps = int(
            stage_two_cfg.get("start_step", total_steps)
            if configured_first_stage is None
            else configured_first_stage
        )
        if first_stage_steps <= 0:
            raise ValueError("training.scheduler.first_stage_steps must be positive.")
        if first_stage_steps >= total_steps:
            raise ValueError(
                "The cosine-then-constant LR schedule requires stage one to end "
                "before training.steps. Set training.loss.stage_two.start_step or "
                "training.scheduler.first_stage_steps below training.steps."
            )
        scheduler_cls = TwoStageCosineConstantLR
        scheduler_kwargs = {
            "first_stage_steps": first_stage_steps,
            "second_stage_lr": float(cfg.training.optimizer.stage_two_lr),
            "eta_min": float(
                scheduler_cfg.get("eta_min", cfg.training.optimizer.stage_two_lr)
            ),
        }

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
    """Build hooks while requiring checkpoint cadence to be step-based."""
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
    logging_cfg = cfg.training.get(
        "logging",
        cfg.training.get("tracking", cfg.training.get("metrics", {})),
    )
    jsonl_path = logging_cfg.get("jsonl_path", None)
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
                logger_axis=str(logging_cfg.get("logger_axis", "step")),
            ),
        ]
    )
    checkpoint_cfg = cfg.training.checkpoint
    if bool(checkpoint_cfg.get("enabled", False)):
        if checkpoint_cfg.get("epoch_interval", None) is not None:
            raise ValueError(
                "10_mace_training is step-based; set "
                "training.checkpoint.epoch_interval=null and use step_interval."
            )
        if checkpoint_cfg.get("step_interval", None) is None:
            raise ValueError("Set training.checkpoint.step_interval.")
        hook_kwargs: dict[str, Any] = {"checkpoint_dir": Path(checkpoint_cfg.dir)}
        hook_kwargs["step_interval"] = int(checkpoint_cfg.step_interval)
        hooks.append(CheckpointHook(**hook_kwargs))
    return hooks


def _build_model(cfg: DictConfig, device: torch.device) -> torch.nn.Module:
    atomic_numbers, atomic_energies = _e0s(cfg.model)
    active_outputs = {"energy"}
    if float(cfg.training.loss.force_weight) != 0.0:
        active_outputs.add("forces")
    if float(cfg.training.loss.get("stress_weight", 0.0)) != 0.0:
        active_outputs.add("stress")
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


def _save_final_checkpoint(
    cfg: DictConfig,
    strategy: TrainingStrategy,
    manager: DistributedManager,
) -> None:
    """Save a final checkpoint on rank zero when configured."""
    checkpoint_cfg = cfg.training.checkpoint
    if not bool(checkpoint_cfg.get("save_final", False)):
        return
    if get_rank(manager) != 0:
        return
    step_interval = checkpoint_cfg.get("step_interval", None)
    if (
        step_interval is not None
        and strategy.step_count > 0
        and strategy.step_count % int(step_interval) == 0
    ):
        return
    strategy.save_checkpoint(checkpoint_cfg.dir)


@hydra.main(version_base=None, config_path=".", config_name="10_vanilla_mace")
def main(cfg: DictConfig) -> None:
    """Run MACE training."""

    # Initialize the distributed manager
    DistributedManager.initialize()
    manager = DistributedManager()
    if get_rank(manager) == 0:
        print(OmegaConf.to_yaml(cfg, resolve=True), flush=True)
    device = torch.device(manager.device)
    torch.manual_seed(int(cfg.training.seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(cfg.training.seed))

    # Build the model
    model = _build_model(cfg, device)
    if get_rank(manager) == 0:
        print(f"Model parameters: {count_model_parameters(model):,}")

    # Build the dataloaders
    skip_validation = bool(cfg.training.dataloader.get("skip_validation", True))
    validation_loader, validation_config = _build_mace_validation_config(
        cfg,
        model,
        manager,
        device,
        loss_fn,
    )
    train_loader = _loader(
        str(cfg.data.zarr_path),
        cfg,
        device,
        skip_validation=skip_validation,
    )

    # Build the training strategy
    loss_fn = build_mace_step_huber_loss(cfg.training.loss)
    hooks = _hooks(cfg, model)
    strategy = TrainingStrategy(
        models=model,
        optimizer_configs=_optimizer(cfg),
        num_epochs=None,
        num_steps=int(cfg.training.steps),
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
        strategy.num_steps = int(cfg.training.steps)

    # Run the training loop
    try:
        strategy.run(train_loader)
        _save_final_checkpoint(cfg, strategy, manager)
    finally:
        train_loader.dataset.reader.close()
        if validation_loader is not None:
            validation_loader.dataset.reader.close()
        DistributedManager.cleanup()


if __name__ == "__main__":
    main()
