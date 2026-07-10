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
"""
MACE Training with ALCHEMI Training Utilities
=============================================

This example walks through a complete model-training lifecycle on the ALCHEMI
Toolkit, using a baseline ScaleShiftMACE model trained on the MatPES r2SCAN
dataset as the reference workflow. The training script run configuration is loaded
with `Hydra <https://hydra.cc/>`__ from :download:`10_vanilla_mace.yaml <10_vanilla_mace.yaml>`.

At a high level, the ALCHEMI training workflow has the following structure:

.. code-block:: text

   [Graph Data] -> [Model Architecture] -> [Supervised Objective] -> [Runtime Hooks] -> [TrainingStrategy]

**Data** — MatPES r2SCAN structures are read from ALCHEMI-compatible Zarr splits.
Each sample contains graph inputs (positions, atom types, periodic boundary
metadata) and supervised labels (energy, forces, stress).

**Model** — A 9.06M-parameter ScaleShiftMACE model from
`ACEsuit <https://github.com/acesuit/mace>`__ is wrapped with
:class:`~nvalchemi.models.mace.MACEWrapper` so it can be used by
:class:`~nvalchemi.training.TrainingStrategy`. NVIDIA cuEquivariance kernels are
enabled by default in the Hydra config (`model.cueq.enabled: true`).

**Loss** — Energies, forces, and stresses are fit with a weighted sum of Huber
losses. :class:`~nvalchemi.training.PiecewiseWeight` schedules are used to change
the loss-term weights at a configured optimizer step for the second training stage.

**Runtime** — Distributed wrapping, EMA, neighbor-list rebuild, gradient
clipping, metrics logging, and checkpointing are attached through runtime hooks
rather than being implemented directly in the core trainin loop. Validation
is configured separately using :class:`~nvalchemi.training.ValidationConfig` on
:class:`~nvalchemi.training.TrainingStrategy`. Validation runs automatically
during :meth:`~nvalchemi.training.TrainingStrategy.run`.

Dataset-derived metadata (`E0s`, `avg_num_neighbors`, `atomic_inter_shift` /
`atomic_inter_scale`), must be precomputed and set in ``cfg.model`` before
training. The default YAML includes values computed from the MatPES r2SCAN training
split.

"""

# sphinx_gallery_start_ignore
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from examples.advanced._mace_models import build_training_mace_model, get_e0s
from examples.advanced._mace_training_helpers import (
    GradientClipHook,
    JsonLinesLogger,
    ScaleField,
    ToDType,
    TrainingMetricsLogger,
    TwoStageCosineConstantLR,
    close_zarr_loaders,
    count_model_parameters,
    get_cfg,
    get_dtype,
    make_validation_sampler,
    save_final_checkpoint,
    stress_target_scale,
)
from nvalchemi.data.datapipes import (
    AtomicDataZarrReader,
    DataLoader,
    InMemoryDataset,
)
from nvalchemi.distributed import DistributedManager
from nvalchemi.hooks import Hook, NeighborListHook
from nvalchemi.models.mace import MACEWrapper
from nvalchemi.training import (
    CheckpointHook,
    ComposedLossFunction,
    DDPHook,
    EMAHook,
    EnergyHuberLoss,
    ForceHuberLoss,
    OptimizerConfig,
    PiecewiseWeight,
    StressHuberLoss,
    TrainingStage,
    TrainingStrategy,
    ValidationConfig,
    default_training_fn,
)
from nvalchemi.training.hooks import TrainingUpdateHook

_DOCS_BUILD = os.environ.get("NVALCHEMI_SPHINX_BUILD") == "1"
# Samples per Zarr read during startup materialization; 32768 is used instead of
# the default 4096 for fewer read calls.
_DATASET_CHUNK_SIZE = 32768
# CUDA prefetch settings for host-to-device transfer in dataloader.
_DATALOADER_NUM_STREAMS = 2
_DATALOADER_PREFETCH_FACTOR = 2
_DATALOADER_USE_STREAMS = True
# sphinx_gallery_end_ignore

# %%
# Loading train and validation data
# ---------------------------------
# This pipeline reads MatPES r2SCAN structures from ALCHEMI-compatible Zarr
# splits. :class:`~nvalchemi.data.datapipes.AtomicDataZarrReader` streams raw
# samples from disk; :class:`~nvalchemi.data.datapipes.InMemoryDataset`
# materializes each split once as a :class:`~nvalchemi.data.Batch` on the target
# device, and :class:`~nvalchemi.data.datapipes.DataLoader` selects shuffled or
# sequential training batches from that in-memory batch.
#
# The default configuration uses a per-process training batch size of 256 and a
# validation batch size of 512. Given that the structure sizes in this dataset range from
# 1 atom to 240 atoms, :class:`~nvalchemi.dynamics.sampler.SizeAwareSampler` can also be
# used as an alternative to cap the atom count per batch when memory is tight.
#
# .. code-block:: python
#
#    from pathlib import Path
#
#    import torch
#    from nvalchemi.data.datapipes import AtomicDataZarrReader, DataLoader, InMemoryDataset
#
#    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
#    train_dataset = InMemoryDataset(
#        reader=AtomicDataZarrReader(Path("/path/to/r2scan-2025.2-train.zarr")),
#        device=device,
#        skip_validation=True,
#    )
#
#    train_batches = DataLoader(
#        train_dataset,
#        batch_size=256,
#        shuffle=True,
#    )
#
#    val_dataset = InMemoryDataset(
#        reader=AtomicDataZarrReader(Path("/path/to/r2scan-2025.2-valid.zarr")),
#        device=device,
#        skip_validation=True,
#    )
#
#    val_batches = DataLoader(
#        val_dataset,
#        batch_size=512,
#        shuffle=False,
#    )
#
# The runnable script wraps this pattern in ``_loader(...)`` so Hydra can supply
# paths, batch sizes, and optional stress scaling transforms.

# sphinx_gallery_start_ignore


def _loader(
    path: str,
    cfg: DictConfig,
    *,
    device: torch.device | None = None,
    batch_size: int | None = None,
    shuffle: bool = True,
) -> DataLoader:
    """Create an ALCHEMI InMemoryDataset/DataLoader from a Zarr path."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = get_dtype(cfg.model.get("dtype", "float32"))
    batch_transforms = [ToDType(dtype)]
    if float(cfg.training.loss.get("stress_weight", 0.0)) != 0.0:
        # Add scaling transform for stress field
        batch_transforms.insert(
            0,
            ScaleField(
                "stress",
                stress_target_scale(cfg.data),
                missing_ok=False,
            ),
        )
    loader_cfg = cfg.training.dataloader
    dataset = InMemoryDataset(
        reader=AtomicDataZarrReader(path),
        device=device,
        chunk_size=_DATASET_CHUNK_SIZE,
        skip_validation=True,
        batch_transforms=batch_transforms,
    )
    resolved_batch_size = int(
        cfg.training.batch_size if batch_size is None else batch_size
    )
    return DataLoader(
        dataset,
        batch_size=resolved_batch_size,
        shuffle=shuffle,
        drop_last=bool(loader_cfg.get("drop_last", False)),
        prefetch_factor=_DATALOADER_PREFETCH_FACTOR,
        num_streams=_DATALOADER_NUM_STREAMS,
        use_streams=_DATALOADER_USE_STREAMS,
        pin_memory=True,
    )


# sphinx_gallery_end_ignore

# %%
# Building the MACE model
# -----------------------
# The default configuration trains ScaleShiftMACE to predict energy, force, and
# stress. Any model object passed to :class:`~nvalchemi.training.TrainingStrategy`
# must follow :class:`~nvalchemi.models.base.BaseModelMixin`.
# :class:`~nvalchemi.models.mace.MACEWrapper` handles input adaptation,
# neighbor-list metadata, and routes model outputs for MACE model variants.
#
# Before building the model, populate the Hydra config with dataset-derived
# metadata: ``E0s`` (from structure-energy regression or isolated-atom DFT),
# ``avg_num_neighbors``, and the ScaleShiftMACE pair ``atomic_inter_shift`` /
# ``atomic_inter_scale``. The default YAML includes values precomputed from the
# training split.
#
# .. code-block:: python
#
#    import torch
#    from mace.modules import ScaleShiftMACE
#
#    from nvalchemi.models.mace import MACEWrapper
#
#    mace_model = ScaleShiftMACE(...)
#    model = MACEWrapper(mace_model.to(device=device, dtype=torch.float32))
#    model.model_config.active_outputs = {"energy", "forces", "stress"}
#
# The runnable script reads architecture hyperparameters from Hydra and builds
# the wrapped model through ``_build_model(cfg, device)``, which calls
# :func:`examples.advanced._mace_models.build_training_mace_model` to set
# ``active_outputs`` and attach a checkpointable model spec.

# sphinx_gallery_start_ignore


def _build_model(cfg: DictConfig, device: torch.device) -> torch.nn.Module:
    """Build ScaleShiftMACE wrapped in MACEWrapper."""
    atomic_numbers, atomic_energies = get_e0s(cfg.model)

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
        dtype=get_dtype(cfg.model.get("dtype", "float32")),
        device=device,
        active_outputs=sorted(active_outputs),
    )


# sphinx_gallery_end_ignore

# %%
# Defining the loss
# -----------------
# The default configuration fits energies, forces, and stresses. The loss is a
# weighted sum of Huber terms composed with ``+`` and ``*`` into a
# :class:`~nvalchemi.training.ComposedLossFunction`. Stage-one weights hold until
# ``stage_two_start``, then switch (for example ``1/10/100`` to ``10/1/10`` at
# step 54,400 of 68,000).
#
# .. code-block:: python
#
#    from nvalchemi.training import (
#        ComposedLossFunction,
#        EnergyHuberLoss,
#        ForceHuberLoss,
#        PiecewiseWeight,
#        StressHuberLoss,
#    )
#
#    stage_two_start = 54_400
#
#    loss_fn: ComposedLossFunction = (
#        PiecewiseWeight(
#            boundaries=(stage_two_start,),
#            values=(1.0, 10.0),
#            per_epoch=False,
#        )
#        * EnergyHuberLoss(per_atom=True, delta=0.01)
#        + PiecewiseWeight(
#            boundaries=(stage_two_start,),
#            values=(10.0, 1.0),
#            per_epoch=False,
#        )
#        * ForceHuberLoss(normalize_by_atom_count=False, delta=0.01)
#        + PiecewiseWeight(
#            boundaries=(stage_two_start,),
#            values=(100.0, 10.0),
#            per_epoch=False,
#        )
#        * StressHuberLoss(delta=0.01)
#    )
#
#    loss_fn.normalize_weights = False
#
# The runnable script builds the same composition from ``cfg.training.loss``
# through ``_build_mace_huber_loss(cfg.training.loss)``.

# sphinx_gallery_start_ignore


def _build_mace_huber_loss(loss_cfg: Any) -> ComposedLossFunction:
    """Build a step-scheduled MACE Huber objective.

    Stage-one weights are held constant until ``loss.stage_two.start_step``.
    At that step, weights switch instantly to their configured stage-two values.

    Parameters
    ----------
    loss_cfg : Any
        ``cfg.training.loss`` node with energy, force, and stress weights.

    Returns
    -------
    ComposedLossFunction
        Weighted sum of Huber losses with step-based weight schedules.
    """
    delta = float(get_cfg(loss_cfg, "huber_delta", 0.01))
    stage_two = get_cfg(loss_cfg, "stage_two", {})
    stage_two_start = int(get_cfg(stage_two, "start_step"))
    boundaries = (stage_two_start,)

    energy_weight = float(get_cfg(loss_cfg, "energy_weight"))
    force_weight = float(get_cfg(loss_cfg, "force_weight"))
    stress_weight = float(get_cfg(loss_cfg, "stress_weight", 0.0))

    loss_fn: ComposedLossFunction = PiecewiseWeight(
        boundaries=boundaries,
        values=(
            energy_weight,
            float(get_cfg(stage_two, "energy_weight", energy_weight)),
        ),
        per_epoch=False,
    ) * EnergyHuberLoss(
        per_atom=True,
        delta=delta,
        ignore_nonfinite=True,
    )
    if force_weight != 0.0:
        loss_fn = loss_fn + (
            PiecewiseWeight(
                boundaries=boundaries,
                values=(
                    force_weight,
                    float(get_cfg(stage_two, "force_weight", force_weight)),
                ),
                per_epoch=False,
            )
            * ForceHuberLoss(
                normalize_by_atom_count=False,
                delta=delta,
                ignore_nonfinite=True,
            )
        )
    if stress_weight != 0.0:
        loss_fn = loss_fn + (
            PiecewiseWeight(
                boundaries=boundaries,
                values=(
                    stress_weight,
                    float(get_cfg(stage_two, "stress_weight", stress_weight)),
                ),
                per_epoch=False,
            )
            * StressHuberLoss(
                delta=delta,
                ignore_nonfinite=True,
            )
        )
    loss_fn.normalize_weights = False
    return loss_fn


# sphinx_gallery_end_ignore

# %%
# Configuring the optimizer and scheduler
# ---------------------------------------
# Schedulers are attached through :class:`~nvalchemi.training.OptimizerConfig`. The
# runnable example uses
# :class:`~examples.advanced._mace_training_helpers.TwoStageCosineConstantLR` —
# cosine annealing for stage one, then a constant stage-two learning rate; any
# ``torch.optim.lr_scheduler.LRScheduler`` subclass can be passed via
# ``scheduler_cls`` and ``scheduler_kwargs``.
#
# .. code-block:: python
#
#    import torch
#
#    from examples.advanced._mace_training_helpers import TwoStageCosineConstantLR
#    from nvalchemi.training import OptimizerConfig
#
#    optimizer_config = OptimizerConfig(
#        optimizer_cls=torch.optim.AdamW,
#        optimizer_kwargs={
#            "lr": 5.0e-3,
#        },
#        scheduler_cls=TwoStageCosineConstantLR,
#        scheduler_kwargs={
#            "first_stage_steps": 54_400,
#            "second_stage_lr": 1.0e-3,
#            "eta_min": 1.0e-3,
#        },
#    )
#
# Hydra supplies learning-rate and schedule values; ``_optimizer(cfg)`` maps them
# onto :class:`~nvalchemi.training.OptimizerConfig`.

# sphinx_gallery_start_ignore


def _optimizer(cfg: DictConfig) -> OptimizerConfig:
    """Build the OptimizerConfig used by TrainingStrategy."""
    if cfg.training.get("epochs", None) is not None:
        raise ValueError("Set training.epochs=null and training.steps to an integer.")
    if cfg.training.get("steps", None) is None:
        raise ValueError("Set training.steps to the desired optimizer-step count.")

    scheduler_cfg = cfg.training.scheduler
    return OptimizerConfig(
        optimizer_cls=torch.optim.AdamW,
        optimizer_kwargs={
            "lr": float(cfg.training.optimizer.lr),
            "weight_decay": float(cfg.training.optimizer.get("weight_decay", 5e-7)),
        },
        scheduler_cls=TwoStageCosineConstantLR,
        scheduler_kwargs={
            "first_stage_steps": int(scheduler_cfg.first_stage_steps),
            "second_stage_lr": float(cfg.training.optimizer.stage_two_lr),
            "eta_min": float(scheduler_cfg.eta_min),
        },
    )


# sphinx_gallery_end_ignore

# %%
# Adding runtime hooks
# --------------------
# Hooks extend the core training loop without embedding that logic in the loop
# itself. For example, :class:`~nvalchemi.training.DDPHook` wraps the model in
# DDP at the :class:`~nvalchemi.training.TrainingStage` ``SETUP`` stage when
# ``training.distributed.enabled`` is true.
# :class:`~nvalchemi.training.EMAHook` maintains shadow weights for validation at
# ``AFTER_OPTIMIZER_STEP``, and :class:`~nvalchemi.hooks.NeighborListHook` rebuilds
# the interaction graph at ``BEFORE_FORWARD`` before every forward pass.
#
# .. code-block:: python
#
#    from pathlib import Path
#
#    from examples.advanced._mace_training_helpers import (
#        GradientClipHook,
#        TrainingMetricsLogger,
#    )
#    from nvalchemi.hooks import NeighborListHook
#    from nvalchemi.training import (
#        CheckpointHook,
#        DDPHook,
#        EMAHook,
#        TrainingStage,
#    )
#
#    hooks = [
#        DDPHook(backend="nccl", sampler_kwargs={"seed": 42}),
#        EMAHook(model_key="main", decay=0.995),
#        GradientClipHook(max_norm=2.0),
#        NeighborListHook(
#            model.model_config.neighbor_config,
#            max_neighbors=256,
#            method="batch_naive_tile",
#            stage=TrainingStage.BEFORE_FORWARD,
#        ),
#        TrainingMetricsLogger(every=100),
#        CheckpointHook(
#            checkpoint_dir=Path("outputs/checkpoints"),
#            step_interval=10_000,
#        ),
#    ]
#
# ``GradientClipHook`` and ``TrainingMetricsLogger`` are implemented in this example's
# helper module. The other hooks shown above are public ALCHEMI training APIs. The runnable
# script assembles the full hook list from Hydra through ``_hooks(cfg, model)``.

# sphinx_gallery_start_ignore


def _hooks(
    cfg: DictConfig,
    model: torch.nn.Module,
) -> list[Hook | TrainingUpdateHook]:
    """Build runtime hooks for DDP, EMA, neighbor lists, logging, and checkpointing."""
    hooks: list[Hook | TrainingUpdateHook] = []

    # Distributed wrapping — DDPHook applies DDP at TrainingStage.SETUP.
    distributed_cfg = cfg.training.get("distributed", {})
    if bool(distributed_cfg.get("enabled", False)):
        backend = str(distributed_cfg.get("backend", "nccl"))
        hooks.append(
            DDPHook(backend=backend, sampler_kwargs={"seed": int(cfg.training.seed)})
        )

    # EMA — shadow weights for validation (use_ema="auto" in ValidationConfig).
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

    # Gradient clipping before the optimizer step.
    clip_grad = cfg.training.optimizer.get("clip_grad", 100.0)
    if clip_grad is not None and float(clip_grad) > 0.0:
        hooks.append(GradientClipHook(max_norm=float(clip_grad)))

    # Graph rebuild — NeighborListHook runs before every forward pass.
    hooks.append(
        NeighborListHook(
            model.model_config.neighbor_config,
            max_neighbors=int(cfg.training.get("max_neighbors", 256)),
            method=cfg.training.get("neighbor_list_method", None),
            stage=TrainingStage.BEFORE_FORWARD,
        )
    )

    # Metrics logging — train/validation scalars to stdout and an optional logger.
    # ``metrics_logger`` is pluggable: this example defaults to ``JsonLinesLogger``
    # when ``jsonl_path`` is set; pass an MLflow or Weights & Biases client instead
    # (any object with ``log_metrics``, ``log``, or ``log_metric`` and ``step=``).
    logging_cfg = cfg.training.get(
        "logging",
        cfg.training.get("tracking", cfg.training.get("metrics", {})),
    )
    jsonl_path = logging_cfg.get("jsonl_path", None)
    metrics_logger = JsonLinesLogger(jsonl_path) if jsonl_path is not None else None
    hooks.append(
        TrainingMetricsLogger(
            every=int(cfg.training.log_every_steps),
            logger=metrics_logger,
            logger_axis=str(logging_cfg.get("logger_axis", "step")),
        )
    )

    # Checkpointing — restartable snapshots on a step cadence.
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


# sphinx_gallery_end_ignore

# %%
# Configuring validation
# ----------------------
# Validation is configured with :class:`~nvalchemi.training.ValidationConfig`. The
# configuration specifies the validation data, validation function, loss function,
# evaluation cadence, and whether to use EMA weights. During
# :class:`~nvalchemi.training.TrainingStrategy.run`, the strategy evaluates validation
# at this cadence and once more at the end of training. The latest validation
# summary is stored on ``strategy.last_validation``.
#
# In multi-GPU runs, each rank evaluates a disjoint validation shard through a
# DistributedSampler. The runnable script builds this configuration with
# ``_build_validation_config(...)`` after the validation loader and loss function
# have been constructed.
#
# .. code-block:: python
#
#    from nvalchemi.training import ValidationConfig, default_training_fn
#
#    validation_config = ValidationConfig(
#        validation_data=val_batches,
#        validation_fn=default_training_fn,
#        loss_fn=loss_fn,
#        every_n_steps=1000,
#        grad_mode="auto",
#        use_ema="auto",
#        name="validation",
#    )

# sphinx_gallery_start_ignore


def _build_validation_config(
    cfg: DictConfig,
    validation_loader: DataLoader | None,
    loss_fn: ComposedLossFunction,
) -> ValidationConfig | None:
    """Build the ValidationConfig used by TrainingStrategy."""
    if not bool(cfg.training.validation.get("enabled", True)):
        return None
    if validation_loader is None:
        raise ValueError(
            "validation_loader is required when training.validation.enabled is true."
        )
    if cfg.training.validation.get("every_epochs", None) is not None:
        raise ValueError(
            "Training is step-based; set "
            "training.validation.every_epochs=null and use every_steps."
        )
    every_steps = cfg.training.validation.get("every_steps", None)
    if every_steps is None:
        raise ValueError("Set training.validation.every_steps for step-based training.")

    return ValidationConfig(
        validation_data=validation_loader,
        validation_fn=default_training_fn,
        loss_fn=loss_fn,
        every_n_steps=int(every_steps),
        grad_mode="auto",
        use_ema="auto",
        name="validation",
    )


# sphinx_gallery_end_ignore

# %%
# Running TrainingStrategy
# ------------------------
# The final step is to assemble the objects created above and hand them to
# :class:`~nvalchemi.training.TrainingStrategy`, which runs the training loop.
# On each step, it calls the training function, steps the optimizer and scheduler,
# invokes hooks at their registered stages, runs validation when configured, and
# tracks checkpointable training state.
#
# :class:`~nvalchemi.distributed.DistributedManager` provides distributed
# runtime information such as rank, local rank, world size, and device placement.
# The same code path works for single-GPU and multi-GPU launches. In distributed
# runs, :class:`~nvalchemi.training.DDPHook` uses the distributed manager to wrap
# the model and coordinate rank-specific behavior.
#
# .. code-block:: python
#
#    from nvalchemi.distributed import DistributedManager
#    from nvalchemi.training import TrainingStrategy, default_training_fn
#
#    DistributedManager.initialize()
#    manager = DistributedManager()
#    device = torch.device(manager.device)
#
#    strategy = TrainingStrategy(
#        models=model,
#        optimizer_configs=optimizer_config,
#        num_steps=68_000,
#        training_fn=default_training_fn,
#        loss_fn=loss_fn,
#        devices=[device],
#        distributed_manager=manager,
#        hooks=hooks,
#        validation_config=validation_config,
#    )
#
#    strategy.run(train_loader)
#
# Run the Hydra entrypoint on one or more GPUs:
#
# Single GPU:
#
# .. code-block:: bash
#
#    uv run --extra cu12 --extra mace python examples/advanced/10_mace_training.py
#
# Multi-GPU:
#
# .. code-block:: bash
#
#    uv run --extra cu12 --extra mace torchrun --standalone --nproc_per_node=8 \
#        examples/advanced/10_mace_training.py \
#        --config-name=10_vanilla_mace distributed.enabled=true
#
# The commands above use ``--extra cu12``; change ``uv run --extra cuXX`` to match
# your NVIDIA driver / CUDA toolkit availability (for example, ``cu13`` on CUDA 13.x).
#
# Note that ``training.batch_size`` is the per-process batch size. The global batch size
# is therefore ``training.batch_size * nproc_per_node``.

# sphinx_gallery_start_ignore


@hydra.main(version_base=None, config_path=".", config_name="10_vanilla_mace")
def main(cfg: DictConfig) -> None:
    """Run MACE training."""
    DistributedManager.initialize()
    manager = DistributedManager()
    if manager.rank == 0:
        print(OmegaConf.to_yaml(cfg, resolve=True), flush=True)
    device = (
        torch.device("cuda", manager.local_rank)
        if torch.cuda.is_available()
        else torch.device(manager.device)
    )
    torch.manual_seed(int(cfg.training.seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(cfg.training.seed))

    # Loading train and validation data
    train_loader = _loader(
        str(cfg.data.zarr_path),
        cfg,
        device=device,
    )
    validation_loader: DataLoader | None = None
    if bool(cfg.training.validation.get("enabled", True)):
        validation_loader = _loader(
            str(cfg.data.validation_zarr_path),
            cfg,
            device=device,
            batch_size=int(cfg.training.validation.batch_size),
            shuffle=False,
        )
        validation_sampler = make_validation_sampler(validation_loader.dataset, manager)
        if validation_sampler is not None:
            validation_loader.sampler = validation_sampler

    # Building the MACE model
    model: MACEWrapper = _build_model(cfg, device)
    if manager.rank == 0:
        print(f"Model parameters: {count_model_parameters(model):,}")

    # Building the loss function
    loss_fn: ComposedLossFunction = _build_mace_huber_loss(cfg.training.loss)

    # Building the runtime hooks
    hooks: list[Hook | TrainingUpdateHook] = _hooks(cfg, model)

    # Building the training strategy
    validation_config: ValidationConfig | None = _build_validation_config(
        cfg,
        validation_loader,
        loss_fn,
    )
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

    # Running the training strategy
    try:
        strategy.run(train_loader)
        save_final_checkpoint(cfg, strategy, manager)
    finally:
        close_zarr_loaders(train_loader, validation_loader)
        DistributedManager.cleanup()


if __name__ == "__main__":
    if _DOCS_BUILD:
        print(
            "Skipping Hydra training during docs build. Run with:\n"
            "uv run --extra cu12 --extra mace python "
            "examples/advanced/10_mace_training.py\n"
            "(use --extra cu13 instead depending on your CUDA availability)",
            flush=True,
        )
    else:
        main()

# sphinx_gallery_end_ignore

# %%
# Validation curves and reference results
# ---------------------------------------
# The figure below shows validation Huber losses from a full default-config run
# on 1× H100 GPU, which took about 80 minutes of wall time. Actual wall time may
# differ depending on system configuration, hardware, and software stack.
# The sharp transition near step 54,400 marks the stage-two loss-weight schedule
# configured by ``training.loss.stage_two.start_step``.
#
# .. image:: ../_static/vanilla_mace_validation_metrics_260701.png
#    :align: center
#    :width: 70%
#
# With this default config (68,000 optimizer steps, about 50 epochs on the MatPES r2SCAN
# train set), the trained model reaches held-out test MAEs of energy 25.5 meV/atom,
# forces 145 meV/Å, and stress 0.703 GPa. These values are comparable to the
# MatPES r2SCAN benchmarks reported in
# `the MatPES paper <https://arxiv.org/pdf/2503.04070>`__ and to training with the
# `MACE CLI <https://github.com/acesuit/mace>`__.
