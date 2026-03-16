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
Training an MLIP: End-to-end pipeline with checkpointing
=========================================================

This example demonstrates a complete training workflow using the
:class:`~nvalchemi.training.Trainer`, including:

* Generating synthetic training and validation data
* Composing a multi-task loss (energy + forces)
* Configuring gradient clipping and checkpointing
* Running an initial training phase
* Resuming training from a checkpoint
* Inspecting the :class:`~nvalchemi.training.TrainingResult`

We use :class:`~nvalchemi.models.demo.DemoModelWrapper` as a lightweight
stand-in for a real MLIP.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import torch

# ---------------------------------------------------------------------------
# CPU fallback for segmented reduction
# ---------------------------------------------------------------------------
# The built-in segmented-sum custom op wraps a warp GPU kernel that has no
# autograd formula registered.  For this CPU-only example we monkey-patch it
# with a pure-torch implementation so backward works.  On a GPU cluster you
# would skip this block and use the native kernels.
import nvalchemi.training.losses._base as _loss_base  # noqa: E402
from nvalchemi.data import AtomicData, Batch
from nvalchemi.models.demo import DemoModelWrapper
from nvalchemi.training import (
    GradClipConfig,
    TerminateOnStepsHook,
    Trainer,
    TrainingConfig,
    TrainingStageEnum,
)
from nvalchemi.training.losses import EnergyLoss, ForceLoss


def _cpu_segmented_sum(
    values: torch.Tensor, idx: torch.Tensor, num_segments: int
) -> torch.Tensor:
    out = torch.zeros(num_segments, device=values.device, dtype=values.dtype)
    out.scatter_add_(0, idx.long(), values)
    return out


_loss_base._segmented_sum = _cpu_segmented_sum

# %%
# Synthetic data
# --------------
# We build a small dataset of random molecular graphs. Each graph has a
# random number of atoms with fabricated positions, energies, and forces.
# In practice these would come from DFT calculations or another data source,
# and optimally stored in a `Zarr` dataset.

torch.manual_seed(0)

NUM_TRAIN = 16
NUM_VAL = 4
ATOMS_PER_SYSTEM = 5


def make_random_batch(num_systems: int) -> Batch:
    """Create a :class:`Batch` of random atomic systems with target labels."""
    data_list = []
    for _ in range(num_systems):
        n = ATOMS_PER_SYSTEM
        data = AtomicData(
            positions=torch.randn(n, 3),
            atomic_numbers=torch.randint(1, 10, (n,)),
            energies=torch.randn(1, 1),
            forces=torch.randn(n, 3),
        )
        data_list.append(data)
    return Batch.from_data_list(data_list)


train_batches = [make_random_batch(4) for _ in range(NUM_TRAIN // 4)]
val_batches = [make_random_batch(4) for _ in range(NUM_VAL // 4)]

print(f"Train loader: {len(train_batches)} batches, {NUM_TRAIN} systems total")
print(f"Val   loader: {len(val_batches)} batches, {NUM_VAL} systems total")

# %%
# Model
# -----
# :class:`~nvalchemi.models.demo.DemoModelWrapper` is a simple MLP that
# predicts per-atom energies (summed per graph) and conservative forces
# via autograd.  It implements :class:`~nvalchemi.models.base.BaseModelMixin`,
# so it works directly with the :class:`~nvalchemi.training.Trainer`.

model = DemoModelWrapper(num_atom_types=10, hidden_dim=32)
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# %%
# Loss function
# -------------
# Compose a multi-task loss using arithmetic operators. The trainer will
# compute each term and report individual contributions alongside the
# weighted total.

loss_fn = 1.0 * EnergyLoss() + 10.0 * ForceLoss()

# %%
# Training configuration
# ----------------------
# We use a temporary directory for checkpoints so the example stays
# self-contained.  In practice, point ``checkpoint_dir`` to a persistent
# location.

tmpdir = Path(tempfile.mkdtemp(prefix="nvalchemi_train_"))
checkpoint_dir = tmpdir / "checkpoints"

config = TrainingConfig(
    max_epochs=6,
    grad_clip=GradClipConfig(method="norm", max_value=1.0),
    checkpoint_every_n_epochs=2,
    checkpoint_dir=checkpoint_dir,
    val_every_n_epochs=2,
    log_every_n_steps=1,
    seed=42,
)
print(f"Checkpoint dir: {checkpoint_dir}")

# %%
# Phase 1 — Initial training
# ---------------------------
# The :class:`~nvalchemi.training.Trainer` is used as a context manager.
# Calling :meth:`~nvalchemi.training.Trainer.fit` runs the training loop
# and returns a :class:`~nvalchemi.training.TrainingResult`.

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.5)

with Trainer(
    model=model,
    loss=loss_fn,
    optimizer=optimizer,
    scheduler=scheduler,
    train_loader=train_batches,
    val_loader=val_batches,
    config=config,
) as trainer:
    result = trainer.fit()

print(f"\nPhase 1 complete — {result.epochs_completed} epochs")
print(f"  Best val loss : {result.best_val_loss}")
print(f"  Best checkpoint: {result.best_checkpoint}")

# %%
# Inspect checkpoints
# -------------------
# The trainer saves checkpoints every ``checkpoint_every_n_epochs`` epochs.

saved = sorted(checkpoint_dir.glob("*.pt"))
print(f"Saved checkpoints: {[p.name for p in saved]}")

# %%
# Phase 2 — Resume from checkpoint
# ----------------------------------
# To resume, create a new :class:`~nvalchemi.training.TrainingConfig` with
# ``resume_from`` pointing to a checkpoint file.  The :class:`Trainer`
# restores the model, optimizer, and scheduler state and continues from
# the saved epoch.

resume_ckpt = saved[-1] if saved else None
print(f"Resuming from: {resume_ckpt}")

resume_config = TrainingConfig(
    max_epochs=10,
    grad_clip=GradClipConfig(method="norm", max_value=1.0),
    checkpoint_every_n_epochs=2,
    checkpoint_dir=checkpoint_dir,
    val_every_n_epochs=2,
    log_every_n_steps=1,
    resume_from=resume_ckpt,
    seed=42,
)

optimizer_2 = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler_2 = torch.optim.lr_scheduler.StepLR(optimizer_2, step_size=4, gamma=0.5)

with Trainer(
    model=model,
    loss=loss_fn,
    optimizer=optimizer_2,
    scheduler=scheduler_2,
    train_loader=train_batches,
    val_loader=val_batches,
    config=resume_config,
) as trainer:
    result_2 = trainer.fit()

print(f"\nPhase 2 complete — {result_2.epochs_completed} epochs")
print(f"  Best val loss : {result_2.best_val_loss}")

# %%
# Using hooks — early termination
# --------------------------------
# :class:`~nvalchemi.training.TerminateOnStepsHook` stops training after
# a fixed number of steps, useful for debugging or benchmarking.

model_3 = DemoModelWrapper(num_atom_types=10, hidden_dim=32)
optimizer_3 = torch.optim.Adam(model_3.parameters(), lr=1e-3)

hook_config = TrainingConfig(
    max_epochs=1000,
    log_every_n_steps=1,
    seed=42,
)

with Trainer(
    model=model_3,
    loss=loss_fn,
    optimizer=optimizer_3,
    train_loader=train_batches,
    config=hook_config,
) as trainer:
    trainer.register_hook(
        TerminateOnStepsHook(max_count=6, stage=TrainingStageEnum.AFTER_STEP)
    )
    result_3 = trainer.fit()

print(f"\nHook demo — stopped after {result_3.epochs_completed} epoch(s)")

# %%
# Clean up
# --------
# Remove the temporary checkpoint directory.

shutil.rmtree(tmpdir, ignore_errors=True)
print("Temporary files cleaned up.")
