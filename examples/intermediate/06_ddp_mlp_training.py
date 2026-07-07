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
Distributed Training: DDPHook with a Dummy MLP
==============================================

This example trains a small MLP on synthetic per-system energy labels and uses
:class:`~nvalchemi.training.hooks.DDPHook` to configure
``torch.nn.parallel.DistributedDataParallel``. The dataset is intentionally
small and generated on the fly so the example focuses on the distributed
training wiring rather than model quality.

The script is written for Sphinx-gallery review: configuration is expressed as
constants, and each section explains the API decisions that matter when adapting
this pattern to a real model. The DDP training cell only runs under
``torchrun``. During documentation builds, where distributed environment
variables are absent, the example prints the launch command and exits cleanly.

Run on a single node with ``torchrun`` through ``uv``:

.. code-block:: bash

   uv run --extra cu12 torchrun --standalone --nproc_per_node=2 \
       examples/intermediate/06_ddp_mlp_training.py
"""

from __future__ import annotations

import os
from typing import Any

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from nvalchemi.data import AtomicData, Batch
from nvalchemi.distributed import DistributedManager
from nvalchemi.models.base import BaseModelMixin, ModelConfig
from nvalchemi.training import (
    DDPHook,
    EnergyMSELoss,
    OptimizerConfig,
    TrainingStage,
    TrainingStrategy,
    default_training_fn,
)

# %%
# Configure a fixed gallery example
# ---------------------------------
# Sphinx-gallery examples should be readable without command-line parsing. These
# constants are the values used when the file is launched with ``torchrun``. To
# experiment locally, edit the constants and rerun the same launch command.

BACKEND = "auto"  # ``auto`` lets DistributedManager choose NCCL or Gloo.
EPOCHS = 4
BATCH_SIZE = 8
NUM_SAMPLES = 64
NUM_ATOMS = 4
HIDDEN_DIM = 32
LEARNING_RATE = 5.0e-3
SEED = 123
LOG_EVERY = 2

# Launcher-only examples must not initialize process groups during docs builds.
# Sphinx sets ``NVALCHEMI_SPHINX_BUILD`` in ``docs/conf.py``; torchrun sets the
# rank/world-size variables during real launches.
_DOCS_BUILD = os.environ.get("NVALCHEMI_SPHINX_BUILD") == "1"
_DISTRIBUTED_ENV = "RANK" in os.environ and "WORLD_SIZE" in os.environ
_RUN_DDP_EXAMPLE = _DISTRIBUTED_ENV and not _DOCS_BUILD


# %%
# Define a tiny AtomicData dataset
# --------------------------------
# ``TrainingStrategy`` and the loss functions expect ALCHEMI ``AtomicData`` or
# ``Batch`` objects. This dataset generates fixed-size systems so the example can
# focus on DDP setup instead of neighbor lists, padding, or chemistry.


class DummyEnergyDataset(Dataset[AtomicData]):
    """Deterministic synthetic systems with per-system energy labels."""

    def __init__(self, *, num_samples: int, num_atoms: int, seed: int) -> None:
        self.num_samples = num_samples
        self.num_atoms = num_atoms
        self.seed = seed

    def __len__(self) -> int:
        """Return the number of synthetic samples."""
        return self.num_samples

    def __getitem__(self, index: int) -> AtomicData:
        """Generate one deterministic synthetic atomic system."""
        generator = torch.Generator().manual_seed(self.seed + index)
        positions = torch.randn(self.num_atoms, 3, generator=generator)
        atomic_numbers = torch.ones(self.num_atoms, dtype=torch.long)
        # The target is deliberately learnable: the MLP only has to regress a
        # smooth function of positions, not a real atomistic potential.
        energy = positions.square().sum().view(1, 1)
        return AtomicData(
            positions=positions,
            atomic_numbers=atomic_numbers,
            atomic_masses=torch.ones(self.num_atoms),
            energy=energy,
            forces=torch.zeros(self.num_atoms, 3),
        )


# %%
# Wrap a PyTorch module with BaseModelMixin
# -----------------------------------------
# TrainingStrategy works with ``BaseModelMixin`` wrappers. The key contract for
# this toy model is ``model_config.outputs={"energy"}``: default_training_fn
# converts that model output into ``predicted_energy`` for ``EnergyMSELoss``.


class SimpleEnergyMLP(torch.nn.Module, BaseModelMixin):
    """Small MLP that predicts one total energy per fixed-size system."""

    def __init__(self, *, num_atoms: int, hidden_dim: int) -> None:
        super().__init__()
        self.num_atoms = num_atoms
        self.network = torch.nn.Sequential(
            torch.nn.Linear(num_atoms * 3, hidden_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden_dim, 1),
        )
        self.model_config = ModelConfig(
            outputs=frozenset({"energy"}),
            autograd_outputs=frozenset(),
            autograd_inputs=frozenset(),
            required_inputs=frozenset({"positions"}),
            optional_inputs=frozenset(),
            supports_pbc=False,
            needs_pbc=False,
            neighbor_config=None,
        )

    @property
    def embedding_shapes(self) -> dict[str, tuple[int, ...]]:
        """Return no named embeddings for this toy model."""
        return {}

    def compute_embeddings(
        self, data: AtomicData | Batch, **kwargs: Any
    ) -> AtomicData | Batch:
        """Return ``data`` unchanged because the toy MLP has no embeddings."""
        return data

    def forward(
        self, data: AtomicData | Batch, **kwargs: Any
    ) -> dict[str, torch.Tensor]:
        """Predict per-graph energies from flattened atomic positions."""
        num_graphs = data.batch_size if isinstance(data, Batch) else 1
        # The dataset uses a fixed atom count, so every graph has the same
        # feature width. Production MLIPs usually avoid this flattening pattern.
        features = data.positions.reshape(num_graphs, self.num_atoms * 3)
        return {"energy": self.network(features)}


# %%
# Add small rank-zero logging hooks
# ---------------------------------
# These hooks are intentionally simple and run through the normal training hook
# lifecycle. ``RankZeroSetupLogger`` fires after ``DDPHook`` prepares the model
# and dataloader; ``RankZeroLossLogger`` prints local progress after optimizer
# steps. Real projects should use the reporting hooks for richer dashboards and
# rank reductions.


class RankZeroSetupLogger:
    """Explain the distributed training setup once DDPHook has run."""

    stage = TrainingStage.SETUP
    frequency = 1

    def __init__(
        self,
        *,
        requested_backend: str,
        resolved_backend: str,
        manager: DistributedManager,
        num_samples: int,
        num_atoms: int,
        batch_size: int,
        hidden_dim: int,
        lr: float,
    ) -> None:
        self.requested_backend = requested_backend
        self.resolved_backend = resolved_backend
        self.manager = manager
        self.num_samples = num_samples
        self.num_atoms = num_atoms
        self.batch_size = batch_size
        self.hidden_dim = hidden_dim
        self.lr = lr

    def __call__(self, ctx: Any, stage: TrainingStage) -> None:
        """Print a rank-zero summary of the setup-stage side effects."""
        if ctx.global_rank != 0:
            return
        strategy = ctx.workflow
        # DDPHook stores the active dataloader on the strategy workflow. Looking
        # here lets the log report whether the hook replaced the sampler.
        sampler = getattr(getattr(strategy, "active_dataloader", None), "sampler", None)
        sampler_fields = [
            f"{name}={getattr(sampler, name)}"
            for name in ("num_replicas", "rank", "shuffle")
            if hasattr(sampler, name)
        ]
        sampler_suffix = f" ({', '.join(sampler_fields)})" if sampler_fields else ""
        sampler_description = (
            "None" if sampler is None else f"{type(sampler).__name__}{sampler_suffix}"
        )
        sampler_status = (
            "DDPHook installed a DistributedSampler"
            if isinstance(sampler, DistributedSampler)
            else "DDPHook left the dataloader sampler unchanged"
        )
        print(
            "\nDDP MLP training example\n"
            "------------------------\n"
            f"requested backend: {self.requested_backend}\n"
            f"resolved backend:  {self.resolved_backend}\n"
            f"world size:        {self.manager.world_size}\n"
            f"rank-0 device:     {self.manager.device}\n"
            f"dataset:           {self.num_samples} synthetic systems, "
            f"{self.num_atoms} atoms each\n"
            "target:            energy = sum(positions ** 2) per system\n"
            f"model:             SimpleEnergyMLP(hidden_dim={self.hidden_dim})\n"
            f"optimizer:         Adam(lr={self.lr})\n"
            f"batch size:        {self.batch_size} systems per rank\n"
            f"sampler after DDP: {sampler_description}\n"
            f"sampler status:    {sampler_status}\n"
            "progress log:      rank-0 local mini-batch loss after each "
            "optimizer step\n",
            flush=True,
        )


class RankZeroLossLogger:
    """Record local losses and print progress on rank zero."""

    stage = TrainingStage.AFTER_BATCH
    frequency = 1

    def __init__(self, *, every: int) -> None:
        self.every = every

    def __call__(self, ctx: Any, stage: TrainingStage) -> None:
        """Print occasional rank-zero local loss progress."""
        if ctx.loss is None or ctx.global_rank != 0 or ctx.step_count % self.every != 0:
            return
        loss = float(ctx.loss.detach().cpu())
        print(
            "progress: "
            f"optimizer_step={ctx.step_count:03d} "
            f"epoch={ctx.epoch:02d} "
            f"rank0_local_loss={loss:.6f}",
            flush=True,
        )


# %%
# Run only under torchrun
# -----------------------
# DDP needs one Python process per rank. Sphinx-gallery executes examples as a
# normal single Python process, so the distributed launch cell is guarded by the
# same environment variables that ``torchrun`` sets. The docs still show all of
# the code users need, but the build does not try to create a process group.

if not _RUN_DDP_EXAMPLE:
    print(
        "Not running under torchrun; skipping DDP training. Run with:\n"
        "uv run --extra cu12 torchrun --standalone --nproc_per_node=2 "
        "examples/intermediate/06_ddp_mlp_training.py",
        flush=True,
    )
else:
    manager: DistributedManager | None = None
    try:
        # DistributedManager reads rank, world size, local rank, address, and
        # port from the torchrun environment. Explicit backend constants are
        # only needed when you want to force Gloo or NCCL.
        if not DistributedManager.is_initialized():
            if BACKEND == "auto":
                DistributedManager.initialize()
            else:
                DistributedManager.setup(
                    rank=int(os.environ.get("RANK", "0")),
                    world_size=int(os.environ.get("WORLD_SIZE", "1")),
                    local_rank=int(os.environ.get("LOCAL_RANK", "0")),
                    addr=os.environ.get("MASTER_ADDR", "localhost"),
                    port=os.environ.get("MASTER_PORT", "12355"),
                    backend=BACKEND,
                )
        manager = DistributedManager()
        backend = (
            dist.get_backend()
            if dist.is_available() and dist.is_initialized()
            else "single-process"
        )
        device = torch.device(manager.device)
        torch.manual_seed(SEED)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(SEED)

        # DDPHook replaces the dataloader sampler with DistributedSampler during
        # setup, so the original dataloader can look like ordinary PyTorch code.
        dataset = DummyEnergyDataset(
            num_samples=NUM_SAMPLES,
            num_atoms=NUM_ATOMS,
            seed=SEED,
        )
        dataloader = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            collate_fn=lambda samples: Batch.from_data_list(list(samples)),
            num_workers=0,
        )
        setup_logger = RankZeroSetupLogger(
            requested_backend=BACKEND,
            resolved_backend=backend,
            manager=manager,
            num_samples=len(dataset),
            num_atoms=NUM_ATOMS,
            batch_size=BATCH_SIZE,
            hidden_dim=HIDDEN_DIM,
            lr=LEARNING_RATE,
        )

        # TrainingStrategy prepares hooks before moving models to devices.
        # DDPHook uses that phase to select the rank-local device and later wrap
        # the model before optimizer construction.
        strategy = TrainingStrategy(
            models=SimpleEnergyMLP(
                num_atoms=NUM_ATOMS,
                hidden_dim=HIDDEN_DIM,
            ),
            optimizer_configs=OptimizerConfig(
                optimizer_cls=torch.optim.Adam,
                optimizer_kwargs={"lr": LEARNING_RATE},
            ),
            num_epochs=EPOCHS,
            training_fn=default_training_fn,
            loss_fn=EnergyMSELoss(),
            devices=[device],
            distributed_manager=manager,
            hooks=[
                DDPHook(backend=None if BACKEND == "auto" else BACKEND),
                setup_logger,
                RankZeroLossLogger(every=LOG_EVERY),
            ],
        )

        strategy.run(dataloader)
    finally:
        if manager is not None:
            DistributedManager.cleanup()
