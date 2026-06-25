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

Run on a single node with ``torchrun`` through ``uv``:

.. code-block:: bash

   uv run --extra cu12 torchrun --standalone --nproc_per_node=2 \
       examples/intermediate/06_ddp_mlp_training.py --backend auto

The ``--backend`` option accepts:

* ``auto``: let :class:`~nvalchemi.distributed.DistributedManager` choose the
  available backend.
* ``gloo``: explicitly use the Gloo process group.
* ``nccl``: explicitly use the NCCL process group.

Without a distributed launcher, the same script runs as a single-process smoke
test.
"""

from __future__ import annotations

import argparse
import os
from collections.abc import Sequence
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

# ---------------------------------------------------------------------------
# Dummy data
# ---------------------------------------------------------------------------
# The training stack expects ALCHEMI AtomicData/Batch objects. This toy dataset
# creates fixed-size systems so the MLP can flatten positions without padding.


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
        # A deliberately learnable target: the model only has to regress a
        # smooth function of positions, not a real atomistic potential.
        energy = positions.square().sum().view(1, 1)
        return AtomicData(
            positions=positions,
            atomic_numbers=atomic_numbers,
            atomic_masses=torch.ones(self.num_atoms),
            energy=energy,
            forces=torch.zeros(self.num_atoms, 3),
        )


# ---------------------------------------------------------------------------
# Model wrapper
# ---------------------------------------------------------------------------
# TrainingStrategy works with BaseModelMixin wrappers. The wrapper advertises
# that the model produces "energy"; default_training_fn will therefore expose it
# to the loss as "predicted_energy".


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


# ---------------------------------------------------------------------------
# Rank-zero reporting hooks
# ---------------------------------------------------------------------------
# Hooks keep the example output tied to the actual TrainingStrategy lifecycle:
# SETUP runs after DDPHook has prepared the model/dataloader, and AFTER_BATCH
# runs after each optimizer step.


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


# ---------------------------------------------------------------------------
# CLI and training assembly
# ---------------------------------------------------------------------------


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the DDP MLP example."""
    parser = argparse.ArgumentParser(
        description="Train a simple MLP with nvalchemi DDPHook on dummy data.",
    )
    parser.add_argument(
        "--backend",
        choices=("auto", "gloo", "nccl"),
        default="auto",
        help="Distributed backend. auto lets DistributedManager choose.",
    )
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-samples", type=int, default=64)
    parser.add_argument("--num-atoms", type=int, default=4)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--log-every", type=int, default=2)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the DDP MLP training example."""
    args = parse_args(argv)

    manager: DistributedManager | None = None

    try:
        # 1. Initialize distributed runtime. The manager reads torchrun, SLURM,
        #    and OpenMPI environments itself, then falls back to one process.
        if not DistributedManager.is_initialized():
            if args.backend == "auto":
                DistributedManager.initialize()
            else:
                DistributedManager.setup(
                    rank=int(os.environ.get("RANK", "0")),
                    world_size=int(os.environ.get("WORLD_SIZE", "1")),
                    local_rank=int(os.environ.get("LOCAL_RANK", "0")),
                    addr=os.environ.get("MASTER_ADDR", "localhost"),
                    port=os.environ.get("MASTER_PORT", "12355"),
                    backend=args.backend,
                )
        manager = DistributedManager()
        backend = (
            dist.get_backend()
            if dist.is_available() and dist.is_initialized()
            else "single-process"
        )
        device = torch.device(manager.device)
        torch.manual_seed(args.seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(args.seed)

        # 2. Build ordinary PyTorch data/model pieces. DDPHook will make the
        #    dataloader distributed-aware during TrainingStage.SETUP.
        dataset = DummyEnergyDataset(
            num_samples=args.num_samples,
            num_atoms=args.num_atoms,
            seed=args.seed,
        )
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=lambda samples: Batch.from_data_list(list(samples)),
            num_workers=0,
        )
        logger = RankZeroLossLogger(every=args.log_every)
        setup_logger = RankZeroSetupLogger(
            requested_backend=args.backend,
            resolved_backend=backend,
            manager=manager,
            num_samples=len(dataset),
            num_atoms=args.num_atoms,
            batch_size=args.batch_size,
            hidden_dim=args.hidden_dim,
            lr=args.lr,
        )
        # 3. Hand the manager and DDPHook to TrainingStrategy. DDPHook runs
        #    before optimizer construction, so Adam sees DDP-wrapped parameters.
        strategy = TrainingStrategy(
            models=SimpleEnergyMLP(
                num_atoms=args.num_atoms,
                hidden_dim=args.hidden_dim,
            ),
            optimizer_configs=OptimizerConfig(
                optimizer_cls=torch.optim.Adam,
                optimizer_kwargs={"lr": args.lr},
            ),
            num_epochs=args.epochs,
            training_fn=default_training_fn,
            loss_fn=EnergyMSELoss(),
            devices=[device],
            distributed_manager=manager,
            hooks=[
                DDPHook(backend=None if args.backend == "auto" else args.backend),
                setup_logger,
                logger,
            ],
        )

        # 4. Run training. The setup logger explains the resolved distributed
        #    configuration before the first batch, then the loss logger reports
        #    rank-zero progress after optimizer steps.
        strategy.run(dataloader)
    finally:
        if manager is not None:
            DistributedManager.cleanup()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
