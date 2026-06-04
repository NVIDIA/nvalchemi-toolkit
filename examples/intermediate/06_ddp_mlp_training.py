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

* ``auto``: choose ``nccl`` when the requested local ranks fit on visible GPUs,
  otherwise choose ``gloo``.
* ``gloo``: run on CPU with the Gloo process group.
* ``nccl``: require one visible CUDA device per requested local rank.

The backend selection is intentionally single-node oriented: ``auto`` treats the
torchrun world size as the requested local rank count.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from typing import Any

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset

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


def _is_torchrun() -> bool:
    """Return whether this process appears to be launched by torchrun."""
    return dist.is_torchelastic_launched()


def resolve_backend(requested: str, *, requested_ranks: int) -> str:
    """Resolve ``auto``/``gloo``/``nccl`` into a concrete process backend."""
    if requested == "gloo":
        return "gloo"

    cuda_count = torch.cuda.device_count()
    nccl_ready = torch.cuda.is_available() and dist.is_nccl_available()
    ranks_fit_on_gpus = cuda_count >= requested_ranks

    if requested == "auto":
        return "nccl" if nccl_ready and ranks_fit_on_gpus else "gloo"

    if not nccl_ready:
        raise RuntimeError(
            "--backend nccl requested, but CUDA or the NCCL process group is "
            "not available in this environment."
        )
    if not ranks_fit_on_gpus:
        raise RuntimeError(
            "--backend nccl requested, but visible CUDA devices "
            f"({cuda_count}) are fewer than requested local ranks "
            f"({requested_ranks})."
        )
    return "nccl"


def setup_distributed_runtime(requested_backend: str) -> tuple[DistributedManager, str]:
    """Initialize process communication from torchrun and return the manager."""
    original_setup = DistributedManager.setup
    original_cuda_is_available = torch.cuda.is_available
    original_init_process_group = dist.init_process_group
    resolved_backend: dict[str, str] = {}

    def init_process_group_without_cpu_device_id(*args: Any, **kwargs: Any) -> Any:
        device_id = kwargs.get("device_id")
        if device_id is not None and torch.device(device_id).type == "cpu":
            kwargs = dict(kwargs)
            kwargs.pop("device_id")
        return original_init_process_group(*args, **kwargs)

    def setup(
        *,
        rank: int = 0,
        world_size: int = 1,
        local_rank: int | None = None,
        addr: str = "localhost",
        port: str = "12355",
        backend: str = "nccl",
        method: str = "env",
    ) -> None:
        selected = resolve_backend(requested_backend, requested_ranks=world_size)
        resolved_backend["value"] = selected
        if selected == "gloo":
            torch.cuda.is_available = lambda: False
            dist.init_process_group = init_process_group_without_cpu_device_id
        original_setup(
            rank=rank,
            world_size=world_size,
            local_rank=local_rank,
            addr=addr,
            port=port,
            backend=selected,
            method=method,
        )

    DistributedManager.setup = staticmethod(setup)
    try:
        DistributedManager.initialize_env()
        return DistributedManager(), resolved_backend["value"]
    except Exception:
        DistributedManager._shared_state = {}
        raise
    finally:
        DistributedManager.setup = staticmethod(original_setup)
        torch.cuda.is_available = original_cuda_is_available
        dist.init_process_group = original_init_process_group


def cleanup_distributed_runtime(manager: DistributedManager) -> None:
    """Destroy the process group created by this example."""
    DistributedManager.cleanup()


def training_device(manager: DistributedManager) -> torch.device:
    """Return the training device implied by the selected backend."""
    return torch.device(manager.device)


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
        energy = positions.square().sum().view(1, 1)
        return AtomicData(
            positions=positions,
            atomic_numbers=atomic_numbers,
            atomic_masses=torch.ones(self.num_atoms),
            energy=energy,
            forces=torch.zeros(self.num_atoms, 3),
        )


def collate_atomic_data(samples: Sequence[AtomicData]) -> Batch:
    """Collate synthetic systems into an ALCHEMI batch."""
    return Batch.from_data_list(list(samples))


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
        features = data.positions.reshape(num_graphs, self.num_atoms * 3)
        return {"energy": self.network(features)}


class RankZeroLossLogger:
    """Record local losses and print progress on rank zero."""

    stage = TrainingStage.AFTER_BATCH
    frequency = 1

    def __init__(self, *, every: int) -> None:
        self.every = every
        self.last_loss: float | None = None

    def __call__(self, ctx: Any, stage: TrainingStage) -> None:
        """Record the latest scalar loss and print occasional progress."""
        if ctx.loss is None:
            return
        self.last_loss = float(ctx.loss.detach().cpu())
        if ctx.global_rank == 0 and ctx.step_count % self.every == 0:
            print(
                "step="
                f"{ctx.step_count:03d} epoch={ctx.epoch:02d} "
                f"local_loss={self.last_loss:.6f}",
                flush=True,
            )


def mean_across_ranks(value: float, device: torch.device) -> float:
    """Return the distributed mean of a scalar value."""
    tensor = torch.tensor(value, device=device)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        tensor /= dist.get_world_size()
    return float(tensor.cpu())


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the DDP MLP example."""
    parser = argparse.ArgumentParser(
        description="Train a simple MLP with nvalchemi DDPHook on dummy data.",
    )
    parser.add_argument(
        "--backend",
        choices=("auto", "gloo", "nccl"),
        default="auto",
        help=(
            "Distributed backend. auto uses nccl when requested local ranks fit "
            "on visible GPUs, otherwise gloo."
        ),
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
    if not _is_torchrun():
        print(
            "This example is intended to run under torchrun. Try:\n"
            "uv run --extra cu12 torchrun --standalone --nproc_per_node=2 "
            "examples/intermediate/06_ddp_mlp_training.py --backend auto",
            flush=True,
        )
        return 0

    manager: DistributedManager | None = None

    try:
        manager, backend = setup_distributed_runtime(args.backend)
        device = training_device(manager)
        torch.manual_seed(args.seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(args.seed)

        dataset = DummyEnergyDataset(
            num_samples=args.num_samples,
            num_atoms=args.num_atoms,
            seed=args.seed,
        )
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=collate_atomic_data,
            num_workers=0,
        )
        logger = RankZeroLossLogger(every=args.log_every)
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
                DDPHook(backend=backend),
                logger,
            ],
        )

        if manager.rank == 0:
            print(
                f"backend={backend} world_size={manager.world_size} "
                f"device={device} samples={len(dataset)}",
                flush=True,
            )
        strategy.run(dataloader)
        if logger.last_loss is not None:
            final_loss = mean_across_ranks(logger.last_loss, device)
            if manager.rank == 0:
                print(f"mean_final_loss={final_loss:.6f}", flush=True)
    finally:
        if manager is not None:
            cleanup_distributed_runtime(manager)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
