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

"""Two parallel FIRE -> NVTLangevin pipelines on 4 GPUs.

Topology::

    Rank 0 (FIRE)  -->  Rank 1 (NVTLangevin)
    Rank 2 (FIRE)  -->  Rank 3 (NVTLangevin)

Each FIRE rank optimises a batch of small molecules, and converged
samples are sent to the paired Langevin rank for short MD production.
A ``ConvergedSnapshotHook`` on the Langevin ranks saves completed
trajectories to a ``HostMemory`` sink.

Run with::

    torchrun --nproc_per_node=4 examples/05_distributed_pipeline_example.py
"""

from __future__ import annotations

import torch
import torch.distributed as dist
from ase.build import molecule
from loguru import logger

from nvalchemi.data import AtomicData, Batch
from nvalchemi.dynamics import (
    FIRE,
    ConvergenceHook,
    DistributedPipeline,
    HostMemory,
    NVTLangevin,
    SizeAwareSampler,
)
from nvalchemi.dynamics.base import HookStageEnum
from nvalchemi.dynamics.hooks import ConvergedSnapshotHook
from nvalchemi.models.demo import DemoModelWrapper

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def atoms_to_data(atoms) -> AtomicData:
    """Convert an ASE Atoms object to AtomicData with dynamics fields."""
    data = AtomicData.from_atoms(atoms)
    n = data.num_nodes
    data.forces = torch.zeros(n, 3)
    data.energies = torch.zeros(1, 1)
    data.add_node_property("velocities", torch.zeros(n, 3))
    return data


class InMemoryDataset:
    """Minimal dataset wrapper for ``SizeAwareSampler``."""

    def __init__(self, data_list: list[AtomicData]) -> None:
        self._data = data_list

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: int) -> tuple[AtomicData, dict]:
        d = self._data[idx]
        return d, {"num_atoms": d.num_nodes, "num_edges": d.num_edges}

    def get_metadata(self, idx: int) -> tuple[int, int]:
        d = self._data[idx]
        return d.num_nodes, d.num_edges


class DownstreamDoneHook:
    """Set ``stage.done = True`` after *patience* consecutive idle steps.

    Downstream (non-inflight) stages never set ``done`` on their own.
    This hook counts consecutive steps where the batch is empty (no
    graphs to integrate) and marks the stage as finished once the
    patience limit is reached.
    """

    stage = HookStageEnum.AFTER_STEP
    frequency = 1

    def __init__(self, patience: int = 5) -> None:
        self.patience = patience
        self._idle_steps = 0

    def __call__(self, batch: Batch, dynamics) -> None:
        if batch.num_graphs == 0:
            self._idle_steps += 1
        else:
            self._idle_steps = 0
        if self._idle_steps >= self.patience:
            dynamics.done = True


# ---------------------------------------------------------------------------
# Build molecules
# ---------------------------------------------------------------------------


def build_dataset() -> list[AtomicData]:
    """Create a handful of small rattled molecules."""
    names = ["H2O", "CH4", "NH3", "H2O", "CH4", "NH3", "H2O", "CH4"]
    data_list = []
    for name in names:
        atoms = molecule(name)
        atoms.rattle(stdev=0.15)
        data_list.append(atoms_to_data(atoms))
    return data_list


# ---------------------------------------------------------------------------
# Stage factories
# ---------------------------------------------------------------------------


def make_fire(model: DemoModelWrapper, rank: int, **kwargs) -> FIRE:
    """Create a FIRE optimiser stage."""
    return FIRE(
        model=model,
        dt=1.0,
        n_steps=50,
        convergence_hook=ConvergenceHook(
            criteria=[
                {
                    "key": "forces",
                    "threshold": 0.05,
                    "reduce_op": "norm",
                    "reduce_dims": -1,
                }
            ],
        ),
        **kwargs,
    )


def make_langevin(
    model: DemoModelWrapper,
    sink: HostMemory,
    rank: int,
    **kwargs,
) -> NVTLangevin:
    """Create an NVTLangevin MD stage with a snapshot hook."""
    return NVTLangevin(
        model=model,
        dt=0.5,
        temperature=300.0,
        friction=0.01,
        n_steps=20,
        hooks=[
            ConvergedSnapshotHook(sink=sink, frequency=1),
            DownstreamDoneHook(patience=10),
        ],
        convergence_hook=ConvergenceHook(
            criteria=[
                {
                    "key": "forces",
                    "threshold": 0.01,
                    "reduce_op": "norm",
                    "reduce_dims": -1,
                }
            ],
        ),
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Launch two parallel FIRE -> Langevin pipelines on 4 GPUs."""
    model = DemoModelWrapper()

    # Sinks (only used by ranks 1 and 3, but created on all for simplicity)
    sink_a = HostMemory(capacity=100)
    sink_b = HostMemory(capacity=100)

    # Dataset (only used by ranks 0 and 2)
    all_data = build_dataset()
    mid = len(all_data) // 2
    dataset_a = InMemoryDataset(all_data[:mid])
    dataset_b = InMemoryDataset(all_data[mid:])

    sampler_a = SizeAwareSampler(
        dataset=dataset_a,
        max_atoms=50,
        max_edges=0,
        max_batch_size=4,
    )
    sampler_b = SizeAwareSampler(
        dataset=dataset_b,
        max_atoms=50,
        max_edges=0,
        max_batch_size=4,
    )

    # Stages — one per rank.
    # By default prior_rank / next_rank are -1 (unset) and
    # DistributedPipeline.setup() would auto-wire a linear chain
    # 0 -> 1 -> 2 -> 3.  Setting them explicitly here creates two
    # independent sub-pipelines: 0 -> 1 and 2 -> 3.
    stages = {
        0: make_fire(
            model,
            rank=0,
            sampler=sampler_a,
            refill_frequency=1,
            prior_rank=None,
            next_rank=1,
        ),
        1: make_langevin(model, sink=sink_a, rank=1, prior_rank=0, next_rank=None),
        2: make_fire(
            model,
            rank=2,
            sampler=sampler_b,
            refill_frequency=1,
            prior_rank=None,
            next_rank=3,
        ),
        3: make_langevin(model, sink=sink_b, rank=3, prior_rank=2, next_rank=None),
    }

    pipeline = DistributedPipeline(stages=stages)
    pipeline.run()

    rank = dist.get_rank() if dist.is_initialized() else 0
    if rank == 1:
        logger.info(f"Rank 1 sink collected {len(sink_a)} samples")
    elif rank == 3:
        logger.info(f"Rank 3 sink collected {len(sink_b)} samples")


if __name__ == "__main__":
    main()
