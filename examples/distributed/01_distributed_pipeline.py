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
Distributed Multi-GPU Pipeline: Parallel FIRE → Langevin
=========================================================

This example orchestrates two independent FIRE → NVTLangevin pipelines
running in parallel across 4 GPUs using
:class:`~nvalchemi.dynamics.DistributedPipeline`.

.. rubric:: Topology

.. code-block:: text

    Rank 0 (FIRE, sampler_a)  ─→  Rank 1 (NVTLangevin, sink_a)
    Rank 2 (FIRE, sampler_b)  ─→  Rank 3 (NVTLangevin, sink_b)

Each FIRE rank draws molecules from a dataset, optimises them until
convergence, and sends them to the paired Langevin rank for short MD
production.  A :class:`~nvalchemi.dynamics.hooks.ConvergedSnapshotHook`
on the Langevin ranks writes completed trajectories to a
:class:`~nvalchemi.dynamics.HostMemory` sink.

.. note::

    This example requires 4 GPUs.  Run with::

        torchrun --nproc_per_node=4 examples/distributed/01_distributed_pipeline.py

    For CPU-only testing, change ``backend="nccl"`` to ``backend="gloo"``.
"""

from __future__ import annotations

import logging
import os

import torch
import torch.distributed as dist
from ase.build import molecule
from loguru import logger

from nvalchemi.data import AtomicData
from nvalchemi.dynamics import (
    FIRE,
    ConvergenceHook,
    DistributedPipeline,
    HostMemory,
    NVTLangevin,
    SizeAwareSampler,
)
from nvalchemi.dynamics.base import BufferConfig, DynamicsStage
from nvalchemi.dynamics.hooks import ConvergedSnapshotHook
from nvalchemi.hooks import HookContext
from nvalchemi.models.demo import DemoModelWrapper

logging.basicConfig(level=logging.INFO)

# When run outside ``torchrun`` (e.g. during a Sphinx docs build), the
# distributed environment variables ``RANK`` and ``WORLD_SIZE`` are absent.
# We detect this and skip the pipeline launch so the example renders in
# the gallery without requiring multiple GPUs.
_DISTRIBUTED_ENV = "RANK" in os.environ and "WORLD_SIZE" in os.environ

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def atoms_to_data(atoms) -> AtomicData:
    """Convert an ASE Atoms object to AtomicData with dynamics fields."""
    data = AtomicData.from_atoms(atoms)
    n = data.num_nodes
    data.forces = torch.zeros(n, 3)
    data.energy = torch.zeros(1, 1)
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
        """Get the metadata associated with idx"""
        d = self._data[idx]
        return d.num_nodes, d.num_edges


class DownstreamDoneHook:
    """Set ``stage.done = True`` after *patience* consecutive idle steps.

    Downstream (non-inflight) stages never set ``done`` on their own.
    This hook counts consecutive steps where the batch is empty (no
    graphs to integrate) and marks the stage as finished once the
    patience limit is reached.

    Because :class:`~nvalchemi.hooks.HookContext` does not carry a
    reference to the dynamics engine, the ``dynamics`` attribute must
    be set after the engine is constructed (see ``make_langevin``).
    """

    stage = DynamicsStage.AFTER_STEP
    frequency = 1

    def __init__(self, patience: int = 5) -> None:
        self.patience = patience
        self._idle_steps = 0
        self.dynamics: object | None = None

    def __call__(self, ctx: HookContext, stage_: DynamicsStage) -> None:
        if ctx.batch.num_graphs == 0:
            self._idle_steps += 1
        else:
            self._idle_steps = 0
        if self._idle_steps >= self.patience and self.dynamics is not None:
            self.dynamics.done = True


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


# %%
# DistributedPipeline topology
# ----------------------------------
# :class:`~nvalchemi.dynamics.DistributedPipeline` maps integer GPU ranks
# to dynamics stage instances.  When ``pipeline.run()`` is called, each
# process (launched by ``torchrun``) looks up its own rank and executes
# only the corresponding stage.  Inter-rank communication is handled
# transparently via NCCL ``isend``/``irecv`` calls.
#
# Here we create two independent sub-pipelines: ranks 0→1 and 2→3.
# Each sub-pipeline is a FIRE optimiser feeding into a Langevin MD stage.
# The ``stages`` dict is built on every rank but only the local stage is
# ever executed; constructing all stages on every rank keeps the code
# identical across processes, which simplifies debugging.

# %%
# Dataset and samplers
# -------------------------
# :class:`~nvalchemi.dynamics.SizeAwareSampler` draws molecules from a
# dataset and packs them into variable-size batches that fit within atom
# and edge count budgets.  In a distributed setting, each upstream rank
# owns its own sampler and dataset partition so that work is distributed
# evenly.  Downstream ranks (the Langevin stages) do not need a sampler —
# they receive systems directly from the paired FIRE rank via NCCL.


# %%
# BufferConfig: fixed-size communication
# -------------------------------------------
# NCCL requires that every ``isend``/``irecv`` pair transfers an identical
# number of bytes.  :class:`~nvalchemi.dynamics.base.BufferConfig` specifies
# the fixed sizes (``num_systems``, ``num_nodes``, ``num_edges``) used to
# pre-allocate communication buffers on both sender and receiver.  Choose
# values that are at least as large as the largest batch you expect to send
# in a single step; excess capacity is padded with zeros and stripped on
# receipt.

# %%
# Stage construction
# -----------------------
# Each stage is wired to its neighbours via ``prior_rank`` and ``next_rank``.
# An upstream stage has ``prior_rank=None`` and ``next_rank=<downstream>``;
# a downstream stage has ``prior_rank=<upstream>`` and ``next_rank=None``.
# The ``buffer_config`` must be identical for all stages that communicate
# with each other.


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
    done_hook = DownstreamDoneHook(patience=10)
    stage = NVTLangevin(
        model=model,
        dt=0.5,
        temperature=300.0,
        friction=0.01,
        n_steps=20,
        hooks=[
            ConvergedSnapshotHook(sink=sink, frequency=1),
            done_hook,
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
    done_hook.dynamics = stage
    return stage


# %%
# Running the pipeline
# -------------------------
# :class:`~nvalchemi.dynamics.DistributedPipeline` is used as a context
# manager.  On ``__enter__`` it initialises the PyTorch process group
# (``dist.init_process_group``) and assigns each rank its GPU device.
# On ``__exit__`` it tears down the process group gracefully.
#
# ``pipeline.run()`` blocks until the local stage signals completion
# (``dynamics.done = True``).  Upstream ranks finish when their sampler
# is exhausted; downstream ranks finish via ``DownstreamDoneHook`` after
# a configurable number of idle steps with no incoming systems.


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

    # Buffer config — matches sampler capacities and ensures fixed-size comm
    # buffers for NCCL compatibility (identical message count every step).
    buffer_cfg = BufferConfig(num_systems=4, num_nodes=50, num_edges=0)

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
            buffer_config=buffer_cfg,
        ),
        1: make_langevin(
            model,
            sink=sink_a,
            rank=1,
            prior_rank=0,
            next_rank=None,
            buffer_config=buffer_cfg,
        ),
        2: make_fire(
            model,
            rank=2,
            sampler=sampler_b,
            refill_frequency=1,
            prior_rank=None,
            next_rank=3,
            buffer_config=buffer_cfg,
        ),
        3: make_langevin(
            model,
            sink=sink_b,
            rank=3,
            prior_rank=2,
            next_rank=None,
            buffer_config=buffer_cfg,
        ),
    }

    if not _DISTRIBUTED_ENV:
        logger.info(
            "Not running under torchrun — skipping pipeline launch. "
            "Run with: torchrun --nproc_per_node=4 "
            "examples/distributed/01_distributed_pipeline.py",
        )
        return

    backend = "nccl"  # use gloo when testing with CPUs
    # debug mode will provide insight into what rank is doing what
    pipeline = DistributedPipeline(stages=stages, backend=backend, debug_mode=True)
    with pipeline:
        pipeline.run()

    rank = dist.get_rank() if dist.is_initialized() else 0
    if rank == 1:
        logger.info(f"Rank 1 sink collected {len(sink_a)} samples")
    elif rank == 3:
        logger.info(f"Rank 3 sink collected {len(sink_b)} samples")


if __name__ == "__main__":
    main()
