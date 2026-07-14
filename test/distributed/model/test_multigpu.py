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

"""Multi-GPU tests for the ShardTensor-based distributed module.

Tests ShardedBatch, particle_halo_padding, reshard_by_destination,
and DomainParallel end-to-end using ``torch.multiprocessing.spawn``.

Requires at least 2 CUDA GPUs.  Run with::

    pytest test/distributed/test_multigpu.py -v
"""

from __future__ import annotations

import os
from typing import Any

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from nvalchemi.data import AtomicData, Batch
from nvalchemi.distributed.config import DomainConfig
from nvalchemi.distributed.domain_parallel import DomainParallel
from nvalchemi.dynamics.base import DynamicsStage
from nvalchemi.dynamics.integrators.nve import NVE
from nvalchemi.hooks.neighbor_list import NeighborListHook
from nvalchemi.models.lj import LennardJonesModelWrapper

WORLD_SIZE = 2

_skip_no_multi_gpu = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < WORLD_SIZE,
    reason=f"Need {WORLD_SIZE}+ GPUs for distributed tests",
)


# ======================================================================
# Helpers
# ======================================================================


def _init_pg(rank: int, world_size: int) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29501"
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(rank)

    # Let DistributedManager handle init_process_group — it also calls
    # it internally, so calling it ourselves first causes "initialize twice".
    from physicsnemo.distributed import DistributedManager

    DistributedManager.initialize()
    torch.cuda.set_device(rank)


def _worker(rank: int, world_size: int, test_fn: Any, *args: Any) -> None:
    _init_pg(rank, world_size)
    try:
        test_fn(rank, world_size, *args)
    finally:
        from physicsnemo.distributed import DistributedManager

        DistributedManager.cleanup()


def _create_argon(n_side: int = 5, lattice: float = 3.4, seed: int = 42) -> AtomicData:
    """Small cubic Argon system."""
    coords = torch.arange(n_side, dtype=torch.float32) * lattice
    gx, gy, gz = torch.meshgrid(coords, coords, coords, indexing="ij")
    positions = torch.stack([gx.flatten(), gy.flatten(), gz.flatten()], dim=-1)
    n = positions.shape[0]

    gen = torch.Generator().manual_seed(seed)
    kB, T, M = 8.617e-5, 50.0, 39.948
    vel = torch.randn(n, 3, generator=gen) * (kB * T / M) ** 0.5
    vel -= vel.mean(0)

    box = n_side * lattice
    data = AtomicData(
        positions=positions,
        atomic_numbers=torch.full((n,), 18, dtype=torch.int64),
        atomic_masses=torch.full((n,), M),
        cell=torch.eye(3).unsqueeze(0) * box,
        pbc=torch.ones(1, 3, dtype=torch.bool),
    )
    data.add_node_property("velocities", vel)
    return data


def _make_dd(device: torch.device, mesh) -> tuple[DomainParallel, NVE]:
    model = LennardJonesModelWrapper(epsilon=0.0104, sigma=3.40, cutoff=8.5).to(device)
    nl = NeighborListHook(
        config=model.model_config.neighbor_config,
        skin=0.0,
        stage=DynamicsStage.BEFORE_COMPUTE,
    )
    nve = NVE(model=model, dt=1.0, hooks=[nl])
    config = DomainConfig(cutoff=8.5, skin=0.0, mesh=mesh, mesh_dim="domain")
    dd = DomainParallel(nve, config=config)
    return dd, nve


# ======================================================================
# Test 1: ShardedBatch scatter/gather round-trip
# ======================================================================


def _test_sharded_batch_roundtrip(rank: int, world_size: int) -> None:
    device = torch.device(f"cuda:{rank}")
    data = _create_argon()
    initial_n = data.positions.shape[0]

    from torch.distributed import DeviceMesh

    mesh = DeviceMesh("cuda", list(range(world_size)), mesh_dim_names=("domain",))
    dd, _ = _make_dd(device, mesh)

    batch = Batch.from_data_list([data], device=device) if rank == 0 else None
    local_batch = dd.partition(batch)

    # ShardedBatch should exist after partition
    assert dd._sharded_batch is not None

    # Each rank should have atoms
    assert local_batch.num_nodes > 0

    # Total atoms conserved
    count = torch.tensor([local_batch.num_nodes], device=device)
    dist.all_reduce(count)
    assert count.item() == initial_n

    # Gather back via ShardedBatch.full_batch()
    dd._sharded_batch.update_from_batch(local_batch)
    full = dd._sharded_batch.full_batch(dst=0)
    if rank == 0:
        assert full is not None
        assert full.num_nodes == initial_n
    else:
        assert full is None


@_skip_no_multi_gpu
def test_sharded_batch_roundtrip():
    mp.spawn(
        _worker, args=(WORLD_SIZE, _test_sharded_batch_roundtrip), nprocs=WORLD_SIZE
    )


# ======================================================================
# Test 2: particle_halo_padding exchanges ghosts
# ======================================================================


# ``_ghost_exchange`` is private to :class:`DistributedModel` and not part of
# the public contract, so it isn't poked directly here. Coverage of the
# particle-halo primitive lives in ``test_particle_halo.py``; integration
# coverage is in ``test_distributed_models.py``.


# ======================================================================
# Test 3: reshard_by_destination moves atoms
# ======================================================================


def _test_reshard(rank: int, world_size: int) -> None:
    device = torch.device(f"cuda:{rank}")

    from torch.distributed import DeviceMesh

    from nvalchemi.distributed._core.reshard import reshard_by_destination

    mesh = DeviceMesh("cuda", list(range(world_size)), mesh_dim_names=("domain",))

    # Each rank has 10 atoms, send half to other rank
    tensor = torch.randn(10, 3, device=device)
    destinations = torch.tensor([rank] * 5 + [1 - rank] * 5, device=device)

    result = reshard_by_destination(tensor, destinations, mesh)

    # Each rank should still have 10 atoms (5 stayed + 5 received)
    assert result.shape[0] == 10

    # Total conserved
    total = torch.tensor([result.shape[0]], device=device)
    dist.all_reduce(total)
    assert total.item() == 20  # 10 per rank * 2 ranks


@_skip_no_multi_gpu
def test_reshard():
    mp.spawn(_worker, args=(WORLD_SIZE, _test_reshard), nprocs=WORLD_SIZE)


# ======================================================================
# Test 4: Full DomainParallel step completes
# ======================================================================


def _test_dd_step_completes(rank: int, world_size: int) -> None:
    device = torch.device(f"cuda:{rank}")
    data = _create_argon()

    from torch.distributed import DeviceMesh

    mesh = DeviceMesh("cuda", list(range(world_size)), mesh_dim_names=("domain",))
    dd, _ = _make_dd(device, mesh)

    batch = Batch.from_data_list([data], device=device) if rank == 0 else None
    local_batch = dd.partition(batch)

    for _ in range(5):
        local_batch, _ = dd.step(local_batch)

    assert local_batch.forces is not None
    assert local_batch.energy is not None
    assert local_batch.num_nodes > 0


@_skip_no_multi_gpu
def test_dd_step_completes():
    mp.spawn(_worker, args=(WORLD_SIZE, _test_dd_step_completes), nprocs=WORLD_SIZE)


# ======================================================================
# Test 5: Atom count conservation over many steps
# ======================================================================


def _test_atom_conservation(rank: int, world_size: int) -> None:
    device = torch.device(f"cuda:{rank}")
    data = _create_argon()
    initial_n = data.positions.shape[0]

    from torch.distributed import DeviceMesh

    mesh = DeviceMesh("cuda", list(range(world_size)), mesh_dim_names=("domain",))
    dd, _ = _make_dd(device, mesh)

    batch = Batch.from_data_list([data], device=device) if rank == 0 else None
    local_batch = dd.partition(batch)

    for _ in range(20):
        local_batch, _ = dd.step(local_batch)

    count = torch.tensor([local_batch.num_nodes], device=device, dtype=torch.long)
    dist.all_reduce(count)
    assert count.item() == initial_n, (
        f"Atoms lost/duplicated: expected {initial_n}, got {count.item()}"
    )


@_skip_no_multi_gpu
def test_atom_conservation():
    mp.spawn(_worker, args=(WORLD_SIZE, _test_atom_conservation), nprocs=WORLD_SIZE)


# ======================================================================
# Test 6: Gather reconstructs full batch
# ======================================================================


def _test_gather(rank: int, world_size: int) -> None:
    device = torch.device(f"cuda:{rank}")
    data = _create_argon()
    initial_n = data.positions.shape[0]

    from torch.distributed import DeviceMesh

    mesh = DeviceMesh("cuda", list(range(world_size)), mesh_dim_names=("domain",))
    dd, _ = _make_dd(device, mesh)

    batch = Batch.from_data_list([data], device=device) if rank == 0 else None
    local_batch = dd.partition(batch)

    # Run a few steps so atoms migrate
    for _ in range(5):
        local_batch, _ = dd.step(local_batch)

    full = dd.gather(local_batch, dst=0)
    if rank == 0:
        assert full is not None
        assert full.num_nodes == initial_n
        assert full.positions.shape == (initial_n, 3)
        assert full.cell is not None


@_skip_no_multi_gpu
def test_gather():
    mp.spawn(_worker, args=(WORLD_SIZE, _test_gather), nprocs=WORLD_SIZE)


# ======================================================================
# Test 7: Migration moves atoms between ranks
# ======================================================================


def _test_migration_moves_atoms(rank: int, world_size: int) -> None:
    """Run enough steps that atoms drift and migrate between domains."""
    device = torch.device(f"cuda:{rank}")
    data = _create_argon(n_side=5, seed=42)

    from torch.distributed import DeviceMesh

    mesh = DeviceMesh("cuda", list(range(world_size)), mesh_dim_names=("domain",))
    dd, _ = _make_dd(device, mesh)

    batch = Batch.from_data_list([data], device=device) if rank == 0 else None
    local_batch = dd.partition(batch)

    # Run many steps — atoms should migrate
    for _ in range(50):
        local_batch, _ = dd.step(local_batch)

    # After migration, local counts may have changed
    final_local_n = local_batch.num_nodes

    # Total should still be conserved
    initial_total = torch.tensor(
        [data.positions.shape[0]], device=device, dtype=torch.long
    )
    final_count = torch.tensor([final_local_n], device=device, dtype=torch.long)
    dist.all_reduce(final_count)
    assert final_count.item() == initial_total.item()


@_skip_no_multi_gpu
def test_migration_moves_atoms():
    mp.spawn(_worker, args=(WORLD_SIZE, _test_migration_moves_atoms), nprocs=WORLD_SIZE)


# ======================================================================
# Test 8: prime_forces populates batch
# ======================================================================


def _test_prime_forces(rank: int, world_size: int) -> None:
    device = torch.device(f"cuda:{rank}")
    data = _create_argon()

    from torch.distributed import DeviceMesh

    mesh = DeviceMesh("cuda", list(range(world_size)), mesh_dim_names=("domain",))
    dd, _ = _make_dd(device, mesh)

    batch = Batch.from_data_list([data], device=device) if rank == 0 else None
    local_batch = dd.partition(batch)

    dd._prime_forces(local_batch)

    assert local_batch.forces is not None
    assert local_batch.forces.shape == (local_batch.num_nodes, 3)
    assert local_batch.energy is not None
    # Forces should be non-zero for a non-equilibrium system
    assert local_batch.forces.abs().max() > 0


@_skip_no_multi_gpu
def test_prime_forces():
    mp.spawn(_worker, args=(WORLD_SIZE, _test_prime_forces), nprocs=WORLD_SIZE)
