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
"""Multi-GPU correctness tests for DomainParallel.

These tests require at least 2 CUDA GPUs.  They use
``torch.multiprocessing.spawn`` to launch worker processes, each of which
initialises its own ``torch.distributed`` process group.

Run with::

    pytest test/dynamics/test_distributed_correctness.py -v
"""

from __future__ import annotations

import os
from typing import Any

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed import DeviceMesh

from nvalchemi._typing import ModelOutputs
from nvalchemi.data import AtomicData, Batch
from nvalchemi.distributed.config import DomainConfig
from nvalchemi.distributed.domain_parallel import DomainParallel
from nvalchemi.dynamics.base import DynamicsStage
from nvalchemi.dynamics.integrators.nve import NVE
from nvalchemi.hooks.neighbor_list import NeighborListHook
from nvalchemi.models.lj import LennardJonesModelWrapper
from nvalchemi.neighbors import compute_neighbors

WORLD_SIZE = 2
LJ_SIGMA = 3.4
LJ_CUTOFF = 8.5
NEIGHBOR_SKIN = 0.0

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_skip_no_multi_gpu = pytest.mark.skipif(
    torch.cuda.device_count() < WORLD_SIZE,
    reason=f"Need {WORLD_SIZE}+ GPUs for distributed tests",
)


def _init_process_group(rank: int, world_size: int) -> None:
    """Initialise NCCL process group for a spawned worker."""
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def _create_argon_system(
    n_atoms: int = 100,
    lattice_constant: float = 2 ** (1.0 / 6.0) * LJ_SIGMA * 1.05,
    seed: int = 42,
) -> AtomicData:
    """Create a stable periodic Argon system with deterministic state."""
    # Derive grid side from desired atom count (round up to next perfect cube)
    side = max(2, round(n_atoms ** (1.0 / 3.0)))
    positions = [
        [ix * lattice_constant, iy * lattice_constant, iz * lattice_constant]
        for ix in range(side)
        for iy in range(side)
        for iz in range(side)
    ]
    positions = torch.tensor(positions, dtype=torch.float32)
    actual_n = positions.shape[0]

    gen = torch.Generator().manual_seed(seed)
    box_length = side * lattice_constant
    positions += 0.01 * torch.randn(positions.shape, generator=gen)
    positions.remainder_(box_length)

    atomic_numbers = torch.full((actual_n,), 18, dtype=torch.long)
    atomic_masses = torch.full((actual_n,), 39.948, dtype=torch.float32)

    velocities = 0.001 * torch.randn(actual_n, 3, generator=gen)
    velocities -= velocities.mean(dim=0)

    # cell is [B, 3, 3] and pbc is [B, 3] per the current AtomicData schema.
    cell = (torch.eye(3, dtype=torch.float32) * box_length).unsqueeze(0)
    pbc = torch.ones(1, 3, dtype=torch.bool)

    data = AtomicData(
        positions=positions,
        atomic_numbers=atomic_numbers,
        atomic_masses=atomic_masses,
        forces=torch.zeros_like(positions),
        energy=torch.zeros(1, 1, dtype=positions.dtype),
        cell=cell,
        pbc=pbc,
    )
    data.add_node_property("velocities", velocities)
    return data


def _make_model(device: torch.device) -> LennardJonesModelWrapper:
    """Instantiate the LJ model on the given device."""
    return LennardJonesModelWrapper(
        epsilon=0.0104,
        sigma=LJ_SIGMA,
        cutoff=LJ_CUTOFF,
    ).to(device)


def _make_nve(model: LennardJonesModelWrapper) -> NVE:
    """Instantiate NVE integrator with a neighbor-list hook."""
    nl_hook = NeighborListHook(
        config=model.model_config.neighbor_config,
        skin=NEIGHBOR_SKIN,
        stage=DynamicsStage.BEFORE_COMPUTE,
    )
    return NVE(model=model, dt=1.0, hooks=[nl_hook])


def _partition_system(
    rank: int,
    world_size: int,
    data: AtomicData,
) -> tuple[DomainParallel, Batch]:
    """Build current-API domain-parallel NVE and partition one system."""
    device = torch.device(f"cuda:{rank}")
    mesh = DeviceMesh("cuda", list(range(world_size)), mesh_dim_names=("domain",))
    model = _make_model(device)
    dynamics = _make_nve(model)
    config = DomainConfig(
        cutoff=float(model.model_config.neighbor_config.cutoff),
        skin=NEIGHBOR_SKIN,
        mesh=mesh,
        mesh_dim="domain",
    )
    parallel = DomainParallel(dynamics, config=config)
    batch = Batch.from_data_list([data], device=device) if rank == 0 else None
    return parallel, parallel.partition(batch)


def _reference_outputs(batch: Batch, device: torch.device) -> ModelOutputs:
    """Evaluate LJ on a fresh batch at the gathered trajectory positions."""
    model = _make_model(device)
    data = AtomicData(
        positions=batch.positions.clone(),
        atomic_numbers=batch.atomic_numbers,
        atomic_masses=batch.atomic_masses,
        cell=batch.cell,
        pbc=batch.pbc,
    )
    ref_batch = Batch.from_data_list([data], device=device)
    compute_neighbors(ref_batch, config=model.model_config.neighbor_config)
    return model(ref_batch)


def _worker(rank: int, world_size: int, test_fn: Any, *args: Any) -> None:
    """Generic spawned-worker entry point."""
    _init_process_group(rank, world_size)
    try:
        test_fn(rank, world_size, *args)
    finally:
        dist.destroy_process_group()


# ---------------------------------------------------------------------------
# Test 1: Single-step force correctness
# ---------------------------------------------------------------------------


def _test_single_step_force_correctness(rank: int, world_size: int) -> None:
    """Compare one-step distributed forces at identical positions."""
    device = torch.device(f"cuda:{rank}")
    data = _create_argon_system(n_atoms=343, seed=42)
    parallel, local_batch = _partition_system(rank, world_size, data)
    local_batch, _ = parallel.step(local_batch)
    full_final = parallel.gather(local_batch, dst=0)

    if rank == 0:
        assert full_final is not None
        ref_forces = _reference_outputs(full_final, device)["forces"]
        torch.testing.assert_close(
            full_final.forces,
            ref_forces,
            atol=1e-5,
            rtol=1e-5,
        )


@_skip_no_multi_gpu
def test_single_step_force_correctness():
    """Per-atom forces must match between 1-GPU reference and 2-GPU DomainParallel."""
    mp.spawn(
        _worker,
        args=(WORLD_SIZE, _test_single_step_force_correctness),
        nprocs=WORLD_SIZE,
    )


# ---------------------------------------------------------------------------
# Test 2: NVE final-energy correctness
# ---------------------------------------------------------------------------


def _test_nve_final_energy_correctness(rank: int, world_size: int) -> None:
    """Compare final distributed energy at identical positions."""
    device = torch.device(f"cuda:{rank}")
    n_steps = 100
    data = _create_argon_system(n_atoms=343, seed=42)
    parallel, local_batch = _partition_system(rank, world_size, data)
    for _ in range(n_steps):
        local_batch, _ = parallel.step(local_batch)
    dist_final_energy = local_batch.energy.sum().clone()
    full_final = parallel.gather(local_batch, dst=0)

    if rank == 0:
        assert full_final is not None
        ref_energy = _reference_outputs(full_final, device)["energy"].sum()
        torch.testing.assert_close(
            dist_final_energy,
            ref_energy,
            atol=1e-4,
            rtol=1e-4,
        )


@_skip_no_multi_gpu
def test_nve_final_energy_correctness():
    """Final potential energy must match after 100 distributed NVE steps."""
    mp.spawn(
        _worker,
        args=(WORLD_SIZE, _test_nve_final_energy_correctness),
        nprocs=WORLD_SIZE,
    )


# ---------------------------------------------------------------------------
# Test 3: Atom count conservation
# ---------------------------------------------------------------------------


def _test_atom_count_conservation(rank: int, world_size: int) -> None:
    """Total atom count across all ranks must equal the initial count."""
    device = torch.device(f"cuda:{rank}")
    n_steps = 50

    data = _create_argon_system(n_atoms=343, seed=42)
    initial_n_atoms = data.positions.shape[0]
    parallel, local_batch = _partition_system(rank, world_size, data)

    # Run several steps (atoms may migrate between domains).
    for _ in range(n_steps):
        local_batch, _ = parallel.step(local_batch)

    # All-reduce local atom counts.
    local_count = torch.tensor(
        [local_batch.positions.shape[0]], dtype=torch.long, device=device
    )
    dist.all_reduce(local_count, op=dist.ReduceOp.SUM)
    total_count = local_count.item()

    if rank == 0:
        assert total_count == initial_n_atoms, (
            f"Atom count changed: expected {initial_n_atoms}, got {total_count}. "
            "Atoms were lost or duplicated during migration."
        )


@_skip_no_multi_gpu
def test_atom_count_conservation():
    """Total atoms across all ranks must equal the initial system size."""
    mp.spawn(
        _worker, args=(WORLD_SIZE, _test_atom_count_conservation), nprocs=WORLD_SIZE
    )


# ---------------------------------------------------------------------------
# Test 4: Partition distributes atoms
# ---------------------------------------------------------------------------


def _test_partition_distributes_atoms(rank: int, world_size: int) -> None:
    """Verify that after partition(), each rank has a disjoint subset of atoms."""
    data = _create_argon_system(n_atoms=343, seed=42)
    initial_n = data.positions.shape[0]
    _, local_batch = _partition_system(rank, world_size, data)

    # Each rank should have some atoms
    n_local = local_batch.num_nodes
    assert n_local > 0, f"Rank {rank} has no atoms"

    # Total across ranks should equal initial
    count = torch.tensor([n_local], device=local_batch.positions.device)
    dist.all_reduce(count)
    assert count.item() == initial_n

    # Local batch should have cell and pbc
    assert local_batch.cell is not None
    assert local_batch.pbc is not None


@_skip_no_multi_gpu
def test_partition_distributes_atoms():
    """After partition(), each rank has atoms and totals match initial count."""
    mp.spawn(
        _worker,
        args=(WORLD_SIZE, _test_partition_distributes_atoms),
        nprocs=WORLD_SIZE,
    )


# ---------------------------------------------------------------------------
# Phase-B removals: ``_ghost_exchange`` / ``_prime_forces`` /
# ``_prepare_padded_batch`` tests. Those methods moved into
# ``DistributedModel`` or were stale. Ghost-exchange coverage lives in
# ``test_particle_halo.py`` + ``test_distributed_models.py``; priming
# and bbox handling are exercised end-to-end by the step-based tests
# below (``test_step_completes``, ``test_nve_final_energy_correctness``).
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Test 5: step() completes without error
# ---------------------------------------------------------------------------


def _test_step_completes(rank: int, world_size: int) -> None:
    """Verify that dd.step() completes without error for 5 steps."""
    data = _create_argon_system(n_atoms=343, seed=42)
    parallel, local_batch = _partition_system(rank, world_size, data)

    for _i in range(5):
        local_batch, _converged = parallel.step(local_batch)

    assert local_batch.num_nodes > 0
    assert local_batch.forces is not None
    assert local_batch.energy is not None
    assert torch.isfinite(local_batch.positions).all()
    assert torch.isfinite(local_batch.forces).all()
    assert torch.isfinite(local_batch.energy).all()


@_skip_no_multi_gpu
def test_step_completes():
    """dd.step() must complete 5 steps without crashing."""
    mp.spawn(
        _worker,
        args=(WORLD_SIZE, _test_step_completes),
        nprocs=WORLD_SIZE,
    )


# (``test_prepare_unprepare_roundtrip_distributed`` lived here in the
# pre-Phase-B design. It referenced ``_prepare_padded_batch`` /
# ``_unprepare_padded_batch`` — methods that no longer exist. Coverage
# of the underlying PBC-flip-for-NL behavior now lives inside
# ``DistributedModel._call_halo`` and is exercised end-to-end by the
# step/run-based tests above.)
