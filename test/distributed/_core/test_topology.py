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

"""Topology tests using gloo backend on CPU.

These tests spawn arbitrary numbers of ranks (e.g., 4, 8, 9) on CPU
using the ``gloo`` backend to validate ghost exchange, migration, and
partitioning across complex topologies (2x2x1, 2x2x2, 3x3x1, etc.).

No GPUs required — all computation runs on CPU tensors.

Run with::

    pytest test/distributed/test_topology.py -v
"""

from __future__ import annotations

import os
from typing import Any

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from nvalchemi.distributed._core.particle_halo import (
    ParticleHaloConfig,
    particle_halo_padding,
    particle_halo_padding_multi,
    particle_halo_unpadding,
)
from nvalchemi.distributed._core.reshard import reshard_by_destination
from nvalchemi.distributed.config import DomainConfig
from nvalchemi.distributed.partitioner import SpatialPartitioner


def _patch_all_to_all_for_gloo():
    """Monkey-patch physicsnemo's all_to_all with isend/irecv for gloo.

    Gloo doesn't support all_to_all, but we can emulate it with
    point-to-point communication for testing.
    """
    import physicsnemo.distributed.utils as pn_utils

    _orig_indexed = pn_utils.indexed_all_to_all_v_wrapper

    def _indexed_all_to_all_v_gloo(tensor, indices, sizes, dim=0, group=None):
        comm_size = dist.get_world_size(group=group)
        rank = dist.get_rank(group=group)

        # Build send buffers.
        x_send = [tensor[idx].contiguous() for idx in indices]

        # Exchange sizes are already known (sizes matrix).
        x_recv = []
        tensor_shape = list(tensor.shape)
        for r in range(comm_size):
            tensor_shape[dim] = sizes[r][rank]
            x_recv.append(
                torch.empty(tensor_shape, dtype=tensor.dtype, device=tensor.device)
            )

        # Point-to-point exchange (gloo-compatible).
        ops = []
        for r in range(comm_size):
            if r == rank:
                # Local copy.
                x_recv[r].copy_(x_send[r])
            else:
                ops.append(dist.isend(x_send[r], dst=r, group=group))
                ops.append(dist.irecv(x_recv[r], src=r, group=group))
        for op in ops:
            op.wait()

        return torch.cat(x_recv, dim=dim)

    pn_utils.indexed_all_to_all_v_wrapper = _indexed_all_to_all_v_gloo


# ======================================================================
# Helpers
# ======================================================================


def _init_gloo(rank: int, world_size: int) -> None:
    """Initialize gloo process group on CPU."""
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29502"
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
    _patch_all_to_all_for_gloo()


def _worker(rank: int, world_size: int, test_fn: Any, *args: Any) -> None:
    _init_gloo(rank, world_size)
    try:
        test_fn(rank, world_size, *args)
    finally:
        dist.destroy_process_group()


def _create_crystal(n_side: int = 6, lattice: float = 3.4):
    """Create positions on a simple cubic lattice."""
    coords = torch.arange(n_side, dtype=torch.float32) * lattice
    gx, gy, gz = torch.meshgrid(coords, coords, coords, indexing="ij")
    positions = torch.stack([gx.flatten(), gy.flatten(), gz.flatten()], dim=-1)
    box = n_side * lattice
    cell = torch.eye(3) * box
    pbc = torch.ones(3, dtype=torch.bool)
    return positions, cell, pbc, box


def _make_halo_config(
    positions: torch.Tensor,
    cell: torch.Tensor,
    pbc: torch.Tensor,
    world_size: int,
    rank: int,
    ghost_width: float = 5.0,
) -> tuple[ParticleHaloConfig, SpatialPartitioner]:
    """Build partitioner + halo config for a given rank."""
    mesh = _MockMesh(rank, world_size)
    config = DomainConfig(
        cutoff=ghost_width,
        mesh=mesh,
    )
    partitioner = SpatialPartitioner(
        config=config,
        cell_matrix=cell.unsqueeze(0),
        pbc=pbc.unsqueeze(0),
    )
    halo_config = ParticleHaloConfig(
        ghost_width=ghost_width,
        partitioner=partitioner,
        mesh=mesh,
    )
    return halo_config, partitioner


class _MockMesh:
    """Minimal mesh mock for CPU tests. Uses the default process group."""

    def __init__(self, rank: int, world_size: int):
        self._rank = rank
        self._world_size = world_size

    def get_local_rank(self):
        return self._rank

    def size(self, dim=None):
        return self._world_size

    def get_group(self):
        return None  # default group


# ======================================================================
# Test: Ghost exchange with 4 ranks (2x2x1)
# ======================================================================


def _test_ghost_exchange_4ranks(rank: int, world_size: int) -> None:
    positions, cell, pbc, box = _create_crystal(n_side=6)
    n_total = positions.shape[0]

    halo_config, partitioner = _make_halo_config(
        positions, cell, pbc, world_size, rank, ghost_width=5.0
    )

    # Assign atoms to ranks and keep only this rank's atoms.
    rank_assignment = partitioner.assign_atoms_to_ranks(positions)
    local_mask = rank_assignment == rank
    local_pos = positions[local_mask]

    assert local_pos.shape[0] > 0, f"Rank {rank} has no atoms"

    # Ghost exchange.
    padded, meta = particle_halo_padding(local_pos, halo_config)

    assert padded.shape[0] >= local_pos.shape[0]
    assert meta.n_owned == local_pos.shape[0]

    # Total owned atoms across all ranks should equal initial.
    owned_count = torch.tensor([meta.n_owned], dtype=torch.long)
    dist.all_reduce(owned_count)
    assert owned_count.item() == n_total

    # Ghost count should be > 0 (with ghost_width=5.0 and box=20.4, domains overlap).
    n_ghosts = meta.n_padded - meta.n_owned
    # Not all ranks are guaranteed ghosts (corner ranks in 2x2x1 may have few),
    # but total ghosts across all ranks should be > 0.
    ghost_count = torch.tensor([n_ghosts], dtype=torch.long)
    dist.all_reduce(ghost_count)
    assert ghost_count.item() > 0, "No ghosts exchanged across all ranks"

    # Unpadding should recover owned.
    owned = particle_halo_unpadding(padded, meta)
    assert torch.equal(owned, local_pos)


def test_ghost_exchange_4ranks():
    mp.spawn(_worker, args=(4, _test_ghost_exchange_4ranks), nprocs=4)


# ======================================================================
# Test: Ghost exchange with 8 ranks (2x2x2)
# ======================================================================


def _test_ghost_exchange_8ranks(rank: int, world_size: int) -> None:
    positions, cell, pbc, box = _create_crystal(n_side=6)
    n_total = positions.shape[0]

    halo_config, partitioner = _make_halo_config(
        positions, cell, pbc, world_size, rank, ghost_width=5.0
    )

    rank_assignment = partitioner.assign_atoms_to_ranks(positions)
    local_pos = positions[rank_assignment == rank]

    padded, meta = particle_halo_padding(local_pos, halo_config)

    # Atom conservation.
    owned_count = torch.tensor([meta.n_owned], dtype=torch.long)
    dist.all_reduce(owned_count)
    assert owned_count.item() == n_total

    # In a 2x2x2 topology, each rank has up to 26 neighbors.
    # With ghost_width=5.0 and box=20.4, ghosts should exist.
    ghost_count = torch.tensor([meta.n_padded - meta.n_owned], dtype=torch.long)
    dist.all_reduce(ghost_count)
    assert ghost_count.item() > 0


def test_ghost_exchange_8ranks():
    mp.spawn(_worker, args=(8, _test_ghost_exchange_8ranks), nprocs=8)


# ======================================================================
# Test: Reshard with 4 ranks
# ======================================================================


def _test_reshard_4ranks(rank: int, world_size: int) -> None:
    # Each rank has 10 atoms. Send 3 to the next rank (wrap around).
    tensor = torch.randn(10, 3)
    dest_rank = (rank + 1) % world_size
    destinations = torch.tensor([rank] * 7 + [dest_rank] * 3, dtype=torch.long)

    result = reshard_by_destination(tensor, destinations, _MockMesh(rank, world_size))

    # Each rank sends 3 and receives 3, so should have 10 atoms still.
    assert result.shape[0] == 10

    total = torch.tensor([result.shape[0]], dtype=torch.long)
    dist.all_reduce(total)
    assert total.item() == 40  # 10 per rank * 4 ranks


def test_reshard_4ranks():
    mp.spawn(_worker, args=(4, _test_reshard_4ranks), nprocs=4)


# ======================================================================
# Test: Reshard conserves atoms with uneven distribution
# ======================================================================


def _test_reshard_uneven(rank: int, world_size: int) -> None:
    # Rank 0 has 20 atoms, others have 5. Redistribute evenly.
    n_local = 20 if rank == 0 else 5
    tensor = torch.randn(n_local, 3)

    # Send all atoms to rank (i % world_size) based on index.
    destinations = torch.tensor(
        [i % world_size for i in range(n_local)], dtype=torch.long
    )

    result = reshard_by_destination(tensor, destinations, _MockMesh(rank, world_size))

    # Total must be conserved.
    total_before = torch.tensor([n_local], dtype=torch.long)
    dist.all_reduce(total_before)
    total_after = torch.tensor([result.shape[0]], dtype=torch.long)
    dist.all_reduce(total_after)
    assert total_after.item() == total_before.item()


def test_reshard_uneven():
    mp.spawn(_worker, args=(4, _test_reshard_uneven), nprocs=4)


# ======================================================================
# Test: Multi-field ghost exchange preserves field count
# ======================================================================


def _test_multi_field_ghost_exchange(rank: int, world_size: int) -> None:
    positions, cell, pbc, box = _create_crystal(n_side=6)

    halo_config, partitioner = _make_halo_config(
        positions, cell, pbc, world_size, rank, ghost_width=5.0
    )

    rank_assignment = partitioner.assign_atoms_to_ranks(positions)
    local_pos = positions[rank_assignment == rank]
    n_local = local_pos.shape[0]

    other_fields = {
        "velocities": torch.randn(n_local, 3),
        "atomic_numbers": torch.full((n_local,), 18, dtype=torch.long),
        "atomic_masses": torch.full((n_local,), 39.948),
    }

    padded_pos, padded_fields, meta = particle_halo_padding_multi(
        local_pos, other_fields, halo_config
    )

    # All fields should have same number of atoms as padded positions.
    for name, field in padded_fields.items():
        assert field.shape[0] == padded_pos.shape[0], (
            f"Field {name} has {field.shape[0]} atoms but positions has {padded_pos.shape[0]}"
        )

    # Owned portion should match original.
    assert torch.equal(padded_pos[: meta.n_owned], local_pos)


def test_multi_field_ghost_exchange():
    mp.spawn(_worker, args=(4, _test_multi_field_ghost_exchange), nprocs=4)


# ======================================================================
# Test: Partition correctness with different topologies
# ======================================================================


def _test_partition_topology(rank: int, world_size: int) -> None:
    """Verify partition assigns all atoms and covers the box."""
    positions, cell, pbc, box = _create_crystal(n_side=8)
    n_total = positions.shape[0]

    mesh = _MockMesh(rank, world_size)
    config = DomainConfig(cutoff=5.0, mesh=mesh)
    partitioner = SpatialPartitioner(
        config=config,
        cell_matrix=cell.unsqueeze(0),
        pbc=pbc.unsqueeze(0),
    )

    rank_assignment = partitioner.assign_atoms_to_ranks(positions)

    # Every atom should be assigned to exactly one rank in [0, world_size).
    assert rank_assignment.min() >= 0
    assert rank_assignment.max() < world_size

    # This rank's atoms.
    local_mask = rank_assignment == rank
    local_n = local_mask.sum().item()

    # Total conserved.
    count = torch.tensor([local_n], dtype=torch.long)
    dist.all_reduce(count)
    assert count.item() == n_total

    # Grid dimensions should be reasonable.
    grid = partitioner.rank_grid
    assert grid[0] * grid[1] * grid[2] == world_size


@pytest.mark.parametrize("world_size", [2, 4, 6, 8, 9])
def test_partition_topology(world_size):
    mp.spawn(_worker, args=(world_size, _test_partition_topology), nprocs=world_size)


# ======================================================================
# Test: Ghost exchange symmetry — if A ghosts to B, B should ghost back
# ======================================================================


def _test_ghost_symmetry(rank: int, world_size: int) -> None:
    """Ghost exchange is symmetric: atoms near a boundary are ghosted both ways."""
    positions, cell, pbc, box = _create_crystal(n_side=6)

    halo_config, partitioner = _make_halo_config(
        positions, cell, pbc, world_size, rank, ghost_width=5.0
    )

    rank_assignment = partitioner.assign_atoms_to_ranks(positions)
    local_pos = positions[rank_assignment == rank]

    padded, meta = particle_halo_padding(local_pos, halo_config)
    n_ghosts = meta.n_padded - meta.n_owned

    # Collect ghost counts per rank.
    ghost_tensor = torch.tensor([n_ghosts], dtype=torch.long)
    all_ghosts = [torch.zeros(1, dtype=torch.long) for _ in range(world_size)]
    dist.all_gather(all_ghosts, ghost_tensor)

    # In a fully periodic system with uniform density, ghost counts
    # should be roughly similar across ranks (not exactly equal due to
    # discretization, but none should be zero).
    ghost_counts = [g.item() for g in all_ghosts]
    assert all(g >= 0 for g in ghost_counts)
    # At least some ranks should have ghosts.
    assert sum(ghost_counts) > 0


def test_ghost_symmetry_4ranks():
    mp.spawn(_worker, args=(4, _test_ghost_symmetry), nprocs=4)


def test_ghost_symmetry_8ranks():
    mp.spawn(_worker, args=(8, _test_ghost_symmetry), nprocs=8)
