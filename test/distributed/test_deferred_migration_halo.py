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
"""Cross-component regressions for halo exchange during deferred migration.

``DomainParallel`` can retain ownership briefly after an atom crosses a domain
boundary. These tests construct that intermediate state directly and verify
both the halo exchange and the resulting distributed force. They do not run an
integrator trajectory.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.distributed as dist
from _gloo_harness import run_gloo
from torch.distributed import DeviceMesh

from nvalchemi.data import AtomicData, Batch
from nvalchemi.distributed._core.particle_halo import (
    ParticleHaloConfig,
    particle_halo_padding,
)
from nvalchemi.distributed.config import DomainConfig
from nvalchemi.distributed.distributed_model import DistributedModel
from nvalchemi.distributed.partitioner import SpatialPartitioner
from nvalchemi.distributed.sharded_batch import ShardedBatch
from nvalchemi.models.lj import LennardJonesModelWrapper
from nvalchemi.neighbors import compute_neighbors


def _crossed_atom_reaches_receiver_before_migration(
    rank: int,
    world_size: int,
    _queue: Any,
) -> None:
    """An atom retained by its old owner must still reach its natural owner."""
    assert world_size == 2
    dtype = torch.float64
    cell = torch.eye(3, dtype=dtype) * 10.0
    pbc = torch.ones(3, dtype=torch.bool)
    mesh = DeviceMesh("cpu", list(range(world_size)), mesh_dim_names=("domain",))
    domain_config = DomainConfig(cutoff=2.0, skin=0.5, mesh=mesh)
    partitioner = SpatialPartitioner(
        config=domain_config,
        cell_matrix=cell.unsqueeze(0),
        pbc=pbc.unsqueeze(0),
    )
    halo_config = ParticleHaloConfig(
        ghost_width=domain_config.effective_ghost_width(),
        partitioner=partitioner,
        mesh=mesh,
    )

    split_dims = [dim for dim, ranks in enumerate(partitioner.rank_grid) if ranks > 1]
    assert len(split_dims) == 1
    split_dim = split_dims[0]

    # The atom starts in rank 0, then moves 0.2 Å across the boundary in one
    # update.
    previous_frac = torch.full((1, 3), 0.5, dtype=dtype)
    previous_frac[0, split_dim] = 0.49
    previous_position = previous_frac @ cell
    crossed_frac = torch.full((1, 3), 0.5, dtype=dtype)
    crossed_frac[0, split_dim] = 0.51
    crossed_position = crossed_frac @ cell
    if rank == 0:
        assert partitioner.assign_atoms_to_ranks(previous_position).item() == 0
        assert partitioner.assign_atoms_to_ranks(crossed_position).item() == 1
        # It crossed only 0.1 Å into rank 1, so the 0.25 Å migration hysteresis
        # deliberately leaves ownership with rank 0 for this compute.
        assert partitioner.keeps_owner(
            crossed_position,
            owner_rank=0,
            hysteresis=domain_config.effective_migration_hysteresis(),
        ).item()
        local_positions = crossed_position
    else:
        resident_frac = torch.full((1, 3), 0.5, dtype=dtype)
        resident_frac[0, split_dim] = 0.75
        local_positions = resident_frac @ cell

    padded_positions, metadata = particle_halo_padding(local_positions, halo_config)

    if rank == 1:
        assert metadata.n_owned == 1
        assert metadata.n_padded == 2, (
            "rank 1 did not receive the rank-0-owned atom after it crossed into "
            "rank 1's core but before deferred migration transferred ownership"
        )
        torch.testing.assert_close(padded_positions[1:], crossed_position)


def _crossed_atom_force_matches_same_geometry_reference(
    rank: int,
    world_size: int,
    _queue: Any,
) -> None:
    """Compare forces at the crossed geometry, without evolving trajectories."""
    assert world_size == 2
    dtype = torch.float64
    cell = torch.eye(3, dtype=dtype) * 20.0
    pbc = torch.ones(3, dtype=torch.bool)
    mesh = DeviceMesh("cpu", list(range(world_size)), mesh_dim_names=("domain",))
    domain_config = DomainConfig(cutoff=2.0, skin=0.5, mesh=mesh)
    partitioner = SpatialPartitioner(
        config=domain_config,
        cell_matrix=cell.unsqueeze(0),
        pbc=pbc.unsqueeze(0),
    )

    split_dims = [dim for dim, ranks in enumerate(partitioner.rank_grid) if ranks > 1]
    assert len(split_dims) == 1
    split_dim = split_dims[0]

    previous_frac = torch.full((1, 3), 0.5, dtype=dtype)
    previous_frac[0, split_dim] = 0.495
    previous_position = previous_frac @ cell
    crossed_frac = previous_frac.clone()
    crossed_frac[0, split_dim] = 0.505
    crossed_position = crossed_frac @ cell
    resident_frac = previous_frac.clone()
    resident_frac[0, split_dim] = 0.565
    resident_position = resident_frac @ cell
    rank0_far_frac = previous_frac.clone()
    rank0_far_frac[0, split_dim] = 0.2
    rank0_far_position = rank0_far_frac @ cell
    rank1_far_frac = previous_frac.clone()
    rank1_far_frac[0, split_dim] = 0.8
    rank1_far_position = rank1_far_frac @ cell

    # The far atoms keep the decomposition non-degenerate while remaining
    # outside every interaction and opposite-rank halo.
    positions_before_crossing = torch.cat(
        [
            previous_position,
            rank0_far_position,
            resident_position,
            rank1_far_position,
        ]
    )
    positions_at_compute = torch.cat(
        [
            crossed_position,
            rank0_far_position,
            resident_position,
            rank1_far_position,
        ]
    )
    assert torch.equal(
        partitioner.assign_atoms_to_ranks(positions_before_crossing),
        torch.tensor([0, 0, 1, 1]),
    )

    def _batch(positions: torch.Tensor) -> Batch:
        n_atoms = positions.shape[0]
        data = AtomicData(
            positions=positions.clone(),
            atomic_numbers=torch.full((n_atoms,), 18, dtype=torch.long),
            atomic_masses=torch.full((n_atoms,), 39.948, dtype=dtype),
            cell=cell.unsqueeze(0),
            pbc=pbc.unsqueeze(0),
        )
        return Batch.from_data_list([data])

    # Partition at the old geometry, then move only rank 0's owned atom. This
    # creates the real deferred-migration state: current position in rank 1,
    # ownership still on rank 0.
    sharded = ShardedBatch.from_batch(
        _batch(positions_before_crossing) if rank == 0 else None,
        mesh=mesh,
        config=domain_config,
        src=0,
    )
    owned_positions = sharded.positions.to_local()
    assert owned_positions.shape[0] == 2
    if rank == 0:
        owned_positions[0].copy_(crossed_position[0])
        assert partitioner.assign_atoms_to_ranks(crossed_position).item() == 1
        assert partitioner.keeps_owner(
            crossed_position,
            owner_rank=0,
            hysteresis=domain_config.effective_migration_hysteresis(),
        ).item()

    # The reference is a fresh single-process forward at exactly the positions
    # used by the distributed compute. It is not a separately evolved trajectory.
    reference_forces = torch.zeros(4, 3, dtype=dtype)
    if rank == 0:
        reference_model = LennardJonesModelWrapper(
            epsilon=1.0,
            sigma=1.0,
            cutoff=2.0,
        )
        reference_batch = _batch(positions_at_compute)
        compute_neighbors(
            reference_batch,
            config=reference_model.model_config.neighbor_config,
        )
        reference_forces.copy_(reference_model(reference_batch)["forces"].detach())
    dist.broadcast(reference_forces, src=0)
    assert torch.linalg.vector_norm(reference_forces[2]).item() > 0.1

    distributed_model = LennardJonesModelWrapper(
        epsilon=1.0,
        sigma=1.0,
        cutoff=2.0,
    )
    with DistributedModel(distributed_model, domain_config) as model:
        distributed_forces = model(sharded)["forces"]

    if rank == 0:
        torch.testing.assert_close(
            distributed_forces[0],
            reference_forces[0],
            rtol=1e-12,
            atol=1e-12,
            msg="rank 0's force should remain correct because it receives rank 1's atom",
        )
    else:
        actual_force = distributed_forces[0]
        expected_force = reference_forces[2]
        torch.testing.assert_close(
            actual_force,
            expected_force,
            rtol=1e-12,
            atol=1e-12,
            msg=(
                "rank 1 computed the wrong force because the interacting atom "
                "remained owned by rank 0 but was missing from rank 1's halo; "
                f"distributed={actual_force.tolist()}, "
                f"same_geometry_reference={expected_force.tolist()}"
            ),
        )


def test_crossed_atom_reaches_receiver_before_deferred_migration() -> None:
    """The current owner must ghost an atom that entered a neighbor's core."""
    run_gloo(world_size=2, fn=_crossed_atom_reaches_receiver_before_migration)


def test_crossed_atom_force_matches_same_geometry_reference() -> None:
    """Distributed force must match a reference at the identical geometry."""
    run_gloo(world_size=2, fn=_crossed_atom_force_matches_same_geometry_reference)
