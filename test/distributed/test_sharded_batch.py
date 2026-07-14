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
"""Tests for ShardedBatch.

Single-GPU tests validate the ShardedBatch data model, the ``local_batch``
property, ``update_from_batch()``, and ``_build_batch_from_tensors()``.
Multi-GPU tests are in test_multigpu.py.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import torch

from nvalchemi.distributed.sharded_batch import ShardedBatch, _has_field

# ======================================================================
# Helpers
# ======================================================================


def _mock_st(tensor: torch.Tensor):
    """Create a mock ShardTensor that returns *tensor* from to_local()."""
    m = MagicMock()
    m.to_local.return_value = tensor
    return m


def _make_sb(
    n_atoms: int = 10,
    include_velocities: bool = False,
    include_forces: bool = False,
) -> ShardedBatch:
    """Create a ShardedBatch with mock ShardTensors."""
    fields = {
        "positions": _mock_st(torch.randn(n_atoms, 3)),
        "atomic_numbers": _mock_st(torch.ones(n_atoms, dtype=torch.long)),
        "atomic_masses": _mock_st(torch.ones(n_atoms)),
    }
    if include_velocities:
        fields["velocities"] = _mock_st(torch.randn(n_atoms, 3))
    if include_forces:
        fields["forces"] = _mock_st(torch.randn(n_atoms, 3))

    return ShardedBatch(
        mesh=MagicMock(),
        atom_fields=fields,
        cell=torch.eye(3).unsqueeze(0) * 30.0,
        pbc=torch.ones(1, 3, dtype=torch.bool),
        n_global=n_atoms,
    )


# ======================================================================
# _has_field helper
# ======================================================================


class TestHasField:
    def test_existing_field(self):
        obj = MagicMock()
        obj.positions = torch.zeros(3)
        assert _has_field(obj, "positions")

    def test_none_field(self):
        obj = MagicMock()
        obj.positions = None
        assert not _has_field(obj, "positions")

    def test_missing_field(self):
        obj = MagicMock(spec=[])
        assert not _has_field(obj, "nonexistent")


# ======================================================================
# Properties
# ======================================================================


class TestShardedBatchProperties:
    def test_positions(self):
        sb = _make_sb()
        assert sb.positions is sb.fields["positions"]

    def test_n_owned(self):
        sb = _make_sb(n_atoms=42)
        assert sb.n_owned == 42

    def test_n_global(self):
        sb = _make_sb(n_atoms=42)
        assert sb.n_global == 42

    def test_velocities_none_when_absent(self):
        sb = _make_sb(include_velocities=False)
        assert sb.velocities is None

    def test_forces_none_when_absent(self):
        sb = _make_sb(include_forces=False)
        assert sb.forces is None

    def test_velocities_present(self):
        sb = _make_sb(include_velocities=True)
        assert sb.velocities is not None

    def test_forces_present(self):
        sb = _make_sb(include_forces=True)
        assert sb.forces is not None

    def test_atomic_numbers(self):
        sb = _make_sb()
        assert sb.atomic_numbers is sb.fields["atomic_numbers"]

    def test_atomic_masses(self):
        sb = _make_sb()
        assert sb.atomic_masses is sb.fields["atomic_masses"]


# ======================================================================
# to_batch
# ======================================================================


class TestShardedBatchLocalBatch:
    def test_produces_valid_batch(self):
        sb = _make_sb(n_atoms=10)
        batch = sb.local_batch
        assert batch.num_nodes == 10
        assert batch.positions.shape == (10, 3)

    def test_includes_velocities(self):
        sb = _make_sb(n_atoms=5, include_velocities=True)
        batch = sb.local_batch
        assert hasattr(batch, "velocities")
        assert batch.velocities.shape == (5, 3)

    def test_includes_forces(self):
        sb = _make_sb(n_atoms=5, include_forces=True)
        batch = sb.local_batch
        assert hasattr(batch, "forces")
        assert batch.forces.shape == (5, 3)

    def test_cell_preserved(self):
        sb = _make_sb()
        batch = sb.local_batch
        assert batch.cell is not None
        assert batch.cell.shape == (1, 3, 3)

    def test_pbc_preserved(self):
        sb = _make_sb()
        batch = sb.local_batch
        assert batch.pbc is not None


# ======================================================================
# _build_batch_from_tensors
# ======================================================================


class TestBuildBatchFromTensors:
    def test_basic(self):
        sb = _make_sb()
        tensors = {
            "positions": torch.randn(8, 3),
            "atomic_numbers": torch.ones(8, dtype=torch.long),
            "atomic_masses": torch.ones(8),
        }
        batch = sb._build_batch_from_tensors(tensors)
        assert batch.num_nodes == 8

    def test_with_velocities(self):
        sb = _make_sb()
        tensors = {
            "positions": torch.randn(5, 3),
            "atomic_numbers": torch.ones(5, dtype=torch.long),
            "atomic_masses": torch.ones(5),
            "velocities": torch.randn(5, 3),
        }
        batch = sb._build_batch_from_tensors(tensors)
        assert hasattr(batch, "velocities")

    def test_with_forces(self):
        sb = _make_sb()
        tensors = {
            "positions": torch.randn(5, 3),
            "atomic_numbers": torch.ones(5, dtype=torch.long),
            "atomic_masses": torch.ones(5),
            "forces": torch.randn(5, 3),
        }
        batch = sb._build_batch_from_tensors(tensors)
        assert hasattr(batch, "forces")


# ======================================================================
# atom_fields
# ======================================================================


class TestShardedBatchAtomFields:
    def test_returns_dict(self):
        sb = _make_sb()
        result = sb.atom_fields()
        assert isinstance(result, dict)
        assert "positions" in result
        assert "atomic_numbers" in result
        assert "atomic_masses" in result

    def test_returns_copy_not_reference(self):
        sb = _make_sb()
        result = sb.atom_fields()
        result["new_key"] = "foo"
        assert "new_key" not in sb.fields


# ======================================================================
# update_from_batch — single process (no ShardTensor, tests logic path)
# ======================================================================


class TestUpdateFromBatch:
    """Test update_from_batch logic without real ShardTensors.

    We use mock ShardTensors to verify the identity-check logic:
    if the batch tensor IS the same object as to_local(), skip update.
    """

    def test_skips_when_same_object(self):
        """If batch field is the same object as st.to_local(), no update."""
        pos_tensor = torch.randn(5, 3)
        z_tensor = torch.ones(5, dtype=torch.long)
        m_tensor = torch.ones(5)

        mock_pos = MagicMock()
        mock_pos.to_local.return_value = pos_tensor
        mock_z = MagicMock()
        mock_z.to_local.return_value = z_tensor
        mock_m = MagicMock()
        mock_m.to_local.return_value = m_tensor

        sb = ShardedBatch(
            mesh=MagicMock(),
            atom_fields={
                "positions": mock_pos,
                "atomic_numbers": mock_z,
                "atomic_masses": mock_m,
            },
            cell=torch.eye(3).unsqueeze(0),
            pbc=torch.ones(1, 3, dtype=torch.bool),
            n_global=5,
        )

        # Build a batch where ALL fields are the SAME objects as to_local()
        batch = MagicMock()
        batch.positions = pos_tensor
        batch.atomic_numbers = z_tensor
        batch.atomic_masses = m_tensor

        # Single-process unit test (mock mesh). ``update_from_batch``
        # unconditionally all-gathers n_owned to keep ranks in collective
        # lockstep; pin ``is_initialized`` False so it takes the
        # single-process early-return instead of driving a real collective on
        # the mock mesh (which otherwise trips on a process group leaked by an
        # earlier test in the same worker).
        with patch(
            "nvalchemi.distributed.sharded_batch.dist.is_initialized",
            return_value=False,
        ):
            sb.update_from_batch(batch)

        # All fields should still be the original mocks (no replacement)
        assert sb.fields["positions"] is mock_pos
        assert sb.fields["atomic_numbers"] is mock_z
        assert sb.fields["atomic_masses"] is mock_m


# Multi-GPU tests are in test_multigpu.py


# ======================================================================
# Gloo-harness tests for ``ShardedBatch.from_batch`` — the scatter path.
#
# These exercise the *production* ``from_batch`` (not the
# ``make_gloo_sharded_batch`` shim) via physicsnemo's real ``ShardTensor``
# on the gloo backend. They catch the class of bugs where the scatter
# silently diverges from the partitioner's rank assignment — e.g. a
# balanced split on a tensor whose per-rank sizes are uneven, which would
# leave one rank owning atoms that spatially belong to another rank.
#
# Can't call ``full_tensor()`` under gloo (all_gather needs equal sizes);
# we verify the invariant directly: each rank's ``local_batch`` contains
# exactly the atoms ``SpatialPartitioner.assign_atoms_to_ranks`` assigned
# to that rank.
# ======================================================================


def _gloo_worker(rank: int, world_size: int, port: str, fn_name: str) -> None:
    """Top-level (pickleable) gloo worker: init process group, dispatch
    by ``fn_name`` to the actual worker body, then tear down."""
    import os

    import torch.distributed as dist_mod

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = port
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    dist_mod.init_process_group(backend="gloo", rank=rank, world_size=world_size)
    try:
        globals()[fn_name](rank, world_size)
    finally:
        dist_mod.destroy_process_group()


def _gloo_spawn(world_size: int, port: str, fn_name: str) -> None:
    """Small test harness: spawn ``world_size`` gloo workers. ``fn_name``
    is the name of a module-level function taking ``(rank, world_size)``."""
    import torch.multiprocessing as mp

    mp.spawn(_gloo_worker, args=(world_size, port, fn_name), nprocs=world_size)


def _from_batch_rank_assignment_worker(rank: int, world_size: int) -> None:
    """Each rank builds the full batch on rank 0 (None elsewhere), calls
    ``ShardedBatch.from_batch``, then asserts its local atoms match the
    partitioner's assignment for that rank."""
    import torch as t
    from torch.distributed import DeviceMesh

    from nvalchemi.data import AtomicData, Batch
    from nvalchemi.distributed.config import DomainConfig
    from nvalchemi.distributed.partitioner import SpatialPartitioner
    from nvalchemi.distributed.sharded_batch import ShardedBatch

    # Cluster that WILL split unevenly: atoms packed at one corner of a
    # larger box. Partitioner cuts the box evenly → rank 0 gets most
    # atoms, rank 1 gets fewer. This is the regime where a balanced
    # scatter would mis-route atoms.
    spacing = 2.0 ** (1.0 / 6.0) * 3.40 * 1.05
    n_per_side = 6
    coords = t.arange(n_per_side, dtype=t.float64) * spacing
    gx, gy, gz = t.meshgrid(coords, coords, coords, indexing="ij")
    positions_global = t.stack([gx.flatten(), gy.flatten(), gz.flatten()], dim=-1)
    t.manual_seed(0)
    positions_global = positions_global + 0.05 * t.randn_like(positions_global)
    n = positions_global.shape[0]
    atomic_numbers_global = t.full((n,), 18, dtype=t.long)
    masses_global = t.full((n,), 39.948, dtype=t.float64)
    cell = t.eye(3, dtype=t.float64) * (n_per_side * spacing + 20.0)
    pbc = t.zeros(3, dtype=t.bool)

    # Independently compute the ground-truth rank assignment on every
    # rank (pure tensor op, deterministic) so we can check against it.
    mesh = DeviceMesh("cpu", list(range(world_size)), mesh_dim_names=("domain",))
    cfg = DomainConfig(cutoff=8.5, mesh=mesh)
    partitioner = SpatialPartitioner(
        config=cfg, cell_matrix=cell.unsqueeze(0), pbc=pbc.unsqueeze(0)
    )
    expected_assignment = partitioner.assign_atoms_to_ranks(positions_global)

    # --- Run from_batch ---
    if rank == 0:
        full_data = AtomicData(
            atomic_numbers=atomic_numbers_global,
            positions=positions_global,
            atomic_masses=masses_global,
            cell=cell.unsqueeze(0),
            pbc=pbc.unsqueeze(0),
        )
        full_batch = Batch.from_data_list([full_data])
    else:
        full_batch = None

    sharded = ShardedBatch.from_batch(full_batch, mesh=mesh, config=cfg)

    # --- Invariant 1: global atom count conserved ---
    assert sharded.n_global == n, f"n_global: {sharded.n_global} != {n}"

    # --- Invariant 2: sum of per-rank n_owned equals n_global ---
    local_n = sharded.n_owned
    import torch.distributed as dist_mod

    total = t.tensor([local_n], dtype=t.int64)
    dist_mod.all_reduce(total)
    assert int(total.item()) == n, f"sum(n_owned) {int(total.item())} != n_global {n}"

    # --- Invariant 3: this rank's owned positions are exactly the atoms
    # partitioner assigned to this rank. Compare (sorted) position sets.
    expected_mask = expected_assignment == rank
    expected_positions = positions_global[expected_mask]
    local_positions = sharded.positions.to_local()

    assert local_positions.shape[0] == int(expected_mask.sum().item()), (
        f"rank {rank}: n_owned {local_positions.shape[0]} != "
        f"expected {int(expected_mask.sum().item())}"
    )

    # Set-equality via sorted flattened coords (stable sort; positions
    # are distinct so the comparison is well-defined).
    exp_keys = expected_positions.flatten().sort().values
    loc_keys = local_positions.flatten().sort().values
    assert t.allclose(exp_keys, loc_keys), (
        f"rank {rank}: owned-atom set doesn't match partitioner's assignment"
    )


def test_from_batch_honors_partition_2ranks() -> None:
    """The scattered ``ShardedBatch`` matches ``SpatialPartitioner``'s
    rank assignment even when the per-rank counts are uneven — a balanced
    split would let a rank own atoms spatially belonging to another
    rank."""
    _gloo_spawn(2, "29680", "_from_batch_rank_assignment_worker")


def test_from_batch_honors_partition_4ranks() -> None:
    _gloo_spawn(4, "29681", "_from_batch_rank_assignment_worker")
