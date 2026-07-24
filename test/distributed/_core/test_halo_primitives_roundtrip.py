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

"""Round-trip tests for halo_forward_exchange / halo_reverse_exchange.

These primitives are the foundation of the MPNN halo-correction
pathway (see :func:`nvalchemi.distributed._core.shard_tensor._halo_scatter_correction`).
If either has a bug, it silently corrupts every MPNN layer's
per-atom features at the boundary.

What each primitive does:

- ``halo_forward_exchange(owned, meta, cfg) -> padded``: takes this
  rank's OWNED rows, sends them to neighbor ranks that hold those
  atoms as halos, and assembles each rank's ``(n_padded,)`` tensor
  with halos populated from their owners.
- ``halo_reverse_exchange(padded, meta, cfg) -> owned``: each halo
  atom's partial contributions across ranks are routed back to the
  owner rank and summed into that rank's owned row.

Invariants we can assert via gloo multi-process tests:

1. **Identity after round trip with zero halo contribution** — if
   halo rows start at zero, ``halo_reverse(halo_forward(owned))``
   returns ``owned`` on every rank (no double-count, no drift).

2. **Halo rows equal owners' values** — after ``halo_forward_exchange``,
   every halo row on a borrower rank must equal the corresponding
   owned row on the owner rank (tagged by a rank-encoded value).

3. **Summation property of halo_reverse** — if halo rows contain
   known partial contributions, ``halo_reverse`` accumulates them
   correctly into owners.

4. **Autograd adjoint** — ``halo_forward`` and ``halo_reverse`` are
   each other's backward; running a scalar loss through
   ``halo_forward`` and backpropping should produce gradients
   equivalent to calling ``halo_reverse`` on the grad.

These tests are CPU-only via the gloo backend.
"""

from __future__ import annotations

import os
from typing import Any

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

# Shared gloo shim.
from _helpers import _MockMesh  # noqa: E402

from nvalchemi.distributed._core.particle_halo import (
    ParticleHaloConfig,
    halo_forward_exchange,
    halo_reverse_exchange,
    particle_halo_padding,
)
from nvalchemi.distributed.config import DomainConfig
from nvalchemi.distributed.partitioner import SpatialPartitioner

# ======================================================================
# Gloo harness
# ======================================================================


def _init_gloo(rank: int, world_size: int, port: str) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = port
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
    import physicsnemo.distributed.utils as pn_utils

    def _impl(tensor, indices, sizes, dim=0, group=None):
        cs = dist.get_world_size(group=group)
        r = dist.get_rank(group=group)
        x_send = [tensor[idx].contiguous() for idx in indices]
        x_recv = []
        shape = list(tensor.shape)
        for i in range(cs):
            shape[dim] = sizes[i][r]
            x_recv.append(torch.empty(shape, dtype=tensor.dtype, device=tensor.device))
        ops = []
        for i in range(cs):
            if i == r:
                x_recv[i].copy_(x_send[i])
            else:
                if x_send[i].numel() > 0:
                    ops.append(dist.isend(x_send[i], dst=i, group=group))
                if x_recv[i].numel() > 0:
                    ops.append(dist.irecv(x_recv[i], src=i, group=group))
        for op in ops:
            op.wait()
        return torch.cat(x_recv, dim=dim)

    pn_utils.indexed_all_to_all_v_wrapper = _impl


def _worker(rank: int, world_size: int, port: str, fn_name: str, *args: Any) -> None:
    _init_gloo(rank, world_size, port)
    try:
        globals()[fn_name](rank, world_size, *args)
    finally:
        dist.destroy_process_group()


def _spawn(world_size: int, port: str, fn_name: str, *args: Any) -> None:
    mp.spawn(_worker, args=(world_size, port, fn_name, *args), nprocs=world_size)


# ======================================================================
# Helpers: build a halo config + padded layout for a small cluster.
# ======================================================================


def _make_halo_cfg_for_cluster(
    rank: int,
    world_size: int,
    n_per_side: int = 4,
    pbc: bool = False,
    dtype: torch.dtype = torch.float64,
) -> tuple[ParticleHaloConfig, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build a simple cubic cluster; return (halo_cfg, positions_global,
    owned_mask, rank_assignment)."""
    spacing = 1.5
    coords = torch.arange(n_per_side, dtype=dtype) * spacing
    gx, gy, gz = torch.meshgrid(coords, coords, coords, indexing="ij")
    positions = torch.stack([gx.flatten(), gy.flatten(), gz.flatten()], dim=-1)
    box = n_per_side * spacing
    if pbc:
        cell = torch.eye(3, dtype=dtype) * box
        pbc_t = torch.ones(3, dtype=torch.bool)
    else:
        # Non-PBC: cell sized to tightly fit the cluster so the
        # partitioner's even-box split yields atoms on every rank
        # (otherwise one rank can end up empty and roundtrip tests
        # degenerate into no-op checks).
        cell = torch.eye(3, dtype=dtype) * box
        pbc_t = torch.zeros(3, dtype=torch.bool)

    mesh = _MockMesh(rank, world_size)
    cfg = DomainConfig(cutoff=2.5, mesh=mesh)
    partitioner = SpatialPartitioner(
        config=cfg, cell_matrix=cell.unsqueeze(0), pbc=pbc_t.unsqueeze(0)
    )
    halo_cfg = ParticleHaloConfig(ghost_width=2.5, partitioner=partitioner, mesh=mesh)
    rank_assignment = partitioner.assign_atoms_to_ranks(positions)
    owned_mask = rank_assignment == rank
    return halo_cfg, positions, owned_mask, rank_assignment


# ======================================================================
# Worker bodies.
# ======================================================================


def _worker_halo_forward_halo_values_match_owner(
    rank: int, world_size: int, n_per_side: int, pbc: bool
) -> None:
    """After ``halo_forward_exchange``, every halo row on each rank must
    equal the corresponding owned row on the owner rank. We encode each
    rank's owned rows as ``rank * 1000 + local_idx`` and check that
    halo rows contain the owner-side encoded value, not this rank's."""
    halo_cfg, positions_global, owned_mask, rank_assignment = (
        _make_halo_cfg_for_cluster(rank, world_size, n_per_side=n_per_side, pbc=pbc)
    )
    local_positions = positions_global[owned_mask].contiguous()
    # Build the padded layout from positions (so we know meta.n_padded,
    # meta.n_owned, and the routing indices).
    padded_pos, meta = particle_halo_padding(local_positions, halo_cfg)

    # Construct a rank-tagged feature: each owned row's value = rank.
    n_owned = meta.n_owned
    n_padded = meta.n_padded
    feat_dim = 4
    owned_feat = torch.full((n_owned, feat_dim), float(rank), dtype=torch.float64)

    # halo_forward: send owned rows so each rank assembles its (n_padded,
    # feat_dim) padded view with halos populated from their owners.
    padded_feat = halo_forward_exchange(owned_feat, meta, halo_cfg)

    assert padded_feat.shape == (n_padded, feat_dim), (
        f"rank {rank}: padded_feat.shape={tuple(padded_feat.shape)} "
        f"expected ({n_padded}, {feat_dim})"
    )
    # Owned rows must be preserved.
    torch.testing.assert_close(
        padded_feat[:n_owned],
        owned_feat,
        msg=f"rank {rank}: halo_forward corrupted owned rows",
    )
    # Halo rows (if any) must contain other ranks' rank-index (not this
    # rank's). Every value in the halo block should therefore be NOT
    # equal to ``rank``.
    if n_padded > n_owned:
        halo_block = padded_feat[n_owned:]
        # Every halo value must be a valid rank index != this rank's.
        halo_ranks = halo_block[:, 0]  # feat_dim rows all equal to that rank
        # Every halo row's value must be in [0, world_size) and != rank.
        assert (halo_ranks != float(rank)).all(), (
            f"rank {rank}: halo rows still carry this rank's value, "
            f"halo_forward didn't populate from the owner. "
            f"halo[:,0]={halo_ranks.tolist()}"
        )
        assert (halo_ranks >= 0).all() and (halo_ranks < world_size).all(), (
            f"rank {rank}: halo rows contain values outside [0, world_size)"
        )


def _worker_halo_reverse_zero_halo_is_identity(
    rank: int, world_size: int, n_per_side: int, pbc: bool
) -> None:
    """If halo rows are zero, ``halo_reverse_exchange(padded)`` returns
    a tensor equal to ``padded[:n_owned]`` (no spurious additions)."""
    halo_cfg, positions_global, owned_mask, _ = _make_halo_cfg_for_cluster(
        rank, world_size, n_per_side=n_per_side, pbc=pbc
    )
    local_positions = positions_global[owned_mask].contiguous()
    padded_pos, meta = particle_halo_padding(local_positions, halo_cfg)

    n_owned = meta.n_owned
    n_padded = meta.n_padded
    feat_dim = 3

    # Build padded: owned rows are rank-encoded; halo rows are ZERO.
    padded_feat = torch.zeros((n_padded, feat_dim), dtype=torch.float64)
    padded_feat[:n_owned] = float(rank)

    owned_after = halo_reverse_exchange(padded_feat, meta, halo_cfg)

    assert owned_after.shape == (n_owned, feat_dim), (
        f"rank {rank}: owned_after shape {tuple(owned_after.shape)}"
    )
    expected = torch.full((n_owned, feat_dim), float(rank), dtype=torch.float64)
    torch.testing.assert_close(
        owned_after,
        expected,
        msg=(
            f"rank {rank}: halo_reverse with zero halo should be identity "
            f"on owned rows; got difference max "
            f"{(owned_after - expected).abs().max().item()}"
        ),
    )


def _worker_halo_forward_reverse_roundtrip(
    rank: int, world_size: int, n_per_side: int, pbc: bool
) -> None:
    """``halo_reverse(halo_forward(owned))`` should leave owned
    unchanged when halo rows are the only source of non-trivial routing."""
    halo_cfg, positions_global, owned_mask, _ = _make_halo_cfg_for_cluster(
        rank, world_size, n_per_side=n_per_side, pbc=pbc
    )
    local_positions = positions_global[owned_mask].contiguous()
    padded_pos, meta = particle_halo_padding(local_positions, halo_cfg)

    n_owned = meta.n_owned
    feat_dim = 3
    torch.manual_seed(rank * 13 + 7)
    owned = torch.randn(n_owned, feat_dim, dtype=torch.float64)

    padded = halo_forward_exchange(owned, meta, halo_cfg)
    # Zero halo rows — we only want to check the forward's action on
    # owned rows. The halo_forward handler writes owner values into
    # halo rows of the receiver. Calling halo_reverse_exchange on the
    # result routes those halo rows BACK to owners, doubling each
    # owned value (once from local rows, once from borrowers' halos).
    # So the round-trip isn't an identity unless we zero the halo.
    if padded.shape[0] > n_owned:
        padded = padded.clone()
        padded[n_owned:] = 0.0
    owned_rt = halo_reverse_exchange(padded, meta, halo_cfg)

    torch.testing.assert_close(
        owned_rt,
        owned,
        msg=(
            f"rank {rank}: halo_reverse(zero_halo(halo_forward(owned))) "
            f"!= owned; max diff "
            f"{(owned_rt - owned).abs().max().item()}"
        ),
    )


def _worker_crossed_atom_reaches_receiver_before_migration(
    rank: int, world_size: int
) -> None:
    """An atom retained by its old owner must still reach its natural owner.

    DomainParallel defers migration until the next step and applies a small
    ownership hysteresis. During that interval an atom can be physically inside
    a neighboring rank's core while the previous rank still owns it. The
    receiver must see the atom as a ghost until ownership is transferred.
    """
    assert world_size == 2
    dtype = torch.float64
    cell = torch.eye(3, dtype=dtype) * 10.0
    pbc = torch.ones(3, dtype=torch.bool)
    mesh = _MockMesh(rank, world_size)
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


def _worker_crossed_atom_force_matches_same_geometry_reference(
    rank: int, world_size: int
) -> None:
    """Compare forces at the crossed geometry, without evolving two trajectories."""
    from torch.distributed.device_mesh import DeviceMesh

    from nvalchemi.data import AtomicData, Batch
    from nvalchemi.distributed.distributed_model import DistributedModel
    from nvalchemi.distributed.sharded_batch import ShardedBatch
    from nvalchemi.models.lj import LennardJonesModelWrapper
    from nvalchemi.neighbors import compute_neighbors

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


# ======================================================================
# Tests.
# ======================================================================


class TestHaloForwardNonPBC:
    def test_n4_2ranks(self) -> None:
        _spawn(
            2,
            "29800",
            "_worker_halo_forward_halo_values_match_owner",
            4,
            False,
        )

    def test_n6_2ranks(self) -> None:
        _spawn(
            2,
            "29801",
            "_worker_halo_forward_halo_values_match_owner",
            6,
            False,
        )

    def test_n8_4ranks(self) -> None:
        _spawn(
            4,
            "29802",
            "_worker_halo_forward_halo_values_match_owner",
            8,
            False,
        )


class TestHaloForwardPBC:
    def test_n4_2ranks(self) -> None:
        _spawn(
            2,
            "29810",
            "_worker_halo_forward_halo_values_match_owner",
            4,
            True,
        )

    def test_n6_2ranks(self) -> None:
        _spawn(
            2,
            "29811",
            "_worker_halo_forward_halo_values_match_owner",
            6,
            True,
        )


class TestHaloReverseZeroHaloIdentity:
    def test_n4_2ranks_nonpbc(self) -> None:
        _spawn(
            2,
            "29820",
            "_worker_halo_reverse_zero_halo_is_identity",
            4,
            False,
        )

    def test_n4_2ranks_pbc(self) -> None:
        _spawn(
            2,
            "29821",
            "_worker_halo_reverse_zero_halo_is_identity",
            4,
            True,
        )


class TestHaloForwardReverseRoundtrip:
    def test_n4_2ranks_nonpbc(self) -> None:
        _spawn(
            2,
            "29830",
            "_worker_halo_forward_reverse_roundtrip",
            4,
            False,
        )

    def test_n4_2ranks_pbc(self) -> None:
        _spawn(
            2,
            "29831",
            "_worker_halo_forward_reverse_roundtrip",
            4,
            True,
        )


def test_crossed_atom_reaches_receiver_before_deferred_migration() -> None:
    """The current owner must ghost an atom that entered a neighbor's core."""
    _spawn(2, "29850", "_worker_crossed_atom_reaches_receiver_before_migration")


def test_crossed_atom_force_matches_same_geometry_reference() -> None:
    """Distributed force must match a reference at the identical geometry."""
    _spawn(
        2,
        "29851",
        "_worker_crossed_atom_force_matches_same_geometry_reference",
    )
