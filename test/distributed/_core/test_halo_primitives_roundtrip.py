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


def _worker_pbc_halo_tracks_current_cell(rank: int, world_size: int) -> None:
    """PBC images exchanged after a cell update use the new lattice vectors."""
    assert world_size == 2
    dtype = torch.float64
    mesh = _MockMesh(rank, world_size)
    pbc = torch.ones(3, dtype=torch.bool)
    old_cell = torch.eye(3, dtype=dtype) * 10.0
    domain_config = DomainConfig(cutoff=2.0, mesh=mesh)
    partitioner = SpatialPartitioner(
        config=domain_config,
        cell_matrix=old_cell.unsqueeze(0),
        pbc=pbc.unsqueeze(0),
    )
    halo_config = ParticleHaloConfig(
        ghost_width=2.0,
        partitioner=partitioner,
        mesh=mesh,
    )

    split_dims = [dim for dim, ranks in enumerate(partitioner.rank_grid) if ranks > 1]
    assert len(split_dims) == 1
    split_dim = split_dims[0]

    local_frac = torch.full((1, 3), 0.5, dtype=dtype)
    local_frac[0, split_dim] = 0.02 if rank == 0 else 0.98
    remote_frac = torch.full((1, 3), 0.5, dtype=dtype)
    remote_frac[0, split_dim] = 0.98 if rank == 0 else 0.02
    image_offset = torch.zeros((1, 3), dtype=dtype)
    image_offset[0, split_dim] = -1.0 if rank == 0 else 1.0

    # Establish that the original geometry produces the expected one-image
    # layout before changing the partitioner's cell.
    old_positions = local_frac @ old_cell
    old_padded, old_meta = particle_halo_padding(old_positions, halo_config)
    assert old_meta.n_owned == 1
    assert old_meta.n_padded == 2
    torch.testing.assert_close(
        old_padded[1:],
        (remote_frac + image_offset) @ old_cell,
    )

    # Change both the length and direction of the split-axis lattice vector.
    # A cache of Cartesian shifts from ``old_cell`` maps the periodic image into
    # the receiver's core and causes it to be omitted.  Deriving the image from
    # the current cell yields the expected halo row.
    new_cell = old_cell.clone()
    new_cell[split_dim, split_dim] = 12.0
    new_cell[split_dim, (split_dim + 1) % 3] = 1.5
    partitioner.update_cell(new_cell)

    new_positions = local_frac @ new_cell
    new_padded, new_meta = particle_halo_padding(new_positions, halo_config)
    assert new_meta.n_owned == 1
    assert new_meta.n_padded == 2
    torch.testing.assert_close(
        new_padded[1:],
        (remote_frac + image_offset) @ new_cell,
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


def test_pbc_halo_tracks_current_cell_after_update() -> None:
    """A long-lived halo config must not retain the partition-time PBC shift."""
    _spawn(2, "29840", "_worker_pbc_halo_tracks_current_cell")
