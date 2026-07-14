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
"""Tests for autograd-aware halo feature exchange.

Uses gloo + torch.multiprocessing.spawn so tests run on CPU without needing
multiple GPUs. The gate: ``halo_forward_exchange`` and ``halo_reverse_exchange``
must be true adjoints, i.e. ``<y, fwd(x)> == <fwd.backward(y), x>`` on all ranks.
"""

from __future__ import annotations

import os
from typing import Any

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from nvalchemi.distributed._core.particle_halo import (
    ParticleHaloConfig,
    halo_forward_exchange,
    halo_reverse_exchange,
    particle_halo_padding,
)
from nvalchemi.distributed.config import DomainConfig
from nvalchemi.distributed.partitioner import SpatialPartitioner

# ======================================================================
# Gloo test harness (mirrors test_topology.py; local copy to keep each
# distributed test file self-contained)
# ======================================================================


def _patch_all_to_all_for_gloo() -> None:
    import physicsnemo.distributed.utils as pn_utils

    def _indexed_all_to_all_v_gloo(tensor, indices, sizes, dim=0, group=None):
        comm_size = dist.get_world_size(group=group)
        rank = dist.get_rank(group=group)
        x_send = [tensor[idx].contiguous() for idx in indices]
        x_recv = []
        tensor_shape = list(tensor.shape)
        for r in range(comm_size):
            tensor_shape[dim] = sizes[r][rank]
            x_recv.append(
                torch.empty(tensor_shape, dtype=tensor.dtype, device=tensor.device)
            )
        ops = []
        for r in range(comm_size):
            if r == rank:
                x_recv[r].copy_(x_send[r])
            elif x_send[r].numel() > 0 or x_recv[r].numel() > 0:
                if x_send[r].numel() > 0:
                    ops.append(dist.isend(x_send[r], dst=r, group=group))
                if x_recv[r].numel() > 0:
                    ops.append(dist.irecv(x_recv[r], src=r, group=group))
        for op in ops:
            op.wait()
        return torch.cat(x_recv, dim=dim)

    pn_utils.indexed_all_to_all_v_wrapper = _indexed_all_to_all_v_gloo


def _init_gloo(rank: int, world_size: int, port: str = "29503") -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = port
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


class _MockMesh:
    def __init__(self, rank: int, world_size: int) -> None:
        self._rank = rank
        self._world_size = world_size

    def get_local_rank(self) -> int:
        return self._rank

    def size(self, dim: int | None = None) -> int:
        return self._world_size

    def get_group(self) -> Any:
        return None


# ======================================================================
# Fixture helpers
# ======================================================================


def _cubic_lattice(
    n_side: int = 6, lattice: float = 3.4
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    coords = torch.arange(n_side, dtype=torch.float64) * lattice
    gx, gy, gz = torch.meshgrid(coords, coords, coords, indexing="ij")
    positions = torch.stack([gx.flatten(), gy.flatten(), gz.flatten()], dim=-1)
    cell = torch.eye(3, dtype=torch.float64) * (n_side * lattice)
    pbc = torch.ones(3, dtype=torch.bool)
    return positions, cell, pbc


def _build_rank_halo(
    rank: int,
    world_size: int,
    ghost_width: float = 5.0,
) -> tuple[torch.Tensor, Any, Any]:
    """Return this rank's owned positions, halo metadata, and halo config."""
    positions, cell, pbc = _cubic_lattice(n_side=6, lattice=3.4)
    mesh = _MockMesh(rank, world_size)
    domain_config = DomainConfig(cutoff=ghost_width, mesh=mesh)
    partitioner = SpatialPartitioner(
        config=domain_config,
        cell_matrix=cell.unsqueeze(0),
        pbc=pbc.unsqueeze(0),
    )
    halo_config = ParticleHaloConfig(
        ghost_width=ghost_width, partitioner=partitioner, mesh=mesh
    )
    rank_assignment = partitioner.assign_atoms_to_ranks(positions)
    local_pos = positions[rank_assignment == rank].contiguous()
    _padded_pos, meta = particle_halo_padding(local_pos, halo_config)
    return local_pos, meta, halo_config


def _compute_out_degree(meta: Any, n_owned: int) -> torch.Tensor:
    """How many neighbor ranks borrowed each owned atom (1 per rank copy)."""
    out_degree = torch.zeros(n_owned, dtype=torch.float64)
    for idx_tensor in meta.gnn_markers.send_indices_owned:
        if idx_tensor.numel() > 0:
            out_degree.index_add_(
                0, idx_tensor, torch.ones(idx_tensor.shape[0], dtype=torch.float64)
            )
    return out_degree


# ======================================================================
# Test 1: markers are populated
# ======================================================================


def _test_markers_populated(rank: int, world_size: int) -> None:
    _local_pos, meta, _config = _build_rank_halo(rank, world_size)
    assert meta.gnn_markers is not None
    assert len(meta.gnn_markers.send_indices_owned) == world_size
    # Every owned-index tensor must be in [0, n_owned).
    for r in range(world_size):
        idx = meta.gnn_markers.send_indices_owned[r]
        if idx.numel() > 0:
            assert int(idx.max()) < meta.n_owned
            assert int(idx.min()) >= 0


def test_markers_populated_2ranks() -> None:
    mp.spawn(_worker, args=(2, _test_markers_populated), nprocs=2)


def test_markers_populated_4ranks() -> None:
    mp.spawn(_worker, args=(4, _test_markers_populated), nprocs=4)


# ======================================================================
# Test 2: forward exchange fills halo with copies of owner values
# ======================================================================


def _test_forward_fills_halo(rank: int, world_size: int) -> None:
    local_pos, meta, config = _build_rank_halo(rank, world_size)
    n_owned = local_pos.shape[0]

    # Features: encode (rank, atom_idx) so we can verify halo contents.
    features = torch.full(
        (n_owned, 1), float(rank + 1) * 1000.0, dtype=torch.float64
    ) + torch.arange(n_owned, dtype=torch.float64).unsqueeze(-1)

    padded = halo_forward_exchange(features, meta, config)
    assert padded.shape[0] == meta.n_padded
    torch.testing.assert_close(padded[:n_owned], features)

    # Verify via a weaker invariant — every halo value has the rank-signature
    # of some OTHER rank. We can't reconstruct the sender's exact values here
    # (we don't see the sender's local atom indices), but the rank-encoded
    # prefix tells us where each halo row came from.
    halo = padded[n_owned:]
    for v in halo.flatten().tolist():
        sender_rank = int(v // 1000.0) - 1
        assert 0 <= sender_rank < world_size
        assert sender_rank != rank


def test_forward_fills_halo_2ranks() -> None:
    mp.spawn(_worker, args=(2, _test_forward_fills_halo), nprocs=2)


def test_forward_fills_halo_4ranks() -> None:
    mp.spawn(_worker, args=(4, _test_forward_fills_halo), nprocs=4)


# ======================================================================
# Test 3: rev(fwd(ones)) == 1 + out_degree
# ======================================================================


def _test_rev_of_fwd_ones(rank: int, world_size: int) -> None:
    local_pos, meta, config = _build_rank_halo(rank, world_size)
    n_owned = local_pos.shape[0]

    x = torch.ones((n_owned, 2), dtype=torch.float64)
    padded = halo_forward_exchange(x, meta, config)
    y = halo_reverse_exchange(padded, meta, config)

    out_degree = _compute_out_degree(meta, n_owned)
    expected = (1.0 + out_degree).unsqueeze(-1).expand_as(y)
    torch.testing.assert_close(y, expected)


def test_rev_of_fwd_ones_2ranks() -> None:
    mp.spawn(_worker, args=(2, _test_rev_of_fwd_ones), nprocs=2)


def test_rev_of_fwd_ones_4ranks() -> None:
    mp.spawn(_worker, args=(4, _test_rev_of_fwd_ones), nprocs=4)


# ======================================================================
# Test 4: adjoint consistency for halo_forward_exchange
#   <y, fwd(x)>_global == <fwd.backward(y), x>_global
# ======================================================================


def _test_forward_adjoint(rank: int, world_size: int) -> None:
    local_pos, meta, config = _build_rank_halo(rank, world_size)
    n_owned = local_pos.shape[0]
    feat_dim = 3

    gen = torch.Generator().manual_seed(100 + rank)
    x = torch.randn(
        (n_owned, feat_dim), dtype=torch.float64, generator=gen, requires_grad=True
    )

    padded = halo_forward_exchange(x, meta, config)
    y = torch.randn(
        padded.shape, dtype=torch.float64, generator=gen
    )  # cotangent on padded

    local_lhs = (y * padded).sum()
    (grad_x,) = torch.autograd.grad(local_lhs, x, retain_graph=False)
    local_rhs = (grad_x * x.detach()).sum()

    global_lhs = local_lhs.detach().clone()
    global_rhs = local_rhs.detach().clone()
    dist.all_reduce(global_lhs, op=dist.ReduceOp.SUM)
    dist.all_reduce(global_rhs, op=dist.ReduceOp.SUM)

    torch.testing.assert_close(global_lhs, global_rhs, rtol=1e-10, atol=1e-10)


def test_forward_adjoint_2ranks() -> None:
    mp.spawn(_worker, args=(2, _test_forward_adjoint), nprocs=2)


def test_forward_adjoint_4ranks() -> None:
    mp.spawn(_worker, args=(4, _test_forward_adjoint), nprocs=4)


# ======================================================================
# Test 5: adjoint consistency for halo_reverse_exchange
#   <z, rev(padded)>_global == <rev.backward(z), padded>_global
# ======================================================================


def _test_reverse_adjoint(rank: int, world_size: int) -> None:
    _local_pos, meta, config = _build_rank_halo(rank, world_size)
    feat_dim = 3

    gen = torch.Generator().manual_seed(200 + rank)
    padded = torch.randn(
        (meta.n_padded, feat_dim),
        dtype=torch.float64,
        generator=gen,
        requires_grad=True,
    )

    owned = halo_reverse_exchange(padded, meta, config)
    z = torch.randn(owned.shape, dtype=torch.float64, generator=gen)

    local_lhs = (z * owned).sum()
    (grad_padded,) = torch.autograd.grad(local_lhs, padded, retain_graph=False)
    local_rhs = (grad_padded * padded.detach()).sum()

    global_lhs = local_lhs.detach().clone()
    global_rhs = local_rhs.detach().clone()
    dist.all_reduce(global_lhs, op=dist.ReduceOp.SUM)
    dist.all_reduce(global_rhs, op=dist.ReduceOp.SUM)

    torch.testing.assert_close(global_lhs, global_rhs, rtol=1e-10, atol=1e-10)


def test_reverse_adjoint_2ranks() -> None:
    mp.spawn(_worker, args=(2, _test_reverse_adjoint), nprocs=2)


def test_reverse_adjoint_4ranks() -> None:
    mp.spawn(_worker, args=(4, _test_reverse_adjoint), nprocs=4)


# ======================================================================
# Test 6: directional finite-difference vs autograd (single perturbation
# direction per rank → 2 collective forwards, cheap sanity on top of adjoint)
# ======================================================================


def _test_directional_fd_vs_autograd(rank: int, world_size: int) -> None:
    local_pos, meta, config = _build_rank_halo(rank, world_size)
    n_owned = local_pos.shape[0]
    feat_dim = 2

    gen = torch.Generator().manual_seed(300 + rank)
    x = torch.randn(
        (n_owned, feat_dim), dtype=torch.float64, generator=gen, requires_grad=True
    )
    dx = torch.randn((n_owned, feat_dim), dtype=torch.float64, generator=gen)

    def local_loss(features: torch.Tensor) -> torch.Tensor:
        padded = halo_forward_exchange(features, meta, config)
        return (padded**2).sum()

    loss = local_loss(x)
    (autograd_grad,) = torch.autograd.grad(loss, x, retain_graph=False)

    h = 1e-6
    with torch.no_grad():
        l_p = local_loss(x + h * dx)
        dist.all_reduce(l_p, op=dist.ReduceOp.SUM)
        l_m = local_loss(x - h * dx)
        dist.all_reduce(l_m, op=dist.ReduceOp.SUM)

    directional_fd = (l_p - l_m) / (2 * h)
    directional_autograd = (autograd_grad * dx).sum()
    dist.all_reduce(directional_autograd, op=dist.ReduceOp.SUM)

    torch.testing.assert_close(
        directional_autograd, directional_fd, rtol=1e-5, atol=1e-5
    )


def test_directional_fd_vs_autograd_2ranks() -> None:
    mp.spawn(_worker, args=(2, _test_directional_fd_vs_autograd), nprocs=2)
