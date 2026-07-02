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

"""World=2 force-equivalence gate for the async-overlap driver with a REAL
collective (gloo/CPU), for graph-parallel.

Drives :func:`overlapped_message` — the whole adapter body — through the strategy's
:meth:`GraphPartitionStrategy.async_exchange` (a real ``gather_to_replicate``
all-gather) + :meth:`locality_partition`, and checks each rank's owned message
output matches the single-process full-message reference. This exercises the
exchange + partition + two-pass driver together over an actual process group
(overlap OFF), the P0.5 keystone claim, without a GPU.
"""

from __future__ import annotations

import os

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def _message_fn(features: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    send, recv = edge_index[0], edge_index[1]
    edge_feat = torch.tanh(features[send]) * 0.7 + features[recv] * 0.3
    out = torch.zeros(
        features.shape[0], features.shape[1], dtype=features.dtype, device=features.device
    )
    return out.scatter_add_(0, recv.unsqueeze(-1).expand_as(edge_feat), edge_feat)


def _gp_worker(rank: int, world: int) -> None:
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29931")
    dist.init_process_group("gloo", rank=rank, world_size=world)
    try:
        from torch.distributed.device_mesh import DeviceMesh

        from nvalchemi.distributed._core.context import (
            DistributedContext,
            activate_dd_context,
        )
        from nvalchemi.distributed._core.overlap import overlapped_message
        from nvalchemi.distributed._core.placement import ShardRouting
        from nvalchemi.distributed._core.storage_policy import GraphParallelPolicy
        from nvalchemi.distributed.config import DomainConfig
        from nvalchemi.distributed.strategy import GraphPartitionStrategy

        n_global, feat = 12, 4
        counts = [n_global // world] * world
        full = torch.randn(n_global, feat, dtype=torch.float64, generator=torch.Generator().manual_seed(0))
        eg = torch.Generator().manual_seed(1)
        global_edges = torch.stack(
            [torch.randint(0, n_global, (80,), generator=eg),
             torch.randint(0, n_global, (80,), generator=eg)]
        )
        # Single-process reference: full message, this rank's owned receiver rows.
        reference = _message_fn(full, global_edges)
        offset = sum(counts[:rank])
        n_owned = counts[rank]
        ref_owned = reference[offset : offset + n_owned]

        # Layout: contiguous owned blocks → the all-gather reproduces `full`.
        owner_rank = torch.cat(
            [torch.full((counts[r],), r, dtype=torch.long) for r in range(world)]
        )
        local_index = torch.cat([torch.arange(counts[r]) for r in range(world)])
        meta = ShardRouting(
            n_owned=n_owned, n_global=n_global, owner_rank=owner_rank, local_index=local_index
        )
        # This rank owns the edges whose receiver is in its block; remap receiver
        # to owned-local. Senders stay global (they index the gathered full set).
        mask = (global_edges[1] >= offset) & (global_edges[1] < offset + n_owned)
        rank_edges = global_edges[:, mask].clone()
        rank_edges[1] -= offset
        owned = full[offset : offset + n_owned].clone()

        mesh = DeviceMesh("cpu", list(range(world)), mesh_dim_names=("domain",))
        cfg = DomainConfig(cutoff=5.0, mesh=mesh, mesh_dim="domain")
        strat = GraphPartitionStrategy(GraphParallelPolicy(), cfg, rank)
        ctx = DistributedContext(mesh=mesh, gather_meta=meta, strategy=strat)
        with activate_dd_context(ctx):
            out = overlapped_message(_message_fn, owned, rank_edges)

        assert out.shape == ref_owned.shape, (out.shape, ref_owned.shape)
        torch.testing.assert_close(out, ref_owned, rtol=1e-9, atol=1e-9)
        dist.barrier()
    finally:
        dist.destroy_process_group()


@pytest.mark.timeout(120)
def test_overlap_gate_graph_parallel_2ranks() -> None:
    mp.spawn(_gp_worker, args=(2,), nprocs=2, join=True)


# ======================================================================
# Halo gate — real halo_forward_exchange over a 2-rank spatial partition.
# Harness mirrors test_halo_autograd.py (gloo a2a shim + mock mesh); the
# claim is overlap == the non-overlapped padded forward (halo force-equiv).
# ======================================================================


def _patch_all_to_all_for_gloo() -> None:
    import physicsnemo.distributed.utils as pn_utils

    def _indexed_all_to_all_v_gloo(tensor, indices, sizes, dim=0, group=None):
        comm_size = dist.get_world_size(group=group)
        rank = dist.get_rank(group=group)
        x_send = [tensor[idx].contiguous() for idx in indices]
        x_recv = []
        shape = list(tensor.shape)
        for r in range(comm_size):
            shape[dim] = sizes[r][rank]
            x_recv.append(torch.empty(shape, dtype=tensor.dtype, device=tensor.device))
        ops = []
        for r in range(comm_size):
            if r == rank:
                x_recv[r].copy_(x_send[r])
            else:
                if x_send[r].numel() > 0:
                    ops.append(dist.isend(x_send[r], dst=r, group=group))
                if x_recv[r].numel() > 0:
                    ops.append(dist.irecv(x_recv[r], src=r, group=group))
        for op in ops:
            op.wait()
        return torch.cat(x_recv, dim=dim)

    pn_utils.indexed_all_to_all_v_wrapper = _indexed_all_to_all_v_gloo


class _MockMesh:
    def __init__(self, rank: int, world_size: int) -> None:
        self._rank = rank
        self._world_size = world_size

    def get_local_rank(self) -> int:
        return self._rank

    def size(self, dim: int | None = None) -> int:
        return self._world_size

    def get_group(self) -> object | None:
        return None


def _halo_worker(rank: int, world: int) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29932"
    dist.init_process_group("gloo", rank=rank, world_size=world)
    _patch_all_to_all_for_gloo()
    try:
        from nvalchemi.distributed._core.context import (
            DistributedContext,
            activate_dd_context,
        )
        from nvalchemi.distributed._core.overlap import overlapped_message
        from nvalchemi.distributed._core.particle_halo import (
            ParticleHaloConfig,
            halo_forward_exchange,
            particle_halo_padding,
        )
        from nvalchemi.distributed._core.storage_policy import HaloStoragePolicy
        from nvalchemi.distributed.config import DomainConfig
        from nvalchemi.distributed.partitioner import SpatialPartitioner
        from nvalchemi.distributed.strategy import HaloStrategy

        # Real 2-rank spatial partition → genuine halo metadata (with gnn_markers).
        n_side, lat = 6, 3.4
        coords = torch.arange(n_side, dtype=torch.float64) * lat
        gx, gy, gz = torch.meshgrid(coords, coords, coords, indexing="ij")
        positions = torch.stack([gx.flatten(), gy.flatten(), gz.flatten()], dim=-1)
        cell = torch.eye(3, dtype=torch.float64) * (n_side * lat)
        pbc = torch.ones(3, dtype=torch.bool)
        mesh = _MockMesh(rank, world)
        dcfg = DomainConfig(cutoff=5.0, mesh=mesh)
        part = SpatialPartitioner(
            config=dcfg, cell_matrix=cell.unsqueeze(0), pbc=pbc.unsqueeze(0)
        )
        hcfg = ParticleHaloConfig(ghost_width=5.0, partitioner=part, mesh=mesh)
        assignment = part.assign_atoms_to_ranks(positions)
        owned_pos = positions[assignment == rank].contiguous()
        _padded_pos, meta = particle_halo_padding(owned_pos, hcfg)
        n_owned, n_padded = meta.n_owned, meta.n_padded

        feat = 4
        fg = torch.Generator().manual_seed(100 + rank)
        owned_features = torch.randn(n_owned, feat, dtype=torch.float64, generator=fg)
        eg = torch.Generator().manual_seed(7)
        edges = torch.stack(
            [torch.randint(0, n_padded, (200,), generator=eg),   # owned or ghost senders
             torch.randint(0, n_owned, (200,), generator=eg)]     # owned receivers
        )
        assert (edges[0] >= n_owned).any(), "need ghost senders for a real split"

        # Non-overlapped reference: full message on the borrowed [owned|ghost] set.
        padded_feats = halo_forward_exchange(owned_features, meta, hcfg)
        reference = _message_fn(padded_feats, edges)[:n_owned]

        # Overlapped: split into owned-sender + ghost-sender buckets.
        strat = HaloStrategy(HaloStoragePolicy(), dcfg, rank)
        ctx = DistributedContext(
            mesh=mesh, halo_meta=meta, halo_config=hcfg, strategy=strat
        )
        with activate_dd_context(ctx):
            out = overlapped_message(_message_fn, owned_features, edges)

        assert out.shape == reference.shape, (out.shape, reference.shape)
        torch.testing.assert_close(out, reference, rtol=1e-9, atol=1e-9)
        dist.barrier()
    finally:
        dist.destroy_process_group()


@pytest.mark.timeout(120)
def test_overlap_gate_halo_2ranks() -> None:
    mp.spawn(_halo_worker, args=(2,), nprocs=2, join=True)
