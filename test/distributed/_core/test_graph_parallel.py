# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Graph-parallel execution gate on a toy MACE-style MPNN.

A small message-passing model (embedding -> L layers of gather-sender / MLP /
scatter-to-receiver -> per-graph energy sum) runs two ways and must agree:

* single-process over all atoms;
* graph-parallel over a balanced atom-index partition, where each rank owns a
  contiguous atom slice plus the edges into its atoms, all-gathers the node
  features to a replicated tensor every layer (:func:`gather_to_replicate`), and
  all-reduces its owned per-graph energy.

The forward is an ordinary single-device MPNN; only the runner adds the
partition + replicate + reduce. Energy and per-atom forces must match the
single-process reference, exercising the replicate collective and its adjoint
under autograd.
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn

from nvalchemi.distributed._core.context import (
    DistributedContext,
    activate_dd_context,
)
from nvalchemi.distributed._core.placement import ShardRouting
from nvalchemi.distributed._core.storage_policy import GraphParallelPolicy
from nvalchemi.distributed.config import DomainConfig
from nvalchemi.distributed.helpers import refresh_neighbors
from nvalchemi.distributed.partitioner import IndexPartitioner


class _ToyMPNN(nn.Module):
    def __init__(self, n_layers: int = 2, hidden: int = 8, n_types: int = 4) -> None:
        super().__init__()
        self.embed = nn.Embedding(n_types, hidden)
        self.layers = nn.ModuleList(
            nn.Sequential(nn.Linear(hidden + 1, hidden), nn.SiLU(),
                          nn.Linear(hidden, hidden))
            for _ in range(n_layers)
        )
        self.readout = nn.Linear(hidden, 1)

    def message(self, layer: nn.Module, h_src: torch.Tensor, pos_src: torch.Tensor,
                pos_dst: torch.Tensor) -> torch.Tensor:
        edge_len = (pos_src - pos_dst).norm(dim=1, keepdim=True)
        return layer(torch.cat([h_src, edge_len], dim=1))

    def node_energy(self, h: torch.Tensor) -> torch.Tensor:
        return self.readout(h).squeeze(1)


def _toy_inputs(n_atoms: int, n_sys: int, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    z = torch.randint(0, 4, (n_atoms,), generator=g)
    pos = torch.randn(n_atoms, 3, dtype=torch.float64, generator=g)
    batch = (torch.arange(n_atoms) * n_sys // n_atoms).clamp(max=n_sys - 1)
    # A dense-ish random graph; both endpoints global.
    e = max(4 * n_atoms, 8)
    snd = torch.randint(0, n_atoms, (e,), generator=g)
    rcv = torch.randint(0, n_atoms, (e,), generator=g)
    edge_index = torch.stack([snd, rcv])  # (2, E)
    return z, pos, edge_index, batch


def _single_process(model: _ToyMPNN, z, pos, edge_index, batch, n_sys):
    pos = pos.clone().requires_grad_(True)
    h = model.embed(z)
    s, r = edge_index[0], edge_index[1]
    for layer in model.layers:
        m = model.message(layer, h[s], pos[s], pos[r])
        h = h + torch.zeros_like(h).index_add(0, r, m)
    energy = torch.zeros(n_sys, dtype=h.dtype).index_add(0, batch, model.node_energy(h))
    (forces,) = torch.autograd.grad(energy.sum(), pos)
    return energy.detach(), -forces


def _graph_parallel(model, z, pos, edge_index, batch, n_sys, rank, world, mesh, group):
    # Balanced index partition + routing, the same seam DistributedModel uses.
    cfg = DomainConfig(cutoff=1.0, mesh=mesh)
    assignment = IndexPartitioner(cfg).assign_atoms_to_ranks(pos)
    meta = ShardRouting.from_assignment(assignment, rank, world)
    offset = sum(int((assignment == r).sum()) for r in range(rank))
    n_owned = meta.n_owned
    owned = slice(offset, offset + n_owned)

    pos_owned = pos[owned].clone().requires_grad_(True)
    z_owned, batch_owned = z[owned], batch[owned]

    # Edges into this rank's atoms: sender stays global, receiver -> owned-local.
    s_g, r_g = edge_index[0], edge_index[1]
    keep = (r_g >= offset) & (r_g < offset + n_owned)
    s_g = s_g[keep]
    r_loc = r_g[keep] - offset

    ctx = DistributedContext(mesh=mesh, gather_meta=meta, policy=GraphParallelPolicy())
    with activate_dd_context(ctx):
        pos_full = refresh_neighbors(pos_owned)  # (N, 3)
        h = model.embed(z_owned)  # (n_owned, H)
        for layer in model.layers:
            h_full = refresh_neighbors(h)  # (N, H)
            m = model.message(
                layer, h_full[s_g], pos_full[s_g], pos_full[offset + r_loc]
            )
            h = h + torch.zeros_like(h).index_add(0, r_loc, m)
        energy = torch.zeros(n_sys, dtype=h.dtype).index_add(
            0, batch_owned, model.node_energy(h)
        )
        dist.all_reduce(energy, op=dist.ReduceOp.SUM, group=group)
    (g_owned,) = torch.autograd.grad(energy.sum(), pos_owned)
    return energy.detach(), -g_owned, offset, n_owned


def _worker(rank: int, world: int) -> None:
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29901")
    dist.init_process_group("gloo", rank=rank, world_size=world)
    from torch.distributed.device_mesh import init_device_mesh

    mesh = init_device_mesh("cpu", (world,))
    group = dist.group.WORLD

    torch.manual_seed(0)
    n_atoms, n_sys = 24, 3
    model = _ToyMPNN().double()
    z, pos, edge_index, batch = _toy_inputs(n_atoms, n_sys)

    e_ref, f_ref = _single_process(model, z, pos, edge_index, batch, n_sys)
    e_gp, f_owned, offset, n_owned = _graph_parallel(
        model, z, pos, edge_index, batch, n_sys, rank, world, mesh, group
    )

    torch.testing.assert_close(e_gp, e_ref, rtol=1e-9, atol=1e-9)
    torch.testing.assert_close(f_owned, f_ref[offset : offset + n_owned],
                               rtol=1e-8, atol=1e-8)
    if rank == 0:
        print(f"[gp w={world}] energy + owned forces match single-process")
    dist.barrier()
    dist.destroy_process_group()


def test_graph_parallel_toy_mpnn_2ranks() -> None:
    mp.spawn(_worker, args=(2,), nprocs=2)


def test_graph_parallel_toy_mpnn_3ranks() -> None:
    mp.spawn(_worker, args=(3,), nprocs=3)


if __name__ == "__main__":
    for w in (2, 3):
        mp.spawn(_worker, args=(w,), nprocs=w)
