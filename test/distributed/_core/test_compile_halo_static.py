# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Fixed-shape inner halo via ``_halo_meta_packed`` (a 2nd ShardTensor
inner tensor) routed through ``nvalchemi::halo_forward_static``.

Validates, at 2-rank gloo over a real halo ShardTensor (``x.index_select`` ->
``_dispatch_halo_gather``):

  1. eager static path == eager ``list[int]`` reference (fwd + bwd) — the static
     op is a correct drop-in for the math;
  2. under ``torch.compile`` (aot_eager — exercises AOTAutograd, the production
     desugaring) the static path == the ``list[int]`` path (fwd + bwd) — correct
     drop-in under AOT;
  3. ``_halo_meta_packed`` is a genuine graph INPUT, not a baked constant:
     perturbing the recv-mask of the packed routing (same shape) changes the
     compiled output.

(We compare static-vs-list[int] *within* each execution mode rather than
compiled-vs-eager: the harness builds the feature ShardTensor via an eager outer
``halo_forward_exchange`` and differentiates across the eager->compiled boundary,
which uniformly scales grads under AOT for BOTH paths — an artifact of this
synthetic harness, not the ops. The real MACE path is covered by
``test_mace_cueq_multigpu``.)
"""

import os
import sys
import traceback

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.device_mesh import DeviceMesh


def _patch_all_to_all_for_gloo():
    import physicsnemo.distributed.utils as pn_utils

    def _gloo(tensor, indices, sizes, dim=0, group=None):
        comm = dist.get_world_size(group=group)
        rank = dist.get_rank(group=group)
        x_send = [tensor[idx].contiguous() for idx in indices]
        x_recv = []
        shape = list(tensor.shape)
        for r in range(comm):
            shape[dim] = sizes[r][rank]
            x_recv.append(torch.empty(shape, dtype=tensor.dtype, device=tensor.device))
        ops = []
        for r in range(comm):
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

    pn_utils.indexed_all_to_all_v_wrapper = _gloo


def _build_rank_halo(mesh, rank, world_size, ghost_width=5.0):
    from nvalchemi.distributed._core.halo_types import ParticleHaloConfig
    from nvalchemi.distributed._core.particle_halo import particle_halo_padding
    from nvalchemi.distributed.partitioner import DomainConfig, SpatialPartitioner

    n_side, lattice = 6, 3.4
    coords = torch.arange(n_side, dtype=torch.float64) * lattice
    gx, gy, gz = torch.meshgrid(coords, coords, coords, indexing="ij")
    positions = torch.stack([gx.flatten(), gy.flatten(), gz.flatten()], dim=-1)
    cell = torch.eye(3, dtype=torch.float64) * (n_side * lattice)
    pbc = torch.ones(3, dtype=torch.bool)
    dc = DomainConfig(cutoff=ghost_width, mesh=mesh)
    part = SpatialPartitioner(
        config=dc, cell_matrix=cell.unsqueeze(0), pbc=pbc.unsqueeze(0)
    )
    hc = ParticleHaloConfig(ghost_width=ghost_width, partitioner=part, mesh=mesh)
    assignment = part.assign_atoms_to_ranks(positions)
    local_pos = positions[assignment == rank].contiguous()
    _padded, meta = particle_halo_padding(local_pos, hc)
    return meta, hc


def _worker(rank, world_size):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29691"
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
    try:
        import nvalchemi.distributed  # noqa: F401
        from nvalchemi.distributed._core.particle_halo import (
            build_halo_meta_tensors,
            halo_forward_exchange,
            pack_halo_meta,
        )
        from nvalchemi.distributed._core.shard_tensor import ShardTensor
        from nvalchemi.distributed.spec import SPEC_MPNN_HALO

        _patch_all_to_all_for_gloo()
        mesh = DeviceMesh("cpu", list(range(world_size)), mesh_dim_names=("dom",))
        meta, config = _build_rank_halo(mesh, rank, world_size)
        n_owned, n_padded, F = meta.n_owned, meta.n_padded, 5
        assert n_padded > n_owned
        max_send = max(1, max(max(row) for row in meta.send_sizes))

        torch.manual_seed(321 + rank)
        owned0 = torch.randn(n_owned, F, dtype=torch.float64)
        gather_idx = torch.arange(n_padded)

        def make_packed(device):
            si, rd, rr, no = build_halo_meta_tensors(
                meta, rank, max_send, n_padded, device
            )
            return pack_halo_meta(si, rd, rr, no)

        def make_x(owned, packed):
            with mesh:
                padded = halo_forward_exchange(owned, meta, config)
                return ShardTensor.wrap(
                    padded,
                    meta=meta,
                    config=config,
                    spec=SPEC_MPNN_HALO,
                    halo_meta_packed=packed,
                )

        def fn(x):
            return x.index_select(0, gather_idx)

        def run(packed, compiled, backend="aot_eager"):
            o = owned0.clone().requires_grad_(True)
            f = torch.compile(fn, backend=backend, fullgraph=True) if compiled else fn
            with mesh:
                out = f(make_x(o, packed))
            (g,) = torch.autograd.grad(out.sum(), o)
            return out.unwrap().detach(), g

        # 1. eager: static == list[int] (fwd + bwd)
        out_es, g_es = run(make_packed(owned0.device), compiled=False)
        out_el, g_el = run(None, compiled=False)
        torch.testing.assert_close(out_es, out_el, rtol=1e-9, atol=1e-9)
        torch.testing.assert_close(g_es, g_el, rtol=1e-9, atol=1e-9)

        # 2. compiled (aot_eager): static == list[int] (fwd + bwd)
        torch._dynamo.reset()
        out_cs, g_cs = run(make_packed(owned0.device), compiled=True)
        torch._dynamo.reset()
        out_cl, g_cl = run(None, compiled=True)
        torch.testing.assert_close(out_cs, out_cl, rtol=1e-9, atol=1e-9)
        torch.testing.assert_close(g_cs, g_cl, rtol=1e-9, atol=1e-9)

        # 3. graph-input proof: perturb recv-mask of the packed routing (same
        # shape) -> compiled output changes (i.e. the routing is a runtime input,
        # not a constant baked at trace time).
        torch._dynamo.reset()
        cf = torch.compile(fn, backend="aot_eager", fullgraph=True)
        with mesh:
            base = (
                cf(make_x(owned0.clone(), make_packed(owned0.device))).unwrap().detach()
            )
            pert = make_packed(owned0.device).clone()
            wm = pert.shape[0] // 3
            pert[2 * wm :] = 0  # zero recv_real -> ghosts gather nothing
            out_p = cf(make_x(owned0.clone(), pert)).unwrap().detach()
        differs = (out_p - base).abs().max().item() > 1e-6
        assert differs, (
            "perturbed _halo_meta_packed gave the same result -> baked, not a graph input"
        )

        print(
            f"[r{rank}] OK static==list[int] (eager & aot_eager, fwd+bwd); graph-input verified",
            flush=True,
        )
    except Exception as e:  # noqa: BLE001
        print(f"[r{rank}] FAILED: {type(e).__name__}: {str(e)[:400]}", flush=True)
        for line in traceback.format_exc().splitlines()[-25:]:
            print(f"[r{rank}]   " + line, flush=True)
        sys.exit(1)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_gloo_available(), reason="gloo backend required")
def test_compile_halo_static_2ranks() -> None:
    mp.spawn(_worker, args=(2,), nprocs=2)


if __name__ == "__main__":
    mp.spawn(_worker, args=(2,), nprocs=2)
