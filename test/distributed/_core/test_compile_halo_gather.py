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

"""Halo gather (index_select on a halo ShardTensor) under torch.compile
via the nvalchemi::halo_forward custom op. Compiled (dispatch) == eager (TF) for
forward AND backward. 2-rank gloo."""

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


def _worker_halo_gather(rank, world_size):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29683"
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
    try:
        import nvalchemi.distributed  # noqa: F401
        from nvalchemi.distributed._core.shard_tensor import ShardTensor
        from nvalchemi.distributed.spec import SPEC_MPNN_HALO

        _patch_all_to_all_for_gloo()
        mesh = DeviceMesh("cpu", list(range(world_size)), mesh_dim_names=("dom",))
        meta, config = _build_rank_halo(mesh, rank, world_size)
        n_owned, n_padded, F = meta.n_owned, meta.n_padded, 5  # F!=3 (avoid coord skip)
        assert n_padded > n_owned

        torch.manual_seed(200 + rank)
        owned0 = torch.randn(n_owned, F, dtype=torch.float64)
        gather_idx = torch.arange(n_padded)  # gather every padded row

        def make_x(owned):
            # padded [owned | stale-halo]; the gather refreshes the halo.
            with mesh:
                from nvalchemi.distributed._core.particle_halo import (
                    halo_forward_exchange,
                )

                padded = halo_forward_exchange(owned, meta, config)
                return ShardTensor.wrap(
                    padded, meta=meta, config=config, spec=SPEC_MPNN_HALO
                )

        def fn(x):
            return x.index_select(0, gather_idx)

        owned_e = owned0.clone().requires_grad_(True)
        with mesh:
            out_e = fn(make_x(owned_e))
            energy_e = out_e.sum()
        (grad_e,) = torch.autograd.grad(energy_e, owned_e)
        out_e_local = out_e.unwrap()

        torch._dynamo.reset()
        owned_c = owned0.clone().requires_grad_(True)
        cf = torch.compile(fn, backend="eager", fullgraph=True)
        with mesh:
            out_c = cf(make_x(owned_c))
            energy_c = out_c.sum()
        (grad_c,) = torch.autograd.grad(energy_c, owned_c)
        out_c_local = out_c.unwrap()

        torch.testing.assert_close(out_c_local, out_e_local, rtol=1e-9, atol=1e-9)
        torch.testing.assert_close(grad_c, grad_e, rtol=1e-9, atol=1e-9)
        print(
            f"[r{rank}] MATCH eager==compiled (fwd+bwd). n_owned={n_owned} "
            f"n_padded={n_padded} out.sum={out_e_local.sum().item():.4f}",
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
def test_compile_halo_gather_2ranks() -> None:
    mp.spawn(_worker_halo_gather, args=(2,), nprocs=2)
