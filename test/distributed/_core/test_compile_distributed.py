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

"""Distributed scatter_add / index_select on a sharded ShardTensor under
torch.compile via nvalchemi::distributed_scatter_add / ::distributed_index_select
custom ops. Compiled (dispatch) == eager (TF) for forward AND backward, ==
central reference. 2-rank gloo."""
import os
import sys
import traceback
import types

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


def _worker_distributed(rank, world_size):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29685"
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
    try:
        import nvalchemi.distributed  # noqa: F401
        from nvalchemi.distributed._core.placement import ShardRouting
        from nvalchemi.distributed._core.shard_tensor import ShardTensor
        from nvalchemi.distributed._core.spec import DistributionSpec
        from nvalchemi.distributed._core.storage_policy import PlainShard
        from nvalchemi.distributed.spec import MLIPSpec

        _patch_all_to_all_for_gloo()
        mesh = DeviceMesh("cpu", list(range(world_size)), mesh_dim_names=("dom",))
        per_rank, F = 3, 4
        n_global = per_rank * world_size
        assignment = torch.arange(n_global, dtype=torch.long) // per_rank
        gm = ShardRouting.from_assignment(assignment, rank=rank)
        cfg = types.SimpleNamespace(mesh=mesh, rank=rank)

        torch.manual_seed(600 + rank)
        K = 5
        gidx = torch.randint(0, n_global, (K,), dtype=torch.long)
        src0 = torch.randn(K, F, dtype=torch.float64)
        idx_exp = gidx.unsqueeze(-1).expand(-1, F).contiguous()

        def make_shard():
            with mesh:
                return ShardTensor.wrap(
                    torch.zeros(per_rank, F, dtype=torch.float64),
                    gather_meta=gm,
                    config=cfg,
                    spec=MLIPSpec(distribution=DistributionSpec(policy=PlainShard())),
                )

        def fn(shard, src):
            return shard.scatter_add_(0, idx_exp, src)

        # eager (TF)
        src_e = src0.clone().requires_grad_(True)
        with mesh:
            out_e = fn(make_shard(), src_e)
            energy_e = out_e.sum()
        (grad_e,) = torch.autograd.grad(energy_e, src_e)
        out_e_local = out_e.unwrap()

        # compiled (dispatch -> custom op)
        torch._dynamo.reset()
        src_c = src0.clone().requires_grad_(True)
        cf = torch.compile(fn, backend="eager", fullgraph=True)
        with mesh:
            out_c = cf(make_shard(), src_c)
            energy_c = out_c.sum()
        (grad_c,) = torch.autograd.grad(energy_c, src_c)
        out_c_local = out_c.unwrap()

        torch.testing.assert_close(out_c_local, out_e_local, rtol=1e-9, atol=1e-9)
        torch.testing.assert_close(grad_c, grad_e, rtol=1e-9, atol=1e-9)

        # central reference (gather all, scatter, slice my block)
        all_idx = [torch.zeros_like(gidx) for _ in range(world_size)]
        all_src = [torch.zeros_like(src0) for _ in range(world_size)]
        dist.all_gather(all_idx, gidx)
        dist.all_gather(all_src, src0)
        ref_full = torch.zeros(n_global, F, dtype=torch.float64)
        for r in range(world_size):
            ref_full.scatter_add_(
                0, all_idx[r].unsqueeze(-1).expand(-1, F), all_src[r]
            )
        expected = ref_full[rank * per_rank : (rank + 1) * per_rank]
        torch.testing.assert_close(out_e_local, expected, rtol=1e-9, atol=1e-9)
        print(
            f"[r{rank}] MATCH eager==compiled (fwd+bwd); ==central ref. "
            f"out.sum={out_e_local.sum().item():.4f}",
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
def test_compile_distributed_2ranks() -> None:
    mp.spawn(_worker_distributed, args=(2,), nprocs=2)
