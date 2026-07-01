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

"""Per-system reduce under torch.compile via the nvalchemi::per_system_reduce
custom op. The eager __torch_function__ reduce handler (_PerSystemReduceSum) is
bypassed under compile; the scatter routes through __torch_dispatch__ to the
custom op, which carries the cross-rank all_reduce adjoint in register_autograd.
Asserts compiled == eager (forward AND backward), == the global per-system sum,
and the correct all_reduced grad. 2-rank gloo."""
import pytest

import os
import sys
import traceback

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.device_mesh import DeviceMesh


def _worker_per_system(rank, world_size):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29681"
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
    try:
        import nvalchemi.distributed  # noqa: F401
        from nvalchemi.distributed._core.shard_tensor import ShardTensor
        from nvalchemi.distributed.spec import SPEC_MPNN_HALO

        mesh = DeviceMesh("cpu", list(range(world_size)), mesh_dim_names=("dom",))
        n_systems, n_atoms, F = 2, 6, 3
        torch.manual_seed(400 + rank)
        per_atom0 = torch.randn(n_atoms, F, dtype=torch.float64)
        sys_idx = torch.randint(0, n_systems, (n_atoms,), dtype=torch.long)
        idx_expanded = sys_idx.unsqueeze(-1).expand(-1, F).contiguous()

        import types as _types

        # Minimal config: the eager _PerSystemReduceSum reads only config.mesh
        # (the compiled custom op uses the default group).
        cfg = _types.SimpleNamespace(mesh=mesh, rank=rank)

        def make_acc():
            with mesh:
                return ShardTensor.wrap(
                    torch.zeros(n_systems, F, dtype=torch.float64),
                    n_systems=n_systems,
                    spec=SPEC_MPNN_HALO,
                    config=cfg,
                )

        def fn(acc, per_atom):
            return acc.scatter_add_(0, idx_expanded, per_atom)

        y = torch.randn(n_systems, F, dtype=torch.float64)

        # eager reference (TF -> _PerSystemReduceSum)
        pa_e = per_atom0.clone().requires_grad_(True)
        with mesh:
            out_e = fn(make_acc(), pa_e)
            energy_e = (y * out_e).sum()
        (grad_e,) = torch.autograd.grad(energy_e, pa_e)
        out_e_local = out_e.unwrap()

        # compiled (dispatch -> nvalchemi::per_system_reduce)
        torch._dynamo.reset()
        pa_c = per_atom0.clone().requires_grad_(True)
        cf = torch.compile(fn, backend="eager", fullgraph=True)
        with mesh:
            out_c = cf(make_acc(), pa_c)
            energy_c = (y * out_c).sum()
        (grad_c,) = torch.autograd.grad(energy_c, pa_c)
        out_c_local = out_c.unwrap()

        torch.testing.assert_close(out_c_local, out_e_local, rtol=1e-9, atol=1e-9)
        torch.testing.assert_close(grad_c, grad_e, rtol=1e-9, atol=1e-9)

        # Non-vacuous: result must equal the GLOBAL per-system sum (all ranks).
        all_pa = [torch.zeros_like(per_atom0) for _ in range(world_size)]
        all_idx = [torch.zeros_like(sys_idx) for _ in range(world_size)]
        dist.all_gather(all_pa, per_atom0)
        dist.all_gather(all_idx, sys_idx)
        ref = torch.zeros(n_systems, F, dtype=torch.float64)
        for r in range(world_size):
            ref.scatter_add_(
                0, all_idx[r].unsqueeze(-1).expand(-1, F), all_pa[r]
            )
        torch.testing.assert_close(out_e_local, ref, rtol=1e-9, atol=1e-9)
        # grad sanity: out is replicated (forward all_reduce), so the backward
        # all_reduces the upstream grad y; grad = (Σ_r y_r)[sys_idx].
        y_global = y.clone()
        dist.all_reduce(y_global, op=dist.ReduceOp.SUM)
        torch.testing.assert_close(
            grad_e, y_global.index_select(0, sys_idx), rtol=1e-9, atol=1e-9
        )
        print(
            f"[r{rank}] MATCH eager==compiled (fwd+bwd); ==global-sum; grad==y[idx]. "
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
def test_compile_per_system_2ranks() -> None:
    mp.spawn(_worker_per_system, args=(2,), nprocs=2)
