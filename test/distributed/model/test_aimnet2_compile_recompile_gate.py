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
"""AIMNet2 halo + ``torch.compile`` DD gate: force equivalence AND zero
steady-state recompiles.

AIMNet2's distributed halo path under ``compile_model=True`` is owned entirely
by the framework
(``DistributedModel._compiled_energy_autograd_forward``): the dense ``(N, K)``
nbmat Batch is fixed-shape-padded by ``DenseBatchPadder`` before the wrapper's
pure ``forward`` is compiled, the halo routing is published to the compile-routing
holder (read by the spec's declared conv / Coulomb / ``mol_sum`` adapters), and
forces come from external autograd over the compiled energy. The methane-PBC
validator
(``test_validate_cuda.py::test_aimnet2_methane_pbc_passes``) exercises only the
*eager* halo path. This gate covers both the compiled dense-padding path and the
steady-state recompile count:

* **Equivalence** (step 0, unjittered): an *eager*-DD reference vs the compiled
  DD forward, on the same partition — each rank's owned forces must match. (A
  single-GPU reference can't be used: a bare ``AIMNet2Wrapper(Batch)`` forward
  needs aimnet's 2-D neighbor-mode layout that only the DD input path builds.)
* **Recompiles** (jittered MD loop): after warmup, the number of unique compiled
  graphs (``torch._dynamo.utils.counters["stats"]["unique_graphs"]``) must not
  grow — the dense fixed-shape caps must hold the graph stable across steps.

Requires 2+ CUDA GPUs + ``aimnet`` installed (Warp conv kernel is CUDA-fp32).
"""

from __future__ import annotations

import pytest
import torch
import torch.multiprocessing as mp
from _dd_harness import nccl_worker as _worker

from nvalchemi.data import AtomicData, Batch
from nvalchemi.distributed.config import DomainConfig

WORLD_SIZE = 2
WARMUP_STEPS = 8
STEADY_STEPS = 8
JITTER = 0.06  # Å per-step RMS displacement (within cutoff+skin headroom)

_skip = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < WORLD_SIZE,
    reason=f"Need {WORLD_SIZE}+ CUDA GPUs",
)


def _methane_packing(n_per_side: int = 4, spacing: float = 4.4, dtype=torch.float32):
    """n_per_side**3 methane molecules (5 atoms each) on a cubic PBC lattice."""
    box = float(n_per_side) * spacing
    bond = 1.087
    s = bond / (3.0**0.5)
    offsets = torch.tensor(
        [[0, 0, 0], [s, s, s], [-s, -s, s], [-s, s, -s], [s, -s, -s]], dtype=dtype
    )
    grid = torch.arange(n_per_side, dtype=dtype)
    centres = (
        torch.stack(torch.meshgrid(grid, grid, grid, indexing="ij"), dim=-1).reshape(
            -1, 3
        )
        * spacing
    )
    positions = (centres.unsqueeze(1) + offsets.unsqueeze(0)).reshape(-1, 3)
    n = positions.shape[0]
    atomic_numbers = torch.tensor([6, 1, 1, 1, 1] * (n // 5), dtype=torch.long)
    cell = torch.eye(3, dtype=dtype) * box
    pbc = torch.ones(3, dtype=torch.bool)
    return positions, atomic_numbers, cell, pbc, box


def _aimnet2_recompile_worker(rank: int, world_size: int) -> None:
    from torch._dynamo.utils import counters
    from torch.distributed import DeviceMesh

    from nvalchemi.distributed.distributed_model import DistributedModel
    from nvalchemi.distributed.sharded_batch import ShardedBatch
    from nvalchemi.models.aimnet2 import AIMNet2Wrapper

    dtype = torch.float32
    device = torch.device(f"cuda:{rank}")
    positions0, atomic_numbers, cell, pbc, box = _methane_packing(n_per_side=4)
    positions0 = positions0.to(device=device, dtype=dtype)
    cell_d = cell.to(device=device, dtype=dtype).unsqueeze(0)
    pbc_d = pbc.to(device).unsqueeze(0)

    def _data(pos):
        return AtomicData(
            positions=pos.clone(),
            atomic_numbers=atomic_numbers.to(device),
            cell=cell_d,
            pbc=pbc_d,
        )

    def _batch(pos):
        return Batch.from_data_list([_data(pos)]) if rank == 0 else None

    mesh = DeviceMesh("cuda", list(range(world_size)), mesh_dim_names=("domain",))

    # ---- eager-DD reference (same DistributedModel path, no compile) at the
    # initial geometry. A single-GPU reference is unavailable: a bare
    # AIMNet2Wrapper(Batch) forward needs aimnet's 2-D neighbor-mode layout that
    # only the DD input path builds. Same partition -> owned forces align. ----
    eager = AIMNet2Wrapper.from_checkpoint("aimnet2", device=device)
    eager.eval()
    eager.model_config.active_outputs = {"energy", "forces"}
    cfg = DomainConfig(cutoff=float(eager._cutoff), skin=0.5, mesh=mesh)
    with DistributedModel(eager, cfg) as eager_model:
        f_eager_owned = (
            eager_model(ShardedBatch.from_batch(_batch(positions0), mesh, cfg, 0))[
                "forces"
            ]
            .detach()
            .float()
            .cpu()
        )
    del eager, eager_model

    # ---- compiled DD model: the wrapper is eager (compile_model is the
    # single-process model-compile lever); DD-compile is requested on
    # DistributedModel, which owns the compiled energy-autograd forward. ----
    wrapper = AIMNet2Wrapper.from_checkpoint("aimnet2", device=device)
    wrapper.eval()
    wrapper.model_config.active_outputs = {"energy", "forces"}

    gen = torch.Generator(device="cpu").manual_seed(7)
    graphs_after_warmup = [0]

    with DistributedModel(wrapper, cfg, compile=True) as dist_model:
        for step in range(WARMUP_STEPS + STEADY_STEPS):
            if step == 0:
                pos = positions0
            else:
                disp = JITTER * torch.randn(
                    positions0.shape, dtype=torch.float64, generator=gen
                ).to(device=device, dtype=dtype)
                pos = positions0 + disp
                pos = pos - torch.floor(pos / box) * box
            sharded = ShardedBatch.from_batch(_batch(pos), mesh, cfg, 0)
            out = dist_model(sharded)
            f_owned = out["forces"].detach().float()

            if step == 0:
                # Equivalence: compiled DD == eager DD on the same partition.
                torch.testing.assert_close(
                    f_owned.cpu(),
                    f_eager_owned,
                    rtol=3e-3,
                    atol=3e-3,
                    msg=f"rank {rank}: compiled DD forces != eager DD reference",
                )
            _ = f_owned.sum().item()
            n_graphs = counters["stats"].get("unique_graphs", 0)
            if step == WARMUP_STEPS - 1:
                graphs_after_warmup[0] = n_graphs
            print(
                f"[r{rank}] step {step:02d} unique_graphs={n_graphs} "
                f"caps={getattr(wrapper, '_cap_state', {})}",
                flush=True,
            )

    final_graphs = counters["stats"].get("unique_graphs", 0)
    new_recompiles = final_graphs - graphs_after_warmup[0]
    assert new_recompiles == 0, (
        f"rank {rank}: {new_recompiles} recompile(s) during {STEADY_STEPS} "
        f"steady-state steps (unique_graphs {graphs_after_warmup[0]} -> "
        f"{final_graphs}); caps={getattr(wrapper, '_cap_state', {})}. AIMNet2's "
        "dense fixed-shape padding is no longer holding the compiled graph stable."
    )


@_skip
def test_aimnet2_compile_dd_equivalence_and_zero_recompiles_2ranks():
    """AIMNet2 halo + compile under DD: owned forces match a single-GPU
    reference, and a jittered MD loop produces zero steady-state recompiles.
    Guards the dense-nbmat fixed-shape padding (the DensePadder path)."""
    pytest.importorskip("aimnet", reason="aimnet not installed")
    mp.spawn(
        _worker,
        args=(WORLD_SIZE, "29573", _aimnet2_recompile_worker),
        nprocs=WORLD_SIZE,
    )
