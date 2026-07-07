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

"""Multi-GPU regression: composed-model DD under ``torch.compile``.

Gates per-model compilation of a composite: MACE (shared-autograd) compiles via
its framework-owned compiled energy-autograd DD path while DFT-D3 (kernel force)
runs eager — the composite glue stays eager. Two properties:

* **Equivalence** — the compiled 2-rank composite reproduces the single-GPU
  pipeline's owned forces (rattled NaCl) at the same geometry.
* **No steady-state recompiles** — across a jittered MD loop the number of unique
  compiled graphs must not grow after warmup; the persistent per-model
  ``DistributedModel`` instances hold their fixed-shape caps across steps.

Requires 2+ CUDA GPUs, ``mace-torch``, and ``nvalchemiops`` (DFTD3 kernels).
"""

from __future__ import annotations

import os
import warnings
from typing import Any

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from nvalchemi.data import AtomicData, Batch
from nvalchemi.distributed.config import DomainConfig

WORLD_SIZE = 2
WARMUP_STEPS = 4
STEADY_STEPS = 4
JITTER = 0.05
_A1, _A2, _S8 = 0.4289, 4.4407, 0.7875
_DFTD3_CUT = 5.0
_SKIN = 4.0

_skip = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < WORLD_SIZE,
    reason=f"Need {WORLD_SIZE}+ CUDA GPUs",
)


def _init_pg(rank: int, world_size: int, port: str) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = port
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(rank)
    torch.cuda.set_device(rank)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)


def _worker(rank: int, world_size: int, port: str, fn: Any, *args: Any) -> None:
    _init_pg(rank, world_size, port)
    try:
        fn(rank, world_size, *args)
    finally:
        dist.destroy_process_group()


def _build_lattice(dtype: torch.dtype = torch.float64, seed: int = 0):
    n_side = int(os.environ.get("NVALCHEMI_PIPE_N_SIDE", 16))
    box = float(os.environ.get("NVALCHEMI_PIPE_BOX", 48.0))
    coords = torch.arange(n_side, dtype=dtype) * (box / n_side)
    gx, gy, gz = torch.meshgrid(coords, coords, coords, indexing="ij")
    positions = torch.stack([gx.flatten(), gy.flatten(), gz.flatten()], dim=-1)
    n = positions.shape[0]
    g = torch.Generator().manual_seed(seed)
    positions = positions + 0.15 * torch.randn(
        positions.shape, dtype=dtype, generator=g
    )
    positions = positions % box
    sign = torch.ones(n, dtype=dtype)
    sign[1::2] = -1.0
    atomic_numbers = torch.where(
        sign > 0,
        torch.full((n,), 11, dtype=torch.long),
        torch.full((n,), 17, dtype=torch.long),
    )
    masses = torch.where(
        sign > 0,
        torch.full((n,), 22.99, dtype=dtype),
        torch.full((n,), 35.45, dtype=dtype),
    )
    cell = torch.eye(3, dtype=dtype) * box
    pbc = torch.ones(3, dtype=torch.bool)
    return positions, atomic_numbers, masses, cell, pbc, box


def _make_data(atomic_numbers, positions, masses, cell, pbc, device, dtype):
    n = positions.shape[0]
    return AtomicData(
        atomic_numbers=atomic_numbers.to(device),
        positions=positions.to(device=device, dtype=dtype).clone(),
        atomic_masses=masses.to(device=device, dtype=dtype),
        cell=cell.to(device=device, dtype=dtype).unsqueeze(0),
        pbc=pbc.to(device).unsqueeze(0),
        forces=torch.zeros(n, 3, device=device, dtype=dtype),
        energy=torch.zeros(1, 1, device=device, dtype=dtype),
    )


def _build_pipeline(mace_cut_holder: list[float], device, dtype):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from nvalchemi.models.mace import MACEWrapper

    from nvalchemi.models.dftd3 import DFTD3ModelWrapper
    from nvalchemi.models.pipeline import PipelineGroup, PipelineModelWrapper

    mace = MACEWrapper.from_checkpoint(
        "small", device=device, dtype=dtype, enable_cueq=False
    )
    mace_cut_holder[0] = float(mace.cutoff)
    dftd3 = DFTD3ModelWrapper(a1=_A1, a2=_A2, s8=_S8, cutoff=_DFTD3_CUT)
    return PipelineModelWrapper(
        groups=[
            PipelineGroup(steps=[mace], use_autograd=True),
            PipelineGroup(steps=[dftd3], use_autograd=False),
        ]
    )


def _compile_worker(rank: int, world_size: int) -> None:
    from torch._dynamo.utils import counters
    from torch.distributed import DeviceMesh

    from nvalchemi.distributed.distributed_pipeline import DistributedPipelineModel
    from nvalchemi.distributed.partitioner import SpatialPartitioner
    from nvalchemi.distributed.sharded_batch import ShardedBatch
    from nvalchemi.neighbors import compute_neighbors

    dtype = torch.float64
    device = torch.device(f"cuda:{rank}")
    positions0, an, masses, cell, pbc, box = _build_lattice(dtype=dtype)
    n_global = positions0.shape[0]
    mace_cut = [0.0]

    # --- Single-GPU reference on rank 0 at the initial (rattled) geometry ---
    e_ref = torch.zeros(1, dtype=dtype, device=device)
    f_ref = torch.zeros(n_global, 3, dtype=dtype, device=device)
    if rank == 0:
        ref_pipe = _build_pipeline(mace_cut, device, dtype)
        batch = Batch.from_data_list(
            [_make_data(an, positions0, masses, cell, pbc, device, dtype)]
        )
        compute_neighbors(batch, config=ref_pipe.model_config.neighbor_config)
        out = ref_pipe(batch)
        e_ref.copy_(out["energy"].sum().detach().view(1))
        f_ref.copy_(out["forces"].detach())
        del ref_pipe
    dist.broadcast(e_ref, src=0)
    dist.broadcast(f_ref, src=0)

    pipeline = _build_pipeline(mace_cut, device, dtype)
    max_cut = max(mace_cut[0], _DFTD3_CUT)
    mesh = DeviceMesh("cuda", list(range(world_size)), mesh_dim_names=("domain",))
    base_config = DomainConfig(cutoff=max_cut, skin=_SKIN, mesh=mesh)
    partitioner = SpatialPartitioner(
        config=base_config,
        cell_matrix=cell.to(device=device, dtype=dtype).unsqueeze(0),
        pbc=pbc.to(device).unsqueeze(0),
    )

    def _sharded(pos):
        full = (
            Batch.from_data_list(
                [_make_data(an, pos, masses, cell, pbc, device, dtype)]
            )
            if rank == 0
            else None
        )
        return ShardedBatch.from_batch(batch=full, mesh=mesh, config=base_config, src=0)

    gen = torch.Generator(device="cpu").manual_seed(11)
    graphs_after_warmup = [0]

    with DistributedPipelineModel(pipeline, base_config, compile=True) as dpm:
        for step in range(WARMUP_STEPS + STEADY_STEPS):
            if step == 0:
                pos = positions0
            else:
                disp = JITTER * torch.randn(
                    positions0.shape, dtype=dtype, generator=gen
                ).to(device)
                pos = (positions0.to(device) + disp) % box
            out = dpm(_sharded(pos))
            if step == 0:
                f_owned = out["forces"].detach()
                e_local = out["energy"].sum().detach()
                local_mask = (
                    partitioner.assign_atoms_to_ranks(
                        positions0.to(device=device, dtype=dtype)
                    )
                    == rank
                )
                f_ref_owned = f_ref[local_mask]
                de = (e_local - e_ref).abs().item()
                df = (f_owned - f_ref_owned).abs().max().item()
                print(
                    f"[pipe-compile rank {rank}] step0 ΔE={de:.3e} |ΔF|max={df:.3e} "
                    f"n_owned={f_owned.shape[0]} mace_cut={mace_cut[0]:.2f}",
                    flush=True,
                )
                torch.testing.assert_close(
                    e_local.view(1).to(e_ref.dtype),
                    e_ref,
                    rtol=1e-5,
                    atol=1e-2,
                    msg=f"rank {rank}: compiled composite energy mismatch ΔE={de:.3e}",
                )
                torch.testing.assert_close(
                    f_owned,
                    f_ref_owned,
                    rtol=1e-3,
                    atol=2e-4,
                    msg=f"rank {rank}: compiled composite forces mismatch |ΔF|max={df:.3e}",
                )
            if step == WARMUP_STEPS - 1:
                graphs_after_warmup[0] = counters["stats"]["unique_graphs"]

    final_graphs = counters["stats"]["unique_graphs"]
    print(
        f"[pipe-compile rank {rank}] unique_graphs warmup={graphs_after_warmup[0]} "
        f"final={final_graphs}",
        flush=True,
    )
    assert final_graphs == graphs_after_warmup[0], (
        f"rank {rank}: compiled composite recompiled in steady state "
        f"({graphs_after_warmup[0]} -> {final_graphs})"
    )


@_skip
def test_distributed_pipeline_compile_mace_dftd3_2ranks():
    """Compiled ``DistributedPipelineModel(MACE + DFTD3)`` == single-GPU; no steady recompiles."""
    pytest.importorskip("nvalchemiops", reason="nvalchemiops not installed")
    pytest.importorskip("mace", reason="mace-torch not installed")
    mp.spawn(
        _worker,
        args=(WORLD_SIZE, "29588", _compile_worker),
        nprocs=WORLD_SIZE,
    )
