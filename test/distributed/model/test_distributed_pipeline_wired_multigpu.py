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

"""Multi-GPU regression: ``DistributedPipelineModel`` (C3, wired field).

Gates distributed composition of a **wired** shared-autograd group — AIMNet2
produces per-atom ``charges`` that PME consumes — the canonical C3 case
(``proposal-distributed-pipeline.md`` §C3). The consumer's energy depends on a
field the producer computes, so the two models form one coupled autograd graph
spanning the halo: PME's ghost charges are gathered (autograd-aware) from the
producer's owned charges, and the force chain ``-(dE_pme/dq)(dq/dr)`` rides that
exchange's backward back to the producing rank.

The 2-rank composite forward must reproduce, on each rank's owned atoms, the
**single-GPU pipeline** result (energy + per-atom forces incl. the charge
pathway). A perfect lattice would make this less discriminating, so the methane
packing is **rattled**.

AIMNet2 is organic-only, so the crystal is methane (C/H), not NaCl. AIMNet2 runs
in float32 internally, so tolerances are float32-scale.

Requires 2+ CUDA GPUs, ``aimnet2`` checkpoint, and ``nvalchemiops`` (PME kernels).
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
_PME_CUT = 6.0
_SKIN = 0.5

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


def _methane_packing(dtype: torch.dtype = torch.float32, seed: int = 0):
    """``n_per_side**3`` methane molecules (5 atoms each) on a cubic PBC lattice,
    rattled so net forces are non-trivial."""
    n_per_side = int(os.environ.get("NVALCHEMI_WIRED_N_SIDE", 4))
    spacing = float(os.environ.get("NVALCHEMI_WIRED_SPACING", 4.4))
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
    g = torch.Generator().manual_seed(seed)
    positions = positions + 0.05 * torch.randn(positions.shape, dtype=dtype, generator=g)
    positions = positions % box
    atomic_numbers = torch.tensor([6, 1, 1, 1, 1] * (n // 5), dtype=torch.long)
    cell = torch.eye(3, dtype=dtype) * box
    pbc = torch.ones(3, dtype=torch.bool)
    return positions, atomic_numbers, cell, pbc


def _make_data(atomic_numbers, positions, cell, pbc, device, dtype):
    n = positions.shape[0]
    return AtomicData(
        atomic_numbers=atomic_numbers.to(device),
        positions=positions.to(device=device, dtype=dtype).clone(),
        cell=cell.to(device=device, dtype=dtype).unsqueeze(0),
        pbc=pbc.to(device).unsqueeze(0),
        forces=torch.zeros(n, 3, device=device, dtype=dtype),
        energy=torch.zeros(1, 1, device=device, dtype=dtype),
    )


def _build_pipeline(aim_cut_holder: list[float], device, dtype):
    """Fresh AIMNet2(charges) -> PME(charges) wired use_autograd pipeline."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from nvalchemi.models.aimnet2 import AIMNet2Wrapper

    from nvalchemi.models.pipeline import PipelineGroup, PipelineModelWrapper
    from nvalchemi.models.pme import PMEModelWrapper

    aim = AIMNet2Wrapper.from_checkpoint("aimnet2", device=device)
    aim.eval()
    aim.model_config.active_outputs = {"energy", "forces", "charges"}
    aim_cut_holder[0] = float(aim._cutoff)
    pme = PMEModelWrapper(cutoff=_PME_CUT)  # hybrid_forces=True default
    return PipelineModelWrapper(
        groups=[PipelineGroup(steps=[aim, pme], use_autograd=True)]
    )


def _wired_pipeline_worker(rank: int, world_size: int) -> None:
    from torch.distributed import DeviceMesh

    from nvalchemi.distributed.distributed_pipeline import DistributedPipelineModel
    from nvalchemi.distributed.partitioner import SpatialPartitioner
    from nvalchemi.distributed.sharded_batch import ShardedBatch
    from nvalchemi.neighbors import compute_neighbors

    dtype = torch.float32
    device = torch.device(f"cuda:{rank}")
    positions, an, cell, pbc = _methane_packing(dtype=dtype)
    n_global = positions.shape[0]
    aim_cut = [0.0]

    # --- Single-GPU reference on rank 0: the full wired pipeline forward ---
    e_ref = torch.zeros(1, dtype=dtype, device=device)
    f_ref = torch.zeros(n_global, 3, dtype=dtype, device=device)
    if rank == 0:
        ref_pipe = _build_pipeline(aim_cut, device, dtype)
        batch = Batch.from_data_list([_make_data(an, positions, cell, pbc, device, dtype)])
        compute_neighbors(batch, config=ref_pipe.model_config.neighbor_config)
        out = ref_pipe(batch)
        e_ref.copy_(out["energy"].sum().detach().view(1))
        f_ref.copy_(out["forces"].detach())
        del ref_pipe
    dist.broadcast(e_ref, src=0)
    dist.broadcast(f_ref, src=0)

    # --- Distributed composite over ONE shared partition (built at max cutoff) ---
    pipeline = _build_pipeline(aim_cut, device, dtype)
    max_cut = max(aim_cut[0], _PME_CUT)
    mesh = DeviceMesh("cuda", list(range(world_size)), mesh_dim_names=("domain",))
    base_config = DomainConfig(cutoff=max_cut, skin=_SKIN, mesh=mesh)
    full = (
        Batch.from_data_list([_make_data(an, positions, cell, pbc, device, dtype)])
        if rank == 0
        else None
    )
    sharded = ShardedBatch.from_batch(batch=full, mesh=mesh, config=base_config, src=0)

    with DistributedPipelineModel(pipeline, base_config) as dpm:
        composite = dpm(sharded)
    e_local = composite["energy"].sum().detach()
    f_owned = composite["forces"].detach()

    partitioner = SpatialPartitioner(
        config=base_config,
        cell_matrix=cell.to(device=device, dtype=dtype).unsqueeze(0),
        pbc=pbc.to(device).unsqueeze(0),
    )
    local_mask = (
        partitioner.assign_atoms_to_ranks(positions.to(device=device, dtype=dtype)) == rank
    )
    f_ref_owned = f_ref[local_mask]

    de = (e_local - e_ref).abs().item()
    df = (f_owned - f_ref_owned).abs().max().item()
    print(
        f"[pipe-wired rank {rank}] ΔE={de:.3e} |ΔF|max={df:.3e} "
        f"n_owned={f_owned.shape[0]} aim_cut={aim_cut[0]:.2f} "
        f"dist_e={e_local.item():+.4f} ref_e={e_ref.item():+.4f}",
        flush=True,
    )
    # Energy total is ~-7e4 eV for the methane supercell; float32 carries only
    # ~1 ulp (~8e-3 eV) at that magnitude, so compare on a relative scale (a real
    # composition error shifts the total far more than float32 rounding). The DD
    # energy is fp64 (the per-system reductions accumulate in fp64 for
    # order-independence); cast to the reference dtype before comparing.
    torch.testing.assert_close(
        e_local.view(1).to(e_ref.dtype), e_ref, rtol=1e-5, atol=0.1,
        msg=f"rank {rank}: wired composite energy mismatch ΔE={de:.3e}",
    )
    torch.testing.assert_close(
        f_owned, f_ref_owned, rtol=1e-2, atol=2e-3,
        msg=f"rank {rank}: wired composite forces mismatch |ΔF|max={df:.3e}",
    )


@_skip
def test_distributed_pipeline_aimnet2_pme_2ranks():
    """``DistributedPipelineModel(AIMNet2 charges -> PME)`` == single-GPU pipeline."""
    pytest.importorskip("nvalchemiops", reason="nvalchemiops not installed")
    pytest.importorskip("aimnet", reason="aimnet not installed")
    mp.spawn(
        _worker,
        args=(WORLD_SIZE, "29587", _wired_pipeline_worker),
        nprocs=WORLD_SIZE,
    )
