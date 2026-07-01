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

"""Multi-GPU regression: ``DistributedPipelineModel`` (C1, direct-force).

Gates distributed composition of direct-force models — here DFT-D3 + Ewald
(the direct-force analog of the MACE+DFTD3 / AIMNet2+PME use cases) — over
**one shared owned partition with per-model halos**. The 2-rank composite
forward must match, on each rank's owned atoms, the **sum of the two
single-GPU single-model results** (energy + per-atom forces).

This validates that each sub-model runs on its own right-sized ghost width
over the same owned set and the owned-aligned outputs add correctly — the C1
tier of ``proposal-distributed-pipeline.md``.

Requires 2+ CUDA GPUs and ``nvalchemiops`` with the DFTD3 + Ewald kernels.
"""

from __future__ import annotations

import os
from typing import Any

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from nvalchemi.data import AtomicData, Batch
from nvalchemi.distributed.config import DomainConfig

WORLD_SIZE = 2
_A1, _A2, _S8 = 0.4289, 4.4407, 0.7875
_DFTD3_CUT = 5.0
_EWALD_CUT = 6.0
_SKIN = 4.0  # CN-depth halo margin for DFTD3 ghost coordination numbers

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


def _build_lattice(dtype: torch.dtype = torch.float32, seed: int = 0):
    n_side = int(os.environ.get("NVALCHEMI_PIPE_N_SIDE", 4))
    box = float(os.environ.get("NVALCHEMI_PIPE_BOX", 12.0))
    coords = torch.arange(n_side, dtype=dtype) * (box / n_side)
    gx, gy, gz = torch.meshgrid(coords, coords, coords, indexing="ij")
    positions = torch.stack([gx.flatten(), gy.flatten(), gz.flatten()], dim=-1)
    n = positions.shape[0]
    g = torch.Generator().manual_seed(seed)
    positions = positions + 0.1 * torch.randn(positions.shape, dtype=dtype, generator=g)
    positions = positions % box
    sign = torch.ones(n, dtype=dtype)
    sign[1::2] = -1.0
    charges = sign  # globally neutral for even n
    atomic_numbers = torch.where(
        sign > 0, torch.full((n,), 11, dtype=torch.long), torch.full((n,), 17, dtype=torch.long)
    )
    masses = torch.where(
        sign > 0, torch.full((n,), 22.99, dtype=dtype), torch.full((n,), 35.45, dtype=dtype)
    )
    cell = torch.eye(3, dtype=dtype) * box
    pbc = torch.ones(3, dtype=torch.bool)
    return positions, atomic_numbers, masses, charges, cell, pbc


def _make_data(atomic_numbers, positions, masses, charges, cell, pbc, device, dtype):
    n = positions.shape[0]
    return AtomicData(
        atomic_numbers=atomic_numbers.to(device),
        positions=positions.to(device=device, dtype=dtype).clone(),
        atomic_masses=masses.to(device=device, dtype=dtype),
        charges=charges.to(device=device, dtype=dtype),
        cell=cell.to(device=device, dtype=dtype).unsqueeze(0),
        pbc=pbc.to(device).unsqueeze(0),
        forces=torch.zeros(n, 3, device=device, dtype=dtype),
        energy=torch.zeros(1, 1, device=device, dtype=dtype),
    )


def _single_ref(wrapper, an, pos, m, q, cell, pbc, device, dtype):
    from nvalchemi.neighbors import compute_neighbors

    batch = Batch.from_data_list([_make_data(an, pos, m, q, cell, pbc, device, dtype)])
    compute_neighbors(batch, config=wrapper.model_config.neighbor_config)
    out = wrapper(batch)
    return out["energy"].sum().detach(), out["forces"].detach()


def _pipeline_worker(rank: int, world_size: int) -> None:
    from torch.distributed import DeviceMesh

    from nvalchemi.distributed.distributed_pipeline import DistributedPipelineModel
    from nvalchemi.distributed.partitioner import SpatialPartitioner
    from nvalchemi.distributed.sharded_batch import ShardedBatch
    from nvalchemi.models.dftd3 import DFTD3ModelWrapper
    from nvalchemi.models.ewald import EwaldModelWrapper
    from nvalchemi.models.pipeline import PipelineGroup, PipelineModelWrapper

    dtype = torch.float32
    device = torch.device(f"cuda:{rank}")
    positions, an, masses, charges, cell, pbc = _build_lattice(dtype=dtype)
    n_global = positions.shape[0]

    def _mk_dftd3():
        return DFTD3ModelWrapper(a1=_A1, a2=_A2, s8=_S8, cutoff=_DFTD3_CUT)

    def _mk_ewald():
        return EwaldModelWrapper(cutoff=_EWALD_CUT, hybrid_forces=False)

    # --- Single-GPU reference on rank 0: sum of the two single models ---
    e_ref = torch.zeros(1, dtype=dtype, device=device)
    f_ref = torch.zeros(n_global, 3, dtype=dtype, device=device)
    if rank == 0:
        e_d, f_d = _single_ref(_mk_dftd3(), an, positions, masses, charges, cell, pbc, device, dtype)
        e_e, f_e = _single_ref(_mk_ewald(), an, positions, masses, charges, cell, pbc, device, dtype)
        e_ref.copy_((e_d + e_e).view(1))
        f_ref.copy_(f_d + f_e)
    dist.broadcast(e_ref, src=0)
    dist.broadcast(f_ref, src=0)

    # --- Distributed composite over ONE shared partition (built at max cutoff) ---
    mesh = DeviceMesh("cuda", list(range(world_size)), mesh_dim_names=("domain",))
    base_config = DomainConfig(cutoff=max(_DFTD3_CUT, _EWALD_CUT), skin=_SKIN, mesh=mesh)
    full = (
        Batch.from_data_list([_make_data(an, positions, masses, charges, cell, pbc, device, dtype)])
        if rank == 0
        else None
    )
    sharded = ShardedBatch.from_batch(batch=full, mesh=mesh, config=base_config, src=0)

    pipeline = PipelineModelWrapper(
        groups=[PipelineGroup(steps=[_mk_dftd3(), _mk_ewald()], use_autograd=False)]
    )
    with DistributedPipelineModel(pipeline, base_config) as dpm:
        out = dpm(sharded)
    e_local = out["energy"].sum().detach()
    f_owned = out["forces"].detach()

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
        f"[pipe-dd rank {rank}] ΔE={de:.3e} |ΔF|max={df:.3e} "
        f"n_owned={f_owned.shape[0]} dist_e={e_local.item():+.4f} ref_e={e_ref.item():+.4f}",
        flush=True,
    )
    torch.testing.assert_close(
        e_local.view(1), e_ref, rtol=1e-4, atol=1e-3,
        msg=f"rank {rank}: composite energy mismatch ΔE={de:.3e}",
    )
    torch.testing.assert_close(
        f_owned, f_ref_owned, rtol=1e-3, atol=1e-4,
        msg=f"rank {rank}: composite forces mismatch |ΔF|max={df:.3e}",
    )


@_skip
def test_distributed_pipeline_dftd3_ewald_2ranks():
    """``DistributedPipelineModel(DFTD3 + Ewald)`` == summed single-models."""
    pytest.importorskip("nvalchemiops", reason="nvalchemiops not installed")
    mp.spawn(
        _worker,
        args=(WORLD_SIZE, "29585", _pipeline_worker),
        nprocs=WORLD_SIZE,
    )
