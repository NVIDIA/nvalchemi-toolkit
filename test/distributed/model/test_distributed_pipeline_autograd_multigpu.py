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

"""Multi-GPU regression: ``DistributedPipelineModel`` (C2, shared-autograd).

Gates distributed composition of a pipeline that mixes a **shared-autograd
group** (MACE, ``use_autograd=True`` — forces are ``-dE/dr``) with a
**direct-force group** (DFT-D3) — the motivating MACE + DFT-D3 potential.
The 2-rank composite forward must reproduce, on each rank's owned atoms, the
**single-GPU pipeline** result (energy + per-atom forces).

This validates the C2 claim (``proposal-distributed-pipeline.md`` §C2 /
decision 4 & 6): because the sub-models share one owned partition and there is
no cross-model coupling (wiring is C3), the group force ``-d(ΣE_m)/dr``
decomposes into ``Σ_m (-dE_m/dr)`` — so running each sub-model through its own
validated single-model DD autograd forward (forces enabled) and summing the
owned-aligned results is exactly correct. A perfect lattice has ~zero net force
(a vacuous check), so the geometry is **rattled**.

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
_A1, _A2, _S8 = 0.4289, 4.4407, 0.7875
# Realistic DFT-D3 dispersion cutoff (~12 Å) — far larger than MACE's ~6 Å, so
# the composite genuinely exercises per-model right-sized halos (MACE rebuilds a
# small ghost layer, DFTD3 a large one) over the one shared owned partition.
_DFTD3_CUT = 12.0
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


def _build_lattice(dtype: torch.dtype = torch.float64, seed: int = 0):
    # Non-degenerate: DFTD3's deep CN halo (cutoff 12 Å + skin 4 Å -> ghost 16 Å)
    # means the remote band (box/2 - 2*ghost) must exceed the lattice spacing, i.e.
    # box > 4*ghost + 2*spacing ~ 74 Å (not just 64). 88 Å / 16 = 5.5 Å spacing
    # leaves a ~12 Å (2-plane) remote band; 4096 atoms.
    n_side = int(os.environ.get("NVALCHEMI_PIPE_N_SIDE", 16))
    box = float(os.environ.get("NVALCHEMI_PIPE_BOX", 88.0))
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
    return positions, atomic_numbers, masses, cell, pbc


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
    """Construct a fresh MACE(use_autograd) + DFTD3(direct) pipeline.

    Records MACE's cutoff into ``mace_cut_holder[0]`` for the caller's max-cutoff
    partition. Each construction loads identical (deterministic) MACE weights and
    parameter-free DFTD3, so a separate reference and distributed instance match.
    """
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


def _autograd_pipeline_worker(rank: int, world_size: int) -> None:
    from torch.distributed import DeviceMesh

    from nvalchemi.distributed.distributed_pipeline import DistributedPipelineModel
    from nvalchemi.distributed.partitioner import SpatialPartitioner
    from nvalchemi.distributed.sharded_batch import ShardedBatch
    from nvalchemi.neighbors import compute_neighbors

    dtype = torch.float64
    device = torch.device(f"cuda:{rank}")
    positions, an, masses, cell, pbc = _build_lattice(dtype=dtype)
    n_global = positions.shape[0]

    mace_cut = [0.0]

    # --- Single-GPU reference on rank 0: the full pipeline forward ---
    e_ref = torch.zeros(1, dtype=dtype, device=device)
    f_ref = torch.zeros(n_global, 3, dtype=dtype, device=device)
    if rank == 0:
        ref_pipe = _build_pipeline(mace_cut, device, dtype)
        batch = Batch.from_data_list(
            [_make_data(an, positions, masses, cell, pbc, device, dtype)]
        )
        compute_neighbors(batch, config=ref_pipe.model_config.neighbor_config)
        out = ref_pipe(batch)
        e_ref.copy_(out["energy"].sum().detach().view(1))
        f_ref.copy_(out["forces"].detach())
        del ref_pipe
    dist.broadcast(e_ref, src=0)
    dist.broadcast(f_ref, src=0)

    # --- Distributed composite over ONE shared partition (built at max cutoff) ---
    pipeline = _build_pipeline(mace_cut, device, dtype)
    max_cut = max(mace_cut[0], _DFTD3_CUT)
    mesh = DeviceMesh("cuda", list(range(world_size)), mesh_dim_names=("domain",))
    base_config = DomainConfig(
        cutoff=max_cut, skin=_SKIN, mesh=mesh, require_nondegenerate=True
    )
    full = (
        Batch.from_data_list(
            [_make_data(an, positions, masses, cell, pbc, device, dtype)]
        )
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
        partitioner.assign_atoms_to_ranks(positions.to(device=device, dtype=dtype))
        == rank
    )
    f_ref_owned = f_ref[local_mask]

    de = (e_local - e_ref).abs().item()
    df = (f_owned - f_ref_owned).abs().max().item()
    print(
        f"[pipe-autograd rank {rank}] ΔE={de:.3e} |ΔF|max={df:.3e} "
        f"n_owned={f_owned.shape[0]} mace_cut={mace_cut[0]:.2f} "
        f"dist_e={e_local.item():+.4f} ref_e={e_ref.item():+.4f}",
        flush=True,
    )
    torch.testing.assert_close(
        e_local.view(1),
        e_ref,
        rtol=1e-5,
        atol=1e-4,
        msg=f"rank {rank}: composite energy mismatch ΔE={de:.3e}",
    )
    torch.testing.assert_close(
        f_owned,
        f_ref_owned,
        rtol=1e-4,
        atol=1e-4,
        msg=f"rank {rank}: composite forces mismatch |ΔF|max={df:.3e}",
    )


@_skip
def test_distributed_pipeline_mace_dftd3_2ranks():
    """``DistributedPipelineModel(MACE[use_autograd] + DFTD3)`` == single-GPU pipeline."""
    pytest.importorskip("nvalchemiops", reason="nvalchemiops not installed")
    pytest.importorskip("mace", reason="mace-torch not installed")
    mp.spawn(
        _worker,
        args=(WORLD_SIZE, "29586", _autograd_pipeline_worker),
        nprocs=WORLD_SIZE,
    )
