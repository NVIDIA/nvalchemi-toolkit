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

"""Multi-GPU regression: PME under ``torch.compile`` + domain decomposition.

PME's reciprocal energy is differentiable (staged structure factors +
``_InjectChargeGrad``), so with ``hybrid_forces=False`` it rides the framework's
compiled energy-autograd DD path. This gate checks the compiled 2-rank forward
reproduces single-GPU PME (rattled NaCl) and shows no steady-state recompiles.

Requires 2+ CUDA GPUs and ``nvalchemiops`` (PME / B-spline kernels).
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
WARMUP_STEPS = 4
STEADY_STEPS = 4
JITTER = 0.05
_PME_CUT = 6.0

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


def _build_nacl(dtype: torch.dtype = torch.float32, seed: int = 0):
    n_side = int(os.environ.get("NVALCHEMI_PME_N_SIDE", 10))
    box = float(os.environ.get("NVALCHEMI_PME_BOX", 32.0))
    coords = torch.arange(n_side, dtype=dtype) * (box / n_side)
    gx, gy, gz = torch.meshgrid(coords, coords, coords, indexing="ij")
    positions = torch.stack([gx.flatten(), gy.flatten(), gz.flatten()], dim=-1)
    n = positions.shape[0]
    g = torch.Generator().manual_seed(seed)
    positions = positions + 0.15 * torch.randn(positions.shape, dtype=dtype, generator=g)
    positions = positions % box
    sign = torch.ones(n, dtype=dtype)
    sign[1::2] = -1.0
    atomic_numbers = torch.where(
        sign > 0, torch.full((n,), 11, dtype=torch.long), torch.full((n,), 17, dtype=torch.long)
    )
    masses = torch.where(
        sign > 0, torch.full((n,), 22.99, dtype=dtype), torch.full((n,), 35.45, dtype=dtype)
    )
    cell = torch.eye(3, dtype=dtype) * box
    pbc = torch.ones(3, dtype=torch.bool)
    return positions, atomic_numbers, masses, sign, cell, pbc, box


def _make_data(an, positions, masses, charges, cell, pbc, device, dtype):
    n = positions.shape[0]
    return AtomicData(
        atomic_numbers=an.to(device),
        positions=positions.to(device=device, dtype=dtype).clone(),
        atomic_masses=masses.to(device=device, dtype=dtype),
        charges=charges.to(device=device, dtype=dtype),
        cell=cell.to(device=device, dtype=dtype).unsqueeze(0),
        pbc=pbc.to(device).unsqueeze(0),
        forces=torch.zeros(n, 3, device=device, dtype=dtype),
        energy=torch.zeros(1, 1, device=device, dtype=dtype),
    )


def _compile_worker(rank: int, world_size: int) -> None:
    from torch._dynamo.utils import counters
    from torch.distributed import DeviceMesh

    from nvalchemi.distributed.distributed_model import DistributedModel
    from nvalchemi.distributed.partitioner import SpatialPartitioner
    from nvalchemi.distributed.sharded_batch import ShardedBatch
    from nvalchemi.models.pme import PMEModelWrapper
    from nvalchemi.neighbors import compute_neighbors

    dtype = torch.float32
    device = torch.device(f"cuda:{rank}")
    positions0, an, masses, charges, cell, pbc, box = _build_nacl(dtype=dtype)
    n_global = positions0.shape[0]

    # Single-GPU reference, COMPILED — not eager. The gate isolates *DD*
    # correctness, so the reference must share the production compiled path's
    # numerics. PME's fp32 reciprocal space (rfft + the fused complex convolve)
    # drifts ~15 eV / ~2e-3 forces between eager and torch.compile on this
    # system regardless of DD — inductor warns it "does not support code
    # generation for complex operators" and falls back, so the compiled
    # reduction order differs. Measured single-GPU (no DD): eager E=-2879.63 vs
    # compiled E=-2864.47 (Δ=15.16), |ΔF|max=2.43e-3 — identical to what a 2-rank
    # run shows. Comparing compiled-DD against an *eager* reference would charge
    # that orthogonal compile-vs-eager drift to the DD path. So the reference is
    # the compiled energy-only forward + autograd forces, exactly mirroring
    # DistributedModel._compiled_energy_autograd_forward.
    e_ref = torch.zeros(1, dtype=dtype, device=device)
    f_ref = torch.zeros(n_global, 3, dtype=dtype, device=device)
    if rank == 0:
        ref = PMEModelWrapper(cutoff=_PME_CUT, hybrid_forces=False)
        batch = Batch.from_data_list(
            [_make_data(an, positions0, masses, charges, cell, pbc, device, dtype)]
        )
        compute_neighbors(batch, config=ref.model_config.neighbor_config)

        def _ref_energy(b):
            return ref(b)["energy"]

        compiled_ref = torch.compile(_ref_energy, dynamic=False)
        ref.model_config.active_outputs = {"energy"}
        pos_leaf = batch.positions.detach().requires_grad_(True)
        batch._atoms_group["positions"] = pos_leaf
        e = compiled_ref(batch)
        (grad,) = torch.autograd.grad([e.sum()], [pos_leaf])
        e_ref.copy_(e.sum().detach().view(1))
        f_ref.copy_((-grad).detach())
        del ref
    dist.broadcast(e_ref, src=0)
    dist.broadcast(f_ref, src=0)

    wrapper = PMEModelWrapper(cutoff=_PME_CUT, hybrid_forces=False)
    cp = wrapper.distribution_spec().compile
    assert cp is not None and cp.forces_via_autograd, (
        "PME(hybrid_forces=False) must declare a forces_via_autograd CompilePolicy"
    )
    mesh = DeviceMesh("cuda", list(range(world_size)), mesh_dim_names=("domain",))
    cfg = DomainConfig(cutoff=_PME_CUT, skin=2.0, mesh=mesh)
    partitioner = SpatialPartitioner(
        config=cfg,
        cell_matrix=cell.to(device=device, dtype=dtype).unsqueeze(0),
        pbc=pbc.to(device).unsqueeze(0),
    )

    def _sharded(pos):
        full = (
            Batch.from_data_list([_make_data(an, pos, masses, charges, cell, pbc, device, dtype)])
            if rank == 0
            else None
        )
        return ShardedBatch.from_batch(batch=full, mesh=mesh, config=cfg, src=0)

    gen = torch.Generator(device="cpu").manual_seed(11)
    graphs_after_warmup = [0]

    with DistributedModel(wrapper, cfg, compile=True) as dm:
        for step in range(WARMUP_STEPS + STEADY_STEPS):
            if step == 0:
                pos = positions0
            else:
                disp = JITTER * torch.randn(positions0.shape, dtype=dtype, generator=gen).to(device)
                pos = (positions0.to(device) + disp) % box
            out = dm(_sharded(pos))
            if step == 0:
                f_owned = out["forces"].detach()
                e_local = out["energy"].sum().detach()
                local_mask = (
                    partitioner.assign_atoms_to_ranks(positions0.to(device=device, dtype=dtype))
                    == rank
                )
                f_ref_owned = f_ref[local_mask]
                de = (e_local - e_ref).abs().item()
                df = (f_owned - f_ref_owned).abs().max().item()
                print(
                    f"[pme-compile rank {rank}] step0 ΔE={de:.3e} |ΔF|max={df:.3e} "
                    f"n_owned={f_owned.shape[0]}",
                    flush=True,
                )
                torch.testing.assert_close(
                    e_local.view(1).to(e_ref.dtype), e_ref, rtol=1e-4, atol=1e-2,
                    msg=f"rank {rank}: compiled PME energy mismatch ΔE={de:.3e}",
                )
                torch.testing.assert_close(
                    f_owned, f_ref_owned, rtol=1e-2, atol=2e-3,
                    msg=f"rank {rank}: compiled PME forces mismatch |ΔF|max={df:.3e}",
                )
            if step == WARMUP_STEPS - 1:
                graphs_after_warmup[0] = counters["stats"]["unique_graphs"]

    final_graphs = counters["stats"]["unique_graphs"]
    print(
        f"[pme-compile rank {rank}] unique_graphs warmup={graphs_after_warmup[0]} final={final_graphs}",
        flush=True,
    )
    assert final_graphs == graphs_after_warmup[0], (
        f"rank {rank}: compiled PME recompiled in steady state "
        f"({graphs_after_warmup[0]} -> {final_graphs})"
    )


@_skip
def test_pme_compile_dd_2ranks():
    """Compiled ``DistributedModel(PME, hybrid_forces=False)`` == single-GPU; no steady recompiles."""
    pytest.importorskip("nvalchemiops", reason="nvalchemiops not installed")
    mp.spawn(_worker, args=(WORLD_SIZE, "29589", _compile_worker), nprocs=WORLD_SIZE)
