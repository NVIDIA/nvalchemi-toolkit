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

"""Multi-GPU regressions for ``DistributedPipelineModel`` (composed-model DD).

Four distinct composition scenarios, each gating a 2-rank composite forward over
ONE shared owned partition (per-model halos) against a single-GPU reference:

* ``test_distributed_pipeline_dftd3_ewald_2ranks`` — C1 direct-force (DFT-D3 +
  Ewald); composite == sum of the two single-model results.
* ``test_distributed_pipeline_mace_dftd3_2ranks`` — C2 shared-autograd (MACE
  ``-dE/dr``) + direct-force (DFT-D3); composite == single-GPU pipeline.
* ``test_distributed_pipeline_aimnet2_pme_2ranks`` — C3 wired field (AIMNet2
  charges consumed by PME across the halo); composite == single-GPU pipeline.
* ``test_distributed_pipeline_compile_mace_dftd3_2ranks`` — compiled composition
  (MACE compiled, DFT-D3 eager); equivalence + no steady-state recompiles.

Requires 2+ CUDA GPUs and ``nvalchemiops``; the MACE / AIMNet2 scenarios also
need ``mace-torch`` / the ``aimnet2`` checkpoint. Geometries are rattled so net
forces are non-trivial."""

from __future__ import annotations

import os
import warnings

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from _dd_harness import nccl_worker as _worker

from nvalchemi.data import AtomicData, Batch
from nvalchemi.distributed.config import DomainConfig

WORLD_SIZE = 2
_A1, _A2, _S8 = 0.4289, 4.4407, 0.7875

_skip = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < WORLD_SIZE,
    reason=f"Need {WORLD_SIZE}+ CUDA GPUs",
)


# ====================================================================
# C1 — DFT-D3 + Ewald (direct-force composition)
# ====================================================================

_DE_DFTD3_CUT = 5.0
_DE_EWALD_CUT = 6.0
_DE_SKIN = 4.0  # CN-depth halo margin for DFTD3 ghost coordination numbers


def _de_build_lattice(dtype: torch.dtype = torch.float32, seed: int = 0):
    # Non-degenerate: max cutoff 6 Å + skin 4 Å -> ghost 10 Å, so a 2-rank split
    # needs box > 40 Å. 48 Å / 12 = 4 Å spacing.
    n_side = int(os.environ.get("NVALCHEMI_PIPE_N_SIDE", 12))
    box = float(os.environ.get("NVALCHEMI_PIPE_BOX", 48.0))
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
    return positions, atomic_numbers, masses, charges, cell, pbc


def _de_make_data(atomic_numbers, positions, masses, charges, cell, pbc, device, dtype):
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


def _de_single_ref(wrapper, an, pos, m, q, cell, pbc, device, dtype):
    from nvalchemi.neighbors import compute_neighbors

    batch = Batch.from_data_list(
        [_de_make_data(an, pos, m, q, cell, pbc, device, dtype)]
    )
    compute_neighbors(batch, config=wrapper.model_config.neighbor_config)
    out = wrapper(batch)
    return out["energy"].sum().detach(), out["forces"].detach()


def _de_pipeline_worker(rank: int, world_size: int) -> None:
    from torch.distributed import DeviceMesh

    from nvalchemi.distributed.distributed_pipeline import DistributedPipelineModel
    from nvalchemi.distributed.partitioner import SpatialPartitioner
    from nvalchemi.distributed.sharded_batch import ShardedBatch
    from nvalchemi.models.dftd3 import DFTD3ModelWrapper
    from nvalchemi.models.ewald import EwaldModelWrapper
    from nvalchemi.models.pipeline import PipelineGroup, PipelineModelWrapper

    dtype = torch.float32
    device = torch.device(f"cuda:{rank}")
    positions, an, masses, charges, cell, pbc = _de_build_lattice(dtype=dtype)
    n_global = positions.shape[0]

    def _mk_dftd3():
        return DFTD3ModelWrapper(a1=_A1, a2=_A2, s8=_S8, cutoff=_DE_DFTD3_CUT)

    def _mk_ewald():
        return EwaldModelWrapper(cutoff=_DE_EWALD_CUT, hybrid_forces=False)

    # --- Single-GPU reference on rank 0: sum of the two single models ---
    e_ref = torch.zeros(1, dtype=dtype, device=device)
    f_ref = torch.zeros(n_global, 3, dtype=dtype, device=device)
    if rank == 0:
        e_d, f_d = _de_single_ref(
            _mk_dftd3(), an, positions, masses, charges, cell, pbc, device, dtype
        )
        e_e, f_e = _de_single_ref(
            _mk_ewald(), an, positions, masses, charges, cell, pbc, device, dtype
        )
        e_ref.copy_((e_d + e_e).view(1))
        f_ref.copy_(f_d + f_e)
    dist.broadcast(e_ref, src=0)
    dist.broadcast(f_ref, src=0)

    # --- Distributed composite over ONE shared partition (built at max cutoff) ---
    mesh = DeviceMesh("cuda", list(range(world_size)), mesh_dim_names=("domain",))
    base_config = DomainConfig(
        cutoff=max(_DE_DFTD3_CUT, _DE_EWALD_CUT),
        skin=_DE_SKIN,
        mesh=mesh,
        require_nondegenerate=True,
    )
    full = (
        Batch.from_data_list(
            [_de_make_data(an, positions, masses, charges, cell, pbc, device, dtype)]
        )
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
        partitioner.assign_atoms_to_ranks(positions.to(device=device, dtype=dtype))
        == rank
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
        e_local.view(1),
        e_ref,
        rtol=1e-4,
        atol=1e-3,
        msg=f"rank {rank}: composite energy mismatch ΔE={de:.3e}",
    )
    torch.testing.assert_close(
        f_owned,
        f_ref_owned,
        rtol=1e-3,
        atol=1e-4,
        msg=f"rank {rank}: composite forces mismatch |ΔF|max={df:.3e}",
    )


@_skip
def test_distributed_pipeline_dftd3_ewald_2ranks():
    """``DistributedPipelineModel(DFTD3 + Ewald)`` == summed single-models."""
    pytest.importorskip("nvalchemiops", reason="nvalchemiops not installed")
    mp.spawn(
        _worker,
        args=(WORLD_SIZE, "29585", _de_pipeline_worker),
        nprocs=WORLD_SIZE,
    )


# ====================================================================
# C2 — MACE (shared-autograd) + DFT-D3 (direct-force)
# ====================================================================

# Realistic DFT-D3 dispersion cutoff (~12 Å) — far larger than MACE's ~6 Å,
# so the composite genuinely exercises per-model right-sized halos (MACE
# rebuilds a small ghost layer, DFTD3 a large one) over one shared partition.
_MD_DFTD3_CUT = 12.0
_MD_SKIN = 4.0  # CN-depth halo margin for DFTD3 ghost coordination numbers


def _md_build_lattice(dtype: torch.dtype = torch.float64, seed: int = 0):
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


def _md_make_data(atomic_numbers, positions, masses, cell, pbc, device, dtype):
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


def _md_build_pipeline(mace_cut_holder: list[float], device, dtype):
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
    dftd3 = DFTD3ModelWrapper(a1=_A1, a2=_A2, s8=_S8, cutoff=_MD_DFTD3_CUT)
    return PipelineModelWrapper(
        groups=[
            PipelineGroup(steps=[mace], use_autograd=True),
            PipelineGroup(steps=[dftd3], use_autograd=False),
        ]
    )


def _md_autograd_pipeline_worker(rank: int, world_size: int) -> None:
    from torch.distributed import DeviceMesh

    from nvalchemi.distributed.distributed_pipeline import DistributedPipelineModel
    from nvalchemi.distributed.partitioner import SpatialPartitioner
    from nvalchemi.distributed.sharded_batch import ShardedBatch
    from nvalchemi.neighbors import compute_neighbors

    dtype = torch.float64
    device = torch.device(f"cuda:{rank}")
    positions, an, masses, cell, pbc = _md_build_lattice(dtype=dtype)
    n_global = positions.shape[0]

    mace_cut = [0.0]

    # --- Single-GPU reference on rank 0: the full pipeline forward ---
    e_ref = torch.zeros(1, dtype=dtype, device=device)
    f_ref = torch.zeros(n_global, 3, dtype=dtype, device=device)
    if rank == 0:
        ref_pipe = _md_build_pipeline(mace_cut, device, dtype)
        batch = Batch.from_data_list(
            [_md_make_data(an, positions, masses, cell, pbc, device, dtype)]
        )
        compute_neighbors(batch, config=ref_pipe.model_config.neighbor_config)
        out = ref_pipe(batch)
        e_ref.copy_(out["energy"].sum().detach().view(1))
        f_ref.copy_(out["forces"].detach())
        del ref_pipe
    dist.broadcast(e_ref, src=0)
    dist.broadcast(f_ref, src=0)

    # --- Distributed composite over ONE shared partition (built at max cutoff) ---
    pipeline = _md_build_pipeline(mace_cut, device, dtype)
    max_cut = max(mace_cut[0], _MD_DFTD3_CUT)
    mesh = DeviceMesh("cuda", list(range(world_size)), mesh_dim_names=("domain",))
    base_config = DomainConfig(
        cutoff=max_cut, skin=_MD_SKIN, mesh=mesh, require_nondegenerate=True
    )
    full = (
        Batch.from_data_list(
            [_md_make_data(an, positions, masses, cell, pbc, device, dtype)]
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
        args=(WORLD_SIZE, "29586", _md_autograd_pipeline_worker),
        nprocs=WORLD_SIZE,
    )


# ====================================================================
# C3 — AIMNet2 charges -> PME (wired shared-autograd)
# ====================================================================

_AP_PME_CUT = 6.0
_AP_SKIN = 0.5


def _ap_methane_packing(dtype: torch.dtype = torch.float32, seed: int = 0):
    """``n_per_side**3`` methane molecules (5 atoms each) on a cubic PBC lattice,
    rattled so net forces are non-trivial."""
    # Non-degenerate: PME cutoff 6 Å + skin 0.5 Å -> ghost 6.5 Å, so a 2-rank
    # split needs box > 26 Å. 7 * 4.4 = 30.8 Å (7**3 * 5 = 1715 atoms).
    n_per_side = int(os.environ.get("NVALCHEMI_WIRED_N_SIDE", 7))
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
    positions = positions + 0.05 * torch.randn(
        positions.shape, dtype=dtype, generator=g
    )
    positions = positions % box
    atomic_numbers = torch.tensor([6, 1, 1, 1, 1] * (n // 5), dtype=torch.long)
    cell = torch.eye(3, dtype=dtype) * box
    pbc = torch.ones(3, dtype=torch.bool)
    return positions, atomic_numbers, cell, pbc


def _ap_make_data(atomic_numbers, positions, cell, pbc, device, dtype):
    n = positions.shape[0]
    return AtomicData(
        atomic_numbers=atomic_numbers.to(device),
        positions=positions.to(device=device, dtype=dtype).clone(),
        cell=cell.to(device=device, dtype=dtype).unsqueeze(0),
        pbc=pbc.to(device).unsqueeze(0),
        forces=torch.zeros(n, 3, device=device, dtype=dtype),
        energy=torch.zeros(1, 1, device=device, dtype=dtype),
    )


def _ap_build_pipeline(aim_cut_holder: list[float], device, dtype):
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
    pme = PMEModelWrapper(cutoff=_AP_PME_CUT)  # hybrid_forces=True default
    return PipelineModelWrapper(
        groups=[PipelineGroup(steps=[aim, pme], use_autograd=True)]
    )


def _ap_wired_pipeline_worker(rank: int, world_size: int) -> None:
    from torch.distributed import DeviceMesh

    from nvalchemi.distributed.distributed_pipeline import DistributedPipelineModel
    from nvalchemi.distributed.partitioner import SpatialPartitioner
    from nvalchemi.distributed.sharded_batch import ShardedBatch
    from nvalchemi.neighbors import compute_neighbors

    dtype = torch.float32
    device = torch.device(f"cuda:{rank}")
    positions, an, cell, pbc = _ap_methane_packing(dtype=dtype)
    n_global = positions.shape[0]
    aim_cut = [0.0]

    # --- Single-GPU reference on rank 0: the full wired pipeline forward ---
    e_ref = torch.zeros(1, dtype=dtype, device=device)
    f_ref = torch.zeros(n_global, 3, dtype=dtype, device=device)
    if rank == 0:
        ref_pipe = _ap_build_pipeline(aim_cut, device, dtype)
        batch = Batch.from_data_list(
            [_ap_make_data(an, positions, cell, pbc, device, dtype)]
        )
        compute_neighbors(batch, config=ref_pipe.model_config.neighbor_config)
        out = ref_pipe(batch)
        e_ref.copy_(out["energy"].sum().detach().view(1))
        f_ref.copy_(out["forces"].detach())
        del ref_pipe
    dist.broadcast(e_ref, src=0)
    dist.broadcast(f_ref, src=0)

    # --- Distributed composite over ONE shared partition (built at max cutoff) ---
    pipeline = _ap_build_pipeline(aim_cut, device, dtype)
    max_cut = max(aim_cut[0], _AP_PME_CUT)
    mesh = DeviceMesh("cuda", list(range(world_size)), mesh_dim_names=("domain",))
    base_config = DomainConfig(
        cutoff=max_cut, skin=_AP_SKIN, mesh=mesh, require_nondegenerate=True
    )
    full = (
        Batch.from_data_list([_ap_make_data(an, positions, cell, pbc, device, dtype)])
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
        e_local.view(1).to(e_ref.dtype),
        e_ref,
        rtol=1e-5,
        atol=0.1,
        msg=f"rank {rank}: wired composite energy mismatch ΔE={de:.3e}",
    )
    torch.testing.assert_close(
        f_owned,
        f_ref_owned,
        rtol=1e-2,
        atol=2e-3,
        msg=f"rank {rank}: wired composite forces mismatch |ΔF|max={df:.3e}",
    )


@_skip
def test_distributed_pipeline_aimnet2_pme_2ranks():
    """``DistributedPipelineModel(AIMNet2 charges -> PME)`` == single-GPU pipeline."""
    pytest.importorskip("nvalchemiops", reason="nvalchemiops not installed")
    pytest.importorskip("aimnet", reason="aimnet not installed")
    mp.spawn(
        _worker,
        args=(WORLD_SIZE, "29587", _ap_wired_pipeline_worker),
        nprocs=WORLD_SIZE,
    )


# ====================================================================
# Compiled composition — MACE (compiled) + DFT-D3 (eager)
# ====================================================================

_MDC_WARMUP_STEPS = 4
_MDC_STEADY_STEPS = 4
_MDC_JITTER = 0.05
_MDC_DFTD3_CUT = 5.0
_MDC_SKIN = 4.0


def _mdc_build_lattice(dtype: torch.dtype = torch.float64, seed: int = 0):
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


def _mdc_make_data(atomic_numbers, positions, masses, cell, pbc, device, dtype):
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


def _mdc_build_pipeline(mace_cut_holder: list[float], device, dtype):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from nvalchemi.models.mace import MACEWrapper

    from nvalchemi.models.dftd3 import DFTD3ModelWrapper
    from nvalchemi.models.pipeline import PipelineGroup, PipelineModelWrapper

    mace = MACEWrapper.from_checkpoint(
        "small", device=device, dtype=dtype, enable_cueq=False
    )
    mace_cut_holder[0] = float(mace.cutoff)
    dftd3 = DFTD3ModelWrapper(a1=_A1, a2=_A2, s8=_S8, cutoff=_MDC_DFTD3_CUT)
    return PipelineModelWrapper(
        groups=[
            PipelineGroup(steps=[mace], use_autograd=True),
            PipelineGroup(steps=[dftd3], use_autograd=False),
        ]
    )


def _mdc_compile_worker(rank: int, world_size: int) -> None:
    from torch._dynamo.utils import counters
    from torch.distributed import DeviceMesh

    from nvalchemi.distributed.distributed_pipeline import DistributedPipelineModel
    from nvalchemi.distributed.partitioner import SpatialPartitioner
    from nvalchemi.distributed.sharded_batch import ShardedBatch
    from nvalchemi.neighbors import compute_neighbors

    dtype = torch.float64
    device = torch.device(f"cuda:{rank}")
    positions0, an, masses, cell, pbc, box = _mdc_build_lattice(dtype=dtype)
    n_global = positions0.shape[0]
    mace_cut = [0.0]

    # --- Single-GPU reference on rank 0 at the initial (rattled) geometry ---
    e_ref = torch.zeros(1, dtype=dtype, device=device)
    f_ref = torch.zeros(n_global, 3, dtype=dtype, device=device)
    if rank == 0:
        ref_pipe = _mdc_build_pipeline(mace_cut, device, dtype)
        batch = Batch.from_data_list(
            [_mdc_make_data(an, positions0, masses, cell, pbc, device, dtype)]
        )
        compute_neighbors(batch, config=ref_pipe.model_config.neighbor_config)
        out = ref_pipe(batch)
        e_ref.copy_(out["energy"].sum().detach().view(1))
        f_ref.copy_(out["forces"].detach())
        del ref_pipe
    dist.broadcast(e_ref, src=0)
    dist.broadcast(f_ref, src=0)

    pipeline = _mdc_build_pipeline(mace_cut, device, dtype)
    max_cut = max(mace_cut[0], _MDC_DFTD3_CUT)
    mesh = DeviceMesh("cuda", list(range(world_size)), mesh_dim_names=("domain",))
    base_config = DomainConfig(cutoff=max_cut, skin=_MDC_SKIN, mesh=mesh)
    partitioner = SpatialPartitioner(
        config=base_config,
        cell_matrix=cell.to(device=device, dtype=dtype).unsqueeze(0),
        pbc=pbc.to(device).unsqueeze(0),
    )

    def _sharded(pos):
        full = (
            Batch.from_data_list(
                [_mdc_make_data(an, pos, masses, cell, pbc, device, dtype)]
            )
            if rank == 0
            else None
        )
        return ShardedBatch.from_batch(batch=full, mesh=mesh, config=base_config, src=0)

    gen = torch.Generator(device="cpu").manual_seed(11)
    graphs_after_warmup = [0]

    with DistributedPipelineModel(pipeline, base_config, compile=True) as dpm:
        for step in range(_MDC_WARMUP_STEPS + _MDC_STEADY_STEPS):
            if step == 0:
                pos = positions0
            else:
                disp = _MDC_JITTER * torch.randn(
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
            if step == _MDC_WARMUP_STEPS - 1:
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
        args=(WORLD_SIZE, "29588", _mdc_compile_worker),
        nprocs=WORLD_SIZE,
    )
