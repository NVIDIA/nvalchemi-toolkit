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

"""Multi-GPU regression: Ewald under halo-storage domain decomposition.

Gates distributed Ewald: a 2-rank ``DomainParallel(EwaldModelWrapper)``
forward must match the single-GPU reference on total energy and per-atom
forces.

Exercises:

- :data:`~nvalchemi.distributed.spec.SPEC_EWALD_HALO` storage / scatter
  modes (halo, halo_correction, halo_read, system_reductions).
- :meth:`~nvalchemi.models.ewald.EwaldModelWrapper.distributed_setup`
  installing the staged-bindings handlers via
  :func:`~nvalchemi.distributed._core.escape_hatches.wrap_custom_op` with
  ``owned_slice_inputs`` + ``all_reduce_outputs``.
- Cross-rank reduction of partial structure factors via
  :func:`~nvalchemi.distributed._core.gather_primitives.distributed_all_reduce`.
- Per-atom reciprocal-space energy from the globally-reduced ``S(k)``
  on all local atoms, followed by per-system
  :func:`~nvalchemi.distributed._core.per_system.per_system_reduce` that
  slices halo rows off the source before reducing.

Requires:
* 2+ CUDA GPUs.
* ``nvalchemiops`` installed with the Ewald kernels.

Run with::

    pytest test/distributed/test_ewald_multigpu.py -v

The default system is non-degenerate (remote atoms on every rank) and the
DomainConfig sets ``require_nondegenerate=True`` so a trivial partition fails
loud rather than passing vacuously. Override the system via env (keep it
non-degenerate: box > 4*(cutoff+skin)):
    NVALCHEMI_EWALD_BOX=28.0 NVALCHEMI_EWALD_N_SIDE=10 pytest ...
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

_skip = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < WORLD_SIZE,
    reason=f"Need {WORLD_SIZE}+ CUDA GPUs",
)


# ======================================================================
# Harness
# ======================================================================


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


# ======================================================================
# System — cubic NaCl-like lattice (charge-neutral, periodic)
# ======================================================================


def _build_nacl(dtype: torch.dtype = torch.float32, seed: int = 0):
    """Simple-cubic lattice of alternating ±1 charges.

    2x2x2 NaCl supercell → 16 atoms. Env-overridable via
    ``NVALCHEMI_EWALD_N_SIDE`` (atoms per side) and
    ``NVALCHEMI_EWALD_BOX`` (Å). Keeps the test bounded; the
    correctness bar here is matching ranks, not lattice realism.
    """
    # Non-degenerate by default: cutoff caps at 5 Å (skin 0), so box must clear
    # 4*ghost = 20 Å for a 2-rank split to develop remote atoms. 28 Å / 10 = 2.8 Å
    # spacing (~NaCl), 1000 atoms.
    n_side = int(os.environ.get("NVALCHEMI_EWALD_N_SIDE", 10))
    box = float(os.environ.get("NVALCHEMI_EWALD_BOX", 28.0))

    coords = torch.arange(n_side, dtype=dtype) * (box / n_side)
    gx, gy, gz = torch.meshgrid(coords, coords, coords, indexing="ij")
    positions = torch.stack([gx.flatten(), gy.flatten(), gz.flatten()], dim=-1)
    n = positions.shape[0]

    g = torch.Generator().manual_seed(seed)
    positions = positions + 0.05 * torch.randn(
        positions.shape, dtype=dtype, generator=g
    )
    positions = positions % box

    # Alternating +1/-1 along the first axis — globally neutral for even n.
    signs = torch.ones(n, dtype=dtype)
    signs[1::2] = -1.0
    charges = signs
    # Use Na (11) / Cl (17) — atomic identity doesn't matter for Ewald.
    atomic_numbers = torch.where(
        signs > 0,
        torch.full((n,), 11, dtype=torch.long),
        torch.full((n,), 17, dtype=torch.long),
    )
    masses = torch.where(
        signs > 0,
        torch.full((n,), 22.99, dtype=dtype),
        torch.full((n,), 35.45, dtype=dtype),
    )
    cell = torch.eye(3, dtype=dtype) * box
    pbc = torch.ones(3, dtype=torch.bool)
    return positions, atomic_numbers, masses, charges, cell, pbc


# ======================================================================
# Worker
# ======================================================================


def _ewald_equivalence_worker(rank: int, world_size: int) -> None:
    """Single-GPU Ewald reference on rank 0 → broadcast → each rank
    runs the distributed forward and asserts its owned slice of forces
    + the total energy match the reference.

    Uses ``hybrid_forces=False`` and no ``stress`` because those modes
    route through the staged bindings under halo (see
    ``EwaldModelWrapper.forward``). Hybrid + stress have their own
    dispatch paths and are single-GPU only in this MVP; they'd
    land in a follow-up test once multi-GPU charge-grad /
    virial-staged wiring is verified.
    """
    from torch.distributed import DeviceMesh

    from nvalchemi.distributed.distributed_model import DistributedModel
    from nvalchemi.distributed.partitioner import SpatialPartitioner
    from nvalchemi.distributed.sharded_batch import ShardedBatch
    from nvalchemi.models.ewald import EwaldModelWrapper

    dtype = torch.float32
    device = torch.device(f"cuda:{rank}")

    positions, atomic_numbers, masses, charges, cell, pbc = _build_nacl(dtype=dtype)
    n_global = positions.shape[0]

    # ---- Single-process reference on rank 0 only ----
    e_ref_host = torch.zeros(1, dtype=dtype)
    f_ref_host = torch.zeros(n_global, 3, dtype=dtype)
    if rank == 0:
        ref_wrapper = EwaldModelWrapper(
            cutoff=min(5.0, 0.45 * cell[0, 0].item()), hybrid_forces=False
        )
        ref_data = AtomicData(
            atomic_numbers=atomic_numbers.to(device),
            positions=positions.to(device=device, dtype=dtype).clone(),
            atomic_masses=masses.to(device=device, dtype=dtype),
            charges=charges.to(device=device, dtype=dtype),
            cell=cell.to(device=device, dtype=dtype).unsqueeze(0),
            pbc=pbc.to(device).unsqueeze(0),
            forces=torch.zeros(n_global, 3, device=device, dtype=dtype),
            energy=torch.zeros(1, 1, device=device, dtype=dtype),
        )
        ref_batch = Batch.from_data_list([ref_data])
        from nvalchemi.neighbors import compute_neighbors

        compute_neighbors(ref_batch, config=ref_wrapper.model_config.neighbor_config)
        ref_out = ref_wrapper(ref_batch)
        e_ref_host = ref_out["energy"].sum().detach().cpu().view(1)
        f_ref_host = ref_out["forces"].detach().cpu()
        del ref_wrapper, ref_batch, ref_out

    e_ref = e_ref_host.to(device=device, dtype=dtype)
    f_ref = f_ref_host.to(device=device, dtype=dtype)
    dist.broadcast(e_ref, src=0)
    dist.broadcast(f_ref, src=0)

    # ---- Distributed forward ----
    dist_wrapper = EwaldModelWrapper(
        cutoff=min(5.0, 0.45 * cell[0, 0].item()), hybrid_forces=False
    )
    mesh = DeviceMesh("cuda", list(range(world_size)), mesh_dim_names=("domain",))

    cutoff = float(dist_wrapper.cutoff)
    domain_config = DomainConfig(
        cutoff=cutoff, skin=0.0, mesh=mesh, require_nondegenerate=True
    )

    if rank == 0:
        full_batch = Batch.from_data_list(
            [
                AtomicData(
                    atomic_numbers=atomic_numbers.to(device),
                    positions=positions.to(device=device, dtype=dtype).clone(),
                    atomic_masses=masses.to(device=device, dtype=dtype),
                    charges=charges.to(device=device, dtype=dtype),
                    cell=cell.to(device=device, dtype=dtype).unsqueeze(0),
                    pbc=pbc.to(device).unsqueeze(0),
                    forces=torch.zeros(n_global, 3, device=device, dtype=dtype),
                    energy=torch.zeros(1, 1, device=device, dtype=dtype),
                )
            ]
        )
    else:
        full_batch = None

    sharded = ShardedBatch.from_batch(
        batch=full_batch, mesh=mesh, config=domain_config, src=0
    )
    local_n = sharded.n_owned

    with DistributedModel(dist_wrapper, domain_config) as dist_model:
        out = dist_model(sharded)

    e_local = out["energy"].sum().detach()
    f_owned = out["forces"].detach()

    # ---- Recover this rank's owned slice of reference forces ----
    partitioner = SpatialPartitioner(
        config=domain_config,
        cell_matrix=cell.to(device=device, dtype=dtype).unsqueeze(0),
        pbc=pbc.to(device).unsqueeze(0),
    )
    rank_assignment = partitioner.assign_atoms_to_ranks(
        positions.to(device=device, dtype=dtype)
    )
    local_mask = rank_assignment == rank
    f_ref_owned = f_ref[local_mask]

    # ---- Diagnostics: energy delta + force error stats on BOTH ranks
    # print before the assert so we see the full picture even on failure.
    e_delta = e_local.item() - e_ref.item()
    print(
        f"[ewald-halo rank {rank}] "
        f"dist_e={e_local.item():+.6f}  ref_e={e_ref.item():+.6f}  "
        f"Δ={e_delta:+.3e}",
        flush=True,
    )

    assert f_owned.shape[0] == local_n, (
        f"rank {rank}: force shape mismatch — got {f_owned.shape}, "
        f"expected ({local_n}, 3)"
    )
    assert f_ref_owned.shape[0] == local_n, (
        f"rank {rank}: partitioner / ShardedBatch disagreement — "
        f"partitioner says {local_mask.sum().item()} atoms, "
        f"ShardedBatch says {local_n}"
    )

    diff = (f_owned - f_ref_owned).detach()
    abs_diff = diff.abs()
    ref_norm = f_ref_owned.norm(dim=1).clamp_min(1e-12)
    rel_per_atom = diff.norm(dim=1) / ref_norm
    worst = int(abs_diff.norm(dim=1).argmax().item())
    local_global_idx = torch.nonzero(local_mask, as_tuple=False).flatten()[worst].item()
    print(
        f"[ewald-halo rank {rank}] "
        f"|ΔF| max={abs_diff.max().item():.3e}  mean={abs_diff.mean().item():.3e}  "
        f"rms={(abs_diff.pow(2).mean().sqrt()).item():.3e}  "
        f"|ΔF|/|F_ref| max={rel_per_atom.max().item():.3e}  "
        f"median={rel_per_atom.median().item():.3e}  "
        f"|F_ref| max={f_ref_owned.norm(dim=1).max().item():.3e}  "
        f"min={f_ref_owned.norm(dim=1).min().item():.3e}\n"
        f"[ewald-halo rank {rank}] worst owned atom local_idx={worst} "
        f"global_idx={local_global_idx}  "
        f"dist_F={f_owned[worst].tolist()}  ref_F={f_ref_owned[worst].tolist()}",
        flush=True,
    )

    # ---- Assertions ----
    # fp32 + long-range kernels with an FFT-free direct k-sum: total
    # energy holds to ~1e-4 absolute; per-atom forces to ~1e-3 relative
    # (same tolerance the cueq/MACE multi-GPU test uses).
    torch.testing.assert_close(
        e_local.view(1),
        e_ref,
        rtol=1e-4,
        atol=1e-4,
        msg=(
            f"rank {rank}: energy mismatch Δ={e_delta:+.3e} "
            f"(dist={e_local.item():.6f}, ref={e_ref.item():.6f})"
        ),
    )
    torch.testing.assert_close(
        f_owned,
        f_ref_owned,
        rtol=1e-3,
        atol=1e-4,
        msg=(
            f"rank {rank}: per-atom forces disagree with single-process Ewald "
            f"reference — max |ΔF|={abs_diff.max().item():.3e}, "
            f"max |ΔF|/|F|={rel_per_atom.max().item():.3e}"
        ),
    )


@_skip
def test_ewald_dist_model_equivalence_2ranks():
    """Regression: ``DistributedModel(EwaldModelWrapper)`` under halo
    matches single-GPU Ewald on total energy and per-atom forces.

    Gates the staged-bindings + wrap_custom_op owned_slice + all_reduce
    path end-to-end: the per-rank partial structure factors sum
    correctly across the mesh, and the per-atom reciprocal energy
    drops halo rows via per_system_reduce at the final scatter.
    """
    pytest.importorskip("nvalchemiops", reason="nvalchemiops not installed")

    mp.spawn(
        _worker,
        args=(WORLD_SIZE, "29573", _ewald_equivalence_worker),
        nprocs=WORLD_SIZE,
    )
