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

"""Multi-GPU regression: PME under halo-storage domain decomposition.

Gates distributed PME: a 2-rank ``DomainParallel(PMEModelWrapper)``
forward must match the single-GPU reference on total energy and per-atom
forces.

Exercises:

- :data:`~nvalchemi.distributed.spec.SPEC_PME_HALO` storage modes.
- :meth:`~nvalchemi.models.pme.PMEModelWrapper.distributed_setup`
  installing handlers on the nvalchemiops
  ``_spline_spread`` / ``_batch_spline_spread`` torch custom ops.
- Cross-rank all-reduce of the partial charge mesh after B-spline
  spreading via the ``owned_slice_inputs`` + ``all_reduce_outputs``
  path of :func:`~nvalchemi.distributed._core.escape_hatches.wrap_custom_op`.
- Post-reduce FFT → Green's function → IFFT pipeline running
  identically on every rank (replicated mesh).
- Per-atom reciprocal energy + forces via ``spline_gather`` on all
  local atoms, with halo-row contributions dropped by the final
  per-system scatter.

Requires:
* 2+ CUDA GPUs.
* ``nvalchemiops`` installed with the PME / B-spline kernels.

Run with::

    pytest test/distributed/test_pme_multigpu.py -v

Override the test system via env:
    NVALCHEMI_PME_BOX=12.0 NVALCHEMI_PME_N_SIDE=3 pytest ...
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
    """Same layout as the Ewald multigpu test — PME's real-space path
    reuses the Ewald kernel, so the two tests cover identical geometry
    with different k-space algorithms."""
    n_side = int(os.environ.get("NVALCHEMI_PME_N_SIDE", 2))
    box = float(os.environ.get("NVALCHEMI_PME_BOX", 5.64))

    coords = torch.arange(n_side, dtype=dtype) * (box / n_side)
    gx, gy, gz = torch.meshgrid(coords, coords, coords, indexing="ij")
    positions = torch.stack([gx.flatten(), gy.flatten(), gz.flatten()], dim=-1)
    n = positions.shape[0]

    g = torch.Generator().manual_seed(seed)
    positions = positions + 0.05 * torch.randn(
        positions.shape, dtype=dtype, generator=g
    )
    positions = positions % box

    signs = torch.ones(n, dtype=dtype)
    signs[1::2] = -1.0
    charges = signs
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


def _pme_equivalence_worker(rank: int, world_size: int) -> None:
    """Single-GPU PME reference on rank 0 → broadcast → each rank
    runs the distributed forward and asserts its owned slice of forces
    + the total energy match the reference.

    Uses ``hybrid_forces=False`` — same constraint as the Ewald
    multigpu test; the hybrid + charge-grad path under distribution is
    covered by the pipeline composition tests.
    """
    from torch.distributed import DeviceMesh

    from nvalchemi.distributed.distributed_model import DistributedModel
    from nvalchemi.distributed.partitioner import SpatialPartitioner
    from nvalchemi.distributed.sharded_batch import ShardedBatch
    from nvalchemi.models.pme import PMEModelWrapper

    dtype = torch.float32
    device = torch.device(f"cuda:{rank}")

    positions, atomic_numbers, masses, charges, cell, pbc = _build_nacl(dtype=dtype)
    n_global = positions.shape[0]

    # ---- Single-process reference on rank 0 only ----
    e_ref_host = torch.zeros(1, dtype=dtype)
    f_ref_host = torch.zeros(n_global, 3, dtype=dtype)
    if rank == 0:
        ref_wrapper = PMEModelWrapper(
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
    dist_wrapper = PMEModelWrapper(
        cutoff=min(5.0, 0.45 * cell[0, 0].item()), hybrid_forces=False
    )
    mesh = DeviceMesh("cuda", list(range(world_size)), mesh_dim_names=("domain",))

    cutoff = float(dist_wrapper.cutoff)
    domain_config = DomainConfig(cutoff=cutoff, skin=0.0, mesh=mesh)

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

    # ---- Diagnostics: energy delta + force error stats on BOTH ranks,
    # printed before the assert so failures surface concrete numbers.
    e_delta = e_local.item() - e_ref.item()
    print(
        f"[pme-halo rank {rank}] "
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
        f"[pme-halo rank {rank}] "
        f"|ΔF| max={abs_diff.max().item():.3e}  mean={abs_diff.mean().item():.3e}  "
        f"rms={(abs_diff.pow(2).mean().sqrt()).item():.3e}  "
        f"|ΔF|/|F_ref| max={rel_per_atom.max().item():.3e}  "
        f"median={rel_per_atom.median().item():.3e}  "
        f"|F_ref| max={f_ref_owned.norm(dim=1).max().item():.3e}  "
        f"min={f_ref_owned.norm(dim=1).min().item():.3e}\n"
        f"[pme-halo rank {rank}] worst owned atom local_idx={worst} "
        f"global_idx={local_global_idx}  "
        f"dist_F={f_owned[worst].tolist()}  ref_F={f_ref_owned[worst].tolist()}",
        flush=True,
    )

    # ---- Assertions ----
    # fp32 + FFT-based PME: tolerances slightly looser than Ewald's
    # direct k-sum because of accumulated rounding in the mesh pipeline.
    torch.testing.assert_close(
        e_local.view(1),
        e_ref,
        rtol=5e-4,
        atol=5e-4,
        msg=(
            f"rank {rank}: energy mismatch Δ={e_delta:+.3e} "
            f"(dist={e_local.item():.6f}, ref={e_ref.item():.6f})"
        ),
    )
    torch.testing.assert_close(
        f_owned,
        f_ref_owned,
        rtol=1e-3,
        atol=5e-4,
        msg=(
            f"rank {rank}: per-atom forces disagree with single-process PME "
            f"reference — max |ΔF|={abs_diff.max().item():.3e}, "
            f"max |ΔF|/|F|={rel_per_atom.max().item():.3e}"
        ),
    )


@_skip
def test_pme_dist_model_equivalence_2ranks():
    """Regression: ``DistributedModel(PMEModelWrapper)`` under halo
    matches single-GPU PME on total energy and per-atom forces.

    Gates the ``_spline_spread`` owned_slice + all_reduce handler
    end-to-end. Verifies that the partial charge mesh summed across
    ranks produces the globally-correct mesh, that every rank's
    subsequent FFT / Green's function / IFFT pipeline is replicated
    correctly, and that per-atom spline_gather + corrections give
    the right per-system energy after the final per_system_reduce
    scatter.
    """
    pytest.importorskip("nvalchemiops", reason="nvalchemiops not installed")

    mp.spawn(
        _worker,
        args=(WORLD_SIZE, "29574", _pme_equivalence_worker),
        nprocs=WORLD_SIZE,
    )
