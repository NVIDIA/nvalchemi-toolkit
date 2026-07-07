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

"""Multi-GPU regression: DFT-D3(BJ) dispersion under halo-storage DD.

Gates distributed DFTD3: a 2-rank ``DistributedModel(DFTD3ModelWrapper)``
forward must match the single-GPU reference on total energy and per-atom
forces.

DFTD3 is a pure-halo / local model (no cross-rank collective). The wrapper
localizes the halo-padded inputs for the Warp kernel, emits per-atom
dispersion energies (the ops ``compute_atomic_energies`` output), and reduces
them with :func:`~nvalchemi.distributed.helpers.system_sum` (owned-slice +
all-reduce); forces are direct per-atom (PER_NODE, OWNED).

One subtlety vs LJ: a ghost atom's coordination number (and the CN-gradient
force term) depend on the ghost's own neighbors, which reach beyond the
dispersion cutoff. Exact forces therefore need a halo a little deeper than the
cutoff — set here via ``DomainConfig.skin``. The default box is sized
non-degenerate (box > 4*(cutoff+skin) so every rank develops remote atoms) and
``require_nondegenerate=True`` makes a trivial partition fail loud; override the
system via ``NVALCHEMI_DFTD3_BOX`` / ``NVALCHEMI_DFTD3_N_SIDE`` (keep it larger
than 4*(cutoff+skin)).

Requires 2+ CUDA GPUs and ``nvalchemiops`` with the DFTD3 kernels.

Run with::

    pytest test/distributed/model/test_dftd3_multigpu.py -v
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

# D3(BJ) parameters for PBE (Grimme 2010); a2 in Bohr.
_A1, _A2, _S8 = 0.4289, 4.4407, 0.7875


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
    """Rattled simple-cubic lattice of alternating Na/Cl, periodic.

    ``NVALCHEMI_DFTD3_N_SIDE`` atoms per side, ``NVALCHEMI_DFTD3_BOX`` Å. The
    correctness bar is matching ranks, not lattice realism. Rattled so the
    forces are non-trivial (a symmetric lattice gives ~zero forces and a
    vacuous check).
    """
    # Non-degenerate: cutoff caps at 5 Å, CN-skin 4 Å -> ghost 9 Å, so a 2-rank
    # split needs box > 36 Å. 44 Å / 14 = 3.14 Å spacing, 2744 atoms (DFTD3 is
    # cheap — no neural net).
    n_side = int(os.environ.get("NVALCHEMI_DFTD3_N_SIDE", 14))
    box = float(os.environ.get("NVALCHEMI_DFTD3_BOX", 44.0))

    coords = torch.arange(n_side, dtype=dtype) * (box / n_side)
    gx, gy, gz = torch.meshgrid(coords, coords, coords, indexing="ij")
    positions = torch.stack([gx.flatten(), gy.flatten(), gz.flatten()], dim=-1)
    n = positions.shape[0]

    g = torch.Generator().manual_seed(seed)
    positions = positions + 0.1 * torch.randn(positions.shape, dtype=dtype, generator=g)
    positions = positions % box

    sign = torch.ones(n, dtype=torch.long)
    sign[1::2] = -1
    atomic_numbers = torch.where(
        sign > 0, torch.full((n,), 11, dtype=torch.long), torch.full((n,), 17, dtype=torch.long)
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


def _dftd3_equivalence_worker(rank: int, world_size: int) -> None:
    from torch.distributed import DeviceMesh

    from nvalchemi.distributed.distributed_model import DistributedModel
    from nvalchemi.distributed.partitioner import SpatialPartitioner
    from nvalchemi.distributed.sharded_batch import ShardedBatch
    from nvalchemi.models.dftd3 import DFTD3ModelWrapper

    dtype = torch.float32
    device = torch.device(f"cuda:{rank}")

    positions, atomic_numbers, masses, cell, pbc = _build_lattice(dtype=dtype)
    n_global = positions.shape[0]
    box = float(cell[0, 0].item())
    cutoff = min(5.0, 0.45 * box)
    # Deeper halo than the cutoff so ghost coordination numbers (and the
    # CN-gradient force term) are complete -> machine-precision forces.
    cn_skin = 4.0

    from nvalchemi.neighbors import compute_neighbors

    # ---- Single-process reference on rank 0 ----
    e_ref_host = torch.zeros(1, dtype=dtype)
    f_ref_host = torch.zeros(n_global, 3, dtype=dtype)
    if rank == 0:
        ref_wrapper = DFTD3ModelWrapper(a1=_A1, a2=_A2, s8=_S8, cutoff=cutoff)
        ref_data = _make_data(atomic_numbers, positions, masses, cell, pbc, device, dtype)
        ref_batch = Batch.from_data_list([ref_data])
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
    dist_wrapper = DFTD3ModelWrapper(a1=_A1, a2=_A2, s8=_S8, cutoff=cutoff)
    mesh = DeviceMesh("cuda", list(range(world_size)), mesh_dim_names=("domain",))
    domain_config = DomainConfig(
        cutoff=cutoff, skin=cn_skin, mesh=mesh, require_nondegenerate=True
    )

    if rank == 0:
        full_batch = Batch.from_data_list(
            [_make_data(atomic_numbers, positions, masses, cell, pbc, device, dtype)]
        )
    else:
        full_batch = None

    sharded = ShardedBatch.from_batch(batch=full_batch, mesh=mesh, config=domain_config, src=0)
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

    e_delta = e_local.item() - e_ref.item()
    print(
        f"[dftd3-halo rank {rank}] dist_e={e_local.item():+.6f} "
        f"ref_e={e_ref.item():+.6f} Δ={e_delta:+.3e}",
        flush=True,
    )
    assert f_owned.shape[0] == local_n, (
        f"rank {rank}: force shape {f_owned.shape}, expected ({local_n}, 3)"
    )
    diff = (f_owned - f_ref_owned).detach()
    print(
        f"[dftd3-halo rank {rank}] |ΔF| max={diff.abs().max().item():.3e} "
        f"|F_ref| max={f_ref_owned.norm(dim=1).max().item():.3e}",
        flush=True,
    )

    torch.testing.assert_close(
        e_local.view(1), e_ref, rtol=1e-4, atol=1e-4,
        msg=f"rank {rank}: energy mismatch Δ={e_delta:+.3e}",
    )
    torch.testing.assert_close(
        f_owned, f_ref_owned, rtol=1e-3, atol=1e-4,
        msg=f"rank {rank}: per-atom forces disagree, max |ΔF|={diff.abs().max().item():.3e}",
    )


@_skip
def test_dftd3_dist_model_equivalence_2ranks():
    """``DistributedModel(DFTD3ModelWrapper)`` under halo matches single-GPU."""
    pytest.importorskip("nvalchemiops", reason="nvalchemiops not installed")
    mp.spawn(
        _worker,
        args=(WORLD_SIZE, "29581", _dftd3_equivalence_worker),
        nprocs=WORLD_SIZE,
    )
