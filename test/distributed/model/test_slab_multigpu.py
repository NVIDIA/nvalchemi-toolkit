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

"""Multi-GPU correctness gate for the slab correction under halo DD.

The slab correction depends on three global per-system moments
(``M = Σ q z``, ``M2 = Σ q z²``, ``Q = Σ q``). Under halo storage each rank
holds owned + ghost atoms, so a naive moment sum over the padded batch would
double-count ghosts. This gate proves the DD path is correct: a 2-rank
``DistributedModel(PMEModelWrapper / EwaldModelWrapper)`` forward with
``slab_correction=True`` must match the single-GPU reference on total energy
and per-atom forces, on a NON-degenerate spatial partition.

System: a 2D-periodic slab (pbc = [True, True, False]) with vacuum along z and
a large in-plane box so the x partition is non-degenerate
(box_xy > ~4 * (cutoff + skin)).

Requires 2+ CUDA GPUs and nvalchemiops with the slab kernels.
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

_CUTOFF = 8.0
_SKIN = 2.0


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


def _build_slab(dtype: torch.dtype = torch.float32, seed: int = 0):
    """2D-periodic NaCl-like slab: large in-plane box, vacuum along z.

    The in-plane box is large so the x partition is non-degenerate
    (> 4*(cutoff+skin)); the atoms occupy only the central slab in z so the
    correction is non-trivial and the net charge is (deliberately) non-zero to
    also exercise the Ballenegger background term.
    """
    # box_xy must clear ~4*(cutoff+skin) so the 2-rank x partition is genuinely
    # non-degenerate (each rank's halo does NOT cover the whole system).
    box_xy = float(os.environ.get("NVALCHEMI_SLAB_BOXXY", 80.0))
    box_z = float(os.environ.get("NVALCHEMI_SLAB_BOXZ", 60.0))
    n_side = int(os.environ.get("NVALCHEMI_SLAB_NSIDE", 10))

    xs = torch.linspace(0.0, box_xy, n_side + 1, dtype=dtype)[:-1]
    zs = torch.linspace(box_z * 0.35, box_z * 0.65, 3, dtype=dtype)
    gx, gy, gz = torch.meshgrid(xs, xs, zs, indexing="ij")
    positions = torch.stack([gx.flatten(), gy.flatten(), gz.flatten()], dim=-1)
    n = positions.shape[0]

    g = torch.Generator().manual_seed(seed)
    positions = positions + 0.1 * torch.randn(
        positions.shape, dtype=dtype, generator=g
    )
    # Wrap only the periodic in-plane axes; leave z inside the slab.
    positions[:, 0] = positions[:, 0] % box_xy
    positions[:, 1] = positions[:, 1] % box_xy

    signs = torch.ones(n, dtype=dtype)
    signs[1::2] = -1.0
    # Introduce a small net charge to also gate the Ballenegger background term.
    charges = signs
    charges[0] = charges[0] + 0.5

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
    cell = torch.diag(torch.tensor([box_xy, box_xy, box_z], dtype=dtype))
    pbc = torch.tensor([True, True, False])
    return positions, atomic_numbers, masses, charges, cell, pbc


def _make_wrapper(method: str):
    if method == "pme":
        from nvalchemi.models.pme import PMEModelWrapper

        return PMEModelWrapper(
            cutoff=_CUTOFF, hybrid_forces=False, slab_correction=True
        )
    from nvalchemi.models.ewald import EwaldModelWrapper

    return EwaldModelWrapper(
        cutoff=_CUTOFF, hybrid_forces=False, slab_correction=True
    )


def _slab_equivalence_worker(rank: int, world_size: int, method: str) -> None:
    from torch.distributed import DeviceMesh

    from nvalchemi.distributed.distributed_model import DistributedModel
    from nvalchemi.distributed.partitioner import SpatialPartitioner
    from nvalchemi.distributed.sharded_batch import ShardedBatch
    from nvalchemi.neighbors import compute_neighbors

    dtype = torch.float32
    device = torch.device(f"cuda:{rank}")

    positions, atomic_numbers, masses, charges, cell, pbc = _build_slab(dtype=dtype)
    n_global = positions.shape[0]

    def _make_data():
        return AtomicData(
            atomic_numbers=atomic_numbers.to(device),
            positions=positions.to(device=device, dtype=dtype).clone(),
            atomic_masses=masses.to(device=device, dtype=dtype),
            charges=charges.to(device=device, dtype=dtype),
            cell=cell.to(device=device, dtype=dtype).unsqueeze(0),
            pbc=pbc.to(device).unsqueeze(0),
            forces=torch.zeros(n_global, 3, device=device, dtype=dtype),
            energy=torch.zeros(1, 1, device=device, dtype=dtype),
        )

    # ---- Single-GPU reference on rank 0 ----
    e_ref_host = torch.zeros(1, dtype=dtype)
    f_ref_host = torch.zeros(n_global, 3, dtype=dtype)
    if rank == 0:
        ref_wrapper = _make_wrapper(method)
        ref_batch = Batch.from_data_list([_make_data()])
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
    dist_wrapper = _make_wrapper(method)
    mesh = DeviceMesh("cuda", list(range(world_size)), mesh_dim_names=("domain",))
    domain_config = DomainConfig(cutoff=_CUTOFF, skin=_SKIN, mesh=mesh)

    full_batch = Batch.from_data_list([_make_data()]) if rank == 0 else None
    sharded = ShardedBatch.from_batch(
        batch=full_batch, mesh=mesh, config=domain_config, src=0
    )
    local_n = sharded.n_owned

    with DistributedModel(dist_wrapper, domain_config) as dist_model:
        out = dist_model(sharded)

    e_local = out["energy"].sum().detach()
    f_owned = out["forces"].detach()

    # ---- Owned slice of the reference forces ----
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

    n_owned_other = torch.tensor([local_n], device=device)
    counts = [torch.zeros_like(n_owned_other) for _ in range(world_size)]
    dist.all_gather(counts, n_owned_other)
    owned_counts = [int(c.item()) for c in counts]

    e_delta = e_local.item() - e_ref.item()
    abs_diff = (f_owned - f_ref_owned).abs()
    print(
        f"[slab-{method} rank {rank}] owned_counts={owned_counts} "
        f"(non-degenerate: every rank > 0 and < {n_global})  "
        f"dist_e={e_local.item():+.6f} ref_e={e_ref.item():+.6f} Δ={e_delta:+.3e}  "
        f"|ΔF| max={abs_diff.max().item():.3e} mean={abs_diff.mean().item():.3e}",
        flush=True,
    )

    # Guard: the partition must be genuinely split across ranks.
    assert all(0 < c < n_global for c in owned_counts), (
        f"degenerate partition owned_counts={owned_counts}; increase box_xy"
    )
    assert f_owned.shape[0] == local_n
    assert f_ref_owned.shape[0] == local_n

    torch.testing.assert_close(
        e_local.view(1),
        e_ref,
        rtol=5e-4,
        atol=5e-4,
        msg=f"rank {rank}: slab DD energy mismatch Δ={e_delta:+.3e}",
    )
    torch.testing.assert_close(
        f_owned,
        f_ref_owned,
        rtol=1e-3,
        atol=5e-4,
        msg=f"rank {rank}: slab DD forces disagree, max|ΔF|={abs_diff.max().item():.3e}",
    )


@_skip
def test_pme_slab_dist_model_equivalence_2ranks():
    """DistributedModel(PMEModelWrapper, slab_correction=True) matches 1-GPU."""
    pytest.importorskip("nvalchemiops", reason="nvalchemiops not installed")
    mp.spawn(
        _worker,
        args=(WORLD_SIZE, "29591", _slab_equivalence_worker, "pme"),
        nprocs=WORLD_SIZE,
    )


@_skip
def test_ewald_slab_dist_model_equivalence_2ranks():
    """DistributedModel(EwaldModelWrapper, slab_correction=True) matches 1-GPU."""
    pytest.importorskip("nvalchemiops", reason="nvalchemiops not installed")
    mp.spawn(
        _worker,
        args=(WORLD_SIZE, "29592", _slab_equivalence_worker, "ewald"),
        nprocs=WORLD_SIZE,
    )
