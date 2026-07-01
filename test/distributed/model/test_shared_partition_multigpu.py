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

"""Prerequisite for ``DistributedPipelineModel``: one shared owned partition,
per-model halos.

Validates the foundational invariant of distributed model composition
(``proposal-distributed-pipeline.md`` §5 step 0): a single ``ShardedBatch``
(one owned partition, built once at the max cutoff) can drive **two different
models with different cutoffs / ghost widths**, each reproducing its
single-GPU result. Each model's halo is rebuilt at its own ghost width over
the shared owned set (``ShardedBatch.invalidate_padded_view`` between models);
the owned partition is never recomputed.

This is exactly the per-model-halo plan a ``DistributedPipelineModel`` will
orchestrate; gating it here de-risks the composite before it exists.

Requires 2+ CUDA GPUs and ``nvalchemiops`` with the DFTD3 kernels.
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
_CN_SKIN = 4.0  # CN-depth halo margin (DFTD3 ghost CN completeness)

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
    n_side = int(os.environ.get("NVALCHEMI_SHARED_N_SIDE", 6))
    box = float(os.environ.get("NVALCHEMI_SHARED_BOX", 18.0))
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
        sign > 0, torch.full((n,), 22.99, dtype=dtype), torch.full((n,), 35.45, dtype=dtype)
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


def _single_ref(wrapper, atomic_numbers, positions, masses, cell, pbc, device, dtype):
    from nvalchemi.neighbors import compute_neighbors

    data = _make_data(atomic_numbers, positions, masses, cell, pbc, device, dtype)
    batch = Batch.from_data_list([data])
    compute_neighbors(batch, config=wrapper.model_config.neighbor_config)
    out = wrapper(batch)
    return out["energy"].sum().detach().cpu().view(1), out["forces"].detach().cpu()


def _shared_partition_worker(rank: int, world_size: int) -> None:
    from torch.distributed import DeviceMesh

    from nvalchemi.distributed.distributed_model import DistributedModel
    from nvalchemi.distributed.partitioner import SpatialPartitioner
    from nvalchemi.distributed.sharded_batch import ShardedBatch
    from nvalchemi.models.dftd3 import DFTD3ModelWrapper

    dtype = torch.float32
    device = torch.device(f"cuda:{rank}")
    positions, atomic_numbers, masses, cell, pbc = _build_lattice(dtype=dtype)
    n_global = positions.shape[0]

    # Two models with DIFFERENT cutoffs -> different ghost widths.
    cut_a, cut_b = 5.0, 3.0
    max_cut = max(cut_a, cut_b)

    # --- Single-GPU references on rank 0 (one per model) ---
    refs = {}
    if rank == 0:
        for tag, cut in (("a", cut_a), ("b", cut_b)):
            w = DFTD3ModelWrapper(a1=_A1, a2=_A2, s8=_S8, cutoff=cut)
            refs[tag] = _single_ref(
                w, atomic_numbers, positions, masses, cell, pbc, device, dtype
            )
    e_ref = {
        t: torch.zeros(1, dtype=dtype, device=device) for t in ("a", "b")
    }
    f_ref = {
        t: torch.zeros(n_global, 3, dtype=dtype, device=device) for t in ("a", "b")
    }
    for t in ("a", "b"):
        if rank == 0:
            e_ref[t].copy_(refs[t][0].to(device))
            f_ref[t].copy_(refs[t][1].to(device))
        dist.broadcast(e_ref[t], src=0)
        dist.broadcast(f_ref[t], src=0)

    mesh = DeviceMesh("cuda", list(range(world_size)), mesh_dim_names=("domain",))
    # ONE shared owned partition, built at the MAX ghost width so the partition
    # cells comfortably hold every model's ghost layer.
    shared_config = DomainConfig(cutoff=max_cut, skin=_CN_SKIN, mesh=mesh)
    full = (
        Batch.from_data_list(
            [_make_data(atomic_numbers, positions, masses, cell, pbc, device, dtype)]
        )
        if rank == 0
        else None
    )
    sharded = ShardedBatch.from_batch(batch=full, mesh=mesh, config=shared_config, src=0)

    # Owned slice of this rank under the SHARED partition (one assignment, reused
    # by both models).
    partitioner = SpatialPartitioner(
        config=shared_config,
        cell_matrix=cell.to(device=device, dtype=dtype).unsqueeze(0),
        pbc=pbc.to(device).unsqueeze(0),
    )
    local_mask = (
        partitioner.assign_atoms_to_ranks(positions.to(device=device, dtype=dtype)) == rank
    )

    for tag, cut in (("a", cut_a), ("b", cut_b)):
        # Each model rebuilds its OWN halo (its ghost width) over the shared
        # owned partition; the owned set is never recomputed.
        sharded.invalidate_padded_view()
        wrapper = DFTD3ModelWrapper(a1=_A1, a2=_A2, s8=_S8, cutoff=cut)
        cfg = DomainConfig(cutoff=cut, skin=_CN_SKIN, mesh=mesh)
        with DistributedModel(wrapper, cfg) as dm:
            out = dm(sharded)
        e_local = out["energy"].sum().detach()
        f_owned = out["forces"].detach()
        f_ref_owned = f_ref[tag][local_mask]

        de = (e_local - e_ref[tag]).abs().item()
        df = (f_owned - f_ref_owned).abs().max().item()
        print(
            f"[shared-part rank {rank}] model={tag} cut={cut} "
            f"ΔE={de:.3e} |ΔF|max={df:.3e} n_owned={f_owned.shape[0]}",
            flush=True,
        )
        torch.testing.assert_close(
            e_local.view(1), e_ref[tag], rtol=1e-4, atol=1e-4,
            msg=f"rank {rank} model {tag}: energy mismatch ΔE={de:.3e}",
        )
        torch.testing.assert_close(
            f_owned, f_ref_owned, rtol=1e-3, atol=1e-4,
            msg=f"rank {rank} model {tag}: force mismatch |ΔF|max={df:.3e}",
        )


@_skip
def test_shared_partition_per_model_halo_2ranks():
    """One owned partition drives two different-cutoff models, each exact."""
    pytest.importorskip("nvalchemiops", reason="nvalchemiops not installed")
    mp.spawn(
        _worker,
        args=(WORLD_SIZE, "29583", _shared_partition_worker),
        nprocs=WORLD_SIZE,
    )
