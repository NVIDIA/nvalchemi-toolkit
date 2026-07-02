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
"""End-to-end gate for the DENSE-``neighbor_matrix`` graph-parallel forward.

The dense-nbmat sibling of ``test_graph_parallel_model``. A toy MLIP over a dense
``[n, K]`` neighbour matrix (:class:`ToyGraphParallelDenseWrapper`, still on
``SPEC_MPNN_GP``) runs single-process over all atoms and graph-parallel through
``DistributedModel``, and the two must agree on energy + owned forces.

Exercises the dense path specifically: ``_graph_parallel_owned_nbmat`` (owned
receiver rows sliced from the full dense matrix, global sender columns) + the
``NeighborListFormat.MATRIX`` branch in ``_graph_partition_run_forward`` + the
per-layer node all-gather's reduce-scatter adjoint (forces) + sharded output
consolidation. This is the machinery PME real-space / AIMNet2 (#140/#141) ride.
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from nvalchemi.data.atomic_data import AtomicData
from nvalchemi.data.batch import Batch
from nvalchemi.distributed.config import DomainConfig
from nvalchemi.distributed.distributed_model import DistributedModel
from nvalchemi.distributed.sharded_batch import ShardedBatch
from nvalchemi.distributed.spec import SPEC_MPNN_GP
from nvalchemi.neighbors import compute_neighbors

_N_ATOMS = 24
_CUTOFF = 2.5


def _full_batch() -> Batch:
    g = torch.Generator().manual_seed(0)
    pos = torch.randn(_N_ATOMS, 3, dtype=torch.float64, generator=g)
    z = torch.randint(1, 4, (_N_ATOMS,), generator=g)
    masses = torch.ones(_N_ATOMS, dtype=torch.float64)
    # Finite (large) box, non-periodic: pure distance cutoff, cell invertible for
    # the spatial partitioner the ShardedBatch builds unconditionally.
    data = AtomicData(
        positions=pos,
        atomic_numbers=z,
        atomic_masses=masses,
        cell=torch.eye(3, dtype=torch.float64).unsqueeze(0) * 100.0,
        pbc=torch.zeros(1, 3, dtype=torch.bool),
    )
    return Batch.from_data_list([data])


def _reference(model):
    """Single-process energy + forces over the full system (dense nbmat)."""
    batch = _full_batch()
    compute_neighbors(batch, config=model.model_config.neighbor_config)
    pos = batch._atoms_group["positions"].detach().requires_grad_(True)
    batch._atoms_group["positions"] = pos
    energy = model(batch)["energy"]
    (grad,) = torch.autograd.grad(energy.sum(), pos)
    return energy.detach(), -grad


def _owned_counts(n: int, world: int) -> list[int]:
    per_rank = max(n // world, 1)
    counts = [per_rank] * world
    counts[-1] = n - per_rank * (world - 1)
    return counts


def _worker(rank: int, world: int) -> None:
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29904")
    dist.init_process_group("gloo", rank=rank, world_size=world)
    from _toy_graph_parallel_dense import ToyGraphParallelDenseWrapper  # noqa: PLC0415
    from torch.distributed.device_mesh import init_device_mesh

    mesh = init_device_mesh("cpu", (world,))

    torch.manual_seed(0)
    model = ToyGraphParallelDenseWrapper().double()

    e_ref, f_ref = _reference(model)

    cfg = DomainConfig(cutoff=_CUTOFF, mesh=mesh)
    full = _full_batch() if rank == 0 else None
    sharded = ShardedBatch.from_batch(
        full, mesh=mesh, config=cfg, src=0, partition_mode="contiguous_block"
    )
    dist_model = DistributedModel(model, cfg, spec=SPEC_MPNN_GP)
    out = dist_model(sharded)

    counts = _owned_counts(_N_ATOMS, world)
    offset = sum(counts[:rank])
    n_owned = counts[rank]

    torch.testing.assert_close(out["energy"], e_ref, rtol=1e-9, atol=1e-9)
    torch.testing.assert_close(
        out["forces"], f_ref[offset : offset + n_owned], rtol=1e-8, atol=1e-8
    )
    if rank == 0:
        print(f"[gp-dense w={world}] energy + owned forces match single-process")
    dist.barrier()
    dist.destroy_process_group()


def test_graph_parallel_dense_model_2ranks() -> None:
    mp.spawn(_worker, args=(2,), nprocs=2)


def test_graph_parallel_dense_model_3ranks() -> None:
    mp.spawn(_worker, args=(3,), nprocs=3)


if __name__ == "__main__":
    for w in (2, 3):
        mp.spawn(_worker, args=(w,), nprocs=w)
