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
"""End-to-end gate for the graph-parallel forward through ``DistributedModel``.

A toy MPNN (:class:`ToyGraphParallelMPNNWrapper`) on ``SPEC_MPNN_GP`` runs two
ways and must agree:

* single-process over all atoms (no DD context — the intent verbs are
  identities);
* graph-parallel through ``DistributedModel`` over a balanced index partition,
  where the framework hands each rank its owned rows plus a ``neighbor_list``
  with global senders / owned-local receivers, all-gathers the node features per
  layer, and all-reduces the owned per-graph energy.

Exercises the whole seam: ``GraphParallelPolicy`` topology, ``ShardedBatch``
contiguous-block partition, owned-target edge prep, the per-layer replicate
collective and its reduce-scatter adjoint (forces), and sharded output
consolidation. Energy and per-atom forces must match the single-process
reference.
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
    # Finite (large) box, non-periodic: positions fit comfortably so the graph
    # is a pure distance cutoff, while the cell stays invertible for the spatial
    # partitioner the ShardedBatch builds unconditionally.
    data = AtomicData(
        positions=pos,
        atomic_numbers=z,
        atomic_masses=masses,
        cell=torch.eye(3, dtype=torch.float64).unsqueeze(0) * 100.0,
        pbc=torch.zeros(1, 3, dtype=torch.bool),
    )
    return Batch.from_data_list([data])


def _reference(model):
    """Single-process energy + forces over the full system."""
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
    os.environ.setdefault("MASTER_PORT", "29903")
    dist.init_process_group("gloo", rank=rank, world_size=world)
    from _toy_graph_parallel_mpnn import ToyGraphParallelMPNNWrapper  # noqa: PLC0415
    from torch.distributed.device_mesh import init_device_mesh

    mesh = init_device_mesh("cpu", (world,))

    torch.manual_seed(0)
    model = ToyGraphParallelMPNNWrapper().double()

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
        print(f"[gp-model w={world}] energy + owned forces match single-process")
    dist.barrier()
    dist.destroy_process_group()


def test_graph_parallel_model_2ranks() -> None:
    mp.spawn(_worker, args=(2,), nprocs=2)


def test_graph_parallel_model_3ranks() -> None:
    mp.spawn(_worker, args=(3,), nprocs=3)


if __name__ == "__main__":
    for w in (2, 3):
        mp.spawn(_worker, args=(w,), nprocs=w)
