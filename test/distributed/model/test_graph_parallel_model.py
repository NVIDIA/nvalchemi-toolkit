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
"""End-to-end gates for the graph-parallel forward through ``DistributedModel``.

Each toy MLIP runs two ways over the same system and must agree on energy +
owned per-atom forces:

* single-process over all atoms (no DD context — the intent verbs are
  identities);
* graph-parallel through ``DistributedModel`` over a balanced index partition,
  where the framework hands each rank its owned rows plus a ``neighbor_list`` /
  ``neighbor_matrix`` (global senders, owned-local receivers), all-gathers node
  features per layer, and all-reduces the owned per-graph energy.

Three toys pin three distinct graph-parallel branches (all gloo / CPU):

* ``mpnn`` — iterated message passing on the COO ``neighbor_list``
  (``_graph_partition_run_forward``, per-layer node all-gather + reduce-scatter
  adjoint).
* ``dense`` — dense ``[n, K]`` ``neighbor_matrix`` (``_graph_parallel_owned_nbmat``
  + the ``NeighborListFormat.MATRIX`` branch); the machinery PME real-space /
  AIMNet2 ride.
* ``dense_full`` — ``gp_replicate_geometry`` full-position path
  (``_graph_parallel_dense_full_autograd``, owned-aware ``node_energy_key``
  reduction + autograd over the full-position leaf); the path PME real-space rides.
"""

from __future__ import annotations

import os

import pytest
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

# Per-toy rendezvous port so concurrently-parametrized spawns do not collide.
_PORTS = {"mpnn": "29903", "dense": "29904", "dense_full": "29905"}


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


def _build_toy(kind: str):
    """Instantiate a toy wrapper and resolve the spec its GP branch declares."""
    if kind == "mpnn":
        from _toy_graph_parallel_mpnn import (
            ToyGraphParallelMPNNWrapper,  # noqa: PLC0415
        )

        return ToyGraphParallelMPNNWrapper().double(), SPEC_MPNN_GP
    if kind == "dense":
        from _toy_graph_parallel_dense import (  # noqa: PLC0415
            ToyGraphParallelDenseWrapper,
        )

        return ToyGraphParallelDenseWrapper().double(), SPEC_MPNN_GP
    from _toy_graph_parallel_dense_full import (  # noqa: PLC0415
        ToyGraphParallelDenseFullWrapper,
    )

    model = ToyGraphParallelDenseFullWrapper().double()
    return model, model.distribution_spec


def _worker(rank: int, world: int, kind: str, port: str) -> None:
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", port)
    dist.init_process_group("gloo", rank=rank, world_size=world)
    from torch.distributed.device_mesh import init_device_mesh  # noqa: PLC0415

    mesh = init_device_mesh("cpu", (world,))

    torch.manual_seed(0)
    model, spec = _build_toy(kind)

    e_ref, f_ref = _reference(model)

    cfg = DomainConfig(cutoff=_CUTOFF, mesh=mesh)
    full = _full_batch() if rank == 0 else None
    sharded = ShardedBatch.from_batch(
        full, mesh=mesh, config=cfg, src=0, partition_mode="contiguous_block"
    )
    dist_model = DistributedModel(model, cfg, spec=spec)
    out = dist_model(sharded)

    counts = _owned_counts(_N_ATOMS, world)
    offset = sum(counts[:rank])
    n_owned = counts[rank]

    torch.testing.assert_close(out["energy"], e_ref, rtol=1e-9, atol=1e-9)
    torch.testing.assert_close(
        out["forces"], f_ref[offset : offset + n_owned], rtol=1e-8, atol=1e-8
    )
    if rank == 0:
        print(f"[gp-{kind} w={world}] energy + owned forces match single-process")
    dist.barrier()
    dist.destroy_process_group()


@pytest.mark.parametrize("kind", ["mpnn", "dense", "dense_full"])
@pytest.mark.parametrize("world", [2, 3])
def test_graph_parallel_model(kind: str, world: int) -> None:
    mp.spawn(_worker, args=(world, kind, _PORTS[kind]), nprocs=world)


if __name__ == "__main__":
    for _kind in ("mpnn", "dense", "dense_full"):
        for _w in (2, 3):
            mp.spawn(_worker, args=(_w, _kind, _PORTS[_kind]), nprocs=_w)
