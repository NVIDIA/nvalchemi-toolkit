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
"""End-to-end async-overlap gate: an OPAQUE MPNN + a spec that declares overlap.

The whole point (proposal §3.0): overlap is added to an ordinary, distribution-
agnostic model by a single line in its ``distribution_spec`` —
``SPEC_MPNN_GP.with_adapters(*overlap_adapters(self.model.layers))`` — and nothing
in the model body changes. This runs the toy two ways and requires they agree:

* single-process over all atoms (no DD context → the overlap adapters are the
  identity, so the plain scatter-add layers run directly);
* graph-parallel through ``DistributedModel`` over a balanced partition, where
  each installed OverlapAdapter intercepts its message layer and, via the
  strategy's all-gather exchange + sender-residency split, computes the layer in
  two buckets and sums them.

Energy and owned per-atom forces must match the single-process reference. Runs on
CPU/gloo (overlap OFF — synchronous two-pass — so this isolates correctness).
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
from nvalchemi.neighbors import compute_neighbors

_N_ATOMS = 24
_CUTOFF = 2.5


def _full_batch() -> Batch:
    g = torch.Generator().manual_seed(0)
    pos = torch.randn(_N_ATOMS, 3, dtype=torch.float64, generator=g)
    z = torch.randint(1, 4, (_N_ATOMS,), generator=g)
    masses = torch.ones(_N_ATOMS, dtype=torch.float64)
    data = AtomicData(
        positions=pos,
        atomic_numbers=z,
        atomic_masses=masses,
        cell=torch.eye(3, dtype=torch.float64).unsqueeze(0) * 100.0,
        pbc=torch.zeros(1, 3, dtype=torch.bool),
    )
    return Batch.from_data_list([data])


def _reference(model):
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
    os.environ.setdefault("MASTER_PORT", "29905")
    dist.init_process_group("gloo", rank=rank, world_size=world)
    try:
        from _toy_overlap_mpnn import ToyOverlapMPNNWrapper  # noqa: PLC0415
        from torch.distributed.device_mesh import init_device_mesh

        mesh = init_device_mesh("cpu", (world,))
        torch.manual_seed(0)
        model = ToyOverlapMPNNWrapper().double()

        e_ref, f_ref = _reference(model)

        cfg = DomainConfig(cutoff=_CUTOFF, mesh=mesh)
        full = _full_batch() if rank == 0 else None
        sharded = ShardedBatch.from_batch(
            full, mesh=mesh, config=cfg, src=0, partition_mode="contiguous_block"
        )
        # The spec carries the overlap adapters — that is the only distributed
        # declaration; the model is opaque.
        dist_model = DistributedModel(model, cfg, spec=model.distribution_spec)
        out = dist_model(sharded)

        counts = _owned_counts(_N_ATOMS, world)
        offset = sum(counts[:rank])
        n_owned = counts[rank]

        torch.testing.assert_close(out["energy"], e_ref, rtol=1e-9, atol=1e-9)
        torch.testing.assert_close(
            out["forces"], f_ref[offset : offset + n_owned], rtol=1e-8, atol=1e-8
        )
        if rank == 0:
            print("[overlap-mpnn] energy + owned forces match single-process")
        dist.barrier()
    finally:
        dist.destroy_process_group()


def test_overlap_mpnn_2ranks() -> None:
    mp.spawn(_worker, args=(2,), nprocs=2)


def test_overlap_mpnn_3ranks() -> None:
    mp.spawn(_worker, args=(3,), nprocs=3)
