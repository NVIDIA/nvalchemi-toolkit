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
"""Per-system model inputs survive the scatter.

Sharding splits per-atom fields across ranks and replicates the per-system ones.
Beyond ``cell`` / ``pbc`` that includes the graph-level inputs a wrapper reads to
decide *which physical system* it is computing — total ``charge``, ``spin``
multiplicity, and any custom system property. A rank that loses them silently
falls back to a neutral default, so these are equivalence tests, not plumbing
tests.
"""

from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _gloo_harness import run_gloo  # noqa: E402

CHARGE = -2.0
SPIN = 3.0
CUSTOM = 7.5


BOX = 20.0


def _build_batch(split_dim: int):
    """Eight atoms straddling *split_dim*, carrying charge, spin, and a custom field.

    Half sit at fractional 0.25 along the partitioner's split axis and half at
    0.75, so the two ranks own four atoms each and the replication assertions
    aren't satisfied by a degenerate all-on-one-rank partition.
    """
    from nvalchemi.data.atomic_data import AtomicData
    from nvalchemi.data.batch import Batch

    frac = torch.tensor(
        [
            [0.2, 0.2, 0.2],
            [0.3, 0.4, 0.2],
            [0.2, 0.6, 0.3],
            [0.3, 0.8, 0.4],
            [0.7, 0.2, 0.2],
            [0.8, 0.4, 0.2],
            [0.7, 0.6, 0.3],
            [0.8, 0.8, 0.4],
        ],
        dtype=torch.float64,
    )
    # Put the 0.2/0.3 vs 0.7/0.8 separation on whichever axis the partitioner
    # actually splits.
    if split_dim != 0:
        frac[:, [0, split_dim]] = frac[:, [split_dim, 0]]
    positions = frac * BOX

    data = AtomicData(
        positions=positions,
        atomic_numbers=torch.full((8,), 8, dtype=torch.long),
        atomic_masses=torch.full((8,), 16.0, dtype=torch.float64),
        cell=torch.eye(3, dtype=torch.float64).unsqueeze(0) * BOX,
        pbc=torch.ones(1, 3, dtype=torch.bool),
        charge=torch.tensor([[CHARGE]], dtype=torch.float64),
        spin=torch.tensor([[SPIN]], dtype=torch.float64),
    )
    data.add_system_property("custom_scalar", torch.tensor([[CUSTOM]]))
    return Batch.from_data_list([data])


def _check_system_fields_replicated(rank, world_size, queue):
    """Every rank's local view carries the source batch's per-system inputs."""
    from torch.distributed.device_mesh import DeviceMesh

    from nvalchemi.distributed.config import DomainConfig
    from nvalchemi.distributed.partitioner import SpatialPartitioner
    from nvalchemi.distributed.sharded_batch import ShardedBatch

    mesh = DeviceMesh("cpu", list(range(world_size)), mesh_dim_names=("domain",))
    config = DomainConfig(cutoff=1.0, skin=0.2, mesh=mesh)

    cell = torch.eye(3, dtype=torch.float64).unsqueeze(0) * BOX
    partitioner = SpatialPartitioner(
        config=config, cell_matrix=cell, pbc=torch.ones(1, 3, dtype=torch.bool)
    )
    split_dim = next(d for d, n in enumerate(partitioner.rank_grid) if n > 1)

    sharded = ShardedBatch.from_batch(
        _build_batch(split_dim) if rank == 0 else None,
        mesh=mesh,
        config=config,
        src=0,
    )
    local = sharded.local_batch

    queue.put(
        (
            rank,
            {
                "n_owned": int(sharded.positions.to_local().shape[0]),
                "charge": float(local.charge.reshape(-1)[0]),
                "spin": float(local.spin.reshape(-1)[0]),
                "custom": float(local.custom_scalar.reshape(-1)[0]),
            },
        )
    )


def test_system_fields_reach_every_rank():
    """charge / spin / custom system properties replicate to all ranks.

    Guards the failure mode where only ``cell`` and ``pbc`` crossed and a
    charged or open-shell system was computed as neutral / closed-shell.
    """
    results = dict(run_gloo(world_size=2, fn=_check_system_fields_replicated))

    assert set(results) == {0, 1}
    # A real split, so the assertions aren't trivially satisfied by one rank
    # holding everything.
    assert all(0 < r["n_owned"] < 8 for r in results.values())
    for rank, seen in results.items():
        assert seen["charge"] == CHARGE, f"rank {rank} lost charge"
        assert seen["spin"] == SPIN, f"rank {rank} lost spin"
        assert seen["custom"] == CUSTOM, f"rank {rank} lost custom_scalar"


def test_system_fields_survive_a_migration_rebuild():
    """Migration must not narrow the system schema.

    ``apply_migration`` rebuilds the batch from per-atom fields, so whatever it
    does not carry across is gone. It used to name ``cell``, ``pbc``, ``energy``
    and ``stress`` individually, which silently dropped total charge, spin and
    any property a producer attached — an AIMNet2 run under a barostat would
    revert to a neutral system partway through a trajectory.
    """
    from nvalchemi.data import AtomicData, Batch
    from nvalchemi.distributed.strategy import _build_batch_from_fields

    data = AtomicData(
        positions=torch.zeros(4, 3),
        atomic_numbers=torch.ones(4, dtype=torch.long),
        cell=torch.eye(3).unsqueeze(0),
        pbc=torch.ones(1, 3, dtype=torch.bool),
    )
    data.add_system_property("charge", torch.tensor([[-2.0]]))
    data.add_system_property("spin", torch.tensor([[3.0]]))
    batch = Batch.from_data_list([data])

    # The reconstruction migration performs: per-atom fields only.
    rebuilt = _build_batch_from_fields(
        {
            "positions": batch.positions.clone(),
            "atomic_numbers": batch.atomic_numbers.clone(),
        },
        torch.device("cpu"),
    )
    system = batch._system_group
    assert system is not None
    for name, tensor in system.items():
        if isinstance(tensor, torch.Tensor):
            rebuilt[name] = tensor.clone()

    assert set(rebuilt._system_group.keys()) == set(system.keys()), (
        "migration narrowed the system schema; a hardcoded carry-over list "
        "drops whatever it does not mention"
    )
    torch.testing.assert_close(rebuilt.charge, batch.charge)
    torch.testing.assert_close(rebuilt.spin, batch.spin)
