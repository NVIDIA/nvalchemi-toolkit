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

"""Subgroup correctness for pipeline × domain (2-D) meshes.

With a ``(pipeline=2, domain=2)`` mesh, the domain groups are global ranks
``{0, 1}`` and ``{2, 3}``.  Both groups number their members locally as
``{0, 1}``, so these tests catch collectives that accidentally use a local rank
as a global rank or fall back to the four-rank world group.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
import torch.distributed as dist

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _gloo_harness import run_gloo  # noqa: E402

pytestmark = pytest.mark.skipif(
    not dist.is_gloo_available(), reason="gloo backend required"
)


class _LocalRows:
    """Small ShardTensor stand-in for properties that only call ``to_local``."""

    def __init__(self, rows: torch.Tensor) -> None:
        self._rows = rows

    def to_local(self) -> torch.Tensor:
        return self._rows


def _domain_submesh() -> Any:
    """Build this rank's two-rank domain row from a 2 × 2 device mesh."""
    from torch.distributed import init_device_mesh

    mesh_2d = init_device_mesh("cpu", (2, 2), mesh_dim_names=("pipeline", "domain"))
    return mesh_2d["domain"]


def _broadcast_state_worker(rank: int, world_size: int, queue: Any) -> None:
    del world_size

    from nvalchemi.distributed._dynamics_coordinator import (
        DynamicsDistributionCoordinator,
    )

    domain = _domain_submesh()
    domain_rank = domain.get_local_rank()
    state = SimpleNamespace(controller=torch.tensor([rank], dtype=torch.int64))
    dynamics = SimpleNamespace(
        __dd_thermo_kind__="nhc",
        __dd_replicated__=("controller",),
        _state=state,
    )
    strategy = SimpleNamespace(process_group=domain.get_group())

    coordinator = DynamicsDistributionCoordinator(dynamics, strategy)
    coordinator.broadcast_state(SimpleNamespace())

    # Every domain row must receive its own local-rank-0 value:
    # group {0, 1} -> 0 and group {2, 3} -> 2.
    expected_lead = rank - domain_rank
    torch.testing.assert_close(
        state.controller, torch.tensor([expected_lead], dtype=torch.int64)
    )
    queue.put((rank, int(state.controller.item())))


def test_controller_state_broadcast_uses_each_domain_group_lead() -> None:
    """The second domain row must broadcast from global rank 2, not rank 0."""
    results = run_gloo(world_size=4, fn=_broadcast_state_worker, timeout_sec=30.0)
    assert sorted(results) == [(0, 0), (1, 0), (2, 2), (3, 2)]


def _rank_assignment_worker(rank: int, world_size: int, queue: Any) -> None:
    del world_size

    from nvalchemi.distributed.sharded_batch import ShardedBatch

    domain = _domain_submesh()
    domain_rank = domain.get_local_rank()
    pipeline_rank = rank // 2

    # Make the two domain rows deliberately different:
    #   group {0, 1}: local counts [1, 2]
    #   group {2, 3}: local counts [3, 4]
    group_counts = [1, 2] if pipeline_rank == 0 else [3, 4]
    local_n = group_counts[domain_rank]
    n_global = sum(group_counts)

    sharded = ShardedBatch(
        mesh=domain,
        atom_fields={"positions": _LocalRows(torch.zeros(local_n, 3))},
        cell=torch.eye(3).unsqueeze(0),
        pbc=torch.zeros(1, 3, dtype=torch.bool),
        n_global=n_global,
    )

    got = sharded.rank_assignment
    expected = torch.tensor(
        [0] * group_counts[0] + [1] * group_counts[1], dtype=torch.int64
    )
    torch.testing.assert_close(got, expected)
    queue.put((rank, got.tolist()))


def test_rank_assignment_gathers_counts_within_domain_group() -> None:
    """Ownership counts from the other pipeline row must not be gathered."""
    results = run_gloo(world_size=4, fn=_rank_assignment_worker, timeout_sec=30.0)
    assert sorted(results) == [
        (0, [0, 1, 1]),
        (1, [0, 1, 1]),
        (2, [0, 0, 0, 1, 1, 1, 1]),
        (3, [0, 0, 0, 1, 1, 1, 1]),
    ]
