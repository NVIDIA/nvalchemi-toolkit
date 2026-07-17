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

"""Unit tests for the placement & routing foundation.

Pure-CPU constructor coverage for :class:`ShardRouting` (global<->local
round-trip over a permuted ownership).
"""

from __future__ import annotations

import torch

from nvalchemi.distributed._core.placement import ShardRouting

# --------------------------------------------------------------------------
# ShardRouting.from_assignment
# --------------------------------------------------------------------------


def _check_round_trip(assignment: torch.Tensor, world_size: int) -> None:
    """For every rank: build routing, then verify owner_rank is the assignment
    and that local_index assigns contiguous 0..n_owned-1 positions in global
    order to each rank's owned rows (the global<->local round-trip)."""
    n_global = assignment.shape[0]
    for rank in range(world_size):
        r = ShardRouting.from_assignment(assignment, rank=rank, world_size=world_size)
        assert r.n_global == n_global
        assert torch.equal(r.owner_rank, assignment.long())
        assert r.n_owned == int((assignment == rank).sum())
        # Each rank's owned global ids, in ascending order, must map to local
        # indices 0, 1, 2, ... (contiguous-per-rank ordering).
        for owner in range(world_size):
            owned_globals = torch.where(assignment == owner)[0]
            expected_local = torch.arange(owned_globals.shape[0], dtype=torch.long)
            assert torch.equal(r.local_index[owned_globals], expected_local)


def test_from_assignment_round_robin() -> None:
    # Round-robin ownership over 3 ranks => permuted (non-contiguous) ownership.
    assignment = torch.arange(9) % 3
    _check_round_trip(assignment, world_size=3)


def test_from_assignment_contiguous_blocks() -> None:
    # Contiguous-block ownership: rows [0,1,2]->0, [3,4]->1, [5,6,7]->2.
    assignment = torch.tensor([0, 0, 0, 1, 1, 2, 2, 2])
    _check_round_trip(assignment, world_size=3)


def test_from_assignment_uneven_and_empty_rank() -> None:
    # Rank 1 owns nothing; ownership is irregular.
    assignment = torch.tensor([0, 2, 0, 0, 2])
    _check_round_trip(assignment, world_size=3)
    r1 = ShardRouting.from_assignment(assignment, rank=1, world_size=3)
    assert r1.n_owned == 0


def test_from_assignment_world_size_inferred_matches_explicit() -> None:
    assignment = torch.tensor([0, 1, 0, 1, 1])
    inferred = ShardRouting.from_assignment(assignment, rank=0)
    explicit = ShardRouting.from_assignment(assignment, rank=0, world_size=2)
    assert torch.equal(inferred.owner_rank, explicit.owner_rank)
    assert torch.equal(inferred.local_index, explicit.local_index)
    assert inferred.n_owned == explicit.n_owned == 2


def test_from_assignment_empty() -> None:
    r = ShardRouting.from_assignment(torch.empty(0, dtype=torch.long), rank=0)
    assert r.n_global == 0
    assert r.n_owned == 0
    assert r.owner_rank.shape == (0,)
    assert r.local_index.shape == (0,)
