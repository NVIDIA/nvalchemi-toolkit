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
"""Tests for :mod:`nvalchemi.training.distributed`."""

from __future__ import annotations

from typing import Any

import torch
from torch import distributed as dist

from nvalchemi.training.distributed import all_reduce_flags
from test.training.conftest import _FakeManager


class _RecordingWorld(_FakeManager):
    """Manager recording the reduction it was asked for and adding a peer's flag."""

    def __init__(self, *, peer: int | None = None, **kwargs: Any) -> None:
        """Report a world where rank *peer*, if given, has raised its flag."""
        super().__init__(**kwargs)
        self.peer = peer
        self.ops: list[Any] = []

    def all_reduce(self, tensor: torch.Tensor, *, op: Any = None) -> torch.Tensor:
        """Record *op* and merge in the peer's flag as a MAX reduce would."""
        self.ops.append(op)
        if self.peer is not None:
            tensor[self.peer] = 1
        return tensor


class TestAllReduceFlags:
    def test_a_single_process_gets_its_own_flag_without_a_collective(self) -> None:
        """One process has nobody to reduce with, so the answer is its own bit."""
        assert all_reduce_flags(True, _FakeManager(world_size=1)).tolist() == [1]
        assert all_reduce_flags(False, _FakeManager(world_size=1)).tolist() == [0]

    def test_no_manager_and_no_process_group_is_a_single_process(self) -> None:
        """Without a launcher the world is one rank, read from the environment."""
        assert not dist.is_initialized()

        assert all_reduce_flags(1).tolist() == [1]

    def test_this_rank_raises_its_own_entry(self) -> None:
        """The flag lands at the global rank's index in a world-sized vector."""
        world = _RecordingWorld(world_size=3, rank=1)

        assert all_reduce_flags(True, world).tolist() == [0, 1, 0]

    def test_the_reduction_is_a_max(self) -> None:
        """A MAX keeps every raised flag and never counts one twice."""
        world = _RecordingWorld(world_size=3, rank=1)

        all_reduce_flags(False, world)

        assert world.ops == [dist.ReduceOp.MAX]

    def test_a_peer_flag_comes_back_beside_this_rank_lowered_one(self) -> None:
        """Every rank reads the same vector, so a lowered rank still sees who raised."""
        world = _RecordingWorld(world_size=4, rank=0, peer=2)

        flags = all_reduce_flags(False, world)

        assert flags.tolist() == [0, 0, 1, 0]
        assert flags.nonzero().flatten().tolist() == [2]

    def test_an_integer_flag_is_read_as_a_bit(self) -> None:
        """Callers pass counts or booleans alike; anything truthy raises the entry."""
        world = _RecordingWorld(world_size=2, rank=1)

        assert all_reduce_flags(3, world).tolist() == [0, 1]
        assert all_reduce_flags(0, _RecordingWorld(world_size=2, rank=1)).tolist() == [
            0,
            0,
        ]

    def test_the_vector_is_integer_typed(self) -> None:
        """An int64 vector all-reduces under every backend and indexes cleanly."""
        assert (
            all_reduce_flags(True, _RecordingWorld(world_size=2)).dtype == torch.int64
        )
        assert all_reduce_flags(True, _FakeManager(world_size=1)).dtype == torch.int64
