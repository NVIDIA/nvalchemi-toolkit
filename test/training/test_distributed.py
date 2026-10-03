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

import os
import socket
from typing import Any

import pytest
import torch
from torch import distributed as dist

from nvalchemi.training.distributed import (
    all_gather_objects,
    all_gather_rows,
    all_reduce_flags,
)
from test.training.conftest import _FakeManager

_SHARD_SIZES = (2, 3)
"""Rows each of two ranks holds, unequal so the padding path is exercised."""


def _free_port() -> int:
    """Return an available localhost TCP port for process-group setup."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _shard(rank: int) -> torch.Tensor:
    """Return *rank*'s rows, distinct across ranks so their order can be read off."""
    return torch.arange(float(_SHARD_SIZES[rank])).reshape(-1, 1) + 10.0 * rank


def _run_gather_worker(
    rank: int, world_size: int, port: int, result_queue: Any
) -> None:
    """Gather this rank's shard and a verdict object, then report what came back."""
    os.environ.update(
        {
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": str(port),
            "RANK": str(rank),
            "WORLD_SIZE": str(world_size),
        }
    )
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    try:
        rows = _shard(rank).requires_grad_(True)
        world, mine = all_gather_rows(rows)
        world.sum().backward()
        plain, _ = all_gather_rows(rows.detach(), differentiable=False)
        result_queue.put(
            (
                rank,
                {
                    "world": world.detach().tolist(),
                    "plain": plain.tolist(),
                    "mine": [mine.start, mine.stop],
                    "grad": rows.grad.tolist(),
                    "objects": all_gather_objects({"rank": rank}),
                },
            )
        )
    finally:
        dist.destroy_process_group()


def _spawn_two_ranks() -> dict[int, dict[str, Any]]:
    """Run the gather worker on two spawned ranks and collect their reports."""
    ctx = torch.multiprocessing.get_context("spawn")
    result_queue = ctx.Queue()
    port = _free_port()
    procs = [
        ctx.Process(target=_run_gather_worker, args=(rank, 2, port, result_queue))
        for rank in range(2)
    ]
    for proc in procs:
        proc.start()
    results: dict[int, dict[str, Any]] = {}
    for _ in procs:
        rank, payload = result_queue.get(timeout=120)
        results[rank] = payload
    for proc in procs:
        proc.join(timeout=60)
        assert proc.exitcode == 0
    return results


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


class TestAllGatherRows:
    """Single-process behavior of the row gather; the collective runs below."""

    def test_a_single_process_gets_its_tensor_back_without_a_collective(self) -> None:
        """One rank is the whole world, so the rows come back as they are."""
        rows = torch.randn(3, 2, requires_grad=True)

        world, mine = all_gather_rows(rows)

        assert world is rows
        assert mine == slice(0, 3)

    def test_a_single_process_ignores_the_differentiable_flag(self) -> None:
        """With no gather there is no graph to cut, so the flag changes nothing."""
        rows = torch.randn(3, 2)
        world, mine = all_gather_rows(rows, differentiable=False)
        assert world is rows and mine == slice(0, 3)

    def test_a_multi_rank_world_without_a_process_group_is_refused(self) -> None:
        """A manager reporting peers cannot be gathered without a group to reach them."""
        with pytest.raises(RuntimeError, match="no process group is initialized"):
            all_gather_rows(torch.zeros(2, 1), _FakeManager(world_size=2))


class TestAllGatherObjects:
    """Single-process behavior of the object gather; the collective runs below."""

    def test_a_single_process_gets_a_one_entry_list(self) -> None:
        """The world of one rank holds that rank's object alone."""
        verdict = {"empty": False}
        assert all_gather_objects(verdict) == [verdict]

    def test_a_multi_rank_world_without_a_process_group_is_refused(self) -> None:
        """A manager reporting peers cannot be gathered without a group to reach them."""
        with pytest.raises(RuntimeError, match="no process group is initialized"):
            all_gather_objects([3], _FakeManager(world_size=2))


@pytest.mark.skipif(not dist.is_gloo_available(), reason="gloo backend required")
def test_two_cpu_ranks_gather_unequal_shards_in_rank_order() -> None:
    """Both ranks see every row once, in rank order, and find their own rows."""
    results = _spawn_two_ranks()

    expected = torch.cat([_shard(0), _shard(1)]).tolist()
    assert set(results) == {0, 1}
    for rank, result in results.items():
        assert result["world"] == expected
        assert result["plain"] == expected
        start = sum(_SHARD_SIZES[:rank])
        assert result["mine"] == [start, start + _SHARD_SIZES[rank]]
        assert result["objects"] == [{"rank": 0}, {"rank": 1}]


@pytest.mark.skipif(not dist.is_gloo_available(), reason="gloo backend required")
def test_two_cpu_ranks_receive_the_gradient_of_every_ranks_use() -> None:
    """A loss over the world tensor on every rank sends each rank world_size copies back."""
    results = _spawn_two_ranks()

    for rank, result in results.items():
        assert result["grad"] == torch.full((_SHARD_SIZES[rank], 1), 2.0).tolist()
