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

"""Correctness of the neighbor-only point-to-point halo exchange primitives.

The variable-size and fixed-shape neighbor exchanges replace a world-wide
``all_to_all`` with batched ``isend`` / ``irecv`` restricted to a rank's
neighbors. These tests build sparse, globally-consistent neighbor sets over a
gloo group and check that the neighbor primitives are byte-identical to the
collective path and do not deadlock on asymmetric (one-directionally empty) edges.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _gloo_harness import run_gloo  # noqa: E402


def _sizes_chain(world_size: int) -> list[list[int]]:
    """Symmetric count matrix: rank i exchanges only with i-1 / i+1 (no wrap),
    plus a self slice. Non-neighbors (|i-j|>1) are 0 so the neighbor primitive
    genuinely skips them."""
    s = [[0] * world_size for _ in range(world_size)]
    for i in range(world_size):
        s[i][i] = i + 1
        if i + 1 < world_size:
            n = (i + 1) * 3
            s[i][i + 1] = n
            s[i + 1][i] = n  # symmetric edge
    return s


def _check(rank: int, world_size: int, queue) -> None:  # type: ignore[no-untyped-def]
    import torch.distributed as dist

    from nvalchemi.distributed._core.gather_primitives import (
        _all_to_all_v_1d,
        _neighbor_p2p_v_1d,
    )

    sizes = _sizes_chain(world_size)
    send_counts = sizes[rank]
    recv_counts = [sizes[i][rank] for i in range(world_size)]

    # Deterministic payload: the slice rank r sends to j is filled with r*100+j.
    parts = [
        torch.full((send_counts[j],), float(rank * 100 + j)) for j in range(world_size)
    ]
    send = torch.cat(parts) if parts else torch.zeros(0)

    group = dist.group.WORLD
    ref = _all_to_all_v_1d(send, send_counts, recv_counts, group)
    got = _neighbor_p2p_v_1d(send, send_counts, recv_counts, group)

    exact = bool(torch.equal(ref, got))

    # Content check: the slice received from source i must be i*100+rank.
    r_off = [0]
    for c in recv_counts:
        r_off.append(r_off[-1] + c)
    content_ok = True
    for i in range(world_size):
        chunk = got[r_off[i] : r_off[i + 1]]
        if chunk.numel() and not torch.all(chunk == float(i * 100 + rank)):
            content_ok = False
    queue.put((rank, exact, content_ok, int(got.numel())))


def test_neighbor_p2p_matches_all_to_all_v_4ranks() -> None:
    results = run_gloo(world_size=4, fn=_check)
    assert len(results) == 4, results
    for rank, exact, content_ok, n in results:
        assert exact, f"rank {rank}: neighbor P2P != all_to_all_v"
        assert content_ok, f"rank {rank}: wrong received content"
        assert n > 0, f"rank {rank}: received nothing (n={n})"


# ----- fixed-shape halo layout (_neighbor_p2p_fixed) -----

_M = 3  # max_send (per-peer slot width)


def _chain_neighbors(rank: int, world_size: int) -> list[int]:
    return [i for i in (rank - 1, rank + 1) if 0 <= i < world_size]


def _check_fixed(rank: int, world_size: int, queue) -> None:  # type: ignore[no-untyped-def]
    import torch.distributed as dist

    from nvalchemi.distributed._core.gather_primitives import _neighbor_p2p_fixed

    # send_rows[j*M : (j+1)*M] = rank's slot destined for peer j, valued rank*10+j.
    send_rows = torch.cat(
        [torch.full((_M, 2), float(rank * 10 + j)) for j in range(world_size)]
    )
    got = _neighbor_p2p_fixed(
        send_rows, world_size, _chain_neighbors(rank, world_size), dist.group.WORLD
    )

    ok = True
    for i in range(world_size):
        chunk = got[i * _M : (i + 1) * _M]
        if i == rank:
            expect = float(rank * 10 + rank)  # self slot passes through
        elif abs(i - rank) == 1:
            expect = float(i * 10 + rank)  # neighbor sent its slot-for-rank
        else:
            expect = 0.0  # non-neighbor: untouched zeros
        if not torch.all(chunk == expect):
            ok = False
    queue.put((rank, ok))


def test_neighbor_p2p_fixed_layout_4ranks() -> None:
    results = run_gloo(world_size=4, fn=_check_fixed)
    assert len(results) == 4, results
    for rank, ok in results:
        assert ok, f"rank {rank}: fixed-layout neighbor exchange wrong"


def _check_fixed_asym(rank: int, world_size: int, queue) -> None:  # type: ignore[no-untyped-def]
    """A neighbor whose payload is empty in one direction must not hang.

    Every rank uses the symmetric chain neighbor set; rank 1 sends a real slot to
    rank 2 while rank 2 sends only zeros back (an asymmetric-empty edge). Because
    the neighbor set is symmetric, both ranks post the matching send and receive,
    so the exchange completes.
    """
    import torch.distributed as dist

    from nvalchemi.distributed._core.gather_primitives import _neighbor_p2p_fixed

    send_rows = torch.zeros(world_size * _M, 2)
    # Only rank 1 -> rank 2 carries a real payload; all other slots are zero.
    if rank == 1:
        send_rows[2 * _M : 3 * _M] = 7.0

    got = _neighbor_p2p_fixed(
        send_rows, world_size, _chain_neighbors(rank, world_size), dist.group.WORLD
    )
    # rank 2 must have received 7.0 in its slot-from-1; everyone else all-zero
    # in cross-rank slots. (No assertion needed beyond "didn't hang".)
    recv_from_1 = got[1 * _M : 2 * _M]
    ok = torch.all(recv_from_1 == 7.0).item() if rank == 2 else True
    queue.put((rank, bool(ok)))


def test_neighbor_p2p_fixed_asymmetric_empty_no_hang() -> None:
    # Would deadlock under a recv-count-derived active set; the symmetric
    # geometric neighbor set completes.
    results = run_gloo(world_size=4, fn=_check_fixed_asym, timeout_sec=30.0)
    assert len(results) == 4, f"deadlock/incomplete: {results}"
    for rank, ok in results:
        assert ok, f"rank {rank}: asymmetric-empty exchange wrong"
