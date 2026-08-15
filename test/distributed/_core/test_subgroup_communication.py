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

"""Communication primitives over domain subgroups of a two-dimensional mesh."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _gloo_harness import run_gloo  # noqa: E402


def _domain_group(rank: int):  # type: ignore[no-untyped-def]
    """Return this rank's domain row in a (pipeline=2, domain=2) layout."""
    import torch.distributed as dist

    first_stage = dist.new_group([0, 1])
    second_stage = dist.new_group([2, 3])
    return first_stage if rank < 2 else second_stage


def _expected_received(rank: int) -> list[float]:
    """Expected source-ordered values within the rank's two-member domain row."""
    stage_first_rank = (rank // 2) * 2
    domain_rank = rank % 2
    return [
        float(source * 10 + domain_rank)
        for source in (stage_first_rank, stage_first_rank + 1)
    ]


def _check_p2p_helper(rank: int, world_size: int, queue, helper_name: str) -> None:  # type: ignore[no-untyped-def]
    import torch.distributed as dist

    from nvalchemi.distributed._core.gather_primitives import (
        _isend_irecv_v_1d,
        _neighbor_p2p_fixed,
        _neighbor_p2p_v_1d,
    )

    assert world_size == 4
    group = _domain_group(rank)
    domain_world_size = dist.get_world_size(group)
    domain_rank = dist.get_rank(group)

    # Slot d contains the value this global rank is sending to domain-local rank d.
    send = torch.tensor(
        [float(rank * 10 + destination) for destination in range(domain_world_size)]
    )

    received: list[float] | None = None
    error: str | None = None
    try:
        if helper_name == "neighbor_variable":
            result = _neighbor_p2p_v_1d(
                send,
                [1] * domain_world_size,
                [1] * domain_world_size,
                group,
            )
        elif helper_name == "neighbor_fixed":
            result = _neighbor_p2p_fixed(
                send,
                domain_world_size,
                list(range(domain_world_size)),
                group,
            )
        else:
            result = torch.empty_like(send)
            _isend_irecv_v_1d(
                send,
                [1] * domain_world_size,
                result,
                [1] * domain_world_size,
                group,
            )
        received = result.tolist()
    except Exception as exc:  # noqa: BLE001
        error = f"{type(exc).__name__}: {exc}"

    # Keep the first domain row alive until the second row has also completed
    # (or reported its rank-translation error).
    dist.barrier()
    queue.put(
        (
            rank,
            domain_rank,
            received,
            _expected_received(rank),
            error,
        )
    )


@pytest.mark.parametrize(
    "helper_name",
    ["neighbor_variable", "neighbor_fixed", "gloo_fallback"],
)
def test_p2p_helpers_use_global_peers_in_domain_subgroups(helper_name: str) -> None:
    """A domain-local peer index must be translated before calling P2P APIs."""
    results = run_gloo(
        world_size=4,
        fn=_check_p2p_helper,
        args=(helper_name,),
        timeout_sec=30.0,
    )
    assert len(results) == 4, results
    for rank, domain_rank, received, expected, error in results:
        assert error is None, (
            f"global rank {rank} (domain rank {domain_rank}) failed: {error}"
        )
        assert received == expected, (
            f"global rank {rank} (domain rank {domain_rank}) "
            f"received {received}, expected {expected}"
        )


def _check_fixed_halo_group(rank: int, world_size: int, queue) -> None:  # type: ignore[no-untyped-def]
    import torch.distributed as dist

    from nvalchemi.distributed._core.gather_primitives import (
        halo_exchange_fixed,
        mesh_group,
    )

    assert world_size == 4
    group = _domain_group(rank)
    assert mesh_group(group) is group
    domain_world_size = dist.get_world_size(group)
    domain_rank = dist.get_rank(group)
    send_rows = torch.tensor(
        [float(rank * 10 + destination) for destination in range(domain_world_size)]
    )

    received: list[float] | None = None
    error: str | None = None
    try:
        result = halo_exchange_fixed(send_rows, domain_world_size, group)
        received = result.tolist()
    except Exception as exc:  # noqa: BLE001
        error = f"{type(exc).__name__}: {exc}"

    dist.barrier()
    queue.put(
        (
            rank,
            domain_rank,
            received,
            _expected_received(rank),
            error,
        )
    )


def test_halo_exchange_fixed_uses_domain_subgroup() -> None:
    """The fixed halo dispatcher must not replace its subgroup with WORLD."""
    results = run_gloo(
        world_size=4,
        fn=_check_fixed_halo_group,
        timeout_sec=30.0,
    )
    assert len(results) == 4, results
    for rank, domain_rank, received, expected, error in results:
        assert error is None, (
            f"global rank {rank} (domain rank {domain_rank}) failed: {error}"
        )
        assert received == expected, (
            f"global rank {rank} (domain rank {domain_rank}) "
            f"received {received}, expected {expected}"
        )


def _check_static_halo_op_group(rank: int, world_size: int, queue) -> None:  # type: ignore[no-untyped-def]
    import torch.distributed as dist

    from nvalchemi.distributed._core.particle_halo import (
        halo_forward_static_op,
        set_halo_process_group,
    )

    assert world_size == 4
    group = _domain_group(rank)
    domain_rank = dist.get_rank(group)
    set_halo_process_group(group)

    # One owned row, one real ghost row from the peer, and one dead padding row.
    # Each peer block has a fixed capacity of one row. The self-source block is
    # masked as padding; the other source block lands in the ghost row.
    padded = torch.tensor([float(rank), -1.0, -1.0], requires_grad=True)
    send_index = torch.tensor([0, 0], dtype=torch.int64)
    if domain_rank == 0:
        recv_dest = torch.tensor([2, 1], dtype=torch.int64)
        recv_real = torch.tensor([False, True])
    else:
        recv_dest = torch.tensor([1, 2], dtype=torch.int64)
        recv_real = torch.tensor([True, False])
    n_owned = torch.tensor(1, dtype=torch.int64)

    out = halo_forward_static_op(
        padded,
        send_index,
        recv_dest,
        recv_real,
        n_owned,
        dist.get_world_size(group),
    )
    out.sum().backward()

    peer = rank + 1 if domain_rank == 0 else rank - 1
    queue.put(
        (rank, out.tolist(), padded.grad.tolist(), [float(rank), float(peer), 0.0])
    )


def test_static_halo_op_uses_published_domain_subgroup() -> None:
    """The real compiled-path custom op must stay inside its pipeline stage."""
    results = run_gloo(
        world_size=4,
        fn=_check_static_halo_op_group,
        timeout_sec=30.0,
    )
    assert len(results) == 4, results
    for rank, received, gradient, expected in results:
        assert received == expected, (
            f"global rank {rank} received {received}, expected {expected}"
        )
        assert gradient == [2.0, 0.0, 0.0], (
            f"global rank {rank} produced input gradient {gradient}"
        )
