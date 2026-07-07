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
"""The degenerate-partition verdict (``_partition_health_verdict``).

The collective that classifies a halo partition as broken (empty shard),
trivial (every rank sees all atoms -> no parallelism), or healthy. 2-rank gloo
on CPU — no GPU needed; this gates the collective LOGIC (shared verdict on every
rank). The end-to-end behavior (DistributedModel raises on empty / warns on
trivial) rides the model multigpu gates.
"""

from __future__ import annotations

import os

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def _verdict_worker(rank: int, world_size: int, scenario: str) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29683"
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
    try:
        from nvalchemi.distributed.distributed_model import (
            _partition_health_verdict,
        )

        group = dist.group.WORLD
        device = torch.device("cpu")
        if scenario == "empty":
            # rank 1 gets 0 owned atoms -> empty shard.
            n_owned, n_padded = (10, 10) if rank == 0 else (0, 0)
            any_empty, any_trivial, n_global = _partition_health_verdict(
                n_owned, n_padded, group, device
            )
            assert any_empty, "empty shard not detected"
            assert n_global == 10
        elif scenario == "trivial":
            # each owns 5 of 10, but the halo padded view covers all 10 -> 0
            # remote atoms on every rank (degenerate-but-correct).
            any_empty, any_trivial, n_global = _partition_health_verdict(
                5, 10, group, device
            )
            assert not any_empty
            assert any_trivial, "trivial (0-remote) partition not detected"
            assert n_global == 10
        else:  # healthy: 50 owned + 10 ghost of 100 total -> 40 remote.
            any_empty, any_trivial, n_global = _partition_health_verdict(
                50, 60, group, device
            )
            assert not any_empty
            assert not any_trivial, "healthy partition misflagged as trivial"
            assert n_global == 100
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("scenario", ["empty", "trivial", "healthy"])
def test_partition_health_verdict_2ranks(scenario: str) -> None:
    """The verdict is shared identically across ranks: empty shard -> any_empty,
    full-coverage -> any_trivial, real split -> neither."""
    mp.spawn(_verdict_worker, args=(2, scenario), nprocs=2)


# ---------------------------------------------------------------------------
# Acting on the verdict (_resolve_partition_health) — pure, no collectives.
# ---------------------------------------------------------------------------


def test_resolve_partition_health_empty_always_raises() -> None:
    """An empty shard is fatal regardless of require_nondegenerate."""
    from nvalchemi.distributed.distributed_model import _resolve_partition_health

    for strict in (False, True):
        with pytest.raises(RuntimeError, match="0 owned"):
            _resolve_partition_health(
                any_empty=True,
                any_trivial=False,
                n_global=10,
                world_size=2,
                require_nondegenerate=strict,
                rank=0,
            )


def test_resolve_partition_health_trivial_strict_raises() -> None:
    """A trivial (0-remote) partition raises when require_nondegenerate=True."""
    from nvalchemi.distributed.distributed_model import _resolve_partition_health

    with pytest.raises(RuntimeError, match="require_nondegenerate=True"):
        _resolve_partition_health(
            any_empty=False,
            any_trivial=True,
            n_global=10,
            world_size=2,
            require_nondegenerate=True,
            rank=0,
        )


def test_resolve_partition_health_trivial_lenient_warns_not_raises() -> None:
    """Without the flag a trivial partition only warns (still correct)."""
    from nvalchemi.distributed.distributed_model import _resolve_partition_health

    # Must not raise; a healthy verdict must be a no-op too.
    _resolve_partition_health(
        any_empty=False,
        any_trivial=True,
        n_global=10,
        world_size=2,
        require_nondegenerate=False,
        rank=0,
    )
    _resolve_partition_health(
        any_empty=False,
        any_trivial=False,
        n_global=100,
        world_size=2,
        require_nondegenerate=True,
        rank=0,
    )
