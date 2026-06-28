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
"""Distributed (multi-rank, gloo/CPU) tests for the standalone ValidationLoop."""

from __future__ import annotations

import math
import os
import socket
from typing import Any

import pytest
import torch
from torch import distributed as dist

from nvalchemi.training import (
    EnergyMSELoss,
    ForceMSELoss,
    ValidationConfig,
    ValidationLoop,
)
from test.training.conftest import _build_dataset, _build_demo_model
from test.training.test_strategy import demo_training_fn


def _free_port() -> int:
    """Return an available localhost TCP port for process-group setup."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _composed_loss() -> Any:
    """Return a composed energy + force MSE loss."""
    return EnergyMSELoss() + ForceMSELoss(normalize_by_atom_count=True)


def _run_validation_worker(
    rank: int,
    world_size: int,
    port: int,
    result_queue: Any,
) -> None:
    """Run a standalone ValidationLoop on one gloo/CPU rank and report its summary.

    With ``distributed_manager=None`` the ValidationLoop falls back to the
    raw ``torch.distributed`` primitives, so the all-reduce and ``__exit__``
    barrier use the initialized process group.

    Parameters
    ----------
    rank : int
        Global rank of this worker.
    world_size : int
        Total number of ranks.
    port : int
        TCP port for the gloo rendezvous.
    result_queue : Any
        Multiprocessing queue used to send ``(rank, summary)`` to the parent.
    """
    os.environ.update(
        {
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": str(port),
            "RANK": str(rank),
            "WORLD_SIZE": str(world_size),
            "LOCAL_RANK": str(rank),
        }
    )
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
    try:
        # Each rank validates a deterministic, re-iterable shard so the
        # all-reduced mean is well-defined and finite.
        data = _build_dataset(n_batches=2, base_seed=100 + rank)
        config = ValidationConfig(validation_data=data, loss_fn=_composed_loss())
        loop = ValidationLoop(
            validation_data=data,
            config=config,
            device=torch.device("cpu"),
            model=_build_demo_model(),
            validation_fn=demo_training_fn,
            grad_enabled=True,
        )
        with loop as active_loop:
            summary = active_loop.execute()
        result_queue.put(
            (
                rank,
                {
                    "total_loss": float(summary["total_loss"]),
                    "num_batches": summary["num_batches"],
                    "distributed_reduced": summary["distributed_reduced"],
                },
            )
        )
    finally:
        dist.barrier()
        dist.destroy_process_group()


@pytest.mark.slow
@pytest.mark.skipif(not dist.is_gloo_available(), reason="gloo backend required")
def test_distributed_validation_summary_available_on_all_ranks() -> None:
    """Every rank receives the same finite reduced validation summary."""
    world_size = 2
    ctx = torch.multiprocessing.get_context("spawn")
    result_queue = ctx.Queue()
    port = _free_port()
    procs = [
        ctx.Process(
            target=_run_validation_worker,
            args=(rank, world_size, port, result_queue),
        )
        for rank in range(world_size)
    ]
    for proc in procs:
        proc.start()
    results: dict[int, Any] = {}
    for _ in range(world_size):
        rank, summary = result_queue.get(timeout=60)
        results[rank] = summary
    for proc in procs:
        proc.join(timeout=60)
    for proc in procs:
        # A clean exit on every rank proves the __exit__ barrier did not deadlock.
        assert proc.exitcode == 0

    assert set(results) == set(range(world_size))
    expected = results[0]
    assert expected is not None
    assert math.isfinite(expected["total_loss"])
    assert expected["num_batches"] == 4
    assert expected["distributed_reduced"] is True
    for rank in range(1, world_size):
        summary = results[rank]
        assert summary is not None
        assert summary["total_loss"] == pytest.approx(expected["total_loss"])
        assert summary["num_batches"] == expected["num_batches"]
        assert summary["distributed_reduced"] is True
