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

"""Reusable Gloo-spawn helpers for distributed-dispatch tests.

Centralises the boilerplate of:

  - ``mp.spawn`` with a generic worker that calls a user-supplied test
    function under an initialised gloo process group on CPU.
  - A monkey-patch of
    ``physicsnemo.distributed.utils.indexed_all_to_all_v_wrapper`` so the
    halo-exchange primitives have a working ``all_to_all_v`` over gloo
    (NCCL has it natively but gloo doesn't).
  - A queue-backed mechanism for collecting per-rank results back to
    the launcher process. Useful for asserts of the form "rank 0 saw
    shape X, rank 1 saw shape Y".

Patterns
--------
A test that just needs all_reduce (per_system_reduce) does::

    def _check(rank, world_size, queue):
        from nvalchemi.distributed._core.dispatch_trace import dispatch_trace
        with dispatch_trace() as records:
            ...
        queue.put((rank, records))

    def test_thing():
        records_per_rank = run_gloo(world_size=2, fn=_check)
        # records_per_rank is a list of (rank, payload) tuples.

For tests that exercise the halo-exchange / sharded-gather paths,
``run_gloo`` already installs the gloo-friendly ``indexed_all_to_all_v``
shim in each worker — no per-test setup needed.
"""

from __future__ import annotations

import os
from typing import Any, Callable

import torch.distributed as dist
import torch.multiprocessing as mp

__all__ = ["run_gloo"]


def _patch_all_to_all_for_gloo() -> None:
    """Replace ``physicsnemo.distributed.utils.indexed_all_to_all_v_wrapper``
    with an ``isend/irecv`` emulation. Gloo lacks ``all_to_all_v``;
    NCCL has it natively. The replacement preserves the same function
    contract so the halo-exchange path is exercised under gloo
    end-to-end.
    """
    import physicsnemo.distributed.utils as pn_utils  # noqa: PLC0415
    import torch  # noqa: PLC0415

    def _indexed_all_to_all_v_gloo(tensor, indices, sizes, dim=0, group=None):
        comm_size = dist.get_world_size(group=group)
        rank = dist.get_rank(group=group)
        x_send = [tensor[idx].contiguous() for idx in indices]
        x_recv = []
        tensor_shape = list(tensor.shape)
        for r in range(comm_size):
            tensor_shape[dim] = sizes[r][rank]
            x_recv.append(
                torch.empty(tensor_shape, dtype=tensor.dtype, device=tensor.device)
            )
        ops = []
        for r in range(comm_size):
            if r == rank:
                x_recv[r].copy_(x_send[r])
            else:
                ops.append(dist.isend(x_send[r], dst=r, group=group))
                ops.append(dist.irecv(x_recv[r], src=r, group=group))
        for op in ops:
            op.wait()
        return torch.cat(x_recv, dim=dim)

    pn_utils.indexed_all_to_all_v_wrapper = _indexed_all_to_all_v_gloo


def _worker(
    rank: int,
    world_size: int,
    fn: Callable[..., None],
    queue: Any,
    args: tuple,
) -> None:
    """Worker entry point called by ``mp.spawn``. Initialises the gloo
    process group, applies the all_to_all patch, and invokes the user
    function with ``(rank, world_size, queue, *args)``.
    """
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    # Distinct port per spawn to avoid clashes when pytest runs the same
    # harness back-to-back.
    os.environ.setdefault("MASTER_PORT", "29503")
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)

    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
    try:
        _patch_all_to_all_for_gloo()
        fn(rank, world_size, queue, *args)
    finally:
        dist.destroy_process_group()


def run_gloo(
    *,
    world_size: int,
    fn: Callable[..., None],
    args: tuple = (),
    timeout_sec: float = 60.0,
) -> list[Any]:
    """Spawn ``world_size`` Gloo workers, run ``fn`` on each, collect
    queue payloads, return them in arrival order.

    The user function signature is
    ``fn(rank, world_size, queue, *args) -> None``. Whatever the
    function ``queue.put``s before exiting becomes part of the
    returned list. Tests typically put ``(rank, payload)`` tuples
    so they can sort by rank afterwards.

    ``timeout_sec`` is the per-spawn join timeout; on hangs, processes
    are terminated and the test fails.
    """
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    procs = []
    for rank in range(world_size):
        p = ctx.Process(target=_worker, args=(rank, world_size, fn, queue, args))
        p.start()
        procs.append(p)

    results: list[Any] = []
    deadline_per_proc = timeout_sec
    try:
        for p in procs:
            p.join(timeout=deadline_per_proc)
            if p.is_alive():
                p.terminate()
                raise TimeoutError(
                    f"gloo worker pid={p.pid} did not finish within "
                    f"{deadline_per_proc:.1f}s"
                )
            if p.exitcode not in (0, None):
                raise RuntimeError(
                    f"gloo worker pid={p.pid} exited with code {p.exitcode}"
                )
        # Drain the queue. Each worker may have put 0..N items; we
        # collect everything available within a small grace period.
        while not queue.empty():
            results.append(queue.get_nowait())
    finally:
        for p in procs:
            if p.is_alive():
                p.terminate()

    return results
