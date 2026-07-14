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
"""Shared multi-GPU process-group harness for the distributed model tests.

The real-model DD tests each spawn ``world_size`` processes that share a NCCL
group (rank ``r`` pinned to physical ``cuda:r``). This module centralises the
otherwise-duplicated init/teardown so a test only supplies its per-rank worker
function. It lives beside :mod:`_helpers` in ``test/distributed`` so both the
top-level and ``model/`` test packages can ``from _dd_harness import ...``."""

from __future__ import annotations

import os
from typing import Any

import torch
import torch.distributed as dist

__all__ = ["init_nccl", "nccl_worker"]


def init_nccl(rank: int, world_size: int, port: str) -> None:
    """Initialise a NCCL process group for one spawned rank.

    Sets the rendezvous environment (``MASTER_ADDR``/``MASTER_PORT`` on
    localhost, ``RANK``/``WORLD_SIZE``/``LOCAL_RANK``), pins this rank to its
    physical GPU, and joins the group.

    Parameters
    ----------
    rank : int
        Global rank of this process, also its local ``cuda`` ordinal.
    world_size : int
        Number of processes in the group.
    port : str
        TCP port for the localhost rendezvous. Callers pass a per-test value
        so concurrently-running test files do not collide.
    """
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = port
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(rank)
    # Pin to this rank's physical GPU before NCCL init.
    torch.cuda.set_device(rank)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)


def nccl_worker(rank: int, world_size: int, port: str, fn: Any, *args: Any) -> None:
    """``mp.spawn`` entry point that runs *fn* inside a NCCL group.

    Initialises the group, invokes ``fn(rank, world_size, *args)``, and tears
    the group down even if the worker raises. Pass this as the ``mp.spawn``
    target with ``args=(world_size, port, fn, *fn_args)``.

    Parameters
    ----------
    rank : int
        Global rank supplied by ``mp.spawn``.
    world_size : int
        Number of processes in the group.
    port : str
        Rendezvous port (see :func:`init_nccl`).
    fn : callable
        Per-rank test body, called as ``fn(rank, world_size, *args)``.
    *args
        Extra positional arguments forwarded to *fn*.
    """
    init_nccl(rank, world_size, port)
    try:
        fn(rank, world_size, *args)
    finally:
        dist.destroy_process_group()
