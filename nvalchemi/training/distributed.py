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
"""Structural helpers for distributed training managers.

This module intentionally does not define a concrete manager class. Phase-2
training can accept a manager supplied by another package while retaining a
``torch.distributed`` fallback for local tests and ``torchrun`` launches.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

import torch
from torch import distributed as dist

if TYPE_CHECKING:
    from nvalchemi.distributed import DistributedManager

__all__ = [
    "all_reduce",
    "barrier",
    "destroy_distributed",
    "distributed_device",
    "get_local_rank",
    "get_rank",
    "get_world_size",
    "init_distributed",
    "is_distributed_initialized",
]


def _read_attr_or_call(manager: Any, *names: str) -> Any:
    """Return the first manager attribute or zero-arg method result found."""
    for name in names:
        if not hasattr(manager, name):
            continue
        value = getattr(manager, name)
        if callable(value):
            try:
                return value()
            except TypeError:
                continue
        return value
    return None


def _call_manager(manager: Any, *names: str, **kwargs: Any) -> bool:
    """Call the first matching manager method and report whether one ran."""
    for name in names:
        method = getattr(manager, name, None)
        if not callable(method):
            continue
        try:
            method(**kwargs)
        except TypeError:
            method()
        return True
    return False


def _env_int(name: str, default: int) -> int:
    """Read an integer torchrun environment variable."""
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def is_distributed_initialized(manager: DistributedManager | None = None) -> bool:
    """Return whether distributed communication is initialized."""
    if manager is not None:
        value = _read_attr_or_call(
            manager,
            "is_initialized",
            "initialized",
            "is_distributed_initialized",
        )
        if value is not None:
            return bool(value)
    return dist.is_available() and dist.is_initialized()


def get_rank(manager: DistributedManager | None = None) -> int:
    """Return the global process rank."""
    if manager is not None:
        value = _read_attr_or_call(manager, "global_rank", "rank", "get_rank")
        if value is not None:
            return int(value)
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return _env_int("RANK", 0)


def get_world_size(manager: DistributedManager | None = None) -> int:
    """Return the distributed world size."""
    if manager is not None:
        value = _read_attr_or_call(manager, "world_size", "get_world_size")
        if value is not None:
            return int(value)
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    return _env_int("WORLD_SIZE", 1)


def get_local_rank(manager: DistributedManager | None = None) -> int:
    """Return the process-local rank."""
    if manager is not None:
        value = _read_attr_or_call(manager, "local_rank", "get_local_rank")
        if value is not None:
            return int(value)
    if dist.is_available() and dist.is_initialized():
        try:
            return int(dist.get_node_local_rank())
        except (AttributeError, RuntimeError):
            pass
    return _env_int("LOCAL_RANK", 0)


def distributed_device(
    manager: DistributedManager | None,
    fallback: torch.device | str,
    *,
    prefer_cuda: bool = True,
) -> torch.device:
    """Resolve the device for the current rank."""
    if manager is not None:
        value = _read_attr_or_call(manager, "device", "get_device")
        if value is not None:
            return torch.device(value)
    fallback_device = torch.device(fallback)
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda", get_local_rank(manager))
    return fallback_device


def init_distributed(
    manager: DistributedManager | None = None,
    *,
    backend: str | None = None,
    **kwargs: Any,
) -> bool:
    """Initialize distributed communication and return whether this call did so."""
    if is_distributed_initialized(manager):
        return False
    if manager is not None:
        return _call_manager(
            manager,
            "init_process_group",
            "initialize",
            "init",
            "setup",
            backend=backend,
            **kwargs,
        )
    if get_world_size(None) <= 1:
        return False
    resolved_backend = backend or ("nccl" if torch.cuda.is_available() else "gloo")
    dist.init_process_group(backend=resolved_backend, **kwargs)
    return True


def destroy_distributed(manager: DistributedManager | None = None) -> bool:
    """Destroy distributed communication if possible."""
    if manager is not None:
        return _call_manager(
            manager,
            "destroy_process_group",
            "destroy",
            "cleanup",
            "teardown",
        )
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()
        return True
    return False


def barrier(manager: DistributedManager | None = None) -> None:
    """Synchronize all ranks when distributed communication is initialized."""
    if manager is not None and _call_manager(manager, "barrier"):
        return
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def all_reduce(
    tensor: torch.Tensor,
    manager: DistributedManager | None = None,
    *,
    op: dist.ReduceOp = dist.ReduceOp.SUM,
) -> torch.Tensor:
    """All-reduce ``tensor`` in place and return it."""
    if manager is not None:
        method = getattr(manager, "all_reduce", None)
        if callable(method):
            result = method(tensor, op=op)
            return tensor if result is None else result
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(tensor, op=op)
    return tensor
