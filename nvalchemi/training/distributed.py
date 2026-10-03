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
from torch.distributed.nn.functional import all_gather as _differentiable_all_gather

from nvalchemi.distributed import collective_device

if TYPE_CHECKING:
    from nvalchemi.distributed import DistributedManager

__all__ = [
    "all_gather_objects",
    "all_gather_rows",
    "all_reduce",
    "all_reduce_flags",
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


def all_reduce_flags(
    flag: bool | int, manager: DistributedManager | None = None
) -> torch.Tensor:
    """Return every rank's *flag* as one tensor, this rank's at its own index.

    Each rank raises its own entry of a zero vector and a ``MAX`` all-reduce
    collects the vector, so the result reads the same on every rank and names
    the ranks whose flag was set. The vector lives on the device
    :func:`~nvalchemi.distributed.collective_device` picks for the backend in
    use, so callers never choose a device for a collective themselves.

    Parameters
    ----------
    flag : bool | int
        This rank's verdict. A truthy value raises this rank's entry.
    manager : DistributedManager | None, optional
        Manager whose rank, world size, and ``all_reduce`` are used when
        given. Default ``None`` reads ``torch.distributed`` or the launcher's
        environment.

    Returns
    -------
    torch.Tensor
        Integer tensor of shape ``(world_size,)`` holding ``1`` where a rank
        raised its flag. A single process gets a length-one tensor holding its
        own flag, without a collective.

    Examples
    --------
    >>> from nvalchemi.training.distributed import all_reduce_flags
    >>> flags = all_reduce_flags(shard_is_empty, manager)  # doctest: +SKIP
    >>> if bool(flags.any()):  # doctest: +SKIP
    ...     raise ValueError(f"Ranks {flags.nonzero().flatten().tolist()} came up empty.")
    """
    world_size = get_world_size(manager)
    if world_size == 1:
        return torch.tensor([int(bool(flag))], dtype=torch.int64)
    flags = torch.zeros(world_size, dtype=torch.int64, device=collective_device())
    flags[get_rank(manager)] = int(bool(flag))
    return all_reduce(flags, manager, op=dist.ReduceOp.MAX)


def _require_process_group(world_size: int, name: str) -> None:
    """Raise when *name* would gather *world_size* ranks without a process group."""
    if not (dist.is_available() and dist.is_initialized()):
        raise RuntimeError(
            f"{name} gathers across {world_size!r} ranks, but no process group is "
            "initialized. Initialize one before the collective, or run on one "
            "process."
        )


def all_gather_rows(
    tensor: torch.Tensor,
    manager: DistributedManager | None = None,
    *,
    differentiable: bool = True,
) -> tuple[torch.Tensor, slice]:
    """Return every rank's *tensor* stacked along the leading dim, and this rank's rows.

    Shards may differ in their leading size. The sizes are gathered first, on
    the device :func:`~nvalchemi.distributed.collective_device` picks for the
    backend, then every shard is padded to the largest, gathered, and trimmed
    back, so the result holds each rank's rows in rank order and nothing
    else. With *differentiable*, the gather goes through
    :func:`torch.distributed.nn.functional.all_gather`, so a gradient reaching
    the gathered tensor flows back to each rank's own rows, summed over the
    ranks that used them. A single process gets its tensor back unchanged,
    with a slice over all of it, without a collective.

    Parameters
    ----------
    tensor : torch.Tensor
        This rank's rows, of shape ``(n, ...)``. The trailing shape and the
        dtype must agree across ranks.
    manager : DistributedManager | None, optional
        Manager whose rank and world size are read when given. Default
        ``None`` reads ``torch.distributed`` or the launcher's environment.
    differentiable : bool, optional
        Whether the gathered tensor carries an autograd graph back to
        *tensor*. Default ``True``.

    Returns
    -------
    tuple[torch.Tensor, slice]
        The world tensor, of shape ``(sum of every rank's n, ...)``, and the
        slice of its rows that came from this rank.

    Raises
    ------
    RuntimeError
        If more than one rank is reported but no process group is initialized.

    Examples
    --------
    >>> from nvalchemi.training.distributed import all_gather_rows
    >>> world, rows = all_gather_rows(energies)  # doctest: +SKIP
    >>> torch.equal(world[rows], energies)  # doctest: +SKIP
    True
    """
    world_size = get_world_size(manager)
    if world_size == 1:
        return tensor, slice(0, tensor.shape[0])
    _require_process_group(world_size, "all_gather_rows")
    count = torch.tensor(
        [tensor.shape[0]], dtype=torch.int64, device=collective_device()
    )
    counts = [torch.zeros_like(count) for _ in range(world_size)]
    dist.all_gather(counts, count)
    sizes = [int(size) for size in counts]
    padding = [0, 0] * (tensor.ndim - 1) + [0, max(sizes) - tensor.shape[0]]
    padded = torch.nn.functional.pad(tensor, padding)
    if differentiable:
        gathered = _differentiable_all_gather(padded)
    else:
        gathered = [torch.zeros_like(padded) for _ in range(world_size)]
        dist.all_gather(gathered, padded)
    world = torch.cat(
        [shard[:size] for shard, size in zip(gathered, sizes, strict=True)]
    )
    start = sum(sizes[: get_rank(manager)])
    return world, slice(start, start + tensor.shape[0])


def all_gather_objects(
    obj: Any, manager: DistributedManager | None = None
) -> list[Any]:
    """Return every rank's *obj*, in rank order.

    A single process gets ``[obj]`` without a collective. Under a process
    group the objects travel through :func:`torch.distributed.all_gather_object`,
    which pickles them, so the call suits small verdicts and summaries;
    :func:`all_gather_rows` carries tensors.

    Parameters
    ----------
    obj : Any
        This rank's picklable object.
    manager : DistributedManager | None, optional
        Manager whose world size is read when given. Default ``None`` reads
        ``torch.distributed`` or the launcher's environment.

    Returns
    -------
    list[Any]
        One entry per rank, this rank's at its own index.

    Raises
    ------
    RuntimeError
        If more than one rank is reported but no process group is initialized.
    """
    world_size = get_world_size(manager)
    if world_size == 1:
        return [obj]
    _require_process_group(world_size, "all_gather_objects")
    gathered: list[Any] = [None] * world_size
    dist.all_gather_object(gathered, obj)
    return gathered
