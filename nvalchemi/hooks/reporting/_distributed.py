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
"""Distributed reporting helpers."""

from __future__ import annotations

from dataclasses import replace

import torch
from torch import distributed as dist

from nvalchemi.hooks.reporting._scalars import ScalarSnapshot
from nvalchemi.training.distributed import (
    all_reduce,
    get_world_size,
    is_distributed_initialized,
)


def _collective_device() -> torch.device:
    """Return the tensor device used for reporting collectives."""
    from physicsnemo.distributed import DistributedManager

    if not DistributedManager.is_initialized():
        raise RuntimeError(
            "Reporting rank reductions require DistributedManager to be initialized."
        )
    return torch.device(DistributedManager().device)


_STRING_REDUCTIONS = {
    "sum": (dist.ReduceOp.SUM, False),
    "min": (dist.ReduceOp.MIN, False),
    "max": (dist.ReduceOp.MAX, False),
    "mean": (dist.ReduceOp.SUM, True),
    "avg": (dist.ReduceOp.SUM, True),
    "average": (dist.ReduceOp.SUM, True),
}


def reduce_scalar_snapshot(
    snapshot: ScalarSnapshot,
    reduction: dist.ReduceOp | str | None,
    *,
    reporter_name: str,
) -> ScalarSnapshot:
    """Reduce snapshot scalar values across distributed ranks.

    Parameters
    ----------
    snapshot : ScalarSnapshot
        Local scalar snapshot.
    reduction : torch.distributed.ReduceOp | str | None
        Reduction operation to apply. ``None`` and ``"none"`` disable
        reduction. ``"mean"``, ``"avg"``, and ``"average"`` use
        :data:`torch.distributed.ReduceOp.SUM` followed by explicit world-size
        division.
    reporter_name : str
        Reporter name used in validation error messages.

    Returns
    -------
        ScalarSnapshot
        Snapshot with reduced scalar values. The original snapshot is returned
        unchanged outside initialized distributed runs or when ``reduction`` is
        ``None``.

    Raises
    ------
    ValueError
        If ranks report different scalar keys.
    """
    op, average = normalize_rank_reduction(reduction)
    if op is None:
        return snapshot
    if not is_distributed_initialized(None):
        return snapshot
    world_size = get_world_size(None)
    keys = tuple(sorted(snapshot.scalars))
    gathered_keys: list[tuple[str, ...]] = [() for _ in range(world_size)]
    dist.all_gather_object(gathered_keys, keys)
    if any(rank_keys != keys for rank_keys in gathered_keys):
        raise ValueError(
            f"{reporter_name} rank reduction requires every rank to report "
            "the same scalar keys."
        )
    if not keys:
        return replace(snapshot, scalars={})
    values = torch.tensor(
        [snapshot.scalars[key] for key in keys],
        device=_collective_device(),
        dtype=torch.float64,
    )
    all_reduce(values, None, op=op)
    if average:
        values /= world_size
    reduced_values = values.cpu().tolist()
    reduced_scalars = {
        key: float(value) for key, value in zip(keys, reduced_values, strict=True)
    }
    return replace(snapshot, scalars=reduced_scalars)


def normalize_rank_reduction(
    reduction: dist.ReduceOp | str | None,
) -> tuple[dist.ReduceOp | None, bool]:
    """Normalize user-facing rank reduction input to a PyTorch reduction op.

    Parameters
    ----------
    reduction : torch.distributed.ReduceOp | str | None
        Reduction configuration supplied by a reporter.

    Returns
    -------
    tuple[torch.distributed.ReduceOp | None, bool]
        Normalized PyTorch reduction op plus whether to divide by world size
        after the collective.

    Raises
    ------
    ValueError
        If a string reduction is not recognized.
    TypeError
        If ``reduction`` is not ``None``, a string, or a PyTorch
        :class:`torch.distributed.ReduceOp`.
    """
    if reduction is None:
        return None, False
    if isinstance(reduction, str):
        key = reduction.lower()
        if key == "none":
            return None, False
        try:
            return _STRING_REDUCTIONS[key]
        except KeyError as exc:
            raise ValueError(
                "rank_reduction must be None, a torch.distributed.ReduceOp, "
                "or one of 'none', 'mean', 'avg', 'average', 'sum', 'min', "
                "or 'max'."
            ) from exc
    if not isinstance(reduction, dist.ReduceOp):
        raise TypeError(
            "rank_reduction must be None, a string, or torch.distributed.ReduceOp."
        )
    return reduction, False
