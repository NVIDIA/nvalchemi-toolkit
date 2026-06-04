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
from enum import Enum

import torch
from torch import distributed as dist

from nvalchemi.hooks.reporting._scalars import ScalarSnapshot


class RankReduction(str, Enum):
    """Distributed scalar reduction mode.

    Attributes
    ----------
    NONE : RankReduction
        Do not reduce across ranks.
    MEAN : RankReduction
        Average each scalar across ranks.
    SUM : RankReduction
        Sum each scalar across ranks.
    MIN : RankReduction
        Take the minimum scalar value across ranks.
    MAX : RankReduction
        Take the maximum scalar value across ranks.
    """

    NONE = "none"
    MEAN = "mean"
    SUM = "sum"
    MIN = "min"
    MAX = "max"


def reduce_scalar_snapshot(
    snapshot: ScalarSnapshot,
    reduction: RankReduction,
    *,
    reporter_name: str,
) -> ScalarSnapshot:
    """Reduce snapshot scalar values across distributed ranks.

    Parameters
    ----------
    snapshot : ScalarSnapshot
        Local scalar snapshot.
    reduction : RankReduction
        Reduction operation to apply.
    reporter_name : str
        Reporter name used in validation error messages.

    Returns
    -------
    ScalarSnapshot
        Snapshot with reduced scalar values. The original snapshot is returned
        unchanged outside initialized distributed runs or when ``reduction`` is
        ``RankReduction.NONE``.

    Raises
    ------
    RuntimeError
        If NCCL reduction is requested without an available CUDA device.
    ValueError
        If ranks report different scalar keys.
    """
    if reduction == RankReduction.NONE:
        return snapshot
    if not dist.is_available() or not dist.is_initialized():
        return snapshot
    keys = tuple(sorted(snapshot.scalars))
    gathered_keys: list[tuple[str, ...]] = [() for _ in range(dist.get_world_size())]
    dist.all_gather_object(gathered_keys, keys)
    if any(rank_keys != keys for rank_keys in gathered_keys):
        raise ValueError(
            f"{reporter_name} rank reduction requires every rank to report "
            "the same scalar keys."
        )
    device = _collective_device()
    reduced_scalars: dict[str, float] = {}
    for key in keys:
        value = torch.tensor(snapshot.scalars[key], device=device)
        dist.all_reduce(value, op=_reduce_op(reduction))
        if reduction == RankReduction.MEAN:
            value /= dist.get_world_size()
        reduced_scalars[key] = float(value.cpu().item())
    return replace(snapshot, scalars=reduced_scalars)


def _collective_device() -> torch.device:
    if dist.get_backend() == "nccl":
        if not torch.cuda.is_available():
            raise RuntimeError("NCCL rank reduction requires an available CUDA device.")
        return torch.device("cuda", torch.cuda.current_device())
    return torch.device("cpu")


def _reduce_op(reduction: RankReduction) -> dist.ReduceOp:
    if reduction in (RankReduction.MEAN, RankReduction.SUM):
        return dist.ReduceOp.SUM
    if reduction == RankReduction.MIN:
        return dist.ReduceOp.MIN
    if reduction == RankReduction.MAX:
        return dist.ReduceOp.MAX
    raise ValueError(f"Unsupported rank reduction: {reduction.value!r}.")
