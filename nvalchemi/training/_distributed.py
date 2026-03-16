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
"""Distributed training wrappers."""

from __future__ import annotations

from typing import Any, Literal

import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from nvalchemi._imports import OptionalDependencyFailure, check_optional_dependencies

try:
    from physicsnemo.distributed.utils import reduce_loss as _reduce_loss
except ImportError:
    _reduce_loss = None  # type: ignore[assignment]
    OptionalDependencyFailure("training")


def wrap_model(
    model: nn.Module,
    strategy: Literal["ddp", "fsdp"],
    dist_manager: Any | None = None,
) -> nn.Module:
    """Wrap a model for distributed training.

    Parameters
    ----------
    model : nn.Module
        The model to wrap.
    strategy : Literal["ddp", "fsdp"]
        Distributed strategy — ``"ddp"`` for
        :class:`~torch.nn.parallel.DistributedDataParallel` or ``"fsdp"``
        for :class:`~torch.distributed.fsdp.FullyShardedDataParallel`.
    dist_manager : Any or None
        A ``physicsnemo.distributed.DistributedManager`` instance, or
        ``None`` for single-process training.

    Returns
    -------
    nn.Module
        The (possibly wrapped) model.  Returns the original model unchanged
        when *dist_manager* is ``None`` or world size is 1.
    """
    if dist_manager is None:
        return model

    world_size: int = getattr(dist_manager, "world_size", 1)
    if world_size <= 1:
        return model

    local_rank: int = getattr(dist_manager, "local_rank", 0)

    if strategy == "ddp":
        broadcast_buffers: bool = getattr(dist_manager, "broadcast_buffers", True)
        find_unused: bool = getattr(dist_manager, "find_unused_parameters", False)
        return DistributedDataParallel(
            model,
            device_ids=[local_rank],
            broadcast_buffers=broadcast_buffers,
            find_unused_parameters=find_unused,
        )

    if strategy == "fsdp":
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

        return FSDP(model)

    msg = f"Unknown distributed strategy: {strategy!r}"
    raise ValueError(msg)


@check_optional_dependencies()
def reduce_loss_across_ranks(
    loss: torch.Tensor,
    dist_manager: Any | None = None,
) -> torch.Tensor:
    """Reduce a loss tensor across distributed ranks.

    Parameters
    ----------
    loss : torch.Tensor
        Scalar loss from the local rank.
    dist_manager : Any or None
        A ``physicsnemo.distributed.DistributedManager`` instance, or
        ``None`` for single-process training.

    Returns
    -------
    torch.Tensor
        The averaged loss across all ranks, or the unmodified *loss* when
        not running in a distributed context.
    """
    if dist_manager is None:
        return loss

    distributed: bool = getattr(dist_manager, "distributed", False)
    if not distributed:
        return loss

    reduced = _reduce_loss(loss.item(), dst_rank=0, mean=True)
    if reduced is None:
        return loss
    return torch.tensor(reduced, dtype=loss.dtype, device=loss.device)
