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
"""Recommended distributed runtime manager for nvalchemi workflows."""

from __future__ import annotations

import os

import torch
from physicsnemo.distributed import (
    DistributedManager,
    PhysicsNeMoUninitializedDistributedManagerWarning,
)
from torch import distributed as dist

__all__ = [
    "DistributedManager",
    "PhysicsNeMoUninitializedDistributedManagerWarning",
    "collective_device",
    "resolve_global_rank",
    "resolve_world_size",
]


def resolve_world_size() -> int:
    """Resolve world size from PhysicsNeMo, torch.distributed, or environment."""
    if DistributedManager.is_initialized():
        return int(DistributedManager().world_size)
    if dist.is_available() and dist.is_initialized():
        return int(dist.get_world_size())
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    return world_size


def resolve_global_rank(global_rank: int | None = None) -> int:
    """Resolve global rank from an explicit value, distributed state, or env."""
    if global_rank is not None:
        return int(global_rank)
    if DistributedManager.is_initialized():
        return int(DistributedManager().rank)
    if dist.is_available() and dist.is_initialized():
        return int(dist.get_rank())
    rank = int(os.environ.get("RANK", 0))
    return rank


def collective_device(fallback: torch.device | str = "cpu") -> torch.device:
    """Resolve the rank-local device for distributed tensor collectives."""
    if dist.is_available() and dist.is_initialized():
        try:
            backend = dist.get_backend()
        except RuntimeError:
            backend = None
        if backend != "nccl":
            return torch.device("cpu")
    if DistributedManager.is_initialized():
        device = torch.device(DistributedManager().device)
    elif torch.cuda.is_available():
        index = int(os.environ.get("LOCAL_RANK", 0))
        device = torch.device("cuda", index)
    else:
        device = torch.device(fallback)
    if device.type == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return device
