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

from physicsnemo.distributed import (
    DistributedManager,
    PhysicsNeMoUninitializedDistributedManagerWarning,
)
from torch import distributed as dist

__all__ = [
    "DistributedManager",
    "PhysicsNeMoUninitializedDistributedManagerWarning",
    "resolve_global_rank",
    "resolve_world_size",
]


def resolve_world_size() -> int:
    """Resolve world size from PhysicsNeMo, torch.distributed, or environment."""
    if DistributedManager.is_initialized():
        return int(DistributedManager().world_size)
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    return int(os.getenv("WORLD_SIZE", 1))


def resolve_global_rank(global_rank: int | None = None) -> int:
    """Resolve global rank from an explicit value, distributed state, or env."""
    if global_rank is not None:
        return int(global_rank)
    if DistributedManager.is_initialized():
        return int(DistributedManager().rank)
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return int(os.getenv("RANK", 0))
