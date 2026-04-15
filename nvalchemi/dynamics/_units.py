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
"""Conversions between public MD time units and kernel time units."""

from __future__ import annotations

import torch

FS_PER_INTERNAL_TIME: float = 10.180505710759414


def fs_to_internal_time(value: float | torch.Tensor) -> float | torch.Tensor:
    """Convert a physical time in femtoseconds to the kernel time unit."""
    return value / FS_PER_INTERNAL_TIME


def per_fs_to_internal_rate(value: float | torch.Tensor) -> float | torch.Tensor:
    """Convert a physical rate in 1/fs to the kernel rate unit."""
    return value * FS_PER_INTERNAL_TIME
