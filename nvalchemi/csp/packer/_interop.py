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

"""Small, private Torch/Warp interoperability helpers for the Packer."""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any

import torch
import warp as wp

wp.init()


def as_warp(value: torch.Tensor, dtype: Any) -> wp.array:
    """Return a zero-copy Warp view of a contiguous Torch tensor."""
    if not value.is_contiguous():
        raise ValueError("Warp inputs must be contiguous")
    return wp.from_torch(value, dtype=dtype)


def launch_device(value: torch.Tensor) -> str:
    """Return the Warp device name matching a Torch tensor."""
    return str(value.device)


def scoped_warp_stream(value: torch.Tensor):
    """Bind Warp launches to the active Torch CUDA stream for ``value``."""
    if value.device.type != "cuda":
        return nullcontext()
    stream = torch.cuda.current_stream(value.device)
    return wp.ScopedStream(wp.stream_from_torch(stream))
