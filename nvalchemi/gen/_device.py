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
"""Shared device parsing, CUDA pinning, and availability validation."""

from __future__ import annotations

import torch


def normalize_device(value: torch.device | str) -> torch.device:
    """Parse a device and pin CUDA to a valid, concrete device index.

    Parameters
    ----------
    value
        A device string or :class:`torch.device`.

    Returns
    -------
    torch.device
        The parsed device, with unindexed CUDA resolved to the current GPU.

    Raises
    ------
    ValueError
        If the value cannot be parsed or the requested CUDA device is
        unavailable.
    """
    if not isinstance(value, (torch.device, str)):
        raise ValueError(
            f"device must be a string or torch.device, got {type(value).__name__}."
        )

    try:
        device = torch.device(value)
    except RuntimeError as error:
        raise ValueError(f"Invalid device string {value!r}: {error}") from error

    if device.type != "cuda":
        return device

    if not torch.cuda.is_available():
        raise ValueError(
            f"device={device} requests CUDA, but "
            "torch.cuda.is_available() is False on this host."
        )

    if device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())

    count = torch.cuda.device_count()
    if device.index < 0 or device.index >= count:
        raise ValueError(
            f"device={device} is out of range: {count} CUDA device(s) "
            "available on this host."
        )

    return device
