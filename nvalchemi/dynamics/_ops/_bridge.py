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
"""
Shared utilities for the PyTorch-Warp bridge layer.

Provides helpers for constructing system-only state batches, broadcasting
scalar parameters to per-system tensors, and resolving Warp dtype variants
from torch dtypes.
"""

from __future__ import annotations

import torch
import warp as wp

from nvalchemi.data import Batch
from nvalchemi.data.level_storage import MultiLevelStorage, UniformLevelStorage


def _make_state_batch(
    system_data: dict[str, torch.Tensor],
    device: torch.device,
) -> Batch:
    """Build a system-only :class:`~nvalchemi.data.Batch` from a dict of tensors.

    All tensors must have shape ``[M, *trailing]`` where M is the number of
    systems.  The resulting batch has no atom or edge groups — only the
    ``"system"`` :class:`~nvalchemi.data.level_storage.UniformLevelStorage`.
    This makes it suitable for holding per-system integrator state (chain
    positions, barostat velocities, per-system timesteps, etc.) while
    reusing the full :class:`~nvalchemi.data.Batch` API for index-selection
    and concatenation during inflight batching.

    Parameters
    ----------
    system_data : dict[str, torch.Tensor]
        Mapping from key name to tensor of shape ``[M, *trailing]``.
    device : torch.device
        Target device; all tensors should already reside on this device.

    Returns
    -------
    Batch
        A system-only batch whose keys are accessible as attributes, e.g.
        ``state_batch.nhc_xi``.
    """
    system_group = UniformLevelStorage(data=system_data, device=device, validate=False)
    multi = MultiLevelStorage(groups={"system": system_group})
    keys = {"system": set(system_data.keys())}
    return Batch._construct(device=device, keys=keys, storage=multi)


def _to_per_system(
    val: float | torch.Tensor,
    M: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Broadcast a scalar or tensor to shape ``[M, *trailing]``.

    Parameters
    ----------
    val : float or torch.Tensor
        Scalar value or tensor of shape ``[]``, ``[1, *trailing]``, or
        ``[M, *trailing]``.
    M : int
        Number of systems.
    device : torch.device
        Target device.
    dtype : torch.dtype
        Target dtype.

    Returns
    -------
    torch.Tensor
        Contiguous tensor of shape ``[M, *trailing]`` on *device* with
        *dtype*.
    """
    if isinstance(val, torch.Tensor):
        tensor = val.to(device=device, dtype=dtype)
        if tensor.ndim == 0:
            return tensor.expand(M).contiguous()
        leading = tensor.shape[0]
        if leading not in (1, M):
            raise ValueError(
                f"Expected leading dimension 1 or {M} for per-system broadcast, "
                f"got shape {tuple(tensor.shape)}."
            )
        return tensor.expand((M, *tensor.shape[1:])).contiguous()
    return torch.full((M,), float(val), dtype=dtype, device=device)


def _vec_type(dtype: torch.dtype) -> type:
    """Return ``wp.vec3f`` or ``wp.vec3d`` from a torch float dtype.

    Parameters
    ----------
    dtype : torch.dtype
        Either ``torch.float32`` or ``torch.float64``.

    Returns
    -------
    type
        The corresponding Warp 3-vector type.
    """
    return wp.vec3d if dtype == torch.float64 else wp.vec3f


def _mat_type(dtype: torch.dtype) -> type:
    """Return ``wp.mat33f`` or ``wp.mat33d`` from a torch float dtype.

    Parameters
    ----------
    dtype : torch.dtype
        Either ``torch.float32`` or ``torch.float64``.

    Returns
    -------
    type
        The corresponding Warp 3×3 matrix type.
    """
    return wp.mat33d if dtype == torch.float64 else wp.mat33f


def _scalar_type(dtype: torch.dtype) -> type:
    """Return ``wp.float32`` or ``wp.float64`` from a torch float dtype.

    Parameters
    ----------
    dtype : torch.dtype
        Either ``torch.float32`` or ``torch.float64``.

    Returns
    -------
    type
        The corresponding Warp scalar type.
    """
    return wp.float64 if dtype == torch.float64 else wp.float32
