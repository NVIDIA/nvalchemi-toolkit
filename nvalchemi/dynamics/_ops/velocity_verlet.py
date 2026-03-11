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
PyTorch bindings for velocity Verlet NVE integrator kernels.

Wraps the Warp kernels from
:mod:`nvalchemiops.dynamics.integrators.velocity_verlet` as
``torch.library.custom_op`` operations, enabling correct behaviour
under ``torch.compile`` and PyTorch's autograd infrastructure.

Functions
---------
vv_position_update
    First half-step: update positions and half-step velocities using
    current forces.
vv_velocity_finalize
    Second half-step: finalize velocities using new forces after the
    force evaluation.
"""

from __future__ import annotations

import torch
import torch.library
import warp as wp
from nvalchemiops.dynamics.integrators import (
    velocity_verlet_position_update as _vv_pos_update,
)
from nvalchemiops.dynamics.integrators import (
    velocity_verlet_velocity_finalize as _vv_vel_finalize,
)

from nvalchemi.dynamics._ops._bridge import _scalar_type, _vec_type

__all__ = ["vv_position_update", "vv_velocity_finalize"]


@torch.library.custom_op(
    "nvalchemi::vv_position_update", mutates_args={"positions", "velocities"}
)
def vv_position_update(
    positions: torch.Tensor,
    velocities: torch.Tensor,
    forces: torch.Tensor,
    masses: torch.Tensor,
    dt: torch.Tensor,
    batch_idx: torch.Tensor,
) -> None:
    """Velocity Verlet position update (first half-step).

    Computes ``r(t+dt) = r(t) + v(t)*dt + 0.5*(F/m)*dt²`` and the
    half-step velocity ``v(t+dt/2) = v(t) + 0.5*(F/m)*dt``.
    Modifies *positions* and *velocities* in-place.

    Parameters
    ----------
    positions : torch.Tensor
        Atomic positions ``[N, 3]``, float32 or float64.
    velocities : torch.Tensor
        Atomic velocities ``[N, 3]``, same dtype as *positions*.
    forces : torch.Tensor
        Atomic forces ``[N, 3]``, same dtype as *positions*.
    masses : torch.Tensor
        Atomic masses ``[N]``, same dtype as *positions*.
    dt : torch.Tensor
        Per-system timestep ``[M]``, same dtype as *positions*.
    batch_idx : torch.Tensor
        Per-atom system index ``[N]``, int32, non-decreasing.
    """
    dtype = positions.dtype
    vec_t = _vec_type(dtype)
    scl_t = _scalar_type(dtype)
    _vv_pos_update(
        wp.from_torch(positions, dtype=vec_t),
        wp.from_torch(velocities, dtype=vec_t),
        wp.from_torch(forces, dtype=vec_t),
        wp.from_torch(masses, dtype=scl_t),
        wp.from_torch(dt, dtype=scl_t),
        batch_idx=wp.from_torch(batch_idx, dtype=wp.int32),
    )


@vv_position_update.register_fake
def _vv_position_update_fake(
    positions, velocities, forces, masses, dt, batch_idx
) -> None:
    pass


@torch.library.custom_op(
    "nvalchemi::vv_velocity_finalize", mutates_args={"velocities"}
)
def vv_velocity_finalize(
    velocities: torch.Tensor,
    forces_new: torch.Tensor,
    masses: torch.Tensor,
    dt: torch.Tensor,
    batch_idx: torch.Tensor,
) -> None:
    """Velocity Verlet velocity finalize (second half-step).

    Computes ``v(t+dt) = v(t+dt/2) + 0.5*(F_new/m)*dt`` using the
    forces evaluated at the updated positions.
    Modifies *velocities* in-place.

    Parameters
    ----------
    velocities : torch.Tensor
        Half-step velocities ``[N, 3]``, float32 or float64.
    forces_new : torch.Tensor
        Forces at updated positions ``[N, 3]``, same dtype.
    masses : torch.Tensor
        Atomic masses ``[N]``, same dtype.
    dt : torch.Tensor
        Per-system timestep ``[M]``, same dtype.
    batch_idx : torch.Tensor
        Per-atom system index ``[N]``, int32, non-decreasing.
    """
    dtype = velocities.dtype
    vec_t = _vec_type(dtype)
    scl_t = _scalar_type(dtype)
    _vv_vel_finalize(
        wp.from_torch(velocities, dtype=vec_t),
        wp.from_torch(forces_new, dtype=vec_t),
        wp.from_torch(masses, dtype=scl_t),
        wp.from_torch(dt, dtype=scl_t),
        batch_idx=wp.from_torch(batch_idx, dtype=wp.int32),
    )


@vv_velocity_finalize.register_fake
def _vv_velocity_finalize_fake(velocities, forces_new, masses, dt, batch_idx) -> None:
    pass
