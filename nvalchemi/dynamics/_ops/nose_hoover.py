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
PyTorch bindings for Nosé-Hoover chain (NHC) NVT integrator kernels.

Wraps :mod:`nvalchemiops.dynamics.integrators.nose_hoover` as
``torch.library.custom_op`` operations, enabling correct behaviour
under ``torch.compile`` and PyTorch's autograd infrastructure.

The NHC thermostat uses Yoshida-Suzuki factorization with the
Martyna-Tobias-Klein (MTK) equations for time-reversible integration.

Functions
---------
nhc_compute_masses
    Compute chain masses Q_k from temperature and coupling time τ_T.
nhc_chain_update
    Propagate the NHC system, scaling particle velocities in-place.
nhc_velocity_half_step
    Apply half-step velocity kick: ``v += 0.5 * (F/m) * dt``.
nhc_position_update
    Apply full-step position update: ``r += v * dt``.
"""

from __future__ import annotations

import torch
import torch.library
import warp as wp
from nvalchemiops.dynamics.integrators import (
    nhc_compute_masses as _nhc_compute_masses,
)
from nvalchemiops.dynamics.integrators import (
    nhc_position_update as _nhc_pos_update,
)
from nvalchemiops.dynamics.integrators import (
    nhc_thermostat_chain_update as _nhc_chain_update,
)
from nvalchemiops.dynamics.integrators import (
    nhc_velocity_half_step as _nhc_vel_half,
)

from nvalchemi.dynamics._ops._bridge import _scalar_type, _vec_type

__all__ = [
    "nhc_compute_masses",
    "nhc_chain_update",
    "nhc_velocity_half_step",
    "nhc_position_update",
]


def nhc_compute_masses(
    temperature: torch.Tensor,
    thermostat_time: torch.Tensor,
    masses: torch.Tensor,
    batch_idx: torch.Tensor,
    chain_length: int,
) -> torch.Tensor:
    """Compute Nosé-Hoover chain masses Q_k for each system.

    Uses the standard MTK formula:
    ``Q_1 = N_f * kT * τ_T²``, ``Q_k = kT * τ_T²`` for k > 1.

    .. note::
        The underlying kernel takes scalar *ndof*, *target_temp*, and *tau*
        arguments, so for heterogeneous batches the first system's values are
        used as the representative parameters.  All systems in the batch must
        share the same temperature and coupling time for correct results.

    .. note::
        Workaround: toolkit-ops ``nhc_compute_masses`` defaults to
        ``device="cuda:0"`` when no device is given.  We pass it explicitly.

    Parameters
    ----------
    temperature : torch.Tensor
        Per-system target temperature in Kelvin ``[M]``.
    thermostat_time : torch.Tensor
        Per-system thermostat coupling time τ_T ``[M]``, same dtype.
    masses : torch.Tensor
        Per-atom masses ``[N]``, same dtype.  Used to determine N_f.
    batch_idx : torch.Tensor
        Per-atom system index ``[N]``, int32, non-decreasing.
    chain_length : int
        Number of links in the Nosé-Hoover chain.

    Returns
    -------
    torch.Tensor
        Chain masses ``[M, chain_length]``, same dtype as *temperature*.
    """
    M = temperature.shape[0]
    dtype = temperature.dtype
    scl_t = _scalar_type(dtype)
    wp_dtype = wp.float64 if dtype == torch.float64 else wp.float32

    counts = torch.bincount(batch_idx.long(), minlength=M).to(torch.int32)
    ndof = (counts * 3).contiguous()

    # Q is the OUTPUT buffer; _nhc_compute_masses fills it and returns it.
    Q = torch.zeros(M, chain_length, dtype=dtype, device=temperature.device)
    # All array args must be wp.array; ndof is int32, others match temperature dtype.
    Q_wp = _nhc_compute_masses(
        ndof=wp.from_torch(ndof, dtype=wp.int32),
        target_temp=wp.from_torch(temperature.contiguous(), dtype=scl_t),
        tau=wp.from_torch(thermostat_time.contiguous(), dtype=scl_t),
        chain_length=chain_length,
        masses=wp.from_torch(Q, dtype=scl_t),
        num_systems=M,
        dtype=wp_dtype,
        device=str(temperature.device),  # workaround: kernel defaults to cuda:0
    )
    return wp.to_torch(Q_wp)


@torch.library.custom_op(
    "nvalchemi::nhc_chain_update",
    mutates_args={"velocities", "eta", "eta_dot", "ke2", "total_scale", "step_scale"},
)
def nhc_chain_update(
    velocities: torch.Tensor,
    masses: torch.Tensor,
    eta: torch.Tensor,
    eta_dot: torch.Tensor,
    Q: torch.Tensor,
    temperature: torch.Tensor,
    dt: torch.Tensor,
    ndof: torch.Tensor,
    ke2: torch.Tensor,
    total_scale: torch.Tensor,
    step_scale: torch.Tensor,
    dt_chain: torch.Tensor,
    batch_idx: torch.Tensor,
) -> None:
    """Propagate the Nosé-Hoover chain and scale particle velocities.

    Applies Yoshida-Suzuki factorization to advance the chain variables
    (``eta``, ``eta_dot``) and rescales particle velocities by the
    accumulated thermostat factor.  Modifies *velocities*, *eta*, *eta_dot*,
    and scratch tensors in-place.

    Parameters
    ----------
    velocities : torch.Tensor
        Atomic velocities ``[N, 3]``, float32 or float64.
    masses : torch.Tensor
        Atomic masses ``[N]``, same dtype.
    eta : torch.Tensor
        Chain positions ``[M, C]``, same dtype.  Updated in-place.
    eta_dot : torch.Tensor
        Chain velocities ``[M, C]``, same dtype.  Updated in-place.
    Q : torch.Tensor
        Chain masses ``[M, C]``, same dtype.
    temperature : torch.Tensor
        Per-system target temperature in Kelvin ``[M]``, same dtype.
    dt : torch.Tensor
        Per-system timestep ``[M]``, same dtype.
    ndof : torch.Tensor
        Per-system degrees of freedom ``[M]``, same float dtype as *velocities*.
    ke2 : torch.Tensor
        Scratch buffer ``[M]``, same dtype.  Written by kernel.
    total_scale : torch.Tensor
        Scratch buffer ``[M]``, same dtype.  Written by kernel.
    step_scale : torch.Tensor
        Scratch buffer ``[M]``, same dtype.  Written by kernel.
    dt_chain : torch.Tensor
        Scratch buffer ``[M]``, same dtype for Yoshida-Suzuki chain dt.
    batch_idx : torch.Tensor
        Per-atom system index ``[N]``, int32, non-decreasing.
    """
    M = temperature.shape[0]
    dtype = velocities.dtype
    vec_t = _vec_type(dtype)
    scl_t = _scalar_type(dtype)
    _nhc_chain_update(
        wp.from_torch(velocities, dtype=vec_t),
        wp.from_torch(masses, dtype=scl_t),
        wp.from_torch(eta, dtype=scl_t),
        wp.from_torch(eta_dot, dtype=scl_t),
        wp.from_torch(Q, dtype=scl_t),            # eta_mass
        wp.from_torch(temperature, dtype=scl_t),  # target_temp (wp.array)
        wp.from_torch(dt, dtype=scl_t),           # dt (wp.array)
        wp.from_torch(ndof.to(dtype=dtype), dtype=scl_t),         # ndof (wp.array, int dtype)
        wp.from_torch(ke2, dtype=scl_t),
        wp.from_torch(total_scale, dtype=scl_t),
        wp.from_torch(step_scale, dtype=scl_t),
        wp.from_torch(dt_chain, dtype=scl_t),
        batch_idx=wp.from_torch(batch_idx, dtype=wp.int32),
        num_systems=M,
    )


@nhc_chain_update.register_fake
def _nhc_chain_update_fake(
    velocities, masses, eta, eta_dot, Q, temperature, dt, ndof,
    ke2, total_scale, step_scale, dt_chain, batch_idx,
) -> None:
    pass


@torch.library.custom_op(
    "nvalchemi::nhc_velocity_half_step", mutates_args={"velocities"}
)
def nhc_velocity_half_step(
    velocities: torch.Tensor,
    forces: torch.Tensor,
    masses: torch.Tensor,
    dt: torch.Tensor,
    batch_idx: torch.Tensor,
) -> None:
    """Apply NHC half-step velocity kick: ``v += 0.5 * (F/m) * dt``.

    Modifies *velocities* in-place.

    Parameters
    ----------
    velocities : torch.Tensor
        Atomic velocities ``[N, 3]``, float32 or float64.
    forces : torch.Tensor
        Atomic forces ``[N, 3]``, same dtype.
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
    _nhc_vel_half(
        wp.from_torch(velocities, dtype=vec_t),
        wp.from_torch(forces, dtype=vec_t),
        wp.from_torch(masses, dtype=scl_t),
        wp.from_torch(dt, dtype=scl_t),
        batch_idx=wp.from_torch(batch_idx, dtype=wp.int32),
    )


@nhc_velocity_half_step.register_fake
def _nhc_velocity_half_step_fake(velocities, forces, masses, dt, batch_idx) -> None:
    pass


@torch.library.custom_op(
    "nvalchemi::nhc_position_update", mutates_args={"positions"}
)
def nhc_position_update(
    positions: torch.Tensor,
    velocities: torch.Tensor,
    dt: torch.Tensor,
    batch_idx: torch.Tensor,
) -> None:
    """Apply NHC full-step position update: ``r += v * dt``.

    Modifies *positions* in-place.

    Parameters
    ----------
    positions : torch.Tensor
        Atomic positions ``[N, 3]``, float32 or float64.
    velocities : torch.Tensor
        Atomic velocities ``[N, 3]``, same dtype.
    dt : torch.Tensor
        Per-system timestep ``[M]``, same dtype.
    batch_idx : torch.Tensor
        Per-atom system index ``[N]``, int32, non-decreasing.
    """
    dtype = positions.dtype
    vec_t = _vec_type(dtype)
    scl_t = _scalar_type(dtype)
    _nhc_pos_update(
        wp.from_torch(positions, dtype=vec_t),
        wp.from_torch(velocities, dtype=vec_t),
        wp.from_torch(dt, dtype=scl_t),
        batch_idx=wp.from_torch(batch_idx, dtype=wp.int32),
    )


@nhc_position_update.register_fake
def _nhc_position_update_fake(positions, velocities, dt, batch_idx) -> None:
    pass
