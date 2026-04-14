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
PyTorch bindings for thermostat utility kernels.

Wraps :mod:`nvalchemiops.dynamics.utils.thermostat_utils` and
:mod:`nvalchemiops.dynamics.integrators.velocity_rescaling` as
``torch.library.custom_op`` operations, enabling correct behaviour
under ``torch.compile`` and PyTorch's autograd infrastructure.

Functions
---------
initialize_velocities
    Sample velocities from Maxwell-Boltzmann and optionally remove COM.
remove_com_motion
    Zero center-of-mass velocity for each system.
compute_kinetic_energy
    Compute per-system kinetic energy KE = Σ 0.5*m*v².
compute_temperature
    Compute instantaneous temperature T = 2·KE / (N_f * kB).
velocity_rescale
    Rescale velocities by a per-system factor (e.g. for velocity rescaling
    thermostat).
"""

from __future__ import annotations

import torch
import torch.library
import warp as wp
from nvalchemiops.dynamics.integrators import (
    velocity_rescale as _vel_rescale,
)
from nvalchemiops.dynamics.utils.thermostat_utils import (
    compute_kinetic_energy as _compute_ke,
)
from nvalchemiops.dynamics.utils.thermostat_utils import (
    compute_temperature as _compute_T,
)
from nvalchemiops.dynamics.utils.thermostat_utils import (
    initialize_velocities as _init_vel,
)
from nvalchemiops.dynamics.utils.thermostat_utils import (
    remove_com_motion as _remove_com,
)

from nvalchemi.dynamics._ops._bridge import _scalar_type, _vec_type

__all__ = [
    "initialize_velocities",
    "remove_com_motion",
    "compute_kinetic_energy",
    "compute_temperature",
    "velocity_rescale",
]


def _remove_angular_momentum(
    velocities: torch.Tensor,
    positions: torch.Tensor,
    masses: torch.Tensor,
    batch_idx: torch.Tensor,
    num_systems: int,
) -> None:
    """Zero angular momentum per system by removing rigid-body rotation.

    For each system, computes the angular velocity omega = I^{-1} L
    (from the moment of inertia tensor I and angular momentum L about
    the center of mass) and subtracts the rotational component
    v_rot = omega x (r - r_COM) from each atom's velocity.

    Operates in-place on *velocities*.  Only meaningful for non-periodic
    (isolated) systems.  Pure-PyTorch implementation (runs once at init).
    """
    dtype = velocities.dtype
    device = velocities.device
    idx = batch_idx.long()  # (N,)
    m = masses.unsqueeze(-1)  # (N, 1)

    # Compute COM position per system.
    total_mass = torch.zeros(num_systems, 1, dtype=dtype, device=device)
    total_mass.scatter_add_(0, idx.unsqueeze(-1), m)
    weighted_pos = torch.zeros(num_systems, 3, dtype=dtype, device=device)
    weighted_pos.scatter_add_(0, idx.unsqueeze(-1).expand_as(positions), m * positions)
    com_pos = weighted_pos / total_mass.clamp_min(1e-30)  # (M, 3)

    # Relative position and velocity w.r.t. COM.
    r = positions - com_pos[idx]  # (N, 3)
    v = velocities  # (N, 3) — will modify in-place

    # Angular momentum per system: L = sum_i m_i * (r_i x v_i).
    L_per_atom = m * torch.linalg.cross(r, v)  # (N, 3)
    L = torch.zeros(num_systems, 3, dtype=dtype, device=device)
    L.scatter_add_(0, idx.unsqueeze(-1).expand_as(L_per_atom), L_per_atom)

    # Moment of inertia tensor per system: I_ab = sum_i m_i * (|r_i|^2 d_ab - r_ia * r_ib).
    r2 = (r * r).sum(dim=-1, keepdim=True)  # (N, 1)
    # Diagonal: m * r^2
    I_diag = torch.zeros(num_systems, 3, dtype=dtype, device=device)
    I_diag.scatter_add_(0, idx.unsqueeze(-1).expand(-1, 3), (m * r2).expand(-1, 3))
    # Off-diagonal: -m * r_a * r_b (build full 3x3 per atom, accumulate)
    I_outer = torch.zeros(num_systems, 3, 3, dtype=dtype, device=device)
    rr = torch.einsum("na,nb->nab", r, r)  # (N, 3, 3)
    mrr = m.unsqueeze(-1) * rr  # (N, 3, 3)
    I_outer.scatter_add_(0, idx.unsqueeze(-1).unsqueeze(-1).expand_as(mrr), mrr)
    I_total = torch.diag_embed(I_diag) - I_outer  # (M, 3, 3)

    # Angular velocity: omega = I^{-1} L.
    # Use pseudo-inverse for robustness (handles linear molecules / single atoms).
    omega = torch.linalg.lstsq(I_total, L.unsqueeze(-1)).solution.squeeze(-1)  # (M, 3)

    # Subtract rotational component: v -= omega x r.
    v_rot = torch.linalg.cross(omega[idx], r)  # (N, 3)
    velocities -= v_rot


def _rescale_to_temperature(
    velocities: torch.Tensor,
    masses: torch.Tensor,
    target_temperature: torch.Tensor,
    batch_idx: torch.Tensor,
    num_systems: int,
) -> None:
    """Rescale velocities so per-system kinetic temperature matches target.

    After COM/rotation removal, the kinetic temperature drifts below
    the target.  This function computes the actual KE per system and
    rescales each atom's velocity by ``sqrt(T_target / T_actual)``.

    Operates in-place on *velocities*.
    """
    from nvalchemi.dynamics.hooks._utils import KB_EV  # noqa: PLC0415

    m = masses.unsqueeze(-1) if masses.dim() == 1 else masses  # (N, 1)
    v2 = (velocities**2).sum(dim=-1, keepdim=True)  # (N, 1)
    ke_per_atom = 0.5 * m * v2  # (N, 1)

    # Sum KE per system.
    ke_per_system = torch.zeros(
        num_systems, 1, dtype=velocities.dtype, device=velocities.device
    )
    ke_per_system.scatter_add_(0, batch_idx.unsqueeze(-1).long(), ke_per_atom)

    # Count atoms per system.
    n_atoms = torch.bincount(batch_idx.long(), minlength=num_systems).float()

    # Actual temperature per system: T = 2 * KE / (3 * N * kB).
    t_actual = (2.0 * ke_per_system.squeeze(-1)) / (3.0 * n_atoms * KB_EV)  # (M,)

    # Target temperature per system.
    t_target = target_temperature  # (M,)

    # Scale factor per system: sqrt(T_target / T_actual).
    # Clamp to avoid division by zero for single-atom systems.
    scale = torch.where(
        t_actual > 1e-10,
        (t_target / t_actual).sqrt(),
        torch.ones_like(t_actual),
    )  # (M,)

    # Apply per-atom scale.
    velocities *= scale[batch_idx.long()].unsqueeze(-1)


@torch.library.custom_op(
    "nvalchemi::initialize_velocities", mutates_args={"velocities"}
)
def initialize_velocities(
    velocities: torch.Tensor,
    masses: torch.Tensor,
    temperature: torch.Tensor,
    batch_idx: torch.Tensor,
    random_seed: int = 42,
    remove_com: bool = True,
    remove_rotations: bool = False,
    rescale: bool = True,
    positions: torch.Tensor | None = None,
) -> None:
    """Initialize velocities from the Maxwell-Boltzmann distribution.

    Draws velocities for each atom such that the per-system kinetic
    temperature matches *temperature*.  Optionally removes center-of-mass
    drift and rigid-body rotation, then rescales to the exact target
    temperature.  Modifies *velocities* in-place.

    Parameters
    ----------
    velocities : torch.Tensor
        Atomic velocities to initialize ``[N, 3]``, float32 or float64.
    masses : torch.Tensor
        Atomic masses ``[N]``, same dtype.
    temperature : torch.Tensor
        Per-system target temperature in Kelvin ``[M]``, same dtype.
    batch_idx : torch.Tensor
        Per-atom system index ``[N]``, int32, non-decreasing.
    random_seed : int, optional
        Global RNG seed.  Default 42.
    remove_com : bool, optional
        If True, subtract the COM velocity from each system after
        sampling.  Default True.
    remove_rotations : bool, optional
        If True, zero the angular momentum of each system after
        sampling.  Only meaningful for non-periodic (isolated) systems.
        Requires *positions* to be provided.  Default False.
    rescale : bool, optional
        If True, rescale velocities after COM/rotation removal so
        that the kinetic temperature matches the target exactly.
        Default True.
    positions : torch.Tensor | None, optional
        Atomic positions ``[N, 3]``.  Required when ``remove_rotations=True``
        (needed to compute the moment of inertia tensor).
    """
    from nvalchemi.dynamics.hooks._utils import KB_EV  # noqa: PLC0415

    dtype = velocities.dtype
    vec_t = _vec_type(dtype)
    scl_t = _scalar_type(dtype)
    M = temperature.shape[0]
    device = velocities.device

    # The nvalchemiops kernel expects kB*T in energy units (eV),
    # but the public API accepts temperature in Kelvin.
    kT = temperature * KB_EV

    total_momentum = torch.zeros(M, 3, dtype=dtype, device=device)
    total_mass = torch.zeros(M, dtype=dtype, device=device)
    com_velocities = torch.zeros(M, 3, dtype=dtype, device=device)

    _init_vel(
        wp.from_torch(velocities, dtype=vec_t),
        wp.from_torch(masses, dtype=scl_t),
        wp.from_torch(kT, dtype=scl_t),
        wp.from_torch(total_momentum, dtype=vec_t),
        wp.from_torch(total_mass, dtype=scl_t),
        wp.from_torch(com_velocities, dtype=vec_t),
        random_seed=random_seed,
        remove_com=remove_com,
        batch_idx=wp.from_torch(batch_idx, dtype=wp.int32),
        num_systems=M,
    )

    if remove_rotations:
        if positions is None:
            raise ValueError("positions must be provided when remove_rotations=True")
        _remove_angular_momentum(velocities, positions, masses, batch_idx, M)

    if rescale and (remove_com or remove_rotations):
        _rescale_to_temperature(velocities, masses, temperature, batch_idx, M)


@initialize_velocities.register_fake
def _initialize_velocities_fake(
    velocities,
    masses,
    temperature,
    batch_idx,
    random_seed=42,
    remove_com=True,
    remove_rotations=False,
    rescale=True,
    positions=None,
) -> None:
    pass


@torch.library.custom_op("nvalchemi::remove_com_motion", mutates_args={"velocities"})
def remove_com_motion(
    velocities: torch.Tensor,
    masses: torch.Tensor,
    batch_idx: torch.Tensor,
    num_systems: int,
) -> None:
    """Remove center-of-mass velocity for each system.

    Computes v_COM = Σ(m*v) / Σ(m) per system and subtracts it from all
    atom velocities.  Modifies *velocities* in-place.

    Parameters
    ----------
    velocities : torch.Tensor
        Atomic velocities ``[N, 3]``, float32 or float64.
    masses : torch.Tensor
        Atomic masses ``[N]``, same dtype.
    batch_idx : torch.Tensor
        Per-atom system index ``[N]``, int32, non-decreasing.
    num_systems : int
        Number of systems M.  Required because M cannot be inferred from
        tensor shapes alone.
    """
    dtype = velocities.dtype
    vec_t = _vec_type(dtype)
    scl_t = _scalar_type(dtype)
    M = num_systems
    device = velocities.device

    total_momentum = torch.zeros(M, 3, dtype=dtype, device=device)
    total_mass = torch.zeros(M, dtype=dtype, device=device)
    com_velocities = torch.zeros(M, 3, dtype=dtype, device=device)

    _remove_com(
        wp.from_torch(velocities, dtype=vec_t),
        wp.from_torch(masses, dtype=scl_t),
        wp.from_torch(total_momentum, dtype=vec_t),
        wp.from_torch(total_mass, dtype=scl_t),
        wp.from_torch(com_velocities, dtype=vec_t),
        batch_idx=wp.from_torch(batch_idx, dtype=wp.int32),
        num_systems=M,
    )


@remove_com_motion.register_fake
def _remove_com_motion_fake(velocities, masses, batch_idx, num_systems) -> None:
    pass


@torch.library.custom_op("nvalchemi::compute_kinetic_energy", mutates_args=())
def compute_kinetic_energy(
    velocities: torch.Tensor,
    masses: torch.Tensor,
    batch_idx: torch.Tensor,
    num_systems: int,
) -> torch.Tensor:
    """Compute per-system kinetic energy KE = Σ_i 0.5 * m_i * v_i².

    Parameters
    ----------
    velocities : torch.Tensor
        Atomic velocities ``[N, 3]``, float32 or float64.
    masses : torch.Tensor
        Atomic masses ``[N]``, same dtype.
    batch_idx : torch.Tensor
        Per-atom system index ``[N]``, int32, non-decreasing.
    num_systems : int
        Number of systems M.  Required because M cannot be inferred from
        tensor shapes alone.

    Returns
    -------
    torch.Tensor
        Per-system kinetic energy ``[M]``, same dtype.
    """
    dtype = velocities.dtype
    vec_t = _vec_type(dtype)
    scl_t = _scalar_type(dtype)
    M = num_systems
    device = velocities.device
    ke = torch.zeros(M, dtype=dtype, device=device)
    _compute_ke(
        wp.from_torch(velocities, dtype=vec_t),
        wp.from_torch(masses, dtype=scl_t),
        wp.from_torch(ke, dtype=scl_t),
        batch_idx=wp.from_torch(batch_idx, dtype=wp.int32),
        num_systems=M,
    )
    return ke


@compute_kinetic_energy.register_fake
def _compute_kinetic_energy_fake(
    velocities, masses, batch_idx, num_systems
) -> torch.Tensor:
    return velocities.new_empty(num_systems)


@torch.library.custom_op("nvalchemi::compute_temperature", mutates_args=())
def compute_temperature(
    kinetic_energy: torch.Tensor,
    num_atoms_per_system: torch.Tensor,
) -> torch.Tensor:
    """Compute instantaneous temperature from kinetic energy.

    Uses the equipartition theorem: ``T = 2·KE / (3·N * kB)``.

    Parameters
    ----------
    kinetic_energy : torch.Tensor
        Per-system kinetic energy ``[M]``, float32 or float64.
    num_atoms_per_system : torch.Tensor
        Number of atoms per system ``[M]``, int32.

    Returns
    -------
    torch.Tensor
        Per-system temperature in Kelvin ``[M]``, same dtype as
        *kinetic_energy*.
    """
    dtype = kinetic_energy.dtype
    scl_t = _scalar_type(dtype)
    M = kinetic_energy.shape[0]
    device = kinetic_energy.device
    temperature = torch.zeros(M, dtype=dtype, device=device)
    _compute_T(
        wp.from_torch(kinetic_energy, dtype=scl_t),
        wp.from_torch(temperature, dtype=scl_t),
        wp.from_torch(num_atoms_per_system, dtype=wp.int32),
    )
    return temperature


@compute_temperature.register_fake
def _compute_temperature_fake(kinetic_energy, num_atoms_per_system) -> torch.Tensor:
    return kinetic_energy.new_empty(kinetic_energy.shape[0])


@torch.library.custom_op("nvalchemi::velocity_rescale", mutates_args={"velocities"})
def velocity_rescale(
    velocities: torch.Tensor,
    scale_factor: torch.Tensor,
    batch_idx: torch.Tensor,
) -> None:
    """Rescale velocities by a per-system factor.

    Computes ``v_i *= scale_factor[sys_i]`` for each atom.
    Modifies *velocities* in-place.

    Parameters
    ----------
    velocities : torch.Tensor
        Atomic velocities ``[N, 3]``, float32 or float64.
    scale_factor : torch.Tensor
        Per-system rescaling factor ``[M]``, same dtype.
    batch_idx : torch.Tensor
        Per-atom system index ``[N]``, int32, non-decreasing.
    """
    dtype = velocities.dtype
    vec_t = _vec_type(dtype)
    scl_t = _scalar_type(dtype)
    _vel_rescale(
        wp.from_torch(velocities, dtype=vec_t),
        wp.from_torch(scale_factor, dtype=scl_t),
        batch_idx=wp.from_torch(batch_idx, dtype=wp.int32),
    )


@velocity_rescale.register_fake
def _velocity_rescale_fake(velocities, scale_factor, batch_idx) -> None:
    pass
