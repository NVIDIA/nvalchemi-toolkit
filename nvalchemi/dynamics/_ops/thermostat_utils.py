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
) -> None:
    """Initialize velocities from the Maxwell-Boltzmann distribution.

    Draws velocities for each atom such that the per-system kinetic
    temperature matches *temperature*.  Optionally removes center-of-mass
    drift.  Modifies *velocities* in-place.

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
    """
    dtype = velocities.dtype
    vec_t = _vec_type(dtype)
    scl_t = _scalar_type(dtype)
    M = temperature.shape[0]
    device = velocities.device

    total_momentum = torch.zeros(M, 3, dtype=dtype, device=device)
    total_mass = torch.zeros(M, dtype=dtype, device=device)
    com_velocities = torch.zeros(M, 3, dtype=dtype, device=device)

    _init_vel(
        wp.from_torch(velocities, dtype=vec_t),
        wp.from_torch(masses, dtype=scl_t),
        wp.from_torch(temperature, dtype=scl_t),
        wp.from_torch(total_momentum, dtype=vec_t),
        wp.from_torch(total_mass, dtype=scl_t),
        wp.from_torch(com_velocities, dtype=vec_t),
        random_seed=random_seed,
        remove_com=remove_com,
        batch_idx=wp.from_torch(batch_idx, dtype=wp.int32),
        num_systems=M,
    )


@initialize_velocities.register_fake
def _initialize_velocities_fake(
    velocities, masses, temperature, batch_idx, random_seed=42, remove_com=True
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
