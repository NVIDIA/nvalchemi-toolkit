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
PyTorch bindings for NPT and NPH barostat/pressure kernels.

Wraps :mod:`nvalchemiops.dynamics.integrators.npt` as
``torch.library.custom_op`` operations, enabling correct behaviour
under ``torch.compile`` and PyTorch's autograd infrastructure.

The NPT integrator uses the Martyna-Tobias-Klein (MTK) equations with a
Nosé-Hoover chain thermostat coupled to the particle and barostat DOFs.
NPH omits the thermostat, allowing temperature to fluctuate.

Functions
---------
compute_pressure_tensor
    Compute the full instantaneous pressure tensor P = (KE + virial) / V.
compute_scalar_pressure
    Compute scalar pressure P = Tr(P_tensor) / 3.
compute_barostat_mass
    Compute barostat inertia W = (N_f + d) * kT * τ_P².
nph_barostat_half_step
    NPH cell-velocity half-step (no thermostat drag term).
nph_velocity_half_step
    NPH particle velocity half-step coupled to barostat strain rate.
npt_barostat_half_step
    NPT cell-velocity half-step with thermostat η̇₁ drag term.
npt_thermostat_half_step
    NHC thermostat half-step used in NPT (updates chain variables).
npt_velocity_half_step
    NPT particle velocity half-step coupled to both thermostat and barostat.
npt_position_update
    Full-step position update including cell strain: shared by NPT/NPH.
npt_cell_update
    Full-step cell matrix update: shared by NPT/NPH.
stress_to_cell_force
    Convert stress tensor to cell force for variable-cell optimization.
"""

from __future__ import annotations

import torch
import torch.library
import warp as wp
from nvalchemiops.dynamics.integrators import (
    compute_barostat_mass as _compute_baro_mass,
)
from nvalchemiops.dynamics.integrators import (
    compute_pressure_tensor as _compute_P,
)
from nvalchemiops.dynamics.integrators import (
    compute_scalar_pressure as _compute_P_scalar,
)
from nvalchemiops.dynamics.integrators import (
    nph_barostat_half_step as _nph_baro_half,
)
from nvalchemiops.dynamics.integrators import (
    nph_velocity_half_step as _nph_vel_half,
)
from nvalchemiops.dynamics.integrators import (
    npt_barostat_half_step as _npt_baro_half,
)
from nvalchemiops.dynamics.integrators import (
    npt_cell_update as _npt_cell_update,
)
from nvalchemiops.dynamics.integrators import (
    npt_position_update as _npt_pos_update,
)
from nvalchemiops.dynamics.integrators import (
    npt_thermostat_half_step as _npt_thermo_half,
)
from nvalchemiops.dynamics.integrators import (
    npt_velocity_half_step as _npt_vel_half,
)
from nvalchemiops.dynamics.integrators.npt import vec9d, vec9f
from nvalchemiops.dynamics.utils.cell_filter import (
    stress_to_cell_force as _stress_to_cell,
)

from nvalchemi.dynamics._ops._bridge import _mat_type, _scalar_type, _vec_type


def _vec9_type(dtype: torch.dtype):
    """Return the warp vec9 type matching the given torch float dtype."""
    return vec9f if dtype == torch.float32 else vec9d


def _target_pressure_wp_array(target_pressure: torch.Tensor):
    """Convert a target-pressure tensor to the matching Warp array dtype.

    Parameters
    ----------
    target_pressure : torch.Tensor
        ``[M]`` (isotropic), ``[M, 3]`` (anisotropic), or
        ``[M, 3, 3]`` (triclinic).

    Returns
    -------
    wp.array
        Warp array with scalar, vec3, or vec9 dtype as appropriate.
    """
    dtype = target_pressure.dtype
    scl_t = _scalar_type(dtype)
    vec_t = _vec_type(dtype)
    vec9_t = _vec9_type(dtype)
    if target_pressure.ndim == 1:
        return wp.from_torch(target_pressure.contiguous(), dtype=scl_t)
    if target_pressure.ndim == 2:
        if target_pressure.shape[-1] != 3:
            raise ValueError(
                "Anisotropic target pressure must have shape [M, 3], got "
                f"{tuple(target_pressure.shape)}."
            )
        return wp.from_torch(target_pressure.contiguous(), dtype=vec_t)
    if target_pressure.ndim == 3:
        if target_pressure.shape[-2:] != (3, 3):
            raise ValueError(
                "Triclinic target pressure must have shape [M, 3, 3], got "
                f"{tuple(target_pressure.shape)}."
            )
        reshaped = target_pressure.reshape(target_pressure.shape[0], 9).contiguous()
        return wp.from_torch(reshaped, dtype=vec9_t)
    raise ValueError(
        "Target pressure must be rank-1, rank-2, or rank-3, got "
        f"{target_pressure.ndim}."
    )


__all__ = [
    "compute_pressure_tensor",
    "compute_scalar_pressure",
    "compute_barostat_mass",
    "nph_barostat_half_step",
    "nph_velocity_half_step",
    "npt_barostat_half_step",
    "npt_thermostat_half_step",
    "npt_velocity_half_step",
    "npt_position_update",
    "npt_cell_update",
    "stress_to_cell_force",
]


@torch.library.custom_op(
    "nvalchemi::compute_pressure_tensor",
    mutates_args={"kinetic_tensors", "pressure_tensors", "volumes"},
)
def compute_pressure_tensor(
    velocities: torch.Tensor,
    masses: torch.Tensor,
    virial: torch.Tensor,
    cell: torch.Tensor,
    kinetic_tensors: torch.Tensor,
    pressure_tensors: torch.Tensor,
    volumes: torch.Tensor,
    batch_idx: torch.Tensor,
) -> torch.Tensor:
    """Compute the full instantaneous pressure tensor for each system.

    ``P = (KE_tensor + virial) / V``

    Pre-allocated scratch arrays (*kinetic_tensors*, *pressure_tensors*,
    *volumes*) are zeroed internally before use; allocate them once and
    reuse across steps to avoid repeated GPU allocation.

    Parameters
    ----------
    velocities : torch.Tensor
        Atomic velocities ``[N, 3]``, float32 or float64.
    masses : torch.Tensor
        Per-atom masses ``[N]``, same dtype.
    virial : torch.Tensor
        Per-system virial tensor ``W = -dE/d(epsilon)`` ``[M, 3, 3]``
        in eV, same dtype.
    cell : torch.Tensor
        Per-system cell matrix ``[M, 3, 3]``, same dtype.
    kinetic_tensors : torch.Tensor
        Scratch buffer ``[M, 9]``, same dtype.  Zeroed by kernel.
        Must be 2-D (not [M, 3, 3]) so warp sees it as ``array2d``.
    pressure_tensors : torch.Tensor
        Output buffer ``[M, 9]``, same dtype.  Written by kernel.
    volumes : torch.Tensor
        Scratch buffer ``[M]``, same dtype. Zeroed by kernel.
    batch_idx : torch.Tensor
        Per-atom system index ``[N]``, int32, non-decreasing.

    Returns
    -------
    torch.Tensor
        Pressure tensor ``[M, 9]`` (vec9 layout), same dtype as *velocities*.
    """
    M = virial.shape[0]
    dtype = velocities.dtype
    vec_t = _vec_type(dtype)
    mat_t = _mat_type(dtype)
    scl_t = _scalar_type(dtype)
    vec9_t = _vec9_type(dtype)
    P_wp = _compute_P(
        wp.from_torch(velocities, dtype=vec_t),
        wp.from_torch(masses, dtype=scl_t),
        wp.from_torch(virial.reshape(M, 9).contiguous(), dtype=vec9_t),
        wp.from_torch(cell, dtype=mat_t),
        wp.from_torch(kinetic_tensors, dtype=scl_t),  # [M, 9] array2d scalar
        wp.from_torch(pressure_tensors, dtype=vec9_t),  # [M, 9] as vec9 [M]
        wp.from_torch(volumes, dtype=scl_t),
        batch_idx=wp.from_torch(batch_idx, dtype=wp.int32),
    )
    return wp.to_torch(P_wp)


@compute_pressure_tensor.register_fake
def _compute_pressure_tensor_fake(
    velocities,
    masses,
    virial,
    cell,
    kinetic_tensors,
    pressure_tensors,
    volumes,
    batch_idx,
) -> torch.Tensor:
    M = virial.shape[0]
    return velocities.new_empty(M, 9)


@torch.library.custom_op(
    "nvalchemi::compute_scalar_pressure",
    mutates_args={"scalar_pressures"},
)
def compute_scalar_pressure(
    pressure_tensor: torch.Tensor,
    scalar_pressures: torch.Tensor,
) -> None:
    """Compute scalar pressure as Tr(P) / 3 for each system in-place.

    Parameters
    ----------
    pressure_tensor : torch.Tensor
        Full pressure tensor ``[M, 3, 3]``, float32 or float64.
    scalar_pressures : torch.Tensor
        Output buffer ``[M]``, same dtype.  Written in-place.
    """
    dtype = pressure_tensor.dtype
    vec9_t = _vec9_type(dtype)
    scl_t = _scalar_type(dtype)
    _compute_P_scalar(
        wp.from_torch(pressure_tensor, dtype=vec9_t),  # [M, 9] as vec9 [M]
        wp.from_torch(scalar_pressures, dtype=scl_t),
    )


@compute_scalar_pressure.register_fake
def _compute_scalar_pressure_fake(pressure_tensor, scalar_pressures) -> None:
    pass


@torch.library.custom_op(
    "nvalchemi::compute_barostat_mass",
    mutates_args={"masses_out"},
)
def compute_barostat_mass(
    temperature: torch.Tensor,
    barostat_time: torch.Tensor,
    num_atoms_per_system: torch.Tensor,
    masses_out: torch.Tensor,
) -> None:
    """Compute barostat inertia W = (N_f + d) * kT * τ_P² in-place.

    .. note::
        The underlying kernel takes scalar temperature and tau_p.
        The first system's values are used as representative parameters.

    Parameters
    ----------
    temperature : torch.Tensor
        Per-system temperature in Kelvin ``[M]``, float32 or float64.
    barostat_time : torch.Tensor
        Per-system barostat coupling time τ_P ``[M]``, same dtype.
    num_atoms_per_system : torch.Tensor
        Number of atoms per system ``[M]``, int32.
    masses_out : torch.Tensor
        Output buffer ``[M]``, same dtype.  Written in-place.
    """
    dtype = masses_out.dtype
    scl_t = _scalar_type(dtype)
    _compute_baro_mass(
        wp.from_torch(temperature.to(dtype), dtype=scl_t),
        wp.from_torch(barostat_time.to(dtype), dtype=scl_t),
        wp.from_torch(num_atoms_per_system, dtype=wp.int32),
        wp.from_torch(masses_out, dtype=scl_t),
    )


@compute_barostat_mass.register_fake
def _compute_barostat_mass_fake(
    temperature, barostat_time, num_atoms_per_system, masses_out
) -> None:
    pass


@torch.library.custom_op(
    "nvalchemi::nph_barostat_half_step", mutates_args={"cell_velocity"}
)
def nph_barostat_half_step(
    cell_velocity: torch.Tensor,
    pressure_tensor: torch.Tensor,
    target_pressure: torch.Tensor,
    volumes: torch.Tensor,
    W: torch.Tensor,
    kinetic_energy: torch.Tensor,
    num_atoms_per_system: torch.Tensor,
    dt: torch.Tensor,
) -> None:
    """NPH barostat cell-velocity half-step.

    Updates ``ḣ`` via ``ḧ = (V/W)(P_inst - P_ext)`` (no thermostat drag).
    Modifies *cell_velocity* in-place.

    Parameters
    ----------
    cell_velocity : torch.Tensor
        Per-system cell velocity matrix ḣ ``[M, 3, 3]``, float32/float64.
    pressure_tensor : torch.Tensor
        Instantaneous pressure tensor ``[M, 3, 3]``, same dtype.
    target_pressure : torch.Tensor
        Target pressure ``[M]`` (isotropic), ``[M, 3]`` (anisotropic),
        or ``[M, 3, 3]`` (triclinic).
    volumes : torch.Tensor
        Per-system cell volumes ``[M]``, same dtype.
    W : torch.Tensor
        Barostat inertia ``[M]``, same dtype.
    kinetic_energy : torch.Tensor
        Per-system kinetic energy ``[M]``, same dtype.
    num_atoms_per_system : torch.Tensor
        Number of atoms per system ``[M]``, int32.
    dt : torch.Tensor
        Per-system timestep ``[M]``, same dtype.
    """
    dtype = cell_velocity.dtype
    mat_t = _mat_type(dtype)
    scl_t = _scalar_type(dtype)
    vec9_t = _vec9_type(dtype)
    _nph_baro_half(
        wp.from_torch(cell_velocity, dtype=mat_t),
        wp.from_torch(pressure_tensor, dtype=vec9_t),  # [M, 9] as vec9 [M]
        _target_pressure_wp_array(target_pressure),
        wp.from_torch(volumes, dtype=scl_t),
        wp.from_torch(W, dtype=scl_t),
        wp.from_torch(kinetic_energy, dtype=scl_t),
        wp.from_torch(num_atoms_per_system, dtype=wp.int32),
        wp.from_torch(dt, dtype=scl_t),
    )


@nph_barostat_half_step.register_fake
def _nph_barostat_half_step_fake(
    cell_velocity,
    pressure_tensor,
    target_pressure,
    volumes,
    W,
    kinetic_energy,
    num_atoms_per_system,
    dt,
) -> None:
    pass


@torch.library.custom_op(
    "nvalchemi::nph_velocity_half_step", mutates_args={"velocities"}
)
def nph_velocity_half_step(
    velocities: torch.Tensor,
    masses: torch.Tensor,
    forces: torch.Tensor,
    cell_velocity: torch.Tensor,
    volumes: torch.Tensor,
    num_atoms_per_system: torch.Tensor,
    dt: torch.Tensor,
    batch_idx: torch.Tensor,
    cells_inv: torch.Tensor,
    pressure_mode: str = "isotropic",
) -> None:
    """NPH particle velocity half-step coupled to barostat strain rate.

    Applies ``v += 0.5*(F/m - (1 + 1/N_f)*ε̇·v)*dt`` where ε̇ = ḣ·h⁻¹.
    Modifies *velocities* in-place.

    Parameters
    ----------
    velocities : torch.Tensor
        Atomic velocities ``[N, 3]``, float32 or float64.
    masses : torch.Tensor
        Atomic masses ``[N]``, same dtype.
    forces : torch.Tensor
        Atomic forces ``[N, 3]``, same dtype.
    cell_velocity : torch.Tensor
        Per-system cell velocity ḣ ``[M, 3, 3]``, same dtype.
    volumes : torch.Tensor
        Per-system cell volumes ``[M]``, same dtype.
    num_atoms_per_system : torch.Tensor
        Number of atoms per system ``[M]``, int32.
    dt : torch.Tensor
        Per-system timestep ``[M]``, same dtype.
    batch_idx : torch.Tensor
        Per-atom system index ``[N]``, int32, non-decreasing.
    cells_inv : torch.Tensor
        Pre-computed inverse cell matrices ``[M, 3, 3]``, same dtype.
    pressure_mode : str, optional
        Pressure control mode: ``"isotropic"``, ``"anisotropic"``,
        or ``"triclinic"``.  Default ``"isotropic"``.
    """
    N = velocities.shape[0]
    dtype = velocities.dtype
    vec_t = _vec_type(dtype)
    mat_t = _mat_type(dtype)
    scl_t = _scalar_type(dtype)
    _nph_vel_half(
        wp.from_torch(velocities, dtype=vec_t),
        wp.from_torch(masses, dtype=scl_t),
        wp.from_torch(forces, dtype=vec_t),
        wp.from_torch(cell_velocity, dtype=mat_t),
        wp.from_torch(volumes, dtype=scl_t),
        N,
        wp.from_torch(dt, dtype=scl_t),
        batch_idx=wp.from_torch(batch_idx, dtype=wp.int32),
        num_atoms_per_system=wp.from_torch(num_atoms_per_system, dtype=wp.int32),
        cells_inv=wp.from_torch(cells_inv, dtype=mat_t),
        mode=pressure_mode,
    )


@nph_velocity_half_step.register_fake
def _nph_velocity_half_step_fake(
    velocities,
    masses,
    forces,
    cell_velocity,
    volumes,
    num_atoms_per_system,
    dt,
    batch_idx,
    cells_inv,
    pressure_mode="isotropic",
) -> None:
    pass


@torch.library.custom_op(
    "nvalchemi::npt_barostat_half_step", mutates_args={"cell_velocity"}
)
def npt_barostat_half_step(
    cell_velocity: torch.Tensor,
    pressure_tensor: torch.Tensor,
    target_pressure: torch.Tensor,
    volumes: torch.Tensor,
    W: torch.Tensor,
    kinetic_energy: torch.Tensor,
    num_atoms_per_system: torch.Tensor,
    eta_dots: torch.Tensor,
    dt: torch.Tensor,
) -> None:
    """NPT barostat cell-velocity half-step with thermostat drag.

    Updates ``ḣ`` via ``ḧ = (V/W)(P_inst - P_ext) - η̇₁·ḣ``.
    Modifies *cell_velocity* in-place.

    Parameters
    ----------
    cell_velocity : torch.Tensor
        Per-system cell velocity ḣ ``[M, 3, 3]``, float32/float64.
    pressure_tensor : torch.Tensor
        Instantaneous pressure ``[M, 3, 3]``, same dtype.
    target_pressure : torch.Tensor
        Target pressure ``[M]`` (isotropic), ``[M, 3]`` (anisotropic),
        or ``[M, 3, 3]`` (triclinic).
    volumes : torch.Tensor
        Per-system cell volumes ``[M]``, same dtype.
    W : torch.Tensor
        Barostat inertia ``[M]``, same dtype.
    kinetic_energy : torch.Tensor
        Per-system kinetic energy ``[M]``, same dtype.
    num_atoms_per_system : torch.Tensor
        Number of atoms per system ``[M]``, int32.
    eta_dots : torch.Tensor
        Full NHC chain velocities ``[M, chain_length]``, same dtype.
        The kernel reads only ``eta_dots[:, 0]`` (first chain link).
    dt : torch.Tensor
        Per-system timestep ``[M]``, same dtype.
    """
    dtype = cell_velocity.dtype
    mat_t = _mat_type(dtype)
    scl_t = _scalar_type(dtype)
    vec9_t = _vec9_type(dtype)
    _npt_baro_half(
        wp.from_torch(cell_velocity, dtype=mat_t),
        wp.from_torch(pressure_tensor, dtype=vec9_t),  # [M, 9] as vec9 [M]
        _target_pressure_wp_array(target_pressure),
        wp.from_torch(volumes, dtype=scl_t),
        wp.from_torch(W, dtype=scl_t),
        wp.from_torch(kinetic_energy, dtype=scl_t),
        wp.from_torch(num_atoms_per_system, dtype=wp.int32),
        wp.from_torch(eta_dots, dtype=scl_t),
        wp.from_torch(dt, dtype=scl_t),
    )


@npt_barostat_half_step.register_fake
def _npt_barostat_half_step_fake(
    cell_velocity,
    pressure_tensor,
    target_pressure,
    volumes,
    W,
    kinetic_energy,
    num_atoms_per_system,
    eta_dots,
    dt,
) -> None:
    pass


@torch.library.custom_op(
    "nvalchemi::npt_thermostat_half_step",
    mutates_args={"eta", "eta_dot"},
)
def npt_thermostat_half_step(
    eta: torch.Tensor,
    eta_dot: torch.Tensor,
    kinetic_energy: torch.Tensor,
    temperature: torch.Tensor,
    thermostat_masses: torch.Tensor,
    num_atoms_per_system: torch.Tensor,
    chain_length: int,
    dt: torch.Tensor,
) -> None:
    """NPT NHC thermostat half-step for particle or cell DOFs.

    Propagates the chain variables (*eta*, *eta_dot*) based on the current
    kinetic energy.  Particle velocity scaling is performed by the caller
    after this function using the updated chain velocities.
    Modifies *eta* and *eta_dot* in-place.

    Parameters
    ----------
    eta : torch.Tensor
        Chain positions ``[M, C]``, float32 or float64.
    eta_dot : torch.Tensor
        Chain velocities ``[M, C]``, same dtype.
    kinetic_energy : torch.Tensor
        Per-system kinetic energy ``[M]``, same dtype.
    temperature : torch.Tensor
        Per-system temperature in Kelvin ``[M]``, same dtype.
    thermostat_masses : torch.Tensor
        Chain masses ``[M, C]``, same dtype.
    num_atoms_per_system : torch.Tensor
        Number of atoms per system ``[M]``, int32.
    chain_length : int
        Number of links in the chain.
    dt : torch.Tensor
        Per-system timestep ``[M]``, same dtype.
    """
    dtype = eta.dtype
    scl_t = _scalar_type(dtype)
    _npt_thermo_half(
        wp.from_torch(eta, dtype=scl_t),
        wp.from_torch(eta_dot, dtype=scl_t),
        wp.from_torch(kinetic_energy, dtype=scl_t),
        wp.from_torch(temperature.to(dtype), dtype=scl_t),
        wp.from_torch(thermostat_masses, dtype=scl_t),
        wp.from_torch(num_atoms_per_system, dtype=wp.int32),
        chain_length,
        wp.from_torch(dt, dtype=scl_t),
    )


@npt_thermostat_half_step.register_fake
def _npt_thermostat_half_step_fake(
    eta,
    eta_dot,
    kinetic_energy,
    temperature,
    thermostat_masses,
    num_atoms_per_system,
    chain_length,
    dt,
) -> None:
    pass


@torch.library.custom_op(
    "nvalchemi::npt_velocity_half_step", mutates_args={"velocities"}
)
def npt_velocity_half_step(
    velocities: torch.Tensor,
    masses: torch.Tensor,
    forces: torch.Tensor,
    cell_velocity: torch.Tensor,
    volumes: torch.Tensor,
    eta_dots: torch.Tensor,
    num_atoms_per_system: torch.Tensor,
    dt: torch.Tensor,
    batch_idx: torch.Tensor,
    cells_inv: torch.Tensor,
    pressure_mode: str = "isotropic",
) -> None:
    """NPT particle velocity half-step coupled to thermostat and barostat.

    Applies ``v += 0.5*(F/m - (1 + 1/N_f)*ε̇·v - η̇₁·v)*dt``.
    Modifies *velocities* in-place.

    Parameters
    ----------
    velocities : torch.Tensor
        Atomic velocities ``[N, 3]``, float32 or float64.
    masses : torch.Tensor
        Atomic masses ``[N]``, same dtype.
    forces : torch.Tensor
        Atomic forces ``[N, 3]``, same dtype.
    cell_velocity : torch.Tensor
        Per-system cell velocity ḣ ``[M, 3, 3]``, same dtype.
    volumes : torch.Tensor
        Per-system cell volumes ``[M]``, same dtype.
    eta_dots : torch.Tensor
        Full NHC chain velocities ``[M, chain_length]``, same dtype.
        The kernel reads only ``eta_dots[:, 0]`` (first chain link).
    num_atoms_per_system : torch.Tensor
        Number of atoms per system ``[M]``, int32.
    dt : torch.Tensor
        Per-system timestep ``[M]``, same dtype.
    batch_idx : torch.Tensor
        Per-atom system index ``[N]``, int32, non-decreasing.
    cells_inv : torch.Tensor
        Pre-computed inverse cell matrices ``[M, 3, 3]``, same dtype.
    pressure_mode : str, optional
        Pressure control mode: ``"isotropic"``, ``"anisotropic"``,
        or ``"triclinic"``.  Default ``"isotropic"``.
    """
    N = velocities.shape[0]
    dtype = velocities.dtype
    vec_t = _vec_type(dtype)
    mat_t = _mat_type(dtype)
    scl_t = _scalar_type(dtype)
    wp_cell_inv = wp.from_torch(cells_inv.contiguous(), dtype=mat_t)
    _npt_vel_half(
        wp.from_torch(velocities, dtype=vec_t),
        wp.from_torch(masses, dtype=scl_t),
        wp.from_torch(forces, dtype=vec_t),
        wp.from_torch(cell_velocity, dtype=mat_t),
        wp.from_torch(volumes, dtype=scl_t),
        wp.from_torch(eta_dots, dtype=scl_t),
        N,
        wp.from_torch(dt, dtype=scl_t),
        batch_idx=wp.from_torch(batch_idx, dtype=wp.int32),
        num_atoms_per_system=wp.from_torch(num_atoms_per_system, dtype=wp.int32),
        cells_inv=wp_cell_inv,
        mode=pressure_mode,
    )


@npt_velocity_half_step.register_fake
def _npt_velocity_half_step_fake(
    velocities,
    masses,
    forces,
    cell_velocity,
    volumes,
    eta_dots,
    num_atoms_per_system,
    dt,
    batch_idx,
    cells_inv,
    pressure_mode="isotropic",
) -> None:
    pass


@torch.library.custom_op("nvalchemi::npt_position_update", mutates_args={"positions"})
def npt_position_update(
    positions: torch.Tensor,
    velocities: torch.Tensor,
    cell: torch.Tensor,
    cell_velocity: torch.Tensor,
    dt: torch.Tensor,
    cells_inv: torch.Tensor,
    batch_idx: torch.Tensor,
) -> None:
    """Full-step position update including cell strain; shared by NPT/NPH.

    Computes ``r(t+dt) = r(t) + (v + ε̇·r)*dt`` where ε̇ = ḣ·h⁻¹.
    Modifies *positions* in-place.

    Parameters
    ----------
    positions : torch.Tensor
        Atomic positions ``[N, 3]``, float32 or float64.
    velocities : torch.Tensor
        Atomic velocities ``[N, 3]``, same dtype.
    cell : torch.Tensor
        Per-system cell matrix ``[M, 3, 3]``, same dtype.
    cell_velocity : torch.Tensor
        Per-system cell velocity ḣ ``[M, 3, 3]``, same dtype.
    dt : torch.Tensor
        Per-system timestep ``[M]``, same dtype.
    cells_inv : torch.Tensor
        Pre-computed inverse cell matrices ``[M, 3, 3]``, same dtype.
        Used for fractional-coordinate updates.
    batch_idx : torch.Tensor
        Per-atom system index ``[N]``, int32, non-decreasing.
    """
    dtype = positions.dtype
    vec_t = _vec_type(dtype)
    mat_t = _mat_type(dtype)
    scl_t = _scalar_type(dtype)
    _npt_pos_update(
        wp.from_torch(positions, dtype=vec_t),
        wp.from_torch(velocities, dtype=vec_t),
        wp.from_torch(cell, dtype=mat_t),
        wp.from_torch(cell_velocity, dtype=mat_t),
        wp.from_torch(dt, dtype=scl_t),
        wp.from_torch(cells_inv.contiguous(), dtype=mat_t),
        batch_idx=wp.from_torch(batch_idx, dtype=wp.int32),
    )


@npt_position_update.register_fake
def _npt_position_update_fake(
    positions, velocities, cell, cell_velocity, dt, cells_inv, batch_idx
) -> None:
    pass


@torch.library.custom_op("nvalchemi::npt_cell_update", mutates_args={"cell"})
def npt_cell_update(
    cell: torch.Tensor,
    cell_velocity: torch.Tensor,
    dt: torch.Tensor,
) -> None:
    """Full-step cell matrix update: ``h(t+dt) = h(t) + ḣ*dt``.

    Shared by NPT and NPH. Modifies *cell* in-place.

    Parameters
    ----------
    cell : torch.Tensor
        Per-system cell matrix ``[M, 3, 3]``, float32 or float64.
    cell_velocity : torch.Tensor
        Per-system cell velocity ḣ ``[M, 3, 3]``, same dtype.
    dt : torch.Tensor
        Per-system timestep ``[M]``, same dtype.
    """
    dtype = cell.dtype
    scl_t = _scalar_type(dtype)
    mat_t = _mat_type(dtype)
    _npt_cell_update(
        wp.from_torch(cell, dtype=mat_t),
        wp.from_torch(cell_velocity, dtype=mat_t),
        wp.from_torch(dt, dtype=scl_t),
    )


@npt_cell_update.register_fake
def _npt_cell_update_fake(cell, cell_velocity, dt) -> None:
    pass


@torch.library.custom_op("nvalchemi::stress_to_cell_force", mutates_args=())
def stress_to_cell_force(
    stress: torch.Tensor,
    cell: torch.Tensor,
    volume: torch.Tensor,
    keep_aligned: bool = True,
) -> torch.Tensor:
    """Convert stress tensor to cell force: ``F_cell = -V * σ * (h⁻¹)ᵀ``.

    Used by variable-cell FIRE/FIRE2 optimizers to obtain the force on
    the cell degrees of freedom from the model's stress output.

    Parameters
    ----------
    stress : torch.Tensor
        Per-system stress tensor ``[M, 3, 3]``, float32 or float64.
    cell : torch.Tensor
        Per-system cell matrix ``[M, 3, 3]``, same dtype.
    volume : torch.Tensor
        Per-system cell volume ``[M]``, same dtype.
    keep_aligned : bool, optional
        If True, enforce upper-triangular symmetry on the cell force.
        Default True.

    Returns
    -------
    torch.Tensor
        Cell force ``[M, 3, 3]``, same dtype as *stress*.
    """
    dtype = stress.dtype
    mat_t = _mat_type(dtype)
    scl_t = _scalar_type(dtype)
    M = stress.shape[0]
    cell_force = torch.empty(M, 3, 3, dtype=dtype, device=stress.device)
    _stress_to_cell(
        wp.from_torch(stress, dtype=mat_t),
        wp.from_torch(cell, dtype=mat_t),
        wp.from_torch(volume, dtype=scl_t),
        wp.from_torch(cell_force, dtype=mat_t),
        keep_aligned=keep_aligned,
    )
    return cell_force


@stress_to_cell_force.register_fake
def _stress_to_cell_force_fake(stress, cell, volume, keep_aligned=True) -> torch.Tensor:
    return stress.new_empty(stress.shape[0], 3, 3)
