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
NPT (isothermal-isobaric) integrator.

Constant temperature and pressure; uses Martyna-Tobias-Klein (MTK)
barostat equations with two Nosé-Hoover chains: one coupled to
particle DOFs and one coupled to cell/barostat DOFs.

The step is split around the force/stress evaluation:

* ``pre_update``:  NHC-p half → NHC-b half → baro half → v half
                   → r full → cell full
* [model evaluates F and stress at r(t+dt), h(t+dt)]
* ``post_update``: v half → baro half → NHC-b half → NHC-p half

Per-system state: ``dt``, ``temperature``, ``pressure``,
``barostat_time``, ``thermostat_time``, barostat inertia ``W``,
``cell_velocity [M,3,3]``, particle NHC state
``nhc_eta [M,C]``, ``nhc_eta_dot [M,C]``, ``nhc_Q [M,C]``,
barostat NHC state ``nhc_b_eta [M,C]``, ``nhc_b_eta_dot [M,C]``,
``nhc_b_Q [M,C]``, and pre-allocated scratch tensors.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import torch

from nvalchemi.data import Batch
from nvalchemi.dynamics._ops._bridge import _make_state_batch, _to_per_system
from nvalchemi.dynamics._ops.nose_hoover import nhc_compute_masses
from nvalchemi.dynamics._ops.npt_nph import (
    compute_barostat_mass,
    compute_pressure_tensor,
    npt_barostat_half_step,
    npt_cell_update,
    npt_position_update,
    npt_thermostat_half_step,
    npt_velocity_half_step,
)
from nvalchemi.dynamics._ops.thermostat_utils import compute_kinetic_energy
from nvalchemi.dynamics.base import BaseDynamics
from nvalchemi.dynamics.hooks._utils import KB_EV

if TYPE_CHECKING:
    from nvalchemi.dynamics.base import ConvergenceHook
    from nvalchemi.hooks import Hook
    from nvalchemi.models.base import BaseModelMixin

__all__ = ["NPT"]


class NPT(BaseDynamics):
    """Isothermal-isobaric (NPT) integrator via MTK barostat and NHC thermostat.

    Samples the NPT ensemble.  Two Nosé-Hoover chains control the
    temperature: one coupled to the particle velocities and one coupled
    to the barostat (cell) degrees of freedom.

    Parameters
    ----------
    model : BaseModelMixin
        The neural network potential model.  Must produce ``"stresses"``
        output in addition to forces.
    dt : float or torch.Tensor
        Integration timestep ``[M]`` or scalar.
    temperature : float or torch.Tensor
        Target temperature in Kelvin ``[M]`` or scalar.
    pressure : float or torch.Tensor
        Target pressure ``[M]`` (isotropic), ``[M, 3]`` (anisotropic),
        or ``[M, 3, 3]`` (triclinic).  Scalar is broadcast to ``[M]``
        isotropic.
    barostat_time : float or torch.Tensor
        Barostat coupling time τ_P ``[M]`` or scalar.
    thermostat_time : float or torch.Tensor
        Thermostat coupling time τ_T ``[M]`` or scalar.
    pressure_coupling : {"isotropic", "anisotropic", "triclinic"}
        Pressure control mode.  Default ``"isotropic"``.
    chain_length : int, optional
        Number of links in each Nosé-Hoover chain.  Default 3.
    n_steps : int, optional
        Total steps for :meth:`run`.
    hooks : list[Hook], optional
        Initial hooks.
    convergence_hook : ConvergenceHook or dict, optional
        Convergence criterion.
    **kwargs
        Forwarded to :class:`~nvalchemi.dynamics.base.BaseDynamics`.

    Attributes
    ----------
    __needs_keys__ : set[str]
        ``{"forces", "stresses"}``.
    __provides_keys__ : set[str]
        ``{"positions", "velocities", "cell"}``.
    """

    __needs_keys__: set[str] = {"forces", "stresses"}
    __provides_keys__: set[str] = {"positions", "velocities", "cell"}

    def __init__(
        self,
        model: BaseModelMixin,
        dt: float | torch.Tensor,
        temperature: float | torch.Tensor,
        pressure: float | torch.Tensor,
        barostat_time: float | torch.Tensor,
        thermostat_time: float | torch.Tensor,
        pressure_coupling: Literal[
            "isotropic", "anisotropic", "triclinic"
        ] = "isotropic",
        chain_length: int = 3,
        n_steps: int | None = None,
        hooks: list[Hook] | None = None,
        convergence_hook: ConvergenceHook | dict | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            model=model,
            n_steps=n_steps,
            hooks=hooks,
            convergence_hook=convergence_hook,
            **kwargs,
        )
        self._dt_init = dt
        self._temperature_init = temperature
        self._pressure_init = pressure
        self._barostat_time_init = barostat_time
        self._thermostat_time_init = thermostat_time
        self.pressure_coupling = pressure_coupling
        self.chain_length = chain_length

    def _init_state(self, batch: Batch) -> None:
        M = batch.num_graphs
        dev = batch.device
        dtype = batch.positions.dtype
        dt = _to_per_system(self._dt_init, M, dev, dtype)
        # All NHC/barostat kernels expect kT in energy units (eV), not T in Kelvin.
        kT = _to_per_system(self._temperature_init * KB_EV, M, dev, dtype)
        pressure = _to_per_system(self._pressure_init, M, dev, dtype)
        tau_p = _to_per_system(self._barostat_time_init, M, dev, dtype)
        tau_t = _to_per_system(self._thermostat_time_init, M, dev, dtype)
        counts = torch.bincount(batch.batch, minlength=M)
        num_atoms_per_system = counts.to(dtype=torch.int32, device=dev)
        W = torch.zeros(M, dtype=dtype, device=dev)
        compute_barostat_mass(kT, tau_p, num_atoms_per_system, W)
        Q = nhc_compute_masses(
            kT, tau_t, batch.atomic_masses, batch.batch.int(), self.chain_length
        )
        # Barostat NHC: 3 dummy atoms per system → N_f = 9 DOFs (3×3 cell).
        dummy_b_masses = torch.ones(M * 3, dtype=dtype, device=dev)
        dummy_b_batch = torch.arange(
            M, device=dev, dtype=torch.int32
        ).repeat_interleave(3)
        Q_b = nhc_compute_masses(
            kT, tau_t, dummy_b_masses, dummy_b_batch, self.chain_length
        )
        self._state = _make_state_batch(
            {
                "dt": dt,
                "temperature": kT,
                "pressure": pressure,
                "barostat_time": tau_p,
                "thermostat_time": tau_t,
                "W": W,
                "cell_velocity": torch.zeros(M, 3, 3, dtype=dtype, device=dev),
                "num_atoms_per_system": num_atoms_per_system,
                "nhc_eta": torch.zeros(M, self.chain_length, dtype=dtype, device=dev),
                "nhc_eta_dot": torch.zeros(
                    M, self.chain_length, dtype=dtype, device=dev
                ),
                "nhc_Q": Q,
                "nhc_b_eta": torch.zeros(M, self.chain_length, dtype=dtype, device=dev),
                "nhc_b_eta_dot": torch.zeros(
                    M, self.chain_length, dtype=dtype, device=dev
                ),
                "nhc_b_Q": Q_b,
                # Pre-allocated scratch tensors; zeroed by kernel each call.
                "kinetic_tensors": torch.zeros(M, 9, dtype=dtype, device=dev),
                "pressure_tensors": torch.zeros(M, 9, dtype=dtype, device=dev),
                "volumes": torch.zeros(M, dtype=dtype, device=dev),
            },
            dev,
        )

    def _make_new_state(self, n: int, template_batch: Batch) -> Batch:
        dev = template_batch.device
        dtype = template_batch.positions.dtype
        kT = _to_per_system(self._temperature_init * KB_EV, n, dev, dtype)
        tau_p = _to_per_system(self._barostat_time_init, n, dev, dtype)
        tau_t = _to_per_system(self._thermostat_time_init, n, dev, dtype)
        approx_n_atoms = template_batch.num_nodes // template_batch.num_graphs
        num_atoms_per_system = torch.full(
            (n,), approx_n_atoms, dtype=torch.int32, device=dev
        )
        dummy_masses = template_batch.atomic_masses[:1].expand(n).contiguous()
        dummy_batch_idx = torch.zeros(n, dtype=torch.int32, device=dev)
        W = torch.zeros(n, dtype=dtype, device=dev)
        compute_barostat_mass(kT, tau_p, num_atoms_per_system, W)
        Q = nhc_compute_masses(
            kT[:1],
            tau_t[:1],
            dummy_masses[:1],
            dummy_batch_idx[:1],
            self.chain_length,
        )
        Q = Q.expand(n, -1).contiguous()
        dummy_b_masses = torch.ones(3, dtype=dtype, device=dev)
        dummy_b_batch = torch.zeros(3, dtype=torch.int32, device=dev)
        Q_b_single = nhc_compute_masses(
            kT[:1], tau_t[:1], dummy_b_masses, dummy_b_batch, self.chain_length
        )
        Q_b = Q_b_single.expand(n, -1).contiguous()
        return _make_state_batch(
            {
                "dt": _to_per_system(self._dt_init, n, dev, dtype),
                "temperature": kT,
                "pressure": _to_per_system(self._pressure_init, n, dev, dtype),
                "barostat_time": tau_p,
                "thermostat_time": tau_t,
                "W": W,
                "cell_velocity": torch.zeros(n, 3, 3, dtype=dtype, device=dev),
                "num_atoms_per_system": num_atoms_per_system,
                "nhc_eta": torch.zeros(n, self.chain_length, dtype=dtype, device=dev),
                "nhc_eta_dot": torch.zeros(
                    n, self.chain_length, dtype=dtype, device=dev
                ),
                "nhc_Q": Q,
                "nhc_b_eta": torch.zeros(n, self.chain_length, dtype=dtype, device=dev),
                "nhc_b_eta_dot": torch.zeros(
                    n, self.chain_length, dtype=dtype, device=dev
                ),
                "nhc_b_Q": Q_b,
                "kinetic_tensors": torch.zeros(n, 9, dtype=dtype, device=dev),
                "pressure_tensors": torch.zeros(n, 9, dtype=dtype, device=dev),
                "volumes": torch.zeros(n, dtype=dtype, device=dev),
            },
            dev,
        )

    def _compute_volumes(self, batch: Batch) -> torch.Tensor:
        """Compute per-system cell volumes as |det(h)|."""
        return torch.linalg.det(batch.cell).abs()

    def _compute_P(self, batch: Batch, volumes: torch.Tensor) -> torch.Tensor:
        """Compute the instantaneous pressure tensor."""
        return compute_pressure_tensor(
            batch.velocities,
            batch.atomic_masses,
            batch.stresses,
            batch.cell,
            self._state.kinetic_tensors,
            self._state.pressure_tensors,
            volumes,
            batch.batch.int(),
        )

    def _compute_ke(self, batch: Batch) -> torch.Tensor:
        """Compute per-system kinetic energy."""
        M = batch.num_graphs
        return compute_kinetic_energy(
            batch.velocities,
            batch.atomic_masses,
            batch.batch.int(),
            M,
        )

    def pre_update(self, batch: Batch) -> None:
        """NHC-p half → NHC-b half → baro half → v half → r full → cell full.

        Parameters
        ----------
        batch : Batch
            Current batch; *positions*, *velocities*, and *cell*
            updated in-place.
        """
        M = batch.num_graphs
        volumes = self._compute_volumes(batch)
        cells_inv = torch.linalg.inv(batch.cell)
        KE = self._compute_ke(batch)

        # Particle thermostat half step.
        npt_thermostat_half_step(
            self._state.nhc_eta,
            self._state.nhc_eta_dot,
            KE,
            self._state.temperature,
            self._state.nhc_Q,
            self._state.num_atoms_per_system,
            self.chain_length,
            self._state.dt,
        )
        # Scale particle velocities by exp(-eta_dot[:,0] * dt/2).
        scale = torch.exp(-self._state.nhc_eta_dot[:, 0] * self._state.dt * 0.5)
        batch.velocities.mul_(scale[batch.batch].unsqueeze(-1))

        # Barostat thermostat half step (acts on cell DOFs; KE_cell ≈ 0.5*W*Tr(h_dot^T*h_dot)).
        # Compute approximate cell kinetic energy from cell_velocity.
        cell_v_flat = self._state.cell_velocity.reshape(M, -1)
        ke_cell = 0.5 * self._state.W * (cell_v_flat * cell_v_flat).sum(dim=-1)
        # cell DOFs = 9 (3x3 matrix), represented as 3 pseudo-atoms with 3 DOFs each.
        cell_ndof_tensor = torch.full((M,), 9, dtype=torch.int32, device=batch.device)
        npt_thermostat_half_step(
            self._state.nhc_b_eta,
            self._state.nhc_b_eta_dot,
            ke_cell,
            self._state.temperature,
            self._state.nhc_b_Q,
            cell_ndof_tensor,
            self.chain_length,
            self._state.dt,
        )
        # Scale cell velocity by exp(-b_eta_dot[:,0] * dt/2).
        b_scale = torch.exp(-self._state.nhc_b_eta_dot[:, 0] * self._state.dt * 0.5)
        self._state.cell_velocity.mul_(b_scale.view(M, 1, 1))

        P_inst = self._compute_P(batch, volumes)
        KE = self._compute_ke(batch)
        npt_barostat_half_step(
            self._state.cell_velocity,
            P_inst,
            self._state.pressure,
            volumes,
            self._state.W,
            KE,
            self._state.num_atoms_per_system,
            self._state.nhc_eta_dot,
            self._state.dt,
        )
        npt_velocity_half_step(
            batch.velocities,
            batch.atomic_masses,
            batch.forces,
            self._state.cell_velocity,
            volumes,
            self._state.nhc_eta_dot,
            self._state.num_atoms_per_system,
            self._state.dt,
            batch.batch.int(),
            cells_inv,
        )
        npt_position_update(
            batch.positions,
            batch.velocities,
            batch.cell,
            self._state.cell_velocity,
            self._state.dt,
            cells_inv,
            batch.batch.int(),
        )
        npt_cell_update(
            batch.cell,
            self._state.cell_velocity,
            self._state.dt,
        )

    def post_update(self, batch: Batch) -> None:
        """v half → baro half → NHC-b half → NHC-p half (symmetric closure).

        Parameters
        ----------
        batch : Batch
            Current batch; *velocities* updated in-place.
        """
        M = batch.num_graphs
        volumes = self._compute_volumes(batch)
        cells_inv = torch.linalg.inv_ex(batch.cell)[0].contiguous()
        KE = self._compute_ke(batch)

        npt_velocity_half_step(
            batch.velocities,
            batch.atomic_masses,
            batch.forces,
            self._state.cell_velocity,
            volumes,
            self._state.nhc_eta_dot,
            self._state.num_atoms_per_system,
            self._state.dt,
            batch.batch.int(),
            cells_inv,
        )
        P_inst = self._compute_P(batch, volumes)
        KE = self._compute_ke(batch)
        npt_barostat_half_step(
            self._state.cell_velocity,
            P_inst,
            self._state.pressure,
            volumes,
            self._state.W,
            KE,
            self._state.num_atoms_per_system,
            self._state.nhc_eta_dot,
            self._state.dt,
        )

        # Barostat thermostat half step.
        cell_v_flat = self._state.cell_velocity.reshape(M, -1)
        ke_cell = 0.5 * self._state.W * (cell_v_flat * cell_v_flat).sum(dim=-1)
        cell_ndof_tensor = torch.full((M,), 9, dtype=torch.int32, device=batch.device)
        npt_thermostat_half_step(
            self._state.nhc_b_eta,
            self._state.nhc_b_eta_dot,
            ke_cell,
            self._state.temperature,
            self._state.nhc_b_Q,
            cell_ndof_tensor,
            self.chain_length,
            self._state.dt,
        )
        b_scale = torch.exp(-self._state.nhc_b_eta_dot[:, 0] * self._state.dt * 0.5)
        self._state.cell_velocity.mul_(b_scale.view(M, 1, 1))

        # Particle thermostat half step.
        KE = self._compute_ke(batch)
        npt_thermostat_half_step(
            self._state.nhc_eta,
            self._state.nhc_eta_dot,
            KE,
            self._state.temperature,
            self._state.nhc_Q,
            self._state.num_atoms_per_system,
            self.chain_length,
            self._state.dt,
        )
        scale = torch.exp(-self._state.nhc_eta_dot[:, 0] * self._state.dt * 0.5)
        batch.velocities.mul_(scale[batch.batch].unsqueeze(-1))
