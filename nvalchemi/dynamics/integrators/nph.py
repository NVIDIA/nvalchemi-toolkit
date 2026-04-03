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
NPH (isenthalpic-isobaric) integrator.

Constant enthalpy and pressure; temperature fluctuates.  No thermostat.
Uses Martyna-Tobias-Klein (MTK) barostat equations.

The step is split around the force/stress evaluation:

* ``pre_update``:   compute P → baro half → v half → r full → cell full
* [model evaluates F and stress at r(t+dt), h(t+dt)]
* ``post_update``:  v half → compute P → baro half

Per-system state: ``dt``, ``pressure``, barostat inertia ``W``,
cell velocity ``cell_velocity [M,3,3]``, and pre-allocated scratch tensors
``kinetic_tensors``, ``pressure_tensors``, ``volumes``, ``cells_inv``,
``scalar_pressures``, ``kinetic_energy``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import torch

from nvalchemi.data import Batch
from nvalchemi.dynamics._ops._bridge import _make_state_batch, _to_per_system
from nvalchemi.dynamics._ops.npt_nph import (
    compute_barostat_mass,
    compute_pressure_tensor,
    nph_barostat_half_step,
    nph_velocity_half_step,
    npt_cell_update,
    npt_position_update,
)
from nvalchemi.dynamics._ops.thermostat_utils import compute_kinetic_energy
from nvalchemi.dynamics.base import BaseDynamics
from nvalchemi.dynamics.hooks._utils import KB_EV

if TYPE_CHECKING:
    from nvalchemi.dynamics.base import ConvergenceHook
    from nvalchemi.hooks import Hook
    from nvalchemi.models.base import BaseModelMixin

__all__ = ["NPH"]


class NPH(BaseDynamics):
    """Isenthalpic-isobaric (NPH) integrator via MTK barostat.

    Temperature fluctuates; enthalpy H = E + PV is conserved.

    Parameters
    ----------
    model : BaseModelMixin
        The neural network potential model.  Must produce ``"stresses"``
        output in addition to forces.
    dt : float or torch.Tensor
        Integration timestep ``[M]`` or scalar.
    pressure : float or torch.Tensor
        Target pressure ``[M]`` (isotropic), ``[M, 3]`` (anisotropic),
        or ``[M, 3, 3]`` (triclinic).  Scalar is broadcast to ``[M]``
        isotropic.
    barostat_time : float or torch.Tensor
        Barostat coupling time τ_P ``[M]`` or scalar.
    pressure_coupling : {"isotropic", "anisotropic", "triclinic"}
        Pressure control mode.  Default ``"isotropic"``.
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
        pressure: float | torch.Tensor,
        barostat_time: float | torch.Tensor,
        pressure_coupling: Literal[
            "isotropic", "anisotropic", "triclinic"
        ] = "isotropic",
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
        self._pressure_init = pressure
        self._barostat_time_init = barostat_time
        self.pressure_coupling = pressure_coupling

    def _init_state(self, batch: Batch) -> None:
        M = batch.num_graphs
        dev = batch.device
        dtype = batch.positions.dtype
        dt = _to_per_system(self._dt_init, M, dev, dtype)
        pressure = _to_per_system(self._pressure_init, M, dev, dtype)
        barostat_time = _to_per_system(self._barostat_time_init, M, dev, dtype)
        counts = torch.bincount(batch.batch, minlength=M)
        num_atoms_per_system = counts.to(dtype=torch.int32, device=dev)
        # Use a representative kT estimate for W; NPH temperature is not
        # controlled, so we use 300 K → kT as a sensible default.
        kT_est = torch.full((M,), 300.0 * KB_EV, dtype=dtype, device=dev)
        W = torch.zeros(M, dtype=dtype, device=dev)
        compute_barostat_mass(kT_est, barostat_time, num_atoms_per_system, W)
        self._state = _make_state_batch(
            {
                "dt": dt,
                "pressure": pressure,
                "barostat_time": barostat_time,
                "W": W,
                "cell_velocity": torch.zeros(M, 3, 3, dtype=dtype, device=dev),
                "num_atoms_per_system": num_atoms_per_system,
                # Pre-allocated scratch tensors; zeroed by the kernel each call.
                "kinetic_tensors": torch.zeros(M, 9, dtype=dtype, device=dev),
                "pressure_tensors": torch.zeros(M, 9, dtype=dtype, device=dev),
                "volumes": torch.zeros(M, dtype=dtype, device=dev),
                "kinetic_energy": torch.zeros(M, dtype=dtype, device=dev),
            },
            dev,
        )

    def _make_new_state(self, n: int, template_batch: Batch) -> Batch:
        dev = template_batch.device
        dtype = template_batch.positions.dtype
        barostat_time = _to_per_system(self._barostat_time_init, n, dev, dtype)
        kT_est = torch.full((n,), 300.0 * KB_EV, dtype=dtype, device=dev)
        # Approximate atom count from template.
        approx_n_atoms = template_batch.num_nodes // template_batch.num_graphs
        num_atoms_per_system = torch.full(
            (n,), approx_n_atoms, dtype=torch.int32, device=dev
        )
        W = torch.zeros(n, dtype=dtype, device=dev)
        compute_barostat_mass(kT_est, barostat_time, num_atoms_per_system, W)
        return _make_state_batch(
            {
                "dt": _to_per_system(self._dt_init, n, dev, dtype),
                "pressure": _to_per_system(self._pressure_init, n, dev, dtype),
                "barostat_time": barostat_time,
                "W": W,
                "cell_velocity": torch.zeros(n, 3, 3, dtype=dtype, device=dev),
                "num_atoms_per_system": num_atoms_per_system,
                "kinetic_tensors": torch.zeros(n, 9, dtype=dtype, device=dev),
                "pressure_tensors": torch.zeros(n, 9, dtype=dtype, device=dev),
                "volumes": torch.zeros(n, dtype=dtype, device=dev),
                "kinetic_energy": torch.zeros(n, dtype=dtype, device=dev),
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
        """Barostat half → velocity half → position full → cell full.

        Parameters
        ----------
        batch : Batch
            Current batch; *positions*, *velocities*, and *cell*
            updated in-place.
        """
        volumes = self._compute_volumes(batch)
        # Compute cells_inv for velocity and position updates.
        cells_inv = torch.linalg.inv_ex(batch.cell)[0].contiguous()
        KE = self._compute_ke(batch)
        P_inst = self._compute_P(batch, volumes)
        nph_barostat_half_step(
            self._state.cell_velocity,
            P_inst,
            self._state.pressure,
            volumes,
            self._state.W,
            KE,
            self._state.num_atoms_per_system,
            self._state.dt,
        )
        nph_velocity_half_step(
            batch.velocities,
            batch.atomic_masses,
            batch.forces,
            self._state.cell_velocity,
            volumes,
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
        """Velocity half → barostat half (symmetric closure).

        Parameters
        ----------
        batch : Batch
            Current batch; *velocities* updated in-place.
        """
        volumes = self._compute_volumes(batch)
        cells_inv = torch.linalg.inv_ex(batch.cell)[0].contiguous()
        KE = self._compute_ke(batch)
        nph_velocity_half_step(
            batch.velocities,
            batch.atomic_masses,
            batch.forces,
            self._state.cell_velocity,
            volumes,
            self._state.num_atoms_per_system,
            self._state.dt,
            batch.batch.int(),
            cells_inv,
        )
        P_inst = self._compute_P(batch, volumes)
        KE = self._compute_ke(batch)
        nph_barostat_half_step(
            self._state.cell_velocity,
            P_inst,
            self._state.pressure,
            volumes,
            self._state.W,
            KE,
            self._state.num_atoms_per_system,
            self._state.dt,
        )
