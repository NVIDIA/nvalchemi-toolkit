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
NVT integrator via Nosé-Hoover chain (NHC) thermostat.

The NHC thermostat is deterministic, time-reversible, and ergodic.
It uses Yoshida-Suzuki factorization with the Martyna-Tobias-Klein
(MTK) equations.

The step is split around the force evaluation:

* ``pre_update``:  NHC half → v half kick → r full step
* [model evaluates F(r(t+dt))]
* ``post_update``: v half kick → NHC half

Per-system state: ``dt``, ``temperature``, ``thermostat_time``,
chain positions ``nhc_eta [M, C]``, chain velocities ``nhc_eta_dot [M, C]``,
chain masses ``nhc_Q [M, C]``, and NHC scratch tensors
``nhc_ke2 [M]``, ``nhc_total_scale [M]``, ``nhc_step_scale [M]``,
``nhc_dt_chain [M]``.

References
----------
Martyna, Tobias, Klein (1994); Tuckerman et al. (2006).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import torch

from nvalchemi.data import Batch
from nvalchemi.dynamics._ops._bridge import _make_state_batch, _to_per_system
from nvalchemi.dynamics._ops.nose_hoover import (
    nhc_chain_update,
    nhc_compute_masses,
    nhc_position_update,
    nhc_velocity_half_step,
)
from nvalchemi.dynamics._units import fs_to_internal_time
from nvalchemi.dynamics.base import BaseDynamics
from nvalchemi.dynamics.hooks._utils import KB_EV

if TYPE_CHECKING:
    from nvalchemi.dynamics.base import ConvergenceHook
    from nvalchemi.hooks import Hook
    from nvalchemi.models.base import BaseModelMixin

__all__ = ["NVTNoseHoover"]


class NVTNoseHoover(BaseDynamics):
    """NVT integrator via Nosé-Hoover chain (NHC) thermostat.

    Deterministic extended-Lagrangian thermostat that rigorously samples
    the canonical ensemble for ergodic systems.

    Parameters
    ----------
    model : BaseModelMixin
        The neural network potential model.
    dt : float or torch.Tensor
        Integration timestep in femtoseconds ``[M]`` or scalar.
    temperature : float or torch.Tensor
        Target temperature in Kelvin ``[M]`` or scalar.
    thermostat_time : float or torch.Tensor
        Thermostat coupling time τ_T in femtoseconds ``[M]`` or scalar.
        Controls how tightly the thermostat couples to the system;
        typical values are 10–100 × dt.
    chain_length : int, optional
        Number of links in the Nosé-Hoover chain.  Default 3.
    yoshida_order : int, optional
        Order of the Yoshida-Suzuki integrator.  Must be 3 or 5.
        Default 3.
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
        ``{"forces"}``.
    __provides_keys__ : set[str]
        ``{"positions", "velocities"}``.
    """

    __needs_keys__: set[str] = {"forces"}
    __provides_keys__: set[str] = {"positions", "velocities"}

    def __init__(
        self,
        model: BaseModelMixin,
        dt: float | torch.Tensor,
        temperature: float | torch.Tensor,
        thermostat_time: float | torch.Tensor,
        chain_length: int = 3,
        yoshida_order: Literal[3, 5] = 3,
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
        self._dt_init = fs_to_internal_time(dt)
        self._temperature_init = temperature
        self._thermostat_time_init = fs_to_internal_time(thermostat_time)
        self.chain_length = chain_length
        self.yoshida_order = yoshida_order

    def _init_state(self, batch: Batch) -> None:
        M = batch.num_graphs
        dev = batch.device
        dtype = batch.positions.dtype
        dt = _to_per_system(self._dt_init, M, dev, dtype)
        # NHC kernels expect kT in energy units (eV), not T in Kelvin.
        kT = _to_per_system(self._temperature_init * KB_EV, M, dev, dtype)
        tau = _to_per_system(self._thermostat_time_init, M, dev, dtype)
        # Compute chain masses using the actual per-atom masses and batch index.
        Q = nhc_compute_masses(
            kT, tau, batch.atomic_masses, batch.batch_idx.int(), self.chain_length
        )
        # Compute per-system ndof as a float tensor (required by nhc_chain_update).
        counts = torch.bincount(batch.batch_idx, minlength=M)
        nhc_ndof = (counts * 3).to(dtype=dtype, device=dev)
        self._state = _make_state_batch(
            {
                "dt": dt,
                "temperature": kT,
                "thermostat_time": tau,
                "nhc_ndof": nhc_ndof,
                "nhc_eta": torch.zeros(M, self.chain_length, dtype=dtype, device=dev),
                "nhc_eta_dot": torch.zeros(
                    M, self.chain_length, dtype=dtype, device=dev
                ),
                "nhc_Q": Q,
                # Scratch tensors for NHC chain update.
                "nhc_ke2": torch.zeros(M, dtype=dtype, device=dev),
                "nhc_total_scale": torch.zeros(M, dtype=dtype, device=dev),
                "nhc_step_scale": torch.zeros(M, dtype=dtype, device=dev),
                "nhc_dt_chain": torch.zeros(M, dtype=dtype, device=dev),
            },
            dev,
        )

    def _make_new_state(self, n: int, template_batch: Batch) -> Batch:
        dev = template_batch.device
        dtype = template_batch.positions.dtype
        kT = _to_per_system(self._temperature_init * KB_EV, n, dev, dtype)
        tau = _to_per_system(self._thermostat_time_init, n, dev, dtype)
        # Approximate Q with a reasonable default using a dummy batch.
        dummy_masses = (
            template_batch.atomic_masses[:n].contiguous()
            if template_batch.atomic_masses.shape[0] >= n
            else template_batch.atomic_masses
        )
        dummy_batch_idx = torch.zeros(
            dummy_masses.shape[0], dtype=torch.int32, device=dev
        )
        Q = nhc_compute_masses(
            kT[:1], tau[:1], dummy_masses, dummy_batch_idx, self.chain_length
        )
        Q = Q.expand(n, -1).contiguous()
        approx_n_atoms = template_batch.num_nodes // template_batch.num_graphs
        nhc_ndof = torch.full((n,), approx_n_atoms * 3, dtype=dtype, device=dev)
        return _make_state_batch(
            {
                "dt": _to_per_system(self._dt_init, n, dev, dtype),
                "temperature": kT,
                "thermostat_time": tau,
                "nhc_ndof": nhc_ndof,
                "nhc_eta": torch.zeros(n, self.chain_length, dtype=dtype, device=dev),
                "nhc_eta_dot": torch.zeros(
                    n, self.chain_length, dtype=dtype, device=dev
                ),
                "nhc_Q": Q,
                "nhc_ke2": torch.zeros(n, dtype=dtype, device=dev),
                "nhc_total_scale": torch.zeros(n, dtype=dtype, device=dev),
                "nhc_step_scale": torch.zeros(n, dtype=dtype, device=dev),
                "nhc_dt_chain": torch.zeros(n, dtype=dtype, device=dev),
            },
            dev,
        )

    def pre_update(self, batch: Batch) -> None:
        """NHC half-step → velocity half-kick → position full-step.

        Parameters
        ----------
        batch : Batch
            Current batch; *positions* and *velocities* updated in-place.
        """
        nhc_chain_update(
            batch.velocities,
            batch.atomic_masses,
            self._state.nhc_eta,
            self._state.nhc_eta_dot,
            self._state.nhc_Q,
            self._state.temperature,
            self._state.dt,
            self._state.nhc_ndof,
            self._state.nhc_ke2,
            self._state.nhc_total_scale,
            self._state.nhc_step_scale,
            self._state.nhc_dt_chain,
            batch.batch_idx.int(),
        )
        nhc_velocity_half_step(
            batch.velocities,
            batch.forces,
            batch.atomic_masses,
            self._state.dt,
            batch.batch_idx.int(),
        )
        nhc_position_update(
            batch.positions,
            batch.velocities,
            self._state.dt,
            batch.batch_idx.int(),
        )

    def post_update(self, batch: Batch) -> None:
        """Velocity half-kick → NHC half-step (symmetric closure).

        Parameters
        ----------
        batch : Batch
            Current batch; *velocities* updated in-place.
        """
        nhc_velocity_half_step(
            batch.velocities,
            batch.forces,
            batch.atomic_masses,
            self._state.dt,
            batch.batch_idx.int(),
        )
        nhc_chain_update(
            batch.velocities,
            batch.atomic_masses,
            self._state.nhc_eta,
            self._state.nhc_eta_dot,
            self._state.nhc_Q,
            self._state.temperature,
            self._state.dt,
            self._state.nhc_ndof,
            self._state.nhc_ke2,
            self._state.nhc_total_scale,
            self._state.nhc_step_scale,
            self._state.nhc_dt_chain,
            batch.batch_idx.int(),
        )
