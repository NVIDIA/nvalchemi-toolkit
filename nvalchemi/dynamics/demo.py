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
Demo dynamics implementations for testing and debugging.

This module provides ``DemoDynamics`` — a concrete, minimal implementation
of ``BaseDynamics`` for testing and debugging, analogous to how
``nvalchemi.models.demo`` provides ``DemoModelWrapper`` for models.

``DemoDynamics`` implements a standard Velocity Verlet integrator, which
is a symplectic, time-reversible algorithm commonly used in molecular
dynamics. This serves as a simple example of how to implement an integrator
by overriding the ``pre_update`` and ``post_update`` methods. Inter-rank
communication capabilities are inherited from ``BaseDynamics``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from nvalchemi._typing import Forces, NodePositions, NodeVelocities
from nvalchemi.data import Batch
from nvalchemi.dynamics.base import BaseDynamics, ConvergenceHook

if TYPE_CHECKING:
    from nvalchemi.dynamics.base import Hook
    from nvalchemi.models.base import BaseModelMixin

__all__ = ["DemoDynamics"]


class DemoDynamics(BaseDynamics):
    """
    Velocity Verlet integrator for molecular dynamics simulations.

    Implements the standard Velocity Verlet algorithm, a symplectic,
    time-reversible integration scheme commonly used in molecular dynamics.
    The algorithm splits the integration into two half-steps:

    1. ``pre_update``: Update positions using current velocities and accelerations
       x(t+dt) = x(t) + v(t)*dt + 0.5*a(t)*dt^2

    2. ``post_update``: Update velocities using averaged accelerations
       v(t+dt) = v(t) + 0.5*(a(t) + a(t+dt))*dt

    On the first step when previous accelerations are unavailable, falls back
    to Euler integration for velocities: v(t+dt) = v(t) + a(t+dt)*dt.

    Inter-rank communication capabilities for use in pipeline workflows are
    inherited from ``BaseDynamics``. Communication attributes include
    ``prior_rank``, ``next_rank``, ``sinks``, ``active_batch``,
    ``max_batch_size``, and ``done``.

    This class is intended **entirely** for testing and debugging, demonstrating
    how to implement an integrator by overriding ``pre_update`` and ``post_update``.
    Do **NOT** use this class for production.

    Attributes
    ----------
    __needs_keys__ : set[str]
        Set of output keys required from the model. Set to ``{"forces"}``.
    __provides_keys__ : set[str]
        Set of keys this dynamics produces beyond model outputs.
        Set to ``{"velocities", "positions"}``.
    model : BaseModelMixin
        The neural network potential model.
    dt : float
        The integration timestep.
    step_count : int
        The current step number.
    hooks : dict[HookStageEnum, list[Hook]]
        Registered hooks organized by stage.
    _prev_accelerations : torch.Tensor | None
        Cached accelerations from the previous step for the velocity
        half-step. ``None`` on the first step.
    prior_rank : int | None
        Rank of the previous pipeline stage (inherited from ``BaseDynamics``).
    next_rank : int | None
        Rank of the next pipeline stage (inherited from ``BaseDynamics``).

    Examples
    --------
    >>> model = DemoModelWrapper()
    >>> dynamics = DemoDynamics(model, dt=0.5, n_steps=100)
    >>> dynamics.run(batch)
    >>> # DistributedPipeline composition:
    >>> pipeline = dynamics | other_dynamics
    """

    __needs_keys__: set[str] = {"forces"}
    __provides_keys__: set[str] = {"velocities", "positions"}

    _prev_accelerations: torch.Tensor | None

    def __init__(
        self,
        model: BaseModelMixin,
        n_steps: int,
        dt: float = 1.0,
        hooks: list[Hook] | None = None,
        convergence_hook: ConvergenceHook | dict | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the Velocity Verlet integrator.

        Parameters
        ----------
        model : BaseModelMixin
            The neural network potential model.
        n_steps : int
            Total number of simulation steps for ``run()``.
        dt : float, optional
            The integration timestep in femtoseconds. Default is 1.0.
        hooks : list[Hook] | None, optional
            Initial list of hooks to register. Each hook will be
            organized by its ``stage`` attribute.
        convergence_hook : ConvergenceHook | dict | None, optional
            Convergence hook configuration. Accepts a ``ConvergenceHook``
            instance, a dict (unpacked as ``ConvergenceHook(**dict)``),
            or ``None`` for no convergence checking. Default is ``None``.
        **kwargs : Any
            Additional keyword arguments forwarded to the next class
            in the MRO (for cooperative multiple inheritance).
        """
        super().__init__(
            model=model,
            hooks=hooks,
            convergence_hook=convergence_hook,
            n_steps=n_steps,
            **kwargs,
        )
        self.dt = dt
        self._prev_accelerations = None

    def pre_update(self, batch: Batch) -> None:
        """
        Perform the position update (first half of Velocity Verlet).

        Updates positions according to:
        x(t+dt) = x(t) + v(t)*dt + 0.5*a(t)*dt^2

        where a(t) = F(t) / m.

        If forces are not yet computed (first call before any ``compute``),
        this method falls back to a simple Euler position update:
        x(t+dt) = x(t) + v(t)*dt.

        The update is performed inside a ``torch.no_grad()`` context to
        avoid conflicts with autograd when ``forces_via_autograd=True``
        (which causes ``compute()`` to set ``requires_grad_(True)`` on
        positions).

        Parameters
        ----------
        batch : Batch
            The current batch of atomic data. Positions are modified in-place.
        """
        positions: NodePositions = batch.positions
        velocities: NodeVelocities = batch.velocities
        forces: Forces | None = batch.forces
        masses = batch.atomic_masses.unsqueeze(-1)  # (V,) -> (V, 1) for broadcasting

        dt = self.dt

        with torch.no_grad():
            if forces is not None and not torch.all(forces == 0):
                # Compute current accelerations: a = F / m
                accelerations = forces / masses

                # Store for velocity half-step in post_update
                self._prev_accelerations = accelerations.clone()

                # x(t+dt) = x(t) + v(t)*dt + 0.5*a(t)*dt^2
                positions.add_(velocities * dt + 0.5 * accelerations * dt * dt)
            else:
                # No forces computed yet — simple Euler position update
                # x(t+dt) = x(t) + v(t)*dt
                positions.add_(velocities * dt)

    def post_update(self, batch: Batch) -> None:
        """
        Perform the velocity update (second half of Velocity Verlet).

        Updates velocities according to:
        v(t+dt) = v(t) + 0.5*(a(t) + a(t+dt))*dt

        where a(t+dt) = F(t+dt) / m are the forces computed after the
        position update.

        If previous accelerations are unavailable (first step), falls back
        to Euler velocity update: v(t+dt) = v(t) + a(t+dt)*dt.

        The update is performed inside a ``torch.no_grad()`` context to
        avoid conflicts with autograd when ``forces_via_autograd=True``.

        Parameters
        ----------
        batch : Batch
            The current batch of atomic data. Velocities are modified in-place.
        """
        velocities: NodeVelocities = batch.velocities
        forces: Forces = batch.forces
        masses = batch.atomic_masses.unsqueeze(-1)  # (V,) -> (V, 1) for broadcasting

        dt = self.dt

        with torch.no_grad():
            # Compute new accelerations from updated forces: a(t+dt) = F(t+dt) / m
            new_accelerations = forces / masses

            if self._prev_accelerations is not None:
                # Full Velocity Verlet: v(t+dt) = v(t) + 0.5*(a(t) + a(t+dt))*dt
                velocities.add_(
                    0.5 * (self._prev_accelerations + new_accelerations) * dt,
                )
            else:
                # First step fallback — Euler: v(t+dt) = v(t) + a(t+dt)*dt
                velocities.add_(new_accelerations * dt)
