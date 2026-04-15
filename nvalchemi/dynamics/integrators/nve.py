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
NVE (microcanonical) integrator via velocity Verlet.

The velocity Verlet algorithm is symplectic and time-reversible,
conserving the total energy (H = KE + PE) over long simulations to
within integration error.

The step is split around the force evaluation:

* ``pre_update``:   r(t+dt) = r(t) + v(t)·dt + ½·(F/m)·dt²
                    v(t+dt/2) = v(t) + ½·(F/m)·dt
* [model evaluates F(r(t+dt))]
* ``post_update``:  v(t+dt) = v(t+dt/2) + ½·(F_new/m)·dt

Per-system state consists solely of the timestep ``dt``, stored in
``self._state`` as a system-level :class:`~nvalchemi.data.Batch`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from nvalchemi.data import Batch
from nvalchemi.dynamics._ops._bridge import _make_state_batch, _to_per_system
from nvalchemi.dynamics._ops.velocity_verlet import (
    vv_position_update,
    vv_velocity_finalize,
)
from nvalchemi.dynamics._units import fs_to_internal_time
from nvalchemi.dynamics.base import BaseDynamics

if TYPE_CHECKING:
    from nvalchemi.dynamics.base import ConvergenceHook
    from nvalchemi.hooks import Hook
    from nvalchemi.models.base import BaseModelMixin

__all__ = ["NVE"]


class NVE(BaseDynamics):
    """Microcanonical (NVE) integrator via velocity Verlet.

    Conserves the total energy E = KE + PE.  Best choice for validating
    model energy conservation and for generating reference trajectories.

    Parameters
    ----------
    model : BaseModelMixin
        The neural network potential model.
    dt : float or torch.Tensor
        Integration timestep in femtoseconds. A scalar is broadcast to all systems;
        a tensor of shape ``[M]`` sets per-system timesteps (useful for
        heterogeneous batches or adaptive stepping via hooks).
    n_steps : int, optional
        Total steps for :meth:`run`.
    hooks : list[Hook], optional
        Initial hooks to register.
    convergence_hook : ConvergenceHook or dict, optional
        Convergence criterion.  For pure NVE trajectories this is
        rarely needed.
    **kwargs
        Forwarded to :class:`~nvalchemi.dynamics.base.BaseDynamics`.

    Attributes
    ----------
    __needs_keys__ : set[str]
        ``{"forces"}`` — forces must be present in model outputs.
    __provides_keys__ : set[str]
        ``{"positions", "velocities"}`` — updated each step.

    Examples
    --------
    >>> nve = NVE(model, dt=1.0, n_steps=1000)
    >>> nve.run(batch)
    """

    __needs_keys__: set[str] = {"forces"}
    __provides_keys__: set[str] = {"positions", "velocities"}

    def __init__(
        self,
        model: BaseModelMixin,
        dt: float | torch.Tensor,
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

    def _init_state(self, batch: Batch) -> None:
        """Allocate per-system timestep tensor.

        Parameters
        ----------
        batch : Batch
            The first concrete batch; provides M, device, and dtype.
        """
        M = batch.num_graphs
        dev = batch.device
        dtype = batch.positions.dtype
        self._state = _make_state_batch(
            {"dt": _to_per_system(self._dt_init, M, dev, dtype)},
            dev,
        )

    def _make_new_state(self, n: int, template_batch: Batch) -> Batch:
        """Return default state for *n* newly admitted systems.

        Parameters
        ----------
        n : int
            Number of new systems.
        template_batch : Batch
            Updated active batch; used for device and dtype.
        """
        dev = template_batch.device
        dtype = template_batch.positions.dtype
        return _make_state_batch(
            {"dt": _to_per_system(self._dt_init, n, dev, dtype)},
            dev,
        )

    def pre_update(self, batch: Batch) -> None:
        """Position update and half-step velocity kick.

        Parameters
        ----------
        batch : Batch
            Current batch; *positions* and *velocities* updated in-place.
        """
        vv_position_update(
            batch.positions,
            batch.velocities,
            batch.forces,
            batch.atomic_masses,
            self._state.dt,
            batch.batch_idx.int(),
        )

    def post_update(self, batch: Batch) -> None:
        """Finalize velocities with new forces.

        Parameters
        ----------
        batch : Batch
            Current batch; *velocities* updated in-place.
        """
        vv_velocity_finalize(
            batch.velocities,
            batch.forces,
            batch.atomic_masses,
            self._state.dt,
            batch.batch_idx.int(),
        )
