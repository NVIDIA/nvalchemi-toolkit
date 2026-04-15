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
NVT integrator via BAOAB Langevin dynamics.

The BAOAB splitting scheme samples the canonical (NVT) ensemble and
provides excellent configurational sampling accuracy while remaining
simple to implement.

The step is split around the force evaluation:

* ``pre_update``:   B-A-O-A — half kick, drift, Ornstein-Uhlenbeck,
                               drift
* [model evaluates F(r(t+dt))]
* ``post_update``:  B         — final half velocity kick

Reference: Leimkuhler & Matthews, *BAOAB algorithm* (2012).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from nvalchemi.data import Batch
from nvalchemi.dynamics._ops._bridge import _make_state_batch, _to_per_system
from nvalchemi.dynamics._ops.langevin import langevin_finalize, langevin_half_step
from nvalchemi.dynamics._units import fs_to_internal_time, per_fs_to_internal_rate
from nvalchemi.dynamics.base import BaseDynamics
from nvalchemi.dynamics.hooks._utils import KB_EV

if TYPE_CHECKING:
    from nvalchemi.dynamics.base import ConvergenceHook
    from nvalchemi.hooks import Hook
    from nvalchemi.models.base import BaseModelMixin

__all__ = ["NVTLangevin"]


class NVTLangevin(BaseDynamics):
    """NVT integrator via BAOAB Langevin dynamics.

    Samples the canonical ensemble via a stochastic Ornstein-Uhlenbeck
    process that acts as an exact thermostat.

    Parameters
    ----------
    model : BaseModelMixin
        The neural network potential model.
    dt : float or torch.Tensor
        Integration timestep in femtoseconds ``[M]`` or scalar.
    temperature : float or torch.Tensor
        Target temperature in Kelvin ``[M]`` or scalar.
    friction : float or torch.Tensor
        Langevin friction coefficient γ in ``1/fs`` ``[M]`` or scalar.
        Controls thermostat coupling strength.
    random_seed : int, optional
        Global RNG seed for the stochastic O step.  Default 42.
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
        friction: float | torch.Tensor,
        random_seed: int = 42,
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
        self._friction_init = per_fs_to_internal_rate(friction)
        self._random_seed = random_seed

    def _init_state(self, batch: Batch) -> None:
        M = batch.num_graphs
        dev = batch.device
        dtype = batch.positions.dtype
        # The Warp kernel expects kT in energy units (eV), not T in Kelvin.
        kT_init = self._temperature_init * KB_EV
        self._state = _make_state_batch(
            {
                "dt": _to_per_system(self._dt_init, M, dev, dtype),
                "temperature": _to_per_system(kT_init, M, dev, dtype),
                "friction": _to_per_system(self._friction_init, M, dev, dtype),
            },
            dev,
        )
        # Cache int32 batch index — graph topology never changes during MD.
        # Refreshed in _get_batch_int32 if the batch composition changes
        # (e.g. after an inflight-batching refill).
        self._batch_int32: torch.Tensor = batch.batch_idx.int()

    def _make_new_state(self, n: int, template_batch: Batch) -> Batch:
        dev = template_batch.device
        dtype = template_batch.positions.dtype
        kT_init = self._temperature_init * KB_EV
        return _make_state_batch(
            {
                "dt": _to_per_system(self._dt_init, n, dev, dtype),
                "temperature": _to_per_system(kT_init, n, dev, dtype),
                "friction": _to_per_system(self._friction_init, n, dev, dtype),
            },
            dev,
        )

    def _get_batch_int32(self, batch: Batch) -> torch.Tensor:
        """Return the per-atom int32 batch index, refreshing only when needed.

        Graph topology never changes during a fixed-batch MD run, so we cache
        the int32 conversion and only reallocate when the atom count changes
        (e.g. after an inflight-batching refill that adds or removes systems).
        """
        if (
            not hasattr(self, "_batch_int32")
            or self._batch_int32.shape[0] != batch.num_nodes
        ):
            self._batch_int32 = batch.batch_idx.int()
        return self._batch_int32

    def pre_update(self, batch: Batch) -> None:
        """BAOAB pre-force half: B-A-O-A sequence.

        Parameters
        ----------
        batch : Batch
            Current batch; *positions* and *velocities* updated in-place.
        """
        langevin_half_step(
            batch.positions,
            batch.velocities,
            batch.forces,
            batch.atomic_masses,
            self._state.dt,
            self._state.temperature,
            self._state.friction,
            self._random_seed + self.step_count,
            self._get_batch_int32(batch),
        )

    def post_update(self, batch: Batch) -> None:
        """BAOAB post-force final B step.

        Parameters
        ----------
        batch : Batch
            Current batch; *velocities* updated in-place.
        """
        langevin_finalize(
            batch.velocities,
            batch.forces,
            batch.atomic_masses,
            self._state.dt,
            self._get_batch_int32(batch),
        )
