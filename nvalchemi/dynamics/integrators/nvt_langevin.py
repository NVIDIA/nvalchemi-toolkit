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

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import torch

from nvalchemi.data import Batch
from nvalchemi.dynamics._ops._bridge import _make_state_batch, _to_per_system
from nvalchemi.dynamics._ops.langevin import (
    langevin_finalize,
    langevin_half_step_device_seed,
)
from nvalchemi.dynamics._units import (
    KB_EV,
    fs_to_internal_time,
    per_fs_to_internal_rate,
)
from nvalchemi.dynamics.base import BaseDynamics

if TYPE_CHECKING:
    from nvalchemi.dynamics.base import ConvergenceHook
    from nvalchemi.hooks import Hook
    from nvalchemi.models.base import BaseModelMixin

__all__ = ["NVTLangevin"]


class NVTLangevin(BaseDynamics):
    r"""NVT integrator via BAOAB Langevin dynamics.

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
        Langevin friction coefficient :math:`\gamma` in ``1/fs`` ``[M]`` or scalar.
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

    @property
    def random_seed(self) -> int:
        """Base seed of the thermostat noise, which every step offsets by ``step_count``."""
        return self._random_seed

    @random_seed.setter
    def random_seed(self, value: int) -> None:
        """Move the noise onto another stream from the next step on."""
        self._random_seed = value

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
        # The RNG seed of the NEXT half step lives on the device and is advanced
        # by the op itself: a captured CUDA graph then draws fresh noise on every
        # replay (a Python int seed would be baked into the graph).
        self._rng_seed: torch.Tensor = torch.tensor(
            [self._random_seed + self.step_count], dtype=torch.int32, device=dev
        )
        self._rng_host_step: int = self.step_count
        self._rng_base_seed: int = self._random_seed

    def apply_per_system_params(
        self, params: Mapping[str, torch.Tensor], batch: Batch
    ) -> None:
        """Rebind each walker's target temperature and rescale its velocities.

        Langevin has no thermostat memory to transform — its noise amplitude
        is read from the per-system ``temperature`` every step — so the
        rebinding is the target update plus the velocity rescaling that keeps
        the kinetic energy consistent with the new target.  Velocities scale
        by ``sqrt(T_new / T_old)``, in the same call, because the two together
        are the indivisible change.

        Parameters
        ----------
        params : Mapping[str, torch.Tensor]
            Must contain ``"temperature"`` in Kelvin per graph, shape ``[B]``.
        batch : Batch
            The live batch; ``velocities`` is modified in place.

        Raises
        ------
        RuntimeError
            If called before the integrator state exists.
        KeyError
            If *params* names something this integrator cannot rebind.
        """
        state = getattr(self, "_state", None)
        if state is None:
            raise RuntimeError(
                "NVTLangevin.apply_per_system_params: integrator state is "
                "not initialised; run or prime the dynamics first."
            )
        temperature = self._validated_temperature(params, state.temperature)

        old_kT = state.temperature.reshape(-1).clone()
        new_kT = temperature * KB_EV
        with torch.no_grad():
            state.temperature.copy_(new_kT.reshape(state.temperature.shape))
        self._rescale_velocities(torch.sqrt(new_kT / old_kT), batch)

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

    def pre_update(self, batch: Batch) -> None:
        """BAOAB pre-force half: B-A-O-A sequence.

        Parameters
        ----------
        batch : Batch
            Current batch; *positions* and *velocities* updated in-place.
        """
        if (
            self._rng_host_step != self.step_count
            or self._rng_base_seed != self._random_seed
        ):
            # step_count or random_seed was changed from outside (restart, rewind,
            # ``seed_offset``): resync the device seed.  A resync recorded into a
            # CUDA graph would reset the seed on every replay, so it is refused there.
            if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "NVTLangevin: the device RNG seed is out of sync with step_count "
                    "or random_seed during CUDA-graph capture.  Call dyn.sync_rng() "
                    "(or run one eager step) before capturing."
                )
            self.sync_rng()
        langevin_half_step_device_seed(
            batch.positions,
            batch.velocities,
            batch.forces,
            batch.atomic_masses,
            self._state.dt,
            self._state.temperature,
            self._state.friction,
            self._rng_seed,
            batch.batch_idx.int(),
        )
        self._rng_host_step = self.step_count + 1

    def sync_rng(self) -> None:
        """Set the device RNG seed to ``random_seed + step_count`` (eager only).

        ``pre_update`` does this by itself before an eager step whenever
        ``step_count`` or ``random_seed`` changed since the last step (a restart,
        a rewind, ``seed_offset``).  Call it explicitly after such a change and
        before capturing ``step`` in a CUDA graph: a resync cannot be recorded
        into a graph, because every replay would reset the seed.

        Raises
        ------
        RuntimeError
            If the integrator state is not initialised yet, or the current CUDA
            stream is capturing.
        """
        if getattr(self, "_rng_seed", None) is None:
            raise RuntimeError(
                "NVTLangevin.sync_rng: integrator state is not initialised; "
                "run or prime the dynamics first."
            )
        if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "NVTLangevin.sync_rng must run eagerly, not in a capture."
            )
        self._rng_seed.fill_(self._random_seed + self.step_count)
        self._rng_host_step = self.step_count
        self._rng_base_seed = self._random_seed

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
            batch.batch_idx.int(),
        )
