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
FIRE and FIRE+variable-cell geometry optimizers.

FIRE (Fast Inertial Relaxation Engine) drives atomic positions toward a
local energy minimum using a modified molecular dynamics trajectory with
adaptive timestep and velocity-mixing.

* ``FIRE``            — fixed-cell coordinate optimizer.
* ``FIREVariableCell`` — variable-cell optimizer using NPH-like cell
  propagation at zero target pressure combined with FIRE velocity
  mixing on the extended coordinate+cell DOFs.

The step is split around the force (and stress) evaluation:

**FIRE** (fixed-cell)

* ``pre_update``:  full FIRE step (MD integration + mixing) using
                   forces from the *previous* model call.
* ``post_update``: no-op; forces from the new positions will be used
                   on the next ``pre_update``.

**FIREVariableCell**

* ``pre_update``:  velocity half-kick → position full-step →
                   cell full-step.
* [model evaluates F and stress at r(t+dt), h(t+dt)]
* ``post_update``: velocity half-kick → FIRE velocity mixing +
                   parameter update.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from nvalchemi.data import Batch
from nvalchemi.dynamics._ops._bridge import _make_state_batch, _to_per_system
from nvalchemi.dynamics._ops.fire import fire_step, fire_update
from nvalchemi.dynamics._ops.npt_nph import (
    nph_velocity_half_step,
    npt_cell_update,
    npt_position_update,
)
from nvalchemi.dynamics._ops.velocity_verlet import vv_velocity_finalize
from nvalchemi.dynamics.base import BaseDynamics

if TYPE_CHECKING:
    from nvalchemi.dynamics.base import ConvergenceHook, Hook
    from nvalchemi.models.base import BaseModelMixin

__all__ = ["FIRE", "FIREVariableCell"]

_FIRE_DEFAULTS = dict(
    maxstep=0.2,
    n_min=5,
    f_dec=0.5,
    f_inc=1.1,
    alpha_start=0.1,
    f_alpha=0.99,
)


class FIRE(BaseDynamics):
    """Fixed-cell FIRE geometry optimizer.

    Drives atomic coordinates to a local energy minimum using the Fast
    Inertial Relaxation Engine algorithm (Bitzek et al., 2006).

    Parameters
    ----------
    model : BaseModelMixin
        The neural network potential model.
    dt : float or torch.Tensor
        Initial adaptive timestep ``[M]`` or scalar.
    dt_max : float or torch.Tensor, optional
        Maximum timestep ``[M]`` or scalar.  Default ``10 × dt``.
    dt_min : float or torch.Tensor, optional
        Minimum timestep ``[M]`` or scalar.  Default ``0.02 × dt``.
    maxstep : float
        Maximum displacement per step.  Default 0.2.
    n_min : int
        Positive-power steps before timestep increase.  Default 5.
    f_dec : float
        Timestep decrease factor.  Default 0.5.
    f_inc : float
        Timestep increase factor.  Default 1.1.
    alpha_start : float
        Initial FIRE mixing parameter.  Default 0.1.
    f_alpha : float
        Alpha decrease factor.  Default 0.99.
    uphill : bool or torch.Tensor, optional
        Enable uphill step detection.  If ``bool``, broadcast to all
        systems; if ``torch.Tensor`` of shape ``[M]`` or ``[]``, use
        per-system values.  Default ``False`` (vanilla FIRE).
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
        dt_max: float | torch.Tensor | None = None,
        dt_min: float | torch.Tensor | None = None,
        maxstep: float = _FIRE_DEFAULTS["maxstep"],
        n_min: int = _FIRE_DEFAULTS["n_min"],
        f_dec: float = _FIRE_DEFAULTS["f_dec"],
        f_inc: float = _FIRE_DEFAULTS["f_inc"],
        alpha_start: float = _FIRE_DEFAULTS["alpha_start"],
        f_alpha: float = _FIRE_DEFAULTS["f_alpha"],
        uphill: bool | torch.Tensor = False,
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
        self._dt_max_init = dt_max
        self._dt_min_init = dt_min
        self.maxstep = maxstep
        self.n_min = n_min
        self.f_dec = f_dec
        self.f_inc = f_inc
        self.alpha_start = alpha_start
        self.f_alpha = f_alpha
        self._uphill_init = uphill

    def _make_uphill_flag(self, M: int, dev: torch.device) -> torch.Tensor:
        """Convert *uphill* init value to a per-system int32 tensor."""
        if isinstance(self._uphill_init, torch.Tensor):
            return (
                self._uphill_init.expand(M)
                .to(dtype=torch.int32, device=dev)
                .contiguous()
            )
        val = 1 if self._uphill_init else 0
        return torch.full((M,), val, dtype=torch.int32, device=dev)

    def _init_state(self, batch: Batch) -> None:
        M = batch.num_graphs
        dev = batch.device
        dtype = batch.positions.dtype
        dt = _to_per_system(self._dt_init, M, dev, dtype)
        dt_max = (
            _to_per_system(self._dt_max_init, M, dev, dtype)
            if self._dt_max_init is not None
            else dt * 10.0
        )
        dt_min = (
            _to_per_system(self._dt_min_init, M, dev, dtype)
            if self._dt_min_init is not None
            else dt * 0.02
        )
        self._state = _make_state_batch(
            {
                "dt": dt,
                "dt_max": dt_max,
                "dt_min": dt_min,
                "alpha": torch.full((M,), self.alpha_start, dtype=dtype, device=dev),
                "alpha_start": torch.full(
                    (M,), self.alpha_start, dtype=dtype, device=dev
                ),
                "f_alpha": torch.full((M,), self.f_alpha, dtype=dtype, device=dev),
                "maxstep": torch.full((M,), self.maxstep, dtype=dtype, device=dev),
                "n_min": torch.full((M,), self.n_min, dtype=torch.int32, device=dev),
                "f_dec": torch.full((M,), self.f_dec, dtype=dtype, device=dev),
                "f_inc": torch.full((M,), self.f_inc, dtype=dtype, device=dev),
                "uphill_flag": self._make_uphill_flag(M, dev),
                "n_steps_positive": torch.zeros(M, dtype=torch.int32, device=dev),
                # Pre-allocated scratch for dot-product reductions.
                "vf": torch.zeros(M, dtype=dtype, device=dev),
                "vv": torch.zeros(M, dtype=dtype, device=dev),
                "ff": torch.zeros(M, dtype=dtype, device=dev),
            },
            dev,
        )

    def _make_new_state(self, n: int, template_batch: Batch) -> Batch:
        dev = template_batch.device
        dtype = template_batch.positions.dtype
        dt = _to_per_system(self._dt_init, n, dev, dtype)
        dt_max = (
            _to_per_system(self._dt_max_init, n, dev, dtype)
            if self._dt_max_init is not None
            else dt * 10.0
        )
        dt_min = (
            _to_per_system(self._dt_min_init, n, dev, dtype)
            if self._dt_min_init is not None
            else dt * 0.02
        )
        return _make_state_batch(
            {
                "dt": dt,
                "dt_max": dt_max,
                "dt_min": dt_min,
                "alpha": torch.full((n,), self.alpha_start, dtype=dtype, device=dev),
                "alpha_start": torch.full(
                    (n,), self.alpha_start, dtype=dtype, device=dev
                ),
                "f_alpha": torch.full((n,), self.f_alpha, dtype=dtype, device=dev),
                "maxstep": torch.full((n,), self.maxstep, dtype=dtype, device=dev),
                "n_min": torch.full((n,), self.n_min, dtype=torch.int32, device=dev),
                "f_dec": torch.full((n,), self.f_dec, dtype=dtype, device=dev),
                "f_inc": torch.full((n,), self.f_inc, dtype=dtype, device=dev),
                "uphill_flag": self._make_uphill_flag(n, dev),
                "n_steps_positive": torch.zeros(n, dtype=torch.int32, device=dev),
                "vf": torch.zeros(n, dtype=dtype, device=dev),
                "vv": torch.zeros(n, dtype=dtype, device=dev),
                "ff": torch.zeros(n, dtype=dtype, device=dev),
            },
            dev,
        )

    def pre_update(self, batch: Batch) -> None:
        """Full FIRE step using current forces.

        Parameters
        ----------
        batch : Batch
            Current batch; *positions* and *velocities* updated in-place.
        """
        fire_step(
            batch.positions,
            batch.velocities,
            batch.forces,
            batch.atomic_masses,
            self._state.alpha,
            self._state.dt,
            self._state.n_steps_positive,
            self._state.alpha_start,
            self._state.f_alpha,
            self._state.dt_min,
            self._state.dt_max,
            self._state.maxstep,
            self._state.n_min,
            self._state.f_dec,
            self._state.f_inc,
            self._state.uphill_flag,
            vf=self._state.vf,
            vv=self._state.vv,
            ff=self._state.ff,
            batch_idx=batch.batch.int(),
        )

    def post_update(self, batch: Batch) -> None:
        """No-op; forces from new positions are used on the next step."""


class FIREVariableCell(BaseDynamics):
    """Variable-cell FIRE geometry optimizer.

    Extends FIRE to simultaneously relax atomic coordinates and the
    simulation cell.  Cell forces are derived from the model's stress
    tensor via ``stress_to_cell_force``.

    Integration order (symmetric around force/stress evaluation):

    * ``pre_update``:  v half-kick → r full-step → cell full-step.
    * [model evaluates F and stress at new r, h]
    * ``post_update``: v half-kick → FIRE mixing + parameter update.

    Parameters
    ----------
    model : BaseModelMixin
        The neural network potential model.  Must produce ``"stress"``.
    dt : float or torch.Tensor
        Initial adaptive timestep ``[M]`` or scalar.
    dt_max : float or torch.Tensor, optional
        Maximum timestep ``[M]`` or scalar.  Default ``10 × dt``.
    dt_min : float or torch.Tensor, optional
        Minimum timestep ``[M]`` or scalar.  Default ``0.02 × dt``.
    maxstep : float
        Maximum displacement per step.  Default 0.2.
    n_min : int
        Positive-power steps before timestep increase.  Default 5.
    f_dec : float
        Timestep decrease factor.  Default 0.5.
    f_inc : float
        Timestep increase factor.  Default 1.1.
    alpha_start : float
        Initial FIRE mixing parameter.  Default 0.1.
    f_alpha : float
        Alpha decrease factor.  Default 0.99.
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
        ``{"forces", "stress"}``.
    __provides_keys__ : set[str]
        ``{"positions", "velocities", "cell"}``.
    """

    __needs_keys__: set[str] = {"forces", "stress"}
    __provides_keys__: set[str] = {"positions", "velocities", "cell"}

    def __init__(
        self,
        model: BaseModelMixin,
        dt: float | torch.Tensor,
        dt_max: float | torch.Tensor | None = None,
        dt_min: float | torch.Tensor | None = None,
        maxstep: float = _FIRE_DEFAULTS["maxstep"],
        n_min: int = _FIRE_DEFAULTS["n_min"],
        f_dec: float = _FIRE_DEFAULTS["f_dec"],
        f_inc: float = _FIRE_DEFAULTS["f_inc"],
        alpha_start: float = _FIRE_DEFAULTS["alpha_start"],
        f_alpha: float = _FIRE_DEFAULTS["f_alpha"],
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
        self._dt_max_init = dt_max
        self._dt_min_init = dt_min
        self.maxstep = maxstep
        self.n_min = n_min
        self.f_dec = f_dec
        self.f_inc = f_inc
        self.alpha_start = alpha_start
        self.f_alpha = f_alpha

    def _init_state(self, batch: Batch) -> None:
        M = batch.num_graphs
        dev = batch.device
        dtype = batch.positions.dtype
        dt = _to_per_system(self._dt_init, M, dev, dtype)
        dt_max = (
            _to_per_system(self._dt_max_init, M, dev, dtype)
            if self._dt_max_init is not None
            else dt * 10.0
        )
        dt_min = (
            _to_per_system(self._dt_min_init, M, dev, dtype)
            if self._dt_min_init is not None
            else dt * 0.02
        )
        self._state = _make_state_batch(
            {
                "dt": dt,
                "dt_max": dt_max,
                "dt_min": dt_min,
                "alpha": torch.full((M,), self.alpha_start, dtype=dtype, device=dev),
                "alpha_start": torch.full(
                    (M,), self.alpha_start, dtype=dtype, device=dev
                ),
                "f_alpha": torch.full((M,), self.f_alpha, dtype=dtype, device=dev),
                "n_min": torch.full((M,), self.n_min, dtype=torch.int32, device=dev),
                "f_dec": torch.full((M,), self.f_dec, dtype=dtype, device=dev),
                "f_inc": torch.full((M,), self.f_inc, dtype=dtype, device=dev),
                "n_steps_positive": torch.zeros(M, dtype=torch.int32, device=dev),
                "cell_velocity": torch.zeros(M, 3, 3, dtype=dtype, device=dev),
                "vf": torch.zeros(M, dtype=dtype, device=dev),
                "vv": torch.zeros(M, dtype=dtype, device=dev),
                "ff": torch.zeros(M, dtype=dtype, device=dev),
                # Scratch for npt_position_update cells_inv.
                "cells_inv": torch.zeros(M, 3, 3, dtype=dtype, device=dev),
            },
            dev,
        )

    def _make_new_state(self, n: int, template_batch: Batch) -> Batch:
        dev = template_batch.device
        dtype = template_batch.positions.dtype
        dt = _to_per_system(self._dt_init, n, dev, dtype)
        dt_max = (
            _to_per_system(self._dt_max_init, n, dev, dtype)
            if self._dt_max_init is not None
            else dt * 10.0
        )
        dt_min = (
            _to_per_system(self._dt_min_init, n, dev, dtype)
            if self._dt_min_init is not None
            else dt * 0.02
        )
        return _make_state_batch(
            {
                "dt": dt,
                "dt_max": dt_max,
                "dt_min": dt_min,
                "alpha": torch.full((n,), self.alpha_start, dtype=dtype, device=dev),
                "alpha_start": torch.full(
                    (n,), self.alpha_start, dtype=dtype, device=dev
                ),
                "f_alpha": torch.full((n,), self.f_alpha, dtype=dtype, device=dev),
                "n_min": torch.full((n,), self.n_min, dtype=torch.int32, device=dev),
                "f_dec": torch.full((n,), self.f_dec, dtype=dtype, device=dev),
                "f_inc": torch.full((n,), self.f_inc, dtype=dtype, device=dev),
                "n_steps_positive": torch.zeros(n, dtype=torch.int32, device=dev),
                "cell_velocity": torch.zeros(n, 3, 3, dtype=dtype, device=dev),
                "vf": torch.zeros(n, dtype=dtype, device=dev),
                "vv": torch.zeros(n, dtype=dtype, device=dev),
                "ff": torch.zeros(n, dtype=dtype, device=dev),
                "cells_inv": torch.zeros(n, 3, 3, dtype=dtype, device=dev),
            },
            dev,
        )

    def pre_update(self, batch: Batch) -> None:
        """Velocity half-kick → position full-step → cell full-step.

        Parameters
        ----------
        batch : Batch
            Current batch; *positions*, *velocities*, and *cell*
            updated in-place.
        """
        cells_inv = torch.linalg.inv(batch.cell)
        volumes = torch.linalg.det(batch.cell).abs()
        num_atoms = torch.bincount(batch.batch.long(), minlength=batch.num_graphs).to(
            dtype=torch.int32, device=batch.device
        )
        nph_velocity_half_step(
            batch.velocities,
            batch.atomic_masses,
            batch.forces,
            self._state.cell_velocity,
            volumes,
            num_atoms,
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
        """Velocity half-kick → FIRE mixing + parameter update.

        Parameters
        ----------
        batch : Batch
            Current batch; *velocities* updated in-place.
        """
        # Second velocity half-kick with new forces.
        vv_velocity_finalize(
            batch.velocities,
            batch.forces,
            batch.atomic_masses,
            self._state.dt,
            batch.batch.int(),
        )
        # FIRE velocity mixing on atomic DOFs.
        fire_update(
            batch.velocities,
            batch.forces,
            self._state.alpha,
            self._state.dt,
            self._state.n_steps_positive,
            self._state.alpha_start,
            self._state.f_alpha,
            self._state.dt_min,
            self._state.dt_max,
            self._state.n_min,
            self._state.f_dec,
            self._state.f_inc,
            vf=self._state.vf,
            vv=self._state.vv,
            ff=self._state.ff,
            batch_idx=batch.batch.int(),
        )
