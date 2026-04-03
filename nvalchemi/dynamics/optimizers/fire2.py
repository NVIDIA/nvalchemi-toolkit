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
FIRE2 and FIRE2+variable-cell geometry optimizers.

FIRE2 (Shuang et al., 2020) improves on FIRE with better restart
conditions and a modified velocity-mixing rule.

* ``FIRE2``            — fixed-cell coordinate optimizer.
* ``FIRE2VariableCell`` — variable-cell optimizer.

Both classes delegate to ``fire2_step_coord`` and
``fire2_step_coord_cell`` from :mod:`nvalchemiops.torch.fire2`, which
wrap the full FIRE2 step (MD integration + mixing).  The step is
therefore placed entirely in ``pre_update``; ``post_update`` is a no-op.

FIRE2 uses different hyperparameter names from FIRE:

* ``delaystep``   — minimum steps before adaptation (like n_min, default 60)
* ``dtgrow``      — timestep growth factor (like f_inc, default 1.05)
* ``dtshrink``    — timestep shrink factor (like f_dec, default 0.75)
* ``alphashrink`` — alpha decrease factor (like f_alpha, default 0.985)
* ``alpha0``      — initial mixing parameter (like alpha_start, default 0.09)
* ``tmax``        — maximum timestep (default 0.08)
* ``tmin``        — minimum timestep (default 0.005)
* ``maxstep``     — maximum displacement per step (default 0.1)

Per-system state: ``dt [M]``, ``alpha [M]``, ``nsteps_inc [M, int32]``.
For :class:`FIRE2VariableCell`: additionally ``cell_velocities [M,3,3]``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from nvalchemi.data import Batch
from nvalchemi.dynamics._ops._bridge import _make_state_batch, _to_per_system
from nvalchemi.dynamics._ops.fire import fire2_step_coord, fire2_step_coord_cell
from nvalchemi.dynamics._ops.npt_nph import stress_to_cell_force
from nvalchemi.dynamics.base import BaseDynamics

if TYPE_CHECKING:
    from nvalchemi.dynamics.base import ConvergenceHook
    from nvalchemi.hooks import Hook
    from nvalchemi.models.base import BaseModelMixin

__all__ = ["FIRE2", "FIRE2VariableCell"]

_FIRE2_DEFAULTS = dict(
    delaystep=60,
    dtgrow=1.05,
    dtshrink=0.75,
    alphashrink=0.985,
    alpha0=0.09,
    tmax=0.08,
    tmin=0.005,
    maxstep=0.1,
)


def _build_state(
    n: int,
    dt: torch.Tensor,
    alpha0: float,
    dtype: torch.dtype,
    dev: torch.device,
    *,
    with_cell: bool = False,
) -> dict[str, torch.Tensor]:
    d = {
        "dt": dt,
        "alpha": torch.full((n,), alpha0, dtype=dtype, device=dev),
        "nsteps_inc": torch.zeros(n, dtype=torch.int32, device=dev),
        # Scratch buffers for FIRE2 dot-product reductions.
        "vf": torch.zeros(n, dtype=dtype, device=dev),
        "v_sumsq": torch.zeros(n, dtype=dtype, device=dev),
        "f_sumsq": torch.zeros(n, dtype=dtype, device=dev),
    }
    if with_cell:
        d["cell_velocities"] = torch.zeros(n, 3, 3, dtype=dtype, device=dev)
    return d


class FIRE2(BaseDynamics):
    """Fixed-cell FIRE2 geometry optimizer.

    Parameters
    ----------
    model : BaseModelMixin
        The neural network potential model.
    dt : float or torch.Tensor
        Initial adaptive timestep ``[M]`` or scalar.
    delaystep : int
        Minimum steps before adaptation.  Default 60.
    dtgrow : float
        Timestep growth factor.  Default 1.05.
    dtshrink : float
        Timestep shrink factor.  Default 0.75.
    alphashrink : float
        Alpha decrease factor.  Default 0.985.
    alpha0 : float
        Initial mixing parameter.  Default 0.09.
    tmax : float
        Maximum timestep.  Default 0.08.
    tmin : float
        Minimum timestep.  Default 0.005.
    maxstep : float
        Maximum displacement per step.  Default 0.1.
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
        delaystep: int = _FIRE2_DEFAULTS["delaystep"],
        dtgrow: float = _FIRE2_DEFAULTS["dtgrow"],
        dtshrink: float = _FIRE2_DEFAULTS["dtshrink"],
        alphashrink: float = _FIRE2_DEFAULTS["alphashrink"],
        alpha0: float = _FIRE2_DEFAULTS["alpha0"],
        tmax: float = _FIRE2_DEFAULTS["tmax"],
        tmin: float = _FIRE2_DEFAULTS["tmin"],
        maxstep: float = _FIRE2_DEFAULTS["maxstep"],
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
        self.delaystep = delaystep
        self.dtgrow = dtgrow
        self.dtshrink = dtshrink
        self.alphashrink = alphashrink
        self.alpha0 = alpha0
        self.tmax = tmax
        self.tmin = tmin
        self.maxstep = maxstep

    def _init_state(self, batch: Batch) -> None:
        M = batch.num_graphs
        dev = batch.device
        dtype = batch.positions.dtype
        dt = _to_per_system(self._dt_init, M, dev, dtype)
        self._state = _make_state_batch(
            _build_state(M, dt, self.alpha0, dtype, dev),
            dev,
        )

    def _make_new_state(self, n: int, template_batch: Batch) -> Batch:
        dev = template_batch.device
        dtype = template_batch.positions.dtype
        dt = _to_per_system(self._dt_init, n, dev, dtype)
        return _make_state_batch(
            _build_state(n, dt, self.alpha0, dtype, dev),
            dev,
        )

    def pre_update(self, batch: Batch) -> None:
        """Full FIRE2 step using current forces.

        Parameters
        ----------
        batch : Batch
            Current batch; *positions* and *velocities* updated in-place.
        """
        fire2_step_coord(
            batch.positions,
            batch.velocities,
            batch.forces,
            batch.batch.int(),
            self._state.alpha,
            self._state.dt,
            self._state.nsteps_inc,
            vf=self._state.vf,
            v_sumsq=self._state.v_sumsq,
            f_sumsq=self._state.f_sumsq,
            delaystep=self.delaystep,
            dtgrow=self.dtgrow,
            dtshrink=self.dtshrink,
            alphashrink=self.alphashrink,
            alpha0=self.alpha0,
            tmax=self.tmax,
            tmin=self.tmin,
            maxstep=self.maxstep,
        )

    def post_update(self, batch: Batch) -> None:
        """No-op; forces from new positions are used on the next step."""


class FIRE2VariableCell(BaseDynamics):
    """Variable-cell FIRE2 geometry optimizer.

    Simultaneously relaxes atomic coordinates and the simulation cell
    using the FIRE2 algorithm with cell-force derived from the model's
    stress tensor.

    Parameters
    ----------
    model : BaseModelMixin
        The neural network potential model.  Must produce ``"stresses"``.
    dt : float or torch.Tensor
        Initial adaptive timestep ``[M]`` or scalar.
    delaystep : int
        Minimum steps before adaptation.  Default 60.
    dtgrow : float
        Timestep growth factor.  Default 1.05.
    dtshrink : float
        Timestep shrink factor.  Default 0.75.
    alphashrink : float
        Alpha decrease factor.  Default 0.985.
    alpha0 : float
        Initial mixing parameter.  Default 0.09.
    tmax : float
        Maximum timestep.  Default 0.08.
    tmin : float
        Minimum timestep.  Default 0.005.
    maxstep : float
        Maximum displacement per step.  Default 0.1.
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
        delaystep: int = _FIRE2_DEFAULTS["delaystep"],
        dtgrow: float = _FIRE2_DEFAULTS["dtgrow"],
        dtshrink: float = _FIRE2_DEFAULTS["dtshrink"],
        alphashrink: float = _FIRE2_DEFAULTS["alphashrink"],
        alpha0: float = _FIRE2_DEFAULTS["alpha0"],
        tmax: float = _FIRE2_DEFAULTS["tmax"],
        tmin: float = _FIRE2_DEFAULTS["tmin"],
        maxstep: float = _FIRE2_DEFAULTS["maxstep"],
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
        self.delaystep = delaystep
        self.dtgrow = dtgrow
        self.dtshrink = dtshrink
        self.alphashrink = alphashrink
        self.alpha0 = alpha0
        self.tmax = tmax
        self.tmin = tmin
        self.maxstep = maxstep

    def _init_state(self, batch: Batch) -> None:
        M = batch.num_graphs
        dev = batch.device
        dtype = batch.positions.dtype
        dt = _to_per_system(self._dt_init, M, dev, dtype)
        self._state = _make_state_batch(
            _build_state(M, dt, self.alpha0, dtype, dev, with_cell=True),
            dev,
        )

    def _make_new_state(self, n: int, template_batch: Batch) -> Batch:
        dev = template_batch.device
        dtype = template_batch.positions.dtype
        dt = _to_per_system(self._dt_init, n, dev, dtype)
        return _make_state_batch(
            _build_state(n, dt, self.alpha0, dtype, dev, with_cell=True),
            dev,
        )

    def pre_update(self, batch: Batch) -> None:
        """Full FIRE2 variable-cell step using current forces and stress.

        Parameters
        ----------
        batch : Batch
            Current batch; *positions*, *velocities*, and *cell*
            updated in-place.
        """
        volumes = torch.linalg.det(batch.cell).abs()
        # batch.stresses is the raw virial W_phys (energy units, eV).
        # stress_to_cell_force expects the mechanical stress σ = W_phys / V
        # (energy/volume units), so divide by volume here.
        stress_sigma = batch.stresses / volumes.view(-1, 1, 1)
        cell_force = stress_to_cell_force(stress_sigma, batch.cell, volumes)
        fire2_step_coord_cell(
            batch.positions,
            batch.velocities,
            batch.forces,
            batch.cell,
            self._state.cell_velocities,
            cell_force,
            batch.batch.int(),
            self._state.alpha,
            self._state.dt,
            self._state.nsteps_inc,
            vf=self._state.vf,
            v_sumsq=self._state.v_sumsq,
            f_sumsq=self._state.f_sumsq,
            delaystep=self.delaystep,
            dtgrow=self.dtgrow,
            dtshrink=self.dtshrink,
            alphashrink=self.alphashrink,
            alpha0=self.alpha0,
            tmax=self.tmax,
            tmin=self.tmin,
            maxstep=self.maxstep,
        )

    def post_update(self, batch: Batch) -> None:
        """No-op; forces from new positions are used on the next step."""
