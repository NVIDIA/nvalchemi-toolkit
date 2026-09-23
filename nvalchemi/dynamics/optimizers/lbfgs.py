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
L-BFGS and L-BFGS+variable-cell geometry optimizers.

L-BFGS steps along a quasi-Newton direction built from the last
``history_size`` position/force differences, bounded by a ``maxstep`` trust
region.  One force evaluation per step; no energy.

* ``LBFGS``             — fixed-cell coordinate optimizer.
* ``LBFGSVariableCell`` — variable-cell optimizer.

Both classes delegate to ``lbfgs_step_coord`` and ``lbfgs_step_coord_cell``
from :mod:`nvalchemiops.torch.lbfgs`.  The step is placed entirely in
``pre_update``; ``post_update`` is a no-op.

Hyperparameters:

* ``history_size``  — stored curvature pairs (default 6); fixed at allocation
* ``curvature_eps`` — pair acceptance floor (default ``None``: by dtype)
* ``maxstep``       — maximum displacement per step (default 0.2)

State spans two levels: per-system scalars and a segmented ``"lbfgs_dofs"``
level (one row per atom, plus two per system for variable cell) holding
the history.  Positions must not be edited between steps (e.g. by
``WrapPeriodicHook``): the next step differences them against the last.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from nvalchemi.data import Batch
from nvalchemi.dynamics._ops._bridge import _make_two_level_state_batch, _state_level
from nvalchemi.dynamics._ops.lbfgs import (
    LBFGSCellState,
    LBFGSState,
    lbfgs_prepare_cell_state,
    lbfgs_prepare_state,
    lbfgs_step_coord,
    lbfgs_step_coord_cell,
)
from nvalchemi.dynamics.base import BaseDynamics
from nvalchemi.dynamics.hooks.cell_align import AlignCellHook, _aligned_periodic

if TYPE_CHECKING:
    from nvalchemi.dynamics.base import ConvergenceHook
    from nvalchemi.hooks import Hook
    from nvalchemi.models.base import BaseModelMixin

__all__ = ["LBFGS", "LBFGSVariableCell"]

_LBFGS_DEFAULTS = dict(
    history_size=6,
    curvature_eps=None,
    maxstep=0.2,
)

_DOF_LEVEL = "lbfgs_dofs"
_PER_DOF = ("x_base", "force_base", "direction", "s_history", "y_history")
_PER_SYSTEM = (
    "ys", "yy", "alpha_hist", "beta_hist", "ss", "gg", "d0", "dmax", "dquad",
    "alpha_step", "iteration", "end", "n_loop", "history_count",
)  # fmt: skip
_CELL_PER_DOF = ("ext_positions", "ext_forces")
_CELL_PER_SYSTEM = (
    "ref_cell", "ref_cell_inv", "kappa", "phi", "phi_inv", "d_phi",
    "cell_dof_a", "cell_dof_b", "cell_force_a", "cell_force_b",
)  # fmt: skip
# Must match nvalchemiops' check in lbfgs_prepare_cell_state.
_ALIGN_ATOL = 1e-10


def _build_state(
    atoms_per_system: torch.Tensor,
    history_size: int,
    dtype: torch.dtype,
    dev: torch.device,
    *,
    cell: torch.Tensor | None = None,
    cell_force_scale: float = 1.0,
) -> Batch:
    segments = atoms_per_system.to(torch.int32) + (0 if cell is None else 2)
    opt = lbfgs_prepare_state(
        int(segments.sum()),
        segments.numel(),
        dtype=dtype,
        device=dev,
        history_size=history_size,
    )
    system = {k: getattr(opt, k) for k in _PER_SYSTEM}
    dofs = {k: getattr(opt, k) for k in _PER_DOF}
    if cell is not None:
        atom_ptr = torch.nn.functional.pad(
            atoms_per_system.cumsum(0, dtype=torch.int32), (1, 0)
        )
        cs = lbfgs_prepare_cell_state(
            atom_ptr, cell, cell_force_scale=cell_force_scale, dtype=dtype, device=dev
        )
        system |= {k: getattr(cs, k) for k in _CELL_PER_SYSTEM}
        dofs |= {k: getattr(cs, k) for k in _CELL_PER_DOF}
    return _make_two_level_state_batch(
        system, dofs, segments, dev, level_name=_DOF_LEVEL
    )


def _refuse(name: str) -> None:
    raise AttributeError(
        f"{name} is fixed when optimizer state is allocated; "
        "construct a new optimizer to change it"
    )


def _ops_state(state: Batch) -> LBFGSState:
    return LBFGSState(**{k: state[k] for k in _PER_DOF + _PER_SYSTEM})


def _ops_cell_state(state: Batch) -> LBFGSCellState:
    # The packed topology is the segmented level's own; never stored.
    return LBFGSCellState(
        ext_batch_idx=_state_level(state, _DOF_LEVEL).batch_idx.int(),
        ext_atom_ptr=state.level_ptr(_DOF_LEVEL),
        **{k: state[k] for k in _CELL_PER_SYSTEM + _CELL_PER_DOF},
    )


class LBFGS(BaseDynamics):
    """Fixed-cell L-BFGS geometry optimizer.

    Parameters
    ----------
    model : BaseModelMixin
        The neural network potential model.
    history_size : int
        Stored curvature pairs.  Fixed once state is allocated.  Default 6.
    curvature_eps : float, optional
        Curvature-pair acceptance floor.  Default ``None`` (by dtype).
    maxstep : float
        Maximum displacement per step.  Default 0.2.
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
        ``{"positions"}``.
    """

    __needs_keys__: set[str] = {"forces"}
    __provides_keys__: set[str] = {"positions"}

    def __init__(
        self,
        model: BaseModelMixin,
        history_size: int = _LBFGS_DEFAULTS["history_size"],
        curvature_eps: float | None = _LBFGS_DEFAULTS["curvature_eps"],
        maxstep: float = _LBFGS_DEFAULTS["maxstep"],
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
        self._history_size = history_size
        self.curvature_eps = curvature_eps
        self.maxstep = maxstep

    @property
    def history_size(self) -> int:
        """Stored curvature pairs; fixed once state is allocated."""
        return self._history_size

    @history_size.setter
    def history_size(self, value: int) -> None:
        _refuse("history_size")

    def _init_state(self, batch: Batch) -> None:
        self._state = _build_state(
            batch.num_nodes_per_graph,
            self.history_size,
            batch.positions.dtype,
            batch.device,
        )

    def _make_new_state(self, n: int, template_batch: Batch) -> Batch:
        return _build_state(
            template_batch.num_nodes_per_graph[-n:],
            self.history_size,
            template_batch.positions.dtype,
            template_batch.device,
        )

    def pre_update(self, batch: Batch) -> None:
        """Full L-BFGS step using current forces.

        Parameters
        ----------
        batch : Batch
            Current batch; *positions* updated in-place.
        """
        # Detach positions to avoid "non-leaf .grad accessed" warning from
        # wp.from_torch.  In-place updates still apply to the batch storage.
        lbfgs_step_coord(
            batch.positions.detach(),
            batch.forces,
            _ops_state(self._state),
            batch.batch_idx.int(),
            maxstep=self.maxstep,
            curvature_eps=self.curvature_eps,
        )

    def post_update(self, batch: Batch) -> None:
        """No-op; forces from new positions are used on the next step."""


class LBFGSVariableCell(BaseDynamics):
    """Variable-cell L-BFGS geometry optimizer.

    Relaxes atomic coordinates and the cell together, driven by the model's
    stress.  Cells must be aligned: install
    :class:`~nvalchemi.dynamics.hooks.AlignCellHook` (``frequency=1``), or
    pass pre-aligned cells.

    Parameters
    ----------
    model : BaseModelMixin
        The neural network potential model.  Must produce ``"stress"``.
    history_size : int
        Stored curvature pairs.  Fixed once state is allocated.  Default 6.
    curvature_eps : float, optional
        Curvature-pair acceptance floor.  Default ``None`` (by dtype).
    maxstep : float
        Maximum displacement per step.  Default 0.2.
    n_steps : int, optional
        Total steps for :meth:`run`.
    hooks : list[Hook], optional
        Initial hooks.
    convergence_hook : ConvergenceHook or dict, optional
        Convergence criterion.
    cell_force_scale : float
        Multiplier on the atom count normalizing stress-derived cell forces;
        raise it to move the cell less per step.  Fixed once state is
        allocated.  Default 1.0.
    **kwargs
        Forwarded to :class:`~nvalchemi.dynamics.base.BaseDynamics`.

    Attributes
    ----------
    __needs_keys__ : set[str]
        ``{"forces", "stress"}``.
    __provides_keys__ : set[str]
        ``{"positions", "cell"}``.
    """

    __needs_keys__: set[str] = {"forces", "stress"}
    __provides_keys__: set[str] = {"positions", "cell"}

    def __init__(
        self,
        model: BaseModelMixin,
        history_size: int = _LBFGS_DEFAULTS["history_size"],
        curvature_eps: float | None = _LBFGS_DEFAULTS["curvature_eps"],
        maxstep: float = _LBFGS_DEFAULTS["maxstep"],
        n_steps: int | None = None,
        hooks: list[Hook] | None = None,
        convergence_hook: ConvergenceHook | dict | None = None,
        *,
        cell_force_scale: float = 1.0,
        **kwargs: Any,
    ) -> None:
        if cell_force_scale <= 0:
            raise ValueError(
                f"cell_force_scale must be positive; got {cell_force_scale}"
            )
        super().__init__(
            model=model,
            n_steps=n_steps,
            hooks=hooks,
            convergence_hook=convergence_hook,
            **kwargs,
        )
        self._history_size = history_size
        self.curvature_eps = curvature_eps
        self.maxstep = maxstep
        self._cell_force_scale = cell_force_scale

    @property
    def history_size(self) -> int:
        """Stored curvature pairs; fixed once state is allocated."""
        return self._history_size

    @history_size.setter
    def history_size(self, value: int) -> None:
        _refuse("history_size")

    @property
    def cell_force_scale(self) -> float:
        """Cell-force normalization multiplier; fixed once state is allocated."""
        return self._cell_force_scale

    @cell_force_scale.setter
    def cell_force_scale(self, value: float) -> None:
        _refuse("cell_force_scale")

    def _reference_cells(self, batch: Batch, n: int) -> torch.Tensor:
        """Aligned cells of the last *n* systems, for the chart.  Never writes *batch*."""
        align_hooks = [h for h in self.hooks if isinstance(h, AlignCellHook)]
        if any(h.frequency != 1 for h in align_hooks):
            raise ValueError(
                "LBFGSVariableCell requires AlignCellHook(frequency=1): its "
                "reference chart assumes every admitted cell is aligned "
                "before the next step."
            )
        cell = batch.cell.detach()
        if align_hooks:
            aligned = _aligned_periodic(batch)
            if aligned is not None:
                cell = aligned[1]
        cell = cell[-n:]
        skew = torch.triu(cell, 1).abs().amax(dim=(1, 2))
        bad = torch.nonzero(skew > _ALIGN_ATOL).flatten()
        if bad.numel():
            systems = (bad + batch.num_graphs - n).tolist()
            fix = (
                "AlignCellHook aligns only periodic systems; set pbc for "
                "these, or align their cells before the run."
                if align_hooks
                else "Pass AlignCellHook() in this optimizer's hooks (a "
                "FusedStage-level hook is not seen here), or align the cells "
                "before the run."
            )
            raise ValueError(
                f"LBFGSVariableCell: cell for system(s) {systems} is not "
                f"aligned (upper off-diagonal up to {skew.max().item():.3e}). " + fix
            )
        return cell

    def _init_state(self, batch: Batch) -> None:
        self._state = _build_state(
            batch.num_nodes_per_graph,
            self.history_size,
            batch.positions.dtype,
            batch.device,
            cell=self._reference_cells(batch, batch.num_graphs),
            cell_force_scale=self.cell_force_scale,
        )

    def _make_new_state(self, n: int, template_batch: Batch) -> Batch:
        return _build_state(
            template_batch.num_nodes_per_graph[-n:],
            self.history_size,
            template_batch.positions.dtype,
            template_batch.device,
            cell=self._reference_cells(template_batch, n),
            cell_force_scale=self.cell_force_scale,
        )

    def pre_update(self, batch: Batch) -> None:
        """Full L-BFGS variable-cell step using current forces and stress.

        Parameters
        ----------
        batch : Batch
            Current batch; *positions* and *cell* updated in-place.
        """
        # batch.stress is tensile-positive Cauchy stress -W/V (eV/A^3);
        # ops converts it to the cell force internally.
        lbfgs_step_coord_cell(
            batch.positions.detach(),
            batch.cell.detach(),
            batch.forces,
            batch.stress,
            _ops_state(self._state),
            _ops_cell_state(self._state),
            batch.batch_idx.int(),
            maxstep=self.maxstep,
            curvature_eps=self.curvature_eps,
        )

    def post_update(self, batch: Batch) -> None:
        """No-op; forces from new positions are used on the next step."""
