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
PyTorch bindings for FIRE and FIRE2 optimizer kernels.

Wraps :mod:`nvalchemiops.dynamics.optimizers.fire` and the higher-level
PyTorch adapters in :mod:`nvalchemiops.torch.fire2` as
``torch.library.custom_op`` operations, enabling correct behaviour
under ``torch.compile`` and PyTorch's autograd infrastructure.

All FIRE hyperparameters (*alpha_start*, *f_alpha*, *dt_min*, *dt_max*,
*maxstep*, *n_min*, *f_dec*, *f_inc*, *uphill_flag*) are **per-system
tensors** of shape ``[M]``.  This allows heterogeneous batches where
different systems can have different hyperparameters or be in different
algorithm modes.

FIRE2 functions delegate directly to :mod:`nvalchemiops.torch.fire2`
(PyTorch-native), so they accept plain ``torch.Tensor`` arguments without
Warp conversion.

Functions
---------
fire_step
    Full FIRE step: MD integration + velocity mixing + parameter adaptation.
fire_update
    FIRE velocity mixing and parameter update without MD integration
    (for variable-cell workflows).
fire2_step_coord
    FIRE2 coordinate-only step (delegates to nvalchemiops.torch.fire2).
fire2_step_coord_cell
    FIRE2 variable-cell step (delegates to nvalchemiops.torch.fire2).
"""

from __future__ import annotations

import torch
import torch.library
import warp as wp
from nvalchemiops.dynamics.optimizers.fire import (
    fire_step as _fire_step,
)
from nvalchemiops.dynamics.optimizers.fire import (
    fire_update as _fire_update,
)
from nvalchemiops.torch.fire2 import (
    fire2_step_coord as _fire2_coord,
)
from nvalchemiops.torch.fire2 import (
    fire2_step_coord_cell as _fire2_coord_cell,
)

from nvalchemi.dynamics._ops._bridge import _scalar_type, _vec_type

__all__ = [
    "fire_step",
    "fire_update",
    "fire2_step_coord",
    "fire2_step_coord_cell",
]


# ---------------------------------------------------------------------------
# Internal custom ops (all tensor args required; hyperparams are [M] tensors)
# ---------------------------------------------------------------------------


@torch.library.custom_op(
    "nvalchemi::_fire_step_op",
    mutates_args={
        "positions",
        "velocities",
        "alpha",
        "dt",
        "n_steps_positive",
        "vf",
        "vv",
        "ff",
    },
)
def _fire_step_op(
    positions: torch.Tensor,
    velocities: torch.Tensor,
    forces: torch.Tensor,
    masses: torch.Tensor,
    alpha: torch.Tensor,
    dt: torch.Tensor,
    alpha_start: torch.Tensor,
    f_alpha: torch.Tensor,
    dt_min: torch.Tensor,
    dt_max: torch.Tensor,
    maxstep: torch.Tensor,
    n_steps_positive: torch.Tensor,
    n_min: torch.Tensor,
    f_dec: torch.Tensor,
    f_inc: torch.Tensor,
    uphill_flag: torch.Tensor,
    vf: torch.Tensor,
    vv: torch.Tensor,
    ff: torch.Tensor,
    batch_idx: torch.Tensor,
) -> None:
    dtype = positions.dtype
    vec_t = _vec_type(dtype)
    scl_t = _scalar_type(dtype)
    _fire_step(
        wp.from_torch(positions, dtype=vec_t),
        wp.from_torch(velocities, dtype=vec_t),
        wp.from_torch(forces, dtype=vec_t),
        wp.from_torch(masses, dtype=scl_t),
        wp.from_torch(alpha, dtype=scl_t),
        wp.from_torch(dt, dtype=scl_t),
        wp.from_torch(alpha_start, dtype=scl_t),
        wp.from_torch(f_alpha, dtype=scl_t),
        wp.from_torch(dt_min, dtype=scl_t),
        wp.from_torch(dt_max, dtype=scl_t),
        wp.from_torch(maxstep, dtype=scl_t),
        wp.from_torch(n_steps_positive, dtype=wp.int32),
        wp.from_torch(n_min, dtype=wp.int32),
        wp.from_torch(f_dec, dtype=scl_t),
        wp.from_torch(f_inc, dtype=scl_t),
        wp.from_torch(uphill_flag, dtype=wp.int32),
        vf=wp.from_torch(vf, dtype=scl_t),
        vv=wp.from_torch(vv, dtype=scl_t),
        ff=wp.from_torch(ff, dtype=scl_t),
        batch_idx=wp.from_torch(batch_idx, dtype=wp.int32),
    )


@_fire_step_op.register_fake
def _fire_step_op_fake(
    positions,
    velocities,
    forces,
    masses,
    alpha,
    dt,
    alpha_start,
    f_alpha,
    dt_min,
    dt_max,
    maxstep,
    n_steps_positive,
    n_min,
    f_dec,
    f_inc,
    uphill_flag,
    vf,
    vv,
    ff,
    batch_idx,
) -> None:
    pass


@torch.library.custom_op(
    "nvalchemi::_fire_update_op",
    mutates_args={"velocities", "alpha", "dt", "n_steps_positive", "vf", "vv", "ff"},
)
def _fire_update_op(
    velocities: torch.Tensor,
    forces: torch.Tensor,
    alpha: torch.Tensor,
    dt: torch.Tensor,
    alpha_start: torch.Tensor,
    f_alpha: torch.Tensor,
    dt_min: torch.Tensor,
    dt_max: torch.Tensor,
    n_steps_positive: torch.Tensor,
    n_min: torch.Tensor,
    f_dec: torch.Tensor,
    f_inc: torch.Tensor,
    vf: torch.Tensor,
    vv: torch.Tensor,
    ff: torch.Tensor,
    batch_idx: torch.Tensor,
) -> None:
    dtype = velocities.dtype
    vec_t = _vec_type(dtype)
    scl_t = _scalar_type(dtype)
    _fire_update(
        wp.from_torch(velocities, dtype=vec_t),
        wp.from_torch(forces, dtype=vec_t),
        wp.from_torch(alpha, dtype=scl_t),
        wp.from_torch(dt, dtype=scl_t),
        wp.from_torch(alpha_start, dtype=scl_t),
        wp.from_torch(f_alpha, dtype=scl_t),
        wp.from_torch(dt_min, dtype=scl_t),
        wp.from_torch(dt_max, dtype=scl_t),
        wp.from_torch(n_steps_positive, dtype=wp.int32),
        wp.from_torch(n_min, dtype=wp.int32),
        wp.from_torch(f_dec, dtype=scl_t),
        wp.from_torch(f_inc, dtype=scl_t),
        vf=wp.from_torch(vf, dtype=scl_t),
        vv=wp.from_torch(vv, dtype=scl_t),
        ff=wp.from_torch(ff, dtype=scl_t),
        batch_idx=wp.from_torch(batch_idx, dtype=wp.int32),
    )


@_fire_update_op.register_fake
def _fire_update_op_fake(
    velocities,
    forces,
    alpha,
    dt,
    alpha_start,
    f_alpha,
    dt_min,
    dt_max,
    n_steps_positive,
    n_min,
    f_dec,
    f_inc,
    vf,
    vv,
    ff,
    batch_idx,
) -> None:
    pass


# ---------------------------------------------------------------------------
# Public API — allocates scratch buffers, then calls the internal custom op
# ---------------------------------------------------------------------------


def fire_step(
    positions: torch.Tensor,
    velocities: torch.Tensor,
    forces: torch.Tensor,
    masses: torch.Tensor,
    alpha: torch.Tensor,
    dt: torch.Tensor,
    n_steps_positive: torch.Tensor,
    alpha_start: torch.Tensor,
    f_alpha: torch.Tensor,
    dt_min: torch.Tensor,
    dt_max: torch.Tensor,
    maxstep: torch.Tensor,
    n_min: torch.Tensor,
    f_dec: torch.Tensor,
    f_inc: torch.Tensor,
    uphill_flag: torch.Tensor,
    *,
    vf: torch.Tensor | None = None,
    vv: torch.Tensor | None = None,
    ff: torch.Tensor | None = None,
    batch_idx: torch.Tensor | None = None,
) -> None:
    """Full FIRE optimization step.

    Performs an MD integration step followed by FIRE velocity mixing and
    adaptive parameter updates based on power P = F·v.

    Modifies *positions*, *velocities*, *alpha*, *dt*, and
    *n_steps_positive* in-place.  Scratch buffers (*vf*, *vv*, *ff*) are
    zeroed by the kernel each call; pass pre-allocated tensors to avoid
    repeated allocation.

    Parameters
    ----------
    positions : torch.Tensor
        Atomic positions ``[N, 3]``, float32 or float64.
    velocities : torch.Tensor
        Atomic velocities ``[N, 3]``, same dtype.
    forces : torch.Tensor
        Atomic forces ``[N, 3]``, same dtype.
    masses : torch.Tensor
        Atomic masses ``[N]``, same dtype.
    alpha : torch.Tensor
        Per-system FIRE mixing parameter ``[M]``, same dtype.
    dt : torch.Tensor
        Per-system adaptive timestep ``[M]``, same dtype.
    n_steps_positive : torch.Tensor
        Per-system consecutive positive-power step counter ``[M]``, int32.
    alpha_start : torch.Tensor
        Per-system initial FIRE mixing parameter ``[M]``, same dtype as *dt*.
    f_alpha : torch.Tensor
        Per-system alpha decrease factor ``[M]``, same dtype.
    dt_min : torch.Tensor
        Per-system minimum timestep ``[M]``, same dtype.
    dt_max : torch.Tensor
        Per-system maximum timestep ``[M]``, same dtype.
    maxstep : torch.Tensor
        Per-system maximum displacement per step ``[M]``, same dtype.
    n_min : torch.Tensor
        Per-system minimum positive-power steps before timestep increase
        ``[M]``, int32.
    f_dec : torch.Tensor
        Per-system timestep decrease factor ``[M]``, same dtype.
    f_inc : torch.Tensor
        Per-system timestep increase factor ``[M]``, same dtype.
    uphill_flag : torch.Tensor
        Per-system algorithm selector ``[M]``, int32.  0 = vanilla FIRE,
        1 = FIRE with uphill step checks.
    vf : torch.Tensor, optional
        Scratch buffer ``[M]`` for Σ(F·v); allocated if None.
    vv : torch.Tensor, optional
        Scratch buffer ``[M]`` for Σ(v·v); allocated if None.
    ff : torch.Tensor, optional
        Scratch buffer ``[M]`` for Σ(F·F); allocated if None.
    batch_idx : torch.Tensor, optional
        Per-atom system index ``[N]``, int32, non-decreasing.
    """
    M = alpha.shape[0]
    dtype = positions.dtype
    device = positions.device
    if vf is None:
        vf = torch.zeros(M, dtype=dtype, device=device)
    if vv is None:
        vv = torch.zeros(M, dtype=dtype, device=device)
    if ff is None:
        ff = torch.zeros(M, dtype=dtype, device=device)
    if batch_idx is None:
        batch_idx = torch.zeros(positions.shape[0], dtype=torch.int32, device=device)
    _fire_step_op(
        positions,
        velocities,
        forces,
        masses,
        alpha,
        dt,
        alpha_start,
        f_alpha,
        dt_min,
        dt_max,
        maxstep,
        n_steps_positive,
        n_min,
        f_dec,
        f_inc,
        uphill_flag,
        vf,
        vv,
        ff,
        batch_idx,
    )


def fire_update(
    velocities: torch.Tensor,
    forces: torch.Tensor,
    alpha: torch.Tensor,
    dt: torch.Tensor,
    n_steps_positive: torch.Tensor,
    alpha_start: torch.Tensor,
    f_alpha: torch.Tensor,
    dt_min: torch.Tensor,
    dt_max: torch.Tensor,
    n_min: torch.Tensor,
    f_dec: torch.Tensor,
    f_inc: torch.Tensor,
    *,
    vf: torch.Tensor | None = None,
    vv: torch.Tensor | None = None,
    ff: torch.Tensor | None = None,
    batch_idx: torch.Tensor | None = None,
) -> None:
    """FIRE velocity mixing and parameter update (no MD integration).

    Updates velocities via the FIRE mixing rule and adapts *alpha*, *dt*,
    and *n_steps_positive* based on the sign of power P = F·v.
    Used by variable-cell FIRE workflows where the MD step is handled
    separately with cell-aware position scaling.

    Modifies *velocities*, *alpha*, *dt*, and *n_steps_positive* in-place.

    Parameters
    ----------
    velocities : torch.Tensor
        Atomic velocities ``[N, 3]``, float32 or float64.
    forces : torch.Tensor
        Atomic forces ``[N, 3]``, same dtype.
    alpha : torch.Tensor
        Per-system FIRE mixing parameter ``[M]``, same dtype.
    dt : torch.Tensor
        Per-system adaptive timestep ``[M]``, same dtype.
    n_steps_positive : torch.Tensor
        Per-system positive-power step counter ``[M]``, int32.
    alpha_start : torch.Tensor
        Per-system initial mixing parameter ``[M]``, same dtype.
    f_alpha : torch.Tensor
        Per-system alpha decrease factor ``[M]``, same dtype.
    dt_min : torch.Tensor
        Per-system minimum timestep ``[M]``, same dtype.
    dt_max : torch.Tensor
        Per-system maximum timestep ``[M]``, same dtype.
    n_min : torch.Tensor
        Per-system minimum positive steps before increase ``[M]``, int32.
    f_dec : torch.Tensor
        Per-system timestep decrease factor ``[M]``, same dtype.
    f_inc : torch.Tensor
        Per-system timestep increase factor ``[M]``, same dtype.
    vf, vv, ff : torch.Tensor, optional
        Scratch buffers ``[M]``; allocated if None.
    batch_idx : torch.Tensor, optional
        Per-atom system index ``[N]``, int32.
    """
    M = alpha.shape[0]
    dtype = velocities.dtype
    device = velocities.device
    if vf is None:
        vf = torch.zeros(M, dtype=dtype, device=device)
    if vv is None:
        vv = torch.zeros(M, dtype=dtype, device=device)
    if ff is None:
        ff = torch.zeros(M, dtype=dtype, device=device)
    if batch_idx is None:
        batch_idx = torch.zeros(velocities.shape[0], dtype=torch.int32, device=device)
    _fire_update_op(
        velocities,
        forces,
        alpha,
        dt,
        alpha_start,
        f_alpha,
        dt_min,
        dt_max,
        n_steps_positive,
        n_min,
        f_dec,
        f_inc,
        vf,
        vv,
        ff,
        batch_idx,
    )


# ---------------------------------------------------------------------------
# FIRE2 — delegates to nvalchemiops.torch.fire2 (PyTorch-native, no warp)
# ---------------------------------------------------------------------------


def fire2_step_coord(
    positions: torch.Tensor,
    velocities: torch.Tensor,
    forces: torch.Tensor,
    batch_idx: torch.Tensor,
    alpha: torch.Tensor,
    dt: torch.Tensor,
    nsteps_inc: torch.Tensor,
    *,
    vf: torch.Tensor | None = None,
    v_sumsq: torch.Tensor | None = None,
    f_sumsq: torch.Tensor | None = None,
    max_norm: torch.Tensor | None = None,
    delaystep: int = 60,
    dtgrow: float = 1.05,
    dtshrink: float = 0.75,
    alphashrink: float = 0.985,
    alpha0: float = 0.09,
    tmax: float = 0.08,
    tmin: float = 0.005,
    maxstep: float = 0.1,
) -> None:
    """Full FIRE2 coordinate-only optimization step.

    Delegates to :func:`nvalchemiops.torch.fire2.fire2_step_coord`.
    Modifies *positions*, *velocities*, *alpha*, *dt*, and *nsteps_inc*
    in-place.

    Parameters
    ----------
    positions : torch.Tensor
        Atomic positions ``[N, 3]``, float32 or float64.
    velocities : torch.Tensor
        Atomic velocities ``[N, 3]``, same dtype.
    forces : torch.Tensor
        Atomic forces ``[N, 3]``, same dtype.
    batch_idx : torch.Tensor
        Per-atom system index ``[N]``, int32, non-decreasing.
    alpha : torch.Tensor
        Per-system FIRE2 mixing parameter ``[M]``, same dtype.
    dt : torch.Tensor
        Per-system adaptive timestep ``[M]``, same dtype.
    nsteps_inc : torch.Tensor
        Per-system consecutive positive-power step counter ``[M]``, int32.
    vf : torch.Tensor, optional
        Scratch buffer ``[M]`` for Σ(F·v); allocated if None.
    v_sumsq : torch.Tensor, optional
        Scratch buffer ``[M]`` for Σ(v·v); allocated if None.
    f_sumsq : torch.Tensor, optional
        Scratch buffer ``[M]`` for Σ(F·F); allocated if None.
    max_norm : torch.Tensor, optional
        Scratch buffer ``[M]`` for max force norm; allocated if None.
    delaystep : int
        Minimum steps before parameter adaptation.  Default 60.
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
    """
    _fire2_coord(
        positions,
        velocities,
        forces,
        batch_idx,
        alpha,
        dt,
        nsteps_inc,
        vf=vf,
        v_sumsq=v_sumsq,
        f_sumsq=f_sumsq,
        max_norm=max_norm,
        delaystep=delaystep,
        dtgrow=dtgrow,
        dtshrink=dtshrink,
        alphashrink=alphashrink,
        alpha0=alpha0,
        tmax=tmax,
        tmin=tmin,
        maxstep=maxstep,
    )


def fire2_step_coord_cell(
    positions: torch.Tensor,
    velocities: torch.Tensor,
    forces: torch.Tensor,
    cell: torch.Tensor,
    cell_velocities: torch.Tensor,
    cell_force: torch.Tensor,
    batch_idx: torch.Tensor,
    alpha: torch.Tensor,
    dt: torch.Tensor,
    nsteps_inc: torch.Tensor,
    *,
    vf: torch.Tensor | None = None,
    v_sumsq: torch.Tensor | None = None,
    f_sumsq: torch.Tensor | None = None,
    max_norm: torch.Tensor | None = None,
    delaystep: int = 60,
    dtgrow: float = 1.05,
    dtshrink: float = 0.75,
    alphashrink: float = 0.985,
    alpha0: float = 0.09,
    tmax: float = 0.08,
    tmin: float = 0.005,
    maxstep: float = 0.1,
) -> None:
    """Full FIRE2 variable-cell optimization step.

    Simultaneously relaxes atomic coordinates and cell degrees of freedom.
    Delegates to :func:`nvalchemiops.torch.fire2.fire2_step_coord_cell`.
    Modifies *positions*, *velocities*, *cell*, *cell_velocities*, *alpha*,
    *dt*, and *nsteps_inc* in-place.

    Parameters
    ----------
    positions : torch.Tensor
        Atomic positions ``[N, 3]``, float32 or float64.
    velocities : torch.Tensor
        Atomic velocities ``[N, 3]``, same dtype.
    forces : torch.Tensor
        Atomic forces ``[N, 3]``, same dtype.
    cell : torch.Tensor
        Per-system cell matrix ``[M, 3, 3]``, same dtype.
    cell_velocities : torch.Tensor
        Per-system cell velocity ḣ ``[M, 3, 3]``, same dtype.
    cell_force : torch.Tensor
        Per-system cell force (from stress) ``[M, 3, 3]``, same dtype.
        Compute via :func:`~nvalchemi.dynamics._ops.npt_nph.stress_to_cell_force`.
    batch_idx : torch.Tensor
        Per-atom system index ``[N]``, int32, non-decreasing.
    alpha : torch.Tensor
        Per-system FIRE2 mixing parameter ``[M]``, same dtype.
    dt : torch.Tensor
        Per-system adaptive timestep ``[M]``, same dtype.
    nsteps_inc : torch.Tensor
        Per-system consecutive positive-power step counter ``[M]``, int32.
    vf : torch.Tensor, optional
        Scratch buffer ``[M]`` for Σ(F·v); allocated if None.
    v_sumsq : torch.Tensor, optional
        Scratch buffer ``[M]`` for Σ(v·v); allocated if None.
    f_sumsq : torch.Tensor, optional
        Scratch buffer ``[M]`` for Σ(F·F); allocated if None.
    max_norm : torch.Tensor, optional
        Scratch buffer ``[M]`` for max force norm; allocated if None.
    delaystep, dtgrow, dtshrink, alphashrink, alpha0, tmax, tmin, maxstep
        FIRE2 hyperparameters (same semantics as :func:`fire2_step_coord`).
    """
    _fire2_coord_cell(
        positions,
        velocities,
        forces,
        cell,
        cell_velocities,
        cell_force,
        batch_idx,
        alpha,
        dt,
        nsteps_inc,
        vf=vf,
        v_sumsq=v_sumsq,
        f_sumsq=f_sumsq,
        max_norm=max_norm,
        delaystep=delaystep,
        dtgrow=dtgrow,
        dtshrink=dtshrink,
        alphashrink=alphashrink,
        alpha0=alpha0,
        tmax=tmax,
        tmin=tmin,
        maxstep=maxstep,
    )
