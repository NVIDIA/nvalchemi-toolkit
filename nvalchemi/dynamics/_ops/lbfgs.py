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
PyTorch bindings for the L-BFGS optimizer.

Delegates to :mod:`nvalchemiops.torch.lbfgs`, which registers its own
``torch.library`` custom ops.

Functions
---------
lbfgs_step_coord
    L-BFGS coordinate-only step.
lbfgs_step_coord_cell
    L-BFGS variable-cell step.
"""

from __future__ import annotations

import torch
from nvalchemiops.torch.lbfgs import (
    LBFGSCellState,
    LBFGSState,
    lbfgs_prepare_cell_state,
    lbfgs_prepare_state,
)
from nvalchemiops.torch.lbfgs import (
    lbfgs_step_coord as _lbfgs_coord,
)
from nvalchemiops.torch.lbfgs import (
    lbfgs_step_coord_cell as _lbfgs_coord_cell,
)

__all__ = [
    "LBFGSCellState",
    "LBFGSState",
    "lbfgs_prepare_cell_state",
    "lbfgs_prepare_state",
    "lbfgs_step_coord",
    "lbfgs_step_coord_cell",
]


def lbfgs_step_coord(
    positions: torch.Tensor,
    forces: torch.Tensor,
    state: LBFGSState,
    batch_idx: torch.Tensor,
    *,
    maxstep: float = 0.2,
    curvature_eps: float | None = None,
) -> None:
    """Full L-BFGS coordinate-only step.

    Delegates to :func:`nvalchemiops.torch.lbfgs.lbfgs_step_coord`.
    Modifies *positions* and *state* in-place.

    Parameters
    ----------
    positions : torch.Tensor
        Atomic positions ``[N, 3]``, float32 or float64.
    forces : torch.Tensor
        Atomic forces ``[N, 3]``, same dtype.
    state : LBFGSState
        Optimizer state sized for ``N`` degrees of freedom.
    batch_idx : torch.Tensor
        Per-atom system index ``[N]``, int32, non-decreasing.
    maxstep : float
        Maximum per-atom displacement per step.  Default 0.2.
    curvature_eps : float, optional
        Curvature-pair acceptance floor; ``None`` picks by dtype.
    """
    _lbfgs_coord(
        positions,
        forces,
        state,
        batch_idx,
        maxstep=maxstep,
        curvature_eps=curvature_eps,
    )


def lbfgs_step_coord_cell(
    positions: torch.Tensor,
    cell: torch.Tensor,
    forces: torch.Tensor,
    stress: torch.Tensor,
    state: LBFGSState,
    cell_state: LBFGSCellState,
    batch_idx: torch.Tensor,
    *,
    maxstep: float = 0.2,
    curvature_eps: float | None = None,
) -> None:
    """Full L-BFGS variable-cell step.

    Delegates to :func:`nvalchemiops.torch.lbfgs.lbfgs_step_coord_cell`.
    Modifies *positions*, *cell*, *state* and *cell_state* in-place.

    Parameters
    ----------
    positions : torch.Tensor
        Atomic positions ``[N, 3]``, float32 or float64.
    cell : torch.Tensor
        Per-system aligned cell ``[M, 3, 3]``, same dtype.
    forces : torch.Tensor
        Atomic forces ``[N, 3]``, same dtype.
    stress : torch.Tensor
        Per-system tensile-positive Cauchy stress ``[M, 3, 3]``, same dtype.
    state : LBFGSState
        Optimizer state sized for ``N + 2M`` degrees of freedom.
    cell_state : LBFGSCellState
        Variable-cell chart and scratch.
    batch_idx : torch.Tensor
        Per-atom system index ``[N]``, int32, non-decreasing.
    maxstep, curvature_eps
        As :func:`lbfgs_step_coord`.
    """
    _lbfgs_coord_cell(
        positions,
        cell,
        forces,
        stress,
        state,
        cell_state,
        batch_idx,
        maxstep=maxstep,
        curvature_eps=curvature_eps,
    )
