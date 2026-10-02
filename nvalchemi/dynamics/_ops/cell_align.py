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
PyTorch binding for cell alignment to lower-triangular form.

Wraps :func:`nvalchemiops.dynamics.utils.align_cell` as a
``torch.library.custom_op``, enabling correct behaviour under
``torch.compile`` and PyTorch's autograd infrastructure.

Functions
---------
align_cell
    Align periodic cells to lower-triangular form and rotate positions
    to preserve fractional coordinates.
cell_alignment_offenders
    Per-system check for whether a cell is already in ``align_cell``'s
    canonical form; the one shared criterion both :class:`AlignCellHook`
    and :class:`LBFGSVariableCell` use so they can't disagree.
"""

from __future__ import annotations

import torch
import torch.library
import warp as wp
from nvalchemiops.dynamics.utils import align_cell as _align_cell

from nvalchemi.dynamics._ops._bridge import _mat_type, _vec_type

__all__ = ["align_cell", "ALIGN_ATOL", "cell_alignment_offenders"]

# Must match nvalchemiops.dynamics.optimizers.lbfgs._check_cell_is_aligned's
# `atol` default — not dtype-dependent there, so not here either.
ALIGN_ATOL = 1e-10


# ---------------------------------------------------------------------------
# Internal custom op
# ---------------------------------------------------------------------------


@torch.library.custom_op(
    "nvalchemi::align_cell", mutates_args={"positions", "cell", "transform"}
)
def _align_cell_op(
    positions: torch.Tensor,
    cell: torch.Tensor,
    transform: torch.Tensor,
    batch_idx: torch.Tensor,
) -> None:
    """Align cells to lower-triangular form and transform positions in-place.

    Parameters
    ----------
    positions : torch.Tensor
        Atomic positions ``[N, 3]``, float32 or float64.
    cell : torch.Tensor
        Per-system cell matrices ``[M, 3, 3]``, same dtype.  Overwritten
        with the aligned (lower-triangular) cells.
    transform : torch.Tensor
        Per-system rotation matrices ``[M, 3, 3]``, same dtype.  Must be
        initialized to identity on entry: the kernel leaves degenerate
        (zero-volume) systems untouched.  Overwritten with the rotation
        ``R`` such that ``positions_new[i] = R[sys(i)] @ positions_old[i]``
        and ``cell_new = cell_old @ R^T``; the same ``R`` rotates any other
        Cartesian vector/tensor field (forces, stress) into the new frame.
    batch_idx : torch.Tensor
        Per-atom system index ``[N]``, int32, non-decreasing.
    """
    dtype = positions.dtype
    vec_t = _vec_type(dtype)
    mat_t = _mat_type(dtype)

    cell_c = cell.contiguous()
    transform_c = transform.contiguous()

    wp_device = wp.device_from_torch(positions.device)
    _align_cell(
        wp.from_torch(positions.contiguous(), dtype=vec_t),
        wp.from_torch(cell_c, dtype=mat_t),
        wp.from_torch(transform_c, dtype=mat_t),
        batch_idx=wp.from_torch(
            batch_idx.to(dtype=torch.int32).contiguous(), dtype=wp.int32
        ),
        device=wp_device,
    )
    # Write aligned cell and rotation back into the original tensors
    cell.copy_(cell_c)
    transform.copy_(transform_c)


@_align_cell_op.register_fake
def _align_cell_op_fake(
    positions: torch.Tensor,
    cell: torch.Tensor,
    transform: torch.Tensor,
    batch_idx: torch.Tensor,
) -> None:
    pass


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def align_cell(
    positions: torch.Tensor,
    cell: torch.Tensor,
    batch_idx: torch.Tensor | None = None,
    *,
    transform: torch.Tensor | None = None,
) -> torch.Tensor:
    """Align periodic cells to lower-triangular form and rotate positions.

    This is a one-time preprocessing step before variable-cell optimization.
    The cell is transformed to the standard lower-triangular form, and
    positions are rotated to maintain their fractional coordinates.

    Parameters
    ----------
    positions : torch.Tensor
        Atomic positions ``[N, 3]``, float32 or float64.  Modified in-place.
    cell : torch.Tensor
        Per-system cell matrices ``[M, 3, 3]``, same dtype.  Overwritten
        with aligned (lower-triangular) cells.
    batch_idx : torch.Tensor, optional
        Per-atom system index ``[N]``, int32.  If ``None``, all atoms are
        assumed to belong to a single system.
    transform : torch.Tensor, optional
        Output buffer ``[M, 3, 3]`` for the per-system rotation matrices.
        If given, it is reset to identity and overwritten in-place; if
        omitted, an internal buffer is allocated.  Reuse the returned
        rotation to keep any other Cartesian vector/tensor field (forces,
        stress) consistent with the new frame — see :func:`_align_cell_op`
        for the exact convention.

    Returns
    -------
    torch.Tensor
        The rotation matrices ``[M, 3, 3]`` applied to *cell* and
        *positions* (identity for systems that needed no rotation).
    """
    if batch_idx is None:
        batch_idx = torch.zeros(
            positions.shape[0], dtype=torch.int32, device=positions.device
        )
    if transform is None:
        transform = torch.eye(3, dtype=cell.dtype, device=cell.device).repeat(
            cell.shape[0], 1, 1
        )
    else:
        transform.copy_(torch.eye(3, dtype=cell.dtype, device=cell.device))
    _align_cell_op(positions, cell, transform, batch_idx)
    return transform


def cell_alignment_offenders(
    cell: torch.Tensor, atol: float = ALIGN_ATOL
) -> torch.Tensor:
    r"""Per-system bool: ``True`` where *cell* is not already aligned.

    "Aligned" means exactly what :func:`align_cell` would leave unchanged:
    the strict upper triangle is (numerically) zero, the cell is
    right-handed (positive determinant), *and* the diagonal is
    non-negative.  All three are necessary — the kernel's canonical output
    always has a non-negative diagonal (``a``, ``b sin gamma``, ``c3`` are
    lengths/sqrt terms by construction), and none of the first two checks
    alone catches every matrix that violates it:

    - A matrix that is already triangular but left-handed (e.g.
      ``diag(-5, 5, 5)``, determinant :math:`-125`) would still be flipped
      by :func:`align_cell`, even though its upper triangle is trivially
      zero — caught by the determinant check.
    - A matrix that is triangular *and* right-handed can still have a
      negative diagonal entry if an even number of them are negative (e.g.
      ``diag(-5, -5, 5)``, determinant :math:`+125`): the determinant check
      alone calls this aligned, but :func:`align_cell` rotates it to
      ``diag(5, 5, 5)`` — a 180-degree rotation about the third axis, not a
      no-op — so it needs its own check.

    A non-positive determinant also catches degenerate (zero-volume)
    cells, which :func:`align_cell` leaves untouched but which are never a
    valid reference cell regardless.

    This is the one criterion :class:`~nvalchemi.dynamics.hooks.AlignCellHook`
    (to decide whether calling :func:`align_cell` would be a no-op) and
    :meth:`~nvalchemi.dynamics.optimizers.lbfgs.LBFGSVariableCell._reference_cells`
    (to validate an admitted cell) both use, so the two can never disagree
    about what counts as aligned — a cell the hook decides to skip can never
    fail admission, and vice versa.

    Parameters
    ----------
    cell : torch.Tensor
        Per-system cell matrices ``[M, 3, 3]``.
    atol : float, optional
        Absolute tolerance on the strict-upper-triangle entries.  Default
        matches ``nvalchemiops``' own (dtype-independent) check.

    Returns
    -------
    torch.Tensor
        Boolean ``[M]``; ``True`` where the system needs realignment.
    """
    skew = torch.triu(cell, 1).abs().amax(dim=(-2, -1))
    not_triangular = skew > atol
    not_right_handed = torch.linalg.det(cell) <= 0
    negative_diagonal = (torch.diagonal(cell, dim1=-2, dim2=-1) < 0).any(dim=-1)
    return not_triangular | not_right_handed | negative_diagonal
