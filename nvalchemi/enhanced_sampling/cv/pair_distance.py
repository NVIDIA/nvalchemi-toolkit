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
"""Differentiable pair-distance collective variable.

:func:`pair_distance` supports:

* Non-periodic systems (``batch.cell`` is ``None`` or ``batch.pbc`` is all
  ``False``).
* Periodic and mixed-periodic systems via the minimum-image convention (MIC)
  for **Minkowski-reduced** triclinic cells (see requirement below).

Scope: Minkowski-reduced MIC, not general triclinic MIC
--------------------------------------------------------
This is a **reduced-cell MIC implementation**.  It is *not* a general
triclinic MIC implementation.  The 27-image exhaustive search (offsets in
``{−1, 0, +1}³``) is correct only when the cell satisfies the Minkowski
reduction conditions.  For unreduced cells the minimum-image offset can
exceed ±1 in one or more fractional components, and the search silently
returns a longer-than-minimum image.

True general triclinic MIC (arbitrary unreduced cells, implemented via LLL
lattice reduction or an extended image search with a data-dependent radius)
is **not yet implemented** — its interaction with the strain-based virial
computation in :class:`ConservativeBias` adds non-trivial complexity.

Minkowski reduction condition
-----------------------------
For every pair of periodic lattice vectors ``(aᵢ, aⱼ)`` with ``i ≠ j``:

.. math::

    |\\mathbf{a}_i \\cdot \\mathbf{a}_j| \\le
    \\tfrac{1}{2}\\,\\min(|\\mathbf{a}_i|^2,\\,|\\mathbf{a}_j|^2)

When this fails, the search returns the wrong image.  Counter-example:
cell ``[[1,0,0],[10,0.1,0],[0,0,10]]``, fractional displacement
``[0,0.49,0]`` — the 27-image search returns ≈ 3.9 Å, but the true image
(offset ``[−5,0,0]``) is ≈ 0.11 Å.

:func:`pair_distance` checks this condition at call time **in eager mode
only** and raises ``ValueError`` for non-reduced cells.

.. warning::

    Under ``torch.compile`` the check is skipped (guarded by
    ``torch.compiler.is_compiling()``).  In compiled mode the caller is
    **solely responsible** for supplying Minkowski-reduced cells.  Passing
    an unreduced cell in compiled mode produces wrong distances with no
    error.  Pre-reduce cells with a Niggli or LLL algorithm (e.g.
    ``ASE: atoms.get_cell().niggli_reduce()``) before simulation.

Triclinic MIC algorithm
-----------------------
For a reduced cell with lattice matrix ``A`` (rows = lattice vectors,
ASE convention)::

    df          = (r_j − r_i) @ A⁻¹          # fractional displacement
    df_rounded  = df − round(df) × pbc_mask   # map to (−0.5, 0.5]
    candidates  = df_rounded + n,  n ∈ {−1,0,+1}³ × pbc_mask
    dr_mic      = argmin_n |candidates @ A|   # shortest image

torch.compile compatibility
---------------------------
Shape-based branches (periodicity, cell presence) resolve at trace time.
Gradient flow through ``pair_distance`` for use inside
:class:`ConservativeBias` is fully supported.  The Minkowski check and
bounds check are guarded by ``torch.compiler.is_compiling()`` and do not
appear in the compiled graph.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor

if TYPE_CHECKING:
    from nvalchemi.data import Batch

__all__ = ["pair_distance"]


def pair_distance(batch: Batch, atom_indices: Tensor) -> Tensor:
    """Differentiable pair distance(s) as a collective variable.

    Parameters
    ----------
    batch:
        Current ``Batch`` containing atomic positions and (optionally)
        cell and PBC flags.  When a periodic cell is present it must be
        **Minkowski-reduced** — see module docstring for details.
    atom_indices:
        * Shape ``[2]`` — selects the same atom pair ``(i, j)`` in every
          graph of the batch.
        * Shape ``[B, 2]`` — selects a different pair per graph.

        Indices are **local to each graph** (0-based within the graph, not
        global row indices in the batched position tensor).

    Returns
    -------
    Tensor
        Shape ``[B, 1]`` — pair distance in the same length unit as
        ``batch.positions`` (Å).  Fully differentiable w.r.t.
        ``batch.positions`` and ``batch.cell``.

    Raises
    ------
    ValueError
        If ``atom_indices`` is not shape ``[2]`` or ``[B, 2]``, or is not
        an integer dtype (eager mode only).
    IndexError
        If any local atom index is negative or >= the graph's atom count
        (eager mode only).
    ValueError
        If any periodic cell is not Minkowski-reduced (eager mode only).
        This check is **skipped under** ``torch.compile``; see module
        docstring for the compiled-mode caller responsibility.

    Notes
    -----
    Non-periodic graphs with an explicit cell
        In **eager mode**, MIC is skipped entirely when ``batch.pbc`` is
        all-False, so a degenerate cell (e.g. zeros) is safe.  In
        **compiled mode**, ``bool(pbc.any())`` cannot be evaluated without
        a graph break, so MIC is entered whenever ``cell`` and ``pbc`` are
        both present.  Compiled callers must therefore supply a
        non-degenerate cell (or omit it entirely, ``cell=None``) for
        non-periodic graphs.
    """
    positions = batch.positions  # [N_total, 3]
    batch_ptr = batch.batch_ptr  # [B+1]
    B = batch.num_graphs

    # --- Eager-only shape / dtype check on atom_indices ------------------
    # Must come BEFORE the dim()==1 broadcast so wrong shapes are caught,
    # not silently coerced.  E.g. [1] would expand to [[0,0]] (self-pair)
    # and [B,3] would silently drop the third column.
    if not torch.compiler.is_compiling():
        _validate_atom_indices(atom_indices, B)

    # --- Resolve atom_indices to global row indices -----------------------
    if atom_indices.dim() == 1:
        atom_indices = atom_indices.unsqueeze(0).expand(B, 2)  # [B, 2]

    # --- Eager-only bounds validation ------------------------------------
    if not torch.compiler.is_compiling():
        # Bounds check: catch silent cross-graph wrapping before any indexing.
        atoms_per_graph = batch_ptr[1:] - batch_ptr[:-1]  # [B]
        for col, label in ((0, "atom_indices[…, 0]"), (1, "atom_indices[…, 1]")):
            idx = atom_indices[:, col]
            neg = idx < 0
            if neg.any():
                bad = neg.nonzero(as_tuple=False).squeeze(-1).tolist()
                raise IndexError(
                    f"pair_distance: {label} has negative values for graph(s) "
                    f"{bad}: {idx[neg].tolist()}"
                )
            oob = idx >= atoms_per_graph
            if oob.any():
                bad = oob.nonzero(as_tuple=False).squeeze(-1).tolist()
                raise IndexError(
                    f"pair_distance: {label} is out of range for graph(s) "
                    f"{bad} — index {idx[oob].tolist()} >= "
                    f"graph size {atoms_per_graph[oob].tolist()}"
                )

    offsets = batch_ptr[:-1]  # [B]
    global_i = offsets + atom_indices[:, 0]  # [B]
    global_j = offsets + atom_indices[:, 1]  # [B]

    pos_i = positions[global_i]  # [B, 3]
    pos_j = positions[global_j]  # [B, 3]
    dr = pos_j - pos_i  # [B, 3], raw Cartesian displacement

    # --- Apply MIC for periodic systems ----------------------------------
    has_cell = getattr(batch, "cell", None) is not None and batch.cell is not None
    has_pbc = getattr(batch, "pbc", None) is not None and batch.pbc is not None

    # In eager mode, also require at least one True pbc flag before calling
    # _apply_mic.  Without this guard, a batch with cell=<degenerate> and
    # pbc=all-False would reach torch.linalg.inv and raise LinAlgError.
    #
    # In compiled mode, bool(pbc.any()) would force a data-dependent Python
    # branch that breaks fullgraph=True.  We skip the guard there and rely
    # on pbc_mask (all-zeros for all-False pbc) to make the MIC computation
    # a mathematical identity for non-periodic graphs.  Compiled callers
    # must therefore supply a non-degenerate cell (or cell=None) for
    # non-periodic graphs; a degenerate cell still causes LinAlgError.
    any_periodic = (
        has_cell
        and has_pbc
        and (torch.compiler.is_compiling() or bool(batch.pbc.any()))
    )

    if any_periodic:
        if not torch.compiler.is_compiling():
            _check_minkowski_reduced(batch.cell, batch.pbc)
        dr = _apply_mic(dr, batch.cell, batch.pbc)

    return torch.linalg.vector_norm(dr, dim=-1, keepdim=True)  # [B, 1]


# ---------------------------------------------------------------------------
# atom_indices validation
# ---------------------------------------------------------------------------

_INTEGER_DTYPES = frozenset(
    {
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    }
)


def _validate_atom_indices(atom_indices: Tensor, B: int) -> None:
    """Raise ``ValueError`` for malformed ``atom_indices`` (eager mode only).

    Accepted shapes
    ---------------
    * ``[2]``    — shared pair; broadcast to every graph.
    * ``[B, 2]`` — one pair per graph.

    Rejected (with clear error messages)
    -------------------------------------
    * Wrong number of dimensions (not 1-D or 2-D).
    * 1-D tensor whose length is not exactly 2.  A length-1 tensor such as
      ``torch.tensor([0])`` would otherwise silently expand to ``[[0, 0]]``
      (a self-distance), not raise.
    * 2-D tensor whose second dimension is not exactly 2.  A ``[B, 3]``
      tensor would otherwise silently drop the third column.
    * 2-D tensor whose first dimension does not match the batch size ``B``.
    * Non-integer dtype.  Float indices would silently be used as memory
      offsets after casting by the indexing operation.

    Parameters
    ----------
    atom_indices:
        The tensor to validate.
    B:
        Number of graphs in the current batch.
    """
    # dtype check
    if atom_indices.dtype not in _INTEGER_DTYPES:
        raise ValueError(
            f"pair_distance: atom_indices must have an integer dtype, "
            f"got {atom_indices.dtype}.  Use e.g. torch.tensor([i, j]) "
            f"(default int64) or pass dtype=torch.long explicitly."
        )

    ndim = atom_indices.dim()
    shape = tuple(atom_indices.shape)

    if ndim == 1:
        if shape[0] != 2:
            raise ValueError(
                f"pair_distance: 1-D atom_indices must have exactly 2 elements "
                f"(shape [2] for a shared pair), got shape {shape}.  "
                f"A length-1 tensor would silently produce a self-distance."
            )
    elif ndim == 2:
        if shape[1] != 2:
            raise ValueError(
                f"pair_distance: 2-D atom_indices must have shape [B, 2], "
                f"got {shape}.  The second dimension must be exactly 2 "
                f"(atom i and atom j); extra columns are not allowed."
            )
        if shape[0] != B:
            raise ValueError(
                f"pair_distance: 2-D atom_indices has shape {shape} but the "
                f"batch has B={B} graphs.  The first dimension must equal B."
            )
    else:
        raise ValueError(
            f"pair_distance: atom_indices must be 1-D (shape [2]) or "
            f"2-D (shape [B, 2]), got {ndim}-D tensor with shape {shape}."
        )


# ---------------------------------------------------------------------------
# Minkowski-reduction check
# ---------------------------------------------------------------------------


def _check_minkowski_reduced(cell: Tensor, pbc: Tensor) -> None:
    """Raise ``ValueError`` if any periodic cell pair violates the Minkowski condition.

    This check is an **eager-mode guard only**.  It is never called under
    ``torch.compile`` (guarded by ``torch.compiler.is_compiling()`` in the
    caller).  Compiled callers are responsible for supplying reduced cells;
    no error is raised if a non-reduced cell is used in compiled mode.

    The 27-image MIC search returns the true minimum-image vector only for
    Minkowski-reduced cells.  For every pair of periodic lattice vectors
    ``(aᵢ, aⱼ)`` with ``i ≠ j``:

    .. math::

        |\\mathbf{a}_i \\cdot \\mathbf{a}_j|
        \\le \\tfrac{1}{2}\\,\\min(|\\mathbf{a}_i|^2,\\,|\\mathbf{a}_j|^2)

    When this fails, the minimum-image offset can exceed ±1 in some
    fractional component and the search silently returns the wrong image.

    Parameters
    ----------
    cell:
        Lattice matrices, shape ``[B, 3, 3]`` or ``[B, 1, 3, 3]``.
    pbc:
        Periodicity flags, shape ``[B, 3]`` or ``[B, 1, 3]``.
    """
    if cell.dim() == 4:
        cell = cell.squeeze(1)
    if pbc.dim() == 3:
        pbc = pbc.squeeze(1)

    for i in range(3):
        for j in range(i + 1, 3):
            # Only enforce for pairs of dimensions that are BOTH periodic.
            both_periodic = pbc[:, i] & pbc[:, j]  # [B] bool
            if not both_periodic.any():
                continue

            ai = cell[:, i, :]  # [B, 3]
            aj = cell[:, j, :]  # [B, 3]
            dot_abs = (ai * aj).sum(-1).abs()  # [B]
            norm_sq_i = (ai * ai).sum(-1)  # [B]
            norm_sq_j = (aj * aj).sum(-1)  # [B]
            threshold = 0.5 * torch.minimum(norm_sq_i, norm_sq_j)  # [B]

            violated = both_periodic & (dot_abs > threshold)
            if violated.any():
                bad = violated.nonzero(as_tuple=False).squeeze(-1).tolist()
                raise ValueError(
                    f"pair_distance: the cell for graph(s) {bad} is not "
                    f"Minkowski-reduced: lattice vectors a[{i}] and a[{j}] satisfy "
                    f"|a[{i}]·a[{j}]| > 0.5·min(|a[{i}]|², |a[{j}]|²).  "
                    f"The 27-image MIC search is only guaranteed correct for "
                    f"Minkowski-reduced cells.  Pre-reduce the cell using a Niggli "
                    f"or LLL algorithm (e.g. ASE niggli_reduce) before simulation."
                )


# ---------------------------------------------------------------------------
# MIC implementation
# ---------------------------------------------------------------------------


def _apply_mic(dr: Tensor, cell: Tensor, pbc: Tensor) -> Tensor:
    """Apply the minimum-image convention via an exhaustive 27-image search.

    Parameters
    ----------
    dr:
        Cartesian displacement vectors, shape ``[B, 3]``.
    cell:
        Lattice matrices, shape ``[B, 3, 3]`` or ``[B, 1, 3, 3]``.
        Rows are lattice vectors (ASE convention).
    pbc:
        Periodicity flags per dimension, shape ``[B, 3]`` or ``[B, 1, 3]``.

    Returns
    -------
    Tensor
        MIC-corrected displacement vectors, shape ``[B, 3]``.

    Notes
    -----
    The cell must be Minkowski-reduced; see :func:`_check_minkowski_reduced`.
    That check is performed in :func:`pair_distance` before this function is
    called, so it is not repeated here.
    """
    if cell.dim() == 4:
        cell = cell.squeeze(1)
    if pbc.dim() == 3:
        pbc = pbc.squeeze(1)

    pbc_mask = pbc.to(dtype=cell.dtype)  # [B, 3]

    # Fractional displacement
    cell_inv = torch.linalg.inv(cell)  # [B, 3, 3]
    df = torch.bmm(dr.unsqueeze(1), cell_inv).squeeze(1)  # [B, 3]

    # Initial half-cell rounding (periodic dims only)
    df_rounded = df - torch.round(df) * pbc_mask  # [B, 3]

    # Exhaustive 27-image search over offsets in {-1, 0, +1}³
    coords = torch.tensor([-1.0, 0.0, 1.0], device=dr.device, dtype=dr.dtype)
    gi, gj, gk = torch.meshgrid(coords, coords, coords, indexing="ij")
    all_offsets = torch.stack(
        [gi.flatten(), gj.flatten(), gk.flatten()], dim=-1
    )  # [27, 3]

    offsets_masked = all_offsets[None] * pbc_mask[:, None, :]  # [B, 27, 3]
    df_cands = df_rounded[:, None, :] + offsets_masked  # [B, 27, 3]
    dr_cands = torch.einsum("bki,bij->bkj", df_cands, cell)  # [B, 27, 3]

    dist_sq = (dr_cands * dr_cands).sum(dim=-1)  # [B, 27]
    best = dist_sq.argmin(dim=-1)[:, None, None].expand(-1, 1, 3)  # [B, 1, 3]
    return dr_cands.gather(1, best).squeeze(1)  # [B, 3]
