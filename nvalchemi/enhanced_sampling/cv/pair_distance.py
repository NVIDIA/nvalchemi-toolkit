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

:func:`pair_distance` is the P0 built-in geometric CV.  It supports:

* Non-periodic systems (``batch.cell`` is ``None`` or ``batch.pbc`` is all
  ``False``).
* Fully periodic and mixed-periodic systems via the minimum-image
  convention (MIC) for general triclinic cells.

Triclinic MIC algorithm
-----------------------
For a triclinic cell with lattice matrix ``A`` (rows = lattice vectors,
ASE convention), the fractional displacement is::

    df = (r_j - r_i) @ A^{-1}

We then apply the standard half-cell rounding::

    df -= torch.round(df)   # map to (−0.5, 0.5]

and convert back to Cartesian::

    dr_mic = df @ A

The distance is ``||dr_mic||``.

**Boundary note:** The half-cell tie (``|df_k| == 0.5`` exactly) maps
to ``+0.5`` or ``−0.5`` depending on floating-point rounding, producing
a discontinuity in the distance function exactly at the Wigner–Seitz
cell boundary.  This is the standard MIC behaviour and is shared by ASE,
LAMMPS, and PLUMED.  Tests are placed away from the half-cell tie; the
boundary behaviour is documented but not worked around.

torch.compile compatibility
---------------------------
This function is a compile target (``compile_biases=True``).  It uses
only standard PyTorch ops; there are no Python-level branches on tensor
values (the periodicity branch is on the *shape* / *bool* of ``pbc``,
which is resolved at trace time for fixed-pbc batches).  Gradient flow
through ``pair_distance`` for use inside :class:`ConservativeBias` is
fully supported.
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
        cell and PBC flags.
    atom_indices:
        * Shape ``[2]`` — selects the same atom pair ``(i, j)`` in every
          graph of the batch.
        * Shape ``[B, 2]`` — selects a different pair per graph.

        Atom indices are **local to each graph** (0-based within the
        graph, not global row indices in the batched position tensor).

    Returns
    -------
    Tensor
        Shape ``[B, 1]`` — pair distance in the same length units as
        ``batch.positions`` (Å).  Fully differentiable w.r.t.
        ``batch.positions`` and ``batch.cell``.

    Notes
    -----
    * MIC is applied independently per graph, using that graph's cell.
    * PBC flags (``batch.pbc``) are used to determine whether MIC is
      applied.  If **all** PBC flags for a graph are ``False``, or if
      ``batch.cell`` is ``None``, the straight Cartesian distance is
      used.
    * For graphs with mixed periodicity (e.g. ``pbc = [True, True, False]``),
      the MIC rounding is still applied in fractional coordinates; the
      non-periodic fractional components will map outside (−0.5, 0.5] in
      the direction(s) with ``pbc=False``, but the ``round()`` is only
      applied along periodic dimensions.
    """
    positions = batch.positions  # [N_total, 3]
    batch_ptr = batch.batch_ptr  # [B+1]
    B = batch.num_graphs

    # --- Resolve atom_indices to global row indices -----------------------
    if atom_indices.dim() == 1:
        # [2] → broadcast same pair across all graphs
        atom_indices = atom_indices.unsqueeze(0).expand(B, 2)  # [B, 2]

    # --- Bounds check (eager only; skipped under torch.compile) ----------
    # Without this, an out-of-range local index silently wraps into the next
    # graph's atom rows, producing a cross-system CV with no error.
    if not torch.compiler.is_compiling():
        atoms_per_graph = batch_ptr[1:] - batch_ptr[:-1]  # [B]
        for col, label in ((0, "atom_indices[…, 0]"), (1, "atom_indices[…, 1]")):
            idx = atom_indices[:, col]  # [B]
            neg = idx < 0
            if neg.any():
                bad_graphs = neg.nonzero(as_tuple=False).squeeze(-1).tolist()
                raise IndexError(
                    f"pair_distance: {label} has negative values for graph(s) "
                    f"{bad_graphs}: {idx[neg].tolist()}"
                )
            oob = idx >= atoms_per_graph
            if oob.any():
                bad_graphs = oob.nonzero(as_tuple=False).squeeze(-1).tolist()
                raise IndexError(
                    f"pair_distance: {label} is out of range for graph(s) "
                    f"{bad_graphs} — index {idx[oob].tolist()} >= "
                    f"graph size {atoms_per_graph[oob].tolist()}"
                )

    # batch_ptr[b] is the start of graph b; atom_indices[:, k] is local idx
    offsets = batch_ptr[:-1]  # [B]
    global_i = offsets + atom_indices[:, 0]  # [B]
    global_j = offsets + atom_indices[:, 1]  # [B]

    pos_i = positions[global_i]  # [B, 3]
    pos_j = positions[global_j]  # [B, 3]

    dr = pos_j - pos_i  # [B, 3], raw Cartesian displacement

    # --- Apply MIC for periodic systems -----------------------------------
    has_cell = getattr(batch, "cell", None) is not None and batch.cell is not None
    has_pbc = getattr(batch, "pbc", None) is not None and batch.pbc is not None

    if has_cell and has_pbc:
        dr = _apply_mic(dr, batch.cell, batch.pbc)

    dist = torch.linalg.vector_norm(dr, dim=-1, keepdim=True)  # [B, 1]
    return dist


# ---------------------------------------------------------------------------
# Internal MIC helper
# ---------------------------------------------------------------------------


def _apply_mic(dr: Tensor, cell: Tensor, pbc: Tensor) -> Tensor:
    """Apply the minimum-image convention for general triclinic cells.

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
    Componentwise fractional rounding (``df -= round(df)``) is only exact for
    orthogonal cells.  For skewed triclinic cells the Wigner–Seitz cell does
    not align with the fractional-coordinate axes, so the shortest image may
    require adding or subtracting a lattice vector even after rounding each
    component to (−0.5, 0.5].  Example: cell rows ``[[1,0,0],[0.9,0.1,0],
    [0,0,10]]``, fractional displacement ``[0.49,0.49,0]`` — rounding gives
    ≈ 0.932 Å but the true MIC vector (offset ``[0,−1,0]``) is ≈ 0.060 Å.

    This implementation performs an exhaustive search over all 27 lattice
    images (offsets in ``{−1, 0, +1}^3`` restricted to periodic dims) and
    returns the Cartesian vector with minimum norm.  The search is fully
    vectorised and passes ``torch.compile(fullgraph=True)``.
    """
    # Normalise shapes
    if cell.dim() == 4:
        cell = cell.squeeze(1)  # [B, 3, 3]
    if pbc.dim() == 3:
        pbc = pbc.squeeze(1)  # [B, 3]

    pbc_mask = pbc.to(dtype=cell.dtype)  # [B, 3]

    # Fractional displacement: df[b] = dr[b] @ cell[b]^{-1}
    cell_inv = torch.linalg.inv(cell)  # [B, 3, 3]
    df = torch.bmm(dr.unsqueeze(1), cell_inv).squeeze(1)  # [B, 3]

    # Initial rounding: map each periodic component to (−0.5, 0.5].
    # For non-periodic dims the component is unchanged.
    df_rounded = df - torch.round(df) * pbc_mask  # [B, 3]

    # --- Exhaustive 27-image search -----------------------------------------
    # Build all 27 offset vectors {-1, 0, 1}^3 and mask non-periodic dims.
    coords = torch.tensor([-1.0, 0.0, 1.0], device=dr.device, dtype=dr.dtype)
    gi, gj, gk = torch.meshgrid(coords, coords, coords, indexing="ij")
    all_offsets = torch.stack(
        [gi.flatten(), gj.flatten(), gk.flatten()], dim=-1
    )  # [27, 3]

    # [B, 27, 3]: apply pbc_mask so non-periodic dims are never shifted
    offsets_masked = all_offsets[None] * pbc_mask[:, None, :]

    # Candidate fractional displacements and their Cartesian counterparts
    df_cands = df_rounded[:, None, :] + offsets_masked  # [B, 27, 3]
    dr_cands = torch.einsum("bki,bij->bkj", df_cands, cell)  # [B, 27, 3]

    # Select the image with minimum squared Cartesian distance (avoid sqrt)
    dist_sq = (dr_cands * dr_cands).sum(dim=-1)  # [B, 27]
    best = dist_sq.argmin(dim=-1)[:, None, None].expand(-1, 1, 3)  # [B, 1, 3]
    dr_mic = dr_cands.gather(1, best).squeeze(1)  # [B, 3]

    return dr_mic
