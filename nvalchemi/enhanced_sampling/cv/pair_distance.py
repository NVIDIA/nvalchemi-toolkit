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
    """
    # Normalise shapes
    if cell.dim() == 4:
        cell = cell.squeeze(1)  # [B, 3, 3]
    if pbc.dim() == 3:
        pbc = pbc.squeeze(1)  # [B, 3]

    # Convert pbc to float mask on the correct device/dtype
    pbc_mask = pbc.to(dtype=cell.dtype)  # [B, 3]

    # Fractional displacement: dr @ A^{-1}
    # cell[b] has shape [3, 3]; A^{-1} = cell^{-T} when rows=lattice vecs.
    # torch.linalg.solve(A.T, dr.T) gives x s.t. A.T x = dr.T
    # Equivalently: df = dr @ A^{-1} = (A^{-T} dr^T)^T
    cell_inv = torch.linalg.inv(cell)  # [B, 3, 3]  (A^{-1})
    # df[b] = dr[b] @ cell_inv[b];  batched matmul: [B, 1, 3] @ [B, 3, 3]
    df = torch.bmm(dr.unsqueeze(1), cell_inv).squeeze(1)  # [B, 3]

    # Apply half-cell rounding only along periodic dimensions.
    # For periodic dims (pbc_mask=1): df_mic = df - round(df)  → maps to (−0.5, 0.5]
    # For non-periodic dims (pbc_mask=0): df_mic = df          → unchanged
    df_mic = df - torch.round(df) * pbc_mask  # [B, 3]

    # Back to Cartesian: dr_mic[b] = df_mic[b] @ cell[b]
    dr_mic = torch.bmm(df_mic.unsqueeze(1), cell).squeeze(1)  # [B, 3]
    return dr_mic
