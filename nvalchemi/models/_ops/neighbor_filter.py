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
"""Utilities for filtering neighbor lists to a tighter cutoff.

This module provides functions to filter pre-built neighbor lists (both dense
MATRIX format and sparse COO format) down to a smaller model cutoff, and a
unified entry point ``prepare_neighbors_for_model`` that handles format
conversion as needed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor

from nvalchemi.models.base import NeighborListFormat

if TYPE_CHECKING:
    from nvalchemi.data import Batch

__all__ = [
    "filter_neighbor_matrix",
    "filter_neighbor_list",
    "neighbor_matrix_to_list",
    "prepare_neighbors_for_model",
]


def filter_neighbor_matrix(
    positions: Tensor,
    cutoff: float,
    neighbor_matrix: Tensor,
    num_neighbors: Tensor,
    fill_value: int,
    cell: Tensor | None = None,
    neighbor_shifts: Tensor | None = None,
    batch_idx: Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor | None]:
    """Filter a dense neighbor matrix to a tighter cutoff.

    Parameters
    ----------
    positions : Tensor
        Atom positions, shape ``(N, 3)``.
    cutoff : float
        Tighter cutoff radius to filter down to.
    neighbor_matrix : Tensor
        Dense neighbor indices, shape ``(N, K)``, int32.  Entries equal to
        ``fill_value`` are treated as empty slots.
    num_neighbors : Tensor
        Number of valid neighbors per atom, shape ``(N,)``, int32.
    fill_value : int
        Sentinel value used to mark empty slots in ``neighbor_matrix``.
    cell : Tensor | None
        Unit cell matrices.  Either ``(3, 3)`` (single cell) or ``(B, 3, 3)``
        (per-system cells).  Required for PBC distance computation.
    neighbor_shifts : Tensor | None
        Integer unit-cell shift vectors, shape ``(N, K, 3)``, int32.
        Required for PBC; ignored when ``cell`` is ``None``.
    batch_idx : Tensor | None
        System index for each atom, shape ``(N,)``.  Required when ``cell``
        has shape ``(B, 3, 3)``.

    Returns
    -------
    neighbor_matrix : Tensor
        Filtered dense neighbor matrix, same shape ``(N, K)``.
    num_neighbors : Tensor
        Updated neighbor counts, shape ``(N,)``.
    neighbor_shifts : Tensor | None
        Filtered shift vectors ``(N, K, 3)`` if shifts were provided, else
        ``None``.
    """
    N, K = neighbor_matrix.shape
    dtype = positions.dtype

    # Boolean mask: True for valid (non-fill) entries.
    valid = neighbor_matrix < fill_value  # (N, K)

    # Clamp j indices to [0, N-1] for safe indexing; invalid slots are masked later.
    j_safe = neighbor_matrix.clamp(0, N - 1)  # (N, K)

    # Gather neighbour positions.
    pos_j = positions[j_safe]  # (N, K, 3)
    pos_i = positions.unsqueeze(1).expand_as(pos_j)  # (N, K, 3)
    delta = pos_j - pos_i  # (N, K, 3)

    # Add PBC Cartesian shift: shift_vec = shifts @ cell
    if cell is not None and neighbor_shifts is not None:
        shifts_f = neighbor_shifts.to(dtype)  # (N, K, 3)
        if cell.dim() == 2:
            # Single cell (3, 3) — broadcast over all atoms and neighbours.
            atom_cell = cell.unsqueeze(0).unsqueeze(0).expand(N, K, 3, 3)  # (N,K,3,3)
        else:
            # Per-system cells (B, 3, 3).
            if batch_idx is None:
                raise ValueError("batch_idx is required when cell has shape (B, 3, 3).")
            atom_cell = cell[batch_idx]  # (N, 3, 3)
            atom_cell = atom_cell.unsqueeze(1).expand(N, K, 3, 3)  # (N, K, 3, 3)
        # Cartesian shift = einsum('nks,nksd->nkd', shifts_f, atom_cell)
        cart_shift = torch.einsum("nks,nksd->nkd", shifts_f, atom_cell)
        delta = delta + cart_shift

    dist2 = (delta * delta).sum(dim=-1)  # (N, K)
    within_cutoff = dist2 < (cutoff * cutoff)  # (N, K)

    # Combined keep mask: valid slot AND within cutoff.
    keep = valid & within_cutoff  # (N, K)

    # Defragment: argsort on (~keep).long() with stable=True so kept entries
    # (value 0) sort before discarded entries (value 1).
    sort_key = (~keep).long()  # (N, K)
    order = sort_key.argsort(dim=1, stable=True)  # (N, K)

    # Reorder neighbour matrix and fill discarded slots with fill_value.
    nm_sorted = neighbor_matrix.gather(1, order)  # (N, K)
    keep_sorted = keep.gather(1, order)  # (N, K)
    nm_filtered = nm_sorted.masked_fill(~keep_sorted, fill_value)

    # Update num_neighbors.
    new_num_neighbors = keep.sum(dim=1).to(num_neighbors.dtype)  # (N,)

    # Reorder shift vectors if present.
    out_shifts: Tensor | None = None
    if neighbor_shifts is not None:
        order_3d = order.unsqueeze(-1).expand_as(neighbor_shifts)  # (N, K, 3)
        shifts_sorted = neighbor_shifts.gather(1, order_3d)
        out_shifts = shifts_sorted.masked_fill(~keep_sorted.unsqueeze(-1), 0)

    return nm_filtered, new_num_neighbors, out_shifts


def filter_neighbor_list(
    positions: Tensor,
    cutoff: float,
    neighbor_list: Tensor,
    neighbor_ptr: Tensor,
    cell: Tensor | None = None,
    unit_shifts: Tensor | None = None,
    batch_idx: Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor | None]:
    """Filter a sparse COO neighbor list to a tighter cutoff.

    Parameters
    ----------
    positions : Tensor
        Atom positions, shape ``(N, 3)``.
    cutoff : float
        Tighter cutoff radius to filter down to.
    neighbor_list : Tensor
        COO edge index, shape ``(M, 2)``, int32.  Column 0 holds source (i)
        indices, column 1 holds target (j) indices.
    neighbor_ptr : Tensor
        CSR row pointer, shape ``(N+1,)``, int32.
    cell : Tensor | None
        Unit cell matrices.  Either ``(3, 3)`` or ``(B, 3, 3)``.
    unit_shifts : Tensor | None
        Integer unit-cell shift vectors, shape ``(M, 3)``, int32.
    batch_idx : Tensor | None
        System index per atom, shape ``(N,)``.  Required for per-system cells.

    Returns
    -------
    neighbor_list : Tensor
        Filtered COO edge index, shape ``(M', 2)``.
    neighbor_ptr : Tensor
        Rebuilt CSR row pointer, shape ``(N+1,)``.
    unit_shifts : Tensor | None
        Filtered shift vectors ``(M', 3)`` if provided, else ``None``.
    """
    N = positions.shape[0]
    dtype = positions.dtype
    device = positions.device

    i_idx = neighbor_list[:, 0]  # (M,)
    j_idx = neighbor_list[:, 1]  # (M,)

    pos_i = positions[i_idx]  # (M, 3)
    pos_j = positions[j_idx]  # (M, 3)
    delta = pos_j - pos_i  # (M, 3)

    if cell is not None and unit_shifts is not None:
        shifts_f = unit_shifts.to(dtype)  # (M, 3)
        if cell.dim() == 2:
            # Single cell (3, 3).
            atom_cell = cell.unsqueeze(0).expand(shifts_f.shape[0], 3, 3)  # (M,3,3)
        else:
            # Per-system cells (B, 3, 3); look up by source atom's system.
            if batch_idx is None:
                raise ValueError("batch_idx is required when cell has shape (B, 3, 3).")
            atom_cell = cell[batch_idx[i_idx]]  # (M, 3, 3)
        cart_shift = torch.einsum("ms,msd->md", shifts_f, atom_cell)
        delta = delta + cart_shift

    dist2 = (delta * delta).sum(dim=-1)  # (M,)
    keep = dist2 < (cutoff * cutoff)  # (M,)

    keep_idx = keep.nonzero(as_tuple=False).squeeze(1)  # (M',) — single sync

    new_nl = neighbor_list[keep_idx]  # (M', 2)
    out_shifts: Tensor | None = None
    if unit_shifts is not None:
        out_shifts = unit_shifts[keep_idx]

    # Rebuild CSR pointer via bincount (avoids allocating a ones tensor).
    counts = torch.bincount(i_idx[keep_idx], minlength=N).to(neighbor_ptr.dtype)
    new_ptr = torch.zeros(N + 1, dtype=neighbor_ptr.dtype, device=device)
    new_ptr[1:] = counts.cumsum(0)

    return new_nl, new_ptr, out_shifts


def neighbor_matrix_to_list(
    neighbor_matrix: Tensor,
    num_neighbors: Tensor,
    fill_value: int,
    neighbor_shifts: Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor | None]:
    """Convert a dense neighbor matrix (MATRIX format) to sparse COO+CSR.

    Parameters
    ----------
    neighbor_matrix : Tensor
        Dense neighbor indices, shape ``(N, K)``, int32.
    num_neighbors : Tensor
        Number of valid neighbors per atom, shape ``(N,)``, int32.
    fill_value : int
        Sentinel value marking empty slots.
    neighbor_shifts : Tensor | None
        Integer unit-cell shift vectors, shape ``(N, K, 3)``, int32.

    Returns
    -------
    neighbor_list : Tensor
        COO edge index ``(M, 2)``, int32.
    neighbor_ptr : Tensor
        CSR row pointer ``(N+1,)``, int32.
    unit_shifts : Tensor | None
        Shift vectors ``(M, 3)`` for valid pairs if provided, else ``None``.
    """
    N, K = neighbor_matrix.shape
    device = neighbor_matrix.device

    # Build source indices matching each column slot.
    i_idx = (
        torch.arange(N, device=device, dtype=neighbor_matrix.dtype)
        .unsqueeze(1)
        .expand(N, K)
    )  # (N, K)

    # Single nonzero call instead of multiple boolean-index operations (avoids
    # repeated CPU-GPU syncs to determine dynamic output shapes).
    valid_flat = (neighbor_matrix < fill_value).reshape(-1)  # (N*K,)
    flat_idx = valid_flat.nonzero(as_tuple=False).squeeze(1)  # (M,) — single sync

    i_sel = i_idx.reshape(-1)[flat_idx]  # (M,)
    j_sel = neighbor_matrix.reshape(-1)[flat_idx]  # (M,)
    nl = torch.stack([i_sel, j_sel], dim=1)  # (M, 2)

    # Build CSR pointer from num_neighbors via cumsum.
    ptr = torch.zeros(N + 1, dtype=torch.int32, device=device)
    ptr[1:] = num_neighbors.cumsum(0).to(torch.int32)

    out_shifts: Tensor | None = None
    if neighbor_shifts is not None:
        ns_flat = neighbor_shifts.reshape(-1, 3)  # (N*K, 3)
        out_shifts = ns_flat[flat_idx]  # (M, 3)

    return nl, ptr, out_shifts


def prepare_neighbors_for_model(
    data: Batch,
    model_cutoff: float,
    target_format: NeighborListFormat,
    fill_value: int,
) -> dict[str, Tensor]:
    """Unified entry point that filters/converts neighbor data for a model.

    Called from a model's ``adapt_input`` to obtain neighbor tensors in the
    format expected by the model, optionally filtering them to a tighter
    cutoff than the one used when the list was built.

    Parameters
    ----------
    data : Batch
        Batch object containing neighbor data (``neighbor_matrix`` /
        ``edge_index`` / etc.).
    model_cutoff : float
        The model's interaction cutoff.  If the neighbor list was built with
        a larger cutoff (detected via ``data._neighbor_list_cutoff``), the
        list is filtered down to ``model_cutoff``.
    target_format : NeighborListFormat
        The format the model expects (``MATRIX`` or ``COO``).
    fill_value : int
        Sentinel used to mark empty slots in dense neighbor matrices.

    Returns
    -------
    dict[str, Tensor]
        For ``MATRIX`` format: keys ``"neighbor_matrix"``, ``"num_neighbors"``,
        ``"neighbor_shifts"`` (only present when shifts exist).
        For ``COO`` format: keys ``"edge_index"`` (shape ``(E, 2)``),
        ``"edge_ptr"``, ``"unit_shifts"`` (only present when shifts exist).

    Raises
    ------
    RuntimeError
        If ``target_format`` is ``MATRIX`` but the batch has no
        ``neighbor_matrix``.
    """
    # Determine whether filtering is needed.
    built_cutoff: float | None = getattr(data, "_neighbor_list_cutoff", None)
    needs_filter = built_cutoff is not None and (built_cutoff - model_cutoff) > 1e-6

    # Collect optional common tensors.
    positions: Tensor = data.positions
    cell: Tensor | None = getattr(data, "cell", None)
    batch_idx: Tensor | None = getattr(data, "batch", None)

    has_matrix = getattr(data, "neighbor_matrix", None) is not None
    has_list = getattr(data, "edge_index", None) is not None

    # ------------------------------------------------------------------ #
    # Case 1: MATRIX target format, matrix available                      #
    # ------------------------------------------------------------------ #
    if target_format == NeighborListFormat.MATRIX:
        if not has_matrix:
            raise RuntimeError(
                "prepare_neighbors_for_model: target format is MATRIX but "
                "the batch has no 'neighbor_matrix'.  Ensure a "
                "NeighborListHook with format=MATRIX is registered."
            )
        nm: Tensor = data.neighbor_matrix
        nn_: Tensor = data.num_neighbors
        ns: Tensor | None = getattr(data, "neighbor_shifts", None)

        if needs_filter:
            nm, nn_, ns = filter_neighbor_matrix(
                positions=positions,
                cutoff=model_cutoff,
                neighbor_matrix=nm,
                num_neighbors=nn_,
                fill_value=fill_value,
                cell=cell,
                neighbor_shifts=ns,
                batch_idx=batch_idx,
            )

        out: dict[str, Tensor] = {
            "neighbor_matrix": nm,
            "num_neighbors": nn_,
        }
        if ns is not None:
            out["neighbor_shifts"] = ns
        return out

    # ------------------------------------------------------------------ #
    # Case 2: COO target format                                            #
    # ------------------------------------------------------------------ #
    # Sub-case A: have matrix — filter if needed then convert.
    if has_matrix:
        nm = data.neighbor_matrix
        nn_ = data.num_neighbors
        ns = getattr(data, "neighbor_shifts", None)

        if needs_filter:
            nm, nn_, ns = filter_neighbor_matrix(
                positions=positions,
                cutoff=model_cutoff,
                neighbor_matrix=nm,
                num_neighbors=nn_,
                fill_value=fill_value,
                cell=cell,
                neighbor_shifts=ns,
                batch_idx=batch_idx,
            )

        nl, ptr, shifts = neighbor_matrix_to_list(
            neighbor_matrix=nm,
            num_neighbors=nn_,
            fill_value=fill_value,
            neighbor_shifts=ns,
        )
        out = {"edge_index": nl, "edge_ptr": ptr}
        if shifts is not None:
            out["unit_shifts"] = shifts
        return out

    # Sub-case B: have list — filter if needed.
    if has_list:
        nl = data.edge_index  # (E, 2)
        ptr = data.edge_ptr
        us: Tensor | None = getattr(data, "unit_shifts", None)

        if needs_filter:
            nl, ptr, us = filter_neighbor_list(
                positions=positions,
                cutoff=model_cutoff,
                neighbor_list=nl,
                neighbor_ptr=ptr,
                cell=cell,
                unit_shifts=us,
                batch_idx=batch_idx,
            )

        out = {"edge_index": nl, "edge_ptr": ptr}
        if us is not None:
            out["unit_shifts"] = us
        return out

    raise RuntimeError(
        "prepare_neighbors_for_model: batch has neither 'neighbor_matrix' nor "
        "'edge_index'.  Ensure a NeighborListHook is registered before calling "
        "the model."
    )
