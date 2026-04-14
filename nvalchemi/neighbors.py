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
"""Neighbor list construction for atomic batches.

Provides :func:`compute_neighbors`, a one-shot function for populating
neighbor data on a :class:`~nvalchemi.data.Batch` outside a dynamics loop.
This is the recommended way to prepare a batch for model evaluation when
you are not running dynamics::

    from nvalchemi.neighbors import compute_neighbors

    batch = Batch.from_data_list([data], device="cuda")
    compute_neighbors(batch, config=model.model_config.neighbor_config)
    out = model(batch)

The :class:`~nvalchemi.dynamics.hooks.NeighborListHook` uses the same
shared helper (:func:`_write_neighbor_data_to_batch`) internally, so
the neighbor data written to the batch is identical regardless of which
path you use.
"""

from __future__ import annotations

import torch
from nvalchemiops.neighbors.neighbor_utils import estimate_max_neighbors
from nvalchemiops.torch.neighbors import neighbor_list
from nvalchemiops.torch.neighbors.neighbor_utils import (
    get_neighbor_list_from_neighbor_matrix,
)

from nvalchemi.data import Batch
from nvalchemi.data.level_storage import SegmentedLevelStorage
from nvalchemi.models.base import NeighborConfig, NeighborListFormat

__all__ = ["compute_neighbors"]


# ---------------------------------------------------------------------------
# Shared helper: write neighbor data into a Batch
# ---------------------------------------------------------------------------


@torch.compiler.disable
def _write_neighbor_data_to_batch(
    batch: Batch,
    neighbor_matrix: torch.Tensor,
    num_neighbors: torch.Tensor,
    neighbor_matrix_shifts: torch.Tensor | None,
    format: NeighborListFormat,
    cutoff: float,
) -> None:
    """Write computed neighbor data into *batch* and stamp the cutoff.

    For ``MATRIX`` format: writes ``neighbor_matrix``, ``num_neighbors``,
    and (optionally) ``neighbor_matrix_shifts`` into ``batch._atoms_group``.

    For ``COO`` format: converts the matrix to sparse edge form via
    :func:`get_neighbor_list_from_neighbor_matrix`, creates a
    :class:`~nvalchemi.data.level_storage.SegmentedLevelStorage`, and
    replaces ``batch._storage.groups["edges"]``.

    Parameters
    ----------
    batch : Batch
        Target batch (mutated in-place).
    neighbor_matrix : torch.Tensor
        Dense neighbor indices, shape ``(N, max_neighbors)`` int32.
    num_neighbors : torch.Tensor
        Per-atom neighbor count, shape ``(N,)`` int32.
    neighbor_matrix_shifts : torch.Tensor | None
        PBC lattice shifts, shape ``(N, max_neighbors, 3)`` int32, or
        ``None`` for non-periodic systems.
    format : NeighborListFormat
        Whether to store as MATRIX or convert to COO.
    cutoff : float
        The cutoff used to build the list (stamped on the batch for
        downstream filtering by ``prepare_neighbors_for_model``).
    """
    if format == NeighborListFormat.COO:
        neighbor_list_coo = get_neighbor_list_from_neighbor_matrix(
            neighbor_matrix=neighbor_matrix,
            num_neighbors=num_neighbors,
            neighbor_shift_matrix=neighbor_matrix_shifts
            if neighbor_matrix_shifts is not None
            else None,
            fill_value=batch.num_nodes,
        )
        neighbor_list_edges = neighbor_list_coo[0].T.contiguous()  # (E, 2) int32
        nl_shifts = (
            neighbor_list_coo[2].to(torch.int32) if len(neighbor_list_coo) > 2 else None
        )

        src_atoms = neighbor_list_edges[:, 0]  # (E,)
        graph_per_edge = batch.batch_idx[src_atoms]  # (E,)
        B = batch.num_graphs
        seg_lengths = torch.bincount(graph_per_edge, minlength=B).to(torch.int32)

        data_dict: dict[str, torch.Tensor] = {"neighbor_list": neighbor_list_edges}
        if nl_shifts is not None:
            data_dict["neighbor_list_shifts"] = nl_shifts

        batch._storage.groups["edges"] = SegmentedLevelStorage(
            data=data_dict,
            device=batch.device,
            segment_lengths=seg_lengths,
            validate=False,
        )
    else:
        atoms_group = batch._atoms_group
        if atoms_group is None:
            raise RuntimeError("Cannot store neighbor data: batch has no atoms group.")
        atoms_group["neighbor_matrix"] = neighbor_matrix
        atoms_group["num_neighbors"] = num_neighbors
        if neighbor_matrix_shifts is not None:
            atoms_group["neighbor_matrix_shifts"] = neighbor_matrix_shifts

    batch._neighbor_list_cutoff = cutoff


# ---------------------------------------------------------------------------
# User-facing one-shot neighbor list computation
# ---------------------------------------------------------------------------


def compute_neighbors(
    batch: Batch,
    cutoff: float | None = None,
    *,
    config: NeighborConfig | None = None,
    format: NeighborListFormat = NeighborListFormat.MATRIX,
    max_neighbors: int | None = None,
    half_list: bool = False,
) -> None:
    """Compute a neighbor list and write results into *batch* in-place.

    One-shot convenience for populating neighbor data outside a dynamics
    loop.  Equivalent to constructing a
    :class:`~nvalchemi.dynamics.hooks.NeighborListHook` and invoking it,
    without the staging-buffer or skin-buffer machinery that is only
    useful for repeated dynamics steps.

    After the call, ``batch.neighbor_matrix`` / ``batch.num_neighbors``
    (MATRIX format) or ``batch.neighbor_list`` / ``batch.unit_shifts``
    (COO format) are populated, and ``batch._neighbor_list_cutoff`` is
    stamped for downstream use by
    :func:`~nvalchemi.models._ops.neighbor_filter.prepare_neighbors_for_model`.

    Pass either *config* (from a model's
    :attr:`~nvalchemi.models.base.ModelConfig.neighbor_config`) or the
    scalar parameters directly.  When *config* is provided, scalar
    parameters are ignored.

    Parameters
    ----------
    batch : Batch
        Batch to populate with neighbor data.
    cutoff : float | None
        Interaction cutoff radius.  Required if *config* is ``None``.
    config : NeighborConfig | None
        Full neighbor configuration (from
        ``model.model_config.neighbor_config``).
    format : NeighborListFormat
        Output format (``MATRIX`` or ``COO``).  Default: ``MATRIX``.
    max_neighbors : int | None
        Maximum neighbors per atom.  Auto-estimated from *cutoff* if
        ``None``.
    half_list : bool
        Whether to build a half neighbor list.  Default: ``False``.

    Raises
    ------
    ValueError
        If neither *cutoff* nor *config* is provided.

    Examples
    --------
    ::

        from nvalchemi.neighbors import compute_neighbors
        from nvalchemi.models.lj import LennardJonesModelWrapper

        lj = LennardJonesModelWrapper(epsilon=0.01, sigma=3.4, cutoff=8.5)
        batch = Batch.from_data_list([data], device="cuda")
        compute_neighbors(batch, config=lj.model_config.neighbor_config)
        out = lj(batch)
    """
    if config is not None:
        cutoff = config.cutoff
        format = config.format
        half_list = config.half_list
    elif cutoff is None:
        raise ValueError("Either 'cutoff' or 'config' must be provided.")

    N = batch.num_nodes
    device = batch.device

    pbc = getattr(batch, "pbc", None)
    cell = getattr(batch, "cell", None)

    if max_neighbors is None:
        max_neighbors = estimate_max_neighbors(cutoff=cutoff)
        # Non-PBC hard cap: an atom can see at most (N_system - 1)
        # neighbors without periodic images.  We use max_num_nodes
        # (not max_num_nodes - 1) so that K has one sentinel slot
        # to distinguish "all used" from "overflow" in the retry check.
        # Round up to nearest 16 for memory-aligned kernel performance.
        if pbc is None and batch.max_num_nodes > 0:
            cap = ((batch.max_num_nodes + 15) // 16) * 16
            max_neighbors = min(max_neighbors, cap)

    # Cast index tensors to int32 (Warp kernels require it).
    batch_ptr = batch.batch_ptr.to(torch.int32)
    batch_idx = batch.batch_idx.to(torch.int32)

    # Build the neighbor list.  If the kernel overflows (some atoms have
    # more neighbors than max_neighbors), grow the buffer and retry.
    while True:
        nb_matrix = torch.full((N, max_neighbors), N, dtype=torch.int32, device=device)
        nb_counts = torch.zeros(N, dtype=torch.int32, device=device)
        nb_shifts: torch.Tensor | None = None
        if pbc is not None:
            nb_shifts = torch.zeros(
                N, max_neighbors, 3, dtype=torch.int32, device=device
            )

        neighbor_list(
            positions=batch.positions,
            cutoff=cutoff,
            cell=cell,
            pbc=pbc,
            max_neighbors=max_neighbors,
            half_fill=half_list,
            batch_ptr=batch_ptr,
            batch_idx=batch_idx,
            neighbor_matrix=nb_matrix,
            num_neighbors=nb_counts,
            neighbor_matrix_shifts=nb_shifts,
            rebuild_flags=None,
        )

        # Check for overflow (sync is acceptable — one-shot call).
        actual_max = int(nb_counts.max())
        if actual_max < max_neighbors:
            break
        # Overflow: grow with headroom and retry.
        max_neighbors = int(actual_max * 1.5) + 1

    _write_neighbor_data_to_batch(
        batch, nb_matrix, nb_counts, nb_shifts, format, cutoff
    )
