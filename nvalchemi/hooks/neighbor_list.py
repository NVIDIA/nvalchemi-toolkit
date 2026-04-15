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
"""Neighbor list hook for on-the-fly neighbor list construction.

This module provides :class:`NeighborListHook`, which runs at the
``BEFORE_COMPUTE`` stage to compute or refresh the neighbor list stored in
the batch before the model forward pass.  It supports an optional Verlet
skin buffer to avoid recomputing neighbors every step.

Both ``MATRIX`` and ``COO`` neighbor formats are supported for dynamic
updates (i.e. updates each dynamics step).  For ``COO`` format the hook
creates or replaces the edges group on the batch each step so that
``batch.neighbor_list`` (shape ``(E, 2)``) and ``batch.neighbor_list_shifts``
(shape ``(E, 3)``, PBC only) are always up to date.  The companion
``Batch.edge_ptr`` property derives the per-atom CSR pointer on demand.

Pre-allocation
--------------
The hook maintains *staging buffers* — persistent GPU tensors that are
refreshed each step via ``Tensor.copy_()`` — to avoid per-step dynamic
allocation inside the ``neighbor_list`` dispatcher.

``neighbor_list`` selects between ``batch_naive`` (avg < 2000 atoms/system)
and ``batch_cell_list`` (avg >= 2000), see https://nvidia.github.io/nvalchemi-toolkit-ops/userguide/components/neighborlist.html.
Both paths normally allocate auxiliary tensors on-demand with CPU-GPU syncs
(e.g. ``.item()`` calls). :meth:`NeighborListHook._alloc_nl_kwargs`
computes these **once** when the batch shape is first seen (or changes)
and caches them in ``NeighborListHook._buf_nl_kwargs``:

* *Naive, no PBC*: no extra kwargs needed.
* *Naive, PBC*: ``shift_range_per_dimension``, ``num_shifts_per_system``,
  ``max_shifts_per_system``, and ``max_atoms_per_system``.
* *Cell list*: seven cell-list scratch tensors via ``allocate_cell_list``.

**NPT note**: geometry-dependent kwargs (shift ranges, cell-list sizes) are
fixed when the staging buffers are first allocated for a given ``(N, B)``
shape.  For NPT (variable-cell) simulations the pre-computed values may
become stale as the cell changes; accuracy is maintained by keeping the
cutoff + skin well below the shortest cell dimension throughout the run.
"""

from __future__ import annotations

from enum import Enum

import torch
from nvalchemiops.neighbors.neighbor_utils import estimate_max_neighbors
from nvalchemiops.torch.neighbors import neighbor_list

try:
    from nvalchemiops.torch.neighbors.batch_cell_list import (
        estimate_batch_cell_list_sizes,
    )
    from nvalchemiops.torch.neighbors.neighbor_utils import (
        allocate_cell_list,
        compute_naive_num_shifts,
    )
except ImportError:
    allocate_cell_list = None
    compute_naive_num_shifts = None
    estimate_batch_cell_list_sizes = None

try:
    from nvalchemiops.torch.neighbors.rebuild_detection import (
        batch_neighbor_list_needs_rebuild as _batch_nl_needs_rebuild,
    )
except ImportError:
    _batch_nl_needs_rebuild = None

try:
    from nvalchemi.dynamics._ops.neighbor_list_rebuild import (
        batch_neighbor_list_rebuild_inplace as _batch_nl_rebuild_inplace,
    )
except ImportError:
    _batch_nl_rebuild_inplace = None

from nvalchemi.data import Batch
from nvalchemi.hooks._context import HookContext
from nvalchemi.models.base import NeighborConfig, NeighborListFormat
from nvalchemi.neighbors import _write_neighbor_data_to_batch


class NeighborListHook:
    """Compute and cache neighbor lists before each model evaluation.

    This hook runs at :attr:`~DynamicsStage.BEFORE_COMPUTE` and writes
    neighbor data into the batch so that the model's ``adapt_input`` can
    read it.  An optional Verlet skin buffer avoids rebuilding the list
    every step: the list is only recomputed when the maximum atomic
    displacement since the last build exceeds ``config.skin / 2``, or when
    the set of active systems changes (detected via ``system_id``).

    For ``MATRIX`` format the following tensors are written to the atoms
    group of the batch (and thus accessible as ``batch.neighbor_matrix``
    etc.):

    * ``neighbor_matrix`` — shape ``(N, max_neighbors)``, int32
    * ``num_neighbors``   — shape ``(N,)``, int32
    * ``neighbor_matrix_shifts`` — shape ``(N, max_neighbors, 3)``, int32
      (only written when PBC is active)

    For ``COO`` format the edges group of the batch is created or replaced
    on every rebuild, making the following accessible:

    * ``batch.neighbor_list`` — shape ``(E, 2)``, int32 (nvalchemi convention)
    * ``batch.neighbor_list_shifts`` — shape ``(E, 3)``, int32 (only when PBC active)
    * ``batch.edge_ptr`` — shape ``(N+1,)``, int32, derived on demand via
      the :attr:`~nvalchemi.data.Batch.edge_ptr` property

    Parameters
    ----------
    config : NeighborConfig
        Neighbor list configuration read from the model config.
    skin : float, optional
        Verlet skin distance in the same length units as positions.
        The neighbor list is searched out to ``cutoff + skin`` so that
        atoms crossing the skin boundary but not the bare cutoff are
        already included.  The list is only rebuilt when any atom has
        moved more than ``skin / 2`` since the previous build (requires
        ``nvalchemiops >= 0.4``); set to ``0.0`` (default) to rebuild
        every step.
    max_neighbors : int | None, optional
        Maximum number of neighbors per atom for MATRIX format.  When
        ``None`` (default), auto-estimated from the cutoff via
        ``estimate_max_neighbors(cutoff)``.  Ignored for COO format.
    stage : Enum | None, optional
        The workflow stage at which this hook runs.  Defaults to
        ``DynamicsStage.BEFORE_COMPUTE``.
    """

    def __init__(
        self,
        config: NeighborConfig,
        skin: float = 0.0,
        max_neighbors: int | None = None,
        stage: Enum | None = None,
    ) -> None:
        self.config = config
        self.skin = skin
        self.stage = stage
        self._max_neighbors_override = max_neighbors
        self.frequency = 1
        self._neighbor_list_flag = config.format == NeighborListFormat.COO

        # Skin-buffer state: populated after the first build.
        self._ref_positions: torch.Tensor | None = None
        self._rebuild_flags: torch.Tensor | None = None

        # Neighbor Matrix state: populated after the first build.
        self._neighbor_matrix: torch.Tensor | None = None
        self._col_range: torch.Tensor | None = None
        self._num_neighbors: torch.Tensor | None = None
        self._neighbor_matrix_shifts: torch.Tensor | None = None

        # Shape the staging buffers were allocated for; used to detect when
        # re-allocation is needed (e.g. inflight batching with variable load).
        self._alloc_N: int | None = None
        self._alloc_B: int | None = None

        # Staging buffers — persistent GPU tensors refreshed each step via
        # copy_() to avoid per-step dynamic allocation inside the dispatcher.
        self._buf_positions: torch.Tensor | None = None
        self._buf_batch_idx: torch.Tensor | None = None
        self._buf_batch_ptr: torch.Tensor | None = None
        self._buf_cell: torch.Tensor | None = None  # PBC only
        self._buf_pbc: torch.Tensor | None = None  # PBC only

        # Algorithm-specific pre-allocated kwargs forwarded to neighbor_list.
        self._buf_nl_kwargs: dict[str, torch.Tensor] = {}

        # Adaptive K-dimension state.
        self._actual_max_k: torch.Tensor | None = None  # GPU scalar from last build
        self._first_build: bool = True  # Force sync check after first kernel call

    # ------------------------------------------------------------------
    # Main hook entry point
    # ------------------------------------------------------------------
    @torch.compile(fullgraph=False, mode="max-autotune-no-cudagraphs")
    def __call__(self, ctx: HookContext, stage: Enum) -> None:
        """Recompute the neighbor list if needed and write it to the batch.

        When ``skin > 0`` and ``nvalchemiops`` provides
        :func:`~nvalchemiops.torch.neighbors.rebuild_detection.batch_neighbor_list_needs_rebuild`,
        the list is only rebuilt when at least one atom has moved more than
        ``skin / 2`` since the previous build.  The reference positions are
        updated in-place on the GPU (no clone) whenever a rebuild occurs.
        """
        self._rebuild(ctx.batch)

        # First build: initialise the skin-buffer reference (one-time clone).
        if self.skin > 0.0 and self._ref_positions is None:
            self._init_ref_positions(ctx.batch.positions)

    @torch.compiler.disable
    def _init_ref_positions(self, positions: torch.Tensor) -> None:
        """One-time clone of positions into the skin-buffer reference.

        Marked ``@torch.compiler.disable`` because the attribute assignment
        is a Python mutation that creates a graph break.  Called only on the
        first step for a given batch shape.
        """
        self._ref_positions = positions.detach().clone()

    # ------------------------------------------------------------------
    # Neighbor list construction
    # ------------------------------------------------------------------

    def _rebuild(self, batch: Batch) -> None:
        """Build the neighbor list and write results into the batch."""
        positions = batch.positions  # (N, 3)
        batch_ptr = batch.batch_ptr  # (B+1,)
        N = batch.num_nodes
        B = batch.num_graphs

        # Detect PBC.  getattr avoids a try/except which is a graph break.
        pbc = getattr(batch, "pbc", None)  # (B, 3) bool or None
        cell = getattr(batch, "cell", None)  # (B, 3, 3) float or None

        # ------------------------------------------------------------------
        # Allocate (or reallocate) the output tensors when shape changes.
        # Reallocation also resets the skin-buffer state so that the first
        # subsequent step forces a full rebuild and re-initialises
        # _ref_positions for the new atom count.
        # ------------------------------------------------------------------
        if self._neighbor_matrix is None or self._neighbor_matrix.shape[0] != N:
            self._alloc_output_tensors(N, batch, pbc)

        # ------------------------------------------------------------------
        # (Re)allocate staging buffers and algorithm kwargs on shape change.
        # ------------------------------------------------------------------
        if self._alloc_N != N or self._alloc_B != B:
            # Composition changed — check K before staging realloc.
            self._check_and_resize_k(N, batch.device, pbc)
            self._alloc_staging_buffers(
                N, B, positions.dtype, batch.device, cell, pbc, batch_ptr
            )
            self._alloc_N = N
            self._alloc_B = B

        # Refresh staging buffers from the current batch.
        self._copy_to_staging_buffers(positions, batch_ptr, batch.batch_idx, cell, pbc)

        # ------------------------------------------------------------------
        # Skin check: decide per-system whether the neighbor list needs
        # rebuilding based on atomic displacement since the last build.
        # Uses the in-place variant to avoid per-step allocation of the
        # rebuild_flags tensor.  Falls back to the upstream function if the
        # in-place op is not available (nvalchemiops < 0.4 or custom op not
        # loaded).
        # ------------------------------------------------------------------
        if self.skin > 0.0 and self._ref_positions is not None:
            cell_inv = (
                torch.linalg.inv_ex(self._buf_cell)[0].contiguous()
                if self._buf_cell is not None
                else None
            )
            if _batch_nl_rebuild_inplace is not None:
                _batch_nl_rebuild_inplace(
                    reference_positions=self._ref_positions,
                    current_positions=self._buf_positions,
                    batch_idx=self._buf_batch_idx,
                    rebuild_flags=self._rebuild_flags,
                    skin_distance_threshold=self.skin / 2,
                    update_reference_positions=True,
                    cell=self._buf_cell,
                    cell_inv=cell_inv,
                    pbc=self._buf_pbc,
                )
            elif _batch_nl_needs_rebuild is not None:
                self._rebuild_flags = _batch_nl_needs_rebuild(
                    reference_positions=self._ref_positions,
                    current_positions=self._buf_positions,
                    batch_idx=self._buf_batch_idx,
                    skin_distance_threshold=self.skin / 2,
                    update_reference_positions=True,
                    cell=self._buf_cell,
                    cell_inv=cell_inv,
                    pbc=self._buf_pbc,
                )

        # ------------------------------------------------------------------
        # Build the neighbor list using pre-allocated buffers.
        # ------------------------------------------------------------------
        neighbor_list(
            positions=self._buf_positions,
            cutoff=self.config.cutoff + self.skin,
            cell=self._buf_cell,
            pbc=self._buf_pbc,
            max_neighbors=self._max_neighbors,
            half_fill=self.config.half_list,
            batch_ptr=self._buf_batch_ptr,
            batch_idx=self._buf_batch_idx,
            neighbor_matrix=self._neighbor_matrix,
            num_neighbors=self._num_neighbors,
            neighbor_matrix_shifts=self._neighbor_matrix_shifts,
            rebuild_flags=self._rebuild_flags,
            **self._buf_nl_kwargs,
        )

        # ------------------------------------------------------------------
        # Adaptive K: first-build check (runs once, then never again).
        # This is the only per-step adaptive K code.  After the first
        # build, all checks are gated on structural events (N/B change)
        # inside _alloc_output_tensors / _alloc_staging_buffers.
        # ------------------------------------------------------------------
        if self._first_build:
            self._first_build = False
            self._actual_max_k = self._num_neighbors.max()
            grew = self._check_and_resize_k(N, batch.device, pbc)
            if grew:
                # K was too small — re-run kernel with larger buffers.
                neighbor_list(
                    positions=self._buf_positions,
                    cutoff=self.config.cutoff + self.skin,
                    cell=self._buf_cell,
                    pbc=self._buf_pbc,
                    max_neighbors=self._max_neighbors,
                    half_fill=self.config.half_list,
                    batch_ptr=self._buf_batch_ptr,
                    batch_idx=self._buf_batch_idx,
                    neighbor_matrix=self._neighbor_matrix,
                    num_neighbors=self._num_neighbors,
                    neighbor_matrix_shifts=self._neighbor_matrix_shifts,
                    rebuild_flags=None,  # Force full rebuild
                    **self._buf_nl_kwargs,
                )

        # ------------------------------------------------------------------
        # Mark Stale Entries
        # ------------------------------------------------------------------
        stale = self._col_range.unsqueeze(0) >= self._num_neighbors.unsqueeze(1)
        self._neighbor_matrix[stale] = batch.num_nodes
        if self._neighbor_matrix_shifts is not None:
            self._neighbor_matrix_shifts[stale] = 0

        # ------------------------------------------------------------------
        # Post-processing: write results to batch (shared with compute_neighbors)
        # ------------------------------------------------------------------
        _write_neighbor_data_to_batch(
            batch=batch,
            neighbor_matrix=self._neighbor_matrix,
            num_neighbors=self._num_neighbors,
            neighbor_matrix_shifts=self._neighbor_matrix_shifts,
            format=NeighborListFormat.COO
            if self._neighbor_list_flag
            else NeighborListFormat.MATRIX,
            cutoff=self.config.cutoff,
        )

    # ------------------------------------------------------------------
    # Staging buffer management
    # ------------------------------------------------------------------

    @torch.compiler.disable
    def _alloc_output_tensors(
        self,
        N: int,
        batch: "Batch",
        pbc: torch.Tensor | None,
    ) -> None:
        """Allocate neighbor-matrix output tensors for atom count *N*.

        Marked ``@torch.compiler.disable`` because it calls
        ``estimate_max_neighbors`` (CPU work), allocates tensors with
        dynamic shapes, and mutates Python attributes — all graph breaks.
        Called only when the atom count changes.
        """
        device = batch.device
        max_nbrs = self._max_neighbors_override
        if max_nbrs is None:
            max_nbrs = estimate_max_neighbors(
                cutoff=self.config.cutoff + self.skin,
            )
        # Non-PBC hard cap: an atom can see at most (N_system - 1)
        # neighbors without periodic images.  We use max_num_nodes
        # (not max_num_nodes - 1) so that K has one sentinel slot
        # to distinguish "all used" from "overflow" in the adaptive check.
        # Round up to nearest 16 for memory-aligned kernel performance.
        if pbc is None and batch.max_num_nodes > 0:
            cap = ((batch.max_num_nodes + 15) // 16) * 16
            max_nbrs = min(max_nbrs, cap)
        self._max_neighbors = max_nbrs
        self._neighbor_matrix = torch.full(
            (N, max_nbrs), N, dtype=torch.int32, device=device
        )
        self._col_range = torch.arange(max_nbrs, device=device, dtype=torch.int32)
        self._num_neighbors = torch.zeros(N, dtype=torch.int32, device=device)
        if pbc is not None:
            self._neighbor_matrix_shifts = torch.zeros(
                N, max_nbrs, 3, dtype=torch.int32, device=device
            )
        # Reset skin-buffer state so __call__ re-initialises _ref_positions.
        self._ref_positions = None
        self._rebuild_flags = None
        # Reset adaptive K state so first build triggers a sync check.
        self._first_build = True
        self._actual_max_k = None

    @torch.compiler.disable
    def _check_and_resize_k(
        self,
        N: int,
        device: torch.device,
        pbc: torch.Tensor | None,
    ) -> bool:
        """Sync on actual max K and grow/shrink the neighbor matrix if needed.

        Called on structural events (first build, N/B change, cell volume
        change).  The sync cost is acceptable because these events are
        infrequent and the calling code path is already off the compile graph.

        Returns ``True`` if K was grown (caller must re-run the kernel).
        Shrinking trims the existing buffers in-place — no re-run needed.
        """
        if self._actual_max_k is None:
            return False
        actual = int(self._actual_max_k.item())

        if actual >= self._max_neighbors:
            # Overflow — grow with 1.5x headroom and round to nearest 16.  Must re-run kernel.
            self._max_neighbors = ((int(actual * 1.5) + 15) // 16) * 16
            self._realloc_k(N, device, pbc)
            return True
        elif actual < (1 / 2) * self._max_neighbors and actual > 0:
            # 2x+ overestimate — trim existing buffers in-place.
            new_k = ((int(actual * 2) + 15) // 16) * 16
            # Never shrink below the user-provided override — it serves as a
            # hard floor.  We may grow above it on overflow, but not below.
            if self._max_neighbors_override is not None:
                new_k = max(new_k, self._max_neighbors_override)
            if new_k < self._max_neighbors:
                self._max_neighbors = new_k
                self._neighbor_matrix = self._neighbor_matrix[:, :new_k].contiguous()
                self._col_range = self._col_range[:new_k]
                if self._neighbor_matrix_shifts is not None:
                    self._neighbor_matrix_shifts = self._neighbor_matrix_shifts[
                        :, :new_k
                    ].contiguous()
            # num_neighbors unchanged — still valid.
        return False

    @torch.compiler.disable
    def _realloc_k(
        self,
        N: int,
        device: torch.device,
        pbc: torch.Tensor | None,
    ) -> None:
        """Reallocate neighbor-matrix buffers at the current N with a new K.

        Preserves N (no staging-buffer realloc needed) but resets the
        skin state to force a full rebuild on the next step.
        """
        max_nbrs = self._max_neighbors
        self._neighbor_matrix = torch.full(
            (N, max_nbrs), N, dtype=torch.int32, device=device
        )
        self._col_range = torch.arange(max_nbrs, device=device, dtype=torch.int32)
        self._num_neighbors = torch.zeros(N, dtype=torch.int32, device=device)
        if pbc is not None:
            self._neighbor_matrix_shifts = torch.zeros(
                N, max_nbrs, 3, dtype=torch.int32, device=device
            )
        else:
            self._neighbor_matrix_shifts = None
        # Reset skin state to force a full rebuild.
        self._ref_positions = None
        self._rebuild_flags = None

    @torch.compiler.disable
    def _alloc_staging_buffers(
        self,
        N: int,
        B: int,
        dtype: torch.dtype,
        device: torch.device,
        cell: torch.Tensor | None,
        pbc: torch.Tensor | None,
        batch_ptr: torch.Tensor | None = None,
    ) -> None:
        """Allocate persistent staging buffers for the current (N, B) shape."""
        self._buf_positions = torch.zeros(N, 3, dtype=dtype, device=device)
        self._buf_batch_idx = torch.zeros(N, dtype=torch.int32, device=device)
        self._buf_batch_ptr = torch.zeros(B + 1, dtype=torch.int32, device=device)
        if cell is not None:
            self._buf_cell = torch.zeros(B, 3, 3, dtype=dtype, device=device)
            self._buf_pbc = torch.zeros(B, 3, dtype=torch.bool, device=device)
        else:
            self._buf_cell = None
            self._buf_pbc = None
        # Pre-allocate rebuild_flags as all-True so that the very first step
        # (before _ref_positions is set and the skin check runs) forces a full
        # neighbor-list build for every system.  The in-place op zeroes this
        # buffer at the start of each subsequent call before writing fresh values.
        self._rebuild_flags = torch.ones(B, dtype=torch.bool, device=device)
        # Pre-allocate algorithm-specific kwargs to eliminate on-demand CPU syncs
        # from the neighbor_list dispatcher.  Use the actual batch_ptr (if provided)
        # to compute max_atoms_per_system correctly — the staging buffer is still
        # all-zeros at this point and would give max_atoms = 0.
        ptr = batch_ptr if batch_ptr is not None else self._buf_batch_ptr
        self._alloc_nl_kwargs(N, B, self._buf_positions, ptr, cell, pbc, device, dtype)

    def _copy_to_staging_buffers(
        self,
        positions: torch.Tensor,
        batch_ptr: torch.Tensor,
        batch_idx: torch.Tensor,
        cell: torch.Tensor | None,
        pbc: torch.Tensor | None,
    ) -> None:
        """Refresh staging buffers from the current batch."""
        self._buf_positions.copy_(positions)
        self._buf_batch_ptr.copy_(batch_ptr)
        self._buf_batch_idx.copy_(batch_idx)
        if self._buf_cell is not None and cell is not None:
            self._buf_cell.copy_(cell)
        if self._buf_pbc is not None and pbc is not None:
            self._buf_pbc.copy_(pbc)

    # ------------------------------------------------------------------
    # Algorithm-specific pre-allocation
    # ------------------------------------------------------------------

    def _alloc_nl_kwargs(
        self,
        N: int,
        B: int,
        positions: torch.Tensor,
        batch_ptr: torch.Tensor,
        cell: torch.Tensor | None,
        pbc: torch.Tensor | None,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        """Pre-allocate algorithm-specific kwargs to remove CPU-GPU syncs.

        The ``neighbor_list`` dispatcher normally infers geometry-dependent
        values (shift ranges, cell-list sizes) at call time using ``.item()``
        synchronisations.  This method computes them **once** when the staging
        buffers are allocated (or re-allocated after a shape change) and stores
        the resulting tensors in ``_buf_nl_kwargs`` so they can be forwarded as
        ``**kwargs`` on every ``neighbor_list`` call.

        Algorithm selection mirrors the dispatcher threshold:

        * ``avg_atoms < 2000`` -> ``batch_naive``
        * ``avg_atoms >= 2000`` -> ``batch_cell_list``

        Parameters
        ----------
        N, B : int
            Total atom count and number of systems at alloc time.
        positions : torch.Tensor
            Staging buffer for positions (used to estimate bounding box for
            non-PBC cell-list systems).
        batch_ptr : torch.Tensor
            Staging buffer for batch_ptr (used to get max_atoms_per_system).
        cell, pbc : torch.Tensor or None
            Cell and PBC flag tensors at alloc time.
        device, dtype : torch.device, torch.dtype
            Allocation target.
        """
        self._buf_nl_kwargs = {}

        avg_atoms = N // max(B, 1)
        use_cell_list = avg_atoms >= 2000

        if use_cell_list:
            if estimate_batch_cell_list_sizes is None or allocate_cell_list is None:
                return  # nvalchemiops too old; fall back to dynamic allocation
            if cell is not None and pbc is not None:
                # PBC: use the actual cell geometry.
                alloc_cell = cell.to(dtype).contiguous()
                alloc_pbc = pbc
            else:
                # Non-PBC: synthesise a bounding-box cell from current positions
                # with a 1.5x pad so that position drift during the simulation
                # doesn't overflow the pre-allocated cell-list arrays.
                expanded_idx = self._buf_batch_idx.unsqueeze(1).expand_as(positions)
                pos_min = torch.full((B, 3), float("inf"), dtype=dtype, device=device)
                pos_min.scatter_reduce_(0, expanded_idx, positions, reduce="amin")
                pos_max = torch.full((B, 3), float("-inf"), dtype=dtype, device=device)
                pos_max.scatter_reduce_(0, expanded_idx, positions, reduce="amax")
                cell_lengths = (pos_max - pos_min) * 1.5 + 0.1 * (
                    self.config.cutoff + self.skin
                )
                alloc_cell = torch.diag_embed(cell_lengths)  # (B, 3, 3)
                alloc_pbc = torch.zeros(B, 3, dtype=torch.bool, device=device)

            max_total_cells, neighbor_search_radius = estimate_batch_cell_list_sizes(
                alloc_cell, alloc_pbc, self.config.cutoff + self.skin
            )
            (
                cells_per_dimension,
                neighbor_search_radius,
                atom_periodic_shifts,
                atom_to_cell_mapping,
                atoms_per_cell_count,
                cell_atom_start_indices,
                cell_atom_list,
            ) = allocate_cell_list(
                N, int(max_total_cells), neighbor_search_radius, device
            )
            self._buf_nl_kwargs = {
                "cells_per_dimension": cells_per_dimension,
                "neighbor_search_radius": neighbor_search_radius,
                "atom_periodic_shifts": atom_periodic_shifts,
                "atom_to_cell_mapping": atom_to_cell_mapping,
                "atoms_per_cell_count": atoms_per_cell_count,
                "cell_atom_start_indices": cell_atom_start_indices,
                "cell_atom_list": cell_atom_list,
            }

        else:
            # Naive algorithm.
            if cell is not None and pbc is not None:
                # PBC naive: pre-compute shift-range tensors so the dispatcher
                # does not call compute_naive_num_shifts (which has .item()) on
                # the hot path.
                if compute_naive_num_shifts is None:
                    return
                shift_range, num_shifts, max_shifts = compute_naive_num_shifts(
                    cell.to(dtype).contiguous(),
                    self.config.cutoff + self.skin,
                    pbc,
                )
                max_atoms = int((batch_ptr[1:] - batch_ptr[:-1]).max().item())
                self._buf_nl_kwargs = {
                    "shift_range_per_dimension": shift_range,
                    "num_shifts_per_system": num_shifts,
                    "max_shifts_per_system": max_shifts,
                    "max_atoms_per_system": max_atoms,
                }
            # No-PBC naive: no extra kwargs required — the kernel has no
            # CPU-sync allocations in this branch.
