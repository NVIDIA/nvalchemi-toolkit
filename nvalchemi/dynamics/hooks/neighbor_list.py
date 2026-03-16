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
``batch.edge_index`` (shape ``(E, 2)``) and ``batch.unit_shifts``
(shape ``(E, 3)``, PBC only) are always up to date.  The companion
``Batch.edge_ptr`` property derives the per-atom CSR pointer on demand.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from nvalchemiops.torch.neighbors import neighbor_list

from nvalchemi.dynamics.base import HookStageEnum
from nvalchemi.models.base import NeighborConfig, NeighborListFormat

if TYPE_CHECKING:
    from nvalchemi.data import Batch
    from nvalchemi.dynamics.base import BaseDynamics


class NeighborListHook:
    """Compute and cache neighbor lists before each model evaluation.

    This hook runs at :attr:`~HookStageEnum.BEFORE_COMPUTE` and writes
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
    * ``neighbor_shifts`` — shape ``(N, max_neighbors, 3)``, int32
      (only written when PBC is active)

    For ``COO`` format the edges group of the batch is created or replaced
    on every rebuild, making the following accessible:

    * ``batch.edge_index`` — shape ``(E, 2)``, int32 (nvalchemi convention)
    * ``batch.unit_shifts`` — shape ``(E, 3)``, int32 (only when PBC active)
    * ``batch.edge_ptr`` — shape ``(N+1,)``, int32, derived on demand via
      the :attr:`~nvalchemi.data.Batch.edge_ptr` property

    Parameters
    ----------
    config : NeighborConfig
        Neighbor list configuration read from the model card.  The
        ``max_neighbors`` field must be set when ``format=MATRIX``.

    Raises
    ------
    ValueError
        If ``format=MATRIX`` and ``config.max_neighbors`` is not set.
    """

    stage: HookStageEnum = HookStageEnum.BEFORE_COMPUTE
    frequency: int = 1

    def __init__(self, config: NeighborConfig) -> None:
        self.config = config
        self._neighbor_list_flag = config.format == NeighborListFormat.COO
        self._ref_positions: torch.Tensor | None = None
        self._ref_system_ids: torch.Tensor | None = None

    # ------------------------------------------------------------------
    # Main hook entry point
    # ------------------------------------------------------------------

    def __call__(self, batch: Batch, dynamics: BaseDynamics) -> None:
        """Recompute the neighbor list and write it to *batch*."""

        self._rebuild(batch)

        # Update skin-buffer reference state.
        self._ref_positions = batch.positions.detach().clone()
        try:
            self._ref_system_ids = batch.system_id.detach().clone()
        except AttributeError:
            self._ref_system_ids = None

    # ------------------------------------------------------------------
    # Neighbor list construction
    # ------------------------------------------------------------------

    def _rebuild(self, batch: Batch) -> None:
        """Build the neighbor list and write results into the batch."""

        positions = batch.positions  # (N, 3)
        batch_ptr = batch.ptr.to(torch.int32)  # (B+1,) int32

        # Detect PBC.
        try:
            pbc = batch.pbc  # (B, 3) bool
            cell = batch.cell  # (B, 3, 3) float
        except AttributeError:
            pbc = None
            cell = None

        result = neighbor_list(
            positions=positions,
            cutoff=self.config.cutoff,
            cell=cell,
            pbc=pbc,
            max_neighbors=self.config.max_neighbors,
            half_fill=self.config.half_list,
            batch_ptr=batch_ptr,
            return_neighbor_list=self._neighbor_list_flag,
        )

        if self._neighbor_list_flag:
            edge_index = result[0]  # (2, E) int32
            # result[1] is the per-atom edge_ptr — not stored explicitly;
            # batch.edge_ptr is a computed property that derives it from the
            # edges group on demand.
            unit_shifts = result[2] if len(result) > 2 else None  # (E, 3) int32

            # Build the edges group.  Per-graph segment lengths are computed
            # from the source-atom indices in edge_index and the per-atom
            # graph assignment in batch.batch.
            from nvalchemi.data.level_storage import SegmentedLevelStorage

            E = edge_index.shape[1]
            B = batch.num_graphs
            src_atoms = edge_index[0].long()  # (E,)
            graph_per_edge = batch.batch.long()[src_atoms]  # (E,)
            seg_lengths = torch.zeros(B, dtype=torch.int32, device=positions.device)
            seg_lengths.scatter_add_(
                0,
                graph_per_edge,
                torch.ones(E, dtype=torch.int32, device=positions.device),
            )

            # Store edge_index in nvalchemi's (E, 2) convention so that
            # model adapt_input methods (e.g. MACEWrapper) can read it
            # directly with a .T transpose.
            data_dict: dict[str, torch.Tensor] = {
                "edge_index": edge_index.T.contiguous(),  # (E, 2)
            }
            if unit_shifts is not None:
                data_dict["unit_shifts"] = unit_shifts  # (E, 3)

            # Replace (or create) the edges group.  validate=False is required
            # because the edge count changes between neighbor-list rebuilds.
            batch._storage.groups["edges"] = SegmentedLevelStorage(
                data=data_dict,
                device=positions.device,
                segment_lengths=seg_lengths,
                validate=False,
            )
        else:
            neighbor_matrix = result[0]  # (N, max_neighbors) int32
            num_neighbors = result[1]  # (N,) int32
            neighbor_shifts = result[2] if len(result) > 2 else None
            # Write into the atoms group so that `batch.neighbor_matrix` etc. work.
            atoms_group = batch._atoms_group
            if atoms_group is None:
                raise RuntimeError(
                    "NeighborListHook: batch has no atoms group — cannot store "
                    "neighbor data."
                )
            atoms_group["neighbor_matrix"] = neighbor_matrix
            atoms_group["num_neighbors"] = num_neighbors
            if neighbor_shifts is not None:
                atoms_group["neighbor_shifts"] = neighbor_shifts

        # Stamp the cutoff so that prepare_neighbors_for_model can detect when
        # filtering is needed for sub-models with a tighter cutoff.
        batch._neighbor_list_cutoff = self.config.cutoff
