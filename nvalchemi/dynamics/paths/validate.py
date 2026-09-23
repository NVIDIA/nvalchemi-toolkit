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
"""Validation helpers for reaction-path batches."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from nvalchemi.data import Batch


# Cells may acquire small floating-point differences through dtype conversion or
# serialization. These tolerances accept numerical noise without treating a
# physically different lattice as the same path cell.
_CELL_RTOL = 1e-5
_CELL_ATOL = 1e-6


def validate_paths(batch: Batch) -> None:
    """Read-only validation of grouped image paths for path-related calculations.

    Each group in the batch is interpreted as one path. A valid path contains
    at least three images, with the same number and ordering of atoms in every
    image. Periodic images must also have consistent cells and periodic-boundary
    settings within each path. A missing ``pbc`` field means that every image is
    nonperiodic. This function never adds fields to or otherwise mutates the
    batch.

    Parameters
    ----------
    batch : Batch
        Grouped batch whose graphs are the images along one or more paths.

    Raises
    ------
    ValueError
        If the batch has no group layout, paths, or atomic numbers; a path has
        fewer than three images; or the images in a path do not have matching
        atoms, cells, or periodic-boundary settings.
    """
    if "atomic_numbers" not in batch:
        raise ValueError("Reaction paths require atomic_numbers")

    layout = batch.group_layout
    if layout.num_groups == 0:
        raise ValueError("A reaction-path batch requires at least one path")
    if torch.any(layout.num_graphs_per_group < 3):
        raise ValueError("Every reaction path must contain at least three images")

    cells = batch.cell if "cell" in batch else None
    pbcs = batch.pbc if "pbc" in batch else None
    if cells is not None and cells.shape != (batch.num_graphs, 3, 3):
        raise ValueError(
            f"cell must have shape ({batch.num_graphs}, 3, 3), got {tuple(cells.shape)}"
        )
    if pbcs is not None and pbcs.shape != (batch.num_graphs, 3):
        raise ValueError(
            f"pbc must have shape ({batch.num_graphs}, 3), got {tuple(pbcs.shape)}"
        )
    if cells is None and pbcs is not None and torch.any(pbcs):
        raise ValueError("Periodic paths require a cell")

    # Generate `first_image` with shape (num_graphs,)
    first_image = layout.group_ptr[layout.group_idx]
    counts = batch.num_nodes_per_graph
    if not torch.all(counts == counts[first_image]):
        raise ValueError("All images in a reaction path must have the same atom count")

    if pbcs is not None and not torch.all(pbcs == pbcs[first_image]):
        raise ValueError(
            "All images in a reaction path must have identical PBC settings"
        )

    if cells is not None and not torch.all(
        torch.isclose(
            cells,
            cells[first_image],
            rtol=_CELL_RTOL,
            atol=_CELL_ATOL,
        )
    ):
        raise ValueError("All images in a reaction path must have the same cell")

    node_image = batch.batch_idx.long()
    node_rank = (
        torch.arange(batch.num_nodes, device=batch.device)
        - batch.batch_ptr[node_image].long()
    )
    first_node = batch.batch_ptr[layout.group_ptr[layout.node_to_group]].long()
    reference_node = first_node + node_rank
    if not torch.equal(batch.atomic_numbers, batch.atomic_numbers[reference_node]):
        raise ValueError(
            "All images in a reaction path must have identical atomic numbers"
        )
