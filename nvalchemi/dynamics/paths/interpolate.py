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
"""Construction and interpolation of reaction paths."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor

from nvalchemi.data import Batch
from nvalchemi.dynamics.paths._geometry import minimum_image_displacement

_CELL_RTOL = 1e-5
_CELL_ATOL = 1e-6
_STALE_FIELDS = frozenset(
    {
        "forces",
        "energy",
        "stress",
        "virial",
        "dipole",
        "node_embeddings",
        "graph_embeddings",
        "velocities",
        "momenta",
        "kinetic_energies",
    }
)


def _normalize_num_images(num_images: int | Sequence[int], num_paths: int) -> list[int]:
    """Return one total image count, including endpoints, for every path."""
    if isinstance(num_images, bool):
        raise TypeError("num_images must be an integer or a sequence of integers")
    if isinstance(num_images, int):
        counts = [num_images] * num_paths
    elif isinstance(num_images, Sequence) and not isinstance(num_images, (str, bytes)):
        counts = list(num_images)
        if len(counts) != num_paths:
            raise ValueError(
                "num_images must contain one value per path; "
                f"expected {num_paths}, got {len(counts)}"
            )
    else:
        raise TypeError("num_images must be an integer or a sequence of integers")

    if any(isinstance(count, bool) or not isinstance(count, int) for count in counts):
        raise TypeError("Every num_images value must be an integer")
    if any(count < 3 for count in counts):
        raise ValueError("Every num_images value must be at least three")
    return counts


def _validate_endpoint_geometry(
    initial: Batch, final: Batch
) -> tuple[Tensor | None, Tensor]:
    """Validate and return cells and PBC shared by all endpoint pairs."""
    initial_has_cell = "cell" in initial
    final_has_cell = "cell" in final
    if initial_has_cell != final_has_cell:
        raise ValueError(
            "Paired endpoints must either both define a cell or neither do"
        )
    initial_has_pbc = "pbc" in initial
    final_has_pbc = "pbc" in final
    if initial_has_pbc != final_has_pbc:
        raise ValueError("Paired endpoints must either both define PBC or neither do")

    num_paths = initial.num_graphs
    cell = initial.cell if initial_has_cell else None
    final_cell = final.cell if final_has_cell else None
    pbc = (
        initial.pbc
        if initial_has_pbc
        else torch.zeros((num_paths, 3), dtype=torch.bool, device=initial.device)
    )
    final_pbc = final.pbc if final_has_pbc else pbc
    if cell is not None and cell.shape != (num_paths, 3, 3):
        raise ValueError(f"cell must have shape ({num_paths}, 3, 3)")
    if final_cell is not None and final_cell.shape != (num_paths, 3, 3):
        raise ValueError(f"cell must have shape ({num_paths}, 3, 3)")
    if pbc.shape != (num_paths, 3) or final_pbc.shape != (num_paths, 3):
        raise ValueError(f"pbc must have shape ({num_paths}, 3)")
    if pbc.dtype != torch.bool or final_pbc.dtype != torch.bool:
        raise TypeError("pbc must use the bool dtype")
    if not torch.equal(pbc, final_pbc):
        raise ValueError("Paired endpoints must have identical PBC settings")
    if (
        cell is not None
        and final_cell is not None
        and not torch.allclose(cell, final_cell, rtol=_CELL_RTOL, atol=_CELL_ATOL)
    ):
        raise ValueError("Paired endpoints must have the same cell")
    periodic = torch.any(pbc, dim=1)
    if torch.any(periodic):
        if cell is None:
            raise ValueError("Periodic endpoints must define a cell")
        try:
            torch.linalg.inv(cell[periodic])
        except RuntimeError as error:
            raise ValueError("Periodic endpoint cells must be invertible") from error
    return cell, pbc


def _discard_stale_fields(batch: Batch) -> None:
    """Remove edge data, model outputs, and dynamical state from path images."""
    edge_group = batch._edges_group
    edge_fields = set() if edge_group is None else set(edge_group.keys())
    discarded = edge_fields | _STALE_FIELDS
    for field in discarded:
        if field in batch:
            del batch[field]
    if batch.keys is not None:
        for fields in batch.keys.values():
            fields.difference_update(discarded)


def interpolate_paths(
    initial: Batch,
    final: Batch,
    num_images: int | Sequence[int],
) -> Batch:
    """Interpolate paired endpoint batches into grouped reaction paths.

    Graph ``i`` in ``initial`` is paired with graph ``i`` in ``final`` and
    becomes one path. Atomic correspondence is index-based: paired graphs must
    contain identical atomic numbers in identical order. Periodic endpoint
    displacements use the minimum-image convention.

    Parameters
    ----------
    initial, final : Batch
        Batches containing corresponding initial and final structures.
    num_images : int or sequence of int
        Total number of images per path, including both endpoints. An integer
        applies the same count to every path. A sequence supplies one count per
        paired endpoint graph. Every count must be at least three.

    Returns
    -------
    Batch
        Images ordered path-by-path with a populated group layout.

    Raises
    ------
    TypeError
        If ``num_images`` is not an integer or integer sequence.
    ValueError
        If the endpoint batches or interpolation options are incompatible.

    Notes
    -----
    Structural and model-input fields are copied from each initial graph.
    Neighbor data, model outputs, and dynamical state are discarded.
    Endpoint coordinates are retained exactly, even when periodic interior
    images follow an unwrapped minimum-image path. The input endpoint batches
    are never modified.
    """
    if initial.num_graphs == 0 or final.num_graphs == 0:
        raise ValueError("Endpoint batches must contain at least one graph")
    if initial.num_graphs != final.num_graphs:
        raise ValueError("Endpoint batches must contain the same number of graphs")
    if initial.device != final.device:
        raise ValueError("Endpoint batches must be on the same device")
    image_counts = _normalize_num_images(num_images, initial.num_graphs)
    if initial.positions.dtype != final.positions.dtype:
        raise ValueError("Paired endpoints must use the same position dtype")
    if not torch.equal(initial.num_nodes_per_graph, final.num_nodes_per_graph):
        raise ValueError("Paired endpoints must contain the same number of atoms")
    if not torch.equal(initial.atomic_numbers, final.atomic_numbers):
        raise ValueError(
            "Paired endpoints must have identical atomic numbers in the same order"
        )
    cell, pbc = _validate_endpoint_geometry(initial, final)
    endpoint_displacement = minimum_image_displacement(
        final.positions - initial.positions,
        initial.batch_idx,
        cell,
        pbc,
    )
    interpolation_target = initial.positions + endpoint_displacement
    # Materialize every ragged path directly from segmented batch tensors.
    image_counts_tensor = torch.tensor(
        image_counts,
        dtype=torch.long,
        device=initial.device,
    )
    path_indices = torch.arange(
        initial.num_graphs, dtype=torch.long, device=initial.device
    )
    image_to_path = torch.repeat_interleave(path_indices, image_counts_tensor)
    image_starts = torch.cumsum(image_counts_tensor, dim=0) - image_counts_tensor
    image_rank = (
        torch.arange(image_to_path.numel(), device=initial.device)
        - image_starts[image_to_path]
    )
    fractions = image_rank.to(initial.positions.dtype) / (
        image_counts_tensor[image_to_path] - 1
    ).to(initial.positions.dtype)

    result = initial.index_select(image_to_path)
    output_graph = result.batch_idx.to(torch.long)
    source_graph = image_to_path[output_graph]
    output_node = torch.arange(
        result.num_nodes, device=result.device
    ) - result.batch_ptr[output_graph].to(torch.long)
    source_node = initial.batch_ptr[source_graph].to(torch.long) + output_node
    start_positions = result.positions
    target_positions = interpolation_target[source_node]
    interpolated = start_positions + fractions[output_graph, None] * (
        target_positions - start_positions
    )
    node_rank = image_rank[output_graph]
    positions = torch.where((node_rank == 0)[:, None], start_positions, interpolated)
    terminal = node_rank == image_counts_tensor[source_graph] - 1
    positions = torch.where(terminal[:, None], final.positions[source_node], positions)

    result.positions = positions
    _discard_stale_fields(result)
    result.set_group_layout(image_to_path)
    return result


__all__ = ["interpolate_paths"]
