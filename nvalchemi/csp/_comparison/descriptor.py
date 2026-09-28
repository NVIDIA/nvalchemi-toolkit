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
"""Bounded radial descriptor construction using the public ops neighbor API."""

from __future__ import annotations

import torch
from nvalchemiops.torch.neighbors import neighbor_list
from torch import Tensor


def build_structure_descriptor(
    positions: Tensor,
    cell: Tensor | None,
    pbc: Tensor,
    cutoff: float,
    atom_types: Tensor | None,
    typed_neighbors: bool,
    memory_budget_bytes: int | None = None,
) -> tuple[Tensor, Tensor | None]:
    """Return sorted cutoff-padded distances and optional neighbor type IDs."""
    n_atoms = positions.shape[0]
    if n_atoms == 0:
        return positions.new_empty((0, 0)), None
    # nvalchemiops uses shifts @ cell; pass Toolkit's row-vector cell directly.
    if cell is None:
        # Ops requires a cell whenever PBC metadata is supplied, even when all
        # flags are false. This identity is inert for nonperiodic searches.
        ops_cell = torch.eye(
            3, dtype=positions.dtype, device=positions.device
        ).unsqueeze(0)
    else:
        ops_cell = cell.contiguous().unsqueeze(0)
    ops_pbc = pbc.to(dtype=torch.bool).reshape(1, 3).contiguous()
    ptr = torch.tensor([0, n_atoms], dtype=torch.int32, device=positions.device)
    batch_idx = torch.zeros(n_atoms, dtype=torch.int32, device=positions.device)
    capacity = min(max(16, n_atoms), 64)
    max_capacity = torch.iinfo(torch.int32).max
    while True:
        # Use a provisional 128-byte-per-slot allowance for Ops outputs (index
        # and image shift), safe indices, displacement vectors, distances,
        # masks, and typed gather/sort buffers. This estimate is not a proven
        # peak bound; overlapping sort/gather temporaries can affect actual use.
        estimated_bytes = n_atoms * capacity * 128 + n_atoms * 64
        if memory_budget_bytes is not None and estimated_bytes > memory_budget_bytes:
            raise MemoryError(
                f"one structure needs an estimated {estimated_bytes} bytes, "
                f"above the configured {memory_budget_bytes}-byte device budget"
            )
        result = neighbor_list(
            positions=positions.contiguous(),
            cutoff=cutoff,
            cell=ops_cell,
            pbc=ops_pbc,
            batch_idx=batch_idx,
            batch_ptr=ptr,
            max_neighbors=capacity,
            half_fill=False,
            method="batch_naive",
        )
        matrix, counts, shifts = result if len(result) == 3 else (*result, None)
        max_count = int(counts.max().item())
        if max_count >= capacity:
            if capacity >= max_capacity:
                raise RuntimeError(
                    f"neighbor-list capacity overflow for {n_atoms} atoms at {capacity}"
                )
            next_capacity = min(capacity * 2, max_capacity)
            del result, matrix, counts, shifts
            capacity = next_capacity
            continue
        break

    columns = torch.arange(
        capacity, dtype=torch.int32, device=positions.device
    ).unsqueeze(0)
    valid = columns < counts.unsqueeze(1)
    safe_indices = matrix.clamp(min=0, max=n_atoms - 1).to(torch.int32)
    flat_indices = safe_indices.reshape(-1)
    delta = torch.index_select(positions, 0, flat_indices).reshape(
        n_atoms, capacity, 3
    ) - positions.unsqueeze(1)
    if shifts is not None and cell is not None:
        translation = torch.matmul(shifts.to(positions.dtype), cell.unsqueeze(0))
        delta = delta + translation
    distances = torch.linalg.vector_norm(delta, dim=-1)
    if torch.any(valid & (distances <= 0)):
        raise ValueError("comparison rejects distinct atoms at zero distance")
    distances = torch.where(
        valid, distances, torch.as_tensor(cutoff, device=positions.device)
    )
    if atom_types is None:
        return distances.sort(dim=1).values.contiguous(), None
    neighbor_types = torch.index_select(atom_types, 0, flat_indices).reshape(
        n_atoms, capacity
    )
    del safe_indices, flat_indices
    neighbor_types = torch.where(
        valid,
        neighbor_types,
        torch.full_like(neighbor_types, torch.iinfo(torch.int32).min),
    )
    by_distance = torch.argsort(distances, dim=1, stable=True)
    by_distance = by_distance.to(torch.int32)
    distances = distances.gather(1, by_distance)
    neighbor_types = neighbor_types.gather(1, by_distance)
    if typed_neighbors:
        # Stable lexicographic ordering: type first, then distance.
        by_type = torch.argsort(neighbor_types, dim=1, stable=True)
        by_type = by_type.to(torch.int32)
        distances = distances.gather(1, by_type)
        neighbor_types = neighbor_types.gather(1, by_type)
    return distances.contiguous(), neighbor_types.contiguous()
