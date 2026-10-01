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

import math
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass

import torch
from nvalchemiops.neighbors import estimate_max_neighbors
from nvalchemiops.torch.neighbors import neighbor_list
from torch import Tensor

_TYPE_PAD = torch.iinfo(torch.int32).min


@dataclass(frozen=True)
class DescriptorTile:
    """One successful batched neighbor query and its distance-sorted rows.

    ``distances`` and optional ``neighbor_types`` have one row per tile atom
    and only ``valid_width`` columns. Each system's active width is in
    ``widths``; rows are padded to the tile maximum with cutoff and
    ``_TYPE_PAD``. ``atom_offsets`` indexes systems in these flat row arrays.
    ``estimated_peak_bytes`` is the builder's workspace sizing estimate;
    ``geometry_bytes`` is the exact size of the returned descriptor tensors.
    """

    distances: Tensor
    neighbor_types: Tensor | None
    center_types: Tensor | None
    atom_offsets: tuple[int, ...]
    widths: tuple[int, ...]
    capacity: int
    valid_width: int
    estimated_peak_bytes: int
    geometry_bytes: int

    def layout(self, *, typed_neighbors: bool) -> tuple[Tensor, Tensor | None]:
        """Return this tile's rows in distance or type-grouped order.

        This rearranges the stored rows and reuses the neighbor geometry.

        Parameters
        ----------
        typed_neighbors : bool
            If True and neighbor types are present, group valid neighbors by
            ascending type ID. Otherwise, retain ascending distance order.

        Returns
        -------
        distances : Tensor
            Distances with shape ``[tile_atoms, valid_width]``. Padding uses
            the cutoff value stored in the tile.
        neighbor_types : Tensor or None
            Type IDs aligned with ``distances``; padding uses ``_TYPE_PAD``.
            None when the tile has no type data.

        Notes
        -----
        Stable sorts preserve ascending distance within each type group and
        move padded entries after valid neighbors.
        """
        if not typed_neighbors or self.neighbor_types is None or not self.valid_width:
            return self.distances, self.neighbor_types

        valid = self.neighbor_types != _TYPE_PAD
        by_type = torch.argsort(self.neighbor_types, dim=1, stable=True)
        distances = self.distances.gather(1, by_type)
        neighbor_types = self.neighbor_types.gather(1, by_type)
        valid = valid.gather(1, by_type)
        # Type sorting also moves the minimum-int padding sentinel first.
        # This stable validity sort moves those entries after all real labels
        # while preserving the type groups and their distance order.
        active_first = torch.argsort(
            valid.to(torch.int8), dim=1, descending=True, stable=True
        )
        return (
            distances.gather(1, active_first).contiguous(),
            neighbor_types.gather(1, active_first).contiguous(),
        )


def _initial_capacity(
    max_atoms_per_system: int,
    cutoff: float,
    *,
    cell_volume: float | None,
    periodic: bool,
) -> int:
    """Choose an initial neighbor capacity for each atom in a structure.

    Parameters
    ----------
    max_atoms_per_system : int
        Number of atoms in the structure; also bounds the fallback capacity.
    cutoff : float
        Neighbor cutoff in the same length unit as the coordinates.
    cell_volume : float or None
        Periodic cell volume in the cube of the coordinate length unit.
        None disables the density-based estimate.
    periodic : bool
        Whether the structure uses periodic geometry.

    Returns
    -------
    int
        Seed capacity per atom, before any overflow retry.

    Notes
    -----
    The fallback is the larger of 16 and the structure's atom count, capped at
    64 slots. For periodic cells, the estimated density and cutoff can raise
    that seed. The neighbor query still checks for overflow before accepting
    the capacity.
    """
    capacity = min(max(16, max_atoms_per_system), 64)
    if not periodic or max_atoms_per_system == 0 or cell_volume is None:
        return capacity
    if not math.isfinite(cell_volume) or cell_volume <= 0:
        return capacity
    density = (max_atoms_per_system / cell_volume) * 1.35
    if not math.isfinite(density) or density <= 0:
        return capacity
    return max(
        capacity,
        estimate_max_neighbors(cutoff, atomic_density=density),
    )


def _estimated_peak_bytes(atom_count: int, capacity: int, system_count: int) -> int:
    """Estimate neighbor-query and descriptor workspace for one tile.

    Parameters
    ----------
    atom_count : int
        Total atoms in the tile.
    capacity : int
        Neighbor slots allowed per atom by the batched query.
    system_count : int
        Number of structures represented by the tile.

    Returns
    -------
    int
        Estimated workspace bytes for the tile.

    Notes
    -----
    The slot allowance includes Ops indices and shifts plus distance, mask,
    gather, and sorting buffers. Ops cell-list internals are not exposed by its
    public API, so this is a sizing estimate rather than a measured allocator
    peak.
    """
    return atom_count * capacity * 128 + atom_count * 64 + system_count * 64


def _tile_ranges(
    pointers: Sequence[int],
    capacity_seeds: Sequence[int],
    memory_budget_bytes: int | None,
) -> Iterator[tuple[int, int]]:
    """Partition structures into consecutive ranges for geometry construction.

    Parameters
    ----------
    pointers : sequence of int
        Cumulative atom offsets, starting at zero. Adjacent values delimit a
        structure.
    capacity_seeds : sequence of int
        Initial per-atom neighbor capacities, one per structure.
    memory_budget_bytes : int or None
        Estimated workspace allowance for one tile. None yields each
        structure separately without budget-based grouping.

    Yields
    ------
    tuple of int
        Half-open structure-index range ``(start, stop)``. Ranges preserve
        input order and never split a structure.

    Raises
    ------
    MemoryError
        If one structure exceeds the estimated workspace allowance.

    Notes
    -----
    With a finite budget, consecutive structures are greedily combined while
    their atom counts and seeded capacities fit the estimate.
    """
    if memory_budget_bytes is None:
        for index in range(len(pointers) - 1):
            yield index, index + 1
        return

    start = 0
    structure_count = len(pointers) - 1
    while start < structure_count:
        stop = start
        capacity = 16
        while stop < structure_count:
            next_capacity = max(capacity, capacity_seeds[stop])
            atom_count = pointers[stop + 1] - pointers[start]
            estimated = _estimated_peak_bytes(
                atom_count, next_capacity, stop + 1 - start
            )
            if estimated > memory_budget_bytes:
                if stop == start:
                    raise MemoryError(
                        f"one structure needs an estimated {estimated} bytes, "
                        f"above the configured {memory_budget_bytes}-byte device budget"
                    )
                break
            capacity = next_capacity
            stop += 1
        yield start, stop
        start = stop


def _build_tile(
    positions: Tensor,
    cells: Tensor | None,
    pbc: Tensor,
    atom_types: Tensor | None,
    pointers: Sequence[int],
    capacity_seeds: Sequence[int],
    structure_start: int,
    structure_stop: int,
    cutoff: float,
    device: torch.device,
    memory_budget_bytes: int | None,
    *,
    initial_capacity: int | None = None,
) -> Iterator[DescriptorTile]:
    """Build descriptor rows for a half-open range of structures.

    Parameters
    ----------
    positions : Tensor
        Coordinates with shape ``[num_atoms, 3]``. Their length unit must
        match ``cutoff`` (angstrom for CSP comparisons).
    cells : Tensor or None
        Optional cells with shape ``[num_structures, 3, 3]``. Missing cells
        disable periodicity.
    pbc : Tensor
        Periodic-axis flags with shape ``[num_structures, 3]``.
    atom_types : Tensor or None
        Optional integer type ID per atom, shape ``[num_atoms]``.
    pointers : sequence of int
        Cumulative atom offsets for all structures.
    capacity_seeds : sequence of int
        Initial neighbor capacities, one per structure.
    structure_start, structure_stop : int
        Half-open structure-index range to build.
    cutoff : float
        Neighbor cutoff in the coordinate length unit.
    device : torch.device
        Device for copied geometry and neighbor-query outputs.
    memory_budget_bytes : int or None
        Estimated workspace allowance. None disables the estimate-based limit.
    initial_capacity : int or None, keyword-only
        Capacity for an overflow retry; None uses the seed for the range.

    Yields
    ------
    DescriptorTile
        Tile-local atom offsets, per-structure maximum widths, and rows of
        FP32 distances. Rows are padded to the tile's maximum width with the
        cutoff; typed neighbor padding uses ``_TYPE_PAD``.

    Raises
    ------
    MemoryError
        If one structure exceeds the estimated geometry budget.
    RuntimeError
        If a full-capacity neighbor row cannot be retried at a larger capacity.
    ValueError
        If converted positions or periodic cells are invalid, or a distinct
        atom pair has zero distance.

    Notes
    -----
    Geometry arithmetic runs in FP32 on ``device``. Periodic image shifts are
    applied through each atom's cell before distances are measured. When any
    structure reaches the current capacity, the query is retried because a
    full row may have been truncated. If the larger estimate exceeds budget,
    a multi-structure range is split recursively. Workspace values are
    estimates, not measured allocator usage.
    """
    host_ptr = [pointers[structure_start] - pointers[structure_start]]
    host_ptr.extend(
        pointers[index] - pointers[structure_start]
        for index in range(structure_start + 1, structure_stop + 1)
    )
    system_count = structure_stop - structure_start
    atom_start = pointers[structure_start]
    atom_stop = pointers[structure_stop]
    atom_count = atom_stop - atom_start
    capacity = initial_capacity or max(
        capacity_seeds[structure_start:structure_stop], default=16
    )

    estimated = _estimated_peak_bytes(atom_count, capacity, system_count)
    if memory_budget_bytes is not None and estimated > memory_budget_bytes:
        if system_count > 1:
            midpoint = structure_start + system_count // 2
            yield from _build_tile(
                positions,
                cells,
                pbc,
                atom_types,
                pointers,
                capacity_seeds,
                structure_start,
                midpoint,
                cutoff,
                device,
                memory_budget_bytes,
            )
            yield from _build_tile(
                positions,
                cells,
                pbc,
                atom_types,
                pointers,
                capacity_seeds,
                midpoint,
                structure_stop,
                cutoff,
                device,
                memory_budget_bytes,
            )
            return
        raise MemoryError(
            f"one structure needs an estimated {estimated} bytes, above the "
            f"configured {memory_budget_bytes}-byte device budget"
        )

    tile_positions = positions[atom_start:atom_stop].to(
        device=device, dtype=torch.float32, copy=True
    )
    tile_pbc = pbc[structure_start:structure_stop].to(
        device=device, dtype=torch.bool, copy=True
    )
    if not torch.isfinite(tile_positions).all():
        raise ValueError("positions must remain finite after FP32 conversion")

    # Nonperiodic systems use an inert identity cell. This also lets one Ops
    # batch carry different PBC flags and cells without imposing invertibility
    # on a cell that does not participate in the geometry.
    tile_cells = (
        torch.eye(3, dtype=torch.float32, device=device)
        .expand(system_count, 3, 3)
        .clone()
    )
    if cells is not None:
        source_cells = cells[structure_start:structure_stop].to(
            device=device, dtype=torch.float32, copy=True
        )
        periodic = tile_pbc.any(dim=1)
        if periodic.any():
            periodic_cells = source_cells[periodic]
            if not torch.isfinite(periodic_cells).all():
                raise ValueError(
                    "periodic cell must remain finite after FP32 conversion"
                )
            try:
                inverses = torch.linalg.inv(periodic_cells)
            except RuntimeError as exc:
                raise ValueError("periodic cell must be invertible in FP32") from exc
            if not torch.isfinite(inverses).all():
                raise ValueError("periodic cell must have a finite FP32 inverse")
            tile_cells[periodic] = periodic_cells

    if atom_types is None:
        tile_types = None
    else:
        tile_types = atom_types[atom_start:atom_stop].to(
            device=device, dtype=torch.int32, copy=True
        )

    tile_ptr = torch.tensor(host_ptr, dtype=torch.int32, device=device)
    batch_idx = torch.repeat_interleave(
        torch.arange(system_count, dtype=torch.int32, device=device),
        tile_ptr[1:] - tile_ptr[:-1],
    )

    # There are no centers to query when every structure is empty.
    if atom_count == 0:
        empty = tile_positions.new_empty((0, 0))
        empty_types = (
            torch.empty((0, 0), dtype=torch.int32, device=device)
            if tile_types is not None
            else None
        )
        center_types = tile_types
        yield DescriptorTile(
            empty,
            empty_types,
            center_types,
            tuple(host_ptr),
            tuple(0 for _ in range(system_count)),
            capacity,
            0,
            estimated,
            0,
        )
        return

    method = "batch_cell_list" if device.type == "cuda" else "batch_naive"
    result = neighbor_list(
        positions=tile_positions.contiguous(),
        cutoff=cutoff,
        cell=tile_cells.contiguous(),
        pbc=tile_pbc.contiguous(),
        batch_idx=batch_idx.contiguous(),
        batch_ptr=tile_ptr,
        max_neighbors=capacity,
        half_fill=False,
        method=method,
    )
    matrix, counts = result[:2]
    shifts = result[2] if len(result) >= 3 else None
    if counts.numel():
        # Only the maximum count per structure needs to cross to the host;
        # reducing by the tile's batch index avoids copying one count per atom.
        widths_device = torch.zeros(system_count, dtype=counts.dtype, device=device)
        widths_device.scatter_reduce_(
            0, batch_idx.to(torch.int64), counts, reduce="amax", include_self=True
        )
        widths = tuple(int(value) for value in widths_device.detach().cpu().tolist())
    else:
        widths = tuple(0 for _ in range(system_count))
    max_count = max(widths, default=0)
    # Ops uses max_neighbors as a hard row capacity, so equality cannot prove
    # that the full neighbor row was returned; retry before accepting it.
    if max_count >= capacity:
        del result, matrix, counts, shifts
        if capacity >= torch.iinfo(torch.int32).max:
            raise RuntimeError(
                f"neighbor-list capacity overflow for {atom_count} atoms at {capacity}"
            )
        next_capacity = min(capacity * 2, torch.iinfo(torch.int32).max)
        next_estimate = _estimated_peak_bytes(atom_count, next_capacity, system_count)
        if memory_budget_bytes is not None and next_estimate > memory_budget_bytes:
            if system_count > 1:
                midpoint = structure_start + system_count // 2
                yield from _build_tile(
                    positions,
                    cells,
                    pbc,
                    atom_types,
                    pointers,
                    capacity_seeds,
                    structure_start,
                    midpoint,
                    cutoff,
                    device,
                    memory_budget_bytes,
                )
                yield from _build_tile(
                    positions,
                    cells,
                    pbc,
                    atom_types,
                    pointers,
                    capacity_seeds,
                    midpoint,
                    structure_stop,
                    cutoff,
                    device,
                    memory_budget_bytes,
                )
                return
            raise MemoryError(
                f"neighbor capacity retry needs an estimated {next_estimate} bytes, "
                f"above the configured {memory_budget_bytes}-byte device budget"
            )
        yield from _build_tile(
            positions,
            cells,
            pbc,
            atom_types,
            pointers,
            capacity_seeds,
            structure_start,
            structure_stop,
            cutoff,
            device,
            memory_budget_bytes,
            initial_capacity=next_capacity,
        )
        return

    valid_width = max(widths, default=0)
    if valid_width:
        # Trim allocator overcapacity before distance evaluation and sorting.
        matrix = matrix[:, :valid_width]
        if shifts is not None:
            shifts = shifts[:, :valid_width]
        columns = torch.arange(valid_width, dtype=counts.dtype, device=device)
        valid = columns.unsqueeze(0) < counts.unsqueeze(1)
        # Invalid slots can contain negative or out-of-range indices. Clamp
        # before gathering, then replace those distances/types using ``valid``.
        safe_indices = matrix.clamp(min=0, max=atom_count - 1).to(torch.int64)
        flat_indices = safe_indices.reshape(-1)
        delta = tile_positions.index_select(0, flat_indices).reshape(
            atom_count, valid_width, 3
        ) - tile_positions.unsqueeze(1)
        if shifts is not None:
            atom_cells = tile_cells.index_select(0, batch_idx.to(torch.int64))
            translations = torch.matmul(
                shifts.to(tile_positions.dtype).unsqueeze(-2),
                atom_cells.unsqueeze(1),
            ).squeeze(-2)
            delta = delta + translations
        distances = torch.linalg.vector_norm(delta, dim=-1)
        if torch.any(valid & (distances <= 0)):
            raise ValueError("comparison rejects distinct atoms at zero distance")
        cutoff_value = torch.as_tensor(cutoff, dtype=distances.dtype, device=device)
        distances = torch.where(valid, distances, cutoff_value)
        neighbor_types = (
            tile_types.index_select(0, flat_indices).reshape(atom_count, valid_width)
            if tile_types is not None
            else None
        )
        if neighbor_types is not None:
            neighbor_types = torch.where(
                valid,
                neighbor_types,
                torch.full_like(neighbor_types, _TYPE_PAD),
            )
        by_distance = torch.argsort(distances, dim=1, stable=True)
        distances = distances.gather(1, by_distance).contiguous()
        if neighbor_types is not None:
            neighbor_types = neighbor_types.gather(1, by_distance).contiguous()
        del safe_indices, flat_indices, delta, shifts, matrix
    else:
        distances = tile_positions.new_empty((atom_count, 0))
        neighbor_types = (
            torch.empty((atom_count, 0), dtype=torch.int32, device=device)
            if tile_types is not None
            else None
        )
        del matrix, shifts

    geometry_bytes = distances.numel() * distances.element_size()
    if neighbor_types is not None:
        geometry_bytes += neighbor_types.numel() * neighbor_types.element_size()
    if tile_types is not None:
        geometry_bytes += tile_types.numel() * tile_types.element_size()
    yield DescriptorTile(
        distances,
        neighbor_types,
        tile_types,
        tuple(host_ptr),
        widths,
        capacity,
        valid_width,
        estimated,
        geometry_bytes,
    )


def build_descriptor_tiles(
    positions: Tensor,
    pointers: Iterable[int],
    cells: Tensor | None,
    pbc: Tensor,
    atom_types: Tensor | None,
    *,
    cutoff: float,
    device: torch.device,
    memory_budget_bytes: int | None,
) -> Iterator[DescriptorTile]:
    """Yield bounded geometry tiles shared by comparison descriptor layouts.

    Parameters
    ----------
    positions : Tensor
        Coordinates with shape ``[num_atoms, 3]``.
    pointers : iterable of int
        Cumulative atom counts, starting at zero and ending at ``num_atoms``.
    cells : Tensor or None
        Optional periodic cells with shape ``[num_structures, 3, 3]``. None
        disables periodicity even if ``pbc`` contains True values.
    pbc : Tensor
        Periodic-axis flags with shape ``[num_structures, 3]``.
    atom_types : Tensor or None
        Optional integer type IDs with shape ``[num_atoms]``.
    cutoff : float
        Neighbor cutoff in the same length unit as ``positions``.
    device : torch.device
        Device used for geometry construction and yielded tiles.
    memory_budget_bytes : int or None
        Estimated workspace limit for a tile; None disables budget-based
        grouping and splitting.

    Yields
    ------
    DescriptorTile
        One or more consecutive structures with tile-local atom offsets,
        distance-sorted FP32 rows, and optional aligned type IDs. Rows are
        padded to the tile width with ``cutoff`` and ``_TYPE_PAD``.

    Raises
    ------
    ValueError
        If pointers or tensor shapes are invalid, positions or periodic cells
        are invalid after FP32 conversion, or a valid distinct pair has zero
        distance.
    MemoryError
        If even one structure exceeds the estimated workspace limit.
    RuntimeError
        If neighbor capacity reaches its integer limit before a row fits.

    Notes
    -----
    Each successful batched Ops query supplies geometry reusable by untyped,
    center-typed, and fully typed layouts. Capacity retries and budget-driven
    splitting happen inside the iterator. A missing cell disables periodicity.
    """
    pointers = tuple(int(value) for value in pointers)
    if len(pointers) < 2 or pointers[0] != 0:
        raise ValueError("pointers must start at zero and include every structure")
    if any(right < left for left, right in zip(pointers, pointers[1:])):
        raise ValueError("pointers must be nondecreasing")
    if pointers[-1] != positions.shape[0]:
        raise ValueError("last pointer must equal the number of positions")
    if pbc.shape != (len(pointers) - 1, 3):
        raise ValueError("pbc must have shape [num_structures, 3]")
    if cells is not None and cells.shape != (len(pointers) - 1, 3, 3):
        raise ValueError("cells must have shape [num_structures, 3, 3]")
    if atom_types is not None and atom_types.shape != (positions.shape[0],):
        raise ValueError("atom_types must have shape [num_atoms]")
    if not _is_positive_finite(cutoff):
        raise ValueError("cutoff must be a positive finite scalar")
    if memory_budget_bytes is not None and memory_budget_bytes <= 0:
        raise MemoryError("device memory budget leaves no space for a geometry tile")

    geometry_pbc = torch.zeros_like(pbc, dtype=torch.bool) if cells is None else pbc
    pbc_cpu = geometry_pbc.detach().to(device="cpu", dtype=torch.bool)
    if cells is None:
        cell_volumes = [1.0] * (len(pointers) - 1)
    else:
        cell_volumes = (
            torch.linalg.det(cells.detach().to(dtype=torch.float64))
            .abs()
            .to(device="cpu")
            .tolist()
        )
    capacity_seeds = tuple(
        _initial_capacity(
            pointers[index + 1] - pointers[index],
            cutoff,
            cell_volume=cell_volumes[index],
            periodic=bool(pbc_cpu[index].any()),
        )
        for index in range(len(pointers) - 1)
    )

    for structure_start, structure_stop in _tile_ranges(
        pointers, capacity_seeds, memory_budget_bytes
    ):
        yield from _build_tile(
            positions,
            cells,
            geometry_pbc,
            atom_types,
            pointers,
            capacity_seeds,
            structure_start,
            structure_stop,
            cutoff,
            device,
            memory_budget_bytes,
        )


def _is_positive_finite(value: float) -> bool:
    """Return whether ``value`` is a positive finite Python number.

    Parameters
    ----------
    value : float
        Scalar to check.

    Returns
    -------
    bool
        True for positive finite integers and floats, excluding booleans.
    """
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and value > 0
        and math.isfinite(value)
    )


def build_structure_descriptor(
    positions: Tensor,
    cell: Tensor | None,
    pbc: Tensor,
    cutoff: float,
    atom_types: Tensor | None,
    typed_neighbors: bool,
    memory_budget_bytes: int | None = None,
) -> tuple[Tensor, Tensor | None]:
    """Build sorted radial rows for one structure.

    Parameters
    ----------
    positions : Tensor
        Atomic coordinates with shape ``[num_atoms, 3]``.
    cell : Tensor or None
        Optional periodic cell with shape ``[3, 3]``.
    pbc : Tensor
        Periodic-axis flags with shape ``[3]`` or ``[1, 3]``.
    cutoff : float
        Neighbor cutoff in the coordinate length unit.
    atom_types : Tensor or None
        Optional integer type ID per atom, shape ``[num_atoms]``.
    typed_neighbors : bool
        If True and types are supplied, group neighbors by type while retaining
        distance order within each type.
    memory_budget_bytes : int or None, optional
        Estimated workspace limit for geometry construction.

    Returns
    -------
    distances : Tensor
        Ascending distances with shape ``[num_atoms, width]``. Missing slots
        are filled with ``cutoff``.
    neighbor_types : Tensor or None
        Aligned int32 type IDs; padding uses the minimum int32 value. None when
        ``atom_types`` is None or the structure is empty.

    Raises
    ------
    ValueError
        If the cutoff or converted geometry is invalid, or distinct atoms have
        zero distance.
    MemoryError
        If the estimated geometry workspace exceeds ``memory_budget_bytes``.
    RuntimeError
        If neighbor capacity reaches its integer limit before a row fits.
    """
    n_atoms = positions.shape[0]
    if n_atoms == 0:
        return positions.new_empty((0, 0)), None
    positions_device = positions.device
    if pbc.ndim == 1:
        pbc = pbc.reshape(1, 3)
    pointers = (0, n_atoms)
    cells = cell.reshape(1, 3, 3).to(device="cpu") if cell is not None else None
    tile = next(
        build_descriptor_tiles(
            positions.detach().to(device="cpu", dtype=torch.float32),
            pointers,
            cells,
            pbc.detach().to(device="cpu", dtype=torch.bool),
            atom_types.detach().to(device="cpu", dtype=torch.int32)
            if atom_types is not None
            else None,
            cutoff=cutoff,
            device=positions_device,
            memory_budget_bytes=memory_budget_bytes,
        )
    )
    distances = tile.distances
    neighbors = tile.neighbor_types
    if typed_neighbors:
        distances, neighbors = tile.layout(typed_neighbors=True)
    return distances, neighbors
