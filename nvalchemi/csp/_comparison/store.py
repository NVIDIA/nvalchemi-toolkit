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
"""Owned storage for atom-neighbor distance descriptors."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from functools import lru_cache

import torch
from torch import Tensor

from nvalchemi.csp._comparison.descriptor import build_descriptor_tiles
from nvalchemi.data import Batch

_TYPE_PAD = torch.iinfo(torch.int32).min


@lru_cache(maxsize=64)
def _log_cutoff_fp32(cutoff: float) -> float:
    """Return the FP32 natural logarithm of a descriptor cutoff.

    Parameters
    ----------
    cutoff : float
        Positive cutoff in the same length unit as the input coordinates.

    Returns
    -------
    float
        ``log(cutoff)`` rounded to FP32 and converted back to a Python float.

    Notes
    -----
    The cached value is shared by tensor construction and Warp scoring so
    cutoff padding uses the same endpoint. At most 64 values are cached.
    """
    return float(torch.tensor(math.log(cutoff), dtype=torch.float32))


def workspace_reserve(memory_budget_bytes: int) -> int:
    """Estimate temporary-workspace headroom within a memory budget.

    Parameters
    ----------
    memory_budget_bytes : int
        CUDA memory budget in bytes.

    Returns
    -------
    int
        One eighth of the budget, bounded below by 1 MiB and above by 1 GiB.

    Notes
    -----
    This allowance reduces the budget available to resident descriptor data;
    it does not allocate or reserve memory with CUDA.
    """
    return min(max(1024**2, memory_budget_bytes // 8), 1024**3)


def available_cuda_bytes(device: torch.device | str) -> int:
    """Estimate reusable PyTorch allocation capacity on a CUDA device.

    Parameters
    ----------
    device : torch.device or str
        CUDA device to query.

    Returns
    -------
    int
        Estimated bytes available, capped by total device memory.

    Raises
    ------
    ValueError
        If ``device`` is not a CUDA device.

    Notes
    -----
    The estimate adds driver-free bytes to unused bytes in PyTorch's caching
    allocator. It is not a measurement of exclusively driver-free memory.
    """
    device = torch.device(device)
    if device.type != "cuda":
        raise ValueError("available_cuda_bytes requires a CUDA device")
    driver_free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    reusable_reserved_bytes = max(
        0,
        torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device),
    )
    return min(total_bytes, driver_free_bytes + reusable_reserved_bytes)


def _typed_summary_peak_bytes(
    neighbor_slots: int, center_count: int, type_count: int, rank_count: int
) -> int:
    """Estimate temporary GPU bytes for one structure's typed summary.

    Parameters
    ----------
    neighbor_slots : int
        Number of padded distance/type slots across all centers.
    center_count : int
        Number of atoms serving as centers in the structure.
    type_count : int
        Number of distinct type labels in the shared vocabulary.
    rank_count : int
        Number of neighbor ranks summarized per type pair.

    Returns
    -------
    int
        Estimated temporary bytes, including a fixed 1 MiB allowance.

    Notes
    -----
    The estimate accounts for neighbor ranking buffers, per-center gathers,
    and type-pair extrema. It is a budget preflight, not a measurement or
    reservation of allocator memory.
    """
    center_type_ranks = center_count * type_count * rank_count
    center_types = center_count * type_count
    type_pair_ranks = type_count * type_count * rank_count
    return (
        32 * neighbor_slots
        + 32 * center_type_ranks
        + 32 * center_types
        + 64 * type_pair_ranks
        + 1024**2
    )


@dataclass(frozen=True)
class DescriptorBlock:
    """Flat descriptors for a local structure block, indexed by local ID.

    Distances are FP32 natural logarithms of angstrom distances. ``row_offsets``
    index flattened atom rows (each row has the structure width); ``atom_offsets``
    index ``center_types``. ``neighbor_types`` align with distances and use
    int32 minimum for padding. ``source_ids`` maps local to original indices
    when staged; ``None`` means resident arrays indexed by original IDs.
    """

    distances: Tensor
    neighbor_types: Tensor
    center_types: Tensor
    row_offsets: Tensor
    atom_offsets: Tensor
    widths: Tensor
    atom_counts: Tensor
    source_ids: tuple[int, ...] | None


@dataclass(frozen=True)
class DescriptorStore:
    """Ragged descriptor arrays owned by a comparison index.

    Distances are FP32 natural logarithms of angstrom distances. ``row_offsets``
    index flattened atom rows (each row has the structure width); ``atom_offsets``
    index ``center_types``. ``neighbor_types`` align with distances and use
    int32 minimum for padding. Summary tensors are filtering data, not score
    outputs.
    """

    distances: Tensor
    neighbor_types: Tensor
    center_types: Tensor
    row_offsets: Tensor
    atom_offsets: Tensor
    widths: Tensor
    atom_counts: Tensor
    summaries: Tensor
    typed_summaries: Tensor
    center_type_presence: Tensor
    typed_type_vocab: tuple[int, ...]
    typed_rank_indices: tuple[int, ...]
    device: torch.device

    @property
    def num_structures(self) -> int:
        """Return the number of structure entries in the store.

        Returns
        -------
        int
            Number of entries in ``atom_counts``, including empty structures.
        """
        return int(self.atom_counts.numel())

    @property
    def storage_bytes(self) -> int:
        """Return the total tensor payload size in bytes.

        Returns
        -------
        int
            Sum of tensor storage for descriptor values, ragged offsets, and
            summary arrays.

        Notes
        -----
        Python object overhead and allocator bookkeeping are excluded.
        """
        return sum(
            value.numel() * value.element_size()
            for value in (
                self.distances,
                self.neighbor_types,
                self.center_types,
                self.row_offsets,
                self.atom_offsets,
                self.widths,
                self.atom_counts,
                self.summaries,
                self.typed_summaries,
                self.center_type_presence,
            )
        )

    def bytes_for(self, structure_ids: set[int]) -> int:
        """Estimate descriptor payload bytes for staging selected structures.

        Parameters
        ----------
        structure_ids : set of int
            Original structure indices to include.

        Returns
        -------
        int
            Estimated bytes for FP32 log distances, int32 neighbor types, and
            int32 center types for the selected structures.

        Notes
        -----
        Each neighbor slot contributes eight bytes and each center contributes
        four. Offsets, summaries, transfer workspaces, and allocator overhead
        are excluded.
        """
        if not structure_ids:
            return 0
        ids = sorted(structure_ids)
        return sum(
            (int(self.atom_counts[i]) * int(self.widths[i]) * 8)
            + (int(self.atom_counts[i]) * 4)
            for i in ids
        )

    def block(self, structure_ids: set[int], device: torch.device) -> DescriptorBlock:
        """Return a resident descriptor view or stage selected structures.

        Parameters
        ----------
        structure_ids : set of int
            Original structure indices requested for a block.
        device : torch.device
            Target device for the returned descriptor arrays.

        Returns
        -------
        DescriptorBlock
            On the store's device, contains views of the full arrays and uses
            original structure IDs. On another device, contains packed arrays
            for sorted ``structure_ids`` and maps local IDs through
            ``source_ids``. An empty request returns empty arrays and zero
            offsets.

        Notes
        -----
        Staged tensors remain alive through references held by the returned
        block. A same-device result references storage owned by this store.
        """
        ids = tuple(sorted(structure_ids))
        if not ids:
            empty = torch.empty(0, dtype=torch.float32, device=device)
            empty_i32 = torch.empty(0, dtype=torch.int32, device=device)
            offsets = torch.zeros(1, dtype=torch.int64, device=device)
            return DescriptorBlock(
                empty,
                empty_i32,
                empty_i32,
                offsets,
                offsets,
                empty_i32,
                empty_i32,
                ids,
            )

        if self.device == device:
            return DescriptorBlock(
                self.distances,
                self.neighbor_types,
                self.center_types,
                self.row_offsets,
                self.atom_offsets,
                self.widths,
                self.atom_counts,
                None,
            )

        row_offsets_cpu = self.row_offsets.cpu()
        atom_offsets_cpu = self.atom_offsets.cpu()
        widths_cpu = self.widths.cpu()
        counts_cpu = self.atom_counts.cpu()
        distance_parts: list[Tensor] = []
        neighbor_parts: list[Tensor] = []
        center_parts: list[Tensor] = []
        row_offsets = [0]
        atom_offsets = [0]
        widths: list[int] = []
        counts: list[int] = []
        for structure_id in ids:
            r0 = int(row_offsets_cpu[structure_id])
            r1 = int(row_offsets_cpu[structure_id + 1])
            a0 = int(atom_offsets_cpu[structure_id])
            a1 = int(atom_offsets_cpu[structure_id + 1])
            width = int(widths_cpu[structure_id])
            count = int(counts_cpu[structure_id])
            distance_parts.append(self.distances[r0:r1].cpu())
            neighbor_parts.append(self.neighbor_types[r0:r1].cpu())
            center_parts.append(self.center_types[a0:a1].cpu())
            row_offsets.append(r1 - r0 + row_offsets[-1])
            atom_offsets.append(count + atom_offsets[-1])
            widths.append(width)
            counts.append(count)

        distances = torch.cat(distance_parts).to(device=device, non_blocking=True)
        neighbor_types = torch.cat(neighbor_parts).to(device=device, non_blocking=True)
        center_types = torch.cat(center_parts).to(device=device, non_blocking=True)
        return DescriptorBlock(
            distances,
            neighbor_types,
            center_types,
            torch.tensor(row_offsets, dtype=torch.int64, device=device),
            torch.tensor(atom_offsets, dtype=torch.int64, device=device),
            torch.tensor(widths, dtype=torch.int32, device=device),
            torch.tensor(counts, dtype=torch.int32, device=device),
            ids,
        )


def _snapshot_batch(
    batch: Batch,
    atom_types: Tensor | None,
    device: torch.device,
) -> tuple[list[int], Tensor, Tensor | None, Tensor, Tensor | None]:
    """Copy batch geometry into descriptor-builder-owned tensors.

    Parameters
    ----------
    batch : Batch
        Batched structures whose geometry is snapshotted.
    atom_types : Tensor or None
        Optional type ID per atom to copy as int32.
    device : torch.device
        Target device for positions, optional cells, PBC flags, and types.

    Returns
    -------
    tuple
        ``(pointers, positions, cells, pbc, types)``. Pointers are a host list
        of cumulative atom offsets. Positions have shape ``[num_atoms, 3]``
        and dtype FP32; cells are optional FP32 tensors with shape
        ``[num_structures, 3, 3]``; PBC has shape ``[num_structures, 3]`` and
        dtype bool; types are optional int32 values with shape
        ``[num_atoms]``.

    Notes
    -----
    Tensor outputs are detached copies independent of the input batch. Missing
    positions produce an empty tensor; missing PBC becomes all-false. A single
    PBC triplet is expanded across structures before copying.
    """
    ptr = batch.batch_ptr.detach().to(device="cpu").tolist()
    pbc_source = (
        torch.zeros((batch.num_graphs, 3), dtype=torch.bool, device=batch.device)
        if "pbc" not in batch
        else batch.pbc.reshape(1, 3).expand(batch.num_graphs, 3)
        if batch.pbc.ndim == 1
        else batch.pbc
    )
    positions_source = (
        batch.positions
        if "positions" in batch
        else torch.empty((0, 3), dtype=torch.float32, device=batch.device)
    )
    positions = positions_source.detach().to(
        device=device, dtype=torch.float32, copy=True
    )
    cells = (
        batch.cell.detach().to(device=device, dtype=torch.float32, copy=True)
        if "cell" in batch
        else None
    )
    pbc = pbc_source.detach().to(device=device, dtype=torch.bool, copy=True)
    types = (
        atom_types.detach().to(device=device, dtype=torch.int32, copy=True)
        if atom_types is not None
        else None
    )
    return ptr, positions, cells, pbc, types


def build_descriptor_store(
    batch: Batch,
    *,
    cutoff: float,
    atom_types: Tensor | None,
    typed_neighbors: bool,
    has_atom_types: bool,
    device: torch.device,
    cuda_memory_budget_bytes: int | None,
    require_device_residency: bool = False,
) -> DescriptorStore:
    """Build one frozen descriptor layout through the shared multi-mode path.

    Parameters
    ----------
    batch : Batch
        Structures whose coordinates and optional cells define the geometry.
    cutoff : float
        Neighbor cutoff in the coordinate length unit.
    atom_types : Tensor or None
        Optional integer type IDs with shape ``[num_atoms]``.
    typed_neighbors : bool
        Whether to group neighbors by type when atom types are available.
    has_atom_types : bool
        Whether to construct a center-typed layout. When False, types are not
        included in the descriptor.
    device : torch.device
        Device used for geometry construction.
    cuda_memory_budget_bytes : int or None
        Configured CUDA construction and residency budget in bytes.
    require_device_residency : bool, default=False
        Whether a CUDA build must retain the packed store on the build device.

    Returns
    -------
    DescriptorStore
        The untyped, center-typed, or fully typed layout selected by the two
        type flags. Its tensor device follows :func:`build_descriptor_stores`.

    Raises
    ------
    MemoryError
        If estimated neighbor-list or typed-summary construction memory
        exceeds the CUDA budget.
    RuntimeError
        If neighbor capacity reaches its integer limit before a row fits.
    ValueError
        If the cutoff or geometry is invalid after conversion to FP32.

    Notes
    -----
    The single-layout request delegates to :func:`build_descriptor_stores` so
    it uses the same geometry and residency policy as multi-mode builds.
    """
    mode = "untyped" if not has_atom_types else "center"
    if has_atom_types and typed_neighbors:
        mode = "full"
    return build_descriptor_stores(
        batch,
        cutoff=cutoff,
        atom_types=atom_types if has_atom_types else None,
        modes=(mode,),
        device=device,
        cuda_memory_budget_bytes=cuda_memory_budget_bytes,
        require_device_residency=require_device_residency,
    )[mode]


def build_descriptor_stores(
    batch: Batch,
    *,
    cutoff: float,
    atom_types: Tensor | None,
    modes: Iterable[str],
    device: torch.device,
    cuda_memory_budget_bytes: int | None,
    require_device_residency: bool = False,
) -> dict[str, DescriptorStore]:
    """Build requested comparison layouts from one batched geometry pass.

    Parameters
    ----------
    batch : Batch
        Structures whose positions, cells, and PBC flags define the geometry.
    cutoff : float
        Neighbor cutoff in the same length unit as the coordinates.
    atom_types : Tensor or None
        Optional integer type IDs with shape ``[num_atoms]`` for typed modes.
    modes : iterable of str
        Requested layouts: ``"untyped"``, ``"center"``, or ``"full"``.
        Repeated names are collapsed while preserving first occurrence.
    device : torch.device
        Device used for the geometry pass and intermediate descriptor parts.
    cuda_memory_budget_bytes : int or None
        Configured limit for CUDA construction and residency estimates.
    require_device_residency : bool, default=False
        On CUDA, require the packed stores to remain resident on ``device``.
        When False, the stores may be assembled on CPU if the estimate does
        not fit the CUDA resident budget.

    Returns
    -------
    dict of str to DescriptorStore
        One store per unique requested mode, in request order. Untyped stores
        omit type labels; center stores retain center labels and distance-sorted
        neighbors; full stores group neighbors by type and include typed
        summaries.

    Raises
    ------
    MemoryError
        If estimated geometry or typed-summary workspace exceeds the configured
        CUDA construction budget, or a required CUDA-resident store exceeds the
        estimated resident budget.
    ValueError
        If a mode or cutoff is invalid, or positions or periodic cells are
        invalid after conversion to FP32.
    RuntimeError
        If neighbor capacity reaches its integer limit before a row fits.

    Notes
    -----
    Geometry tiles are shared across requested modes. Distances are stored as
    FP32 natural logarithms, and invalid neighbor slots use the FP32 log-cutoff
    sentinel. A missing cell disables periodic geometry. Memory values are
    estimates: on CUDA, required residency is checked against estimated
    available and configured budget; otherwise, stores remain on CUDA when
    the estimated peak fits, and are assembled on CPU when it does not. The
    input snapshot and intermediate parts remain owned by this call until the
    packed stores are complete.
    """
    requested_modes = tuple(dict.fromkeys(modes))
    supported_modes = {"untyped", "center", "full"}
    unsupported = set(requested_modes) - supported_modes
    if unsupported:
        raise ValueError(f"unsupported descriptor mode(s): {sorted(unsupported)}")
    if not requested_modes:
        return {}

    device = torch.device(device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    free_bytes_at_start = None
    if device.type == "cuda":
        free_bytes_at_start = available_cuda_bytes(device)
    ptr, positions, cells, pbc, types = _snapshot_batch(batch, atom_types, device)
    snapshot_bytes = sum(
        tensor.numel() * tensor.element_size()
        for tensor in (positions, cells, pbc, types)
        if tensor is not None
    )
    type_modes = {mode for mode in requested_modes if mode != "untyped"}
    geometry_types = types if type_modes else None
    type_vocab_device = (
        torch.unique(types, sorted=True)
        if "full" in requested_modes and types is not None
        else torch.empty(0, dtype=torch.int32, device=device)
    )
    typed_type_vocab = tuple(type_vocab_device.detach().to(device="cpu").tolist())
    typed_rank_indices = (0, 1, 4, 16, 64, 256, 1024)
    log_cutoff = _log_cutoff_fp32(cutoff)
    log_cutoff_device = torch.tensor(log_cutoff, dtype=torch.float32, device=device)
    cutoff_value_device = torch.tensor(cutoff, dtype=torch.float32, device=device)
    parts: dict[str, dict[str, list[Tensor] | list[int]]] = {
        mode: {
            "distances": [],
            "neighbors": [],
            "centers": [],
            "typed_summaries": [],
            "center_presence": [],
            "widths": [],
            "atom_counts": [],
        }
        for mode in requested_modes
    }

    resident_budget = 0
    if device.type == "cuda":
        resident_budget = min(
            free_bytes_at_start,
            cuda_memory_budget_bytes
            if cuda_memory_budget_bytes is not None
            else (free_bytes_at_start if require_device_residency else 0),
        )

    def estimated_store_bytes() -> int:
        """Estimate packed descriptor payload from accumulated tile metadata.

        Returns
        -------
        int
            Estimated bytes for packed values across the requested modes.

        Notes
        -----
        Per-structure widths and atom counts determine value slots, offsets,
        and summary arrays. Full-mode typed summaries also scale with the type
        vocabulary. This estimate excludes allocator peak behavior.
        """
        estimated = 0
        for mode in requested_modes:
            mode_parts = parts[mode]
            widths = mode_parts["widths"]
            atom_counts = mode_parts["atom_counts"]
            structure_count = len(widths)
            atom_count = sum(atom_counts)
            slot_count = sum(
                width * count for width, count in zip(widths, atom_counts, strict=True)
            )
            estimated += (
                slot_count * 8
                + atom_count * 4
                + 2 * (structure_count + 1) * 8
                + structure_count * 72
            )
            if mode == "full" and typed_type_vocab:
                type_count = len(typed_type_vocab)
                estimated += structure_count * (
                    type_count * type_count * len(typed_rank_indices) * 8 + type_count
                )
        return estimated

    def estimated_peak_build_bytes() -> int:
        """Estimate the modeled CUDA build peak for current tile metadata.

        Returns
        -------
        int
            Estimated bytes for the current build state.

        Notes
        -----
        The estimate combines the input snapshot, two packed-store equivalents
        for accumulated parts and concatenated copies, the largest geometry
        tile, and workspace headroom. It guides budget decisions only; it does
        not reserve memory or measure allocator peak usage.
        """
        return (
            snapshot_bytes
            + 2 * estimated_store_bytes()
            + max_tile_peak_bytes
            + workspace_reserve(resident_budget)
        )

    max_tile_peak_bytes = 0
    if len(ptr) > 1:
        tile_budget = cuda_memory_budget_bytes
        if device.type == "cuda" and require_device_residency:
            effective_budget = min(
                free_bytes_at_start,
                cuda_memory_budget_bytes
                if cuda_memory_budget_bytes is not None
                else free_bytes_at_start,
            )
            # Leave estimated snapshot and temporary headroom before sizing
            # geometry tiles against the remaining construction allowance.
            tile_budget = max(
                0,
                effective_budget - snapshot_bytes - workspace_reserve(effective_budget),
            )
            if tile_budget == 0 and len(ptr) > 1:
                raise MemoryError(
                    "CUDA-resident descriptor build has no budget remaining "
                    "for geometry construction"
                )
        tiles = build_descriptor_tiles(
            positions,
            ptr,
            cells,
            pbc,
            geometry_types,
            cutoff=cutoff,
            device=device,
            memory_budget_bytes=tile_budget,
        )
        for tile in tiles:
            max_tile_peak_bytes = max(max_tile_peak_bytes, tile.estimated_peak_bytes)
            for mode in requested_modes:
                has_center_types = mode != "untyped" and types is not None
                typed_neighbors = mode == "full" and has_center_types
                rows, neighbors = tile.layout(typed_neighbors=typed_neighbors)
                for local_graph, width in enumerate(tile.widths):
                    atom_start = tile.atom_offsets[local_graph]
                    atom_stop = tile.atom_offsets[local_graph + 1]
                    graph_rows = rows[atom_start:atom_stop, :width].contiguous()
                    if neighbors is None or not has_center_types:
                        graph_neighbors = torch.full(
                            graph_rows.shape,
                            _TYPE_PAD,
                            dtype=torch.int32,
                            device=device,
                        )
                    else:
                        graph_neighbors = neighbors[
                            atom_start:atom_stop, :width
                        ].contiguous()

                    log_rows = torch.log(graph_rows)
                    if has_center_types:
                        # Typed rows use the explicit type sentinel for padding;
                        # untyped rows have no sentinel buffer, so cutoff marks it.
                        active = graph_neighbors != _TYPE_PAD
                    else:
                        active = graph_rows < cutoff_value_device
                    log_rows = torch.where(active, log_rows, log_cutoff_device)
                    graph_centers = (
                        tile.center_types[atom_start:atom_stop]
                        if has_center_types and tile.center_types is not None
                        else torch.full(
                            (atom_stop - atom_start,),
                            _TYPE_PAD,
                            dtype=torch.int32,
                            device=device,
                        )
                    )

                    if typed_type_vocab and mode == "full":
                        summary_peak = _typed_summary_peak_bytes(
                            graph_rows.numel(),
                            atom_stop - atom_start,
                            len(typed_type_vocab),
                            len(typed_rank_indices),
                        )
                        if (
                            device.type == "cuda"
                            and cuda_memory_budget_bytes is not None
                            and summary_peak > cuda_memory_budget_bytes
                        ):
                            raise MemoryError(
                                f"typed summary for structure needs an estimated "
                                f"{summary_peak} CUDA bytes, above the configured "
                                f"{cuda_memory_budget_bytes}-byte build budget"
                            )
                        typed_summary, center_presence = (
                            _build_typed_summaries_for_structure(
                                log_rows,
                                graph_neighbors,
                                graph_centers,
                                type_vocab_device,
                                typed_rank_indices,
                                log_cutoff,
                            )
                        )
                        parts[mode]["typed_summaries"].append(typed_summary.detach())
                        parts[mode]["center_presence"].append(center_presence.detach())

                    parts[mode]["distances"].append(log_rows.detach().reshape(-1))
                    parts[mode]["neighbors"].append(
                        graph_neighbors.detach().reshape(-1)
                    )
                    parts[mode]["centers"].append(graph_centers.detach())
                    parts[mode]["widths"].append(width)
                    parts[mode]["atom_counts"].append(atom_stop - atom_start)
            if device.type == "cuda" and require_device_residency:
                required_bytes = estimated_peak_build_bytes()
                if required_bytes > resident_budget:
                    raise MemoryError(
                        f"CUDA-resident descriptor stores need an estimated "
                        f"{required_bytes} bytes, above the configured "
                        f"{resident_budget}-byte device budget"
                    )
            del tile

    store_device = torch.device("cpu")
    if device.type == "cuda":
        required_bytes = estimated_peak_build_bytes()
        if require_device_residency and required_bytes > resident_budget:
            raise MemoryError(
                f"CUDA-resident descriptor stores need an estimated "
                f"{required_bytes} bytes, above the configured "
                f"{resident_budget}-byte device budget"
            )
        if require_device_residency or required_bytes <= resident_budget:
            store_device = device

    values_by_mode: dict[str, tuple[Tensor, ...]] = {}
    for mode in requested_modes:
        mode_parts = parts[mode]
        widths = mode_parts["widths"]
        atom_counts = mode_parts["atom_counts"]
        row_offsets = [0]
        atom_offsets = [0]
        for width, count in zip(widths, atom_counts, strict=True):
            row_offsets.append(row_offsets[-1] + width * count)
            atom_offsets.append(atom_offsets[-1] + count)
        row_offsets_t = torch.tensor(
            row_offsets, dtype=torch.int64, device=store_device
        )
        atom_offsets_t = torch.tensor(
            atom_offsets, dtype=torch.int64, device=store_device
        )
        widths_t = torch.tensor(widths, dtype=torch.int32, device=store_device)
        counts_t = torch.tensor(atom_counts, dtype=torch.int32, device=store_device)
        distances = (
            torch.cat(
                [value.to(device=store_device) for value in mode_parts["distances"]]
            )
            if mode_parts["distances"]
            else torch.empty(0, dtype=torch.float32, device=store_device)
        )
        neighbor_types = (
            torch.cat(
                [value.to(device=store_device) for value in mode_parts["neighbors"]]
            )
            if mode_parts["neighbors"]
            else torch.empty(0, dtype=torch.int32, device=store_device)
        )
        center_types = (
            torch.cat(
                [value.to(device=store_device) for value in mode_parts["centers"]]
            )
            if mode_parts["centers"]
            else torch.empty(0, dtype=torch.int32, device=store_device)
        )
        summaries = _build_summaries(
            distances,
            neighbor_types,
            row_offsets,
            widths,
            atom_counts,
            log_cutoff,
            mode != "untyped" and types is not None,
            distance_sorted=mode != "full" or types is None,
        )
        mode_type_vocab = typed_type_vocab if mode == "full" else ()
        if mode_type_vocab:
            typed_summaries = (
                torch.stack(
                    [
                        value.to(device=store_device)
                        for value in mode_parts["typed_summaries"]
                    ]
                )
                if mode_parts["typed_summaries"]
                else torch.empty(
                    (
                        len(widths),
                        len(mode_type_vocab),
                        len(mode_type_vocab),
                        len(typed_rank_indices),
                        2,
                    ),
                    dtype=torch.float32,
                    device=store_device,
                )
            )
            center_type_presence = (
                torch.stack(
                    [
                        value.to(device=store_device)
                        for value in mode_parts["center_presence"]
                    ]
                )
                if mode_parts["center_presence"]
                else torch.empty(
                    (len(widths), len(mode_type_vocab)),
                    dtype=torch.bool,
                    device=store_device,
                )
            )
        else:
            typed_summaries = torch.empty(
                (len(widths), 0, 0, 2), dtype=torch.float32, device=store_device
            )
            center_type_presence = torch.empty(
                (len(widths), 0), dtype=torch.bool, device=store_device
            )
        values_by_mode[mode] = (
            distances,
            neighbor_types,
            center_types,
            row_offsets_t,
            atom_offsets_t,
            widths_t,
            counts_t,
            summaries,
            typed_summaries,
            center_type_presence,
        )

    stores: dict[str, DescriptorStore] = {}
    for mode, values in values_by_mode.items():
        stores[mode] = DescriptorStore(
            *values,
            typed_type_vocab=typed_type_vocab if mode == "full" else (),
            typed_rank_indices=(
                typed_rank_indices if mode == "full" and typed_type_vocab else ()
            ),
            device=store_device,
        )
    return stores


def _build_typed_summaries_for_structure(
    distances: Tensor,
    neighbor_types: Tensor,
    center_types: Tensor,
    type_vocab: Tensor,
    rank_indices: tuple[int, ...],
    log_cutoff: float,
) -> tuple[Tensor, Tensor]:
    """Build typed distance extrema for the requested per-type ranks.

    Parameters
    ----------
    distances : Tensor
        FP32 natural-log distances, grouped by neighbor type per center.
    neighbor_types : Tensor
        Aligned int32 labels with ``_TYPE_PAD`` in trailing padded slots.
    center_types : Tensor
        One int32 type label per center atom.
    type_vocab : Tensor
        Sorted vocabulary of type labels used to index summary axes.
    rank_indices : tuple of int
        Zero-based neighbor positions to summarize within each type group.
    log_cutoff : float
        Natural logarithm of the cutoff, used for missing ranks.

    Returns
    -------
    summaries : Tensor
        Minimum and maximum values for each center type, neighbor type, and
        requested rank, with shape ``[num_types, num_types, num_ranks, 2]``.
    group_presence : Tensor
        Boolean vector with shape ``[num_types]`` indicating which center
        types occur.

    Notes
    -----
    Empty center/type/rank groups receive ``log_cutoff`` for both extrema.
    """
    center_count = center_types.numel()
    type_count = type_vocab.numel()
    rank_count = len(rank_indices)
    # The batched scatter_reduce_ below requires int64 center indices.
    center_type_indices = torch.searchsorted(type_vocab, center_types)
    group_presence = torch.bincount(center_type_indices, minlength=type_count).gt(0)
    if distances.shape[1] == 0:
        minima = torch.full(
            (type_count, type_count, rank_count),
            log_cutoff,
            dtype=torch.float32,
            device=distances.device,
        )
        maxima = minima
    else:
        # Map labels to sorted vocabulary ranks so padding remains after every
        # valid label without widening this per-neighbor buffer to int64.
        neighbor_ranks = torch.searchsorted(type_vocab, neighbor_types, out_int32=True)
        sorted_ranks = torch.where(
            neighbor_types == _TYPE_PAD,
            type_count,
            neighbor_ranks,
        ).contiguous()
        queries = (
            torch.arange(type_count, dtype=torch.int32, device=distances.device)
            .view(1, -1)
            .expand(center_count, -1)
            .contiguous()
        )
        starts = torch.searchsorted(sorted_ranks, queries, right=False, out_int32=True)
        stops = torch.searchsorted(sorted_ranks, queries, right=True, out_int32=True)
        counts = stops - starts
        ranks = torch.tensor(rank_indices, dtype=torch.int32, device=distances.device)
        last_index = distances.shape[1] - 1
        safe_starts = starts.clamp(max=last_index)
        safe_ranks = torch.minimum(
            ranks.view(1, 1, -1),
            (last_index - safe_starts).unsqueeze(-1),
        )
        safe_indices = safe_starts.unsqueeze(-1) + safe_ranks
        values = distances.gather(1, safe_indices.reshape(center_count, -1)).reshape(
            center_count, type_count, rank_count
        )
        values = torch.where(
            ranks.view(1, 1, -1) < counts.unsqueeze(-1),
            values,
            torch.as_tensor(log_cutoff, dtype=torch.float32, device=distances.device),
        )
        index = center_type_indices.view(-1, 1, 1).expand_as(values)
        minima = torch.full(
            (type_count, type_count, rank_count),
            log_cutoff,
            dtype=torch.float32,
            device=distances.device,
        )
        maxima = torch.full(
            (type_count, type_count, rank_count),
            -float("inf"),
            dtype=torch.float32,
            device=distances.device,
        )
        minima.scatter_reduce_(0, index, values, reduce="amin", include_self=True)
        maxima.scatter_reduce_(0, index, values, reduce="amax", include_self=True)
        maxima = torch.where(group_presence.view(-1, 1, 1), maxima, log_cutoff)
    return torch.stack((minima, maxima), dim=-1), group_presence


def _build_summaries(
    distances: Tensor,
    neighbor_types: Tensor,
    row_offsets: list[int],
    widths: list[int],
    atom_counts: list[int],
    log_cutoff: float,
    has_atom_types: bool,
    feature_count: int = 8,
    *,
    distance_sorted: bool = False,
) -> Tensor:
    """Summarize nearest-neighbor order statistics for each structure.

    Parameters
    ----------
    distances : Tensor
        Flattened FP32 natural-log distances, grouped by structure and atom row.
    neighbor_types : Tensor
        Flattened int32 type IDs aligned with ``distances``; padded entries use
        ``_TYPE_PAD``.
    row_offsets : list of int
        Start offsets into the flattened distance/type arrays, one per
        structure plus a terminal offset.
    widths : list of int
        Padded neighbor width for each structure.
    atom_counts : list of int
        Number of center atoms in each structure.
    log_cutoff : float
        Natural logarithm of the cutoff, used for missing ranks.
    has_atom_types : bool
        If True, use the type padding sentinel to identify valid slots;
        otherwise, identify them by comparing log distances with
        ``log_cutoff``.
    feature_count : int, default=8
        Number of nearest-neighbor ranks retained per center.
    distance_sorted : bool, default=False
        Whether each row is already ordered by ascending distance. Typed rows
        may contain padding between valid neighbors, so their requested ranks
        are selected by cumulative valid counts. Untyped validity is a prefix
        because padding is identified by ``log_cutoff``.

    Returns
    -------
    Tensor
        FP32 array of shape ``[num_structures, 2 * feature_count]``. Each rank
        contributes its minimum and maximum across centers.

    Notes
    -----
    Missing neighbor ranks use ``log_cutoff``. These extrema are order
    statistics that are 1-Lipschitz under row mismatch and are used to filter
    candidate pairs before detailed scoring. For typed inputs, validity comes
    from the type sentinel; otherwise, a value equal to ``log_cutoff`` is
    treated as padding. The distance-sorted path avoids sorting rows already
    ordered by geometry while retaining typed validity when padding is
    interleaved with neighbors at the cutoff.
    """
    summaries: list[Tensor] = []
    cutoff_value = torch.tensor(
        log_cutoff, dtype=torch.float32, device=distances.device
    )
    for structure_id, (count, width) in enumerate(
        zip(atom_counts, widths, strict=True)
    ):
        row_start = row_offsets[structure_id]
        values = distances[row_start : row_start + count * width].reshape(count, width)
        neighbors = neighbor_types[row_start : row_start + count * width].reshape(
            count, width
        )
        if width:
            if has_atom_types:
                valid = neighbors != _TYPE_PAD
            else:
                valid = values < cutoff_value
            features = torch.full(
                (count, feature_count),
                log_cutoff,
                dtype=torch.float32,
                device=distances.device,
            )
            copied = min(width, feature_count)
            if distance_sorted:
                if has_atom_types and count:
                    # Cutoff-valued padding can precede rounded valid neighbors;
                    # select ranks by cumulative validity instead of slicing.
                    cumulative_valid = valid.cumsum(dim=1, dtype=torch.int32)
                    ranks = (
                        torch.arange(
                            1, copied + 1, dtype=torch.int32, device=values.device
                        )
                        .expand(count, -1)
                        .contiguous()
                    )
                    indices = torch.searchsorted(
                        cumulative_valid.contiguous(), ranks, out_int32=True
                    )
                    indices = indices.clamp_max(width - 1).to(torch.int64)
                    selected = values.gather(1, indices)
                    present = cumulative_valid[:, -1:] >= ranks
                    features[:, :copied] = torch.where(
                        present, selected, torch.full_like(selected, log_cutoff)
                    )
                else:
                    # Untyped rows have a valid-by-value prefix, so this mask
                    # preserves cutoff padding without a row sort.
                    features[:, :copied] = torch.where(
                        valid[:, :copied],
                        values[:, :copied],
                        torch.full_like(values[:, :copied], log_cutoff),
                    )
            else:
                ordered = torch.sort(
                    torch.where(valid, values, torch.full_like(values, float("inf"))),
                    dim=1,
                ).values
                features[:, :copied] = ordered[:, :copied]
            features = torch.where(torch.isfinite(features), features, cutoff_value)
        else:
            features = torch.full(
                (count, feature_count),
                log_cutoff,
                dtype=torch.float32,
                device=distances.device,
            )
        if count:
            summary = torch.stack((features.amin(dim=0), features.amax(dim=0)), dim=-1)
        else:
            summary = torch.full(
                (feature_count, 2),
                log_cutoff,
                dtype=torch.float32,
                device=distances.device,
            )
        summaries.append(summary.reshape(-1).contiguous())
    return (
        torch.stack(summaries)
        if summaries
        else torch.empty(
            (0, feature_count * 2), dtype=torch.float32, device=distances.device
        )
    )
