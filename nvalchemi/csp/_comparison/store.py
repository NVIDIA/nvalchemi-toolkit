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

from dataclasses import dataclass

import torch
from torch import Tensor

from nvalchemi.csp._comparison.descriptor import build_structure_descriptor
from nvalchemi.data import Batch

_TYPE_PAD = torch.iinfo(torch.int32).min


def workspace_reserve(memory_budget_bytes: int) -> int:
    """Return the byte allowance kept for descriptor and scoring temporaries."""
    return min(max(1024**2, memory_budget_bytes // 8), 1024**3)


def _typed_summary_peak_bytes(
    neighbor_slots: int, center_count: int, type_count: int, rank_count: int
) -> int:
    """Estimate per-structure GPU memory for typed-summary construction."""
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
        """Number of structure rows in the descriptor store."""
        return int(self.atom_counts.numel())

    @property
    def storage_bytes(self) -> int:
        """Bytes occupied by descriptor and summary tensors."""
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
        """Estimate staged descriptor bytes for the selected structures."""
        if not structure_ids:
            return 0
        ids = sorted(structure_ids)
        return sum(
            (int(self.atom_counts[i]) * int(self.widths[i]) * 8)
            + (int(self.atom_counts[i]) * 4)
            for i in ids
        )

    def block(self, structure_ids: set[int], device: torch.device) -> DescriptorBlock:
        """Return a resident view or stage selected structures to ``device``."""
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
    chunk_nodes: int = 131_072,
) -> tuple[list[int], Tensor, Tensor | None, Tensor, Tensor | None]:
    """Copy geometry through the target FP32 route, then own it on the host."""
    ptr = batch.batch_ptr.detach().to(device="cpu").tolist()
    position_parts: list[Tensor] = []
    cell_parts: list[Tensor] | None = [] if "cell" in batch else None
    pbc_source = (
        torch.zeros((batch.num_graphs, 3), dtype=torch.bool, device=batch.device)
        if "pbc" not in batch
        else batch.pbc.reshape(1, 3).expand(batch.num_graphs, 3)
        if batch.pbc.ndim == 1
        else batch.pbc
    )
    pbc_parts: list[Tensor] = []
    for graph_start in range(
        0, batch.num_graphs, max(1, chunk_nodes // max(batch.max_num_nodes, 1))
    ):
        graph_stop = min(
            batch.num_graphs,
            graph_start + max(1, chunk_nodes // max(batch.max_num_nodes, 1)),
        )
        node_start, node_stop = ptr[graph_start], ptr[graph_stop]
        positions = (
            batch.positions[node_start:node_stop]
            .detach()
            .to(device=device, dtype=torch.float32, copy=True)
        )
        position_parts.append(positions.to(device="cpu", copy=True))
        pbc_parts.append(
            pbc_source[graph_start:graph_stop]
            .detach()
            .to(device="cpu", dtype=torch.bool, copy=True)
        )
        if cell_parts is not None:
            cells = (
                batch.cell[graph_start:graph_stop]
                .detach()
                .to(device=device, dtype=torch.float32, copy=True)
            )
            cell_parts.append(cells.to(device="cpu", copy=True))
        del positions

    positions_cpu = (
        torch.cat(position_parts)
        if position_parts
        else torch.empty((0, 3), dtype=torch.float32)
    )
    pbc_cpu = (
        torch.cat(pbc_parts) if pbc_parts else torch.empty((0, 3), dtype=torch.bool)
    )
    cells_cpu = torch.cat(cell_parts) if cell_parts else None
    atom_types_cpu = (
        atom_types.detach().to(device="cpu", dtype=torch.int32, copy=True)
        if atom_types is not None
        else None
    )
    return ptr, positions_cpu, cells_cpu, pbc_cpu, atom_types_cpu


def _build_graph_descriptor_parts(
    positions: Tensor,
    cells: Tensor | None,
    pbc: Tensor,
    types: Tensor | None,
    *,
    start: int,
    stop: int,
    graph_id: int,
    cutoff: float,
    typed_neighbors: bool,
    device: torch.device,
    build_budget: int | None,
    typed_type_vocab: tuple[int, ...],
    type_vocab_device: Tensor,
    typed_rank_indices: tuple[int, ...],
) -> tuple[Tensor, Tensor, Tensor, int, int, Tensor | None, Tensor | None]:
    """Build one graph's descriptors and return only CPU-owned parts."""
    graph_positions = positions[start:stop].to(device=device, copy=True)
    graph_pbc = pbc[graph_id].to(device=device, copy=True)
    graph_cell = (
        cells[graph_id].to(device=device, copy=True)
        if cells is not None and bool(graph_pbc.any())
        else None
    )
    graph_types = (
        types[start:stop].to(device=device, copy=True) if types is not None else None
    )
    if not torch.isfinite(graph_positions).all():
        raise ValueError("positions must remain finite after FP32 conversion")
    if graph_cell is not None:
        if not torch.isfinite(graph_cell).all():
            raise ValueError("periodic cell must remain finite after FP32 conversion")
        try:
            inverse = torch.linalg.inv(graph_cell)
        except RuntimeError as exc:
            raise ValueError("periodic cell must be invertible in FP32") from exc
        if not torch.isfinite(inverse).all():
            raise ValueError("periodic cell must have a finite FP32 inverse")

    rows, neighbors = build_structure_descriptor(
        graph_positions,
        graph_cell,
        graph_pbc,
        cutoff,
        graph_types,
        typed_neighbors,
        memory_budget_bytes=build_budget,
    )
    log_rows = torch.log(rows)
    if neighbors is None:
        log_cutoff = torch.log(
            torch.as_tensor(cutoff, dtype=torch.float32, device=device)
        )
        active = log_rows < log_cutoff
        width = int(active.sum(dim=1).max()) if active.numel() else 0
        compact_rows = log_rows[:, :width]
        compact_neighbors = torch.full(
            compact_rows.shape, _TYPE_PAD, dtype=torch.int32, device=device
        )
    else:
        valid = neighbors != _TYPE_PAD
        width = int(valid.sum(dim=1).max()) if valid.numel() else 0
        if width:
            order = torch.argsort(
                valid.to(torch.int8), dim=1, descending=True, stable=True
            )
            order = order[:, :width].to(torch.int32)
            compact_rows = log_rows.gather(1, order)
            compact_neighbors = neighbors.gather(1, order)
        else:
            compact_rows = log_rows[:, :0]
            compact_neighbors = neighbors[:, :0]
    if typed_type_vocab and device.type == "cuda" and build_budget is not None:
        summary_peak = _typed_summary_peak_bytes(
            rows.numel(),
            stop - start,
            len(typed_type_vocab),
            len(typed_rank_indices),
        )
        if summary_peak > build_budget:
            raise MemoryError(
                f"typed summary for structure {graph_id} needs an estimated "
                f"{summary_peak} CUDA bytes, above the configured "
                f"{build_budget}-byte build budget"
            )

    compact_rows_cpu = compact_rows.detach().to(device="cpu", copy=True).reshape(-1)
    compact_neighbors_cpu = (
        compact_neighbors.detach().to(device="cpu", copy=True).reshape(-1)
    )
    center_types_cpu = (
        (
            graph_types
            if graph_types is not None
            else torch.full(
                (stop - start,), _TYPE_PAD, dtype=torch.int32, device=device
            )
        )
        .detach()
        .to(device="cpu", copy=True)
    )
    if typed_type_vocab:
        typed_summary, center_presence = _build_typed_summaries_for_structure(
            compact_rows,
            compact_neighbors,
            graph_types,
            type_vocab_device,
            typed_rank_indices,
            float(torch.log(torch.tensor(cutoff, dtype=torch.float32))),
        )
        typed_summary_cpu = typed_summary.detach().to(device="cpu", copy=True)
        center_presence_cpu = center_presence.detach().to(device="cpu", copy=True)
    else:
        typed_summary_cpu = None
        center_presence_cpu = None
    return (
        compact_rows_cpu,
        compact_neighbors_cpu,
        center_types_cpu,
        width,
        stop - start,
        typed_summary_cpu,
        center_presence_cpu,
    )


def build_descriptor_store(
    batch: Batch,
    *,
    cutoff: float,
    atom_types: Tensor | None,
    typed_neighbors: bool,
    has_atom_types: bool,
    device: torch.device,
    cuda_memory_budget_bytes: int | None,
) -> DescriptorStore:
    """Build frozen descriptors and keep the flat store on GPU when it fits.

    Raises
    ------
    MemoryError
        If estimated neighbor-list or typed-summary construction memory
        exceeds the CUDA budget.
    """
    ptr, positions, cells, pbc, types = _snapshot_batch(batch, atom_types, device)
    row_parts: list[Tensor] = []
    neighbor_parts: list[Tensor] = []
    center_parts: list[Tensor] = []
    typed_summary_parts: list[Tensor] = []
    center_presence_parts: list[Tensor] = []
    widths: list[int] = []
    atom_counts: list[int] = []

    typed_type_vocab = (
        tuple(sorted(set(types.tolist())))
        if typed_neighbors and types is not None
        else ()
    )
    typed_rank_indices = (0, 1, 4, 16, 64, 256, 1024)
    type_vocab_device = (
        torch.tensor(typed_type_vocab, dtype=torch.int32, device=device)
        if typed_type_vocab
        else torch.empty(0, dtype=torch.int32, device=device)
    )

    build_budget = cuda_memory_budget_bytes

    for graph_id, (start, stop) in enumerate(zip(ptr[:-1], ptr[1:], strict=True)):
        (
            rows_cpu,
            neighbors_cpu,
            centers_cpu,
            width,
            atom_count,
            typed_summary_cpu,
            center_presence_cpu,
        ) = _build_graph_descriptor_parts(
            positions,
            cells,
            pbc,
            types,
            start=start,
            stop=stop,
            graph_id=graph_id,
            cutoff=cutoff,
            typed_neighbors=typed_neighbors,
            device=device,
            build_budget=build_budget,
            typed_type_vocab=typed_type_vocab,
            type_vocab_device=type_vocab_device,
            typed_rank_indices=typed_rank_indices,
        )
        row_parts.append(rows_cpu)
        neighbor_parts.append(neighbors_cpu)
        center_parts.append(centers_cpu)
        if typed_summary_cpu is not None and center_presence_cpu is not None:
            typed_summary_parts.append(typed_summary_cpu)
            center_presence_parts.append(center_presence_cpu)
        widths.append(width)
        atom_counts.append(atom_count)

    row_offsets = [0]
    atom_offsets = [0]
    for width, count in zip(widths, atom_counts, strict=True):
        row_offsets.append(row_offsets[-1] + width * count)
        atom_offsets.append(atom_offsets[-1] + count)

    distances_cpu = (
        torch.cat(row_parts) if row_parts else torch.empty(0, dtype=torch.float32)
    )
    neighbor_types_cpu = (
        torch.cat(neighbor_parts)
        if neighbor_parts
        else torch.empty(0, dtype=torch.int32)
    )
    center_types_cpu = (
        torch.cat(center_parts) if center_parts else torch.empty(0, dtype=torch.int32)
    )
    if typed_type_vocab:
        typed_summaries_cpu = torch.stack(typed_summary_parts)
        center_type_presence_cpu = torch.stack(center_presence_parts)
    else:
        typed_summaries_cpu = torch.empty((len(widths), 0, 0, 2), dtype=torch.float32)
        center_type_presence_cpu = torch.empty((len(widths), 0), dtype=torch.bool)
    cpu_values = (
        distances_cpu,
        neighbor_types_cpu,
        center_types_cpu,
        torch.tensor(row_offsets, dtype=torch.int64),
        torch.tensor(atom_offsets, dtype=torch.int64),
        torch.tensor(widths, dtype=torch.int32),
        torch.tensor(atom_counts, dtype=torch.int32),
        _build_summaries(
            distances_cpu,
            neighbor_types_cpu,
            torch.tensor(row_offsets, dtype=torch.int64),
            torch.tensor(widths, dtype=torch.int32),
            torch.tensor(atom_counts, dtype=torch.int32),
            float(torch.log(torch.tensor(cutoff, dtype=torch.float32))),
            has_atom_types,
        ),
        typed_summaries_cpu,
        center_type_presence_cpu,
    )

    if device.type == "cuda":
        free_bytes, _ = torch.cuda.mem_get_info(device)
        resident_budget = min(free_bytes, cuda_memory_budget_bytes or 0)
        storage_bytes = sum(
            value.numel() * value.element_size() for value in cpu_values
        )
        if storage_bytes + workspace_reserve(resident_budget) <= resident_budget:
            values = tuple(value.to(device=device) for value in cpu_values)
        else:
            values = cpu_values
            device = torch.device("cpu")
    else:
        values = cpu_values
    return DescriptorStore(
        *values,
        typed_type_vocab=typed_type_vocab,
        typed_rank_indices=typed_rank_indices if typed_type_vocab else (),
        device=device,
    )


def _build_typed_summaries_for_structure(
    distances: Tensor,
    neighbor_types: Tensor,
    center_types: Tensor,
    type_vocab: Tensor,
    rank_indices: tuple[int, ...],
    log_cutoff: float,
) -> tuple[Tensor, Tensor]:
    """Build typed rank extrema by center type and neighbor type."""
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
    row_offsets: Tensor,
    widths: Tensor,
    atom_counts: Tensor,
    log_cutoff: float,
    has_atom_types: bool,
    feature_count: int = 8,
) -> Tensor:
    """Summarize order statistics that are 1-Lipschitz under row mismatch."""
    summaries: list[Tensor] = []
    cutoff_value = torch.tensor(log_cutoff, dtype=torch.float32)
    for structure_id in range(int(atom_counts.numel())):
        count = int(atom_counts[structure_id])
        width = int(widths[structure_id])
        row_start = int(row_offsets[structure_id])
        values = distances[row_start : row_start + count * width].reshape(count, width)
        neighbors = neighbor_types[row_start : row_start + count * width].reshape(
            count, width
        )
        if width:
            if has_atom_types:
                valid = neighbors != _TYPE_PAD
            else:
                valid = values < cutoff_value
            ordered = torch.sort(
                torch.where(valid, values, torch.full_like(values, float("inf"))),
                dim=1,
            ).values
            features = torch.full(
                (count, feature_count), log_cutoff, dtype=torch.float32
            )
            copied = min(width, feature_count)
            features[:, :copied] = ordered[:, :copied]
            features = torch.where(torch.isfinite(features), features, cutoff_value)
        else:
            features = torch.full(
                (count, feature_count), log_cutoff, dtype=torch.float32
            )
        summaries.append(
            torch.stack((features.amin(dim=0), features.amax(dim=0)), dim=-1)
            .reshape(-1)
            .contiguous()
        )
    return (
        torch.stack(summaries)
        if summaries
        else torch.empty((0, feature_count * 2), dtype=torch.float32)
    )
