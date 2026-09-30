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
"""Approximate atomistic structure comparison and deterministic deduplication."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass

import torch
from torch import Tensor

from nvalchemi.csp._comparison.score import (
    score_descriptor_pairs,
    threshold_score,
    threshold_score_descriptor_pairs,
)
from nvalchemi.csp._comparison.store import (
    DescriptorStore,
    available_cuda_bytes,
    build_descriptor_store,
    workspace_reserve,
)
from nvalchemi.data import Batch

__all__ = [
    "RadialComparisonIndex",
    "DeduplicationResult",
    "deduplicate_batch",
    "deduplicate_stream",
    "iter_matches_stream",
]

_CUDA_TYPED_FILTER_MAX_TILE = 1024
_CUDA_TYPED_FILTER_OUTPUT_RESERVE_BYTES = 320 * 1024


@dataclass(frozen=True)
class DeduplicationResult:
    """Retained structures, each input’s retained representative, and group
    sizes.

    All fields are int32 tensors on the comparison device. ``retained_indices``
    has shape ``[R]`` and gives original-pool indices in retention order.
    ``representative_indices`` has shape ``[N]`` and gives each input's
    original-pool representative, not an offset into ``retained_indices``.
    ``multiplicities`` has shape ``[R]`` and counts inputs assigned to each
    retained structure, including the representative itself.
    """

    retained_indices: Tensor
    representative_indices: Tensor
    multiplicities: Tensor


def _validate_threshold(value: float) -> float:
    """Validate one public fractional score threshold.

    Parameters
    ----------
    value : float
        Inclusive, nonnegative mismatch limit supplied by the caller.

    Returns
    -------
    float
        The threshold normalized to a Python float for score-bound creation.

    Raises
    ------
    ValueError
        If the value is boolean, nonnumeric, nonfinite, or negative.
    """
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
    ):
        raise ValueError("threshold must be a finite nonnegative fraction")
    return float(value)


def _conservative_summary_log_bound(threshold: float) -> float | None:
    """Convert an inclusive score limit to a conservative log-summary bound.

    The public endpoint is rounded outward in FP32, then widened by a small
    additional ULP allowance before ``log1p`` is rounded upward. Summary
    filtering therefore rejects only pairs proven outside the inclusive score
    limit; extreme thresholds return ``None`` and disable that filter.

    Parameters
    ----------
    threshold : float
        Validated fractional mismatch threshold.

    Returns
    -------
    float or None
        Upward-rounded log-distance bound, or ``None`` when it is not finite.
    """
    score = torch.tensor(threshold, dtype=torch.float32)
    infinity = torch.tensor(float("inf"), dtype=torch.float32)
    score = torch.nextafter(score, infinity)
    endpoint = float(score)
    if not math.isfinite(endpoint) or endpoint >= 1.0e30:
        return None
    for _ in range(32):
        score = torch.nextafter(score, infinity)
    bound = torch.tensor(math.log1p(float(score)), dtype=torch.float32)
    for _ in range(4):
        bound = torch.nextafter(bound, infinity)
    return float(bound)


def _validate_confirmed_subset(
    radial: Tensor, confirmed: Tensor, comparison_device: torch.device
) -> None:
    """Validate a callback result as an ordered subset of radial proposals.

    Parameters
    ----------
    radial : torch.Tensor, shape (K, 2), dtype int32
        Immutable snapshot of logical ``(candidate, retained representative)``
        proposals in priority order.
    confirmed : torch.Tensor, shape (M, 2), dtype int32
        Callback result containing only proposal rows, in their original order.
    comparison_device : torch.device
        Device required for ``confirmed``.

    Raises
    ------
    ValueError
        If dtype, shape, device, or ordered-subset membership is invalid.
    """
    if (
        not isinstance(confirmed, Tensor)
        or confirmed.dtype != torch.int32
        or confirmed.ndim != 2
        or confirmed.shape[1] != 2
        or confirmed.shape[0] > radial.shape[0]
    ):
        raise ValueError("confirm must return an int32 [K, 2] subset of its input")
    if confirmed.device != comparison_device:
        raise ValueError("confirm result must be on the comparison device")
    source = [tuple(pair) for pair in radial.tolist()]
    cursor = 0
    for pair in confirmed.tolist():
        try:
            cursor = source.index(tuple(pair), cursor) + 1
        except ValueError as exc:
            raise ValueError(
                "confirm must return an ordered subset of its input"
            ) from exc


class RadialComparisonIndex:
    """Compare structures by local radial distributions of atom-neighbor distances.

    ``cutoff`` and coordinates are in angstroms. For two positive distances,
    the fractional mismatch is ``max(d1, d2) / min(d1, d2) - 1``. The score is
    the largest mismatch after matching compatible atom environments in both
    directions. Thus ``threshold=0.1`` allows the larger distance to be at
    most 10% above the smaller one. This radial comparison is approximate: a
    low score can be a false positive and does not establish structural
    equivalence.

    Without ``atom_types``, all atoms share one comparison type; atomic
    numbers are not inferred. Supplied types always constrain center matching;
    ``typed_neighbors`` controls whether neighbor distances are separated by
    type. Missing cell or PBC metadata means nonperiodic geometry; per-axis PBC
    flags support partially periodic structures. If a center type is absent
    from the other structure, the pair's score is infinite. Cross-index
    comparisons require equal cutoff, device, and typing modes and a shared
    semantic mapping of type IDs.

    For matching, a finite threshold is converted to FP32 and advanced one
    representable value toward positive infinity.
    """

    def __init__(
        self,
        descriptor_store: DescriptorStore,
        cutoff: float,
        atom_types: Tensor | None,
        typed_neighbors: bool,
        device: torch.device,
        block_size: int,
        max_memory_fraction: float,
        cuda_memory_budget_bytes: int | None,
        summary: Tensor,
        typed_summary: Tensor,
        center_type_presence: Tensor,
    ) -> None:
        """Bind a descriptor pool and its comparison-time accounting state.

        Parameters
        ----------
        descriptor_store : DescriptorStore
            Frozen per-structure descriptors, row IDs, and summary backing.
        cutoff : float
            Radial cutoff in angstroms used to build the descriptors.
        atom_types : torch.Tensor or None
            Shared int32 type IDs aligned with descriptor atoms, if supplied.
        typed_neighbors : bool
            Whether neighbor ranks are separated by type when types exist.
        device : torch.device
            Device used by the compact summaries and scoring operations.
        block_size : int
            Target maximum number of structures staged for a pair block.
        max_memory_fraction : float
            Configured fraction used for the index's estimated CUDA allowance.
        cuda_memory_budget_bytes : int or None
            Call-local CUDA byte allowance, or ``None`` for CPU execution.
        summary, typed_summary : torch.Tensor
            Compact per-structure extrema on the comparison device or host.
        center_type_presence : torch.Tensor
            Boolean per-structure center-type presence matrix.

        Notes
        -----
        Resident descriptor bytes are tracked separately from reusable filter
        workspace so cross-pool queries can account for all live indexes.
        """
        self._store = descriptor_store
        self.cutoff = cutoff
        self._typed_neighbors = typed_neighbors and atom_types is not None
        self._has_center_types = atom_types is not None
        self.device = device
        self.structure_block_size = block_size
        self.max_memory_fraction = max_memory_fraction
        self._cuda_memory_budget_bytes = cuda_memory_budget_bytes
        self._summary = summary
        self._typed_summary = typed_summary
        self._center_type_presence = center_type_presence
        self._typed_type_vocab = descriptor_store.typed_type_vocab
        self._typed_rank_indices = descriptor_store.typed_rank_indices
        self._resident_descriptor_bytes = (
            descriptor_store.storage_bytes
            if descriptor_store.device.type == "cuda"
            else sum(
                value.numel() * value.element_size()
                for value in (summary, typed_summary, center_type_presence)
                if value.device.type == "cuda"
            )
        )
        self._active_resident_descriptor_bytes = self._resident_descriptor_bytes
        self._typed_filter_tile_size: int | None = None
        self._typed_filter_workspace_bytes = 0
        self._typed_filter_bytes_per_pair = 0
        self._configure_typed_filter_workspace()
        self._num_structures = descriptor_store.num_structures
        self._last_search_stats = {
            "pairs_considered": 0,
            "pairs_rejected": 0,
            "pairs_scored": 0,
            "pairs_certified_matches": 0,
        }
        self._last_dedup_stats = {
            "pairs_considered": 0,
            "pairs_rejected": 0,
            "pairs_scored": 0,
            "pairs_certified_matches": 0,
        }

    def _configure_typed_filter_workspace(
        self, active_resident_bytes: int | None = None
    ) -> None:
        """Size typed-summary gathers within the remaining CUDA allowance.

        Parameters
        ----------
        active_resident_bytes : int, optional
            Bytes occupied by every descriptor bundle held in the current tile.
            Defaults to this index's current resident descriptor count.

        Notes
        -----
        The pair cap reserves output space and obeys a fixed launch limit after
        accounting for all resident descriptors.
        """
        self._typed_filter_tile_size = None
        self._typed_filter_workspace_bytes = 0
        self._typed_filter_bytes_per_pair = 0
        if self.device.type == "cuda" and self._typed_neighbors:
            if self._typed_summary.shape[0]:
                summary_elements = self._typed_summary[0].numel()
                summary_bytes = summary_elements * self._typed_summary.element_size()
                vocab_size = self._center_type_presence.shape[-1]
                # Two gathered FP32 summaries and one boolean separation tensor
                # are live at once; subtract/abs operate in place, so no extra
                # FP32 difference arrays are needed. Four boolean presence masks
                # are live per pair (two gathers, shared types, and type mismatch).
                # The fixed reserve bounds IDs, padded pairs, and selected outputs.
                self._typed_filter_bytes_per_pair = (
                    2 * summary_bytes + summary_elements + 4 * vocab_size + 64
                )
            available_budget = max(
                0,
                (self._cuda_memory_budget_bytes or 0)
                - (
                    self._active_resident_descriptor_bytes
                    if active_resident_bytes is None
                    else active_resident_bytes
                ),
            )
            self._typed_filter_workspace_bytes = min(
                available_budget, workspace_reserve(available_budget)
            )
            usable_workspace = (
                self._typed_filter_workspace_bytes
                - _CUDA_TYPED_FILTER_OUTPUT_RESERVE_BYTES
            )
            if self._typed_filter_bytes_per_pair > 0 and usable_workspace > 0:
                self._typed_filter_tile_size = min(
                    _CUDA_TYPED_FILTER_MAX_TILE,
                    usable_workspace // self._typed_filter_bytes_per_pair,
                )
            else:
                self._typed_filter_tile_size = 0

    def _set_active_memory_allowance(
        self, budget_bytes: int | None, active_resident_bytes: int
    ) -> None:
        """Recompute workspace for indexes live in one stream tile.

        Parameters
        ----------
        budget_bytes : int or None
            Common estimated CUDA allowance for the active stream call.
        active_resident_bytes : int
            Combined descriptor bytes held by modes and endpoints in the tile.

        Notes
        -----
        Typed filtering cannot claim capacity already occupied by another live
        mode or endpoint.
        """
        self._cuda_memory_budget_bytes = budget_bytes
        self._active_resident_descriptor_bytes = active_resident_bytes
        self._configure_typed_filter_workspace(active_resident_bytes)

    @classmethod
    def _from_store(
        cls,
        descriptor_store: DescriptorStore,
        *,
        cutoff: float,
        atom_types: Tensor | None,
        typed_neighbors: bool,
        device: torch.device,
        max_memory_fraction: float,
        cuda_memory_budget_bytes: int | None,
    ) -> RadialComparisonIndex:
        """Create an index around an already-built descriptor store.

        Parameters
        ----------
        descriptor_store : DescriptorStore
            Store containing the mode's descriptor layout and summaries.
        cutoff : float
            Radial cutoff in angstroms used for the shared neighbor geometry.
        atom_types : torch.Tensor or None
            Pool-aligned atom type IDs for this comparison mode.
        typed_neighbors : bool
            Whether neighbor ranks are separated by type.
        device : torch.device
            Comparison and summary device.
        max_memory_fraction : float
            Configured fraction used for estimated CUDA workspace.
        cuda_memory_budget_bytes : int or None
            Shared CUDA allowance in bytes, or ``None`` on CPU.

        Returns
        -------
        RadialComparisonIndex
            Index bound to the supplied mode layout.

        Notes
        -----
        Stream modes use separate layouts built from shared neighbor geometry;
        the store's logical row IDs and compact summaries remain aligned.
        """
        return cls(
            descriptor_store,
            cutoff,
            atom_types,
            typed_neighbors,
            device,
            max(1, descriptor_store.num_structures),
            max_memory_fraction,
            cuda_memory_budget_bytes,
            descriptor_store.summaries,
            descriptor_store.typed_summaries,
            descriptor_store.center_type_presence,
        )

    @classmethod
    def build(
        cls,
        batch: Batch,
        *,
        cutoff: float,
        atom_types: Tensor | None = None,
        typed_neighbors: bool = True,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str | None = None,
        max_memory_fraction: float = 0.70,
        structure_block_size: int | None = None,
    ) -> RadialComparisonIndex:
        """Build a reusable index of local atom-neighbor distances for each
        structure.

        Parameters
        ----------
        batch : Batch
            Source structures. Geometry and optional types are copied during
            construction; later source-batch mutations do not affect results.
        cutoff : float
            Maximum atom-neighbor distance included, in angstroms.
        atom_types : torch.Tensor, optional
            Caller-supplied integer type for each atom, using one shared ID
            mapping across structures. Values must fit signed int32; the int32
            minimum is reserved for descriptor padding.
        typed_neighbors : bool, default=True
            When types are supplied, compare neighbor distances separately by type.
        dtype : torch.dtype, default=torch.float32
            Descriptor and score dtype. Only FP32 is supported.
        device : torch.device or str, optional
            Device used for descriptor construction and scoring. Defaults to
            the batch device.
        max_memory_fraction : float, default=0.70
            Fraction in (0, 0.70] of CUDA memory available to PyTorch at build
            time used as an estimated descriptor-construction and owned-storage
            budget. The estimate includes reusable caching-allocator bytes.
            Host-backed descriptor staging and typed-summary tiles use the
            remaining allowance; caller inputs and returned results are
            excluded. Later staging also depends on current free memory.
            This is an estimate, not a memory reservation: backend workspace
            and allocator behavior can make actual peak use differ, and an
            allocation can still fail.
        structure_block_size : int, optional
            Target limit on distinct structures staged for a descriptor block.
            One pair may require more; resident descriptor blocks are not sliced.

        Returns
        -------
        RadialComparisonIndex
            Eagerly built index for scoring and matching the frozen structures
            copied from ``batch``.

        Raises
        ------
        TypeError
            If the batch, dtype, type IDs, or option types are unsupported.
        ValueError
            If geometry or numerical options are invalid.
        MemoryError
            If estimated CUDA descriptor or typed-summary workspace exceeds
            the configured memory budget.
        """
        if not isinstance(batch, Batch):
            raise TypeError("batch must be a nvalchemi.data.Batch")
        if dtype is not torch.float32:
            raise TypeError("structure comparison supports dtype=torch.float32 only")
        if (
            isinstance(cutoff, bool)
            or not isinstance(cutoff, (int, float))
            or not math.isfinite(cutoff)
            or cutoff <= 0
        ):
            raise ValueError("cutoff must be finite and positive")
        cutoff_fp32 = torch.tensor(cutoff, dtype=torch.float32)
        if not torch.isfinite(cutoff_fp32) or cutoff_fp32 <= 0:
            raise ValueError("cutoff must be representable as positive finite FP32")
        if not (0.0 < max_memory_fraction <= 0.70):
            raise ValueError("max_memory_fraction must be in (0, 0.70]")
        if not isinstance(typed_neighbors, bool):
            raise TypeError("typed_neighbors must be bool")
        if batch.num_graphs:
            if (
                "positions" not in batch
                or batch.positions.ndim != 2
                or batch.positions.shape[1] != 3
            ):
                raise ValueError(
                    "batch must contain positions with shape [num_nodes, 3]"
                )
            if not batch.positions.dtype.is_floating_point:
                raise TypeError("batch.positions must have a floating dtype")
            if not torch.isfinite(batch.positions).all():
                raise ValueError("batch.positions must be finite")
        if batch.num_graphs and torch.any(batch.batch_ptr[1:] <= batch.batch_ptr[:-1]):
            raise ValueError("comparison structures must contain atoms")
        if "cell" in batch:
            if (
                batch.cell.shape != (batch.num_graphs, 3, 3)
                or not batch.cell.dtype.is_floating_point
            ):
                raise ValueError(
                    "batch.cell must have shape [num_graphs, 3, 3] and a floating dtype"
                )
            if not torch.isfinite(batch.cell).all():
                raise ValueError("batch.cell must contain finite values")
        if "pbc" in batch:
            pbc = batch.pbc
            if (
                pbc.shape not in ((3,), (batch.num_graphs, 3))
                or pbc.dtype != torch.bool
            ):
                raise ValueError(
                    "batch.pbc must be bool with shape [3] or [num_graphs, 3]"
                )
            pbc_by_graph = (
                pbc.reshape(1, 3).expand(batch.num_graphs, 3) if pbc.ndim == 1 else pbc
            )
            if "cell" in batch:
                periodic = pbc_by_graph.any(dim=1)
                if periodic.any() and torch.any(
                    torch.linalg.det(batch.cell[periodic]).abs() <= 1e-10
                ):
                    raise ValueError("periodic structures require nonsingular cells")
        if atom_types is not None:
            if (
                not isinstance(atom_types, Tensor)
                or atom_types.ndim != 1
                or atom_types.numel() != batch.num_nodes
            ):
                raise ValueError(
                    "atom_types must be a one-dimensional tensor with one entry per atom"
                )
            if atom_types.dtype not in (torch.int32, torch.int64):
                raise TypeError("atom_types must have dtype torch.int32 or torch.int64")
            int32_info = torch.iinfo(torch.int32)
            if atom_types.dtype is torch.int64 and torch.any(
                (atom_types < int32_info.min) | (atom_types > int32_info.max)
            ):
                raise ValueError("atom_types values must fit signed int32")
            if torch.any(atom_types == int32_info.min):
                raise ValueError("atom_types cannot use the reserved int32 minimum")
            atom_types = atom_types.detach().to(dtype=torch.int32).contiguous()
        target_device = torch.device(device) if device is not None else batch.device
        if target_device.type not in ("cpu", "cuda"):
            raise ValueError("device must be a CPU or CUDA device")
        if target_device.type == "cuda" and target_device.index is None:
            target_device = torch.device("cuda", torch.cuda.current_device())
        limit = 256 if target_device.type == "cuda" else 64
        default_block = limit
        if structure_block_size is None:
            block_size = default_block
        else:
            if (
                isinstance(structure_block_size, bool)
                or not isinstance(structure_block_size, int)
                or structure_block_size <= 0
            ):
                raise ValueError("structure_block_size must be a positive integer")
            block_size = min(structure_block_size, limit)
        cuda_memory_budget_bytes = None
        if target_device.type == "cuda":
            available_bytes = available_cuda_bytes(target_device)
            cuda_memory_budget_bytes = int(available_bytes * max_memory_fraction)
        descriptor_store = build_descriptor_store(
            batch,
            cutoff=float(cutoff),
            atom_types=atom_types,
            typed_neighbors=typed_neighbors and atom_types is not None,
            has_atom_types=atom_types is not None,
            device=target_device,
            cuda_memory_budget_bytes=cuda_memory_budget_bytes,
        )
        summary = descriptor_store.summaries
        typed_summary = descriptor_store.typed_summaries
        center_type_presence = descriptor_store.center_type_presence
        if target_device.type == "cuda" and descriptor_store.device.type == "cpu":
            available_bytes = available_cuda_bytes(target_device)
            summary_bytes = sum(
                value.numel() * value.element_size()
                for value in (summary, typed_summary, center_type_presence)
            )
            budget = min(available_bytes, cuda_memory_budget_bytes or 0)
            if summary_bytes + workspace_reserve(budget) <= budget:
                summary = summary.to(device=target_device)
                typed_summary = typed_summary.to(device=target_device)
                center_type_presence = center_type_presence.to(device=target_device)
        return cls(
            descriptor_store,
            float(cutoff),
            atom_types,
            typed_neighbors,
            target_device,
            block_size,
            max_memory_fraction,
            cuda_memory_budget_bytes,
            summary,
            typed_summary,
            center_type_presence,
        )

    @property
    def num_structures(self) -> int:
        """Return the number of frozen structure rows represented by the index.

        Returns
        -------
        int
            Number of logical structures in the source pool.

        Notes
        -----
        The value is independent of how many descriptor blocks are resident
        during a comparison.
        """
        return self._num_structures

    def _device_memory_budget(self) -> int | None:
        """Return current CUDA workspace after reserving resident descriptors.

        Returns
        -------
        int or None
            Remaining estimated CUDA bytes, or ``None`` for CPU indexes.

        Notes
        -----
        The configured fraction is capped by memory currently available to
        PyTorch, including reusable allocator bytes. Resident descriptor storage
        is subtracted before scoring or typed filtering selects temporary tiles.
        """
        if self.device.type != "cuda":
            return None
        available_bytes = available_cuda_bytes(self.device)
        budget = self._cuda_memory_budget_bytes or 0
        return min(
            available_bytes,
            max(0, budget - self._active_resident_descriptor_bytes),
        )

    def _check_other(self, other: RadialComparisonIndex) -> None:
        """Check cross-pool compatibility before sharing a scoring kernel.

        Parameters
        ----------
        other : RadialComparisonIndex
            Right-hand index for a cross-pool query.

        Raises
        ------
        TypeError
            If ``other`` is not a comparison index.
        ValueError
            If cutoff, device, or typing modes differ.

        Notes
        -----
        Pair IDs keep separate left and right pool meanings. Callers must also
        use the same semantic mapping for type IDs.
        """
        if not isinstance(other, RadialComparisonIndex):
            raise TypeError("other must be a RadialComparisonIndex")
        if (
            self.cutoff,
            self.device,
            self._typed_neighbors,
            self._has_center_types,
        ) != (
            other.cutoff,
            other.device,
            other._typed_neighbors,
            other._has_center_types,
        ):
            raise ValueError(
                "comparison indexes must have matching cutoff, device, and typing modes"
            )

    def _validate_pairs(self, pair_indices: Tensor, right_count: int) -> Tensor:
        """Validate ordered logical pair IDs and make a device-local int64 view.

        Parameters
        ----------
        pair_indices : torch.Tensor, shape (K, 2)
            Ordered left and right logical row IDs, as int32 or int64.
        right_count : int
            Number of rows in the right-hand pool.

        Returns
        -------
        torch.Tensor, shape (K, 2), dtype int64
            Validated pair IDs on this index's device, preserving input order.

        Raises
        ------
        ValueError
            If the input does not have shape ``(K, 2)``.
        TypeError
            If the input dtype is not int32 or int64.
        IndexError
            If either column contains a row ID outside its pool.
        """
        if (
            not isinstance(pair_indices, Tensor)
            or pair_indices.ndim != 2
            or pair_indices.shape[1] != 2
        ):
            raise ValueError("pair_indices must have shape [K, 2]")
        if pair_indices.dtype not in (torch.int32, torch.int64):
            raise TypeError("pair_indices must have dtype int32 or int64")
        pairs = pair_indices.to(device=self.device, dtype=torch.int64)
        if pairs.numel() and (
            torch.any(pairs[:, 0] < 0)
            or torch.any(pairs[:, 0] >= self.num_structures)
            or torch.any(pairs[:, 1] < 0)
            or torch.any(pairs[:, 1] >= right_count)
        ):
            raise IndexError("pair_indices contain an out-of-range structure index")
        return pairs

    def _descriptor_pair_blocks(
        self,
        pairs: list[tuple[int, int]],
        right: RadialComparisonIndex,
        max_pairs: int,
        memory_budget_bytes: int | None,
        threshold_log_bound: float | None = None,
        threshold_accept_log_bound: float | None = None,
    ) -> Iterator[tuple[list[tuple[int, int]], Tensor, Tensor | None]]:
        """Stage and score ordered pairs within pair and memory limits.

        Parameters
        ----------
        pairs : list of tuple of int
            Logical ``(left_id, right_id)`` pairs in caller order.
        right : RadialComparisonIndex
            Index supplying the right-hand descriptor rows.
        max_pairs : int
            Maximum number of logical pairs in a scored block.
        memory_budget_bytes : int or None
            Estimated temporary CUDA allowance for staging and scoring.
        threshold_log_bound, threshold_accept_log_bound : float or None
            Conservative log-distance bounds enabling early score outcomes.

        Yields
        ------
        tuple
            Pair rows, FP32 scores, and optional threshold outcomes for one
            block. Pair order matches the corresponding input subsequence.

        Raises
        ------
        MemoryError
            If even one descriptor pair exceeds the staging allowance.

        Notes
        -----
        Host-backed stores are staged by unique structure ID. Self-comparison
        reuses one endpoint block rather than staging it twice.
        """
        block_limit = min(max_pairs, 8192)
        staging = self.device.type == "cuda" and (
            self._store.device.type == "cpu" or right._store.device.type == "cpu"
        )
        structure_limit = min(self.structure_block_size, right.structure_block_size)
        reserve = (
            0 if memory_budget_bytes is None else workspace_reserve(memory_budget_bytes)
        )
        block: list[tuple[int, int]] = []
        left_ids: set[int] = set()
        right_ids: set[int] = set()
        combined_ids: set[int] = set()
        staged_bytes_current = 0
        left_on_host = self._store.device.type == "cpu"
        right_on_host = right._store.device.type == "cpu"

        def next_staging_state(
            left_id: int, right_id: int
        ) -> tuple[int, int, int, list[int], list[int]]:
            """Estimate unique host rows and bytes if a pair joins this block.

            Parameters
            ----------
            left_id, right_id : int
                Logical endpoint IDs for the next pair.

            Returns
            -------
            tuple
                Updated unique left/right counts, staged bytes, and newly
                required endpoint ID lists.

            Notes
            -----
            Returned ID lists let the outer loop test caps without loading
            descriptors prematurely.
            """
            if self is right:
                additions = list(dict.fromkeys((left_id, right_id)))
                new_combined = [
                    value for value in additions if value not in combined_ids
                ]
                right_additions = [
                    value
                    for value in dict.fromkeys((right_id,))
                    if value not in right_ids
                ]
                next_bytes = staged_bytes_current + sum(
                    self._store.bytes_for({value}) for value in new_combined
                )
                return (
                    len(combined_ids) + len(new_combined),
                    len(right_ids) + len(right_additions),
                    next_bytes,
                    new_combined,
                    right_additions,
                )
            new_left = [left_id] if left_id not in left_ids else []
            new_right = [right_id] if right_id not in right_ids else []
            next_bytes = staged_bytes_current
            if left_on_host:
                next_bytes += sum(self._store.bytes_for({value}) for value in new_left)
            if right_on_host:
                next_bytes += sum(
                    right._store.bytes_for({value}) for value in new_right
                )
            return (
                len(left_ids) + len(new_left),
                len(right_ids) + len(new_right),
                next_bytes,
                new_left,
                new_right,
            )

        def emit(
            current: list[tuple[int, int]],
        ) -> tuple[list[tuple[int, int]], Tensor, Tensor | None]:
            """Materialize one staging block and score it in pair order.

            Parameters
            ----------
            current : list of tuple of int
                Logical endpoint IDs for this block, in input order.

            Returns
            -------
            tuple
                The logical pairs, FP32 scores, and optional threshold outcomes.

            Notes
            -----
            Self-comparison reuses one staged block for both endpoints. The
            threshold path can certify accepts or rejects without full scoring.
            """
            current_left = {i for i, _ in current}
            current_right = {j for _, j in current}
            if self is right:
                current_left |= current_right
            left_block = self._store.block(current_left, self.device)
            right_block = (
                left_block
                if self is right
                else right._store.block(current_right, self.device)
            )
            left_map = (
                None
                if left_block.source_ids is None
                else {value: index for index, value in enumerate(left_block.source_ids)}
            )
            right_map = (
                None
                if right_block.source_ids is None
                else {
                    value: index for index, value in enumerate(right_block.source_ids)
                }
            )
            mapped = torch.tensor(
                [
                    (
                        i if left_map is None else left_map[i],
                        j if right_map is None else right_map[j],
                    )
                    for i, j in current
                ],
                dtype=torch.int32,
                device=self.device,
            )
            if threshold_log_bound is None or threshold_accept_log_bound is None:
                scores, _ = score_descriptor_pairs(
                    left_block,
                    right_block,
                    mapped,
                    self.cutoff,
                    self._has_center_types,
                    self._typed_neighbors or right._typed_neighbors,
                )
                return current, scores, None
            scores, outcomes = threshold_score_descriptor_pairs(
                left_block,
                right_block,
                mapped,
                self.cutoff,
                threshold_log_bound,
                threshold_accept_log_bound,
                self._has_center_types,
                self._typed_neighbors or right._typed_neighbors,
            )
            return current, scores, outcomes

        for i, j in pairs:
            if staging:
                (
                    next_left_count,
                    next_right_count,
                    staged_bytes,
                    added_left,
                    added_right,
                ) = next_staging_state(i, j)
            else:
                next_left_count = next_right_count = 0
                staged_bytes = 0
                added_left = added_right = []
            over_budget = (
                staging
                and memory_budget_bytes is not None
                and staged_bytes + reserve > memory_budget_bytes
            )
            if block and (
                len(block) >= block_limit
                or (
                    staging
                    and (
                        next_left_count > structure_limit
                        or next_right_count > structure_limit
                    )
                )
                or over_budget
            ):
                yield emit(block)
                block, left_ids, right_ids = [], set(), set()
                combined_ids = set()
                staged_bytes_current = 0
                if staging:
                    (
                        next_left_count,
                        next_right_count,
                        staged_bytes,
                        added_left,
                        added_right,
                    ) = next_staging_state(i, j)
                over_budget = (
                    staging
                    and memory_budget_bytes is not None
                    and staged_bytes + reserve > memory_budget_bytes
                )
            if over_budget:
                raise MemoryError(
                    "one comparison pair needs descriptors above the configured "
                    "device memory budget"
                )
            block.append((i, j))
            if staging:
                if self is right:
                    for value in added_left:
                        combined_ids.add(value)
                    right_ids.update(added_right)
                else:
                    left_ids.update(added_left)
                    right_ids.update(added_right)
                staged_bytes_current = staged_bytes
        if block:
            yield emit(block)

    def score_pairs(
        self,
        pair_indices: Tensor,
        other: RadialComparisonIndex | None = None,
        *,
        pair_chunk_size: int | None = None,
    ) -> Tensor:
        """Return one local-distance mismatch score for each requested pair.

        Parameters
        ----------
        pair_indices : torch.Tensor
            ``int32`` or ``int64`` ``[K, 2]`` indices. Column zero indexes this
            index and column one indexes ``other`` when supplied.
        other : RadialComparisonIndex, optional
            Index providing the second structure pool. Defaults to this index.
        pair_chunk_size : int, optional
            Upper bound on pairs processed together. The device default is
            capped at 4096 pairs on CPU and 8192 on CUDA.

        Returns
        -------
        torch.Tensor
            FP32 scores in input order on this index's device.

        Raises
        ------
        TypeError
            If pair indices or ``other`` have an unsupported type.
        ValueError
            If index settings are incompatible or the chunk size is invalid.
        IndexError
            If a pair contains an out-of-range structure index.
        MemoryError
            If one descriptor pair exceeds the configured CUDA workspace.
        """
        right = self if other is None else other
        if other is not None:
            self._check_other(other)
        if pair_chunk_size is not None and (
            isinstance(pair_chunk_size, bool)
            or not isinstance(pair_chunk_size, int)
            or pair_chunk_size <= 0
        ):
            raise ValueError("pair_chunk_size must be a positive integer")
        chunk_size = pair_chunk_size or (8192 if self.device.type == "cuda" else 4096)
        chunk_size = min(chunk_size, 8192 if self.device.type == "cuda" else 4096)
        if (
            not isinstance(pair_indices, Tensor)
            or pair_indices.ndim != 2
            or pair_indices.shape[1] != 2
        ):
            raise ValueError("pair_indices must have shape [K, 2]")
        if pair_indices.dtype not in (torch.int32, torch.int64):
            raise TypeError("pair_indices must have dtype int32 or int64")
        memory_budget = self._device_memory_budget()
        output = torch.empty(
            (pair_indices.shape[0],), dtype=torch.float32, device=self.device
        )
        for start in range(0, pair_indices.shape[0], chunk_size):
            valid_chunk = self._validate_pairs(
                pair_indices[start : start + chunk_size], right.num_structures
            )
            pair_chunk = [tuple(pair) for pair in valid_chunk.tolist()]
            offset = start
            for pair_block, scores, _ in self._descriptor_pair_blocks(
                pair_chunk, right, chunk_size, memory_budget
            ):
                output[offset : offset + len(pair_block)] = scores
                offset += len(scores)
        return output

    def _all_pairs(
        self, other: RadialComparisonIndex | None, chunk_size: int
    ) -> Iterator[Tensor]:
        """Generate logical pair tiles in deterministic row-major order.

        Parameters
        ----------
        other : RadialComparisonIndex or None
            Right-hand pool; ``None`` selects this index and self-pair order.
        chunk_size : int
            Maximum number of right-hand rows in each yielded tile.

        Yields
        ------
        torch.Tensor, shape (K, 2), dtype int32
            Logical pair IDs on the comparison device.

        Notes
        -----
        Self-comparison omits the diagonal and lower triangle. Cross-comparison
        traverses the full Cartesian product.
        """
        right_count = self.num_structures if other is None else other.num_structures
        for i in range(self.num_structures):
            start = i + 1 if other is None else 0
            for j_start in range(start, right_count, chunk_size):
                js = torch.arange(
                    j_start,
                    min(right_count, j_start + chunk_size),
                    dtype=torch.int32,
                    device=self.device,
                )
                is_ = torch.full_like(js, i)
                yield torch.stack((is_, js), dim=1)

    def _summary_log_bound(self, threshold: float) -> float | None:
        """Conservatively invert the FP32 inclusive expm1 score endpoint.

        Parameters
        ----------
        threshold : float
            Validated inclusive fractional mismatch threshold.

        Returns
        -------
        float or None
            Outward log-distance bound for conservative rejection, or None when
            pruning is disabled.

        Notes
        -----
        Accepted scores compare the FP32 ``expm1(log_score)`` output with one
        outward-rounded threshold ULP. The filter allows 32 further FP32 ULPs
        in score space for expm1 rounding, then rounds ``log1p`` upward before
        comparing summary extrema. Larger endpoints skip pruning.
        """
        return _conservative_summary_log_bound(threshold)

    def _early_accept_log_bound(self, threshold: float) -> float | None:
        """Return an inward FP32 endpoint for threshold-only match proofs.

        Parameters
        ----------
        threshold : float
            Validated inclusive fractional mismatch threshold.

        Returns
        -------
        float or None
            Inward log-distance bound for certified acceptance, or None when
            that proof is disabled.

        Notes
        -----
        The score endpoint is moved 32 ULPs inward before ``log1p`` and its
        FP32 result is moved four ULPs down. A directed score below this bound
        therefore remains below the public outward-rounded ``expm1`` limit.
        Values in the gap still use the exact-score path.
        """
        score = torch.tensor(threshold, dtype=torch.float32)
        infinity = torch.tensor(float("inf"), dtype=torch.float32)
        endpoint = torch.nextafter(score, infinity)
        value = float(endpoint)
        if not math.isfinite(value) or value >= 1.0e30:
            return None
        negative_infinity = torch.tensor(float("-inf"), dtype=torch.float32)
        for _ in range(32):
            endpoint = torch.nextafter(endpoint, negative_infinity)
        endpoint = endpoint.clamp_min(0.0)
        bound = torch.tensor(math.log1p(float(endpoint)), dtype=torch.float32)
        for _ in range(4):
            bound = torch.nextafter(bound, negative_infinity)
        return max(0.0, float(bound))

    def _filter_pairs(
        self,
        pairs: Tensor,
        right: RadialComparisonIndex,
        log_bound: float | None,
    ) -> Tensor:
        """Reject pairs whose radial summaries prove they exceed the limit.

        Parameters
        ----------
        pairs : torch.Tensor, shape (K, 2), dtype int32
            Ordered logical row-ID pairs on the comparison device.
        right : RadialComparisonIndex
            Compatible comparison index supplying the right-side summaries.
        log_bound : float or None
            Conservative log-distance difference limit; None skips summary
            pruning.

        Returns
        -------
        torch.Tensor
            Surviving pairs on the comparison device, preserving input order.

        Notes
        -----
        Updates the call-local search counters.

        L is a conservative log-score bound; compared descriptor values are
        natural logs of distances in angstroms. For a typed log-score at most L,
        every center row has a compatible row on the other side whose
        corresponding ``(neighbor type, rank)`` values differ by at most L,
        with missing ranks padded by ``log(cutoff)``. The
        reverse directed score gives this coverage in both directions, so the
        minimum and maximum of each selected rank over rows of each center type
        must also differ by at most L. A center type present on only one side
        has no compatible center and cannot produce a finite score. Untyped
        rows use the same argument on their global distance order statistics.
        Values beyond the conservative inclusive FP32 endpoint prove rejection;
        weaker summaries only reduce pruning.
        """
        self._last_search_stats["pairs_considered"] += pairs.shape[0]
        if log_bound is None or pairs.numel() == 0:
            return pairs
        ids_a = pairs[:, 0].to(dtype=torch.int64)
        ids_b = pairs[:, 1].to(dtype=torch.int64)
        typed_compatible = (
            self._typed_neighbors
            and right._typed_neighbors
            and self._typed_type_vocab == right._typed_type_vocab
            and self._typed_rank_indices == right._typed_rank_indices
        )
        if typed_compatible:
            if self.device.type == "cuda":
                # Keep the large typed-summary gathers at one fixed shape. Dedup
                # feeds variable candidate counts for each representative; doing
                # those gathers at the varying count retains many large allocator
                # blocks over the full search. Padding the last tile repeats a
                # valid pair, and its result is discarded below.
                tile_size = self._typed_filter_tile_size or 0
                if tile_size == 0:
                    raise MemoryError(
                        "one typed comparison pair exceeds the configured CUDA "
                        "filter workspace: requires approximately "
                        f"{self._typed_filter_bytes_per_pair} bytes plus the fixed "
                        f"{_CUDA_TYPED_FILTER_OUTPUT_RESERVE_BYTES}-byte output "
                        f"reserve, but only {self._typed_filter_workspace_bytes} "
                        "bytes are available"
                    )
                filtered_tiles = []
                for start in range(0, pairs.shape[0], tile_size):
                    tile = pairs[start : start + tile_size]
                    real_count = tile.shape[0]
                    if real_count < tile_size:
                        padding = tile[-1:].expand(tile_size - real_count, 2)
                        tile = torch.cat((tile, padding), dim=0)
                    rejected = self._typed_pair_rejections(tile, right, log_bound)
                    filtered_tiles.append(tile[:real_count][~rejected[:real_count]])
                filtered = torch.cat(filtered_tiles, dim=0)
            else:
                rejected = self._typed_pair_rejections(pairs, right, log_bound)
                filtered = pairs[~rejected]
        elif self._typed_neighbors or right._typed_neighbors:
            # The pools may contain different sets of type IDs. Their summary
            # columns then do not align, so score the pairs directly. Equal IDs
            # must still mean the same type across indexes.
            return pairs
        else:
            summaries_a = self._summary[ids_a.to(device=self._summary.device)].to(
                device=self.device
            )
            summaries_b = right._summary[ids_b.to(device=right._summary.device)].to(
                device=self.device
            )
            separated = (summaries_a - summaries_b).abs() > log_bound
            rejected = separated.any(dim=1)
            filtered = pairs[~rejected]
        self._last_search_stats["pairs_rejected"] += pairs.shape[0] - filtered.shape[0]
        return filtered

    def _typed_pair_rejections(
        self,
        pairs: Tensor,
        right: RadialComparisonIndex,
        log_bound: float,
    ) -> Tensor:
        """Mark typed pairs that compact summaries prove cannot match.

        Parameters
        ----------
        pairs : torch.Tensor, shape (K, 2)
            Logical row IDs for the left and right pools.
        right : RadialComparisonIndex
            Index providing right-hand summaries.
        log_bound : float
            Conservative log-distance mismatch limit.

        Returns
        -------
        torch.Tensor, shape (K,), dtype bool
            True where type presence or shared-type rank bounds prove rejection.

        Notes
        -----
        Surviving pairs retain input order for exact descriptor scoring.
        """
        ids_a = pairs[:, 0].to(dtype=torch.int64)
        ids_b = pairs[:, 1].to(dtype=torch.int64)
        summary_a = self._typed_summary[ids_a.to(device=self._typed_summary.device)].to(
            device=self.device
        )
        summary_b = right._typed_summary[
            ids_b.to(device=right._typed_summary.device)
        ].to(device=self.device)
        presence_a = self._center_type_presence[
            ids_a.to(device=self._center_type_presence.device)
        ].to(device=self.device)
        presence_b = right._center_type_presence[
            ids_b.to(device=right._center_type_presence.device)
        ].to(device=self.device)
        shared_types = (presence_a & presence_b).view(pairs.shape[0], -1, 1, 1, 1)
        summary_a.sub_(summary_b).abs_()
        separated = summary_a > log_bound
        separated &= shared_types
        return (presence_a != presence_b).any(dim=1) | separated.any(dim=(1, 2, 3, 4))

    def iter_matches(
        self,
        other: RadialComparisonIndex | None = None,
        *,
        threshold: float,
        pair_indices: Tensor | None = None,
        pair_chunk_size: int | None = None,
    ) -> Iterator[Tensor]:
        """Yield matching pair indices in deterministic bounded chunks.

        Parameters
        ----------
        other : RadialComparisonIndex, optional
            Cross-compare against this second pool. Defaults to self-comparison
            over unordered pairs.
        threshold : float
            Inclusive finite nonnegative fractional score threshold.
        pair_indices : torch.Tensor, optional
            Explicit ordered ``int32`` or ``int64`` ``[K, 2]`` pairs to test.
        pair_chunk_size : int, optional
            Upper bound on pair work per chunk. Defaults to at most 4096 pairs
            on CPU or 8192 on CUDA.

        Yields
        ------
        torch.Tensor
            Matching ``int32 [M, 2]`` index chunks on this index's device.

        Raises
        ------
        TypeError
            If an index or explicit pair tensor has an unsupported type.
        ValueError
            If index settings, threshold, or chunk size are invalid.
        IndexError
            If an explicit pair contains an out-of-range index.
        MemoryError
            If one descriptor pair or typed-filter tile exceeds the configured
            CUDA workspace.
        """
        if other is not None:
            self._check_other(other)
        t = _validate_threshold(threshold)
        right = self if other is None else other
        if pair_chunk_size is None:
            pair_chunk_size = 8192 if self.device.type == "cuda" else 4096
        if (
            isinstance(pair_chunk_size, bool)
            or not isinstance(pair_chunk_size, int)
            or pair_chunk_size <= 0
        ):
            raise ValueError("pair_chunk_size must be a positive integer")
        cap = 8192 if self.device.type == "cuda" else 4096
        pair_chunk_size = min(pair_chunk_size, cap)
        if pair_indices is None:
            pairs_iter = self._all_pairs(other, pair_chunk_size)
        else:
            if (
                not isinstance(pair_indices, Tensor)
                or pair_indices.ndim != 2
                or pair_indices.shape[1] != 2
            ):
                raise ValueError("pair_indices must have shape [K, 2]")
            if pair_indices.dtype not in (torch.int32, torch.int64):
                raise TypeError("pair_indices must have dtype int32 or int64")

            def explicit_pairs() -> Iterator[Tensor]:
                """Yield validated slices of the caller's ordered pair list."""
                for start in range(0, pair_indices.shape[0], pair_chunk_size):
                    validated = self._validate_pairs(
                        pair_indices[start : start + pair_chunk_size],
                        right.num_structures,
                    )
                    yield validated

            pairs_iter = explicit_pairs()
        memory_budget = self._device_memory_budget()
        limit = threshold_score(t, self.device)
        log_bound = self._summary_log_bound(t)
        accept_log_bound = (
            None if log_bound is None else self._early_accept_log_bound(t)
        )
        self._last_search_stats = {
            "pairs_considered": 0,
            "pairs_rejected": 0,
            "pairs_scored": 0,
            "pairs_certified_matches": 0,
        }
        for pairs in pairs_iter:
            matches = self._match_chunk(
                pairs, right, limit, log_bound, accept_log_bound, memory_budget
            )
            if matches.numel():
                yield matches

    def _match_chunk(
        self,
        pairs: Tensor,
        right: RadialComparisonIndex,
        limit: Tensor,
        log_bound: float | None,
        accept_log_bound: float | None,
        memory_budget: int | None,
    ) -> Tensor:
        """Filter and score one ordered pair tile.

        Parameters
        ----------
        pairs : torch.Tensor, shape (K, 2)
            Logical pair IDs on this index's device.
        right : RadialComparisonIndex
            Right-hand structure pool.
        limit : torch.Tensor
            Inclusive FP32 score endpoint on the comparison device.
        log_bound, accept_log_bound : float or None
            Conservative summary rejection and early-accept bounds.
        memory_budget : int or None
            Temporary CUDA allowance in bytes.

        Returns
        -------
        torch.Tensor, shape (M, 2), dtype int32
            Accepted logical pairs in the original tile order.

        Notes
        -----
        Summary checks remove only proven nonmatches; survivors are scored or
        resolved by conservative threshold outcomes.
        """
        pairs = self._filter_pairs(pairs, right, log_bound)
        if pairs.numel() == 0:
            return torch.empty((0, 2), dtype=torch.int32, device=self.device)
        pair_list = [tuple(pair) for pair in pairs.tolist()]
        result: list[tuple[int, int]] = []
        for pair_block, scores, outcomes in self._descriptor_pair_blocks(
            pair_list,
            right,
            len(pair_list),
            memory_budget,
            threshold_log_bound=log_bound,
            threshold_accept_log_bound=accept_log_bound,
        ):
            if outcomes is None:
                matched = torch.isfinite(scores) & (scores <= limit)
                self._last_search_stats["pairs_scored"] += len(pair_block)
                match_values = matched.tolist()
            else:
                matched = (outcomes == 2) | (
                    (outcomes == 1) & torch.isfinite(scores) & (scores <= limit)
                )
                flags = (
                    torch.stack((outcomes, matched.to(torch.int32)), dim=1)
                    .cpu()
                    .tolist()
                )
                self._last_search_stats["pairs_scored"] += sum(
                    outcome == 1 for outcome, _ in flags
                )
                self._last_search_stats["pairs_certified_matches"] += sum(
                    outcome == 2 for outcome, _ in flags
                )
                self._last_search_stats["pairs_rejected"] += sum(
                    outcome == 0 for outcome, _ in flags
                )
                match_values = [bool(match) for _, match in flags]
            result.extend(
                pair
                for pair, keep in zip(pair_block, match_values, strict=True)
                if keep
            )
        return torch.tensor(result, dtype=torch.int32, device=self.device).reshape(
            -1, 2
        )

    def find_matches(
        self,
        other: RadialComparisonIndex | None = None,
        *,
        threshold: float,
        pair_indices: Tensor | None = None,
        pair_chunk_size: int | None = None,
    ) -> Tensor:
        """Collect proposed matches in deterministic pair order.

        Parameters
        ----------
        other : RadialComparisonIndex, optional
            Cross-compare against this second pool. Defaults to self-comparison
            over unordered pairs.
        threshold : float
            Inclusive finite nonnegative fractional score threshold.
        pair_indices : torch.Tensor, optional
            Explicit ordered ``int32`` or ``int64`` ``[K, 2]`` pairs to test.
        pair_chunk_size : int, optional
            Upper bound on temporary pair work; returned matches are collected
            in one tensor regardless of this value.

        Returns
        -------
        torch.Tensor
            Matching ``int32 [M, 2]`` pairs on this index's device.

        Raises
        ------
        TypeError
            If an index or explicit pair tensor has an unsupported type.
        ValueError
            If index settings, threshold, or chunk size are invalid.
        IndexError
            If an explicit pair contains an out-of-range index.
        MemoryError
            If one descriptor pair or typed-filter tile exceeds the configured
            CUDA workspace.
        """
        chunks = list(
            self.iter_matches(
                other,
                threshold=threshold,
                pair_indices=pair_indices,
                pair_chunk_size=pair_chunk_size,
            )
        )
        return (
            torch.cat(chunks, dim=0)
            if chunks
            else torch.empty((0, 2), dtype=torch.int32, device=self.device)
        )

    def deduplicate(
        self,
        *,
        threshold: float,
        confirm: Callable[[Tensor], Tensor] | None = None,
        pair_chunk_size: int | None = None,
    ) -> DeduplicationResult:
        """Keep each structure that does not match an earlier retained structure.

        Visit input structures in order. Compare each with structures already
        kept and assign it to its first confirmed match, or retain it if none
        matches. If A matches B and B matches C but A does not match C, input
        order A, B, C retains A and C.

        Parameters
        ----------
        threshold : float
            Inclusive finite nonnegative fractional score threshold.
        confirm : callable, optional
            Receives each proposed-match chunk as ordered ``int32 [K, 2]``
            original-pool indices on this index's device. It must return an
            ordered subset on the same device. Column zero is the candidate and
            column one is an earlier retained representative. Without a callback,
            all proposed matches are accepted.
        pair_chunk_size : int, optional
            Upper bound on candidate-to-representative pair work per chunk.

        Returns
        -------
        DeduplicationResult
            Retained indices, each structure's representative index, and
            multiplicities aligned with retained structures.

        Raises
        ------
        TypeError
            If ``confirm`` is not callable.
        ValueError
            If the threshold, pair chunk size, or callback result is invalid.
        MemoryError
            If one descriptor pair or typed-filter tile exceeds the configured
            CUDA workspace.
        """
        t = _validate_threshold(threshold)
        if confirm is not None and not callable(confirm):
            raise TypeError("confirm must be callable or None")
        if pair_chunk_size is None:
            pair_chunk_size = 8192 if self.device.type == "cuda" else 4096
        cap = 8192 if self.device.type == "cuda" else 4096
        if (
            isinstance(pair_chunk_size, bool)
            or not isinstance(pair_chunk_size, int)
            or pair_chunk_size <= 0
        ):
            raise ValueError("pair_chunk_size must be a positive integer")
        pair_chunk_size = min(pair_chunk_size, cap)
        retained: list[int] = []
        retained_positions: dict[int, int] = {}
        representatives = [-1] * self.num_structures
        multiplicities: list[int] = []
        self._last_dedup_stats = {
            "pairs_considered": 0,
            "pairs_rejected": 0,
            "pairs_scored": 0,
            "pairs_certified_matches": 0,
        }
        for candidate in range(self.num_structures):
            if not retained:
                retained_positions[candidate] = 0
                retained.append(candidate)
                representatives[candidate] = candidate
                multiplicities.append(1)
                continue
            assigned = False
            for start in range(0, len(retained), pair_chunk_size):
                representatives_block = retained[start : start + pair_chunk_size]
                rep_indices = torch.tensor(
                    representatives_block, dtype=torch.int32, device=self.device
                )
                candidate_pairs = torch.empty(
                    (len(representatives_block), 2),
                    dtype=torch.int32,
                    device=self.device,
                )
                candidate_pairs[:, 0] = candidate
                candidate_pairs[:, 1] = rep_indices
                radial = self.find_matches(
                    threshold=t,
                    pair_indices=candidate_pairs,
                    pair_chunk_size=pair_chunk_size,
                )
                for key in self._last_search_stats:
                    self._last_dedup_stats[key] += self._last_search_stats[key]
                if radial.numel() == 0:
                    continue
                if confirm is None:
                    confirmed = radial
                    validation_source = radial
                else:
                    validation_source = radial.clone()
                    confirmed = confirm(radial)
                _validate_confirmed_subset(validation_source, confirmed, self.device)
                accepted = confirmed.tolist()
                if accepted:
                    representative = accepted[0][1]
                    representatives[candidate] = representative
                    multiplicities[retained_positions[representative]] += 1
                    assigned = True
                    break
            if not assigned:
                retained_positions[candidate] = len(retained)
                retained.append(candidate)
                representatives[candidate] = candidate
                multiplicities.append(1)
        return DeduplicationResult(
            torch.tensor(retained, dtype=torch.int32, device=self.device),
            torch.tensor(representatives, dtype=torch.int32, device=self.device),
            torch.tensor(multiplicities, dtype=torch.int32, device=self.device),
        )


def deduplicate_stream(
    count: int,
    read_typed_batch: Callable[[Tensor], tuple[Batch, Tensor | None]],
    *,
    cutoff: float,
    threshold: float,
    type_vocabulary: Sequence[int] | None = None,
    priority_order: Sequence[int] | Tensor | None = None,
    device: torch.device | str = "cpu",
    input_batch_size: int | None = None,
    pair_block_size: int | None = None,
    max_batch_atoms: int = 200_000,
    max_memory_fraction: float = 0.85,
    summary_coordinate_count: int = 32,
    confirm: Callable[[Tensor], Tensor] | None = None,
) -> DeduplicationResult:
    """Deduplicate a pool in priority order while bounding descriptor blocks.

    Candidate structures are resolved in ``priority_order`` against earlier
    retained representatives. Each proposed pair passes untyped, center-typed,
    and center-plus-neighbor-typed radial screens in that order when types are
    supplied, and only the untyped screen otherwise. Retention is greedy and
    order-dependent; radial matches are approximate and need not be transitive.

    When ``summary_coordinate_count`` is positive, a preliminary conservative
    rejection check uses up to that many selected coordinates; the selected
    count may be lower. Pairs that pass this check go to the radial index
    matcher; summary bounds can reject a pair but never establish a match.
    Setting the count to zero disables the pilot and outer check and sends all
    still-eligible pairs to the index matcher, which may apply its built-in
    conservative summary bounds before scoring full descriptors. Candidate
    chunks contain up to ``input_batch_size`` rows. Screening keeps each active
    candidate tile live while it scans earlier representative blocks of up to
    ``pair_block_size`` rows.

    Parameters
    ----------
    count : int
        Number of logical rows available from ``read_typed_batch``.
    read_typed_batch : callable
        Given CPU int64 row IDs, returns a ``(Batch, atom_types)`` pair. The
        int32 or int64 type vector aligns with all atoms, or is ``None`` on
        every call for untyped comparison. Loader values must remain stable.
    type_vocabulary : sequence of int, optional
        Fixed set of signed int32 type IDs used across batches. Required when
        the loader returns atom types.
    priority_order : sequence of int or torch.Tensor, optional
        Permutation of logical row IDs that defines candidate and representative
        priority. Defaults to logical row order.
    cutoff : float
        Descriptor cutoff in angstroms.
    threshold : float
        Inclusive finite nonnegative radial mismatch threshold.
    device : torch.device or str
        Device used to build and score descriptors. Defaults to CPU.
    input_batch_size : int, optional
        Maximum rows requested per summary read and per candidate chunk.
        Defaults to 1024 on CUDA and 64 on CPU.
    pair_block_size : int, optional
        Maximum earlier representatives considered together; it does not limit
        candidate chunk size. Defaults to 256 on CUDA and 32 on CPU.
    max_batch_atoms : int, default=200000
        Maximum atoms in a descriptor tile.
    max_memory_fraction : float, default=0.85
        Maximum fraction of CUDA memory available to PyTorch assigned to
        descriptor construction and scoring workspace, including reusable
        caching-allocator bytes. This is an estimate, not a reservation.
    summary_coordinate_count : int, default=32
        Maximum number of coordinates used by the conservative preliminary
        rejection check, from 0 through 128. The selected count may be lower.
        Pairs that pass go to the radial index matcher; summary bounds can
        reject a pair but never establish a match. Zero disables the pilot and
        outer check and sends all still-eligible pairs to the index matcher,
        which may use its built-in conservative summary bounds before scoring
        full descriptors.
    confirm : callable, optional
        Receives ordered radial-match proposals as int32 [K, 2] original row
        IDs on the comparison device. Column zero is the candidate; column one
        is an earlier retained representative. Return an ordered subset on the
        same device. A rejected proposal does not discard the candidate; later
        representatives are tried. Without a callback, radial matches suffice.
        Calls may be grouped across candidates; decisions must depend on pair
        data rather than the global callback invocation order.
    Returns
    -------
    DeduplicationResult
        ``retained_indices`` contains logical IDs in priority order.
        ``representative_indices`` is indexed by original logical ID and stores
        each row's representative logical ID. ``multiplicities`` aligns with
        ``retained_indices``.

    Raises
    ------
    TypeError
        If the loader, returned batch/type vector, or confirm callback has an
        unsupported type.
    ValueError
        If the input ordering, type vocabulary, threshold, cutoff, block sizes,
        or summary coordinate count are invalid.
    """
    from nvalchemi.csp._comparison.stream import deduplicate_stream as _stream

    return _stream(
        count,
        read_typed_batch,
        cutoff=cutoff,
        threshold=threshold,
        type_vocabulary=type_vocabulary,
        priority_order=priority_order,
        device=device,
        input_batch_size=input_batch_size,
        pair_block_size=pair_block_size,
        max_batch_atoms=max_batch_atoms,
        max_memory_fraction=max_memory_fraction,
        summary_coordinate_count=summary_coordinate_count,
        confirm=confirm,
    )


def deduplicate_batch(
    batch: Batch,
    *,
    atom_types: Tensor | None = None,
    cutoff: float,
    threshold: float,
    priority_order: Sequence[int] | Tensor | None = None,
    device: torch.device | str | None = None,
    max_batch_atoms: int = 200_000,
    max_memory_fraction: float = 0.85,
    summary_coordinate_count: int = 32,
    confirm: Callable[[Tensor], Tensor] | None = None,
) -> DeduplicationResult:
    """Deduplicate every structure in one Batch with the tiled stream engine.

    Geometry and supplied atom types are read from ``batch`` by structure ID.
    The result uses original Batch row IDs and contains int32 tensors on the
    comparison device. With supplied types, one shared vocabulary is derived
    from the full type vector and the same pair must pass untyped, center-typed,
    then fully typed radial screens. Without supplied types, only the untyped
    screen is used.

    Parameters
    ----------
    batch : Batch
        Input structure pool.
    atom_types : torch.Tensor, optional
        Generic int32 or int64 label for every atom. Values are compared as
        caller-defined IDs and are not inferred from atomic numbers.
    cutoff : float
        Radial descriptor cutoff in angstroms.
    threshold : float
        Inclusive finite nonnegative fractional mismatch threshold.
    priority_order : sequence of int or torch.Tensor, optional
        Permutation of Batch row IDs defining greedy retention order.
    device : torch.device or str, optional
        Descriptor and score device. Defaults to ``batch.device``.
    max_batch_atoms : int, default=200000
        Maximum atoms in one descriptor tile.
    max_memory_fraction : float, default=0.85
        Maximum fraction of CUDA memory available to PyTorch assigned to
        descriptor construction and scoring workspace, including reusable
        caching-allocator bytes. This is an estimate, not a reservation.
    summary_coordinate_count : int, default=32
        Maximum coordinates used by the conservative preliminary rejection
        check, from 0 through 128. The selected count may be lower. Pairs that
        pass go to the radial index matcher; summary bounds can reject a pair
        but never establish a match. Zero disables the pilot and outer check
        and sends all still-eligible pairs to the index matcher, which may use
        its built-in conservative summary bounds before scoring full
        descriptors.
    confirm : callable, optional
        Receives ordered radial-match proposals as int32 [K, 2] original row
        IDs on the comparison device. Column zero is the candidate; column one
        is an earlier retained representative. Return an ordered subset on the
        same device. A rejected proposal does not discard the candidate; later
        representatives are tried. Without a callback, radial matches suffice.
        Calls may be grouped across candidates; decisions must depend on pair
        data rather than the global callback invocation order.

    Returns
    -------
    DeduplicationResult
        Retained Batch row IDs, each row's representative ID, and retained
        multiplicities, all as int32 tensors on the comparison device.
    """
    if not isinstance(batch, Batch):
        raise TypeError("batch must be a nvalchemi.data.Batch")
    if confirm is not None and not callable(confirm):
        raise TypeError("confirm must be callable or None")
    if atom_types is not None:
        if (
            not isinstance(atom_types, Tensor)
            or atom_types.ndim != 1
            or atom_types.numel() != batch.num_nodes
        ):
            raise ValueError(
                "atom_types must be a one-dimensional tensor with one entry per atom"
            )
        if atom_types.dtype not in (torch.int32, torch.int64):
            raise TypeError("atom_types must have dtype torch.int32 or torch.int64")
        int32_info = torch.iinfo(torch.int32)
        if atom_types.dtype is torch.int64 and torch.any(
            (atom_types < int32_info.min) | (atom_types > int32_info.max)
        ):
            raise ValueError("atom_types values must fit signed int32")
        if torch.any(atom_types == int32_info.min):
            raise ValueError("atom_types cannot use the reserved int32 minimum")
        atom_types = atom_types.detach().to(dtype=torch.int32).contiguous()
        vocabulary: tuple[int, ...] | None = tuple(
            sorted(set(atom_types.detach().cpu().tolist()))
        )
    else:
        vocabulary = None

    ptr = batch.batch_ptr.detach().cpu().tolist()

    def read(ids: Tensor) -> tuple[Batch, Tensor | None]:
        """Select requested structures and aligned caller-supplied atom types."""
        row_ids = ids.tolist()
        selected = batch.index_select(row_ids)
        if atom_types is None:
            return selected, None
        if not row_ids:
            return selected, atom_types.new_empty((0,))
        selected_types = torch.cat(
            [atom_types[ptr[row] : ptr[row + 1]] for row in row_ids], dim=0
        )
        return selected, selected_types

    target_device = torch.device(device) if device is not None else batch.device
    return deduplicate_stream(
        batch.num_graphs,
        read,
        type_vocabulary=vocabulary,
        cutoff=cutoff,
        threshold=threshold,
        priority_order=priority_order,
        device=target_device,
        max_batch_atoms=max_batch_atoms,
        max_memory_fraction=max_memory_fraction,
        summary_coordinate_count=summary_coordinate_count,
        confirm=confirm,
    )


def iter_matches_stream(
    count: int,
    read_typed_batch: Callable[[Tensor], tuple[Batch, Tensor | None]],
    *,
    cutoff: float,
    threshold: float,
    type_vocabulary: Sequence[int] | None = None,
    other_count: int | None = None,
    read_other_typed_batch: Callable[[Tensor], tuple[Batch, Tensor | None]]
    | None = None,
    device: torch.device | str = "cpu",
    max_batch_atoms: int = 200_000,
    max_memory_fraction: float = 0.85,
    pair_chunk_size: int = 8192,
    summary_coordinate_count: int = 32,
) -> Iterator[Tensor]:
    """Yield radial matches from one or two loader-backed structure pools.

    The function begins reading when the iterator is advanced. Loaders receive
    CPU int64 logical row IDs and return a ``(Batch, atom_types)`` pair, with
    an int32 or int64 type vector aligned to all Batch atoms, or ``None`` for
    untyped comparison. Typed loaders must use the same fixed type vocabulary
    in both pools and must return stable values for every requested ID.

    Self-comparison emits each unordered pair once as ``(left, right)`` with
    ``left < right``. Cross-pool comparison emits ``(left_row, right_row)``.
    Both forms are lexicographic. Output tensors are nonempty int32 ``[K, 2]``
    chunks on ``device`` and contain at most ``pair_chunk_size`` pairs.
    Full descriptors are built for active atom-bounded pair tiles. Compact
    summaries and sorted search columns reside on the comparison device and
    grow with the pool sizes and selected coordinate count.
    Setting ``summary_coordinate_count=0`` disables the outer pilot-selected
    check and sends every eligible pair to the radial index matcher, which may
    still apply its built-in conservative summary bound before full scoring.
    Closing the iterator releases its call-local tensors without closing either
    caller-owned loader.

    Parameters
    ----------
    count : int
        Number of rows available from ``read_typed_batch``.
    read_typed_batch : callable
        Loader for the left pool. It receives CPU int64 row IDs.
    cutoff : float
        Radial descriptor cutoff in angstroms.
    threshold : float
        Inclusive finite nonnegative fractional mismatch threshold.
    type_vocabulary : sequence of int, optional
        Fixed set of signed int32 atom type IDs used by typed loaders.
    other_count : int, optional
        Number of rows in a second pool. Must be supplied with
        ``read_other_typed_batch``; omit both for self-comparison.
    read_other_typed_batch : callable, optional
        Loader for the right pool in a cross-pool comparison.
    device : torch.device or str, default="cpu"
        Device used to build and score descriptor tiles.
    max_batch_atoms : int, default=200000
        Maximum atom count in each active descriptor tile. Cross-comparison may
        keep up to this many atoms on both sides at once; the shared CUDA memory
        budget accounts for both live descriptor sets.
    max_memory_fraction : float, default=0.85
        Maximum fraction of CUDA memory available to PyTorch assigned to
        descriptor construction and scoring workspace, including reusable
        caching-allocator bytes. This is an estimate, not a reservation.
    pair_chunk_size : int, default=8192
        Maximum number of logical pairs screened or yielded at once.
    summary_coordinate_count : int, default=32
        Maximum coordinates for the preliminary conservative rejection check,
        from 0 through 128. The selected count may be lower. Passing pairs go
        to the radial index matcher; summary bounds can reject but not match.
        Zero disables only the outer pilot check; the matcher may retain its
        built-in conservative summary bound.

    Yields
    ------
    torch.Tensor
        Nonempty int32 ``[K, 2]`` matching index chunks, in lexicographic order.

    Raises
    ------
    TypeError
        If a loader, vocabulary, or option has an unsupported type.
    ValueError
        If sizes, callbacks, cutoff, threshold, or memory options are invalid.
    MemoryError
        If one structure or pair cannot fit the atom or CUDA memory limit.
    """
    from nvalchemi.csp._comparison.stream import (
        iter_matches_stream as _iter_matches_stream,
    )

    return _iter_matches_stream(
        count,
        read_typed_batch,
        cutoff=cutoff,
        threshold=threshold,
        type_vocabulary=type_vocabulary,
        other_count=other_count,
        read_other_typed_batch=read_other_typed_batch,
        device=device,
        max_batch_atoms=max_batch_atoms,
        max_memory_fraction=max_memory_fraction,
        pair_chunk_size=pair_chunk_size,
        summary_coordinate_count=summary_coordinate_count,
    )
