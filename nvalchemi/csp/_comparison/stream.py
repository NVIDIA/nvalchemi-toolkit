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
"""Bounded-memory, order-preserving radial structure deduplication."""

from __future__ import annotations

import copy
import math
import time
from collections.abc import Callable, Iterator, Sequence
from contextvars import ContextVar
from typing import Any

import torch
from torch import Tensor

from nvalchemi.csp._comparison.store import (
    _log_cutoff_fp32,
    available_cuda_bytes,
    build_descriptor_stores,
    comparison_geometry,
    validate_include_hydrogens,
)
from nvalchemi.csp._validation import (
    finite_nonnegative as _finite_nonnegative,
)
from nvalchemi.csp._validation import (
    nonnegative_int as _nonnegative_int,
)
from nvalchemi.csp._validation import (
    positive_integer,
)
from nvalchemi.csp.comparison import (
    DeduplicationResult,
    RadialComparisonIndex,
    _conservative_summary_log_bound,
    _validate_confirmed_subset,
)
from nvalchemi.data import Batch, resolve_device

_PILOT_CONTIGUOUS = 160
_PILOT_SPREAD = 160
_PILOT_REFERENCE = 128
_PILOT_TRAIN_QUERY = 16
_PILOT_HELDOUT_QUERY = 16
_DEFAULT_SUMMARY_COORDINATE_COUNT = 32
_MAX_SUMMARY_COORDINATE_COUNT = 128
_TYPED_RANK_COUNT = 7
_FILTER_BLOCK_SIZE = 65_536
_SUMMARY_PAIR_FILTER_MAX_PAIRS = 8_192
_SUMMARY_PAIR_FILTER_WORKSPACE_BYTES = 8 * 1024 * 1024
_MAX_BATCH_ATOMS = 200_000


class _RetryActiveTile(Exception):
    """Signal a candidate tile split after a descriptor capacity failure."""


_LAST_STREAM_STATS: ContextVar[dict[str, Any] | None] = ContextVar(
    "csp_deduplicate_stream_stats", default=None
)


def _positive_int(value: Any, name: str) -> int:
    """Validate a positive Python integer stream option."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be a positive integer")
    try:
        positive_integer(value, name=name)
    except ValueError:
        raise ValueError(f"{name} must be a positive integer") from None
    return value


def _pilot_role_counts(sample_count: int) -> tuple[int, int, int]:
    """Split a pilot sample into reference, training, and holdout rows.

    Parameters
    ----------
    sample_count : int
        Number of priority-ordered pilot rows available.

    Returns
    -------
    tuple of int
        Counts for reference, training-query, and held-out-query roles.

    Notes
    -----
    The roles are disjoint so selectivity estimates never use a query as its
    own reference.
    """
    if sample_count >= _PILOT_CONTIGUOUS + _PILOT_SPREAD:
        return _PILOT_REFERENCE, _PILOT_TRAIN_QUERY, _PILOT_HELDOUT_QUERY
    if sample_count >= 3 * _PILOT_TRAIN_QUERY:
        holdout = _PILOT_HELDOUT_QUERY
        training = _PILOT_TRAIN_QUERY
        return (
            min(_PILOT_REFERENCE, sample_count - training - holdout),
            training,
            holdout,
        )
    training = min(_PILOT_TRAIN_QUERY, sample_count // 2)
    return sample_count - training, training, 0


def _get_last_stream_stats() -> dict[str, Any] | None:
    """Return a defensive copy of counters for the latest call in this context.

    Returns
    -------
    dict or None
        Copy of the most recent stream counters, or ``None`` before a call.
    """
    stats = _LAST_STREAM_STATS.get()
    return copy.deepcopy(stats) if stats is not None else None


def _pilot_ids(priority: list[int]) -> tuple[list[int], list[str]]:
    """Choose pilot rows while retaining their contiguous or spread stratum.

    Parameters
    ----------
    priority : list of int
        Original row IDs in requested greedy priority order.

    Returns
    -------
    tuple
        Role-ordered row IDs and parallel ``contiguous``/``spread`` labels.

    Notes
    -----
    Roles are ordered reference, training, held-out, then remaining pilot rows.
    """
    contiguous = priority[: min(_PILOT_CONTIGUOUS, len(priority))]
    available = priority[len(contiguous) :]
    spread_count = min(_PILOT_SPREAD, len(available))
    if spread_count > 1:
        spread = [
            available[(i * (len(available) - 1)) // (spread_count - 1)]
            for i in range(spread_count)
        ]
    else:
        spread = available[:spread_count]
    contiguous_cursor = spread_cursor = 0

    def take(count: int) -> list[tuple[int, str]]:
        """Take the next pilot rows from both sampling strata.

        Parameters
        ----------
        count : int
            Number of rows requested for the next role.

        Returns
        -------
        list of tuple
            Row IDs paired with their contiguous/spread source label.

        Notes
        -----
        Cursors advance so a row is assigned to at most one pilot role.
        """
        nonlocal contiguous_cursor, spread_cursor
        contiguous_count = min(count // 2, len(contiguous) - contiguous_cursor)
        spread_count = min(count - contiguous_count, len(spread) - spread_cursor)
        contiguous_count += min(
            count - contiguous_count - spread_count,
            len(contiguous) - contiguous_cursor - contiguous_count,
        )
        output = [
            (row, "contiguous")
            for row in contiguous[
                contiguous_cursor : contiguous_cursor + contiguous_count
            ]
        ]
        contiguous_cursor += contiguous_count
        output.extend(
            (row, "spread")
            for row in spread[spread_cursor : spread_cursor + spread_count]
        )
        spread_cursor += spread_count
        return output

    reference_count, training_count, holdout_count = _pilot_role_counts(len(priority))
    roles = take(reference_count)
    roles.extend(take(training_count))
    roles.extend(take(holdout_count))
    roles.extend((row, "contiguous") for row in contiguous[contiguous_cursor:])
    roles.extend((row, "spread") for row in spread[spread_cursor:])
    return [row for row, _ in roles], [label for _, label in roles]


def _read_batch(
    loader: Callable[[Tensor], tuple[Batch, Tensor | None]],
    ids: Sequence[int],
    vocabulary: tuple[int, ...] | None,
    *,
    include_hydrogens: bool = True,
) -> tuple[Batch, Tensor | None]:
    """Read and validate one set of logical rows from a caller-owned loader.

    Parameters
    ----------
    loader : callable
        Reader accepting CPU int64 row IDs and returning a Batch and types.
    ids : sequence of int
        Logical row IDs in desired output order.
    vocabulary : tuple of int or None
        Fixed signed-int32 type vocabulary required when types are present.
    include_hydrogens : bool
        Whether atomic-number-one atoms remain in the comparison geometry.

    Returns
    -------
    tuple
        Batch and either atom-aligned int32 types on its device or ``None``.

    Raises
    ------
    TypeError
        If the loader result does not contain a Batch.
    ValueError
        If row count, type-vector representation or alignment, vocabulary, or
        type values are invalid.

    Notes
    -----
    Loader ownership remains with the caller; this function only reads rows.
    """
    batch, atom_types = loader(torch.tensor(ids, dtype=torch.int64))
    if not isinstance(batch, Batch):
        raise TypeError("read_typed_batch must return a Batch as its first value")
    if batch.num_graphs != len(ids):
        raise ValueError("read_typed_batch must return the requested row count")
    if atom_types is None:
        if vocabulary is not None:
            raise ValueError("type_vocabulary requires atom_types from every batch")
        return comparison_geometry(
            batch,
            None,
            include_hydrogens=include_hydrogens,
            atom_types_validated=True,
        )
    if vocabulary is None:
        raise ValueError("atom_types require a fixed type_vocabulary")
    if (
        not isinstance(atom_types, Tensor)
        or atom_types.ndim != 1
        or atom_types.numel() != batch.num_nodes
        or atom_types.dtype not in (torch.int32, torch.int64)
    ):
        raise ValueError("atom_types must be an aligned int32 or int64 vector")
    int32_info = torch.iinfo(torch.int32)
    if atom_types.dtype == torch.int64 and torch.any(
        (atom_types < int32_info.min) | (atom_types > int32_info.max)
    ):
        raise ValueError("atom_types values must fit signed int32")
    if torch.any(atom_types == int32_info.min):
        raise ValueError("atom_types cannot contain the reserved int32 minimum")
    types = atom_types.to(device=batch.device, dtype=torch.int32).contiguous()
    if types.numel():
        if types.device.type == "cuda":
            allowed_types = torch.tensor(
                vocabulary, dtype=torch.int32, device=types.device
            )
            if not bool(torch.isin(types, allowed_types).all()):
                raise ValueError("atom_types contain a value outside type_vocabulary")
        elif not set(types.tolist()).issubset(vocabulary):
            raise ValueError("atom_types contain a value outside type_vocabulary")
    return comparison_geometry(
        batch,
        types,
        include_hydrogens=include_hydrogens,
        atom_types_validated=True,
    )


def _build_atom_counts(
    count: int,
    loader: Callable[[Tensor], tuple[Batch, Tensor | None]],
    vocabulary: tuple[int, ...] | None,
    chunk_size: int,
    max_batch_atoms: int,
    include_hydrogens: bool = True,
) -> Tensor:
    """Read atom counts in bounded loader chunks without building descriptors.

    Parameters
    ----------
    count : int
        Number of original logical rows.
    loader : callable
        Loader used to fetch bounded row groups.
    vocabulary : tuple of int or None
        Fixed vocabulary used to validate optional atom types.
    chunk_size : int
        Maximum rows requested from the loader at once.
    max_batch_atoms : int
        Maximum allowed atoms in any one structure.
    include_hydrogens : bool
        Whether atom counts include hydrogen atoms.

    Returns
    -------
    torch.Tensor, shape (count,), dtype int32, device CPU
        Atom counts indexed by original logical row ID.

    Raises
    ------
    MemoryError
        If one structure exceeds ``max_batch_atoms``.

    Notes
    -----
    Counts allow active descriptor tiles to be bounded before construction.
    """
    atom_counts = torch.empty((count,), dtype=torch.int32)
    for start in range(0, count, chunk_size):
        ids = list(range(start, min(count, start + chunk_size)))
        batch, types = _read_batch(
            loader, ids, vocabulary, include_hydrogens=include_hydrogens
        )
        pointers = batch.batch_ptr.detach().cpu().tolist()
        counts = [pointers[i + 1] - pointers[i] for i in range(len(ids))]
        for row_id, atom_count in zip(ids, counts, strict=True):
            if atom_count > max_batch_atoms:
                raise MemoryError(
                    f"structure {row_id} requires {atom_count} atoms, above "
                    f"max_batch_atoms capacity {max_batch_atoms}"
                )
        atom_counts[torch.tensor(ids, dtype=torch.int64)] = torch.tensor(
            counts, dtype=torch.int32
        )
        del batch, types
    return atom_counts


def _canonical_typed_summary(
    descriptor_store: Any,
    vocabulary: tuple[int, ...],
    cutoff: float,
    *,
    device: torch.device | None = None,
) -> tuple[Tensor, Tensor]:
    """Map local typed summary axes into the caller's fixed type vocabulary.

    Parameters
    ----------
    descriptor_store : DescriptorStore
        Store containing a batch-local typed layout and center presence.
    vocabulary : tuple of int
        Stable sorted type IDs used by every loader batch.
    cutoff : float
        Radial cutoff in angstroms used for the summary padding value.
    device : torch.device, optional
        Destination device; defaults to the typed summary's current device.

    Returns
    -------
    tuple of torch.Tensor
        Flattened FP32 features of shape ``(N, F)`` and bool center presence
        of shape ``(N, T)`` in stable vocabulary order.

    Notes
    -----
    Missing types receive cutoff-padded summaries and false center presence.
    """
    local_vocab = descriptor_store.typed_type_vocab
    if device is None:
        device = descriptor_store.typed_summaries.device
    summary = descriptor_store.typed_summaries.detach().to(
        device=device, dtype=torch.float32
    )
    local_presence = descriptor_store.center_type_presence.detach().to(device=device)
    vocab_size = len(vocabulary)
    log_cutoff = _log_cutoff_fp32(cutoff)
    full = torch.full(
        (descriptor_store.num_structures, vocab_size, vocab_size, _TYPED_RANK_COUNT, 2),
        log_cutoff,
        device=device,
        dtype=torch.float32,
    )
    presence = torch.zeros(
        (descriptor_store.num_structures, vocab_size), dtype=torch.bool, device=device
    )
    if local_vocab:
        to_global = {value: position for position, value in enumerate(vocabulary)}
        mapping = [to_global[value] for value in local_vocab]
        mapping_tensor = torch.tensor(mapping, dtype=torch.int64, device=device)
        presence[:, mapping_tensor] = local_presence
        full[:, mapping_tensor[:, None], mapping_tensor[None, :]] = summary
    return full.reshape(descriptor_store.num_structures, -1), presence


def _mode_names(atom_types: Tensor | None) -> tuple[str, ...]:
    """Select radial screening modes supported by the available atom types.

    Parameters
    ----------
    atom_types : torch.Tensor or None
        Atom-aligned types; ``None`` selects untyped comparison.

    Returns
    -------
    tuple of str
        Mode names to build for the ordered screening cascade.
    """
    return ("untyped",) if atom_types is None else ("untyped", "center", "full")


def _cuda_budget(device: torch.device, max_memory_fraction: float) -> int | None:
    """Estimate this call's CUDA allocation allowance from PyTorch availability.

    Parameters
    ----------
    device : torch.device
        Target construction and scoring device.
    max_memory_fraction : float
        Fraction of PyTorch-available bytes assigned to this call.

    Returns
    -------
    int or None
        Estimated CUDA byte allowance, or ``None`` for CPU.

    Notes
    -----
    Available bytes include reusable caching-allocator memory.
    """
    if device.type != "cuda":
        return None
    return int(available_cuda_bytes(device) * max_memory_fraction)


def _build_mode_indices(
    batch: Batch,
    atom_types: Tensor | None,
    *,
    cutoff: float,
    device: torch.device,
    max_memory_fraction: float,
    cuda_memory_budget_bytes: int | None = None,
    include_hydrogens: bool = True,
) -> dict[str, RadialComparisonIndex]:
    """Build each radial mode from one input tile and shared descriptor stores.

    Parameters
    ----------
    batch : Batch
        Input structures in logical row order.
    atom_types : torch.Tensor or None
        Atom-aligned type IDs for typed modes.
    cutoff : float
        Radial cutoff in angstroms.
    device : torch.device
        Target device for descriptors and summary features.
    max_memory_fraction : float
        Fraction of current PyTorch-available CUDA bytes for this call.
    cuda_memory_budget_bytes : int, optional
        Shared byte allowance, overriding a fresh estimate when provided.
    include_hydrogens : bool
        Hydrogen-selection setting associated with the shared geometry.

    Returns
    -------
    dict
        Indexes keyed by mode name, sharing the input row ordering.

    Raises
    ------
    MemoryError
        If no positive CUDA budget is available for a CUDA build.

    Notes
    -----
    CUDA stores remain device-resident, and their bytes are combined when
    subsequent scoring keeps multiple mode bundles alive.
    """
    budget = (
        _cuda_budget(device, max_memory_fraction)
        if cuda_memory_budget_bytes is None
        else cuda_memory_budget_bytes
    )
    if device.type == "cuda" and budget is not None and budget <= 0:
        raise MemoryError("CUDA descriptor build has no positive memory budget")
    stores = build_descriptor_stores(
        batch,
        cutoff=cutoff,
        atom_types=atom_types,
        modes=_mode_names(atom_types),
        device=device,
        cuda_memory_budget_bytes=budget,
        require_device_residency=device.type == "cuda",
    )
    modes = (
        (("untyped", None, False),)
        if atom_types is None
        else (
            ("untyped", None, False),
            ("center", atom_types, False),
            ("full", atom_types, True),
        )
    )
    return {
        name: RadialComparisonIndex._from_store(
            stores[name],
            cutoff=cutoff,
            atom_types=mode_types,
            typed_neighbors=typed_neighbors,
            device=device,
            max_memory_fraction=max_memory_fraction,
            cuda_memory_budget_bytes=budget,
            include_hydrogens=include_hydrogens,
        )
        for name, mode_types, typed_neighbors in modes
    }


def _resident_descriptor_bytes(
    indexes: dict[str, RadialComparisonIndex],
) -> int:
    """Count descriptor storage retained by all modes in one bundle.

    Parameters
    ----------
    indexes : dict
        Mode name to comparison index mapping.

    Returns
    -------
    int
        Total resident descriptor bytes for all indexes in the mapping.
    """
    return sum(index._resident_descriptor_bytes for index in indexes.values())


def _remaining_cuda_budget(budget: int | None, resident_bytes: int) -> int | None:
    """Subtract active descriptor bytes from a shared CUDA allowance.

    Parameters
    ----------
    budget : int or None
        Shared CUDA byte allowance, or ``None`` for CPU execution.
    resident_bytes : int
        Descriptor bytes already held by active bundles.

    Returns
    -------
    int or None
        Positive remaining bytes, or ``None`` on CPU.

    Raises
    ------
    MemoryError
        If active descriptors leave no positive CUDA capacity.
    """
    if budget is None:
        return None
    remaining = budget - resident_bytes
    if remaining <= 0:
        raise MemoryError(
            f"active descriptors use {resident_bytes} bytes of the {budget}-byte "
            "CUDA budget; no capacity remains for another descriptor tile"
        )
    return remaining


def _configure_active_bundles(
    bundles: Sequence[dict[str, RadialComparisonIndex]], budget: int | None
) -> None:
    """Set each live index's workspace after summing resident bundles.

    Parameters
    ----------
    bundles : sequence of dict
        Active mode-index bundles sharing a tile's lifetime.
    budget : int or None
        Shared CUDA byte allowance, or ``None`` on CPU.

    Notes
    -----
    Simultaneously live left, right, and mode descriptors cannot each claim the
    full call allowance independently.
    """
    active_bytes = sum(_resident_descriptor_bytes(bundle) for bundle in bundles)
    for bundle in bundles:
        for index in bundle.values():
            index._set_active_memory_allowance(budget, active_bytes)


def _screen_prebuilt_pairs(
    left_indexes: dict[str, RadialComparisonIndex],
    pairs: Tensor,
    threshold: float,
    *,
    right_indexes: dict[str, RadialComparisonIndex] | None = None,
    stats: dict[str, Any] | None = None,
    descriptor_structures: int = 0,
) -> Tensor:
    """Run ordered pairs through the available untyped-to-typed screens.

    Parameters
    ----------
    left_indexes : dict
        Left mode name to comparison index mapping.
    pairs : torch.Tensor, shape (K, 2)
        Logical row pairs in caller order.
    threshold : float
        Inclusive fractional score limit.
    right_indexes : dict, optional
        Right indexes for cross-pool screening; defaults to the left bundle.
    stats : dict, optional
        Counters updated with pair and descriptor work.
    descriptor_structures : int
        Number of descriptors held for this screening operation.

    Returns
    -------
    torch.Tensor, shape (M, 2), dtype int32
        Surviving logical pairs in input order.

    Notes
    -----
    Each mode only removes pairs; survivors preserve order through the screen
    cascade.
    """
    if pairs.numel() == 0:
        return pairs
    right_indexes = left_indexes if right_indexes is None else right_indexes
    for mode in left_indexes:
        screen_name = {
            "untyped": "untyped",
            "center": "center_typed",
            "full": "fully_typed",
        }[mode]
        if stats is not None:
            stats["screen_pair_count"] += pairs.shape[0]
            stats[f"screen_pairs_{screen_name}"] += pairs.shape[0]
            stats[f"descriptor_structures_{screen_name}"] += descriptor_structures
        score_start = time.perf_counter()
        pairs = left_indexes[mode].find_matches(
            right_indexes[mode],
            threshold=threshold,
            pair_indices=pairs,
            pair_chunk_size=max(1, pairs.shape[0]),
        )
        if stats is not None:
            stats["screen_seconds"] += time.perf_counter() - score_start
            stats["scored_pair_count"] += left_indexes[mode]._last_search_stats[
                "pairs_scored"
            ]
        if not pairs.numel():
            break
    return pairs


def _build_summaries(
    ids: Sequence[int],
    loader: Callable[[Tensor], tuple[Batch, Tensor | None]],
    vocabulary: tuple[int, ...] | None,
    cutoff: float,
    device: torch.device,
    chunk_size: int,
    *,
    max_batch_atoms: int = _MAX_BATCH_ATOMS,
    max_memory_fraction: float = 0.85,
    return_atom_counts: bool = False,
    include_hydrogens: bool = True,
) -> tuple[Tensor, Tensor, int] | tuple[Tensor, Tensor, Tensor, int]:
    """Build pool summaries in atom-bounded tiles with recursive OOM retries.

    Parameters
    ----------
    ids : sequence of int
        Original logical row IDs to summarize, in output order.
    loader : callable
        Loader for bounded structure batches and optional atom types.
    vocabulary : tuple of int or None
        Fixed type vocabulary, required when loaded rows include atom types.
    cutoff : float
        Radial cutoff in angstroms.
    device : torch.device
        Device for compact summaries and feature search.
    chunk_size : int
        Maximum rows requested in one loader call.
    max_batch_atoms : int
        Maximum atom count allowed in one descriptor tile.
    max_memory_fraction : float
        Estimated CUDA fraction assigned to builds and workspace.
    return_atom_counts : bool
        Whether to return a CPU int32 atom-count vector as well.
    include_hydrogens : bool
        Whether hydrogens remain in summaries and atom counts.

    Returns
    -------
    tuple
        FP32 feature rows, bool presence rows, optional atom counts, and the
        descriptor rebuild count. Row order matches ``ids``.

    Raises
    ------
    MemoryError
        If one structure exceeds the atom limit or its singleton descriptor
        build cannot fit the estimated CUDA capacity.

    Notes
    -----
    CUDA capacity failures split a tile after failed exception state is
    released, avoiding stale traceback tensors during recursive retries. Only
    the summary layout is built here; scoring typed candidate pairs later uses
    the U/C/F cascade after the conservative shortlist.
    """
    summary_width = (
        len(vocabulary) * len(vocabulary) * _TYPED_RANK_COUNT * 2
        if vocabulary is not None
        else 16
    )
    presence_width = (
        len(vocabulary) if vocabulary is not None else (summary_width + 13) // 14
    )
    if not ids:
        empty_features = torch.empty((0, summary_width), device=device)
        empty_presence = torch.empty(
            (0, presence_width), dtype=torch.bool, device=device
        )
        if return_atom_counts:
            return (
                empty_features,
                empty_presence,
                torch.empty((0,), dtype=torch.int32),
                0,
            )
        return empty_features, empty_presence, 0
    feature_parts: list[Tensor] = []
    presence_parts: list[Tensor] = []
    atom_count_parts: list[Tensor] = []
    rebuilds = 0
    for start in range(0, len(ids), chunk_size):
        block_ids = ids[start : start + chunk_size]
        batch, types = _read_batch(
            loader,
            block_ids,
            vocabulary,
            include_hydrogens=include_hydrogens,
        )
        ptr = batch.batch_ptr.detach().cpu().tolist()
        local_counts = [ptr[i + 1] - ptr[i] for i in range(len(block_ids))]

        def build_tile(tile_start: int, tile_stop: int) -> None:
            """Build a descriptor tile, splitting only on capacity failures.

            Parameters
            ----------
            tile_start, tile_stop : int
                Half-open positions in the current loader block.

            Raises
            ------
            MemoryError
                If one structure's descriptor build exceeds CUDA capacity.

            Notes
            -----
            Failure text is retained without its traceback; successful
            summaries are appended in original ``ids`` order.
            """
            nonlocal rebuilds
            selected = list(range(tile_start, tile_stop))
            if tile_start == 0 and tile_stop == len(block_ids):
                tile_batch = batch
                tile_types = types
            else:
                tile_batch = batch.index_select(selected)
                if types is None:
                    tile_types = None
                else:
                    type_start, type_stop = ptr[tile_start], ptr[tile_stop]
                    tile_types = types[type_start:type_stop].contiguous()
            failure_message: str | None = None
            try:
                # Summary-only builds need one layout; the full U/C/F cascade
                # is built later only for candidate pairs that reach scoring.
                summary_mode = "untyped" if tile_types is None else "full"
                stores = build_descriptor_stores(
                    tile_batch,
                    cutoff=cutoff,
                    atom_types=tile_types,
                    modes=(summary_mode,),
                    device=device,
                    cuda_memory_budget_bytes=_cuda_budget(device, max_memory_fraction),
                    require_device_residency=device.type == "cuda",
                )
            except (MemoryError, torch.cuda.OutOfMemoryError) as exc:
                # Keep only text while leaving the handler. The exception
                # traceback can hold descriptor-builder tensors alive; clear
                # it before a recursive tile retry allocates another attempt.
                failure_message = str(exc)
            if failure_message is not None:
                del tile_batch, tile_types
                if tile_stop - tile_start <= 1:
                    raise MemoryError(
                        f"structure {block_ids[tile_start]} with "
                        f"{local_counts[tile_start]} atoms exceeds the configured "
                        f"descriptor memory capacity: {failure_message}"
                    )
                midpoint = tile_start + (tile_stop - tile_start) // 2
                build_tile(tile_start, midpoint)
                build_tile(midpoint, tile_stop)
                return
            summary_store = stores[summary_mode]
            if tile_types is None:
                block_features = summary_store.summaries.detach().to(
                    device=device, dtype=torch.float32
                )
                block_presence = torch.ones(
                    (len(selected), presence_width), dtype=torch.bool, device=device
                )
            else:
                block_features, block_presence = _canonical_typed_summary(
                    summary_store, vocabulary, cutoff, device=device
                )
            feature_parts.append(block_features)
            presence_parts.append(block_presence)
            atom_count_parts.append(summary_store.atom_counts.detach().cpu())
            rebuilds += 1
            del stores, tile_batch, tile_types

        tile_start = 0
        while tile_start < len(block_ids):
            tile_stop = tile_start
            node_count = 0
            while tile_stop < len(block_ids):
                next_count = local_counts[tile_stop]
                if next_count > max_batch_atoms:
                    raise MemoryError(
                        f"structure {block_ids[tile_stop]} requires {next_count} "
                        "atoms, above max_batch_atoms capacity "
                        f"{max_batch_atoms}"
                    )
                if tile_stop > tile_start and node_count + next_count > max_batch_atoms:
                    break
                node_count += next_count
                tile_stop += 1
            build_tile(tile_start, tile_stop)
            tile_start = tile_stop
        del batch, types
    features = torch.cat(feature_parts)
    presence = torch.cat(presence_parts)
    if return_atom_counts:
        return features, presence, torch.cat(atom_count_parts), rebuilds
    return features, presence, rebuilds


def _select_coordinates(
    features: Tensor,
    presence: Tensor,
    threshold: float,
    vocabulary_size: int,
    summary_coordinate_count: int = _DEFAULT_SUMMARY_COORDINATE_COUNT,
) -> list[int]:
    """Greedily select summary coordinates using training query pairs.

    Parameters
    ----------
    features : torch.Tensor, shape (N, F)
        Pilot log-distance summary features.
    presence : torch.Tensor, shape (N, T), dtype bool
        Center-type presence for typed features, or the untyped presence mask.
    threshold : float
        Inclusive fractional mismatch threshold for conservative bounds.
    vocabulary_size : int
        Type-axis width used to map flattened coordinates to center types.
    summary_coordinate_count : int
        Maximum number of coordinates to select, from zero through the cap.

    Returns
    -------
    list of int
        Selected feature-column IDs in greedy selection order.

    Raises
    ------
    ValueError
        If the requested coordinate count is outside its supported range.

    Notes
    -----
    Training pairs choose coordinates by rejection count; held-out rows are
    excluded. Unused coordinates fill the requested width if filters are weak.
    """
    summary_coordinate_count = _nonnegative_int(
        summary_coordinate_count, "summary_coordinate_count"
    )
    if summary_coordinate_count > _MAX_SUMMARY_COORDINATE_COUNT:
        raise ValueError(
            f"summary_coordinate_count must be at most {_MAX_SUMMARY_COORDINATE_COUNT}"
        )
    limit = min(summary_coordinate_count, features.shape[1])
    if not limit:
        return []
    refs, training_count, _ = _pilot_role_counts(features.shape[0])
    train_stop = refs + training_count
    if refs == 0 or training_count == 0:
        return list(range(limit))
    bound = _conservative_summary_log_bound(threshold)
    if bound is None:
        return list(range(limit))

    q = features[refs:train_stop]
    r = features[:refs]
    q_presence = presence[refs:train_stop]
    r_presence = presence[:refs]
    shared = q_presence[:, None, :] & r_presence[None, :, :]
    pair_live = ~(q_presence[:, None, :] != r_presence[None, :, :]).any(dim=2)
    feature_count = features.shape[1]
    center_by_coordinate = torch.arange(feature_count, device=features.device) // (
        vocabulary_size * 14
    )
    rejected = torch.empty(
        (q.shape[0], r.shape[0], feature_count),
        dtype=torch.bool,
        device=features.device,
    )
    for start in range(0, feature_count, 512):
        stop = min(start + 512, feature_count)
        centers = center_by_coordinate[start:stop]
        differences = (q[:, None, start:stop] - r[None, :, start:stop]).abs()
        rejected[:, :, start:stop] = shared[:, :, centers] & (differences > bound)

    selected: list[int] = []
    available = torch.ones(feature_count, dtype=torch.bool, device=features.device)
    for _ in range(limit):
        scores = (rejected & pair_live.unsqueeze(2)).sum(dim=(0, 1))
        scores[~available] = -1
        best_rejections = int(scores.max())
        if best_rejections <= 0:
            break
        best = int(torch.argmax(scores))
        selected.append(best)
        pair_live &= ~rejected[:, :, best]
        available[best] = False
    if len(selected) < limit:
        selected_tensor = torch.tensor(
            selected, dtype=torch.int64, device=features.device
        )
        if selected:
            available[selected_tensor] = False
        selected.extend(
            torch.nonzero(available, as_tuple=False)
            .flatten()[: limit - len(selected)]
            .detach()
            .cpu()
            .tolist()
        )
    return selected


def _pilot_heldout_selectivity(
    features: Tensor,
    presence: Tensor,
    labels: list[str],
    selected: list[int],
    threshold: float,
    vocabulary_size: int,
) -> dict[str, dict[str, float | int | None]]:
    """Measure selected-summary rejection on held-out pilot queries.

    Parameters
    ----------
    features : torch.Tensor, shape (N, F)
        Pilot log-distance summary features.
    presence : torch.Tensor, shape (N, T), dtype bool
        Per-row center-type presence.
    labels : list of str
        Pilot-row origin labels used to split contiguous and spread queries.
    selected : list of int
        Summary coordinate IDs used by the preliminary filter.
    threshold : float
        Inclusive fractional mismatch threshold.
    vocabulary_size : int
        Type-axis width for mapping feature columns to center types.

    Returns
    -------
    dict
        Rejection counts and fractions for each held-out sample stratum.

    Notes
    -----
    These diagnostics do not change the conservative candidate set.
    """
    refs, training_count, holdout_count = _pilot_role_counts(features.shape[0])
    start = refs + training_count
    stop = start + holdout_count
    bound = _conservative_summary_log_bound(threshold)
    centers = torch.tensor(
        [coordinate // (vocabulary_size * 14) for coordinate in selected],
        dtype=torch.int64,
        device=features.device,
    )
    coordinates = torch.tensor(selected, dtype=torch.int64, device=features.device)
    result: dict[str, dict[str, float | int | None]] = {}
    for label in ("contiguous", "spread"):
        queries = [row for row in range(start, stop) if labels[row] == label]
        if not queries or not refs:
            result[label] = {"pairs": 0, "rejected": 0, "fraction": None}
            continue
        query_ids = torch.tensor(queries, dtype=torch.int64, device=features.device)
        query_presence = presence[query_ids]
        ref_presence = presence[:refs]
        rejected = (query_presence[:, None] != ref_presence[None, :]).any(dim=2)
        if selected and bound is not None:
            shared = query_presence[:, None, centers] & ref_presence[None, :, centers]
            differences = (
                features[query_ids][:, None, coordinates]
                - features[:refs][None, :, coordinates]
            ).abs()
            rejected |= (shared & (differences > bound)).any(dim=2)
        total = int(rejected.numel())
        count = int(rejected.sum())
        result[label] = {
            "pairs": total,
            "rejected": count,
            "fraction": count / total if total else None,
        }
    return result


def _select_summary_intervals(
    query_features: Tensor,
    applicable: Tensor,
    sorted_values: Tensor,
    outward_bound: Tensor,
    has_applicable: Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Select each query's narrowest applicable outward-rounded interval.

    Parameters
    ----------
    query_features : torch.Tensor, shape (F,) or (Q, F)
        Already selected log-distance summary rows.
    applicable : torch.Tensor, shape (F,) or (Q, F), dtype bool
        Per-coordinate applicability masks for those query rows.
    sorted_values : torch.Tensor, shape (F, N_right)
        Sorted right-pool values for each summary coordinate.
    outward_bound : torch.Tensor, scalar FP32
        Existing FP32 bound after its outward ``nextafter`` step.
    has_applicable : torch.Tensor, shape (Q,), dtype bool, optional
        Precomputed batch mask indicating which query rows have an applicable
        coordinate; batched callers supply this mask. Scalar callers have
        already checked applicability and omit it.

    Returns
    -------
    tuple of torch.Tensor
        Selected coordinate, left endpoint, right endpoint, and interval width.
        Scalar queries return zero-dimensional tensors; batched queries return
        vectors. Inapplicable batch rows have zero width.
    """
    device = query_features.device
    negative_infinity = torch.tensor(float("-inf"), dtype=torch.float32, device=device)
    positive_infinity = torch.tensor(float("inf"), dtype=torch.float32, device=device)
    low = torch.nextafter(
        query_features - outward_bound,
        negative_infinity,
    )
    high = torch.nextafter(
        query_features + outward_bound,
        positive_infinity,
    )
    if query_features.ndim == 1:
        left_all = torch.searchsorted(
            sorted_values, low.unsqueeze(1), right=False
        ).squeeze(1)
        right_all = torch.searchsorted(
            sorted_values, high.unsqueeze(1), right=True
        ).squeeze(1)
        del low, high
        widths = right_all - left_all
        widths = torch.where(
            applicable,
            widths,
            torch.full_like(widths, sorted_values.shape[1] + 1),
        )
        selected_coordinate = torch.argmin(widths)
        left = left_all[selected_coordinate]
        right = right_all[selected_coordinate]
        return (
            selected_coordinate,
            left,
            right,
            right - left,
        )

    left_all = torch.searchsorted(
        sorted_values, low.transpose(0, 1).contiguous(), right=False
    )
    right_all = torch.searchsorted(
        sorted_values, high.transpose(0, 1).contiguous(), right=True
    )
    del low, high
    widths = right_all - left_all
    widths = torch.where(
        applicable.transpose(0, 1),
        widths,
        torch.full_like(widths, sorted_values.shape[1] + 1),
    )
    selected_coordinates = widths.argmin(dim=0)
    query_columns = torch.arange(
        query_features.shape[0], dtype=torch.int64, device=device
    )
    left = left_all[selected_coordinates, query_columns]
    right = right_all[selected_coordinates, query_columns]
    interval_widths = right - left
    if has_applicable is not None:
        interval_widths = torch.where(has_applicable, interval_widths, 0)
    return selected_coordinates, left, right, interval_widths


def _safe_summary_features_match(
    query_features: Tensor,
    representative_features: Tensor,
    active_coordinates: Tensor | None,
    outward_bound: Tensor,
) -> Tensor:
    """Compare gathered safe summary features with inclusive FP32 bounds.

    Parameters
    ----------
    query_features : torch.Tensor, shape (S,) or (K, S)
        One query row broadcast over representatives or aligned query rows.
    representative_features : torch.Tensor, shape (S,) or (K, S)
        Gathered representative features aligned with the query rows.
    active_coordinates : torch.Tensor, shape (S,) or (K, S), dtype bool, optional
        Coordinates that apply to each comparison. ``None`` means all gathered
        coordinates were narrowed to active safe coordinates by the caller.
    outward_bound : torch.Tensor, scalar FP32
        Existing FP32 bound after its outward ``nextafter`` step.

    Returns
    -------
    torch.Tensor
        Scalar bool tensor for one comparison or shape (K,) for aligned rows.
    """
    differences = query_features - representative_features
    del query_features, representative_features
    differences.abs_()
    matches = differences <= outward_bound
    if active_coordinates is not None:
        matches |= ~active_coordinates
    return matches.all(dim=-1)


def _candidate_representatives(
    candidate_rank: int,
    retained: list[int],
    retained_mask: Tensor,
    rank_by_id: Tensor,
    features: Tensor,
    presence: Tensor,
    coordinate_centers: Tensor,
    sorted_values: Tensor,
    sorted_ids: Tensor,
    coordinate_safe: Tensor,
    bound: float | None,
) -> tuple[list[int], int, int, int]:
    """Conservatively shortlist retained rows for one candidate rank.

    Parameters
    ----------
    candidate_rank : int
        Feature-row position for the current candidate.
    retained : list of int
        Retained representative IDs in greedy priority order.
    retained_mask : torch.Tensor, shape (N,), dtype bool
        Mask indexed by logical row ID indicating retained structures.
    rank_by_id : torch.Tensor, shape (N,)
        Map from logical row ID to feature-row rank.
    features : torch.Tensor, shape (N, F)
        Compact log-distance summaries.
    presence : torch.Tensor, shape (N, T), dtype bool
        Per-row center-type presence.
    coordinate_centers : torch.Tensor, shape (F,)
        Center-type index for each flattened feature coordinate.
    sorted_values, sorted_ids : torch.Tensor, shape (F, N)
        Per-coordinate sorted features and matching logical row IDs.
    coordinate_safe : torch.Tensor, shape (F,), dtype bool
        Coordinates safe for conservative interval pruning.
    bound : float or None
        Conservative log-distance difference bound.

    Returns
    -------
    tuple
        Retained logical IDs in priority order and counts for interval rows,
        retained interval rows, and final summary survivors.

    Notes
    -----
    The narrowest safe interval seeds the query; subsequent checks only remove
    proven nonmatches, leaving exact radial scoring to the caller.
    """
    if not retained:
        return [], 0, 0, 0
    feature_count = sorted_values.shape[0]
    if bound is None or not feature_count:
        return retained, 0, len(retained), len(retained)
    candidate_presence = presence[candidate_rank]
    applicable = candidate_presence[coordinate_centers] & coordinate_safe
    if not torch.any(applicable):
        return retained, 0, len(retained), len(retained)

    candidate_features = features[candidate_rank]
    safe_coordinates = torch.nonzero(coordinate_safe, as_tuple=False).flatten()
    range_bound = torch.nextafter(
        torch.tensor(bound, dtype=torch.float32),
        torch.tensor(float("inf"), dtype=torch.float32),
    )
    coordinate_tensor, left_tensor, right_tensor, _ = _select_summary_intervals(
        candidate_features,
        applicable,
        sorted_values,
        range_bound,
    )
    coordinate = int(coordinate_tensor)
    interval_ids = sorted_ids[
        coordinate,
        left_tensor:right_tensor,
    ].to(torch.int64)
    interval_count = interval_ids.numel()
    candidates = interval_ids[retained_mask[interval_ids]]
    retained_interval_count = candidates.numel()
    if not retained_interval_count:
        return [], interval_count, 0, 0

    selected_centers = coordinate_centers[safe_coordinates]
    surviving: list[Tensor] = []
    for start in range(0, retained_interval_count, _FILTER_BLOCK_SIZE):
        block_ids = candidates[start : start + _FILTER_BLOCK_SIZE]
        rows = rank_by_id[block_ids]
        same_presence = (presence[rows] == candidate_presence).all(dim=1)
        block_ids = block_ids[same_presence]
        rows = rows[same_presence]
        if not block_ids.numel():
            continue
        keep = _safe_summary_features_match(
            candidate_features[safe_coordinates],
            features[rows][:, safe_coordinates],
            candidate_presence[selected_centers],
            range_bound,
        )
        if torch.any(keep):
            surviving.append(block_ids[keep])
    if not surviving:
        return [], interval_count, retained_interval_count, 0
    result_ids = torch.cat(surviving)
    priority = rank_by_id[result_ids]
    result_ids = result_ids[torch.argsort(priority, stable=True)]
    return (
        result_ids.tolist(),
        interval_count,
        retained_interval_count,
        result_ids.numel(),
    )


def _candidate_representatives_batch(
    candidate_ids: Sequence[int],
    candidate_ranks: Tensor,
    retained: Sequence[int],
    retained_mask: Tensor,
    rank_by_id: Tensor,
    features: Tensor,
    presence: Tensor,
    coordinate_centers: Tensor,
    sorted_values: Tensor,
    sorted_ids: Tensor,
    coordinate_safe: Tensor,
    bound: float | None,
) -> tuple[list[list[int]], list[int], list[int], list[int]]:
    """Build conservative shortlist lists for a tile of candidates.

    Parameters
    ----------
    candidate_ids : sequence of int
        Logical IDs whose representative candidates are requested.
    candidate_ranks : torch.Tensor, shape (C,)
        Feature-row rank for each candidate ID.
    retained : sequence of int
        Retained logical IDs in greedy priority order.
    retained_mask : torch.Tensor, shape (N,), dtype bool
        Mask indexed by logical row ID indicating retained structures.
    rank_by_id : torch.Tensor, shape (N,)
        Map from logical row ID to feature-row rank.
    features : torch.Tensor, shape (N, F)
        Compact log-distance summaries.
    presence : torch.Tensor, shape (N, T), dtype bool
        Per-row center-type presence.
    coordinate_centers : torch.Tensor, shape (F,)
        Center-type index for each flattened feature coordinate.
    sorted_values, sorted_ids : torch.Tensor, shape (F, N)
        Per-coordinate sorted features and logical row IDs.
    coordinate_safe : torch.Tensor, shape (F,), dtype bool
        Coordinates safe for conservative interval pruning.
    bound : float or None
        Conservative log-distance difference bound.

    Returns
    -------
    tuple
        Per-candidate logical shortlist lists and parallel counts for interval
        rows, retained interval rows, and summary survivors.

    Notes
    -----
    Device-side search is batched; each output list is restored to retained
    priority before exact pair scoring.
    """
    candidate_count = len(candidate_ids)
    output: list[list[int]] = [[] for _ in candidate_ids]
    interval_counts = [0] * candidate_count
    retained_counts = [0] * candidate_count
    full_counts = [0] * candidate_count
    if not candidate_count or not retained:
        return output, interval_counts, retained_counts, full_counts

    device = features.device
    retained_order = {
        logical_id: position for position, logical_id in enumerate(retained)
    }
    if bound is None or not features.shape[1]:
        for position in range(candidate_count):
            output[position] = list(retained)
            retained_counts[position] = len(retained)
            full_counts[position] = len(retained)
        return output, interval_counts, retained_counts, full_counts

    candidate_presence = presence[candidate_ranks]
    applicable = candidate_presence[:, coordinate_centers] & coordinate_safe
    has_applicable = applicable.any(dim=1)
    fallback_positions = (
        torch.nonzero(~has_applicable, as_tuple=False).flatten().detach().cpu().tolist()
    )
    for position in fallback_positions:
        output[position] = list(retained)
        retained_counts[position] = len(retained)
        full_counts[position] = len(retained)

    if not torch.any(has_applicable):
        return output, interval_counts, retained_counts, full_counts

    outward_bound = torch.nextafter(
        torch.tensor(bound, dtype=torch.float32, device=device),
        torch.tensor(float("inf"), dtype=torch.float32, device=device),
    )
    (
        selected_coordinates,
        left,
        right,
        interval_widths,
    ) = _select_summary_intervals(
        features[candidate_ranks],
        applicable,
        sorted_values,
        outward_bound,
        has_applicable=has_applicable,
    )
    interval_counts = interval_widths.detach().cpu().tolist()

    # Keep interval expansion and its summary-filter workspace bounded while
    # amortizing device-to-host transfers across many candidates.
    query_tile_size = 128
    pair_limit = _summary_pair_filter_block_size(
        presence.shape[1], int(coordinate_safe.sum())
    )
    for query_start in range(0, candidate_count, query_tile_size):
        query_stop = min(candidate_count, query_start + query_tile_size)
        query_positions = torch.arange(query_start, query_stop, device=device)
        active_mask = has_applicable[query_positions]
        query_positions = query_positions[active_mask]
        if not query_positions.numel():
            continue
        max_width = int(interval_widths[query_positions].max())
        if max_width == 0:
            continue
        rows_per_block = max(1, pair_limit // query_positions.numel())
        retained_count_by_query = torch.zeros(
            (candidate_count,), dtype=torch.int32, device=device
        )
        matched_count_by_query = torch.zeros_like(retained_count_by_query)
        for start in range(0, max_width, rows_per_block):
            stop = min(max_width, start + rows_per_block)
            positions = (
                left[query_positions, None]
                + torch.arange(start, stop, dtype=torch.int64, device=device)[None, :]
            )
            in_interval = positions < right[query_positions, None]
            bounded_positions = positions.clamp_(min=0, max=sorted_ids.shape[1] - 1)
            representatives = sorted_ids[
                selected_coordinates[query_positions, None], bounded_positions
            ].to(torch.int64)
            in_retained_interval = in_interval & retained_mask[representatives]
            query_rows, rep_columns = torch.nonzero(in_retained_interval, as_tuple=True)
            if not query_rows.numel():
                continue
            local_queries = query_positions[query_rows]
            representative_ids = representatives[query_rows, rep_columns]
            retained_count_by_query.scatter_add_(
                0, local_queries, torch.ones_like(local_queries, dtype=torch.int32)
            )
            pair_ranks = torch.stack(
                (candidate_ranks[local_queries], rank_by_id[representative_ids]), dim=1
            )
            keep = _summary_pairs_may_match_batch(
                pair_ranks,
                features,
                presence,
                coordinate_centers,
                coordinate_safe,
                bound,
            )
            matched_queries = local_queries[keep]
            matched_ids = representative_ids[keep]
            matched_count_by_query.scatter_add_(
                0,
                matched_queries,
                torch.ones_like(matched_queries, dtype=torch.int32),
            )
            matched_pairs = torch.stack(
                (
                    matched_queries.to(torch.int32),
                    matched_ids.to(torch.int32),
                ),
                dim=1,
            )
            for candidate_position, representative_id in (
                matched_pairs.detach().cpu().tolist()
            ):
                output[candidate_position].append(representative_id)

        retained_count_rows = retained_count_by_query.detach().cpu().tolist()
        matched_count_rows = matched_count_by_query.detach().cpu().tolist()
        for query_position in query_positions.detach().cpu().tolist():
            retained_counts[query_position] = retained_count_rows[query_position]
            full_counts[query_position] = matched_count_rows[query_position]

    for candidate_position, candidates in enumerate(output):
        if candidates and candidate_position not in fallback_positions:
            candidates.sort(key=retained_order.__getitem__)
    return output, interval_counts, retained_counts, full_counts


def _summary_pair_may_match(
    candidate_rank: int,
    representative_rank: int,
    features: Tensor,
    presence: Tensor,
    coordinate_centers: Tensor,
    coordinate_safe: Tensor,
    bound: float | None,
) -> bool:
    """Test whether one ranked pair survives all safe summary coordinates.

    Parameters
    ----------
    candidate_rank, representative_rank : int
        Feature-row ranks for the ordered pair.
    features : torch.Tensor, shape (N, F)
        Compact log-distance summaries.
    presence : torch.Tensor, shape (N, T), dtype bool
        Center-type presence by feature row.
    coordinate_centers : torch.Tensor, shape (F,)
        Center-type index for each summary coordinate.
    coordinate_safe : torch.Tensor, shape (F,), dtype bool
        Coordinates allowed to reject a pair conservatively.
    bound : float or None
        Conservative log-distance difference bound.

    Returns
    -------
    bool
        Whether the pair remains a possible radial match.

    Notes
    -----
    A true result is not a match decision; descriptor scoring is still required.
    """
    if bound is None or not features.shape[1]:
        return True
    candidate_presence = presence[candidate_rank]
    safe_coordinates = torch.nonzero(coordinate_safe, as_tuple=False).flatten()
    safe_centers = coordinate_centers[safe_coordinates]
    active = candidate_presence[safe_centers]
    if not torch.any(active):
        return True
    if not torch.equal(candidate_presence, presence[representative_rank]):
        return False
    compared_coordinates = safe_coordinates[active]
    outward_bound = torch.nextafter(
        torch.tensor(bound, dtype=torch.float32, device=features.device),
        torch.tensor(float("inf"), dtype=torch.float32, device=features.device),
    )
    return bool(
        _safe_summary_features_match(
            features[candidate_rank, compared_coordinates],
            features[representative_rank, compared_coordinates],
            None,
            outward_bound,
        )
    )


def _summary_pairs_may_match_batch(
    pair_ranks: Tensor,
    features: Tensor,
    presence: Tensor,
    coordinate_centers: Tensor,
    coordinate_safe: Tensor,
    bound: float | None,
) -> Tensor:
    """Apply the safe summary filter to a bounded ordered rank-pair tensor.

    Parameters
    ----------
    pair_ranks : torch.Tensor, shape (K, 2)
        Candidate and representative feature-row ranks.
    features : torch.Tensor, shape (N, F)
        Compact log-distance summaries.
    presence : torch.Tensor, shape (N, T), dtype bool
        Center-type presence by feature row.
    coordinate_centers : torch.Tensor, shape (F,)
        Center-type index for each summary coordinate.
    coordinate_safe : torch.Tensor, shape (F,), dtype bool
        Coordinates allowed to reject pairs conservatively.
    bound : float or None
        Conservative log-distance difference bound.

    Returns
    -------
    torch.Tensor, shape (K,), dtype bool
        True for pairs that remain possible, in input order.

    Notes
    -----
    Surviving pairs still require exact radial scoring.
    """
    pair_count = pair_ranks.shape[0]
    if not pair_count:
        return torch.empty((0,), dtype=torch.bool, device=features.device)
    if bound is None or not features.shape[1]:
        return torch.ones((pair_count,), dtype=torch.bool, device=features.device)

    safe_coordinates = torch.nonzero(coordinate_safe, as_tuple=False).flatten()
    safe_centers = coordinate_centers[safe_coordinates]
    candidate_ranks = pair_ranks[:, 0]
    representative_ranks = pair_ranks[:, 1]
    active_coordinates = presence[candidate_ranks][:, safe_centers]
    active_pairs = torch.nonzero(
        active_coordinates.any(dim=1), as_tuple=False
    ).flatten()
    keep = torch.ones((pair_count,), dtype=torch.bool, device=features.device)
    if not active_pairs.numel():
        return keep

    keep[active_pairs] = False
    active_candidates = candidate_ranks[active_pairs]
    active_representatives = representative_ranks[active_pairs]
    same_presence = (
        presence[active_candidates] == presence[active_representatives]
    ).all(dim=1)
    matching_presence_pairs = torch.nonzero(same_presence, as_tuple=False).flatten()
    if not matching_presence_pairs.numel():
        return keep

    matched_pair_indices = active_pairs[matching_presence_pairs]
    matched_candidates = candidate_ranks[matched_pair_indices]
    matched_representatives = representative_ranks[matched_pair_indices]
    active_feature_coordinates = active_coordinates[matched_pair_indices]
    outward_bound = torch.nextafter(
        torch.tensor(bound, dtype=torch.float32, device=features.device),
        torch.tensor(float("inf"), dtype=torch.float32, device=features.device),
    )
    keep[matched_pair_indices] = _safe_summary_features_match(
        features[matched_candidates][:, safe_coordinates],
        features[matched_representatives][:, safe_coordinates],
        active_feature_coordinates,
        outward_bound,
    )
    return keep


def _summary_pair_filter_block_size(
    presence_width: int, safe_coordinate_count: int
) -> int:
    """Choose a pair tile fitting the fixed summary-filter workspace estimate.

    Parameters
    ----------
    presence_width : int
        Center-type presence width per structure.
    safe_coordinate_count : int
        Number of selected coordinates that can participate in filtering.

    Returns
    -------
    int
        Pair count bounded by the estimated temporary workspace and hard cap.

    Notes
    -----
    The estimate grows with both input widths, not with the total pool size.
    """
    bytes_per_pair = 64 + 3 * presence_width + 16 * safe_coordinate_count
    return max(
        1,
        min(
            _SUMMARY_PAIR_FILTER_MAX_PAIRS,
            _SUMMARY_PAIR_FILTER_WORKSPACE_BYTES // bytes_per_pair,
        ),
    )


def _within_chunk_candidate_pairs(
    unresolved: Sequence[int],
    rank_by_id: Tensor,
    features: Tensor,
    presence: Tensor,
    coordinate_centers: Tensor,
    coordinate_safe: Tensor,
    bound: float | None,
) -> list[tuple[int, int]]:
    """Enumerate possible within-tile pairs in candidate-major order.

    Parameters
    ----------
    unresolved : sequence of int
        Logical candidate IDs in current greedy priority order.
    rank_by_id : torch.Tensor, shape (N,)
        Map from logical row ID to feature-row rank.
    features : torch.Tensor, shape (N, F)
        Compact log-distance summaries.
    presence : torch.Tensor, shape (N, T), dtype bool
        Center-type presence by feature row.
    coordinate_centers : torch.Tensor, shape (F,)
        Center-type index for each summary coordinate.
    coordinate_safe : torch.Tensor, shape (F,), dtype bool
        Coordinates allowed to reject pairs conservatively.
    bound : float or None
        Conservative log-distance difference bound.

    Returns
    -------
    list of tuple of int
        Radial shortlist proposals in candidate-major and earlier-major order.

    Notes
    -----
    Each later candidate is compared only with earlier unresolved positions.
    Replay restricts confirmation and assignment to rows actually retained;
    speculative radial edges cannot establish representatives.
    """
    candidate_count = len(unresolved)
    if candidate_count < 2:
        return []

    logical_ids = torch.tensor(unresolved, dtype=torch.int64, device=features.device)
    ranks = rank_by_id[logical_ids]
    safe_coordinate_count = int(coordinate_safe.sum())
    pair_block_size = _summary_pair_filter_block_size(
        presence.shape[1], safe_coordinate_count
    )
    matches: list[tuple[int, int]] = []

    def append_matching_pairs(
        candidate_positions: Tensor, earlier_positions: Tensor
    ) -> None:
        """Append summary survivors in the pair positions' input order.

        Parameters
        ----------
        candidate_positions, earlier_positions : torch.Tensor, shape (K,)
            Aligned feature-row positions defining ordered local pairs.

        Notes
        -----
        The closure appends logical row IDs to its candidate-major result list.
        """
        pair_ranks = torch.stack(
            (
                ranks[candidate_positions],
                ranks[earlier_positions],
            ),
            dim=1,
        )
        keep = _summary_pairs_may_match_batch(
            pair_ranks,
            features,
            presence,
            coordinate_centers,
            coordinate_safe,
            bound,
        )
        if not torch.any(keep):
            return
        pair_ids = torch.stack(
            (
                logical_ids[candidate_positions],
                logical_ids[earlier_positions],
            ),
            dim=1,
        )
        matches.extend(
            (int(candidate), int(earlier))
            for candidate, earlier in pair_ids[keep].tolist()
        )

    if candidate_count <= pair_block_size:
        # Row-major nonzero order is the same candidate-major, earlier-major
        # order used by the original nested Python loops.
        rows_per_block = max(1, pair_block_size // candidate_count)
        all_columns = torch.arange(
            candidate_count, dtype=torch.int64, device=features.device
        )
        for row_start in range(1, candidate_count, rows_per_block):
            row_stop = min(candidate_count, row_start + rows_per_block)
            candidate_positions = torch.arange(
                row_start, row_stop, dtype=torch.int64, device=features.device
            )
            row_positions, earlier_positions = torch.nonzero(
                all_columns.unsqueeze(0) < candidate_positions.unsqueeze(1),
                as_tuple=True,
            )
            append_matching_pairs(candidate_positions[row_positions], earlier_positions)
    else:
        # A user may choose an input batch larger than the pair workspace. Keep
        # each candidate's earlier rows ordered while bounding that row's pair
        # filter workspace as well.
        for candidate_position in range(1, candidate_count):
            for earlier_start in range(0, candidate_position, pair_block_size):
                earlier_stop = min(candidate_position, earlier_start + pair_block_size)
                earlier_positions = torch.arange(
                    earlier_start,
                    earlier_stop,
                    dtype=torch.int64,
                    device=features.device,
                )
                candidate_positions = torch.full_like(
                    earlier_positions, candidate_position
                )
                append_matching_pairs(candidate_positions, earlier_positions)
    return matches


def _confirm_pair_chunk(
    pairs: list[tuple[int, int]],
    confirm: Callable[[Tensor], Tensor] | None,
    device: torch.device,
) -> list[tuple[int, int]]:
    """Apply ``confirm`` to one bounded batch of radial proposals.

    Parameters
    ----------
    pairs : list of tuple of int
        Logical ``(candidate, retained representative)`` proposals in priority
        order.
    confirm : callable or None
        Callback returning an ordered tensor subset of the supplied proposals.
    device : torch.device
        Comparison device for the int32 callback tensor and result.

    Returns
    -------
    list of tuple of int
        Confirmed pairs in proposal order. With no callback, returns ``pairs``.

    Raises
    ------
    ValueError
        If a callback result is not an ordered subset on the comparison device.
    Exception
        Any exception raised by caller-supplied callback code propagates.

    Notes
    -----
    A pre-callback snapshot protects subset validation from input mutation;
    empty proposals never invoke user code.
    """
    if not pairs or confirm is None:
        return pairs
    proposals = torch.tensor(pairs, dtype=torch.int32, device=device)
    validation_source = proposals.clone()
    confirmed = confirm(proposals)
    _validate_confirmed_subset(validation_source, confirmed, device)
    return [tuple(pair) for pair in confirmed.detach().cpu().tolist()]


def _screen_candidate(
    candidate: int,
    representatives: list[int],
    loader: Callable[[Tensor], tuple[Batch, Tensor | None]],
    vocabulary: tuple[int, ...] | None,
    cutoff: float,
    threshold: float,
    device: torch.device,
    block_size: int,
    *,
    atom_counts: Tensor | None = None,
    max_batch_atoms: int = _MAX_BATCH_ATOMS,
    max_memory_fraction: float = 0.85,
    include_hydrogens: bool = True,
    confirm: Callable[[Tensor], Tensor] | None = None,
) -> tuple[int | None, dict[str, float | int]]:
    """Screen one candidate against ordered retained representatives.

    Parameters
    ----------
    candidate : int
        Candidate's logical row ID.
    representatives : list of int
        Previously retained IDs in greedy priority order.
    loader : callable
        Reader for bounded structure batches and optional atom types.
    vocabulary : tuple of int or None
        Fixed type vocabulary for typed loader rows.
    cutoff : float
        Radial cutoff in angstroms.
    threshold : float
        Inclusive fractional mismatch limit.
    device : torch.device
        Descriptor construction and comparison device.
    block_size : int
        Maximum representatives in one active screen block.
    atom_counts : torch.Tensor, optional
        CPU counts indexed by original logical row ID.
    max_batch_atoms : int
        Maximum atom count in one active descriptor tile.
    max_memory_fraction : float
        Fraction of PyTorch-available CUDA memory assigned to the operation.
    include_hydrogens : bool
        Whether hydrogen atoms remain in the comparison geometry.
    confirm : callable, optional
        Callback that accepts an ordered subset of radial proposals.

    Returns
    -------
    tuple
        First confirmed representative ID, or ``None``, and local screen stats.

    Raises
    ------
    MemoryError
        If a singleton structure or pair cannot fit the atom or memory limit.
    Exception
        Loader and confirmation callback exceptions propagate unchanged.

    Notes
    -----
    Representative blocks preserve priority; a rejected proposal continues to
    later representatives. Confirmation runs outside capacity retries.
    """
    stats: dict[str, float | int] = {
        "descriptor_rebuilds": 0,
        "descriptor_structures_built": 0,
        "descriptor_rebuild_seconds": 0.0,
        "screen_seconds": 0.0,
        "screen_pair_count": 0,
        "screened_pair_candidates": 0,
        "scored_pair_count": 0,
        "screen_pairs_untyped": 0,
        "screen_pairs_center_typed": 0,
        "screen_pairs_fully_typed": 0,
        "descriptor_structures_untyped": 0,
        "descriptor_structures_center_typed": 0,
        "descriptor_structures_fully_typed": 0,
    }
    candidate_atoms = None if atom_counts is None else int(atom_counts[candidate])
    if candidate_atoms is not None and candidate_atoms > max_batch_atoms:
        raise MemoryError(
            f"structure {candidate} requires {candidate_atoms} atoms, above "
            f"max_batch_atoms capacity {max_batch_atoms}"
        )
    start = 0
    while start < len(representatives):
        block: list[int] = []
        representative_atoms_total = 0
        while start < len(representatives) and len(block) < block_size:
            representative = representatives[start]
            representative_atoms = (
                0 if atom_counts is None else int(atom_counts[representative])
            )
            if representative_atoms > max_batch_atoms:
                raise MemoryError(
                    f"structure {representative} requires {representative_atoms} "
                    f"atoms, above max_batch_atoms capacity {max_batch_atoms}"
                )
            if (
                block
                and representative_atoms_total + representative_atoms > max_batch_atoms
            ):
                break
            block.append(representative)
            representative_atoms_total += representative_atoms
            start += 1
        logical_pairs = [(candidate, representative) for representative in block]
        stats["screened_pair_candidates"] += len(logical_pairs)
        screen_start = time.perf_counter()
        matching = _screen_pair_group(
            logical_pairs,
            loader,
            vocabulary,
            cutoff,
            threshold,
            device,
            stats,
            atom_counts=atom_counts,
            max_batch_atoms=max_batch_atoms,
            max_memory_fraction=max_memory_fraction,
            include_hydrogens=include_hydrogens,
        )
        stats["screen_seconds"] += time.perf_counter() - screen_start
        if matching:
            confirmed = _confirm_pair_chunk(matching, confirm, device)
            if confirmed:
                matched_representatives = {
                    representative for _, representative in confirmed
                }
                return next(
                    representative
                    for representative in block
                    if representative in matched_representatives
                ), stats
    return None, stats


def _screen_candidates_against_representatives(
    candidates: Sequence[int],
    shortlists: dict[int, Tensor],
    representatives: Sequence[int],
    loader: Callable[[Tensor], tuple[Batch, Tensor | None]],
    vocabulary: tuple[int, ...] | None,
    cutoff: float,
    threshold: float,
    device: torch.device,
    stats: dict[str, Any],
    *,
    atom_counts: Tensor,
    pair_block_size: int,
    max_batch_atoms: int,
    max_memory_fraction: float,
    include_hydrogens: bool = True,
    confirm: Callable[[Tensor], Tensor] | None = None,
) -> dict[int, int]:
    """Compare candidate tiles with retained blocks under one shared budget.

    Parameters
    ----------
    candidates : sequence of int
        Candidate logical IDs, in greedy priority order.
    shortlists : dict
        Candidate ID to conservative retained-ID tensor mapping.
    representatives : sequence of int
        Retained logical IDs in priority order.
    loader : callable
        Reader for candidate and representative structure batches.
    vocabulary : tuple of int or None
        Fixed type vocabulary for typed rows.
    cutoff : float
        Radial cutoff in angstroms.
    threshold : float
        Inclusive fractional mismatch limit.
    device : torch.device
        Descriptor construction and scoring device.
    stats : dict
        Aggregate diagnostic counters updated during grouped work.
    atom_counts : torch.Tensor
        CPU atom counts indexed by original logical row ID.
    pair_block_size : int
        Maximum representatives built for one scoring block.
    max_batch_atoms : int
        Maximum atoms in each candidate or representative descriptor tile.
    max_memory_fraction : float
        Fraction of PyTorch-available CUDA bytes assigned to the operation.
    include_hydrogens : bool
        Whether hydrogen atoms remain in the comparison geometry.
    confirm : callable, optional
        Callback filtering radial proposals against already-retained rows.

    Returns
    -------
    dict
        Candidate logical ID to its earliest confirmed old representative.

    Raises
    ------
    MemoryError
        If singleton descriptors or a singleton pair cannot fit the limits.
    Exception
        Loader and callback failures propagate without capacity retries.

    Notes
    -----
    Candidate descriptors stay live while representative blocks are built, so
    workspace is shared across both. Candidate and representative order define
    greedy priority.
    """
    if not candidates or not representatives:
        return {}

    # Candidate tiles come from one input chunk, so their row count is bounded
    # by input_batch_size. Candidate and representative descriptor tiles each
    # have their own atom cap; their shared CUDA budget below accounts for both
    # tiles being live. pair_block_size applies only to representative blocks.
    candidate_limit = max_batch_atoms
    candidate_tiles: list[list[int]] = []
    current: list[int] = []
    current_atoms = 0
    for candidate in candidates:
        row_atoms = int(atom_counts[candidate])
        if row_atoms > max_batch_atoms:
            raise MemoryError(
                f"structure {candidate} requires {row_atoms} atoms, above "
                f"max_batch_atoms capacity {max_batch_atoms}"
            )
        if current and current_atoms + row_atoms > candidate_limit:
            candidate_tiles.append(current)
            current, current_atoms = [], 0
        current.append(candidate)
        current_atoms += row_atoms
        if current_atoms >= candidate_limit or row_atoms > candidate_limit:
            candidate_tiles.append(current)
            current, current_atoms = [], 0
    if current:
        candidate_tiles.append(current)

    matches: dict[int, int] = {}
    pair_limit = 8192 if device.type == "cuda" else 4096

    def process_tile(tile: list[int]) -> dict[int, int]:
        """Build and screen a candidate tile, splitting on capacity failures.

        Parameters
        ----------
        tile : list of int
            Candidate logical IDs in greedy priority order.

        Returns
        -------
        dict
            Candidate ID to earliest confirmed retained representative.

        Raises
        ------
        MemoryError
            If a candidate or comparison pair cannot fit as a singleton.

        Notes
        -----
        The enclosing call releases candidate descriptors before a retry; user
        loader and confirmation exceptions are not reinterpreted as capacity.
        """
        if not tile:
            return {}
        tile_atoms = sum(int(atom_counts[row]) for row in tile)
        if tile_atoms > max_batch_atoms:
            if len(tile) == 1:
                row = tile[0]
                raise MemoryError(
                    f"structure {row} requires {tile_atoms} atoms, above "
                    f"max_batch_atoms capacity {max_batch_atoms}"
                )
            midpoint = len(tile) // 2
            return process_tile(tile[:midpoint]) | process_tile(tile[midpoint:])

        build_start = time.perf_counter()
        candidate_batch: Batch | None = None
        candidate_types: Tensor | None = None
        candidate_indexes: dict[str, RadialComparisonIndex] | None = None
        split_tile = False
        output: dict[int, int] = {}
        try:
            candidate_batch, candidate_types = _read_batch(
                loader, tile, vocabulary, include_hydrogens=include_hydrogens
            )
            budget = _cuda_budget(device, max_memory_fraction)
            try:
                candidate_indexes = _build_mode_indices(
                    candidate_batch,
                    candidate_types,
                    cutoff=cutoff,
                    device=device,
                    max_memory_fraction=max_memory_fraction,
                    cuda_memory_budget_bytes=budget,
                    include_hydrogens=include_hydrogens,
                )
            except (MemoryError, torch.cuda.OutOfMemoryError) as exc:
                if len(tile) == 1:
                    row = tile[0]
                    required = int(atom_counts[row])
                    raise MemoryError(
                        f"structure {row} with {required} atoms exceeds descriptor "
                        f"memory capacity: {exc}"
                    ) from exc
                split_tile = True
                raise _RetryActiveTile from exc
            stats["descriptor_rebuild_seconds"] += time.perf_counter() - build_start
            stats["descriptor_rebuilds"] += 1
            stats["descriptor_structures_built"] += len(tile)
            unresolved = list(tile)
            shortlist_sets = {
                candidate: set(shortlists[candidate].tolist()) for candidate in tile
            }
            relevant_representatives = set().union(
                *(shortlist_sets[row] for row in unresolved)
            )
            rep_order = [
                representative
                for representative in representatives
                if representative in relevant_representatives
            ]
            rep_rank = {row: rank for rank, row in enumerate(rep_order)}
            cursor = 0
            while cursor < len(rep_order) and unresolved:
                first_rep = rep_order[cursor]
                first_rep_atoms = int(atom_counts[first_rep])
                if first_rep_atoms > max_batch_atoms:
                    raise MemoryError(
                        f"structure {first_rep} requires {first_rep_atoms} atoms, "
                        f"above max_batch_atoms capacity {max_batch_atoms}"
                    )
                rep_block: list[int] = []
                rep_atoms = 0
                scan = cursor
                while scan < len(rep_order) and len(rep_block) < pair_block_size:
                    row = rep_order[scan]
                    row_atoms = int(atom_counts[row])
                    if rep_atoms + row_atoms > max_batch_atoms:
                        break
                    rep_block.append(row)
                    rep_atoms += row_atoms
                    scan += 1
                if not rep_block:
                    # The first representative is handled by the oversized
                    # branch above; reaching this branch means invalid counts.
                    raise RuntimeError("failed to form an atom-bounded rep block")

                incident = [
                    (candidate, rep)
                    for candidate in unresolved
                    for rep in rep_block
                    if rep in shortlist_sets[candidate]
                ]
                rep_cursor = cursor
                while incident:
                    rep_build_start = time.perf_counter()
                    rep_batch: Batch | None = None
                    rep_types: Tensor | None = None
                    rep_indexes: dict[str, RadialComparisonIndex] | None = None
                    pair_tensor: Tensor | None = None
                    rep_batch, rep_types = _read_batch(
                        loader,
                        rep_block,
                        vocabulary,
                        include_hydrogens=include_hydrogens,
                    )
                    matched_pairs: list[tuple[int, int]] = []
                    try:
                        candidate_local = {
                            row: position for position, row in enumerate(tile)
                        }
                        candidate_bytes = _resident_descriptor_bytes(candidate_indexes)
                        rep_budget = _remaining_cuda_budget(budget, candidate_bytes)
                        rep_indexes = _build_mode_indices(
                            rep_batch,
                            rep_types,
                            cutoff=cutoff,
                            device=device,
                            max_memory_fraction=max_memory_fraction,
                            cuda_memory_budget_bytes=rep_budget,
                            include_hydrogens=include_hydrogens,
                        )
                        _configure_active_bundles(
                            (candidate_indexes, rep_indexes), budget
                        )
                        stats["descriptor_rebuilds"] += 1
                        stats["descriptor_structures_built"] += len(rep_block)
                        stats["descriptor_rebuild_seconds"] += (
                            time.perf_counter() - rep_build_start
                        )
                        rep_map = {
                            row: position for position, row in enumerate(rep_block)
                        }
                        scored_candidates: set[int] = set()
                        for pair_start in range(0, len(incident), pair_limit):
                            group = incident[pair_start : pair_start + pair_limit]
                            if confirm is None:
                                group = [
                                    pair
                                    for pair in group
                                    if pair[0] not in scored_candidates
                                ]
                            if not group:
                                continue
                            pair_tensor = torch.tensor(
                                [
                                    (candidate_local[left], rep_map[right])
                                    for left, right in group
                                ],
                                dtype=torch.int32,
                                device=device,
                            )
                            survivors = _screen_prebuilt_pairs(
                                candidate_indexes,
                                pair_tensor,
                                threshold,
                                right_indexes=rep_indexes,
                                stats=stats,
                                descriptor_structures=len(tile) + len(rep_block),
                            )
                            radial_pairs = [
                                (tile[left], rep_block[right])
                                for left, right in survivors.detach().cpu().tolist()
                            ]
                            matched_pairs.extend(radial_pairs)
                            if confirm is None:
                                scored_candidates.update(
                                    candidate for candidate, _ in radial_pairs
                                )
                            del pair_tensor
                            pair_tensor = None
                    except (MemoryError, torch.cuda.OutOfMemoryError) as exc:
                        if pair_tensor is not None:
                            del pair_tensor
                        if rep_indexes is not None:
                            del rep_indexes
                        if rep_batch is not None:
                            del rep_batch
                        if rep_types is not None:
                            del rep_types
                        if len(rep_block) > 1:
                            rep_block = rep_block[: max(1, len(rep_block) // 2)]
                            rep_set = set(rep_block)
                            incident = [pair for pair in incident if pair[1] in rep_set]
                            scan = rep_cursor + len(rep_block)
                            continue
                        if len(tile) > 1:
                            split_tile = True
                            break
                        candidate = tile[0]
                        representative = rep_block[0]
                        required = int(atom_counts[candidate]) + int(
                            atom_counts[representative]
                        )
                        raise MemoryError(
                            f"comparison pair ({candidate}, {representative}) with "
                            f"{required} atoms exceeds descriptor memory capacity: {exc}"
                        ) from exc
                    else:
                        del rep_indexes, rep_batch, rep_types
                        confirmed_pairs: list[tuple[int, int]] = []
                        for pair_start in range(0, len(matched_pairs), pair_limit):
                            # Bound callback proposal tensors independently of
                            # grouped radial output; user code stays outside the
                            # capacity-retry handler above.
                            confirmed_pairs.extend(
                                _confirm_pair_chunk(
                                    matched_pairs[pair_start : pair_start + pair_limit],
                                    confirm,
                                    device,
                                )
                            )
                        by_candidate: dict[int, list[int]] = {}
                        for candidate, representative in confirmed_pairs:
                            by_candidate.setdefault(candidate, []).append(
                                representative
                            )
                        for candidate, matching_reps in by_candidate.items():
                            output[candidate] = min(
                                matching_reps,
                                key=rep_rank.__getitem__,
                            )
                        resolved = set(by_candidate)
                        unresolved = [row for row in unresolved if row not in resolved]
                        break
                if split_tile:
                    break
                cursor = scan
        except _RetryActiveTile:
            pass
        finally:
            if candidate_indexes is not None:
                del candidate_indexes
            if candidate_batch is not None:
                del candidate_batch
            if candidate_types is not None:
                del candidate_types
        if split_tile:
            if len(tile) == 1:
                raise MemoryError(
                    f"candidate structure {tile[0]} could not fit an active tile"
                )
            midpoint = len(tile) // 2
            return process_tile(tile[:midpoint]) | process_tile(tile[midpoint:])
        return output

    for tile in candidate_tiles:
        matches.update(process_tile(tile))
    return matches


def _screen_pair_group(
    logical_pairs: list[tuple[int, int]],
    loader: Callable[[Tensor], tuple[Batch, Tensor | None]],
    vocabulary: tuple[int, ...] | None,
    cutoff: float,
    threshold: float,
    device: torch.device,
    stats: dict[str, Any],
    *,
    atom_counts: Tensor | None = None,
    max_batch_atoms: int = _MAX_BATCH_ATOMS,
    max_memory_fraction: float = 0.85,
    include_hydrogens: bool = True,
) -> list[tuple[int, int]]:
    """Score ordered pairs from atom-bounded endpoint descriptor tiles.

    Parameters
    ----------
    logical_pairs : list of tuple of int
        Ordered logical left/right row IDs to screen.
    loader : callable
        Reader for endpoint batches and optional atom types.
    vocabulary : tuple of int or None
        Fixed type vocabulary for typed rows.
    cutoff : float
        Radial cutoff in angstroms.
    threshold : float
        Inclusive fractional mismatch limit.
    device : torch.device
        Descriptor construction and scoring device.
    stats : dict
        Aggregate screen and descriptor counters updated in place.
    atom_counts : torch.Tensor, optional
        CPU atom counts indexed by logical row ID.
    max_batch_atoms : int
        Maximum atom count per endpoint tile.
    max_memory_fraction : float
        Fraction of PyTorch-available CUDA bytes assigned to the operation.

    Returns
    -------
    list of tuple of int
        Radial match proposals in input pair order.

    Raises
    ------
    MemoryError
        If a singleton descriptor pair cannot fit the atom or memory allowance.
    Exception
        Loader failures propagate unchanged and are never retried.

    Notes
    -----
    Endpoint unions are built once when they fit; otherwise both live bundles
    share the remaining CUDA allowance. Capacity failures split pair work.
    """
    if not logical_pairs:
        return []

    def split_by_atoms(
        pairs: list[tuple[int, int]],
    ) -> list[list[tuple[int, int]]]:
        """Partition ordered pairs by each side's distinct endpoint atom total.

        Parameters
        ----------
        pairs : list of tuple of int
            Logical endpoint pairs in caller order.

        Returns
        -------
        list of list of tuple of int
            Ordered pair tiles whose distinct left and right IDs each fit the
            active atom limit.

        Raises
        ------
        MemoryError
            If either endpoint of a pair exceeds the per-tile atom limit.
        """
        if atom_counts is None:
            return [pairs]
        tiles: list[list[tuple[int, int]]] = []
        current: list[tuple[int, int]] = []
        left_endpoints: set[int] = set()
        right_endpoints: set[int] = set()
        left_nodes = 0
        right_nodes = 0
        for pair in pairs:
            left, right = pair
            if (
                int(atom_counts[left]) > max_batch_atoms
                or int(atom_counts[right]) > max_batch_atoms
            ):
                offender = next(
                    value for value in pair if int(atom_counts[value]) > max_batch_atoms
                )
                raise MemoryError(
                    f"structure {offender} requires {int(atom_counts[offender])} "
                    f"atoms, above max_batch_atoms capacity {max_batch_atoms}"
                )

            add_left = 0 if left in left_endpoints else int(atom_counts[left])
            add_right = 0 if right in right_endpoints else int(atom_counts[right])
            if current and (
                left_nodes + add_left > max_batch_atoms
                or right_nodes + add_right > max_batch_atoms
            ):
                tiles.append(current)
                current = []
                left_endpoints, right_endpoints = set(), set()
                left_nodes = right_nodes = 0
                add_left = int(atom_counts[left])
                add_right = int(atom_counts[right])

            current.append(pair)
            left_endpoints.add(left)
            right_endpoints.add(right)
            left_nodes += add_left
            right_nodes += add_right
        if current:
            tiles.append(current)
        return tiles

    matches: list[tuple[int, int]] = []
    for initial_tile in split_by_atoms(logical_pairs):
        pending = [initial_tile]
        while pending:
            tile_pairs = pending.pop(0)
            left_ids = list(dict.fromkeys(left for left, _ in tile_pairs))
            right_ids = list(dict.fromkeys(right for _, right in tile_pairs))
            combined_ids = list(dict.fromkeys((*left_ids, *right_ids)))
            combined_atoms = (
                None
                if atom_counts is None
                else sum(int(atom_counts[row_id]) for row_id in combined_ids)
            )
            combine_endpoints = (
                combined_atoms is not None and combined_atoms <= max_batch_atoms
            )
            if combine_endpoints:
                # Most pair tiles fit in one atom-bounded descriptor tile. Read
                # and build their endpoint union once; pair lists can share
                # these indexes for both columns. Keep separate bundles only
                # when the union exceeds the cap but each side still fits.
                left_ids = combined_ids
                right_ids = combined_ids
            left_by_id = {row_id: index for index, row_id in enumerate(left_ids)}
            right_by_id = (
                left_by_id
                if combine_endpoints
                else {row_id: index for index, row_id in enumerate(right_ids)}
            )
            build_start = time.perf_counter()
            left_batch: Batch | None = None
            left_types: Tensor | None = None
            right_batch: Batch | None = None
            right_types: Tensor | None = None
            pairs: Tensor | None = None
            left_indexes: dict[str, RadialComparisonIndex] | None = None
            right_indexes: dict[str, RadialComparisonIndex] | None = None
            try:
                # Loader failures are caller errors and propagate unchanged;
                # only descriptor/scoring capacity failures split this group.
                left_batch, left_types = _read_batch(
                    loader,
                    left_ids,
                    vocabulary,
                    include_hydrogens=include_hydrogens,
                )
                if not combine_endpoints:
                    right_batch, right_types = _read_batch(
                        loader,
                        right_ids,
                        vocabulary,
                        include_hydrogens=include_hydrogens,
                    )
                try:
                    budget = _cuda_budget(device, max_memory_fraction)
                    left_indexes = _build_mode_indices(
                        left_batch,
                        left_types,
                        cutoff=cutoff,
                        device=device,
                        max_memory_fraction=max_memory_fraction,
                        cuda_memory_budget_bytes=budget,
                        include_hydrogens=include_hydrogens,
                    )
                    if combine_endpoints:
                        _configure_active_bundles((left_indexes,), budget)
                    else:
                        right_budget = _remaining_cuda_budget(
                            budget, _resident_descriptor_bytes(left_indexes)
                        )
                        right_indexes = _build_mode_indices(
                            right_batch,
                            right_types,
                            cutoff=cutoff,
                            device=device,
                            max_memory_fraction=max_memory_fraction,
                            cuda_memory_budget_bytes=right_budget,
                            include_hydrogens=include_hydrogens,
                        )
                        _configure_active_bundles((left_indexes, right_indexes), budget)
                    pairs = torch.tensor(
                        [
                            (left_by_id[left], right_by_id[right])
                            for left, right in tile_pairs
                        ],
                        dtype=torch.int32,
                        device=device,
                    )
                    descriptor_structures = (
                        len(left_ids)
                        if combine_endpoints
                        else len(left_ids) + len(right_ids)
                    )
                    stats["descriptor_rebuild_seconds"] += (
                        time.perf_counter() - build_start
                    )
                    stats["descriptor_rebuilds"] += 1 if combine_endpoints else 2
                    stats["descriptor_structures_built"] += descriptor_structures
                    pairs = _screen_prebuilt_pairs(
                        left_indexes,
                        pairs,
                        threshold,
                        right_indexes=right_indexes,
                        stats=stats,
                        descriptor_structures=descriptor_structures,
                    )
                    tile_matches = [
                        (left_ids[left], right_ids[right])
                        for left, right in pairs.detach().cpu().tolist()
                    ]
                    matches.extend(tile_matches)
                except (MemoryError, torch.cuda.OutOfMemoryError) as exc:
                    if len(tile_pairs) <= 1:
                        pair = tile_pairs[0]
                        required_atoms = (
                            0
                            if atom_counts is None
                            else sum(int(atom_counts[value]) for value in set(pair))
                        )
                        raise MemoryError(
                            f"comparison pair {pair} with {required_atoms} atoms "
                            f"exceeds descriptor memory capacity: {exc}"
                        ) from exc
                    middle = len(tile_pairs) // 2
                    pending[0:0] = [tile_pairs[:middle], tile_pairs[middle:]]
            finally:
                if left_indexes is not None:
                    del left_indexes
                if right_indexes is not None:
                    del right_indexes
                if pairs is not None:
                    del pairs
                if left_batch is not None:
                    del left_batch
                if left_types is not None:
                    del left_types
                if right_batch is not None:
                    del right_batch
                if right_types is not None:
                    del right_types
    return matches


def _candidate_matches_across(
    query_features: Tensor,
    query_presence: Tensor,
    candidate_features: Tensor,
    candidate_presence: Tensor,
    coordinate_centers: Tensor,
    sorted_values: Tensor,
    sorted_ids: Tensor,
    coordinate_safe: Tensor,
    bound: float | None,
) -> list[int]:
    """Return cross-pool rows surviving one query's safe summary interval.

    Parameters
    ----------
    query_features : torch.Tensor, shape (F,)
        Query's compact log-distance summary.
    query_presence : torch.Tensor, shape (T,), dtype bool
        Query center-type presence.
    candidate_features : torch.Tensor, shape (N, F)
        Right-pool compact summaries.
    candidate_presence : torch.Tensor, shape (N, T), dtype bool
        Right-pool center-type presence.
    coordinate_centers : torch.Tensor, shape (F,)
        Center-type index for each feature column.
    sorted_values, sorted_ids : torch.Tensor, shape (F, N)
        Per-coordinate sorted features and their right-pool row IDs.
    coordinate_safe : torch.Tensor, shape (F,), dtype bool
        Coordinates allowed to reject pairs conservatively.
    bound : float or None
        Conservative log-distance difference bound.

    Returns
    -------
    list of int
        Possible right-pool row IDs in ascending logical order.

    Notes
    -----
    The narrowest applicable interval seeds the result; presence and selected
    feature checks remove only pairs proven not to match.
    """
    candidate_count = candidate_features.shape[0]
    if not candidate_count:
        return []
    all_ids = torch.arange(candidate_count, dtype=torch.int64)
    if bound is None or not candidate_features.shape[1]:
        return all_ids.tolist()
    applicable = query_presence[coordinate_centers] & coordinate_safe
    if not torch.any(applicable):
        return all_ids.tolist()

    safe_coordinates = torch.nonzero(coordinate_safe, as_tuple=False).flatten()
    range_bound = torch.nextafter(
        torch.tensor(bound, dtype=torch.float32),
        torch.tensor(float("inf"), dtype=torch.float32),
    )
    coordinate_tensor, left_tensor, right_tensor, _ = _select_summary_intervals(
        query_features,
        applicable,
        sorted_values,
        range_bound,
    )
    coordinate = int(coordinate_tensor)
    interval_ids = sorted_ids[
        coordinate,
        left_tensor:right_tensor,
    ].to(torch.int64)
    if not interval_ids.numel():
        return []
    same_presence = (candidate_presence[interval_ids] == query_presence).all(dim=1)
    interval_ids = interval_ids[same_presence]
    if interval_ids.numel() == 0:
        return []
    centers = coordinate_centers[safe_coordinates]
    matches = _safe_summary_features_match(
        query_features[safe_coordinates],
        candidate_features[interval_ids][:, safe_coordinates],
        query_presence[centers],
        range_bound,
    )
    interval_ids = interval_ids[matches]
    return interval_ids.sort().values.tolist()


def _candidate_matches_across_batch(
    query_ids: Sequence[int],
    query_features: Tensor,
    query_presence: Tensor,
    candidate_features: Tensor,
    candidate_presence: Tensor,
    coordinate_centers: Tensor,
    sorted_values: Tensor,
    sorted_ids: Tensor,
    coordinate_safe: Tensor,
    bound: float | None,
) -> list[list[int]]:
    """Retrieve candidate lists for ordered query rows with bounded device work.

    Parameters
    ----------
    query_ids : sequence of int
        Original logical row IDs for the left-pool queries.
    query_features : torch.Tensor, shape (Nq, F)
        Left-pool compact summaries.
    query_presence : torch.Tensor, shape (Nq, T), dtype bool
        Left-pool center-type presence.
    candidate_features : torch.Tensor, shape (Nr, F)
        Right-pool compact summaries.
    candidate_presence : torch.Tensor, shape (Nr, T), dtype bool
        Right-pool center-type presence.
    coordinate_centers : torch.Tensor, shape (F,)
        Center-type index for each feature column.
    sorted_values, sorted_ids : torch.Tensor, shape (F, Nr)
        Per-coordinate sorted right-pool features and row IDs.
    coordinate_safe : torch.Tensor, shape (F,), dtype bool
        Coordinates allowed to reject pairs conservatively.
    bound : float or None
        Conservative log-distance difference bound.

    Returns
    -------
    list of list of int
        One ascending list of possible right-pool row IDs per query row.

    Notes
    -----
    Fixed-size filtering blocks avoid materializing the full query-by-pool
    Cartesian product.
    """
    output: list[list[int]] = [[] for _ in query_ids]
    candidate_count = candidate_features.shape[0]
    if not query_ids or not candidate_count:
        return output
    if bound is None or not candidate_features.shape[1]:
        return [list(range(candidate_count)) for _ in query_ids]

    device = query_features.device
    query_id_tensor = torch.tensor(query_ids, dtype=torch.int64, device=device)
    query_rows = query_features[query_id_tensor]
    query_presence_rows = query_presence[query_id_tensor]
    applicable = query_presence_rows[:, coordinate_centers] & coordinate_safe
    has_applicable = applicable.any(dim=1)
    fallback_rows = torch.nonzero(~has_applicable, as_tuple=False).flatten()
    fallback_list = fallback_rows.detach().cpu().tolist()
    for query_position in fallback_list:
        output[query_position] = list(range(candidate_count))
    if not torch.any(has_applicable):
        return output

    outward_bound = torch.nextafter(
        torch.tensor(bound, dtype=torch.float32, device=device),
        torch.tensor(float("inf"), dtype=torch.float32, device=device),
    )
    selected_coordinates, left, right, interval_widths = _select_summary_intervals(
        query_rows,
        applicable,
        sorted_values,
        outward_bound,
        has_applicable=has_applicable,
    )

    pair_limit = _summary_pair_filter_block_size(
        query_presence.shape[1], int(coordinate_safe.sum())
    )
    max_pairs_per_tile = max(1, _SUMMARY_PAIR_FILTER_WORKSPACE_BYTES // 8)
    query_tile_size = max(
        1,
        min(128, max_pairs_per_tile // max(1, candidate_count)),
    )
    for query_start in range(0, len(query_ids), query_tile_size):
        query_stop = min(len(query_ids), query_start + query_tile_size)
        query_positions = torch.arange(query_start, query_stop, device=device)
        query_positions = query_positions[has_applicable[query_positions]]
        if not query_positions.numel():
            continue
        max_width = int(interval_widths[query_positions].max())
        if max_width == 0:
            continue
        rows_per_block = max(1, pair_limit // query_positions.numel())
        for start in range(0, max_width, rows_per_block):
            stop = min(max_width, start + rows_per_block)
            positions = (
                left[query_positions, None]
                + torch.arange(start, stop, dtype=torch.int64, device=device)[None, :]
            )
            in_interval = positions < right[query_positions, None]
            bounded_positions = positions.clamp_(min=0, max=candidate_count - 1)
            candidate_ids = sorted_ids[
                selected_coordinates[query_positions, None], bounded_positions
            ].to(torch.int64)
            query_rows_in_block, candidate_columns = torch.nonzero(
                in_interval, as_tuple=True
            )
            if not query_rows_in_block.numel():
                continue
            local_queries = query_positions[query_rows_in_block]
            candidate_rows = candidate_ids[query_rows_in_block, candidate_columns]
            pair_presence = (
                query_presence[query_id_tensor[local_queries]]
                == candidate_presence[candidate_rows]
            ).all(dim=1)
            matching_presence = torch.nonzero(pair_presence, as_tuple=False).flatten()
            if not matching_presence.numel():
                continue
            local_queries = local_queries[matching_presence]
            candidate_rows = candidate_rows[matching_presence]
            safe_coordinates = torch.nonzero(coordinate_safe, as_tuple=False).flatten()
            safe_centers = coordinate_centers[safe_coordinates]
            active_coordinates = query_presence[query_id_tensor[local_queries]][
                :, safe_centers
            ]
            keep = _safe_summary_features_match(
                query_features[query_id_tensor[local_queries]][:, safe_coordinates],
                candidate_features[candidate_rows][:, safe_coordinates],
                active_coordinates,
                outward_bound,
            )
            pair_rows = torch.stack(
                (local_queries.to(torch.int32), candidate_rows.to(torch.int32)), dim=1
            )[keep]
            for query_position, candidate_id in pair_rows.detach().cpu().tolist():
                output[query_position].append(candidate_id)

    for candidate_rows in output:
        candidate_rows.sort()
    return output


def _add_screen_stats(target: dict[str, Any], source: dict[str, Any]) -> None:
    """Accumulate numeric descriptor and radial-screen counters in place.

    Parameters
    ----------
    target : dict
        Aggregate statistics owned by the stream call.
    source : dict
        One candidate's local screen counters.

    Notes
    -----
    The source dictionary is not modified.
    """
    for name in (
        "descriptor_rebuilds",
        "descriptor_structures_built",
        "descriptor_rebuild_seconds",
        "screen_seconds",
        "screen_pair_count",
        "scored_pair_count",
        "screen_pairs_untyped",
        "screen_pairs_center_typed",
        "screen_pairs_fully_typed",
        "descriptor_structures_untyped",
        "descriptor_structures_center_typed",
        "descriptor_structures_fully_typed",
    ):
        target[name] += source[name]


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
    max_batch_atoms: int = _MAX_BATCH_ATOMS,
    max_memory_fraction: float = 0.85,
    summary_coordinate_count: int = _DEFAULT_SUMMARY_COORDINATE_COUNT,
    include_hydrogens: bool = True,
    confirm: Callable[[Tensor], Tensor] | None = None,
) -> DeduplicationResult:
    """Deduplicate a pool in priority order using bounded descriptor blocks.

    Parameters
    ----------
    count : int
        Number of logical rows that ``read_typed_batch`` can read.
    read_typed_batch : callable
        Maps CPU int64 logical IDs to ``(Batch, atom_types)``. The int32 or
        int64 type vector aligns with all atoms in the returned batch. Return
        ``None`` as the type vector on every call for untyped comparison.
    type_vocabulary : sequence of int, optional
        Fixed set of signed int32 type IDs used across batches. Required when
        the loader returns atom types.
    cutoff : float
        Radial descriptor cutoff in angstroms.
    threshold : float
        Inclusive finite nonnegative approximate mismatch threshold.
    priority_order : sequence of int or torch.Tensor, optional
        A permutation of logical row IDs. Defaults to ``range(count)`` and
        determines candidate order and representative priority.
    device : torch.device or str, default="cpu"
        Device for descriptor construction and exact radial screens.
    input_batch_size : int, optional
        Maximum logical rows requested for each summary-building read and each
        candidate chunk. Defaults to 1024 on CUDA and 64 on CPU.
    pair_block_size : int, optional
        Maximum prior representatives considered together. Defaults to 256 on
        CUDA and 32 on CPU.
    max_batch_atoms : int, default=200000
        Maximum total atoms in one descriptor tile. A single structure above
        this limit raises ``MemoryError``.
    max_memory_fraction : float, default=0.85
        Maximum fraction of CUDA memory available to PyTorch assigned to
        descriptor construction and scoring workspace, including reusable
        caching-allocator bytes. This is an estimate, not a reservation.
    summary_coordinate_count : int, default=32
        Maximum number of coordinates used by the preliminary conservative
        rejection check, from 0 through 128. The selected count may be lower.
        Passing pairs go to the radial index matcher; summary bounds can reject
        but never establish a match. Zero disables only the outer pilot check;
        the index matcher may retain its built-in conservative summary bound.
    Returns
    -------
    DeduplicationResult
        Retained logical IDs in priority order, representative logical IDs
        indexed by original logical ID, and multiplicities aligned with retained
        IDs.

    Notes
    -----
    When ``summary_coordinate_count`` is positive, a deterministic pilot uses
    contiguous and spread rows, then greedily selects typed-summary coordinates
    with training queries; held-out queries measure shortlist selectivity. The
    stream retains those summaries, a presence table, and compact sorted search
    columns. It does not retain full descriptors. Candidate chunks pack
    shortlisted earlier representatives into priority-ordered groups; an
    adaptive schedule uses sequential screening when grouping does not reduce
    the block count. Zero disables only the pilot-selected outer rejection
        check and submits every still-eligible pair to the radial index matcher,
        which may apply its built-in conservative summary bound.
    confirm : callable, optional
        Receives ordered radial-match proposals as int32 [K, 2] original row
        IDs on the comparison device. Column zero is the candidate; column one
        is an earlier retained representative. Return an ordered subset on the
        same device. A rejected proposal does not discard the candidate; later
        representatives are tried. Without a callback, radial matches suffice.
        Calls may be grouped across candidates; decisions must depend on pair
        data rather than the global callback invocation order.
    Grouped chunks screen unresolved within-chunk pairs after a conservative
    summary filter. Every proposed pair passes the requested untyped,
    center-typed, and center-plus-neighbor-typed screens in order. Each active
    pair group builds one descriptor bundle when its endpoint union fits the
    atom cap, or separate left and right bundles (each within the atom cap)
    under one shared CUDA budget when that union is larger. All requested
    layouts reuse each bundle's geometry tile.
    Private stream statistics keep ``full_feature_candidates`` scoped to the
    prior-representative shortlist; local-pair counters distinguish all
    considered combinations from the pairs retained by summary filtering.
    The requested and selected summary coordinate counts are also recorded.
    """
    _LAST_STREAM_STATS.set(None)
    validate_include_hydrogens(include_hydrogens)
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise ValueError("count must be a nonnegative integer")
    if count > torch.iinfo(torch.int32).max:
        raise ValueError("count exceeds the int32 logical-index range")
    summary_coordinate_count = _nonnegative_int(
        summary_coordinate_count, "summary_coordinate_count"
    )
    if summary_coordinate_count > _MAX_SUMMARY_COORDINATE_COUNT:
        raise ValueError(
            f"summary_coordinate_count must be at most {_MAX_SUMMARY_COORDINATE_COUNT}"
        )
    cutoff = _finite_nonnegative(cutoff, "cutoff")
    cutoff_fp32 = torch.tensor(cutoff, dtype=torch.float32)
    if cutoff <= 0 or not torch.isfinite(cutoff_fp32) or cutoff_fp32 <= 0:
        raise ValueError("cutoff must be representable as positive finite FP32")
    threshold = _finite_nonnegative(threshold, "threshold")
    if not callable(read_typed_batch):
        raise TypeError("read_typed_batch must be callable")
    if confirm is not None and not callable(confirm):
        raise TypeError("confirm must be callable or None")

    if type_vocabulary is None:
        vocabulary = None
    else:
        try:
            vocabulary_input = tuple(type_vocabulary)
        except TypeError as exc:
            raise TypeError(
                "type_vocabulary must be a sequence of signed int32 IDs"
            ) from exc
        if any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or value <= torch.iinfo(torch.int32).min
            or value > torch.iinfo(torch.int32).max
            for value in vocabulary_input
        ):
            raise ValueError("type_vocabulary values must fit signed int32")
        vocabulary = tuple(sorted(vocabulary_input))
        if len(set(vocabulary)) != len(vocabulary):
            raise ValueError("type_vocabulary must contain unique signed int32 IDs")
    if priority_order is None:
        order = list(range(count))
    elif isinstance(priority_order, Tensor):
        order = priority_order.detach().cpu().tolist()
    else:
        order = list(priority_order)
    if (
        len(order) != count
        or any(isinstance(value, bool) or not isinstance(value, int) for value in order)
        or sorted(order) != list(range(count))
    ):
        raise ValueError("priority_order must be a permutation of logical row IDs")
    target_device = torch.device(device)
    if target_device.type not in ("cpu", "cuda"):
        raise ValueError("device must be CPU or CUDA")
    if target_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but CUDA is unavailable")
    target_device = resolve_device(target_device)
    if input_batch_size is None:
        input_batch_size = 1024 if target_device.type == "cuda" else 64
    else:
        input_batch_size = _positive_int(input_batch_size, "input_batch_size")
    if pair_block_size is None:
        pair_block_size = 256 if target_device.type == "cuda" else 32
    else:
        pair_block_size = _positive_int(pair_block_size, "pair_block_size")
    max_batch_atoms = _positive_int(max_batch_atoms, "max_batch_atoms")
    if (
        isinstance(max_memory_fraction, bool)
        or not isinstance(max_memory_fraction, (int, float))
        or not math.isfinite(max_memory_fraction)
        or not 0.0 < max_memory_fraction <= 0.85
    ):
        raise ValueError("max_memory_fraction must be in (0, 0.85]")
    max_memory_fraction = float(max_memory_fraction)

    stats: dict[str, Any] = {
        "completed": False,
        "timing_semantics": "host wall time; CUDA phase timings are asynchronous",
        "selected_coordinate_ids": [],
        "requested_summary_coordinate_count": summary_coordinate_count,
        "selected_summary_coordinate_count": 0,
        "pilot_heldout_selectivity": {},
        "summary_descriptor_rebuilds": 0,
        "summary_preparation_seconds": 0.0,
        "sorted_search_build_seconds": 0.0,
        "candidate_search_seconds": 0.0,
        "descriptor_rebuilds": 0,
        "descriptor_structures_built": 0,
        "descriptor_rebuild_seconds": 0.0,
        "screen_seconds": 0.0,
        "candidate_interval_rows": 0,
        "retained_interval_rows": 0,
        "full_feature_candidates": 0,
        "screen_pair_count": 0,
        "scored_pair_count": 0,
        "screen_pairs_untyped": 0,
        "screen_pairs_center_typed": 0,
        "screen_pairs_fully_typed": 0,
        "descriptor_structures_untyped": 0,
        "descriptor_structures_center_typed": 0,
        "descriptor_structures_fully_typed": 0,
        "old_shortlisted_pairs": 0,
        "old_shortlist_hit_blocks": 0,
        "old_shortlist_compact_groups": 0,
        "old_shortlist_sequential_blocks_predicted": 0,
        "within_chunk_pairs_considered": 0,
        "within_chunk_pairs_shortlisted": 0,
        "within_chunk_pairs_rejected_by_features": 0,
        "pilot_selection_seconds": 0.0,
        "grouped_memory_fallbacks": 0,
        "grouped_mode_chunks": 0,
        "sequential_mode_chunks": 0,
        "survivor_counts_by_chunk": [],
        "survivor_growth_by_chunk": [],
    }
    _LAST_STREAM_STATS.set(stats)
    if summary_coordinate_count == 0:
        selected: list[int] = []
        atom_counts = _build_atom_counts(
            count,
            read_typed_batch,
            vocabulary,
            input_batch_size,
            max_batch_atoms,
            include_hydrogens,
        )
        presence_width = len(vocabulary) if vocabulary is not None else (16 + 13) // 14
        features = torch.empty((count, 0), dtype=torch.float32, device=target_device)
        presence = torch.ones(
            (count, presence_width), dtype=torch.bool, device=target_device
        )
    else:
        pilot_ids, pilot_labels = _pilot_ids(order)
        start = time.perf_counter()
        pilot_features, pilot_presence, _, rebuilds = _build_summaries(
            pilot_ids,
            read_typed_batch,
            vocabulary,
            cutoff,
            target_device,
            input_batch_size,
            max_batch_atoms=max_batch_atoms,
            max_memory_fraction=max_memory_fraction,
            return_atom_counts=True,
            include_hydrogens=include_hydrogens,
        )
        stats["summary_preparation_seconds"] += time.perf_counter() - start
        stats["summary_descriptor_rebuilds"] += rebuilds
        select_start = time.perf_counter()
        selected = _select_coordinates(
            pilot_features,
            pilot_presence,
            threshold,
            max(1, len(vocabulary or ())),
            summary_coordinate_count,
        )
        stats["pilot_selection_seconds"] = time.perf_counter() - select_start
        stats["selected_coordinate_ids"] = selected
        stats["selected_summary_coordinate_count"] = len(selected)
        stats["pilot_heldout_selectivity"] = _pilot_heldout_selectivity(
            pilot_features,
            pilot_presence,
            pilot_labels,
            selected,
            threshold,
            max(1, len(vocabulary or ())),
        )
        presence_width = pilot_presence.shape[1]
        del pilot_features, pilot_presence

        features = torch.empty(
            (count, len(selected)), dtype=torch.float32, device=target_device
        )
        presence = torch.empty(
            (count, presence_width), dtype=torch.bool, device=target_device
        )
        atom_counts = torch.empty((count,), dtype=torch.int32)
        for chunk_start in range(0, count, input_batch_size):
            chunk_ids = order[chunk_start : chunk_start + input_batch_size]
            start = time.perf_counter()
            chunk_features, chunk_presence, chunk_atom_counts, rebuilds = (
                _build_summaries(
                    chunk_ids,
                    read_typed_batch,
                    vocabulary,
                    cutoff,
                    target_device,
                    input_batch_size,
                    max_batch_atoms=max_batch_atoms,
                    max_memory_fraction=max_memory_fraction,
                    return_atom_counts=True,
                    include_hydrogens=include_hydrogens,
                )
            )
            stats["summary_preparation_seconds"] += time.perf_counter() - start
            stats["summary_descriptor_rebuilds"] += rebuilds
            chunk_stop = chunk_start + len(chunk_ids)
            features[chunk_start:chunk_stop] = chunk_features[:, selected]
            presence[chunk_start:chunk_stop] = chunk_presence
            atom_counts[torch.tensor(chunk_ids, dtype=torch.int64)] = chunk_atom_counts
            del chunk_features, chunk_presence

    priority_ids = torch.tensor(order, dtype=torch.int32, device=target_device)
    rank_by_id = torch.empty(count, dtype=torch.int64, device=target_device)
    rank_by_id_cpu = [-1] * count
    if count:
        rank_by_id[priority_ids.to(torch.int64)] = torch.arange(
            count, dtype=torch.int64, device=target_device
        )
        for rank, logical_id in enumerate(order):
            rank_by_id_cpu[logical_id] = rank
    coordinate_centers = torch.tensor(
        [coordinate // (max(1, len(vocabulary or ())) * 14) for coordinate in selected],
        dtype=torch.int64,
        device=target_device,
    )
    sorted_values = torch.empty(
        (len(selected), count), dtype=torch.float32, device=target_device
    )
    sorted_ids = torch.empty(
        (len(selected), count), dtype=torch.int32, device=target_device
    )
    search_build_start = time.perf_counter()
    for column in range(len(selected)):
        permutation = torch.argsort(features[:, column], stable=True)
        sorted_values[column] = features[permutation, column]
        sorted_ids[column] = priority_ids[permutation]
        del permutation
    stats["sorted_search_build_seconds"] = time.perf_counter() - search_build_start

    coordinate_safe = torch.isfinite(features).all(dim=0)
    bound = _conservative_summary_log_bound(threshold)
    retained: list[int] = []
    retained_mask = torch.zeros(count, dtype=torch.bool, device=target_device)
    retained_slot_by_id = torch.full((count,), -1, dtype=torch.int32)
    representatives = [-1] * count
    multiplicities: list[int] = []

    candidate_chunk_size = input_batch_size
    for chunk_start in range(0, count, candidate_chunk_size):
        chunk_ids = order[chunk_start : chunk_start + candidate_chunk_size]
        retained_before_chunk = len(retained)
        shortlists: dict[int, Tensor] = {}
        if target_device.type == "cuda" and summary_coordinate_count > 0:
            search_start = time.perf_counter()
            candidate_ids_tensor = torch.tensor(
                chunk_ids, dtype=torch.int64, device=target_device
            )
            candidate_ranks = rank_by_id[candidate_ids_tensor]
            (candidate_lists, interval_counts, retained_counts, full_counts) = (
                _candidate_representatives_batch(
                    chunk_ids,
                    candidate_ranks,
                    retained,
                    retained_mask,
                    rank_by_id,
                    features,
                    presence,
                    coordinate_centers,
                    sorted_values,
                    sorted_ids,
                    coordinate_safe,
                    bound,
                )
            )
            stats["candidate_search_seconds"] += time.perf_counter() - search_start
            stats["candidate_interval_rows"] += sum(interval_counts)
            stats["retained_interval_rows"] += sum(retained_counts)
            stats["full_feature_candidates"] += sum(full_counts)
            shortlists = {
                candidate: torch.tensor(candidates, dtype=torch.int32)
                for candidate, candidates in zip(
                    chunk_ids, candidate_lists, strict=True
                )
            }
        else:
            for candidate in chunk_ids:
                candidate_rank = rank_by_id_cpu[candidate]
                search_start = time.perf_counter()
                candidates, interval_rows, retained_interval, full_candidates = (
                    _candidate_representatives(
                        candidate_rank,
                        retained,
                        retained_mask,
                        rank_by_id,
                        features,
                        presence,
                        coordinate_centers,
                        sorted_values,
                        sorted_ids,
                        coordinate_safe,
                        bound,
                    )
                )
                stats["candidate_search_seconds"] += time.perf_counter() - search_start
                stats["candidate_interval_rows"] += interval_rows
                stats["retained_interval_rows"] += retained_interval
                stats["full_feature_candidates"] += full_candidates
                shortlists[candidate] = torch.tensor(candidates, dtype=torch.int32)

        shortlisted_slots: set[int] = set()
        hit_historical_blocks: set[int] = set()
        predicted_sequential_blocks = 0
        for candidate in chunk_ids:
            shortlist = shortlists[candidate]
            shortlist_size = shortlist.numel()
            predicted_sequential_blocks += (
                shortlist_size + pair_block_size - 1
            ) // pair_block_size
            if shortlist_size:
                shortlist_slots = retained_slot_by_id[shortlist.to(torch.int64)]
                shortlisted_slots.update(int(slot) for slot in shortlist_slots)
                hit_historical_blocks.update(
                    int(slot) // pair_block_size for slot in shortlist_slots
                )
        # Compact groups are formed from the union of actual shortlist hits;
        # each group remains priority ordered because retained slots are.
        compact_representatives = [retained[slot] for slot in sorted(shortlisted_slots)]
        compact_rep_blocks = [
            compact_representatives[start : start + pair_block_size]
            for start in range(0, len(compact_representatives), pair_block_size)
        ]
        grouped_block_count = len(compact_rep_blocks)
        stats["old_shortlist_hit_blocks"] += len(hit_historical_blocks)
        stats["old_shortlist_compact_groups"] += grouped_block_count
        stats["old_shortlist_sequential_blocks_predicted"] += (
            predicted_sequential_blocks
        )
        sequential_schedule = (
            predicted_sequential_blocks > 0
            and grouped_block_count >= predicted_sequential_blocks
        )

        old_matches: dict[int, int] = {}
        group_old_shortlisted_pairs = 0
        if not sequential_schedule:
            grouped_candidates = [
                candidate for candidate in chunk_ids if shortlists[candidate].numel()
            ]
            group_old_shortlisted_pairs = sum(
                shortlists[candidate].numel() for candidate in grouped_candidates
            )
            old_matches = _screen_candidates_against_representatives(
                grouped_candidates,
                shortlists,
                compact_representatives,
                read_typed_batch,
                vocabulary,
                cutoff,
                threshold,
                target_device,
                stats,
                atom_counts=atom_counts,
                pair_block_size=pair_block_size,
                max_batch_atoms=max_batch_atoms,
                max_memory_fraction=max_memory_fraction,
                confirm=confirm,
                include_hydrogens=include_hydrogens,
            )

        assignments: dict[int, int] = {}
        if not sequential_schedule:
            unresolved = [
                candidate for candidate in chunk_ids if candidate not in old_matches
            ]
            within_chunk_matches: set[tuple[int, int]] = set()
            if len(unresolved) > 1:
                all_local_pair_count = len(unresolved) * (len(unresolved) - 1) // 2
                stats["within_chunk_pairs_considered"] += all_local_pair_count
                local_pairs = _within_chunk_candidate_pairs(
                    unresolved,
                    rank_by_id,
                    features,
                    presence,
                    coordinate_centers,
                    coordinate_safe,
                    bound,
                )
                stats["within_chunk_pairs_shortlisted"] += len(local_pairs)
                stats["within_chunk_pairs_rejected_by_features"] += (
                    all_local_pair_count - len(local_pairs)
                )
                if local_pairs:
                    # This grouped pass finds radial edges only. Greedy replay
                    # below filters them against the prefix actually retained
                    # and confirms only those non-speculative pairs.
                    matches = _screen_pair_group(
                        local_pairs,
                        read_typed_batch,
                        vocabulary,
                        cutoff,
                        threshold,
                        target_device,
                        stats,
                        atom_counts=atom_counts,
                        max_batch_atoms=max_batch_atoms,
                        max_memory_fraction=max_memory_fraction,
                        include_hydrogens=include_hydrogens,
                    )
                    within_chunk_matches = set(matches)

            assignments = dict(old_matches)
            local_retained: list[int] = []
            confirm_pair_limit = 8192 if target_device.type == "cuda" else 4096
            for candidate in unresolved:
                # Earlier candidates are provisional until this priority replay
                # retains them. Never score or confirm against speculative
                # within-chunk radial edges to a candidate that was discarded.
                matching_local_reps = [
                    earlier
                    for earlier in local_retained
                    if (candidate, earlier) in within_chunk_matches
                ]
                representative = None
                if confirm is None:
                    if matching_local_reps:
                        representative = matching_local_reps[0]
                else:
                    for start in range(0, len(matching_local_reps), confirm_pair_limit):
                        confirmed = _confirm_pair_chunk(
                            [
                                (candidate, earlier)
                                for earlier in matching_local_reps[
                                    start : start + confirm_pair_limit
                                ]
                            ],
                            confirm,
                            target_device,
                        )
                        if confirmed:
                            representative = confirmed[0][1]
                            break
                if representative is None:
                    local_retained.append(candidate)
                else:
                    assignments[candidate] = representative

        if sequential_schedule:
            stats["sequential_mode_chunks"] += 1
            local_retained: list[int] = []
            new_representatives: list[int] = []
            for candidate in chunk_ids:
                old_representative, screen_stats = _screen_candidate(
                    candidate,
                    shortlists[candidate].tolist(),
                    read_typed_batch,
                    vocabulary,
                    cutoff,
                    threshold,
                    target_device,
                    pair_block_size,
                    atom_counts=atom_counts,
                    max_batch_atoms=max_batch_atoms,
                    max_memory_fraction=max_memory_fraction,
                    confirm=confirm,
                    include_hydrogens=include_hydrogens,
                )
                stats["old_shortlisted_pairs"] += screen_stats[
                    "screened_pair_candidates"
                ]
                _add_screen_stats(stats, screen_stats)
                representative = old_representative
                if representative is None and local_retained:
                    if target_device.type == "cuda" and summary_coordinate_count > 0:
                        local_ids = torch.tensor(
                            [(candidate, earlier) for earlier in local_retained],
                            dtype=torch.int64,
                            device=target_device,
                        )
                        local_ranks = rank_by_id[local_ids]
                        local_keep = (
                            _summary_pairs_may_match_batch(
                                local_ranks,
                                features,
                                presence,
                                coordinate_centers,
                                coordinate_safe,
                                bound,
                            )
                            .detach()
                            .cpu()
                            .tolist()
                        )
                        local_candidates = [
                            earlier
                            for earlier, keep in zip(
                                local_retained, local_keep, strict=True
                            )
                            if keep
                        ]
                    else:
                        candidate_rank = rank_by_id_cpu[candidate]
                        local_candidates = [
                            earlier
                            for earlier in local_retained
                            if _summary_pair_may_match(
                                candidate_rank,
                                rank_by_id_cpu[earlier],
                                features,
                                presence,
                                coordinate_centers,
                                coordinate_safe,
                                bound,
                            )
                        ]
                    stats["within_chunk_pairs_considered"] += len(local_retained)
                    stats["within_chunk_pairs_shortlisted"] += len(local_candidates)
                    stats["within_chunk_pairs_rejected_by_features"] += len(
                        local_retained
                    ) - len(local_candidates)
                    representative, screen_stats = _screen_candidate(
                        candidate,
                        local_candidates,
                        read_typed_batch,
                        vocabulary,
                        cutoff,
                        threshold,
                        target_device,
                        pair_block_size,
                        atom_counts=atom_counts,
                        max_batch_atoms=max_batch_atoms,
                        max_memory_fraction=max_memory_fraction,
                        confirm=confirm,
                        include_hydrogens=include_hydrogens,
                    )
                    _add_screen_stats(stats, screen_stats)
                if representative is None:
                    slot = len(retained)
                    retained.append(candidate)
                    local_retained.append(candidate)
                    new_representatives.append(candidate)
                    retained_slot_by_id[candidate] = slot
                    representatives[candidate] = candidate
                    multiplicities.append(1)
                else:
                    representatives[candidate] = representative
                    slot = int(retained_slot_by_id[representative])
                    multiplicities[slot] += 1
        else:
            stats["grouped_mode_chunks"] += 1
            stats["old_shortlisted_pairs"] += group_old_shortlisted_pairs
            new_representatives = []
            for candidate in chunk_ids:
                representative = assignments.get(candidate)
                if representative is None:
                    slot = len(retained)
                    retained.append(candidate)
                    new_representatives.append(candidate)
                    retained_slot_by_id[candidate] = slot
                    representatives[candidate] = candidate
                    multiplicities.append(1)
                else:
                    representatives[candidate] = representative
                    slot = int(retained_slot_by_id[representative])
                    multiplicities[slot] += 1
        if new_representatives:
            new_representative_ids = torch.tensor(
                new_representatives, dtype=torch.int64, device=target_device
            )
            retained_mask[new_representative_ids] = True
        stats["survivor_counts_by_chunk"].append(len(retained))
        stats["survivor_growth_by_chunk"].append(len(retained) - retained_before_chunk)

    stats["completed"] = True
    _LAST_STREAM_STATS.set(copy.deepcopy(stats))
    return DeduplicationResult(
        torch.tensor(retained, dtype=torch.int32, device=target_device),
        torch.tensor(representatives, dtype=torch.int32, device=target_device),
        torch.tensor(multiplicities, dtype=torch.int32, device=target_device),
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
    max_batch_atoms: int = _MAX_BATCH_ATOMS,
    max_memory_fraction: float = 0.85,
    pair_chunk_size: int = 8192,
    summary_coordinate_count: int = _DEFAULT_SUMMARY_COORDINATE_COUNT,
    include_hydrogens: bool = True,
) -> Iterator[Tensor]:
    """Yield radial matches from one or two loader-backed structure pools.

    The function begins reading when the iterator is advanced. Loaders receive
    CPU int64 logical row IDs and return a ``(Batch, atom_types)`` pair, with an
    int32 or int64 type vector aligned to all Batch atoms, or ``None`` for
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
    validate_include_hydrogens(include_hydrogens)
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise ValueError("count must be a nonnegative integer")
    if count > torch.iinfo(torch.int32).max:
        raise ValueError("count exceeds the int32 logical-index range")
    if (other_count is None) != (read_other_typed_batch is None):
        raise ValueError(
            "other_count and read_other_typed_batch must be supplied together"
        )
    cross_pool = other_count is not None
    if other_count is None:
        right_count = count
        read_right = read_typed_batch
    else:
        if (
            isinstance(other_count, bool)
            or not isinstance(other_count, int)
            or other_count < 0
        ):
            raise ValueError("other_count must be a nonnegative integer")
        if other_count > torch.iinfo(torch.int32).max:
            raise ValueError("other_count exceeds the int32 logical-index range")
        right_count = other_count
        read_right = read_other_typed_batch
    if not callable(read_typed_batch) or not callable(read_right):
        raise TypeError("pool loaders must be callable")
    if type_vocabulary is None:
        vocabulary = None
    else:
        try:
            vocabulary_input = tuple(type_vocabulary)
        except TypeError as exc:
            raise TypeError(
                "type_vocabulary must be a sequence of signed int32 IDs"
            ) from exc
        if any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or value <= torch.iinfo(torch.int32).min
            or value > torch.iinfo(torch.int32).max
            for value in vocabulary_input
        ):
            raise ValueError("type_vocabulary values must fit signed int32")
        vocabulary = tuple(sorted(vocabulary_input))
        if len(set(vocabulary)) != len(vocabulary):
            raise ValueError("type_vocabulary must contain unique signed int32 IDs")
    cutoff = _finite_nonnegative(cutoff, "cutoff")
    cutoff_fp32 = torch.tensor(cutoff, dtype=torch.float32)
    if cutoff <= 0 or not torch.isfinite(cutoff_fp32) or cutoff_fp32 <= 0:
        raise ValueError("cutoff must be representable as positive finite FP32")
    threshold = _finite_nonnegative(threshold, "threshold")
    max_batch_atoms = _positive_int(max_batch_atoms, "max_batch_atoms")
    pair_chunk_size = _positive_int(pair_chunk_size, "pair_chunk_size")
    summary_coordinate_count = _nonnegative_int(
        summary_coordinate_count, "summary_coordinate_count"
    )
    if summary_coordinate_count > _MAX_SUMMARY_COORDINATE_COUNT:
        raise ValueError(
            f"summary_coordinate_count must be at most {_MAX_SUMMARY_COORDINATE_COUNT}"
        )
    if (
        isinstance(max_memory_fraction, bool)
        or not isinstance(max_memory_fraction, (int, float))
        or not math.isfinite(max_memory_fraction)
        or not 0.0 < max_memory_fraction <= 0.85
    ):
        raise ValueError("max_memory_fraction must be in (0, 0.85]")
    max_memory_fraction = float(max_memory_fraction)
    target_device = torch.device(device)
    if target_device.type not in ("cpu", "cuda"):
        raise ValueError("device must be CPU or CUDA")
    if target_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but CUDA is unavailable")
    target_device = resolve_device(target_device)

    # Everything below belongs to the iterator call and is dropped on exhaustion
    # or explicit close; loader objects remain caller-owned.
    left_features: Tensor | None = None
    right_features: Tensor | None = None
    left_presence: Tensor | None = None
    right_presence: Tensor | None = None
    left_atom_counts: Tensor | None = None
    right_atom_counts: Tensor | None = None
    sorted_values: Tensor | None = None
    sorted_ids: Tensor | None = None
    coordinate_centers: Tensor | None = None
    coordinate_safe: Tensor | None = None
    try:
        if not count or not right_count:
            return
        input_batch_size = 1024 if target_device.type == "cuda" else 64
        type_count = max(1, len(vocabulary or ()))
        if summary_coordinate_count == 0:
            selected: list[int] = []
            left_atom_counts = _build_atom_counts(
                count,
                read_typed_batch,
                vocabulary,
                input_batch_size,
                max_batch_atoms,
                include_hydrogens,
            )
            if cross_pool:
                right_atom_counts = _build_atom_counts(
                    right_count,
                    read_right,
                    vocabulary,
                    input_batch_size,
                    max_batch_atoms,
                    include_hydrogens,
                )
            else:
                right_atom_counts = left_atom_counts
            left_features = torch.empty(
                (count, 0), dtype=torch.float32, device=target_device
            )
            left_presence = torch.empty(
                (count, 0), dtype=torch.bool, device=target_device
            )
            if cross_pool:
                right_features = torch.empty(
                    (right_count, 0), dtype=torch.float32, device=target_device
                )
                right_presence = torch.empty(
                    (right_count, 0), dtype=torch.bool, device=target_device
                )
            else:
                right_features = left_features
                right_presence = left_presence
        else:
            left_pilot_ids, left_labels = _pilot_ids(list(range(count)))
            left_pilot, left_pilot_presence, _, _ = _build_summaries(
                left_pilot_ids,
                read_typed_batch,
                vocabulary,
                cutoff,
                target_device,
                input_batch_size,
                max_batch_atoms=max_batch_atoms,
                max_memory_fraction=max_memory_fraction,
                return_atom_counts=True,
                include_hydrogens=include_hydrogens,
            )
            if cross_pool:
                right_pilot_ids, right_labels = _pilot_ids(list(range(right_count)))
                right_pilot, right_pilot_presence, _, _ = _build_summaries(
                    right_pilot_ids,
                    read_right,
                    vocabulary,
                    cutoff,
                    target_device,
                    input_batch_size,
                    max_batch_atoms=max_batch_atoms,
                    max_memory_fraction=max_memory_fraction,
                    return_atom_counts=True,
                    include_hydrogens=include_hydrogens,
                )
                pilot_features = torch.cat((left_pilot, right_pilot))
                pilot_presence = torch.cat((left_pilot_presence, right_pilot_presence))
                del right_labels, right_pilot_ids
            else:
                pilot_features = left_pilot
                pilot_presence = left_pilot_presence
            selected = _select_coordinates(
                pilot_features,
                pilot_presence,
                threshold,
                type_count,
                summary_coordinate_count,
            )
            del pilot_features, pilot_presence, left_pilot, left_pilot_presence
            if cross_pool:
                del right_pilot, right_pilot_presence

            feature_count = len(selected)
            presence_width = (
                len(vocabulary) if vocabulary is not None else (16 + 13) // 14
            )
            left_features = torch.empty(
                (count, feature_count), dtype=torch.float32, device=target_device
            )
            left_presence = torch.empty(
                (count, presence_width), dtype=torch.bool, device=target_device
            )
            left_atom_counts = torch.empty((count,), dtype=torch.int32)
            if cross_pool:
                right_features = torch.empty(
                    (right_count, feature_count),
                    dtype=torch.float32,
                    device=target_device,
                )
                right_presence = torch.empty(
                    (right_count, presence_width),
                    dtype=torch.bool,
                    device=target_device,
                )
                right_atom_counts = torch.empty((right_count,), dtype=torch.int32)
            else:
                # Self-comparison uses the same summary rows for both sides;
                # avoid allocating and then discarding a duplicate pool copy.
                right_features = left_features
                right_presence = left_presence
                right_atom_counts = left_atom_counts
            for start in range(0, count, input_batch_size):
                ids = list(range(start, min(count, start + input_batch_size)))
                features, presence, counts, _ = _build_summaries(
                    ids,
                    read_typed_batch,
                    vocabulary,
                    cutoff,
                    target_device,
                    input_batch_size,
                    max_batch_atoms=max_batch_atoms,
                    max_memory_fraction=max_memory_fraction,
                    return_atom_counts=True,
                    include_hydrogens=include_hydrogens,
                )
                stop = start + len(ids)
                left_features[start:stop] = features[:, selected]
                left_presence[start:stop] = presence
                left_atom_counts[start:stop] = counts
                del features, presence, counts
            if cross_pool:
                for start in range(0, right_count, input_batch_size):
                    ids = list(range(start, min(right_count, start + input_batch_size)))
                    features, presence, counts, _ = _build_summaries(
                        ids,
                        read_right,
                        vocabulary,
                        cutoff,
                        target_device,
                        input_batch_size,
                        max_batch_atoms=max_batch_atoms,
                        max_memory_fraction=max_memory_fraction,
                        return_atom_counts=True,
                        include_hydrogens=include_hydrogens,
                    )
                    stop = start + len(ids)
                    right_features[start:stop] = features[:, selected]
                    right_presence[start:stop] = presence
                    right_atom_counts[start:stop] = counts
                    del features, presence, counts

        feature_count = len(selected)
        coordinate_centers = torch.tensor(
            [coordinate // (type_count * 14) for coordinate in selected],
            dtype=torch.int64,
            device=target_device,
        )
        coordinate_safe = torch.isfinite(left_features).all(dim=0) & torch.isfinite(
            right_features
        ).all(dim=0)
        sorted_values = torch.empty(
            (feature_count, right_count), dtype=torch.float32, device=target_device
        )
        sorted_ids = torch.empty(
            (feature_count, right_count), dtype=torch.int32, device=target_device
        )
        for column in range(feature_count):
            permutation = torch.argsort(right_features[:, column], stable=True)
            sorted_values[column] = right_features[permutation, column]
            sorted_ids[column] = permutation.to(torch.int32)
        bound = _conservative_summary_log_bound(threshold)

        gpu_query_batch_size = max(
            1,
            min(
                input_batch_size,
                _SUMMARY_PAIR_FILTER_WORKSPACE_BYTES // 40 // max(1, right_count),
            ),
        )
        query_batch_start = -1
        batched_candidate_ids: list[list[int]] = []
        for left_id in range(count):
            if target_device.type == "cuda" and summary_coordinate_count > 0:
                if left_id >= query_batch_start + len(batched_candidate_ids):
                    query_batch_start = left_id
                    query_ids = list(
                        range(
                            left_id,
                            min(count, left_id + gpu_query_batch_size),
                        )
                    )
                    batched_candidate_ids = _candidate_matches_across_batch(
                        query_ids,
                        left_features,
                        left_presence,
                        right_features,
                        right_presence,
                        coordinate_centers,
                        sorted_values,
                        sorted_ids,
                        coordinate_safe,
                        bound,
                    )
                candidate_ids = batched_candidate_ids[left_id - query_batch_start]
            else:
                candidate_ids = _candidate_matches_across(
                    left_features[left_id],
                    left_presence[left_id],
                    right_features,
                    right_presence,
                    coordinate_centers,
                    sorted_values,
                    sorted_ids,
                    coordinate_safe,
                    bound,
                )
            if not cross_pool:
                candidate_ids = [
                    right_id for right_id in candidate_ids if right_id > left_id
                ]
            if not candidate_ids:
                continue
            left_batch: Batch | None = None
            left_types: Tensor | None = None
            left_indexes: dict[str, RadialComparisonIndex] | None = None
            try:
                left_batch, left_types = _read_batch(
                    read_typed_batch,
                    [left_id],
                    vocabulary,
                    include_hydrogens=include_hydrogens,
                )
                left_atoms = int(left_atom_counts[left_id])
                if left_atoms > max_batch_atoms:
                    raise MemoryError(
                        f"structure {left_id} requires {left_atoms} atoms, above "
                        f"max_batch_atoms capacity {max_batch_atoms}"
                    )
                budget = _cuda_budget(target_device, max_memory_fraction)
                try:
                    left_indexes = _build_mode_indices(
                        left_batch,
                        left_types,
                        cutoff=cutoff,
                        device=target_device,
                        max_memory_fraction=max_memory_fraction,
                        cuda_memory_budget_bytes=budget,
                        include_hydrogens=include_hydrogens,
                    )
                except (MemoryError, torch.cuda.OutOfMemoryError) as exc:
                    # Only descriptor construction is normalized here. Loader
                    # failures occur before this boundary and propagate intact.
                    raise MemoryError(
                        f"structure {left_id} with {left_atoms} atoms exceeds "
                        f"descriptor memory capacity: {exc}"
                    ) from exc
                right_cursor = 0
                pair_limit = min(
                    pair_chunk_size,
                    8192 if target_device.type == "cuda" else 4096,
                )
                while right_cursor < len(candidate_ids):
                    right_block: list[int] = []
                    right_atoms = 0
                    while (
                        right_cursor + len(right_block) < len(candidate_ids)
                        and len(right_block) < pair_limit
                    ):
                        right_id = candidate_ids[right_cursor + len(right_block)]
                        row_atoms = int(right_atom_counts[right_id])
                        if row_atoms > max_batch_atoms:
                            if not right_block:
                                raise MemoryError(
                                    f"structure {right_id} requires {row_atoms} atoms, "
                                    "above "
                                    f"max_batch_atoms capacity {max_batch_atoms}"
                                )
                            break
                        if right_atoms + row_atoms > max_batch_atoms:
                            break
                        right_block.append(right_id)
                        right_atoms += row_atoms
                    if not right_block:
                        raise RuntimeError(
                            "failed to form an atom-bounded similarity block"
                        )

                    while True:
                        right_batch: Batch | None = None
                        right_types: Tensor | None = None
                        right_indexes: dict[str, RadialComparisonIndex] | None = None
                        pair_tensor: Tensor | None = None
                        output_chunk: Tensor | None = None
                        right_batch, right_types = _read_batch(
                            read_right,
                            right_block,
                            vocabulary,
                            include_hydrogens=include_hydrogens,
                        )
                        try:
                            if (left_types is None) != (right_types is None):
                                raise ValueError(
                                    "both comparison pools must either supply types or omit them"
                                )
                            left_bytes = _resident_descriptor_bytes(left_indexes)
                            right_budget = _remaining_cuda_budget(budget, left_bytes)
                            right_indexes = _build_mode_indices(
                                right_batch,
                                right_types,
                                cutoff=cutoff,
                                device=target_device,
                                max_memory_fraction=max_memory_fraction,
                                cuda_memory_budget_bytes=right_budget,
                                include_hydrogens=include_hydrogens,
                            )
                            _configure_active_bundles(
                                (left_indexes, right_indexes), budget
                            )
                            pair_tensor = torch.tensor(
                                [(0, index) for index in range(len(right_block))],
                                dtype=torch.int32,
                                device=target_device,
                            )
                            survivors = _screen_prebuilt_pairs(
                                left_indexes,
                                pair_tensor,
                                threshold,
                                right_indexes=right_indexes,
                            )
                            matches = [
                                (left_id, right_block[right])
                                for _, right in survivors.detach().cpu().tolist()
                            ]
                            if matches:
                                output_chunk = torch.tensor(
                                    matches,
                                    dtype=torch.int32,
                                    device=target_device,
                                ).reshape(-1, 2)
                            right_cursor += len(right_block)
                        except (MemoryError, torch.cuda.OutOfMemoryError) as exc:
                            if len(right_block) > 1:
                                right_block = right_block[
                                    : max(1, len(right_block) // 2)
                                ]
                                right_atoms = sum(
                                    int(right_atom_counts[row]) for row in right_block
                                )
                                continue
                            right_id = right_block[0]
                            required_atoms = left_atoms + int(
                                right_atom_counts[right_id]
                            )
                            pair_label = (
                                "cross-pool pair" if cross_pool else "comparison pair"
                            )
                            raise MemoryError(
                                f"{pair_label} ({left_id}, {right_id}) with "
                                f"{required_atoms} atoms exceeds descriptor memory "
                                f"capacity: {exc}"
                            ) from exc
                        finally:
                            if pair_tensor is not None:
                                del pair_tensor
                            if right_indexes is not None:
                                del right_indexes
                            if right_batch is not None:
                                del right_batch
                            if right_types is not None:
                                del right_types
                        if output_chunk is not None:
                            yield output_chunk
                        break
            finally:
                if left_indexes is not None:
                    del left_indexes
                if left_batch is not None:
                    del left_batch
                if left_types is not None:
                    del left_types
    finally:
        del left_features, right_features, left_presence, right_presence
        del left_atom_counts, right_atom_counts, sorted_values, sorted_ids
        del coordinate_centers, coordinate_safe
