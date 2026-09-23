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
"""Zarr backend for AtomicData (de)serialization.

This module provides the concrete implementation of ``AtomicData``
(de)serialization using high performance ``zarr`` array I/O.

The ``AtomicDataZarrWriter`` class is designed to allow for efficient,
amortized data writes with the ability to directly save/append ``Batch``
objects to disk.

The ``AtomicDataZarrReader`` provides a concrete ``Reader`` implementation
that reads in arrays from disk, and maps them to ``torch.Tensor``s that
are intended to composed with :class:`nvalchemi.data.datapipes.Dataset`.

To understand usage, users should refer to ``examples/data/datapipes/read_zarr_store.py``.
"""

from __future__ import annotations

import dataclasses
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Annotated, Any, Literal, TypeAlias

import numpy as np
import torch
import zarr
import zarr.abc.codec
from plum import dispatch, overload
from pydantic import BaseModel, ConfigDict, Field, model_validator
from zarr.abc.store import Store
from zarr.storage import StorePath

# These need to be available at runtime for plum dispatch
from nvalchemi.data.atomic_data import AtomicData
from nvalchemi.data.batch import Batch
from nvalchemi.data.datapipes.backends.base import Reader
from nvalchemi.data.level_storage import LevelSchema

_BUILTIN_LEVELS = frozenset({"atoms", "edges", "system"})

# Type alias for zarr store-like objects
StoreLike: TypeAlias = Store | StorePath | Path | str | dict[str, Any]

_REPORTED_MISMATCHES = 4
"""Number of disagreeing store arrays named before an integrity error truncates."""


@dataclasses.dataclass(frozen=True)
class FieldSchema:
    """Store level, dtype, and row shape of one field an ALCHEMI Zarr store holds.

    Parameters
    ----------
    level : str
        Built-in level alias (``"atom"``, ``"edge"``, or ``"system"``) or
        registered custom level name the field is stored at.
    dtype : torch.dtype
        Dtype the field's array reads back as.
    row_shape : tuple[int, ...]
        Shape of one stored row: the array's shape without the axis samples
        are concatenated along.
    """

    level: str
    dtype: torch.dtype
    row_shape: tuple[int, ...]


def _torn_store_error(detail: str) -> ValueError:
    """Return the error raised for a store an interrupted append left inconsistent."""
    return ValueError(
        f"Zarr store is inconsistent: {detail}. This is what an append interrupted "
        "mid-write leaves behind; truncate the store back to its committed samples "
        "or write a fresh one."
    )


class ZarrArrayConfig(BaseModel):
    """Per-array storage settings for compression, chunking, and sharding.

    A ``ZarrArrayConfig`` bundles the codec and layout choices applied to a
    single Zarr array written by :class:`AtomicDataZarrWriter`. The codec
    fields (``compressors``, ``filters``, ``serializer``) accept ``zarr`` v3
    codec instances and control how bytes are transformed on write; leaving
    them ``None`` uses Zarr's defaults. ``chunk_size`` sets the chunk length
    along the leading (row / sample) dimension, with all other dimensions
    stored at full extent, and ``shard_size`` optionally groups several chunks
    into one storage object to reduce file count for object stores.

    You rarely construct this directly for a whole store; instead you attach
    one or more ``ZarrArrayConfig`` instances to a :class:`ZarrWriteConfig`,
    which routes them to the metadata, core, and custom array groups (and to
    per-field overrides). Tuning these settings trades write size and speed
    against read throughput -- see the ``nvalchemi-zarr-perf`` guidance for
    chunk/shard sizing under shuffled or random access.

    Examples
    --------
    Zstandard compression with 1024-row chunks::

        from zarr.codecs import ZstdCodec

        cfg = ZarrArrayConfig(compressors=(ZstdCodec(level=3),), chunk_size=1024)

    Group four chunks into each shard (``shard_size`` must be a multiple of
    ``chunk_size``)::

        cfg = ZarrArrayConfig(chunk_size=256, shard_size=1024)

    Notes
    -----
    When both ``chunk_size`` and ``shard_size`` are set, ``shard_size`` must be
    an exact multiple of ``chunk_size``; an ``after`` validator raises
    ``ValueError`` otherwise. ``compressors`` and ``filters`` are tuples of
    codecs applied in order, and ``arbitrary_types_allowed`` is enabled so that
    native ``zarr`` codec objects can be stored as field values.

    .. seealso::

       :ref:`zarr_compression_guide` -- choosing codecs, chunk sizes, and shard
       sizes to balance store size against read and write throughput.
    """

    compressors: Annotated[
        tuple[zarr.abc.codec.Codec, ...] | None,
        Field(description="Compressor codec(s) to apply."),
    ] = None
    filters: Annotated[
        tuple[zarr.abc.codec.Codec, ...] | None,
        Field(description="Array-to-array filter codec(s)."),
    ] = None
    serializer: Annotated[
        zarr.abc.codec.Codec | None,
        Field(description="Bytes serializer codec."),
    ] = None
    chunk_size: Annotated[
        int | None,
        Field(
            description="Chunk length along dimension 0. Other dims use full extent."
        ),
    ] = None
    shard_size: Annotated[
        int | None,
        Field(
            description=(
                "Shard length along dimension 0. "
                "When set, multiple chunks are stored in a single storage object. "
                "Must be a multiple of chunk_size when both are specified."
            ),
        ),
    ] = None
    write_empty_chunks: Annotated[
        bool,
        Field(description="Whether to write chunks that are entirely fill-valued."),
    ] = True

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @model_validator(mode="after")
    def _validate_shard_chunk_alignment(self) -> ZarrArrayConfig:
        """Validate that shard_size is a multiple of chunk_size."""
        if self.shard_size is not None and self.chunk_size is not None:
            if self.shard_size % self.chunk_size != 0:
                msg = (
                    f"shard_size ({self.shard_size}) must be a multiple of "
                    f"chunk_size ({self.chunk_size})"
                )
                raise ValueError(msg)
        return self


class ZarrWriteConfig(BaseModel):
    """Top-level storage plan handed to ``AtomicDataZarrWriter``.

    A ``ZarrWriteConfig`` collects the :class:`ZarrArrayConfig` settings for a
    whole store and decides which one applies to each array the writer emits.
    Arrays are grouped by role: ``meta`` covers bookkeeping arrays (pointers
    and masks), ``core`` covers the standard ``AtomicData`` fields (positions,
    energy, forces, ...), and ``custom`` covers any user-added arrays. Each
    group defaults to a plain :class:`ZarrArrayConfig`, so an empty
    ``ZarrWriteConfig()`` is a valid "use Zarr defaults everywhere" plan.

    For finer control, ``field_overrides`` maps an individual field name to its
    own :class:`ZarrArrayConfig`; a matching override wins over the group-level
    config for that field. This lets you, for example, compress ``positions``
    differently from the rest of the core arrays while leaving everything else
    on the shared ``core`` settings. Pass the completed config to the writer to
    control on-disk compression, chunking, and sharding across the store.

    Examples
    --------
    Compress core arrays with Zstd, but give ``positions`` its own Blosc/LZ4
    codec via an override::

        >>> from zarr.codecs import ZstdCodec, BloscCodec
        >>> config = ZarrWriteConfig(
        ...     core=ZarrArrayConfig(compressors=(ZstdCodec(level=3),), chunk_size=1024),
        ...     field_overrides={
        ...         "positions": ZarrArrayConfig(compressors=(BloscCodec(cname="lz4"),))
        ...     },
        ... )

    Accept Zarr defaults for every array group::

        >>> config = ZarrWriteConfig()

    Notes
    -----
    ``meta``, ``core``, and ``custom`` each default to a fresh
    :class:`ZarrArrayConfig` via ``default_factory``, so omitting a group is
    equivalent to passing an unconfigured one. ``field_overrides`` is keyed by
    field name and takes precedence over the group config only for the fields
    it names. ``arbitrary_types_allowed`` is enabled so nested configs may hold
    native ``zarr`` codec objects.

    .. seealso::

       :ref:`zarr_compression_guide` -- choosing codecs, chunk sizes, and shard
       sizes to balance store size against read and write throughput.
    """

    meta: Annotated[
        ZarrArrayConfig,
        Field(
            default_factory=ZarrArrayConfig,
            description="Config for metadata arrays (pointers, masks).",
        ),
    ]
    core: Annotated[
        ZarrArrayConfig,
        Field(
            default_factory=ZarrArrayConfig,
            description="Config for core data arrays (positions, energy, etc.).",
        ),
    ]
    custom: Annotated[
        ZarrArrayConfig,
        Field(
            default_factory=ZarrArrayConfig,
            description="Config for user-added custom arrays.",
        ),
    ]
    field_overrides: Annotated[
        dict[str, ZarrArrayConfig],
        Field(
            default_factory=dict,
            description="Per-field overrides. Takes precedence over group-level config.",
        ),
    ]

    model_config = ConfigDict(arbitrary_types_allowed=True)


def _get_field_level(key: str) -> str:
    """Return 'atom', 'edge', or 'system' for a core field key.

    Parameters
    ----------
    key : str
        Field name.

    Returns
    -------
    str
        One of 'atom', 'edge', or 'system'.
    """
    match key:
        case k if k in AtomicData._default_node_keys:
            return "atom"
        case k if k in AtomicData._default_edge_keys:
            return "edge"
        case k if k in AtomicData._default_system_keys:
            return "system"
        case _:
            # Default to atom level for unknown keys
            return "atom"


# ---------------------------------------------------------------------------
# Gap-merge run construction
# ---------------------------------------------------------------------------
#
# Policy: merge adjacent sorted physical indices into contiguous ranges when
# the gap between them is <= *gap_threshold* (defaults to the batch size).
# This reduces the number of Zarr codec-pipeline / shard-index round trips
# — the dominant cost for random access — at the expense of reading some
# unrequested rows ("read amplification").
#
# To keep amplification bounded, each merged range is capped so that
#   span / requested_count <= max_amplification
# where *span* is `last_physical - first_physical + 1` and
# *requested_count* is the number of positions in the run.  Default
# cap is 8x, meaning we never decompress more than 8x the data we
# actually need within a single range.
_DEFAULT_MAX_AMPLIFICATION: int = 8


def _leading_storage_size(arr: Any) -> int | None:
    """Return the leading Zarr storage-object length when available."""
    metadata = getattr(arr, "metadata", None)
    chunk_grid = getattr(metadata, "chunk_grid", None)
    chunk_shape = getattr(chunk_grid, "chunk_shape", None)
    if chunk_shape is not None and len(chunk_shape) > 0:
        return int(chunk_shape[0])

    shards = getattr(arr, "shards", None)
    if shards is not None and len(shards) > 0 and shards[0] is not None:
        return int(shards[0])

    chunks = getattr(arr, "chunks", None)
    if chunks is not None and len(chunks) > 0 and chunks[0] is not None:
        return int(chunks[0])

    return None


def _chunk_span_for_slice(
    start: int, end: int, chunk_size: int
) -> tuple[int, int] | None:
    """Return inclusive leading-axis chunk span for a half-open row slice."""
    if end <= start:
        return None
    return start // chunk_size, (end - 1) // chunk_size


def _sample_chunk_spans(
    physical_idx: int,
    fields: Sequence[tuple[str, str, Any]],
    level_ptrs: Mapping[str, torch.Tensor],
) -> list[tuple[int, int, int]]:
    """Return per-field chunk spans touched by one physical sample."""
    spans: list[tuple[int, int, int]] = []

    for field_idx, (_key, level, arr) in enumerate(fields):
        ptr = level_ptrs.get(level)
        if ptr is None:
            continue
        start = int(ptr[physical_idx].item())
        end = int(ptr[physical_idx + 1].item())

        chunk_size = _leading_storage_size(arr)
        if chunk_size is None or chunk_size <= 0:
            continue
        chunk_span = _chunk_span_for_slice(start, end, chunk_size)
        if chunk_span is not None:
            spans.append((field_idx, *chunk_span))

    return spans


def _spans_overlap(
    run_spans: Mapping[int, tuple[int, int]],
    sample_spans: Sequence[tuple[int, int, int]],
) -> bool:
    """Return True when a sample touches a chunk already covered by a run."""
    for field_idx, first, last in sample_spans:
        if field_idx not in run_spans:
            continue
        run_first, run_last = run_spans[field_idx]
        if first <= run_last and last >= run_first:
            return True
    return False


def _merge_chunk_spans(
    run_spans: dict[int, tuple[int, int]],
    sample_spans: Sequence[tuple[int, int, int]],
) -> None:
    """Extend run chunk spans in-place with spans from another sample."""
    for field_idx, first, last in sample_spans:
        if field_idx not in run_spans:
            run_spans[field_idx] = (first, last)
            continue
        run_first, run_last = run_spans[field_idx]
        run_spans[field_idx] = (min(run_first, first), max(run_last, last))


def _merge_physical_runs_by_chunks(
    sorted_physical: Sequence[int],
    fields: Sequence[tuple[str, str, Any]],
    level_ptrs: Mapping[str, torch.Tensor],
    *,
    max_amplification: int = _DEFAULT_MAX_AMPLIFICATION,
) -> list[list[int]]:
    """Group physical indices while preserving Zarr chunk locality."""
    if not sorted_physical:
        return []

    gap_threshold = max(len(sorted_physical), 1)
    runs: list[list[int]] = [[0]]
    run_first_physical = sorted_physical[0]
    sample_spans = [
        _sample_chunk_spans(physical_idx, fields, level_ptrs)
        for physical_idx in sorted_physical
    ]
    run_spans: dict[int, tuple[int, int]] = {}
    _merge_chunk_spans(run_spans, sample_spans[0])

    for position in range(1, len(sorted_physical)):
        gap = sorted_physical[position] - sorted_physical[position - 1]
        span = sorted_physical[position] - run_first_physical + 1
        count = len(runs[-1]) + 1
        within_gap_policy = gap <= gap_threshold and span <= count * max_amplification
        overlaps_existing_chunk = _spans_overlap(run_spans, sample_spans[position])

        if overlaps_existing_chunk or within_gap_policy:
            runs[-1].append(position)
            _merge_chunk_spans(run_spans, sample_spans[position])
        else:
            runs.append([position])
            run_first_physical = sorted_physical[position]
            run_spans = {}
            _merge_chunk_spans(run_spans, sample_spans[position])

    return runs


def _row_indices_for_ranges(starts: Sequence[int], ends: Sequence[int]) -> np.ndarray:
    """Return concatenated row indices for a sequence of half-open ranges."""
    ranges = [
        np.arange(start, end, dtype=np.int64)
        for start, end in zip(starts, ends, strict=True)
        if end > start
    ]
    if not ranges:
        return np.empty(0, dtype=np.int64)
    return np.concatenate(ranges)


# NOTE: the generic *index*/*face* regex fallback returning -1 is local to
# the Zarr backend. No current AtomicData edge field reaches it, and the Zarr
# read paths (_slice_edge_array) reject cat_dim != 0 with a RuntimeError.
def _get_cat_dim(key: str) -> int:
    """Return concatenation dimension for a field.

    Parameters
    ----------
    key : str
        Field name.

    Returns
    -------
    int
        Concatenation dimension.
    """
    if key == "neighbor_list":
        return 0
    if bool(re.search("(index|face)", key)):
        return -1
    return 0


def _slice_edge_array(arr: Any, key: str, edge_start: int, edge_end: int) -> Any:
    """Slice an edge-level array on dim 0, rejecting non-zero cat dims.

    Parameters
    ----------
    arr : Any
        Numpy array or zarr array to slice.
    key : str
        Field name (used for error messages and cat_dim lookup).
    edge_start : int
        Start index along the edge dimension.
    edge_end : int
        End index along the edge dimension.

    Returns
    -------
    Any
        Sliced array ``arr[edge_start:edge_end]``.

    Raises
    ------
    RuntimeError
        If ``_get_cat_dim(key)`` returns anything other than 0.
    """
    cat_dim = _get_cat_dim(key)
    if cat_dim != 0:
        raise RuntimeError(
            f"Unexpected cat_dim={cat_dim} for edge field '{key}'. "
            "All edge fields should use (E, ...) layout with cat_dim=0."
        )
    return arr[edge_start:edge_end]


class AtomicDataZarrWriter:
    """Writer for serializing AtomicData into Zarr stores.

    Writes AtomicData objects into a structured Zarr store with CSR-style
    pointer arrays for variable-size graph data. Supports single writes,
    batch writes, appending, custom fields, soft-delete, and defragmentation.

    The Zarr store layout is:

    .. code-block:: text

        dataset.zarr/
        ├── meta/                       # Pointer arrays + masks
        │   ├── atoms_ptr               # int64 [N+1] — cumulative node counts
        │   ├── edges_ptr               # int64 [N+1] — cumulative edge counts
        │   ├── samples_mask            # bool [N] — False = deleted sample
        │   ├── atoms_mask              # bool [V_total] — False = deleted atom
        │   └── edges_mask              # bool [E_total] — False = deleted edge
        │
        ├── core/                       # AtomicData fields (auto-populated)
        │   ├── atomic_numbers          # int64 [V_total]
        │   ├── positions               # float32 [V_total, 3]
        │   └── ...
        │
        ├── custom/                     # User-defined arrays (optional)
        │   └── <user_key>              # any dtype, any shape
        │
        └── .zattrs                     # root metadata

    Parameters
    ----------
    store : StoreLike
        Any zarr-compatible store: filesystem path (str or Path), or a zarr
        Store instance (LocalStore, MemoryStore, FsspecStore, etc.), StorePath,
        or a dict for in-memory buffer storage.
    config : ZarrWriteConfig | Mapping[str, Any] | None
        Compression/chunking configuration. Can be a ``ZarrWriteConfig``
        instance or a dict that will be converted to one. Default is ``None``
        (use Zarr defaults).

    Attributes
    ----------
    _store : StoreLike
        The zarr store used for I/O.
    _config : ZarrWriteConfig
        The write configuration for compression and chunking.
    """

    def __init__(
        self,
        store: StoreLike,
        config: ZarrWriteConfig | Mapping[str, Any] | None = None,
    ) -> None:
        """Initialize the writer with a target store.

        Parameters
        ----------
        store : StoreLike
            Any zarr-compatible store: filesystem path (str or Path), or a zarr
            Store instance (LocalStore, MemoryStore, FsspecStore, etc.),
            StorePath, or a dict for in-memory buffer storage.
        config : ZarrWriteConfig | Mapping[str, Any] | None
            Compression/chunking configuration. Can be a ``ZarrWriteConfig``
            instance or a dict that will be converted to one. Default is ``None``
            (use Zarr defaults).
        """
        self._store: StoreLike = store
        if isinstance(config, Mapping):
            config = ZarrWriteConfig.model_validate(config)
        if config is None:
            config = ZarrWriteConfig()
        self._config = config

    def _open(self, mode: Literal["r", "r+", "w", "w-", "a"] = "r") -> zarr.Group:
        """Open the zarr store with the given mode.

        Parameters
        ----------
        mode : Literal["r", "r+", "w", "w-", "a"]
            Zarr access mode ('r', 'r+', 'w', 'w-', 'a').

        Returns
        -------
        zarr.Group
            The opened zarr group.
        """
        return zarr.open(self._store, mode=mode)  # type: ignore[return-value]

    def _store_exists(self) -> bool:
        """Check whether the store already contains data.

        For filesystem paths, checks if the path exists. For abstract stores
        (MemoryStore, FsspecStore, etc.), attempts to open in read mode and
        check for content.

        Returns
        -------
        bool
            True if the store exists and contains data.
        """
        if isinstance(self._store, (str, Path)):
            return Path(self._store).exists()
        # For abstract stores (MemoryStore, FsspecStore, etc.),
        # try opening read-only and check for content
        try:
            root = zarr.open(self._store, mode="r")  # type: ignore[call-overload]
            # If we can list any members, the store has data
            return len(list(root.group_keys())) > 0 or len(list(root.array_keys())) > 0
        except Exception:
            return False

    def _resolve_array_kwargs(
        self, key: str, group: str, data: np.ndarray, *, cat_dim: int = 0
    ) -> dict[str, Any]:
        """Resolve compression/chunking kwargs for a ``create_array`` call.

        Parameters
        ----------
        key : str
            Array name (e.g. ``"positions"``, ``"atoms_ptr"``).
        group : str
            Group name: ``"meta"``, ``"core"``, or ``"custom"``.
        data : np.ndarray
            The data to be written (used to determine chunk shape).
        cat_dim : int, optional
            The concatenation axis (variable-length dimension) for chunking.
            Defaults to 0. For ``neighbor_list`` (stored as ``[E, 2]``), use 0.

        Returns
        -------
        dict[str, Any]
            Keyword arguments to pass to ``zarr.Group.create_array``.
        """
        base_cfg: ZarrArrayConfig = getattr(self._config, group)
        cfg = self._config.field_overrides.get(key, base_cfg)

        kwargs: dict[str, Any] = {}
        if cfg.compressors is not None:
            kwargs["compressors"] = cfg.compressors
        if cfg.filters is not None:
            kwargs["filters"] = cfg.filters
        if cfg.serializer is not None:
            kwargs["serializer"] = cfg.serializer
        if cfg.chunk_size is not None:
            shape = list(data.shape)
            shape[cat_dim] = cfg.chunk_size
            kwargs["chunks"] = tuple(shape)
        if cfg.shard_size is not None:
            shape = list(data.shape)
            shape[cat_dim] = cfg.shard_size
            kwargs["shards"] = tuple(shape)
        if not cfg.write_empty_chunks:
            kwargs["config"] = {"write_empty_chunks": False}
        return kwargs

    @staticmethod
    def _custom_level_names(schema: LevelSchema) -> tuple[str, ...]:
        return tuple(name for name in schema.level_names if name not in _BUILTIN_LEVELS)

    @staticmethod
    def _schema_definitions(schema: LevelSchema) -> dict[str, dict[str, str]]:
        """Serialize custom definitions in schema registration order."""
        definitions: dict[str, dict[str, str]] = {}
        for name in AtomicDataZarrWriter._custom_level_names(schema):
            kind = schema.level_kind(name)
            if kind == "product":
                left, right = schema.product_parents[name]
                definitions[name] = {"kind": kind, "left": left, "right": right}
            else:
                definitions[name] = {"kind": kind}
        return definitions

    @staticmethod
    def _schema_from_levels(levels: Any) -> LevelSchema:
        """Rebuild the persisted level schema, rejecting unknown revisions."""
        if not isinstance(levels, Mapping) or levels.get("version") != 1:
            found = levels.get("version") if isinstance(levels, Mapping) else None
            raise ValueError(
                f"Unsupported custom-level metadata version {found}; "
                "supported version is 1"
            )
        definitions = levels.get("definitions")
        if not isinstance(definitions, Mapping):
            raise ValueError("Invalid Zarr custom level schema definitions")

        schema = LevelSchema()
        for name, definition in definitions.items():
            if not isinstance(name, str) or not isinstance(definition, Mapping):
                raise ValueError("Invalid Zarr custom level schema definition")
            kind = definition.get("kind")
            if kind not in {
                "uniform",
                "segmented",
                "product",
            }:
                raise ValueError("Invalid Zarr custom level schema definition")
            if name in _BUILTIN_LEVELS:
                raise ValueError("Custom level definitions cannot contain built-ins")
            if kind == "product":
                left, right = definition.get("left"), definition.get("right")
                if not isinstance(left, str) or not isinstance(right, str):
                    raise ValueError(f"Product level '{name}' has invalid parents")
                schema.add_product_level(name, left=left, right=right)
            else:
                schema.add_level(name, segmented=kind == "segmented")
        return schema

    @staticmethod
    def _field_items(data: Batch) -> list[tuple[str, str, torch.Tensor]]:
        """Return materialized batch fields in deterministic schema order."""
        items: list[tuple[str, str, torch.Tensor]] = []
        for level in data._storage.attr_map.level_names:
            group = data._storage.groups.get(level)
            if group is None:
                continue
            for key, value in group.items():
                if isinstance(value, torch.Tensor):
                    items.append((level, key, value))
        return items

    def _write_level_ptr(self, group: zarr.Group, name: str, ptr: torch.Tensor) -> None:
        ptr_np = self._to_numpy(ptr.to(torch.long))
        group.create_array(
            name,
            data=ptr_np,
            **self._resolve_array_kwargs(name, "meta", ptr_np),
        )

    @overload
    def write(self, data: AtomicData) -> None:  # noqa: F811
        """Write a single AtomicData."""
        self.write([data])

    @overload
    def write(self, data: list[AtomicData]) -> None:  # noqa: F811
        """Write a list of AtomicData to a new Zarr store."""
        self.write(Batch.from_data_list(data, device="cpu"))

    @overload
    def write(self, data: Batch) -> None:  # noqa: F811
        """Write a Batch to a new Zarr store.

        This is the efficient bulk-write path. Since a Batch already has
        all tensors concatenated (node/edge level) or stacked (system level),
        each field is written to zarr in a single I/O operation with no
        per-sample iteration.

        Parameters
        ----------
        batch : Batch
            Batched atomic data to write.

        Raises
        ------
        FileExistsError
            If store already exists.
        ValueError
            If batch is empty.
        """
        if self._store_exists():
            raise FileExistsError(f"Zarr store already exists at {self._store}")
        num_samples = data.num_graphs
        if num_samples == 0:
            raise ValueError("No data provided to write.")

        schema = data._storage.attr_map.clone()
        custom_levels = self._custom_level_names(schema)
        root = self._open(mode="w")
        meta_group = root.create_group("meta")
        core_group = root.create_group("core")
        root.create_group("custom")
        levels_group = root.create_group("levels") if custom_levels else None

        atoms_ptr = data.level_ptr("atoms").to(torch.long)
        edges_ptr = data.level_ptr("edges").to(torch.long)
        for key, ptr in (("atoms_ptr", atoms_ptr), ("edges_ptr", edges_ptr)):
            ptr_np = self._to_numpy(ptr)
            meta_group.create_array(
                key, data=ptr_np, **self._resolve_array_kwargs(key, "meta", ptr_np)
            )
        masks = {
            "samples_mask": np.ones(num_samples, dtype=bool),
            "atoms_mask": np.ones(int(atoms_ptr[-1]), dtype=bool),
            "edges_mask": np.ones(int(edges_ptr[-1]), dtype=bool),
        }
        for key, mask in masks.items():
            meta_group.create_array(
                key, data=mask, **self._resolve_array_kwargs(key, "meta", mask)
            )

        fields_metadata: dict[str, dict[str, str]] = {"core": {}, "custom": {}}
        for level, key, value in self._field_items(data):
            if level in _BUILTIN_LEVELS:
                stored_level = {"atoms": "atom", "edges": "edge", "system": "system"}[
                    level
                ]
                fields_metadata["core"][key] = stored_level
                if stored_level == "system" and value.dim() > 2:
                    while value.dim() > 2 and value.shape[1] == 1:
                        value = value.squeeze(1)
                array = self._to_numpy(value)
                cat_dim = _get_cat_dim(key)
                if cat_dim < 0:
                    cat_dim += array.ndim
                core_group.create_array(
                    key,
                    data=array,
                    **self._resolve_array_kwargs(key, "core", array, cat_dim=cat_dim),
                )
            else:
                if levels_group is None:
                    raise RuntimeError("Custom level storage was not initialized")
                array = self._to_numpy(value)
                level_group = levels_group.require_group(level)
                level_group.create_array(
                    key,
                    data=array,
                    **self._resolve_array_kwargs(key, "custom", array),
                )
                fields_metadata.setdefault("levels", {})[key] = level

        if custom_levels:
            ptr_group = meta_group.create_group("level_ptrs")
            if levels_group is None:
                raise RuntimeError("Custom level storage was not initialized")
            for name in custom_levels:
                if name in data.level_keys:
                    levels_group.require_group(name)
                if schema.level_kind(name) == "uniform":
                    continue
                try:
                    self._write_level_ptr(ptr_group, name, data.level_ptr(name))
                except KeyError:
                    # An unmaterialized definition has no resolved cardinality.
                    continue
            root.attrs["levels"] = {
                "version": 1,
                "definitions": self._schema_definitions(schema),
            }
        root.attrs["num_samples"] = num_samples
        root.attrs["fields"] = fields_metadata

    @dispatch
    def write(self, data: AtomicData | list[AtomicData] | Batch) -> None:  # noqa: F811
        """Write atomic data to a new Zarr store.

        Creates the Zarr store with core/, meta/, custom/ groups.
        Builds atoms_ptr, edges_ptr, and initializes all masks to True.

        If data is a Batch, calls to_data_list() first.
        If data is a single AtomicData, wraps in a list.

        Parameters
        ----------
        data : AtomicData | list[AtomicData] | Batch
            Data to write.

        Raises
        ------
        FileExistsError
            If store already exists.
        """
        pass

    @overload
    def append(self, data: AtomicData) -> None:  # noqa: F811
        """Append a single AtomicData to an existing Zarr store.

        While this dispatch is available for convenience, we recommend
        users to try and amortize I/O operations by packing multiple
        data to write, instead of one at a time. This can be achieved
        by passing either a ``Batch`` object, or a list of ``AtomicData``
        which will automatically form a batch.

        Parameters
        ----------
        data : AtomicData
            Single atomic data to append.

        Raises
        ------
        FileNotFoundError
            If store does not exist.
        """
        self.append(Batch.from_data_list([data], device=data.device))

    @overload
    def append(self, data: list[AtomicData]) -> None:  # noqa: F811
        """Append a list of AtomicData to an existing Zarr store."""
        if not data:
            return
        self.append(Batch.from_data_list(data, device=data[0].device))

    @overload
    def append(self, data: Batch) -> None:  # noqa: F811
        """Append a Batch to an existing Zarr store.

        This is the efficient bulk-append path. Since a Batch already has
        all tensors concatenated (node/edge level) or stacked (system level),
        each field is extended in a single I/O operation with no per-sample
        iteration.

        Parameters
        ----------
        data : Batch
            Batched atomic data to append.

        Raises
        ------
        FileNotFoundError
            If store does not exist.
        ValueError
            If required custom fields, pointers, dtypes, or trailing shapes
            are incompatible with the existing store. These checks complete
            before any target array is extended.
        """
        self._append_batch(data)

    @dispatch
    def append(self, data: AtomicData | list[AtomicData] | Batch) -> None:  # noqa: F811
        """Append data to an existing Zarr store.

        Extends all arrays along concatenation axis.
        Extends pointer arrays and masks.
        Updates num_samples in .zattrs.

        Parameters
        ----------
        data : AtomicData | list[AtomicData] | Batch
            Data to append.

        Raises
        ------
        FileNotFoundError
            If store does not exist.
        ValueError
            If required custom fields, pointers, dtypes, or trailing shapes
            are incompatible with the existing store.
        """
        pass

    def add_custom(
        self,
        key: str,
        data: torch.Tensor,
        level: str,
        *,
        attr_map: LevelSchema | None = None,
        level_ptrs: Mapping[str, torch.Tensor] | None = None,
    ) -> None:
        """Add a field to an existing Zarr store.

        Parameters
        ----------
        key : str
            Name for the custom array.
        data : torch.Tensor
            Tensor data. Its leading dimension must match the physical size of
            ``level`` across the complete store, including deleted samples.
        level : str
            A built-in alias or registered custom level name. Built-in fields
            are stored under ``custom/``; custom-level fields are stored under
            ``levels/<level>/``.
        attr_map : LevelSchema | None, optional
            Schema defining a new custom level and any required custom parents.
            Existing stored definitions can be reused without supplying it.
        level_ptrs : Mapping[str, torch.Tensor] | None, optional
            Complete prefix pointers for newly resolved custom segmented
            levels. Each pointer must cover every physical sample.

        Raises
        ------
        ValueError
            If the level definition, pointer, field name, or tensor shape is
            incompatible with the store. Tensor conversion and all supplied
            pointer validation complete before the store is changed.
        TypeError
            If the tensor cannot be converted to a NumPy array for Zarr I/O.
        FileNotFoundError
            If store does not exist.
        """
        if not self._store_exists():
            raise FileNotFoundError(f"Zarr store does not exist at {self._store}")
        root = self._open(mode="r+")
        meta_group = root["meta"]
        num_samples = int(root.attrs["num_samples"])
        if not isinstance(key, str) or not key:
            raise ValueError("Custom field key must be a non-empty string")
        if not isinstance(data, torch.Tensor) or data.ndim == 0:
            raise ValueError("Custom data must be a tensor with a leading dimension")
        # Convert before creating any groups or updating metadata. In
        # particular, bfloat16 tensors cannot be converted through NumPy and
        # must leave the store unchanged when rejected.
        array = self._to_numpy(data)
        resolved_level = {
            "atom": "atoms",
            "edge": "edges",
            "system": "system",
        }.get(level, level)
        fields = dict(root.attrs.get("fields", {"core": {}, "custom": {}}))
        all_keys: set[str] = set()
        for group_name in ("core", "custom"):
            if group_name in root:
                all_keys.update(root[group_name].array_keys())
        if "levels" in root:
            for level_group in root["levels"].group_keys():
                all_keys.update(root["levels"][level_group].array_keys())
        for field_metadata in fields.values():
            if isinstance(field_metadata, Mapping):
                all_keys.update(field_metadata)
        if key in all_keys:
            raise ValueError(f"Field '{key}' already exists")

        if resolved_level in _BUILTIN_LEVELS:
            expected = {
                "atoms": int(meta_group["atoms_ptr"][-1]),
                "edges": int(meta_group["edges_ptr"][-1]),
                "system": num_samples,
            }[resolved_level]
            if data.shape[0] != expected:
                raise ValueError(
                    f"Data shape[0]={data.shape[0]} does not match expected size={expected}"
                )
            root["custom"].create_array(
                key, data=array, **self._resolve_array_kwargs(key, "custom", array)
            )
            fields.setdefault("custom", {})[key] = {
                "atoms": "atom",
                "edges": "edge",
                "system": "system",
            }[resolved_level]
            root.attrs["fields"] = fields
            return

        if "levels" in root.attrs:
            schema = self._schema_from_levels(root.attrs["levels"])
        else:
            schema = LevelSchema()
        if attr_map is None:
            if resolved_level not in schema.level_kinds:
                if "levels" not in root.attrs:
                    raise ValueError(
                        f"Custom level '{level}' is not defined in this legacy store; "
                        "pass attr_map with its definition, or use 'atom', 'edge', "
                        "or 'system'."
                    )
                raise ValueError(
                    "A schema containing the custom level is required: "
                    f"expected one of {self._custom_level_names(schema)}, "
                    f"got {resolved_level}"
                )
            incoming = schema.clone()
        else:
            if resolved_level not in attr_map.level_kinds:
                raise ValueError(
                    "A schema containing the custom level is required: "
                    f"expected one of {self._custom_level_names(attr_map)}, "
                    f"got {resolved_level}"
                )
            incoming = attr_map.clone()
            stored_names = self._custom_level_names(schema)
            incoming_existing = tuple(
                name
                for name in self._custom_level_names(incoming)
                if name in stored_names
            )
            expected_existing = tuple(
                name for name in stored_names if name in incoming_existing
            )
            if incoming_existing != expected_existing:
                raise ValueError(
                    f"Custom level order mismatch: expected {expected_existing}, "
                    f"got {incoming_existing}"
                )
            for name in incoming_existing:
                if (
                    incoming.level_kind(name),
                    incoming.product_parents.get(name),
                ) != (
                    schema.level_kind(name),
                    schema.product_parents.get(name),
                ):
                    expected = (
                        schema.level_kind(name),
                        schema.product_parents.get(name),
                    )
                    actual = (
                        incoming.level_kind(name),
                        incoming.product_parents.get(name),
                    )
                    raise ValueError(
                        f"Incompatible existing custom level '{name}': "
                        f"expected {expected}, got {actual}"
                    )

        closure: list[str] = []

        def add_definition(name: str) -> None:
            if name in closure or name in _BUILTIN_LEVELS:
                return
            kind = incoming.level_kind(name)
            if kind == "product":
                left, right = incoming.product_parents[name]
                add_definition(left)
                add_definition(right)
            closure.append(name)

        add_definition(resolved_level)
        for name in closure:
            kind = incoming.level_kind(name)
            if name in schema.level_kinds:
                if schema.level_kind(name) != kind or schema.product_parents.get(
                    name
                ) != incoming.product_parents.get(name):
                    expected = (
                        schema.level_kind(name),
                        schema.product_parents.get(name),
                    )
                    actual = (kind, incoming.product_parents.get(name))
                    raise ValueError(
                        f"Incompatible existing custom level '{name}': "
                        f"expected {expected}, got {actual}"
                    )
            elif kind == "product":
                left, right = incoming.product_parents[name]
                schema.add_product_level(name, left=left, right=right)
            else:
                schema.add_level(name, segmented=kind == "segmented")
            for field in incoming.group_to_attrs.get(name, set()):
                schema.set(
                    field,
                    name,
                    dtype=incoming.dtypes.get(field),
                    is_segmented=kind != "uniform",
                )

        kind = schema.level_kind(resolved_level)
        ptr_inputs = dict(level_ptrs or {})
        ptr_cache: dict[str, torch.Tensor] = {
            "atoms": torch.from_numpy(meta_group["atoms_ptr"][:]).to(torch.long),
            "edges": torch.from_numpy(meta_group["edges_ptr"][:]).to(torch.long),
        }
        if "level_ptrs" in meta_group:
            ptr_cache.update(
                {
                    name: torch.from_numpy(meta_group["level_ptrs"][name][:]).to(
                        torch.long
                    )
                    for name in meta_group["level_ptrs"].array_keys()
                }
            )

        known_pointer_names = set(schema.level_kinds)
        known_pointer_names.update(("atoms", "edges"))
        unknown_pointers = set(ptr_inputs) - known_pointer_names
        if unknown_pointers:
            raise ValueError(
                "Pointers were supplied for unknown levels: "
                + ", ".join(sorted(unknown_pointers))
            )

        def checked_ptr(name: str, ptr: torch.Tensor) -> torch.Tensor:
            if (
                not isinstance(ptr, torch.Tensor)
                or ptr.ndim != 1
                or ptr.dtype
                not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8)
            ):
                raise ValueError(
                    f"Level pointer for '{name}' must be a 1-D integer tensor: "
                    f"expected a 1-D integer tensor, got "
                    f"shape={getattr(ptr, 'shape', None)}, "
                    f"dtype={getattr(ptr, 'dtype', None)}"
                )
            ptr = ptr.to(torch.long).cpu()
            if (
                len(ptr) != num_samples + 1
                or ptr[0] != 0
                or torch.any(ptr[1:] < ptr[:-1])
            ):
                raise ValueError(
                    f"Level pointer for '{name}' must be a full nondecreasing "
                    f"prefix pointer: expected length {num_samples + 1} starting "
                    f"at 0, got {ptr.tolist()}"
                )
            return ptr

        checked_inputs: dict[str, torch.Tensor] = {}
        for name, supplied in ptr_inputs.items():
            if name in schema.level_kinds and schema.level_kind(name) == "uniform":
                raise ValueError(f"Uniform level '{name}' cannot have a pointer")
            checked_inputs[name] = checked_ptr(name, supplied)
        ptr_inputs = checked_inputs

        def resolve_ptr(name: str) -> torch.Tensor:
            if name in ptr_cache:
                supplied = ptr_inputs.get(name)
                if supplied is not None:
                    checked_supplied = checked_ptr(name, supplied)
                    if not torch.equal(checked_supplied, ptr_cache[name]):
                        raise ValueError(
                            f"Supplied pointer for '{name}' conflicts with the "
                            f"stored pointer: expected {ptr_cache[name].tolist()}, "
                            f"got {checked_supplied.tolist()}"
                        )
                return ptr_cache[name]
            kind_ = schema.level_kind(name)
            if kind_ == "product":
                left, right = schema.product_parents[name]
                left_ptr, right_ptr = resolve_ptr(left), resolve_ptr(right)
                lengths = (left_ptr[1:] - left_ptr[:-1]) * (
                    right_ptr[1:] - right_ptr[:-1]
                )
                computed = torch.cat(
                    [torch.zeros(1, dtype=torch.long), torch.cumsum(lengths, 0)]
                )
                supplied = ptr_inputs.get(name)
                if supplied is not None:
                    checked_supplied = checked_ptr(name, supplied)
                    if not torch.equal(checked_supplied, computed):
                        raise ValueError(
                            f"Supplied product pointer for '{name}' does not match "
                            f"its parents: expected {computed.tolist()}, got "
                            f"{checked_supplied.tolist()}"
                        )
                ptr_cache[name] = computed
                return computed
            supplied = ptr_inputs.get(name)
            if supplied is None:
                raise ValueError(f"Segmented custom level '{name}' requires a pointer")
            ptr_cache[name] = checked_ptr(name, supplied)
            return ptr_cache[name]

        pointer_names = [
            name for name in closure if schema.level_kind(name) != "uniform"
        ]
        for name in pointer_names:
            resolve_ptr(name)
        expected = (
            num_samples if kind == "uniform" else int(resolve_ptr(resolved_level)[-1])
        )
        if data.shape[0] != expected:
            raise ValueError(
                f"Data shape[0]={data.shape[0]} does not match expected size={expected}"
            )

        # Complete schema and array-creation preparation before materializing
        # any groups. Both operations are local and cannot partially update
        # the store if validation fails.
        schema.set(
            key, resolved_level, dtype=data.dtype, is_segmented=kind != "uniform"
        )
        array_kwargs = self._resolve_array_kwargs(key, "custom", array)

        # All validation is complete; only now materialize groups, metadata,
        # pointers, and field.
        levels_group = root.require_group("levels")
        ptr_group = meta_group.require_group("level_ptrs")
        for name in closure:
            levels_group.require_group(name)
            if schema.level_kind(name) != "uniform" and name not in ptr_group:
                self._write_level_ptr(ptr_group, name, resolve_ptr(name))
        levels_group.require_group(resolved_level).create_array(
            key, data=array, **array_kwargs
        )
        root.attrs["levels"] = {
            "version": 1,
            "definitions": self._schema_definitions(schema),
        }
        fields.setdefault("levels", {})[key] = resolved_level
        root.attrs["fields"] = fields

    def delete(self, indices: list[int] | torch.Tensor) -> None:
        """Soft-delete samples by index.

        Sets masks to False and zeros out data slices in core/ and custom/.
        Pointer arrays are NOT modified.

        Parameters
        ----------
        indices : list[int] | torch.Tensor
            Sample indices to delete.
        """
        if not self._store_exists():
            raise FileNotFoundError(f"Zarr store does not exist at {self._store}")

        # Convert to torch tensor for consistent handling
        if isinstance(indices, list):
            indices_tensor = torch.as_tensor(indices, dtype=torch.long)
        else:
            indices_tensor = indices.to(torch.long)

        if len(indices_tensor) == 0:
            return

        root = self._open(mode="r+")
        meta_group = root["meta"]
        core_group = root["core"]

        atoms_ptr = meta_group["atoms_ptr"][:]
        edges_ptr = meta_group["edges_ptr"][:]

        samples_mask = meta_group["samples_mask"][:]
        atoms_mask = meta_group["atoms_mask"][:]
        edges_mask = meta_group["edges_mask"][:]

        fields_metadata = dict(root.attrs.get("fields", {"core": {}, "custom": {}}))
        for idx in indices_tensor:
            idx = int(idx)
            # Mark sample as deleted
            samples_mask[idx] = False

            # Get slice ranges
            atom_start, atom_end = int(atoms_ptr[idx]), int(atoms_ptr[idx + 1])
            edge_start, edge_end = int(edges_ptr[idx]), int(edges_ptr[idx + 1])

            # Zero out atoms_mask and edges_mask
            atoms_mask[atom_start:atom_end] = False
            edges_mask[edge_start:edge_end] = False

            # Zero out core fields
            for key in core_group.keys():
                level = fields_metadata.get("core", {}).get(key, _get_field_level(key))
                arr = core_group[key]

                if level == "atom":
                    self._zero_slice(arr, atom_start, atom_end, axis=0)
                elif level == "edge":
                    cat_dim = _get_cat_dim(key)
                    self._zero_slice(arr, edge_start, edge_end, axis=cat_dim)
                elif level == "system":
                    self._zero_slice(arr, idx, idx + 1, axis=0)

            # Zero out custom fields
            if "custom" in root:
                custom_group = root["custom"]
                for key in custom_group.keys():
                    level = fields_metadata.get("custom", {}).get(key, "system")
                    arr = custom_group[key]

                    if level == "atom":
                        self._zero_slice(arr, atom_start, atom_end, axis=0)
                    elif level == "edge":
                        self._zero_slice(arr, edge_start, edge_end, axis=0)
                    elif level == "system":
                        self._zero_slice(arr, idx, idx + 1, axis=0)

            if "levels" in root and "level_ptrs" in meta_group:
                levels_group = root["levels"]
                ptr_group = meta_group["level_ptrs"]
                for level_name in levels_group.group_keys():
                    if level_name not in ptr_group:
                        for key in levels_group[level_name].array_keys():
                            self._zero_slice(
                                levels_group[level_name][key], idx, idx + 1
                            )
                        continue
                    ptr = ptr_group[level_name]
                    start, end = int(ptr[idx]), int(ptr[idx + 1])
                    for key in levels_group[level_name].array_keys():
                        self._zero_slice(levels_group[level_name][key], start, end)

        # Write back masks
        meta_group["samples_mask"][:] = samples_mask
        meta_group["atoms_mask"][:] = atoms_mask
        meta_group["edges_mask"][:] = edges_mask

    def defragment(
        self, config: ZarrWriteConfig | Mapping[str, Any] | None = None
    ) -> None:
        """Rewrite store excluding deleted samples.

        Rebuilds all arrays, pointer arrays, and resets all masks to True.

        Parameters
        ----------
        config : ZarrWriteConfig | Mapping[str, Any] | None
            Optional new write configuration for the rebuilt arrays. When
            provided, also updates the writer's stored config for future
            operations. When ``None``, reuses the existing writer config.
        """
        if config is not None:
            if isinstance(config, Mapping):
                config = ZarrWriteConfig.model_validate(config)
            self._config = config
        if not self._store_exists():
            raise FileNotFoundError(f"Zarr store does not exist at {self._store}")

        root = self._open(mode="r")
        meta_group = root["meta"]
        core_group = root["core"]

        atoms_ptr = meta_group["atoms_ptr"][:]
        edges_ptr = meta_group["edges_ptr"][:]
        samples_mask = meta_group["samples_mask"][:]

        fields_metadata = dict(root.attrs.get("fields", {"core": {}, "custom": {}}))
        levels_metadata = root.attrs.get("levels")
        custom_level_arrays: dict[str, dict[str, np.ndarray]] = {}
        custom_level_ptrs: dict[str, np.ndarray] = {}
        if "levels" in root:
            custom_level_arrays = {
                name: {
                    key: root["levels"][name][key][:]
                    for key in root["levels"][name].array_keys()
                }
                for name in root["levels"].group_keys()
            }
        if "level_ptrs" in meta_group:
            custom_level_ptrs = {
                name: meta_group["level_ptrs"][name][:]
                for name in meta_group["level_ptrs"].array_keys()
            }
        # Snapshot descriptors before the all-deleted path reopens the store
        # in ``mode="w"``. Zarr group/array handles become empty after that
        # replacement, so their shapes and dtypes must be retained here
        # without copying payloads that will not be reused.
        core_snapshot = {
            key: (core_group[key].shape, core_group[key].dtype)
            for key in core_group.array_keys()
        }
        custom_snapshot = (
            {
                key: (root["custom"][key].shape, root["custom"][key].dtype)
                for key in root["custom"].array_keys()
            }
            if "custom" in root
            else {}
        )

        # Find active sample indices
        active_indices = np.where(samples_mask)[0]

        if len(active_indices) == 0:
            new_root = self._open(mode="w")
            new_meta = new_root.create_group("meta")
            new_core = new_root.create_group("core")
            new_custom = new_root.create_group("custom")
            empty_meta = {
                "atoms_ptr": np.zeros(1, dtype=np.int64),
                "edges_ptr": np.zeros(1, dtype=np.int64),
                "samples_mask": np.zeros(0, dtype=bool),
                "atoms_mask": np.zeros(0, dtype=bool),
                "edges_mask": np.zeros(0, dtype=bool),
            }
            for key, array in empty_meta.items():
                new_meta.create_array(
                    key,
                    data=array,
                    **self._resolve_array_kwargs(key, "meta", array),
                )
            for key, (source_shape, source_dtype) in core_snapshot.items():
                axis = _get_cat_dim(key)
                if axis < 0:
                    axis += len(source_shape)
                empty_shape = list(source_shape)
                empty_shape[axis] = 0
                empty = np.empty(tuple(empty_shape), dtype=source_dtype)
                new_core.create_array(
                    key,
                    data=empty,
                    **self._resolve_array_kwargs(key, "core", empty),
                )
            for key, (source_shape, source_dtype) in custom_snapshot.items():
                empty = np.empty((0, *source_shape[1:]), dtype=source_dtype)
                new_custom.create_array(
                    key,
                    data=empty,
                    **self._resolve_array_kwargs(key, "custom", empty),
                )
            new_root.attrs["num_samples"] = 0
            new_root.attrs["fields"] = fields_metadata
            if levels_metadata is not None:
                ptr_group = new_meta.create_group("level_ptrs")
                new_levels = new_root.create_group("levels")
                for name, arrays in custom_level_arrays.items():
                    group = new_levels.create_group(name)
                    for key, array in arrays.items():
                        empty = np.empty((0, *array.shape[1:]), dtype=array.dtype)
                        group.create_array(
                            key,
                            data=empty,
                            **self._resolve_array_kwargs(key, "custom", empty),
                        )
                for name, ptr in custom_level_ptrs.items():
                    zero = np.zeros(1, dtype=ptr.dtype)
                    ptr_group.create_array(
                        name,
                        data=zero,
                        **self._resolve_array_kwargs(name, "meta", zero),
                    )
                new_root.attrs["levels"] = levels_metadata
            return

        # Collect active data for each field
        new_core_data: dict[str, list[np.ndarray]] = {
            key: [] for key in core_group.keys()
        }
        new_custom_data: dict[str, list[np.ndarray]] = {}
        new_level_data: dict[str, dict[str, list[np.ndarray]]] = {
            name: {key: [] for key in arrays}
            for name, arrays in custom_level_arrays.items()
        }
        new_level_lengths: dict[str, list[int]] = {
            name: [] for name in custom_level_ptrs
        }

        if "custom" in root:
            custom_group = root["custom"]
            new_custom_data = {key: [] for key in custom_group.keys()}

        new_num_nodes: list[int] = []
        new_num_edges: list[int] = []

        # Pre-read all arrays once to avoid re-reading per sample
        core_arrays = {key: core_group[key][:] for key in core_group.keys()}
        custom_arrays: dict[str, np.ndarray] = {}
        if "custom" in root:
            custom_arrays = {
                key: root["custom"][key][:] for key in root["custom"].keys()
            }

        for idx in active_indices:
            idx = int(idx)
            atom_start, atom_end = int(atoms_ptr[idx]), int(atoms_ptr[idx + 1])
            edge_start, edge_end = int(edges_ptr[idx]), int(edges_ptr[idx + 1])

            new_num_nodes.append(atom_end - atom_start)
            new_num_edges.append(edge_end - edge_start)

            for key in core_group.keys():
                level = fields_metadata.get("core", {}).get(key, _get_field_level(key))
                arr = core_arrays[key]

                if level == "atom":
                    new_core_data[key].append(arr[atom_start:atom_end])
                elif level == "edge":
                    new_core_data[key].append(
                        _slice_edge_array(arr, key, edge_start, edge_end)
                    )
                elif level == "system":
                    # System level: index by sample
                    new_core_data[key].append(arr[idx : idx + 1])

            if custom_arrays:
                for key in custom_arrays:
                    level = fields_metadata.get("custom", {}).get(key, "system")
                    arr = custom_arrays[key]

                    if level == "atom":
                        new_custom_data[key].append(arr[atom_start:atom_end])
                    elif level == "edge":
                        new_custom_data[key].append(
                            _slice_edge_array(arr, key, edge_start, edge_end)
                        )
                    elif level == "system":
                        new_custom_data[key].append(arr[idx : idx + 1])

            for level_name, ptr in custom_level_ptrs.items():
                new_level_lengths[level_name].append(int(ptr[idx + 1]) - int(ptr[idx]))
            for level_name, arrays in custom_level_arrays.items():
                if level_name in custom_level_ptrs:
                    ptr = custom_level_ptrs[level_name]
                    start, end = int(ptr[idx]), int(ptr[idx + 1])
                else:
                    start, end = idx, idx + 1
                for key, array in arrays.items():
                    new_level_data[level_name][key].append(array[start:end])

        # Clear store and create new structure (mode="w" clears existing data)
        new_root = self._open(mode="w")
        new_meta = new_root.create_group("meta")
        new_core = new_root.create_group("core")
        new_custom = new_root.create_group("custom")
        new_levels = (
            new_root.create_group("levels") if levels_metadata is not None else None
        )

        # Build new pointer arrays
        new_atoms_ptr = np.array([0] + list(np.cumsum(new_num_nodes)), dtype=np.int64)
        new_edges_ptr = np.array([0] + list(np.cumsum(new_num_edges)), dtype=np.int64)

        new_total_atoms = int(new_atoms_ptr[-1])
        new_total_edges = int(new_edges_ptr[-1])
        new_num_samples = len(active_indices)

        new_samples_mask = np.ones(new_num_samples, dtype=np.bool_)
        new_atoms_mask = np.ones(new_total_atoms, dtype=np.bool_)
        new_edges_mask = np.ones(new_total_edges, dtype=np.bool_)

        new_meta.create_array(
            "atoms_ptr",
            data=new_atoms_ptr,
            **self._resolve_array_kwargs("atoms_ptr", "meta", new_atoms_ptr),
        )
        new_meta.create_array(
            "edges_ptr",
            data=new_edges_ptr,
            **self._resolve_array_kwargs("edges_ptr", "meta", new_edges_ptr),
        )
        new_meta.create_array(
            "samples_mask",
            data=new_samples_mask,
            **self._resolve_array_kwargs("samples_mask", "meta", new_samples_mask),
        )
        new_meta.create_array(
            "atoms_mask",
            data=new_atoms_mask,
            **self._resolve_array_kwargs("atoms_mask", "meta", new_atoms_mask),
        )
        new_meta.create_array(
            "edges_mask",
            data=new_edges_mask,
            **self._resolve_array_kwargs("edges_mask", "meta", new_edges_mask),
        )

        # Concatenate and write core arrays
        for key, arrays in new_core_data.items():
            if arrays:
                cat_dim = _get_cat_dim(key)
                concatenated = np.concatenate(arrays, axis=cat_dim)
                resolved_cat_dim = (
                    cat_dim if cat_dim >= 0 else cat_dim + concatenated.ndim
                )
                new_core.create_array(
                    key,
                    data=concatenated,
                    **self._resolve_array_kwargs(
                        key, "core", concatenated, cat_dim=resolved_cat_dim
                    ),
                )

        # Concatenate and write custom arrays
        for key, arrays in new_custom_data.items():
            if arrays:
                concatenated = np.concatenate(arrays, axis=0)
                new_custom.create_array(
                    key,
                    data=concatenated,
                    **self._resolve_array_kwargs(key, "custom", concatenated),
                )

        if new_levels is not None:
            ptr_group = new_meta.create_group("level_ptrs")
            for level_name, arrays in new_level_data.items():
                group = new_levels.create_group(level_name)
                for key, parts in arrays.items():
                    source = custom_level_arrays[level_name][key]
                    rebuilt = (
                        np.concatenate(parts, axis=0)
                        if parts
                        else np.empty((0, *source.shape[1:]), dtype=source.dtype)
                    )
                    group.create_array(
                        key,
                        data=rebuilt,
                        **self._resolve_array_kwargs(key, "custom", rebuilt),
                    )
            for level_name, lengths in new_level_lengths.items():
                rebuilt_ptr = np.array([0, *np.cumsum(lengths)], dtype=np.int64)
                ptr_group.create_array(
                    level_name,
                    data=rebuilt_ptr,
                    **self._resolve_array_kwargs(level_name, "meta", rebuilt_ptr),
                )

        # Update metadata
        new_root.attrs["num_samples"] = new_num_samples
        new_root.attrs["fields"] = fields_metadata
        if levels_metadata is not None:
            new_root.attrs["levels"] = levels_metadata

    def _append_batch(self, data: Batch) -> None:
        """Append one batch after validating custom-store compatibility."""
        if not self._store_exists():
            raise FileNotFoundError(f"Zarr store does not exist at {self._store}")
        if data.num_graphs == 0:
            return

        root = self._open(mode="r+")
        meta_group = root["meta"]
        core_group = root["core"]
        schema = data._storage.attr_map.clone()
        stored_custom = "levels" in root.attrs
        custom_levels = self._custom_level_names(schema)
        if stored_custom:
            stored_schema = self._schema_from_levels(root.attrs["levels"])
            stored_names = self._custom_level_names(stored_schema)
            if custom_levels != stored_names:
                raise ValueError(
                    f"Custom level order mismatch: expected {stored_names}, "
                    f"got {custom_levels}"
                )
            for name in stored_names:
                if (
                    schema.level_kind(name),
                    schema.product_parents.get(name),
                ) != (
                    stored_schema.level_kind(name),
                    stored_schema.product_parents.get(name),
                ):
                    expected = (
                        stored_schema.level_kind(name),
                        stored_schema.product_parents.get(name),
                    )
                    actual = (
                        schema.level_kind(name),
                        schema.product_parents.get(name),
                    )
                    raise ValueError(
                        f"Custom append has incompatible level '{name}': "
                        f"expected {expected}, got {actual}"
                    )
        elif custom_levels:
            raise ValueError("Cannot append custom levels to a legacy Zarr store")

        source_fields: dict[str, dict[str, torch.Tensor]] = {}
        for level, key, value in self._field_items(data):
            source_fields.setdefault(level, {})[key] = value

        # Materialize required incoming fields before any target array is
        # resized. Additional fields in legacy stores remain permissively
        # ignored, including fields whose dtype cannot be converted for Zarr.
        source_arrays: dict[str, dict[str, np.ndarray]] = {}

        def materialize_source(level: str, key: str) -> np.ndarray:
            try:
                value = source_fields[level][key]
            except KeyError as exc:
                raise ValueError(
                    f"Required source field '{key}' is missing at level '{level}'"
                ) from exc
            array = self._to_numpy(value)
            source_arrays.setdefault(level, {})[key] = array
            return array

        source_ptrs: dict[str, torch.Tensor] = {}
        for name in custom_levels:
            if schema.level_kind(name) == "uniform":
                continue
            try:
                source_ptrs[name] = data.level_ptr(name).to(torch.long)
            except KeyError:
                pass

        def checked_source_ptr(name: str, pointer: torch.Tensor) -> np.ndarray:
            pointer_np = self._to_numpy(pointer)
            if (
                pointer_np.ndim != 1
                or not np.issubdtype(pointer_np.dtype, np.integer)
                or len(pointer_np) != data.num_graphs + 1
                or pointer_np[0] != 0
                or np.any(pointer_np[1:] < pointer_np[:-1])
            ):
                raise ValueError(
                    f"Custom pointer '{name}' must be a full nondecreasing prefix "
                    f"pointer: expected integer shape ({data.num_graphs + 1},), "
                    f"got dtype={pointer_np.dtype}, shape={pointer_np.shape}, "
                    f"values={pointer_np.tolist()}"
                )
            return pointer_np.astype(np.int64, copy=False)

        source_ptr_arrays = {
            name: checked_source_ptr(name, pointer)
            for name, pointer in source_ptrs.items()
        }

        fields_metadata_raw = root.attrs.get("fields", {"core": {}, "custom": {}})
        if not isinstance(fields_metadata_raw, Mapping):
            raise ValueError("Invalid Zarr fields metadata")
        fields_metadata: dict[str, dict[str, str]] = {
            name: dict(values)
            for name, values in fields_metadata_raw.items()
            if isinstance(values, Mapping)
        }
        if set(fields_metadata) != set(fields_metadata_raw):
            raise ValueError("Invalid Zarr fields metadata")

        if stored_custom:
            levels_group = root["levels"]
            target_ptrs = (
                set(meta_group["level_ptrs"].array_keys())
                if "level_ptrs" in meta_group
                else set()
            )
            if set(source_ptrs) != target_ptrs:
                raise ValueError(
                    "Custom resolved pointer levels mismatch: "
                    f"expected {sorted(target_ptrs)}, got {sorted(source_ptrs)}"
                )
            for name in custom_levels:
                target_fields = (
                    set(levels_group[name].array_keys())
                    if name in levels_group
                    else set()
                )
                actual_fields = set(source_fields.get(name, {}))
                if actual_fields != target_fields:
                    raise ValueError(
                        f"Custom field set mismatch for level '{name}': "
                        f"expected {sorted(target_fields)}, got "
                        f"{sorted(actual_fields)}"
                    )
                for key in source_fields.get(name, {}):
                    materialize_source(name, key)
                    target = levels_group[name][key]
                    if (
                        np.dtype(source_arrays[name][key].dtype) != target.dtype
                        or source_arrays[name][key].shape[1:] != target.shape[1:]
                    ):
                        raise ValueError(
                            f"Custom append field '{key}' has incompatible dtype or "
                            f"trailing shape: expected dtype={target.dtype}, "
                            f"shape[1:]={target.shape[1:]}, got "
                            f"dtype={source_arrays[name][key].dtype}, "
                            f"shape[1:]={source_arrays[name][key].shape[1:]}"
                        )

            for name in custom_levels:
                fields = source_arrays.get(name, {})
                if not fields:
                    continue
                if schema.level_kind(name) == "uniform":
                    expected_size = data.num_graphs
                elif name in source_ptr_arrays:
                    expected_size = int(source_ptr_arrays[name][-1])
                else:
                    raise ValueError(
                        f"Custom append requires a resolved pointer for '{name}': "
                        "expected a pointer in the incoming batch, got none"
                    )
                for key, array in fields.items():
                    target = levels_group[name][key]
                    if array.shape[0] != expected_size:
                        raise ValueError(
                            f"Custom append field '{key}' does not match its level "
                            f"pointer: expected {expected_size}, got {array.shape[0]}"
                        )

        # Root custom arrays are the legacy extension point and remain
        # permissive about additional incoming fields. Existing arrays are
        # required, however, because otherwise their appended rows would be
        # silently missing and the store would become unreadable.
        root_custom_appends: dict[str, np.ndarray] = {}
        custom_fields = fields_metadata.get("custom", {})
        for key in root["custom"].array_keys():
            level = custom_fields.get(key, "system")
            source_level = {"atom": "atoms", "edge": "edges", "system": "system"}.get(
                level
            )
            if source_level is None:
                raise ValueError(f"Custom field '{key}' has an invalid level")
            if key not in source_fields.get(source_level, {}):
                raise ValueError(
                    f"Custom append requires existing field '{key}' at level '{level}'"
                )
            array = materialize_source(source_level, key)
            target = root["custom"][key]
            if (
                array.ndim != target.ndim
                or array.shape[1:] != target.shape[1:]
                or np.dtype(array.dtype) != target.dtype
            ):
                raise ValueError(
                    f"Custom append field '{key}' has incompatible dtype or trailing "
                    f"shape: expected dtype={target.dtype}, "
                    f"shape[1:]={target.shape[1:]}, got dtype={array.dtype}, "
                    f"shape[1:]={array.shape[1:]}"
                )
            root_custom_appends[key] = array

        old_num_samples = int(root.attrs["num_samples"])
        old_atoms_ptr = torch.from_numpy(meta_group["atoms_ptr"][:]).to(torch.long)
        old_edges_ptr = torch.from_numpy(meta_group["edges_ptr"][:]).to(torch.long)
        new_atoms_ptr = data.level_ptr("atoms").to(torch.long)[1:] + old_atoms_ptr[-1]
        new_edges_ptr = data.level_ptr("edges").to(torch.long)[1:] + old_edges_ptr[-1]
        self._extend_array(meta_group["atoms_ptr"], self._to_numpy(new_atoms_ptr))
        self._extend_array(meta_group["edges_ptr"], self._to_numpy(new_edges_ptr))
        self._extend_array(
            meta_group["samples_mask"], np.ones(data.num_graphs, dtype=bool)
        )
        self._extend_array(
            meta_group["atoms_mask"],
            np.ones(int(new_atoms_ptr[-1] - old_atoms_ptr[-1]), dtype=bool),
        )
        self._extend_array(
            meta_group["edges_mask"],
            np.ones(int(new_edges_ptr[-1] - old_edges_ptr[-1]), dtype=bool),
        )

        # Preserve the legacy built-in append route. Custom fields above are
        # fully converted and checked before mutation; built-in fields retain
        # their historical conversion and assignment behavior.
        for key in core_group.array_keys():
            level = fields_metadata.get("core", {}).get(key, _get_field_level(key))
            source_level = {"atom": "atoms", "edge": "edges", "system": "system"}[level]
            value = source_fields.get(source_level, {}).get(key)
            if value is None:
                continue
            if level == "system" and value.dim() > 2:
                while value.dim() > 2 and value.shape[1] == 1:
                    value = value.squeeze(1)
            self._extend_array(
                core_group[key], self._to_numpy(value), axis=_get_cat_dim(key)
            )

        for key, array in root_custom_appends.items():
            self._extend_array(root["custom"][key], array)

        if stored_custom:
            ptr_group = meta_group["level_ptrs"]
            for name, ptr in source_ptr_arrays.items():
                old_ptr = int(ptr_group[name][-1])
                self._extend_array(ptr_group[name], ptr[1:] + old_ptr)
            levels_group = root["levels"]
            for name, fields in source_arrays.items():
                if name not in custom_levels:
                    continue
                for key, value in fields.items():
                    self._extend_array(levels_group[name][key], value)
        root.attrs["num_samples"] = old_num_samples + data.num_graphs

    @staticmethod
    def _to_numpy(tensor: torch.Tensor) -> np.ndarray:
        """Convert a torch tensor to numpy for zarr I/O.

        Parameters
        ----------
        tensor : torch.Tensor
            Tensor to convert.

        Returns
        -------
        np.ndarray
            Numpy array for zarr storage.
        """
        return tensor.detach().cpu().numpy()

    @staticmethod
    def _extend_array(arr: zarr.Array, data: np.ndarray, axis: int = 0) -> None:
        """Extend a zarr array along an axis.

        Parameters
        ----------
        arr : zarr.Array
            Zarr array to extend.
        data : np.ndarray
            Data to append.
        axis : int
            Axis along which to extend.
        """
        old_shape = arr.shape
        new_len = data.shape[axis]

        # Build new shape
        new_shape = list(old_shape)
        new_shape[axis] = old_shape[axis] + new_len

        # Resize array
        arr.resize(tuple(new_shape))

        # Write new data
        slices: list[slice | int] = [slice(None)] * len(old_shape)
        slices[axis] = slice(old_shape[axis], new_shape[axis])
        arr[tuple(slices)] = data

    @staticmethod
    def _zero_slice(arr: zarr.Array, start: int, end: int, axis: int = 0) -> None:
        """Zero out a slice of a zarr array.

        Parameters
        ----------
        arr : zarr.Array
            Zarr array to modify.
        start : int
            Start index.
        end : int
            End index.
        axis : int
            Axis along which to slice.
        """
        if start >= end:
            return

        slices: list[slice | int] = [slice(None)] * len(arr.shape)
        slices[axis] = slice(start, end)

        # Create zeros with correct shape
        shape = list(arr.shape)
        shape[axis] = end - start
        zeros = np.zeros(shape, dtype=arr.dtype)

        arr[tuple(slices)] = zeros


class AtomicDataZarrReader(Reader):
    """Reader for loading AtomicData from Zarr stores.

    This reader provides random-access loading of
    AtomicData samples from Zarr stores created by :class:`AtomicDataZarrWriter`.
    It supports soft-deleted samples via the samples_mask and provides
    efficient random access using pointer arrays.

    The Zarr store layout expected is:

    .. code-block:: text

        dataset.zarr/
        ├── meta/                       # Pointer arrays + masks
        │   ├── atoms_ptr               # int64 [N+1] — cumulative node counts
        │   ├── edges_ptr               # int64 [N+1] — cumulative edge counts
        │   └── samples_mask            # bool [N] — False = deleted sample
        │
        ├── core/                       # AtomicData fields
        │   ├── atomic_numbers          # int64 [V_total]
        │   ├── positions               # float32 [V_total, 3]
        │   └── ...
        │
        └── custom/                     # User-defined arrays (optional)

    Parameters
    ----------
    store : StoreLike
        Any zarr-compatible store: filesystem path (str or Path), or a zarr
        Store instance (LocalStore, MemoryStore, FsspecStore, etc.), StorePath,
        or a dict for in-memory buffer storage.
    pin_memory : bool, default=False
        If True, place tensors in pinned (page-locked) memory for faster
        async CPU→GPU transfers.
    include_index_in_metadata : bool, default=True
        If True, include sample index in the metadata dict.

    Attributes
    ----------
    _store : StoreLike
        The underlying zarr store reference.

    Examples
    --------
    >>> from nvalchemi.data.datapipes.backends.zarr import AtomicDataZarrReader  # doctest: +SKIP
    >>> reader = AtomicDataZarrReader(store="dataset.zarr")  # doctest: +SKIP
    >>> data_dict, metadata = reader[0]  # returns dict and metadata  # doctest: +SKIP
    >>> atomic_data = AtomicDataZarrReader.to_atomic_data(data_dict)  # doctest: +SKIP
    """

    def __init__(
        self,
        store: StoreLike,
        *,
        pin_memory: bool = False,
        include_index_in_metadata: bool = True,
    ) -> None:
        """Initialize the reader with a Zarr store.

        Parameters
        ----------
        store : StoreLike
            Any zarr-compatible store: filesystem path (str or Path), or a zarr
            Store instance (LocalStore, MemoryStore, FsspecStore, etc.), StorePath,
            or a dict for in-memory buffer storage.
        pin_memory : bool, default=False
            If True, place tensors in pinned (page-locked) memory.
        include_index_in_metadata : bool, default=True
            If True, include sample index in the metadata dict.

        Raises
        ------
        FileNotFoundError
            If the Zarr store does not exist (for filesystem paths).
        ValueError
            If the store is missing required groups (meta, core).
        """
        super().__init__(
            pin_memory=pin_memory,
            include_index_in_metadata=include_index_in_metadata,
        )

        self._store: StoreLike = store

        # For filesystem paths, provide a friendly existence check
        if isinstance(store, (str, Path)) and not Path(store).exists():
            raise FileNotFoundError(f"Zarr store does not exist at {store}")

        # Open the Zarr store in read mode
        self._root = zarr.open(self._store, mode="r")

        # Validate store structure
        if "meta" not in self._root:
            raise ValueError(f"Zarr store at {self._store} is missing 'meta' group")
        if "core" not in self._root:
            raise ValueError(f"Zarr store at {self._store} is missing 'core' group")

        # Load cached state from the store
        self.refresh()

    def refresh(self) -> None:
        """Reload cached pointer arrays, masks, and metadata from the store.

        Call this method after external modifications to the Zarr store
        (e.g., appending or deleting samples via :class:`AtomicDataZarrWriter`)
        to ensure the reader reflects the current state of the data.

        Raises
        ------
        RuntimeError
            If the reader has been closed.
        ValueError
            If a versioned custom-level layout is malformed.
        """
        if self._root is None:
            raise RuntimeError("Cannot refresh a closed reader.")

        # Build every piece of state locally. A malformed external update must
        # not leave a previously usable reader half-refreshed.
        root = zarr.open(self._store, mode="r")
        if "meta" not in root or "core" not in root:
            raise ValueError("Zarr store must contain 'meta' and 'core' groups")
        meta_group = root["meta"]
        atoms_ptr = torch.from_numpy(meta_group["atoms_ptr"][:]).to(torch.long)
        edges_ptr = torch.from_numpy(meta_group["edges_ptr"][:]).to(torch.long)
        samples_mask = torch.from_numpy(meta_group["samples_mask"][:]).to(torch.bool)
        num_samples = len(samples_mask)

        try:
            fields_metadata_raw = root.attrs.get("fields", {"core": {}, "custom": {}})
            if not isinstance(fields_metadata_raw, Mapping):
                raise TypeError("fields metadata must be a mapping")
            fields_metadata: dict[str, dict[str, str]] = {
                name: dict(values)
                for name, values in fields_metadata_raw.items()
                if isinstance(values, Mapping)
            }
            if set(fields_metadata) != set(fields_metadata_raw):
                raise TypeError("fields metadata entries must be mappings")
        except (TypeError, ValueError) as exc:
            raise ValueError("Invalid Zarr fields metadata") from exc

        level_schema: LevelSchema | None = None
        level_ptrs: dict[str, torch.Tensor] = {}
        levels = root.attrs.get("levels")
        if levels is not None:
            try:
                level_schema = AtomicDataZarrWriter._schema_from_levels(levels)
            except (KeyError, TypeError) as exc:
                raise ValueError("Invalid Zarr custom level schema") from exc
            if "levels" not in root:
                raise ValueError(
                    "Custom Zarr level metadata is missing its levels group"
                )

            levels_group = root["levels"]
            registered_names = set(
                AtomicDataZarrWriter._custom_level_names(level_schema)
            )
            stored_names = set(levels_group.group_keys())
            unknown_groups = stored_names - registered_names
            if unknown_groups:
                raise ValueError(
                    "Stored field group(s) are not registered: "
                    + ", ".join(sorted(unknown_groups))
                )

            level_fields = fields_metadata.get("levels", {})
            if not isinstance(level_fields, Mapping):
                raise ValueError("Invalid custom-level field metadata")
            existing_fields = set(fields_metadata.get("core", {})) | set(
                fields_metadata.get("custom", {})
            )
            registered_fields: set[str] = set()
            for level_name in stored_names:
                level_group = levels_group[level_name]
                kind = level_schema.level_kind(level_name)
                for key in level_group.array_keys():
                    if key in registered_fields or key in existing_fields:
                        raise ValueError(f"Duplicate stored field '{key}'")
                    if level_fields.get(key) != level_name:
                        raise ValueError(
                            f"Custom field '{key}' is missing level metadata"
                        )
                    array = level_group[key]
                    try:
                        dtype = torch.from_numpy(np.empty(0, dtype=array.dtype)).dtype
                    except (TypeError, ValueError) as exc:
                        raise ValueError(
                            f"Custom field '{key}' has an unsupported dtype"
                        ) from exc
                    level_schema.set(
                        key,
                        level_name,
                        dtype=dtype,
                        is_segmented=kind != "uniform",
                    )
                    registered_fields.add(key)
                    if kind == "uniform" and array.shape[0] != num_samples:
                        raise ValueError(
                            f"Uniform custom field '{key}' must have one row per "
                            f"sample: expected {num_samples}, got {array.shape[0]}"
                        )

            stale_fields = set(level_fields) - registered_fields
            if stale_fields:
                raise ValueError(
                    "Custom-level metadata names missing arrays: "
                    + ", ".join(sorted(stale_fields))
                )

            ptr_group = meta_group.get("level_ptrs")
            if ptr_group is not None:
                for name in ptr_group.array_keys():
                    if name not in registered_names:
                        raise ValueError(
                            f"Pointer '{name}' is not a registered custom level"
                        )
                    if level_schema.level_kind(name) == "uniform":
                        raise ValueError(
                            f"Uniform level '{name}' cannot have a pointer"
                        )
                    pointer_array = ptr_group[name]
                    pointer = pointer_array[:]
                    if not np.issubdtype(pointer.dtype, np.integer):
                        raise ValueError(
                            f"Pointer '{name}' must have an integer dtype: "
                            f"expected an integer dtype, got {pointer.dtype}"
                        )
                    if (
                        pointer.ndim != 1
                        or len(pointer) != num_samples + 1
                        or pointer[0] != 0
                        or np.any(pointer[1:] < pointer[:-1])
                    ):
                        raise ValueError(
                            f"Pointer '{name}' must be a full nondecreasing prefix "
                            f"pointer: expected shape ({num_samples + 1},), "
                            f"starting at 0 with nondecreasing values, got "
                            f"shape={pointer.shape}, values={pointer.tolist()}"
                        )
                    level_ptrs[name] = torch.from_numpy(pointer).to(torch.long)

            for level_name in stored_names:
                kind = level_schema.level_kind(level_name)
                level_group = levels_group[level_name]
                pointer = level_ptrs.get(level_name)
                if kind != "uniform" and level_group.array_keys() and pointer is None:
                    raise ValueError(
                        f"Segmented custom level '{level_name}' is missing its pointer"
                    )
                if pointer is not None:
                    expected_size = int(pointer[-1])
                    for key in level_group.array_keys():
                        if level_group[key].shape[0] != expected_size:
                            raise ValueError(
                                f"Custom field '{key}' does not match pointer "
                                f"'{level_name}': expected {expected_size}, got "
                                f"{level_group[key].shape[0]}"
                            )

            for name in registered_names:
                if level_schema.level_kind(name) != "product":
                    continue
                pointer = level_ptrs.get(name)
                if pointer is None:
                    continue
                left, right = level_schema.product_parents[name]
                left_ptr = {
                    "atoms": atoms_ptr,
                    "edges": edges_ptr,
                }.get(left, level_ptrs.get(left))
                right_ptr = {
                    "atoms": atoms_ptr,
                    "edges": edges_ptr,
                }.get(right, level_ptrs.get(right))
                if left_ptr is None or right_ptr is None:
                    raise ValueError(
                        f"Product level '{name}' has unresolved parent pointers"
                    )
                lengths = (left_ptr[1:] - left_ptr[:-1]) * (
                    right_ptr[1:] - right_ptr[:-1]
                )
                expected = torch.cat(
                    [torch.zeros(1, dtype=torch.long), torch.cumsum(lengths, 0)]
                )
                if not torch.equal(pointer, expected):
                    raise ValueError(
                        f"Product pointer '{name}' does not match its parent pointers: "
                        f"expected {expected.tolist()}, got {pointer.tolist()}"
                    )

        # Swap only after all prospective state and layout checks succeed.
        self._root = root
        self._atoms_ptr = atoms_ptr
        self._edges_ptr = edges_ptr
        self._samples_mask = samples_mask
        self._active_indices = torch.where(samples_mask)[0]
        self._fields_metadata = fields_metadata
        self._level_schema = level_schema
        self._level_ptrs = level_ptrs
        self._metadata_revision += 1

    @property
    def field_levels(self) -> dict[str, str]:
        """Per-field level classification from store metadata.

        Returns
        -------
        dict[str, str]
            Mapping of field name to a built-in alias or registered custom
            level name.
        """
        flat: dict[str, str] = {}
        for fields in self._fields_metadata.values():
            flat.update(fields)
        return flat

    @property
    def level_schema(self) -> LevelSchema | None:
        """Return the persisted custom-level schema.

        Returns
        -------
        LevelSchema | None
            An independent schema clone for a versioned custom-level store, or
            ``None`` for a legacy store.
        """
        return self._level_schema.clone() if self._level_schema is not None else None

    @property
    def num_samples(self) -> int:
        """Number of samples the store holds, soft-deleted ones included.

        Returns
        -------
        int
            Length of ``meta/samples_mask``; ``len(reader)`` counts only the
            active samples.
        """
        return int(self._samples_mask.numel())

    def field_array(self, field: str) -> zarr.Array:
        """Return the Zarr array backing *field*.

        Parameters
        ----------
        field : str
            Name of a field the store holds, at any level.

        Returns
        -------
        zarr.Array
            The array under ``core/``, ``custom/``, or ``levels/<level>/``.

        Raises
        ------
        KeyError
            If the store holds no array named *field*.
        RuntimeError
            If the reader has been closed.
        """
        for key, _, array in self._field_entries():
            if key == field:
                return array
        raise KeyError(
            f"Field {field!r} is not in the store; stored fields are "
            f"{sorted(key for key, _, _ in self._field_entries())!r}."
        )

    def schema(self) -> dict[str, FieldSchema]:
        """Return the level, dtype, and row shape of every field the store holds.

        Dtypes come from the array metadata, so no chunk is read.

        Returns
        -------
        dict[str, FieldSchema]
            One :class:`FieldSchema` per stored field, keyed by field name.

        Raises
        ------
        RuntimeError
            If the reader has been closed.
        """
        schema: dict[str, FieldSchema] = {}
        for key, level, array in self._field_entries():
            dtype = torch.from_numpy(np.empty(0, dtype=array.dtype)).dtype
            cat_dim = _get_cat_dim(key) % len(array.shape)
            row_shape = tuple(
                size for axis, size in enumerate(array.shape) if axis != cat_dim
            )
            schema[key] = FieldSchema(level, dtype, row_shape)
        return schema

    def level_sizes(self) -> dict[str, int]:
        """Return the number of rows every level of the store holds.

        Returns
        -------
        dict[str, int]
            ``"atom"`` and ``"edge"`` follow the atom and edge pointers,
            ``"system"`` has one row per stored sample, and each registered
            custom level follows its own pointer, or has one row per sample
            when it is uniform. A segmented custom level whose pointer was
            never written, because no field materialized it, is left out.
        """
        return self._level_sizes(self.num_samples)

    def check_integrity(self) -> None:
        """Raise when the store's arrays disagree about how many samples it holds.

        An append interrupted between extending the pointers, masks, and
        field arrays and committing ``num_samples`` leaves them at different
        lengths, after which every sample past the torn one reads misaligned.
        Only array metadata is inspected, so no chunk is read.

        Raises
        ------
        ValueError
            If the store records no committed sample count, the atom or edge
            pointer is not non-decreasing from zero, the store declares a
            field it holds no array for, or a pointer, mask, or field array
            holds a number of rows other than the committed samples account
            for.
        RuntimeError
            If the reader has been closed.
        """
        if self._root is None:
            raise RuntimeError("Cannot read from a closed reader.")
        committed = self._root.attrs.get("num_samples")
        if committed is None:
            raise _torn_store_error("the store records no committed sample count")
        num_samples = int(committed)
        meta = self._root["meta"]
        pointers = {"atoms_ptr": self._atoms_ptr, "edges_ptr": self._edges_ptr}
        for name, pointer in pointers.items():
            if int(pointer[0].item()) != 0 or bool((pointer[1:] < pointer[:-1]).any()):
                raise _torn_store_error(
                    f"meta/{name} is not a non-decreasing pointer array starting at "
                    f"zero; got {pointer.tolist()!r}"
                )
        entries = self._field_entries()
        held = {key for key, _, _ in entries}
        for field in self.field_levels:
            if field not in held:
                raise _torn_store_error(
                    f"the store declares field {field!r} but holds no array for it"
                )
        totals = self._level_sizes(num_samples)
        lengths = {
            "meta/atoms_ptr": (int(self._atoms_ptr.numel()), num_samples + 1),
            "meta/edges_ptr": (int(self._edges_ptr.numel()), num_samples + 1),
            "meta/samples_mask": (self.num_samples, num_samples),
        }
        for name, level in (("atoms_mask", "atom"), ("edges_mask", "edge")):
            if name in meta:
                lengths[f"meta/{name}"] = (int(meta[name].shape[0]), totals[level])
        for key, level, array in entries:
            cat_dim = _get_cat_dim(key) % len(array.shape)
            lengths[key] = (int(array.shape[cat_dim]), totals[level])
        mismatched = [
            f"{name} holds {found!r} rows where {expected!r} are committed"
            for name, (found, expected) in lengths.items()
            if found != expected
        ]
        if mismatched:
            reported = ", ".join(mismatched[:_REPORTED_MISMATCHES])
            remaining = len(mismatched) - _REPORTED_MISMATCHES
            raise _torn_store_error(
                f"{num_samples!r} samples are committed but {reported}"
                + (
                    f", and {remaining!r} further arrays disagree"
                    if remaining > 0
                    else ""
                )
            )

    def _level_sizes(self, num_samples: int) -> dict[str, int]:
        """Return the rows per level when *num_samples* samples are stored."""
        sizes = {
            "atom": int(self._atoms_ptr[-1].item()),
            "edge": int(self._edges_ptr[-1].item()),
            "system": num_samples,
        }
        if self._level_schema is not None:
            for name in AtomicDataZarrWriter._custom_level_names(self._level_schema):
                pointer = self._level_ptrs.get(name)
                if pointer is not None:
                    sizes[name] = int(pointer[-1].item())
                elif self._level_schema.level_kind(name) == "uniform":
                    sizes[name] = num_samples
        return sizes

    def _field_entries(self) -> list[tuple[str, str, Any]]:
        """Return ``(field, level, array)`` for every array the store holds.

        Levels come from the store's field metadata, falling back to the
        core-field defaults for a legacy store that recorded none.
        """
        if self._root is None:
            raise RuntimeError("Cannot read from a closed reader.")
        fields: list[tuple[str, str, Any]] = []
        core_group = self._root["core"]
        for key in core_group.array_keys():
            level = self._fields_metadata.get("core", {}).get(
                key, _get_field_level(key)
            )
            fields.append((key, level, core_group[key]))
        if "custom" in self._root:
            custom_group = self._root["custom"]
            for key in custom_group.array_keys():
                level = self._fields_metadata.get("custom", {}).get(key, "system")
                fields.append((key, level, custom_group[key]))
        if self._level_schema is not None:
            levels_group = self._root["levels"]
            for level_name in levels_group.group_keys():
                level_group = levels_group[level_name]
                fields.extend(
                    (key, level_name, level_group[key])
                    for key in level_group.array_keys()
                )
        return fields

    def _resolve_logical_index(self, index: int) -> int:
        """Resolve a logical index according to this store's active sample mask."""
        if index < 0:
            index = len(self) + index
        if index < 0 or index >= len(self):
            raise IndexError(
                f"Index {index} out of range for reader with {len(self)} samples"
            )
        return index

    def _load_sample(self, index: int) -> dict[str, torch.Tensor]:
        """Load raw data for a single sample through the batch read path.

        Parameters
        ----------
        index : int
            Logical sample index (0 to len-1), accounting for deleted samples.

        Returns
        -------
        dict[str, torch.Tensor]
            Dictionary mapping field names to CPU tensors.

        Raises
        ------
        IndexError
            If index is out of range.
        """
        return self._load_many_samples([self._resolve_logical_index(index)])[0]

    def _reshape_product(
        self,
        value: torch.Tensor,
        level: str,
        physical_idx: int,
        level_ptrs: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        """Restore one flattened product segment to its parent-axis layout."""
        if (
            self._level_schema is None
            or level not in self._level_schema.level_kinds
            or self._level_schema.level_kind(level) != "product"
        ):
            return value
        left, right = self._level_schema.product_parents[level]
        left_ptr, right_ptr = level_ptrs.get(left), level_ptrs.get(right)
        if left_ptr is None or right_ptr is None:
            raise ValueError(f"Product level '{level}' has unresolved parents")
        left_count = int(left_ptr[physical_idx + 1] - left_ptr[physical_idx])
        right_count = int(right_ptr[physical_idx + 1] - right_ptr[physical_idx])
        return value.reshape(left_count, right_count, *value.shape[1:])

    def _read_many_orthogonal(
        self,
        normalized_indices: Sequence[int],
        sorted_order: Sequence[int],
        sorted_physical: Sequence[int],
        fields: Sequence[tuple[str, str, Any]],
        level_ptrs: Mapping[str, torch.Tensor],
    ) -> list[dict[str, torch.Tensor]]:
        """Load fragmented samples using one orthogonal selection per field."""
        data_by_sorted: list[dict[str, torch.Tensor]] = [{} for _ in sorted_order]
        pointer_ranges: dict[str, tuple[list[int], list[int], np.ndarray]] = {}
        for _key, level, _arr in fields:
            ptr = level_ptrs.get(level)
            if ptr is None or level in pointer_ranges:
                continue
            starts = [int(ptr[index]) for index in sorted_physical]
            ends = [int(ptr[index + 1]) for index in sorted_physical]
            rows = _row_indices_for_ranges(starts, ends)
            pointer_ranges[level] = (starts, ends, rows)

        for key, level, arr in fields:
            ptr = level_ptrs.get(level)
            if ptr is not None:
                if level == "edge":
                    # Validate edge-field layout even when this fragmented read selects no edge rows.
                    _slice_edge_array(arr, key, 0, 0)
                starts, ends, rows = pointer_ranges[level]
                block = torch.from_numpy(arr.oindex[rows] if len(rows) else arr[:0])
                offset = 0
                for i, (start, end) in enumerate(zip(starts, ends, strict=True)):
                    count = end - start
                    tensor = block[offset : offset + count]
                    if key == "neighbor_list" and level == "edge":
                        tensor = tensor - int(self._atoms_ptr[sorted_physical[i]])
                    tensor = self._reshape_product(
                        tensor, level, sorted_physical[i], level_ptrs
                    )
                    data_by_sorted[i][key] = tensor
                    offset += count
            else:
                rows = np.asarray(sorted_physical, dtype=np.int64)
                block = torch.from_numpy(arr.oindex[rows])
                for i in range(len(sorted_physical)):
                    data_by_sorted[i][key] = block[i : i + 1]

        inverse = [0] * len(sorted_order)
        for new_pos, old_pos in enumerate(sorted_order):
            inverse[old_pos] = new_pos

        return [data_by_sorted[inverse[i]] for i in range(len(normalized_indices))]

    def _load_many_samples(
        self, indices: Sequence[int]
    ) -> list[dict[str, torch.Tensor]]:
        """Load raw data for multiple samples in requested order.

        Contiguous physical samples are read as ranges so each Zarr array is
        opened once and sliced once per range. The range tensors are then
        split back into per-sample dictionaries. The base ``Reader`` attaches
        metadata and optional pinned memory.

        Parameters
        ----------
        indices : Sequence[int]
            Logical sample indices to load. Negative values are supported.

        Returns
        -------
        list[dict[str, torch.Tensor]]
            Ordered raw tensor dictionaries with CPU tensors.

        Raises
        ------
        RuntimeError
            If the reader has been closed.
        IndexError
            If any requested index is out of range.
        """
        if self._root is None:
            raise RuntimeError("Cannot read from a closed reader.")

        normalized_indices = [self._resolve_logical_index(index) for index in indices]
        if not normalized_indices:
            return []

        fields = self._field_entries()

        level_ptrs = {
            "atom": self._atoms_ptr,
            "atoms": self._atoms_ptr,
            "edge": self._edges_ptr,
            "edges": self._edges_ptr,
            **self._level_ptrs,
        }
        physical_indices = [
            int(self._active_indices[index]) for index in normalized_indices
        ]
        sorted_order = sorted(
            range(len(physical_indices)), key=physical_indices.__getitem__
        )
        sorted_physical = [physical_indices[index] for index in sorted_order]
        runs = _merge_physical_runs_by_chunks(sorted_physical, fields, level_ptrs)
        if len(runs) > 4:
            return self._read_many_orthogonal(
                normalized_indices,
                sorted_order,
                sorted_physical,
                fields,
                level_ptrs,
            )

        data_by_sorted: list[dict[str, torch.Tensor]] = [{} for _ in sorted_order]
        for positions in runs:
            first = sorted_physical[positions[0]]
            last = sorted_physical[positions[-1]]
            for key, level, arr in fields:
                ptr = level_ptrs.get(level)
                if ptr is None:
                    block = torch.from_numpy(arr[first : last + 1])
                else:
                    block_start, block_end = int(ptr[first]), int(ptr[last + 1])
                    if level == "edge":
                        array = _slice_edge_array(arr, key, block_start, block_end)
                    else:
                        array = arr[block_start:block_end]
                    block = torch.from_numpy(array)
                for position in positions:
                    physical_idx = sorted_physical[position]
                    if ptr is None:
                        value = block[physical_idx - first : physical_idx - first + 1]
                    else:
                        start, end = int(ptr[physical_idx]), int(ptr[physical_idx + 1])
                        value = block[start - block_start : end - block_start]
                    if key == "neighbor_list" and level == "edge":
                        value = value - int(self._atoms_ptr[physical_idx])
                    data_by_sorted[position][key] = self._reshape_product(
                        value, level, physical_idx, level_ptrs
                    )
        inverse = [0] * len(sorted_order)
        for new_position, old_position in enumerate(sorted_order):
            inverse[old_position] = new_position
        return [
            data_by_sorted[inverse[index]] for index in range(len(normalized_indices))
        ]

    def __len__(self) -> int:
        """Return the number of active (non-deleted) samples.

        Returns
        -------
        int
            Number of samples available for reading.
        """
        return len(self._active_indices)

    def _get_sample_metadata(self, index: int) -> dict[str, str | int]:
        """Return metadata for a sample.

        Parameters
        ----------
        index : int
            Logical sample index.

        Returns
        -------
        dict[str, str]
            Dictionary containing source file information.
        """
        index = self._resolve_logical_index(index)
        physical_idx = int(self._active_indices[index].item())
        return {
            "index": index,
            "source_file": str(self._store),
            "physical_index": str(physical_idx),
        }

    def get_metadata(self, index: int) -> tuple[int, int]:
        """Return atom and edge counts from cached pointer arrays.

        Parameters
        ----------
        index : int
            Logical sample index. Negative values are supported.

        Returns
        -------
        tuple[int, int]
            ``(num_atoms, num_edges)`` for the sample.
        """
        index = self._resolve_logical_index(index)
        physical_idx = int(self._active_indices[index].item())
        num_atoms = int(
            (self._atoms_ptr[physical_idx + 1] - self._atoms_ptr[physical_idx]).item()
        )
        num_edges = int(
            (self._edges_ptr[physical_idx + 1] - self._edges_ptr[physical_idx]).item()
        )
        return num_atoms, num_edges

    def close(self) -> None:
        """Release the Zarr store reference and clean up resources."""
        self._root = None
        super().close()
