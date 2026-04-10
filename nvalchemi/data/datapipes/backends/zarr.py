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

import re
from collections.abc import Mapping
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

# Type alias for zarr store-like objects
StoreLike: TypeAlias = Store | StorePath | Path | str | dict[str, Any]

# TODO: make classes inherit from PNM when stable


class ZarrArrayConfig(BaseModel):
    """Configuration for Zarr array compression, chunking, and sharding.

    Parameters
    ----------
    compressors : tuple[zarr.abc.codec.Codec, ...] | None
        Compressor codec(s) to apply. E.g. ``(zarr.codecs.ZstdCodec(level=3),)``.
    filters : tuple[zarr.abc.codec.Codec, ...] | None
        Array-to-array filter codec(s). E.g. ``(zarr.codecs.TransposeCodec(order=(1, 0)),)``.
    serializer : zarr.abc.codec.Codec | None
        Bytes serializer codec. E.g. ``zarr.codecs.BytesCodec(endian="little")``.
    chunk_size : int | None
        Chunk length along dimension 0. Other dimensions use their full extent.
        ``None`` uses Zarr defaults.
    shard_size : int | None
        Shard length along dimension 0. When set, multiple chunks are stored
        in a single storage object. Must be a multiple of ``chunk_size`` when
        both are specified. ``None`` disables sharding.
    write_empty_chunks : bool
        Whether to write chunks that are entirely fill-valued. Default ``True``.
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
    """Top-level write configuration for ``AtomicDataZarrWriter``.

    Provides per-group defaults and optional per-field overrides.

    Parameters
    ----------
    meta : ZarrArrayConfig
        Config for metadata arrays (pointers, masks). Usually no compression.
    core : ZarrArrayConfig
        Config for core data arrays (positions, energy, etc.).
    custom : ZarrArrayConfig
        Config for user-added custom arrays.
    field_overrides : dict[str, ZarrArrayConfig]
        Per-field overrides. Keys are field names (e.g. ``"positions"``).
        Takes precedence over group-level config.

    Examples
    --------
    >>> from zarr.codecs import ZstdCodec, BloscCodec
    >>> config = ZarrWriteConfig(
    ...     core=ZarrArrayConfig(compressors=(ZstdCodec(level=3),), chunk_size=1024),
    ...     field_overrides={
    ...         "positions": ZarrArrayConfig(compressors=(BloscCodec(cname="lz4"),))
    ...     },
    ... )
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
        if num_samples is None or num_samples == 0:
            raise ValueError("No data provided to write.")

        root = self._open(mode="w")
        meta_group = root.create_group("meta")
        core_group = root.create_group("core")
        root.create_group("custom")

        # Build pointer arrays directly from batch metadata — no iteration
        nodes_tensor = torch.tensor(data.num_nodes_list, dtype=torch.long)
        # Handle case where num_edges_list is empty (no edges in data)
        if data.num_edges_list:
            edges_tensor = torch.tensor(data.num_edges_list, dtype=torch.long)
        else:
            # No edges: create zeros for each sample
            edges_tensor = torch.zeros(num_samples, dtype=torch.long)
        atoms_ptr = torch.cat(
            [torch.zeros(1, dtype=torch.long), torch.cumsum(nodes_tensor, dim=0)]
        )
        edges_ptr = torch.cat(
            [torch.zeros(1, dtype=torch.long), torch.cumsum(edges_tensor, dim=0)]
        )

        total_atoms = int(atoms_ptr[-1].item())
        total_edges = int(edges_ptr[-1].item())

        # Write meta arrays
        atoms_ptr_np = self._to_numpy(atoms_ptr)
        edges_ptr_np = self._to_numpy(edges_ptr)
        samples_mask_np = np.ones(num_samples, dtype=bool)
        atoms_mask_np = np.ones(total_atoms, dtype=bool)
        edges_mask_np = np.ones(total_edges, dtype=bool)

        meta_group.create_array(
            "atoms_ptr",
            data=atoms_ptr_np,
            **self._resolve_array_kwargs("atoms_ptr", "meta", atoms_ptr_np),
        )
        meta_group.create_array(
            "edges_ptr",
            data=edges_ptr_np,
            **self._resolve_array_kwargs("edges_ptr", "meta", edges_ptr_np),
        )
        meta_group.create_array(
            "samples_mask",
            data=samples_mask_np,
            **self._resolve_array_kwargs("samples_mask", "meta", samples_mask_np),
        )
        meta_group.create_array(
            "atoms_mask",
            data=atoms_mask_np,
            **self._resolve_array_kwargs("atoms_mask", "meta", atoms_mask_np),
        )
        meta_group.create_array(
            "edges_mask",
            data=edges_mask_np,
            **self._resolve_array_kwargs("edges_mask", "meta", edges_mask_np),
        )

        # Build field level mapping
        fields_metadata: dict[str, dict[str, str]] = {"core": {}, "custom": {}}

        # Collect all field keys from the batch's level categorization.
        # batch.keys is {"node": set, "edge": set, "system": set} — these are the
        # field names present in the batch, already categorized by level.
        excluded = {"batch_idx", "batch_ptr", "device", "dtype", "info", "num_nodes"}
        all_field_keys: set[str] = set()
        level_map: dict[str, str] = {}  # key -> "atom"/"edge"/"system"
        for level_name, key_set in (data.keys or {}).items():
            for k in key_set:
                all_field_keys.add(k)
                # batch.keys uses "node"/"edge"/"system"; zarr format uses "atom"/"edge"/"system"
                level_map[k] = "atom" if level_name == "node" else level_name

        # Get all tensor attributes from the batch (Pydantic's to_dict works)
        batch_dict = data.to_dict()

        # Write each field: one zarr I/O per field, no per-sample loop
        for key in all_field_keys:
            val = batch_dict.get(key)
            if val is None or not isinstance(val, torch.Tensor):
                continue
            if key in excluded:
                continue

            level = level_map.get(key, _get_field_level(key))
            fields_metadata["core"][key] = level

            # System-level: Batch stacks (1, 3, 3) -> (N, 1, 3, 3); squeeze dim 1 so zarr has (N, 3, 3)
            if level == "system" and val.dim() > 2:
                while val.dim() > 2 and val.shape[1] == 1:
                    val = val.squeeze(1)
            np_val = self._to_numpy(val)
            cat_dim = _get_cat_dim(key)
            if cat_dim < 0:
                cat_dim += np_val.ndim
            core_group.create_array(
                key,
                data=np_val,
                **self._resolve_array_kwargs(key, "core", np_val, cat_dim=cat_dim),
            )

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
        if not self._store_exists():
            raise FileNotFoundError(f"Zarr store does not exist at {self._store}")

        root = self._open(mode="r+")
        meta_group = root["meta"]
        core_group = root["core"]

        # Read existing pointer tails
        old_atoms_ptr = torch.from_numpy(meta_group["atoms_ptr"][:])
        old_edges_ptr = torch.from_numpy(meta_group["edges_ptr"][:])
        old_num_samples = int(root.attrs["num_samples"])

        last_atom_ptr = int(old_atoms_ptr[-1].item())
        last_edge_ptr = int(old_edges_ptr[-1].item())

        # Get counts directly from the single AtomicData
        data_dict = data.to_dict()
        num_atoms = int(data.num_nodes)
        # Determine num_edges from neighbor_list if present
        neighbor_list = data_dict.get("neighbor_list")
        if neighbor_list is not None and isinstance(neighbor_list, torch.Tensor):
            num_edges = neighbor_list.shape[0]
        else:
            num_edges = 0

        # Extend pointer arrays with single new entries
        new_atom_ptr = torch.tensor([last_atom_ptr + num_atoms], dtype=torch.long)
        new_edge_ptr = torch.tensor([last_edge_ptr + num_edges], dtype=torch.long)
        self._extend_array(meta_group["atoms_ptr"], self._to_numpy(new_atom_ptr))
        self._extend_array(meta_group["edges_ptr"], self._to_numpy(new_edge_ptr))

        # Extend masks (single sample, its atoms, its edges)
        self._extend_array(
            meta_group["samples_mask"],
            self._to_numpy(torch.ones(1, dtype=torch.bool)),
        )
        self._extend_array(
            meta_group["atoms_mask"],
            self._to_numpy(torch.ones(num_atoms, dtype=torch.bool)),
        )
        self._extend_array(
            meta_group["edges_mask"],
            self._to_numpy(torch.ones(num_edges, dtype=torch.bool)),
        )

        # Extend each existing core field
        excluded = {"batch_idx", "batch_ptr", "device", "dtype", "info", "num_nodes"}
        for key in core_group.keys():
            val = data_dict.get(key)
            if val is None or not isinstance(val, torch.Tensor):
                continue
            if key in excluded:
                continue
            # System-level fields need unsqueeze(0) to add the sample dimension
            level = _get_field_level(key)
            if level == "system":
                val = val.unsqueeze(0) if val.dim() == 0 else val
            cat_dim = _get_cat_dim(key)
            self._extend_array(core_group[key], self._to_numpy(val), axis=cat_dim)

        root.attrs["num_samples"] = old_num_samples + 1

    @overload
    def append(self, data: list[AtomicData]) -> None:  # noqa: F811
        """Append a list of AtomicData to an existing Zarr store."""
        device = data[0].device
        self.append(Batch.from_data_list(data, device))

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
        """
        if not self._store_exists():
            raise FileNotFoundError(f"Zarr store does not exist at {self._store}")

        num_samples = data.num_graphs
        if num_samples is None or num_samples == 0:
            return

        root = self._open(mode="r+")
        meta_group = root["meta"]
        core_group = root["core"]

        # Read existing pointer tails
        old_atoms_ptr = torch.from_numpy(meta_group["atoms_ptr"][:])
        old_edges_ptr = torch.from_numpy(meta_group["edges_ptr"][:])
        old_num_samples = int(root.attrs["num_samples"])

        last_atom_ptr = int(old_atoms_ptr[-1].item())
        last_edge_ptr = int(old_edges_ptr[-1].item())

        # Compute new pointer entries from batch metadata
        nodes_tensor = torch.tensor(data.num_nodes_list, dtype=torch.long)
        # Handle case where num_edges_list is empty (no edges in data)
        if data.num_edges_list:
            edges_tensor = torch.tensor(data.num_edges_list, dtype=torch.long)
        else:
            # No edges: create zeros for each sample
            edges_tensor = torch.zeros(num_samples, dtype=torch.long)
        new_atoms_ptr = last_atom_ptr + torch.cumsum(nodes_tensor, dim=0)
        new_edges_ptr = last_edge_ptr + torch.cumsum(edges_tensor, dim=0)

        new_total_atoms = int(new_atoms_ptr[-1].item())
        new_total_edges = int(new_edges_ptr[-1].item())

        # Extend pointer arrays
        self._extend_array(meta_group["atoms_ptr"], self._to_numpy(new_atoms_ptr))
        self._extend_array(meta_group["edges_ptr"], self._to_numpy(new_edges_ptr))

        # Extend masks
        self._extend_array(
            meta_group["samples_mask"],
            self._to_numpy(torch.ones(num_samples, dtype=torch.bool)),
        )
        self._extend_array(
            meta_group["atoms_mask"],
            self._to_numpy(
                torch.ones(new_total_atoms - last_atom_ptr, dtype=torch.bool)
            ),
        )
        self._extend_array(
            meta_group["edges_mask"],
            self._to_numpy(
                torch.ones(new_total_edges - last_edge_ptr, dtype=torch.bool)
            ),
        )

        # Get all tensor attributes from the batch (Pydantic's to_dict works)
        batch_dict = data.to_dict()

        # Extend each field — single I/O per field
        excluded = {"batch_idx", "batch_ptr", "device", "dtype", "info", "num_nodes"}
        for key in core_group.keys():
            val = batch_dict.get(key)
            if val is None or not isinstance(val, torch.Tensor):
                continue
            if key in excluded:
                continue
            level = _get_field_level(key)
            if level == "system" and val.dim() > 2:
                while val.dim() > 2 and val.shape[1] == 1:
                    val = val.squeeze(1)
            cat_dim = _get_cat_dim(key)
            self._extend_array(core_group[key], self._to_numpy(val), axis=cat_dim)

        root.attrs["num_samples"] = old_num_samples + num_samples

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
        """
        pass

    def add_custom(
        self, key: str, data: torch.Tensor, level: Literal["atom", "edge", "system"]
    ) -> None:
        """Add a custom array to the custom/ group.

        Parameters
        ----------
        key : str
            Name for the custom array.
        data : torch.Tensor
            Tensor data. First dimension must match:
            - num_samples for "system" level
            - total atoms for "atom" level
            - total edges for "edge" level
        level : str
            One of "atom", "edge", "system".

        Raises
        ------
        ValueError
            If level is invalid or data shape doesn't match.
        FileNotFoundError
            If store does not exist.
        """
        if level not in ("atom", "edge", "system"):
            raise ValueError(
                f"Invalid level '{level}'. Must be 'atom', 'edge', or 'system'."
            )

        if not self._store_exists():
            raise FileNotFoundError(f"Zarr store does not exist at {self._store}")

        root = self._open(mode="r+")
        meta_group = root["meta"]
        custom_group = root["custom"]

        # Validate shape
        num_samples = int(root.attrs["num_samples"])
        atoms_ptr = meta_group["atoms_ptr"][:]
        edges_ptr = meta_group["edges_ptr"][:]
        total_atoms = int(atoms_ptr[-1])
        total_edges = int(edges_ptr[-1])

        expected_size = {
            "system": num_samples,
            "atom": total_atoms,
            "edge": total_edges,
        }[level]

        if data.shape[0] != expected_size:
            raise ValueError(
                f"Data shape[0]={data.shape[0]} does not match expected "
                f"size={expected_size} for level='{level}'."
            )

        # Write to custom group (convert to numpy at zarr boundary)
        np_data = self._to_numpy(data)
        custom_group.create_array(
            key, data=np_data, **self._resolve_array_kwargs(key, "custom", np_data)
        )

        # Update fields metadata
        fields_metadata = dict(root.attrs.get("fields", {"core": {}, "custom": {}}))
        if "custom" not in fields_metadata:
            fields_metadata["custom"] = {}
        fields_metadata["custom"][key] = level
        root.attrs["fields"] = fields_metadata

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

        # Find active sample indices
        active_indices = np.where(samples_mask)[0]

        if len(active_indices) == 0:
            # All samples deleted, just reset to empty
            # Clear store by opening with mode="w" (overwrite)
            new_root = self._open(mode="w")
            new_root.create_group("meta")
            new_root.create_group("core")
            new_root.create_group("custom")
            new_root.attrs["num_samples"] = 0
            new_root.attrs["fields"] = {"core": {}, "custom": {}}
            return

        # Collect active data for each field
        new_core_data: dict[str, list[np.ndarray]] = {
            key: [] for key in core_group.keys()
        }
        new_custom_data: dict[str, list[np.ndarray]] = {}

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

        # Clear store and create new structure (mode="w" clears existing data)
        new_root = self._open(mode="w")
        new_meta = new_root.create_group("meta")
        new_core = new_root.create_group("core")
        new_custom = new_root.create_group("custom")

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

        # Update metadata
        new_root.attrs["num_samples"] = new_num_samples
        new_root.attrs["fields"] = fields_metadata

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
        """
        if self._root is None:
            raise RuntimeError("Cannot refresh a closed reader.")

        # Re-open the store to pick up structural changes
        self._root = zarr.open(self._store, mode="r")

        # Cache pointer arrays as torch tensors
        self._atoms_ptr = torch.from_numpy(self._root["meta"]["atoms_ptr"][:]).to(
            torch.long
        )
        self._edges_ptr = torch.from_numpy(self._root["meta"]["edges_ptr"][:]).to(
            torch.long
        )

        # Cache samples mask
        self._samples_mask = torch.from_numpy(self._root["meta"]["samples_mask"][:]).to(
            torch.bool
        )

        # Build logical->physical index mapping (indices where mask is True)
        self._active_indices = torch.where(self._samples_mask)[0]

        # Cache fields metadata
        self._fields_metadata: dict[str, dict[str, str]] = dict(
            self._root.attrs.get("fields", {"core": {}, "custom": {}})
        )

    def _load_sample(self, index: int) -> dict[str, torch.Tensor]:
        """Load raw data for a single sample.

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
        # Map logical index to physical index
        physical_idx = int(self._active_indices[index].item())

        # Get slice ranges from pointer arrays
        atom_start = int(self._atoms_ptr[physical_idx].item())
        atom_end = int(self._atoms_ptr[physical_idx + 1].item())
        edge_start = int(self._edges_ptr[physical_idx].item())
        edge_end = int(self._edges_ptr[physical_idx + 1].item())

        data: dict[str, torch.Tensor] = {}

        # Load core fields
        core_group = self._root["core"]
        for key in core_group.array_keys():
            level = self._fields_metadata.get("core", {}).get(
                key, _get_field_level(key)
            )
            arr = core_group[key]

            if level == "atom":
                data[key] = torch.from_numpy(arr[atom_start:atom_end])
            elif level == "edge":
                tensor = torch.from_numpy(
                    _slice_edge_array(arr, key, edge_start, edge_end)
                )

                # neighbor_list needs to be converted from global to local indices
                # by subtracting the atom offset for this sample
                if key == "neighbor_list":
                    tensor = tensor - atom_start

                data[key] = tensor
            else:  # system level
                # Keep batch dim for system-level fields
                data[key] = torch.from_numpy(arr[physical_idx : physical_idx + 1])

        # Load custom fields if present
        if "custom" in self._root:
            custom_group = self._root["custom"]
            for key in custom_group.array_keys():
                level = self._fields_metadata.get("custom", {}).get(key, "system")
                arr = custom_group[key]

                if level == "atom":
                    data[key] = torch.from_numpy(arr[atom_start:atom_end])
                elif level == "edge":
                    data[key] = torch.from_numpy(
                        _slice_edge_array(arr, key, edge_start, edge_end)
                    )
                else:  # system level
                    data[key] = torch.from_numpy(arr[physical_idx : physical_idx + 1])

        return data

    def __len__(self) -> int:
        """Return the number of active (non-deleted) samples.

        Returns
        -------
        int
            Number of samples available for reading.
        """
        return len(self._active_indices)

    def _get_sample_metadata(self, index: int) -> dict[str, str]:
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
        physical_idx = int(self._active_indices[index].item())
        return {
            "source_file": str(self._store),
            "physical_index": str(physical_idx),
        }

    def close(self) -> None:
        """Release the Zarr store reference and clean up resources."""
        self._root = None
        super().close()
