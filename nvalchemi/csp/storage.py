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
"""Zarr persistence for rigid-molecule ASU representations.

Stores share formula-unit conformers and retain each structure's cell,
symmetry, and independent molecular placements; full-cell atom coordinates are
materialized only when selected rows are expanded.
"""

from __future__ import annotations

import os
import shutil
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any
from uuid import uuid4

import numpy as np
import torch
import zarr
from zarr.storage import MemoryStore

from nvalchemi.csp.data import MolecularPackingInput, RigidMoleculeASUBatch
from nvalchemi.data.batch import Batch
from nvalchemi.data.datapipes.backends.zarr import (
    StoreLike,
    ZarrWriteConfig,
)

_REPRESENTATION = "nvalchemi.csp.rigid_molecule_asu"
_SCHEMA_VERSION = 1
_MAX_POINTER = int(np.iinfo(np.int32).max)
_FORMULA_FIELDS = (
    "conformer_positions",
    "conformer_ptr",
    "molecule_conformer_ptr",
    "molecule_atom_ptr",
    "atomic_numbers",
    "contact_distances",
    "component_index",
)
_STRUCTURE_FIELDS = (
    "structure_ids",
    "conformer_indices",
    "rotations",
    "fractional_centers",
    "cells",
    "space_groups",
    "z",
    "z_prime",
)


def _numpy(value: torch.Tensor) -> np.ndarray:
    """Expose a contiguous host NumPy array; CPU tensor storage may be shared."""
    return value.detach().to(device="cpu").contiguous().numpy()


def _config(value: ZarrWriteConfig | Mapping[str, Any] | None) -> ZarrWriteConfig:
    """Normalize an optional writer configuration."""
    if value is None:
        return ZarrWriteConfig()
    if isinstance(value, ZarrWriteConfig):
        return value
    return ZarrWriteConfig.model_validate(value)


def _config_kwargs(
    config: ZarrWriteConfig, name: str, group: str, data: np.ndarray
) -> dict[str, Any]:
    """Build Zarr array creation options for one field and array role."""
    array_config = config.field_overrides.get(name, getattr(config, group))
    kwargs: dict[str, Any] = {}
    if array_config.compressors is not None:
        kwargs["compressors"] = array_config.compressors
    if array_config.filters is not None:
        kwargs["filters"] = array_config.filters
    if array_config.serializer is not None:
        kwargs["serializer"] = array_config.serializer
    if array_config.chunk_size is not None and data.ndim:
        chunks = list(data.shape)
        chunks[0] = array_config.chunk_size
        kwargs["chunks"] = tuple(chunks)
    if array_config.shard_size is not None and data.ndim:
        shards = list(data.shape)
        shards[0] = array_config.shard_size
        kwargs["shards"] = tuple(shards)
    if not array_config.write_empty_chunks:
        kwargs["config"] = {"write_empty_chunks": False}
    return kwargs


def _create(
    group: zarr.Group, key: str, value: np.ndarray, cfg: ZarrWriteConfig, role: str
) -> Any:
    """Create a Zarr array using the configured layout for its field."""
    return group.create_array(key, data=value, **_config_kwargs(cfg, key, role, value))


def _torch_array_metadata(value: torch.Tensor) -> tuple[tuple[int, ...], np.dtype]:
    """Return a tensor's shape and corresponding NumPy dtype without copying it."""
    dtype = torch.empty((), dtype=value.dtype).numpy().dtype
    return tuple(value.shape), np.dtype(dtype)


def _preflight_config(batch: RigidMoleculeASUBatch, config: ZarrWriteConfig) -> None:
    """Validate all configured Zarr array layouts without copying payloads."""
    formula = batch.packing_input
    arrays: dict[str, dict[str, tuple[tuple[int, ...], np.dtype]]] = {
        "packing_input": {
            name: _torch_array_metadata(getattr(formula, name))
            for name in _FORMULA_FIELDS
        },
        "meta": {
            "molecules_ptr": ((batch.num_structures + 1,), np.dtype("int32")),
            "samples_mask": ((batch.num_structures,), np.dtype("bool")),
            "molecules_mask": (
                (int(batch.structure_molecule_ptr[-1]),),
                np.dtype("bool"),
            ),
            "structure_ids": _torch_array_metadata(batch.structure_ids),
        },
        "core": {
            name: _torch_array_metadata(getattr(batch, name))
            for name in _STRUCTURE_FIELDS
            if name != "structure_ids"
        },
        "custom": {
            name: _torch_array_metadata(value)
            for name, value in batch.properties.items()
        },
    }
    store = MemoryStore()
    root = zarr.open_group(store, mode="w")
    groups = {
        name: root.create_group(name)
        for name in ("packing_input", "meta", "core", "custom")
    }
    for group_name, fields in arrays.items():
        role = (
            "meta"
            if group_name == "meta"
            else "custom"
            if group_name == "custom"
            else "core"
        )
        group = groups[group_name]
        for name, (shape, dtype) in fields.items():
            exemplar = np.empty((0, *shape[1:]), dtype=dtype)
            kwargs = _config_kwargs(config, name, role, exemplar)
            group.create_array(name, shape=shape, dtype=dtype, **kwargs)


def _input_attrs(value: MolecularPackingInput) -> dict[str, Any]:
    """Return the JSON-compatible versioned packing-input attributes."""
    return {
        "version": 1,
        "formula_unit_volume": float(value.formula_unit_volume),
        "metadata": None if value.metadata is None else _thaw(value.metadata),
    }


def _thaw(value: Any) -> Any:
    """Convert frozen nested metadata containers into mutable JSON values."""
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    if isinstance(value, list):
        return [_thaw(item) for item in value]
    return value


def _freeze(value: Any) -> Any:
    """Make nested metadata mappings read-only and lists immutable."""
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def _input_arrays(value: MolecularPackingInput) -> dict[str, np.ndarray]:
    """Expose formula fields as host arrays; CPU storage may be shared."""
    return {name: _numpy(getattr(value, name)) for name in _FORMULA_FIELDS}


def _same_input(left: MolecularPackingInput, right: MolecularPackingInput) -> bool:
    """Check exact metadata and array equality for two packing inputs."""
    if _input_attrs(left) != _input_attrs(right):
        return False
    left_arrays = _input_arrays(left)
    right_arrays = _input_arrays(right)
    return all(
        np.array_equal(left_arrays[name], right_arrays[name])
        for name in _FORMULA_FIELDS
    )


def _batch_arrays(batch: RigidMoleculeASUBatch) -> dict[str, np.ndarray]:
    """Expose structure fields and pointers as host arrays; CPU storage may be shared."""
    values = {name: _numpy(getattr(batch, name)) for name in _STRUCTURE_FIELDS}
    values["molecules_ptr"] = _numpy(batch.structure_molecule_ptr)
    return values


def _same_batch(left: RigidMoleculeASUBatch, right: RigidMoleculeASUBatch) -> bool:
    """Check exact packing-input equality and bitwise structure/property equality."""
    if not _same_input(left.packing_input, right.packing_input):
        return False
    left_arrays = _batch_arrays(left)
    right_arrays = _batch_arrays(right)
    if left_arrays.keys() != right_arrays.keys() or not all(
        _bitwise_equal(left_arrays[name], right_arrays[name]) for name in left_arrays
    ):
        return False
    if left.properties.keys() != right.properties.keys():
        return False
    return all(
        _bitwise_equal(_numpy(left.properties[name]), _numpy(right.properties[name]))
        for name in left.properties
    )


def _row_equal(
    left_arrays: Mapping[str, np.ndarray],
    left_props: Mapping[str, np.ndarray],
    left_index: int,
    right_arrays: Mapping[str, np.ndarray],
    right_props: Mapping[str, np.ndarray],
    right_index: int,
) -> bool:
    """Compare one ASU structure row and its system properties exactly."""
    left_start, left_end = map(
        int, left_arrays["molecules_ptr"][left_index : left_index + 2]
    )
    right_start, right_end = map(
        int, right_arrays["molecules_ptr"][right_index : right_index + 2]
    )
    if left_end - left_start != right_end - right_start:
        return False
    for name in _STRUCTURE_FIELDS:
        left, right = left_arrays[name], right_arrays[name]
        if name in {"conformer_indices", "rotations", "fractional_centers"}:
            left = left[left_start:left_end]
            right = right[right_start:right_end]
        else:
            left = left[left_index]
            right = right[right_index]
        if not _bitwise_equal(left, right):
            return False
    return all(
        _bitwise_equal(left_props[name][left_index], right_props[name][right_index])
        for name in left_props
    )


def _unique_batch(batch: RigidMoleculeASUBatch) -> RigidMoleculeASUBatch:
    """Keep the first row for each ID, rejecting conflicting duplicate payloads."""
    arrays = _batch_arrays(batch)
    properties = {name: _numpy(value) for name, value in batch.properties.items()}
    first_for_id: dict[tuple[int, int], int] = {}
    unique_indices: list[int] = []
    for index in range(batch.num_structures):
        key = _id_key(arrays["structure_ids"], index)
        previous = first_for_id.get(key)
        if previous is None:
            first_for_id[key] = index
            unique_indices.append(index)
        elif not _row_equal(arrays, properties, previous, arrays, properties, index):
            raise ValueError(
                f"structure ID {key} conflicts with another row in this batch"
            )
    if len(unique_indices) == batch.num_structures:
        return batch
    indices = torch.tensor(unique_indices, dtype=torch.int64, device=batch.cells.device)
    return batch.select(indices)


def _validate_property_names(properties: Mapping[str, torch.Tensor]) -> None:
    """Reject custom fields that cannot safely be used as Zarr array keys."""
    for name in properties:
        if name in {".", "..", "zarr.json"} or "/" in name:
            raise ValueError(
                f"custom property name {name!r} cannot be stored as a Zarr array key"
            )


def _validate_nonnegative_ids(batch: RigidMoleculeASUBatch) -> None:
    """Require both components of every stable structure ID to be nonnegative."""
    if np.any(_numpy(batch.structure_ids) < 0):
        raise ValueError("structure_ids values must be nonnegative")


def _select_rows(array: Any, indices: np.ndarray) -> np.ndarray:
    """Select first-axis rows while preserving requested order and repeats."""
    if indices.size == 0:
        return np.empty((0, *array.shape[1:]), dtype=array.dtype)
    return array.oindex[indices]


def _read_store_row(
    root: zarr.Group, row: int
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Read one physical structure row and its custom properties from Zarr."""
    ptr = _select_rows(
        root["meta"]["molecules_ptr"], np.asarray([row, row + 1], dtype=np.int64)
    )
    start, stop = map(int, ptr)
    arrays: dict[str, np.ndarray] = {
        "molecules_ptr": np.asarray([0, stop - start], dtype=np.int32),
        "structure_ids": _select_rows(
            root["meta"]["structure_ids"], np.asarray([row], dtype=np.int64)
        ),
    }
    for name in _STRUCTURE_FIELDS:
        if name == "structure_ids":
            continue
        if name in {"conformer_indices", "rotations", "fractional_centers"}:
            arrays[name] = root["core"][name][start:stop]
        else:
            arrays[name] = _select_rows(
                root["core"][name], np.asarray([row], dtype=np.int64)
            )
    properties = {
        name: _select_rows(root["custom"][name], np.asarray([row], dtype=np.int64))
        for name in root["custom"].array_keys()
    }
    return arrays, properties


def _bitwise_equal(left: np.ndarray, right: np.ndarray) -> bool:
    """Compare NumPy arrays by dtype, shape, and stored bytes."""
    return (
        left.dtype == right.dtype
        and left.shape == right.shape
        and left.tobytes() == right.tobytes()
    )


def _id_key(ids: np.ndarray, index: int) -> tuple[int, int]:
    """Convert one two-component structure ID row to a dictionary key."""
    return int(ids[index, 0]), int(ids[index, 1])


class RigidMoleculeASUZarrWriter:
    """Write and maintain rigid-molecule ASU representations in a Zarr store.

    Custom property names must not contain ``/`` or equal ``.``, ``..``, or
    ``zarr.json``, which Zarr reserves or interprets as path/metadata keys.
    These additional naming restrictions apply when writing to Zarr.

    Use one active writer per store. Reopen a writer after another writer
    changes the store; cached IDs and property schema are not refreshed
    automatically. Readers must refresh after any store mutation.

    Parameters
    ----------
    store : StoreLike
        Filesystem path or Zarr-compatible store.
    config : ZarrWriteConfig or mapping, optional
        Toolkit chunking and compression settings.
    """

    def __init__(
        self,
        store: StoreLike,
        config: ZarrWriteConfig | Mapping[str, Any] | None = None,
    ) -> None:
        """Bind a target store and its optional Zarr write configuration."""
        self._store = store
        self._config = _config(config)
        self._closed = False
        self._cache: (
            tuple[
                MolecularPackingInput,
                dict[str, tuple[np.dtype, tuple[int, ...]]],
                dict[tuple[int, int], int],
            ]
            | None
        ) = None

    def _check_open(self) -> None:
        """Raise if this writer has been closed."""
        if self._closed:
            raise RuntimeError("Cannot use a closed RigidMoleculeASUZarrWriter")

    def _open(self, mode: str = "r") -> zarr.Group:
        """Open the configured store using a Zarr access mode."""
        return zarr.open_group(self._store, mode=mode)

    def _build_cache(
        self, root: zarr.Group
    ) -> tuple[
        MolecularPackingInput,
        dict[str, tuple[np.dtype, tuple[int, ...]]],
        dict[tuple[int, int], int],
    ]:
        """Validate a store and cache its input, property schema, and active IDs."""
        packing_input = _validate_root(root)
        properties = {
            name: (root["custom"][name].dtype, root["custom"][name].shape[1:])
            for name in root["custom"].array_keys()
        }
        mask = root["meta"]["samples_mask"][:].astype(bool)
        ids = root["meta"]["structure_ids"][:]
        active_by_id = {_id_key(ids, row): row for row in range(len(mask)) if mask[row]}
        return packing_input, properties, active_by_id

    def _write_new(self, batch: RigidMoleculeASUBatch, *, mode: str = "w-") -> None:
        """Write a complete ASU batch and versioned schema to a new store."""
        root = self._open(mode)
        meta = root.create_group("meta")
        core = root.create_group("core")
        custom = root.create_group("custom")
        input_group = root.create_group("packing_input")
        input_group.attrs.update(_input_attrs(batch.packing_input))
        for name, value in _input_arrays(batch.packing_input).items():
            _create(input_group, name, value, self._config, "core")

        arrays = _batch_arrays(batch)
        rows = batch.num_structures
        ptr = arrays.pop("molecules_ptr")
        _create(meta, "molecules_ptr", ptr, self._config, "meta")
        _create(
            meta, "samples_mask", np.ones(rows, dtype=np.bool_), self._config, "meta"
        )
        _create(
            meta,
            "molecules_mask",
            np.ones(int(ptr[-1]), dtype=np.bool_),
            self._config,
            "meta",
        )
        _create(
            meta, "structure_ids", arrays.pop("structure_ids"), self._config, "meta"
        )
        for name, value in arrays.items():
            _create(core, name, value, self._config, "core")
        for name, tensor in batch.properties.items():
            _create(custom, name, _numpy(tensor), self._config, "custom")
        fields = {
            "meta": {"structure_ids": "system"},
            "core": {
                "conformer_indices": "molecule",
                "rotations": "molecule",
                "fractional_centers": "molecule",
                "cells": "system",
                "space_groups": "system",
                "z": "system",
                "z_prime": "system",
            },
            "custom": {name: "system" for name in batch.properties},
        }
        root.attrs.update(
            {
                "representation": _REPRESENTATION,
                "schema_version": _SCHEMA_VERSION,
                "num_samples": rows,
                "fields": fields,
            }
        )

    def write(self, structures: RigidMoleculeASUBatch) -> None:
        """Create a new store from rigid-molecule ASU structures.

        Identical repeated structure IDs in the input are stored once. Conflicting
        rows with the same ID fail before the store is created. This method does
        not modify ``structures``.

        Parameters
        ----------
        structures : RigidMoleculeASUBatch
            ASU structures and their versioned packing input.

        Raises
        ------
        FileExistsError
            If the target filesystem path already exists, including an empty
            directory, or an existing store is found.
        ValueError
            If duplicate IDs conflict, an ID is negative, or a custom property
            name is unsupported by Zarr.
        OverflowError
            If molecule pointers exceed the on-disk int32 range.
        """
        self._check_open()
        _validate_nonnegative_ids(structures)
        _validate_property_names(structures.properties)
        if isinstance(self._store, (str, Path)) and Path(self._store).exists():
            raise FileExistsError(f"Zarr store already exists at {self._store}")
        try:
            root = self._open("r")
        except Exception:
            root = None
        if root is not None:
            raise FileExistsError(f"Zarr store already exists at {self._store}")
        structures = _unique_batch(structures)
        if (
            structures.structure_molecule_ptr.numel()
            and int(structures.structure_molecule_ptr[-1]) > _MAX_POINTER
        ):
            raise OverflowError("molecules_ptr exceeds int32 capacity")
        _preflight_config(structures, self._config)
        self._write_new(structures)
        self._cache = self._build_cache(self._open("r"))

    def append(self, structures: RigidMoleculeASUBatch) -> None:
        """Append new ASU rows or skip rows already stored with identical IDs
        and values.

        Appending the same stored row again has no effect. A reused ID with
        different values fails before any mutation. IDs of deleted rows may be
        reused. Recovery from interrupted storage I/O is outside this API's
        guarantees. The input batch is not modified.

        Parameters
        ----------
        structures : RigidMoleculeASUBatch
            Rows to append, with the same packing input and property names,
            dtypes, and per-structure shapes.

        Raises
        ------
        ValueError
            If IDs conflict, are negative, or the packing input or property
            names, dtypes, or per-structure shapes differ from the store.
        OverflowError
            If the resulting molecule pointers exceed the on-disk int32 range.
        """
        self._check_open()
        _validate_nonnegative_ids(structures)
        _validate_property_names(structures.properties)
        root = self._open("r+")
        cache = self._cache
        if cache is None:
            cache = self._build_cache(root)
        existing_input, property_schema, active_by_id = cache
        if not _same_input(existing_input, structures.packing_input):
            raise ValueError("packing_input does not exactly match the stored input")
        existing_props = set(property_schema)
        incoming_props = set(structures.properties)
        if existing_props != incoming_props:
            raise ValueError("property keys do not match the stored schema")
        incoming_props_np = {
            name: _numpy(value) for name, value in structures.properties.items()
        }
        for name in existing_props:
            value = incoming_props_np[name]
            dtype, trailing_shape = property_schema[name]
            if dtype != value.dtype or trailing_shape != value.shape[1:]:
                raise ValueError(f"property {name!r} dtype or trailing shape differs")

        incoming = _batch_arrays(structures)
        incoming_ids = incoming["structure_ids"]
        pending_by_id: dict[tuple[int, int], int] = {}
        stored_rows: dict[
            tuple[int, int], tuple[dict[str, np.ndarray], dict[str, np.ndarray]]
        ] = {}
        append_indices: list[int] = []
        for index in range(structures.num_structures):
            key = _id_key(incoming_ids, index)
            prior = active_by_id.get(key)
            if prior is not None:
                stored = stored_rows.get(key)
                if stored is None:
                    stored = _read_store_row(root, prior)
                    stored_rows[key] = stored
                same = _row_equal(
                    stored[0], stored[1], 0, incoming, incoming_props_np, index
                )
            elif key in pending_by_id:
                same = _row_equal(
                    incoming,
                    incoming_props_np,
                    pending_by_id[key],
                    incoming,
                    incoming_props_np,
                    index,
                )
            else:
                pending_by_id[key] = index
                append_indices.append(index)
                continue
            if not same:
                raise ValueError(
                    f"structure ID {key} conflicts with an existing different row"
                )

        if not append_indices:
            self._cache = cache
            return
        old_physical = int(root.attrs["num_samples"])
        append_index = torch.tensor(
            append_indices, dtype=torch.int64, device=structures.cells.device
        )
        selected = structures.select(append_index)
        arrays = _batch_arrays(selected)
        additions = selected.num_structures
        molecule_offset = int(root["meta"]["molecules_ptr"][-1])
        if molecule_offset + int(arrays["molecules_ptr"][-1]) > _MAX_POINTER:
            raise OverflowError("molecules_ptr would exceed int32 capacity")
        selected_arrays = {
            name: value for name, value in arrays.items() if name != "molecules_ptr"
        }

        # All validation and capacity checks precede the first mutation. Any
        # storage error after this point may leave partial writes, so force a
        # full schema/ID validation before the next append attempt.
        try:
            meta = root["meta"]
            _extend(
                meta["molecules_ptr"],
                (arrays["molecules_ptr"][1:] + molecule_offset).astype(np.int32),
            )
            _extend(meta["samples_mask"], np.ones(additions, dtype=np.bool_))
            _extend(
                meta["molecules_mask"],
                np.ones(int(arrays["molecules_ptr"][-1]), dtype=np.bool_),
            )
            _extend(meta["structure_ids"], selected_arrays.pop("structure_ids"))
            for name, value in selected_arrays.items():
                _extend(root["core"][name], value)
            for name, value in incoming_props_np.items():
                _extend(root["custom"][name], value[append_indices])
            root.attrs["num_samples"] = old_physical + additions
        except Exception:
            self._cache = None
            raise
        active_by_id = dict(active_by_id)
        for offset, index in enumerate(append_indices):
            active_by_id[_id_key(incoming_ids, index)] = old_physical + offset
        self._cache = (existing_input, property_schema, active_by_id)

    def delete(self, indices: torch.Tensor) -> None:
        """Remove rows from the active set and overwrite their stored values with
        zeros.

        Parameters
        ----------
        indices : torch.Tensor
            One-dimensional int32/int64 indices in the current sequence of
            undeleted rows. Deleting logical row 0 makes the old row 1 the new
            row 0; repeated indices delete a row once.

        Raises
        ------
        IndexError
            If any index is outside the active logical row range.
        TypeError
            If ``indices`` is not a one-dimensional int32/int64 tensor.
        """
        self._check_open()
        index = _indices(indices, "indices")
        root = self._open("r+")
        cache = self._cache
        if cache is None:
            cache = self._build_cache(root)
        logical_to_physical = np.flatnonzero(root["meta"]["samples_mask"][:])
        if index.size and (index.min() < 0 or index.max() >= logical_to_physical.size):
            raise IndexError("delete indices must address active logical rows")
        if not index.size:
            self._cache = cache
            return
        physical_rows = np.unique(logical_to_physical[index])
        mask = root["meta"]["samples_mask"][:].astype(bool)
        ptr = root["meta"]["molecules_ptr"][:]
        molecules_mask = root["meta"]["molecules_mask"][:].astype(bool)
        deleted_rows = set(map(int, physical_rows))
        updated_ids = {
            key: row for key, row in cache[2].items() if row not in deleted_rows
        }
        try:
            for row_value in physical_rows:
                row = int(row_value)
                mask[row] = False
                start, end = map(int, ptr[row : row + 2])
                molecules_mask[start:end] = False
                for name in root["core"].array_keys():
                    arr = root["core"][name]
                    if _fields_alignment(root, name) == "molecule":
                        arr[start:end] = np.zeros_like(arr[start:end])
                    else:
                        arr[row] = np.zeros_like(arr[row])
                for name in root["custom"].array_keys():
                    arr = root["custom"][name]
                    arr[row] = np.zeros_like(arr[row])
                root["meta"]["structure_ids"][row] = np.zeros_like(
                    root["meta"]["structure_ids"][row]
                )
            root["meta"]["samples_mask"][:] = mask
            root["meta"]["molecules_mask"][:] = molecules_mask
        except Exception:
            self._cache = None
            raise
        self._cache = (cache[0], cache[1], updated_ids)

    def defragment(
        self, config: ZarrWriteConfig | Mapping[str, Any] | None = None
    ) -> None:
        """Rewrite active rows densely in their current logical order.

        Deleted rows are removed, so physical row indices may change. Existing
        readers should call :meth:`RigidMoleculeASUZarrReader.refresh` after defragmentation.
        The packing input and property schema are retained. ``config`` replaces
        this writer's settings after the new store has been installed. Only local
        filesystem stores are supported; callers must provide exclusive access.

        Parameters
        ----------
        config : ZarrWriteConfig or mapping, optional
            Chunking and compression settings for the rewritten arrays.

        Raises
        ------
        ValueError
            If any configured array layout is invalid.
        TypeError
            If the store is not a local filesystem path.
        RuntimeError
            If replacement fails, including when rollback restores the original.
            Backup-cleanup failure raises after replacement succeeds; the new
            store and writer configuration remain active.
        """
        self._check_open()
        if not isinstance(self._store, (str, Path)):
            raise TypeError("defragment supports local filesystem paths only")
        target = Path(self._store)
        if not target.exists() or not target.is_dir():
            raise FileNotFoundError(f"CSP Zarr directory does not exist: {target}")
        new_config = self._config if config is None else _config(config)
        reader = RigidMoleculeASUZarrReader(target)
        try:
            compact = reader.read()
        finally:
            reader.close()
        _preflight_config(compact, new_config)
        token = uuid4().hex
        staged = target.with_name(f".{target.name}.csp-stage-{token}")
        backup = target.with_name(f".{target.name}.csp-backup-{token}")
        try:
            RigidMoleculeASUZarrWriter(staged, new_config)._write_new(
                compact, mode="w-"
            )
            with RigidMoleculeASUZarrReader(staged) as staged_reader:
                validated = staged_reader.read()
                if not _same_batch(compact, validated):
                    raise ValueError("staged CSP Zarr payload differs from the source")
            os.replace(target, backup)
            try:
                os.replace(staged, target)
            except Exception as swap_error:
                try:
                    os.replace(backup, target)
                except Exception as rollback_error:
                    raise RuntimeError(
                        "defragment replacement and rollback failed; the original "
                        f"store remains recoverable at {backup}"
                    ) from rollback_error
                raise RuntimeError(
                    "defragment replacement failed; the original store was restored"
                ) from swap_error
        except Exception:
            if staged.exists():
                shutil.rmtree(staged, ignore_errors=True)
            raise

        # The replacement is now authoritative. Update the writer settings and
        # cache before best-effort cleanup so the live state matches the store.
        self._config = new_config
        new_root = self._open("r")
        self._cache = self._build_cache(new_root)
        try:
            shutil.rmtree(backup)
        except Exception as exc:
            raise RuntimeError(
                f"defragment completed but backup cleanup failed; "
                f"the previous store remains at {backup}"
            ) from exc

    def close(self) -> None:
        """Close this writer; subsequent operations raise ``RuntimeError``."""
        self._closed = True

    def __enter__(self) -> RigidMoleculeASUZarrWriter:
        """Return this open writer for use as a context manager."""
        self._check_open()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        """Close the writer when leaving its context."""
        self.close()


class RigidMoleculeASUZarrReader:
    """Read active rigid-molecule ASU representations from a CSP Zarr store.

    Parameters
    ----------
    store : StoreLike
        Path or Zarr-compatible store containing an existing CSP ASU dataset.

    Notes
    -----
    Selections index the current sequence of undeleted rows; call ``refresh``
    after a writer changes the store.

    ``read_batch`` requires an explicit selection to bound P1 expansion.
    Returned tensors are newly constructed and do not alias Zarr buffers.
    The reader snapshots root metadata and the active row mapping, not an
    immutable copy of stored payloads.
    """

    def __init__(self, store: StoreLike) -> None:
        """Open a CSP ASU store and snapshot its active row mapping."""
        self._store = store
        self._root: zarr.Group | None = None
        self._metadata: Mapping[str, Any] = MappingProxyType({})
        self._fields: list[str] = []
        self._active: np.ndarray = np.empty(0, dtype=np.int64)
        self._refresh()

    def _check_open(self) -> zarr.Group:
        """Return the current root group or raise if the reader is closed."""
        if self._root is None:
            raise RuntimeError("Cannot use a closed RigidMoleculeASUZarrReader")
        return self._root

    def _refresh(self) -> None:
        """Replace the reader snapshot with the store's current active rows."""
        try:
            root = zarr.open_group(self._store, mode="r")
        except Exception as exc:
            raise FileNotFoundError(
                f"Cannot open CSP Zarr store at {self._store}"
            ) from exc
        self._packing_input = _validate_root(root)
        self._root = root
        self._metadata = _freeze(dict(root.attrs))
        fields = root.attrs["fields"]
        self._fields = [
            *fields.get("meta", {}),
            *fields.get("core", {}),
            *fields.get("custom", {}),
        ]
        self._active = np.flatnonzero(root["meta"]["samples_mask"][:])

    def __len__(self) -> int:
        """Return the number of active logical rows in the current snapshot."""
        self._check_open()
        return int(self._active.size)

    @property
    def metadata(self) -> Mapping[str, object]:
        """Read-only copy of root attributes."""
        self._check_open()
        return self._metadata

    @property
    def field_names(self) -> list[str]:
        """Return stored field names in schema order."""
        self._check_open()
        return list(self._fields)

    def _resolve(self, indices: torch.Tensor | None) -> np.ndarray:
        """Map active logical row indices to physical store rows."""
        if indices is None:
            return self._active.copy()
        logical = _indices(indices, "indices")
        if logical.size and (logical.min() < 0 or logical.max() >= len(self)):
            raise IndexError("read indices must address active logical rows")
        return self._active[logical]

    def get_metadata(self, indices: torch.Tensor) -> torch.Tensor:
        """Return selected rows' atom and edge counts without P1 expansion.

        Parameters
        ----------
        indices : torch.Tensor
            One-dimensional int32/int64 indices in the active logical row space.

        Returns
        -------
        torch.Tensor
            int64 ``[selection_size, 2]`` metadata in selection order, including
            repeated indices. Column 0 is the full-cell atom count
            (formula-unit atom count times Z); column 1 is zero because the ASU
            store has no materialized neighbor edges.
        """
        root = self._check_open()
        physical = self._resolve(indices)
        z = _select_rows(root["core"]["z"], physical).astype(np.int64)
        atom_count = int(root["packing_input"]["atomic_numbers"].shape[0])
        result = np.stack((atom_count * z, np.zeros_like(z)), axis=1)
        return torch.from_numpy(result.copy()).to(dtype=torch.int64)

    def read(
        self,
        indices: torch.Tensor | None = None,
        *,
        device: torch.device | str = "cpu",
    ) -> RigidMoleculeASUBatch:
        """Load selected active rows as an ASU batch.

        Parameters
        ----------
        indices : torch.Tensor or None, optional
            One-dimensional int32/int64 logical row indices. Order and repeated
            indices are preserved. ``None`` selects every active row.
        device : torch.device or str, default="cpu"
            Device for returned tensors.

        Returns
        -------
        RigidMoleculeASUBatch
            An owned ASU batch; formula input and selected row payloads are
            copied from the store.
        """
        root = self._check_open()
        physical = self._resolve(indices)
        ptr_positions = np.unique(np.concatenate((physical, physical + 1)))
        selected_ptr_values = _select_rows(root["meta"]["molecules_ptr"], ptr_positions)
        ptr_by_position = dict(
            zip(ptr_positions.tolist(), selected_ptr_values.tolist(), strict=True)
        )
        starts = np.asarray(
            [ptr_by_position[int(row)] for row in physical], dtype=np.int64
        )
        stops = np.asarray(
            [ptr_by_position[int(row + 1)] for row in physical], dtype=np.int64
        )
        molecule_indices = (
            np.concatenate(
                [np.arange(a, b) for a, b in zip(starts, stops, strict=True)]
            )
            if physical.size
            else np.empty(0, dtype=np.int64)
        )
        counts = stops - starts
        selected_ptr = np.concatenate(
            (np.zeros(1, dtype=np.int64), np.cumsum(counts))
        ).astype(np.int32)
        formula = self._packing_input
        props = {
            name: torch.from_numpy(
                _select_rows(root["custom"][name], physical).copy()
            ).to(device)
            for name in root["custom"].array_keys()
        }
        return RigidMoleculeASUBatch._from_validated(
            packing_input=formula.to(device),
            structure_molecule_ptr=torch.from_numpy(selected_ptr).to(device),
            conformer_indices=torch.from_numpy(
                _select_rows(root["core"]["conformer_indices"], molecule_indices).copy()
            ).to(device),
            rotations=torch.from_numpy(
                _select_rows(root["core"]["rotations"], molecule_indices).copy()
            ).to(device),
            fractional_centers=torch.from_numpy(
                _select_rows(
                    root["core"]["fractional_centers"], molecule_indices
                ).copy()
            ).to(device),
            cells=torch.from_numpy(
                _select_rows(root["core"]["cells"], physical).copy()
            ).to(device),
            space_groups=torch.from_numpy(
                _select_rows(root["core"]["space_groups"], physical).copy()
            ).to(device),
            z=torch.from_numpy(_select_rows(root["core"]["z"], physical).copy()).to(
                device
            ),
            z_prime=torch.from_numpy(
                _select_rows(root["core"]["z_prime"], physical).copy()
            ).to(device),
            structure_ids=torch.from_numpy(
                _select_rows(root["meta"]["structure_ids"], physical).copy()
            ).to(device),
            properties=MappingProxyType(props),
        )

    def read_batch(
        self, indices: torch.Tensor, *, device: torch.device | str = "cpu"
    ) -> Batch:
        """Expand explicitly selected active logical rows into a Toolkit P1 batch.

        Parameters
        ----------
        indices : torch.Tensor
            Required one-dimensional int32/int64 indices in active logical row
            space. Order and repeated indices are preserved. ``None`` is rejected
            so callers must bound potentially large P1 expansion.
        device : torch.device or str, default="cpu"
            Device for the returned batch.

        Returns
        -------
        Batch
            A newly expanded P1 batch carrying source structure IDs.

        Raises
        ------
        TypeError
            If ``indices`` is ``None`` or is not a one-dimensional int32/int64
            tensor.
        IndexError
            If an index is outside the active logical row range.
        """
        if indices is None:
            raise TypeError("indices is required for P1 expansion")
        compact = self.read(indices, device=device)
        return compact.to_batch(device=device)

    def refresh(self) -> None:
        """Reload reader metadata and active rows after appends, deletions, or
        defragmentation.

        Existing selected results remain independent tensors; this reader's
        logical-to-physical mapping and root metadata are replaced.
        """
        self._check_open()
        self._refresh()

    def close(self) -> None:
        """Close this reader; subsequent operations raise ``RuntimeError``."""
        self._root = None

    def __enter__(self) -> RigidMoleculeASUZarrReader:
        """Return this open reader for use as a context manager."""
        self._check_open()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        """Close the reader when leaving its context."""
        self.close()


def _indices(indices: torch.Tensor, name: str) -> np.ndarray:
    """Validate one-dimensional integer indices and expose them as int64 NumPy.

    Contiguous CPU int64 storage may be shared.
    """
    if not isinstance(indices, torch.Tensor) or indices.ndim != 1:
        raise TypeError(f"{name} must be a one-dimensional torch.Tensor")
    if indices.dtype not in (torch.int32, torch.int64):
        raise TypeError(f"{name} must have dtype torch.int32 or torch.int64")
    return _numpy(indices).astype(np.int64, copy=False)


def _extend(array: Any, values: np.ndarray) -> None:
    """Append first-axis values to a resizable Zarr array."""
    if not values.shape[0]:
        return
    start = array.shape[0]
    array.resize((start + values.shape[0], *array.shape[1:]))
    array[start:] = values


def _read_input(group: zarr.Group) -> MolecularPackingInput:
    """Reconstruct and validate version-1 molecular packing input."""
    attrs = group.attrs
    if int(attrs.get("version", -1)) != 1:
        raise ValueError("unsupported packing_input version")
    if "formula_unit_volume" not in attrs or "metadata" not in attrs:
        raise ValueError("packing_input is missing versioned state attributes")
    state: dict[str, Any] = {
        "version": 1,
        **{name: torch.from_numpy(group[name][:].copy()) for name in _FORMULA_FIELDS},
        "formula_unit_volume": attrs["formula_unit_volume"],
        "metadata": attrs["metadata"],
    }
    try:
        return MolecularPackingInput.from_state_dict(state)
    except (TypeError, ValueError, KeyError) as exc:
        raise ValueError("packing_input version-1 state is invalid") from exc


def _fields_alignment(root: zarr.Group, name: str) -> str:
    """Return the stored row alignment for a core field."""
    return str(root.attrs["fields"].get("core", {}).get(name, "system"))


def _validate_root(root: zarr.Group) -> MolecularPackingInput:
    """Validate the CSP ASU schema and return its packing input."""
    attrs = root.attrs
    if attrs.get("representation") != _REPRESENTATION:
        raise ValueError("Zarr store is not a CSP rigid-molecule ASU representation")
    if attrs.get("schema_version") != _SCHEMA_VERSION:
        raise ValueError("unsupported CSP Zarr schema_version")
    if not all(name in root for name in ("meta", "core", "custom", "packing_input")):
        raise ValueError("CSP Zarr store is missing a required group")
    required_meta = {"molecules_ptr", "samples_mask", "molecules_mask", "structure_ids"}
    required_core = {
        "conformer_indices",
        "rotations",
        "fractional_centers",
        "cells",
        "space_groups",
        "z",
        "z_prime",
    }
    if (
        set(root["meta"].array_keys()) != required_meta
        or set(root["core"].array_keys()) != required_core
    ):
        raise ValueError("CSP Zarr store has an invalid v1 field set")
    n = int(attrs.get("num_samples", -1))
    ptr = root["meta"]["molecules_ptr"]
    masks = root["meta"]["samples_mask"]
    molecule_mask = root["meta"]["molecules_mask"]
    ids = root["meta"]["structure_ids"]
    if n < 0 or ptr.shape != (n + 1,) or masks.shape != (n,) or ids.shape != (n, 2):
        raise ValueError("CSP Zarr sample arrays have inconsistent shapes")
    ptr_values = ptr[:]
    if (
        ptr.dtype != np.dtype("int32")
        or ptr_values[0] != 0
        or np.any(np.diff(ptr_values) < 0)
    ):
        raise ValueError("molecules_ptr must be monotone int32 and start at zero")
    q = int(ptr_values[-1])
    if (
        molecule_mask.shape != (q,)
        or ids.dtype != np.dtype("int64")
        or masks.dtype != np.dtype("bool")
    ):
        raise ValueError("CSP Zarr masks or structure_ids have invalid dtype or shape")
    if molecule_mask.dtype != np.dtype("bool"):
        raise ValueError("molecules_mask must have bool dtype")
    ids_values = ids[:]
    if np.any(ids_values < 0):
        raise ValueError("structure_ids must be nonnegative")
    if len({tuple(row) for row in ids_values[masks[:].astype(bool)]}) != int(
        masks[:].sum()
    ):
        raise ValueError("active structure_ids must be unique")

    expected_dtype = {
        "conformer_indices": np.dtype("int32"),
        "rotations": np.dtype("float32"),
        "fractional_centers": np.dtype("float32"),
        "cells": np.dtype("float32"),
        "space_groups": np.dtype("int32"),
        "z": np.dtype("int32"),
        "z_prime": np.dtype("int32"),
    }
    expected_shapes = {
        "conformer_indices": (q,),
        "rotations": (q, 3, 3),
        "fractional_centers": (q, 3),
        "cells": (n, 3, 3),
        "space_groups": (n,),
        "z": (n,),
        "z_prime": (n,),
    }
    for name, shape in expected_shapes.items():
        array = root["core"][name]
        if array.shape != shape or array.dtype != expected_dtype[name]:
            raise ValueError(f"{name} has invalid dtype or shape")

    input_group = root["packing_input"]
    if not set(_FORMULA_FIELDS).issubset(input_group.array_keys()):
        raise ValueError("packing_input is missing required arrays")
    packing_input = _read_input(input_group)

    fields = attrs.get("fields")
    if not isinstance(fields, Mapping):
        raise ValueError("CSP Zarr store is missing its fields schema")
    if set(fields) != {"meta", "core", "custom"}:
        raise ValueError("CSP Zarr fields schema has invalid groups")
    if set(fields.get("meta", {})) != {"structure_ids"}:
        raise ValueError("fields schema does not declare structure_ids")
    if set(fields.get("core", {})) != set(root["core"].array_keys()):
        raise ValueError("fields schema does not match core arrays")
    if set(fields.get("custom", {})) != set(root["custom"].array_keys()):
        raise ValueError("fields schema does not match custom arrays")
    if fields.get("meta", {}).get("structure_ids") != "system":
        raise ValueError("structure_ids must be declared system-aligned")
    expected_core_alignment = {
        "conformer_indices": "molecule",
        "rotations": "molecule",
        "fractional_centers": "molecule",
        "cells": "system",
        "space_groups": "system",
        "z": "system",
        "z_prime": "system",
    }
    if fields.get("core") != expected_core_alignment:
        raise ValueError("core fields have invalid molecule/system alignment")
    for name in root["custom"].array_keys():
        if root["custom"][name].shape[0] != n:
            raise ValueError(
                f"custom property {name!r} length does not match num_samples"
            )
        if fields["custom"].get(name) != "system":
            raise ValueError(f"custom property {name!r} must be system-aligned")
    return packing_input
