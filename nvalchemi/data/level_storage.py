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
"""Unified tensor-based data containers for ML workflows on atomic simulations.

This module provides efficient, torch-only data management for molecular dynamics
and quantum chemistry calculations. Data is organized by **level**: per-atom (node),
per-edge (bond/neighbor), and per-system (graph) tensors.

Data model
----------

Conceptually, a batched atomic dataset is a set of tensors split by level. Each
level has either a **uniform** layout (one row per system, e.g. system-level
fields like ``cell``, ``energy``) or a **segmented** layout (concatenated
variable-length segments, e.g. per-atom ``positions``, per-edge ``neighbor_list``).
The following diagram shows how the classes fit together::

    ┌─────────────────────────────────────────────────────────────────────────┐
    │ LevelSchema                                                             │
    │   Maps tensor names → level ("atoms" | "edges" | "system"), dtype,       │
    │   and whether that level is segmented. No tensors; schema only.         │
    └─────────────────────────────────────────────────────────────────────────┘
                                        │
                                        │ used by
                                        ▼
    ┌─────────────────────────────────────────────────────────────────────────┐
    │ MultiLevelStorage                                                         │
    │   Holds one container per level; routes name → correct container.          │
    │   Typical keys: "atoms", "edges", "system".                               │
    └─────────────────────────────────────────────────────────────────────────┘
         │                    │                    │
         │ atoms              │ edges              │ system
         ▼                    ▼                    ▼
    ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
    │ SegmentedLevel  │ │ SegmentedLevel  │ │ UniformLevel    │
    │ Storage         │ │ Storage         │ │ Storage         │
    │ (variable len)  │ │ (variable len)  │ │ (one per system)│
    └────────┬────────┘ └────────┬────────┘ └────────┬────────┘
             │                   │                   │
             └───────────────────┼───────────────────┘
                                 │
                       inherit from
                                 ▼
                       ┌──────────────────┐
                       │ BaseLevelStorage │
                       │ (ABC: one level, │
                       │  dict-like)      │
                       └──────────────────┘

Classes
-------
LevelSchema
    Registry that maps attribute names to levels (groups), dtypes, and
    segmentation flags. In chemistry terms: which level (atom / edge / system)
    each tensor belongs to.
BaseLevelStorage
    Abstract base for a single level: dict-like container of tensors with
    uniform or segmented layout.
UniformLevelStorage
    One level with **uniform** first dimension (e.g. system-level: one row
    per system).
SegmentedLevelStorage
    One level with **segmented** layout: concatenated variable-length segments
    and batch pointer bookkeeping (e.g. atom-level, edge-level).
MultiLevelStorage
    Multi-level container: holds ``BaseLevelStorage`` instances keyed by level
    (e.g. ``"atoms"``, ``"edges"``, ``"system"``) and routes attribute access.

The class names use a **Level** / **Storage** convention: ``LevelSchema`` for the
registry, ``BaseLevelStorage`` for the abstract single-level container, and
``UniformLevelStorage`` / ``SegmentedLevelStorage`` / ``MultiLevelStorage`` for
the concrete implementations.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Iterator
from typing import Any

import numpy as np
import torch
import warp as wp
from tensordict import TensorDict
from torch import Tensor

from nvalchemi.data.buffer_kernels import (
    TORCH_TO_WP,
    compute_put_fit_mask_per_system,
    compute_put_fit_mask_segmented,
    defrag_per_system,
    defrag_segmented,
    put_masked_per_system,
    put_masked_segmented,
)

wp.config.quiet = True
try:
    wp.init()
except RuntimeError as e:
    raise RuntimeError(
        "Failed to initialize warp, likely due to missing drivers and/or devices."
        " Make sure you have the correct CUDA version, and that GPUs are available."
    ) from e

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
IndexType = int | slice | Tensor | list
DeviceType = torch.device | str
# Segment/graph index dtype for SegmentedLevelStorage (matches segment_lengths).
INDEX_DTYPE = torch.int32

# ---------------------------------------------------------------------------
# Dtype mapping constants
# ---------------------------------------------------------------------------
TORCH_DTYPE_MAP: dict[str, torch.dtype] = {
    "float16": torch.float16,
    "float32": torch.float32,
    "float64": torch.float64,
    "half": torch.float16,
    "float": torch.float32,
    "double": torch.float64,
    "int8": torch.int8,
    "int16": torch.int16,
    "int32": torch.int32,
    "int64": torch.int64,
    "byte": torch.int8,
    "short": torch.int16,
    "int": torch.int32,
    "long": torch.int64,
    "uint8": torch.uint8,
    "uint16": torch.uint16,
    "uint32": torch.uint32,
    "uint64": torch.uint64,
    "ubyte": torch.uint8,
    "complex64": torch.complex64,
    "complex128": torch.complex128,
    "cfloat": torch.complex64,
    "cdouble": torch.complex128,
    "bool": torch.bool,
    "bool_": torch.bool,
}

TORCH_DTYPE_MAP_INVERSE: dict[torch.dtype, str] = {
    torch.float16: "float16",
    torch.float32: "float32",
    torch.float64: "float64",
    torch.int8: "int8",
    torch.int16: "int16",
    torch.int32: "int32",
    torch.int64: "int64",
    torch.uint8: "uint8",
    torch.uint16: "uint16",
    torch.uint32: "uint32",
    torch.uint64: "uint64",
    torch.complex64: "complex64",
    torch.complex128: "complex128",
    torch.bool: "bool",
}

# ---------------------------------------------------------------------------
# Domain-specific defaults (aligned with nvalchemi naming conventions)
# ---------------------------------------------------------------------------
DEFAULT_ATTRIBUTE_MAP: dict[str, set[str]] = {
    "atoms": {
        "positions",
        "atomic_numbers",
        "forces",
        "velocities",
        "charges",
        "atomic_masses",
    },
    "edges": {
        "neighbor_list",
        "edge_embeddings",
        "shifts",
        "neighbor_list_shifts",
    },
    "system": {
        "cell",
        "pbc",
        "energy",
        "dipole",
        "stress",
        "virial",
    },
}

DEFAULT_DTYPES: dict[str, str] = {
    "positions": "float32",
    "atomic_numbers": "int64",
    "forces": "float32",
    "velocities": "float32",
    "charges": "float32",
    "atomic_masses": "float32",
    "neighbor_list": "int64",
    "edge_embeddings": "float32",
    "shifts": "float32",
    "neighbor_list_shifts": "float32",
    "cell": "float32",
    "pbc": "bool",
    "energy": "float64",
    "dipole": "float32",
    "stress": "float32",
    "virial": "float32",
}

DEFAULT_SEGMENTED_GROUPS: set[str] = {"atoms", "edges"}


# ---------------------------------------------------------------------------
# LevelSchema
# ---------------------------------------------------------------------------
class LevelSchema:
    """Registry mapping attribute names to groups, dtypes, and segmentation flags.

    Parameters
    ----------
    group_to_attrs : dict[str, set[str]], optional
        Mapping from group names to sets of attribute names.
        Defaults to ``DEFAULT_ATTRIBUTE_MAP``.
    segmented_groups : set[str], optional
        Group names whose data has variable segment lengths.
        Defaults to ``DEFAULT_SEGMENTED_GROUPS``.
    dtypes : dict[str, str], optional
        Per-attribute dtype strings (keys of ``TORCH_DTYPE_MAP``).
        Defaults to ``DEFAULT_DTYPES``.

    Raises
    ------
    ValueError
        If *dtypes* is provided but its keys don't match the attribute set.
    """

    def __init__(
        self,
        group_to_attrs: dict[str, set[str]] | None = None,
        segmented_groups: set[str] | None = None,
        dtypes: dict[str, str] | None = None,
    ) -> None:
        if group_to_attrs is None:
            group_to_attrs = DEFAULT_ATTRIBUTE_MAP
        self.group_to_attrs: dict[str, set[str]] = {
            k: v.copy() for k, v in group_to_attrs.items()
        }
        self.attr_to_group: dict[str, str] = {
            attr: group
            for group, attrs in self.group_to_attrs.items()
            for attr in attrs
        }
        self.segmented_groups: set[str] = (
            segmented_groups
            if segmented_groups is not None
            else DEFAULT_SEGMENTED_GROUPS.copy()
        )

        if dtypes is not None:
            provided = set(dtypes.keys())
            expected = set(self.attr_to_group.keys())
            if provided != expected:
                raise ValueError(
                    f"dtype keys must match attribute set. Missing: {expected - provided}, "
                    f"extra: {provided - expected}"
                )
            self.dtypes: dict[str, str] = dtypes
        else:
            self.dtypes = DEFAULT_DTYPES.copy()

    # -- Mutation -----------------------------------------------------------

    def set(
        self,
        attr_name: str,
        group_name: str,
        dtype: str | torch.dtype | None = None,
        is_segmented: bool | None = None,
    ) -> None:
        """Register or update an attribute's group, dtype, and segmentation.

        Parameters
        ----------
        attr_name : str
            Name of the attribute.
        group_name : str
            Target group for the attribute.
        dtype : str or torch.dtype, optional
            Data type for the attribute.
        is_segmented : bool, optional
            Whether *group_name* should be marked as segmented.
        """
        existing_group = self.attr_to_group.get(attr_name)
        if existing_group is not None and existing_group != group_name:
            self.group_to_attrs[existing_group].discard(attr_name)

        self.attr_to_group[attr_name] = group_name
        if group_name not in self.group_to_attrs:
            self.group_to_attrs[group_name] = set()
        self.group_to_attrs[group_name].add(attr_name)

        if is_segmented is not None:
            if is_segmented:
                self.segmented_groups.add(group_name)
            else:
                self.segmented_groups.discard(group_name)

        if dtype is not None:
            self.dtypes[attr_name] = (
                TORCH_DTYPE_MAP_INVERSE[dtype]
                if isinstance(dtype, torch.dtype)
                else dtype
            )

    def mark_group_segmented(self, group_name: str) -> None:
        """Mark *group_name* as segmented."""
        self.segmented_groups.add(group_name)

    def unmark_group_segmented(self, group_name: str) -> None:
        """Remove the segmented flag from *group_name*.

        Raises
        ------
        KeyError
            If *group_name* is not currently segmented.
        """
        self.segmented_groups.remove(group_name)

    # -- Queries ------------------------------------------------------------

    def is_segmented_group(self, group_name: str) -> bool:
        """Return whether *group_name* is segmented."""
        return group_name in self.segmented_groups

    def is_segmented_attr(self, attr_name: str) -> bool:
        """Return whether *attr_name* belongs to a segmented group.

        Raises
        ------
        KeyError
            If *attr_name* is not registered.
        """
        if attr_name not in self.attr_to_group:
            raise KeyError(f"Attribute '{attr_name}' not found")
        return self.attr_to_group[attr_name] in self.segmented_groups

    def group(self, attr_name: str) -> str:
        """Return the group name for *attr_name*.

        Raises
        ------
        KeyError
            If *attr_name* is not registered.
        """
        if attr_name not in self.attr_to_group:
            raise KeyError(f"Attribute '{attr_name}' not found")
        return self.attr_to_group[attr_name]

    def dtype(self, attr_name: str) -> str:
        """Return the dtype string for *attr_name*.

        Raises
        ------
        KeyError
            If *attr_name* has no registered dtype.
        """
        if attr_name not in self.dtypes:
            raise KeyError(f"Attribute '{attr_name}' not found in dtype registry")
        return self.dtypes[attr_name]

    # -- Copy ---------------------------------------------------------------

    def clone(self) -> LevelSchema:
        """Return an independent deep copy."""
        cloned = LevelSchema.__new__(LevelSchema)
        cloned.group_to_attrs = {g: a.copy() for g, a in self.group_to_attrs.items()}
        cloned.attr_to_group = self.attr_to_group.copy()
        cloned.segmented_groups = self.segmented_groups.copy()
        cloned.dtypes = self.dtypes.copy()
        return cloned


# ---------------------------------------------------------------------------
# warp-accelerated helper
# ---------------------------------------------------------------------------
_INT32_MAX: int = 2**31 - 1


@wp.kernel(enable_backward=False)
def _expand_segments_kernel(
    starts: wp.array(dtype=Any),
    lengths: wp.array(dtype=Any),
    offsets: wp.array(dtype=Any),
    output: wp.array(dtype=Any),
) -> None:
    """Write ``range(starts[tid], starts[tid]+lengths[tid])`` at ``offsets[tid]``.

    Parameters
    ----------
    starts : wp.array
        Per-segment start element index.
    lengths : wp.array
        Per-segment element count.
    offsets : wp.array
        Per-segment write offset into *output*.
    output : wp.array
        Pre-allocated 1-D buffer receiving the expanded element indices.
    """
    tid = wp.tid()
    s = starts[tid]
    n = lengths[tid]
    o = offsets[tid]
    for _i in range(wp.int32(n)):
        i = type(s)(_i)
        output[o + i] = s + i


_WP_INT_TYPES = [wp.int32, wp.int64]

_expand_segments_overloads: dict[type, Any] = {}
for _wp_t in _WP_INT_TYPES:
    _expand_segments_overloads[_wp_t] = wp.overload(
        _expand_segments_kernel,
        [
            wp.array(dtype=_wp_t),
            wp.array(dtype=_wp_t),
            wp.array(dtype=_wp_t),
            wp.array(dtype=_wp_t),
        ],
    )

_TORCH_TO_WP: dict[torch.dtype, type] = {
    torch.int32: wp.int32,
    torch.int64: wp.int64,
}


def _expand_segments_warp(
    seg_idx: torch.Tensor,
    batch_ptr: torch.Tensor,
    device: torch.device,
    index_dtype: torch.dtype,
) -> torch.Tensor:
    """Expand segment indices to element indices using a Warp kernel.

    The kernel overload matching *index_dtype* is selected automatically,
    so no dtype conversion is performed on the input tensors.

    Parameters
    ----------
    seg_idx : torch.Tensor
        1-D tensor of selected segment indices (on *device*).
    batch_ptr : torch.Tensor
        Cumulative segment pointer of length ``num_segments + 1``.
    device : torch.device
        Target device (CPU or CUDA) for the kernel launch.
    index_dtype : torch.dtype
        Integer dtype (``torch.int32`` or ``torch.int64``) for the output
        tensor and kernel selection.

    Returns
    -------
    torch.Tensor
        1-D tensor of element-level indices on *device*.

    Raises
    ------
    ValueError
        If *index_dtype* is not ``torch.int32`` or ``torch.int64``.
    """
    wp_dtype = _TORCH_TO_WP.get(index_dtype)
    if wp_dtype is None:
        raise ValueError(
            f"Unsupported index_dtype {index_dtype}; "
            f"expected torch.int32 or torch.int64"
        )

    kernel = _expand_segments_overloads[wp_dtype]

    starts = batch_ptr[seg_idx]
    ends = batch_ptr[seg_idx + 1]
    lengths = ends - starts

    cumlen = torch.cumsum(lengths, dim=0, dtype=index_dtype)
    total = int(cumlen[-1].item())
    if total > _INT32_MAX:
        raise ValueError(
            f"Total element count {total} exceeds int32 maximum "
            f"({_INT32_MAX}); the Warp kernel uses int32 loop bounds"
        )
    if total == 0:
        return torch.empty(0, device=device, dtype=index_dtype)

    offsets = cumlen - lengths
    output = torch.empty(total, device=device, dtype=index_dtype)

    if device.type == "cuda":
        wp_device = f"cuda:{device.index or 0}"
    else:
        wp_device = "cpu"
    wp.launch(
        kernel=kernel,
        dim=seg_idx.numel(),
        inputs=[
            wp.from_torch(starts, dtype=wp_dtype),
            wp.from_torch(lengths, dtype=wp_dtype),
            wp.from_torch(offsets, dtype=wp_dtype),
            wp.from_torch(output, dtype=wp_dtype),
        ],
        device=wp_device,
    )

    return output


# ---------------------------------------------------------------------------
# Tensor utility helpers
# ---------------------------------------------------------------------------
def to_tensor(
    value: Any,
    device: DeviceType | None = None,
    dtype: str | None = None,
) -> Tensor:
    """Convert an arbitrary value to a :class:`torch.Tensor`.

    Accepts tensors, numpy arrays, lists, scalars, and anything that
    :func:`torch.as_tensor` supports.

    Parameters
    ----------
    value : Any
        Value to convert.
    device : DeviceType, optional
        Target device.
    dtype : str, optional
        Dtype string (looked up in ``TORCH_DTYPE_MAP``).

    Returns
    -------
    Tensor

    Raises
    ------
    TypeError
        If *value* cannot be converted to a tensor.
    """
    if isinstance(value, list) and len(value) > 0 and isinstance(value[0], np.ndarray):
        value = np.array(value)

    # Only apply the schema dtype for non-tensor inputs (lists, numpy arrays,
    # scalars) where there is no existing dtype to preserve.  When the input
    # is already a Tensor, respect its dtype — the user chose it deliberately.
    if dtype and not isinstance(value, Tensor):
        torch_dtype = TORCH_DTYPE_MAP[dtype]
    else:
        torch_dtype = None
    return torch.as_tensor(value, device=device, dtype=torch_dtype)


def _clone_tensor(tensor: Tensor) -> Tensor:
    """Return a detached clone of *tensor*."""
    return tensor.clone()


def _concatenate_tensors(t1: Tensor, t2: Tensor) -> Tensor:
    """Concatenate two tensors along dim 0, coercing *t2* to *t1*'s device/dtype.

    Parameters
    ----------
    t1, t2 : Tensor
        Tensors with matching shapes except for the first dimension.

    Returns
    -------
    Tensor

    Raises
    ------
    ValueError
        If trailing shapes don't match.
    """
    if t1.shape[1:] != t2.shape[1:]:
        raise ValueError(
            f"Shape mismatch for concatenation: {t1.shape[1:]} vs {t2.shape[1:]}"
        )
    return torch.cat([t1, t2.to(device=t1.device, dtype=t1.dtype)], dim=0)


def _validate_trailing_shapes(tensors: list[Tensor]) -> bool:
    """Return ``True`` if all tensors share the same shape after dim 0."""
    if not tensors:
        return True
    trailing = tensors[0].shape[1:]
    return all(t.shape[1:] == trailing for t in tensors)


# ---------------------------------------------------------------------------
# BaseLevelStorage (ABC)
# ---------------------------------------------------------------------------
class BaseLevelStorage(ABC):
    """Abstract base for dict-like tensor containers.

    Subclasses must implement ``__len__``, ``select``, ``update_at``,
    ``concatenate``, ``is_segmented``, and ``_validate_setitem``.

    Parameters
    ----------
    data : dict[str, Tensor], optional
        Initial attribute tensors keyed by name.
    device : DeviceType, optional
        Target device (defaults to ``"cpu"``).
    attr_map : LevelSchema, optional
        Shared attribute registry.
    validate : bool
        Whether to run shape checks on every ``__setitem__``.
    """

    def __init__(
        self,
        data: dict[str, Tensor] | TensorDict | None = None,
        device: DeviceType | None = None,
        attr_map: LevelSchema | None = None,
        validate: bool = True,
    ) -> None:
        self._attr_map = attr_map if attr_map else LevelSchema()
        self.device = torch.device(device) if device else torch.device("cpu")
        self.validate = validate

        if data is None:
            self._data = TensorDict({}, batch_size=torch.Size([0]), device=self.device)
        elif isinstance(data, TensorDict):
            self._data = data
        else:
            if not isinstance(data, dict):
                raise TypeError(
                    f"data must be a dict or TensorDict, got {type(data).__name__}"
                )
            for key in data:
                if not isinstance(key, str):
                    raise TypeError(
                        f"Attribute keys must be strings, got {type(key).__name__}"
                    )
            if not data:
                self._data = TensorDict(
                    {}, batch_size=torch.Size([0]), device=self.device
                )
            else:
                converted = {
                    k: to_tensor(
                        v,
                        device=self.device,
                        dtype=self._attr_map.dtypes.get(k),
                    )
                    for k, v in data.items()
                }
                first_dim = next(iter(converted.values())).shape[0]
                self._data = TensorDict(
                    converted,
                    batch_size=[first_dim],
                    device=self.device,
                )

    # -- Properties ---------------------------------------------------------

    @property
    def attr_map(self) -> LevelSchema:
        """The :class:`LevelSchema` associated with this container."""
        return self._attr_map

    @attr_map.setter
    def attr_map(self, attr_map: LevelSchema) -> None:
        self._attr_map = attr_map

    @property
    def num_systems(self) -> int:
        """Number of systems (same as ``len(self)``)."""
        return len(self)

    @property
    def num_atoms(self) -> int:
        """Total number of atoms across all systems.

        For segmented containers this is the first dimension of any atoms-group
        attribute.  For uniform containers it is ``dim0 * dim1``.

        Raises
        ------
        ValueError
            If no atoms-group attribute is present.
        """
        if self._data.is_empty():
            return 0
        atoms_attrs = self._attr_map.group_to_attrs.get("atoms", set())
        for attr_name, value in self._data.items():
            if attr_name in atoms_attrs:
                if self.is_segmented():
                    return value.shape[0]
                return value.shape[0] * value.shape[1]
        raise ValueError("Cannot determine num_atoms: no atoms-group attributes found")

    # -- Abstract -----------------------------------------------------------

    @abstractmethod
    def __len__(self) -> int: ...

    @abstractmethod
    def select(self, idx: IndexType) -> BaseLevelStorage:
        """Return a new container with only the selected samples."""

    @abstractmethod
    def update_at(self, key: str, value: Tensor, idx: IndexType) -> None:
        """In-place update of *key* at the given indices."""

    @abstractmethod
    def concatenate(
        self,
        other: BaseLevelStorage | dict[str, Tensor],
        strict: bool = False,
    ) -> BaseLevelStorage:
        """Concatenate *other* into this container (in-place)."""

    @abstractmethod
    def is_segmented(self) -> bool:
        """Return ``True`` if this container uses segmented storage."""

    @abstractmethod
    def _validate_setitem(self, key: str, value: Tensor) -> None: ...

    # -- Dict-like interface ------------------------------------------------

    def __getitem__(self, key: str | IndexType) -> Tensor | BaseLevelStorage:
        """Retrieve an attribute by name or select a subset by index."""
        if isinstance(key, str):
            if key not in self._data.keys():
                raise KeyError(f"Key '{key}' not found")
            return self._data[key]
        if isinstance(key, (slice, int, Tensor, list)):
            return self.select(key)
        raise TypeError(f"Invalid index type: {type(key).__name__}")

    def __setitem__(self, key: str, value: Any) -> None:
        """Set an attribute, converting *value* to a tensor."""
        dtype = self._attr_map.dtypes.get(key, None)
        tensor = to_tensor(value, device=self.device, dtype=dtype)
        if self.validate:
            self._validate_setitem(key, tensor)
        self._data[key] = tensor

    def __contains__(self, key: str) -> bool:
        return key in self._data.keys()

    def __iter__(self) -> Iterator[str]:
        return iter(self._data.keys())

    def __delitem__(self, key: str) -> None:
        if key not in self._data.keys():
            raise KeyError(f"Key '{key}' not found")
        del self._data[key]

    def keys(self) -> Iterator[str]:
        """Iterate over attribute names."""
        return iter(self._data.keys())

    def values(self) -> Iterator[Tensor]:
        """Iterate over attribute tensors."""
        return self._data.values()

    def items(self) -> Iterator[tuple[str, Tensor]]:
        """Iterate over ``(name, tensor)`` pairs."""
        return self._data.items()

    def get(self, key: str, default: Tensor | None = None) -> Tensor | None:
        """Return the tensor for *key*, or *default* if absent."""
        return self._data.get(key, default)

    def pop(self, key: str, default: Tensor | None = None) -> Tensor:
        """Remove and return the tensor for *key*.

        Parameters
        ----------
        key : str
            Attribute name.
        default : Tensor, optional
            Fallback value if *key* is missing.

        Raises
        ------
        KeyError
            If *key* is absent and no *default* was given.
        """
        if key not in self._data.keys():
            if default is not None:
                return default
            raise KeyError(f"Key '{key}' not found")
        value = self._data[key].clone()
        del self._data[key]
        return value

    # -- Copy / device ------------------------------------------------------

    def deepcopy(self) -> BaseLevelStorage:
        """Return a new container with cloned tensors (attr_map is shared)."""
        cloned = {k: _clone_tensor(v) for k, v in self._data.items()}
        return self.__class__(cloned, device=self.device, validate=False)

    def copy(self) -> BaseLevelStorage:
        """Return a shallow copy (tensors are **not** cloned)."""
        return self.__class__(self._data, device=self.device, validate=False)

    def clone(self) -> BaseLevelStorage:
        """Return a deep copy including an independent attr_map clone."""
        cloned = {k: _clone_tensor(v) for k, v in self._data.items()}
        return self.__class__(
            cloned,
            device=self.device,
            attr_map=self._attr_map.clone(),
            validate=False,
        )

    def to_device(self, device: DeviceType) -> BaseLevelStorage:
        """Move all tensors to *device* (in-place).

        Parameters
        ----------
        device : DeviceType
            Target device.

        Returns
        -------
        Self
            For method chaining.
        """
        device = torch.device(device)
        self.device = device
        self._data = self._data.to(device)
        return self


# ---------------------------------------------------------------------------
# UniformLevelStorage (uniform / dense)
# ---------------------------------------------------------------------------
class UniformLevelStorage(BaseLevelStorage):
    """Container for uniform data where every attribute shares the same first dimension.

    Parameters
    ----------
    data : dict[str, Tensor], optional
        Initial attribute tensors.
    device : DeviceType, optional
        Target device.
    attr_map : LevelSchema, optional
        Shared attribute registry.
    validate : bool
        If ``True``, ensure all attributes have the same ``shape[0]``.

    Raises
    ------
    ValueError
        If validation is enabled and first-dimension sizes differ.
    """

    def __init__(
        self,
        data: dict[str, Tensor] | None = None,
        device: DeviceType | None = None,
        attr_map: LevelSchema | None = None,
        validate: bool = True,
    ) -> None:
        # Validate first dims before building TensorDict (TensorDict requires consistent batch)
        if data is not None and isinstance(data, dict) and validate and len(data) > 1:
            first_dim = next(iter(data.values())).shape[0]
            for key, value in data.items():
                if value.shape[0] != first_dim:
                    raise ValueError(
                        f"Inconsistent first dimension: expected {first_dim}, "
                        f"got {value.shape[0]} for '{key}'"
                    )
        super().__init__(data, device, attr_map, validate)

        if data is not None and validate and not self._data.is_empty():
            first_key = next(iter(self._data.keys()))
            batch_size = self._data[first_key].shape[0]
            for key in self._data.keys():
                if self._data[key].shape[0] != batch_size:
                    raise ValueError(
                        f"Inconsistent first dimension between '{first_key}' and "
                        f"'{key}': {batch_size} vs {self._data[key].shape[0]}"
                    )

    def __len__(self) -> int:
        if self._data.is_empty():
            # batch_size is set to [N] when constructed from a sized TensorDict
            # (e.g. the empty system group created by from_data_list).
            bs = self._data.batch_size
            return int(bs[0]) if bs else 0
        n = getattr(self, "_num_kept", None)
        if n is not None:
            return n
        return self._data.shape[0]

    def num_elements(self) -> int:
        """Total number of elements (rows); after defrag, same as len(self)."""
        return len(self)

    def _validate_setitem(self, key: str, value: Tensor) -> None:
        if not self._data.is_empty() and len(value) != len(self):
            raise ValueError(f"Length mismatch: {len(value)} vs {len(self)}")

    def select(self, idx: IndexType) -> UniformLevelStorage:
        """Select a subset of samples by index.

        Parameters
        ----------
        idx : int, slice, or Tensor
            Index specification.

        Returns
        -------
        UniformLevelStorage
        """
        if isinstance(idx, int):
            idx = slice(idx, idx + 1)
        elif not isinstance(idx, slice):
            idx = self._prepare_index(idx)

        selected_td = self._data[idx]
        return self.__class__(selected_td, device=self.device, validate=False)

    def _prepare_index(self, idx: Any) -> Tensor:
        """Coerce *idx* to a tensor on ``self.device``, preserving integer dtype."""
        idx = to_tensor(idx)
        return idx.to(device=self.device)

    def update_at(self, key: str, value: Any, idx: IndexType) -> None:
        """Update attribute *key* at the given indices.

        Parameters
        ----------
        key : str
            Attribute name.
        value : Any
            Replacement values (converted to tensor).
        idx : IndexType
            Target indices.

        Raises
        ------
        KeyError
            If *key* is not present.
        ValueError
            If *idx* is not 1-D when given as a tensor.
        """
        if key not in self._data.keys():
            raise KeyError(f"Key '{key}' not found")
        value = to_tensor(value, self.device)
        if isinstance(idx, (slice, int)):
            self._data.set_at_(key, value, index=idx)
        else:
            idx = to_tensor(idx)
            if idx.ndim != 1:
                raise ValueError(f"Index must be 1-D, got {idx.ndim}-D")
            self._data.set_at_(key, value, index=idx.to(device=self.device))

    def concatenate(
        self,
        other: UniformLevelStorage | dict[str, Tensor],
        strict: bool = False,
    ) -> UniformLevelStorage:
        """Concatenate *other* along dimension 0 (in-place).

        Parameters
        ----------
        other : UniformLevelStorage or dict[str, Tensor]
            Data to append.
        strict : bool
            If ``True``, require identical attribute sets.

        Returns
        -------
        Self
            For method chaining.

        Raises
        ------
        ValueError
            On key mismatch (strict) or incompatible trailing shapes.
        """
        self_keys = set(self.keys())
        other_keys = set(other.keys())
        common = self_keys & other_keys
        if strict and self_keys != other_keys:
            raise ValueError(f"Keys mismatch: {self_keys} vs {other_keys}")
        if not common:
            # No keys to concatenate, but extend with zeros so that batch_size
            # stays aligned (matches the behavior of extend_for_appended_graphs).
            n_other = len(other)
            return self.extend_for_appended_graphs(n_other)
        # TensorDict enforces batch_size; replace with new TensorDict of concatenated data
        new_data = {}
        for key in common:
            other_tensor = to_tensor(other[key], self.device)
            if not _validate_trailing_shapes([self[key], other_tensor]):
                raise ValueError(
                    f"Key '{key}' has incompatible shape: "
                    f"{self[key].shape} vs {other_tensor.shape}"
                )
            new_data[key] = _concatenate_tensors(self._data[key], other_tensor)
        new_len = next(iter(new_data.values())).shape[0]
        self._data = TensorDict(
            new_data,
            batch_size=[new_len],
            device=self.device,
        )
        if hasattr(self, "_num_kept"):
            object.__delattr__(self, "_num_kept")
        return self

    def extend_for_appended_graphs(self, n: int) -> UniformLevelStorage:
        """Extend all tensors by *n* rows of zeros (for batches that lack this group).

        Used when appending a batch that does not have this group: the first
        dimension (num graphs) is extended so segment counts stay aligned.

        Parameters
        ----------
        n : int
            Number of graph slots to add (zero-padded).

        Returns
        -------
        Self
            For method chaining.
        """
        if n <= 0:
            return self
        if self._data.is_empty():
            return self
        new_data = {}
        for key, tensor in self._data.items():
            pad = torch.zeros(
                n, *tensor.shape[1:], dtype=tensor.dtype, device=tensor.device
            )
            new_data[key] = torch.cat([tensor, pad], dim=0)
        new_len = next(iter(new_data.values())).shape[0]
        self._data = TensorDict(
            new_data,
            batch_size=[new_len],
            device=self.device,
        )
        if hasattr(self, "_num_kept"):
            object.__delattr__(self, "_num_kept")
        return self

    def compute_put_per_system_fit_mask(
        self,
        source: UniformLevelStorage,
        source_mask: Tensor,
        dest_mask: Tensor | None,
        fit_mask: Tensor,
    ) -> None:
        """Compute which source rows would fit in this storage; write result into fit_mask in place.

        Parameters
        ----------
        source : UniformLevelStorage
            Source storage; length gives num_systems.
        source_mask : Tensor, shape (num_systems,), dtype bool
            True for each system considered for copy.
        dest_mask : Tensor, optional, shape (len(self),), dtype bool
            True = slot occupied. If None, all slots are treated as empty.
        fit_mask : Tensor, shape (num_systems,), dtype bool
            Output written in place: True where that source row would fit.

        Notes
        -----
        No data is copied. Use with put after combining (e.g. logical_and) with other
        levels' fit masks so only systems that fit in every level are copied.
        """
        n_src = len(source)
        if source_mask.shape[0] != n_src:
            raise ValueError(
                f"source_mask shape {source_mask.shape[0]} != len(source) {n_src}"
            )
        if fit_mask.shape[0] != n_src:
            raise ValueError(f"fit_mask shape {fit_mask.shape[0]} != {n_src}")
        dest_capacity = self._data.shape[0]
        source_mask = source_mask.to(device=self.device, dtype=torch.bool)
        fit_mask = fit_mask.to(device=self.device, dtype=torch.bool)
        if dest_mask is None:
            dest_mask = torch.zeros(dest_capacity, device=self.device, dtype=torch.bool)
        else:
            dest_mask = dest_mask.to(device=self.device, dtype=torch.bool)
            if dest_mask.shape[0] != dest_capacity:
                raise ValueError(
                    f"dest_mask shape {dest_mask.shape[0]} != dest capacity {dest_capacity}"
                )
        compute_put_fit_mask_per_system(source_mask, dest_mask, fit_mask)

    def put(
        self,
        src: UniformLevelStorage,
        mask: Tensor,
        *,
        copied_mask: Tensor | None = None,
        dest_mask: Tensor | None = None,
    ) -> None:
        """Put rows where mask[i] is True from src into this storage (buffer).

        Copies only float32 attributes; only as many rows as fit in this
        storage's empty slots (dest_mask[i] False = empty). Uses Warp buffer
        kernels (no host sync). If copied_mask is provided, it is updated in
        place with True for each row that was copied.

        Parameters
        ----------
        src : UniformLevelStorage
            Source storage; same attribute keys as self.
        mask : Tensor
            (num_systems,) bool, True = copy this row.
        copied_mask : Tensor, optional
            (num_systems,) bool; if provided, modified in place with which
            rows were actually copied. If None, stored as ``_copied_mask`` for
            use by :meth:`defrag`.
        dest_mask : Tensor, optional
            (capacity,) bool, True = slot occupied; capacity = ``len(self)``
            or ``self._data.shape[0]`` when ``_num_kept`` is set (pre-allocated
            buffer). If None, all slots are treated as empty.
        """
        if self._data.is_empty() or src._data.is_empty():
            raise ValueError("put requires non-empty source and dest")
        common = set(self._data.keys()) & set(src._data.keys())
        if not common:
            raise ValueError("put requires at least one common attribute")
        n_src = len(src)
        dest_capacity = self._data.shape[0]
        if mask.shape[0] != n_src:
            raise ValueError(f"mask shape {mask.shape[0]} != len(src) {n_src}")
        mask = mask.to(device=self.device, dtype=torch.bool)
        if copied_mask is not None:
            if copied_mask.shape[0] != n_src:
                raise ValueError(f"copied_mask shape {copied_mask.shape[0]} != {n_src}")
            out_mask = copied_mask.to(device=self.device, dtype=torch.bool)
        else:
            out_mask = torch.zeros(n_src, device=self.device, dtype=torch.bool)
            object.__setattr__(src, "_copied_mask", out_mask)
        if dest_mask is None:
            dest_mask = torch.zeros(dest_capacity, device=self.device, dtype=torch.bool)
        else:
            dest_mask = dest_mask.to(device=self.device, dtype=torch.bool)
            if dest_mask.shape[0] != dest_capacity:
                raise ValueError(
                    f"dest_mask shape {dest_mask.shape[0]} != dest capacity {dest_capacity}"
                )
        for key in common:
            src_t = src._data[key]
            dest_t = self._data[key]
            if dest_t.shape[0] < dest_capacity:
                raise ValueError(
                    f"dest attribute '{key}' first dim {dest_t.shape[0]} < {dest_capacity}"
                )
            put_masked_per_system(
                src_t,
                mask,
                dest_t,
                dest_mask,
                out_mask,
            )
        num_copied = out_mask.sum().item()
        if num_copied > 0 and getattr(self, "_num_kept", None) is not None:
            object.__setattr__(self, "_num_kept", self._num_kept + num_copied)

    def defrag(
        self,
        copied_mask: Tensor | None = None,
    ) -> UniformLevelStorage:
        """Defrag in-place: remove rows where copied_mask[i] is True.

        Rows with copied_mask[i] True (previously put) are dropped; remaining
        rows move to the front in place. Only float32 attributes are
        compacted. Buffer shape is unchanged (fixed-size batches). Other
        attributes are indexed with the same indices (so must be kept in sync
        if present).

        Parameters
        ----------
        copied_mask : Tensor, optional
            (num_systems,) bool; if None, uses ``_copied_mask`` from the last
            :meth:`put`.

        Returns
        -------
        Self
            For method chaining.
        """
        if self._data.is_empty():
            return self
        n_src = len(self)
        if copied_mask is None:
            copied_mask = getattr(self, "_copied_mask", None)
            if copied_mask is None:
                raise ValueError("defrag requires copied_mask or a prior put")
        else:
            copied_mask = copied_mask.to(device=self.device, dtype=torch.bool)
        if copied_mask.shape[0] != n_src:
            raise ValueError(f"copied_mask shape {copied_mask.shape[0]} != {n_src}")
        for key in list(self._data.keys()):
            t = self._data[key]
            if t.dtype not in TORCH_TO_WP:
                continue
            # Pass a clone so the kernel's in-place mask update doesn't affect other keys
            defrag_per_system(t, copied_mask.clone())
        object.__setattr__(self, "_num_kept", int((~copied_mask).sum().item()))
        if hasattr(self, "_copied_mask"):
            object.__delattr__(self, "_copied_mask")
        return self

    def is_segmented(self) -> bool:
        return False

    def to_device(self, device: DeviceType) -> UniformLevelStorage:
        """Move all tensors to *device* (in-place)."""
        super().to_device(device)
        return self

    def clone(self) -> UniformLevelStorage:
        """Return a deep copy; copies _num_kept if set (e.g. after defrag)."""
        out = super().clone()
        if getattr(self, "_num_kept", None) is not None:
            object.__setattr__(out, "_num_kept", self._num_kept)
        return out

    def __repr__(self) -> str:
        attrs = ", ".join(f"'{k}': {self._data[k].shape}" for k in self._data.keys())
        return f"UniformLevelStorage({{{attrs}}}, device={self.device})"


# ---------------------------------------------------------------------------
# SegmentedLevelStorage (variable-length segments)
# ---------------------------------------------------------------------------
class SegmentedLevelStorage(BaseLevelStorage):
    """Container for segmented data with variable-length segments.

    Each "sample" corresponds to a segment of contiguous elements.  The
    ``segment_lengths`` tensor records how many elements belong to each
    segment, and ``batch_idx`` / ``batch_ptr`` are lazily computed from it.

    Parameters
    ----------
    data : dict[str, Tensor], optional
        Initial attribute tensors (all must have the same ``shape[0]``).
    device : DeviceType, optional
        Target device.
    attr_map : LevelSchema, optional
        Shared attribute registry.
    segment_lengths : list[int] or Tensor, optional
        Number of elements per segment.  If ``None`` and *data* is provided,
        a single segment spanning the full data is assumed.
    batch_idx : Tensor, optional
        Pre-computed segment assignment per element.
    batch_ptr : Tensor, optional
        Pre-computed cumulative element counts (length = num_segments + 1).
    batch_ptr_capacity : int, optional
        If set, pre-allocate batch_ptr with this length so put() can append
        segments without reallocating. Ignored when batch_ptr is provided.
    validate : bool
        If ``True``, verify consistency between data and segment lengths.

    Raises
    ------
    ValueError
        If segment lengths are negative or don't sum to ``data.shape[0]``.
    """

    def __init__(
        self,
        data: dict[str, Tensor] | None = None,
        device: DeviceType | None = None,
        attr_map: LevelSchema | None = None,
        segment_lengths: list[int] | Tensor | None = None,
        batch_idx: Tensor | None = None,
        batch_ptr: Tensor | None = None,
        batch_ptr_capacity: int | None = None,
        validate: bool = True,
    ) -> None:
        super().__init__(data, device, attr_map, validate)

        if data is not None and validate and not self._data.is_empty():
            sizes = {self._data[k].shape[0] for k in self._data.keys()}
            if len(sizes) > 1:
                raise ValueError(
                    f"All attributes must have the same first dimension, got {sizes}"
                )

        if segment_lengths is None:
            if not self._data.is_empty():
                num_el = self._data.shape[0]
                self.segment_lengths = torch.tensor(
                    [num_el], device=self.device, dtype=torch.int32
                )
            else:
                self.segment_lengths = torch.tensor(
                    [], device=self.device, dtype=torch.int32
                )
        else:
            self.segment_lengths = to_tensor(
                segment_lengths, self.device, dtype="int32"
            )

        self._batch_idx: Tensor | None = (
            batch_idx.to(device=self.device, dtype=torch.int32)
            if batch_idx is not None
            else None
        )
        if batch_ptr is not None:
            self._batch_ptr = batch_ptr.to(device=self.device, dtype=torch.int32)
        elif batch_ptr_capacity is not None and not self._data.is_empty():
            n_seg = len(self.segment_lengths)
            cap = max(batch_ptr_capacity, n_seg + 1)
            self._batch_ptr = torch.empty(cap, device=self.device, dtype=torch.int32)
            self._batch_ptr[0] = 0
            cum = torch.cumsum(self.segment_lengths, dim=0)
            self._batch_ptr[1 : n_seg + 1] = cum
            if cap > n_seg + 1:
                self._batch_ptr[n_seg + 1 :].fill_(cum[-1].item() if n_seg else 0)
        else:
            self._batch_ptr = None
        self._batch_ptr_np: np.ndarray | None = None
        self._segment_indices: Tensor | None = None

        if not self._data.is_empty() and validate:
            self._validate_initialization()

    # -- Validation ---------------------------------------------------------

    def _validate_initialization(self) -> None:
        """Check segment lengths, batch_idx, and batch_ptr for consistency."""
        total_elements = self.num_elements()
        if (self.segment_lengths < 0).any():
            raise ValueError(
                f"Segment lengths cannot be negative: {self.segment_lengths}"
            )

        segment_sum = self.segment_lengths.sum().item()
        if segment_sum != total_elements:
            raise ValueError(
                f"Sum of segment_lengths ({segment_sum}) != data length ({total_elements})"
            )

        if self._batch_idx is not None:
            n_seg = len(self.segment_lengths)
            if self._batch_idx.shape[0] != total_elements:
                raise ValueError(
                    f"batch_idx length {self._batch_idx.shape[0]} != data length {total_elements}"
                )
            if total_elements > 0:
                if self._batch_idx[0] != 0:
                    raise ValueError(
                        f"batch_idx must start at 0, got {self._batch_idx[0].item()}"
                    )
                if self._batch_idx[-1] != n_seg - 1:
                    raise ValueError(
                        f"batch_idx last element {self._batch_idx[-1].item()} "
                        f"!= num_segments - 1 ({n_seg - 1})"
                    )

        if self._batch_ptr is not None:
            expected_len = len(self.segment_lengths) + 1
            if len(self._batch_ptr) < expected_len:
                raise ValueError(
                    f"batch_ptr length {len(self._batch_ptr)} < num_segments + 1 ({expected_len})"
                )
            if self._batch_ptr[0] != 0:
                raise ValueError(
                    f"batch_ptr must start at 0, got {self._batch_ptr[0].item()}"
                )
            logical_end = self._batch_ptr[expected_len - 1].item()
            if logical_end != total_elements:
                raise ValueError(
                    f"batch_ptr logical end (index {expected_len - 1}) {logical_end} "
                    f"!= data length ({total_elements})"
                )

    # -- Special methods ----------------------------------------------------

    def __len__(self) -> int:
        n = getattr(self, "_num_segments", None)
        if n is not None:
            return n
        return len(self.segment_lengths)

    def __getattr__(self, name: str) -> Any:
        """Lazy-initialise ``batch_idx`` and ``batch_ptr`` on first access."""
        match name:
            case "batch_idx":
                if not hasattr(self, "_batch_idx") or self._batch_idx is None:
                    self._lazy_init_batch_idx()
                return self._batch_idx
            case "batch_ptr":
                if not hasattr(self, "_batch_ptr") or self._batch_ptr is None:
                    self._lazy_init_batch_ptr()
                return self._batch_ptr
            case _:
                raise AttributeError(
                    f"'{type(self).__name__}' has no attribute '{name}'"
                )

    def __repr__(self) -> str:
        attrs = ", ".join(f"'{k}': {self._data[k].shape}" for k in self._data.keys())
        return (
            f"SegmentedLevelStorage({{{attrs}}}, "
            f"segments={len(self.segment_lengths)}, device={self.device})"
        )

    # -- Queries ------------------------------------------------------------

    def num_elements(self) -> int:
        """Total number of elements (active count; after defrag, same as batch_ptr[-1])."""
        if self._data.is_empty():
            return 0
        if len(self.segment_lengths) == 0:
            return 0
        n = getattr(self, "_num_elements_kept", None)
        if n is not None:
            return n
        return self._data.shape[0]

    def is_segmented(self) -> bool:
        return True

    # -- Lazy initialisers --------------------------------------------------

    def _lazy_init_batch_idx(self) -> None:
        if self._batch_idx is None:
            self._batch_idx = torch.repeat_interleave(
                torch.arange(
                    len(self.segment_lengths), device=self.device, dtype=torch.int32
                ),
                self.segment_lengths,
            )

    def _lazy_init_batch_ptr(self) -> None:
        if self._batch_ptr is None:
            self._batch_ptr = torch.cat(
                [
                    torch.zeros(1, device=self.device, dtype=torch.int32),
                    torch.cumsum(self.segment_lengths, dim=0),
                ]
            )

    def _lazy_init_segment_indices(self) -> None:
        if self._segment_indices is None:
            self._segment_indices = torch.arange(
                len(self.segment_lengths), device=self.device, dtype=torch.int32
            )

    # -- Validation ---------------------------------------------------------

    def _validate_setitem(self, key: str, value: Tensor) -> None:
        if not self._data.is_empty() and len(value) != self.num_elements():
            raise ValueError(f"Length mismatch: {len(value)} vs {self.num_elements()}")

    # -- Indexing -----------------------------------------------------------

    def _normalize_segment_index(self, idx: IndexType) -> torch.Tensor:
        """Normalize *idx* to a 1-D tensor of segment indices."""
        if isinstance(idx, int):
            i = idx + len(self) if idx < 0 else idx
            return torch.tensor([i], device=self.device, dtype=INDEX_DTYPE)
        if isinstance(idx, slice):
            start, stop, step = idx.indices(len(self))
            return torch.arange(
                start, stop, step, device=self.device, dtype=INDEX_DTYPE
            )
        if isinstance(idx, list):
            t = torch.as_tensor(idx, device=self.device)
            if t.dtype != INDEX_DTYPE and t.dtype != torch.bool:
                t = t.to(INDEX_DTYPE)
        elif isinstance(idx, torch.Tensor):
            t = idx.to(device=self.device)
        else:
            raise TypeError(f"Unsupported segment index type '{type(idx).__name__}'")
        if t.dtype == torch.bool:
            if t.numel() != len(self):
                raise ValueError(
                    f"Boolean index length {t.numel()} != num segments {len(self)}"
                )
            return torch.nonzero(t, as_tuple=False).flatten().to(torch.int64)
        return t

    def _expand_idx(self, idx: IndexType) -> Tensor:
        """Convert segment-level *idx* into element-level indices.

        Parameters
        ----------
        idx : int, slice, or Tensor
            Segment-level index.

        Returns
        -------
        Tensor
            Element-level indices on ``self.device``.
        """

        match idx:
            case int():
                self._lazy_init_batch_ptr()
                return torch.arange(
                    self._batch_ptr[idx],
                    self._batch_ptr[idx + 1],
                    device=self.device,
                )
            case slice():
                start, stop, step = idx.indices(len(self.segment_lengths))
                if step == 1:
                    if start == 0 and stop == len(self.segment_lengths):
                        return torch.arange(
                            0,
                            self.num_elements(),
                            device=self.device,
                            dtype=torch.int64,
                        )
                    else:
                        self._lazy_init_batch_ptr()
                        el_start = int(self._batch_ptr[start].item())
                        el_end = int(self._batch_ptr[stop].item())
                        return torch.arange(
                            el_start, el_end, device=self.device, dtype=torch.int64
                        )

        seg_idx = self._normalize_segment_index(idx)
        if self.device.type == "cuda":
            self._lazy_init_batch_ptr()
            return _expand_segments_warp(
                seg_idx, self._batch_ptr, self.device, torch.int64
            )

        else:
            starts = self.batch_ptr[seg_idx]
            ends = self.batch_ptr[seg_idx + 1]
            lengths = ends - starts
            total = int(lengths.sum().item())
            if total == 0:
                return torch.empty(0, device=self.device, dtype=torch.int64)

            repeated_starts = torch.repeat_interleave(
                starts, lengths, output_size=total
            )
            cum_lengths = torch.cumsum(lengths, 0, dtype=lengths.dtype)
            prefix = torch.repeat_interleave(
                cum_lengths - lengths,
                lengths,
                output_size=total,
            )
            local = torch.arange(total, device=self.device, dtype=torch.int64) - prefix
            return repeated_starts + local

    def _select_segment_lengths(self, idx: IndexType) -> Tensor:
        """Return segment_lengths for the selected segments."""
        if isinstance(idx, int):
            return self.segment_lengths[idx : idx + 1]
        if isinstance(idx, slice):
            return self.segment_lengths[idx]
        return self.segment_lengths[to_tensor(idx).to(device=self.device)]

    def select(self, idx: IndexType) -> SegmentedLevelStorage:
        """Select a subset of segments.

        Parameters
        ----------
        idx : int, slice, or Tensor
            Segment-level index.

        Returns
        -------
        SegmentedLevelStorage
        """
        element_idx = self._expand_idx(idx)
        segment_lengths = self._select_segment_lengths(idx)
        selected_td = self._data[element_idx.to(device=self.device)]
        return self.__class__(
            selected_td,
            device=self.device,
            segment_lengths=segment_lengths,
            validate=False,
        )

    def update_at(self, key: str, value: Any, idx: IndexType) -> None:
        """Update attribute *key* at the given segment indices.

        Parameters
        ----------
        key : str
            Attribute name.
        value : Any
            Replacement values (converted to tensor).
        idx : IndexType
            Segment-level indices.

        Raises
        ------
        KeyError
            If *key* is not present.
        """
        if key not in self._data.keys():
            raise KeyError(f"Key '{key}' not found")
        element_idx = self._expand_idx(idx)
        self._data[key][element_idx.to(device=self.device)] = to_tensor(
            value, self.device
        )

    # -- Device / copy ------------------------------------------------------

    def to_device(self, device: DeviceType) -> SegmentedLevelStorage:
        """Move all tensors (including bookkeeping) to *device*.

        Returns
        -------
        Self
            For method chaining.
        """
        super().to_device(device)
        self.segment_lengths = self.segment_lengths.to(device)
        if self._batch_idx is not None:
            self._batch_idx = self._batch_idx.to(device)
        if self._batch_ptr is not None:
            self._batch_ptr = self._batch_ptr.to(device)
        if self._segment_indices is not None:
            self._segment_indices = self._segment_indices.to(device)
        self._batch_ptr_np = None
        return self

    def clone(self) -> SegmentedLevelStorage:
        """Return a deep copy with independent tensors, segment info, and attr_map."""
        cloned_data = {k: _clone_tensor(self._data[k]) for k in self._data.keys()}
        cloned_seg = _clone_tensor(self.segment_lengths)
        cloned_bidx = (
            _clone_tensor(self._batch_idx) if self._batch_idx is not None else None
        )
        cloned_bptr = (
            _clone_tensor(self._batch_ptr) if self._batch_ptr is not None else None
        )
        out = self.__class__(
            data=cloned_data,
            device=self.device,
            attr_map=self._attr_map.clone(),
            segment_lengths=cloned_seg,
            batch_idx=cloned_bidx,
            batch_ptr=cloned_bptr,
            validate=False,
        )
        if getattr(self, "_num_segments", None) is not None:
            object.__setattr__(out, "_num_segments", self._num_segments)
            object.__setattr__(out, "_num_elements_kept", self._num_elements_kept)
        return out

    # -- Concatenation ------------------------------------------------------

    def concatenate(
        self,
        other: SegmentedLevelStorage,
        strict: bool = False,
    ) -> SegmentedLevelStorage:
        """Concatenate *other* into this container (in-place).

        Parameters
        ----------
        other : SegmentedLevelStorage
            Container to append.
        strict : bool
            If ``True``, require identical attribute sets.

        Returns
        -------
        Self
            For method chaining.

        Raises
        ------
        ValueError
            On key mismatch (strict) or incompatible trailing shapes.
        """
        self_keys = set(self.keys())
        other_keys = set(other.keys())
        common = self_keys & other_keys
        if strict and self_keys != other_keys:
            raise ValueError(f"Keys mismatch: {self_keys} vs {other_keys}")
        if not common:
            return self

        prev_elements = self.num_elements()
        prev_segments = len(self)

        # TensorDict enforces batch_size; replace with new TensorDict of concatenated data
        new_data = {}
        for key in common:
            other_tensor = to_tensor(other[key], self.device)
            if not _validate_trailing_shapes([self[key], other_tensor]):
                raise ValueError(
                    f"Key '{key}' has incompatible shape: "
                    f"{self[key].shape} vs {other_tensor.shape}"
                )
            new_data[key] = _concatenate_tensors(self._data[key], other_tensor)
        new_total = next(iter(new_data.values())).shape[0]
        self._data = TensorDict(
            new_data,
            batch_size=[new_total],
            device=self.device,
        )
        self.segment_lengths = torch.cat([self.segment_lengths, other.segment_lengths])

        if self._batch_idx is not None:
            other._lazy_init_batch_idx()
            self._batch_idx = torch.cat(
                [self._batch_idx, other._batch_idx.to(self.device) + prev_segments]
            )
        if self._batch_ptr is not None:
            other._lazy_init_batch_ptr()
            self._batch_ptr = torch.cat(
                [self._batch_ptr, other._batch_ptr[1:].to(self.device) + prev_elements]
            )
        self._batch_ptr_np = None
        if hasattr(self, "_num_segments"):
            object.__delattr__(self, "_num_segments")
            object.__delattr__(self, "_num_elements_kept")
        return self

    def extend_for_appended_graphs(self, n: int) -> SegmentedLevelStorage:
        """Extend segment count by *n* zero-length segments (for batches that lack this group).

        Used when appending a batch that does not have this group: segment_lengths
        is extended with *n* zeros so graph counts stay aligned. No new elements
        are added to the data tensors.

        Parameters
        ----------
        n : int
            Number of graph slots (segments of length 0) to add.

        Returns
        -------
        Self
            For method chaining.
        """
        if n <= 0:
            return self
        extra = torch.zeros(n, device=self.device, dtype=self.segment_lengths.dtype)
        self.segment_lengths = torch.cat([self.segment_lengths, extra])
        self._batch_idx = None
        self._batch_ptr = None
        self._batch_ptr_np = None
        self._segment_indices = None
        if hasattr(self, "_num_segments"):
            object.__delattr__(self, "_num_segments")
            object.__delattr__(self, "_num_elements_kept")
        return self

    def compute_put_per_system_fit_mask(
        self,
        source: SegmentedLevelStorage,
        source_mask: Tensor,
        dest_mask: Tensor | None,
        fit_mask: Tensor,
    ) -> None:
        """Compute which source segments would fit in this storage; write result into fit_mask in place.

        Parameters
        ----------
        source : SegmentedLevelStorage
            Source storage; len(source) gives num_systems (segments).
        source_mask : Tensor, shape (num_systems,), dtype bool
            True for each segment considered for copy.
        dest_mask : Tensor, optional
            Ignored for segmented level (fit uses batch_ptr and data capacity only).
        fit_mask : Tensor, shape (num_systems,), dtype bool
            Output written in place: True where that segment fits in this storage.

        Notes
        -----
        No data is copied. If this storage's batch_ptr has insufficient capacity for
        new segment boundaries, fit_mask is zeroed. Use with put after combining
        (e.g. logical_and) with other levels' fit masks.
        """
        n_seg = len(source)
        if source_mask.shape[0] != n_seg:
            raise ValueError(
                f"source_mask shape {source_mask.shape[0]} != len(source) {n_seg}"
            )
        if fit_mask.shape[0] != n_seg:
            raise ValueError(f"fit_mask shape {fit_mask.shape[0]} != {n_seg}")
        source_mask = source_mask.to(device=self.device, dtype=torch.bool)
        fit_mask = fit_mask.to(device=self.device, dtype=torch.bool)
        source._lazy_init_batch_ptr()
        self._lazy_init_batch_ptr()
        num_dest_segments = len(self)
        min_batch_ptr_size = num_dest_segments + n_seg + 2
        if self._batch_ptr.shape[0] < min_batch_ptr_size:
            fit_mask.zero_()
            return
        dest_capacity = self._data.shape[0]
        compute_put_fit_mask_segmented(
            source._batch_ptr,
            source_mask,
            self._batch_ptr,
            num_dest_segments,
            dest_capacity,
            fit_mask,
        )

    def put(
        self,
        src: SegmentedLevelStorage,
        mask: Tensor,
        *,
        copied_mask: Tensor | None = None,
    ) -> None:
        """Put segments where mask[i] is True from src into this storage (buffer).

        New segment boundaries are appended to this storage's batch_ptr. Only
        float32 attributes are copied. Uses Warp buffer kernels; one host sync
        after all attributes to update this storage's segment count. If
        copied_mask is provided, it is updated in place with True for each
        segment that was copied.

        Parameters
        ----------
        src : SegmentedLevelStorage
            Source storage; same attribute keys as self.
        mask : Tensor
            (num_segments,) bool, True = copy this segment.
        copied_mask : Tensor, optional
            (num_segments,) bool; if provided, modified in place with which
            segments were actually copied. If None, stored on *src* as
            ``_copied_mask`` for use by :meth:`defrag`.
        """
        if self._data.is_empty() or src._data.is_empty():
            raise ValueError("put requires non-empty source and dest")
        common = set(self._data.keys()) & set(src._data.keys())
        if not common:
            raise ValueError("put requires at least one common attribute")
        n_seg = len(src)
        if mask.shape[0] != n_seg:
            raise ValueError(f"mask shape {mask.shape[0]} != num segments {n_seg}")
        mask = mask.to(device=self.device, dtype=torch.bool)
        if copied_mask is not None:
            if copied_mask.shape[0] != n_seg:
                raise ValueError(f"copied_mask shape {copied_mask.shape[0]} != {n_seg}")
            out_mask = copied_mask.to(device=self.device, dtype=torch.bool)
        else:
            out_mask = torch.zeros(n_seg, device=self.device, dtype=torch.bool)
            object.__setattr__(src, "_copied_mask", out_mask)
        src._lazy_init_batch_ptr()
        self._lazy_init_batch_ptr()
        num_dest_segments = len(self)
        min_batch_ptr_size = num_dest_segments + n_seg + 2
        dest_batch_ptr = self._batch_ptr
        if dest_batch_ptr.shape[0] < min_batch_ptr_size:
            return
        new_num_dest = None
        for key in common:
            src_t = src._data[key]
            if src_t.dtype != torch.float32:
                continue
            dest_t = self._data[key]
            if dest_t.dtype != torch.float32:
                continue
            new_num_dest = put_masked_segmented(
                src_t,
                src._batch_ptr,
                mask,
                dest_t,
                dest_batch_ptr,
                num_dest_segments,
                out_mask,
            )
        if new_num_dest is not None:
            new_n = int(new_num_dest.item())
            object.__setattr__(self, "_batch_ptr", dest_batch_ptr[: new_n + 1].clone())
            self.segment_lengths = self._batch_ptr[1:] - self._batch_ptr[:-1]
            object.__setattr__(self, "_batch_ptr_np", None)
            if hasattr(self, "_num_segments"):
                object.__delattr__(self, "_num_segments")
                object.__delattr__(self, "_num_elements_kept")

    def defrag(
        self,
        copied_mask: Tensor | None = None,
    ) -> SegmentedLevelStorage:
        """Defrag in-place: remove segments where copied_mask[i] is True.

        Kept segments move to the front; batch_ptr is updated in place (tail
        filled so batch_ptr[-1] == total_kept_elems); segment_lengths derived
        from it; no trim. All attributes must be float32 (uses Warp kernels).

        Parameters
        ----------
        copied_mask : Tensor, optional
            (num_segments,) bool; if None, uses ``_copied_mask`` from the
            last :meth:`put`.

        Returns
        -------
        Self
            For method chaining.
        """
        if self._data.is_empty():
            return self
        n_seg = len(self)
        if copied_mask is None:
            copied_mask = getattr(self, "_copied_mask", None)
            if copied_mask is None:
                raise ValueError("defrag requires copied_mask or a prior put")
        else:
            copied_mask = copied_mask.to(device=self.device, dtype=torch.bool)
        if copied_mask.shape[0] != n_seg:
            raise ValueError(f"copied_mask shape {copied_mask.shape[0]} != {n_seg}")
        self._lazy_init_batch_ptr()
        keys = list(self._data.keys())
        original_bp = self._batch_ptr.clone()
        num_kept_t = defrag_segmented(self._data[keys[0]], self._batch_ptr, copied_mask)
        for key in keys[1:]:
            defrag_segmented(self._data[key], original_bp.clone(), copied_mask)

        self.segment_lengths = self._batch_ptr[1:] - self._batch_ptr[:-1]
        n_kept = int(num_kept_t.item())
        object.__setattr__(self, "_num_segments", n_kept)
        object.__setattr__(
            self, "_num_elements_kept", int(self._batch_ptr[n_kept].item())
        )
        self._batch_idx = None
        self._batch_ptr_np = None
        self._segment_indices = None
        # Kernel already compacted each tensor in place (kept rows at front, rest zeroed);
        # buffer shape is unchanged for fixed-size batches.
        if hasattr(self, "_copied_mask"):
            object.__delattr__(self, "_copied_mask")
        return self


# ---------------------------------------------------------------------------
# MultiLevelStorage
# ---------------------------------------------------------------------------
class MultiLevelStorage:
    """Multi-group container that routes attribute access to the correct group.

    Manages a dict of :class:`UniformLevelStorage` and :class:`SegmentedLevelStorage`
    instances, exposing a flat attribute namespace across all groups.

    Parameters
    ----------
    groups : dict[str, UniformLevelStorage | SegmentedLevelStorage], optional
        Named groups.  ``None`` creates an empty batch.
    validate : bool
        If ``True``, check length/device consistency and duplicate attributes.
    attr_map : LevelSchema, optional
        Shared attribute registry.
    device : DeviceType, optional
        Target device.  If ``None``, inferred from existing groups.

    Raises
    ------
    ValueError
        On inconsistent group lengths or duplicate attribute names.
    """

    def __init__(
        self,
        groups: dict[str, UniformLevelStorage | SegmentedLevelStorage] | None = None,
        validate: bool = True,
        attr_map: LevelSchema | None = None,
        device: DeviceType | None = None,
    ) -> None:
        self.groups: dict[str, UniformLevelStorage | SegmentedLevelStorage] = (
            groups if groups is not None else {}
        )
        self.attr_map = attr_map if attr_map is not None else LevelSchema()
        self.device = torch.device(device) if device else torch.device("cpu")

        if validate and self.groups:
            self._validate_consistency()

        if device is not None:
            self.to_device(device)
        elif self.groups:
            self._infer_and_set_device()

    # -- Validation ---------------------------------------------------------

    def _infer_and_set_device(self) -> None:
        first_device = next(iter(self.groups.values())).device
        self.to_device(first_device)

    def _validate_consistency(self) -> None:
        self._validate_length_consistency()
        self._validate_no_duplicate_attributes()
        self._validate_consistent_devices()

    def _validate_length_consistency(self) -> None:
        if not self.groups:
            return
        expected = len(next(iter(self.groups.values())))
        for name, group in self.groups.items():
            if len(group) != expected:
                raise ValueError(
                    f"Group '{name}' has length {len(group)}, expected {expected}"
                )

    def _validate_no_duplicate_attributes(self) -> None:
        seen: set[str] = set()
        for group_name, group in self.groups.items():
            for key in group.keys():
                if key in seen:
                    raise ValueError(
                        f"Attribute '{key}' is duplicated in group '{group_name}'"
                    )
                seen.add(key)

    def _validate_consistent_devices(self) -> None:
        if not self.groups:
            return
        first_device = next(iter(self.groups.values())).device
        for group in self.groups.values():
            group.to_device(first_device)

    # -- Length / properties ------------------------------------------------

    def __len__(self) -> int:
        if not self.groups:
            return 0
        return len(next(iter(self.groups.values())))

    @property
    def num_systems(self) -> int:
        """Number of systems from the ``'system'`` group.

        Raises
        ------
        ValueError
            If no ``'system'`` group exists.
        """
        if "system" not in self.groups:
            raise ValueError("'system' group not found in batch")
        return len(self.groups["system"])

    @property
    def num_atoms(self) -> int:
        """Total atom count from the ``'atoms'`` group."""
        group = self.groups.get("atoms")
        if group is None:
            return 0
        if isinstance(group, SegmentedLevelStorage):
            return group.num_elements()
        first_val = next(iter(group._data.values()))
        return first_val.shape[0] * first_val.shape[1]

    # -- Group lookup -------------------------------------------------------

    def _group_name_from_attr(self, attr_name: str) -> str | None:
        for group_name, group in self.groups.items():
            if attr_name in group:
                return group_name
        return None

    def group_from_attr(
        self, attr_name: str
    ) -> UniformLevelStorage | SegmentedLevelStorage | None:
        """Return the group that contains *attr_name*, or ``None``."""
        name = self._group_name_from_attr(attr_name)
        return self.groups[name] if name is not None else None

    # -- Dict-like interface ------------------------------------------------

    def __getitem__(self, key: str | IndexType) -> Tensor | MultiLevelStorage:
        """Access an attribute by name or select a subset by index."""
        if isinstance(key, str):
            group_name = self._group_name_from_attr(key)
            if group_name is None:
                raise KeyError(f"Attribute '{key}' not found in batch")
            return self.groups[group_name][key]
        return self.select(key)

    def __setitem__(self, key: str, value: Any) -> None:
        """Set an attribute, routing to the correct group via ``attr_map``."""
        try:
            group_name = self.attr_map.group(key)
        except KeyError:
            group_name = "system"  # Unknown keys default to system-level
        dtype = self.attr_map.dtypes.get(key, None)
        tensor = to_tensor(value, device=self.device, dtype=dtype)

        if group_name in self.groups:
            self.groups[group_name][key] = tensor
        else:
            if self.attr_map.is_segmented_group(group_name):
                raise ValueError(
                    f"Group '{group_name}' is segmented but not found in batch"
                )
            self.groups[group_name] = UniformLevelStorage(
                data={key: tensor},
                device=self.device,
                attr_map=self.attr_map,
                validate=False,
            )

    def __delitem__(self, key: str) -> None:
        group_name = self._group_name_from_attr(key)
        if group_name is None:
            raise KeyError(f"Attribute '{key}' not found in batch")
        del self.groups[group_name][key]

    def __contains__(self, key: str) -> bool:
        return any(key in g for g in self.groups.values())

    def __iter__(self) -> Iterator[str]:
        for group in self.groups.values():
            yield from group

    def __repr__(self) -> str:
        return f"MultiLevelStorage(groups={self.groups}, attr_map={self.attr_map})"

    def __str__(self) -> str:
        return repr(self)

    def keys(self) -> list[str]:
        """All attribute names across groups."""
        return [k for g in self.groups.values() for k in g.keys()]

    def values(self) -> list[Tensor]:
        """All attribute tensors across groups."""
        return [v for g in self.groups.values() for v in g.values()]

    def items(self) -> Iterator[tuple[str, Tensor]]:
        """Iterate over ``(attr_name, tensor)`` across all groups."""
        for group in self.groups.values():
            yield from group.items()

    def get(self, key: str, default: Tensor | None = None) -> Tensor | None:
        """Return the tensor for *key*, or *default* if absent."""
        group_name = self._group_name_from_attr(key)
        if group_name is None:
            return default
        return self.groups[group_name].get(key, default)

    def pop(self, key: str, default: Any = None) -> Tensor:
        """Remove and return the tensor for *key*.

        Raises
        ------
        KeyError
            If *key* is absent and no *default* was given.
        """
        group_name = self._group_name_from_attr(key)
        if group_name is None:
            if default is not None:
                return default
            raise KeyError(f"Attribute '{key}' not found in batch")
        return self.groups[group_name].pop(key, default)

    # -- Selection / cloning ------------------------------------------------

    def select(self, idx: IndexType) -> MultiLevelStorage:
        """Select a subset of samples across all groups.

        Parameters
        ----------
        idx : IndexType
            Sample-level index specification.

        Returns
        -------
        MultiLevelStorage
        """
        if isinstance(idx, int):
            idx = slice(idx, idx + 1)
        return self.__class__(
            groups={k: v.select(idx) for k, v in self.groups.items()},
            attr_map=self.attr_map,
            validate=False,
        )

    def is_segmented(self) -> bool:
        """Return ``True`` if any group is segmented."""
        return any(g.is_segmented() for g in self.groups.values())

    def to_device(self, device: DeviceType) -> MultiLevelStorage:
        """Move all groups to *device*.

        Returns
        -------
        Self
            For method chaining.
        """
        device = torch.device(device)
        self.device = device
        for group in self.groups.values():
            group.to_device(device)
        return self

    def clone(self) -> MultiLevelStorage:
        """Return a deep copy with independent groups and attr_map."""
        return self.__class__(
            groups={n: g.clone() for n, g in self.groups.items()},
            validate=False,
            attr_map=self.attr_map.clone(),
            device=self.device,
        )

    def update_at(self, key: str, value: Any, idx: IndexType) -> None:
        """Update attribute *key* at the given indices.

        Parameters
        ----------
        key : str
            Attribute name.
        value : Any
            Replacement values.
        idx : IndexType
            Target indices.

        Raises
        ------
        KeyError
            If *key* is not found in any group.
        """
        group_name = self._group_name_from_attr(key)
        if group_name is None:
            raise KeyError(f"Attribute '{key}' not found in batch")
        dtype = self.attr_map.dtypes.get(key, None)
        tensor = to_tensor(value, device=self.device, dtype=dtype)
        self.groups[group_name].update_at(key, tensor, idx)

    def concatenate(
        self, other: MultiLevelStorage, strict: bool = False
    ) -> MultiLevelStorage:
        """Concatenate with another batch.

        Parameters
        ----------
        other : MultiLevelStorage
            Data to append.
        strict : bool
            If ``True``, require identical attribute sets.

        Returns
        -------
        MultiLevelStorage
        """
        return self.from_batches([self, other], strict=strict)

    # -- Factory methods ----------------------------------------------------

    @classmethod
    def from_data(
        cls,
        data: dict[str, Tensor],
        attr_map: LevelSchema | None = None,
        segment_lengths: dict[str, list[int] | Tensor] | None = None,
        device: DeviceType | None = "cpu",
        validate: bool = True,
    ) -> MultiLevelStorage:
        """Create an :class:`MultiLevelStorage` from a flat attribute dict.

        Parameters
        ----------
        data : dict[str, Tensor]
            Attribute tensors keyed by name.
        attr_map : LevelSchema, optional
            Registry used to route attributes to groups.
        segment_lengths : dict[str, list[int] | Tensor], optional
            Per-group segment lengths for segmented groups.
        device : DeviceType, optional
            Target device.
        validate : bool
            Whether to validate consistency.

        Returns
        -------
        MultiLevelStorage

        Raises
        ------
        ValueError
            If a segmented group has no corresponding entry in *segment_lengths*.
        """
        if attr_map is None:
            attr_map = LevelSchema()

        grouped: dict[str, dict[str, Tensor]] = defaultdict(dict)
        for key, value in data.items():
            grouped[attr_map.group(key)][key] = value

        groups: dict[str, UniformLevelStorage | SegmentedLevelStorage] = {}
        for group_name, group_dict in grouped.items():
            if segment_lengths is not None and attr_map.is_segmented_group(group_name):
                if group_name not in segment_lengths:
                    raise ValueError(
                        f"Segment lengths not provided for segmented group '{group_name}'"
                    )
                groups[group_name] = SegmentedLevelStorage(
                    data=group_dict,
                    device=device,
                    segment_lengths=segment_lengths[group_name],
                    validate=validate,
                    attr_map=attr_map,
                )
            else:
                groups[group_name] = UniformLevelStorage(
                    data=group_dict, device=device, validate=validate, attr_map=attr_map
                )
        return cls(groups=groups, attr_map=attr_map)

    def to_segmented(self, validate: bool = True) -> MultiLevelStorage:
        """Convert uniform groups into segmented format.

        Attributes belonging to segmented groups are flattened along dims 0-1,
        and uniform segment lengths are inferred from the original shape.

        Parameters
        ----------
        validate : bool
            Whether to validate the resulting batch.

        Returns
        -------
        MultiLevelStorage
        """
        flat_data: dict[str, Tensor] = {}
        seg_lengths: dict[str, Tensor] = {}
        group_shapes: dict[str, tuple[int, int]] = {}
        device: DeviceType = "cpu"

        for attr_name, tensor in self.items():
            if self.attr_map.is_segmented_attr(attr_name):
                group_name = self.attr_map.attr_to_group[attr_name]
                batch_size, num_atoms = tensor.shape[:2]
                if group_name in group_shapes:
                    if group_shapes[group_name][0] != batch_size:
                        raise ValueError(
                            f"Batch size mismatch for group '{group_name}': "
                            f"{group_shapes[group_name]} vs ({batch_size}, {num_atoms}) "
                            f"for key '{attr_name}'"
                        )
                else:
                    group_shapes[group_name] = (batch_size, num_atoms)
                tensor = tensor.flatten(0, 1)
                if group_name not in seg_lengths:
                    seg_lengths[group_name] = torch.full(
                        (batch_size,),
                        num_atoms,
                        device=tensor.device,
                        dtype=torch.int32,
                    )
            flat_data[attr_name] = tensor
            device = tensor.device

        return self.__class__.from_data(
            data=flat_data,
            attr_map=self.attr_map,
            segment_lengths=seg_lengths,
            device=device,
            validate=validate,
        )

    @classmethod
    def from_batches(
        cls,
        batches: list[MultiLevelStorage],
        strict: bool = False,
        allow_conversion: bool = True,
    ) -> MultiLevelStorage:
        """Merge multiple batches into one.

        Parameters
        ----------
        batches : list[MultiLevelStorage]
            Instances to combine.
        strict : bool
            If ``True``, require identical attribute sets.
        allow_conversion : bool
            If ``True``, automatically convert uniform batches to segmented
            format when mixing with segmented batches.

        Returns
        -------
        MultiLevelStorage

        Notes
        -----
        Uses the ``attr_map`` from the first batch.
        """
        if not batches:
            return cls(attr_map=LevelSchema())
        if len(batches) == 1:
            return batches[0]

        attr_map = batches[0].attr_map
        common_attrs = set.intersection(*(set(b.keys()) for b in batches))

        if strict and common_attrs != set(batches[0].keys()):
            raise ValueError(
                f"Attribute sets differ. Batch 0: {set(batches[0].keys())}, "
                f"common: {common_attrs}"
            )
        if not common_attrs:
            return cls(attr_map=attr_map)

        device = batches[0].device

        need_conversion = False
        if allow_conversion:
            seg_flags = [b.is_segmented() for b in batches]
            need_conversion = any(seg_flags)
            if not need_conversion:
                for attr in common_attrs:
                    vals = [b[attr] for b in batches]
                    if not cls._compatible_shapes_uniform(vals, seg_flags):
                        need_conversion = True
                        break

        if need_conversion:
            batches = [
                b.to_segmented(validate=True) if not b.is_segmented() else b
                for b in batches
            ]

        merged_data: dict[str, Tensor] = {}
        merged_seg_lengths: dict[str, Tensor] = {}
        for attr in common_attrs:
            dtype_str = attr_map.dtypes.get(attr, None)
            tensors = [
                to_tensor(b[attr], dtype=dtype_str, device=device) for b in batches
            ]
            merged_data[attr] = torch.cat(tensors, dim=0)

            attr_groups = [b.group_from_attr(attr) for b in batches]
            if all(g.is_segmented() for g in attr_groups):
                group_name = attr_map.group(attr)
                merged_seg_lengths[group_name] = torch.cat(
                    [g.segment_lengths for g in attr_groups], dim=0
                )

        return cls.from_data(
            data=merged_data,
            attr_map=attr_map,
            segment_lengths=merged_seg_lengths or None,
            device=device,
        )

    @staticmethod
    def _compatible_shapes_uniform(
        tensors: list[Tensor],
        is_segmented: list[bool],
    ) -> bool:
        """Check whether all tensors can be concatenated without segmentation.

        Returns ``False`` if any tensor comes from a segmented source or if
        trailing shapes differ.
        """
        if not tensors:
            return True
        trailing = None
        for tensor, seg in zip(tensors, is_segmented):
            if seg:
                return False
            if trailing is None:
                trailing = tensor.shape[1:]
            if tensor.shape[1:] != trailing:
                return False
        return True
