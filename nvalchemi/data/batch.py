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
"""Graph-aware Pydantic batch backed by :class:`MultiLevelStorage`.

This module provides a :class:`Batch` class that combines the Pydantic-model
interface of ``nvalchemi.data.batch.Batch`` with the performant tensor storage
of :class:`~nvalchemi.data.level_storage.MultiLevelStorage`.

Performance advantages over the Pydantic-based ``nvalchemi.data.batch.Batch``:

* **index_select** operates directly on concatenated tensors via segment
  selection -- no per-graph object reconstruction and re-batching.
* **to / clone** move / copy tensors in a single pass -- no
  ``model_dump`` / ``map_structure`` / ``model_validate`` round-trip.
* **batch_idx / batch_ptr** are lazily derived from ``segment_lengths`` -- never
  eagerly built or manually maintained.
* **No slices / cumsum bookkeeping** -- edge-index offsets are recovered
  from ``atoms.batch_ptr`` at unbatching time.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator, Sequence
from typing import Any

import numpy as np
import torch
from tensordict import TensorDict
from torch import Tensor
from torch import distributed as dist
from torch.distributed import ProcessGroup, Work

from nvalchemi.data.atomic_data import AtomicData
from nvalchemi.data.data import DataMixin
from nvalchemi.data.level_storage import (
    TORCH_DTYPE_MAP,
    LevelSchema,
    MultiLevelStorage,
    SegmentedLevelStorage,
    UniformLevelStorage,
    _checked_segment_metadata,
    _resolve_device,
)

# Edge-level keys whose values are node indices and therefore need
# cumulative node-offset correction when batching (from_data_list) and
# the reverse correction when extracting a single graph (get_data).
# This set does NOT control concatenation dimension or shape semantics;
# all edge tensors are stored as (E, ...) and concatenated on dim 0.
_INDEX_KEYS = frozenset({"neighbor_list"})
_EXCLUDED_KEYS = frozenset({"batch_idx", "batch_ptr", "device", "dtype", "info"})
_BUILTIN_LEVELS = frozenset({"atoms", "edges", "system"})
_LEVEL_ALIASES = {
    "atom": "atoms",
    "node": "atoms",
    "atoms": "atoms",
    "edge": "edges",
    "edges": "edges",
    "system": "system",
}
_INT32_MAX = torch.iinfo(torch.int32).max
_UNIFORM_BUFFER_DTYPES = frozenset(
    {torch.bool, torch.float32, torch.float64, torch.int32, torch.int64}
)


_OWN_ATTRS = frozenset({"device", "keys", "_storage", "_data_class"})


def _canonical_schema_dtype(schema: LevelSchema, key: str) -> torch.dtype | str | None:
    """Return the effective dtype for a schema field, preserving unknown names."""
    declared = schema.dtypes.get(key)
    if declared is None:
        return None
    return TORCH_DTYPE_MAP.get(declared, declared)


def _checked_product_lengths(
    left_lengths: Sequence[int],
    right_lengths: Sequence[int],
    level_name: str,
) -> list[int]:
    """Return product cardinalities after checking int32 pointer limits."""
    products: list[int] = []
    total = 0
    for left, right in zip(left_lengths, right_lengths, strict=True):
        product = int(left) * int(right)
        if product > _INT32_MAX:
            raise OverflowError(
                f"Product level '{level_name}' cardinality {product} exceeds "
                f"int32 maximum ({_INT32_MAX})"
            )
        total += product
        if total > _INT32_MAX:
            raise OverflowError(
                f"Product level '{level_name}' cumulative cardinality {total} "
                f"exceeds int32 maximum ({_INT32_MAX})"
            )
        products.append(product)
    return products


def _build_batch_storage(
    samples: Iterator[tuple[Iterator[tuple[str, Tensor]], int, int | None]],
    *,
    node_keys: frozenset[str] | set[str],
    edge_keys: frozenset[str] | set[str],
    system_keys: frozenset[str] | set[str],
    device: torch.device,
    validate: bool,
    attr_map: LevelSchema,
    field_levels: dict[str, str] | None = None,
    fallback_level: str | None = None,
) -> tuple[MultiLevelStorage, dict[str, set[str]]]:
    """Shared batch-construction pipeline for from_data_list / from_raw_dicts.

    Parameters
    ----------
    samples : Iterator
        Yields ``(key_value_pairs, num_nodes, num_edges)`` per sample.
        ``key_value_pairs`` is an iterator of ``(key, tensor)`` pairs.
    node_keys, edge_keys, system_keys : set-like
        Key sets for level classification.
    device : torch.device
        Target device for tensors.
    validate : bool
        Whether to validate storage shapes.
    attr_map : LevelSchema
        Attribute registry.
    field_levels : dict[str, str] or None, default=None
        Explicit per-field level overrides (``"atom"`` / ``"edge"`` /
        ``"system"``), typically from reader metadata.  Checked for keys
        not found in the static key sets.
    fallback_level : str or None, default=None
        Level to assign keys not in any key set and not in
        *field_levels*.  ``"system"`` for raw-dict paths.
        ``None`` to silently drop unclassified keys.

    Returns
    -------
    tuple[MultiLevelStorage, dict[str, set[str]]]
        The constructed storage and tracked key sets.
    """
    # Keep the first sample's field order in each group, while delaying all
    # concatenation until cardinalities have been checked.  Besides producing
    # clearer errors, this is important for product levels whose parent counts
    # may be inferred from another custom level.
    records: list[tuple[list[tuple[str, Tensor]], int, int | None]] = []

    def _resolve_level(level: str) -> str:
        resolved = _LEVEL_ALIASES.get(level)
        if resolved is not None:
            return resolved
        if level in attr_map.level_kinds:
            return level
        raise ValueError(
            f"Unknown field level '{level}'; expected 'atom', 'edge', 'system', "
            "or a registered level"
        )

    classifications: dict[str, str | None] = {}

    def _classify(key: str) -> str | None:
        if key in classifications:
            return classifications[key]
        if key in node_keys:
            level = "atom"
        elif key in edge_keys:
            level = "edge"
        elif key in system_keys:
            level = "system"
        elif field_levels is not None and key in field_levels:
            resolved = _resolve_level(field_levels[key])
            # Preserve explicit metadata-driven routing in the private schema
            # carried by extracted AtomicData.  Legacy key sets still win and
            # are intentionally not rewritten.
            level_kind = attr_map.level_kind(resolved)
            attr_map.set(key, resolved, is_segmented=level_kind != "uniform")
            level = resolved
        else:
            try:
                level = attr_map.group(key)
            except KeyError:
                level = fallback_level
        classifications[key] = level
        return level

    node_offset = 0
    selected_keys: tuple[str, ...] | None = None
    for sample_index, (key_value_pairs, n_nodes, n_edges) in enumerate(samples):
        pairs = list(key_value_pairs)
        if selected_keys is None:
            selected_keys = tuple(key for key, _ in pairs)
        values_by_key = dict(pairs)

        # The legacy collator selects fields from the first sample.  Keep that
        # ordering and selection for built-in fields, but reject a custom field
        # that appears only in a later sample instead of silently dropping it.
        for key, _ in pairs:
            if key in selected_keys:
                continue
            level = _classify(key)
            if (
                level is not None
                and _LEVEL_ALIASES.get(level, level) not in _BUILTIN_LEVELS
            ):
                raise ValueError(
                    f"Custom field '{key}' in level "
                    f"'{_LEVEL_ALIASES.get(level, level)}' appears only in "
                    f"later sample {sample_index}"
                )

        sample_pairs: list[tuple[str, Tensor]] = []
        for key in selected_keys:
            if key not in values_by_key:
                continue
            value = values_by_key[key]
            level = _classify(key)
            if level is None:
                continue
            value = value.to(device, non_blocking=True)
            if level == "atom":
                level = "atoms"
            elif level == "edge":
                level = "edges"
            elif level == "system":
                level = "system"
            if level == "edges" and key in _INDEX_KEYS:
                value = value + node_offset
            sample_pairs.append((key, value))
        records.append((sample_pairs, n_nodes, n_edges))
        node_offset += n_nodes

    num_samples = len(records)
    grouped: dict[str, dict[str, list[Tensor]]] = defaultdict(lambda: defaultdict(list))
    field_samples: dict[tuple[str, str], list[int]] = defaultdict(list)
    for sample_index, (pairs, _, _) in enumerate(records):
        seen = set()
        for key, value in pairs:
            group_name = _classify(key)
            if group_name is None:
                continue
            group_name = _LEVEL_ALIASES.get(group_name, group_name)
            if key in seen:
                raise ValueError(
                    f"Field '{key}' appears more than once in sample {sample_index}"
                )
            seen.add(key)
            grouped[group_name][key].append(value)
            field_samples[(group_name, key)].append(sample_index)

    # All fields selected from the first sample must be available in every
    # sample.  The old hard-coded path eventually failed on inconsistent
    # segment lengths; prevalidation gives the caller the offending field.
    for group_name, fields in grouped.items():
        for key, values in fields.items():
            if len(values) != num_samples:
                field_kind = (
                    "Custom field" if group_name not in _BUILTIN_LEVELS else "Field"
                )
                present_samples = field_samples[(group_name, key)]
                missing_samples = [
                    index
                    for index in range(num_samples)
                    if index not in present_samples
                ]
                missing_location = (
                    f"sample {missing_samples[0]}"
                    if len(missing_samples) == 1
                    else f"samples {missing_samples}"
                )
                raise ValueError(
                    f"{field_kind} '{key}' in level '{group_name}' is missing from "
                    f"{missing_location}"
                )
            if attr_map.level_kinds.get(group_name) != "product" and any(
                value.ndim == 0 for value in values
            ):
                raise ValueError(
                    f"Field '{key}' in level '{group_name}' must have a leading "
                    "cardinality dimension"
                )

    # Custom fields own their dtype and trailing shape in the effective schema
    # of this batch.  Normalize that schema from the first value only after all
    # samples have been checked, so an input schema is never mutated and a
    # rejected batch cannot leave partially updated metadata.
    inferred_dtypes: list[tuple[str, str, bool, torch.dtype]] = []
    for group_name, fields in grouped.items():
        if group_name in _BUILTIN_LEVELS:
            continue
        kind = attr_map.level_kinds[group_name]
        for key, values in fields.items():
            first_value = values[0]
            if any(value.dtype != first_value.dtype for value in values[1:]):
                raise ValueError(
                    f"Custom field '{key}' in level '{group_name}' has "
                    f"incompatible dtypes: "
                    f"{[value.dtype for value in values]}"
                )
            trailing_start = 2 if kind == "product" else 1
            trailing_shape = first_value.shape[trailing_start:]
            if any(
                value.shape[trailing_start:] != trailing_shape for value in values[1:]
            ):
                raise ValueError(
                    f"Custom field '{key}' in level '{group_name}' has "
                    f"incompatible trailing shapes: "
                    f"{[tuple(value.shape[trailing_start:]) for value in values]}"
                )

            declared_dtype = attr_map.dtypes.get(key)
            if declared_dtype is not None:
                try:
                    expected_dtype = TORCH_DTYPE_MAP[declared_dtype]
                except KeyError as exc:
                    raise ValueError(
                        f"Custom field '{key}' in level '{group_name}' has "
                        f"unsupported declared dtype '{declared_dtype}'"
                    ) from exc
                if expected_dtype != first_value.dtype:
                    raise ValueError(
                        f"Custom field '{key}' in level '{group_name}' has dtype "
                        f"{first_value.dtype}, expected declared dtype "
                        f"{declared_dtype}"
                    )

            if declared_dtype is None:
                inferred_dtypes.append(
                    (key, group_name, kind != "uniform", first_value.dtype)
                )

    for key, group_name, is_segmented, dtype in inferred_dtypes:
        attr_map.set(key, group_name, dtype=dtype, is_segmented=is_segmented)

    level_counts: dict[str, list[int | None]] = {
        "atoms": [record[1] for record in records],
        # Without a neighbor_list, AtomicData intentionally reports zero
        # edges.  Keep that cardinality unresolved here so a custom edge
        # field, or a product using edges as a fieldless parent, can infer it
        # locally without changing AtomicData's public behavior.
        "edges": [count if count not in (None, 0) else None for _, _, count in records],
    }
    if "neighbor_list" in grouped.get("edges", {}):
        level_counts["edges"] = [record[2] for record in records]
    level_kinds = attr_map.level_kinds

    # Infer ordinary segmented levels from their first field and validate all
    # remaining fields against that cardinality.  Uniform levels have one row
    # per graph, including custom ordinary levels.
    for group_name, fields in grouped.items():
        kind = level_kinds.get(group_name)
        if kind is None:
            # This is only reachable for a legacy fallback group.  Raw-dict
            # fallback is system-level, which is registered by the default
            # schema; retain a defensive error for custom callers.
            raise ValueError(f"Level '{group_name}' is not registered in the schema")
        if kind == "uniform":
            for key, values in fields.items():
                if any(value.shape[0] != 1 for value in values):
                    raise ValueError(
                        f"Uniform level '{group_name}' field '{key}' must have "
                        "one row per graph"
                    )
            level_counts[group_name] = [1] * num_samples
            continue
        if kind == "segmented":
            if (
                group_name == "edges"
                and "neighbor_list" not in fields
                and not all(count is not None for count in level_counts["edges"])
            ):
                first_values = next(iter(fields.values()))
                level_counts["edges"] = [int(value.shape[0]) for value in first_values]
            if group_name in level_counts and all(
                count is not None for count in level_counts[group_name]
            ):
                expected = [int(count) for count in level_counts[group_name]]
            else:
                first_values = next(iter(fields.values()))
                expected = [int(value.shape[0]) for value in first_values]
                level_counts[group_name] = expected
            for key, values in fields.items():
                actual = [int(value.shape[0]) for value in values]
                if actual != expected:
                    raise ValueError(
                        f"Segmented level '{group_name}' field '{key}' has "
                        f"cardinalities {actual}, expected {expected}"
                    )

    # Product fields arrive in their logical two-axis shape ``[L, R, ...]``.
    # Infer each registered parent directly from those axes, validate all
    # fields before flattening, and retain empty parent groups when needed for
    # round-tripping.  Process products in schema order so shared parents are
    # resolved deterministically.
    product_groups = [
        name
        for name in attr_map.level_names
        if level_kinds.get(name) == "product" and name in grouped
    ]
    for group_name in product_groups:
        fields = grouped[group_name]
        left, right = attr_map.product_parents[group_name]
        first_values = next(iter(fields.values()))
        if any(value.ndim < 2 for values in fields.values() for value in values):
            raise ValueError(
                f"Product level '{group_name}' fields must have rank >= 2 "
                "with shape [left, right, ...]"
            )

        left_counts = [int(value.shape[0]) for value in first_values]
        right_counts = [int(value.shape[1]) for value in first_values]
        if left == right and left_counts != right_counts:
            raise ValueError(
                f"Self-product level '{group_name}' requires equal left and "
                f"right cardinalities, got {left_counts} and {right_counts}"
            )

        for key, values in fields.items():
            actual_left = [int(value.shape[0]) for value in values]
            actual_right = [int(value.shape[1]) for value in values]
            if actual_left != left_counts or actual_right != right_counts:
                raise ValueError(
                    f"Product level '{group_name}' field '{key}' has axes "
                    f"{list(zip(actual_left, actual_right, strict=True))}, "
                    f"expected {list(zip(left_counts, right_counts, strict=True))}"
                )

        def _known_counts(name: str) -> bool:
            return name in level_counts and all(
                count is not None for count in level_counts[name]
            )

        if _known_counts(left) and [int(c) for c in level_counts[left]] != left_counts:
            raise ValueError(
                f"Product level '{group_name}' left axis cardinalities "
                f"{left_counts} do not match parent '{left}' cardinalities "
                f"{level_counts[left]}"
            )
        if (
            _known_counts(right)
            and [int(c) for c in level_counts[right]] != right_counts
        ):
            raise ValueError(
                f"Product level '{group_name}' right axis cardinalities "
                f"{right_counts} do not match parent '{right}' cardinalities "
                f"{level_counts[right]}"
            )
        if left == right:
            level_counts[left] = left_counts
        else:
            level_counts[left] = left_counts
            level_counts[right] = right_counts

        for key, values in fields.items():
            payload_shape = values[0].shape[2:]
            for value in values:
                if value.shape[2:] != payload_shape:
                    raise ValueError(
                        f"Product level '{group_name}' field '{key}' has "
                        f"trailing shape {tuple(value.shape[2:])}, expected "
                        f"{tuple(payload_shape)}"
                    )

        expected = _checked_product_lengths(left_counts, right_counts, group_name)
        level_counts[group_name] = expected
        for key, values in fields.items():
            # Flatten only after both axes and payload shapes are validated.
            fields[key] = [
                value.reshape(value.shape[0] * value.shape[1], *value.shape[2:])
                for value in values
            ]

    required_groups = set(grouped)
    for group_name in tuple(required_groups):
        if level_kinds.get(group_name) == "product":
            left, right = attr_map.product_parents[group_name]
            required_groups.update((left, right))

    groups: dict[str, UniformLevelStorage | SegmentedLevelStorage] = {}
    ordered_groups = [name for name in attr_map.level_names if name in required_groups]
    for group_name in ordered_groups:
        kind = level_kinds.get(group_name)
        fields = grouped.get(group_name, {})
        if kind == "uniform":
            if not fields:
                continue
            data = {key: torch.cat(values, dim=0) for key, values in fields.items()}
            groups[group_name] = UniformLevelStorage(
                data=data,
                device=device,
                validate=validate,
                attr_map=attr_map,
            )
        elif kind in {"segmented", "product"}:
            data = {key: torch.cat(values, dim=0) for key, values in fields.items()}
            groups[group_name] = SegmentedLevelStorage(
                data=data or None,
                device=device,
                segment_lengths=level_counts[group_name],
                validate=validate,
                attr_map=attr_map,
            )
        else:
            raise ValueError(f"Level '{group_name}' is not registered in the schema")

    storage = MultiLevelStorage(groups=groups, attr_map=attr_map, validate=validate)
    tracked_keys = {
        "node": set(grouped.get("atoms", {})),
        "edge": set(grouped.get("edges", {})),
        "system": set(grouped.get("system", {})),
    }
    return storage, tracked_keys


def _batch_device(
    device: torch.device | str | None, storage: MultiLevelStorage | None
) -> torch.device:
    """Return the device a batch records when built around *storage*.

    Batch-level allocations follow :attr:`Batch.device` rather than the
    storage's: the index tensor :meth:`Batch.index_select` builds, ``edge_ptr``,
    and the empty ``batch_idx`` / ``batch_ptr`` fallbacks. A storage holding
    data has already resolved which GPU its tensors reached, so it decides the
    recorded device; a request that names only the device type, or none at all,
    adopts it. An empty storage holds nothing to disagree with, so the request
    wins and the caller places the storage on it.

    Parameters
    ----------
    device : torch.device | str | None
        Requested device. ``None`` adopts the storage's device.
    storage : MultiLevelStorage | None
        Storage the batch will wrap, when the caller supplies one.

    Returns
    -------
    torch.device
        The storage's device when one is supplied, else the resolved request.

    Raises
    ------
    ValueError
        If *device* names an indexed device the storage is not on.
    """
    if storage is None or not storage.groups:
        return _resolve_device(device)
    if device is None:
        return storage.device
    requested = torch.device(device)
    if requested.type == "cuda" and requested.index is None:
        return storage.device
    if requested != storage.device:
        raise ValueError(
            f"Batch device {str(requested)!r} conflicts with the supplied "
            f"storage's device {str(storage.device)!r}; pass a matching device "
            "or move the storage first."
        )
    return storage.device


def set_transient(batch: "Batch", name: str, value: torch.Tensor) -> None:
    """Overlay *value* on *batch* for the duration of a computation.

    :meth:`Batch.__setattr__` routes a tensor into the batch's storage, which is
    what ``.to()``, gathering and any rebuild of the batch read back. That is the
    wrong lifetime for a value a single forward should see and no later reader
    should inherit — an affine strain applied for one autograd pass, say, which
    a rebuild would otherwise apply a second time on top of the first.

    This writes an instance attribute instead, which shadows the storage
    delegation for attribute reads while leaving the stored tensor untouched.

    Parameters
    ----------
    batch : Batch
        Batch to overlay.
    name : str
        Attribute the computation will read.
    value : torch.Tensor
        Value to expose.

    Returns
    -------
    None
    """
    object.__setattr__(batch, name, value)


def _complete_schema(schema: LevelSchema) -> LevelSchema:
    """Add missing built-in levels to an independent schema copy."""
    for level_name in ("atoms", "edges"):
        if level_name not in schema.level_kinds:
            schema.add_level(level_name, segmented=True)
        elif schema.level_kind(level_name) != "segmented":
            raise ValueError(f"Built-in level '{level_name}' must be segmented")
    if "system" not in schema.level_kinds:
        schema.add_level("system", segmented=False)
    elif schema.level_kind("system") != "uniform":
        raise ValueError("Built-in level 'system' must be uniform")
    return schema


def _effective_schema(
    data_list: Sequence[AtomicData], attr_map: LevelSchema | None
) -> LevelSchema:
    """Return an independent schema for a new batch.

    An explicitly supplied schema is authoritative.  Otherwise, a schema
    carried privately by an ``AtomicData`` instance is used when available;
    ordinary AtomicData instances fall back to the built-in schema.
    """
    if attr_map is not None:
        schema = attr_map.clone()
    else:
        schema = None
        for data in data_list:
            candidate = getattr(data, "_level_schema", None)
            if isinstance(candidate, LevelSchema):
                schema = candidate.clone()
                break
        if schema is None:
            schema = LevelSchema()

    # A deliberately minimal custom schema need not repeat the built-in levels;
    # add them only to this independent effective copy.  LevelSchema itself
    # continues to preserve its explicit constructor topology.
    return _complete_schema(schema)


def _empty_effective_schema(
    template: AtomicData | Batch | None,
    attr_map: LevelSchema | None,
) -> LevelSchema:
    """Clone the schema source selected by :meth:`Batch.empty`.

    Explicit schemas take precedence, followed by a ``Batch`` template,
    private ``AtomicData`` metadata, and finally the default schema.
    """
    if attr_map is not None:
        return _complete_schema(attr_map.clone())
    if isinstance(template, Batch):
        return _complete_schema(template._storage.attr_map.clone())
    if isinstance(template, AtomicData):
        candidate = getattr(template, "_level_schema", None)
        if isinstance(candidate, LevelSchema):
            return _complete_schema(candidate.clone())
    return _complete_schema(LevelSchema())


def _validate_level_capacities(
    schema: LevelSchema,
    level_capacities: dict[str, int] | None,
) -> dict[str, int]:
    """Validate and copy explicit custom level capacities."""
    if level_capacities is None:
        return {}
    if not isinstance(level_capacities, dict):
        raise TypeError("level_capacities must be a dictionary or None")

    capacities: dict[str, int] = {}
    for level_name, capacity in level_capacities.items():
        if not isinstance(level_name, str) or level_name not in schema.level_kinds:
            raise ValueError(f"Unknown level capacity entry '{level_name}'")
        if level_name in _BUILTIN_LEVELS:
            raise ValueError(
                f"Capacity for built-in level '{level_name}' is controlled by "
                "num_systems, num_nodes, or num_edges"
            )
        if isinstance(capacity, bool) or not isinstance(capacity, int):
            raise TypeError(
                f"Capacity for level '{level_name}' must be an integer, "
                f"got {type(capacity).__name__}"
            )
        if capacity < 0:
            raise ValueError(f"Capacity for level '{level_name}' must be non-negative")
        capacities[level_name] = capacity
    return capacities


def _validate_required_level_capacities(
    schema: LevelSchema,
    template: Batch,
    capacities: dict[str, int],
) -> None:
    """Require capacities for materialized custom payload groups."""
    for level_name, group in template._storage.groups.items():
        if level_name in _BUILTIN_LEVELS:
            continue
        kind = schema.level_kinds.get(level_name)
        if kind not in {"segmented", "product"}:
            continue
        if list(group.keys()) and level_name not in capacities:
            raise ValueError(
                f"Missing capacity for payload-bearing custom level '{level_name}'"
            )


def _validate_atomic_data_level_capacities(
    schema: LevelSchema,
    template: AtomicData,
    capacities: dict[str, int],
) -> None:
    """Require capacities for custom payload fields in an AtomicData template."""
    required: set[str] = set()
    for key, value in template.model_dump(exclude_none=True).items():
        if not isinstance(value, Tensor):
            continue
        level_name = schema.attr_to_group.get(key)
        if (
            level_name is not None
            and level_name not in _BUILTIN_LEVELS
            and schema.level_kinds.get(level_name) in {"segmented", "product"}
        ):
            required.add(level_name)
    for level_name in required:
        if level_name not in capacities:
            raise ValueError(
                f"Missing capacity for payload-bearing custom level '{level_name}'"
            )


def _batch_graph_slot_capacity(batch: Batch) -> int:
    """Return the reusable graph-slot capacity represented by *batch*."""
    capacity = batch.num_graphs
    for group in batch._storage.groups.values():
        if isinstance(group, UniformLevelStorage):
            capacity = max(capacity, group._data.shape[0])
            continue
        pointer_capacity = group._batch_ptr_capacity
        if group._batch_ptr is not None:
            pointer_capacity = group._batch_ptr.shape[0]
        if pointer_capacity is not None:
            capacity = max(capacity, max(pointer_capacity - 2, 0))
        capacity = max(capacity, len(group))
    return capacity


def _transport_payload_tag_span(
    group: UniformLevelStorage | SegmentedLevelStorage | None,
) -> int:
    """Return the tag space reserved for one level payload."""
    return len(list(group.keys())) + 1 if group is not None else 1


def _custom_transport_groups(
    batch: Batch,
) -> list[tuple[str, UniformLevelStorage | SegmentedLevelStorage]]:
    """Return materialized custom groups in schema definition order."""
    return [
        (name, batch._storage.groups[name])
        for name in batch._storage.attr_map.level_names
        if name not in _BUILTIN_LEVELS and name in batch._storage.groups
    ]


class Batch(DataMixin):
    """Graph-aware batch built on :class:`MultiLevelStorage`.

    The three built-in attribute groups are:

    * ``"atoms"`` (:class:`SegmentedLevelStorage`) -- node-level tensors
    * ``"edges"`` (:class:`SegmentedLevelStorage`) -- edge-level tensors
    * ``"system"`` (:class:`UniformLevelStorage`) -- graph-level tensors

    A :class:`LevelSchema` can register additional uniform, segmented, and
    product levels. ``batch_idx``, ``batch_ptr``, ``num_nodes_list``, and
    ``num_edges_list`` remain aliases for the built-in atom and edge levels.

    Attributes
    ----------
    device : torch.device
        Device of the underlying storage. When a storage is supplied it decides
        this value, and otherwise a bare ``cuda`` is resolved to the GPU the
        storage's tensors reached, so batch-level allocations such as
        ``edge_ptr`` and the index tensor of :meth:`index_select` never land on
        a different device than the data they are built for. Constructing a
        batch around a storage held on another indexed device raises
        ``ValueError``.
    keys : dict[str, set[str]] | None
        Level categorisation: ``{"node": ..., "edge": ..., "system": ...}``.
    """

    def __init__(
        self,
        *,
        device: torch.device | str,
        storage: MultiLevelStorage | None = None,
        keys: dict[str, set[str]] | None = None,
    ) -> None:
        resolved = _batch_device(device, storage)
        if storage is None:
            storage = MultiLevelStorage(device=resolved)
        elif not storage.groups:
            storage.to_device(resolved)
        object.__setattr__(self, "_storage", storage)
        object.__setattr__(self, "_data_class", AtomicData)
        object.__setattr__(self, "device", resolved)
        object.__setattr__(self, "keys", keys)

    def __setattr__(self, name: str, value: Any) -> None:
        if name in _OWN_ATTRS:
            object.__setattr__(self, name, value)
        elif isinstance(value, torch.Tensor):
            self._storage[name] = value
        else:
            object.__setattr__(self, name, value)

    @classmethod
    def _construct(
        cls,
        *,
        device: torch.device | str | None,
        keys: dict[str, set[str]] | None,
        storage: MultiLevelStorage,
        data_class: type = AtomicData,
    ) -> Batch:
        """Fast constructor that bypasses __init__."""
        batch = cls.__new__(cls)
        resolved = _batch_device(device, storage)
        if not storage.groups:
            storage.to_device(resolved)
        object.__setattr__(batch, "_storage", storage)
        object.__setattr__(batch, "_data_class", data_class)
        object.__setattr__(batch, "device", resolved)
        object.__setattr__(batch, "keys", keys)
        return batch

    # ------------------------------------------------------------------
    # Properties derived from storage
    # ------------------------------------------------------------------

    @property
    def num_graphs(self) -> int:
        """Number of graphs in the batch."""
        return len(self._storage)

    @property
    def batch_size(self) -> int:
        """Alias for :attr:`num_graphs`."""
        return self.num_graphs

    @property
    def num_nodes(self) -> int:
        """Total number of nodes across all graphs."""
        atoms = self._atoms_group
        return atoms.num_elements() if atoms is not None else 0

    @property
    def num_edges(self) -> int:
        """Total number of edges across all graphs."""
        edges = self._edges_group
        return edges.num_elements() if edges is not None else 0

    @property
    def system_capacity(self) -> int:
        """Maximum number of systems (graphs) this buffer can hold (e.g. from :meth:`empty`)."""
        system = self._system_group
        if system is None:
            return 0
        return system._data.shape[0]

    @property
    def batch_idx(self) -> Tensor:
        """Per-node graph assignment tensor (lazily computed)."""
        atoms = self._atoms_group
        if atoms is None:
            return torch.tensor([], dtype=torch.long, device=self.device)
        return atoms.batch_idx

    @property
    def batch_ptr(self) -> Tensor:
        """Cumulative node count per graph (lazily computed)."""
        atoms = self._atoms_group
        if atoms is None:
            return torch.zeros(1, dtype=torch.int32, device=self.device)
        return atoms.batch_ptr

    @property
    def level_keys(self) -> dict[str, set[str]]:
        """Return fields for every level whose cardinality is resolvable.

        The mapping follows the effective :class:`LevelSchema` order.  Empty
        sets represent registered uniform levels, fieldless built-in levels,
        or product levels whose segmented parents provide their cardinality.
        Unmaterialized custom segmented levels are omitted because their
        cardinality is unresolved.
        """
        result: dict[str, set[str]] = {}
        for name in self._storage.attr_map.level_names:
            group = self._storage.groups.get(name)
            if group is not None:
                result[name] = set(group.keys())
                continue
            try:
                self.level_ptr(name)
            except KeyError:
                continue
            result[name] = set()
        return result

    def level_ptr(self, name: str) -> Tensor:
        """Return the cumulative per-graph pointer for a registered level.

        Materialized segmented and product levels expose their storage
        pointers.  Uniform levels have one row per graph and therefore use
        ``arange(B + 1)``.  Missing built-in ``atoms`` and ``edges`` levels
        are treated as fieldless zero-cardinality levels.  An unmaterialized
        product can be resolved from its two segmented parent pointers.

        Parameters
        ----------
        name : str
            Registered level name.

        Returns
        -------
        torch.Tensor
            Cumulative element counts with length ``num_graphs + 1`` and
            ``torch.int32`` dtype.

        Raises
        ------
        OverflowError
            If the level's graph count, per-graph product cardinality, or
            cumulative cardinality exceeds the signed int32 pointer range.
        KeyError
            If *name* is not registered or its segmented cardinality cannot
            be resolved from materialized data or resolved product parents.
        """
        schema = self._storage.attr_map
        if name not in schema.level_kinds:
            raise KeyError(f"Level '{name}' not found")

        group = self._storage.groups.get(name)
        if group is not None:
            if isinstance(group, SegmentedLevelStorage):
                return group.batch_ptr[: self.num_graphs + 1]
            if self.num_graphs > _INT32_MAX:
                raise OverflowError(
                    f"Uniform level '{name}' graph count exceeds int32 maximum "
                    f"({_INT32_MAX})"
                )
            return torch.arange(
                self.num_graphs + 1, dtype=torch.int32, device=self.device
            )

        kind = schema.level_kind(name)
        if kind == "uniform":
            if self.num_graphs > _INT32_MAX:
                raise OverflowError(
                    f"Uniform level '{name}' graph count exceeds int32 maximum "
                    f"({_INT32_MAX})"
                )
            return torch.arange(
                self.num_graphs + 1, dtype=torch.int32, device=self.device
            )

        if name in _BUILTIN_LEVELS:
            return torch.zeros(
                self.num_graphs + 1, dtype=torch.int32, device=self.device
            )

        if kind == "product":
            left, right = schema.product_parents[name]
            try:
                left_ptr = self.level_ptr(left)
                right_ptr = self.level_ptr(right)
            except KeyError as exc:
                raise KeyError(
                    f"Level '{name}' has unresolved parent cardinality"
                ) from exc
            left_lengths = left_ptr[1:] - left_ptr[:-1]
            right_lengths = right_ptr[1:] - right_ptr[:-1]
            product_lengths = left_lengths.to(torch.int64) * right_lengths.to(
                torch.int64
            )
            if (
                product_lengths.numel()
                and int(product_lengths.max().item()) > _INT32_MAX
            ):
                raise OverflowError(
                    f"Product level '{name}' cardinality exceeds int32 maximum "
                    f"({_INT32_MAX})"
                )
            cumulative = torch.cumsum(product_lengths, dim=0, dtype=torch.int64)
            if cumulative.numel() and int(cumulative[-1].item()) > _INT32_MAX:
                raise OverflowError(
                    f"Product level '{name}' cumulative cardinality exceeds "
                    f"int32 maximum ({_INT32_MAX})"
                )
            return torch.cat(
                [
                    torch.zeros(1, dtype=torch.int32, device=self.device),
                    cumulative.to(torch.int32),
                ]
            )

        raise KeyError(f"Level '{name}' has unresolved segmented cardinality")

    @property
    def edge_ptr(self) -> Tensor:
        """Per-atom CSR pointer into the edge list (N+1,), int32.

        Returns a tensor where ``edge_ptr[i] : edge_ptr[i+1]`` is the slice of
        edge rows in ``neighbor_list`` that belong to atom ``i`` (i.e. where atom
        ``i`` is the sender).  Valid only after a COO-format
        :class:`~nvalchemi.hooks.NeighborListHook` has populated the
        edges group.

        An all-zeros pointer of length ``num_nodes + 1`` is returned when the
        edges group is absent or empty.
        """
        edges = self._edges_group
        if edges is None or edges.num_elements() == 0:
            N = self.num_nodes
            return torch.zeros(N + 1, dtype=torch.int32, device=self.device)
        ei = edges["neighbor_list"]  # (E, 2)
        N = self.num_nodes
        src = ei[:, 0]  # (E,)
        counts = torch.zeros(N, dtype=torch.int32, device=self.device)
        counts.scatter_add_(
            0, src, torch.ones(src.shape[0], dtype=torch.int32, device=self.device)
        )
        ptr = torch.zeros(N + 1, dtype=torch.int32, device=self.device)
        ptr[1:] = counts.cumsum(0)
        return ptr

    @property
    def num_nodes_list(self) -> list[int]:
        """Per-graph node counts as a Python list."""
        atoms = self._atoms_group
        if atoms is None:
            return []
        return atoms.segment_lengths[: len(atoms)].tolist()

    @property
    def num_edges_list(self) -> list[int]:
        """Per-graph edge counts as a Python list."""
        edges = self._edges_group
        if edges is None:
            return []
        return edges.segment_lengths[: len(edges)].tolist()

    @property
    def num_nodes_per_graph(self) -> Tensor:
        """Per-graph node counts as a tensor."""
        atoms = self._atoms_group
        if atoms is None:
            return torch.tensor([], dtype=torch.long, device=self.device)
        return atoms.segment_lengths[: len(atoms)]

    @property
    def num_edges_per_graph(self) -> Tensor:
        """Per-graph edge counts as a tensor."""
        edges = self._edges_group
        if edges is None:
            return torch.tensor([], dtype=torch.long, device=self.device)
        return edges.segment_lengths[: len(edges)]

    @property
    def max_num_nodes(self) -> int:
        """Maximum node count in any graph."""
        nodes = self.num_nodes_list
        return max(nodes) if nodes else 0

    # ------------------------------------------------------------------
    # Internal group accessors
    # ------------------------------------------------------------------

    @property
    def _atoms_group(self) -> SegmentedLevelStorage | None:
        g = self._storage.groups.get("atoms")
        return g if isinstance(g, SegmentedLevelStorage) else None

    @property
    def _edges_group(self) -> SegmentedLevelStorage | None:
        g = self._storage.groups.get("edges")
        return g if isinstance(g, SegmentedLevelStorage) else None

    @property
    def _system_group(self) -> UniformLevelStorage | None:
        return self._storage.groups.get("system")

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_data_list(
        cls,
        data_list: list[AtomicData],
        device: torch.device | str | None = None,
        skip_validation: bool = False,
        attr_map: LevelSchema | None = None,
        exclude_keys: list[str] | None = None,
        field_levels: dict[str, str] | None = None,
    ) -> Batch:
        """Construct a batch from a list of :class:`AtomicData` objects.

        Parameters
        ----------
        data_list : list[AtomicData]
            Individual graphs to batch.
        device : torch.device | str, optional
            Target device.  Inferred from *data_list* if ``None``.
        skip_validation : bool
            If ``True``, skip shape validation for speed.
        attr_map : LevelSchema, optional
            Attribute registry. If omitted, reuse schema metadata carried by
            an input object or fall back to ``LevelSchema()``.
        exclude_keys : list[str], optional
            Keys to exclude from batching.
        field_levels : dict[str, str], optional
            Explicit per-field level map. Values may be built-in aliases such
            as ``"atom"``, ``"edge"``, and ``"system"``, or names registered
            in *attr_map*. Used to classify keys absent from the data class key
            sets.

        Returns
        -------
        Batch
        """
        if not data_list:
            raise ValueError("Cannot create batch from empty data list")

        if device is None:
            device = data_list[0].device
        device = torch.device(device) if isinstance(device, str) else device

        attr_map = _effective_schema(data_list, attr_map)

        representative = data_list[0]
        data_cls = representative.__class__
        node_key_set = representative.__node_keys__
        edge_key_set = representative.__edge_keys__
        system_key_set = representative.__system_keys__

        excluded = _EXCLUDED_KEYS | set(exclude_keys or [])

        # Iterate keys in dict (= pydantic field declaration) order so that
        # downstream insertion order into grouped storage is deterministic
        # across processes. Using ``set(...)`` here would iterate in
        # PYTHONHASHSEED-dependent order, producing rank-divergent group dicts
        # that break collective issue ordering in DD.
        def _iter_samples() -> Iterator[tuple[Iterator[tuple[str, Tensor]], int, int]]:
            for data in data_list:
                pairs = (
                    (key, value)
                    for key, value in data.model_dump(exclude_none=True).items()
                    if key not in excluded and isinstance(value, Tensor)
                )
                yield pairs, data.num_nodes, data.num_edges

        storage, tracked_keys = _build_batch_storage(
            _iter_samples(),
            node_keys=node_key_set,
            edge_keys=edge_key_set,
            system_keys=system_key_set,
            device=device,
            validate=not skip_validation,
            attr_map=attr_map,
            field_levels=field_levels,
        )
        batch = cls._construct(
            device=device,
            keys=tracked_keys,
            storage=storage,
            data_class=data_cls,
        )
        return batch._make_contiguous()

    @classmethod
    def from_raw_dicts(
        cls,
        data_list: list[dict[str, Tensor]],
        device: torch.device | str | None = None,
        attr_map: LevelSchema | None = None,
        exclude_keys: list[str] | None = None,
        field_levels: dict[str, str] | None = None,
    ) -> Batch:
        """Construct a batch directly from raw tensor dictionaries.

        Bypasses :class:`AtomicData` construction and Pydantic validation
        entirely, using ``AtomicData._default_*_keys`` for level
        classification.  Keys not found in the default key sets are
        classified using *field_levels* (e.g. from
        :attr:`Reader.field_levels`).  This is significantly faster when
        the data is already known to be well-formed (e.g. read from a
        validated Zarr store).

        Parameters
        ----------
        data_list : list[dict[str, Tensor]]
            Per-sample tensor dictionaries.
        device : torch.device | str, optional
            Target device.  Inferred from first dict if ``None``.
        attr_map : LevelSchema, optional
            Attribute registry. Defaults to ``LevelSchema()``.
        exclude_keys : list[str], optional
            Keys to exclude from batching.
        field_levels : dict[str, str], optional
            Explicit per-field level map. Values may be built-in aliases such
            as ``"atom"``, ``"edge"``, and ``"system"``, or names registered
            in *attr_map*. Used to classify keys absent from the default key
            sets.

        Returns
        -------
        Batch
        """
        if not data_list:
            raise ValueError("Cannot create batch from empty data list")

        first = data_list[0]
        if device is None:
            for v in first.values():
                if isinstance(v, Tensor):
                    device = v.device
                    break
            else:
                device = torch.device("cpu")
        device = torch.device(device) if isinstance(device, str) else device

        attr_map = _complete_schema(
            attr_map.clone() if attr_map is not None else LevelSchema()
        )

        node_key_set = AtomicData._default_node_keys
        edge_key_set = AtomicData._default_edge_keys
        system_key_set = AtomicData._default_system_keys

        excluded = _EXCLUDED_KEYS | set(exclude_keys or [])

        def _iter_samples() -> Iterator[tuple[Iterator[tuple[str, Tensor]], int, int]]:
            for data in data_list:
                n_nodes = data["atomic_numbers"].shape[0]
                nl = data.get("neighbor_list")
                n_edges = nl.shape[0] if isinstance(nl, Tensor) else 0
                pairs = (
                    (key, value)
                    for key, value in data.items()
                    if key not in excluded and isinstance(value, Tensor)
                )
                yield pairs, n_nodes, n_edges

        storage, tracked_keys = _build_batch_storage(
            _iter_samples(),
            node_keys=node_key_set,
            edge_keys=edge_key_set,
            system_keys=system_key_set,
            device=device,
            validate=False,
            attr_map=attr_map,
            field_levels=field_levels,
            fallback_level="system",
        )
        batch = cls._construct(
            device=device,
            keys=tracked_keys,
            storage=storage,
            data_class=AtomicData,
        )
        return batch._make_contiguous()

    @classmethod
    def empty(
        cls,
        *,
        num_systems: int,
        num_nodes: int,
        num_edges: int,
        template: AtomicData | Batch | None = None,
        device: torch.device | str = "cpu",
        attr_map: LevelSchema | None = None,
        level_capacities: dict[str, int] | None = None,
    ) -> Batch:
        """Construct an empty batch with pre-allocated capacity (zero graphs, fixed storage).

        Storage tensors are allocated with the given capacities; no graphs are
        stored initially (``num_graphs == 0``). Use :meth:`put` to copy graphs
        into the buffer; pass ``dest_mask`` of shape ``(num_systems,)`` with
        ``False`` for empty slots.

        Parameters
        ----------
        num_systems : int
            Maximum number of systems (graphs) the buffer can hold.
        num_nodes : int
            Total node (atom) capacity across all graphs.
        num_edges : int
            Total edge capacity across all graphs.
        template : AtomicData or Batch, optional
            Template for attribute keys and per-key shapes/dtypes. If ``None``,
            a minimal :class:`AtomicData` with ``positions``, ``atomic_numbers``,
            and ``energy`` is used.
        device : torch.device or str, optional
            Device for allocated tensors.
        attr_map : LevelSchema, optional
            Attribute registry; used when template is provided.
        level_capacities : dict[str, int], optional
            Explicit capacities for payload-bearing custom segmented and
            product levels. Custom uniform levels use ``num_systems``;
            fieldless segmented levels need no entry.

        Returns
        -------
        Batch
            Batch with ``num_graphs == 0`` and capacity for the given sizes.

        Raises
        ------
        TypeError
            If a custom capacity is not an integer.
        ValueError
            If a capacity is negative, names an unknown or built-in level, or
            is missing for a payload-bearing custom segmented or product
            level in the template.
        """
        if num_systems < 0 or num_nodes < 0 or num_edges < 0:
            raise ValueError(
                "num_systems, num_nodes, and num_edges must be non-negative"
            )
        device = torch.device(device) if isinstance(device, str) else device
        effective_schema = _empty_effective_schema(template, attr_map)
        capacities = _validate_level_capacities(effective_schema, level_capacities)
        if isinstance(template, AtomicData):
            _validate_atomic_data_level_capacities(
                effective_schema, template, capacities
            )
        elif isinstance(template, Batch):
            _validate_required_level_capacities(effective_schema, template, capacities)

        if template is None:
            template = AtomicData(
                positions=torch.zeros(1, 3),
                atomic_numbers=torch.zeros(1, dtype=torch.long),
                energy=torch.tensor([[0.0]]),
            )
        if isinstance(template, AtomicData):
            ref = cls.from_data_list(
                [template], device=device, attr_map=effective_schema
            )
        else:
            ref = template

        groups: dict[str, UniformLevelStorage | SegmentedLevelStorage] = {}
        ordered_names = [
            name for name in effective_schema.level_names if name in ref._storage.groups
        ]
        ordered_names.extend(
            name for name in ref._storage.groups if name not in ordered_names
        )
        for name in ordered_names:
            group = ref._storage.groups[name]
            keys = list(group.keys())
            kind = effective_schema.level_kinds.get(name)
            if kind is None:
                raise ValueError(f"Level '{name}' is not registered in the schema")
            if not keys and name in _BUILTIN_LEVELS:
                continue
            if name == "system" or kind == "uniform":
                data = {
                    k: torch.zeros(
                        (num_systems,) + group[k].shape[1:],
                        device=device,
                        dtype=group[k].dtype,
                    )
                    for k in keys
                }
                if not data:
                    # A fieldless uniform group is metadata-only but still
                    # occupies one graph slot per system.
                    storage = UniformLevelStorage(
                        data=None,
                        device=device,
                        validate=False,
                        attr_map=effective_schema,
                    )
                    storage._data = TensorDict(
                        {}, batch_size=[num_systems], device=device
                    )
                else:
                    storage = UniformLevelStorage(
                        data=data,
                        device=device,
                        validate=False,
                        attr_map=effective_schema,
                    )
                object.__setattr__(storage, "_num_kept", 0)
                groups[name] = storage
            else:
                if keys:
                    if name == "atoms":
                        data_capacity = num_nodes
                    elif name == "edges":
                        data_capacity = num_edges
                    else:
                        data_capacity = capacities[name]
                else:
                    data_capacity = 0
                data = {
                    k: torch.zeros(
                        (data_capacity,) + group[k].shape[1:],
                        device=device,
                        dtype=group[k].dtype,
                    )
                    for k in keys
                }
                groups[name] = SegmentedLevelStorage(
                    data=data or None,
                    segment_lengths=torch.tensor([], device=device, dtype=torch.int32),
                    device=device,
                    batch_ptr_capacity=max(num_systems + 2, 2),
                    validate=False,
                    attr_map=effective_schema,
                )

        storage = MultiLevelStorage(
            groups=groups, attr_map=effective_schema, validate=False
        )
        return cls._construct(
            device=device,
            keys={k: v.copy() for k, v in ref.keys.items()} if ref.keys else None,
            storage=storage,
            data_class=ref._data_class,
        )

    def zero(self) -> None:
        """Reset this batch to an empty-but-allocated state.

        Zeros all leaf data tensors while preserving the allocated storage
        capacity.  After calling ``zero()``, ``num_graphs`` returns 0 but
        ``system_capacity`` remains unchanged.

        This method is used to reset pre-allocated communication buffers
        (created via :meth:`empty`) between pipeline steps without
        reallocating memory.

        Notes
        -----
        Modeled after :meth:`GPUBuffer.zero` in ``nvalchemi.dynamics.sinks``.
        Resets bookkeeping for both :class:`UniformLevelStorage` (``_num_kept``)
        and :class:`SegmentedLevelStorage` (``segment_lengths``, ``_batch_ptr``).

        Examples
        --------
        >>> batch = Batch.empty(num_systems=10, num_nodes=100, num_edges=200)
        >>> batch.zero()
        >>> batch.num_graphs
        0
        >>> batch.system_capacity
        10
        """
        for group in self._storage.groups.values():
            group._data.apply_(lambda x: x.zero_())

            if hasattr(group, "_num_kept"):
                object.__setattr__(group, "_num_kept", 0)

            if hasattr(group, "segment_lengths"):
                group.segment_lengths = torch.empty(
                    0,
                    dtype=group.segment_lengths.dtype,
                    device=group.segment_lengths.device,
                )
                if group._batch_ptr is not None:
                    batch_ptr_capacity = group._batch_ptr.shape[0]
                    group._batch_ptr = torch.zeros(
                        batch_ptr_capacity,
                        dtype=torch.int32,
                        device=group.device,
                    )
                    group._batch_ptr_capacity = batch_ptr_capacity
                if hasattr(group, "_batch_idx"):
                    group._batch_idx = None
                group._batch_ptr_np = None
                group._segment_indices = None
                if hasattr(group, "_num_segments"):
                    object.__delattr__(group, "_num_segments")
                if hasattr(group, "_num_elements_kept"):
                    object.__delattr__(group, "_num_elements_kept")
                if hasattr(group, "_copied_mask"):
                    object.__delattr__(group, "_copied_mask")
        if hasattr(self, "_copied_mask"):
            object.__delattr__(self, "_copied_mask")

    # ------------------------------------------------------------------
    # Per-graph reconstruction
    # ------------------------------------------------------------------

    def get_data(self, idx: int) -> AtomicData:
        """Reconstruct the :class:`AtomicData` object at position *idx*.

        Edge-index offsets applied during batching are automatically undone.

        Parameters
        ----------
        idx : int
            Graph index (supports negative indexing).

        Returns
        -------
        AtomicData
        """
        if not (-self.num_graphs <= idx < self.num_graphs):
            raise IndexError(
                f"graph index {idx} is out of range for batch with "
                f"{self.num_graphs} graph(s)"
            )
        if idx < 0:
            idx = self.num_graphs + idx

        data: dict[str, Any] = {}

        atoms = self._atoms_group
        if atoms is not None:
            atoms._lazy_init_batch_ptr()
        node_offset = atoms._batch_ptr[idx] if atoms is not None else 0

        # Iterate in schema order, not the historical atoms/edges/system order.
        # This preserves deterministic field order for custom ordinary and
        # product levels while retaining the built-in neighbor-list correction.
        for group_name in self._storage.attr_map.level_names:
            group = self._storage.groups.get(group_name)
            if group is None:
                continue
            if isinstance(group, SegmentedLevelStorage):
                group._lazy_init_batch_ptr()
                start = group._batch_ptr[idx].item()
                end = group._batch_ptr[idx + 1].item()
                product_shape: tuple[int, int] | None = None
                if self._storage.attr_map.level_kind(group_name) == "product":
                    left, right = self._storage.attr_map.product_parents[group_name]
                    left_group = self._storage.groups.get(left)
                    right_group = self._storage.groups.get(right)
                    if not isinstance(
                        left_group, SegmentedLevelStorage
                    ) or not isinstance(right_group, SegmentedLevelStorage):
                        raise RuntimeError(
                            f"Product level '{group_name}' has missing segmented "
                            "parent storage"
                        )
                    left_group._lazy_init_batch_ptr()
                    right_group._lazy_init_batch_ptr()
                    product_shape = (
                        int(
                            left_group._batch_ptr[idx + 1] - left_group._batch_ptr[idx]
                        ),
                        int(
                            right_group._batch_ptr[idx + 1]
                            - right_group._batch_ptr[idx]
                        ),
                    )
                for key, tensor in group.items():
                    value = tensor[start:end]
                    if group_name == "edges" and key in _INDEX_KEYS:
                        value = value - node_offset
                    if product_shape is not None:
                        value = value.reshape(*product_shape, *value.shape[1:])
                    data[key] = value
            else:
                for key, tensor in group.items():
                    data[key] = tensor[idx].unsqueeze(0)

        # Pass storage-group key sets so dynamically-added keys
        # (e.g. system_id) survive the round-trip through model_post_init.
        if atoms is not None:
            data["__node_keys__"] = set(atoms.keys())
        edges = self._edges_group
        if edges is not None:
            data["__edge_keys__"] = set(edges.keys())
        system = self._system_group
        if system is not None:
            data["__system_keys__"] = set(system.keys())

        result = self._data_class(**data)
        # Custom level definitions are intentionally private: they are needed
        # for an immediate unbatch/rebatch cycle but are not AtomicData fields.
        schema = self._storage.attr_map.clone()
        if hasattr(result, "_level_schema"):
            result._level_schema = schema
        return result

    def to_data_list(self) -> list[AtomicData]:
        """Reconstruct all individual :class:`AtomicData` objects.

        Returns
        -------
        list[AtomicData]
        """
        return [self.get_data(i) for i in range(self.num_graphs)]

    # ------------------------------------------------------------------
    # Selection / indexing
    # ------------------------------------------------------------------

    def index_select(
        self,
        idx: int | slice | Tensor | list[int] | np.ndarray | Sequence[int],
    ) -> Batch:
        """Select a subset of graphs by index.

        Operates directly on concatenated tensors via segment selection --
        no per-graph :class:`AtomicData` reconstruction.

        Parameters
        ----------
        idx : int, slice, Tensor, list[int], np.ndarray, or Sequence[int]
            Graph-level index specification.

        Returns
        -------
        Batch
        """
        idx_list = self._normalize_index(idx)
        idx_tensor = torch.tensor(idx_list, dtype=torch.int32, device=self.device)

        new_groups: dict[str, UniformLevelStorage | SegmentedLevelStorage] = {}

        atoms = self._atoms_group
        offset_diff: Tensor | None = None
        if atoms is not None:
            old_offsets = atoms.batch_ptr[idx_tensor]
            new_atoms = atoms.select(idx_tensor)
            new_atoms._lazy_init_batch_ptr()
            new_offsets = new_atoms._batch_ptr[:-1]
            offset_diff = old_offsets - new_offsets

        for group_name in self._storage.attr_map.level_names:
            group = self._storage.groups.get(group_name)
            if group is None:
                continue
            if isinstance(group, SegmentedLevelStorage):
                new_group = group.select(idx_tensor)
                if (
                    group_name == "edges"
                    and "neighbor_list" in new_group
                    and offset_diff is not None
                ):
                    new_group._lazy_init_batch_ptr()
                    edge_batch_idx = new_group.batch_idx
                    correction = offset_diff[edge_batch_idx]
                    new_group._data["neighbor_list"] = new_group[
                        "neighbor_list"
                    ] - correction.unsqueeze(1)
            else:
                new_group = group.select(idx_tensor)
            new_groups[group_name] = new_group

        new_schema = self._storage.attr_map.clone()
        for group in new_groups.values():
            group.attr_map = new_schema
        new_storage = MultiLevelStorage(
            groups=new_groups,
            attr_map=new_schema,
            validate=False,
        )
        return Batch._construct(
            device=self.device,
            keys={k: v.copy() for k, v in self.keys.items()} if self.keys else None,
            storage=new_storage,
            data_class=self._data_class,
        )

    def put(
        self,
        src_batch: Batch,
        mask: Tensor,
        *,
        copied_mask: Tensor | None = None,
        dest_mask: Tensor | None = None,
    ) -> None:
        """Put graphs where mask[i] is True from src_batch into this batch (buffer).

        Computes a fit mask for every materialized level and copies only graphs
        that fit in all levels. Built-in and custom uniform payloads may use
        ``bool``, ``float32``, ``float64``, ``int32``, or ``int64``. Materialized
        custom segmented and product payloads must use ``torch.float32``. If
        *copied_mask* is provided, it is updated with the combined copy mask for
        :meth:`defrag`.

        Parameters
        ----------
        src_batch : Batch
            Source batch. Custom schemas and materialized custom groups must
            match the destination buffer.
        mask : Tensor
            (num_graphs,) bool, True = consider copying this graph.
        copied_mask : Tensor, optional
            (num_graphs,) bool; if provided, modified in place with the actual
            copy mask (fit in all levels). If None, stored on *src_batch*.
        dest_mask : Tensor, optional
            Shared occupancy mask for uniform levels, with ``True`` denoting an
            occupied destination slot. Occupied slots must form a dense prefix.
            If ``None``, all slots are available.

        Raises
        ------
        ValueError
            If *src_batch* is on another device than this batch, or if a mask's
            length does not match ``src_batch.num_graphs``.

        Notes
        -----
        The copy runs as a Warp kernel over both batches' raw pointers, so a
        source on another device is rejected rather than moved: moving it would
        hide a per-step host-device transfer inside what callers use as an
        in-place buffer write.
        """
        device = self.device
        if src_batch.device != device:
            raise ValueError(
                f"put requires src_batch on {str(device)!r}, got "
                f"{str(src_batch.device)!r}; move it with src_batch.to(...) first."
            )
        n = src_batch.num_graphs
        if mask.shape[0] != n:
            raise ValueError(f"mask shape {mask.shape[0]} != num_graphs {n}")
        self._validate_custom_put(src_batch)
        self._prevalidate_buffer_put(src_batch)
        mask = mask.to(device=device, dtype=torch.bool)
        if copied_mask is not None:
            if copied_mask.shape[0] != n:
                raise ValueError(f"copied_mask shape {copied_mask.shape[0]} != {n}")
            copy_mask = copied_mask.to(device=device, dtype=torch.bool)
        else:
            copy_mask = torch.zeros(n, device=device, dtype=torch.bool)
            object.__setattr__(src_batch, "_copied_mask", copy_mask)

        # Uniform groups all represent the same graph slots.  Snapshot the
        # caller's occupancy before either group mutates it, so each fit and
        # copy sees the same free slots.  The caller's mask is updated once
        # after all uniform groups have copied.
        uniform_dest_mask = (
            dest_mask.to(device=device, dtype=torch.bool)
            if dest_mask is not None
            else None
        )

        fit_mask = torch.ones(n, device=device, dtype=torch.bool)

        # Keep the historical built-in order for communication layouts, then
        # process custom levels in schema order.
        legacy_groups = ("system", "atoms", "edges")
        custom_groups = tuple(
            name
            for name in self._storage.attr_map.level_names
            if name not in _BUILTIN_LEVELS
        )
        fit_groups: list[tuple[str, Any, Any]] = []
        for group_name in (*legacy_groups, *custom_groups):
            dest_group = self._storage.groups.get(group_name)
            src_group = src_batch._storage.groups.get(group_name)
            if (
                dest_group is not None
                and src_group is not None
                and not (
                    isinstance(dest_group, UniformLevelStorage)
                    and dest_group._data.is_empty()
                    and src_group._data.is_empty()
                )
            ):
                fit_groups.append((group_name, dest_group, src_group))

        # Compute every fit before entering the copy phase so a rejected
        # custom group cannot leave built-in data partially written.
        for _, group, src_group in fit_groups:
            level_fit = torch.empty(n, device=device, dtype=torch.bool)
            group_dest_mask = uniform_dest_mask if not group.is_segmented() else None
            group.compute_put_per_system_fit_mask(
                src_group, mask, group_dest_mask, level_fit
            )
            fit_mask.logical_and_(level_fit)
        copy_mask.copy_(fit_mask)

        final_uniform_dest_mask: Tensor | None = None
        for _, group, src_group in fit_groups:
            if group.is_segmented():
                group.put(src_group, copy_mask, copied_mask=copy_mask)
            else:
                group_dest_mask = (
                    uniform_dest_mask.clone()
                    if uniform_dest_mask is not None
                    else torch.zeros(
                        group._data.shape[0], device=device, dtype=torch.bool
                    )
                )
                group.put(
                    src_group,
                    copy_mask,
                    copied_mask=copy_mask,
                    dest_mask=group_dest_mask,
                )
                if final_uniform_dest_mask is None:
                    final_uniform_dest_mask = group_dest_mask

        if dest_mask is not None and final_uniform_dest_mask is not None:
            dest_mask.copy_(
                final_uniform_dest_mask.to(
                    device=dest_mask.device, dtype=dest_mask.dtype
                )
            )

    def defrag(
        self,
        copied_mask: Tensor | None = None,
    ) -> Batch:
        """Defrag this batch in-place by removing graphs that were put.

        Drops graphs where ``copied_mask[i]`` is ``True`` from every
        materialized level, including fieldless segmented metadata. Payloads
        must use a buffer-kernel-supported dtype: ``bool``, ``float32``,
        ``float64``, ``int32``, or ``int64``.

        Parameters
        ----------
        copied_mask : Tensor, optional
            (num_graphs,) bool; if None, uses stored value from last :meth:`put`.

        Returns
        -------
        Self
            For method chaining.
        """
        if copied_mask is None:
            copied_mask = getattr(self, "_copied_mask", None)
            if copied_mask is None:
                raise ValueError("defrag requires copied_mask or a prior put")
        self._prevalidate_buffer_defrag()
        # Keep the historical built-in order, then compact each materialized
        # custom level in schema order using the same graph mask.
        legacy_groups = ("system", "atoms", "edges")
        custom_groups = tuple(
            name
            for name in self._storage.attr_map.level_names
            if name not in _BUILTIN_LEVELS
        )
        for group_name in (*legacy_groups, *custom_groups):
            group = self._storage.groups.get(group_name)
            if group is not None:
                group.defrag(copied_mask=copied_mask)
        if hasattr(self, "_copied_mask"):
            object.__delattr__(self, "_copied_mask")
        return self

    def trim(
        self,
        copied_mask: Tensor | None = None,
    ) -> Batch | None:
        """Remove marked graphs and return a new :class:`Batch` with tight storage.

        Unlike :meth:`defrag`, which compacts data to the front of
        pre-allocated buffers while preserving their capacity (ideal for
        fixed-size GPU buffers that will be reused with :meth:`put`),
        ``trim`` produces a brand-new :class:`Batch` whose underlying
        storage tensors are sized to exactly fit the remaining graphs —
        no padding, no unused trailing slots.

        Use :meth:`defrag` when you need to keep the buffer alive for
        further :meth:`put` / :meth:`defrag` cycles (e.g. communication
        buffers).  Use ``trim`` when the batch will be consumed directly
        by a model or integrator and must have self-consistent tensor
        shapes across all storage groups.

        Parameters
        ----------
        copied_mask : Tensor, optional
            ``(num_graphs,)`` boolean tensor where ``True`` marks graphs
            to remove.  If *None*, uses the ``_copied_mask`` stored by
            the most recent :meth:`put`.

        Returns
        -------
        Batch or None
            A new :class:`Batch` containing only the kept graphs with
            all tensors sized to exactly fit, or *None* if every graph
            was removed.

        Raises
        ------
        ValueError
            If no *copied_mask* is provided and no prior :meth:`put`
            has stored one.

        See Also
        --------
        defrag : In-place compaction that preserves buffer capacity.
        """
        if copied_mask is None:
            copied_mask = getattr(self, "_copied_mask", None)
            if copied_mask is None:
                raise ValueError("trim requires copied_mask or a prior put")
        keep_mask = ~copied_mask
        if not keep_mask.any():
            return None
        keep_indices = torch.where(keep_mask)[0]
        return self.index_select(keep_indices)

    def _normalize_index(
        self,
        idx: int | slice | Tensor | list[int] | np.ndarray | Sequence[int],
    ) -> list[int]:
        """Convert various index types to a flat list of integer indices."""
        match idx:
            case int():
                result = [idx]
            case slice():
                result = list(range(self.num_graphs)[idx])
            case Tensor():
                if idx.dtype == torch.bool:
                    result = idx.flatten().nonzero(as_tuple=False).flatten().tolist()
                elif idx.dtype.is_floating_point:
                    raise IndexError(
                        f"Tensor index must be integer or bool, got {idx.dtype}"
                    )
                else:
                    result = idx.flatten().tolist()
            case np.ndarray():
                if idx.dtype == np.bool_:
                    result = idx.flatten().nonzero()[0].flatten().tolist()
                else:
                    result = idx.flatten().tolist()
            case list():
                result = idx
            case _ if isinstance(idx, Sequence) and not isinstance(idx, str):
                result = list(idx)
            case _:
                raise IndexError(f"Unsupported index type: {type(idx).__name__}")
        if not result:
            raise IndexError("Index is empty")
        return [self.num_graphs + i if i < 0 else i for i in result]

    def __getitem__(self, key: str | int | slice | Tensor | list) -> Any:
        """Access an attribute by name, or select graphs by index.

        Parameters
        ----------
        key : str or index
            Attribute name (returns tensor) or graph index (returns
            :class:`AtomicData` for int, :class:`Batch` for slice/tensor).
        """
        match key:
            case str():
                return self._get_attr(key)
            case int():
                return self.get_data(key)
            case _:
                return self.index_select(key)

    def __setitem__(self, key: str, value: Any) -> None:
        """Set an attribute, routing to the correct group."""
        self._storage[key] = value

    def __contains__(self, key: str) -> bool:
        return key in self._storage

    def __len__(self) -> int:
        return self.num_graphs

    def __iter__(self) -> Iterator[tuple[str, Any]]:
        yield from self._storage.items()

    def __repr__(self) -> str:
        return (
            f"Batch(num_graphs={self.num_graphs}, "
            f"num_nodes={self.num_nodes}, "
            f"num_edges={self.num_edges}, "
            f"device={self.device})"
        )

    def _get_attr(self, key: str) -> Tensor:
        """Look up *key* across all groups."""
        for group in self._storage.groups.values():
            if key in group:
                return group[key]
        raise KeyError(f"Attribute '{key}' not found in batch")

    def __getattr__(self, name: str) -> Any:
        """Delegate unknown attribute access to the storage groups."""
        if name.startswith("_") or name in {"device", "keys"}:
            raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")
        try:
            return self._get_attr(name)
        except KeyError:
            raise AttributeError(
                f"'{type(self).__name__}' has no attribute '{name}'"
            ) from None

    def __delitem__(self, key: str) -> None:
        """Delete an attribute from the underlying storage."""
        del self._storage[key]

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def _validate_custom_append(self, other: Batch) -> None:
        """Validate custom append compatibility before any mutation."""
        schema = self._storage.attr_map
        other_schema = other._storage.attr_map

        custom_names = tuple(
            name for name in schema.level_names if name not in _BUILTIN_LEVELS
        )
        other_custom_names = tuple(
            name for name in other_schema.level_names if name not in _BUILTIN_LEVELS
        )
        if schema.level_names != other_schema.level_names:
            raise ValueError(
                "Custom append requires identical schema level order: "
                f"{schema.level_names} vs {other_schema.level_names}"
            )
        if custom_names != other_custom_names:
            raise ValueError(
                "Custom append requires identical custom level definitions: "
                f"{custom_names} vs {other_custom_names}"
            )

        def _custom_resolution(
            level_schema: LevelSchema,
        ) -> dict[str, tuple[str, torch.dtype | str | None]]:
            return {
                key: (group, _canonical_schema_dtype(level_schema, key))
                for key, group in level_schema.attr_to_group.items()
                if group not in _BUILTIN_LEVELS
            }

        resolution = _custom_resolution(schema)
        other_resolution = _custom_resolution(other_schema)
        if resolution != other_resolution:
            raise ValueError(
                "Custom schema has incompatible resolved groups, dtypes, or field "
                "sets: "
                f"{resolution} vs {other_resolution}"
            )

        self_materialized = {
            name for name in self._storage.groups if name not in _BUILTIN_LEVELS
        }
        other_materialized = {
            name for name in other._storage.groups if name not in _BUILTIN_LEVELS
        }
        if self_materialized != other_materialized:
            raise ValueError(
                "Custom append requires identical materialized level sets: "
                f"{self_materialized} vs {other_materialized}"
            )

        for name in custom_names:
            definition = (
                schema.level_kinds.get(name),
                schema.product_parents.get(name),
            )
            other_definition = (
                other_schema.level_kinds.get(name),
                other_schema.product_parents.get(name),
            )
            if definition != other_definition:
                raise ValueError(
                    f"Custom level '{name}' has incompatible definitions: "
                    f"{definition} vs {other_definition}"
                )

            if name not in self_materialized:
                continue

            group = self._storage.groups[name]
            other_group = other._storage.groups[name]
            if type(group) is not type(other_group):
                raise ValueError(
                    f"Custom level '{name}' has incompatible storage kinds: "
                    f"{type(group).__name__} vs {type(other_group).__name__}"
                )
            fields = set(group.keys())
            other_fields = set(other_group.keys())
            if fields != other_fields:
                raise ValueError(
                    f"Custom level '{name}' has incompatible field sets: "
                    f"{fields} vs {other_fields}"
                )

            for field in fields:
                value = group[field]
                other_value = other_group[field]
                if value.dtype != other_value.dtype:
                    raise ValueError(
                        f"Level '{name}' field '{field}' has incompatible buffer "
                        f"dtypes: {value.dtype} vs {other_value.dtype}"
                    )
                if value.shape[1:] != other_value.shape[1:]:
                    raise ValueError(
                        f"Custom level '{name}' field '{field}' has incompatible "
                        f"trailing shapes: {value.shape[1:]} vs {other_value.shape[1:]}"
                    )

    def _validate_custom_put(self, other: Batch) -> None:
        """Validate custom schema and storage layout before a buffer put."""
        schema = self._storage.attr_map
        other_schema = other._storage.attr_map
        custom_names = tuple(
            name for name in schema.level_names if name not in _BUILTIN_LEVELS
        )
        other_custom_names = tuple(
            name for name in other_schema.level_names if name not in _BUILTIN_LEVELS
        )
        if custom_names != other_custom_names:
            raise ValueError(
                "Custom put requires identical schema level order: "
                f"{custom_names} vs {other_custom_names}"
            )

        custom_resolution = {
            key: (group, _canonical_schema_dtype(schema, key))
            for key, group in schema.attr_to_group.items()
            if group not in _BUILTIN_LEVELS
        }
        other_resolution = {
            key: (group, _canonical_schema_dtype(other_schema, key))
            for key, group in other_schema.attr_to_group.items()
            if group not in _BUILTIN_LEVELS
        }
        if custom_resolution != other_resolution:
            raise ValueError(
                "Custom put requires identical resolved groups, dtypes, or fields"
            )

        self_materialized = {
            name for name in self._storage.groups if name not in _BUILTIN_LEVELS
        }
        other_materialized = {
            name for name in other._storage.groups if name not in _BUILTIN_LEVELS
        }
        if self_materialized != other_materialized:
            raise ValueError(
                "Custom put requires identical materialized level sets: "
                f"{self_materialized} vs {other_materialized}"
            )

        for name in custom_names:
            definition = (
                schema.level_kinds.get(name),
                schema.product_parents.get(name),
            )
            other_definition = (
                other_schema.level_kinds.get(name),
                other_schema.product_parents.get(name),
            )
            if definition != other_definition:
                raise ValueError(
                    f"Custom level '{name}' has incompatible definitions: "
                    f"{definition} vs {other_definition}"
                )
            if name not in self_materialized:
                continue

            group = self._storage.groups[name]
            other_group = other._storage.groups[name]
            if type(group) is not type(other_group):
                raise ValueError(
                    f"Custom level '{name}' has incompatible storage kinds: "
                    f"{type(group).__name__} vs {type(other_group).__name__}"
                )
            fields = set(group.keys())
            other_fields = set(other_group.keys())
            if fields != other_fields:
                raise ValueError(
                    f"Custom level '{name}' has incompatible fields: "
                    f"{fields} vs {other_fields}"
                )
            for field in fields:
                value = group[field]
                other_value = other_group[field]
                if value.dtype != other_value.dtype:
                    raise ValueError(
                        f"Custom level '{name}' field '{field}' has incompatible "
                        f"dtypes: {value.dtype} vs {other_value.dtype}"
                    )
                if value.shape[1:] != other_value.shape[1:]:
                    raise ValueError(
                        f"Custom level '{name}' field '{field}' has incompatible "
                        f"trailing shapes: {value.shape[1:]} vs "
                        f"{other_value.shape[1:]}"
                    )
                kind = schema.level_kind(name)
                if kind == "uniform" and value.dtype not in _UNIFORM_BUFFER_DTYPES:
                    raise ValueError(
                        f"Custom level '{name}' field '{field}' dtype {value.dtype} "
                        "is not supported by uniform buffer kernels"
                    )
                if kind in {"segmented", "product"} and (
                    value.dtype != torch.float32 or other_value.dtype != torch.float32
                ):
                    raise ValueError(
                        f"Custom level '{name}' field '{field}' buffer payloads "
                        "must use float32"
                    )

        # Preserve the existing built-in tolerance for missing groups and
        # fields, but reject incompatible paired storage before any fit or copy
        # can partially update the destination.
        for name, group in self._storage.groups.items():
            other_group = other._storage.groups.get(name)
            if other_group is None:
                continue
            if type(group) is not type(other_group):
                raise ValueError(
                    f"Level '{name}' has incompatible storage kinds: "
                    f"{type(group).__name__} vs {type(other_group).__name__}"
                )
            for field in set(group.keys()) & set(other_group.keys()):
                value = group[field]
                other_value = other_group[field]
                if value.shape[1:] != other_value.shape[1:]:
                    raise ValueError(
                        f"Level '{name}' field '{field}' has incompatible trailing "
                        f"shapes: {value.shape[1:]} vs {other_value.shape[1:]}"
                    )

    def _prevalidate_buffer_put(self, other: Batch) -> None:
        """Validate every payload field before any batch buffer mutation."""
        for level, dest_group in self._storage.groups.items():
            source_group = other._storage.groups.get(level)
            if source_group is None:
                continue
            if (
                isinstance(dest_group, UniformLevelStorage)
                and dest_group._data.is_empty()
                and source_group._data.is_empty()
            ):
                continue
            for field in dest_group.keys():
                if field not in source_group:
                    continue
                dest_dtype = dest_group[field].dtype
                source_dtype = source_group[field].dtype
                if dest_dtype != source_dtype:
                    raise ValueError(
                        f"Level '{level}' field '{field}' has incompatible buffer "
                        f"dtypes: {dest_dtype} vs {source_dtype}"
                    )
                if dest_dtype not in _UNIFORM_BUFFER_DTYPES:
                    raise ValueError(
                        f"Level '{level}' field '{field}' dtype {dest_dtype} is not "
                        "supported by buffer kernels"
                    )

    def _prevalidate_buffer_defrag(self) -> None:
        """Validate every payload field before compacting any batch level."""
        for level, group in self._storage.groups.items():
            for field in group.keys():
                dtype = group[field].dtype
                if dtype not in _UNIFORM_BUFFER_DTYPES:
                    raise ValueError(
                        f"Level '{level}' field '{field}' dtype {dtype} is not "
                        "supported by buffer kernels"
                    )

    def append(self, other: Batch) -> None:
        """Append another batch (in-place via concatenation).

        If *other* is missing a group that this batch has (e.g. system-level
        data), this batch's tensors in that group are extended with zeros so
        that the first dimension (num graphs) stays aligned.

        Parameters
        ----------
        other : Batch
            Batch to append.  Must not share storage with *self* —
            use ``batch.append(batch.clone())`` to double a batch.
        """
        if other is self or other._storage is self._storage:
            raise ValueError(
                "Cannot append a Batch that shares storage with the "
                "receiver (would corrupt both).  Use "
                "batch.append(batch.clone()) instead."
            )

        self._validate_custom_append(other)

        # Verify every segmented pointer that the append will extend before
        # any payload or neighbor-list offset can mutate either batch.
        for group_name, group in self._storage.groups.items():
            other_group = other._storage.groups.get(group_name)
            if not (
                isinstance(group, SegmentedLevelStorage)
                and isinstance(other_group, SegmentedLevelStorage)
            ):
                continue

            group_fields = set(group.keys())
            other_fields = set(other_group.keys())
            both_fieldless = not group_fields and not other_fields
            if not both_fieldless and not group_fields.intersection(other_fields):
                # Segmented concatenate is a no-op without a shared payload.
                continue
            _checked_segment_metadata(
                torch.cat(
                    [
                        group.segment_lengths.to(
                            device=group.device, dtype=torch.int64
                        ),
                        other_group.segment_lengths.to(
                            device=group.device, dtype=torch.int64
                        ),
                    ]
                ),
                group.device,
            )

        atoms = self._atoms_group
        other_atoms = other._atoms_group
        saved_ei = None
        if atoms is not None and other_atoms is not None:
            total_nodes = atoms.num_elements()
            other_edges = other._edges_group
            if other_edges is not None and "neighbor_list" in other_edges:
                saved_ei = other_edges._data["neighbor_list"]
                other_edges._data["neighbor_list"] = saved_ei + total_nodes

        try:
            n_other = other.num_graphs
            for group_name, group in self._storage.groups.items():
                other_group = other._storage.groups.get(group_name)
                if other_group is not None:
                    if (
                        isinstance(group, SegmentedLevelStorage)
                        and isinstance(other_group, SegmentedLevelStorage)
                        and not set(group.keys())
                        and not set(other_group.keys())
                    ):
                        # Fieldless segmented levels retain per-graph
                        # cardinality even without payload tensors.
                        group._replace_segment_lengths(
                            torch.cat(
                                [
                                    group.segment_lengths,
                                    other_group.segment_lengths.to(group.device),
                                ]
                            )
                        )
                        continue
                    group.concatenate(other_group)
                else:
                    group.extend_for_appended_graphs(n_other)
        finally:
            # Restore other's neighbor_list to avoid mutating the input batch,
            # including when a non-custom legacy append fails mid-operation.
            if saved_ei is not None:
                other_edges._data["neighbor_list"] = saved_ei

    def append_data(
        self,
        data_list: list[AtomicData],
        exclude_keys: list[str] | None = None,
    ) -> None:
        """Append individual :class:`AtomicData` objects to this batch.

        Parameters
        ----------
        data_list : list[AtomicData]
            Data objects to append.
        exclude_keys : list[str], optional
            Keys to exclude.

        Raises
        ------
        ValueError
            If *data_list* is empty.
        """
        if not data_list:
            raise ValueError("No data provided to append.")
        other = Batch.from_data_list(
            data_list,
            device=self.device,
            attr_map=self._storage.attr_map,
            exclude_keys=exclude_keys,
        )
        self.append(other)

    def add_key(
        self,
        key: str,
        values: list[Tensor],
        level: str = "node",
        overwrite: bool = False,
    ) -> None:
        """Add a new key-value pair to the batch.

        Registered custom level names are accepted in addition to the
        built-in aliases. An unrecognized level retains the legacy behavior
        of assigning the key to the atom level.

        Parameters
        ----------
        key : str
            Name of the new attribute.
        values : list[Tensor]
            One value per graph.
        level : str
            Built-in alias or registered custom level name.
        overwrite : bool
            If ``True``, overwrite existing keys.

        Raises
        ------
        ValueError
            If key exists and *overwrite* is ``False``, or if the number
            of values does not match the batch size, shape, or level
            cardinality.
        TypeError
            If *level* is not a string or a value is not a tensor.
        """
        if key in self._storage and not overwrite:
            raise ValueError(
                f"Key '{key}' already exists in batch. "
                "Set overwrite=True to replace existing values."
            )
        if len(values) != self.num_graphs:
            raise ValueError(
                f"Number of values ({len(values)}) must match "
                f"number of graphs in batch ({self.num_graphs})"
            )
        if not values:
            raise ValueError("Values must be non-empty")

        device = self.device
        if not isinstance(level, str):
            raise TypeError(f"level must be a string, got {type(level).__name__}")
        group_name = _LEVEL_ALIASES.get(level, level)
        if group_name not in self._storage.attr_map.level_kinds:
            # Preserve the historical fallback for unknown levels.
            group_name = "atoms"

        existing_group_name = self._storage._group_name_from_attr(key)
        if existing_group_name is not None and existing_group_name != group_name:
            raise ValueError(
                f"Key '{key}' already belongs to level '{existing_group_name}', "
                f"not '{group_name}'"
            )

        schema = self._storage.attr_map.clone()
        kind = schema.level_kind(group_name)
        values = [v.to(device) if isinstance(v, Tensor) else v for v in values]

        def _validate_value(value: Any) -> Tensor:
            if not isinstance(value, Tensor):
                raise TypeError(
                    f"Values for key '{key}' must be tensors, got "
                    f"{type(value).__name__}"
                )
            if value.ndim == 0 and kind != "uniform":
                raise ValueError(f"Values for key '{key}' need a leading dimension")
            return value

        values = [_validate_value(value) for value in values]
        if group_name not in _BUILTIN_LEVELS:
            first_dtype = values[0].dtype
            if any(value.dtype != first_dtype for value in values[1:]):
                raise ValueError(
                    f"Custom field '{key}' in level '{group_name}' has "
                    f"incompatible dtypes: "
                    f"{[value.dtype for value in values]}"
                )
            declared_dtype = schema.dtypes.get(key)
            if declared_dtype is not None:
                try:
                    expected_dtype = TORCH_DTYPE_MAP[declared_dtype]
                except KeyError as exc:
                    raise ValueError(
                        f"Custom field '{key}' in level '{group_name}' has "
                        f"unsupported declared dtype '{declared_dtype}'"
                    ) from exc
                if expected_dtype != first_dtype:
                    raise ValueError(
                        f"Custom field '{key}' in level '{group_name}' has dtype "
                        f"{first_dtype}, expected declared dtype {declared_dtype}"
                    )
            trailing_start = 2 if kind == "product" else 1
            first_shape = values[0].shape[trailing_start:]
            if any(value.shape[trailing_start:] != first_shape for value in values[1:]):
                raise ValueError(
                    f"Custom field '{key}' in level '{group_name}' has "
                    f"incompatible trailing shapes: "
                    f"{[tuple(value.shape[trailing_start:]) for value in values]}"
                )

        schema.set(
            key,
            group_name,
            dtype=(
                values[0].dtype
                if group_name in _BUILTIN_LEVELS or key not in schema.dtypes
                else None
            ),
            is_segmented=kind != "uniform",
        )
        group = self._storage.groups.get(group_name)
        if group is None and group_name in _BUILTIN_LEVELS:
            raise ValueError(f"Group '{group_name}' not found in batch")

        if kind == "uniform":
            if any(value.ndim > 0 and value.shape[0] != 1 for value in values):
                raise ValueError(
                    f"Uniform level '{group_name}' field '{key}' must have one "
                    "row per graph"
                )
            # squeeze (1, *trailing) per-graph to (num_graphs, *trailing)
            squeezed = [
                v.squeeze(0) if v.dim() >= 1 and v.shape[0] == 1 else v for v in values
            ]
            new_data = torch.stack(squeezed, dim=0)
            if group is None:
                group = UniformLevelStorage(
                    data={key: new_data},
                    device=device,
                    validate=False,
                    attr_map=schema,
                )
                self._storage.groups[group_name] = group
            else:
                group._data[key] = new_data
        else:
            parents_to_materialize: list[tuple[str, list[int]]] = []
            parent_counts: list[list[int] | None] = []
            if kind == "product":
                if any(value.ndim < 2 for value in values):
                    raise ValueError(
                        f"Product level '{group_name}' values must have rank >= 2 "
                        "with shape [left, right, ...]"
                    )
                parent_names = schema.product_parents[group_name]

                def _parent_cardinalities(name: str) -> list[int] | None:
                    parent_group = self._storage.groups.get(name)
                    if parent_group is not None:
                        if not isinstance(parent_group, SegmentedLevelStorage):
                            raise ValueError(
                                f"Product parent '{name}' must use segmented storage"
                            )
                        return parent_group.segment_lengths[: self.num_graphs].tolist()
                    if name == "atoms" and self._atoms_group is not None:
                        return self.num_nodes_list
                    if name == "edges" and self._edges_group is not None:
                        return self.num_edges_list
                    return None

                parent_counts = [
                    _parent_cardinalities(parent) for parent in parent_names
                ]

            if group is None:
                if kind == "product":
                    left_counts = [int(value.shape[0]) for value in values]
                    right_counts = [int(value.shape[1]) for value in values]
                else:
                    expected = [int(value.shape[0]) for value in values]
                group = None
            else:
                if not isinstance(group, SegmentedLevelStorage):
                    raise ValueError(
                        f"Level '{group_name}' is segmented but storage is not"
                    )
                expected = group.segment_lengths[: self.num_graphs].tolist()
                if kind == "product":
                    left_counts = [int(value.shape[0]) for value in values]
                    right_counts = [int(value.shape[1]) for value in values]

            if kind == "product":
                if parent_names[0] == parent_names[1] and left_counts != right_counts:
                    raise ValueError(
                        f"Self-product level '{group_name}' requires equal left and "
                        f"right cardinalities, got {left_counts} and {right_counts}"
                    )
                for axis, counts in enumerate((left_counts, right_counts)):
                    known_counts = parent_counts[axis]
                    if (
                        known_counts is not None
                        and [int(c) for c in known_counts] != counts
                    ):
                        raise ValueError(
                            f"Product level '{group_name}' axis {axis} cardinalities "
                            f"{counts} do not match parent '{parent_names[axis]}' "
                            f"cardinalities {known_counts}"
                        )
                    parent_counts[axis] = counts
                expected = _checked_product_lengths(
                    left_counts, right_counts, group_name
                )
                if group is not None and expected != [
                    int(count) for count in group.segment_lengths[: self.num_graphs]
                ]:
                    raise ValueError(
                        f"Product level '{group_name}' cardinalities {expected} "
                        f"do not match existing storage"
                    )
                payload_shape = values[0].shape[2:]
                for value in values:
                    if value.shape[2:] != payload_shape:
                        raise ValueError(
                            f"Product level '{group_name}' field '{key}' has "
                            f"trailing shape {tuple(value.shape[2:])}, expected "
                            f"{tuple(payload_shape)}"
                        )
                if group is None:
                    for parent, counts in zip(parent_names, parent_counts, strict=True):
                        if parent not in self._storage.groups:
                            if not any(
                                existing_parent == parent
                                for existing_parent, _ in parents_to_materialize
                            ):
                                parents_to_materialize.append((parent, counts))
                concatenated = torch.cat(
                    [
                        value.reshape(value.shape[0] * value.shape[1], *value.shape[2:])
                        for value in values
                    ],
                    dim=0,
                )
            else:
                actual = [int(value.shape[0]) for value in values]
                if actual != expected:
                    raise ValueError(
                        f"Segmented level '{group_name}' field '{key}' has "
                        f"cardinalities {actual}, expected {expected}"
                    )
                trailing = values[0].shape[1:]
                if any(value.shape[1:] != trailing for value in values):
                    raise ValueError(f"Field '{key}' has incompatible trailing shapes")
                concatenated = torch.cat(values, dim=0)
            if values:
                if group is None:
                    for parent, counts in parents_to_materialize:
                        self._storage.groups[parent] = SegmentedLevelStorage(
                            data=None,
                            device=device,
                            segment_lengths=counts,
                            validate=False,
                            attr_map=schema,
                        )
                    group = SegmentedLevelStorage(
                        data={key: concatenated},
                        device=device,
                        segment_lengths=expected,
                        validate=False,
                        attr_map=schema,
                    )
                    self._storage.groups[group_name] = group
                else:
                    group._data[key] = concatenated

        self._storage.attr_map = schema
        groups = {
            name: self._storage.groups[name]
            for name in schema.level_names
            if name in self._storage.groups
        }
        groups.update(
            {
                name: group
                for name, group in self._storage.groups.items()
                if name not in groups
            }
        )
        self._storage.groups = groups
        for storage_group in self._storage.groups.values():
            storage_group.attr_map = schema

        if self.keys is not None:
            legacy_level = {
                "atoms": "node",
                "edges": "edge",
                "system": "system",
            }.get(group_name)
            if legacy_level is not None:
                self.keys[legacy_level].add(key)

    # ------------------------------------------------------------------
    # DataMixin overrides (performance-critical)
    # ------------------------------------------------------------------

    def to(
        self,
        device: torch.device | str,
        dtype: torch.dtype | None = None,
        non_blocking: bool = False,
    ) -> Batch:
        """Move all tensors to *device*.

        Overrides :meth:`DataMixin.to` for performance: delegates to
        :meth:`MultiLevelStorage.to_device` instead of the
        ``model_dump`` / ``map_structure`` / ``model_validate`` round-trip.

        Parameters
        ----------
        device : torch.device | str
            Target device. A bare ``"cuda"`` is recorded as the CUDA device
            current at the time of the move, which is where the tensors land.
        dtype : torch.dtype, optional
            Ignored (present for API compatibility).
        non_blocking : bool
            Whether tensor copies may be asynchronous when supported.

        Returns
        -------
        Batch
        """
        new = self.clone()
        new._storage.to_device(device, non_blocking=non_blocking)
        new.device = new._storage.device
        return new

    def clone(self) -> Batch:
        """Return a deep copy.

        Overrides :meth:`DataMixin.clone` for performance.

        Returns
        -------
        Batch
        """
        return Batch._construct(
            device=self.device,
            keys={k: v.copy() for k, v in self.keys.items()} if self.keys else None,
            storage=self._storage.clone(),
            data_class=self._data_class,
        )

    def cpu(self) -> Batch:
        """Return a copy on CPU."""
        return self.to("cpu")

    def cuda(self, device: int | None = None, non_blocking: bool = False) -> Batch:
        """Return a copy on CUDA."""
        dev = f"cuda:{device}" if device is not None else "cuda"
        return self.to(dev)

    def contiguous(self) -> Batch:
        """Ensure contiguous memory layout for all tensors.

        Returns
        -------
        Self
            For method chaining.
        """
        self._make_contiguous()
        return self

    def pin_memory(self) -> Batch:
        """Pin all tensors to page-locked memory.

        Returns
        -------
        Self
            For method chaining.
        """
        for group in self._storage.groups.values():
            for key, tensor in list(group.items()):
                group._data[key] = tensor.pin_memory()
        return self

    def _make_contiguous(self) -> Batch:
        """Ensure all tensors are contiguous. Returns self for chaining."""
        for group in self._storage.groups.values():
            for key, tensor in list(group.items()):
                if not tensor.is_contiguous():
                    group._data[key] = tensor.contiguous()
        return self

    # ------------------------------------------------------------------
    # Custom serialization
    # ------------------------------------------------------------------

    def model_dump(self, **kwargs: Any) -> dict[str, Any]:
        """Serialize the batch into a flat dictionary.

        Collects all tensors from the underlying :class:`MultiLevelStorage`
        groups, plus metadata fields (``device``, ``keys``, ``batch_idx``,
        ``batch_ptr``, ``num_nodes_list``, ``num_edges_list``, ``num_graphs``).

        Returns
        -------
        dict[str, Any]
        """
        result: dict[str, Any] = {
            "device": self.device,
            "keys": self.keys,
            "batch_idx": self.batch_idx,
            "batch_ptr": self.batch_ptr,
            "num_graphs": self.num_graphs,
            "num_nodes_list": self.num_nodes_list,
            "num_edges_list": self.num_edges_list,
        }
        result.update(
            {
                key: tensor
                for group in self._storage.groups.values()
                for key, tensor in group.items()
            }
        )
        exclude_none = kwargs.get("exclude_none", False)
        if exclude_none:
            result = {k: v for k, v in result.items() if v is not None}
        return result

    # ------------------------------------------------------------------
    # Distributed communication
    # ------------------------------------------------------------------

    def isend(
        self,
        dst: int,
        *,
        tag: int = 0,
        group: ProcessGroup | None = None,
    ) -> _BatchSendHandle:
        """Non-blocking send of this batch to *dst*.

        Transmits the fixed metadata header and built-in level data first.
        Materialized custom level segment lengths and tensor payloads follow
        in schema order. A zero-graph batch sends only the metadata header.
        The wire format does not negotiate custom layouts; the receiver must
        provide a matching custom template as a caller precondition.

        Parameters
        ----------
        dst : int
            Destination rank.
        tag : int
            Base message tag.  Incremented deterministically per group.
        group : ProcessGroup, optional
            Process group.  ``None`` uses the default group.

        Returns
        -------
        _BatchSendHandle
            Handle whose ``.wait()`` blocks until all sends complete.
        """
        handles: list[Work | list[Work] | int | None] = []

        meta = torch.tensor(
            [self.num_graphs, self.num_nodes, self.num_edges],
            dtype=torch.int64,
            device=self.device,
        )
        handles.append(dist.isend(meta, dst=dst, tag=tag, group=group))
        tag_offset = 1

        if self.num_graphs == 0:
            return _BatchSendHandle(handles)

        for name in ("atoms", "edges"):
            grp = self._storage.groups.get(name)
            if grp is not None and isinstance(grp, SegmentedLevelStorage):
                seg_len = grp.segment_lengths[: self.num_graphs].contiguous()
                handles.append(
                    dist.isend(seg_len, dst=dst, tag=tag + tag_offset, group=group)
                )
            tag_offset += 1

        for name in ("atoms", "edges", "system"):
            grp = self._storage.groups.get(name)
            if grp is None:
                tag_offset += _transport_payload_tag_span(grp)
                continue
            if isinstance(grp, SegmentedLevelStorage):
                n = grp.num_elements()
            else:
                n = self.num_graphs
            occupied_td = grp._data[:n]
            result = occupied_td.isend(
                dst=dst,
                init_tag=tag + tag_offset,
                group=group,
                return_early=True,
            )
            if isinstance(result, list):
                handles.extend(result)
            else:
                handles.append(result)
            tag_offset += _transport_payload_tag_span(grp)

        custom_groups = _custom_transport_groups(self)
        if custom_groups:
            for _, custom_group in custom_groups:
                if isinstance(custom_group, SegmentedLevelStorage):
                    segment_lengths = custom_group.segment_lengths[
                        : self.num_graphs
                    ].contiguous()
                    handles.append(
                        dist.isend(
                            segment_lengths,
                            dst=dst,
                            tag=tag + tag_offset,
                            group=group,
                        )
                    )
                    tag_offset += 1

            for _, custom_group in custom_groups:
                keys = list(custom_group.keys())
                if not keys:
                    tag_offset += _transport_payload_tag_span(custom_group)
                    continue
                n = (
                    custom_group.num_elements()
                    if isinstance(custom_group, SegmentedLevelStorage)
                    else self.num_graphs
                )
                result = custom_group._data[:n].isend(
                    dst=dst,
                    init_tag=tag + tag_offset,
                    group=group,
                    return_early=True,
                )
                if isinstance(result, list):
                    handles.extend(result)
                else:
                    handles.append(result)
                tag_offset += _transport_payload_tag_span(custom_group)

        return _BatchSendHandle(handles)

    @classmethod
    def irecv(
        cls,
        src: int,
        device: torch.device | str,
        *,
        template: Batch | None = None,
        tag: int = 0,
        group: ProcessGroup | None = None,
    ) -> _BatchRecvHandle:
        """Non-blocking receive of a batch from *src*.

        Posts non-blocking receives for the metadata header, then returns
        a :class:`_BatchRecvHandle` whose ``.wait()`` blocks until all
        data arrives and reconstructs a :class:`Batch` using *template*.

        Parameters
        ----------
        src : int
            Source rank.
        device : torch.device | str
            Device to receive tensors onto.
        template : Batch, optional
            Template batch providing attribute keys, dtypes, group structure,
            and custom level definitions. Required when receiving custom
            levels and for the first structured receive. Its materialized
            levels, field order, dtypes, and trailing shapes must match the
            sender; callers may cache it for subsequent calls. No custom
            layout negotiation occurs on the wire.
        tag : int
            Base message tag.
        group : ProcessGroup, optional
            Process group.

        Returns
        -------
        _BatchRecvHandle
            Handle whose ``.wait()`` returns the received :class:`Batch`.
        """
        device = _resolve_device(device)

        meta = torch.empty(3, dtype=torch.int64, device=device)
        meta_handle = dist.irecv(meta, src=src, tag=tag, group=group)

        return _BatchRecvHandle(
            meta=meta,
            meta_handle=meta_handle,
            src=src,
            device=device,
            template=template,
            base_tag=tag,
            group=group,
        )

    def send(
        self,
        dst: int,
        *,
        tag: int = 0,
        group: ProcessGroup | None = None,
    ) -> None:
        """Blocking send to *dst*.

        Equivalent to ``self.isend(dst, tag=tag, group=group).wait()``.
        When custom levels are present, the receiver must supply a matching
        template; the wire format performs no custom layout negotiation.

        Parameters
        ----------
        dst : int
            Destination rank.
        tag : int
            Base message tag.
        group : ProcessGroup, optional
            Process group.
        """
        self.isend(dst=dst, tag=tag, group=group).wait()

    @classmethod
    def recv(
        cls,
        src: int,
        device: torch.device | str,
        *,
        template: Batch | None = None,
        tag: int = 0,
        group: ProcessGroup | None = None,
    ) -> Batch:
        """Blocking receive from *src*.

        Equivalent to ``cls.irecv(src, device, ...).wait()``.

        Parameters
        ----------
        src : int
            Source rank.
        device : torch.device | str
            Device to receive tensors onto.
        template : Batch, optional
            Template batch providing attribute keys, dtypes, group structure,
            and custom level definitions. Required when receiving custom
            levels and must match the sender's materialized layout. This
            matching template is a caller precondition; no custom layout
            negotiation occurs on the wire.
        tag : int
            Base message tag.
        group : ProcessGroup, optional
            Process group.

        Returns
        -------
        Batch

        """
        return cls.irecv(
            src=src,
            device=device,
            template=template,
            tag=tag,
            group=group,
        ).wait()

    @classmethod
    def empty_like(
        cls,
        batch: Batch,
        *,
        device: torch.device | str | None = None,
    ) -> Batch:
        """Create an empty batch with the schema and capacities of *batch*.

        Parameters
        ----------
        batch : Batch
            Template batch for attribute keys and dtypes.
        device : torch.device | str, optional
            Device for the new batch.  Defaults to ``batch.device``.

        Returns
        -------
        Batch
            A zero-graph batch preserving every materialized level and its
            graph-slot or payload capacity.
        """
        dev = device if device is not None else batch.device
        level_capacities: dict[str, int] = {}
        for name, group in batch._storage.groups.items():
            if name in _BUILTIN_LEVELS:
                continue
            if isinstance(group, SegmentedLevelStorage) and list(group.keys()):
                level_capacities[name] = group._data.shape[0]

        atoms = batch._storage.groups.get("atoms")
        edges = batch._storage.groups.get("edges")
        num_nodes = atoms._data.shape[0] if atoms is not None else 0
        num_edges = edges._data.shape[0] if edges is not None else 0
        return cls.empty(
            num_systems=_batch_graph_slot_capacity(batch),
            num_nodes=num_nodes,
            num_edges=num_edges,
            template=batch,
            device=dev,
            attr_map=batch._storage.attr_map,
            level_capacities=level_capacities,
        )


# ======================================================================
# Distributed communication handle classes
# ======================================================================


class _BatchSendHandle:
    """Aggregates multiple async distributed send handles.

    Calling ``.wait()`` blocks until all underlying sends have completed.

    Parameters
    ----------
    handles : list
        A list of ``torch.distributed.Work`` objects (or ``int`` /
        ``None`` values which are silently skipped).
    """

    def __init__(self, handles: list) -> None:
        self._handles = handles

    def wait(self) -> None:
        """Block until all sends complete."""
        for h in self._handles:
            if h is not None and hasattr(h, "wait"):
                h.wait()


class _BatchRecvHandle:
    """Deferred receive that reconstructs a :class:`Batch` on ``.wait()``.

    Created by :meth:`Batch.irecv`.  The metadata header receive is
    already posted; ``.wait()`` blocks on it, then posts and completes
    the segment-length and bulk-data receives.

    Parameters
    ----------
    meta : Tensor
        Pre-allocated ``(3,)`` int64 tensor for the metadata header.
    meta_handle : Work
        Async receive handle for *meta*.
    src : int
        Source rank.
    device : torch.device
        Device to receive tensors onto.
    template : Batch | None
        Template batch for attribute keys, dtypes, group structure, and custom
        level definitions.
    base_tag : int
        Base message tag (must match sender's *tag*).
    group : ProcessGroup | None
        Process group.
    """

    def __init__(
        self,
        *,
        meta: Tensor,
        meta_handle: Work,
        src: int,
        device: torch.device,
        template: Batch | None,
        base_tag: int,
        group: ProcessGroup | None,
    ) -> None:
        self._meta = meta
        self._meta_handle = meta_handle
        self._src = src
        self._device = device
        self._template = template
        self._base_tag = base_tag
        self._group = group

    def wait(self) -> Batch:
        """Block until all data arrives and return the received :class:`Batch`.

        Returns
        -------
        Batch
            The reconstructed batch.  If the sender sent a sentinel
            (0-graph batch), returns ``Batch.empty(...)`` with 0 capacity.

        """
        self._meta_handle.wait()
        num_graphs, num_nodes, num_edges = self._meta.tolist()
        num_graphs = int(num_graphs)
        num_nodes = int(num_nodes)
        num_edges = int(num_edges)

        tag_offset = 1

        if num_graphs == 0:
            if self._template is not None:
                return Batch.empty(
                    num_systems=0,
                    num_nodes=0,
                    num_edges=0,
                    template=self._template,
                    device=self._device,
                    attr_map=self._template._storage.attr_map,
                    level_capacities={
                        name: 0
                        for name, custom_group in _custom_transport_groups(
                            self._template
                        )
                        if isinstance(custom_group, SegmentedLevelStorage)
                    },
                )
            return Batch(device=self._device)

        handles: list = []

        atoms_seg: Tensor | None = None
        edges_seg: Tensor | None = None

        if self._template is not None:
            atoms_grp = self._template._storage.groups.get("atoms")
            if atoms_grp is not None and isinstance(atoms_grp, SegmentedLevelStorage):
                atoms_seg = torch.empty(
                    num_graphs, dtype=torch.int32, device=self._device
                )
                handles.append(
                    dist.irecv(
                        atoms_seg,
                        src=self._src,
                        tag=self._base_tag + tag_offset,
                        group=self._group,
                    )
                )
        tag_offset += 1

        if self._template is not None:
            edges_grp = self._template._storage.groups.get("edges")
            if edges_grp is not None and isinstance(edges_grp, SegmentedLevelStorage):
                edges_seg = torch.empty(
                    num_graphs, dtype=torch.int32, device=self._device
                )
                handles.append(
                    dist.irecv(
                        edges_seg,
                        src=self._src,
                        tag=self._base_tag + tag_offset,
                        group=self._group,
                    )
                )
        tag_offset += 1

        groups: dict[str, UniformLevelStorage | SegmentedLevelStorage] = {}
        attr_map = (
            self._template._storage.attr_map.clone()
            if self._template is not None
            else LevelSchema()
        )
        builtin_specs: list[
            tuple[
                str,
                TensorDict | None,
                Tensor | None,
            ]
        ] = []

        for name, capacity, seg_lens in [
            ("atoms", num_nodes, atoms_seg),
            ("edges", num_edges, edges_seg),
            ("system", num_graphs, None),
        ]:
            template_grp = (
                self._template._storage.groups.get(name)
                if self._template is not None
                else None
            )
            if template_grp is None:
                tag_offset += _transport_payload_tag_span(template_grp)
                continue

            keys = list(template_grp.keys())
            if not keys:
                tag_offset += _transport_payload_tag_span(template_grp)
                builtin_specs.append((name, None, seg_lens))
                continue

            recv_data = {}
            for k in keys:
                ref_tensor = template_grp[k]
                trailing_shape = ref_tensor.shape[1:]
                recv_data[k] = torch.empty(
                    (capacity,) + trailing_shape,
                    dtype=ref_tensor.dtype,
                    device=self._device,
                )

            recv_td = TensorDict(recv_data, batch_size=[capacity], device=self._device)
            td_handles = recv_td.irecv(
                src=self._src,
                init_tag=self._base_tag + tag_offset,
                group=self._group,
                return_premature=True,
            )
            if isinstance(td_handles, list):
                handles.extend(td_handles)
            else:
                handles.append(td_handles)
            tag_offset += _transport_payload_tag_span(template_grp)
            builtin_specs.append((name, recv_td, seg_lens))

        for h in handles:
            if h is not None and hasattr(h, "wait"):
                h.wait()

        # Segment-length receives populate uninitialized tensors.  Construct
        # segmented storage only after those receives complete, including
        # fieldless built-in atoms/edges where no TensorDict I/O was needed.
        for name, recv_td, seg_lens in builtin_specs:
            if name == "system":
                if recv_td is None:
                    continue
                groups[name] = UniformLevelStorage(
                    data={key: recv_td[key] for key in recv_td.keys()},
                    device=self._device,
                    validate=False,
                    attr_map=attr_map,
                )
            elif seg_lens is not None:
                groups[name] = SegmentedLevelStorage(
                    data=(
                        {key: recv_td[key] for key in recv_td.keys()}
                        if recv_td is not None
                        else None
                    ),
                    segment_lengths=seg_lens,
                    device=self._device,
                    validate=False,
                    attr_map=attr_map,
                )

        custom_groups = (
            _custom_transport_groups(self._template)
            if self._template is not None
            else []
        )
        if custom_groups:
            control_handles: list[Work | list[Work] | int | None] = []
            custom_segment_lengths: dict[str, Tensor] = {}
            for name, custom_group in custom_groups:
                if isinstance(custom_group, SegmentedLevelStorage):
                    segment_lengths = torch.empty(
                        num_graphs, dtype=torch.int32, device=self._device
                    )
                    custom_segment_lengths[name] = segment_lengths
                    control_handles.append(
                        dist.irecv(
                            segment_lengths,
                            src=self._src,
                            tag=self._base_tag + tag_offset,
                            group=self._group,
                        )
                    )
                    tag_offset += 1

            for handle in control_handles:
                if handle is not None and hasattr(handle, "wait"):
                    handle.wait()

            custom_payload_handles: list[Work | list[Work] | int | None] = []
            for name, template_group in custom_groups:
                keys = list(template_group.keys())
                kind = attr_map.level_kind(name)
                if isinstance(template_group, SegmentedLevelStorage):
                    segment_lengths = custom_segment_lengths[name]
                    capacity = int(segment_lengths.sum().item())
                else:
                    segment_lengths = None
                    capacity = num_graphs

                if not keys:
                    tag_offset += _transport_payload_tag_span(template_group)
                    if kind == "uniform":
                        storage = UniformLevelStorage(
                            data=None,
                            device=self._device,
                            validate=False,
                            attr_map=attr_map,
                        )
                        storage._data = TensorDict(
                            {}, batch_size=[capacity], device=self._device
                        )
                    else:
                        storage = SegmentedLevelStorage(
                            data=None,
                            segment_lengths=segment_lengths,
                            device=self._device,
                            validate=False,
                            attr_map=attr_map,
                        )
                    groups[name] = storage
                    continue

                recv_data = {
                    key: torch.empty(
                        (capacity,) + template_group[key].shape[1:],
                        dtype=template_group[key].dtype,
                        device=self._device,
                    )
                    for key in keys
                }
                recv_td = TensorDict(
                    recv_data, batch_size=[capacity], device=self._device
                )
                payload_handles = recv_td.irecv(
                    src=self._src,
                    init_tag=self._base_tag + tag_offset,
                    group=self._group,
                    return_premature=True,
                )
                if isinstance(payload_handles, list):
                    custom_payload_handles.extend(payload_handles)
                else:
                    custom_payload_handles.append(payload_handles)
                tag_offset += _transport_payload_tag_span(template_group)

                if kind == "uniform":
                    groups[name] = UniformLevelStorage(
                        data={key: recv_td[key] for key in keys},
                        device=self._device,
                        validate=False,
                        attr_map=attr_map,
                    )
                else:
                    groups[name] = SegmentedLevelStorage(
                        data={key: recv_td[key] for key in keys},
                        segment_lengths=segment_lengths,
                        device=self._device,
                        validate=False,
                        attr_map=attr_map,
                    )

            for handle in custom_payload_handles:
                if handle is not None and hasattr(handle, "wait"):
                    handle.wait()

        mls = MultiLevelStorage(groups=groups, attr_map=attr_map, validate=False)
        return Batch._construct(
            device=self._device,
            keys=(
                {k: v.copy() for k, v in self._template.keys.items()}
                if self._template is not None and self._template.keys is not None
                else None
            ),
            storage=mls,
            data_class=(
                self._template._data_class if self._template is not None else AtomicData
            ),
        )
