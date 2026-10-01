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

"""Private native packing payload metadata and point-to-point transport."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Any

import torch
import torch.distributed as dist
from torch import Tensor

from nvalchemi.csp._batch import expand_asu_batch
from nvalchemi.csp.data import MolecularPackingInput, RigidMoleculeASUBatch
from nvalchemi.csp.packer.result import PackingReport, PackingResult
from nvalchemi.data.batch import Batch
from nvalchemi.data.level_storage import SegmentedLevelStorage, UniformLevelStorage
from nvalchemi.distributed import ProcessGroupContext

_RIGID_FIELDS = (
    "structure_molecule_ptr",
    "conformer_indices",
    "rotations",
    "fractional_centers",
    "cells",
    "space_groups",
    "z",
    "z_prime",
    "structure_ids",
)


def _tensor_descriptor(value: Tensor) -> tuple[str, tuple[int, ...]]:
    """Return the comparable dtype and trailing shape for one payload tensor."""
    return str(value.dtype), tuple(int(size) for size in value.shape[1:])


def _rigid_payload_fields(
    structures: RigidMoleculeASUBatch,
) -> list[tuple[tuple[str, ...], Tensor]]:
    """Return all compact fields in the stable wire order."""
    fields = [(name, (getattr(structures, name))) for name in _RIGID_FIELDS]
    fields.extend(
        (("property", name), structures.properties[name])
        for name in sorted(structures.properties)
    )
    return [
        ((name,) if isinstance(name, str) else name, value) for name, value in fields
    ]


def _batch_groups_in_transport_order(
    structures: Batch,
) -> list[tuple[str, UniformLevelStorage | SegmentedLevelStorage]]:
    """Return materialized groups in Batch's built-in then schema order."""
    ordered_names = ["atoms", "edges", "system"]
    ordered_names.extend(
        name
        for name in structures._storage.attr_map.level_names
        if name not in ordered_names
    )
    ordered_names.extend(
        name for name in structures._storage.groups if name not in ordered_names
    )
    return [
        (name, structures._storage.groups[name])
        for name in ordered_names
        if name in structures._storage.groups
    ]


def _batch_layout(structures: Batch) -> tuple[Any, ...]:
    """Return the complete comparable native Batch schema and payload layout."""
    schema = structures._storage.attr_map
    level_definitions = tuple(
        (
            name,
            schema.level_kinds[name],
            schema.product_parents.get(name),
        )
        for name in schema.level_names
    )
    schema_extras = tuple(
        (name, schema.level_kinds[name], schema.product_parents.get(name))
        for name in sorted(set(schema.level_kinds) - set(schema.level_names))
    )
    group_attributes = tuple(
        (name, tuple(sorted(schema.group_to_attrs.get(name, ()))))
        for name in schema.level_names
    ) + tuple(
        (name, tuple(sorted(attrs)))
        for name, attrs in sorted(schema.group_to_attrs.items())
        if name not in schema.level_names
    )
    attribute_groups = tuple(sorted(schema.attr_to_group.items()))
    registered_dtypes = tuple(
        (name, str(dtype) if dtype is not None else None)
        for name, dtype in sorted(schema.dtypes.items())
    )

    materialized = []
    for name, group in _batch_groups_in_transport_order(structures):
        if isinstance(group, UniformLevelStorage):
            storage_kind = "uniform"
        elif isinstance(group, SegmentedLevelStorage):
            storage_kind = "segmented"
        else:
            raise TypeError(
                f"unsupported Batch storage for level {name!r}: {type(group).__name__}"
            )
        payload_fields = tuple(
            (field, *_tensor_descriptor(group[field])) for field in group.keys()
        )
        materialized.append(
            (
                name,
                storage_kind,
                schema.level_kinds.get(name),
                payload_fields,
            )
        )

    return (
        "batch",
        level_definitions,
        schema_extras,
        group_attributes,
        attribute_groups,
        registered_dtypes,
        tuple(materialized),
    )


def _batch_counts(structures: Batch) -> dict[str, Any]:
    """Return only Batch's typed transport header values."""
    return {
        "num_graphs": int(structures.num_graphs),
        "num_nodes": int(structures.num_nodes),
        "num_edges": int(structures.num_edges),
    }


def _batch_template(structures: Batch) -> Batch:
    """Create a CPU zero-capacity Batch that preserves the local full schema."""
    level_capacities = {
        name: 0
        for name, group in _batch_groups_in_transport_order(structures)
        if name not in {"atoms", "edges", "system"}
        and isinstance(group, SegmentedLevelStorage)
        and list(group.keys())
    }
    return Batch.empty(
        num_systems=0,
        num_nodes=0,
        num_edges=0,
        template=structures,
        device="cpu",
        level_capacities=level_capacities,
    )


def _describe_native(
    structures: RigidMoleculeASUBatch | Batch,
) -> dict[str, Any]:
    """Describe the local native payload without transferring any tensor data."""
    if isinstance(structures, RigidMoleculeASUBatch):
        fields = _rigid_payload_fields(structures)
        layout = (
            "rigid",
            tuple(
                (field_name, *_tensor_descriptor(value)) for field_name, value in fields
            ),
        )
        field_rows = tuple(
            (field_name, int(value.shape[0])) for field_name, value in fields
        )
        return {
            "kind": "rigid",
            "layout": layout,
            "template": None,
            "counts": {
                "num_structures": int(structures.num_structures),
                "num_molecules": int(structures.conformer_indices.shape[0]),
                "field_rows": field_rows,
            },
        }
    if isinstance(structures, Batch):
        return {
            "kind": "batch",
            "layout": _batch_layout(structures),
            "template": _batch_template(structures),
            "counts": _batch_counts(structures),
        }
    raise TypeError("native packing structures must be RigidMoleculeASUBatch or Batch")


def _empty_batch(
    structures: RigidMoleculeASUBatch | Batch,
    *,
    device: torch.device,
) -> Batch:
    """Return an empty Batch with the native structure's observable schema."""
    target_device = torch.device(device)
    if isinstance(structures, Batch):
        return _batch_template(structures).to(target_device)
    if isinstance(structures, RigidMoleculeASUBatch):
        empty_indices = torch.empty(
            (0,), dtype=torch.int64, device=structures.cells.device
        )
        return expand_asu_batch(
            structures,
            indices=empty_indices,
            device=target_device,
        )
    raise TypeError("native packing structures must be RigidMoleculeASUBatch or Batch")


def _rigid_rows_by_field(
    metadata: Mapping[str, Any],
) -> dict[tuple[str, ...], int]:
    return {
        tuple(field_name): int(rows)
        for field_name, rows in metadata["counts"]["field_rows"]
    }


def _build_rigid_batch(
    fields: Mapping[tuple[str, ...], Tensor],
    *,
    inputs: MolecularPackingInput,
) -> RigidMoleculeASUBatch:
    """Rebuild one rank's ragged compact ASU payload from received fields."""
    properties = {
        field_name[1]: value
        for field_name, value in fields.items()
        if field_name[0] == "property"
    }
    return RigidMoleculeASUBatch._from_validated(
        packing_input=inputs,
        structure_molecule_ptr=fields[("structure_molecule_ptr",)],
        conformer_indices=fields[("conformer_indices",)],
        rotations=fields[("rotations",)],
        fractional_centers=fields[("fractional_centers",)],
        cells=fields[("cells",)],
        space_groups=fields[("space_groups",)],
        z=fields[("z",)],
        z_prime=fields[("z_prime",)],
        structure_ids=fields[("structure_ids",)],
        properties=MappingProxyType(properties),
    )


def _gather_rigid_payloads(
    structures: RigidMoleculeASUBatch,
    *,
    metadata: Sequence[dict[str, Any]],
    inputs: MolecularPackingInput,
    group_context: ProcessGroupContext,
    destination: int,
) -> list[RigidMoleculeASUBatch] | None:
    """Send or receive compact tensor fields in deterministic rank/field order."""
    group_size = group_context.world_size
    group_rank = group_context.rank
    destination_global = group_context.global_rank(destination)
    local_fields = _rigid_payload_fields(structures)
    local_values = dict(local_fields)

    if group_rank != destination:
        handles = []
        for field_index, (field_name, _) in enumerate(local_fields):
            value = local_values[field_name]
            if value.numel() == 0:
                continue
            handle = dist.isend(
                value,
                dst=destination_global,
                tag=field_index,
                group=group_context.process_group,
            )
            if handle is not None:
                handles.append(handle)
        for handle in handles:
            handle.wait()
        return None

    pieces: list[RigidMoleculeASUBatch | None] = [None] * group_size
    pieces[group_rank] = structures
    received_by_rank: dict[int, dict[tuple[str, ...], Tensor]] = {}
    handles = []
    for source_rank in range(group_size):
        if source_rank == destination:
            continue
        source_metadata = metadata[source_rank]
        row_counts = _rigid_rows_by_field(source_metadata)
        received: dict[tuple[str, ...], Tensor] = {}
        for field_index, (field_name, reference) in enumerate(local_fields):
            row_count = row_counts[field_name]
            value = torch.empty(
                (row_count, *reference.shape[1:]),
                dtype=reference.dtype,
                device=group_context.collective_device,
            )
            received[field_name] = value
            if value.numel() == 0:
                continue
            handle = dist.irecv(
                value,
                src=group_context.global_rank(source_rank),
                tag=field_index,
                group=group_context.process_group,
            )
            if handle is not None:
                handles.append(handle)
        received_by_rank[source_rank] = received
    for handle in handles:
        handle.wait()
    for source_rank in range(group_size):
        if source_rank != destination:
            pieces[source_rank] = _build_rigid_batch(
                received_by_rank[source_rank], inputs=inputs
            )
    if any(piece is None for piece in pieces):
        raise RuntimeError("compact packing transfer did not receive every rank")
    return [piece for piece in pieces if piece is not None]


def _gather_batch_payloads(
    structures: Batch,
    *,
    metadata: Sequence[dict[str, Any]],
    group_context: ProcessGroupContext,
    destination: int,
) -> list[Batch] | None:
    """Use Batch's typed transport with its agreed, schema-preserving template."""
    group_size = group_context.world_size
    group_rank = group_context.rank
    destination_global = group_context.global_rank(destination)
    if group_rank != destination:
        handle = structures.isend(
            dst=destination_global,
            group=group_context.process_group,
        )
        handle.wait()
        return None

    pieces: list[Batch | None] = [None] * group_size
    pieces[group_rank] = structures
    for source_rank in range(group_size):
        if source_rank == destination:
            continue
        template = metadata[source_rank]["template"].to(group_context.collective_device)
        receive = Batch.irecv(
            src=group_context.global_rank(source_rank),
            device=group_context.collective_device,
            template=template,
            group=group_context.process_group,
        )
        pieces[source_rank] = receive.wait()
    if any(piece is None for piece in pieces):
        raise RuntimeError("Batch packing transfer did not receive every rank")
    return [piece for piece in pieces if piece is not None]


def _gather_payloads(
    local_result: PackingResult,
    *,
    metadata: Sequence[dict[str, Any]],
    inputs: MolecularPackingInput,
    group_context: ProcessGroupContext,
    destination: int,
) -> list[RigidMoleculeASUBatch | Batch] | None:
    """Transfer rank-local native tensors to one group-local destination."""
    group_size = group_context.world_size
    if len(metadata) != group_size:
        raise ValueError("native metadata must contain one entry per process rank")
    if isinstance(destination, bool) or not 0 <= destination < group_size:
        raise ValueError("destination must be a valid group-local rank")

    local_structures = local_result.structures
    kind = metadata[0]["kind"]
    if any(
        item["kind"] != kind or item["layout"] != metadata[0]["layout"]
        for item in metadata
    ):
        raise ValueError("native payload metadata differs across process ranks")
    with group_context.communication_scope():
        if kind == "rigid" and isinstance(local_structures, RigidMoleculeASUBatch):
            return _gather_rigid_payloads(
                local_structures,
                metadata=metadata,
                inputs=inputs,
                group_context=group_context,
                destination=destination,
            )
        if kind == "batch" and isinstance(local_structures, Batch):
            return _gather_batch_payloads(
                local_structures,
                metadata=metadata,
                group_context=group_context,
                destination=destination,
            )
        raise TypeError("local native payload kind does not match gathered metadata")


def _assemble_result(
    parts: Sequence[RigidMoleculeASUBatch | Batch],
    *,
    run_id: int,
    reports: tuple[PackingReport, ...],
) -> PackingResult:
    """Assemble rank-major native pieces into one gathered result."""
    if not parts:
        raise ValueError("at least one native rank payload is required")
    first = parts[0]
    if isinstance(first, RigidMoleculeASUBatch):
        if any(not isinstance(part, RigidMoleculeASUBatch) for part in parts):
            raise TypeError("native result parts must all have the same payload type")
        structures = RigidMoleculeASUBatch.concatenate(parts)
    elif isinstance(first, Batch):
        if any(not isinstance(part, Batch) for part in parts):
            raise TypeError("native result parts must all have the same payload type")
        expected_layout = _batch_layout(first)
        if any(_batch_layout(part) != expected_layout for part in parts[1:]):
            raise ValueError("Batch result parts have incompatible native schemas")
        structures = first.clone()
        for part in parts[1:]:
            structures.append(part)
    else:
        raise TypeError("native result parts must be RigidMoleculeASUBatch or Batch")
    return PackingResult(
        structures=structures,
        run_id=run_id,
        reports=reports,
        scope="gathered",
    )
