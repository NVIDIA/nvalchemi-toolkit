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
"""Private flat-tensor P1 expansion from CSP rigid-molecule ASU batches."""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

import torch
from torch import Tensor

from nvalchemi.data.atomic_data import _default_mass_table
from nvalchemi.data.batch import Batch
from nvalchemi.data.level_storage import LevelSchema, MultiLevelStorage

if TYPE_CHECKING:
    from nvalchemi.csp.data import RigidMoleculeASUBatch

_INT32_MAX = 2**31 - 1
_ATOM_SOURCE_FIELDS = (
    "csp_source_asu_atom_index",
    "csp_source_molecule_index",
    "csp_source_component_index",
    "csp_source_conformer_index",
    "csp_source_symmetry_operation_index",
)
_SYSTEM_SOURCE_FIELDS = (
    "csp_source_space_group",
    "csp_source_z",
    "csp_source_z_prime",
)
_ATOM_CHUNK_SIZE = 32768


def _canonical_device(device: torch.device | str) -> torch.device:
    """Resolve an unindexed CUDA alias to its current concrete device."""
    resolved = torch.device(device)
    if resolved.type == "cuda" and resolved.index is None:
        return torch.device("cuda", torch.cuda.current_device())
    return resolved


@lru_cache(maxsize=None)
def _operation_tables_cached(device: torch.device) -> tuple[Tensor, Tensor, Tensor]:
    """Return cached symmetry operation tensors on a canonical device."""
    from nvalchemi.csp._space_group_tables import SG_OPS_IDX, SG_OPS_PTR, SYMM_OPS

    flat_operations = torch.as_tensor(SYMM_OPS, dtype=torch.float32, device=device)
    operations = torch.empty(
        (flat_operations.shape[0], 3, 4), dtype=torch.float32, device=device
    )
    operations[:, :, :3] = flat_operations[:, :9].reshape(-1, 3, 3)
    operations[:, :, 3] = flat_operations[:, 9:12]
    return (
        operations,
        torch.as_tensor(SG_OPS_IDX, dtype=torch.int32, device=device).clone(),
        torch.as_tensor(SG_OPS_PTR, dtype=torch.int32, device=device).clone(),
    )


def _operation_tables(
    device: torch.device | str,
) -> tuple[Tensor, Tensor, Tensor]:
    """Return cached symmetry operation tensors on ``device``."""
    return _operation_tables_cached(_canonical_device(device))


@lru_cache(maxsize=None)
def _mass_table_cached(device: torch.device) -> Tensor:
    """Return the Toolkit atomic-mass lookup table on a canonical device."""
    return _default_mass_table().to(device=device)


def _mass_table(device: torch.device | str) -> Tensor:
    """Return the Toolkit atomic-mass lookup table cached on ``device``."""
    return _mass_table_cached(_canonical_device(device))


def _selected_indices(compact: RigidMoleculeASUBatch, indices: Tensor | None) -> Tensor:
    """Return row indices on the ASU batch device.

    ``None`` selects every row. Supplied indices must be one-dimensional
    int32 or int64 tensors on CPU or on the ASU batch device.
    """
    if indices is None:
        return torch.arange(
            compact.num_structures, dtype=torch.int64, device=compact.cells.device
        )
    if not isinstance(indices, Tensor) or indices.ndim != 1:
        raise TypeError("indices must be a one-dimensional torch.Tensor")
    if indices.dtype not in (torch.int32, torch.int64):
        raise TypeError("indices must have dtype torch.int32 or torch.int64")
    if indices.device.type != "cpu" and indices.device != compact.cells.device:
        raise ValueError("indices must be on CPU or on the compact batch device")
    return indices.to(device=compact.cells.device)


def _batch_schema(
    properties: dict[str, Tensor],
) -> tuple[LevelSchema, dict[str, set[str]]]:
    """Build flat-Batch storage metadata for CSP fields and system properties."""
    schema = LevelSchema()
    node_keys = {
        "positions",
        "atomic_numbers",
        "atomic_masses",
        "atom_categories",
        "velocities",
        *_ATOM_SOURCE_FIELDS,
    }
    system_keys = {
        "cell",
        "pbc",
        *_SYSTEM_SOURCE_FIELDS,
        "csp_source_structure_id",
        *properties,
    }
    schema.set("atom_categories", "atoms", dtype=torch.int64, is_segmented=True)
    for name in _ATOM_SOURCE_FIELDS:
        schema.set(name, "atoms", dtype=torch.int32, is_segmented=True)
    for name in _SYSTEM_SOURCE_FIELDS:
        schema.set(name, "system", dtype=torch.int32)
    schema.set("csp_source_structure_id", "system", dtype=torch.int64)
    for name in properties:
        schema.set(name, "system")
    return schema, {"node": node_keys, "edge": set(), "system": system_keys}


def expand_asu_batch(
    compact: RigidMoleculeASUBatch,
    *,
    indices: Tensor | None = None,
    device: torch.device | str | None = None,
) -> Batch:
    """Expand selected ASU rows to full-cell coordinates with provenance maps."""
    target_device = (
        _canonical_device(device) if device is not None else compact.cells.device
    )
    selected_index = _selected_indices(compact, indices)
    selected_count = int(selected_index.numel())
    source_device = compact.cells.device
    if selected_count > _INT32_MAX:
        raise OverflowError(
            f"P1 expansion has more than int32 capacity ({_INT32_MAX}) systems"
        )

    selected_cells = compact.cells.index_select(0, selected_index).to(target_device)
    selected_space_groups = compact.space_groups.index_select(0, selected_index).to(
        target_device
    )
    selected_z = compact.z.index_select(0, selected_index).to(target_device)
    selected_z_prime = compact.z_prime.index_select(0, selected_index).to(target_device)
    selected_structure_ids = compact.structure_ids.index_select(0, selected_index).to(
        target_device
    )
    selected_properties = {
        name: value.index_select(0, selected_index).to(target_device)
        for name, value in compact.properties.items()
    }

    formula = compact.packing_input
    num_formula_atoms = formula.num_atoms
    num_formula_molecules = formula.num_molecules
    node_counts64 = selected_z.to(torch.int64) * num_formula_atoms
    if selected_count:
        # Clamp before summing so the capacity check itself cannot overflow int64.
        # With P and z represented by int32, the clamped sum is bounded by < 2**62.
        checked_total = node_counts64.clamp(max=_INT32_MAX + 1).sum(dtype=torch.int64)
        num_nodes = int(checked_total.item())
        if num_nodes > _INT32_MAX:
            raise OverflowError(
                f"P1 expansion has more than int32 capacity ({_INT32_MAX}) atoms"
            )
        if num_nodes < 0:
            raise ValueError("P1 expansion node counts must be nonnegative")
        node_ptr64 = torch.cat(
            (
                torch.zeros(1, dtype=torch.int64, device=target_device),
                node_counts64.cumsum(dim=0),
            )
        )
    else:
        num_nodes = 0
        node_ptr64 = torch.zeros(1, dtype=torch.int64, device=target_device)

    node_counts = node_counts64.to(torch.int32)
    atom_map = torch.arange(num_nodes, dtype=torch.int64, device=target_device)
    if num_nodes:
        structure_molecule_ptr = compact.structure_molecule_ptr
        selected_molecule_starts = structure_molecule_ptr.index_select(
            0, selected_index
        )
        selected_compact_fields: tuple[Tensor, Tensor, Tensor] | None = None
        if source_device == target_device:
            molecule_row_offsets = selected_molecule_starts.to(
                device=target_device, dtype=torch.int64
            )
        else:
            selected_molecule_stops = structure_molecule_ptr.index_select(
                0, selected_index + 1
            )
            if source_device.type == "cpu":
                starts_cpu = selected_molecule_starts
                stops_cpu = selected_molecule_stops
            else:
                selected_bounds_cpu = torch.stack(
                    (selected_molecule_starts, selected_molecule_stops), dim=1
                ).to(device="cpu")
                starts_cpu = selected_bounds_cpu[:, 0]
                stops_cpu = selected_bounds_cpu[:, 1]
            molecule_counts_cpu = stops_cpu.to(torch.int64) - starts_cpu.to(torch.int64)
            num_selected_molecules = int(molecule_counts_cpu.sum().item())
            molecule_ptr_cpu = torch.cat(
                (
                    torch.zeros(1, dtype=torch.int64),
                    molecule_counts_cpu.cumsum(dim=0),
                )
            )
            selected_rows_cpu = torch.repeat_interleave(
                torch.arange(selected_count, dtype=torch.int64),
                molecule_counts_cpu,
                output_size=num_selected_molecules,
            )
            local_molecule_cpu = torch.arange(
                num_selected_molecules, dtype=torch.int64
            ) - molecule_ptr_cpu[:-1].index_select(0, selected_rows_cpu)
            selected_molecule_map_cpu = (
                starts_cpu.to(torch.int64).index_select(0, selected_rows_cpu)
                + local_molecule_cpu
            )
            selected_molecule_map_source = selected_molecule_map_cpu.to(
                device=source_device
            )
            selected_compact_fields = (
                compact.conformer_indices.index_select(
                    0, selected_molecule_map_source
                ).to(device=target_device),
                compact.rotations.index_select(0, selected_molecule_map_source).to(
                    device=target_device
                ),
                compact.fractional_centers.index_select(
                    0, selected_molecule_map_source
                ).to(device=target_device),
            )
            molecule_row_offsets = molecule_ptr_cpu[:-1].to(device=target_device)

        row_map = torch.repeat_interleave(
            torch.arange(selected_count, dtype=torch.int64, device=target_device),
            node_counts64,
            output_size=num_nodes,
        )
        local_node = atom_map - node_ptr64.index_select(0, row_map)
        asu_atoms_per_row = selected_z_prime.to(torch.int64) * num_formula_atoms
        operation_local = local_node // asu_atoms_per_row.index_select(0, row_map)
        repeated_asu_atom = local_node % asu_atoms_per_row.index_select(0, row_map)
        formula_atom = repeated_asu_atom % num_formula_atoms

        atom_ptr = formula.molecule_atom_ptr.to(device=target_device)
        formula_molecule = torch.searchsorted(
            atom_ptr[1:], formula_atom, right=True
        ).to(torch.int64)
        asu_repeat = repeated_asu_atom // num_formula_atoms
        # The ASU rows contain formula-unit molecule order repeated z_prime times.
        asu_molecule = asu_repeat * num_formula_molecules + formula_molecule

        compact_molecule = molecule_row_offsets.index_select(0, row_map) + asu_molecule

        conformer_ptr = formula.conformer_ptr.to(device=target_device)
        molecule_atom_start = atom_ptr.index_select(0, formula_molecule)

        atomic_numbers_formula = formula.atomic_numbers.to(device=target_device)
        atomic_numbers = atomic_numbers_formula.index_select(0, formula_atom)
        atomic_masses_formula = _mass_table(formula.atomic_numbers.device)[
            formula.atomic_numbers
        ].to(device=target_device, dtype=torch.float32)
        atomic_masses = atomic_masses_formula.index_select(0, formula_atom)
        component_index = formula.component_index.to(device=target_device)
        source_component = component_index.index_select(0, formula_molecule)

        source_asu_atom = repeated_asu_atom.to(torch.int32)
        source_molecule = asu_molecule.to(torch.int32)
        source_operation = operation_local.to(torch.int32)

        operation_data, operation_index, operation_ptr = _operation_tables(
            target_device
        )
        selected_group_index = selected_space_groups.to(torch.int64) - 1
        selected_operation_start = operation_ptr.index_select(
            0, selected_group_index
        ).to(torch.int64)
        atom_operation_index = (
            selected_operation_start.index_select(0, row_map) + operation_local
        )
        operation_ids = operation_index.index_select(
            0, atom_operation_index.to(torch.int64)
        )

        inverse_cells = torch.linalg.inv_ex(selected_cells, check_errors=False).inverse
        conformer_positions = formula.conformer_positions.to(device=target_device)
        if selected_compact_fields is None:
            compact_conformers = compact.conformer_indices
            compact_rotations = compact.rotations
            compact_centers = compact.fractional_centers
        else:
            compact_conformers, compact_rotations, compact_centers = (
                selected_compact_fields
            )
        source_conformer = torch.empty(
            num_nodes, dtype=torch.int32, device=target_device
        )
        positions = torch.empty(
            (num_nodes, 3), dtype=torch.float32, device=target_device
        )
        for chunk_start in range(0, num_nodes, _ATOM_CHUNK_SIZE):
            chunk_stop = min(chunk_start + _ATOM_CHUNK_SIZE, num_nodes)
            chunk_slice = slice(chunk_start, chunk_stop)
            row = row_map[chunk_slice]
            compact_mol = compact_molecule[chunk_slice]
            conformer_chunk = compact_conformers.index_select(0, compact_mol).to(
                device=target_device
            )
            source_conformer[chunk_slice] = conformer_chunk.to(torch.int32)
            rotation_chunk = compact_rotations.index_select(0, compact_mol).to(
                device=target_device
            )
            center_chunk = compact_centers.index_select(0, compact_mol).to(
                device=target_device
            )
            molecule_start_chunk = molecule_atom_start[chunk_slice]
            conformer_start_chunk = conformer_ptr.index_select(
                0, conformer_chunk.to(torch.int64)
            )
            formula_conf_atom = (
                conformer_start_chunk.to(torch.int64)
                + formula_atom[chunk_slice]
                - molecule_start_chunk
            )
            local_positions = conformer_positions.index_select(0, formula_conf_atom)
            local_cartesian = torch.bmm(
                local_positions.unsqueeze(1), rotation_chunk.transpose(1, 2)
            ).squeeze(1)
            fractional_displacement = torch.bmm(
                local_cartesian.unsqueeze(1), inverse_cells.index_select(0, row)
            ).squeeze(1)

            operation = operation_data.index_select(
                0, operation_ids[chunk_slice].to(torch.int64)
            )
            symmetry_rotation = operation[:, :, :3]
            symmetry_translation = operation[:, :, 3]
            transformed_center = (
                torch.bmm(
                    center_chunk.unsqueeze(1), symmetry_rotation.transpose(1, 2)
                ).squeeze(1)
                + symmetry_translation
            )
            wrapped_center = torch.remainder(transformed_center, 1.0)
            transformed_displacement = torch.bmm(
                fractional_displacement.unsqueeze(1), symmetry_rotation.transpose(1, 2)
            ).squeeze(1)
            fractional_position = wrapped_center + transformed_displacement
            positions[chunk_slice] = torch.bmm(
                fractional_position.unsqueeze(1), selected_cells.index_select(0, row)
            ).squeeze(1)

        atom_fields = {
            "positions": positions,
            "atomic_numbers": atomic_numbers,
            "atomic_masses": atomic_masses,
            "atom_categories": torch.zeros(
                num_nodes, dtype=torch.int64, device=target_device
            ),
            "velocities": torch.zeros(
                (num_nodes, 3), dtype=torch.float32, device=target_device
            ),
            "csp_source_asu_atom_index": source_asu_atom,
            "csp_source_molecule_index": source_molecule,
            "csp_source_component_index": source_component.to(torch.int32),
            "csp_source_conformer_index": source_conformer,
            "csp_source_symmetry_operation_index": source_operation,
        }
    else:
        empty_i32 = torch.empty(0, dtype=torch.int32, device=target_device)
        atom_fields = {
            "positions": torch.empty((0, 3), dtype=torch.float32, device=target_device),
            "atomic_numbers": torch.empty(0, dtype=torch.int64, device=target_device),
            "atomic_masses": torch.empty(0, dtype=torch.float32, device=target_device),
            "atom_categories": torch.empty(0, dtype=torch.int64, device=target_device),
            "velocities": torch.empty(
                (0, 3), dtype=torch.float32, device=target_device
            ),
            "csp_source_asu_atom_index": empty_i32,
            "csp_source_molecule_index": empty_i32.clone(),
            "csp_source_component_index": empty_i32.clone(),
            "csp_source_conformer_index": empty_i32.clone(),
            "csp_source_symmetry_operation_index": empty_i32.clone(),
            "csp_source_structure_id": torch.empty(
                (0, 2), dtype=torch.int64, device=target_device
            ),
        }

    system_fields = {
        "cell": selected_cells,
        "pbc": torch.ones((selected_count, 3), dtype=torch.bool, device=target_device),
        "csp_source_space_group": selected_space_groups.to(torch.int32),
        "csp_source_z": selected_z.to(torch.int32),
        "csp_source_z_prime": selected_z_prime.to(torch.int32),
        "csp_source_structure_id": selected_structure_ids,
        **selected_properties,
    }
    fields = {**atom_fields, **system_fields}
    schema, keys = _batch_schema(selected_properties)
    storage = MultiLevelStorage.from_data(
        fields,
        attr_map=schema,
        segment_lengths={"atoms": node_counts},
        device=target_device,
        validate=False,
    )
    return Batch(device=target_device, storage=storage, keys=keys)
