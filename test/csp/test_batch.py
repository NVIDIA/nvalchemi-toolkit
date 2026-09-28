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
"""Flat P1 expansion and provenance tests for CSP compact structures."""

from __future__ import annotations

import pytest
import torch

from nvalchemi.csp.data import MolecularPackingInput, RigidMoleculeASUBatch
from nvalchemi.csp.symmetry import get_space_group_operations

ATOM_SOURCE_FIELDS = (
    "csp_source_asu_atom_index",
    "csp_source_molecule_index",
    "csp_source_component_index",
    "csp_source_conformer_index",
    "csp_source_symmetry_operation_index",
)
SYSTEM_SOURCE_FIELDS = (
    "csp_source_space_group",
    "csp_source_z",
    "csp_source_z_prime",
    "csp_source_structure_id",
)


def make_skew_compact() -> RigidMoleculeASUBatch:
    """Make two compatible monoclinic SG 4 structures with valid conformer pools."""
    packing_input = MolecularPackingInput(
        conformer_positions=torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0],
                [0.25, 0.5, 1.0],
                [-0.25, -0.5, -1.0],
                [0.0, 1.0, 1.0],
                [0.0, -1.0, -1.0],
                [0.5, 0.0, 0.5],
                [-0.5, 0.0, -0.5],
            ],
            dtype=torch.float32,
        ),
        conformer_ptr=torch.tensor([0, 2, 4, 6, 8], dtype=torch.int32),
        molecule_conformer_ptr=torch.tensor([0, 2, 4], dtype=torch.int32),
        molecule_atom_ptr=torch.tensor([0, 2, 4], dtype=torch.int32),
        atomic_numbers=torch.tensor([6, 1, 8, 7], dtype=torch.int64),
        contact_distances=torch.ones((4, 4), dtype=torch.float32),
        component_index=torch.tensor([0, 1], dtype=torch.int32),
        formula_unit_volume=64.0,
        metadata={"source": "independent-skew-oracle"},
    )
    rotations = torch.tensor(
        [
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
            [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]],
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
            [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]],
        ],
        dtype=torch.float32,
    )
    cells = torch.tensor(
        [
            [[4.0, 0.0, 0.0], [0.0, 5.0, 0.0], [1.0, 0.0, 6.0]],
            [[5.0, 0.0, 0.0], [0.0, 6.0, 0.0], [1.25, 0.0, 7.0]],
        ],
        dtype=torch.float32,
    )
    cosine = 2.0**-0.5
    orientation = torch.tensor(
        [[cosine, cosine, 0.0], [-cosine, cosine, 0.0], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
    )
    cells = cells @ orientation
    return RigidMoleculeASUBatch(
        packing_input=packing_input,
        structure_molecule_ptr=torch.tensor([0, 2, 4], dtype=torch.int32),
        conformer_indices=torch.tensor([1, 3, 0, 2], dtype=torch.int32),
        rotations=rotations,
        fractional_centers=torch.tensor(
            [
                [1.05, -0.1, 0.4],
                [0.3, 0.7, 0.6],
                [0.2, 0.4, 0.8],
                [-0.2, 1.1, 0.35],
            ],
            dtype=torch.float32,
        ),
        cells=cells,
        space_groups=torch.full((2,), 4, dtype=torch.int32),
        z=torch.full((2,), 2, dtype=torch.int32),
        z_prime=torch.ones(2, dtype=torch.int32),
        properties={"temperature": torch.tensor([280.0, 300.0], dtype=torch.float64)},
    )


def independent_p1_oracle(
    compact: RigidMoleculeASUBatch, selected_rows: list[int]
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Expand using explicit row, operation, molecule, and atom loops."""
    identity = torch.eye(3, dtype=torch.float32)
    screw_rotation = torch.diag(torch.tensor([-1.0, 1.0, -1.0]))
    operations = (
        (identity, torch.zeros(3)),
        (screw_rotation, torch.tensor([0.0, 0.5, 0.0])),
    )
    expected_positions: list[torch.Tensor] = []
    expected_asu_atom: list[int] = []
    expected_molecule: list[int] = []
    expected_component: list[int] = []
    expected_conformer: list[int] = []
    expected_operation: list[int] = []

    formula = compact.packing_input
    atom_ptr = formula.molecule_atom_ptr.tolist()
    conf_ptr = formula.conformer_ptr.tolist()
    for row in selected_rows:
        cell = compact.cells[row]
        inverse = torch.linalg.inv(cell)
        molecule_start = compact.structure_molecule_ptr[row].item()
        for operation_index, (symmetry_rotation, translation) in enumerate(operations):
            for molecule in range(formula.num_molecules):
                compact_molecule = molecule_start + molecule
                conformer = int(compact.conformer_indices[compact_molecule])
                molecule_rotation = compact.rotations[compact_molecule]
                center = compact.fractional_centers[compact_molecule]
                local_positions = formula.conformer_positions[
                    conf_ptr[conformer] : conf_ptr[conformer + 1]
                ]
                local_cartesian = local_positions @ molecule_rotation.T
                local_fractional = local_cartesian @ inverse
                center_fractional = center @ symmetry_rotation.T + translation
                center_fractional = torch.remainder(center_fractional, 1.0)
                transformed_displacement = local_fractional @ symmetry_rotation.T
                expected_positions.extend(
                    (center_fractional + transformed_displacement) @ cell
                )
                for atom in range(atom_ptr[molecule], atom_ptr[molecule + 1]):
                    expected_asu_atom.append(atom)
                    expected_molecule.append(molecule)
                    expected_component.append(int(formula.component_index[molecule]))
                    expected_conformer.append(conformer)
                    expected_operation.append(operation_index)

    num_rows = len(selected_rows)
    provenance = {
        "csp_source_asu_atom_index": torch.tensor(expected_asu_atom, dtype=torch.int32),
        "csp_source_molecule_index": torch.tensor(expected_molecule, dtype=torch.int32),
        "csp_source_component_index": torch.tensor(
            expected_component, dtype=torch.int32
        ),
        "csp_source_conformer_index": torch.tensor(
            expected_conformer, dtype=torch.int32
        ),
        "csp_source_symmetry_operation_index": torch.tensor(
            expected_operation, dtype=torch.int32
        ),
        "csp_source_space_group": torch.full((num_rows,), 4, dtype=torch.int32),
        "csp_source_z": torch.full((num_rows,), 2, dtype=torch.int32),
        "csp_source_z_prime": torch.ones(num_rows, dtype=torch.int32),
        "csp_source_structure_id": torch.empty((num_rows, 2), dtype=torch.int64),
    }
    provenance["csp_source_structure_id"] = compact.structure_ids[selected_rows].clone()
    return torch.stack(expected_positions), provenance


class TestP1Expansion:
    def test_skew_cell_geometry_and_provenance_match_independent_oracle(self) -> None:
        compact = make_skew_compact()
        expected_operations = get_space_group_operations(4)
        torch.testing.assert_close(
            expected_operations[0],
            torch.cat((torch.eye(3), torch.zeros((3, 1))), dim=1),
        )
        torch.testing.assert_close(
            expected_operations[1],
            torch.cat(
                (
                    torch.diag(torch.tensor([-1.0, 1.0, -1.0])),
                    torch.tensor([[0.0], [0.5], [0.0]]),
                ),
                dim=1,
            ),
        )
        assert not torch.equal(expected_operations[1, :, 3], torch.zeros(3))
        assert not torch.equal(
            expected_operations[1, :, :3] @ compact.cells[0],
            compact.cells[0] @ expected_operations[1, :, :3],
        )
        molecular_rotation = compact.rotations[0]
        assert not torch.equal(molecular_rotation, molecular_rotation.T)

        selected_rows = [1, 0, 1]
        expected_positions, expected_sources = independent_p1_oracle(
            compact, selected_rows
        )
        batch = compact.to_batch(indices=torch.tensor(selected_rows, dtype=torch.int32))

        assert batch.num_graphs == 3
        assert batch.num_nodes == 24
        torch.testing.assert_close(batch.batch_ptr, torch.tensor([0, 8, 16, 24]))
        torch.testing.assert_close(
            batch.positions, expected_positions, atol=3.0e-6, rtol=1.0e-6
        )
        torch.testing.assert_close(
            batch.atomic_numbers,
            torch.tensor([6, 1, 8, 7] * 6, dtype=torch.int64),
        )
        torch.testing.assert_close(batch.cell, compact.cells[[1, 0, 1]])
        assert batch.pbc.shape == (3, 3)
        assert bool(batch.pbc.all())
        torch.testing.assert_close(batch.velocities, torch.zeros((24, 3)))
        torch.testing.assert_close(
            batch.atom_categories, torch.zeros(24, dtype=torch.int64)
        )
        torch.testing.assert_close(
            batch.temperature, torch.tensor([300.0, 280.0, 300.0], dtype=torch.float64)
        )
        for name in (*ATOM_SOURCE_FIELDS, *SYSTEM_SOURCE_FIELDS):
            dtype = torch.int64 if name == "csp_source_structure_id" else torch.int32
            assert batch[name].dtype == dtype
            torch.testing.assert_close(batch[name], expected_sources[name])

    def test_batch_tensor_index_select_preserves_repeated_metadata_and_properties(
        self,
    ) -> None:
        compact = make_skew_compact()
        original = compact.to_batch(indices=torch.tensor([1, 0, 1], dtype=torch.int64))
        selected = original.index_select(torch.tensor([2, 0, 2], dtype=torch.int32))

        expected_positions, expected_sources = independent_p1_oracle(compact, [1, 1, 1])
        torch.testing.assert_close(selected.batch_ptr, torch.tensor([0, 8, 16, 24]))
        torch.testing.assert_close(
            selected.positions, expected_positions, atol=3.0e-6, rtol=1.0e-6
        )
        torch.testing.assert_close(
            selected.temperature,
            torch.tensor([300.0, 300.0, 300.0], dtype=torch.float64),
        )
        for name in (*ATOM_SOURCE_FIELDS, *SYSTEM_SOURCE_FIELDS):
            torch.testing.assert_close(selected[name], expected_sources[name])

    def test_empty_selection_returns_zero_graph_with_full_field_schema(self) -> None:
        empty = make_skew_compact().to_batch(indices=torch.empty(0, dtype=torch.int32))
        assert empty.num_graphs == 0
        assert empty.num_nodes == 0
        assert empty.batch_ptr.tolist() == [0]
        assert empty.positions.shape == (0, 3)
        assert empty.atomic_numbers.shape == (0,)
        assert empty.cell.shape == (0, 3, 3)
        assert empty.pbc.shape == (0, 3)
        assert empty.temperature.shape == (0,)
        for name in ATOM_SOURCE_FIELDS:
            assert empty[name].shape == (0,)
            assert empty[name].dtype == torch.int32
        for name in SYSTEM_SOURCE_FIELDS:
            if name == "csp_source_structure_id":
                assert empty[name].shape == (0, 2)
                assert empty[name].dtype == torch.int64
            else:
                assert empty[name].shape == (0,)
                assert empty[name].dtype == torch.int32
        assert "atom_categories" in empty.keys["node"]
        assert "csp_source_z" in empty.keys["system"]

    def test_z_prime_repeats_formula_unit_order_inside_each_operation(self) -> None:
        base = make_skew_compact()
        repeated_molecules = torch.tensor([0, 1, 0, 1], dtype=torch.int64)
        compact = RigidMoleculeASUBatch(
            packing_input=base.packing_input,
            structure_molecule_ptr=torch.tensor([0, 4], dtype=torch.int32),
            conformer_indices=base.conformer_indices.index_select(
                0, repeated_molecules
            ),
            rotations=base.rotations.index_select(0, repeated_molecules),
            fractional_centers=base.fractional_centers.index_select(
                0, repeated_molecules
            ),
            cells=base.cells[:1],
            space_groups=torch.tensor([1], dtype=torch.int32),
            z=torch.tensor([2], dtype=torch.int32),
            z_prime=torch.tensor([2], dtype=torch.int32),
        )
        batch = compact.to_batch()
        expected_positions = []
        for compact_molecule in repeated_molecules.tolist():
            conformer = int(base.conformer_indices[compact_molecule])
            start = int(compact.packing_input.conformer_ptr[conformer])
            positions = compact.packing_input.conformer_positions[start : start + 2]
            positions = positions @ compact.rotations[compact_molecule].T
            center = torch.remainder(compact.fractional_centers[compact_molecule], 1.0)
            fractional_displacement = positions @ torch.linalg.inv(compact.cells[0])
            expected_positions.extend(
                (center + fractional_displacement) @ compact.cells[0]
            )

        assert batch.num_nodes == 8
        torch.testing.assert_close(batch.positions, torch.stack(expected_positions))
        torch.testing.assert_close(
            batch.csp_source_asu_atom_index, torch.arange(8, dtype=torch.int32)
        )
        torch.testing.assert_close(
            batch.csp_source_molecule_index,
            torch.tensor([0, 0, 1, 1, 2, 2, 3, 3], dtype=torch.int32),
        )
        torch.testing.assert_close(
            batch.csp_source_component_index,
            torch.tensor([0, 0, 1, 1, 0, 0, 1, 1], dtype=torch.int32),
        )
        assert batch.csp_source_symmetry_operation_index.tolist() == [0] * 8

    def test_int32_node_capacity_is_checked_before_expanded_allocation(self) -> None:
        base = make_skew_compact()
        too_large = RigidMoleculeASUBatch(
            packing_input=base.packing_input,
            structure_molecule_ptr=base.structure_molecule_ptr[:2],
            conformer_indices=base.conformer_indices[:2],
            rotations=base.rotations[:2],
            fractional_centers=base.fractional_centers[:2],
            cells=base.cells[:1],
            space_groups=base.space_groups[:1],
            z=torch.tensor([torch.iinfo(torch.int32).max], dtype=torch.int32),
            z_prime=torch.ones(1, dtype=torch.int32),
        )
        import pytest

        with pytest.raises(OverflowError, match="int32 capacity"):
            too_large.to_batch()

    def test_explicit_cpu_target_preserves_compact_inputs(self) -> None:
        compact = make_skew_compact()
        before = compact.conformer_indices.clone()
        result = compact.to_batch(device="cpu")
        assert result.device.type == "cpu"
        torch.testing.assert_close(compact.conformer_indices, before)
        assert result.num_nodes == 16


def test_conditional_cuda_transfer_selection_and_materialization() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    cpu_compact = make_skew_compact()
    selected_rows = [1, 0, 1]
    expected = cpu_compact.to_batch(
        indices=torch.tensor(selected_rows, dtype=torch.int32)
    )
    cpu_to_cuda = cpu_compact.to_batch(
        indices=torch.tensor(selected_rows, dtype=torch.int32), device="cuda"
    )

    compact = cpu_compact.to("cuda")
    selected = compact.select(
        torch.tensor(selected_rows, dtype=torch.int32, device="cuda")
    )
    pretransferred_cuda = selected.to_batch()

    for batch in (cpu_to_cuda, pretransferred_cuda):
        assert batch.device.type == "cuda"
        assert batch.num_graphs == expected.num_graphs
        assert batch.num_nodes == expected.num_nodes
        torch.testing.assert_close(
            batch.positions.cpu(), expected.positions, atol=3.0e-6, rtol=1.0e-6
        )
        torch.testing.assert_close(batch.cell.cpu(), expected.cell, atol=0.0, rtol=0.0)
        assert torch.equal(batch.pbc.cpu(), expected.pbc)
        for name in (*ATOM_SOURCE_FIELDS, *SYSTEM_SOURCE_FIELDS):
            assert torch.equal(batch[name].cpu(), expected[name])


def test_cuda_operation_cache_resolves_unindexed_device_aliases() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    from nvalchemi.csp._batch import _operation_tables

    current = torch.cuda.current_device()
    with torch.cuda.device(current):
        unindexed = _operation_tables(torch.device("cuda"))
        explicit = _operation_tables(torch.device("cuda", current))
        assert unindexed is explicit
        torch.testing.assert_close(
            get_space_group_operations(4, device="cuda").cpu(),
            get_space_group_operations(4),
        )

    if torch.cuda.device_count() > 1:
        other = (current + 1) % torch.cuda.device_count()
        with torch.cuda.device(other):
            other_unindexed = _operation_tables(torch.device("cuda"))
            other_explicit = _operation_tables(torch.device("cuda", other))
            assert other_unindexed is other_explicit
        assert other_unindexed is not explicit
