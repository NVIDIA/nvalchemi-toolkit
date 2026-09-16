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
"""Tests for reaction-path validation and interpolation."""

from __future__ import annotations

from unittest.mock import patch

import pytest
import torch
from torch import Tensor

from nvalchemi.data import AtomicData, Batch
from nvalchemi.dynamics.paths import interpolate_paths, validate_paths
from nvalchemi.dynamics.paths._geometry import prepare_mic

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _path_batch(
    atomic_numbers: list[list[int]], group_idx: list[int] | None = None
) -> Batch:
    images = [
        AtomicData(
            atomic_numbers=torch.tensor(numbers),
            positions=torch.zeros(len(numbers), 3),
        )
        for numbers in atomic_numbers
    ]
    batch = Batch.from_data_list(images)
    batch.set_group_layout(
        torch.tensor(group_idx if group_idx is not None else [0] * len(images))
    )
    return batch


def _batch(
    positions: list[Tensor],
    atomic_numbers: list[Tensor],
    *,
    cell: Tensor | None = None,
    pbc: Tensor | None = None,
) -> Batch:
    return Batch.from_data_list(
        [
            AtomicData(
                positions=coordinates,
                atomic_numbers=numbers,
                cell=cell,
                pbc=pbc,
            )
            for coordinates, numbers in zip(positions, atomic_numbers, strict=True)
        ]
    )


# ---------------------------------------------------------------------------
# Test validate_paths
# ---------------------------------------------------------------------------


class TestValidatePaths:
    """Validate reaction-path cardinality, atomic composition, and periodicity."""

    def test_accepts_multiple_valid_paths(self) -> None:
        batch = _path_batch(
            [[1, 8], [1, 8], [1, 8], [6], [6], [6], [6]],
            [0, 0, 0, 1, 1, 1, 1],
        )

        assert validate_paths(batch) is None

    def test_rejects_empty_path_batch(self) -> None:
        batch = Batch.empty(num_systems=0, num_nodes=0, num_edges=0)
        batch.set_group_layout(torch.empty(0, dtype=torch.long))

        with pytest.raises(ValueError, match="at least one path"):
            validate_paths(batch)

    def test_rejects_different_atom_counts(self) -> None:
        batch = _path_batch([[1], [1, 1], [1]])

        with pytest.raises(ValueError, match="same atom count"):
            validate_paths(batch)

    def test_rejects_different_atomic_numbers(self) -> None:
        batch = _path_batch([[1, 8], [1, 6], [1, 8]])

        with pytest.raises(ValueError, match="identical atomic numbers"):
            validate_paths(batch)

    def test_accepts_independent_path_cells_and_pbc(self) -> None:
        batch = _path_batch([[1]] * 6, [0, 0, 0, 1, 1, 1])
        batch.cell = torch.stack(
            [
                torch.eye(3),
                torch.eye(3) + 1e-7,
                torch.eye(3),
                torch.eye(3) * 2,
                torch.eye(3) * 2,
                torch.eye(3) * 2,
            ]
        )
        batch.pbc = torch.tensor(
            [
                [True, True, True],
                [True, True, True],
                [True, True, True],
                [True, False, False],
                [True, False, False],
                [True, False, False],
            ]
        )

        assert validate_paths(batch) is None

    def test_accepts_cell_without_pbc_without_mutating_batch(self) -> None:
        batch = _path_batch([[1]] * 3)
        batch.cell = torch.eye(3).repeat(3, 1, 1)
        original_cell = batch.cell.clone()

        assert "pbc" not in batch
        assert validate_paths(batch) is None
        assert "pbc" not in batch
        assert torch.equal(batch.cell, original_cell)

    @pytest.mark.parametrize(
        ("field", "value", "expected_shape"),
        [
            ("cell", torch.zeros(3, 3), r"\(3, 3, 3\)"),
            ("pbc", torch.zeros(3, dtype=torch.bool), r"\(3, 3\)"),
        ],
    )
    def test_rejects_invalid_periodic_field_shape(
        self,
        field: str,
        value: torch.Tensor,
        expected_shape: str,
    ) -> None:
        batch = _path_batch([[1]] * 3)
        setattr(batch, field, value)

        with pytest.raises(
            ValueError, match=rf"{field} must have shape {expected_shape}"
        ):
            validate_paths(batch)

    def test_rejects_different_pbc_within_path(self) -> None:
        batch = _path_batch([[1]] * 3)
        batch.cell = torch.eye(3).repeat(3, 1, 1)
        batch.pbc = torch.tensor(
            [[True, True, True], [True, False, True], [True, True, True]]
        )

        with pytest.raises(ValueError, match="identical PBC settings"):
            validate_paths(batch)

    def test_rejects_different_cells_within_path(self) -> None:
        batch = _path_batch([[1]] * 3)
        batch.cell = torch.stack([torch.eye(3), torch.eye(3) * 1.01, torch.eye(3)])

        with pytest.raises(ValueError, match="same cell"):
            validate_paths(batch)


# ---------------------------------------------------------------------------
# Test interpolate_paths
# ---------------------------------------------------------------------------


class TestInterpolatePaths:
    """Interpolate one or more index-corresponded endpoint pairs."""

    def test_linear_interpolates_multiple_paths_and_sets_layout(self) -> None:
        initial = _batch(
            [torch.zeros(2, 3), torch.zeros(1, 3)],
            [torch.tensor([1, 8]), torch.tensor([6])],
        )
        final = _batch(
            [torch.full((2, 3), 2.0), torch.full((1, 3), 4.0)],
            [torch.tensor([1, 8]), torch.tensor([6])],
        )

        paths = interpolate_paths(initial, final, [3, 5])

        assert paths.group_idx.tolist() == [0, 0, 0, 1, 1, 1, 1, 1]
        assert paths.group_layout.num_graphs_per_group.tolist() == [3, 5]
        assert torch.equal(paths.get_data(0).positions, initial.get_data(0).positions)
        assert torch.equal(paths.get_data(2).positions, final.get_data(0).positions)
        assert torch.equal(paths.get_data(3).positions, initial.get_data(1).positions)
        assert torch.equal(paths.get_data(7).positions, final.get_data(1).positions)
        assert torch.allclose(paths.get_data(1).positions, torch.ones(2, 3))
        assert torch.allclose(paths.get_data(5).positions, torch.full((1, 3), 2.0))
        assert validate_paths(paths) is None

    def test_uses_batch_storage_without_reconstructing_atomic_data(self) -> None:
        """Replicate variable-size graphs directly and discard stale computed fields."""
        initial = _batch(
            [torch.zeros(2, 3), torch.zeros(1, 3)],
            [torch.tensor([1, 8]), torch.tensor([6])],
        )
        final = _batch(
            [torch.full((2, 3), 2.0), torch.full((1, 3), 4.0)],
            [torch.tensor([1, 8]), torch.tensor([6])],
        )
        initial.forces = torch.ones_like(initial.positions)
        initial.velocities = torch.ones_like(initial.positions)
        initial.energy = torch.ones(initial.num_graphs, 1)

        with (
            patch.object(
                Batch, "to_data_list", side_effect=AssertionError("reconstructed")
            ),
            patch.object(
                Batch, "from_data_list", side_effect=AssertionError("reconstructed")
            ),
        ):
            paths = interpolate_paths(initial, final, [3, 4])

        assert validate_paths(paths) is None
        assert paths.num_graphs == 7
        assert paths.num_nodes == 10
        assert paths.group_idx.tolist() == [0, 0, 0, 1, 1, 1, 1]
        assert "forces" not in paths
        assert "velocities" not in paths
        assert "energy" not in paths
        assert "forces" in initial

    def test_uses_minimum_image_displacement_per_graph_and_keeps_endpoints(
        self,
    ) -> None:
        cell = 10.0 * torch.eye(3).unsqueeze(0)
        periodic = torch.tensor([[True, False, False]])
        nonperiodic = torch.tensor([[False, False, False]])
        numbers = torch.tensor([1])
        initial = Batch.from_data_list(
            [
                AtomicData(
                    positions=torch.tensor([[9.0, 0.0, 0.0]]),
                    atomic_numbers=numbers,
                    cell=cell,
                    pbc=periodic,
                ),
                AtomicData(
                    positions=torch.tensor([[1.0, 0.0, 0.0]]),
                    atomic_numbers=numbers,
                    cell=cell,
                    pbc=nonperiodic,
                ),
            ]
        )
        final = Batch.from_data_list(
            [
                AtomicData(
                    positions=torch.tensor([[1.0, 0.0, 0.0]]),
                    atomic_numbers=numbers,
                    cell=cell,
                    pbc=periodic,
                ),
                AtomicData(
                    positions=torch.tensor([[3.0, 0.0, 0.0]]),
                    atomic_numbers=numbers,
                    cell=cell,
                    pbc=nonperiodic,
                ),
            ]
        )

        path = interpolate_paths(initial, final, 3)

        assert validate_paths(path) is None
        assert torch.equal(path.get_data(0).positions, initial.get_data(0).positions)
        assert torch.equal(path.get_data(2).positions, final.get_data(0).positions)
        assert torch.equal(path.get_data(3).positions, initial.get_data(1).positions)
        assert torch.equal(path.get_data(5).positions, final.get_data(1).positions)
        assert torch.allclose(
            path.get_data(1).positions, torch.tensor([[10.0, 0.0, 0.0]])
        )
        assert torch.allclose(
            path.get_data(4).positions, torch.tensor([[2.0, 0.0, 0.0]])
        )

    def test_optionally_removes_translation_and_rotation(self) -> None:
        initial_positions = torch.tensor(
            [[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0]]
        )
        rotation = torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        final_positions = initial_positions @ rotation.T + torch.tensor(
            [4.0, -3.0, 2.0]
        )
        numbers = [torch.tensor([1, 1, 8])]
        initial = _batch([initial_positions], numbers)
        final = _batch([final_positions], numbers)

        path = interpolate_paths(
            initial,
            final,
            3,
            remove_translation_and_rotation=True,
        )

        for image in path.to_data_list():
            torch.testing.assert_close(image.positions, initial_positions)
        torch.testing.assert_close(final.positions, final_positions)

    def test_fit_mask_aligns_multiple_endpoint_pairs_independently(self) -> None:
        reference_core = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [4.0, 2.0, 1.0],
            ],
            dtype=torch.float64,
        )
        reference_second = torch.tensor(
            [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 1.0, 1.0]],
            dtype=torch.float64,
        )
        rotation_first = torch.tensor(
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
            dtype=torch.float64,
        )
        rotation_second = torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]],
            dtype=torch.float64,
        )
        mobile_core = reference_core @ rotation_first + torch.tensor(
            [8.0, -3.0, 2.0], dtype=torch.float64
        )
        # Add internal motion to the atom excluded from the alignment fit.
        mobile_core[3, 2] += 1.5
        mobile_second = reference_second @ rotation_second + torch.tensor(
            [-2.0, 5.0, 1.0], dtype=torch.float64
        )
        initial = _batch(
            [reference_core, reference_second],
            [torch.tensor([46, 6, 6, 1]), torch.tensor([8, 1, 1])],
        )
        final = _batch(
            [mobile_core, mobile_second],
            [torch.tensor([46, 6, 6, 1]), torch.tensor([8, 1, 1])],
        )
        fit_mask = torch.tensor([True, True, True, False, True, True, True])

        paths = interpolate_paths(
            initial,
            final,
            [3, 4],
            remove_translation_and_rotation=True,
            fit_mask=fit_mask,
        )

        first_terminal = paths.get_data(2).positions
        second_terminal = paths.get_data(6).positions
        torch.testing.assert_close(
            first_terminal[:3], reference_core[:3], rtol=0, atol=1.0e-12
        )
        torch.testing.assert_close(
            torch.linalg.vector_norm(first_terminal[3] - reference_core[3]),
            torch.tensor(1.5, dtype=torch.float64),
            rtol=0,
            atol=1.0e-12,
        )
        torch.testing.assert_close(
            second_terminal, reference_second, rtol=0, atol=1.0e-12
        )

    def test_fit_mask_requires_endpoint_alignment(self) -> None:
        endpoint = _batch([torch.zeros(2, 3)], [torch.tensor([1, 8])])

        with pytest.raises(
            ValueError, match="requires remove_translation_and_rotation=True"
        ):
            interpolate_paths(
                endpoint,
                endpoint,
                3,
                fit_mask=torch.ones(2, dtype=torch.bool),
            )

    def test_alignment_removes_periodic_minimum_image_translation(self) -> None:
        cell = 10.0 * torch.eye(3).unsqueeze(0)
        pbc = torch.tensor([[True, False, False]])
        numbers = [torch.tensor([1, 8])]
        initial_positions = torch.tensor([[9.0, 0.0, 0.0], [2.0, 1.0, 0.0]])
        final_positions = torch.tensor([[1.0, 0.0, 0.0], [4.0, 1.0, 0.0]])
        initial = _batch([initial_positions], numbers, cell=cell, pbc=pbc)
        final = _batch([final_positions], numbers, cell=cell, pbc=pbc)

        path = interpolate_paths(
            initial,
            final,
            3,
            remove_translation_and_rotation=True,
        )

        for image in path.to_data_list():
            torch.testing.assert_close(image.positions, initial_positions)

    def test_prepares_periodic_geometry_once_for_alignment_and_interpolation(
        self,
    ) -> None:
        """Reuse one prepared MIC object throughout endpoint construction."""
        positions = [torch.zeros((2, 3), dtype=torch.float64)]
        numbers = [torch.tensor([1, 8])]
        cell = torch.tensor(
            [[[1.0, 0.0, 0.0], [0.5, 1.0, 0.0], [0.0, 0.0, 1.0]]],
            dtype=torch.float64,
        )
        pbc = torch.ones((1, 3), dtype=torch.bool)
        initial = _batch(positions, numbers, cell=cell, pbc=pbc)
        final = _batch(positions, numbers, cell=cell, pbc=pbc)

        with patch(
            "nvalchemi.dynamics.paths.interpolate.prepare_mic",
            wraps=prepare_mic,
        ) as prepare:
            interpolate_paths(
                initial,
                final,
                3,
                remove_translation_and_rotation=True,
            )

        prepare.assert_called_once_with(initial.cell, initial.pbc)

    def test_rejects_atomic_number_reordering(self) -> None:
        positions = [torch.zeros(2, 3)]
        initial = _batch(positions, [torch.tensor([1, 8])])
        final = _batch(positions, [torch.tensor([8, 1])])

        with pytest.raises(ValueError, match="same order"):
            interpolate_paths(initial, final, 3)

    def test_rejects_inconsistent_cells(self) -> None:
        positions = [torch.zeros(1, 3)]
        numbers = [torch.tensor([1])]
        pbc = torch.tensor([[True, True, True]])
        initial = _batch(positions, numbers, cell=torch.eye(3).unsqueeze(0), pbc=pbc)
        final = _batch(positions, numbers, cell=2 * torch.eye(3).unsqueeze(0), pbc=pbc)

        with pytest.raises(ValueError, match="same cell"):
            interpolate_paths(initial, final, 3)
