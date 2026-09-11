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
"""Tests for reaction-path interpolation."""

from __future__ import annotations

from unittest.mock import patch

import pytest
import torch
from torch import Tensor

from nvalchemi.data import AtomicData, Batch
from nvalchemi.dynamics.paths import interpolate_paths, validate_paths


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
        """Replicate ragged graphs directly and discard stale computed fields."""
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
