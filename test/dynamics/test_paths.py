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
"""Tests for reaction-path batch validation."""

from __future__ import annotations

import pytest
import torch

from nvalchemi.data import AtomicData, Batch
from nvalchemi.dynamics.paths import validate_paths


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
