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
"""Tests for the image-dependent pair-potential model."""

from __future__ import annotations

import pytest
import torch
from torch import Tensor

from nvalchemi.data import AtomicData, Batch
from nvalchemi.dynamics.paths import (
    IDPPModel,
    interpolate_paths,
    prepare_idpp_targets,
)

# =============================================================================
# Helpers
# =============================================================================


def _endpoints(
    positions: list[Tensor], atomic_numbers: list[Tensor]
) -> tuple[Batch, Batch]:
    """Return endpoint batches from alternating initial and final positions."""
    initial = Batch.from_data_list(
        [
            AtomicData(positions=positions[2 * index], atomic_numbers=numbers)
            for index, numbers in enumerate(atomic_numbers)
        ]
    )
    final = Batch.from_data_list(
        [
            AtomicData(positions=positions[2 * index + 1], atomic_numbers=numbers)
            for index, numbers in enumerate(atomic_numbers)
        ]
    )
    return initial, final


def _prepared_path(image: AtomicData, target_distances: Tensor) -> Batch:
    """Repeat one prepared image into a complete three-image path."""
    image.add_edge_property("idpp_target_distances", target_distances)
    paths = Batch.from_data_list([image] * 3)
    paths.set_group_layout(torch.zeros(3, dtype=torch.long, device=paths.device))
    return paths


def _pair_data(
    distance: float,
    target: float,
    *,
    cell: Tensor | None = None,
    pbc: Tensor | None = None,
) -> Batch:
    """Return a complete path with one prepared IDPP pair per image."""
    dtype = torch.float64
    image = AtomicData(
        positions=torch.tensor([[0.0, 0.0, 0.0], [distance, 0.0, 0.0]], dtype=dtype),
        atomic_numbers=torch.tensor([1, 1]),
        neighbor_list=torch.tensor([[0, 1]], dtype=torch.int32),
        cell=cell,
        pbc=pbc,
    )
    return _prepared_path(
        image, torch.tensor([target], dtype=dtype, device=image.device)
    )


# =============================================================================
# IDPP target preparation
# =============================================================================


class TestPrepareIDPPTargets:
    """Test preparation of image-dependent target pair distances."""

    def test_pair_distances_for_three_image_two_atom_path(self) -> None:
        """Pair targets interpolate across all three two-atom images."""
        initial, final = _endpoints(
            [
                torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
                torch.tensor([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]]),
            ],
            [torch.tensor([1, 1])],
        )
        paths = interpolate_paths(initial, final, 3)

        result = prepare_idpp_targets(paths)

        assert result is paths
        assert paths.num_edges_per_graph.tolist() == [1, 1, 1]
        assert paths.neighbor_list.tolist() == [[0, 1], [2, 3], [4, 5]]
        assert torch.allclose(
            paths.idpp_target_distances,
            torch.tensor([1.0, 2.0, 3.0]),
        )

    def test_prepare_supports_multiple_paths_with_different_atom_counts(self) -> None:
        """One prepared batch can hold ragged pair targets for multiple paths."""
        initial, final = _endpoints(
            [
                torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
                torch.tensor([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]]),
                torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
                torch.tensor([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]]),
            ],
            [torch.tensor([1, 1]), torch.tensor([1, 6, 8])],
        )
        paths = interpolate_paths(initial, final, [3, 4])

        prepare_idpp_targets(paths)
        model = IDPPModel()
        outputs = model(paths)

        assert paths.num_edges_per_graph.tolist() == [1, 1, 1, 3, 3, 3, 3]
        assert outputs["energy"].shape == (7, 1)
        assert outputs["forces"].shape == paths.positions.shape
        assert torch.isfinite(outputs["energy"]).all()
        assert torch.isfinite(outputs["forces"]).all()


# =============================================================================
# IDPP model
# =============================================================================


class TestIDPPModel:
    """Test IDPP energies, forces, configuration, and validation."""

    def test_analytic_forces_match_finite_difference(self) -> None:
        """Analytic IDPP forces agree with central energy differences."""
        image = AtomicData(
            positions=torch.tensor(
                [[0.0, 0.0, 0.0], [1.1, 0.2, 0.0], [0.1, 1.3, 0.3]],
                dtype=torch.float64,
            ),
            atomic_numbers=torch.tensor([1, 6, 8]),
            neighbor_list=torch.tensor([[0, 1], [0, 2], [1, 2]], dtype=torch.int32),
        )
        data = _prepared_path(
            image,
            torch.tensor([1.0, 1.2, 1.4], dtype=torch.float64),
        )
        model = IDPPModel()
        analytic_forces = model(data)["forces"]
        finite_difference_forces = torch.zeros_like(data.positions)
        displacement = 1.0e-6

        for atom in range(data.num_nodes):
            for axis in range(3):
                original = data.positions[atom, axis].item()
                data.positions[atom, axis] = original + displacement
                energy_plus = model(data)["energy"].sum()
                data.positions[atom, axis] = original - displacement
                energy_minus = model(data)["energy"].sum()
                data.positions[atom, axis] = original
                finite_difference_forces[atom, axis] = -(energy_plus - energy_minus) / (
                    2 * displacement
                )

        assert torch.allclose(
            analytic_forces,
            finite_difference_forces,
            rtol=1.0e-7,
            atol=1.0e-9,
        )

    def test_minimum_image_matches_equivalent_nonperiodic_pair(self) -> None:
        """A wrapped pair in a non-unit cell matches its direct displacement."""
        direct = IDPPModel()(_pair_data(distance=-0.2, target=0.4))
        wrapped = IDPPModel()(
            _pair_data(
                distance=1.8,
                target=0.4,
                cell=torch.diag(
                    torch.tensor([2.0, 3.0, 4.0], dtype=torch.float64)
                ).unsqueeze(0),
                pbc=torch.tensor([[True, False, False]]),
            )
        )

        assert torch.allclose(wrapped["energy"], direct["energy"])
        assert torch.allclose(wrapped["forces"], direct["forces"])

    def test_model_ignores_nonperiodic_placeholder_cell(self) -> None:
        """Nonperiodic placeholder cells do not enter minimum-image calculations."""
        reference = IDPPModel()(_pair_data(distance=1.5, target=1.0))
        with_cell = IDPPModel()(
            _pair_data(
                distance=1.5,
                target=1.0,
                cell=torch.zeros(1, 3, 3, dtype=torch.float64),
                pbc=torch.zeros(1, 3, dtype=torch.bool),
            )
        )

        assert torch.allclose(with_cell["energy"], reference["energy"])
        assert torch.allclose(with_cell["forces"], reference["forces"])

    def test_model_honors_active_outputs_and_declares_direct_forces(self) -> None:
        """IDPP computes only active outputs and identifies its analytic force."""
        model = IDPPModel()
        data = _pair_data(distance=1.5, target=1.0)

        model.set_config("active_outputs", {"energy"})
        assert set(model(data)) == {"energy"}

        model.set_config("active_outputs", {"forces"})
        assert set(model(data)) == {"forces"}
        assert model.direct_derivative_keys() == {"forces"}

    def test_model_rejects_coincident_pairs(self) -> None:
        """Coincident atoms fail instead of returning a zero separating force."""
        data = _pair_data(distance=0.0, target=1.0)

        with pytest.raises(RuntimeError, match="coincident atom pairs"):
            IDPPModel()(data)

    def test_model_requires_prepared_targets_and_has_no_neighbor_hook(self) -> None:
        """IDPPModel is stateless and does not request dynamic neighbor rebuilding."""
        initial, final = _endpoints(
            [torch.zeros(2, 3), torch.ones(2, 3)], [torch.tensor([1, 1])]
        )
        paths = interpolate_paths(initial, final, 3)
        model = IDPPModel()

        assert model.state_dict() == {}
        assert model.make_neighbor_hooks() == []
        with pytest.raises(KeyError, match="prepare_idpp_targets"):
            model(paths)
