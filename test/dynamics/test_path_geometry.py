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
"""Tests for exact minimum-image reaction-path geometry."""

from __future__ import annotations

from itertools import product

import pytest
import torch
import warp as wp

from nvalchemi.dynamics.paths._geometry import (
    minimum_image_displacement,
    prepare_mic,
)
from nvalchemi.dynamics.paths.neb._ops.kernels import _mic


def _enumerated_mic(
    displacement: torch.Tensor,
    cell: torch.Tensor,
    pbc: torch.Tensor,
) -> torch.Tensor:
    """Return the shortest candidate from a proven-complete enumeration."""
    basis = cell[pbc]
    if basis.shape[0] == 0:
        return displacement
    cartesian_to_fractional = torch.linalg.pinv(basis)
    fractional = displacement @ cartesian_to_fractional
    nearest = torch.round(fractional)
    radius = torch.linalg.vector_norm(displacement - nearest @ basis)
    bounds = torch.ceil(
        radius * torch.linalg.vector_norm(cartesian_to_fractional, dim=0)
    ).to(torch.int64)
    lower = torch.floor(fractional - bounds).to(torch.int64)
    upper = torch.ceil(fractional + bounds).to(torch.int64)
    best = displacement
    best_sq = torch.dot(best, best)
    ranges = [
        range(int(lower[index]), int(upper[index]) + 1)
        for index in range(basis.shape[0])
    ]
    for coefficients in product(*ranges):
        candidate = (
            displacement - torch.tensor(coefficients, dtype=displacement.dtype) @ basis
        )
        candidate_sq = torch.dot(candidate, candidate)
        if candidate_sq < best_sq:
            best, best_sq = candidate, candidate_sq
    return best


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_skewed_cell_returns_shortest_cartesian_displacement(
    dtype: torch.dtype,
) -> None:
    """Regression for fractional component-wise wrapping in a skewed cell."""
    cell = torch.tensor(
        [[[1.0, 0.0, 0.0], [0.5, 1.0, 0.0], [0.0, 0.0, 1.0]]],
        dtype=dtype,
    )
    displacement = torch.tensor([[0.7, 0.49, 0.0]], dtype=dtype)

    actual = minimum_image_displacement(
        displacement,
        torch.zeros(1, dtype=torch.long),
        cell,
        torch.ones((1, 3), dtype=torch.bool),
    )

    expected = torch.tensor([[0.2, -0.51, 0.0]], dtype=dtype)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)


def test_prepared_mic_uses_fixed_minkowski_neighbor_capacity() -> None:
    """General rank-two and rank-three cells use 8 and 26 neighbors."""
    cells = torch.tensor(
        [
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            [[1.0, 0.0, 0.0], [0.4, 1.0, 0.0], [0.0, 0.0, 1.0]],
            [[1.0, 0.0, 0.0], [0.4, 1.0, 0.0], [0.2, 0.3, 1.0]],
        ],
        dtype=torch.float64,
    )
    pbc = torch.tensor([[False, False, False], [True, True, False], [True, True, True]])

    prepared = prepare_mic(cells, pbc)

    assert prepared.candidate_count.tolist() == [0, 8, 26]
    assert prepared.candidate_shifts.shape == (3, 26, 3)
    assert torch.count_nonzero(prepared.candidate_shifts[0]) == 0
    assert torch.count_nonzero(prepared.candidate_shifts[1, 8:]) == 0


@pytest.mark.parametrize("mask_value", range(8))
def test_all_periodicity_masks_match_enumeration(mask_value: int) -> None:
    """PBC flags select arbitrary lattice rows and preserve the shortest image."""
    cell = torch.tensor(
        [[1.2, 0.1, 0.2], [0.45, 1.1, -0.1], [0.2, 0.35, 1.3]],
        dtype=torch.float64,
    )
    pbc = torch.tensor([(mask_value >> bit) & 1 for bit in range(3)]).bool()
    displacement = torch.tensor([3.7, -2.4, 1.9], dtype=torch.float64)

    actual = minimum_image_displacement(
        displacement[None],
        torch.zeros(1, dtype=torch.long),
        cell[None],
        pbc[None],
    )[0]
    expected = _enumerated_mic(displacement, cell, pbc)

    torch.testing.assert_close(actual, expected, atol=1e-11, rtol=1e-11)


def test_seeded_skewed_cells_match_independent_enumeration() -> None:
    """Random well-conditioned cells and multi-image shifts match enumeration."""
    generator = torch.Generator().manual_seed(20260915)
    pbc = torch.ones(3, dtype=torch.bool)
    for _ in range(12):
        cell = torch.eye(3, dtype=torch.float64)
        cell *= 0.8 + torch.rand(3, generator=generator, dtype=torch.float64)
        cell += torch.tril(
            0.4 * (torch.rand((3, 3), generator=generator, dtype=torch.float64) - 0.5),
            diagonal=-1,
        )
        displacement = 3 * torch.randn(3, generator=generator, dtype=torch.float64)
        lattice_shift = torch.randint(-5, 6, (3,), generator=generator).to(
            torch.float64
        )
        displacement += lattice_shift @ cell

        actual = minimum_image_displacement(
            displacement[None],
            torch.zeros(1, dtype=torch.long),
            cell[None],
            pbc[None],
        )[0]
        expected = _enumerated_mic(displacement, cell, pbc)

        torch.testing.assert_close(actual, expected, atol=1e-11, rtol=1e-11)


def test_partial_pbc_ignores_oblique_nonperiodic_rows() -> None:
    """A nonperiodic row cannot contaminate periodic projection coordinates."""
    cell = torch.tensor(
        [[[1.0, 0.0, 0.0], [20.0, 1.0, 0.0], [0.0, 0.0, 0.0]]],
        dtype=torch.float64,
    )
    pbc = torch.tensor([[True, False, False]])
    displacement = torch.tensor([[2.6, 40.0, -3.0]], dtype=torch.float64)

    actual = minimum_image_displacement(
        displacement, torch.zeros(1, dtype=torch.long), cell, pbc
    )

    torch.testing.assert_close(
        actual, torch.tensor([[-0.4, 40.0, -3.0]], dtype=torch.float64)
    )


def test_unreduced_basis_and_multi_image_shift_match_enumeration() -> None:
    """Reduction preserves a lattice represented by long unimodular rows."""
    base = torch.tensor(
        [[1.0, 0.0, 0.0], [0.2, 1.1, 0.0], [0.1, 0.3, 1.2]],
        dtype=torch.float64,
    )
    transform = torch.tensor([[1, 8, -3], [0, 1, 5], [0, 0, 1]])
    cell = transform.to(torch.float64) @ base
    displacement = torch.tensor([12.3, -7.1, 4.4], dtype=torch.float64)

    actual = minimum_image_displacement(
        displacement[None],
        torch.zeros(1, dtype=torch.long),
        cell[None],
        torch.ones((1, 3), dtype=torch.bool),
    )[0]
    expected = _enumerated_mic(displacement, base, torch.ones(3, dtype=torch.bool))

    torch.testing.assert_close(actual, expected, atol=1e-11, rtol=1e-11)


def test_rotational_covariance_and_translation_invariance() -> None:
    """Rigid rotations and multi-cell translations preserve MIC geometry."""
    cell = torch.tensor(
        [[1.0, 0.0, 0.0], [0.45, 1.1, 0.0], [0.1, 0.2, 1.2]],
        dtype=torch.float64,
    )
    pbc = torch.tensor([[True, True, False]])
    displacement = torch.tensor([[0.71, 0.49, 2.3]], dtype=torch.float64)
    angle = torch.tensor(0.37, dtype=torch.float64)
    rotation = torch.tensor(
        [
            [torch.cos(angle), -torch.sin(angle), 0.0],
            [torch.sin(angle), torch.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float64,
    )
    translated = (
        displacement + torch.tensor([[4.0, -3.0, 0.0]], dtype=torch.float64) @ cell
    )

    reference = minimum_image_displacement(
        displacement, torch.zeros(1, dtype=torch.long), cell[None], pbc
    )
    shifted = minimum_image_displacement(
        translated, torch.zeros(1, dtype=torch.long), cell[None], pbc
    )
    rotated = minimum_image_displacement(
        displacement @ rotation,
        torch.zeros(1, dtype=torch.long),
        (cell @ rotation)[None],
        pbc,
    )

    torch.testing.assert_close(shifted, reference, atol=1e-11, rtol=1e-11)
    torch.testing.assert_close(rotated, reference @ rotation, atol=1e-11, rtol=1e-11)


def test_zero_nonperiodic_rows_are_accepted() -> None:
    """A fully nonperiodic zero cell leaves displacements unchanged."""
    zero_cell = torch.zeros((1, 3, 3), dtype=torch.float64)
    nonperiodic = prepare_mic(zero_cell, torch.zeros((1, 3), dtype=torch.bool))
    displacement = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float64)
    actual = minimum_image_displacement(
        displacement, torch.zeros(1, dtype=torch.long), prepared=nonperiodic
    )
    assert torch.equal(actual, displacement)


def test_minimum_image_displacement_compiles_as_full_graph() -> None:
    """Prepared MIC execution has no tensor-dependent Python graph breaks."""
    cell = torch.tensor(
        [[[1.0, 0.0, 0.0], [0.4, 1.0, 0.0], [0.0, 0.0, 1.0]]],
        dtype=torch.float64,
    )
    pbc = torch.ones((1, 3), dtype=torch.bool)
    prepared = prepare_mic(cell, pbc)
    displacement = torch.tensor([[0.7, 0.49, 0.0]], dtype=torch.float64)
    graph_idx = torch.zeros(1, dtype=torch.long)
    expected = minimum_image_displacement(displacement, graph_idx, prepared=prepared)
    compiled = torch.compile(
        minimum_image_displacement, backend="eager", fullgraph=True
    )

    actual = compiled(displacement, graph_idx, prepared=prepared)

    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("mode", ["none", "orthogonal", "mixed"])
def test_prepared_dispatch_preserves_mixed_geometry(
    device: str, compiled: bool, mode: str
) -> None:
    """Prepared dispatch preserves each path's geometry in eager and compiled calls."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    cells = torch.tensor(
        [
            [[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]],
            [[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]],
            [[1.0, 0, 0], [0.5, 1.0, 0], [0, 0, 0]],
        ],
        device=device,
    )
    pbc = torch.tensor([[False] * 3, [True] * 3, [True, True, False]], device=device)
    if mode == "none":
        pbc[:] = False
    elif mode == "orthogonal":
        pbc[2] = False
    prepared = prepare_mic(cells, pbc)
    displacement = torch.tensor([[0.7, 0.49, 10000.0]] * 3, device=device)
    expected = displacement.clone()
    if mode != "none":
        expected[1] = torch.tensor([-0.3, 0.49, 0.0], device=device)
    if mode == "mixed":
        expected[2] = torch.tensor([0.2, -0.51, 10000.0], device=device)
    function = minimum_image_displacement
    if compiled:
        function = torch.compile(function, backend="eager", fullgraph=True)
    actual = function(displacement, torch.arange(3, device=device), prepared=prepared)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=0)


@wp.kernel
def _partial_periodic_mic_kernel(
    displacement: wp.array(dtype=wp.vec3f),
    basis: wp.array(dtype=wp.mat33f),
    coordinate_map: wp.array(dtype=wp.mat33f),
    shifts: wp.array2d(dtype=wp.vec3f),
    output: wp.array(dtype=wp.vec3f),
):
    """Exercise the shared Warp primitive with a rank-two periodic lattice."""
    i = wp.tid()
    output[i] = _mic(
        displacement[i],
        wp.vec3f(0.0),
        wp.int32(2),
        basis[0],
        coordinate_map[0],
        wp.int32(0),
        wp.int32(8),
        shifts,
    )


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_warp_mic_preserves_large_perpendicular_component(device: str) -> None:
    """A perpendicular residual must not hide the shorter periodic image."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    cell = torch.tensor([[[1.0, 0, 0], [0.5, 1.0, 0], [0, 0, 0]]], device=device)
    prepared = prepare_mic(cell, torch.tensor([[True, True, False]], device=device))
    displacement = torch.tensor(
        [[0.7, 0.49, 10000.0], [-0.7, -0.49, -10000.0]], device=device
    )
    output = torch.empty_like(displacement)
    wp.launch(
        _partial_periodic_mic_kernel,
        dim=2,
        inputs=[
            wp.from_torch(displacement, dtype=wp.vec3f),
            wp.from_torch(prepared.periodic_basis, dtype=wp.mat33f),
            wp.from_torch(prepared.cartesian_to_fractional, dtype=wp.mat33f),
            wp.from_torch(prepared.candidate_shifts, dtype=wp.vec3f),
            wp.from_torch(output, dtype=wp.vec3f),
        ],
        device=device,
    )
    wp.synchronize_device(device)
    expected = torch.tensor(
        [[0.2, -0.51, 10000.0], [-0.2, 0.51, -10000.0]], device=device
    )
    torch.testing.assert_close(output, expected, atol=2e-6, rtol=0)
