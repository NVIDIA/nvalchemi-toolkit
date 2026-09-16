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
"""Tests for rigid alignment of paired position batches."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest
import torch

from nvalchemi.dynamics.paths._alignment import align_batch_positions
from nvalchemi.dynamics.paths._geometry import prepare_mic


def _nonperiodic_batched_pair(
    dtype: torch.dtype, device: str
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    references = [
        torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 2.0, 0.0],
                [0.0, 0.0, 3.0],
            ],
            dtype=dtype,
            device=device,
        ),
        torch.tensor(
            [
                [-1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [2.0, 0.0, 1.0],
                [0.0, -2.0, 0.0],
                [0.0, 0.0, 3.0],
            ],
            dtype=dtype,
            device=device,
        ),
    ]
    applied_rotations = [
        torch.tensor(
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
            dtype=dtype,
            device=device,
        ),
        torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]],
            dtype=dtype,
            device=device,
        ),
    ]
    shifts = [
        torch.tensor([4.0, -3.0, 2.0], dtype=dtype, device=device),
        torch.tensor([-2.0, 5.0, 1.0], dtype=dtype, device=device),
    ]
    mobiles = [
        reference @ rotation.T + shift
        for reference, rotation, shift in zip(
            references, applied_rotations, shifts, strict=True
        )
    ]
    counts = torch.tensor([4, 5], dtype=torch.long, device=device)
    batch_idx = torch.repeat_interleave(torch.arange(2, device=device), counts)
    return torch.cat(references), torch.cat(mobiles), batch_idx, counts


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_align_batch_positions_handles_variable_size_nonperiodic_graphs(
    dtype: torch.dtype, device: str
) -> None:
    """Align variable-size graphs independently while preserving dtype and device."""
    reference, mobile, batch_idx, counts = _nonperiodic_batched_pair(dtype, device)
    original_reference = reference.clone()
    original_mobile = mobile.clone()

    alignment = align_batch_positions(reference, mobile, batch_idx, counts)

    atol = 1.0e-5 if dtype == torch.float32 else 1.0e-12
    torch.testing.assert_close(alignment.positions, reference, rtol=0, atol=atol)
    torch.testing.assert_close(
        alignment.rotation @ alignment.rotation.transpose(-2, -1),
        torch.eye(3, dtype=dtype, device=device).expand(2, -1, -1),
        rtol=0,
        atol=atol,
    )
    torch.testing.assert_close(
        torch.linalg.det(alignment.rotation),
        torch.ones(2, dtype=dtype, device=device),
        rtol=0,
        atol=atol,
    )
    transformed = (
        torch.bmm(alignment.rotation[batch_idx], mobile.unsqueeze(-1)).squeeze(-1)
        + alignment.translation[batch_idx]
    )
    torch.testing.assert_close(transformed, alignment.positions, rtol=0, atol=atol)
    assert alignment.positions.dtype == dtype
    assert alignment.positions.device.type == device
    assert alignment.rotation.shape == (2, 3, 3)
    assert alignment.translation.shape == (2, 3)
    assert torch.equal(reference, original_reference)
    assert torch.equal(mobile, original_mobile)


def test_align_batch_positions_uses_fit_mask_and_transforms_all_atoms() -> None:
    """Fit on a rigid core without discarding motion of unselected atoms."""
    reference = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [4.0, 2.0, 1.0],
        ],
        dtype=torch.float64,
    )
    applied_rotation = torch.tensor(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=torch.float64,
    )
    mobile = reference @ applied_rotation + torch.tensor(
        [8.0, -3.0, 2.0], dtype=torch.float64
    )
    mobile[3] += torch.tensor([0.5, -1.0, 1.5], dtype=torch.float64)
    fit_mask = torch.tensor([True, True, True, False])

    alignment = align_batch_positions(
        reference,
        mobile,
        torch.zeros(4, dtype=torch.long),
        torch.tensor([4]),
        fit_mask=fit_mask,
    )

    torch.testing.assert_close(
        alignment.positions[fit_mask], reference[fit_mask], rtol=0, atol=1.0e-12
    )
    torch.testing.assert_close(
        alignment.positions[3],
        reference[3] + torch.tensor([1.0, 0.5, 1.5], dtype=torch.float64),
        rtol=0,
        atol=1.0e-12,
    )


def test_align_batch_positions_supports_masked_mixed_periodicity(device: str) -> None:
    """Route masked periodic and nonperiodic fits correctly on each device."""
    nonperiodic_reference = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]],
        dtype=torch.float64,
        device=device,
    )
    applied_rotation = torch.tensor(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=torch.float64,
        device=device,
    )
    nonperiodic_mobile = nonperiodic_reference @ applied_rotation.T + torch.tensor(
        [4.0, -3.0, 2.0], dtype=torch.float64, device=device
    )
    periodic_reference = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=torch.float64, device=device
    )
    periodic_mobile = torch.tensor(
        [[8.0, 2.0, 0.0], [0.0, 3.0, 0.0]], dtype=torch.float64, device=device
    )
    reference = torch.cat((nonperiodic_reference, periodic_reference))
    mobile = torch.cat((nonperiodic_mobile, periodic_mobile))
    batch_idx = torch.tensor([0, 0, 0, 0, 1, 1], device=device)
    counts = torch.tensor([4, 2], device=device)
    cell = torch.stack(
        (torch.zeros(3, 3, device=device), 10.0 * torch.eye(3, device=device))
    ).to(torch.float64)
    pbc = torch.tensor([[False, False, False], [True, False, False]], device=device)

    alignment = align_batch_positions(
        reference,
        mobile,
        batch_idx,
        counts,
        cell=cell,
        pbc=pbc,
        fit_mask=torch.tensor([True, True, True, False, True, False], device=device),
    )

    expected_periodic = torch.tensor(
        [[0.0, 0.0, 0.0], [2.0, 1.0, 0.0]], dtype=torch.float64, device=device
    )
    torch.testing.assert_close(
        alignment.positions[:4], nonperiodic_reference, rtol=0, atol=1.0e-12
    )
    torch.testing.assert_close(
        alignment.positions[4:], expected_periodic, rtol=0, atol=1.0e-12
    )
    torch.testing.assert_close(
        alignment.rotation[1], torch.eye(3, dtype=torch.float64, device=device)
    )
    torch.testing.assert_close(
        alignment.translation[1],
        torch.tensor([2.0, -2.0, 0.0], dtype=torch.float64, device=device),
    )


def test_align_batch_positions_uses_exact_mic_for_skewed_cells() -> None:
    """Select the shortest Cartesian image rather than rounding fractions."""
    reference = torch.zeros((2, 3), dtype=torch.float64)
    mobile = torch.tensor([[0.0, 0.0, 0.0], [0.7, 0.49, 0.0]], dtype=torch.float64)
    cell = torch.tensor(
        [[[1.0, 0.0, 0.0], [0.5, 1.0, 0.0], [0.0, 0.0, 1.0]]],
        dtype=torch.float64,
    )

    alignment = align_batch_positions(
        reference,
        mobile,
        torch.zeros(2, dtype=torch.long),
        torch.tensor([2]),
        cell=cell,
        pbc=torch.ones((1, 3), dtype=torch.bool),
    )

    torch.testing.assert_close(
        alignment.positions,
        torch.tensor([[-0.1, 0.255, 0.0], [0.1, -0.255, 0.0]], dtype=torch.float64),
        rtol=0,
        atol=1.0e-12,
    )


def test_align_batch_positions_accepts_singular_nonperiodic_cell_rows() -> None:
    """Require only the cell rows selected by PBC to be independent."""
    reference = torch.zeros((2, 3), dtype=torch.float64)
    mobile = torch.tensor([[0.0, 0.0, 0.0], [2.6, 40.0, -3.0]], dtype=torch.float64)
    cell = torch.tensor(
        [[[1.0, 0.0, 0.0], [20.0, 1.0, 0.0], [0.0, 0.0, 0.0]]],
        dtype=torch.float64,
    )

    alignment = align_batch_positions(
        reference,
        mobile,
        torch.zeros(2, dtype=torch.long),
        torch.tensor([2]),
        cell=cell,
        pbc=torch.tensor([[True, False, False]]),
    )

    torch.testing.assert_close(
        alignment.positions,
        torch.tensor([[0.2, -20.0, 1.5], [-0.2, 20.0, -1.5]], dtype=torch.float64),
    )


def test_align_batch_positions_reuses_prepared_mic() -> None:
    """Avoid repeating cell-dependent lattice analysis in alignment loops."""
    positions = torch.zeros((2, 3), dtype=torch.float64)
    cell = torch.eye(3, dtype=torch.float64).unsqueeze(0)
    pbc = torch.ones((1, 3), dtype=torch.bool)
    prepared = prepare_mic(cell, pbc)

    with patch(
        "nvalchemi.dynamics.paths._geometry.prepare_mic",
        side_effect=AssertionError("MIC geometry was prepared again"),
    ):
        alignment = align_batch_positions(
            positions,
            positions,
            torch.zeros(2, dtype=torch.long),
            torch.tensor([2]),
            cell=cell,
            pbc=pbc,
            prepared_mic=prepared,
        )

    assert torch.equal(alignment.positions, positions)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_align_batch_positions_applies_zero_covariance_fallback_per_graph(
    dtype: torch.dtype,
    device: str,
) -> None:
    """Coincident and single-atom graphs get identity without affecting other fits."""
    reference, mobile, batch_idx, counts = _nonperiodic_batched_pair(dtype, device)
    reference[:4].zero_()
    mobile[:4].fill_(2.0)
    reference = torch.cat((reference, reference.new_tensor([[1.0, 2.0, 3.0]])))
    mobile = torch.cat((mobile, mobile.new_tensor([[5.0, -1.0, 7.0]])))
    batch_idx = torch.cat((batch_idx, batch_idx.new_tensor([2])))
    counts = torch.cat((counts, counts.new_tensor([1])))

    alignment = align_batch_positions(reference, mobile, batch_idx, counts)

    tolerance = 1.0e-5 if dtype == torch.float32 else 1.0e-12
    identity = torch.eye(3, dtype=dtype, device=device)
    torch.testing.assert_close(
        alignment.rotation[[0, 2]],
        identity.expand(2, -1, -1),
        rtol=0,
        atol=tolerance,
    )
    assert not torch.allclose(alignment.rotation[1], identity)
    torch.testing.assert_close(alignment.positions, reference, rtol=0, atol=tolerance)
    transformed = (
        torch.bmm(alignment.rotation[batch_idx], mobile.unsqueeze(-1)).squeeze(-1)
        + alignment.translation[batch_idx]
    )
    torch.testing.assert_close(transformed, alignment.positions, rtol=0, atol=tolerance)


@pytest.mark.parametrize(
    ("batch_idx", "counts", "message"),
    [
        (torch.tensor([0, 0, 1]), torch.tensor([1, 2]), "must agree"),
        (torch.tensor([0, -1, 1]), torch.tensor([1, 2]), "must be in"),
        (torch.tensor([0, 0, 0]), torch.tensor([3, 0]), "at least one atom"),
    ],
)
def test_align_batch_positions_validates_graph_membership(
    batch_idx: torch.Tensor, counts: torch.Tensor, message: str
) -> None:
    """Reject inconsistent, out-of-range, and empty graph descriptions."""
    positions = torch.zeros(3, 3)

    with pytest.raises(ValueError, match=message):
        align_batch_positions(positions, positions, batch_idx, counts)


def test_align_batch_positions_requires_independent_periodic_vectors() -> None:
    """Report missing cells and dependent vectors only for periodic graphs."""
    positions = torch.zeros(2, 3)
    batch_idx = torch.tensor([0, 1])
    counts = torch.ones(2, dtype=torch.long)
    pbc = torch.tensor([[False, False, False], [True, False, False]])

    with pytest.raises(ValueError, match="cell is required"):
        align_batch_positions(positions, positions, batch_idx, counts, pbc=pbc)

    with pytest.raises(ValueError, match="numerically dependent"):
        align_batch_positions(
            positions,
            positions,
            batch_idx,
            counts,
            cell=torch.zeros(2, 3, 3),
            pbc=pbc,
        )


@pytest.mark.parametrize(
    ("fit_mask", "error_type", "message"),
    [
        (torch.ones(3, dtype=torch.bool), ValueError, "shape"),
        (torch.ones(4), TypeError, "bool dtype"),
        (
            torch.tensor([True, True, False, False]),
            ValueError,
            "at least one atom in every graph",
        ),
    ],
)
@pytest.mark.parametrize("skip_extra_checks", [False, True])
def test_align_batch_positions_validates_fit_mask(
    fit_mask: torch.Tensor,
    error_type: type[Exception],
    message: str,
    skip_extra_checks: bool,
) -> None:
    """Validate fitting masks on both structural-validation paths."""
    positions = torch.zeros(4, 3)
    batch_idx = torch.tensor([0, 0, 1, 1])
    counts = torch.tensor([2, 2])

    with pytest.raises(error_type, match=message):
        align_batch_positions(
            positions,
            positions,
            batch_idx,
            counts,
            fit_mask=fit_mask,
            skip_extra_checks=skip_extra_checks,
        )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("masked", [False, True])
@pytest.mark.parametrize("reflected", [False, True])
def test_align_batch_positions_matches_kabsch_for_deformed_graphs(
    dtype: torch.dtype, device: str, masked: bool, reflected: bool
) -> None:
    """Match an independent proper Kabsch fit for noisy, optionally mirrored graphs."""
    rng = np.random.default_rng(42)
    references, mobiles, masks, expected = [], [], [], []
    for count in (7, 11):
        reference = rng.normal(size=(count, 3)) * [1.0, 2.0, 3.0]
        axis = np.array([1.0, 2.0, -3.0])
        axis /= np.linalg.norm(axis)
        # Use an oblique axis so the rotation mixes all three coordinates.
        rotation = 2 * np.outer(axis, axis) - np.eye(3)
        mobile = reference @ rotation + rng.normal(scale=0.15, size=reference.shape)
        if reflected:
            mobile[:, 0] *= -1
        mobile += [4.0, -2.0, 1.0]
        reference_tensor = torch.tensor(reference, dtype=dtype, device=device)
        mobile_tensor = torch.tensor(mobile, dtype=dtype, device=device)
        # Compare solvers on identical rounded inputs, with a float64 CPU oracle.
        reference = reference_tensor.cpu().double().numpy()
        mobile = mobile_tensor.cpu().double().numpy()
        selected = np.ones(count, dtype=bool)
        if masked:
            selected[-2:] = False
        reference_center = reference[selected].mean(axis=0)
        mobile_center = mobile[selected].mean(axis=0)
        covariance = (mobile[selected] - mobile_center).T @ (
            reference[selected] - reference_center
        )
        u, _, vh = np.linalg.svd(covariance)
        correction = np.eye(3)
        correction[-1, -1] = np.sign(np.linalg.det(u @ vh))
        if reflected:
            assert correction[-1, -1] == -1
        expected.append(
            (mobile - mobile_center) @ u @ correction @ vh + reference_center
        )
        references.append(reference_tensor)
        mobiles.append(mobile_tensor)
        masks.append(torch.tensor(selected, device=device))

    counts = torch.tensor([7, 11], device=device)
    batch_idx = torch.repeat_interleave(torch.arange(2, device=device), counts)
    reference = torch.cat(references)
    mobile = torch.cat(mobiles)
    fit_mask = torch.cat(masks) if masked else None
    alignment = align_batch_positions(
        reference, mobile, batch_idx, counts, fit_mask=fit_mask
    )

    tolerance = 1.0e-5 if dtype == torch.float32 else 1.0e-12
    torch.testing.assert_close(
        alignment.positions,
        torch.tensor(np.concatenate(expected), dtype=dtype, device=device),
        rtol=0,
        atol=tolerance,
    )
    torch.testing.assert_close(
        alignment.rotation @ alignment.rotation.transpose(-2, -1),
        torch.eye(3, dtype=dtype, device=device).expand(2, -1, -1),
        rtol=0,
        atol=tolerance,
    )
    torch.testing.assert_close(
        torch.linalg.det(alignment.rotation),
        torch.ones(2, dtype=dtype, device=device),
        rtol=0,
        atol=tolerance,
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("geometry", ["half_turn", "collinear"])
def test_align_batch_positions_handles_half_turns_and_degenerate_geometry(
    dtype: torch.dtype, device: str, geometry: str
) -> None:
    """Recover positions without requiring a unique rotation for rank-deficient fits."""
    if geometry == "half_turn":
        coordinates = [[0, 0, 0], [1, 0, 0], [0, 2, 0], [0, 0, 3]]
    else:
        coordinates = [[-2, 0, 0], [0, 0, 0], [1, 0, 0], [3, 0, 0]]
    reference = torch.tensor(coordinates, dtype=dtype, device=device)
    axis = torch.tensor([1.0, 2.0, -3.0], dtype=dtype, device=device)
    axis /= torch.linalg.vector_norm(axis)
    rotation = 2 * torch.outer(axis, axis) - torch.eye(3, dtype=dtype, device=device)
    mobile = reference @ rotation.T + reference.new_tensor([4.0, -3.0, 2.0])
    count = reference.shape[0]

    alignment = align_batch_positions(
        reference,
        mobile,
        torch.zeros(count, dtype=torch.long, device=device),
        torch.tensor([count], device=device),
    )

    tolerance = 1.0e-5 if dtype == torch.float32 else 1.0e-12
    torch.testing.assert_close(alignment.positions, reference, rtol=0, atol=tolerance)
    torch.testing.assert_close(
        mobile @ alignment.rotation[0].T + alignment.translation[0],
        alignment.positions,
        rtol=0,
        atol=tolerance,
    )
    torch.testing.assert_close(
        alignment.rotation[0] @ alignment.rotation[0].T,
        torch.eye(3, dtype=dtype, device=device),
        rtol=0,
        atol=tolerance,
    )
    torch.testing.assert_close(
        torch.linalg.det(alignment.rotation[0]),
        reference.new_tensor(1.0),
        rtol=0,
        atol=tolerance,
    )
