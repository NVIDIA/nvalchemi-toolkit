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
"""Tests for tensor operations shared by reaction-path methods."""

from __future__ import annotations

import pytest
import torch

from nvalchemi.dynamics.paths._ops.torch_paths import path_energy_stats


@pytest.mark.parametrize(
    ("case", "match"),
    [
        ("pointer-rank", "must be one-dimensional"),
        ("energy-dtype", "image_energies must have dtype"),
        ("pointer-dtype", "path_ptr must have dtype int32"),
        ("energy-layout", "must be contiguous"),
    ],
)
def test_path_energy_stats_rejects_invalid_inputs(case: str, match: str) -> None:
    energies = torch.tensor([0.0, 1.0, 0.0, 2.0, 3.0, 2.0])
    path_ptr = torch.tensor([0, 3, 6], dtype=torch.int32)
    if case == "pointer-rank":
        path_ptr = path_ptr.unsqueeze(0)
    elif case == "energy-dtype":
        energies = energies.to(torch.int64)
    elif case == "pointer-dtype":
        path_ptr = path_ptr.to(torch.int64)
    else:
        energies = torch.zeros(12)[::2]

    with pytest.raises(ValueError, match=match):
        path_energy_stats(
            energies,
            path_ptr,
            torch.empty(2),
            torch.empty(2),
            torch.empty(2, dtype=torch.int32),
        )


def test_path_energy_stats_rejects_zero_paths() -> None:
    with pytest.raises(ValueError, match="at least one path"):
        path_energy_stats(
            torch.empty(0),
            torch.tensor([0], dtype=torch.int32),
            torch.empty(0),
            torch.empty(0),
            torch.empty(0, dtype=torch.int32),
        )


@pytest.mark.parametrize(
    ("case", "match"),
    [
        ("float-shape", "endpoint_reference_energy must have shape"),
        ("float-layout", "endpoint_reference_energy must match"),
        ("index-shape", "highest_interior_image_idx must have shape"),
        ("index-dtype", "highest_interior_image_idx must be contiguous int32"),
    ],
)
def test_path_energy_stats_rejects_invalid_outputs(case: str, match: str) -> None:
    energies = torch.tensor([0.0, 1.0, 0.0, 2.0, 3.0, 2.0])
    path_ptr = torch.tensor([0, 3, 6], dtype=torch.int32)
    endpoint_reference = torch.empty(2)
    interior_energy = torch.empty(2)
    interior_idx = torch.empty(2, dtype=torch.int32)
    if case == "float-shape":
        endpoint_reference = endpoint_reference[:-1]
    elif case == "float-layout":
        endpoint_reference = torch.empty(4)[::2]
    elif case == "index-shape":
        interior_idx = interior_idx[:-1]
    else:
        interior_idx = interior_idx.to(torch.int64)

    with pytest.raises(ValueError, match=match):
        path_energy_stats(
            energies,
            path_ptr,
            endpoint_reference,
            interior_energy,
            interior_idx,
        )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_path_energy_stats_matches_expected_values(
    dtype: torch.dtype,
    device: str,
) -> None:
    """Reduce ragged paths consistently on every supported device and dtype."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    energies = torch.tensor(
        [5.0, 3.0, 3.0, 1.0, 2.0, 4.0, 7.0, 6.0, 8.0],
        dtype=dtype,
        device=device,
    )
    path_ptr = torch.tensor([0, 4, 9], dtype=torch.int32, device=device)
    endpoint_reference = torch.empty(2, dtype=dtype, device=device)
    interior_energy = torch.empty(2, dtype=dtype, device=device)
    interior_idx = torch.empty(2, dtype=torch.int32, device=device)

    path_energy_stats(
        energies,
        path_ptr,
        endpoint_reference,
        interior_energy,
        interior_idx,
    )

    assert endpoint_reference.cpu().tolist() == [1.0, 2.0]
    assert interior_energy.cpu().tolist() == [3.0, 7.0]
    assert interior_idx.cpu().tolist() == [1, 6]


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_path_energy_stats_resolves_strided_ties_to_lowest_index(
    device: str,
) -> None:
    """Select the lower index when tied candidates belong to different lanes."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    energies = torch.zeros(35, device=device)
    energies[31:33] = 5.0
    path_ptr = torch.tensor([0, 35], dtype=torch.int32, device=device)
    endpoint_reference = torch.empty(1, device=device)
    interior_energy = torch.empty(1, device=device)
    interior_idx = torch.empty(1, dtype=torch.int32, device=device)

    path_energy_stats(
        energies,
        path_ptr,
        endpoint_reference,
        interior_energy,
        interior_idx,
    )

    assert endpoint_reference.item() == 0.0
    assert interior_energy.item() == 5.0
    assert interior_idx.item() == 31


def test_path_energy_stats_captures_fullgraph() -> None:
    """Execute the shared custom operation through full-graph compilation."""
    energies = torch.tensor([0.0, 2.0, 1.0, 0.0])
    path_ptr = torch.tensor([0, 4], dtype=torch.int32)
    endpoint_reference = torch.empty(1)
    interior_energy = torch.empty(1)
    interior_idx = torch.empty(1, dtype=torch.int32)
    compiled = torch.compile(path_energy_stats, backend="eager", fullgraph=True)

    compiled(
        energies,
        path_ptr,
        endpoint_reference,
        interior_energy,
        interior_idx,
    )

    assert endpoint_reference.tolist() == [0.0]
    assert interior_energy.tolist() == [2.0]
    assert interior_idx.tolist() == [1]
