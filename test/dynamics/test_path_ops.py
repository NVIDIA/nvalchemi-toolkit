# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for tensor operations shared by reaction-path methods."""

from __future__ import annotations

import pytest
import torch

from nvalchemi.dynamics.paths._ops.torch_paths import path_energy_stats


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
