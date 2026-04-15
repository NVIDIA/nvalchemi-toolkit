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
"""
Unit tests for cell alignment custom op and AlignCellHook.

Covers:
- ``nvalchemi.dynamics._ops.cell_align.align_cell`` custom op
- ``nvalchemi.dynamics.hooks.cell_align.AlignCellHook``
- ``nvalchemi.dynamics.hooks.cell_align.align_atomic_data_cell``
"""

from __future__ import annotations

import math

import pytest
import torch

from nvalchemi.data import AtomicData, Batch
from nvalchemi.dynamics._ops.cell_align import align_cell
from nvalchemi.dynamics.base import BaseDynamics, DynamicsStage
from nvalchemi.dynamics.hooks import AlignCellHook
from nvalchemi.hooks import Hook, HookContext
from nvalchemi.models.demo import DemoModel, DemoModelWrapper

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _upper_triangular(cell: torch.Tensor, atol: float = 1e-5) -> bool:
    """Check whether a [M, 3, 3] cell tensor is upper-triangular."""
    # Lower triangle elements: (1,0), (2,0), (2,1) should be zero
    return (
        cell[:, 0, 1].abs().max() < atol
        and cell[:, 0, 2].abs().max() < atol
        and cell[:, 1, 2].abs().max() < atol
    )


def _make_rotated_cell(
    a: float = 5.0,
    b: float = 6.0,
    c: float = 7.0,
    alpha: float = 80.0,
    beta: float = 85.0,
    gamma: float = 75.0,
    dtype: torch.dtype = torch.float64,
    device: str = "cpu",
) -> torch.Tensor:
    """Build a triclinic cell from lattice parameters (not upper-triangular).

    Returns shape ``[1, 3, 3]``.
    """
    alpha_r = math.radians(alpha)
    beta_r = math.radians(beta)
    gamma_r = math.radians(gamma)

    cos_a, cos_b, cos_g = math.cos(alpha_r), math.cos(beta_r), math.cos(gamma_r)
    sin_g = math.sin(gamma_r)

    # Standard triclinic in upper-triangular form first
    ax = a
    bx = b * cos_g
    by = b * sin_g
    cx = c * cos_b
    cy = c * (cos_a - cos_b * cos_g) / sin_g
    cz = math.sqrt(max(0.0, c * c - cx * cx - cy * cy))

    cell_ut = torch.tensor(
        [[ax, 0.0, 0.0], [bx, by, 0.0], [cx, cy, cz]],
        dtype=dtype,
        device=device,
    )

    # Apply an arbitrary rotation so the cell is NOT upper-triangular
    angle = math.radians(37.0)
    cos_t, sin_t = math.cos(angle), math.sin(angle)
    rot = torch.tensor(
        [[cos_t, -sin_t, 0.0], [sin_t, cos_t, 0.0], [0.0, 0.0, 1.0]],
        dtype=dtype,
        device=device,
    )
    cell_rotated = cell_ut @ rot  # rotate lattice vectors
    return cell_rotated.unsqueeze(0)  # [1, 3, 3]


def _make_periodic_batch(
    n_graphs: int = 1,
    atoms_per_graph: int = 4,
    cell_size: float = 10.0,
    pbc: tuple[bool, bool, bool] = (True, True, True),
    dtype: torch.dtype = torch.float64,
    device: str = "cpu",
) -> Batch:
    """Create a batch with periodic boundary conditions."""
    data_list = []
    for _ in range(n_graphs):
        data = AtomicData(
            atomic_numbers=torch.tensor([6] * atoms_per_graph, dtype=torch.long),
            positions=torch.randn(atoms_per_graph, 3, dtype=dtype) * 2.0
            + cell_size / 2,
            cell=torch.eye(3, dtype=dtype).unsqueeze(0) * cell_size,
            pbc=torch.tensor([pbc]),
        )
        data_list.append(data)
    batch = Batch.from_data_list(data_list).to(device)
    total_atoms = n_graphs * atoms_per_graph
    batch["forces"] = torch.zeros(total_atoms, 3, dtype=dtype, device=device)
    batch["energy"] = torch.zeros(n_graphs, 1, dtype=dtype, device=device)
    batch["cell"] = (
        torch.eye(3, dtype=dtype, device=device).unsqueeze(0).expand(n_graphs, -1, -1)
        * cell_size
    )
    batch["pbc"] = torch.tensor([pbc], device=device).expand(n_graphs, -1)
    return batch


def _make_dynamics() -> BaseDynamics:
    return BaseDynamics(DemoModelWrapper(DemoModel()))


def _make_ctx(batch: Batch, dynamics: BaseDynamics) -> HookContext:
    converged = dynamics._last_converged
    if converged is not None:
        mask = torch.zeros(
            batch.num_graphs, dtype=torch.bool, device=batch.positions.device
        )
        mask[converged] = True
    else:
        mask = None
    return HookContext(
        batch=batch,
        step_count=dynamics.step_count,
        model=dynamics.model,
        converged_mask=mask,
        global_rank=dynamics.global_rank,
    )


# ===========================================================================
# align_cell custom op
# ===========================================================================


class TestAlignCellOp:
    """Tests for the ``align_cell`` PyTorch custom op."""

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_already_upper_triangular_is_noop(self, dtype, device: str) -> None:
        """An already upper-triangular cell should remain unchanged."""
        cell = torch.tensor(
            [[[5.0, 0.0, 0.0], [2.0, 6.0, 0.0], [1.0, 0.5, 7.0]]],
            dtype=dtype,
            device=device,
        )
        positions = torch.tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            dtype=dtype,
            device=device,
        )
        cell_before = cell.clone()

        align_cell(positions, cell)

        assert _upper_triangular(cell)
        # Lattice parameters should be preserved
        assert torch.allclose(
            torch.linalg.norm(cell[0], dim=-1),
            torch.linalg.norm(cell_before[0], dim=-1),
            atol=1e-4,
        )

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_rotated_cell_becomes_upper_triangular(self, dtype, device: str) -> None:
        """A rotated triclinic cell should be aligned to upper-triangular form."""
        cell = _make_rotated_cell(dtype=dtype, device=device)
        positions = torch.randn(4, 3, dtype=dtype, device=device)

        # Before: not upper-triangular
        assert not _upper_triangular(cell, atol=0.1)

        align_cell(positions, cell)

        # After: upper-triangular
        assert _upper_triangular(cell, atol=1e-4 if dtype == torch.float32 else 1e-8)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_lattice_parameters_preserved(self, dtype, device: str) -> None:
        """Cell alignment preserves lattice vector lengths and angles."""
        cell = _make_rotated_cell(a=5.0, b=6.0, c=7.0, dtype=dtype, device=device)
        positions = torch.randn(4, 3, dtype=dtype, device=device)

        lengths_before = torch.linalg.norm(cell[0], dim=-1)
        vol_before = torch.det(cell[0]).abs()

        align_cell(positions, cell)

        lengths_after = torch.linalg.norm(cell[0], dim=-1)
        vol_after = torch.det(cell[0]).abs()

        atol = 1e-4 if dtype == torch.float32 else 1e-8
        assert torch.allclose(lengths_before, lengths_after, atol=atol)
        assert torch.allclose(vol_before, vol_after, atol=atol)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_fractional_coordinates_preserved(self, dtype, device: str) -> None:
        """Positions should maintain their fractional coordinates after alignment."""
        cell = _make_rotated_cell(dtype=dtype, device=device)
        positions = torch.tensor(
            [[1.0, 2.0, 3.0], [2.5, 1.5, 0.5]], dtype=dtype, device=device
        )

        # Compute fractional coords before
        frac_before = positions @ torch.linalg.inv(cell[0])

        align_cell(positions, cell)

        # Compute fractional coords after
        frac_after = positions @ torch.linalg.inv(cell[0])

        atol = 1e-4 if dtype == torch.float32 else 1e-8
        assert torch.allclose(frac_before, frac_after, atol=atol)

    def test_multi_system_batch(self, device: str) -> None:
        """Alignment works for batches with multiple systems."""
        dtype = torch.float64
        cell1 = _make_rotated_cell(
            a=5.0, b=5.0, c=5.0, gamma=90.0, dtype=dtype, device=device
        )
        cell2 = _make_rotated_cell(
            a=8.0, b=8.0, c=8.0, gamma=70.0, dtype=dtype, device=device
        )
        cell = torch.cat([cell1, cell2], dim=0)  # [2, 3, 3]

        positions = torch.randn(6, 3, dtype=dtype, device=device)
        batch_idx = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.int32, device=device)

        align_cell(positions, cell, batch_idx)

        assert _upper_triangular(cell, atol=1e-8)

    def test_in_place_mutation(self, device: str) -> None:
        """Verify positions and cell tensors are modified in-place."""
        dtype = torch.float64
        cell = _make_rotated_cell(dtype=dtype, device=device)
        positions = torch.randn(4, 3, dtype=dtype, device=device)

        pos_data_ptr = positions.data_ptr()
        cell_data_ptr = cell.data_ptr()

        align_cell(positions, cell)

        assert positions.data_ptr() == pos_data_ptr
        assert cell.data_ptr() == cell_data_ptr


# ===========================================================================
# AlignCellHook
# ===========================================================================


class TestAlignCellHook:
    """Tests for :class:`AlignCellHook`."""

    def test_is_hook(self) -> None:
        hook = AlignCellHook()
        assert isinstance(hook, Hook)

    def test_default_stage_and_frequency(self) -> None:
        hook = AlignCellHook()
        assert hook.stage == DynamicsStage.BEFORE_STEP
        assert hook.frequency == 1

    def test_custom_frequency(self) -> None:
        hook = AlignCellHook(frequency=5)
        assert hook.frequency == 5

    def test_aligns_rotated_cell(self, device: str) -> None:
        """Hook aligns a rotated cell to upper-triangular form."""
        dtype = torch.float64
        batch = _make_periodic_batch(dtype=dtype, device=device)
        dynamics = _make_dynamics()

        # Replace with a rotated cell
        rotated = _make_rotated_cell(a=10.0, b=10.0, c=10.0, dtype=dtype, device=device)
        batch["cell"] = rotated

        hook = AlignCellHook()
        ctx = _make_ctx(batch, dynamics)
        hook(ctx, DynamicsStage.BEFORE_STEP)

        assert _upper_triangular(batch.cell, atol=1e-8)

    def test_noop_without_cell(self, device: str) -> None:
        """Hook is a no-op when batch has no cell attribute."""
        # Build a batch with no cell
        data = AtomicData(
            atomic_numbers=torch.tensor([6, 6], dtype=torch.long),
            positions=torch.randn(2, 3, dtype=torch.float64),
        )
        batch = Batch.from_data_list([data]).to(device)
        batch["forces"] = torch.zeros(2, 3, dtype=torch.float64, device=device)
        batch["energy"] = torch.zeros(1, 1, dtype=torch.float64, device=device)
        dynamics = _make_dynamics()

        hook = AlignCellHook()
        ctx = _make_ctx(batch, dynamics)
        hook(ctx, DynamicsStage.BEFORE_STEP)  # should not raise

    def test_noop_non_periodic(self, device: str) -> None:
        """Hook is a no-op for non-periodic systems."""
        dtype = torch.float64
        batch = _make_periodic_batch(
            pbc=(False, False, False), dtype=dtype, device=device
        )
        dynamics = _make_dynamics()

        rotated = _make_rotated_cell(a=10.0, b=10.0, c=10.0, dtype=dtype, device=device)
        batch["cell"] = rotated
        cell_before = batch.cell.clone()

        hook = AlignCellHook()
        ctx = _make_ctx(batch, dynamics)
        hook(ctx, DynamicsStage.BEFORE_STEP)

        # Cell should not be modified
        assert torch.allclose(batch.cell, cell_before)
