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
"""Tests for SpatialPartitioner."""

from __future__ import annotations

import math

import pytest
import torch
from torch.distributed import DeviceMesh  # noqa: F401 — needed to resolve forward ref

from nvalchemi.distributed.config import DomainConfig
from nvalchemi.distributed.partitioner import SpatialPartitioner

# Resolve the forward reference for DeviceMesh so pydantic can validate.
DomainConfig.model_rebuild(_types_namespace={"DeviceMesh": DeviceMesh})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_orthorhombic_cell(lx: float, ly: float, lz: float) -> torch.Tensor:
    """Create a diagonal (orthorhombic) cell matrix."""
    return torch.diag(torch.tensor([lx, ly, lz], dtype=torch.float64))


def _make_skew_cell(a: float, c: float, gamma_deg: float = 120.0) -> torch.Tensor:
    """Create a hexagonal (``gamma != 90``) cell matrix with rows as lattice
    vectors. The inverse is non-symmetric, so ``inv(cell)`` and ``inv(cell).T``
    give different fractional coordinates — unlike a diagonal cell."""
    g = math.radians(gamma_deg)
    return torch.tensor(
        [
            [a, 0.0, 0.0],
            [a * math.cos(g), a * math.sin(g), 0.0],
            [0.0, 0.0, c],
        ],
        dtype=torch.float64,
    )


def _make_partitioner(
    cell_matrix: torch.Tensor,
    cutoff: float,
    world_size: int = 1,
    pbc: torch.Tensor | None = None,
    grid_dims: tuple[int, int, int] | None = None,
) -> SpatialPartitioner:
    """Build a SpatialPartitioner with a simple DomainConfig (no real mesh)."""
    if pbc is None:
        pbc = torch.tensor([True, True, True])
    config = DomainConfig(cutoff=cutoff, grid_dims=grid_dims)
    # Monkey-patch world_size since we don't have a real DeviceMesh in tests.
    part = SpatialPartitioner.__new__(SpatialPartitioner)
    part.config = config
    part.cell_matrix = cell_matrix
    part.pbc = pbc
    part.world_size = world_size

    # Reproduce __init__ logic after world_size is set.
    if config.grid_dims is not None:
        part.cells_per_dim = config.grid_dims
    else:
        part.cells_per_dim = SpatialPartitioner._compute_cells_per_dim(
            cell_matrix, config.cutoff
        )

    total_cells = part.cells_per_dim[0] * part.cells_per_dim[1] * part.cells_per_dim[2]
    if total_cells < world_size:
        part.cells_per_dim = SpatialPartitioner.refine_grid_for_ranks(
            part.cells_per_dim, world_size
        )

    part.rank_grid = SpatialPartitioner.compute_rank_grid(
        part.cells_per_dim, world_size
    )
    if config.grid_dims is None:
        part.cells_per_dim = SpatialPartitioner.balance_cells_for_ranks(
            part.cells_per_dim, part.rank_grid
        )
    part._neighbor_ranks = part._compute_all_neighbor_ranks()
    # __init__ caches the cell-matrix inverse for ``assign_atoms_to_ranks``;
    # mirror it here since this helper bypasses __init__.
    part._inv_cell = torch.linalg.inv(
        cell_matrix.squeeze(0) if cell_matrix.ndim == 3 else cell_matrix
    )
    return part


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestComputeRankGrid:
    """Tests for compute_rank_grid static method."""

    def test_rank_grid_cubic(self):
        """8 GPUs, cubic cell grid -> [2, 2, 2]."""
        grid = SpatialPartitioner.compute_rank_grid((20, 20, 20), 8)
        assert grid == (2, 2, 2)

    def test_rank_grid_elongated(self):
        """8 GPUs, elongated 10x10x40 cells.

        Should prefer a factorization that aligns more ranks along the
        long z dimension. The surface-area minimizer should pick the
        best among all valid 3-factor factorizations of 8.
        """
        grid = SpatialPartitioner.compute_rank_grid((10, 10, 40), 8)
        Px, Py, Pz = grid
        assert Px * Py * Pz == 8

        # Verify it actually minimises surface area among all factorizations.
        Nx, Ny, Nz = 10, 10, 40
        best_surface = float("inf")
        best = None
        for px in range(1, 9):
            if 8 % px != 0:
                continue
            for py in range(1, 8 // px + 1):
                if (8 // px) % py != 0:
                    continue
                pz = 8 // (px * py)
                dx, dy, dz = Nx / px, Ny / py, Nz / pz
                s = 2.0 * (dx * dy + dy * dz + dx * dz)
                if s < best_surface:
                    best_surface = s
                    best = (px, py, pz)
        assert grid == best

    def test_rank_grid_single_gpu(self):
        """1 GPU -> (1, 1, 1)."""
        grid = SpatialPartitioner.compute_rank_grid((5, 5, 5), 1)
        assert grid == (1, 1, 1)

    def test_rank_grid_prime(self):
        """Prime world_size -> one dimension gets all ranks."""
        grid = SpatialPartitioner.compute_rank_grid((10, 10, 10), 7)
        Px, Py, Pz = grid
        assert Px * Py * Pz == 7
        # 7 is prime, so exactly one factor is 7 and the others are 1.
        assert sorted([Px, Py, Pz]) == [1, 1, 7]


class TestBalanceCellsForRanks:
    """balance_cells_for_ranks rounds partition-axis cell counts to a
    multiple of the rank factor (rounding down, floored at the factor)."""

    def test_three_cells_two_ranks_rounds_to_two(self):
        # 3 cells / 2 ranks would split 2:1 → balance to 2.
        assert SpatialPartitioner.balance_cells_for_ranks((3, 3, 3), (2, 1, 1)) == (
            2,
            3,
            3,
        )

    def test_single_rank_axis_unchanged(self):
        assert SpatialPartitioner.balance_cells_for_ranks((5, 5, 5), (1, 1, 1)) == (
            5,
            5,
            5,
        )

    def test_already_divisible_unchanged(self):
        assert SpatialPartitioner.balance_cells_for_ranks((4, 6, 8), (2, 2, 2)) == (
            4,
            6,
            8,
        )

    def test_floors_at_rank_factor(self):
        # Ni < Pi can't round down below Pi (each rank keeps >= 1 cell).
        assert SpatialPartitioner.balance_cells_for_ranks((1, 1, 1), (2, 1, 1)) == (
            2,
            1,
            1,
        )

    def test_balanced_assignment_cubic_two_ranks(self):
        # End-to-end: a cubic box across 2 ranks splits ~50/50, not 2:1.
        cell = torch.eye(3) * 22.96  # ~ BCC Fe 8^3 box
        coords = torch.linspace(0.3, 22.6, 10)
        gx, gy, gz = torch.meshgrid(coords, coords, coords, indexing="ij")
        positions = torch.stack([gx.flatten(), gy.flatten(), gz.flatten()], dim=-1)
        part = _make_partitioner(cell, cutoff=6.0, world_size=2)
        ranks = part.assign_atoms_to_ranks(positions)
        n0 = int((ranks == 0).sum())
        n1 = int((ranks == 1).sum())
        # Balanced split: neither rank owns more than ~60%.
        assert min(n0, n1) / max(n0, n1) > 0.8, f"imbalanced: {n0} vs {n1}"


class TestCellToRankRoundtrip:
    """cell_to_rank -> rank_to_cell_bounds round-trip consistency."""

    @pytest.mark.parametrize(
        "cells_per_dim, world_size",
        [
            ((6, 6, 6), 8),
            ((10, 10, 40), 8),
            ((5, 5, 5), 4),
            ((7, 3, 5), 6),
        ],
    )
    def test_cell_to_rank_roundtrip(self, cells_per_dim, world_size):
        """For every cell, the owning rank's bounds contain that cell."""
        cell = _make_orthorhombic_cell(50.0, 50.0, 50.0)
        part = _make_partitioner(
            cell, cutoff=5.0, world_size=world_size, grid_dims=cells_per_dim
        )

        Nx, Ny, Nz = part.cells_per_dim
        for ix in range(Nx):
            for iy in range(Ny):
                for iz in range(Nz):
                    rank = part.cell_to_rank(ix, iy, iz)
                    lo, hi = part.rank_to_cell_bounds(rank)
                    assert lo[0] <= ix < hi[0], f"x: {ix} not in [{lo[0]}, {hi[0]})"
                    assert lo[1] <= iy < hi[1], f"y: {iy} not in [{lo[1]}, {hi[1]})"
                    assert lo[2] <= iz < hi[2], f"z: {iz} not in [{lo[2]}, {hi[2]})"


class TestNeighborRanks:
    """Tests for get_neighbor_ranks."""

    def test_neighbor_ranks_pbc(self):
        """2x2x2 grid, full PBC: every rank has exactly 26 neighbors."""
        cell = _make_orthorhombic_cell(20.0, 20.0, 20.0)
        part = _make_partitioner(cell, cutoff=5.0, world_size=8, grid_dims=(4, 4, 4))
        # Rank grid should be (2, 2, 2) for 8 GPUs.
        assert part.rank_grid == (2, 2, 2)

        for rank in range(8):
            neighbors = part.get_neighbor_ranks(rank)
            # With PBC wrapping on a 2x2x2 grid, each rank sees all
            # 26 neighbor slots.  However some neighbor coordinates may
            # map back to the same rank (e.g., wrapping in a size-2 dim
            # gives coord 0 -> neighbor 1 and coord -1 wraps to 1).
            # All 7 *other* ranks should appear since 2^3 = 8 unique.
            assert len(neighbors) == 7, (
                f"Rank {rank} has {len(neighbors)} neighbors, expected 7 "
                f"(all other ranks in a 2x2x2 PBC grid)."
            )

    def test_neighbor_ranks_no_pbc(self):
        """2x2x2 grid, no PBC: corner rank should have 7 neighbors."""
        cell = _make_orthorhombic_cell(20.0, 20.0, 20.0)
        pbc = torch.tensor([False, False, False])
        part = _make_partitioner(
            cell, cutoff=5.0, world_size=8, grid_dims=(4, 4, 4), pbc=pbc
        )
        assert part.rank_grid == (2, 2, 2)

        # Corner rank 0 = (0,0,0): only positive neighbors are valid.
        neighbors_0 = part.get_neighbor_ranks(0)
        assert len(neighbors_0) == 7, (
            f"Corner rank 0 has {len(neighbors_0)} neighbors, expected 7 "
            "(3 face + 3 edge + 1 corner with no PBC)."
        )

        # A body-centered rank doesn't exist in a 2x2x2 grid — all ranks
        # are corners. Verify that all ranks have 7 neighbors (since
        # non-PBC 2x2x2 means each rank is in a corner).
        for rank in range(8):
            assert len(part.get_neighbor_ranks(rank)) == 7

    def test_neighbor_ranks_mixed_pbc(self):
        """3x3x3 rank grid, PBC only along z."""
        cell = _make_orthorhombic_cell(30.0, 30.0, 30.0)
        pbc = torch.tensor([False, False, True])
        part = _make_partitioner(
            cell, cutoff=3.0, world_size=27, grid_dims=(9, 9, 9), pbc=pbc
        )
        assert part.rank_grid == (3, 3, 3)

        # Corner rank (0,0,0) = rank 0.
        neighbors = part.get_neighbor_ranks(0)
        # x: only +1 valid (non-PBC) -> 2 choices {0, 1}
        # y: only +1 valid (non-PBC) -> 2 choices {0, 1}
        # z: all 3 valid (PBC wraps) -> 3 choices {0, 1, 2}
        # Total neighbor slots = 2*2*3 - 1 (exclude self) = 11
        assert len(neighbors) == 11


class TestAssignAtomsToRanks:
    """Tests for the vectorized assign_atoms_to_ranks."""

    def test_assign_atoms_known_box(self):
        """Known positions in a known orthorhombic box."""
        cell = _make_orthorhombic_cell(20.0, 20.0, 20.0)
        # cutoff=10 -> cells_per_dim = (2, 2, 2) for a 20A box.
        part = _make_partitioner(cell, cutoff=10.0, world_size=8)
        assert part.cells_per_dim == (2, 2, 2)
        assert part.rank_grid == (2, 2, 2)

        # Place atoms at known cell centres.
        positions = torch.tensor(
            [
                [5.0, 5.0, 5.0],  # cell (0,0,0)
                [15.0, 5.0, 5.0],  # cell (1,0,0)
                [5.0, 15.0, 5.0],  # cell (0,1,0)
                [15.0, 15.0, 15.0],  # cell (1,1,1)
            ],
            dtype=torch.float64,
        )

        ranks = part.assign_atoms_to_ranks(positions)
        assert ranks.shape == (4,)

        # Manually compute expected ranks.
        expected = []
        for pos in positions:
            frac = pos / 20.0  # orthorhombic -> fractional = pos / L
            ix = int(math.floor(frac[0].item() * 2))
            iy = int(math.floor(frac[1].item() * 2))
            iz = int(math.floor(frac[2].item() * 2))
            expected.append(part.cell_to_rank(ix, iy, iz))

        torch.testing.assert_close(ranks, torch.tensor(expected, dtype=ranks.dtype))

    def test_assign_atoms_vectorized_matches_scalar(self):
        """Vectorized and scalar cell_to_rank produce the same results."""
        cell = _make_orthorhombic_cell(30.0, 30.0, 60.0)
        part = _make_partitioner(cell, cutoff=5.0, world_size=8)

        torch.manual_seed(42)
        positions = torch.rand(200, 3, dtype=torch.float64) * torch.tensor(
            [30.0, 30.0, 60.0], dtype=torch.float64
        )

        ranks = part.assign_atoms_to_ranks(positions)

        # Compute expected via scalar path.
        inv_cell_T = torch.linalg.inv(cell).T
        cells_t = torch.tensor(part.cells_per_dim, dtype=torch.float64)
        cells_int = torch.tensor(part.cells_per_dim, dtype=torch.int64)

        frac = positions @ inv_cell_T
        cell_coords = torch.floor(frac * cells_t).to(torch.int64)
        cell_coords = cell_coords % cells_int  # PBC wrap

        expected = []
        for i in range(len(positions)):
            ix, iy, iz = cell_coords[i].tolist()
            expected.append(part.cell_to_rank(int(ix), int(iy), int(iz)))

        torch.testing.assert_close(ranks, torch.tensor(expected, dtype=ranks.dtype))

    def test_assign_atoms_skew_cell_uses_inv_not_inv_transpose(self):
        """On a skew (hex) cell, atom assignment must use ``inv(cell)`` for the
        fractional coordinates, not ``inv(cell).T``.

        Regression for the skew-cell fractional-coordinate bug: on an
        orthorhombic cell the two conventions coincide, so this is only
        observable on a non-symmetric inverse. Assignment must match the
        ``inv(cell)`` reference and must *differ* from the ``inv(cell).T``
        reference (proving the skew cell exercises the distinction).
        """
        # 6x6x6 hex supercell, cutoff 5 Å -> a real multi-cell grid.
        cell = _make_skew_cell(a=4.9019 * 6, c=5.3988 * 6, gamma_deg=120.0)
        part = _make_partitioner(cell, cutoff=5.0, world_size=8)

        torch.manual_seed(0)
        # Fill the (skewed) box via fractional coords so every atom is inside.
        frac_rand = torch.rand(300, 3, dtype=torch.float64)
        positions = frac_rand @ cell

        ranks = part.assign_atoms_to_ranks(positions)

        cells_t = torch.tensor(part.cells_per_dim, dtype=torch.float64)
        cells_int = torch.tensor(part.cells_per_dim, dtype=torch.int64)

        def _ranks_from_frac(frac: torch.Tensor) -> torch.Tensor:
            cell_coords = torch.floor(frac * cells_t).to(torch.int64) % cells_int
            return torch.tensor(
                [
                    part.cell_to_rank(*cell_coords[i].tolist())
                    for i in range(len(positions))
                ],
                dtype=ranks.dtype,
            )

        inv = torch.linalg.inv(cell)
        expected_correct = _ranks_from_frac(positions @ inv)
        expected_wrong = _ranks_from_frac(positions @ inv.T)

        torch.testing.assert_close(ranks, expected_correct)
        # The skew cell must actually distinguish the two conventions, else the
        # test would silently pass even under the buggy ``.T`` path.
        assert not torch.equal(expected_correct, expected_wrong)


class TestRefineGrid:
    """Tests for refine_grid_for_ranks."""

    def test_refine_2x2x2_to_16(self):
        """2x2x2 = 8 cells, 16 GPUs -> refined grid has >= 16 cells."""
        refined = SpatialPartitioner.refine_grid_for_ranks((2, 2, 2), 16)
        total = refined[0] * refined[1] * refined[2]
        assert total >= 16

    def test_refine_no_op_when_sufficient(self):
        """If already enough cells, refine is a no-op."""
        refined = SpatialPartitioner.refine_grid_for_ranks((4, 4, 4), 8)
        assert refined == (4, 4, 4)

    def test_refine_doubles_smallest(self):
        """Doubling strategy targets the smallest dimension first."""
        # (1, 1, 1) with world_size=8 -> should reach at least 8.
        refined = SpatialPartitioner.refine_grid_for_ranks((1, 1, 1), 8)
        total = refined[0] * refined[1] * refined[2]
        assert total >= 8
        # All dims should be equal since they start equal and we double
        # the smallest (which is always any of them when tied).
        assert refined[0] == refined[1] == refined[2] == 2

    def test_refine_asymmetric(self):
        """Asymmetric starting grid."""
        # (1, 2, 2) = 4 cells, need 8.
        refined = SpatialPartitioner.refine_grid_for_ranks((1, 2, 2), 8)
        total = refined[0] * refined[1] * refined[2]
        assert total >= 8
        # First doubling should target dim 0 (size 1).
        assert refined[0] >= 2


class TestRankToGridCoords:
    """Tests for rank_to_grid_coords."""

    def test_roundtrip(self):
        """Linearize -> decompose round-trip for all ranks."""
        cell = _make_orthorhombic_cell(30.0, 30.0, 30.0)
        part = _make_partitioner(cell, cutoff=5.0, world_size=8, grid_dims=(6, 6, 6))
        Px, Py, Pz = part.rank_grid
        total = Px * Py * Pz

        for rank in range(total):
            rx, ry, rz = part.rank_to_grid_coords(rank)
            assert rx + Px * (ry + Py * rz) == rank
