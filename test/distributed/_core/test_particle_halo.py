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
"""Tests for particle halo (ghost) exchange primitives.

Single-GPU tests validate ghost identification logic (fractional coords,
PBC shifts, halo region checks).  Multi-GPU tests validate the actual
exchange via indexed_all_to_all_v (skipped without multiple GPUs).
"""

from __future__ import annotations

from collections.abc import Mapping
from unittest.mock import MagicMock, patch

import pytest
import torch

from nvalchemi.distributed._core.halo_types import (
    ParticleHaloConfig,
    ParticleHaloMetadata,
    _compute_pbc_shift_vectors,
)
from nvalchemi.distributed._core.particle_halo import (
    _check_halo_region,
    _compute_ghost_masks_batched,
    _ghost_width_fractional,
    _identify_ghosts_split,
    _rank_fractional_bounds,
    particle_halo_padding,
    particle_halo_padding_multi,
    particle_halo_unpadding,
)
from nvalchemi.distributed.config import DomainConfig
from nvalchemi.distributed.partitioner import SpatialPartitioner


def _make_partitioner(
    box_length: float = 30.0,
    cutoff: float = 8.5,
    pbc: tuple[bool, bool, bool] = (True, True, True),
    world_size: int = 2,
) -> SpatialPartitioner:
    """Create a SpatialPartitioner for testing."""
    cell = torch.eye(3) * box_length
    pbc_t = torch.tensor(pbc, dtype=torch.bool)
    config = DomainConfig(
        cutoff=cutoff, mesh=MagicMock(size=MagicMock(return_value=world_size))
    )
    # Monkey-patch world_size for the partitioner's grid computation.
    with patch("torch.distributed.get_world_size", return_value=world_size):
        part = SpatialPartitioner(
            config=config, cell_matrix=cell.unsqueeze(0), pbc=pbc_t.unsqueeze(0)
        )
    return part


def _make_halo_config(
    box_length: float = 30.0,
    cutoff: float = 8.5,
    ghost_width: float = 8.5,
    pbc: tuple[bool, bool, bool] = (True, True, True),
    world_size: int = 2,
) -> ParticleHaloConfig:
    """Create a ParticleHaloConfig for testing."""
    part = _make_partitioner(
        box_length=box_length, cutoff=cutoff, pbc=pbc, world_size=world_size
    )
    mesh = MagicMock()
    mesh.get_local_rank.return_value = 0
    mesh.size.return_value = world_size
    return ParticleHaloConfig(
        ghost_width=ghost_width,
        partitioner=part,
        mesh=mesh,
    )


# ======================================================================
# PBC shift vector tests
# ======================================================================


class TestComputePBCShiftVectors:
    """Test precomputation of PBC shift vectors."""

    def test_shifts_exist_for_wrapping_pair(self):
        part = _make_partitioner(world_size=2)
        shifts = _compute_pbc_shift_vectors(part)
        # With 2 ranks along one dim and PBC, there should be shifts.
        assert len(shifts) > 0

    def test_no_shifts_without_pbc(self):
        part = _make_partitioner(pbc=(False, False, False), world_size=2)
        shifts = _compute_pbc_shift_vectors(part)
        assert len(shifts) == 0

    def test_shift_magnitude_equals_box_length(self):
        box = 30.0
        part = _make_partitioner(box_length=box, world_size=2)
        shifts = _compute_pbc_shift_vectors(part)
        for (sender, receiver), shift_list in shifts.items():  # noqa: PERF102
            for shift in shift_list:
                # Each shift component should be 0 or ±box_length.
                for d in range(3):
                    assert shift[d].abs().item() == pytest.approx(0.0) or shift[
                        d
                    ].abs().item() == pytest.approx(box, abs=0.01)


# ======================================================================
# Ghost identification tests
# ======================================================================


class TestCheckHaloRegion:
    """Test the halo region check function."""

    def test_atom_in_halo(self):
        frac_pos = torch.tensor([[0.48, 0.5, 0.5]])  # near lo boundary of [0.5, 1.0]
        frac_lo = torch.tensor([0.5, 0.0, 0.0])
        frac_hi = torch.tensor([1.0, 1.0, 1.0])
        gw = torch.tensor([0.1, 0.1, 0.1])
        mask = _check_halo_region(frac_pos, frac_lo, frac_hi, gw)
        assert mask[0].item() is True

    def test_atom_in_core(self):
        frac_pos = torch.tensor([[0.75, 0.5, 0.5]])  # deep inside [0.5, 1.0]
        frac_lo = torch.tensor([0.5, 0.0, 0.0])
        frac_hi = torch.tensor([1.0, 1.0, 1.0])
        gw = torch.tensor([0.1, 0.1, 0.1])
        mask = _check_halo_region(frac_pos, frac_lo, frac_hi, gw)
        assert mask[0].item() is False  # in core, not halo

    def test_atom_outside(self):
        frac_pos = torch.tensor([[0.1, 0.5, 0.5]])  # far from [0.5, 1.0]
        frac_lo = torch.tensor([0.5, 0.0, 0.0])
        frac_hi = torch.tensor([1.0, 1.0, 1.0])
        gw = torch.tensor([0.1, 0.1, 0.1])
        mask = _check_halo_region(frac_pos, frac_lo, frac_hi, gw)
        assert mask[0].item() is False

    def test_batch_of_atoms(self):
        frac_pos = torch.tensor(
            [
                [0.48, 0.5, 0.5],  # in halo
                [0.75, 0.5, 0.5],  # in core
                [0.1, 0.5, 0.5],  # outside
            ]
        )
        frac_lo = torch.tensor([0.5, 0.0, 0.0])
        frac_hi = torch.tensor([1.0, 1.0, 1.0])
        gw = torch.tensor([0.1, 0.1, 0.1])
        mask = _check_halo_region(frac_pos, frac_lo, frac_hi, gw)
        assert mask.tolist() == [True, False, False]


class TestIdentifyGhostsSplit:
    """Test ghost identification with direct and PBC masks."""

    def test_direct_ghosts_near_boundary(self):
        config = _make_halo_config(box_length=30.0, ghost_width=8.5, world_size=2)
        # Place atom near the boundary between rank 0 [0, 15) and rank 1 [15, 30).
        positions = torch.tensor([[14.0, 15.0, 15.0]])  # near z=15 boundary
        direct_mask, pbc_list = _identify_ghosts_split(positions, 1, config)
        # This atom should be a ghost for rank 1.
        assert direct_mask.any().item()

    def test_center_atom_not_ghost(self):
        config = _make_halo_config(box_length=30.0, ghost_width=3.0, world_size=2)
        # Deep in the interior of rank 0's domain, away from every boundary.
        # The 2-rank grid partitions along Z (rank_grid (1,1,2), balanced
        # boundary at z=15), so an interior atom must sit well inside z<15 —
        # not on the partition plane.
        positions = torch.tensor([[7.5, 7.5, 7.5]])  # deep in rank 0's domain
        direct_mask, pbc_list = _identify_ghosts_split(positions, 1, config)
        combined = direct_mask
        for m, _ in pbc_list:
            combined = combined | m
        assert not combined.any().item()

    def test_empty_positions(self):
        config = _make_halo_config(world_size=2)
        positions = torch.zeros(0, 3)
        direct_mask, pbc_list = _identify_ghosts_split(positions, 1, config)
        assert direct_mask.shape == (0,)


class TestComputeGhostMasksBatched:
    """Test batched ghost mask computation."""

    def test_returns_dict_for_all_neighbors(self):
        config = _make_halo_config(world_size=2)
        positions = torch.rand(50, 3) * 15.0  # all in rank 0's domain
        masks = _compute_ghost_masks_batched(positions, config)
        assert isinstance(masks, dict)
        for nr in config.neighbor_ranks:
            assert nr in masks


class TestGhostWidthFractional:
    """Test fractional ghost width computation."""

    def test_orthorhombic(self):
        part = _make_partitioner(box_length=30.0)
        gw = _ghost_width_fractional(part, 8.5)
        # For orthorhombic cell, fractional width = ghost_width / box_length.
        expected = 8.5 / 30.0
        assert gw[0].item() == pytest.approx(expected, rel=1e-4)
        assert gw[1].item() == pytest.approx(expected, rel=1e-4)
        assert gw[2].item() == pytest.approx(expected, rel=1e-4)


class TestRankFractionalBounds:
    """Test fractional bounds computation."""

    def test_two_ranks(self):
        part = _make_partitioner(box_length=30.0, world_size=2)
        lo0, hi0 = _rank_fractional_bounds(part, 0)
        lo1, hi1 = _rank_fractional_bounds(part, 1)
        # One dimension should be split, others should be [0, 1].
        # Find the split dimension.
        for d in range(3):
            if hi0[d] < 0.99:
                # This is the split dimension.
                assert hi0[d].item() == pytest.approx(lo1[d].item(), abs=0.01)
                break


# ======================================================================
# Public API tests (single-process — no dist)
# ======================================================================


class TestParticleHaloPaddingSingleProcess:
    """Test particle_halo_padding without torch.distributed initialized."""

    def test_returns_positions_unchanged(self):
        config = _make_halo_config(world_size=2)
        positions = torch.rand(10, 3)
        padded, meta = particle_halo_padding(positions, config)
        assert torch.equal(padded, positions)
        assert meta.n_owned == 10
        assert meta.n_padded == 10

    def test_multi_returns_fields_unchanged(self):
        config = _make_halo_config(world_size=2)
        positions = torch.rand(10, 3)
        fields = {
            "velocities": torch.randn(10, 3),
            "atomic_numbers": torch.ones(10, dtype=torch.long),
        }
        padded_pos, padded_fields, meta = particle_halo_padding_multi(
            positions, fields, config
        )
        assert torch.equal(padded_pos, positions)
        assert torch.equal(padded_fields["velocities"], fields["velocities"])
        assert meta.n_owned == 10


class TestParticleHaloUnpadding:
    """Test particle_halo_unpadding."""

    def test_strips_ghosts(self):
        padded = torch.randn(20, 3)
        meta = ParticleHaloMetadata(
            n_owned=15, n_padded=20, send_indices=[], send_sizes=[], recv_sizes=[]
        )
        result = particle_halo_unpadding(padded, meta)
        assert result.shape == (15, 3)
        assert torch.equal(result, padded[:15])

    def test_noop_when_no_ghosts(self):
        padded = torch.randn(10, 3)
        meta = ParticleHaloMetadata(
            n_owned=10, n_padded=10, send_indices=[], send_sizes=[], recv_sizes=[]
        )
        result = particle_halo_unpadding(padded, meta)
        assert torch.equal(result, padded)


class TestParticleHaloConfig:
    """Test ParticleHaloConfig initialization."""

    def test_computes_neighbor_ranks(self):
        config = _make_halo_config(world_size=2)
        assert isinstance(config.neighbor_ranks, list)
        assert config.rank not in config.neighbor_ranks

    def test_computes_pbc_shifts(self):
        config = _make_halo_config(world_size=2)
        shifts = config.pbc_shifts
        assert isinstance(shifts, Mapping)
        key = next(iter(shifts))
        with pytest.raises(TypeError):
            shifts[key] = ()  # type: ignore[index]
        with pytest.raises(AttributeError):
            shifts[key].append(torch.zeros(3))  # type: ignore[attr-defined]

    def test_pbc_shifts_follow_cell_updates(self):
        config = _make_halo_config(world_size=2)
        old_shifts = config.pbc_shifts

        new_cell = config.partitioner.cell_matrix * 1.2
        config.partitioner.update_cell(new_cell)
        new_shifts = config.pbc_shifts

        expected = _compute_pbc_shift_vectors(config.partitioner)
        assert new_shifts.keys() == expected.keys()
        assert any(
            not torch.equal(old_shift, new_shift)
            for key, shifts in new_shifts.items()
            for old_shift, new_shift in zip(old_shifts[key], shifts, strict=True)
        )
        for key, shifts in new_shifts.items():
            for shift, expected_shift in zip(shifts, expected[key], strict=True):
                torch.testing.assert_close(shift, expected_shift)


# Multi-GPU tests live in test/distributed/test_multigpu.py
# which tests the full pipeline via torch.multiprocessing.spawn.
