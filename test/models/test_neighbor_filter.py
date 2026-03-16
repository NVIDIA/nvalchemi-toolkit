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
"""Comprehensive unit tests for nvalchemi.models._ops.neighbor_filter."""

from __future__ import annotations

import types

import pytest
import torch

from nvalchemi.models._ops.neighbor_filter import (
    filter_neighbor_list,
    filter_neighbor_matrix,
    neighbor_matrix_to_list,
    prepare_neighbors_for_model,
)
from nvalchemi.models.base import NeighborListFormat

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(params=[torch.float32, torch.float64])
def dtype(request):
    return request.param


@pytest.fixture(params=["cpu"])
def device(request):
    return torch.device(request.param)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_simple_4atom(dtype, device):
    """4-atom 1-D chain: 0--1---2---3.

    Positions along x-axis: 0, 1, 3, 5.
    Atoms 0 and 1 are close (d=1); pairs (0,2), (1,2) have d=3,2; pairs to
    atom 3 are further away.

    Neighbor matrix (K=3, fill=4):
        atom 0 sees: 1(d=1), 2(d=3), <fill>
        atom 1 sees: 0(d=1), 2(d=2), <fill>
        atom 2 sees: 0(d=3), 1(d=2), <fill>
        atom 3 sees: <fill>, <fill>, <fill>
    """
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [3.0, 0.0, 0.0], [5.0, 0.0, 0.0]],
        dtype=dtype,
        device=device,
    )
    nm = torch.tensor(
        [[1, 2, 4], [0, 2, 4], [0, 1, 4], [4, 4, 4]],
        dtype=torch.int32,
        device=device,
    )
    nn_ = torch.tensor([2, 2, 2, 0], dtype=torch.int32, device=device)
    fill_value = 4
    return positions, nm, nn_, fill_value


def _make_simple_coo(dtype, device):
    """COO version of the same 4-atom system as above.

    Edges: (0,1) d=1, (1,0) d=1, (1,2) d=2, (2,1) d=2, (0,2) d=3, (2,0) d=3
    ptr: [0, 2, 4, 6, 6]  (atom 3 has no neighbors)
    """
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [3.0, 0.0, 0.0], [5.0, 0.0, 0.0]],
        dtype=dtype,
        device=device,
    )
    # i: source, j: target
    edge_index = torch.tensor(
        [[0, 0, 1, 1, 2, 2], [1, 2, 0, 2, 0, 1]],
        dtype=torch.int32,
        device=device,
    )
    ptr = torch.tensor([0, 2, 4, 6, 6], dtype=torch.int32, device=device)
    return positions, edge_index, ptr


# ---------------------------------------------------------------------------
# TestFilterNeighborMatrix
# ---------------------------------------------------------------------------


class TestFilterNeighborMatrix:
    def test_basic_filtering_removes_distant_pairs(self, dtype, device):
        """Pairs beyond cutoff must be replaced with fill_value."""
        positions, nm, nn_, fill_value = _make_simple_4atom(dtype, device)

        # cutoff=1.5 keeps only d=1 pair (atoms 0-1 / 1-0); d=2 and d=3 are removed.
        nm_out, nn_out, _ = filter_neighbor_matrix(
            positions=positions,
            cutoff=1.5,
            neighbor_matrix=nm,
            num_neighbors=nn_,
            fill_value=fill_value,
        )

        # atom 0: only neighbor 1 survives
        assert int(nn_out[0]) == 1
        assert int(nm_out[0, 0]) == 1

        # atom 1: only neighbor 0 survives
        assert int(nn_out[1]) == 1
        assert int(nm_out[1, 0]) == 0

        # atom 2: neighbors 0 (d=3) and 1 (d=2) both removed
        assert int(nn_out[2]) == 0

        # atom 3: was already empty
        assert int(nn_out[3]) == 0

    def test_num_neighbors_decreases(self, dtype, device):
        """filter_neighbor_matrix must reduce num_neighbors when pairs are removed."""
        positions, nm, nn_, fill_value = _make_simple_4atom(dtype, device)

        _, nn_out, _ = filter_neighbor_matrix(
            positions=positions,
            cutoff=2.5,
            neighbor_matrix=nm,
            num_neighbors=nn_,
            fill_value=fill_value,
        )

        # cutoff=2.5: d=1 and d=2 survive; d=3 does not
        # atom 0 had 2 neighbors (d=1, d=3) -> keeps 1
        assert int(nn_out[0]) == 1
        # atom 1 had 2 neighbors (d=1, d=2) -> keeps 2
        assert int(nn_out[1]) == 2
        # atom 2 had 2 neighbors (d=3, d=2) -> keeps 1
        assert int(nn_out[2]) == 1

    def test_defragmentation_packs_valid_entries_to_front(self, dtype, device):
        """Valid entries must be contiguous at the start; fill_value pads the rest."""
        positions, nm, nn_, fill_value = _make_simple_4atom(dtype, device)

        nm_out, nn_out, _ = filter_neighbor_matrix(
            positions=positions,
            cutoff=2.5,
            neighbor_matrix=nm,
            num_neighbors=nn_,
            fill_value=fill_value,
        )

        N, K = nm_out.shape
        for i in range(N):
            count = int(nn_out[i])
            # Entries in [0, count) must be valid (< fill_value).
            for k in range(count):
                assert int(nm_out[i, k]) < fill_value, (
                    f"atom {i} slot {k} should be valid but got {nm_out[i, k]}"
                )
            # Entries in [count, K) must be fill_value.
            for k in range(count, K):
                assert int(nm_out[i, k]) == fill_value, (
                    f"atom {i} slot {k} should be fill_value but got {nm_out[i, k]}"
                )

    def test_pbc_single_cell_expands_effective_distance(self, dtype, device):
        """With a PBC cell and non-zero shifts, distances should include the shift."""
        # Place two atoms far apart in direct space but a shift of (1,0,0) brings
        # them close.  Cell = 5*I, so shift (1,0,0) -> Cartesian (+5, 0, 0).
        # atom 0 at (0,0,0), atom 1 at (4.5, 0, 0).
        # Direct distance = 4.5 > cutoff=2.
        # With shift (-1, 0, 0): delta + shift*cell = (4.5,0,0) + (-5,0,0) = (-0.5,0,0)
        # -> distance 0.5 < cutoff=2, so pair survives.
        positions = torch.tensor(
            [[0.0, 0.0, 0.0], [4.5, 0.0, 0.0]], dtype=dtype, device=device
        )
        cell = 5.0 * torch.eye(3, dtype=dtype, device=device)
        nm = torch.tensor([[1, 2], [0, 2]], dtype=torch.int32, device=device)
        nn_ = torch.tensor([1, 1], dtype=torch.int32, device=device)
        shifts = torch.tensor(
            [[[-1, 0, 0], [0, 0, 0]], [[1, 0, 0], [0, 0, 0]]],
            dtype=torch.int32,
            device=device,
        )
        fill_value = 2

        nm_out, nn_out, shifts_out = filter_neighbor_matrix(
            positions=positions,
            cutoff=2.0,
            neighbor_matrix=nm,
            num_neighbors=nn_,
            fill_value=fill_value,
            cell=cell,
            neighbor_shifts=shifts,
        )

        # Both atoms should retain their neighbor through the PBC image.
        assert int(nn_out[0]) == 1
        assert int(nn_out[1]) == 1

    def test_pbc_per_system_cells(self, dtype, device):
        """Per-system (B,3,3) cell with batch_idx routes correctly."""
        # Two independent systems: system 0 has cell=5*I, system 1 has cell=10*I.
        # Use the same geometry as the single-cell PBC test for system 0.
        # System 1 pair has no shift effect needed (direct distance already within cutoff).
        positions = torch.tensor(
            [
                [0.0, 0.0, 0.0],  # sys 0, atom 0
                [4.5, 0.0, 0.0],  # sys 0, atom 1
                [0.0, 0.0, 0.0],  # sys 1, atom 0
                [1.0, 0.0, 0.0],  # sys 1, atom 1
            ],
            dtype=dtype,
            device=device,
        )
        cell = torch.stack(
            [
                5.0 * torch.eye(3, dtype=dtype, device=device),
                10.0 * torch.eye(3, dtype=dtype, device=device),
            ],
            dim=0,
        )
        batch_idx = torch.tensor([0, 0, 1, 1], dtype=torch.int32, device=device)
        fill_value = 4
        nm = torch.tensor(
            [[1, fill_value], [0, fill_value], [3, fill_value], [2, fill_value]],
            dtype=torch.int32,
            device=device,
        )
        nn_ = torch.tensor([1, 1, 1, 1], dtype=torch.int32, device=device)
        shifts = torch.zeros(4, 2, 3, dtype=torch.int32, device=device)
        # Atom 0 (sys 0) seeing atom 1 through shift (-1,0,0).
        shifts[0, 0] = torch.tensor([-1, 0, 0], dtype=torch.int32, device=device)
        # Atom 1 (sys 0) seeing atom 0 through shift (+1,0,0).
        shifts[1, 0] = torch.tensor([1, 0, 0], dtype=torch.int32, device=device)

        nm_out, nn_out, _ = filter_neighbor_matrix(
            positions=positions,
            cutoff=2.0,
            neighbor_matrix=nm,
            num_neighbors=nn_,
            fill_value=fill_value,
            cell=cell,
            neighbor_shifts=shifts,
            batch_idx=batch_idx,
        )

        # System 0 pair (PBC image, effective d=0.5) should survive.
        assert int(nn_out[0]) == 1
        assert int(nn_out[1]) == 1
        # System 1 pair (direct d=1.0 < 2.0) should survive.
        assert int(nn_out[2]) == 1
        assert int(nn_out[3]) == 1

    def test_neighbor_shifts_output_matches_defragmented_matrix(self, dtype, device):
        """Returned shifts must correspond to the same reordering as nm_out."""
        positions, nm, nn_, fill_value = _make_simple_4atom(dtype, device)
        N, K = nm.shape
        # Assign distinct shifts so we can track which got kept.
        shifts = torch.zeros(N, K, 3, dtype=torch.int32, device=device)
        for i in range(N):
            for k in range(K):
                shifts[i, k] = torch.tensor([i * 10 + k, 0, 0], dtype=torch.int32)

        nm_out, nn_out, shifts_out = filter_neighbor_matrix(
            positions=positions,
            cutoff=2.5,
            neighbor_matrix=nm,
            num_neighbors=nn_,
            fill_value=fill_value,
            neighbor_shifts=shifts,
        )

        assert shifts_out is not None
        assert shifts_out.shape == (N, K, 3)

        # For each atom, check that the shift at slot k aligns with the neighbor
        # at nm_out[i, k] (cross-check against the original nm/shifts).
        orig_nm_flat = {
            (i, int(nm[i, k])): k
            for i in range(N)
            for k in range(K)
            if int(nm[i, k]) < fill_value
        }
        for i in range(N):
            for k in range(int(nn_out[i])):
                j = int(nm_out[i, k])
                orig_k = orig_nm_flat.get((i, j))
                assert orig_k is not None
                assert torch.equal(shifts_out[i, k], shifts[i, orig_k]), (
                    f"shift mismatch at atom {i} slot {k}: got {shifts_out[i, k]}, "
                    f"expected {shifts[i, orig_k]}"
                )

        # Slots beyond nn_out should have zero-filled shifts.
        for i in range(N):
            count = int(nn_out[i])
            assert torch.all(shifts_out[i, count:] == 0)

    def test_float32_and_float64(self, dtype, device):
        """Function must work for both float32 and float64 positions."""
        positions, nm, nn_, fill_value = _make_simple_4atom(dtype, device)
        nm_out, nn_out, _ = filter_neighbor_matrix(
            positions=positions,
            cutoff=1.5,
            neighbor_matrix=nm,
            num_neighbors=nn_,
            fill_value=fill_value,
        )
        assert nm_out.shape == nm.shape
        assert nn_out.shape == nn_.shape

    def test_all_neighbors_within_cutoff_no_change(self, dtype, device):
        """When all pairs are within cutoff, nm and nn_ must be unchanged."""
        positions, nm, nn_, fill_value = _make_simple_4atom(dtype, device)

        # cutoff=100 captures every pair.
        nm_out, nn_out, _ = filter_neighbor_matrix(
            positions=positions,
            cutoff=100.0,
            neighbor_matrix=nm,
            num_neighbors=nn_,
            fill_value=fill_value,
        )

        assert torch.equal(nn_out, nn_)
        # Valid entries per atom must be the same set (order may differ after sort).
        for i in range(4):
            orig_set = set(int(nm[i, k]) for k in range(int(nn_[i])))
            out_set = set(int(nm_out[i, k]) for k in range(int(nn_out[i])))
            assert orig_set == out_set

    def test_no_neighbors_within_cutoff_all_fill(self, dtype, device):
        """When no pair is within cutoff, all slots become fill_value and counts zero."""
        positions, nm, nn_, fill_value = _make_simple_4atom(dtype, device)

        nm_out, nn_out, _ = filter_neighbor_matrix(
            positions=positions,
            cutoff=0.01,
            neighbor_matrix=nm,
            num_neighbors=nn_,
            fill_value=fill_value,
        )

        assert torch.all(nn_out == 0)
        assert torch.all(nm_out == fill_value)

    def test_single_atom(self, dtype, device):
        """Single atom with empty neighbor list must return unchanged tensors."""
        positions = torch.tensor([[0.0, 0.0, 0.0]], dtype=dtype, device=device)
        nm = torch.tensor([[1]], dtype=torch.int32, device=device)
        nn_ = torch.tensor([0], dtype=torch.int32, device=device)
        fill_value = 1

        nm_out, nn_out, _ = filter_neighbor_matrix(
            positions=positions,
            cutoff=5.0,
            neighbor_matrix=nm,
            num_neighbors=nn_,
            fill_value=fill_value,
        )

        assert int(nn_out[0]) == 0
        assert int(nm_out[0, 0]) == fill_value

    def test_output_shapes(self, dtype, device):
        """Output shapes must match input shapes."""
        positions, nm, nn_, fill_value = _make_simple_4atom(dtype, device)
        shifts = torch.zeros(*nm.shape, 3, dtype=torch.int32, device=device)

        nm_out, nn_out, shifts_out = filter_neighbor_matrix(
            positions=positions,
            cutoff=2.0,
            neighbor_matrix=nm,
            num_neighbors=nn_,
            fill_value=fill_value,
            neighbor_shifts=shifts,
        )

        assert nm_out.shape == nm.shape
        assert nn_out.shape == nn_.shape
        assert shifts_out is not None
        assert shifts_out.shape == shifts.shape

    def test_no_shifts_input_returns_none_shifts(self, dtype, device):
        """When neighbor_shifts is None, the output shifts must also be None."""
        positions, nm, nn_, fill_value = _make_simple_4atom(dtype, device)

        _, _, shifts_out = filter_neighbor_matrix(
            positions=positions,
            cutoff=2.0,
            neighbor_matrix=nm,
            num_neighbors=nn_,
            fill_value=fill_value,
            neighbor_shifts=None,
        )

        assert shifts_out is None


# ---------------------------------------------------------------------------
# TestFilterNeighborList
# ---------------------------------------------------------------------------


class TestFilterNeighborList:
    def test_basic_filtering_removes_distant_edges(self, dtype, device):
        """Edges beyond cutoff must be removed from the COO list."""
        positions, edge_index, ptr = _make_simple_coo(dtype, device)

        # cutoff=1.5: only d=1 edges survive ((0,1) and (1,0)).
        nl_out, ptr_out, _ = filter_neighbor_list(
            positions=positions,
            cutoff=1.5,
            neighbor_list=edge_index,
            neighbor_ptr=ptr,
        )

        assert nl_out.shape[0] == 2  # still 2 rows (i and j)
        assert nl_out.shape[1] == 2  # 2 edges survive

        # The surviving edges must be the d=1 pair.
        edges_set = {
            (int(nl_out[0, e]), int(nl_out[1, e])) for e in range(nl_out.shape[1])
        }
        assert edges_set == {(0, 1), (1, 0)}

    def test_neighbor_ptr_rebuilt_correctly(self, dtype, device):
        """CSR pointer after filtering must reflect the new per-atom edge counts."""
        positions, edge_index, ptr = _make_simple_coo(dtype, device)

        # cutoff=2.5: d=1 and d=2 edges survive.
        # Surviving: (0,1)d=1, (1,0)d=1, (1,2)d=2, (2,1)d=2  -> 4 edges
        # atom 0: 1 edge (to 1), atom 1: 2 edges (to 0, to 2), atom 2: 1 edge (to 1), atom 3: 0
        nl_out, ptr_out, _ = filter_neighbor_list(
            positions=positions,
            cutoff=2.5,
            neighbor_list=edge_index,
            neighbor_ptr=ptr,
        )

        assert int(ptr_out[0]) == 0
        assert int(ptr_out[4]) == nl_out.shape[1]  # last entry == total edges
        assert ptr_out.shape == (5,)  # N+1 = 5

        # Count edges per atom from nl_out.
        counts_from_nl = [0, 0, 0, 0]
        for e in range(nl_out.shape[1]):
            counts_from_nl[int(nl_out[0, e])] += 1

        for i in range(4):
            assert int(ptr_out[i + 1]) - int(ptr_out[i]) == counts_from_nl[i]

    def test_pbc_unit_shifts_applied(self, dtype, device):
        """Edges kept only because of PBC shift must survive when shifts are provided."""
        # Same setup as the single-cell PBC matrix test.
        positions = torch.tensor(
            [[0.0, 0.0, 0.0], [4.5, 0.0, 0.0]], dtype=dtype, device=device
        )
        cell = 5.0 * torch.eye(3, dtype=dtype, device=device)
        # Both edges: (0->1) with shift (-1,0,0) and (1->0) with shift (+1,0,0).
        edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.int32, device=device)
        ptr = torch.tensor([0, 1, 2], dtype=torch.int32, device=device)
        unit_shifts = torch.tensor(
            [[-1, 0, 0], [1, 0, 0]], dtype=torch.int32, device=device
        )

        nl_out, ptr_out, us_out = filter_neighbor_list(
            positions=positions,
            cutoff=2.0,
            neighbor_list=edge_index,
            neighbor_ptr=ptr,
            cell=cell,
            unit_shifts=unit_shifts,
        )

        # Both edges survive because PBC image distances are 0.5.
        assert nl_out.shape[1] == 2
        assert us_out is not None
        assert us_out.shape == (2, 3)

    def test_pbc_per_system_cells(self, dtype, device):
        """Per-system (B,3,3) cell routes cell correctly by source atom's system."""
        positions = torch.tensor(
            [
                [0.0, 0.0, 0.0],  # sys 0, atom 0
                [4.5, 0.0, 0.0],  # sys 0, atom 1
                [0.0, 0.0, 0.0],  # sys 1, atom 0
                [1.0, 0.0, 0.0],  # sys 1, atom 1
            ],
            dtype=dtype,
            device=device,
        )
        cell = torch.stack(
            [
                5.0 * torch.eye(3, dtype=dtype, device=device),
                10.0 * torch.eye(3, dtype=dtype, device=device),
            ],
            dim=0,
        )
        batch_idx = torch.tensor([0, 0, 1, 1], dtype=torch.int32, device=device)
        # Edges for both systems.
        edge_index = torch.tensor(
            [[0, 1, 2, 3], [1, 0, 3, 2]], dtype=torch.int32, device=device
        )
        ptr = torch.tensor([0, 1, 2, 3, 4], dtype=torch.int32, device=device)
        unit_shifts = torch.tensor(
            [[-1, 0, 0], [1, 0, 0], [0, 0, 0], [0, 0, 0]],
            dtype=torch.int32,
            device=device,
        )

        nl_out, ptr_out, us_out = filter_neighbor_list(
            positions=positions,
            cutoff=2.0,
            neighbor_list=edge_index,
            neighbor_ptr=ptr,
            cell=cell,
            unit_shifts=unit_shifts,
            batch_idx=batch_idx,
        )

        # All 4 edges should survive (sys0 via PBC d=0.5, sys1 direct d=1.0).
        assert nl_out.shape[1] == 4

    def test_none_unit_shifts_returns_none(self, dtype, device):
        """When unit_shifts input is None, output must also be None."""
        positions, edge_index, ptr = _make_simple_coo(dtype, device)

        _, _, us_out = filter_neighbor_list(
            positions=positions,
            cutoff=5.0,
            neighbor_list=edge_index,
            neighbor_ptr=ptr,
            unit_shifts=None,
        )

        assert us_out is None

    def test_output_shapes(self, dtype, device):
        """Output ptr must always have shape (N+1,)."""
        positions, edge_index, ptr = _make_simple_coo(dtype, device)
        N = positions.shape[0]

        _, ptr_out, _ = filter_neighbor_list(
            positions=positions,
            cutoff=2.0,
            neighbor_list=edge_index,
            neighbor_ptr=ptr,
        )

        assert ptr_out.shape == (N + 1,)

    def test_all_edges_removed(self, dtype, device):
        """With cutoff below all pair distances, output edge list must be empty."""
        positions, edge_index, ptr = _make_simple_coo(dtype, device)
        N = positions.shape[0]

        nl_out, ptr_out, _ = filter_neighbor_list(
            positions=positions,
            cutoff=0.001,
            neighbor_list=edge_index,
            neighbor_ptr=ptr,
        )

        assert nl_out.shape[1] == 0
        assert torch.all(ptr_out == 0)
        assert ptr_out.shape == (N + 1,)

    def test_all_edges_survive(self, dtype, device):
        """With large cutoff, all edges must survive and ptr must match input."""
        positions, edge_index, ptr = _make_simple_coo(dtype, device)

        nl_out, ptr_out, _ = filter_neighbor_list(
            positions=positions,
            cutoff=100.0,
            neighbor_list=edge_index,
            neighbor_ptr=ptr,
        )

        assert nl_out.shape[1] == edge_index.shape[1]
        assert torch.equal(ptr_out, ptr)


# ---------------------------------------------------------------------------
# TestNeighborMatrixToList
# ---------------------------------------------------------------------------


class TestNeighborMatrixToList:
    def test_edge_index_source_and_target(self, dtype, device):
        """edge_index[0] must be source (i) indices, edge_index[1] target (j)."""
        positions, nm, nn_, fill_value = _make_simple_4atom(dtype, device)

        nl, ptr, _ = neighbor_matrix_to_list(
            neighbor_matrix=nm,
            num_neighbors=nn_,
            fill_value=fill_value,
        )

        assert nl.shape[0] == 2
        M = nl.shape[1]

        # All source and target indices must be in [0, N).
        N = nm.shape[0]
        assert torch.all(nl[0] >= 0) and torch.all(nl[0] < N)
        assert torch.all(nl[1] >= 0) and torch.all(nl[1] < N)

        # Every (i, j) pair must correspond to nm[i, :] containing j.
        for e in range(M):
            i = int(nl[0, e])
            j = int(nl[1, e])
            valid_js = [int(nm[i, k]) for k in range(int(nn_[i]))]
            assert j in valid_js, f"target {j} not a valid neighbor of {i}"

    def test_neighbor_ptr_format(self, dtype, device):
        """ptr[0] must be 0, ptr[-1] must equal total number of edges."""
        positions, nm, nn_, fill_value = _make_simple_4atom(dtype, device)

        nl, ptr, _ = neighbor_matrix_to_list(
            neighbor_matrix=nm,
            num_neighbors=nn_,
            fill_value=fill_value,
        )

        assert int(ptr[0]) == 0
        assert int(ptr[-1]) == nl.shape[1]
        assert ptr.shape == (nm.shape[0] + 1,)

    def test_neighbor_ptr_per_atom_counts(self, dtype, device):
        """ptr[i+1] - ptr[i] must equal nn_[i] for all atoms."""
        positions, nm, nn_, fill_value = _make_simple_4atom(dtype, device)

        nl, ptr, _ = neighbor_matrix_to_list(
            neighbor_matrix=nm,
            num_neighbors=nn_,
            fill_value=fill_value,
        )

        for i in range(nm.shape[0]):
            count = int(ptr[i + 1]) - int(ptr[i])
            assert count == int(nn_[i]), (
                f"atom {i}: ptr says {count} edges but nn_={int(nn_[i])}"
            )

    def test_total_edges_equals_sum_of_num_neighbors(self, dtype, device):
        """Total edges in nl must equal sum(nn_)."""
        positions, nm, nn_, fill_value = _make_simple_4atom(dtype, device)

        nl, ptr, _ = neighbor_matrix_to_list(
            neighbor_matrix=nm,
            num_neighbors=nn_,
            fill_value=fill_value,
        )

        assert nl.shape[1] == int(nn_.sum())

    def test_without_shifts_returns_none(self, dtype, device):
        """When neighbor_shifts is None, unit_shifts output must be None."""
        positions, nm, nn_, fill_value = _make_simple_4atom(dtype, device)

        _, _, us_out = neighbor_matrix_to_list(
            neighbor_matrix=nm,
            num_neighbors=nn_,
            fill_value=fill_value,
            neighbor_shifts=None,
        )

        assert us_out is None

    def test_with_shifts_output_shape_and_values(self, dtype, device):
        """Shift vectors for valid pairs must be extracted correctly."""
        positions, nm, nn_, fill_value = _make_simple_4atom(dtype, device)
        N, K = nm.shape
        # Unique shifts so we can verify correct extraction.
        shifts = torch.zeros(N, K, 3, dtype=torch.int32, device=device)
        for i in range(N):
            for k in range(K):
                shifts[i, k] = torch.tensor([i * 10 + k, i, k], dtype=torch.int32)

        nl, ptr, us_out = neighbor_matrix_to_list(
            neighbor_matrix=nm,
            num_neighbors=nn_,
            fill_value=fill_value,
            neighbor_shifts=shifts,
        )

        M = nl.shape[1]
        assert us_out is not None
        assert us_out.shape == (M, 3)

        # Verify each edge's shift matches the source matrix entry.
        # Build a lookup: (i, j) -> original k in nm.
        lookup = {}
        for i in range(N):
            for k in range(K):
                j = int(nm[i, k])
                if j < fill_value:
                    lookup[(i, j)] = k

        for e in range(M):
            i = int(nl[0, e])
            j = int(nl[1, e])
            k = lookup[(i, j)]
            assert torch.equal(us_out[e], shifts[i, k]), (
                f"edge ({i},{j}): shift {us_out[e]} != expected {shifts[i, k]}"
            )

    def test_variable_num_neighbors_per_atom(self, dtype, device):
        """Atoms with different valid neighbor counts must all be handled correctly."""
        # atom 0: 3 neighbors, atom 1: 1 neighbor, atom 2: 0 neighbors
        nm = torch.tensor(
            [[1, 2, 3], [0, 4, 4], [4, 4, 4]],
            dtype=torch.int32,
            device=device,
        )
        nn_ = torch.tensor([3, 1, 0], dtype=torch.int32, device=device)
        fill_value = 4

        nl, ptr, _ = neighbor_matrix_to_list(
            neighbor_matrix=nm,
            num_neighbors=nn_,
            fill_value=fill_value,
        )

        assert nl.shape[1] == 4  # 3 + 1 + 0
        assert int(ptr[0]) == 0
        assert int(ptr[1]) == 3
        assert int(ptr[2]) == 4
        assert int(ptr[3]) == 4

    def test_output_dtype_is_int32(self, dtype, device):
        """Output nl and ptr must be int32 regardless of positions dtype."""
        positions, nm, nn_, fill_value = _make_simple_4atom(dtype, device)

        nl, ptr, _ = neighbor_matrix_to_list(
            neighbor_matrix=nm,
            num_neighbors=nn_,
            fill_value=fill_value,
        )

        assert nl.dtype == torch.int32
        assert ptr.dtype == torch.int32


# ---------------------------------------------------------------------------
# TestPrepareNeighborsForModel
# ---------------------------------------------------------------------------


def _make_matrix_data(dtype, device, cutoff=None):
    """Build a SimpleNamespace mimicking a Batch with MATRIX neighbor data."""
    positions, nm, nn_, fill_value = _make_simple_4atom(dtype, device)
    data = types.SimpleNamespace(
        positions=positions,
        neighbor_matrix=nm,
        num_neighbors=nn_,
        batch=torch.zeros(4, dtype=torch.int32, device=device),
        num_nodes=4,
        edge_index=None,
    )
    if cutoff is not None:
        data._neighbor_list_cutoff = cutoff
    return data, fill_value


def _make_coo_data(dtype, device, cutoff=None):
    """Build a SimpleNamespace mimicking a Batch with COO neighbor data."""
    positions, edge_index, ptr = _make_simple_coo(dtype, device)
    data = types.SimpleNamespace(
        positions=positions,
        neighbor_matrix=None,
        edge_index=edge_index,
        edge_ptr=ptr,
        batch=torch.zeros(4, dtype=torch.int32, device=device),
        num_nodes=4,
    )
    if cutoff is not None:
        data._neighbor_list_cutoff = cutoff
    return data


class TestPrepareNeighborsForModel:
    # ---- MATRIX -> MATRIX ----

    def test_matrix_to_matrix_no_filtering(self, dtype, device):
        """MATRIX->MATRIX with matching cutoff returns same tensors."""
        data, fill_value = _make_matrix_data(dtype, device, cutoff=10.0)

        result = prepare_neighbors_for_model(
            data=data,
            model_cutoff=10.0,
            target_format=NeighborListFormat.MATRIX,
            fill_value=fill_value,
        )

        assert "neighbor_matrix" in result
        assert "num_neighbors" in result
        assert torch.equal(result["neighbor_matrix"], data.neighbor_matrix)
        assert torch.equal(result["num_neighbors"], data.num_neighbors)

    def test_matrix_to_matrix_with_filtering(self, dtype, device):
        """MATRIX->MATRIX with built_cutoff > model_cutoff applies filtering."""
        # Built with cutoff=10, model only needs cutoff=1.5.
        data, fill_value = _make_matrix_data(dtype, device, cutoff=10.0)

        result = prepare_neighbors_for_model(
            data=data,
            model_cutoff=1.5,
            target_format=NeighborListFormat.MATRIX,
            fill_value=fill_value,
        )

        assert "neighbor_matrix" in result
        assert "num_neighbors" in result
        # After filtering to 1.5, only the d=1 pair should survive.
        nn_out = result["num_neighbors"]
        assert int(nn_out[0]) == 1
        assert int(nn_out[1]) == 1
        assert int(nn_out[2]) == 0
        assert int(nn_out[3]) == 0

    def test_matrix_to_matrix_no_cutoff_attr_no_filtering(self, dtype, device):
        """Without _neighbor_list_cutoff attribute, no filtering is applied."""
        data, fill_value = _make_matrix_data(dtype, device, cutoff=None)

        result = prepare_neighbors_for_model(
            data=data,
            model_cutoff=1.5,
            target_format=NeighborListFormat.MATRIX,
            fill_value=fill_value,
        )

        assert torch.equal(result["neighbor_matrix"], data.neighbor_matrix)
        assert torch.equal(result["num_neighbors"], data.num_neighbors)

    # ---- MATRIX -> COO ----

    def test_matrix_to_coo_conversion(self, dtype, device):
        """MATRIX->COO must produce a valid COO edge list without filtering."""
        data, fill_value = _make_matrix_data(dtype, device, cutoff=None)

        result = prepare_neighbors_for_model(
            data=data,
            model_cutoff=100.0,
            target_format=NeighborListFormat.COO,
            fill_value=fill_value,
        )

        assert "edge_index" in result
        assert "edge_ptr" in result
        nl = result["edge_index"]
        ptr = result["edge_ptr"]
        assert nl.shape[0] == 2
        assert int(ptr[0]) == 0
        assert int(ptr[-1]) == nl.shape[1]
        assert nl.shape[1] == int(data.num_neighbors.sum())

    def test_matrix_to_coo_with_filtering(self, dtype, device):
        """MATRIX->COO with built_cutoff > model_cutoff filters before conversion."""
        data, fill_value = _make_matrix_data(dtype, device, cutoff=10.0)

        result = prepare_neighbors_for_model(
            data=data,
            model_cutoff=1.5,
            target_format=NeighborListFormat.COO,
            fill_value=fill_value,
        )

        nl = result["edge_index"]
        # Only d=1 pair survives: edges (0,1) and (1,0).
        assert nl.shape[1] == 2
        edges_set = {(int(nl[0, e]), int(nl[1, e])) for e in range(nl.shape[1])}
        assert edges_set == {(0, 1), (1, 0)}

    # ---- COO -> COO ----

    def test_coo_to_coo_no_filtering(self, dtype, device):
        """COO->COO with matching cutoff returns same edge list."""
        data = _make_coo_data(dtype, device, cutoff=10.0)

        result = prepare_neighbors_for_model(
            data=data,
            model_cutoff=10.0,
            target_format=NeighborListFormat.COO,
            fill_value=999,
        )

        assert "edge_index" in result
        assert "edge_ptr" in result
        assert torch.equal(result["edge_index"], data.edge_index)
        assert torch.equal(result["edge_ptr"], data.edge_ptr)

    def test_coo_to_coo_with_filtering(self, dtype, device):
        """COO->COO with built_cutoff > model_cutoff filters edges."""
        data = _make_coo_data(dtype, device, cutoff=10.0)

        result = prepare_neighbors_for_model(
            data=data,
            model_cutoff=1.5,
            target_format=NeighborListFormat.COO,
            fill_value=999,
        )

        nl = result["edge_index"]
        # Only d=1 pair survives.
        assert nl.shape[1] == 2

    def test_coo_no_cutoff_attr_no_filtering(self, dtype, device):
        """Without _neighbor_list_cutoff, COO->COO makes no change."""
        data = _make_coo_data(dtype, device, cutoff=None)

        result = prepare_neighbors_for_model(
            data=data,
            model_cutoff=1.5,
            target_format=NeighborListFormat.COO,
            fill_value=999,
        )

        assert torch.equal(result["edge_index"], data.edge_index)
        assert torch.equal(result["edge_ptr"], data.edge_ptr)

    # ---- Error cases ----

    def test_raises_runtime_error_when_no_neighbor_data(self, dtype, device):
        """RuntimeError must be raised when neither neighbor_matrix nor edge_index present."""
        data = types.SimpleNamespace(
            positions=torch.zeros(4, 3, dtype=dtype, device=device),
            neighbor_matrix=None,
            edge_index=None,
            batch=torch.zeros(4, dtype=torch.int32, device=device),
            num_nodes=4,
        )

        with pytest.raises(RuntimeError, match="neither 'neighbor_matrix' nor"):
            prepare_neighbors_for_model(
                data=data,
                model_cutoff=5.0,
                target_format=NeighborListFormat.COO,
                fill_value=999,
            )

    def test_raises_runtime_error_matrix_format_without_matrix(self, dtype, device):
        """RuntimeError must be raised when MATRIX format requested but no matrix present."""
        data = _make_coo_data(dtype, device)
        # Ensure neighbor_matrix is absent.
        assert data.neighbor_matrix is None

        with pytest.raises(RuntimeError, match="target format is MATRIX"):
            prepare_neighbors_for_model(
                data=data,
                model_cutoff=5.0,
                target_format=NeighborListFormat.MATRIX,
                fill_value=999,
            )

    # ---- neighbor_shifts propagation ----

    def test_neighbor_shifts_in_matrix_output(self, dtype, device):
        """neighbor_shifts key present in output when data has neighbor_shifts."""
        data, fill_value = _make_matrix_data(dtype, device, cutoff=None)
        N, K = data.neighbor_matrix.shape
        data.neighbor_shifts = torch.zeros(N, K, 3, dtype=torch.int32, device=device)

        result = prepare_neighbors_for_model(
            data=data,
            model_cutoff=100.0,
            target_format=NeighborListFormat.MATRIX,
            fill_value=fill_value,
        )

        assert "neighbor_shifts" in result

    def test_unit_shifts_in_coo_output_from_matrix(self, dtype, device):
        """unit_shifts key present in COO output when matrix had neighbor_shifts."""
        data, fill_value = _make_matrix_data(dtype, device, cutoff=None)
        N, K = data.neighbor_matrix.shape
        data.neighbor_shifts = torch.zeros(N, K, 3, dtype=torch.int32, device=device)

        result = prepare_neighbors_for_model(
            data=data,
            model_cutoff=100.0,
            target_format=NeighborListFormat.COO,
            fill_value=fill_value,
        )

        assert "unit_shifts" in result
        assert result["unit_shifts"].shape[1] == 3

    def test_unit_shifts_in_coo_output_from_coo(self, dtype, device):
        """unit_shifts key present in COO output when COO input had unit_shifts."""
        data = _make_coo_data(dtype, device, cutoff=None)
        M = data.edge_index.shape[1]
        data.unit_shifts = torch.zeros(M, 3, dtype=torch.int32, device=device)

        result = prepare_neighbors_for_model(
            data=data,
            model_cutoff=100.0,
            target_format=NeighborListFormat.COO,
            fill_value=999,
        )

        assert "unit_shifts" in result

    def test_no_shifts_key_absent_in_output(self, dtype, device):
        """When no shifts present in data, output dict must not contain shift keys."""
        data, fill_value = _make_matrix_data(dtype, device, cutoff=None)

        result_matrix = prepare_neighbors_for_model(
            data=data,
            model_cutoff=100.0,
            target_format=NeighborListFormat.MATRIX,
            fill_value=fill_value,
        )
        assert "neighbor_shifts" not in result_matrix

        result_coo = prepare_neighbors_for_model(
            data=data,
            model_cutoff=100.0,
            target_format=NeighborListFormat.COO,
            fill_value=fill_value,
        )
        assert "unit_shifts" not in result_coo
