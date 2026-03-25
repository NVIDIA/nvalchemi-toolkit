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
"""Tests for NeighborListHook edge_index shape and value contracts."""

from __future__ import annotations

import torch

from nvalchemi.data import AtomicData, Batch
from nvalchemi.dynamics.hooks import NeighborListHook
from nvalchemi.models.base import NeighborConfig, NeighborListFormat


def _make_two_atom_batch() -> Batch:
    """Two atoms 1 Å apart, no PBC."""
    data = AtomicData(
        positions=torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        atomic_numbers=torch.tensor([6, 6]),
    )
    return Batch.from_data_list([data])


def _make_pbc_batch() -> Batch:
    """Two atoms in a 5 Å cubic cell with PBC."""
    data = AtomicData(
        positions=torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        atomic_numbers=torch.tensor([6, 6]),
        cell=(torch.eye(3) * 5.0).unsqueeze(0),
        pbc=torch.tensor([[True, True, True]]),
    )
    return Batch.from_data_list([data])


class TestNeighborListHookCOO:
    """Verify COO hook writes edge_index as (E, 2) into the batch."""

    def test_edge_index_shape_is_E2(self) -> None:
        """After hook runs, batch.edge_index must be (E, 2)."""
        batch = _make_two_atom_batch()
        config = NeighborConfig(cutoff=2.0, format=NeighborListFormat.COO)
        hook = NeighborListHook(config)

        hook(batch, None)

        ei = batch.edge_index
        assert ei.dim() == 2
        assert ei.shape[1] == 2, f"Expected (E, 2), got {ei.shape}"

    def test_edge_index_values_are_valid_node_indices(self) -> None:
        """Source and target indices must be in [0, num_nodes)."""
        batch = _make_two_atom_batch()
        config = NeighborConfig(cutoff=2.0, format=NeighborListFormat.COO)
        hook = NeighborListHook(config)

        hook(batch, None)

        ei = batch.edge_index
        assert (ei >= 0).all()
        assert (ei < batch.num_nodes).all()

    def test_edge_index_contains_expected_pairs(self) -> None:
        """Two atoms 1 Å apart with cutoff=2 Å should produce edges (0,1) and (1,0)."""
        batch = _make_two_atom_batch()
        config = NeighborConfig(cutoff=2.0, format=NeighborListFormat.COO)
        hook = NeighborListHook(config)

        hook(batch, None)

        ei = batch.edge_index
        pairs = set(map(tuple, ei.tolist()))
        assert (0, 1) in pairs
        assert (1, 0) in pairs

    def test_edge_index_half_list(self) -> None:
        """With half_list=True only (0,1) appears, not (1,0)."""
        batch = _make_two_atom_batch()
        config = NeighborConfig(
            cutoff=2.0, format=NeighborListFormat.COO, half_list=True
        )
        hook = NeighborListHook(config)

        hook(batch, None)

        ei = batch.edge_index
        assert ei.shape == (1, 2), f"Expected (1, 2) for half list, got {ei.shape}"

    def test_pbc_edge_index_shape_is_E2(self) -> None:
        """PBC system: edge_index must still be (E, 2)."""
        batch = _make_pbc_batch()
        config = NeighborConfig(cutoff=2.0, format=NeighborListFormat.COO)
        hook = NeighborListHook(config)

        hook(batch, None)

        ei = batch.edge_index
        assert ei.dim() == 2
        assert ei.shape[1] == 2, f"Expected (E, 2), got {ei.shape}"

    def test_pbc_unit_shifts_aligned_with_edge_index(self) -> None:
        """unit_shifts must have the same E as edge_index."""
        batch = _make_pbc_batch()
        config = NeighborConfig(cutoff=2.0, format=NeighborListFormat.COO)
        hook = NeighborListHook(config)

        hook(batch, None)

        ei = batch.edge_index
        us = batch.unit_shifts
        assert us.shape == (ei.shape[0], 3)

    def test_multi_graph_edge_index_shape(self) -> None:
        """Batching multiple graphs: edge_index is (E_total, 2)."""
        data_a = AtomicData(
            positions=torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
            atomic_numbers=torch.tensor([6, 6]),
        )
        data_b = AtomicData(
            positions=torch.tensor(
                [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [1.0, 0.0, 0.0]]
            ),
            atomic_numbers=torch.tensor([8, 8, 8]),
        )
        batch = Batch.from_data_list([data_a, data_b])
        config = NeighborConfig(cutoff=2.0, format=NeighborListFormat.COO)
        hook = NeighborListHook(config)

        hook(batch, None)

        ei = batch.edge_index
        assert ei.dim() == 2
        assert ei.shape[1] == 2
        assert ei.shape[0] > 0
