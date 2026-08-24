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
"""Tests for group-cardinality metadata derived from Batch.group_idx."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest
import torch

from nvalchemi.data import AtomicData, Batch, GroupLayout
from nvalchemi.data.group_layout import _select_groups


def _batch() -> Batch:
    counts = [1, 2, 1, 3, 2]
    return Batch.from_data_list(
        [
            AtomicData(
                positions=torch.randn(count, 3),
                atomic_numbers=torch.ones(count, dtype=torch.long),
            )
            for count in counts
        ]
    )


class TestBatchGroupLayout:
    """Tests for the Batch grouping public surface and cache lifecycle."""

    def test_set_group_layout_normalizes_and_caches_immediately(self):
        batch = _batch()
        assert "system" not in batch._storage.groups

        batch.set_group_layout(torch.tensor([4, 4, 1, 1, 1], dtype=torch.int32))

        assert batch.group_idx.tolist() == [0, 0, 1, 1, 1]
        assert set(batch._storage.groups["system"].keys()) == {"group_idx"}
        cached = object.__getattribute__(batch, "_group_layout")
        assert isinstance(cached, GroupLayout)
        assert batch.group_layout is cached

    def test_group_layout_derives_all_cardinality_mappings(self):
        batch = _batch()
        batch.set_group_layout(torch.tensor([0, 0, 1, 1, 1]))
        layout = batch.group_layout

        assert layout.num_groups == 2
        assert layout.group_idx.tolist() == [0, 0, 1, 1, 1]
        assert layout.group_idx is batch.group_idx
        assert layout.device == batch.device
        assert layout.graph_rank.tolist() == [0, 1, 0, 1, 2]
        assert layout.node_to_group.tolist() == [0, 0, 0, 1, 1, 1, 1, 1, 1]
        assert layout.group_ptr.tolist() == [0, 2, 5]
        assert layout.num_graphs_per_group.tolist() == [2, 3]

    def test_group_operations(self):
        batch = _batch()
        batch.set_group_layout(torch.tensor([0, 0, 1, 1, 1]))
        layout = batch.group_layout
        graph_mask = torch.tensor([True, True, True, False, True])

        assert layout.reduce_all(graph_mask).tolist() == [True, False]
        assert layout.reduce_any(graph_mask).tolist() == [True, True]
        assert layout.broadcast(torch.tensor([7, 9])).tolist() == [7, 7, 9, 9, 9]
        assert layout.graph_mask(torch.tensor([False, True])).tolist() == [
            False,
            False,
            True,
            True,
            True,
        ]
        assert layout.selected_group_idx(torch.tensor([False, True])).tolist() == [
            0,
            0,
            0,
        ]

    def test_select_groups_rebases_nonadjacent_groups(self):
        batch = _batch()
        batch.set_group_layout(torch.tensor([0, 0, 1, 2, 2]))

        selected = _select_groups(batch, torch.tensor([True, False, True]))

        assert selected.num_graphs == 4
        assert selected.num_nodes == 8
        assert selected.group_idx.tolist() == [0, 0, 1, 1]
        assert selected.group_layout.num_graphs_per_group.tolist() == [2, 2]
        assert batch.group_idx.tolist() == [0, 0, 1, 2, 2]

    def test_index_select_preserves_selected_group_labels(self):
        batch = _batch()
        batch.set_group_layout(torch.tensor([0, 0, 1, 1, 1]))

        selected = batch.index_select([2, 3, 4, 0, 1])

        assert selected.group_idx.tolist() == [1, 1, 1, 0, 0]
        assert getattr(selected, "_group_layout", None) is None
        with pytest.raises(ValueError, match="dense, contiguous"):
            _ = selected.group_layout

        selected.normalize_group_idx()

        assert selected.group_idx.tolist() == [0, 0, 0, 1, 1]
        assert selected.group_layout.num_graphs_per_group.tolist() == [3, 2]
        assert batch.group_idx.tolist() == [0, 0, 1, 1, 1]

    def test_normalize_group_idx_rejects_interleaved_groups(self):
        batch = _batch()
        batch.set_group_layout(torch.tensor([0, 0, 1, 1, 1]))

        selected = batch.index_select([0, 2, 1, 3, 4])

        assert selected.group_idx.tolist() == [0, 1, 0, 1, 1]
        with pytest.raises(ValueError, match="same group must be contiguous"):
            selected.normalize_group_idx()

    def test_select_groups_returns_grouped_empty_batch(self):
        batch = _batch()
        batch.set_group_layout(torch.tensor([0, 0, 1, 2, 2]))

        selected = _select_groups(batch, torch.tensor([False, False, False]))

        assert selected.num_graphs == 0
        assert selected.num_nodes == 0
        assert selected.num_edges == 0
        assert selected.group_idx.numel() == 0
        assert selected.group_layout.num_groups == 0

    @pytest.mark.parametrize(
        "operation", ["clone", "to", "cpu", "cuda", "index_select"]
    )
    def test_constructed_batches_reset_and_lazily_rebuild_cache(self, operation):
        batch = _batch()
        batch.set_group_layout(torch.tensor([0, 0, 1, 1, 1]))
        original_layout = batch.group_layout

        if operation == "index_select":
            rebuilt_batch = batch.index_select([0, 1, 2, 3])
        elif operation == "cuda":
            if not torch.cuda.is_available():
                pytest.skip("CUDA is not available")
            rebuilt_batch = batch.cuda()
        else:
            rebuilt_batch = getattr(batch, operation)(
                *(("cpu",) if operation == "to" else ())
            )

        assert getattr(rebuilt_batch, "_group_layout", None) is None
        assert (
            rebuilt_batch.group_idx.tolist()
            == [0, 0, 1, 1, 1][: rebuilt_batch.num_graphs]
        )
        assert rebuilt_batch.group_layout is not original_layout
        assert (
            rebuilt_batch.group_layout.group_idx.tolist()
            == [0, 0, 1, 1, 1][: rebuilt_batch.num_graphs]
        )

    def test_put_invalidates_cached_layout(self):
        batch = _batch()
        batch.set_group_layout(torch.tensor([0, 0, 1, 1, 1]))
        original_layout = batch.group_layout
        source = batch.clone()

        batch.put(source, torch.zeros(batch.num_graphs, dtype=torch.bool))

        assert object.__getattribute__(batch, "_group_layout") is None
        assert batch.group_layout is not original_layout
        assert batch.group_layout.group_idx.tolist() == [0, 0, 1, 1, 1]

    def test_append_rebases_group_idx(self):
        batch = _batch()
        batch.set_group_layout(torch.tensor([0, 0, 1, 1, 1]))
        other = _batch()
        other.set_group_layout(torch.tensor([8, 8, 3, 3, 3]))
        other_group_idx = other.group_idx.clone()

        batch.append(other)

        assert batch.group_idx.tolist() == [0, 0, 1, 1, 1, 2, 2, 3, 3, 3]
        assert batch.group_layout.num_groups == 4
        assert batch.group_layout.num_graphs_per_group.tolist() == [2, 3, 2, 3]
        assert torch.equal(other.group_idx, other_group_idx)

    @pytest.mark.parametrize("grouped_receiver", [True, False])
    def test_append_rejects_mixed_grouping_before_mutation(self, grouped_receiver):
        batch = _batch()
        other = _batch()
        grouped = batch if grouped_receiver else other
        grouped.set_group_layout(torch.tensor([0, 0, 1, 1, 1]))
        original_num_graphs = batch.num_graphs

        with pytest.raises(ValueError, match="both batches or neither"):
            batch.append(other)

        assert batch.num_graphs == original_num_graphs

    def test_append_data_rejects_grouped_batch(self):
        batch = _batch()
        batch.set_group_layout(torch.tensor([0, 0, 1, 1, 1]))

        with pytest.raises(ValueError, match="grouped Batch"):
            batch.append_data(
                [
                    AtomicData(
                        positions=torch.randn(2, 3),
                        atomic_numbers=torch.ones(2, dtype=torch.long),
                    )
                ]
            )

        assert batch.num_graphs == 5

    def test_group_layout_is_not_serialized(self):
        batch = _batch()
        batch.set_group_layout(torch.tensor([0, 0, 1, 1, 1]))

        dumped = batch.model_dump()

        assert "group_idx" in dumped
        assert "group_layout" not in dumped
        assert "_group_layout" not in dumped

    def test_group_layout_is_frozen_dataclass(self):
        batch = _batch()
        batch.set_group_layout(torch.tensor([0, 0, 1, 1, 1]))

        with pytest.raises(FrozenInstanceError):
            batch.group_layout.group_idx = torch.zeros(5, dtype=torch.long)

    def test_missing_group_idx_raises_on_property_access(self):
        batch = _batch()

        with pytest.raises(ValueError, match="no group_idx"):
            _ = batch.group_layout

    def test_normalize_group_idx_updates_stored_metadata(self):
        batch = _batch()
        batch["group_idx"] = torch.tensor([4, 4, 1, 1, 1])

        with pytest.raises(ValueError, match="dense, contiguous"):
            _ = batch.group_layout

        batch.normalize_group_idx()

        assert batch.group_idx.tolist() == [0, 0, 1, 1, 1]
        assert batch.group_layout.num_graphs_per_group.tolist() == [2, 3]

    @pytest.mark.parametrize(
        ("group_idx", "error", "match"),
        [
            (torch.tensor([0, 1]), ValueError, "shape"),
            (torch.tensor([0.0, 0.0, 1.0, 1.0, 1.0]), TypeError, "integer"),
            (torch.tensor([0, 1, 0, 1, 1]), ValueError, "contiguous"),
        ],
    )
    def test_set_group_layout_validates_input(self, group_idx, error, match):
        batch = _batch()

        with pytest.raises(error, match=match):
            batch.set_group_layout(group_idx)
