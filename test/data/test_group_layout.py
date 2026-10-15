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
from unittest.mock import patch

import pytest
import torch

from nvalchemi.data import AtomicData, Batch, GroupLayout


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

        with pytest.raises(ValueError, match="set_group_layout"):
            batch.add_key(
                "group_idx",
                [torch.tensor([label]) for label in [0, 1, 1, 1, 1]],
                level="system",
                overwrite=True,
            )

        assert batch.group_idx.tolist() == [0, 0, 1, 1, 1]
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

    def test_index_select_preserves_selected_group_labels(self):
        batch = _batch()
        batch.set_group_layout(torch.tensor([0, 0, 1, 1, 1]))

        selected = batch.index_select([2, 3, 4, 0, 1])

        assert selected.group_idx.tolist() == [1, 1, 1, 0, 0]
        assert getattr(selected, "_group_layout", None) is None
        with pytest.raises(ValueError, match="dense, contiguous"):
            _ = selected.group_layout

        selected.set_group_layout(selected.group_idx)

        assert selected.group_idx.tolist() == [0, 0, 0, 1, 1]
        assert selected.group_layout.num_graphs_per_group.tolist() == [3, 2]
        assert batch.group_idx.tolist() == [0, 0, 1, 1, 1]

    def test_set_group_layout_rejects_interleaved_selected_groups(self):
        batch = _batch()
        batch.set_group_layout(torch.tensor([0, 0, 1, 1, 1]))

        selected = batch.index_select([0, 2, 1, 3, 4])

        assert selected.group_idx.tolist() == [0, 1, 0, 1, 1]
        with pytest.raises(ValueError, match="same group must be contiguous"):
            selected.set_group_layout(selected.group_idx)

    @pytest.mark.parametrize(
        "selection,graphs,group_sizes",
        [
            ([1, 0], [2, 3, 4, 0, 1], [3, 2]),
            ([False, True], [2, 3, 4], [3]),
            ([], [], []),
            ([False, False], [], []),
        ],
    )
    def test_index_select_groups_preserves_whole_groups(
        self, selection, graphs, group_sizes, device: str
    ) -> None:
        """Ordered indices and group masks select complete graphs and rebase labels."""
        batch = _batch().to(device)
        batch.set_group_layout(torch.tensor([0, 0, 1, 1, 1], device=device))

        selected = batch.index_select_groups(selection)

        assert selected.num_graphs == len(graphs)
        assert selected.device == batch.device
        assert selected.group_layout.num_graphs_per_group.tolist() == group_sizes
        assert batch.group_idx.tolist() == [0, 0, 1, 1, 1]
        if graphs:
            expected = batch.index_select(graphs)
            torch.testing.assert_close(selected.positions, expected.positions)
            torch.testing.assert_close(selected.batch_ptr, expected.batch_ptr)

    @pytest.mark.parametrize(
        "selection,error,match",
        [
            ([True], ValueError, "one entry per group"),
            ([0, 0], ValueError, "cannot repeat a group"),
            ([-1], IndexError, "out of range"),
            ([2], IndexError, "out of range"),
        ],
    )
    def test_index_select_groups_rejects_invalid_selection(
        self, selection, error, match
    ) -> None:
        """Group selection rejects invalid masks, duplicate groups, and bounds."""
        batch = _batch()
        batch.set_group_layout(torch.tensor([0, 0, 1, 1, 1]))
        with pytest.raises(error, match=match):
            batch.index_select_groups(selection)

    @pytest.mark.parametrize("operation", ["put", "trim"])
    def test_partial_group_mask_rejected_before_mutation(
        self, operation: str, device: str
    ) -> None:
        """Buffer copying and removal must not split a logical group."""
        batch = _batch().to(device)
        batch.set_group_layout(torch.tensor([0, 0, 1, 1, 1], device=device))
        layout = batch.group_layout
        positions = batch.positions.clone()
        mask = torch.tensor([True, False, False, False, False], device=device)
        buffer = Batch.empty_like(batch, device=device)

        with pytest.raises(ValueError, match="complete groups"):
            if operation == "put":
                buffer.put(batch, mask)
            else:
                batch.trim(mask)

        assert buffer.num_graphs == 0
        assert batch.group_layout is layout
        assert batch.group_idx.tolist() == [0, 0, 1, 1, 1]
        torch.testing.assert_close(batch.positions, positions)

    def test_put_copies_only_groups_that_fit(self, device: str) -> None:
        """put moves whole groups only and records them in the source mask."""
        # Mimic in-flight batching: ask put to move every graph out of the
        # source, but size the buffer so only group 0 fits (2 graphs / 3 atoms;
        # group 1 needs 3 / 6). put stores the copied graphs on the source.
        source = _batch().to(device)
        source.set_group_layout(torch.tensor([0, 0, 1, 1, 1], device=device))
        positions = source.positions.clone()
        buffer = Batch.empty(
            num_systems=2, num_nodes=3, num_edges=0, template=source, device=device
        )
        buffer.put(source, torch.ones(5, dtype=torch.bool, device=device))

        assert source._copied_mask.tolist() == [True, True, False, False, False]
        assert buffer.group_layout.num_graphs_per_group.tolist() == [2]
        torch.testing.assert_close(buffer.positions, positions[:3])

    @pytest.mark.parametrize("operation", ["trim", "defrag"])
    def test_trim_and_defrag_remove_put_groups(
        self, operation: str, device: str
    ) -> None:
        """Test removing put groups with trim and defrag.

        The remaining old group 1 is relabeled as group 0. ``trim`` returns a
        new batch; ``defrag`` compacts in place and keeps capacity.
        """
        # Same setup as above: put copies only group 0 out of the source.
        source = _batch().to(device)
        source.set_group_layout(torch.tensor([0, 0, 1, 1, 1], device=device))
        positions = source.positions.clone()
        buffer = Batch.empty(
            num_systems=2, num_nodes=3, num_edges=0, template=source, device=device
        )
        buffer.put(source, torch.ones(5, dtype=torch.bool, device=device))
        capacity = source.system_capacity

        # With no mask argument, trim/defrag remove the graphs put copied
        # (group 0), leaving the old group 1.
        if operation == "trim":
            remaining = source.trim()
            assert source.num_graphs == 5
        else:
            remaining = source.defrag()
            assert remaining is source
            assert source.system_capacity == capacity

        assert remaining.num_graphs == 3
        assert remaining.group_idx[: remaining.num_graphs].tolist() == [0, 0, 0]
        assert remaining.group_layout.num_graphs_per_group.tolist() == [3]
        torch.testing.assert_close(
            remaining.positions[: remaining.num_nodes], positions[3:]
        )

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

    @pytest.mark.parametrize(
        ("grouped_receiver", "grouped_source"),
        [(True, False), (False, True)],
    )
    def test_put_rejects_mixed_grouping_before_mutation(
        self,
        grouped_receiver,
        grouped_source,
    ):
        batch = _batch()
        source = _batch()
        original_positions = batch.positions.clone()
        original_layout = None
        if grouped_receiver:
            batch.set_group_layout(torch.tensor([0, 0, 1, 1, 1]))
            original_layout = batch.group_layout
        if grouped_source:
            source.set_group_layout(torch.tensor([0, 0, 1, 1, 1]))
        copied_mask = torch.zeros(source.num_graphs, dtype=torch.bool)

        with pytest.raises(ValueError, match="both be grouped or ungrouped"):
            batch.put(
                source,
                torch.ones(source.num_graphs, dtype=torch.bool),
                copied_mask=copied_mask,
            )

        torch.testing.assert_close(batch.positions, original_positions)
        assert not copied_mask.any()
        if original_layout is not None:
            assert batch.group_layout is original_layout

    @pytest.mark.parametrize("preallocated", [False, True])
    @pytest.mark.parametrize("device", ["cpu", "cuda"])
    def test_zero_preserves_capacity_and_builds_empty_layout(
        self, preallocated: bool, device: str
    ) -> None:
        """Allocated label rows must not become groups after zeroing a batch."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA is not available")
        batch = _batch().index_select([0, 1, 2]).to(device)
        batch.set_group_layout(torch.tensor([0, 0, 1], device=device))
        if preallocated:
            batch = Batch.empty(
                num_systems=6,
                num_nodes=12,
                num_edges=0,
                template=batch,
                device=device,
            )
        else:
            assert batch.group_layout.num_groups == 2
        capacity = batch.system_capacity
        labels = batch.group_idx
        positions = batch.positions

        for _ in range(2):
            batch.zero()
            layout = batch.group_layout

            assert batch.num_graphs == 0
            assert batch.system_capacity == capacity
            assert batch.group_idx is labels
            assert batch.positions is positions
            assert layout.num_groups == 0
            assert layout.group_idx.numel() == 0
            assert layout.graph_rank.numel() == 0
            assert layout.node_to_group.numel() == 0
            assert layout.num_graphs_per_group.numel() == 0
            assert layout.group_ptr.tolist() == [0]
            assert layout.device == batch.device

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

    @pytest.mark.parametrize("cached_receiver", [False, True])
    def test_repeated_append_rebases_matching_labels_without_building_layout(
        self, cached_receiver: bool, device: str
    ) -> None:
        """Equal join labels stay separate with cached and lazy receiver layouts."""
        batch = _batch().to(device)
        batch.set_group_layout(torch.zeros(5, dtype=torch.long, device=device))
        other = batch.clone()
        original_positions = other.positions.clone()
        if not cached_receiver:
            batch.group_idx = batch.group_idx.clone()

        with patch.object(
            GroupLayout, "from_batch", side_effect=AssertionError("eager group layout")
        ):
            for groups in [2, 3]:
                batch.append(other)
                assert getattr(batch, "_group_layout", None) is None
                assert batch.group_idx.tolist() == [
                    i for i in range(groups) for _ in range(5)
                ]

        assert batch.group_layout.num_graphs_per_group.tolist() == [5, 5, 5]
        assert other.group_idx.tolist() == [0] * 5
        torch.testing.assert_close(other.positions, original_positions)

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

    def test_set_group_layout_normalizes_stored_metadata(self):
        batch = _batch()
        batch["group_idx"] = torch.tensor([4, 4, 1, 1, 1])

        with pytest.raises(ValueError, match="dense, contiguous"):
            _ = batch.group_layout

        batch.set_group_layout(batch.group_idx)

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
