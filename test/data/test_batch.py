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
"""Comprehensive tests for Batch (graph-aware Pydantic batch on MultiLevelStorage)."""

from __future__ import annotations

import pytest
import torch

from nvalchemi.data.atomic_data import AtomicData
from nvalchemi.data.batch import Batch
from nvalchemi.data.level_storage import MultiLevelStorage, UniformLevelStorage


def _minimal_atomic_data(
    num_nodes: int = 4,
    num_edges: int = 0,
    device: str | torch.device = "cpu",
) -> AtomicData:
    """Build minimal AtomicData for tests."""
    positions = torch.randn(num_nodes, 3, device=device)
    atomic_numbers = torch.ones(num_nodes, dtype=torch.long, device=device)
    kwargs: dict = {"positions": positions, "atomic_numbers": atomic_numbers}
    if num_edges > 0:
        edge_index = torch.zeros(num_edges, 2, dtype=torch.long, device=device)
        kwargs["edge_index"] = edge_index
    return AtomicData(**kwargs)


def _atomic_data_with_system(
    num_nodes: int = 2,
    device: str | torch.device = "cpu",
) -> AtomicData:
    """AtomicData with a system-level field so Batch has a 'system' group."""
    return AtomicData(
        positions=torch.randn(num_nodes, 3, device=device),
        atomic_numbers=torch.ones(num_nodes, dtype=torch.long, device=device),
        energies=torch.tensor([[0.0]], device=device),
    )


def _atomic_data_with_edges_and_system(
    num_nodes: int = 2,
    num_edges: int = 2,
    device: str | torch.device = "cpu",
) -> AtomicData:
    """AtomicData with node, edge, and system fields so Batch has all three groups."""
    return AtomicData(
        positions=torch.randn(num_nodes, 3, device=device),
        atomic_numbers=torch.ones(num_nodes, dtype=torch.long, device=device),
        edge_index=torch.zeros(num_edges, 2, dtype=torch.long, device=device),
        energies=torch.tensor([[0.0]], device=device),
    )


# -----------------------------------------------------------------------------
# Construction
# -----------------------------------------------------------------------------
class TestBatchConstruction:
    """Tests for Batch.from_data_list and default construction."""

    def test_from_data_list_empty_raises(self):
        with pytest.raises(ValueError, match="empty data list"):
            Batch.from_data_list([])

    def test_from_data_list_single(self):
        d = _minimal_atomic_data(4)
        batch = Batch.from_data_list([d])
        assert batch.num_graphs == 1
        assert batch.num_nodes == 4
        assert batch.num_edges == 0
        assert batch.batch.shape == (4,)
        assert batch.ptr.tolist() == [0, 4]
        assert batch.num_nodes_list == [4]
        # No edges group when input has no edge data, so num_edges_list is []
        assert batch.num_edges_list == []

    def test_from_data_list_multiple(self):
        d1 = _minimal_atomic_data(3)
        d2 = _minimal_atomic_data(5)
        batch = Batch.from_data_list([d1, d2])
        assert batch.num_graphs == 2
        assert batch.num_nodes == 8
        assert batch.batch.shape == (8,)
        assert (batch.batch[:3] == 0).all()
        assert (batch.batch[3:8] == 1).all()
        assert batch.ptr.tolist() == [0, 3, 8]
        assert batch.num_nodes_list == [3, 5]
        assert batch.max_num_nodes == 5

    def test_from_data_list_infers_device(self):
        d = _minimal_atomic_data(2)
        batch = Batch.from_data_list([d], device=None)
        assert batch.device == d.positions.device

    def test_from_data_list_exclude_keys(self):
        d = _minimal_atomic_data(2)
        batch = Batch.from_data_list([d], exclude_keys=["positions"])
        assert "positions" not in batch

    def test_storage_default_empty(self):
        batch = Batch(device="cpu")
        assert batch.num_graphs == 0
        assert batch.num_nodes == 0
        assert batch.num_edges == 0

    def test_batch_with_system_only_storage(self):
        """Batch built with only system group: batch, ptr, num_nodes_list, etc. hit None branches."""
        system = UniformLevelStorage(
            data={"energies": torch.randn(2, 1)},
            device="cpu",
            validate=False,
        )
        storage = MultiLevelStorage(
            groups={"system": system},
            attr_map=None,
            validate=False,
        )
        batch = Batch._construct(
            device=torch.device("cpu"),
            keys={"system": {"energies"}},
            storage=storage,
        )
        assert batch.num_graphs == 2
        assert batch.num_nodes == 0
        assert batch.num_edges == 0
        assert batch.batch.shape == (0,)
        assert batch.ptr.tolist() == [0]
        assert batch.num_nodes_list == []
        assert batch.num_edges_list == []
        assert batch.num_nodes_per_graph.shape == (0,)
        assert batch.num_edges_per_graph.shape == (0,)
        assert batch.max_num_nodes == 0
        assert batch.batch_size == 2


# -----------------------------------------------------------------------------
# Batch.zero() tests
# -----------------------------------------------------------------------------
class TestBatchZero:
    """Tests for Batch.zero() method that resets pre-allocated batches."""

    def test_batch_zero_resets_state(self):
        """zero() should reset num_graphs to 0 while preserving system_capacity."""
        template = _atomic_data_with_system(num_nodes=2)
        batch = Batch.empty(
            num_systems=10,
            num_nodes=100,
            num_edges=200,
            template=template,
        )

        # Verify initial state (empty but allocated)
        initial_capacity = batch.system_capacity
        assert initial_capacity == 10
        assert batch.num_graphs == 0

        # Call zero and verify state is reset
        batch.zero()

        assert batch.num_graphs == 0
        assert batch.system_capacity == 10

    def test_batch_zero_zeros_tensor_data(self):
        """zero() should zero all leaf tensors in the storage."""
        template = _atomic_data_with_system(num_nodes=2)
        batch = Batch.empty(
            num_systems=5,
            num_nodes=50,
            num_edges=100,
            template=template,
        )

        # Get a reference to the positions tensor and verify it's zeroed
        # after calling zero()
        batch.zero()

        # Check that the underlying data tensors are zeroed
        atoms_group = batch._atoms_group
        if atoms_group is not None:
            for key, tensor in atoms_group._data.items():
                assert (tensor == 0).all(), f"Tensor '{key}' should be zeroed"

        system_group = batch._system_group
        if system_group is not None:
            for key, tensor in system_group._data.items():
                assert (tensor == 0).all(), f"System tensor '{key}' should be zeroed"

    def test_batch_zero_resets_segment_lengths(self):
        """zero() should reset segment_lengths to empty for segmented groups."""
        template = _atomic_data_with_system(num_nodes=2)
        batch = Batch.empty(
            num_systems=5,
            num_nodes=50,
            num_edges=100,
            template=template,
        )

        batch.zero()

        atoms_group = batch._atoms_group
        if atoms_group is not None:
            assert len(atoms_group.segment_lengths) == 0

    def test_batch_zero_preserves_capacity_with_edges(self):
        """zero() should work correctly with batches that have edge data."""
        template = _atomic_data_with_edges_and_system(num_nodes=3, num_edges=4)
        batch = Batch.empty(
            num_systems=8,
            num_nodes=80,
            num_edges=160,
            template=template,
        )

        batch.zero()

        assert batch.num_graphs == 0
        assert batch.system_capacity == 8
        assert batch.num_nodes == 0
        assert batch.num_edges == 0

    def test_batch_zero_idempotent(self):
        """Calling zero() multiple times should be idempotent."""
        template = _atomic_data_with_system(num_nodes=2)
        batch = Batch.empty(
            num_systems=5,
            num_nodes=50,
            num_edges=100,
            template=template,
        )

        batch.zero()
        batch.zero()
        batch.zero()

        assert batch.num_graphs == 0
        assert batch.system_capacity == 5


# -----------------------------------------------------------------------------
# Per-graph reconstruction
# -----------------------------------------------------------------------------
class TestBatchReconstruction:
    """Tests for get_data and to_data_list."""

    def test_get_data_single(self):
        d = _minimal_atomic_data(4)
        batch = Batch.from_data_list([d])
        out = batch.get_data(0)
        assert isinstance(out, AtomicData)
        assert out.num_nodes == 4
        assert torch.allclose(out.positions, d.positions)
        assert torch.equal(out.atomic_numbers, d.atomic_numbers)

    def test_get_data_multiple(self):
        d1 = _minimal_atomic_data(3)
        d2 = _minimal_atomic_data(5)
        batch = Batch.from_data_list([d1, d2])
        o1 = batch.get_data(0)
        o2 = batch.get_data(1)
        assert o1.num_nodes == 3 and o2.num_nodes == 5
        assert torch.allclose(o1.positions, d1.positions)
        assert torch.allclose(o2.positions, d2.positions)

    def test_get_data_negative_index(self):
        d1 = _minimal_atomic_data(2)
        d2 = _minimal_atomic_data(3)
        batch = Batch.from_data_list([d1, d2])
        last = batch.get_data(-1)
        assert last.num_nodes == 3

    def test_to_data_list(self):
        d1 = _minimal_atomic_data(2)
        d2 = _minimal_atomic_data(3)
        batch = Batch.from_data_list([d1, d2])
        lst = batch.to_data_list()
        assert len(lst) == 2
        assert lst[0].num_nodes == 2 and lst[1].num_nodes == 3


# -----------------------------------------------------------------------------
# Indexing / selection
# -----------------------------------------------------------------------------
class TestBatchIndexing:
    """Tests for index_select and __getitem__ with indices."""

    def test_index_select_slice(self):
        batch = Batch.from_data_list(
            [
                _minimal_atomic_data(2),
                _minimal_atomic_data(3),
                _minimal_atomic_data(4),
            ]
        )
        sub = batch[1:3]
        assert isinstance(sub, Batch)
        assert sub.num_graphs == 2
        assert sub.num_nodes_list == [3, 4]
        assert sub.num_nodes == 7

    def test_index_select_int(self):
        batch = Batch.from_data_list(
            [
                _minimal_atomic_data(2),
                _minimal_atomic_data(3),
            ]
        )
        one = batch[1]
        assert isinstance(one, AtomicData)
        assert one.num_nodes == 3

    def test_index_select_tensor(self):
        batch = Batch.from_data_list(
            [
                _minimal_atomic_data(2),
                _minimal_atomic_data(3),
                _minimal_atomic_data(4),
            ]
        )
        sub = batch[torch.tensor([0, 2])]
        assert sub.num_graphs == 2
        assert sub.num_nodes_list == [2, 4]

    def test_index_select_list(self):
        batch = Batch.from_data_list(
            [
                _minimal_atomic_data(2),
                _minimal_atomic_data(3),
            ]
        )
        sub = batch[[1, 0]]
        assert sub.num_graphs == 2
        assert sub.num_nodes_list == [3, 2]

    def test_index_select_with_edges_applies_edge_index_correction(self):
        """index_select on a batch with edges corrects edge_index offsets."""
        data_list = [
            _atomic_data_with_edges_and_system(num_nodes=2, num_edges=3),
            _atomic_data_with_edges_and_system(num_nodes=3, num_edges=2),
            _atomic_data_with_edges_and_system(num_nodes=1, num_edges=1),
        ]
        batch = Batch.from_data_list(data_list)
        sub = batch[torch.tensor([0, 2])]
        assert sub.num_graphs == 2
        assert sub.num_nodes_list == [2, 1]
        assert sub.num_edges_list == [3, 1]
        d0 = sub.get_data(0)
        d1 = sub.get_data(1)
        assert d0.edge_index is not None and d1.edge_index is not None

    def test_index_select_normalize_bool_tensor(self):
        batch = Batch.from_data_list(
            [
                _minimal_atomic_data(2),
                _minimal_atomic_data(3),
                _minimal_atomic_data(1),
            ]
        )
        mask = torch.tensor([True, False, True])
        sub = batch[mask]
        assert sub.num_graphs == 2
        assert sub.num_nodes_list == [2, 1]

    def test_index_select_float_tensor_raises(self):
        batch = Batch.from_data_list([_minimal_atomic_data(2)])
        with pytest.raises(IndexError, match="Tensor index must be integer or bool"):
            _ = batch[torch.tensor([0.0, 1.0])]

    def test_index_select_empty_raises(self):
        batch = Batch.from_data_list([_minimal_atomic_data(2)])
        with pytest.raises(IndexError, match="Index is empty"):
            _ = batch[torch.tensor([], dtype=torch.long)]

    def test_index_select_unsupported_type_raises(self):
        batch = Batch.from_data_list([_minimal_atomic_data(2)])
        with pytest.raises(IndexError, match="Unsupported index type"):
            _ = batch[1.5]

    def test_getitem_attr_by_name(self):
        batch = Batch.from_data_list([_minimal_atomic_data(3)])
        pos = batch["positions"]
        assert pos.shape == (3, 3)

    def test_model_dump_exclude_none(self):
        batch = Batch.from_data_list([_minimal_atomic_data(2)])
        d = batch.model_dump(exclude_none=True)
        assert isinstance(d, dict)
        assert "positions" in d
        assert "device" in d

    def test_pin_memory(self):
        batch = Batch.from_data_list([_minimal_atomic_data(2)])
        pinned = batch.pin_memory()
        assert pinned.num_graphs == batch.num_graphs
        assert pinned["positions"].is_pinned()


# -----------------------------------------------------------------------------
# Mutation and add_key
# -----------------------------------------------------------------------------
class TestBatchMutation:
    """Tests for append, append_data, add_key, __setitem__."""

    def test_append(self):
        b1 = Batch.from_data_list([_minimal_atomic_data(2), _minimal_atomic_data(3)])
        b2 = Batch.from_data_list([_minimal_atomic_data(4)])
        b1.append(b2)
        assert b1.num_graphs == 3
        assert b1.num_nodes_list == [2, 3, 4]
        assert b1.num_nodes == 9

    def test_append_data(self):
        batch = Batch.from_data_list([_minimal_atomic_data(2)])
        batch.append_data([_minimal_atomic_data(3), _minimal_atomic_data(1)])
        assert batch.num_graphs == 3
        assert batch.num_nodes_list == [2, 3, 1]

    def test_add_key_system(self):
        # Batch must have a system group (from data with system-level keys)
        batch = Batch.from_data_list(
            [
                _atomic_data_with_system(2),
                _atomic_data_with_system(3),
            ]
        )
        # (1, 3, 3) per graph (AtomicData-style) -> leading 1 squeezed, then stack -> (2, 3, 3)
        batch.add_key(
            "virials",
            [torch.randn(1, 3, 3), torch.randn(1, 3, 3)],
            level="system",
        )
        assert "virials" in batch
        assert batch["virials"].shape == (2, 3, 3)

    def test_add_key_node(self):
        batch = Batch.from_data_list(
            [
                _minimal_atomic_data(2),
                _minimal_atomic_data(3),
            ]
        )
        batch.add_key(
            "forces",
            [torch.randn(2, 3), torch.randn(3, 3)],
            level="node",
        )
        assert batch["forces"].shape == (5, 3)

    def test_add_key_overwrite(self):
        batch = Batch.from_data_list([_atomic_data_with_system(2)])
        batch.add_key("virials", [torch.zeros(1, 3, 3)], level="system")
        batch.add_key("virials", [torch.ones(1, 3, 3)], level="system", overwrite=True)
        assert batch["virials"].eq(1).all()

    def test_add_key_exists_raises(self):
        batch = Batch.from_data_list([_atomic_data_with_system(2)])
        batch.add_key("virials", [torch.zeros(1, 3, 3)], level="system")
        with pytest.raises(ValueError, match="already exists"):
            batch.add_key(
                "virials", [torch.ones(1, 3, 3)], level="system", overwrite=False
            )

    def test_append_data_empty_raises(self):
        batch = Batch.from_data_list([_minimal_atomic_data(2)])
        with pytest.raises(ValueError, match="No data provided"):
            batch.append_data([])

    def test_add_key_group_not_found_raises(self):
        """add_key with level='edge' when batch has no edges group raises."""
        batch = Batch.from_data_list([_atomic_data_with_system(2)])
        with pytest.raises(ValueError, match="Group 'edges' not found"):
            batch.add_key("edge_attr", [torch.randn(1, 4)], level="edge")


# -----------------------------------------------------------------------------
# Round-trip: added keys appear correctly in to_data_list()
# -----------------------------------------------------------------------------
class TestBatchRoundTripAddedKeys:
    """Test that keys added to a Batch (e.g. by MD code) are correctly stored in
    AtomicData when converting back via to_data_list() / get_data().
    """

    def test_added_node_edge_system_keys_round_trip(self):
        # Batch with nodes, edges, and system so we can add keys at all levels
        data_list_in = [
            _atomic_data_with_edges_and_system(num_nodes=2, num_edges=4),
            _atomic_data_with_edges_and_system(num_nodes=3, num_edges=5),
        ]
        batch = Batch.from_data_list(data_list_in)
        assert batch.num_graphs == 2
        assert batch.num_nodes_list == [2, 3]
        # Per-graph values we will add and then verify after round-trip
        velocities_list = [torch.randn(2, 3), torch.randn(3, 3)]
        edge_emb_list = [torch.randn(4, 8), torch.randn(5, 8)]
        temperature_list = [torch.tensor([[1.5]]), torch.tensor([[2.0]])]

        batch.add_key("velocities_tmp", velocities_list, level="node")
        batch.add_key("edge_embeddings", edge_emb_list, level="edge")
        batch.add_key("temperature", temperature_list, level="system")

        data_list_out = batch.to_data_list()

        assert len(data_list_out) == 2
        for i in range(2):
            out = data_list_out[i]
            assert hasattr(out, "velocities_tmp") and out.velocities_tmp is not None
            assert out.velocities_tmp.shape == velocities_list[i].shape
            assert torch.allclose(out.velocities_tmp, velocities_list[i])

            assert hasattr(out, "edge_embeddings") and out.edge_embeddings is not None
            assert out.edge_embeddings.shape == edge_emb_list[i].shape
            assert torch.allclose(out.edge_embeddings, edge_emb_list[i])

            assert hasattr(out, "temperature") and out.temperature is not None
            t = out.temperature
            expected = temperature_list[i].squeeze(0)
            if t.dim() == 0:
                assert expected.numel() == 1
                assert torch.allclose(t, expected.view(()))
            else:
                assert torch.allclose(t, expected)

        # get_data(i) should match to_data_list()[i] for added keys
        for i in range(2):
            single = batch.get_data(i)
            assert torch.allclose(
                single.velocities_tmp, data_list_out[i].velocities_tmp
            )
            assert torch.allclose(
                single.edge_embeddings, data_list_out[i].edge_embeddings
            )
            assert torch.allclose(
                torch.as_tensor(single.temperature),
                torch.as_tensor(data_list_out[i].temperature),
            )

    def test_dynamic_system_key_survives_full_round_trip(self) -> None:
        """Dynamically-added system properties (e.g. system_id) must survive
        the full HostMemory-style round-trip:
        from_data_list → index_select → to_data_list → .to(cpu) → from_data_list.

        Regression test for the crash in examples/intermediate/04_inflight_batching.py.
        """
        d1 = AtomicData(
            positions=torch.randn(3, 3),
            atomic_numbers=torch.tensor([6, 6, 6]),
        )
        d1.add_system_property("system_id", torch.tensor([[0]], dtype=torch.long))

        d2 = AtomicData(
            positions=torch.randn(5, 3),
            atomic_numbers=torch.tensor([8, 8, 8, 8, 8]),
        )
        d2.add_system_property("system_id", torch.tensor([[1]], dtype=torch.long))

        batch = Batch.from_data_list([d1, d2])
        assert hasattr(batch, "system_id")
        assert batch.system_id.shape == (2, 1)

        # index_select → to_data_list (what ConvergedSnapshotHook does)
        sub = batch.index_select([0])
        data_list = sub.to_data_list()
        assert "system_id" in data_list[0].__system_keys__

        # .to(cpu) (what HostMemory.write does)
        cpu_data = [d.to(torch.device("cpu")) for d in data_list]
        assert "system_id" in cpu_data[0].__system_keys__

        # from_data_list (what HostMemory.read / drain does)
        result = Batch.from_data_list(cpu_data)
        assert hasattr(result, "system_id")
        assert result.system_id.squeeze(-1).tolist() == [0]


# -----------------------------------------------------------------------------
# Device, clone, contiguous, serialization
# -----------------------------------------------------------------------------
class TestBatchDeviceAndCopy:
    """Tests for to, clone, cpu, cuda, contiguous, pin_memory."""

    def test_to_device(self):
        batch = Batch.from_data_list([_minimal_atomic_data(2)])
        batch = batch.to("cpu")
        assert batch.device.type == "cpu"
        assert batch["positions"].device.type == "cpu"

    def test_clone(self):
        batch = Batch.from_data_list([_minimal_atomic_data(2)])
        c = batch.clone()
        assert c is not batch
        assert c.num_graphs == batch.num_graphs
        assert c["positions"] is not batch["positions"]

    def test_cpu_cuda(self):
        batch = Batch.from_data_list([_minimal_atomic_data(2)])
        batch_cpu = batch.cpu()
        assert batch_cpu.device.type == "cpu"

    def test_contiguous(self):
        batch = Batch.from_data_list([_minimal_atomic_data(2)])
        batch = batch.contiguous()
        assert batch["positions"].is_contiguous()


class TestBatchSerialization:
    """Tests for model_dump and round-trip."""

    def test_model_dump_contains_tensors_and_metadata(self):
        batch = Batch.from_data_list([_minimal_atomic_data(2)])
        d = batch.model_dump()
        assert "device" in d
        assert "num_graphs" in d
        assert "batch" in d
        assert "ptr" in d
        assert "positions" in d
        assert "atomic_numbers" in d
        assert d["num_graphs"] == 1
        assert d["positions"].shape == (2, 3)


# -----------------------------------------------------------------------------
# Len, iter, contains, repr
# -----------------------------------------------------------------------------
class TestBatchProtocols:
    """Tests for __len__, __iter__, __contains__, __repr__."""

    def test_len(self):
        batch = Batch.from_data_list(
            [
                _minimal_atomic_data(2),
                _minimal_atomic_data(3),
            ]
        )
        assert len(batch) == 2

    def test_contains(self):
        batch = Batch.from_data_list([_minimal_atomic_data(2)])
        assert "positions" in batch
        assert "nonexistent" not in batch

    def test_iter_items(self):
        batch = Batch.from_data_list([_minimal_atomic_data(2)])
        items = list(batch)
        assert any(k == "positions" for k, _ in items)

    def test_contains_missing_key(self):
        batch = Batch.from_data_list([_minimal_atomic_data(2)])
        assert "positions" in batch
        assert "nonexistent_key" not in batch

    def test_setitem_roundtrip(self):
        batch = Batch.from_data_list([_minimal_atomic_data(2)])
        new_forces = torch.randn(2, 3)
        batch["forces"] = new_forces
        assert "forces" in batch
        assert torch.allclose(batch["forces"], new_forces)

    def test_getattr_missing_raises(self):
        batch = Batch.from_data_list([_minimal_atomic_data(2)])
        with pytest.raises(AttributeError, match="no attribute 'nonexistent'"):
            _ = batch.nonexistent


# -----------------------------------------------------------------------------
# put and defrag
# -----------------------------------------------------------------------------
class TestBatchPutDefrag:
    """Tests for buffer.put (two-phase: fit mask per level, logical_and, then put) and defrag."""

    def test_put_stores_copied_mask_on_src(self):
        """When copied_mask is None, put stores _copied_mask (combined fit mask) on src_batch."""
        buffer = Batch.from_data_list(
            [
                _minimal_atomic_data(2),
                _minimal_atomic_data(2),
            ]
        )
        src_batch = Batch.from_data_list(
            [
                _minimal_atomic_data(2),
                _minimal_atomic_data(2),
            ]
        )
        mask = torch.tensor([False, False])
        buffer.put(src_batch, mask)
        assert hasattr(src_batch, "_copied_mask")
        assert src_batch._copied_mask.shape == (2,)
        assert src_batch._copied_mask.sum().item() == 0

    def test_put_with_copied_mask_in_place(self):
        """put with copied_mask provided sets it to the combined fit mask (in place)."""
        buffer = Batch.from_data_list(
            [
                _minimal_atomic_data(2),
                _minimal_atomic_data(2),
            ]
        )
        src_batch = Batch.from_data_list([_minimal_atomic_data(2)])
        mask = torch.tensor([False])
        copied_mask = torch.zeros(1, dtype=torch.bool)
        buffer.put(src_batch, mask, copied_mask=copied_mask)
        assert copied_mask.shape == (1,)
        assert copied_mask.sum().item() == 0

    def test_put_copied_mask_when_same_size_buffer_no_room(self):
        """When buffer and src have same size, fixed storage has no room to append; copied_mask all False."""
        buffer = Batch.from_data_list(
            [
                _minimal_atomic_data(2),
                _minimal_atomic_data(2),
            ]
        )
        src_batch = Batch.from_data_list(
            [_minimal_atomic_data(2), _minimal_atomic_data(2)],
        )
        mask = torch.tensor([True, True])
        copied_mask = torch.zeros(2, dtype=torch.bool)
        buffer.put(src_batch, mask, copied_mask=copied_mask)
        # No room to append (buffer has no extra batch_ptr/data capacity); combined fit is all False
        assert copied_mask.sum().item() == 0
        assert copied_mask.shape == (2,)

    def test_put_no_room_after_full(self):
        """When buffer has no room (fixed storage), second put copies nothing; copied_mask all False."""
        buffer = Batch.from_data_list(
            [_minimal_atomic_data(2), _minimal_atomic_data(2)],
        )
        src_first = Batch.from_data_list(
            [_minimal_atomic_data(2), _minimal_atomic_data(2)],
        )
        mask = torch.tensor([True, True])
        buffer.put(src_first, mask)
        assert buffer.num_graphs == 2
        src_second = Batch.from_data_list(
            [_minimal_atomic_data(2), _minimal_atomic_data(2)],
        )
        copied_mask = torch.ones(2, dtype=torch.bool)
        buffer.put(src_second, mask, copied_mask=copied_mask)
        # No room in buffer; combined fit mask is all False
        assert copied_mask.sum().item() == 0
        assert buffer.num_graphs == 2

    def test_defrag_with_copied_mask(self):
        """defrag(copied_mask) compacts batch by removing graphs where copied_mask is True."""
        batch = Batch.from_data_list(
            [
                _minimal_atomic_data(2),
                _minimal_atomic_data(3),
                _minimal_atomic_data(1),
            ]
        )
        assert batch.num_graphs == 3
        assert batch.num_nodes_list == [2, 3, 1]
        # Mark graphs 0 and 2 as "copied" (to be removed); keep graph 1
        copied_mask = torch.tensor([True, False, True])
        batch.defrag(copied_mask=copied_mask)
        assert batch.num_graphs == 1
        assert batch.num_nodes_list == [3]
        assert batch.num_nodes == 3

    def test_defrag_requires_copied_mask_or_prior_put(self):
        """defrag() without copied_mask and without prior put raises."""
        batch = Batch.from_data_list([_minimal_atomic_data(2)])
        with pytest.raises(
            ValueError, match="defrag requires copied_mask or a prior put"
        ):
            batch.defrag()

    def test_defrag_segment_lengths_consistency(self):
        """After defrag, num_nodes_per_graph length equals num_graphs."""
        # Create batch of 4 graphs with different atom counts
        batch = Batch.from_data_list(
            [
                _minimal_atomic_data(3),
                _minimal_atomic_data(5),
                _minimal_atomic_data(4),
                _minimal_atomic_data(3),
            ]
        )
        assert batch.num_graphs == 4
        assert batch.num_nodes == 15

        # Create send buffer and put first 2 graphs
        template = _minimal_atomic_data(1)
        buffer = Batch.empty(
            num_systems=4, num_nodes=50, num_edges=0, template=template
        )
        mask = torch.tensor([True, True, False, False])
        buffer.put(batch, mask)

        # Defrag the source batch (removes graphs 0 and 1)
        batch.defrag()

        # After defrag: 2 graphs remain (graphs 2 and 3 with 4 and 3 atoms)
        assert batch.num_graphs == 2
        assert len(batch.num_nodes_per_graph) == 2
        assert len(batch.num_nodes_list) == 2
        assert batch.num_nodes_per_graph.sum().item() == batch.num_nodes
        # This operation should succeed without error
        expanded = torch.repeat_interleave(
            torch.ones(2, dtype=torch.bool), batch.num_nodes_per_graph
        )
        assert len(expanded) == batch.num_nodes

    def test_trim_removes_marked_graphs(self):
        """trim() returns a new batch with only kept graphs."""
        batch = Batch.from_data_list(
            [
                _minimal_atomic_data(2),
                _minimal_atomic_data(3),
                _minimal_atomic_data(1),
            ]
        )
        copied_mask = torch.tensor([True, False, True])
        trimmed = batch.trim(copied_mask=copied_mask)
        assert trimmed is not None
        assert trimmed.num_graphs == 1
        assert trimmed.num_nodes == 3
        assert trimmed.num_nodes_list == [3]

    def test_trim_returns_none_when_all_removed(self):
        """trim() returns None when all graphs are marked for removal."""
        batch = Batch.from_data_list([_minimal_atomic_data(2), _minimal_atomic_data(3)])
        copied_mask = torch.tensor([True, True])
        result = batch.trim(copied_mask=copied_mask)
        assert result is None

    def test_trim_requires_copied_mask_or_prior_put(self):
        """trim() without copied_mask and without prior put raises."""
        batch = Batch.from_data_list([_minimal_atomic_data(2)])
        with pytest.raises(
            ValueError, match="trim requires copied_mask or a prior put"
        ):
            batch.trim()

    def test_trim_preserves_original_batch(self):
        """trim() does not modify the original batch."""
        batch = Batch.from_data_list(
            [
                _minimal_atomic_data(2),
                _minimal_atomic_data(3),
            ]
        )
        original_num_graphs = batch.num_graphs
        original_num_nodes = batch.num_nodes
        copied_mask = torch.tensor([True, False])
        batch.trim(copied_mask=copied_mask)
        assert batch.num_graphs == original_num_graphs
        assert batch.num_nodes == original_num_nodes

    def test_trim_tensors_are_tight(self):
        """After trim, all storage tensors match logical counts exactly."""
        batch = Batch.from_data_list(
            [
                _minimal_atomic_data(4),
                _minimal_atomic_data(3),
                _minimal_atomic_data(5),
                _minimal_atomic_data(3),
            ]
        )
        copied_mask = torch.tensor([True, True, False, False])
        trimmed = batch.trim(copied_mask=copied_mask)
        assert trimmed is not None
        # Node-level tensors match num_nodes exactly
        assert trimmed.positions.shape[0] == trimmed.num_nodes
        assert trimmed.num_nodes == 8  # 5 + 3
        # Graph-level: num_nodes_per_graph length matches num_graphs
        assert len(trimmed.num_nodes_per_graph) == trimmed.num_graphs
        assert trimmed.num_graphs == 2
        # batch assignment tensor matches num_nodes
        assert trimmed.batch.shape[0] == trimmed.num_nodes

    def test_trim_uses_copied_mask_from_put(self):
        """trim() uses _copied_mask from a prior put() if no mask is given."""
        batch = Batch.from_data_list(
            [
                _minimal_atomic_data(3),
                _minimal_atomic_data(5),
                _minimal_atomic_data(4),
                _minimal_atomic_data(3),
            ]
        )
        template = _minimal_atomic_data(1)
        buffer = Batch.empty(
            num_systems=4, num_nodes=50, num_edges=0, template=template
        )
        mask = torch.tensor([True, True, False, False])
        buffer.put(batch, mask)

        # trim should use the _copied_mask stored by put
        trimmed = batch.trim()
        assert trimmed is not None
        assert trimmed.num_graphs == 2
        assert trimmed.num_nodes == 7  # 4 + 3
        # Tensors are tight
        assert trimmed.positions.shape[0] == 7


class TestBatchRecvHandleWait:
    """Tests for _BatchRecvHandle.wait() non-blocking receive protocol."""

    def test_wait_uses_irecv_not_recv(self):
        """wait() uses dist.irecv (non-blocking) and waits on all handles."""
        from unittest.mock import MagicMock, patch

        from nvalchemi.data.batch import _BatchRecvHandle

        template = Batch.from_data_list([_atomic_data_with_system(num_nodes=5)])

        meta = torch.tensor([2, 10, 0], dtype=torch.int64, device="cpu")
        mock_meta_handle = MagicMock()

        handle = _BatchRecvHandle(
            meta=meta,
            meta_handle=mock_meta_handle,
            src=0,
            device=torch.device("cpu"),
            template=template,
            base_tag=100,
            group=None,
        )

        irecv_handles = []

        def make_irecv_handle(*args, **kwargs):
            h = MagicMock()
            irecv_handles.append(h)
            return h

        mock_td_handles = [MagicMock(), MagicMock()]

        with (
            patch("torch.distributed.recv") as mock_recv,
            patch(
                "torch.distributed.irecv", side_effect=make_irecv_handle
            ) as mock_irecv,
            patch("tensordict.TensorDict.irecv", return_value=mock_td_handles),
        ):
            _ = handle.wait()

        mock_recv.assert_not_called()
        assert mock_irecv.call_count >= 1
        mock_meta_handle.wait.assert_called_once()
        for h in irecv_handles:
            assert h.wait.called, "irecv handle should have wait() called"
        for h in mock_td_handles:
            assert h.wait.called, "TensorDict handle should have wait() called"


class TestBatchRecvHandleEmpty:
    """Tests for _BatchRecvHandle.wait() with empty (sentinel) batch."""

    def test_empty_batch_returns_immediately(self):
        """wait() with 0-graph meta returns empty Batch without extra irecv."""
        from unittest.mock import MagicMock, patch

        from nvalchemi.data.batch import _BatchRecvHandle

        template = Batch.from_data_list([_atomic_data_with_system(num_nodes=2)])

        meta = torch.tensor([0, 0, 0], dtype=torch.int64, device="cpu")
        mock_meta_handle = MagicMock()

        handle = _BatchRecvHandle(
            meta=meta,
            meta_handle=mock_meta_handle,
            src=0,
            device=torch.device("cpu"),
            template=template,
            base_tag=100,
            group=None,
        )

        with (
            patch("torch.distributed.recv") as mock_recv,
            patch("torch.distributed.irecv") as mock_irecv,
        ):
            result = handle.wait()

        assert result.num_graphs == 0
        mock_recv.assert_not_called()
        mock_irecv.assert_not_called()
        mock_meta_handle.wait.assert_called_once()


class TestBatchIsendIrecvTagAlignment:
    """Tests that isend and irecv+wait use matching tag sequences."""

    def test_tag_sequences_match(self):
        """isend and irecv+wait protocol must use identical tag sequences."""
        from unittest.mock import MagicMock, patch

        batch = Batch.from_data_list(
            [_atomic_data_with_edges_and_system(num_nodes=3, num_edges=2)]
        )

        send_tags: list[int] = []
        recv_tags: list[int] = []

        def capture_isend_tag(*args, **kwargs):
            if "tag" in kwargs:
                send_tags.append(kwargs["tag"])
            mock_handle = MagicMock()
            return mock_handle

        def capture_irecv_tag(*args, **kwargs):
            if "tag" in kwargs:
                recv_tags.append(kwargs["tag"])
            mock_handle = MagicMock()
            return mock_handle

        def capture_td_isend(
            self, dst=None, init_tag=None, group=None, return_early=False
        ):
            for i in range(3):
                send_tags.append(init_tag + i)  # noqa: PERF401
            return [MagicMock()] if return_early else None

        def capture_td_irecv(
            self, src=None, init_tag=None, group=None, return_premature=False
        ):
            for i in range(3):
                recv_tags.append(init_tag + i)  # noqa: PERF401
            return [MagicMock()] if return_premature else None

        with patch("torch.distributed.isend", side_effect=capture_isend_tag):
            with patch("tensordict.TensorDict.isend", capture_td_isend):
                batch.isend(dst=1, tag=0)

        from nvalchemi.data.batch import _BatchRecvHandle

        meta = torch.tensor(
            [batch.num_graphs, batch.num_nodes, batch.num_edges],
            dtype=torch.int64,
            device="cpu",
        )
        mock_meta_handle = MagicMock()

        handle = _BatchRecvHandle(
            meta=meta,
            meta_handle=mock_meta_handle,
            src=0,
            device=torch.device("cpu"),
            template=batch,
            base_tag=0,
            group=None,
        )

        with patch("torch.distributed.irecv", side_effect=capture_irecv_tag):
            with patch("tensordict.TensorDict.irecv", capture_td_irecv):
                handle.wait()

        send_tags_after_meta = send_tags[1:]
        assert send_tags_after_meta == recv_tags, (
            f"Tag mismatch: send (after meta)={send_tags_after_meta}, recv={recv_tags}"
        )
