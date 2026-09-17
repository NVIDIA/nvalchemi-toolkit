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

import socket
from datetime import timedelta

import pytest
import torch
from tensordict import TensorDict
from torch import distributed as dist
from torch import multiprocessing as mp

from nvalchemi.data.atomic_data import AtomicData
from nvalchemi.data.batch import Batch, set_transient
from nvalchemi.data.level_storage import (
    LevelSchema,
    MultiLevelStorage,
    SegmentedLevelStorage,
    UniformLevelStorage,
)


def _minimal_atomic_data(
    num_nodes: int = 4,
    num_edges: int = 0,
    device: str | torch.device = "cpu",
) -> AtomicData:
    """Build minimal AtomicData for tests."""
    positions = torch.randn(num_nodes, 3, device=device)
    numbers = torch.ones(num_nodes, dtype=torch.long, device=device)
    kwargs: dict = {"positions": positions, "atomic_numbers": numbers}
    if num_edges > 0:
        neighbor_list = torch.zeros(num_edges, 2, dtype=torch.long, device=device)
        kwargs["neighbor_list"] = neighbor_list
    return AtomicData(**kwargs)


def _atomic_data_with_system(
    num_nodes: int = 2,
    device: str | torch.device = "cpu",
) -> AtomicData:
    """AtomicData with a system-level field so Batch has a 'system' group."""
    return AtomicData(
        positions=torch.randn(num_nodes, 3, device=device),
        atomic_numbers=torch.ones(num_nodes, dtype=torch.long, device=device),
        energy=torch.tensor([[0.0]], device=device),
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
        neighbor_list=torch.zeros(num_edges, 2, dtype=torch.long, device=device),
        energy=torch.tensor([[0.0]], device=device),
    )


def _fieldless_builtin_batch(
    atom_lengths: list[int],
    edge_lengths: list[int],
    *,
    pointer_capacity: int | None = None,
) -> Batch:
    """Build a metadata-only legacy batch for fieldless atoms/edges coverage."""
    schema = LevelSchema()
    groups = {
        "atoms": SegmentedLevelStorage(
            data=None,
            segment_lengths=atom_lengths,
            batch_ptr_capacity=pointer_capacity,
            device="cpu",
            attr_map=schema,
            validate=False,
        ),
        "edges": SegmentedLevelStorage(
            data=None,
            segment_lengths=edge_lengths,
            batch_ptr_capacity=pointer_capacity,
            device="cpu",
            attr_map=schema,
            validate=False,
        ),
    }
    return Batch._construct(
        device="cpu",
        keys={"node": set(), "edge": set(), "system": set()},
        storage=MultiLevelStorage(groups=groups, attr_map=schema, validate=False),
    )


def _custom_boundary_batch(
    segment_length: int,
    energy: float,
    *,
    payload_value: float | None = None,
) -> Batch:
    """Build a small batch with boundary-sized custom segment metadata."""
    schema = LevelSchema()
    schema.add_level("samples", segmented=True)
    sample_data = None
    if payload_value is not None:
        schema.set("sample_values", "samples", dtype=torch.float32)
        base = torch.tensor([[payload_value]], dtype=torch.float32)
        sample_data = {"sample_values": base.expand(segment_length, 1)}

    groups = {
        "system": UniformLevelStorage(
            data={"energy": torch.tensor([[energy]], dtype=torch.float32)},
            device="cpu",
            attr_map=schema,
            validate=True,
        ),
        "samples": SegmentedLevelStorage(
            data=sample_data,
            segment_lengths=[segment_length],
            device="cpu",
            attr_map=schema,
            validate=True,
        ),
    }
    return Batch._construct(
        device="cpu",
        keys={"node": set(), "edge": set(), "system": {"energy"}},
        storage=MultiLevelStorage(groups=groups, attr_map=schema, validate=True),
    )


def _builtin_edge_product_schema() -> LevelSchema:
    """Build the public fieldless-edges product schema."""
    schema = LevelSchema()
    schema.add_product_level("atom_edges", left="atoms", right="edges")
    schema.set("atom_edge_values", "atom_edges")
    return schema


def _builtin_edge_product_data(
    num_nodes: int, num_edges: int, offset: float
) -> AtomicData:
    """Build an AtomicData item with a fieldless built-in edge parent."""
    data = _minimal_atomic_data(num_nodes)
    data.atom_edge_values = (
        torch.arange(num_nodes * num_edges, dtype=torch.float32)
        .add_(offset)
        .reshape(num_nodes, num_edges, 1)
    )
    return data


def _builtin_edge_product_batch() -> Batch:
    """Build public atoms-times-fieldless-edges data with zero-edge graph."""
    return Batch.from_data_list(
        [
            _builtin_edge_product_data(2, 3, 10.0),
            _builtin_edge_product_data(3, 0, 20.0),
            _builtin_edge_product_data(1, 2, 30.0),
        ],
        attr_map=_builtin_edge_product_schema(),
    )


def _custom_buffer_schema(*, fieldless_parent: bool = False) -> LevelSchema:
    """Build a schema used by the custom buffer lifecycle tests."""
    schema = LevelSchema()
    schema.add_level("molecules", segmented=True)
    schema.add_product_level("pairs", left="atoms", right="molecules")
    if not fieldless_parent:
        schema.set("molecule_values", "molecules")
    schema.set("pair_values", "pairs")
    return schema


def _custom_buffer_data(
    num_nodes: int,
    num_molecules: int,
    *,
    fieldless_parent: bool = False,
) -> AtomicData:
    """Build one graph with custom segmented and product values."""
    data = _minimal_atomic_data(num_nodes)
    if not fieldless_parent:
        data.molecule_values = torch.arange(num_molecules, dtype=torch.float32).reshape(
            -1, 1
        )
    data.pair_values = (
        torch.arange(num_nodes * num_molecules, dtype=torch.float32)
        .add_(num_nodes * 10 + num_molecules)
        .reshape(num_nodes, num_molecules, 1)
    )
    return data


def _custom_uniform_schema() -> LevelSchema:
    """Build a schema with one custom uniform field."""
    schema = LevelSchema()
    schema.add_level("metadata", segmented=False)
    schema.set("metadata_values", "metadata")
    return schema


def _custom_transport_schema() -> LevelSchema:
    """Build the mixed custom schema used by transport tests."""
    schema = LevelSchema()
    schema.add_level("metadata", segmented=False)
    schema.add_level("molecules", segmented=True)
    schema.add_level("fieldless", segmented=True)
    schema.add_product_level("pairs", left="atoms", right="molecules")
    schema.add_product_level("fieldless_pairs", left="atoms", right="fieldless")
    schema.set("metadata_values", "metadata")
    schema.set("molecule_values", "molecules")
    schema.set("pair_values", "pairs")
    schema.set("fieldless_pair_values", "fieldless_pairs")
    return schema


def _custom_transport_batch() -> Batch:
    """Build a mixed custom batch with nonzero and zero custom segments."""
    schema = _custom_transport_schema()
    first = _minimal_atomic_data(2)
    first.metadata_values = torch.tensor([[1.0]])
    first.molecule_values = torch.tensor([[2.0], [3.0]])
    first.pair_values = torch.arange(4, dtype=torch.float32).reshape(2, 2, 1)
    first.fieldless_pair_values = torch.arange(6, dtype=torch.float32).reshape(2, 3, 1)

    second = _minimal_atomic_data(3)
    second.metadata_values = torch.tensor([[4.0]])
    second.molecule_values = torch.empty(0, 1)
    second.pair_values = torch.empty(3, 0, 1)
    second.fieldless_pair_values = torch.empty(3, 0, 1)
    return Batch.from_data_list([first, second], attr_map=schema)


def _custom_transport_gloo_worker(
    rank: int, world_size: int, port: int, sentinel: bool
) -> None:
    """Send or receive the custom transport fixture in a Gloo worker."""
    dist.init_process_group(
        backend="gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=30),
    )
    try:
        template = _custom_transport_batch()
        if rank == 0:
            outgoing = template
            if sentinel:
                outgoing = Batch.empty(
                    num_systems=0,
                    num_nodes=0,
                    num_edges=0,
                    template=template,
                    attr_map=template._storage.attr_map,
                    level_capacities={
                        "molecules": 0,
                        "fieldless": 0,
                        "pairs": 0,
                        "fieldless_pairs": 0,
                    },
                )
            outgoing.send(dst=1, tag=37)
        else:
            received = Batch.recv(src=0, device="cpu", template=template, tag=37)
            assert received._storage.attr_map is not template._storage.attr_map
            assert (
                received._storage.attr_map.level_names
                == template._storage.attr_map.level_names
            )
            assert list(received._storage.groups) == list(template._storage.groups)
            if sentinel:
                assert received.num_graphs == 0
                for name, group in template._storage.groups.items():
                    received_group = received._storage.groups[name]
                    assert type(received_group) is type(group)
                    assert list(received_group.keys()) == list(group.keys())
                    for key in group.keys():
                        assert received_group[key].dtype == group[key].dtype
                        assert received_group[key].shape[1:] == group[key].shape[1:]
                    if isinstance(received_group, SegmentedLevelStorage):
                        assert received_group.segment_lengths.numel() == 0
                assert received._storage.attr_map.product_parents[
                    "fieldless_pairs"
                ] == (
                    "atoms",
                    "fieldless",
                )
            else:
                assert received.level_keys == template.level_keys
                assert received.keys == template.keys
                torch.testing.assert_close(
                    received.metadata_values, template.metadata_values
                )
                torch.testing.assert_close(
                    received.molecule_values, template.molecule_values
                )
                torch.testing.assert_close(received.pair_values, template.pair_values)
                torch.testing.assert_close(
                    received.fieldless_pair_values, template.fieldless_pair_values
                )
                assert received._storage.groups[
                    "molecules"
                ].segment_lengths.tolist() == [2, 0]
                assert received._storage.groups[
                    "fieldless"
                ].segment_lengths.tolist() == [3, 0]
                for level in ("molecules", "fieldless", "pairs", "fieldless_pairs"):
                    torch.testing.assert_close(
                        received.level_ptr(level), template.level_ptr(level)
                    )
                first = received.get_data(0)
                second = received.get_data(1)
                assert first.fieldless_pair_values.shape == (2, 3, 1)
                assert second.fieldless_pair_values.shape == (3, 0, 1)
                torch.testing.assert_close(
                    first.fieldless_pair_values,
                    template.get_data(0).fieldless_pair_values,
                )
    finally:
        dist.destroy_process_group()


def _builtin_edge_product_gloo_worker(rank: int, world_size: int, port: int) -> None:
    """Send the public fieldless built-in edge parent fixture over Gloo."""
    dist.init_process_group(
        backend="gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=30),
    )
    try:
        template = _builtin_edge_product_batch()
        if rank == 0:
            template.send(dst=1, tag=43)
        else:
            received = Batch.recv(src=0, device="cpu", template=template, tag=43)
            assert "neighbor_list" not in received
            assert received.level_ptr("atoms").tolist() == [0, 2, 5, 6]
            assert received.level_ptr("edges").tolist() == [0, 3, 3, 5]
            assert received.level_ptr("atom_edges").tolist() == [0, 6, 6, 8]
            torch.testing.assert_close(
                received.atom_edge_values, template.atom_edge_values
            )
            for graph_idx in range(received.num_graphs):
                torch.testing.assert_close(
                    received.get_data(graph_idx).atom_edge_values,
                    template.get_data(graph_idx).atom_edge_values,
                )
    finally:
        dist.destroy_process_group()


def _available_tcp_port() -> int:
    """Reserve and return an available local TCP port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


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
        assert batch.batch_idx.shape == (4,)
        assert batch.batch_ptr.tolist() == [0, 4]
        assert batch.num_nodes_list == [4]
        # No edges group when input has no edge data, so num_edges_list is []
        assert batch.num_edges_list == []

    def test_from_data_list_multiple(self):
        d1 = _minimal_atomic_data(3)
        d2 = _minimal_atomic_data(5)
        batch = Batch.from_data_list([d1, d2])
        assert batch.num_graphs == 2
        assert batch.num_nodes == 8
        assert batch.batch_idx.shape == (8,)
        assert (batch.batch_idx[:3] == 0).all()
        assert (batch.batch_idx[3:8] == 1).all()
        assert batch.batch_ptr.tolist() == [0, 3, 8]
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

    def test_level_ptr_rejects_unknown_and_unresolved_levels(self):
        schema = LevelSchema()
        schema.add_level("samples", segmented=True)
        batch = Batch.from_data_list([_minimal_atomic_data(2)], attr_map=schema)

        with pytest.raises(KeyError, match="missing"):
            batch.level_ptr("missing")
        with pytest.raises(KeyError, match="samples"):
            batch.level_ptr("samples")
        assert "samples" not in batch.level_keys

    def test_level_ptr_resolves_fieldless_uniform_and_builtin_levels(self):
        schema = LevelSchema()
        schema.add_level("metadata", segmented=False)
        batch = Batch.from_data_list(
            [_minimal_atomic_data(2), _minimal_atomic_data(3)], attr_map=schema
        )

        assert batch.level_keys["metadata"] == set()
        assert batch.level_ptr("metadata").tolist() == [0, 1, 2]
        assert batch.level_keys["edges"] == set()
        assert batch.level_ptr("edges").tolist() == [0, 0, 0]

    def test_level_ptr_resolves_fieldless_builtin_atoms(self):
        system = UniformLevelStorage(
            data={"energy": torch.zeros(2, 1)}, device="cpu", validate=False
        )
        storage = MultiLevelStorage(
            groups={"system": system}, attr_map=None, validate=False
        )
        batch = Batch._construct(
            device=torch.device("cpu"), keys={"system": {"energy"}}, storage=storage
        )

        assert batch.level_keys["atoms"] == set()
        assert batch.level_ptr("atoms").tolist() == [0, 0, 0]

    def test_level_ptr_rejects_uniform_graph_count_overflow(self):
        system = UniformLevelStorage(
            data={"energy": torch.zeros(1, 1)}, device="cpu", validate=False
        )
        object.__setattr__(system, "_num_kept", torch.iinfo(torch.int32).max + 1)
        batch = Batch._construct(
            device="cpu",
            keys={"system": {"energy"}},
            storage=MultiLevelStorage(groups={"system": system}, validate=False),
        )

        with pytest.raises(OverflowError, match="Uniform level 'system'"):
            batch.level_ptr("system")

    def test_level_ptr_derives_fieldless_product_from_resolved_parents(self):
        schema = LevelSchema()
        schema.add_level("left_items", segmented=True)
        schema.add_level("right_items", segmented=True)
        schema.add_product_level("left_right", left="left_items", right="right_items")
        schema.set("left_values", "left_items")
        schema.set("right_values", "right_items")
        data_list = []
        for num_left, num_right in ((2, 3), (1, 2)):
            data = _minimal_atomic_data(2)
            data.left_values = torch.zeros(num_left, 1)
            data.right_values = torch.zeros(num_right, 1)
            data_list.append(data)

        batch = Batch.from_data_list(data_list, attr_map=schema)

        assert batch.level_keys["left_right"] == set()
        assert batch.level_ptr("left_right").tolist() == [0, 6, 8]

    @pytest.mark.parametrize(
        "node_counts",
        [[50_000], [40_000, 40_000]],
        ids=["per-graph", "cumulative"],
    )
    def test_level_ptr_rejects_int32_product_overflow(self, node_counts: list[int]):
        schema = LevelSchema()
        schema.add_product_level("pairs", left="atoms", right="atoms")
        batch = Batch.from_data_list(
            [_minimal_atomic_data(num_nodes) for num_nodes in node_counts],
            attr_map=schema,
        )

        with pytest.raises(OverflowError, match="int32 maximum"):
            batch.level_ptr("pairs")

    def test_custom_schema_fields_retain_first_sample_order_and_infer_dtype(self):
        schema = LevelSchema()
        schema.add_level("samples", segmented=True)
        schema.set("first_values", "samples")
        schema.set("second_values", "samples")

        first = _minimal_atomic_data(2)
        first.first_values = torch.tensor([[1.0], [2.0]], dtype=torch.float64)
        first.second_values = torch.tensor([[10.0], [20.0]], dtype=torch.float64)
        second = _minimal_atomic_data(1)
        second.second_values = torch.tensor([[30.0]], dtype=torch.float64)
        second.first_values = torch.tensor([[3.0]], dtype=torch.float64)

        batch = Batch.from_data_list([first, second], attr_map=schema)

        assert list(batch._storage.groups["samples"].keys()) == [
            "first_values",
            "second_values",
        ]
        assert batch._storage.attr_map.dtypes["first_values"] == "float64"
        assert batch._storage.attr_map.dtypes["second_values"] == "float64"
        assert "first_values" not in schema.dtypes
        assert "second_values" not in schema.dtypes
        assert batch.first_values[:, 0].tolist() == [1.0, 2.0, 3.0]
        assert batch.second_values[:, 0].tolist() == [10.0, 20.0, 30.0]

    def test_custom_later_only_field_is_rejected(self):
        schema = LevelSchema()
        schema.add_level("samples", segmented=True)
        schema.set("later_values", "samples")
        first = _minimal_atomic_data(2)
        second = _minimal_atomic_data(2)
        second.later_values = torch.ones(1, 1)

        with pytest.raises(ValueError, match="appears only in later sample 1"):
            Batch.from_data_list([first, second], attr_map=schema)

    def test_custom_missing_field_is_rejected(self):
        schema = LevelSchema()
        schema.add_level("samples", segmented=True)
        schema.set("sample_values", "samples")
        first = _minimal_atomic_data(2)
        first.sample_values = torch.ones(1, 1)
        second = _minimal_atomic_data(2)

        with pytest.raises(
            ValueError, match="Custom field 'sample_values'.*missing from sample 1"
        ):
            Batch.from_data_list([first, second], attr_map=schema)

    def test_batch_with_system_only_storage(self):
        """Batch built with only system group: batch, ptr, num_nodes_list, etc. hit None branches."""
        system = UniformLevelStorage(
            data={"energy": torch.randn(2, 1)},
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
            keys={"system": {"energy"}},
            storage=storage,
        )
        assert batch.num_graphs == 2
        assert batch.num_nodes == 0
        assert batch.num_edges == 0
        assert batch.batch_idx.shape == (0,)
        assert batch.batch_ptr.tolist() == [0]
        assert batch.num_nodes_list == []
        assert batch.num_edges_list == []
        assert batch.num_nodes_per_graph.shape == (0,)
        assert batch.num_edges_per_graph.shape == (0,)
        assert batch.max_num_nodes == 0
        assert batch.batch_size == 2

    def test_custom_levels_pack_in_schema_order_and_round_trip(self):
        schema = LevelSchema()
        schema.add_level("molecules", segmented=True)
        schema.add_product_level("atom_molecule", left="atoms", right="molecules")
        schema.set("molecule_features", "molecules")
        schema.set("pair_features", "atom_molecule")

        data_list = []
        for num_nodes, num_molecules in ((2, 3), (4, 1)):
            data = _minimal_atomic_data(num_nodes)
            data.molecule_features = torch.arange(num_molecules).reshape(-1, 1)
            data.pair_features = torch.arange(num_nodes * num_molecules).reshape(
                num_nodes, num_molecules, 1
            )
            data_list.append(data)

        batch = Batch.from_data_list(data_list, attr_map=schema)

        assert list(batch._storage.groups) == ["atoms", "molecules", "atom_molecule"]
        assert list(batch.level_keys) == [
            "atoms",
            "edges",
            "system",
            "molecules",
            "atom_molecule",
        ]
        assert batch.level_keys["molecules"] == {"molecule_features"}
        assert batch.level_ptr("molecules").tolist() == [0, 3, 4]
        assert batch.level_ptr("atom_molecule").tolist() == [0, 6, 10]

        output = batch.get_data(1)
        assert output.molecule_features.shape == (1, 1)
        assert output.pair_features.shape == (4, 1, 1)
        assert output._level_schema.level_kind("atom_molecule") == "product"
        cloned_output = output.clone()
        assert cloned_output._level_schema is not output._level_schema
        moved_output = output.to("cpu")
        assert moved_output._level_schema is not output._level_schema
        rebatch = Batch.from_data_list([output])
        assert rebatch.level_keys == batch.level_keys

        selected = batch[[1, 0]]
        assert selected.level_ptr("atom_molecule").tolist() == [0, 4, 10]
        assert selected.get_data(0).pair_features.shape == (4, 1, 1)

    def test_product_with_fieldless_parent_round_trips_logical_shape(self):
        schema = LevelSchema()
        schema.add_level("molecules", segmented=True)
        schema.add_product_level("atom_molecule", left="atoms", right="molecules")
        schema.set("pair_features", "atom_molecule")

        data_list = []
        for num_nodes, num_molecules in ((2, 3), (4, 1)):
            data = _minimal_atomic_data(num_nodes)
            data.pair_features = torch.zeros(num_nodes, num_molecules, 2)
            data_list.append(data)

        batch = Batch.from_data_list(data_list, attr_map=schema)

        assert batch.level_keys["molecules"] == set()
        assert batch.level_ptr("molecules").tolist() == [0, 3, 4]
        assert batch.pair_features.shape == (10, 2)
        assert batch.get_data(0).pair_features.shape == (2, 3, 2)
        assert batch.get_data(1).pair_features.shape == (4, 1, 2)
        assert Batch.from_data_list(batch.to_data_list()).level_keys == batch.level_keys

    def test_self_product_uses_equal_logical_axes(self):
        schema = LevelSchema()
        schema.add_product_level("atom_atom", left="atoms", right="atoms")
        schema.set("pair_features", "atom_atom")
        data_list = []
        for num_nodes in (2, 3):
            data = _minimal_atomic_data(num_nodes)
            data.pair_features = torch.zeros(num_nodes, num_nodes, 1)
            data_list.append(data)

        batch = Batch.from_data_list(data_list, attr_map=schema)

        assert batch.level_ptr("atom_atom").tolist() == [0, 4, 13]
        assert batch.get_data(0).pair_features.shape == (2, 2, 1)
        assert batch.get_data(1).pair_features.shape == (3, 3, 1)

    def test_custom_levels_complete_three_system_lifecycle(self):
        schema = LevelSchema()
        schema.add_level("empty", segmented=True)
        schema.add_level("left_items", segmented=True)
        schema.add_level("right_items", segmented=True)
        schema.add_product_level("atom_atom", left="atoms", right="atoms")
        schema.add_product_level("left_right", left="left_items", right="right_items")
        schema.add_level("A", segmented=True)
        schema.add_product_level("A_A", left="A", right="A")
        for field, level in (
            ("empty_values", "empty"),
            ("left_values", "left_items"),
            ("right_values", "right_items"),
            ("atom_pairs", "atom_atom"),
            ("cross_values", "left_right"),
            ("A_values", "A"),
            ("A_pairs", "A_A"),
        ):
            schema.set(field, level)

        atom_counts = [2, 4, 1]
        left_counts = [2, 4, 1]
        right_counts = [3, 1, 2]
        data_list = []
        for index, (num_atoms, num_left, num_right) in enumerate(
            zip(atom_counts, left_counts, right_counts, strict=True)
        ):
            data = _minimal_atomic_data(num_atoms)
            a_count = num_atoms + 1
            data.empty_values = torch.empty(0, 2)
            data.left_values = torch.arange(index * 10, index * 10 + num_left).reshape(
                -1, 1
            )
            data.right_values = torch.arange(
                index * 10, index * 10 + num_right
            ).reshape(-1, 1)
            data.atom_pairs = torch.arange(num_atoms * num_atoms).reshape(
                num_atoms, num_atoms, 1
            )
            data.cross_values = torch.arange(num_left * num_right).reshape(
                num_left, num_right, 1
            )
            data.A_values = torch.arange(a_count).reshape(-1, 1)
            data.A_pairs = torch.arange(a_count * a_count).reshape(a_count, a_count, 1)
            data_list.append(data)

        batch = Batch.from_data_list(data_list, attr_map=schema)
        expected_ptrs = {
            "atoms": [0, 2, 6, 7],
            "empty": [0, 0, 0, 0],
            "left_items": [0, 2, 6, 7],
            "right_items": [0, 3, 4, 6],
            "atom_atom": [0, 4, 20, 21],
            "left_right": [0, 6, 10, 12],
            "A": [0, 3, 8, 10],
            "A_A": [0, 9, 34, 38],
        }
        for level, expected in expected_ptrs.items():
            assert batch.level_ptr(level).tolist() == expected
        assert batch.level_ptr("system").tolist() == [0, 1, 2, 3]
        assert batch.level_keys["empty"] == {"empty_values"}
        assert batch.empty_values.shape == (0, 2)
        assert batch.left_values.shape == (7, 1)
        assert batch.left_values[:, 0].tolist() == [0, 1, 10, 11, 12, 13, 20]
        assert batch.right_values.shape == (6, 1)
        assert batch.atom_pairs.shape == (21, 1)
        assert batch.cross_values.shape == (12, 1)
        assert batch.A_pairs.shape == (38, 1)

        expected_shapes = [
            ((2, 2, 1), (2, 3, 1), (3, 3, 1)),
            ((4, 4, 1), (4, 1, 1), (5, 5, 1)),
            ((1, 1, 1), (1, 2, 1), (2, 2, 1)),
        ]
        for index, (atom_pair_shape, cross_shape, a_pair_shape) in enumerate(
            expected_shapes
        ):
            data = batch.get_data(index)
            assert data.empty_values.shape == (0, 2)
            assert data.atom_pairs.shape == atom_pair_shape
            assert data.cross_values.shape == cross_shape
            assert data.A_pairs.shape == a_pair_shape

        rebatch = Batch.from_data_list(batch.to_data_list())
        assert rebatch.level_keys == batch.level_keys
        for level, expected in expected_ptrs.items():
            assert rebatch.level_ptr(level).tolist() == expected

        selected = batch.index_select([2, 0, 2])
        selected_ptrs = {
            "atoms": [0, 1, 3, 4],
            "empty": [0, 0, 0, 0],
            "left_items": [0, 1, 3, 4],
            "right_items": [0, 2, 5, 7],
            "atom_atom": [0, 1, 5, 6],
            "left_right": [0, 2, 8, 10],
            "A": [0, 2, 5, 7],
            "A_A": [0, 4, 13, 17],
        }
        for level, expected in selected_ptrs.items():
            assert selected.level_ptr(level).tolist() == expected
        assert selected.get_data(0).atom_pairs.shape == (1, 1, 1)
        assert selected.get_data(1).cross_values.shape == (2, 3, 1)
        assert selected.get_data(2).A_pairs.shape == (2, 2, 1)

        cloned = batch.clone()
        assert cloned._storage.attr_map is not batch._storage.attr_map
        cloned.add_key(
            "clone_values",
            [
                torch.zeros(2, 1),
                torch.zeros(4, 1),
                torch.zeros(1, 1),
            ],
            level="left_items",
        )
        assert "clone_values" in cloned.level_keys["left_items"]
        assert "clone_values" not in batch.level_keys["left_items"]

        cpu = batch.cpu()
        assert cpu.device == torch.device("cpu")
        assert cpu.contiguous() is cpu
        assert all(value.is_contiguous() for _, value in cpu)

    def test_product_rank_and_cardinality_rejections(self):
        schema = LevelSchema()
        schema.add_level("molecules", segmented=True)
        schema.add_product_level("atom_molecule", left="atoms", right="molecules")
        schema.set("pair_features", "atom_molecule")

        rank_error = _minimal_atomic_data(2)
        rank_error.pair_features = torch.zeros(6)
        with pytest.raises(ValueError, match="rank >= 2"):
            Batch.from_data_list([rank_error], attr_map=schema)

        cardinality_error = _minimal_atomic_data(2)
        cardinality_error.pair_features = torch.zeros(3, 2, 1)
        with pytest.raises(ValueError, match="left axis cardinalities"):
            Batch.from_data_list([cardinality_error], attr_map=schema)

    def test_product_rejects_inconsistent_axes_and_payload_shapes(self):
        schema = LevelSchema()
        schema.add_level("molecules", segmented=True)
        schema.add_product_level("atom_molecule", left="atoms", right="molecules")
        schema.set("pair_features", "atom_molecule")

        first = _minimal_atomic_data(2)
        second = _minimal_atomic_data(2)
        first.pair_features = torch.zeros(2, 3, 1)
        second.pair_features = torch.zeros(2, 3, 2)
        with pytest.raises(ValueError, match="trailing shape"):
            Batch.from_data_list([first, second], attr_map=schema)

    def test_custom_cardinality_mismatch_is_rejected(self):
        schema = LevelSchema()
        schema.add_level("samples", segmented=True)
        schema.set("sample_features", "samples")
        schema.set("other_features", "samples")
        first = _minimal_atomic_data(2)
        second = _minimal_atomic_data(3)
        first.sample_features = torch.zeros(2, 1)
        second.sample_features = torch.zeros(3, 1)
        first.other_features = torch.zeros(2, 1)
        second.other_features = torch.zeros(4, 1)

        with pytest.raises(ValueError, match="cardinalities"):
            Batch.from_data_list([first, second], attr_map=schema)

    def test_custom_dtype_mismatch_is_rejected_without_schema_mutation(self):
        schema = LevelSchema()
        schema.add_level("samples", segmented=True)
        schema.set("sample_values", "samples")
        before = schema.clone()
        first = _minimal_atomic_data(2)
        first.sample_values = torch.ones(1, 1, dtype=torch.float32)
        second = _minimal_atomic_data(2)
        second.sample_values = torch.ones(1, 1, dtype=torch.float64)

        with pytest.raises(ValueError, match="incompatible dtypes"):
            Batch.from_data_list([first, second], attr_map=schema)

        assert schema.dtypes == before.dtypes
        assert schema.attr_to_group == before.attr_to_group

    def test_custom_declared_dtype_must_match_tensor_dtype(self):
        schema = LevelSchema()
        schema.add_level("samples", segmented=True)
        schema.set("sample_values", "samples", dtype="float64")
        before = schema.clone()
        data = _minimal_atomic_data(2)
        data.sample_values = torch.ones(1, 1, dtype=torch.float32)

        with pytest.raises(ValueError, match="expected declared dtype float64"):
            Batch.from_data_list([data], attr_map=schema)

        assert schema.dtypes == before.dtypes
        assert schema.attr_to_group == before.attr_to_group

    def test_edge_cardinality_inference_stays_local_to_batch(self):
        data = _minimal_atomic_data(2)
        data.add_edge_property("edge_weights", torch.zeros(3, 1))

        assert data.num_edges == 0
        batch = Batch.from_data_list([data])

        assert batch.num_edges_list == [3]
        assert batch.get_data(0).edge_weights.shape == (3, 1)
        assert batch.get_data(0).num_edges == 0


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

    def test_batch_zero_preserves_capacity_after_smaller_put(self):
        """A reused buffer accepts a larger payload after a smaller one."""
        template = _minimal_atomic_data(2)
        buffer = Batch.empty(
            num_systems=4,
            num_nodes=16,
            num_edges=0,
            template=template,
        )
        small = Batch.from_data_list([_minimal_atomic_data(2)])
        larger = Batch.from_data_list(
            [_minimal_atomic_data(2), _minimal_atomic_data(2)]
        )

        buffer.put(small, torch.ones(1, dtype=torch.bool))
        assert buffer.num_graphs == 1

        buffer.zero()
        buffer.put(larger, torch.ones(2, dtype=torch.bool))

        assert buffer.num_graphs == 2
        assert buffer.num_nodes_list == [2, 2]


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

    @pytest.mark.parametrize("idx", [-3, -4, -5, -100, 2, 3, 100])
    def test_get_data_out_of_range_raises(self, idx):
        d1 = _minimal_atomic_data(2)
        d2 = _minimal_atomic_data(3)
        batch = Batch.from_data_list([d1, d2])
        with pytest.raises(IndexError):
            batch.get_data(idx)

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

    def test_index_select_slice(self, device):
        batch = Batch.from_data_list(
            [
                _minimal_atomic_data(2),
                _minimal_atomic_data(3),
                _minimal_atomic_data(4),
            ],
            device=device,
        )
        sub = batch[1:3]
        assert isinstance(sub, Batch)
        assert sub.num_graphs == 2
        assert sub.num_nodes_list == [3, 4]
        assert sub.num_nodes == 7
        assert sub.level_ptr("atoms").dtype == torch.int32

    def test_index_select_int(self, device):
        batch = Batch.from_data_list(
            [
                _minimal_atomic_data(2),
                _minimal_atomic_data(3),
            ],
            device=device,
        )
        one = batch[1]
        assert isinstance(one, AtomicData)
        assert one.num_nodes == 3

    def test_index_select_tensor(self, device):
        batch = Batch.from_data_list(
            [
                _minimal_atomic_data(2),
                _minimal_atomic_data(3),
                _minimal_atomic_data(4),
            ],
            device=device,
        )
        sub = batch[torch.tensor([0, 2], device=device)]
        assert sub.num_graphs == 2
        assert sub.num_nodes_list == [2, 4]

    def test_index_select_list(self, device):
        batch = Batch.from_data_list(
            [
                _minimal_atomic_data(2),
                _minimal_atomic_data(3),
            ],
            device=device,
        )
        sub = batch[[1, 0]]
        assert sub.num_graphs == 2
        assert sub.num_nodes_list == [3, 2]

    def test_index_select_with_edges_applies_edge_index_correction(self):
        """index_select on a batch with edges corrects neighbor_list offsets."""
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
        assert d0.neighbor_list is not None and d1.neighbor_list is not None

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

    def test_append_select_and_rebatch_fieldless_builtin_segment_metadata(self):
        left = _fieldless_builtin_batch([2, 1], [3, 0], pointer_capacity=5)
        right = _fieldless_builtin_batch([4], [2])

        left.append(right)

        assert left.level_ptr("atoms").tolist() == [0, 2, 3, 7]
        assert left.level_ptr("edges").tolist() == [0, 3, 3, 5]
        assert left._storage.groups["atoms"].batch_ptr.shape[0] == 5
        selected = left.index_select([1, 2])
        assert selected.level_ptr("atoms").tolist() == [0, 1, 5]
        assert selected.level_ptr("edges").tolist() == [0, 0, 2]

        selected.append(selected.clone())
        assert selected.level_ptr("atoms").tolist() == [0, 1, 5, 6, 10]
        assert selected.level_ptr("edges").tolist() == [0, 0, 2, 2, 4]

    def test_fieldless_builtin_edge_product_lifecycle(self):
        """Atoms-times-edges products retain fieldless edge cardinalities."""
        original = _builtin_edge_product_batch()
        left = Batch.from_data_list(
            original.to_data_list()[:2], attr_map=original._storage.attr_map
        )
        right = Batch.from_data_list(
            original.to_data_list()[2:], attr_map=original._storage.attr_map
        )

        left.append(right)

        assert "neighbor_list" not in left
        assert left.level_ptr("atoms").tolist() == [0, 2, 5, 6]
        assert left.level_ptr("edges").tolist() == [0, 3, 3, 5]
        assert left.level_ptr("atom_edges").tolist() == [0, 6, 6, 8]
        for graph_idx in range(left.num_graphs):
            torch.testing.assert_close(
                left.get_data(graph_idx).atom_edge_values,
                original.get_data(graph_idx).atom_edge_values,
            )

        selected = left.index_select([2, 0])
        assert selected.level_ptr("edges").tolist() == [0, 2, 5]
        for graph_idx, source_idx in enumerate((2, 0)):
            torch.testing.assert_close(
                selected.get_data(graph_idx).atom_edge_values,
                original.get_data(source_idx).atom_edge_values,
            )

        repeated = selected.clone()
        repeated.append(selected)
        assert repeated.level_ptr("edges").tolist() == [0, 2, 5, 7, 10]
        rebatch = Batch.from_data_list(
            repeated.to_data_list(), attr_map=repeated._storage.attr_map
        )
        assert rebatch.level_ptr("atom_edges").tolist() == [0, 2, 8, 10, 16]
        for graph_idx, source_idx in enumerate((2, 0, 2, 0)):
            torch.testing.assert_close(
                rebatch.get_data(graph_idx).atom_edge_values,
                original.get_data(source_idx).atom_edge_values,
            )

    def test_append_fieldless_metadata_prevalidates_pointer_overflow(self):
        maximum = torch.iinfo(torch.int32).max
        left = _fieldless_builtin_batch([maximum], [0], pointer_capacity=4)
        right = _fieldless_builtin_batch([1], [0])

        with pytest.raises(OverflowError, match="Segment pointer exceeds"):
            left.append(right)

        assert left.level_ptr("atoms").tolist() == [0, maximum]
        assert left.level_ptr("edges").tolist() == [0, 0]

    def test_append_payload_overflow_is_atomic_across_levels(self):
        maximum = torch.iinfo(torch.int32).max
        left = _custom_boundary_batch(maximum, 11.0, payload_value=7.0)
        right = _custom_boundary_batch(1, 22.0, payload_value=9.0)

        with pytest.raises(OverflowError, match="Segment pointer exceeds"):
            left.append(right)

        assert left.num_graphs == 1
        assert left.energy.tolist() == [[11.0]]
        assert left.level_ptr("samples").tolist() == [0, maximum]
        assert left.sample_values.shape == (maximum, 1)
        assert left.sample_values[0].item() == 7.0
        assert right.num_graphs == 1
        assert right.energy.tolist() == [[22.0]]
        assert right.level_ptr("samples").tolist() == [0, 1]
        assert right.sample_values.tolist() == [[9.0]]

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
            "virial",
            [torch.randn(1, 3, 3), torch.randn(1, 3, 3)],
            level="system",
        )
        assert "virial" in batch
        assert batch["virial"].shape == (2, 3, 3)

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
        batch.add_key("virial", [torch.zeros(1, 3, 3)], level="system")
        batch.add_key("virial", [torch.ones(1, 3, 3)], level="system", overwrite=True)
        assert batch["virial"].eq(1).all()

    def test_add_key_exists_raises(self):
        batch = Batch.from_data_list([_atomic_data_with_system(2)])
        batch.add_key("virial", [torch.zeros(1, 3, 3)], level="system")
        with pytest.raises(ValueError, match="already exists"):
            batch.add_key(
                "virial", [torch.ones(1, 3, 3)], level="system", overwrite=False
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

    def test_add_key_registered_custom_level(self):
        schema = LevelSchema()
        schema.add_level("samples", segmented=True)
        batch = Batch.from_data_list(
            [_minimal_atomic_data(2), _minimal_atomic_data(3)], attr_map=schema
        )
        schema.set("sample_features", "samples")

        batch.add_key(
            "sample_features",
            [torch.zeros(4, 2), torch.ones(1, 2)],
            level="samples",
        )

        assert batch.level_keys["samples"] == {"sample_features"}
        assert batch.level_ptr("samples").tolist() == [0, 4, 5]
        assert batch.sample_features.shape == (5, 2)

    def test_add_key_registered_product_level_uses_logical_axes(self):
        schema = LevelSchema()
        schema.add_level("samples", segmented=True)
        schema.add_product_level("atom_sample", left="atoms", right="samples")
        batch = Batch.from_data_list(
            [_minimal_atomic_data(2), _minimal_atomic_data(3)], attr_map=schema
        )

        batch.add_key(
            "pair_features",
            [torch.zeros(2, 4, 1), torch.ones(3, 1, 1)],
            level="atom_sample",
        )

        assert batch.level_keys["samples"] == set()
        assert batch.level_ptr("samples").tolist() == [0, 4, 5]
        assert batch.level_ptr("atom_sample").tolist() == [0, 8, 11]
        assert batch.get_data(0).pair_features.shape == (2, 4, 1)
        assert batch.get_data(1).pair_features.shape == (3, 1, 1)

    def test_add_key_product_rejects_insufficient_rank(self):
        schema = LevelSchema()
        schema.add_product_level("atom_atom", left="atoms", right="atoms")
        batch = Batch.from_data_list([_minimal_atomic_data(2)], attr_map=schema)

        with pytest.raises(ValueError, match="rank >= 2"):
            batch.add_key("pair_features", [torch.zeros(4)], level="atom_atom")

    def test_add_key_self_product_rejects_unequal_axes(self):
        schema = LevelSchema()
        schema.add_product_level("atom_atom", left="atoms", right="atoms")
        batch = Batch.from_data_list([_minimal_atomic_data(2)], attr_map=schema)

        with pytest.raises(ValueError, match="requires equal left and right"):
            batch.add_key("pair_features", [torch.zeros(2, 3, 1)], level="atom_atom")

    def test_add_key_product_rejects_parent_cardinality_mismatch(self):
        schema = LevelSchema()
        schema.add_level("samples", segmented=True)
        schema.add_product_level("atom_sample", left="atoms", right="samples")
        batch = Batch.from_data_list([_minimal_atomic_data(2)], attr_map=schema)

        with pytest.raises(ValueError, match="do not match parent 'atoms'"):
            batch.add_key("pair_features", [torch.zeros(3, 4, 1)], level="atom_sample")

    def test_add_key_product_rejects_incompatible_payload_shapes(self):
        schema = LevelSchema()
        schema.add_product_level("atom_atom", left="atoms", right="atoms")
        batch = Batch.from_data_list(
            [_minimal_atomic_data(2), _minimal_atomic_data(3)], attr_map=schema
        )

        with pytest.raises(ValueError, match="incompatible trailing shapes"):
            batch.add_key(
                "pair_features",
                [torch.zeros(2, 2, 1), torch.zeros(3, 3, 2)],
                level="atom_atom",
            )

    def test_add_key_custom_dtype_validation_precedes_storage_mutation(self):
        schema = LevelSchema()
        schema.add_level("samples", segmented=True)
        batch = Batch.from_data_list(
            [_minimal_atomic_data(2), _minimal_atomic_data(2)], attr_map=schema
        )
        before_groups = tuple(batch._storage.groups)
        before_keys = {
            name: set(group.keys()) for name, group in batch._storage.groups.items()
        }

        with pytest.raises(ValueError, match="incompatible dtypes"):
            batch.add_key(
                "sample_values",
                [
                    torch.ones(1, 1, dtype=torch.float32),
                    torch.ones(1, 1, dtype=torch.float64),
                ],
                level="samples",
            )

        assert tuple(batch._storage.groups) == before_groups
        assert {
            name: set(group.keys()) for name, group in batch._storage.groups.items()
        } == before_keys

    def test_add_key_respects_declared_custom_dtype_before_storage_mutation(self):
        schema = LevelSchema()
        schema.add_level("samples", segmented=True)
        schema.set("sample_values", "samples", dtype="float64")
        batch = Batch.from_data_list(
            [_minimal_atomic_data(2), _minimal_atomic_data(2)], attr_map=schema
        )
        before_groups = tuple(batch._storage.groups)
        before_keys = {
            name: set(group.keys()) for name, group in batch._storage.groups.items()
        }
        before_dtypes = batch._storage.attr_map.dtypes.copy()

        with pytest.raises(ValueError, match="expected declared dtype float64"):
            batch.add_key(
                "sample_values",
                [
                    torch.ones(1, 1, dtype=torch.float32),
                    torch.ones(1, 1, dtype=torch.float32),
                ],
                level="samples",
            )

        assert tuple(batch._storage.groups) == before_groups
        assert {
            name: set(group.keys()) for name, group in batch._storage.groups.items()
        } == before_keys
        assert batch._storage.attr_map.dtypes == before_dtypes

    def test_add_key_custom_overwrite_and_unknown_level_fallback(self):
        schema = LevelSchema()
        schema.add_level("samples", segmented=True)
        batch = Batch.from_data_list(
            [_minimal_atomic_data(2), _minimal_atomic_data(3)], attr_map=schema
        )

        batch.add_key(
            "sample_values",
            [torch.zeros(2, 1), torch.ones(3, 1)],
            level="samples",
        )
        batch.add_key(
            "sample_values",
            [
                torch.full((2, 1), 2.0, dtype=torch.float32),
                torch.full((3, 1), 3.0, dtype=torch.float32),
            ],
            level="samples",
            overwrite=True,
        )
        assert batch.sample_values[:, 0].tolist() == [2, 2, 3, 3, 3]
        assert batch.level_ptr("samples").tolist() == [0, 2, 5]

        batch.add_key(
            "legacy_values",
            [torch.zeros(2, 1), torch.ones(3, 1)],
            level="unregistered-level",
        )
        assert "legacy_values" in batch.level_keys["atoms"]
        assert batch.legacy_values.shape == (5, 1)

    def test_append_custom_product_with_fieldless_parent(self):
        schema = LevelSchema()
        schema.add_level("samples", segmented=True)
        schema.add_product_level("atom_sample", left="atoms", right="samples")
        schema.set("pair_values", "atom_sample")

        def make_batch(num_atoms: int, num_samples: int, value: int) -> Batch:
            data = _minimal_atomic_data(num_atoms)
            data.pair_values = torch.full((num_atoms, num_samples, 1), value)
            return Batch.from_data_list([data], attr_map=schema)

        left = make_batch(2, 3, 1)
        right = make_batch(1, 2, 2)
        right_samples_ptr = right.level_ptr("samples").clone()

        left.append(right)

        assert left.level_keys["samples"] == set()
        assert left.level_ptr("samples").tolist() == [0, 3, 5]
        assert left.level_ptr("atom_sample").tolist() == [0, 6, 8]
        assert left.get_data(1).pair_values.shape == (1, 2, 1)
        assert right.level_ptr("samples").tolist() == right_samples_ptr.tolist()

    @pytest.mark.parametrize(
        "other_parents,other_counts",
        [
            (("right", "left"), {"left": 3, "right": 2, "other": 4}),
            (("left", "other"), {"left": 2, "right": 4, "other": 3}),
        ],
        ids=["reversed-parents", "different-parents"],
    )
    def test_append_rejects_same_product_name_with_different_parents(
        self, other_parents, other_counts
    ):
        def make_schema(parents: tuple[str, str]) -> LevelSchema:
            schema = LevelSchema()
            for name in ("left", "right", "other"):
                schema.add_level(name, segmented=True)
            schema.add_product_level("pairs", left=parents[0], right=parents[1])
            for field, level in (
                ("left_values", "left"),
                ("right_values", "right"),
                ("other_values", "other"),
                ("pair_values", "pairs"),
            ):
                schema.set(field, level)
            return schema

        def make_batch(schema: LevelSchema, counts: dict[str, int]) -> Batch:
            data = _minimal_atomic_data(2)
            data.left_values = torch.zeros(counts["left"], 1)
            data.right_values = torch.zeros(counts["right"], 1)
            data.other_values = torch.zeros(counts["other"], 1)
            data.pair_values = torch.zeros(2, 3, 1)
            return Batch.from_data_list([data], attr_map=schema)

        receiver = make_batch(
            make_schema(("left", "right")), {"left": 2, "right": 3, "other": 4}
        )
        other = make_batch(make_schema(other_parents), other_counts)
        receiver_before = receiver.pair_values.clone()
        other_before = other.pair_values.clone()
        receiver_ptr_before = receiver.level_ptr("pairs").clone()

        with pytest.raises(ValueError, match="definitions"):
            receiver.append(other)

        torch.testing.assert_close(receiver.pair_values, receiver_before)
        torch.testing.assert_close(other.pair_values, other_before)
        assert receiver.level_ptr("pairs").tolist() == receiver_ptr_before.tolist()

    def test_append_rejects_incompatible_custom_fields_before_mutation(self):
        schema = LevelSchema()
        schema.add_level("samples", segmented=True)
        schema.set("sample_features", "samples")
        first = _minimal_atomic_data(2)
        first.sample_features = torch.zeros(2, 1)
        second = _minimal_atomic_data(2)
        second.sample_features = torch.zeros(3, 1)
        left = Batch.from_data_list([first], attr_map=schema)

        other_schema = schema.clone()
        other_schema.set("other_features", "samples")
        second.other_features = torch.zeros(3, 1)
        right = Batch.from_data_list([second], attr_map=other_schema)
        left_before = left.sample_features.clone()
        right_before = right.other_features.clone()

        with pytest.raises(ValueError, match="field sets"):
            left.append(right)
        assert left.num_graphs == 1
        torch.testing.assert_close(left.sample_features, left_before)
        torch.testing.assert_close(right.other_features, right_before)

    def test_append_rejects_custom_trailing_shape_before_mutation(self):
        schema = LevelSchema()
        schema.add_level("samples", segmented=True)
        schema.set("sample_features", "samples")
        left_data = _minimal_atomic_data(2)
        left_data.sample_features = torch.zeros(2, 1)
        right_data = _minimal_atomic_data(3)
        right_data.sample_features = torch.zeros(3, 2)
        left = Batch.from_data_list([left_data], attr_map=schema)
        right = Batch.from_data_list([right_data], attr_map=schema)
        left_before = left.sample_features.clone()
        right_before = right.sample_features.clone()

        with pytest.raises(ValueError, match="trailing shapes"):
            left.append(right)
        torch.testing.assert_close(left.sample_features, left_before)
        torch.testing.assert_close(right.sample_features, right_before)

    def test_append_accepts_equivalent_custom_dtype_aliases(self):
        left_schema = _custom_uniform_schema()
        left_schema.set("metadata_values", "metadata", dtype="float")
        right_schema = _custom_uniform_schema()
        right_schema.set("metadata_values", "metadata", dtype="float32")

        left_data = _minimal_atomic_data(2)
        left_data.metadata_values = torch.tensor([[1.0]], dtype=torch.float32)
        right_data = _minimal_atomic_data(2)
        right_data.metadata_values = torch.tensor([[2.0]], dtype=torch.float32)
        left = Batch.from_data_list([left_data], attr_map=left_schema)
        right = Batch.from_data_list([right_data], attr_map=right_schema)

        left.append(right)

        assert left.num_graphs == 2
        assert left.metadata_values[:, 0].tolist() == [1.0, 2.0]

    def test_append_preserves_other_edge_index(self):
        """append() must not mutate the other batch's neighbor_list."""
        b1 = Batch.from_data_list(
            [_atomic_data_with_edges_and_system(num_nodes=2, num_edges=3)]
        )
        b2 = Batch.from_data_list(
            [_atomic_data_with_edges_and_system(num_nodes=3, num_edges=2)]
        )
        ei_before = b2["neighbor_list"].clone()
        b1.append(b2)
        assert torch.equal(b2["neighbor_list"], ei_before), (
            "append() mutated other batch's neighbor_list"
        )
        # Verify result is structurally correct.
        assert b1.num_graphs == 2
        assert b1.num_nodes_list == [2, 3]
        assert b1.num_edges_list == [3, 2]

    def test_append_self_raises(self):
        """Appending a batch to itself must raise ValueError."""
        batch = Batch.from_data_list(
            [_atomic_data_with_edges_and_system(num_nodes=2, num_edges=2)]
        )
        with pytest.raises(ValueError, match="shares storage"):
            batch.append(batch)

    def test_append_shared_storage_raises(self):
        """Appending a batch that shares the same storage must raise ValueError."""
        batch = Batch.from_data_list(
            [_atomic_data_with_edges_and_system(num_nodes=2, num_edges=2)]
        )
        alias = Batch(device=batch.device, storage=batch._storage)
        with pytest.raises(ValueError, match="shares storage"):
            batch.append(alias)


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

    def test_clone_preserves_custom_keys(self) -> None:
        """AtomicData.clone() must preserve dynamically-added key sets."""
        data = AtomicData(
            positions=torch.randn(3, 3),
            atomic_numbers=torch.tensor([6, 6, 6]),
        )
        data.add_system_property("system_id", torch.tensor([[0]], dtype=torch.long))

        cloned = data.clone()

        # Key set metadata is preserved
        assert "system_id" in cloned.__system_keys__
        # Value is preserved and independent
        assert torch.equal(cloned.system_id, data.system_id)
        assert cloned.system_id is not data.system_id
        # Key sets are independent copies (mutating one doesn't affect the other)
        assert cloned.__system_keys__ is not data.__system_keys__

    def test_model_copy_preserves_custom_keys(self) -> None:
        """AtomicData.model_copy(deep=True) must preserve dynamically-added key sets."""
        data = AtomicData(
            positions=torch.randn(3, 3),
            atomic_numbers=torch.tensor([6, 6, 6]),
        )
        data.add_system_property("system_id", torch.tensor([[0]], dtype=torch.long))

        copied = data.model_copy(deep=True)

        # Key set metadata is preserved
        assert "system_id" in copied.__system_keys__
        # Value is preserved
        assert torch.equal(copied.system_id, data.system_id)

    def test_batch_clone_preserves_custom_keys(self) -> None:
        """Batch.clone() must preserve dynamically-added keys."""
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
        cloned = batch.clone()

        assert hasattr(cloned, "system_id")
        assert torch.equal(cloned.system_id, batch.system_id)
        assert cloned.system_id is not batch.system_id


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
        assert "batch_idx" in d
        assert "batch_ptr" in d
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

    def test_put_repeatedly_into_empty_buffer(self):
        """A Batch.empty buffer with remaining capacity accepts repeated puts."""
        template = _minimal_atomic_data(1)
        buffer = Batch.empty(
            num_systems=4, num_nodes=50, num_edges=0, template=template
        )
        for expected in range(1, 4):
            src = Batch.from_data_list([_minimal_atomic_data(2)])
            mask = torch.ones(1, dtype=torch.bool)
            copied_mask = torch.zeros(1, dtype=torch.bool)
            buffer.put(src, mask, copied_mask=copied_mask)
            assert copied_mask.all()
            assert buffer.num_graphs == expected

    def test_put_zero_put_reuses_buffer(self):
        """zero() restores a put-filled buffer to its freshly-allocated state."""
        template = _minimal_atomic_data(1)
        buffer = Batch.empty(
            num_systems=4, num_nodes=50, num_edges=0, template=template
        )
        src = Batch.from_data_list(
            [_minimal_atomic_data(2), _minimal_atomic_data(3)],
        )
        mask = torch.ones(2, dtype=torch.bool)
        buffer.put(src, mask)
        assert buffer.num_graphs == 2
        buffer.zero()
        assert buffer.num_graphs == 0
        buffer.put(src, mask)
        assert buffer.num_graphs == 2
        assert buffer.num_nodes_list == [2, 3]
        torch.testing.assert_close(buffer.positions[:5], src.positions[:5])

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
        assert trimmed.batch_idx.shape[0] == trimmed.num_nodes

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

    def test_empty_allocates_custom_segmented_and_product_capacities(self):
        schema = _custom_buffer_schema()
        template = _custom_buffer_data(2, 3)
        source = Batch.from_data_list([template], attr_map=schema)

        buffer = Batch.empty(
            num_systems=4,
            num_nodes=20,
            num_edges=0,
            template=source,
            level_capacities={"molecules": 10, "pairs": 20},
        )

        assert buffer._storage.attr_map is not source._storage.attr_map
        assert buffer._storage.groups["molecules"]._data.shape[0] == 10
        assert buffer._storage.groups["pairs"]._data.shape[0] == 20
        assert buffer._storage.groups["molecules"].batch_ptr.shape[0] == 6

    def test_empty_allocates_custom_uniform_from_num_systems(self):
        schema = _custom_uniform_schema()
        template = _minimal_atomic_data(2)
        template.metadata_values = torch.tensor([[3.0]])
        source = Batch.from_data_list([template], attr_map=schema)

        buffer = Batch.empty(
            num_systems=5,
            num_nodes=10,
            num_edges=0,
            template=source,
        )

        group = buffer._storage.groups["metadata"]
        assert isinstance(group, UniformLevelStorage)
        assert group._data["metadata_values"].shape == (5, 1)
        assert len(group) == 0

    def test_empty_atomic_data_template_preserves_custom_float64_uniform_payload(self):
        schema = _custom_uniform_schema()
        schema.set("metadata_values", "metadata", dtype=torch.float64)
        template = AtomicData(
            positions=torch.zeros(2, 3, dtype=torch.float64),
            atomic_numbers=torch.tensor([1, 8], dtype=torch.int64),
        )
        template.metadata_values = torch.tensor([[3.5]], dtype=torch.float64)
        source = Batch.from_data_list([template], attr_map=schema)
        carried_template = source.get_data(0)
        buffer = Batch.empty(
            num_systems=1,
            num_nodes=2,
            num_edges=0,
            template=carried_template,
        )

        buffer.put(source, torch.ones(1, dtype=torch.bool))

        assert buffer.metadata_values.dtype == torch.float64
        torch.testing.assert_close(
            buffer.get_data(0).metadata_values, carried_template.metadata_values
        )

    @pytest.mark.parametrize(
        "capacities, error",
        [
            ({}, "Missing capacity"),
            ({"unknown": 1}, "Unknown level"),
            ({"atoms": 1}, "built-in"),
            ({"molecules": 1.5}, "integer"),
            ({"molecules": True}, "integer"),
            ({"molecules": -1}, "non-negative"),
        ],
    )
    def test_empty_rejects_invalid_custom_capacities(self, capacities, error):
        schema = _custom_buffer_schema()
        source = Batch.from_data_list([_custom_buffer_data(2, 3)], attr_map=schema)
        with pytest.raises((ValueError, TypeError), match=error):
            Batch.empty(
                num_systems=2,
                num_nodes=10,
                num_edges=0,
                template=source,
                level_capacities=capacities,
            )

    def test_empty_like_preserves_custom_layout_and_no_system_graph_capacity(self):
        schema = _custom_buffer_schema()
        source = Batch.from_data_list(
            [
                _custom_buffer_data(2, 3),
                _custom_buffer_data(3, 1),
            ],
            attr_map=schema,
        )
        buffer = Batch.empty(
            num_systems=4,
            num_nodes=20,
            num_edges=0,
            template=source,
            level_capacities={"molecules": 10, "pairs": 20},
        )

        clone = Batch.empty_like(buffer)

        assert clone._storage.attr_map is not buffer._storage.attr_map
        assert tuple(clone._storage.groups) == tuple(buffer._storage.groups)
        assert clone._storage.groups["molecules"]._data.shape[0] == 10
        assert clone._storage.groups["pairs"]._data.shape[0] == 20
        assert clone._storage.groups["atoms"]._data.shape[0] == 20
        assert clone._storage.groups["atoms"]._batch_ptr.shape[0] == 6

    def test_empty_keeps_product_capacity_independent_of_parent_capacity(self):
        schema = _custom_buffer_schema()
        source = Batch.from_data_list(
            [_custom_buffer_data(2, 3)],
            attr_map=schema,
        )

        buffer = Batch.empty(
            num_systems=2,
            num_nodes=100,
            num_edges=0,
            template=source,
            level_capacities={"molecules": 3, "pairs": 7},
        )

        assert buffer._storage.groups["molecules"]._data.shape[0] == 3
        assert buffer._storage.groups["pairs"]._data.shape[0] == 7
        with pytest.raises(ValueError, match="pairs"):
            Batch.empty(
                num_systems=2,
                num_nodes=100,
                num_edges=0,
                template=source,
                level_capacities={"molecules": 3},
            )

    def test_trim_preserves_custom_uniform_segmented_product_and_fieldless_parent(
        self,
    ):
        schema = LevelSchema()
        schema.add_level("metadata", segmented=False)
        schema.add_level("molecules", segmented=True)
        schema.add_level("fieldless", segmented=True)
        schema.add_product_level("pairs", left="atoms", right="fieldless")
        schema.set("metadata_values", "metadata")
        schema.set("molecule_values", "molecules")
        schema.set("pair_values", "pairs")

        data_list = []
        for num_nodes, num_molecules, num_fieldless in (
            (2, 3, 2),
            (4, 1, 3),
            (1, 2, 0),
        ):
            data = _minimal_atomic_data(num_nodes)
            data.metadata_values = torch.tensor([[float(num_nodes)]])
            data.molecule_values = torch.arange(
                num_molecules, dtype=torch.float32
            ).reshape(-1, 1)
            data.pair_values = torch.arange(
                num_nodes * num_fieldless, dtype=torch.float32
            ).reshape(num_nodes, num_fieldless, 1)
            data_list.append(data)

        batch = Batch.from_data_list(data_list, attr_map=schema)
        trimmed = batch.trim(torch.tensor([True, False, True]))

        assert trimmed is not None
        assert trimmed.num_graphs == 1
        assert isinstance(trimmed._storage.groups["metadata"], UniformLevelStorage)
        assert isinstance(trimmed._storage.groups["molecules"], SegmentedLevelStorage)
        assert isinstance(trimmed._storage.groups["fieldless"], SegmentedLevelStorage)
        assert isinstance(trimmed._storage.groups["pairs"], SegmentedLevelStorage)
        assert trimmed.metadata_values.shape == (1, 1)
        assert trimmed.molecule_values.shape == (1, 1)
        assert trimmed.level_ptr("fieldless").tolist() == [0, 3]
        assert trimmed.level_ptr("pairs").tolist() == [0, 12]
        assert trimmed.get_data(0).pair_values.shape == (4, 3, 1)

    def test_put_combines_custom_rejection_before_legacy_copy(self):
        schema = _custom_buffer_schema(fieldless_parent=True)
        source = Batch.from_data_list(
            [
                _custom_buffer_data(2, 3, fieldless_parent=True),
                _custom_buffer_data(2, 3, fieldless_parent=True),
            ],
            attr_map=schema,
        )
        buffer = Batch.empty(
            num_systems=2,
            num_nodes=4,
            num_edges=0,
            template=source,
            level_capacities={"pairs": 3},
        )
        copied = torch.zeros(2, dtype=torch.bool)

        buffer.put(source, torch.ones(2, dtype=torch.bool), copied_mask=copied)

        assert copied.tolist() == [False, False]
        assert buffer.num_graphs == 0
        assert buffer.positions.eq(0).all()

    def test_put_fieldless_overflow_is_rejected_before_uniform_copy(self):
        maximum = torch.iinfo(torch.int32).max
        first = _custom_boundary_batch(maximum, 11.0)
        second = _custom_boundary_batch(1, 22.0)
        buffer = Batch.empty(
            num_systems=2,
            num_nodes=0,
            num_edges=0,
            template=first,
        )
        occupancy = torch.zeros(2, dtype=torch.bool)
        copied = torch.zeros(1, dtype=torch.bool)

        buffer.put(
            first,
            torch.tensor([True]),
            copied_mask=copied,
            dest_mask=occupancy,
        )
        assert copied.tolist() == [True]
        assert occupancy.tolist() == [True, False]

        copied.zero_()
        buffer.put(
            second,
            torch.tensor([True]),
            copied_mask=copied,
            dest_mask=occupancy,
        )

        assert copied.tolist() == [False]
        assert occupancy.tolist() == [True, False]
        assert buffer.num_graphs == 1
        assert buffer.energy.tolist() == [[11.0], [0.0]]
        assert buffer.level_ptr("samples").tolist() == [0, maximum]

    def test_put_uniform_groups_share_occupancy_snapshot(self):
        schema = _custom_uniform_schema()
        data_list = []
        for energy, metadata in ((10.0, 20.0), (30.0, 40.0)):
            data = _atomic_data_with_system(2)
            data.energy = torch.tensor([[energy]])
            data.metadata_values = torch.tensor([[metadata]])
            data_list.append(data)
        source = Batch.from_data_list(data_list, attr_map=schema)
        buffer = Batch.empty(
            num_systems=3,
            num_nodes=10,
            num_edges=0,
            template=source,
        )
        copied = torch.zeros(2, dtype=torch.bool)
        dest_mask = torch.tensor([True, False, True])

        buffer.put(
            source,
            torch.ones(2, dtype=torch.bool),
            copied_mask=copied,
            dest_mask=dest_mask,
        )

        assert copied.tolist() == [True, False]
        assert dest_mask.tolist() == [True, True, True]
        assert buffer.energy[1].item() == 10.0
        assert buffer.metadata_values[1].item() == 20.0
        assert buffer.energy[0].item() == 0.0
        assert buffer.energy[2].item() == 0.0

    def test_put_and_defrag_preserve_mixed_builtin_buffer_dtypes(self):
        first = _minimal_atomic_data(2)
        second = _minimal_atomic_data(3)
        first.atomic_numbers = torch.tensor([1, 6], dtype=torch.int64)
        second.atomic_numbers = torch.tensor([8, 1, 1], dtype=torch.int64)
        source = Batch.from_data_list([first, second])
        buffer = Batch.empty(
            num_systems=2,
            num_nodes=5,
            num_edges=0,
            template=source,
        )
        copied = torch.zeros(2, dtype=torch.bool)

        buffer.put(source, torch.tensor([True, False]), copied_mask=copied)

        assert copied.tolist() == [True, False]
        torch.testing.assert_close(buffer.atomic_numbers[:2], first.atomic_numbers)
        torch.testing.assert_close(buffer.positions[:2], first.positions)
        source.defrag(copied)
        torch.testing.assert_close(source.atomic_numbers[:3], second.atomic_numbers)
        torch.testing.assert_close(source.positions[:3], second.positions)

    def test_put_omits_paired_fieldless_uniform_schema_group(self):
        schema = LevelSchema()
        schema.add_level("metadata", segmented=False)
        source = Batch.from_data_list(
            [_minimal_atomic_data(2), _minimal_atomic_data(2)], attr_map=schema
        )
        buffer = Batch.empty(
            num_systems=2,
            num_nodes=4,
            num_edges=0,
            template=source,
        )
        source_metadata = UniformLevelStorage(
            data=None, device="cpu", attr_map=schema, validate=False
        )
        source_metadata._data = TensorDict({}, batch_size=[2], device="cpu")
        buffer_metadata = UniformLevelStorage(
            data=None, device="cpu", attr_map=schema, validate=False
        )
        buffer_metadata._data = TensorDict({}, batch_size=[2], device="cpu")
        source._storage.groups["metadata"] = source_metadata
        buffer._storage.groups["metadata"] = buffer_metadata
        copied = torch.zeros(2, dtype=torch.bool)

        buffer.put(source, torch.ones(2, dtype=torch.bool), copied_mask=copied)

        assert copied.tolist() == [True, True]
        assert buffer.num_graphs == 2
        assert buffer._storage.groups["metadata"]._data.is_empty()

    def test_put_rejects_builtin_dtype_mismatch_before_any_group_mutates(self):
        source = Batch.from_data_list([_minimal_atomic_data(2)])
        source._storage.groups["atoms"]._data["atomic_numbers"] = torch.ones(
            2, dtype=torch.int32
        )
        buffer = Batch.empty(
            num_systems=1,
            num_nodes=2,
            num_edges=0,
            template=source,
        )
        buffer._storage.groups["atoms"]._data["atomic_numbers"] = torch.zeros(
            2, dtype=torch.int64
        )
        copied = torch.zeros(1, dtype=torch.bool)
        before_positions = buffer.positions.clone()
        before_ptr = buffer.batch_ptr.clone()

        with pytest.raises(
            ValueError,
            match=(
                "Level 'atoms' field 'atomic_numbers' has incompatible buffer dtypes: "
                "torch.int64 vs torch.int32"
            ),
        ):
            buffer.put(source, torch.ones(1, dtype=torch.bool), copied_mask=copied)

        assert copied.tolist() == [False]
        torch.testing.assert_close(buffer.positions, before_positions)
        torch.testing.assert_close(buffer.batch_ptr, before_ptr)

    def test_defrag_prevalidates_all_groups_before_compacting_any_payload(self):
        batch = Batch.from_data_list([_minimal_atomic_data(2), _minimal_atomic_data(2)])
        batch._storage.groups["atoms"]._data["atomic_numbers"] = torch.ones(
            4, dtype=torch.float16
        )
        before_positions = batch.positions.clone()
        before_ptr = batch.batch_ptr.clone()

        with pytest.raises(
            ValueError,
            match=(
                "Level 'atoms' field 'atomic_numbers' dtype torch.float16 is not "
                "supported by buffer kernels"
            ),
        ):
            batch.defrag(torch.tensor([True, False]))

        torch.testing.assert_close(batch.positions, before_positions)
        torch.testing.assert_close(batch.batch_ptr, before_ptr)

    def test_put_and_defrag_support_product_and_fieldless_parent(self):
        schema = _custom_buffer_schema(fieldless_parent=True)
        source = Batch.from_data_list(
            [
                _custom_buffer_data(2, 3, fieldless_parent=True),
                _custom_buffer_data(3, 1, fieldless_parent=True),
            ],
            attr_map=schema,
        )
        buffer = Batch.empty(
            num_systems=3,
            num_nodes=20,
            num_edges=0,
            template=source,
            level_capacities={"pairs": 20},
        )
        copied = torch.zeros(2, dtype=torch.bool)
        expected_first = source.get_data(0).pair_values.clone()
        expected_second = source.get_data(1).pair_values.clone()

        buffer.put(source, torch.tensor([True, False]), copied_mask=copied)

        assert copied.tolist() == [True, False]
        assert buffer.num_graphs == 1
        assert buffer._storage.groups["molecules"].batch_ptr[:2].tolist() == [0, 3]
        assert buffer.pair_values.shape == (20, 1)
        assert buffer.level_ptr("pairs").tolist() == [0, 6]
        assert buffer.level_ptr("pairs").numel() == buffer.num_graphs + 1
        torch.testing.assert_close(buffer.get_data(0).pair_values, expected_first)
        source.defrag(copied)
        assert source.num_graphs == 1
        assert source.level_ptr("pairs")[:2].tolist() == [0, 3]
        torch.testing.assert_close(source.get_data(0).pair_values, expected_second)

    def test_zero_clears_segment_caches_and_reuses_custom_buffer(self):
        schema = _custom_buffer_schema(fieldless_parent=True)
        source = Batch.from_data_list(
            [_custom_buffer_data(2, 3, fieldless_parent=True)], attr_map=schema
        )
        buffer = Batch.empty(
            num_systems=2,
            num_nodes=10,
            num_edges=0,
            template=source,
            level_capacities={"pairs": 10},
        )
        buffer.put(source, torch.ones(1, dtype=torch.bool))
        source.defrag()
        buffer.zero()

        assert buffer.num_graphs == 0
        assert buffer._storage.groups["pairs"].batch_ptr.shape[0] == 4
        assert buffer.pair_values.eq(0).all()
        assert not hasattr(buffer._storage.groups["pairs"], "_num_segments")
        buffer.put(
            Batch.from_data_list(
                [_custom_buffer_data(2, 3, fieldless_parent=True)], attr_map=schema
            ),
            torch.ones(1, dtype=torch.bool),
        )
        assert buffer.num_graphs == 1

    def test_put_mismatched_custom_layout_is_atomic(self):
        target_schema = _custom_buffer_schema()
        source_schema = _custom_buffer_schema()
        source_schema.set("molecule_values", "molecules", dtype="float64")
        target_source = Batch.from_data_list(
            [_custom_buffer_data(2, 2)], attr_map=target_schema
        )
        source_data = _custom_buffer_data(2, 2)
        source_data.molecule_values = source_data.molecule_values.to(torch.float64)
        source = Batch.from_data_list([source_data], attr_map=source_schema)
        buffer = Batch.empty(
            num_systems=2,
            num_nodes=10,
            num_edges=0,
            template=target_source,
            level_capacities={"molecules": 5, "pairs": 5},
        )
        with pytest.raises(ValueError, match="Custom put"):
            buffer.put(source, torch.ones(1, dtype=torch.bool))
        assert buffer.num_graphs == 0
        assert buffer.positions.eq(0).all()

    def test_put_accepts_equivalent_custom_dtype_aliases(self):
        target_schema = _custom_buffer_schema()
        target_schema.set("molecule_values", "molecules", dtype="float32")
        source_schema = _custom_buffer_schema()
        source_schema.set("molecule_values", "molecules", dtype="float")
        target = Batch.from_data_list(
            [_custom_buffer_data(2, 2)], attr_map=target_schema
        )
        source = Batch.from_data_list(
            [_custom_buffer_data(2, 2)], attr_map=source_schema
        )
        buffer = Batch.empty(
            num_systems=2,
            num_nodes=10,
            num_edges=0,
            template=target,
            level_capacities={"molecules": 5, "pairs": 5},
        )

        buffer.put(source, torch.ones(1, dtype=torch.bool))

        assert buffer.num_graphs == 1
        assert buffer.molecule_values[:2, 0].tolist() == [0.0, 1.0]

    @pytest.mark.parametrize("kind", ["uniform", "segmented", "product"])
    def test_put_rejects_non_float32_custom_buffer_payload_before_mutation(self, kind):
        schema = LevelSchema()
        data = _minimal_atomic_data(2)
        if kind == "uniform":
            schema.add_level("observables", segmented=False)
            schema.set("observable_values", "observables")
            data.observable_values = torch.tensor([[1]], dtype=torch.float16)
            level = "observables"
            capacities = {}
            error = "not supported by uniform buffer kernels"
        elif kind == "segmented":
            schema.add_level("samples", segmented=True)
            schema.set("sample_values", "samples")
            data.sample_values = torch.tensor([[1], [2]], dtype=torch.int64)
            level = "samples"
            capacities = {"samples": 4}
            error = "must use float32"
        else:
            schema.add_level("samples", segmented=True)
            schema.add_product_level("pairs", left="atoms", right="samples")
            schema.set("pair_values", "pairs")
            data.pair_values = torch.arange(6, dtype=torch.int64).reshape(2, 3, 1)
            level = "pairs"
            capacities = {"pairs": 8}
            error = "must use float32"

        source = Batch.from_data_list([data], attr_map=schema)
        buffer = Batch.empty(
            num_systems=2,
            num_nodes=4,
            num_edges=0,
            template=source,
            level_capacities=capacities,
        )
        before_positions = buffer.positions.clone()
        before_payload = {
            key: value.clone() for key, value in buffer._storage.groups[level].items()
        }
        before_ptr = buffer.level_ptr(level).clone()
        copied = torch.zeros(1, dtype=torch.bool)

        with pytest.raises(ValueError, match=error):
            buffer.put(source, torch.ones(1, dtype=torch.bool), copied_mask=copied)

        assert copied.tolist() == [False]
        assert buffer.num_graphs == 0
        torch.testing.assert_close(buffer.positions, before_positions)
        for key, value in before_payload.items():
            torch.testing.assert_close(buffer._storage.groups[level][key], value)
        torch.testing.assert_close(buffer.level_ptr(level), before_ptr)


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

    def test_wait_constructs_fieldless_builtin_groups_after_length_receives(self):
        from unittest.mock import MagicMock, patch

        from nvalchemi.data.batch import _BatchRecvHandle

        template = _fieldless_builtin_batch([1, 1], [1, 1])
        pending_lengths = iter(([2, 3], [1, 0]))

        class ReceiveWork:
            def __init__(self, target, values):
                self.target = target
                self.values = values

            def wait(self):
                self.target.copy_(torch.tensor(self.values, dtype=torch.int32))

        def receive(target, *args, **kwargs):
            return ReceiveWork(target, next(pending_lengths))

        handle = _BatchRecvHandle(
            meta=torch.tensor([2, 5, 1], dtype=torch.int64),
            meta_handle=MagicMock(),
            src=0,
            device=torch.device("cpu"),
            template=template,
            base_tag=100,
            group=None,
        )
        with patch("torch.distributed.irecv", side_effect=receive):
            received = handle.wait()

        assert set(received._storage.groups) == {"atoms", "edges"}
        assert received.level_ptr("atoms").tolist() == [0, 2, 5]
        assert received.level_ptr("edges").tolist() == [0, 1, 1]
        assert received.level_keys == {"atoms": set(), "edges": set(), "system": set()}

    def test_wait_receives_public_fieldless_builtin_edge_product(self):
        """Receive waits for built-in lengths before product reconstruction."""
        from unittest.mock import MagicMock, patch

        from nvalchemi.data.batch import _BatchRecvHandle

        template = _builtin_edge_product_batch()
        pending_lengths = iter(([2, 3, 1], [3, 0, 2], [6, 0, 2]))

        class ReceiveWork:
            def __init__(self, target, values):
                self.target = target
                self.values = values

            def wait(self):
                self.target.copy_(torch.tensor(self.values, dtype=torch.int32))

        def receive(target, *args, **kwargs):
            return ReceiveWork(target, next(pending_lengths))

        payloads = iter(
            [
                template._storage.groups["atoms"]._data[: template.num_nodes],
                template._storage.groups["atom_edges"]._data[
                    : template._storage.groups["atom_edges"].num_elements()
                ],
            ]
        )

        def receive_tensordict(
            destination, src=None, init_tag=None, group=None, return_premature=False
        ):
            source = next(payloads)
            for key in destination.keys():
                destination[key].copy_(source[key])
            return [MagicMock()]

        handle = _BatchRecvHandle(
            meta=torch.tensor([3, 6, 5], dtype=torch.int64),
            meta_handle=MagicMock(),
            src=0,
            device=torch.device("cpu"),
            template=template,
            base_tag=100,
            group=None,
        )
        with (
            patch("torch.distributed.irecv", side_effect=receive),
            patch("tensordict.TensorDict.irecv", receive_tensordict),
        ):
            received = handle.wait()

        assert "neighbor_list" not in received
        assert received.level_ptr("edges").tolist() == [0, 3, 3, 5]
        assert received.level_ptr("atom_edges").tolist() == [0, 6, 6, 8]
        torch.testing.assert_close(received.atom_edge_values, template.atom_edge_values)
        for graph_idx in range(received.num_graphs):
            torch.testing.assert_close(
                received.get_data(graph_idx).atom_edge_values,
                template.get_data(graph_idx).atom_edge_values,
            )


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


class TestCustomBatchTransport:
    def test_legacy_transport_tags_have_no_custom_extension(self):
        from unittest.mock import MagicMock, patch

        batch = Batch.from_data_list([_minimal_atomic_data(2)])
        send_tags: list[int] = []
        td_tags: list[int] = []

        def capture_isend(*args, **kwargs):
            send_tags.append(kwargs["tag"])
            return MagicMock()

        def capture_td_isend(
            self, dst=None, init_tag=None, group=None, return_early=False
        ):
            td_tags.extend(init_tag + index for index, _ in enumerate(self.keys()))
            return [MagicMock()]

        with (
            patch("torch.distributed.isend", side_effect=capture_isend),
            patch("tensordict.TensorDict.isend", capture_td_isend),
        ):
            batch.isend(dst=1, tag=11)

        assert send_tags == [11, 12]
        assert td_tags == list(
            range(14, 14 + len(list(batch._storage.groups["atoms"].keys())))
        )

    def test_custom_extension_starts_after_legacy_and_reserves_fieldless_slot(self):
        from unittest.mock import MagicMock, patch

        batch = _custom_transport_batch()
        send_tags: list[int] = []
        td_tags: list[int] = []

        def capture_isend(*args, **kwargs):
            send_tags.append(kwargs["tag"])
            return MagicMock()

        def capture_td_isend(
            self, dst=None, init_tag=None, group=None, return_early=False
        ):
            td_tags.extend(init_tag + index for index, _ in enumerate(self.keys()))
            return [MagicMock()]

        with (
            patch("torch.distributed.isend", side_effect=capture_isend),
            patch("tensordict.TensorDict.isend", capture_td_isend),
        ):
            batch.isend(dst=1, tag=11)

        legacy_end = 3
        for name in ("atoms", "edges", "system"):
            group = batch._storage.groups.get(name)
            legacy_end += len(list(group.keys())) + 1 if group is not None else 1
        custom_start = 11 + legacy_end

        assert send_tags[:2] == [11, 12]
        assert send_tags[2:] == [
            custom_start,
            custom_start + 1,
            custom_start + 2,
            custom_start + 3,
        ]
        assert custom_start + 8 not in td_tags
        assert custom_start + 9 in td_tags
        assert custom_start + 11 in td_tags

    def test_custom_send_and_receive_tags_align_in_schema_order(self):
        from unittest.mock import MagicMock, patch

        from nvalchemi.data.batch import _BatchRecvHandle

        batch = _custom_transport_batch()
        send_tags: list[int] = []
        recv_tags: list[int] = []

        def capture_isend(*args, **kwargs):
            send_tags.append(kwargs["tag"])
            return MagicMock()

        def capture_td_isend(
            self, dst=None, init_tag=None, group=None, return_early=False
        ):
            send_tags.extend(init_tag + index for index, _ in enumerate(self.keys()))
            return [MagicMock()]

        with (
            patch("torch.distributed.isend", side_effect=capture_isend),
            patch("tensordict.TensorDict.isend", capture_td_isend),
        ):
            batch.isend(dst=1, tag=23)

        def capture_irecv(tensor, *args, **kwargs):
            recv_tags.append(kwargs["tag"])
            if tensor.dtype == torch.int32:
                tensor.zero_()
            return MagicMock()

        def capture_td_irecv(
            self, src=None, init_tag=None, group=None, return_premature=False
        ):
            recv_tags.extend(init_tag + index for index, _ in enumerate(self.keys()))
            return [MagicMock()]

        handle = _BatchRecvHandle(
            meta=torch.tensor(
                [batch.num_graphs, batch.num_nodes, batch.num_edges], dtype=torch.int64
            ),
            meta_handle=MagicMock(),
            src=0,
            device=torch.device("cpu"),
            template=batch,
            base_tag=23,
            group=None,
        )
        with (
            patch("torch.distributed.irecv", side_effect=capture_irecv),
            patch("tensordict.TensorDict.irecv", capture_td_irecv),
        ):
            handle.wait()

        assert send_tags[1:] == recv_tags

    def test_custom_sentinel_preserves_schema_without_receives(self):
        from unittest.mock import MagicMock, patch

        from nvalchemi.data.batch import _BatchRecvHandle

        template = _custom_transport_batch()
        handle = _BatchRecvHandle(
            meta=torch.zeros(3, dtype=torch.int64),
            meta_handle=MagicMock(),
            src=0,
            device=torch.device("cpu"),
            template=template,
            base_tag=0,
            group=None,
        )
        with patch("torch.distributed.irecv") as mock_irecv:
            received = handle.wait()

        mock_irecv.assert_not_called()
        assert received._storage.attr_map is not template._storage.attr_map
        assert (
            received._storage.attr_map.level_names
            == template._storage.attr_map.level_names
        )
        assert list(received._storage.groups) == list(template._storage.groups)
        assert received._storage.groups["fieldless"].segment_lengths.numel() == 0
        assert received._storage.attr_map.product_parents["fieldless_pairs"] == (
            "atoms",
            "fieldless",
        )

    @pytest.mark.skipif(
        not dist.is_available() or not dist.is_gloo_available(),
        reason="gloo backend is required",
    )
    @pytest.mark.parametrize("sentinel", [False, True])
    def test_custom_transport_round_trip_over_gloo(self, sentinel):
        mp.spawn(
            _custom_transport_gloo_worker,
            args=(2, _available_tcp_port(), sentinel),
            nprocs=2,
            join=True,
        )

    @pytest.mark.skipif(
        not dist.is_available() or not dist.is_gloo_available(),
        reason="gloo backend is required",
    )
    def test_fieldless_builtin_edge_product_round_trip_over_gloo(self):
        mp.spawn(
            _builtin_edge_product_gloo_worker,
            args=(2, _available_tcp_port()),
            nprocs=2,
            join=True,
        )


class TestBatchFromRawDicts:
    """Tests for Batch.from_raw_dicts (validation-free batch construction)."""

    def test_matches_from_data_list(self):
        """from_raw_dicts produces identical tensors to from_data_list."""
        data_list = [_atomic_data_with_edges_and_system(3, 4) for _ in range(5)]
        ref = Batch.from_data_list(data_list, skip_validation=True)

        raw_dicts = [
            {
                "positions": d.positions,
                "atomic_numbers": d.atomic_numbers,
                "neighbor_list": d.neighbor_list,
                "energy": d.energy,
            }
            for d in data_list
        ]
        result = Batch.from_raw_dicts(raw_dicts)

        assert result.num_graphs == ref.num_graphs
        assert result.num_nodes == ref.num_nodes
        assert result.num_edges == ref.num_edges
        torch.testing.assert_close(result.positions, ref.positions)
        torch.testing.assert_close(result.atomic_numbers, ref.atomic_numbers)
        torch.testing.assert_close(result.neighbor_list, ref.neighbor_list)
        torch.testing.assert_close(result.energy, ref.energy)

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="empty data list"):
            Batch.from_raw_dicts([])

    def test_node_offset_applied_to_neighbor_list(self):
        """neighbor_list indices are offset by cumulative node count."""
        d0 = {
            "atomic_numbers": torch.tensor([1, 2]),
            "neighbor_list": torch.tensor([[0, 1]]),
        }
        d1 = {
            "atomic_numbers": torch.tensor([3]),
            "neighbor_list": torch.tensor([[0, 0]]),
        }
        batch = Batch.from_raw_dicts([d0, d1])
        # d1's neighbor_list should be offset by 2 (num_nodes in d0)
        assert batch.neighbor_list[-1, 0].item() == 2
        assert batch.neighbor_list[-1, 1].item() == 2

    def test_keys_tracking(self):
        """Batch.keys correctly reports node/edge/system sets."""
        raw = [
            {
                "positions": torch.randn(2, 3),
                "atomic_numbers": torch.tensor([1, 2]),
                "energy": torch.tensor([[0.5]]),
                "neighbor_list": torch.zeros(1, 2, dtype=torch.long),
            }
        ]
        batch = Batch.from_raw_dicts(raw)
        assert "positions" in batch.keys["node"]
        assert "neighbor_list" in batch.keys["edge"]
        assert "energy" in batch.keys["system"]

    def test_segment_lengths(self):
        """Per-graph node/edge counts are correct."""
        d0 = {
            "atomic_numbers": torch.tensor([1, 2, 3]),
            "positions": torch.randn(3, 3),
            "neighbor_list": torch.zeros(2, 2, dtype=torch.long),
        }
        d1 = {
            "atomic_numbers": torch.tensor([4]),
            "positions": torch.randn(1, 3),
            "neighbor_list": torch.zeros(5, 2, dtype=torch.long),
        }
        batch = Batch.from_raw_dicts([d0, d1])
        assert batch.num_nodes_list == [3, 1]
        assert batch.num_edges_list == [2, 5]

    def test_custom_key_preserved_as_system(self) -> None:
        """Keys not in _default_*_keys are preserved as system-level."""
        d0 = {
            "atomic_numbers": torch.tensor([1, 2]),
            "positions": torch.randn(2, 3),
            "my_custom_scalar": torch.tensor([42.0]),
        }
        d1 = {
            "atomic_numbers": torch.tensor([3]),
            "positions": torch.randn(1, 3),
            "my_custom_scalar": torch.tensor([99.0]),
        }
        batch = Batch.from_raw_dicts([d0, d1])
        assert "my_custom_scalar" in batch.keys["system"]
        assert batch.my_custom_scalar.shape == (2,)
        assert batch.my_custom_scalar[0].item() == 42.0
        assert batch.my_custom_scalar[1].item() == 99.0

    def test_custom_later_only_field_in_raw_dict_is_rejected_with_sample_index(self):
        schema = LevelSchema()
        schema.add_level("samples", segmented=True)
        schema.set("later_values", "samples")
        raw = [
            {
                "atomic_numbers": torch.tensor([1, 2]),
                "positions": torch.zeros(2, 3),
            },
            {
                "atomic_numbers": torch.tensor([3]),
                "positions": torch.zeros(1, 3),
                "later_values": torch.ones(1, 1),
            },
        ]

        with pytest.raises(ValueError, match="appears only in later sample 1"):
            Batch.from_raw_dicts(raw, attr_map=schema)

    def test_field_levels_classifies_custom_atom_key(self) -> None:
        """field_levels routes custom per-atom tensors to atom level."""
        d0 = {
            "atomic_numbers": torch.tensor([1, 2, 3]),
            "positions": torch.randn(3, 3),
            "partial_charges": torch.tensor([0.1, 0.2, 0.3]),
        }
        d1 = {
            "atomic_numbers": torch.tensor([4, 5]),
            "positions": torch.randn(2, 3),
            "partial_charges": torch.tensor([0.4, 0.5]),
        }
        batch = Batch.from_raw_dicts([d0, d1], field_levels={"partial_charges": "atom"})
        assert "partial_charges" in batch.keys["node"]
        assert batch.partial_charges.shape == (5,)
        torch.testing.assert_close(
            batch.partial_charges,
            torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5]),
        )

    def test_field_levels_classifies_custom_edge_key(self) -> None:
        """field_levels routes custom per-edge tensors to edge level."""
        d0 = {
            "atomic_numbers": torch.tensor([1, 2]),
            "positions": torch.randn(2, 3),
            "neighbor_list": torch.tensor([[0, 1], [1, 0]]),
            "edge_weights": torch.tensor([1.0, 2.0]),
        }
        d1 = {
            "atomic_numbers": torch.tensor([3]),
            "positions": torch.randn(1, 3),
            "neighbor_list": torch.tensor([[0, 0]]),
            "edge_weights": torch.tensor([3.0]),
        }
        batch = Batch.from_raw_dicts([d0, d1], field_levels={"edge_weights": "edge"})
        assert "edge_weights" in batch.keys["edge"]
        assert batch.edge_weights.shape == (3,)

    def test_field_levels_fallback_still_system(self) -> None:
        """Keys absent from both default sets and field_levels fall back to system."""
        d0 = {
            "atomic_numbers": torch.tensor([1, 2]),
            "positions": torch.randn(2, 3),
            "unknown_scalar": torch.tensor([1.0]),
        }
        d1 = {
            "atomic_numbers": torch.tensor([3]),
            "positions": torch.randn(1, 3),
            "unknown_scalar": torch.tensor([2.0]),
        }
        # field_levels is provided but doesn't mention unknown_scalar
        batch = Batch.from_raw_dicts([d0, d1], field_levels={"some_other_key": "atom"})
        assert "unknown_scalar" in batch.keys["system"]

    def test_schema_routes_custom_raw_levels_and_field_levels_override(self) -> None:
        schema = LevelSchema()
        schema.add_level("samples", segmented=True)
        schema.set("sample_values", "samples")
        raw = [
            {
                "atomic_numbers": torch.tensor([1, 2]),
                "positions": torch.zeros(2, 3),
                "sample_values": torch.zeros(3, 1),
                "system_value": torch.zeros(1, 1),
            },
            {
                "atomic_numbers": torch.tensor([3]),
                "positions": torch.zeros(1, 3),
                "sample_values": torch.ones(1, 1),
                "system_value": torch.ones(1, 1),
            },
        ]
        batch = Batch.from_raw_dicts(raw, attr_map=schema)
        assert batch.level_ptr("samples").tolist() == [0, 3, 4]

        override_schema = schema.clone()
        override_schema.set("system_value", "samples")
        overridden = Batch.from_raw_dicts(
            raw,
            attr_map=override_schema,
            field_levels={"system_value": "system"},
        )
        assert "system_value" in overridden.level_keys["system"]
        assert (
            "system_value"
            in Batch.from_data_list([overridden.get_data(0)]).level_keys["system"]
        )

    def test_raw_product_uses_logical_axes_and_round_trips(self) -> None:
        schema = LevelSchema()
        schema.add_level("left_items", segmented=True)
        schema.add_level("right_items", segmented=True)
        schema.add_product_level("left_right", left="left_items", right="right_items")
        schema.set("pair_values", "left_right")
        raw = [
            {
                "atomic_numbers": torch.tensor([1, 2]),
                "positions": torch.zeros(2, 3),
                "pair_values": torch.zeros(2, 3, 4),
            },
            {
                "atomic_numbers": torch.tensor([3]),
                "positions": torch.zeros(1, 3),
                "pair_values": torch.ones(1, 2, 4),
            },
        ]

        batch = Batch.from_raw_dicts(raw, attr_map=schema)

        assert batch.level_ptr("left_items").tolist() == [0, 2, 3]
        assert batch.level_ptr("right_items").tolist() == [0, 3, 5]
        assert batch.level_ptr("left_right").tolist() == [0, 6, 8]
        assert batch.get_data(0).pair_values.shape == (2, 3, 4)
        assert batch.get_data(1).pair_values.shape == (1, 2, 4)
        rebatch = Batch.from_data_list(batch.to_data_list())
        assert rebatch.level_keys["left_right"] == {"pair_values"}
        assert rebatch.level_ptr("left_right").tolist() == [0, 6, 8]


class TestSetTransient:
    """``set_transient`` overlays a value without disturbing stored state.

    The distinction is load-bearing: a strain applied for one forward must not
    become the batch's real geometry, or a later rebuild deforms an already
    deformed cell. Plain assignment routes tensors into storage and would.
    """

    def _batch(self) -> Batch:
        data = AtomicData(
            positions=torch.zeros(4, 3),
            atomic_numbers=torch.ones(4, dtype=torch.long),
            cell=torch.eye(3).unsqueeze(0),
            pbc=torch.ones(1, 3, dtype=torch.bool),
        )
        return Batch.from_data_list([data])

    def test_overlay_is_visible_to_readers(self) -> None:
        batch = self._batch()
        set_transient(batch, "cell", batch.cell * 7.0)
        assert batch.cell[0, 0, 0].item() == pytest.approx(7.0)

    def test_overlay_leaves_storage_untouched(self) -> None:
        batch = self._batch()
        stored = batch._storage["cell"].clone()
        set_transient(batch, "cell", batch.cell * 7.0)
        torch.testing.assert_close(batch._storage["cell"], stored)

    def test_plain_assignment_writes_through_to_storage(self) -> None:
        """Contrast: the reason ``set_transient`` exists rather than ``setattr``."""
        batch = self._batch()
        batch.cell = batch.cell * 7.0
        assert batch._storage["cell"][0, 0, 0].item() == pytest.approx(7.0)
