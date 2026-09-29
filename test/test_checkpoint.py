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
"""Unit tests for the shared transactional checkpoint layer.

Covers the parts that belong to no workflow: state encoding without pickle,
the checksum, the manifest as commit marker, and the ``Stateful`` contract
that ``save_checkpoint`` and ``load_checkpoint`` are written against.

Driving the same machinery through a real workflow — an integrator, adaptive
biases, a replica-exchange ladder — lives in
``test/enhanced_sampling/test_checkpoint.py``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest
import torch
import zarr

from nvalchemi._checkpoint import (
    CHECKPOINT_FORMAT_VERSION,
    CheckpointManifest,
    Stateful,
    _component_checksum,
    _decode_state,
    _encode_state,
    load_checkpoint,
    save_checkpoint,
)
from nvalchemi.data import AtomicData, Batch

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _Counter:
    """A minimal ``Stateful``: one integer and one tensor."""

    def __init__(self, count: int = 0) -> None:
        self.count = count
        self.history = torch.zeros(3)

    def state_dict(self) -> Mapping[str, Any]:
        """Return the state."""
        return {"count": self.count, "history": self.history}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore the state."""
        self.count = int(state["count"])
        self.history = state["history"]


def _make_batch(n_graphs: int = 2, atoms_per_graph: int = 3) -> Batch:
    """Return a small batch with a per-graph label field."""
    torch.manual_seed(0)
    data_list = [
        AtomicData(
            positions=torch.randn(atoms_per_graph, 3),
            atomic_numbers=torch.full((atoms_per_graph,), 6, dtype=torch.long),
            atomic_masses=torch.ones(atoms_per_graph),
            forces=torch.zeros(atoms_per_graph, 3),
            energy=torch.zeros(1, 1),
        )
        for _ in range(n_graphs)
    ]
    batch = Batch.from_data_list(data_list)
    batch["walker_id"] = torch.arange(n_graphs, dtype=torch.long)
    return batch


# ===========================================================================
# 1. State encoding
# ===========================================================================


class TestStateEncoding:
    """Nested state survives the Zarr round-trip without pickle."""

    def test_tensors_scalars_and_nesting(self, tmp_path) -> None:
        state = {
            "counter": 7,
            "label": "umbrella",
            "ratio": 0.25,
            "flag": True,
            "nothing": None,
            "listy": [1, 2, 3],
            "weights": torch.arange(6, dtype=torch.float64).reshape(2, 3),
            "ids": torch.tensor([4, 5], dtype=torch.int64),
            "nested": {"inner": torch.ones(2), "depth": 2},
        }
        group = zarr.open_group(str(tmp_path / "s.zarr"), mode="w")
        _encode_state(group, state)
        restored = _decode_state(group, "cpu")

        assert restored["counter"] == 7
        assert restored["label"] == "umbrella"
        assert restored["flag"] is True
        assert restored["nothing"] is None
        assert restored["listy"] == [1, 2, 3]
        assert torch.equal(restored["weights"], state["weights"])
        assert restored["weights"].dtype == torch.float64
        assert restored["ids"].dtype == torch.int64
        assert torch.equal(restored["nested"]["inner"], state["nested"]["inner"])
        assert restored["nested"]["depth"] == 2

    def test_empty_tensor_round_trips(self, tmp_path) -> None:
        group = zarr.open_group(str(tmp_path / "s.zarr"), mode="w")
        _encode_state(group, {"empty": torch.zeros(0, 3)})
        restored = _decode_state(group, "cpu")
        assert restored["empty"].shape == (0, 3)

    def test_zero_dimensional_tensor_round_trips(self, tmp_path) -> None:
        """Zarr stores a 0-d array as shape (1,); the rank must be restored.

        Scalar buffers are how a compile-safe bias holds its counters — a
        Python int would be a data-dependent value in the traced graph.  If
        the rank comes back wrong the component no longer matches the digest
        taken when it was written, and restore fails its own checksum.
        """
        state = {"count": torch.tensor(5, dtype=torch.int64)}
        group = zarr.open_group(str(tmp_path / "s.zarr"), mode="w")
        _encode_state(group, state)
        restored = _decode_state(group, "cpu")

        assert restored["count"].shape == ()
        assert torch.equal(restored["count"], state["count"])
        assert _component_checksum(restored) == _component_checksum(state)

    def test_checksum_distinguishes_rank(self) -> None:
        """A scalar and a one-element vector are not the same state."""
        assert _component_checksum({"x": torch.tensor(5)}) != _component_checksum(
            {"x": torch.tensor([5])}
        )

    def test_unsupported_type_raises_rather_than_pickling(self, tmp_path) -> None:
        """Refusing is the point: a pickle payload would make a checkpoint
        executable and unreadable outside Python."""
        group = zarr.open_group(str(tmp_path / "s.zarr"), mode="w")
        with pytest.raises(TypeError, match="no pickle payloads"):
            _encode_state(group, {"bad": object()})

    def test_checksum_is_order_independent(self) -> None:
        a = {"x": torch.ones(3), "y": 2}
        b = {"y": 2, "x": torch.ones(3)}
        assert _component_checksum(a) == _component_checksum(b)

    def test_checksum_detects_value_change(self) -> None:
        base = _component_checksum({"x": torch.ones(3)})
        assert base != _component_checksum({"x": torch.zeros(3)})
        assert base != _component_checksum({"x": torch.ones(3) * 2})

    def test_checksum_detects_dtype_change(self) -> None:
        assert _component_checksum({"x": torch.ones(3)}) != _component_checksum(
            {"x": torch.ones(3, dtype=torch.float64)}
        )


# ===========================================================================
# 2. The Stateful contract
# ===========================================================================


class TestStatefulProtocol:
    """A checkpoint is a mapping of name to something that knows its state."""

    def test_the_toolkit_types_already_satisfy_it(self) -> None:
        """This is the argument for the protocol: nothing had to be adapted."""
        from nvalchemi.dynamics import NVTLangevin
        from nvalchemi.enhanced_sampling import BiasHook, ReplicaExchange
        from nvalchemi.enhanced_sampling import ThermodynamicState as State
        from nvalchemi.models.demo import DemoModel, DemoModelWrapper

        model = DemoModelWrapper(DemoModel())
        ladder = ReplicaExchange(
            [State(state_id=i, temperature=300.0 * 1.2**i) for i in range(2)],
            torch.arange(2),
        )
        engine = NVTLangevin(model=model, dt=0.1, temperature=300.0, friction=0.1)
        for obj in (model, engine, BiasHook({}), ladder):
            assert isinstance(obj, Stateful), type(obj).__name__

    def test_an_object_without_the_pair_is_not_stateful(self) -> None:
        class NotStateful:
            def state_dict(self) -> dict[str, Any]:
                return {}

        assert not isinstance(NotStateful(), Stateful)


# ===========================================================================
# 3. save_checkpoint / load_checkpoint
# ===========================================================================


class TestRoundTrip:
    """What goes in comes back, into the objects the caller names."""

    def test_state_only_checkpoint_needs_no_batch(self, tmp_path) -> None:
        """MD restart wants a batch; a bare strategy restart does not."""
        source = _Counter(count=9)
        source.history = torch.tensor([1.0, 2.0, 3.0])
        save_checkpoint(tmp_path / "c.zarr", {"counter": source})

        target = _Counter()
        contents = load_checkpoint(tmp_path / "c.zarr", {"counter": target})
        assert target.count == 9
        assert torch.equal(target.history, source.history)
        assert contents.batch is None
        assert contents.manifest.num_graphs is None
        assert contents.manifest.batch_checksum == ""

    def test_batch_round_trips_with_named_extra_fields(self, tmp_path) -> None:
        """AtomicDataZarrWriter drops what it does not recognise."""
        batch = _make_batch()
        save_checkpoint(
            tmp_path / "c.zarr",
            {"counter": _Counter(1)},
            batch=batch,
            batch_fields=("walker_id",),
        )
        contents = load_checkpoint(tmp_path / "c.zarr")
        assert contents.batch is not None
        assert contents.batch.num_graphs == 2
        assert contents.batch.walker_id.reshape(-1).tolist() == [0, 1]
        assert contents.manifest.batch_fields == ["walker_id"]

    def test_nested_component_names_become_nested_groups(self, tmp_path) -> None:
        save_checkpoint(
            tmp_path / "c.zarr",
            {"biases/umbrella": _Counter(2), "biases/wall": _Counter(3)},
        )
        root = zarr.open_group(str(tmp_path / "c.zarr"), mode="r")
        assert "checkpoint/biases/umbrella" in root
        assert "checkpoint/biases/wall" in root

    def test_states_are_returned_even_when_not_applied(self, tmp_path) -> None:
        """The escape hatch for a component with an ordering constraint."""
        save_checkpoint(tmp_path / "c.zarr", {"counter": _Counter(4)})
        contents = load_checkpoint(tmp_path / "c.zarr")
        assert contents.states["counter"]["count"] == 4

    def test_compatibility_and_metadata_round_trip_verbatim(self, tmp_path) -> None:
        save_checkpoint(
            tmp_path / "c.zarr",
            {"counter": _Counter()},
            compatibility={"engine": "NVTLangevin"},
            metadata={"step": 400},
        )
        manifest = load_checkpoint(tmp_path / "c.zarr").manifest
        assert manifest.compatibility == {"engine": "NVTLangevin"}
        assert manifest.metadata == {"step": 400}
        assert manifest.format_version == CHECKPOINT_FORMAT_VERSION

    def test_validate_runs_before_any_state_is_applied(self, tmp_path) -> None:
        """A mismatched checkpoint must leave the caller's objects untouched."""
        save_checkpoint(
            tmp_path / "c.zarr",
            {"counter": _Counter(7)},
            compatibility={"engine": "NVE"},
        )

        def _refuse(manifest: CheckpointManifest) -> None:
            raise ValueError(f"wrong engine: {manifest.compatibility['engine']}")

        target = _Counter(0)
        with pytest.raises(ValueError, match="wrong engine: NVE"):
            load_checkpoint(tmp_path / "c.zarr", {"counter": target}, validate=_refuse)
        assert target.count == 0, "state was applied despite validation failing"

    def test_requesting_an_absent_component_is_an_error(self, tmp_path) -> None:
        save_checkpoint(tmp_path / "c.zarr", {"counter": _Counter()})
        with pytest.raises(ValueError, match="does not hold component"):
            load_checkpoint(tmp_path / "c.zarr", {"ghost": _Counter()})


# ===========================================================================
# 4. Names the store cannot hold
# ===========================================================================


class TestComponentNames:
    """Names become Zarr groups, so some of them cannot be used."""

    def test_manifest_is_reserved(self, tmp_path) -> None:
        """It is the commit marker; a component there would overwrite it."""
        with pytest.raises(ValueError, match="reserved"):
            save_checkpoint(tmp_path / "c.zarr", {"manifest": _Counter()})

    @pytest.mark.parametrize("name", ["", "a//b", "/a"])
    def test_empty_path_segments_are_rejected(self, tmp_path, name: str) -> None:
        with pytest.raises(ValueError, match="empty"):
            save_checkpoint(tmp_path / "c.zarr", {name: _Counter()})

    def test_batch_fields_without_a_batch_is_an_error(self, tmp_path) -> None:
        with pytest.raises(ValueError, match="without a batch"):
            save_checkpoint(
                tmp_path / "c.zarr", {"counter": _Counter()}, batch_fields=("x",)
            )

    def test_a_batch_field_the_batch_lacks_is_an_error(self, tmp_path) -> None:
        """Silently skipping it would come back missing on restore."""
        with pytest.raises(ValueError, match="does not carry"):
            save_checkpoint(
                tmp_path / "c.zarr",
                {"counter": _Counter()},
                batch=_make_batch(),
                batch_fields=("nonexistent",),
            )


# ===========================================================================
# 5. The manifest is the commit marker
# ===========================================================================


class TestManifestIntegrity:
    """Cover is mandatory: a gap is a tampered manifest, not a free pass."""

    def _committed(self, tmp_path) -> Any:
        save_checkpoint(
            tmp_path / "c.zarr",
            {"counter": _Counter(5)},
            batch=_make_batch(),
            batch_fields=("walker_id",),
        )
        return tmp_path / "c.zarr"

    @staticmethod
    def _tamper(path, mutate) -> None:
        """Apply *mutate* to the manifest dict and write it back."""
        root = zarr.open_group(str(path), mode="a")
        manifest = dict(root["checkpoint/manifest"].attrs["manifest"])
        mutate(manifest)
        root["checkpoint/manifest"].attrs["manifest"] = manifest

    def test_store_without_a_manifest_is_refused(self, tmp_path) -> None:
        zarr.open_group(str(tmp_path / "torn.zarr"), mode="w")
        with pytest.raises(ValueError, match="no committed manifest"):
            load_checkpoint(tmp_path / "torn.zarr")

    def test_component_without_a_checksum_is_invalid(self, tmp_path) -> None:
        path = self._committed(tmp_path)
        self._tamper(path, lambda m: m["checksums"].pop("counter"))
        with pytest.raises(ValueError, match="with no checksum"):
            load_checkpoint(path)

    def test_orphaned_checksum_is_invalid(self, tmp_path) -> None:
        path = self._committed(tmp_path)
        self._tamper(path, lambda m: m["checksums"].__setitem__("ghost", "0" * 64))
        with pytest.raises(ValueError, match="does not declare as components"):
            load_checkpoint(path)

    def test_stripped_batch_checksum_is_invalid(self, tmp_path) -> None:
        path = self._committed(tmp_path)
        self._tamper(path, lambda m: m.__setitem__("batch_checksum", ""))
        with pytest.raises(ValueError, match="no batch_checksum"):
            load_checkpoint(path)

    def test_batch_checksum_without_a_batch_is_invalid(self, tmp_path) -> None:
        """The two must agree, in both directions."""
        save_checkpoint(tmp_path / "c.zarr", {"counter": _Counter()})
        self._tamper(
            tmp_path / "c.zarr", lambda m: m.__setitem__("batch_checksum", "0" * 64)
        )
        with pytest.raises(ValueError, match="records no batch"):
            load_checkpoint(tmp_path / "c.zarr")

    def test_corrupted_component_fails_its_checksum(self, tmp_path) -> None:
        path = self._committed(tmp_path)
        root = zarr.open_group(str(path), mode="a")
        root["checkpoint/counter/history"][...] = 99.0
        with pytest.raises(ValueError, match="failed its checksum"):
            load_checkpoint(path)

    def test_corrupted_batch_fails_its_checksum(self, tmp_path) -> None:
        path = self._committed(tmp_path)
        root = zarr.open_group(str(path), mode="a")
        root["core/positions"][...] = root["core/positions"][...] + 1.0
        with pytest.raises(ValueError, match="batch failed its checksum"):
            load_checkpoint(path)

    def test_missing_declared_group_is_caught(self, tmp_path) -> None:
        path = self._committed(tmp_path)
        root = zarr.open_group(str(path), mode="a")
        del root["checkpoint/counter"]
        with pytest.raises(ValueError, match="manifest but the group is missing"):
            load_checkpoint(path)

    def test_a_future_format_version_is_refused(self, tmp_path) -> None:
        path = self._committed(tmp_path)
        self._tamper(path, lambda m: m.__setitem__("format_version", 99))
        with pytest.raises(ValueError, match="format_version"):
            load_checkpoint(path)

    def test_error_names_the_store(self, tmp_path) -> None:
        """One ValueError naming the path, not a nested pydantic report."""
        path = self._committed(tmp_path)
        self._tamper(path, lambda m: m["checksums"].pop("counter"))
        with pytest.raises(ValueError) as excinfo:
            load_checkpoint(path)
        assert str(path) in str(excinfo.value)

    def test_no_pickle_in_the_store(self, tmp_path) -> None:
        """Loading a checkpoint must not be able to execute code."""
        path = self._committed(tmp_path)
        assert not list(path.rglob("*.pkl"))
        assert not list(path.rglob("*.pt"))
