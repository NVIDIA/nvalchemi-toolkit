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
"""Tests for :mod:`nvalchemi.training.distillation.labeling`."""

from __future__ import annotations

import time
import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
import torch
import zarr

from nvalchemi.data import AtomicData, Batch
from nvalchemi.data.datapipes.backends.zarr import (
    AtomicDataZarrReader,
    AtomicDataZarrWriter,
)
from nvalchemi.data.datapipes.dataset import Dataset
from nvalchemi.data.datapipes.in_memory_dataset import InMemoryDataset
from nvalchemi.data.datapipes.multidataset import MultiDataset
from nvalchemi.data.level_storage import LevelSchema
from nvalchemi.models.base import NeighborListFormat
from nvalchemi.models.lj import LennardJonesModelWrapper
from nvalchemi.neighbors import compute_neighbors
from nvalchemi.training.distillation import (
    InProcessTeacherScorer,
    SignalLevel,
    TeacherLabels,
    TeacherSignal,
    label_dataset,
)
from nvalchemi.training.distillation.labeling import _AUTO_PROBE_CHUNKS
from test.training.conftest import _build_atomic_data
from test.training.distillation.conftest import (
    _WIRED_CHARGE,
    _build_atom_only_dataset,
    _build_direct_force_teacher,
    _build_periodic_dataset,
    _build_small_dataset,
    _ChargeSourceModel,
    _DirectForceTeacher,
)

_SIGNALS = ["energy", "forces", "atomic_energies", "embeddings"]
"""Signal set exercised by the labeling tests."""

_TEACHER_FIELDS = (
    "teacher_energy",
    "teacher_forces",
    "teacher_atomic_energies",
    "teacher_node_embeddings",
)
"""Batch fields the ``_SIGNALS`` scorer writes into the store."""

_SPARSE_FIELDS = {"neighbor_list", "neighbor_list_shifts"}
"""Edge-level neighbor fields a periodic ``COO`` source carries."""

_SOURCE_FIELDS = {
    "atom_categories",
    "atomic_masses",
    "atomic_numbers",
    "energy",
    "forces",
    "positions",
    "velocities",
}
"""Fields the small source dataset contributes to a labeled store."""


def _make_scorer(teacher: _DirectForceTeacher) -> InProcessTeacherScorer:
    """Return a scorer over every signal the direct-force demo teacher supports."""
    return InProcessTeacherScorer(teacher, _SIGNALS)


def _read_all(store: Path) -> Batch:
    """Return every stored sample as a single CPU batch, with reader field levels.

    ``Dataset(device=None)`` auto-selects CUDA when available, so the device is
    pinned to keep comparisons against CPU scorer outputs hardware-independent.
    """
    reader = AtomicDataZarrReader(store)
    dataset = Dataset(reader=reader, device="cpu")
    return dataset.load_batches([list(range(len(dataset)))])[0]


def _label_prefix(
    dataset: InMemoryDataset,
    scorer: InProcessTeacherScorer,
    store: Path,
    count: int = 3,
) -> None:
    """Label the first *count* samples of *dataset* into *store*."""
    prefix = InMemoryDataset(
        in_memory_batch=dataset.in_memory_batch.index_select(list(range(count)))
    )
    label_dataset(prefix, scorer, store, batch_size=count)


def _make_neighbor_dataset(periodic: bool = False) -> InMemoryDataset:
    """Return a dataset whose samples carry a sparse neighbor list of varying size.

    A *periodic* source also carries ``neighbor_list_shifts``, the second
    edge-level tensor a ``COO`` build writes for a cell with boundaries.
    """
    source = _build_periodic_dataset() if periodic else _build_small_dataset()
    batch = source.in_memory_batch
    compute_neighbors(batch, cutoff=6.0, format=NeighborListFormat.COO)
    batch.keys["edge"].update(_SPARSE_FIELDS if periodic else {"neighbor_list"})
    return InMemoryDataset(in_memory_batch=batch)


def _make_custom_level_dataset(n_systems: int = 4) -> InMemoryDataset:
    """Return the small dataset's samples carrying ``site_weight`` at a custom level.

    Sample *i* has ``i + 1`` sites, so the segmented level ``sites`` follows a
    pointer of its own rather than the atom pointer.
    """
    schema = LevelSchema()
    schema.add_level("sites", segmented=True)
    batch = Batch.from_data_list(
        [
            _build_atomic_data(n_atoms=2 + index, seed=200 + index)
            for index in range(n_systems)
        ],
        attr_map=schema,
    )
    batch.add_key(
        "site_weight",
        [torch.full((index + 1, 1), float(index + 1)) for index in range(n_systems)],
        level="sites",
    )
    return InMemoryDataset(in_memory_batch=batch)


def _make_zarr_dataset(source: InMemoryDataset, root: Path) -> Dataset:
    """Return *source* written to a Zarr store under *root* and read back as a dataset.

    A Zarr-backed dataset reads ahead on a worker thread, so it exercises the
    asynchronous prefetch path an in-memory dataset satisfies synchronously.
    """
    path = root / "source.zarr"
    AtomicDataZarrWriter(path).write(source.in_memory_batch)
    return Dataset(reader=AtomicDataZarrReader(path), device="cpu")


def _make_uniform_dataset(n_systems: int = 10, n_atoms: int = 4) -> InMemoryDataset:
    """Return *n_systems* samples of *n_atoms* atoms each, so chunks cost the same."""
    return InMemoryDataset(
        in_memory_batch=Batch.from_data_list(
            [
                _build_atomic_data(n_atoms=n_atoms, seed=500 + index)
                for index in range(n_systems)
            ]
        )
    )


def _make_slow_loader(dataset: InMemoryDataset, delay: float) -> Any:
    """Return ``dataset.load_batches`` slowed by *delay* seconds per call."""
    load_batches = dataset.load_batches

    def slow_load_batches(*args: Any, **kwargs: Any) -> list[Batch]:
        time.sleep(delay)
        return load_batches(*args, **kwargs)

    return slow_load_batches


class _SequentialOnlyDataset:
    """Dataset exposing ``load_batches`` but none of the fused-prefetch surface."""

    def __init__(self, source: InMemoryDataset) -> None:
        self.source = source

    def __len__(self) -> int:
        """Return the wrapped dataset's sample count."""
        return len(self.source)

    def load_batches(
        self,
        batch_index_lists: Sequence[Sequence[int]],
        stream: torch.cuda.Stream | None = None,  # noqa: ARG002
    ) -> list[Batch]:
        """Load through the wrapped dataset."""
        return self.source.load_batches(batch_index_lists)


class _DelayedScorer:
    """Scorer that sleeps *delay* seconds before delegating to *inner*."""

    def __init__(self, inner: InProcessTeacherScorer, delay: float) -> None:
        self.inner = inner
        self.delay = delay
        self.label_fields = inner.label_fields

    def label(self, batch: Batch) -> TeacherLabels:
        """Return the inner scorer's labels after the configured delay."""
        time.sleep(self.delay)
        return self.inner.label(batch)


class _EmptyDataset:
    """Zero-length stand-in for a :class:`BatchDatasetProtocol` dataset."""

    def __len__(self) -> int:
        """Return zero samples."""
        return 0

    def load_batches(
        self,
        batch_index_lists: Sequence[Sequence[int]],  # noqa: ARG002
        stream: torch.cuda.Stream | None = None,  # noqa: ARG002
    ) -> list[Batch]:
        """Fail loudly, since a zero-length dataset must never be read."""
        raise AssertionError("load_batches must not be called for an empty dataset")


class _ForeignLabelScorer:
    """Scorer whose returned labels land outside the teacher namespace."""

    signals = frozenset({"reference_energy"})

    def label(self, batch: Batch) -> TeacherLabels:
        """Return a label keyed on the batch's own ``energy`` field."""
        return {"energy": (torch.zeros(batch.num_graphs, 1), "system")}


class _RowCountScorer:
    """Scorer whose label rows drift from the batch's after a healthy prefix."""

    signals = frozenset({"row_count"})

    def __init__(
        self,
        field: str,
        level: SignalLevel,
        offset: int,
        healthy_chunks: int = 0,
    ) -> None:
        self.label_fields = (field,)
        self.field = field
        self.level = level
        self.offset = offset
        self.healthy_chunks = healthy_chunks
        self.calls = 0

    def label(self, batch: Batch) -> TeacherLabels:
        """Return a label whose leading dimension is offset past the healthy chunks."""
        self.calls += 1
        rows = batch.num_nodes if self.level == "node" else batch.num_graphs
        if self.calls > self.healthy_chunks:
            rows += self.offset
        width = 3 if self.level == "node" else 1
        return {self.field: (torch.zeros(rows, width), self.level)}


class _ForeignFieldsScorer:
    """Scorer declaring a label field outside the teacher namespace."""

    signals = frozenset({"reference_energy"})
    label_fields = ("positions", "teacher_energy")

    def label(self, batch: Batch) -> TeacherLabels:  # noqa: ARG002
        """Fail loudly, since a foreign declaration must stop the run first."""
        raise AssertionError("label must not be called for a foreign declaration")


class TestLabelDataset:
    """Offline labeling of a dataset into a Zarr store."""

    def test_every_sample_is_labeled_and_stored(
        self,
        small_dataset: InMemoryDataset,
        direct_force_teacher: _DirectForceTeacher,
        tmp_path: Path,
    ) -> None:
        """Labeling returns the sample count and the store holds them all."""
        store = tmp_path / "labeled.zarr"
        labeled = label_dataset(
            small_dataset, _make_scorer(direct_force_teacher), store, batch_size=2
        )
        assert labeled == len(small_dataset)
        assert len(AtomicDataZarrReader(store)) == len(small_dataset)

    def test_teacher_fields_are_stored_at_the_expected_levels(
        self,
        small_dataset: InMemoryDataset,
        direct_force_teacher: _DirectForceTeacher,
        tmp_path: Path,
    ) -> None:
        """Node signals land at atom level and energy at system level."""
        store = tmp_path / "labeled.zarr"
        label_dataset(
            small_dataset, _make_scorer(direct_force_teacher), store, batch_size=2
        )
        levels = AtomicDataZarrReader(store).field_levels
        assert levels["teacher_energy"] == "system"
        assert levels["teacher_forces"] == "atom"
        assert levels["teacher_atomic_energies"] == "atom"
        assert levels["teacher_node_embeddings"] == "atom"

    def test_stored_values_match_a_direct_scorer_call(
        self,
        small_dataset: InMemoryDataset,
        direct_force_teacher: _DirectForceTeacher,
        tmp_path: Path,
    ) -> None:
        """Round-tripped teacher fields equal the scorer's own output."""
        store = tmp_path / "labeled.zarr"
        scorer = _make_scorer(direct_force_teacher)
        label_dataset(small_dataset, scorer, store, batch_size=2)
        expected = scorer.label(
            small_dataset.load_batches([list(range(len(small_dataset)))])[0]
        )
        stored = _read_all(store)
        for field, (values, _) in expected.items():
            torch.testing.assert_close(stored[field], values)

    def test_stress_round_trips_as_a_system_level_matrix(
        self,
        periodic_dataset: InMemoryDataset,
        lj_teacher: LennardJonesModelWrapper,
        tmp_path: Path,
    ) -> None:
        """A ``(B, 3, 3)`` signal keeps its shape, level, and values through the store."""
        store = tmp_path / "stress.zarr"
        scorer = InProcessTeacherScorer(lj_teacher, ["energy", "stress"])
        label_dataset(periodic_dataset, scorer, store, batch_size=2)
        assert AtomicDataZarrReader(store).field_levels["teacher_stress"] == "system"
        expected = scorer.label(
            periodic_dataset.load_batches([list(range(len(periodic_dataset)))])[0]
        )["teacher_stress"][0]
        stored = _read_all(store)
        assert stored.teacher_stress.shape == (len(periodic_dataset), 3, 3)
        torch.testing.assert_close(stored.teacher_stress, expected)

    def test_atom_only_dataset_gains_a_system_level_field(
        self,
        atom_only_dataset: InMemoryDataset,
        direct_force_teacher: _DirectForceTeacher,
        tmp_path: Path,
    ) -> None:
        """A dataset with no system-level field still receives ``teacher_energy``."""
        assert "system" not in atom_only_dataset.in_memory_batch._storage.groups
        store = tmp_path / "atom-only.zarr"
        scorer = _make_scorer(direct_force_teacher)
        assert label_dataset(atom_only_dataset, scorer, store, batch_size=2) == len(
            atom_only_dataset
        )
        assert AtomicDataZarrReader(store).field_levels["teacher_energy"] == "system"
        expected = scorer.label(
            atom_only_dataset.load_batches([list(range(len(atom_only_dataset)))])[0]
        )["teacher_energy"][0]
        stored = _read_all(store)
        assert stored.teacher_energy.shape == (len(atom_only_dataset), 1)
        torch.testing.assert_close(stored.teacher_energy, expected)

    @pytest.mark.parametrize("keep_neighbors", [False, True], ids=["dropped", "kept"])
    def test_dense_neighbor_fields_are_not_persisted(
        self,
        small_dataset: InMemoryDataset,
        direct_force_teacher: _DirectForceTeacher,
        tmp_path: Path,
        keep_neighbors: bool,
    ) -> None:
        """Dense neighbor tensors are dropped whatever ``keep_neighbors`` asks for."""
        store = tmp_path / "labeled.zarr"
        batch = small_dataset.in_memory_batch
        batch._atoms_group["num_neighbors"] = torch.zeros(
            batch.num_nodes, dtype=torch.int32
        )
        batch.keys["node"].add("num_neighbors")
        label_dataset(
            small_dataset,
            _make_scorer(direct_force_teacher),
            store,
            batch_size=2,
            keep_neighbors=keep_neighbors,
        )
        assert "num_neighbors" not in AtomicDataZarrReader(store).field_levels

    def test_a_source_neighbor_list_is_dropped_by_default(
        self,
        direct_force_teacher: _DirectForceTeacher,
        tmp_path: Path,
    ) -> None:
        """A source list records no cutoff a consumer could check, so it is not stored."""
        dataset = _make_neighbor_dataset()
        store = tmp_path / "edges.zarr"
        label_dataset(dataset, _make_scorer(direct_force_teacher), store, batch_size=2)
        assert set(AtomicDataZarrReader(store).field_levels) == {
            *_SOURCE_FIELDS,
            *_TEACHER_FIELDS,
        }
        stored = _read_all(store)
        assert stored.num_edges_list == []
        assert "edges" not in stored._storage.groups

    def test_keep_neighbors_carries_the_source_list_over(
        self,
        direct_force_teacher: _DirectForceTeacher,
        tmp_path: Path,
    ) -> None:
        """``keep_neighbors=True`` stores the sparse list with consistent pointers."""
        dataset = _make_neighbor_dataset()
        expected_edges = list(dataset.in_memory_batch.num_edges_list)
        store = tmp_path / "edges.zarr"
        label_dataset(
            dataset,
            _make_scorer(direct_force_teacher),
            store,
            batch_size=2,
            keep_neighbors=True,
        )
        assert set(AtomicDataZarrReader(store).field_levels) == {
            "neighbor_list",
            *_SOURCE_FIELDS,
            *_TEACHER_FIELDS,
        }
        stored = _read_all(store)
        assert stored.num_edges_list == expected_edges

    def test_neighbor_list_shifts_are_dropped_with_the_list(
        self,
        direct_force_teacher: _DirectForceTeacher,
        tmp_path: Path,
    ) -> None:
        """A periodic source's shift tensor follows its list into or out of the store."""
        dataset = _make_neighbor_dataset(periodic=True)
        scorer = _make_scorer(direct_force_teacher)
        dropped = tmp_path / "dropped.zarr"
        label_dataset(dataset, scorer, dropped, batch_size=2)
        assert not _SPARSE_FIELDS & set(AtomicDataZarrReader(dropped).field_levels)
        kept = tmp_path / "kept.zarr"
        label_dataset(dataset, scorer, kept, batch_size=2, keep_neighbors=True)
        assert _SPARSE_FIELDS <= set(AtomicDataZarrReader(kept).field_levels)

    def test_labeling_moves_each_chunk_to_the_requested_device(
        self,
        small_dataset: InMemoryDataset,
        direct_force_teacher: _DirectForceTeacher,
        tmp_path: Path,
    ) -> None:
        """Every loaded chunk is moved to the requested device before scoring."""
        store = tmp_path / "labeled.zarr"
        with patch.object(Batch, "to", autospec=True, side_effect=Batch.to) as spy:
            label_dataset(
                small_dataset,
                _make_scorer(direct_force_teacher),
                store,
                batch_size=2,
                device="cpu",
            )
        assert spy.call_count == 3
        assert all(call.args[1] == "cpu" for call in spy.call_args_list)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_labeling_on_cuda_matches_the_cpu_store(
        self,
        small_dataset: InMemoryDataset,
        direct_force_teacher: _DirectForceTeacher,
        tmp_path: Path,
    ) -> None:
        """Labeling on CUDA stores the same values as labeling on CPU."""
        cpu_store = tmp_path / "cpu.zarr"
        label_dataset(
            small_dataset, _make_scorer(direct_force_teacher), cpu_store, batch_size=2
        )
        cuda_store = tmp_path / "cuda.zarr"
        label_dataset(
            small_dataset,
            _make_scorer(direct_force_teacher.to("cuda")),
            cuda_store,
            batch_size=2,
            device="cuda",
        )
        stored_cpu = _read_all(cpu_store)
        stored_cuda = _read_all(cuda_store)
        for field in _TEACHER_FIELDS:
            torch.testing.assert_close(stored_cuda[field], stored_cpu[field])

    def test_resume_on_a_complete_store_labels_nothing(
        self,
        small_dataset: InMemoryDataset,
        direct_force_teacher: _DirectForceTeacher,
        tmp_path: Path,
    ) -> None:
        """A second pass over a fully labeled store is a no-op that appends nothing."""
        store = tmp_path / "labeled.zarr"
        scorer = _make_scorer(direct_force_teacher)
        label_dataset(small_dataset, scorer, store, batch_size=2)
        assert label_dataset(small_dataset, scorer, store, batch_size=2) == 0
        assert len(AtomicDataZarrReader(store)) == len(small_dataset)

    def test_resume_on_a_store_longer_than_the_dataset_raises(
        self,
        small_dataset: InMemoryDataset,
        direct_force_teacher: _DirectForceTeacher,
        tmp_path: Path,
    ) -> None:
        """A store holding more samples than the dataset came from another dataset."""
        store = tmp_path / "labeled.zarr"
        scorer = _make_scorer(direct_force_teacher)
        label_dataset(small_dataset, scorer, store, batch_size=2)
        shorter = InMemoryDataset(
            in_memory_batch=small_dataset.in_memory_batch.index_select([0, 1, 2])
        )
        with pytest.raises(ValueError, match="holds 5 samples but the dataset has 3"):
            label_dataset(shorter, scorer, store, batch_size=2)
        assert len(AtomicDataZarrReader(store)) == len(small_dataset)

    def test_resume_continues_a_partial_store(
        self,
        small_dataset: InMemoryDataset,
        direct_force_teacher: _DirectForceTeacher,
        tmp_path: Path,
    ) -> None:
        """Resuming labels only the samples the store does not already hold."""
        store = tmp_path / "labeled.zarr"
        scorer = _make_scorer(direct_force_teacher)
        prefix = InMemoryDataset(
            in_memory_batch=small_dataset.in_memory_batch.index_select([0, 1])
        )
        label_dataset(prefix, scorer, store, batch_size=2)
        assert label_dataset(small_dataset, scorer, store, batch_size=2) == 3
        assert len(AtomicDataZarrReader(store)) == len(small_dataset)

    def test_resumed_store_matches_a_single_pass_store(
        self,
        small_dataset: InMemoryDataset,
        direct_force_teacher: _DirectForceTeacher,
        tmp_path: Path,
    ) -> None:
        """A resumed run reproduces the store a single uninterrupted run writes."""
        scorer = _make_scorer(direct_force_teacher)
        single = tmp_path / "single.zarr"
        label_dataset(small_dataset, scorer, single, batch_size=2)
        resumed = tmp_path / "resumed.zarr"
        prefix = InMemoryDataset(
            in_memory_batch=small_dataset.in_memory_batch.index_select([0, 1])
        )
        label_dataset(prefix, scorer, resumed, batch_size=2)
        label_dataset(small_dataset, scorer, resumed, batch_size=2)
        expected = _read_all(single)
        actual = _read_all(resumed)
        assert actual.num_nodes_list == expected.num_nodes_list
        for field, values in expected:
            torch.testing.assert_close(actual[field], values)

    def test_resume_with_a_different_signal_set_raises(
        self,
        small_dataset: InMemoryDataset,
        direct_force_teacher: _DirectForceTeacher,
        tmp_path: Path,
    ) -> None:
        """A resumed run that would write a different field set is refused."""
        store = tmp_path / "labeled.zarr"
        prefix = InMemoryDataset(
            in_memory_batch=small_dataset.in_memory_batch.index_select([0, 1])
        )
        label_dataset(prefix, _make_scorer(direct_force_teacher), store, batch_size=2)
        narrowed = InProcessTeacherScorer(direct_force_teacher, ["energy"])
        with pytest.raises(ValueError, match="teacher_forces"):
            label_dataset(small_dataset, narrowed, store, batch_size=2)

    def test_resume_false_on_an_existing_store_raises(
        self,
        small_dataset: InMemoryDataset,
        direct_force_teacher: _DirectForceTeacher,
        tmp_path: Path,
    ) -> None:
        """Overwriting an existing store is refused rather than silently appended."""
        store = tmp_path / "labeled.zarr"
        scorer = _make_scorer(direct_force_teacher)
        label_dataset(small_dataset, scorer, store, batch_size=2)
        with pytest.raises(ValueError, match="resume"):
            label_dataset(small_dataset, scorer, store, resume=False)

    def test_resume_false_on_an_emptied_store_raises(
        self,
        small_dataset: InMemoryDataset,
        direct_force_teacher: _DirectForceTeacher,
        tmp_path: Path,
    ) -> None:
        """A store whose samples were all deleted still counts as existing."""
        store = tmp_path / "labeled.zarr"
        scorer = _make_scorer(direct_force_teacher)
        label_dataset(small_dataset, scorer, store, batch_size=2)
        AtomicDataZarrWriter(store).delete(list(range(len(small_dataset))))
        assert len(AtomicDataZarrReader(store)) == 0
        with pytest.raises(ValueError, match="resume"):
            label_dataset(small_dataset, scorer, store, resume=False)

    def test_resume_on_a_store_with_deleted_samples_raises(
        self,
        small_dataset: InMemoryDataset,
        direct_force_teacher: _DirectForceTeacher,
        tmp_path: Path,
    ) -> None:
        """Soft-deleted samples break index alignment, so resuming is refused."""
        store = tmp_path / "labeled.zarr"
        scorer = _make_scorer(direct_force_teacher)
        prefix = InMemoryDataset(
            in_memory_batch=small_dataset.in_memory_batch.index_select([0, 1, 2])
        )
        label_dataset(prefix, scorer, store, batch_size=2)
        AtomicDataZarrWriter(store).delete([1])
        with pytest.raises(ValueError, match="soft-deleted"):
            label_dataset(small_dataset, scorer, store, batch_size=2)

    def test_unreadable_store_path_raises(
        self,
        small_dataset: InMemoryDataset,
        direct_force_teacher: _DirectForceTeacher,
        tmp_path: Path,
    ) -> None:
        """A path that exists but is not a Zarr store is reported clearly."""
        store = tmp_path / "not-a-store.zarr"
        store.mkdir()
        with pytest.raises(ValueError, match="not a readable"):
            label_dataset(small_dataset, _make_scorer(direct_force_teacher), store)

    def test_empty_dataset_labels_nothing_and_writes_no_store(
        self,
        direct_force_teacher: _DirectForceTeacher,
        tmp_path: Path,
    ) -> None:
        """A zero-length dataset is a no-op that leaves no store behind."""
        store = tmp_path / "labeled.zarr"
        labeled = label_dataset(
            _EmptyDataset(), _make_scorer(direct_force_teacher), store
        )
        assert labeled == 0
        assert not store.exists()

    def test_non_positive_batch_size_raises(
        self,
        small_dataset: InMemoryDataset,
        direct_force_teacher: _DirectForceTeacher,
        tmp_path: Path,
    ) -> None:
        """A batch size of zero is rejected before any store is created."""
        with pytest.raises(ValueError, match="batch_size"):
            label_dataset(
                small_dataset,
                _make_scorer(direct_force_teacher),
                tmp_path / "labeled.zarr",
                batch_size=0,
            )


class TestLabelDatasetCustomSignals:
    """Labeling with a signal the built-in table does not cover."""

    def test_a_custom_signal_lands_in_the_store_at_its_level(
        self, small_dataset: InMemoryDataset, tmp_path: Path
    ) -> None:
        """``teacher_charges`` is stored as an atom-level field with the teacher's values."""
        charges = TeacherSignal("charges", "charges", "teacher_charges", "node")
        scorer = InProcessTeacherScorer(_ChargeSourceModel(), ["energy", charges])
        store = tmp_path / "labeled.zarr"
        assert label_dataset(small_dataset, scorer, store, batch_size=2) == 5
        reader = AtomicDataZarrReader(store)
        assert reader.field_levels["teacher_charges"] == "atom"
        assert reader.schema()["teacher_charges"].row_shape == ()
        stored = _read_all(store)
        assert stored["teacher_charges"].shape == (stored.num_nodes,)
        assert torch.all(stored["teacher_charges"] == _WIRED_CHARGE)


class TestLabelDatasetStoreIntegrity:
    """Resuming a store an interrupted labeling run left inconsistent."""

    def test_resume_after_an_interrupted_append_raises(
        self,
        small_dataset: InMemoryDataset,
        direct_force_teacher: _DirectForceTeacher,
        tmp_path: Path,
    ) -> None:
        """Masks extended past the committed sample count are refused, not resumed."""
        store = tmp_path / "labeled.zarr"
        scorer = _make_scorer(direct_force_teacher)
        _label_prefix(small_dataset, scorer, store)
        root = zarr.open(store, mode="r+")
        atoms_ptr = root["meta"]["atoms_ptr"]
        stored_atoms = int(atoms_ptr[-1])
        atoms_ptr.resize((atoms_ptr.shape[0] + 2,))
        atoms_ptr[-2:] = [stored_atoms + 5, stored_atoms + 11]
        root["meta"]["samples_mask"].resize((5,))
        with pytest.raises(ValueError, match="meta/samples_mask holds"):
            label_dataset(small_dataset, scorer, store, batch_size=2)

    def test_resume_with_a_short_field_array_raises(
        self,
        small_dataset: InMemoryDataset,
        direct_force_teacher: _DirectForceTeacher,
        tmp_path: Path,
    ) -> None:
        """A field array shorter than its level total is refused, not resumed."""
        store = tmp_path / "labeled.zarr"
        scorer = _make_scorer(direct_force_teacher)
        _label_prefix(small_dataset, scorer, store)
        positions = zarr.open(store, mode="r+")["core"]["positions"]
        positions.resize((positions.shape[0] - 4, 3))
        with pytest.raises(ValueError, match="positions holds"):
            label_dataset(small_dataset, scorer, store, batch_size=2)

    def test_resume_with_a_non_monotonic_pointer_array_raises(
        self,
        small_dataset: InMemoryDataset,
        direct_force_teacher: _DirectForceTeacher,
        tmp_path: Path,
    ) -> None:
        """Pointers left out of order by a torn write are refused, not resumed."""
        store = tmp_path / "labeled.zarr"
        scorer = _make_scorer(direct_force_teacher)
        _label_prefix(small_dataset, scorer, store)
        zarr.open(store, mode="r+")["meta"]["atoms_ptr"][2] = 0
        with pytest.raises(ValueError, match="non-decreasing"):
            label_dataset(small_dataset, scorer, store, batch_size=2)


class TestLabelDatasetCustomLevels:
    """Stores whose source fields live at a user-registered level."""

    def test_resume_continues_a_store_with_a_custom_level_field(
        self, direct_force_teacher: _DirectForceTeacher, tmp_path: Path
    ) -> None:
        """A healthy store carrying a custom-level field resumes like any other."""
        dataset = _make_custom_level_dataset()
        store = tmp_path / "labeled.zarr"
        scorer = _make_scorer(direct_force_teacher)
        _label_prefix(dataset, scorer, store, count=2)
        assert label_dataset(dataset, scorer, store, batch_size=2) == 2
        reader = AtomicDataZarrReader(store)
        assert reader.field_levels["site_weight"] == "sites"
        assert reader.level_sizes()["sites"] == 10
        stored = _read_all(store)
        torch.testing.assert_close(
            stored["site_weight"], dataset.in_memory_batch["site_weight"]
        )

    def test_resume_missing_the_stored_custom_level_field_raises(
        self, direct_force_teacher: _DirectForceTeacher, tmp_path: Path
    ) -> None:
        """A chunk without the store's custom-level field is refused as drift."""
        store = tmp_path / "labeled.zarr"
        scorer = _make_scorer(direct_force_teacher)
        _label_prefix(_make_custom_level_dataset(), scorer, store, count=2)
        with pytest.raises(ValueError, match=r"is missing \['site_weight'\]"):
            label_dataset(
                _build_small_dataset(n_systems=4), scorer, store, batch_size=2
            )
        assert len(AtomicDataZarrReader(store)) == 2

    def test_resume_adding_a_custom_level_field_raises(
        self, direct_force_teacher: _DirectForceTeacher, tmp_path: Path
    ) -> None:
        """A chunk carrying a custom-level field the store lacks is refused as drift."""
        store = tmp_path / "labeled.zarr"
        scorer = _make_scorer(direct_force_teacher)
        _label_prefix(_build_small_dataset(n_systems=4), scorer, store, count=2)
        with pytest.raises(ValueError, match=r"writes extra \['site_weight'\]"):
            label_dataset(_make_custom_level_dataset(), scorer, store, batch_size=2)
        assert len(AtomicDataZarrReader(store)) == 2


class TestLabelDatasetChunkSchema:
    """Per-chunk field, level, and dtype agreement with the store schema."""

    def test_field_drift_between_chunks_of_a_fresh_run_raises(
        self,
        direct_force_teacher: _DirectForceTeacher,
        tmp_path: Path,
    ) -> None:
        """A heterogeneous source whose chunks differ is refused mid-run."""
        dataset = MultiDataset(_build_small_dataset(), _build_atom_only_dataset())
        store = tmp_path / "drift.zarr"
        with pytest.raises(ValueError, match="covering samples 4-5"):
            label_dataset(
                dataset, _make_scorer(direct_force_teacher), store, batch_size=2
            )
        assert len(AtomicDataZarrReader(store)) == 4

    def test_unstorable_label_dtype_is_refused_before_the_first_write(
        self,
        small_dataset: InMemoryDataset,
        direct_force_teacher: _DirectForceTeacher,
        tmp_path: Path,
    ) -> None:
        """Labels in a dtype no store can hold leave no store behind."""
        store = tmp_path / "labeled.zarr"
        scorer = InProcessTeacherScorer(
            direct_force_teacher, _SIGNALS, dtype=torch.bfloat16
        )
        with pytest.raises(ValueError, match="bfloat16"):
            label_dataset(small_dataset, scorer, store, batch_size=2)
        assert not store.exists()

    def test_dtype_drift_on_resume_raises(
        self,
        small_dataset: InMemoryDataset,
        direct_force_teacher: _DirectForceTeacher,
        tmp_path: Path,
    ) -> None:
        """A resume that would cast labels into the stored precision is refused."""
        store = tmp_path / "labeled.zarr"
        half = InProcessTeacherScorer(
            direct_force_teacher, _SIGNALS, dtype=torch.float16
        )
        _label_prefix(small_dataset, half, store, count=2)
        with pytest.raises(ValueError, match="torch.float16"):
            label_dataset(
                small_dataset,
                _make_scorer(direct_force_teacher),
                store,
                batch_size=2,
            )

    def test_shape_drift_on_resume_raises_and_leaves_the_store_unchanged(
        self,
        small_dataset: InMemoryDataset,
        direct_force_teacher: _DirectForceTeacher,
        tmp_path: Path,
    ) -> None:
        """A resume whose embeddings are wider than the stored ones is refused."""
        store = tmp_path / "labeled.zarr"
        _label_prefix(small_dataset, _make_scorer(direct_force_teacher), store, count=2)
        before = _read_all(store)
        wider = _make_scorer(_build_direct_force_teacher(hidden_dim=16))
        with pytest.raises(ValueError, match=r"\(8,\)\) but arrives as .*\(16,\)"):
            label_dataset(small_dataset, wider, store, batch_size=2)
        after = _read_all(store)
        assert len(AtomicDataZarrReader(store)) == 2
        assert after["teacher_node_embeddings"].shape == (before.num_nodes, 8)
        torch.testing.assert_close(
            after["teacher_node_embeddings"], before["teacher_node_embeddings"]
        )


class TestLabelDatasetFieldNamespace:
    """Teacher labels are held to the ``teacher_*`` namespace."""

    def test_a_scorer_returning_a_foreign_field_raises_before_any_write(
        self,
        small_dataset: InMemoryDataset,
        tmp_path: Path,
    ) -> None:
        """A label that would overwrite ``energy`` leaves no store behind."""
        store = tmp_path / "foreign.zarr"
        with pytest.raises(ValueError, match="Teacher labels must populate"):
            label_dataset(small_dataset, _ForeignLabelScorer(), store, batch_size=2)
        assert not store.exists()

    def test_a_scorer_declaring_a_foreign_field_raises_before_the_first_chunk(
        self,
        small_dataset: InMemoryDataset,
        tmp_path: Path,
    ) -> None:
        """A foreign ``label_fields`` is refused without the scorer being called."""
        store = tmp_path / "declared.zarr"
        with pytest.raises(ValueError, match=r"label_fields must populate"):
            label_dataset(small_dataset, _ForeignFieldsScorer(), store, batch_size=2)
        assert not store.exists()


class TestLabelDatasetLabelRowCounts:
    """Teacher labels are held to one row per atom or per graph."""

    def test_a_system_label_with_a_surplus_row_is_refused_before_any_write(
        self,
        small_dataset: InMemoryDataset,
        tmp_path: Path,
    ) -> None:
        """A system label one row too long names its field, level, and shape."""
        store = tmp_path / "wide_system.zarr"
        scorer = _RowCountScorer("teacher_energy", "system", offset=1)
        with pytest.raises(
            ValueError,
            match=r"'teacher_energy' at level 'system' has shape \(3, 1\); "
            r"expected 2 rows, one per graph",
        ):
            label_dataset(small_dataset, scorer, store, batch_size=2)
        assert not store.exists()

    def test_a_node_label_with_a_surplus_row_is_refused_before_any_write(
        self,
        small_dataset: InMemoryDataset,
        tmp_path: Path,
    ) -> None:
        """A node label one row too long is named rather than left to torch.split."""
        store = tmp_path / "wide_node.zarr"
        scorer = _RowCountScorer("teacher_forces", "node", offset=1)
        with pytest.raises(
            ValueError,
            match=r"'teacher_forces' at level 'node' has shape \(6, 3\); "
            r"expected 5 rows, one per atom",
        ):
            label_dataset(small_dataset, scorer, store, batch_size=2)
        assert not store.exists()

    def test_a_label_short_of_a_row_is_refused(
        self,
        small_dataset: InMemoryDataset,
        tmp_path: Path,
    ) -> None:
        """A label one row too short is refused on the same terms as a long one."""
        store = tmp_path / "short.zarr"
        scorer = _RowCountScorer("teacher_energy", "system", offset=-1)
        with pytest.raises(ValueError, match="expected 2 rows, one per graph"):
            label_dataset(small_dataset, scorer, store, batch_size=2)
        assert not store.exists()

    def test_a_surplus_label_leaves_the_chunks_already_written_untouched(
        self,
        small_dataset: InMemoryDataset,
        tmp_path: Path,
    ) -> None:
        """A row count that drifts mid-run stops the store at its last good chunk."""
        store = tmp_path / "mid_run.zarr"
        scorer = _RowCountScorer("teacher_energy", "system", offset=1, healthy_chunks=1)
        with pytest.raises(ValueError, match="expected 2 rows, one per graph"):
            label_dataset(small_dataset, scorer, store, batch_size=2)
        stored = _read_all(store)
        assert len(AtomicDataZarrReader(store)) == 2
        assert stored["teacher_energy"].shape == (2, 1)


class TestLabelDatasetPrefetch:
    """Reading one chunk ahead of the scoring and writing of the previous one."""

    @pytest.mark.parametrize("prefetch", [True, "auto"], ids=["pipelined", "auto"])
    def test_store_matches_the_sequential_store(
        self,
        small_dataset: InMemoryDataset,
        direct_force_teacher: _DirectForceTeacher,
        tmp_path: Path,
        prefetch: bool | str,
    ) -> None:
        """Every mode writes the same sample count, schema, and values."""
        dataset = _make_zarr_dataset(small_dataset, tmp_path)
        scorer = _make_scorer(direct_force_teacher)
        sequential = tmp_path / "sequential.zarr"
        label_dataset(dataset, scorer, sequential, batch_size=2, prefetch=False)
        store = tmp_path / "labeled.zarr"
        assert label_dataset(
            dataset, scorer, store, batch_size=2, prefetch=prefetch
        ) == len(dataset)
        reference, reader = (
            AtomicDataZarrReader(sequential),
            AtomicDataZarrReader(store),
        )
        assert len(reader) == len(reference)
        assert reader.field_levels == reference.field_levels
        expected, actual = _read_all(sequential), _read_all(store)
        assert actual.num_nodes_list == expected.num_nodes_list
        for field, values in expected:
            torch.testing.assert_close(actual[field], values)

    def test_prefetch_true_reads_one_chunk_ahead(
        self,
        small_dataset: InMemoryDataset,
        direct_force_teacher: _DirectForceTeacher,
        tmp_path: Path,
    ) -> None:
        """Each chunk but the last submits the next one and nothing is left pending."""
        dataset = _make_zarr_dataset(small_dataset, tmp_path)
        store = tmp_path / "labeled.zarr"
        with patch.object(
            dataset, "prefetch_fused_batches", wraps=dataset.prefetch_fused_batches
        ) as spy:
            label_dataset(
                dataset,
                _make_scorer(direct_force_teacher),
                store,
                batch_size=2,
                prefetch=True,
            )
        assert [call.args[0] for call in spy.call_args_list] == [[[2, 3]], [[4]]]
        assert not dataset.has_pending_fused_batches()

    def test_prefetch_true_without_the_surface_warns_and_labels_sequentially(
        self,
        small_dataset: InMemoryDataset,
        direct_force_teacher: _DirectForceTeacher,
        tmp_path: Path,
    ) -> None:
        """A dataset offering only ``load_batches`` is labeled with a warning."""
        store = tmp_path / "labeled.zarr"
        with pytest.warns(UserWarning, match="no fused-prefetch surface"):
            labeled = label_dataset(
                _SequentialOnlyDataset(small_dataset),
                _make_scorer(direct_force_teacher),
                store,
                batch_size=2,
                prefetch=True,
            )
        assert labeled == len(small_dataset)
        assert len(AtomicDataZarrReader(store)) == len(small_dataset)

    def test_prefetch_auto_without_the_surface_is_silent(
        self,
        small_dataset: InMemoryDataset,
        direct_force_teacher: _DirectForceTeacher,
        tmp_path: Path,
    ) -> None:
        """The default mode stays sequential without a warning on such a dataset."""
        store = tmp_path / "labeled.zarr"
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            labeled = label_dataset(
                _SequentialOnlyDataset(small_dataset),
                _make_scorer(direct_force_teacher),
                store,
                batch_size=2,
            )
        assert labeled == len(small_dataset)

    def test_invalid_prefetch_value_raises(
        self,
        small_dataset: InMemoryDataset,
        direct_force_teacher: _DirectForceTeacher,
        tmp_path: Path,
    ) -> None:
        """A prefetch value outside ``True``, ``False``, ``"auto"`` is refused."""
        with pytest.raises(ValueError, match="prefetch must be True, False, or 'auto'"):
            label_dataset(
                small_dataset,
                _make_scorer(direct_force_teacher),
                tmp_path / "labeled.zarr",
                prefetch="always",  # type: ignore[arg-type]
            )

    @pytest.mark.parametrize(
        "prefetch", [False, True, "auto"], ids=["sequential", "pipelined", "auto"]
    )
    def test_resume_matches_a_single_pass_store_in_every_mode(
        self,
        small_dataset: InMemoryDataset,
        direct_force_teacher: _DirectForceTeacher,
        tmp_path: Path,
        prefetch: bool | str,
    ) -> None:
        """A resumed run in any mode reproduces the single-pass store."""
        dataset = _make_zarr_dataset(small_dataset, tmp_path)
        scorer = _make_scorer(direct_force_teacher)
        single = tmp_path / "single.zarr"
        label_dataset(dataset, scorer, single, batch_size=2, prefetch=False)
        resumed = tmp_path / "resumed.zarr"
        _label_prefix(small_dataset, scorer, resumed, count=2)
        assert (
            label_dataset(dataset, scorer, resumed, batch_size=2, prefetch=prefetch)
            == 3
        )
        expected, actual = _read_all(single), _read_all(resumed)
        assert actual.num_nodes_list == expected.num_nodes_list
        for field, values in expected:
            torch.testing.assert_close(actual[field], values)

    def test_auto_pipelines_when_loading_dominates(
        self,
        direct_force_teacher: _DirectForceTeacher,
        tmp_path: Path,
    ) -> None:
        """A slow loader makes ``"auto"`` read ahead after the probe chunks."""
        dataset = _make_uniform_dataset()
        slow = _make_slow_loader(dataset, delay=0.15)
        with (
            patch.object(dataset, "load_batches", side_effect=slow),
            patch.object(
                dataset, "prefetch_fused_batches", wraps=dataset.prefetch_fused_batches
            ) as spy,
        ):
            label_dataset(
                dataset,
                _make_scorer(direct_force_teacher),
                tmp_path / "labeled.zarr",
                batch_size=2,
            )
        assert spy.call_count == 5 - _AUTO_PROBE_CHUNKS - 1

    def test_auto_falls_back_when_reading_ahead_is_not_faster(
        self,
        direct_force_teacher: _DirectForceTeacher,
        tmp_path: Path,
    ) -> None:
        """Reads ahead that take longer than the sequential probe stop after one chunk."""
        dataset = _make_uniform_dataset()
        slow = _make_slow_loader(dataset, delay=0.15)
        slower = _make_slow_loader(dataset, delay=0.3)
        prefetch = dataset.prefetch_fused_batches

        def slow_prefetch(*args: Any, **kwargs: Any) -> None:
            slower(*args, **kwargs)
            prefetch(*args, **kwargs)

        with (
            patch.object(dataset, "load_batches", side_effect=slow),
            patch.object(
                dataset, "prefetch_fused_batches", side_effect=slow_prefetch
            ) as spy,
        ):
            label_dataset(
                dataset,
                _make_scorer(direct_force_teacher),
                tmp_path / "labeled.zarr",
                batch_size=2,
            )
        assert spy.call_count == 2
        assert not dataset.has_pending_fused_batches()

    def test_auto_stays_sequential_when_scoring_dominates(
        self,
        direct_force_teacher: _DirectForceTeacher,
        tmp_path: Path,
    ) -> None:
        """A slow scorer over a fast dataset keeps ``"auto"`` on the sequential loop."""
        dataset = _make_uniform_dataset()
        scorer = _DelayedScorer(_make_scorer(direct_force_teacher), delay=0.15)
        with patch.object(
            dataset, "prefetch_fused_batches", wraps=dataset.prefetch_fused_batches
        ) as spy:
            label_dataset(dataset, scorer, tmp_path / "labeled.zarr", batch_size=2)
        assert spy.call_count == 0

    def test_auto_skips_zero_atom_probe_chunks(self, tmp_path: Path) -> None:
        """Zero-atom probe chunks defer the decision and the store still matches."""
        data = [
            AtomicData(
                positions=torch.zeros(n_atoms, 3),
                atomic_numbers=torch.ones(n_atoms, dtype=torch.long),
            )
            for n_atoms in [0 if index // 2 in (1, 3) else 4 for index in range(10)]
        ]
        dataset = InMemoryDataset(in_memory_batch=Batch.from_data_list(data))
        scorer = _RowCountScorer("teacher_energy", "system", offset=0)
        sequential = tmp_path / "sequential.zarr"
        label_dataset(dataset, scorer, sequential, batch_size=2, prefetch=False)
        store = tmp_path / "auto.zarr"
        assert label_dataset(dataset, scorer, store, batch_size=2) == 10
        expected, actual = _read_all(sequential), _read_all(store)
        assert actual.num_nodes_list == expected.num_nodes_list
        for field, values in expected:
            torch.testing.assert_close(actual[field], values)

    def test_a_failing_chunk_leaves_no_read_pending(
        self,
        small_dataset: InMemoryDataset,
        tmp_path: Path,
    ) -> None:
        """The read submitted ahead of a chunk that fails is cancelled."""
        dataset = _make_zarr_dataset(small_dataset, tmp_path)
        scorer = _RowCountScorer("teacher_energy", "system", offset=1, healthy_chunks=1)
        with pytest.raises(ValueError, match="expected 2 rows, one per graph"):
            label_dataset(
                dataset, scorer, tmp_path / "labeled.zarr", batch_size=2, prefetch=True
            )
        assert not dataset.has_pending_fused_batches()
