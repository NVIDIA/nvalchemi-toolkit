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
"""Tests for nvalchemi.data.datapipes.collection."""

from __future__ import annotations

import pytest
import torch

from nvalchemi.data.batch import Batch
from nvalchemi.data.datapipes.backends.base import Reader
from nvalchemi.data.datapipes.collection import (
    ConcatReader,
    DataCollection,
    RandomSplit,
    Split,
    SplitStrategy,
    SubsetReader,
)


class _InMemoryReader(Reader):
    """Trivial in-memory reader for unit tests."""

    def __init__(self, n: int, seed: int = 0) -> None:
        super().__init__()
        gen = torch.Generator().manual_seed(seed)
        self._samples: list[dict[str, torch.Tensor]] = []
        for i in range(n):
            num_atoms = (i % 4) + 2  # 2..5
            num_edges = (i % 3) + 1  # 1..3
            self._samples.append(
                {
                    "atomic_numbers": torch.randint(1, 20, (num_atoms,), generator=gen),
                    "positions": torch.randn(num_atoms, 3, generator=gen),
                    "forces": torch.randn(num_atoms, 3, generator=gen),
                    "energies": torch.randn(1, 1, generator=gen),
                    "cell": torch.eye(3).unsqueeze(0),
                    "pbc": torch.tensor([[True, True, True]]),
                    "edge_index": torch.stack(
                        [
                            torch.randint(0, num_atoms, (num_edges,), generator=gen),
                            torch.randint(0, num_atoms, (num_edges,), generator=gen),
                        ]
                    ),
                    "shifts": torch.randn(num_edges, 3, generator=gen),
                }
            )

    def _load_sample(self, index: int) -> dict[str, torch.Tensor]:
        return self._samples[index]

    def __len__(self) -> int:
        return len(self._samples)


# ---------------------------------------------------------------------------
# Split enum
# ---------------------------------------------------------------------------


class TestSplitEnum:
    """Tests for the Split enum."""

    def test_values(self) -> None:
        """Enum values are the expected strings."""
        assert Split.TRAIN.value == "train"
        assert Split.VAL.value == "val"
        assert Split.TEST.value == "test"

    def test_str_subclass(self) -> None:
        """Split members are str instances (usable as dict keys)."""
        assert isinstance(Split.TRAIN, str)

    def test_membership(self) -> None:
        """Only three members exist."""
        assert len(Split) == 3


# ---------------------------------------------------------------------------
# SubsetReader
# ---------------------------------------------------------------------------


class TestSubsetReader:
    """Tests for SubsetReader."""

    def test_length(self) -> None:
        """Subset length equals number of indices."""
        reader = _InMemoryReader(10)
        subset = SubsetReader(reader, [0, 3, 7])
        assert len(subset) == 3

    def test_remapping(self) -> None:
        """Items are correctly remapped through the index list."""
        reader = _InMemoryReader(10)
        indices = [4, 2, 8]
        subset = SubsetReader(reader, indices)

        for i, parent_idx in enumerate(indices):
            sub_data, _ = subset[i]
            parent_data, _ = reader[parent_idx]
            for key in parent_data:
                assert torch.equal(sub_data[key], parent_data[key])

    def test_metadata_passthrough(self) -> None:
        """Metadata index reflects the remapped (subset-local) position."""
        reader = _InMemoryReader(10)
        subset = SubsetReader(reader, [5, 9])
        _, meta = subset[0]
        assert meta["index"] == 0

    def test_out_of_bounds_construction(self) -> None:
        """ValueError on indices outside parent range."""
        reader = _InMemoryReader(5)
        with pytest.raises(ValueError, match="out of range"):
            SubsetReader(reader, [0, 5])

    def test_negative_index_rejected(self) -> None:
        """Negative indices are rejected at construction."""
        reader = _InMemoryReader(5)
        with pytest.raises(ValueError, match="out of range"):
            SubsetReader(reader, [-1])

    def test_getitem_out_of_bounds(self) -> None:
        """IndexError when accessing beyond the subset length."""
        reader = _InMemoryReader(10)
        subset = SubsetReader(reader, [1, 2])
        with pytest.raises(IndexError):
            subset[2]


# ---------------------------------------------------------------------------
# RandomSplit
# ---------------------------------------------------------------------------


class TestRandomSplit:
    """Tests for the RandomSplit strategy."""

    def test_reproducible(self) -> None:
        """Same seed produces identical splits."""
        splitter = RandomSplit()
        a = splitter(100, {Split.TRAIN: 0.8, Split.VAL: 0.2}, seed=42)
        b = splitter(100, {Split.TRAIN: 0.8, Split.VAL: 0.2}, seed=42)
        assert a == b

    def test_different_seeds(self) -> None:
        """Different seeds produce different permutations."""
        splitter = RandomSplit()
        a = splitter(100, {Split.TRAIN: 0.8, Split.VAL: 0.2}, seed=0)
        b = splitter(100, {Split.TRAIN: 0.8, Split.VAL: 0.2}, seed=1)
        assert a[Split.TRAIN] != b[Split.TRAIN]

    def test_ratios_respected(self) -> None:
        """Output sizes match expected proportions."""
        splitter = RandomSplit()
        result = splitter(
            100,
            {Split.TRAIN: 0.7, Split.VAL: 0.2, Split.TEST: 0.1},
            seed=0,
        )
        assert len(result[Split.TRAIN]) == 70
        assert len(result[Split.VAL]) == 20
        assert len(result[Split.TEST]) == 10

    def test_disjoint_partitions(self) -> None:
        """Splits share no indices."""
        splitter = RandomSplit()
        result = splitter(50, {Split.TRAIN: 0.5, Split.VAL: 0.3}, seed=7)
        assert set(result[Split.TRAIN]).isdisjoint(set(result[Split.VAL]))

    def test_remainder_discarded(self) -> None:
        """Indices beyond ratio coverage are unused."""
        splitter = RandomSplit()
        result = splitter(100, {Split.TRAIN: 0.5}, seed=0)
        assert len(result[Split.TRAIN]) == 50
        all_indices = set(result[Split.TRAIN])
        assert len(all_indices) == 50  # no duplicates

    def test_ratios_exceed_one_raises(self) -> None:
        """ValueError when ratios sum above 1.0."""
        splitter = RandomSplit()
        with pytest.raises(ValueError, match="exceeds 1.0"):
            splitter(10, {Split.TRAIN: 0.6, Split.VAL: 0.5}, seed=0)

    def test_protocol_conformance(self) -> None:
        """RandomSplit satisfies SplitStrategy protocol."""
        assert isinstance(RandomSplit(), SplitStrategy)


# ---------------------------------------------------------------------------
# DataCollection — single reader with splits
# ---------------------------------------------------------------------------


class TestDataCollectionSingleReader:
    """Tests for DataCollection with a single Reader + splits."""

    def setup_method(self) -> None:
        """Create a shared reader and collection."""
        self.reader = _InMemoryReader(20)
        self.collection = DataCollection(
            reader=self.reader,
            splits={Split.TRAIN: 0.6, Split.VAL: 0.2, Split.TEST: 0.2},
            split_strategy=RandomSplit(),
            seed=42,
        )

    def test_split_sizes(self) -> None:
        """Splits have the expected number of samples."""
        assert len(self.collection.get_reader(Split.TRAIN)) == 12
        assert len(self.collection.get_reader(Split.VAL)) == 4
        assert len(self.collection.get_reader(Split.TEST)) == 4

    def test_no_index_overlap(self) -> None:
        """Splits are disjoint."""
        train_reader = self.collection.get_reader(Split.TRAIN)
        val_reader = self.collection.get_reader(Split.VAL)
        test_reader = self.collection.get_reader(Split.TEST)

        train_indices = set(train_reader._indices)
        val_indices = set(val_reader._indices)
        test_indices = set(test_reader._indices)

        assert train_indices.isdisjoint(val_indices)
        assert train_indices.isdisjoint(test_indices)
        assert val_indices.isdisjoint(test_indices)

    def test_split_names(self) -> None:
        """split_names returns all configured splits."""
        assert set(self.collection.split_names) == {
            Split.TRAIN,
            Split.VAL,
            Split.TEST,
        }

    def test_splits_without_strategy_raises(self) -> None:
        """ValueError when splits are given without split_strategy."""
        with pytest.raises(ValueError, match="split_strategy must be provided"):
            DataCollection(
                reader=_InMemoryReader(10),
                splits={Split.TRAIN: 0.8, Split.VAL: 0.2},
            )

    def test_get_dataset(self) -> None:
        """get_dataset returns a Dataset of the correct length."""
        ds = self.collection.get_dataset(Split.TRAIN)
        assert len(ds) == 12

    def test_convenience_properties(self) -> None:
        """train_dataset / val_dataset / test_dataset are usable."""
        assert len(self.collection.train_dataset) == 12
        assert len(self.collection.val_dataset) == 4
        assert len(self.collection.test_dataset) == 4

    def test_is_split(self) -> None:
        """is_split is True for a split collection."""
        assert self.collection.is_split is True


# ---------------------------------------------------------------------------
# DataCollection — unsplit (inference mode)
# ---------------------------------------------------------------------------


class TestDataCollectionUnsplit:
    """Tests for DataCollection with no splits (inference mode)."""

    def test_unsplit_get_reader(self) -> None:
        """get_reader() without split returns the full reader."""
        reader = _InMemoryReader(10)
        collection = DataCollection(reader=reader)
        assert len(collection.get_reader()) == 10

    def test_unsplit_is_split(self) -> None:
        """is_split is False for an unsplit collection."""
        collection = DataCollection(reader=_InMemoryReader(5))
        assert collection.is_split is False

    def test_unsplit_split_names_raises(self) -> None:
        """split_names raises RuntimeError for unsplit collection."""
        collection = DataCollection(reader=_InMemoryReader(5))
        with pytest.raises(RuntimeError, match="no named splits"):
            collection.split_names

    def test_unsplit_get_reader_with_split_raises(self) -> None:
        """Passing a split to an unsplit collection raises RuntimeError."""
        collection = DataCollection(reader=_InMemoryReader(5))
        with pytest.raises(RuntimeError, match="no named splits"):
            collection.get_reader(Split.TRAIN)

    def test_unsplit_get_dataset(self) -> None:
        """get_dataset() works without split argument."""
        collection = DataCollection(reader=_InMemoryReader(8))
        ds = collection.get_dataset()
        assert len(ds) == 8

    def test_unsplit_get_loader(self) -> None:
        """get_loader() works without split argument."""
        collection = DataCollection(reader=_InMemoryReader(6))
        loader = collection.get_loader(batch_size=3, use_streams=False)
        batches = list(loader)
        assert len(batches) == 2
        for batch in batches:
            assert isinstance(batch, Batch)

    def test_unsplit_list_reader(self) -> None:
        """List of readers without splits is concatenated unsplit."""
        r1 = _InMemoryReader(4, seed=0)
        r2 = _InMemoryReader(3, seed=1)
        collection = DataCollection(reader=[r1, r2])
        assert len(collection.get_reader()) == 7


# ---------------------------------------------------------------------------
# DataCollection — validation
# ---------------------------------------------------------------------------


class TestDataCollectionValidation:
    """Tests for split key validation."""

    def test_invalid_dict_key_raises(self) -> None:
        """ValueError when dict keys are plain strings, not Split enum."""
        with pytest.raises(ValueError, match="Invalid split key"):
            DataCollection(reader={"train": _InMemoryReader(5)})  # type: ignore[arg-type]

    def test_invalid_splits_key_raises(self) -> None:
        """ValueError when splits dict uses plain strings."""
        with pytest.raises(ValueError, match="Invalid split key"):
            DataCollection(
                reader=_InMemoryReader(10),
                splits={"train": 0.8},  # type: ignore[arg-type]
                split_strategy=RandomSplit(),
            )

    def test_get_reader_non_enum_raises(self) -> None:
        """ValueError when passing a non-Split to get_reader."""
        collection = DataCollection(
            reader={Split.TRAIN: _InMemoryReader(5)},
        )
        with pytest.raises(ValueError, match="Split enum member"):
            collection.get_reader("train")  # type: ignore[arg-type]

    def test_get_reader_missing_split_raises(self) -> None:
        """KeyError for a split not in the collection."""
        collection = DataCollection(
            reader={Split.TRAIN: _InMemoryReader(5)},
        )
        with pytest.raises(KeyError, match="not in collection"):
            collection.get_reader(Split.VAL)

    def test_get_reader_split_required_on_split_collection(self) -> None:
        """RuntimeError when calling get_reader() without split on a split collection."""
        collection = DataCollection(
            reader={Split.TRAIN: _InMemoryReader(5)},
        )
        with pytest.raises(RuntimeError, match="named splits"):
            collection.get_reader()


# ---------------------------------------------------------------------------
# DataCollection — dict of readers
# ---------------------------------------------------------------------------


class TestDataCollectionDictReader:
    """Tests for DataCollection constructed from a dict of readers."""

    def test_named_readers_passed_through(self) -> None:
        """Dict readers are used as-is (no splitting)."""
        r1 = _InMemoryReader(8)
        r2 = _InMemoryReader(4)
        collection = DataCollection(
            reader={Split.TRAIN: r1, Split.VAL: r2},
        )

        assert len(collection.get_reader(Split.TRAIN)) == 8
        assert len(collection.get_reader(Split.VAL)) == 4

    def test_unknown_split_raises(self) -> None:
        """KeyError for a split not in the dict."""
        collection = DataCollection(
            reader={Split.TRAIN: _InMemoryReader(5)},
        )
        with pytest.raises(KeyError):
            collection.get_reader(Split.VAL)


# ---------------------------------------------------------------------------
# DataLoader integration
# ---------------------------------------------------------------------------


class TestDataCollectionLoader:
    """Integration tests: DataCollection -> DataLoader -> Batch."""

    def test_get_loader_returns_dataloader(self) -> None:
        """get_loader returns a DataLoader."""
        reader = _InMemoryReader(10)
        collection = DataCollection(
            reader=reader,
            splits={Split.TRAIN: 0.8, Split.VAL: 0.2},
            split_strategy=RandomSplit(),
            seed=0,
        )
        loader = collection.get_loader(
            Split.TRAIN,
            batch_size=2,
            use_streams=False,
        )
        assert hasattr(loader, "__iter__")

    def test_train_loader_yields_batches(self) -> None:
        """Iterating train_loader yields Batch objects."""
        reader = _InMemoryReader(10)
        collection = DataCollection(
            reader=reader,
            splits={Split.TRAIN: 1.0},
            split_strategy=RandomSplit(),
            seed=0,
        )
        loader = collection.train_loader(batch_size=4, use_streams=False)
        batches = list(loader)
        assert len(batches) > 0
        for batch in batches:
            assert isinstance(batch, Batch)

    def test_round_trip_batch_shapes(self) -> None:
        """Full round-trip: collection -> loader -> verify batch tensor shapes."""
        reader = _InMemoryReader(8)
        collection = DataCollection(
            reader=reader,
            splits={Split.TRAIN: 1.0},
            split_strategy=RandomSplit(),
            seed=0,
        )
        loader = collection.train_loader(
            batch_size=4,
            use_streams=False,
            drop_last=False,
        )
        batch = next(iter(loader))
        assert isinstance(batch, Batch)
        assert batch.positions.ndim == 2
        assert batch.positions.shape[1] == 3
        assert batch.energies.ndim == 2
        assert batch.energies.shape[1] == 1

    def test_val_loader_no_shuffle(self) -> None:
        """val_loader defaults to shuffle=False and drop_last=False."""
        reader = _InMemoryReader(6)
        collection = DataCollection(
            reader=reader,
            splits={Split.VAL: 1.0},
            split_strategy=RandomSplit(),
            seed=0,
        )
        loader = collection.val_loader(batch_size=3, use_streams=False)
        batches = list(loader)
        assert len(batches) == 2


# ---------------------------------------------------------------------------
# ConcatReader
# ---------------------------------------------------------------------------


class TestConcatReader:
    """Tests for ConcatReader."""

    def test_length(self) -> None:
        """Total length is sum of sub-reader lengths."""
        r1 = _InMemoryReader(5, seed=0)
        r2 = _InMemoryReader(3, seed=1)
        concat = ConcatReader([r1, r2])
        assert len(concat) == 8

    def test_indexing_within_first_reader(self) -> None:
        """Indices in the first reader's range delegate correctly."""
        r1 = _InMemoryReader(4, seed=0)
        r2 = _InMemoryReader(3, seed=1)
        concat = ConcatReader([r1, r2])

        for i in range(4):
            data_c, _ = concat[i]
            data_r, _ = r1[i]
            for key in data_r:
                assert torch.equal(data_c[key], data_r[key])

    def test_indexing_across_boundary(self) -> None:
        """Indices in the second reader's range delegate correctly."""
        r1 = _InMemoryReader(4, seed=0)
        r2 = _InMemoryReader(3, seed=1)
        concat = ConcatReader([r1, r2])

        for i in range(3):
            data_c, _ = concat[4 + i]
            data_r, _ = r2[i]
            for key in data_r:
                assert torch.equal(data_c[key], data_r[key])

    def test_three_readers(self) -> None:
        """Works with more than two readers."""
        readers = [_InMemoryReader(2, seed=s) for s in range(3)]
        concat = ConcatReader(readers)
        assert len(concat) == 6

        data_c, _ = concat[5]
        data_r, _ = readers[2][1]
        for key in data_r:
            assert torch.equal(data_c[key], data_r[key])

    def test_metadata_includes_reader_index(self) -> None:
        """Metadata contains reader_index identifying the source reader."""
        r1 = _InMemoryReader(3, seed=0)
        r2 = _InMemoryReader(2, seed=1)
        concat = ConcatReader([r1, r2])

        _, meta0 = concat[0]
        assert meta0["reader_index"] == 0

        _, meta3 = concat[3]
        assert meta3["reader_index"] == 1

    def test_close_propagates(self) -> None:
        """Closing ConcatReader closes all sub-readers."""
        r1 = _InMemoryReader(2)
        r2 = _InMemoryReader(2)
        closed = []
        r1.close = lambda: closed.append(0)  # type: ignore[assignment]
        r2.close = lambda: closed.append(1)  # type: ignore[assignment]

        concat = ConcatReader([r1, r2])
        concat.close()
        assert sorted(closed) == [0, 1]

    def test_empty_readers_raises(self) -> None:
        """ValueError when no readers are provided."""
        with pytest.raises(ValueError, match="at least one reader"):
            ConcatReader([])

    def test_mismatched_field_names_raises(self) -> None:
        """ValueError when readers have different field names."""
        r1 = _InMemoryReader(2, seed=0)
        r2 = _InMemoryReader(2, seed=1)
        original_load = r2._load_sample

        def _patched(index: int) -> dict[str, torch.Tensor]:
            sample = original_load(index)
            sample["extra_field"] = torch.tensor([1.0])
            return sample

        r2._load_sample = _patched  # type: ignore[assignment]
        with pytest.raises(ValueError, match="field names"):
            ConcatReader([r1, r2])

    def test_single_reader_passthrough(self) -> None:
        """ConcatReader with one reader behaves identically to that reader."""
        r1 = _InMemoryReader(5, seed=0)
        concat = ConcatReader([r1])
        assert len(concat) == 5
        for i in range(5):
            data_c, _ = concat[i]
            data_r, _ = r1[i]
            for key in data_r:
                assert torch.equal(data_c[key], data_r[key])

    def test_out_of_bounds(self) -> None:
        """IndexError when accessing beyond total length."""
        concat = ConcatReader([_InMemoryReader(3), _InMemoryReader(2)])
        with pytest.raises(IndexError):
            concat[5]

    def test_negative_indexing(self) -> None:
        """Negative indices work via Reader.__getitem__."""
        r1 = _InMemoryReader(3, seed=0)
        r2 = _InMemoryReader(2, seed=1)
        concat = ConcatReader([r1, r2])

        data_neg, _ = concat[-1]
        data_pos, _ = concat[4]
        for key in data_pos:
            assert torch.equal(data_neg[key], data_pos[key])


# ---------------------------------------------------------------------------
# DataCollection — multi-reader (list per split)
# ---------------------------------------------------------------------------


class TestDataCollectionMultiReader:
    """Tests for DataCollection with multiple readers per split."""

    def test_dict_with_list_values(self) -> None:
        """Dict values that are lists get concatenated."""
        r1 = _InMemoryReader(5, seed=0)
        r2 = _InMemoryReader(3, seed=1)
        r3 = _InMemoryReader(4, seed=2)

        collection = DataCollection(
            reader={Split.TRAIN: [r1, r2], Split.VAL: r3},
        )
        assert len(collection.get_reader(Split.TRAIN)) == 8
        assert len(collection.get_reader(Split.VAL)) == 4

    def test_list_reader_with_splits(self) -> None:
        """Top-level list of readers is concatenated then split."""
        r1 = _InMemoryReader(10, seed=0)
        r2 = _InMemoryReader(10, seed=1)

        collection = DataCollection(
            reader=[r1, r2],
            splits={Split.TRAIN: 0.8, Split.VAL: 0.2},
            split_strategy=RandomSplit(),
            seed=42,
        )
        assert len(collection.get_reader(Split.TRAIN)) == 16
        assert len(collection.get_reader(Split.VAL)) == 4

    def test_multi_reader_loader_yields_batches(self) -> None:
        """Loader from multi-reader collection yields valid Batches."""
        r1 = _InMemoryReader(6, seed=0)
        r2 = _InMemoryReader(4, seed=1)

        collection = DataCollection(
            reader={Split.TRAIN: [r1, r2]},
        )
        loader = collection.train_loader(batch_size=5, use_streams=False)
        batches = list(loader)
        assert len(batches) > 0
        for batch in batches:
            assert isinstance(batch, Batch)

    def test_multi_reader_no_data_loss(self) -> None:
        """All samples from all readers are accessible."""
        r1 = _InMemoryReader(4, seed=0)
        r2 = _InMemoryReader(3, seed=1)

        collection = DataCollection(
            reader={Split.TRAIN: [r1, r2]},
        )
        reader = collection.get_reader(Split.TRAIN)
        assert len(reader) == 7

        for i in range(7):
            data, meta = reader[i]
            assert "positions" in data
            assert "reader_index" in meta


class TestDataCollectionDeviceType:
    """Tests for the DataCollection device parameter handling."""

    def test_string_device_stored_as_string(self) -> None:
        """Passing a string device type stores it as a string."""
        reader = _InMemoryReader(4)
        collection = DataCollection(reader=reader, device="cpu")
        assert collection._device == "cpu"
        assert isinstance(collection._device, str)

    def test_torch_device_stored_as_torch_device(self) -> None:
        """Passing a torch.device preserves it (backward compat)."""
        reader = _InMemoryReader(4)
        dev = torch.device("cpu")
        collection = DataCollection(reader=reader, device=dev)
        assert collection._device == dev
        assert isinstance(collection._device, torch.device)

    def test_none_device_auto_detects_as_string(self) -> None:
        """device=None auto-detects and stores a string, not torch.device."""
        reader = _InMemoryReader(4)
        collection = DataCollection(reader=reader, device=None)
        assert isinstance(collection._device, str)
        assert collection._device in ("cuda", "cpu")

    def test_string_device_forwarded_to_dataset(self) -> None:
        """String device is forwarded to Dataset via get_dataset."""
        reader = _InMemoryReader(4)
        collection = DataCollection(reader=reader, device="cpu")
        dataset = collection.get_dataset()
        assert dataset.target_device == torch.device("cpu")
