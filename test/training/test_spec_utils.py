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
"""Tests for :mod:`nvalchemi.training._spec_utils`."""

from __future__ import annotations

from pathlib import Path

import pytest

from nvalchemi.data.datapipes.backends.zarr import (
    AtomicDataZarrReader,
    AtomicDataZarrWriter,
)
from nvalchemi.data.datapipes.dataset import Dataset
from nvalchemi.data.datapipes.in_memory_dataset import InMemoryDataset
from nvalchemi.data.datapipes.multidataset import MultiDataset
from nvalchemi.training._spec_utils import (
    DatasetRef,
    SpecSerializable,
    dataset_from_spec_dict,
    dataset_spec_dict,
)
from test.training.conftest import _build_batch


def _make_store(tmp_path: Path, name: str = "structures.zarr") -> Dataset:
    """Return a path-backed dataset of three systems, the kind a spec can name."""
    store = tmp_path / name
    AtomicDataZarrWriter(store).write(_build_batch(n_systems=3))
    return Dataset(reader=AtomicDataZarrReader(store))


def _make_composition(tmp_path: Path) -> tuple[MultiDataset, list[str]]:
    """Return a MultiDataset over two stores and the paths it concatenates, in order."""
    names = ["first.zarr", "second.zarr"]
    composed = MultiDataset(*[_make_store(tmp_path, name) for name in names])
    return composed, [str(tmp_path / name) for name in names]


class TestDatasetSpecDict:
    def test_a_path_backed_dataset_is_named_by_its_store(self, tmp_path: Path) -> None:
        """The reference carries the store path and the collation device."""
        dataset = _make_store(tmp_path)

        spec = dataset_spec_dict(dataset, field="Strategy.dataset")

        assert spec == {
            "path": str(tmp_path / "structures.zarr"),
            "device": str(dataset.target_device),
        }
        assert set(spec) < set(DatasetRef.model_fields)

    def test_a_composition_is_named_by_the_stores_it_concatenates(
        self, tmp_path: Path
    ) -> None:
        """A MultiDataset serializes as its children's paths in global index order."""
        composed, paths = _make_composition(tmp_path)

        spec = dataset_spec_dict(composed, field="Strategy.dataset")

        assert spec == {
            "paths": paths,
            "device": str(composed.datasets[0].target_device),
        }

    def test_a_composition_holding_a_memory_dataset_is_refused(self) -> None:
        """Every store a composition concatenates has to be one a path names."""
        composed = MultiDataset(InMemoryDataset(in_memory_batch=_build_batch()))

        with pytest.raises(
            ValueError, match="Strategy.dataset is a InMemoryDataset holding"
        ):
            dataset_spec_dict(composed, field="Strategy.dataset")

    def test_an_in_memory_dataset_is_refused_naming_the_field(self) -> None:
        """A spec references a dataset by its store, so a memory-held one has none."""
        dataset = InMemoryDataset(in_memory_batch=_build_batch())

        with pytest.raises(
            ValueError, match="Strategy.dataset is a InMemoryDataset holding"
        ):
            dataset_spec_dict(dataset, field="Strategy.dataset")

    def test_the_default_remedy_names_the_zarr_writer(self) -> None:
        """Without a caller remedy the error still says how to get a store."""
        dataset = InMemoryDataset(in_memory_batch=_build_batch())

        with pytest.raises(ValueError, match="AtomicDataZarrWriter"):
            dataset_spec_dict(dataset, field="Strategy.dataset")

    def test_a_caller_remedy_replaces_the_default_one(self) -> None:
        """The caller's sentence ends the error, so it can name its own writer."""
        dataset = InMemoryDataset(in_memory_batch=_build_batch())

        with pytest.raises(ValueError, match=r"Use my_writer\.$"):
            dataset_spec_dict(
                dataset, field="Strategy.dataset", remedy="Use my_writer."
            )


class TestDatasetFromSpecDict:
    def test_a_reference_reopens_the_store(self, tmp_path: Path) -> None:
        """The rebuilt dataset reads the rows the store holds."""
        spec = dataset_spec_dict(_make_store(tmp_path), field="Strategy.dataset")

        rebuilt = dataset_from_spec_dict(spec, field="Strategy.dataset")

        assert len(rebuilt) == 3
        assert dataset_spec_dict(rebuilt, field="Strategy.dataset") == spec

    def test_a_composition_reference_reopens_a_multidataset(
        self, tmp_path: Path
    ) -> None:
        """The rebuilt composition reads every row of every store and re-serializes alike."""
        composed, _ = _make_composition(tmp_path)
        spec = dataset_spec_dict(composed, field="Strategy.dataset")

        rebuilt = dataset_from_spec_dict(spec, field="Strategy.dataset")

        assert isinstance(rebuilt, MultiDataset)
        assert len(rebuilt) == len(composed) == 6
        assert dataset_spec_dict(rebuilt, field="Strategy.dataset") == spec

    def test_a_reference_without_a_path_is_refused_naming_the_field(self) -> None:
        """A store a spec forgot to name is a spec error, not a raw KeyError."""
        with pytest.raises(ValueError, match="Strategy.dataset must reference"):
            dataset_from_spec_dict({"device": "cpu"}, field="Strategy.dataset")

    def test_a_reference_naming_one_store_and_a_composition_is_refused(self) -> None:
        """``path`` and ``paths`` are alternatives, so a reference giving both is ambiguous."""
        with pytest.raises(ValueError, match="either one store under path"):
            dataset_from_spec_dict(
                {"path": "x.zarr", "paths": ["y.zarr"]}, field="Strategy.dataset"
            )

    def test_a_reference_with_an_unknown_key_is_refused(self) -> None:
        """A misspelled key is refused rather than silently dropped."""
        with pytest.raises(ValueError, match="paht"):
            dataset_from_spec_dict(
                {"path": "x.zarr", "paht": "y.zarr"}, field="Strategy.dataset"
            )


class _Recipeable:
    """Collaborator offering both halves of the spec protocol."""

    def to_spec_dict(self) -> dict[str, str]:
        """Return a block naming this object."""
        return {"kind": "recipeable"}

    @classmethod
    def from_spec_dict(cls, spec: dict[str, str]) -> _Recipeable:  # noqa: ARG003
        """Rebuild from the block."""
        return cls()


class _WriteOnly:
    """Collaborator that serializes but cannot be rebuilt."""

    def to_spec_dict(self) -> dict[str, str]:
        """Return a block nothing reads back."""
        return {}


class TestSpecSerializable:
    def test_an_object_with_both_methods_satisfies_the_protocol(self) -> None:
        """The runtime check accepts a class carrying to_spec_dict and from_spec_dict."""
        assert isinstance(_Recipeable(), SpecSerializable)
        assert isinstance(
            _Recipeable.from_spec_dict(_Recipeable().to_spec_dict()), _Recipeable
        )

    def test_an_object_missing_from_spec_dict_does_not(self) -> None:
        """Serializing alone is not enough to be named by a spec."""
        assert not isinstance(_WriteOnly(), SpecSerializable)
        assert not isinstance(object(), SpecSerializable)

    def test_the_distillation_package_re_exports_the_core_protocol(self) -> None:
        """The recipe layer names the same protocol the core defines."""
        from nvalchemi.training.distillation import SpecSerializable as exported

        assert exported is SpecSerializable
