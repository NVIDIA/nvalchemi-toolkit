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
from nvalchemi.training._spec_utils import (
    DatasetRef,
    dataset_from_spec_dict,
    dataset_spec_dict,
)
from test.training.conftest import _build_batch


def _make_store(tmp_path: Path) -> Dataset:
    """Return a path-backed dataset of three systems, the kind a spec can name."""
    store = tmp_path / "structures.zarr"
    AtomicDataZarrWriter(store).write(_build_batch(n_systems=3))
    return Dataset(reader=AtomicDataZarrReader(store))


class TestDatasetSpecDict:
    def test_a_path_backed_dataset_is_named_by_its_store(self, tmp_path: Path) -> None:
        """The reference carries the store path and the collation device."""
        dataset = _make_store(tmp_path)

        spec = dataset_spec_dict(dataset, field="Strategy.dataset")

        assert spec == {
            "path": str(tmp_path / "structures.zarr"),
            "device": str(dataset.target_device),
        }
        assert set(spec) == set(DatasetRef.model_fields)

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

    def test_a_reference_without_a_path_is_refused_naming_the_field(self) -> None:
        """A store a spec forgot to name is a spec error, not a raw KeyError."""
        with pytest.raises(ValueError, match="Strategy.dataset must reference"):
            dataset_from_spec_dict({"device": "cpu"}, field="Strategy.dataset")

    def test_a_reference_with_an_unknown_key_is_refused(self) -> None:
        """A misspelled key is refused rather than silently dropped."""
        with pytest.raises(ValueError, match="paht"):
            dataset_from_spec_dict(
                {"path": "x.zarr", "paht": "y.zarr"}, field="Strategy.dataset"
            )
