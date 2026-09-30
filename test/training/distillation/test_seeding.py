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
"""Tests for :mod:`nvalchemi.training.distillation.seeding`."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pytest
from pydantic import ValidationError

from nvalchemi.data import Batch
from nvalchemi.data.datapipes.backends.zarr import (
    AtomicDataZarrReader,
    AtomicDataZarrWriter,
)
from nvalchemi.data.datapipes.dataset import Dataset
from nvalchemi.data.datapipes.in_memory_dataset import InMemoryDataset
from nvalchemi.dynamics import OrderedStructureSampler, StructureSource
from nvalchemi.training.distillation import (
    InitialStructures,
    InitialStructuresSource,
)
from nvalchemi.training.distillation.seeding import _InitialStructuresSpec
from test.training.conftest import _build_atomic_data
from test.training.distillation.conftest import _build_small_dataset


def _make_dataset(sizes: Sequence[int]) -> InMemoryDataset:
    """Return a dataset of systems holding *sizes* atoms, in that order."""
    return InMemoryDataset(
        in_memory_batch=Batch.from_data_list(
            [
                _build_atomic_data(n_atoms=size, seed=300 + index)
                for index, size in enumerate(sizes)
            ]
        )
    )


def _make_store(tmp_path: Path, sizes: Sequence[int] = (2, 3, 4)) -> Dataset:
    """Return a path-backed dataset, which is the only kind a recipe can name."""
    store = tmp_path / "structures.zarr"
    AtomicDataZarrWriter(store).write(_make_dataset(sizes).in_memory_batch)
    return Dataset(reader=AtomicDataZarrReader(store))


class TestInitialStructuresIsTheCoreSampler:
    def test_the_loop_names_are_the_core_ones(self) -> None:
        """The distillation names stay importable and resolve to the core objects."""
        assert issubclass(InitialStructures, OrderedStructureSampler)
        assert InitialStructuresSource is StructureSource
        assert isinstance(InitialStructures(_build_small_dataset()), StructureSource)

    def test_the_budget_refusal_names_the_subclass(self) -> None:
        """An error raised from the core sampler still names the class the user built."""
        with pytest.raises(ValueError, match="InitialStructures max_atoms"):
            InitialStructures(_build_small_dataset(), max_atoms=0)


class TestInitialStructuresSpec:
    def test_a_path_backed_source_round_trips_through_a_spec(
        self, tmp_path: Path
    ) -> None:
        """A recipe names the store the structures are read from and the budgets set."""
        source = InitialStructures(
            _make_store(tmp_path), max_atoms=32, max_batch_size=2
        )

        rebuilt = InitialStructures.from_spec_dict(source.to_spec_dict())

        assert set(source.to_spec_dict()) == set(_InitialStructuresSpec.model_fields)
        assert rebuilt.to_spec_dict() == source.to_spec_dict()
        assert (rebuilt.max_atoms, rebuilt.max_batch_size) == (32, 2)
        assert len(rebuilt) == 3

    def test_a_misspelled_budget_is_refused_by_name(self, tmp_path: Path) -> None:
        """A budget that reaches no field leaves the run silently unbudgeted."""
        spec = InitialStructures(_make_store(tmp_path)).to_spec_dict()
        spec["max_atom"] = 10

        with pytest.raises(ValidationError, match="max_atom"):
            InitialStructures.from_spec_dict(spec)

    def test_a_non_positive_budget_is_refused(self, tmp_path: Path) -> None:
        """A budget bounds a batch, so it has to name a count a batch can hold."""
        spec = InitialStructures(_make_store(tmp_path)).to_spec_dict()
        spec["max_atoms"] = -5

        with pytest.raises(ValidationError):
            InitialStructures.from_spec_dict(spec)

    def test_a_non_numeric_budget_is_refused(self, tmp_path: Path) -> None:
        """A budget the policy compares totals against cannot be a word."""
        spec = InitialStructures(_make_store(tmp_path)).to_spec_dict()
        spec["max_batch_size"] = "four"

        with pytest.raises(ValidationError):
            InitialStructures.from_spec_dict(spec)

    def test_a_dataset_reference_without_a_path_is_refused(
        self, tmp_path: Path
    ) -> None:
        """A store a recipe forgot to name is a recipe error, not a raw KeyError."""
        spec = InitialStructures(_make_store(tmp_path)).to_spec_dict()
        del spec["dataset"]["path"]

        with pytest.raises(ValidationError):
            InitialStructures.from_spec_dict(spec)

    def test_an_in_memory_source_cannot_be_named(self) -> None:
        """A spec references a dataset by the store it reads."""
        source = InitialStructures(_build_small_dataset())

        with pytest.raises(ValueError, match="OnPolicyConfig.initial_structures is a"):
            source.to_spec_dict()
