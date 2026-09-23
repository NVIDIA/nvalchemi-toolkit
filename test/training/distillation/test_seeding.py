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

import dataclasses
from collections.abc import Sequence
from pathlib import Path

import pytest
import torch
from pydantic import ValidationError

from nvalchemi.data import AtomicData, Batch
from nvalchemi.data.datapipes.backends.zarr import (
    AtomicDataZarrReader,
    AtomicDataZarrWriter,
)
from nvalchemi.data.datapipes.dataset import Dataset
from nvalchemi.data.datapipes.in_memory_dataset import InMemoryDataset
from nvalchemi.training.distillation import InitialStructures, WithinBudget
from nvalchemi.training.distillation.seeding import _InitialStructuresSpec
from test.training.conftest import _build_atomic_data
from test.training.distillation.conftest import _build_small_dataset

_SHARD_SIZES = (2, 3, 4, 5, 6)
"""Atom counts of ``_build_small_dataset``, distinct so a row is identifiable."""


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


def _served_sizes(drawn: list[AtomicData]) -> list[int]:
    """Return the atom count of every structure a draw handed back."""
    return [int(data.positions.shape[0]) for data in drawn]


def _make_store(tmp_path: Path, sizes: Sequence[int] = (2, 3, 4)) -> Dataset:
    """Return a path-backed dataset, which is the only kind a recipe can name."""
    store = tmp_path / "structures.zarr"
    AtomicDataZarrWriter(store).write(_make_dataset(sizes).in_memory_batch)
    return Dataset(reader=AtomicDataZarrReader(store))


@dataclasses.dataclass(frozen=True)
class _EstimatedMemory:
    """Fit policy bounding a per-atom memory estimate, a budget axis of its own."""

    bytes_per_atom: int
    budget: int

    def __call__(self, num_atoms: int, num_edges: int) -> bool:  # noqa: ARG002
        """Return whether the estimated footprint of *num_atoms* fits the budget."""
        return num_atoms * self.bytes_per_atom <= self.budget


class TestInitialStructuresCursor:
    def test_an_unbudgeted_source_seeds_every_row_it_owns(self) -> None:
        """A bare dataset is propagated whole."""
        source = InitialStructures(_build_small_dataset())

        state = source.initial_batch()

        assert state.num_graphs == 5
        assert source.cursor == 5

    def test_the_cursor_opens_past_the_seeded_batch(self) -> None:
        """A budgeted source leaves the rows it did not pack for a later draw."""
        source = InitialStructures(_build_small_dataset(), max_batch_size=2)

        state = source.initial_batch()

        assert state.num_graphs == 2
        assert source.cursor == 2

    def test_an_unbudgeted_source_hands_out_nothing(self) -> None:
        """The initial batch consumed every row, so a draw has no remainder."""
        source = InitialStructures(_build_small_dataset())
        source.initial_batch()

        assert source.draw() == []
        assert source.exhausted

    def test_an_initial_batch_stops_at_the_first_structure_over_budget(self) -> None:
        """Packing stops on the miss and leaves it at the cursor for a later draw."""
        source = InitialStructures(_make_dataset([3, 8, 2]), max_atoms=4)

        state = source.initial_batch()

        assert state.num_graphs == 1
        assert source.cursor == 1

    def test_two_draws_never_serve_one_structure_twice(self) -> None:
        """The cursor is shared, so a second draw opens where the first stopped."""
        source = InitialStructures(_build_small_dataset(), max_batch_size=1)
        source.initial_batch()

        first = source.draw(limit=2)
        second = source.draw(limit=2)

        assert _served_sizes(first) == [3, 4]
        assert _served_sizes(second) == [5, 6]

    def test_drawn_structures_continue_the_seeded_numbering(self) -> None:
        """Ids number the trajectories the run started, seeded and drawn alike."""
        source = InitialStructures(_build_small_dataset(), max_batch_size=2)
        state = source.initial_batch()

        drawn = source.draw(limit=1)

        assert state["system_id"].view(-1).tolist() == [0, 1]
        assert int(drawn[0].system_id.view(-1)[0]) == 2

    def test_a_seed_batch_arrives_without_the_previous_run_bookkeeping(self) -> None:
        """Status describes the run that wrote it, so the source installs its own."""
        dataset = _build_small_dataset()
        frames = dataset.in_memory_batch
        frames.add_key(
            "status",
            [torch.full((1, 1), 3, dtype=torch.long) for _ in range(5)],
            level="system",
        )

        state = InitialStructures(
            InMemoryDataset(in_memory_batch=frames)
        ).initial_batch()

        assert state["status"].view(-1).tolist() == [0] * 5

    def test_a_budget_that_fits_nothing_is_rejected(self) -> None:
        """A run has to propagate something, and says so before it starts."""
        source = InitialStructures(_make_dataset([9, 9]), max_atoms=4)

        with pytest.raises(ValueError, match="has to propagate something"):
            source.initial_batch()

    def test_a_non_positive_budget_is_rejected(self) -> None:
        """A budget bounds a batch, so it has to name a count a batch can hold."""
        with pytest.raises(ValueError, match="must be positive"):
            InitialStructures(_build_small_dataset(), max_atoms=0)


class TestInitialStructuresDraw:
    def test_a_miss_stops_the_draw_and_stays_at_the_cursor(self) -> None:
        """Under ``on_miss="stop"`` the oversized structure is left for the next draw."""
        source = InitialStructures(_make_dataset([3, 8, 2]), max_batch_size=1)
        source.initial_batch()

        stopped = source.draw(fits=WithinBudget(atoms=4))
        widened = source.draw(fits=WithinBudget(atoms=8))

        assert stopped == []
        assert _served_sizes(widened) == [8]

    def test_a_miss_is_passed_over_when_asked(self) -> None:
        """Under ``on_miss="skip"`` an oversized structure does not starve the refill."""
        source = InitialStructures(_make_dataset([3, 8, 2]), max_batch_size=1)
        source.initial_batch()

        drawn = source.draw(fits=WithinBudget(atoms=4), on_miss="skip")

        assert _served_sizes(drawn) == [2]
        assert source.exhausted

    def test_the_policy_sees_the_running_totals(self) -> None:
        """A budget is spent across the draw, not checked per structure."""
        source = InitialStructures(_make_dataset([2, 2, 2, 2]), max_batch_size=1)
        source.initial_batch()

        drawn = source.draw(fits=WithinBudget(atoms=5))

        assert _served_sizes(drawn) == [2, 2]
        assert source.cursor == 3

    def test_limit_caps_the_draw(self) -> None:
        """A draw serves at most *limit* structures however many fit."""
        source = InitialStructures(_build_small_dataset(), max_batch_size=1)
        source.initial_batch()

        assert len(source.draw(limit=3)) == 3
        assert source.cursor == 4

    def test_nothing_that_fits_hands_back_nothing(self) -> None:
        """A skipping draw that reaches no structure small enough returns empty."""
        source = InitialStructures(_make_dataset([2, 8, 9]), max_batch_size=1)
        source.initial_batch()

        assert source.draw(fits=WithinBudget(atoms=1), on_miss="skip") == []
        assert source.exhausted

    def test_a_custom_policy_decides_the_fit(self) -> None:
        """Any predicate over the totals is a policy, not only an atom or edge bound."""
        source = InitialStructures(_make_dataset([2, 3, 4, 5]), max_batch_size=1)
        source.initial_batch()

        drawn = source.draw(fits=_EstimatedMemory(bytes_per_atom=16, budget=120))

        assert _served_sizes(drawn) == [3, 4]

    def test_within_budget_bounds_edges_only_when_asked(self) -> None:
        """An edge bound is opt-in, since a dataset reports stored edges only."""
        assert WithinBudget(atoms=10)(num_atoms=10, num_edges=10**6)
        assert not WithinBudget(edges=5)(num_atoms=1, num_edges=6)


class TestInitialStructuresShard:
    def test_rows_are_dealt_strided_and_disjoint(self) -> None:
        """Rank r takes every world-th row from offset r, unpadded and unshuffled."""
        source = InitialStructures(_build_small_dataset())

        source.shard(1, 2)

        assert source.rows == (1, 3)
        assert len(source) == 2

    def test_a_single_rank_run_owns_the_whole_dataset(self) -> None:
        """The strided deal degenerates to the dataset itself on one process."""
        source = InitialStructures(_build_small_dataset())

        source.shard(0, 1)

        assert source.rows == tuple(range(5))

    def test_exhaustion_counts_shard_positions(self) -> None:
        """A rank is done when *its* rows are gone, not when the dataset's are."""
        source = InitialStructures(_build_small_dataset())
        source.shard(1, 2)

        source.initial_batch()

        assert source.exhausted
        assert len(source) == 2

    def test_two_ranks_draw_disjoint_rows_covering_the_set(self) -> None:
        """A structure served to a rank that does not own it is propagated twice."""
        dataset = _build_small_dataset()
        served: list[list[int]] = []
        for rank in (0, 1):
            source = InitialStructures(dataset, max_batch_size=1)
            source.shard(rank, 2)
            state = source.initial_batch()
            served.append([int(state.num_nodes)] + _served_sizes(source.draw()))

        assert set(served[0]).isdisjoint(served[1])
        assert sorted(served[0] + served[1]) == sorted(_SHARD_SIZES)

    def test_a_rank_outside_its_world_is_rejected(self) -> None:
        """A shard is dealt to one rank of a world, so it has to name a position."""
        source = InitialStructures(_build_small_dataset())

        with pytest.raises(ValueError, match="rank=2 of world_size=2"):
            source.shard(2, 2)

    def test_installing_a_shard_reopens_the_cursor(self) -> None:
        """A rerun reseeds the trajectory, so the source opens at its front again."""
        source = InitialStructures(_build_small_dataset())
        source.initial_batch()

        source.shard(0, 1)

        assert (source.cursor, source.next_system_id) == (0, 0)


class TestInitialStructuresState:
    def test_the_cursor_round_trips_through_a_state_dict(self) -> None:
        """A restart resumes the position and the next id."""
        source = InitialStructures(_build_small_dataset(), max_batch_size=1)
        source.initial_batch()
        source.draw(limit=2)

        restored = InitialStructures(_build_small_dataset(), max_batch_size=1)
        restored.load_state_dict(source.state_dict())

        assert restored.state_dict() == source.state_dict()
        assert (restored.cursor, restored.next_system_id) == (3, 3)

    def test_a_restored_source_resumes_at_its_cursor_not_at_its_ids(self) -> None:
        """Ids skip the structures a policy passed over, so they name no row."""
        sizes = [2, 9, 3, 4]
        source = InitialStructures(_make_dataset(sizes), max_batch_size=1)
        source.initial_batch()
        source.draw(limit=1, fits=WithinBudget(atoms=5), on_miss="skip")
        state = source.state_dict()

        restored = InitialStructures(_make_dataset(sizes), max_batch_size=1)
        restored.load_state_dict(state)

        assert state["next_system_id"] < state["cursor"]
        assert _served_sizes(restored.draw(limit=1)) == _served_sizes(
            source.draw(limit=1)
        )

    def test_a_bundle_from_another_shard_is_refused(self) -> None:
        """A cursor counts positions in one rank's rows and no others."""
        source = InitialStructures(_build_small_dataset())
        source.shard(0, 1)

        with pytest.raises(ValueError, match="written for rank 1 of 2"):
            source.load_state_dict(
                {"cursor": 0, "next_system_id": 0, "rank": 1, "world_size": 2}
            )


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
