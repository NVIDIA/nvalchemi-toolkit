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
"""Tests for :mod:`nvalchemi.dynamics.structure_sampler`."""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence

import pytest
import torch

from nvalchemi.data import AtomicData, Batch
from nvalchemi.data.datapipes.in_memory_dataset import InMemoryDataset
from nvalchemi.dynamics import (
    FitPolicy,
    OrderedStructureSampler,
    StructureSource,
    WithinBudget,
)

_SIZES = (2, 3, 4, 5, 6)
"""Atom counts of the default dataset, distinct so a row is identifiable."""


def _make_atomic_data(n_atoms: int, seed: int) -> AtomicData:
    """Return one structure of *n_atoms* atoms with masses and forces filled in."""
    g = torch.Generator().manual_seed(seed)
    return AtomicData(
        positions=torch.randn(n_atoms, 3, generator=g),
        atomic_numbers=torch.randint(1, 10, (n_atoms,), dtype=torch.long, generator=g),
        atomic_masses=torch.ones(n_atoms),
        energy=torch.randn(1, 1, generator=g),
        forces=torch.randn(n_atoms, 3, generator=g),
    )


def _make_dataset(sizes: Sequence[int] = _SIZES) -> InMemoryDataset:
    """Return a dataset of systems holding *sizes* atoms, in that order."""
    return InMemoryDataset(
        in_memory_batch=Batch.from_data_list(
            [
                _make_atomic_data(n_atoms=size, seed=300 + index)
                for index, size in enumerate(sizes)
            ]
        )
    )


def _served_sizes(drawn: list[AtomicData]) -> list[int]:
    """Return the atom count of every structure a draw handed back."""
    return [int(data.positions.shape[0]) for data in drawn]


@dataclasses.dataclass(frozen=True)
class _EstimatedMemory:
    """Fit policy bounding a per-atom memory estimate, a budget axis of its own."""

    bytes_per_atom: int
    budget: int

    def __call__(self, num_atoms: int, num_edges: int) -> bool:  # noqa: ARG002
        """Return whether the estimated footprint of *num_atoms* fits the budget."""
        return num_atoms * self.bytes_per_atom <= self.budget


def _admit(policy: FitPolicy, num_atoms: int, num_edges: int) -> bool:
    """Call *policy* the way a size-aware caller typed against the protocol does."""
    return policy(num_atoms, num_edges)


class TestFitPolicy:
    def test_within_budget_is_a_fit_policy(self) -> None:
        """A caller typed against the protocol takes the built-in bound as is."""
        policy = WithinBudget(atoms=10)

        assert _admit(policy, 8, 0)
        assert not _admit(policy, 12, 0)

    def test_within_budget_bounds_edges_only_when_asked(self) -> None:
        """An edge bound is opt-in, since a dataset reports stored edges only."""
        assert WithinBudget(atoms=10)(num_atoms=10, num_edges=10**6)
        assert not WithinBudget(edges=5)(num_atoms=1, num_edges=6)

    def test_a_custom_predicate_is_a_fit_policy_too(self) -> None:
        """Any predicate over the totals is a policy, not only an atom or edge bound."""
        assert _admit(_EstimatedMemory(bytes_per_atom=16, budget=120), 7, 0)
        assert not _admit(_EstimatedMemory(bytes_per_atom=16, budget=120), 8, 0)


class TestOrderedStructureSamplerInitialBatch:
    def test_the_sampler_is_a_structure_source(self) -> None:
        """The reference implementation satisfies the protocol it documents."""
        assert isinstance(OrderedStructureSampler(_make_dataset()), StructureSource)

    def test_an_unbudgeted_sampler_serves_every_row_it_owns(self) -> None:
        """A bare dataset is propagated whole."""
        sampler = OrderedStructureSampler(_make_dataset())

        state = sampler.initial_batch()

        assert state.num_graphs == 5
        assert sampler.next_row == 5

    def test_next_row_opens_past_the_initial_batch(self) -> None:
        """A budgeted sampler leaves the rows it did not pack for a later draw."""
        sampler = OrderedStructureSampler(_make_dataset(), max_batch_size=2)

        state = sampler.initial_batch()

        assert state.num_graphs == 2
        assert sampler.next_row == 2

    def test_an_unbudgeted_sampler_hands_out_nothing_afterwards(self) -> None:
        """The initial batch consumed every row, so a draw has no remainder."""
        sampler = OrderedStructureSampler(_make_dataset())
        sampler.initial_batch()

        assert sampler.draw() == []
        assert sampler.exhausted

    def test_an_initial_batch_stops_at_the_first_structure_over_budget(self) -> None:
        """Packing stops on the miss and leaves it at next_row for a later draw."""
        sampler = OrderedStructureSampler(_make_dataset([3, 8, 2]), max_atoms=4)

        state = sampler.initial_batch()

        assert state.num_graphs == 1
        assert sampler.next_row == 1

    def test_two_draws_never_serve_one_structure_twice(self) -> None:
        """The position is shared, so a second draw opens where the first stopped."""
        sampler = OrderedStructureSampler(_make_dataset(), max_batch_size=1)
        sampler.initial_batch()

        first = sampler.draw(limit=2)
        second = sampler.draw(limit=2)

        assert _served_sizes(first) == [3, 4]
        assert _served_sizes(second) == [5, 6]

    def test_drawn_structures_continue_the_initial_numbering(self) -> None:
        """Ids number the structures the run started, initial and drawn alike."""
        sampler = OrderedStructureSampler(_make_dataset(), max_batch_size=2)
        state = sampler.initial_batch()

        drawn = sampler.draw(limit=1)

        assert state["system_id"].view(-1).tolist() == [0, 1]
        assert int(drawn[0].system_id.view(-1)[0]) == 2

    def test_an_initial_batch_arrives_without_the_previous_run_bookkeeping(
        self,
    ) -> None:
        """Status describes the run that wrote it, so the sampler installs its own."""
        dataset = _make_dataset()
        frames = dataset.in_memory_batch
        frames.add_key(
            "status",
            [torch.full((1, 1), 3, dtype=torch.long) for _ in range(5)],
            level="system",
        )

        state = OrderedStructureSampler(
            InMemoryDataset(in_memory_batch=frames)
        ).initial_batch()

        assert state["status"].view(-1).tolist() == [0] * 5

    def test_a_budget_that_fits_nothing_is_rejected(self) -> None:
        """A run has to propagate something, and says so before it starts."""
        sampler = OrderedStructureSampler(_make_dataset([9, 9]), max_atoms=4)

        with pytest.raises(ValueError, match="has to propagate something"):
            sampler.initial_batch()

    def test_a_non_positive_budget_is_rejected(self) -> None:
        """A budget bounds a batch, so it has to name a count a batch can hold."""
        with pytest.raises(ValueError, match="must be positive"):
            OrderedStructureSampler(_make_dataset(), max_atoms=0)


class TestOrderedStructureSamplerDraw:
    def test_a_miss_stops_the_draw_and_stays_on_the_row(self) -> None:
        """Under ``on_miss="stop"`` the oversized structure is left for the next draw."""
        sampler = OrderedStructureSampler(_make_dataset([3, 8, 2]), max_batch_size=1)
        sampler.initial_batch()

        stopped = sampler.draw(fits=WithinBudget(atoms=4))
        widened = sampler.draw(fits=WithinBudget(atoms=8))

        assert stopped == []
        assert _served_sizes(widened) == [8]

    def test_a_miss_is_passed_over_when_asked(self) -> None:
        """Under ``on_miss="skip"`` an oversized structure does not starve the refill."""
        sampler = OrderedStructureSampler(_make_dataset([3, 8, 2]), max_batch_size=1)
        sampler.initial_batch()

        drawn = sampler.draw(fits=WithinBudget(atoms=4), on_miss="skip")

        assert _served_sizes(drawn) == [2]
        assert sampler.exhausted

    def test_the_policy_sees_the_running_totals(self) -> None:
        """A budget is spent across the draw, not checked per structure."""
        sampler = OrderedStructureSampler(_make_dataset([2, 2, 2, 2]), max_batch_size=1)
        sampler.initial_batch()

        drawn = sampler.draw(fits=WithinBudget(atoms=5))

        assert _served_sizes(drawn) == [2, 2]
        assert sampler.next_row == 3

    def test_limit_caps_the_draw(self) -> None:
        """A draw serves at most *limit* structures however many fit."""
        sampler = OrderedStructureSampler(_make_dataset(), max_batch_size=1)
        sampler.initial_batch()

        assert len(sampler.draw(limit=3)) == 3
        assert sampler.next_row == 4

    def test_nothing_that_fits_hands_back_nothing(self) -> None:
        """A skipping draw that reaches no structure small enough returns empty."""
        sampler = OrderedStructureSampler(_make_dataset([2, 8, 9]), max_batch_size=1)
        sampler.initial_batch()

        assert sampler.draw(fits=WithinBudget(atoms=1), on_miss="skip") == []
        assert sampler.exhausted

    def test_a_custom_policy_decides_the_fit(self) -> None:
        """Any predicate over the totals is a policy, not only an atom or edge bound."""
        sampler = OrderedStructureSampler(_make_dataset([2, 3, 4, 5]), max_batch_size=1)
        sampler.initial_batch()

        drawn = sampler.draw(fits=_EstimatedMemory(bytes_per_atom=16, budget=120))

        assert _served_sizes(drawn) == [3, 4]


class TestOrderedStructureSamplerShard:
    def test_rows_are_dealt_strided_and_disjoint(self) -> None:
        """Rank r takes every world-th row from offset r, unpadded and unshuffled."""
        sampler = OrderedStructureSampler(_make_dataset())

        sampler.shard(1, 2)

        assert sampler.rows == (1, 3)
        assert len(sampler) == 2

    def test_a_single_rank_run_owns_the_whole_dataset(self) -> None:
        """The strided deal degenerates to the dataset itself on one process."""
        sampler = OrderedStructureSampler(_make_dataset())

        sampler.shard(0, 1)

        assert sampler.rows == tuple(range(5))

    def test_exhaustion_counts_shard_positions(self) -> None:
        """A rank is done when *its* rows are gone, not when the dataset's are."""
        sampler = OrderedStructureSampler(_make_dataset())
        sampler.shard(1, 2)

        sampler.initial_batch()

        assert sampler.exhausted
        assert len(sampler) == 2

    def test_two_ranks_draw_disjoint_rows_covering_the_set(self) -> None:
        """A structure served to a rank that does not own it is propagated twice."""
        dataset = _make_dataset()
        served: list[list[int]] = []
        for rank in (0, 1):
            sampler = OrderedStructureSampler(dataset, max_batch_size=1)
            sampler.shard(rank, 2)
            state = sampler.initial_batch()
            served.append([int(state.num_nodes)] + _served_sizes(sampler.draw()))

        assert set(served[0]).isdisjoint(served[1])
        assert sorted(served[0] + served[1]) == sorted(_SIZES)

    def test_a_rank_outside_its_world_is_rejected(self) -> None:
        """A shard is dealt to one rank of a world, so it has to name a position."""
        sampler = OrderedStructureSampler(_make_dataset())

        with pytest.raises(ValueError, match="rank=2 of world_size=2"):
            sampler.shard(2, 2)

    def test_a_shard_dealt_to_no_rows_cannot_be_probed(self) -> None:
        """A probe needs one row, and a rank past the dataset's length owns none."""
        sampler = OrderedStructureSampler(_make_dataset([2]))
        sampler.shard(1, 2)

        with pytest.raises(ValueError, match="leaves rank=1 empty"):
            sampler.probe()

    def test_installing_a_shard_reopens_the_sampler(self) -> None:
        """A rerun restarts the pass, so the sampler opens at its first row again."""
        sampler = OrderedStructureSampler(_make_dataset())
        sampler.initial_batch()

        sampler.shard(0, 1)

        assert (sampler.next_row, sampler.next_system_id) == (0, 0)


class TestOrderedStructureSamplerState:
    def test_the_position_round_trips_through_a_state_dict(self) -> None:
        """A restart resumes the row and the next id."""
        sampler = OrderedStructureSampler(_make_dataset(), max_batch_size=1)
        sampler.initial_batch()
        sampler.draw(limit=2)

        restored = OrderedStructureSampler(_make_dataset(), max_batch_size=1)
        restored.load_state_dict(sampler.state_dict())

        assert restored.state_dict() == sampler.state_dict()
        assert (restored.next_row, restored.next_system_id) == (3, 3)

    def test_the_state_dict_records_next_row(self) -> None:
        """The position travels under ``next_row`` and nothing else."""
        sampler = OrderedStructureSampler(_make_dataset(), max_batch_size=1)
        sampler.initial_batch()

        assert sampler.state_dict() == {
            "next_row": 1,
            "next_system_id": 1,
            "rank": 0,
            "world_size": 1,
        }

    def test_a_restored_sampler_resumes_at_its_row_not_at_its_ids(self) -> None:
        """Ids skip the structures a policy passed over, so they name no row."""
        sizes = [2, 9, 3, 4]
        sampler = OrderedStructureSampler(_make_dataset(sizes), max_batch_size=1)
        sampler.initial_batch()
        sampler.draw(limit=1, fits=WithinBudget(atoms=5), on_miss="skip")
        state = sampler.state_dict()

        restored = OrderedStructureSampler(_make_dataset(sizes), max_batch_size=1)
        restored.load_state_dict(state)

        assert state["next_system_id"] < state["next_row"]
        assert _served_sizes(restored.draw(limit=1)) == _served_sizes(
            sampler.draw(limit=1)
        )

    def test_a_bundle_from_another_shard_is_refused(self) -> None:
        """A position counts rows in one rank's shard and no other."""
        sampler = OrderedStructureSampler(_make_dataset())
        sampler.shard(0, 1)

        with pytest.raises(ValueError, match="written for rank 1 of 2"):
            sampler.load_state_dict(
                {"next_row": 0, "next_system_id": 0, "rank": 1, "world_size": 2}
            )

    def test_a_bundle_under_the_former_cursor_key_is_refused(self) -> None:
        """The legacy key is not aliased; the refusal names the key now read."""
        sampler = OrderedStructureSampler(_make_dataset())

        with pytest.raises(KeyError, match="next_row"):
            sampler.load_state_dict(
                {"cursor": 2, "next_system_id": 2, "rank": 0, "world_size": 1}
            )
        assert sampler.next_row == 0
