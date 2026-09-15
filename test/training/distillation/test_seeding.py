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
import torch
from pydantic import ValidationError

from nvalchemi.data import AtomicData, Batch
from nvalchemi.data.datapipes.backends.zarr import (
    AtomicDataZarrReader,
    AtomicDataZarrWriter,
)
from nvalchemi.data.datapipes.dataset import Dataset
from nvalchemi.data.datapipes.in_memory_dataset import InMemoryDataset
from nvalchemi.dynamics.base import ConvergenceHook
from nvalchemi.dynamics.demo import DemoDynamics
from nvalchemi.dynamics.optimizers.fire import FIRE, FIREVariableCell
from nvalchemi.dynamics.sampler import SizeAwareSampler
from nvalchemi.training.distillation import SeedSource
from nvalchemi.training.distillation.seeding import (
    _check_seed_fields,
    _check_seed_status,
    _seed_field_requirements,
    _SeedSourceSpec,
)
from test.training.conftest import _build_atomic_data, _build_demo_model
from test.training.distillation.conftest import (
    _build_periodic_batch,
    _build_small_dataset,
)

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


def _served_sizes(replacements: list[AtomicData]) -> list[int]:
    """Return the atom count of every structure a request handed back."""
    return [int(data.positions.shape[0]) for data in replacements]


def _make_store(tmp_path: Path, sizes: Sequence[int] = (2, 3, 4)) -> Dataset:
    """Return a path-backed dataset, which is the only kind a recipe can name."""
    store = tmp_path / "seeds.zarr"
    AtomicDataZarrWriter(store).write(_make_dataset(sizes).in_memory_batch)
    return Dataset(reader=AtomicDataZarrReader(store))


def _make_variable_cell_seed() -> Batch:
    """Return a periodic seed batch carrying what a variable-cell FIRE reads."""
    batch = _build_periodic_batch(n_systems=2, n_atoms=4)
    batch["forces"] = torch.zeros(batch.num_nodes, 3)
    batch["stress"] = torch.zeros(batch.num_graphs, 3, 3)
    return batch


class _RenamedForceFIRE(FIRE):
    """FIRE writing the model's forces to a batch field of its own naming."""

    _OUTPUT_KEY_TO_BATCH_ATTR = {"forces": "reference_forces"}


class TestSeedSourceCursor:
    def test_an_unbudgeted_source_seeds_every_row_it_owns(self) -> None:
        """A bare seed dataset is propagated whole, which is today's behavior."""
        source = SeedSource(_build_small_dataset())

        state = source.initial_batch()

        assert state.num_graphs == 5
        assert source.cursor == 5

    def test_the_cursor_opens_past_the_seeded_batch(self) -> None:
        """A budgeted source leaves the rows it did not pack for the backfill."""
        source = SeedSource(_build_small_dataset(), max_batch_size=2)

        state = source.initial_batch()

        assert state.num_graphs == 2
        assert source.cursor == 2

    def test_an_unbudgeted_source_hands_out_nothing(self) -> None:
        """The initial batch consumed every row, so a refill has no remainder."""
        source = SeedSource(_build_small_dataset())
        source.initial_batch()

        assert source.request_replacements_budget() == []
        assert source.exhausted

    def test_an_unbudgeted_initial_batch_records_the_envelope(self) -> None:
        """The seeded batch is the size a backfill may never widen past."""
        source = SeedSource(_build_small_dataset())

        state = source.initial_batch()

        assert source.max_atoms == int(state.num_nodes)
        assert source.max_batch_size == state.num_graphs
        assert source.max_edges is None

    def test_a_declared_budget_survives_seeding(self) -> None:
        """A source the caller sized keeps that size rather than the batch's."""
        source = SeedSource(_build_small_dataset(), max_atoms=64, max_batch_size=2)

        source.initial_batch()

        assert (source.max_atoms, source.max_batch_size) == (64, 2)

    def test_an_over_budget_structure_is_skipped_not_blocking(self) -> None:
        """A large structure at the cursor must not starve every refill behind it."""
        source = SeedSource(_make_dataset([3, 8, 2]), max_batch_size=1)
        source.initial_batch()

        replacements = source.request_replacements_budget(atom_budget=4, max_count=1)

        assert _served_sizes(replacements) == [2]

    def test_nothing_that_fits_hands_back_nothing(self) -> None:
        """A pass that reaches no structure small enough returns empty."""
        source = SeedSource(_make_dataset([2, 8, 9]), max_batch_size=1)
        source.initial_batch()

        assert source.request_replacements_budget(atom_budget=1) == []

    def test_a_recycling_scan_stops_after_one_pass(self) -> None:
        """A wrap mid-scan must not serve one structure twice in one call."""
        source = SeedSource(_make_dataset([2, 2, 2]), max_batch_size=1, recycle=True)
        source.initial_batch()

        replacements = source.request_replacements_budget(max_count=10)

        assert len(replacements) == 3

    def test_a_skipped_structure_still_spends_the_one_pass_quota(self) -> None:
        """A skip that did not count would let a wrap serve a row twice in one call."""
        source = SeedSource(
            _make_dataset([4, 12, 13, 14]), max_batch_size=1, recycle=True
        )
        source.initial_batch()

        replacements = source.request_replacements_budget(atom_budget=15, max_count=2)

        assert _served_sizes(replacements) == [12]

    def test_two_requests_never_serve_one_structure_twice(self) -> None:
        """The cursor is shared, so a second request opens where the first stopped."""
        source = SeedSource(_build_small_dataset(), max_batch_size=1)
        source.initial_batch()

        first = source.request_replacements_budget(max_count=2)
        second = source.request_replacements_budget(max_count=2)

        assert _served_sizes(first) == [3, 4]
        assert _served_sizes(second) == [5, 6]

    def test_backfilled_structures_continue_the_seeded_numbering(self) -> None:
        """Ids number the trajectories the run started, seeded and backfilled alike."""
        source = SeedSource(_build_small_dataset(), max_batch_size=2)
        state = source.initial_batch()

        replacements = source.request_replacements_budget(max_count=1)

        assert state["system_id"].view(-1).tolist() == [0, 1]
        assert int(replacements[0].system_id.view(-1)[0]) == 2

    def test_a_seed_batch_arrives_without_the_previous_run_bookkeeping(self) -> None:
        """Status describes the run that wrote it, so the source installs its own."""
        dataset = _build_small_dataset()
        frames = dataset.in_memory_batch
        frames.add_key(
            "status",
            [torch.full((1, 1), 3, dtype=torch.long) for _ in range(5)],
            level="system",
        )

        state = SeedSource(InMemoryDataset(in_memory_batch=frames)).initial_batch()

        assert state["status"].view(-1).tolist() == [0] * 5

    def test_a_budget_that_fits_nothing_is_rejected(self) -> None:
        """A run has to propagate something, and says so before it starts."""
        source = SeedSource(_make_dataset([9, 9]), max_atoms=4)

        with pytest.raises(ValueError, match="has to propagate something"):
            source.initial_batch()


class TestSeedSourceShard:
    def test_rows_are_dealt_strided_and_disjoint(self) -> None:
        """Rank r takes every world-th row from offset r, unpadded and unshuffled."""
        source = SeedSource(_build_small_dataset())

        source.shard(1, 2)

        assert source.rows == (1, 3)
        assert len(source) == 2

    def test_a_single_rank_run_owns_the_whole_dataset(self) -> None:
        """The strided deal degenerates to the dataset itself on one process."""
        source = SeedSource(_build_small_dataset())

        source.shard(0, 1)

        assert source.rows == tuple(range(5))

    def test_exhaustion_counts_shard_positions(self) -> None:
        """A rank is done when *its* rows are gone, not when the dataset's are."""
        source = SeedSource(_build_small_dataset())
        source.shard(1, 2)

        source.initial_batch()

        assert source.exhausted
        assert len(source) == 2

    def test_two_ranks_backfill_disjoint_rows_covering_the_set(self) -> None:
        """A structure served to a rank that does not own it is propagated twice."""
        dataset = _build_small_dataset()
        served: list[list[int]] = []
        for rank in (0, 1):
            source = SeedSource(dataset, max_batch_size=1)
            source.shard(rank, 2)
            state = source.initial_batch()
            served.append(
                [int(state.num_nodes)]
                + _served_sizes(source.request_replacements_budget())
            )

        assert set(served[0]).isdisjoint(served[1])
        assert sorted(served[0] + served[1]) == sorted(_SHARD_SIZES)

    def test_a_rank_outside_its_world_is_rejected(self) -> None:
        """A shard is dealt to one rank of a world, so it has to name a position."""
        source = SeedSource(_build_small_dataset())

        with pytest.raises(ValueError, match="rank=2 of world_size=2"):
            source.shard(2, 2)

    def test_installing_a_shard_reopens_the_cursor(self) -> None:
        """A rerun reseeds the trajectory, so the source opens at its front again."""
        source = SeedSource(_build_small_dataset())
        source.initial_batch()

        source.shard(0, 1)

        assert (source.cursor, source.next_system_id) == (0, 0)
        assert source.max_atoms is None


class TestSeedSourceRecycle:
    def test_a_recycled_cursor_wraps_inside_its_own_shard(self) -> None:
        """Recycling restarts the rank's rows, never its neighbor's."""
        source = SeedSource(_build_small_dataset(), max_batch_size=1, recycle=True)
        source.shard(1, 2)
        source.initial_batch()

        served = _served_sizes(source.request_replacements_budget(max_count=3))

        assert source.wraps == 1
        assert set(served) <= {3, 5}

    def test_a_recycling_source_never_reports_itself_exhausted(self) -> None:
        """The batch keeps its trajectory count instead of narrowing away."""
        source = SeedSource(_build_small_dataset(), recycle=True)
        source.initial_batch()

        assert not source.exhausted


class TestSeedSourceState:
    def test_the_cursor_round_trips_through_a_state_dict(self) -> None:
        """A restart resumes the position, the wrap count, and the next id."""
        source = SeedSource(_build_small_dataset(), max_batch_size=1)
        source.initial_batch()
        source.request_replacements_budget(max_count=2)

        restored = SeedSource(_build_small_dataset(), max_batch_size=1)
        restored.load_state_dict(source.state_dict())

        assert restored.state_dict() == source.state_dict()
        assert (restored.cursor, restored.next_system_id) == (3, 3)

    def test_a_restored_source_resumes_at_its_cursor_not_at_its_ids(self) -> None:
        """Ids skip the structures a budget passed over, so they name no row."""
        sizes = [2, 9, 3, 4]
        source = SeedSource(_make_dataset(sizes), max_batch_size=1, recycle=True)
        source.initial_batch()
        source.request_replacements_budget(atom_budget=5, max_count=1)
        state = source.state_dict()

        restored = SeedSource(_make_dataset(sizes), max_batch_size=1, recycle=True)
        restored.load_state_dict(state)

        assert state["next_system_id"] < state["cursor"]
        assert _served_sizes(
            restored.request_replacements_budget(max_count=1)
        ) == _served_sizes(source.request_replacements_budget(max_count=1))

    def test_a_restored_source_keeps_the_envelope_the_seeds_established(self) -> None:
        """A run restored after a graduation refills under the width it started at."""
        sizes = [2, 6, 2]
        source = SeedSource(_make_dataset(sizes), recycle=True)
        dynamics = DemoDynamics(_build_demo_model(), n_steps=1, dt=0.5)
        state = source.initial_batch()
        dynamics.sampler = source
        state = dynamics.run(state, n_steps=1)
        state["status"][1] = dynamics.exit_status
        state = dynamics.refill_check(state, dynamics.exit_status)
        state["status"][0] = dynamics.exit_status

        restored = SeedSource(_make_dataset(sizes), recycle=True)
        restored.load_state_dict(source.state_dict())
        restored.record_envelope(state)
        dynamics.sampler = restored
        refilled = dynamics.refill_check(state, dynamics.exit_status)

        assert (restored.max_atoms, restored.max_batch_size) == (10, 3)
        assert sorted(int(n) for n in refilled.num_nodes_per_graph) == [2, 2, 6]

    def test_a_declared_budget_is_left_out_of_the_bundle(self) -> None:
        """The envelope is state only where the caller declared no budget at all."""
        source = SeedSource(_build_small_dataset(), max_batch_size=1)
        source.initial_batch()

        bundle = source.state_dict()

        assert "max_atoms" not in bundle and "max_batch_size" not in bundle

    def test_a_budgeted_source_ignores_the_envelope_a_bundle_carries(self) -> None:
        """A recipe that declared a budget outranks the envelope a stale bundle holds."""
        seeded = SeedSource(_make_dataset([2, 6, 2]))
        seeded.initial_batch()

        restored = SeedSource(_make_dataset([2, 6, 2]), max_atoms=4)
        restored.load_state_dict(seeded.state_dict())

        assert restored.max_atoms == 4

    def test_a_bundle_from_another_shard_is_refused(self) -> None:
        """A cursor counts positions in one rank's rows and no others."""
        source = SeedSource(_build_small_dataset())
        source.shard(0, 1)

        with pytest.raises(ValueError, match="written for rank 1 of 2"):
            source.load_state_dict(
                {
                    "cursor": 0,
                    "wraps": 0,
                    "next_system_id": 0,
                    "rank": 1,
                    "world_size": 2,
                }
            )


class TestSeedSourceSpec:
    def test_a_path_backed_source_round_trips_through_a_spec(
        self, tmp_path: Path
    ) -> None:
        """A recipe names the store the seeds are read from and the budgets set."""
        source = SeedSource(
            _make_store(tmp_path), max_atoms=32, max_batch_size=2, recycle=True
        )

        rebuilt = SeedSource.from_spec_dict(source.to_spec_dict())

        assert set(source.to_spec_dict()) == set(_SeedSourceSpec.model_fields)
        assert rebuilt.to_spec_dict() == source.to_spec_dict()
        assert (rebuilt.max_atoms, rebuilt.max_batch_size, rebuilt.recycle) == (
            32,
            2,
            True,
        )
        assert len(rebuilt) == 3

    def test_a_recorded_envelope_never_reaches_the_spec(self, tmp_path: Path) -> None:
        """The envelope is state a run measured, not configuration a recipe set."""
        source = SeedSource(_make_store(tmp_path))
        source.initial_batch()

        assert source.to_spec_dict()["max_atoms"] is None

    def test_a_flag_spelled_as_a_string_is_read_as_the_boolean_it_spells(
        self, tmp_path: Path
    ) -> None:
        """A recipe carrying its flags as text still says what it means."""
        spec = SeedSource(_make_store(tmp_path)).to_spec_dict()

        spec["recycle"] = "true"
        assert SeedSource.from_spec_dict(spec).recycle is True
        spec["recycle"] = "false"
        assert SeedSource.from_spec_dict(spec).recycle is False

    def test_a_flag_nothing_reads_as_a_boolean_is_refused(self, tmp_path: Path) -> None:
        """A recycling run needs a lifecycle, so the flag must not be guessed at."""
        spec = SeedSource(_make_store(tmp_path)).to_spec_dict()
        spec["recycle"] = "maybe"

        with pytest.raises(ValidationError):
            SeedSource.from_spec_dict(spec)

    def test_a_misspelled_budget_is_refused_by_name(self, tmp_path: Path) -> None:
        """A budget that reaches no field leaves the run silently unbudgeted."""
        spec = SeedSource(_make_store(tmp_path)).to_spec_dict()
        spec["max_atom"] = 10

        with pytest.raises(ValidationError, match="max_atom"):
            SeedSource.from_spec_dict(spec)

    def test_a_non_positive_budget_is_refused(self, tmp_path: Path) -> None:
        """A budget bounds a batch, so it has to name a count a batch can hold."""
        spec = SeedSource(_make_store(tmp_path)).to_spec_dict()
        spec["max_atoms"] = -5

        with pytest.raises(ValidationError):
            SeedSource.from_spec_dict(spec)

    def test_a_non_numeric_budget_is_refused(self, tmp_path: Path) -> None:
        """A budget the refill subtracts atom counts from cannot be a word."""
        spec = SeedSource(_make_store(tmp_path)).to_spec_dict()
        spec["max_batch_size"] = "four"

        with pytest.raises(ValidationError):
            SeedSource.from_spec_dict(spec)

    def test_a_dataset_reference_without_a_path_is_refused(
        self, tmp_path: Path
    ) -> None:
        """A store a recipe forgot to name is a recipe error, not a raw KeyError."""
        spec = SeedSource(_make_store(tmp_path)).to_spec_dict()
        del spec["dataset"]["path"]

        with pytest.raises(ValidationError):
            SeedSource.from_spec_dict(spec)

    def test_an_in_memory_source_cannot_be_named(self) -> None:
        """A spec references a dataset by the store it reads."""
        source = SeedSource(_build_small_dataset())

        with pytest.raises(ValueError, match="OnPolicyConfig.seeds is a"):
            source.to_spec_dict()

    def test_from_sampler_carries_the_dataset_and_the_budgets_over(self) -> None:
        """The sampler is an input to the source, not a live delegate behind it."""
        dataset = _build_small_dataset()
        sampler = SizeAwareSampler(dataset, max_atoms=12, max_batch_size=3)

        with pytest.warns(DeprecationWarning, match="takes a SeedSource"):
            source = SeedSource.from_sampler(sampler)

        assert source.dataset is dataset
        assert (source.max_atoms, source.max_batch_size) == (12, 3)
        assert source.initial_batch().num_graphs == 3


class TestSeedSourceRefillContract:
    def test_a_source_backfills_the_graph_a_propagator_graduated(self) -> None:
        """The five members refill_check reads are answered by the seed source."""
        source = SeedSource(_make_dataset([2, 3, 4]), recycle=True)
        dynamics = DemoDynamics(_build_demo_model(), n_steps=1, dt=0.5)
        state = source.initial_batch()
        dynamics.sampler = source
        state = dynamics.run(state, n_steps=1)
        state["status"][0] = dynamics.exit_status

        refilled = dynamics.refill_check(state, dynamics.exit_status)

        assert refilled.num_graphs == 3
        assert source.wraps == 1

    def test_an_exhausted_source_narrows_the_batch_instead(self) -> None:
        """Without recycling the run keeps generating from what is still moving."""
        source = SeedSource(_make_dataset([2, 3, 4]))
        dynamics = DemoDynamics(_build_demo_model(), n_steps=1, dt=0.5)
        state = source.initial_batch()
        dynamics.sampler = source
        state = dynamics.run(state, n_steps=1)
        state["status"][0] = dynamics.exit_status

        refilled = dynamics.refill_check(state, dynamics.exit_status)

        assert refilled.num_graphs == 2


class TestSeedFieldRequirements:
    def test_a_fixed_cell_optimizer_reads_forces_and_the_momentum_state(self) -> None:
        """FIRE opens on forces it has not computed and on velocities it updates."""
        requirements = _seed_field_requirements(FIRE(_build_demo_model(), dt=0.1))

        assert requirements == ("atomic_masses", "forces", "velocities")

    def test_a_variable_cell_propagator_also_reads_the_cell(self) -> None:
        """A cell is state the propagator updates in place, and inverts first."""
        requirements = _seed_field_requirements(
            FIREVariableCell(_build_demo_model(), dt=0.1)
        )

        assert requirements == (
            "atomic_masses",
            "cell",
            "forces",
            "stress",
            "velocities",
        )

    def test_a_seed_batch_without_a_cell_is_rejected(self) -> None:
        """An aperiodic seed cannot start a variable-cell relaxation."""
        dynamics = FIREVariableCell(_build_demo_model(), dt=0.1)
        seed = _make_variable_cell_seed()
        del seed["cell"]

        with pytest.raises(ValueError, match="missing \\['cell'\\]"):
            _check_seed_fields(seed, dynamics)

    def test_a_periodic_seed_carrying_the_declared_fields_is_accepted(self) -> None:
        """The same batch with its cell passes, so the check is not blanket."""
        dynamics = FIREVariableCell(_build_demo_model(), dt=0.1)

        _check_seed_fields(_make_variable_cell_seed(), dynamics)

    def test_the_propagators_own_output_map_names_the_batch_field(self) -> None:
        """A propagator renaming an output is checked against the name it reads."""
        requirements = _seed_field_requirements(
            _RenamedForceFIRE(_build_demo_model(), dt=0.1)
        )

        assert "reference_forces" in requirements
        assert "forces" not in requirements


class TestSeedStatusContract:
    def _seeded_batch(self) -> Batch:
        """Return a two-system seed batch carrying the run's own bookkeeping."""
        return SeedSource(_make_dataset([3, 3])).initial_batch()

    def test_the_stamped_status_is_the_one_the_shorthand_migrates_off(self) -> None:
        """Seeds enter on status 0, which is what the fmax shorthand reads."""
        state = self._seeded_batch()

        assert state["status"].view(-1).tolist() == [0, 0]
        _check_seed_status(
            state,
            ConvergenceHook.from_fmax(0.05, source_status=0, target_status=1),
        )

    def test_a_criterion_aimed_at_an_unseeded_status_raises(self) -> None:
        """A criterion migrating off status 1 would freeze and graduate nothing."""
        state = self._seeded_batch()

        with pytest.raises(ValueError, match=r"source_status=1 against seed statuses"):
            _check_seed_status(
                state,
                ConvergenceHook.from_fmax(0.05, source_status=1, target_status=2),
            )
