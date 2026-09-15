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
"""Tests for :mod:`nvalchemi.training.distillation.config`."""

from __future__ import annotations

from typing import Any

import pytest
import torch
from pydantic import ValidationError

from nvalchemi.dynamics import FusedStage
from nvalchemi.dynamics.base import ConvergenceHook
from nvalchemi.dynamics.demo import DemoDynamics
from nvalchemi.dynamics.integrators.nve import NVE
from nvalchemi.dynamics.optimizers.fire import FIRE
from nvalchemi.dynamics.sampler import SizeAwareSampler
from nvalchemi.training.distillation import (
    InProcessTeacherScorer,
    OnPolicyConfig,
    OnPolicyKnobs,
    SeedSource,
)
from test.training.conftest import _build_demo_model
from test.training.distillation.conftest import (
    _build_atom_only_dataset,
    _build_small_dataset,
)

_OBJECT_FIELDS = frozenset({"dynamics", "teacher_scorer", "seeds", "convergence_hook"})
"""The whole of what a live segment loop adds to the declarative knobs."""


def _make_knob_kwargs(**overrides: Any) -> dict[str, Any]:
    """Return a minimal valid ``OnPolicyKnobs`` payload with *overrides* applied."""
    kwargs: dict[str, Any] = {"replay_ratio": 0.25, "steps_per_segment": 4}
    kwargs.update(overrides)
    return kwargs


def _make_config_kwargs(**overrides: Any) -> dict[str, Any]:
    """Return a minimal valid ``OnPolicyConfig`` payload with *overrides* applied."""
    kwargs: dict[str, Any] = {
        "dynamics": DemoDynamics(_build_demo_model(), n_steps=10, dt=0.5),
        "teacher_scorer": InProcessTeacherScorer(
            _build_demo_model(), ["energy", "forces"]
        ),
        "seeds": SeedSource(_build_small_dataset()),
        "replay_ratio": 0.25,
        "steps_per_segment": 4,
    }
    kwargs.update(overrides)
    return kwargs


class TestOnPolicyKnobs:
    def test_a_plain_dict_is_all_a_pre_flight_needs(self) -> None:
        """No propagator, no teacher, no store: the scalars validate on their own."""
        knobs = OnPolicyKnobs.model_validate(_make_knob_kwargs())

        assert knobs.batch_size == 8
        assert knobs.segment_steps == 100
        assert knobs.label_frequency == 100
        assert knobs.replay_capacity is None
        assert knobs.replay_eviction == "fifo"
        assert knobs.replay_device is None
        assert knobs.convergence is None
        assert knobs.weight_sync_frequency == 1

    def test_the_knobs_round_trip_through_json(self) -> None:
        """A recipe carries the dumped scalars and rebuilds the same knobs."""
        knobs = OnPolicyKnobs(
            **_make_knob_kwargs(replay_device="cpu", convergence=0.05)
        )

        assert OnPolicyKnobs.model_validate(knobs.model_dump(mode="json")) == knobs

    def test_a_torch_device_is_read_back_as_its_name(self) -> None:
        """``replay_device`` is a string knob a recipe can carry as it stands."""
        knobs = OnPolicyKnobs(**_make_knob_kwargs(replay_device=torch.device("cpu")))

        assert knobs.replay_device == "cpu"

    def test_the_config_adds_live_objects_and_nothing_else(self) -> None:
        """Every scalar belongs to the knobs, so a pre-flight sees all of them."""
        assert (
            set(OnPolicyConfig.model_fields) - set(OnPolicyKnobs.model_fields)
            == _OBJECT_FIELDS
        )

    @pytest.mark.parametrize(
        "overrides",
        [
            {"replay_ratio": -0.1},
            {"replay_ratio": 1.5},
            {"segment_steps": 0},
            {"steps_per_segment": 0},
            {"batch_size": 0},
            {"label_frequency": 0},
            {"replay_capacity": 0},
            {"convergence": 0.0},
            {"replay_eviction": "oldest"},
            {"unknown_knob": 1},
        ],
        ids=[
            "negative_ratio",
            "ratio_above_one",
            "zero_segment_steps",
            "zero_training_steps",
            "zero_batch_size",
            "zero_label_frequency",
            "zero_replay_capacity",
            "zero_convergence",
            "unknown_eviction",
            "extra_field",
        ],
    )
    def test_out_of_range_knobs_are_rejected(self, overrides: dict[str, Any]) -> None:
        """Every declarative constraint fails at construction, not mid-run."""
        with pytest.raises(ValidationError):
            OnPolicyKnobs(**_make_knob_kwargs(**overrides))

    def test_uncertainty_eviction_is_rejected(self) -> None:
        """The reserved policy fails here, not after a segment of teacher passes."""
        with pytest.raises(ValidationError, match="reserved for committee-based"):
            OnPolicyKnobs(**_make_knob_kwargs(replay_eviction="uncertainty"))

    def test_weight_sync_frequency_above_one_raises(self) -> None:
        """The reserved sync knob is held at 1 while the propagator shares a module."""
        with pytest.raises(ValidationError, match="weight_sync_frequency must be 1"):
            OnPolicyKnobs(**_make_knob_kwargs(weight_sync_frequency=2))

    def test_a_zero_replay_ratio_is_rejected(self) -> None:
        """Generating frames no batch ever draws is offline training with extra steps."""
        with pytest.raises(ValidationError, match="drop on_policy"):
            OnPolicyKnobs(**_make_knob_kwargs(replay_ratio=0.0))

    @pytest.mark.parametrize(
        ("replay_ratio", "batch_size"),
        [(0.05, 8), (0.95, 8)],
        ids=["replay_rounds_away", "reference_rounds_away"],
    )
    def test_a_ratio_that_rounds_a_source_out_of_the_batch_is_rejected(
        self, replay_ratio: float, batch_size: int
    ) -> None:
        """The mixture is whole samples, so the ratio only means something with the size."""
        with pytest.raises(ValidationError, match="leaves one source out of training"):
            OnPolicyKnobs(
                **_make_knob_kwargs(replay_ratio=replay_ratio, batch_size=batch_size)
            )

    def test_the_rejected_batch_size_names_one_that_works(self) -> None:
        """The rejection's own remedy constructs instead of raising the same error."""
        with pytest.raises(ValidationError, match="raise batch_size to at least 11"):
            OnPolicyKnobs(**_make_knob_kwargs(replay_ratio=0.95, batch_size=10))

        knobs = OnPolicyKnobs(**_make_knob_kwargs(replay_ratio=0.95, batch_size=11))

        assert knobs.batch_size == 11


class TestOnPolicyConfigComposition:
    def test_the_knobs_property_matches_a_standalone_build(self) -> None:
        """A knob is validated identically standalone and composed."""
        config = OnPolicyConfig(**_make_config_kwargs(segment_steps=7))

        assert config.knobs == OnPolicyKnobs(**_make_knob_kwargs(segment_steps=7))

    def test_a_bare_dataset_is_wrapped_in_an_unbudgeted_source(self) -> None:
        """Seeding from a dataset whole is the 90% case and stays silent."""
        dataset = _build_small_dataset()

        config = OnPolicyConfig(**_make_config_kwargs(seeds=dataset))

        assert isinstance(config.seeds, SeedSource)
        assert config.seeds.dataset is dataset

    def test_relaxation_optimizer_is_accepted_as_the_propagator(self) -> None:
        """The knob is ``dynamics``, so a FIRE relaxation drives the loop too."""
        propagator = FIRE(_build_demo_model(), dt=0.1, n_steps=10)

        config = OnPolicyConfig(**_make_config_kwargs(dynamics=propagator))

        assert config.dynamics is propagator

    def test_scorer_must_satisfy_the_teacher_scorer_protocol(self) -> None:
        """A stand-in without ``label`` and ``signals`` is not a scorer."""
        with pytest.raises(ValidationError):
            OnPolicyConfig(**_make_config_kwargs(teacher_scorer=object()))

    def test_async_knobs_are_not_configurable_yet(self) -> None:
        """``async_mode`` and ``staleness_threshold`` land with the remote scorer."""
        with pytest.raises(ValidationError):
            OnPolicyConfig(**_make_config_kwargs(async_mode=True))

    def test_seeds_missing_a_propagator_field_are_rejected_at_construction(
        self,
    ) -> None:
        """A missing ``forces`` surfaces here, not from inside the first kernel."""
        with pytest.raises(ValidationError, match="propagates from"):
            OnPolicyConfig(
                **_make_config_kwargs(seeds=SeedSource(_build_atom_only_dataset()))
            )

    def test_recycling_without_a_criterion_is_rejected(self) -> None:
        """Only a run managing a trajectory lifecycle ever backfills."""
        seeds = SeedSource(_build_small_dataset(), recycle=True)

        with pytest.raises(ValidationError, match="ever backfills"):
            OnPolicyConfig(**_make_config_kwargs(seeds=seeds))

    def test_a_multi_sub_stage_fused_propagator_is_rejected(self) -> None:
        """Every non-last sub-stage carries a migrator the lifecycle does not own."""
        student = _build_demo_model()
        propagator = FusedStage(
            sub_stages=[(0, FIRE(student, dt=0.1)), (1, NVE(student, dt=0.1))]
        )

        with pytest.raises(
            ValidationError, match="no other status-migrating ConvergenceHook"
        ):
            OnPolicyConfig(**_make_config_kwargs(dynamics=propagator, convergence=1e-6))

    def test_a_threshold_builds_the_criterion_the_lifecycle_drives(self) -> None:
        """The shorthand migrates 0 to the propagator's own exit status."""
        config = OnPolicyConfig(**_make_config_kwargs(convergence=0.05))

        criterion = config.convergence_criterion

        assert criterion is config.convergence_criterion
        assert criterion.source_status == 0
        assert criterion.target_status == config.dynamics.exit_status

    def test_a_hook_passed_whole_is_the_criterion(self) -> None:
        """A live criterion is runtime-only, and stands for itself."""
        hook = ConvergenceHook.from_fmax(0.05, source_status=0, target_status=1)

        config = OnPolicyConfig(**_make_config_kwargs(convergence_hook=hook))

        assert config.convergence_criterion is hook

    def test_a_threshold_and_a_hook_together_are_rejected(self) -> None:
        """Two spellings of one criterion, so exactly one of them names it."""
        hook = ConvergenceHook.from_fmax(0.05, source_status=0, target_status=1)

        with pytest.raises(ValidationError, match="two spellings of one"):
            OnPolicyConfig(
                **_make_config_kwargs(convergence=0.05, convergence_hook=hook)
            )

    def test_a_hook_that_migrates_no_status_is_rejected(self) -> None:
        """A graph graduates out of the batch on its status, so one has to move."""
        hook = ConvergenceHook.from_fmax(0.05)

        with pytest.raises(ValidationError, match="has to migrate status"):
            OnPolicyConfig(**_make_config_kwargs(convergence_hook=hook))

    def test_a_hook_that_migrates_short_of_the_exit_status_is_rejected(self) -> None:
        """Converged graphs must reach the status that graduates them."""
        hook = ConvergenceHook.from_fmax(0.05, source_status=0, target_status=0)

        with pytest.raises(ValidationError, match="at least the propagator's"):
            OnPolicyConfig(**_make_config_kwargs(convergence_hook=hook))

    def test_a_hook_that_does_not_run_every_step_is_rejected(self) -> None:
        """A structure is captured and frozen on the step it converges."""
        hook = ConvergenceHook.from_fmax(
            0.05, source_status=0, target_status=1, frequency=2
        )

        with pytest.raises(ValidationError, match="has to run on every"):
            OnPolicyConfig(**_make_config_kwargs(convergence_hook=hook))


class TestOnPolicyConfigDeprecations:
    def test_seed_dataset_maps_onto_an_unbudgeted_source(self) -> None:
        """The pre-SeedSource spelling keeps working while call sites migrate."""
        dataset = _build_small_dataset()

        with pytest.warns(DeprecationWarning, match="seed_dataset is now"):
            config = OnPolicyConfig(
                **_make_config_kwargs(seeds=None, seed_dataset=dataset)
            )

        assert config.seeds.dataset is dataset

    def test_a_sampler_maps_onto_an_equivalent_source(self) -> None:
        """The sampler's dataset and budgets describe a source exactly."""
        dataset = _build_small_dataset()
        sampler = SizeAwareSampler(dataset, max_atoms=12, max_batch_size=3)

        with pytest.warns(DeprecationWarning, match="takes a SeedSource"):
            config = OnPolicyConfig(**_make_config_kwargs(seeds=None, sampler=sampler))

        assert config.seeds.dataset is dataset
        assert config.seeds.max_batch_size == 3

    def test_recycle_seeds_sets_the_flag_on_the_source(self) -> None:
        """The cursor that wraps lives on the source, so the flag does too."""
        with pytest.warns(DeprecationWarning, match="recycle_seeds is now"):
            config = OnPolicyConfig(
                **_make_config_kwargs(recycle_seeds=True, convergence=0.05)
            )

        assert config.seeds.recycle

    def test_a_hook_valued_convergence_moves_to_the_runtime_field(self) -> None:
        """The threshold is what a recipe carries; the hook is runtime-only."""
        hook = ConvergenceHook.from_fmax(0.05, source_status=0, target_status=1)

        with pytest.warns(DeprecationWarning, match="convergence is the fmax"):
            config = OnPolicyConfig(**_make_config_kwargs(convergence=hook))

        assert config.convergence is None
        assert config.convergence_hook is hook

    def test_a_sampler_alongside_a_seed_dataset_raises(self) -> None:
        """A sampler brings its own dataset, so the seed dataset would be dead."""
        sampler = SizeAwareSampler(
            _build_small_dataset(), max_atoms=64, max_batch_size=4
        )

        with pytest.raises(ValidationError, match="Exactly one of seed_dataset"):
            OnPolicyConfig(
                **_make_config_kwargs(
                    seeds=None,
                    seed_dataset=_build_small_dataset(),
                    sampler=sampler,
                )
            )

    def test_a_config_without_any_seed_source_raises(self) -> None:
        """The loop has to be told what to propagate from."""
        with pytest.raises(ValidationError, match="seeds"):
            OnPolicyConfig(**_make_config_kwargs(seeds=None))
