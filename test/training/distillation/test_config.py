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

import warnings
from typing import Any

import pytest
import torch
from pydantic import ValidationError

from nvalchemi.data import Batch
from nvalchemi.dynamics.demo import DemoDynamics
from nvalchemi.dynamics.optimizers.fire import FIRE, FIREVariableCell
from nvalchemi.dynamics.sinks import HostMemory
from nvalchemi.models.demo import DemoModelWrapper
from nvalchemi.training.distillation import (
    FIFO,
    InitialStructures,
    InitialStructuresSource,
    InProcessTeacherScorer,
    OnPolicyConfig,
    OnPolicySettings,
)
from test.training.conftest import _build_atomic_data, _build_demo_model
from test.training.distillation.conftest import (
    _build_atom_only_dataset,
    _build_lj_teacher,
    _build_small_dataset,
    _ListSource,
)

_OBJECT_FIELDS = frozenset(
    {
        "dynamics",
        "teacher_scorer",
        "initial_structures",
        "capture_sink",
        "replay_admission",
        "convergence_hook",
    }
)
"""The whole of what a live segment loop adds to the declarative settings."""


def _make_settings_kwargs(**overrides: Any) -> dict[str, Any]:
    """Return a minimal valid ``OnPolicySettings`` payload with *overrides* applied."""
    kwargs: dict[str, Any] = {"replay_ratio": 0.25, "training_steps_per_segment": 4}
    kwargs.update(overrides)
    return kwargs


def _make_config_kwargs(**overrides: Any) -> dict[str, Any]:
    """Return a minimal valid ``OnPolicyConfig`` payload with *overrides* applied."""
    kwargs: dict[str, Any] = {
        "dynamics": DemoDynamics(_build_demo_model(), n_steps=10, dt=0.5),
        "teacher_scorer": InProcessTeacherScorer(
            _build_demo_model(), ["energy", "forces"]
        ),
        "initial_structures": InitialStructures(_build_small_dataset()),
        "replay_ratio": 0.25,
        "training_steps_per_segment": 4,
    }
    kwargs.update(overrides)
    return kwargs


def _admit_everything(frames: Batch) -> torch.Tensor:
    """Admission predicate keeping every frame."""
    return torch.ones(frames.num_graphs, dtype=torch.bool)


class _DropNewest:
    """Eviction policy retiring the frames that arrived last."""

    def select(self, buffer: Batch, incoming: Batch, capacity: int) -> torch.Tensor:  # noqa: ARG002
        """Name the newest frames past capacity."""
        return torch.arange(capacity, buffer.num_graphs)


class _StressNeedingDynamics(DemoDynamics):
    """Propagator declaring a stress it reads that a demo student never computes."""

    __needs_keys__: set[str] = {"forces", "stress"}


class _ChargeReadingWrapper(DemoModelWrapper):
    """Demo student whose forward reads a ``charges`` field nothing declared."""

    def forward(self, data: Any, **kwargs: Any) -> Any:
        """Read the undeclared field before the ordinary forward."""
        _ = data["charges"]
        return super().forward(data, **kwargs)


class _RowsOnlySource:
    """Stand-in with the seeding members but none of the cursor ones."""

    def probe(self) -> Batch:
        """Return one structure."""
        return Batch.from_data_list([_build_atomic_data(seed=3)])

    def initial_batch(self) -> Batch:
        """Return the same structure."""
        return self.probe()


class TestOnPolicySettings:
    def test_a_plain_dict_is_all_a_pre_flight_needs(self) -> None:
        """No propagator, no teacher, no store: the scalars validate on their own."""
        settings = OnPolicySettings.model_validate(_make_settings_kwargs())

        assert settings.batch_size == 8
        assert settings.generation_steps == 100
        assert settings.label_frequency == 100
        assert settings.replay_capacity is None
        assert settings.replay_eviction == "fifo"
        assert settings.replay_device is None
        assert settings.weight_sync_frequency == 1

    def test_the_settings_round_trip_through_json(self) -> None:
        """A recipe carries the dumped scalars and rebuilds the same settings."""
        settings = OnPolicySettings(
            **_make_settings_kwargs(replay_device="cpu", seed=3)
        )

        assert (
            OnPolicySettings.model_validate(settings.model_dump(mode="json"))
            == settings
        )

    def test_a_torch_device_is_read_back_as_its_name(self) -> None:
        """``replay_device`` is a string setting a recipe can carry as it stands."""
        settings = OnPolicySettings(
            **_make_settings_kwargs(replay_device=torch.device("cpu"))
        )

        assert settings.replay_device == "cpu"

    def test_the_config_adds_live_objects_and_nothing_else(self) -> None:
        """Every scalar belongs to the settings, so a pre-flight sees all of them."""
        assert (
            set(OnPolicyConfig.model_fields) - set(OnPolicySettings.model_fields)
            == _OBJECT_FIELDS
        )

    @pytest.mark.parametrize(
        "overrides",
        [
            {"replay_ratio": -0.1},
            {"replay_ratio": 1.5},
            {"generation_steps": 0},
            {"training_steps_per_segment": 0},
            {"batch_size": 0},
            {"label_frequency": 0},
            {"replay_capacity": 0},
            {"replay_eviction": "oldest"},
            {"unknown_setting": 1},
        ],
        ids=[
            "negative_ratio",
            "ratio_above_one",
            "zero_generation_steps",
            "zero_training_steps",
            "zero_batch_size",
            "zero_label_frequency",
            "zero_replay_capacity",
            "unknown_eviction",
            "extra_field",
        ],
    )
    def test_out_of_range_settings_are_rejected(
        self, overrides: dict[str, Any]
    ) -> None:
        """Every declarative constraint fails at construction, not mid-run."""
        with pytest.raises(ValidationError):
            OnPolicySettings(**_make_settings_kwargs(**overrides))

    def test_an_eviction_string_other_than_fifo_is_rejected(self) -> None:
        """The recipe spelling is ``"fifo"`` alone; a policy object is not a setting."""
        with pytest.raises(ValidationError):
            OnPolicySettings(**_make_settings_kwargs(replay_eviction="uncertainty"))
        with pytest.raises(ValidationError):
            OnPolicySettings(**_make_settings_kwargs(replay_eviction=FIFO()))

    def test_weight_sync_frequency_above_one_raises(self) -> None:
        """The reserved sync setting stays 1 while the propagator shares a module."""
        with pytest.raises(ValidationError, match="weight_sync_frequency must be 1"):
            OnPolicySettings(**_make_settings_kwargs(weight_sync_frequency=2))

    def test_a_zero_replay_ratio_is_rejected(self) -> None:
        """Generating frames no batch ever draws is offline training with extra steps."""
        with pytest.raises(ValidationError, match="drop on_policy"):
            OnPolicySettings(**_make_settings_kwargs(replay_ratio=0.0))

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
            OnPolicySettings(
                **_make_settings_kwargs(
                    replay_ratio=replay_ratio, batch_size=batch_size
                )
            )

    def test_the_rejected_batch_size_names_one_that_works(self) -> None:
        """The rejection's own remedy constructs instead of raising the same error."""
        with pytest.raises(ValidationError, match="raise batch_size to at least 11"):
            OnPolicySettings(**_make_settings_kwargs(replay_ratio=0.95, batch_size=10))

        settings = OnPolicySettings(
            **_make_settings_kwargs(replay_ratio=0.95, batch_size=11)
        )

        assert settings.batch_size == 11


class TestOnPolicyConfigComposition:
    def test_the_settings_property_matches_a_standalone_build(self) -> None:
        """A setting is validated identically standalone and composed."""
        config = OnPolicyConfig(**_make_config_kwargs(generation_steps=7))

        assert config.settings == OnPolicySettings(
            **_make_settings_kwargs(generation_steps=7)
        )

    def test_a_bare_dataset_is_wrapped_in_an_unbudgeted_source(self) -> None:
        """Seeding from a dataset whole is the 90% case and stays silent."""
        dataset = _build_small_dataset()

        config = OnPolicyConfig(**_make_config_kwargs(initial_structures=dataset))

        assert isinstance(config.initial_structures, InitialStructures)
        assert config.initial_structures.dataset is dataset

    def test_a_custom_source_passes_through_untouched(self) -> None:
        """An object implementing the protocol is the loop's source as is."""
        source = _ListSource([_build_atomic_data(seed=3)])

        config = OnPolicyConfig(**_make_config_kwargs(initial_structures=source))

        assert config.initial_structures is source
        assert isinstance(source, InitialStructuresSource)

    def test_an_object_that_is_neither_source_nor_dataset_is_refused(self) -> None:
        """The refusal names the protocol and the dataset alternative."""
        with pytest.raises(ValueError, match="InitialStructuresSource.*load_batches"):
            OnPolicyConfig(**_make_config_kwargs(initial_structures=object()))

    def test_a_source_missing_the_cursor_members_is_refused_not_wrapped(self) -> None:
        """Seeding members alone do not make a source, and there are no rows to wrap."""
        with pytest.raises(ValueError, match="InitialStructuresSource"):
            OnPolicyConfig(**_make_config_kwargs(initial_structures=_RowsOnlySource()))

    def test_capture_sink_is_a_runtime_object_outside_the_settings(self) -> None:
        """A sink rides on the config only; the declarative half never carries it."""
        sink = HostMemory(capacity=4)

        config = OnPolicyConfig(**_make_config_kwargs(capture_sink=sink))

        assert config.capture_sink is sink
        assert "capture_sink" not in OnPolicySettings.model_fields
        assert config.settings == OnPolicySettings(**_make_settings_kwargs())
        with pytest.raises(ValidationError):
            OnPolicySettings(**_make_settings_kwargs(capture_sink=sink))

    def test_a_policy_instance_rides_on_the_config_and_reads_back_as_fifo(self) -> None:
        """A custom eviction is runtime-only; the settings copy warns and records fifo."""
        policy = _DropNewest()

        config = OnPolicyConfig(**_make_config_kwargs(replay_eviction=policy))

        assert config.replay_eviction is policy
        with pytest.warns(UserWarning, match="_DropNewest instance.*record 'fifo'"):
            settings = config.settings
        assert settings.replay_eviction == "fifo"

    def test_the_fifo_object_reads_back_as_fifo_without_a_warning(self) -> None:
        """``FIFO()`` is exactly what ``"fifo"`` names, so nothing is lost."""
        config = OnPolicyConfig(**_make_config_kwargs(replay_eviction=FIFO()))

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert config.settings.replay_eviction == "fifo"

    def test_replay_admission_is_a_runtime_object_outside_the_settings(self) -> None:
        """An admission predicate never reaches the declarative half."""
        config = OnPolicyConfig(
            **_make_config_kwargs(replay_admission=_admit_everything)
        )

        assert config.replay_admission is _admit_everything
        assert "replay_admission" not in OnPolicySettings.model_fields
        assert config.settings == OnPolicySettings(**_make_settings_kwargs())

    def test_capture_sink_must_be_a_data_sink(self) -> None:
        """A list is not a sink, whatever it can append."""
        with pytest.raises(ValidationError):
            OnPolicyConfig(**_make_config_kwargs(capture_sink=[]))

    def test_relaxation_optimizer_is_accepted_as_the_propagator(self) -> None:
        """The field is ``dynamics``, so a FIRE relaxation drives the loop too."""
        propagator = FIRE(_build_demo_model(), dt=0.1, n_steps=10)

        config = OnPolicyConfig(**_make_config_kwargs(dynamics=propagator))

        assert config.dynamics is propagator

    def test_scorer_must_satisfy_the_teacher_scorer_protocol(self) -> None:
        """A stand-in without ``label`` and ``signals`` is not a scorer."""
        with pytest.raises(ValidationError):
            OnPolicyConfig(**_make_config_kwargs(teacher_scorer=object()))

    def test_async_settings_are_not_configurable_yet(self) -> None:
        """``async_mode`` and ``staleness_threshold`` land with the remote scorer."""
        with pytest.raises(ValidationError):
            OnPolicyConfig(**_make_config_kwargs(async_mode=True))

    def test_structures_missing_a_propagator_field_are_rejected_at_construction(
        self,
    ) -> None:
        """A missing ``cell`` surfaces here, not from inside the first kernel."""
        with pytest.raises(ValidationError, match="propagates from"):
            OnPolicyConfig(
                **_make_config_kwargs(
                    dynamics=FIREVariableCell(_build_demo_model(), dt=0.1, n_steps=10),
                    initial_structures=InitialStructures(_build_atom_only_dataset()),
                )
            )

    def test_structures_missing_only_model_outputs_are_accepted(self) -> None:
        """The propagator primes ``forces`` itself, so an atom-only structure passes."""
        config = OnPolicyConfig(
            **_make_config_kwargs(
                initial_structures=InitialStructures(_build_atom_only_dataset())
            )
        )

        assert isinstance(config.initial_structures, InitialStructures)


class TestOnPolicyConfigPropagatorProbe:
    """One ``compute()`` at construction holds the propagator to its declarations."""

    def test_a_needs_key_the_student_never_produces_is_refused_naming_it(self) -> None:
        """A declared ``stress`` the demo student lacks fails here, not at step one."""
        propagator = _StressNeedingDynamics(_build_demo_model(), n_steps=10, dt=0.5)

        with pytest.raises(ValueError, match="'stress'.*__needs_keys__"):
            OnPolicyConfig(**_make_config_kwargs(dynamics=propagator))

    def test_a_field_read_that_nothing_declared_is_refused_naming_it(self) -> None:
        """A ``charges`` read inside compute() surfaces as a construction error."""
        torch.manual_seed(0)
        student = _ChargeReadingWrapper(_build_demo_model().model)
        propagator = DemoDynamics(student, n_steps=10, dt=0.5)

        with pytest.raises(
            ValueError, match="read a field.*charges.*__provides_keys__"
        ):
            OnPolicyConfig(**_make_config_kwargs(dynamics=propagator))

    def test_a_graph_student_is_probed_with_a_list_built_for_it(self) -> None:
        """A neighbor-list model needs no hook to pass the construction probe."""
        propagator = DemoDynamics(_build_lj_teacher(), n_steps=10, dt=0.5)

        config = OnPolicyConfig(**_make_config_kwargs(dynamics=propagator))

        assert config.dynamics is propagator

    def test_the_probe_leaves_the_propagator_and_student_as_it_found_them(self) -> None:
        """One forward at construction primes nothing and flips no mode, per module."""
        student = _build_demo_model().train()
        student.model.eval()
        propagator = DemoDynamics(student, n_steps=10, dt=0.5)

        OnPolicyConfig(**_make_config_kwargs(dynamics=propagator))

        assert student.training is True
        assert student.model.training is False
        assert propagator.step_count == 0
        assert propagator._forces_primed is False
        assert propagator._last_outputs is None


class TestOnPolicyConfigRequiredObjects:
    def test_a_config_without_initial_structures_raises(self) -> None:
        """The loop has to be told what to propagate from."""
        with pytest.raises(ValidationError, match="initial_structures"):
            OnPolicyConfig(**_make_config_kwargs(initial_structures=None))
