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
"""Unit tests for ``DynamicsStrategy``, the base every workflow recipe shares.

Two properties carry the weight here. The engine cache is what lets
consecutive ``run()`` calls continue one trajectory, which means it has to
refuse a second potential rather than quietly answer with the first. And
``to_spec_dict`` advertises JSON — so it has to produce JSON for every
configuration the engine itself accepts, which includes tensor-valued
controls.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from nvalchemi.data import AtomicData, Batch
from nvalchemi.dynamics import NVE, DynamicsStrategy, NVTLangevin
from nvalchemi.dynamics.base import DynamicsStage
from nvalchemi.models.demo import DemoModel, DemoModelWrapper

_ENGINE_KWARGS = {"dt": 0.1, "temperature": 300.0, "friction": 0.1}


def _model() -> DemoModelWrapper:
    """Return a demo potential."""
    return DemoModelWrapper(DemoModel())


def _strategy(**kwargs: object) -> DynamicsStrategy:
    """Return a strategy over ``NVTLangevin``."""
    kwargs.setdefault("engine_kwargs", _ENGINE_KWARGS)
    return DynamicsStrategy(engine=NVTLangevin, **kwargs)


def _batch(n_graphs: int = 2, atoms: int = 3) -> Batch:
    """Return a batch with the buffers an integrator writes into."""
    torch.manual_seed(0)
    items = []
    for _ in range(n_graphs):
        data = AtomicData(
            positions=torch.randn(atoms, 3),
            atomic_numbers=torch.full((atoms,), 6, dtype=torch.long),
            atomic_masses=torch.ones(atoms),
            forces=torch.zeros(atoms, 3),
            energy=torch.zeros(1, 1),
        )
        data.add_node_property("velocities", torch.zeros(atoms, 3))
        items.append(data)
    return Batch.from_data_list(items)


class TestEngineCache:
    """One strategy drives one engine, and says so when asked for a second."""

    def test_the_engine_is_cached(self) -> None:
        """Consecutive run() calls must continue, not restart, a trajectory."""
        strategy, model = _strategy(), _model()
        assert strategy.dynamics(model) is strategy.dynamics(model)

    def test_step_state_survives_across_calls(self) -> None:
        strategy, model = _strategy(n_steps=2), _model()
        batch = _batch()
        strategy.run(batch, model)
        strategy.run(batch, model)
        assert strategy.dynamics(model).step_count == 4

    def test_a_different_model_is_refused(self) -> None:
        """Otherwise the trajectory is produced for a potential nobody asked for.

        The cached engine holds the *first* model, so returning it would
        evaluate that one while the caller believes they passed another.
        """
        strategy = _strategy()
        first = _model()
        strategy.dynamics(first)
        with pytest.raises(ValueError, match="already driving an engine"):
            strategy.dynamics(_model())

    def test_run_refuses_a_different_model_too(self) -> None:
        strategy = _strategy(n_steps=1)
        batch = _batch()
        strategy.run(batch, _model())
        with pytest.raises(ValueError, match="already driving an engine"):
            strategy.run(batch, _model())

    def test_build_remains_the_escape_hatch(self) -> None:
        """A caller who wants an independent engine has a way to say so."""
        strategy = _strategy()
        strategy.dynamics(_model())
        other = _model()
        assert strategy.build(other).model is other

    def test_build_does_not_disturb_the_cache(self) -> None:
        strategy, model = _strategy(), _model()
        cached = strategy.dynamics(model)
        strategy.build(_model())
        assert strategy.dynamics(model) is cached


class TestBuildHooks:
    """The base contributes only ``extra_hooks``, in order."""

    def test_extra_hooks_are_the_whole_contribution(self) -> None:
        class _NoopHook:
            stage = DynamicsStage.AFTER_STEP
            frequency = 1

            def __call__(self, ctx: object, stage: DynamicsStage) -> None:
                pass

        hooks = [_NoopHook(), _NoopHook()]
        strategy = _strategy(extra_hooks=hooks)
        assert strategy.build_hooks() == hooks
        assert strategy.dynamics(_model()).hooks == hooks


class TestToSpecDict:
    """It advertises JSON, so it has to produce JSON."""

    def test_the_spec_is_json(self) -> None:
        spec = _strategy(n_steps=17).to_spec_dict()
        assert json.loads(json.dumps(spec)) == spec
        assert spec["engine"].endswith("NVTLangevin")
        assert spec["n_steps"] == 17

    def test_tensor_valued_controls_are_converted(self) -> None:
        """``NVTLangevin`` annotates ``temperature`` as ``float | Tensor``.

        Copying ``engine_kwargs`` verbatim produced a dict ``json.dumps``
        refused, for a configuration the engine itself accepts.
        """
        spec = _strategy(
            engine_kwargs={
                "dt": torch.tensor(0.5),
                "temperature": torch.tensor([300.0, 350.0]),
                "friction": 0.1,
            }
        ).to_spec_dict()
        json.dumps(spec)
        assert spec["engine_kwargs"]["dt"] == 0.5
        assert spec["engine_kwargs"]["temperature"] == [300.0, 350.0]

    def test_nested_containers_are_converted(self) -> None:
        spec = _strategy(
            engine_kwargs={"opts": {"dtype": torch.float64, "bounds": (1, 2)}}
        ).to_spec_dict()
        json.dumps(spec)
        assert spec["engine_kwargs"]["opts"] == {
            "dtype": "torch.float64",
            "bounds": [1, 2],
        }

    def test_a_path_is_converted(self, tmp_path) -> None:
        destination = tmp_path / "run.zarr"
        spec = _strategy(engine_kwargs={"out": Path(destination)}).to_spec_dict()
        json.dumps(spec)
        assert spec["engine_kwargs"]["out"] == str(destination)

    def test_a_value_with_no_json_form_is_refused_by_name(self) -> None:
        """A repr nothing can read back is worse than saying so."""
        strategy = _strategy(engine_kwargs={"callback": lambda batch: None})
        with pytest.raises(TypeError, match=r"engine_kwargs\['callback'\]"):
            strategy.to_spec_dict()

    def test_extra_hooks_are_excluded(self) -> None:
        """A hook is a live object, not a knob."""

        class _NoopHook:
            stage = DynamicsStage.AFTER_STEP
            frequency = 1

            def __call__(self, ctx: object, stage: DynamicsStage) -> None:
                pass

        spec = _strategy(extra_hooks=[_NoopHook()]).to_spec_dict()
        json.dumps(spec)
        assert "extra_hooks" not in spec


class TestRun:
    """The strategy configures; ``BaseDynamics`` steps."""

    def test_run_delegates_and_returns_the_batch(self) -> None:
        strategy, model = _strategy(), _model()
        batch = _batch()
        assert strategy.run(batch, model, n_steps=3) is batch
        assert strategy.dynamics(model).step_count == 3

    def test_n_steps_falls_back_to_the_field(self) -> None:
        strategy, model = _strategy(n_steps=4), _model()
        strategy.run(_batch(), model)
        assert strategy.dynamics(model).step_count == 4

    def test_no_step_count_anywhere_raises(self) -> None:
        strategy, model = _strategy(), _model()
        with pytest.raises(ValueError, match="No step count provided"):
            strategy.run(_batch(), model)

    def test_a_different_engine_class_is_honoured(self) -> None:
        strategy = DynamicsStrategy(engine=NVE, engine_kwargs={"dt": 0.1})
        assert isinstance(strategy.dynamics(_model()), NVE)
