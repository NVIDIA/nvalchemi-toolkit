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
"""Tests for declarative dynamics strategies."""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest
import torch

from nvalchemi.data import AtomicData, Batch
from nvalchemi.dynamics import BaseDynamics, DynamicsStage, DynamicsStrategy
from nvalchemi.dynamics.demo import DemoDynamics
from nvalchemi.dynamics.hooks import NaNDetectorHook
from nvalchemi.models.demo import DemoModel, DemoModelWrapper


class _Strategy(DynamicsStrategy):
    def build_engine(self) -> BaseDynamics:
        raise NotImplementedError


class _CachedStrategy(DynamicsStrategy):
    """Opt into persistent engines without overriding construction or execution."""

    cache_engine: bool = True


def test_unconfigured_strategy_fails_at_build_time() -> None:
    strategy = DynamicsStrategy(model=DemoModelWrapper(DemoModel()).eval())

    with pytest.raises(
        NotImplementedError,
        match=r"DynamicsStrategy must set `engine=` or override build_engine\(\)",
    ):
        strategy.build_engine()


@pytest.mark.parametrize("strategy_type", [DynamicsStrategy, _CachedStrategy])
def test_repeated_runs_preserve_state_only_when_caching(
    strategy_type: type[DynamicsStrategy],
) -> None:
    model = DemoModelWrapper(DemoModel()).eval()
    strategy = strategy_type(
        model=model, engine=DemoDynamics, engine_kwargs={"dt": 0.01}, n_steps=2
    )
    batch = Batch.from_data_list(
        [
            AtomicData(
                atomic_numbers=torch.tensor([6, 8]),
                positions=torch.ones(2, 3),
            )
        ],
    )
    batch.velocities = torch.zeros_like(batch.positions)
    batch.forces = torch.zeros_like(batch.positions)
    batch.energy = torch.zeros(1, 1)

    with patch.object(
        DemoDynamics, "run", autospec=True, side_effect=DemoDynamics.run
    ) as run:
        assert strategy.run(batch) is batch
        first_engine = run.call_args.args[0]
        assert first_engine.step_count == 2

        assert strategy.run(batch, n_steps=3) is batch
        second_engine = run.call_args.args[0]

    assert (second_engine is first_engine) is strategy.cache_engine
    assert second_engine.step_count == (5 if strategy.cache_engine else 3)
    assert second_engine.n_steps == 2

    spec = json.loads(json.dumps(strategy.to_spec_dict()))
    assert "_engine" not in spec
    restored = strategy_type.from_spec_dict(spec, model=model)
    assert restored._engine is None
    assert restored.cache_engine is strategy.cache_engine


def test_spec_round_trip_requires_model_and_preserves_concrete_hook_stage_enum() -> (
    None
):
    model = DemoModelWrapper(DemoModel()).eval()
    strategy = DynamicsStrategy(
        model=model,
        engine=DemoDynamics,
        engine_kwargs={"dt": 0.25},
        n_steps=3,
        extra_hooks=[NaNDetectorHook()],
        cache_engine=True,
    )

    spec = json.loads(json.dumps(strategy.to_spec_dict()))

    assert "model" not in spec
    assert "model_spec" not in spec
    assert "extra_hooks" not in spec
    assert spec["engine"] == "nvalchemi.dynamics.demo.DemoDynamics"
    # Hook constructors accept abstract Enum, so preserve the concrete enum type.
    assert spec["extra_hook_specs"][0]["stage"] == {
        "__enum__": "nvalchemi.dynamics.base.DynamicsStage",
        "value": DynamicsStage.AFTER_COMPUTE.value,
    }

    with pytest.raises(TypeError, match="required keyword-only argument: 'model'"):
        DynamicsStrategy.from_spec_dict(spec)

    restored = DynamicsStrategy.from_spec_dict(spec, model=model)

    assert restored.model is model
    assert len(restored.extra_hooks) == 1
    assert isinstance(restored.extra_hooks[0], NaNDetectorHook)
    # Restoration must return the concrete member rather than its raw value.
    assert restored.extra_hooks[0].stage is DynamicsStage.AFTER_COMPUTE
    assert restored.engine is DemoDynamics
    assert restored.cache_engine is True
    engine = restored.build_engine()
    assert engine.model is model
    assert engine.n_steps == 3
    assert engine.dt == 0.25
    assert engine.hooks == restored.extra_hooks


def test_from_spec_dict_appends_runtime_extra_hooks() -> None:
    model = DemoModelWrapper(DemoModel()).eval()
    serialized_hook = NaNDetectorHook()
    runtime_hook = NaNDetectorHook()
    spec = _Strategy(
        model=model,
        extra_hooks=[serialized_hook],
    ).to_spec_dict()

    restored = _Strategy.from_spec_dict(
        spec,
        model=model,
        extra_hooks=[runtime_hook],
    )

    assert len(restored.extra_hooks) == 2
    assert isinstance(restored.extra_hooks[0], NaNDetectorHook)
    assert restored.extra_hooks[1] is runtime_hook
