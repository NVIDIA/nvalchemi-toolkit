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

import pytest

from nvalchemi.dynamics import BaseDynamics, DynamicsStage, DynamicsStrategy
from nvalchemi.dynamics.hooks import NaNDetectorHook
from nvalchemi.models.demo import DemoModel, DemoModelWrapper


class _Strategy(DynamicsStrategy):
    def build_engine(self) -> BaseDynamics:
        raise NotImplementedError


def test_spec_round_trip_requires_model_and_preserves_concrete_hook_stage_enum() -> (
    None
):
    model = DemoModelWrapper(DemoModel()).eval()
    strategy = _Strategy(model=model, extra_hooks=[NaNDetectorHook()])

    spec = json.loads(json.dumps(strategy.to_spec_dict()))

    assert "model" not in spec
    assert "model_spec" not in spec
    assert "extra_hooks" not in spec
    # Hook constructors accept abstract Enum, so preserve the concrete enum type.
    assert spec["extra_hook_specs"][0]["stage"] == {
        "__enum__": "nvalchemi.dynamics.base.DynamicsStage",
        "value": DynamicsStage.AFTER_COMPUTE.value,
    }

    with pytest.raises(TypeError, match="required keyword-only argument: 'model'"):
        _Strategy.from_spec_dict(spec)

    restored = _Strategy.from_spec_dict(spec, model=model)

    assert restored.model is model
    assert len(restored.extra_hooks) == 1
    assert isinstance(restored.extra_hooks[0], NaNDetectorHook)
    # Restoration must return the concrete member rather than its raw value.
    assert restored.extra_hooks[0].stage is DynamicsStage.AFTER_COMPUTE


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
