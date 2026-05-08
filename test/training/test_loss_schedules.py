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
"""Tests for :mod:`nvalchemi.training.losses.schedules`."""

from __future__ import annotations

import json
from typing import Any

import pytest
from pydantic import ValidationError

from nvalchemi.training import (
    ConstantWeight,
    CosineWeight,
    LinearWeight,
    LossWeightSchedule,
    PiecewiseWeight,
    create_model_spec,
    create_model_spec_from_json,
)


class TestSchedules:
    """Tests for the weight schedules, protocol, and BaseSpec round-trip."""

    def test_protocol_runtime_check(self) -> None:
        w = ConstantWeight(value=1.0)
        assert isinstance(w, LossWeightSchedule)
        assert w.per_epoch is False

    def test_constant_weight(self) -> None:
        w = ConstantWeight(value=2.5)
        assert w(0, 0) == 2.5
        assert w(100, 3) == 2.5
        assert w(100_000, 99) == 2.5

    @pytest.mark.parametrize("cls", [LinearWeight, CosineWeight])
    def test_ramp_endpoints_and_clamp(
        self, cls: type[LinearWeight | CosineWeight]
    ) -> None:
        w = cls(start=0.0, end=1.0, num_steps=10)
        assert w(0, 0) == 0.0
        assert abs(w(10, 0) - 1.0) < 1e-6
        assert w(100, 0) == 1.0
        assert w(-5, 0) == 0.0

    def test_linear_midpoint(self) -> None:
        w = LinearWeight(start=0.0, end=1.0, num_steps=10)
        assert abs(w(5, 0) - 0.5) < 1e-6

    def test_cosine_midpoint(self) -> None:
        w = CosineWeight(start=0.0, end=1.0, num_steps=10)
        assert abs(w(5, 0) - 0.5) < 1e-6

    @pytest.mark.parametrize("cls", [LinearWeight, CosineWeight])
    def test_per_epoch_ramps_use_epoch_counter(
        self, cls: type[LinearWeight | CosineWeight]
    ) -> None:
        w = cls(start=0.0, end=1.0, num_steps=10, per_epoch=True)
        assert w.per_epoch is True
        assert w(step=10, epoch=0) == 0.0
        assert abs(w(step=0, epoch=5) - 0.5) < 1e-6
        assert w(step=0, epoch=10) == 1.0

    @pytest.mark.parametrize(
        "boundaries,values,step,expected",
        [
            ((100,), (0.1, 0.9), 0, 0.1),
            ((100,), (0.1, 0.9), 99, 0.1),
            ((100,), (0.1, 0.9), 100, 0.9),
            ((100,), (0.1, 0.9), 500, 0.9),
            ((10, 20, 30), (0.0, 0.25, 0.5, 1.0), 5, 0.0),
            ((10, 20, 30), (0.0, 0.25, 0.5, 1.0), 10, 0.25),
            ((10, 20, 30), (0.0, 0.25, 0.5, 1.0), 20, 0.5),
            ((10, 20, 30), (0.0, 0.25, 0.5, 1.0), 30, 1.0),
        ],
    )
    def test_piecewise_weight(
        self,
        boundaries: tuple[int, ...],
        values: tuple[float, ...],
        step: int,
        expected: float,
    ) -> None:
        w = PiecewiseWeight(boundaries=boundaries, values=values)
        assert w(step, 0) == expected

    def test_per_epoch_piecewise_uses_epoch_counter(self) -> None:
        w = PiecewiseWeight(
            boundaries=(2,),
            values=(0.0, 1.0),
            per_epoch=True,
        )
        assert w(step=100, epoch=1) == 0.0
        assert w(step=0, epoch=2) == 1.0

    @pytest.mark.parametrize(
        "cls,kwargs",
        [
            (LinearWeight, {"start": 0.0, "end": 1.0, "num_steps": 0}),
            (LinearWeight, {"start": 0.0, "end": 1.0, "num_steps": -3}),
            (CosineWeight, {"start": 0.0, "end": 1.0, "num_steps": 0}),
            (
                PiecewiseWeight,
                {"boundaries": (10, 20), "values": (0.1, 0.5)},
            ),
            (
                PiecewiseWeight,
                {"boundaries": (10, 5), "values": (0.1, 0.5, 0.9)},
            ),
            (PiecewiseWeight, {"boundaries": (-1,), "values": (0.1, 0.5)}),
        ],
    )
    def test_schedule_validators_reject_bad_input(
        self, cls: type, kwargs: dict[str, Any]
    ) -> None:
        with pytest.raises(ValidationError):
            cls(**kwargs)

    def test_schedule_frozen(self) -> None:
        w = ConstantWeight(value=1.0)
        with pytest.raises(ValidationError):
            w.value = 2.0  # type: ignore[misc]

    def test_piecewise_hashable(self) -> None:
        w = PiecewiseWeight(boundaries=(10, 20), values=(0.1, 0.5, 0.9))
        assert hash(w) == hash(w)

    @pytest.mark.parametrize(
        "cls,kwargs",
        [
            (ConstantWeight, {"value": 0.5}),
            (
                LinearWeight,
                {"start": 0.1, "end": 0.9, "num_steps": 100, "per_epoch": True},
            ),
            (CosineWeight, {"start": 1.0, "end": 0.0, "num_steps": 50}),
            (
                PiecewiseWeight,
                {"boundaries": (10, 20), "values": (0.1, 0.5, 0.9)},
            ),
        ],
    )
    def test_schedule_basespec_roundtrip(
        self, cls: type, kwargs: dict[str, Any]
    ) -> None:
        spec = create_model_spec(cls, **kwargs)
        dumped = spec.model_dump_json()
        rebuilt_spec = create_model_spec_from_json(json.loads(dumped))
        built = rebuilt_spec.build()
        assert isinstance(built, cls)
        for k, v in kwargs.items():
            assert getattr(built, k) == v
        assert isinstance(built(5, 0), float)
