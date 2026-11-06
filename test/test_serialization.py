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
"""Tests for :mod:`nvalchemi._serialization`."""

from __future__ import annotations

import json
import math

import pytest

from nvalchemi._serialization import MeasurementRecord
from nvalchemi.training.distillation.evaluation import (
    MeasurementRecord as EvaluationMeasurementRecord,
)


class _Timing(MeasurementRecord):
    """Record with a required integer, an optional float, and a declared tuple."""

    steps: int
    seconds: float | None = None
    repeats: tuple[int, int, int] = (1, 1, 1)


class TestMeasurementRecord:
    """Exporting a record and rebuilding it from its own export."""

    def test_the_evaluation_suite_re_exports_the_core_record(self) -> None:
        """The evaluation suite's record base is the core class itself."""
        assert EvaluationMeasurementRecord is MeasurementRecord

    def test_a_record_survives_a_json_round_trip(self) -> None:
        """Every field comes back as it was exported, ``None`` included."""
        record = _Timing(steps=3, repeats=(2, 1, 1))
        assert _Timing.from_dict(json.loads(json.dumps(record.to_dict()))) == record
        assert record.to_dict()["seconds"] is None

    def test_a_nonfinite_spelled_by_a_strict_writer_reads_back_as_the_number(
        self,
    ) -> None:
        """A writer that spells ``nan`` or ``inf`` for strict JSON still rebuilds."""
        rebuilt = _Timing.from_dict({"steps": 1, "seconds": "nan"})
        assert math.isnan(rebuilt.seconds)
        assert _Timing.from_dict({"steps": 1, "seconds": "-inf"}).seconds == -math.inf

    def test_a_rebuilt_record_keeps_the_tuple_its_field_declares(self) -> None:
        """A list read out of JSON returns as the tuple the record declares."""
        exported = json.loads(json.dumps(_Timing(steps=1, repeats=(2, 1, 1)).to_dict()))
        assert exported["repeats"] == [2, 1, 1]
        assert _Timing.from_dict(exported).repeats == (2, 1, 1)

    def test_a_key_the_record_does_not_declare_is_rejected(self) -> None:
        """An export written by another version fails where it is read."""
        with pytest.raises(ValueError, match=r"carrying \['minutes'\]"):
            _Timing.from_dict({"steps": 1, "minutes": 0.1})

    def test_a_missing_required_field_is_rejected(self) -> None:
        """A truncated export cannot be rebuilt into a partly-defaulted object."""
        with pytest.raises(ValueError, match=r"missing the required \['steps'\]"):
            _Timing.from_dict({"seconds": 2.0})

    def test_a_value_failing_its_field_is_named(self) -> None:
        """A fault that is neither an unknown nor a missing key names the field."""
        with pytest.raises(ValueError, match=r"_Timing\.steps:"):
            _Timing.from_dict({"steps": "three"})

    def test_a_record_is_frozen(self) -> None:
        """A measurement is a result, not a mutable state."""
        record = _Timing(steps=1)
        with pytest.raises(ValueError, match="frozen"):
            record.steps = 2
