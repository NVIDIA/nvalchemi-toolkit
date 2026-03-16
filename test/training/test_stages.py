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
"""Tests for :class:`TrainingStageEnum`."""

from __future__ import annotations

import pytest

from nvalchemi.training._stages import TrainingStageEnum


class TestTrainingStageEnumMembership:
    """Verify that all expected members exist with the correct values."""

    @pytest.mark.parametrize(
        ("name", "value"),
        [
            ("BEFORE_EPOCH", 0),
            ("AFTER_EPOCH", 1),
            ("BEFORE_STEP", 10),
            ("AFTER_DATA_LOAD", 11),
            ("BEFORE_FORWARD", 12),
            ("AFTER_FORWARD", 13),
            ("BEFORE_LOSS", 14),
            ("AFTER_LOSS", 15),
            ("BEFORE_BACKWARD", 16),
            ("AFTER_BACKWARD", 17),
            ("BEFORE_OPTIMIZER_STEP", 18),
            ("AFTER_OPTIMIZER_STEP", 19),
            ("AFTER_STEP", 20),
            ("BEFORE_VALIDATION", 30),
            ("AFTER_VALIDATION", 31),
            ("BEFORE_CHECKPOINT", 40),
            ("AFTER_CHECKPOINT", 41),
            ("ON_TRAINING_END", 50),
        ],
    )
    def test_member_exists_with_correct_value(self, name: str, value: int) -> None:
        """Each enum member has the expected name and integer value."""
        member = TrainingStageEnum[name]
        assert member.value == value

    def test_total_member_count(self) -> None:
        """Enum has exactly 18 members."""
        assert len(TrainingStageEnum) == 18


class TestTrainingStageEnumOrdering:
    """Verify that stage values are ordered logically."""

    def test_epoch_stages_precede_step_stages(self) -> None:
        """Epoch boundaries have lower values than within-step stages."""
        assert (
            TrainingStageEnum.BEFORE_EPOCH.value < TrainingStageEnum.BEFORE_STEP.value
        )
        assert TrainingStageEnum.AFTER_EPOCH.value < TrainingStageEnum.BEFORE_STEP.value

    def test_step_stages_are_monotonically_ordered(self) -> None:
        """Within-step stages follow the expected execution order."""
        step_stages = [
            TrainingStageEnum.BEFORE_STEP,
            TrainingStageEnum.AFTER_DATA_LOAD,
            TrainingStageEnum.BEFORE_FORWARD,
            TrainingStageEnum.AFTER_FORWARD,
            TrainingStageEnum.BEFORE_LOSS,
            TrainingStageEnum.AFTER_LOSS,
            TrainingStageEnum.BEFORE_BACKWARD,
            TrainingStageEnum.AFTER_BACKWARD,
            TrainingStageEnum.BEFORE_OPTIMIZER_STEP,
            TrainingStageEnum.AFTER_OPTIMIZER_STEP,
            TrainingStageEnum.AFTER_STEP,
        ]
        values = [s.value for s in step_stages]
        assert values == sorted(values)

    def test_validation_stages_follow_step_stages(self) -> None:
        """Validation boundaries have higher values than step stages."""
        assert (
            TrainingStageEnum.AFTER_STEP.value
            < TrainingStageEnum.BEFORE_VALIDATION.value
        )

    def test_checkpoint_stages_follow_validation(self) -> None:
        """Checkpoint boundaries follow validation."""
        assert (
            TrainingStageEnum.AFTER_VALIDATION.value
            < TrainingStageEnum.BEFORE_CHECKPOINT.value
        )

    def test_training_end_is_last(self) -> None:
        """ON_TRAINING_END has the highest value."""
        assert TrainingStageEnum.ON_TRAINING_END.value == max(
            s.value for s in TrainingStageEnum
        )

    def test_lookup_by_value(self) -> None:
        """Members can be retrieved by integer value."""
        assert TrainingStageEnum(10) is TrainingStageEnum.BEFORE_STEP
        assert TrainingStageEnum(50) is TrainingStageEnum.ON_TRAINING_END
