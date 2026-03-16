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

"""Tests for graceful termination primitives."""

from __future__ import annotations

import pytest

from nvalchemi.training._hooks import TrainingContext, TrainingHook
from nvalchemi.training._stages import TrainingStageEnum
from nvalchemi.training._terminate import StopTraining, TerminateOnStepsHook


class TestStopTraining:
    """Tests for the :class:`StopTraining` sentinel exception."""

    def test_is_exception_subclass(self) -> None:
        """``StopTraining`` must be a subclass of ``Exception``."""
        assert issubclass(StopTraining, Exception)

    def test_can_be_raised_and_caught(self) -> None:
        """The sentinel can be raised and caught like a normal exception."""
        with pytest.raises(StopTraining):
            raise StopTraining("done")

    def test_message_preserved(self) -> None:
        """The message passed to ``StopTraining`` is accessible via ``args``."""
        exc = StopTraining("time to stop")
        assert exc.args[0] == "time to stop"


class TestTerminateOnStepsHook:
    """Tests for :class:`TerminateOnStepsHook`."""

    def test_conforms_to_training_hook_protocol(self) -> None:
        """``TerminateOnStepsHook`` instances satisfy ``TrainingHook``."""
        hook = TerminateOnStepsHook(max_count=10)
        assert isinstance(hook, TrainingHook)

    def test_default_stage_is_after_step(self) -> None:
        """Default monitored stage is ``AFTER_STEP``."""
        hook = TerminateOnStepsHook(max_count=5)
        assert hook.stage is TrainingStageEnum.AFTER_STEP

    def test_frequency_is_one(self) -> None:
        """Hook fires on every invocation (``frequency == 1``)."""
        hook = TerminateOnStepsHook(max_count=5)
        assert hook.frequency == 1

    def test_does_not_raise_below_threshold(self) -> None:
        """No exception when stage count is below *max_count*."""
        hook = TerminateOnStepsHook(max_count=3, stage=TrainingStageEnum.AFTER_STEP)
        ctx = TrainingContext()
        ctx.stage_counts[TrainingStageEnum.AFTER_STEP] = 2
        # Should not raise
        hook(ctx, model=None, trainer=None)  # type: ignore[arg-type]

    def test_raises_at_exact_threshold(self) -> None:
        """Raises ``StopTraining`` when count equals *max_count*."""
        hook = TerminateOnStepsHook(max_count=3, stage=TrainingStageEnum.AFTER_STEP)
        ctx = TrainingContext()
        ctx.stage_counts[TrainingStageEnum.AFTER_STEP] = 3
        with pytest.raises(StopTraining):
            hook(ctx, model=None, trainer=None)  # type: ignore[arg-type]

    def test_raises_above_threshold(self) -> None:
        """Raises ``StopTraining`` when count exceeds *max_count*."""
        hook = TerminateOnStepsHook(max_count=3, stage=TrainingStageEnum.AFTER_STEP)
        ctx = TrainingContext()
        ctx.stage_counts[TrainingStageEnum.AFTER_STEP] = 5
        with pytest.raises(StopTraining):
            hook(ctx, model=None, trainer=None)  # type: ignore[arg-type]

    def test_custom_stage(self) -> None:
        """Hook can monitor any ``TrainingStageEnum`` stage."""
        hook = TerminateOnStepsHook(max_count=2, stage=TrainingStageEnum.AFTER_EPOCH)
        assert hook.stage is TrainingStageEnum.AFTER_EPOCH

        ctx = TrainingContext()
        ctx.stage_counts[TrainingStageEnum.AFTER_EPOCH] = 2
        with pytest.raises(StopTraining):
            hook(ctx, model=None, trainer=None)  # type: ignore[arg-type]

    def test_zero_count_no_raise(self) -> None:
        """When stage has not fired (missing key), hook does not raise."""
        hook = TerminateOnStepsHook(max_count=1, stage=TrainingStageEnum.AFTER_STEP)
        ctx = TrainingContext()
        # stage_counts is empty — count defaults to 0
        hook(ctx, model=None, trainer=None)  # type: ignore[arg-type]
