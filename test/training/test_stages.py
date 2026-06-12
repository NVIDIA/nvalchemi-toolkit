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
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from nvalchemi.dynamics.base import DynamicsStage
from nvalchemi.hooks import HookRegistryMixin
from nvalchemi.training import TrainingStage

# Canonical name/order snapshot. Must be edited by hand if TrainingStage members
# change — that is the point: an accidental reorder or rename fails this test.
_EXPECTED_MEMBERS: tuple[str, ...] = (
    "SETUP",
    "BEFORE_TRAINING",
    "BEFORE_EPOCH",
    "BEFORE_BATCH",
    "BEFORE_FORWARD",
    "AFTER_FORWARD",
    "BEFORE_LOSS",
    "AFTER_LOSS",
    "BEFORE_BACKWARD",
    "DO_BACKWARD",
    "AFTER_BACKWARD",
    "BEFORE_OPTIMIZER_STEP",
    "DO_OPTIMIZER_STEP",
    "AFTER_OPTIMIZER_STEP",
    "AFTER_BATCH",
    "AFTER_EPOCH",
    "AFTER_TRAINING",
    "AFTER_VALIDATION",
)


class _TrainingHost(HookRegistryMixin):
    _stage_type = TrainingStage

    def __init__(self):
        self.step_count = 0
        self._init_hooks()


class _DynamicsHost(HookRegistryMixin):
    _stage_type = DynamicsStage

    def __init__(self):
        self.step_count = 0
        self._init_hooks()


class TestTrainingStageEnum:
    def test_members_in_declared_order(self):
        assert tuple(s.name for s in TrainingStage) == _EXPECTED_MEMBERS

    def test_values_are_unique(self):
        assert len({s.value for s in TrainingStage}) == len(TrainingStage)

    def test_members_count(self):
        assert len(TrainingStage) == 18

    def test_all_members_are_before_or_after_or_do(self):
        for member in TrainingStage:
            assert member is TrainingStage.SETUP or member.name.startswith(
                ("BEFORE_", "AFTER_", "DO_")
            )

    def test_do_backward_between_before_and_after(self):
        members = list(TrainingStage)
        assert (
            members.index(TrainingStage.DO_BACKWARD)
            == members.index(TrainingStage.BEFORE_BACKWARD) + 1
        )
        assert (
            members.index(TrainingStage.DO_BACKWARD)
            == members.index(TrainingStage.AFTER_BACKWARD) - 1
        )

    def test_do_optimizer_step_between_before_and_after(self):
        members = list(TrainingStage)
        assert (
            members.index(TrainingStage.DO_OPTIMIZER_STEP)
            == members.index(TrainingStage.BEFORE_OPTIMIZER_STEP) + 1
        )
        assert (
            members.index(TrainingStage.DO_OPTIMIZER_STEP)
            == members.index(TrainingStage.AFTER_OPTIMIZER_STEP) - 1
        )

    def test_public_optimizer_boundaries_remain_distinct(self):
        members = list(TrainingStage)
        assert members.index(TrainingStage.AFTER_BACKWARD) + 1 == members.index(
            TrainingStage.BEFORE_OPTIMIZER_STEP
        )
        assert members.index(TrainingStage.AFTER_OPTIMIZER_STEP) + 1 == members.index(
            TrainingStage.AFTER_BATCH
        )


class TestTrainingStageRegistration:
    def test_register_training_hook_succeeds(self):
        host = _TrainingHost()

        class TrainingHook:
            frequency = 1
            stage = TrainingStage.BEFORE_BATCH

            def __call__(self, ctx, stage):
                pass

        hook = TrainingHook()
        host.register_hook(hook)

        assert len(host.hooks) == 1
        assert host.hooks[0] is hook

    def test_call_hooks_dispatches_by_stage(self):
        host = _TrainingHost()
        host.step_count = 1
        call_log: list[TrainingStage] = []

        class BeforeBatchHook:
            frequency = 1
            stage = TrainingStage.BEFORE_BATCH

            def __call__(self, ctx, stage):
                call_log.append(stage)

        class AfterBatchHook:
            frequency = 1
            stage = TrainingStage.AFTER_BATCH

            def __call__(self, ctx, stage):
                call_log.append(stage)

        host.register_hook(BeforeBatchHook())
        host.register_hook(AfterBatchHook())

        host._call_hooks(TrainingStage.BEFORE_BATCH, MagicMock())

        assert call_log == [TrainingStage.BEFORE_BATCH]

    def test_dynamics_stage_rejected_on_training_host(self):
        host = _TrainingHost()

        class DynamicsHook:
            frequency = 1
            stage = DynamicsStage.BEFORE_STEP

            def __call__(self, ctx, stage):
                pass

        with pytest.raises(
            TypeError, match=r"type DynamicsStage.*only accepts TrainingStage"
        ):
            host.register_hook(DynamicsHook())

    def test_training_host_requires_stage(self):
        """Pins that a TrainingStage-typed host inherits the generic "stage required" contract."""
        host = _TrainingHost()

        class NoStageHook:
            frequency = 1
            stage = None

            def __call__(self, ctx, stage):
                pass

        with pytest.raises(TypeError, match="no stage assigned"):
            host.register_hook(NoStageHook())


class TestStageIsolation:
    def test_training_stage_rejected_on_dynamics_host(self):
        host = _DynamicsHost()

        class TrainingHook:
            frequency = 1
            stage = TrainingStage.BEFORE_BATCH

            def __call__(self, ctx, stage):
                pass

        with pytest.raises(
            TypeError, match=r"type TrainingStage.*only accepts DynamicsStage"
        ):
            host.register_hook(TrainingHook())

    def test_runs_on_stage_bypass_allows_cross_category(self):
        """`_runs_on_stage` bypasses the stage-type check at registration."""
        host = _TrainingHost()

        class CrossCategoryHook:
            frequency = 1
            stage = DynamicsStage.BEFORE_STEP  # foreign enum; normally rejected

            def _runs_on_stage(self, stage):
                return True

            def __call__(self, ctx, stage):
                pass

        host.register_hook(CrossCategoryHook())
        assert len(host.hooks) == 1
