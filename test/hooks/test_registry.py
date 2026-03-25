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

from enum import Enum
from unittest.mock import MagicMock

import pytest

from nvalchemi.hooks import HookContext, HookRegistryMixin


class _TestStage(Enum):
    A = 0
    B = 1


class _OtherStage(Enum):
    X = 0


class _MinimalEngine(HookRegistryMixin):
    def __init__(self):
        self.step_count = 0
        self._init_hooks()


class _TypedEngine(HookRegistryMixin):
    _stage_type = _TestStage

    def __init__(self):
        self.step_count = 0
        self._init_hooks()


class TestHookRegistryMixin:
    def test_init_hooks_creates_empty_list(self):
        engine = _MinimalEngine()
        assert engine.hooks == []

    def test_register_hook_appends_to_list(self):
        engine = _MinimalEngine()

        class SimpleHook:
            frequency = 1
            stage = _TestStage.A

            def __call__(self, ctx, stage):
                pass

        hook = SimpleHook()
        engine.register_hook(hook)

        assert len(engine.hooks) == 1
        assert engine.hooks[0] is hook

    def test_register_hook_validates_frequency_zero(self):
        engine = _MinimalEngine()

        class ZeroFrequencyHook:
            frequency = 0
            stage = _TestStage.A

            def __call__(self, ctx, stage):
                pass

        with pytest.raises(ValueError, match="positive integer"):
            engine.register_hook(ZeroFrequencyHook())

    def test_register_hook_validates_frequency_negative(self):
        engine = _MinimalEngine()

        class NegativeFrequencyHook:
            frequency = -1
            stage = _TestStage.A

            def __call__(self, ctx, stage):
                pass

        with pytest.raises(ValueError, match="positive integer"):
            engine.register_hook(NegativeFrequencyHook())

    def test_call_hooks_filters_by_stage(self):
        engine = _MinimalEngine()
        engine.step_count = 1
        call_log = []

        class HookA:
            frequency = 1
            stage = _TestStage.A

            def __call__(self, ctx, stage):
                call_log.append(("A", stage))

        class HookB:
            frequency = 1
            stage = _TestStage.B

            def __call__(self, ctx, stage):
                call_log.append(("B", stage))

        engine.register_hook(HookA())
        engine.register_hook(HookB())

        engine._call_hooks(_TestStage.A, MagicMock())

        # Only HookA fires — HookB's stage doesn't match
        assert call_log == [("A", _TestStage.A)]

    def test_call_hooks_uses_runs_on_stage_override(self):
        engine = _MinimalEngine()
        engine.step_count = 1
        call_log = []

        class MultiStageHook:
            frequency = 1
            stage = _TestStage.A  # primary stage for protocol

            def _runs_on_stage(self, stage):
                return stage in {_TestStage.A, _TestStage.B}

            def __call__(self, ctx, stage):
                call_log.append(stage)

        engine.register_hook(MultiStageHook())

        engine._call_hooks(_TestStage.A, MagicMock())
        engine._call_hooks(_TestStage.B, MagicMock())

        assert call_log == [_TestStage.A, _TestStage.B]

    def test_call_hooks_runs_on_stage_false_skips_hook(self):
        engine = _MinimalEngine()
        engine.step_count = 1
        call_log = []

        class SelectiveHook:
            frequency = 1
            stage = _TestStage.A

            def _runs_on_stage(self, stage):
                return False  # never fires

            def __call__(self, ctx, stage):
                call_log.append(stage)

        engine.register_hook(SelectiveHook())
        engine._call_hooks(_TestStage.A, MagicMock())

        assert call_log == []

    def test_call_hooks_hook_receives_ctx_and_stage(self):
        engine = _MinimalEngine()
        engine.step_count = 42
        received = []

        class InspectHook:
            frequency = 1
            stage = _TestStage.A

            def __call__(self, ctx, stage):
                received.append((ctx, stage))

        engine.register_hook(InspectHook())
        mock_batch = MagicMock()
        engine._call_hooks(_TestStage.A, mock_batch)

        assert len(received) == 1
        ctx, stage = received[0]
        assert isinstance(ctx, HookContext)
        assert ctx.step_count == 42
        assert ctx.batch is mock_batch
        assert stage == _TestStage.A

    def test_build_context_returns_hook_context(self):
        engine = _MinimalEngine()
        engine.step_count = 42

        mock_batch = MagicMock()
        ctx = engine._build_context(mock_batch)

        assert isinstance(ctx, HookContext)
        assert ctx.batch is mock_batch
        assert ctx.step_count == 42

    def test_build_context_includes_model_if_present(self):
        engine = _MinimalEngine()
        engine.step_count = 1
        engine.model = MagicMock()

        ctx = engine._build_context(MagicMock())
        assert ctx.model is engine.model

    def test_init_hooks_with_provided_hooks(self):
        class SimpleHook:
            frequency = 1
            stage = _TestStage.A

            def __call__(self, ctx, stage):
                pass

        engine = HookRegistryMixin()
        engine.step_count = 0
        engine._init_hooks(hooks=[SimpleHook()])

        assert len(engine.hooks) == 1


class TestStageTypeValidation:
    def test_register_hook_accepts_matching_stage_type(self):
        engine = _TypedEngine()

        class GoodHook:
            frequency = 1
            stage = _TestStage.A

            def __call__(self, ctx, stage):
                pass

        engine.register_hook(GoodHook())
        assert len(engine.hooks) == 1

    def test_register_hook_rejects_wrong_stage_type(self):
        engine = _TypedEngine()

        class BadHook:
            frequency = 1
            stage = _OtherStage.X

            def __call__(self, ctx, stage):
                pass

        with pytest.raises(TypeError, match="only accepts _TestStage"):
            engine.register_hook(BadHook())

    def test_no_validation_when_stage_type_is_none(self):
        engine = _MinimalEngine()  # _stage_type defaults to None

        class AnyStageHook:
            frequency = 1
            stage = _OtherStage.X

            def __call__(self, ctx, stage):
                pass

        engine.register_hook(AnyStageHook())
        assert len(engine.hooks) == 1

    def test_tuple_stage_type_accepts_multiple_enums(self):
        class MultiEngine(HookRegistryMixin):
            _stage_type = (_TestStage, _OtherStage)

            def __init__(self):
                self.step_count = 0
                self._init_hooks()

        engine = MultiEngine()

        class HookA:
            frequency = 1
            stage = _TestStage.A

            def __call__(self, ctx, stage):
                pass

        class HookX:
            frequency = 1
            stage = _OtherStage.X

            def __call__(self, ctx, stage):
                pass

        engine.register_hook(HookA())
        engine.register_hook(HookX())
        assert len(engine.hooks) == 2

    def test_tuple_stage_type_rejects_unlisted_enum(self):
        class ThirdStage(Enum):
            Z = 0

        class MultiEngine(HookRegistryMixin):
            _stage_type = (_TestStage, _OtherStage)

            def __init__(self):
                self.step_count = 0
                self._init_hooks()

        engine = MultiEngine()

        class BadHook:
            frequency = 1
            stage = ThirdStage.Z

            def __call__(self, ctx, stage):
                pass

        with pytest.raises(TypeError, match="only accepts _TestStage | _OtherStage"):
            engine.register_hook(BadHook())

    def test_runs_on_stage_bypasses_stage_type_check(self):
        """Hooks with _runs_on_stage skip stage-type validation."""
        engine = _TypedEngine()  # _stage_type = _TestStage

        class CrossCategoryHook:
            frequency = 1
            stage = _OtherStage.X  # mismatched type

            def _runs_on_stage(self, stage):
                return True

            def __call__(self, ctx, stage):
                pass

        engine.register_hook(CrossCategoryHook())
        assert len(engine.hooks) == 1
