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

from nvalchemi.hooks import Hook


class _TestStage(Enum):
    A = 0
    B = 1


class TestHookProtocol:
    def test_valid_hook_satisfies_protocol(self):
        class ValidHook:
            frequency: int = 1
            stage: Enum = _TestStage.A

            def __call__(self, ctx, stage):
                pass

        hook = ValidHook()
        assert isinstance(hook, Hook)

    def test_missing_frequency_does_not_satisfy_protocol(self):
        class MissingFrequency:
            stage: Enum = _TestStage.A

            def __call__(self, ctx, stage):
                pass

        hook = MissingFrequency()
        assert not isinstance(hook, Hook)

    def test_missing_stage_does_not_satisfy_protocol(self):
        class MissingStage:
            frequency: int = 1

            def __call__(self, ctx, stage):
                pass

        hook = MissingStage()
        assert not isinstance(hook, Hook)

    def test_missing_call_does_not_satisfy_protocol(self):
        class MissingCall:
            frequency: int = 1
            stage: Enum = _TestStage.A

        hook = MissingCall()
        assert not isinstance(hook, Hook)

    def test_hook_with_runs_on_stage_satisfies_protocol(self):
        class MultiStageHook:
            frequency: int = 1
            stage: Enum = _TestStage.A

            def __call__(self, ctx, stage):
                pass

            def _runs_on_stage(self, stage):
                return stage in {_TestStage.A, _TestStage.B}

        hook = MultiStageHook()
        assert isinstance(hook, Hook)

    def test_hook_without_runs_on_stage_satisfies_protocol(self):
        """_runs_on_stage is optional — not part of the protocol."""

        class SimpleHook:
            frequency: int = 1
            stage: Enum = _TestStage.A

            def __call__(self, ctx, stage):
                pass

        hook = SimpleHook()
        assert isinstance(hook, Hook)
