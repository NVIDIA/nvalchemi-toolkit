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
"""
Shared pytest fixtures and helpers for the dynamics test suite.
"""

from __future__ import annotations

from enum import Enum

from nvalchemi.dynamics.base import HookStageEnum
from nvalchemi.hooks import HookContext


class RecordingHook:
    """
    A concrete hook implementation that records when it was called.

    This hook appends its name to a shared list each time it is invoked,
    allowing tests to verify execution order.

    Attributes
    ----------
    frequency : int
        Execute every N steps.
    stage : HookStageEnum
        Stage at which to fire.
    name : str
        Identifier for this hook.
    record_list : list[str]
        Shared list to append name to when called.

    Examples
    --------
    >>> record_list = []
    >>> hook = RecordingHook(HookStageEnum.AFTER_STEP, record_list, name="my_hook")
    >>> hook.stage
    <HookStageEnum.AFTER_STEP: 7>
    """

    def __init__(
        self,
        stage: HookStageEnum,
        record_list: list[str],
        name: str | None = None,
        frequency: int = 1,
    ) -> None:
        """
        Initialize the recording hook.

        Parameters
        ----------
        stage : HookStageEnum
            The stage at which this hook fires.
        record_list : list[str]
            List to append to when called.
        name : str | None, optional
            Name identifier. Defaults to stage name.
        frequency : int, optional
            How often to fire (every N steps). Default 1.
        """
        self.stage = stage
        self.frequency = frequency
        self.name = name if name is not None else stage.name
        self.record_list = record_list

    def __call__(self, ctx: HookContext, stage: Enum) -> None:
        """
        Record that this hook was called.

        Parameters
        ----------
        ctx : HookContext
            The hook context (unused).
        stage : Enum
            The stage being dispatched (unused).
        """
        self.record_list.append(self.name)
