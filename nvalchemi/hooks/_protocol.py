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
"""Hook protocol definition."""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from nvalchemi.hooks._context import HookContext


@runtime_checkable
class Hook(Protocol):
    """Protocol for hooks that observe or modify workflow state.

    Attributes
    ----------
    frequency : int
        How often the hook runs (every N steps).
    stage : Enum | None
        The stage enum value at which this hook runs, or ``None`` for
        hooks that are stage-agnostic until registered with a specific
        engine.
    """

    frequency: int
    stage: Enum | None

    def __call__(self, ctx: HookContext, stage: Enum) -> None:
        """Execute the hook.

        Only called when the registry determines the hook should fire at
        the dispatched stage.  By default, hooks fire when
        ``stage == self.stage``.  To fire at multiple stages, define a
        ``_runs_on_stage(self, stage: Enum) -> bool`` method that returns
        ``True`` for each relevant stage.

        Frequency gating is handled by the registry: hooks are only
        called when ``step_count % frequency == 0``.

        Parameters
        ----------
        ctx : HookContext
            Snapshot of the current workflow state.
        stage : Enum
            The stage being dispatched.
        """
        ...
