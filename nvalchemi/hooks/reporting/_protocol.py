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
"""Reporter protocol definitions."""

from __future__ import annotations

from enum import Enum
from typing import Protocol, runtime_checkable

from nvalchemi.hooks._context import HookContext
from nvalchemi.hooks.reporting._state import ReportingState


@runtime_checkable
class Reporter(Protocol):
    """Protocol for reporting sinks owned by ``ReportingOrchestrator``.

    Reporters consume the existing hook context directly. They should not
    require the orchestrator to construct separate workflow event objects.
    Reporters may optionally expose ``rank_zero_only: bool`` to request
    per-reporter rank gating. Reporters that run distributed collectives must
    expose ``requires_all_ranks: bool`` so orchestrator-level rank gating does
    not skip nonzero ranks before a collective. Reporters may also implement
    ``__enter__``, ``__exit__``, or ``close`` for resource lifecycle
    management.
    """

    def report(self, ctx: HookContext, stage: Enum, state: ReportingState) -> None:
        """Consume one reporting event.

        Parameters
        ----------
        ctx : HookContext
            Workflow hook context.
        stage : Enum
            Hook stage being reported.
        state : ReportingState
            Shared reporting state for the orchestrator run.
        """
        ...
