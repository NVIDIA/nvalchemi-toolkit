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
"""Shared reporting runtime state."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from nvalchemi.hooks._context import HookContext

MessageLevel = Literal["info", "warning", "error"]


@dataclass(frozen=True, kw_only=True)
class ReporterMessage:
    """Message emitted by reporting infrastructure.

    Attributes
    ----------
    level : {"info", "warning", "error"}
        Severity level for the message.
    message : str
        Human-readable message.
    reporter : str | None
        Reporter class name associated with the message, when available.
    stage : str | None
        Hook stage name associated with the message, when available.
    step_count : int | None
        Workflow step count associated with the message, when available.
    global_rank : int | None
        Distributed rank associated with the message, when available.
    timestamp_s : float
        Wall-clock timestamp from :func:`time.time`.
    """

    level: MessageLevel
    message: str
    reporter: str | None = None
    stage: str | None = None
    step_count: int | None = None
    global_rank: int | None = None
    timestamp_s: float = field(default_factory=time.time)


@dataclass(kw_only=True)
class ReportingState:
    """Mutable state shared by a reporting orchestrator and its reporters.

    The state object intentionally stores only orchestration metadata:
    counters, timestamps, recent messages, and an extensible metadata mapping.
    Workflow values such as losses, schedulers, or dynamics counters should be
    read from the hook context rather than duplicated here.

    Attributes
    ----------
    max_messages : int
        Maximum number of recent messages retained.
    started_at_s : float
        Monotonic time when the state was created.
    event_count : int
        Number of reporting events dispatched by the orchestrator.
    last_event_at_s : float | None
        Monotonic time of the latest reporting event.
    last_stage : str | None
        Name of the latest reported hook stage.
    last_step_count : int | None
        Step count from the latest reported context, when available.
    last_global_rank : int | None
        Rank from the latest reported context, when available.
    messages : list[ReporterMessage]
        Bounded list of recent reporting messages.
    metadata : dict[str, Any]
        Scratch space for reporters that need shared per-run state.
    """

    max_messages: int = 100
    started_at_s: float = field(default_factory=time.monotonic)
    event_count: int = 0
    last_event_at_s: float | None = None
    last_stage: str | None = None
    last_step_count: int | None = None
    last_global_rank: int | None = None
    messages: list[ReporterMessage] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def mark_event(self, ctx: HookContext, stage: Enum) -> None:
        """Record that a reporting event was dispatched.

        Parameters
        ----------
        ctx : HookContext
            Workflow context passed to the reporting orchestrator.
        stage : Enum
            Hook stage being reported.
        """
        self.event_count += 1
        self.last_event_at_s = time.monotonic()
        self.last_stage = stage.name
        self.last_step_count = getattr(ctx, "step_count", None)
        self.last_global_rank = ctx.global_rank

    def add_message(
        self,
        level: MessageLevel,
        message: str,
        *,
        reporter: object | None = None,
        ctx: HookContext | None = None,
        stage: Enum | None = None,
    ) -> ReporterMessage:
        """Append a bounded recent message.

        Parameters
        ----------
        level : {"info", "warning", "error"}
            Message severity.
        message : str
            Human-readable message.
        reporter : object | None, optional
            Reporter associated with the message.
        ctx : HookContext | None, optional
            Context associated with the message.
        stage : Enum | None, optional
            Hook stage associated with the message.

        Returns
        -------
        ReporterMessage
            The message object appended to :attr:`messages`.
        """
        entry = ReporterMessage(
            level=level,
            message=message,
            reporter=type(reporter).__name__ if reporter is not None else None,
            stage=stage.name if stage is not None else None,
            step_count=getattr(ctx, "step_count", None) if ctx is not None else None,
            global_rank=ctx.global_rank if ctx is not None else None,
        )
        self.messages.append(entry)
        if len(self.messages) > self.max_messages:
            del self.messages[: len(self.messages) - self.max_messages]
        return entry
