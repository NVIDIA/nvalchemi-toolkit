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
"""Hook-native reporting orchestrator."""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from enum import Enum
from types import TracebackType

from torch import distributed as dist

from nvalchemi.hooks._context import HookContext
from nvalchemi.hooks.reporting._protocol import Reporter
from nvalchemi.hooks.reporting._state import ReportingState

ReportingStage = Enum | str

DEFAULT_REPORT_STAGES: frozenset[str] = frozenset(
    {"AFTER_OPTIMIZER_STEP", "AFTER_STEP"}
)


class ReportingErrorPolicy(str, Enum):
    """Policy used when an individual reporter raises.

    Attributes
    ----------
    RAISE : ReportingErrorPolicy
        Re-raise reporter exceptions.
    WARN : ReportingErrorPolicy
        Emit :class:`UserWarning` and continue to later reporters.
    IGNORE : ReportingErrorPolicy
        Record the error in :class:`ReportingState` and continue silently.
    """

    RAISE = "raise"
    WARN = "warn"
    IGNORE = "ignore"


class ReportingOrchestrator:
    """Fan out hook contexts to reporting sinks.

    ``ReportingOrchestrator`` is itself a normal hook. It uses
    ``_runs_on_stage`` so it can be registered with both training and dynamics
    hook registries while still choosing the workflow stages it observes.

    Parameters
    ----------
    reporters : Sequence[Reporter]
        Reporters to call in order for each reporting event.
    frequency : int, optional
        Run every ``frequency`` workflow steps, using the existing hook
        registry gating. Default ``1``.
    stages : set[Enum | str] | None, optional
        Stages to report. Enum values are matched by identity; strings are
        matched against enum member names. Defaults to
        ``{"AFTER_OPTIMIZER_STEP", "AFTER_STEP"}``, which gives once-per-step
        training and dynamics reporting without importing either workflow.
    rank_zero_only : bool, optional
        If ``True``, suppress child reporters on nonzero ranks unless they
        expose ``requires_all_ranks=True`` for distributed collectives.
        Individual reporters may also expose ``rank_zero_only=True`` to
        request their own gating. Default ``False``.
    error_policy : ReportingErrorPolicy | str, optional
        Reporter failure handling policy. Default ``"raise"``.
    state : ReportingState | None, optional
        Shared reporting state. If omitted, a new state object is created.
    """

    def __init__(
        self,
        reporters: Sequence[Reporter],
        *,
        frequency: int = 1,
        stages: set[ReportingStage] | None = None,
        rank_zero_only: bool = False,
        error_policy: ReportingErrorPolicy | str = ReportingErrorPolicy.RAISE,
        state: ReportingState | None = None,
    ) -> None:
        self.reporters = list(reporters)
        self.frequency = frequency
        self.stage: Enum | None = None
        self.rank_zero_only = rank_zero_only
        self.error_policy = ReportingErrorPolicy(error_policy)
        self.state = state if state is not None else ReportingState()
        self._stages = frozenset(
            stages if stages is not None else DEFAULT_REPORT_STAGES
        )
        self._context_depth = 0
        self._entered_reporters: list[Reporter] = []
        self._disabled_reporter_ids: set[int] = set()
        self._closed = False

    @property
    def global_rank(self) -> int:
        """Return the current distributed rank, or zero outside distributed runs."""
        if dist.is_available() and dist.is_initialized():
            return dist.get_rank()
        return 0

    @property
    def is_rank_zero(self) -> bool:
        """Return whether this process is rank zero."""
        return self.global_rank == 0

    def _runs_on_stage(self, stage: Enum) -> bool:
        """Return whether reporting should run for ``stage``.

        Parameters
        ----------
        stage : Enum
            Hook stage under consideration.

        Returns
        -------
        bool
            ``True`` when the orchestrator should receive this stage.
        """
        return stage in self._stages or stage.name in self._stages

    def __call__(self, ctx: HookContext, stage: Enum) -> None:
        """Dispatch one hook event to child reporters.

        Parameters
        ----------
        ctx : HookContext
            Workflow hook context.
        stage : Enum
            Hook stage being dispatched.
        """
        active_reporters = [
            reporter
            for reporter in self.reporters
            if id(reporter) not in self._disabled_reporter_ids
            and not self._skip_reporter_for_rank(reporter)
        ]
        if not active_reporters:
            return
        self.state.mark_event(ctx, stage)
        for reporter in active_reporters:
            try:
                reporter.report(ctx, stage, self.state)
            except Exception as exc:
                self._handle_reporter_error(
                    reporter, exc, ctx, stage, operation="report"
                )

    def __enter__(self) -> ReportingOrchestrator:
        """Enter reporters that implement the context manager protocol."""
        if self._context_depth > 0:
            self._context_depth += 1
            return self
        self._closed = False
        self._entered_reporters = []
        self._disabled_reporter_ids = set()
        for reporter in self.reporters:
            if self._skip_reporter_for_rank(reporter):
                self._disabled_reporter_ids.add(id(reporter))
                continue
            enter = getattr(reporter, "__enter__", None)
            if enter is not None:
                try:
                    enter()
                except Exception as exc:
                    self._disabled_reporter_ids.add(id(reporter))
                    self._handle_enter_error(reporter, exc)
                else:
                    self._entered_reporters.append(reporter)
        self._context_depth = 1
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Exit reporters without replacing an active workflow exception."""
        if self._context_depth == 0:
            return
        self._context_depth -= 1
        if self._context_depth > 0:
            return
        self._finish_close(exc_type, exc, tb)

    def close(self) -> None:
        """Close reporters in reverse order."""
        self._finish_close(None, None, None)

    def _reporter_rank_zero_only(self, reporter: Reporter) -> bool:
        """Return whether ``reporter`` requests rank-zero-only dispatch."""
        return bool(getattr(reporter, "rank_zero_only", False))

    def _reporter_requires_all_ranks(self, reporter: Reporter) -> bool:
        """Return whether ``reporter`` must be dispatched on every rank."""
        return bool(getattr(reporter, "requires_all_ranks", False))

    def _skip_reporter_for_rank(self, reporter: Reporter) -> bool:
        """Return whether ``reporter`` should be skipped on this rank."""
        if self.is_rank_zero:
            return False
        if self._reporter_requires_all_ranks(reporter):
            return False
        return self.rank_zero_only or self._reporter_rank_zero_only(reporter)

    def _handle_enter_error(self, reporter: Reporter, exc: Exception) -> None:
        """Handle a reporter ``__enter__`` failure."""
        if self.error_policy == ReportingErrorPolicy.RAISE:
            try:
                self._close_reporters(
                    list(self._entered_reporters),
                    type(exc),
                    exc,
                    exc.__traceback__,
                    preserve_workflow_exception=True,
                )
            finally:
                self._entered_reporters = []
                self._closed = True
        self._handle_reporter_error(reporter, exc, operation="enter")

    def _finish_close(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Close reporters once and reset lifecycle state."""
        if self._closed:
            self._context_depth = 0
            return
        try:
            self._close_reporters(
                self.reporters,
                exc_type,
                exc,
                tb,
                preserve_workflow_exception=exc_type is not None,
            )
        finally:
            self._entered_reporters = []
            self._context_depth = 0
            self._closed = True

    def _close_reporters(
        self,
        reporters: Sequence[Reporter],
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
        *,
        preserve_workflow_exception: bool = False,
    ) -> None:
        """Close reporters, preferring ``__exit__`` for entered reporters."""
        errors: list[tuple[str, Exception]] = []
        entered_ids = {id(reporter) for reporter in self._entered_reporters}
        for reporter in reversed(reporters):
            reporter_id = id(reporter)
            if (
                reporter_id in self._disabled_reporter_ids
                and reporter_id not in entered_ids
            ):
                continue
            exit_fn = getattr(reporter, "__exit__", None)
            close_fn = getattr(reporter, "close", None)
            was_entered = reporter_id in entered_ids
            if (not was_entered or exit_fn is None) and close_fn is None:
                continue
            try:
                if was_entered and exit_fn is not None:
                    exit_fn(exc_type, exc, tb)
                else:
                    close_fn()
            except Exception as close_exc:
                message = self._record_reporter_error(
                    reporter,
                    close_exc,
                    operation="close",
                )
                errors.append((message, close_exc))
        self._apply_close_error_policy(errors, preserve_workflow_exception)

    def _apply_close_error_policy(
        self,
        errors: Sequence[tuple[str, Exception]],
        preserve_workflow_exception: bool,
    ) -> None:
        """Apply failure policy after all close attempts have completed."""
        if not errors or self.error_policy == ReportingErrorPolicy.IGNORE:
            return
        if (
            self.error_policy == ReportingErrorPolicy.WARN
            or preserve_workflow_exception
        ):
            for message, _ in errors:
                warnings.warn(message, UserWarning, stacklevel=2)
            return
        raise errors[0][1]

    def _handle_reporter_error(
        self,
        reporter: Reporter,
        exc: Exception,
        ctx: HookContext | None = None,
        stage: Enum | None = None,
        *,
        operation: str,
        preserve_workflow_exception: bool = False,
    ) -> None:
        """Apply the configured reporter failure policy."""
        message = self._record_reporter_error(
            reporter,
            exc,
            ctx=ctx,
            stage=stage,
            operation=operation,
        )
        if self.error_policy == ReportingErrorPolicy.IGNORE:
            return
        if (
            self.error_policy == ReportingErrorPolicy.WARN
            or preserve_workflow_exception
        ):
            warnings.warn(message, UserWarning, stacklevel=2)
            return
        raise exc

    def _record_reporter_error(
        self,
        reporter: Reporter,
        exc: Exception,
        ctx: HookContext | None = None,
        stage: Enum | None = None,
        *,
        operation: str,
    ) -> str:
        """Record a reporter error message and return its text."""
        message = (
            f"{type(reporter).__name__} failed during {operation}: "
            f"{type(exc).__name__}: {exc}"
        )
        self.state.add_message(
            "error",
            message,
            reporter=reporter,
            ctx=ctx,
            stage=stage,
        )
        return message
