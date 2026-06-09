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
"""Tests for hook-native reporting orchestration."""

from __future__ import annotations

import warnings
from enum import Enum, auto
from typing import Any

import pytest

from nvalchemi.hooks import HookContext, HookRegistryMixin, TrainContext
from nvalchemi.hooks.reporting import (
    Reporter,
    ReportingErrorPolicy,
    ReportingOrchestrator,
    ReportingState,
)


class _ReportStage(Enum):
    AFTER_OPTIMIZER_STEP = auto()
    AFTER_STEP = auto()
    BEFORE_STEP = auto()
    EXACT = auto()


class _Reporter:
    def __init__(
        self,
        name: str = "reporter",
        events: list[str] | None = None,
        *,
        rank_zero_only: bool = False,
        requires_all_ranks: bool = False,
        fail_report: bool = False,
    ) -> None:
        self.name = name
        self.events = events
        self.rank_zero_only = rank_zero_only
        self.requires_all_ranks = requires_all_ranks
        self.fail_report = fail_report
        self.calls: list[tuple[HookContext, Enum, ReportingState]] = []

    def report(self, ctx: HookContext, stage: Enum, state: ReportingState) -> None:
        if self.fail_report:
            raise RuntimeError("report failed")
        self.calls.append((ctx, stage, state))
        if self.events is not None:
            self.events.append(f"report:{self.name}:{stage.name}:{state.event_count}")


class _ContextReporter:
    def __init__(
        self,
        name: str,
        events: list[str],
        *,
        rank_zero_only: bool = False,
        requires_all_ranks: bool = False,
        fail_enter: bool = False,
        fail_exit: bool = False,
    ) -> None:
        self.name = name
        self.events = events
        self.rank_zero_only = rank_zero_only
        self.requires_all_ranks = requires_all_ranks
        self.fail_enter = fail_enter
        self.fail_exit = fail_exit

    def __enter__(self) -> _ContextReporter:
        self.events.append(f"enter:{self.name}")
        if self.fail_enter:
            raise RuntimeError("enter failed")
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: Any,
    ) -> None:
        self.events.append(f"exit:{self.name}")
        if self.fail_exit:
            raise RuntimeError("exit failed")

    def close(self) -> None:
        self.events.append(f"close:{self.name}")

    def report(self, ctx: HookContext, stage: Enum, state: ReportingState) -> None:
        self.events.append(f"report:{self.name}")


class _CloseOnlyReporter:
    def __init__(
        self,
        name: str,
        events: list[str],
        *,
        fail_close: bool = False,
    ) -> None:
        self.name = name
        self.events = events
        self.fail_close = fail_close

    def close(self) -> None:
        self.events.append(f"close:{self.name}")
        if self.fail_close:
            raise RuntimeError("close failed")

    def report(self, ctx: HookContext, stage: Enum, state: ReportingState) -> None:
        self.events.append(f"report:{self.name}")


class _Engine(HookRegistryMixin):
    def __init__(self, hooks: list[Reporter]) -> None:
        self.step_count = 0
        self._init_hooks(hooks)

    def _build_context(self, batch: object) -> HookContext:
        return _ctx(step_count=self.step_count)


class _RankedReportingOrchestrator(ReportingOrchestrator):
    def __init__(
        self,
        reporters: list[Reporter],
        *,
        global_rank: int,
        **kwargs: Any,
    ) -> None:
        super().__init__(reporters, **kwargs)
        self._global_rank = global_rank

    @property
    def global_rank(self) -> int:
        return self._global_rank


def _ctx(*, global_rank: int = 0, step_count: int = 7) -> TrainContext:
    return TrainContext(
        batch=object(),
        global_rank=global_rank,
        step_count=step_count,
    )


class TestReportingOrchestratorDispatch:
    def test_default_stages_cover_training_and_dynamics(self) -> None:
        hook = ReportingOrchestrator([])

        assert hook._runs_on_stage(_ReportStage.AFTER_OPTIMIZER_STEP)
        assert hook._runs_on_stage(_ReportStage.AFTER_STEP)
        assert not hook._runs_on_stage(_ReportStage.BEFORE_STEP)

    def test_exact_enum_stages_override_defaults(self) -> None:
        hook = ReportingOrchestrator([], stages={_ReportStage.EXACT})

        assert hook._runs_on_stage(_ReportStage.EXACT)
        assert not hook._runs_on_stage(_ReportStage.AFTER_STEP)

    def test_stage_name_strings_match_enum_names(self) -> None:
        hook = ReportingOrchestrator([], stages={"EXACT"})

        assert hook._runs_on_stage(_ReportStage.EXACT)
        assert not hook._runs_on_stage(_ReportStage.AFTER_STEP)

    def test_reporters_receive_original_context_stage_and_shared_state(self) -> None:
        events: list[str] = []
        first = _Reporter("first", events)
        second = _Reporter("second", events)
        hook = ReportingOrchestrator([first, second])
        ctx = _ctx(step_count=11)

        hook(ctx, _ReportStage.AFTER_STEP)

        assert events == [
            "report:first:AFTER_STEP:1",
            "report:second:AFTER_STEP:1",
        ]
        assert first.calls == [(ctx, _ReportStage.AFTER_STEP, hook.state)]
        assert second.calls == [(ctx, _ReportStage.AFTER_STEP, hook.state)]
        assert hook.state.last_stage == "AFTER_STEP"
        assert hook.state.last_step_count == 11

    def test_frequency_gating_comes_from_hook_registry(self) -> None:
        reporter = _Reporter("reporter")
        hook = ReportingOrchestrator([reporter], frequency=2)
        engine = _Engine([hook])

        engine.step_count = 1
        engine._call_hooks(_ReportStage.AFTER_STEP, object())
        engine.step_count = 2
        engine._call_hooks(_ReportStage.AFTER_STEP, object())

        assert len(reporter.calls) == 1
        assert reporter.calls[0][0].step_count == 2

    def test_orchestrator_rank_zero_only_skips_state_and_reporters(self) -> None:
        reporter = _Reporter("reporter")
        nonzero = _RankedReportingOrchestrator(
            [reporter],
            global_rank=1,
            rank_zero_only=True,
        )

        nonzero(_ctx(global_rank=0), _ReportStage.AFTER_STEP)
        assert nonzero.state.event_count == 0
        assert reporter.calls == []

        rank_zero = _RankedReportingOrchestrator(
            [reporter],
            global_rank=0,
            rank_zero_only=True,
        )
        rank_zero(_ctx(global_rank=1), _ReportStage.AFTER_STEP)
        assert rank_zero.state.event_count == 1
        assert len(reporter.calls) == 1

    def test_orchestrator_rank_zero_only_dispatches_all_rank_reporters(self) -> None:
        gated = _Reporter("gated")
        collective = _Reporter("collective", requires_all_ranks=True)
        hook = _RankedReportingOrchestrator(
            [gated, collective],
            global_rank=1,
            rank_zero_only=True,
        )

        hook(_ctx(global_rank=1), _ReportStage.AFTER_STEP)

        assert gated.calls == []
        assert len(collective.calls) == 1
        assert hook.state.event_count == 1

    def test_reporter_rank_zero_only_skips_only_that_reporter(self) -> None:
        gated = _Reporter("gated", rank_zero_only=True)
        ungated = _Reporter("ungated")
        hook = _RankedReportingOrchestrator([gated, ungated], global_rank=1)

        hook(_ctx(global_rank=0), _ReportStage.AFTER_STEP)

        assert gated.calls == []
        assert len(ungated.calls) == 1
        assert hook.state.event_count == 1


class TestReportingOrchestratorFailures:
    @pytest.mark.parametrize(
        ("policy", "expected_later_calls"),
        [
            (ReportingErrorPolicy.RAISE, 0),
            (ReportingErrorPolicy.WARN, 1),
            (ReportingErrorPolicy.IGNORE, 1),
        ],
    )
    def test_report_failure_policy_records_message_and_controls_fanout(
        self,
        policy: ReportingErrorPolicy,
        expected_later_calls: int,
    ) -> None:
        later = _Reporter("later")
        hook = ReportingOrchestrator(
            [_Reporter(fail_report=True), later],
            error_policy=policy,
        )
        ctx = _ctx(global_rank=2)

        if policy == ReportingErrorPolicy.RAISE:
            with pytest.raises(RuntimeError, match="report failed"):
                hook(ctx, _ReportStage.AFTER_STEP)
        elif policy == ReportingErrorPolicy.WARN:
            with pytest.warns(UserWarning, match="failed during report"):
                hook(ctx, _ReportStage.AFTER_STEP)
        else:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                hook(ctx, _ReportStage.AFTER_STEP)
            assert caught == []

        assert len(later.calls) == expected_later_calls
        assert len(hook.state.messages) == 1
        message = hook.state.messages[0]
        assert message.message.startswith("_Reporter failed during report")
        assert message.stage == "AFTER_STEP"
        assert message.step_count == 7
        assert message.global_rank == 2


class TestReportingOrchestratorLifecycle:
    def test_nested_context_enters_and_exits_once(self) -> None:
        events: list[str] = []
        hook = ReportingOrchestrator([_ContextReporter("reporter", events)])

        with hook:
            with hook:
                assert events == ["enter:reporter"]

        assert events == ["enter:reporter", "exit:reporter"]

    def test_context_exits_in_reverse_order_and_prefers_exit_over_close(self) -> None:
        events: list[str] = []
        hook = ReportingOrchestrator(
            [
                _ContextReporter("first", events),
                _ContextReporter("second", events),
            ]
        )

        with hook:
            pass

        assert events == [
            "enter:first",
            "enter:second",
            "exit:second",
            "exit:first",
        ]

    def test_close_only_reporters_close_in_reverse_order_once(self) -> None:
        events: list[str] = []
        hook = ReportingOrchestrator(
            [
                _CloseOnlyReporter("first", events),
                _CloseOnlyReporter("second", events),
            ]
        )

        hook.close()
        hook.close()

        assert events == ["close:second", "close:first"]

    def test_close_inside_context_prevents_double_exit(self) -> None:
        events: list[str] = []
        hook = ReportingOrchestrator([_ContextReporter("reporter", events)])

        with hook:
            hook.close()

        assert events == ["enter:reporter", "exit:reporter"]

    def test_enter_failure_unwinds_already_entered_reporters(self) -> None:
        events: list[str] = []
        hook = ReportingOrchestrator(
            [
                _ContextReporter("first", events),
                _ContextReporter("second", events, fail_enter=True),
            ]
        )

        with pytest.raises(RuntimeError, match="enter failed"):
            with hook:
                pass

        assert events == ["enter:first", "enter:second", "exit:first"]
        assert hook.state.messages[-1].message.startswith(
            "_ContextReporter failed during enter"
        )

    def test_close_failure_still_attempts_remaining_reporters(self) -> None:
        events: list[str] = []
        hook = ReportingOrchestrator(
            [
                _CloseOnlyReporter("first", events),
                _CloseOnlyReporter("second", events, fail_close=True),
            ]
        )

        with pytest.raises(RuntimeError, match="close failed"):
            hook.close()

        assert events == ["close:second", "close:first"]
        assert hook.state.messages[-1].message.startswith(
            "_CloseOnlyReporter failed during close"
        )

    def test_cleanup_failure_warns_without_replacing_workflow_exception(self) -> None:
        events: list[str] = []
        hook = ReportingOrchestrator(
            [_ContextReporter("reporter", events, fail_exit=True)]
        )

        with pytest.warns(UserWarning, match="failed during close"):
            with pytest.raises(ValueError, match="workflow failed"):
                with hook:
                    raise ValueError("workflow failed")

        assert events == ["enter:reporter", "exit:reporter"]

    def test_failed_enter_reporter_is_disabled_under_non_raising_policy(self) -> None:
        events: list[str] = []
        failed = _ContextReporter("failed", events, fail_enter=True)
        active = _Reporter("active", events)
        hook = ReportingOrchestrator(
            [failed, active],
            error_policy=ReportingErrorPolicy.WARN,
        )

        with pytest.warns(UserWarning, match="failed during enter"):
            with hook:
                hook(_ctx(), _ReportStage.AFTER_STEP)

        assert events == [
            "enter:failed",
            "report:active:AFTER_STEP:1",
        ]

    def test_rank_zero_only_orchestrator_skips_lifecycle_on_nonzero_rank(
        self,
    ) -> None:
        events: list[str] = []
        hook = _RankedReportingOrchestrator(
            [_ContextReporter("reporter", events)],
            global_rank=1,
            rank_zero_only=True,
        )

        with hook:
            pass
        hook.close()

        assert events == []

    def test_rank_zero_only_orchestrator_enters_all_rank_reporters_on_nonzero_rank(
        self,
    ) -> None:
        events: list[str] = []
        hook = _RankedReportingOrchestrator(
            [_ContextReporter("reporter", events, requires_all_ranks=True)],
            global_rank=1,
            rank_zero_only=True,
        )

        with hook:
            hook(_ctx(global_rank=1), _ReportStage.AFTER_STEP)

        assert events == ["enter:reporter", "report:reporter", "exit:reporter"]

    def test_rank_zero_only_reporter_skips_lifecycle_on_nonzero_rank(
        self,
    ) -> None:
        events: list[str] = []
        hook = _RankedReportingOrchestrator(
            [_ContextReporter("reporter", events, rank_zero_only=True)],
            global_rank=1,
        )

        with hook:
            pass
        hook.close()

        assert events == []


class TestReportingState:
    def test_state_tracks_event_metadata_and_bounds_messages(self) -> None:
        state = ReportingState(max_messages=2)
        ctx = _ctx(global_rank=3, step_count=19)

        state.mark_event(ctx, _ReportStage.AFTER_STEP)

        assert state.event_count == 1
        assert state.last_stage == "AFTER_STEP"
        assert state.last_step_count == 19
        assert state.last_global_rank == 3

        state.add_message("info", "first", ctx=ctx, stage=_ReportStage.AFTER_STEP)
        state.add_message("warning", "second", ctx=ctx, stage=_ReportStage.BEFORE_STEP)
        state.add_message("error", "third", ctx=ctx, stage=_ReportStage.EXACT)

        assert [message.message for message in state.messages] == ["second", "third"]
        assert state.messages[-1].stage == "EXACT"
        assert state.messages[-1].step_count == 19
        assert state.messages[-1].global_rank == 3
