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
"""Tests for TensorBoard reporting."""

from __future__ import annotations

from enum import Enum, auto

import pytest
import torch

from nvalchemi._optional import OptionalDependency, OptionalDependencyError
from nvalchemi.hooks import TrainContext
from nvalchemi.hooks.reporting import (
    ReportingState,
    TensorBoardReporter,
)


class _ReportStage(Enum):
    AFTER_OPTIMIZER_STEP = auto()


class _RecordingWriter:
    def __init__(self) -> None:
        self.scalars: list[tuple[str, float, int | None]] = []
        self.flushed = 0
        self.closed = 0

    def add_scalar(
        self,
        tag: str,
        scalar_value: float,
        global_step: int | None = None,
    ) -> None:
        self.scalars.append((tag, scalar_value, global_step))

    def flush(self) -> None:
        self.flushed += 1

    def close(self) -> None:
        self.closed += 1


def _ctx(*, global_rank: int = 0, loss: torch.Tensor | None = None) -> TrainContext:
    return TrainContext(
        batch=object(),
        global_rank=global_rank,
        step_count=17,
        batch_count=19,
        epoch_step_count=3,
        epoch=5,
        loss=loss,
    )


def _state(ctx: TrainContext) -> ReportingState:
    state = ReportingState()
    state.mark_event(ctx, _ReportStage.AFTER_OPTIMIZER_STEP)
    return state


@pytest.fixture(autouse=True)
def _tensorboard_available(monkeypatch) -> None:
    dep = OptionalDependency.TENSORBOARD
    monkeypatch.setattr(dep, "_available", True)
    monkeypatch.setattr(dep, "_import_error", None)


def test_tensorboard_reporter_writes_scalar_tags_with_step(tmp_path) -> None:
    writer = _RecordingWriter()
    ctx = _ctx(loss=torch.tensor(2.5))
    reporter = TensorBoardReporter(
        tmp_path / "runs",
        custom_scalars={"metric": lambda context, stage: 9.0},  # noqa: ARG005
        tag_prefix="train",
        writer=writer,
    )

    reporter.report(ctx, _ReportStage.AFTER_OPTIMIZER_STEP, _state(ctx))

    assert writer.scalars == [
        ("train/loss/total", 2.5, 17),
        ("train/metric", 9.0, 17),
    ]
    assert writer.flushed == 1


def test_tensorboard_reporter_defaults_to_rank_zero_only(tmp_path) -> None:
    writer = _RecordingWriter()
    ctx = _ctx(global_rank=1, loss=torch.tensor(2.5))
    reporter = TensorBoardReporter(tmp_path / "runs", writer=writer)

    reporter.report(ctx, _ReportStage.AFTER_OPTIMIZER_STEP, _state(ctx))

    assert reporter.rank_zero_only is True
    assert writer.scalars == []


def test_tensorboard_reporter_requires_rank_token_for_all_rank_writes(
    tmp_path,
) -> None:
    with pytest.raises(ValueError, match="must contain '\\{rank\\}'"):
        TensorBoardReporter(tmp_path / "runs", rank_zero_only=False)


def test_tensorboard_reporter_all_rank_write_accepts_rank_safe_log_dir(
    tmp_path,
) -> None:
    writer = _RecordingWriter()
    ctx = _ctx(global_rank=3, loss=torch.tensor(2.5))
    reporter = TensorBoardReporter(
        tmp_path / "runs-rank-{rank}",
        rank_zero_only=False,
        writer=writer,
    )

    reporter.report(ctx, _ReportStage.AFTER_OPTIMIZER_STEP, _state(ctx))

    assert writer.scalars == [("loss/total", 2.5, 17)]


def test_tensorboard_reduction_uses_all_rank_dispatch_and_rank_zero_write(
    tmp_path,
) -> None:
    writer = _RecordingWriter()
    ctx = _ctx(global_rank=0, loss=torch.tensor(2.5))
    reporter = TensorBoardReporter(
        tmp_path / "runs",
        rank_reduction="mean",
        writer=writer,
    )

    reporter.report(ctx, _ReportStage.AFTER_OPTIMIZER_STEP, _state(ctx))

    assert reporter.rank_zero_only is False
    assert writer.scalars == [("loss/total", 2.5, 17)]


def test_tensorboard_close_closes_writer(tmp_path) -> None:
    writer = _RecordingWriter()
    reporter = TensorBoardReporter(tmp_path / "runs", writer=writer)

    reporter.close()

    assert writer.closed == 1


def test_tensorboard_missing_extra_uses_optional_dependency_error(
    tmp_path,
    monkeypatch,
) -> None:
    dep = OptionalDependency.TENSORBOARD
    monkeypatch.setattr(dep, "_available", False)
    monkeypatch.setattr(dep, "_import_error", ImportError("missing tensorboard"))

    with pytest.raises(
        OptionalDependencyError, match="nvalchemi-toolkit\\[tensorboard\\]"
    ):
        TensorBoardReporter(tmp_path / "runs")
