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
"""Tests for Rich reporting."""

from __future__ import annotations

from enum import Enum, auto
from io import StringIO
from types import SimpleNamespace

import pytest
import torch
from rich.console import Console

from nvalchemi.hooks import DynamicsContext, TrainContext
from nvalchemi.hooks.reporting import (
    BaseRichLayout,
    DynamicsRichLayout,
    RankReduction,
    ReportingState,
    RichReporter,
    TrainingRichLayout,
)


class _ReportStage(Enum):
    AFTER_OPTIMIZER_STEP = auto()
    AFTER_STEP = auto()


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


def _state(
    ctx: DynamicsContext | TrainContext,
    stage: _ReportStage = _ReportStage.AFTER_OPTIMIZER_STEP,
) -> ReportingState:
    state = ReportingState()
    state.mark_event(ctx, stage)
    return state


def _dynamics_ctx(*, global_rank: int = 0) -> DynamicsContext:
    batch = SimpleNamespace(
        num_graphs=2,
        energy=torch.tensor([[-1.0], [-3.0]]),
        forces=torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [0.0, 2.0, 0.0],
                [0.0, 0.0, 3.0],
            ]
        ),
        velocities=torch.zeros(3, 3),
        atomic_masses=torch.ones(3),
        batch_idx=torch.tensor([0, 0, 1]),
        num_nodes_per_graph=torch.tensor([2, 1]),
        status=torch.tensor([[0], [1]]),
    )
    return DynamicsContext(
        batch=batch,
        global_rank=global_rank,
        step_count=23,
        converged_mask=torch.tensor([False, True]),
        workflow=SimpleNamespace(exit_status=1),
    )


def _console(buffer: StringIO) -> Console:
    return Console(
        file=buffer,
        force_terminal=False,
        color_system=None,
        width=120,
    )


def test_rich_reporter_prints_live_dashboard() -> None:
    buffer = StringIO()
    ctx = _ctx(loss=torch.tensor(2.5))
    reporter = RichReporter(
        custom_scalars={"metric": lambda context, stage: 9.0},  # noqa: ARG005
        title="training",
        console=_console(buffer),
    )

    reporter.report(ctx, _ReportStage.AFTER_OPTIMIZER_STEP, _state(ctx))

    output = buffer.getvalue()
    assert "training" in output
    assert "AFTER_OPTIMIZER_STEP" in output
    assert "step 17" in output
    assert "loss/total" in output
    assert "2.5" in output
    assert "metric" in output
    assert "9" in output
    assert "rank=0" in output
    assert "event=1" in output
    assert "History" in output
    assert reporter.history["loss/total"] == ((17, 2.5),)


def test_rich_reporter_defaults_to_rank_zero_only() -> None:
    buffer = StringIO()
    ctx = _ctx(global_rank=1, loss=torch.tensor(2.5))
    reporter = RichReporter(console=_console(buffer))

    reporter.report(ctx, _ReportStage.AFTER_OPTIMIZER_STEP, _state(ctx))

    assert reporter.rank_zero_only is True
    assert buffer.getvalue() == ""


def test_rich_reporter_reduction_uses_all_rank_dispatch_and_rank_zero_write() -> None:
    buffer = StringIO()
    ctx = _ctx(loss=torch.tensor(2.5))
    reporter = RichReporter(
        rank_reduction=RankReduction.MEAN,
        console=_console(buffer),
    )

    reporter.report(ctx, _ReportStage.AFTER_OPTIMIZER_STEP, _state(ctx))

    assert reporter.rank_zero_only is False
    assert "loss/total" in buffer.getvalue()


def test_rich_reporter_reduction_skips_nonzero_rank_write() -> None:
    buffer = StringIO()
    ctx = _ctx(global_rank=1, loss=torch.tensor(2.5))
    reporter = RichReporter(
        rank_reduction=RankReduction.MEAN,
        console=_console(buffer),
    )

    reporter.report(ctx, _ReportStage.AFTER_OPTIMIZER_STEP, _state(ctx))

    assert buffer.getvalue() == ""


def test_rich_reporter_reduction_context_starts_live_only_on_rank_zero() -> None:
    buffer = StringIO()
    reporter = RichReporter(
        rank_reduction=RankReduction.MEAN,
        console=_console(buffer),
        transient=True,
    )

    with reporter:
        assert reporter.rank_zero_only is False
        assert reporter._live is None

        nonzero_ctx = _ctx(global_rank=1, loss=torch.tensor(2.5))
        reporter.report(
            nonzero_ctx,
            _ReportStage.AFTER_OPTIMIZER_STEP,
            _state(nonzero_ctx),
        )
        assert reporter._live is None
        assert buffer.getvalue() == ""

        rank_zero_ctx = _ctx(loss=torch.tensor(2.5))
        reporter.report(
            rank_zero_ctx,
            _ReportStage.AFTER_OPTIMIZER_STEP,
            _state(rank_zero_ctx),
        )
        assert reporter._live is not None

    assert reporter._live is None


def test_rich_reporter_max_scalars_truncates_output() -> None:
    buffer = StringIO()
    ctx = _ctx(loss=torch.tensor(2.5))
    reporter = RichReporter(
        custom_scalars={
            "first": lambda context, stage: 1.0,  # noqa: ARG005
            "second": lambda context, stage: 2.0,  # noqa: ARG005
        },
        max_scalars=1,
        console=_console(buffer),
    )

    reporter.report(ctx, _ReportStage.AFTER_OPTIMIZER_STEP, _state(ctx))

    output = buffer.getvalue()
    assert "omitted" in output
    assert "2 omitted" in output


def test_rich_reporter_seed_history_supports_preview_data() -> None:
    buffer = StringIO()
    reporter = RichReporter(title="preview", console=_console(buffer))

    snapshot = reporter.seed_history(
        {
            "loss/total": [1.0, 0.5, 0.25],
            "optimizer/lr": [1e-3, 5e-4, 1e-4],
        },
        steps=[10, 20, 30],
        epoch=2,
        batch_count=64,
    )
    reporter.console.print(reporter.renderable())

    output = buffer.getvalue()
    assert snapshot.scalars == {"loss/total": 0.25, "optimizer/lr": 1e-4}
    assert reporter.history["loss/total"] == ((10, 1.0), (20, 0.5), (30, 0.25))
    assert "preview" in output
    assert "loss/total" in output
    assert "optimizer/lr" in output


def test_rich_reporter_preview_renders_default_dashboard() -> None:
    buffer = StringIO()

    RichReporter.preview(console=_console(buffer), title="preview")

    output = buffer.getvalue()
    assert "preview" in output
    assert "loss/total" in output
    assert "optimizer/lr" in output


def test_rich_reporter_layout_names_resolve_to_layouts() -> None:
    training = RichReporter(layout="training")
    dynamics = RichReporter(layout="dynamics")
    custom = DynamicsRichLayout()

    custom_reporter = RichReporter(layout=custom)

    assert isinstance(training.layout, TrainingRichLayout)
    assert isinstance(dynamics.layout, DynamicsRichLayout)
    assert custom_reporter.layout is custom
    assert isinstance(training.layout, BaseRichLayout)


def test_rich_layouts_are_available_from_workflow_submodules() -> None:
    from nvalchemi.hooks.reporting.layouts.dynamics import (
        DynamicsRichLayout as Dynamics,
    )
    from nvalchemi.hooks.reporting.layouts.train import TrainingRichLayout as Training

    assert isinstance(Training(), TrainingRichLayout)
    assert isinstance(Dynamics(), DynamicsRichLayout)


def test_rich_reporter_dynamics_preview_uses_dynamics_metrics() -> None:
    buffer = StringIO()

    RichReporter.preview(console=_console(buffer), layout="dynamics", title="preview")

    output = buffer.getvalue()
    assert "preview" in output
    assert "dynamics" in output
    assert "AFTER_STEP" in output
    assert "AFTER_OPTIMIZER_STEP" not in output
    assert "fmax" in output
    assert "temperature" in output
    assert "converged_fraction" in output
    assert "loss/total" not in output
    assert "epoch=" not in output
    assert "batch=" not in output


def test_rich_reporter_dynamics_layout_collects_default_metrics() -> None:
    buffer = StringIO()
    ctx = _dynamics_ctx()
    reporter = RichReporter(
        layout="dynamics",
        console=_console(buffer),
        max_plots=0,
    )

    reporter.report(ctx, _ReportStage.AFTER_STEP, _state(ctx, _ReportStage.AFTER_STEP))

    output = buffer.getvalue()
    assert "dynamics" in output
    assert "Observables" in output
    assert "Convergence" in output
    assert "Dynamics Traces" in output
    assert "energy" in output
    assert "fmax" in output
    assert "temperature" in output
    assert "converged_fraction" in output
    assert "active_fraction" in output
    assert reporter.history["energy"] == ((23, -2.0),)
    assert reporter.history["fmax"] == ((23, 3.0),)
    assert reporter.history["temperature"] == ((23, 0.0),)
    assert reporter.history["converged_fraction"] == ((23, 0.5),)
    assert reporter.history["active_fraction"] == ((23, 0.5),)


def test_rich_reporter_rejects_unknown_layout() -> None:
    with pytest.raises(ValueError, match="layout"):
        RichReporter(layout="unknown")
    with pytest.raises(TypeError, match="layout objects"):
        RichReporter(layout=object())


def test_rich_reporter_live_context_updates_and_closes() -> None:
    buffer = StringIO()
    ctx = _ctx(loss=torch.tensor(2.5))
    reporter = RichReporter(console=_console(buffer), transient=True)

    with reporter:
        assert reporter._live is not None
        reporter.report(ctx, _ReportStage.AFTER_OPTIMIZER_STEP, _state(ctx))

    assert reporter._live is None
    assert reporter.history["loss/total"] == ((17, 2.5),)


def test_rich_reporter_validates_formatting_options() -> None:
    with pytest.raises(ValueError, match="precision"):
        RichReporter(precision=-1)
    with pytest.raises(ValueError, match="max_scalars"):
        RichReporter(max_scalars=0)
    with pytest.raises(ValueError, match="history_size"):
        RichReporter(history_size=0)
    with pytest.raises(ValueError, match="max_plots"):
        RichReporter(max_plots=-1)
    with pytest.raises(ValueError, match="plot_height"):
        RichReporter(plot_height=3)
    with pytest.raises(ValueError, match="refresh_per_second"):
        RichReporter(refresh_per_second=0)
