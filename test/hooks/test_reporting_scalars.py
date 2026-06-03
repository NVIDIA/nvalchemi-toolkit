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
"""Tests for reporting scalar extraction and JSONL output."""

from __future__ import annotations

import json
from enum import Enum, auto

import pytest
import torch

from nvalchemi.hooks import TrainContext
from nvalchemi.hooks.reporting import (
    JSONLMode,
    JSONLReporter,
    RankReduction,
    ReportingState,
    collect_scalars,
    extract_loss_scalars,
    extract_scalars,
)


class _ReportStage(Enum):
    AFTER_OPTIMIZER_STEP = auto()


def _ctx(
    *,
    global_rank: int = 2,
    loss: torch.Tensor | None = None,
    losses: dict[str, object] | None = None,
    optimizers: list[torch.optim.Optimizer] | None = None,
) -> TrainContext:
    return TrainContext(
        batch=object(),
        global_rank=global_rank,
        step_count=17,
        batch_count=19,
        epoch_step_count=3,
        epoch=5,
        loss=loss,
        losses=losses,
        optimizers=optimizers or [],
    )


def test_extract_loss_scalars_handles_simple_training_losses() -> None:
    ctx = _ctx(
        loss=torch.tensor(1.5),
        losses={
            "energy": torch.tensor(0.4),
            "force": torch.tensor(0.1),
        },
    )

    scalars = extract_loss_scalars(ctx)

    assert scalars == pytest.approx(
        {
            "loss/total": 1.5,
            "loss/energy": 0.4,
            "loss/force": 0.1,
        }
    )


def test_extract_loss_scalars_handles_composed_loss_output() -> None:
    ctx = _ctx(
        loss=torch.tensor(99.0),
        losses={
            "total_loss": torch.tensor(3.0),
            "per_component_total": {
                "energy": torch.tensor(1.0),
                "force": torch.tensor([2.0]),
            },
            "per_component_weight": {"energy": 0.25, "force": 0.75},
            "per_component_raw_weight": {"energy": 1.0, "force": 3.0},
            "per_component_sample": {
                "energy": torch.tensor([1.0, 3.0]),
                "force": torch.tensor([2.0, 6.0]),
            },
        },
    )

    scalars = extract_loss_scalars(ctx)

    assert scalars == pytest.approx(
        {
            "loss/total": 3.0,
            "loss/energy/total": 1.0,
            "loss/force/total": 2.0,
            "loss/energy/weight": 0.25,
            "loss/force/weight": 0.75,
            "loss/energy/raw_weight": 1.0,
            "loss/force/raw_weight": 3.0,
            "loss/energy/sample_mean": 2.0,
            "loss/force/sample_mean": 4.0,
        }
    )


def test_extract_scalars_flattens_nested_mapping() -> None:
    scalars = extract_scalars(
        {
            "outer": {
                "inner": torch.tensor(2.0),
                "flag": True,
            },
            "plain": 3,
        },
        prefix="custom",
    )

    assert scalars == {
        "custom/outer/inner": 2.0,
        "custom/outer/flag": 1.0,
        "custom/plain": 3.0,
    }


def test_extract_scalars_rejects_non_scalar_tensor() -> None:
    with pytest.raises(ValueError, match="'vector' must be scalar"):
        extract_scalars({"vector": torch.tensor([1.0, 2.0])})


def test_collect_scalars_includes_metadata_custom_scalars_and_lrs() -> None:
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.SGD([parameter], lr=0.125)
    ctx = _ctx(loss=torch.tensor(2.5), optimizers=[optimizer])
    state = ReportingState()
    state.mark_event(ctx, _ReportStage.AFTER_OPTIMIZER_STEP)

    snapshot = collect_scalars(
        ctx,
        _ReportStage.AFTER_OPTIMIZER_STEP,
        state,
        custom_scalars={
            "metric": lambda context, stage: torch.tensor(4.5),  # noqa: ARG005
            "nested": lambda context, stage: {"value": 6.0},  # noqa: ARG005
        },
    )

    assert snapshot.stage == "AFTER_OPTIMIZER_STEP"
    assert snapshot.event_count == 1
    assert snapshot.step_count == 17
    assert snapshot.batch_count == 19
    assert snapshot.epoch_step_count == 3
    assert snapshot.epoch == 5
    assert snapshot.global_rank == 2
    assert snapshot.elapsed_s is not None
    assert snapshot.scalars == pytest.approx(
        {
            "loss/total": 2.5,
            "optimizer/lr": 0.125,
            "metric": 4.5,
            "nested/value": 6.0,
        }
    )


def test_jsonl_reporter_writes_scalar_snapshot(tmp_path) -> None:
    output_path = tmp_path / "reports" / "metrics.jsonl"
    ctx = _ctx(global_rank=0, loss=torch.tensor(2.5))
    state = ReportingState()
    state.mark_event(ctx, _ReportStage.AFTER_OPTIMIZER_STEP)
    reporter = JSONLReporter(
        output_path,
        custom_scalars={"metric": lambda context, stage: 9.0},  # noqa: ARG005
        mode="w",
    )

    reporter.report(ctx, _ReportStage.AFTER_OPTIMIZER_STEP, state)
    reporter.close()
    reporter.close()

    records = [
        json.loads(line)
        for line in output_path.read_text(encoding="utf-8").splitlines()
    ]
    assert len(records) == 1
    record = records[0]
    assert record["stage"] == "AFTER_OPTIMIZER_STEP"
    assert record["event_count"] == 1
    assert record["step_count"] == 17
    assert record["global_rank"] == 0
    assert record["scalars"] == pytest.approx(
        {
            "loss/total": 2.5,
            "metric": 9.0,
        }
    )


def test_jsonl_reporter_context_manager_closes_file(tmp_path) -> None:
    output_path = tmp_path / "metrics.jsonl"
    ctx = _ctx(global_rank=0)
    state = ReportingState()
    state.mark_event(ctx, _ReportStage.AFTER_OPTIMIZER_STEP)

    with JSONLReporter(
        output_path,
        include_losses=False,
        include_optimizer_lrs=False,
        mode="w",
    ) as reporter:
        reporter.report(ctx, _ReportStage.AFTER_OPTIMIZER_STEP, state)

    records = [
        json.loads(line)
        for line in output_path.read_text(encoding="utf-8").splitlines()
    ]
    assert records[0]["scalars"] == {}


def test_jsonl_reporter_defaults_to_rank_zero_only(tmp_path) -> None:
    output_path = tmp_path / "metrics.jsonl"
    ctx = _ctx(global_rank=1, loss=torch.tensor(2.5))
    state = ReportingState()
    state.mark_event(ctx, _ReportStage.AFTER_OPTIMIZER_STEP)
    reporter = JSONLReporter(output_path, mode="w")

    reporter.report(ctx, _ReportStage.AFTER_OPTIMIZER_STEP, state)

    assert reporter.rank_zero_only is True
    assert not output_path.exists()


def test_jsonl_reporter_requires_rank_token_for_all_rank_writes(tmp_path) -> None:
    with pytest.raises(ValueError, match="must contain '\\{rank\\}'"):
        JSONLReporter(tmp_path / "metrics.jsonl", rank_zero_only=False)


def test_jsonl_reporter_expands_rank_token_for_all_rank_writes(tmp_path) -> None:
    output_template = tmp_path / "metrics.rank-{rank}.jsonl"
    ctx = _ctx(global_rank=3, loss=torch.tensor(2.5))
    state = ReportingState()
    state.mark_event(ctx, _ReportStage.AFTER_OPTIMIZER_STEP)
    reporter = JSONLReporter(
        output_template,
        mode=JSONLMode.WRITE,
        rank_zero_only=False,
    )

    reporter.report(ctx, _ReportStage.AFTER_OPTIMIZER_STEP, state)
    reporter.close()

    output_path = tmp_path / "metrics.rank-3.jsonl"
    records = [
        json.loads(line)
        for line in output_path.read_text(encoding="utf-8").splitlines()
    ]
    assert records[0]["global_rank"] == 3
    assert records[0]["scalars"] == pytest.approx({"loss/total": 2.5})


def test_jsonl_reporter_reduction_uses_all_rank_dispatch_and_rank_zero_write(
    tmp_path,
) -> None:
    output_path = tmp_path / "metrics.jsonl"
    ctx = _ctx(global_rank=0, loss=torch.tensor(2.5))
    state = ReportingState()
    state.mark_event(ctx, _ReportStage.AFTER_OPTIMIZER_STEP)
    reporter = JSONLReporter(
        output_path,
        mode="w",
        rank_reduction=RankReduction.MEAN,
    )

    reporter.report(ctx, _ReportStage.AFTER_OPTIMIZER_STEP, state)
    reporter.close()

    records = [
        json.loads(line)
        for line in output_path.read_text(encoding="utf-8").splitlines()
    ]
    assert reporter.rank_zero_only is False
    assert records[0]["scalars"] == pytest.approx({"loss/total": 2.5})


def test_jsonl_reporter_validates_mode(tmp_path) -> None:
    with pytest.raises(ValueError, match="mode must be one of"):
        JSONLReporter(tmp_path / "metrics.jsonl", mode="r")
