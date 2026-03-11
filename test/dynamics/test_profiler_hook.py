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
"""Unit tests for ProfilerHook — interval-based profiling controller."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import torch

from nvalchemi.data import AtomicData, Batch
from nvalchemi.dynamics.base import Hook, HookStageEnum
from nvalchemi.dynamics.demo import DemoDynamics
from nvalchemi.dynamics.hooks.profiling import ProfilerHook, _Probe
from nvalchemi.models.demo import DemoModelWrapper


def _make_batch(
    n_graphs: int = 2, atoms_per_graph: int = 3, device: str = "cpu"
) -> Batch:
    data_list = [
        AtomicData(
            atomic_numbers=torch.tensor([6] * atoms_per_graph, dtype=torch.long),
            positions=torch.randn(atoms_per_graph, 3),
        )
        for _ in range(n_graphs)
    ]
    batch = Batch.from_data_list(data_list).to(device)
    batch.__dict__["forces"] = torch.randn(batch.num_nodes, 3, device=device)
    batch.__dict__["energies"] = torch.randn(batch.num_graphs, 1, device=device)
    batch.__dict__["velocities"] = torch.randn(batch.num_nodes, 3, device=device) * 0.01
    batch.__dict__["atomic_masses"] = torch.full(
        (batch.num_nodes,), 12.0, device=device
    )
    return batch


def _make_dynamics(hooks=None, n_steps: int = 5) -> DemoDynamics:
    model = DemoModelWrapper()
    return DemoDynamics(model=model, n_steps=n_steps, dt=1.0, hooks=hooks)


class TestProbe:
    def test_has_correct_stage(self) -> None:
        profiler = ProfilerHook.step_timer()
        probes = profiler.hooks
        stages = {p.stage for p in probes}
        assert HookStageEnum.BEFORE_STEP in stages
        assert HookStageEnum.AFTER_STEP in stages

    def test_has_correct_frequency(self) -> None:
        profiler = ProfilerHook.step_timer(frequency=7)
        for probe in profiler.hooks:
            assert probe.frequency == 7

    def test_satisfies_hook_protocol(self) -> None:
        profiler = ProfilerHook.step_timer()
        for probe in profiler.hooks:
            assert isinstance(probe, Hook)


class TestProfilerHookConstruction:
    def test_single_interval(self) -> None:
        S = HookStageEnum
        profiler = ProfilerHook(intervals=[("test", S.BEFORE_STEP, S.AFTER_STEP)])
        assert len(profiler.hooks) == 2

    def test_shared_stage_deduplication(self) -> None:
        S = HookStageEnum
        profiler = ProfilerHook(
            intervals=[
                ("a", S.BEFORE_STEP, S.AFTER_COMPUTE),
                ("b", S.AFTER_COMPUTE, S.AFTER_STEP),
            ]
        )
        assert len(profiler.hooks) == 3
        stages = {p.stage for p in profiler.hooks}
        assert stages == {S.BEFORE_STEP, S.AFTER_COMPUTE, S.AFTER_STEP}

    def test_duplicate_labels_raises(self) -> None:
        S = HookStageEnum
        with pytest.raises(ValueError, match="Duplicate"):
            ProfilerHook(
                intervals=[
                    ("same", S.BEFORE_STEP, S.AFTER_STEP),
                    ("same", S.BEFORE_COMPUTE, S.AFTER_COMPUTE),
                ]
            )

    def test_start_not_before_end_raises(self) -> None:
        S = HookStageEnum
        with pytest.raises(ValueError, match="must come before"):
            ProfilerHook(intervals=[("bad", S.AFTER_STEP, S.BEFORE_STEP)])

    def test_start_equals_end_raises(self) -> None:
        S = HookStageEnum
        with pytest.raises(ValueError, match="must come before"):
            ProfilerHook(intervals=[("bad", S.AFTER_STEP, S.AFTER_STEP)])

    def test_step_timer_classmethod(self) -> None:
        profiler = ProfilerHook.step_timer()
        assert len(profiler.hooks) == 2
        stages = {p.stage for p in profiler.hooks}
        assert stages == {HookStageEnum.BEFORE_STEP, HookStageEnum.AFTER_STEP}

    def test_detailed_classmethod(self) -> None:
        profiler = ProfilerHook.detailed()
        assert len(profiler.hooks) == 8
        expected = {
            HookStageEnum.BEFORE_STEP,
            HookStageEnum.AFTER_STEP,
            HookStageEnum.BEFORE_PRE_UPDATE,
            HookStageEnum.AFTER_PRE_UPDATE,
            HookStageEnum.BEFORE_COMPUTE,
            HookStageEnum.AFTER_COMPUTE,
            HookStageEnum.BEFORE_POST_UPDATE,
            HookStageEnum.AFTER_POST_UPDATE,
        }
        stages = {p.stage for p in profiler.hooks}
        assert stages == expected


class TestProfilerHookCPUTiming:
    def test_records_values(self) -> None:
        profiler = ProfilerHook.step_timer(timer_backend="perf_counter")
        dynamics = _make_dynamics(hooks=profiler.hooks, n_steps=5)
        batch = _make_batch()
        dynamics.run(batch)
        summary = profiler.summary()
        assert summary["step"]["n_samples"] == 5

    def test_summary_keys(self) -> None:
        profiler = ProfilerHook.step_timer(timer_backend="perf_counter")
        dynamics = _make_dynamics(hooks=profiler.hooks, n_steps=3)
        batch = _make_batch()
        dynamics.run(batch)
        summary = profiler.summary()
        expected_keys = {"mean_s", "std_s", "min_s", "max_s", "total_s", "n_samples"}
        assert set(summary["step"].keys()) == expected_keys

    def test_positive_deltas(self) -> None:
        profiler = ProfilerHook.step_timer(timer_backend="perf_counter")
        dynamics = _make_dynamics(hooks=profiler.hooks, n_steps=5)
        batch = _make_batch()
        dynamics.run(batch)
        summary = profiler.summary()
        assert summary["step"]["mean_s"] >= 0
        assert summary["step"]["min_s"] >= 0
        assert summary["step"]["max_s"] >= 0
        assert summary["step"]["total_s"] >= 0

    def test_frequency_gating(self) -> None:
        profiler = ProfilerHook.step_timer(timer_backend="perf_counter", frequency=3)
        dynamics = _make_dynamics(hooks=profiler.hooks, n_steps=9)
        batch = _make_batch()
        dynamics.run(batch)
        summary = profiler.summary()
        assert summary["step"]["n_samples"] == 3

    def test_multiple_intervals(self) -> None:
        profiler = ProfilerHook.detailed(timer_backend="perf_counter")
        dynamics = _make_dynamics(hooks=profiler.hooks, n_steps=5)
        batch = _make_batch()
        dynamics.run(batch)
        summary = profiler.summary()
        assert set(summary.keys()) == {"step", "pre_update", "compute", "post_update"}
        for label in summary:
            assert summary[label]["n_samples"] == 5

    def test_timer_disabled(self) -> None:
        profiler = ProfilerHook.step_timer(enable_timer=False)
        dynamics = _make_dynamics(hooks=profiler.hooks, n_steps=5)
        batch = _make_batch()
        dynamics.run(batch)
        summary = profiler.summary()
        assert summary == {}

    def test_reset(self) -> None:
        profiler = ProfilerHook.step_timer(timer_backend="perf_counter")
        dynamics = _make_dynamics(hooks=profiler.hooks, n_steps=5)
        batch = _make_batch()
        dynamics.run(batch)
        assert profiler.summary()["step"]["n_samples"] == 5
        profiler.reset()
        summary = profiler.summary()
        assert "step" not in summary


class TestProfilerHookNVTX:
    def test_nvtx_push_pop_called(self) -> None:
        try:
            import nvtx  # noqa: F401
        except ImportError:
            pytest.skip("nvtx not available")

        with (
            patch("nvalchemi.dynamics.hooks.profiling.nvtx") as mock_nvtx,
        ):
            mock_nvtx.push_range = MagicMock()
            mock_nvtx.pop_range = MagicMock()

            profiler = ProfilerHook.step_timer(enable_nvtx=True, enable_timer=False)
            dynamics = _make_dynamics(hooks=profiler.hooks, n_steps=1)
            batch = _make_batch()
            dynamics.run(batch)

            mock_nvtx.push_range.assert_called_with("step/0")
            mock_nvtx.pop_range.assert_called_once()

    def test_nvtx_disabled(self) -> None:
        with patch("nvalchemi.dynamics.hooks.profiling.nvtx") as mock_nvtx:
            mock_nvtx.push_range = MagicMock()
            mock_nvtx.pop_range = MagicMock()

            profiler = ProfilerHook.step_timer(enable_nvtx=False, enable_timer=False)
            dynamics = _make_dynamics(hooks=profiler.hooks, n_steps=1)
            batch = _make_batch()
            dynamics.run(batch)

            mock_nvtx.push_range.assert_not_called()
            mock_nvtx.pop_range.assert_not_called()

    def test_nvtx_nested_ranges(self) -> None:
        try:
            import nvtx  # noqa: F401
        except ImportError:
            pytest.skip("nvtx not available")

        with patch("nvalchemi.dynamics.hooks.profiling.nvtx") as mock_nvtx:
            push_calls = []
            pop_calls = []
            mock_nvtx.push_range = MagicMock(side_effect=lambda x: push_calls.append(x))
            mock_nvtx.pop_range = MagicMock(side_effect=lambda: pop_calls.append("pop"))

            S = HookStageEnum
            profiler = ProfilerHook(
                intervals=[
                    ("outer", S.BEFORE_STEP, S.AFTER_STEP),
                    ("inner", S.BEFORE_COMPUTE, S.AFTER_COMPUTE),
                ],
                enable_nvtx=True,
                enable_timer=False,
            )
            dynamics = _make_dynamics(hooks=profiler.hooks, n_steps=1)
            batch = _make_batch()
            dynamics.run(batch)

            assert "outer/0" in push_calls
            assert "inner/0" in push_calls
            outer_push_idx = push_calls.index("outer/0")
            inner_push_idx = push_calls.index("inner/0")
            assert outer_push_idx < inner_push_idx
            assert len(pop_calls) == 2


class TestProfilerHookAutoBackend:
    def test_auto_selects_perf_counter_on_cpu(self) -> None:
        profiler = ProfilerHook.step_timer(timer_backend="auto")
        dynamics = _make_dynamics(hooks=profiler.hooks, n_steps=1)
        batch = _make_batch(device="cpu")
        dynamics.run(batch)
        assert profiler._backend_resolved == "perf_counter"


class TestProfilerHookIntegration:
    def test_full_loop_with_dynamics(self) -> None:
        profiler = ProfilerHook.step_timer(timer_backend="perf_counter")
        dynamics = _make_dynamics(hooks=profiler.hooks, n_steps=5)
        batch = _make_batch()
        dynamics.run(batch)
        summary = profiler.summary()
        assert "step" in summary
        assert summary["step"]["n_samples"] == 5
        assert "mean_s" in summary["step"]

    def test_probes_register_at_correct_stages(self) -> None:
        profiler = ProfilerHook.step_timer()
        dynamics = _make_dynamics(hooks=profiler.hooks, n_steps=1)

        before_step_hooks = dynamics.hooks[HookStageEnum.BEFORE_STEP]
        after_step_hooks = dynamics.hooks[HookStageEnum.AFTER_STEP]

        before_step_probes = [h for h in before_step_hooks if isinstance(h, _Probe)]
        after_step_probes = [h for h in after_step_hooks if isinstance(h, _Probe)]

        assert len(before_step_probes) == 1
        assert len(after_step_probes) == 1
