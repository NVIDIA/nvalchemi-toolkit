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
"""Unit tests for ProfilerHook."""

from __future__ import annotations

import csv
from unittest.mock import MagicMock, patch

import pytest
import torch

from nvalchemi.data import AtomicData, Batch
from nvalchemi.dynamics.base import DynamicsStage
from nvalchemi.dynamics.demo import DemoDynamics
from nvalchemi.dynamics.hooks.profiling import ProfilerHook
from nvalchemi.models.demo import DemoModel, DemoModelWrapper


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


def _make_dynamics(hooks=None, n_steps: int = 5, device: str = "cpu") -> DemoDynamics:
    model = DemoModelWrapper(DemoModel())
    if device != "cpu":
        model = model.to(device)
    return DemoDynamics(
        model=model, n_steps=n_steps, dt=1.0, hooks=hooks, device_type=device
    )


# ------------------------------------------------------------------
# Construction / presets
# ------------------------------------------------------------------


class TestConstruction:
    def test_step_preset(self) -> None:
        profiler = ProfilerHook("step")
        assert set(profiler._profiled_stages) == {
            DynamicsStage.BEFORE_STEP,
            DynamicsStage.AFTER_STEP,
        }

    def test_detailed_preset(self) -> None:
        profiler = ProfilerHook("detailed")
        expected = {
            DynamicsStage.BEFORE_STEP,
            DynamicsStage.BEFORE_PRE_UPDATE,
            DynamicsStage.AFTER_PRE_UPDATE,
            DynamicsStage.BEFORE_COMPUTE,
            DynamicsStage.AFTER_COMPUTE,
            DynamicsStage.BEFORE_POST_UPDATE,
            DynamicsStage.AFTER_POST_UPDATE,
            DynamicsStage.AFTER_STEP,
        }
        assert set(profiler._profiled_stages) == expected

    def test_all_preset(self) -> None:
        profiler = ProfilerHook("all")
        assert DynamicsStage.ON_CONVERGE not in profiler._profiled_stages
        assert len(profiler._profiled_stages) == len(DynamicsStage) - 1

    def test_custom_stages(self) -> None:
        S = DynamicsStage
        custom = {S.BEFORE_COMPUTE, S.AFTER_COMPUTE}
        profiler = ProfilerHook(custom)
        assert set(profiler._profiled_stages) == custom

    def test_unknown_preset_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown stages preset"):
            ProfilerHook("bogus")  # type: ignore[arg-type]

    def test_single_stage_raises(self) -> None:
        with pytest.raises(ValueError, match="At least two stages"):
            ProfilerHook({DynamicsStage.BEFORE_STEP})

    def test_stages_sorted_by_execution_order(self) -> None:
        profiler = ProfilerHook("detailed")
        values = [s.value for s in profiler._profiled_stages]
        assert values == sorted(values)


# ------------------------------------------------------------------
# Registration
# ------------------------------------------------------------------


class TestRegistration:
    def test_registers_at_all_stages(self, device: str) -> None:
        profiler = ProfilerHook("step")
        dynamics = _make_dynamics(hooks=[profiler], n_steps=1, device=device)
        assert profiler in dynamics.hooks
        # Verify _runs_on_stage covers the expected stages
        assert profiler._runs_on_stage(DynamicsStage.BEFORE_STEP)
        assert profiler._runs_on_stage(DynamicsStage.AFTER_STEP)

    def test_does_not_register_at_other_stages(self, device: str) -> None:
        profiler = ProfilerHook("step")
        _make_dynamics(hooks=[profiler], n_steps=1, device=device)
        assert not profiler._runs_on_stage(DynamicsStage.BEFORE_COMPUTE)

    def test_composable_with_other_hooks(self, device: str) -> None:
        from nvalchemi.dynamics.hooks.safety import NaNDetectorHook

        profiler = ProfilerHook("step")
        nan_hook = NaNDetectorHook()
        dynamics = _make_dynamics(hooks=[profiler, nan_hook], n_steps=3, device=device)
        batch = _make_batch(device=device)
        dynamics.run(batch)
        assert len(profiler.summary()) > 0


# ------------------------------------------------------------------
# CPU timing
# ------------------------------------------------------------------


class TestTiming:
    def test_records_values(self, device: str) -> None:
        profiler = ProfilerHook("step")
        dynamics = _make_dynamics(hooks=[profiler], n_steps=5, device=device)
        batch = _make_batch(device=device)
        dynamics.run(batch)
        summary = profiler.summary()
        key = "BEFORE_STEP->AFTER_STEP"
        assert key in summary
        assert summary[key]["n_samples"] == 5

    def test_summary_keys(self, device: str) -> None:
        profiler = ProfilerHook("step")
        dynamics = _make_dynamics(hooks=[profiler], n_steps=3, device=device)
        batch = _make_batch(device=device)
        dynamics.run(batch)
        summary = profiler.summary()
        key = next(iter(summary))
        expected_keys = {"mean_s", "std_s", "min_s", "max_s", "total_s", "n_samples"}
        assert set(summary[key].keys()) == expected_keys

    def test_positive_deltas(self, device: str) -> None:
        profiler = ProfilerHook("step")
        dynamics = _make_dynamics(hooks=[profiler], n_steps=5, device=device)
        batch = _make_batch(device=device)
        dynamics.run(batch)
        for stats in profiler.summary().values():
            assert stats["mean_s"] >= 0
            assert stats["min_s"] >= 0

    def test_frequency_gating(self, device: str) -> None:
        profiler = ProfilerHook("step", frequency=3)
        dynamics = _make_dynamics(hooks=[profiler], n_steps=9, device=device)
        batch = _make_batch(device=device)
        dynamics.run(batch)
        summary = profiler.summary()
        assert summary["BEFORE_STEP->AFTER_STEP"]["n_samples"] == 3

    def test_detailed_timing(self, device: str) -> None:
        profiler = ProfilerHook("detailed")
        dynamics = _make_dynamics(hooks=[profiler], n_steps=5, device=device)
        batch = _make_batch(device=device)
        dynamics.run(batch)
        summary = profiler.summary()
        # 8 stages -> 7 transitions.
        assert len(summary) == 7
        for stats in summary.values():
            assert stats["n_samples"] == 5

    def test_reset(self, device: str) -> None:
        profiler = ProfilerHook("step")
        dynamics = _make_dynamics(hooks=[profiler], n_steps=5, device=device)
        batch = _make_batch(device=device)
        dynamics.run(batch)
        assert len(profiler.summary()) > 0
        profiler.reset()
        assert profiler.summary() == {}


# ------------------------------------------------------------------
# Auto backend
# ------------------------------------------------------------------


class TestAutoBackend:
    def test_auto_selects_perf_counter_on_cpu(self) -> None:
        profiler = ProfilerHook("step", timer_backend="auto")
        dynamics = _make_dynamics(hooks=[profiler], n_steps=1, device="cpu")
        batch = _make_batch(device="cpu")
        dynamics.run(batch)
        assert profiler._backend_resolved == "perf_counter"

    def test_auto_selects_cuda_event_on_gpu(self, gpu_device: str) -> None:
        profiler = ProfilerHook("step", timer_backend="auto")
        dynamics = _make_dynamics(hooks=[profiler], n_steps=1, device=gpu_device)
        batch = _make_batch(device=gpu_device)
        dynamics.run(batch)
        assert profiler._backend_resolved == "cuda_event"


# ------------------------------------------------------------------
# NVTX
# ------------------------------------------------------------------


class TestNVTX:
    def test_nvtx_push_pop_called(self) -> None:
        try:
            import nvtx  # noqa: F401
        except ImportError:
            pytest.skip("nvtx not available")

        with patch("nvalchemi.dynamics.hooks.profiling.nvtx") as mock_nvtx:
            mock_nvtx.push_range = MagicMock()
            mock_nvtx.pop_range = MagicMock()

            profiler = ProfilerHook("step", enable_nvtx=True)
            dynamics = _make_dynamics(hooks=[profiler], n_steps=1)
            batch = _make_batch()
            dynamics.run(batch)

            assert mock_nvtx.push_range.call_count >= 1
            assert mock_nvtx.pop_range.call_count >= 1

    def test_nvtx_disabled(self) -> None:
        with patch("nvalchemi.dynamics.hooks.profiling.nvtx") as mock_nvtx:
            mock_nvtx.push_range = MagicMock()
            mock_nvtx.pop_range = MagicMock()

            profiler = ProfilerHook("step", enable_nvtx=False)
            dynamics = _make_dynamics(hooks=[profiler], n_steps=1)
            batch = _make_batch()
            dynamics.run(batch)

            mock_nvtx.push_range.assert_not_called()
            mock_nvtx.pop_range.assert_not_called()


# ------------------------------------------------------------------
# CSV logging
# ------------------------------------------------------------------


class TestCSVLogging:
    def test_writes_csv(self, tmp_path, device: str) -> None:
        log_file = tmp_path / "profiler.csv"
        profiler = ProfilerHook("step", log_path=log_file)
        dynamics = _make_dynamics(hooks=[profiler], n_steps=3, device=device)
        batch = _make_batch(device=device)
        dynamics.run(batch)
        profiler.close()

        with open(log_file) as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 3
        assert "rank" in rows[0]
        assert "step" in rows[0]
        assert "stage" in rows[0]
        assert "t_since_init_s" in rows[0]
        assert "delta_s" in rows[0]

    def test_detailed_csv_rows(self, tmp_path, device: str) -> None:
        log_file = tmp_path / "detailed.csv"
        profiler = ProfilerHook("detailed", log_path=log_file)
        dynamics = _make_dynamics(hooks=[profiler], n_steps=2, device=device)
        batch = _make_batch(device=device)
        dynamics.run(batch)
        profiler.close()

        with open(log_file) as f:
            rows = list(csv.DictReader(f))
        # 8 stages -> 7 transitions per step, 2 steps -> 14 rows.
        assert len(rows) == 14


# ------------------------------------------------------------------
# Console output
# ------------------------------------------------------------------


class TestConsoleOutput:
    def test_show_console(self, device: str) -> None:
        with patch("nvalchemi.dynamics.hooks.profiling.logger") as mock_logger:
            profiler = ProfilerHook("step", show_console=True)
            dynamics = _make_dynamics(hooks=[profiler], n_steps=2, device=device)
            batch = _make_batch(device=device)
            dynamics.run(batch)
            assert mock_logger.info.call_count == 2

    def test_console_frequency(self, device: str) -> None:
        with patch("nvalchemi.dynamics.hooks.profiling.logger") as mock_logger:
            profiler = ProfilerHook(
                "step",
                show_console=True,
                console_frequency=3,
            )
            dynamics = _make_dynamics(hooks=[profiler], n_steps=9, device=device)
            batch = _make_batch(device=device)
            dynamics.run(batch)
            assert mock_logger.info.call_count == 3


# ------------------------------------------------------------------
# Integration
# ------------------------------------------------------------------


class TestIntegration:
    def test_full_loop(self, device: str) -> None:
        profiler = ProfilerHook("step")
        dynamics = _make_dynamics(hooks=[profiler], n_steps=5, device=device)
        batch = _make_batch(device=device)
        dynamics.run(batch)
        summary = profiler.summary()
        assert len(summary) > 0
        for stats in summary.values():
            assert "mean_s" in stats
            assert stats["n_samples"] == 5
