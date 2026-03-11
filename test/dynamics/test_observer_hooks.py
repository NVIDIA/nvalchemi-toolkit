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
"""Unit tests for observer hooks — SnapshotHook, ConvergedSnapshotHook,
LoggingHook, and EnergyDriftMonitorHook.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest
import torch

from nvalchemi.data import AtomicData, Batch
from nvalchemi.dynamics.base import BaseDynamics, HookStageEnum
from nvalchemi.dynamics.hooks import (
    ConvergedSnapshotHook,
    EnergyDriftMonitorHook,
    LoggingHook,
    SnapshotHook,
)
from nvalchemi.dynamics.hooks._base import _ObserverHook
from nvalchemi.dynamics.sinks import HostMemory
from nvalchemi.models.demo import DemoModelWrapper

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_batch(
    n_graphs: int = 2,
    atoms_per_graph: int = 3,
    with_velocities: bool = False,
    device: str = "cpu",
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
    if with_velocities:
        batch.__dict__["velocities"] = (
            torch.randn(batch.num_nodes, 3, device=device) * 0.01
        )
        batch.__dict__["atomic_masses"] = torch.full(
            (batch.num_nodes,), 12.0, device=device
        )
    return batch


def _make_dynamics(device: str = "cpu") -> BaseDynamics:
    model = DemoModelWrapper()
    if device != "cpu":
        model = model.to(device)
    return BaseDynamics(model, device_type=device)


# ---------------------------------------------------------------------------
# _ObserverHook base class
# ---------------------------------------------------------------------------


class TestObserverHookBase:
    def test_default_stage(self) -> None:
        hook = _ObserverHook()
        assert hook.stage == HookStageEnum.AFTER_STEP

    def test_on_converge_stage(self) -> None:
        hook = _ObserverHook(stage=HookStageEnum.ON_CONVERGE)
        assert hook.stage == HookStageEnum.ON_CONVERGE

    def test_invalid_stage_raises(self) -> None:
        with pytest.raises(ValueError, match="Observer hooks only support"):
            _ObserverHook(stage=HookStageEnum.BEFORE_STEP)

    def test_frequency(self) -> None:
        hook = _ObserverHook(frequency=5)
        assert hook.frequency == 5


# ---------------------------------------------------------------------------
# SnapshotHook
# ---------------------------------------------------------------------------


class TestSnapshotHook:
    def test_writes_to_sink(self, device: str) -> None:
        sink = HostMemory(capacity=100)
        hook = SnapshotHook(sink=sink, frequency=1)
        batch = _make_batch(device=device)
        dynamics = _make_dynamics(device=device)

        hook(batch, dynamics)
        assert len(sink) == batch.num_graphs

    def test_frequency_respected(self, device: str) -> None:
        sink = HostMemory(capacity=100)
        hook = SnapshotHook(sink=sink, frequency=2)
        dynamics = _make_dynamics(device=device)

        # Register and run — frequency gating is done by dynamics._call_hooks
        dynamics.register_hook(hook)
        assert hook.stage == HookStageEnum.AFTER_STEP
        assert hook.frequency == 2

    def test_multiple_writes(self, device: str) -> None:
        sink = HostMemory(capacity=100)
        hook = SnapshotHook(sink=sink)
        batch = _make_batch(n_graphs=3, device=device)
        dynamics = _make_dynamics(device=device)

        hook(batch, dynamics)
        hook(batch, dynamics)
        assert len(sink) == 6

    def test_protocol_compliance(self) -> None:
        from nvalchemi.dynamics.base import Hook

        sink = HostMemory(capacity=10)
        hook = SnapshotHook(sink=sink)
        assert isinstance(hook, Hook)


# ---------------------------------------------------------------------------
# ConvergedSnapshotHook
# ---------------------------------------------------------------------------


class TestConvergedSnapshotHook:
    def test_stage_is_on_converge(self) -> None:
        sink = HostMemory(capacity=100)
        hook = ConvergedSnapshotHook(sink=sink)
        assert hook.stage == HookStageEnum.ON_CONVERGE

    def test_writes_only_converged_samples(self, device: str) -> None:
        sink = HostMemory(capacity=100)
        hook = ConvergedSnapshotHook(sink=sink)
        batch = _make_batch(n_graphs=4, device=device)
        dynamics = _make_dynamics(device=device)

        # Simulate convergence of graphs 1 and 3
        dynamics._last_converged = torch.tensor([1, 3])
        hook(batch, dynamics)

        assert len(sink) == 2

    def test_no_write_when_no_converged(self, device: str) -> None:
        sink = HostMemory(capacity=100)
        hook = ConvergedSnapshotHook(sink=sink)
        batch = _make_batch(device=device)
        dynamics = _make_dynamics(device=device)

        dynamics._last_converged = None
        hook(batch, dynamics)
        assert len(sink) == 0

    def test_no_write_when_empty_converged(self, device: str) -> None:
        sink = HostMemory(capacity=100)
        hook = ConvergedSnapshotHook(sink=sink)
        batch = _make_batch(device=device)
        dynamics = _make_dynamics(device=device)

        dynamics._last_converged = torch.tensor([], dtype=torch.long)
        hook(batch, dynamics)
        assert len(sink) == 0

    def test_all_converged(self, device: str) -> None:
        sink = HostMemory(capacity=100)
        hook = ConvergedSnapshotHook(sink=sink)
        batch = _make_batch(n_graphs=3, device=device)
        dynamics = _make_dynamics(device=device)

        dynamics._last_converged = torch.tensor([0, 1, 2])
        hook(batch, dynamics)
        assert len(sink) == 3


# ---------------------------------------------------------------------------
# LoggingHook
# ---------------------------------------------------------------------------


class TestLoggingHook:
    """Tests for LoggingHook with per-sample row semantics."""

    @staticmethod
    def _capture_hook(
        **kwargs,
    ) -> tuple[LoggingHook, list[tuple[int, list[dict[str, float]]]]]:
        """Create a LoggingHook with a custom backend that captures rows."""
        captured: list[tuple[int, list[dict[str, float]]]] = []

        def writer(step: int, rows: list[dict[str, float]]) -> None:
            captured.append((step, rows))

        return LoggingHook(backend="custom", writer_fn=writer, **kwargs), captured

    def test_context_manager(self, device: str, tmp_path: Path) -> None:
        csv_path = tmp_path / "ctx.csv"
        with LoggingHook(backend="csv", log_path=str(csv_path)) as hook:
            batch = _make_batch(device=device)
            dynamics = _make_dynamics(device=device)
            hook(batch, dynamics)
        # After exiting, file should be flushed and closed
        rows = list(csv.DictReader(csv_path.open()))
        assert len(rows) == 2  # 2 graphs

    def test_context_manager_closes_resources(
        self, device: str, tmp_path: Path
    ) -> None:
        csv_path = tmp_path / "ctx2.csv"
        hook = LoggingHook(backend="csv", log_path=str(csv_path))
        with hook:
            batch = _make_batch(device=device)
            dynamics = _make_dynamics(device=device)
            hook(batch, dynamics)
        assert hook._csv_file is None
        assert hook._stream is None

    def test_usable_without_context_manager(self, device: str) -> None:
        """Executor is created in __init__, so hook works without `with`."""
        hook, captured = self._capture_hook()
        batch = _make_batch(device=device)
        dynamics = _make_dynamics(device=device)

        hook(batch, dynamics)
        hook.close()
        assert len(captured) == 1

    def test_executor_survives_close(self, device: str) -> None:
        """Executor is recreated after close() so hook remains usable."""
        hook, captured = self._capture_hook()
        batch = _make_batch(device=device)
        dynamics = _make_dynamics(device=device)

        with hook:
            hook(batch, dynamics)
        assert len(captured) == 1

        # Still usable after close via a new context
        with hook:
            hook(batch, dynamics)
        assert len(captured) == 2

    def test_one_row_per_graph(self, device: str) -> None:
        hook, captured = self._capture_hook()
        batch = _make_batch(n_graphs=3, device=device)
        dynamics = _make_dynamics(device=device)

        with hook:
            hook(batch, dynamics)

        assert len(captured) == 1
        step, rows = captured[0]
        assert step == 0
        assert len(rows) == 3  # one row per graph

    def test_row_contains_step_graph_idx_status(self, device: str) -> None:
        hook, captured = self._capture_hook()
        batch = _make_batch(n_graphs=2, device=device)
        dynamics = _make_dynamics(device=device)

        with hook:
            hook(batch, dynamics)

        rows = captured[0][1]
        for i, row in enumerate(rows):
            assert row["step"] == 0.0
            assert row["graph_idx"] == float(i)
            assert "status" in row  # 0.0 when no status on batch

    def test_per_graph_energy_and_fmax(self, device: str) -> None:
        hook, captured = self._capture_hook()
        batch = _make_batch(n_graphs=2, device=device)
        dynamics = _make_dynamics(device=device)

        with hook:
            hook(batch, dynamics)

        rows = captured[0][1]
        assert "energy" in rows[0]
        assert "fmax" in rows[0]
        # Per-graph energies should differ (random)
        # Just verify they're valid floats
        for row in rows:
            assert isinstance(row["energy"], float)
            assert isinstance(row["fmax"], float)

    def test_csv_per_sample_rows(self, device: str, tmp_path: Path) -> None:
        csv_path = tmp_path / "log.csv"
        with LoggingHook(backend="csv", log_path=str(csv_path)) as hook:
            batch = _make_batch(n_graphs=2, device=device)
            dynamics = _make_dynamics(device=device)

            hook(batch, dynamics)
            dynamics.step_count = 1
            hook(batch, dynamics)

        rows = list(csv.DictReader(csv_path.open()))
        # 2 graphs * 2 steps = 4 rows
        assert len(rows) == 4
        assert "step" in rows[0]
        assert "graph_idx" in rows[0]
        assert "status" in rows[0]
        assert "energy" in rows[0]
        assert "fmax" in rows[0]
        # First 2 rows are step 0, next 2 are step 1
        assert float(rows[0]["step"]) == 0.0
        assert float(rows[1]["step"]) == 0.0
        assert float(rows[2]["step"]) == 1.0
        assert float(rows[3]["step"]) == 1.0

    def test_custom_backend_receives_rows(self, device: str) -> None:
        hook, captured = self._capture_hook()
        batch = _make_batch(device=device)
        dynamics = _make_dynamics(device=device)

        with hook:
            hook(batch, dynamics)

        assert len(captured) == 1
        assert captured[0][0] == 0
        assert isinstance(captured[0][1], list)
        assert isinstance(captured[0][1][0], dict)

    def test_custom_scalar_float_broadcast(self, device: str) -> None:
        hook, captured = self._capture_hook(
            custom_scalars={"n_atoms": lambda b, d: float(b.num_nodes)},
        )
        batch = _make_batch(n_graphs=2, atoms_per_graph=5, device=device)
        dynamics = _make_dynamics(device=device)

        with hook:
            hook(batch, dynamics)

        rows = captured[0][1]
        # Float is broadcast to all graphs
        assert rows[0]["n_atoms"] == 10.0
        assert rows[1]["n_atoms"] == 10.0

    def test_custom_scalar_tensor_per_graph(self, device: str) -> None:
        hook, captured = self._capture_hook(
            custom_scalars={
                "per_graph_val": lambda b, d: torch.arange(
                    b.num_graphs,
                    dtype=torch.float32,
                    device=b.positions.device,
                ),
            },
        )
        batch = _make_batch(n_graphs=3, device=device)
        dynamics = _make_dynamics(device=device)

        with hook:
            hook(batch, dynamics)

        rows = captured[0][1]
        assert rows[0]["per_graph_val"] == 0.0
        assert rows[1]["per_graph_val"] == 1.0
        assert rows[2]["per_graph_val"] == 2.0

    def test_custom_scalar_overrides_default(self, device: str) -> None:
        hook, captured = self._capture_hook(
            custom_scalars={"energy": lambda b, d: 42.0},
        )
        batch = _make_batch(device=device)
        dynamics = _make_dynamics(device=device)

        with hook:
            hook(batch, dynamics)

        rows = captured[0][1]
        for row in rows:
            assert row["energy"] == 42.0

    def test_temperature_logged_with_velocities(self, device: str) -> None:
        hook, captured = self._capture_hook()
        batch = _make_batch(with_velocities=True, device=device)
        dynamics = _make_dynamics(device=device)

        with hook:
            hook(batch, dynamics)

        assert "temperature" in captured[0][1][0]

    def test_zero_velocities_give_zero_temperature(self, device: str) -> None:
        hook, captured = self._capture_hook()
        batch = _make_batch(with_velocities=False, device=device)
        dynamics = _make_dynamics(device=device)

        with hook:
            hook(batch, dynamics)

        # Batch always has velocities (defaulting to zeros), so temperature
        # is always logged. Zero velocities give T=0.
        assert captured[0][1][0]["temperature"] == 0.0

    def test_csv_requires_log_path(self) -> None:
        with pytest.raises(ValueError, match="csv backend requires log_path"):
            LoggingHook(backend="csv")

    def test_tensorboard_requires_log_path(self) -> None:
        with pytest.raises(ValueError, match="tensorboard backend requires log_path"):
            LoggingHook(backend="tensorboard")

    def test_custom_requires_writer_fn(self) -> None:
        with pytest.raises(ValueError, match="custom backend requires writer_fn"):
            LoggingHook(backend="custom")

    def test_invalid_backend_raises(self) -> None:
        with pytest.raises(ValueError, match="only supports backends"):
            LoggingHook(backend="blagh")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# EnergyDriftMonitorHook
# ---------------------------------------------------------------------------


class TestEnergyDriftMonitorHook:
    def test_first_call_captures_reference(self, device: str) -> None:
        hook = EnergyDriftMonitorHook(threshold=1.0)
        batch = _make_batch(device=device)
        dynamics = _make_dynamics(device=device)

        # First call should just capture reference, not raise
        hook(batch, dynamics)
        assert hook._reference_total_energy is not None

    def test_no_drift_no_action(self, device: str) -> None:
        hook = EnergyDriftMonitorHook(threshold=1.0, metric="absolute")
        batch = _make_batch(device=device)
        # Set constant energies
        batch.__dict__["energies"] = torch.tensor([[1.0], [2.0]], device=device)
        dynamics = _make_dynamics(device=device)

        hook(batch, dynamics)  # capture reference
        dynamics.step_count = 1
        hook(batch, dynamics)  # same energy, no drift

    def test_drift_exceeds_threshold_warn(
        self, device: str, capfd: pytest.CaptureFixture
    ) -> None:
        hook = EnergyDriftMonitorHook(threshold=0.01, metric="absolute", action="warn")
        batch = _make_batch(device=device)
        batch.__dict__["energies"] = torch.tensor([[1.0], [2.0]], device=device)
        dynamics = _make_dynamics(device=device)

        hook(batch, dynamics)  # capture reference

        # Introduce drift
        batch.__dict__["energies"] = torch.tensor([[2.0], [3.0]], device=device)
        dynamics.step_count = 1
        hook(batch, dynamics)  # should warn, not raise

    def test_drift_exceeds_threshold_raise(self, device: str) -> None:
        hook = EnergyDriftMonitorHook(threshold=0.01, metric="absolute", action="raise")
        batch = _make_batch(device=device)
        batch.__dict__["energies"] = torch.tensor([[1.0], [2.0]], device=device)
        dynamics = _make_dynamics(device=device)

        hook(batch, dynamics)  # capture reference

        batch.__dict__["energies"] = torch.tensor([[2.0], [3.0]], device=device)
        dynamics.step_count = 1
        with pytest.raises(RuntimeError, match="Energy drift"):
            hook(batch, dynamics)

    def test_per_atom_per_step_normalization(self, device: str) -> None:
        hook = EnergyDriftMonitorHook(
            threshold=1e10, metric="per_atom_per_step", action="raise"
        )
        batch = _make_batch(n_graphs=1, atoms_per_graph=10, device=device)
        batch.__dict__["energies"] = torch.tensor([[0.0]], device=device)
        dynamics = _make_dynamics(device=device)

        hook(batch, dynamics)  # capture reference

        batch.__dict__["energies"] = torch.tensor([[1.0]], device=device)
        dynamics.step_count = 10
        # drift = |1.0 - 0.0| / (10 atoms * 10 steps) = 0.01
        # This is well below 1e10, so no raise
        hook(batch, dynamics)

    def test_per_atom_per_step_exceeds(self, device: str) -> None:
        hook = EnergyDriftMonitorHook(
            threshold=0.005, metric="per_atom_per_step", action="raise"
        )
        batch = _make_batch(n_graphs=1, atoms_per_graph=10, device=device)
        batch.__dict__["energies"] = torch.tensor([[0.0]], device=device)
        dynamics = _make_dynamics(device=device)

        hook(batch, dynamics)

        batch.__dict__["energies"] = torch.tensor([[1.0]], device=device)
        dynamics.step_count = 10
        # drift = |1.0| / (10 * 10) = 0.01 > 0.005
        with pytest.raises(RuntimeError, match="Energy drift"):
            hook(batch, dynamics)

    def test_include_kinetic_false(self, device: str) -> None:
        hook = EnergyDriftMonitorHook(
            threshold=1e10, metric="absolute", include_kinetic=False
        )
        batch = _make_batch(with_velocities=True, device=device)
        batch.__dict__["energies"] = torch.tensor([[1.0], [2.0]], device=device)
        dynamics = _make_dynamics(device=device)

        # Should work without KE even though velocities are present
        hook(batch, dynamics)
        dynamics.step_count = 1
        hook(batch, dynamics)

    def test_include_kinetic_true(self, device: str) -> None:
        hook = EnergyDriftMonitorHook(
            threshold=1e10, metric="absolute", include_kinetic=True
        )
        batch = _make_batch(with_velocities=True, device=device)
        batch.__dict__["energies"] = torch.tensor([[1.0], [2.0]], device=device)
        dynamics = _make_dynamics(device=device)

        hook(batch, dynamics)
        dynamics.step_count = 1
        hook(batch, dynamics)

    def test_multi_graph_max_drift(self, device: str) -> None:
        hook = EnergyDriftMonitorHook(threshold=0.5, metric="absolute", action="raise")
        batch = _make_batch(n_graphs=2, device=device)
        batch.__dict__["energies"] = torch.tensor([[0.0], [0.0]], device=device)
        dynamics = _make_dynamics(device=device)

        hook(batch, dynamics)

        # Only one graph drifts above threshold
        batch.__dict__["energies"] = torch.tensor([[0.1], [1.0]], device=device)
        dynamics.step_count = 1
        with pytest.raises(RuntimeError, match="Energy drift"):
            hook(batch, dynamics)


# ---------------------------------------------------------------------------
# Hook lifecycle management (_open_hooks / _close_hooks)
# ---------------------------------------------------------------------------


class _MockCMHook:
    """Mock hook with context-manager protocol for testing lifecycle."""

    def __init__(self) -> None:
        self.frequency = 1
        self.stage = HookStageEnum.AFTER_STEP
        self.enter_count = 0
        self.exit_count = 0
        self.exit_args: list[tuple] = []
        self.call_count = 0
        self.close_count = 0

    def __call__(self, batch: Batch, dynamics: BaseDynamics) -> None:
        self.call_count += 1

    def __enter__(self) -> "_MockCMHook":
        self.enter_count += 1
        return self

    def __exit__(self, *args: object) -> None:
        self.exit_count += 1
        self.exit_args.append(args)

    def close(self) -> None:
        self.close_count += 1


class _MockCloseOnlyHook:
    """Mock hook with close() only (no context-manager protocol)."""

    def __init__(self) -> None:
        self.frequency = 1
        self.stage = HookStageEnum.AFTER_STEP
        self.call_count = 0
        self.close_count = 0

    def __call__(self, batch: Batch, dynamics: BaseDynamics) -> None:
        self.call_count += 1

    def close(self) -> None:
        self.close_count += 1


class TestHookLifecycle:
    """Tests for automatic hook context-manager lifecycle in run()."""

    @pytest.fixture(params=["cpu"])
    def device(self, request: pytest.FixtureRequest) -> str:
        return request.param

    def test_run_calls_enter_and_exit_on_cm_hooks(self, device: str) -> None:
        """Context-manager hooks get __enter__ at start and __exit__ at end."""
        hook = _MockCMHook()

        dynamics = _make_dynamics(device=device)
        dynamics.register_hook(hook)

        batch = _make_batch(device=device)
        dynamics.run(batch, n_steps=3)

        assert hook.enter_count == 1
        assert hook.exit_count == 1
        assert hook.exit_args == [(None, None, None)]

    def test_run_calls_close_on_non_cm_hooks(self, device: str) -> None:
        """Hooks with close() but no __enter__/__exit__ get close() called."""
        hook = _MockCloseOnlyHook()

        dynamics = _make_dynamics(device=device)
        dynamics.register_hook(hook)

        batch = _make_batch(device=device)
        dynamics.run(batch, n_steps=3)

        assert hook.close_count == 1
        assert not hasattr(hook, "__enter__")
        assert not hasattr(hook, "__exit__")

    def test_run_prefers_exit_over_close(self, device: str) -> None:
        """Hooks with both __exit__ and close() only get __exit__ called."""
        hook = _MockCMHook()

        dynamics = _make_dynamics(device=device)
        dynamics.register_hook(hook)

        batch = _make_batch(device=device)
        dynamics.run(batch, n_steps=3)

        assert hook.exit_count == 1
        assert hook.close_count == 0  # __exit__ called, not close()

    def test_idempotent_close_via_user_with_and_engine(self, device: str) -> None:
        """LoggingHook guards against double-close (user with + engine run)."""
        records: list[dict] = []

        def noop_writer(record: dict) -> None:
            records.append(record)

        hook = LoggingHook(backend="custom", writer_fn=noop_writer)

        # User manually enters
        hook.__enter__()

        dynamics = _make_dynamics(device=device)
        dynamics.register_hook(hook)

        batch = _make_batch(device=device)
        dynamics.run(batch, n_steps=2)  # engine calls __exit__

        # User manually exits again — should not raise
        hook.__exit__(None, None, None)

    def test_multi_stage_hook_entered_once(self, device: str) -> None:
        """Hook registered at multiple stages is entered/exited only once."""
        hook = _MockCMHook()
        # Register hook at multiple stages via `stages` attribute
        hook.stages = [HookStageEnum.AFTER_STEP, HookStageEnum.AFTER_COMPUTE]

        dynamics = _make_dynamics(device=device)
        dynamics.register_hook(hook)

        batch = _make_batch(device=device)
        dynamics.run(batch, n_steps=2)

        # Despite being in two stage lists, only one enter/exit
        assert hook.enter_count == 1
        assert hook.exit_count == 1
