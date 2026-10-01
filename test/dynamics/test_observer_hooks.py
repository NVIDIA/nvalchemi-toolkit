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
LoggingHook, EnergyDriftMonitorHook, and StabilityMonitor.
"""

from __future__ import annotations

import csv
import math
import warnings
from collections.abc import Sequence
from enum import Enum
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from nvalchemi.data import AtomicData, Batch
from nvalchemi.dynamics.base import BaseDynamics, DynamicsStage
from nvalchemi.dynamics.hooks import (
    ConvergedSnapshotHook,
    EnergyDriftMonitorHook,
    LoggingHook,
    SnapshotHook,
    StabilityMetrics,
    StabilityMonitor,
    nonfinite_graph_mask,
    total_momentum,
)
from nvalchemi.dynamics.integrators import NVE
from nvalchemi.dynamics.sinks import HostMemory
from nvalchemi.hooks import DynamicsContext, Hook
from nvalchemi.hooks.neighbor_list import NeighborListHook
from nvalchemi.models.demo import DemoModel, DemoModelWrapper
from nvalchemi.models.lj import LennardJonesModelWrapper
from test.dynamics.conftest import (
    ARGON_MASS,
    make_dynamics_context,
    make_lattice_batch,
    make_lattice_data,
)

_LATTICE_ATOMS = 27
"""Atom count of the default 3x3x3 argon lattice."""

_SWING = 0.06
"""Per-atom amplitude of the scripted energy oscillations, in eV."""

_SWING_PERIOD_FS = 500.0
"""Period of the scripted oscillation, in femtoseconds."""

_SWING_SAMPLES = 16
"""Samples per oscillation period."""

_SWING_PERIODS = 4
"""Whole periods the scripted oscillation runs for."""

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
    batch.__dict__["energy"] = torch.randn(batch.num_graphs, 1, device=device)
    if with_velocities:
        batch.__dict__["velocities"] = (
            torch.randn(batch.num_nodes, 3, device=device) * 0.01
        )
        batch.__dict__["atomic_masses"] = torch.full(
            (batch.num_nodes,), 12.0, device=device
        )
    return batch


def _make_dynamics(device: str = "cpu") -> BaseDynamics:
    model = DemoModelWrapper(DemoModel())
    if device != "cpu":
        model = model.to(device)
    return BaseDynamics(model, device_type=device)


_make_ctx = make_dynamics_context


# ---------------------------------------------------------------------------
# SnapshotHook
# ---------------------------------------------------------------------------


class TestSnapshotHook:
    def test_writes_to_sink(self, device: str) -> None:
        sink = HostMemory(capacity=100)
        hook = SnapshotHook(sink=sink, frequency=1)
        batch = _make_batch(device=device)
        dynamics = _make_dynamics(device=device)
        ctx = _make_ctx(batch, dynamics)

        hook(ctx, DynamicsStage.AFTER_STEP)
        assert len(sink) == batch.num_graphs

    def test_frequency_respected(self, device: str) -> None:
        sink = HostMemory(capacity=100)
        hook = SnapshotHook(sink=sink, frequency=2)
        dynamics = _make_dynamics(device=device)

        # Register and run — frequency gating is done by dynamics._call_hooks
        dynamics.register_hook(hook)
        assert hook.stage == DynamicsStage.AFTER_STEP
        assert hook.frequency == 2

    def test_multiple_writes(self, device: str) -> None:
        sink = HostMemory(capacity=100)
        hook = SnapshotHook(sink=sink)
        batch = _make_batch(n_graphs=3, device=device)
        dynamics = _make_dynamics(device=device)
        ctx = _make_ctx(batch, dynamics)

        hook(ctx, DynamicsStage.AFTER_STEP)
        hook(ctx, DynamicsStage.AFTER_STEP)
        assert len(sink) == 6

    def test_protocol_compliance(self) -> None:
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
        assert hook.stage == DynamicsStage.ON_CONVERGE

    def test_writes_only_converged_samples(self, device: str) -> None:
        sink = HostMemory(capacity=100)
        hook = ConvergedSnapshotHook(sink=sink)
        batch = _make_batch(n_graphs=4, device=device)
        dynamics = _make_dynamics(device=device)

        # Simulate convergence of graphs 1 and 3
        converged = torch.tensor([1, 3])
        ctx = _make_ctx(batch, dynamics, converged=converged)
        hook(ctx, DynamicsStage.ON_CONVERGE)

        assert len(sink) == 2

    def test_no_write_when_no_converged(self, device: str) -> None:
        sink = HostMemory(capacity=100)
        hook = ConvergedSnapshotHook(sink=sink)
        batch = _make_batch(device=device)
        dynamics = _make_dynamics(device=device)

        ctx = _make_ctx(batch, dynamics, converged=None)
        hook(ctx, DynamicsStage.ON_CONVERGE)
        assert len(sink) == 0

    def test_no_write_when_empty_converged(self, device: str) -> None:
        sink = HostMemory(capacity=100)
        hook = ConvergedSnapshotHook(sink=sink)
        batch = _make_batch(device=device)
        dynamics = _make_dynamics(device=device)

        converged = torch.tensor([], dtype=torch.long)
        ctx = _make_ctx(batch, dynamics, converged=converged)
        hook(ctx, DynamicsStage.ON_CONVERGE)
        assert len(sink) == 0

    def test_all_converged(self, device: str) -> None:
        sink = HostMemory(capacity=100)
        hook = ConvergedSnapshotHook(sink=sink)
        batch = _make_batch(n_graphs=3, device=device)
        dynamics = _make_dynamics(device=device)

        converged = torch.tensor([0, 1, 2])
        ctx = _make_ctx(batch, dynamics, converged=converged)
        hook(ctx, DynamicsStage.ON_CONVERGE)
        assert len(sink) == 3

    def test_neighbor_keys_preserved_on_batch(self, device: str) -> None:
        """Regression: stripping neighbor data must not mutate the live batch.

        Before the fix, ``del batch[key]`` removed neighbor keys from the
        shared batch object, causing KeyError in any hook that read
        neighbor data after ON_CONVERGE.
        """
        sink = HostMemory(capacity=100)
        hook = ConvergedSnapshotHook(sink=sink)
        batch = _make_batch(n_graphs=3, device=device)

        # Attach fake neighbor data to the batch.
        n_atoms = batch.num_nodes
        K = 4  # neighbors per atom
        batch.__dict__["neighbor_matrix"] = torch.zeros(
            n_atoms, K, dtype=torch.long, device=device
        )
        batch.__dict__["num_neighbors"] = torch.full(
            (n_atoms,), K, dtype=torch.long, device=device
        )

        dynamics = _make_dynamics(device=device)
        converged = torch.tensor([1])
        ctx = _make_ctx(batch, dynamics, converged=converged)
        hook(ctx, DynamicsStage.ON_CONVERGE)

        # The live batch must still have its neighbor keys.
        assert batch.neighbor_matrix is not None
        assert batch.num_neighbors is not None
        assert batch.neighbor_matrix.shape == (n_atoms, K)
        assert len(sink) == 1


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
            ctx = _make_ctx(batch, dynamics)
            hook(ctx, DynamicsStage.AFTER_STEP)
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
            ctx = _make_ctx(batch, dynamics)
            hook(ctx, DynamicsStage.AFTER_STEP)
        assert hook._csv_file is None
        assert hook._stream is None

    def test_usable_without_context_manager(self, device: str) -> None:
        """Executor is created in __init__, so hook works without `with`."""
        hook, captured = self._capture_hook()
        batch = _make_batch(device=device)
        dynamics = _make_dynamics(device=device)
        ctx = _make_ctx(batch, dynamics)

        hook(ctx, DynamicsStage.AFTER_STEP)
        hook.close()
        assert len(captured) == 1

    def test_executor_survives_close(self, device: str) -> None:
        """Executor is recreated after close() so hook remains usable."""
        hook, captured = self._capture_hook()
        batch = _make_batch(device=device)
        dynamics = _make_dynamics(device=device)
        ctx = _make_ctx(batch, dynamics)

        with hook:
            hook(ctx, DynamicsStage.AFTER_STEP)
        assert len(captured) == 1

        # Still usable after close via a new context
        with hook:
            hook(ctx, DynamicsStage.AFTER_STEP)
        assert len(captured) == 2

    def test_one_row_per_graph(self, device: str) -> None:
        hook, captured = self._capture_hook()
        batch = _make_batch(n_graphs=3, device=device)
        dynamics = _make_dynamics(device=device)
        ctx = _make_ctx(batch, dynamics)

        with hook:
            hook(ctx, DynamicsStage.AFTER_STEP)

        assert len(captured) == 1
        step, rows = captured[0]
        assert step == 0
        assert len(rows) == 3  # one row per graph

    def test_row_contains_step_graph_idx_status(self, device: str) -> None:
        hook, captured = self._capture_hook()
        batch = _make_batch(n_graphs=2, device=device)
        dynamics = _make_dynamics(device=device)
        ctx = _make_ctx(batch, dynamics)

        with hook:
            hook(ctx, DynamicsStage.AFTER_STEP)

        rows = captured[0][1]
        for i, row in enumerate(rows):
            assert row["step"] == 0.0
            assert row["graph_idx"] == float(i)
            assert "status" in row  # 0.0 when no status on batch

    def test_per_graph_energy_and_fmax(self, device: str) -> None:
        hook, captured = self._capture_hook()
        batch = _make_batch(n_graphs=2, device=device)
        dynamics = _make_dynamics(device=device)
        ctx = _make_ctx(batch, dynamics)

        with hook:
            hook(ctx, DynamicsStage.AFTER_STEP)

        rows = captured[0][1]
        assert "energy" in rows[0]
        assert "fmax" in rows[0]
        # Per-graph energy should differ (random)
        # Just verify they're valid floats
        for row in rows:
            assert isinstance(row["energy"], float)
            assert isinstance(row["fmax"], float)

    def test_csv_per_sample_rows(self, device: str, tmp_path: Path) -> None:
        csv_path = tmp_path / "log.csv"
        with LoggingHook(backend="csv", log_path=str(csv_path)) as hook:
            batch = _make_batch(n_graphs=2, device=device)
            dynamics = _make_dynamics(device=device)
            ctx = _make_ctx(batch, dynamics)

            hook(ctx, DynamicsStage.AFTER_STEP)
            dynamics.step_count = 1
            ctx = _make_ctx(batch, dynamics)
            hook(ctx, DynamicsStage.AFTER_STEP)

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
        ctx = _make_ctx(batch, dynamics)

        with hook:
            hook(ctx, DynamicsStage.AFTER_STEP)

        assert len(captured) == 1
        assert captured[0][0] == 0
        assert isinstance(captured[0][1], list)
        assert isinstance(captured[0][1][0], dict)

    def test_custom_scalar_float_broadcast(self, device: str) -> None:
        hook, captured = self._capture_hook(
            custom_scalars={"n_atoms": lambda ctx: float(ctx.batch.num_nodes)},
        )
        batch = _make_batch(n_graphs=2, atoms_per_graph=5, device=device)
        dynamics = _make_dynamics(device=device)
        ctx = _make_ctx(batch, dynamics)

        with hook:
            hook(ctx, DynamicsStage.AFTER_STEP)

        rows = captured[0][1]
        # Float is broadcast to all graphs
        assert rows[0]["n_atoms"] == 10.0
        assert rows[1]["n_atoms"] == 10.0

    def test_custom_scalar_tensor_per_graph(self, device: str) -> None:
        hook, captured = self._capture_hook(
            custom_scalars={
                "per_graph_val": lambda ctx: torch.arange(
                    ctx.batch.num_graphs,
                    dtype=torch.float32,
                    device=ctx.batch.positions.device,
                ),
            },
        )
        batch = _make_batch(n_graphs=3, device=device)
        dynamics = _make_dynamics(device=device)
        ctx = _make_ctx(batch, dynamics)

        with hook:
            hook(ctx, DynamicsStage.AFTER_STEP)

        rows = captured[0][1]
        assert rows[0]["per_graph_val"] == 0.0
        assert rows[1]["per_graph_val"] == 1.0
        assert rows[2]["per_graph_val"] == 2.0

    def test_custom_scalar_overrides_default(self, device: str) -> None:
        hook, captured = self._capture_hook(
            custom_scalars={"energy": lambda ctx: 42.0},
        )
        batch = _make_batch(device=device)
        dynamics = _make_dynamics(device=device)
        ctx = _make_ctx(batch, dynamics)

        with hook:
            hook(ctx, DynamicsStage.AFTER_STEP)

        rows = captured[0][1]
        for row in rows:
            assert row["energy"] == 42.0

    def test_temperature_logged_with_velocities(self, device: str) -> None:
        hook, captured = self._capture_hook()
        batch = _make_batch(with_velocities=True, device=device)
        dynamics = _make_dynamics(device=device)
        ctx = _make_ctx(batch, dynamics)

        with hook:
            hook(ctx, DynamicsStage.AFTER_STEP)

        assert "temperature" in captured[0][1][0]

    def test_zero_velocities_give_zero_temperature(self, device: str) -> None:
        hook, captured = self._capture_hook()
        batch = _make_batch(with_velocities=False, device=device)
        dynamics = _make_dynamics(device=device)
        ctx = _make_ctx(batch, dynamics)

        with hook:
            hook(ctx, DynamicsStage.AFTER_STEP)

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

    # ------------------------------------------------------------------
    # Snapshot decoupling (regression: CUDA stream race → -inf fmax)
    # ------------------------------------------------------------------

    def test_cuda_snapshot_records_logging_stream(self, gpu_device: str) -> None:
        """CUDA snapshot storage must be recorded on the D2H copy stream."""
        hook, _ = self._capture_hook(
            custom_scalars={"cpu_value": lambda _ctx: torch.ones(2)}
        )
        batch = _make_batch(n_graphs=2, device=gpu_device)
        dynamics = _make_dynamics(device=gpu_device)
        ctx = _make_ctx(batch, dynamics)

        with patch.object(torch.Tensor, "record_stream", autospec=True) as record:
            with hook:
                stream = hook._stream
                assert stream is not None
                hook(ctx, DynamicsStage.AFTER_STEP)

        assert record.call_count == 6
        assert all(call.args == (stream,) for call in record.call_args_list)

    def test_snapshot_decouples_energy_from_batch(self, device: str) -> None:
        """Snapshot must break view-aliasing between td["energy"] and batch.energy.

        Regression test: ``_compute_columns`` stores ``batch.energy.squeeze(-1)``
        which is a *view* sharing storage with the live batch tensor.  Without
        ``_snapshot_tensordict``, the async D2H on the logging side-stream
        races with the next step's overwrites on the main stream, producing
        corrupted (NaN / -inf) log rows.
        """
        from nvalchemi.dynamics.hooks.logging import _snapshot_tensordict

        batch = _make_batch(n_graphs=2, device=device)
        dynamics = _make_dynamics(device=device)
        ctx = _make_ctx(batch, dynamics)

        hook, _ = self._capture_hook()
        td = hook._compute_columns(batch, step_count=0, ctx=ctx)

        # energy column is a view of batch.energy — verify the premise
        assert (
            td["energy"].untyped_storage().data_ptr()
            == batch.energy.untyped_storage().data_ptr()
        )

        td_snap = _snapshot_tensordict(td)
        energy_snap = td_snap["energy"].clone()

        # Simulate the next dynamics step overwriting batch.energy
        batch.energy.fill_(float("nan"))

        # Snapshot must be unaffected
        assert torch.equal(td_snap["energy"], energy_snap)
        assert not td_snap["energy"].isnan().any()

    def test_logged_values_are_finite(self, device: str) -> None:
        """All logged scalars must be finite for a well-formed batch.

        Regression test: the -inf fmax symptom observed in production was
        caused by the amax sentinel (``-inf``) leaking into log rows when
        the D2H transfer raced with batch mutations.
        """
        hook, captured = self._capture_hook()
        batch = _make_batch(n_graphs=2, with_velocities=True, device=device)
        # Use known positive forces so fmax is well-defined
        batch.__dict__["forces"] = torch.ones(batch.num_nodes, 3, device=device)
        dynamics = _make_dynamics(device=device)
        ctx = _make_ctx(batch, dynamics)

        with hook:
            hook(ctx, DynamicsStage.AFTER_STEP)

        rows = captured[0][1]
        for row in rows:
            for key, val in row.items():
                assert not (val != val), f"{key} is NaN"  # NaN != NaN
                assert val != float("inf"), f"{key} is +inf"
                assert val != float("-inf"), f"{key} is -inf"


# ---------------------------------------------------------------------------
# EnergyDriftMonitorHook
# ---------------------------------------------------------------------------


class TestEnergyDriftMonitorHook:
    def test_first_call_captures_reference(self, device: str) -> None:
        hook = EnergyDriftMonitorHook(threshold=1.0)
        batch = _make_batch(device=device)
        dynamics = _make_dynamics(device=device)
        ctx = _make_ctx(batch, dynamics)

        # First call should just capture reference, not raise
        hook(ctx, DynamicsStage.AFTER_STEP)
        assert hook._reference_total_energy is not None

    def test_no_drift_no_action(self, device: str) -> None:
        hook = EnergyDriftMonitorHook(threshold=1.0, metric="absolute")
        batch = _make_batch(device=device)
        # Set constant energy
        batch.__dict__["energy"] = torch.tensor([[1.0], [2.0]], device=device)
        dynamics = _make_dynamics(device=device)
        ctx = _make_ctx(batch, dynamics)

        hook(ctx, DynamicsStage.AFTER_STEP)  # capture reference
        dynamics.step_count = 1
        ctx = _make_ctx(batch, dynamics)
        hook(ctx, DynamicsStage.AFTER_STEP)  # same energy, no drift

    def test_drift_exceeds_threshold_warn(
        self, device: str, capfd: pytest.CaptureFixture
    ) -> None:
        hook = EnergyDriftMonitorHook(threshold=0.01, metric="absolute", action="warn")
        batch = _make_batch(device=device)
        batch.__dict__["energy"] = torch.tensor([[1.0], [2.0]], device=device)
        dynamics = _make_dynamics(device=device)
        ctx = _make_ctx(batch, dynamics)

        hook(ctx, DynamicsStage.AFTER_STEP)  # capture reference

        # Introduce drift
        batch.__dict__["energy"] = torch.tensor([[2.0], [3.0]], device=device)
        dynamics.step_count = 1
        ctx = _make_ctx(batch, dynamics)
        hook(ctx, DynamicsStage.AFTER_STEP)  # should warn, not raise

    def test_drift_exceeds_threshold_raise(self, device: str) -> None:
        hook = EnergyDriftMonitorHook(threshold=0.01, metric="absolute", action="raise")
        batch = _make_batch(device=device)
        batch.__dict__["energy"] = torch.tensor([[1.0], [2.0]], device=device)
        dynamics = _make_dynamics(device=device)
        ctx = _make_ctx(batch, dynamics)

        hook(ctx, DynamicsStage.AFTER_STEP)  # capture reference

        batch.__dict__["energy"] = torch.tensor([[2.0], [3.0]], device=device)
        dynamics.step_count = 1
        ctx = _make_ctx(batch, dynamics)
        with pytest.raises(RuntimeError, match="Energy drift"):
            hook(ctx, DynamicsStage.AFTER_STEP)

    def test_per_atom_per_step_normalization(self, device: str) -> None:
        hook = EnergyDriftMonitorHook(
            threshold=1e10, metric="per_atom_per_step", action="raise"
        )
        batch = _make_batch(n_graphs=1, atoms_per_graph=10, device=device)
        batch.__dict__["energy"] = torch.tensor([[0.0]], device=device)
        dynamics = _make_dynamics(device=device)
        ctx = _make_ctx(batch, dynamics)

        hook(ctx, DynamicsStage.AFTER_STEP)  # capture reference

        batch.__dict__["energy"] = torch.tensor([[1.0]], device=device)
        dynamics.step_count = 10
        ctx = _make_ctx(batch, dynamics)
        # drift = |1.0 - 0.0| / (10 atoms * 10 steps) = 0.01
        # This is well below 1e10, so no raise
        hook(ctx, DynamicsStage.AFTER_STEP)

    def test_per_atom_per_step_exceeds(self, device: str) -> None:
        hook = EnergyDriftMonitorHook(
            threshold=0.005, metric="per_atom_per_step", action="raise"
        )
        batch = _make_batch(n_graphs=1, atoms_per_graph=10, device=device)
        batch.__dict__["energy"] = torch.tensor([[0.0]], device=device)
        dynamics = _make_dynamics(device=device)
        ctx = _make_ctx(batch, dynamics)

        hook(ctx, DynamicsStage.AFTER_STEP)

        batch.__dict__["energy"] = torch.tensor([[1.0]], device=device)
        dynamics.step_count = 10
        ctx = _make_ctx(batch, dynamics)
        # drift = |1.0| / (10 * 10) = 0.01 > 0.005
        with pytest.raises(RuntimeError, match="Energy drift"):
            hook(ctx, DynamicsStage.AFTER_STEP)

    def test_include_kinetic_false(self, device: str) -> None:
        hook = EnergyDriftMonitorHook(
            threshold=1e10, metric="absolute", include_kinetic=False
        )
        batch = _make_batch(with_velocities=True, device=device)
        batch.__dict__["energy"] = torch.tensor([[1.0], [2.0]], device=device)
        dynamics = _make_dynamics(device=device)
        ctx = _make_ctx(batch, dynamics)

        # Should work without KE even though velocities are present
        hook(ctx, DynamicsStage.AFTER_STEP)
        dynamics.step_count = 1
        ctx = _make_ctx(batch, dynamics)
        hook(ctx, DynamicsStage.AFTER_STEP)

    def test_include_kinetic_true(self, device: str) -> None:
        hook = EnergyDriftMonitorHook(
            threshold=1e10, metric="absolute", include_kinetic=True
        )
        batch = _make_batch(with_velocities=True, device=device)
        batch.__dict__["energy"] = torch.tensor([[1.0], [2.0]], device=device)
        dynamics = _make_dynamics(device=device)
        ctx = _make_ctx(batch, dynamics)

        hook(ctx, DynamicsStage.AFTER_STEP)
        dynamics.step_count = 1
        ctx = _make_ctx(batch, dynamics)
        hook(ctx, DynamicsStage.AFTER_STEP)

    def test_multi_graph_max_drift(self, device: str) -> None:
        hook = EnergyDriftMonitorHook(threshold=0.5, metric="absolute", action="raise")
        batch = _make_batch(n_graphs=2, device=device)
        batch.__dict__["energy"] = torch.tensor([[0.0], [0.0]], device=device)
        dynamics = _make_dynamics(device=device)
        ctx = _make_ctx(batch, dynamics)

        hook(ctx, DynamicsStage.AFTER_STEP)

        # Only one graph drifts above threshold
        batch.__dict__["energy"] = torch.tensor([[0.1], [1.0]], device=device)
        dynamics.step_count = 1
        ctx = _make_ctx(batch, dynamics)
        with pytest.raises(RuntimeError, match="Energy drift"):
            hook(ctx, DynamicsStage.AFTER_STEP)


# ---------------------------------------------------------------------------
# Hook lifecycle management (_open_hooks / _close_hooks)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# StabilityMonitor
# ---------------------------------------------------------------------------


def _drive(monitor: StabilityMonitor, batch: Batch, energies: Sequence[float]) -> None:
    """Fire *monitor* once per scripted total energy, one step apart."""
    for step, energy in enumerate(energies):
        batch.energy = torch.full((batch.num_graphs, 1), energy)
        monitor(DynamicsContext(batch=batch, step_count=step), DynamicsStage.AFTER_STEP)


def _swing(*, closed: bool) -> list[float]:
    """Return a bounded per-atom oscillation as one total energy per step.

    The closed series is a cosine over whole periods, ending on the sample it
    started from; the open one is a sine over the same span, stopping one
    sample short of closing, which is what a window cut mid-oscillation looks
    like. Both swing by the same amplitude about the same mean.
    """
    samples = _SWING_PERIODS * _SWING_SAMPLES + (1 if closed else 0)
    wave = math.cos if closed else math.sin
    return [
        _LATTICE_ATOMS * (-1.0 + _SWING * wave(2.0 * math.pi * step / _SWING_SAMPLES))
        for step in range(samples)
    ]


def _make_geometry_only_batch() -> Batch:
    """Return the moving lattice with every field an NVE run needs but no energy."""
    lattice = make_lattice_data(speed=0.002, jitter=0.15)
    data = AtomicData(
        positions=lattice.positions,
        atomic_numbers=lattice.atomic_numbers,
        atomic_masses=lattice.atomic_masses,
        cell=lattice.cell,
        pbc=lattice.pbc,
        forces=torch.zeros_like(lattice.positions),
    )
    data.add_node_property("velocities", lattice.velocities)
    return Batch.from_data_list([data])


def _make_pair_at_rest(dtype: torch.dtype) -> Batch:
    """Return a two-atom batch at rest whose energy buffer is held in *dtype*."""
    data = AtomicData(
        positions=torch.zeros(2, 3, dtype=dtype),
        atomic_numbers=torch.ones(2, dtype=torch.long),
        atomic_masses=torch.ones(2, dtype=dtype),
        energy=torch.zeros(1, 1, dtype=dtype),
    )
    data.add_node_property("velocities", torch.zeros(2, 3, dtype=dtype))
    return Batch.from_data_list([data])


def _make_identified_batch(
    system_ids: Sequence[int], cells: Sequence[int] = (2, 2)
) -> Batch:
    """Return one lattice graph per entry of *system_ids*, tagged and sized to match."""
    structures = []
    for system_id, count in zip(system_ids, cells, strict=True):
        data = make_lattice_data(cells=count)
        data.add_system_property("system_id", torch.tensor([[system_id]]))
        structures.append(data)
    return Batch.from_data_list(structures)


def _make_lj_nve(monitor: StabilityMonitor | None = None) -> NVE:
    """Return an NVE integrator over an argon Lennard-Jones model with its neighbor hook."""
    model = LennardJonesModelWrapper(epsilon=0.01, sigma=3.4, cutoff=5.0)
    hooks = [
        NeighborListHook(
            config=model.model_config.neighbor_config,
            skin=1.0,
            stage=DynamicsStage.BEFORE_COMPUTE,
        )
    ]
    if monitor is not None:
        hooks.append(monitor)
    return NVE(model=model, dt=1.0, hooks=hooks)


class TestStabilityMonitor:
    """Drift and momentum metrics over a recorded trajectory."""

    def test_scripted_linear_drift_matches_the_analytic_rate(self) -> None:
        """A total energy rising by a fixed amount per step reports that slope."""
        monitor = StabilityMonitor(timestep_fs=2.0)
        batch = make_lattice_batch()
        _drive(monitor, batch, [1.0 + 0.027 * step for step in range(11)])
        metrics = monitor.metrics()
        assert metrics.num_samples == 11
        assert metrics.energy_drift_per_atom == pytest.approx(0.27 / _LATTICE_ATOMS)
        assert metrics.energy_drift_per_atom_per_step == pytest.approx(0.001)
        assert metrics.energy_drift_per_atom_per_ns == pytest.approx(500.0)

    @pytest.mark.parametrize(
        ("dtype", "include_kinetic"),
        [(torch.float64, False), (torch.float32, False), (torch.float64, True)],
        ids=["float64-potential", "float32-potential", "float64-total"],
    )
    def test_samples_are_copied_off_a_buffer_written_in_place(
        self, dtype: torch.dtype, include_kinetic: bool
    ) -> None:
        """An energy the propagator overwrites with copy_ leaves earlier samples intact."""
        batch = _make_pair_at_rest(dtype)
        monitor = StabilityMonitor(include_kinetic=include_kinetic)
        for step, energy in enumerate([0.0, 2.0, 4.0]):
            batch.energy.copy_(torch.full_like(batch.energy, energy))
            monitor(
                DynamicsContext(batch=batch, step_count=step), DynamicsStage.AFTER_STEP
            )
        assert monitor.metrics().energy_drift_per_atom_per_step == pytest.approx(1.0)

    def test_kinetic_energy_is_included_by_default(self) -> None:
        """Only the kinetic-aware monitor sees a constant-potential run heating up."""
        batch = make_lattice_batch()
        total = StabilityMonitor()
        potential = StabilityMonitor(include_kinetic=False)
        for step in range(2):
            batch.velocities = torch.full((batch.num_nodes, 3), 0.1 * step)
            batch.energy = torch.ones(1, 1)
            for monitor in (total, potential):
                monitor(
                    DynamicsContext(batch=batch, step_count=step),
                    DynamicsStage.AFTER_STEP,
                )
        assert potential.metrics().energy_drift_per_atom == 0.0
        assert total.metrics().energy_drift_per_atom == pytest.approx(
            0.5 * ARGON_MASS * 3.0 * 0.1**2
        )

    def test_momentum_drift_matches_the_scripted_velocity_change(self) -> None:
        """Momentum drift is the total mass times the velocity it drifted by."""
        monitor = StabilityMonitor()
        batch = make_lattice_batch()
        for step in range(3):
            batch.velocities = torch.zeros(batch.num_nodes, 3)
            batch.velocities[:, 0] = 0.25 * step
            batch.energy = torch.zeros(1, 1)
            monitor(
                DynamicsContext(batch=batch, step_count=step), DynamicsStage.AFTER_STEP
            )
        expected = ARGON_MASS * _LATTICE_ATOMS * 0.5
        assert monitor.metrics().max_momentum_drift == pytest.approx(expected, rel=1e-5)

    def test_a_single_sample_cannot_be_scored(self) -> None:
        """One recorded frame gives no interval to measure drift over."""
        monitor = StabilityMonitor()
        _drive(monitor, make_lattice_batch(), [1.0])
        with pytest.raises(ValueError, match="at least two recorded samples"):
            monitor.metrics()

    def test_the_metrics_accessor_stays_a_method(self) -> None:
        """``metrics`` is a method, so reading it uncalled is not the metrics."""
        monitor = StabilityMonitor()
        _drive(monitor, make_lattice_batch(), [1.0, 2.0])
        assert not isinstance(StabilityMonitor.__dict__["metrics"], property)
        assert isinstance(monitor.metrics(), StabilityMetrics)
        assert not isinstance(monitor.metrics, StabilityMetrics)

    def test_drift_rate_is_omitted_without_a_timestep(self) -> None:
        """Steps become nanoseconds only when a timestep says how long one is."""
        monitor = StabilityMonitor()
        _drive(monitor, make_lattice_batch(), [1.0, 2.0])
        assert monitor.metrics().energy_drift_per_atom_per_ns is None

    def test_changing_graph_count_stops_recording_and_warns(self) -> None:
        """A batch that graduated systems is not folded into the same series."""
        monitor = StabilityMonitor()
        _drive(monitor, make_lattice_batch(), [1.0, 2.0])
        graduated = make_lattice_batch()
        graduated = Batch.from_data_list(graduated.to_data_list() * 2)
        with pytest.warns(UserWarning, match="went from .* graphs"):
            monitor(
                DynamicsContext(batch=graduated, step_count=9), DynamicsStage.AFTER_STEP
            )
        assert monitor.metrics().num_samples == 2

    def test_a_refill_of_differently_sized_systems_stops_recording(self) -> None:
        """Same graph count, different atom counts, is still a different series."""
        monitor = StabilityMonitor()
        _drive(monitor, _make_identified_batch([0, 1]), [1.0, 2.0])
        refilled = _make_identified_batch([0, 1], cells=(2, 3))
        with pytest.warns(UserWarning, match="went from .* graphs"):
            monitor(
                DynamicsContext(batch=refilled, step_count=9), DynamicsStage.AFTER_STEP
            )
        assert monitor.metrics().num_samples == 2

    def test_a_shape_preserving_refill_stops_recording(self) -> None:
        """Fresh systems in the same slots break the series even at the same size."""
        monitor = StabilityMonitor()
        _drive(monitor, _make_identified_batch([0, 1]), [1.0, 2.0])
        refilled = _make_identified_batch([2, 3])
        with pytest.warns(
            UserWarning, match="replaced by others of the same atom counts"
        ):
            monitor(
                DynamicsContext(batch=refilled, step_count=9), DynamicsStage.AFTER_STEP
            )
        assert monitor.metrics().num_samples == 2

    def test_the_same_systems_keep_being_recorded(self) -> None:
        """An unchanged inflight batch is not mistaken for a refilled one."""
        monitor = StabilityMonitor()
        _drive(monitor, _make_identified_batch([0, 1]), [1.0, 2.0, 3.0])
        assert monitor.metrics().num_samples == 3

    def test_a_symmetric_excursion_fits_a_zero_drift_rate(self) -> None:
        """A run that heats up and cools back down is scored as no net drift."""
        monitor = StabilityMonitor(timestep_fs=1.0)
        _drive(monitor, make_lattice_batch(), [0.0, 2.0, 3.0, 2.0, 0.0])
        metrics = monitor.metrics()
        assert metrics.energy_drift_per_atom == pytest.approx(0.0)
        assert metrics.energy_drift_per_atom_per_ns == pytest.approx(0.0, abs=1e-9)

    def test_a_closed_oscillation_is_only_seen_by_the_diagnostics(self) -> None:
        """Both drift figures read zero on a swing the fluctuation sizes exactly."""
        monitor = StabilityMonitor(timestep_fs=_SWING_PERIOD_FS / _SWING_SAMPLES)
        _drive(monitor, make_lattice_batch(), _swing(closed=True))
        metrics = monitor.metrics()
        assert metrics.energy_drift_per_atom == pytest.approx(0.0, abs=1e-9)
        assert metrics.energy_drift_per_atom_per_ns == pytest.approx(0.0, abs=1e-9)
        assert metrics.energy_fluctuation_per_atom == pytest.approx(
            _SWING / math.sqrt(2.0), rel=0.02
        )
        assert metrics.max_energy_excursion_per_atom == pytest.approx(
            2.0 * _SWING, rel=1e-5
        )

    def test_the_fluctuation_does_not_move_with_where_the_window_ends(self) -> None:
        """The same swing fits a zero rate or a huge one; the fluctuation is fixed."""
        timestep = _SWING_PERIOD_FS / _SWING_SAMPLES
        closed = StabilityMonitor(timestep_fs=timestep)
        open_ended = StabilityMonitor(timestep_fs=timestep)
        _drive(closed, make_lattice_batch(), _swing(closed=True))
        _drive(open_ended, make_lattice_batch(), _swing(closed=False))
        cut = open_ended.metrics()
        assert cut.energy_drift_per_atom_per_ns > 1.0
        assert cut.energy_fluctuation_per_atom == pytest.approx(
            closed.metrics().energy_fluctuation_per_atom, rel=0.05
        )
        assert cut.max_energy_excursion_per_atom == pytest.approx(_SWING, rel=1e-5)

    def test_a_linear_ramp_has_nothing_to_fluctuate_about(self) -> None:
        """A series that is its own fit leaves no residual, and drifts by its rise."""
        monitor = StabilityMonitor(timestep_fs=1.0)
        _drive(
            monitor, make_lattice_batch(), [1.0 + 0.027 * step for step in range(11)]
        )
        metrics = monitor.metrics()
        assert metrics.energy_fluctuation_per_atom == pytest.approx(0.0, abs=1e-6)
        assert metrics.max_energy_excursion_per_atom == pytest.approx(
            metrics.energy_drift_per_atom
        )

    def test_the_metrics_round_trip_through_an_export(self) -> None:
        """Every field, diagnostics included, survives to_dict and back."""
        monitor = StabilityMonitor(timestep_fs=1.0)
        _drive(monitor, make_lattice_batch(), [1.0, 2.0, 4.0])
        metrics = monitor.metrics()
        assert StabilityMetrics.from_dict(metrics.to_dict()) == metrics

    def test_an_export_written_before_the_diagnostics_still_loads(self) -> None:
        """A dict lacking the two newer keys rebuilds with them unmeasured."""
        monitor = StabilityMonitor(timestep_fs=1.0)
        _drive(monitor, make_lattice_batch(), [1.0, 2.0, 4.0])
        exported = monitor.metrics().to_dict()
        older = {
            key: value
            for key, value in exported.items()
            if key
            not in {"energy_fluctuation_per_atom", "max_energy_excursion_per_atom"}
        }
        restored = StabilityMetrics.from_dict(older)
        assert restored.energy_fluctuation_per_atom is None
        assert restored.max_energy_excursion_per_atom is None

    def test_a_geometry_only_batch_names_the_field_it_is_missing(self) -> None:
        """An unpropagated frame is refused by field name rather than sampled."""
        with pytest.raises(ValueError, match=r"carrying no \['energy'\]"):
            StabilityMonitor()(
                DynamicsContext(batch=_make_geometry_only_batch(), step_count=0),
                DynamicsStage.AFTER_STEP,
            )

    def test_a_geometry_only_batch_gains_its_energy_during_the_run(self) -> None:
        """compute() allocates the energy a seed batch lacks, so the run records it."""
        batch = _make_geometry_only_batch()
        monitor = StabilityMonitor()
        _make_lj_nve(monitor).run(batch, n_steps=2)
        assert batch.energy is not None
        assert monitor.metrics().num_samples == 2

    def test_a_status_filtered_dispatch_is_refused(self) -> None:
        """Like its sibling, the monitor cannot yet track a masked subset of graphs."""
        batch = make_lattice_batch()
        dynamics = _make_lj_nve()
        ctx = make_dynamics_context(
            batch, dynamics, active_graph_mask=torch.tensor([True])
        )
        with pytest.raises(NotImplementedError, match="status-filtered"):
            StabilityMonitor()(ctx, DynamicsStage.AFTER_STEP)

    def test_an_equilibration_transient_hides_the_drift_that_follows_it(self) -> None:
        """Discarding the relaxation window recovers the rate the whole fit cancels."""
        rise = 0.03125
        relaxation = [1.0 + rise * (5 - step) for step in range(5)]
        heating = [1.0 + rise * step for step in range(6)]
        whole = StabilityMonitor(timestep_fs=1.0)
        equilibrated = StabilityMonitor(timestep_fs=1.0, warmup_steps=5)
        for monitor in (whole, equilibrated):
            _drive(monitor, make_lattice_batch(), relaxation + heating)
        assert whole.metrics().energy_drift_per_atom == pytest.approx(0.0, abs=1e-9)
        assert whole.metrics().energy_drift_per_atom_per_ns == pytest.approx(
            0.0, abs=1e-6
        )
        metrics = equilibrated.metrics()
        assert metrics.first_step == 5
        assert metrics.num_samples == 6
        assert metrics.energy_drift_per_atom == pytest.approx(5 * rise / _LATTICE_ATOMS)
        assert metrics.energy_drift_per_atom_per_ns == pytest.approx(
            rise / _LATTICE_ATOMS * 1.0e6
        )

    def test_the_series_is_fingerprinted_from_the_first_recorded_sample(self) -> None:
        """A refill inside the warmup window is discarded, not treated as a break."""
        monitor = StabilityMonitor(warmup_steps=2)
        _drive(monitor, _make_identified_batch([0, 1]), [1.0, 2.0])
        refilled = _make_identified_batch([2, 3])
        for step in (2, 3):
            refilled.energy = torch.full((refilled.num_graphs, 1), float(step))
            monitor(
                DynamicsContext(batch=refilled, step_count=step),
                DynamicsStage.AFTER_STEP,
            )
        metrics = monitor.metrics()
        assert metrics.num_samples == 2
        assert metrics.first_step == 2

    def test_lattice_at_rest_holds_its_energy_through_an_nve_run(self) -> None:
        """A Lennard-Jones lattice at its minimum drifts by nothing measurable."""
        monitor = StabilityMonitor(frequency=2, timestep_fs=1.0)
        _make_lj_nve(monitor).run(make_lattice_batch(), n_steps=20)
        metrics = monitor.metrics()
        assert metrics.num_samples == 10
        assert metrics.energy_drift_per_atom_per_step < 1e-9
        assert metrics.max_momentum_drift < 1e-9

    def test_perturbed_lattice_conserves_energy_under_nve(self) -> None:
        """A moving, displaced lattice still conserves energy to MD tolerance."""
        monitor = StabilityMonitor(frequency=5, timestep_fs=1.0)
        _make_lj_nve(monitor).run(
            make_lattice_batch(speed=0.002, jitter=0.15), n_steps=50
        )
        assert monitor.metrics().energy_drift_per_atom_per_step < 1e-6

    def test_the_default_divergence_is_the_core_nonfinite_graph_mask(self) -> None:
        """An unset predicate is the framework's own non-finite check, not a copy."""
        assert StabilityMonitor().divergence is nonfinite_graph_mask

    def test_a_nonfinite_frame_stops_the_series_at_its_step(self) -> None:
        """The default predicate ends the series where a position went non-finite."""
        monitor = StabilityMonitor()
        batch = make_lattice_batch()
        _drive(monitor, batch, [1.0, 1.1, 1.2])
        batch.positions[0, 0] = math.nan
        _drive(monitor, batch, [1.3])
        batch.positions[0, 0] = 0.0
        for step, energy in enumerate([1.4, 1.5], start=1):
            batch.energy = torch.full((1, 1), energy)
            monitor(
                DynamicsContext(batch=batch, step_count=step), DynamicsStage.AFTER_STEP
            )
        metrics = monitor.metrics()
        assert metrics.num_samples == 3
        assert metrics.first_divergence_step == 0
        assert StabilityMetrics.from_dict(metrics.to_dict()) == metrics

    def test_an_undiverged_series_records_no_divergence_step(self) -> None:
        """``None`` says the predicate never fired, not that it was never asked."""
        monitor = StabilityMonitor()
        _drive(monitor, make_lattice_batch(), [1.0, 2.0])
        assert monitor.metrics().first_divergence_step is None

    def test_a_custom_divergence_predicate_decides_the_stop(self) -> None:
        """The predicate is the caller's; here a graph diverges past an energy."""

        def hot(batch: Batch) -> torch.Tensor:
            return batch.energy.reshape(-1) > 2.5

        monitor = StabilityMonitor(divergence=hot)
        _drive(monitor, make_lattice_batch(), [1.0, 2.0, 3.0, 4.0])
        metrics = monitor.metrics()
        assert metrics.num_samples == 2
        assert metrics.first_divergence_step == 2

    def test_the_mean_aggregate_averages_the_graphs_the_max_picks_from(self) -> None:
        """Two graphs drifting differently report their worst or their mean."""
        batch = _make_identified_batch([0, 1])
        counts = batch.num_nodes_per_graph.to(torch.float64)
        worst = StabilityMonitor()
        mean = StabilityMonitor(aggregate="mean")
        for step in range(3):
            batch.energy = torch.tensor([[0.0], [0.5 * step]])
            for monitor in (worst, mean):
                monitor(
                    DynamicsContext(batch=batch, step_count=step),
                    DynamicsStage.AFTER_STEP,
                )
        per_graph = torch.tensor([0.0, 1.0]) / counts
        assert worst.metrics().energy_drift_per_atom == pytest.approx(
            float(per_graph.max())
        )
        assert mean.metrics().energy_drift_per_atom == pytest.approx(
            float(per_graph.mean())
        )
        assert mean.metrics().aggregate == "mean"

    def test_an_unknown_aggregate_is_rejected(self) -> None:
        """Only the two reductions are offered."""
        with pytest.raises(ValueError, match="aggregate must be one of"):
            StabilityMonitor(aggregate="median")

    def test_a_shape_preserving_refill_can_be_recorded_through(self) -> None:
        """Turning the composition stop off keeps an equal-size refill in the series."""
        monitor = StabilityMonitor(stop_on_composition_change=False)
        _drive(monitor, _make_identified_batch([0, 1]), [1.0, 2.0])
        refilled = _make_identified_batch([2, 3])
        refilled.energy = torch.full((2, 1), 3.0)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            monitor(
                DynamicsContext(batch=refilled, step_count=2), DynamicsStage.AFTER_STEP
            )
        assert monitor.metrics().num_samples == 3

    def test_a_resized_refill_stops_recording_whatever_the_setting(self) -> None:
        """Different per-graph atom counts cannot join the series either way."""
        monitor = StabilityMonitor(stop_on_composition_change=False)
        _drive(monitor, _make_identified_batch([0, 1]), [1.0, 2.0])
        refilled = _make_identified_batch([0, 1], cells=(2, 3))
        with pytest.warns(UserWarning, match="went from .* graphs"):
            monitor(
                DynamicsContext(batch=refilled, step_count=9), DynamicsStage.AFTER_STEP
            )
        assert monitor.metrics().num_samples == 2


class TestTotalMomentum:
    """Mass-weighted velocity sums per graph."""

    def test_total_momentum_sums_mass_weighted_velocities_per_graph(self) -> None:
        """A batch at rest carries no momentum, one row per graph."""
        batch = make_lattice_batch()
        torch.testing.assert_close(total_momentum(batch), torch.zeros(1, 3))
        batch.velocities = torch.ones(batch.num_nodes, 3)
        expected = torch.full((1, 3), ARGON_MASS * _LATTICE_ATOMS)
        torch.testing.assert_close(total_momentum(batch), expected)

    def test_each_graph_sums_only_its_own_atoms(self) -> None:
        """Two graphs moving in opposite directions report opposite momenta."""
        batch = _make_identified_batch([0, 1])
        batch.velocities = torch.zeros(batch.num_nodes, 3)
        first = batch.batch_idx == 0
        batch.velocities[first, 0] = 1.0
        batch.velocities[~first, 0] = -1.0
        momentum = total_momentum(batch)
        atoms = batch.num_nodes_per_graph.to(momentum.dtype)
        torch.testing.assert_close(
            momentum[:, 0], ARGON_MASS * atoms * torch.tensor([1.0, -1.0])
        )
        torch.testing.assert_close(momentum[:, 1:], torch.zeros(2, 2))


class _MockCMHook:
    """Mock hook with context-manager protocol for testing lifecycle."""

    def __init__(self) -> None:
        self.frequency = 1
        self.stage = DynamicsStage.AFTER_STEP
        self.enter_count = 0
        self.exit_count = 0
        self.exit_args: list[tuple] = []
        self.call_count = 0
        self.close_count = 0

    def __call__(self, ctx: DynamicsContext, stage: Enum) -> None:
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
        self.stage = DynamicsStage.AFTER_STEP
        self.call_count = 0
        self.close_count = 0

    def __call__(self, ctx: DynamicsContext, stage: Enum) -> None:
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
        hook.stages = [DynamicsStage.AFTER_STEP, DynamicsStage.AFTER_COMPUTE]

        dynamics = _make_dynamics(device=device)
        dynamics.register_hook(hook)

        batch = _make_batch(device=device)
        dynamics.run(batch, n_steps=2)

        # Despite being in two stage lists, only one enter/exit
        assert hook.enter_count == 1
        assert hook.exit_count == 1
