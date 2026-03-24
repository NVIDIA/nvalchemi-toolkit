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
"""Cross-context tests for hooks usable in both dynamics and training.

Hooks whose ``__call__`` accepts a generic ``Enum`` stage (or an explicit
union of stage types) should work identically regardless of which
workflow dispatches them.  This module verifies that these hooks behave
correctly when called with both :class:`DynamicsStage` and
:class:`TrainingStage` contexts.
"""

from __future__ import annotations

import pytest
import torch

from nvalchemi.data import AtomicData, Batch
from nvalchemi.dynamics.base import DynamicsStage
from nvalchemi.dynamics.hooks.logging import LoggingHook
from nvalchemi.dynamics.hooks.neighbor_list import NeighborListHook
from nvalchemi.dynamics.hooks.periodic import WrapPeriodicHook
from nvalchemi.dynamics.hooks.profiling import ProfilerHook
from nvalchemi.dynamics.hooks.safety import MaxForceClampHook, NaNDetectorHook
from nvalchemi.dynamics.hooks.snapshot import SnapshotHook
from nvalchemi.dynamics.sinks import HostMemory
from nvalchemi.hooks import HookContext
from nvalchemi.models.base import NeighborConfig
from nvalchemi.training import TrainingStage

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_batch(n_graphs: int = 2, atoms_per_graph: int = 3) -> Batch:
    """Create a test batch with forces and energies."""
    data_list = [
        AtomicData(
            atomic_numbers=torch.tensor([6] * atoms_per_graph, dtype=torch.long),
            positions=torch.randn(atoms_per_graph, 3),
        )
        for _ in range(n_graphs)
    ]
    batch = Batch.from_data_list(data_list)
    batch.__dict__["forces"] = torch.randn(batch.num_nodes, 3)
    batch.__dict__["energies"] = torch.randn(batch.num_graphs, 1)
    return batch


def _make_ctx(batch: Batch, step_count: int = 0) -> HookContext:
    """Build a minimal HookContext (no model, no dynamics-specific fields)."""
    return HookContext(batch=batch, step_count=step_count)


# ===========================================================================
# NaNDetectorHook
# ===========================================================================


class TestNaNDetectorHookCrossContext:
    """NaNDetectorHook fires correctly under both stage enums."""

    def test_dynamics_stage_no_nan(self) -> None:
        """Clean batch passes under DynamicsStage."""
        hook = NaNDetectorHook(stage=DynamicsStage.AFTER_COMPUTE)
        ctx = _make_ctx(_make_batch())
        hook(ctx, DynamicsStage.AFTER_COMPUTE)

    def test_training_stage_no_nan(self) -> None:
        """Clean batch passes under TrainingStage."""
        hook = NaNDetectorHook(stage=TrainingStage.AFTER_FORWARD)
        ctx = _make_ctx(_make_batch())
        hook(ctx, TrainingStage.AFTER_FORWARD)

    def test_training_stage_detects_nan(self) -> None:
        """NaN in forces is caught under TrainingStage."""
        hook = NaNDetectorHook(stage=TrainingStage.AFTER_FORWARD)
        batch = _make_batch()
        batch.forces[0, 0] = float("nan")
        ctx = _make_ctx(batch)
        with pytest.raises(RuntimeError, match="Non-finite values detected"):
            hook(ctx, TrainingStage.AFTER_FORWARD)


# ===========================================================================
# MaxForceClampHook
# ===========================================================================


class TestMaxForceClampHookCrossContext:
    """MaxForceClampHook fires correctly under both stage enums."""

    def test_dynamics_stage_clamps(self) -> None:
        """Forces are clamped under DynamicsStage."""
        hook = MaxForceClampHook(max_force=1.0, stage=DynamicsStage.AFTER_COMPUTE)
        batch = _make_batch(n_graphs=1, atoms_per_graph=1)
        batch.__dict__["forces"] = torch.tensor([[10.0, 0.0, 0.0]])
        ctx = _make_ctx(batch)
        hook(ctx, DynamicsStage.AFTER_COMPUTE)
        norm = torch.linalg.vector_norm(batch.forces[0])
        assert torch.isclose(norm, torch.tensor(1.0), atol=1e-5)

    def test_training_stage_clamps(self) -> None:
        """Forces are clamped under TrainingStage."""
        hook = MaxForceClampHook(max_force=1.0, stage=TrainingStage.AFTER_FORWARD)
        batch = _make_batch(n_graphs=1, atoms_per_graph=1)
        batch.__dict__["forces"] = torch.tensor([[10.0, 0.0, 0.0]])
        ctx = _make_ctx(batch)
        hook(ctx, TrainingStage.AFTER_FORWARD)
        norm = torch.linalg.vector_norm(batch.forces[0])
        assert torch.isclose(norm, torch.tensor(1.0), atol=1e-5)


# ===========================================================================
# SnapshotHook
# ===========================================================================


class TestSnapshotHookCrossContext:
    """SnapshotHook writes to sink under both stage enums."""

    def test_dynamics_stage_writes(self) -> None:
        """Sink receives data under DynamicsStage."""
        sink = HostMemory(capacity=100)
        hook = SnapshotHook(sink=sink, stage=DynamicsStage.AFTER_STEP)
        batch = _make_batch()
        ctx = _make_ctx(batch)
        hook(ctx, DynamicsStage.AFTER_STEP)
        assert len(sink) == batch.num_graphs

    def test_training_stage_writes(self) -> None:
        """Sink receives data under TrainingStage."""
        sink = HostMemory(capacity=100)
        hook = SnapshotHook(sink=sink, stage=TrainingStage.AFTER_BATCH)
        batch = _make_batch()
        ctx = _make_ctx(batch)
        hook(ctx, TrainingStage.AFTER_BATCH)
        assert len(sink) == batch.num_graphs


# ===========================================================================
# LoggingHook
# ===========================================================================


class TestLoggingHookCrossContext:
    """LoggingHook logs scalars under both stage enums."""

    @staticmethod
    def _capture_hook(
        **kwargs,
    ) -> tuple[LoggingHook, list[tuple[int, list[dict[str, float]]]]]:
        """Create a LoggingHook with a custom backend that captures rows."""
        captured: list[tuple[int, list[dict[str, float]]]] = []

        def writer(step: int, rows: list[dict[str, float]]) -> None:
            captured.append((step, rows))

        return LoggingHook(backend="custom", writer_fn=writer, **kwargs), captured

    def test_dynamics_stage_logs(self) -> None:
        """Rows are emitted under DynamicsStage."""
        hook, captured = self._capture_hook()
        batch = _make_batch(n_graphs=2)
        ctx = _make_ctx(batch)
        with hook:
            hook(ctx, DynamicsStage.AFTER_STEP)
        assert len(captured) == 1
        assert len(captured[0][1]) == 2

    def test_training_stage_logs(self) -> None:
        """Rows are emitted under TrainingStage."""
        hook, captured = self._capture_hook(stage=TrainingStage.AFTER_BATCH)
        batch = _make_batch(n_graphs=3)
        ctx = _make_ctx(batch)
        with hook:
            hook(ctx, TrainingStage.AFTER_BATCH)
        assert len(captured) == 1
        assert len(captured[0][1]) == 3


# ===========================================================================
# WrapPeriodicHook
# ===========================================================================


class TestWrapPeriodicHookCrossContext:
    """WrapPeriodicHook wraps positions under both stage enums."""

    @staticmethod
    def _make_periodic_batch() -> Batch:
        """Create a batch with PBC and positions outside the cell."""
        data = AtomicData(
            atomic_numbers=torch.tensor([6, 6], dtype=torch.long),
            positions=torch.tensor([[12.0, 0.5, 0.5], [-1.0, 0.5, 0.5]]),
            cell=torch.eye(3).unsqueeze(0) * 10.0,
            pbc=torch.tensor([[True, True, True]]),
        )
        return Batch.from_data_list([data])

    def test_dynamics_stage_wraps(self) -> None:
        """Positions are wrapped under DynamicsStage."""
        hook = WrapPeriodicHook(stage=DynamicsStage.AFTER_POST_UPDATE)
        batch = self._make_periodic_batch()
        ctx = _make_ctx(batch)
        hook(ctx, DynamicsStage.AFTER_POST_UPDATE)
        # 12.0 should wrap to 2.0, -1.0 should wrap to 9.0
        assert batch.positions[0, 0].item() < 10.0
        assert batch.positions[1, 0].item() > 0.0

    def test_training_stage_wraps(self) -> None:
        """Positions are wrapped under TrainingStage."""
        hook = WrapPeriodicHook(stage=TrainingStage.BEFORE_FORWARD)
        batch = self._make_periodic_batch()
        ctx = _make_ctx(batch)
        hook(ctx, TrainingStage.BEFORE_FORWARD)
        assert batch.positions[0, 0].item() < 10.0
        assert batch.positions[1, 0].item() > 0.0


# ===========================================================================
# NeighborListHook
# ===========================================================================


class TestNeighborListHookCrossContext:
    """NeighborListHook builds neighbor lists under both stage enums."""

    @staticmethod
    def _make_periodic_batch() -> Batch:
        """Create a batch with PBC for neighbor list computation."""
        data = AtomicData(
            atomic_numbers=torch.tensor([6, 6, 6], dtype=torch.long),
            positions=torch.tensor([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [0.0, 1.5, 0.0]]),
            cell=torch.eye(3).unsqueeze(0) * 10.0,
            pbc=torch.tensor([[True, True, True]]),
        )
        return Batch.from_data_list([data])

    def test_dynamics_stage_builds_neighbors(self) -> None:
        """Neighbor list is built under DynamicsStage."""
        config = NeighborConfig(cutoff=5.0)
        hook = NeighborListHook(config, stage=DynamicsStage.BEFORE_COMPUTE)
        batch = self._make_periodic_batch()
        ctx = _make_ctx(batch)
        hook(ctx, DynamicsStage.BEFORE_COMPUTE)
        assert batch.edge_index is not None

    def test_training_stage_builds_neighbors(self) -> None:
        """Neighbor list is built under TrainingStage."""
        config = NeighborConfig(cutoff=5.0)
        hook = NeighborListHook(config, stage=TrainingStage.BEFORE_FORWARD)
        batch = self._make_periodic_batch()
        ctx = _make_ctx(batch)
        hook(ctx, TrainingStage.BEFORE_FORWARD)
        assert batch.edge_index is not None


# ===========================================================================
# ProfilerHook
# ===========================================================================


class TestProfilerHookCrossContext:
    """ProfilerHook records timing under both stage enums."""

    def test_dynamics_stage_records(self) -> None:
        """Timing is recorded under DynamicsStage."""
        profiler = ProfilerHook({DynamicsStage.BEFORE_STEP, DynamicsStage.AFTER_STEP})
        batch = _make_batch()
        ctx = _make_ctx(batch, step_count=0)
        profiler(ctx, DynamicsStage.BEFORE_STEP)
        profiler(ctx, DynamicsStage.AFTER_STEP)
        summary = profiler.summary()
        assert "BEFORE_STEP->AFTER_STEP" in summary
        assert summary["BEFORE_STEP->AFTER_STEP"]["n_samples"] == 1

    def test_training_stage_records(self) -> None:
        """Timing is recorded under TrainingStage."""
        profiler = ProfilerHook(
            {TrainingStage.BEFORE_FORWARD, TrainingStage.AFTER_FORWARD}
        )
        batch = _make_batch()
        ctx = _make_ctx(batch, step_count=0)
        profiler(ctx, TrainingStage.BEFORE_FORWARD)
        profiler(ctx, TrainingStage.AFTER_FORWARD)
        summary = profiler.summary()
        assert "BEFORE_FORWARD->AFTER_FORWARD" in summary
        assert summary["BEFORE_FORWARD->AFTER_FORWARD"]["n_samples"] == 1

    def test_training_preset(self) -> None:
        """The 'training' preset resolves to TrainingStage members."""
        profiler = ProfilerHook("training")
        assert len(profiler._profiled_stages) >= 2
        assert all(isinstance(s, TrainingStage) for s in profiler._profiled_stages)
        assert TrainingStage.ON_CONVERGE not in profiler._profiled_stages
