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
"""Tests for the PhysicsNeMo PyTorch profiler hook."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import pytest
import torch

from nvalchemi import distributed as distributed_module
from nvalchemi.dynamics.base import DynamicsStage
from nvalchemi.hooks import HookContext, TorchProfilerHook
from nvalchemi.hooks import physicsnemo_profiling as profiling_module
from nvalchemi.training import TrainingStage


class _CustomStage(Enum):
    """Custom enum with names that should not be claimed by the hook."""

    BEFORE_TRAINING = 0
    AFTER_BATCH = 1
    BEFORE_STEP = 2
    AFTER_STEP = 3


@dataclass
class _FakeConfig:
    """Minimal stand-in for PhysicsNeMo TorchProfilerConfig."""

    name: str = "torch"
    torch_prof_activities: tuple[Any, ...] | None = None
    record_shapes: bool = True
    with_stack: bool = False
    profile_memory: bool = True
    with_flops: bool = True
    schedule: Any = None
    on_trace_ready_path: Path | None = None


class _FakeTorchProfileWrapper:
    """Minimal stand-in for PhysicsNeMo TorchProfileWrapper."""

    last_instance: _FakeTorchProfileWrapper | None = None

    def __init__(self, config: _FakeConfig) -> None:
        self.config = config
        self.enabled = False
        type(self).last_instance = self


class _FakeProfiler:
    """Minimal stand-in for PhysicsNeMo Profiler."""

    def __init__(self) -> None:
        self.initialized = False
        self.enabled = False
        self.output_path: Path | None = None
        self.wrapper: _FakeTorchProfileWrapper | None = None
        self.enter_count = 0
        self.exit_count = 0
        self.step_count = 0
        self.finalize_count = 0

    def enable(
        self, wrapper: _FakeTorchProfileWrapper | str
    ) -> _FakeTorchProfileWrapper:
        self.enabled = True
        if isinstance(wrapper, str):
            resolved = _FakeTorchProfileWrapper.last_instance
            if resolved is None:
                raise RuntimeError("Fake torch profiler was not configured.")
            wrapper = resolved
        wrapper.enabled = True
        self.wrapper = wrapper
        return wrapper

    def __enter__(self) -> _FakeProfiler:
        self.initialized = True
        self.enter_count += 1
        return self

    def __exit__(self, *exc: object) -> None:
        self.exit_count += 1

    def step(self) -> None:
        self.step_count += 1

    def finalize(self) -> None:
        self.finalize_count += 1


@dataclass
class _FakeManager:
    """Structural distributed manager for rank layout tests."""

    world_size: int = 1


@dataclass
class _FakeWorkflow:
    """Workflow object carrying a distributed manager."""

    distributed_manager: _FakeManager | None = None


def _ctx(rank: int = 0, world_size: int = 1) -> HookContext:
    """Build a base hook context for profiler tests."""
    return HookContext(
        batch=None,
        global_rank=rank,
        workflow=_FakeWorkflow(_FakeManager(world_size=world_size)),
    )


@pytest.fixture()
def fake_profiler(monkeypatch: pytest.MonkeyPatch) -> _FakeProfiler:
    """Patch PhysicsNeMo profiler classes with fakes."""
    profiler = _FakeProfiler()

    monkeypatch.setattr(profiling_module, "Profiler", lambda: profiler)
    monkeypatch.setattr(
        profiling_module, "TorchProfileWrapper", _FakeTorchProfileWrapper
    )
    monkeypatch.setattr(profiling_module, "TorchProfilerConfig", _FakeConfig)
    return profiler


def _reset_physicsnemo_profiler_state() -> None:
    """Reset PhysicsNeMo profiler singleton state for smoke tests."""
    try:
        from physicsnemo.utils.profiling import Profiler, TorchProfileWrapper
    except ImportError:
        return
    Profiler._profilers.clear()
    Profiler._decoration_registry.clear()
    Profiler._output_top = Path("./physicsnemo_profiling_outputs/")
    Profiler._initialized = False
    Profiler._clear_instance()
    TorchProfileWrapper._clear_instance()


@pytest.fixture(autouse=True)
def reset_physicsnemo_profiler_state() -> Iterator[None]:
    """Keep real PhysicsNeMo profiler singletons isolated between tests."""
    _reset_physicsnemo_profiler_state()
    yield
    _reset_physicsnemo_profiler_state()


class TestTorchProfilerHookConstruction:
    """TorchProfilerHook construction and stage dispatch."""

    def test_activity_aliases_are_normalized(self, tmp_path: Path) -> None:
        """String activities are normalized to PyTorch profiler enums."""
        hook = TorchProfilerHook(output_dir=tmp_path, activities=("cpu",))
        assert hook.activities == (torch.profiler.ProfilerActivity.CPU,)

    def test_unknown_activity_raises(self, tmp_path: Path) -> None:
        """Unknown activity strings fail validation."""
        with pytest.raises(ValueError, match="Unknown profiler activity"):
            TorchProfilerHook(output_dir=tmp_path, activities=("bogus",))

    def test_runs_on_training_and_dynamics_stages(self, tmp_path: Path) -> None:
        """The hook claims training and dynamics profiler stages only."""
        hook = TorchProfilerHook(output_dir=tmp_path)
        assert hook._runs_on_stage(TrainingStage.BEFORE_TRAINING)
        assert hook._runs_on_stage(TrainingStage.BEFORE_BATCH)
        assert hook._runs_on_stage(TrainingStage.AFTER_BATCH)
        assert hook._runs_on_stage(TrainingStage.AFTER_TRAINING)
        assert hook._runs_on_stage(DynamicsStage.BEFORE_STEP)
        assert hook._runs_on_stage(DynamicsStage.AFTER_STEP)
        assert not hook._runs_on_stage(DynamicsStage.BEFORE_COMPUTE)
        assert not hook._runs_on_stage(_CustomStage.AFTER_BATCH)
        assert not hook._runs_on_stage(_CustomStage.AFTER_STEP)


class TestTorchProfilerHookLifecycle:
    """TorchProfilerHook lifecycle behavior with fake PhysicsNeMo objects."""

    def test_training_lifecycle_starts_steps_and_finalizes(
        self, tmp_path: Path, fake_profiler: _FakeProfiler
    ) -> None:
        """Training stages drive start, step, and finalization."""
        hook = TorchProfilerHook(output_dir=tmp_path)
        ctx = _ctx(rank=1, world_size=2)

        hook(ctx, TrainingStage.BEFORE_TRAINING)
        assert fake_profiler.enter_count == 1
        assert fake_profiler.output_path == tmp_path / "rank_1"

        hook(ctx, TrainingStage.AFTER_BATCH)
        assert fake_profiler.step_count == 1

        hook(ctx, TrainingStage.AFTER_TRAINING)
        hook.close()
        assert fake_profiler.exit_count == 1
        assert fake_profiler.finalize_count == 1

    def test_train_batch_fallback_starts_on_before_batch(
        self, tmp_path: Path, fake_profiler: _FakeProfiler
    ) -> None:
        """Standalone train_batch calls can start without BEFORE_TRAINING."""
        hook = TorchProfilerHook(output_dir=tmp_path)
        ctx = _ctx()

        with hook:
            hook(ctx, TrainingStage.BEFORE_BATCH)
            hook(ctx, TrainingStage.AFTER_BATCH)

        assert fake_profiler.enter_count == 1
        assert fake_profiler.step_count == 1
        assert fake_profiler.finalize_count == 1

    def test_dynamics_lifecycle_starts_steps_and_finalizes(
        self, tmp_path: Path, fake_profiler: _FakeProfiler
    ) -> None:
        """Dynamics stages drive start, step, and context finalization."""
        hook = TorchProfilerHook(output_dir=tmp_path)
        ctx = _ctx(rank=0, world_size=1)

        with hook:
            hook(ctx, DynamicsStage.BEFORE_STEP)
            hook(ctx, DynamicsStage.AFTER_STEP)

        assert fake_profiler.output_path == tmp_path / "rank_0"
        assert fake_profiler.step_count == 1
        assert fake_profiler.exit_count == 1
        assert fake_profiler.finalize_count == 1

    def test_trace_ready_path_is_rank_suffixed(
        self, tmp_path: Path, fake_profiler: _FakeProfiler
    ) -> None:
        """TensorBoard trace handler paths get an explicit rank directory."""
        hook = TorchProfilerHook(
            output_dir=tmp_path / "out",
            on_trace_ready_path=tmp_path / "traces",
            activities=("cpu",),
        )
        hook(_ctx(rank=2, world_size=4), DynamicsStage.BEFORE_STEP)

        assert fake_profiler.wrapper is not None
        assert fake_profiler.wrapper.config.on_trace_ready_path == (
            tmp_path / "traces" / "rank_2"
        )
        assert fake_profiler.output_path == tmp_path / "out" / "rank_2"

    def test_rank_subdirs_can_be_disabled_for_single_process(
        self, tmp_path: Path, fake_profiler: _FakeProfiler
    ) -> None:
        """Single-process runs can write directly under output_dir."""
        hook = TorchProfilerHook(output_dir=tmp_path, rank_subdirs=False)

        hook(_ctx(rank=0, world_size=1), DynamicsStage.BEFORE_STEP)

        assert fake_profiler.output_path == tmp_path

    def test_rank_subdirs_disabled_still_suffixes_distributed_runs(
        self,
        tmp_path: Path,
        fake_profiler: _FakeProfiler,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Distributed runs keep rank suffixing even when single-rank layout opts out."""
        monkeypatch.setenv("WORLD_SIZE", "2")
        hook = TorchProfilerHook(output_dir=tmp_path, rank_subdirs=False)

        hook(_ctx(rank=1, world_size=2), DynamicsStage.BEFORE_STEP)

        assert fake_profiler.output_path == tmp_path / "rank_1"

    def test_physicsnemo_single_process_manager_uses_base_output_dir(
        self,
        tmp_path: Path,
        fake_profiler: _FakeProfiler,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """When PhysicsNeMo is initialized but not distributed, use output_dir."""

        class _InitializedManager:
            distributed = False
            world_size = 1
            rank = 0

            @classmethod
            def is_initialized(cls) -> bool:
                return True

        monkeypatch.setattr(profiling_module, "DistributedManager", _InitializedManager)
        monkeypatch.setattr(
            distributed_module, "DistributedManager", _InitializedManager
        )
        hook = TorchProfilerHook(output_dir=tmp_path)

        hook(_ctx(rank=1, world_size=2), DynamicsStage.BEFORE_STEP)

        assert fake_profiler.output_path == tmp_path

    def test_already_initialized_physicsnemo_profiler_raises(
        self, tmp_path: Path, fake_profiler: _FakeProfiler
    ) -> None:
        """The hook refuses to reconfigure an active PhysicsNeMo profiler."""
        fake_profiler.initialized = True
        hook = TorchProfilerHook(output_dir=tmp_path)

        with pytest.raises(RuntimeError, match="already initialized or enabled"):
            hook(_ctx(), DynamicsStage.BEFORE_STEP)


class TestTorchProfilerHookSmoke:
    """Smoke tests with the real PhysicsNeMo profiler."""

    def test_cpu_trace_is_written(self, tmp_path: Path) -> None:
        """A CPU-only profile writes PhysicsNeMo torch outputs."""
        pytest.importorskip("physicsnemo")
        hook = TorchProfilerHook(
            output_dir=tmp_path,
            activities=("cpu",),
            record_shapes=False,
            profile_memory=False,
            with_flops=False,
        )
        ctx = _ctx()

        with hook:
            hook(ctx, TrainingStage.BEFORE_TRAINING)
            with torch.profiler.record_function("nvalchemi_profiler_smoke"):
                (torch.ones(4) + 1).sum().item()
            hook(ctx, TrainingStage.AFTER_BATCH)

        out_dir = tmp_path / "rank_0" / "torch"
        assert (out_dir / "trace.json").is_file()
        assert (out_dir / "cpu_time.txt").is_file()
