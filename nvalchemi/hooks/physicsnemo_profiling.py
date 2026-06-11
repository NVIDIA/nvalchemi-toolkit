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
"""PhysicsNeMo-backed PyTorch profiler hook."""

from __future__ import annotations

from collections.abc import Callable
from enum import Enum
from pathlib import Path
from typing import Annotated, Any, ClassVar

from physicsnemo.utils.profiling import (
    Profiler,
    TorchProfilerConfig,
    TorchProfileWrapper,
)
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, field_validator
from torch.profiler import ProfilerActivity

from nvalchemi.distributed import (
    DistributedManager,
    resolve_global_rank,
    resolve_world_size,
)
from nvalchemi.hooks._context import HookContext

__all__ = ["TorchProfilerHook"]


def _parse_activity(activity: ProfilerActivity | str) -> ProfilerActivity:
    """Normalize a profiler activity enum or string alias."""
    if isinstance(activity, ProfilerActivity):
        return activity
    normalized = activity.lower()
    match normalized:
        case "cpu":
            return ProfilerActivity.CPU
        case "cuda":
            return ProfilerActivity.CUDA
        case _:
            raise ValueError(
                f"Unknown profiler activity {activity!r}; expected 'cpu' or 'cuda'."
            )


class TorchProfilerHook(BaseModel):
    """Capture PyTorch profiler traces through PhysicsNeMo's profiler wrapper.

    The hook supports both training and dynamics workflows. It starts the
    PhysicsNeMo profiler when entering its context, or lazily at the first
    supported stage if called without a context manager. It advances the PyTorch
    profiler schedule at each batch or dynamics step and finalizes traces at the
    end of training or when the hook context closes.

    Parameters
    ----------
    output_dir : str | Path
        Root directory for profiler outputs.
    activities : tuple[ProfilerActivity | str, ...] | None, optional
        PyTorch profiler activities. ``None`` lets PhysicsNeMo choose CPU and
        CUDA when CUDA is available. Strings may be ``"cpu"`` or ``"cuda"``.
    schedule : Callable | None, optional
        PyTorch profiler schedule created by :func:`torch.profiler.schedule`.
    record_shapes : bool, optional
        Whether to record tensor shapes.
    profile_memory : bool, optional
        Whether to profile memory allocations.
    with_flops : bool, optional
        Whether to estimate FLOPs for supported operations.
    with_stack : bool, optional
        Whether to record Python stack traces.
    on_trace_ready_path : str | Path | None, optional
        Directory passed to PyTorch's tensorboard trace handler. When provided,
        it is rank-suffixed because those traces bypass PhysicsNeMo's final
        ``trace.json`` export.
    frequency : int, optional
        Hook dispatch frequency. Keep the default ``1`` unless you explicitly
        want the profiler schedule to advance less often.
    rank_subdirs : bool, optional
        Whether to place nvalchemi-managed outputs under ``rank_<global_rank>``.
        Enabled by default for a consistent single- and multi-process layout.

    Attributes
    ----------
    stage : Enum | None
        ``None`` because this hook dispatches across training and dynamics
        stages through :meth:`_runs_on_stage`.
    frequency : int
        Hook dispatch cadence.
    """

    output_dir: Annotated[
        Path,
        Field(description="Root directory for PhysicsNeMo profiler outputs."),
    ]
    activities: Annotated[
        tuple[ProfilerActivity, ...] | None,
        Field(
            default=None,
            description=(
                "PyTorch profiler activities, or None to let PhysicsNeMo "
                "choose CPU and CUDA when available."
            ),
        ),
    ] = None
    schedule: Annotated[
        Callable[..., Any] | None,
        Field(default=None, description="Optional torch.profiler schedule."),
    ] = None
    record_shapes: Annotated[
        bool, Field(description="Record input tensor shapes in the trace.")
    ] = True
    profile_memory: Annotated[
        bool, Field(description="Profile memory allocations.")
    ] = True
    with_flops: Annotated[
        bool, Field(description="Estimate FLOPs for supported operations.")
    ] = True
    with_stack: Annotated[bool, Field(description="Record Python stack traces.")] = (
        False
    )
    on_trace_ready_path: Annotated[
        Path | None,
        Field(
            default=None,
            description="Optional path for PyTorch tensorboard trace handler output.",
        ),
    ] = None
    frequency: Annotated[
        int,
        Field(
            default=1,
            ge=1,
            description="Run every N workflow steps.",
        ),
    ] = 1
    name: Annotated[
        str,
        Field(default="torch", description="PhysicsNeMo profiler output name."),
    ] = "torch"
    rank_subdirs: Annotated[
        bool,
        Field(
            default=True,
            description="Write nvalchemi-managed outputs under rank_<global_rank>.",
        ),
    ] = True

    stage: ClassVar[Enum | None] = None

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=False,
        extra="forbid",
    )

    _profiler: Any | None = PrivateAttr(default=None)
    _torch_profiler: Any | None = PrivateAttr(default=None)
    _started: bool = PrivateAttr(default=False)
    _closed: bool = PrivateAttr(default=False)
    _entered_context: bool = PrivateAttr(default=False)

    @field_validator("activities", mode="before")
    @classmethod
    def _normalize_activities(cls, value: Any) -> tuple[ProfilerActivity, ...] | None:
        """Normalize activity aliases before pydantic validation."""
        if value is None:
            return None
        if isinstance(value, (str, ProfilerActivity)):
            raw_values = (value,)
        else:
            raw_values = tuple(value)
        return tuple(_parse_activity(activity) for activity in raw_values)

    def __enter__(self) -> TorchProfilerHook:
        """Enter the hook context and start profiling."""
        self._start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: Any,
    ) -> None:
        """Finalize profiler output when a workflow context exits."""
        self.close()

    def _runs_on_stage(self, stage: Enum) -> bool:
        """Return whether this hook handles ``stage``.

        Parameters
        ----------
        stage : Enum
            Workflow stage enum value.

        Returns
        -------
        bool
            ``True`` for supported training and dynamics stages.
        """
        from nvalchemi.dynamics.base import DynamicsStage
        from nvalchemi.training._stages import TrainingStage

        match stage:
            case (
                TrainingStage.BEFORE_TRAINING
                | TrainingStage.BEFORE_BATCH
                | TrainingStage.AFTER_BATCH
                | TrainingStage.AFTER_TRAINING
                | DynamicsStage.BEFORE_STEP
                | DynamicsStage.AFTER_STEP
            ):
                return True
            case _:
                return False

    def __call__(self, ctx: HookContext, stage: Enum) -> None:
        """Handle a supported training or dynamics stage.

        Parameters
        ----------
        ctx : HookContext
            Workflow context containing rank and workflow metadata.
        stage : Enum
            Current workflow stage.
        """
        from nvalchemi.dynamics.base import DynamicsStage
        from nvalchemi.training._stages import TrainingStage

        match stage:
            case TrainingStage.BEFORE_TRAINING | DynamicsStage.BEFORE_STEP:
                self._start(ctx)
            case TrainingStage.BEFORE_BATCH if not self._started:
                self._start(ctx)
            case TrainingStage.AFTER_BATCH | DynamicsStage.AFTER_STEP:
                if not self._started:
                    self._start(ctx)
                if self._profiler is not None:
                    self._profiler.step()
            case TrainingStage.AFTER_TRAINING:
                self.close()
            case _:
                return

    def _start(self, ctx: HookContext | None = None) -> None:
        """Start the PhysicsNeMo profiler."""
        if self._started:
            return
        if self._closed:
            raise RuntimeError(
                "TorchProfilerHook cannot be restarted after it has finalized."
            )

        profiler = Profiler()
        if getattr(profiler, "initialized", False) or getattr(
            profiler, "enabled", False
        ):
            raise RuntimeError(
                "PhysicsNeMo Profiler is already initialized or enabled. "
                "Create and register TorchProfilerHook before other "
                "PhysicsNeMo profiler configuration, or finalize the existing "
                "profiler before starting this hook."
            )

        rank = resolve_global_rank(None if ctx is None else ctx.global_rank)
        output_path = self._resolve_output_path(rank)
        trace_path = self._resolve_trace_path(rank)
        output_path.mkdir(parents=True, exist_ok=True)
        if trace_path is not None:
            trace_path.mkdir(parents=True, exist_ok=True)

        config = TorchProfilerConfig(
            name=self.name,
            torch_prof_activities=self.activities,
            record_shapes=self.record_shapes,
            with_stack=self.with_stack,
            profile_memory=self.profile_memory,
            with_flops=self.with_flops,
            schedule=self.schedule,
            on_trace_ready_path=trace_path,
        )
        torch_profiler = TorchProfileWrapper(config)
        enabled_torch_profiler = profiler.enable("torch")
        profiler.output_path = output_path
        profiler.__enter__()

        self._profiler = profiler
        self._torch_profiler = enabled_torch_profiler or torch_profiler
        self._started = True
        self._entered_context = True

    def _resolve_output_path(self, rank: int) -> Path:
        """Return the PhysicsNeMo output path for this process."""
        output_dir = self.output_dir
        if DistributedManager.is_initialized() and not DistributedManager().distributed:
            return output_dir
        if self.rank_subdirs or resolve_world_size() > 1:
            return output_dir / f"rank_{rank}"
        return output_dir

    def _resolve_trace_path(self, rank: int) -> Path | None:
        """Return the rank-specific tensorboard trace path, if configured."""
        if self.on_trace_ready_path is None:
            return None
        return self.on_trace_ready_path / f"rank_{rank}"

    def close(self) -> None:
        """Finalize profiler outputs once."""
        if not self._started:
            return
        if self._profiler is None:
            return
        if self._entered_context:
            self._profiler.__exit__(None, None, None)
            self._entered_context = False
        self._profiler.finalize()
        self._started = False
        self._closed = True
