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

    ``TorchProfilerHook`` drives PhysicsNeMo's
    :class:`~physicsnemo.utils.profiling.Profiler` (backed by
    :class:`~physicsnemo.utils.profiling.TorchProfileWrapper`) so that
    ``torch.profiler`` traces are collected for an nvalchemi workflow without
    hand-rolling profiler setup, stepping, and finalization. The same hook attaches to both training and dynamics workflows:
    it recognizes :attr:`TrainingStage.BEFORE_TRAINING`,
    :attr:`~TrainingStage.BEFORE_BATCH`, :attr:`~TrainingStage.AFTER_BATCH`, and
    :attr:`~TrainingStage.AFTER_TRAINING`, plus
    :attr:`DynamicsStage.BEFORE_STEP` and :attr:`~DynamicsStage.AFTER_STEP`.

    The profiler starts when the hook enters its context (``__enter__``) or,
    if it is dispatched by a workflow without being used as a context manager,
    lazily on the first supported start stage. It advances the ``torch.profiler``
    schedule once per batch or dynamics step (at ``AFTER_BATCH`` /
    ``AFTER_STEP``) and finalizes traces at ``AFTER_TRAINING`` or when the hook
    context closes. Register it like any other hook by adding it to a strategy's
    or dynamics object's ``hooks=[...]`` list; for dynamics runs it is also valid
    to wrap the run in a ``with`` block so start/finalize bracket exactly the
    profiled region.

    Outputs are written under ``output_dir`` (named by ``name``). In distributed
    runs, or whenever ``rank_subdirs`` is set, per-process outputs land in
    ``output_dir / rank_<global_rank>``, and the optional
    ``on_trace_ready_path`` TensorBoard handler directory is rank-suffixed the
    same way. Activity selection accepts either
    :class:`~torch.profiler.ProfilerActivity` values or the string aliases
    ``"cpu"`` / ``"cuda"``; ``None`` lets PhysicsNeMo pick CPU and CUDA when
    available.

    Examples
    --------
    Profile a training run by registering the hook alongside the strategy's
    other hooks:

    >>> import torch  # doctest: +SKIP
    >>> from nvalchemi.hooks.physicsnemo_profiling import TorchProfilerHook  # doctest: +SKIP
    >>> from nvalchemi.training import (  # doctest: +SKIP
    ...     EnergyMSELoss, OptimizerConfig, TrainingStrategy, default_training_fn,
    ... )
    >>> profiler = TorchProfilerHook(  # doctest: +SKIP
    ...     output_dir="prof/train",
    ...     activities=("cpu", "cuda"),
    ...     record_shapes=True,
    ...     profile_memory=True,
    ...     with_flops=True,
    ... )
    >>> strategy = TrainingStrategy(  # doctest: +SKIP
    ...     models=model,
    ...     optimizer_configs=OptimizerConfig(
    ...         optimizer_cls=torch.optim.Adam, optimizer_kwargs={"lr": 1e-3},
    ...     ),
    ...     training_fn=default_training_fn,
    ...     loss_fn=EnergyMSELoss(),
    ...     num_epochs=1,
    ...     devices=[torch.device("cuda")],
    ...     hooks=[profiler],
    ... )
    >>> strategy.run(train_loader)  # doctest: +SKIP

    For dynamics, use the hook as a context manager so the profiler brackets the
    exact steps you care about:

    >>> hook = TorchProfilerHook(output_dir="prof/md", activities=("cuda",))  # doctest: +SKIP
    >>> with hook:  # doctest: +SKIP
    ...     dynamics.run(batch, num_steps=100)

    Notes
    -----
    Only one PhysicsNeMo profiler may be active at a time: ``_start`` raises a
    :class:`RuntimeError` if the global
    :class:`~physicsnemo.utils.profiling.Profiler` is already initialized or
    enabled, so construct and register this
    hook before any other PhysicsNeMo profiler configuration. The hook is
    single-use — once finalized it cannot be restarted, and calling it (or
    re-entering it) after ``close`` raises. Finalization happens at
    ``AFTER_TRAINING`` or on context exit; dynamics workflows that never emit an
    ``AFTER_TRAINING`` stage should be run under the ``with`` block (or have
    ``close`` called) to flush traces. ``frequency`` is a :class:`ClassVar`-style
    workflow field, and ``stage`` is ``None`` because the hook handles multiple
    stages itself rather than binding to a single one.
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
