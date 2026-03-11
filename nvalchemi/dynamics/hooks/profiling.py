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
"""
Interval-based performance profiling for dynamics simulations.

This module provides :class:`ProfilerHook`, a controller that creates
lightweight probe hooks to measure timing intervals between arbitrary
hook stages. Supports both NVTX ranges for Nsight Systems profiling
and wall-clock timing via CUDA events or CPU perf counters.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Literal, Sequence

import torch

from nvalchemi.dynamics.base import HookStageEnum

if TYPE_CHECKING:
    from nvalchemi.data import Batch
    from nvalchemi.dynamics.base import BaseDynamics

try:
    import nvtx
except ImportError:
    nvtx = None

__all__ = ["ProfilerHook"]


class _Probe:
    """Internal hook that records a timestamp at a single stage."""

    def __init__(
        self, stage: HookStageEnum, frequency: int, parent: ProfilerHook
    ) -> None:
        self.stage = stage
        self.frequency = frequency
        self._parent = parent

    @torch.compiler.disable
    def __call__(self, batch: Batch, dynamics: BaseDynamics) -> None:
        self._parent._record(self.stage, dynamics.step_count, batch.device)


class ProfilerHook:
    """Interval-based profiler for dynamics simulations.

    This class is a **controller**, not a hook itself. It creates lightweight
    ``_Probe`` objects that satisfy the ``Hook`` protocol. Pass ``profiler.hooks``
    when constructing dynamics to register the probes.

    Each interval is defined by a label, a start stage, and an end stage.
    The profiler measures the elapsed time between these stages and can
    also emit NVTX ranges for visualization in NVIDIA Nsight Systems.

    Parameters
    ----------
    intervals : Sequence[tuple[str, HookStageEnum, HookStageEnum]]
        Each tuple is ``(label, start_stage, end_stage)``. The profiler
        will measure the time from ``start_stage`` to ``end_stage`` and
        report it under ``label``.
    enable_nvtx : bool, optional
        Whether to emit NVTX ranges for Nsight profiling. Default ``True``.
    enable_timer : bool, optional
        Whether to record wall-clock timing. Default ``True``.
    timer_backend : {"cuda_event", "perf_counter", "auto"}, optional
        Timing backend. ``"cuda_event"`` uses GPU-synchronized events,
        ``"perf_counter"`` uses ``time.perf_counter_ns``, and ``"auto"``
        selects based on device type. Default ``"auto"``.
    frequency : int, optional
        Profile every ``frequency`` steps. Default ``1``.

    Attributes
    ----------
    enable_nvtx : bool
        Whether NVTX annotation is active.
    enable_timer : bool
        Whether step timing is active.
    timer_backend : str
        The configured timing backend.
    frequency : int
        Profiling frequency in steps.

    Examples
    --------
    >>> from nvalchemi.dynamics.hooks import ProfilerHook
    >>> profiler = ProfilerHook.step_timer()
    >>> dynamics = DemoDynamics(model=model, n_steps=100, dt=0.5, hooks=profiler.hooks)
    >>> dynamics.run(batch)
    >>> print(profiler.summary())
    """

    def __init__(
        self,
        intervals: Sequence[tuple[str, HookStageEnum, HookStageEnum]],
        enable_nvtx: bool = True,
        enable_timer: bool = True,
        timer_backend: Literal["cuda_event", "perf_counter", "auto"] = "auto",
        frequency: int = 1,
    ) -> None:
        self.enable_nvtx = enable_nvtx
        self.enable_timer = enable_timer
        self.timer_backend = timer_backend
        self.frequency = frequency
        self._intervals = list(intervals)

        labels = [label for label, _, _ in self._intervals]
        if len(labels) != len(set(labels)):
            raise ValueError("Duplicate interval labels are not allowed.")

        for label, start, end in self._intervals:
            if start.value >= end.value:
                raise ValueError(
                    f"Interval '{label}': start stage {start.name} must come "
                    f"before end stage {end.name} in execution order."
                )

        unique_stages: set[HookStageEnum] = set()
        for _, start, end in self._intervals:
            unique_stages.add(start)
            unique_stages.add(end)

        self._probes = [_Probe(stage, frequency, self) for stage in unique_stages]

        self._cuda_events: dict[HookStageEnum, list[torch.cuda.Event]] = {
            stage: [] for stage in unique_stages
        }
        self._cpu_timestamps: dict[HookStageEnum, list[int]] = {
            stage: [] for stage in unique_stages
        }
        self._backend_resolved: str | None = None

    @property
    def hooks(self) -> list[_Probe]:
        """Return probe hooks for registration with dynamics."""
        return list(self._probes)

    def _resolve_backend(self, device: torch.device) -> str:
        """Resolve the timing backend based on configuration and device."""
        if self.timer_backend != "auto":
            return self.timer_backend
        if device.type == "cuda":
            return "cuda_event"
        return "perf_counter"

    def _record(self, stage: HookStageEnum, step: int, device: torch.device) -> None:
        """Record timing and NVTX events for a stage."""
        if self.enable_nvtx and nvtx is not None:
            for label, start, end in self._intervals:
                if stage == start:
                    nvtx.push_range(f"{label}/{step}")
                elif stage == end:
                    nvtx.pop_range()

        if self.enable_timer:
            if self._backend_resolved is None:
                self._backend_resolved = self._resolve_backend(device)
            if self._backend_resolved == "cuda_event":
                event = torch.cuda.Event(enable_timing=True)
                event.record()
                self._cuda_events[stage].append(event)
            else:
                self._cpu_timestamps[stage].append(time.perf_counter_ns())

    def summary(self) -> dict[str, dict[str, float]]:
        """Return timing statistics per interval.

        For CUDA event timing, this method synchronizes the device once
        before resolving event pairs. Call after the simulation run completes.

        Returns
        -------
        dict[str, dict[str, float]]
            Mapping from interval label to stats dict with keys:
            ``mean_s``, ``std_s``, ``min_s``, ``max_s``, ``total_s``, ``n_samples``.
        """
        result: dict[str, dict[str, float]] = {}

        if not self.enable_timer:
            return result

        backend = self._backend_resolved
        if backend == "cuda_event":
            torch.cuda.synchronize()

        for label, start, end in self._intervals:
            if backend == "cuda_event":
                start_events = self._cuda_events[start]
                end_events = self._cuda_events[end]
                n = min(len(start_events), len(end_events))
                if n == 0:
                    continue
                deltas = torch.tensor(
                    [
                        start_events[i].elapsed_time(end_events[i]) / 1000.0
                        for i in range(n)
                    ]
                )
            else:
                start_ts = self._cpu_timestamps[start]
                end_ts = self._cpu_timestamps[end]
                n = min(len(start_ts), len(end_ts))
                if n == 0:
                    continue
                deltas = torch.tensor(
                    [(end_ts[i] - start_ts[i]) / 1e9 for i in range(n)]
                )

            result[label] = {
                "mean_s": deltas.mean().item(),
                "std_s": deltas.std().item() if n > 1 else 0.0,
                "min_s": deltas.min().item(),
                "max_s": deltas.max().item(),
                "total_s": deltas.sum().item(),
                "n_samples": n,
            }
        return result

    def reset(self) -> None:
        """Clear all accumulated timing data."""
        for stage in self._cuda_events:
            self._cuda_events[stage].clear()
        for stage in self._cpu_timestamps:
            self._cpu_timestamps[stage].clear()
        self._backend_resolved = None

    @classmethod
    def step_timer(cls, **kwargs) -> ProfilerHook:
        """Create a profiler that measures full step duration."""
        S = HookStageEnum
        return cls(intervals=[("step", S.BEFORE_STEP, S.AFTER_STEP)], **kwargs)

    @classmethod
    def detailed(cls, **kwargs) -> ProfilerHook:
        """Create a profiler measuring step, pre_update, compute, and post_update."""
        S = HookStageEnum
        return cls(
            intervals=[
                ("step", S.BEFORE_STEP, S.AFTER_STEP),
                ("pre_update", S.BEFORE_PRE_UPDATE, S.AFTER_PRE_UPDATE),
                ("compute", S.BEFORE_COMPUTE, S.AFTER_COMPUTE),
                ("post_update", S.BEFORE_POST_UPDATE, S.AFTER_POST_UPDATE),
            ],
            **kwargs,
        )
