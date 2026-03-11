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
Per-stage wall-clock profiling for dynamics simulations.

Provides :class:`ProfilerHook`, a single hook that registers at multiple
stages and records the elapsed time between consecutive stages at each
step.  Supports NVTX range annotations for Nsight Systems, CSV logging,
and formatted console output via ``loguru``.
"""

from __future__ import annotations

import csv
import io
import statistics
import time
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import torch
from loguru import logger

from nvalchemi.dynamics.base import HookStageEnum

if TYPE_CHECKING:
    from nvalchemi.data import Batch
    from nvalchemi.dynamics.base import BaseDynamics

try:
    import nvtx
except ImportError:
    nvtx = None

__all__ = ["ProfilerHook"]

# HookStageEnum members in execution order.  The numeric values of the
# enum are monotonically increasing with execution order.
_STAGE_ORDER: list[HookStageEnum] = sorted(HookStageEnum, key=lambda s: s.value)


class ProfilerHook:
    """Per-stage timing hook for dynamics simulations.

    A single ``ProfilerHook`` instance registers itself at every
    requested stage.  On each call it records a timestamp; when the
    last profiled stage in a step fires, it computes the elapsed time
    between consecutive stages and (optionally) writes to CSV / console.

    The hook uses ``stages`` (plural) so that
    :meth:`~nvalchemi.dynamics.base.BaseDynamics.register_hook`
    registers it at all listed stages in one call.

    Parameters
    ----------
    stages : set[HookStageEnum] | {"all", "step", "detailed"}
        Which stages to instrument.

        * ``"all"`` (default): every stage except ``ON_CONVERGE``.
        * ``"step"``: ``BEFORE_STEP`` and ``AFTER_STEP`` only.
        * ``"detailed"``: all stages from ``BEFORE_STEP`` through
          ``AFTER_STEP`` (excluding ``ON_CONVERGE``).
        * A custom ``set[HookStageEnum]`` for fine-grained control.
    frequency : int, optional
        Profile every ``frequency`` steps.  Default ``1``.
    enable_nvtx : bool, optional
        Emit NVTX push/pop ranges for Nsight Systems.  Default ``True``.
    timer_backend : {"cuda_event", "perf_counter", "auto"}, optional
        Timing backend.  ``"auto"`` selects ``cuda_event`` on GPU
        devices and ``perf_counter`` on CPU.  Default ``"auto"``.
    log_path : str | Path | None, optional
        Path to a CSV file for persistent timing logs.  Each row
        records the rank, step, stage transition, wall-clock offset,
        and delta.  Default ``None`` (no file).
    show_console : bool, optional
        Print a formatted timing table via ``loguru`` at each
        profiled step.  Default ``False``.
    console_frequency : int, optional
        When ``show_console`` is ``True``, print every
        ``console_frequency`` profiled steps.  Default ``1``.

    Attributes
    ----------
    stages : list[HookStageEnum]
        Profiled stages in execution order (used by ``register_hook``).
    frequency : int
        Execution frequency in steps.
    timings : dict[HookStageEnum, list[float]]
        Accumulated per-transition timing data (seconds).

    Examples
    --------
    >>> from nvalchemi.dynamics.hooks import ProfilerHook
    >>> profiler = ProfilerHook()
    >>> dynamics = DemoDynamics(model=model, n_steps=100, dt=0.5, hooks=[profiler])
    >>> dynamics.run(batch)
    >>> print(profiler.summary())

    With CSV logging and console output:

    >>> profiler = ProfilerHook(
    ...     "detailed",
    ...     log_path="profiler.csv",
    ...     show_console=True,
    ...     console_frequency=10,
    ... )
    >>> dynamics = DemoDynamics(model=model, n_steps=1000, dt=0.5, hooks=[profiler])
    >>> dynamics.run(batch)
    """

    def __init__(
        self,
        stages: set[HookStageEnum] | Literal["all", "step", "detailed"] = "all",
        *,
        frequency: int = 1,
        enable_nvtx: bool = True,
        timer_backend: Literal["cuda_event", "perf_counter", "auto"] = "auto",
        log_path: str | Path | None = None,
        show_console: bool = False,
        console_frequency: int = 1,
    ) -> None:
        # Init file handle early so __del__ is safe on validation errors.
        self._csv_file: io.TextIOWrapper | None = None
        self._csv_writer: csv.DictWriter | None = None

        S = HookStageEnum
        if isinstance(stages, str):
            if stages == "all":
                resolved = {s for s in S if s != S.ON_CONVERGE}
            elif stages == "step":
                resolved = {S.BEFORE_STEP, S.AFTER_STEP}
            elif stages == "detailed":
                resolved = {
                    S.BEFORE_STEP,
                    S.BEFORE_PRE_UPDATE,
                    S.AFTER_PRE_UPDATE,
                    S.BEFORE_COMPUTE,
                    S.AFTER_COMPUTE,
                    S.BEFORE_POST_UPDATE,
                    S.AFTER_POST_UPDATE,
                    S.AFTER_STEP,
                }
            else:
                raise ValueError(
                    f"Unknown stages preset {stages!r}. "
                    f"Use 'all', 'step', 'detailed', or a set of HookStageEnum."
                )
        else:
            resolved = set(stages)

        if len(resolved) < 2:
            raise ValueError(
                "At least two stages are required to measure timing deltas."
            )

        # Sorted by execution order — used by register_hook.
        self.stages: list[HookStageEnum] = [s for s in _STAGE_ORDER if s in resolved]
        self.frequency = frequency
        self.enable_nvtx = enable_nvtx
        self.timer_backend = timer_backend
        self.log_path = Path(log_path) if log_path is not None else None
        self.show_console = show_console
        self.console_frequency = console_frequency

        # Per-step scratch — separate dicts for type safety.
        self._current_step: int = -1
        self._step_cuda_events: dict[HookStageEnum, torch.cuda.Event] = {}
        self._step_cpu_timestamps: dict[HookStageEnum, int] = {}

        # Accumulated timing: transition endpoint -> list of delta_s.
        self.timings: dict[HookStageEnum, list[float]] = {s: [] for s in self.stages}

        self._t0_ns: int = time.perf_counter_ns()
        self._backend_resolved: str | None = None
        self._steps_recorded: int = 0

    # ------------------------------------------------------------------
    # Hook entry point
    # ------------------------------------------------------------------

    @torch.compiler.disable
    def __call__(self, batch: Batch, dynamics: BaseDynamics) -> None:
        """Record a timestamp for the current stage.

        Parameters
        ----------
        batch : Batch
            The current batch of atomic data.
        dynamics : BaseDynamics
            The dynamics engine instance.
        """
        stage: HookStageEnum = dynamics.current_hook_stage  # type: ignore[assignment]
        step = dynamics.step_count

        # New step: flush the previous one, then reset scratch.
        if step != self._current_step:
            if self._current_step >= 0:
                self._flush_step(dynamics.global_rank)
            self._current_step = step
            self._step_cuda_events.clear()
            self._step_cpu_timestamps.clear()

        # NVTX annotation.
        if self.enable_nvtx and nvtx is not None:
            idx = self.stages.index(stage)
            if idx > 0:
                nvtx.pop_range()
            nvtx.push_range(f"{stage.name}/{step}")

        # Timestamp.
        dev = batch.device
        if isinstance(dev, str):
            dev = torch.device(dev)
        if self._backend_resolved is None:
            self._backend_resolved = self._resolve_backend(dev)
        if self._backend_resolved == "cuda_event":
            event = torch.cuda.Event(enable_timing=True)
            event.record()
            self._step_cuda_events[stage] = event
        else:
            self._step_cpu_timestamps[stage] = time.perf_counter_ns()

        # If this is the last profiled stage in the step, flush now.
        if stage == self.stages[-1]:
            self._flush_step(dynamics.global_rank)
            self._current_step = -1
            self._step_cuda_events.clear()
            self._step_cpu_timestamps.clear()

    # ------------------------------------------------------------------
    # Backend resolution
    # ------------------------------------------------------------------

    def _resolve_backend(self, device: torch.device) -> str:
        """Resolve the timing backend based on configuration and device."""
        if self.timer_backend != "auto":
            return self.timer_backend
        if device.type == "cuda":
            return "cuda_event"
        return "perf_counter"

    # ------------------------------------------------------------------
    # Step flush — compute deltas, log
    # ------------------------------------------------------------------

    def _flush_step(self, rank: int) -> None:
        """Compute per-transition deltas for the current step and log."""
        use_cuda = self._backend_resolved == "cuda_event"

        if use_cuda:
            ordered = [s for s in self.stages if s in self._step_cuda_events]
        else:
            ordered = [s for s in self.stages if s in self._step_cpu_timestamps]

        if len(ordered) < 2:
            return

        if use_cuda:
            torch.cuda.synchronize()

        deltas: dict[HookStageEnum, float] = {}
        for i in range(1, len(ordered)):
            prev_stage, curr_stage = ordered[i - 1], ordered[i]
            if use_cuda:
                prev_ev = self._step_cuda_events[prev_stage]
                curr_ev = self._step_cuda_events[curr_stage]
                delta_s = prev_ev.elapsed_time(curr_ev) / 1000.0
            else:
                prev_ts = self._step_cpu_timestamps[prev_stage]
                curr_ts = self._step_cpu_timestamps[curr_stage]
                delta_s = (curr_ts - prev_ts) / 1e9
            deltas[curr_stage] = delta_s
            self.timings[curr_stage].append(delta_s)

        t_since_init_s = (time.perf_counter_ns() - self._t0_ns) / 1e9
        self._steps_recorded += 1

        if self.log_path is not None:
            self._write_csv(rank, self._current_step, t_since_init_s, ordered, deltas)

        if self.show_console and (self._steps_recorded % self.console_frequency == 0):
            self._print_console(
                rank, self._current_step, t_since_init_s, ordered, deltas
            )

        # Close NVTX range for the last stage in this step.
        if self.enable_nvtx and nvtx is not None:
            nvtx.pop_range()

    # ------------------------------------------------------------------
    # CSV output
    # ------------------------------------------------------------------

    def _write_csv(
        self,
        rank: int,
        step: int,
        t_since_init: float,
        ordered: list[HookStageEnum],
        deltas: dict[HookStageEnum, float],
    ) -> None:
        """Append one row per transition to the CSV log."""
        rows = []
        for i, stage in enumerate(ordered[1:], start=1):
            rows.append(
                {
                    "rank": rank,
                    "step": step,
                    "stage": f"{ordered[i - 1].name}->{stage.name}",
                    "t_since_init_s": f"{t_since_init:.6f}",
                    "delta_s": f"{deltas[stage]:.6f}",
                }
            )
        if self._csv_writer is None:
            log_path = self.log_path
            if log_path is None:
                return
            fh = open(log_path, "w", newline="")  # noqa: SIM115
            self._csv_file = fh
            self._csv_writer = csv.DictWriter(
                fh,
                fieldnames=["rank", "step", "stage", "t_since_init_s", "delta_s"],
            )
            self._csv_writer.writeheader()
        self._csv_writer.writerows(rows)
        if self._csv_file is not None:
            self._csv_file.flush()

    # ------------------------------------------------------------------
    # Console output
    # ------------------------------------------------------------------

    def _print_console(
        self,
        rank: int,
        step: int,
        t_since_init: float,
        ordered: list[HookStageEnum],
        deltas: dict[HookStageEnum, float],
    ) -> None:
        """Print a formatted timing table for the current step."""
        lines = [f"[Profiler] rank={rank}  step={step}  t={t_since_init:.3f}s"]
        for i, stage in enumerate(ordered[1:], start=1):
            prev_name = ordered[i - 1].name
            lines.append(
                f"  {prev_name} -> {stage.name}: {deltas[stage] * 1000:.3f} ms"
            )
        logger.info("\n".join(lines))

    # ------------------------------------------------------------------
    # Summary / reset / close
    # ------------------------------------------------------------------

    def summary(self) -> dict[str, dict[str, float]]:
        """Return per-transition timing statistics.

        Returns
        -------
        dict[str, dict[str, float]]
            Mapping from ``"PREV_STAGE->STAGE"`` label to a stats dict
            with keys ``mean_s``, ``std_s``, ``min_s``, ``max_s``,
            ``total_s``, ``n_samples``.
        """
        result: dict[str, dict[str, float]] = {}
        for idx, stage in enumerate(self.stages):
            samples = self.timings[stage]
            if not samples:
                continue
            prev_name = self.stages[idx - 1].name
            label = f"{prev_name}->{stage.name}"
            n = len(samples)
            result[label] = {
                "mean_s": statistics.mean(samples),
                "std_s": statistics.stdev(samples) if n > 1 else 0.0,
                "min_s": min(samples),
                "max_s": max(samples),
                "total_s": sum(samples),
                "n_samples": float(n),
            }
        return result

    def reset(self) -> None:
        """Clear all accumulated timing data."""
        for stage in self.timings:
            self.timings[stage].clear()
        self._step_cuda_events.clear()
        self._step_cpu_timestamps.clear()
        self._current_step = -1
        self._backend_resolved = None
        self._t0_ns = time.perf_counter_ns()
        self._steps_recorded = 0

    def close(self) -> None:
        """Flush and close the CSV log file, if open."""
        if self._csv_file is not None:
            self._csv_file.close()
            self._csv_file = None
            self._csv_writer = None

    def __del__(self) -> None:
        self.close()
