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
"""Generic capacity planning and queue construction for simulation campaigns."""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import torch

if TYPE_CHECKING:
    from nvalchemi.data import Batch

__all__ = [
    "BatchMemoryEstimate",
    "BatchMeasurement",
    "RunAssignment",
    "SimulationBatchPlanner",
]


def _is_out_of_memory(error: BaseException | None) -> bool:
    """Return whether ``error`` is, or wraps, a CUDA out-of-memory error.

    An OOM raised while ``torch.compile`` lowers a graph (e.g. inductor's
    pad_mm pass benchmarking real padded operands) surfaces as
    ``torch._dynamo.exc.BackendCompilerFailed``, which carries the original
    error on ``inner_exception`` rather than as a ``torch.cuda.OutOfMemoryError``.
    """
    seen: set[int] = set()
    while error is not None and id(error) not in seen:
        if isinstance(error, torch.cuda.OutOfMemoryError):
            return True
        seen.add(id(error))
        error = (
            getattr(error, "inner_exception", None)
            or error.__cause__
            or error.__context__
        )
    return False


@dataclass(frozen=True)
class BatchMemoryEstimate:
    """Conservative memory model inferred from successful profile points."""

    model_resident_bytes: int
    bytes_per_walker: int


@dataclass(frozen=True)
class BatchMeasurement:
    """Measured throughput and memory at one independent-run batch width."""

    batch_width: int
    status: Literal["ok", "oom"]
    walker_blocks_per_second: float | None = None
    peak_reserved_bytes: int | None = None
    atoms_per_walker: int | None = None
    error: str | None = None


@dataclass(frozen=True)
class RunAssignment:
    """One contiguous independent-run range assigned to one GPU and wave."""

    gpu_id: int
    wave: int
    start: int
    stop: int

    @property
    def count(self) -> int:
        """Return the number of independent runs in this assignment."""
        return self.stop - self.start


class SimulationBatchPlanner:
    """Select a memory-safe batch width and schedule independent run waves.

    A measurement sweep is preferred over a model-independent memory formula:
    MLIP activation memory depends on the selected model, atom count, neighbor
    graph, simulation method, precision, and compilation mode. The static
    estimate is a conservative pre-run bound once a previous profile supplies
    model-resident and per-run memory.
    """

    def __init__(
        self,
        memory_fraction: float = 0.85,
        throughput_fraction: float = 0.95,
    ) -> None:
        """Initialize capacity and throughput selection policy."""
        if not 0.0 < memory_fraction <= 1.0:
            raise ValueError("memory_fraction must lie in (0, 1]")
        if not 0.0 < throughput_fraction <= 1.0:
            raise ValueError("throughput_fraction must lie in (0, 1]")
        self.memory_fraction = memory_fraction
        self.throughput_fraction = throughput_fraction

    def estimate_width(
        self,
        *,
        total_memory_bytes: int,
        model_resident_bytes: int,
        bytes_per_walker: int,
    ) -> int:
        """Estimate a safe width from a previously measured memory profile."""
        if total_memory_bytes < 1 or model_resident_bytes < 0 or bytes_per_walker < 1:
            raise ValueError(
                "memory arguments must be non-negative, with positive total and per-run memory"
            )
        usable = int(total_memory_bytes * self.memory_fraction) - model_resident_bytes
        if usable < bytes_per_walker:
            return 0
        return usable // bytes_per_walker

    def profile(
        self,
        workload_factory: Callable[[int, torch.device], tuple[Any, Batch]],
        widths: Sequence[int],
        *,
        device: torch.device | str = "cuda",
        warmup_blocks: int = 1,
        measured_blocks: int = 1,
        stop_on_oom: bool = True,
    ) -> list[BatchMeasurement]:
        """Measure simulation capacity and throughput over candidate widths.

        The factory returns a fresh ``(runner, batch)`` pair that represents
        the intended production workload. The runner contract is
        ``runner.run(batch, n_blocks=...)``; adapters can provide that contract
        for MC, MD, optimization, or hybrid simulation stages.

        ``peak_reserved_bytes`` spans warmup as well as the measured blocks, so
        a compiled model's first-call compilation peak (which a fresh
        production run pays again) counts against the memory budget. An OOM,
        including one raised inside ``torch.compile``, is recorded as
        ``status="oom"``; with ``stop_on_oom`` the sweep ends there, since
        larger widths in an ascending ladder cannot fit either.
        """
        if not widths or any(width < 1 for width in widths):
            raise ValueError("widths must contain positive batch widths")
        if warmup_blocks < 0 or measured_blocks < 1:
            raise ValueError(
                "warmup_blocks must be non-negative and measured_blocks positive"
            )
        resolved_device = torch.device(device)
        if resolved_device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError(
                "simulation batch profiling requires an available CUDA GPU"
            )

        measurements: list[BatchMeasurement] = []
        for width in widths:
            runner = None
            batch = None
            try:
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(resolved_device)
                runner, batch = workload_factory(width, resolved_device)
                if warmup_blocks:
                    runner.run(batch, n_blocks=warmup_blocks)
                torch.cuda.synchronize(resolved_device)
                start = time.perf_counter()
                runner.run(batch, n_blocks=measured_blocks)
                torch.cuda.synchronize(resolved_device)
                wall_seconds = time.perf_counter() - start
                measurements.append(
                    BatchMeasurement(
                        batch_width=width,
                        status="ok",
                        walker_blocks_per_second=width * measured_blocks / wall_seconds,
                        peak_reserved_bytes=torch.cuda.max_memory_reserved(
                            resolved_device
                        ),
                        atoms_per_walker=batch.num_nodes // width,
                    )
                )
            except Exception as error:
                if not _is_out_of_memory(error):
                    raise
                measurements.append(
                    BatchMeasurement(batch_width=width, status="oom", error=str(error))
                )
                if stop_on_oom:
                    break
            finally:
                del runner, batch
                torch.cuda.empty_cache()
        return measurements

    def recommend_width(
        self,
        measurements: Sequence[BatchMeasurement],
        *,
        total_memory_bytes: int,
    ) -> int:
        """Return the smallest efficient width satisfying the memory policy."""
        memory_limit = int(total_memory_bytes * self.memory_fraction)
        eligible = [
            measurement
            for measurement in measurements
            if measurement.status == "ok"
            and measurement.peak_reserved_bytes is not None
            and measurement.peak_reserved_bytes <= memory_limit
            and measurement.walker_blocks_per_second is not None
        ]
        if not eligible:
            raise RuntimeError(
                "no measured batch width satisfies the configured memory limit"
            )
        threshold = self.throughput_fraction * max(
            measurement.walker_blocks_per_second for measurement in eligible
        )
        return min(
            measurement.batch_width
            for measurement in eligible
            if measurement.walker_blocks_per_second >= threshold
        )

    @staticmethod
    def infer_memory_model(
        measurements: Sequence[BatchMeasurement],
    ) -> BatchMemoryEstimate:
        """Infer a conservative linear memory model from a profile sweep."""
        points = sorted(
            (
                (measurement.batch_width, measurement.peak_reserved_bytes)
                for measurement in measurements
                if measurement.status == "ok"
                and measurement.peak_reserved_bytes is not None
            ),
        )
        if len(points) < 2:
            raise ValueError("at least two successful profile points are required")
        if len({width for width, _ in points}) != len(points):
            raise ValueError("measurements must have unique batch widths")

        incremental_bytes = max(
            (right_memory - left_memory) / (right_width - left_width)
            for (left_width, left_memory), (right_width, right_memory) in zip(
                points, points[1:]
            )
        )
        incremental_bytes = max(1, math.ceil(incremental_bytes))
        resident_bytes = max(
            0,
            max(memory - incremental_bytes * width for width, memory in points),
        )
        return BatchMemoryEstimate(
            model_resident_bytes=resident_bytes,
            bytes_per_walker=incremental_bytes,
        )

    @staticmethod
    def assign_runs(
        total_runs: int,
        batch_width: int,
        gpu_ids: Sequence[int],
    ) -> list[RunAssignment]:
        """Pack independent runs into active batches and serial GPU waves."""
        if total_runs < 1 or batch_width < 1:
            raise ValueError("total_runs and batch_width must be positive")
        if not gpu_ids or len(set(gpu_ids)) != len(gpu_ids):
            raise ValueError("gpu_ids must contain at least one unique GPU id")
        assignments: list[RunAssignment] = []
        start = 0
        wave = 0
        while start < total_runs:
            for gpu_id in gpu_ids:
                if start == total_runs:
                    break
                stop = min(start + batch_width, total_runs)
                assignments.append(
                    RunAssignment(gpu_id=gpu_id, wave=wave, start=start, stop=stop)
                )
                start = stop
            wave += 1
        return assignments
