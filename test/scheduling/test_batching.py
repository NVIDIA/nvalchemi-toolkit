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
"""Tests for simulation batch capacity planning and GPU queue construction."""

from __future__ import annotations

import torch

from nvalchemi.scheduling import BatchMeasurement, SimulationBatchPlanner
from nvalchemi.scheduling.batching import _is_out_of_memory


def test_estimate_width_uses_model_and_per_walker_memory() -> None:
    """The static estimate retains the configured memory headroom."""
    planner = SimulationBatchPlanner(memory_fraction=0.9)

    width = planner.estimate_width(
        total_memory_bytes=40_000,
        model_resident_bytes=4_000,
        bytes_per_walker=2_000,
    )

    assert width == 16


def test_recommend_width_prefers_smallest_near_peak_throughput() -> None:
    """The lowest width meeting the throughput threshold is selected."""
    planner = SimulationBatchPlanner(memory_fraction=0.9, throughput_fraction=0.95)
    measurements = [
        BatchMeasurement(1, "ok", 100.0, 1_000),
        BatchMeasurement(2, "ok", 180.0, 2_000),
        BatchMeasurement(4, "ok", 190.0, 3_000),
        BatchMeasurement(8, "oom"),
    ]

    assert planner.recommend_width(measurements, total_memory_bytes=4_000) == 4


def test_infer_memory_model_can_port_a_profile_to_another_gpu() -> None:
    """A conservative fitted model gives the required static estimate inputs."""
    measurements = [
        BatchMeasurement(1, "ok", 10.0, 1_200),
        BatchMeasurement(2, "ok", 20.0, 1_400),
        BatchMeasurement(4, "ok", 30.0, 1_800),
    ]

    estimate = SimulationBatchPlanner.infer_memory_model(measurements)

    assert estimate.model_resident_bytes == 1_000
    assert estimate.bytes_per_walker == 200


def test_assign_runs_forms_serial_waves_per_gpu() -> None:
    """Excess runs are queued after one full batch per requested GPU."""
    assignments = SimulationBatchPlanner.assign_runs(
        total_runs=19, batch_width=4, gpu_ids=[2, 5]
    )

    assert [
        (item.gpu_id, item.wave, item.start, item.stop) for item in assignments
    ] == [
        (2, 0, 0, 4),
        (5, 0, 4, 8),
        (2, 1, 8, 12),
        (5, 1, 12, 16),
        (2, 2, 16, 19),
    ]


def test_compile_wrapped_oom_is_classified_as_oom() -> None:
    """An OOM raised inside torch.compile arrives wrapped, not as OutOfMemoryError."""

    class BackendCompilerFailed(Exception):  # mirrors torch._dynamo.exc's shape
        def __init__(self, inner: BaseException) -> None:
            super().__init__(f"backend='inductor' raised:\n{inner}")
            self.inner_exception = inner

    oom = torch.cuda.OutOfMemoryError(
        "CUDA out of memory. Tried to allocate 19.78 GiB."
    )
    assert _is_out_of_memory(oom)
    assert _is_out_of_memory(BackendCompilerFailed(oom))
    try:
        try:
            raise oom
        except torch.cuda.OutOfMemoryError as inner:
            raise RuntimeError("wrapped") from inner
    except RuntimeError as chained:
        assert _is_out_of_memory(chained)
    assert not _is_out_of_memory(BackendCompilerFailed(ValueError("bad graph")))
