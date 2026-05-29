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
"""Tests for the nvalchemi I/O benchmark CLI helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from nvalchemi.data.io_test import _make_atomic_data, _run_benchmark


def test_make_atomic_data_generates_edge_rows() -> None:
    """Generated edge tensors use edge-major row layout."""
    data = _make_atomic_data(num_atoms=4, num_edges=7)

    assert data.neighbor_list.shape == (7, 2)
    assert data.shifts.shape == (7, 3)


def test_run_benchmark_profiles_readback(tmp_path: Path) -> None:
    """Benchmark results include a timed full-store readback."""
    results = _run_benchmark(
        num_systems_list=[2],
        min_atoms=3,
        max_atoms=4,
        seed=42,
        config=None,
        store_dir=tmp_path,
    )

    result = results[0]
    assert result["read_bytes"] >= result["raw_bytes"]
    assert result["read_time"] >= 0
    assert result["profile_time"] == pytest.approx(
        result["write_time"] + result["read_time"]
    )
    assert result["read_throughput"] >= 0
    assert result["profile_throughput"] >= 0
