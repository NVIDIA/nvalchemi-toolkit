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

from nvalchemi.data.io_test import (
    _build_read_indices,
    _expand_read_modes,
    _make_atomic_data,
    _run_benchmark,
    _run_read_benchmark,
)


def test_expand_read_modes_defaults_to_batch() -> None:
    """No explicit read mode uses the batch readback fast path."""
    assert _expand_read_modes(()) == ("batch",)


def test_expand_read_modes_supports_both() -> None:
    """The convenience mode expands into batch and single readback paths."""
    assert _expand_read_modes(("both",)) == ("batch", "single")


def test_build_read_indices_supports_sequential_order() -> None:
    """Sequential read order preserves logical storage order."""
    assert _build_read_indices(5, "sequential", seed=123, read_order_block_size=2) == [
        0,
        1,
        2,
        3,
        4,
    ]


def test_build_read_indices_supports_full_shuffle() -> None:
    """Shuffle read order randomizes individual sample indices."""
    indices = _build_read_indices(8, "shuffle", seed=123, read_order_block_size=4)

    assert sorted(indices) == list(range(8))
    assert indices != list(range(8))


def test_build_read_indices_supports_block_shuffle() -> None:
    """Block shuffle preserves locality inside shuffled contiguous blocks."""
    indices = _build_read_indices(8, "block-shuffle", seed=123, read_order_block_size=2)
    blocks = [indices[start : start + 2] for start in range(0, len(indices), 2)]

    assert sorted(indices) == list(range(8))
    assert indices != list(range(8))
    assert all(block[1] == block[0] + 1 for block in blocks)


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
    assert result["read_mode"] == "batch"
    assert result["read_order"] == "sequential"
    assert result["batch_size"] == 64
    assert result["prefetch_factor"] == 16
    assert result["effective_read_window"] == 1024
    assert result["read_bytes"] >= result["raw_bytes"]
    assert result["read_time"] >= 0
    assert result["profile_time"] == pytest.approx(
        result["write_time"] + result["read_time"]
    )
    assert result["read_throughput"] >= 0
    assert result["profile_throughput"] >= 0


def test_run_benchmark_can_compare_batch_and_single_readback(tmp_path: Path) -> None:
    """Benchmark can report batch and single-sample readback rows."""
    results = _run_benchmark(
        num_systems_list=[2],
        min_atoms=3,
        max_atoms=4,
        seed=42,
        config=None,
        store_dir=tmp_path,
        read_modes=("batch", "single"),
        batch_size=2,
        prefetch_factor=3,
        read_order="shuffle",
        read_seed=123,
    )

    assert [result["read_mode"] for result in results] == ["batch", "single"]
    assert [result["read_order"] for result in results] == ["shuffle", "shuffle"]
    assert [result["batch_size"] for result in results] == [2, 1]
    assert [result["prefetch_factor"] for result in results] == [3, 0]
    assert [result["effective_read_window"] for result in results] == [6, 1]
    assert {result["num_systems"] for result in results} == {2}


def test_run_benchmark_records_block_shuffle_settings(tmp_path: Path) -> None:
    """Benchmark rows record block-shuffle readback settings."""
    results = _run_benchmark(
        num_systems_list=[4],
        min_atoms=3,
        max_atoms=4,
        seed=42,
        config=None,
        store_dir=tmp_path,
        read_order="block-shuffle",
        read_seed=123,
        read_order_block_size=2,
        batch_size=2,
        prefetch_factor=2,
    )

    result = results[0]
    assert result["read_order"] == "block-shuffle"
    assert result["read_order_block_size"] == 2


@pytest.fixture()
def small_zarr_store(tmp_path: Path) -> Path:
    """Write a 4-system Zarr store for read-only benchmarking."""
    from nvalchemi.data.datapipes.backends.zarr import AtomicDataZarrWriter

    store_path = tmp_path / "small.zarr"
    data_list = [_make_atomic_data(num_atoms=5, num_edges=8) for _ in range(4)]
    writer = AtomicDataZarrWriter(store_path)
    writer.write(data_list)
    return store_path


def test_run_read_benchmark_reads_existing_store(small_zarr_store: Path) -> None:
    """Read benchmark discovers sample count and reports read throughput."""
    results = _run_read_benchmark(store_path=small_zarr_store)

    assert len(results) == 1
    result = results[0]
    assert result["num_systems"] == 4
    assert result["read_mode"] == "batch"
    assert result["read_order"] == "sequential"
    assert result["read_time"] >= 0
    assert result["read_bytes"] > 0
    assert result["read_throughput"] >= 0
    assert result["store_path"] == str(small_zarr_store)


def test_run_read_benchmark_supports_shuffle(small_zarr_store: Path) -> None:
    """Read benchmark works with shuffled access order."""
    results = _run_read_benchmark(
        store_path=small_zarr_store,
        read_order="shuffle",
        read_seed=42,
    )

    result = results[0]
    assert result["read_order"] == "shuffle"
    assert result["read_order_block_size"] is None
    assert result["read_bytes"] > 0


def test_run_read_benchmark_compares_batch_and_single(small_zarr_store: Path) -> None:
    """Read benchmark can report both batch and single readback modes."""
    results = _run_read_benchmark(
        store_path=small_zarr_store,
        read_modes=("batch", "single"),
        batch_size=2,
        prefetch_factor=3,
    )

    assert [r["read_mode"] for r in results] == ["batch", "single"]
    assert [r["batch_size"] for r in results] == [2, 1]
    assert [r["prefetch_factor"] for r in results] == [3, 0]
    assert [r["effective_read_window"] for r in results] == [6, 1]
    assert all(r["num_systems"] == 4 for r in results)
    assert all(r["read_bytes"] > 0 for r in results)
