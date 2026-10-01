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
"""Batched descriptor geometry and public scoring parity checks."""

from __future__ import annotations

import itertools
import math

import pytest
import torch
from nvalchemiops.neighbors import estimate_max_neighbors

from nvalchemi.csp._comparison import descriptor as descriptor_module  # noqa: E402
from nvalchemi.csp._comparison.store import (  # noqa: E402
    DescriptorStore,
    available_cuda_bytes,
    build_descriptor_stores,
)
from nvalchemi.csp.comparison import RadialComparisonIndex  # noqa: E402
from nvalchemi.data import AtomicData, Batch  # noqa: E402

_TYPE_PAD = torch.iinfo(torch.int32).min
_CUTOFF = 1.1


@pytest.mark.parametrize(
    ("driver_free", "total", "reserved", "allocated", "expected"),
    [
        (100, 1000, 300, 250, 150),
        (950, 1000, 200, 100, 1000),
        (100, 1000, 50, 80, 100),
    ],
)
def test_available_cuda_bytes_includes_reusable_allocator_memory(
    monkeypatch,
    driver_free: int,
    total: int,
    reserved: int,
    allocated: int,
    expected: int,
) -> None:
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (driver_free, total))
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda device: reserved)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda device: allocated)

    assert available_cuda_bytes(torch.device("cuda:0")) == expected


def _fixtures() -> tuple[list[torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
    positions = [
        torch.tensor([[0.0, 0, 0], [0.45, 0, 0]], dtype=torch.float32),
        torch.tensor([[0.0, 0, 0], [0.5, 0, 0], [1.05, 0, 0]], dtype=torch.float32),
        torch.tensor([[0.0, 0, 0]], dtype=torch.float32),
    ]
    cells = torch.tensor(
        [
            [[1.0, 0, 0], [0, 4.0, 0], [0, 0, 5.0]],
            [[3.2, 0, 0], [0, 2.7, 0], [0, 0, 4.0]],
            [[4.0, 0, 0], [0, 0.8, 0], [0, 0, 4.0]],
        ],
        dtype=torch.float32,
    )
    pbc = torch.tensor(
        [[True, False, False], [False, False, False], [False, True, False]],
        dtype=torch.bool,
    )
    atom_types = torch.tensor([1, 2, 1, 2, 1, 2], dtype=torch.int32)
    return positions, cells, pbc, atom_types


def _batch(
    positions: list[torch.Tensor], cells: torch.Tensor, pbc: torch.Tensor
) -> Batch:
    return Batch.from_data_list(
        [
            AtomicData(
                positions=values,
                atomic_numbers=torch.ones(len(values), dtype=torch.int64),
                cell=cells[index : index + 1],
                pbc=pbc[index : index + 1],
            )
            for index, values in enumerate(positions)
        ]
    )


def _explicit_rows(
    positions: torch.Tensor,
    cell: torch.Tensor,
    pbc: torch.Tensor,
    atom_types: torch.Tensor,
    cutoff: float,
) -> list[list[tuple[float, int]]]:
    """Enumerate periodic images independently in float64 for this fixture."""
    positions64 = positions.to(torch.float64)
    cell64 = cell.to(torch.float64)
    if bool(pbc.any()):
        max_displacement = max(
            (
                float(torch.linalg.vector_norm(positions64[i] - positions64[j]))
                for i in range(len(positions64))
                for j in range(len(positions64))
            ),
            default=0.0,
        )
        image_bound = math.ceil(
            (cutoff + max_displacement) / float(torch.linalg.svdvals(cell64).min())
        )
    else:
        image_bound = 0
    ranges = [
        range(-image_bound, image_bound + 1) if bool(periodic) else range(1)
        for periodic in pbc
    ]
    shifts = list(itertools.product(*ranges))
    rows = []
    for i in range(len(positions64)):
        row = []
        for j in range(len(positions64)):
            for shift in shifts:
                if i == j and shift == (0, 0, 0):
                    continue
                shift_vector = torch.tensor(shift, dtype=torch.float64) @ cell64
                distance = float(
                    torch.linalg.vector_norm(
                        positions64[i] - positions64[j] + shift_vector
                    )
                )
                if distance < cutoff:
                    row.append((distance, int(atom_types[j])))
        rows.append(sorted(row))
    return rows


def _rows_from_store(
    store: DescriptorStore, structure_id: int
) -> tuple[torch.Tensor, torch.Tensor]:
    row_start = int(store.row_offsets[structure_id])
    row_stop = int(store.row_offsets[structure_id + 1])
    atom_start = int(store.atom_offsets[structure_id])
    atom_stop = int(store.atom_offsets[structure_id + 1])
    width = int(store.widths[structure_id])
    count = atom_stop - atom_start
    return (
        store.distances[row_start:row_stop].reshape(count, width).cpu(),
        store.neighbor_types[row_start:row_stop].reshape(count, width).cpu(),
    )


def _oracle_score(
    rows_a: list[list[tuple[float, int]]],
    rows_b: list[list[tuple[float, int]]],
    types_a: torch.Tensor | None,
    types_b: torch.Tensor | None,
    cutoff: float,
    typed_neighbors: bool,
) -> float:
    def mismatch(
        row_a: list[tuple[float, int]], row_b: list[tuple[float, int]]
    ) -> float:
        if typed_neighbors:
            groups = set(type_id for _, type_id in row_a) | set(
                type_id for _, type_id in row_b
            )
            distances_a = {type_id: [] for type_id in groups}
            distances_b = {type_id: [] for type_id in groups}
            for distance, type_id in row_a:
                distances_a[type_id].append(distance)
            for distance, type_id in row_b:
                distances_b[type_id].append(distance)
            log_mismatch = 0.0
            for type_id in groups:
                left = sorted(distances_a[type_id])
                right = sorted(distances_b[type_id])
                width = max(len(left), len(right))
                left.extend([cutoff] * (width - len(left)))
                right.extend([cutoff] * (width - len(right)))
                log_mismatch = max(
                    log_mismatch,
                    max(
                        (
                            abs(math.log(x) - math.log(y))
                            for x, y in zip(left, right, strict=True)
                        ),
                        default=0.0,
                    ),
                )
            return log_mismatch

        left = sorted(distance for distance, _ in row_a)
        right = sorted(distance for distance, _ in row_b)
        width = max(len(left), len(right))
        left.extend([cutoff] * (width - len(left)))
        right.extend([cutoff] * (width - len(right)))
        return max(
            (abs(math.log(x) - math.log(y)) for x, y in zip(left, right, strict=True)),
            default=0.0,
        )

    def directed(
        source_rows: list[list[tuple[float, int]]],
        source_types: torch.Tensor | None,
        target_rows: list[list[tuple[float, int]]],
        target_types: torch.Tensor | None,
    ) -> float:
        worst = 0.0
        for row_index, source_row in enumerate(source_rows):
            if source_types is None:
                candidates = target_rows
            else:
                center_type = int(source_types[row_index])
                candidates = [
                    target_row
                    for target_index, target_row in enumerate(target_rows)
                    if int(target_types[target_index]) == center_type
                ]
            if not candidates:
                return math.inf
            worst = max(
                worst,
                min(mismatch(source_row, candidate) for candidate in candidates),
            )
        return worst

    log_score = max(
        directed(rows_a, types_a, rows_b, types_b),
        directed(rows_b, types_b, rows_a, types_a),
    )
    return math.expm1(log_score)


@pytest.mark.parametrize("device_name", ["cpu", "cuda"])
def test_batched_modes_match_explicit_images_and_public_scores(
    monkeypatch, device_name: str
) -> None:
    if device_name == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    positions, cells, pbc, atom_types = _fixtures()
    batch = _batch(positions, cells, pbc)
    pointers = (0, 2, 5, 6)
    calls = []
    original_neighbor_list = descriptor_module.neighbor_list

    def record_query(**kwargs):
        calls.append(
            (
                kwargs["positions"].shape[0],
                kwargs["cell"].shape,
                kwargs["pbc"].detach().cpu().clone(),
                kwargs["batch_idx"].detach().cpu().clone(),
                kwargs["method"],
            )
        )
        return original_neighbor_list(**kwargs)

    monkeypatch.setattr(descriptor_module, "neighbor_list", record_query)
    stores = build_descriptor_stores(
        batch,
        cutoff=_CUTOFF,
        atom_types=atom_types,
        modes=("untyped", "center", "full"),
        device=torch.device(device_name),
        cuda_memory_budget_bytes=100_000_000,
    )

    assert set(stores) == {"untyped", "center", "full"}
    assert len(calls) == 1
    atom_count, cell_shape, queried_pbc, batch_idx, method = calls[0]
    assert atom_count == pointers[-1]
    assert cell_shape == (len(positions), 3, 3)
    assert torch.equal(queried_pbc, pbc)
    assert batch_idx.tolist() == [0, 0, 1, 1, 1, 2]
    assert method == ("batch_cell_list" if device_name == "cuda" else "batch_naive")

    explicit = [
        _explicit_rows(
            pos,
            cells[index],
            pbc[index],
            atom_types[pointers[index] : pointers[index + 1]],
            _CUTOFF,
        )
        for index, pos in enumerate(positions)
    ]
    expected_widths = [max(map(len, rows), default=0) for rows in explicit]
    assert stores["untyped"].widths.tolist() == expected_widths
    assert stores["center"].widths.tolist() == expected_widths
    assert stores["full"].widths.tolist() == expected_widths
    for structure_id, rows in enumerate(explicit):
        width = expected_widths[structure_id]
        for mode in ("untyped", "center", "full"):
            stored_distances, stored_types = _rows_from_store(
                stores[mode], structure_id
            )
            for atom_id, neighbors in enumerate(rows):
                ordered = (
                    sorted(neighbors, key=lambda item: (item[1], item[0]))
                    if mode == "full"
                    else neighbors
                )
                expected_distances = [math.log(distance) for distance, _ in ordered]
                expected_types = [type_id for _, type_id in ordered]
                padding = width - len(ordered)
                expected_distances.extend([math.log(_CUTOFF)] * padding)
                expected_types.extend([_TYPE_PAD] * padding)
                torch.testing.assert_close(
                    stored_distances[atom_id],
                    torch.tensor(expected_distances, dtype=torch.float32),
                    atol=2e-6,
                    rtol=2e-6,
                )
                if mode == "untyped":
                    assert torch.all(stored_types[atom_id] == _TYPE_PAD)
                else:
                    assert stored_types[atom_id].tolist() == expected_types
        assert stores["untyped"].center_types[
            int(stores["untyped"].atom_offsets[structure_id]) : int(
                stores["untyped"].atom_offsets[structure_id + 1]
            )
        ].tolist() == [_TYPE_PAD] * len(positions[structure_id])

    pair = torch.tensor([[0, 1]], dtype=torch.int32, device=device_name)
    for mode in ("untyped", "center", "full"):
        typed = None if mode == "untyped" else atom_types.to(device_name)
        index = RadialComparisonIndex.build(
            batch.to(device_name),
            cutoff=_CUTOFF,
            atom_types=typed,
            typed_neighbors=mode == "full",
            device=device_name,
        )
        expected = _oracle_score(
            explicit[0],
            explicit[1],
            None if typed is None else atom_types[:2],
            None if typed is None else atom_types[2:5],
            _CUTOFF,
            mode == "full",
        )
        assert index.score_pairs(pair).item() == pytest.approx(expected, abs=3e-6)


@pytest.mark.parametrize("device_name", ["cpu", "cuda"])
def test_capacity_retry_and_exact_tile_accounting(
    monkeypatch, device_name: str
) -> None:
    if device_name == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    capacities = []

    def fake_neighbor_list(**kwargs):
        positions = kwargs["positions"]
        capacity = kwargs["max_neighbors"]
        capacities.append(capacity)
        count = positions.shape[0]
        matrix = torch.full(
            (count, capacity), count, dtype=torch.int32, device=positions.device
        )
        counts = torch.zeros(count, dtype=torch.int32, device=positions.device)
        shifts = torch.zeros(
            (count, capacity, 3), dtype=torch.int32, device=positions.device
        )
        if len(capacities) == 1:
            counts.fill_(capacity)
        else:
            counts.fill_(1)
            matrix[:, 0] = torch.tensor([1, 0], device=positions.device)
        return matrix, counts, shifts

    monkeypatch.setattr(descriptor_module, "neighbor_list", fake_neighbor_list)
    positions = torch.tensor([[0.0, 0, 0], [1.0, 0, 0]], dtype=torch.float32)
    tiles = list(
        descriptor_module.build_descriptor_tiles(
            positions,
            (0, 2),
            None,
            torch.zeros((1, 3), dtype=torch.bool),
            None,
            cutoff=2.0,
            device=torch.device(device_name),
            memory_budget_bytes=None,
        )
    )
    assert capacities == [16, 32]
    assert len(tiles) == 1
    tile = tiles[0]
    assert tile.capacity == 32
    assert tile.valid_width == 1
    assert tile.widths == (1,)
    assert tile.estimated_peak_bytes == 2 * 32 * 128 + 2 * 64 + 64
    assert tile.geometry_bytes == tile.distances.numel() * 4


@pytest.mark.parametrize("device_name", ["cpu", "cuda"])
def test_fp32_cutoff_boundary_remains_padding_for_typed_rows(
    device_name: str,
) -> None:
    if device_name == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    positions = [
        torch.tensor([[0.0, 0, 0], [0.75, 0, 0], [1.75, 0, 0]], dtype=torch.float32)
    ]
    cells = torch.eye(3).unsqueeze(0) * 8
    pbc = torch.zeros((1, 3), dtype=torch.bool)
    batch = _batch(positions, cells, pbc).to(device_name)
    store = build_descriptor_stores(
        batch,
        cutoff=1.0,
        atom_types=torch.tensor([1, 2, 3], dtype=torch.int32, device=device_name),
        modes=("full",),
        device=torch.device(device_name),
        cuda_memory_budget_bytes=100_000_000 if device_name == "cuda" else None,
    )["full"]
    distances, neighbors = _rows_from_store(store, 0)
    assert store.widths.tolist() == [1]
    assert neighbors[:, 0].tolist() == [2, 1, _TYPE_PAD]
    assert distances[2, 0].item() == pytest.approx(0.0, abs=1e-7)


def test_cuda_required_residency_keeps_descriptor_and_summaries_on_device() -> None:
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    positions, cells, pbc, atom_types = _fixtures()
    batch = _batch(positions, cells, pbc)
    store = build_descriptor_stores(
        batch,
        cutoff=_CUTOFF,
        atom_types=atom_types,
        modes=("full",),
        device=torch.device("cuda"),
        cuda_memory_budget_bytes=100_000_000,
        require_device_residency=True,
    )["full"]

    assert store.device.type == "cuda"
    assert all(
        tensor.device.type == "cuda"
        for tensor in (
            store.distances,
            store.neighbor_types,
            store.center_types,
            store.row_offsets,
            store.atom_offsets,
            store.widths,
            store.atom_counts,
            store.summaries,
            store.typed_summaries,
            store.center_type_presence,
        )
    )


def test_cuda_required_residency_rejects_insufficient_budget() -> None:
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    positions, cells, pbc, atom_types = _fixtures()
    batch = _batch(positions, cells, pbc)
    with pytest.raises(MemoryError, match="CUDA-resident descriptor"):
        build_descriptor_stores(
            batch,
            cutoff=_CUTOFF,
            atom_types=atom_types,
            modes=("full",),
            device=torch.device("cuda"),
            cuda_memory_budget_bytes=256,
            require_device_residency=True,
        )


def test_dense_periodic_density_seed_and_cpu_cuda_parity(monkeypatch) -> None:
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    queries: list[tuple[str, str, int]] = []
    original_neighbor_list = descriptor_module.neighbor_list

    def record_query(**kwargs):
        queries.append(
            (
                kwargs["positions"].device.type,
                kwargs["method"],
                kwargs["max_neighbors"],
            )
        )
        return original_neighbor_list(**kwargs)

    monkeypatch.setattr(descriptor_module, "neighbor_list", record_query)
    positions = [
        torch.tensor([[0.0, 0, 0], [0.125, 0, 0]], dtype=torch.float32),
        torch.tensor([[0.0, 0, 0], [0.125, 0, 0]], dtype=torch.float32),
    ]
    cells = torch.eye(3).unsqueeze(0).expand(2, 3, 3).clone() * 0.25
    pbc = torch.ones((2, 3), dtype=torch.bool)
    batch = _batch(positions, cells, pbc)
    atomic_density = (2 / abs(float(torch.linalg.det(cells[0])))) * 1.35
    expected_seed = estimate_max_neighbors(1.2, atomic_density=atomic_density)
    stores = {}
    query_ranges = {}
    for device_name in ("cpu", "cuda"):
        query_start = len(queries)
        stores[device_name] = build_descriptor_stores(
            batch.to(device_name),
            cutoff=1.2,
            atom_types=None,
            modes=("untyped",),
            device=torch.device(device_name),
            cuda_memory_budget_bytes=100_000_000,
        )["untyped"]
        query_ranges[device_name] = (query_start, len(queries))

    for device_name, (start, stop) in query_ranges.items():
        assert stop > start
        assert queries[start][2] == expected_seed
        assert all(capacity > 64 for _, _, capacity in queries[start:stop]), (
            "periodic density should seed the first query above the old cap"
        )
        expected_method = "batch_cell_list" if device_name == "cuda" else "batch_naive"
        assert all(
            query_device == device_name and method == expected_method
            for query_device, method, _ in queries[start:stop]
        )

    cpu_store = stores["cpu"]
    cuda_store = stores["cuda"]
    assert cpu_store.widths.tolist() == cuda_store.widths.tolist()
    assert cpu_store.row_offsets.tolist() == cuda_store.row_offsets.tolist()
    torch.testing.assert_close(
        cpu_store.distances,
        cuda_store.distances.cpu(),
        atol=2e-6,
        rtol=2e-6,
    )

    scores = []
    pair = torch.tensor([[0, 1]], dtype=torch.int32)
    for device_name in ("cpu", "cuda"):
        index = RadialComparisonIndex.build(
            batch.to(device_name),
            cutoff=1.2,
            device=device_name,
        )
        scores.append(index.score_pairs(pair.to(device_name)).item())
    assert scores[0] == pytest.approx(scores[1], abs=2e-6)
    assert scores[0] == pytest.approx(0.0, abs=2e-6)
