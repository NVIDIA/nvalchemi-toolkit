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
"""Public behavior checks for atomistic structure comparison."""

from __future__ import annotations

import math
from itertools import product

import pytest
import torch

from nvalchemi.csp.comparison import RadialComparisonIndex  # noqa: E402
from nvalchemi.csp.symmetry import SpaceGroupPolicy  # noqa: E402
from nvalchemi.data import AtomicData, Batch  # noqa: E402


def _batch(
    structures: list[torch.Tensor],
    *,
    cells: torch.Tensor | None = None,
    pbc: torch.Tensor | None = None,
) -> Batch:
    data = []
    for i, positions in enumerate(structures):
        kwargs = {
            "positions": positions,
            "atomic_numbers": torch.ones(len(positions), dtype=torch.int64),
        }
        if cells is not None:
            kwargs["cell"] = cells[i : i + 1]
            kwargs["pbc"] = (
                pbc[i : i + 1]
                if pbc is not None
                else torch.ones((1, 3), dtype=torch.bool)
            )
        data.append(AtomicData(**kwargs))
    return Batch.from_data_list(data)


def _oracle(a: torch.Tensor, b: torch.Tensor, cutoff: float) -> float:
    def rows(pos: torch.Tensor) -> list[list[float]]:
        output = []
        for i in range(len(pos)):
            values = [
                float(torch.linalg.vector_norm(pos[i] - pos[j]))
                for j in range(len(pos))
                if j != i
            ]
            output.append(
                sorted(
                    [d for d in values if d < cutoff]
                    + [cutoff] * (len(pos) - 1 - sum(d < cutoff for d in values))
                )
            )
        return output

    ra, rb = rows(a), rows(b)

    def directed(source: list[list[float]], target: list[list[float]]) -> float:
        if not source:
            return 0.0
        return max(
            min(
                max(
                    max(x, y) / min(x, y) - 1.0
                    for x, y in zip(row, candidate, strict=True)
                )
                for candidate in target
            )
            for row in source
        )

    return max(directed(ra, rb), directed(rb, ra))


def _typed_oracle(
    a: torch.Tensor,
    b: torch.Tensor,
    types_a: torch.Tensor,
    types_b: torch.Tensor,
    cutoff: float,
    typed_neighbors: bool,
) -> float:
    def rows(
        positions: torch.Tensor, types: torch.Tensor
    ) -> list[dict[int, list[float]]]:
        result = []
        for i in range(len(positions)):
            row: dict[int, list[float]] = {}
            for j in range(len(positions)):
                if i == j:
                    continue
                distance = float(torch.linalg.vector_norm(positions[i] - positions[j]))
                if distance < cutoff:
                    group = int(types[j]) if typed_neighbors else 0
                    row.setdefault(group, []).append(distance)
            for distances in row.values():
                distances.sort()
            result.append(row)
        return result

    rows_a, rows_b = rows(a, types_a), rows(b, types_b)

    def mismatch(row_a: dict[int, list[float]], row_b: dict[int, list[float]]) -> float:
        if not typed_neighbors:
            distances_a = sorted(row_a.get(0, []))
            distances_b = sorted(row_b.get(0, []))
            width = max(len(distances_a), len(distances_b))
            distances_a += [cutoff] * (width - len(distances_a))
            distances_b += [cutoff] * (width - len(distances_b))
            return max(
                (
                    abs(math.log(x) - math.log(y))
                    for x, y in zip(distances_a, distances_b)
                ),
                default=0.0,
            )
        worst = 0.0
        for group in row_a.keys() | row_b.keys():
            distances_a, distances_b = row_a.get(group, []), row_b.get(group, [])
            width = max(len(distances_a), len(distances_b))
            padded_a = distances_a + [cutoff] * (width - len(distances_a))
            padded_b = distances_b + [cutoff] * (width - len(distances_b))
            worst = max(
                worst,
                max(
                    (
                        abs(math.log(x) - math.log(y))
                        for x, y in zip(padded_a, padded_b)
                    ),
                    default=0.0,
                ),
            )
        return worst

    def directed(
        source_rows: list[dict[int, list[float]]],
        source_types: torch.Tensor,
        target_rows: list[dict[int, list[float]]],
        target_types: torch.Tensor,
    ) -> float:
        worst = 0.0
        for source, center_type in zip(source_rows, source_types.tolist()):
            compatible = [
                target
                for target, target_type in zip(target_rows, target_types.tolist())
                if int(target_type) == int(center_type)
            ]
            if not compatible:
                return math.inf
            worst = max(
                worst,
                min(mismatch(source, target) for target in compatible),
            )
        return worst

    log_score = max(
        directed(rows_a, types_a, rows_b, types_b),
        directed(rows_b, types_b, rows_a, types_a),
    )
    return math.expm1(log_score)


def _periodic_rows_oracle(
    positions: torch.Tensor,
    cell: torch.Tensor,
    cutoff: float,
    *,
    enumerate_images: bool = True,
) -> list[list[float]]:
    """Build radial rows from explicit periodic images in float64."""
    positions64 = positions.to(dtype=torch.float64)
    cell64 = cell.to(dtype=torch.float64)
    if enumerate_images:
        max_displacement = max(
            float(torch.linalg.vector_norm(positions64[i] - positions64[j]))
            for i in range(len(positions64))
            for j in range(len(positions64))
        )
        smallest_singular_value = float(torch.linalg.svdvals(cell64).min())
        shift_bound = math.ceil((cutoff + max_displacement) / smallest_singular_value)
    else:
        shift_bound = 0

    shifts = product(range(-shift_bound, shift_bound + 1), repeat=3)
    shifts = list(shifts)
    rows = []
    for i in range(len(positions64)):
        distances = []
        for j in range(len(positions64)):
            displacement = positions64[i] - positions64[j]
            for shift in shifts:
                if i == j and shift == (0, 0, 0):
                    continue
                lattice_shift = torch.tensor(shift, dtype=torch.float64) @ cell64
                distance = float(torch.linalg.vector_norm(displacement + lattice_shift))
                if distance < cutoff:
                    distances.append(distance)
        rows.append(sorted(distances))
    return rows


def _periodic_score_oracle(
    a_rows: list[list[float]], b_rows: list[list[float]], cutoff: float
) -> float:
    """Compare radial rows with max log-distance matching in both directions."""

    def mismatch(row_a: list[float], row_b: list[float]) -> float:
        width = max(len(row_a), len(row_b))
        padded_a = row_a + [cutoff] * (width - len(row_a))
        padded_b = row_b + [cutoff] * (width - len(row_b))
        return max(
            (abs(math.log(x) - math.log(y)) for x, y in zip(padded_a, padded_b)),
            default=0.0,
        )

    def directed(source: list[list[float]], target: list[list[float]]) -> float:
        return max(
            (min(mismatch(row, candidate) for candidate in target) for row in source),
            default=0.0,
        )

    return math.expm1(max(directed(a_rows, b_rows), directed(b_rows, a_rows)))


@pytest.mark.parametrize("device_name", ["cpu", "cuda"])
def test_scores_match_independent_distance_oracle_and_invariances(
    device_name: str,
) -> None:
    if device_name == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    a = torch.tensor([[0.0, 0, 0], [1.0, 0, 0], [0.0, 2.0, 0]], dtype=torch.float32)
    b = torch.tensor([[4.0, 3, 0], [4.0, 4.1, 0], [4.0, 3, 2.0]], dtype=torch.float32)
    batch = _batch([a, b, a.flip(0) + 9])
    before = batch.positions.clone()
    index = RadialComparisonIndex.build(
        batch.to(device_name), cutoff=3.0, device=device_name
    )
    expected = _oracle(a, b, 3.0)
    score = index.score_pairs(
        torch.tensor([[0, 1]], dtype=torch.int32, device=device_name)
    )
    assert score.dtype == torch.float32
    assert score.item() == pytest.approx(expected, abs=2e-6)
    assert index.score_pairs(
        torch.tensor([[0, 2]], dtype=torch.int32, device=device_name)
    ).item() == pytest.approx(0.0, abs=2e-6)
    assert index.find_matches(threshold=0.0).tolist() == [[0, 2]]
    assert torch.equal(batch.positions, before)


def test_index_uses_frozen_descriptor_snapshot_for_repeated_queries() -> None:
    a = torch.tensor([[0.0, 0, 0], [1.0, 0, 0], [0.0, 2.0, 0]])
    batch = _batch([a, a + 4])
    atom_types = torch.tensor([6, 1, 1, 6, 1, 1], dtype=torch.int64)
    index = RadialComparisonIndex.build(
        batch, cutoff=3.0, atom_types=atom_types, typed_neighbors=True
    )
    pair = torch.tensor([[0, 1]], dtype=torch.int32)
    score = index.score_pairs(pair).clone()
    matches = index.find_matches(threshold=0.0).clone()
    deduplicated = index.deduplicate(threshold=0.0)

    batch.positions[4, 0] += 0.4
    atom_types[3:] = torch.tensor([1, 6, 1])

    assert torch.equal(index.score_pairs(pair), score)
    assert torch.equal(index.find_matches(threshold=0.0), matches)
    repeated = index.deduplicate(threshold=0.0)
    assert torch.equal(repeated.retained_indices, deduplicated.retained_indices)
    assert torch.equal(
        repeated.representative_indices, deduplicated.representative_indices
    )
    assert torch.equal(repeated.multiplicities, deduplicated.multiplicities)
    mutated_index = RadialComparisonIndex.build(
        batch, cutoff=3.0, atom_types=atom_types, typed_neighbors=True
    )
    assert mutated_index.score_pairs(pair).item() > score.item()
    assert mutated_index.deduplicate(threshold=0.0).retained_indices.tolist() == [0, 1]


def test_typing_threshold_order_and_empty_pools() -> None:
    a = torch.tensor([[0.0, 0, 0], [1.0, 0, 0], [0.0, 2.0, 0]], dtype=torch.float32)
    batch = _batch([a, a + 3])
    types = torch.tensor([5, 7, 5, 5, 7, 5], dtype=torch.int64)
    index = RadialComparisonIndex.build(batch, cutoff=3.0, atom_types=types)
    assert index.find_matches(threshold=0.0).tolist() == [[0, 1]]
    supplied = torch.tensor([[1, 0], [0, 1], [1, 0]], dtype=torch.int64)
    assert index.find_matches(threshold=0.0, pair_indices=supplied).tolist() == [
        [1, 0],
        [0, 1],
        [1, 0],
    ]
    assert index.score_pairs(supplied, pair_chunk_size=1).tolist() == pytest.approx(
        [0.0, 0.0, 0.0], abs=2e-6
    )
    assert index.find_matches(
        threshold=0.0, pair_indices=torch.empty((0, 2), dtype=torch.int32)
    ).shape == (0, 2)
    with pytest.raises(TypeError, match="float32 only"):
        RadialComparisonIndex.build(batch, cutoff=3.0, dtype=torch.float64)


@pytest.mark.parametrize("device_name", ["cpu", "cuda"])
def test_typed_neighbors_change_score_against_independent_oracle(
    device_name: str,
) -> None:
    if device_name == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    square = torch.tensor([[0.0, 0, 0], [1.0, 0, 0], [1.0, 1.0, 0], [0.0, 1.0, 0]])
    a, b = square, square + 3
    types_a = torch.tensor([1, 1, 2, 2], dtype=torch.int64)
    types_b = torch.tensor([1, 2, 1, 2], dtype=torch.int64)
    batch = _batch([a, b])
    atom_types = torch.cat((types_a, types_b))
    pair = torch.tensor([[0, 1]], dtype=torch.int32, device=device_name)
    untyped = RadialComparisonIndex.build(
        batch.to(device_name),
        cutoff=2.0,
        atom_types=atom_types.to(device_name),
        typed_neighbors=False,
        device=device_name,
    )
    typed = RadialComparisonIndex.build(
        batch.to(device_name),
        cutoff=2.0,
        atom_types=atom_types.to(device_name),
        typed_neighbors=True,
        device=device_name,
    )
    expected_untyped = _typed_oracle(a, b, types_a, types_b, 2.0, False)
    expected_typed = _typed_oracle(a, b, types_a, types_b, 2.0, True)
    assert untyped.score_pairs(pair).item() == pytest.approx(expected_untyped, abs=2e-6)
    assert typed.score_pairs(pair).item() == pytest.approx(expected_typed, abs=2e-6)
    assert untyped.find_matches(
        threshold=expected_untyped, pair_indices=pair
    ).tolist() == [[0, 1]]
    assert typed.find_matches(threshold=expected_typed, pair_indices=pair).tolist() == [
        [0, 1]
    ]
    assert expected_untyped < 0.1 < expected_typed
    assert untyped.find_matches(threshold=0.1, pair_indices=pair).tolist() == [[0, 1]]
    assert typed.find_matches(threshold=0.1, pair_indices=pair).shape == (0, 2)


@pytest.mark.parametrize("device_name", ["cpu", "cuda"])
def test_typed_rows_with_repeated_labels_and_unequal_neighbor_counts(
    device_name: str,
) -> None:
    if device_name == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    a = torch.tensor([[0.0, 0, 0], [1.0, 0, 0], [3.0, 0, 0], [10.0, 0, 0]])
    b = torch.tensor([[0.0, 0, 0], [1.0, 0, 0], [4.5, 0, 0], [10.0, 0, 0]]) + 20
    types = torch.tensor([10, 20, 20, 30] * 2, dtype=torch.int32)
    index = RadialComparisonIndex.build(
        _batch([a, b]).to(device_name),
        cutoff=4.0,
        atom_types=types.to(device_name),
        typed_neighbors=True,
        device=device_name,
    )
    pair = torch.tensor([[0, 1]], dtype=torch.int32, device=device_name)
    expected = _typed_oracle(
        a, b, types[:4].to(torch.int64), types[4:].to(torch.int64), 4.0, True
    )
    score = index.score_pairs(pair).item()
    assert score == pytest.approx(expected, abs=2e-6)
    assert index.find_matches(threshold=score, pair_indices=pair).tolist() == [[0, 1]]


def test_inclusive_threshold_and_cross_index_pair_order() -> None:
    a = torch.tensor([[0.0, 0, 0], [1.0, 0, 0]], dtype=torch.float32)
    b = torch.tensor([[3.0, 0, 0], [4.0499997, 0, 0]], dtype=torch.float32)
    left = RadialComparisonIndex.build(_batch([a]), cutoff=2.0)
    right = RadialComparisonIndex.build(_batch([b, a]), cutoff=2.0)
    assert left.find_matches(right, threshold=0.05).tolist() == [[0, 0], [0, 1]]


def test_two_by_two_cross_search_uses_query_major_order() -> None:
    a = torch.tensor([[0.0, 0, 0], [1.0, 0, 0]], dtype=torch.float32)
    left = RadialComparisonIndex.build(_batch([a, a + 3]), cutoff=2.0)
    right = RadialComparisonIndex.build(_batch([a + 6, a + 9]), cutoff=2.0)
    assert left.find_matches(right, threshold=0.0).tolist() == [
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1],
    ]


def test_threshold_is_inclusive_but_does_not_admit_materially_higher_score() -> None:
    a = torch.tensor([[0.0, 0, 0], [1.0, 0, 0]], dtype=torch.float32)
    equal = torch.tensor([[0.0, 0, 0], [1.05, 0, 0]], dtype=torch.float32)
    above = torch.tensor([[0.0, 0, 0], [1.0500003, 0, 0]], dtype=torch.float32)
    equal_index = RadialComparisonIndex.build(_batch([a, equal]), cutoff=2.0)
    above_index = RadialComparisonIndex.build(_batch([a, above]), cutoff=2.0)
    equal_score = equal_index.score_pairs(
        torch.tensor([[0, 1]], dtype=torch.int32)
    ).item()
    above_score = above_index.score_pairs(
        torch.tensor([[0, 1]], dtype=torch.int32)
    ).item()
    assert equal_score <= 0.05
    assert equal_index.find_matches(threshold=0.05).tolist() == [[0, 1]]
    assert above_score > 0.05
    assert above_index.find_matches(threshold=0.05).shape == (0, 2)


def test_no_compatible_centers_never_match_finite_threshold() -> None:
    a = torch.tensor([[0.0, 0, 0], [1.0, 0, 0]], dtype=torch.float32)
    batch = _batch([a, a + 3])
    index = RadialComparisonIndex.build(
        batch,
        cutoff=2.0,
        atom_types=torch.tensor([1, 1, 2, 2], dtype=torch.int64),
    )
    assert torch.isinf(
        index.score_pairs(torch.tensor([[0, 1]], dtype=torch.int32))
    ).all()
    assert index.find_matches(threshold=1e308).shape == (0, 2)


@pytest.mark.parametrize(
    "out_of_range_type",
    [torch.iinfo(torch.int64).min, torch.iinfo(torch.int64).max],
)
def test_atom_type_labels_outside_int32_are_rejected(out_of_range_type: int) -> None:
    batch = _batch([torch.tensor([[0.0, 0, 0]], dtype=torch.float32)])
    with pytest.raises(ValueError, match="fit signed int32"):
        RadialComparisonIndex.build(
            batch,
            cutoff=2.0,
            atom_types=torch.tensor([out_of_range_type], dtype=torch.int64),
        )


def test_reserved_int32_minimum_atom_type_is_rejected() -> None:
    batch = _batch([torch.tensor([[0.0, 0, 0]], dtype=torch.float32)])
    with pytest.raises(ValueError, match="reserved int32 minimum"):
        RadialComparisonIndex.build(
            batch,
            cutoff=2.0,
            atom_types=torch.tensor([torch.iinfo(torch.int32).min], dtype=torch.int32),
        )


@pytest.mark.parametrize("device_name", ["cpu", "cuda"])
def test_int32_max_atom_type_scores_and_matches(device_name: str) -> None:
    if device_name == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    atom_types = torch.tensor(
        [torch.iinfo(torch.int32).max, torch.iinfo(torch.int32).max] * 2,
        dtype=torch.int64,
    )
    shape = torch.tensor([[0.0, 0, 0], [1.0, 0, 0]], dtype=torch.float32)
    index = RadialComparisonIndex.build(
        _batch([shape, shape + 4]).to(device_name),
        cutoff=2.0,
        atom_types=atom_types.to(device_name),
        device=device_name,
    )
    pair = torch.tensor([[0, 1]], dtype=torch.int32, device=device_name)
    assert index.score_pairs(pair).tolist() == pytest.approx([0.0], abs=2e-6)
    assert index.find_matches(threshold=0.0, pair_indices=pair).tolist() == [[0, 1]]


@pytest.mark.parametrize("device_name", ["cpu", "cuda"])
def test_cross_index_partially_overlapping_type_labels_preserve_equality(
    device_name: str,
) -> None:
    if device_name == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    positions = torch.tensor(
        [[0.0, 0, 0], [1.0, 0, 0], [0.0, 2.0, 0]], dtype=torch.float32
    )
    left = RadialComparisonIndex.build(
        _batch([positions]).to(device_name),
        cutoff=4.0,
        atom_types=torch.tensor([10, 20, 30], dtype=torch.int32, device=device_name),
        device=device_name,
    )
    right = RadialComparisonIndex.build(
        _batch([positions + 5]).to(device_name),
        cutoff=4.0,
        atom_types=torch.tensor([20, 30, 40], dtype=torch.int64, device=device_name),
        device=device_name,
    )
    pair = torch.tensor([[0, 0]], dtype=torch.int32, device=device_name)
    assert torch.isinf(left.score_pairs(pair, other=right)).all()
    assert left.find_matches(other=right, threshold=1e30, pair_indices=pair).shape == (
        0,
        2,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_cuda_high_type_vocab_respects_filter_workspace_budget(monkeypatch) -> None:
    atom_count = 128
    positions = torch.zeros((atom_count, 3), dtype=torch.float32)
    positions[:, 0] = torch.arange(atom_count, dtype=torch.float32) * 0.05
    batch = _batch([positions, positions + 20.0])
    labels = torch.arange(atom_count, dtype=torch.int32).repeat(2)
    monkeypatch.setattr(
        torch.cuda,
        "mem_get_info",
        lambda device=None: (20_000_000, 20_000_000),
    )
    index = RadialComparisonIndex.build(
        batch,
        cutoff=10.0,
        atom_types=labels,
        device="cuda",
        max_memory_fraction=0.70,
    )
    pair = torch.tensor([[0, 1]], dtype=torch.int32, device="cuda")
    with pytest.raises(MemoryError, match="one typed comparison pair"):
        index.find_matches(threshold=0.0, pair_indices=pair)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_cuda_typed_summary_rejects_over_budget_build(monkeypatch) -> None:
    atom_count = 24
    positions = torch.zeros((atom_count, 3), dtype=torch.float32)
    positions[:, 0] = torch.arange(atom_count, dtype=torch.float32) * 0.5
    batch = _batch([positions])
    labels = torch.arange(atom_count, dtype=torch.int32)
    monkeypatch.setattr(
        torch.cuda,
        "mem_get_info",
        lambda device=None: (100_000_000, 100_000_000),
    )
    with pytest.raises(
        MemoryError, match="typed summary.*configured 100000-byte build budget"
    ):
        RadialComparisonIndex.build(
            batch,
            cutoff=20.0,
            atom_types=labels,
            typed_neighbors=True,
            device="cuda",
            max_memory_fraction=0.001,
        )


@pytest.mark.parametrize("cutoff", [1e100, 1e-100])
def test_cutoff_must_be_representable_as_positive_fp32(cutoff: float) -> None:
    batch = _batch([torch.tensor([[0.0, 0, 0]], dtype=torch.float32)])
    with pytest.raises(ValueError, match="positive finite FP32"):
        RadialComparisonIndex.build(batch, cutoff=cutoff)


def test_extreme_fp64_geometry_rejected_after_fp32_conversion() -> None:
    positions = torch.tensor([[0.0, 0, 0], [1e100, 0, 0]], dtype=torch.float64)
    with pytest.raises(ValueError, match="positions must remain finite"):
        RadialComparisonIndex.build(_batch([positions]), cutoff=2.0).score_pairs(
            torch.tensor([[0, 0]], dtype=torch.int32)
        )

    cell = torch.eye(3, dtype=torch.float64).unsqueeze(0) * 1e100
    periodic = _batch(
        [torch.tensor([[0.0, 0, 0], [1.0, 0, 0]], dtype=torch.float64)],
        cells=cell,
    )
    with pytest.raises(ValueError, match="periodic cell must remain finite"):
        RadialComparisonIndex.build(periodic, cutoff=2.0).score_pairs(
            torch.tensor([[0, 0]], dtype=torch.int32)
        )


def test_fp64_source_and_nonperiodic_singular_cell() -> None:
    a = torch.tensor([[0.0, 0, 0], [1.0, 0, 0]], dtype=torch.float64)
    b = a + 2
    cells = torch.zeros((2, 3, 3), dtype=torch.float64)
    pbc = torch.zeros((2, 3), dtype=torch.bool)
    index = RadialComparisonIndex.build(
        _batch([a, b], cells=cells, pbc=pbc), cutoff=2.0
    )
    assert index.score_pairs(
        torch.tensor([[0, 1]], dtype=torch.int32)
    ).item() == pytest.approx(0.0, abs=2e-6)


@pytest.mark.parametrize("device_name", ["cpu", "cuda"])
def test_missing_cell_treats_periodic_flags_as_nonperiodic(
    device_name: str,
) -> None:
    if device_name == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    positions = [
        torch.tensor([[0.0, 0, 0], [0.5, 0, 0]], dtype=torch.float32),
        torch.tensor([[0.0, 0, 0], [0.75, 0, 0]], dtype=torch.float32),
    ]

    def make_batch(pbc_value: bool) -> Batch:
        return Batch.from_data_list(
            [
                AtomicData(
                    positions=structure,
                    atomic_numbers=torch.ones(len(structure), dtype=torch.int64),
                    pbc=torch.full((1, 3), pbc_value, dtype=torch.bool),
                )
                for structure in positions
            ]
        )

    flagged_index = RadialComparisonIndex.build(
        make_batch(True).to(device_name), cutoff=2.0, device=device_name
    )
    nonperiodic_index = RadialComparisonIndex.build(
        make_batch(False).to(device_name), cutoff=2.0, device=device_name
    )
    pair = torch.tensor([[0, 1]], dtype=torch.int32, device=device_name)
    flagged_score = flagged_index.score_pairs(pair, other=nonperiodic_index)
    nonperiodic_score = nonperiodic_index.score_pairs(pair)
    torch.testing.assert_close(flagged_score, nonperiodic_score)
    assert flagged_score.item() == pytest.approx(0.5, abs=2e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_cuda_comparison_scores_and_preserves_order() -> None:
    a = torch.tensor([[0.1, 0, 0], [9.9, 0, 0]], dtype=torch.float32, device="cuda")
    b = torch.tensor([[0.1, 0, 0], [-0.1, 0, 0]], dtype=torch.float32, device="cuda")
    cell = torch.eye(3, dtype=torch.float32, device="cuda").unsqueeze(0) * 10
    pbc = torch.tensor([[True, False, False]], dtype=torch.bool, device="cuda")
    batch = Batch.from_data_list(
        [
            AtomicData(
                positions=a,
                atomic_numbers=torch.ones(2, dtype=torch.int64, device="cuda"),
                cell=cell,
                pbc=pbc,
            ),
            AtomicData(
                positions=b,
                atomic_numbers=torch.ones(2, dtype=torch.int64, device="cuda"),
                cell=cell,
                pbc=pbc,
            ),
        ],
        device="cuda",
    )
    atom_types = torch.tensor([5, 7, 5, 7], dtype=torch.int64, device="cuda")
    index = RadialComparisonIndex.build(
        batch, cutoff=1.0, device="cuda", atom_types=atom_types
    )
    pairs = torch.tensor([[1, 0], [0, 1]], dtype=torch.int32, device="cuda")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        scores = index.score_pairs(pairs, pair_chunk_size=1)
        matches = index.find_matches(threshold=0.01, pair_indices=pairs)
        endpoint = index.find_matches(
            threshold=float(scores[0].item()), pair_indices=pairs
        )
    stream.synchronize()
    assert scores.device.type == "cuda"
    assert scores.tolist()[0] == pytest.approx(scores.tolist()[1], abs=2e-6)
    assert matches.tolist() == [[1, 0], [0, 1]]
    expected = pairs[
        scores <= torch.nextafter(scores[0], torch.tensor(float("inf"), device="cuda"))
    ]
    assert torch.equal(endpoint, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_cuda_tiny_workspace_budget_fails_before_descriptor_construction() -> None:
    a = torch.tensor([[0.0, 0, 0], [1.0, 0, 0]], dtype=torch.float32, device="cuda")
    batch = _batch([a], cells=None)
    with pytest.raises(MemoryError, match="budget"):
        RadialComparisonIndex.build(
            batch,
            cutoff=2.0,
            device="cuda",
            max_memory_fraction=1e-12,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_cuda_host_backed_summary_stages_for_public_queries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    batch = _batch([torch.zeros((1, 3)) for _ in range(810)])
    free_bytes = 1_571_429  # 70% gives a 1,100,000-byte index budget.
    monkeypatch.setattr(
        torch.cuda,
        "mem_get_info",
        lambda device=None: (free_bytes, free_bytes),
    )
    index = RadialComparisonIndex.build(
        batch,
        cutoff=2.0,
        atom_types=torch.ones(batch.num_nodes, dtype=torch.int64),
        device="cuda",
        max_memory_fraction=0.70,
        structure_block_size=8,
    )
    pair = torch.tensor([[0, 1]], dtype=torch.int32, device="cuda")
    assert index.score_pairs(pair).item() == pytest.approx(0.0, abs=2e-6)
    assert index.find_matches(threshold=0.0, pair_indices=pair).tolist() == [[0, 1]]


def test_partial_pbc_and_zero_distance_rejection() -> None:
    a = torch.tensor([[0.0, 0, 0], [1.0, 0, 0]], dtype=torch.float32)
    cell = torch.eye(3, dtype=torch.float32).unsqueeze(0).repeat(2, 1, 1) * 10
    pbc = torch.tensor([[True, False, False], [True, False, False]])
    index = RadialComparisonIndex.build(
        _batch(
            [
                a,
                a + 2,
            ],
            cells=cell,
            pbc=pbc,
        ),
        cutoff=2.0,
    )
    assert index.find_matches(threshold=0.0).tolist() == [[0, 1]]
    overlap = _batch([torch.tensor([[0.0, 0, 0], [0.0, 0, 0]], dtype=torch.float32)])
    with pytest.raises(ValueError, match="zero distance"):
        RadialComparisonIndex.build(overlap, cutoff=1.0).score_pairs(
            torch.tensor([[0, 0]], dtype=torch.int32)
        )


def test_skew_cell_periodic_boundary_uses_row_cell_and_shift_sign() -> None:
    cell_one = torch.tensor([3.0, 9.0, 0.0], dtype=torch.float32)
    cell = torch.tensor(
        [[10.0, 0.0, 0.0], [3.0, 9.0, 0.0], [1.0, 2.0, 8.0]], dtype=torch.float32
    )
    p0 = torch.tensor([0.1, 0.1, 0.1], dtype=torch.float32)
    short = cell_one / torch.linalg.vector_norm(cell_one) * 0.25
    p1 = p0 + short - cell_one
    first = torch.stack((p0, p1))
    second = torch.stack((p0, p1 + cell_one))
    cells = cell.unsqueeze(0).repeat(2, 1, 1)
    pbc = torch.tensor([[False, True, False], [False, True, False]])
    index = RadialComparisonIndex.build(
        _batch([first, second], cells=cells, pbc=pbc), cutoff=1.0
    )
    assert index.score_pairs(
        torch.tensor([[0, 1]], dtype=torch.int32)
    ).item() == pytest.approx(0.0, abs=2e-6)


@pytest.mark.parametrize("device_name", ["cpu", "cuda"])
def test_periodic_scores_match_explicit_image_oracle_and_chunking(
    device_name: str,
) -> None:
    if device_name == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires CUDA")

    cell = torch.tensor(
        [[3.0, 0.0, 0.0], [1.1, 2.8, 0.0], [0.3, 0.4, 3.1]],
        dtype=torch.float32,
    )
    a = torch.tensor([[0.1, 0.2, 0.2], [2.8, 0.2, 0.2]], dtype=torch.float32)
    b = torch.tensor([[0.1, 0.2, 0.2], [2.6, 0.2, 0.2]], dtype=torch.float32)
    cutoff = 3.2
    expected_ab = _periodic_score_oracle(
        _periodic_rows_oracle(a, cell, cutoff),
        _periodic_rows_oracle(b, cell, cutoff),
        cutoff,
    )
    zero_image_ab = _periodic_score_oracle(
        _periodic_rows_oracle(a, cell, cutoff, enumerate_images=False),
        _periodic_rows_oracle(b, cell, cutoff, enumerate_images=False),
        cutoff,
    )
    assert expected_ab == pytest.approx(2.0 / 3.0, abs=2e-6)
    assert zero_image_ab == pytest.approx(0.08, abs=2e-6)
    assert expected_ab - zero_image_ab > 0.5

    cells = cell.unsqueeze(0).repeat(3, 1, 1)
    pbc = torch.ones((3, 3), dtype=torch.bool)
    batch = _batch([a, b, a], cells=cells, pbc=pbc).to(device_name)
    index = RadialComparisonIndex.build(batch, cutoff=cutoff, device=device_name)
    pairs = torch.tensor(
        [[0, 1], [0, 2], [1, 2]], dtype=torch.int32, device=device_name
    )
    expected_scores = [expected_ab, 0.0, expected_ab]
    scores_short_chunk = index.score_pairs(pairs, pair_chunk_size=2)
    scores_large_chunk = index.score_pairs(pairs, pair_chunk_size=4)
    assert scores_short_chunk.tolist() == pytest.approx(expected_scores, abs=2e-5)
    assert torch.allclose(scores_short_chunk, scores_large_chunk, atol=2e-6, rtol=0.0)
    assert index.find_matches(threshold=0.60, pair_chunk_size=2).tolist() == [[0, 2]]
    assert index.find_matches(threshold=0.70, pair_chunk_size=2).tolist() == [
        [0, 1],
        [0, 2],
        [1, 2],
    ]


def test_dense_periodic_images_grow_neighbor_capacity_past_old_limit() -> None:
    positions = torch.tensor([[0.0, 0, 0], [0.125, 0, 0]], dtype=torch.float32)
    cells = torch.eye(3, dtype=torch.float32).unsqueeze(0).repeat(2, 1, 1) * 0.25
    pbc = torch.ones((2, 3), dtype=torch.bool)
    batch = _batch([positions, positions + 0.03], cells=cells, pbc=pbc)
    index = RadialComparisonIndex.build(batch, cutoff=1.2)
    assert index.score_pairs(
        torch.tensor([[0, 1]], dtype=torch.int32)
    ).item() == pytest.approx(0.0, abs=2e-6)


def test_empty_batch_pool() -> None:
    index = RadialComparisonIndex.build(Batch(device="cpu"), cutoff=2.0)
    assert index.num_structures == 0
    assert index.find_matches(threshold=0.0).shape == (0, 2)
    result = index.deduplicate(threshold=0.0)
    assert result.retained_indices.shape == (0,)
    assert result.representative_indices.shape == (0,)
    assert result.multiplicities.shape == (0,)


def test_single_structure_blocks_allow_one_descriptor_per_side() -> None:
    a = torch.tensor([[0.0, 0, 0], [1.0, 0, 0]], dtype=torch.float32)
    batch = _batch([a, a + 2])
    index = RadialComparisonIndex.build(batch, cutoff=2.0, structure_block_size=1)
    assert index.score_pairs(
        torch.tensor([[0, 1]], dtype=torch.int32)
    ).tolist() == pytest.approx([0.0], abs=2e-6)


def test_empty_graph_in_nonempty_batch_is_rejected() -> None:
    empty = AtomicData(
        positions=torch.empty((0, 3), dtype=torch.float32),
        atomic_numbers=torch.empty(0, dtype=torch.int64),
    )
    nonempty = AtomicData(
        positions=torch.tensor([[0.0, 0, 0]], dtype=torch.float32),
        atomic_numbers=torch.ones(1, dtype=torch.int64),
    )
    with pytest.raises(ValueError, match="structures must contain atoms"):
        RadialComparisonIndex.build(Batch.from_data_list([empty, nonempty]), cutoff=2.0)


def test_callback_order_and_greedy_nontransitivity() -> None:
    shapes = [
        torch.tensor([[0.0, 0, 0], [1.0, 0, 0]], dtype=torch.float32),
        torch.tensor([[0.0, 0, 0], [1.05, 0, 0]], dtype=torch.float32),
        torch.tensor([[0.0, 0, 0], [1.11, 0, 0]], dtype=torch.float32),
    ]
    index = RadialComparisonIndex.build(
        _batch(shapes), cutoff=2.0, structure_block_size=1
    )
    seen: list[list[list[int]]] = []
    result = index.deduplicate(
        threshold=0.06,
        pair_chunk_size=1,
        confirm=lambda pairs: seen.append(pairs.tolist()) or pairs[:1],
    )
    assert result.retained_indices.dtype == torch.int32
    assert result.retained_indices.tolist() == [0, 2]
    assert result.representative_indices.tolist() == [0, 0, 2]
    assert result.multiplicities.tolist() == [2, 1]
    assert seen == [[[1, 0]]]


@pytest.mark.parametrize("device_name", ["cpu", "cuda"])
@pytest.mark.parametrize("pair_chunk_size", [1, 2, None])
def test_ordered_matches_and_dedup_callback_are_chunk_invariant(
    device_name: str, pair_chunk_size: int | None
) -> None:
    if device_name == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    shape = torch.tensor([[0.0, 0, 0], [1.0, 0, 0]], dtype=torch.float32)
    batch = _batch(
        [
            shape,
            shape * torch.tensor([1.05, 1.0, 1.0]),
            shape * torch.tensor([1.11, 1.0, 1.0]),
            shape + 7.0,
        ]
    ).to(device_name)
    index = RadialComparisonIndex.build(batch, cutoff=2.0, device=device_name)
    ordered_pairs = torch.tensor(
        [[1, 0], [2, 0], [2, 1], [3, 0], [3, 1], [3, 2]],
        dtype=torch.int32,
        device=device_name,
    )
    matches = index.find_matches(
        threshold=0.06,
        pair_indices=ordered_pairs,
        pair_chunk_size=pair_chunk_size,
    )
    assert matches.tolist() == [[1, 0], [2, 1], [3, 0], [3, 1]]

    received: list[list[int]] = []
    returned: list[list[int]] = []

    def confirm(pairs: torch.Tensor) -> torch.Tensor:
        received.extend(pairs.tolist())
        accepted = pairs
        returned.extend(accepted.tolist())
        return accepted

    result = index.deduplicate(
        threshold=0.06,
        confirm=confirm,
        pair_chunk_size=pair_chunk_size,
    )
    assert result.retained_indices.tolist() == [0, 2]
    assert result.representative_indices.tolist() == [0, 0, 2, 0]
    assert result.multiplicities.tolist() == [3, 1]
    assert received == [[1, 0], [3, 0]]
    assert returned == received


def test_confirmation_callback_must_return_ordered_subset() -> None:
    shapes = [
        torch.tensor([[0.0, 0, 0], [1.0, 0, 0]], dtype=torch.float32),
        torch.tensor([[0.0, 0, 0], [1.11, 0, 0]], dtype=torch.float32),
        torch.tensor([[0.0, 0, 0], [1.05, 0, 0]], dtype=torch.float32),
    ]
    index = RadialComparisonIndex.build(_batch(shapes), cutoff=2.0)
    with pytest.raises(ValueError, match="ordered subset"):
        index.deduplicate(threshold=0.06, confirm=lambda pairs: pairs.flip(0))

    def mutate_proposals(pairs: torch.Tensor) -> torch.Tensor:
        pairs[0, 1] = pairs[0, 0]
        return pairs

    with pytest.raises(ValueError, match="ordered subset"):
        index.deduplicate(threshold=0.06, confirm=mutate_proposals)


def test_packer_produced_batch_is_accepted() -> None:
    from nvalchemi.csp.data import MolecularPackingInput
    from nvalchemi.csp.packer import CrystalPacker, PackingConfig

    inputs = MolecularPackingInput(
        conformer_positions=torch.tensor([[0.0, 0, 0]], dtype=torch.float32),
        conformer_ptr=torch.tensor([0, 1], dtype=torch.int32),
        molecule_conformer_ptr=torch.tensor([0, 1], dtype=torch.int32),
        molecule_atom_ptr=torch.tensor([0, 1], dtype=torch.int32),
        atomic_numbers=torch.tensor([6], dtype=torch.int64),
        contact_distances=torch.ones((1, 1), dtype=torch.float32),
        component_index=torch.tensor([0], dtype=torch.int32),
        formula_unit_volume=10_000.0,
    )
    result = CrystalPacker(
        PackingConfig(
            z=1,
            z_prime=1,
            batch_size=2,
            max_candidates=2,
            cell_volume_range=(10_000.0, 10_000.0),
            space_groups=SpaceGroupPolicy.fixed(1),
        ),
        device="cpu",
    )(inputs, num_samples=2, rng=torch.Generator().manual_seed(9))
    packed_batch = result.structures.to_batch()
    assert RadialComparisonIndex.build(packed_batch, cutoff=3.0).num_structures == 2
