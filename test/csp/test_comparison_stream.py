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
"""Public behavior checks for bounded streaming deduplication."""

from __future__ import annotations

import math
import weakref

import pytest
import torch

from nvalchemi.csp._comparison import stream as stream_module  # noqa: E402
from nvalchemi.csp._comparison.stream import (  # noqa: E402
    _candidate_representatives,
    _candidate_representatives_batch,
    _get_last_stream_stats,
    _pilot_heldout_selectivity,
    _pilot_ids,
    _screen_candidate,
    _select_coordinates,
    _summary_pairs_may_match_batch,
)
from nvalchemi.csp.comparison import (  # noqa: E402
    RadialComparisonIndex,
    _conservative_summary_log_bound,
    deduplicate_batch,
    deduplicate_stream,
    iter_matches_stream,
)
from nvalchemi.data import AtomicData, Batch  # noqa: E402


def _structures(lengths: list[float]) -> Batch:
    return Batch.from_data_list(
        [
            AtomicData(
                positions=torch.tensor(
                    [[float(i) * 5, 0.0, 0.0], [float(i) * 5 + length, 0.0, 0.0]],
                    dtype=torch.float32,
                ),
                atomic_numbers=torch.ones(2, dtype=torch.int64),
            )
            for i, length in enumerate(lengths)
        ]
    )


def _feature_rich_structures() -> tuple[Batch, torch.Tensor]:
    coordinate_rows = (
        (0.0, 1.0, 2.1, 3.5),
        (0.0, 1.02, 2.1, 3.5),
        (0.0, 1.06, 2.1, 3.5),
        (0.0, 1.14, 2.1, 3.5),
        (0.0, 1.0, 2.16, 3.5),
        (0.0, 1.0, 2.2, 3.5),
        (0.0, 1.0, 2.1, 3.5),
    )
    batch = Batch.from_data_list(
        [
            AtomicData(
                positions=torch.tensor(
                    [[x, 0.0, 0.0] for x in row], dtype=torch.float32
                ),
                atomic_numbers=torch.ones(4, dtype=torch.int64),
            )
            for row in coordinate_rows
        ]
    )
    types = torch.tensor([1, 2, 3, 4] * len(coordinate_rows), dtype=torch.int32)
    return batch, types


def _variable_structures() -> tuple[Batch, torch.Tensor]:
    coordinates = (
        [0, 1],
        [0, 1.02, 3.1],
        [0, 0.8, 2, 5],
        [0],
        [0, 1.1, 2.4, 4, 7],
        [0, 1, 2.1],
    )
    type_rows = ([1, 1], [1, 2, 2], [2, 2, 1, 1], [2], [1, 2, 1, 2, 2], [1, 2, 1])
    structures = [
        AtomicData(
            positions=torch.tensor([[x, 0.0, 0.0] for x in row], dtype=torch.float32),
            atomic_numbers=torch.ones(len(row), dtype=torch.int64),
        )
        for row in coordinates
    ]
    return Batch.from_data_list(structures), torch.tensor(
        [type_id for row in type_rows for type_id in row], dtype=torch.int32
    )


def _typed_loader(batch: Batch, types: torch.Tensor):
    def read(ids: torch.Tensor) -> tuple[Batch, torch.Tensor]:
        selected = batch.index_select(ids.tolist())
        atom_ids = [
            atom
            for logical_id in ids.tolist()
            for atom in range(
                int(batch.batch_ptr[logical_id]), int(batch.batch_ptr[logical_id + 1])
            )
        ]
        return selected, types[atom_ids]

    return read


def _direct_greedy(
    batch: Batch,
    types: torch.Tensor,
    order: list[int],
    cutoff: float,
    threshold: float,
) -> tuple[list[int], list[int], list[int]]:
    retained: list[int] = []
    representatives = [-1] * batch.num_graphs
    multiplicities: list[int] = []
    endpoint = torch.nextafter(
        torch.tensor(threshold, dtype=torch.float32), torch.tensor(float("inf"))
    )
    for candidate in order:
        chosen = None
        for representative in retained:
            pair_batch = batch.index_select([candidate, representative])
            atom_ids = [
                atom
                for row in (candidate, representative)
                for atom in range(
                    int(batch.batch_ptr[row]), int(batch.batch_ptr[row + 1])
                )
            ]
            pair_types = types[atom_ids]
            pairs = torch.tensor([[0, 1]], dtype=torch.int32)
            keep = True
            for typed_neighbors in (None, False, True):
                index = RadialComparisonIndex.build(
                    pair_batch,
                    cutoff=cutoff,
                    atom_types=pair_types if typed_neighbors is not None else None,
                    typed_neighbors=typed_neighbors is True,
                    device="cpu",
                )
                keep = bool((index.score_pairs(pairs) <= endpoint).item())
                if not keep:
                    break
            if keep:
                chosen = representative
                break
        if chosen is None:
            retained.append(candidate)
            representatives[candidate] = candidate
            multiplicities.append(1)
        else:
            representatives[candidate] = chosen
            multiplicities[retained.index(chosen)] += 1
    return retained, representatives, multiplicities


def _public_score_greedy(
    batch: Batch,
    types: torch.Tensor,
    order: list[int],
    cutoff: float,
    threshold: float,
    device: str = "cpu",
) -> tuple[list[int], list[int], list[int]]:
    """Resolve ordered greedy results from public raw scores for all pairs."""
    logical_pairs = [
        (candidate, earlier)
        for candidate_rank, candidate in enumerate(order)
        for earlier in order[:candidate_rank]
    ]
    pair_tensor = torch.tensor(logical_pairs, dtype=torch.int32, device=device).reshape(
        -1, 2
    )
    accepted_endpoint = torch.nextafter(
        torch.tensor(threshold, dtype=torch.float32, device=device),
        torch.tensor(float("inf"), dtype=torch.float32, device=device),
    )
    pair_matches = torch.ones(len(logical_pairs), dtype=torch.bool, device=device)
    for atom_types, typed_neighbors in (
        (None, False),
        (types, False),
        (types, True),
    ):
        index = RadialComparisonIndex.build(
            batch,
            cutoff=cutoff,
            atom_types=atom_types,
            typed_neighbors=typed_neighbors,
            device=device,
        )
        scores = index.score_pairs(pair_tensor)
        pair_matches &= scores <= accepted_endpoint
    matches = {
        pair: bool(match)
        for pair, match in zip(logical_pairs, pair_matches.tolist(), strict=True)
    }

    retained: list[int] = []
    representatives = [-1] * batch.num_graphs
    multiplicities: list[int] = []
    for candidate in order:
        representative = next(
            (earlier for earlier in retained if matches[(candidate, earlier)]),
            None,
        )
        if representative is None:
            retained.append(candidate)
            representatives[candidate] = candidate
            multiplicities.append(1)
        else:
            representatives[candidate] = representative
            multiplicities[retained.index(representative)] += 1
    return retained, representatives, multiplicities


def _mixed_pbc_batch(
    lengths: list[float], pbc_rows: list[tuple[bool, bool, bool]]
) -> Batch:
    """Build separate two-atom structures with mixed periodic dimensions."""
    cell = torch.diag(torch.tensor([4.0, 5.0, 6.0], dtype=torch.float32)).unsqueeze(0)
    return Batch.from_data_list(
        [
            AtomicData(
                positions=torch.tensor(
                    [[0.0, 0.0, 0.0], [length, 0.0, 0.0]], dtype=torch.float32
                ),
                atomic_numbers=torch.ones(2, dtype=torch.int64),
                cell=cell,
                pbc=torch.tensor([pbc], dtype=torch.bool),
            )
            for length, pbc in zip(lengths, pbc_rows, strict=True)
        ]
    )


def _exhaustive_match_pairs(
    left: Batch,
    right: Batch,
    cutoff: float,
    threshold: float,
    *,
    self_comparison: bool,
    left_types: torch.Tensor | None = None,
    right_types: torch.Tensor | None = None,
) -> list[tuple[int, int]]:
    """Score every possible pair through the public comparison API."""
    all_pairs = [
        (i, j)
        for i in range(left.num_graphs)
        for j in range(right.num_graphs)
        if not self_comparison or i < j
    ]
    if not all_pairs:
        return []
    pairs = torch.tensor(all_pairs, dtype=torch.int32)
    endpoint = torch.nextafter(
        torch.tensor(threshold, dtype=torch.float32), torch.tensor(float("inf"))
    )
    modes: list[tuple[torch.Tensor | None, torch.Tensor | None, bool]] = [
        (None, None, False)
    ]
    if left_types is not None or right_types is not None:
        assert left_types is not None and right_types is not None
        modes.extend(
            ((left_types, right_types, False), (left_types, right_types, True))
        )
    accepted = torch.ones(len(all_pairs), dtype=torch.bool)
    for left_labels, right_labels, typed_neighbors in modes:
        left_index = RadialComparisonIndex.build(
            left,
            cutoff=cutoff,
            atom_types=left_labels,
            typed_neighbors=typed_neighbors,
            device="cpu",
        )
        right_index = (
            left_index
            if self_comparison
            else RadialComparisonIndex.build(
                right,
                cutoff=cutoff,
                atom_types=right_labels,
                typed_neighbors=typed_neighbors,
                device="cpu",
            )
        )
        accepted &= left_index.score_pairs(pairs, other=right_index) <= endpoint
    return [
        pair for pair, keep in zip(all_pairs, accepted.tolist(), strict=True) if keep
    ]


def _literal_confirmed_greedy(
    batch: Batch,
    types: torch.Tensor | None,
    order: list[int],
    cutoff: float,
    threshold: float,
    accepts,
) -> tuple[list[int], list[int], list[int], dict[int, tuple[int, ...]]]:
    """Apply a pair predicate to exhaustive public scores in literal priority order."""
    radial_matches = set(
        _exhaustive_match_pairs(
            batch,
            batch,
            cutoff,
            threshold,
            self_comparison=True,
            left_types=types,
            right_types=types,
        )
    )
    retained: list[int] = []
    representatives = [-1] * batch.num_graphs
    multiplicities: list[int] = []
    prior_retained: dict[int, tuple[int, ...]] = {}
    for candidate in order:
        prior_retained[candidate] = tuple(retained)
        representative = next(
            (
                earlier
                for earlier in retained
                if (min(candidate, earlier), max(candidate, earlier)) in radial_matches
                and accepts(candidate, earlier)
            ),
            None,
        )
        if representative is None:
            retained.append(candidate)
            representatives[candidate] = candidate
            multiplicities.append(1)
        else:
            representatives[candidate] = representative
            multiplicities[retained.index(representative)] += 1
    return retained, representatives, multiplicities, prior_retained


def _run(
    batch: Batch,
    types: torch.Tensor | None = None,
    *,
    order: list[int] | None = None,
    threshold: float = 0.06,
    device: str = "cpu",
    input_batch_size: int = 3,
    pair_block_size: int = 2,
):
    if types is None:
        types = torch.ones(batch.num_nodes, dtype=torch.int32)
    count = batch.num_graphs
    return deduplicate_stream(
        count,
        _typed_loader(batch, types),
        type_vocabulary=sorted(set(types.tolist())),
        cutoff=2.0,
        threshold=threshold,
        priority_order=order,
        device=device,
        input_batch_size=input_batch_size,
        pair_block_size=pair_block_size,
    )


def test_stream_matches_exhaustive_greedy_and_is_nontransitive() -> None:
    batch = _structures([1.0, 1.05, 1.1025, 1.1025, 1.30])
    types = torch.ones(batch.num_nodes, dtype=torch.int32)
    expected = _direct_greedy(batch, types, list(range(5)), 2.0, 0.06)
    result = _run(batch, types)
    assert result.retained_indices.tolist() == expected[0] == [0, 2, 4]
    assert result.representative_indices.tolist() == expected[1] == [0, 0, 2, 2, 4]
    assert result.multiplicities.tolist() == expected[2] == [2, 2, 1]
    stats = _get_last_stream_stats()
    assert len(stats["selected_coordinate_ids"]) == 14
    assert stats["timing_semantics"].startswith("host wall time")
    assert len(stats["survivor_counts_by_chunk"]) == 2


def test_priority_order_controls_ties_and_default_order_is_logical() -> None:
    batch = _structures([1.0, 1.05, 1.1025])
    expected = _direct_greedy(
        batch, torch.ones(6, dtype=torch.int32), [2, 1, 0], 2.0, 0.06
    )
    result = _run(batch, order=[2, 1, 0], input_batch_size=2)
    assert result.retained_indices.tolist() == expected[0] == [2, 0]
    assert result.representative_indices.tolist() == expected[1] == [0, 2, 2]
    assert result.multiplicities.tolist() == expected[2] == [2, 1]
    assert _run(_structures([1.0, 1.0])).retained_indices.tolist() == [0]


def test_grouped_old_matches_choose_first_retained_representative() -> None:
    batch = _structures([1.0, 1.1025, 1.05])
    types = torch.ones(batch.num_nodes, dtype=torch.int32)
    expected = _direct_greedy(batch, types, [0, 1, 2], 2.0, 0.06)
    result = _run(batch, types, input_batch_size=3, pair_block_size=2)
    assert expected == ([0, 1], [0, 1, 0], [2, 1])
    assert result.retained_indices.tolist() == expected[0]
    assert result.representative_indices.tolist() == expected[1]
    assert result.multiplicities.tolist() == expected[2]


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_active_candidate_descriptors_are_reused_across_rep_blocks(
    monkeypatch, device: str
) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    batch = _structures([2.0, 3.0, 4.0, 1.0, 1.0, 1.1])
    types = torch.ones(batch.num_nodes, dtype=torch.int32)

    def controlled_shortlists(candidate_rank, retained, *args, **kwargs):
        if candidate_rank < 4:
            return [], 0, 0, 0
        return retained.copy(), len(retained), len(retained), len(retained)

    def controlled_batch_shortlists(candidate_ids, candidate_ranks, retained, *args):
        ranks = candidate_ranks.detach().cpu().tolist()
        candidates = [[] if rank < 4 else retained.copy() for rank in ranks]
        counts = [0 if rank < 4 else len(retained) for rank in ranks]
        return candidates, counts, counts.copy(), counts.copy()

    monkeypatch.setattr(
        stream_module, "_candidate_representatives", controlled_shortlists
    )
    monkeypatch.setattr(
        stream_module,
        "_candidate_representatives_batch",
        controlled_batch_shortlists,
    )
    monkeypatch.setattr(
        stream_module,
        "_summary_pair_may_match",
        lambda *args, **kwargs: False,
    )
    result = deduplicate_stream(
        batch.num_graphs,
        _typed_loader(batch, types),
        type_vocabulary=[1],
        cutoff=5.0,
        threshold=0.0,
        device=device,
        input_batch_size=4,
        pair_block_size=2,
        # Candidate and representative tiles each contain four atoms. Their
        # combined eight atoms exceed this per-tile cap.
        max_batch_atoms=4,
    )

    expected = _direct_greedy(batch, types, list(range(batch.num_graphs)), 5.0, 0.0)
    assert result.retained_indices.tolist() == expected[0]
    assert result.representative_indices.tolist() == expected[1]
    # Two candidate structures are built once, then each two-row rep block once.
    # Rebuilding candidate endpoints in both rep blocks would build eight rows.
    stats = _get_last_stream_stats()
    assert stats["descriptor_structures_built"] == 6
    assert stats["descriptor_rebuilds"] == 3


@pytest.mark.parametrize(
    ("device", "input_batch_size"),
    [
        ("cpu", 1024),
        pytest.param(
            "cuda",
            None,
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="requires CUDA"
            ),
        ),
    ],
)
def test_input_chunk_uses_input_batch_size_not_pair_block_size(
    device: str, input_batch_size: int | None
) -> None:
    count = 300
    batch = Batch.from_data_list(
        [
            AtomicData(
                positions=torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32),
                atomic_numbers=torch.ones(1, dtype=torch.int64),
            )
            for _ in range(count)
        ]
    )
    types = torch.ones(count, dtype=torch.int32)

    result = deduplicate_stream(
        count,
        _typed_loader(batch, types),
        type_vocabulary=[1],
        cutoff=2.0,
        threshold=0.0,
        device=device,
        input_batch_size=input_batch_size,
        pair_block_size=256,
    )

    # Every singleton has the same untyped, center-typed, and fully typed
    # descriptor. Greedy replay therefore retains row zero for the full chunk.
    assert result.retained_indices.tolist() == [0]
    assert result.representative_indices.tolist() == [0] * count
    assert result.multiplicities.tolist() == [count]
    assert len(_get_last_stream_stats()["survivor_counts_by_chunk"]) == 1


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_active_candidate_tile_can_exceed_representative_block(
    device: str,
) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    input_batch_size = 300
    pair_block_size = 256
    count = 2 * input_batch_size
    batch = Batch.from_data_list(
        [
            AtomicData(
                positions=torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32),
                atomic_numbers=torch.ones(1, dtype=torch.int64),
            )
            for _ in range(count)
        ]
    )
    types = torch.ones(count, dtype=torch.int32)

    result = deduplicate_stream(
        count,
        _typed_loader(batch, types),
        type_vocabulary=[1],
        cutoff=2.0,
        threshold=0.0,
        device=device,
        input_batch_size=input_batch_size,
        pair_block_size=pair_block_size,
    )

    # The first 300 rows form one retained representative; the next 300 are
    # scored in one candidate tile against that representative. This exact
    # scalar oracle is independent of candidate/reps block scheduling.
    assert result.retained_indices.tolist() == [0]
    assert result.representative_indices.tolist() == [0] * count
    assert result.multiplicities.tolist() == [count]
    stats = _get_last_stream_stats()
    # Within-chunk screening reads its 300-row union once. The second chunk
    # builds a 300-row candidate tile and one representative structure.
    assert stats["descriptor_structures_built"] == 2 * input_batch_size + 1
    assert stats["descriptor_rebuilds"] == 3


def test_active_candidate_tile_capacity_failure_splits_without_dropping_rows(
    monkeypatch,
) -> None:
    batch = _structures([2.0, 3.0, 4.0, 1.0, 1.0, 1.1])
    types = torch.ones(batch.num_nodes, dtype=torch.int32)
    monkeypatch.setattr(
        stream_module,
        "_candidate_representatives",
        lambda rank, retained, *args, **kwargs: (
            [] if rank < 4 else retained.copy(),
            0 if rank < 4 else len(retained),
            0 if rank < 4 else len(retained),
            0 if rank < 4 else len(retained),
        ),
    )
    monkeypatch.setattr(
        stream_module,
        "_summary_pair_may_match",
        lambda *args, **kwargs: False,
    )
    original_build = stream_module._build_mode_indices
    failed = False

    def fail_first_multirow_tile(active_batch, *args, **kwargs):
        nonlocal failed
        if active_batch.num_graphs > 1 and not failed:
            failed = True
            raise MemoryError("injected descriptor capacity failure")
        return original_build(active_batch, *args, **kwargs)

    monkeypatch.setattr(stream_module, "_build_mode_indices", fail_first_multirow_tile)
    result = deduplicate_stream(
        batch.num_graphs,
        _typed_loader(batch, types),
        type_vocabulary=[1],
        cutoff=5.0,
        threshold=0.0,
        input_batch_size=4,
        pair_block_size=2,
        max_batch_atoms=8,
    )

    expected = _direct_greedy(batch, types, list(range(batch.num_graphs)), 5.0, 0.0)
    assert failed
    assert result.retained_indices.tolist() == expected[0]
    assert result.representative_indices.tolist() == expected[1]
    assert result.multiplicities.tolist() == expected[2]


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_disjoint_tile_type_vocabularies_use_exact_typed_screen(
    monkeypatch, device: str
) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    batch = _structures([1.0, 1.0])
    types = torch.tensor([1, 1, 2, 2], dtype=torch.int32)
    monkeypatch.setattr(
        stream_module,
        "_candidate_representatives",
        lambda rank, retained, *args, **kwargs: (
            retained.copy() if rank else [],
            len(retained),
            len(retained),
            len(retained),
        ),
    )

    result = deduplicate_stream(
        batch.num_graphs,
        _typed_loader(batch, types),
        type_vocabulary=[1, 2],
        cutoff=2.0,
        threshold=0.0,
        device=device,
        input_batch_size=1,
        pair_block_size=1,
    )

    # The candidate and representative tiles have local vocabularies {2} and
    # {1}. Their summaries cannot be aligned, so the exact center-type screen
    # determines that the pair has no match.
    assert result.retained_indices.tolist() == [0, 1]
    assert result.representative_indices.tolist() == [0, 1]
    assert result.multiplicities.tolist() == [1, 1]


@pytest.mark.parametrize(
    ("lengths", "input_batch_size", "pair_block_size"),
    [
        ([1.0, 1.3, 1.6, 1.0, 1.3], 3, 1),
        ([1.0, 1.4, 1.0, 1.4], 2, 2),
    ],
    ids=["sparse-disjoint-blocks", "dense-shared-block"],
)
def test_adaptive_old_shortlist_schedule_preserves_public_greedy(
    lengths: list[float],
    input_batch_size: int,
    pair_block_size: int,
) -> None:
    batch = _structures(lengths)
    types = torch.ones(batch.num_nodes, dtype=torch.int32)
    expected = _public_score_greedy(
        batch, types, list(range(batch.num_graphs)), 2.0, 0.06
    )

    result = deduplicate_stream(
        batch.num_graphs,
        _typed_loader(batch, types),
        type_vocabulary=[1],
        cutoff=2.0,
        threshold=0.06,
        input_batch_size=input_batch_size,
        pair_block_size=pair_block_size,
    )

    assert result.retained_indices.tolist() == expected[0]
    assert result.representative_indices.tolist() == expected[1]
    assert result.multiplicities.tolist() == expected[2]
    stats = _get_last_stream_stats()
    candidate_chunk_size = input_batch_size
    assert len(stats["survivor_counts_by_chunk"]) == math.ceil(
        len(lengths) / candidate_chunk_size
    )


def test_compact_shortlist_groups_preserve_first_match_across_boundaries() -> None:
    order = [13, 0, 15, 2, 12, 1, 16, 4, 10, 14, 3, 8, 6, 5, 11, 9, 7]
    priority_rows = [
        (1, 3.0),
        (2, None),
        (3, None),
        (1, 2.0),
        (4, None),
        (5, None),
        (1, 0.5),
        (6, None),
        (7, None),
        (1, 1.0),
        (8, None),
        (9, None),
        (1, 1.1025),
        (1, 3.0),
        (1, 1.05),
        (1, 0.5),
        (1, 2.0),
    ]
    structures_by_id: list[AtomicData | None] = [None] * len(order)
    types_by_id: list[list[int] | None] = [None] * len(order)
    for logical_id, (type_id, length) in zip(order, priority_rows, strict=True):
        if length is None:
            positions = [[0.0, 0.0, 0.0]]
            atom_types = [type_id]
        else:
            positions = [[0.0, 0.0, 0.0], [length, 0.0, 0.0]]
            atom_types = [type_id, type_id]
        structures_by_id[logical_id] = AtomicData(
            positions=torch.tensor(positions, dtype=torch.float32),
            atomic_numbers=torch.ones(len(atom_types), dtype=torch.int64),
        )
        types_by_id[logical_id] = atom_types
    batch = Batch.from_data_list(
        [structure for structure in structures_by_id if structure is not None]
    )
    types = torch.tensor(
        [type_id for row in types_by_id for type_id in row or []],
        dtype=torch.int32,
    )
    cutoff = 4.0
    threshold = 0.06
    expected = _public_score_greedy(batch, types, order, cutoff, threshold)

    accepting_pairs = torch.tensor([[11, 14], [11, 6]], dtype=torch.int32)
    typed_index = RadialComparisonIndex.build(
        batch,
        cutoff=cutoff,
        atom_types=types,
        typed_neighbors=True,
        device="cpu",
    )
    accepted_endpoint = torch.nextafter(
        torch.tensor(threshold, dtype=torch.float32),
        torch.tensor(float("inf"), dtype=torch.float32),
    )
    assert torch.all(typed_index.score_pairs(accepting_pairs) <= accepted_endpoint)

    result = deduplicate_stream(
        batch.num_graphs,
        _typed_loader(batch, types),
        type_vocabulary=list(range(1, 10)),
        cutoff=cutoff,
        threshold=threshold,
        priority_order=order,
        input_batch_size=13,
        pair_block_size=2,
        summary_coordinate_count=128,
    )

    assert result.retained_indices.tolist() == expected[0]
    assert result.representative_indices.tolist() == expected[1]
    assert result.multiplicities.tolist() == expected[2]
    assert result.representative_indices[11].item() == 14
    stats = _get_last_stream_stats()
    assert stats["old_shortlist_hit_blocks"] > 0
    assert stats["old_shortlist_compact_groups"] > 0
    assert stats["old_shortlist_sequential_blocks_predicted"] > 0


def test_summary_coordinate_width_preserves_ordered_greedy_results() -> None:
    batch, types = _feature_rich_structures()
    order = [6, 1, 4, 0, 3, 2, 5]
    expected = _public_score_greedy(batch, types, order, 4.0, 0.08)

    for width in (32, 64, 128):
        result = deduplicate_stream(
            batch.num_graphs,
            _typed_loader(batch, types),
            type_vocabulary=[1, 2, 3, 4],
            cutoff=4.0,
            threshold=0.08,
            priority_order=order,
            input_batch_size=4,
            pair_block_size=2,
            summary_coordinate_count=width,
        )
        assert result.retained_indices.tolist() == expected[0]
        assert result.representative_indices.tolist() == expected[1]
        assert result.multiplicities.tolist() == expected[2]
        stats = _get_last_stream_stats()
        assert stats["requested_summary_coordinate_count"] == width
        assert stats["selected_summary_coordinate_count"] == width
        assert len(stats["selected_coordinate_ids"]) == width


@pytest.mark.parametrize("summary_coordinate_count", [129, True, 1.5, "32", None])
def test_invalid_summary_coordinate_counts_are_rejected(
    summary_coordinate_count: object,
) -> None:
    with pytest.raises(ValueError, match="summary_coordinate_count"):
        deduplicate_stream(
            0,
            lambda _: (Batch.from_data_list([]), torch.empty(0, dtype=torch.int32)),
            type_vocabulary=[],
            cutoff=2.0,
            threshold=0.0,
            summary_coordinate_count=summary_coordinate_count,
        )


def test_zero_summary_coordinates_deduplicate_matches_exhaustive() -> None:
    batch = _structures([1.0, 1.05, 1.1025, 1.30, 1.02])
    types = torch.ones(batch.num_nodes, dtype=torch.int32)
    order = [3, 1, 4, 0, 2]
    expected = _public_score_greedy(batch, types, order, 2.0, 0.06)

    result = deduplicate_stream(
        batch.num_graphs,
        _typed_loader(batch, types),
        type_vocabulary=[1],
        cutoff=2.0,
        threshold=0.06,
        priority_order=order,
        input_batch_size=2,
        pair_block_size=2,
        summary_coordinate_count=0,
    )
    assert result.retained_indices.tolist() == expected[0]
    assert result.representative_indices.tolist() == expected[1]
    assert result.multiplicities.tolist() == expected[2]

    batched = deduplicate_batch(
        batch,
        atom_types=types,
        cutoff=2.0,
        threshold=0.06,
        priority_order=order,
        summary_coordinate_count=0,
    )
    assert batched.retained_indices.tolist() == expected[0]
    assert batched.representative_indices.tolist() == expected[1]
    assert batched.multiplicities.tolist() == expected[2]


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="requires CUDA"
            ),
        ),
    ],
)
def test_untyped_zero_summary_dedup_matches_default_and_exhaustive(
    device: str,
) -> None:
    batch = _structures([1.0] * 24)
    order = list(reversed(range(batch.num_graphs)))
    cutoff = 2.0
    threshold = 0.06
    matching_pairs = set(
        _exhaustive_match_pairs(
            batch,
            batch,
            cutoff,
            threshold,
            self_comparison=True,
        )
    )
    matching_pairs |= {(right, left) for left, right in matching_pairs}
    expected_retained: list[int] = []
    expected_representatives = [-1] * batch.num_graphs
    expected_multiplicities: list[int] = []
    for candidate in order:
        representative = next(
            (
                earlier
                for earlier in expected_retained
                if (candidate, earlier) in matching_pairs
            ),
            None,
        )
        if representative is None:
            expected_retained.append(candidate)
            expected_representatives[candidate] = candidate
            expected_multiplicities.append(1)
        else:
            expected_representatives[candidate] = representative
            expected_multiplicities[expected_retained.index(representative)] += 1

    def read_untyped(ids: torch.Tensor) -> tuple[Batch, None]:
        return batch.index_select(ids.tolist()), None

    arguments = dict(
        cutoff=cutoff,
        threshold=threshold,
        priority_order=order,
        input_batch_size=4,
        pair_block_size=2,
        device=device,
    )
    default = deduplicate_stream(batch.num_graphs, read_untyped, **arguments)
    exhaustive = deduplicate_stream(
        batch.num_graphs,
        read_untyped,
        summary_coordinate_count=0,
        **arguments,
    )
    for result in (default, exhaustive):
        assert result.retained_indices.tolist() == expected_retained
        assert result.representative_indices.tolist() == expected_representatives
        assert result.multiplicities.tolist() == expected_multiplicities


def test_summary_retrieval_has_no_candidate_cap() -> None:
    row_count = 130
    features = torch.zeros((row_count, 1), dtype=torch.float32)
    presence = torch.ones((row_count, 1), dtype=torch.bool)
    retained_mask = torch.zeros(row_count, dtype=torch.bool)
    retained_mask[: row_count - 1] = True
    rank_by_id = torch.arange(row_count, dtype=torch.int64)
    sorted_values = features.T.contiguous()
    sorted_ids = torch.arange(row_count, dtype=torch.int32).reshape(1, -1)
    candidates, *_ = _candidate_representatives(
        row_count - 1,
        list(range(row_count - 1)),
        retained_mask,
        rank_by_id,
        features,
        presence,
        torch.tensor([0], dtype=torch.int64),
        sorted_values,
        sorted_ids,
        torch.tensor([True]),
        0.0,
    )
    assert candidates == list(range(row_count - 1))


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_batched_summary_search_ignores_inactive_coordinate_sentinel(
    device: str,
) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    target = torch.device(device)
    row_count = 100
    features = torch.stack(
        (torch.zeros(row_count), torch.arange(row_count, dtype=torch.float32)), dim=1
    ).to(target)
    presence = torch.tensor([[True, False]] * row_count, device=target)
    retained = list(range(1, row_count))
    retained_mask = torch.zeros(row_count, dtype=torch.bool, device=target)
    retained_mask[retained] = True
    rank_by_id = torch.arange(row_count, dtype=torch.int64, device=target)
    sorted_values, sorted_ids = [], []
    for coordinate in range(features.shape[1]):
        order = torch.argsort(features[:, coordinate], stable=True)
        sorted_values.append(features[order, coordinate])
        sorted_ids.append(order.to(torch.int32))

    candidates, interval_counts, retained_counts, full_counts = (
        _candidate_representatives_batch(
            [0],
            torch.tensor([0], dtype=torch.int64, device=target),
            retained,
            retained_mask,
            rank_by_id,
            features,
            presence,
            torch.tensor([0, 1], dtype=torch.int64, device=target),
            torch.stack(sorted_values),
            torch.stack(sorted_ids),
            torch.tensor([True, True], device=target),
            0.0,
        )
    )

    assert candidates == [retained]
    assert interval_counts == [row_count]
    assert retained_counts == [len(retained)]
    assert full_counts == [len(retained)]


def test_old_match_priority_uses_rank_when_group_results_are_reversed(monkeypatch):
    batch = _structures([1.1025, 1.05, 1.0])
    types = torch.ones(batch.num_nodes, dtype=torch.int32)
    order = [2, 0, 1]
    expected = _direct_greedy(batch, types, order, 2.0, 0.06)
    original_screen = stream_module._screen_pair_group

    def reverse_old_group_results(structure_ids, logical_pairs, *args, **kwargs):
        matches = original_screen(structure_ids, logical_pairs, *args, **kwargs)
        if len(logical_pairs) == 2:
            return list(reversed(matches))
        return matches

    monkeypatch.setattr(stream_module, "_screen_pair_group", reverse_old_group_results)
    result = _run(
        batch,
        types,
        order=order,
        input_batch_size=2,
        pair_block_size=2,
    )

    assert expected == ([2, 0], [0, 2, 2], [2, 1])
    assert result.retained_indices.tolist() == expected[0]
    assert result.representative_indices.tolist() == expected[1]
    assert result.multiplicities.tolist() == expected[2]


def test_nontransitive_chain_crosses_candidate_chunk_boundary() -> None:
    structures = [
        AtomicData(
            positions=torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32),
            atomic_numbers=torch.ones(1, dtype=torch.int64),
        )
        for _ in range(30)
    ]
    atom_types = list(range(2, 32))
    for length in (1.0, 1.05, 1.1025):
        structures.append(
            AtomicData(
                positions=torch.tensor(
                    [[0.0, 0.0, 0.0], [length, 0.0, 0.0]], dtype=torch.float32
                ),
                atomic_numbers=torch.ones(2, dtype=torch.int64),
            )
        )
        atom_types.extend((1, 1))
    batch = Batch.from_data_list(structures)
    types = torch.tensor(atom_types, dtype=torch.int32)

    result = _run(batch, types, input_batch_size=32, pair_block_size=32)

    assert result.retained_indices.tolist() == [*range(31), 32]
    assert result.representative_indices.tolist() == [
        *range(30),
        30,
        30,
        32,
    ]
    assert result.multiplicities.tolist() == [*([1] * 30), 2, 1]
    stats = _get_last_stream_stats()
    assert stats["survivor_growth_by_chunk"] == [31, 1]


def test_all_duplicates_in_one_candidate_chunk_use_one_pair_group() -> None:
    count = 32
    result = _run(
        _structures([1.0] * count),
        input_batch_size=64,
        pair_block_size=32,
    )
    pair_count = count * (count - 1) // 2
    assert result.retained_indices.tolist() == [0]
    assert result.representative_indices.tolist() == [0] * count
    assert result.multiplicities.tolist() == [count]
    stats = _get_last_stream_stats()
    assert stats["screen_pairs_untyped"] == pair_count
    assert stats["screen_pairs_center_typed"] == pair_count
    assert stats["screen_pairs_fully_typed"] == pair_count
    assert stats["descriptor_rebuilds"] == 1
    assert stats["descriptor_structures_built"] == count
    assert stats["old_shortlisted_pairs"] == 0
    assert stats["within_chunk_pairs_considered"] == pair_count
    assert stats["within_chunk_pairs_shortlisted"] == pair_count
    assert stats["within_chunk_pairs_rejected_by_features"] == 0


def test_sparse_chunk_summary_filter_preserves_near_bound_match() -> None:
    cutoff = 5.0
    lengths = [1.0, 1.05] + [
        0.5 * (1.06**exponent) for exponent in range(32) if exponent not in (12, 13)
    ]
    batch = _structures(lengths)
    types = torch.ones(batch.num_nodes, dtype=torch.int32)
    score_index = RadialComparisonIndex.build(
        batch,
        cutoff=cutoff,
        atom_types=types,
        typed_neighbors=True,
        device="cpu",
    )
    threshold = float(score_index.score_pairs(torch.tensor([[0, 1]]))[0])
    expected = _public_score_greedy(
        batch, types, list(range(batch.num_graphs)), cutoff, threshold
    )

    result = deduplicate_stream(
        batch.num_graphs,
        _typed_loader(batch, types),
        type_vocabulary=[1],
        cutoff=cutoff,
        threshold=threshold,
        input_batch_size=32,
        pair_block_size=32,
    )

    assert result.retained_indices.tolist() == expected[0]
    assert result.representative_indices.tolist() == expected[1]
    assert result.multiplicities.tolist() == expected[2]
    assert result.representative_indices[1].item() == 0
    stats = _get_last_stream_stats()
    assert stats["within_chunk_pairs_considered"] == 32 * 31 // 2
    assert stats["within_chunk_pairs_shortlisted"] == 1
    assert stats["within_chunk_pairs_rejected_by_features"] == (
        stats["within_chunk_pairs_considered"] - stats["within_chunk_pairs_shortlisted"]
    )
    assert stats["screen_pairs_untyped"] == 1
    assert stats["screen_pairs_center_typed"] == 1
    assert stats["screen_pairs_fully_typed"] == 1


def test_descriptor_capacity_error_splits_pair_tile(monkeypatch) -> None:
    batch = _structures([1.0, 1.05, 1.1025, 1.1025])
    types = torch.ones(batch.num_nodes, dtype=torch.int32)
    expected = _direct_greedy(batch, types, list(range(4)), 2.0, 0.06)
    original_build = stream_module._build_mode_indices
    did_fail = False

    batch_sizes = []

    def fail_group_build_once(build_batch, *args, **kwargs):
        nonlocal did_fail
        batch_sizes.append(build_batch.num_graphs)
        if not did_fail and build_batch.num_graphs >= 2:
            did_fail = True
            raise MemoryError("injected grouped descriptor workspace limit")
        return original_build(build_batch, *args, **kwargs)

    monkeypatch.setattr(stream_module, "_build_mode_indices", fail_group_build_once)
    result = _run(batch, types, input_batch_size=4, pair_block_size=4)
    assert did_fail
    assert result.retained_indices.tolist() == expected[0]
    assert result.representative_indices.tolist() == expected[1]
    assert result.multiplicities.tolist() == expected[2]
    assert max(batch_sizes) >= 3
    assert min(batch_sizes[1:]) < max(batch_sizes)


def test_scoring_capacity_error_splits_pair_tile(monkeypatch) -> None:
    batch = _structures([1.0, 1.0, 1.0, 1.0])
    types = torch.ones(batch.num_nodes, dtype=torch.int32)
    expected = _direct_greedy(batch, types, list(range(4)), 2.0, 0.06)
    original_find_matches = RadialComparisonIndex.find_matches
    did_fail = False

    def fail_group_score_once(index, *args, **kwargs):
        nonlocal did_fail
        pair_indices = kwargs.get("pair_indices")
        if not did_fail and pair_indices is not None and pair_indices.shape[0] == 6:
            did_fail = True
            raise MemoryError("injected grouped score workspace limit")
        return original_find_matches(index, *args, **kwargs)

    monkeypatch.setattr(RadialComparisonIndex, "find_matches", fail_group_score_once)
    result = _run(batch, types, input_batch_size=4, pair_block_size=4)

    assert did_fail
    assert result.retained_indices.tolist() == expected[0]
    assert result.representative_indices.tolist() == expected[1]
    assert result.multiplicities.tolist() == expected[2]
    assert _get_last_stream_stats()["descriptor_rebuilds"] > 0


def test_group_loader_memory_error_propagates_without_group_fallback() -> None:
    batch = _structures([1.0, 1.0, 1.0])
    types = torch.ones(batch.num_nodes, dtype=torch.int32)
    read = _typed_loader(batch, types)
    read_calls = 0

    def fail_group_read(ids: torch.Tensor):
        nonlocal read_calls
        read_calls += 1
        if read_calls == 3:
            raise MemoryError("injected loader allocation failure")
        return read(ids)

    with pytest.raises(MemoryError, match="injected loader allocation failure"):
        deduplicate_stream(
            3,
            fail_group_read,
            type_vocabulary=[1],
            cutoff=2.0,
            threshold=0.0,
            input_batch_size=3,
        )

    assert read_calls == 3
    assert _get_last_stream_stats()["grouped_memory_fallbacks"] == 0


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="requires CUDA"
            ),
        ),
    ],
)
def test_grouped_stage_compaction_preserves_public_greedy_and_type_alignment(
    device: str,
) -> None:
    square = torch.tensor(
        [[0.0, 0, 0], [1.0, 0, 0], [1.0, 1, 0], [0.0, 1, 0]],
        dtype=torch.float32,
    )
    triangle = torch.tensor(
        [[0.0, 0, 0], [0.8, 0, 0], [1.7, 0.1, 0]], dtype=torch.float32
    )
    order = [4, 1, 5, 0, 3, 2]
    priority_rows = [
        (triangle, [1, 1, 1]),
        (square, [1, 1, 1, 1]),
        (square, [1, 2, 1, 2]),
        (square, [1, 1, 2, 2]),
        (square, [1, 1, 2, 2]),
        (square, [1, 1, 2, 2]),
    ]
    structures_by_id: list[AtomicData | None] = [None] * len(order)
    types_by_id: list[list[int] | None] = [None] * len(order)
    for logical_id, (positions, atom_types) in zip(order, priority_rows, strict=True):
        structures_by_id[logical_id] = AtomicData(
            positions=positions,
            atomic_numbers=torch.ones(len(atom_types), dtype=torch.int64),
        )
        types_by_id[logical_id] = atom_types
    batch = Batch.from_data_list(
        [structure for structure in structures_by_id if structure is not None]
    ).to(device)
    types = torch.tensor(
        [type_id for row in types_by_id for type_id in row or []],
        dtype=torch.int32,
        device=device,
    )
    read_batch = _typed_loader(batch, types)
    direct = _public_score_greedy(batch, types, order, 2.0, 0.0, device=device)

    result = deduplicate_stream(
        batch.num_graphs,
        read_batch,
        type_vocabulary=[1, 2],
        cutoff=2.0,
        threshold=0.0,
        priority_order=order,
        device=device,
        input_batch_size=4,
        pair_block_size=4,
    )

    assert result.retained_indices.tolist() == direct[0]
    assert result.representative_indices.tolist() == direct[1]
    assert result.multiplicities.tolist() == direct[2]
    assert result.representative_indices[2].item() == 0


def test_center_typed_screen_rejects_pair_after_untyped_match() -> None:
    batch = _structures([1.0, 1.0])
    types = torch.tensor([1, 1, 1, 2], dtype=torch.int32)
    representative, stats = _screen_candidate(
        1, [0], _typed_loader(batch, types), (1, 2), 2.0, 0.0, torch.device("cpu"), 2
    )
    assert representative is None
    assert stats["screen_pairs_untyped"] == 1
    assert stats["screen_pairs_center_typed"] == 1
    assert stats["screen_pairs_fully_typed"] == 0


def test_fully_typed_screen_rejects_after_untyped_and_center_typed() -> None:
    square = torch.tensor(
        [[0.0, 0, 0], [1.0, 0, 0], [1.0, 1, 0], [0.0, 1, 0]],
        dtype=torch.float32,
    )
    batch = Batch.from_data_list(
        [
            AtomicData(
                positions=square, atomic_numbers=torch.ones(4, dtype=torch.int64)
            ),
            AtomicData(
                positions=square, atomic_numbers=torch.ones(4, dtype=torch.int64)
            ),
        ]
    )
    # The geometries and center-type populations are identical. The A/B
    # assignment changes which neighbor-distance lists are compared by type.
    types = torch.tensor([1, 1, 2, 2, 1, 2, 1, 2], dtype=torch.int32)
    representative, stats = _screen_candidate(
        1, [0], _typed_loader(batch, types), (1, 2), 2.0, 0.0, torch.device("cpu"), 2
    )
    assert representative is None
    assert stats["screen_pairs_untyped"] == 1
    assert stats["screen_pairs_center_typed"] == 1
    assert stats["screen_pairs_fully_typed"] == 1


def test_summary_retrieval_contains_every_raw_three_screen_match() -> None:
    batch, types = _variable_structures()
    loader = _typed_loader(batch, types)
    order = [4, 1, 5, 0, 3, 2]
    vocabulary = (1, 2)
    from nvalchemi.csp._comparison.stream import _build_summaries, _select_coordinates

    pilot_ids, _ = _pilot_ids(order)
    pilot, pilot_presence, _ = _build_summaries(
        pilot_ids, loader, vocabulary, 5.0, torch.device("cpu"), 3
    )
    selected = _select_coordinates(pilot, pilot_presence, 0.08, len(vocabulary))
    all_features, all_presence, _ = _build_summaries(
        order, loader, vocabulary, 5.0, torch.device("cpu"), 3
    )
    features = all_features[:, selected]
    presence = all_presence
    rank_by_id = torch.empty(batch.num_graphs, dtype=torch.int64)
    rank_by_id[torch.tensor(order)] = torch.arange(batch.num_graphs)
    priority_ids = torch.tensor(order, dtype=torch.int32)
    centers = torch.tensor([c // (len(vocabulary) * 14) for c in selected])
    sorted_values = torch.empty((len(selected), batch.num_graphs), dtype=torch.float32)
    sorted_ids = torch.empty((len(selected), batch.num_graphs), dtype=torch.int32)
    for column in range(len(selected)):
        permutation = torch.argsort(features[:, column], stable=True)
        sorted_values[column] = features[permutation, column]
        sorted_ids[column] = priority_ids[permutation]
    typed_indexes = [
        RadialComparisonIndex.build(
            batch,
            cutoff=5.0,
            atom_types=None if mode is None else types,
            typed_neighbors=mode is True,
            device="cpu",
        )
        for mode in (None, False, True)
    ]
    target_pair = torch.tensor([[order[0], order[1]]], dtype=torch.int32)
    target_scores = [
        float(index.score_pairs(target_pair)[0]) for index in typed_indexes
    ]
    finite_scores = [
        score for score in target_scores if torch.isfinite(torch.tensor(score))
    ]
    threshold = max(finite_scores) if finite_scores else 0.08
    accepted_endpoint = torch.nextafter(
        torch.tensor(threshold), torch.tensor(float("inf"))
    )
    for candidate_rank, candidate in enumerate(order):
        retained = order[:candidate_rank]
        retained_mask = torch.zeros(batch.num_graphs, dtype=torch.bool)
        retained_mask[retained] = True
        retrieved, *_ = _candidate_representatives(
            int(rank_by_id[candidate]),
            retained,
            retained_mask,
            rank_by_id,
            features,
            presence,
            centers,
            sorted_values,
            sorted_ids,
            torch.isfinite(features).all(dim=0),
            _conservative_summary_log_bound(threshold),
        )
        accepted = set(retained)
        candidate_pairs = torch.tensor(
            [[candidate, representative] for representative in retained],
            dtype=torch.int32,
        ).reshape(-1, 2)
        if not retained:
            assert not accepted and not retrieved
            continue
        for index in typed_indexes:
            raw = index.score_pairs(candidate_pairs)
            accepted = {
                representative
                for representative, score in zip(retained, raw.tolist(), strict=True)
                if representative in accepted and score <= float(accepted_endpoint)
            }
        assert accepted.issubset(set(retrieved)), (candidate, accepted, retrieved)

    expected = _direct_greedy(batch, types, order, 5.0, threshold)
    result = deduplicate_stream(
        batch.num_graphs,
        loader,
        type_vocabulary=vocabulary,
        cutoff=5.0,
        threshold=threshold,
        priority_order=order,
        input_batch_size=3,
        pair_block_size=2,
    )
    assert result.retained_indices.tolist() == expected[0]
    assert result.representative_indices.tolist() == expected[1]
    assert result.multiplicities.tolist() == expected[2]


def test_fp32_cutoff_padding_agrees_with_public_scores_and_stream() -> None:
    cutoff = 2.37486376976615
    cutoff_fp32 = torch.tensor(cutoff, dtype=torch.float32)
    distance = torch.nextafter(cutoff_fp32, torch.tensor(0.0, dtype=torch.float32))
    batch = Batch.from_data_list(
        [
            AtomicData(
                positions=torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32),
                atomic_numbers=torch.ones(1, dtype=torch.int64),
            ),
            AtomicData(
                positions=torch.tensor(
                    [[0.0, 0.0, 0.0], [distance.item(), 0.0, 0.0]],
                    dtype=torch.float32,
                ),
                atomic_numbers=torch.ones(2, dtype=torch.int64),
            ),
        ]
    )
    atom_types = torch.ones(batch.num_nodes, dtype=torch.int32)
    pair = torch.tensor([[0, 1]], dtype=torch.int32)
    for types, typed_neighbors in (
        (None, False),
        (atom_types, False),
        (atom_types, True),
    ):
        index = RadialComparisonIndex.build(
            batch,
            cutoff=cutoff,
            atom_types=types,
            typed_neighbors=typed_neighbors,
            device="cpu",
        )
        assert index.score_pairs(pair).tolist() == [0.0]
        assert index.find_matches(threshold=0.0, pair_indices=pair).tolist() == [[0, 1]]

    result = deduplicate_stream(
        2,
        _typed_loader(batch, atom_types),
        type_vocabulary=[1],
        cutoff=cutoff,
        threshold=0.0,
        input_batch_size=1,
    )
    assert result.retained_indices.tolist() == [0]
    assert result.representative_indices.tolist() == [0, 0]
    assert result.multiplicities.tolist() == [2]


def test_grouped_old_shortlist_keeps_near_cutoff_match() -> None:
    cutoff = 2.37486376976615
    cutoff_fp32 = torch.tensor(cutoff, dtype=torch.float32)
    distance = torch.nextafter(cutoff_fp32, torch.tensor(0.0, dtype=torch.float32))
    batch = Batch.from_data_list(
        [
            AtomicData(
                positions=torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32),
                atomic_numbers=torch.ones(1, dtype=torch.int64),
            ),
            AtomicData(
                positions=torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32),
                atomic_numbers=torch.ones(1, dtype=torch.int64),
            ),
            *[
                AtomicData(
                    positions=torch.tensor(
                        [[0.0, 0.0, 0.0], [distance.item(), 0.0, 0.0]],
                        dtype=torch.float32,
                    ),
                    atomic_numbers=torch.ones(2, dtype=torch.int64),
                )
                for _ in range(2)
            ],
        ]
    )
    types = torch.tensor([1, 2, 1, 1, 1, 1], dtype=torch.int32)

    result = deduplicate_stream(
        4,
        _typed_loader(batch, types),
        type_vocabulary=[1, 2],
        cutoff=cutoff,
        threshold=0.0,
        input_batch_size=2,
        pair_block_size=2,
    )

    assert result.retained_indices.tolist() == [0, 1]
    assert result.representative_indices.tolist() == [0, 1, 0, 0]
    assert result.multiplicities.tolist() == [3, 1]


@pytest.mark.parametrize("cutoff", [2.37486376976615, 1.9440808525179338])
def test_ragged_descriptor_padding_uses_score_cutoff_log(cutoff: float) -> None:
    cutoff_fp32 = torch.tensor(cutoff, dtype=torch.float32)
    score_log = torch.tensor(math.log(cutoff), dtype=torch.float32)
    descriptor_log = torch.log(cutoff_fp32)
    assert score_log != descriptor_log
    distance = torch.nextafter(cutoff_fp32, torch.tensor(0.0, dtype=torch.float32))
    batch = Batch.from_data_list(
        [
            AtomicData(
                positions=torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32),
                atomic_numbers=torch.ones(1, dtype=torch.int64),
            ),
            AtomicData(
                positions=torch.tensor(
                    [[0.0, 0.0, 0.0], [distance.item(), 0.0, 0.0]],
                    dtype=torch.float32,
                ),
                atomic_numbers=torch.ones(2, dtype=torch.int64),
            ),
        ]
    )
    atom_types = torch.ones(batch.num_nodes, dtype=torch.int32)
    pair = torch.tensor([[0, 1]], dtype=torch.int32)
    for typed_neighbors in (False, True):
        index = RadialComparisonIndex.build(
            batch,
            cutoff=cutoff,
            atom_types=atom_types,
            typed_neighbors=typed_neighbors,
            device="cpu",
        )
        score = float(index.score_pairs(pair)[0])
        assert index.find_matches(threshold=score, pair_indices=pair).tolist() == [
            [0, 1]
        ]

        result = deduplicate_stream(
            2,
            _typed_loader(batch, atom_types),
            type_vocabulary=[1],
            cutoff=cutoff,
            threshold=score,
            input_batch_size=1,
        )
        assert result.retained_indices.tolist() == [0]
        assert result.representative_indices.tolist() == [0, 0]


def test_ragged_typed_summary_normalizes_explicit_padding_slots() -> None:
    cutoff = 3.760784504413605
    cutoff_fp32 = torch.tensor(cutoff, dtype=torch.float32)
    distance = torch.nextafter(cutoff_fp32, torch.tensor(0.0, dtype=torch.float32))
    height = distance * torch.sqrt(torch.tensor(3.0, dtype=torch.float32)) / 2
    triangle = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [distance.item(), 0.0, 0.0],
            [distance.item() / 2, height.item(), 0.0],
        ],
        dtype=torch.float32,
    )
    batch = Batch.from_data_list(
        [
            AtomicData(
                positions=triangle, atomic_numbers=torch.ones(3, dtype=torch.int64)
            ),
            AtomicData(
                positions=torch.cat(
                    (
                        triangle,
                        torch.tensor([[100.0, 100.0, 100.0]], dtype=torch.float32),
                    )
                ),
                atomic_numbers=torch.ones(4, dtype=torch.int64),
            ),
        ]
    )
    atom_types = torch.ones(batch.num_nodes, dtype=torch.int32)
    pair = torch.tensor([[0, 1]], dtype=torch.int32)
    for typed_neighbors in (False, True):
        index = RadialComparisonIndex.build(
            batch,
            cutoff=cutoff,
            atom_types=atom_types,
            typed_neighbors=typed_neighbors,
            device="cpu",
        )
        score = float(index.score_pairs(pair)[0])
        assert index.find_matches(threshold=score, pair_indices=pair).tolist() == [
            [0, 1]
        ]


def test_fp32_outward_interval_contains_adjacent_summary_boundary() -> None:
    q = torch.tensor(0.09531043469905853, dtype=torch.float32)
    y = torch.tensor(-(2.0**-30), dtype=torch.float32)
    bound = 0.09531043469905853
    assert (q - y).abs().item() == bound
    old_low = torch.nextafter(
        q - torch.tensor(bound, dtype=torch.float32),
        torch.tensor(float("-inf")),
    )
    assert y < old_low
    features = torch.stack((y, q)).reshape(2, 1)
    retrieved, *_ = _candidate_representatives(
        candidate_rank=1,
        retained=[0],
        retained_mask=torch.tensor([True, False]),
        rank_by_id=torch.tensor([0, 1]),
        features=features,
        presence=torch.tensor([[True], [True]]),
        coordinate_centers=torch.tensor([0]),
        sorted_values=torch.tensor([[y.item(), q.item()]]),
        sorted_ids=torch.tensor([[0, 1]], dtype=torch.int32),
        coordinate_safe=torch.tensor([True]),
        bound=bound,
    )
    assert retrieved == [0]


def test_vectorized_within_chunk_filter_preserves_public_greedy_oracle() -> None:
    batch, types = _feature_rich_structures()
    order = [4, 0, 6, 1, 5, 2, 3]
    expected = _public_score_greedy(batch, types, order, 2.0, 0.06)
    result = deduplicate_stream(
        batch.num_graphs,
        _typed_loader(batch, types),
        type_vocabulary=[1, 2, 3, 4],
        cutoff=2.0,
        threshold=0.06,
        priority_order=order,
        input_batch_size=len(order),
        pair_block_size=2,
    )

    assert result.retained_indices.tolist() == expected[0]
    assert result.representative_indices.tolist() == expected[1]
    assert result.multiplicities.tolist() == expected[2]
    stats = _get_last_stream_stats()
    assert stats["within_chunk_pairs_considered"] == 21
    assert (
        stats["within_chunk_pairs_shortlisted"]
        + stats["within_chunk_pairs_rejected_by_features"]
        == 21
    )


def test_vectorized_summary_mask_matches_independent_oracle_at_fp32_boundary() -> None:
    zero = torch.tensor(0.0, dtype=torch.float32)
    nominal_bound = torch.tensor(0.25, dtype=torch.float32)
    infinity = torch.tensor(float("inf"), dtype=torch.float32)
    outward_bound = torch.nextafter(nominal_bound, infinity)
    beyond_outward = torch.nextafter(outward_bound, infinity)
    features = torch.tensor(
        [
            [zero.item(), zero.item(), zero.item(), 0.0],
            [outward_bound.item(), zero.item(), zero.item(), 0.0],
            [zero.item(), zero.item(), zero.item(), 0.0],
            [4.0, 4.0, 4.0, 4.0],
            [beyond_outward.item(), zero.item(), zero.item(), 0.0],
        ],
        dtype=torch.float32,
    )
    presence = torch.tensor(
        [
            [True, False],
            [True, False],
            [False, True],
            [False, False],
            [True, False],
        ],
        dtype=torch.bool,
    )
    centers = torch.tensor([0, 1, 0, 1], dtype=torch.int64)
    safe = torch.tensor([True, True, True, False])
    pairs = torch.tensor(
        [[1, 0], [4, 0], [4, 1], [2, 0], [3, 2], [1, 3]],
        dtype=torch.int64,
    )

    def independent_keep(candidate: int, representative: int) -> bool:
        active = [
            center
            for coordinate, center in enumerate(centers.tolist())
            if bool(safe[coordinate]) and bool(presence[candidate, center])
        ]
        if not active:
            return True
        if not torch.equal(presence[candidate], presence[representative]):
            return False
        return all(
            float(
                (
                    features[candidate, coordinate]
                    - features[representative, coordinate]
                ).abs()
            )
            <= float(outward_bound)
            for coordinate in torch.nonzero(safe, as_tuple=False).flatten().tolist()
            if int(centers[coordinate]) in active
        )

    expected = [independent_keep(left, right) for left, right in pairs.tolist()]
    kept = _summary_pairs_may_match_batch(
        pairs,
        features,
        presence,
        centers,
        safe,
        float(nominal_bound),
    )

    assert expected == [True, False, True, False, True, False]
    assert kept.tolist() == expected


def test_nonfinite_selected_summary_falls_back_to_all_retained_rows() -> None:
    retrieved, interval_rows, retained_rows, filtered_rows = _candidate_representatives(
        candidate_rank=1,
        retained=[0],
        retained_mask=torch.tensor([True, False]),
        rank_by_id=torch.tensor([0, 1]),
        features=torch.tensor([[float("inf")], [0.0]]),
        presence=torch.tensor([[True], [True]]),
        coordinate_centers=torch.tensor([0]),
        sorted_values=torch.tensor([[0.0, float("inf")]]),
        sorted_ids=torch.tensor([[0, 1]], dtype=torch.int32),
        coordinate_safe=torch.tensor([False]),
        bound=0.1,
    )
    assert retrieved == [0]
    assert (interval_rows, retained_rows, filtered_rows) == (0, 1, 1)


def test_pilot_reserves_contiguous_spread_training_and_heldout_rows() -> None:
    ids, labels = _pilot_ids(list(range(1000)))
    assert len(ids) == 320
    assert labels[:128].count("contiguous") == 64
    assert labels[:128].count("spread") == 64
    assert labels[128:144].count("contiguous") == 8
    assert labels[128:144].count("spread") == 8
    assert labels[144:160].count("contiguous") == 8
    assert labels[144:160].count("spread") == 8
    assert len(set(ids)) == len(ids)


def test_32_row_pilot_reserves_training_queries_for_informed_selection() -> None:
    ids, labels = _pilot_ids(list(range(32)))
    assert ids == list(range(32))
    assert labels[:16] == ["contiguous"] * 16
    assert labels[16:] == ["contiguous"] * 16

    features = torch.zeros((32, 56), dtype=torch.float32)
    features[16:, 47] = 1.0
    presence = torch.ones((32, 2), dtype=torch.bool)
    selected = _select_coordinates(features, presence, 0.05, 2)
    assert len(selected) == 32
    assert selected[0] == 47
    assert len(set(selected)) == len(selected)


def test_pilot_reports_heldout_selectivity_by_sample_stratum() -> None:
    _, labels = _pilot_ids(list(range(1000)))
    features = torch.arange(320 * 14, dtype=torch.float32).reshape(320, 14) / 1000
    presence = torch.ones((320, 1), dtype=torch.bool)
    measured = _pilot_heldout_selectivity(
        features, presence, labels, list(range(14)), 0.0, 1
    )
    assert measured["contiguous"]["pairs"] > 0
    assert measured["spread"]["pairs"] > 0


def test_fixed_vocabulary_chunk_boundaries_and_no_files(tmp_path) -> None:
    batch = _structures([1.0, 1.0, 1.0])
    full_types = torch.tensor([1, 1, 1, 1, 1, 2], dtype=torch.int32)
    read = _typed_loader(batch, full_types)
    result = deduplicate_stream(
        3,
        read,
        type_vocabulary=[2, 1],
        cutoff=2.0,
        threshold=0.0,
        input_batch_size=1,
    )
    assert result.retained_indices.tolist() == [0, 2]
    assert result.representative_indices.tolist() == [0, 0, 2]
    assert result.multiplicities.tolist() == [2, 1]
    assert list(tmp_path.iterdir()) == []


def test_loader_failure_propagates_without_creating_artifacts(tmp_path) -> None:
    def fail(_ids: torch.Tensor):
        raise RuntimeError("injected loader failure")

    with pytest.raises(RuntimeError, match="injected loader failure"):
        deduplicate_stream(1, fail, type_vocabulary=[1], cutoff=2.0, threshold=0.0)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("operation", ["deduplicate", "matches"])
def test_stream_rejects_int64_atom_type_wraparound_before_narrowing(
    operation: str,
) -> None:
    batch = _structures([1.0])

    def reader(ids: torch.Tensor):
        selected = batch.index_select(ids.tolist())
        return selected, torch.full(
            (selected.num_nodes,), 4_294_967_297, dtype=torch.int64
        )

    with pytest.raises(ValueError, match="fit signed int32"):
        if operation == "deduplicate":
            deduplicate_stream(
                1,
                reader,
                type_vocabulary=[1],
                cutoff=2.0,
                threshold=0.0,
            )
        else:
            list(
                iter_matches_stream(
                    1,
                    reader,
                    type_vocabulary=[1],
                    cutoff=2.0,
                    threshold=0.0,
                )
            )


def test_stream_accepts_representable_int64_atom_types() -> None:
    batch = _structures([1.0])

    def reader(ids: torch.Tensor):
        selected = batch.index_select(ids.tolist())
        return selected, torch.ones(selected.num_nodes, dtype=torch.int64)

    result = deduplicate_stream(
        1,
        reader,
        type_vocabulary=[1],
        cutoff=2.0,
        threshold=0.0,
    )
    assert result.retained_indices.tolist() == [0]
    assert (
        list(
            iter_matches_stream(
                1,
                reader,
                type_vocabulary=[1],
                cutoff=2.0,
                threshold=0.0,
            )
        )
        == []
    )


def test_summary_memory_retry_releases_failed_traceback_before_split(
    monkeypatch,
) -> None:
    batch = _structures([1.0, 1.0])
    original_build = stream_module.build_descriptor_stores
    sentinel_refs: list[weakref.ReferenceType] = []
    attempts = 0

    class AllocationSentinel:
        pass

    def fail_first_build(*args, **kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            allocation = AllocationSentinel()
            sentinel_refs.append(weakref.ref(allocation))
            raise MemoryError("injected summary allocation limit")
        assert sentinel_refs[0]() is None
        return original_build(*args, **kwargs)

    monkeypatch.setattr(stream_module, "build_descriptor_stores", fail_first_build)
    features, _, rebuilds = stream_module._build_summaries(
        [0, 1],
        lambda ids: (batch.index_select(ids.tolist()), None),
        None,
        2.0,
        torch.device("cpu"),
        2,
        max_batch_atoms=4,
    )

    assert features.shape == (2, 16)
    assert attempts == 3
    assert rebuilds == 2
    assert sentinel_refs[0]() is None


def test_empty_pool_and_empty_vocabulary() -> None:
    result = deduplicate_stream(
        0,
        lambda ids: (Batch.from_data_list([]), torch.empty(0, dtype=torch.int32)),
        type_vocabulary=[],
        cutoff=2.0,
        threshold=0.0,
    )
    assert result.retained_indices.numel() == 0
    assert result.representative_indices.numel() == 0
    assert result.multiplicities.numel() == 0


def test_deduplicate_batch_reports_required_atom_count_and_tile_capacity() -> None:
    batch = _structures([1.0])
    with pytest.raises(
        MemoryError,
        match="structure 0 requires 2 atoms, above max_batch_atoms capacity 1",
    ):
        deduplicate_batch(
            batch,
            cutoff=2.0,
            threshold=0.0,
            max_batch_atoms=1,
        )


def test_stream_preserves_single_structure_descriptor_capacity_detail(
    monkeypatch,
) -> None:
    batch = _structures([1.0])

    def fail_descriptor_build(*args, **kwargs):
        raise MemoryError(
            "neighbor capacity needs an estimated 1234 bytes, above the "
            "configured 500-byte device budget"
        )

    monkeypatch.setattr(stream_module, "build_descriptor_stores", fail_descriptor_build)
    with pytest.raises(
        MemoryError,
        match=r"structure 0 with 2 atoms.*1234 bytes.*500-byte device budget",
    ):
        deduplicate_stream(
            1,
            lambda ids: (batch.index_select(ids.tolist()), None),
            cutoff=2.0,
            threshold=0.0,
        )


def test_self_pair_atom_limit_applies_per_materialized_tile() -> None:
    batch = _structures([1.0, 1.0])
    matches = list(
        iter_matches_stream(
            batch.num_graphs,
            lambda ids: (batch.index_select(ids.tolist()), None),
            cutoff=2.0,
            threshold=0.0,
            max_batch_atoms=3,
        )
    )
    assert torch.cat(matches).tolist() == [[0, 1]]


def test_sequential_dedup_pair_uses_independent_atom_caps() -> None:
    batch = _structures([1.0, 1.0])
    result = deduplicate_stream(
        2,
        lambda ids: (batch.index_select(ids.tolist()), None),
        cutoff=2.0,
        threshold=0.0,
        input_batch_size=1,
        pair_block_size=1,
        max_batch_atoms=3,
    )
    assert result.retained_indices.tolist() == [0]
    assert result.representative_indices.tolist() == [0, 0]


def test_within_chunk_pair_uses_independent_atom_caps() -> None:
    batch = _structures([1.0, 1.0])
    result = deduplicate_stream(
        2,
        lambda ids: (batch.index_select(ids.tolist()), None),
        cutoff=2.0,
        threshold=0.0,
        input_batch_size=2,
        pair_block_size=1,
        max_batch_atoms=3,
    )
    assert result.retained_indices.tolist() == [0]
    assert result.representative_indices.tolist() == [0, 0]


def test_cross_pair_atom_limit_applies_per_materialized_tile() -> None:
    left = _structures([1.0])
    right = _structures([1.0])
    matches = list(
        iter_matches_stream(
            1,
            lambda ids: (left.index_select(ids.tolist()), None),
            cutoff=2.0,
            threshold=0.0,
            other_count=1,
            read_other_typed_batch=lambda ids: (
                right.index_select(ids.tolist()),
                None,
            ),
            max_batch_atoms=3,
        )
    )
    assert torch.cat(matches).tolist() == [[0, 0]]


def test_cuda_budget_uses_allocator_reusable_bytes_and_85_percent_cap(monkeypatch):
    requested_devices: list[torch.device] = []

    def available_bytes(device: torch.device) -> int:
        requested_devices.append(device)
        return 10_000

    monkeypatch.setattr(stream_module, "available_cuda_bytes", available_bytes)
    monkeypatch.setattr(
        torch.cuda,
        "mem_get_info",
        lambda *_args, **_kwargs: pytest.fail("raw driver free memory was queried"),
    )

    cuda_device = torch.device("cuda:1")
    assert stream_module._cuda_budget(cuda_device, 0.85) == 8_500
    assert stream_module._cuda_budget(torch.device("cpu"), 0.85) is None
    assert requested_devices == [cuda_device]


def test_dedup_zero_remaining_cuda_budget_uses_memory_error(monkeypatch) -> None:
    batch = _structures([1.0, 1.0, 1.0, 1.0])
    monkeypatch.setattr(stream_module, "_cuda_budget", lambda *_: 512)
    monkeypatch.setattr(
        stream_module,
        "_candidate_representatives",
        lambda rank, retained, *args, **kwargs: (
            (retained.copy(), len(retained), len(retained), len(retained))
            if rank >= 2
            else ([], 0, 0, 0)
        ),
    )
    monkeypatch.setattr(stream_module, "_summary_pair_may_match", lambda *args: False)
    monkeypatch.setattr(
        stream_module,
        "_summary_pairs_may_match_batch",
        lambda pairs, *args, **kwargs: torch.zeros(pairs.shape[0], dtype=torch.bool),
    )
    build_budgets: list[int | None] = []

    def fake_summaries(ids, *args, **kwargs):
        return (
            torch.zeros((len(ids), 16)),
            torch.ones((len(ids), 2), dtype=torch.bool),
            torch.full((len(ids),), 2, dtype=torch.int32),
            0,
        )

    def saturate_candidate_budget(*args, **kwargs):
        budget = kwargs.get("cuda_memory_budget_bytes")
        build_budgets.append(budget)
        assert budget is None or budget > 0
        return {"untyped": type("Index", (), {"_resident_descriptor_bytes": 512})()}

    monkeypatch.setattr(stream_module, "_build_summaries", fake_summaries)
    monkeypatch.setattr(stream_module, "_build_mode_indices", saturate_candidate_budget)
    with pytest.raises(
        MemoryError,
        match=r"comparison pair \(2, 0\) with 4 atoms exceeds descriptor memory capacity",
    ):
        deduplicate_stream(
            batch.num_graphs,
            lambda ids: (batch.index_select(ids.tolist()), None),
            cutoff=2.0,
            threshold=0.0,
            input_batch_size=2,
            pair_block_size=2,
            device="cpu",
        )
    assert build_budgets
    assert all(budget == 512 for budget in build_budgets)


def test_iterator_zero_remaining_cuda_budget_uses_memory_error(monkeypatch) -> None:
    left = _structures([1.0])
    right = _structures([1.0])
    monkeypatch.setattr(stream_module, "_cuda_budget", lambda *_: 512)
    monkeypatch.setattr(
        stream_module,
        "_candidate_matches_across",
        lambda *args, **kwargs: [0],
    )
    build_budgets: list[int | None] = []

    def fake_summaries(ids, *args, **kwargs):
        return (
            torch.zeros((len(ids), 16)),
            torch.ones((len(ids), 2), dtype=torch.bool),
            torch.full((len(ids),), 2, dtype=torch.int32),
            0,
        )

    def saturate_left_budget(*args, **kwargs):
        budget = kwargs.get("cuda_memory_budget_bytes")
        build_budgets.append(budget)
        assert budget is None or budget > 0
        return {"untyped": type("Index", (), {"_resident_descriptor_bytes": 512})()}

    monkeypatch.setattr(stream_module, "_build_summaries", fake_summaries)
    monkeypatch.setattr(stream_module, "_build_mode_indices", saturate_left_budget)
    with pytest.raises(
        MemoryError,
        match=r"cross-pool pair \(0, 0\) with 4 atoms exceeds descriptor memory capacity",
    ):
        list(
            iter_matches_stream(
                left.num_graphs,
                lambda ids: (left.index_select(ids.tolist()), None),
                cutoff=2.0,
                threshold=0.0,
                other_count=right.num_graphs,
                read_other_typed_batch=lambda ids: (
                    right.index_select(ids.tolist()),
                    None,
                ),
                pair_chunk_size=1,
            )
        )
    assert build_budgets == [512]


def test_dedup_descriptor_tiles_use_independent_atom_caps_and_shared_budget(
    monkeypatch,
) -> None:
    batch = _structures([1.0] * 6)
    builds: list[tuple[int, int | None, object]] = []

    class FakeIndex:
        def __init__(self, resident_bytes: int) -> None:
            self._resident_descriptor_bytes = resident_bytes
            self.allowances: list[tuple[int | None, int]] = []

        def _set_active_memory_allowance(
            self, budget: int | None, active_resident_bytes: int
        ) -> None:
            self.allowances.append((budget, active_resident_bytes))

    def fake_summaries(ids, *args, **kwargs):
        return (
            torch.zeros((len(ids), 16)),
            torch.ones((len(ids), 2), dtype=torch.bool),
            torch.full((len(ids),), 2, dtype=torch.int32),
            0,
        )

    def fake_build(active_batch, *args, **kwargs):
        budget = kwargs["cuda_memory_budget_bytes"]
        index = FakeIndex(400 if not builds else 300)
        builds.append((active_batch.num_nodes, budget, index))
        return {"untyped": index}

    monkeypatch.setattr(stream_module, "_cuda_budget", lambda *_: 1000)
    monkeypatch.setattr(stream_module, "_build_summaries", fake_summaries)
    monkeypatch.setattr(
        stream_module,
        "_candidate_representatives",
        lambda rank, retained, *args, **kwargs: (
            (retained.copy(), len(retained), len(retained), len(retained))
            if rank >= 4
            else ([], 0, 0, 0)
        ),
    )
    monkeypatch.setattr(stream_module, "_summary_pair_may_match", lambda *args: False)
    monkeypatch.setattr(
        stream_module,
        "_summary_pairs_may_match_batch",
        lambda pairs, *args, **kwargs: torch.zeros(pairs.shape[0], dtype=torch.bool),
    )
    monkeypatch.setattr(stream_module, "_build_mode_indices", fake_build)
    monkeypatch.setattr(
        stream_module,
        "_screen_prebuilt_pairs",
        lambda _left, pairs, *_args, **_kwargs: pairs[:0],
    )

    result = deduplicate_stream(
        batch.num_graphs,
        lambda ids: (batch.index_select(ids.tolist()), None),
        cutoff=2.0,
        threshold=0.0,
        input_batch_size=4,
        pair_block_size=2,
        max_batch_atoms=4,
        device="cpu",
    )

    assert result.retained_indices.tolist() == list(range(6))
    # One 4-atom candidate tile coexists with each 4-atom rep tile. The two
    # tiles total 8 atoms, while rep construction receives only the bytes left
    # by the candidate descriptors and scoring gets the aggregate residency.
    assert [(atoms, budget) for atoms, budget, _ in builds] == [
        (4, 1000),
        (4, 600),
        (4, 600),
    ]
    assert [index.allowances for _, _, index in builds] == [
        [(1000, 700), (1000, 700)],
        [(1000, 700)],
        [(1000, 700)],
    ]


def test_pair_group_uses_independent_atom_tiles_and_shared_cuda_budget(
    monkeypatch,
) -> None:
    batch = _structures([1.0, 1.0])
    builds: list[tuple[int, int | None, object]] = []

    class FakeIndex:
        def __init__(self, resident_bytes: int) -> None:
            self._resident_descriptor_bytes = resident_bytes
            self.allowances: list[tuple[int | None, int]] = []

        def _set_active_memory_allowance(
            self, budget: int | None, active_resident_bytes: int
        ) -> None:
            self.allowances.append((budget, active_resident_bytes))

    def fake_build(active_batch, *args, **kwargs):
        budget = kwargs["cuda_memory_budget_bytes"]
        index = FakeIndex(400 if not builds else 300)
        builds.append((active_batch.num_nodes, budget, index))
        return {"untyped": index}

    monkeypatch.setattr(stream_module, "_cuda_budget", lambda *_: 1000)
    monkeypatch.setattr(stream_module, "_build_mode_indices", fake_build)
    monkeypatch.setattr(
        stream_module,
        "_screen_prebuilt_pairs",
        lambda _left, pairs, *_args, **_kwargs: pairs,
    )
    stats = {
        "descriptor_rebuild_seconds": 0.0,
        "descriptor_rebuilds": 0,
        "descriptor_structures_built": 0,
    }

    matches = stream_module._screen_pair_group(
        [0, 1],
        [(1, 0)],
        lambda ids: (batch.index_select(ids.tolist()), None),
        None,
        2.0,
        0.0,
        torch.device("cpu"),
        stats,
        atom_counts=torch.tensor([2, 2], dtype=torch.int32),
        max_batch_atoms=3,
    )

    assert matches == [(1, 0)]
    assert [(atoms, budget) for atoms, budget, _ in builds] == [
        (2, 1000),
        (2, 600),
    ]
    assert [index.allowances for _, _, index in builds] == [
        [(1000, 700)],
        [(1000, 700)],
    ]
    assert stats["descriptor_rebuilds"] == 2


def test_pair_group_builds_endpoint_union_once_when_it_fits_atom_cap(
    monkeypatch,
) -> None:
    batch = _structures([1.0, 1.0])
    reads: list[list[int]] = []
    builds: list[tuple[int, int | None, object]] = []
    screened: list[tuple[object, torch.Tensor, object]] = []

    class FakeIndex:
        _resident_descriptor_bytes = 400

        def _set_active_memory_allowance(
            self, budget: int | None, active_resident_bytes: int
        ) -> None:
            self.allowance = (budget, active_resident_bytes)

    def loader(ids: torch.Tensor) -> tuple[Batch, None]:
        requested = ids.tolist()
        reads.append(requested)
        return batch.index_select(requested), None

    def fake_build(active_batch, *args, **kwargs):
        index = FakeIndex()
        builds.append(
            (active_batch.num_nodes, kwargs["cuda_memory_budget_bytes"], index)
        )
        return {"untyped": index}

    def fake_screen(left, pairs, _threshold, *, right_indexes, **_kwargs):
        screened.append((left, pairs.detach().cpu(), right_indexes))
        return pairs

    monkeypatch.setattr(stream_module, "_cuda_budget", lambda *_: 1000)
    monkeypatch.setattr(stream_module, "_build_mode_indices", fake_build)
    monkeypatch.setattr(stream_module, "_screen_prebuilt_pairs", fake_screen)
    stats = {
        "descriptor_rebuild_seconds": 0.0,
        "descriptor_rebuilds": 0,
        "descriptor_structures_built": 0,
    }

    matches = stream_module._screen_pair_group(
        [0, 1],
        [(1, 0)],
        loader,
        None,
        2.0,
        0.0,
        torch.device("cpu"),
        stats,
        atom_counts=torch.tensor([2, 2], dtype=torch.int32),
        max_batch_atoms=4,
    )

    assert matches == [(1, 0)]
    assert reads == [[1, 0]]
    assert [(atoms, budget) for atoms, budget, _ in builds] == [(4, 1000)]
    assert screened[0][0] is not None
    assert screened[0][1].tolist() == [[0, 1]]
    assert screened[0][2] is None
    assert builds[0][2].allowance == (1000, 400)
    assert stats["descriptor_rebuilds"] == 1
    assert stats["descriptor_structures_built"] == 2


def test_iterator_descriptor_tiles_use_independent_atom_caps_and_shared_budget(
    monkeypatch,
) -> None:
    left = _structures([1.0])
    right = _structures([1.0])
    builds: list[tuple[int, int | None, object]] = []

    class FakeIndex:
        def __init__(self, resident_bytes: int) -> None:
            self._resident_descriptor_bytes = resident_bytes
            self.allowances: list[tuple[int | None, int]] = []

        def _set_active_memory_allowance(
            self, budget: int | None, active_resident_bytes: int
        ) -> None:
            self.allowances.append((budget, active_resident_bytes))

    def fake_summaries(ids, *args, **kwargs):
        return (
            torch.zeros((len(ids), 16)),
            torch.ones((len(ids), 2), dtype=torch.bool),
            torch.full((len(ids),), 2, dtype=torch.int32),
            0,
        )

    def fake_build(active_batch, *args, **kwargs):
        budget = kwargs["cuda_memory_budget_bytes"]
        index = FakeIndex(400 if not builds else 300)
        builds.append((active_batch.num_nodes, budget, index))
        return {"untyped": index}

    monkeypatch.setattr(stream_module, "_cuda_budget", lambda *_: 1000)
    monkeypatch.setattr(stream_module, "_build_summaries", fake_summaries)
    monkeypatch.setattr(
        stream_module, "_candidate_matches_across", lambda *args, **kwargs: [0]
    )
    monkeypatch.setattr(stream_module, "_build_mode_indices", fake_build)
    monkeypatch.setattr(
        stream_module,
        "_screen_prebuilt_pairs",
        lambda _left, pairs, *_args, **_kwargs: pairs,
    )

    matches = list(
        iter_matches_stream(
            1,
            lambda ids: (left.index_select(ids.tolist()), None),
            cutoff=2.0,
            threshold=0.0,
            other_count=1,
            read_other_typed_batch=lambda ids: (
                right.index_select(ids.tolist()),
                None,
            ),
            pair_chunk_size=1,
            max_batch_atoms=2,
        )
    )

    assert torch.cat(matches).tolist() == [[0, 0]]
    # Left and right tiles each meet the two-atom cap; their joint atom count
    # is four, and right-side descriptor construction receives the remaining
    # shared byte allowance after the 400-byte left residency.
    assert [(atoms, budget) for atoms, budget, _ in builds] == [(2, 1000), (2, 600)]
    assert [index.allowances for _, _, index in builds] == [
        [(1000, 700)],
        [(1000, 700)],
    ]


def test_deduplicate_batch_matches_stream_and_exhaustive_priority_oracle() -> None:
    batch, types = _feature_rich_structures()
    order = [6, 1, 4, 0, 3, 2, 5]
    expected = _direct_greedy(batch, types, order, 4.0, 0.08)

    result = deduplicate_batch(
        batch,
        atom_types=types,
        cutoff=4.0,
        threshold=0.08,
        priority_order=order,
        max_batch_atoms=8,
    )
    stream_result = deduplicate_stream(
        batch.num_graphs,
        _typed_loader(batch, types),
        type_vocabulary=[1, 2, 3, 4],
        cutoff=4.0,
        threshold=0.08,
        priority_order=order,
        input_batch_size=4,
        pair_block_size=2,
        max_batch_atoms=8,
    )

    for candidate in (result, stream_result):
        assert candidate.retained_indices.dtype == torch.int32
        assert candidate.representative_indices.dtype == torch.int32
        assert candidate.multiplicities.dtype == torch.int32
        assert candidate.retained_indices.device.type == "cpu"
        assert candidate.retained_indices.tolist() == expected[0]
        assert candidate.representative_indices.tolist() == expected[1]
        assert candidate.multiplicities.tolist() == expected[2]


@pytest.mark.parametrize(
    ("device", "summary_coordinate_count", "typed"),
    [
        ("cpu", None, False),
        ("cpu", 0, True),
        ("cuda", None, True),
        ("cuda", 0, False),
    ],
    ids=[
        "cpu-default-untyped",
        "cpu-exhaustive-typed",
        "cuda-default-typed",
        "cuda-exhaustive-untyped",
    ],
)
def test_confirmed_batch_and_stream_match_literal_greedy_oracle(
    device: str, summary_coordinate_count: int | None, typed: bool
) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    batch = _structures([1.0, 1.04, 1.08, 1.12, 1.08])
    types = torch.ones(batch.num_nodes, dtype=torch.int32) if typed else None
    order = [2, 0, 4, 1, 3]

    def accepts(candidate: int, representative: int) -> bool:
        if candidate == 1:
            return representative == 0
        if candidate == 3:
            return representative == 4
        return candidate != 4

    expected = _literal_confirmed_greedy(batch, types, order, 2.0, 0.06, accepts)
    assert expected[1][1] == 0
    assert expected[1][4] == 4
    assert expected[1][3] == 4

    radial_matches = set(
        _exhaustive_match_pairs(
            batch,
            batch,
            2.0,
            0.06,
            self_comparison=True,
            left_types=types,
            right_types=types,
        )
    )
    assert (0, 1) in radial_matches
    assert (1, 2) in radial_matches
    assert (0, 2) not in radial_matches

    summary_options = (
        {}
        if summary_coordinate_count is None
        else {"summary_coordinate_count": summary_coordinate_count}
    )
    batch_seen: list[list[int]] = []
    stream_seen: list[list[int]] = []

    def callback_for(seen: list[list[int]]):
        def confirm(pairs: torch.Tensor) -> torch.Tensor:
            assert pairs.dtype == torch.int32
            assert pairs.device.type == device
            assert pairs.shape[0] > 0
            rows = pairs.tolist()
            seen.extend(rows)
            for candidate, representative in rows:
                assert representative in expected[3][candidate]
                assert (
                    min(candidate, representative),
                    max(candidate, representative),
                ) in (radial_matches)
            accepted_positions = [
                position
                for position, (candidate, representative) in enumerate(rows)
                if accepts(candidate, representative)
            ]
            return pairs[accepted_positions] if accepted_positions else pairs[:0]

        return confirm

    batch_result = deduplicate_batch(
        batch,
        atom_types=types,
        cutoff=2.0,
        threshold=0.06,
        priority_order=order,
        max_batch_atoms=4,
        device=device,
        confirm=callback_for(batch_seen),
        **summary_options,
    )
    loader = (
        _typed_loader(batch, types)
        if types is not None
        else lambda ids: (batch.index_select(ids.tolist()), None)
    )
    stream_result = deduplicate_stream(
        batch.num_graphs,
        loader,
        type_vocabulary=[1] if types is not None else None,
        cutoff=2.0,
        threshold=0.06,
        priority_order=order,
        device=device,
        input_batch_size=2,
        pair_block_size=2,
        max_batch_atoms=4,
        confirm=callback_for(stream_seen),
        **summary_options,
    )

    for result in (batch_result, stream_result):
        assert result.retained_indices.tolist() == expected[0]
        assert result.representative_indices.tolist() == expected[1]
        assert result.multiplicities.tolist() == expected[2]
    for seen in (batch_seen, stream_seen):
        candidate_one = [
            representative for candidate, representative in seen if candidate == 1
        ]
        assert candidate_one[:2] == [2, 0]
        assert all(
            candidate != 4 or representative in (2, 0)
            for candidate, representative in seen
        )


@pytest.mark.parametrize(
    ("bad_result", "device"),
    [
        ("non-tensor", "cpu"),
        ("wrong-dtype", "cpu"),
        ("wrong-shape", "cpu"),
        ("wrong-order", "cpu"),
        ("new-pair", "cpu"),
        ("wrong-device", "cuda"),
    ],
)
def test_stream_confirm_requires_ordered_int32_pair_subset(
    bad_result: str, device: str
) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    batch = _structures([1.0, 1.08, 1.04])

    def malformed_result(pairs: torch.Tensor):
        if bad_result == "non-tensor":
            return pairs.tolist()
        if bad_result == "wrong-dtype":
            return pairs.to(torch.int64)
        if bad_result == "wrong-shape":
            return pairs[:, 0]
        if bad_result == "wrong-order":
            return pairs.flip(0)
        if bad_result == "new-pair":
            return torch.tensor([[2, 2]], dtype=torch.int32, device=pairs.device)
        return pairs.cpu()

    message = "comparison device" if bad_result == "wrong-device" else "confirm"
    with pytest.raises(ValueError, match=message):
        deduplicate_stream(
            batch.num_graphs,
            lambda ids: (batch.index_select(ids.tolist()), None),
            cutoff=2.0,
            threshold=0.06,
            device=device,
            summary_coordinate_count=0,
            input_batch_size=3,
            confirm=malformed_result,
        )


def test_deduplication_rejects_noncallable_confirm_before_loader_work() -> None:
    batch = _structures([1.0])
    reader_calls = 0

    def reader(_ids: torch.Tensor):
        nonlocal reader_calls
        reader_calls += 1
        raise AssertionError("invalid confirm must fail before reading")

    with pytest.raises(TypeError, match="confirm must be callable"):
        deduplicate_stream(1, reader, cutoff=2.0, threshold=0.0, confirm=object())
    with pytest.raises(TypeError, match="confirm must be callable"):
        deduplicate_batch(batch, cutoff=2.0, threshold=0.0, confirm=object())
    assert reader_calls == 0


def test_deduplication_does_not_call_confirm_for_empty_proposals() -> None:
    batch = _structures([1.0])

    def fail_empty_confirm(_pairs: torch.Tensor) -> torch.Tensor:
        raise AssertionError("confirm must not receive empty proposals")

    result = deduplicate_stream(
        batch.num_graphs,
        lambda ids: (batch.index_select(ids.tolist()), None),
        cutoff=2.0,
        threshold=0.0,
        confirm=fail_empty_confirm,
    )
    assert result.retained_indices.tolist() == [0]


def test_grouped_confirm_memory_error_propagates_without_retry(monkeypatch) -> None:
    batch = _structures([1.0, 1.2, 1.04, 1.06])
    read_ids: list[list[int]] = []
    original_build = stream_module._build_mode_indices
    build_count = 0
    callback_count = 0
    failure = MemoryError("caller confirmation allocation failure")

    def reader(ids: torch.Tensor):
        row_ids = ids.tolist()
        read_ids.append(row_ids)
        return batch.index_select(row_ids), None

    def count_builds(*args, **kwargs):
        nonlocal build_count
        build_count += 1
        return original_build(*args, **kwargs)

    def fail_confirm(_pairs: torch.Tensor) -> torch.Tensor:
        nonlocal callback_count
        callback_count += 1
        raise failure

    monkeypatch.setattr(stream_module, "_build_mode_indices", count_builds)
    with pytest.raises(MemoryError) as exc_info:
        deduplicate_stream(
            batch.num_graphs,
            reader,
            cutoff=2.0,
            threshold=0.06,
            summary_coordinate_count=0,
            input_batch_size=2,
            pair_block_size=2,
            max_batch_atoms=4,
            confirm=fail_confirm,
        )

    assert exc_info.value is failure
    assert callback_count == 1
    assert build_count == 3
    assert read_ids == [[0, 1], [2, 3], [1, 0], [2, 3], [0, 1]]


def test_grouped_representative_loader_memory_error_propagates_without_retry() -> None:
    batch = _structures([1.0, 1.2, 1.04, 1.06])
    read_ids: list[list[int]] = []
    failure = MemoryError("caller representative loader failure")

    def reader(ids: torch.Tensor):
        row_ids = ids.tolist()
        read_ids.append(row_ids)
        if row_ids == [0, 1] and len(read_ids) > 1:
            raise failure
        return batch.index_select(row_ids), None

    with pytest.raises(MemoryError) as exc_info:
        deduplicate_stream(
            batch.num_graphs,
            reader,
            cutoff=2.0,
            threshold=0.06,
            summary_coordinate_count=0,
            input_batch_size=2,
            pair_block_size=2,
            max_batch_atoms=4,
        )

    assert exc_info.value is failure
    assert read_ids == [[0, 1], [2, 3], [1, 0], [2, 3], [0, 1]]


def test_iter_matches_stream_right_loader_memory_error_propagates_unchanged() -> None:
    left = _structures([1.0])
    right = _structures([1.0])
    right_reads = 0
    failure = MemoryError("caller right-loader allocation failure")

    def left_reader(ids: torch.Tensor):
        return left.index_select(ids.tolist()), None

    def right_reader(ids: torch.Tensor):
        nonlocal right_reads
        right_reads += 1
        if right_reads == 2:
            raise failure
        return right.index_select(ids.tolist()), None

    iterator = iter_matches_stream(
        left.num_graphs,
        left_reader,
        cutoff=2.0,
        threshold=0.0,
        other_count=right.num_graphs,
        read_other_typed_batch=right_reader,
        summary_coordinate_count=0,
    )
    with pytest.raises(MemoryError) as exc_info:
        next(iterator)

    assert exc_info.value is failure
    assert right_reads == 2


@pytest.mark.parametrize(
    "failure_kind", ["cuda-oom", "runtime"], ids=["normalize-oom", "preserve-other"]
)
def test_iter_matches_stream_normalizes_only_singleton_left_builder_capacity_errors(
    monkeypatch, failure_kind: str
) -> None:
    batch = _structures([1.0, 1.0])
    failure = (
        torch.cuda.OutOfMemoryError("injected singleton builder OOM")
        if failure_kind == "cuda-oom"
        else RuntimeError("injected unrelated builder failure")
    )

    def reader(ids: torch.Tensor):
        return batch.index_select(ids.tolist()), None

    def fail_build(*_args, **_kwargs):
        raise failure

    monkeypatch.setattr(stream_module, "_build_mode_indices", fail_build)
    iterator = iter_matches_stream(
        batch.num_graphs,
        reader,
        cutoff=2.0,
        threshold=0.0,
        summary_coordinate_count=0,
    )
    if failure_kind == "cuda-oom":
        with pytest.raises(MemoryError, match="structure 0 with 2 atoms") as exc_info:
            next(iterator)
        assert exc_info.value is not failure
    else:
        with pytest.raises(RuntimeError) as exc_info:
            next(iterator)
        assert exc_info.value is failure


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="requires CUDA"
            ),
        ),
    ],
)
def test_iter_matches_stream_self_is_lazy_lexicographic_and_exhaustive(
    device: str,
) -> None:
    batch = _mixed_pbc_batch(
        [1.0, 5.0, 1.15, 1.4],
        [
            (True, False, False),
            (True, False, False),
            (False, False, False),
            (True, False, False),
        ],
    )
    types = torch.ones(batch.num_nodes, dtype=torch.int32)
    read_calls = 0
    reader_closed = False

    class Reader:
        def __call__(self, ids: torch.Tensor):
            nonlocal read_calls
            read_calls += 1
            assert ids.device.type == "cpu"
            assert ids.dtype == torch.int64
            return _typed_loader(batch, types)(ids)

        def close(self):
            nonlocal reader_closed
            reader_closed = True

    reader = Reader()
    iterator = iter_matches_stream(
        batch.num_graphs,
        reader,
        type_vocabulary=[1],
        cutoff=2.0,
        threshold=0.2,
        pair_chunk_size=1,
        max_batch_atoms=4,
        device=device,
    )
    assert read_calls == 0
    first = next(iterator)
    assert read_calls > 0
    assert first.dtype == torch.int32
    assert first.device.type == device
    assert 0 < first.shape[0] <= 1
    rest = list(iterator)
    chunks = [first, *rest]
    assert all(0 < chunk.shape[0] <= 1 for chunk in chunks)
    actual = torch.cat(chunks).tolist()
    expected = _exhaustive_match_pairs(
        batch,
        batch,
        2.0,
        0.2,
        self_comparison=True,
        left_types=types,
        right_types=types,
    )
    assert actual == [list(pair) for pair in expected]
    assert actual == sorted(actual)
    assert all(left < right for left, right in actual)
    assert not reader_closed


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="requires CUDA"
            ),
        ),
    ],
)
def test_iter_matches_stream_cross_pool_is_exhaustive_and_lexicographic(
    device: str,
) -> None:
    left = _mixed_pbc_batch(
        [1.0, 5.0, 1.15, 1.4],
        [
            (True, False, False),
            (True, False, False),
            (False, False, False),
            (True, False, False),
        ],
    )
    right = _mixed_pbc_batch(
        [1.02, 1.22, 5.05],
        [(True, False, False), (False, False, False), (True, False, False)],
    )

    def untyped_loader(batch: Batch):
        return lambda ids: (batch.index_select(ids.tolist()), None)

    iterator = iter_matches_stream(
        left.num_graphs,
        untyped_loader(left),
        cutoff=2.0,
        threshold=0.2,
        other_count=right.num_graphs,
        read_other_typed_batch=untyped_loader(right),
        pair_chunk_size=2,
        max_batch_atoms=4,
        device=device,
    )
    chunks = list(iterator)
    assert chunks
    assert all(
        chunk.dtype == torch.int32
        and chunk.device.type == device
        and 0 < chunk.shape[0] <= 2
        for chunk in chunks
    )
    actual = torch.cat(chunks).tolist()
    expected = _exhaustive_match_pairs(left, right, 2.0, 0.2, self_comparison=False)
    assert actual == [list(pair) for pair in expected]
    assert actual == sorted(actual)


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="requires CUDA"
            ),
        ),
    ],
)
def test_zero_summary_coordinates_iter_matches_exhaustively(device: str) -> None:
    left = _mixed_pbc_batch(
        [1.0, 5.0, 1.15, 1.4],
        [
            (True, False, False),
            (True, False, False),
            (False, False, False),
            (True, False, False),
        ],
    )
    right = _mixed_pbc_batch(
        [1.02, 1.22, 5.05],
        [(True, False, False), (False, False, False), (True, False, False)],
    )
    left_types = torch.ones(left.num_nodes, dtype=torch.int32)
    right_types = torch.ones(right.num_nodes, dtype=torch.int32)

    self_matches = torch.cat(
        list(
            iter_matches_stream(
                left.num_graphs,
                _typed_loader(left, left_types),
                type_vocabulary=[1],
                cutoff=2.0,
                threshold=0.2,
                pair_chunk_size=2,
                summary_coordinate_count=0,
                device=device,
            )
        )
    ).tolist()
    cross_matches = torch.cat(
        list(
            iter_matches_stream(
                left.num_graphs,
                _typed_loader(left, left_types),
                type_vocabulary=[1],
                cutoff=2.0,
                threshold=0.2,
                other_count=right.num_graphs,
                read_other_typed_batch=_typed_loader(right, right_types),
                pair_chunk_size=2,
                summary_coordinate_count=0,
                device=device,
            )
        )
    ).tolist()
    self_expected = _exhaustive_match_pairs(
        left,
        left,
        2.0,
        0.2,
        self_comparison=True,
        left_types=left_types,
        right_types=left_types,
    )
    cross_expected = _exhaustive_match_pairs(
        left,
        right,
        2.0,
        0.2,
        self_comparison=False,
        left_types=left_types,
        right_types=right_types,
    )
    assert self_matches == [list(pair) for pair in self_expected]
    assert cross_matches == [list(pair) for pair in cross_expected]


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_iter_matches_stream_reuses_left_descriptors_across_right_blocks(
    monkeypatch, device: str
) -> None:
    def offset_structure(offset: float, length: float) -> AtomicData:
        return AtomicData(
            positions=torch.tensor(
                [[offset, 0.0, 0.0], [offset + length, 0.0, 0.0]],
                dtype=torch.float32,
            ),
            atomic_numbers=torch.ones(2, dtype=torch.int64),
        )

    left = Batch.from_data_list([offset_structure(100.0, 1.0)])
    right = Batch.from_data_list(
        [
            offset_structure(0.0, 1.0),
            offset_structure(10.0, 1.1),
            offset_structure(20.0, 1.2),
        ]
    )
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    built_origins: list[float] = []
    original_build = stream_module._build_mode_indices

    def count_builds(batch, *args, **kwargs):
        built_origins.append(float(batch.positions[0, 0]))
        return original_build(batch, *args, **kwargs)

    monkeypatch.setattr(stream_module, "_build_mode_indices", count_builds)
    chunks = list(
        iter_matches_stream(
            left.num_graphs,
            lambda ids: (left.index_select(ids.tolist()), None),
            cutoff=2.0,
            threshold=1.0,
            other_count=right.num_graphs,
            read_other_typed_batch=lambda ids: (right.index_select(ids.tolist()), None),
            pair_chunk_size=1,
            device=device,
        )
    )

    actual = torch.cat(chunks).tolist()
    assert actual == [[0, 0], [0, 1], [0, 2]]
    # The left descriptor bundle is built once; only each right block is rebuilt.
    assert built_origins == [100.0, 0.0, 10.0, 20.0]


def test_iter_matches_stream_releases_right_tile_before_next_left_and_on_close(
    monkeypatch,
) -> None:
    def offset_batch(offsets: list[float]) -> Batch:
        return Batch.from_data_list(
            [
                AtomicData(
                    positions=torch.tensor(
                        [[offset, 0.0, 0.0], [offset + 1.0, 0.0, 0.0]],
                        dtype=torch.float32,
                    ),
                    atomic_numbers=torch.ones(2, dtype=torch.int64),
                )
                for offset in offsets
            ]
        )

    left = offset_batch([100.0, 200.0])
    right = offset_batch([0.0])

    def typed_reader(batch: Batch):
        pointers = batch.batch_ptr.tolist()

        def read(ids: torch.Tensor):
            row_ids = ids.tolist()
            selected = batch.index_select(row_ids)
            selected_types = torch.cat(
                [
                    torch.ones(pointers[row + 1] - pointers[row], dtype=torch.int32)
                    for row in row_ids
                ]
            ).clone()
            return selected, selected_types

        return read

    right_refs: list[weakref.ReferenceType] = []
    left_refs: list[weakref.ReferenceType] = []
    build_origins: list[float] = []

    class FakeIndex:
        _resident_descriptor_bytes = 0

        def _set_active_memory_allowance(self, budget, active_resident_bytes) -> None:
            pass

    def fake_summaries(ids, *args, **kwargs):
        return (
            torch.zeros((len(ids), 14)),
            torch.ones((len(ids), 1), dtype=torch.bool),
            torch.full((len(ids),), 2, dtype=torch.int32),
            0,
        )

    def track_build(active_batch, active_types, *args, **kwargs):
        origin = float(active_batch.positions[0, 0])
        build_origins.append(origin)
        indexes = {
            name: FakeIndex() for name in stream_module._mode_names(active_types)
        }
        representative_index = next(iter(indexes.values()))
        refs = [
            weakref.ref(active_batch.positions),
            weakref.ref(active_types),
            weakref.ref(representative_index),
        ]
        if origin < 50.0:
            right_refs[:] = refs
        else:
            if left_refs:
                assert all(reference() is None for reference in left_refs)
            assert not right_refs or all(
                reference() is None for reference in right_refs
            )
            left_refs[:] = refs
        return indexes

    monkeypatch.setattr(stream_module, "_build_summaries", fake_summaries)
    monkeypatch.setattr(stream_module, "_select_coordinates", lambda *args: [0])
    monkeypatch.setattr(stream_module, "_candidate_matches_across", lambda *args: [0])
    monkeypatch.setattr(stream_module, "_build_mode_indices", track_build)
    monkeypatch.setattr(
        stream_module,
        "_screen_prebuilt_pairs",
        lambda _left, pairs, *_args, **_kwargs: pairs,
    )

    iterator = iter_matches_stream(
        2,
        typed_reader(left),
        type_vocabulary=[1],
        cutoff=2.0,
        threshold=0.0,
        other_count=1,
        read_other_typed_batch=typed_reader(right),
        pair_chunk_size=1,
    )
    first = next(iterator)
    assert first.tolist() == [[0, 0]]
    assert all(reference() is None for reference in right_refs)

    second = next(iterator)
    assert second.tolist() == [[1, 0]]
    assert build_origins == [100.0, 0.0, 200.0, 0.0]
    assert all(reference() is None for reference in right_refs)

    iterator.close()
    assert all(reference() is None for reference in left_refs)


@pytest.mark.parametrize("cross_pool", [False, True], ids=["self", "cross"])
def test_iter_matches_stream_releases_pilot_summaries_before_first_yield_and_keeps_loaders_open(
    monkeypatch, cross_pool: bool
) -> None:
    left = _mixed_pbc_batch(
        [1.0, 2.0] if not cross_pool else [1.0],
        [(False, False, False)] * (2 if not cross_pool else 1),
    )
    right = _mixed_pbc_batch([1.0], [(False, False, False)])

    class Reader:
        def __init__(self, batch: Batch) -> None:
            self.batch = batch
            self.calls = 0
            self.close_calls = 0

        def __call__(self, ids: torch.Tensor):
            self.calls += 1
            return self.batch.index_select(ids.tolist()), None

        def close(self) -> None:
            self.close_calls += 1

    left_reader = Reader(left)
    right_reader = Reader(right)
    pilot_summary_refs: list[weakref.ReferenceType] = []
    summary_builds = 0
    pilot_build_count = 2 if cross_pool else 1

    class FakeIndex:
        _resident_descriptor_bytes = 0

        def _set_active_memory_allowance(self, budget, active_resident_bytes) -> None:
            pass

    def fake_summaries(ids, *args, **kwargs):
        nonlocal summary_builds
        features = torch.zeros((len(ids), 14), dtype=torch.float32)
        presence = torch.ones((len(ids), 2), dtype=torch.bool)
        atom_counts = torch.full((len(ids),), 2, dtype=torch.int32)
        if summary_builds < pilot_build_count:
            pilot_summary_refs.extend((weakref.ref(features), weakref.ref(presence)))
        summary_builds += 1
        return features, presence, atom_counts, 0

    monkeypatch.setattr(stream_module, "_build_summaries", fake_summaries)
    monkeypatch.setattr(stream_module, "_select_coordinates", lambda *args: [0])
    monkeypatch.setattr(
        stream_module,
        "_candidate_matches_across",
        lambda *args: [0] if cross_pool else [1],
    )
    monkeypatch.setattr(
        stream_module,
        "_build_mode_indices",
        lambda _batch, active_types, **_kwargs: {
            name: FakeIndex() for name in stream_module._mode_names(active_types)
        },
    )
    monkeypatch.setattr(
        stream_module,
        "_screen_prebuilt_pairs",
        lambda _left, pairs, *_args, **_kwargs: pairs,
    )

    iterator = iter_matches_stream(
        left.num_graphs,
        left_reader,
        cutoff=2.0,
        threshold=0.0,
        other_count=right.num_graphs if cross_pool else None,
        read_other_typed_batch=right_reader if cross_pool else None,
        pair_chunk_size=1,
    )
    first = next(iterator)
    assert first.tolist() == ([[0, 0]] if cross_pool else [[0, 1]])
    assert all(reference() is None for reference in pilot_summary_refs)

    iterator.close()
    assert left_reader.close_calls == 0
    assert right_reader.close_calls == 0


@pytest.mark.parametrize(
    ("count", "other_count"),
    [(0, None), (0, 2), (2, 0)],
    ids=["empty-self", "empty-left", "empty-right"],
)
def test_iter_matches_stream_empty_pools_skip_loaders(
    count: int, other_count: int | None
) -> None:
    class Reader:
        def __init__(self) -> None:
            self.calls = 0

        def __call__(self, _ids: torch.Tensor):
            self.calls += 1
            raise AssertionError("empty-pool iteration must not call its loader")

    left_reader = Reader()
    right_reader = Reader()
    iterator = iter_matches_stream(
        count,
        left_reader,
        cutoff=2.0,
        threshold=0.1,
        other_count=other_count,
        read_other_typed_batch=right_reader if other_count is not None else None,
    )

    assert list(iterator) == []
    assert left_reader.calls == 0
    assert right_reader.calls == 0


def test_iter_matches_stream_validates_options_on_first_next_for_empty_pool() -> None:
    class Reader:
        def __init__(self) -> None:
            self.calls = 0

        def __call__(self, _ids: torch.Tensor):
            self.calls += 1
            raise AssertionError("invalid empty-pool options must fail before loading")

    reader = Reader()
    iterator = iter_matches_stream(0, reader, cutoff=2.0, threshold=-0.1)

    with pytest.raises(ValueError, match="threshold"):
        next(iterator)
    assert reader.calls == 0


@pytest.mark.parametrize(
    ("other_count", "other_reader"),
    [(None, lambda ids: (Batch.from_data_list([]), None)), (0, None)],
)
def test_iter_matches_stream_requires_both_cross_pool_arguments(
    other_count, other_reader
) -> None:
    with pytest.raises(ValueError, match="must be supplied together"):
        list(
            iter_matches_stream(
                0,
                lambda ids: (Batch.from_data_list([]), None),
                cutoff=2.0,
                threshold=0.0,
                other_count=other_count,
                read_other_typed_batch=other_reader,
            )
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_stream_cuda_matches_exhaustive_greedy() -> None:
    batch = _structures([1.0, 1.05, 1.1025, 1.1025, 1.30])
    types = torch.ones(batch.num_nodes, dtype=torch.int32)
    expected = _direct_greedy(batch, types, list(range(5)), 2.0, 0.06)
    result = _run(batch, types, device="cuda")
    assert result.retained_indices.tolist() == expected[0]
    assert result.representative_indices.tolist() == expected[1]
    assert result.multiplicities.tolist() == expected[2]
