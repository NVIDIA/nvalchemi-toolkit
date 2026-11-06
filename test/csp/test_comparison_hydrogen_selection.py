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
"""Public hydrogen-selection behavior for CSP radial comparisons."""

from __future__ import annotations

import pytest
import torch

from nvalchemi.csp.comparison import (
    RadialComparisonIndex,
    deduplicate_batch,
    deduplicate_stream,
    iter_matches_stream,
)
from nvalchemi.data import AtomicData, Batch
from nvalchemi.data.level_storage import MultiLevelStorage


def _make_batch(
    positions: list[torch.Tensor],
    atomic_numbers: list[torch.Tensor],
    *,
    pbc: torch.Tensor | None = None,
) -> Batch:
    data = []
    for row, (row_positions, row_numbers) in enumerate(
        zip(positions, atomic_numbers, strict=True)
    ):
        values = {
            "positions": row_positions,
            "atomic_numbers": row_numbers,
        }
        if pbc is not None:
            values["cell"] = (
                torch.eye(3, dtype=row_positions.dtype).mul(12).unsqueeze(0)
            )
            values["pbc"] = pbc[row : row + 1]
        data.append(AtomicData(**values))
    return Batch.from_data_list(data)


def _hydrogen_variant_batch(*, pbc: torch.Tensor | None = None) -> Batch:
    return _make_batch(
        [
            torch.tensor([[0.2, 6, 6], [11.8, 6, 6], [0.2, 7.0, 6]]),
            torch.tensor([[4.2, 6, 6], [3.7, 6, 6], [4.2, 9.0, 6]]),
        ],
        [torch.tensor([6, 6, 1]), torch.tensor([6, 6, 1])],
        pbc=pbc,
    )


def _same_heavy_variant_batch() -> Batch:
    return _make_batch(
        [
            torch.tensor([[0.2, 6, 6], [0.6, 6, 6], [0.2, 7.0, 6]]),
            torch.tensor([[4.2, 6, 6], [4.6, 6, 6], [4.2, 9.0, 6]]),
        ],
        [torch.tensor([6, 6, 1]), torch.tensor([6, 6, 1])],
    )


def _read_from_batch(batch: Batch, atom_types: torch.Tensor):
    pointers = batch.batch_ptr.tolist()

    def read(ids: torch.Tensor) -> tuple[Batch, torch.Tensor]:
        rows = ids.tolist()
        selected = batch.index_select(rows)
        selected_types = torch.cat(
            [atom_types[pointers[row] : pointers[row + 1]] for row in rows]
        )
        return selected, selected_types

    return read


@pytest.mark.parametrize(
    "pbc",
    [
        None,
        torch.tensor([[True, True, True]] * 2),
        torch.tensor([[True, False, False]] * 2),
    ],
)
@pytest.mark.parametrize("device_name", ["cpu", "cuda"])
@pytest.mark.parametrize("typing_mode", ["untyped", "center", "full"])
def test_index_hydrogen_filter_matches_explicit_mask_for_periodicity_and_types(
    pbc: torch.Tensor | None,
    device_name: str,
    typing_mode: str,
) -> None:
    if device_name == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    batch = _hydrogen_variant_batch(pbc=pbc)
    atom_types = (
        torch.tensor([41, 42, 901, 41, 42, 901], dtype=torch.int64)
        if typing_mode != "untyped"
        else None
    )
    batch = batch.to(device_name)
    if atom_types is not None:
        atom_types = atom_types.to(device_name)
    before_positions = batch.positions.clone()
    before_numbers = batch.atomic_numbers.clone()
    keep = batch.atomic_numbers != 1
    filtered_data = []
    for row in range(batch.num_graphs):
        original = batch[row]
        row_keep = original.atomic_numbers != 1
        filtered_data.append(
            AtomicData(
                positions=original.positions[row_keep],
                atomic_numbers=original.atomic_numbers[row_keep],
                **(
                    {"cell": original.cell, "pbc": original.pbc}
                    if pbc is not None
                    else {}
                ),
            )
        )
    manual_batch = Batch.from_data_list(filtered_data)

    typed_neighbors = typing_mode == "full"
    automatic = RadialComparisonIndex.build(
        batch,
        cutoff=4.0,
        atom_types=atom_types,
        typed_neighbors=typed_neighbors,
        include_hydrogens=False,
    )
    manual = RadialComparisonIndex.build(
        manual_batch.to(device_name),
        cutoff=4.0,
        atom_types=None if atom_types is None else atom_types[keep],
        typed_neighbors=typed_neighbors,
        device=device_name,
    )
    pair = torch.tensor([[0, 1]], dtype=torch.int32)

    assert automatic.include_hydrogens is False
    assert automatic.score_pairs(pair).item() == pytest.approx(
        manual.score_pairs(pair).item(), abs=2e-6
    )
    assert automatic.score_pairs(pair).item() > 0.01
    assert automatic.score_pairs(pair).device.type == device_name
    assert torch.equal(batch.positions, before_positions)
    assert torch.equal(batch.atomic_numbers, before_numbers)
    with pytest.raises(AttributeError):
        automatic.include_hydrogens = True


def test_hydrogen_geometry_changes_full_score_and_original_row_deduplication() -> None:
    batch = _same_heavy_variant_batch()
    types = torch.tensor([41, 42, 901, 41, 42, 901], dtype=torch.int32)
    pair = torch.tensor([[0, 1]], dtype=torch.int32)
    all_atom_index = RadialComparisonIndex.build(
        batch, cutoff=4.0, atom_types=types, typed_neighbors=True
    )
    heavy_index = RadialComparisonIndex.build(
        batch,
        cutoff=4.0,
        atom_types=types,
        typed_neighbors=True,
        include_hydrogens=False,
    )
    assert all_atom_index.score_pairs(pair).item() > 0.1
    assert heavy_index.score_pairs(pair).item() == pytest.approx(0.0, abs=2e-6)

    proposals: list[list[list[int]]] = []

    def confirm(pairs: torch.Tensor) -> torch.Tensor:
        proposals.append(pairs.cpu().tolist())
        return pairs

    threshold = 0.01
    all_atom_result = deduplicate_batch(
        batch,
        atom_types=types,
        cutoff=4.0,
        threshold=threshold,
        priority_order=[1, 0],
        summary_coordinate_count=0,
    )
    result = deduplicate_batch(
        batch,
        atom_types=types,
        cutoff=4.0,
        threshold=threshold,
        priority_order=[1, 0],
        include_hydrogens=False,
        summary_coordinate_count=0,
        confirm=confirm,
    )
    assert all_atom_result.retained_indices.tolist() == [1, 0]
    assert result.retained_indices.tolist() == [1]
    assert result.representative_indices.tolist() == [1, 1]
    assert result.multiplicities.tolist() == [2]
    assert proposals == [[[0, 1]]]
    assert batch[result.retained_indices[0]].atomic_numbers.tolist() == [6, 6, 1]


def test_index_flags_must_match_and_original_types_are_validated_before_masking() -> (
    None
):
    batch = _hydrogen_variant_batch()
    types = torch.tensor([41, 42, 901, 41, 42, 901], dtype=torch.int32)
    with pytest.raises(ValueError, match="one entry per atom"):
        RadialComparisonIndex.build(
            batch,
            cutoff=4.0,
            atom_types=types[:-1],
            include_hydrogens=False,
        )

    all_atoms = RadialComparisonIndex.build(batch, cutoff=4.0)
    heavy_atoms = RadialComparisonIndex.build(
        batch, cutoff=4.0, include_hydrogens=False
    )
    with pytest.raises(ValueError, match="include_hydrogens"):
        all_atoms.find_matches(heavy_atoms, threshold=0.0)


def test_default_and_true_match_and_false_preserves_already_heavy_geometry() -> None:
    batch = _make_batch(
        [
            torch.tensor([[0.0, 0, 0], [1.0, 0, 0]]),
            torch.tensor([[4.0, 0, 0], [5.2, 0, 0]]),
        ],
        [torch.tensor([6, 8]), torch.tensor([6, 8])],
    )
    pair = torch.tensor([[0, 1]], dtype=torch.int32)
    default = RadialComparisonIndex.build(batch, cutoff=2.0)
    explicit_true = RadialComparisonIndex.build(
        batch, cutoff=2.0, include_hydrogens=True
    )
    explicit_false = RadialComparisonIndex.build(
        batch, cutoff=2.0, include_hydrogens=False
    )
    expected = default.score_pairs(pair)
    assert default.include_hydrogens is True
    assert explicit_true.include_hydrogens is True
    assert explicit_false.include_hydrogens is False
    assert torch.equal(explicit_true.score_pairs(pair), expected)
    assert torch.equal(explicit_false.score_pairs(pair), expected)


@pytest.mark.parametrize(
    "operation",
    [
        lambda batch: RadialComparisonIndex.build(
            batch, cutoff=4.0, include_hydrogens=1
        ),
        lambda batch: deduplicate_batch(
            batch, cutoff=4.0, threshold=0.1, include_hydrogens=1
        ),
    ],
)
def test_non_bool_hydrogen_flags_are_rejected(operation) -> None:
    with pytest.raises(TypeError, match="include_hydrogens must be bool"):
        operation(_hydrogen_variant_batch())


def test_stream_hydrogen_flag_validation_preserves_lazy_iterator() -> None:
    def unexpected_read(_ids: torch.Tensor):
        raise AssertionError("invalid options are checked before loader reads")

    with pytest.raises(TypeError, match="include_hydrogens must be bool"):
        deduplicate_stream(
            0,
            unexpected_read,
            cutoff=2.0,
            threshold=0.1,
            include_hydrogens=1,
        )
    iterator = iter_matches_stream(
        0,
        unexpected_read,
        cutoff=2.0,
        threshold=0.1,
        include_hydrogens=1,
    )
    with pytest.raises(TypeError, match="include_hydrogens must be bool"):
        next(iterator)


@pytest.mark.parametrize("summary_coordinate_count", [0, 32])
@pytest.mark.parametrize("device_name", ["cpu", "cuda"])
def test_stream_helpers_filter_once_preserve_ids_and_allow_unused_h_labels(
    summary_coordinate_count: int,
    device_name: str,
) -> None:
    if device_name == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    batch = _hydrogen_variant_batch(pbc=torch.tensor([[True, False, False]] * 2)).to(
        device_name
    )
    types = torch.tensor(
        [41, 42, 901, 41, 42, 901], dtype=torch.int32, device=device_name
    )
    read = _read_from_batch(batch, types)
    deduplicated = deduplicate_stream(
        batch.num_graphs,
        read,
        type_vocabulary=(41, 42, 901),
        cutoff=4.0,
        threshold=0.3,
        include_hydrogens=False,
        device=device_name,
        max_batch_atoms=2,
        summary_coordinate_count=summary_coordinate_count,
    )
    assert deduplicated.retained_indices.tolist() == [0]
    assert deduplicated.representative_indices.tolist() == [0, 0]

    matches = torch.cat(
        list(
            iter_matches_stream(
                batch.num_graphs,
                read,
                type_vocabulary=(41, 42, 901),
                cutoff=4.0,
                threshold=0.3,
                other_count=batch.num_graphs,
                read_other_typed_batch=read,
                include_hydrogens=False,
                device=device_name,
                max_batch_atoms=2,
                summary_coordinate_count=summary_coordinate_count,
            )
        )
    )
    assert matches.tolist() == [[0, 0], [0, 1], [1, 0], [1, 1]]


@pytest.mark.parametrize("summary_coordinate_count", [0, 32])
@pytest.mark.parametrize("device_name", ["cpu", "cuda"])
def test_stream_empty_rows_and_empty_pool_keep_existing_behavior(
    summary_coordinate_count: int, device_name: str
) -> None:
    if device_name == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    batch = _make_batch(
        [
            torch.tensor([[0.0, 0, 0]]),
            torch.tensor([[4.0, 0, 0]]),
            torch.tensor([[8.0, 0, 0]]),
        ],
        [torch.tensor([1]), torch.tensor([1]), torch.tensor([6])],
    ).to(device_name)
    types = torch.tensor([901, 901, 41], dtype=torch.int32, device=device_name)
    read = _read_from_batch(batch, types)
    vocabulary = (41, 901)
    result = deduplicate_stream(
        3,
        read,
        type_vocabulary=vocabulary,
        cutoff=2.0,
        threshold=0.0,
        include_hydrogens=False,
        device=device_name,
        summary_coordinate_count=summary_coordinate_count,
    )
    assert result.retained_indices.tolist() == [0, 2]
    assert result.representative_indices.tolist() == [0, 0, 2]
    assert result.multiplicities.tolist() == [2, 1]
    matches = torch.cat(
        list(
            iter_matches_stream(
                3,
                read,
                type_vocabulary=vocabulary,
                cutoff=2.0,
                threshold=0.0,
                include_hydrogens=False,
                device=device_name,
                summary_coordinate_count=summary_coordinate_count,
            )
        )
    )
    assert matches.tolist() == [[0, 1]]

    empty = Batch(device=device_name)
    assert (
        RadialComparisonIndex.build(
            empty, cutoff=2.0, include_hydrogens=False
        ).num_structures
        == 0
    )

    def unexpected_read(_ids: torch.Tensor):
        raise AssertionError("empty pools must not call their loaders")

    empty_result = deduplicate_stream(
        0,
        unexpected_read,
        cutoff=2.0,
        threshold=0.0,
        include_hydrogens=False,
        device=device_name,
    )
    assert empty_result.retained_indices.numel() == 0
    assert (
        list(
            iter_matches_stream(
                0,
                unexpected_read,
                cutoff=2.0,
                threshold=0.0,
                include_hydrogens=False,
                device=device_name,
            )
        )
        == []
    )


def test_index_rejects_rows_with_no_retained_atoms_and_false_requires_numbers() -> None:
    hydrogen_only = _make_batch([torch.tensor([[0.0, 0, 0]])], [torch.tensor([1])])
    with pytest.raises(ValueError, match="structures must contain atoms"):
        RadialComparisonIndex.build(hydrogen_only, cutoff=2.0, include_hydrogens=False)

    storage = MultiLevelStorage.from_data(
        {"positions": torch.tensor([[0.0, 0, 0], [1.0, 0, 0]])},
        segment_lengths={"atoms": [2]},
        device="cpu",
        validate=False,
    )
    missing_numbers = Batch(device="cpu", storage=storage)
    with pytest.raises(ValueError, match="atomic_numbers is required"):
        RadialComparisonIndex.build(
            missing_numbers, cutoff=2.0, include_hydrogens=False
        )
