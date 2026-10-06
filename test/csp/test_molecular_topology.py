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
"""Tests for exact atomic-number-labelled topology orbits."""

from __future__ import annotations

import random
import subprocess
import sys
from itertools import permutations

import pytest
import torch

from nvalchemi import OptionalDependency, OptionalDependencyError
from nvalchemi.csp._chem import _topology
from nvalchemi.csp.chem import (
    TopologicalAtomTypeMap,
    topological_atom_types_from_connectivity,
    topological_atom_types_from_mol,
)
from nvalchemi.data import AtomicData, Batch


def test_core_and_chem_import_without_loading_rdkit() -> None:
    script = """
import sys
from importlib.abc import MetaPathFinder

class BlockRdkit(MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'rdkit' or fullname.startswith('rdkit.'):
            raise AssertionError('unexpected RDKit import: ' + fullname)
        return None

sys.meta_path.insert(0, BlockRdkit())
import nvalchemi.csp
import nvalchemi.csp.chem
assert not any(name == 'rdkit' or name.startswith('rdkit.') for name in sys.modules)
"""
    subprocess.run([sys.executable, "-c", script], check=True)  # noqa: S603


def _adjacency(count: int, edges: list[tuple[int, int]]) -> torch.Tensor:
    result = torch.zeros((count, count), dtype=torch.bool)
    for left, right in edges:
        result[left, right] = result[right, left] = True
    return result


def _brute_force_orbits(
    atomic_numbers: list[int], adjacency: list[list[bool]]
) -> list[int]:
    """Independent exhaustive permutation oracle for small graphs."""
    count = len(atomic_numbers)
    related = [[False] * count for _ in range(count)]
    for permutation in permutations(range(count)):
        if any(
            atomic_numbers[i] != atomic_numbers[permutation[i]] for i in range(count)
        ):
            continue
        if any(
            adjacency[i][j] != adjacency[permutation[i]][permutation[j]]
            for i in range(count)
            for j in range(count)
        ):
            continue
        for index, mapped in enumerate(permutation):
            related[index][mapped] = True
    ids = [-1] * count
    next_type = 0
    for index in range(count):
        if ids[index] >= 0:
            continue
        for other in range(index, count):
            if related[index][other]:
                ids[other] = next_type
        next_type += 1
    return ids


@pytest.mark.parametrize(
    "numbers, edges",
    [
        ([6], []),
        ([6, 6, 6], [(0, 1), (1, 2)]),
        ([6, 6, 6, 6], [(0, 1), (1, 2), (2, 3), (3, 0)]),
        (
            [6, 6, 6, 6, 6, 6, 6],
            [(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 6), (6, 3)],
        ),
        ([6, 8, 6, 8, 6], [(0, 1), (1, 2), (2, 3), (3, 4)]),
        ([6, 6, 6, 6, 6, 6], [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]),
    ],
)
def test_types_match_independent_permutation_oracle(numbers, edges) -> None:
    adjacency = _adjacency(len(numbers), edges)
    result = topological_atom_types_from_connectivity(
        torch.tensor(numbers, dtype=torch.int64), adjacency
    )
    assert result.tolist() == _brute_force_orbits(numbers, adjacency.tolist())


def test_seeded_small_graphs_match_independent_permutation_oracle() -> None:
    generator = random.Random(20260927)
    for _ in range(12):
        count = generator.randrange(2, 7)
        numbers = [generator.choice((6, 7, 8)) for _ in range(count)]
        edges = [
            (left, right)
            for left in range(count)
            for right in range(left + 1, count)
            if generator.random() < 0.35
        ]
        adjacency = _adjacency(count, edges)
        result = topological_atom_types_from_connectivity(
            torch.tensor(numbers, dtype=torch.int64), adjacency
        )
        assert result.tolist() == _brute_force_orbits(numbers, adjacency.tolist())


def test_disconnected_triangle_and_square_split_degree_two_atoms() -> None:
    numbers = torch.full((7,), 6, dtype=torch.int64)
    adjacency = _adjacency(
        7,
        [(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 6), (6, 3)],
    )
    types = topological_atom_types_from_connectivity(numbers, adjacency)
    assert types.tolist() == [0, 0, 0, 1, 1, 1, 1]


def test_identical_disconnected_copies_share_types_and_ids_are_first_index_order() -> (
    None
):
    numbers = torch.tensor([8, 6, 8, 6], dtype=torch.int32)
    adjacency = _adjacency(4, [(0, 1), (2, 3)])
    result = topological_atom_types_from_connectivity(numbers, adjacency)
    assert result.tolist() == [0, 1, 0, 1]


@pytest.mark.parametrize("numbers, edges, expected", [([], [], []), ([1], [], [0])])
def test_empty_and_single_atom_graphs(numbers, edges, expected) -> None:
    result = topological_atom_types_from_connectivity(
        torch.tensor(numbers, dtype=torch.int64), _adjacency(len(numbers), edges)
    )
    assert result.dtype == torch.int32
    assert result.tolist() == expected


@pytest.mark.parametrize(
    "numbers, adjacency, error, message",
    [
        (
            torch.tensor([[6]]),
            torch.zeros((1, 1), dtype=torch.int64),
            ValueError,
            r"shape \[N\]",
        ),
        (
            torch.tensor([0]),
            torch.zeros((1, 1), dtype=torch.int64),
            ValueError,
            "1 through 118",
        ),
        (
            torch.tensor([119]),
            torch.zeros((1, 1), dtype=torch.int64),
            ValueError,
            "1 through 118",
        ),
        (
            torch.tensor([6.0]),
            torch.zeros((1, 1), dtype=torch.int64),
            TypeError,
            "integral dtype",
        ),
        (
            torch.tensor([True]),
            torch.zeros((1, 1), dtype=torch.int64),
            TypeError,
            "integral dtype",
        ),
        (
            torch.tensor([6]),
            torch.ones((1, 1), dtype=torch.int64),
            ValueError,
            "zero diagonal",
        ),
        (torch.tensor([6, 6]), torch.tensor([[0, 1], [0, 0]]), ValueError, "symmetric"),
        (torch.tensor([6]), torch.tensor([[2]]), ValueError, "binary values"),
        (
            torch.tensor([6, 6]),
            torch.zeros((1, 1), dtype=torch.int64),
            ValueError,
            r"shape \[N, N\]",
        ),
        (
            torch.tensor([6]),
            torch.zeros((1, 1), dtype=torch.float32),
            TypeError,
            "boolean or integral",
        ),
    ],
)
def test_connectivity_rejects_invalid_inputs(
    numbers, adjacency, error, message
) -> None:
    with pytest.raises(error, match=message):
        topological_atom_types_from_connectivity(numbers, adjacency)


def test_connectivity_requires_tensor_inputs() -> None:
    with pytest.raises(TypeError, match="atomic_numbers"):
        topological_atom_types_from_connectivity([6], torch.zeros((1, 1)))  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="adjacency"):
        topological_atom_types_from_connectivity(torch.tensor([6]), [[False]])  # type: ignore[arg-type]


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_connectivity_returns_int32_on_atomic_number_device(device: str) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    numbers = torch.tensor([6, 6, 6], dtype=torch.int64, device=device)
    adjacency = _adjacency(3, [(0, 1), (1, 2)]).to(device)
    result = topological_atom_types_from_connectivity(numbers, adjacency)
    assert result.device == numbers.device
    assert result.dtype == torch.int32
    assert result.tolist() == [0, 1, 0]


def _ethanol_graph() -> tuple[torch.Tensor, torch.Tensor]:
    numbers = torch.tensor([6, 1, 1, 1, 6, 1, 1, 8, 1], dtype=torch.int64)
    adjacency = _adjacency(
        9,
        [(0, 1), (0, 2), (0, 3), (0, 4), (4, 5), (4, 6), (4, 7), (7, 8)],
    )
    return numbers, adjacency


def test_shared_map_preserves_labels_for_reordered_asymmetric_molecule() -> None:
    numbers, adjacency = _ethanol_graph()
    atom_map = TopologicalAtomTypeMap(numbers, adjacency)
    permutation = [7, 8, 4, 6, 5, 0, 3, 1, 2]
    target_numbers = numbers[permutation]
    target_adjacency = adjacency[permutation][:, permutation]

    labels = atom_map.for_connectivity(target_numbers, target_adjacency)

    expected = topological_atom_types_from_connectivity(numbers, adjacency)[permutation]
    assert torch.equal(labels, expected)
    assert labels.tolist() == [4, 5, 2, 3, 3, 0, 1, 1, 1]


def test_map_snapshots_inputs_and_supports_multiple_reference_components() -> None:
    # Ordered reference components are water and carbon dioxide.
    numbers = torch.tensor([8, 1, 1, 8, 6, 8], dtype=torch.int64)
    adjacency = _adjacency(6, [(0, 1), (0, 2), (3, 4), (4, 5)])
    expected = topological_atom_types_from_connectivity(numbers, adjacency)
    atom_map = TopologicalAtomTypeMap(numbers, adjacency)
    numbers.fill_(7)
    adjacency.fill_(False)

    # A reordered CO2 component followed by a reordered water component.
    target_numbers = torch.tensor([8, 6, 8, 1, 8, 1], dtype=torch.int64)
    target_adjacency = _adjacency(6, [(0, 1), (1, 2), (3, 4), (4, 5)])
    labels = atom_map.for_connectivity(target_numbers, target_adjacency)
    assert labels.tolist() == [
        expected[5].item(),
        expected[4].item(),
        expected[3].item(),
        expected[2].item(),
        expected[0].item(),
        expected[1].item(),
    ]


def test_map_rejects_unknown_target_component() -> None:
    numbers, adjacency = _ethanol_graph()
    atom_map = TopologicalAtomTypeMap(numbers, adjacency)
    with pytest.raises(ValueError, match="no isomorphic reference component"):
        atom_map.for_connectivity(
            torch.tensor([6, 6, 6, 6]),
            _adjacency(4, [(0, 1), (1, 2), (2, 3), (3, 0)]),
        )


def test_for_connectivity_does_not_compute_target_orbit_labels(monkeypatch) -> None:
    numbers, adjacency = _ethanol_graph()
    atom_map = TopologicalAtomTypeMap(numbers, adjacency)

    def unexpected_orbit_calculation(*args, **kwargs):
        raise AssertionError("target orbit labels are not needed for shared mapping")

    monkeypatch.setattr(
        _topology, "topological_atom_types", unexpected_orbit_calculation
    )
    permutation = [7, 8, 4, 6, 5, 0, 3, 1, 2]
    result = atom_map.for_connectivity(
        numbers[permutation], adjacency[permutation][:, permutation]
    )
    assert result.tolist() == [4, 5, 2, 3, 3, 0, 1, 1, 1]


def _batch(numbers: list[int], source_indices: list[int]) -> Batch:
    structure = AtomicData(
        positions=torch.zeros((len(numbers), 3), dtype=torch.float32),
        atomic_numbers=torch.tensor(numbers, dtype=torch.int64),
    )
    batch = Batch.from_data_list([structure])
    batch.add_key(
        "csp_source_asu_atom_index",
        [torch.tensor(source_indices, dtype=torch.int32)],
        level="node",
    )
    return batch


def test_for_batch_maps_mixed_formula_copies_by_aligned_provenance() -> None:
    numbers, adjacency = _ethanol_graph()
    atom_map = TopologicalAtomTypeMap(numbers, adjacency)
    source_indices = [[7, 8, 4, 6, 5, 0, 3, 1, 2], list(reversed(range(18)))]
    structures = [
        AtomicData(
            positions=torch.zeros((len(indices), 3), dtype=torch.float32),
            atomic_numbers=torch.tensor(
                [numbers[index % len(numbers)].item() for index in indices],
                dtype=torch.int64,
            ),
        )
        for indices in source_indices
    ]
    batch = Batch.from_data_list(structures)
    batch.add_key(
        "csp_source_asu_atom_index",
        [torch.tensor(indices, dtype=torch.int32) for indices in source_indices],
        level="node",
    )
    mapped = atom_map.for_batch(batch)
    expected = topological_atom_types_from_connectivity(numbers, adjacency)
    assert mapped.tolist() == [
        expected[index % len(numbers)].item()
        for structure_indices in source_indices
        for index in structure_indices
    ]


@pytest.mark.parametrize(
    "numbers, source_indices, message",
    [
        ([6, 1, 1, 1, 6, 1, 1, 8, 1], [-1, 1, 2, 3, 4, 5, 6, 7, 8], "nonnegative"),
        (
            [8, 1, 1, 1, 6, 1, 1, 8, 1],
            [0, 1, 2, 3, 4, 5, 6, 7, 8],
            "disagree with source provenance",
        ),
    ],
)
def test_for_batch_rejects_invalid_provenance(
    numbers: list[int], source_indices: list[int], message: str
) -> None:
    reference_numbers, adjacency = _ethanol_graph()
    atom_map = TopologicalAtomTypeMap(reference_numbers, adjacency)
    with pytest.raises(ValueError, match=message):
        atom_map.for_batch(_batch(numbers, source_indices))


def test_rdkit_adapter_matches_connectivity_and_ignores_annotations() -> None:
    Chem = pytest.importorskip("rdkit.Chem")
    molecule = Chem.MolFromSmiles("CCO")
    assert molecule is not None
    baseline = topological_atom_types_from_mol(molecule)
    assert baseline.dtype == torch.int32 and baseline.device.type == "cpu"

    editable = Chem.RWMol()
    for number in (6, 6, 8):
        atom = Chem.Atom(number)
        atom.SetIsotope(13)
        atom.SetFormalCharge(1)
        editable.AddAtom(atom)
    editable.AddBond(0, 1, Chem.BondType.DOUBLE)
    editable.AddBond(1, 2, Chem.BondType.TRIPLE)
    annotated = editable.GetMol()
    expected = topological_atom_types_from_connectivity(
        torch.tensor([6, 6, 8]), _adjacency(3, [(0, 1), (1, 2)])
    )
    assert torch.equal(baseline, expected)
    assert torch.equal(topological_atom_types_from_mol(annotated), expected)


def test_rdkit_adapter_rejects_non_molecule() -> None:
    pytest.importorskip("rdkit.Chem")
    with pytest.raises(TypeError, match="rdkit.Chem.Mol"):
        topological_atom_types_from_mol(object())  # type: ignore[arg-type]


def test_rdkit_adapter_uses_optional_dependency_guard(monkeypatch) -> None:
    monkeypatch.setattr(OptionalDependency.RDKIT, "_available", False)
    monkeypatch.setattr(
        OptionalDependency.RDKIT,
        "_import_error",
        ModuleNotFoundError("No module named 'rdkit'"),
    )
    with pytest.raises(OptionalDependencyError, match=r"nvalchemi-toolkit\[rdkit\]"):
        topological_atom_types_from_mol(None)  # type: ignore[arg-type]
