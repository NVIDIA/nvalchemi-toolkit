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
"""Tests for :func:`nvalchemi.data.transforms.make_supercell`."""

from __future__ import annotations

import itertools

import pytest
import torch

from nvalchemi.data.atomic_data import AtomicData
from nvalchemi.data.transforms import (
    DEFAULT_EXTENSIVE_SYSTEM_KEYS,
    DEFAULT_INTENSIVE_SYSTEM_KEYS,
    make_supercell,
)

_CELL = torch.tensor([[[4.0, 0.0, 0.0], [0.5, 3.0, 0.0], [0.0, 0.5, 5.0]]])
"""A skewed cell, so a wrong axis or offset order shows in the positions."""


def _make_data(**fields: torch.Tensor) -> AtomicData:
    """Build a two-atom periodic structure carrying *fields* on top."""
    return AtomicData(
        positions=torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.5, 2.0]]),
        atomic_numbers=torch.tensor([11, 17]),
        cell=_CELL.clone(),
        pbc=torch.tensor([[True, True, True]]),
        **fields,
    )


def _hand_tiled_positions(
    data: AtomicData, repeats: tuple[int, int, int]
) -> torch.Tensor:
    """Tile positions copy-major with the first axis slowest, as ASE does."""
    cell = data.cell.reshape(3, 3)
    copies = [
        data.positions + i * cell[0] + j * cell[1] + k * cell[2]
        for i, j, k in itertools.product(*(range(count) for count in repeats))
    ]
    return torch.cat(copies)


class TestMakeSupercellGeometry:
    """Positions and cell of the tiled structure."""

    def test_positions_are_tiled_copy_major_with_lattice_offsets(self) -> None:
        """Every copy is the whole primitive cell shifted by one lattice image."""
        data = _make_data()
        supercell = make_supercell(data, (2, 3, 1))
        torch.testing.assert_close(
            supercell.positions, _hand_tiled_positions(data, (2, 3, 1))
        )
        assert supercell.num_nodes == 12

    def test_the_cell_is_scaled_per_axis(self) -> None:
        """Each lattice vector grows by its own factor and no other."""
        supercell = make_supercell(_make_data(), (2, 3, 4))
        expected = _CELL.reshape(3, 3) * torch.tensor([[2.0], [3.0], [4.0]])
        torch.testing.assert_close(supercell.cell.reshape(3, 3), expected)

    def test_the_source_structure_is_left_unmodified(self) -> None:
        """Replication builds a new structure rather than growing the input."""
        data = _make_data()
        make_supercell(data, (2, 2, 2))
        assert data.num_nodes == 2
        torch.testing.assert_close(data.cell, _CELL)

    def test_positions_match_ase_repeat(self) -> None:
        """The layout is the one :meth:`ase.Atoms.repeat` produces."""
        ase = pytest.importorskip("ase")
        data = _make_data()
        atoms = ase.Atoms(
            numbers=data.atomic_numbers.tolist(),
            positions=data.positions.numpy(),
            cell=data.cell.reshape(3, 3).numpy(),
            pbc=True,
        ).repeat((2, 3, 2))
        supercell = make_supercell(data, (2, 3, 2))
        torch.testing.assert_close(
            supercell.positions.double(), torch.as_tensor(atoms.get_positions())
        )
        torch.testing.assert_close(
            supercell.cell.reshape(3, 3).double(), torch.as_tensor(atoms.cell.array)
        )


class TestMakeSupercellFields:
    """How node- and system-level fields travel into the supercell."""

    def test_node_fields_are_repeated_with_their_sites(self) -> None:
        """A declared and a custom node field both follow the atom they belong to."""
        data = _make_data(forces=torch.tensor([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]]))
        data.add_node_property("site_tag", torch.tensor([7, 9]))
        supercell = make_supercell(data, (1, 2, 1))
        assert supercell.atomic_numbers.tolist() == [11, 17, 11, 17]
        torch.testing.assert_close(supercell.forces, data.forces.repeat(2, 1))
        assert supercell.site_tag.tolist() == [7, 9, 7, 9]
        assert "site_tag" in supercell.__node_keys__

    def test_extensive_system_fields_are_multiplied_by_the_copy_count(self) -> None:
        """Energy and charge grow k-fold, as a size-extensive quantity must."""
        data = _make_data(energy=torch.tensor([[-3.0]]), charge=torch.tensor([[1.0]]))
        supercell = make_supercell(data, (2, 2, 1))
        assert supercell.energy.item() == pytest.approx(-12.0)
        assert supercell.charge.item() == pytest.approx(4.0)

    def test_intensive_system_fields_are_carried_unchanged(self) -> None:
        """Stress and periodicity describe the material, not the cell size."""
        stress = torch.arange(9.0).reshape(1, 3, 3)
        supercell = make_supercell(_make_data(stress=stress), (2, 1, 1))
        torch.testing.assert_close(supercell.stress, stress)
        assert supercell.pbc.tolist() == [[True, True, True]]

    def test_a_field_in_neither_set_is_refused_by_name(self) -> None:
        """The error names the field and the two keyword sets that would place it."""
        data = _make_data(spin=torch.tensor([[1.0]]))
        with pytest.raises(
            ValueError, match=r"\['spin'\].*extensive_keys.*intensive_keys"
        ):
            make_supercell(data, (2, 1, 1))

    def test_the_keyword_sets_place_a_custom_system_field(self) -> None:
        """A field the defaults do not know scales the way the caller declares."""
        data = _make_data()
        data.add_system_property("n_electrons", torch.tensor([[10.0]]))
        extensive = make_supercell(
            data,
            (2, 1, 1),
            extensive_keys={*DEFAULT_EXTENSIVE_SYSTEM_KEYS, "n_electrons"},
        )
        intensive = make_supercell(
            data,
            (2, 1, 1),
            intensive_keys={*DEFAULT_INTENSIVE_SYSTEM_KEYS, "n_electrons"},
        )
        assert extensive.n_electrons.item() == pytest.approx(20.0)
        assert intensive.n_electrons.item() == pytest.approx(10.0)
        assert "n_electrons" in extensive.__system_keys__

    def test_dropped_and_edge_fields_are_left_out(self) -> None:
        """Neighbor state does not survive tiling, and a dropped field is omitted."""
        data = _make_data(forces=torch.zeros(2, 3))
        data.neighbor_list = torch.tensor([[0, 1], [1, 0]])
        supercell = make_supercell(data, (2, 1, 1), drop_keys={"forces"})
        assert supercell.forces is None
        assert supercell.neighbor_list is None

    def test_a_dropped_system_field_is_not_refused(self) -> None:
        """Dropping a field with no defined scaling is the third remedy."""
        data = _make_data(spin=torch.tensor([[1.0]]))
        supercell = make_supercell(data, (2, 1, 1), drop_keys={"spin"})
        assert supercell.spin is None


class TestMakeSupercellValidation:
    """Inputs the transform refuses."""

    @pytest.mark.parametrize("repeats", [(2, 1), (0, 1, 1), (2, -1, 1), (1, 1, 1, 1)])
    def test_invalid_repeat_counts_are_rejected(self, repeats: tuple[int, ...]) -> None:
        """Three positive integers, nothing else."""
        with pytest.raises(ValueError, match="three positive integers"):
            make_supercell(_make_data(), repeats)

    def test_a_structure_without_a_cell_is_rejected(self) -> None:
        """A cluster has no lattice vectors to tile along."""
        data = AtomicData(
            positions=torch.zeros(2, 3), atomic_numbers=torch.tensor([1, 1])
        )
        with pytest.raises(ValueError, match="no cell"):
            make_supercell(data, (2, 1, 1))
