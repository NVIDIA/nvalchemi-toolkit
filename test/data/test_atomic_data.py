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
"""Comprehensive tests for AtomicData (Pydantic + DataMixin atomic system representation)."""

from __future__ import annotations

import runpy
import textwrap
import warnings
from pathlib import Path

import numpy as np
import pytest
import torch
from ase import Atoms

from nvalchemi import _typing as t
from nvalchemi.data.atomic_data import (
    AtomicData,
    AtomicNumberTable,
    atomic_numbers_to_indices,
    to_one_hot,
    voigt_to_matrix,
)


def _minimal_atomic_data(
    num_nodes: int = 4,
    num_edges: int = 0,
    device: str | torch.device = "cpu",
) -> AtomicData:
    """Build minimal AtomicData for tests."""
    positions = torch.randn(num_nodes, 3, device=device)
    atomic_numbers = torch.ones(num_nodes, dtype=torch.long, device=device)
    kwargs: dict = {"positions": positions, "atomic_numbers": atomic_numbers}
    if num_edges > 0:
        edge_index = torch.zeros(num_edges, 2, dtype=torch.long, device=device)
        kwargs["edge_index"] = edge_index
    return AtomicData(**kwargs)


# -----------------------------------------------------------------------------
# Construction and required fields
# -----------------------------------------------------------------------------
class TestAtomicDataConstruction:
    """Tests for AtomicData construction and required fields."""

    def test_minimal_construction(self):
        positions = torch.randn(4, 3)
        atomic_numbers = torch.ones(4, dtype=torch.long)
        data = AtomicData(positions=positions, atomic_numbers=atomic_numbers)
        assert data.positions.shape == (4, 3)
        assert data.atomic_numbers.shape == (4,)
        assert data.edge_index is None

    def test_with_edge_index(self):
        data = _minimal_atomic_data(4, num_edges=6)
        assert data.edge_index is not None
        assert data.edge_index.shape == (6, 2)

    def test_optional_system_fields(self):
        data = AtomicData(
            positions=torch.randn(2, 3),
            atomic_numbers=torch.ones(2, dtype=torch.long),
            cell=torch.eye(3).unsqueeze(0),
            pbc=torch.tensor([[True, True, False]]),
            energies=torch.tensor([[1.0]]),
        )
        assert data.cell.shape == (1, 3, 3)
        assert data.pbc.shape == (1, 3)
        assert data.energies.shape == (1, 1)

    @pytest.mark.parametrize("int_dtype", [torch.int32, torch.int64])
    def test_integer_dtypes_accepted(self, int_dtype: torch.dtype):
        """AtomicData should accept both int32 and int64 for integer fields."""
        data = AtomicData(
            positions=torch.randn(3, 3),
            atomic_numbers=torch.tensor([1, 6, 8], dtype=int_dtype),
            edge_index=torch.zeros(4, 2, dtype=int_dtype),
        )
        assert data.atomic_numbers.dtype == int_dtype
        assert data.edge_index.dtype == int_dtype

    def test_optional_node_fields(self):
        data = AtomicData(
            positions=torch.randn(3, 3),
            atomic_numbers=torch.ones(3, dtype=torch.long),
            forces=torch.randn(3, 3),
        )
        assert data.forces.shape == (3, 3)


# -----------------------------------------------------------------------------
# Properties: num_nodes, num_edges, device, dtype
# -----------------------------------------------------------------------------
class TestAtomicDataProperties:
    """Tests for num_nodes, num_edges, device, dtype."""

    def test_num_nodes(self):
        data = _minimal_atomic_data(5)
        assert data.num_nodes == 5

    def test_num_edges_none(self):
        data = _minimal_atomic_data(4, num_edges=0)
        assert data.num_edges == 0

    def test_num_edges_with_edges(self):
        data = _minimal_atomic_data(4, num_edges=10)
        assert data.num_edges == 10

    def test_device(self):
        data = _minimal_atomic_data(2)
        assert data.device == data.positions.device
        data_cuda = AtomicData(
            positions=torch.randn(2, 3, device="cpu"),
            atomic_numbers=torch.ones(2, dtype=torch.long, device="cpu"),
        )
        assert data_cuda.device.type == "cpu"

    def test_dtype(self):
        data = AtomicData(
            positions=torch.randn(2, 3, dtype=torch.float64),
            atomic_numbers=torch.ones(2, dtype=torch.long),
        )
        assert data.dtype == torch.float64


# -----------------------------------------------------------------------------
# Class-level keys
# -----------------------------------------------------------------------------
class TestAtomicDataKeys:
    """Tests for _default_node_keys, _default_edge_keys, _default_system_keys."""

    def test_node_keys_contains_positions(self):
        assert "positions" in AtomicData._default_node_keys
        assert "atomic_numbers" in AtomicData._default_node_keys

    def test_edge_keys_contains_edge_index(self):
        assert "edge_index" in AtomicData._default_edge_keys
        assert "shifts" in AtomicData._default_edge_keys

    def test_system_keys_contains_cell_energies(self):
        assert "cell" in AtomicData._default_system_keys
        assert "energies" in AtomicData._default_system_keys


# -----------------------------------------------------------------------------
# Node / edge / system properties and add_*_property
# -----------------------------------------------------------------------------
class TestAtomicDataPropertiesDict:
    """Tests for node_properties, edge_properties, system_properties, add_*_property."""

    def test_node_properties(self):
        data = _minimal_atomic_data(3)
        node_props = data.node_properties
        assert "positions" in node_props
        assert "atomic_numbers" in node_props
        assert node_props["positions"].shape == (3, 3)

    def test_edge_properties_empty_without_edges(self):
        data = _minimal_atomic_data(3)
        edge_props = data.edge_properties
        assert isinstance(edge_props, dict)

    def test_system_properties(self):
        data = AtomicData(
            positions=torch.randn(2, 3),
            atomic_numbers=torch.ones(2, dtype=torch.long),
            energies=torch.tensor([[1.0]]),
        )
        sys_props = data.system_properties
        assert "energies" in sys_props

    def test_add_node_property(self):
        data = _minimal_atomic_data(3)
        data.add_node_property("custom_node", torch.randn(3, 5))
        assert "custom_node" in data.__node_keys__
        assert data["custom_node"].shape == (3, 5)

    def test_add_edge_property(self):
        data = _minimal_atomic_data(4, num_edges=6)
        data.add_edge_property("custom_edge", torch.randn(6, 2))
        assert "custom_edge" in data.__edge_keys__

    def test_add_system_property(self):
        data = _minimal_atomic_data(2)
        data.add_system_property("custom_sys", torch.tensor([1.0]))
        assert "custom_sys" in data.__system_keys__

    def test_add_node_property_does_not_pollute_other_instances(self):
        a = _minimal_atomic_data(3)
        b = _minimal_atomic_data(4)
        a.add_node_property("custom_feat", torch.randn(3, 4))
        assert "custom_feat" in a.__node_keys__
        assert "custom_feat" not in b.__node_keys__

    def test_add_edge_property_does_not_pollute_other_instances(self):
        a = _minimal_atomic_data(3, num_edges=4)
        b = _minimal_atomic_data(3, num_edges=4)
        a.add_edge_property("custom_edge", torch.randn(4, 2))
        assert "custom_edge" in a.__edge_keys__
        assert "custom_edge" not in b.__edge_keys__

    def test_add_system_property_does_not_pollute_other_instances(self):
        a = _minimal_atomic_data(3)
        b = _minimal_atomic_data(3)
        a.add_system_property("custom_sys", torch.tensor([1.0]))
        assert "custom_sys" in a.__system_keys__
        assert "custom_sys" not in b.__system_keys__

    def test_add_node_property_does_not_pollute_future_instances(self):
        a = _minimal_atomic_data(3)
        a.add_node_property("custom_feat", torch.randn(3, 4))
        c = _minimal_atomic_data(2)
        assert "custom_feat" not in c.__node_keys__

    def test_add_edge_property_does_not_pollute_future_instances(self):
        a = _minimal_atomic_data(3, num_edges=4)
        a.add_edge_property("custom_edge", torch.randn(4, 2))
        c = _minimal_atomic_data(3, num_edges=4)
        assert "custom_edge" not in c.__edge_keys__

    def test_add_system_property_does_not_pollute_future_instances(self):
        a = _minimal_atomic_data(3)
        a.add_system_property("custom_sys", torch.tensor([1.0]))
        c = _minimal_atomic_data(3)
        assert "custom_sys" not in c.__system_keys__

    def test_multiple_add_node_property_preserves_all_keys(self):
        data = _minimal_atomic_data(3)
        data.add_node_property("feat_a", torch.randn(3, 4))
        data.add_node_property("feat_b", torch.randn(3, 2))
        data.add_node_property("feat_c", torch.randn(3, 1))
        assert "feat_a" in data.__node_keys__
        assert "feat_b" in data.__node_keys__
        assert "feat_c" in data.__node_keys__

    def test_class_level_defaults_are_immutable(self):
        assert isinstance(AtomicData._default_node_keys, frozenset)
        assert isinstance(AtomicData._default_edge_keys, frozenset)
        assert isinstance(AtomicData._default_system_keys, frozenset)


# -----------------------------------------------------------------------------
# Validation
# -----------------------------------------------------------------------------
class TestAtomicDataValidation:
    """Tests for Pydantic/model validators (node/edge consistency, dtypes)."""

    def test_node_inconsistency_raises(self):
        with pytest.raises(ValueError, match="Inconsistent number of atoms"):
            AtomicData(
                positions=torch.randn(4, 3),
                atomic_numbers=torch.ones(3, dtype=torch.long),
            )

    def test_edge_inconsistency_raises(self):
        with pytest.raises(ValueError, match="Inconsistent number of edges"):
            AtomicData(
                positions=torch.randn(4, 3),
                atomic_numbers=torch.ones(4, dtype=torch.long),
                edge_index=torch.zeros(5, 2, dtype=torch.long),
                shifts=torch.randn(3, 3),
            )

    def test_default_masses_filled(self):
        data = AtomicData(
            positions=torch.randn(2, 3),
            atomic_numbers=torch.tensor([1, 6], dtype=torch.long),
        )
        assert data.atomic_masses is not None
        assert data.atomic_masses.shape == (2,)

    def test_atom_categories_list_of_enum(self):
        """atom_categories as list of AtomCategory is converted to tensor."""
        data = AtomicData(
            positions=torch.randn(3, 3),
            atomic_numbers=torch.ones(3, dtype=torch.long),
            atom_categories=[
                t.AtomCategory.GAS,
                t.AtomCategory.SURFACE,
                t.AtomCategory.BULK,
            ],
        )
        assert data.atom_categories is not None
        assert data.atom_categories.shape == (3,)
        assert data.atom_categories.dtype == torch.long

    def test_use_default_velocities(self):
        """When velocities is None, validator sets zeros like positions."""
        data = AtomicData(
            positions=torch.randn(2, 3),
            atomic_numbers=torch.ones(2, dtype=torch.long),
        )
        assert data.velocities is not None
        assert data.velocities.shape == data.positions.shape
        assert data.velocities.eq(0).all()


# -----------------------------------------------------------------------------
# Chemical hash and equality
# -----------------------------------------------------------------------------
class TestAtomicDataHashAndEq:
    """Tests for chemical_hash and __eq__."""

    def test_chemical_hash_deterministic(self):
        data = _minimal_atomic_data(3)
        h1 = data.chemical_hash
        h2 = data.chemical_hash
        assert h1 == h2
        assert isinstance(h1, str)
        assert len(h1) == 64

    def test_eq_same_structure(self):
        d1 = AtomicData(
            positions=torch.randn(2, 3),
            atomic_numbers=torch.ones(2, dtype=torch.long),
        )
        d2 = AtomicData(
            positions=d1.positions.clone(),
            atomic_numbers=d1.atomic_numbers.clone(),
        )
        assert d1 == d2

    def test_eq_different_type(self):
        data = _minimal_atomic_data(2)
        assert data != "not AtomicData"
        assert data is not None

    def test_chemical_hash_non_periodic(self):
        """chemical_hash when pbc/cell are None uses empty pbc/cell string."""
        data = AtomicData(
            positions=torch.randn(2, 3),
            atomic_numbers=torch.ones(2, dtype=torch.long),
        )
        assert data.pbc is None
        assert data.cell is None
        h = data.chemical_hash
        assert isinstance(h, str)
        assert len(h) == 64

    def test_chemical_hash_periodic(self):
        """chemical_hash when pbc and cell are set includes them in the hash."""
        data = AtomicData(
            positions=torch.randn(2, 3),
            atomic_numbers=torch.ones(2, dtype=torch.long),
            pbc=torch.tensor([[True, True, False]]),
            cell=torch.eye(3).unsqueeze(0),
        )
        h = data.chemical_hash
        assert isinstance(h, str)
        assert len(h) == 64
        # Same structure should give same hash
        data2 = AtomicData(
            positions=data.positions.clone(),
            atomic_numbers=data.atomic_numbers.clone(),
            pbc=data.pbc.clone(),
            cell=data.cell.clone(),
        )
        assert data2.chemical_hash == h


# -----------------------------------------------------------------------------
# __getitem__, __setitem__, indexing
# -----------------------------------------------------------------------------
class TestAtomicDataAccess:
    """Tests for __getitem__, __setitem__."""

    def test_getitem(self):
        data = _minimal_atomic_data(2)
        assert data["positions"] is data.positions
        assert data["atomic_numbers"] is data.atomic_numbers

    def test_setitem(self):
        data = _minimal_atomic_data(2)
        new_pos = torch.randn(2, 3)
        data["positions"] = new_pos
        assert data.positions is new_pos

    def test_tensor_serialization_via_model_dump_json(self):
        """JSON serialization uses _tensor_serialization (tolist) for tensor fields."""
        data = _minimal_atomic_data(2)
        json_str = data.model_dump_json()
        assert isinstance(json_str, str)
        import json

        parsed = json.loads(json_str)
        assert "positions" in parsed
        assert isinstance(parsed["positions"], list)


# -----------------------------------------------------------------------------
# AtomicNumberTable
# -----------------------------------------------------------------------------
class TestAtomicNumberTable:
    """Tests for AtomicNumberTable helper."""

    def test_len(self):
        table = AtomicNumberTable([1, 6, 8])
        assert len(table) == 3

    def test_index_to_z(self):
        table = AtomicNumberTable([1, 6, 8])
        assert table.index_to_z(0) == 1
        assert table.index_to_z(1) == 6

    def test_z_to_index(self):
        table = AtomicNumberTable([1, 6, 8])
        assert table.z_to_index(1) == 0
        assert table.z_to_index(6) == 1

    def test_atomic_numbers_to_indices(self):
        import numpy as np

        table = AtomicNumberTable([1, 6, 8])
        z_arr = np.array([6, 1, 8])
        indices = atomic_numbers_to_indices(z_arr, table)
        assert list(indices) == [1, 0, 2]

    def test_str(self):
        table = AtomicNumberTable([1, 6, 8])
        assert "AtomicNumberTable" in str(table)
        assert "1" in str(table) and "6" in str(table)


# -----------------------------------------------------------------------------
# Module helpers: to_one_hot, voigt_to_matrix
# -----------------------------------------------------------------------------


class TestAtomicDataModuleHelpers:
    """Tests for to_one_hot and voigt_to_matrix."""

    def test_to_one_hot(self):
        indices = torch.tensor([[0], [2], [1]], dtype=torch.long)
        oh = to_one_hot(indices, num_classes=4)
        assert oh.shape == (3, 4)
        assert oh[0].tolist() == [1, 0, 0, 0]
        assert oh[1].tolist() == [0, 0, 1, 0]
        assert oh[2].tolist() == [0, 1, 0, 0]

    def test_voigt_to_matrix_3x3_passthrough(self):
        m = torch.eye(3)
        out = voigt_to_matrix(m)
        assert out.shape == (3, 3)
        assert torch.equal(out, m)

    def test_voigt_to_matrix_6(self, device):
        v = torch.tensor([1.0, 2.0, 3.0, 0.0, 0.0, 0.0], device=device)
        m = voigt_to_matrix(v)
        assert m.shape == (3, 3)
        assert m.device.type == device
        assert m[0, 0].item() == 1.0
        assert m[1, 1].item() == 2.0
        assert m[2, 2].item() == 3.0

    def test_voigt_to_matrix_9(self):
        v = torch.arange(9, dtype=torch.float).view(9)
        m = voigt_to_matrix(v)
        assert m.shape == (3, 3)

    def test_voigt_to_matrix_invalid_shape_raises(self):
        with pytest.raises(ValueError, match="Stress tensor must be"):
            voigt_to_matrix(torch.zeros(5))


# -----------------------------------------------------------------------------
# from_atoms
# -----------------------------------------------------------------------------
class TestFromAtoms:
    """Tests for AtomicData.from_atoms — no zero fabrication, no input mutation."""

    def test_no_labels_fields_are_none(self):
        """Bare Atoms with no DFT data produces None for all optional label fields."""
        atoms = Atoms(numbers=[1, 1], positions=[[0, 0, 0], [0, 0, 1]])
        data = AtomicData.from_atoms(atoms)
        assert data.energies is None
        assert data.forces is None
        assert data.stresses is None
        assert data.virials is None
        assert data.dipoles is None
        assert data.node_charges is None
        assert data.graph_charges is None

    def test_explicit_charge_no_per_atom_charges(self):
        """atoms.info['charge'] populates graph_charges even without per-atom charges."""
        atoms = Atoms(numbers=[8, 1, 1], positions=[[0, 0, 0], [1, 0, 0], [0, 1, 0]])
        atoms.info["charge"] = 0
        data = AtomicData.from_atoms(atoms)
        assert data.graph_charges is not None
        assert data.graph_charges.shape == (1, 1)
        assert data.graph_charges.item() == 0.0
        assert data.node_charges is None

    def test_explicit_charge_numpy_int(self):
        """np.int64 charge is accepted via numbers.Integral."""
        atoms = Atoms(numbers=[1, 1], positions=[[0, 0, 0], [0, 0, 1]])
        atoms.info["charge"] = np.int64(1)
        data = AtomicData.from_atoms(atoms)
        assert data.graph_charges is not None
        assert data.graph_charges.item() == 1.0

    def test_non_integer_charge_raises(self):
        """atoms.info['charge'] with a float value must raise ValueError."""
        atoms = Atoms(numbers=[1, 1], positions=[[0, 0, 0], [0, 0, 1]])
        atoms.info["charge"] = 1.5
        with pytest.raises(ValueError, match="must be an integer"):
            AtomicData.from_atoms(atoms)

    def test_does_not_mutate_input_atoms_info(self):
        """from_atoms must not modify the caller's atoms.info dict."""
        atoms = Atoms(numbers=[1, 1], positions=[[0, 0, 0], [0, 0, 1]])
        atoms.info["energy"] = -1.5
        atoms.info["custom_array"] = np.array([1.0, 2.0])
        atoms.info["label"] = "test_run"
        atoms.info["flag"] = True
        atoms.info["charge"] = 0

        snapshot_keys = set(atoms.info.keys())
        snapshot_types = {k: type(v) for k, v in atoms.info.items()}
        snapshot_scalars = {
            k: v for k, v in atoms.info.items() if not isinstance(v, np.ndarray)
        }
        snapshot_arrays = {
            k: v.copy() for k, v in atoms.info.items() if isinstance(v, np.ndarray)
        }

        AtomicData.from_atoms(atoms)

        assert set(atoms.info.keys()) == snapshot_keys
        for k in snapshot_keys:
            assert type(atoms.info[k]) is snapshot_types[k], (
                f"atoms.info['{k}'] type changed from {snapshot_types[k]} "
                f"to {type(atoms.info[k])}"
            )
        for k, v in snapshot_scalars.items():
            assert atoms.info[k] == v
        for k, v in snapshot_arrays.items():
            assert np.array_equal(atoms.info[k], v)

    def test_returned_info_filtering(self):
        """Returned data.info includes ints/floats/arrays, drops bools/strings/consumed keys."""
        atoms = Atoms(numbers=[1, 1], positions=[[0, 0, 0], [0, 0, 1]])
        atoms.info["my_float"] = 3.14
        atoms.info["my_int"] = 42
        atoms.info["my_np_int"] = np.int64(7)
        atoms.info["my_array"] = np.array([1.0, 2.0, 3.0])
        atoms.info["my_list"] = [10.0, 20.0]
        atoms.info["my_bool"] = True
        atoms.info["my_np_bool"] = np.bool_(False)
        atoms.info["my_str"] = "hello"
        atoms.info["my_dict"] = {"a": 1}
        atoms.info["energy"] = -1.5
        atoms.info["charge"] = 1

        data = AtomicData.from_atoms(atoms)

        assert "my_float" in data.info
        assert "my_int" in data.info
        assert "my_np_int" in data.info
        assert "my_array" in data.info
        assert "my_list" in data.info
        assert "my_bool" not in data.info
        assert "my_np_bool" not in data.info
        assert "my_str" not in data.info
        assert "my_dict" not in data.info
        assert "energy" not in data.info
        assert "charge" not in data.info

        assert isinstance(data.info["my_float"], torch.Tensor)
        assert isinstance(data.info["my_int"], torch.Tensor)
        assert isinstance(data.info["my_array"], torch.Tensor)

        assert data.energies is not None
        assert data.graph_charges is not None

    def test_present_fields_have_canonical_shapes(self, device):
        """When ASE data is present, from_atoms normalizes to canonical shapes."""
        atoms = Atoms(numbers=[8, 1, 1], positions=[[0, 0, 0], [1, 0, 0], [0, 1, 0]])
        atoms.info["energy"] = -10.5
        atoms.arrays["forces"] = np.zeros((3, 3))
        atoms.info["stress"] = np.zeros(6)
        atoms.info["virials"] = np.eye(3)
        atoms.info["dipole"] = np.array([0.1, 0.2, 0.3])
        atoms.arrays["charges"] = np.zeros(3)

        data = AtomicData.from_atoms(atoms, device=device)

        assert data.energies.shape == (1, 1)
        assert data.forces.shape == (3, 3)
        assert data.stresses.shape == (1, 3, 3)
        assert data.stresses.device.type == device
        assert data.virials.shape == (1, 3, 3)
        assert data.virials.device.type == device
        assert data.dipoles.shape == (1, 3)
        assert data.node_charges.shape == (3, 1)


# -----------------------------------------------------------------------------
# dtype cast warning
# -----------------------------------------------------------------------------
class TestDtypeCastWarning:
    """Tests for check_fp_dtype_consistency warning."""

    def test_warns_on_dtype_mismatch(self):
        """Mismatched dtypes emit exactly one UserWarning listing casted fields."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            AtomicData(
                positions=torch.randn(2, 3, dtype=torch.float32),
                atomic_numbers=torch.ones(2, dtype=torch.long),
                forces=torch.randn(2, 3, dtype=torch.float64),
            )
        user_warnings = [w for w in caught if issubclass(w.category, UserWarning)]
        assert len(user_warnings) == 1
        msg = str(user_warnings[0].message)
        assert "forces" in msg
        assert "torch.float32" in msg

    def test_warning_points_to_atomic_data_callsite(self, tmp_path: Path):
        """Warning location resolves to the caller's AtomicData construction site."""
        script = tmp_path / "dtype_warning_probe.py"
        script_text = textwrap.dedent(
            """
            import warnings

            import torch

            from nvalchemi.data.atomic_data import AtomicData

            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                AtomicData(
                    positions=torch.randn(2, 3, dtype=torch.float32),
                    atomic_numbers=torch.ones(2, dtype=torch.long),
                    forces=torch.randn(2, 3, dtype=torch.float64),
                )

            warning = next(w for w in caught if issubclass(w.category, UserWarning))
            warning_filename = warning.filename
            warning_lineno = warning.lineno
            """
        ).lstrip()
        script.write_text(script_text)

        result = runpy.run_path(str(script))
        expected_lineno = next(
            idx
            for idx, line in enumerate(script_text.splitlines(), start=1)
            if "AtomicData(" in line
        )

        assert Path(result["warning_filename"]).resolve() == script.resolve()
        assert result["warning_lineno"] == expected_lineno

    def test_no_warning_when_dtypes_match(self):
        """No warning when all FP tensors share the same dtype."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            AtomicData(
                positions=torch.randn(2, 3, dtype=torch.float32),
                atomic_numbers=torch.ones(2, dtype=torch.long),
                forces=torch.randn(2, 3, dtype=torch.float32),
            )
        user_warnings = [w for w in caught if issubclass(w.category, UserWarning)]
        assert len(user_warnings) == 0


# -----------------------------------------------------------------------------
# from_atoms: cell and pbc handling
# -----------------------------------------------------------------------------
class TestFromAtomsCellPbc:
    """Tests for cell/pbc handling in AtomicData.from_atoms()."""

    def test_non_periodic_sets_cell_pbc_none(self):
        """Fully non-periodic molecule should have cell=None, pbc=None."""
        from ase.build import molecule

        atoms = molecule("H2O")
        data = AtomicData.from_atoms(atoms)
        assert data.cell is None
        assert data.pbc is None

    def test_partial_periodic_zero_cell_raises(self):
        """Partially periodic system with zero cell vectors should raise."""
        atoms = Atoms(
            "Cu4",
            positions=[[0, 0, 0], [1.28, 1.28, 0], [0, 1.28, 1.28], [1.28, 0, 1.28]],
            pbc=[True, True, False],
        )
        with pytest.raises(ValueError, match="undefined.*zero.*lattice vectors"):
            AtomicData.from_atoms(atoms)

    def test_partial_periodic_with_vacuum_works(self):
        """Partially periodic system with proper cell should work."""
        from ase.build import fcc111

        atoms = fcc111("Cu", size=(2, 2, 3), vacuum=10.0)
        data = AtomicData.from_atoms(atoms)
        assert data.cell is not None
        assert data.pbc is not None
        assert data.cell.shape == (1, 3, 3)
        assert data.pbc.shape == (1, 3)

    def test_fully_periodic_works(self):
        """Fully periodic bulk crystal should work."""
        from ase.build import bulk

        atoms = bulk("Cu", "fcc", a=3.6)
        data = AtomicData.from_atoms(atoms)
        assert data.cell is not None
        assert data.pbc is not None
        assert data.cell.shape == (1, 3, 3)
        assert data.pbc.shape == (1, 3)


# -----------------------------------------------------------------------------
# from_structure: pymatgen Structure/Molecule conversion
# -----------------------------------------------------------------------------
class TestFromStructure:
    """Tests for AtomicData.from_structure()."""

    @pytest.fixture(autouse=True)
    def _require_pymatgen(self):
        pytest.importorskip("pymatgen")

    _cu_fcc_coords = [[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]]

    def test_bulk_periodic(self):
        """Fully periodic bulk crystal populates cell, pbc, and core fields."""
        from pymatgen.core import Lattice, Structure

        struct = Structure(Lattice.cubic(3.6), 4 * ["Cu"], self._cu_fcc_coords)
        data = AtomicData.from_structure(struct)
        assert data.cell is not None
        assert data.pbc is not None
        assert data.cell.shape == (1, 3, 3)
        assert data.pbc.shape == (1, 3)
        assert data.atomic_numbers.tolist() == [29, 29, 29, 29]
        assert data.positions.shape == (4, 3)
        assert data.atomic_masses is not None

    def test_molecule_no_lattice(self):
        """Pymatgen Molecule should have cell=None, pbc=None."""
        from pymatgen.core import Molecule

        mol = Molecule(["O", "H", "H"], [[0, 0, 0], [0.96, 0, 0], [0, 0.96, 0]])
        data = AtomicData.from_structure(mol)
        assert data.cell is None
        assert data.pbc is None
        assert data.atomic_numbers.tolist() == [8, 1, 1]
        assert data.positions.shape == (3, 3)

    def test_mixed_pbc_with_cell(self):
        """Partially periodic structure with valid cell should work."""
        from pymatgen.core import Lattice, Structure

        lat = Lattice.from_parameters(5, 5, 20, 90, 90, 90, pbc=(True, True, False))
        struct = Structure(lat, 4 * ["Cu"], self._cu_fcc_coords)
        data = AtomicData.from_structure(struct)
        assert data.cell is not None
        assert data.pbc is not None
        assert data.cell.shape == (1, 3, 3)
        assert data.pbc.shape == (1, 3)

    def test_optional_fields_absent(self):
        """Bare structure should have all optional label fields as None."""
        from pymatgen.core import Lattice, Structure

        struct = Structure(Lattice.cubic(3.6), 4 * ["Cu"], self._cu_fcc_coords)
        data = AtomicData.from_structure(struct)
        assert data.energies is None
        assert data.forces is None
        assert data.stresses is None
        assert data.virials is None
        assert data.dipoles is None
        assert data.node_charges is None
        assert data.graph_charges is None

    def test_optional_fields_present(self):
        """Structure with properties/site_properties populates label fields."""
        from pymatgen.core import Lattice, Structure

        struct = Structure(
            Lattice.cubic(3.6),
            4 * ["Cu"],
            self._cu_fcc_coords,
            site_properties={
                "forces": [[0.1, 0.2, 0.3]] * 4,
                "charges": [0.5, -0.5, 0.5, -0.5],
            },
            properties={
                "energy": -3.5,
                "stress": np.eye(3) * -0.1,
                "virials": np.eye(3) * 0.2,
                "dipole": [0.1, 0.2, 0.3],
            },
        )
        data = AtomicData.from_structure(struct)
        assert data.energies is not None
        assert data.energies.shape == (1, 1)
        assert data.energies.item() == pytest.approx(-3.5)
        assert data.forces is not None
        assert data.forces.shape == (4, 3)
        assert torch.allclose(
            data.forces, torch.tensor([[0.1, 0.2, 0.3]] * 4, dtype=torch.float32)
        )
        assert data.stresses is not None
        assert data.stresses.shape == (1, 3, 3)
        assert torch.allclose(
            data.stresses,
            torch.eye(3, dtype=torch.float32).unsqueeze(0) * -0.1,
        )
        assert data.virials is not None
        assert data.virials.shape == (1, 3, 3)
        assert torch.allclose(
            data.virials,
            torch.eye(3, dtype=torch.float32).unsqueeze(0) * 0.2,
        )
        assert data.dipoles is not None
        assert data.dipoles.shape == (1, 3)
        assert torch.allclose(
            data.dipoles, torch.tensor([[0.1, 0.2, 0.3]], dtype=torch.float32)
        )
        assert data.node_charges is not None
        assert data.node_charges.shape == (4, 1)
        assert torch.allclose(
            data.node_charges,
            torch.tensor([[0.5], [-0.5], [0.5], [-0.5]], dtype=torch.float32),
        )

    def test_charge_explicit(self):
        """Explicitly set charge should populate graph_charges."""
        from pymatgen.core import Lattice, Structure

        struct = Structure(
            Lattice.cubic(3.6), 4 * ["Cu"], self._cu_fcc_coords, charge=2
        )
        data = AtomicData.from_structure(struct)
        assert data.graph_charges is not None
        assert data.graph_charges.item() == 2.0

    def test_charge_not_set(self):
        """No explicit charge should leave graph_charges as None."""
        from pymatgen.core import Lattice, Structure

        struct = Structure(Lattice.cubic(3.6), 4 * ["Cu"], self._cu_fcc_coords)
        data = AtomicData.from_structure(struct)
        assert data.graph_charges is None

    def test_non_integer_charge_raises(self):
        """Non-integer charge should raise ValueError."""
        from pymatgen.core import Lattice, Structure

        struct = Structure(
            Lattice.cubic(3.6), 4 * ["Cu"], self._cu_fcc_coords, charge=1.5
        )
        with pytest.raises(ValueError, match="must be an integer"):
            AtomicData.from_structure(struct)

    def test_extra_properties_in_info(self):
        """Extra properties beyond consumed keys should be in data.info."""
        from pymatgen.core import Lattice, Structure

        struct = Structure(
            Lattice.cubic(3.6),
            4 * ["Cu"],
            self._cu_fcc_coords,
            properties={"energy": -3.5, "my_custom": 42.0},
        )
        data = AtomicData.from_structure(struct)
        assert "my_custom" in data.info
        assert data.info["my_custom"].item() == pytest.approx(42.0)
        assert "energy" not in data.info

    def test_equivalence_with_from_atoms(self):
        """from_structure and from_atoms should produce equivalent AtomicData."""
        from pymatgen.core import Lattice, Structure
        from pymatgen.io.ase import AseAtomsAdaptor

        struct = Structure(Lattice.cubic(3.6), 4 * ["Cu"], self._cu_fcc_coords)
        atoms = AseAtomsAdaptor.get_atoms(struct)

        data_struct = AtomicData.from_structure(struct)
        data_atoms = AtomicData.from_atoms(atoms)

        assert data_struct == data_atoms
        assert torch.allclose(
            data_struct.atomic_masses, data_atoms.atomic_masses, atol=1e-2
        )
