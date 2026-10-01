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
"""Focused tests for optional RDKit CSP preparation."""

from __future__ import annotations

import pytest
import torch

from nvalchemi import OptionalDependency, OptionalDependencyError
from nvalchemi.csp.chem import (
    RDKitConformerConfig,
    build_contact_distance_matrix,
    build_molecular_packing_input,
    generate_conformers_from_smiles,
)

Chem = pytest.importorskip("rdkit.Chem")


def _mol(smiles: str) -> Chem.Mol:
    molecule = Chem.MolFromSmiles(smiles)
    assert molecule is not None
    return molecule


def _with_3d_conformer(molecule: Chem.Mol) -> Chem.Mol:
    """Attach a small 3D conformer to a connected test molecule."""
    conformer = Chem.Conformer(molecule.GetNumAtoms())
    conformer.Set3D(True)
    for atom_index in range(molecule.GetNumAtoms()):
        conformer.SetAtomPosition(atom_index, (float(atom_index), 0.0, 0.0))
    molecule.AddConformer(conformer, assignId=True)
    return molecule


def test_rdkit_helpers_use_optional_dependency_guard(monkeypatch) -> None:
    monkeypatch.setattr(OptionalDependency.RDKIT, "_available", False)
    monkeypatch.setattr(
        OptionalDependency.RDKIT,
        "_import_error",
        ModuleNotFoundError("No module named 'rdkit'"),
    )
    with pytest.raises(OptionalDependencyError, match=r"nvalchemi-toolkit\[rdkit\]"):
        generate_conformers_from_smiles("CO")
    with pytest.raises(OptionalDependencyError):
        build_contact_distance_matrix([_mol("C")])
    with pytest.raises(OptionalDependencyError):
        build_molecular_packing_input([_mol("C")])


def test_conformer_config_is_frozen_and_validated() -> None:
    config = RDKitConformerConfig()
    assert config.num_conformers == 100
    with pytest.raises(Exception):
        config.num_conformers = 2
    with pytest.raises(ValueError, match="num_conformers"):
        RDKitConformerConfig(num_conformers=0)
    with pytest.raises(ValueError, match="random_seed"):
        RDKitConformerConfig(random_seed=-1)
    with pytest.raises(ValueError, match="prune_rms_thresh"):
        RDKitConformerConfig(prune_rms_thresh=-1.1)
    with pytest.raises(ValueError, match="embed_force_tol"):
        RDKitConformerConfig(embed_force_tol=float("inf"))
    with pytest.raises(ValueError, match="max_embed_attempts"):
        RDKitConformerConfig(max_embed_attempts=0)
    with pytest.raises(ValueError, match="num_threads"):
        RDKitConformerConfig(num_threads=-2)
    with pytest.raises(Exception):
        RDKitConformerConfig(num_conformers=True)


def test_smiles_embedding_adds_hydrogens_and_returns_new_3d_molecule() -> None:
    config = RDKitConformerConfig(num_conformers=2, random_seed=42)
    molecule = generate_conformers_from_smiles("CO", config=config)

    assert molecule.GetNumAtoms() == 6
    assert molecule.GetNumConformers() > 0
    assert all(conformer.Is3D() for conformer in molecule.GetConformers())
    assert all(atom.GetAtomicNum() == 1 for atom in list(molecule.GetAtoms())[2:])


@pytest.mark.parametrize("smiles", ["", "not-a-smiles", "C.O"])
def test_smiles_embedding_rejects_invalid_or_disconnected_input(smiles: str) -> None:
    with pytest.raises((TypeError, ValueError)):
        generate_conformers_from_smiles(smiles)


def test_smiles_embedding_requires_rdkit_config_type() -> None:
    with pytest.raises(TypeError, match="config"):
        generate_conformers_from_smiles("C", config={})


def test_contact_matrix_has_vdw_baseline_and_preserves_atom_order() -> None:
    molecule = _mol("CO")
    contacts = build_contact_distance_matrix([molecule])

    assert contacts.shape == (2, 2)
    assert contacts.dtype == torch.float32
    assert contacts.device.type == "cpu"
    assert contacts.is_contiguous()
    assert torch.equal(contacts, contacts.T)
    assert contacts[0, 0].item() == pytest.approx(3.54)
    assert contacts[1, 1].item() == pytest.approx(3.0)
    assert contacts[0, 1].item() == pytest.approx(3.27)


def test_contact_matrix_applies_symmetric_hydrogen_bond_rule() -> None:
    water = Chem.AddHs(_mol("O"))
    contacts = build_contact_distance_matrix([water])

    # RDKit's O-H atom order is O, H, H. H donor and O acceptor bonuses sum.
    expected = 1.20 + 1.50 - 0.55 - 0.15
    assert contacts[0, 1].item() == pytest.approx(expected)
    assert contacts[1, 0].item() == pytest.approx(expected)
    assert contacts[1, 1].item() == pytest.approx(2.40)


def test_contact_matrix_warns_when_periodic_table_radius_is_used() -> None:
    with pytest.warns(RuntimeWarning, match="Na"):
        contacts = build_contact_distance_matrix([_mol("[Na+]")])
    assert contacts.shape == (1, 1)
    assert torch.isfinite(contacts).all()
    assert bool((contacts > 0).all())


@pytest.mark.parametrize("molecules", [[], [_mol("C"), None], [_mol("C.O")]])
def test_contact_matrix_rejects_invalid_molecules(molecules) -> None:
    with pytest.raises((TypeError, ValueError)):
        build_contact_distance_matrix(molecules)


def test_contact_matrix_can_return_on_requested_device() -> None:
    contacts = build_contact_distance_matrix([_mol("C")], device="cpu")
    assert contacts.device.type == "cpu"


def test_formula_input_assembly_packs_conformers_and_centers_once() -> None:
    first = generate_conformers_from_smiles(
        "CO", config=RDKitConformerConfig(num_conformers=2, random_seed=13)
    )
    second = generate_conformers_from_smiles(
        "N", config=RDKitConformerConfig(num_conformers=1, random_seed=17)
    )
    original_first = [
        torch.tensor(
            [
                tuple(first.GetConformer(index).GetAtomPosition(atom))
                for atom in range(first.GetNumAtoms())
            ],
            dtype=torch.float32,
        )
        for index in range(first.GetNumConformers())
    ]
    custom_contacts = torch.full((first.GetNumAtoms() + second.GetNumAtoms(),) * 2, 2.5)
    custom_contacts.fill_diagonal_(3.0)

    packing_input = build_molecular_packing_input(
        [first, second],
        contact_distances=custom_contacts,
        formula_unit_volume=51.0,
        component_index=torch.tensor([0, 1], dtype=torch.int32),
        metadata={"source": {"batch": "chem-test"}},
    )

    assert packing_input.num_molecules == 2
    assert packing_input.num_conformers == 3
    assert packing_input.molecule_atom_ptr.tolist() == [0, 6, 10]
    assert packing_input.molecule_conformer_ptr.tolist() == [0, 2, 3]
    assert packing_input.conformer_ptr.tolist() == [0, 6, 12, 16]
    assert packing_input.formula_unit_volume == pytest.approx(51.0)
    assert torch.equal(packing_input.contact_distances, custom_contacts)
    for index, original in enumerate(original_first):
        start = packing_input.conformer_ptr[index].item()
        stop = packing_input.conformer_ptr[index + 1].item()
        centered = packing_input.conformer_positions[start:stop]
        torch.testing.assert_close(centered, original - original.mean(dim=0))
        torch.testing.assert_close(
            centered.mean(dim=0), torch.zeros(3), atol=1e-6, rtol=0
        )
    assert packing_input.metadata["source"]["batch"] == "chem-test"
    assert torch.equal(
        custom_contacts, torch.full_like(custom_contacts, 2.5).fill_diagonal_(3.0)
    )


def test_formula_input_uses_default_contacts_volume_and_component_ids() -> None:
    molecule = generate_conformers_from_smiles(
        "CO", config=RDKitConformerConfig(num_conformers=1, random_seed=21)
    )
    packing_input = build_molecular_packing_input([molecule])

    assert packing_input.formula_unit_volume == pytest.approx(46.94)
    assert packing_input.component_index.tolist() == [0]
    assert packing_input.contact_distances.shape == (6, 6)
    assert torch.equal(
        packing_input.contact_distances, packing_input.contact_distances.T
    )


def test_formula_input_extracts_component_charges_with_repeated_stoichiometry() -> None:
    molecules = [
        _with_3d_conformer(_mol("[Na+]")),
        _with_3d_conformer(_mol("[Na+]")),
        _with_3d_conformer(_mol("[O-2]")),
    ]
    packing_input = build_molecular_packing_input(
        molecules,
        component_index=torch.tensor([0, 0, 1], dtype=torch.int32),
        contact_distances=torch.ones((3, 3), dtype=torch.float32),
        formula_unit_volume=20.0,
    )

    assert packing_input.component_charge.tolist() == [1, -2]
    assert packing_input.molecule_charge.tolist() == [1, 1, -2]
    assert packing_input.formula_unit_charge == 0


def test_formula_input_rejects_different_charges_for_one_component_id() -> None:
    molecules = [
        _with_3d_conformer(_mol("[Na+]")),
        _with_3d_conformer(_mol("[Cl-]")),
    ]
    with pytest.raises(ValueError, match="conflicting formal charges"):
        build_molecular_packing_input(
            molecules,
            component_index=torch.tensor([0, 0], dtype=torch.int32),
            contact_distances=torch.ones((2, 2), dtype=torch.float32),
            formula_unit_volume=20.0,
        )


@pytest.mark.parametrize(
    "molecule, message",
    [(_mol("C"), "3D conformer"), (Chem.AddHs(_mol("C")), "3D conformer")],
)
def test_formula_input_requires_3d_conformers(molecule, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        build_molecular_packing_input([molecule])


def test_formula_input_validates_custom_contacts_and_components() -> None:
    molecule = generate_conformers_from_smiles(
        "C", config=RDKitConformerConfig(num_conformers=1, random_seed=8)
    )
    with pytest.raises(ValueError, match="contact_distances must have shape"):
        build_molecular_packing_input(
            [molecule], contact_distances=torch.ones((1, 1), dtype=torch.float32)
        )
    with pytest.raises(ValueError, match="positive"):
        build_molecular_packing_input(
            [molecule], contact_distances=torch.zeros((5, 5), dtype=torch.float32)
        )
    with pytest.raises(ValueError, match="component_index"):
        build_molecular_packing_input(
            [molecule], component_index=torch.tensor([1], dtype=torch.int32)
        )
