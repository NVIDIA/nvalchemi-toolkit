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
"""Assembly of canonical tensor-backed molecular packing inputs."""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from typing import Any

import torch
from rdkit.Chem.rdchem import Mol
from torch import Tensor

from nvalchemi.csp._chem._contacts import contact_distances as _default_contacts
from nvalchemi.csp._chem._validation import connected_molecules
from nvalchemi.csp.data import MolecularPackingInput
from nvalchemi.csp.volume import estimate_formula_unit_volume


def _normalize_contact_distances(values: Tensor, num_atoms: int) -> Tensor:
    """Copy contact distances to symmetric contiguous CPU float32 form in angstroms."""
    if not isinstance(values, Tensor):
        raise TypeError("contact_distances must be a torch.Tensor")
    if values.shape != (num_atoms, num_atoms):
        raise ValueError(
            f"contact_distances must have shape ({num_atoms}, {num_atoms}); "
            f"got {tuple(values.shape)}"
        )
    try:
        source = values.detach()
        if source.layout != torch.strided:
            source = source.to_dense()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            matrix = source.to(device="cpu", dtype=torch.float32).contiguous().clone()
    except (RuntimeError, TypeError, ValueError) as error:
        raise TypeError("contact_distances must be convertible to float32") from error
    if not torch.isfinite(matrix).all():
        raise ValueError("contact_distances must contain finite values")
    if (matrix <= 0).any():
        raise ValueError("contact_distances must contain positive values")
    if not torch.allclose(matrix, matrix.T, rtol=1.0e-3, atol=1.0e-3):
        raise ValueError("contact_distances must be symmetric within tolerance")
    averaged = 0.5 * matrix + 0.5 * matrix.T
    lower = torch.tril(averaged)
    return (lower + torch.tril(averaged, diagonal=-1).T).contiguous()


def build_molecular_packing_input(
    molecules: Sequence[Mol],
    *,
    contact_distances: Tensor | None = None,
    formula_unit_volume: float | None = None,
    component_index: Tensor | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> MolecularPackingInput:
    """Build formula-unit tensors from connected RDKit molecules and conformers."""
    components = connected_molecules(molecules)
    positions: list[torch.Tensor] = []
    conformer_ptr = [0]
    molecule_conformer_ptr = [0]
    molecule_atom_ptr = [0]
    atomic_numbers: list[int] = []

    for molecule_index, molecule in enumerate(components):
        atom_count = molecule.GetNumAtoms()
        atomic_numbers.extend(int(atom.GetAtomicNum()) for atom in molecule.GetAtoms())
        molecule_atom_ptr.append(molecule_atom_ptr[-1] + atom_count)
        conformers = tuple(molecule.GetConformers())
        if not conformers:
            raise ValueError(f"molecules[{molecule_index}] must have a 3D conformer")
        for conformer in conformers:
            if not conformer.Is3D():
                raise ValueError(
                    f"molecules[{molecule_index}] contains a non-3D conformer"
                )
            xyz = [
                conformer.GetAtomPosition(atom_index)
                for atom_index in range(atom_count)
            ]
            values = torch.tensor(
                [(point.x, point.y, point.z) for point in xyz],
                dtype=torch.float32,
            )
            if not torch.isfinite(values).all():
                raise ValueError(
                    f"molecules[{molecule_index}] contains non-finite conformer coordinates"
                )
            positions.append(values)
            conformer_ptr.append(conformer_ptr[-1] + atom_count)
        molecule_conformer_ptr.append(molecule_conformer_ptr[-1] + len(conformers))

    atom_numbers = torch.tensor(atomic_numbers, dtype=torch.int64)
    contacts = (
        _default_contacts(components)
        if contact_distances is None
        else _normalize_contact_distances(contact_distances, len(atomic_numbers))
    )
    volume = (
        estimate_formula_unit_volume(atom_numbers)
        if formula_unit_volume is None
        else formula_unit_volume
    )
    components_tensor = (
        torch.arange(len(components), dtype=torch.int32)
        if component_index is None
        else component_index
    )

    return MolecularPackingInput(
        conformer_positions=torch.cat(positions, dim=0),
        conformer_ptr=torch.tensor(conformer_ptr, dtype=torch.int32),
        molecule_conformer_ptr=torch.tensor(molecule_conformer_ptr, dtype=torch.int32),
        molecule_atom_ptr=torch.tensor(molecule_atom_ptr, dtype=torch.int32),
        atomic_numbers=atom_numbers,
        contact_distances=contacts,
        component_index=components_tensor,
        formula_unit_volume=volume,
        metadata=metadata,
    )
