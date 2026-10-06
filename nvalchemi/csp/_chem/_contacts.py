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
"""Private topology-aware atom-pair contact-distance rules."""

from __future__ import annotations

import math
import warnings
from collections.abc import Sequence

import torch
from rdkit import Chem
from rdkit.Chem.rdchem import Atom, BondType, HybridizationType, Mol

from nvalchemi.csp._chem._validation import connected_molecules

_RADII = {
    1: 1.20,
    5: 1.91,
    6: 1.77,
    7: 1.66,
    8: 1.50,
    9: 1.46,
    14: 2.19,
    15: 1.90,
    16: 1.89,
    17: 1.82,
    34: 1.82,
    35: 1.86,
    53: 2.04,
}
_HALOGENS = {9, 17, 35, 53}
_EW_SUBSTITUENTS = {5, 7, 8, 9, 15, 16, 17, 35, 53}


def _has_double_bonded_neighbor(
    atom: Atom,
    elements: set[int],
    *,
    excluded_index: int | None = None,
) -> bool:
    """Whether an atom has a double-bonded neighbor in ``elements``."""
    return any(
        bond.GetBondType() == BondType.DOUBLE
        and bond.GetOtherAtom(atom).GetAtomicNum() in elements
        and bond.GetOtherAtom(atom).GetIdx() != excluded_index
        for bond in atom.GetBonds()
    )


def _cyano_carbon(atom: Atom) -> bool:
    """Whether a carbon atom is triple-bonded to nitrogen."""
    return atom.GetAtomicNum() == 6 and any(
        bond.GetBondType() == BondType.TRIPLE
        and bond.GetOtherAtom(atom).GetAtomicNum() == 7
        for bond in atom.GetBonds()
    )


def _amide_like_nitrogen(atom: Atom) -> bool:
    """Whether a nitrogen is attached to a carbonyl-like center."""
    if atom.GetAtomicNum() != 7:
        return False
    return any(
        neighbor.GetAtomicNum() == 6
        and _has_double_bonded_neighbor(
            neighbor, {7, 8, 16}, excluded_index=atom.GetIdx()
        )
        for neighbor in atom.GetNeighbors()
    )


def _nitro_nitrogen(atom: Atom) -> bool:
    """Whether a positive nitrogen has at least two oxygen neighbors."""
    return (
        atom.GetAtomicNum() == 7
        and atom.GetFormalCharge() > 0
        and sum(neighbor.GetAtomicNum() == 8 for neighbor in atom.GetNeighbors()) >= 2
    )


def _strong_oxygen_acceptor(atom: Atom) -> bool:
    """Whether oxygen is double-bonded to one of the supported elements."""
    return atom.GetAtomicNum() == 8 and _has_double_bonded_neighbor(
        atom, {6, 7, 15, 16, 34}
    )


def _acceptor(atom: Atom) -> float:
    """Return the rule-based acceptor contribution for one RDKit atom."""
    number = atom.GetAtomicNum()
    charge = atom.GetFormalCharge()
    if number in _HALOGENS and charge < 0:
        return 0.30
    if number == 8:
        if charge < 0:
            return 0.30
        if _strong_oxygen_acceptor(atom):
            return 0.25
        if charge == 0 and atom.GetTotalValence() == 2:
            return 0.15
        return 0.0
    if (
        number == 7
        and charge <= 0
        and not (atom.GetIsAromatic() and atom.GetTotalNumHs(includeNeighbors=True) > 0)
        and not _amide_like_nitrogen(atom)
        and not _nitro_nitrogen(atom)
    ):
        if atom.GetIsAromatic() or atom.GetHybridization() in {
            HybridizationType.SP,
            HybridizationType.SP2,
        }:
            return 0.20
        if charge == 0 and atom.GetHybridization() == HybridizationType.SP3:
            return 0.10
    if (
        number in {16, 34}
        and charge == 0
        and atom.GetTotalValence() == 2
        and not any(neighbor.GetAtomicNum() == 8 for neighbor in atom.GetNeighbors())
    ):
        return 0.10
    return 0.0


def _activated_c_hydrogen(carbon: Atom, hydrogen: Atom) -> bool:
    """Whether a carbon-bound hydrogen meets the activated C-H rules."""
    if (
        carbon.GetIsAromatic()
        or carbon.GetHybridization() in {HybridizationType.SP, HybridizationType.SP2}
        or carbon.GetFormalCharge() > 0
    ):
        return True
    return any(
        atom.GetIdx() != hydrogen.GetIdx()
        and (atom.GetFormalCharge() > 0 or atom.GetAtomicNum() in _EW_SUBSTITUENTS)
        for atom in carbon.GetNeighbors()
    )


def _activated_halogen(atom: Atom) -> bool:
    """Whether a singly bonded halogen meets the activated-halogen rules."""
    if atom.GetDegree() != 1:
        return False
    parent = atom.GetNeighbors()[0]
    if (
        parent.GetIsAromatic()
        or parent.GetHybridization() in {HybridizationType.SP, HybridizationType.SP2}
        or parent.GetFormalCharge() > 0
    ):
        return True
    return any(
        neighbor.GetIdx() != atom.GetIdx()
        and (
            neighbor.GetFormalCharge() > 0
            or neighbor.GetAtomicNum() in _EW_SUBSTITUENTS
        )
        for neighbor in parent.GetNeighbors()
    )


def _donor(atom: Atom) -> float:
    """Return the rule-based donor contribution for one RDKit atom."""
    number = atom.GetAtomicNum()
    charge = atom.GetFormalCharge()
    if number == 1 and atom.GetDegree() == 1:
        parent = atom.GetNeighbors()[0]
        parent_number = parent.GetAtomicNum()
        if parent_number == 8:
            return 0.55
        if parent_number == 7:
            return 0.35
        if parent_number == 16:
            return 0.15
        if parent_number == 6 and _activated_c_hydrogen(parent, atom):
            return 0.05
        return 0.0
    if number in {17, 35, 53} and charge == 0 and atom.GetDegree() == 1:
        activated = _activated_halogen(atom)
        return {
            17: 0.10 if activated else 0.0,
            35: 0.20 if activated else 0.10,
            53: 0.35 if activated else 0.25,
        }[number]
    if number in {16, 34} and charge >= 0:
        activated = (
            charge > 0
            or atom.GetTotalValence() > 2
            or any(
                neighbor.GetFormalCharge() > 0
                or neighbor.GetAtomicNum() in {8, 9, 17, 35, 53}
                or _cyano_carbon(neighbor)
                for neighbor in atom.GetNeighbors()
            )
        )
        return (
            (0.10 if activated else 0.05)
            if number == 16
            else (0.15 if activated else 0.10)
        )
    if number == 15:
        if charge > 0:
            return 0.15
        if atom.GetTotalValence() > 3 or any(
            neighbor.GetAtomicNum() in {7, 8, 9, 17, 35, 53} or _cyano_carbon(neighbor)
            for neighbor in atom.GetNeighbors()
        ):
            return 0.05
    if number == 14:
        if charge > 0 or atom.GetTotalValence() > 4:
            return 0.15
        if any(
            neighbor.GetAtomicNum() in {7, 8, 9, 17, 35, 53}
            for neighbor in atom.GetNeighbors()
        ):
            return 0.05
    if (
        number == 5
        and charge in {0, 1}
        and atom.GetDegree() == 3
        and atom.GetTotalValence() == 3
    ):
        return 0.45
    return 0.0


def contact_distances(molecules: Sequence[Mol]) -> torch.Tensor:
    """Construct a CPU float32 all-atom contact-cutoff matrix in angstroms."""
    connected = connected_molecules(molecules)
    atoms = [atom for molecule in connected for atom in molecule.GetAtoms()]
    unsupported: set[str] = set()
    periodic_table = Chem.GetPeriodicTable()
    radii: list[float] = []
    for atom in atoms:
        number = atom.GetAtomicNum()
        radius = _RADII.get(number)
        if radius is None:
            radius = float(periodic_table.GetRvdw(number))
            unsupported.add(atom.GetSymbol())
        if not math.isfinite(radius) or radius <= 0.0:
            raise ValueError(f"No positive van der Waals radius for {atom.GetSymbol()}")
        radii.append(radius)
    if unsupported:
        symbols = ", ".join(sorted(unsupported))
        warnings.warn(
            f"Using RDKit van der Waals radii for unsupported elements: {symbols}",
            RuntimeWarning,
            stacklevel=2,
        )

    radius_tensor = torch.tensor(radii, dtype=torch.float32)
    donor = torch.tensor([_donor(atom) for atom in atoms], dtype=torch.float32)
    acceptor = torch.tensor([_acceptor(atom) for atom in atoms], dtype=torch.float32)
    forward = torch.where(
        (donor[:, None] > 0) & (acceptor[None, :] > 0),
        donor[:, None] + acceptor[None, :],
        0.0,
    )
    backward = torch.where(
        (donor[None, :] > 0) & (acceptor[:, None] > 0),
        donor[None, :] + acceptor[:, None],
        0.0,
    )
    reduction = torch.maximum(forward, backward).clamp(max=0.90)
    return radius_tensor[:, None] + radius_tensor[None, :] - reduction
