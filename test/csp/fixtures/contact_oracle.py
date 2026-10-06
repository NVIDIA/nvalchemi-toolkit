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

"""Small independent periodic contact enumerator for CSP public tests."""

from __future__ import annotations

from itertools import product
from math import ceil

import torch


def enumerate_periodic_contacts(
    positions: torch.Tensor,
    cell: torch.Tensor,
    atom_molecules: torch.Tensor,
    contact_distances: torch.Tensor,
    *,
    asu_atom_ids: torch.Tensor | None = None,
) -> list[tuple[int, int, torch.Tensor, float]]:
    """Enumerate unique overlapping atom pairs and lattice images.

    Coordinates are unwrapped Cartesian positions of the rigid molecules; the
    cell vectors are rows. ``atom_molecules`` identifies physical molecules
    for same-image intramolecular exclusion, while ``asu_atom_ids`` optionally
    supplies separate indices into the contact-distance matrix. The search
    uses the smallest singular value:
    if ``|r_j-r_i+s@cell| < cutoff``, then
    ``sigma_min*|s| <= |s@cell| < cutoff+|r_j-r_i|``. This gives a complete
    integer cube bound for every atom pair, including non-reduced cells.
    Self-image pairs retain only the lexicographically positive shift so each
    undirected contact is counted once.
    """
    positions64 = positions.to(dtype=torch.float64, device="cpu")
    cell64 = cell.to(dtype=torch.float64, device="cpu")
    molecules = atom_molecules.to(dtype=torch.int64, device="cpu")
    asu_atoms = (
        molecules
        if asu_atom_ids is None
        else asu_atom_ids.to(dtype=torch.int64, device="cpu")
    )
    cutoffs = contact_distances.to(dtype=torch.float64, device="cpu")
    sigma_min = float(torch.linalg.svdvals(cell64).min())
    assert sigma_min > 0.0
    contacts: list[tuple[int, int, torch.Tensor, float]] = []
    for i in range(len(positions64)):
        for j in range(i, len(positions64)):
            displacement = positions64[j] - positions64[i]
            cutoff = float(cutoffs[asu_atoms[i], asu_atoms[j]])
            shift_bound = ceil(
                (cutoff + float(torch.linalg.vector_norm(displacement))) / sigma_min
            )
            for shift_values in product(range(-shift_bound, shift_bound + 1), repeat=3):
                if molecules[i] == molecules[j] and shift_values == (0, 0, 0):
                    continue
                if i == j and not (
                    shift_values[0] > 0
                    or (shift_values[0] == 0 and shift_values[1] > 0)
                    or (
                        shift_values[0] == 0
                        and shift_values[1] == 0
                        and shift_values[2] > 0
                    )
                ):
                    continue
                shift = torch.tensor(shift_values, dtype=torch.float64)
                vector = displacement + shift @ cell64
                distance = float(torch.linalg.vector_norm(vector))
                overlap = cutoff - distance
                if overlap > 0.0:
                    contacts.append((i, j, vector, overlap))
    return contacts


def contact_observables(
    positions: torch.Tensor,
    cell: torch.Tensor,
    atom_molecules: torch.Tensor,
    contact_distances: torch.Tensor,
    *,
    expanded_atom_count: int | None = None,
    asu_atom_ids: torch.Tensor | None = None,
) -> tuple[int, torch.Tensor, torch.Tensor, float, float]:
    """Return contact count, molecule forces, virial, total and max overlap."""
    molecules = atom_molecules.to(dtype=torch.int64, device="cpu")
    contacts = enumerate_periodic_contacts(
        positions,
        cell,
        molecules,
        contact_distances,
        asu_atom_ids=asu_atom_ids,
    )
    forces = torch.zeros((int(molecules.max()) + 1, 3), dtype=torch.float64)
    virial = torch.zeros((3, 3), dtype=torch.float64)
    total = 0.0
    maximum = 0.0
    for i, j, vector, overlap in contacts:
        distance = float(torch.linalg.vector_norm(vector))
        pair_force = vector * (overlap / distance)
        forces[molecules[i]] -= pair_force
        forces[molecules[j]] += pair_force
        virial += torch.outer(vector, pair_force)
        total += overlap
        maximum = max(maximum, overlap)
    if expanded_atom_count is not None:
        virial /= expanded_atom_count
    return len(contacts), forces, virial, total, maximum


def contact_torques(
    positions: torch.Tensor,
    cell: torch.Tensor,
    atom_molecules: torch.Tensor,
    contact_distances: torch.Tensor,
    relative_arms: torch.Tensor,
    *,
    asu_atom_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    """Accumulate molecule torques from enumerated atom forces and rigid arms."""
    molecules = atom_molecules.to(dtype=torch.int64, device="cpu")
    arms = relative_arms.to(dtype=torch.float64, device="cpu")
    torques = torch.zeros((int(molecules.max()) + 1, 3), dtype=torch.float64)
    contacts = enumerate_periodic_contacts(
        positions,
        cell,
        molecules,
        contact_distances,
        asu_atom_ids=asu_atom_ids,
    )
    for i, j, vector, overlap in contacts:
        distance = float(torch.linalg.vector_norm(vector))
        pair_force = vector * (overlap / distance)
        torques[molecules[i]] += torch.linalg.cross(arms[i], -pair_force)
        torques[molecules[j]] += torch.linalg.cross(arms[j], pair_force)
    return torques
