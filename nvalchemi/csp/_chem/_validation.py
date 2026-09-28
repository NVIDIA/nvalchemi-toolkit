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
"""Input checks shared by private RDKit CSP adapters."""

from __future__ import annotations

from collections.abc import Sequence

from rdkit import Chem
from rdkit.Chem.rdchem import Mol


def connected_molecules(molecules: Sequence[Mol]) -> tuple[Mol, ...]:
    """Return a validated nonempty sequence of single-component molecules."""
    if isinstance(molecules, (str, bytes)) or not isinstance(molecules, Sequence):
        raise TypeError("molecules must be a sequence of RDKit Mol objects")
    result = tuple(molecules)
    if not result:
        raise ValueError("molecules must contain at least one molecule")
    for index, molecule in enumerate(result):
        if not isinstance(molecule, Chem.Mol):
            raise TypeError(f"molecules[{index}] must be an rdkit.Chem.Mol")
        if molecule.GetNumAtoms() == 0 or len(Chem.GetMolFrags(molecule)) != 1:
            raise ValueError(f"molecules[{index}] must be nonempty and connected")
    return result
