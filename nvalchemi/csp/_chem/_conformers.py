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
"""RDKit ETKDGv3 conformer generation used by the optional public adapter."""

from __future__ import annotations

from rdkit import Chem
from rdkit.Chem import rdDistGeom
from rdkit.Chem.rdchem import Mol

from nvalchemi.csp._chem._validation import connected_molecules
from nvalchemi.csp.chem import RDKitConformerConfig


def generate_conformers_from_smiles(smiles: str, config: RDKitConformerConfig) -> Mol:
    """Build a new explicit-hydrogen molecule and embed ETKDGv3 conformers."""
    if not isinstance(smiles, str):
        raise TypeError("smiles must be a string")
    if not smiles.strip():
        raise ValueError("smiles must be nonempty")

    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(f"Invalid SMILES: {smiles!r}")
    connected_molecules([molecule])
    molecule = Chem.AddHs(molecule)

    parameters = rdDistGeom.ETKDGv3()
    parameters.randomSeed = -1 if config.random_seed is None else config.random_seed
    parameters.maxIterations = config.max_embed_attempts
    parameters.numThreads = config.num_threads
    parameters.pruneRmsThresh = config.prune_rms_thresh
    parameters.optimizerForceTol = config.embed_force_tol
    rdDistGeom.EmbedMultipleConfs(
        molecule,
        numConfs=config.num_conformers,
        params=parameters,
    )
    if molecule.GetNumConformers() == 0:
        raise RuntimeError("RDKit retained no conformers for the requested SMILES")
    return molecule
