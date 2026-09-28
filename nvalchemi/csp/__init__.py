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
"""Prepare molecular crystal searches and inspect their results.

Build formula-unit inputs from molecular conformers, estimate starting cell
volumes, and choose space groups. The packer selects conformers, places the
molecules in periodic cells, then translates and rotates whole molecules and
changes cell geometry to resolve overlaps to a chosen tolerance. Save the
resulting asymmetric-unit representations, expand selected structures to
full-cell Toolkit batches for physical optimization, and screen atomistic
structures for possible duplicates with approximate comparison.
"""

from nvalchemi.csp.data import MolecularPackingInput, RigidMoleculeASUBatch
from nvalchemi.csp.storage import CSPZarrReader, CSPZarrWriter
from nvalchemi.csp.symmetry import (
    CrystalSystem,
    csd_space_group_probabilities,
    get_crystal_system,
    get_space_group_candidates,
    get_space_group_operation_count,
    get_space_group_operations,
    is_sohncke_space_group,
    sample_space_groups,
)
from nvalchemi.csp.volume import (
    estimate_formula_unit_volume,
    get_default_atomic_volumes,
)

__all__ = [
    "CrystalSystem",
    "CSPZarrReader",
    "CSPZarrWriter",
    "MolecularPackingInput",
    "RigidMoleculeASUBatch",
    "csd_space_group_probabilities",
    "estimate_formula_unit_volume",
    "get_crystal_system",
    "get_default_atomic_volumes",
    "get_space_group_candidates",
    "get_space_group_operation_count",
    "get_space_group_operations",
    "is_sohncke_space_group",
    "sample_space_groups",
]
