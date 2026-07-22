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

"""Staged PyTorch bindings for electrostatic interactions.

Exposes the Ewald and PME reciprocal-space computations split into stages, so
distributed callers can insert a cross-rank reduction between the per-rank
partial (structure factors for Ewald, total charge / charge mesh for PME) and
the downstream per-atom energy. The underlying kernels are imported directly
from ``nvalchemiops``.

The upstream reciprocal-space entry points are monolithic and hide this seam;
splitting the stages here lets us all-reduce the partial between them
(``per_system_reduce`` for Ewald structure factors, ``distributed_all_reduce``
for the PME charge mesh).

Modules
-------
ewald
    Ewald reciprocal-space split: per-rank partial structure factors +
    per-atom energy on globally-reduced ``S(k)``.
pme
    PME reciprocal-space split: per-rank partial total charge + full PME
    pipeline taking the globally-reduced total charge as input.
"""

from __future__ import annotations

from nvalchemi.models._ops.electrostatics.ewald import (
    ewald_compute_partial_structure_factors,
    ewald_reciprocal_space_from_structure_factors,
)
from nvalchemi.models._ops.electrostatics.pme import (
    particle_mesh_ewald_from_total_charge,
    pme_compute_partial_total_charge,
    pme_energy_corrections_from_total_charge,
    pme_energy_corrections_with_charge_grad_from_total_charge,
    pme_reciprocal_space_from_total_charge,
)

__all__ = [
    "ewald_compute_partial_structure_factors",
    "ewald_reciprocal_space_from_structure_factors",
    "particle_mesh_ewald_from_total_charge",
    "pme_compute_partial_total_charge",
    "pme_energy_corrections_from_total_charge",
    "pme_energy_corrections_with_charge_grad_from_total_charge",
    "pme_reciprocal_space_from_total_charge",
]
