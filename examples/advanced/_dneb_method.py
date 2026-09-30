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

"""Custom doubly nudged NEB force equation for the batched NEB example."""

from __future__ import annotations

from typing import Any

import warp as wp


@wp.func
def dneb_effective_force(
    physical_force: Any,
    tangent: Any,
    d_plus: Any,
    d_minus: Any,
    force_dot_tangent: Any,
    dplus_dot_tangent: Any,
    dminus_dot_tangent: Any,
    norm_d_plus: Any,
    norm_d_minus: Any,
    dplus_dot_dminus: Any,
    force_dot_dplus: Any,
    force_dot_dminus: Any,
    force_squared_norm: Any,
    k_plus: Any,
    k_minus: Any,
    energy_prev: Any,
    energy_curr: Any,
    energy_next: Any,
    path_energy_ref: Any,
    path_energy_max: Any,
):
    """Add the doubly nudged perpendicular spring correction to regular NEB.

    This is equation (13) of Trygubenko and Wales (2004), expressed as forces.
    It retains the improved-tangent parallel spring term used by this example.
    The correction is zero when the perpendicular physical force vanishes,
    because its direction is then undefined.

    Parameters
    ----------
    physical_force, tangent, d_plus, d_minus : vector
        Per-atom physical force, normalized tangent, and adjacent link vectors.
    force_dot_tangent, dplus_dot_tangent, dminus_dot_tangent : scalar
        Image-wide projections onto the normalized tangent.
    norm_d_plus, norm_d_minus, dplus_dot_dminus : scalar
        Image-wide link norms and link dot product.
    force_dot_dplus, force_dot_dminus, force_squared_norm : scalar
        Image-wide physical-force products and squared norm.
    k_plus, k_minus : scalar
        Forward and backward spring constants.
    energy_prev, energy_curr, energy_next : scalar
        Energies of the adjacent and current images; unused by this equation.
    path_energy_ref, path_energy_max : scalar
        Path energy statistics; unused by this equation.

    Returns
    -------
    vector
        Doubly nudged effective force on the current atom.
    """
    spring_parallel = k_plus * norm_d_plus - k_minus * norm_d_minus
    physical_perp = physical_force - force_dot_tangent * tangent
    spring_force = k_plus * d_plus - k_minus * d_minus
    spring_dot_tangent = k_plus * dplus_dot_tangent - k_minus * dminus_dot_tangent
    spring_perp = spring_force - spring_dot_tangent * tangent
    force_perp_squared = force_squared_norm - force_dot_tangent * force_dot_tangent

    correction = physical_force * type(force_squared_norm)(0.0)
    if force_perp_squared > type(force_squared_norm)(1.0e-20):
        spring_dot_force_perp = (
            k_plus * force_dot_dplus
            - k_minus * force_dot_dminus
            - spring_dot_tangent * force_dot_tangent
        )
        correction = (
            spring_perp - (spring_dot_force_perp / force_perp_squared) * physical_perp
        )

    return physical_perp + spring_parallel * tangent + correction
