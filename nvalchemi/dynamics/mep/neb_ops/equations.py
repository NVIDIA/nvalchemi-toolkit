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

"""Built-in device equations for specialized NEB force kernels."""

from __future__ import annotations

from typing import Any

import warp as wp

__all__ = [
    "climbing_image_effective_force",
    "improved_tangent_weights",
    "neb_effective_force",
    "neb_effective_force_from_gram_stats",
]


# =============================================================================
# Equations for improved tangent NEB
# Reference: Henkelman and Jónsson, “Improved Tangent Estimate in the Nudged
#   Elastic Band Method for Finding Minimum Energy Paths and Saddle Points,”
#   *J. Chem. Phys.* 113, 9978 (2000).
# =============================================================================


@wp.func
def improved_tangent_weights(
    energy_prev: Any,
    energy_curr: Any,
    energy_next: Any,
):
    r"""Return the Henkelman--Jonsson improved-tangent link weights.

    The unnormalized tangent is formed as

    .. math::

        \boldsymbol{\tau} = w_+\mathbf{d}_+ + w_-\mathbf{d}_-,

    where the weights select the higher-energy neighbor on monotonic sections
    of the path and smoothly mix both links at extrema.

    Parameters
    ----------
    energy_prev : scalar
        Energy of the preceding image.
    energy_curr : scalar
        Energy of the current image.
    energy_next : scalar
        Energy of the following image.

    Returns
    -------
    tuple[scalar, scalar]
        The forward and backward link weights ``(weight_plus, weight_minus)``.
    """
    zero = type(energy_curr)(0.0)
    one = type(energy_curr)(1.0)
    delta_plus = energy_next - energy_curr
    delta_minus = energy_curr - energy_prev
    weight_plus = zero
    weight_minus = zero
    if delta_minus > zero and delta_plus > zero:
        weight_plus = one
    elif delta_minus < zero and delta_plus < zero:
        weight_minus = one
    else:
        delta_max = wp.max(wp.abs(delta_plus), wp.abs(delta_minus))
        delta_min = wp.min(wp.abs(delta_plus), wp.abs(delta_minus))
        if energy_next > energy_prev:
            weight_plus = delta_max
            weight_minus = delta_min
        else:
            weight_plus = delta_min
            weight_minus = delta_max
    return weight_plus, weight_minus


@wp.func
def neb_effective_force(
    physical_force: Any,
    tangent: Any,
    force_dot_tangent: Any,
    k_plus: Any,
    k_minus: Any,
    norm_d_plus: Any,
    norm_d_minus: Any,
    energy_prev: Any,
    energy_curr: Any,
    energy_next: Any,
    path_energy_ref: Any,
    path_energy_max: Any,
):
    r"""Return the effective force for regular NEB.

    Computes

    .. math::

        \mathbf{F}^{\mathrm{NEB}}
        = \mathbf{F}
        - (\mathbf{F}\cdot\hat{\boldsymbol{\tau}})
          \hat{\boldsymbol{\tau}}
        + F^{\mathrm{s}}_{\parallel}\hat{\boldsymbol{\tau}}.

    Here, the scalar spring force parallel to the path is

    .. math::

        F^{\mathrm{s}}_{\parallel}
        = k_{+}\lVert\mathbf{d}_{+}\rVert
        - k_{-}\lVert\mathbf{d}_{-}\rVert.

    Parameters
    ----------
    physical_force : vector
        Physical force on one atom in the current image.
    tangent : vector
        Per-atom component of the normalized image tangent.
    force_dot_tangent : scalar
        Image-wide inner product of the physical force and normalized tangent.
    k_plus : scalar
        Spring constant for the link to the following image.
    k_minus : scalar
        Spring constant for the link to the preceding image.
    norm_d_plus : scalar
        Norm of the forward image-displacement vector ``d_plus``.
    norm_d_minus : scalar
        Norm of the backward image-displacement vector ``d_minus``.
    energy_prev : scalar
        Energy of the preceding image.
    energy_curr : scalar
        Energy of the current image.
    energy_next : scalar
        Energy of the following image.
    path_energy_ref : scalar
        Reference energy for the path.
    path_energy_max : scalar
        Maximum interior-image energy for the path.

    Returns
    -------
    vector
        Effective force on the atom.

    Notes
    -----
    The regular constant-spring law does not use the image or path energies;
    they remain available to alternative effective-force hooks using the same
    stored-tangent kernel contract.
    """
    spring_parallel = k_plus * norm_d_plus - k_minus * norm_d_minus
    return physical_force + (spring_parallel - force_dot_tangent) * tangent


@wp.func
def neb_effective_force_from_gram_stats(
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
    """Adapt the NEB effective-force law to the full Gram-statistics contract.

    Parameters
    ----------
    physical_force : vector
        Physical force on one atom in the current image.
    tangent : vector
        Per-atom component of the normalized image tangent.
    d_plus, d_minus : vector
        Per-atom forward and backward image-displacement vectors.
    force_dot_tangent : scalar
        Image-wide inner product of the physical force and normalized tangent.
    dplus_dot_tangent, dminus_dot_tangent : scalar
        Image-wide link projections onto the normalized tangent.
    norm_d_plus, norm_d_minus : scalar
        Image-wide forward and backward link norms.
    dplus_dot_dminus : scalar
        Image-wide inner product of the two link vectors.
    force_dot_dplus, force_dot_dminus : scalar
        Image-wide inner products of each link with the physical force.
    force_squared_norm : scalar
        Image-wide squared norm of the physical force.
    k_plus, k_minus : scalar
        Forward and backward link spring constants.
    energy_prev, energy_curr, energy_next : scalar
        Energies of the preceding, current, and following images.
    path_energy_ref, path_energy_max : scalar
        Reference and maximum interior-image energies for the path.

    Returns
    -------
    vector
        Effective force on the atom.

    Notes
    -----
    Regular NEB uses only the stored-tangent subset of this contract. The remaining
    arguments are available to projection-heavy force laws such as DNEB.
    """
    return neb_effective_force(
        physical_force,
        tangent,
        force_dot_tangent,
        k_plus,
        k_minus,
        norm_d_plus,
        norm_d_minus,
        energy_prev,
        energy_curr,
        energy_next,
        path_energy_ref,
        path_energy_max,
    )


# =============================================================================
# Equations for climbing NEB
# =============================================================================


@wp.func
def climbing_image_effective_force(
    physical_force: Any,
    tangent: Any,
    force_dot_tangent: Any,
):
    r"""Return the climbing-image NEB force on one atom.

    Removes the spring force and reverses the physical-force component parallel
    to the path.

    Parameters
    ----------
    physical_force : vector
        Physical force on one atom in the climbing image.
    tangent : vector
        Per-atom component of the normalized image tangent.
    force_dot_tangent : scalar
        Image-wide inner product of the physical force and normalized tangent.

    Returns
    -------
    vector
        Climbing-image force on the atom.
    """
    two = type(force_dot_tangent)(2.0)
    return physical_force - two * force_dot_tangent * tangent
