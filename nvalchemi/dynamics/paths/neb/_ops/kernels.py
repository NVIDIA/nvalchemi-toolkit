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

"""Warp kernels for batched NEB force construction.

Two methods are used for NEB force calculations:

- The stored-tangent approach computes the minimal quantities needed and stores
  the computed tangent in a buffer for reuse later.
- The Gram-statistics approach computes dot products among the physical force
  and the forward and backward path displacements, avoiding the need to store
  a per-atom tangent buffer.

Two execution variants are provided for both strategies:

- Tiled kernels use CUDA block-cooperative execution and are selected for CUDA.
- Scalar kernels are device-generic one-thread-per-image implementations selected
  for CPU execution.
"""

from __future__ import annotations

from typing import Any

import warp as wp

from .modes import CLIMBING_NEB, ENDPOINT

__all__ = [
    "build_gram_stats_neb_kernel",
    "build_stored_tangent_neb_kernel",
]

_SCALAR_DTYPES = (wp.float32, wp.float64)


# =============================================================================
# Shared NEB functions
# =============================================================================


@wp.func
def _mic(
    r_a: Any,
    r_b: Any,
    cell: Any,
    inv_cell: Any,
    pbc: wp.vec3b,
) -> Any:
    """Return the minimum-image displacement from ``r_b`` to ``r_a``.

    Parameters
    ----------
    r_a, r_b : wp.vec3f or wp.vec3d
        Cartesian positions whose displacement is computed.
    cell, inv_cell : wp.mat33f or wp.mat33d
        Cell matrix and its inverse for the path.
    pbc : wp.vec3b
        Periodic boundary flags for each cell dimension.

    Returns
    -------
    wp.vec3f or wp.vec3d
        Minimum-image displacement with the same type as ``r_a`` and ``r_b``.
    """
    displacement = r_a - r_b
    fractional = displacement * inv_cell
    half = type(fractional[0])(0.5)
    if pbc[0]:
        fractional[0] = fractional[0] - wp.floor(fractional[0] + half)
    if pbc[1]:
        fractional[1] = fractional[1] - wp.floor(fractional[1] + half)
    if pbc[2]:
        fractional[2] = fractional[2] - wp.floor(fractional[2] + half)
    return fractional * cell


@wp.func
def _default_rtol(value: wp.float32) -> wp.float32:
    """Return the default relative tolerance for float32 calculations."""
    return wp.float32(1.0e-6)


@wp.func
def _default_rtol(value: wp.float64) -> wp.float64:
    """Return the default relative tolerance for float64 calculations."""
    return wp.float64(1.0e-12)


@wp.func
def _select_tangent_fallback(
    weight_plus: Any,
    weight_minus: Any,
    dplus_norm: Any,
    dminus_norm: Any,
    tangent_norm: Any,
):
    """Return the stable tangent choice and its resolved norm.

    The fallback code determines the tangent used by the caller:

    - ``0``: use the original tangent.
    - ``1``: use ``dplus`` as the tangent.
    - ``2``: use ``dminus`` as the tangent.
    - ``3``: no tangent can be computed, the effective NEB force is set to zero.
    """
    zero = type(tangent_norm)(0.0)
    tangent_scale = (
        wp.abs(weight_plus) * dplus_norm + wp.abs(weight_minus) * dminus_norm
    )
    fallback = wp.int32(0)
    resolved_norm = tangent_norm
    if tangent_norm <= _default_rtol(tangent_norm) * tangent_scale:
        if dplus_norm >= dminus_norm and dplus_norm > zero:
            fallback = wp.int32(1)
            resolved_norm = dplus_norm
        elif dminus_norm > zero:
            fallback = wp.int32(2)
            resolved_norm = dminus_norm
        else:
            fallback = wp.int32(3)
    return fallback, resolved_norm


# =============================================================================
# NEB force kernels
# =============================================================================


def build_stored_tangent_neb_kernel(
    tangent_weights_fn: wp.Function,
    effective_force_fn: wp.Function,
    climbing_force_fn: wp.Function,
) -> wp.Kernel:
    r"""Build an NEB effective-force kernel with stored tangents.

    The generated kernel evaluates the registered NEB tangent and effective-force
    functions for a batch of path images. It computes and stores each per-atom
    Cartesian tangent in a caller-provided scratch buffer, then reuses the stored
    unit tangent for force projection and effective-force construction instead of
    recomputing it.

    For each active interior image, the unnormalized tangent is built once and
    stored per-atom:

    .. math::

        \boldsymbol{\tau} = w_+\mathbf{d}_+ + w_-\mathbf{d}_-,
        \qquad
        \hat{\boldsymbol{\tau}} = \boldsymbol{\tau} / \lVert\boldsymbol{\tau}\rVert,

    where :math:`w_\pm` are the tangent weights from ``tangent_weights_fn`` (see
    :func:`~.equations.improved_tangent_weights`) and :math:`\mathbf{d}_\pm` are
    the per-atom forward/backward minimum-image displacements. The stored unit
    tangent :math:`\hat{\boldsymbol{\tau}}` is then reduced against the physical
    force to form the scalar :math:`\mathbf{F}\cdot\hat{\boldsymbol{\tau}}` and
    passed, together with the link norms and spring constants, to
    ``effective_force_fn`` (see :func:`~.equations.neb_effective_force`) to
    evaluate the regular effective force :math:`\mathbf{F}^{\mathrm{NEB}}`.
    Climbing images instead use the injected ``climbing_force_fn``.

    The implementation assigns one cooperative block to each image. On CPU,
    Warp executes the block with one lane, which serially scans the image atoms.
    """

    @wp.kernel(enable_backward=False)
    def stored_tangent_neb_forces_kernel(
        positions: wp.array(dtype=Any),
        physical_forces: wp.array(dtype=Any),
        image_energies: wp.array(dtype=Any),
        image_ptr: wp.array(dtype=wp.int32),
        path_ptr: wp.array(dtype=wp.int32),
        image_path_idx: wp.array(dtype=wp.int32),
        spring_constants: wp.array(dtype=Any),
        image_force_mode: wp.array(dtype=wp.int32),
        path_energy_ref: wp.array(dtype=Any),
        path_energy_max: wp.array(dtype=Any),
        cell: wp.array(dtype=Any),
        inv_cell: wp.array(dtype=Any),
        pbc: wp.array(dtype=wp.vec3b),
        tangent_buffer: wp.array(dtype=Any),
        effective_forces: wp.array(dtype=Any),
        link_lengths: wp.array(dtype=Any),
    ):
        """Compute fused NEB forces using a stored per-atom tangent.
        
        One block owns each image. The kernel stores per-atom tangents and
        reduces only the scalar statistics needed to construct effective forces.

        Parameters
        ----------
        positions, physical_forces : wp.array, shape (num_atoms,), dtype wp.vec3f or wp.vec3d
            Atomic positions and corresponding physical forces.
        image_energies : wp.array, shape (num_images,), dtype wp.float32 or wp.float64
            Physical energy of each image.
        image_ptr : wp.array, shape (num_images + 1,), dtype wp.int32
            CSR-style offsets delimiting the contiguous atom range for each image.
        path_ptr : wp.array, shape (num_paths + 1,), dtype wp.int32
            CSR-style offsets delimiting the contiguous image range for each path.
        image_path_idx : wp.array, shape (num_images,), dtype wp.int32
            Path index of each image.
        spring_constants : wp.array, shape (num_images - num_paths,), dtype wp.float32 or wp.float64
            Spring constant for each forward link in path-major order.
        image_force_mode : wp.array, shape (num_images,), dtype wp.int32
            Effective-force mode of each image.
        path_energy_ref, path_energy_max : wp.array, shape (num_paths,), dtype wp.float32 or wp.float64
            Reference and maximum interior-image energies for each path.
        cell, inv_cell : wp.array, shape (num_paths,), dtype wp.mat33f or wp.mat33d
            Cell matrix and its inverse for each path.
        pbc : wp.array, shape (num_paths,), dtype wp.vec3b
            Periodic boundary flags for each path and Cartesian dimension.
        tangent_buffer : wp.array, shape (num_atoms,), dtype wp.vec3f or wp.vec3d
            Scratch space for constructed tangents, overwritten with unit tangents.
        effective_forces : wp.array, shape (num_atoms,), dtype wp.vec3f or wp.vec3d
            Output effective NEB force for each atom.
        link_lengths : wp.array, shape (num_images - num_paths,), dtype wp.float32 or wp.float64
            Output minimum-image length of each forward link.

        Modifies
        --------
        tangent_buffer, effective_forces, link_lengths
            Writes tangent scratch, effective atomic forces, and forward-link
            lengths.
        """
        image_idx, lane = wp.tid()
        path_idx = image_path_idx[image_idx]
        path_pbc = pbc[path_idx]
        mode = image_force_mode[image_idx]
        atom_start = image_ptr[image_idx]
        atom_stop = image_ptr[image_idx + 1]
        is_first = image_idx == path_ptr[path_idx]
        is_last = image_idx == path_ptr[path_idx + 1] - 1
        is_interior = not is_first and not is_last
        should_compute_force = is_interior
        zero = type(image_energies[0])(0.0)

        # --- Tangent-weight selection ---

        # Initialize endpoint-safe values
        energy_prev = image_energies[image_idx]
        energy_curr = image_energies[image_idx]
        energy_next = image_energies[image_idx]
        weight_plus = zero
        weight_minus = zero
        if should_compute_force:
            energy_prev = image_energies[image_idx - 1]
            energy_next = image_energies[image_idx + 1]
            weight_plus, weight_minus = tangent_weights_fn(
                energy_prev, energy_curr, energy_next
            )

        # --- Link output, tangent storage, image-wide reductions ---

        # Every non-last image reduces and writes its image-wide forward-link norm.
        # Active interior images also store per-atom tangent vectors and reduce the
        # image-wide quantities.
        local_dplus_sq = zero
        local_dminus_sq = zero
        local_tangent_sq = zero
        if not is_last:
            atom_idx = atom_start + lane
            while atom_idx < atom_stop:
                atom_local = atom_idx - atom_start
                next_atom_idx = image_ptr[image_idx + 1] + atom_local
                d_plus = _mic(
                    positions[next_atom_idx],
                    positions[atom_idx],
                    cell[path_idx],
                    inv_cell[path_idx],
                    path_pbc,
                )
                local_dplus_sq = local_dplus_sq + wp.dot(d_plus, d_plus)
                if should_compute_force:
                    prev_atom_idx = image_ptr[image_idx - 1] + atom_local
                    d_minus = _mic(
                        positions[atom_idx],
                        positions[prev_atom_idx],
                        cell[path_idx],
                        inv_cell[path_idx],
                        path_pbc,
                    )
                    tangent = weight_plus * d_plus + weight_minus * d_minus
                    tangent_buffer[atom_idx] = tangent
                    local_dminus_sq = local_dminus_sq + wp.dot(d_minus, d_minus)
                    local_tangent_sq = local_tangent_sq + wp.dot(tangent, tangent)
                atom_idx = atom_idx + wp.block_dim()

            dplus_sq = wp.tile_sum(wp.tile(local_dplus_sq))[0]
            if lane == 0:
                link_lengths[image_idx - path_idx] = wp.sqrt(dplus_sq)

        # --- Endpoint forces ---

        if not should_compute_force:
            atom_idx = atom_start + lane
            while atom_idx < atom_stop:
                if mode == ENDPOINT:
                    effective_forces[atom_idx] = physical_forces[atom_idx]
                else:
                    effective_forces[atom_idx] = positions[atom_idx] * zero
                atom_idx = atom_idx + wp.block_dim()
            return

        # --- The rest of image-wide reductions ---

        dminus_sq = wp.tile_sum(wp.tile(local_dminus_sq))[0]
        tangent_sq = wp.tile_sum(wp.tile(local_tangent_sq))[0]
        dplus_norm = wp.sqrt(dplus_sq)
        dminus_norm = wp.sqrt(dminus_sq)
        tangent_norm = wp.sqrt(tangent_sq)

        # --- Zero-tangent fallback ---

        fallback, tangent_norm = _select_tangent_fallback(
            weight_plus,
            weight_minus,
            dplus_norm,
            dminus_norm,
            tangent_norm,
        )

        # Store one unit tangent, including the selected fallback.
        atom_idx = atom_start + lane
        while atom_idx < atom_stop:
            atom_local = atom_idx - atom_start
            tangent = positions[atom_idx] * zero
            if fallback == 0:
                tangent = tangent_buffer[atom_idx] / tangent_norm
            elif fallback == 1:
                next_atom_idx = image_ptr[image_idx + 1] + atom_local
                d_plus = _mic(
                    positions[next_atom_idx],
                    positions[atom_idx],
                    cell[path_idx],
                    inv_cell[path_idx],
                    path_pbc,
                )
                tangent = d_plus / tangent_norm
            elif fallback == 2:
                prev_atom_idx = image_ptr[image_idx - 1] + atom_local
                d_minus = _mic(
                    positions[atom_idx],
                    positions[prev_atom_idx],
                    cell[path_idx],
                    inv_cell[path_idx],
                    path_pbc,
                )
                tangent = d_minus / tangent_norm
            tangent_buffer[atom_idx] = tangent
            atom_idx = atom_idx + wp.block_dim()

        # --- Force projection and effective forces ---

        local_force_dot_unit_tangent = zero
        atom_idx = atom_start + lane
        while atom_idx < atom_stop:
            local_force_dot_unit_tangent = local_force_dot_unit_tangent + wp.dot(
                physical_forces[atom_idx], tangent_buffer[atom_idx]
            )
            atom_idx = atom_idx + wp.block_dim()
        force_dot_unit_tangent = wp.tile_sum(wp.tile(local_force_dot_unit_tangent))[0]

        forward_link = image_idx - path_idx
        k_plus = spring_constants[forward_link]
        k_minus = spring_constants[forward_link - 1]
        atom_idx = atom_start + lane
        while atom_idx < atom_stop:
            if fallback == 3:
                effective_forces[atom_idx] = positions[atom_idx] * zero
            elif mode == CLIMBING_NEB:
                effective_forces[atom_idx] = climbing_force_fn(
                    physical_forces[atom_idx],
                    tangent_buffer[atom_idx],
                    force_dot_unit_tangent,
                )
            else:
                effective_forces[atom_idx] = effective_force_fn(
                    physical_forces[atom_idx],
                    tangent_buffer[atom_idx],
                    force_dot_unit_tangent,
                    k_plus,
                    k_minus,
                    dplus_norm,
                    dminus_norm,
                    energy_prev,
                    energy_curr,
                    energy_next,
                    path_energy_ref[path_idx],
                    path_energy_max[path_idx],
                )
            atom_idx = atom_idx + wp.block_dim()

    return stored_tangent_neb_forces_kernel


def build_gram_stats_neb_kernel(
    tangent_weights_fn: wp.Function,
    effective_force_fn: wp.Function,
    climbing_force_fn: wp.Function,
) -> wp.Kernel:
    r"""Build an NEB effective-force kernel using Gram statistics.

    The kernel reduces image-wide inner products among the physical force and the
    forward and backward path displacements. These statistics expose the
    projections needed by a wider range of force laws, such as DNEB, without
    storing a per-atom tangent buffer. For regular NEB, the additional reductions
    and repeated displacement work may be slower than the stored-tangent strategy.

    Instead of forming :math:`\hat{\boldsymbol{\tau}}` and storing it per atom,
    the kernel reduces the image-wide Gram products
    :math:`\mathbf{d}_+\cdot\mathbf{d}_-`, :math:`\mathbf{F}\cdot\mathbf{d}_+`,
    :math:`\mathbf{F}\cdot\mathbf{d}_-`, and :math:`\lVert\mathbf{F}\rVert^2`
    (alongside :math:`\lVert\mathbf{d}_+\rVert`, :math:`\lVert\mathbf{d}_-\rVert`)
    and combines them with the tangent weights :math:`w_\pm` to obtain the tangent
    norm and the scalar projections analytically:

    .. math::

        \lVert\boldsymbol{\tau}\rVert^2
        = w_+^2\lVert\mathbf{d}_+\rVert^2
        + 2 w_+ w_-\,(\mathbf{d}_+\cdot\mathbf{d}_-)
        + w_-^2\lVert\mathbf{d}_-\rVert^2,
        \qquad
        \mathbf{F}\cdot\hat{\boldsymbol{\tau}}
        = \frac{w_+(\mathbf{F}\cdot\mathbf{d}_+) + w_-(\mathbf{F}\cdot\mathbf{d}_-)}
               {\lVert\boldsymbol{\tau}\rVert}.

    These projections, together with the per-atom displacements
    :math:`\mathbf{d}_\pm` and their link projections onto
    :math:`\hat{\boldsymbol{\tau}}`, are passed to ``effective_force_fn`` (see
    :func:`~.equations.neb_effective_force_from_gram_stats`) to evaluate the
    regular :math:`\mathbf{F}^{\mathrm{NEB}}` for each atom, without ever
    materializing :math:`\hat{\boldsymbol{\tau}}` in a scratch buffer. Climbing
    images instead use the injected ``climbing_force_fn``.

    The implementation assigns one cooperative block to each image. On CPU,
    Warp executes the block with one lane, which serially scans the image atoms.
    """

    @wp.kernel(enable_backward=False)
    def gram_stats_neb_forces_kernel(
        positions: wp.array(dtype=Any),
        physical_forces: wp.array(dtype=Any),
        image_energies: wp.array(dtype=Any),
        image_ptr: wp.array(dtype=wp.int32),
        path_ptr: wp.array(dtype=wp.int32),
        image_path_idx: wp.array(dtype=wp.int32),
        spring_constants: wp.array(dtype=Any),
        image_force_mode: wp.array(dtype=wp.int32),
        path_energy_ref: wp.array(dtype=Any),
        path_energy_max: wp.array(dtype=Any),
        cell: wp.array(dtype=Any),
        inv_cell: wp.array(dtype=Any),
        pbc: wp.array(dtype=wp.vec3b),
        effective_forces: wp.array(dtype=Any),
        link_lengths: wp.array(dtype=Any),
    ):
        """Compute fused NEB forces from per-image Gram statistics.

        One block owns each image. The kernel stores no per-atom tangent scratch.
        Instead, it reduces pairwise products of the link and force vectors.

        Parameters
        ----------
        positions, physical_forces : wp.array, shape (num_atoms,), dtype wp.vec3f or wp.vec3d
            Atomic positions and corresponding physical forces.
        image_energies : wp.array, shape (num_images,), dtype wp.float32 or wp.float64
            Physical energy of each image.
        image_ptr : wp.array, shape (num_images + 1,), dtype wp.int32
            CSR-style offsets delimiting the contiguous atom range for each image.
        path_ptr : wp.array, shape (num_paths + 1,), dtype wp.int32
            CSR-style offsets delimiting the contiguous image range for each path.
        image_path_idx : wp.array, shape (num_images,), dtype wp.int32
            Path index of each image.
        spring_constants : wp.array, shape (num_images - num_paths,), dtype wp.float32 or wp.float64
            Spring constant for each forward link in path-major order.
        image_force_mode : wp.array, shape (num_images,), dtype wp.int32
            Effective-force mode of each image.
        path_energy_ref, path_energy_max : wp.array, shape (num_paths,), dtype wp.float32 or wp.float64
            Reference and maximum interior-image energies for each path.
        cell, inv_cell : wp.array, shape (num_paths,), dtype wp.mat33f or wp.mat33d
            Cell matrix and its inverse for each path.
        pbc : wp.array, shape (num_paths,), dtype wp.vec3b
            Periodic boundary flags for each path and Cartesian dimension.
        effective_forces : wp.array, shape (num_atoms,), dtype wp.vec3f or wp.vec3d
            Output effective NEB force for each atom.
        link_lengths : wp.array, shape (num_images - num_paths,), dtype wp.float32 or wp.float64
            Output minimum-image length of each forward link.

        Modifies
        --------
        effective_forces, link_lengths
            Writes the effective atomic forces and forward-link lengths.
        """
        image_idx, lane = wp.tid()
        path_idx = image_path_idx[image_idx]
        path_pbc = pbc[path_idx]
        mode = image_force_mode[image_idx]
        atom_start = image_ptr[image_idx]
        atom_stop = image_ptr[image_idx + 1]
        is_first = image_idx == path_ptr[path_idx]
        is_last = image_idx == path_ptr[path_idx + 1] - 1
        is_interior = not is_first and not is_last
        should_compute_force = is_interior
        zero = type(image_energies[0])(0.0)

        # --- Tangent-weight selection ---

        # Initialize endpoint-safe values
        energy_prev = image_energies[image_idx]
        energy_curr = image_energies[image_idx]
        energy_next = image_energies[image_idx]
        weight_plus = zero
        weight_minus = zero
        if should_compute_force:
            energy_prev = image_energies[image_idx - 1]
            energy_next = image_energies[image_idx + 1]
            weight_plus, weight_minus = tangent_weights_fn(
                energy_prev, energy_curr, energy_next
            )

        # --- Link output and image-wide reductions ---

        # Interior images accumulate all pairwise products of d+, d-, and F.
        local_dplus_sq = zero
        local_dplus_dot_dminus = zero
        local_dminus_sq = zero
        local_force_dot_dplus = zero
        local_force_dot_dminus = zero
        local_force_sq = zero
        if not is_last:
            atom_idx = atom_start + lane
            while atom_idx < atom_stop:
                atom_local = atom_idx - atom_start
                next_atom_idx = image_ptr[image_idx + 1] + atom_local
                d_plus = _mic(
                    positions[next_atom_idx],
                    positions[atom_idx],
                    cell[path_idx],
                    inv_cell[path_idx],
                    path_pbc,
                )
                local_dplus_sq = local_dplus_sq + wp.dot(d_plus, d_plus)
                if should_compute_force:
                    prev_atom_idx = image_ptr[image_idx - 1] + atom_local
                    d_minus = _mic(
                        positions[atom_idx],
                        positions[prev_atom_idx],
                        cell[path_idx],
                        inv_cell[path_idx],
                        path_pbc,
                    )
                    physical_force = physical_forces[atom_idx]
                    local_dplus_dot_dminus = local_dplus_dot_dminus + wp.dot(
                        d_plus, d_minus
                    )
                    local_dminus_sq = local_dminus_sq + wp.dot(d_minus, d_minus)
                    local_force_dot_dplus = local_force_dot_dplus + wp.dot(
                        physical_force,
                        d_plus,
                    )
                    local_force_dot_dminus = local_force_dot_dminus + wp.dot(
                        physical_force,
                        d_minus,
                    )
                    local_force_sq = local_force_sq + wp.dot(
                        physical_force, physical_force
                    )
                atom_idx = atom_idx + wp.block_dim()

            dplus_sq = wp.tile_sum(wp.tile(local_dplus_sq))[0]
            if lane == 0:
                link_lengths[image_idx - path_idx] = wp.sqrt(dplus_sq)

        # --- Endpoint forces ---

        if not should_compute_force:
            atom_idx = atom_start + lane
            while atom_idx < atom_stop:
                if mode == ENDPOINT:
                    effective_forces[atom_idx] = physical_forces[atom_idx]
                else:
                    effective_forces[atom_idx] = positions[atom_idx] * zero
                atom_idx = atom_idx + wp.block_dim()
            return

        # --- The rest of image-wide reductions ---

        dplus_dot_dminus = wp.tile_sum(wp.tile(local_dplus_dot_dminus))[0]
        dminus_sq = wp.tile_sum(wp.tile(local_dminus_sq))[0]
        force_dot_dplus = wp.tile_sum(wp.tile(local_force_dot_dplus))[0]
        force_dot_dminus = wp.tile_sum(wp.tile(local_force_dot_dminus))[0]
        force_sq = wp.tile_sum(wp.tile(local_force_sq))[0]
        dplus_norm = wp.sqrt(dplus_sq)
        dminus_norm = wp.sqrt(dminus_sq)
        tangent_sq = (
            weight_plus * weight_plus * dplus_sq
            + type(dplus_sq)(2.0) * weight_plus * weight_minus * dplus_dot_dminus
            + weight_minus * weight_minus * dminus_sq
        )
        tangent_norm = wp.sqrt(wp.max(tangent_sq, zero))

        # --- Zero-tangent fallback ---

        fallback, tangent_norm = _select_tangent_fallback(
            weight_plus,
            weight_minus,
            dplus_norm,
            dminus_norm,
            tangent_norm,
        )

        # --- Additional per-image quantities for effective forces ---

        force_dot_tangent = zero
        dplus_dot_tangent = zero
        dminus_dot_tangent = zero
        if fallback == 0:
            force_dot_tangent = (
                weight_plus * force_dot_dplus + weight_minus * force_dot_dminus
            ) / tangent_norm
            dplus_dot_tangent = (
                weight_plus * dplus_sq + weight_minus * dplus_dot_dminus
            ) / tangent_norm
            dminus_dot_tangent = (
                weight_plus * dplus_dot_dminus + weight_minus * dminus_sq
            ) / tangent_norm
        elif fallback == 1:
            force_dot_tangent = force_dot_dplus / dplus_norm
            dplus_dot_tangent = dplus_norm
            dminus_dot_tangent = dplus_dot_dminus / dplus_norm
        elif fallback == 2:
            force_dot_tangent = force_dot_dminus / dminus_norm
            dplus_dot_tangent = dplus_dot_dminus / dminus_norm
            dminus_dot_tangent = dminus_norm

        # --- Effective forces ---

        forward_link = image_idx - path_idx
        k_plus = spring_constants[forward_link]
        k_minus = spring_constants[forward_link - 1]
        atom_idx = atom_start + lane
        while atom_idx < atom_stop:
            atom_local = atom_idx - atom_start
            next_atom_idx = image_ptr[image_idx + 1] + atom_local
            prev_atom_idx = image_ptr[image_idx - 1] + atom_local
            d_plus = _mic(
                positions[next_atom_idx],
                positions[atom_idx],
                cell[path_idx],
                inv_cell[path_idx],
                path_pbc,
            )
            d_minus = _mic(
                positions[atom_idx],
                positions[prev_atom_idx],
                cell[path_idx],
                inv_cell[path_idx],
                path_pbc,
            )
            tangent = positions[atom_idx] * zero
            if fallback == 0:
                tangent = (weight_plus * d_plus + weight_minus * d_minus) / tangent_norm
            elif fallback == 1:
                tangent = d_plus / dplus_norm
            elif fallback == 2:
                tangent = d_minus / dminus_norm

            if fallback == 3:
                effective_forces[atom_idx] = positions[atom_idx] * zero
            elif mode == CLIMBING_NEB:
                effective_forces[atom_idx] = climbing_force_fn(
                    physical_forces[atom_idx],
                    tangent,
                    force_dot_tangent,
                )
            else:
                effective_forces[atom_idx] = effective_force_fn(
                    physical_forces[atom_idx],
                    tangent,
                    d_plus,
                    d_minus,
                    force_dot_tangent,
                    dplus_dot_tangent,
                    dminus_dot_tangent,
                    dplus_norm,
                    dminus_norm,
                    dplus_dot_dminus,
                    force_dot_dplus,
                    force_dot_dminus,
                    force_sq,
                    k_plus,
                    k_minus,
                    energy_prev,
                    energy_curr,
                    energy_next,
                    path_energy_ref[path_idx],
                    path_energy_max[path_idx],
                )
            atom_idx = atom_idx + wp.block_dim()

    return gram_stats_neb_forces_kernel
