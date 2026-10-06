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

"""Warp kernels for batched rigid-molecule and cell relaxation."""

from __future__ import annotations

import torch
import warp as wp

from nvalchemi.csp.packer._interop import as_warp, scoped_warp_stream


@wp.func
def _identity() -> wp.mat33f:
    """Return the 3 by 3 identity matrix."""
    return wp.mat33f(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)


@wp.func
def _matrix(values: wp.array(dtype=wp.float32, ndim=3), row: int) -> wp.mat33f:
    """Load one 3 by 3 matrix from a batched tensor."""
    return wp.mat33f(
        values[row, 0, 0],
        values[row, 0, 1],
        values[row, 0, 2],
        values[row, 1, 0],
        values[row, 1, 1],
        values[row, 1, 2],
        values[row, 2, 0],
        values[row, 2, 1],
        values[row, 2, 2],
    )


@wp.func
def _store_matrix(
    values: wp.array(dtype=wp.float32, ndim=3), row: int, matrix: wp.mat33f
) -> None:
    """Write a 3 by 3 matrix to one row of a batched tensor."""
    for i in range(3):
        for j in range(3):
            values[row, i, j] = matrix[i, j]


@wp.func
def _rotation(
    values: wp.array(dtype=wp.float32, ndim=4), row: int, molecule: int
) -> wp.mat33f:
    """Load one molecule rotation matrix from a batched tensor."""
    return wp.mat33f(
        values[row, molecule, 0, 0],
        values[row, molecule, 0, 1],
        values[row, molecule, 0, 2],
        values[row, molecule, 1, 0],
        values[row, molecule, 1, 1],
        values[row, molecule, 1, 2],
        values[row, molecule, 2, 0],
        values[row, molecule, 2, 1],
        values[row, molecule, 2, 2],
    )


@wp.func
def _store_rotation(
    values: wp.array(dtype=wp.float32, ndim=4),
    row: int,
    molecule: int,
    rotation: wp.mat33f,
) -> None:
    """Write one molecule rotation matrix to a batched tensor."""
    for i in range(3):
        for j in range(3):
            values[row, molecule, i, j] = rotation[i, j]


@wp.func
def _atom_position(
    positions: wp.array(dtype=wp.float32, ndim=2), atom: int
) -> wp.vec3f:
    """Load one Cartesian atom position from the conformer coordinate array."""
    return wp.vec3f(positions[atom, 0], positions[atom, 1], positions[atom, 2])


@wp.func
def _vector3(
    values: wp.array(dtype=wp.float32, ndim=3), row: int, molecule: int
) -> wp.vec3f:
    """Load one three-component per-molecule vector from a batched tensor."""
    return wp.vec3f(
        values[row, molecule, 0],
        values[row, molecule, 1],
        values[row, molecule, 2],
    )


@wp.func
def _wrap(position: wp.vec3f) -> wp.vec3f:
    """Wrap fractional coordinates into the unit cell interval [0, 1)."""
    return position - wp.vec3f(
        wp.floor(position[0]), wp.floor(position[1]), wp.floor(position[2])
    )


@wp.func
def _rotation_update(delta: wp.vec3f) -> wp.mat33f:
    """Convert a rotation vector in radians to an incremental rotation matrix."""
    theta = wp.length(delta)
    skew = wp.mat33f(
        0.0,
        -delta[2],
        delta[1],
        delta[2],
        0.0,
        -delta[0],
        -delta[1],
        delta[0],
        0.0,
    )
    theta_sq = theta * theta
    a = 1.0 - theta_sq / 6.0
    b = 0.5 - theta_sq / 24.0
    if theta >= 1.0e-6:
        a = wp.sin(theta) / theta
        b = (1.0 - wp.cos(theta)) / theta_sq
    return _identity() + a * skew + b * (skew * skew)


@wp.func
def _orthonormalize(rotation: wp.mat33f) -> wp.mat33f:
    """Restore an approximately orthonormal rotation matrix with Gram-Schmidt."""
    row0 = wp.normalize(wp.vec3f(rotation[0, 0], rotation[0, 1], rotation[0, 2]))
    row1 = wp.vec3f(rotation[1, 0], rotation[1, 1], rotation[1, 2])
    row1 = wp.normalize(row1 - wp.dot(row1, row0) * row0)
    row1 = wp.normalize(row1 - wp.dot(row1, row0) * row0)
    row2 = wp.cross(row0, row1)
    return wp.mat33f(
        row0[0],
        row0[1],
        row0[2],
        row1[0],
        row1[1],
        row1[2],
        row2[0],
        row2[1],
        row2[2],
    )


@wp.func
def _fractional_delta(delta: wp.vec3f, inverse: wp.mat33f) -> wp.vec3f:
    """Convert a Cartesian displacement to fractional coordinates with the inverse cell."""
    return wp.vec3f(
        delta[0] * inverse[0, 0] + delta[1] * inverse[1, 0] + delta[2] * inverse[2, 0],
        delta[0] * inverse[0, 1] + delta[1] * inverse[1, 1] + delta[2] * inverse[2, 1],
        delta[0] * inverse[0, 2] + delta[1] * inverse[1, 2] + delta[2] * inverse[2, 2],
    )


@wp.kernel(enable_backward=False)
def _relax_molecules(
    active: wp.array(dtype=wp.bool),
    conformer_positions: wp.array(dtype=wp.float32, ndim=2),
    conformer_ptr: wp.array(dtype=wp.int32),
    conformer_ids: wp.array(dtype=wp.int32, ndim=2),
    molecule_atom_ptr: wp.array(dtype=wp.int32),
    forces: wp.array(dtype=wp.float32, ndim=3),
    torques: wp.array(dtype=wp.float32, ndim=3),
    max_overlap: wp.array(dtype=wp.float32),
    inverse_cells: wp.array(dtype=wp.float32, ndim=3),
    centers: wp.array(dtype=wp.float32, ndim=3),
    rotations: wp.array(dtype=wp.float32, ndim=4),
    step_scale: wp.float32,
    max_step: wp.float32,
) -> None:
    """Update active molecule centers and orientations from contact forces and torques."""
    row, molecule = wp.tid()
    if not active[row]:
        return
    overlap = max_overlap[row]
    if overlap <= 0.0:
        return

    conf_id = conformer_ids[row, molecule]
    atom_start = conformer_ptr[conf_id]
    atom_count = molecule_atom_ptr[molecule + 1] - molecule_atom_ptr[molecule]
    rotation = _rotation(rotations, row, molecule)
    inertia = wp.mat33f()
    for local_atom in range(atom_count):
        relative = rotation * _atom_position(
            conformer_positions, atom_start + local_atom
        )
        inertia += wp.dot(relative, relative) * _identity() - wp.outer(
            relative, relative
        )
    inertia[0, 0] += 1.0e-6
    inertia[1, 1] += 1.0e-6
    inertia[2, 2] += 1.0e-6

    translation = step_scale * _vector3(forces, row, molecule)
    rotation_delta = step_scale * (
        wp.inverse(inertia) * _vector3(torques, row, molecule)
    )
    maximum_displacement = float(0.0)
    for local_atom in range(atom_count):
        relative = rotation * _atom_position(
            conformer_positions, atom_start + local_atom
        )
        induced = translation + wp.cross(rotation_delta, relative)
        maximum_displacement = wp.max(maximum_displacement, wp.length(induced))
    atom_step = wp.min(max_step, overlap)
    if maximum_displacement > atom_step:
        scale = atom_step / wp.max(maximum_displacement, 1.0e-12)
        translation = scale * translation
        rotation_delta = scale * rotation_delta

    center = wp.vec3f(
        centers[row, molecule, 0], centers[row, molecule, 1], centers[row, molecule, 2]
    )
    center = _wrap(center + _fractional_delta(translation, _matrix(inverse_cells, row)))
    centers[row, molecule, 0] = center[0]
    centers[row, molecule, 1] = center[1]
    centers[row, molecule, 2] = center[2]
    _store_rotation(
        rotations,
        row,
        molecule,
        _orthonormalize(_rotation_update(rotation_delta) * rotation),
    )


@wp.func
def _crystal_system(group: int) -> int:
    """Map an international space-group number to its crystal-system index."""
    if group <= 2:
        return 0
    if group <= 15:
        return 1
    if group <= 74:
        return 2
    if group <= 142:
        return 3
    if group <= 167:
        return 4
    if group <= 194:
        return 5
    return 6


@wp.func
def _project_stress(stress: wp.mat33f, system: int) -> wp.mat33f:
    """Keep only stress components allowed by the crystal-system index."""
    symmetric = 0.5 * (stress + wp.transpose(stress))
    projected = wp.mat33f()
    if system == 0:
        return symmetric
    if system == 1:
        projected[0, 0] = symmetric[0, 0]
        projected[1, 1] = symmetric[1, 1]
        projected[2, 2] = symmetric[2, 2]
        projected[0, 2] = symmetric[0, 2]
        projected[2, 0] = symmetric[0, 2]
    elif system == 2:
        projected[0, 0] = symmetric[0, 0]
        projected[1, 1] = symmetric[1, 1]
        projected[2, 2] = symmetric[2, 2]
    elif system == 3 or system == 4 or system == 5:
        ab = 0.5 * (symmetric[0, 0] + symmetric[1, 1])
        projected[0, 0] = ab
        projected[1, 1] = ab
        projected[2, 2] = symmetric[2, 2]
    elif system == 6:
        isotropic = (symmetric[0, 0] + symmetric[1, 1] + symmetric[2, 2]) / 3.0
        projected[0, 0] = isotropic
        projected[1, 1] = isotropic
        projected[2, 2] = isotropic
    return projected


@wp.func
def _cell_lengths(cell: wp.mat33f) -> wp.vec3f:
    """Return the lengths of the three cell vectors in the cell length units."""
    return wp.vec3f(
        wp.length(wp.vec3f(cell[0, 0], cell[0, 1], cell[0, 2])),
        wp.length(wp.vec3f(cell[1, 0], cell[1, 1], cell[1, 2])),
        wp.length(wp.vec3f(cell[2, 0], cell[2, 1], cell[2, 2])),
    )


@wp.func
def _cell_angles(cell: wp.mat33f) -> wp.vec3f:
    """Return the cell angles alpha, beta, and gamma in radians."""
    a = wp.vec3f(cell[0, 0], cell[0, 1], cell[0, 2])
    b = wp.vec3f(cell[1, 0], cell[1, 1], cell[1, 2])
    c = wp.vec3f(cell[2, 0], cell[2, 1], cell[2, 2])
    alpha = wp.clamp(wp.dot(b, c) / (wp.length(b) * wp.length(c)), -1.0, 1.0)
    beta = wp.clamp(wp.dot(a, c) / (wp.length(a) * wp.length(c)), -1.0, 1.0)
    gamma = wp.clamp(wp.dot(a, b) / (wp.length(a) * wp.length(b)), -1.0, 1.0)
    return wp.vec3f(wp.acos(alpha), wp.acos(beta), wp.acos(gamma))


@wp.func
def _cell_from_metric(lengths: wp.vec3f, angles: wp.vec3f) -> wp.mat33f:
    """Build a lower-triangular cell matrix from lengths and angles in radians."""
    alpha = wp.cos(angles[0])
    beta = wp.cos(angles[1])
    gamma = wp.cos(angles[2])
    sin_gamma = wp.max(wp.sin(angles[2]), 1.0e-8)
    gram = wp.max(
        1.0 + 2.0 * alpha * beta * gamma - alpha * alpha - beta * beta - gamma * gamma,
        1.0e-12,
    )
    return wp.mat33f(
        lengths[0],
        0.0,
        0.0,
        lengths[1] * gamma,
        lengths[1] * sin_gamma,
        0.0,
        lengths[2] * beta,
        lengths[2] * (alpha - beta * gamma) / sin_gamma,
        lengths[2] * wp.sqrt(gram) / sin_gamma,
    )


@wp.func
def _project_cell(cell: wp.mat33f, system: int) -> wp.mat33f:
    """Project cell lengths and angles onto the requested crystal system."""
    lengths = _cell_lengths(cell)
    angles = _cell_angles(cell)
    half_pi = 0.5 * 3.141592653589793
    if system == 1:
        angles[0] = half_pi
        angles[2] = half_pi
    elif system == 2:
        angles = wp.vec3f(half_pi, half_pi, half_pi)
    elif system == 3:
        mean = 0.5 * (lengths[0] + lengths[1])
        lengths[0] = mean
        lengths[1] = mean
        angles = wp.vec3f(half_pi, half_pi, half_pi)
    elif system == 4 or system == 5:
        mean = 0.5 * (lengths[0] + lengths[1])
        lengths[0] = mean
        lengths[1] = mean
        angles = wp.vec3f(half_pi, half_pi, 2.0 * 3.141592653589793 / 3.0)
    elif system == 6:
        mean = (lengths[0] + lengths[1] + lengths[2]) / 3.0
        lengths = wp.vec3f(mean, mean, mean)
        angles = wp.vec3f(half_pi, half_pi, half_pi)
    return _cell_from_metric(lengths, angles)


@wp.func
def _minimum_height(cell: wp.mat33f) -> wp.float32:
    """Return the shortest perpendicular cell height in the cell length units."""
    a = wp.vec3f(cell[0, 0], cell[0, 1], cell[0, 2])
    b = wp.vec3f(cell[1, 0], cell[1, 1], cell[1, 2])
    c = wp.vec3f(cell[2, 0], cell[2, 1], cell[2, 2])
    volume = wp.abs(wp.determinant(cell))
    h_a = volume / wp.max(wp.length(wp.cross(b, c)), 1.0e-12)
    h_b = volume / wp.max(wp.length(wp.cross(a, c)), 1.0e-12)
    h_c = volume / wp.max(wp.length(wp.cross(a, b)), 1.0e-12)
    return wp.min(wp.min(h_a, h_b), h_c)


@wp.kernel(enable_backward=False)
def _relax_cells(
    active: wp.array(dtype=wp.bool),
    space_groups: wp.array(dtype=wp.int32),
    reference_volumes: wp.array(dtype=wp.float32),
    max_overlap: wp.array(dtype=wp.float32),
    virial: wp.array(dtype=wp.float32, ndim=3),
    cells: wp.array(dtype=wp.float32, ndim=3),
    inverse_cells: wp.array(dtype=wp.float32, ndim=3),
    steps: wp.array(dtype=wp.int32),
    expanded_atom_count: wp.int32,
    cell_step_scale: wp.float32,
    max_cell_strain: wp.float32,
    volume_compression_scale: wp.float32,
) -> None:
    """Update active cells using projected virial while retaining crystal symmetry."""
    row = wp.tid()
    if not active[row]:
        return
    steps[row] = steps[row] + 1
    overlap = max_overlap[row]
    if cell_step_scale <= 0.0 or overlap <= 0.0:
        return

    cell = _matrix(cells, row)
    system = _crystal_system(space_groups[row])
    volume = wp.abs(wp.determinant(cell))
    stress = wp.mat33f(
        virial[row, 0, 0],
        virial[row, 0, 1],
        virial[row, 0, 2],
        virial[row, 1, 0],
        virial[row, 1, 1],
        virial[row, 1, 2],
        virial[row, 2, 0],
        virial[row, 2, 1],
        virial[row, 2, 2],
    )
    projected = _project_stress(stress, system)
    isotropic = (projected[0, 0] + projected[1, 1] + projected[2, 2]) / 3.0
    shape = _project_stress(projected - isotropic * _identity(), system)
    shape_max = float(0.0)
    for i in range(3):
        for j in range(3):
            shape_max = wp.max(shape_max, wp.abs(shape[i, j]))
    strain = wp.mat33f()
    if shape_max > 1.0e-12:
        overlap_scale = wp.clamp(overlap, 0.0, 1.0)
        overlap_scale = overlap_scale * overlap_scale
        strain = cell_step_scale * overlap_scale * shape / shape_max
        for i in range(3):
            for j in range(3):
                strain[i, j] = wp.clamp(strain[i, j], -max_cell_strain, max_cell_strain)
    candidate = _project_cell(cell * (_identity() + strain), system)
    candidate_volume = wp.abs(wp.determinant(candidate))
    if candidate_volume <= 1.0e-8:
        return
    candidate = candidate * wp.cbrt(volume / candidate_volume)
    excess = wp.max(volume - reference_volumes[row], 0.0)
    pressure_strain = wp.clamp(
        cell_step_scale
        * volume_compression_scale
        * excess
        / wp.float32(expanded_atom_count),
        0.0,
        max_cell_strain,
    )
    if pressure_strain > 0.0:
        candidate = candidate * (1.0 - pressure_strain)
    if _minimum_height(candidate) < 2.5:
        return
    _store_matrix(cells, row, candidate)
    _store_matrix(inverse_cells, row, wp.inverse(candidate))


def relax_step(
    *,
    active: torch.Tensor,
    conformer_positions: torch.Tensor,
    conformer_ptr: torch.Tensor,
    conformer_ids: torch.Tensor,
    molecule_atom_ptr: torch.Tensor,
    forces: torch.Tensor,
    torques: torch.Tensor,
    max_overlap: torch.Tensor,
    virial: torch.Tensor,
    cells: torch.Tensor,
    inverse_cells: torch.Tensor,
    centers: torch.Tensor,
    rotations: torch.Tensor,
    space_groups: torch.Tensor,
    reference_volumes: torch.Tensor,
    steps: torch.Tensor,
    expanded_atom_count: int,
    step_scale: float,
    max_step: float,
    cell_step_scale: float,
    max_cell_strain: float,
    volume_compression_scale: float,
) -> None:
    """Apply one batched rigid-body and crystal-system cell update."""
    batch_size, molecule_count = conformer_ids.shape
    if batch_size == 0:
        return
    with scoped_warp_stream(cells):
        wp.launch(
            _relax_molecules,
            dim=(batch_size, molecule_count),
            inputs=[
                as_warp(active, wp.bool),
                as_warp(conformer_positions, wp.float32),
                as_warp(conformer_ptr, wp.int32),
                as_warp(conformer_ids, wp.int32),
                as_warp(molecule_atom_ptr, wp.int32),
                as_warp(forces, wp.float32),
                as_warp(torques, wp.float32),
                as_warp(max_overlap, wp.float32),
                as_warp(inverse_cells, wp.float32),
                as_warp(centers, wp.float32),
                as_warp(rotations, wp.float32),
                wp.float32(step_scale),
                wp.float32(max_step),
            ],
            device=str(cells.device),
        )
        wp.launch(
            _relax_cells,
            dim=batch_size,
            inputs=[
                as_warp(active, wp.bool),
                as_warp(space_groups, wp.int32),
                as_warp(reference_volumes, wp.float32),
                as_warp(max_overlap, wp.float32),
                as_warp(virial, wp.float32),
                as_warp(cells, wp.float32),
                as_warp(inverse_cells, wp.float32),
                as_warp(steps, wp.int32),
                wp.int32(expanded_atom_count),
                wp.float32(cell_step_scale),
                wp.float32(max_cell_strain),
                wp.float32(volume_compression_scale),
            ],
            device=str(cells.device),
        )
