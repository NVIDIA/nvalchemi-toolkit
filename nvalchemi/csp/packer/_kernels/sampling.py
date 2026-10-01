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

"""Warp sampling kernels for candidate cells and molecular rigid states.

These kernels retain the pinned Packer's counter-based Warp draws so a single
Torch RNG seed can reproduce the same candidate initialization on CPU and GPU.
"""

from __future__ import annotations

import math

import torch
import warp as wp

from nvalchemi.csp.packer._interop import as_warp, scoped_warp_stream

_PI = math.pi
_TWO_PI = 2.0 * math.pi
_HALF_PI = 0.5 * math.pi
_TWO_PI_OVER_THREE = 2.0 * math.pi / 3.0

TRICLINIC = 0
MONOCLINIC = 1
ORTHORHOMBIC = 2
TETRAGONAL = 3
TRIGONAL = 4
HEXAGONAL = 5
CUBIC = 6


@wp.func
def _crystal_system(group: int) -> int:
    """Map an international space-group number to its crystal-system index."""
    if group <= 2:
        return TRICLINIC
    if group <= 15:
        return MONOCLINIC
    if group <= 74:
        return ORTHORHOMBIC
    if group <= 142:
        return TETRAGONAL
    if group <= 167:
        return TRIGONAL
    if group <= 194:
        return HEXAGONAL
    return CUBIC


@wp.func
def _angle_cosines(rng: wp.uint32) -> wp.vec3f:
    """Draw three unconstrained cell-angle cosines in [-0.5, 0.5]."""
    draws = wp.vec3f(wp.randf(rng), wp.randf(rng), wp.randf(rng))
    return draws - wp.vec3f(0.5)


@wp.func
def _free_lengths(rng: wp.uint32, ratio: wp.float32) -> wp.vec3f:
    """Draw dimensionless cell-axis ratios with unit geometric mean."""
    half_width = 0.5 * wp.log(ratio)
    u = wp.vec3f(wp.randf(rng), wp.randf(rng), wp.randf(rng))
    x = (2.0 * u - wp.vec3f(1.0)) * half_width
    mean = (x[0] + x[1] + x[2]) / 3.0
    centered = x - wp.vec3f(mean)
    return wp.vec3f(wp.exp(centered[0]), wp.exp(centered[1]), wp.exp(centered[2]))


@wp.func
def _unique_lengths(rng: wp.uint32, ratio: wp.float32) -> wp.vec3f:
    """Draw dimensionless tetragonal or hexagonal lengths whose product is one."""
    q = wp.exp((2.0 * wp.randf(rng) - 1.0) * wp.log(ratio))
    ab = wp.pow(q, -1.0 / 3.0)
    return wp.vec3f(ab, ab, wp.pow(q, 2.0 / 3.0))


@wp.func
def _metric(
    system: int,
    ratio_lengths: wp.vec3f,
    free_angles: wp.vec3f,
    unique: wp.vec3f,
) -> wp.mat33f:
    """Assemble normalized cell lengths and angle cosines for a crystal system."""
    lengths = ratio_lengths
    angles = free_angles
    if system == MONOCLINIC:
        angles = wp.vec3f(0.0, angles[1], 0.0)
    elif system == ORTHORHOMBIC:
        angles = wp.vec3f(0.0, 0.0, 0.0)
    elif system == TETRAGONAL:
        lengths = unique
        angles = wp.vec3f(0.0, 0.0, 0.0)
    elif system == TRIGONAL or system == HEXAGONAL:
        lengths = unique
        angles = wp.vec3f(0.0, 0.0, -0.5)
    elif system == CUBIC:
        lengths = wp.vec3f(1.0, 1.0, 1.0)
        angles = wp.vec3f(0.0, 0.0, 0.0)
    return wp.mat33f(
        lengths[0],
        lengths[1],
        lengths[2],
        angles[0],
        angles[1],
        angles[2],
        0.0,
        0.0,
        0.0,
    )


@wp.func
def _cell_from_metric(values: wp.mat33f) -> wp.mat33f:
    """Build a lower-triangular cell matrix from lengths and angle cosines."""
    lengths = wp.vec3f(values[0, 0], values[0, 1], values[0, 2])
    cosines = wp.vec3f(values[1, 0], values[1, 1], values[1, 2])
    ca, cb, cg = cosines[0], cosines[1], cosines[2]
    sin_gamma = wp.sqrt(1.0 - cg * cg)
    gram = wp.max(
        1.0 + 2.0 * ca * cb * cg - ca * ca - cb * cb - cg * cg,
        1.0e-12,
    )
    a, b, c = lengths[0], lengths[1], lengths[2]
    return wp.mat33f(
        a,
        0.0,
        0.0,
        b * cg,
        b * sin_gamma,
        0.0,
        c * cb,
        c * (ca - cb * cg) / sin_gamma,
        c * wp.sqrt(gram) / sin_gamma,
    )


@wp.func
def _volume(cell: wp.mat33f) -> wp.float32:
    """Return the absolute cell determinant in cubic cell-length units."""
    return wp.abs(wp.determinant(cell))


@wp.func
def _min_height(cell: wp.mat33f, volume: wp.float32) -> wp.float32:
    """Return the shortest perpendicular cell height in cell-length units."""
    a = wp.vec3f(cell[0, 0], cell[0, 1], cell[0, 2])
    b = wp.vec3f(cell[1, 0], cell[1, 1], cell[1, 2])
    c = wp.vec3f(cell[2, 0], cell[2, 1], cell[2, 2])
    ha = volume / wp.max(wp.length(wp.cross(b, c)), 1.0e-12)
    hb = volume / wp.max(wp.length(wp.cross(a, c)), 1.0e-12)
    hc = volume / wp.max(wp.length(wp.cross(a, b)), 1.0e-12)
    return wp.min(wp.min(ha, hb), hc)


@wp.kernel(enable_backward=False)
def _sample_cells_kernel(
    groups: wp.array(dtype=wp.int32),
    cdf: wp.array(dtype=wp.float32),
    v_min: wp.float32,
    v_max: wp.float32,
    min_height: wp.float32,
    max_axis_ratio: wp.float32,
    seed: wp.uint64,
    round_index: wp.int32,
    cells: wp.array(dtype=wp.mat33f),
    selected_groups: wp.array(dtype=wp.int32),
    valid: wp.array(dtype=wp.int32),
) -> None:
    """Sample and validate cell matrices for one trial per Warp thread."""
    tid = wp.tid()
    rng = wp.rand_init(int(seed) + int(round_index) * 104729, tid)
    target = v_min + (v_max - v_min) * wp.randf(rng)
    draw = wp.randf(rng)
    group = groups[groups.shape[0] - 1]
    for index in range(cdf.shape[0]):
        if draw <= cdf[index]:
            group = groups[index]
            break
    system = _crystal_system(group)
    params = _metric(
        system,
        _free_lengths(rng, max_axis_ratio),
        _angle_cosines(rng),
        _unique_lengths(rng, max_axis_ratio),
    )
    cell = _cell_from_metric(params)
    raw_volume = _volume(cell)
    good = wp.isfinite(raw_volume) and raw_volume > 1.0e-8
    if good:
        scale = wp.cbrt(target / raw_volume)
        cell = wp.mat33f(
            cell[0, 0] * scale,
            cell[0, 1] * scale,
            cell[0, 2] * scale,
            cell[1, 0] * scale,
            cell[1, 1] * scale,
            cell[1, 2] * scale,
            cell[2, 0] * scale,
            cell[2, 1] * scale,
            cell[2, 2] * scale,
        )
        scaled_volume = _volume(cell)
        height = _min_height(cell, scaled_volume)
        good = (
            wp.isfinite(scaled_volume)
            and scaled_volume > 0.0
            and wp.isfinite(height)
            and height > min_height
        )
    cells[tid] = cell
    selected_groups[tid] = group
    valid[tid] = wp.int32(1) if good else wp.int32(0)


@wp.func
def _rotation_from_quaternion(
    w: wp.float32, x: wp.float32, y: wp.float32, z: wp.float32
) -> wp.mat33f:
    """Convert a quaternion to its 3 by 3 rotation matrix."""
    return wp.mat33f(
        1.0 - 2.0 * (y * y + z * z),
        2.0 * (x * y - z * w),
        2.0 * (x * z + y * w),
        2.0 * (x * y + z * w),
        1.0 - 2.0 * (x * x + z * z),
        2.0 * (y * z - x * w),
        2.0 * (x * z - y * w),
        2.0 * (y * z + x * w),
        1.0 - 2.0 * (x * x + y * y),
    )


@wp.func
def _random_rotation(rng: wp.uint32) -> wp.mat33f:
    """Draw a random molecular rotation matrix from the Warp RNG state."""
    u1, u2, u3 = wp.randf(rng), wp.randf(rng), wp.randf(rng)
    r1, r2 = wp.sqrt(1.0 - u1), wp.sqrt(u1)
    t1, t2 = _TWO_PI * u2, _TWO_PI * u3
    x, y = r1 * wp.sin(t1), r1 * wp.cos(t1)
    z, w = r2 * wp.sin(t2), r2 * wp.cos(t2)
    return _rotation_from_quaternion(w, x, y, z)


@wp.kernel(enable_backward=False)
def _initialize_kernel(
    rows: wp.array(dtype=wp.int32),
    sampled_cells: wp.array(dtype=wp.mat33f),
    sampled_groups: wp.array(dtype=wp.int32),
    op_indices: wp.array(dtype=wp.int32),
    op_ptr: wp.array(dtype=wp.int32),
    conformer_starts: wp.array(dtype=wp.int32),
    conformer_stops: wp.array(dtype=wp.int32),
    seed: wp.uint64,
    conformer_ids: wp.array(dtype=wp.int32, ndim=2),
    centers: wp.array(dtype=wp.float32, ndim=3),
    rotations: wp.array(dtype=wp.float32, ndim=4),
    cells: wp.array(dtype=wp.mat33f),
    inverse_cells: wp.array(dtype=wp.mat33f),
    reference_volumes: wp.array(dtype=wp.float32),
    groups: wp.array(dtype=wp.int32),
    selected_ops: wp.array(dtype=wp.int32, ndim=2),
    steps: wp.array(dtype=wp.int32),
) -> None:
    """Initialize selected candidate rows from sampled cells and random molecular states."""
    tid = wp.tid()
    row = rows[tid]
    rng = wp.rand_init(int(seed), tid)
    cell = sampled_cells[tid]
    cells[row] = cell
    inverse_cells[row] = wp.inverse(cell)
    reference_volumes[row] = _volume(cell)
    group = sampled_groups[tid]
    groups[row] = group
    op_start = op_ptr[group - 1]
    for symop in range(selected_ops.shape[1]):
        selected_ops[row, symop] = op_indices[op_start + symop]
    for molecule in range(conformer_ids.shape[1]):
        first = conformer_starts[molecule]
        count = conformer_stops[molecule] - first
        offset = wp.min(wp.int32(wp.randf(rng) * wp.float32(count)), count - 1)
        conformer_ids[row, molecule] = first + offset
        centers[row, molecule, 0] = wp.randf(rng)
        centers[row, molecule, 1] = wp.randf(rng)
        centers[row, molecule, 2] = wp.randf(rng)
        rotation = _random_rotation(rng)
        for i in range(3):
            for j in range(3):
                rotations[row, molecule, i, j] = rotation[i, j]
    steps[row] = 0


def sample_cell_trials(
    *,
    num_trials: int,
    candidate_groups: torch.Tensor,
    probabilities: torch.Tensor,
    volume_range: tuple[float, float],
    min_cell_height: float,
    max_axis_ratio: float,
    seed: int,
    round_index: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample one round of cells and return its validity mask."""
    device = candidate_groups.device
    weights = probabilities / probabilities.sum()
    cdf = torch.cumsum(weights, dim=0)
    cdf[-1] = 1.0
    cells = torch.empty((num_trials, 3, 3), dtype=torch.float32, device=device)
    groups = torch.empty((num_trials,), dtype=torch.int32, device=device)
    valid = torch.empty((num_trials,), dtype=torch.int32, device=device)
    with scoped_warp_stream(cells):
        wp.launch(
            _sample_cells_kernel,
            dim=num_trials,
            inputs=[
                as_warp(candidate_groups, wp.int32),
                as_warp(cdf.contiguous(), wp.float32),
                wp.float32(volume_range[0]),
                wp.float32(volume_range[1]),
                wp.float32(min_cell_height),
                wp.float32(max_axis_ratio),
                wp.uint64(seed),
                wp.int32(round_index),
                as_warp(cells, wp.mat33f),
                as_warp(groups, wp.int32),
                as_warp(valid, wp.int32),
            ],
            device=str(device),
        )
    return cells, groups, valid


def initialize_rows(
    *,
    rows: torch.Tensor,
    sampled_cells: torch.Tensor,
    sampled_groups: torch.Tensor,
    op_indices: torch.Tensor,
    op_ptr: torch.Tensor,
    conformer_starts: torch.Tensor,
    conformer_stops: torch.Tensor,
    seed: int,
    conformer_ids: torch.Tensor,
    centers: torch.Tensor,
    rotations: torch.Tensor,
    cells: torch.Tensor,
    inverse_cells: torch.Tensor,
    reference_volumes: torch.Tensor,
    groups: torch.Tensor,
    selected_ops: torch.Tensor,
    steps: torch.Tensor,
) -> None:
    """Initialize/refill selected rows using the pinned Warp draw order."""
    if rows.numel() == 0:
        return
    with scoped_warp_stream(rows):
        wp.launch(
            _initialize_kernel,
            dim=rows.numel(),
            inputs=[
                as_warp(rows.contiguous(), wp.int32),
                as_warp(sampled_cells.contiguous(), wp.mat33f),
                as_warp(sampled_groups.contiguous(), wp.int32),
                as_warp(op_indices, wp.int32),
                as_warp(op_ptr, wp.int32),
                as_warp(conformer_starts, wp.int32),
                as_warp(conformer_stops, wp.int32),
                wp.uint64(seed),
                as_warp(conformer_ids, wp.int32),
                as_warp(centers, wp.float32),
                as_warp(rotations, wp.float32),
                as_warp(cells, wp.mat33f),
                as_warp(inverse_cells, wp.mat33f),
                as_warp(reference_volumes, wp.float32),
                as_warp(groups, wp.int32),
                as_warp(selected_ops, wp.int32),
                as_warp(steps, wp.int32),
            ],
            device=str(rows.device),
        )


def generate_cells(
    *,
    count: int,
    candidate_groups: torch.Tensor,
    probabilities: torch.Tensor,
    volume_range: tuple[float, float],
    min_cell_height: float,
    max_axis_ratio: float,
    oversample_factor: float,
    seed: int,
    max_rounds: int = 128,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rejection-sample an ordered block of geometrically valid cells."""
    cell_blocks: list[torch.Tensor] = []
    group_blocks: list[torch.Tensor] = []
    total = 0
    for round_index in range(max_rounds):
        remaining = count - total
        if remaining <= 0:
            break
        trials = max(1, math.ceil(remaining * oversample_factor))
        round_cells, round_groups, valid = sample_cell_trials(
            num_trials=trials,
            candidate_groups=candidate_groups,
            probabilities=probabilities,
            volume_range=volume_range,
            min_cell_height=min_cell_height,
            max_axis_ratio=max_axis_ratio,
            seed=seed,
            round_index=round_index,
        )
        accepted = valid.to(dtype=torch.bool)
        if bool(accepted.any()):
            cell_blocks.append(round_cells[accepted])
            group_blocks.append(round_groups[accepted])
            total += int(accepted.sum().item())
    if total < count:
        raise RuntimeError(f"cell rejection sampling accepted {total} of {count} rows")
    return torch.cat(cell_blocks, dim=0)[:count].contiguous(), torch.cat(
        group_blocks, dim=0
    )[:count].contiguous()
