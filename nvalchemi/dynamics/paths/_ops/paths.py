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

"""Warp kernels for per-path energy reductions."""

from __future__ import annotations

from functools import cache
from typing import Any

import warp as wp
from nvalchemiops.warp_dispatch import register_overloads

_SCALAR_DTYPES = (wp.float32, wp.float64)


@wp.kernel(enable_backward=False)
def _path_energy_stats_kernel(
    image_energies: wp.array(dtype=Any),
    path_ptr: wp.array(dtype=wp.int32),
    endpoint_reference_energy: wp.array(dtype=Any),
    highest_interior_energy: wp.array(dtype=Any),
    highest_interior_image_idx: wp.array(dtype=wp.int32),
):
    """Reduce energy statistics with one block per path.

    Parameters
    ----------
    image_energies : wp.array, shape (num_images,), dtype wp.float32 or wp.float64
        Physical energy of each packed image.
    path_ptr : wp.array, shape (num_paths + 1,), dtype wp.int32
        CSR-style offsets delimiting paths in ``image_energies``. Every path
        must contain at least three images.

    Modifies
    --------
    endpoint_reference_energy : wp.array, shape (num_paths,)
        Output for the smaller endpoint energy of each path.
    highest_interior_energy : wp.array, shape (num_paths,)
        Output for the maximum interior-image energy of each path.
    highest_interior_image_idx : wp.array, shape (num_paths,), dtype wp.int32
        Output for the lowest packed image index attaining the maximum
        interior energy.
    """
    path_idx, lane = wp.tid()
    image_start = path_ptr[path_idx]
    image_stop = path_ptr[path_idx + 1]

    local_endpoint_min = type(image_energies[0])(wp.inf)
    local_highest_energy = type(image_energies[0])(-wp.inf)
    local_highest_idx = image_stop
    local_has_interior = wp.bool(False)

    image_idx = image_start + lane
    # Scan in block-width strides to support paths longer than one block.
    while image_idx < image_stop:
        energy = image_energies[image_idx]
        if image_idx == image_start or image_idx == image_stop - wp.int32(1):
            if energy < local_endpoint_min:
                local_endpoint_min = energy
        elif not local_has_interior or energy > local_highest_energy:
            local_highest_energy = energy
            local_highest_idx = image_idx
            local_has_interior = wp.bool(True)
        image_idx = image_idx + wp.block_dim()

    endpoint_min = wp.tile_min(wp.tile(local_endpoint_min))[0]
    highest_energy = wp.tile_max(wp.tile(local_highest_energy))[0]

    # Select the lowest image index among lanes tied for the maximum energy.
    candidate_idx = image_stop
    if local_has_interior and local_highest_energy == highest_energy:
        candidate_idx = local_highest_idx
    # In the case of a tie, tile_min selects the smallest image index.
    highest_idx = wp.tile_min(wp.tile(candidate_idx))[0]

    if lane == 0:
        endpoint_reference_energy[path_idx] = endpoint_min
        highest_interior_energy[path_idx] = highest_energy
        highest_interior_image_idx[path_idx] = highest_idx


@cache
def _get_path_energy_stats_overloads() -> dict[object, wp.Kernel]:
    """Return cached scalar-dtype overloads for path-energy statistics."""

    def build_arg_types(dtype: object) -> list[object]:
        return [
            wp.array(dtype=dtype),
            wp.array(dtype=wp.int32),
            wp.array(dtype=dtype),
            wp.array(dtype=dtype),
            wp.array(dtype=wp.int32),
        ]

    return register_overloads(
        _path_energy_stats_kernel,
        build_arg_types,
        dtypes=_SCALAR_DTYPES,
    )
