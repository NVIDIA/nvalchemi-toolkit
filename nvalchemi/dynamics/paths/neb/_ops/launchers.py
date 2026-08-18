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

"""Framework-neutral Warp dispatch and launch for batched NEB forces."""

from __future__ import annotations

from functools import cache

import warp as wp
from nvalchemiops.warp_dispatch import register_overloads

from .kernels import (
    build_scalar_gram_stats_neb_kernel,
    build_scalar_stored_tangent_neb_kernel,
    build_tiled_gram_stats_neb_kernel,
    build_tiled_stored_tangent_neb_kernel,
)
from .registry import _GramStatsMethod, _StoredTangentMethod, get_neb_method

__all__ = ["launch_neb_forces_kernel"]

_NEB_TILE_DIM = 128
_NEB_DTYPES = (wp.float32, wp.float64)


def _neb_kernel_arg_types(dtype: object, stored: bool) -> list[object]:
    """Build the ordered Warp argument types for an NEB kernel overload.

    Supports the stored-tangent and Gram-statistics kernel families. Both
    layouts produce effective-force and link-length outputs. Stored-tangent
    kernels additionally include a per-atom tangent scratch buffer immediately
    before those outputs.
    """
    vec = wp.vec3f if dtype == wp.float32 else wp.vec3d
    mat = wp.mat33f if dtype == wp.float32 else wp.mat33d
    types = [
        wp.array(dtype=vec),
        wp.array(dtype=vec),
        wp.array(dtype=dtype),
        wp.array(dtype=wp.int32),
        wp.array(dtype=wp.int32),
        wp.array(dtype=wp.int32),
        wp.array(dtype=dtype),
        wp.array(dtype=wp.bool),
        wp.array(dtype=wp.int32),
        wp.array(dtype=dtype),
        wp.array(dtype=dtype),
        wp.array(dtype=mat),
        wp.array(dtype=mat),
        wp.array(dtype=wp.vec3b),
    ]
    if stored:
        types.append(wp.array(dtype=vec))
    return types + [wp.array(dtype=vec), wp.array(dtype=dtype)]


@cache
def _get_neb_forces_kernel_overloads(
    method: str,
    device_type: str,
) -> dict[object, wp.Kernel]:
    """Return cached dtype overloads for an NEB method and device."""
    spec = get_neb_method(method)
    if isinstance(spec, _StoredTangentMethod):
        stored_tangent = True
        scalar_builder = build_scalar_stored_tangent_neb_kernel
        tiled_builder = build_tiled_stored_tangent_neb_kernel
    elif isinstance(spec, _GramStatsMethod):
        stored_tangent = False
        scalar_builder = build_scalar_gram_stats_neb_kernel
        tiled_builder = build_tiled_gram_stats_neb_kernel
    else:
        raise RuntimeError(
            f"NEB method {method!r} has unsupported specification {type(spec).__name__}"
        )

    if device_type == "cpu":
        builder = scalar_builder
    elif device_type == "cuda":
        builder = tiled_builder
    else:
        raise ValueError(f"unsupported NEB device type {device_type!r}")
    kernel = builder(
        spec.tangent_fn,
        spec.force_fn,
        spec.climbing_force_fn,
    )

    def build_arg_types(dtype: object) -> list[object]:
        return _neb_kernel_arg_types(dtype, stored_tangent)

    return register_overloads(
        kernel,
        build_arg_types,
        dtypes=_NEB_DTYPES,
    )


def launch_neb_forces_kernel(
    method: str,
    args: list[object],
    *,
    scalar_dtype: object,
    num_images: int,
) -> None:
    """Launch the fused NEB force kernel on a CPU or CUDA device.

    The kernel implementation and dtype overload are selected from the registered NEB method
    and the device associated with ``args``.

    Parameters
    ----------
    method : str
        Registered NEB method name used to select the kernel implementation.
    args : list[object]
        Ordered, already-converted Warp kernel arguments.
    scalar_dtype : object
        Warp scalar dtype, either ``wp.float32`` or ``wp.float64``, used to select the
        registered kernel overload.
    num_images : int
        Number of images in the launch grid.

    Raises
    ------
    ValueError
        If the method or Warp device is unsupported.
    """
    get_neb_method(method)
    if num_images == 0:
        return

    # Launch kernel with the correct device type and dtype
    device = args[0].device
    if device.is_cpu:
        kernel = _get_neb_forces_kernel_overloads(method, "cpu")[scalar_dtype]
        wp.launch(kernel, dim=num_images, inputs=args, device=device)
        return
    if device.is_cuda:
        kernel = _get_neb_forces_kernel_overloads(method, "cuda")[scalar_dtype]
        wp.launch_tiled(
            kernel,
            dim=num_images,
            inputs=args,
            block_dim=_NEB_TILE_DIM,
            device=device,
        )
        return
    raise ValueError(f"unsupported NEB Warp device {device}")
