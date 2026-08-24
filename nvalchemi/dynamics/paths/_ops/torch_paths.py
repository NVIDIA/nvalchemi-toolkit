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
"""PyTorch boundary for per-path energy reductions."""

from __future__ import annotations

import torch
import warp as wp
from nvalchemiops.torch._warp_op_helpers import (
    register_noop_fake,
    scoped_warp_stream,
    torch_custom_op,
)

from .paths import _get_path_energy_stats_overloads

__all__ = ["path_energy_stats"]

_TORCH_TO_WP_SCALAR = {torch.float32: wp.float32, torch.float64: wp.float64}
_PATH_ENERGY_TILE_DIM = 32


@torch_custom_op(
    "nvalchemiops::path_energy_stats_out",
    mutates_args=(
        "endpoint_reference_energy",
        "highest_interior_energy",
        "highest_interior_image_idx",
    ),
)
def _path_energy_stats_out_op(
    image_energies: torch.Tensor,
    path_ptr: torch.Tensor,
    endpoint_reference_energy: torch.Tensor,
    highest_interior_energy: torch.Tensor,
    highest_interior_image_idx: torch.Tensor,
) -> None:
    """Launch path-energy reductions into caller-provided output buffers."""
    scalar = _TORCH_TO_WP_SCALAR[image_energies.dtype]
    args = [
        wp.from_torch(image_energies, dtype=scalar),
        wp.from_torch(path_ptr, dtype=wp.int32),
        wp.from_torch(endpoint_reference_energy, dtype=scalar),
        wp.from_torch(highest_interior_energy, dtype=scalar),
        wp.from_torch(highest_interior_image_idx, dtype=wp.int32),
    ]
    num_paths = path_ptr.shape[0] - 1
    if num_paths == 0:
        return
    device = args[0].device
    kernel = _get_path_energy_stats_overloads()[scalar]
    with scoped_warp_stream(image_energies.device):
        wp.launch_tiled(
            kernel,
            dim=num_paths,
            inputs=args,
            block_dim=_PATH_ENERGY_TILE_DIM if device.is_cuda else 1,
            device=device,
        )


register_noop_fake(_path_energy_stats_out_op)


def path_energy_stats(
    image_energies: torch.Tensor,
    path_ptr: torch.Tensor,
    endpoint_reference_energy: torch.Tensor,
    highest_interior_energy: torch.Tensor,
    highest_interior_image_idx: torch.Tensor,
) -> None:
    """Compute per-path endpoint and highest-interior energy statistics in place.

    Parameters
    ----------
    image_energies : torch.Tensor, shape (num_images,)
        Packed scalar image energies in path-major order.
    path_ptr : torch.Tensor, shape (num_paths + 1,), dtype int32
        CSR-style offsets delimiting the image range of every path. Each path
        must contain at least three images.
    endpoint_reference_energy : torch.Tensor, shape (num_paths,)
        Output buffer receiving the smaller endpoint energy of every path.
    highest_interior_energy : torch.Tensor, shape (num_paths,)
        Output buffer receiving the maximum interior-image energy of every path.
    highest_interior_image_idx : torch.Tensor, shape (num_paths,), dtype int32
        Output buffer receiving the packed index of the first interior image
        attaining ``highest_interior_energy``.
    """
    if image_energies.ndim != 1 or path_ptr.ndim != 1:
        raise ValueError("image_energies and path_ptr must be one-dimensional")
    if image_energies.dtype not in _TORCH_TO_WP_SCALAR:
        raise ValueError("image_energies must have dtype float32 or float64")
    if path_ptr.dtype != torch.int32:
        raise ValueError("path_ptr must have dtype int32")
    if image_energies.device.type not in {"cpu", "cuda"}:
        raise ValueError("path energy statistics require CPU or CUDA tensors")
    if not image_energies.is_contiguous() or not path_ptr.is_contiguous():
        raise ValueError("image_energies and path_ptr must be contiguous")

    num_paths = path_ptr.shape[0] - 1
    float_outputs = {
        "endpoint_reference_energy": endpoint_reference_energy,
        "highest_interior_energy": highest_interior_energy,
    }
    for name, output in float_outputs.items():
        if output.shape != (num_paths,):
            raise ValueError(f"{name} must have shape ({num_paths},)")
        if (
            output.dtype != image_energies.dtype
            or output.device != image_energies.device
            or not output.is_contiguous()
        ):
            raise ValueError(
                f"{name} must match image_energies in dtype, device, and layout"
            )
    if highest_interior_image_idx.shape != (num_paths,):
        raise ValueError(f"highest_interior_image_idx must have shape ({num_paths},)")
    if (
        highest_interior_image_idx.dtype != torch.int32
        or highest_interior_image_idx.device != image_energies.device
        or not highest_interior_image_idx.is_contiguous()
    ):
        raise ValueError(
            "highest_interior_image_idx must be contiguous int32 on the "
            "image_energies device"
        )

    _path_energy_stats_out_op(
        image_energies,
        path_ptr,
        endpoint_reference_energy,
        highest_interior_energy,
        highest_interior_image_idx,
    )
