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

"""PyTorch boundary for the fused Warp batched NEB force operation.

Execution flow
--------------
::

    neb_forces
      -> stored_tangent_neb_forces | gram_stats_neb_forces
         -> validate inputs and allocate missing output buffers
         -> *_neb_forces_out_op             (PyTorch custom-op boundary)
            -> _launch_neb_from_torch       (Torch-to-Warp conversion)
               -> launch_neb_forces_kernel  (device and dtype selection)
                  -> wp.launch               (CPU scalar kernel)
                  -> wp.launch_tiled         (CUDA tiled kernel)

During FakeTensor tracing, PyTorch dispatches ``*_out_op`` to its registered no-op fake
implementation instead of executing the real implementation, so no Warp kernel is launched.
"""

from __future__ import annotations

import torch
import warp as wp
from nvalchemiops.torch._warp_op_helpers import (
    register_noop_fake,
    scoped_warp_stream,
    torch_custom_op,
)

from .launchers import _neb_kernel_arg_types, launch_neb_forces_kernel
from .registry import _GramStatsMethod, _StoredTangentMethod, get_neb_method

__all__ = [
    "neb_forces",
]

_TORCH_TO_WP_SCALAR = {torch.float32: wp.float32, torch.float64: wp.float64}


# =============================================================================
# Input and output validation
# =============================================================================


def _validate_neb_inputs(x: tuple[torch.Tensor, ...]) -> tuple[int, int]:
    """Validate shape/dtype/device/layout of packed NEB inputs and return (num_atoms, num_links)."""
    (
        pos,
        forces,
        energies,
        image_ptr,
        path_ptr,
        image_path_idx,
        springs,
        fixed,
        modes,
        energy_ref,
        energy_max,
        cell,
        inv_cell,
        pbc,
    ) = x
    if pos.ndim != 2 or pos.shape[1] != 3:
        raise ValueError("positions must have shape (num_atoms, 3)")
    if pos.dtype not in _TORCH_TO_WP_SCALAR:
        raise ValueError("positions must have dtype float32 or float64")
    if pos.device.type not in {"cpu", "cuda"}:
        raise ValueError("NEB Torch adapters require CPU or CUDA tensors")
    if not pos.is_contiguous() or image_ptr.ndim != 1 or path_ptr.ndim != 1:
        raise ValueError(
            "positions must be contiguous and pointer tensors one-dimensional"
        )
    a, i, p = pos.shape[0], image_ptr.shape[0] - 1, path_ptr.shape[0] - 1
    num_links = i - p
    if min(i, p, num_links) < 0:
        raise ValueError("image_ptr and path_ptr describe invalid batch sizes")
    floats = {
        "physical_forces": (forces, (a, 3)),
        "image_energies": (energies, (i,)),
        "spring_constants": (springs, (num_links,)),
        "path_energy_ref": (energy_ref, (p,)),
        "path_energy_max": (energy_max, (p,)),
        "cell": (cell, (p, 3, 3)),
        "inv_cell": (inv_cell, (p, 3, 3)),
    }
    for name, (tensor, shape) in floats.items():
        if tensor.shape != shape:
            raise ValueError(f"{name} must have shape {shape}")
        if (
            tensor.dtype != pos.dtype
            or tensor.device != pos.device
            or not tensor.is_contiguous()
        ):
            raise ValueError(f"{name} must match positions in dtype/device/layout")
    ints = {
        "image_ptr": (image_ptr, (i + 1,)),
        "path_ptr": (path_ptr, (p + 1,)),
        "image_path_idx": (image_path_idx, (i,)),
        "image_force_mode": (modes, (i,)),
    }
    for name, (tensor, shape) in ints.items():
        if (
            tensor.shape != shape
            or tensor.dtype != torch.int32
            or tensor.device != pos.device
            or not tensor.is_contiguous()
        ):
            raise ValueError(f"{name} must be a matching contiguous int32 tensor")
    if (
        fixed.shape != (a,)
        or fixed.dtype != torch.bool
        or fixed.device != pos.device
        or not fixed.is_contiguous()
    ):
        raise ValueError("fixed_atom_mask must be a matching contiguous bool tensor")
    if (
        pbc.shape != (p, 3)
        or pbc.dtype != torch.bool
        or pbc.device != pos.device
        or not pbc.is_contiguous()
    ):
        raise ValueError(
            "pbc must be a matching contiguous bool tensor of shape (num_paths, 3)"
        )
    return a, num_links


def _validate_neb_outputs(pos, a, num_links, forces, links, tangent=None):
    """Validate shape/dtype/device/layout of caller-supplied NEB output tensors."""
    outputs = [
        ("effective_forces", forces, (a, 3)),
        ("link_lengths", links, (num_links,)),
    ]
    if tangent is not None:
        outputs.append(("tangent_buffer", tangent, (a, 3)))
    for name, tensor, shape in outputs:
        if tensor.shape != shape:
            raise ValueError(f"{name} must have shape {shape}")
        if (
            tensor.dtype != pos.dtype
            or tensor.device != pos.device
            or not tensor.is_contiguous()
        ):
            raise ValueError(f"{name} must match positions in dtype/device/layout")


# =============================================================================
# Torch-to-Warp dispatch
# =============================================================================


def _launch_neb_from_torch(
    method: str,
    x,
    forces,
    links,
    tangent=None,
):
    """Convert Torch tensors and dispatch to the launcher."""
    pos = x[0]
    scalar = _TORCH_TO_WP_SCALAR[pos.dtype]
    spec = get_neb_method(method)
    if isinstance(spec, _StoredTangentMethod):
        stored = True
    elif isinstance(spec, _GramStatsMethod):
        stored = False
    else:
        raise RuntimeError(
            f"NEB method {method!r} has unsupported specification {type(spec).__name__}"
        )
    if stored and tangent is None:
        raise RuntimeError("stored-tangent kernels require tangent scratch")
    if not stored and tangent is not None:
        raise RuntimeError("Gram-statistics kernels do not accept tangent scratch")

    # Prepare arguments for kernel
    tensors = [*x]
    if tangent is not None:
        tensors.append(tangent)
    tensors.extend((forces, links))
    arg_types = _neb_kernel_arg_types(scalar, stored)
    args = [
        wp.from_torch(tensor, dtype=arg_type.dtype)
        for tensor, arg_type in zip(tensors, arg_types, strict=True)
    ]

    # Ensure Warp launches kernel on the same CUDA stream that PyTorch uses
    with scoped_warp_stream(pos.device):
        launch_neb_forces_kernel(
            method,
            args,
            scalar_dtype=scalar,
            num_images=x[2].shape[0],
        )


@torch_custom_op(
    "nvalchemiops::stored_tangent_neb_forces_out",
    mutates_args=("tangent", "forces_out", "links_out"),
)
def _stored_tangent_neb_forces_out_op(
    method: str,
    positions: torch.Tensor,
    physical_forces: torch.Tensor,
    image_energies: torch.Tensor,
    image_ptr: torch.Tensor,
    path_ptr: torch.Tensor,
    image_path_idx: torch.Tensor,
    spring_constants: torch.Tensor,
    fixed_atom_mask: torch.Tensor,
    image_force_mode: torch.Tensor,
    path_energy_ref: torch.Tensor,
    path_energy_max: torch.Tensor,
    cell: torch.Tensor,
    inv_cell: torch.Tensor,
    pbc: torch.Tensor,
    tangent: torch.Tensor,
    forces_out: torch.Tensor,
    links_out: torch.Tensor,
) -> None:
    """Custom op: launch the stored-tangent NEB kernel into caller-provided buffers."""
    _launch_neb_from_torch(
        method,
        (
            positions,
            physical_forces,
            image_energies,
            image_ptr,
            path_ptr,
            image_path_idx,
            spring_constants,
            fixed_atom_mask,
            image_force_mode,
            path_energy_ref,
            path_energy_max,
            cell,
            inv_cell,
            pbc,
        ),
        forces_out,
        links_out,
        tangent,
    )


@torch_custom_op(
    "nvalchemiops::gram_stats_neb_forces_out", mutates_args=("forces_out", "links_out")
)
def _gram_stats_neb_forces_out_op(
    method: str,
    positions: torch.Tensor,
    physical_forces: torch.Tensor,
    image_energies: torch.Tensor,
    image_ptr: torch.Tensor,
    path_ptr: torch.Tensor,
    image_path_idx: torch.Tensor,
    spring_constants: torch.Tensor,
    fixed_atom_mask: torch.Tensor,
    image_force_mode: torch.Tensor,
    path_energy_ref: torch.Tensor,
    path_energy_max: torch.Tensor,
    cell: torch.Tensor,
    inv_cell: torch.Tensor,
    pbc: torch.Tensor,
    forces_out: torch.Tensor,
    links_out: torch.Tensor,
) -> None:
    """Custom op: launch the Gram-stats NEB kernel into caller-provided buffers."""
    _launch_neb_from_torch(
        method,
        (
            positions,
            physical_forces,
            image_energies,
            image_ptr,
            path_ptr,
            image_path_idx,
            spring_constants,
            fixed_atom_mask,
            image_force_mode,
            path_energy_ref,
            path_energy_max,
            cell,
            inv_cell,
            pbc,
        ),
        forces_out,
        links_out,
    )


# Register no-op FakeTensor implementations. The custom ops mutate preallocated
# buffers and return None, so tracing does not need to create fake outputs.
register_noop_fake(_stored_tangent_neb_forces_out_op)
register_noop_fake(_gram_stats_neb_forces_out_op)


# =============================================================================
# Public Torch adapters
# =============================================================================


def stored_tangent_neb_forces(
    positions: torch.Tensor,
    physical_forces: torch.Tensor,
    image_energies: torch.Tensor,
    image_ptr: torch.Tensor,
    path_ptr: torch.Tensor,
    image_path_idx: torch.Tensor,
    spring_constants: torch.Tensor,
    fixed_atom_mask: torch.Tensor,
    image_force_mode: torch.Tensor,
    path_energy_ref: torch.Tensor,
    path_energy_max: torch.Tensor,
    cell: torch.Tensor,
    inv_cell: torch.Tensor,
    pbc: torch.Tensor,
    *,
    method: str = "improved_tangent",
    tangent_buffer: torch.Tensor | None = None,
    effective_forces: torch.Tensor | None = None,
    link_lengths: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute local NEB forces using stored tangent scratch.

    Parameters
    ----------
    positions, physical_forces : torch.Tensor, shape (num_atoms, 3)
        Packed atomic positions and physical forces.
    image_energies : torch.Tensor, shape (num_images,)
        Physical energy of every image.
    image_ptr : torch.Tensor, shape (num_images + 1,), dtype int32
        CSR offsets mapping images to packed atoms.
    path_ptr : torch.Tensor, shape (num_paths + 1,), dtype int32
        CSR offsets mapping paths to packed images.
    image_path_idx : torch.Tensor, shape (num_images,), dtype int32
        Path index of every image.
    spring_constants : torch.Tensor, shape (num_images - num_paths,)
        Effective spring constant of every path link.
    fixed_atom_mask : torch.Tensor, shape (num_atoms,), dtype bool
        Whether each atom's effective force is constrained to zero.
    image_force_mode : torch.Tensor, shape (num_images,), dtype int32
        Effective-force mode of every image.
    path_energy_ref, path_energy_max : torch.Tensor, shape (num_paths,)
        Reference and maximum interior-image energies of every path.
    cell, inv_cell : torch.Tensor, shape (num_paths, 3, 3)
        Cell matrices and their inverses.
    pbc : torch.Tensor, shape (num_paths, 3), dtype bool
        Periodic boundary flags of every path.
    method : str, optional
        User-facing NEB formulation identifier.
    tangent_buffer : torch.Tensor, shape (num_atoms, 3), optional
        Reusable tangent scratch. Allocated when omitted.
    effective_forces : torch.Tensor, shape (num_atoms, 3), optional
        Reusable effective-force output. Allocated when omitted.
    link_lengths : torch.Tensor, shape (num_images - num_paths,), optional
        Reusable link-length output. Allocated when omitted.

    Returns
    -------
    effective_forces : torch.Tensor, shape (num_atoms, 3)
        Effective optimizer-facing NEB forces.
    link_lengths : torch.Tensor, shape (num_images - num_paths,)
        Minimum-image link lengths in path-major order.
    """
    x = (
        positions,
        physical_forces,
        image_energies,
        image_ptr,
        path_ptr,
        image_path_idx,
        spring_constants,
        fixed_atom_mask,
        image_force_mode,
        path_energy_ref,
        path_energy_max,
        cell,
        inv_cell,
        pbc,
    )
    num_atoms, num_links = _validate_neb_inputs(x)
    tangent_buffer = (
        torch.empty_like(positions) if tangent_buffer is None else tangent_buffer
    )
    effective_forces = (
        torch.empty_like(positions) if effective_forces is None else effective_forces
    )
    link_lengths = (
        torch.empty_like(spring_constants) if link_lengths is None else link_lengths
    )
    _validate_neb_outputs(
        positions,
        num_atoms,
        num_links,
        effective_forces,
        link_lengths,
        tangent_buffer,
    )
    _stored_tangent_neb_forces_out_op(
        method,
        *x,
        tangent_buffer,
        effective_forces,
        link_lengths,
    )
    return effective_forces, link_lengths


def gram_stats_neb_forces(
    positions: torch.Tensor,
    physical_forces: torch.Tensor,
    image_energies: torch.Tensor,
    image_ptr: torch.Tensor,
    path_ptr: torch.Tensor,
    image_path_idx: torch.Tensor,
    spring_constants: torch.Tensor,
    fixed_atom_mask: torch.Tensor,
    image_force_mode: torch.Tensor,
    path_energy_ref: torch.Tensor,
    path_energy_max: torch.Tensor,
    cell: torch.Tensor,
    inv_cell: torch.Tensor,
    pbc: torch.Tensor,
    *,
    method: str = "improved_tangent",
    effective_forces: torch.Tensor | None = None,
    link_lengths: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute local NEB forces from Gram statistics.

    Parameters
    ----------
    positions, physical_forces : torch.Tensor, shape (num_atoms, 3)
        Packed atomic positions and physical forces.
    image_energies : torch.Tensor, shape (num_images,)
        Physical energy of every image.
    image_ptr : torch.Tensor, shape (num_images + 1,), dtype int32
        CSR offsets mapping images to packed atoms.
    path_ptr : torch.Tensor, shape (num_paths + 1,), dtype int32
        CSR offsets mapping paths to packed images.
    image_path_idx : torch.Tensor, shape (num_images,), dtype int32
        Path index of every image.
    spring_constants : torch.Tensor, shape (num_images - num_paths,)
        Effective spring constant of every path link.
    fixed_atom_mask : torch.Tensor, shape (num_atoms,), dtype bool
        Whether each atom's effective force is constrained to zero.
    image_force_mode : torch.Tensor, shape (num_images,), dtype int32
        Effective-force mode of every image.
    path_energy_ref, path_energy_max : torch.Tensor, shape (num_paths,)
        Reference and maximum interior-image energies of every path.
    cell, inv_cell : torch.Tensor, shape (num_paths, 3, 3)
        Cell matrices and their inverses.
    pbc : torch.Tensor, shape (num_paths, 3), dtype bool
        Periodic boundary flags of every path.
    method : str, optional
        User-facing NEB formulation identifier.
    effective_forces : torch.Tensor, shape (num_atoms, 3), optional
        Reusable effective-force output. Allocated when omitted.
    link_lengths : torch.Tensor, shape (num_images - num_paths,), optional
        Reusable link-length output. Allocated when omitted.

    Returns
    -------
    effective_forces : torch.Tensor, shape (num_atoms, 3)
        Effective optimizer-facing NEB forces.
    link_lengths : torch.Tensor, shape (num_images - num_paths,)
        Minimum-image link lengths in path-major order.
    """
    x = (
        positions,
        physical_forces,
        image_energies,
        image_ptr,
        path_ptr,
        image_path_idx,
        spring_constants,
        fixed_atom_mask,
        image_force_mode,
        path_energy_ref,
        path_energy_max,
        cell,
        inv_cell,
        pbc,
    )
    num_atoms, num_links = _validate_neb_inputs(x)
    effective_forces = (
        torch.empty_like(positions) if effective_forces is None else effective_forces
    )
    link_lengths = (
        torch.empty_like(spring_constants) if link_lengths is None else link_lengths
    )
    _validate_neb_outputs(
        positions, num_atoms, num_links, effective_forces, link_lengths
    )
    _gram_stats_neb_forces_out_op(
        method,
        *x,
        effective_forces,
        link_lengths,
    )
    return effective_forces, link_lengths


def neb_forces(
    positions: torch.Tensor,
    physical_forces: torch.Tensor,
    image_energies: torch.Tensor,
    image_ptr: torch.Tensor,
    path_ptr: torch.Tensor,
    image_path_idx: torch.Tensor,
    spring_constants: torch.Tensor,
    fixed_atom_mask: torch.Tensor,
    image_force_mode: torch.Tensor,
    path_energy_ref: torch.Tensor,
    path_energy_max: torch.Tensor,
    cell: torch.Tensor,
    inv_cell: torch.Tensor,
    pbc: torch.Tensor,
    *,
    method: str = "improved_tangent",
    tangent_buffer: torch.Tensor | None = None,
    effective_forces: torch.Tensor | None = None,
    link_lengths: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute effective NEB forces, dispatching to the strategy for `method`
    registered in the NEB method registry.

    Parameters
    ----------
    positions, physical_forces : torch.Tensor, shape (num_atoms, 3)
        Packed atomic positions and physical forces.
    image_energies : torch.Tensor, shape (num_images,)
        Physical energy of every image.
    image_ptr : torch.Tensor, shape (num_images + 1,), dtype int32
        CSR offsets mapping images to packed atoms.
    path_ptr : torch.Tensor, shape (num_paths + 1,), dtype int32
        CSR offsets mapping paths to packed images.
    image_path_idx : torch.Tensor, shape (num_images,), dtype int32
        Path index of every image.
    spring_constants : torch.Tensor, shape (num_images - num_paths,)
        Effective spring constant of every path link.
    fixed_atom_mask : torch.Tensor, shape (num_atoms,), dtype bool
        Whether each atom's effective force is constrained to zero.
    image_force_mode : torch.Tensor, shape (num_images,), dtype int32
        Effective-force mode of every image.
    path_energy_ref, path_energy_max : torch.Tensor, shape (num_paths,)
        Reference and maximum interior-image energies of every path.
    cell, inv_cell : torch.Tensor, shape (num_paths, 3, 3)
        Cell matrices and their inverses.
    pbc : torch.Tensor, shape (num_paths, 3), dtype bool
        Periodic boundary flags of every path.
    method : str, optional
        User-facing NEB formulation identifier; selects the underlying
        storage strategy (stored-tangent or Gram-stats).
    tangent_buffer : torch.Tensor, shape (num_atoms, 3), optional
        Reusable tangent scratch, only used by stored-tangent methods.
        Allocated when omitted.
    effective_forces : torch.Tensor, shape (num_atoms, 3), optional
        Reusable effective-force output. Allocated when omitted.
    link_lengths : torch.Tensor, shape (num_images - num_paths,), optional
        Reusable link-length output. Allocated when omitted.

    Returns
    -------
    effective_forces : torch.Tensor, shape (num_atoms, 3)
        Effective optimizer-facing NEB forces.
    link_lengths : torch.Tensor, shape (num_images - num_paths,)
        Minimum-image link lengths in path-major order.
    """
    spec = get_neb_method(method)
    common_args = (
        positions,
        physical_forces,
        image_energies,
        image_ptr,
        path_ptr,
        image_path_idx,
        spring_constants,
        fixed_atom_mask,
        image_force_mode,
        path_energy_ref,
        path_energy_max,
        cell,
        inv_cell,
        pbc,
    )
    if isinstance(spec, _StoredTangentMethod):
        return stored_tangent_neb_forces(
            *common_args,
            method=method,
            tangent_buffer=tangent_buffer,
            effective_forces=effective_forces,
            link_lengths=link_lengths,
        )
    if isinstance(spec, _GramStatsMethod):
        return gram_stats_neb_forces(
            *common_args,
            method=method,
            effective_forces=effective_forces,
            link_lengths=link_lengths,
        )
    raise RuntimeError(
        f"NEB method {method!r} has unsupported specification {type(spec).__name__}"
    )
