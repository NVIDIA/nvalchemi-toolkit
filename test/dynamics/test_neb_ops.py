# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for generated NEB Torch adapters."""

from __future__ import annotations

import pytest
import torch
import warp as wp
from nvalchemiops.torch._warp_op_helpers import scoped_warp_stream

from nvalchemi.dynamics.paths.neb._ops.equations import (
    neb_effective_force_from_gram_stats,
)
from nvalchemi.dynamics.paths.neb._ops.launchers import (
    _NEB_TILE_DIM,
    _get_neb_forces_kernel_overloads,
    _neb_kernel_arg_types,
)
from nvalchemi.dynamics.paths.neb._ops.modes import (
    CLIMBING_NEB,
    FIXED_ENDPOINT,
    REGULAR_NEB,
)
from nvalchemi.dynamics.paths.neb._ops.registry import (
    _GramStatsMethod,
    _StoredTangentMethod,
    available_neb_methods,
    get_neb_method,
    register_neb_method,
)
from nvalchemi.dynamics.paths.neb._ops.torch_ops import neb_forces

# =============================================================================
# Fixtures and launch helpers
# =============================================================================


def _inputs(device: str = "cuda", dtype: torch.dtype = torch.float64):
    positions = torch.tensor(
        [[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1.1, 0], [2.2, 0, 0], [2.2, 1.2, 0]],
        device=device,
        dtype=dtype,
    )
    cell = (10 * torch.eye(3, device=device, dtype=dtype)[None]).contiguous()
    return (
        positions,
        torch.arange(positions.numel(), device=device, dtype=dtype).reshape_as(
            positions
        )
        / 10
        - 0.8,
        torch.tensor([0, 1, 0.2], device=device, dtype=dtype),
        torch.tensor([0, 2, 4, 6], device=device, dtype=torch.int32),
        torch.tensor([0, 3], device=device, dtype=torch.int32),
        torch.zeros(3, device=device, dtype=torch.int32),
        torch.tensor([0.1, 0.15], device=device, dtype=dtype),
        torch.zeros(6, device=device, dtype=torch.bool),
        torch.tensor([0, 2, 0], device=device, dtype=torch.int32),
        torch.tensor([0], device=device, dtype=dtype),
        torch.tensor([1], device=device, dtype=dtype),
        cell,
        torch.linalg.inv(cell).contiguous(),
        torch.zeros((1, 3), device=device, dtype=torch.bool),
    )


def _launch_neb_kernel_on_cuda(
    method: str,
    inputs: tuple[torch.Tensor, ...],
    *,
    tiled: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Launch one NEB kernel variant directly against CUDA Torch storage."""
    scalar_dtype = wp.float32 if inputs[0].dtype == torch.float32 else wp.float64
    stored_tangent = isinstance(get_neb_method(method), _StoredTangentMethod)
    tangent = torch.empty_like(inputs[0]) if stored_tangent else None
    effective_forces = torch.empty_like(inputs[0])
    link_lengths = torch.empty_like(inputs[6])
    tensors = [*inputs]
    if tangent is not None:
        tensors.append(tangent)
    tensors.extend((effective_forces, link_lengths))
    arg_types = _neb_kernel_arg_types(scalar_dtype, stored_tangent)
    args = [
        wp.from_torch(tensor, dtype=arg_type.dtype)
        for tensor, arg_type in zip(tensors, arg_types, strict=True)
    ]
    variant = "cuda" if tiled else "cpu"
    kernel = _get_neb_forces_kernel_overloads(method, variant)[scalar_dtype]

    with scoped_warp_stream(inputs[0].device):
        if tiled:
            wp.launch_tiled(
                kernel,
                dim=inputs[2].shape[0],
                inputs=args,
                block_dim=_NEB_TILE_DIM,
                device=args[0].device,
            )
        else:
            wp.launch(
                kernel,
                dim=inputs[2].shape[0],
                inputs=args,
                device=args[0].device,
            )

    return effective_forces, link_lengths


# =============================================================================
# Pure-Torch reference implementation
# =============================================================================


def _minimum_image_displacement(
    r_a: torch.Tensor,
    r_b: torch.Tensor,
    cell: torch.Tensor,
    inv_cell: torch.Tensor,
    pbc: torch.Tensor,
) -> torch.Tensor:
    """Return minimum-image displacements using only Torch operations."""
    fractional = (r_a - r_b) @ inv_cell
    wrapped = fractional - torch.floor(fractional + 0.5)
    return torch.where(pbc, wrapped, fractional) @ cell


def _naive_improved_tangent_neb_forces(
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
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute improved-tangent NEB forces with a readable Torch oracle."""
    del image_path_idx, path_energy_ref, path_energy_max
    num_images = image_energies.shape[0]
    num_paths = path_ptr.shape[0] - 1
    effective_forces = torch.zeros_like(physical_forces)
    link_lengths = positions.new_empty(num_images - num_paths)
    displacements: dict[tuple[int, int], torch.Tensor] = {}

    for path_idx in range(num_paths):
        image_start = int(path_ptr[path_idx])
        image_stop = int(path_ptr[path_idx + 1])
        for image_idx in range(image_start, image_stop - 1):
            atom_start = int(image_ptr[image_idx])
            atom_stop = int(image_ptr[image_idx + 1])
            next_atom_start = int(image_ptr[image_idx + 1])
            displacement = _minimum_image_displacement(
                positions[next_atom_start : next_atom_start + atom_stop - atom_start],
                positions[atom_start:atom_stop],
                cell[path_idx],
                inv_cell[path_idx],
                pbc[path_idx],
            )
            displacements[path_idx, image_idx] = displacement
            link_lengths[image_idx - path_idx] = torch.linalg.vector_norm(displacement)

        for image_idx in range(image_start + 1, image_stop - 1):
            atom_start = int(image_ptr[image_idx])
            atom_stop = int(image_ptr[image_idx + 1])
            mode = int(image_force_mode[image_idx])
            if mode not in (REGULAR_NEB, CLIMBING_NEB):
                continue

            d_plus = displacements[path_idx, image_idx]
            d_minus = displacements[path_idx, image_idx - 1]
            energy_prev = image_energies[image_idx - 1]
            energy_curr = image_energies[image_idx]
            energy_next = image_energies[image_idx + 1]
            delta_plus = energy_next - energy_curr
            delta_minus = energy_curr - energy_prev

            if bool((delta_minus > 0) & (delta_plus > 0)):
                tangent = d_plus
            elif bool((delta_minus < 0) & (delta_plus < 0)):
                tangent = d_minus
            else:
                delta_max = torch.maximum(delta_plus.abs(), delta_minus.abs())
                delta_min = torch.minimum(delta_plus.abs(), delta_minus.abs())
                if bool(energy_next > energy_prev):
                    tangent = delta_max * d_plus + delta_min * d_minus
                else:
                    tangent = delta_min * d_plus + delta_max * d_minus

            tangent_norm = torch.linalg.vector_norm(tangent)
            if bool(tangent_norm == 0):
                continue
            unit_tangent = tangent / tangent_norm
            physical_force = physical_forces[atom_start:atom_stop]
            force_dot_tangent = torch.sum(physical_force * unit_tangent)
            if mode == CLIMBING_NEB:
                effective_forces[atom_start:atom_stop] = (
                    physical_force - 2 * force_dot_tangent * unit_tangent
                )
            else:
                forward_link = image_idx - path_idx
                spring_parallel = (
                    spring_constants[forward_link] * link_lengths[forward_link]
                    - spring_constants[forward_link - 1]
                    * link_lengths[forward_link - 1]
                )
                effective_forces[atom_start:atom_stop] = (
                    physical_force
                    + (spring_parallel - force_dot_tangent) * unit_tangent
                )

    return torch.where(
        fixed_atom_mask[:, None],
        torch.zeros_like(effective_forces),
        effective_forces,
    ), link_lengths


# =============================================================================
# Method registry
# =============================================================================


class TestNEBMethodRegistry:
    """Test NEB method registration and overload selection."""

    def test_static_method_table_contains_improved_tangent(self) -> None:
        assert "improved_tangent" in available_neb_methods()
        assert isinstance(get_neb_method("improved_tangent"), _StoredTangentMethod)

    def test_registers_user_method_as_gram_stats(self) -> None:
        stored_method = get_neb_method("improved_tangent")
        assert isinstance(stored_method, _StoredTangentMethod)

        name = "test_user_gram_stats"
        result = register_neb_method(
            name=name,
            tangent_fn=stored_method.tangent_fn,
            force_fn=neb_effective_force_from_gram_stats,
            climbing_force_fn=stored_method.climbing_force_fn,
        )

        assert result is None
        assert name in available_neb_methods()
        assert isinstance(get_neb_method(name), _GramStatsMethod)
        assert (
            register_neb_method(
                name=name,
                tangent_fn=stored_method.tangent_fn,
                force_fn=neb_effective_force_from_gram_stats,
                climbing_force_fn=stored_method.climbing_force_fn,
            )
            is None
        )

    def test_rejects_conflicting_method_name(self) -> None:
        stored_method = get_neb_method("improved_tangent")
        assert isinstance(stored_method, _StoredTangentMethod)

        with pytest.raises(ValueError, match="already registered"):
            register_neb_method(
                name="improved_tangent",
                tangent_fn=stored_method.tangent_fn,
                force_fn=neb_effective_force_from_gram_stats,
                climbing_force_fn=stored_method.climbing_force_fn,
            )

    @pytest.mark.parametrize(
        "device",
        [
            "cpu",
            pytest.param(
                "cuda",
                marks=pytest.mark.skipif(
                    not torch.cuda.is_available(), reason="CUDA is required"
                ),
            ),
        ],
    )
    def test_method_overloads_are_cached_by_device(self, device: str) -> None:
        first = _get_neb_forces_kernel_overloads("improved_tangent", device)
        second = _get_neb_forces_kernel_overloads("improved_tangent", device)
        assert first is second


# =============================================================================
# Torch adapter
# =============================================================================


class TestNEBTorchAdapter:
    """Test the user-facing Torch adapter behavior."""

    @pytest.mark.parametrize(
        "device",
        [
            "cpu",
            pytest.param(
                "cuda",
                marks=pytest.mark.skipif(
                    not torch.cuda.is_available(), reason="CUDA is required"
                ),
            ),
        ],
    )
    def test_neb_forces_reuses_outputs(self, device: str) -> None:
        inputs = _inputs(device=device)
        reference = neb_forces(*inputs)
        tangent = torch.empty_like(inputs[0])
        forces = torch.empty_like(inputs[0])
        links = torch.empty_like(inputs[6])

        actual = neb_forces(
            *inputs,
            tangent_buffer=tangent,
            effective_forces=forces,
            link_lengths=links,
        )

        assert actual == (forces, links)
        torch.testing.assert_close(actual, reference)


# =============================================================================
# Numerical correctness
# =============================================================================


class TestImprovedTangentNumerics:
    """Compare improved-tangent NEB kernels with independent Torch equations."""

    @pytest.mark.parametrize(
        "mode",
        [
            pytest.param(REGULAR_NEB, id="regular"),
            pytest.param(CLIMBING_NEB, id="climbing"),
        ],
    )
    @pytest.mark.parametrize(
        "dtype",
        [
            pytest.param(torch.float32, id="float32"),
            pytest.param(torch.float64, id="float64"),
        ],
    )
    @pytest.mark.parametrize(
        "device",
        [
            "cpu",
            pytest.param(
                "cuda",
                marks=pytest.mark.skipif(
                    not torch.cuda.is_available(), reason="CUDA is required"
                ),
            ),
        ],
    )
    def test_improved_tangent_neb_forces_match_naive_torch(
        self,
        mode: int,
        dtype: torch.dtype,
        device: str,
    ) -> None:
        """Warp improved-tangent and climbing forces match an independent oracle."""
        inputs = list(_inputs(device=device, dtype=dtype))
        inputs[8] = torch.tensor(
            [FIXED_ENDPOINT, mode, FIXED_ENDPOINT],
            dtype=torch.int32,
            device=device,
        )

        actual = neb_forces(*inputs)
        expected = _naive_improved_tangent_neb_forces(*inputs)

        tolerance = 2.0e-5 if dtype == torch.float32 else 1.0e-11
        torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)


# =============================================================================
# Kernel implementation parity
# =============================================================================


class TestNEBKernelParity:
    """Compare equivalent NEB kernel strategies and launch implementations."""

    @pytest.mark.parametrize(
        "dtype",
        [
            pytest.param(torch.float32, id="float32"),
            pytest.param(torch.float64, id="float64"),
        ],
    )
    @pytest.mark.parametrize(
        "device",
        [
            "cpu",
            pytest.param(
                "cuda",
                marks=pytest.mark.skipif(
                    not torch.cuda.is_available(), reason="CUDA is required"
                ),
            ),
        ],
    )
    def test_improved_tangent_kernel_strategies_match(
        self,
        dtype: torch.dtype,
        device: str,
    ) -> None:
        """Stored-tangent and Gram-statistics kernels produce the same outputs."""
        stored_method = get_neb_method("improved_tangent")
        assert isinstance(stored_method, _StoredTangentMethod)
        method = "improved_tangent_gram_stats"
        register_neb_method(
            name=method,
            tangent_fn=stored_method.tangent_fn,
            force_fn=neb_effective_force_from_gram_stats,
            climbing_force_fn=stored_method.climbing_force_fn,
        )
        inputs = _inputs(device=device, dtype=dtype)

        stored_tangent = neb_forces(*inputs, method="improved_tangent")
        gram_stats = neb_forces(*inputs, method=method)

        tolerance = 2.0e-5 if dtype == torch.float32 else 1.0e-11
        torch.testing.assert_close(
            stored_tangent,
            gram_stats,
            atol=tolerance,
            rtol=tolerance,
        )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
    @pytest.mark.parametrize(
        "use_gram_stats",
        [
            pytest.param(False, id="stored-tangent"),
            pytest.param(True, id="gram-stats"),
        ],
    )
    @pytest.mark.parametrize(
        "mode",
        [
            pytest.param(REGULAR_NEB, id="regular"),
            pytest.param(CLIMBING_NEB, id="climbing"),
        ],
    )
    @pytest.mark.parametrize(
        "dtype",
        [
            pytest.param(torch.float32, id="float32"),
            pytest.param(torch.float64, id="float64"),
        ],
    )
    def test_scalar_and_tiled_neb_kernels_match_on_cuda(
        self,
        use_gram_stats: bool,
        mode: int,
        dtype: torch.dtype,
    ) -> None:
        """Scalar and tiled kernels produce the same outputs on one CUDA device."""
        method = "improved_tangent"
        if use_gram_stats:
            stored_method = get_neb_method("improved_tangent")
            assert isinstance(stored_method, _StoredTangentMethod)
            method = "improved_tangent_gram_stats"
            register_neb_method(
                name=method,
                tangent_fn=stored_method.tangent_fn,
                force_fn=neb_effective_force_from_gram_stats,
                climbing_force_fn=stored_method.climbing_force_fn,
            )
        inputs = list(_inputs(device="cuda", dtype=dtype))
        inputs[8] = torch.tensor(
            [FIXED_ENDPOINT, mode, FIXED_ENDPOINT],
            dtype=torch.int32,
            device="cuda",
        )
        cuda_inputs = tuple(inputs)

        scalar = _launch_neb_kernel_on_cuda(method, cuda_inputs, tiled=False)
        tiled = _launch_neb_kernel_on_cuda(method, cuda_inputs, tiled=True)

        tolerance = 2.0e-5 if dtype == torch.float32 else 1.0e-11
        torch.testing.assert_close(scalar, tiled, atol=tolerance, rtol=tolerance)


# =============================================================================
# Graph capture
# =============================================================================


class TestNEBGraphCapture:
    """Test compilation and graph capture of the Torch adapter."""

    def test_neb_forces_captures_fullgraph(self) -> None:
        inputs = _inputs(device="cpu")

        def compiled_neb_forces(*args):
            return neb_forces(*args, method="improved_tangent")

        compiled = torch.compile(compiled_neb_forces, backend="eager", fullgraph=True)
        torch.testing.assert_close(compiled(*inputs), neb_forces(*inputs))
