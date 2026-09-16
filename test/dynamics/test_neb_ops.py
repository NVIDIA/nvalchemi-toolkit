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

"""Tests for generated NEB Torch adapters."""

from __future__ import annotations

import pytest
import torch

from nvalchemi.dynamics.paths.neb._ops.equations import (
    neb_effective_force_from_gram_stats,
)
from nvalchemi.dynamics.paths.neb._ops.launchers import (
    _get_neb_forces_kernel_overloads,
)
from nvalchemi.dynamics.paths.neb._ops.modes import (
    CLIMBING_NEB,
    ENDPOINT,
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
    mic_mode = torch.zeros(1, device=device, dtype=torch.int32)
    periodic_basis = torch.zeros((1, 3, 3), device=device, dtype=dtype)
    cartesian_to_fractional = torch.zeros_like(periodic_basis)
    mic_candidate_count = torch.zeros(1, device=device, dtype=torch.int32)
    candidate_shifts = torch.zeros((1, 26, 3), device=device, dtype=dtype)
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
        torch.tensor([0, 2, 0], device=device, dtype=torch.int32),
        torch.tensor([0], device=device, dtype=dtype),
        torch.tensor([1], device=device, dtype=dtype),
        mic_mode,
        periodic_basis,
        cartesian_to_fractional,
        mic_candidate_count,
        candidate_shifts,
    )


# =============================================================================
# Pure-Torch reference implementation
# =============================================================================


def _minimum_image_displacement(
    r_a: torch.Tensor,
    r_b: torch.Tensor,
    path_idx: int,
    mic_mode: torch.Tensor,
    periodic_basis: torch.Tensor,
    cartesian_to_fractional: torch.Tensor,
    mic_candidate_count: torch.Tensor,
    candidate_shifts: torch.Tensor,
) -> torch.Tensor:
    """Return minimum-image displacements using only Torch operations."""
    displacement = r_a - r_b
    if int(mic_mode[path_idx]) == 0:
        return displacement
    fractional = displacement @ cartesian_to_fractional[path_idx]
    offset = 0.5 if int(mic_mode[path_idx]) == 1 else 0.0
    wrapped = fractional - torch.floor(fractional + offset)
    base = (
        displacement
        - fractional @ periodic_basis[path_idx]
        + wrapped @ periodic_basis[path_idx]
    )
    best_sq = torch.sum(base.square(), dim=-1)
    for shift in candidate_shifts[path_idx, : int(mic_candidate_count[path_idx])]:
        candidate = base + shift
        candidate_sq = torch.sum(candidate.square(), dim=-1)
        improve = candidate_sq < best_sq
        base = torch.where(improve[:, None], candidate, base)
        best_sq = torch.where(improve, candidate_sq, best_sq)
    return base


def _naive_improved_tangent_neb_forces(
    positions: torch.Tensor,
    physical_forces: torch.Tensor,
    image_energies: torch.Tensor,
    image_ptr: torch.Tensor,
    path_ptr: torch.Tensor,
    image_path_idx: torch.Tensor,
    spring_constants: torch.Tensor,
    image_force_mode: torch.Tensor,
    path_energy_ref: torch.Tensor,
    path_energy_max: torch.Tensor,
    mic_mode: torch.Tensor,
    periodic_basis: torch.Tensor,
    cartesian_to_fractional: torch.Tensor,
    mic_candidate_count: torch.Tensor,
    candidate_shifts: torch.Tensor,
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
        for image_idx in (image_start, image_stop - 1):
            atom_start = int(image_ptr[image_idx])
            atom_stop = int(image_ptr[image_idx + 1])
            effective_forces[atom_start:atom_stop] = physical_forces[
                atom_start:atom_stop
            ]
        for image_idx in range(image_start, image_stop - 1):
            atom_start = int(image_ptr[image_idx])
            atom_stop = int(image_ptr[image_idx + 1])
            next_atom_start = int(image_ptr[image_idx + 1])
            displacement = _minimum_image_displacement(
                positions[next_atom_start : next_atom_start + atom_stop - atom_start],
                positions[atom_start:atom_stop],
                path_idx,
                mic_mode,
                periodic_basis,
                cartesian_to_fractional,
                mic_candidate_count,
                candidate_shifts,
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

    return effective_forces, link_lengths


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

    def test_method_overloads_are_cached(self) -> None:
        first = _get_neb_forces_kernel_overloads("improved_tangent")
        second = _get_neb_forces_kernel_overloads("improved_tangent")
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
        inputs[7] = torch.tensor(
            [ENDPOINT, mode, ENDPOINT],
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
    """Compare equivalent NEB kernel strategies."""

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


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
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
def test_skewed_cell_link_lengths_use_exact_mic_for_both_kernel_strategies(
    device: str,
    dtype: torch.dtype,
) -> None:
    """Stored-tangent and Gram-statistics kernels search skew-cell images."""
    # Compare both kernel strategies using the same improved-tangent setup.
    stored_method = get_neb_method("improved_tangent")
    assert isinstance(stored_method, _StoredTangentMethod)
    gram_method = "test_skewed_mic_gram_stats"
    register_neb_method(
        name=gram_method,
        tangent_fn=stored_method.tangent_fn,
        force_fn=neb_effective_force_from_gram_stats,
        climbing_force_fn=stored_method.climbing_force_fn,
    )
    inputs = list(_inputs(device=device, dtype=dtype))
    # Use identical two-atom images so both path links have the same known
    # displacement and therefore the same expected minimum-image length.
    one_image = torch.tensor(
        [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=dtype,
        device=device,
    )
    offset = torch.tensor([0.7, 0.49, 0.0], dtype=dtype, device=device)
    inputs[0] = torch.cat((one_image, one_image + offset, one_image + 2 * offset))
    # This tilted lattice makes component-wise fractional rounding insufficient.
    cell = torch.tensor(
        [[[1.0, 0.0, 0.0], [0.5, 1.0, 0.0], [0.0, 0.0, 1.0]]],
        dtype=dtype,
        device=device,
    )
    # Enumerate nearby lattice translations for the general skew-cell MIC scan.
    coefficients = torch.cartesian_prod(
        *[torch.arange(-1, 2, device=device) for _ in range(3)]
    )
    coefficients = coefficients[torch.any(coefficients != 0, dim=1)]
    candidate_shifts = torch.zeros((1, 26, 3), dtype=dtype, device=device)
    candidate_shifts[0] = coefficients.to(dtype) @ cell[0]
    # Select general MIC mode and provide its cell, inverse, and candidate shifts.
    inputs[10:15] = [
        torch.tensor([2], device=device, dtype=torch.int32),
        cell,
        torch.linalg.inv(cell).contiguous(),
        torch.tensor([26], device=device, dtype=torch.int32),
        candidate_shifts.contiguous(),
    ]
    # The best per-atom image is (0.2, -0.51, 0); two atoms contribute to each
    # link norm, which explains the factor of two.
    expected_link = torch.sqrt(
        torch.tensor(2 * (0.2**2 + 0.51**2), dtype=dtype, device=device)
    )

    # Only link lengths are relevant here; the two force strategies are tested
    # independently through the second return value.
    stored_links = neb_forces(*inputs, method="improved_tangent")[1]
    gram_links = neb_forces(*inputs, method=gram_method)[1]

    tolerance = 2.0e-6 if dtype == torch.float32 else 1.0e-12

    # The three images form two links, and both must use the exact skew-cell MIC.
    torch.testing.assert_close(
        stored_links, expected_link.expand(2), atol=tolerance, rtol=tolerance
    )
    torch.testing.assert_close(
        gram_links, expected_link.expand(2), atol=tolerance, rtol=tolerance
    )


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
