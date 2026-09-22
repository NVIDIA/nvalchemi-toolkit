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

from typing import Any

import pytest
import torch
import warp as wp

from nvalchemi.dynamics.paths.neb._ops import (
    neb_forces,
    register_neb_method,
)
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
)


@wp.func
def _twice_neb_effective_force(
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
    """Return twice the built-in regular NEB force for registry testing."""
    return type(force_dot_tangent)(2.0) * neb_effective_force_from_gram_stats(
        physical_force,
        tangent,
        d_plus,
        d_minus,
        force_dot_tangent,
        dplus_dot_tangent,
        dminus_dot_tangent,
        norm_d_plus,
        norm_d_minus,
        dplus_dot_dminus,
        force_dot_dplus,
        force_dot_dminus,
        force_squared_norm,
        k_plus,
        k_minus,
        energy_prev,
        energy_curr,
        energy_next,
        path_energy_ref,
        path_energy_max,
    )


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


def _ragged_multi_path_inputs(
    device: str = "cuda", dtype: torch.dtype = torch.float64
) -> tuple[torch.Tensor, ...]:
    """Return two paths with distinct image counts, atom counts, springs, and cells."""
    path_0_base = torch.tensor(
        [[0.1, 0.2, 0.3], [0.7, 1.1, 0.5]], device=device, dtype=dtype
    )
    path_0_step = torch.tensor([2.7, 0.3, 0.1], device=device, dtype=dtype)
    path_0 = torch.cat([path_0_base + i * path_0_step for i in range(3)])

    path_1_base = torch.tensor(
        [[0.2, 0.1, 0.4], [0.9, 0.8, 0.6], [1.4, 1.5, 1.0]],
        device=device,
        dtype=dtype,
    )
    path_1_step = torch.tensor([2.6, 0.55, 0.25], device=device, dtype=dtype)
    path_1 = torch.cat([path_1_base + i * path_1_step for i in range(4)])
    positions = torch.cat((path_0, path_1)).contiguous()
    physical_forces = torch.linspace(
        -0.9,
        1.1,
        positions.numel(),
        device=device,
        dtype=dtype,
    ).reshape_as(positions)

    orthogonal_cell = torch.diag(
        torch.tensor([3.0, 4.0, 5.0], device=device, dtype=dtype)
    )
    skewed_cell = torch.tensor(
        [[3.0, 0.0, 0.0], [0.6, 2.8, 0.0], [0.2, 0.4, 3.2]],
        device=device,
        dtype=dtype,
    )
    periodic_basis = torch.stack((orthogonal_cell, skewed_cell))
    cartesian_to_fractional = torch.linalg.inv(periodic_basis).contiguous()
    candidate_shifts = torch.zeros((2, 26, 3), device=device, dtype=dtype)
    coefficients = torch.cartesian_prod(
        *[torch.arange(-1, 2, device=device) for _ in range(3)]
    )
    coefficients = coefficients[torch.any(coefficients != 0, dim=1)]
    candidate_shifts[1] = coefficients.to(dtype) @ skewed_cell

    return (
        positions,
        physical_forces,
        torch.tensor(
            [0.0, 1.2, 0.1, 2.0, 2.8, 3.4, 2.2],
            device=device,
            dtype=dtype,
        ),
        torch.tensor([0, 2, 4, 6, 9, 12, 15, 18], device=device, dtype=torch.int32),
        torch.tensor([0, 3, 7], device=device, dtype=torch.int32),
        torch.tensor([0, 0, 0, 1, 1, 1, 1], device=device, dtype=torch.int32),
        torch.tensor([0.11, 0.17, 0.23, 0.31, 0.47], device=device, dtype=dtype),
        torch.tensor(
            [
                ENDPOINT,
                REGULAR_NEB,
                ENDPOINT,
                ENDPOINT,
                REGULAR_NEB,
                CLIMBING_NEB,
                ENDPOINT,
            ],
            device=device,
            dtype=torch.int32,
        ),
        torch.tensor([1.2, 2.8], device=device, dtype=dtype),
        torch.tensor([1.2, 3.4], device=device, dtype=dtype),
        torch.tensor([1, 2], device=device, dtype=torch.int32),
        periodic_basis,
        cartesian_to_fractional,
        torch.tensor([0, 26], device=device, dtype=torch.int32),
        candidate_shifts.contiguous(),
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

            zero = energy_curr.new_tensor(0.0)
            one = energy_curr.new_tensor(1.0)
            if bool((delta_minus > 0) & (delta_plus > 0)):
                weight_plus, weight_minus = one, zero
            elif bool((delta_minus < 0) & (delta_plus < 0)):
                weight_plus, weight_minus = zero, one
            else:
                delta_max = torch.maximum(delta_plus.abs(), delta_minus.abs())
                delta_min = torch.minimum(delta_plus.abs(), delta_minus.abs())
                if bool(energy_next > energy_prev):
                    weight_plus, weight_minus = delta_max, delta_min
                else:
                    weight_plus, weight_minus = delta_min, delta_max
            tangent = weight_plus * d_plus + weight_minus * d_minus

            tangent_norm = torch.linalg.vector_norm(tangent)
            dplus_norm = torch.linalg.vector_norm(d_plus)
            dminus_norm = torch.linalg.vector_norm(d_minus)
            tangent_scale = (
                weight_plus.abs() * dplus_norm + weight_minus.abs() * dminus_norm
            )
            rtol = 1.0e-6 if positions.dtype == torch.float32 else 1.0e-12
            if bool(tangent_norm <= rtol * tangent_scale):
                if bool((dplus_norm >= dminus_norm) & (dplus_norm > 0)):
                    tangent = d_plus
                    tangent_norm = dplus_norm
                elif bool(dminus_norm > 0):
                    tangent = d_minus
                    tangent_norm = dminus_norm
                else:
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
    def test_runs_user_defined_equation_through_neb_forces(self, device: str) -> None:
        stored_method = get_neb_method("improved_tangent")
        assert isinstance(stored_method, _StoredTangentMethod)
        method = "test_twice_effective_force"
        register_neb_method(
            name=method,
            tangent_fn=stored_method.tangent_fn,
            force_fn=_twice_neb_effective_force,
            climbing_force_fn=stored_method.climbing_force_fn,
        )
        inputs = list(_inputs(device=device))
        inputs[7][1] = REGULAR_NEB

        reference_forces, reference_links = neb_forces(*inputs)
        actual_forces, actual_links = neb_forces(*inputs, method=method)

        torch.testing.assert_close(actual_links, reference_links, atol=0, rtol=0)
        torch.testing.assert_close(actual_forces[:2], reference_forces[:2])
        torch.testing.assert_close(actual_forces[2:4], 2 * reference_forces[2:4])
        torch.testing.assert_close(actual_forces[4:], reference_forces[4:])

    def test_rejects_same_type_reregistration_with_different_equations(self) -> None:
        stored_method = get_neb_method("improved_tangent")
        assert isinstance(stored_method, _StoredTangentMethod)
        name = "test_conflicting_gram_stats"
        register_neb_method(
            name=name,
            tangent_fn=stored_method.tangent_fn,
            force_fn=neb_effective_force_from_gram_stats,
            climbing_force_fn=stored_method.climbing_force_fn,
        )

        with pytest.raises(ValueError, match="already registered"):
            register_neb_method(
                name=name,
                tangent_fn=stored_method.tangent_fn,
                force_fn=_twice_neb_effective_force,
                climbing_force_fn=stored_method.climbing_force_fn,
            )

    @pytest.mark.parametrize(
        "field_name",
        ["tangent_fn", "force_fn", "climbing_force_fn"],
    )
    def test_rejects_non_warp_equations(self, field_name: str) -> None:
        stored_method = get_neb_method("improved_tangent")
        assert isinstance(stored_method, _StoredTangentMethod)
        equations = {
            "tangent_fn": stored_method.tangent_fn,
            "force_fn": neb_effective_force_from_gram_stats,
            "climbing_force_fn": stored_method.climbing_force_fn,
        }
        equations[field_name] = object()

        with pytest.raises(TypeError, match=field_name):
            register_neb_method(name=f"test_invalid_{field_name}", **equations)

    @pytest.mark.parametrize("name", ["", "   "])
    def test_rejects_empty_method_names(self, name: str) -> None:
        stored_method = get_neb_method("improved_tangent")
        assert isinstance(stored_method, _StoredTangentMethod)

        with pytest.raises(ValueError, match="must not be empty"):
            register_neb_method(
                name=name,
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
    """Test the Torch adapter behavior."""

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
            vector_scratch=tangent,
            effective_forces=forces,
            link_lengths=links,
        )

        assert actual == (forces, links)
        torch.testing.assert_close(actual, reference)

    @pytest.mark.parametrize(
        ("case", "match"),
        [
            ("position-shape", "positions must have shape"),
            ("position-dtype", "positions must have dtype"),
            ("pointer-rank", "pointer tensors one-dimensional"),
            ("invalid-batch-size", "invalid batch sizes"),
            ("float-shape", "physical_forces must have shape"),
            ("float-layout", "physical_forces must match"),
            ("integer-dtype", "image_ptr must be"),
            ("candidate-shape", "candidate_shifts must be"),
        ],
    )
    def test_rejects_invalid_inputs(self, case: str, match: str) -> None:
        inputs = list(_inputs(device="cpu"))
        if case == "position-shape":
            inputs[0] = inputs[0][:, :2]
        elif case == "position-dtype":
            inputs[0] = inputs[0].to(torch.int64)
        elif case == "pointer-rank":
            inputs[3] = inputs[3].unsqueeze(0)
        elif case == "invalid-batch-size":
            inputs[3] = torch.empty(0, dtype=torch.int32)
        elif case == "float-shape":
            inputs[1] = inputs[1][:-1]
        elif case == "float-layout":
            inputs[1] = inputs[1].T.contiguous().T
        elif case == "integer-dtype":
            inputs[3] = inputs[3].to(torch.int64)
        else:
            inputs[14] = inputs[14][:, :-1]

        with pytest.raises(ValueError, match=match):
            neb_forces(*inputs)

    def test_rejects_zero_paths(self) -> None:
        inputs = list(_inputs(device="cpu"))
        inputs[4] = torch.tensor([0], dtype=torch.int32)

        with pytest.raises(ValueError, match="at least one path"):
            neb_forces(*inputs)

    @pytest.mark.parametrize(
        ("case", "match"),
        [
            ("force-shape", "effective_forces must have shape"),
            ("scratch-layout", "tangent_buffer must match"),
        ],
    )
    def test_rejects_invalid_writable_buffers(self, case: str, match: str) -> None:
        inputs = _inputs(device="cpu")
        forces = torch.empty_like(inputs[0])
        links = torch.empty_like(inputs[6])
        scratch = torch.empty_like(inputs[0])
        if case == "force-shape":
            forces = forces[:-1]
        else:
            scratch = scratch.T.contiguous().T

        with pytest.raises(ValueError, match=match):
            neb_forces(
                *inputs,
                vector_scratch=scratch,
                effective_forces=forces,
                link_lengths=links,
            )

    def test_rejects_vector_scratch_for_gram_stats_method(self) -> None:
        stored_method = get_neb_method("improved_tangent")
        assert isinstance(stored_method, _StoredTangentMethod)
        method = "test_rejects_gram_stats_vector_scratch"
        register_neb_method(
            name=method,
            tangent_fn=stored_method.tangent_fn,
            force_fn=neb_effective_force_from_gram_stats,
            climbing_force_fn=stored_method.climbing_force_fn,
        )
        inputs = _inputs(device="cpu")

        with pytest.raises(
            ValueError,
            match="Gram-statistics NEB methods do not accept vector_scratch",
        ):
            neb_forces(
                *inputs,
                method=method,
                vector_scratch=torch.empty_like(inputs[0]),
            )

    def test_rejects_unknown_method(self) -> None:
        with pytest.raises(ValueError, match="unknown NEB method.*available methods"):
            neb_forces(*_inputs(device="cpu"), method="not_registered")


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
    def test_ragged_multi_path_bookkeeping_matches_naive_torch(
        self,
        dtype: torch.dtype,
        device: str,
    ) -> None:
        """Ragged paths keep link, spring, cell, and force bookkeeping isolated."""
        inputs = _ragged_multi_path_inputs(device=device, dtype=dtype)
        stored_method = get_neb_method("improved_tangent")
        assert isinstance(stored_method, _StoredTangentMethod)
        gram_method = "ragged_multi_path_gram_stats"
        register_neb_method(
            name=gram_method,
            tangent_fn=stored_method.tangent_fn,
            force_fn=neb_effective_force_from_gram_stats,
            climbing_force_fn=stored_method.climbing_force_fn,
        )

        expected = _naive_improved_tangent_neb_forces(*inputs)
        for method in ("improved_tangent", gram_method):
            actual = neb_forces(*inputs, method=method)
            tolerance = 2.0e-5 if dtype == torch.float32 else 1.0e-11
            torch.testing.assert_close(
                actual,
                expected,
                atol=tolerance,
                rtol=tolerance,
            )

        assert inputs[3].diff().cpu().tolist() == [2, 2, 2, 3, 3, 3, 3]
        assert inputs[4].diff().cpu().tolist() == [3, 4]
        assert expected[1].shape == (5,)

    @pytest.mark.parametrize(
        ("image_offsets", "energies", "expect_zero"),
        [
            pytest.param([0.0, 0.0, 1.0], [2.0, 1.0, 0.0], False, id="fallback-1"),
            pytest.param([0.0, 1.0, 1.0], [0.0, 1.0, 2.0], False, id="fallback-2"),
            pytest.param([0.0, 0.0, 0.0], [0.0, 1.0, 2.0], True, id="fallback-3"),
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
    def test_degenerate_tangent_fallbacks_match_oracle(
        self,
        image_offsets: list[float],
        energies: list[float],
        expect_zero: bool,
        device: str,
    ) -> None:
        """Both kernel strategies use the documented zero-tangent fallback."""
        inputs = list(_inputs(device=device))
        base_image = inputs[0][:2]
        offsets = base_image.new_tensor(image_offsets).reshape(-1, 1, 1)
        inputs[0] = (base_image.unsqueeze(0) + offsets).reshape(-1, 3).contiguous()
        inputs[2] = inputs[2].new_tensor(energies)

        stored_method = get_neb_method("improved_tangent")
        assert isinstance(stored_method, _StoredTangentMethod)
        gram_method = "improved_tangent_fallback_gram_stats"
        register_neb_method(
            name=gram_method,
            tangent_fn=stored_method.tangent_fn,
            force_fn=neb_effective_force_from_gram_stats,
            climbing_force_fn=stored_method.climbing_force_fn,
        )

        expected = _naive_improved_tangent_neb_forces(*inputs)
        for method in ("improved_tangent", gram_method):
            actual = neb_forces(*inputs, method=method)
            torch.testing.assert_close(actual, expected, atol=1.0e-11, rtol=1.0e-11)

        interior_forces = expected[0][2:4]
        assert bool(torch.all(interior_forces == 0)) is expect_zero


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


@pytest.mark.parametrize("mic_case", ["orthogonal", "partial-periodic"])
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
def test_orthogonal_and_partial_periodic_mic_match_torch_reference(
    device: str,
    dtype: torch.dtype,
    mic_case: str,
) -> None:
    """Both kernel strategies cover orthogonal and zero-padded partial MIC."""
    stored_method = get_neb_method("improved_tangent")
    assert isinstance(stored_method, _StoredTangentMethod)
    gram_method = "test_orthogonal_partial_mic_gram_stats"
    register_neb_method(
        name=gram_method,
        tangent_fn=stored_method.tangent_fn,
        force_fn=neb_effective_force_from_gram_stats,
        climbing_force_fn=stored_method.climbing_force_fn,
    )
    inputs = list(_inputs(device=device, dtype=dtype))
    one_image = torch.tensor(
        [[0.1, 0.2, 0.3], [0.6, 0.8, 0.9]],
        dtype=dtype,
        device=device,
    )
    candidate_shifts = torch.zeros((1, 26, 3), dtype=dtype, device=device)

    if mic_case == "orthogonal":
        offset = torch.tensor([1.1, 2.1, 2.6], dtype=dtype, device=device)
        basis = torch.diag(
            torch.tensor([2.0, 3.0, 4.0], dtype=dtype, device=device)
        ).unsqueeze(0)
        cartesian_to_fractional = torch.linalg.inv(basis).contiguous()
        mic_mode = 1
        candidate_count = 0
    else:
        offset = torch.tensor([1.3, 1.2, 2.5], dtype=dtype, device=device)
        active_basis = torch.tensor(
            [[2.0, 0.0, 0.0], [0.5, 2.0, 0.0]],
            dtype=dtype,
            device=device,
        )
        basis = torch.zeros((1, 3, 3), dtype=dtype, device=device)
        basis[0, :2] = active_basis
        cartesian_to_fractional = torch.zeros_like(basis)
        cartesian_to_fractional[0, :, :2] = torch.linalg.pinv(active_basis)
        coefficients = torch.cartesian_prod(
            torch.arange(-1, 2, device=device),
            torch.arange(-1, 2, device=device),
        )
        coefficients = coefficients[torch.any(coefficients != 0, dim=1)]
        candidate_shifts[0, :8] = coefficients.to(dtype) @ active_basis
        mic_mode = 2
        candidate_count = 8

    inputs[0] = torch.cat(
        (one_image, one_image + offset, one_image + 2 * offset)
    ).contiguous()
    inputs[10:15] = [
        torch.tensor([mic_mode], device=device, dtype=torch.int32),
        basis.contiguous(),
        cartesian_to_fractional.contiguous(),
        torch.tensor([candidate_count], device=device, dtype=torch.int32),
        candidate_shifts.contiguous(),
    ]
    expected = _naive_improved_tangent_neb_forces(*inputs)
    tolerance = 2.0e-5 if dtype == torch.float32 else 1.0e-11

    for method in ("improved_tangent", gram_method):
        actual = neb_forces(*inputs, method=method)
        torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)

    if mic_case == "orthogonal":
        wrapped_offset = offset.new_tensor([-0.9, -0.9, -1.4])
        expected_link = torch.sqrt(2 * torch.sum(wrapped_offset.square()))
        torch.testing.assert_close(
            expected[1], expected_link.expand(2), atol=tolerance, rtol=tolerance
        )
    else:
        assert bool(torch.all(basis[0, 2] == 0))
        assert bool(torch.all(cartesian_to_fractional[0, :, 2] == 0))
        displacement = _minimum_image_displacement(
            inputs[0][2:4],
            inputs[0][:2],
            0,
            *inputs[10:15],
        )
        torch.testing.assert_close(displacement[:, 2], offset[2].expand(2))


# =============================================================================
# Compilation and CUDA graph capture
# =============================================================================


class TestNEBCompilation:
    """Test compilation and CUDA graph capture of the Torch adapter."""

    @pytest.mark.parametrize("method", ["improved_tangent", "test_compile_gram_stats"])
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
    def test_neb_forces_compiles_with_inductor_fullgraph(
        self, device: str, method: str
    ) -> None:
        """Inductor compiles the complete adapter graph on CPU and CUDA."""
        if method != "improved_tangent":
            stored_method = get_neb_method("improved_tangent")
            assert isinstance(stored_method, _StoredTangentMethod)
            register_neb_method(
                name=method,
                tangent_fn=stored_method.tangent_fn,
                force_fn=neb_effective_force_from_gram_stats,
                climbing_force_fn=stored_method.climbing_force_fn,
            )
        inputs = _inputs(device=device)

        def compiled_neb_forces(*args):
            return neb_forces(*args, method=method)

        expected = neb_forces(*inputs, method=method)
        compiled = torch.compile(compiled_neb_forces, fullgraph=True)
        torch.testing.assert_close(compiled(*inputs), expected, atol=0, rtol=0)

    @pytest.mark.parametrize(
        "method", ["improved_tangent", "test_cuda_graph_gram_stats"]
    )
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
    def test_neb_forces_cuda_graph_capture_and_replay(self, method: str) -> None:
        """A warmed adapter captures and replays on a prebound Warp stream."""
        if method != "improved_tangent":
            stored_method = get_neb_method("improved_tangent")
            assert isinstance(stored_method, _StoredTangentMethod)
            register_neb_method(
                name=method,
                tangent_fn=stored_method.tangent_fn,
                force_fn=neb_effective_force_from_gram_stats,
                climbing_force_fn=stored_method.climbing_force_fn,
            )
        inputs = _inputs(device="cuda")
        vector_scratch = (
            torch.empty_like(inputs[0]) if method == "improved_tangent" else None
        )
        effective_forces = torch.empty_like(inputs[0])
        link_lengths = torch.empty_like(inputs[6])

        # Warp kernels must be specialized and JIT-compiled before capture.
        neb_forces(
            *inputs,
            method=method,
            vector_scratch=vector_scratch,
            effective_forces=effective_forces,
            link_lengths=link_lengths,
        )
        torch.cuda.synchronize()

        # Compute the replay oracle with changed tensor contents but identical
        # shapes and addresses for the inputs captured below.
        replay_forces = inputs[1] + 0.25
        replay_inputs = list(inputs)
        replay_inputs[1] = replay_forces
        expected = neb_forces(*replay_inputs, method=method)
        torch.cuda.synchronize()

        stream = torch.cuda.Stream()
        warp_stream = wp.stream_from_torch(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.stream(stream), wp.ScopedStream(warp_stream):
            with torch.cuda.graph(graph, stream=stream):
                neb_forces(
                    *inputs,
                    method=method,
                    vector_scratch=vector_scratch,
                    effective_forces=effective_forces,
                    link_lengths=link_lengths,
                )

            # The graph retains the input addresses, so changing their contents
            # lets the assertion below prove that replay executes the kernels.
            inputs[1].copy_(replay_forces)
            graph.replay()

        torch.cuda.synchronize()
        torch.testing.assert_close(effective_forces, expected[0], atol=0, rtol=0)
        torch.testing.assert_close(link_lengths, expected[1], atol=0, rtol=0)
