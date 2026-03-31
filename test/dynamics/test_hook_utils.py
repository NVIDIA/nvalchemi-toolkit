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
"""Unit tests for ``nvalchemi.dynamics.hooks._utils``."""

from __future__ import annotations

import torch

from nvalchemi.dynamics.hooks._utils import (
    KB_EV,
    kinetic_energy_per_graph,
    scatter_reduce_per_graph,
    temperature_per_graph,
    wrap_positions_into_cell,
)

# ---------------------------------------------------------------------------
# scatter_reduce_per_graph
# ---------------------------------------------------------------------------


class TestScatterReducePerGraph:
    """Test suite for :func:`scatter_reduce_per_graph`."""

    def test_amax_single_graph(self) -> None:
        """Verify amax reduction for a single graph."""
        values = torch.tensor([1.0, 5.0, 3.0])
        batch_idx = torch.tensor([0, 0, 0])
        result = scatter_reduce_per_graph(
            values, batch_idx, num_graphs=1, reduce="amax"
        )
        assert result.shape == (1,)
        assert torch.isclose(result[0], torch.tensor(5.0))

    def test_amax_multi_graph(self) -> None:
        """Verify amax reduction across multiple graphs."""
        values = torch.tensor([5.0, 1.0, 10.0, 2.0])
        batch_idx = torch.tensor([0, 0, 1, 1])
        result = scatter_reduce_per_graph(
            values, batch_idx, num_graphs=2, reduce="amax"
        )
        assert result.shape == (2,)
        assert torch.isclose(result[0], torch.tensor(5.0))
        assert torch.isclose(result[1], torch.tensor(10.0))

    def test_sum_single_graph(self) -> None:
        """Verify sum reduction for a single graph."""
        values = torch.tensor([1.0, 2.0, 3.0])
        batch_idx = torch.tensor([0, 0, 0])
        result = scatter_reduce_per_graph(values, batch_idx, num_graphs=1, reduce="sum")
        assert result.shape == (1,)
        assert torch.isclose(result[0], torch.tensor(6.0))

    def test_sum_multi_graph(self) -> None:
        """Verify sum reduction across multiple graphs."""
        values = torch.tensor([1.0, 2.0, 10.0, 20.0])
        batch_idx = torch.tensor([0, 0, 1, 1])
        result = scatter_reduce_per_graph(values, batch_idx, num_graphs=2, reduce="sum")
        assert result.shape == (2,)
        assert torch.isclose(result[0], torch.tensor(3.0))
        assert torch.isclose(result[1], torch.tensor(30.0))

    def test_amin_multi_graph(self) -> None:
        """Verify amin reduction across multiple graphs."""
        values = torch.tensor([5.0, 1.0, 10.0, 2.0])
        batch_idx = torch.tensor([0, 0, 1, 1])
        result = scatter_reduce_per_graph(
            values, batch_idx, num_graphs=2, reduce="amin"
        )
        assert result.shape == (2,)
        assert torch.isclose(result[0], torch.tensor(1.0))
        assert torch.isclose(result[1], torch.tensor(2.0))

    def test_mean_multi_graph(self) -> None:
        """Verify mean reduction across multiple graphs."""
        values = torch.tensor([2.0, 4.0, 10.0, 20.0])
        batch_idx = torch.tensor([0, 0, 1, 1])
        result = scatter_reduce_per_graph(
            values, batch_idx, num_graphs=2, reduce="mean"
        )
        assert result.shape == (2,)
        assert torch.isclose(result[0], torch.tensor(3.0))
        assert torch.isclose(result[1], torch.tensor(15.0))

    def test_default_reduce_is_amax(self) -> None:
        """Verify default reduction is amax."""
        values = torch.tensor([1.0, 5.0, 3.0])
        batch_idx = torch.tensor([0, 0, 0])
        result = scatter_reduce_per_graph(values, batch_idx, num_graphs=1)
        assert torch.isclose(result[0], torch.tensor(5.0))

    def test_all_zeros(self) -> None:
        """Verify zero values are handled correctly with sum."""
        values = torch.zeros(4)
        batch_idx = torch.tensor([0, 0, 1, 1])
        result = scatter_reduce_per_graph(values, batch_idx, num_graphs=2, reduce="sum")
        assert torch.allclose(result, torch.zeros(2))

    def test_composable_with_vector_norm(self) -> None:
        """Verify scatter_reduce composes with vector_norm for fmax-like queries."""
        # This replaces the old fmax_per_graph function
        forces = torch.tensor(
            [
                [3.0, 4.0, 0.0],  # norm 5.0
                [0.0, 0.0, 1.0],  # norm 1.0
                [6.0, 8.0, 0.0],  # norm 10.0
                [0.0, 2.0, 0.0],  # norm 2.0
            ]
        )
        batch_idx = torch.tensor([0, 0, 1, 1])
        norms = torch.linalg.vector_norm(forces, dim=-1)
        result = scatter_reduce_per_graph(norms, batch_idx, num_graphs=2, reduce="amax")
        assert torch.isclose(result[0], torch.tensor(5.0))
        assert torch.isclose(result[1], torch.tensor(10.0))


# ---------------------------------------------------------------------------
# kinetic_energy_per_graph
# ---------------------------------------------------------------------------


class TestKineticEnergyPerGraph:
    """Test suite for :func:`kinetic_energy_per_graph`."""

    def test_single_atom_single_graph(self) -> None:
        """Verify KE = 0.5 * m * v^2 for a single atom."""
        velocities = torch.tensor([[1.0, 0.0, 0.0]])
        masses = torch.tensor([2.0])
        batch_idx = torch.tensor([0])
        result = kinetic_energy_per_graph(velocities, masses, batch_idx, num_graphs=1)
        # KE = 0.5 * 2.0 * (1^2 + 0 + 0) = 1.0
        assert result.shape == (1, 1)
        assert torch.isclose(result[0, 0], torch.tensor(1.0))

    def test_multi_atom_single_graph(self) -> None:
        """Verify KE sums over atoms in a single graph."""
        velocities = torch.tensor([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
        masses = torch.tensor([1.0, 3.0])
        batch_idx = torch.tensor([0, 0])
        result = kinetic_energy_per_graph(velocities, masses, batch_idx, num_graphs=1)
        # KE = 0.5*1*(1) + 0.5*3*(4) = 0.5 + 6.0 = 6.5
        assert torch.isclose(result[0, 0], torch.tensor(6.5))

    def test_multi_graph(self) -> None:
        """Verify KE per graph for a batched system."""
        velocities = torch.tensor(
            [
                [1.0, 0.0, 0.0],  # graph 0
                [0.0, 1.0, 0.0],  # graph 1
            ]
        )
        masses = torch.tensor([2.0, 4.0])
        batch_idx = torch.tensor([0, 1])
        result = kinetic_energy_per_graph(velocities, masses, batch_idx, num_graphs=2)
        # Graph 0: 0.5*2*1 = 1.0
        # Graph 1: 0.5*4*1 = 2.0
        assert result.shape == (2, 1)
        assert torch.isclose(result[0, 0], torch.tensor(1.0))
        assert torch.isclose(result[1, 0], torch.tensor(2.0))

    def test_masses_2d(self) -> None:
        """Verify masses with shape (V, 1) are handled correctly."""
        velocities = torch.tensor([[1.0, 0.0, 0.0]])
        masses = torch.tensor([[2.0]])  # (1, 1)
        batch_idx = torch.tensor([0])
        result = kinetic_energy_per_graph(velocities, masses, batch_idx, num_graphs=1)
        assert torch.isclose(result[0, 0], torch.tensor(1.0))

    def test_zero_velocity(self) -> None:
        """Verify zero velocity gives zero KE."""
        velocities = torch.zeros(3, 3)
        masses = torch.tensor([1.0, 2.0, 3.0])
        batch_idx = torch.tensor([0, 0, 0])
        result = kinetic_energy_per_graph(velocities, masses, batch_idx, num_graphs=1)
        assert torch.isclose(result[0, 0], torch.tensor(0.0))


# ---------------------------------------------------------------------------
# temperature_per_graph
# ---------------------------------------------------------------------------


class TestTemperaturePerGraph:
    """Test suite for :func:`temperature_per_graph`."""

    def test_known_temperature(self) -> None:
        """Verify temperature computation with a known analytic result.

        For N atoms with uniform mass m and speed v, the temperature is
        T = 2 * KE / (3*N*kB) = 2 * (N * 0.5 * m * v^2) / (3*N*kB) = m*v^2 / (3*kB).
        """
        m = 1.0
        v = 1.0
        n_atoms = 10
        velocities = torch.full((n_atoms, 3), v / 3.0**0.5)
        masses = torch.full((n_atoms,), m)
        batch_idx = torch.zeros(n_atoms, dtype=torch.long)
        atoms_per_graph = torch.tensor([n_atoms])

        result = temperature_per_graph(
            velocities, masses, batch_idx, num_graphs=1, atoms_per_graph=atoms_per_graph
        )
        # Each atom: 0.5 * m * (v/sqrt(3))^2 * 3 = 0.5 * m * v^2
        # Total KE = n_atoms * 0.5 * m * v^2
        # T = 2 * KE / (3 * n_atoms * kB) = m * v^2 / (3 * kB)
        expected_temp = m * v**2 / (3.0 * KB_EV)
        assert result.shape == (1,)
        assert torch.isclose(result[0], torch.tensor(expected_temp), rtol=1e-5)

    def test_custom_conversion_factor(self) -> None:
        """Verify that a caller-supplied conversion_factor is honoured."""
        m, v, n_atoms = 1.0, 1.0, 10
        velocities = torch.full((n_atoms, 3), v / 3.0**0.5)
        masses = torch.full((n_atoms,), m)
        batch_idx = torch.zeros(n_atoms, dtype=torch.long)
        atoms_per_graph = torch.tensor([n_atoms])

        custom_cf = 2.0 * KB_EV
        result = temperature_per_graph(
            velocities,
            masses,
            batch_idx,
            num_graphs=1,
            atoms_per_graph=atoms_per_graph,
            conversion_factor=custom_cf,
        )
        expected = m * v**2 / (3.0 * custom_cf)
        assert torch.isclose(result[0], torch.tensor(expected), rtol=1e-5)

    def test_zero_velocity_gives_zero_temperature(self) -> None:
        """Verify zero velocities produce zero temperature."""
        velocities = torch.zeros(5, 3)
        masses = torch.ones(5)
        batch_idx = torch.zeros(5, dtype=torch.long)
        atoms_per_graph = torch.tensor([5])

        result = temperature_per_graph(
            velocities, masses, batch_idx, num_graphs=1, atoms_per_graph=atoms_per_graph
        )
        assert torch.isclose(result[0], torch.tensor(0.0))


# ---------------------------------------------------------------------------
# wrap_positions_into_cell
# ---------------------------------------------------------------------------


class TestWrapPositionsIntoCell:
    """Test suite for :func:`wrap_positions_into_cell`."""

    def test_orthorhombic_cell_wrap(self) -> None:
        """Verify positions outside an orthorhombic cell are wrapped in."""
        # 10 A cubic cell
        cell = torch.tensor([[[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]]])
        pbc = torch.tensor([[True, True, True]])
        positions = torch.tensor([[12.0, -3.0, 25.0]])  # outside the cell
        batch_idx = torch.tensor([0])

        wrapped = wrap_positions_into_cell(positions, cell, pbc, batch_idx)
        expected = torch.tensor([[2.0, 7.0, 5.0]])
        assert torch.allclose(wrapped, expected, atol=1e-5)

    def test_already_inside_cell(self) -> None:
        """Verify positions already inside the cell are unchanged."""
        cell = torch.tensor([[[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]]])
        pbc = torch.tensor([[True, True, True]])
        positions = torch.tensor([[5.0, 5.0, 5.0]])
        batch_idx = torch.tensor([0])

        wrapped = wrap_positions_into_cell(positions, cell, pbc, batch_idx)
        assert torch.allclose(wrapped, positions, atol=1e-5)

    def test_partial_pbc(self) -> None:
        """Verify only periodic dimensions are wrapped (slab geometry)."""
        cell = torch.tensor([[[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 20.0]]])
        pbc = torch.tensor([[True, True, False]])
        positions = torch.tensor([[12.0, -3.0, 25.0]])
        batch_idx = torch.tensor([0])

        wrapped = wrap_positions_into_cell(positions, cell, pbc, batch_idx)
        # x and y wrapped, z left as-is
        assert torch.isclose(wrapped[0, 0], torch.tensor(2.0), atol=1e-5)
        assert torch.isclose(wrapped[0, 1], torch.tensor(7.0), atol=1e-5)
        assert torch.isclose(wrapped[0, 2], torch.tensor(25.0), atol=1e-5)

    def test_no_pbc_is_noop(self) -> None:
        """Verify non-periodic system leaves positions unchanged."""
        cell = torch.tensor([[[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]]])
        pbc = torch.tensor([[False, False, False]])
        positions = torch.tensor([[12.0, -3.0, 25.0]])
        batch_idx = torch.tensor([0])

        wrapped = wrap_positions_into_cell(positions, cell, pbc, batch_idx)
        assert torch.allclose(wrapped, positions, atol=1e-5)

    def test_idempotent(self) -> None:
        """Verify wrapping is idempotent — wrapping twice gives same result."""
        cell = torch.tensor([[[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]]])
        pbc = torch.tensor([[True, True, True]])
        positions = torch.tensor([[12.0, -3.0, 25.0]])
        batch_idx = torch.tensor([0])

        wrapped_once = wrap_positions_into_cell(positions, cell, pbc, batch_idx)
        wrapped_twice = wrap_positions_into_cell(wrapped_once, cell, pbc, batch_idx)
        assert torch.allclose(wrapped_once, wrapped_twice, atol=1e-5)

    def test_triclinic_cell(self) -> None:
        """Verify wrapping works for a triclinic cell."""
        # Triclinic cell: a = [10, 0, 0], b = [2, 10, 0], c = [0, 0, 10]
        cell = torch.tensor([[[10.0, 0.0, 0.0], [2.0, 10.0, 0.0], [0.0, 0.0, 10.0]]])
        pbc = torch.tensor([[True, True, True]])
        # Position that is exactly at fractional [1.5, 0.5, 0.5]
        # Cartesian: 1.5 * [10,0,0] + 0.5 * [2,10,0] + 0.5 * [0,0,10]
        #          = [15, 0, 0] + [1, 5, 0] + [0, 0, 5] = [16, 5, 5]
        positions = torch.tensor([[16.0, 5.0, 5.0]])
        batch_idx = torch.tensor([0])

        wrapped = wrap_positions_into_cell(positions, cell, pbc, batch_idx)
        # Should wrap to fractional [0.5, 0.5, 0.5]
        # Cartesian: 0.5 * [10,0,0] + 0.5 * [2,10,0] + 0.5 * [0,0,10]
        #          = [5, 0, 0] + [1, 5, 0] + [0, 0, 5] = [6, 5, 5]
        expected = torch.tensor([[6.0, 5.0, 5.0]])
        assert torch.allclose(wrapped, expected, atol=1e-4)

    def test_multi_graph_different_cells(self) -> None:
        """Verify wrapping with heterogeneous cells in a batch."""
        # Graph 0: 10A cubic cell, Graph 1: 5A cubic cell
        cell = torch.tensor(
            [
                [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]],
                [[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
            ]
        )
        pbc = torch.tensor([[True, True, True], [True, True, True]])
        # Atom 0 in graph 0: 12 -> 2 (mod 10)
        # Atom 1 in graph 1: 7 -> 2 (mod 5)
        positions = torch.tensor([[12.0, 5.0, 5.0], [7.0, 2.5, 2.5]])
        batch_idx = torch.tensor([0, 1])

        wrapped = wrap_positions_into_cell(positions, cell, pbc, batch_idx)
        assert torch.isclose(wrapped[0, 0], torch.tensor(2.0), atol=1e-5)
        assert torch.isclose(wrapped[1, 0], torch.tensor(2.0), atol=1e-5)


# ---------------------------------------------------------------------------
# torch.compile smoke tests
# ---------------------------------------------------------------------------


class TestUtilsCompile:
    """Verify _utils functions compile with fullgraph=True.

    Uses the ``device`` fixture from ``test/conftest.py`` to run on both
    CPU (inductor backend) and CUDA (cudagraphs backend) when available.

    Warp kernel calls are wrapped with ``@torch.library.custom_op`` to create
    opaque boundaries that torch.compile treats as single graph nodes.
    """

    @staticmethod
    def _compile_kwargs(device: str) -> dict:
        kw: dict = {"fullgraph": True}
        if device == "cuda":
            kw["backend"] = "cudagraphs"
        return kw

    def test_scatter_reduce_sum_compiles(self, device: str) -> None:
        values = torch.tensor([1.0, 2.0, 10.0, 20.0], device=device)
        batch_idx = torch.tensor([0, 0, 1, 1], device=device)
        fn = torch.compile(scatter_reduce_per_graph, **self._compile_kwargs(device))
        result = fn(values, batch_idx, num_graphs=2, reduce="sum")
        assert torch.isclose(result[0], torch.tensor(3.0, device=device))
        assert torch.isclose(result[1], torch.tensor(30.0, device=device))

    def test_scatter_reduce_amax_compiles(self, device: str) -> None:
        values = torch.tensor([5.0, 1.0, 10.0, 2.0], device=device)
        batch_idx = torch.tensor([0, 0, 1, 1], device=device)
        fn = torch.compile(scatter_reduce_per_graph, **self._compile_kwargs(device))
        result = fn(values, batch_idx, num_graphs=2, reduce="amax")
        assert torch.isclose(result[0], torch.tensor(5.0, device=device))
        assert torch.isclose(result[1], torch.tensor(10.0, device=device))

    def test_kinetic_energy_compiles(self, device: str) -> None:
        velocities = torch.tensor([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]], device=device)
        masses = torch.tensor([1.0, 3.0], device=device)
        batch_idx = torch.tensor([0, 0], device=device)
        fn = torch.compile(kinetic_energy_per_graph, **self._compile_kwargs(device))
        result = fn(velocities, masses, batch_idx, num_graphs=1)
        assert torch.isclose(result[0, 0], torch.tensor(6.5, device=device))

    def test_temperature_compiles(self, device: str) -> None:
        n_atoms = 5
        velocities = torch.randn(n_atoms, 3, device=device)
        masses = torch.ones(n_atoms, device=device)
        batch_idx = torch.zeros(n_atoms, dtype=torch.long, device=device)
        atoms_per_graph = torch.tensor([n_atoms], device=device)
        fn = torch.compile(temperature_per_graph, **self._compile_kwargs(device))
        result = fn(
            velocities, masses, batch_idx, num_graphs=1, atoms_per_graph=atoms_per_graph
        )
        assert result.shape == (1,)
        assert torch.isfinite(result).all()

    def test_wrap_positions_compiles(self, device: str) -> None:
        cell = torch.tensor(
            [[[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]]], device=device
        )
        pbc = torch.tensor([[True, True, True]], device=device)
        positions = torch.tensor([[12.0, -3.0, 25.0]], device=device)
        batch_idx = torch.tensor([0], device=device)
        fn = torch.compile(wrap_positions_into_cell, **self._compile_kwargs(device))
        wrapped = fn(positions, cell, pbc, batch_idx)
        expected = torch.tensor([[2.0, 7.0, 5.0]], device=device)
        assert torch.allclose(wrapped, expected, atol=1e-5)
