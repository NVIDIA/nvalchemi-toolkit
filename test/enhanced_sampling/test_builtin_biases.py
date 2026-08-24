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
"""Unit tests for the built-in biases and the periodic CV difference helper.

Covers :class:`HarmonicUmbrellaBias`, :class:`UpperWall`, :class:`LowerWall`,
:class:`FlatBottomRestraint`, and :func:`periodic_difference`.
"""

from __future__ import annotations

import math

import pytest
import torch

from nvalchemi.data import AtomicData, Batch
from nvalchemi.enhanced_sampling import (
    FlatBottomRestraint,
    HarmonicUmbrellaBias,
    LowerWall,
    UpperWall,
    pair_distance,
    periodic_difference,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _pair_batch(distances: list[float], device: str = "cpu") -> Batch:
    """Return one graph per entry, atoms 0 and 1 separated along x."""
    data_list = [
        AtomicData(
            positions=torch.tensor([[0.0, 0.0, 0.0], [d, 0.0, 0.0]]),
            atomic_numbers=torch.tensor([6, 6], dtype=torch.long),
        )
        for d in distances
    ]
    return Batch.from_data_list(data_list).to(device)


def _cv(batch: Batch) -> torch.Tensor:
    return pair_distance(batch, torch.tensor([0, 1], device=batch.positions.device))


# ===========================================================================
# 1. periodic_difference
# ===========================================================================


class TestPeriodicDifference:
    """Wrapping CV differences onto a circle."""

    def test_none_periods_is_plain_difference(self) -> None:
        values = torch.tensor([[3.0]])
        centers = torch.tensor([[-3.0]])
        assert torch.allclose(
            periodic_difference(values, centers, None), torch.tensor([[6.0]])
        )

    def test_wraps_the_short_way_round(self) -> None:
        """+3.0 and -3.0 rad are 0.283 apart, not 6.0."""
        two_pi = 2 * math.pi
        delta = periodic_difference(
            torch.tensor([[3.0]]), torch.tensor([[-3.0]]), torch.tensor([two_pi])
        )
        assert abs(float(delta) - (6.0 - two_pi)) < 1e-6
        assert abs(float(delta)) < math.pi

    def test_result_within_half_period(self) -> None:
        two_pi = 2 * math.pi
        values = torch.linspace(-10, 10, 41).unsqueeze(-1)
        centers = torch.zeros_like(values)
        delta = periodic_difference(values, centers, torch.tensor([two_pi]))
        assert bool((delta.abs() <= math.pi + 1e-6).all())

    def test_zero_period_marks_non_periodic(self) -> None:
        delta = periodic_difference(
            torch.tensor([[100.0, 3.0]]),
            torch.tensor([[0.0, -3.0]]),
            torch.tensor([0.0, 2 * math.pi]),
        )
        assert abs(float(delta[0, 0]) - 100.0) < 1e-6  # unwrapped
        assert abs(float(delta[0, 1])) < math.pi  # wrapped

    def test_non_finite_period_marks_non_periodic(self) -> None:
        delta = periodic_difference(
            torch.tensor([[100.0]]),
            torch.tensor([[0.0]]),
            torch.tensor([float("inf")]),
        )
        assert torch.isfinite(delta).all()
        assert abs(float(delta) - 100.0) < 1e-6

    def test_gradient_matches_unwrapped(self) -> None:
        """round() has zero gradient, so wrapping must not change d(delta)/d(x)."""
        values = torch.tensor([[3.0]], requires_grad=True)
        delta = periodic_difference(
            values, torch.tensor([[-3.0]]), torch.tensor([2 * math.pi])
        )
        (grad,) = torch.autograd.grad(delta.sum(), values)
        assert torch.allclose(grad, torch.ones_like(grad))


# ===========================================================================
# 2. HarmonicUmbrellaBias
# ===========================================================================


class TestHarmonicUmbrellaBias:
    """Analytical energies, per-window selection, and validation."""

    def test_energy_matches_analytical(self, device: str) -> None:
        batch = _pair_batch([3.0], device=device)
        bias = HarmonicUmbrellaBias(
            cv=_cv, centers=torch.tensor([2.0]), stiffness=4.0, name="u"
        )
        result = bias.evaluate(batch)
        # 0.5 * 4 * (3 - 2)^2 = 2.0
        assert abs(float(result.energy) - 2.0) < 1e-5

    def test_zero_energy_at_center(self, device: str) -> None:
        batch = _pair_batch([2.0], device=device)
        bias = HarmonicUmbrellaBias(
            cv=_cv, centers=torch.tensor([2.0]), stiffness=7.0, name="u"
        )
        assert abs(float(bias.evaluate(batch).energy)) < 1e-6

    def test_force_pulls_toward_center(self, device: str) -> None:
        """A restraint must shorten a too-long distance, not lengthen it."""
        batch = _pair_batch([3.0], device=device)
        bias = HarmonicUmbrellaBias(
            cv=_cv, centers=torch.tensor([2.0]), stiffness=4.0, name="u"
        )
        forces = bias.evaluate(batch).forces
        # Atom 1 sits at +x of atom 0 and is too far: its force must point -x.
        assert float(forces[1, 0]) < 0
        assert float(forces[0, 0]) > 0

    def test_per_window_centers_selected_by_state_id(self, device: str) -> None:
        batch = _pair_batch([2.0, 2.0, 2.0], device=device)
        batch["thermodynamic_state_id"] = torch.tensor([0, 1, 2], device=device)
        bias = HarmonicUmbrellaBias(
            cv=_cv,
            centers=torch.tensor([[2.0], [3.0], [4.0]]),
            stiffness=2.0,
            name="u",
        )
        energy = bias.evaluate(batch).energy.reshape(-1)
        # distances all 2.0; deltas are 0, -1, -2
        expected = torch.tensor([0.0, 1.0, 4.0], device=energy.device)
        assert torch.allclose(energy, expected, atol=1e-5)

    def test_defaults_to_state_zero_without_field(self, device: str) -> None:
        batch = _pair_batch([2.0], device=device)
        bias = HarmonicUmbrellaBias(
            cv=_cv, centers=torch.tensor([[5.0], [9.0]]), stiffness=1.0, name="u"
        )
        # Uses centers[0] = 5.0 -> 0.5 * 1 * (2-5)^2 = 4.5
        assert abs(float(bias.evaluate(batch).energy) - 4.5) < 1e-5

    def test_out_of_range_state_id_raises(self, device: str) -> None:
        batch = _pair_batch([2.0], device=device)
        batch["thermodynamic_state_id"] = torch.tensor([5], device=device)
        bias = HarmonicUmbrellaBias(
            cv=_cv, centers=torch.tensor([[2.0]]), stiffness=1.0, name="u"
        )
        with pytest.raises(IndexError, match="out of range"):
            bias.evaluate(batch)

    @pytest.mark.parametrize(
        "stiffness,shape",
        [(3.0, (1, 1, 1)), (torch.tensor([3.0]), (1, 1, 1))],
    )
    def test_stiffness_forms_expand(self, stiffness, shape) -> None:
        bias = HarmonicUmbrellaBias(
            cv=_cv, centers=torch.tensor([2.0]), stiffness=stiffness, name="u"
        )
        assert bias.stiffness.shape == shape

    def test_full_matrix_stiffness(self) -> None:
        k = torch.tensor([[2.0, 0.5], [0.5, 3.0]])
        bias = HarmonicUmbrellaBias(
            cv=_cv, centers=torch.zeros(2), stiffness=k, name="u"
        )
        assert torch.allclose(bias.stiffness[0], k)

    def test_asymmetric_stiffness_raises(self) -> None:
        k = torch.tensor([[1.0, 2.0], [0.0, 1.0]])
        with pytest.raises(ValueError, match="must be symmetric"):
            HarmonicUmbrellaBias(cv=_cv, centers=torch.zeros(2), stiffness=k, name="u")

    def test_negative_eigenvalue_raises(self) -> None:
        """A negative eigenvalue turns the restraint into a runaway repulsion."""
        k = torch.tensor([[1.0, 0.0], [0.0, -1.0]])
        with pytest.raises(ValueError, match="positive-semidefinite"):
            HarmonicUmbrellaBias(cv=_cv, centers=torch.zeros(2), stiffness=k, name="u")

    def test_mismatched_diagonal_stiffness_raises(self) -> None:
        with pytest.raises(ValueError, match="match the CV dimension"):
            HarmonicUmbrellaBias(
                cv=_cv, centers=torch.zeros(2), stiffness=torch.ones(3), name="u"
            )

    def test_mismatched_periods_raises(self) -> None:
        with pytest.raises(ValueError, match="match the CV dimension"):
            HarmonicUmbrellaBias(
                cv=_cv,
                centers=torch.zeros(2),
                stiffness=1.0,
                periods=torch.ones(5),
                name="u",
            )

    def test_bad_centers_rank_raises(self) -> None:
        with pytest.raises(ValueError, match=r"centers must be \[D\] or \[S, D\]"):
            HarmonicUmbrellaBias(
                cv=_cv, centers=torch.zeros(2, 2, 2), stiffness=1.0, name="u"
            )

    def test_buffers_move_with_module(self) -> None:
        bias = HarmonicUmbrellaBias(
            cv=_cv, centers=torch.tensor([2.0]), stiffness=1.0, name="u"
        )
        assert "centers" in dict(bias.named_buffers())
        assert "stiffness" in dict(bias.named_buffers())

    def test_state_dict_round_trip(self) -> None:
        bias = HarmonicUmbrellaBias(
            cv=_cv, centers=torch.tensor([2.0]), stiffness=3.0, name="u"
        )
        other = HarmonicUmbrellaBias(
            cv=_cv, centers=torch.tensor([9.0]), stiffness=1.0, name="u"
        )
        other.load_state_dict(bias.state_dict())
        assert torch.allclose(other.centers, bias.centers)
        assert torch.allclose(other.stiffness, bias.stiffness)


# ===========================================================================
# 3. Walls
# ===========================================================================


class TestWalls:
    """One- and two-sided CV penalties."""

    def test_upper_wall_zero_inside(self, device: str) -> None:
        batch = _pair_batch([2.0], device=device)
        wall = UpperWall(cv=_cv, threshold=5.0, stiffness=10.0)
        assert abs(float(wall.evaluate(batch).energy)) < 1e-8

    def test_upper_wall_penalises_outside(self, device: str) -> None:
        batch = _pair_batch([7.0], device=device)
        wall = UpperWall(cv=_cv, threshold=5.0, stiffness=10.0)
        # (10/2) * (7-5)^2 = 20
        assert abs(float(wall.evaluate(batch).energy) - 20.0) < 1e-4

    def test_upper_wall_pushes_inward(self, device: str) -> None:
        batch = _pair_batch([7.0], device=device)
        forces = UpperWall(cv=_cv, threshold=5.0, stiffness=10.0).evaluate(batch).forces
        assert float(forces[1, 0]) < 0, "upper wall must pull the pair closer"

    def test_lower_wall_zero_outside(self, device: str) -> None:
        batch = _pair_batch([7.0], device=device)
        wall = LowerWall(cv=_cv, threshold=5.0, stiffness=10.0)
        assert abs(float(wall.evaluate(batch).energy)) < 1e-8

    def test_lower_wall_pushes_outward(self, device: str) -> None:
        batch = _pair_batch([2.0], device=device)
        forces = LowerWall(cv=_cv, threshold=5.0, stiffness=10.0).evaluate(batch).forces
        assert float(forces[1, 0]) > 0, "lower wall must push the pair apart"

    def test_wall_inside_gives_zero_forces_not_an_error(self, device: str) -> None:
        """The clamp keeps the graph connected where the wall is inactive.

        A wall written as ``if inside: return zeros(B, 1)`` would produce an
        energy with no grad_fn, which autograd rejects outright.
        """
        batch = _pair_batch([2.0], device=device)
        result = UpperWall(cv=_cv, threshold=5.0).evaluate(batch)
        assert result.forces is not None
        assert torch.count_nonzero(result.forces) == 0

    def test_force_continuous_across_boundary(self, device: str) -> None:
        """Quadratic walls have zero force at the wall; no impulse."""
        wall = UpperWall(cv=_cv, threshold=5.0, stiffness=10.0, exponent=2.0)
        just_inside = wall.evaluate(_pair_batch([4.999], device=device)).forces
        just_outside = wall.evaluate(_pair_batch([5.001], device=device)).forces
        assert torch.allclose(just_inside, just_outside, atol=1e-2)

    def test_exponent_below_one_raises(self) -> None:
        with pytest.raises(ValueError, match="exponent must be >= 1"):
            UpperWall(cv=_cv, threshold=1.0, exponent=0.5)

    def test_negative_stiffness_raises(self) -> None:
        with pytest.raises(ValueError, match="stiffness must be non-negative"):
            LowerWall(cv=_cv, threshold=1.0, stiffness=-5.0)

    def test_flat_bottom_zero_inside(self, device: str) -> None:
        batch = _pair_batch([3.0], device=device)
        restraint = FlatBottomRestraint(cv=_cv, lower=2.0, upper=5.0, stiffness=10.0)
        assert abs(float(restraint.evaluate(batch).energy)) < 1e-8

    def test_flat_bottom_penalises_both_sides(self, device: str) -> None:
        restraint = FlatBottomRestraint(cv=_cv, lower=2.0, upper=5.0, stiffness=10.0)
        below = restraint.evaluate(_pair_batch([1.0])).energy
        above = restraint.evaluate(_pair_batch([6.0])).energy
        assert abs(float(below) - 5.0) < 1e-4  # (10/2)*(2-1)^2
        assert abs(float(above) - 5.0) < 1e-4  # (10/2)*(6-5)^2

    def test_flat_bottom_matches_two_walls(self, device: str) -> None:
        batch = _pair_batch([6.5], device=device)
        combined = FlatBottomRestraint(
            cv=_cv, lower=2.0, upper=5.0, stiffness=3.0
        ).evaluate(batch)
        upper = UpperWall(cv=_cv, threshold=5.0, stiffness=3.0).evaluate(batch)
        lower = LowerWall(cv=_cv, threshold=2.0, stiffness=3.0).evaluate(batch)
        assert abs(float(combined.energy) - float(upper.energy + lower.energy)) < 1e-5

    def test_inverted_bounds_raise(self) -> None:
        with pytest.raises(ValueError, match="strictly below"):
            FlatBottomRestraint(cv=_cv, lower=5.0, upper=2.0)

    def test_wall_finite_difference_force(self, device: str) -> None:
        """Autograd wall forces agree with central differences."""
        eps = 1e-4
        wall = UpperWall(cv=_cv, threshold=3.0, stiffness=6.0)
        batch = _pair_batch([4.0], device=device)
        analytic = wall.evaluate(batch).forces[1, 0]

        plus = wall.evaluate(_pair_batch([4.0 + eps], device=device)).energy
        minus = wall.evaluate(_pair_batch([4.0 - eps], device=device)).energy
        numeric = -(float(plus) - float(minus)) / (2 * eps)
        assert abs(float(analytic) - numeric) < 1e-2


# ===========================================================================
# 4. torch.compile on the built-ins
# ===========================================================================


class TestBuiltinBiasCompile:
    """``compile_biases=True`` hands ``energy()`` to ``torch.compile``.

    That makes every built-in's ``energy()`` a compiled path in practice, so
    a data-dependent Python branch there is a real defect rather than a
    stylistic one — it breaks ``fullgraph=True`` outright.  These are
    regressions against that, per built-in.
    """

    @staticmethod
    def _state_batch(device: str) -> Batch:
        """Two graphs at distance 3.0, in windows 0 and 1."""
        batch = _pair_batch([3.0, 3.0], device=device)
        batch["thermodynamic_state_id"] = torch.tensor([0, 1], device=device)
        return batch

    @staticmethod
    def _umbrella(device: str = "cpu") -> HarmonicUmbrellaBias:
        """Calling energy() directly bypasses evaluate()'s device alignment."""
        return HarmonicUmbrellaBias(
            cv=_cv,
            centers=torch.tensor([[2.0], [3.0]]),
            stiffness=4.0,
            name="u",
        ).to(device)

    def test_umbrella_energy_compiles_fullgraph_with_state_ids(
        self, device: str
    ) -> None:
        """Per-state selection must not introduce a data-dependent branch."""
        torch._dynamo.reset()
        batch = self._state_batch(device)
        bias = self._umbrella(device)
        compiled = torch.compile(bias.energy, fullgraph=True)
        energy = compiled(batch)
        assert energy.shape == (2, 1)

    def test_umbrella_compiled_matches_eager(self, device: str) -> None:
        torch._dynamo.reset()
        batch = self._state_batch(device)
        bias = self._umbrella(device)
        eager = bias.energy(batch)
        compiled = torch.compile(bias.energy, fullgraph=True)(batch)
        assert torch.allclose(eager, compiled, atol=1e-6)
        # window 0: 0.5*4*(3-2)^2 = 2.0 ; window 1 sits at its center
        assert torch.allclose(
            eager.flatten(), torch.tensor([2.0, 0.0], device=eager.device), atol=1e-5
        )

    def test_umbrella_compiles_without_state_ids(self, device: str) -> None:
        torch._dynamo.reset()
        bias = self._umbrella(device)
        compiled = torch.compile(bias.energy, fullgraph=True)
        assert compiled(_pair_batch([3.0], device=device)).shape == (1, 1)

    def test_state_id_validation_survives_compilation(self, device: str) -> None:
        """Moving the check to evaluate() must not have removed it.

        The bounds check cannot live in energy(), but hoisting it to the eager
        evaluate() means it still runs when energy() is compiled — strictly
        better than the eager-only guards elsewhere, which skip under compile.
        """
        torch._dynamo.reset()
        batch = _pair_batch([3.0], device=device)
        batch["thermodynamic_state_id"] = torch.tensor([7], device=device)
        bias = self._umbrella(device)
        bias.energy = torch.compile(bias.energy, fullgraph=True)
        with pytest.raises(IndexError, match="out of range"):
            bias.evaluate(batch)

    def test_state_id_error_names_graph_and_valid_range(self, device: str) -> None:
        batch = _pair_batch([3.0, 3.0], device=device)
        batch["thermodynamic_state_id"] = torch.tensor([0, 9], device=device)
        with pytest.raises(IndexError) as excinfo:
            self._umbrella().evaluate(batch)
        message = str(excinfo.value)
        assert "[1]" in message  # the offending graph
        assert "0..1" in message  # the valid range

    @pytest.mark.parametrize("wall_factory", ["upper", "lower", "flat"])
    def test_wall_energy_compiles_fullgraph(
        self, wall_factory: str, device: str
    ) -> None:
        torch._dynamo.reset()
        walls = {
            "upper": lambda: UpperWall(cv=_cv, threshold=2.0, stiffness=6.0),
            "lower": lambda: LowerWall(cv=_cv, threshold=4.0, stiffness=6.0),
            "flat": lambda: FlatBottomRestraint(
                cv=_cv, lower=2.0, upper=4.0, stiffness=6.0
            ),
        }
        bias = walls[wall_factory]().to(device)
        batch = _pair_batch([3.0, 5.0], device=device)
        eager = bias.energy(batch)
        compiled = torch.compile(bias.energy, fullgraph=True)(batch)
        assert torch.allclose(eager, compiled, atol=1e-6)

    def test_runner_compile_biases_end_to_end(self, device: str) -> None:
        """The path a user actually takes: compile_biases=True on the runner."""
        from nvalchemi.dynamics import NVTLangevin
        from nvalchemi.enhanced_sampling import EnhancedSampling
        from nvalchemi.models.demo import DemoModel, DemoModelWrapper

        torch._dynamo.reset()

        def make_batch() -> Batch:
            data_list = []
            for _ in range(2):
                data = AtomicData(
                    positions=torch.tensor([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]]),
                    atomic_numbers=torch.tensor([6, 6], dtype=torch.long),
                    atomic_masses=torch.ones(2),
                    forces=torch.zeros(2, 3),
                    energy=torch.zeros(1, 1),
                )
                data.add_node_property("velocities", torch.zeros(2, 3))
                data_list.append(data)
            batch = Batch.from_data_list(data_list).to(device)
            batch["thermodynamic_state_id"] = torch.tensor([0, 1], device=device)
            return batch

        results = []
        for compile_biases in (False, True):
            batch = make_batch()
            # DemoModel has random weights; seed so the physical contribution
            # is identical and any difference is attributable to the bias.
            torch.manual_seed(0)
            model = DemoModelWrapper(DemoModel()).to(device)
            dynamics = NVTLangevin(model=model, dt=0.1, temperature=300.0, friction=0.1)
            runner = EnhancedSampling(
                dynamics, {"u": self._umbrella()}, compile_biases=compile_biases
            )
            runner.prime_forces(batch)
            results.append(batch.forces.clone())
        assert torch.allclose(results[0], results[1], atol=1e-5)
