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
"""Unit tests for :class:`AdaptiveBiasingForce`.

The estimator is checked against a system whose potential of mean force is
known in closed form — a harmonic pair in three dimensions, where

``dA/dr = k (r - r0) - 2 kB T / r``

holds *pointwise*, so a single sample must reproduce it exactly rather than
only on average.  That makes the metric correction directly testable: drop
it and the ideal-gas limit reports a flat PMF instead of ``-2 kB T ln r``.
"""

from __future__ import annotations

import pytest
import torch

from nvalchemi.data import AtomicData, Batch
from nvalchemi.dynamics import NVTLangevin
from nvalchemi.dynamics.base import DynamicsStage
from nvalchemi.dynamics.hooks._utils import KB_EV
from nvalchemi.enhanced_sampling import (
    AdaptiveBiasingForce,
    BiasResult,
    EnhancedSampling,
    ReplicaExchange,
    ThermodynamicState,
    pair_displacement,
    pair_distance,
)
from nvalchemi.models.demo import DemoModel, DemoModelWrapper

TEMPERATURE = 300.0
KT = KB_EV * TEMPERATURE
SPRING = 3.0
REST_LENGTH = 2.0

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _harmonic_frame(
    distances: list[float], device: str = "cpu", forces: bool = True
) -> Batch:
    """Return pairs at *distances* carrying the harmonic physical force.

    ``U = 0.5 k (r - r0)^2`` gives ``F_j = -k (r - r0) u`` and
    ``F_i = +k (r - r0) u``, with the pair laid out along ``x``.
    """
    items = []
    for r in distances:
        kwargs = {
            "positions": torch.tensor([[0.0, 0.0, 0.0], [r, 0.0, 0.0]]),
            "atomic_numbers": torch.ones(2, dtype=torch.long),
        }
        if forces:
            gradient = SPRING * (r - REST_LENGTH)
            kwargs["forces"] = torch.tensor(
                [[gradient, 0.0, 0.0], [-gradient, 0.0, 0.0]]
            )
        items.append(AtomicData(**kwargs))
    return Batch.from_data_list(items).to(device)


def _abf(device: str = "cpu", **kwargs) -> AdaptiveBiasingForce:
    """Return an ABF bias with the threshold disabled unless overridden."""
    params = {
        "atom_indices": torch.tensor([0, 1]),
        "temperature": TEMPERATURE,
        "cv_range": (1.0, 4.0),
        "n_bins": 60,
        "min_samples": 0,
        "full_samples": 0,
    }
    params.update(kwargs)
    return AdaptiveBiasingForce(**params).to(device)


def _analytic_gradient(r: float) -> float:
    """Return the exact ``dA/dr`` for the harmonic pair."""
    return SPRING * (r - REST_LENGTH) - 2.0 * KT / r


def _runner_batch(
    n_graphs: int = 2, atoms: int = 4, device: str = "cpu", seed: int = 0
) -> Batch:
    """Return a batch with the buffers dynamics writes back into."""
    torch.manual_seed(seed)
    items = []
    for _ in range(n_graphs):
        data = AtomicData(
            positions=torch.randn(atoms, 3),
            atomic_numbers=torch.full((atoms,), 6, dtype=torch.long),
            atomic_masses=torch.ones(atoms),
            forces=torch.zeros(atoms, 3),
            energy=torch.zeros(1, 1),
        )
        data.add_node_property("velocities", torch.zeros(atoms, 3))
        items.append(data)
    return Batch.from_data_list(items).to(device)


def _make_dynamics(device: str = "cpu") -> NVTLangevin:
    """Return a demo-model Langevin integrator."""
    model = DemoModelWrapper(DemoModel()).to(device)
    return NVTLangevin(model=model, dt=0.1, temperature=TEMPERATURE, friction=0.1)


# ===========================================================================
# 1. Construction
# ===========================================================================


class TestConstruction:
    """Constructor validation."""

    @pytest.mark.parametrize("indices", [[0], [0, 1, 2]])
    def test_wrong_pair_size_raises(self, indices: list[int]) -> None:
        with pytest.raises(ValueError, match="exactly two atoms"):
            _abf(atom_indices=torch.tensor(indices))

    def test_wrong_rank_raises(self) -> None:
        with pytest.raises(ValueError, match=r"shape \[2\] or \[B, 2\]"):
            _abf(atom_indices=torch.zeros(2, 2, 2, dtype=torch.long))

    def test_negative_index_raises(self) -> None:
        with pytest.raises(ValueError, match="must be non-negative"):
            _abf(atom_indices=torch.tensor([0, -1]))

    def test_self_pair_raises(self) -> None:
        """A zero-length CV has no direction to project onto."""
        with pytest.raises(ValueError, match="same atom twice"):
            _abf(atom_indices=torch.tensor([2, 2]))

    @pytest.mark.parametrize("temperature", [0.0, -10.0])
    def test_non_positive_temperature_raises(self, temperature: float) -> None:
        """Temperature scales the metric correction, so it must be real."""
        with pytest.raises(ValueError, match="temperature must be positive"):
            _abf(temperature=temperature)

    @pytest.mark.parametrize("cv_range", [(4.0, 1.0), (2.0, 2.0)])
    def test_inverted_range_raises(self, cv_range: tuple[float, float]) -> None:
        with pytest.raises(ValueError, match="cv_range must be increasing"):
            _abf(cv_range=cv_range)

    def test_negative_lower_bound_raises(self) -> None:
        with pytest.raises(ValueError, match="lower bound must be non-negative"):
            _abf(cv_range=(-1.0, 3.0))

    def test_zero_bins_raises(self) -> None:
        with pytest.raises(ValueError, match="n_bins must be at least 1"):
            _abf(n_bins=0)

    def test_full_below_min_samples_raises(self) -> None:
        with pytest.raises(ValueError, match="must be at least min_samples"):
            _abf(min_samples=100, full_samples=50)

    def test_full_samples_defaults_to_double(self) -> None:
        assert _abf(min_samples=50, full_samples=None).full_samples == 100

    def test_non_positive_max_force_raises(self) -> None:
        with pytest.raises(ValueError, match="max_force must be positive"):
            _abf(max_force=0.0)

    def test_zero_update_frequency_raises(self) -> None:
        with pytest.raises(ValueError, match="update_frequency must be at least 1"):
            _abf(update_frequency=0)

    def test_mixin_order_is_correct(self) -> None:
        """AdaptivePotentialMixin must precede nn.Module in the MRO."""
        from nvalchemi.enhanced_sampling import AdaptivePotentialMixin

        mro = AdaptiveBiasingForce.__mro__
        assert mro.index(AdaptivePotentialMixin) < mro.index(torch.nn.Module)

    def test_is_not_a_conservative_bias(self) -> None:
        """ABF has no energy to differentiate, so it is not a model."""
        from nvalchemi.enhanced_sampling import ConservativeBias

        assert not issubclass(AdaptiveBiasingForce, ConservativeBias)

    def test_satisfies_the_bias_protocol(self) -> None:
        from nvalchemi.enhanced_sampling import BiasPotential

        assert isinstance(_abf(), BiasPotential)


# ===========================================================================
# 2. The estimator, against a closed-form PMF
# ===========================================================================


class TestEstimator:
    """``dA/dr = k (r - r0) - 2 kB T / r``, exactly, for a harmonic pair."""

    @pytest.mark.parametrize("distance", [1.6, 2.0, 2.9, 3.5])
    def test_single_sample_is_exact(self, device: str, distance: float) -> None:
        """The harmonic pair makes the estimator exact pointwise, not just
        on average — one sample must land on the analytic value."""
        bias = _abf(device)
        bias.update(_harmonic_frame([distance], device), BiasResult())

        index = int(bias.bin_index(torch.tensor([distance]))[0])
        assert float(bias.mean_force()[index]) == pytest.approx(
            _analytic_gradient(distance), rel=1e-5
        )

    def test_metric_correction_is_applied(self, device: str) -> None:
        """Two non-interacting particles have PMF ``-2 kB T ln r``.

        A naive Cartesian projection reports zero mean force here — a flat
        PMF — which is not noise but a smoothly wrong answer.
        """
        bias = _abf(device)
        frame = _harmonic_frame([2.5], device)
        frame.forces = torch.zeros_like(frame.forces)
        bias.update(frame, BiasResult())

        index = int(bias.bin_index(torch.tensor([2.5]))[0])
        got = float(bias.mean_force()[index])
        assert got == pytest.approx(-2.0 * KT / 2.5, rel=1e-5)
        assert got != pytest.approx(0.0, abs=1e-6)

    def test_metric_correction_scales_with_temperature(self, device: str) -> None:
        """The Jacobian term is ``2 kB T / r``, linear in T."""
        gradients = []
        for temperature in (300.0, 600.0):
            bias = _abf(device, temperature=temperature)
            frame = _harmonic_frame([2.5], device)
            frame.forces = torch.zeros_like(frame.forces)
            bias.update(frame, BiasResult())
            index = int(bias.bin_index(torch.tensor([2.5]))[0])
            gradients.append(float(bias.mean_force()[index]))
        assert gradients[1] == pytest.approx(2.0 * gradients[0], rel=1e-5)

    def test_samples_average_within_a_bin(self, device: str) -> None:
        bias = _abf(device, n_bins=1, cv_range=(1.0, 4.0))
        bias.update(_harmonic_frame([1.5, 2.5, 3.5], device), BiasResult())

        expected = sum(_analytic_gradient(r) for r in (1.5, 2.5, 3.5)) / 3.0
        assert int(bias.bin_counts[0]) == 3
        assert float(bias.mean_force()[0]) == pytest.approx(expected, rel=1e-5)

    def test_unvisited_bins_are_nan_not_zero(self, device: str) -> None:
        """Zero is a plausible mean force, so it cannot mean "no data"."""
        bias = _abf(device)
        bias.update(_harmonic_frame([2.0], device), BiasResult())
        estimate = bias.mean_force()
        assert bool(torch.isnan(estimate).any())
        assert int((~torch.isnan(estimate)).sum()) == 1

    def test_out_of_range_samples_are_discarded(self, device: str) -> None:
        bias = _abf(device, cv_range=(2.0, 3.0), n_bins=10)
        bias.update(_harmonic_frame([1.0, 2.5, 5.0], device), BiasResult())
        assert int(bias.bin_counts.sum()) == 1

    def test_update_without_forces_raises(self, device: str) -> None:
        """There is nothing to project if the frame carries no forces."""
        bias = _abf(device)
        with pytest.raises(ValueError, match="has no forces"):
            bias.update(_harmonic_frame([2.0], device, forces=False), BiasResult())


# ===========================================================================
# 3. The applied force
# ===========================================================================


class TestAppliedForce:
    """Direction, magnitude, and the sample threshold."""

    def test_result_is_force_only(self, device: str) -> None:
        """No energy: the applied force is not the gradient of anything held."""
        bias = _abf(device)
        bias.update(_harmonic_frame([2.5], device), BiasResult())
        result = bias.evaluate(_harmonic_frame([2.5], device))

        assert result.energy is None
        assert result.stress is None
        assert result.virial is None
        assert result.forces is not None

    def test_force_opposes_the_mean_force(self, device: str) -> None:
        """The bias must cancel the drift, not reinforce it.

        For an ideal pair ``dA/dr = -2 kB T / r < 0``, so the free energy
        falls with separation and the entropic drift is outward; the bias
        must therefore pull inward.
        """
        bias = _abf(device)
        frame = _harmonic_frame([2.5], device)
        frame.forces = torch.zeros_like(frame.forces)
        bias.update(frame, BiasResult())

        forces = bias.evaluate(_harmonic_frame([2.5], device)).forces
        assert float(forces[1, 0]) < 0.0
        assert float(forces[0, 0]) > 0.0

    def test_applied_force_equals_the_estimate(self, device: str) -> None:
        bias = _abf(device)
        bias.update(_harmonic_frame([2.9], device), BiasResult())
        forces = bias.evaluate(_harmonic_frame([2.9], device)).forces
        assert float(forces[1, 0]) == pytest.approx(_analytic_gradient(2.9), rel=1e-5)

    def test_bias_exerts_no_net_force(self, device: str) -> None:
        """Equal and opposite along the pair: no spurious translation."""
        bias = _abf(device)
        bias.update(_harmonic_frame([2.5, 3.1], device), BiasResult())
        forces = bias.evaluate(_harmonic_frame([2.5, 3.1], device)).forces
        assert float(forces.sum(dim=0).abs().max()) < 1e-9

    def test_no_force_before_the_threshold(self, device: str) -> None:
        """An estimate from a handful of samples is noise."""
        bias = _abf(device, min_samples=4, full_samples=8)
        bias.update(_harmonic_frame([2.5], device), BiasResult())
        result = bias.evaluate(_harmonic_frame([2.5], device))
        assert torch.count_nonzero(result.forces) == 0

    def test_force_ramps_between_the_thresholds(self, device: str) -> None:
        """A jump to the full estimate would be the discontinuity the
        threshold exists to avoid."""
        bias = _abf(device, min_samples=2, full_samples=6)
        magnitudes = []
        for _ in range(6):
            bias.update(_harmonic_frame([2.9], device), BiasResult())
            magnitudes.append(
                abs(float(bias.evaluate(_harmonic_frame([2.9], device)).forces[1, 0]))
            )

        assert magnitudes[0] == pytest.approx(0.0, abs=1e-12)
        assert magnitudes == sorted(magnitudes)
        assert magnitudes[-1] == pytest.approx(abs(_analytic_gradient(2.9)), rel=1e-5)

    def test_ramp_fraction_endpoints(self, device: str) -> None:
        bias = _abf(device, min_samples=2, full_samples=4, n_bins=1)
        assert float(bias.ramp_fraction()[0]) == 0.0
        for _ in range(4):
            bias.update(_harmonic_frame([2.5], device), BiasResult())
        assert float(bias.ramp_fraction()[0]) == pytest.approx(1.0)

    def test_out_of_range_walkers_feel_nothing(self, device: str) -> None:
        bias = _abf(device, cv_range=(2.0, 3.0), n_bins=10)
        bias.update(_harmonic_frame([2.5], device), BiasResult())
        result = bias.evaluate(_harmonic_frame([2.5, 9.0], device))
        assert float(result.forces[2:].abs().max()) == 0.0
        assert float(result.forces[:2].abs().max()) > 0.0

    def test_max_force_caps_the_estimate(self, device: str) -> None:
        """One visit at a bad geometry cannot dominate the trajectory."""
        bias = _abf(device, max_force=0.1)
        bias.update(_harmonic_frame([3.5], device), BiasResult())
        forces = bias.evaluate(_harmonic_frame([3.5], device)).forces
        assert abs(float(forces[1, 0])) == pytest.approx(0.1, rel=1e-6)

    def test_evaluate_does_not_mutate_state(self, device: str) -> None:
        """``evaluate`` is read-only; only ``update`` changes the estimate."""
        bias = _abf(device)
        bias.update(_harmonic_frame([2.5], device), BiasResult())
        before = (bias.bin_counts.clone(), bias.force_sum.clone(), bias.state_version)

        for _ in range(3):
            bias.evaluate(_harmonic_frame([2.5], device))

        assert torch.equal(bias.bin_counts, before[0])
        assert torch.equal(bias.force_sum, before[1])
        assert bias.state_version == before[2]

    def test_diagnostics_are_reported(self, device: str) -> None:
        bias = _abf(device)
        bias.update(_harmonic_frame([2.5], device), BiasResult())
        result = bias.evaluate(_harmonic_frame([2.5, 9.0], device))

        assert set(result.observables) >= {
            "cv",
            "applied_gradient",
            "samples",
            "ramp",
            "in_range",
        }
        # Deliberately not called "mean_force": this is the ramped, capped
        # value, which mean_force() is not.
        assert "mean_force" not in result.observables
        assert float(result.observables["cv"][0]) == pytest.approx(2.5, abs=1e-6)
        assert result.observables["in_range"].reshape(-1).tolist() == [1.0, 0.0]


# ===========================================================================
# 4. Free energy
# ===========================================================================


class TestFreeEnergy:
    """Integrating the estimate recovers the analytic PMF."""

    def test_profile_matches_the_analytic_pmf(self, device: str) -> None:
        """``A(r) = 0.5 k (r - r0)^2 - 2 kB T ln r``, up to a constant.

        Nothing is deconvolved: what ABF accumulates already *is* the
        free-energy gradient.
        """
        bias = _abf(device, cv_range=(1.5, 3.0), n_bins=30)
        centers = bias.bin_centers
        bias.update(_harmonic_frame(centers.tolist(), device), BiasResult())

        profile = bias.free_energy()
        exact = 0.5 * SPRING * (centers - REST_LENGTH) ** 2 - 2 * KT * torch.log(
            centers
        )
        exact = exact - exact.min()
        # Free energy is defined up to an additive constant.
        profile = profile - (profile - exact).mean()
        assert float((profile - exact).abs().max()) < 5e-4

    def test_unsampled_bins_are_nan(self, device: str) -> None:
        bias = _abf(device, cv_range=(1.0, 4.0), n_bins=30)
        centers = bias.bin_centers
        bias.update(_harmonic_frame(centers[5:20].tolist(), device), BiasResult())

        profile = bias.free_energy()
        assert bool(torch.isnan(profile[:5]).all())
        assert bool(torch.isfinite(profile[5:20]).all())
        assert bool(torch.isnan(profile[20:]).all())

    def test_interior_gap_raises(self, device: str) -> None:
        """Integration carries the profile across a hole, so every value
        beyond it would be wrong by an unknown constant."""
        bias = _abf(device, cv_range=(1.0, 4.0), n_bins=30)
        centers = bias.bin_centers
        sampled = centers[[5, 6, 7, 20, 21]].tolist()
        bias.update(_harmonic_frame(sampled, device), BiasResult())

        with pytest.raises(RuntimeError, match="never visited but lie between"):
            bias.free_energy()

    def test_no_samples_raises(self, device: str) -> None:
        with pytest.raises(RuntimeError, match="no bin has been sampled"):
            _abf(device).free_energy()

    def test_single_bin_is_flat(self, device: str) -> None:
        bias = _abf(device, n_bins=1, cv_range=(1.0, 4.0))
        bias.update(_harmonic_frame([2.5], device), BiasResult())
        assert float(bias.free_energy()[0]) == 0.0


# ===========================================================================
# 5. Runner integration and observation ordering
# ===========================================================================


class TestRunnerIntegration:
    """The runner must hand ABF unbiased forces, exactly once per due step."""

    def test_observation_stage_is_after_compute(self) -> None:
        assert _abf().observation_stage is DynamicsStage.AFTER_COMPUTE

    def test_observes_physical_not_total_forces(self, device: str) -> None:
        """An estimator fed its own output converges to what it already said.

        This is the whole reason ABF observes at ``AFTER_COMPUTE``.
        """
        observed: list[torch.Tensor] = []

        class Spy(AdaptiveBiasingForce):
            def update(self, frames: Batch, result: BiasResult) -> None:
                observed.append(frames.forces.clone())
                super().update(frames, result)

        bias = Spy(
            atom_indices=torch.tensor([0, 3]),
            temperature=TEMPERATURE,
            cv_range=(0.5, 6.0),
            n_bins=40,
            min_samples=0,
            full_samples=0,
            name="abf",
        ).to(device)
        runner = EnhancedSampling(_make_dynamics(device), {"abf": bias})
        runner.run(_runner_batch(device=device), n_steps=3)

        physical = runner.last_outputs["physical/forces"]
        total = runner.last_outputs["total/forces"]
        assert float(runner.last_outputs["bias/abf/forces"].abs().max()) > 0.0
        assert torch.allclose(observed[-1], physical, atol=1e-6)
        assert not torch.allclose(observed[-1], total, atol=1e-6)

    def test_contributes_no_energy_to_the_total(self, device: str) -> None:
        bias = _abf(device, atom_indices=torch.tensor([0, 3]), name="abf")
        runner = EnhancedSampling(_make_dynamics(device), {"abf": bias})
        runner.run(_runner_batch(device=device), n_steps=4)

        outputs = runner.last_outputs
        assert "bias/abf/energy" not in outputs
        assert torch.allclose(outputs["total/energy"], outputs["physical/energy"])

    def test_samples_accumulate_over_a_run(self, device: str) -> None:
        bias = _abf(device, atom_indices=torch.tensor([0, 3]), name="abf")
        runner = EnhancedSampling(_make_dynamics(device), {"abf": bias})
        batch = _runner_batch(device=device)
        runner.run(batch, n_steps=6)

        # One sample per walker per step; two walkers.
        assert int(bias.bin_counts.sum()) == 6 * batch.num_graphs

    def test_update_frequency_is_respected(self, device: str) -> None:
        bias = _abf(
            device, atom_indices=torch.tensor([0, 3]), name="abf", update_frequency=3
        )
        runner = EnhancedSampling(_make_dynamics(device), {"abf": bias})
        batch = _runner_batch(device=device)
        runner.run(batch, n_steps=9)
        assert int(bias.bin_counts.sum()) == 3 * batch.num_graphs

    def test_no_sampling_during_priming(self, device: str) -> None:
        bias = _abf(device, atom_indices=torch.tensor([0, 3]), name="abf")
        runner = EnhancedSampling(_make_dynamics(device), {"abf": bias})
        runner.prime_forces(_runner_batch(device=device))
        assert int(bias.bin_counts.sum()) == 0

    def test_below_threshold_updates_do_not_bump_the_version(self, device: str) -> None:
        """Re-priming forces over a bin that applies nothing is pure cost.

        This is the case ``bump_state_version`` documents.
        """
        bias = _abf(device, min_samples=1000, full_samples=2000)
        bias.update(_harmonic_frame([2.5], device), BiasResult())
        assert bias.state_version == 0

    def test_above_threshold_updates_bump_the_version(self, device: str) -> None:
        bias = _abf(device, min_samples=1, full_samples=2)
        bias.update(_harmonic_frame([2.5], device), BiasResult())
        bias.update(_harmonic_frame([2.5], device), BiasResult())
        assert bias.state_version > 0

    def test_composes_with_a_conservative_bias(self, device: str) -> None:
        """ABF's force-only result must sum with an energy-carrying one."""
        from nvalchemi.enhanced_sampling import UpperWall

        indices = torch.tensor([0, 3])
        wall = UpperWall(
            cv=lambda b: pair_distance(b, indices),
            threshold=0.5,
            stiffness=5.0,
            name="wall",
        ).to(device)
        bias = _abf(device, atom_indices=indices, name="abf")
        runner = EnhancedSampling(_make_dynamics(device), {"abf": bias, "wall": wall})
        runner.run(_runner_batch(device=device), n_steps=3)

        outputs = runner.last_outputs
        assert float(outputs["bias/wall/energy"].abs().sum()) > 0.0
        assert torch.allclose(
            outputs["total/energy"],
            outputs["physical/energy"] + outputs["bias_total/energy"],
            atol=1e-5,
        )


# ===========================================================================
# 6. Replica-exchange rejection
# ===========================================================================


class TestExchangeRejection:
    """A force-only bias has no cross-state energy for the acceptance rule."""

    @staticmethod
    def _ladder() -> ReplicaExchange:
        states = [
            ThermodynamicState(state_id=i, temperature=t)
            for i, t in enumerate((300.0, 350.0))
        ]
        return ReplicaExchange(
            states=states,
            attempt_interval=2,
            initial_state_ids=torch.tensor([0, 1]),
        )

    def test_declares_no_exchange_energy(self) -> None:
        assert _abf().supplies_exchange_energy is False

    def test_validate_for_rejects_it(self) -> None:
        with pytest.raises(ValueError, match="supplies no exchange energy"):
            self._ladder().validate_for({"abf": _abf()})

    def test_runner_rejects_abf_with_exchange(self, device: str) -> None:
        bias = _abf(device, atom_indices=torch.tensor([0, 3]), name="abf")
        with pytest.raises(ValueError, match="supplies no exchange energy"):
            EnhancedSampling(
                _make_dynamics(device),
                {"abf": bias},
                replica_exchange=self._ladder(),
            )


# ===========================================================================
# 7. Restart
# ===========================================================================


class TestRestart:
    """Bin counts and accumulated statistics survive a round trip."""

    def test_state_dict_round_trip(self, device: str) -> None:
        bias = _abf(device)
        bias.update(_harmonic_frame([1.6, 2.5, 3.1], device), BiasResult())

        restored = _abf(device)
        restored.load_state_dict(bias.state_dict())

        assert torch.equal(restored.bin_counts, bias.bin_counts)
        assert torch.allclose(restored.force_sum, bias.force_sum)
        assert restored.state_version == bias.state_version
        assert torch.allclose(
            restored.evaluate(_harmonic_frame([2.5], device)).forces,
            bias.evaluate(_harmonic_frame([2.5], device)).forces,
        )

    def test_restart_continues_averaging(self, device: str) -> None:
        """A restored run must extend the average, not restart it."""
        bias = _abf(device, n_bins=1, cv_range=(1.0, 4.0))
        bias.update(_harmonic_frame([1.5, 2.5], device), BiasResult())

        restored = _abf(device, n_bins=1, cv_range=(1.0, 4.0))
        restored.load_state_dict(bias.state_dict())
        restored.update(_harmonic_frame([3.5], device), BiasResult())
        bias.update(_harmonic_frame([3.5], device), BiasResult())

        assert int(restored.bin_counts[0]) == 3
        assert float(restored.mean_force()[0]) == pytest.approx(
            float(bias.mean_force()[0]), rel=1e-6
        )

    def test_restoring_a_different_cv_range_raises(self, device: str) -> None:
        """The histogram's bins mean nothing without the range that made them.

        The counts are shape-compatible, so nothing structural objects; bin
        5 simply stops meaning one distance and starts meaning another,
        carrying its accumulated mean force with it.
        """
        source = _abf(device, cv_range=(1.0, 4.0))
        source.update(_harmonic_frame([1.5, 2.5], device), BiasResult())

        target = _abf(device, cv_range=(2.0, 8.0))
        with pytest.raises(ValueError, match="cv_range"):
            target.load_state_dict(source.state_dict())

    def test_restoring_a_different_temperature_raises(self, device: str) -> None:
        """The metric correction is already folded into ``force_sum``."""
        source = _abf(device)
        source.update(_harmonic_frame([2.5], device), BiasResult())

        target = _abf(device, temperature=900.0)
        with pytest.raises(ValueError, match="temperature"):
            target.load_state_dict(source.state_dict())

    def test_restoring_a_different_atom_pair_raises(self, device: str) -> None:
        """atom_indices is a buffer, so an unchecked load would overwrite it.

        The caller's selection would be silently replaced by the
        checkpoint's — the opposite of what asking for it meant.
        """
        source = _abf(device, atom_indices=torch.tensor([0, 1]))
        source.update(_harmonic_frame([2.5], device), BiasResult())

        target = _abf(device, atom_indices=torch.tensor([5, 7]))
        with pytest.raises(ValueError, match="atom_indices"):
            target.load_state_dict(source.state_dict())
        # The rejection must come before the buffer is overwritten.
        assert target.atom_indices.tolist() == [5, 7]

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("n_bins", 20),
            ("min_samples", 7),
            ("full_samples", 99),
            ("max_force", 0.25),
        ],
    )
    def test_restoring_a_different_setting_raises(
        self, device: str, field: str, value: object
    ) -> None:
        base = {"min_samples": 1, "full_samples": 100}
        source = _abf(device, **base)
        source.update(_harmonic_frame([2.5], device), BiasResult())

        target = _abf(device, **{**base, field: value})
        with pytest.raises(ValueError, match=field):
            target.load_state_dict(source.state_dict())

    def test_the_error_names_every_difference(self, device: str) -> None:
        source = _abf(device, cv_range=(1.0, 4.0))
        source.update(_harmonic_frame([2.5], device), BiasResult())
        target = _abf(device, cv_range=(2.0, 8.0), temperature=900.0)

        with pytest.raises(ValueError) as excinfo:
            target.load_state_dict(source.state_dict())
        message = str(excinfo.value)
        assert "cv_range" in message and "temperature" in message

    def test_identical_configuration_still_restores(self, device: str) -> None:
        """The check must not reject a legitimate continuation."""
        source = _abf(device)
        source.update(_harmonic_frame([1.5, 2.5], device), BiasResult())

        target = _abf(device)
        target.load_state_dict(source.state_dict())
        assert torch.equal(target.bin_counts, source.bin_counts)

    def test_fingerprint_survives_the_zarr_round_trip(
        self, tmp_path, device: str
    ) -> None:
        """A mismatch must be caught through the runner, not only in memory."""
        bias = _abf(device, atom_indices=torch.tensor([0, 3]), name="abf")
        runner = EnhancedSampling(
            _make_dynamics(device), {"abf": bias}, steps_per_epoch=4
        )
        batch = runner.run(_runner_batch(device=device), n_steps=4)
        path = tmp_path / "abf.zarr"
        runner.checkpoint(path, batch)

        wrong = AdaptiveBiasingForce(
            atom_indices=torch.tensor([0, 3]),
            temperature=TEMPERATURE,
            cv_range=(9.0, 12.0),
            n_bins=60,
            min_samples=0,
            full_samples=0,
            name="abf",
        ).to(device)
        fresh = EnhancedSampling(
            _make_dynamics(device), {"abf": wrong}, steps_per_epoch=4
        )
        with pytest.raises(ValueError, match="cv_range"):
            fresh.restore(path, device=device)

    def test_checkpoint_round_trip_through_the_runner(
        self, tmp_path, device: str
    ) -> None:
        bias = _abf(device, atom_indices=torch.tensor([0, 3]), name="abf")
        runner = EnhancedSampling(
            _make_dynamics(device), {"abf": bias}, steps_per_epoch=4
        )
        batch = runner.run(_runner_batch(device=device), n_steps=4)

        path = tmp_path / "abf.zarr"
        runner.checkpoint(path, batch)

        fresh_bias = _abf(device, atom_indices=torch.tensor([0, 3]), name="abf")
        fresh = EnhancedSampling(
            _make_dynamics(device), {"abf": fresh_bias}, steps_per_epoch=4
        )
        fresh.restore(path, device=device)

        assert torch.equal(fresh_bias.bin_counts, bias.bin_counts)
        assert torch.allclose(fresh_bias.force_sum, bias.force_sum)
        assert fresh_bias.state_version == bias.state_version


# ===========================================================================
# 8. Geometry
# ===========================================================================


class TestGeometry:
    """ABF's CV must agree with ``pair_distance`` exactly."""

    def test_displacement_norm_is_the_distance(self, device: str) -> None:
        batch = _harmonic_frame([1.7, 2.9], device)
        indices = torch.tensor([0, 1], device=batch.positions.device)
        assert torch.allclose(
            torch.linalg.vector_norm(
                pair_displacement(batch, indices), dim=-1, keepdim=True
            ),
            pair_distance(batch, indices),
        )

    def test_per_graph_pairs_are_supported(self, device: str) -> None:
        bias = _abf(device, atom_indices=torch.tensor([[0, 1], [1, 0]]))
        bias.update(_harmonic_frame([2.5, 2.5], device), BiasResult())
        assert int(bias.bin_counts.sum()) == 2

    def test_bias_follows_the_batch_device(self, device: str) -> None:
        """A bias built before the batch moved to GPU must still work."""
        bias = _abf("cpu")
        batch = _harmonic_frame([2.5], device)
        bias.update(batch, BiasResult())
        result = bias.evaluate(batch)
        assert result.forces.device.type == batch.positions.device.type
