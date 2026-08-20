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
"""Unit tests for the two metadynamics biases.

Covers :class:`WellTemperedMetaDynamicsBias` (hill scaling, the three storage
policies, multi-walker history, periodic CVs, ramping, free energy, restart)
and :class:`RMSDMetaDynamicsBias` (alignment invariance, warm starts, atom
selection, FIFO retention, periodic rejection, restart), plus the deposition
schedule both get from :class:`EnhancedSampling`.
"""

from __future__ import annotations

import math

import pytest
import torch
from torch import Tensor

from nvalchemi.data import AtomicData, Batch
from nvalchemi.dynamics import NVTLangevin
from nvalchemi.dynamics.hooks._utils import KB_EV
from nvalchemi.enhanced_sampling import (
    EnhancedSampling,
    RMSDMetaDynamicsBias,
    WellTemperedMetaDynamicsBias,
    pair_distance,
)
from nvalchemi.enhanced_sampling.biases.rmsd_metad import _squared_rmsd
from nvalchemi.models.demo import DemoModel, DemoModelWrapper

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _pair_batch(distances: list[float], device: str = "cpu") -> Batch:
    """Return one graph per entry, atoms 0 and 1 separated along x."""
    data_list = []
    for d in distances:
        positions = torch.tensor([[0.0, 0.0, 0.0], [d, 0.0, 0.0]])
        data_list.append(
            AtomicData(
                positions=positions,
                atomic_numbers=torch.ones(2, dtype=torch.long),
            )
        )
    return Batch.from_data_list(data_list).to(device)


def _random_batch(
    n_graphs: int = 2,
    atoms_per_graph: int = 4,
    device: str = "cpu",
    seed: int = 0,
) -> Batch:
    """Return a batch with the buffers dynamics writes back into."""
    torch.manual_seed(seed)
    data_list = []
    for _ in range(n_graphs):
        data = AtomicData(
            positions=torch.randn(atoms_per_graph, 3),
            atomic_numbers=torch.full((atoms_per_graph,), 6, dtype=torch.long),
            atomic_masses=torch.ones(atoms_per_graph),
            forces=torch.zeros(atoms_per_graph, 3),
            energy=torch.zeros(1, 1),
        )
        data.add_node_property("velocities", torch.zeros(atoms_per_graph, 3))
        data_list.append(data)
    return Batch.from_data_list(data_list).to(device)


def _make_dynamics(device: str = "cpu") -> NVTLangevin:
    """Return a demo-model Langevin integrator."""
    model = DemoModelWrapper(DemoModel()).to(device)
    return NVTLangevin(model=model, dt=0.1, temperature=300.0, friction=0.1)


def _pair_cv(indices: tuple[int, int] = (0, 1)):
    """Return a pair-distance CV callable over *indices*."""
    idx = torch.tensor(list(indices))
    return lambda batch: pair_distance(batch, idx)


def _metad(device: str = "cpu", **kwargs) -> WellTemperedMetaDynamicsBias:
    """Return a well-tempered bias with test-friendly defaults."""
    params = {
        "cv": _pair_cv(),
        "height": 0.05,
        "sigma": 0.2,
        "temperature": 300.0,
        "bias_factor": 10.0,
        "max_hills": 16,
    }
    params.update(kwargs)
    return WellTemperedMetaDynamicsBias(**params).to(device)


def _rot() -> Tensor:
    """Return a random proper rotation matrix."""
    q, _ = torch.linalg.qr(torch.randn(3, 3))
    if torch.det(q) < 0:
        q[:, 0] *= -1
    return q


# ===========================================================================
# 1. Well-tempered metadynamics: construction
# ===========================================================================


class TestWellTemperedConstruction:
    """Constructor validation for the well-tempered bias."""

    @pytest.mark.parametrize("height", [0.0, -0.01])
    def test_non_positive_height_raises(self, height: float) -> None:
        """A non-positive hill attracts the walker back where it has been."""
        with pytest.raises(ValueError, match="height must be positive"):
            _metad(height=height)

    @pytest.mark.parametrize("gamma", [1.0, 0.5, -2.0])
    def test_bias_factor_at_or_below_one_raises(self, gamma: float) -> None:
        """gamma = 1 divides by zero in the well-tempered height."""
        with pytest.raises(ValueError, match="bias_factor must be greater"):
            _metad(bias_factor=gamma)

    @pytest.mark.parametrize("sigma", [0.0, -0.3, [0.2, 0.0]])
    def test_non_positive_sigma_raises(self, sigma) -> None:
        """A zero width is a delta function with no usable gradient."""
        with pytest.raises(ValueError, match="sigma must be positive"):
            _metad(sigma=sigma)

    def test_unknown_storage_raises(self) -> None:
        with pytest.raises(ValueError, match="storage must be one of"):
            _metad(storage="lru")

    def test_unknown_history_raises(self) -> None:
        with pytest.raises(ValueError, match="history must be one of"):
            _metad(history="global")

    @pytest.mark.parametrize("storage", ["preallocated", "fifo"])
    def test_bounded_storage_requires_max_hills(self, storage: str) -> None:
        """A ceiling policy without a ceiling is meaningless."""
        with pytest.raises(ValueError, match="needs an explicit max_hills"):
            WellTemperedMetaDynamicsBias(
                cv=_pair_cv(),
                height=0.05,
                sigma=0.2,
                temperature=300.0,
                storage=storage,
            )

    def test_grow_defaults_its_chunk(self) -> None:
        """grow is the one policy that can run without an explicit capacity."""
        bias = WellTemperedMetaDynamicsBias(
            cv=_pair_cv(), height=0.05, sigma=0.2, temperature=300.0, storage="grow"
        )
        assert bias.capacity > 0

    @pytest.mark.parametrize("frequency", [0, -1])
    def test_non_positive_update_frequency_raises(self, frequency: int) -> None:
        with pytest.raises(ValueError, match="update_frequency must be at least 1"):
            _metad(update_frequency=frequency)

    def test_negative_ramp_raises(self) -> None:
        with pytest.raises(ValueError, match="ramp_depositions must be non-negative"):
            _metad(ramp_depositions=-1)

    def test_mixin_order_is_correct(self) -> None:
        """AdaptivePotentialMixin must precede nn.Module in the MRO."""
        mro = WellTemperedMetaDynamicsBias.__mro__
        from nvalchemi.enhanced_sampling import AdaptivePotentialMixin

        assert mro.index(AdaptivePotentialMixin) < mro.index(torch.nn.Module)


# ===========================================================================
# 2. Well-tempered metadynamics: energy and hill scaling
# ===========================================================================


class TestWellTemperedEnergy:
    """Hill accumulation, the well-tempered height, and derived forces."""

    def test_empty_history_is_exactly_zero(self, device: str) -> None:
        """A bias with no hills must contribute nothing, not a small number."""
        bias = _metad(device)
        result = bias.evaluate(_pair_batch([1.0, 2.0], device))
        assert torch.count_nonzero(result.energy) == 0
        assert torch.count_nonzero(result.forces) == 0

    def test_single_hill_matches_closed_form(self, device: str) -> None:
        """V(s) = h * exp(-(s - c)^2 / 2 sigma^2) for one deposited hill."""
        bias = _metad(device, sigma=0.5)
        deposit = _pair_batch([1.0], device)
        bias.update(deposit, bias.evaluate(deposit))

        probe = _pair_batch([1.0, 1.5, 3.0], device)
        got = bias.evaluate(probe).energy.reshape(-1)
        expected = torch.tensor(
            [
                0.05 * math.exp(-((s - 1.0) ** 2) / (2 * 0.5**2))
                for s in (1.0, 1.5, 3.0)
            ],
            device=got.device,
            dtype=got.dtype,
        )
        assert torch.allclose(got, expected, atol=1e-6)

    def test_well_tempered_height_matches_formula(self, device: str) -> None:
        """h = h0 exp(-V / (kB T (gamma - 1))) at the deposition point."""
        gamma, temperature, h0 = 10.0, 300.0, 0.05
        bias = _metad(device, bias_factor=gamma, temperature=temperature, height=h0)
        deposit = _pair_batch([1.0], device)

        bias.update(deposit, bias.evaluate(deposit))
        assert float(bias.hill_heights[0]) == pytest.approx(h0, abs=1e-9)

        # The second hill lands on top of the first, so V = h0 there.
        bias.update(deposit, bias.evaluate(deposit))
        expected = h0 * math.exp(-h0 / (KB_EV * temperature * (gamma - 1.0)))
        assert float(bias.hill_heights[1]) == pytest.approx(expected, rel=1e-6)

    def test_heights_decay_monotonically(self, device: str) -> None:
        """Repeated deposition in one spot must yield shrinking hills."""
        bias = _metad(device, max_hills=8)
        deposit = _pair_batch([1.0], device)
        for _ in range(8):
            bias.update(deposit, bias.evaluate(deposit))
        heights = bias.hill_heights.tolist()
        assert heights == sorted(heights, reverse=True)
        assert heights[-1] < heights[0]

    def test_standard_metadynamics_keeps_constant_height(self, device: str) -> None:
        """bias_factor=None is the gamma -> infinity limit: no damping."""
        bias = _metad(device, bias_factor=None, max_hills=8)
        deposit = _pair_batch([1.0], device)
        for _ in range(4):
            bias.update(deposit, bias.evaluate(deposit))
        assert torch.allclose(
            bias.hill_heights[:4], torch.full_like(bias.hill_heights[:4], 0.05)
        )

    def test_forces_match_numerical_gradient(self, device: str) -> None:
        """Forces are -dE/dx of the same energy the bias reports."""
        bias = _metad(device, sigma=0.4, max_hills=8)
        for d in (1.0, 1.4):
            frame = _pair_batch([d], device)
            bias.update(frame, bias.evaluate(frame))

        batch = _pair_batch([1.2], device)
        result = bias.evaluate(batch)

        eps = 1e-4
        step = _pair_batch([1.2 + eps], device)
        back = _pair_batch([1.2 - eps], device)
        numerical = (
            float(bias.energy(step).sum()) - float(bias.energy(back).sum())
        ) / (2 * eps)
        # Atom 1 carries the whole +x displacement of the pair distance.
        assert float(result.forces[1, 0]) == pytest.approx(-numerical, abs=1e-4)

    def test_bias_is_repulsive(self, device: str) -> None:
        """The force must push the walker away from a deposited hill."""
        bias = _metad(device, sigma=0.3)
        frame = _pair_batch([1.0], device)
        bias.update(frame, bias.evaluate(frame))

        outward = bias.evaluate(_pair_batch([1.1], device))
        # Sitting just beyond the hill, the pair is pushed further apart.
        assert float(outward.forces[1, 0]) > 0.0

    def test_multidimensional_cv(self, device: str) -> None:
        """A 2-component CV uses one sigma per component."""
        idx_a, idx_b = torch.tensor([0, 1]), torch.tensor([0, 2])

        def cv(batch: Batch) -> Tensor:
            return torch.cat(
                [pair_distance(batch, idx_a), pair_distance(batch, idx_b)], dim=-1
            )

        data = AtomicData(
            positions=torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0]]),
            atomic_numbers=torch.ones(3, dtype=torch.long),
        )
        batch = Batch.from_data_list([data]).to(device)
        bias = WellTemperedMetaDynamicsBias(
            cv=cv,
            height=0.05,
            sigma=torch.tensor([0.3, 0.6]),
            temperature=300.0,
            max_hills=4,
        ).to(device)
        bias.update(batch, bias.evaluate(batch))
        assert tuple(bias.hill_centers.shape) == (4, 2)
        assert float(bias.evaluate(batch).energy.sum()) == pytest.approx(0.05, abs=1e-6)


# ===========================================================================
# 3. Well-tempered metadynamics: periodic CVs
# ===========================================================================


class TestWellTemperedPeriodic:
    """A hill near a branch cut must repel from both sides."""

    @staticmethod
    def _phase_cv(batch: Batch) -> Tensor:
        """Treat atom 1's x coordinate as an angle-like periodic CV."""
        return batch.positions[batch.batch_ptr[:-1] + 1, 0:1]

    @staticmethod
    def _phase_batch(values: list[float], device: str) -> Batch:
        items = [
            AtomicData(
                positions=torch.tensor([[0.0, 0.0, 0.0], [v, 0.0, 0.0]]),
                atomic_numbers=torch.ones(2, dtype=torch.long),
            )
            for v in values
        ]
        return Batch.from_data_list(items).to(device)

    def _biases(self, device: str) -> tuple:
        periodic = WellTemperedMetaDynamicsBias(
            cv=self._phase_cv,
            height=0.05,
            sigma=0.2,
            temperature=300.0,
            max_hills=4,
            periods=torch.tensor([2 * math.pi]),
        ).to(device)
        plain = WellTemperedMetaDynamicsBias(
            cv=self._phase_cv,
            height=0.05,
            sigma=0.2,
            temperature=300.0,
            max_hills=4,
        ).to(device)
        return periodic, plain

    def test_hill_wraps_across_the_branch_cut(self, device: str) -> None:
        """A CV at -3.10 is 0.083 from a hill at +3.10, not 6.20 away."""
        periodic, plain = self._biases(device)
        deposit = self._phase_batch([3.10], device)
        for bias in (periodic, plain):
            bias.update(deposit, bias.evaluate(deposit))

        probe = self._phase_batch([-3.10], device)
        assert float(periodic.evaluate(probe).energy.sum()) > 0.04
        assert float(plain.evaluate(probe).energy.sum()) < 1e-12

    def test_wrapped_force_points_the_short_way(self, device: str) -> None:
        """The repulsion must push away from the hill the short way round."""
        periodic, _ = self._biases(device)
        deposit = self._phase_batch([3.10], device)
        periodic.update(deposit, periodic.evaluate(deposit))

        # Wrapped, -3.10 sits 0.083 *above* the hill at 3.10 (it is 3.183
        # once carried across the cut), so repulsion drives the CV further
        # positive rather than back down the long way round.
        forces = periodic.evaluate(self._phase_batch([-3.10], device)).forces
        assert float(forces[1, 0]) > 0.0

        # The unwrapped bias sees the same configuration 6.20 away and does
        # essentially nothing, which is the failure the wrap exists to avoid.
        _, plain = self._biases(device)
        plain.update(deposit, plain.evaluate(deposit))
        plain_forces = plain.evaluate(self._phase_batch([-3.10], device)).forces
        assert float(plain_forces[1, 0]) == pytest.approx(0.0, abs=1e-9)

    def test_non_periodic_component_is_left_unwrapped(self, device: str) -> None:
        """A period of 0 marks a component as non-periodic."""
        bias = WellTemperedMetaDynamicsBias(
            cv=self._phase_cv,
            height=0.05,
            sigma=0.2,
            temperature=300.0,
            max_hills=4,
            periods=torch.tensor([0.0]),
        ).to(device)
        deposit = self._phase_batch([3.10], device)
        bias.update(deposit, bias.evaluate(deposit))
        probe = self._phase_batch([-3.10], device)
        assert float(bias.evaluate(probe).energy.sum()) < 1e-12


# ===========================================================================
# 4. Well-tempered metadynamics: storage policies
# ===========================================================================


class TestStoragePolicies:
    """preallocated raises, grow resizes, fifo evicts."""

    def test_preallocated_overflow_raises(self, device: str) -> None:
        """Silently dropping hills would change a converging run's physics."""
        bias = _metad(device, max_hills=2, storage="preallocated")
        frame = _pair_batch([1.0], device)
        bias.update(frame, bias.evaluate(frame))
        bias.update(frame, bias.evaluate(frame))
        with pytest.raises(RuntimeError, match="storage='preallocated'"):
            bias.update(frame, bias.evaluate(frame))

    def test_preallocated_overflow_leaves_state_untouched(self, device: str) -> None:
        """The failed deposition must not half-apply."""
        bias = _metad(device, max_hills=2, storage="preallocated")
        frame = _pair_batch([1.0], device)
        bias.update(frame, bias.evaluate(frame))
        bias.update(frame, bias.evaluate(frame))
        before = (int(bias.hill_count), int(bias.deposits), bias.state_version)
        with pytest.raises(RuntimeError):
            bias.update(frame, bias.evaluate(frame))
        assert (int(bias.hill_count), int(bias.deposits), bias.state_version) == before

    def test_preallocated_capacity_never_changes(self, device: str) -> None:
        """The compile-stable policy must keep tensor shapes fixed."""
        bias = _metad(device, max_hills=6, storage="preallocated")
        frame = _pair_batch([1.0], device)
        shapes = {tuple(bias.hill_centers.shape)}
        for _ in range(6):
            bias.update(frame, bias.evaluate(frame))
            shapes.add(tuple(bias.hill_centers.shape))
        assert shapes == {(6, 1)}

    def test_grow_extends_capacity(self, device: str) -> None:
        """grow allocates another chunk instead of raising."""
        bias = _metad(device, max_hills=2, storage="grow")
        frame = _pair_batch([1.0], device)
        for _ in range(5):
            bias.update(frame, bias.evaluate(frame))
        assert bias.capacity >= 5
        assert int(bias.hill_count) == 5

    def test_grow_preserves_existing_hills(self, device: str) -> None:
        """Resizing must copy the history, not restart it."""
        bias = _metad(device, max_hills=2, storage="grow", sigma=0.5)
        for d in (1.0, 2.0):
            frame = _pair_batch([d], device)
            bias.update(frame, bias.evaluate(frame))
        before = bias.evaluate(_pair_batch([1.0], device)).energy.clone()

        bias.update(
            _pair_batch([5.0], device), bias.evaluate(_pair_batch([5.0], device))
        )
        after = bias.evaluate(_pair_batch([1.0], device)).energy
        assert bias.capacity > 2
        assert torch.allclose(before, after, atol=1e-6)

    def test_fifo_evicts_the_oldest(self, device: str) -> None:
        """The ring keeps exactly the most recent max_hills deposits."""
        bias = _metad(device, max_hills=3, storage="fifo", sigma=0.2)
        for d in (1.0, 2.0, 3.0, 4.0):
            frame = _pair_batch([d], device)
            bias.update(frame, bias.evaluate(frame))

        assert bias.capacity == 3
        assert int(bias.hill_count) == 3
        centers = sorted(round(float(c), 3) for c in bias.hill_centers.reshape(-1))
        assert centers == [2.0, 3.0, 4.0]
        # The evicted hill leaves no trace at its old location.
        assert float(bias.evaluate(_pair_batch([1.0], device)).energy.sum()) < 1e-6

    def test_fifo_ring_wraps_more_than_once(self, device: str) -> None:
        """Slot assignment must stay correct after several wraps."""
        bias = _metad(device, max_hills=2, storage="fifo", sigma=0.2)
        for d in (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0):
            frame = _pair_batch([d], device)
            bias.update(frame, bias.evaluate(frame))
        centers = sorted(round(float(c), 3) for c in bias.hill_centers.reshape(-1))
        assert centers == [6.0, 7.0]
        assert int(bias.hills_written) == 7

    def test_fifo_refuses_free_energy(self, device: str) -> None:
        """Discarded hills invalidate the well-tempered relation."""
        bias = _metad(device, max_hills=2, storage="fifo")
        with pytest.raises(RuntimeError, match="not valid under storage='fifo'"):
            bias.free_energy(torch.zeros(1, 1, device=device))


# ===========================================================================
# 5. Well-tempered metadynamics: multi-walker history
# ===========================================================================


class TestMultiWalkerHistory:
    """shared, walker-private, and state-owned hill visibility."""

    @staticmethod
    def _two_close_walkers(device: str) -> Batch:
        """Two walkers at nearly the same CV, so cross-hills matter."""
        batch = _pair_batch([1.0, 1.05], device)
        batch.walker_id = torch.tensor([0, 1], device=batch.positions.device)
        batch.thermodynamic_state_id = torch.tensor(
            [0, 1], device=batch.positions.device
        )
        return batch

    def test_shared_history_sees_every_hill(self, device: str) -> None:
        """The multiple-walker scheme: B walkers fill a basin B times faster."""
        batch = self._two_close_walkers(device)
        shared = _metad(device, history="shared", sigma=0.5)
        shared.update(batch, shared.evaluate(batch))
        assert int(shared.hill_count) == 2
        assert bool((shared.hill_owner[:2] == -1).all())

        private = _metad(device, history="walker", sigma=0.5)
        private.update(batch, private.evaluate(batch))

        shared_e = shared.evaluate(batch).energy.reshape(-1)
        private_e = private.evaluate(batch).energy.reshape(-1)
        # Each walker feels its own hill either way; only shared adds the other.
        assert bool((shared_e > private_e + 1e-4).all())

    def test_walker_history_is_private(self, device: str) -> None:
        """Under "walker", a hill is invisible to every other walker."""
        batch = self._two_close_walkers(device)
        bias = _metad(device, history="walker", sigma=0.5)
        bias.update(batch, bias.evaluate(batch))
        assert bias.hill_owner[:2].tolist() == [0, 1]

        # Probe walker 0 alone against the history: it must not feel hill 1.
        probe = _pair_batch([1.05], device)
        probe.walker_id = torch.tensor([0], device=probe.positions.device)
        alone = float(bias.evaluate(probe).energy.sum())

        probe.walker_id = torch.tensor([1], device=probe.positions.device)
        owner = float(bias.evaluate(probe).energy.sum())
        assert owner > alone

    def test_state_history_tags_by_thermodynamic_state(self, device: str) -> None:
        batch = self._two_close_walkers(device)
        bias = _metad(device, history="state", sigma=0.5)
        bias.update(batch, bias.evaluate(batch))
        assert bias.hill_owner[:2].tolist() == [0, 1]

    def test_state_history_declares_exchange_dependence(self) -> None:
        """A per-state history changes which hills a swap exposes a walker to."""
        assert _metad(history="state").state_dependent_for_exchange is True
        assert _metad(history="shared").state_dependent_for_exchange is False
        assert _metad(history="walker").state_dependent_for_exchange is False


# ===========================================================================
# 6. Well-tempered metadynamics: ramping and free energy
# ===========================================================================


class TestRampAndFreeEnergy:
    """Smooth hill activation and the free-energy estimator."""

    def test_no_ramp_activates_immediately(self, device: str) -> None:
        bias = _metad(device, ramp_depositions=0)
        frame = _pair_batch([1.0], device)
        bias.update(frame, bias.evaluate(frame))
        assert float(bias.evaluate(frame).energy.sum()) == pytest.approx(0.05, abs=1e-6)

    def test_ramp_grows_the_contribution(self, device: str) -> None:
        """A ramped hill must not switch on at full height where it landed."""
        bias = _metad(device, ramp_depositions=4)
        frame = _pair_batch([1.0], device)
        bias.update(frame, bias.evaluate(frame))

        series = []
        for _ in range(4):
            series.append(float(bias.evaluate(frame).energy.sum()))
            bias.deposits += 1
        assert series == sorted(series)
        assert series[0] < 0.05 * 0.5
        assert series[-1] == pytest.approx(0.05, abs=1e-6)

    def test_free_energy_uses_the_well_tempered_factor(self, device: str) -> None:
        """F = -(gamma / (gamma - 1)) V."""
        gamma = 10.0
        bias = _metad(device, bias_factor=gamma)
        frame = _pair_batch([1.0], device)
        bias.update(frame, bias.evaluate(frame))

        values = torch.tensor([[1.0]], device=device)
        key = torch.full((1,), -1, dtype=torch.long, device=device)
        bias_value = bias.gaussian_sum(values, key)
        assert torch.allclose(
            bias.free_energy(values), -bias_value * (gamma / (gamma - 1.0))
        )

    def test_free_energy_of_standard_metadynamics(self, device: str) -> None:
        """F = -V when there is no well-tempered damping."""
        bias = _metad(device, bias_factor=None)
        frame = _pair_batch([1.0], device)
        bias.update(frame, bias.evaluate(frame))
        values = torch.tensor([[1.0]], device=device)
        key = torch.full((1,), -1, dtype=torch.long, device=device)
        assert torch.allclose(bias.free_energy(values), -bias.gaussian_sum(values, key))

    def test_free_energy_is_lower_where_hills_accumulated(self, device: str) -> None:
        """The estimator must report a basin where the bias filled one in."""
        bias = _metad(device, max_hills=16, sigma=0.3)
        frame = _pair_batch([1.0], device)
        for _ in range(8):
            bias.update(frame, bias.evaluate(frame))
        probed = bias.free_energy(torch.tensor([[1.0], [4.0]], device=device))
        assert float(probed[0]) < float(probed[1])


# ===========================================================================
# 7. Well-tempered metadynamics: compile and restart
# ===========================================================================


class TestWellTemperedCompile:
    """energy() is the compile boundary and must hold fullgraph=True."""

    def test_energy_compiles_fullgraph(self, device: str) -> None:
        bias = _metad(device, sigma=0.4)
        frame = _pair_batch([1.0], device)
        bias.update(frame, bias.evaluate(frame))

        batch = _pair_batch([1.2, 2.0], device)
        eager = bias.energy(batch)
        compiled = torch.compile(bias.energy, fullgraph=True)(batch)
        assert torch.allclose(eager, compiled, atol=1e-6)

    def test_compiled_energy_tracks_new_hills(self, device: str) -> None:
        """Depositing must change the compiled result, not hit a stale trace."""
        bias = _metad(device, sigma=0.4, max_hills=8)
        compiled = torch.compile(bias.energy, fullgraph=True)
        batch = _pair_batch([1.0], device)
        assert float(compiled(batch).sum()) == pytest.approx(0.0, abs=1e-9)

        bias.update(batch, bias.evaluate(batch))
        assert float(compiled(batch).sum()) == pytest.approx(0.05, abs=1e-6)

    def test_per_state_history_compiles(self, device: str) -> None:
        """The owner mask must not introduce a data-dependent branch."""
        bias = _metad(device, history="state", sigma=0.4)
        batch = _pair_batch([1.0, 1.2], device)
        batch.thermodynamic_state_id = torch.tensor(
            [0, 1], device=batch.positions.device
        )
        bias.update(batch, bias.evaluate(batch))
        compiled = torch.compile(bias.energy, fullgraph=True)
        assert torch.allclose(compiled(batch), bias.energy(batch), atol=1e-6)


class TestWellTemperedRestart:
    """state_dict / load_state_dict round trips, including resized buffers."""

    def test_round_trip_reproduces_the_energy(self, device: str) -> None:
        bias = _metad(device, max_hills=8, sigma=0.4)
        for d in (1.0, 2.0, 3.0):
            frame = _pair_batch([d], device)
            bias.update(frame, bias.evaluate(frame))

        restored = _metad(device, max_hills=8, sigma=0.4)
        restored.load_state_dict(bias.state_dict())

        probe = _pair_batch([1.0, 2.5], device)
        assert torch.allclose(
            restored.evaluate(probe).energy, bias.evaluate(probe).energy
        )
        assert int(restored.hill_count) == int(bias.hill_count)
        assert int(restored.deposits) == int(bias.deposits)
        assert restored.state_version == bias.state_version

    def test_round_trip_after_growth_resizes_buffers(self, device: str) -> None:
        """A grown checkpoint has a capacity the constructor never produces."""
        bias = _metad(device, max_hills=2, storage="grow", sigma=0.4)
        for d in (1.0, 2.0, 3.0, 4.0, 5.0):
            frame = _pair_batch([d], device)
            bias.update(frame, bias.evaluate(frame))
        assert bias.capacity > 2

        restored = _metad(device, max_hills=2, storage="grow", sigma=0.4)
        restored.load_state_dict(bias.state_dict())
        assert restored.capacity == bias.capacity

        probe = _pair_batch([1.0, 3.0], device)
        assert torch.allclose(
            restored.evaluate(probe).energy, bias.evaluate(probe).energy
        )

    def test_restart_continues_the_well_tempered_sequence(self, device: str) -> None:
        """Heights after a restart must follow on, not reset to h0."""
        bias = _metad(device, max_hills=8)
        frame = _pair_batch([1.0], device)
        for _ in range(3):
            bias.update(frame, bias.evaluate(frame))

        restored = _metad(device, max_hills=8)
        restored.load_state_dict(bias.state_dict())
        bias.update(frame, bias.evaluate(frame))
        restored.update(frame, restored.evaluate(frame))
        assert float(restored.hill_heights[3]) == pytest.approx(
            float(bias.hill_heights[3]), rel=1e-9
        )


# ===========================================================================
# 8. RMSD metadynamics: alignment
# ===========================================================================


class TestSquaredRMSD:
    """The QCP kernel underneath the RMSD bias."""

    def test_matches_svd_kabsch(self) -> None:
        """QCP and an explicit det-corrected SVD Kabsch must agree."""
        torch.manual_seed(3)
        coords = torch.randn(3, 6, 3, dtype=torch.float64)
        refs = torch.randn(4, 6, 3, dtype=torch.float64)
        refs = refs - refs.mean(dim=1, keepdim=True)

        got = _squared_rmsd(coords, refs)

        centered = coords - coords.mean(dim=1, keepdim=True)
        expected = torch.zeros(3, 4, dtype=torch.float64)
        for b in range(3):
            for r in range(4):
                cov = centered[b].T @ refs[r]
                u, s, vt = torch.linalg.svd(cov)
                sign = torch.sign(torch.det(u @ vt))
                trace = s[0] + s[1] + sign * s[2]
                expected[b, r] = (
                    (centered[b] ** 2).sum() + (refs[r] ** 2).sum() - 2 * trace
                ) / 6
        assert torch.allclose(got, expected, atol=1e-10)

    def test_identical_structures_give_zero(self) -> None:
        """Never negative: the clamp must absorb the rounding step."""
        torch.manual_seed(4)
        refs = torch.randn(3, 5, 3, dtype=torch.float64)
        refs = refs - refs.mean(dim=1, keepdim=True)
        diagonal = _squared_rmsd(refs, refs).diagonal()
        assert bool((diagonal >= 0).all())
        assert float(diagonal.abs().max()) < 1e-18

    def test_reflection_is_not_treated_as_identical(self) -> None:
        """A mirror image is a different structure; only proper rotations align."""
        torch.manual_seed(5)
        ref = torch.randn(1, 5, 3, dtype=torch.float64)
        ref = ref - ref.mean(dim=1, keepdim=True)
        mirrored = ref.clone()
        mirrored[..., 0] *= -1
        assert float(_squared_rmsd(mirrored, ref)) > 1e-3


class TestRMSDInvariance:
    """Translation and rotation must not change the bias."""

    def _molecule(self, device: str, seed: int = 0) -> Batch:
        torch.manual_seed(seed)
        data = AtomicData(
            positions=torch.randn(5, 3),
            atomic_numbers=torch.ones(5, dtype=torch.long),
        )
        return Batch.from_data_list([data]).to(device)

    def test_energy_is_invariant_to_rigid_motion(self, device: str) -> None:
        bias = RMSDMetaDynamicsBias(
            k_push=0.02, alpha=0.5, max_references=4, ramp_depositions=0
        ).to(device)
        frame = self._molecule(device)
        bias.update(frame, bias.evaluate(frame))
        original = bias.evaluate(frame).energy.clone()

        torch.manual_seed(11)
        rotation = _rot().to(device=frame.positions.device, dtype=frame.positions.dtype)
        shift = torch.tensor([3.0, -2.0, 7.0], device=frame.positions.device)
        moved = self._molecule(device)
        moved.positions = frame.positions @ rotation.T + shift

        assert torch.allclose(bias.evaluate(moved).energy, original, atol=1e-5)

    def test_bias_exerts_no_net_force(self, device: str) -> None:
        """A translation-invariant energy cannot push the molecule bodily."""
        bias = RMSDMetaDynamicsBias(
            k_push=0.05, alpha=0.5, max_references=4, ramp_depositions=0
        ).to(device)
        frame = self._molecule(device)
        bias.update(frame, bias.evaluate(frame))

        probe = self._molecule(device, seed=2)
        forces = bias.evaluate(probe).forces
        assert float(forces.sum(dim=0).abs().max()) < 1e-5

    def test_forces_match_numerical_gradient(self, device: str) -> None:
        bias = RMSDMetaDynamicsBias(
            k_push=0.05, alpha=0.3, max_references=4, ramp_depositions=0
        ).to(device)
        frame = self._molecule(device)
        bias.update(frame, bias.evaluate(frame))

        probe = self._molecule(device, seed=2)
        analytic = bias.evaluate(probe).forces

        eps = 1e-4
        base = probe.positions.clone()
        for atom, axis in ((0, 0), (3, 2)):
            probe.positions = base.clone()
            probe.positions[atom, axis] += eps
            plus = float(bias.energy(probe).sum())
            probe.positions = base.clone()
            probe.positions[atom, axis] -= eps
            minus = float(bias.energy(probe).sum())
            assert float(analytic[atom, axis]) == pytest.approx(
                -(plus - minus) / (2 * eps), abs=1e-4
            )


# ===========================================================================
# 9. RMSD metadynamics: construction, selection, and periodicity
# ===========================================================================


class TestRMSDConstruction:
    """Constructor validation for the RMSD bias."""

    @pytest.mark.parametrize("k_push", [0.0, -0.01])
    def test_non_positive_k_push_raises(self, k_push: float) -> None:
        with pytest.raises(ValueError, match="k_push must be positive"):
            RMSDMetaDynamicsBias(k_push=k_push, alpha=0.5, max_references=4)

    @pytest.mark.parametrize("alpha", [0.0, -1.0])
    def test_non_positive_alpha_raises(self, alpha: float) -> None:
        with pytest.raises(ValueError, match="alpha must be positive"):
            RMSDMetaDynamicsBias(k_push=0.02, alpha=alpha, max_references=4)

    @pytest.mark.parametrize("storage", ["preallocated", "fifo"])
    def test_bounded_storage_requires_capacity(self, storage: str) -> None:
        with pytest.raises(ValueError, match="needs an explicit max_references"):
            RMSDMetaDynamicsBias(k_push=0.02, alpha=0.5, storage=storage)

    def test_empty_atom_indices_raises(self) -> None:
        with pytest.raises(ValueError, match="atom_indices is empty"):
            RMSDMetaDynamicsBias(
                k_push=0.02,
                alpha=0.5,
                max_references=4,
                atom_indices=torch.tensor([], dtype=torch.long),
            )

    def test_duplicate_atom_indices_raise(self) -> None:
        """A repeated atom is silently double-weighted in the RMSD."""
        with pytest.raises(ValueError, match="contains duplicates"):
            RMSDMetaDynamicsBias(
                k_push=0.02,
                alpha=0.5,
                max_references=4,
                atom_indices=torch.tensor([0, 1, 1]),
            )

    def test_negative_atom_indices_raise(self) -> None:
        with pytest.raises(ValueError, match="must be non-negative"):
            RMSDMetaDynamicsBias(
                k_push=0.02,
                alpha=0.5,
                max_references=4,
                atom_indices=torch.tensor([0, -1]),
            )

    def test_default_storage_is_fifo(self) -> None:
        """The xTB-compatible policy is the default here."""
        bias = RMSDMetaDynamicsBias(k_push=0.02, alpha=0.5, max_references=4)
        assert bias.storage == "fifo"


class TestRMSDSelectionAndPeriodicity:
    """Atom selection is per-graph, and periodic batches are rejected."""

    @staticmethod
    def _batch(device: str, sizes: list[int], seed: int = 0) -> Batch:
        torch.manual_seed(seed)
        items = [
            AtomicData(
                positions=torch.randn(n, 3),
                atomic_numbers=torch.ones(n, dtype=torch.long),
            )
            for n in sizes
        ]
        return Batch.from_data_list(items).to(device)

    def test_selection_is_per_graph_local(self, device: str) -> None:
        """Index 1 means atom 1 of each graph, not global atom 1."""
        bias = RMSDMetaDynamicsBias(
            k_push=0.02,
            alpha=0.5,
            max_references=4,
            atom_indices=torch.tensor([0, 1, 2]),
            ramp_depositions=0,
        ).to(device)
        batch = self._batch(device, [4, 4])
        bias.update(batch, bias.evaluate(batch))
        assert tuple(bias.reference_coords.shape) == (4, 3, 3)

    def test_out_of_range_selection_raises(self, device: str) -> None:
        bias = RMSDMetaDynamicsBias(
            k_push=0.02, alpha=0.5, max_references=4, atom_indices=torch.tensor([0, 7])
        ).to(device)
        with pytest.raises(ValueError, match="local index 7"):
            bias.evaluate(self._batch(device, [4, 4]))

    def test_ragged_batch_without_selection_raises(self, device: str) -> None:
        """No fixed correspondence exists across differently sized graphs."""
        bias = RMSDMetaDynamicsBias(k_push=0.02, alpha=0.5, max_references=4).to(device)
        with pytest.raises(ValueError, match="differing atom counts"):
            bias.evaluate(self._batch(device, [4, 5]))

    def test_ragged_batch_with_selection_is_fine(self, device: str) -> None:
        """A common selection restores the correspondence."""
        bias = RMSDMetaDynamicsBias(
            k_push=0.02,
            alpha=0.5,
            max_references=4,
            atom_indices=torch.tensor([0, 1, 2]),
        ).to(device)
        result = bias.evaluate(self._batch(device, [4, 5]))
        assert result.energy.shape[0] == 2

    def test_periodic_batch_is_rejected(self, device: str) -> None:
        """Cartesian RMSD is undefined once atoms can cross a cell face."""
        bias = RMSDMetaDynamicsBias(k_push=0.02, alpha=0.5, max_references=4).to(device)
        batch = self._batch(device, [4, 4])
        batch.cell = torch.eye(3, device=batch.positions.device).repeat(2, 1, 1) * 10.0
        with pytest.raises(ValueError, match="not defined under periodic"):
            bias.evaluate(batch)

    def test_zero_cell_is_not_periodic(self, device: str) -> None:
        """A zero cell is how a molecular batch spells "no cell"."""
        bias = RMSDMetaDynamicsBias(k_push=0.02, alpha=0.5, max_references=4).to(device)
        batch = self._batch(device, [4, 4])
        batch.cell = torch.zeros(2, 3, 3, device=batch.positions.device)
        bias.evaluate(batch)


# ===========================================================================
# 10. RMSD metadynamics: deposition, warm start, retention, restart
# ===========================================================================


class TestRMSDDeposition:
    """Reference accumulation, ramping, FIFO retention, and warm starts."""

    @staticmethod
    def _molecule(device: str, seed: int = 0, n_graphs: int = 1) -> Batch:
        torch.manual_seed(seed)
        items = [
            AtomicData(
                positions=torch.randn(5, 3),
                atomic_numbers=torch.ones(5, dtype=torch.long),
            )
            for _ in range(n_graphs)
        ]
        return Batch.from_data_list(items).to(device)

    def test_empty_history_is_zero(self, device: str) -> None:
        bias = RMSDMetaDynamicsBias(k_push=0.02, alpha=0.5, max_references=4).to(device)
        result = bias.evaluate(self._molecule(device))
        assert torch.count_nonzero(result.energy) == 0

    def test_deposited_structure_feels_full_amplitude(self, device: str) -> None:
        """At RMSD zero the kernel is exp(0) = 1, so V = k_push."""
        bias = RMSDMetaDynamicsBias(
            k_push=0.02, alpha=0.5, max_references=4, ramp_depositions=0
        ).to(device)
        frame = self._molecule(device)
        bias.update(frame, bias.evaluate(frame))
        assert float(bias.evaluate(frame).energy.sum()) == pytest.approx(0.02, abs=1e-7)

    def test_ramp_is_on_by_default(self, device: str) -> None:
        """A new reference lands exactly where the system is standing."""
        bias = RMSDMetaDynamicsBias(k_push=0.02, alpha=0.5, max_references=4)
        assert bias.ramp_depositions == 1

    def test_ramp_delays_full_amplitude(self, device: str) -> None:
        bias = RMSDMetaDynamicsBias(
            k_push=0.02, alpha=0.5, max_references=8, ramp_depositions=4
        ).to(device)
        frame = self._molecule(device)
        bias.update(frame, bias.evaluate(frame))

        series = []
        for _ in range(4):
            series.append(float(bias.evaluate(frame).energy.sum()))
            bias.deposits += 1
        assert series == sorted(series)
        assert series[0] < 0.02
        assert series[-1] == pytest.approx(0.02, abs=1e-7)

    def test_fifo_retains_the_most_recent(self, device: str) -> None:
        bias = RMSDMetaDynamicsBias(
            k_push=0.02, alpha=2.0, max_references=2, ramp_depositions=0
        ).to(device)
        frames = [self._molecule(device, seed=s) for s in range(4)]
        for frame in frames:
            bias.update(frame, bias.evaluate(frame))

        assert bias.capacity == 2
        assert int(bias.reference_count) == 2
        assert int(bias.references_written) == 4
        # The oldest reference has been overwritten, so its site is free again.
        assert float(bias.evaluate(frames[0]).energy.sum()) < float(
            bias.evaluate(frames[3]).energy.sum()
        )

    def test_preallocated_overflow_raises(self, device: str) -> None:
        bias = RMSDMetaDynamicsBias(
            k_push=0.02, alpha=0.5, max_references=2, storage="preallocated"
        ).to(device)
        frame = self._molecule(device)
        bias.update(frame, bias.evaluate(frame))
        bias.update(frame, bias.evaluate(frame))
        with pytest.raises(RuntimeError, match="storage='preallocated'"):
            bias.update(frame, bias.evaluate(frame))

    def test_warm_start_references_are_active_immediately(self, device: str) -> None:
        """A seeded reference is history, not a fresh deposit needing a ramp."""
        torch.manual_seed(0)
        reference = torch.randn(1, 5, 3)
        bias = RMSDMetaDynamicsBias(
            k_push=0.02,
            alpha=0.5,
            max_references=4,
            references=reference,
            ramp_depositions=4,
        ).to(device)
        assert int(bias.reference_count) == 1

        frame = self._molecule(device)
        frame.positions = reference[0].to(frame.positions.device)
        assert float(bias.evaluate(frame).energy.sum()) == pytest.approx(0.02, abs=1e-7)

    def test_warm_start_is_translation_normalised(self, device: str) -> None:
        """Seeded references are centered on storage like deposited ones."""
        torch.manual_seed(0)
        reference = torch.randn(1, 5, 3) + 100.0
        bias = RMSDMetaDynamicsBias(
            k_push=0.02, alpha=0.5, max_references=4, references=reference
        ).to(device)
        assert float(bias.reference_coords[0].mean(dim=0).abs().max()) < 1e-5

    def test_warm_start_shape_is_validated(self) -> None:
        with pytest.raises(ValueError, match=r"shape \[R, M, 3\]"):
            RMSDMetaDynamicsBias(
                k_push=0.02, alpha=0.5, max_references=4, references=torch.randn(5, 3)
            )

    def test_warm_start_beyond_capacity_raises(self) -> None:
        with pytest.raises(ValueError, match="exceed max_references"):
            RMSDMetaDynamicsBias(
                k_push=0.02,
                alpha=0.5,
                max_references=2,
                references=torch.randn(3, 5, 3),
            )

    def test_warm_start_conflicting_with_selection_raises(self) -> None:
        with pytest.raises(ValueError, match="but atom_indices selects"):
            RMSDMetaDynamicsBias(
                k_push=0.02,
                alpha=0.5,
                max_references=4,
                atom_indices=torch.tensor([0, 1]),
                references=torch.randn(1, 5, 3),
            )

    def test_walker_history_is_private(self, device: str) -> None:
        batch = self._molecule(device, n_graphs=2)
        bias = RMSDMetaDynamicsBias(
            k_push=0.02,
            alpha=0.5,
            max_references=8,
            history="walker",
            ramp_depositions=0,
        ).to(device)
        batch.walker_id = torch.tensor([0, 1], device=batch.positions.device)
        bias.update(batch, bias.evaluate(batch))
        assert bias.reference_owner[:2].tolist() == [0, 1]

        shared = RMSDMetaDynamicsBias(
            k_push=0.02, alpha=0.5, max_references=8, ramp_depositions=0
        ).to(device)
        shared.update(batch, shared.evaluate(batch))
        assert bool(
            (shared.evaluate(batch).energy >= bias.evaluate(batch).energy - 1e-9).all()
        )

    def test_state_history_declares_exchange_dependence(self) -> None:
        assert (
            RMSDMetaDynamicsBias(
                k_push=0.02, alpha=0.5, max_references=4, history="state"
            ).state_dependent_for_exchange
            is True
        )
        assert (
            RMSDMetaDynamicsBias(
                k_push=0.02, alpha=0.5, max_references=4
            ).state_dependent_for_exchange
            is False
        )


class TestRMSDRestart:
    """state_dict / load_state_dict round trips for the reference set."""

    @staticmethod
    def _molecule(device: str, seed: int = 0) -> Batch:
        torch.manual_seed(seed)
        data = AtomicData(
            positions=torch.randn(5, 3), atomic_numbers=torch.ones(5, dtype=torch.long)
        )
        return Batch.from_data_list([data]).to(device)

    def test_round_trip_reproduces_the_energy(self, device: str) -> None:
        bias = RMSDMetaDynamicsBias(
            k_push=0.02, alpha=0.5, max_references=8, ramp_depositions=0
        ).to(device)
        for seed in range(3):
            frame = self._molecule(device, seed=seed)
            bias.update(frame, bias.evaluate(frame))

        restored = RMSDMetaDynamicsBias(
            k_push=0.02, alpha=0.5, max_references=8, ramp_depositions=0
        ).to(device)
        restored.load_state_dict(bias.state_dict())

        probe = self._molecule(device, seed=1)
        assert torch.allclose(
            restored.evaluate(probe).energy, bias.evaluate(probe).energy, atol=1e-7
        )
        assert int(restored.reference_count) == int(bias.reference_count)
        assert restored.state_version == bias.state_version

    def test_round_trip_after_growth_resizes_buffers(self, device: str) -> None:
        bias = RMSDMetaDynamicsBias(
            k_push=0.02,
            alpha=0.5,
            max_references=2,
            storage="grow",
            ramp_depositions=0,
        ).to(device)
        for seed in range(5):
            frame = self._molecule(device, seed=seed)
            bias.update(frame, bias.evaluate(frame))
        assert bias.capacity > 2

        restored = RMSDMetaDynamicsBias(
            k_push=0.02,
            alpha=0.5,
            max_references=2,
            storage="grow",
            ramp_depositions=0,
        ).to(device)
        restored.load_state_dict(bias.state_dict())
        assert restored.capacity == bias.capacity

        probe = self._molecule(device, seed=2)
        assert torch.allclose(
            restored.evaluate(probe).energy, bias.evaluate(probe).energy, atol=1e-7
        )


# ===========================================================================
# 11. Runner integration: deposition schedule and force priming
# ===========================================================================


class TestRunnerIntegration:
    """The runner drives deposition exactly once per due step."""

    def test_deposition_follows_update_frequency(self, device: str) -> None:
        bias = _metad(device, name="meta", update_frequency=3, max_hills=64)
        runner = EnhancedSampling(_make_dynamics(device), {"meta": bias})
        runner.run(_random_batch(device=device), n_steps=9)

        # Two walkers deposit one hill each per due step.
        assert int(bias.deposits) == 3
        assert int(bias.hill_count) == 6

    def test_no_deposition_during_priming(self, device: str) -> None:
        """Priming evaluates forces; it must not advance the history."""
        bias = _metad(device, name="meta", update_frequency=1)
        runner = EnhancedSampling(_make_dynamics(device), {"meta": bias})
        runner.prime_forces(_random_batch(device=device))
        assert int(bias.deposits) == 0

    def test_hills_are_deposited_at_post_step_coordinates(self, device: str) -> None:
        """observation_stage is AFTER_STEP: a hill marks where it arrived."""
        assert _metad(device).observation_stage.name == "AFTER_STEP"

    def test_new_hill_is_felt_on_the_next_step(self, device: str) -> None:
        """Depositing bumps the state version, so the runner re-primes forces."""
        bias = _metad(device, name="meta", update_frequency=1, sigma=0.6, height=0.5)
        runner = EnhancedSampling(
            _make_dynamics(device), {"meta": bias}, prime_after_update=True
        )
        batch = _random_batch(device=device)
        runner.prime_forces(batch)
        before = int(bias.state_version)

        runner.run(batch, n_steps=1, prime=False)
        assert int(bias.state_version) > before
        assert float(runner.last_outputs["bias/meta/energy"].abs().sum()) > 0.0

    def test_total_is_physical_plus_bias(self, device: str) -> None:
        bias = _metad(device, name="meta", update_frequency=1, sigma=0.6)
        runner = EnhancedSampling(_make_dynamics(device), {"meta": bias})
        runner.run(_random_batch(device=device), n_steps=4)

        outputs = runner.last_outputs
        assert float(outputs["total/energy"].sum()) == pytest.approx(
            float(outputs["physical/energy"].sum())
            + float(outputs["bias_total/energy"].sum()),
            abs=1e-5,
        )

    def test_rmsd_bias_runs_through_the_runner(self, device: str) -> None:
        bias = RMSDMetaDynamicsBias(
            k_push=0.03,
            alpha=0.4,
            max_references=16,
            update_frequency=2,
            name="rmsd",
        ).to(device)
        runner = EnhancedSampling(_make_dynamics(device), {"rmsd": bias})
        runner.run(_random_batch(device=device), n_steps=6)

        assert int(bias.deposits) == 3
        assert int(bias.reference_count) == 6
        assert float(runner.last_outputs["bias/rmsd/energy"].abs().sum()) > 0.0

    def test_two_metadynamics_biases_compose(self, device: str) -> None:
        """Independent schedules, summed against unmodified physical output."""
        meta = _metad(device, name="meta", update_frequency=2, max_hills=64)
        rmsd = RMSDMetaDynamicsBias(
            k_push=0.03,
            alpha=0.4,
            max_references=16,
            update_frequency=3,
            name="rmsd",
        ).to(device)
        runner = EnhancedSampling(_make_dynamics(device), {"meta": meta, "rmsd": rmsd})
        runner.run(_random_batch(device=device), n_steps=6)

        assert int(meta.deposits) == 3
        assert int(rmsd.deposits) == 2

    def test_checkpoint_round_trip_preserves_history(
        self, tmp_path, device: str
    ) -> None:
        """The runner's Zarr checkpoint must carry the hill table."""
        bias = _metad(device, name="meta", update_frequency=1, max_hills=32)
        runner = EnhancedSampling(
            _make_dynamics(device), {"meta": bias}, steps_per_epoch=4
        )
        batch = _random_batch(device=device)
        batch = runner.run(batch, n_steps=4)

        path = tmp_path / "metad.zarr"
        runner.checkpoint(path, batch)

        fresh_bias = _metad(device, name="meta", update_frequency=1, max_hills=32)
        fresh = EnhancedSampling(
            _make_dynamics(device), {"meta": fresh_bias}, steps_per_epoch=4
        )
        restored = fresh.restore(path, device=device)

        assert int(fresh_bias.hill_count) == int(bias.hill_count)
        assert int(fresh_bias.deposits) == int(bias.deposits)
        assert fresh_bias.state_version == bias.state_version
        assert torch.allclose(
            fresh_bias.evaluate(restored).energy,
            bias.evaluate(restored).energy,
            atol=1e-6,
        )
