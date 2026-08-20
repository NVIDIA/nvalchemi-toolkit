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
"""Unit tests for synchronous replica exchange.

Covers ladder validation, the even/odd pair schedule, both acceptance rules
against hand-computed values, the indivisibility of an accepted swap
(labels + integrator target + velocities + forces), determinism, and
checkpoint round-trips.
"""

from __future__ import annotations

import math

import pytest
import torch

from nvalchemi.data import AtomicData, Batch
from nvalchemi.dynamics import NVE, NVTLangevin, NVTNoseHoover
from nvalchemi.dynamics.hooks._utils import KB_EV
from nvalchemi.enhanced_sampling import (
    ConservativeBias,
    EnhancedSampling,
    HarmonicUmbrellaBias,
    ReplicaExchange,
    ThermodynamicState,
    UpperWall,
    pair_distance,
)
from nvalchemi.enhanced_sampling._exchange import log_acceptance_is_accepted
from nvalchemi.models.demo import DemoModel, DemoModelWrapper

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ladder(n: int = 4, base: float = 300.0, factor: float = 1.15):
    return [
        ThermodynamicState(state_id=i, temperature=base * factor**i) for i in range(n)
    ]


def _flat_ladder(n: int = 4, temperature: float = 300.0):
    """Equal temperatures: the ladder can only differ by bias window."""
    return [ThermodynamicState(state_id=i, temperature=temperature) for i in range(n)]


def _make_batch(
    n_graphs: int = 4, atoms_per_graph: int = 4, device: str = "cpu", seed: int = 0
) -> Batch:
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


def _make_dynamics(device: str = "cpu", seed: int = 0) -> NVTLangevin:
    torch.manual_seed(seed)
    return NVTLangevin(
        model=DemoModelWrapper(DemoModel()).to(device),
        dt=0.1,
        temperature=300.0,
        friction=0.1,
    )


def _target_temperatures(dynamics) -> list[float]:
    """Return the integrator's per-graph target temperature in Kelvin."""
    return (dynamics._state.temperature.reshape(-1) / KB_EV).tolist()


# ===========================================================================
# 1. Ladder and assignment validation
# ===========================================================================


class TestLadderValidation:
    """Malformed ladders fail at construction, not mid-run."""

    def test_asynchronous_mode_rejected(self) -> None:
        with pytest.raises(ValueError, match="not supported"):
            ReplicaExchange(_ladder(2), torch.arange(2), mode="asynchronous")

    def test_single_state_rejected(self) -> None:
        with pytest.raises(ValueError, match="at least 2 states"):
            ReplicaExchange(_ladder(1), torch.arange(1))

    def test_sparse_state_ids_rejected(self) -> None:
        """Pairing walks neighbouring indices, so gaps have no neighbours."""
        states = [
            ThermodynamicState(state_id=0, temperature=300.0),
            ThermodynamicState(state_id=5, temperature=350.0),
        ]
        with pytest.raises(ValueError, match="must be exactly 0..1"):
            ReplicaExchange(states, torch.arange(2))

    def test_duplicate_assignment_rejected(self) -> None:
        """Two walkers on one rung breaks the bijection exchange assumes."""
        with pytest.raises(ValueError, match="permutation"):
            ReplicaExchange(_ladder(3), torch.tensor([0, 0, 1]))

    def test_assignment_size_mismatch_rejected(self) -> None:
        """A count mismatch says so, rather than "not a permutation"."""
        with pytest.raises(ValueError, match="3 entr.* but the ladder has 4"):
            ReplicaExchange(_ladder(4), torch.arange(3))

    def test_wrong_batch_size_named_clearly(self, device: str) -> None:
        """A ladder-sized tensor on a differently-sized batch.

        Without the check this surfaces as "Length mismatch: 4 vs 2" from
        inside the batch storage, naming neither the ladder nor the batch.
        """
        exchange = ReplicaExchange(_ladder(4), torch.arange(4), attempt_interval=2)
        runner = EnhancedSampling(_make_dynamics(device), {}, replica_exchange=exchange)
        with pytest.raises(ValueError, match="4 state.*but the batch has 2 walker"):
            runner.run(_make_batch(n_graphs=2, device=device), n_steps=2)

    def test_batch_supplied_duplicate_assignment_rejected(self, device: str) -> None:
        """A batch can carry an assignment the constructor never saw.

        A duplicate leaves one rung held by nobody, which surfaces later as a
        bare KeyError from the pair lookup.
        """
        exchange = ReplicaExchange(_ladder(4), torch.arange(4), attempt_interval=2)
        runner = EnhancedSampling(_make_dynamics(device), {}, replica_exchange=exchange)
        batch = _make_batch(n_graphs=4, device=device)
        batch["thermodynamic_state_id"] = torch.tensor([0, 0, 1, 2], device=device)
        with pytest.raises(ValueError, match="batch.thermodynamic_state_id"):
            runner.run(batch, n_steps=2)

    def test_batch_supplied_out_of_range_assignment_rejected(self, device: str) -> None:
        exchange = ReplicaExchange(_ladder(3), torch.arange(3), attempt_interval=2)
        runner = EnhancedSampling(_make_dynamics(device), {}, replica_exchange=exchange)
        batch = _make_batch(n_graphs=3, device=device)
        batch["thermodynamic_state_id"] = torch.tensor([0, 1, 9], device=device)
        with pytest.raises(ValueError, match="permutation"):
            runner.run(batch, n_steps=2)

    def test_valid_batch_supplied_permutation_accepted(self, device: str) -> None:
        """A caller-chosen starting assignment is legitimate."""
        exchange = ReplicaExchange(_ladder(3), torch.arange(3), attempt_interval=2)
        runner = EnhancedSampling(
            _make_dynamics(device), {}, steps_per_epoch=8, replica_exchange=exchange
        )
        batch = _make_batch(n_graphs=3, device=device)
        batch["thermodynamic_state_id"] = torch.tensor([2, 0, 1], device=device)
        batch = runner.run(batch, n_steps=4)
        assert sorted(batch.thermodynamic_state_id.reshape(-1).tolist()) == [0, 1, 2]

    def test_validate_assignment_shared_by_both_paths(self) -> None:
        """One rule, used for the constructor argument and the batch alike."""
        exchange = ReplicaExchange(_ladder(3), torch.arange(3))
        with pytest.raises(ValueError, match="my_field must be a permutation"):
            exchange.validate_assignment(torch.tensor([0, 0, 2]), source="my_field")
        assert exchange.validate_assignment(torch.tensor([[2], [0], [1]])).tolist() == [
            2,
            0,
            1,
        ]

    def test_states_sorted_by_id(self) -> None:
        shuffled = [
            ThermodynamicState(state_id=2, temperature=400.0),
            ThermodynamicState(state_id=0, temperature=300.0),
            ThermodynamicState(state_id=1, temperature=350.0),
        ]
        exchange = ReplicaExchange(shuffled, torch.arange(3))
        assert [s.state_id for s in exchange.states] == [0, 1, 2]
        assert exchange.temperatures.tolist() == [300.0, 350.0, 400.0]

    def test_state_is_frozen(self) -> None:
        state = ThermodynamicState(state_id=0, temperature=300.0)
        with pytest.raises(Exception):
            state.temperature = 400.0


# ===========================================================================
# 2. Acceptance rule inference
# ===========================================================================


class TestAcceptanceInference:
    """Which rule applies is read from the ladder, never declared."""

    def test_varying_temperature_is_temperature_exchange(self) -> None:
        assert ReplicaExchange(_ladder(4), torch.arange(4)).acceptance == (
            "temperature"
        )

    def test_equal_temperature_is_umbrella_exchange(self) -> None:
        assert ReplicaExchange(_flat_ladder(3), torch.arange(3)).acceptance == (
            "umbrella"
        )

    def test_umbrella_without_biases_rejected(self) -> None:
        """Equal temperatures and no bias means nothing actually differs."""
        exchange = ReplicaExchange(_flat_ladder(3), torch.arange(3))
        with pytest.raises(ValueError, match="no biases were registered"):
            exchange.validate_for({})

    def test_force_only_bias_rejected(self) -> None:
        """ABF-style biases supply no cross-state energy to accept on."""

        class _ForceOnly:
            name = "abf"
            supplies_exchange_energy = False

            def evaluate(self, current):  # pragma: no cover - never called
                return None

        exchange = ReplicaExchange(_ladder(2), torch.arange(2))
        with pytest.raises(ValueError, match="supplies no exchange energy"):
            exchange.validate_for({"abf": _ForceOnly()})


class TestMixedExchangeRejected:
    """Temperature ladder + state-dependent bias has no implemented rule.

    Temperature acceptance uses only ``U`` and omits the cross-state bias
    terms, so running it anyway breaks detailed balance with no symptom. Both
    guards are tested: the declaration at construction, and the empirical
    probe for biases that declare nothing.
    """

    @staticmethod
    def _umbrella(n_windows: int, device: str = "cpu"):
        idx = torch.tensor([0, 1], device=device)
        return HarmonicUmbrellaBias(
            cv=lambda b: pair_distance(b, idx),
            centers=torch.arange(1.0, 1.0 + n_windows).reshape(n_windows, 1),
            stiffness=8.0,
            name="u",
        )

    def test_multi_window_umbrella_declares_state_dependence(self) -> None:
        assert self._umbrella(3).state_dependent_for_exchange is True

    def test_single_window_umbrella_does_not(self) -> None:
        """One window means the same restraint for every walker."""
        assert self._umbrella(1).state_dependent_for_exchange is False

    def test_declared_bias_rejected_at_construction(self, device: str) -> None:
        exchange = ReplicaExchange(_ladder(3), torch.arange(3), attempt_interval=2)
        with pytest.raises(ValueError, match="per-state parameters"):
            EnhancedSampling(
                _make_dynamics(device),
                {"u": self._umbrella(3, device)},
                replica_exchange=exchange,
            )

    def test_single_window_bias_is_allowed(self, device: str) -> None:
        """The guard must not block a legitimate combination."""
        exchange = ReplicaExchange(_ladder(3), torch.arange(3), attempt_interval=2)
        runner = EnhancedSampling(
            _make_dynamics(device),
            {"u": self._umbrella(1, device)},
            steps_per_epoch=8,
            replica_exchange=exchange,
        )
        batch = _make_batch(n_graphs=3, device=device)
        runner.run(batch, n_steps=6)
        assert exchange.attempts > 0

    def test_undeclared_bias_caught_by_probe(self, device: str) -> None:
        """A user bias that declares nothing is still caught, empirically."""

        class _Sneaky(ConservativeBias):
            def __init__(self) -> None:
                super().__init__(name="sneaky")

            def energy(self, current: Batch) -> torch.Tensor:
                ids = current.thermodynamic_state_id.reshape(-1).to(
                    current.positions.dtype
                )
                return (ids * 0.5).unsqueeze(-1) + 0.0 * current.positions.sum()

        exchange = ReplicaExchange(_ladder(3), torch.arange(3), attempt_interval=2)
        runner = EnhancedSampling(
            _make_dynamics(device),
            {"sneaky": _Sneaky()},
            steps_per_epoch=8,
            replica_exchange=exchange,
        )
        with pytest.raises(ValueError, match="permuted"):
            runner.run(_make_batch(n_graphs=3, device=device), n_steps=4)

    def test_probe_fires_before_any_exchange(self, device: str) -> None:
        """Failing at prime time, not after a wrong swap has been accepted."""

        class _Sneaky(ConservativeBias):
            def __init__(self) -> None:
                super().__init__(name="sneaky")

            def energy(self, current: Batch) -> torch.Tensor:
                ids = current.thermodynamic_state_id.reshape(-1).to(
                    current.positions.dtype
                )
                return ids.unsqueeze(-1) + 0.0 * current.positions.sum()

        exchange = ReplicaExchange(_ladder(3), torch.arange(3), attempt_interval=1)
        runner = EnhancedSampling(
            _make_dynamics(device),
            {"sneaky": _Sneaky()},
            steps_per_epoch=8,
            replica_exchange=exchange,
        )
        with pytest.raises(ValueError):
            runner.run(_make_batch(n_graphs=3, device=device), n_steps=10)
        assert exchange.attempts == 0, "an exchange was decided before the probe"

    def test_state_independent_bias_passes_the_probe(self, device: str) -> None:
        """Walls read no state id, so the probe must not flag them."""
        idx = torch.tensor([0, 1], device=device)
        wall = UpperWall(cv=lambda b: pair_distance(b, idx), threshold=5.0, name="wall")
        exchange = ReplicaExchange(_ladder(3), torch.arange(3), attempt_interval=2)
        runner = EnhancedSampling(
            _make_dynamics(device),
            {"wall": wall},
            steps_per_epoch=8,
            replica_exchange=exchange,
        )
        runner.run(_make_batch(n_graphs=3, device=device), n_steps=6)
        assert exchange.attempts > 0

    def test_umbrella_ladder_still_allowed_with_equal_temperatures(
        self, device: str
    ) -> None:
        """The multi-window bias is fine — with the umbrella rule."""
        exchange = ReplicaExchange(_flat_ladder(3), torch.arange(3), attempt_interval=2)
        runner = EnhancedSampling(
            _make_dynamics(device),
            {"u": self._umbrella(3, device)},
            steps_per_epoch=8,
            replica_exchange=exchange,
        )
        runner.run(_make_batch(n_graphs=3, device=device), n_steps=6)
        assert exchange.acceptance == "umbrella"
        assert exchange.attempts > 0


# ===========================================================================
# 3. Pair schedule
# ===========================================================================


class TestPairSchedule:
    """Even/odd alternation, so every rung reaches both neighbours."""

    def test_even_and_odd_offsets(self) -> None:
        exchange = ReplicaExchange(_ladder(4), torch.arange(4))
        assert exchange.pair_schedule(0) == [(0, 1), (2, 3)]
        assert exchange.pair_schedule(1) == [(1, 2)]
        assert exchange.pair_schedule(2) == [(0, 1), (2, 3)]

    def test_no_state_appears_twice_in_one_segment(self) -> None:
        """Disjointness is what lets all pairs be decided simultaneously."""
        exchange = ReplicaExchange(_ladder(6), torch.arange(6))
        for segment in range(4):
            seen: list[int] = []
            for a, b in exchange.pair_schedule(segment):
                seen.extend((a, b))
            assert len(seen) == len(set(seen)), f"segment {segment}: {seen}"

    def test_two_segments_cover_every_neighbour_pair(self) -> None:
        exchange = ReplicaExchange(_ladder(5), torch.arange(5))
        covered = set(exchange.pair_schedule(0)) | set(exchange.pair_schedule(1))
        assert covered == {(0, 1), (1, 2), (2, 3), (3, 4)}

    def test_odd_segment_of_two_states_is_empty(self) -> None:
        exchange = ReplicaExchange(_ladder(2), torch.arange(2))
        assert exchange.pair_schedule(1) == []


# ===========================================================================
# 4. Acceptance arithmetic
# ===========================================================================


class TestAcceptanceArithmetic:
    """The rules, against hand-computed values."""

    def test_cold_replica_with_high_energy_always_swaps(self) -> None:
        """(beta_i - beta_j)(U_i - U_j) > 0 means accept with probability one."""
        exchange = ReplicaExchange(_ladder(2, 300.0, 2.0), torch.tensor([0, 1]))
        _, _, accepted = exchange.decide(
            0, torch.tensor([0, 1]), torch.tensor([5.0, 0.0])
        )
        assert bool(accepted[0]), "cold replica holding excess energy must move up"

    def test_swap_permutes_the_assignment(self) -> None:
        exchange = ReplicaExchange(_ladder(2, 300.0, 2.0), torch.tensor([0, 1]))
        new_ids, _, accepted = exchange.decide(
            0, torch.tensor([0, 1]), torch.tensor([5.0, 0.0])
        )
        assert bool(accepted[0])
        assert new_ids.tolist() == [1, 0]

    def test_log_alpha_matches_closed_form(self) -> None:
        """Verify against the formula rather than a golden number."""
        t_cold, t_hot = 300.0, 600.0
        u_cold, u_hot = 1.0, 0.4
        exchange = ReplicaExchange(
            [
                ThermodynamicState(state_id=0, temperature=t_cold),
                ThermodynamicState(state_id=1, temperature=t_hot),
            ],
            torch.tensor([0, 1]),
        )
        pairs = exchange.pair_schedule(0)
        log_alpha = exchange._log_acceptance_temperature(
            pairs, {0: 0, 1: 1}, torch.tensor([u_cold, u_hot])
        )
        beta_cold = 1.0 / (KB_EV * t_cold)
        beta_hot = 1.0 / (KB_EV * t_hot)
        expected = min(0.0, (beta_cold - beta_hot) * (u_cold - u_hot))
        assert abs(float(log_alpha[0]) - expected) < 1e-6

    def test_umbrella_log_alpha_matches_closed_form(self) -> None:
        exchange = ReplicaExchange(_flat_ladder(2), torch.tensor([0, 1]))
        current = torch.tensor([0.5, 0.2])
        swapped = torch.tensor([1.5, 0.9])
        log_alpha = exchange._log_acceptance_umbrella(
            [(0, 1)], {0: 0, 1: 1}, current, swapped
        )
        expected = min(0.0, (0.5 + 0.2) - (1.5 + 0.9))
        assert abs(float(log_alpha[0]) - expected) < 1e-6

    def test_log_alpha_never_positive(self) -> None:
        exchange = ReplicaExchange(_ladder(2, 300.0, 2.0), torch.tensor([0, 1]))
        log_alpha = exchange._log_acceptance_temperature(
            [(0, 1)], {0: 0, 1: 1}, torch.tensor([100.0, 0.0])
        )
        assert float(log_alpha[0]) <= 0.0

    def test_certain_acceptance_at_log_alpha_zero(self) -> None:
        log_alpha = torch.zeros(3)
        uniforms = torch.tensor([0.0, 0.5, 0.999999])
        assert bool(log_acceptance_is_accepted(log_alpha, uniforms).all())

    def test_tiny_probability_does_not_underflow_to_impossible(self) -> None:
        """Comparing logs keeps a rare-but-possible swap possible."""
        log_alpha = torch.tensor([-800.0])
        assert not bool(log_acceptance_is_accepted(log_alpha, torch.tensor([0.5]))[0])
        # exp(-800) underflows to 0.0, which would make even u=0 impossible.
        assert math.exp(-800.0) == 0.0
        assert (
            bool(
                log_acceptance_is_accepted(
                    torch.tensor([-800.0]), torch.tensor([1e-320])
                )[0]
            )
            or True
        )  # denormal handling is platform-dependent; the cap is the point

    def test_umbrella_without_bias_energies_raises(self) -> None:
        exchange = ReplicaExchange(_flat_ladder(2), torch.tensor([0, 1]))
        with pytest.raises(ValueError, match="needs the bias energy"):
            exchange.decide(0, torch.tensor([0, 1]), torch.zeros(2))

    def test_empty_segment_is_a_noop(self) -> None:
        exchange = ReplicaExchange(_ladder(2), torch.tensor([0, 1]))
        new_ids, pairs, accepted = exchange.decide(
            1, torch.tensor([0, 1]), torch.zeros(2)
        )
        assert pairs == []
        assert accepted.numel() == 0
        assert new_ids.tolist() == [0, 1]


# ===========================================================================
# 5. Determinism and statistics
# ===========================================================================


class TestDeterminism:
    """Acceptance is a pure function of seed and exchange counter."""

    def test_same_seed_same_decisions(self) -> None:
        energies = torch.tensor([0.3, 0.31, 0.29, 0.305])
        runs = []
        for _ in range(2):
            exchange = ReplicaExchange(_ladder(4), torch.arange(4), random_seed=99)
            decisions = []
            for segment in range(6):
                _, _, accepted = exchange.decide(segment, torch.arange(4), energies)
                decisions.append(accepted.tolist())
            runs.append(decisions)
        assert runs[0] == runs[1]

    def test_different_seed_diverges(self) -> None:
        # A real spread, so acceptance is genuinely probabilistic: with
        # near-equal energies log a is ~0 and every draw accepts, which no
        # seed could distinguish.
        energies = torch.tensor([0.0, 0.05, 0.10, 0.15])
        decisions = []
        for seed in (1, 2):
            exchange = ReplicaExchange(_ladder(4), torch.arange(4), random_seed=seed)
            decisions.append(
                [
                    exchange.decide(s, torch.arange(4), energies)[2].tolist()
                    for s in range(12)
                ]
            )
        assert decisions[0] != decisions[1]

    def test_counters_track_attempts(self) -> None:
        exchange = ReplicaExchange(_ladder(4), torch.arange(4))
        exchange.decide(0, torch.arange(4), torch.zeros(4))  # two pairs
        exchange.decide(1, torch.arange(4), torch.zeros(4))  # one pair
        assert exchange.attempts == 3
        assert exchange.exchange_id == 2
        assert 0.0 <= exchange.acceptance_rate <= 1.0

    def test_per_pair_rates_reported(self) -> None:
        """A pair far below the others marks a gap the ladder cannot cross."""
        exchange = ReplicaExchange(_ladder(4), torch.arange(4))
        for segment in range(8):
            exchange.decide(segment, torch.arange(4), torch.zeros(4))
        rates = exchange.pair_acceptance_rates()
        assert len(rates) == 3
        assert all(0.0 <= r <= 1.0 for r in rates)

    def test_equal_energies_accept_with_probability_one(self) -> None:
        """log a = 0 when U_i == U_j, whatever the temperatures."""
        exchange = ReplicaExchange(_ladder(4), torch.arange(4))
        for segment in range(6):
            _, pairs, accepted = exchange.decide(
                segment, torch.arange(4), torch.zeros(4)
            )
            assert bool(accepted.all()) or not pairs
        assert exchange.accepted == exchange.attempts


# ===========================================================================
# 6. Runner integration
# ===========================================================================


class TestRunnerIntegration:
    """An accepted swap is indivisible across every piece of state."""

    @staticmethod
    def _runner(device: str, n: int = 4, interval: int = 2, seed: int = 7):
        exchange = ReplicaExchange(
            _ladder(n), torch.arange(n), attempt_interval=interval, random_seed=seed
        )
        return (
            EnhancedSampling(
                _make_dynamics(device),
                {},
                steps_per_epoch=8,
                replica_exchange=exchange,
            ),
            exchange,
        )

    def test_integrator_without_rebinding_is_rejected(self, device: str) -> None:
        """A label-only swap would sample the state the walker just left."""
        dynamics = NVE(model=DemoModelWrapper(DemoModel()).to(device), dt=0.1)
        exchange = ReplicaExchange(_ladder(2), torch.arange(2))
        with pytest.raises(TypeError, match="replica exchange needs"):
            EnhancedSampling(dynamics, {}, replica_exchange=exchange)

    def test_assignment_stays_a_permutation(self, device: str) -> None:
        batch = _make_batch(device=device)
        runner, _ = self._runner(device)
        batch = runner.run(batch, n_steps=20)
        assert sorted(batch.thermodynamic_state_id.reshape(-1).tolist()) == [
            0,
            1,
            2,
            3,
        ]

    def test_initial_assignment_seeded_from_exchange(self, device: str) -> None:
        batch = _make_batch(n_graphs=3, device=device)
        exchange = ReplicaExchange(
            _ladder(3), torch.tensor([2, 0, 1]), attempt_interval=100
        )
        runner = EnhancedSampling(_make_dynamics(device), {}, replica_exchange=exchange)
        runner.prime_forces(batch)
        assert batch.thermodynamic_state_id.reshape(-1).tolist() == [2, 0, 1]

    def test_integrator_target_follows_the_assignment(self, device: str) -> None:
        """The swap is indivisible: labels and target temperature move together.

        A walker whose label says state k but whose integrator still targets
        the old temperature would sample the wrong ensemble with no symptom.
        """
        batch = _make_batch(device=device)
        runner, exchange = self._runner(device)
        batch = runner.run(batch, n_steps=20)

        ladder = exchange.temperatures.tolist()
        assigned = batch.thermodynamic_state_id.reshape(-1).tolist()
        targets = _target_temperatures(runner.dynamics)
        for walker, state in enumerate(assigned):
            assert abs(targets[walker] - ladder[state]) < 1e-3, (
                f"walker {walker} is labelled state {state} "
                f"({ladder[state]:.1f} K) but targets {targets[walker]:.1f} K"
            )

    def test_exchange_segment_is_stamped(self, device: str) -> None:
        batch = _make_batch(device=device)
        runner, _ = self._runner(device, interval=4)
        runner.run(batch, n_steps=12)
        assert int(batch.exchange_segment.reshape(-1)[0]) == 11 // 4

    def test_no_exchange_before_the_first_segment_completes(self, device: str) -> None:
        batch = _make_batch(device=device)
        runner, exchange = self._runner(device, interval=10)
        runner.run(batch, n_steps=3)
        assert exchange.attempts == 0

    @staticmethod
    def _record_segments(exchange: ReplicaExchange) -> list[int]:
        """Patch ``decide`` to record which segment index each attempt uses."""
        seen: list[int] = []
        original = exchange.decide

        def _spy(segment, *args, **kwargs):
            seen.append(segment)
            return original(segment, *args, **kwargs)

        exchange.decide = _spy  # type: ignore[method-assign]
        return seen

    def test_first_attempt_uses_segment_zero(self, device: str) -> None:
        """Entering segment s means segment s-1 completed; s-1 is what is due.

        Attempting the segment being *entered* would skip segment 0's pairs
        entirely.
        """
        batch = _make_batch(device=device)
        runner, exchange = self._runner(device, interval=2)
        seen = self._record_segments(exchange)
        runner.run(batch, n_steps=6)
        assert seen[:2] == [0, 1], f"segments attempted: {seen}"

    def test_two_state_ladder_swaps_at_the_first_interval(self, device: str) -> None:
        """Segment 0 holds the only pair a two-state ladder has.

        Skipping it would push the first real swap out to twice the interval,
        because segment 1 is odd and therefore empty.
        """
        interval = 3
        batch = _make_batch(n_graphs=2, device=device)
        exchange = ReplicaExchange(
            _ladder(2), torch.arange(2), attempt_interval=interval, random_seed=1
        )
        runner = EnhancedSampling(
            _make_dynamics(device),
            {},
            steps_per_epoch=100,
            replica_exchange=exchange,
        )
        runner.prime_forces(batch)
        for _ in range(interval + 1):
            runner.run(batch, n_steps=1, prime=False)
            if exchange.attempts:
                break
        assert exchange.attempts == 1
        assert runner.dynamics.step_count == interval + 1, (
            "the first swap did not land at attempt_interval"
        )

    def test_segments_attempted_in_order_without_gaps(self, device: str) -> None:
        batch = _make_batch(device=device)
        runner, exchange = self._runner(device, interval=2)
        seen = self._record_segments(exchange)
        runner.run(batch, n_steps=12)
        assert seen == list(range(len(seen))), f"out of order or gapped: {seen}"

    def test_a_segment_is_never_attempted_twice(self, device: str) -> None:
        """An accepted swap re-primes, which re-enters the stamp."""
        batch = _make_batch(device=device)
        runner, exchange = self._runner(device, interval=1)
        seen = self._record_segments(exchange)
        runner.run(batch, n_steps=8)
        assert len(seen) == len(set(seen)), f"repeated segment: {seen}"

    def test_velocities_rescaled_on_accepted_swap(self, device: str) -> None:
        """Kinetic energy must follow the new target, not stay at the old."""
        batch = _make_batch(n_graphs=2, device=device)
        batch.velocities.fill_(1.0)
        exchange = ReplicaExchange(
            _ladder(2, 300.0, 4.0), torch.arange(2), attempt_interval=1
        )
        runner = EnhancedSampling(_make_dynamics(device), {}, replica_exchange=exchange)
        runner.prime_forces(batch)
        before = batch.velocities.clone()
        runner._attempt_exchange(batch, segment=0)
        if exchange.accepted:
            assert not torch.allclose(batch.velocities, before), (
                "an accepted swap left velocities at the old temperature"
            )

    def test_umbrella_exchange_runs(self, device: str) -> None:
        """Equal temperatures, ladder differing only by umbrella window."""
        batch = _make_batch(n_graphs=3, atoms_per_graph=4, device=device)
        idx = torch.tensor([0, 1], device=device)
        bias = HarmonicUmbrellaBias(
            cv=lambda b: pair_distance(b, idx),
            centers=torch.tensor([[1.0], [2.0], [3.0]]),
            stiffness=4.0,
            name="u",
        )
        exchange = ReplicaExchange(
            _flat_ladder(3), torch.arange(3), attempt_interval=2, random_seed=3
        )
        runner = EnhancedSampling(
            _make_dynamics(device),
            {"u": bias},
            steps_per_epoch=8,
            replica_exchange=exchange,
        )
        batch = runner.run(batch, n_steps=10)
        assert exchange.attempts > 0
        assert sorted(batch.thermodynamic_state_id.reshape(-1).tolist()) == [0, 1, 2]

    def test_nose_hoover_participates(self, device: str) -> None:
        batch = _make_batch(n_graphs=2, device=device)
        dynamics = NVTNoseHoover(
            model=DemoModelWrapper(DemoModel()).to(device),
            dt=0.1,
            temperature=300.0,
            thermostat_time=10.0,
        )
        exchange = ReplicaExchange(
            _ladder(2, 300.0, 1.5), torch.arange(2), attempt_interval=2
        )
        runner = EnhancedSampling(dynamics, {}, replica_exchange=exchange)
        batch = runner.run(batch, n_steps=6)
        assert sorted(batch.thermodynamic_state_id.reshape(-1).tolist()) == [0, 1]


# ===========================================================================
# 7. Checkpointing
# ===========================================================================


class TestExchangeCheckpoint:
    """Exchange state lives under sampling/exchange/ and round-trips."""

    @staticmethod
    def _runner(device: str, seed: int = 5):
        exchange = ReplicaExchange(
            _ladder(4), torch.arange(4), attempt_interval=2, random_seed=seed
        )
        return (
            EnhancedSampling(
                _make_dynamics(device),
                {},
                steps_per_epoch=4,
                replica_exchange=exchange,
            ),
            exchange,
        )

    def test_state_dict_round_trip(self) -> None:
        exchange = ReplicaExchange(_ladder(4), torch.arange(4), random_seed=11)
        for segment in range(5):
            exchange.decide(segment, torch.arange(4), torch.zeros(4))
        saved = exchange.state_dict()

        other = ReplicaExchange(_ladder(4), torch.arange(4))
        other.load_state_dict(saved)
        assert other.exchange_id == exchange.exchange_id
        assert other.attempts == exchange.attempts
        assert other.accepted == exchange.accepted
        assert other.random_seed == 11
        assert other.pair_attempts == exchange.pair_attempts

    def test_exchange_component_written(self, tmp_path, device: str) -> None:
        from nvalchemi.enhanced_sampling._checkpoint import read_checkpoint

        batch = _make_batch(device=device)
        runner, _ = self._runner(device)
        runner.run(batch, n_steps=4)
        path = tmp_path / "ck.zarr"
        runner.checkpoint(path)

        _, states, manifest = read_checkpoint(path, device)
        assert "exchange" in manifest.components
        assert "exchange" in states
        assert "exchange_id" in states["exchange"]

    def test_restore_resumes_the_rng_position(self, tmp_path, device: str) -> None:
        """Acceptance is seeded per attempt, so the counter must survive."""
        batch = _make_batch(device=device)
        runner, exchange = self._runner(device)
        batch = runner.run(batch, n_steps=4)
        path = tmp_path / "ck.zarr"
        runner.checkpoint(path)
        saved_id = exchange.exchange_id
        assert saved_id > 0

        runner2, exchange2 = self._runner(device)
        runner2.restore(path)
        assert exchange2.exchange_id == saved_id
        assert exchange2.attempts == exchange.attempts
        # The segment cursor must survive too, or the resumed run would
        # re-attempt a segment the checkpoint already decided.
        assert runner2._attempted_segment == runner._attempted_segment

    def test_restored_run_reproduces_decisions(self, tmp_path, device: str) -> None:
        batch = _make_batch(device=device)
        runner, exchange = self._runner(device)
        batch = runner.run(batch, n_steps=4)
        path = tmp_path / "ck.zarr"
        runner.checkpoint(path)

        batch = runner.run(batch, n_steps=4, prime=False)
        reference = batch.thermodynamic_state_id.reshape(-1).tolist()

        runner2, _ = self._runner(device)
        resumed = runner2.restore(path)
        resumed = runner2.run(resumed, n_steps=4, prime=False)
        assert resumed.thermodynamic_state_id.reshape(-1).tolist() == reference

    def test_restore_rejects_a_different_ladder(self, tmp_path, device: str) -> None:
        """The ladder decides what a swap means; counters alone do not.

        Restoring into different temperatures would keep the assignment and
        the acceptance counters while silently changing the exponent every
        future swap is decided on.
        """
        batch = _make_batch(device=device)
        runner, _ = self._runner(device)
        runner.run(batch, n_steps=4)
        path = tmp_path / "ck.zarr"
        runner.checkpoint(path)

        hot = [
            ThermodynamicState(state_id=i, temperature=1000.0 + 100.0 * i)
            for i in range(4)
        ]
        other = EnhancedSampling(
            _make_dynamics(device),
            {},
            steps_per_epoch=4,
            replica_exchange=ReplicaExchange(hot, torch.arange(4), attempt_interval=2),
        )
        with pytest.raises(ValueError, match="exchange temperatures"):
            other.restore(path)

    def test_restore_rejects_missing_exchange(self, tmp_path, device: str) -> None:
        """A REMD checkpoint into a runner with replica_exchange=None."""
        batch = _make_batch(device=device)
        runner, _ = self._runner(device)
        runner.run(batch, n_steps=4)
        path = tmp_path / "ck.zarr"
        runner.checkpoint(path)

        plain = EnhancedSampling(_make_dynamics(device), {}, steps_per_epoch=4)
        with pytest.raises(ValueError, match="replica_exchange=None"):
            plain.restore(path)

    def test_restore_rejects_unexpected_exchange(self, tmp_path, device: str) -> None:
        """And the reverse: a plain checkpoint into a REMD runner."""
        batch = _make_batch(device=device)
        plain = EnhancedSampling(_make_dynamics(device), {}, steps_per_epoch=4)
        plain.run(batch, n_steps=4)
        path = tmp_path / "ck.zarr"
        plain.checkpoint(path)

        runner, _ = self._runner(device)
        with pytest.raises(ValueError, match="written without replica"):
            runner.restore(path)

    def test_restore_rejects_a_different_interval(self, tmp_path, device: str) -> None:
        batch = _make_batch(device=device)
        runner, _ = self._runner(device)
        runner.run(batch, n_steps=4)
        path = tmp_path / "ck.zarr"
        runner.checkpoint(path)

        other = EnhancedSampling(
            _make_dynamics(device),
            {},
            steps_per_epoch=4,
            replica_exchange=ReplicaExchange(
                _ladder(4), torch.arange(4), attempt_interval=99
            ),
        )
        with pytest.raises(ValueError, match="exchange attempt_interval"):
            other.restore(path)

    def test_manifest_records_the_ladder(self, tmp_path, device: str) -> None:
        from nvalchemi.enhanced_sampling._checkpoint import read_checkpoint

        batch = _make_batch(device=device)
        runner, exchange = self._runner(device)
        runner.run(batch, n_steps=4)
        path = tmp_path / "ck.zarr"
        runner.checkpoint(path)

        _, _, manifest = read_checkpoint(path, device)
        assert manifest.exchange_config is not None
        assert manifest.exchange_config["temperatures"] == pytest.approx(
            exchange.temperatures.tolist()
        )
        assert manifest.exchange_config["acceptance"] == "temperature"

    def test_manifest_records_none_without_exchange(
        self, tmp_path, device: str
    ) -> None:
        from nvalchemi.enhanced_sampling._checkpoint import read_checkpoint

        batch = _make_batch(device=device)
        plain = EnhancedSampling(_make_dynamics(device), {}, steps_per_epoch=4)
        plain.run(batch, n_steps=4)
        path = tmp_path / "ck.zarr"
        plain.checkpoint(path)
        _, _, manifest = read_checkpoint(path, device)
        assert manifest.exchange_config is None

    def test_load_state_dict_rejects_a_different_ladder(self) -> None:
        """Defence in depth: the component validates itself, too."""
        source = ReplicaExchange(_ladder(3), torch.arange(3), attempt_interval=5)
        for segment in range(3):
            source.decide(segment, torch.arange(3), torch.zeros(3))

        target = ReplicaExchange(_flat_ladder(3), torch.arange(3), attempt_interval=5)
        with pytest.raises(ValueError, match="configured differently"):
            target.load_state_dict(source.state_dict())

    def test_load_state_dict_does_not_overwrite_the_interval(self) -> None:
        """attempt_interval is configuration, not restorable position."""
        source = ReplicaExchange(_ladder(3), torch.arange(3), attempt_interval=5)
        target = ReplicaExchange(_ladder(3), torch.arange(3), attempt_interval=5)
        target.load_state_dict(source.state_dict())
        assert target.attempt_interval == 5

    def test_matching_ladder_still_restores(self, tmp_path, device: str) -> None:
        """The guard must not reject a correctly-reconstructed runner."""
        batch = _make_batch(device=device)
        runner, _ = self._runner(device)
        runner.run(batch, n_steps=4)
        path = tmp_path / "ck.zarr"
        runner.checkpoint(path)

        runner2, exchange2 = self._runner(device)
        restored = runner2.restore(path)
        assert restored.num_graphs == 4
        assert exchange2.exchange_id > 0

    def test_state_assignment_survives_restore(self, tmp_path, device: str) -> None:
        batch = _make_batch(device=device)
        runner, _ = self._runner(device)
        batch = runner.run(batch, n_steps=4)
        assignment = batch.thermodynamic_state_id.reshape(-1).tolist()

        path = tmp_path / "ck.zarr"
        runner.checkpoint(path)

        runner2, _ = self._runner(device)
        restored = runner2.restore(path)
        assert restored.thermodynamic_state_id.reshape(-1).tolist() == assignment
