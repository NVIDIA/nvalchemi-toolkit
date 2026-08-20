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
"""Unit tests for the ``EnhancedSampling`` runner and the adaptive battery.

Covers walker identity stamping, the force-step ordering guarantees,
exactly-once ``update()`` delivery, observation staging, force priming,
epoch commits, and ``warm_start``.
"""

from __future__ import annotations

import pytest
import torch
from torch import Tensor

from nvalchemi.data import AtomicData, Batch
from nvalchemi.dynamics import NVTLangevin
from nvalchemi.dynamics.base import DynamicsStage
from nvalchemi.enhanced_sampling import (
    AdaptivePotentialMixin,
    BiasResult,
    ConservativeBias,
    EnhancedSampling,
    HarmonicUmbrellaBias,
    pair_distance,
)
from nvalchemi.models.demo import DemoModel, DemoModelWrapper

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_batch(
    n_graphs: int = 2,
    atoms_per_graph: int = 4,
    device: str = "cpu",
    seed: int = 0,
    with_cell: bool = False,
) -> Batch:
    """Return a batch with the output buffers dynamics writes back into."""
    torch.manual_seed(seed)
    data_list = []
    for _ in range(n_graphs):
        kwargs = {
            "positions": torch.randn(atoms_per_graph, 3),
            "atomic_numbers": torch.full((atoms_per_graph,), 6, dtype=torch.long),
            "atomic_masses": torch.ones(atoms_per_graph),
            "forces": torch.zeros(atoms_per_graph, 3),
            "energy": torch.zeros(1, 1),
        }
        if with_cell:
            kwargs["cell"] = torch.eye(3).unsqueeze(0) * 10.0
            kwargs["pbc"] = torch.tensor([[True, True, True]])
            kwargs["stress"] = torch.zeros(1, 3, 3)
        data = AtomicData(**kwargs)
        data.add_node_property("velocities", torch.zeros(atoms_per_graph, 3))
        data_list.append(data)
    return Batch.from_data_list(data_list).to(device)


def _make_dynamics(device: str = "cpu") -> NVTLangevin:
    model = DemoModelWrapper(DemoModel()).to(device)
    return NVTLangevin(model=model, dt=0.1, temperature=300.0, friction=0.1)


class _ConstantForceBias(ConservativeBias):
    """E = c * sum(x) — constant, known bias force of -c along x."""

    def __init__(self, coefficient: float = 1.0, name: str = "constant_force") -> None:
        super().__init__(name=name)
        self.coefficient = coefficient

    def energy(self, current: Batch) -> Tensor:
        ptr = current.batch_ptr
        return torch.stack(
            [
                self.coefficient * current.positions[ptr[b] : ptr[b + 1], 0].sum()
                for b in range(current.num_graphs)
            ]
        ).unsqueeze(-1)


class _RecordingAdaptiveBias(AdaptivePotentialMixin, ConservativeBias):
    """Conservative and adaptive; records every update() call for assertions."""

    def __init__(
        self,
        name: str = "recording",
        update_frequency: int = 1,
        observation_stage: DynamicsStage = DynamicsStage.AFTER_STEP,
        bump: bool = True,
    ) -> None:
        super().__init__(name=name)
        self.update_frequency = update_frequency
        self.observation_stage = observation_stage
        self._bump = bump
        self.update_steps: list[int] = []
        self.observed_forces: list[Tensor] = []
        self.commit_calls = 0

    def energy(self, current: Batch) -> Tensor:
        return (
            torch.zeros(
                current.num_graphs,
                1,
                dtype=current.positions.dtype,
                device=current.positions.device,
            )
            + 0.0 * current.positions.sum()
        )

    def update(self, frames: Batch, result: BiasResult) -> None:
        step = int(frames.sampling_step.reshape(-1)[0])
        self.update_steps.append(step)
        forces = getattr(frames, "forces", None)
        if forces is not None:
            self.observed_forces.append(forces.clone())
        if self._bump:
            self.bump_state_version()

    def commit_epoch(self) -> None:
        self.commit_calls += 1


# ===========================================================================
# 1. Construction and validation
# ===========================================================================


class TestRunnerConstruction:
    """Runner rejects malformed bias mappings at construction."""

    def test_empty_biases_allowed(self) -> None:
        runner = EnhancedSampling(_make_dynamics(), {})
        assert runner.biases == {}

    def test_none_biases_allowed(self) -> None:
        assert EnhancedSampling(_make_dynamics()).biases == {}

    def test_non_protocol_bias_raises(self) -> None:
        class NotABias:
            pass

        with pytest.raises(TypeError, match="does not satisfy the BiasPotential"):
            EnhancedSampling(_make_dynamics(), {"x": NotABias()})

    def test_key_name_mismatch_raises(self) -> None:
        bias = _ConstantForceBias(name="actual_name")
        with pytest.raises(ValueError, match="key and the bias name must agree"):
            EnhancedSampling(_make_dynamics(), {"different_key": bias})

    @pytest.mark.parametrize("steps_per_epoch", [0, -1, -10])
    def test_non_positive_steps_per_epoch_rejected(self, steps_per_epoch: int) -> None:
        """It is a divisor: zero raises deep in a run rather than here."""
        with pytest.raises(ValueError, match="steps_per_epoch must be at least 1"):
            EnhancedSampling(_make_dynamics(), {}, steps_per_epoch=steps_per_epoch)

    def test_steps_per_epoch_of_one_is_allowed(self) -> None:
        runner = EnhancedSampling(_make_dynamics(), {}, steps_per_epoch=1)
        assert runner.steps_per_epoch == 1

    def test_hook_inserted_at_front(self) -> None:
        """The bias hook must run before any other AFTER_COMPUTE hook."""
        dynamics = _make_dynamics()
        runner = EnhancedSampling(dynamics, {})
        assert dynamics.hooks[0] is runner._hook

    def test_repr_lists_biases(self) -> None:
        runner = EnhancedSampling(
            _make_dynamics(), {"cf": _ConstantForceBias(name="cf")}
        )
        assert "cf" in repr(runner)


# ===========================================================================
# 2. Walker identity
# ===========================================================================


class TestWalkerIdentity:
    """Identity fields are stamped, and the persistent ones stay persistent."""

    def test_fields_stamped(self, device: str) -> None:
        batch = _make_batch(n_graphs=3, device=device)
        runner = EnhancedSampling(_make_dynamics(device), {})
        runner.run(batch, n_steps=1)

        for field in (
            "walker_id",
            "thermodynamic_state_id",
            "sampling_step",
            "exchange_segment",
            "sampling_epoch",
        ):
            value = getattr(batch, field, None)
            assert value is not None, f"{field} not stamped"
            assert value.reshape(-1).shape == (3,)

    def test_walker_ids_unique_and_stable(self, device: str) -> None:
        """walker_id is an identity: assigned once, never reshuffled."""
        batch = _make_batch(n_graphs=4, device=device)
        runner = EnhancedSampling(_make_dynamics(device), {})
        runner.run(batch, n_steps=1)
        first = batch.walker_id.clone()
        assert len(set(first.reshape(-1).tolist())) == 4

        runner.run(batch, n_steps=3, prime=False)
        assert torch.equal(batch.walker_id, first)

    def test_user_supplied_state_ids_preserved(self, device: str) -> None:
        """A caller assigning windows must not have them overwritten."""
        batch = _make_batch(n_graphs=3, device=device)
        batch["thermodynamic_state_id"] = torch.tensor([2, 0, 1], device=device)
        runner = EnhancedSampling(_make_dynamics(device), {})
        runner.run(batch, n_steps=2)
        assert batch.thermodynamic_state_id.reshape(-1).tolist() == [2, 0, 1]

    def test_sampling_step_tracks_dynamics(self, device: str) -> None:
        batch = _make_batch(device=device)
        dynamics = _make_dynamics(device)
        runner = EnhancedSampling(dynamics, {})
        runner.run(batch, n_steps=5)
        assert int(batch.sampling_step.reshape(-1)[0]) == dynamics.step_count - 1

    def test_epoch_advances_with_steps_per_epoch(self, device: str) -> None:
        batch = _make_batch(device=device)
        runner = EnhancedSampling(_make_dynamics(device), {}, steps_per_epoch=3)
        runner.run(batch, n_steps=7)
        assert int(batch.sampling_epoch.reshape(-1)[0]) == 6 // 3


# ===========================================================================
# 3. Force-step ordering
# ===========================================================================


class TestForceStepOrdering:
    """The guarantees that make multi-bias aggregation order-independent."""

    def test_bias_force_applied_to_batch(self, device: str) -> None:
        """Total force is physical + bias, with the documented sign."""
        batch = _make_batch(n_graphs=1, atoms_per_graph=3, device=device)
        dynamics = _make_dynamics(device)
        runner = EnhancedSampling(dynamics, {"cf": _ConstantForceBias(2.0, name="cf")})
        runner.prime_forces(batch)

        physical = runner.last_outputs["physical/forces"]
        total = batch.forces
        # E = 2*sum(x)  =>  F = -dE/dx = -2 on the x component only.
        expected_bias = torch.zeros_like(total)
        expected_bias[:, 0] = -2.0
        assert torch.allclose(total - physical, expected_bias, atol=1e-5)

    def test_registration_order_does_not_change_total(self, device: str) -> None:
        """Two biases summed against unmodified outputs commute."""
        results = []
        for order in ([("a", 1.0), ("b", 3.0)], [("b", 3.0), ("a", 1.0)]):
            batch = _make_batch(n_graphs=1, atoms_per_graph=3, device=device, seed=7)
            biases = {n: _ConstantForceBias(c, name=n) for n, c in order}
            runner = EnhancedSampling(_make_dynamics(device), biases)
            runner.prime_forces(batch)
            results.append(batch.forces.clone())
        assert torch.allclose(results[0], results[1], atol=1e-6)

    def test_bias_cannot_observe_another_bias_force(self, device: str) -> None:
        """Every bias sees the same unmodified physical forces."""
        seen: list[Tensor] = []

        class _ForceReadingBias(ConservativeBias):
            def __init__(self) -> None:
                super().__init__(name="reader")

            def energy(self, current: Batch) -> Tensor:
                seen.append(current.forces.clone())
                return (
                    torch.zeros(current.num_graphs, 1, device=current.positions.device)
                    + 0.0 * current.positions.sum()
                )

        batch = _make_batch(n_graphs=1, atoms_per_graph=3, device=device)
        runner = EnhancedSampling(
            _make_dynamics(device),
            {"cf": _ConstantForceBias(5.0, name="cf"), "reader": _ForceReadingBias()},
        )
        runner.prime_forces(batch)

        physical = runner.last_outputs["physical/forces"]
        assert seen, "reader bias never ran"
        for observed in seen:
            assert torch.allclose(observed, physical, atol=1e-6), (
                "a bias observed another bias's force contribution"
            )

    def test_diagnostics_namespaced(self, device: str) -> None:
        batch = _make_batch(device=device)
        runner = EnhancedSampling(
            _make_dynamics(device), {"cf": _ConstantForceBias(name="cf")}
        )
        runner.prime_forces(batch)
        keys = set(runner.last_outputs)
        assert "physical/forces" in keys
        assert "bias/cf/forces" in keys
        assert "bias_total/forces" in keys
        assert "total/forces" in keys

    def test_total_is_physical_plus_bias(self, device: str) -> None:
        """'total' must mean physical + bias, not the bias sum alone.

        These differ by exactly the physical contribution, so a caller
        plotting 'total/energy' as the system energy would silently get only
        the restraint term.
        """
        batch = _make_batch(n_graphs=2, atoms_per_graph=3, device=device)
        runner = EnhancedSampling(
            _make_dynamics(device),
            {
                "a": _ConstantForceBias(1.0, name="a"),
                "b": _ConstantForceBias(3.0, name="b"),
            },
        )
        runner.prime_forces(batch)

        physical = runner.last_outputs["physical/forces"]
        bias_total = runner.last_outputs["bias_total/forces"]
        total = runner.last_outputs["total/forces"]

        assert torch.allclose(total, physical + bias_total, atol=1e-6)
        assert torch.allclose(total, batch.forces, atol=1e-6)
        assert not torch.allclose(total, bias_total, atol=1e-6), (
            "total/* is still the bias sum; physical contribution missing"
        )

    def test_bias_total_is_sum_across_biases(self, device: str) -> None:
        batch = _make_batch(n_graphs=1, atoms_per_graph=3, device=device)
        runner = EnhancedSampling(
            _make_dynamics(device),
            {
                "a": _ConstantForceBias(1.0, name="a"),
                "b": _ConstantForceBias(3.0, name="b"),
            },
        )
        runner.prime_forces(batch)
        summed = (
            runner.last_outputs["bias/a/forces"] + runner.last_outputs["bias/b/forces"]
        )
        assert torch.allclose(
            runner.last_outputs["bias_total/forces"], summed, atol=1e-6
        )

    def test_total_energy_matches_batch(self, device: str) -> None:
        batch = _make_batch(n_graphs=2, device=device)
        runner = EnhancedSampling(
            _make_dynamics(device), {"cf": _ConstantForceBias(2.0, name="cf")}
        )
        runner.prime_forces(batch)
        assert torch.allclose(
            runner.last_outputs["total/energy"].reshape(batch.energy.shape),
            batch.energy,
            atol=1e-6,
        )

    def test_total_snapshot_not_aliased_to_batch(self, device: str) -> None:
        """A later in-place hook must not retroactively rewrite the record."""
        batch = _make_batch(device=device)
        runner = EnhancedSampling(
            _make_dynamics(device), {"cf": _ConstantForceBias(name="cf")}
        )
        runner.prime_forces(batch)
        recorded = runner.last_outputs["total/forces"].clone()
        batch.forces.mul_(0.0)
        assert torch.allclose(runner.last_outputs["total/forces"], recorded)

    def test_observables_namespaced_by_bias_name(self, device: str) -> None:
        """Two biases emitting the same observable name must not collide."""

        class _ObservableBias(ConservativeBias):
            def __init__(self, name: str) -> None:
                super().__init__(name=name)

            def energy(self, current: Batch) -> Tensor:
                return (
                    torch.zeros(current.num_graphs, 1, device=current.positions.device)
                    + 0.0 * current.positions.sum()
                )

            def evaluate(self, current: Batch) -> BiasResult:
                base = super().evaluate(current)
                import dataclasses

                return dataclasses.replace(
                    base, observables={"cv": torch.zeros(current.num_graphs)}
                )

        batch = _make_batch(device=device)
        runner = EnhancedSampling(
            _make_dynamics(device),
            {"first": _ObservableBias("first"), "second": _ObservableBias("second")},
        )
        runner.prime_forces(batch)
        assert "bias/first/cv" in runner.last_outputs
        assert "bias/second/cv" in runner.last_outputs

    def test_virial_result_rejected_with_named_error(self, device: str) -> None:
        """The runner applies stress; a virial has no volume here to convert."""

        class _VirialBias:
            name = "virial_bias"

            def evaluate(self, current: Batch) -> BiasResult:
                return BiasResult(virial=torch.zeros(current.num_graphs, 3, 3))

        batch = _make_batch(device=device)
        runner = EnhancedSampling(
            _make_dynamics(device), {"virial_bias": _VirialBias()}
        )
        with pytest.raises(ValueError, match="applies 'stress' to the batch"):
            runner.prime_forces(batch)

    def test_stress_applied_for_periodic_batch(self, device: str) -> None:
        batch = _make_batch(n_graphs=1, device=device, with_cell=True)
        idx = torch.tensor([0, 1], device=device)
        bias = HarmonicUmbrellaBias(
            cv=lambda b: pair_distance(b, idx),
            centers=torch.tensor([1.0]),
            stiffness=4.0,
            name="umbrella",
        )
        runner = EnhancedSampling(_make_dynamics(device), {"umbrella": bias})
        runner.prime_forces(batch)
        assert "bias/umbrella/stress" in runner.last_outputs
        assert torch.count_nonzero(runner.last_outputs["bias/umbrella/stress"]) > 0


# ===========================================================================
# 4. Priming
# ===========================================================================


class TestMissingDestinationBuffers:
    """A bias output with nowhere to go must fail loudly, never silently."""

    @staticmethod
    def _periodic_batch_without(field: str, device: str) -> Batch:
        """Return a periodic batch missing exactly one output buffer."""
        kwargs = {
            "positions": torch.tensor([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]]),
            "atomic_numbers": torch.tensor([6, 6], dtype=torch.long),
            "atomic_masses": torch.ones(2),
            "forces": torch.zeros(2, 3),
            "energy": torch.zeros(1, 1),
            "cell": torch.eye(3).unsqueeze(0) * 10.0,
            "pbc": torch.tensor([[True, True, True]]),
            "stress": torch.zeros(1, 3, 3),
        }
        kwargs.pop(field)
        data = AtomicData(**kwargs)
        data.add_node_property("velocities", torch.zeros(2, 3))
        return Batch.from_data_list([data]).to(device)

    @staticmethod
    def _umbrella(device: str, **kwargs) -> HarmonicUmbrellaBias:
        idx = torch.tensor([0, 1], device=device)
        return HarmonicUmbrellaBias(
            cv=lambda b: pair_distance(b, idx),
            centers=torch.tensor([2.0]),
            stiffness=5.0,
            name="umbrella",
            **kwargs,
        )

    def test_missing_stress_buffer_raises(self, device: str) -> None:
        """The dangerous case: a periodic bias stress with nowhere to land.

        Silently dropping it makes the bias invisible to an NPT/NPH barostat
        — the same failure the deprecated BiasedPotentialHook has, reached
        from a different direction.
        """
        batch = self._periodic_batch_without("stress", device)
        runner = EnhancedSampling(
            _make_dynamics(device), {"umbrella": self._umbrella(device)}
        )
        with pytest.raises(ValueError, match="no destination buffer"):
            runner.prime_forces(batch)

    def test_missing_stress_error_names_bias_and_barostat_risk(
        self, device: str
    ) -> None:
        batch = self._periodic_batch_without("stress", device)
        runner = EnhancedSampling(
            _make_dynamics(device), {"umbrella": self._umbrella(device)}
        )
        with pytest.raises(ValueError) as excinfo:
            runner.prime_forces(batch)
        message = str(excinfo.value)
        assert "'stress'" in message
        assert "umbrella" in message
        assert "barostat" in message
        assert "compute_stress=False" in message

    def test_compute_stress_false_is_the_documented_escape(self, device: str) -> None:
        """An NVT run with a periodic cell may legitimately want no stress."""
        batch = self._periodic_batch_without("stress", device)
        runner = EnhancedSampling(
            _make_dynamics(device),
            {"umbrella": self._umbrella(device, compute_stress=False)},
        )
        runner.prime_forces(batch)
        assert "bias_total/stress" not in runner.last_outputs

    def test_missing_energy_buffer_raises(self, device: str) -> None:
        batch = self._periodic_batch_without("energy", device)
        runner = EnhancedSampling(
            _make_dynamics(device), {"cf": _ConstantForceBias(name="cf")}
        )
        with pytest.raises(ValueError, match="'energy'"):
            runner.prime_forces(batch)

    def test_error_fires_before_any_step_is_taken(self, device: str) -> None:
        """Priming is what makes this a setup error, not a mid-run surprise."""
        batch = self._periodic_batch_without("stress", device)
        dynamics = _make_dynamics(device)
        runner = EnhancedSampling(dynamics, {"umbrella": self._umbrella(device)})
        with pytest.raises(ValueError, match="no destination buffer"):
            runner.run(batch, n_steps=100)
        assert dynamics.step_count == 0, "a step ran before the error surfaced"

    def test_allocated_stress_buffer_receives_contribution(self, device: str) -> None:
        """The positive case: with the buffer present, stress lands on it."""
        batch = _make_batch(n_graphs=1, device=device, with_cell=True)
        runner = EnhancedSampling(
            _make_dynamics(device), {"umbrella": self._umbrella(device)}
        )
        runner.prime_forces(batch)
        assert torch.count_nonzero(batch.stress) > 0


class TestPriming:
    """Force priming, and the error when buffers are missing."""

    def test_missing_forces_buffer_raises_named_error(self, device: str) -> None:
        data = AtomicData(
            positions=torch.randn(3, 3),
            atomic_numbers=torch.full((3,), 6, dtype=torch.long),
        )
        batch = Batch.from_data_list([data]).to(device)
        runner = EnhancedSampling(_make_dynamics(device), {})
        with pytest.raises(ValueError, match="batch has no 'forces' field"):
            runner.prime_forces(batch)

    def test_prime_populates_total_force(self, device: str) -> None:
        batch = _make_batch(device=device)
        assert torch.count_nonzero(batch.forces) == 0
        runner = EnhancedSampling(
            _make_dynamics(device), {"cf": _ConstantForceBias(name="cf")}
        )
        runner.prime_forces(batch)
        assert torch.count_nonzero(batch.forces) > 0

    def test_run_primes_by_default(self, device: str) -> None:
        """Without priming, step 0 would integrate against a zero force buffer."""
        batch = _make_batch(device=device)
        runner = EnhancedSampling(
            _make_dynamics(device), {"cf": _ConstantForceBias(name="cf")}
        )
        runner.run(batch, n_steps=1)
        assert runner.last_outputs, "no force evaluation recorded"

    def test_prime_is_idempotent(self, device: str) -> None:
        batch = _make_batch(device=device)
        runner = EnhancedSampling(
            _make_dynamics(device), {"cf": _ConstantForceBias(name="cf")}
        )
        runner.prime_forces(batch)
        first = batch.forces.clone()
        runner.prime_forces(batch)
        assert torch.allclose(batch.forces, first, atol=1e-6), (
            "priming twice at the same coordinates must not accumulate the bias"
        )


# ===========================================================================
# 5. Adaptive biases: update exactly once
# ===========================================================================


class TestAdaptiveUpdates:
    """update() delivery, staging, and epoch commits."""

    def test_update_called_once_per_step(self, device: str) -> None:
        batch = _make_batch(device=device)
        bias = _RecordingAdaptiveBias(update_frequency=1)
        runner = EnhancedSampling(_make_dynamics(device), {"recording": bias})
        runner.run(batch, n_steps=5)
        assert bias.update_steps == sorted(bias.update_steps)
        assert len(bias.update_steps) == len(set(bias.update_steps)), (
            f"update() delivered more than once for some step: {bias.update_steps}"
        )
        assert len(bias.update_steps) == 5

    def test_update_frequency_respected(self, device: str) -> None:
        batch = _make_batch(device=device)
        bias = _RecordingAdaptiveBias(update_frequency=3)
        runner = EnhancedSampling(_make_dynamics(device), {"recording": bias})
        runner.run(batch, n_steps=9)
        assert all(step % 3 == 0 for step in bias.update_steps), bias.update_steps

    def test_after_compute_observation_sees_unbiased_forces(self, device: str) -> None:
        """ABF's requirement: observe physical forces, never the bias's own."""
        batch = _make_batch(n_graphs=1, atoms_per_graph=3, device=device)
        observer = _RecordingAdaptiveBias(
            name="observer", observation_stage=DynamicsStage.AFTER_COMPUTE
        )
        runner = EnhancedSampling(
            _make_dynamics(device),
            {"cf": _ConstantForceBias(5.0, name="cf"), "observer": observer},
        )
        runner.run(batch, n_steps=2)

        assert observer.observed_forces
        physical = runner.last_outputs["physical/forces"]
        # The constant bias adds -5 to every x component; an observation that
        # captured post-application forces would differ by exactly that.
        for observed in observer.observed_forces:
            assert not torch.allclose(
                observed[:, 0], physical[:, 0] - 5.0, atol=1e-6
            ), "AFTER_COMPUTE observation captured biased forces"

    def test_after_step_update_receives_its_own_bias_result(self, device: str) -> None:
        """update() must get the result its bias produced, at either stage.

        AFTER_STEP is the default and what metadynamics uses; a bias sizing
        its next hill from the bias energy it just applied needs the real
        value, not an empty placeholder.
        """
        received: list[BiasResult] = []

        class _ResultRecordingBias(AdaptivePotentialMixin, _ConstantForceBias):
            def __init__(self) -> None:
                super().__init__(2.0, name="recorder")

            def update(self, frames: Batch, result: BiasResult) -> None:
                received.append(result)

        batch = _make_batch(n_graphs=1, atoms_per_graph=3, device=device)
        runner = EnhancedSampling(
            _make_dynamics(device), {"recorder": _ResultRecordingBias()}
        )
        runner.run(batch, n_steps=2)

        assert received, "update() never called"
        for result in received:
            assert result.energy is not None, "update() got an empty BiasResult"
            assert result.forces is not None
            # E = 2 * sum(x)  =>  F = -2 on x only.
            assert torch.allclose(
                result.forces[:, 0],
                torch.full_like(result.forces[:, 0], -2.0),
                atol=1e-5,
            )

    def test_after_compute_update_also_receives_its_result(self, device: str) -> None:
        received: list[BiasResult] = []

        class _ResultRecordingBias(AdaptivePotentialMixin, _ConstantForceBias):
            observation_stage = DynamicsStage.AFTER_COMPUTE

            def __init__(self) -> None:
                super().__init__(3.0, name="recorder")

            def update(self, frames: Batch, result: BiasResult) -> None:
                received.append(result)

        batch = _make_batch(n_graphs=1, atoms_per_graph=3, device=device)
        runner = EnhancedSampling(
            _make_dynamics(device), {"recorder": _ResultRecordingBias()}
        )
        runner.run(batch, n_steps=2)

        assert received
        for result in received:
            assert result.energy is not None
            assert torch.allclose(
                result.forces[:, 0],
                torch.full_like(result.forces[:, 0], -3.0),
                atol=1e-5,
            )

    def test_each_bias_gets_its_own_result_not_anothers(self, device: str) -> None:
        """With several adaptive biases, results must not be crossed."""
        seen: dict[str, list[float]] = {"a": [], "b": []}

        def _make(tag: str, coefficient: float):
            class _Bias(AdaptivePotentialMixin, _ConstantForceBias):
                def __init__(self) -> None:
                    super().__init__(coefficient, name=tag)

                def update(self, frames: Batch, result: BiasResult) -> None:
                    seen[tag].append(float(result.forces[0, 0]))

            return _Bias()

        batch = _make_batch(n_graphs=1, atoms_per_graph=3, device=device)
        runner = EnhancedSampling(
            _make_dynamics(device), {"a": _make("a", 1.0), "b": _make("b", 4.0)}
        )
        runner.run(batch, n_steps=2)

        assert all(abs(v - (-1.0)) < 1e-5 for v in seen["a"]), seen["a"]
        assert all(abs(v - (-4.0)) < 1e-5 for v in seen["b"]), seen["b"]

    def test_non_adaptive_bias_never_asked_to_update(self, device: str) -> None:
        batch = _make_batch(device=device)
        runner = EnhancedSampling(
            _make_dynamics(device), {"cf": _ConstantForceBias(name="cf")}
        )
        runner.run(batch, n_steps=3)  # must not raise NotImplementedError
        assert runner._adaptive_biases() == {}

    def test_commit_epoch_fires_at_boundary(self, device: str) -> None:
        batch = _make_batch(device=device)
        bias = _RecordingAdaptiveBias()
        runner = EnhancedSampling(
            _make_dynamics(device), {"recording": bias}, steps_per_epoch=2
        )
        runner.run(batch, n_steps=6)
        assert bias.commit_calls >= 2, (
            f"commit_epoch fired {bias.commit_calls} times over 3 epochs"
        )

    def test_state_version_bump_triggers_reprime(self, device: str) -> None:
        batch = _make_batch(device=device)
        bias = _RecordingAdaptiveBias(bump=True)
        runner = EnhancedSampling(
            _make_dynamics(device), {"recording": bias}, prime_after_update=True
        )
        runner.run(batch, n_steps=2)
        assert bias.state_version == 2

    def test_no_bump_no_reprime(self, device: str) -> None:
        """An update that does not change the applied bias must not re-prime."""
        batch = _make_batch(device=device)
        bias = _RecordingAdaptiveBias(bump=False)
        runner = EnhancedSampling(_make_dynamics(device), {"recording": bias})
        runner.run(batch, n_steps=3)
        assert bias.state_version == 0
        assert len(bias.update_steps) == 3


# ===========================================================================
# 6. warm_start and state
# ===========================================================================


class TestWarmStart:
    """Approximate continuation from prior frames."""

    def test_replays_frames_in_order(self, device: str) -> None:
        history = _make_batch(n_graphs=4, device=device)
        history["sampling_step"] = torch.arange(4, device=device)
        bias = _RecordingAdaptiveBias()
        runner = EnhancedSampling(_make_dynamics(device), {"recording": bias})
        runner.warm_start(history)
        assert bias.update_steps == [0, 1, 2, 3]

    def test_warm_start_without_adaptive_is_noop(self, device: str) -> None:
        history = _make_batch(n_graphs=2, device=device)
        runner = EnhancedSampling(
            _make_dynamics(device), {"cf": _ConstantForceBias(name="cf")}
        )
        runner.warm_start(history)  # must not raise

    def test_warm_start_after_restore_raises(self, device: str) -> None:
        runner = EnhancedSampling(_make_dynamics(device), {})
        runner._restored = True
        with pytest.raises(RuntimeError, match="mutually exclusive"):
            runner.warm_start(_make_batch(device=device))

    def test_checkpoint_requires_a_batch(self) -> None:
        """checkpoint() is implemented; it needs something to save."""
        runner = EnhancedSampling(_make_dynamics(), {})
        with pytest.raises(RuntimeError, match="no batch to save"):
            runner.checkpoint("x.zarr")

    def test_state_dict_includes_bias_state(self) -> None:
        bias = _RecordingAdaptiveBias()
        bias.bump_state_version()
        runner = EnhancedSampling(_make_dynamics(), {"recording": bias})
        state = runner.state_dict()
        assert state["biases"]["recording"]["state_version"] == 1


class TestAdaptiveMixinComposition:
    """The mixin must compose with nn.Module without losing state."""

    def test_wrong_mro_order_raises(self) -> None:
        """nn.Module.state_dict would otherwise silently shadow the mixin's."""
        with pytest.raises(TypeError, match="must come before nn.Module"):

            class Wrong(ConservativeBias, AdaptivePotentialMixin):
                def energy(self, current: Batch) -> Tensor:
                    return torch.zeros(current.num_graphs, 1)

    def test_correct_mro_order_accepted(self) -> None:
        class Right(AdaptivePotentialMixin, ConservativeBias):
            def energy(self, current: Batch) -> Tensor:
                return torch.zeros(current.num_graphs, 1)

        assert issubclass(Right, ConservativeBias)

    def test_state_dict_merges_buffers_and_history(self) -> None:
        """Both halves must survive one round trip."""

        class _Both(AdaptivePotentialMixin, ConservativeBias):
            def __init__(self) -> None:
                super().__init__(name="both")
                self.register_buffer("center", torch.tensor([2.0]))

            def energy(self, current: Batch) -> Tensor:
                return torch.zeros(current.num_graphs, 1)

            def update(self, frames: Batch, result: BiasResult) -> None:
                self.bump_state_version()

        bias = _Both()
        bias.bump_state_version()
        bias.bump_state_version()
        state = bias.state_dict()
        assert "center" in state, "nn.Module buffers dropped"
        assert state["state_version"] == 2, "bias history dropped"

        restored = _Both()
        restored.load_state_dict(state)
        assert restored.state_version == 2
        assert torch.allclose(restored.center, torch.tensor([2.0]))

    def test_load_state_dict_does_not_trip_strict_module_check(self) -> None:
        """'state_version' must be stripped before nn.Module sees the mapping."""

        class _Both(AdaptivePotentialMixin, ConservativeBias):
            def __init__(self) -> None:
                super().__init__(name="both")
                self.register_buffer("center", torch.zeros(1))

            def energy(self, current: Batch) -> Tensor:
                return torch.zeros(current.num_graphs, 1)

            def update(self, frames: Batch, result: BiasResult) -> None:
                pass

        bias = _Both()
        bias.load_state_dict(_Both().state_dict())  # must not raise

    def test_adaptive_without_module_half(self) -> None:
        """A non-conservative adaptive bias needs no nn.Module at all."""

        class _ForceOnlyAdaptive(AdaptivePotentialMixin):
            name = "force_only"

            def evaluate(self, current: Batch) -> BiasResult:
                return BiasResult(forces=torch.zeros_like(current.positions))

            def update(self, frames: Batch, result: BiasResult) -> None:
                self.bump_state_version()

        bias = _ForceOnlyAdaptive()
        assert not isinstance(bias, torch.nn.Module)
        assert bias.state_dict() == {"state_version": 0}


# ===========================================================================
# 8. compile_biases
# ===========================================================================


class TestCompileBiases:
    """compile_biases wraps energy(), never evaluate()."""

    def test_compiled_run_matches_eager(self, device: str) -> None:
        torch._dynamo.reset()
        results = []
        for compile_biases in (False, True):
            batch = _make_batch(n_graphs=1, atoms_per_graph=3, device=device, seed=3)
            runner = EnhancedSampling(
                _make_dynamics(device),
                {"cf": _ConstantForceBias(2.0, name="cf")},
                compile_biases=compile_biases,
            )
            runner.prime_forces(batch)
            results.append(batch.forces.clone())
        assert torch.allclose(results[0], results[1], atol=1e-5)
