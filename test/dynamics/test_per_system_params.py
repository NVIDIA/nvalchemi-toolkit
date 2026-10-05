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
"""Unit tests for ``BaseDynamics.apply_per_system_params``.

The adapter any method that permutes per-system state across walkers depends
on — replica exchange over a temperature ladder, basin hopping with swaps, an
annealing schedule.  What is asserted here is the part that makes it safe to
build on: the rebinding is indivisible, so the target, the velocities, and
any thermostat memory move in one call or not at all.
"""

from __future__ import annotations

import math

import pytest
import torch

from nvalchemi.data import AtomicData, Batch
from nvalchemi.dynamics import NVTLangevin, NVTNoseHoover
from nvalchemi.dynamics.base import BaseDynamics
from nvalchemi.models.demo import DemoModel, DemoModelWrapper


def _make_batch(
    n_graphs: int = 2, atoms_per_graph: int = 4, device: str = "cpu"
) -> Batch:
    """Return a batch with the fields an integrator reads and writes."""
    torch.manual_seed(0)
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


def _langevin(device: str = "cpu") -> NVTLangevin:
    """Return an initialised Langevin integrator."""
    model = DemoModelWrapper(DemoModel()).to(device)
    return NVTLangevin(model=model, dt=0.1, temperature=300.0, friction=0.1)


def _nose_hoover(device: str = "cpu") -> NVTNoseHoover:
    """Return an initialised Nosé-Hoover integrator."""
    model = DemoModelWrapper(DemoModel()).to(device)
    return NVTNoseHoover(model=model, dt=0.1, temperature=300.0, thermostat_time=10.0)


class TestBaseRefuses:
    """An integrator that cannot rebind must fail, not accept silently."""

    def test_base_dynamics_raises(self, device: str) -> None:
        dynamics = BaseDynamics(DemoModelWrapper(DemoModel()).to(device))
        batch = _make_batch(device=device)
        with pytest.raises(NotImplementedError, match="does not support"):
            dynamics.apply_per_system_params(
                {"temperature": torch.tensor([300.0], device=device)}, batch
            )

    def test_an_unrebindable_parameter_raises(self, device: str) -> None:
        """Silently ignoring it would leave the walker in the state it left."""
        batch = _make_batch(device=device)
        dynamics = _langevin(device)
        dynamics._ensure_state_initialized(batch)
        with pytest.raises(KeyError, match="cannot rebind"):
            dynamics.apply_per_system_params(
                {
                    "temperature": torch.tensor([300.0, 300.0], device=device),
                    "timestep": torch.tensor([0.5, 0.5], device=device),
                },
                batch,
            )

    def test_rebinding_before_initialisation_raises(self, device: str) -> None:
        batch = _make_batch(device=device)
        dynamics = _langevin(device)
        with pytest.raises(RuntimeError, match="not initialised"):
            dynamics.apply_per_system_params(
                {"temperature": torch.tensor([300.0, 300.0], device=device)}, batch
            )


class TestTemperatureIsValidated:
    """A target that is not a temperature must not reach the state.

    None of these fail on their own: a negative target makes the velocity
    rescale imaginary, zero makes a thermostat chain massless, infinity
    reaches the velocities directly — and the run carries on until ``nan``
    coordinates surface somewhere else entirely.
    """

    @pytest.mark.parametrize(
        ("label", "value"),
        [
            ("negative", -100.0),
            ("zero", 0.0),
            ("infinite", float("inf")),
            ("nan", float("nan")),
        ],
    )
    @pytest.mark.parametrize("make", [_langevin, _nose_hoover])
    def test_it_is_refused(
        self, make: object, label: str, value: float, device: str
    ) -> None:
        batch = _make_batch(device=device)
        dynamics = make(device)
        dynamics._ensure_state_initialized(batch)
        with pytest.raises(ValueError, match="positive and finite"):
            dynamics.apply_per_system_params(
                {"temperature": torch.tensor([value, 300.0], device=device)}, batch
            )

    @pytest.mark.parametrize("make", [_langevin, _nose_hoover])
    def test_a_refusal_leaves_the_state_untouched(
        self, make: object, device: str
    ) -> None:
        """What ``apply_per_system_params`` promises, and what the swap hook
        builds its own atomicity on."""
        batch = _make_batch(device=device)
        dynamics = make(device)
        dynamics._ensure_state_initialized(batch)
        before_target = dynamics._state.temperature.clone()
        before_velocities = batch.velocities.clone()
        before_chain = (
            dynamics._state.nhc_Q.clone() if hasattr(dynamics._state, "nhc_Q") else None
        )

        with pytest.raises(ValueError, match="positive and finite"):
            dynamics.apply_per_system_params(
                {"temperature": torch.tensor([-1.0, 300.0], device=device)}, batch
            )

        assert torch.equal(dynamics._state.temperature, before_target)
        assert torch.equal(batch.velocities, before_velocities)
        if before_chain is not None:
            assert torch.equal(dynamics._state.nhc_Q, before_chain), (
                "the chain masses were scaled before the target was checked"
            )

    @pytest.mark.parametrize("make", [_langevin, _nose_hoover])
    def test_a_valid_temperature_still_applies(self, make: object, device: str) -> None:
        """The guard must not be stricter than the contract it enforces."""
        batch = _make_batch(n_graphs=1, atoms_per_graph=3, device=device)
        batch.velocities.fill_(1.0)
        dynamics = make(device)
        dynamics._ensure_state_initialized(batch)
        dynamics.apply_per_system_params(
            {"temperature": torch.tensor([1200.0], device=device)}, batch
        )
        assert torch.allclose(
            batch.velocities, torch.full_like(batch.velocities, 2.0), atol=1e-5
        )

    @pytest.mark.parametrize("make", [_langevin, _nose_hoover])
    def test_both_integrators_share_one_check(self, make: object, device: str) -> None:
        """Duplicated validation is validation that drifts."""
        from nvalchemi.dynamics.base import BaseDynamics

        dynamics = make(device)
        assert type(dynamics)._validated_temperature is (
            BaseDynamics._validated_temperature
        )


class TestLangevin:
    """No thermostat memory, so the rebinding is target plus velocities."""

    def test_rebinds_the_per_graph_target(self, device: str) -> None:
        batch = _make_batch(device=device)
        dynamics = _langevin(device)
        dynamics._ensure_state_initialized(batch)
        before = dynamics._state.temperature.reshape(-1).clone()

        dynamics.apply_per_system_params(
            {"temperature": torch.tensor([600.0, 300.0], device=device)}, batch
        )
        after = dynamics._state.temperature.reshape(-1)
        assert abs(float(after[0] / before[0]) - 2.0) < 1e-5
        assert abs(float(after[1] / before[1]) - 1.0) < 1e-5

    def test_velocities_move_in_the_same_call(self, device: str) -> None:
        """Indivisible: a caller cannot forget the second half, because there
        is no second half. A walker whose target moved while its velocities
        did not samples the wrong ensemble with no symptom."""
        batch = _make_batch(n_graphs=1, atoms_per_graph=3, device=device)
        batch.velocities.fill_(1.0)
        dynamics = _langevin(device)
        dynamics._ensure_state_initialized(batch)

        dynamics.apply_per_system_params(
            {"temperature": torch.tensor([1200.0], device=device)}, batch
        )
        # T: 300 -> 1200, so v scales by sqrt(4) = 2.
        assert torch.allclose(
            batch.velocities, torch.full_like(batch.velocities, 2.0), atol=1e-5
        )

    def test_an_unchanged_target_leaves_velocities_alone(self, device: str) -> None:
        batch = _make_batch(n_graphs=1, atoms_per_graph=3, device=device)
        batch.velocities.fill_(1.0)
        dynamics = _langevin(device)
        dynamics._ensure_state_initialized(batch)

        dynamics.apply_per_system_params(
            {"temperature": torch.tensor([300.0], device=device)}, batch
        )
        assert torch.allclose(
            batch.velocities, torch.ones_like(batch.velocities), atol=1e-6
        )


class TestNoseHoover:
    """The chain carries memory, so three more quantities travel with kT."""

    def test_chain_state_transforms_with_kt(self, device: str) -> None:
        """Q and eta_dot must move with kT or detailed balance breaks."""
        batch = _make_batch(device=device)
        dynamics = _nose_hoover(device)
        dynamics._ensure_state_initialized(batch)
        with torch.no_grad():
            dynamics._state.nhc_eta_dot.fill_(2.0)
        q_before = dynamics._state.nhc_Q.clone()
        eta_dot_before = dynamics._state.nhc_eta_dot.clone()

        dynamics.apply_per_system_params(
            {"temperature": torch.tensor([1200.0, 1200.0], device=device)}, batch
        )
        ratio = 4.0  # 300 -> 1200
        assert torch.allclose(dynamics._state.nhc_Q, q_before * ratio, rtol=1e-5), (
            "chain masses must scale with kT"
        )
        assert torch.allclose(
            dynamics._state.nhc_eta_dot,
            eta_dot_before / math.sqrt(ratio),
            rtol=1e-5,
        ), "chain velocities must scale as 1/sqrt(kT)"

    def test_chain_kinetic_energy_is_invariant(self, device: str) -> None:
        """Q eta_dot^2 must not change: rebinding injects no thermostat energy."""
        batch = _make_batch(device=device)
        dynamics = _nose_hoover(device)
        dynamics._ensure_state_initialized(batch)
        with torch.no_grad():
            dynamics._state.nhc_eta_dot.fill_(1.5)
        before = (dynamics._state.nhc_Q * dynamics._state.nhc_eta_dot**2).sum()

        dynamics.apply_per_system_params(
            {"temperature": torch.tensor([900.0, 900.0], device=device)}, batch
        )
        after = (dynamics._state.nhc_Q * dynamics._state.nhc_eta_dot**2).sum()
        assert torch.allclose(before, after, rtol=1e-5)

    def test_velocities_move_too(self, device: str) -> None:
        batch = _make_batch(n_graphs=1, atoms_per_graph=3, device=device)
        batch.velocities.fill_(1.0)
        dynamics = _nose_hoover(device)
        dynamics._ensure_state_initialized(batch)

        dynamics.apply_per_system_params(
            {"temperature": torch.tensor([1200.0], device=device)}, batch
        )
        assert torch.allclose(
            batch.velocities, torch.full_like(batch.velocities, 2.0), atol=1e-5
        )
