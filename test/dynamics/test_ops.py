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
"""
Comprehensive unit tests for all nvalchemi.dynamics._ops bindings.

Tests verify:
- Correct output shapes
- Correct dtypes (float32 and float64)
- In-place mutation of the expected tensors
- Single-system (M=1) and multi-system (M>1) batch cases
- Correct batch_idx routing for multi-system batches
"""

from __future__ import annotations

import pytest
import torch

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _batch_idx(sizes: list[int], device: torch.device) -> torch.Tensor:
    """Build a non-decreasing per-atom batch index from per-system atom counts."""
    return torch.repeat_interleave(
        torch.arange(len(sizes), device=device),
        torch.tensor(sizes, device=device),
    ).to(torch.int32)


@pytest.fixture(params=[torch.float32, torch.float64])
def dtype(request):
    return request.param


@pytest.fixture(params=["cpu"])
def device(request):
    return torch.device(request.param)


# ---------------------------------------------------------------------------
# Velocity Verlet
# ---------------------------------------------------------------------------


class TestVelocityVerlet:
    def _make(self, M: int, N: int, dtype, device):
        """Return (positions, velocities, forces, masses, dt, batch_idx)."""
        torch.manual_seed(0)
        positions = torch.randn(N, 3, dtype=dtype, device=device)
        velocities = torch.randn(N, 3, dtype=dtype, device=device)
        forces = torch.randn(N, 3, dtype=dtype, device=device)
        masses = torch.ones(N, dtype=dtype, device=device)
        dt = torch.full((M,), 0.1, dtype=dtype, device=device)
        sizes = [N // M] * M
        sizes[-1] += N - sum(sizes)
        batch = _batch_idx(sizes, device)
        return positions, velocities, forces, masses, dt, batch

    def test_position_update_shape(self, dtype, device):
        from nvalchemi.dynamics._ops.velocity_verlet import vv_position_update

        pos, vel, frc, mass, dt, batch = self._make(2, 8, dtype, device)
        pos_orig = pos.clone()
        vel_orig = vel.clone()
        vv_position_update(pos, vel, frc, mass, dt, batch)
        assert pos.shape == pos_orig.shape
        assert vel.shape == vel_orig.shape
        assert pos.dtype == dtype
        # positions must change
        assert not torch.allclose(pos, pos_orig)

    def test_position_update_single_system(self, dtype, device):
        from nvalchemi.dynamics._ops.velocity_verlet import vv_position_update

        pos, vel, frc, mass, dt, _ = self._make(1, 4, dtype, device)
        batch = torch.zeros(4, dtype=torch.int32, device=device)
        pos_orig = pos.clone()
        vv_position_update(pos, vel, frc, mass, dt, batch)
        assert not torch.allclose(pos, pos_orig)

    def test_velocity_finalize_shape(self, dtype, device):
        from nvalchemi.dynamics._ops.velocity_verlet import vv_velocity_finalize

        _, vel, frc, mass, dt, batch = self._make(2, 8, dtype, device)
        vel_orig = vel.clone()
        vv_velocity_finalize(vel, frc, mass, dt, batch)
        assert vel.shape == vel_orig.shape
        assert vel.dtype == dtype
        assert not torch.allclose(vel, vel_orig)

    def test_velocity_finalize_multi_system(self, dtype, device):
        from nvalchemi.dynamics._ops.velocity_verlet import vv_velocity_finalize

        _, vel, frc, mass, dt, batch = self._make(3, 9, dtype, device)
        vel_orig = vel.clone()
        vv_velocity_finalize(vel, frc, mass, dt, batch)
        assert not torch.allclose(vel, vel_orig)


# ---------------------------------------------------------------------------
# Langevin
# ---------------------------------------------------------------------------


class TestLangevin:
    def _make(self, M: int, N: int, dtype, device):
        torch.manual_seed(1)
        positions = torch.randn(N, 3, dtype=dtype, device=device)
        velocities = torch.randn(N, 3, dtype=dtype, device=device)
        forces = torch.randn(N, 3, dtype=dtype, device=device)
        masses = torch.ones(N, dtype=dtype, device=device)
        dt = torch.full((M,), 0.5, dtype=dtype, device=device)
        temperature = torch.full((M,), 300.0, dtype=dtype, device=device)
        friction = torch.full((M,), 0.1, dtype=dtype, device=device)
        sizes = [N // M] * M
        sizes[-1] += N - sum(sizes)
        batch = _batch_idx(sizes, device)
        return positions, velocities, forces, masses, dt, temperature, friction, batch

    def test_half_step_mutates_positions_and_velocities(self, dtype, device):
        from nvalchemi.dynamics._ops.langevin import langevin_half_step

        pos, vel, frc, mass, dt, temp, fric, batch = self._make(2, 8, dtype, device)
        pos_orig = pos.clone()
        vel_orig = vel.clone()
        langevin_half_step(pos, vel, frc, mass, dt, temp, fric, 42, batch)
        assert pos.shape == pos_orig.shape
        assert vel.shape == vel_orig.shape
        assert not torch.allclose(pos, pos_orig)
        assert not torch.allclose(vel, vel_orig)

    def test_half_step_single_system(self, dtype, device):
        from nvalchemi.dynamics._ops.langevin import langevin_half_step

        pos, vel, frc, mass, dt, temp, fric, _ = self._make(1, 5, dtype, device)
        batch = torch.zeros(5, dtype=torch.int32, device=device)
        pos_orig = pos.clone()
        langevin_half_step(pos, vel, frc, mass, dt, temp, fric, 123, batch)
        assert not torch.allclose(pos, pos_orig)

    def test_half_step_seed_reproducibility(self, dtype, device):
        from nvalchemi.dynamics._ops.langevin import langevin_half_step

        pos1, vel1, frc, mass, dt, temp, fric, batch = self._make(1, 4, dtype, device)
        pos2, vel2 = pos1.clone(), vel1.clone()
        langevin_half_step(pos1, vel1, frc, mass, dt, temp, fric, 42, batch)
        langevin_half_step(pos2, vel2, frc, mass, dt, temp, fric, 42, batch)
        assert torch.allclose(pos1, pos2)
        assert torch.allclose(vel1, vel2)

    def test_finalize_mutates_velocities_only(self, dtype, device):
        from nvalchemi.dynamics._ops.langevin import langevin_finalize

        _, vel, frc, mass, dt, _, _, batch = self._make(2, 8, dtype, device)
        vel_orig = vel.clone()
        langevin_finalize(vel, frc, mass, dt, batch)
        assert vel.shape == vel_orig.shape
        assert vel.dtype == dtype
        assert not torch.allclose(vel, vel_orig)

    def test_finalize_signature(self, dtype, device):
        """langevin_finalize takes (velocities, forces_new, masses, dt, batch_idx)."""
        import inspect

        from nvalchemi.dynamics._ops.langevin import langevin_finalize

        sig = inspect.signature(langevin_finalize)
        params = list(sig.parameters.keys())
        assert "temperature" not in params
        assert "friction" not in params
        assert "random_seed" not in params
        assert "rng_seed" not in params


# ---------------------------------------------------------------------------
# Thermostat utilities
# ---------------------------------------------------------------------------


class TestThermostatUtils:
    def _make(self, M: int, N: int, dtype, device):
        torch.manual_seed(2)
        velocities = torch.randn(N, 3, dtype=dtype, device=device)
        masses = torch.ones(N, dtype=dtype, device=device)
        temperature = torch.full((M,), 300.0, dtype=dtype, device=device)
        sizes = [N // M] * M
        sizes[-1] += N - sum(sizes)
        batch = _batch_idx(sizes, device)
        return velocities, masses, temperature, batch

    def test_initialize_velocities_shape(self, dtype, device):
        from nvalchemi.dynamics._ops.thermostat_utils import initialize_velocities

        M, N = 2, 8
        vel, mass, temp, batch = self._make(M, N, dtype, device)
        initialize_velocities(vel, mass, temp, batch, random_seed=0, remove_com=True)
        assert vel.shape == (N, 3)
        assert vel.dtype == dtype

    def test_initialize_velocities_rescale(self, dtype, device):
        """After COM removal + rescale, temperature should match target exactly."""
        from nvalchemi.dynamics._ops.thermostat_utils import (
            compute_kinetic_energy,
            initialize_velocities,
        )
        from nvalchemi.dynamics.hooks._utils import KB_EV

        M, N = 1, 200  # large enough for good statistics
        vel = torch.zeros(N, 3, dtype=dtype, device=device)
        mass = torch.full((N,), 28.0, dtype=dtype, device=device)  # silicon
        temp = torch.full((M,), 300.0, dtype=dtype, device=device)
        batch = torch.zeros(N, dtype=torch.int32, device=device)

        initialize_velocities(
            vel, mass, temp, batch, random_seed=42, remove_com=True, rescale=True
        )
        ke = compute_kinetic_energy(vel, mass, batch, M)
        t_actual = float(2.0 * ke[0] / (3.0 * N * KB_EV))
        assert abs(t_actual - 300.0) < 1.0, f"T={t_actual}, expected ~300"

    def test_initialize_velocities_no_rescale_drifts(self, dtype, device):
        """Without rescale, COM removal causes temperature to drift below target."""
        from nvalchemi.dynamics._ops.thermostat_utils import (
            compute_kinetic_energy,
            initialize_velocities,
        )
        from nvalchemi.dynamics.hooks._utils import KB_EV

        M, N = 1, 200
        vel = torch.zeros(N, 3, dtype=dtype, device=device)
        mass = torch.full((N,), 28.0, dtype=dtype, device=device)
        temp = torch.full((M,), 300.0, dtype=dtype, device=device)
        batch = torch.zeros(N, dtype=torch.int32, device=device)

        initialize_velocities(
            vel, mass, temp, batch, random_seed=42, remove_com=True, rescale=False
        )
        ke = compute_kinetic_energy(vel, mass, batch, M)
        t_actual = float(2.0 * ke[0] / (3.0 * N * KB_EV))
        # Without rescale, T should be slightly below target (lost 3 DOF).
        assert t_actual < 300.0, f"Expected T < 300 without rescale, got {t_actual}"

    def test_initialize_velocities_remove_rotations(self, dtype, device):
        """After rotation removal, angular momentum should be near zero."""
        from nvalchemi.dynamics._ops.thermostat_utils import initialize_velocities

        M, N = 1, 100
        torch.manual_seed(7)
        pos = torch.randn(N, 3, dtype=dtype, device=device)
        vel = torch.zeros(N, 3, dtype=dtype, device=device)
        mass = torch.full((N,), 28.0, dtype=dtype, device=device)
        temp = torch.full((M,), 300.0, dtype=dtype, device=device)
        batch = torch.zeros(N, dtype=torch.int32, device=device)

        initialize_velocities(
            vel,
            mass,
            temp,
            batch,
            random_seed=42,
            remove_com=True,
            remove_rotations=True,
            positions=pos,
        )

        # Angular momentum L = sum(m * r x v) should be ~0.
        m = mass.unsqueeze(-1)
        L = (m * torch.linalg.cross(pos, vel)).sum(dim=0)
        assert L.abs().max() < 1e-3, f"Angular momentum not zero: {L}"

    def test_initialize_velocities_rotation_removal_rescales(self, dtype, device):
        """Rotation removal + rescale should still match target temperature."""
        from nvalchemi.dynamics._ops.thermostat_utils import (
            compute_kinetic_energy,
            initialize_velocities,
        )
        from nvalchemi.dynamics.hooks._utils import KB_EV

        M, N = 1, 200
        torch.manual_seed(7)
        pos = torch.randn(N, 3, dtype=dtype, device=device)
        vel = torch.zeros(N, 3, dtype=dtype, device=device)
        mass = torch.full((N,), 28.0, dtype=dtype, device=device)
        temp = torch.full((M,), 300.0, dtype=dtype, device=device)
        batch = torch.zeros(N, dtype=torch.int32, device=device)

        initialize_velocities(
            vel,
            mass,
            temp,
            batch,
            random_seed=42,
            remove_com=True,
            remove_rotations=True,
            rescale=True,
            positions=pos,
        )
        ke = compute_kinetic_energy(vel, mass, batch, M)
        t_actual = float(2.0 * ke[0] / (3.0 * N * KB_EV))
        assert abs(t_actual - 300.0) < 1.0, f"T={t_actual}, expected ~300"

    def test_initialize_velocities_multi_system(self, dtype, device):
        """Rescale works per-system with different target temperatures."""
        from nvalchemi.dynamics._ops.thermostat_utils import (
            compute_kinetic_energy,
            initialize_velocities,
        )
        from nvalchemi.dynamics.hooks._utils import KB_EV

        N_per = 100
        M = 3
        N = M * N_per
        vel = torch.zeros(N, 3, dtype=dtype, device=device)
        mass = torch.full((N,), 28.0, dtype=dtype, device=device)
        temp = torch.tensor([100.0, 300.0, 500.0], dtype=dtype, device=device)
        batch = _batch_idx([N_per] * M, device)

        initialize_velocities(
            vel, mass, temp, batch, random_seed=42, remove_com=True, rescale=True
        )
        ke = compute_kinetic_energy(vel, mass, batch, M)
        for i in range(M):
            t_actual = float(2.0 * ke[i] / (3.0 * N_per * KB_EV))
            assert abs(t_actual - float(temp[i])) < 2.0, (
                f"System {i}: T={t_actual}, expected {float(temp[i])}"
            )

    def test_compute_kinetic_energy_shape(self, dtype, device):
        from nvalchemi.dynamics._ops.thermostat_utils import compute_kinetic_energy

        M, N = 3, 9
        vel, mass, _, batch = self._make(M, N, dtype, device)
        ke = compute_kinetic_energy(vel, mass, batch, M)
        assert ke.shape == (M,)
        assert ke.dtype == dtype
        assert (ke >= 0).all()

    def test_compute_kinetic_energy_single(self, dtype, device):
        from nvalchemi.dynamics._ops.thermostat_utils import compute_kinetic_energy

        vel = torch.tensor([[1.0, 0.0, 0.0]], dtype=dtype, device=device)
        mass = torch.tensor([2.0], dtype=dtype, device=device)
        batch = torch.zeros(1, dtype=torch.int32, device=device)
        ke = compute_kinetic_energy(vel, mass, batch, 1)
        expected = 0.5 * 2.0 * 1.0
        assert torch.allclose(
            ke, torch.tensor([expected], dtype=dtype, device=device), rtol=1e-5
        )

    def test_compute_temperature_shape(self, dtype, device):
        from nvalchemi.dynamics._ops.thermostat_utils import (
            compute_kinetic_energy,
            compute_temperature,
        )

        M, N = 2, 6
        vel, mass, _, batch = self._make(M, N, dtype, device)
        ke = compute_kinetic_energy(vel, mass, batch, M)
        atoms_per = torch.full((M,), N // M, dtype=torch.int32, device=device)
        temp = compute_temperature(ke, atoms_per)
        assert temp.shape == (M,)
        assert temp.dtype == dtype

    def test_remove_com_motion(self, dtype, device):
        from nvalchemi.dynamics._ops.thermostat_utils import remove_com_motion

        N = 6
        vel = torch.randn(N, 3, dtype=dtype, device=device)
        mass = torch.ones(N, dtype=dtype, device=device)
        batch = torch.zeros(N, dtype=torch.int32, device=device)
        remove_com_motion(vel, mass, batch, num_systems=1)
        # After COM removal, center-of-mass velocity should be near zero.
        com = vel.mean(dim=0)
        assert torch.allclose(com, torch.zeros(3, dtype=dtype), atol=1e-5)

    def test_velocity_rescale(self, dtype, device):
        from nvalchemi.dynamics._ops.thermostat_utils import velocity_rescale

        N = 4
        vel = torch.ones(N, 3, dtype=dtype, device=device)
        scale = torch.tensor([2.0], dtype=dtype, device=device)
        batch = torch.zeros(N, dtype=torch.int32, device=device)
        velocity_rescale(vel, scale, batch)
        assert torch.allclose(vel, torch.full((N, 3), 2.0, dtype=dtype, device=device))


# ---------------------------------------------------------------------------
# Nosé-Hoover chain
# ---------------------------------------------------------------------------


class TestNoseHoover:
    def _make_nhc_state(self, M: int, N: int, C: int, dtype, device):
        torch.manual_seed(3)
        velocities = torch.randn(N, 3, dtype=dtype, device=device)
        masses = torch.ones(N, dtype=dtype, device=device)
        eta = torch.zeros(M, C, dtype=dtype, device=device)
        eta_dot = torch.zeros(M, C, dtype=dtype, device=device)
        temperature = torch.full((M,), 300.0, dtype=dtype, device=device)
        tau = torch.full((M,), 1.0, dtype=dtype, device=device)
        dt = torch.full((M,), 0.1, dtype=dtype, device=device)
        sizes = [N // M] * M
        sizes[-1] += N - sum(sizes)
        batch = _batch_idx(sizes, device)
        return velocities, masses, eta, eta_dot, temperature, tau, dt, batch

    def test_compute_masses_shape(self, dtype, device):
        from nvalchemi.dynamics._ops.nose_hoover import nhc_compute_masses

        M, N, C = 2, 8, 3
        vel, mass, eta, eta_dot, temp, tau, dt, batch = self._make_nhc_state(
            M, N, C, dtype, device
        )
        Q = nhc_compute_masses(temp, tau, mass, batch, C)
        assert Q.shape == (M, C)

    def test_compute_masses_positive(self, dtype, device):
        from nvalchemi.dynamics._ops.nose_hoover import nhc_compute_masses

        M, N, C = 1, 4, 3
        vel, mass, eta, eta_dot, temp, tau, dt, batch = self._make_nhc_state(
            M, N, C, dtype, device
        )
        Q = nhc_compute_masses(temp, tau, mass, batch, C)
        assert (Q > 0).all()

    def test_chain_update_mutates_velocities(self, dtype, device):
        from nvalchemi.dynamics._ops.nose_hoover import (
            nhc_chain_update,
            nhc_compute_masses,
        )

        M, N, C = 2, 8, 3
        vel, mass, eta, eta_dot, temp, tau, dt, batch = self._make_nhc_state(
            M, N, C, dtype, device
        )
        Q = nhc_compute_masses(temp, tau, mass, batch, C)
        ndof = torch.full((M,), (N // M) * 3, dtype=torch.int32, device=device)
        ke2 = torch.zeros(M, dtype=dtype, device=device)
        total_scale = torch.zeros(M, dtype=dtype, device=device)
        step_scale = torch.zeros(M, dtype=dtype, device=device)
        dt_chain = torch.zeros(M, dtype=dtype, device=device)
        vel_orig = vel.clone()
        nhc_chain_update(
            vel,
            mass,
            eta,
            eta_dot,
            Q,
            temp,
            dt,
            ndof,
            ke2,
            total_scale,
            step_scale,
            dt_chain,
            batch,
        )
        assert vel.shape == vel_orig.shape
        # Velocities should be scaled (modified) by thermostat.
        assert not torch.allclose(vel, vel_orig)

    def test_velocity_half_step_shape(self, dtype, device):
        from nvalchemi.dynamics._ops.nose_hoover import nhc_velocity_half_step

        M, N = 2, 6
        torch.manual_seed(4)
        vel = torch.randn(N, 3, dtype=dtype, device=device)
        frc = torch.randn(N, 3, dtype=dtype, device=device)
        mass = torch.ones(N, dtype=dtype, device=device)
        dt = torch.full((M,), 0.1, dtype=dtype, device=device)
        sizes = [N // M] * M
        batch = _batch_idx(sizes, device)
        vel_orig = vel.clone()
        nhc_velocity_half_step(vel, frc, mass, dt, batch)
        assert vel.shape == vel_orig.shape
        assert not torch.allclose(vel, vel_orig)

    def test_position_update_shape(self, dtype, device):
        from nvalchemi.dynamics._ops.nose_hoover import nhc_position_update

        M, N = 2, 6
        torch.manual_seed(5)
        pos = torch.randn(N, 3, dtype=dtype, device=device)
        vel = torch.randn(N, 3, dtype=dtype, device=device)
        dt = torch.full((M,), 0.1, dtype=dtype, device=device)
        sizes = [N // M] * M
        batch = _batch_idx(sizes, device)
        pos_orig = pos.clone()
        nhc_position_update(pos, vel, dt, batch)
        assert pos.shape == pos_orig.shape
        assert not torch.allclose(pos, pos_orig)


# ---------------------------------------------------------------------------
# FIRE ops
# ---------------------------------------------------------------------------


class TestFireOps:
    def _make_fire_state(self, M: int, N: int, dtype, device):
        torch.manual_seed(6)
        positions = torch.randn(N, 3, dtype=dtype, device=device)
        velocities = torch.zeros(N, 3, dtype=dtype, device=device)
        forces = torch.randn(N, 3, dtype=dtype, device=device)
        masses = torch.ones(N, dtype=dtype, device=device)
        alpha = torch.full((M,), 0.1, dtype=dtype, device=device)
        dt = torch.full((M,), 0.1, dtype=dtype, device=device)
        n_steps_pos = torch.zeros(M, dtype=torch.int32, device=device)
        alpha_start = torch.full((M,), 0.1, dtype=dtype, device=device)
        f_alpha = torch.full((M,), 0.99, dtype=dtype, device=device)
        dt_min = torch.full((M,), 0.002, dtype=dtype, device=device)
        dt_max = torch.full((M,), 1.0, dtype=dtype, device=device)
        maxstep = torch.full((M,), 0.2, dtype=dtype, device=device)
        n_min = torch.full((M,), 5, dtype=torch.int32, device=device)
        f_dec = torch.full((M,), 0.5, dtype=dtype, device=device)
        f_inc = torch.full((M,), 1.1, dtype=dtype, device=device)
        uphill_flag = torch.zeros(M, dtype=torch.int32, device=device)
        sizes = [N // M] * M
        sizes[-1] += N - sum(sizes)
        batch = _batch_idx(sizes, device)
        return (
            positions,
            velocities,
            forces,
            masses,
            alpha,
            dt,
            n_steps_pos,
            alpha_start,
            f_alpha,
            dt_min,
            dt_max,
            maxstep,
            n_min,
            f_dec,
            f_inc,
            uphill_flag,
            batch,
        )

    def test_fire_step_mutates_positions(self, dtype, device):
        from nvalchemi.dynamics._ops.fire import fire_step

        M, N = 2, 8
        (
            pos,
            vel,
            frc,
            mass,
            alpha,
            dt,
            n_pos,
            a_start,
            f_alp,
            dt_min,
            dt_max,
            maxstep,
            n_min,
            f_dec,
            f_inc,
            uphll,
            batch,
        ) = self._make_fire_state(M, N, dtype, device)
        pos_orig = pos.clone()
        fire_step(
            pos,
            vel,
            frc,
            mass,
            alpha,
            dt,
            n_pos,
            a_start,
            f_alp,
            dt_min,
            dt_max,
            maxstep,
            n_min,
            f_dec,
            f_inc,
            uphll,
            batch_idx=batch,
        )
        assert pos.shape == pos_orig.shape
        assert pos.dtype == dtype
        assert not torch.allclose(pos, pos_orig)

    def test_fire_step_single_system(self, dtype, device):
        from nvalchemi.dynamics._ops.fire import fire_step

        M, N = 1, 5
        (
            pos,
            vel,
            frc,
            mass,
            alpha,
            dt,
            n_pos,
            a_start,
            f_alp,
            dt_min,
            dt_max,
            maxstep,
            n_min,
            f_dec,
            f_inc,
            uphll,
            batch,
        ) = self._make_fire_state(M, N, dtype, device)
        pos_orig = pos.clone()
        fire_step(
            pos,
            vel,
            frc,
            mass,
            alpha,
            dt,
            n_pos,
            a_start,
            f_alp,
            dt_min,
            dt_max,
            maxstep,
            n_min,
            f_dec,
            f_inc,
            uphll,
            batch_idx=batch,
        )
        assert not torch.allclose(pos, pos_orig)

    def test_fire_step_uphill_flag_shape(self, dtype, device):
        """uphill_flag must be [M] int32."""
        from nvalchemi.dynamics._ops.fire import fire_step

        M, N = 2, 6
        (
            pos,
            vel,
            frc,
            mass,
            alpha,
            dt,
            n_pos,
            a_start,
            f_alp,
            dt_min,
            dt_max,
            maxstep,
            n_min,
            f_dec,
            f_inc,
            uphll,
            batch,
        ) = self._make_fire_state(M, N, dtype, device)
        assert uphll.shape == (M,)
        assert uphll.dtype == torch.int32
        pos_orig = pos.clone()
        fire_step(
            pos,
            vel,
            frc,
            mass,
            alpha,
            dt,
            n_pos,
            a_start,
            f_alp,
            dt_min,
            dt_max,
            maxstep,
            n_min,
            f_dec,
            f_inc,
            uphll,
            batch_idx=batch,
        )
        assert not torch.allclose(pos, pos_orig)

    def test_fire_update_mutates_velocities(self, dtype, device):
        from nvalchemi.dynamics._ops.fire import fire_update

        M, N = 2, 8
        (
            _,
            vel,
            frc,
            _,
            alpha,
            dt,
            n_pos,
            a_start,
            f_alp,
            dt_min,
            dt_max,
            _,
            n_min,
            f_dec,
            f_inc,
            _,
            batch,
        ) = self._make_fire_state(M, N, dtype, device)
        # Give velocities some initial value to make mixing non-trivial.
        vel.uniform_(0.01, 0.1)
        vel_orig = vel.clone()
        fire_update(
            vel,
            frc,
            alpha,
            dt,
            n_pos,
            a_start,
            f_alp,
            dt_min,
            dt_max,
            n_min,
            f_dec,
            f_inc,
            batch_idx=batch,
        )
        assert vel.shape == vel_orig.shape
        assert not torch.allclose(vel, vel_orig)

    def test_fire_step_all_hyperparams_are_tensors(self, dtype, device):
        """fire_step must accept [M] tensors for all hyperparams (not scalars)."""
        M, N = 2, 6
        (
            pos,
            vel,
            frc,
            mass,
            alpha,
            dt,
            n_pos,
            a_start,
            f_alp,
            dt_min,
            dt_max,
            maxstep,
            n_min,
            f_dec,
            f_inc,
            uphll,
            batch,
        ) = self._make_fire_state(M, N, dtype, device)
        # These must all be tensors - passing a scalar should not be accepted
        # at the _fire_step_op level (but the public fire_step itself takes tensors).
        assert a_start.shape == (M,)
        assert f_alp.shape == (M,)
        assert maxstep.shape == (M,)
        assert n_min.shape == (M,)
        assert n_min.dtype == torch.int32
        assert f_dec.shape == (M,)
        assert f_inc.shape == (M,)


# ---------------------------------------------------------------------------
# FIRE2 ops (pass-through to nvalchemiops.torch.fire2)
# ---------------------------------------------------------------------------


class TestFire2Ops:
    def _make(self, M: int, N: int, dtype, device):
        torch.manual_seed(7)
        positions = torch.randn(N, 3, dtype=dtype, device=device)
        velocities = torch.zeros(N, 3, dtype=dtype, device=device)
        forces = torch.randn(N, 3, dtype=dtype, device=device)
        alpha = torch.full((M,), 0.09, dtype=dtype, device=device)
        dt = torch.full((M,), 0.05, dtype=dtype, device=device)
        nsteps_inc = torch.zeros(M, dtype=torch.int32, device=device)
        sizes = [N // M] * M
        sizes[-1] += N - sum(sizes)
        batch = _batch_idx(sizes, device)
        return positions, velocities, forces, alpha, dt, nsteps_inc, batch

    def test_fire2_coord_mutates_positions(self, dtype, device):
        from nvalchemi.dynamics._ops.fire import fire2_step_coord

        M, N = 2, 8
        pos, vel, frc, alpha, dt, nsteps_inc, batch = self._make(M, N, dtype, device)
        pos_orig = pos.clone()
        fire2_step_coord(pos, vel, frc, batch, alpha, dt, nsteps_inc)
        assert pos.shape == pos_orig.shape
        assert not torch.allclose(pos, pos_orig)

    def test_fire2_coord_single_system(self, dtype, device):
        from nvalchemi.dynamics._ops.fire import fire2_step_coord

        M, N = 1, 4
        pos, vel, frc, alpha, dt, nsteps_inc, batch = self._make(M, N, dtype, device)
        pos_orig = pos.clone()
        fire2_step_coord(pos, vel, frc, batch, alpha, dt, nsteps_inc)
        assert not torch.allclose(pos, pos_orig)

    def test_fire2_coord_scratch_buffers(self, dtype, device):
        from nvalchemi.dynamics._ops.fire import fire2_step_coord

        M, N = 2, 6
        pos, vel, frc, alpha, dt, nsteps_inc, batch = self._make(M, N, dtype, device)
        vf = torch.zeros(M, dtype=dtype, device=device)
        v_sumsq = torch.zeros(M, dtype=dtype, device=device)
        f_sumsq = torch.zeros(M, dtype=dtype, device=device)
        pos_orig = pos.clone()
        fire2_step_coord(
            pos,
            vel,
            frc,
            batch,
            alpha,
            dt,
            nsteps_inc,
            vf=vf,
            v_sumsq=v_sumsq,
            f_sumsq=f_sumsq,
        )
        assert not torch.allclose(pos, pos_orig)


# ---------------------------------------------------------------------------
# NPT/NPH ops - pressure and barostat utilities
# ---------------------------------------------------------------------------


class TestNptNphOps:
    def _make_pressure(self, M: int, N: int, dtype, device):
        torch.manual_seed(8)
        velocities = torch.randn(N, 3, dtype=dtype, device=device)
        masses = torch.ones(N, dtype=dtype, device=device)
        # Random symmetric virial tensors.
        S = torch.randn(M, 3, 3, dtype=dtype, device=device)
        virial = 0.5 * (S + S.transpose(-1, -2))
        # Identity cells.
        cell = (
            torch.eye(3, dtype=dtype, device=device)
            .unsqueeze(0)
            .expand(M, -1, -1)
            .contiguous()
        )
        kinetic_tensors = torch.zeros(M, 9, dtype=dtype, device=device)  # [M,9] array2d
        pressure_tensors = torch.zeros(M, 9, dtype=dtype, device=device)  # [M,9] vec9
        volumes = torch.full((M,), 2.0, dtype=dtype, device=device)
        sizes = [N // M] * M
        sizes[-1] += N - sum(sizes)
        batch = _batch_idx(sizes, device)
        return (
            velocities,
            masses,
            virial,
            cell,
            kinetic_tensors,
            pressure_tensors,
            volumes,
            batch,
        )

    def test_compute_pressure_tensor_shape(self, dtype, device):
        from nvalchemi.dynamics._ops.npt_nph import compute_pressure_tensor

        M, N = 2, 8
        vel, mass, virial, cell, kin, P_scr, vol, batch = self._make_pressure(
            M, N, dtype, device
        )
        P = compute_pressure_tensor(vel, mass, virial, cell, kin, P_scr, vol, batch)
        assert P.shape == (M, 9)
        assert P.dtype == dtype

    def test_compute_pressure_tensor_single(self, dtype, device):
        from nvalchemi.dynamics._ops.npt_nph import compute_pressure_tensor

        M, N = 1, 4
        vel, mass, virial, cell, kin, P_scr, vol, batch = self._make_pressure(
            M, N, dtype, device
        )
        P = compute_pressure_tensor(vel, mass, virial, cell, kin, P_scr, vol, batch)
        assert P.shape == (1, 9)

    def test_compute_scalar_pressure(self, dtype, device):
        from nvalchemi.dynamics._ops.npt_nph import compute_scalar_pressure

        M = 3
        # vec9 layout: [xx,xy,xz,yx,yy,yz,zx,zy,zz]; diagonals at indices 0,4,8.
        P_mat = (
            torch.eye(3, dtype=dtype, device=device)
            .unsqueeze(0)
            .expand(M, -1, -1)
            .contiguous()
            * 2.0
        )
        P_tensor = P_mat.reshape(M, 9)  # [M, 9] vec9
        scalar_P = torch.zeros(M, dtype=dtype, device=device)
        compute_scalar_pressure(P_tensor, scalar_P)
        assert scalar_P.shape == (M,)
        # Tr(2*I)/3 = 2
        assert torch.allclose(
            scalar_P, torch.full((M,), 2.0, dtype=dtype, device=device), atol=1e-5
        )

    def test_compute_barostat_mass_mutates_output(self, dtype, device):
        from nvalchemi.dynamics._ops.npt_nph import compute_barostat_mass

        M = 2
        temperature = torch.full((M,), 300.0, dtype=dtype, device=device)
        tau_p = torch.full((M,), 1.0, dtype=dtype, device=device)
        num_atoms = torch.full((M,), 4, dtype=torch.int32, device=device)
        W = torch.zeros(M, dtype=dtype, device=device)
        compute_barostat_mass(temperature, tau_p, num_atoms, W)
        assert W.shape == (M,)
        assert (W > 0).all()

    def test_npt_cell_update_shape(self, dtype, device):
        from nvalchemi.dynamics._ops.npt_nph import npt_cell_update

        M = 2
        cell = (
            torch.eye(3, dtype=dtype, device=device)
            .unsqueeze(0)
            .expand(M, -1, -1)
            .contiguous()
        )
        cell_vel = torch.randn(M, 3, 3, dtype=dtype, device=device) * 0.01
        dt = torch.full((M,), 0.1, dtype=dtype, device=device)
        cell_orig = cell.clone()
        npt_cell_update(cell, cell_vel, dt)
        assert cell.shape == cell_orig.shape
        assert not torch.allclose(cell, cell_orig)

    def test_npt_cell_update_no_batch_idx(self, dtype, device):
        """npt_cell_update should NOT require batch_idx."""
        import inspect

        from nvalchemi.dynamics._ops.npt_nph import npt_cell_update

        sig = inspect.signature(npt_cell_update)
        params = list(sig.parameters.keys())
        assert "batch_idx" not in params

    def test_npt_position_update_requires_cells_inv(self, dtype, device):
        """npt_position_update must accept cells_inv (custom_op, so test by calling)."""
        from nvalchemi.dynamics._ops.npt_nph import npt_position_update

        M, N = 1, 4
        positions = torch.randn(N, 3, dtype=dtype, device=device)
        velocities = torch.randn(N, 3, dtype=dtype, device=device)
        cell = torch.eye(3, dtype=dtype, device=device).unsqueeze(0)
        cell_vel = torch.zeros(M, 3, 3, dtype=dtype, device=device)
        dt = torch.full((M,), 0.1, dtype=dtype, device=device)
        cells_inv = torch.linalg.inv(cell)
        batch_idx = torch.zeros(N, dtype=torch.int32, device=device)
        pos_orig = positions.clone()
        # Should not raise; positions should be updated.
        npt_position_update(
            positions, velocities, cell, cell_vel, dt, cells_inv, batch_idx
        )
        assert positions.shape == pos_orig.shape

    def test_compute_pressure_tensor_virial_sign_convention(self, dtype, device):
        """Bridge passes virial directly to the ops kernel.

        With zero velocities (no kinetic contribution), the pressure
        tensor should be P = virial / V.  A positive (compressive) virial
        should yield positive pressure.
        """
        from nvalchemi.dynamics._ops.npt_nph import compute_pressure_tensor

        M, N = 1, 4
        vel = torch.zeros(N, 3, dtype=dtype, device=device)
        mass = torch.ones(N, dtype=dtype, device=device)
        virial = torch.eye(3, dtype=dtype, device=device).unsqueeze(0) * 3.0
        cell = torch.eye(3, dtype=dtype, device=device).unsqueeze(0) * 10.0
        kin = torch.zeros(M, 9, dtype=dtype, device=device)
        P_scr = torch.zeros(M, 9, dtype=dtype, device=device)
        vol = torch.full((M,), 1.0, dtype=dtype, device=device)
        batch = torch.zeros(N, dtype=torch.int32, device=device)
        P = compute_pressure_tensor(vel, mass, virial, cell, kin, P_scr, vol, batch)
        P_mat = P.reshape(M, 3, 3)
        P_diag = torch.diagonal(P_mat, dim1=-2, dim2=-1)
        assert (P_diag > 0).all(), (
            f"Compressive virial should give positive pressure, got diag {P_diag}"
        )

    def test_stress_to_cell_force_shape(self, dtype, device):
        from nvalchemi.dynamics._ops.npt_nph import stress_to_cell_force

        M = 2
        S = torch.randn(M, 3, 3, dtype=dtype, device=device)
        stress = 0.5 * (S + S.transpose(-1, -2))
        cell = (
            torch.eye(3, dtype=dtype, device=device)
            .unsqueeze(0)
            .expand(M, -1, -1)
            .contiguous()
        )
        volume = torch.linalg.det(cell).abs()
        cell_force = stress_to_cell_force(stress, cell, volume)
        assert cell_force.shape == (M, 3, 3)
        assert cell_force.dtype == dtype


# ---------------------------------------------------------------------------
# Integration tests: NVE, NVT (Langevin), NVT (NHC)
# ---------------------------------------------------------------------------


class TestIntegrators:
    """Smoke tests that run a few steps of each integrator."""

    def _make_batch(self, n_atoms: int, seed: int = 0):
        """Return an AtomicData with all fields needed by integrators."""
        from nvalchemi.data import AtomicData

        g = torch.Generator()
        g.manual_seed(seed)
        data = AtomicData(
            positions=torch.randn(n_atoms, 3, generator=g),
            atomic_numbers=torch.randint(
                1, 10, (n_atoms,), dtype=torch.long, generator=g
            ),
            atomic_masses=torch.ones(n_atoms),
            forces=torch.zeros(n_atoms, 3),
            energy=torch.zeros(1, 1),
        )
        data.add_node_property("velocities", torch.zeros(n_atoms, 3))
        return data

    def test_nve_step(self):
        from nvalchemi.data import Batch
        from nvalchemi.dynamics.integrators.nve import NVE
        from nvalchemi.models.demo import DemoModel, DemoModelWrapper

        model = DemoModelWrapper(DemoModel())
        model.eval()
        data = self._make_batch(4, seed=0)
        batch = Batch.from_data_list([data])
        nve = NVE(model=model, dt=0.1)
        for _ in range(3):
            nve.step(batch)
        assert batch.positions.shape == (4, 3)

    def test_nvt_langevin_step(self):
        from nvalchemi.data import Batch
        from nvalchemi.dynamics.integrators.nvt_langevin import NVTLangevin
        from nvalchemi.models.demo import DemoModel, DemoModelWrapper

        model = DemoModelWrapper(DemoModel())
        model.eval()
        data = self._make_batch(4, seed=1)
        batch = Batch.from_data_list([data])
        lang = NVTLangevin(
            model=model, dt=0.1, temperature=300.0, friction=0.1, random_seed=42
        )
        for _ in range(3):
            lang.step(batch)
        assert batch.positions.shape == (4, 3)

    def test_fire_step(self):
        from nvalchemi.data import Batch
        from nvalchemi.dynamics.optimizers.fire import FIRE
        from nvalchemi.models.demo import DemoModel, DemoModelWrapper

        model = DemoModelWrapper(DemoModel())
        model.eval()
        data = self._make_batch(4, seed=2)
        batch = Batch.from_data_list([data])
        fire = FIRE(model=model, dt=0.1)
        for _ in range(3):
            fire.step(batch)
        assert batch.positions.shape == (4, 3)

    def test_fire_uphill_flag(self):
        from nvalchemi.data import Batch
        from nvalchemi.dynamics.optimizers.fire import FIRE
        from nvalchemi.models.demo import DemoModel, DemoModelWrapper

        model = DemoModelWrapper(DemoModel())
        model.eval()
        data = self._make_batch(4, seed=3)
        batch = Batch.from_data_list([data])
        fire = FIRE(model=model, dt=0.1, uphill=True)
        fire._init_state(batch)
        assert fire._state.uphill_flag.shape == (1,)
        assert fire._state.uphill_flag.dtype == torch.int32
        assert int(fire._state.uphill_flag[0]) == 1

    def test_fire_uphill_tensor(self):
        from nvalchemi.data import Batch
        from nvalchemi.dynamics.optimizers.fire import FIRE
        from nvalchemi.models.demo import DemoModel, DemoModelWrapper

        model = DemoModelWrapper(DemoModel())
        model.eval()
        data_list = [self._make_batch(4, seed=i) for i in range(3)]
        batch = Batch.from_data_list(data_list)
        uphill_per_system = torch.tensor([0, 1, 0], dtype=torch.int32)
        fire = FIRE(model=model, dt=0.1, uphill=uphill_per_system)
        fire._init_state(batch)
        assert fire._state.uphill_flag.shape == (3,)
        assert list(fire._state.uphill_flag.tolist()) == [0, 1, 0]

    def test_nvt_nhc_step(self):
        from nvalchemi.data import Batch
        from nvalchemi.dynamics.integrators.nvt_nose_hoover import NVTNoseHoover
        from nvalchemi.models.demo import DemoModel, DemoModelWrapper

        model = DemoModelWrapper(DemoModel())
        model.eval()
        data = self._make_batch(6, seed=4)
        batch = Batch.from_data_list([data])
        nhc = NVTNoseHoover(model=model, dt=0.1, temperature=300.0, thermostat_time=1.0)
        for _ in range(3):
            nhc.step(batch)
        assert batch.positions.shape == (6, 3)

    def test_multi_system_fire(self):
        from nvalchemi.data import Batch
        from nvalchemi.dynamics.optimizers.fire import FIRE
        from nvalchemi.models.demo import DemoModel, DemoModelWrapper

        model = DemoModelWrapper(DemoModel())
        model.eval()
        data_list = [self._make_batch(n, seed=i) for n, i in [(4, 0), (5, 1), (3, 2)]]
        batch = Batch.from_data_list(data_list)
        fire = FIRE(model=model, dt=0.1)
        for _ in range(5):
            fire.step(batch)
        assert batch.positions.shape == (4 + 5 + 3, 3)


# ---------------------------------------------------------------------------
# FIRE hyperparameter tensor shapes in optimizer state
# ---------------------------------------------------------------------------


class TestFireOptimizerState:
    def _make_batch(self, n_atoms: int):
        from nvalchemi.data import AtomicData

        g = torch.Generator()
        g.manual_seed(99)
        data = AtomicData(
            positions=torch.randn(n_atoms, 3, generator=g),
            atomic_numbers=torch.randint(
                1, 10, (n_atoms,), dtype=torch.long, generator=g
            ),
            atomic_masses=torch.ones(n_atoms),
            forces=torch.zeros(n_atoms, 3),
            energy=torch.zeros(1, 1),
        )
        data.add_node_property("velocities", torch.zeros(n_atoms, 3))
        return data

    def test_all_hyperparams_are_tensors_in_state(self):
        from nvalchemi.data import Batch
        from nvalchemi.dynamics.optimizers.fire import FIRE
        from nvalchemi.models.demo import DemoModel, DemoModelWrapper

        model = DemoModelWrapper(DemoModel())
        model.eval()
        data_list = [self._make_batch(4), self._make_batch(5)]
        batch = Batch.from_data_list(data_list)
        M = batch.num_graphs

        fire = FIRE(model=model, dt=0.1, alpha_start=0.15, f_alpha=0.98)
        fire._init_state(batch)

        for key in ["alpha_start", "f_alpha", "maxstep", "f_dec", "f_inc"]:
            val = getattr(fire._state, key)
            assert isinstance(val, torch.Tensor), f"{key} should be a tensor"
            assert val.shape == (M,), f"{key} shape should be ({M},)"

        for key in ["n_min", "uphill_flag", "n_steps_positive"]:
            val = getattr(fire._state, key)
            assert val.dtype == torch.int32, f"{key} should be int32"
            assert val.shape == (M,)
