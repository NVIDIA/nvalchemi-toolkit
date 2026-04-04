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
Tests for BaseDynamics per-system _state batch lifecycle:
  - Lazy initialization via _ensure_state_initialized
  - Correct tensor shapes for every integrator
  - _state.num_graphs == batch.num_graphs invariant
  - State sync after inflight batch refills (refill_check)
  - FusedStage sub-stage initialization via masked_update
"""

from __future__ import annotations

import pytest
import torch

from nvalchemi.data import AtomicData, Batch

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_atomic_data(
    n_atoms: int, seed: int = 0, with_cell: bool = False
) -> AtomicData:
    """Return a minimal AtomicData suitable for dynamics tests."""
    g = torch.Generator()
    g.manual_seed(seed)
    kwargs = dict(
        positions=torch.randn(n_atoms, 3, generator=g),
        atomic_numbers=torch.randint(1, 10, (n_atoms,), dtype=torch.long, generator=g),
        atomic_masses=torch.ones(n_atoms),
        forces=torch.zeros(n_atoms, 3),
        energies=torch.zeros(1, 1),
    )
    if with_cell:
        kwargs["cell"] = torch.eye(3).unsqueeze(0)
        kwargs["stresses"] = torch.zeros(1, 3, 3)
    data = AtomicData(**kwargs)
    data.add_node_property("velocities", torch.zeros(n_atoms, 3))
    return data


def _make_batch(
    n_systems: int, n_atoms_each: int = 4, seed: int = 0, with_cell: bool = False
) -> Batch:
    data_list = [
        _make_atomic_data(n_atoms_each, seed + i, with_cell=with_cell)
        for i in range(n_systems)
    ]
    return Batch.from_data_list(data_list)


def _make_stress_model():
    """Return a DemoModelWrapper subclass that also reports zero stress.

    NPT/NPH declare ``"stress"`` in ``__needs_keys__``, so ``step()``
    requires the model to produce stress in its output dict.  This
    factory builds a minimal subclass that appends a (M, 3, 3) zero
    stress tensor so that ``_validate_model_outputs`` passes.  The
    actual stress value used by NPT/NPH kernels is read from
    ``batch.stresses``, which is initialised to zeros when the batch is
    built with ``with_cell=True``.
    """
    from collections import OrderedDict

    from nvalchemi.models.base import ModelCard
    from nvalchemi.models.demo import DemoModelWrapper

    class _Wrapper(DemoModelWrapper):
        @property
        def model_card(self):
            base = super().model_card
            return ModelCard(
                forces_via_autograd=base.forces_via_autograd,
                supports_energies=base.supports_energies,
                supports_forces=base.supports_forces,
                supports_stresses=True,
                supports_hessians=base.supports_hessians,
                supports_dipoles=base.supports_dipoles,
                supports_non_batch=base.supports_non_batch,
                neighbor_config=base.neighbor_config,
                needs_pbc=base.needs_pbc,
            )

        def adapt_output(self, model_output, data):
            M = data.num_graphs if hasattr(data, "num_graphs") else 1
            return OrderedDict(
                [
                    ("energies", model_output["energies"]),
                    ("forces", model_output["forces"]),
                    (
                        "stresses",
                        torch.zeros(
                            M,
                            3,
                            3,
                            device=data.positions.device,
                            dtype=data.positions.dtype,
                        ),
                    ),
                ]
            )

    return _Wrapper()


def _make_model(needs_stress: bool = False):
    from nvalchemi.models.demo import DemoModelWrapper

    if needs_stress:
        return _make_stress_model()
    return DemoModelWrapper()


# ---------------------------------------------------------------------------
# TestStateLazyInit
# ---------------------------------------------------------------------------


class TestStateLazyInit:
    """Calling step() without manually calling _init_state should succeed."""

    def _run_step(self, dynamics, batch):
        dynamics.step(batch)

    def test_nve_state_initialized_on_first_step(self):
        from nvalchemi.dynamics.integrators.nve import NVE

        model = _make_model()
        batch = _make_batch(2)
        nve = NVE(model=model, dt=0.1)
        assert not hasattr(nve, "_state")
        self._run_step(nve, batch)
        assert hasattr(nve, "_state")

    def test_nvt_langevin_state_initialized_on_first_step(self):
        from nvalchemi.dynamics.integrators.nvt_langevin import NVTLangevin

        model = _make_model()
        batch = _make_batch(2)
        dyn = NVTLangevin(model=model, dt=0.1, temperature=300.0, friction=0.1)
        assert not hasattr(dyn, "_state")
        self._run_step(dyn, batch)
        assert hasattr(dyn, "_state")

    def test_nvt_nhc_state_initialized_on_first_step(self):
        from nvalchemi.dynamics.integrators.nvt_nose_hoover import NVTNoseHoover

        model = _make_model()
        batch = _make_batch(2)
        dyn = NVTNoseHoover(model=model, dt=0.1, temperature=300.0, thermostat_time=1.0)
        assert not hasattr(dyn, "_state")
        self._run_step(dyn, batch)
        assert hasattr(dyn, "_state")

    def test_fire_state_initialized_on_first_step(self):
        from nvalchemi.dynamics.optimizers.fire import FIRE

        model = _make_model()
        batch = _make_batch(2)
        dyn = FIRE(model=model, dt=0.1)
        assert not hasattr(dyn, "_state")
        self._run_step(dyn, batch)
        assert hasattr(dyn, "_state")

    def test_fire2_state_initialized_on_first_step(self):
        from nvalchemi.dynamics.optimizers.fire2 import FIRE2

        model = _make_model()
        batch = _make_batch(2)
        dyn = FIRE2(model=model, dt=0.05)
        assert not hasattr(dyn, "_state")
        self._run_step(dyn, batch)
        assert hasattr(dyn, "_state")

    def test_npt_state_initialized_on_first_step(self):
        # NPT/NPH step() exercises warp kernels that require a GPU or a
        # specific dtype configuration not available in the CPU test env.
        # Validate lazy-init via _ensure_state_initialized directly, which
        # covers the same code path as the top of step().
        from nvalchemi.dynamics.integrators.npt import NPT

        model = _make_model(needs_stress=True)
        batch = _make_batch(2, with_cell=True)
        dyn = NPT(
            model=model,
            dt=0.1,
            temperature=300.0,
            pressure=0.0,
            barostat_time=1.0,
            thermostat_time=1.0,
        )
        assert not hasattr(dyn, "_state")
        dyn._ensure_state_initialized(batch)
        assert hasattr(dyn, "_state")

    def test_nph_state_initialized_on_first_step(self):
        from nvalchemi.dynamics.integrators.nph import NPH

        model = _make_model(needs_stress=True)
        batch = _make_batch(2, with_cell=True)
        dyn = NPH(model=model, dt=0.1, pressure=0.0, barostat_time=1.0)
        assert not hasattr(dyn, "_state")
        dyn._ensure_state_initialized(batch)
        assert hasattr(dyn, "_state")

    def test_demo_dynamics_no_state(self):
        from nvalchemi.dynamics.demo import DemoDynamics

        model = _make_model()
        batch = _make_batch(2)
        dyn = DemoDynamics(model=model, n_steps=1, dt=0.5)
        self._run_step(dyn, batch)
        # DemoDynamics has no _init_state override → _state should not be set
        assert not hasattr(dyn, "_state")

    def test_second_step_does_not_reinitialize(self):
        """_ensure_state_initialized must not overwrite _state on subsequent steps."""
        from nvalchemi.dynamics.integrators.nve import NVE

        model = _make_model()
        batch = _make_batch(1)
        nve = NVE(model=model, dt=0.1)
        nve.step(batch)
        state_id = id(nve._state)
        nve.step(batch)
        # _state object should be the same (not reallocated)
        assert id(nve._state) == state_id


# ---------------------------------------------------------------------------
# TestStateShapes
# ---------------------------------------------------------------------------


class TestStateShapes:
    """Verify _state tensor shapes for each integrator."""

    @pytest.mark.parametrize("M", [1, 3])
    def test_nve_shapes(self, M):
        from nvalchemi.dynamics.integrators.nve import NVE

        model = _make_model()
        batch = _make_batch(M)
        dyn = NVE(model=model, dt=0.1)
        dyn._init_state(batch)
        assert dyn._state.dt.shape == (M,)
        assert dyn._state.num_graphs == M

    @pytest.mark.parametrize("M", [1, 3])
    def test_nvt_langevin_shapes(self, M):
        from nvalchemi.dynamics.integrators.nvt_langevin import NVTLangevin

        model = _make_model()
        batch = _make_batch(M)
        dyn = NVTLangevin(model=model, dt=0.1, temperature=300.0, friction=0.1)
        dyn._init_state(batch)
        assert dyn._state.dt.shape == (M,)
        assert dyn._state.temperature.shape == (M,)
        assert dyn._state.friction.shape == (M,)
        assert dyn._state.num_graphs == M

    @pytest.mark.parametrize("M,C", [(1, 3), (3, 5)])
    def test_nvt_nhc_shapes(self, M, C):
        from nvalchemi.dynamics.integrators.nvt_nose_hoover import NVTNoseHoover

        model = _make_model()
        batch = _make_batch(M)
        dyn = NVTNoseHoover(
            model=model, dt=0.1, temperature=300.0, thermostat_time=1.0, chain_length=C
        )
        dyn._init_state(batch)
        assert dyn._state.dt.shape == (M,)
        assert dyn._state.temperature.shape == (M,)
        assert dyn._state.nhc_eta.shape == (M, C)
        assert dyn._state.nhc_eta_dot.shape == (M, C)
        assert dyn._state.nhc_Q.shape == (M, C)
        assert dyn._state.nhc_ndof.shape == (M,)
        assert dyn._state.num_graphs == M

    @pytest.mark.parametrize("M", [1, 2])
    def test_npt_shapes(self, M):
        from nvalchemi.dynamics.integrators.npt import NPT

        model = _make_model()
        batch = _make_batch(M, with_cell=True)
        C = 3
        dyn = NPT(
            model=model,
            dt=0.1,
            temperature=300.0,
            pressure=0.0,
            barostat_time=1.0,
            thermostat_time=1.0,
            chain_length=C,
        )
        dyn._init_state(batch)
        assert dyn._state.dt.shape == (M,)
        assert dyn._state.nhc_eta.shape == (M, C)
        assert dyn._state.nhc_b_eta.shape == (M, C)
        assert dyn._state.cell_velocity.shape == (M, 3, 3)
        assert dyn._state.num_graphs == M

    @pytest.mark.parametrize("M", [1, 2])
    def test_nph_shapes(self, M):
        from nvalchemi.dynamics.integrators.nph import NPH

        model = _make_model()
        batch = _make_batch(M, with_cell=True)
        dyn = NPH(model=model, dt=0.1, pressure=0.0, barostat_time=1.0)
        dyn._init_state(batch)
        assert dyn._state.dt.shape == (M,)
        assert dyn._state.W.shape == (M,)
        assert dyn._state.cell_velocity.shape == (M, 3, 3)
        assert dyn._state.pressure_tensors.shape == (M, 9)
        assert dyn._state.num_graphs == M

    @pytest.mark.parametrize("M", [1, 3])
    def test_fire_shapes(self, M):
        from nvalchemi.dynamics.optimizers.fire import FIRE

        model = _make_model()
        batch = _make_batch(M)
        dyn = FIRE(model=model, dt=0.1)
        dyn._init_state(batch)
        for key in [
            "dt",
            "alpha",
            "alpha_start",
            "f_alpha",
            "maxstep",
            "f_dec",
            "f_inc",
            "dt_max",
            "dt_min",
        ]:
            assert getattr(dyn._state, key).shape == (M,), key
        for key in ["n_min", "uphill_flag", "n_steps_positive"]:
            val = getattr(dyn._state, key)
            assert val.shape == (M,), key
            assert val.dtype == torch.int32, key
        assert dyn._state.num_graphs == M

    @pytest.mark.parametrize("M", [1, 3])
    def test_fire_variable_cell_shapes(self, M):
        from nvalchemi.dynamics.optimizers.fire import FIREVariableCell

        model = _make_model()
        batch = _make_batch(M, with_cell=True)
        dyn = FIREVariableCell(model=model, dt=0.1)
        dyn._init_state(batch)
        assert dyn._state.cell_velocity.shape == (M, 3, 3)
        assert dyn._state.dt.shape == (M,)
        assert dyn._state.num_graphs == M

    @pytest.mark.parametrize("M", [1, 3])
    def test_fire2_shapes(self, M):
        from nvalchemi.dynamics.optimizers.fire2 import FIRE2

        model = _make_model()
        batch = _make_batch(M)
        dyn = FIRE2(model=model, dt=0.05)
        dyn._init_state(batch)
        assert dyn._state.dt.shape == (M,)
        assert dyn._state.alpha.shape == (M,)
        assert dyn._state.nsteps_inc.shape == (M,)
        assert dyn._state.nsteps_inc.dtype == torch.int32
        assert dyn._state.num_graphs == M

    @pytest.mark.parametrize("M", [1, 2])
    def test_fire2_variable_cell_shapes(self, M):
        from nvalchemi.dynamics.optimizers.fire2 import FIRE2VariableCell

        model = _make_model()
        batch = _make_batch(M, with_cell=True)
        dyn = FIRE2VariableCell(model=model, dt=0.05)
        dyn._init_state(batch)
        assert dyn._state.cell_velocities.shape == (M, 3, 3)
        assert dyn._state.num_graphs == M

    def test_nve_state_dtype_matches_positions(self):
        from nvalchemi.dynamics.integrators.nve import NVE

        model = _make_model()
        batch = _make_batch(1)
        dyn = NVE(model=model, dt=0.1)
        dyn._init_state(batch)
        # State dtype must match the actual positions dtype of the batch.
        assert dyn._state.dt.dtype == batch.positions.dtype


# ---------------------------------------------------------------------------
# TestStateInvariant
# ---------------------------------------------------------------------------


class TestStateInvariant:
    """_state.num_graphs must always equal batch.num_graphs after step()."""

    def test_nve_invariant_multi_system(self):
        from nvalchemi.dynamics.integrators.nve import NVE

        model = _make_model()
        batch = _make_batch(3)
        dyn = NVE(model=model, dt=0.1)
        for _ in range(5):
            dyn.step(batch)
            assert dyn._state.num_graphs == batch.num_graphs

    def test_fire_invariant_heterogeneous(self):
        from nvalchemi.dynamics.optimizers.fire import FIRE

        model = _make_model()
        data_list = [_make_atomic_data(n, seed=i) for n, i in [(3, 0), (5, 1), (4, 2)]]
        batch = Batch.from_data_list(data_list)
        dyn = FIRE(model=model, dt=0.1)
        for _ in range(3):
            dyn.step(batch)
            assert dyn._state.num_graphs == batch.num_graphs == 3


# ---------------------------------------------------------------------------
# TestMakeNewState
# ---------------------------------------------------------------------------


class TestMakeNewState:
    """Unit tests for _make_new_state on each integrator."""

    def _template(
        self, n_systems: int = 2, n_atoms: int = 4, with_cell: bool = False
    ) -> Batch:
        return _make_batch(n_systems, n_atoms, with_cell=with_cell)

    @pytest.mark.parametrize("n_new", [1, 4])
    def test_nve_make_new_state(self, n_new):
        from nvalchemi.dynamics.integrators.nve import NVE

        template = self._template()
        dyn = NVE(model=_make_model(), dt=0.1)
        state = dyn._make_new_state(n_new, template)
        assert state is not None
        assert state.dt.shape == (n_new,)
        assert state.num_graphs == n_new

    @pytest.mark.parametrize("n_new", [1, 3])
    def test_nvt_langevin_make_new_state(self, n_new):
        from nvalchemi.dynamics.integrators.nvt_langevin import NVTLangevin

        template = self._template()
        dyn = NVTLangevin(model=_make_model(), dt=0.1, temperature=300.0, friction=0.1)
        state = dyn._make_new_state(n_new, template)
        assert state is not None
        assert state.dt.shape == (n_new,)
        assert state.temperature.shape == (n_new,)

    @pytest.mark.parametrize("n_new", [1, 2])
    def test_nvt_nhc_make_new_state(self, n_new):
        from nvalchemi.dynamics.integrators.nvt_nose_hoover import NVTNoseHoover

        template = self._template()
        C = 3
        dyn = NVTNoseHoover(
            model=_make_model(),
            dt=0.1,
            temperature=300.0,
            thermostat_time=1.0,
            chain_length=C,
        )
        state = dyn._make_new_state(n_new, template)
        assert state is not None
        assert state.nhc_eta.shape == (n_new, C)
        assert state.num_graphs == n_new

    @pytest.mark.parametrize("n_new", [1, 2])
    def test_npt_make_new_state(self, n_new):
        from nvalchemi.dynamics.integrators.npt import NPT

        template = self._template(with_cell=True)
        dyn = NPT(
            model=_make_model(),
            dt=0.1,
            temperature=300.0,
            pressure=0.0,
            barostat_time=1.0,
            thermostat_time=1.0,
        )
        state = dyn._make_new_state(n_new, template)
        assert state is not None
        assert state.cell_velocity.shape == (n_new, 3, 3)

    @pytest.mark.parametrize("n_new", [1, 3])
    def test_fire_make_new_state(self, n_new):
        from nvalchemi.dynamics.optimizers.fire import FIRE

        template = self._template()
        dyn = FIRE(model=_make_model(), dt=0.1)
        state = dyn._make_new_state(n_new, template)
        assert state is not None
        assert state.dt.shape == (n_new,)
        assert state.n_steps_positive.shape == (n_new,)
        # Fresh state: n_steps_positive should be zero
        assert (state.n_steps_positive == 0).all()

    @pytest.mark.parametrize("n_new", [1, 2])
    def test_fire2_make_new_state(self, n_new):
        from nvalchemi.dynamics.optimizers.fire2 import FIRE2

        template = self._template()
        dyn = FIRE2(model=_make_model(), dt=0.05)
        state = dyn._make_new_state(n_new, template)
        assert state is not None
        assert state.alpha.shape == (n_new,)
        assert state.num_graphs == n_new

    def test_demo_dynamics_make_new_state_returns_none(self):
        from nvalchemi.dynamics.demo import DemoDynamics

        template = self._template()
        dyn = DemoDynamics(model=_make_model(), n_steps=1)
        result = dyn._make_new_state(2, template)
        assert result is None


# ---------------------------------------------------------------------------
# TestFusedStageStateInit
# ---------------------------------------------------------------------------


class TestFusedStageStateInit:
    """FusedStage sub-stages must have _state initialized via masked_update."""

    def _make_status_batch(self, n_systems: int, n_atoms: int = 4) -> Batch:
        """Batch with status and fmax tensors; half status=0, half status=1."""
        batch = _make_batch(n_systems, n_atoms)
        status = torch.zeros(n_systems, 1, dtype=torch.long)
        for i in range(n_systems // 2):
            status[i] = 1
        batch["status"] = status
        # fmax is required by the auto-registered ConvergenceHook in FusedStage.
        batch["fmax"] = torch.full((n_systems, 1), float("inf"))
        return batch

    def test_sub_stage_state_initialized_after_fused_step(self):
        from nvalchemi.dynamics.integrators.nvt_langevin import NVTLangevin
        from nvalchemi.dynamics.optimizers.fire import FIRE

        model = _make_model()
        fire = FIRE(model=model, dt=0.1)
        lang = NVTLangevin(model=model, dt=0.1, temperature=300.0, friction=0.1)

        fused = fire + lang
        batch = self._make_status_batch(4)

        assert not hasattr(fire, "_state")
        assert not hasattr(lang, "_state")

        fused.step(batch)

        assert hasattr(fire, "_state")
        assert hasattr(lang, "_state")

    def test_sub_stage_state_shapes_match_full_batch(self):
        """Sub-stage _state.num_graphs == batch.num_graphs (not masked count)."""
        from nvalchemi.dynamics.integrators.nvt_langevin import NVTLangevin
        from nvalchemi.dynamics.optimizers.fire import FIRE

        model = _make_model()
        fire = FIRE(model=model, dt=0.1)
        lang = NVTLangevin(model=model, dt=0.1, temperature=300.0, friction=0.1)

        fused = fire + lang
        M = 4
        batch = self._make_status_batch(M)
        fused.step(batch)

        assert fire._state.num_graphs == M
        assert lang._state.num_graphs == M


# ---------------------------------------------------------------------------
# TestStateSyncInflight
# ---------------------------------------------------------------------------


class _MockSampler:
    """Minimal sampler stub for inflight batching tests."""

    def __init__(self, replacements: list):
        self._queue = list(replacements)
        # Eagerly reflect exhausted state so base.py can snapshot it before requesting
        self.exhausted = len(self._queue) == 0

    def request_replacement(self, n_atoms: int, n_edges: int):
        if not self._queue:
            self.exhausted = True
            return None
        result = self._queue.pop(0)
        self.exhausted = len(self._queue) == 0
        return result


class TestStateSyncInflight:
    """refill_check must keep _state in sync with batch composition."""

    def _make_dynamics_with_sampler(self, replacements):
        from nvalchemi.dynamics.optimizers.fire import FIRE

        model = _make_model()
        sampler = _MockSampler(replacements)
        dyn = FIRE(model=model, dt=0.1, sampler=sampler)
        return dyn, sampler

    def _make_status_batch(self, n_systems: int, n_atoms: int = 4) -> Batch:
        """Batch with all-zero status (none graduated)."""
        batch = _make_batch(n_systems, n_atoms)
        batch["status"] = torch.zeros(n_systems, 1, dtype=torch.long)
        return batch

    def _graduate_first(self, batch: Batch) -> None:
        """Mark system 0 as graduated in-place."""
        batch.status[0] = 1

    def test_state_shrinks_when_system_graduates_no_replacement(self):
        """When a system graduates and the sampler is empty, _state loses a row."""
        dyn, sampler = self._make_dynamics_with_sampler(replacements=[])
        batch = self._make_status_batch(3)

        # Initialize state via step
        dyn.step(batch)
        assert dyn._state.num_graphs == 3

        # Graduate system 0
        self._graduate_first(batch)

        # refill_check(): 1 graduated, no replacement — 2 systems remain active.
        result = dyn.refill_check(batch, exit_status=1)

        # 2 systems still running; the batch is NOT None.
        assert result is not None
        assert result.num_graphs == 2
        # State must shrink to match the 2 remaining systems.
        assert dyn._state.num_graphs == 2
        # Run is not done — 2 systems still need to finish.
        assert dyn.done is False

    def test_state_shrinks_when_system_graduates_with_replacement(self):
        """When 1 of 3 systems graduates and 1 replacement is provided."""
        replacement = _make_atomic_data(4, seed=99)
        dyn, sampler = self._make_dynamics_with_sampler(replacements=[replacement])

        batch = self._make_status_batch(3)
        dyn.step(batch)
        assert dyn._state.num_graphs == 3

        self._graduate_first(batch)
        result = dyn.refill_check(batch, exit_status=1)

        # 2 remaining + 1 replacement = 3 systems
        assert result is not None
        assert result.num_graphs == 3
        assert dyn._state.num_graphs == 3

    def test_state_preserves_remaining_values(self):
        """After refill, the remaining systems' state is unchanged."""
        replacement = _make_atomic_data(4, seed=88)
        dyn, _ = self._make_dynamics_with_sampler(replacements=[replacement])

        batch = self._make_status_batch(3)
        dyn.step(batch)

        # Run a few more steps to evolve state (n_steps_positive changes with FIRE).
        for _ in range(5):
            dyn.step(batch)

        # Capture state for systems 1 and 2 before refill.
        alpha_before = dyn._state.alpha[1:].clone()

        self._graduate_first(batch)
        result = dyn.refill_check(batch, exit_status=1)

        assert result is not None
        # After refill: system 0 (old system 1) and system 1 (old system 2)
        # should have preserved alpha values.
        torch.testing.assert_close(dyn._state.alpha[:2], alpha_before)

    def test_new_system_gets_fresh_state(self):
        """Replacement system should have default (reset) state values."""
        replacement = _make_atomic_data(4, seed=77)
        dyn, _ = self._make_dynamics_with_sampler(replacements=[replacement])

        batch = self._make_status_batch(2)
        dyn.step(batch)

        # Run steps to evolve state away from defaults.
        for _ in range(10):
            dyn.step(batch)

        self._graduate_first(batch)
        result = dyn.refill_check(batch, exit_status=1)

        assert result is not None
        assert dyn._state.num_graphs == 2

        # New system (index 1) should have n_steps_positive reset to 0.
        assert int(dyn._state.n_steps_positive[-1]) == 0
        # And alpha reset to alpha_start.
        expected_alpha = dyn.alpha_start
        assert abs(float(dyn._state.alpha[-1]) - expected_alpha) < 1e-6

    def test_full_graduation_clears_state(self):
        """When all systems graduate and no replacements, _state is deleted."""
        dyn, _ = self._make_dynamics_with_sampler(replacements=[])

        batch = self._make_status_batch(2)
        dyn.step(batch)
        assert hasattr(dyn, "_state")

        # Graduate all systems.
        batch.status[:] = 1
        dyn.refill_check(batch, exit_status=1)

        assert not hasattr(dyn, "_state")
        assert dyn.done is True


# ---------------------------------------------------------------------------
# TestFusedStageStateSyncInflight
# ---------------------------------------------------------------------------


class TestFusedStageStateSyncInflight:
    """FusedStage._sync_state_to_batch must propagate to all sub-stages.

    State is only initialized in a sub-stage when ``masked_update`` is
    called with a non-empty mask (i.e. when at least one system has that
    sub-stage's status code).  All test batches here therefore include at
    least one system at each sub-stage status so that both FIRE (status=0)
    and NVTLangevin (status=1) have their ``_state`` populated after the
    first ``fused.step()`` call.
    """

    def _make_fused_with_sampler(self, replacements):
        from nvalchemi.dynamics.integrators.nvt_langevin import NVTLangevin
        from nvalchemi.dynamics.optimizers.fire import FIRE

        model = _make_model()
        fire = FIRE(model=model, dt=0.1)
        lang = NVTLangevin(model=model, dt=0.1, temperature=300.0, friction=0.1)
        fused = fire + lang
        fused.sampler = _MockSampler(replacements)
        return fused, fire, lang

    def _make_mixed_status_batch(self, n_systems: int, n_atoms: int = 4) -> Batch:
        """Batch with alternating status=0 / status=1 so both sub-stages are active."""
        batch = _make_batch(n_systems, n_atoms)
        status = torch.zeros(n_systems, 1, dtype=torch.long)
        for i in range(n_systems):
            status[i] = i % 2  # alternates 0, 1, 0, 1, ...
        batch["status"] = status
        batch["fmax"] = torch.full((n_systems, 1), float("inf"))
        return batch

    def test_sub_stages_shrink_when_system_graduates_with_replacement(self):
        """After refill, both sub-stage _state tensors match the new batch size."""
        replacement = _make_atomic_data(4, seed=77)
        fused, fire, lang = self._make_fused_with_sampler(replacements=[replacement])

        # 4 systems: status [0, 1, 0, 1] — both sub-stages see at least one system.
        M = 4
        batch = self._make_mixed_status_batch(M)
        fused.step(batch)

        assert fire._state.num_graphs == M
        assert lang._state.num_graphs == M

        # Graduate system 0 (status must reach fused.exit_status=2 to be ejected).
        batch.status[0] = fused.exit_status
        result = fused.refill_check(batch, exit_status=fused.exit_status)

        # 3 remaining + 1 replacement = M systems again.
        assert result is not None
        assert result.num_graphs == M
        assert fire._state.num_graphs == M
        assert lang._state.num_graphs == M

    def test_sub_stages_shrink_when_system_graduates_no_replacement(self):
        """With no replacement, both sub-stage _state tensors shrink."""
        fused, fire, lang = self._make_fused_with_sampler(replacements=[])

        M = 4
        batch = self._make_mixed_status_batch(M)
        fused.step(batch)

        batch.status[0] = fused.exit_status
        result = fused.refill_check(batch, exit_status=fused.exit_status)

        assert result is not None
        assert result.num_graphs == M - 1
        assert fire._state.num_graphs == M - 1
        assert lang._state.num_graphs == M - 1

    def test_sub_stages_state_cleared_on_full_graduation(self):
        """When all systems graduate and sampler is empty, sub-stage _state is deleted."""
        fused, fire, lang = self._make_fused_with_sampler(replacements=[])

        # 2 systems: status [0, 1] — both sub-stages are active.
        batch = self._make_mixed_status_batch(2)
        fused.step(batch)

        assert hasattr(fire, "_state")
        assert hasattr(lang, "_state")

        batch.status[:] = fused.exit_status
        fused.refill_check(batch, exit_status=fused.exit_status)

        assert not hasattr(fire, "_state")
        assert not hasattr(lang, "_state")
        assert fused.done is True
