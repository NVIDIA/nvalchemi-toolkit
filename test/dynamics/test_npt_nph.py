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
Unit tests for the NPH and NPT integrators.

Tests cover constructor parameter storage, class-level key declarations,
state initialisation via ``_init_state`` / ``_make_new_state``, and the
``pre_update`` / ``post_update`` step routines.  All tests require the
``nvalchemiops`` warp extension and are skipped when it is absent.
"""

from __future__ import annotations

import pytest
import torch

from nvalchemi.data import AtomicData, Batch
from nvalchemi.dynamics._units import fs_to_internal_time
from nvalchemi.dynamics.integrators.nph import NPH
from nvalchemi.dynamics.integrators.npt import NPT
from nvalchemi.models.demo import DemoModel, DemoModelWrapper

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_barostat_batch(
    n_atoms: int = 4,
    n_graphs: int = 1,
    device: str = "cpu",
    int_dtype: torch.dtype = torch.long,
) -> Batch:
    """Build a minimal Batch suitable for NPH / NPT integrator tests."""
    dtype = torch.float32
    atoms_per = n_atoms // n_graphs
    data_list = [
        AtomicData(
            atomic_numbers=torch.tensor([18] * atoms_per, dtype=int_dtype),
            positions=torch.randn(atoms_per, 3),
        )
        for _ in range(n_graphs)
    ]
    batch = Batch.from_data_list(data_list).to(device)
    N = batch.num_nodes
    B = batch.num_graphs
    batch["velocities"] = torch.randn(N, 3, dtype=dtype, device=device) * 0.1
    batch["forces"] = torch.zeros(N, 3, dtype=dtype, device=device)
    batch["atomic_masses"] = torch.full(
        (N,), 39.948, dtype=dtype, device=device
    )  # Argon
    batch["cell"] = (
        torch.eye(3, dtype=dtype, device=device)
        .unsqueeze(0)
        .expand(B, -1, -1)
        .contiguous()
        * 10.0
    )
    batch["stress"] = torch.zeros(B, 3, 3, dtype=dtype, device=device)
    return batch


# ---------------------------------------------------------------------------
# NPH tests
# ---------------------------------------------------------------------------


class TestNPHIntegrator:
    """Tests for the NPH (isenthalpic-isobaric) integrator."""

    @pytest.fixture(autouse=True)
    def _require_ops(self):
        """Skip the entire class when nvalchemiops is not installed."""
        pytest.importorskip("nvalchemiops")

    @pytest.fixture
    def nph(self):
        """Return a freshly constructed NPH integrator."""
        return NPH(
            model=DemoModelWrapper(DemoModel()),
            dt=0.001,
            pressure=1.0,
            barostat_time=100.0,
            pressure_coupling="isotropic",
        )

    @pytest.fixture
    def nph_with_state(self, nph):
        """Return an NPH integrator whose state has been initialised."""
        batch = _make_barostat_batch()
        nph._init_state(batch)
        return nph, batch

    # ------------------------------------------------------------------
    # Constructor
    # ------------------------------------------------------------------

    def test_init_stores_parameters(self, nph):
        """Constructor arguments are stored on the integrator."""
        assert nph._dt_init == fs_to_internal_time(0.001)
        assert nph._pressure_init == 1.0
        assert nph._barostat_time_init == fs_to_internal_time(100.0)
        assert nph.pressure_coupling == "isotropic"

    def test_needs_keys(self):
        """NPH declares the correct set of required input keys."""
        assert NPH.__needs_keys__ == {"forces", "stress"}

    def test_provides_keys(self):
        """NPH declares the correct set of provided output keys."""
        assert NPH.__provides_keys__ == {"positions", "velocities", "cell"}

    # ------------------------------------------------------------------
    # _init_state
    # ------------------------------------------------------------------

    def test_init_state_creates_state(self, nph):
        """_init_state populates ``_state``."""
        batch = _make_barostat_batch()
        nph._init_state(batch)
        assert nph._state is not None

    def test_init_state_state_has_expected_keys(self, nph):
        """After _init_state the state object carries all expected attributes."""
        batch = _make_barostat_batch()
        nph._init_state(batch)
        state = nph._state
        for attr in ("dt", "pressure", "W", "cell_velocity", "kinetic_tensors"):
            assert hasattr(state, attr), f"state is missing attribute '{attr}'"

    def test_init_state_shapes(self, nph):
        """W is a (M,) tensor and cell_velocity is a (M, 3, 3) tensor."""
        batch = _make_barostat_batch(n_atoms=4, n_graphs=1)
        nph._init_state(batch)
        M = batch.num_graphs
        assert nph._state.W.shape == (M,)
        assert nph._state.cell_velocity.shape == (M, 3, 3)

    # ------------------------------------------------------------------
    # _make_new_state
    # ------------------------------------------------------------------

    def test_make_new_state_returns_batch(self, nph):
        """_make_new_state returns a Batch-like object with the required attrs."""
        template = _make_barostat_batch()
        new_state = nph._make_new_state(2, template)
        assert new_state is not None
        assert hasattr(new_state, "W")
        assert hasattr(new_state, "cell_velocity")
        assert hasattr(new_state, "dt")
        assert hasattr(new_state, "pressure")

    # ------------------------------------------------------------------
    # pre_update / post_update
    # ------------------------------------------------------------------

    def test_pre_update_runs_without_error(self, nph_with_state):
        """pre_update completes without raising an exception."""
        nph, batch = nph_with_state
        nph.pre_update(batch)

    def test_post_update_runs_without_error(self, nph_with_state):
        """post_update completes without raising an exception after pre_update."""
        nph, batch = nph_with_state
        nph.pre_update(batch)
        nph.post_update(batch)

    @pytest.mark.parametrize("int_dtype", [torch.int32, torch.int64])
    def test_pre_post_update_with_int_dtypes(self, nph, device, int_dtype: torch.dtype):
        """NPH pre_update/post_update work with both int32 and int64 indices."""
        batch = _make_barostat_batch(int_dtype=int_dtype, device=device)
        nph._init_state(batch)
        nph.pre_update(batch)
        nph.post_update(batch)

    def test_pre_update_modifies_positions(self, nph_with_state):
        """pre_update changes at least some atomic positions in the batch."""
        nph, batch = nph_with_state
        positions_before = batch.positions.clone()
        nph.pre_update(batch)
        assert not torch.allclose(batch.positions, positions_before), (
            "pre_update did not modify positions"
        )

    # ------------------------------------------------------------------
    # Multi-graph
    # ------------------------------------------------------------------

    def test_multi_graph_init_state(self, nph):
        """_init_state correctly handles a batch containing multiple graphs."""
        batch = _make_barostat_batch(n_atoms=8, n_graphs=2)
        nph._init_state(batch)
        M = batch.num_graphs
        assert nph._state.W.shape == (M,)
        assert nph._state.cell_velocity.shape == (M, 3, 3)

    # ------------------------------------------------------------------
    # Non-isotropic modes
    # ------------------------------------------------------------------

    @pytest.mark.parametrize(
        "mode,pressure",
        [
            ("anisotropic", torch.tensor([[1.0, 1.0, 1.0]])),
            ("triclinic", torch.eye(3).unsqueeze(0)),
        ],
    )
    def test_W_divided_for_non_isotropic(self, mode, pressure):
        """Barostat mass W is divided by 3 for non-isotropic modes."""
        nph_iso = NPH(
            model=DemoModelWrapper(DemoModel()),
            dt=0.001,
            pressure=1.0,
            barostat_time=100.0,
            pressure_coupling="isotropic",
        )
        nph_aniso = NPH(
            model=DemoModelWrapper(DemoModel()),
            dt=0.001,
            pressure=pressure,
            barostat_time=100.0,
            pressure_coupling=mode,
        )
        batch = _make_barostat_batch()
        nph_iso._init_state(batch)
        nph_aniso._init_state(batch)
        assert torch.allclose(nph_iso._state.W / 3, nph_aniso._state.W, atol=1e-6)

    @pytest.mark.parametrize(
        "mode,pressure",
        [
            ("anisotropic", torch.tensor([[1.0, 1.0, 1.0]])),
            ("triclinic", torch.eye(3).unsqueeze(0)),
        ],
    )
    def test_pre_post_update_runs_for_non_isotropic_modes(self, mode, pressure):
        """pre_update and post_update complete for non-isotropic modes."""
        nph = NPH(
            model=DemoModelWrapper(DemoModel()),
            dt=0.001,
            pressure=pressure,
            barostat_time=100.0,
            pressure_coupling=mode,
        )
        batch = _make_barostat_batch()
        nph._init_state(batch)
        nph.pre_update(batch)
        nph.post_update(batch)


# ---------------------------------------------------------------------------
# NPT tests
# ---------------------------------------------------------------------------


class TestNPTIntegrator:
    """Tests for the NPT (isothermal-isobaric) integrator."""

    @pytest.fixture(autouse=True)
    def _require_ops(self):
        """Skip the entire class when nvalchemiops is not installed."""
        pytest.importorskip("nvalchemiops")

    @pytest.fixture
    def npt(self):
        """Return a freshly constructed NPT integrator with chain_length=3."""
        return NPT(
            model=DemoModelWrapper(DemoModel()),
            dt=0.001,
            temperature=300.0,
            pressure=1.0,
            barostat_time=100.0,
            thermostat_time=100.0,
            pressure_coupling="isotropic",
            chain_length=3,
        )

    @pytest.fixture
    def npt_with_state(self, npt):
        """Return an NPT integrator whose state has been initialised."""
        batch = _make_barostat_batch()
        npt._init_state(batch)
        return npt, batch

    # ------------------------------------------------------------------
    # Constructor
    # ------------------------------------------------------------------

    def test_init_stores_parameters(self, npt):
        """Constructor arguments are stored on the integrator."""
        assert npt._dt_init == fs_to_internal_time(0.001)
        assert npt._temperature_init == 300.0
        assert npt._pressure_init == 1.0
        assert npt._barostat_time_init == fs_to_internal_time(100.0)
        assert npt._thermostat_time_init == fs_to_internal_time(100.0)
        assert npt.pressure_coupling == "isotropic"
        assert npt.chain_length == 3

    def test_needs_keys(self):
        """NPT declares the correct set of required input keys."""
        assert NPT.__needs_keys__ == {"forces", "stress"}

    def test_provides_keys(self):
        """NPT declares the correct set of provided output keys."""
        assert NPT.__provides_keys__ == {"positions", "velocities", "cell"}

    # ------------------------------------------------------------------
    # _init_state – NHC
    # ------------------------------------------------------------------

    def test_init_state_creates_nhc_state(self, npt):
        """_init_state populates the NHC attributes on _state."""
        batch = _make_barostat_batch()
        npt._init_state(batch)
        state = npt._state
        for attr in (
            "nhc_eta",
            "nhc_Q",
            "nhc_b_Q",
            "nhc_b_eta",
            "nhc_eta_dot",
            "nhc_b_eta_dot",
        ):
            assert hasattr(state, attr), f"state is missing NHC attribute '{attr}'"

    def test_init_state_nhc_shapes(self, npt):
        """nhc_eta has shape (M, chain_length) after _init_state."""
        batch = _make_barostat_batch(n_atoms=4, n_graphs=1)
        npt._init_state(batch)
        M = batch.num_graphs
        assert npt._state.nhc_eta.shape == (M, npt.chain_length)
        assert npt._state.nhc_Q.shape == (M, npt.chain_length)

    # ------------------------------------------------------------------
    # _make_new_state
    # ------------------------------------------------------------------

    def test_make_new_state_returns_batch(self, npt):
        """_make_new_state returns a Batch-like object with NHC attributes."""
        template = _make_barostat_batch()
        new_state = npt._make_new_state(2, template)
        assert new_state is not None
        assert hasattr(new_state, "nhc_Q")
        assert hasattr(new_state, "nhc_b_Q")
        assert hasattr(new_state, "W")
        assert hasattr(new_state, "cell_velocity")

    # ------------------------------------------------------------------
    # pre_update / post_update
    # ------------------------------------------------------------------

    def test_pre_update_runs_without_error(self, npt_with_state):
        """pre_update completes without raising an exception."""
        npt, batch = npt_with_state
        npt.pre_update(batch)

    def test_post_update_runs_without_error(self, npt_with_state):
        """post_update completes without raising an exception after pre_update."""
        npt, batch = npt_with_state
        npt.pre_update(batch)
        npt.post_update(batch)

    # ------------------------------------------------------------------
    # chain_length respected
    # ------------------------------------------------------------------

    def test_chain_length_respected(self):
        """The NHC arrays have the requested chain_length as their second dimension."""
        npt5 = NPT(
            model=DemoModelWrapper(DemoModel()),
            dt=0.001,
            temperature=300.0,
            pressure=1.0,
            barostat_time=100.0,
            thermostat_time=100.0,
            chain_length=5,
        )
        batch = _make_barostat_batch()
        npt5._init_state(batch)
        assert npt5._state.nhc_Q.shape[1] == 5
        assert npt5._state.nhc_b_Q.shape[1] == 5

    # ------------------------------------------------------------------
    # Non-isotropic modes
    # ------------------------------------------------------------------

    @pytest.mark.parametrize(
        "mode,pressure",
        [
            ("anisotropic", torch.tensor([[1.0, 1.0, 1.0]])),
            ("triclinic", torch.eye(3).unsqueeze(0)),
        ],
    )
    def test_init_state_preserves_pressure_shape(self, mode, pressure):
        """_init_state keeps anisotropic [M,3] and triclinic [M,3,3] shapes."""
        npt = NPT(
            model=DemoModelWrapper(DemoModel()),
            dt=0.001,
            temperature=300.0,
            pressure=pressure,
            barostat_time=100.0,
            thermostat_time=100.0,
            pressure_coupling=mode,
        )
        batch = _make_barostat_batch()
        npt._init_state(batch)
        assert npt._state.pressure.shape == pressure.shape

    @pytest.mark.parametrize(
        "mode,pressure",
        [
            ("isotropic", 1.0),
            ("anisotropic", torch.tensor([[1.0, 1.0, 1.0]])),
            ("triclinic", torch.eye(3).unsqueeze(0)),
        ],
    )
    def test_nhc_b_Q_shape_all_modes(self, mode, pressure):
        """Barostat NHC chain masses have shape [M, chain_length] for all modes."""
        chain_length = 3
        npt = NPT(
            model=DemoModelWrapper(DemoModel()),
            dt=0.001,
            temperature=300.0,
            pressure=pressure,
            barostat_time=100.0,
            thermostat_time=100.0,
            pressure_coupling=mode,
            chain_length=chain_length,
        )
        batch = _make_barostat_batch()
        npt._init_state(batch)
        M = batch.num_graphs
        assert npt._state.nhc_b_Q.shape == (M, chain_length)

    @pytest.mark.parametrize(
        "mode,pressure",
        [
            ("anisotropic", torch.tensor([[1.0, 1.0, 1.0]])),
            ("triclinic", torch.eye(3).unsqueeze(0)),
        ],
    )
    def test_W_divided_for_non_isotropic(self, mode, pressure):
        """Barostat mass W is divided by 3 for non-isotropic modes."""
        npt_iso = NPT(
            model=DemoModelWrapper(DemoModel()),
            dt=0.001,
            temperature=300.0,
            pressure=1.0,
            barostat_time=100.0,
            thermostat_time=100.0,
            pressure_coupling="isotropic",
        )
        npt_aniso = NPT(
            model=DemoModelWrapper(DemoModel()),
            dt=0.001,
            temperature=300.0,
            pressure=pressure,
            barostat_time=100.0,
            thermostat_time=100.0,
            pressure_coupling=mode,
        )
        batch = _make_barostat_batch()
        npt_iso._init_state(batch)
        npt_aniso._init_state(batch)
        assert torch.allclose(npt_iso._state.W / 3, npt_aniso._state.W, atol=1e-6)

    @pytest.mark.parametrize(
        "mode,pressure",
        [
            ("anisotropic", torch.tensor([[1.0, 1.0, 1.0]])),
            ("triclinic", torch.eye(3).unsqueeze(0)),
        ],
    )
    def test_pre_post_update_runs_for_non_isotropic_modes(self, mode, pressure):
        """pre_update and post_update complete for non-isotropic modes."""
        npt = NPT(
            model=DemoModelWrapper(DemoModel()),
            dt=0.001,
            temperature=300.0,
            pressure=pressure,
            barostat_time=100.0,
            thermostat_time=100.0,
            pressure_coupling=mode,
        )
        batch = _make_barostat_batch()
        npt._init_state(batch)
        npt.pre_update(batch)
        npt.post_update(batch)
