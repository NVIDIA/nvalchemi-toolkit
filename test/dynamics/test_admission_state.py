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
"""Integrator state for systems admitted into a running stage (issue #233).

Inflight refill and pipeline receive append arrivals after the resident
systems and build their state with ``_make_new_state``.  Degrees of freedom,
chain masses, and barostat masses depend on each arrival's own atom count, so
that state must equal what ``_init_state`` builds for the arrivals alone.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
import torch

from nvalchemi.data import AtomicData, Batch
from nvalchemi.dynamics import (
    NPH,
    NPT,
    DynamicsStage,
    FusedStage,
    NVTNoseHoover,
    SizeAwareSampler,
)
from nvalchemi.hooks import DynamicsContext

from .conftest import _make_atomic_data, _make_model

_RESIDENT_SIZES = (4, 7)
_ARRIVAL_SIZES = (2, 9, 13)


def _nose_hoover(**kwargs) -> NVTNoseHoover:
    """Return a Nosé-Hoover NVT integrator targeting 300 K."""
    return NVTNoseHoover(
        model=_make_model(), dt=0.5, temperature=300.0, thermostat_time=50.0, **kwargs
    )


def _npt() -> NPT:
    """Return an isotropic NPT integrator targeting 300 K."""
    return NPT(
        model=_make_model(needs_stress=True),
        dt=0.5,
        temperature=300.0,
        pressure=0.0,
        barostat_time=500.0,
        thermostat_time=50.0,
    )


def _nph() -> NPH:
    """Return an isotropic NPH integrator."""
    return NPH(
        model=_make_model(needs_stress=True),
        dt=0.5,
        pressure=0.0,
        barostat_time=500.0,
    )


_INTEGRATORS = pytest.mark.parametrize(
    "make_dynamics", [_nose_hoover, _npt, _nph], ids=["NVTNoseHoover", "NPT", "NPH"]
)


class TestAdmissionStateInvariant:
    """``_make_new_state`` matches ``_init_state`` on the arrivals alone."""

    @_INTEGRATORS
    @pytest.mark.parametrize("n_arrivals", [1, 2, 3])
    def test_arrival_state_matches_init_state(self, make_dynamics, n_arrivals, device):
        """Every state field equals the arrivals-only ``_init_state`` value."""
        residents = [
            _make_atomic_data(n, seed=i, with_cell=True)
            for i, n in enumerate(_RESIDENT_SIZES)
        ]
        arrivals = [
            _make_atomic_data(n, seed=10 + i, with_cell=True)
            for i, n in enumerate(_ARRIVAL_SIZES[:n_arrivals])
        ]
        dynamics = make_dynamics()

        # Refill and receive both append the arrivals after the residents.
        template = Batch.from_data_list(residents + arrivals).to(device)
        new_state = dynamics._make_new_state(n_arrivals, template)
        dynamics._init_state(Batch.from_data_list(arrivals).to(device))
        expected = dynamics._state

        assert {key for key, _ in new_state} == {key for key, _ in expected}
        for key, value in expected:
            torch.testing.assert_close(
                getattr(new_state, key), value, msg=lambda m: f"{key}: {m}"
            )

    @_INTEGRATORS
    def test_arrival_state_does_not_alias_the_batch(self, make_dynamics):
        """In-place writes to the arrival state leave the batch untouched."""
        template = Batch.from_data_list(
            [
                _make_atomic_data(n, seed=i, with_cell=True)
                for i, n in enumerate(_RESIDENT_SIZES + _ARRIVAL_SIZES)
            ]
        )
        counts = template.num_nodes_per_graph.clone()
        new_state = make_dynamics()._make_new_state(len(_ARRIVAL_SIZES), template)

        for _, value in new_state:
            value.zero_()

        torch.testing.assert_close(template.num_nodes_per_graph, counts)

    @_INTEGRATORS
    def test_rejects_template_without_the_arrivals(self, make_dynamics):
        """A template with fewer graphs than arrivals raises ``ValueError``."""
        template = Batch.from_data_list(
            [
                _make_atomic_data(n, seed=i, with_cell=True)
                for i, n in enumerate(_RESIDENT_SIZES)
            ]
        )
        with pytest.raises(ValueError, match="fewer than"):
            make_dynamics()._make_new_state(3, template)


class _SizedDataset:
    """Systems with the given atom counts, as ``SizeAwareSampler`` reads them."""

    def __init__(self, sizes: tuple[int, ...]) -> None:
        self.sizes = sizes

    def __len__(self) -> int:
        return len(self.sizes)

    def __getitem__(self, idx: int) -> tuple[AtomicData, dict]:
        return _make_atomic_data(self.sizes[idx], seed=idx), {}

    def get_metadata(self, idx: int) -> tuple[int, int]:
        """Return ``(num_atoms, num_edges)`` for sample *idx*."""
        return self.sizes[idx], 0


class _NdofRecorder:
    """Record the thermostat's ndof next to the batch's atom counts."""

    stage = DynamicsStage.AFTER_STEP
    frequency = 1

    def __init__(self, dynamics: NVTNoseHoover) -> None:
        self.dynamics = dynamics
        self.records: list[tuple[torch.Tensor, torch.Tensor]] = []

    def __call__(self, ctx: DynamicsContext, stage: DynamicsStage) -> None:
        """Snapshot ``nhc_ndof`` and ``num_nodes_per_graph``."""
        self.records.append(
            (
                self.dynamics._state.nhc_ndof.clone(),
                ctx.batch.num_nodes_per_graph.clone(),
            )
        )


class TestAdmissionPath:
    """Systems admitted into a stage that has stepped get their own ndof."""

    def test_inflight_refill(self):
        """Mixed-size refills keep ``nhc_ndof == 3 * N`` on every step."""
        sampler = SizeAwareSampler(_SizedDataset((2, 9, 4, 12, 6)), max_batch_size=2)
        dynamics = _nose_hoover(n_steps=2)
        recorder = _NdofRecorder(dynamics)
        stage = FusedStage(
            sub_stages=[(0, dynamics)],
            sampler=sampler,
            refill_frequency=1,
            device_type="cpu",
        )
        stage.register_hook(recorder)
        batch = sampler.build_initial_batch()
        # Stagger the step budgets so systems graduate one at a time and every
        # arrival joins a resident that has already been stepped.
        batch["n_steps_counter_0"] = torch.arange(batch.num_graphs).unsqueeze(-1)

        with patch.object(
            dynamics, "_make_new_state", wraps=dynamics._make_new_state
        ) as make_new_state:
            assert stage.run(batch, n_steps=50) is None

        assert any(
            call.args[1].num_graphs > call.args[0]
            for call in make_new_state.call_args_list
        )
        assert recorder.records
        for ndof, counts in recorder.records:
            torch.testing.assert_close(ndof, 3.0 * counts.to(ndof.dtype))

    def test_pipeline_receive(self):
        """Received systems get the chains ``_init_state`` would give them."""
        dynamics = _nose_hoover()
        stage = FusedStage(sub_stages=[(0, dynamics)], device_type="cpu")
        stage.active_batch = Batch.from_data_list(
            [_make_atomic_data(n, seed=i) for i, n in enumerate(_RESIDENT_SIZES)]
        )
        # Step first: arrivals into a stage without state take _init_state.
        stage.step(stage.active_batch)

        stage._recv_to_batch(
            Batch.from_data_list(
                [
                    _make_atomic_data(n, seed=10 + i)
                    for i, n in enumerate(_ARRIVAL_SIZES)
                ]
            )
        )

        reference = _nose_hoover()
        reference._init_state(stage.active_batch)
        torch.testing.assert_close(dynamics._state.nhc_ndof, reference._state.nhc_ndof)
        torch.testing.assert_close(dynamics._state.nhc_Q, reference._state.nhc_Q)
