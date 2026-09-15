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
"""Tests for :mod:`nvalchemi.training.distillation.evaluation.throughput`."""

from __future__ import annotations

import time
import warnings
from typing import Any

import pytest
import torch

from nvalchemi.data import Batch
from nvalchemi.dynamics.base import ConvergenceHook, DynamicsStage
from nvalchemi.dynamics.integrators import NVE
from nvalchemi.dynamics.optimizers.fire import FIRE
from nvalchemi.hooks.neighbor_list import NeighborListHook
from nvalchemi.training.distillation.evaluation import measure_throughput
from test.training.distillation.conftest import (
    _build_lattice_batch,
    _build_lj_teacher,
)

_LATTICE_ATOMS = 27
"""Atom count of the default 3x3x3 lattice."""

_STALL_CYCLES = 50_000_000
"""Device cycles one stalled step queues, tens of milliseconds on a current GPU."""

_STALL_TRIALS = 3
"""Timings the calibration takes the fastest of."""


def _calibrated_stall_seconds() -> float:
    """Return the wall-clock cost of one queued device stall on this GPU.

    The stall is a fixed number of device cycles, so how long it takes is set
    by whatever clock the GPU is running at, and every bound below is written
    as a multiple of it rather than as an absolute duration. The fastest of a
    few timings is the one to keep: the host sits inside ``synchronize`` for
    the whole stall, so a busy machine can preempt it and inflate any single
    measurement, but nothing can make one finish early.
    """
    torch.cuda.synchronize()
    timings = []
    for _ in range(_STALL_TRIALS):
        started = time.perf_counter()
        torch.cuda._sleep(_STALL_CYCLES)
        torch.cuda.synchronize()
        timings.append(time.perf_counter() - started)
    return min(timings)


def _make_nve(convergence_hook: Any = None) -> NVE:
    """Return an NVE integrator over the Lennard-Jones teacher."""
    model = _build_lj_teacher()
    return NVE(
        model=model,
        dt=1.0,
        convergence_hook=convergence_hook,
        hooks=[
            NeighborListHook(
                config=model.model_config.neighbor_config,
                skin=1.0,
                stage=DynamicsStage.BEFORE_COMPUTE,
            )
        ],
    )


def _make_relaxer() -> FIRE:
    """Return the house relaxation recipe: FIRE under an fmax convergence hook."""
    model = _build_lj_teacher()
    return FIRE(
        model=model,
        dt=0.1,
        convergence_hook=ConvergenceHook.from_fmax(0.05),
        hooks=[
            NeighborListHook(
                config=model.model_config.neighbor_config,
                skin=1.0,
                stage=DynamicsStage.BEFORE_COMPUTE,
            )
        ],
    )


class _DeviceStallDynamics:
    """Propagator whose step only queues device work and never syncs the host."""

    def __init__(self, cycles: int = _STALL_CYCLES) -> None:
        self.cycles = cycles
        self.step_count = 0

    def run(self, batch: Batch, n_steps: int) -> Batch:
        """Queue one long device busy-wait per requested step and return the batch."""
        for _ in range(n_steps):
            torch.cuda._sleep(self.cycles)
            self.step_count += 1
        return batch


class _StalledDynamics:
    """Propagator that returns without advancing, as an exhausted stage would."""

    step_count = 0

    def run(self, batch: Batch, n_steps: int) -> Batch:  # noqa: ARG002
        """Return the batch exactly as it arrived."""
        return batch


class _ExhaustedDynamics:
    """Propagator whose sampler runs dry, so a run returns no batch at all."""

    step_count = 0

    def run(self, batch: Batch, n_steps: int) -> None:  # noqa: ARG002
        """Return nothing, the way a refill past the end of a sampler does."""
        return None


class TestMeasureThroughput:
    """Steady-state rates of a propagator over a fixed batch."""

    def test_rates_are_consistent_with_the_measured_window(self) -> None:
        """Every reported rate is the step rate rescaled by a known constant."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            speed = measure_throughput(
                _make_nve(),
                _build_lattice_batch(),
                warmup_steps=2,
                measured_steps=5,
                timestep_fs=2.0,
            )
        assert caught == []
        assert speed.steps_per_second * speed.elapsed_seconds == pytest.approx(5.0)
        assert speed.atoms_per_second == pytest.approx(
            speed.steps_per_second * _LATTICE_ATOMS
        )
        assert speed.ns_per_day == pytest.approx(
            speed.steps_per_second * 2.0 * 86_400.0 / 1.0e6
        )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_the_timed_window_waits_for_queued_device_work(self) -> None:
        """The clock stops once the device is done, not once the launches return."""
        stall = _calibrated_stall_seconds()
        speed = measure_throughput(
            _DeviceStallDynamics(),
            _build_lattice_batch().to(torch.device("cuda")),
            warmup_steps=2,
            measured_steps=5,
        )
        assert speed.device.startswith("cuda")
        assert speed.elapsed_seconds > 4.0 * stall

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_the_timed_window_excludes_work_queued_by_the_warmup(self) -> None:
        """The warmup's queued work is drained before the clock starts."""
        stall = _calibrated_stall_seconds()
        speed = measure_throughput(
            _DeviceStallDynamics(),
            _build_lattice_batch().to(torch.device("cuda")),
            warmup_steps=10,
            measured_steps=2,
        )
        assert 1.6 * stall < speed.elapsed_seconds < 6.0 * stall

    def test_reported_sizes_describe_the_measured_batch(self) -> None:
        """Atom and graph counts are read at the start of the timed window."""
        speed = measure_throughput(
            _make_nve(), _build_lattice_batch(), warmup_steps=1, measured_steps=2
        )
        assert speed.num_atoms == _LATTICE_ATOMS
        assert speed.num_graphs == 1
        assert speed.device == "cpu"

    def test_ns_per_day_is_omitted_without_a_timestep(self) -> None:
        """A step has no physical duration until a timestep says what it is."""
        speed = measure_throughput(
            _make_nve(), _build_lattice_batch(), warmup_steps=0, measured_steps=2
        )
        assert speed.ns_per_day is None
        assert speed.warmup_steps == 0

    def test_warmup_and_measured_steps_both_advance_the_propagator(self) -> None:
        """The batch really is propagated for the whole warmup plus window."""
        dynamics = _make_nve()
        measure_throughput(
            dynamics, _build_lattice_batch(), warmup_steps=3, measured_steps=4
        )
        assert dynamics.step_count == 7

    def test_a_converging_propagator_is_timed_over_the_steps_it_ran(self) -> None:
        """A window cut short by convergence reports the shorter window, and warns."""
        dynamics = _make_nve(ConvergenceHook.from_fmax(1.0e6))
        with pytest.warns(UserWarning, match="converged after 1 of the 5"):
            speed = measure_throughput(
                dynamics, _build_lattice_batch(), warmup_steps=0, measured_steps=5
            )
        assert speed.measured_steps == 1
        assert dynamics.step_count == 1
        assert speed.steps_per_second * speed.elapsed_seconds == pytest.approx(1.0)

    def test_a_relaxation_reaching_its_minimum_mid_window_is_not_extrapolated(
        self,
    ) -> None:
        """The rate covers the relaxation's own length, not the length requested."""
        dynamics = _make_relaxer()
        with pytest.warns(UserWarning, match="not a steady-state measurement"):
            speed = measure_throughput(
                dynamics,
                _build_lattice_batch(jitter=0.2),
                warmup_steps=0,
                measured_steps=500,
                timestep_fs=1.0,
            )
        assert 1 < speed.measured_steps < 500
        assert dynamics.step_count == speed.measured_steps
        assert speed.atoms_per_second == pytest.approx(
            speed.measured_steps * _LATTICE_ATOMS / speed.elapsed_seconds
        )

    def test_a_propagator_that_never_advances_cannot_be_timed(self) -> None:
        """No executed step means no rate, so the measurement raises."""
        with pytest.raises(ValueError, match="advanced no steps"):
            measure_throughput(
                _StalledDynamics(), _build_lattice_batch(), warmup_steps=0
            )

    def test_a_warmup_that_consumes_the_batch_is_reported(self) -> None:
        """A sampler exhausted during the warmup leaves nothing to measure."""
        with pytest.raises(RuntimeError, match="exhausted its sampler"):
            measure_throughput(
                _ExhaustedDynamics(), _build_lattice_batch(), warmup_steps=1
            )

    @pytest.mark.parametrize(
        ("warmup_steps", "measured_steps"),
        [(2, 0), (-1, 5)],
        ids=["no-measured-steps", "negative-warmup"],
    )
    def test_invalid_step_counts_are_rejected(
        self, warmup_steps: int, measured_steps: int
    ) -> None:
        """A window with no steps, or a negative warmup, raises."""
        with pytest.raises(ValueError, match="measured_steps"):
            measure_throughput(
                _make_nve(),
                _build_lattice_batch(),
                warmup_steps=warmup_steps,
                measured_steps=measured_steps,
            )

    def test_non_positive_timestep_is_rejected(self) -> None:
        """A zero timestep would report an infinite simulated rate."""
        with pytest.raises(ValueError, match="timestep_fs"):
            measure_throughput(
                _make_nve(), _build_lattice_batch(), measured_steps=2, timestep_fs=0.0
            )
