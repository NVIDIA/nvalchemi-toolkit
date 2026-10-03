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
"""Steady-state throughput measurement for a propagator driving a batch."""

from __future__ import annotations

import time
import warnings
from enum import Enum
from typing import TYPE_CHECKING

import torch

from nvalchemi._serialization import MeasurementRecord
from nvalchemi.dynamics.base import DynamicsStage
from nvalchemi.dynamics.hooks._utils import _FS_PER_NS

if TYPE_CHECKING:
    from nvalchemi.data import Batch
    from nvalchemi.dynamics.base import BaseDynamics
    from nvalchemi.hooks._context import DynamicsContext

__all__ = ["ThroughputMetrics", "measure_throughput"]

_SECONDS_PER_DAY = 86_400.0
"""Seconds in a day, for the ns/day figure MD papers quote."""


class ThroughputMetrics(MeasurementRecord):
    """Steady-state speed of one model driving one batch.

    Attributes
    ----------
    steps_per_second : float
        Propagator steps completed per wall-clock second.
    atoms_per_second : float
        ``steps_per_second`` times the atom count of the measured batch. It
        depends on batch size: it climbs with the batch until the device
        saturates. It therefore ranks two students only when both were measured
        on the same batch.
    ns_per_day : float | None
        Simulated nanoseconds per wall-clock day. ``None`` when no timestep was
        supplied.
    num_atoms : int
        Atoms in the batch at the start of the measured window.
    num_graphs : int
        Graphs in the batch at the start of the measured window.
    warmup_steps, measured_steps : int
        Warmup steps run and discarded before timing, and steps actually
        executed inside the timed window. ``measured_steps`` is read from the
        propagator's own step counter.
    elapsed_seconds : float
        Wall-clock duration of the measured window.
    device : str
        Device the batch was propagated on.
    """

    steps_per_second: float
    atoms_per_second: float
    ns_per_day: float | None
    num_atoms: int
    num_graphs: int
    warmup_steps: int
    measured_steps: int
    elapsed_seconds: float
    device: str


def measure_throughput(
    dynamics: BaseDynamics,
    batch: Batch,
    *,
    warmup_steps: int = 5,
    measured_steps: int = 20,
    timestep_fs: float | None = None,
) -> ThroughputMetrics:
    """Time a propagator at steady state and report atoms/s and ns/day.

    The measurement makes one
    :meth:`~nvalchemi.dynamics.base.BaseDynamics.run` call of
    ``warmup_steps + measured_steps`` steps on the live batch and starts the
    clock from inside it, at the first step past the warmup. The warmup
    absorbs the cost of the neighbor-list build, lazy state allocation,
    autotuning, and kernel compilation, and also the admission hooks and the
    force priming a run performs once before its first step. A second
    ``run`` call would perform those again, since every run is a fresh
    admission, and the timed window would pay one model evaluation more than
    the steps it counts. The device is synchronized before the clock starts
    and before it stops, so the rate is the execution rate rather than the
    launch rate. The rate uses the change in the propagator's own
    ``step_count``, because a run stops early once every graph has converged.
    The atom count is read at the start of the window, so a propagator that
    graduates systems mid-window reports the rate it started with.

    Parameters
    ----------
    dynamics : BaseDynamics
        Propagator to time, with its model and hooks already attached.
    batch : Batch
        Batch to propagate. Advanced in place by up to ``warmup_steps +
        measured_steps`` steps, fewer if the propagator converges first.
    warmup_steps : int, optional
        Steps run and discarded before timing. Even at ``0`` the window opens
        after the run's admission and force priming. Default ``5``.
    measured_steps : int, optional
        Steps requested inside the timed window. Default ``20``.
    timestep_fs : float | None, optional
        Integration timestep in femtoseconds, used to convert steps into
        simulated time. Pass the value the propagator was built with. Default
        ``None`` (``ns_per_day`` is left unreported).

    Returns
    -------
    ThroughputMetrics
        Rates measured over the timed window.

    Raises
    ------
    ValueError
        If ``measured_steps`` is not positive, if ``warmup_steps`` is negative,
        if ``timestep_fs`` is not positive, or if the propagator advanced no
        steps inside the window, as one that converged during the warmup does.
    RuntimeError
        If the propagator exhausted its sampler during the warmup, leaving no
        batch to time.

    Warns
    -----
    UserWarning
        If the propagator converged inside the timed window, so the rate covers
        fewer steps than requested and is not a steady-state figure.

    Examples
    --------
    >>> from nvalchemi.dynamics import measure_throughput
    >>> speed = measure_throughput(  # doctest: +SKIP
    ...     NVE(model=student, dt=1.0),
    ...     batch,
    ...     measured_steps=50,
    ...     timestep_fs=1.0,
    ... )
    >>> speed.ns_per_day  # doctest: +SKIP
    12.4
    """
    if measured_steps <= 0 or warmup_steps < 0:
        raise ValueError(
            "measured_steps must be positive and warmup_steps non-negative; got "
            f"measured_steps={measured_steps!r}, warmup_steps={warmup_steps!r}."
        )
    if timestep_fs is not None and timestep_fs <= 0.0:
        raise ValueError(f"timestep_fs must be positive; got {timestep_fs!r}.")
    clock = _WindowClock(dynamics.step_count + warmup_steps)
    dynamics.register_hook(clock)
    try:
        state = dynamics.run(batch, n_steps=warmup_steps + measured_steps)
    finally:
        dynamics.hooks.remove(clock)
    if clock.started is None and state is None:
        raise RuntimeError(
            "The propagator exhausted its sampler during the warmup, so no batch "
            f"survived to time; got warmup_steps={warmup_steps!r}. Shorten the "
            "warmup or hand the measurement a sampler-free propagator."
        )
    executed = 0 if clock.started is None else dynamics.step_count - clock.first_count
    if executed <= 0:
        raise ValueError(
            "The propagator advanced no steps inside the timed window, so there "
            f"is no rate to report; got measured_steps={measured_steps!r}. A "
            "propagator that has already converged has to be reset, or measured "
            "without its convergence hook."
        )
    _synchronize(clock.device)
    elapsed = time.perf_counter() - clock.started
    num_atoms = clock.num_atoms
    num_graphs = clock.num_graphs
    device = clock.device
    if executed < measured_steps:
        warnings.warn(
            f"The propagator converged after {executed} of the {measured_steps} "
            "requested steps, so the reported rate covers that shorter window "
            "and is not a steady-state measurement. Measure a propagator "
            "without a convergence hook, or start from a structure further from "
            "its minimum.",
            UserWarning,
            stacklevel=2,
        )
    steps_per_second = executed / elapsed
    return ThroughputMetrics(
        steps_per_second=steps_per_second,
        atoms_per_second=steps_per_second * num_atoms,
        ns_per_day=(
            None
            if timestep_fs is None
            else steps_per_second * timestep_fs * _SECONDS_PER_DAY / _FS_PER_NS
        ),
        num_atoms=num_atoms,
        num_graphs=num_graphs,
        warmup_steps=warmup_steps,
        measured_steps=executed,
        elapsed_seconds=elapsed,
        device=str(device),
    )


class _WindowClock:
    """Hook that opens the timed window at the first step past the warmup.

    Registered at ``BEFORE_STEP``, it fires after the admission and force
    priming the run's first step performs, drains the device, and reads the
    clock and the batch's size once. Nothing fires if the run ends inside the
    warmup, which the caller reads off ``started``.
    """

    frequency = 1
    stage = DynamicsStage.BEFORE_STEP

    def __init__(self, first_step: int) -> None:
        self.first_step = first_step
        self.started: float | None = None
        self.first_count = 0
        self.num_atoms = 0
        self.num_graphs = 0
        self.device = torch.device("cpu")

    def __call__(self, ctx: DynamicsContext, stage: Enum) -> None:  # noqa: ARG002
        """Start the clock once the warmup steps have elapsed."""
        if self.started is not None or ctx.step_count < self.first_step:
            return
        self.first_count = ctx.step_count
        self.num_atoms = ctx.batch.num_nodes
        self.num_graphs = ctx.batch.num_graphs
        self.device = ctx.batch.device
        _synchronize(self.device)
        self.started = time.perf_counter()


def _synchronize(device: torch.device) -> None:
    """Wait for queued work on *device* so the clock brackets real execution."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
