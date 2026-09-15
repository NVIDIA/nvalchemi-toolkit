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
"""Steady-state throughput measurement for a model driving dynamics."""

from __future__ import annotations

import dataclasses
import time
import warnings
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import torch

from nvalchemi.training.distillation.evaluation._export import _rebuild

if TYPE_CHECKING:
    from nvalchemi.data import Batch
    from nvalchemi.dynamics.base import BaseDynamics

__all__ = ["ThroughputMetrics", "measure_throughput"]

_SECONDS_PER_DAY = 86_400.0
"""Seconds in a day, for the ns/day figure MD papers quote."""

_FS_PER_NS = 1.0e6
"""Femtoseconds in a nanosecond."""


@dataclasses.dataclass(frozen=True)
class ThroughputMetrics:
    """Steady-state speed of one model driving one batch.

    Attributes
    ----------
    steps_per_second : float
        Propagator steps completed per wall-clock second.
    atoms_per_second : float
        ``steps_per_second`` times the atom count of the measured batch. It is
        not a size-independent figure: on a device the batch does not saturate
        the rate climbs steeply with the batch, so it ranks two students only
        when both were measured on the same one.
    ns_per_day : float | None
        Simulated nanoseconds per wall-clock day. ``None`` when no timestep was
        supplied, since a step has no physical duration without one.
    num_atoms : int
        Atoms in the batch at the start of the measured window.
    num_graphs : int
        Graphs in the batch at the start of the measured window.
    warmup_steps, measured_steps : int
        Steps discarded and steps actually executed inside the timed window,
        the latter read from the propagator's own counter rather than from the
        request, since a converging propagator stops early.
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

    def to_dict(self) -> dict[str, Any]:
        """Return every field as a plain dictionary."""
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ThroughputMetrics:
        """Rebuild the metrics from a :meth:`to_dict` export."""
        return _rebuild(cls, data)


def measure_throughput(
    dynamics: BaseDynamics,
    batch: Batch,
    *,
    warmup_steps: int = 5,
    measured_steps: int = 20,
    timestep_fs: float | None = None,
) -> ThroughputMetrics:
    """Time a propagator at steady state and report atoms/s and ns/day.

    The measurement is two chunked :meth:`~nvalchemi.dynamics.base.BaseDynamics.run`
    calls on the same live batch: a warmup that is thrown away, then a timed
    window. Discarding the warmup is what makes the number steady-state — the
    first steps of a run pay for the neighbor-list build, lazy state
    allocation, autotuning, and any kernel compilation, none of which recur.

    Timing is honest about the device: CUDA work is launched asynchronously, so
    the device is synchronized both before the clock starts and before it
    stops. Without the second synchronization the measurement would report the
    launch rate rather than the execution rate. It is equally honest about the
    step count: a run stops early once every graph has converged, so the rate
    is formed from the propagator's own ``step_count`` delta rather than from
    the number of steps that were asked for.

    Parameters
    ----------
    dynamics : BaseDynamics
        Propagator to time, with its model and hooks already attached.
    batch : Batch
        Batch to propagate. Advanced in place by up to ``warmup_steps +
        measured_steps`` steps, fewer if the propagator converges first.
    warmup_steps : int, optional
        Steps run and discarded before timing. Default ``5``.
    measured_steps : int, optional
        Steps requested inside the timed window. Default ``20``.
    timestep_fs : float | None, optional
        Integration timestep in femtoseconds, which is what converts steps into
        simulated time. Pass the same value the propagator was built with.
        Default ``None`` (``ns_per_day`` is left unreported rather than
        guessed).

    Returns
    -------
    ThroughputMetrics
        Rates measured over the timed window.

    Raises
    ------
    ValueError
        If ``measured_steps`` is not positive, if ``warmup_steps`` is negative,
        or if ``timestep_fs`` is not positive.
    RuntimeError
        If the propagator exhausted its sampler during the warmup, leaving no
        batch to time.

    Warns
    -----
    UserWarning
        If the propagator converged inside the timed window, so the rate covers
        fewer steps than were requested and is not a steady-state figure.

    Examples
    --------
    >>> from nvalchemi.training.distillation.evaluation import measure_throughput
    >>> speed = measure_throughput(  # doctest: +SKIP
    ...     NVE(model=student, dt=1.0),
    ...     batch,
    ...     measured_steps=50,
    ...     timestep_fs=1.0,
    ... )
    >>> speed.ns_per_day  # doctest: +SKIP
    12.4

    Notes
    -----
    The atom count is read at the start of the timed window. A propagator that
    graduates systems mid-window therefore reports the rate it started with,
    which is the intended reading for an acceptance gate: measure a fixed batch
    over a fixed number of steps rather than a shrinking one. Convergence is a
    different matter — a relaxer carrying a
    :class:`~nvalchemi.dynamics.base.ConvergenceHook` leaves the loop as soon
    as every graph is converged, and a window that ends there has measured a
    shorter run than it asked for rather than a faster one, so it is reported
    for what it is and warned about.
    """
    if measured_steps <= 0 or warmup_steps < 0:
        raise ValueError(
            "measured_steps must be positive and warmup_steps non-negative; got "
            f"measured_steps={measured_steps!r}, warmup_steps={warmup_steps!r}."
        )
    if timestep_fs is not None and timestep_fs <= 0.0:
        raise ValueError(f"timestep_fs must be positive; got {timestep_fs!r}.")
    state = batch
    if warmup_steps > 0:
        state = dynamics.run(state, n_steps=warmup_steps)
    if state is None:
        raise RuntimeError(
            "The propagator exhausted its sampler during the warmup, so no batch "
            f"survived to time; got warmup_steps={warmup_steps!r}. Shorten the "
            "warmup or hand the measurement a sampler-free propagator."
        )
    num_atoms = state.num_nodes
    num_graphs = state.num_graphs
    device = state.device
    _synchronize(device)
    started = time.perf_counter()
    before = dynamics.step_count
    dynamics.run(state, n_steps=measured_steps)
    _synchronize(device)
    elapsed = time.perf_counter() - started
    executed = dynamics.step_count - before
    if executed <= 0:
        raise ValueError(
            "The propagator advanced no steps inside the timed window, so there "
            f"is no rate to report; got measured_steps={measured_steps!r}. A "
            "propagator that has already converged has to be reset, or measured "
            "without its convergence hook."
        )
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


def _synchronize(device: torch.device) -> None:
    """Wait for queued work on *device* so the clock brackets real execution."""
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)
