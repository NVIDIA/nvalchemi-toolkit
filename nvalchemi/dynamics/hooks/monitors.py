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
Diagnostic monitor hooks for long-running simulations.

Provides :class:`EnergyDriftMonitorHook`, which tracks cumulative
energy drift over time and can warn or halt the simulation if the
drift exceeds a configurable threshold, and :class:`StabilityMonitor`,
which records the whole energy and momentum series of a run and reports
its conservation metrics once the run is over.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from enum import Enum
from typing import TYPE_CHECKING, Literal

import torch
from loguru import logger

from nvalchemi._serialization import MeasurementRecord
from nvalchemi.data import Batch
from nvalchemi.dynamics.base import DynamicsStage
from nvalchemi.dynamics.hooks._utils import _FS_PER_NS, kinetic_energy_per_graph
from nvalchemi.dynamics.hooks.safety import nonfinite_graph_mask
from nvalchemi.hooks._context import DynamicsContext

if TYPE_CHECKING:
    from jaxtyping import Bool, Float

__all__ = [
    "EnergyDriftMonitorHook",
    "StabilityMetrics",
    "StabilityMonitor",
    "total_momentum",
]

_SAMPLED_FIELDS = ("energy", "velocities", "atomic_masses")
"""Batch fields every recorded stability sample is formed from."""

_AGGREGATES = ("max", "mean")
"""Reductions across graphs a stability figure can be reported with."""


class EnergyDriftMonitorHook:
    """Track energy drift and warn or stop if it exceeds a threshold.

    In a well-behaved NVE (microcanonical) simulation with a symplectic
    integrator, the total energy should be conserved to within numerical
    precision.  Significant energy drift indicates problems with:

    * The integration timestep (too large for the force magnitudes).
    * The ML potential (non-smooth or discontinuous energy surface).
    * Numerical precision (single vs. double precision accumulation).
    * Force clamping or other hook-induced modifications breaking
      energy conservation.

    This hook monitors the **total energy** (potential + kinetic) over
    the simulation and computes drift metrics.  It supports two modes:

    **Absolute drift mode** (``metric="absolute"``)
        Tracks ``|E(t) - E(0)|``, the absolute deviation from the
        initial total energy.  Suitable for NVE validation runs.

    **Per-atom-per-step drift mode** (``metric="per_atom_per_step"``)
        Tracks ``|E(t) - E(0)| / (N_atoms * step_count)``, a
        normalized metric that allows comparison across systems of
        different size and simulation length.  This is the standard
        metric reported in ML potential benchmarks.

    When the drift exceeds ``threshold``, the hook either emits a
    warning or raises a :class:`RuntimeError`, controlled by the
    ``action`` parameter.

    The hook records the reference energy on the first firing and
    computes drift on all subsequent firings.  For NVT or NPT
    simulations, energy drift is expected (the thermostat/barostat
    injects or removes energy), so use this hook primarily for NVE
    validation.

    Parameters
    ----------
    threshold : float
        Maximum acceptable drift before triggering the ``action``.
        Units depend on ``metric``: eV for ``"absolute"``,
        eV/atom/step for ``"per_atom_per_step"``.
    metric : {"absolute", "per_atom_per_step"}, optional
        Drift metric to use. Default ``"per_atom_per_step"``.
    action : {"warn", "raise"}, optional
        What to do when the threshold is exceeded. ``"warn"`` emits a
        :mod:`loguru` warning; ``"raise"`` raises a
        :class:`RuntimeError`. Default ``"warn"``.
    frequency : int, optional
        Evaluate drift every ``frequency`` steps. Default ``1``.
    include_kinetic : bool, optional
        Whether to include kinetic energy in the total energy
        calculation. Set to ``False`` if only monitoring potential
        energy drift (e.g. for optimizers). Default ``True``.

    Attributes
    ----------
    threshold : float
        Drift threshold.
    metric : str
        Drift metric mode.
    action : str
        Threshold violation behavior.
    include_kinetic : bool
        Whether kinetic energy is included.
    frequency : int
        Evaluation frequency in steps.
    stage : DynamicsStage
        Fixed to ``AFTER_STEP``.

    Examples
    --------
    NVE validation with strict drift tolerance:

    >>> from nvalchemi.dynamics.hooks import EnergyDriftMonitorHook
    >>> hook = EnergyDriftMonitorHook(
    ...     threshold=1e-5,
    ...     metric="per_atom_per_step",
    ...     action="raise",
    ...     frequency=100,
    ... )
    >>> dynamics = DemoDynamics(model=model, n_steps=10_000, dt=0.5, hooks=[hook])
    >>> dynamics.run(batch)

    Soft monitoring during production:

    >>> hook = EnergyDriftMonitorHook(
    ...     threshold=1e-3,
    ...     action="warn",
    ...     frequency=1000,
    ... )

    Notes
    -----
    * The reference energy is captured on the **first** hook firing
      (step 0 by default), not at construction time.  This allows the
      hook to be registered before the batch is available.
    * For batched simulations, drift is computed **per graph** and the
      maximum drift across all graphs is compared to the threshold.
    * Status-filtered and inflight dynamics are not yet supported. Their
      reference energies and elapsed steps must be tracked per system rather
      than by batch position.
    """

    def __init__(
        self,
        threshold: float,
        metric: Literal["absolute", "per_atom_per_step"] = "per_atom_per_step",
        action: Literal["warn", "raise"] = "warn",
        frequency: int = 1,
        include_kinetic: bool = True,
        stage: Enum = DynamicsStage.AFTER_STEP,
    ) -> None:
        self.frequency = frequency
        self.stage = stage
        self.threshold = threshold
        self.metric = metric
        self.action = action
        self.include_kinetic = include_kinetic
        self._reference_total_energy: torch.Tensor | None = None

    @torch.compiler.disable
    def _check_drift(self, batch: Batch, step_count: int, global_rank: int) -> None:
        """Compute energy drift and compare against the threshold.

        On the first firing, this method captures the reference total
        energy and returns immediately.  On all subsequent firings, it
        computes drift relative to that reference and compares against
        the configured threshold.

        Parameters
        ----------
        batch : Batch
            The current batch of atomic data.  Must have ``energy``
            (and ``velocities`` if ``include_kinetic=True``).
        step_count : int
            The current step number.
        global_rank : int
            The distributed rank of this process.

        Raises
        ------
        RuntimeError
            If ``action="raise"`` and drift exceeds the threshold.
        """
        energy = batch.energy.squeeze(-1)  # (B,)

        if self.include_kinetic and getattr(batch, "velocities", None) is not None:
            ke = kinetic_energy_per_graph(
                batch.velocities,
                batch.atomic_masses,
                batch.batch_idx,
                batch.num_graphs,
            ).squeeze(-1)  # (B,)
            total = energy + ke
        else:
            total = energy

        # Capture reference on first firing
        if self._reference_total_energy is None:
            self._reference_total_energy = total.clone()
            return

        drift = (total - self._reference_total_energy).abs()  # (B,)

        if self.metric == "per_atom_per_step":
            effective_step = max(step_count, 1)
            drift = drift / (batch.num_nodes_per_graph * effective_step)

        max_drift = drift.max().item()

        if max_drift > self.threshold:
            msg = (
                f"Energy drift {max_drift:.2e} exceeds threshold "
                f"{self.threshold:.2e} at step {step_count}"
                f" on rank {global_rank}."
            )
            if self.action == "raise":
                raise RuntimeError(msg)
            else:
                # TODO: use a distributed aware logger
                logger.warning(msg)

    def __call__(self, ctx: DynamicsContext, stage: Enum) -> None:
        """Check energy drift against the configured threshold."""
        if ctx.active_graph_mask is not None:
            raise NotImplementedError(
                "EnergyDriftMonitorHook does not yet support status-filtered "
                "or inflight dynamics because reference energies and elapsed "
                "steps must be tracked per system."
            )
        self._check_drift(ctx.batch, ctx.step_count, ctx.global_rank or 0)


def total_momentum(batch: Batch) -> Float[torch.Tensor, "G 3"]:
    """Return the total linear momentum of each graph.

    Parameters
    ----------
    batch : Batch
        Batch carrying ``velocities`` and ``atomic_masses``.

    Returns
    -------
    Float[torch.Tensor, "G 3"]
        Mass-weighted velocity sum per graph, in the batch's own units.

    Examples
    --------
    >>> from nvalchemi.dynamics.hooks import total_momentum
    >>> total_momentum(batch).shape  # doctest: +SKIP
    torch.Size([2, 3])
    """
    momentum = batch.atomic_masses.unsqueeze(-1) * batch.velocities
    total = momentum.new_zeros(batch.num_graphs, 3)
    return total.index_add_(0, batch.batch_idx, momentum)


class StabilityMetrics(MeasurementRecord):
    """Conservation diagnostics of one trajectory.

    By default, every figure is reported for the worst graph in the batch,
    matching :class:`EnergyDriftMonitorHook`. When the monitor was built with
    ``aggregate="mean"``, each is the mean over graphs instead. The
    per-nanosecond rate is the slope of a least-squares fit through every
    sample, not an endpoint difference, so a noisy series is not scored on
    whichever two samples bracket it. Because it is a slope, the rate reads
    zero for an excursion that is symmetric about the window. Read it together
    with ``energy_fluctuation_per_atom``, the RMS residual about the same fit,
    and with ``max_energy_excursion_per_atom``. A drift no larger than the
    fluctuation is a line fitted through an oscillation. Both drift figures
    include whatever transient the series starts with. A frame that is not
    equilibrated under the propagated potential relaxes during the run, and
    the fit reports that relaxation as drift. Give :class:`StabilityMonitor` a
    ``warmup_steps`` window that covers the relaxation.

    Attributes
    ----------
    num_samples : int
        Recorded samples, after any discarded warmup.
    first_step, last_step : int
        Step counts of the first and last sample.
    energy_drift_per_atom : float
        ``|E(t_end) - E(t_0)| / N`` of the worst graph, or the mean over graphs
        under ``aggregate="mean"``.
    energy_drift_per_atom_per_step : float
        The same figure divided by the elapsed steps.
    energy_drift_per_atom_per_ns : float | None
        Fitted drift rate in eV/atom/ns of the worst graph, or the mean over
        graphs under ``aggregate="mean"``. ``None`` when the monitor was given
        no timestep.
    max_momentum_drift : float
        Largest deviation of a graph's total momentum from its initial value,
        for the worst graph or as the mean over graphs under
        ``aggregate="mean"``.
    timestep_fs : float | None
        Timestep the rates were derived with.
    energy_fluctuation_per_atom : float | None
        RMS residual of a graph's per-atom energy about the fitted line, for the
        worst graph or as the mean over graphs under ``aggregate="mean"``.
        ``None`` only when rebuilt from an export written before this field
        existed.
    max_energy_excursion_per_atom : float | None
        Largest ``|E(t) - E(t_0)| / N`` a graph reached, for the worst graph or
        as the mean over graphs under ``aggregate="mean"``. ``None`` only when
        rebuilt from an older export.
    first_divergence_step : int | None
        Step count of the first firing at which the monitor's divergence
        predicate flagged a graph. The series stopped at that step. ``None``
        when no graph diverged.
    aggregate : Literal["max", "mean"]
        Reduction across graphs used to form the drift, fluctuation,
        excursion, and momentum figures.
    """

    num_samples: int
    first_step: int
    last_step: int
    energy_drift_per_atom: float
    energy_drift_per_atom_per_step: float
    energy_drift_per_atom_per_ns: float | None
    max_momentum_drift: float
    timestep_fs: float | None
    energy_fluctuation_per_atom: float | None = None
    max_energy_excursion_per_atom: float | None = None
    first_divergence_step: int | None = None
    aggregate: Literal["max", "mean"] = "max"


def _composition(batch: Batch, counts: torch.Tensor) -> torch.Tensor:
    """Return the signature a per-graph series has to keep to stay comparable.

    Atom counts alone miss an inflight refill that replaces graduated systems
    with new systems of the same size. The signature therefore includes
    ``system_id`` when the batch carries it.
    """
    identity = getattr(batch, "system_id", None)
    if identity is None:
        return counts
    return torch.cat([counts, identity.detach().reshape(-1).to("cpu", torch.float64)])


class StabilityMonitor:
    """Dynamics hook recording energy and momentum along a trajectory.

    A *stability monitor* records the total energy and momentum of a
    trajectory and reports how well the run conserves them. Register it on a
    :class:`~nvalchemi.dynamics.base.BaseDynamics` run like any observation
    hook, then call :meth:`metrics` once the run is over. Unlike
    :class:`EnergyDriftMonitorHook`, which compares one live value against a
    threshold, this hook keeps the whole series on the host as float64, one
    small tensor per firing. For a long run, raise ``frequency`` rather than
    record every step.

    Parameters
    ----------
    frequency : int, optional
        Record every ``frequency`` steps. The dynamics hook registry applies
        this gating. Default ``1``.
    stage : Enum, optional
        Stage to record at. Default
        :attr:`~nvalchemi.dynamics.base.DynamicsStage.AFTER_STEP`.
    timestep_fs : float | None, optional
        Integration timestep in femtoseconds, which turns per-step drift into a
        per-nanosecond rate. Default ``None``.
    include_kinetic : bool, optional
        Add the kinetic energy to the potential energy before measuring drift.
        This is what makes the metric meaningful for NVE. Default ``True``.
    warmup_steps : int, optional
        Discard every firing before this step count, as read from the
        propagator's own step counter. A run seeded from frames that are not
        equilibria of the propagated potential needs this equilibration
        window, because a fit that includes the relaxation reports it as
        drift. Default ``0``.
    divergence : Callable[[Batch], Bool[Tensor, "G"]] | None, optional
        Predicate that flags the graphs that have diverged, evaluated at every
        firing after the warmup. The first firing that flags any graph is
        recorded as ``first_divergence_step`` and stops the series, so the
        metrics describe the trajectory up to the divergence. Default ``None``
        (:func:`~nvalchemi.dynamics.hooks.nonfinite_graph_mask`, which flags
        a non-finite position or force).
    aggregate : {"max", "mean"}, optional
        Reduction across graphs for the drift, fluctuation, excursion, and
        momentum figures: the worst graph, or the mean over graphs. Default
        ``"max"``.
    stop_on_composition_change : bool, optional
        Stop recording with a warning when the ``system_id`` in a slot changes
        while the batch keeps its shape. That is what an inflight refill with
        an equal-size system looks like. ``False`` keeps recording through
        such a refill. A change in the batch's graph count or per-graph atom
        counts always stops the series, because the per-graph arrays could no
        longer be stacked. Default ``True``.

    Raises
    ------
    ValueError
        If ``aggregate`` is not one of the two reductions.

    Examples
    --------
    >>> from nvalchemi.dynamics.hooks import StabilityMonitor
    >>> monitor = StabilityMonitor(frequency=10, timestep_fs=1.0, warmup_steps=100)
    >>> dynamics.register_hook(monitor)  # doctest: +SKIP
    >>> dynamics.run(batch)  # doctest: +SKIP
    >>> monitor.metrics().energy_drift_per_atom_per_ns  # doctest: +SKIP
    0.0042

    Notes
    -----
    Every sample is formed from the batch's own ``energy``, ``velocities``, and
    ``atomic_masses``. A batch built from geometry alone gains its ``energy``
    on the propagator's first
    :meth:`~nvalchemi.dynamics.base.BaseDynamics.compute`, which allocates the
    output fields the batch arrived without, so a firing inside a run always
    finds one. A batch fired outside a run has no model outputs and is refused.

    Momentum is conserved only by an integrator that conserves it. Under a
    stochastic thermostat, ``max_momentum_drift`` describes the bath, and no
    acceptance bar should be set on it.

    Recording stops with a warning as soon as the batch composition changes.
    A change means a different graph count, different per-graph atom counts,
    or a different ``system_id`` in the slots (unless
    ``stop_on_composition_change`` is off). A propagator that graduates
    systems mid-run is therefore scored on the segment before the first
    graduation. Like :class:`EnergyDriftMonitorHook`, the monitor does not yet
    support a status-filtered dispatch, whose per-graph series would have to
    be tracked per system.
    """

    def __init__(
        self,
        *,
        frequency: int = 1,
        stage: Enum = DynamicsStage.AFTER_STEP,
        timestep_fs: float | None = None,
        include_kinetic: bool = True,
        warmup_steps: int = 0,
        divergence: Callable[[Batch], Bool[torch.Tensor, "G"]] | None = None,
        aggregate: Literal["max", "mean"] = "max",
        stop_on_composition_change: bool = True,
    ) -> None:
        if aggregate not in _AGGREGATES:
            raise ValueError(
                f"aggregate must be one of {list(_AGGREGATES)!r}; got {aggregate!r}."
            )
        self.frequency = frequency
        self.stage = stage
        self.timestep_fs = timestep_fs
        self.include_kinetic = include_kinetic
        self.warmup_steps = warmup_steps
        self.divergence = nonfinite_graph_mask if divergence is None else divergence
        self.aggregate = aggregate
        self.stop_on_composition_change = stop_on_composition_change
        self._steps: list[int] = []
        self._energies: list[torch.Tensor] = []
        self._momenta: list[torch.Tensor] = []
        self._num_nodes: torch.Tensor | None = None
        self._composition: torch.Tensor | None = None
        self._stopped = False
        self._first_divergence_step: int | None = None

    @torch.compiler.disable
    def _record(self, batch: Batch, step_count: int) -> None:
        """Append one sample of the batch's total energy and momentum.

        A firing inside the warmup window is dropped whole, so the composition
        the series is fingerprinted against is the one it starts recording at.
        A firing at which any graph diverged records the step and ends the
        series without a sample. Every sample is copied off the batch, since
        the propagator writes its next energy into the same buffer in place.

        Raises
        ------
        ValueError
            If the batch is missing a field the sample is formed from.
        """
        if self._stopped or step_count < self.warmup_steps:
            return
        if bool(self.divergence(batch).any()):
            self._first_divergence_step = step_count
            self._stopped = True
            return
        missing = [
            name for name in _SAMPLED_FIELDS if getattr(batch, name, None) is None
        ]
        if missing:
            raise ValueError(
                f"StabilityMonitor cannot sample a batch carrying no {missing!r}. "
                "A batch fired outside a run has no model outputs; register the "
                "monitor on the propagator."
            )
        counts = batch.num_nodes_per_graph.detach().to("cpu", torch.float64)
        composition = _composition(batch, counts)
        if self._composition is None:
            self._num_nodes = counts
            self._composition = composition
        elif not torch.equal(composition, self._composition) and (
            self.stop_on_composition_change or not torch.equal(counts, self._num_nodes)
        ):
            self._stopped = True
            same_counts = torch.equal(counts, self._num_nodes)
            change = (
                f"the {counts.numel()} systems were replaced by others of the same "
                "atom counts"
                if same_counts
                else f"the batch went from {self._num_nodes.numel()} graphs of "
                f"{[int(size) for size in self._num_nodes]} atoms to "
                f"{counts.numel()} of {[int(size) for size in counts]}"
            )
            remedy = (
                " Pass stop_on_composition_change=False to keep recording through "
                "a refill that preserves the atom counts."
                if same_counts
                else ""
            )
            warnings.warn(
                f"StabilityMonitor stopped recording: {change}, so the per-graph "
                f"series would no longer describe the same systems.{remedy}",
                UserWarning,
                stacklevel=2,
            )
            return
        energy = batch.energy.reshape(-1)
        if self.include_kinetic:
            energy = energy + kinetic_energy_per_graph(
                batch.velocities,
                batch.atomic_masses,
                batch.batch_idx,
                batch.num_graphs,
            ).reshape(-1)
        self._steps.append(step_count)
        self._energies.append(energy.detach().to("cpu", torch.float64, copy=True))
        self._momenta.append(total_momentum(batch).detach().to("cpu", torch.float64))

    def __call__(self, ctx: DynamicsContext, stage: Enum) -> None:  # noqa: ARG002
        """Record the state the propagator has just resolved."""
        if ctx.active_graph_mask is not None:
            raise NotImplementedError(
                "StabilityMonitor does not yet support status-filtered or inflight "
                "dynamics because the energy and momentum series must be tracked "
                "per system."
            )
        self._record(ctx.batch, ctx.step_count)

    def metrics(self) -> StabilityMetrics:
        """Return the drift and conservation metrics of the recorded series.

        The call stacks the recorded samples and fits a rate over them. It
        refuses a series too short to fit. Call it once the run is over; a call
        made mid-run scores only the segment recorded so far.

        Returns
        -------
        StabilityMetrics
            Metrics over every recorded sample.

        Raises
        ------
        ValueError
            If fewer than two samples were recorded, or if every sample landed
            on the same step so no rate can be formed.
        """
        if len(self._steps) < 2 or self._num_nodes is None:
            raise ValueError(
                "StabilityMonitor needs at least two recorded samples to measure "
                f"drift; got {len(self._steps)}. Run the dynamics with the monitor "
                "registered, and check that neither 'frequency' nor 'warmup_steps' "
                "is longer than the run."
            )
        elapsed = self._steps[-1] - self._steps[0]
        if elapsed <= 0:
            raise ValueError(
                f"Recorded samples span no steps; got step counts "
                f"{self._steps[0]!r} to {self._steps[-1]!r}."
            )
        per_atom = torch.stack(self._energies) / self._num_nodes
        drift = self._across_graphs((per_atom[-1] - per_atom[0]).abs())
        momenta = torch.stack(self._momenta)
        steps = torch.tensor(self._steps, dtype=torch.float64)
        centered = steps - steps.mean()
        deviation = per_atom - per_atom.mean(dim=0)
        slope = (centered.unsqueeze(-1) * deviation).sum(dim=0) / centered.pow(2).sum()
        residual = deviation - centered.unsqueeze(-1) * slope
        rate = (
            None
            if self.timestep_fs is None
            else self._across_graphs((slope * _FS_PER_NS / self.timestep_fs).abs())
        )
        return StabilityMetrics(
            num_samples=len(self._steps),
            first_step=self._steps[0],
            last_step=self._steps[-1],
            energy_drift_per_atom=drift,
            energy_drift_per_atom_per_step=drift / elapsed,
            energy_drift_per_atom_per_ns=rate,
            max_momentum_drift=self._across_graphs(
                (momenta - momenta[0]).norm(dim=-1).amax(dim=0)
            ),
            timestep_fs=self.timestep_fs,
            energy_fluctuation_per_atom=self._across_graphs(
                residual.pow(2).mean(dim=0).sqrt()
            ),
            max_energy_excursion_per_atom=self._across_graphs(
                (per_atom - per_atom[0]).abs().amax(dim=0)
            ),
            first_divergence_step=self._first_divergence_step,
            aggregate=self.aggregate,
        )

    def _across_graphs(self, values: torch.Tensor) -> float:
        """Reduce one figure per graph to the number the monitor reports."""
        return float(values.max() if self.aggregate == "max" else values.mean())
