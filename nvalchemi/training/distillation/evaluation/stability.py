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
"""MD-stability evaluators for a student driving its own dynamics.

:class:`StabilityMonitor` is a dynamics hook watching energy and momentum along
a student-driven run, :func:`extensivity_error` checks that the student's energy
scales with system size, and :func:`radial_distribution` with
:func:`compare_radial_distributions` compares the structure a trajectory samples
against a reference trajectory's, pooled over every species or resolved to one
species pair.
"""

from __future__ import annotations

import dataclasses
import itertools
import warnings
from collections.abc import Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, Any

import torch

from nvalchemi.data import AtomicData, Batch
from nvalchemi.dynamics.base import DynamicsStage
from nvalchemi.dynamics.hooks._utils import kinetic_energy_per_graph
from nvalchemi.models.base import NeighborConfig, NeighborListFormat
from nvalchemi.training.distillation.evaluation._export import _rebuild
from nvalchemi.training.distillation.evaluation.accuracy import _as_scorer
from nvalchemi.training.distillation.scoring import (
    _DENSE_NEIGHBOR_KEYS,
    _isolated_neighbors,
)

if TYPE_CHECKING:
    from enum import Enum

    from nvalchemi.hooks._context import DynamicsContext
    from nvalchemi.models.base import BaseModelMixin
    from nvalchemi.training.distillation.scoring import TeacherScorer

__all__ = [
    "ExtensivityMetrics",
    "RDFComparison",
    "RadialDistribution",
    "StabilityMetrics",
    "StabilityMonitor",
    "compare_radial_distributions",
    "extensivity_error",
    "radial_distribution",
    "total_momentum",
]

_FS_PER_NS = 1.0e6
"""Femtoseconds in a nanosecond."""

_EPS = 1e-12
"""Denominator guard for normalized histograms."""

_SAMPLED_FIELDS = ("energy", "velocities", "atomic_masses")
"""Batch fields every recorded stability sample is formed from."""

_EXTENSIVE_SYSTEM_KEYS = frozenset({"charge", "dipole", "energy", "virial"})
"""System-level fields a k-fold supercell carries k times over."""

_INTENSIVE_SYSTEM_KEYS = frozenset({"pbc", "stress"})
"""System-level fields a supercell carries unchanged."""


def total_momentum(batch: Batch) -> torch.Tensor:
    """Return the total linear momentum of each graph.

    Parameters
    ----------
    batch : Batch
        Batch carrying ``velocities`` and ``atomic_masses``.

    Returns
    -------
    Float[torch.Tensor, "B 3"]
        Mass-weighted velocity sum per graph, in the batch's own units.
    """
    momentum = batch.atomic_masses.unsqueeze(-1) * batch.velocities
    totals = momentum.new_zeros((batch.num_graphs, 3))
    return totals.index_add_(0, batch.batch_idx, momentum)


@dataclasses.dataclass(frozen=True)
class StabilityMetrics:
    """Conservation diagnostics of one student-driven trajectory.

    Drift is the worst graph in the batch, matching
    :class:`~nvalchemi.dynamics.hooks.EnergyDriftMonitorHook`. The
    per-nanosecond rate is the slope of a least-squares fit through every
    sample rather than an endpoint difference, so a noisy series is not scored
    off whichever two samples bracket it; being a slope it reads zero for an
    excursion symmetric about the window, so read it with
    ``energy_fluctuation_per_atom``, the RMS residual about the same fit — a
    drift no larger than the fluctuation is a line through an oscillation — and
    ``max_energy_excursion_per_atom``. Both drift figures integrate whatever
    the series starts with, so a frame not equilibrated under the student's own
    potential relaxes into the fit as drift; give :class:`StabilityMonitor` a
    ``warmup_steps`` window covering the relaxation.

    Attributes
    ----------
    num_samples : int
        Recorded samples, after any discarded warmup.
    first_step, last_step : int
        Step counts of the first and last sample.
    energy_drift_per_atom : float
        ``|E(t_end) - E(t_0)| / N`` of the worst graph.
    energy_drift_per_atom_per_step : float
        The same difference divided by the elapsed steps.
    energy_drift_per_atom_per_ns : float | None
        Fitted drift rate of the worst graph in eV/atom/ns; ``None`` when the
        monitor was given no timestep.
    max_momentum_drift : float
        Largest deviation of any graph's total momentum from its initial value.
    timestep_fs : float | None
        Timestep the rates were derived with.
    energy_fluctuation_per_atom : float | None
        RMS residual of the worst graph's per-atom energy about the fitted line;
        ``None`` only when rebuilt from an export written before the field.
    max_energy_excursion_per_atom : float | None
        Largest ``|E(t) - E(t_0)| / N`` any graph reached; ``None`` only when
        rebuilt from an older export.
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

    def to_dict(self) -> dict[str, Any]:
        """Return every field as a plain dictionary."""
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> StabilityMetrics:
        """Rebuild the metrics from a :meth:`to_dict` export."""
        return _rebuild(cls, data)


def _composition(batch: Batch, counts: torch.Tensor) -> torch.Tensor:
    """Return the signature a per-graph series has to keep to stay comparable.

    Atom counts alone miss an inflight refill that replaces graduated systems
    with fresh ones of the same size, so ``system_id`` is folded in where the
    batch carries it.
    """
    identity = getattr(batch, "system_id", None)
    if identity is None:
        return counts
    return torch.cat([counts, identity.detach().reshape(-1).to("cpu", torch.float64)])


class StabilityMonitor:
    """Dynamics hook recording energy and momentum along a trajectory.

    Register it on a :class:`~nvalchemi.dynamics.base.BaseDynamics` run like
    any observation hook, then *call* :meth:`metrics` once the run is over.
    Unlike :class:`~nvalchemi.dynamics.hooks.EnergyDriftMonitorHook`, which
    compares one live value against a threshold, this hook keeps the whole
    series on the host as float64, one small tensor per firing, so a long run
    should raise ``frequency`` rather than record every step.

    Parameters
    ----------
    frequency : int, optional
        Record every ``frequency`` steps; the dynamics registry does the
        gating. Default ``1``.
    stage : Enum, optional
        Stage to record at. Default
        :attr:`~nvalchemi.dynamics.base.DynamicsStage.AFTER_STEP`.
    timestep_fs : float | None, optional
        Integration timestep in femtoseconds, which turns per-step drift into a
        per-nanosecond rate. Default ``None``.
    include_kinetic : bool, optional
        Add the kinetic energy to the potential energy before measuring drift,
        which is what makes the metric meaningful for NVE. Default ``True``.
    warmup_steps : int, optional
        Discard every firing before this step count, read off the propagator's
        own counter: the equilibration window a student seeded from frames that
        are not equilibria of its own potential needs, since a fit that
        includes the relaxation reports it as drift. Default ``0``.

    Examples
    --------
    >>> from nvalchemi.training.distillation.evaluation import StabilityMonitor
    >>> monitor = StabilityMonitor(frequency=10, timestep_fs=1.0, warmup_steps=100)
    >>> dynamics.register_hook(monitor)  # doctest: +SKIP
    >>> dynamics.run(batch)  # doctest: +SKIP
    >>> monitor.metrics().energy_drift_per_atom_per_ns  # doctest: +SKIP
    0.0042

    Notes
    -----
    Every sample is formed from the batch's own ``energy``, ``velocities``, and
    ``atomic_masses``; a batch built from geometry alone carries no energy
    field, since :meth:`~nvalchemi.dynamics.base.BaseDynamics.compute` copies
    the model's energy into a field the batch already has, and is refused on
    the first firing rather than read back from the propagator's cached
    outputs. Momentum is only conserved by an integrator that conserves it, so
    under a stochastic thermostat ``max_momentum_drift`` describes the bath and
    no bar should be set on it. Recording stops with a warning as soon as the
    batch composition changes — a different graph count, different per-graph
    atom counts, or different ``system_id`` in the slots — so a propagator that
    graduates systems mid-run is scored on the segment before the first
    graduation.
    """

    def __init__(
        self,
        *,
        frequency: int = 1,
        stage: Enum = DynamicsStage.AFTER_STEP,
        timestep_fs: float | None = None,
        include_kinetic: bool = True,
        warmup_steps: int = 0,
    ) -> None:
        self.frequency = frequency
        self.stage = stage
        self.timestep_fs = timestep_fs
        self.include_kinetic = include_kinetic
        self.warmup_steps = warmup_steps
        self._steps: list[int] = []
        self._energies: list[torch.Tensor] = []
        self._momenta: list[torch.Tensor] = []
        self._num_nodes: torch.Tensor | None = None
        self._composition: torch.Tensor | None = None
        self._stopped = False

    @torch.compiler.disable
    def _record(self, batch: Batch, step_count: int) -> None:
        """Append one sample of the batch's total energy and momentum.

        A firing inside the warmup window is dropped whole, so the composition
        the series is fingerprinted against is the one it starts recording at.
        Every sample is copied off the batch, since the propagator writes its
        next energy into the same buffer in place.

        Raises
        ------
        ValueError
            If the batch is missing a field the sample is formed from.
        """
        if self._stopped or step_count < self.warmup_steps:
            return
        missing = [
            name for name in _SAMPLED_FIELDS if getattr(batch, name, None) is None
        ]
        if missing:
            raise ValueError(
                f"StabilityMonitor cannot sample a batch carrying no {missing!r}. "
                "A frame reaches the monitor with nothing to record when the "
                "propagated model publishes no energy, or when the monitor is "
                "fired outside a run on a batch built from geometry alone. Seed "
                "every structure with AtomicData(..., energy=torch.zeros(1, 1)), "
                "which batches to the [num_graphs, 1] tensor the propagator "
                "copies into in the batch's own dtype; velocities and "
                "atomic_masses default themselves when omitted."
            )
        counts = batch.num_nodes_per_graph.detach().to("cpu", torch.float64)
        composition = _composition(batch, counts)
        if self._composition is None:
            self._num_nodes = counts
            self._composition = composition
        elif not torch.equal(composition, self._composition):
            self._stopped = True
            warnings.warn(
                "StabilityMonitor stopped recording: the batch composition "
                f"changed from {self._num_nodes.numel()} graphs of "
                f"{[int(size) for size in self._num_nodes]} atoms to "
                f"{counts.numel()} of {[int(size) for size in counts]}, or the "
                "systems in those slots were replaced, so the per-graph series "
                "would no longer describe the same systems.",
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
        self._record(ctx.batch, ctx.step_count)

    def metrics(self) -> StabilityMetrics:
        """Return the drift and conservation metrics of the recorded series.

        This is a method rather than a property: it stacks the recorded
        samples and fits a rate over them, and refuses a series too short to
        fit one. Call it once the run is over, since a call made mid-run scores
        the segment recorded so far.

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
        drift = (per_atom[-1] - per_atom[0]).abs()
        momenta = torch.stack(self._momenta)
        steps = torch.tensor(self._steps, dtype=torch.float64)
        centered = steps - steps.mean()
        deviation = per_atom - per_atom.mean(dim=0)
        slope = (centered.unsqueeze(-1) * deviation).sum(dim=0) / centered.pow(2).sum()
        residual = deviation - centered.unsqueeze(-1) * slope
        rate = (
            None
            if self.timestep_fs is None
            else float((slope * _FS_PER_NS / self.timestep_fs).abs().max())
        )
        return StabilityMetrics(
            num_samples=len(self._steps),
            first_step=self._steps[0],
            last_step=self._steps[-1],
            energy_drift_per_atom=float(drift.max()),
            energy_drift_per_atom_per_step=float(drift.max()) / elapsed,
            energy_drift_per_atom_per_ns=rate,
            max_momentum_drift=float((momenta - momenta[0]).norm(dim=-1).max()),
            timestep_fs=self.timestep_fs,
            energy_fluctuation_per_atom=float(residual.pow(2).mean(dim=0).sqrt().max()),
            max_energy_excursion_per_atom=float((per_atom - per_atom[0]).abs().max()),
        )


@dataclasses.dataclass(frozen=True)
class ExtensivityMetrics:
    """Energy-scaling error of a model across replicated cells.

    Attributes
    ----------
    repeats : tuple[int, int, int]
        Replication factors applied along each lattice vector.
    num_graphs : int
        Number of structures checked.
    max_error_per_atom, mean_error_per_atom : float
        Absolute deviation of the supercell energy from the replication factor
        times the primitive-cell energy, divided by the supercell's atom count.
    max_relative_error : float
        The same deviation of the worst structure, divided by the magnitude of
        the expected supercell energy.
    """

    repeats: tuple[int, int, int]
    num_graphs: int
    max_error_per_atom: float
    mean_error_per_atom: float
    max_relative_error: float

    def to_dict(self) -> dict[str, Any]:
        """Return every field as a plain dictionary."""
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ExtensivityMetrics:
        """Rebuild the metrics from a :meth:`to_dict` export."""
        return _rebuild(cls, data)


def _replicate(data: AtomicData, repeats: Sequence[int]) -> AtomicData:
    """Return *data* tiled ``repeats`` times along each lattice vector.

    Node-level fields are repeated copy-major with the positions and
    system-level fields scaled by their own extensivity, so the supercell
    reaches the model carrying the inputs the primitive cell did; edge-level
    fields and neighbor state are dropped for the scorer to rebuild.
    """
    cell = data.cell.reshape(3, 3)
    factors = torch.tensor(repeats, device=cell.device, dtype=cell.dtype)
    offsets = torch.stack(
        [
            torch.tensor(image, device=cell.device, dtype=cell.dtype) @ cell
            for image in itertools.product(*(range(count) for count in repeats))
        ]
    )
    copies = len(offsets)
    fields: dict[str, Any] = {
        "positions": (data.positions.unsqueeze(0) + offsets.unsqueeze(1)).reshape(
            -1, 3
        ),
        "cell": (cell * factors.unsqueeze(-1)).unsqueeze(0),
        "__node_keys__": set(data.__node_keys__) - _DENSE_NEIGHBOR_KEYS,
        "__system_keys__": set(data.__system_keys__),
    }
    for key in sorted(set(data.__node_keys__) - _DENSE_NEIGHBOR_KEYS - {"positions"}):
        value = getattr(data, key, None)
        if value is not None:
            fields[key] = value.repeat((copies,) + (1,) * (value.ndim - 1))
    unsupported = []
    for key in sorted(set(data.__system_keys__) - {"cell"}):
        value = getattr(data, key, None)
        if value is None:
            continue
        if key in _EXTENSIVE_SYSTEM_KEYS:
            fields[key] = value * copies
        elif key in _INTENSIVE_SYSTEM_KEYS:
            fields[key] = value
        else:
            unsupported.append(key)
    if unsupported:
        raise ValueError(
            f"Replicating a structure carrying {unsupported!r} is not defined: a "
            "supercell scored under a system-level field that does not scale with "
            "it would report the mismatch as an extensivity error. Drop the field "
            "from the structures handed to extensivity_error."
        )
    return AtomicData(**fields)


def extensivity_error(
    model: TeacherScorer | BaseModelMixin,
    data: Iterable[Batch] | Batch,
    *,
    repeats: Sequence[int] = (2, 1, 1),
) -> ExtensivityMetrics:
    """Check that a model's energy scales with the number of replicated cells.

    A size-extensive potential returns exactly ``k`` times the energy for a
    ``k``-fold supercell. A student that learned a global readout breaks that
    identity, and so does a potential whose numerics are re-derived from the
    cell it is handed, such as an Ewald or PME tail whose splitting parameter
    follows the atom count and volume; the break shows up in MD long before it
    shows up in a held-out energy MAE.

    Parameters
    ----------
    model : TeacherScorer | BaseModelMixin
        Model to check. A bare model is wrapped in an
        :class:`~nvalchemi.training.distillation.InProcessTeacherScorer`, which
        builds and rolls back the neighbor list it needs.
    data : Iterable[Batch] | Batch
        Periodic structures to replicate, left unmodified. Node-level fields
        are carried into the supercell and system-level ones scaled by their
        extensivity; a system-level field with no defined scaling is rejected.
    repeats : Sequence[int], optional
        Replication factors along the three lattice vectors. Default
        ``(2, 1, 1)``.

    Returns
    -------
    ExtensivityMetrics
        Worst-case and mean energy-scaling error, in eV/atom.

    Raises
    ------
    ValueError
        If *repeats* is not three positive integers, if a structure carries no
        cell or a system-level field that does not scale with the supercell, or
        if *data* holds no graphs.

    Examples
    --------
    >>> from nvalchemi.training.distillation.evaluation import extensivity_error
    >>> extensivity_error(student, holdout, repeats=(2, 2, 1))  # doctest: +SKIP
    """
    factors = tuple(int(count) for count in repeats)
    if len(factors) != 3 or any(count < 1 for count in factors):
        raise ValueError(
            f"repeats must be three positive integers; got {list(repeats)!r}."
        )
    scorer = _as_scorer(model, ["energy"])
    copies = factors[0] * factors[1] * factors[2]
    errors: list[torch.Tensor] = []
    relative: list[torch.Tensor] = []
    for batch in [data] if isinstance(data, Batch) else data:
        if getattr(batch, "cell", None) is None:
            raise ValueError(
                "Extensivity requires periodic structures; the batch carries no cell."
            )
        structures = batch.to_data_list()
        supercell = Batch.from_data_list(
            [_replicate(structure, factors) for structure in structures],
            device=batch.device,
        )
        expected = copies * scorer.label(batch)["teacher_energy"][0].reshape(-1)
        observed = scorer.label(supercell)["teacher_energy"][0].reshape(-1)
        deviation = (observed - expected).abs().to(torch.float64)
        errors.append(deviation / (copies * batch.num_nodes_per_graph))
        relative.append(deviation / expected.abs().to(torch.float64).clamp_min(_EPS))
    if not errors:
        raise ValueError("data must hold at least one graph to replicate.")
    per_atom = torch.cat(errors)
    return ExtensivityMetrics(
        repeats=factors,
        num_graphs=int(per_atom.numel()),
        max_error_per_atom=float(per_atom.max()),
        mean_error_per_atom=float(per_atom.mean()),
        max_relative_error=float(torch.cat(relative).max()),
    )


@dataclasses.dataclass(frozen=True)
class RadialDistribution:
    """Radial distribution function accumulated over one or more frames.

    Attributes
    ----------
    edges : Float[torch.Tensor, "num_bins+1"]
        Bin edges from ``0`` to ``r_max``, in A.
    g_r : Float[torch.Tensor, "num_bins"]
        Pair correlation function, normalized so an ideal gas gives ``1``.
    counts : Float[torch.Tensor, "num_bins"]
        Ordered-pair counts summed over every graph and frame, each pair
        apportioned between the two bins whose centres bracket its distance.
    num_frames : int
        Number of graphs the histogram was accumulated over.
    num_atoms : int
        Number of atoms summed over the same graphs, whatever the pair filter.
    r_max : float
        Cutoff the pairs were collected within.
    pair : tuple[int, int] | None
        Atomic numbers the pairs were restricted to, or ``None`` for the total
        ``g(r)`` over every species at once.
    """

    edges: torch.Tensor
    g_r: torch.Tensor
    counts: torch.Tensor
    num_frames: int
    num_atoms: int
    r_max: float
    pair: tuple[int, int] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return the scalar fields and the curve as lists."""
        return {
            "edges": self.edges.tolist(),
            "g_r": self.g_r.tolist(),
            "counts": self.counts.tolist(),
            "num_frames": self.num_frames,
            "num_atoms": self.num_atoms,
            "r_max": self.r_max,
            "pair": self.pair,
        }


@dataclasses.dataclass(frozen=True)
class RDFComparison:
    """Scalar divergences between two radial distribution functions.

    The comparison inherits the species resolution of its curves: two total
    ``g(r)`` curves are compared species-blind, so a student that swapped two
    sublattices scores well while its partials are wrong, and ``pair`` records
    which resolution was measured.

    Attributes
    ----------
    jensen_shannon : float
        Base-2 Jensen-Shannon divergence between the two normalized
        pair-distance histograms, in ``[0, 1]``.
    l1 : float
        Integrated absolute difference of the two ``g(r)`` curves, in A.
    max_deviation : float
        Largest absolute difference between the two curves in any bin.
    num_bins : int
        Bins the comparison ran over.
    pair : tuple[int, int] | None
        Atomic numbers both curves were resolved to, or ``None`` for
        species-blind totals.
    """

    jensen_shannon: float
    l1: float
    max_deviation: float
    num_bins: int
    pair: tuple[int, int] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return every field as a plain dictionary."""
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> RDFComparison:
        """Rebuild the comparison from a :meth:`to_dict` export."""
        return _rebuild(cls, data)


def radial_distribution(
    frames: Iterable[Batch] | Batch,
    *,
    r_max: float = 6.0,
    num_bins: int = 60,
    pair: Sequence[int] | None = None,
) -> RadialDistribution:
    """Accumulate the radial distribution function of a trajectory.

    Every graph is one frame, so a trajectory captured with
    :class:`~nvalchemi.dynamics.hooks.SnapshotHook` into a
    :class:`~nvalchemi.dynamics.sinks.DataSink` is read straight out of
    ``sink.read()``. Pairs are collected with the framework's own neighbor
    build at ``r_max`` and released afterwards, so the caller's neighbor state
    survives. By default every pair goes into one histogram, the
    number-weighted total ``g(r) = \\sum_{ab} x_a x_b g_{ab}(r)``, which is
    blind to chemical ordering; pass *pair* to resolve one species pair and
    gate a multi-species student on the partials. The normalization is
    ``g(r) = n(r) / (N \\rho V_{shell})`` generalized over frames, and each pair
    is apportioned between the two bins whose centres bracket its distance,
    with the neighbor list built one bin past ``r_max``, so the histogram is
    continuous in the positions rather than stepping when a shell crosses a
    bin edge.

    Parameters
    ----------
    frames : Iterable[Batch] | Batch
        Periodic frames to accumulate.
    r_max : float, optional
        Largest pair distance binned, in A; the neighbor build enumerates
        periodic images as deep as the cutoff needs, so it may exceed the cell
        vectors. Default ``6.0``.
    num_bins : int, optional
        Uniform bins between ``0`` and ``r_max``. Default ``60``.
    pair : Sequence[int] | None, optional
        Two atomic numbers ``(a, b)``, counting only the pairs whose first atom
        carries ``a`` and whose second carries ``b``. Default ``None`` (every
        pair, pooled into the species-blind total).

    Returns
    -------
    RadialDistribution
        Curve, raw counts, and the sizes they were accumulated over.

    Raises
    ------
    ValueError
        If ``r_max`` or ``num_bins`` is not positive, if *pair* is not two
        atomic numbers or names a species the frames do not carry, if a frame
        carries no cell or a cell of zero volume, or if no frame was supplied.

    Examples
    --------
    >>> from nvalchemi.training.distillation.evaluation import radial_distribution
    >>> student_rdf = radial_distribution(sink.read(), r_max=6.0)  # doctest: +SKIP
    >>> na_cl = radial_distribution(sink.read(), pair=(11, 17))  # doctest: +SKIP
    """
    if r_max <= 0.0 or num_bins <= 0:
        raise ValueError(
            f"r_max and num_bins must be positive; got r_max={r_max!r}, "
            f"num_bins={num_bins!r}."
        )
    species = None if pair is None else tuple(int(number) for number in pair)
    if species is not None and len(species) != 2:
        raise ValueError(
            "pair must be two atomic numbers to resolve a partial g(r); got "
            f"{list(pair)!r}."
        )
    width = r_max / num_bins
    config = NeighborConfig(
        cutoff=r_max + width, format=NeighborListFormat.COO, half_list=False
    )
    counts = torch.zeros(num_bins, dtype=torch.float64)
    ideal = 0.0
    num_frames = 0
    num_atoms = 0
    for batch in [frames] if isinstance(frames, Batch) else frames:
        cell = getattr(batch, "cell", None)
        if cell is None:
            raise ValueError(
                "A radial distribution needs periodic frames; the batch carries "
                "no cell."
            )
        volumes = cell.reshape(-1, 3, 3).det().abs().to(torch.float64)
        if bool((volumes <= 0.0).any()):
            raise ValueError(
                "A radial distribution is normalized by the ideal-gas density, "
                "which a cell enclosing no volume does not define; got cell "
                f"volumes {volumes.tolist()!r}. Give the frames a periodic cell — "
                "an isolated molecule read with a zero cell would otherwise be "
                "scored as a perfect match against anything."
            )
        with _isolated_neighbors(batch, config):
            distances = _pair_distances(batch, species)
        counts += _cloud_in_cell(distances, num_bins, width)
        ideal += float((_pair_populations(batch, species) / volumes.cpu()).sum())
        num_frames += batch.num_graphs
        num_atoms += batch.num_nodes
    if num_frames == 0:
        raise ValueError("frames must hold at least one graph.")
    if species is not None and ideal <= 0.0:
        raise ValueError(
            "The frames carry no atom of one of the atomic numbers "
            f"{list(species)!r}, so a partial g(r) over that pair has no "
            "ideal-gas density to normalize against."
        )
    edges = torch.linspace(0.0, r_max, num_bins + 1, dtype=torch.float64)
    shells = (4.0 / 3.0) * torch.pi * (edges[1:].pow(3) - edges[:-1].pow(3))
    return RadialDistribution(
        edges=edges,
        g_r=counts / (ideal * shells).clamp_min(_EPS),
        counts=counts,
        num_frames=num_frames,
        num_atoms=num_atoms,
        r_max=r_max,
        pair=species,
    )


def _pair_populations(batch: Batch, species: tuple[int, int] | None) -> torch.Tensor:
    """Return each graph's ordered-pair population for the counted species.

    ``N_g^2`` pooled over every species, or ``N_{a,g} N_{b,g}`` for a resolved
    pair, which is the ideal-gas expectation the counts are divided by. Both
    carry the usual ``O(1/N)`` bias of counting ``N^2`` ordered pairs where
    ``N(N - 1)`` are distinct.
    """
    sizes = batch.num_nodes_per_graph.to("cpu", torch.float64)
    if species is None:
        return sizes.pow(2)
    numbers = batch.atomic_numbers.reshape(-1)
    populations = []
    for number in species:
        selected = (numbers == number).to("cpu", torch.float64)
        populations.append(
            sizes.new_zeros(batch.num_graphs).index_add_(
                0, batch.batch_idx.cpu(), selected
            )
        )
    return populations[0] * populations[1]


def _pair_distances(batch: Batch, species: tuple[int, int] | None) -> torch.Tensor:
    """Return the length of every pair in the batch's sparse neighbor list.

    A *species* filter keeps the ordered pairs running from its first atomic
    number to its second, matching the ordered population the counts are
    normalized by. The list is a full one, so it is closed under transposition
    and ``(a, b)`` selects the same distances as ``(b, a)``.
    """
    neighbors = batch.neighbor_list
    source = neighbors[:, 0]
    target = neighbors[:, 1]
    delta = batch.positions[target] - batch.positions[source]
    shifts = getattr(batch, "neighbor_list_shifts", None)
    if shifts is not None:
        cells = batch.cell.reshape(-1, 3, 3)[batch.batch_idx[source]]
        delta = delta + torch.einsum("ms,msd->md", shifts.to(delta.dtype), cells)
    distances = delta.norm(dim=-1)
    if species is None:
        return distances
    numbers = batch.atomic_numbers.reshape(-1)
    return distances[(numbers[source] == species[0]) & (numbers[target] == species[1])]


def _cloud_in_cell(
    distances: torch.Tensor, num_bins: int, width: float
) -> torch.Tensor:
    """Return the pair histogram, each distance split between two bins.

    A pair contributes to the two bins whose centres bracket its distance,
    weighted by how close it sits to each; the weights sum to one, and a pair
    in the half-bin margin past the last centre deposits only the share that
    falls inside.
    """
    offsets = distances.to("cpu", torch.float64) / width - 0.5
    lower = offsets.floor()
    upper_weight = offsets - lower
    index = lower.long() + 1
    padded = torch.zeros(num_bins + 2, dtype=torch.float64)
    padded.index_add_(0, index.clamp(0, num_bins + 1), 1.0 - upper_weight)
    padded.index_add_(0, (index + 1).clamp(0, num_bins + 1), upper_weight)
    return padded[1:-1]


def compare_radial_distributions(
    reference: RadialDistribution, candidate: RadialDistribution
) -> RDFComparison:
    """Score how far a candidate trajectory's structure sits from a reference.

    The headline number is the Jensen-Shannon divergence of the two normalized
    pair-distance histograms: symmetric, bounded in ``[0, 1]`` with a base-2
    logarithm, and finite where one histogram has an empty bin, which a
    Kullback-Leibler divergence or a chi-squared distance is not on RDF data.
    The ``g(r)`` curves themselves are compared with an integrated absolute
    difference. The comparison is only as species-resolved as its curves; build
    one curve pair per species pair with :func:`radial_distribution`'s ``pair``
    to gate chemical ordering.

    Parameters
    ----------
    reference : RadialDistribution
        Reference-trajectory curve.
    candidate : RadialDistribution
        Student-trajectory curve, binned identically and resolved to the same
        species pair.

    Returns
    -------
    RDFComparison
        The divergence and the two curve distances.

    Raises
    ------
    ValueError
        If the two curves do not share bin edges, if they resolve different
        species, or if either accumulated no pairs at all.

    Examples
    --------
    >>> from nvalchemi.training.distillation.evaluation import (
    ...     compare_radial_distributions,
    ...     radial_distribution,
    ... )
    >>> match = compare_radial_distributions(  # doctest: +SKIP
    ...     radial_distribution(teacher_frames),
    ...     radial_distribution(student_frames),
    ... )
    """
    if reference.edges.shape != candidate.edges.shape or not torch.allclose(
        reference.edges, candidate.edges
    ):
        raise ValueError(
            "Radial distributions must share bin edges to be compared; got "
            f"r_max={reference.r_max!r} over {reference.counts.numel()} bins "
            f"against r_max={candidate.r_max!r} over "
            f"{candidate.counts.numel()} bins."
        )
    if reference.pair != candidate.pair:
        raise ValueError(
            "Radial distributions must resolve the same species to be compared; "
            f"got pair={reference.pair!r} against pair={candidate.pair!r}."
        )
    reference_total = float(reference.counts.sum())
    candidate_total = float(candidate.counts.sum())
    if reference_total <= 0.0 or candidate_total <= 0.0:
        raise ValueError(
            "Both radial distributions must hold pairs; got "
            f"{reference_total!r} and {candidate_total!r} counted pairs."
        )
    first = reference.counts / reference_total
    second = candidate.counts / candidate_total
    mixture = 0.5 * (first + second)
    divergence = 0.5 * (
        _relative_entropy(first, mixture) + _relative_entropy(second, mixture)
    )
    difference = (reference.g_r - candidate.g_r).abs()
    width = reference.edges[1] - reference.edges[0]
    return RDFComparison(
        jensen_shannon=float(divergence),
        l1=float((difference * width).sum()),
        max_deviation=float(difference.max()),
        num_bins=int(reference.counts.numel()),
        pair=reference.pair,
    )


def _relative_entropy(
    distribution: torch.Tensor, mixture: torch.Tensor
) -> torch.Tensor:
    """Return the base-2 Kullback-Leibler divergence, treating empty bins as zero."""
    ratio = distribution.clamp_min(_EPS) / mixture.clamp_min(_EPS)
    return (distribution * ratio.log2()).sum()
