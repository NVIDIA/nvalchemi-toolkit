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

:class:`~nvalchemi.dynamics.hooks.StabilityMonitor` is a dynamics hook that
watches energy and momentum along a student-driven run; it lives in
:mod:`nvalchemi.dynamics.hooks` and is re-exported here with its record and
:func:`~nvalchemi.dynamics.hooks.total_momentum`. :func:`extensivity_error`
checks that the student's energy scales with system size.
:func:`radial_distribution` and :func:`compare_radial_distributions` compare
the structure a trajectory samples against the structure of a reference
trajectory, either pooled over every species or resolved to one species pair.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Collection, Iterable, Sequence
from typing import TYPE_CHECKING, Any

import torch

from nvalchemi.data import Batch
from nvalchemi.data.transforms import (
    DEFAULT_EXTENSIVE_SYSTEM_KEYS,
    DEFAULT_INTENSIVE_SYSTEM_KEYS,
    make_supercell,
)
from nvalchemi.dynamics.hooks import StabilityMetrics, StabilityMonitor, total_momentum
from nvalchemi.models.base import NeighborConfig, NeighborListFormat
from nvalchemi.neighbors import compute_neighbors
from nvalchemi.training.distillation.evaluation._export import MeasurementRecord
from nvalchemi.training.distillation.scoring import _NEIGHBOR_KEYS, _as_scorer
from nvalchemi.training.losses.reductions import per_graph_sum

if TYPE_CHECKING:
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

_EPS = 1e-12
"""Denominator guard for normalized histograms."""


def _is_periodic(batch: Batch) -> bool:
    """Return whether *batch* is periodic, read off ``pbc`` before ``cell``.

    A batch that carries ``pbc`` is periodic when any axis is periodic. A batch
    without ``pbc`` is periodic when it carries a cell, because a cluster read
    without a cell has no ``pbc`` either.
    """
    pbc = getattr(batch, "pbc", None)
    if pbc is not None:
        return bool(pbc.any())
    return getattr(batch, "cell", None) is not None


def _refuse_aperiodic(batch: Batch, purpose: str) -> None:
    """Raise unless *batch* is periodic, naming what the periodicity was for."""
    if _is_periodic(batch):
        return
    if getattr(batch, "cell", None) is None:
        raise ValueError(f"{purpose}; the batch carries no cell.")
    raise ValueError(
        f"{purpose}; the batch carries a cell but its pbc marks every axis "
        "non-periodic."
    )


class ExtensivityMetrics(MeasurementRecord):
    """Energy-scaling error of a model across replicated cells.

    *Extensivity* is the property that a structure's energy grows in proportion
    to its size: a ``k``-fold supercell has ``k`` times the energy of the cell
    it replicates. This record measures how far a model departs from it.

    Attributes
    ----------
    repeats : tuple[int, int, int]
        Replication factors applied along each lattice vector.
    num_graphs : int
        Number of structures checked.
    max_error_per_atom, mean_error_per_atom : float
        Largest and mean absolute deviation of the supercell energy from the
        copy count times the input cell's energy, divided by the supercell's
        atom count.
    max_relative_error : float
        Largest ratio of the same deviation to the magnitude of the expected
        supercell energy.
    """

    repeats: tuple[int, int, int]
    num_graphs: int
    max_error_per_atom: float
    mean_error_per_atom: float
    max_relative_error: float


def extensivity_error(
    model: TeacherScorer | BaseModelMixin,
    data: Iterable[Batch] | Batch,
    *,
    repeats: Sequence[int] = (2, 1, 1),
    extensive_keys: Collection[str] = DEFAULT_EXTENSIVE_SYSTEM_KEYS,
    intensive_keys: Collection[str] = DEFAULT_INTENSIVE_SYSTEM_KEYS,
    drop_keys: Collection[str] = (),
) -> ExtensivityMetrics:
    """Check that a model's energy scales with the number of replicated cells.

    A size-extensive potential returns exactly ``k`` times the energy for a
    ``k``-fold supercell. A student that learned a global readout breaks that
    identity. So does a potential whose numerics are re-derived from the cell
    it is given, such as an Ewald or PME tail whose splitting parameter follows
    the atom count and volume. The break shows up in MD long before it shows up
    in a held-out energy MAE.

    Parameters
    ----------
    model : TeacherScorer | BaseModelMixin
        Model to check. A bare model is wrapped in an
        :class:`~nvalchemi.training.distillation.InProcessTeacherScorer`, which
        builds and rolls back the neighbor list it needs.
    data : Iterable[Batch] | Batch
        Periodic structures to replicate; they are left unmodified.
        :func:`~nvalchemi.data.transforms.make_supercell` carries node-level
        fields into the supercell and scales system-level fields by their
        extensivity. A system-level field with no defined scaling is rejected.
    repeats : Sequence[int], optional
        Replication factors along the three lattice vectors. Default
        ``(2, 1, 1)``.
    extensive_keys : Collection[str], optional
        System-level fields multiplied by the copy count. Default
        :data:`~nvalchemi.data.transforms.DEFAULT_EXTENSIVE_SYSTEM_KEYS`.
    intensive_keys : Collection[str], optional
        System-level fields carried unchanged. Default
        :data:`~nvalchemi.data.transforms.DEFAULT_INTENSIVE_SYSTEM_KEYS`.
    drop_keys : Collection[str], optional
        Fields left out of the supercell, on top of the ephemeral neighbor
        tensors, which are always dropped. A batch returned by
        :meth:`~nvalchemi.dynamics.base.BaseDynamics.run` carries the
        propagator's bookkeeping, such as ``status`` and ``system_id``, which
        no scaling rule covers; pass ``propagator.bookkeeping_keys()`` to
        replicate such a batch. Default ``()``.

    Returns
    -------
    ExtensivityMetrics
        Worst-case and mean energy-scaling error, in eV/atom.

    Raises
    ------
    ValueError
        If *repeats* is not three positive integers. If a structure is not
        periodic (it has no cell, or its ``pbc`` marks every axis
        non-periodic), or carries a system-level field that neither scales
        with the supercell nor is dropped. If *data* holds no graphs.

    Examples
    --------
    >>> from nvalchemi.training.distillation.evaluation import extensivity_error
    >>> extensivity_error(student, holdout, repeats=(2, 2, 1))  # doctest: +SKIP
    >>> extensivity_error(  # doctest: +SKIP
    ...     student, nve.run(batch), drop_keys=nve.bookkeeping_keys()
    ... )
    """
    factors = tuple(int(count) for count in repeats)
    if len(factors) != 3 or any(count < 1 for count in factors):
        raise ValueError(
            f"repeats must be three positive integers; got {list(repeats)!r}."
        )
    scorer = _as_scorer(model, ["energy"])
    copies = factors[0] * factors[1] * factors[2]
    dropped = _NEIGHBOR_KEYS | set(drop_keys)
    errors: list[torch.Tensor] = []
    relative: list[torch.Tensor] = []
    for batch in [data] if isinstance(data, Batch) else data:
        _refuse_aperiodic(batch, "Extensivity requires periodic structures")
        supercell = Batch.from_data_list(
            [
                make_supercell(
                    structure,
                    factors,
                    extensive_keys=extensive_keys,
                    intensive_keys=intensive_keys,
                    drop_keys=dropped,
                )
                for structure in batch.to_data_list()
            ],
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

    The *radial distribution* ``g(r)`` is the pair correlation function: how
    often pairs of atoms sit at separation ``r``, normalized so an ideal gas
    gives ``1``.

    Attributes
    ----------
    edges : Float[torch.Tensor, "num_bins+1"]
        Bin edges from ``0`` to ``r_max``, in A.
    g_r : Float[torch.Tensor, "num_bins"]
        Pair correlation function, normalized so an ideal gas gives ``1``.
    counts : Float[torch.Tensor, "num_bins"]
        Ordered-pair counts summed over every graph and frame. Each pair is
        apportioned between the two bins whose centers bracket its distance.
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


class RDFComparison(MeasurementRecord):
    """Scalar divergences between two radial distribution functions.

    The comparison has the species resolution of its curves. Two total
    ``g(r)`` curves are compared without regard to species, so a student that
    swapped two sublattices can score well while its partial curves are wrong.
    ``pair`` records which resolution was measured.

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


def radial_distribution(
    frames: Iterable[Batch] | Batch,
    *,
    r_max: float = 6.0,
    num_bins: int = 60,
    pair: Sequence[int] | None = None,
) -> RadialDistribution:
    """Accumulate the radial distribution function of a trajectory.

    Every graph is one frame, so a trajectory that
    :class:`~nvalchemi.dynamics.hooks.SnapshotHook` captured into a
    :class:`~nvalchemi.dynamics.sinks.DataSink` can be passed straight from
    ``sink.read()``. Pairs are collected with the framework's own neighbor
    build at ``r_max`` on a copy of each frame, so the caller's neighbor state
    is left as it was. By default every pair goes into one
    histogram: the number-weighted total
    ``g(r) = \\sum_{ab} x_a x_b g_{ab}(r)``, which cannot see chemical
    ordering. Pass *pair* to resolve one species pair and gate a multi-species
    student on its partial curves. The normalization is
    ``g(r) = n(r) / (N \\rho V_{shell})``, generalized over frames. Each pair
    is apportioned between the two bins whose centers bracket its distance,
    and the neighbor list is built one bin past ``r_max``. The histogram is
    therefore continuous in the positions rather than stepping when a shell
    crosses a bin edge.

    Parameters
    ----------
    frames : Iterable[Batch] | Batch
        Periodic frames to accumulate.
    r_max : float, optional
        Largest pair distance binned, in A. It may exceed the cell vectors,
        because the neighbor build enumerates as many periodic images as the
        cutoff needs. Default ``6.0``.
    num_bins : int, optional
        Uniform bins between ``0`` and ``r_max``. Default ``60``.
    pair : Sequence[int] | None, optional
        Two atomic numbers ``(a, b)``. Only pairs whose first atom has atomic
        number ``a`` and whose second atom has ``b`` are counted. Default
        ``None`` (every pair, pooled into the species-blind total).

    Returns
    -------
    RadialDistribution
        Curve, raw counts, and the sizes they were accumulated over.

    Raises
    ------
    ValueError
        If ``r_max`` or ``num_bins`` is not positive. If *pair* is not two
        atomic numbers, or names a species the frames do not carry. If a frame
        is not periodic or its cell encloses no volume, or if no frame was
        supplied.

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
        _refuse_aperiodic(batch, "A radial distribution needs periodic frames")
        volumes = batch.cell.reshape(-1, 3, 3).det().abs().to(torch.float64)
        if bool((volumes <= 0.0).any()):
            raise ValueError(
                "Every frame needs a cell enclosing a positive volume; got cell "
                f"volumes {volumes.tolist()!r}. A radial distribution is normalized "
                "by the ideal-gas density, which a cell enclosing no volume does "
                "not define, and an isolated molecule read with a zero cell would "
                "score as a perfect match against anything. Give the frames a "
                "periodic cell."
            )
        frame = batch.clone(drop=_NEIGHBOR_KEYS)
        compute_neighbors(frame, config=config)
        distances = _pair_distances(frame, species)
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

    The population is ``N_g^2`` when every species is pooled, or
    ``N_{a,g} N_{b,g}`` for a resolved pair. It sets the ideal-gas expectation
    that the counts are divided by. Both forms carry the usual ``O(1/N)`` bias
    of counting ``N^2`` ordered pairs where ``N(N - 1)`` are distinct.
    """
    sizes = batch.num_nodes_per_graph.to("cpu", torch.float64)
    if species is None:
        return sizes.pow(2)
    numbers = batch.atomic_numbers.reshape(-1)
    populations = []
    for number in species:
        selected = (numbers == number).to("cpu", torch.float64)
        populations.append(
            per_graph_sum(selected, batch.batch_idx, num_graphs=batch.num_graphs)
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

    A pair contributes to the two bins whose centers bracket its distance,
    weighted by how close it sits to each. The weights sum to one. A pair in
    the half-bin margin past the last center deposits only the share that
    falls inside the range.
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
    pair-distance histograms. It is symmetric, bounded in ``[0, 1]`` with a
    base-2 logarithm, and finite where one histogram has an empty bin. A
    Kullback-Leibler divergence or a chi-squared distance is not finite there
    on RDF data. The ``g(r)`` curves themselves are compared with an
    integrated absolute difference. The comparison is only as species-resolved
    as its curves. To gate chemical ordering, build one curve pair per species
    pair with the ``pair`` argument of :func:`radial_distribution`.

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
