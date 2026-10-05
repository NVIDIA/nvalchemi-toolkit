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
"""Tests for :mod:`nvalchemi.training.distillation.evaluation.stability`."""

from __future__ import annotations

import itertools
from typing import Any

import pytest
import torch

from nvalchemi.data import AtomicData, Batch
from nvalchemi.data.datapipes.in_memory_dataset import InMemoryDataset
from nvalchemi.data.transforms import DEFAULT_INTENSIVE_SYSTEM_KEYS
from nvalchemi.dynamics import OrderedStructureSampler
from nvalchemi.dynamics.base import DynamicsStage
from nvalchemi.dynamics.integrators import NVE
from nvalchemi.hooks.neighbor_list import NeighborListHook
from nvalchemi.models.base import NeighborConfig, NeighborListFormat
from nvalchemi.neighbors import compute_neighbors
from nvalchemi.training.distillation.evaluation import (
    compare_radial_distributions,
    extensivity_error,
    radial_distribution,
)
from test.training.conftest import _build_batch
from test.training.distillation.conftest import (
    _build_lattice_batch,
    _build_lattice_data,
    _build_lj_teacher,
    _build_pair_batch,
)

_LATTICE_ATOMS = 27
"""Atom count of the default 3x3x3 lattice."""

_BINARY_CELLS = 4
"""Cells per axis of the two-species lattice, even so both orderings tile it."""

_BINARY_SPACING = 3.0
"""Nearest-neighbour distance of the two-species lattice, in A."""

_TRANSLATION = (0.37, 0.11, 0.23)
"""Rigid shift that moves a lattice off its own cell origin, in A."""

_FCC_LATTICE = 4.05
"""Conventional cubic lattice constant of the FCC test crystal, in A."""

_FCC_BASIS = ((0.0, 0.0, 0.0), (0.0, 0.5, 0.5), (0.5, 0.0, 0.5), (0.5, 0.5, 0.0))
"""Fractional sites of the conventional FCC cell."""


def _make_binary_lattice(*, alternate: bool) -> Batch:
    """Return a two-species cubic lattice ordered by site parity or by plane.

    Both orderings hold the same positions and the same population of each
    species, so their pooled pair distances are identical to the bin and only
    the species-resolved ones tell them apart: every nearest neighbour of the
    alternating lattice is unlike, while the layered one has four like
    neighbours out of six.
    """
    sites = list(itertools.product(range(_BINARY_CELLS), repeat=3))
    positions = torch.tensor(
        [[index * _BINARY_SPACING for index in site] for site in sites],
        dtype=torch.float32,
    )
    kinds = [(sum(site) if alternate else site[0]) % 2 for site in sites]
    data = AtomicData(
        positions=positions,
        atomic_numbers=torch.tensor(
            [11 if kind == 0 else 17 for kind in kinds], dtype=torch.long
        ),
        atomic_masses=torch.full((len(sites),), 20.0),
        cell=torch.eye(3).unsqueeze(0) * (_BINARY_CELLS * _BINARY_SPACING),
        pbc=torch.ones(1, 3, dtype=torch.bool),
    )
    return Batch.from_data_list([data])


def _make_shifted_lattice(
    offset: tuple[float, float, float], dtype: torch.dtype
) -> Batch:
    """Return the two-species lattice rigidly translated by *offset*, in *dtype*.

    A rigid translation leaves every pair distance exactly as it was, so any
    curve that moves under it is measuring round-off rather than structure.
    Its nearest-neighbour shell divides the default binning and its second
    shell sits on ``r_max``, which is where the wrapping is worst.
    """
    data = _make_binary_lattice(alternate=True).to_data_list()[0]
    positions = data.positions.to(torch.float64) + torch.tensor(
        offset, dtype=torch.float64
    )
    return Batch.from_data_list(
        [
            AtomicData(
                positions=positions.to(dtype),
                atomic_numbers=data.atomic_numbers,
                atomic_masses=data.atomic_masses.to(dtype),
                cell=data.cell.to(dtype),
                pbc=data.pbc,
            )
        ]
    )


def _make_metal(positions: torch.Tensor, cell: torch.Tensor) -> Batch:
    """Return a single-species periodic frame holding *positions* in *cell*."""
    count = positions.shape[0]
    data = AtomicData(
        positions=positions,
        atomic_numbers=torch.full((count,), 13, dtype=torch.long),
        atomic_masses=torch.full((count,), 26.98, dtype=torch.float64),
        cell=cell.unsqueeze(0),
        pbc=torch.ones(1, 3, dtype=torch.bool),
    )
    return Batch.from_data_list([data])


def _make_fcc(lattice: float, cells: int = 2) -> Batch:
    """Return an FCC crystal of *cells* conventional cells per axis."""
    positions = torch.tensor(
        [
            [(origin[axis] + site[axis]) * lattice for axis in range(3)]
            for origin in itertools.product(range(cells), repeat=3)
            for site in _FCC_BASIS
        ],
        dtype=torch.float64,
    )
    cell = torch.eye(3, dtype=torch.float64) * (lattice * cells)
    return _make_metal(positions, cell)


def _make_primitive_fcc(lattice: float) -> Batch:
    """Return the one-atom primitive cell of the same FCC crystal.

    Its cell vectors are ``lattice / sqrt(2)`` long, so any useful ``r_max``
    is a multiple of the cell and only a build that enumerates every periodic
    image reproduces the conventional crystal's curve.
    """
    half = lattice / 2.0
    cell = torch.tensor(
        [[0.0, half, half], [half, 0.0, half], [half, half, 0.0]], dtype=torch.float64
    )
    return _make_metal(torch.zeros(1, 3, dtype=torch.float64), cell)


def _make_graded_lattice() -> Batch:
    """Return the two-species lattice with a charge that differs on every atom.

    The lattice spacing divides its cell exactly in binary, so a site term
    wrapped on the primitive cell is exact to the last bit, and no two atoms
    share a charge, so a supercell that tiled the field out of step with the
    positions scores differently rather than coincidentally the same.
    """
    data = _make_binary_lattice(alternate=True).to_data_list()[0]
    data.add_node_property("charges", torch.arange(data.num_nodes, dtype=torch.float32))
    return Batch.from_data_list([data])


class _ChargeSumScorer:
    """Scorer summing a per-atom energy that reads an optional charge.

    Size-extensive by construction, and it defaults a missing charge to zero
    the way :class:`~nvalchemi.models.uma.UMAWrapper` defaults a missing tag,
    so any error the extensivity check reports comes from the supercell losing
    the field rather than from the model.
    """

    signals = frozenset({"energy"})

    def label(self, batch: Batch) -> dict[str, Any]:
        """Return each graph's summed ``1 + charge`` per-atom energy."""
        charges = getattr(batch, "charges", None)
        if charges is None:
            charges = torch.zeros(batch.num_nodes)
        per_atom = 1.0 + charges.reshape(-1)
        energy = per_atom.new_zeros(batch.num_graphs).index_add_(
            0, batch.batch_idx, per_atom
        )
        return {"teacher_energy": (energy.reshape(-1, 1),)}


class _SizeSquaredScorer:
    """Deliberately non-extensive scorer whose energy is the atom count squared."""

    signals = frozenset({"energy"})

    def label(self, batch: Batch) -> dict[str, Any]:
        """Return each graph's squared atom count."""
        counts = batch.num_nodes_per_graph.to(torch.float64)
        return {"teacher_energy": (counts.pow(2).reshape(-1, 1),)}


class _TotalChargeScorer:
    """Scorer whose energy is one per atom plus the system's total charge."""

    signals = frozenset({"energy"})

    def label(self, batch: Batch) -> dict[str, Any]:
        """Return each graph's atom count plus its total charge."""
        counts = batch.num_nodes_per_graph.to(torch.float64)
        charge = batch.charge.reshape(-1).to(torch.float64)
        return {"teacher_energy": ((counts + charge).reshape(-1, 1),)}


class _SiteResolvedScorer:
    """Per-atom energy pairing each atom's charge with the site it occupies."""

    signals = frozenset({"energy"})

    def __init__(self, period: float) -> None:
        self.period = period

    def label(self, batch: Batch) -> dict[str, Any]:
        """Return each graph's summed charge-weighted site energy."""
        sites = torch.remainder(batch.positions.to(torch.float64), self.period)
        per_atom = batch.charges.reshape(-1).to(torch.float64) * sites.sum(dim=-1)
        energy = per_atom.new_zeros(batch.num_graphs).index_add_(
            0, batch.batch_idx, per_atom
        )
        return {"teacher_energy": (energy.reshape(-1, 1),)}


def _make_nve(model: object) -> NVE:
    """Return an NVE integrator with the Lennard-Jones neighbor-list hook."""
    return NVE(
        model=model,
        dt=1.0,
        hooks=[
            NeighborListHook(
                config=model.model_config.neighbor_config,
                skin=1.0,
                stage=DynamicsStage.BEFORE_COMPUTE,
            )
        ],
    )


class TestExtensivity:
    """Energy scaling of a model across replicated cells."""

    def test_a_propagated_batch_replicates_once_its_bookkeeping_is_dropped(
        self,
    ) -> None:
        """A sampler-seeded run carries bookkeeping the supercell refuses until dropped."""
        sampler = OrderedStructureSampler(
            InMemoryDataset(in_memory_batch=_build_lattice_batch())
        )
        nve = _make_nve(_build_lj_teacher())
        state = nve.run(sampler.initial_batch(), n_steps=1)
        assert {"status", "system_id"} <= set(nve.bookkeeping_keys())
        assert state.status is not None
        with pytest.raises(ValueError, match="is not defined"):
            extensivity_error(_build_lj_teacher(), state)
        metrics = extensivity_error(
            _build_lj_teacher(), state, drop_keys=nve.bookkeeping_keys()
        )
        assert metrics.num_graphs == 1
        assert metrics.max_error_per_atom == pytest.approx(0.0, abs=1e-6)

    def test_lennard_jones_supercell_energy_is_exactly_extensive(self) -> None:
        """Doubling the cell doubles the pair energy to floating-point precision."""
        metrics = extensivity_error(
            _build_lj_teacher(), _build_lattice_batch(), repeats=(2, 1, 1)
        )
        assert metrics.num_graphs == 1
        assert metrics.max_error_per_atom == pytest.approx(0.0, abs=1e-9)
        assert metrics.max_relative_error == pytest.approx(0.0, abs=1e-6)

    def test_replication_along_every_axis_is_supported(self) -> None:
        """A 2x2x2 supercell is eight copies and eight times the energy."""
        metrics = extensivity_error(
            _build_lj_teacher(), _build_lattice_batch(cells=2), repeats=(2, 2, 2)
        )
        assert metrics.repeats == (2, 2, 2)
        assert metrics.mean_error_per_atom == pytest.approx(0.0, abs=1e-9)

    def test_a_non_extensive_model_is_scored_per_supercell_atom(self) -> None:
        """A model growing as N^2 is off by (k-1)N per atom of the worst graph."""
        batch = Batch.from_data_list(
            [_build_lattice_data(cells=2), _build_lattice_data(cells=3)]
        )
        metrics = extensivity_error(_SizeSquaredScorer(), batch, repeats=(2, 1, 1))
        assert metrics.num_graphs == 2
        assert metrics.max_error_per_atom == pytest.approx(27.0)
        assert metrics.mean_error_per_atom == pytest.approx(17.5)
        assert metrics.max_relative_error == pytest.approx(1.0)

    def test_an_extensive_system_field_is_scaled_into_the_supercell(self) -> None:
        """A model reading the total charge sees k times it, not the cell's."""
        data = _build_lattice_data(cells=2)
        data.charge = torch.full((1, 1), 4.0)
        metrics = extensivity_error(
            _TotalChargeScorer(), Batch.from_data_list([data]), repeats=(2, 1, 1)
        )
        assert metrics.max_error_per_atom == pytest.approx(0.0, abs=1e-9)

    def test_a_node_field_is_tiled_alongside_the_positions_it_belongs_to(self) -> None:
        """Copy-major tiling keeps every atom's field on the site it came from."""
        metrics = extensivity_error(
            _SiteResolvedScorer(_BINARY_CELLS * _BINARY_SPACING),
            _make_graded_lattice(),
            repeats=(2, 1, 1),
        )
        assert metrics.max_error_per_atom == pytest.approx(0.0, abs=1e-9)

    def test_a_model_reading_a_per_atom_field_still_sees_it_in_the_supercell(
        self,
    ) -> None:
        """A field the primitive cell carries is replicated, not defaulted away."""
        data = _build_lattice_data(cells=2)
        data.add_node_property("charges", torch.full((data.num_nodes,), 0.25))
        metrics = extensivity_error(
            _ChargeSumScorer(), Batch.from_data_list([data]), repeats=(2, 1, 1)
        )
        assert metrics.max_error_per_atom == pytest.approx(0.0, abs=1e-9)

    def test_a_field_that_does_not_scale_with_the_supercell_is_rejected(self) -> None:
        """A spin multiplicity has no k-fold value, so replication raises."""
        data = _build_lattice_data(cells=2)
        data.add_system_property("spin", torch.ones(1, 1))
        with pytest.raises(ValueError, match="is not defined"):
            extensivity_error(_build_lj_teacher(), Batch.from_data_list([data]))

    def test_the_caller_can_declare_how_a_system_field_scales(self) -> None:
        """A field the defaults refuse passes once a keyword set places it."""
        data = _build_lattice_data(cells=2)
        data.add_system_property("spin", torch.ones(1, 1))
        metrics = extensivity_error(
            _build_lj_teacher(),
            Batch.from_data_list([data]),
            intensive_keys={*DEFAULT_INTENSIVE_SYSTEM_KEYS, "spin"},
        )
        assert metrics.max_error_per_atom == pytest.approx(0.0, abs=1e-6)

    def test_a_cutoff_past_half_the_supercell_stays_extensive(self) -> None:
        """The neighbor build enumerates every image, so a long cutoff is fine."""
        metrics = extensivity_error(
            _build_lj_teacher(cutoff=14.0), _build_lattice_batch(), repeats=(2, 2, 2)
        )
        assert metrics.max_error_per_atom == pytest.approx(0.0, abs=1e-6)

    def test_a_cell_marked_non_periodic_on_every_axis_is_rejected(self) -> None:
        """Periodicity is read off pbc, so an all-False pbc is a cluster with a box."""
        data = _build_lattice_data(cells=2)
        data.pbc = torch.zeros(1, 3, dtype=torch.bool)
        with pytest.raises(ValueError, match="pbc marks every axis non-periodic"):
            extensivity_error(_build_lj_teacher(), Batch.from_data_list([data]))
        with pytest.raises(ValueError, match="pbc marks every axis non-periodic"):
            radial_distribution(Batch.from_data_list([data]))

    def test_non_periodic_structures_are_rejected(self) -> None:
        """Replicating a cluster is not defined, so it raises instead."""
        with pytest.raises(ValueError, match="no cell"):
            extensivity_error(_build_lj_teacher(), _build_batch())

    @pytest.mark.parametrize(
        "repeats",
        [(2, 1), (0, 1, 1), (-1, 1, 1), (1.5, 1, 1)],
        ids=["too-short", "zero", "negative", "fractional"],
    )
    def test_invalid_repeat_counts_are_rejected(self, repeats: tuple[int, ...]) -> None:
        """Replication factors must be three positive integers; a fraction is not truncated."""
        with pytest.raises(ValueError, match="three positive integers"):
            extensivity_error(
                _build_lj_teacher(), _build_lattice_batch(), repeats=repeats
            )


class TestRadialDistribution:
    """Pair correlation accumulated over frames."""

    def test_simple_cubic_lattice_has_six_nearest_neighbors(self) -> None:
        """Every atom of the lattice has exactly six neighbors inside 5 A."""
        rdf = radial_distribution(_build_lattice_batch(), r_max=5.0, num_bins=25)
        assert float(rdf.counts.sum()) == 6.0 * _LATTICE_ATOMS
        assert rdf.num_atoms == _LATTICE_ATOMS
        assert float(rdf.edges[int(rdf.g_r.argmax())]) == pytest.approx(3.8)

    def test_isolated_pair_integrates_to_one_neighbor(self) -> None:
        """The normalization reproduces the coordination number of a lone pair."""
        rdf = radial_distribution(_build_pair_batch(3.0), r_max=6.0, num_bins=12)
        shells = (4.0 / 3.0) * torch.pi * (rdf.edges[1:].pow(3) - rdf.edges[:-1].pow(3))
        density = 2.0 / 20.0**3
        assert float((rdf.g_r * shells).sum() * density) == pytest.approx(1.0)

    def test_frames_are_averaged_over_graphs(self) -> None:
        """Two identical frames give the same curve as one, with twice the counts."""
        single = radial_distribution(_build_pair_batch(3.0), r_max=6.0, num_bins=12)
        doubled = radial_distribution(
            Batch.from_data_list(_build_pair_batch(3.0).to_data_list() * 2),
            r_max=6.0,
            num_bins=12,
        )
        assert doubled.num_frames == 2
        assert float(doubled.counts.sum()) == 2.0 * float(single.counts.sum())
        torch.testing.assert_close(doubled.g_r, single.g_r)

    def test_frames_keep_the_neighbor_state_they_arrived_with(self) -> None:
        """The pair build runs on a copy, so the frame gains no neighbor list."""
        batch = _build_lattice_batch()
        radial_distribution(batch, r_max=5.0, num_bins=25)
        assert "neighbor_list" not in batch
        assert "neighbor_matrix" not in batch

    def test_a_stale_dense_list_on_the_frame_keeps_its_identity(self) -> None:
        """A caller's own neighbor_matrix is neither rebuilt nor replaced by a copy."""
        batch = _build_lattice_batch()
        compute_neighbors(
            batch, config=NeighborConfig(cutoff=4.0, format=NeighborListFormat.MATRIX)
        )
        before = batch.neighbor_matrix
        radial_distribution(batch, r_max=6.0)
        assert batch.neighbor_matrix is before
        assert "neighbor_list" not in batch

    def test_non_periodic_frames_are_rejected(self) -> None:
        """Without a cell there is no density to normalize against."""
        with pytest.raises(ValueError, match="no cell"):
            radial_distribution(_build_batch())

    def test_frames_whose_cell_encloses_no_volume_are_rejected(self) -> None:
        """A zero cell has no density, so two unrelated molecules would match."""
        data = _build_lattice_data()
        data.cell = torch.zeros(1, 3, 3)
        with pytest.raises(ValueError, match="enclosing no volume"):
            radial_distribution(Batch.from_data_list([data]), r_max=5.0, num_bins=25)

    @pytest.mark.parametrize(
        ("r_max", "num_bins"), [(0.0, 10), (5.0, 0)], ids=["no-range", "no-bins"]
    )
    def test_degenerate_binning_is_rejected(self, r_max: float, num_bins: int) -> None:
        """A histogram needs a positive range and at least one bin."""
        with pytest.raises(ValueError, match="must be positive"):
            radial_distribution(_build_lattice_batch(), r_max=r_max, num_bins=num_bins)


class TestSpeciesResolvedRadialDistribution:
    """Partial pair correlations against the species-blind total."""

    def test_the_total_curve_cannot_tell_two_orderings_apart(self) -> None:
        """Permuting species over fixed positions leaves the pooled g(r) identical."""
        alternating = radial_distribution(
            _make_binary_lattice(alternate=True), r_max=5.0, num_bins=24
        )
        layered = radial_distribution(
            _make_binary_lattice(alternate=False), r_max=5.0, num_bins=24
        )
        assert alternating.pair is None
        assert compare_radial_distributions(alternating, layered).jensen_shannon == 0.0

    def test_a_resolved_pair_separates_the_orderings(self) -> None:
        """The partial g_ab(r) sees the chemical ordering the total pooled away."""
        curves = [
            radial_distribution(
                _make_binary_lattice(alternate=alternate),
                r_max=5.0,
                num_bins=24,
                pair=(11, 17),
            )
            for alternate in (True, False)
        ]
        match = compare_radial_distributions(*curves)
        assert match.jensen_shannon > 0.5
        assert match.pair == (11, 17)

    def test_the_partials_account_for_every_pooled_pair(self) -> None:
        """Each ordered pair lands in exactly one species-resolved histogram."""
        frames = _make_binary_lattice(alternate=True)
        total = radial_distribution(frames, r_max=5.0, num_bins=24)
        partials = [
            radial_distribution(frames, r_max=5.0, num_bins=24, pair=pair).counts
            for pair in ((11, 11), (11, 17), (17, 11), (17, 17))
        ]
        torch.testing.assert_close(sum(partials), total.counts)

    def test_a_partial_integrates_to_the_unlike_coordination_number(self) -> None:
        """The partial normalization counts the neighbours of the other species."""
        rdf = radial_distribution(
            _make_binary_lattice(alternate=True), r_max=5.0, num_bins=24, pair=(11, 17)
        )
        shells = (4.0 / 3.0) * torch.pi * (rdf.edges[1:].pow(3) - rdf.edges[:-1].pow(3))
        density = (rdf.num_atoms / 2) / (_BINARY_CELLS * _BINARY_SPACING) ** 3
        assert float((rdf.g_r * shells).sum() * density) == pytest.approx(6.0)

    def test_a_species_the_frames_do_not_carry_is_rejected(self) -> None:
        """A partial over an absent species has no density to normalize against."""
        with pytest.raises(ValueError, match="no atom of one of the atomic numbers"):
            radial_distribution(
                _make_binary_lattice(alternate=True), r_max=5.0, pair=(11, 8)
            )

    def test_a_pair_that_is_not_two_species_is_rejected(self) -> None:
        """A partial is defined by exactly two atomic numbers."""
        with pytest.raises(ValueError, match="two atomic numbers"):
            radial_distribution(_make_binary_lattice(alternate=True), pair=(11,))

    def test_curves_resolved_differently_cannot_be_compared(self) -> None:
        """A total and a partial are different observables, not two measurements."""
        frames = _make_binary_lattice(alternate=True)
        total = radial_distribution(frames, r_max=5.0, num_bins=24)
        partial = radial_distribution(frames, r_max=5.0, num_bins=24, pair=(11, 17))
        with pytest.raises(ValueError, match="must resolve the same species"):
            compare_radial_distributions(total, partial)


class TestRadialDistributionComparison:
    """Scalar divergences between two pair correlation functions."""

    def test_a_curve_matches_itself_exactly(self) -> None:
        """Comparing a curve to itself gives zero on every measure."""
        rdf = radial_distribution(_build_lattice_batch(), r_max=5.0, num_bins=25)
        match = compare_radial_distributions(rdf, rdf)
        assert match.jensen_shannon == 0.0
        assert match.l1 == 0.0
        assert match.max_deviation == 0.0
        assert match.num_bins == 25

    def test_disjoint_histograms_reach_the_maximum_divergence(self) -> None:
        """Peaks in different bins have no overlap, which is one bit apart."""
        near = radial_distribution(_build_pair_batch(3.0), r_max=6.0, num_bins=12)
        far = radial_distribution(_build_pair_batch(5.0), r_max=6.0, num_bins=12)
        assert compare_radial_distributions(near, far).jensen_shannon == pytest.approx(
            1.0
        )

    def test_curves_binned_differently_cannot_be_compared(self) -> None:
        """Two curves must share bin edges before their bins mean the same thing."""
        coarse = radial_distribution(_build_pair_batch(3.0), r_max=6.0, num_bins=12)
        fine = radial_distribution(_build_pair_batch(3.0), r_max=6.0, num_bins=24)
        with pytest.raises(ValueError, match="share bin edges"):
            compare_radial_distributions(coarse, fine)

    def test_a_curve_with_no_pairs_cannot_be_compared(self) -> None:
        """An empty histogram has no distribution to diverge from."""
        populated = radial_distribution(_build_pair_batch(3.0), r_max=6.0, num_bins=12)
        empty = radial_distribution(
            _build_pair_batch(15.0, cell_length=60.0), r_max=6.0, num_bins=12
        )
        with pytest.raises(ValueError, match="must hold pairs"):
            compare_radial_distributions(populated, empty)


class TestRadialDistributionContinuity:
    """The curve follows the structure rather than the binning."""

    @pytest.mark.parametrize(
        "dtype", [torch.float32, torch.float64], ids=["float32", "float64"]
    )
    def test_a_rigid_translation_leaves_the_curve_where_it_was(
        self, dtype: torch.dtype
    ) -> None:
        """Shifting every atom by one vector moves the curve only by round-off."""
        rest = radial_distribution(
            _make_shifted_lattice((0.0, 0.0, 0.0), dtype), r_max=6.0
        )
        moved = radial_distribution(
            _make_shifted_lattice(_TRANSLATION, dtype), r_max=6.0
        )
        assert float(moved.counts.sum()) == pytest.approx(float(rest.counts.sum()))
        assert compare_radial_distributions(rest, moved).jensen_shannon < 1e-9

    def test_a_lattice_constant_sweep_never_leaps(self) -> None:
        """Straining a crystal in half-permille steps raises the divergence smoothly."""
        reference = radial_distribution(_make_fcc(_FCC_LATTICE), r_max=6.0)
        divergences = []
        for step in range(31):
            strained = _make_fcc(_FCC_LATTICE * (1.0 + 0.0005 * step))
            divergences.append(
                compare_radial_distributions(
                    reference, radial_distribution(strained, r_max=6.0)
                ).jensen_shannon
            )
        steps = [after - before for before, after in itertools.pairwise(divergences)]
        assert min(steps) >= 0.0
        assert max(steps) < 0.05

    def test_a_cutoff_past_the_cell_reaches_every_periodic_image(self) -> None:
        """A one-atom primitive cell gives the supercell's curve up to round-off."""
        primitive = radial_distribution(_make_primitive_fcc(_FCC_LATTICE), r_max=6.0)
        supercell = radial_distribution(_make_fcc(_FCC_LATTICE, cells=3), r_max=6.0)
        assert float((primitive.g_r - supercell.g_r).abs().max()) < 1e-11
