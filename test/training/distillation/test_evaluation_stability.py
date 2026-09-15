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
import math
from collections.abc import Sequence
from typing import Any

import pytest
import torch

from nvalchemi.data import AtomicData, Batch
from nvalchemi.dynamics.base import DynamicsStage
from nvalchemi.dynamics.integrators import NVE
from nvalchemi.hooks import DynamicsContext
from nvalchemi.hooks.neighbor_list import NeighborListHook
from nvalchemi.training.distillation.evaluation import (
    StabilityMetrics,
    StabilityMonitor,
    compare_radial_distributions,
    extensivity_error,
    radial_distribution,
    total_momentum,
)
from test.training.conftest import _build_batch
from test.training.distillation.conftest import (
    _build_lattice_batch,
    _build_lattice_data,
    _build_lj_teacher,
    _build_pair_batch,
)

_ARGON_MASS = 39.948
"""Mass carried by every atom of the shared lattice builder."""

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

_SWING = 0.06
"""Per-atom amplitude of the scripted energy oscillation, in eV."""

_SWING_PERIOD_FS = 500.0
"""Period of that oscillation, in fs."""

_SWING_SAMPLES = 16
"""Samples one period is recorded with, one per step."""

_SWING_PERIODS = 4
"""Whole periods the scripted oscillation covers."""


def _drive(monitor: StabilityMonitor, batch: Batch, energies: Sequence[float]) -> None:
    """Fire *monitor* once per scripted total energy, one step apart."""
    for step, energy in enumerate(energies):
        batch.energy = torch.full((batch.num_graphs, 1), energy)
        monitor(DynamicsContext(batch=batch, step_count=step), DynamicsStage.AFTER_STEP)


def _swing(*, closed: bool) -> list[float]:
    """Return a bounded per-atom oscillation as one total energy per step.

    The closed series is a cosine over whole periods, ending on the sample it
    started from; the open one is a sine over the same span, stopping one
    sample short of closing, which is what a window cut mid-oscillation looks
    like. Both swing by the same amplitude about the same mean.
    """
    samples = _SWING_PERIODS * _SWING_SAMPLES + (1 if closed else 0)
    wave = math.cos if closed else math.sin
    return [
        _LATTICE_ATOMS * (-1.0 + _SWING * wave(2.0 * math.pi * step / _SWING_SAMPLES))
        for step in range(samples)
    ]


def _make_geometry_only_batch() -> Batch:
    """Return the moving lattice with every field an NVE run needs but no energy.

    :meth:`~nvalchemi.dynamics.base.BaseDynamics.compute` allocates the output
    fields a batch arrives without, so a frame assembled this way reaches the
    monitor with nothing to record only while it is still unpropagated.
    """
    lattice = _build_lattice_data(speed=0.002, jitter=0.15)
    data = AtomicData(
        positions=lattice.positions,
        atomic_numbers=lattice.atomic_numbers,
        atomic_masses=lattice.atomic_masses,
        cell=lattice.cell,
        pbc=lattice.pbc,
        forces=torch.zeros_like(lattice.positions),
    )
    data.add_node_property("velocities", lattice.velocities)
    return Batch.from_data_list([data])


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


def _make_identified_batch(
    system_ids: Sequence[int], cells: Sequence[int] = (2, 2)
) -> Batch:
    """Return one lattice graph per entry of *system_ids*, tagged and sized to match."""
    structures = []
    for system_id, count in zip(system_ids, cells, strict=True):
        data = _build_lattice_data(cells=count)
        data.add_system_property("system_id", torch.tensor([[system_id]]))
        structures.append(data)
    return Batch.from_data_list(structures)


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


def _make_nve(model: object, monitor: StabilityMonitor | None = None) -> NVE:
    """Return an NVE integrator with the Lennard-Jones neighbor-list hook."""
    hooks = [
        NeighborListHook(
            config=model.model_config.neighbor_config,
            skin=1.0,
            stage=DynamicsStage.BEFORE_COMPUTE,
        )
    ]
    if monitor is not None:
        hooks.append(monitor)
    return NVE(model=model, dt=1.0, hooks=hooks)


class TestStabilityMonitor:
    """Drift and momentum metrics over a recorded trajectory."""

    def test_scripted_linear_drift_matches_the_analytic_rate(self) -> None:
        """A total energy rising by a fixed amount per step reports that slope."""
        monitor = StabilityMonitor(timestep_fs=2.0)
        batch = _build_lattice_batch()
        _drive(monitor, batch, [1.0 + 0.027 * step for step in range(11)])
        metrics = monitor.metrics()
        assert metrics.num_samples == 11
        assert metrics.energy_drift_per_atom == pytest.approx(0.27 / _LATTICE_ATOMS)
        assert metrics.energy_drift_per_atom_per_step == pytest.approx(0.001)
        assert metrics.energy_drift_per_atom_per_ns == pytest.approx(500.0)

    def test_kinetic_energy_is_included_by_default(self) -> None:
        """Only the kinetic-aware monitor sees a constant-potential run heating up."""
        batch = _build_lattice_batch()
        total = StabilityMonitor()
        potential = StabilityMonitor(include_kinetic=False)
        for step in range(2):
            batch.velocities = torch.full((batch.num_nodes, 3), 0.1 * step)
            batch.energy = torch.ones(1, 1)
            for monitor in (total, potential):
                monitor(
                    DynamicsContext(batch=batch, step_count=step),
                    DynamicsStage.AFTER_STEP,
                )
        assert potential.metrics().energy_drift_per_atom == 0.0
        assert total.metrics().energy_drift_per_atom == pytest.approx(
            0.5 * _ARGON_MASS * 3.0 * 0.1**2
        )

    def test_momentum_drift_matches_the_scripted_velocity_change(self) -> None:
        """Momentum drift is the total mass times the velocity it drifted by."""
        monitor = StabilityMonitor()
        batch = _build_lattice_batch()
        for step in range(3):
            batch.velocities = torch.zeros(batch.num_nodes, 3)
            batch.velocities[:, 0] = 0.25 * step
            batch.energy = torch.zeros(1, 1)
            monitor(
                DynamicsContext(batch=batch, step_count=step), DynamicsStage.AFTER_STEP
            )
        expected = _ARGON_MASS * _LATTICE_ATOMS * 0.5
        assert monitor.metrics().max_momentum_drift == pytest.approx(expected, rel=1e-5)

    def test_a_single_sample_cannot_be_scored(self) -> None:
        """One recorded frame gives no interval to measure drift over."""
        monitor = StabilityMonitor()
        _drive(monitor, _build_lattice_batch(), [1.0])
        with pytest.raises(ValueError, match="at least two recorded samples"):
            monitor.metrics()

    def test_the_metrics_accessor_stays_a_method(self) -> None:
        """``metrics`` is a method, so reading it uncalled is not the metrics."""
        monitor = StabilityMonitor()
        _drive(monitor, _build_lattice_batch(), [1.0, 2.0])
        assert not isinstance(StabilityMonitor.__dict__["metrics"], property)
        assert isinstance(monitor.metrics(), StabilityMetrics)
        assert not isinstance(monitor.metrics, StabilityMetrics)

    def test_drift_rate_is_omitted_without_a_timestep(self) -> None:
        """Steps become nanoseconds only when a timestep says how long one is."""
        monitor = StabilityMonitor()
        _drive(monitor, _build_lattice_batch(), [1.0, 2.0])
        assert monitor.metrics().energy_drift_per_atom_per_ns is None

    def test_changing_graph_count_stops_recording_and_warns(self) -> None:
        """A batch that graduated systems is not folded into the same series."""
        monitor = StabilityMonitor()
        _drive(monitor, _build_lattice_batch(), [1.0, 2.0])
        graduated = _build_lattice_batch()
        graduated = Batch.from_data_list(graduated.to_data_list() * 2)
        with pytest.warns(UserWarning, match="stopped recording"):
            monitor(
                DynamicsContext(batch=graduated, step_count=9), DynamicsStage.AFTER_STEP
            )
        assert monitor.metrics().num_samples == 2

    def test_a_refill_of_differently_sized_systems_stops_recording(self) -> None:
        """Same graph count, different atom counts, is still a different series."""
        monitor = StabilityMonitor()
        _drive(monitor, _make_identified_batch([0, 1]), [1.0, 2.0])
        refilled = _make_identified_batch([0, 1], cells=(2, 3))
        with pytest.warns(UserWarning, match="stopped recording"):
            monitor(
                DynamicsContext(batch=refilled, step_count=9), DynamicsStage.AFTER_STEP
            )
        assert monitor.metrics().num_samples == 2

    def test_a_shape_preserving_refill_stops_recording(self) -> None:
        """Fresh systems in the same slots break the series even at the same size."""
        monitor = StabilityMonitor()
        _drive(monitor, _make_identified_batch([0, 1]), [1.0, 2.0])
        refilled = _make_identified_batch([2, 3])
        with pytest.warns(UserWarning, match="stopped recording"):
            monitor(
                DynamicsContext(batch=refilled, step_count=9), DynamicsStage.AFTER_STEP
            )
        assert monitor.metrics().num_samples == 2

    def test_the_same_systems_keep_being_recorded(self) -> None:
        """An unchanged inflight batch is not mistaken for a refilled one."""
        monitor = StabilityMonitor()
        _drive(monitor, _make_identified_batch([0, 1]), [1.0, 2.0, 3.0])
        assert monitor.metrics().num_samples == 3

    def test_a_symmetric_excursion_fits_a_zero_drift_rate(self) -> None:
        """A run that heats up and cools back down is scored as no net drift."""
        monitor = StabilityMonitor(timestep_fs=1.0)
        _drive(monitor, _build_lattice_batch(), [0.0, 2.0, 3.0, 2.0, 0.0])
        metrics = monitor.metrics()
        assert metrics.energy_drift_per_atom == pytest.approx(0.0)
        assert metrics.energy_drift_per_atom_per_ns == pytest.approx(0.0, abs=1e-9)

    def test_a_closed_oscillation_is_only_seen_by_the_diagnostics(self) -> None:
        """Both drift figures read zero on a swing the fluctuation sizes exactly."""
        monitor = StabilityMonitor(timestep_fs=_SWING_PERIOD_FS / _SWING_SAMPLES)
        _drive(monitor, _build_lattice_batch(), _swing(closed=True))
        metrics = monitor.metrics()
        assert metrics.energy_drift_per_atom == pytest.approx(0.0, abs=1e-9)
        assert metrics.energy_drift_per_atom_per_ns == pytest.approx(0.0, abs=1e-9)
        assert metrics.energy_fluctuation_per_atom == pytest.approx(
            _SWING / math.sqrt(2.0), rel=0.02
        )
        assert metrics.max_energy_excursion_per_atom == pytest.approx(
            2.0 * _SWING, rel=1e-5
        )

    def test_the_fluctuation_does_not_move_with_where_the_window_ends(self) -> None:
        """The same swing fits a zero rate or a huge one; the fluctuation is fixed."""
        timestep = _SWING_PERIOD_FS / _SWING_SAMPLES
        closed = StabilityMonitor(timestep_fs=timestep)
        open_ended = StabilityMonitor(timestep_fs=timestep)
        _drive(closed, _build_lattice_batch(), _swing(closed=True))
        _drive(open_ended, _build_lattice_batch(), _swing(closed=False))
        cut = open_ended.metrics()
        assert cut.energy_drift_per_atom_per_ns > 1.0
        assert cut.energy_fluctuation_per_atom == pytest.approx(
            closed.metrics().energy_fluctuation_per_atom, rel=0.05
        )
        assert cut.max_energy_excursion_per_atom == pytest.approx(_SWING, rel=1e-5)

    def test_a_linear_ramp_has_nothing_to_fluctuate_about(self) -> None:
        """A series that is its own fit leaves no residual, and drifts by its rise."""
        monitor = StabilityMonitor(timestep_fs=1.0)
        _drive(
            monitor, _build_lattice_batch(), [1.0 + 0.027 * step for step in range(11)]
        )
        metrics = monitor.metrics()
        assert metrics.energy_fluctuation_per_atom == pytest.approx(0.0, abs=1e-6)
        assert metrics.max_energy_excursion_per_atom == pytest.approx(
            metrics.energy_drift_per_atom
        )

    def test_the_metrics_round_trip_through_an_export(self) -> None:
        """Every field, diagnostics included, survives to_dict and back."""
        monitor = StabilityMonitor(timestep_fs=1.0)
        _drive(monitor, _build_lattice_batch(), [1.0, 2.0, 4.0])
        metrics = monitor.metrics()
        assert StabilityMetrics.from_dict(metrics.to_dict()) == metrics

    def test_an_export_written_before_the_diagnostics_still_loads(self) -> None:
        """A dict lacking the two newer keys rebuilds with them unmeasured."""
        monitor = StabilityMonitor(timestep_fs=1.0)
        _drive(monitor, _build_lattice_batch(), [1.0, 2.0, 4.0])
        exported = monitor.metrics().to_dict()
        older = {
            key: value
            for key, value in exported.items()
            if key
            not in {"energy_fluctuation_per_atom", "max_energy_excursion_per_atom"}
        }
        restored = StabilityMetrics.from_dict(older)
        assert restored.energy_fluctuation_per_atom is None
        assert restored.max_energy_excursion_per_atom is None

    def test_a_geometry_only_batch_names_the_field_it_is_missing(self) -> None:
        """An unpropagated frame is refused by field name rather than sampled."""
        with pytest.raises(ValueError, match=r"carrying no \['energy'\]"):
            StabilityMonitor()(
                DynamicsContext(batch=_make_geometry_only_batch(), step_count=0),
                DynamicsStage.AFTER_STEP,
            )

    def test_a_propagated_geometry_only_batch_has_an_energy_to_record(self) -> None:
        """compute() allocates the output fields a seed batch was built without."""
        batch = _make_geometry_only_batch()
        _make_nve(_build_lj_teacher(), StabilityMonitor()).run(batch, n_steps=2)
        assert batch.energy is not None

    def test_an_equilibration_transient_hides_the_drift_that_follows_it(self) -> None:
        """Discarding the relaxation window recovers the rate the whole fit cancels."""
        rise = 0.03125
        relaxation = [1.0 + rise * (5 - step) for step in range(5)]
        heating = [1.0 + rise * step for step in range(6)]
        whole = StabilityMonitor(timestep_fs=1.0)
        equilibrated = StabilityMonitor(timestep_fs=1.0, warmup_steps=5)
        for monitor in (whole, equilibrated):
            _drive(monitor, _build_lattice_batch(), relaxation + heating)
        assert whole.metrics().energy_drift_per_atom == pytest.approx(0.0, abs=1e-9)
        assert whole.metrics().energy_drift_per_atom_per_ns == pytest.approx(
            0.0, abs=1e-6
        )
        metrics = equilibrated.metrics()
        assert metrics.first_step == 5
        assert metrics.num_samples == 6
        assert metrics.energy_drift_per_atom == pytest.approx(5 * rise / _LATTICE_ATOMS)
        assert metrics.energy_drift_per_atom_per_ns == pytest.approx(
            rise / _LATTICE_ATOMS * 1.0e6
        )

    def test_the_series_is_fingerprinted_from_the_first_recorded_sample(self) -> None:
        """A refill inside the warmup window is discarded, not treated as a break."""
        monitor = StabilityMonitor(warmup_steps=2)
        _drive(monitor, _make_identified_batch([0, 1]), [1.0, 2.0])
        refilled = _make_identified_batch([2, 3])
        for step in (2, 3):
            refilled.energy = torch.full((refilled.num_graphs, 1), float(step))
            monitor(
                DynamicsContext(batch=refilled, step_count=step),
                DynamicsStage.AFTER_STEP,
            )
        metrics = monitor.metrics()
        assert metrics.num_samples == 2
        assert metrics.first_step == 2

    def test_lattice_at_rest_holds_its_energy_through_an_nve_run(self) -> None:
        """A Lennard-Jones lattice at its minimum drifts by nothing measurable."""
        model = _build_lj_teacher()
        monitor = StabilityMonitor(frequency=2, timestep_fs=1.0)
        _make_nve(model, monitor).run(_build_lattice_batch(), n_steps=20)
        metrics = monitor.metrics()
        assert metrics.num_samples == 10
        assert metrics.energy_drift_per_atom_per_step < 1e-9
        assert metrics.max_momentum_drift < 1e-9

    def test_perturbed_lattice_conserves_energy_under_nve(self) -> None:
        """A moving, displaced lattice still conserves energy to MD tolerance."""
        model = _build_lj_teacher()
        monitor = StabilityMonitor(frequency=5, timestep_fs=1.0)
        _make_nve(model, monitor).run(
            _build_lattice_batch(speed=0.002, jitter=0.15), n_steps=50
        )
        assert monitor.metrics().energy_drift_per_atom_per_step < 1e-6

    def test_total_momentum_sums_mass_weighted_velocities_per_graph(self) -> None:
        """A batch at rest carries no momentum, one row per graph."""
        batch = _build_lattice_batch()
        torch.testing.assert_close(total_momentum(batch), torch.zeros(1, 3))
        batch.velocities = torch.ones(batch.num_nodes, 3)
        expected = torch.full((1, 3), _ARGON_MASS * _LATTICE_ATOMS)
        torch.testing.assert_close(total_momentum(batch), expected)


class TestExtensivity:
    """Energy scaling of a model across replicated cells."""

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

    def test_a_cutoff_past_half_the_supercell_stays_extensive(self) -> None:
        """The neighbor build enumerates every image, so a long cutoff is fine."""
        metrics = extensivity_error(
            _build_lj_teacher(cutoff=14.0), _build_lattice_batch(), repeats=(2, 2, 2)
        )
        assert metrics.max_error_per_atom == pytest.approx(0.0, abs=1e-6)

    def test_non_periodic_structures_are_rejected(self) -> None:
        """Replicating a cluster is not defined, so it raises instead."""
        with pytest.raises(ValueError, match="no cell"):
            extensivity_error(_build_lj_teacher(), _build_batch())

    @pytest.mark.parametrize(
        "repeats",
        [(2, 1), (0, 1, 1), (-1, 1, 1)],
        ids=["too-short", "zero", "negative"],
    )
    def test_invalid_repeat_counts_are_rejected(self, repeats: tuple[int, ...]) -> None:
        """Replication factors must be three positive integers."""
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
        """The neighbor list built to count pairs is rolled back afterwards."""
        batch = _build_lattice_batch()
        radial_distribution(batch, r_max=5.0, num_bins=25)
        assert "neighbor_list" not in batch
        assert "neighbor_matrix" not in batch

    def test_non_periodic_frames_are_rejected(self) -> None:
        """Without a cell there is no density to normalize against."""
        with pytest.raises(ValueError, match="no cell"):
            radial_distribution(_build_batch())

    def test_frames_whose_cell_encloses_no_volume_are_rejected(self) -> None:
        """A zero cell has no density, so two unrelated molecules would match."""
        data = _build_lattice_data()
        data.cell = torch.zeros(1, 3, 3)
        data.pbc = torch.zeros(1, 3, dtype=torch.bool)
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
