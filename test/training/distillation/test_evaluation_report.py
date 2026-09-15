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
"""Tests for :mod:`nvalchemi.training.distillation.evaluation.report`."""

from __future__ import annotations

import dataclasses
import json
import math
from typing import Any, get_args

import pytest
from pydantic import ValidationError
from rich.console import Console

from nvalchemi.training.distillation.evaluation import (
    BAR_FAMILIES,
    AcceptanceReport,
    AcceptanceThresholds,
    AccuracyMetrics,
    AccuracyQuantity,
    DrafterMetrics,
    ExtensivityMetrics,
    MetricFamily,
    RDFComparison,
    StabilityMetrics,
    StabilityMonitor,
    StudentEvaluation,
    ThroughputMetrics,
    build_acceptance_report,
    measured_bars,
)

_FAMILIES: tuple[MetricFamily, ...] = get_args(MetricFamily)
"""Every measurement family, read off the alias rather than restated."""

_QUANTITIES: tuple[AccuracyQuantity, ...] = get_args(AccuracyQuantity)
"""Every accuracy quantity, read off the alias rather than restated."""

_QUANTITY_FIELDS: dict[AccuracyQuantity, tuple[tuple[str, float], ...]] = {
    "energy": (
        ("energy_mae", 0.01),
        ("energy_rmse", 0.02),
        ("energy_per_atom_mae", 0.001),
        ("energy_per_atom_rmse", 0.002),
    ),
    "forces": (
        ("forces_mae", 0.02),
        ("forces_rmse", 0.03),
        ("force_cosine_mean", 0.99),
        ("force_cosine_aggregate", 0.99),
    ),
    "stress": (("stress_mae", 0.004), ("stress_rmse", 0.005)),
    "atomic_energies": (("atomic_energy_mae", 0.001), ("atomic_energy_rmse", 0.002)),
}
"""Accuracy fields an evaluation of each quantity fills, and a value for each."""


_DERIVATION_CASES: list[
    tuple[tuple[MetricFamily, ...], tuple[AccuracyQuantity, ...]]
] = [
    (("accuracy",), _QUANTITIES),
    (("accuracy",), ("energy", "forces")),
    (("accuracy",), ("energy",)),
    (("accuracy",), ("atomic_energies",)),
    (("accuracy", "stability"), _QUANTITIES),
    (("accuracy", "throughput", "extensivity"), _QUANTITIES),
    (("accuracy", "baseline_accuracy"), _QUANTITIES),
    (("accuracy", "baseline_accuracy"), ("stress",)),
    (("accuracy", "baseline_accuracy"), ("atomic_energies",)),
    (("accuracy", "rdf", "drafter"), _QUANTITIES),
    (_FAMILIES, _QUANTITIES),
]
"""Measurement sets the advertised bars are checked against the report for."""


def _make_accuracy(
    name: str = "student", forces_mae: float = 0.02, energy_mae: float = 0.001
) -> AccuracyMetrics:
    """Return accuracy metrics with the two fields the gates read."""
    return AccuracyMetrics(
        name=name,
        num_graphs=4,
        num_atoms=40,
        energy_per_atom_mae=energy_mae,
        forces_mae=forces_mae,
        force_cosine_mean=0.99,
        force_cosine_aggregate=0.99,
    )


def _make_stability(drift: float = 0.001) -> StabilityMetrics:
    """Return stability metrics with a chosen per-nanosecond drift."""
    return StabilityMetrics(
        num_samples=10,
        first_step=0,
        last_step=90,
        energy_drift_per_atom=1e-4,
        energy_drift_per_atom_per_step=1e-6,
        energy_drift_per_atom_per_ns=drift,
        max_momentum_drift=1e-8,
        timestep_fs=1.0,
    )


def _make_throughput(atoms_per_second: float = 2.0e6) -> ThroughputMetrics:
    """Return throughput metrics with a chosen atoms-per-second rate."""
    return ThroughputMetrics(
        steps_per_second=atoms_per_second / 1000.0,
        atoms_per_second=atoms_per_second,
        ns_per_day=50.0,
        num_atoms=1000,
        num_graphs=1,
        warmup_steps=5,
        measured_steps=20,
        elapsed_seconds=0.5,
        device="cpu",
    )


def _make_rdf(
    jensen_shannon: float = 0.02, pair: tuple[int, int] | None = None
) -> RDFComparison:
    """Return a structural comparison at a chosen species resolution."""
    return RDFComparison(
        jensen_shannon=jensen_shannon,
        l1=0.3,
        max_deviation=0.1,
        num_bins=24,
        pair=pair,
    )


def _make_extensivity() -> ExtensivityMetrics:
    """Return an energy-scaling result over a doubled cell."""
    return ExtensivityMetrics(
        repeats=(2, 1, 1),
        num_graphs=2,
        max_error_per_atom=1e-6,
        mean_error_per_atom=5e-7,
        max_relative_error=1e-8,
    )


def _make_student(
    name: str = "student",
    forces_mae: float = 0.02,
    atoms_per_second: float = 2.0e6,
    **kwargs: object,
) -> StudentEvaluation:
    """Return a fully measured student evaluation with overridable slots."""
    return StudentEvaluation(
        name=name,
        accuracy=_make_accuracy(name, forces_mae=forces_mae),
        stability=_make_stability(),
        throughput=_make_throughput(atoms_per_second),
        **kwargs,
    )


def _make_student_on(name: str, **workload: int) -> StudentEvaluation:
    """Return a student whose speed was measured on a chosen ``(atoms, graphs)``."""
    return StudentEvaluation(
        name=name,
        accuracy=_make_accuracy(name),
        throughput=dataclasses.replace(_make_throughput(), **workload),
    )


def _render(report: AcceptanceReport, width: int = 200) -> str:
    """Return the report's Rich rendering as plain text at a chosen width."""
    console = Console(width=width, record=True, force_terminal=False)
    console.print(report)
    return console.export_text()


def _make_scoped_accuracy(
    name: str = "student", quantities: tuple[AccuracyQuantity, ...] = _QUANTITIES
) -> AccuracyMetrics:
    """Return accuracy metrics carrying only the fields *quantities* fill."""
    measured = {
        field: value
        for quantity in quantities
        for field, value in _QUANTITY_FIELDS[quantity]
    }
    return AccuracyMetrics(name=name, num_graphs=4, num_atoms=40, **measured)


def _measured_evaluation(
    families: tuple[MetricFamily, ...],
    quantities: tuple[AccuracyQuantity, ...] = _QUANTITIES,
) -> StudentEvaluation:
    """Return an evaluation whose filled slots are exactly *families*.

    Accuracy is filled whatever *families* says, since the dataclass requires
    it; the family set is therefore read as the optional slots measured on top
    of it, and *quantities* as how much of the accuracy pass was run.
    """
    return StudentEvaluation(
        name="student",
        accuracy=_make_scoped_accuracy(quantities=quantities),
        stability=_make_stability() if "stability" in families else None,
        throughput=_make_throughput() if "throughput" in families else None,
        extensivity=_make_extensivity() if "extensivity" in families else None,
        rdf=_make_rdf() if "rdf" in families else None,
        baseline_accuracy=(
            _make_scoped_accuracy("baseline", quantities)
            if "baseline_accuracy" in families
            else None
        ),
        drafter=DrafterMetrics(acceptance_rate=0.8) if "drafter" in families else None,
    )


def _probe_bar(bar: str) -> Any:
    """Return a valid value that moves acceptance bar *bar* off its default."""
    return True if AcceptanceThresholds.model_fields[bar].annotation is bool else 0.5


def _probe_thresholds(bar: str) -> AcceptanceThresholds:
    """Return thresholds stating exactly the check *bar* takes part in.

    ``from_scratch_margin`` is the from-scratch gate's limit rather than its
    switch, so it is probed with ``require_from_scratch_baseline`` turned on.
    Every other bar stands on its own.
    """
    if bar == "from_scratch_margin":
        return AcceptanceThresholds(
            from_scratch_margin=_probe_bar(bar), require_from_scratch_baseline=True
        )
    return AcceptanceThresholds(**{bar: _probe_bar(bar)})


def _bars_the_report_fills(
    families: tuple[MetricFamily, ...],
    quantities: tuple[AccuracyQuantity, ...] = _QUANTITIES,
) -> set[str]:
    """Return the bars ``build_acceptance_report`` decides from those measurements."""
    evaluation = _measured_evaluation(families, quantities)
    filled = set()
    for bar in AcceptanceThresholds.model_fields:
        try:
            report = build_acceptance_report([evaluation], _probe_thresholds(bar))
        except ValueError:
            continue
        checks = report.verdicts[0].checks
        if checks and all(check.value is not None for check in checks):
            filled.add(bar)
    return filled


class TestAcceptanceThresholds:
    """Validation of the bars themselves."""

    def test_every_bar_defaults_to_unset(self) -> None:
        """A default threshold set tests nothing and accepts everyone."""
        report = build_acceptance_report([_make_student()])
        assert report.accepted
        assert report.verdicts[0].checks == ()

    def test_unknown_bar_is_rejected(self) -> None:
        """The threshold model forbids fields it does not know how to check."""
        with pytest.raises(ValidationError):
            AcceptanceThresholds(max_dipole_mae=0.1)

    def test_negative_bar_is_rejected(self) -> None:
        """An error bar has to be a positive number."""
        with pytest.raises(ValidationError):
            AcceptanceThresholds(max_forces_mae=-1.0)


class TestMeasuredBars:
    """The table saying which measurements each acceptance bar needs."""

    def test_every_bar_declares_the_families_it_reads(self) -> None:
        """The table covers the threshold model exactly, so a new bar cannot slip in."""
        assert set(BAR_FAMILIES) == set(AcceptanceThresholds.model_fields)

    def test_every_family_is_a_measurement_slot_of_an_evaluation(self) -> None:
        """Families name the evaluation slots a bar reads, and nothing else."""
        slots = {field.name for field in dataclasses.fields(StudentEvaluation)}
        assert slots - set(_FAMILIES) == {"name", "num_parameters", "weights"}
        assert set().union(*BAR_FAMILIES.values()) == set(_FAMILIES)

    @pytest.mark.parametrize(
        ("families", "quantities"),
        _DERIVATION_CASES,
        ids=[
            f"{'+'.join(families)}/{'+'.join(quantities)}"
            for families, quantities in _DERIVATION_CASES
        ],
    )
    def test_the_advertised_bars_are_the_ones_the_report_fills(
        self,
        families: tuple[MetricFamily, ...],
        quantities: tuple[AccuracyQuantity, ...],
    ) -> None:
        """What is advertised as measured is what the report decides on a number."""
        assert measured_bars(
            *families, accuracy_quantities=quantities
        ) == _bars_the_report_fills(families, quantities)

    def test_the_accuracy_bars_are_the_four_a_holdout_pass_fills(self) -> None:
        """A holdout pass alone decides the accuracy bars and no others."""
        assert measured_bars("accuracy") == {
            "max_energy_per_atom_mae",
            "max_forces_mae",
            "max_stress_mae",
            "min_force_cosine",
        }

    def test_a_quantity_the_accuracy_pass_skipped_decides_no_bar(self) -> None:
        """A bar reads a quantity, so a pass that skipped it fills nothing."""
        assert measured_bars("accuracy", accuracy_quantities=["energy"]) == {
            "max_energy_per_atom_mae"
        }
        assert measured_bars("accuracy", accuracy_quantities=["forces"]) == {
            "max_forces_mae",
            "min_force_cosine",
        }
        assert measured_bars("accuracy", accuracy_quantities=[]) == frozenset()

    def test_a_quantity_that_is_not_one_is_rejected(self) -> None:
        """A misspelled quantity raises rather than narrowing to nothing."""
        with pytest.raises(ValueError, match="Unknown accuracy quantities"):
            measured_bars("accuracy", accuracy_quantities=["dipole"])

    def test_measuring_nothing_decides_nothing(self) -> None:
        """Every bar reads at least one measurement, so naming none decides none."""
        assert measured_bars() == frozenset()

    def test_two_families_decide_the_bars_of_both(self) -> None:
        """Families accumulate: neither hides nor unlocks the other's bars."""
        assert measured_bars("accuracy", "throughput") == measured_bars(
            "accuracy"
        ) | measured_bars("throughput")

    def test_the_from_scratch_gate_needs_both_families_it_compares(self) -> None:
        """The gate is a ratio, so neither side of it alone decides the bar."""
        gate = {"require_from_scratch_baseline", "from_scratch_margin"}
        assert not gate & measured_bars("accuracy")
        assert not gate & measured_bars("baseline_accuracy")
        assert gate <= measured_bars("accuracy", "baseline_accuracy")

    def test_the_from_scratch_gate_needs_one_quantity_the_two_share(self) -> None:
        """Any one comparable error decides the gate; a pass sharing none does not."""
        gate = {"require_from_scratch_baseline", "from_scratch_margin"}
        assert gate <= measured_bars(
            "accuracy", "baseline_accuracy", accuracy_quantities=["stress"]
        )
        assert not gate & measured_bars(
            "accuracy", "baseline_accuracy", accuracy_quantities=["atomic_energies"]
        )

    def test_the_drafter_bar_is_out_of_reach_of_the_suite(self) -> None:
        """Nothing here fills DrafterMetrics, so no suite measurement decides its bar."""
        suite = tuple(family for family in _FAMILIES if family != "drafter")
        assert "min_drafter_acceptance_rate" not in measured_bars(*suite)
        assert "min_drafter_acceptance_rate" in measured_bars("drafter")

    def test_a_family_that_is_not_one_is_rejected(self) -> None:
        """A misspelled family raises rather than quietly measuring nothing."""
        with pytest.raises(ValueError, match="Unknown measurement families"):
            measured_bars("accuracy", "speed")

    def test_measuring_everything_decides_every_bar(self) -> None:
        """With every slot filled, no bar is left without a number behind it."""
        assert measured_bars(*_FAMILIES) == set(AcceptanceThresholds.model_fields)


class TestAcceptanceVerdicts:
    """Per-student verdicts against configured bars."""

    def test_a_student_inside_every_bar_is_accepted(self) -> None:
        """Clearing accuracy, stability, and speed bars accepts the student."""
        report = build_acceptance_report(
            [_make_student()],
            AcceptanceThresholds(
                max_forces_mae=0.05,
                max_energy_per_atom_mae=0.005,
                min_force_cosine=0.95,
                max_energy_drift_per_atom_per_ns=0.01,
                min_atoms_per_second=1.0e6,
            ),
        )
        assert report.accepted
        assert all(check.passed for check in report.verdicts[0].checks)

    def test_a_student_outside_one_bar_is_rejected(self) -> None:
        """One failed check is enough to reject."""
        report = build_acceptance_report(
            [_make_student(forces_mae=0.2)],
            AcceptanceThresholds(max_forces_mae=0.05, min_atoms_per_second=1.0e6),
        )
        assert not report.accepted
        failed = [check for check in report.verdicts[0].checks if not check.passed]
        assert [check.name for check in failed] == ["forces_mae"]

    def test_a_bar_with_no_measurement_behind_it_fails(self) -> None:
        """An unmeasured metric fails its bar rather than skipping it."""
        report = build_acceptance_report(
            [StudentEvaluation(name="student", accuracy=_make_accuracy())],
            AcceptanceThresholds(min_atoms_per_second=1.0e6),
        )
        check = report.verdicts[0].checks[0]
        assert not check.passed
        assert check.detail == "not measured"
        assert check.value is None

    def test_a_bar_on_a_quantity_the_pass_skipped_names_the_quantity(self) -> None:
        """A holdout scored on energy alone says so rather than "not measured"."""
        report = build_acceptance_report(
            [
                StudentEvaluation(
                    name="student",
                    accuracy=_make_scoped_accuracy(quantities=("energy",)),
                )
            ],
            AcceptanceThresholds(max_forces_mae=0.05),
        )
        check = report.verdicts[0].checks[0]
        assert not check.passed
        assert check.detail == "the accuracy pass did not compare forces"

    def test_a_rate_no_timestep_could_form_is_told_from_no_measurement(self) -> None:
        """A trajectory recorded without a timestep is a different gap from no run."""
        untimed = dataclasses.replace(
            _make_stability(), energy_drift_per_atom_per_ns=None, timestep_fs=None
        )
        thresholds = AcceptanceThresholds(max_energy_drift_per_atom_per_ns=0.01)
        recorded = build_acceptance_report(
            [
                StudentEvaluation(
                    name="student", accuracy=_make_accuracy(), stability=untimed
                )
            ],
            thresholds,
        )
        unrecorded = build_acceptance_report(
            [StudentEvaluation(name="student", accuracy=_make_accuracy())], thresholds
        )
        assert "without a timestep" in recorded.verdicts[0].checks[0].detail
        assert unrecorded.verdicts[0].checks[0].detail == "not measured"

    def test_the_structure_bar_says_it_read_a_species_blind_curve(self) -> None:
        """A pooled g(r) is labelled as one, so the bar is not read as more."""
        report = build_acceptance_report(
            [_make_student(rdf=_make_rdf())],
            AcceptanceThresholds(max_rdf_jensen_shannon=0.1),
        )
        assert report.verdicts[0].checks[0].detail == "species-blind total g(r)"
        assert "species-blind" in _render(report)

    def test_the_structure_bar_names_the_species_pair_it_resolved(self) -> None:
        """A partial g_ab(r) is reported as the observable it actually is."""
        report = build_acceptance_report(
            [_make_student(rdf=_make_rdf(pair=(11, 17)))],
            AcceptanceThresholds(max_rdf_jensen_shannon=0.1),
        )
        check = report.verdicts[0].checks[0]
        assert check.detail == "partial g(r) of atomic numbers [11, 17]"
        assert check.passed

    def test_the_force_cosine_bar_reads_the_magnitude_weighted_alignment(self) -> None:
        """The bar reads the aggregate, not the mean the low-force tail dominates."""
        accuracy = AccuracyMetrics(
            name="student",
            num_graphs=2,
            num_atoms=54,
            forces_mae=0.004,
            force_cosine_mean=0.51,
            force_cosine_aggregate=0.97,
        )
        report = build_acceptance_report(
            [StudentEvaluation(name="student", accuracy=accuracy)],
            AcceptanceThresholds(min_force_cosine=0.9),
        )
        check = report.verdicts[0].checks[0]
        assert report.verdicts[0].accepted is True
        assert check.name == "force_cosine_aggregate"
        assert check.value == 0.97

    @pytest.mark.parametrize(
        ("thresholds", "name"),
        [
            (AcceptanceThresholds(max_forces_mae=0.02), "forces_mae"),
            (AcceptanceThresholds(min_atoms_per_second=2.0e6), "atoms_per_second"),
        ],
        ids=["at-the-maximum", "at-the-minimum"],
    )
    def test_a_measurement_sitting_exactly_on_its_bar_passes(
        self, thresholds: AcceptanceThresholds, name: str
    ) -> None:
        """Both directions are inclusive, so meeting a bar exactly clears it."""
        report = build_acceptance_report([_make_student()], thresholds)
        check = report.verdicts[0].checks[0]
        assert check.name == name
        assert check.value == check.limit
        assert check.passed is True

    def test_minimum_bars_compare_in_the_other_direction(self) -> None:
        """A throughput floor passes when the measurement is above it."""
        report = build_acceptance_report(
            [_make_student(atoms_per_second=5.0e5)],
            AcceptanceThresholds(min_atoms_per_second=1.0e6),
        )
        check = report.verdicts[0].checks[0]
        assert check.comparison == ">="
        assert not check.passed


class TestFromScratchGate:
    """The distilled student has to beat its equal-size from-scratch twin."""

    def test_a_student_beating_the_baseline_passes_the_gate(self) -> None:
        """Lower error than the from-scratch student clears the gate."""
        report = build_acceptance_report(
            [
                _make_student(
                    forces_mae=0.02,
                    baseline_accuracy=_make_accuracy("scratch", forces_mae=0.05),
                )
            ],
            AcceptanceThresholds(require_from_scratch_baseline=True),
        )
        check = report.verdicts[0].checks[0]
        assert check.name == "from_scratch_ratio"
        assert check.value == pytest.approx(1.0)
        assert check.passed

    def test_losing_on_any_shared_metric_fails_the_gate(self) -> None:
        """The worst shared metric decides, so winning on energy is not enough."""
        baseline = AccuracyMetrics(
            name="scratch",
            num_graphs=4,
            num_atoms=40,
            energy_per_atom_mae=0.01,
            forces_mae=0.01,
        )
        report = build_acceptance_report(
            [_make_student(forces_mae=0.02, baseline_accuracy=baseline)],
            AcceptanceThresholds(require_from_scratch_baseline=True),
        )
        assert report.verdicts[0].checks[0].value == pytest.approx(2.0)
        assert not report.accepted

    def test_a_demanded_margin_tightens_the_gate(self) -> None:
        """A margin below one demands the distilled student win by that factor."""
        report = build_acceptance_report(
            [
                _make_student(
                    forces_mae=0.02,
                    baseline_accuracy=_make_accuracy("scratch", forces_mae=0.025),
                )
            ],
            AcceptanceThresholds(
                require_from_scratch_baseline=True, from_scratch_margin=0.5
            ),
        )
        assert not report.accepted

    def test_a_missing_baseline_fails_the_gate(self) -> None:
        """The gate cannot be satisfied by simply not running the baseline."""
        report = build_acceptance_report(
            [_make_student()],
            AcceptanceThresholds(require_from_scratch_baseline=True),
        )
        check = report.verdicts[0].checks[0]
        assert not check.passed
        assert check.detail == "no from-scratch baseline supplied"

    def test_a_baseline_sharing_no_metric_fails_the_gate(self) -> None:
        """Two evaluations measured on different quantities cannot be compared."""
        baseline = AccuracyMetrics(name="scratch", num_graphs=4, num_atoms=40)
        report = build_acceptance_report(
            [_make_student(baseline_accuracy=baseline)],
            AcceptanceThresholds(require_from_scratch_baseline=True),
        )
        assert report.verdicts[0].checks[0].detail.startswith("baseline shares no")

    def test_the_gate_is_off_unless_it_is_asked_for(self) -> None:
        """A baseline that is present but unrequested produces no check."""
        report = build_acceptance_report(
            [_make_student(baseline_accuracy=_make_accuracy("scratch"))]
        )
        assert report.verdicts[0].checks == ()

    def test_a_baseline_with_no_error_at_all_is_unbeatable(self) -> None:
        """A zero baseline is a bar nothing clears, not a metric that drops out."""
        report = build_acceptance_report(
            [
                _make_student(
                    forces_mae=0.02,
                    baseline_accuracy=_make_accuracy("scratch", forces_mae=0.0),
                )
            ],
            AcceptanceThresholds(require_from_scratch_baseline=True),
        )
        check = report.verdicts[0].checks[0]
        assert check.value == math.inf
        assert not check.passed

    def test_two_students_with_no_error_tie_at_one(self) -> None:
        """Zero over zero is a tie: it clears the inclusive gate and fails a margin."""
        evaluation = StudentEvaluation(
            name="student",
            accuracy=_make_accuracy("student", forces_mae=0.0, energy_mae=0.0),
            baseline_accuracy=_make_accuracy("scratch", forces_mae=0.0, energy_mae=0.0),
        )
        tied = build_acceptance_report(
            [evaluation], AcceptanceThresholds(require_from_scratch_baseline=True)
        )
        demanded = build_acceptance_report(
            [evaluation],
            AcceptanceThresholds(
                require_from_scratch_baseline=True, from_scratch_margin=0.5
            ),
        )
        assert tied.verdicts[0].checks[0].value == pytest.approx(1.0)
        assert tied.accepted
        assert not demanded.accepted

    def test_a_baseline_scored_on_another_holdout_fails_however_good_it_is(
        self,
    ) -> None:
        """A ratio across two holdouts compares the sets, so the gate refuses it."""
        baseline = dataclasses.replace(
            _make_accuracy("scratch", forces_mae=0.001), num_graphs=8, num_atoms=90
        )
        report = build_acceptance_report(
            [_make_student(forces_mae=0.02, baseline_accuracy=baseline)],
            AcceptanceThresholds(require_from_scratch_baseline=True),
        )
        check = report.verdicts[0].checks[0]
        assert not check.passed
        assert "(8, 90)" in check.detail
        assert "(4, 40)" in check.detail


class TestNonFiniteMeasurements:
    """A number that is not one is failed on its own detail, never read as absent."""

    def test_a_nan_measurement_fails_its_bar_on_its_own_detail(self) -> None:
        """NaN fails every comparison, so it would otherwise read as an ordinary miss."""
        accuracy = dataclasses.replace(_make_accuracy(), forces_mae=math.nan)
        report = build_acceptance_report(
            [StudentEvaluation(name="student", accuracy=accuracy)],
            AcceptanceThresholds(max_forces_mae=0.05),
        )
        check = report.verdicts[0].checks[0]
        assert not check.passed
        assert check.detail == "not finite"
        assert math.isnan(check.value)

    def test_an_infinity_read_back_from_an_export_fails_its_bar_too(self) -> None:
        """``-inf`` sits under every maximum it is put to, so a bar cannot accept it."""
        accuracy = AccuracyMetrics.from_dict(
            _make_accuracy().to_dict() | {"forces_mae": -math.inf}
        )
        report = build_acceptance_report(
            [StudentEvaluation(name="student", accuracy=accuracy)],
            AcceptanceThresholds(max_forces_mae=0.05),
        )
        check = report.verdicts[0].checks[0]
        assert not check.passed
        assert check.detail == "not finite"

    @pytest.mark.parametrize(
        "field", ["energy_per_atom_mae", "forces_mae", "stress_mae"]
    )
    def test_a_nan_error_fails_the_from_scratch_gate_from_any_position(
        self, field: str
    ) -> None:
        """The worst ratio is a maximum, which a NaN slips through from most seats."""
        accuracy = dataclasses.replace(
            _make_accuracy(), **{"stress_mae": 0.004} | {field: math.nan}
        )
        report = build_acceptance_report(
            [
                StudentEvaluation(
                    name="student",
                    accuracy=accuracy,
                    baseline_accuracy=dataclasses.replace(
                        _make_accuracy("scratch"), stress_mae=0.004
                    ),
                )
            ],
            AcceptanceThresholds(require_from_scratch_baseline=True),
        )
        check = report.verdicts[0].checks[0]
        assert report.accepted is False
        assert math.isnan(check.value)
        assert check.detail == f"no finite ratio for {[field]!r}"

    def test_a_student_with_a_nan_error_is_left_off_the_front(self) -> None:
        """Nothing dominates a NaN, so an unguarded front would rank it first."""
        report = build_acceptance_report(
            [
                _make_student("finite", forces_mae=0.05, atoms_per_second=1.0e6),
                StudentEvaluation(
                    name="broken",
                    accuracy=dataclasses.replace(
                        _make_accuracy("broken"), forces_mae=math.nan
                    ),
                    throughput=_make_throughput(),
                ),
            ]
        )
        assert report.pareto_front == ("finite",)

    def test_a_family_with_no_finite_pair_has_an_empty_front(self) -> None:
        """The front ranks two numbers, so a family carrying none is on nobody's."""
        report = build_acceptance_report(
            [
                StudentEvaluation(
                    name=name,
                    accuracy=dataclasses.replace(
                        _make_accuracy(name), forces_mae=math.nan
                    ),
                    throughput=_make_throughput(),
                )
                for name in ("broken-a", "broken-b")
            ]
        )
        assert report.pareto_front == ()


class TestParetoTable:
    """Speed against accuracy across a family of students."""

    def test_dominated_students_are_left_off_the_front(self) -> None:
        """A student both slower and less accurate than another is dominated."""
        report = build_acceptance_report(
            [
                _make_student("small", forces_mae=0.04, atoms_per_second=4.0e6),
                _make_student("medium", forces_mae=0.02, atoms_per_second=2.0e6),
                _make_student("dominated", forces_mae=0.05, atoms_per_second=1.0e6),
            ]
        )
        assert report.pareto_front == ("small", "medium")

    def test_two_students_measured_alike_both_stay_on_the_front(self) -> None:
        """Domination needs a strict win somewhere, so a tie knocks nobody off."""
        report = build_acceptance_report(
            [
                _make_student("twin-a", forces_mae=0.02, atoms_per_second=2.0e6),
                _make_student("twin-b", forces_mae=0.02, atoms_per_second=2.0e6),
            ]
        )
        assert report.pareto_front == ("twin-a", "twin-b")

    def test_students_without_a_speed_measurement_are_not_ranked(self) -> None:
        """The trade-off needs both axes, so an unmeasured student is skipped."""
        report = build_acceptance_report(
            [
                _make_student("measured"),
                StudentEvaluation(name="accuracy-only", accuracy=_make_accuracy()),
            ]
        )
        assert report.pareto_front == ("measured",)

    def test_the_table_lists_every_student_with_its_verdict(self) -> None:
        """The rendered table carries every student, ranked or not."""
        report = build_acceptance_report(
            [
                _make_student("small", forces_mae=0.04, atoms_per_second=4.0e6),
                _make_student("large", forces_mae=0.2, atoms_per_second=1.0e6),
            ],
            AcceptanceThresholds(max_forces_mae=0.05),
        )
        rendered = _render(report)
        assert "small" in rendered
        assert "large" in rendered
        assert "REJECT" in rendered

    def test_a_default_width_console_never_crops_the_verdict(self) -> None:
        """Rich's default 80 columns abbreviate a header, not ``ACCEPT``."""
        report = build_acceptance_report(
            [
                _make_student("small", forces_mae=0.04, atoms_per_second=4.0e6),
                _make_student("large", forces_mae=0.01, atoms_per_second=1.0e6),
            ],
            AcceptanceThresholds(max_forces_mae=0.03),
        )
        rendered = _render(report, width=80)
        assert "ACCEPT" in rendered
        assert "REJECT" in rendered
        assert "1,000" in rendered


class TestHoldoutComparability:
    """Errors are ranked across the family, so they have to be one holdout's."""

    def test_students_scored_on_different_holdouts_are_rejected(self) -> None:
        """A front over two sets' errors ranks the sets rather than the students."""
        other = dataclasses.replace(_make_accuracy("other"), num_atoms=80)
        with pytest.raises(ValueError, match="scored on one holdout"):
            build_acceptance_report(
                [
                    _make_student("student"),
                    StudentEvaluation(name="other", accuracy=other),
                ]
            )

    def test_a_baseline_measured_elsewhere_does_not_count_as_a_holdout(self) -> None:
        """The family invariant reads each student's own pass, not its baseline's."""
        baseline = dataclasses.replace(_make_accuracy("scratch"), num_atoms=80)
        report = build_acceptance_report([_make_student(baseline_accuracy=baseline)])
        assert report.accepted


class TestThroughputComparability:
    """A speed column ranks students only when one batch produced every number."""

    def test_students_timed_on_different_atom_counts_are_rejected(self) -> None:
        """The rate scales with the batch, so two batches produce no ranking."""
        with pytest.raises(ValueError, match="different batches"):
            build_acceptance_report(
                [
                    _make_student_on("small", num_atoms=1000),
                    _make_student_on("large", num_atoms=64000),
                ]
            )

    def test_the_same_atoms_split_into_different_graph_counts_are_rejected(
        self,
    ) -> None:
        """Graph count moves the rate at a fixed atom count, so it is checked too."""
        with pytest.raises(ValueError, match="different batches"):
            build_acceptance_report(
                [
                    _make_student_on("one-graph", num_graphs=1),
                    _make_student_on("many-graphs", num_graphs=64),
                ]
            )

    def test_a_lone_student_is_compared_against_nothing(self) -> None:
        """A family of one has no second workload to disagree with."""
        report = build_acceptance_report([_make_student_on("only", num_atoms=64000)])
        assert report.pareto_front == ("only",)

    def test_an_unmeasured_student_does_not_count_as_a_second_workload(self) -> None:
        """A student with no throughput at all leaves the measured ones comparable."""
        report = build_acceptance_report(
            [
                _make_student_on("measured", num_atoms=64000),
                StudentEvaluation(name="unmeasured", accuracy=_make_accuracy()),
            ]
        )
        assert report.pareto_front == ("measured",)

    def test_the_pareto_table_names_the_workload_every_rate_was_measured_on(
        self,
    ) -> None:
        """The speed column carries the batch behind it, so it cannot be misread."""
        rendered = _render(
            build_acceptance_report([_make_student_on("only", num_atoms=64000)])
        )
        assert "Atoms/graphs" in rendered
        assert "64,000 / 1" in rendered


class TestDrafterRows:
    """The deferred speculative-MD rows of the report."""

    def test_drafter_rows_are_omitted_when_no_student_carries_them(self) -> None:
        """The speculative table is left out entirely, not rendered empty."""
        assert "Speculative MD" not in _render(
            build_acceptance_report([_make_student()])
        )

    def test_drafter_rows_render_once_metrics_are_supplied(self) -> None:
        """A student carrying drafter metrics gets its acceptance-rate row."""
        report = build_acceptance_report(
            [
                _make_student(
                    drafter=DrafterMetrics(
                        acceptance_rate=0.8, speculative_speedup=2.5, draft_steps=4
                    )
                )
            ]
        )
        rendered = _render(report)
        assert "Speculative MD" in rendered
        assert "0.8" in rendered

    def test_an_acceptance_rate_bar_is_checked_against_the_drafter(self) -> None:
        """The drafter bar behaves like every other minimum bar."""
        report = build_acceptance_report(
            [_make_student(drafter=DrafterMetrics(acceptance_rate=0.4))],
            AcceptanceThresholds(min_drafter_acceptance_rate=0.6),
        )
        assert not report.accepted
        assert report.verdicts[0].checks[0].name == "drafter_acceptance_rate"

    def test_a_mixed_family_gates_only_the_students_that_draft(self) -> None:
        """The bar skips the plain students of a sweep it was never aimed at."""
        report = build_acceptance_report(
            [
                _make_student("student-s"),
                _make_student("student-m", forces_mae=0.03),
                _make_student(
                    "drafter-xs", drafter=DrafterMetrics(acceptance_rate=0.8)
                ),
            ],
            AcceptanceThresholds(min_drafter_acceptance_rate=0.6, max_forces_mae=0.05),
        )
        assert [check.name for check in report.verdicts[0].checks] == ["forces_mae"]
        assert report.verdicts[2].checks[-1].name == "drafter_acceptance_rate"
        assert report.accepted
        assert report.scalars()["student-m/accepted"] == 1.0

    def test_a_drafter_below_the_bar_still_fails_in_a_mixed_family(self) -> None:
        """Scoping the bar to the drafters does not soften it for a drafter."""
        report = build_acceptance_report(
            [
                _make_student("student-s"),
                _make_student(
                    "drafter-xs", drafter=DrafterMetrics(acceptance_rate=0.4)
                ),
            ],
            AcceptanceThresholds(min_drafter_acceptance_rate=0.6),
        )
        assert report.verdicts[0].accepted
        assert not report.verdicts[1].accepted
        assert not report.accepted

    def test_a_drafter_bar_on_a_family_without_a_drafter_is_rejected(self) -> None:
        """A bar scoped away to nothing is a caller mistake, not a silent pass."""
        with pytest.raises(ValueError, match="no student of the family carries"):
            build_acceptance_report(
                [_make_student()],
                AcceptanceThresholds(min_drafter_acceptance_rate=0.6),
            )


class TestReportExports:
    """Plain-dictionary and scalar exports of a finished report."""

    def test_to_dict_carries_thresholds_students_and_verdicts(self) -> None:
        """The export is a plain structure with every measured section in it."""
        report = build_acceptance_report(
            [_make_student()], AcceptanceThresholds(max_forces_mae=0.05)
        )
        exported = report.to_dict()
        assert exported["accepted"] is True
        assert exported["thresholds"]["max_forces_mae"] == 0.05
        student = exported["students"][0]
        assert student["accuracy"]["forces_mae"] == 0.02
        assert student["throughput"]["num_atoms"] == 1000
        assert student["verdict"]["checks"][0]["name"] == "forces_mae"

    def test_scalars_are_flat_and_numeric(self) -> None:
        """Every exported scalar is a float keyed by student, group, and metric."""
        report = build_acceptance_report([_make_student(num_parameters=1234)])
        scalars = report.scalars()
        assert scalars["student/accepted"] == 1.0
        assert scalars["student/accuracy/forces_mae"] == 0.02
        assert scalars["student/stability/energy_drift_per_atom_per_ns"] == 0.001
        assert scalars["student/num_parameters"] == 1234.0
        assert all(isinstance(value, float) for value in scalars.values())
        assert "student/name" not in scalars

    def test_unmeasured_sections_are_left_out_of_the_export(self) -> None:
        """A student with only accuracy exports only accuracy."""
        report = build_acceptance_report(
            [StudentEvaluation(name="student", accuracy=_make_accuracy())]
        )
        assert "throughput" not in report.to_dict()["students"][0]

    def test_the_verdict_table_names_every_check(self) -> None:
        """Each applied bar becomes a row of the acceptance table."""
        report = build_acceptance_report(
            [_make_student()],
            AcceptanceThresholds(max_forces_mae=0.05, min_atoms_per_second=1.0e6),
        )
        rendered = _render(report)
        assert "forces_mae" in rendered
        assert "atoms_per_second" in rendered
        assert "ACCEPT" in rendered


class TestMeasurementRoundTrip:
    """Rebuilding evaluations from the exports separate jobs wrote."""

    def test_a_fully_measured_student_survives_a_json_round_trip(self) -> None:
        """Every nested measurement comes back as the object it was exported from."""
        student = _make_student(
            "student-l",
            extensivity=_make_extensivity(),
            rdf=_make_rdf(pair=(11, 17)),
            drafter=DrafterMetrics(acceptance_rate=0.8, draft_steps=4),
            baseline_accuracy=_make_accuracy("scratch"),
            num_parameters=1234,
        )
        rebuilt = StudentEvaluation.from_dict(json.loads(json.dumps(student.to_dict())))
        assert rebuilt == student
        assert rebuilt.extensivity.repeats == (2, 1, 1)
        assert rebuilt.rdf.pair == (11, 17)

    def test_an_unmeasured_slot_comes_back_unmeasured(self) -> None:
        """Fields the export drops rebuild as ``None`` rather than as zeros."""
        student = StudentEvaluation(name="student", accuracy=_make_accuracy())
        rebuilt = StudentEvaluation.from_dict(student.to_dict())
        assert rebuilt == student
        assert rebuilt.throughput is None
        assert rebuilt.accuracy.stress_mae is None

    def test_a_student_entry_of_a_report_rebuilds_without_its_verdict(self) -> None:
        """Verdicts are formed from the bars of the report being built, not carried."""
        report = build_acceptance_report(
            [_make_student()], AcceptanceThresholds(max_forces_mae=0.05)
        )
        exported = report.to_dict()["students"][0]
        assert StudentEvaluation.from_dict(exported) == report.evaluations[0]

    def test_a_family_aggregated_from_exports_decides_the_same_way(self) -> None:
        """One job per student and one report at the end reaches the live verdicts."""
        thresholds = AcceptanceThresholds(
            max_forces_mae=0.05, min_atoms_per_second=1.0e6
        )
        family = [
            _make_student("student-s"),
            _make_student("student-m", forces_mae=0.2),
        ]
        rebuilt = [
            StudentEvaluation.from_dict(json.loads(json.dumps(student.to_dict())))
            for student in family
        ]
        assert (
            build_acceptance_report(rebuilt, thresholds).to_dict()
            == build_acceptance_report(family, thresholds).to_dict()
        )

    def test_a_nonfinite_metric_stays_in_the_export_as_a_nan(self) -> None:
        """``None`` is what the export drops; a measured NaN keeps its key.

        Python's ``json`` writes and reads the bare ``NaN`` token, which is an
        extension to the format rather than part of it: a strict reader on the
        far side of the export rejects the file.
        """
        metrics = dataclasses.replace(_make_accuracy(), force_cosine_aggregate=math.nan)
        exported = metrics.to_dict()
        assert math.isnan(exported["force_cosine_aggregate"])
        rebuilt = AccuracyMetrics.from_dict(json.loads(json.dumps(exported)))
        assert math.isnan(rebuilt.force_cosine_aggregate)
        assert rebuilt.stress_mae is None

    def test_an_export_written_without_the_nonfinite_count_still_rebuilds(self) -> None:
        """The count defaults to zero, so a job that predates it still loads."""
        exported = _make_accuracy().to_dict()
        del exported["force_nonfinite_atoms"]
        assert AccuracyMetrics.from_dict(exported).force_nonfinite_atoms == 0

    def test_a_key_the_measurement_does_not_declare_is_rejected(self) -> None:
        """An export written by another version fails where it is read."""
        exported = _make_accuracy().to_dict() | {"energy_mape": 0.1}
        with pytest.raises(ValueError, match="cannot be rebuilt from a mapping"):
            AccuracyMetrics.from_dict(exported)

    def test_a_missing_required_field_is_rejected(self) -> None:
        """A truncated export cannot be rebuilt into a partly-defaulted object."""
        exported = _make_accuracy().to_dict()
        del exported["num_atoms"]
        with pytest.raises(ValueError, match="missing the required"):
            AccuracyMetrics.from_dict(exported)


class TestReportConstruction:
    """Guards on building a report at all."""

    def test_an_empty_family_is_rejected(self) -> None:
        """There is nothing to decide without a student."""
        with pytest.raises(ValueError, match="At least one student"):
            build_acceptance_report([])

    def test_duplicate_student_names_are_rejected(self) -> None:
        """Names key the exports, so two students cannot share one."""
        with pytest.raises(ValueError, match="must be unique"):
            build_acceptance_report([_make_student(), _make_student()])


class TestMeasurementSlots:
    """Guards on what a student evaluation is allowed to carry."""

    def test_an_uncalled_metrics_accessor_is_rejected(self) -> None:
        """``stability=monitor.metrics`` fails at the slot, not inside the report."""
        monitor = StabilityMonitor()
        with pytest.raises(TypeError, match="StudentEvaluation.stability"):
            StudentEvaluation(
                name="student", accuracy=_make_accuracy(), stability=monitor.metrics
            )

    def test_a_measurement_of_the_wrong_kind_is_rejected(self) -> None:
        """A speed measurement in the stability slot is caught on construction."""
        with pytest.raises(TypeError, match="must be a StabilityMetrics"):
            StudentEvaluation(
                name="student", accuracy=_make_accuracy(), stability=_make_throughput()
            )


class TestWeightsMarker:
    """The record of which of a student's weights the numbers were measured on."""

    def test_a_recorded_marker_survives_a_json_round_trip(self) -> None:
        """The marker rebuilds as the value it was exported under."""
        student = _make_student(weights="ema")
        rebuilt = StudentEvaluation.from_dict(json.loads(json.dumps(student.to_dict())))
        assert rebuilt == student
        assert rebuilt.weights == "ema"

    def test_an_unrecorded_marker_leaves_no_key_behind(self) -> None:
        """``None`` is dropped from the export and rebuilds as ``None``."""
        student = _make_student()
        exported = student.to_dict()
        assert "weights" not in exported
        rebuilt = StudentEvaluation.from_dict(json.loads(json.dumps(exported)))
        assert rebuilt == student
        assert rebuilt.weights is None

    def test_each_student_carries_its_own_marker_in_the_report(self) -> None:
        """The report export places each marker under the student it belongs to."""
        report = build_acceptance_report(
            [
                _make_student("averaged", weights="ema"),
                _make_student("live", weights="raw"),
            ]
        )
        markers = {
            student["name"]: student["weights"]
            for student in report.to_dict()["students"]
        }
        assert markers == {"averaged": "ema", "live": "raw"}

    def test_an_unknown_weight_set_is_rejected(self) -> None:
        """Only the two weight sets are recordable; a third fails on construction."""
        with pytest.raises(ValueError, match="StudentEvaluation.weights"):
            _make_student(weights="swa")

    def test_an_export_naming_an_unknown_weight_set_is_rejected(self) -> None:
        """A marker another version wrote fails where it is read."""
        exported = _make_student(weights="ema").to_dict() | {"weights": "swa"}
        with pytest.raises(ValueError, match="StudentEvaluation.weights"):
            StudentEvaluation.from_dict(exported)

    def test_the_marker_reaches_no_scalar_sink(self) -> None:
        """A recorded marker leaves the flat scalar export exactly as it was."""
        thresholds = AcceptanceThresholds(max_forces_mae=0.05)
        marked = build_acceptance_report([_make_student(weights="ema")], thresholds)
        plain = build_acceptance_report([_make_student()], thresholds)
        assert marked.scalars() == plain.scalars()
        assert "student/weights" not in marked.scalars()

    def test_the_marker_carries_no_verdict(self) -> None:
        """It is not a measurement family, so no bar can be aimed at it."""
        with pytest.raises(ValueError, match="Unknown measurement families"):
            measured_bars("weights")
