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
"""Acceptance verdicts and the Pareto table across a family of students.

A caller collects one :class:`StudentEvaluation` per candidate, states the bars
as :class:`AcceptanceThresholds`, and :func:`build_acceptance_report` returns an
:class:`AcceptanceReport` that renders as Rich tables and exports as a plain
dictionary. :func:`measured_bars` says which bars a partial measurement can
decide, and every measurement rebuilds from its own export, so a sweep can
evaluate each student in a separate job and assemble the report in a final one.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping, Sequence
from typing import Annotated, Any, Literal, TypeAlias, get_args

from pydantic import BaseModel, ConfigDict, Field
from rich import box
from rich.console import Group
from rich.table import Table

from nvalchemi.training.distillation.evaluation._export import _rebuild
from nvalchemi.training.distillation.evaluation.accuracy import (
    AccuracyMetrics,
    AccuracyQuantity,
)
from nvalchemi.training.distillation.evaluation.stability import (
    ExtensivityMetrics,
    RDFComparison,
    StabilityMetrics,
)
from nvalchemi.training.distillation.evaluation.throughput import ThroughputMetrics

__all__ = [
    "AcceptanceCheck",
    "AcceptanceReport",
    "AcceptanceThresholds",
    "BAR_FAMILIES",
    "MetricFamily",
    "StudentEvaluation",
    "StudentVerdict",
    "build_acceptance_report",
    "measured_bars",
]

_MISSING = "-"
"""Cell rendered where a student has no value for a column."""

_WEIGHT_SOURCES = ("ema", "raw")
"""Weight sets a student evaluation can record having been measured on."""

MetricFamily: TypeAlias = Literal[
    "accuracy",
    "stability",
    "throughput",
    "extensivity",
    "rdf",
    "baseline_accuracy",
]
"""Measurement slot of a :class:`StudentEvaluation` an acceptance bar reads."""


_STUDENT_SECTIONS: dict[MetricFamily, type] = {
    "accuracy": AccuracyMetrics,
    "stability": StabilityMetrics,
    "throughput": ThroughputMetrics,
    "extensivity": ExtensivityMetrics,
    "rdf": RDFComparison,
    "baseline_accuracy": AccuracyMetrics,
}
"""Measurement class behind each nested slot of a student evaluation."""


@dataclasses.dataclass(frozen=True)
class StudentEvaluation:
    """Everything measured about one candidate student.

    Only *name* and *accuracy* are required. Each remaining slot is a
    measurement a caller may or may not have run, and a bar aimed at an empty
    slot fails the student rather than passing it silently; *weights* is a
    note on where the numbers came from rather than a slot, and no bar reads it.

    Attributes
    ----------
    name : str
        Label the student is reported under.
    accuracy : AccuracyMetrics
        Held-out errors, from
        :func:`~nvalchemi.training.distillation.evaluation.evaluate_accuracy`.
    stability : StabilityMetrics | None
        Trajectory conservation metrics, from
        :meth:`~nvalchemi.training.distillation.evaluation.StabilityMonitor.metrics`.
    throughput : ThroughputMetrics | None
        Steady-state speed, from
        :func:`~nvalchemi.training.distillation.evaluation.measure_throughput`.
    extensivity : ExtensivityMetrics | None
        Energy-scaling error, from
        :func:`~nvalchemi.training.distillation.evaluation.extensivity_error`.
    rdf : RDFComparison | None
        Structural match against a reference trajectory, from
        :func:`~nvalchemi.training.distillation.evaluation.compare_radial_distributions`.
    baseline_accuracy : AccuracyMetrics | None
        The same accuracy evaluation run on an equal-size student trained from
        scratch, which the from-scratch gate compares against; a baseline whose
        graph and atom counts differ from *accuracy*'s fails the gate rather
        than being ratioed against it.
    num_parameters : int | None
        Parameter count, reported alongside the speed/accuracy trade-off.
    weights : Literal["ema", "raw"] | None
        Which of the student's weights the numbers were measured on, its
        EMA-averaged or its live ones. Only the caller that handed
        :func:`~nvalchemi.training.distillation.evaluation.evaluate_accuracy` a
        ``strategy.inference_model`` entry knows, so record it here; ``None``
        records nothing, which is not the same as ``"raw"``.

    Raises
    ------
    TypeError
        If a measurement slot holds anything but its own metrics class.
    ValueError
        If *weights* names neither of the two weight sets.
    """

    name: str
    accuracy: AccuracyMetrics
    stability: StabilityMetrics | None = None
    throughput: ThroughputMetrics | None = None
    extensivity: ExtensivityMetrics | None = None
    rdf: RDFComparison | None = None
    baseline_accuracy: AccuracyMetrics | None = None
    num_parameters: int | None = None
    weights: Literal["ema", "raw"] | None = None

    def __post_init__(self) -> None:
        """Reject a mistyped measurement slot or an unrecognized weights marker.

        The slots are read attribute by attribute much later, when the report
        is built, so an object of the wrong kind would otherwise surface as an
        ``AttributeError`` inside :func:`build_acceptance_report` rather than
        at the line that filled the slot.
        """
        for slot, metric in _STUDENT_SECTIONS.items():
            value = getattr(self, slot)
            if value is not None and not isinstance(value, metric):
                raise TypeError(
                    f"StudentEvaluation.{slot} must be a {metric.__name__} or "
                    f"None; got {value!r}. An accessor left uncalled, such as "
                    "StabilityMonitor.metrics rather than the metrics it "
                    "returns, is the usual cause."
                )
        if self.weights is not None and self.weights not in _WEIGHT_SOURCES:
            raise ValueError(
                f"StudentEvaluation.weights must be one of {list(_WEIGHT_SOURCES)!r} "
                f"or None; got {self.weights!r}."
            )

    def to_dict(self) -> dict[str, Any]:
        """Return the populated measurements and markers as plain dictionaries."""
        measured = {
            "name": self.name,
            "accuracy": self.accuracy.to_dict(),
            "stability": self.stability,
            "throughput": self.throughput,
            "extensivity": self.extensivity,
            "rdf": self.rdf,
            "baseline_accuracy": self.baseline_accuracy,
            "num_parameters": self.num_parameters,
            "weights": self.weights,
        }
        return {
            key: value.to_dict() if hasattr(value, "to_dict") else value
            for key, value in measured.items()
            if value is not None
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> StudentEvaluation:
        """Rebuild an evaluation, and its measurements, from a :meth:`to_dict` export.

        An entry taken straight out of :meth:`AcceptanceReport.to_dict` is
        accepted too: its ``verdict`` is dropped, since verdicts are formed from
        the thresholds of the report being built rather than carried between jobs.

        Raises
        ------
        ValueError
            If the export, or one of its nested measurements, carries a key the
            dataclass does not declare or omits a required one.
        """
        rebuilt = {key: value for key, value in data.items() if key != "verdict"}
        for key, metric in _STUDENT_SECTIONS.items():
            if rebuilt.get(key) is not None:
                rebuilt[key] = metric.from_dict(rebuilt[key])
        return _rebuild(cls, rebuilt)


class AcceptanceThresholds(BaseModel):
    """Bars a student has to clear to be accepted.

    Every bar defaults to ``None``, which means "do not test this". A bar that
    is set and has no matching measurement fails the student: an acceptance
    gate that silently skips the check it was asked for is worse than no gate.

    Examples
    --------
    >>> from nvalchemi.training.distillation.evaluation import AcceptanceThresholds
    >>> thresholds = AcceptanceThresholds(
    ...     max_energy_per_atom_mae=0.005,
    ...     max_forces_mae=0.05,
    ...     max_energy_drift_per_atom_per_ns=0.01,
    ...     min_atoms_per_second=1.0e6,
    ...     max_from_scratch_ratio=1.0,
    ... )
    >>> thresholds.max_from_scratch_ratio
    1.0
    """

    max_energy_per_atom_mae: Annotated[
        float | None,
        Field(default=None, gt=0, description="Largest accepted energy MAE per atom."),
    ] = None
    max_forces_mae: Annotated[
        float | None,
        Field(
            default=None,
            gt=0,
            description="Largest accepted force MAE per Cartesian component.",
        ),
    ] = None
    max_stress_mae: Annotated[
        float | None,
        Field(default=None, gt=0, description="Largest accepted stress MAE."),
    ] = None
    min_force_cosine: Annotated[
        float | None,
        Field(
            default=None,
            ge=-1.0,
            le=1.0,
            description=(
                "Smallest accepted magnitude-weighted cosine similarity between "
                "the student's and the teacher's force fields, read off "
                "force_cosine_aggregate rather than off the per-atom mean, which "
                "the holdout's near-zero forces dominate."
            ),
        ),
    ] = None
    max_energy_drift_per_atom_per_ns: Annotated[
        float | None,
        Field(
            default=None,
            gt=0,
            description="Largest accepted fitted energy drift rate, in eV/atom/ns.",
        ),
    ] = None
    max_energy_drift_per_atom_per_step: Annotated[
        float | None,
        Field(
            default=None,
            gt=0,
            description="Largest accepted energy drift per atom per step.",
        ),
    ] = None
    max_momentum_drift: Annotated[
        float | None,
        Field(
            default=None,
            gt=0,
            description=(
                "Largest accepted deviation of a graph's total momentum. Only "
                "meaningful under a momentum-conserving integrator: a stochastic "
                "thermostat exchanges momentum with its bath by design."
            ),
        ),
    ] = None
    max_extensivity_error_per_atom: Annotated[
        float | None,
        Field(
            default=None,
            gt=0,
            description="Largest accepted supercell energy-scaling error per atom.",
        ),
    ] = None
    max_rdf_jensen_shannon: Annotated[
        float | None,
        Field(
            default=None,
            gt=0,
            le=1.0,
            description=(
                "Largest accepted Jensen-Shannon divergence between the student's "
                "and the reference trajectory's pair-distance histograms. Blind "
                "to which species a pair joins unless the compared distributions "
                "were themselves resolved to one pair of species."
            ),
        ),
    ] = None
    min_atoms_per_second: Annotated[
        float | None,
        Field(default=None, gt=0, description="Smallest accepted throughput floor."),
    ] = None
    min_ns_per_day: Annotated[
        float | None,
        Field(
            default=None,
            gt=0,
            description="Smallest accepted simulated nanoseconds per day.",
        ),
    ] = None
    max_from_scratch_ratio: Annotated[
        float | None,
        Field(
            default=None,
            gt=0,
            description=(
                "Largest accepted ratio of the distilled student's error to the "
                "equal-size from-scratch student's on the same holdout, taken as "
                "the worst ratio over every accuracy metric the two share; the "
                "student passes when that ratio is at most the bar, so 1.0 demands "
                "a match and a value below 1.0 demands a margin."
            ),
        ),
    ] = None

    model_config = ConfigDict(extra="forbid")


@dataclasses.dataclass(frozen=True)
class _Bar:
    """Where one acceptance bar reaches the number it gates.

    *check* names the row the bar reports under and, unless *attribute*
    overrides it, the field it reads off the metrics object of its family; both
    are empty for the from-scratch bar, whose ratio spans two families and
    still declares what it reads so :func:`measured_bars` can answer for it.
    *quantities* are the accuracy quantities any one of which decides the bar,
    and *missing* is the detail reported when the family was supplied but the
    field it reads was not.
    """

    families: tuple[MetricFamily, ...]
    check: str = ""
    attribute: str = ""
    comparison: Literal["<=", ">="] = "<="
    quantities: tuple[AccuracyQuantity, ...] = ()
    missing: str = ""


_BARS: dict[str, _Bar] = {
    "max_energy_per_atom_mae": _Bar(
        ("accuracy",),
        "energy_per_atom_mae",
        quantities=("energy",),
        missing="the accuracy pass did not compare energy",
    ),
    "max_forces_mae": _Bar(
        ("accuracy",),
        "forces_mae",
        quantities=("forces",),
        missing="the accuracy pass did not compare forces",
    ),
    "max_stress_mae": _Bar(
        ("accuracy",),
        "stress_mae",
        quantities=("stress",),
        missing="the accuracy pass did not compare stress",
    ),
    "min_force_cosine": _Bar(
        ("accuracy",),
        "force_cosine_aggregate",
        comparison=">=",
        quantities=("forces",),
        missing="the accuracy pass did not compare forces",
    ),
    "max_energy_drift_per_atom_per_ns": _Bar(
        ("stability",),
        "energy_drift_per_atom_per_ns",
        missing="the trajectory was recorded without a timestep, so no rate was fitted",
    ),
    "max_energy_drift_per_atom_per_step": _Bar(
        ("stability",), "energy_drift_per_atom_per_step"
    ),
    "max_momentum_drift": _Bar(("stability",), "max_momentum_drift"),
    "max_extensivity_error_per_atom": _Bar(
        ("extensivity",), "extensivity_error_per_atom", "max_error_per_atom"
    ),
    "max_rdf_jensen_shannon": _Bar(("rdf",), "rdf_jensen_shannon", "jensen_shannon"),
    "min_atoms_per_second": _Bar(("throughput",), "atoms_per_second", comparison=">="),
    "min_ns_per_day": _Bar(
        ("throughput",),
        "ns_per_day",
        comparison=">=",
        missing="the propagator was timed without a timestep, so no rate was formed",
    ),
    "max_from_scratch_ratio": _Bar(
        ("accuracy", "baseline_accuracy"), quantities=("energy", "forces", "stress")
    ),
}
"""Every field of :class:`AcceptanceThresholds`, in the order checks are applied."""

BAR_FAMILIES: Mapping[str, frozenset[MetricFamily]] = {
    bar: frozenset(spec.families) for bar, spec in _BARS.items()
}
"""Measurement families each acceptance bar reads, keyed by threshold field."""


def measured_bars(
    *families: MetricFamily,
    accuracy_quantities: Sequence[AccuracyQuantity] | None = None,
) -> frozenset[str]:
    """Return the acceptance bars *families* hold enough measurements to decide.

    A bar counts as measured only when every family in its :data:`BAR_FAMILIES`
    entry was supplied, because :func:`build_acceptance_report` fails a student
    on a bar whose measurement is missing rather than skipping it;
    ``max_from_scratch_ratio`` therefore needs both ``"accuracy"`` and
    ``"baseline_accuracy"``. *accuracy_quantities* narrows further, since an
    accuracy pass fills only the quantities it compared and a holdout scored on
    energy alone leaves ``max_forces_mae`` as unfillable as no pass at all. Two
    bars carry a precondition no argument here can express and are reported on
    the strength of their family: ``max_energy_drift_per_atom_per_ns`` needs a
    :class:`~nvalchemi.training.distillation.evaluation.StabilityMonitor` built
    with ``timestep_fs`` and ``min_ns_per_day`` needs
    :func:`~nvalchemi.training.distillation.evaluation.measure_throughput`
    called with one; a check that falls to either says so.

    Parameters
    ----------
    *families : MetricFamily
        Slots of a :class:`StudentEvaluation` the caller fills. Naming none
        returns an empty set.
    accuracy_quantities : Sequence[AccuracyQuantity] | None, optional
        Quantities the accuracy pass compared. Default ``None`` (every
        quantity).

    Returns
    -------
    frozenset[str]
        Field names of :class:`AcceptanceThresholds` that may be set.

    Raises
    ------
    ValueError
        If a name is not a measurement family, or not an accuracy quantity.

    Examples
    --------
    >>> from nvalchemi.training.distillation.evaluation import measured_bars
    >>> sorted(measured_bars("accuracy"))
    ['max_energy_per_atom_mae', 'max_forces_mae', 'max_stress_mae', 'min_force_cosine']
    >>> sorted(measured_bars("accuracy", accuracy_quantities=["energy"]))
    ['max_energy_per_atom_mae']
    """
    supplied = frozenset(families)
    unknown = sorted(supplied - set(_STUDENT_SECTIONS))
    if unknown:
        raise ValueError(
            f"Unknown measurement families {unknown!r}; expected names from "
            f"{sorted(_STUDENT_SECTIONS)!r}."
        )
    known = frozenset(get_args(AccuracyQuantity))
    if accuracy_quantities is None:
        compared = known
    else:
        compared = frozenset(accuracy_quantities)
        unknown = sorted(compared - known)
        if unknown:
            raise ValueError(
                f"Unknown accuracy quantities {unknown!r}; expected names from "
                f"{sorted(known)!r}."
            )
    return frozenset(
        bar
        for bar, spec in _BARS.items()
        if frozenset(spec.families) <= supplied
        and (not spec.quantities or compared & frozenset(spec.quantities))
    )


@dataclasses.dataclass(frozen=True)
class AcceptanceCheck:
    """One threshold applied to one measurement.

    Attributes
    ----------
    name : str
        Metric the check reads.
    value : float | None
        Measured value, or ``None`` when the measurement is missing.
    limit : float | None
        Bar the value was compared against.
    comparison : {"<=", ">="}
        Direction the check passes in.
    passed : bool
        Whether the student cleared the bar.
    detail : str
        Why a check failed, when the reason is not the number itself.
    """

    name: str
    value: float | None
    limit: float | None
    comparison: Literal["<=", ">="]
    passed: bool
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Return every field as a plain dictionary."""
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class StudentVerdict:
    """Outcome of every check applied to one student.

    Attributes
    ----------
    name : str
        Student the verdict belongs to.
    accepted : bool
        ``True`` when every check passed. A student with no checks at all is
        accepted, since no bar was asked for.
    checks : tuple[AcceptanceCheck, ...]
        Checks in the order they were applied.
    """

    name: str
    accepted: bool
    checks: tuple[AcceptanceCheck, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return the verdict and its checks as plain dictionaries."""
        return {
            "name": self.name,
            "accepted": self.accepted,
            "checks": [check.to_dict() for check in self.checks],
        }


@dataclasses.dataclass(frozen=True)
class AcceptanceReport:
    """Verdicts, the Pareto front, and the exports a workflow logs.

    A terminal artifact rather than a streaming one, so it does not implement
    the :class:`~nvalchemi.hooks.Reporter` protocol; print it to a
    :class:`rich.console.Console` for the dashboard view and pass
    :meth:`scalars` to a :class:`~nvalchemi.hooks.TensorBoardReporter` or any
    scalar sink for the durable one.

    Attributes
    ----------
    thresholds : AcceptanceThresholds
        Bars the verdicts were formed against.
    evaluations : tuple[StudentEvaluation, ...]
        Measurements the report was built from, in the order supplied.
    verdicts : tuple[StudentVerdict, ...]
        One verdict per evaluation, aligned with ``evaluations``.
    pareto_front : tuple[str, ...]
        Names of the students no other student beats on both accuracy and
        speed.

    Examples
    --------
    >>> from rich.console import Console
    >>> from nvalchemi.training.distillation.evaluation import (
    ...     build_acceptance_report,
    ... )
    >>> report = build_acceptance_report(evaluations, thresholds)  # doctest: +SKIP
    >>> Console().print(report)  # doctest: +SKIP
    >>> report.accepted  # doctest: +SKIP
    True
    """

    thresholds: AcceptanceThresholds
    evaluations: tuple[StudentEvaluation, ...]
    verdicts: tuple[StudentVerdict, ...]
    pareto_front: tuple[str, ...]

    @property
    def accepted(self) -> bool:
        """Return whether every student cleared every bar it was given."""
        return all(verdict.accepted for verdict in self.verdicts)

    def to_dict(self) -> dict[str, Any]:
        """Return the whole report as nested plain dictionaries."""
        return {
            "accepted": self.accepted,
            "thresholds": self.thresholds.model_dump(exclude_none=True),
            "pareto_front": list(self.pareto_front),
            "students": [
                evaluation.to_dict() | {"verdict": verdict.to_dict()}
                for evaluation, verdict in zip(
                    self.evaluations, self.verdicts, strict=True
                )
            ],
        }

    def scalars(self) -> dict[str, float]:
        """Return a flat ``{student/group/metric: value}`` map of every number.

        Returns
        -------
        dict[str, float]
            Numeric metrics only, keyed for a scalar sink such as
            :class:`~nvalchemi.hooks.TensorBoardReporter`; verdicts appear as
            ``<student>/accepted`` with value ``1.0`` or ``0.0`` and a top-level
            number such as ``num_parameters`` as ``<student>/num_parameters``.
        """
        flat: dict[str, float] = {}
        for evaluation, verdict in zip(self.evaluations, self.verdicts, strict=True):
            flat[f"{evaluation.name}/accepted"] = float(verdict.accepted)
            for group, metrics in evaluation.to_dict().items():
                if not isinstance(metrics, dict):
                    if isinstance(metrics, (int, float)) and not isinstance(
                        metrics, bool
                    ):
                        flat[f"{evaluation.name}/{group}"] = float(metrics)
                    continue
                for key, value in metrics.items():
                    if isinstance(value, (int, float)) and not isinstance(value, bool):
                        flat[f"{evaluation.name}/{group}/{key}"] = float(value)
        return flat

    def __rich__(self) -> Group:
        """Render the verdict and Pareto tables as one renderable."""
        return Group(_verdict_table(self.verdicts), _pareto_table(self))


def _format(value: float | None) -> str:
    """Return a compact cell for an optional number."""
    return _MISSING if value is None else f"{value:.4g}"


def _finite(value: float | None) -> bool:
    """Return whether *value* is a number a bar can be decided from."""
    return value is not None and math.isfinite(value)


def _check(
    name: str,
    value: float | None,
    limit: float | None,
    comparison: Literal["<=", ">="],
    detail: str = "",
    missing: str = "not measured",
) -> AcceptanceCheck | None:
    """Return the check for one bar, or ``None`` when no bar was set.

    A missing measurement reports *missing*, a measured value carries *detail*,
    and a non-finite value fails on ``"not finite"``: a NaN would read as an
    ordinary miss and an infinity would pass every ``max_*`` bar.
    """
    if limit is None:
        return None
    if not _finite(value):
        return AcceptanceCheck(
            name=name,
            value=value,
            limit=limit,
            comparison=comparison,
            passed=False,
            detail=missing if value is None else "not finite",
        )
    passed = value <= limit if comparison == "<=" else value >= limit
    return AcceptanceCheck(
        name=name,
        value=value,
        limit=limit,
        comparison=comparison,
        passed=passed,
        detail=detail,
    )


def _baseline_check(
    evaluation: StudentEvaluation, thresholds: AcceptanceThresholds
) -> AcceptanceCheck | None:
    """Return the from-scratch gate: the student must match or beat its baseline.

    Every accuracy metric both share is ratioed and the worst ratio kept. A
    baseline scored on a different number of graphs or atoms fails this
    student's own check rather than the family's report, a baseline of exactly
    zero is unbeatable (a matching student ties at ``1.0``, any error fails at
    infinity), and a non-finite error on either side is no ratio at all.
    """
    limit = thresholds.max_from_scratch_ratio
    if limit is None:
        return None
    baseline = evaluation.baseline_accuracy
    if baseline is None:
        return AcceptanceCheck(
            name="from_scratch_ratio",
            value=None,
            limit=limit,
            comparison="<=",
            passed=False,
            detail="no from-scratch baseline supplied",
        )
    student_workload = (evaluation.accuracy.num_graphs, evaluation.accuracy.num_atoms)
    baseline_workload = (baseline.num_graphs, baseline.num_atoms)
    if student_workload != baseline_workload:
        return AcceptanceCheck(
            name="from_scratch_ratio",
            value=None,
            limit=limit,
            comparison="<=",
            passed=False,
            detail=(
                f"baseline scored {baseline_workload!r} against the student's "
                f"{student_workload!r} as (graphs, atoms)"
            ),
        )
    ratios: dict[str, float] = {}
    for field in ("energy_per_atom_mae", "forces_mae", "stress_mae"):
        error = getattr(evaluation.accuracy, field)
        reference = getattr(baseline, field)
        if error is None or reference is None:
            continue
        if not _finite(error) or not _finite(reference):
            ratios[field] = math.nan
        elif reference == 0.0:
            ratios[field] = 1.0 if error == 0.0 else math.inf
        else:
            ratios[field] = error / reference
    if not ratios:
        return AcceptanceCheck(
            name="from_scratch_ratio",
            value=None,
            limit=limit,
            comparison="<=",
            passed=False,
            detail="baseline shares no comparable accuracy metric",
        )
    unusable = sorted(field for field, ratio in ratios.items() if math.isnan(ratio))
    if unusable:
        return AcceptanceCheck(
            name="from_scratch_ratio",
            value=math.nan,
            limit=limit,
            comparison="<=",
            passed=False,
            detail=f"no finite ratio for {unusable!r}",
        )
    worst = max(ratios.values())
    return AcceptanceCheck(
        name="from_scratch_ratio",
        value=worst,
        limit=limit,
        comparison="<=",
        passed=worst <= limit,
        detail="worst error ratio against the equal-size from-scratch student",
    )


def _rdf_detail(comparison: RDFComparison | None) -> str:
    """Return which pair distribution an RDF bar was measured over."""
    if comparison is None or comparison.pair is None:
        return "species-blind total g(r)"
    return f"partial g(r) of atomic numbers {list(comparison.pair)!r}"


def _student_checks(
    evaluation: StudentEvaluation, thresholds: AcceptanceThresholds
) -> tuple[AcceptanceCheck, ...]:
    """Apply every bar in *thresholds* to one student's measurements.

    :data:`BAR_FAMILIES` locates the metrics object each bar reads, so a bar
    added to :class:`AcceptanceThresholds` without an entry is neither applied
    nor advertised. A bar whose family was measured but whose own number was
    not reports which quantity or timestep was missing.
    """
    candidates = []
    for bar, spec in _BARS.items():
        if not spec.check:
            continue
        family = spec.families[0]
        metrics = getattr(evaluation, family)
        candidates.append(
            _check(
                spec.check,
                None
                if metrics is None
                else getattr(metrics, spec.attribute or spec.check),
                getattr(thresholds, bar),
                spec.comparison,
                _rdf_detail(evaluation.rdf) if family == "rdf" else "",
                "not measured" if metrics is None else spec.missing or "not measured",
            )
        )
    candidates.append(_baseline_check(evaluation, thresholds))
    return tuple(check for check in candidates if check is not None)


def _pareto_front(evaluations: Sequence[StudentEvaluation]) -> tuple[str, ...]:
    """Return the students no other student beats on both accuracy and speed.

    Ranking needs two finite numbers, so a student with a non-finite error or
    rate is left off the front like one never timed: every comparison against
    a NaN is false, so it would head a front nothing can dominate it on.
    """
    points = [
        (
            evaluation.name,
            evaluation.accuracy.forces_mae,
            evaluation.throughput.atoms_per_second,
        )
        for evaluation in evaluations
        if evaluation.throughput is not None
        and _finite(evaluation.accuracy.forces_mae)
        and _finite(evaluation.throughput.atoms_per_second)
    ]
    front = []
    for name, error, speed in points:
        dominated = any(
            other_error <= error
            and other_speed >= speed
            and (other_error < error or other_speed > speed)
            for other_name, other_error, other_speed in points
            if other_name != name
        )
        if not dominated:
            front.append(name)
    return tuple(front)


def _verdict_table(verdicts: Sequence[StudentVerdict]) -> Table:
    """Build the per-check verdict table."""
    table = Table(title="Acceptance", box=box.SIMPLE_HEAD, expand=True)
    for column in ("Student", "Check", "Value", "Bar", "Result"):
        table.add_column(column)
    for verdict in verdicts:
        if not verdict.checks:
            table.add_row(verdict.name, "no bars set", _MISSING, _MISSING, "ACCEPT")
            continue
        for index, check in enumerate(verdict.checks):
            table.add_row(
                verdict.name if index == 0 else "",
                check.name if not check.detail else f"{check.name} ({check.detail})",
                _format(check.value),
                f"{check.comparison} {_format(check.limit)}",
                "pass" if check.passed else "FAIL",
            )
    return table


def _pareto_table(report: AcceptanceReport) -> Table:
    """Build the speed-versus-accuracy table across the student family.

    The verdict column is pinned unwrappable so a narrow console abbreviates a
    header rather than truncating the ``ACCEPT``/``REJECT`` a reader came for.
    """
    table = Table(title="Speed / accuracy", box=box.SIMPLE_HEAD, expand=True)
    for column in (
        "Student",
        "Params",
        "E/atom MAE",
        "F MAE",
        "Atoms/graphs",
        "atoms/s",
        "ns/day",
        "Pareto",
    ):
        table.add_column(column)
    table.add_column("Verdict", no_wrap=True)
    for evaluation, verdict in zip(report.evaluations, report.verdicts, strict=True):
        throughput = evaluation.throughput
        table.add_row(
            evaluation.name,
            _MISSING
            if evaluation.num_parameters is None
            else f"{evaluation.num_parameters:,}",
            _format(evaluation.accuracy.energy_per_atom_mae),
            _format(evaluation.accuracy.forces_mae),
            _MISSING
            if throughput is None
            else f"{throughput.num_atoms:,} / {throughput.num_graphs:,}",
            _format(None if throughput is None else throughput.atoms_per_second),
            _format(None if throughput is None else throughput.ns_per_day),
            "yes" if evaluation.name in report.pareto_front else "",
            "ACCEPT" if verdict.accepted else "REJECT",
        )
    return table


def build_acceptance_report(
    evaluations: Sequence[StudentEvaluation],
    thresholds: AcceptanceThresholds | None = None,
) -> AcceptanceReport:
    """Turn a family of student evaluations into verdicts and a Pareto front.

    Parameters
    ----------
    evaluations : Sequence[StudentEvaluation]
        One evaluation per candidate student. Names must be unique, since they
        key the report's exports.
    thresholds : AcceptanceThresholds | None, optional
        Bars to apply. Default ``None`` (no bars: every student is accepted and
        the report is a comparison table).

    Returns
    -------
    AcceptanceReport
        Verdicts aligned with *evaluations*, plus the Pareto front over force
        MAE and atoms per second.

    Raises
    ------
    ValueError
        If *evaluations* is empty, if two students share a name, if the students
        were not all scored on the same holdout, or if the students that carry
        a throughput measurement were not all measured on the same batch.

    Examples
    --------
    >>> from nvalchemi.training.distillation.evaluation import (
    ...     AcceptanceThresholds,
    ...     StudentEvaluation,
    ...     build_acceptance_report,
    ... )
    >>> report = build_acceptance_report(  # doctest: +SKIP
    ...     [StudentEvaluation(name="student-s", accuracy=metrics)],
    ...     AcceptanceThresholds(max_forces_mae=0.05),
    ... )
    >>> report.accepted  # doctest: +SKIP
    True
    """
    if not evaluations:
        raise ValueError("At least one student evaluation is required to report on.")
    names = [evaluation.name for evaluation in evaluations]
    if len(set(names)) != len(names):
        raise ValueError(f"Student names must be unique; got {names!r}.")
    resolved = thresholds if thresholds is not None else AcceptanceThresholds()
    holdouts = {
        (evaluation.accuracy.num_graphs, evaluation.accuracy.num_atoms)
        for evaluation in evaluations
    }
    if len(holdouts) > 1:
        raise ValueError(
            "Errors are comparable only across students scored on one holdout, "
            "which is what the Pareto front ranks them on; got different sets "
            f"{sorted(holdouts)!r} as (num_graphs, num_atoms). Re-run "
            "evaluate_accuracy for every student over the same held-out data."
        )
    workloads = {
        (evaluation.throughput.num_atoms, evaluation.throughput.num_graphs)
        for evaluation in evaluations
        if evaluation.throughput is not None
    }
    if len(workloads) > 1:
        raise ValueError(
            "Throughput scales with the batch it was measured on, so a family is "
            "comparable only when every student was timed on one; got different "
            f"batches {sorted(workloads)!r} as (num_atoms, num_graphs). Re-measure "
            "every student with measure_throughput on the same batch."
        )
    verdicts = []
    for evaluation in evaluations:
        checks = _student_checks(evaluation, resolved)
        verdicts.append(
            StudentVerdict(
                name=evaluation.name,
                accepted=all(check.passed for check in checks),
                checks=checks,
            )
        )
    return AcceptanceReport(
        thresholds=resolved,
        evaluations=tuple(evaluations),
        verdicts=tuple(verdicts),
        pareto_front=_pareto_front(evaluations),
    )
