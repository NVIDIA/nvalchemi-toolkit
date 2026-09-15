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

This is where the accuracy, stability, and throughput measurements of the
sibling modules turn into a decision. A caller collects one
:class:`StudentEvaluation` per candidate, states the bars as
:class:`AcceptanceThresholds`, and gets back an :class:`AcceptanceReport` that
renders as a Rich table and exports as a plain dictionary. A caller that runs
only part of the suite reads the bars it may state off :func:`measured_bars`
instead of restating which measurement each bar needs.

Drafter acceptance-rate and effective speculative-speedup rows are part of the
report's shape but are not produced here: the metric that fills
:class:`DrafterMetrics` ships with the speculative-MD drafter objectives. Until
an evaluation carries one, those rows are omitted rather than reported empty,
and the matching threshold is the only line a caller has to add later. That
threshold is scoped to the drafters of a mixed family rather than applied to
every student in it, which is what makes the one line safe to add to a sweep
whose other students were never meant to draft.

Every measurement a report is built from also rebuilds from its own export, so
a sweep that evaluates each student in a separate job can persist the results
and assemble the report in a final one.
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
    "DrafterMetrics",
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
    "drafter",
]
"""Measurement slot of a :class:`StudentEvaluation` an acceptance bar reads."""


@dataclasses.dataclass(frozen=True)
class DrafterMetrics:
    """Speculative-MD rates of one drafter student.

    Populated by the drafter objectives rather than by this package; the
    acceptance report renders its rows only when an evaluation carries one.

    Attributes
    ----------
    acceptance_rate : float
        Fraction of drafted steps a verifier accepts.
    speculative_speedup : float | None
        End-to-end speedup of the draft-and-verify loop over verifier-only
        propagation, which is the acceptance rate discounted by the verifier's
        own cost.
    draft_steps : int | None
        Steps drafted between verifications.
    """

    acceptance_rate: float
    speculative_speedup: float | None = None
    draft_steps: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return every field as a plain dictionary."""
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> DrafterMetrics:
        """Rebuild the metrics from a :meth:`to_dict` export."""
        return _rebuild(cls, data)


_STUDENT_SECTIONS: dict[MetricFamily, type] = {
    "accuracy": AccuracyMetrics,
    "stability": StabilityMetrics,
    "throughput": ThroughputMetrics,
    "extensivity": ExtensivityMetrics,
    "rdf": RDFComparison,
    "baseline_accuracy": AccuracyMetrics,
    "drafter": DrafterMetrics,
}
"""Measurement class behind each nested slot of a student evaluation."""


@dataclasses.dataclass(frozen=True)
class StudentEvaluation:
    """Everything measured about one candidate student.

    Only *name* and *accuracy* are required. Each remaining slot is a
    measurement a caller may or may not have run, and a threshold aimed at a
    slot that is empty fails the student rather than passing it silently. The
    exception is *drafter*, which records what kind of student this is rather
    than a measurement every student could have run: a bar on it is checked
    against the students that carry it and skipped for the rest. *weights* is
    not a slot at all but a note on where the rest of the numbers came from,
    and no bar can name it.

    Attributes
    ----------
    name : str
        Label the student is reported under.
    accuracy : AccuracyMetrics
        Held-out errors, from
        :func:`~nvalchemi.training.distillation.evaluation.evaluate_accuracy`.
    stability : StabilityMetrics | None
        Trajectory conservation metrics, from a call to
        :meth:`~nvalchemi.training.distillation.evaluation.StabilityMonitor.metrics`.
    throughput : ThroughputMetrics | None
        Steady-state speed, from
        :func:`~nvalchemi.training.distillation.evaluation.measure_throughput`.
    extensivity : ExtensivityMetrics | None
        Energy-scaling error, from
        :func:`~nvalchemi.training.distillation.evaluation.extensivity_error`.
    rdf : RDFComparison | None
        Structural match against a reference trajectory, from the radial
        distribution comparison in
        :mod:`nvalchemi.training.distillation.evaluation.stability`.
    baseline_accuracy : AccuracyMetrics | None
        The same accuracy evaluation run on an equal-size student trained from
        scratch, which is what the from-scratch gate compares against. "The
        same" is enforced rather than assumed: a baseline whose graph and atom
        counts differ from *accuracy*'s fails the gate instead of being ratioed
        against it.
    drafter : DrafterMetrics | None
        Speculative-MD rates, when the student is a drafter.
    num_parameters : int | None
        Parameter count, reported alongside the speed/accuracy trade-off.
    weights : Literal["ema", "raw"] | None
        Which of the student's weights the numbers above were measured on:
        its EMA-averaged ones or the live ones it trained with. A record
        rather than a measurement — no bar reads it and no verdict moves
        with it — carried so that two exports of the same student say which
        artifact each one gated on. Nothing here can infer it: the caller
        that handed :func:`~nvalchemi.training.distillation.evaluation.evaluate_accuracy`
        a ``strategy.inference_model`` entry is the one who knows the
        averaged weights were swapped in. ``None`` records nothing, which is
        not the same as ``"raw"``.

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
    drafter: DrafterMetrics | None = None
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
            "drafter": self.drafter,
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

        Parameters
        ----------
        data : Mapping[str, Any]
            Export of one student. An entry taken straight out of an
            :meth:`AcceptanceReport.to_dict` is accepted too: its ``verdict``
            is dropped, since verdicts are formed from the thresholds of the
            report being built rather than carried between jobs.

        Returns
        -------
        StudentEvaluation
            Evaluation equal to the one the export came from.

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

    The one exception is a bar scoped to a capability rather than to a
    measurement — currently only *min_drafter_acceptance_rate*, which applies
    to the students of the family that carry drafter metrics. A plain student
    has no acceptance rate to measure, so failing it on that bar would be a
    category error rather than a caught omission. Since the capability is read
    off the metrics themselves, a drafter whose measurement failed to attach
    reads as a plain student — which is why the bar still cannot be satisfied
    by silence: :func:`build_acceptance_report` rejects it outright on a family
    in which no student is a drafter.

    Examples
    --------
    >>> from nvalchemi.training.distillation.evaluation import AcceptanceThresholds
    >>> thresholds = AcceptanceThresholds(
    ...     max_energy_per_atom_mae=0.005,
    ...     max_forces_mae=0.05,
    ...     max_energy_drift_per_atom_per_ns=0.01,
    ...     min_atoms_per_second=1.0e6,
    ...     require_from_scratch_baseline=True,
    ... )
    >>> thresholds.from_scratch_margin
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
    min_drafter_acceptance_rate: Annotated[
        float | None,
        Field(
            default=None,
            gt=0,
            le=1.0,
            description=(
                "Smallest accepted speculative-MD draft acceptance rate. Checked "
                "only against the evaluations of the family that carry drafter "
                "metrics, and rejected outright when none of them does."
            ),
        ),
    ] = None
    require_from_scratch_baseline: Annotated[
        bool,
        Field(
            default=False,
            description=(
                "Require every student to match or beat the equal-size "
                "from-scratch student its evaluation carries, over the same "
                "holdout that student was scored on."
            ),
        ),
    ] = False
    from_scratch_margin: Annotated[
        float,
        Field(
            default=1.0,
            gt=0,
            description=(
                "Largest accepted ratio of a distilled student's error to the "
                "from-scratch student's. Below ``1`` demands a margin."
            ),
        ),
    ] = 1.0

    model_config = ConfigDict(extra="forbid")


@dataclasses.dataclass(frozen=True)
class _Bar:
    """Where one acceptance bar reaches the number it gates.

    *check* names the row the bar reports under and, unless *attribute*
    overrides it, the field it reads off the metrics object of its single
    family. Both are empty for a bar whose gate is not one field of one family
    — currently the from-scratch pair, whose ratio divides two families'
    accuracy metrics field by field — and such a bar still declares what it
    reads, so :func:`measured_bars` can answer for it.

    *quantities* are the accuracy quantities the bar can be decided from, since
    an accuracy pass fills only the fields of the quantities it was asked to
    compare; any one of them is enough. *missing* is the detail a check reports
    when the family was supplied but the field it reads was not, which is a
    different omission from the family never having been measured. *scoped*
    marks the bar that reads a capability rather than a measurement: a student
    without the family is left unchecked instead of failed.
    """

    families: tuple[MetricFamily, ...]
    check: str = ""
    attribute: str = ""
    comparison: Literal["<=", ">="] = "<="
    quantities: tuple[AccuracyQuantity, ...] = ()
    missing: str = ""
    scoped: bool = False


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
    "min_drafter_acceptance_rate": _Bar(
        ("drafter",),
        "drafter_acceptance_rate",
        "acceptance_rate",
        ">=",
        scoped=True,
    ),
    "require_from_scratch_baseline": _Bar(
        ("accuracy", "baseline_accuracy"), quantities=("energy", "forces", "stress")
    ),
    "from_scratch_margin": _Bar(
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

    A bar counts as measured only when every family in its
    :data:`BAR_FAMILIES` entry was supplied, because
    :func:`build_acceptance_report` fails a student on a bar whose measurement
    is missing rather than skipping it. The from-scratch pair therefore needs
    both ``"accuracy"`` and ``"baseline_accuracy"``, since the gate is a ratio
    between the two; ``min_drafter_acceptance_rate`` needs ``"drafter"``, which
    this package never measures at all — :class:`DrafterMetrics` is filled by
    the speculative-MD drafter objectives.

    Naming a family is necessary but not always sufficient, which is what
    *accuracy_quantities* is for: an accuracy pass fills only the fields of the
    quantities it compared, so a holdout scored on energy alone leaves
    ``max_forces_mae`` as unfillable as no accuracy pass at all. Two bars carry
    a precondition no argument here can express, and are reported as measured
    on the strength of their family: ``max_energy_drift_per_atom_per_ns`` needs
    a :class:`~nvalchemi.training.distillation.evaluation.StabilityMonitor`
    built with ``timestep_fs``, and ``min_ns_per_day`` needs
    :func:`~nvalchemi.training.distillation.evaluation.measure_throughput`
    called with one. A check that falls to either says so rather than reporting
    the measurement missing.

    A caller that measures only part of the suite should read the bars it may
    accept off this function rather than restate the mapping. The
    ``distill evaluate`` command is the one in the tree: it runs the holdout
    pass and nothing else, so the bars a recipe may carry are
    ``measured_bars("accuracy", accuracy_quantities=spec.quantities)`` — passing
    the quantities matters, since they are what the recipe chose to compare —
    and a bar added to :class:`AcceptanceThresholds` cannot then go silently
    unrefused.

    Parameters
    ----------
    *families : MetricFamily
        Slots of a :class:`StudentEvaluation` the caller fills. Naming none
        returns an empty set, since every bar reads at least one measurement.
    accuracy_quantities : Sequence[AccuracyQuantity] | None, optional
        Quantities the accuracy pass compared, which narrows the bars the
        ``"accuracy"`` family decides. Default ``None`` (every quantity).

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

    The report is a terminal artifact rather than a streaming one, so it does
    not implement the :class:`~nvalchemi.hooks.Reporter` protocol that
    :class:`~nvalchemi.hooks.ReportingOrchestrator` drives during a run.
    It plugs into the same stack from the other end: print it to a
    :class:`rich.console.Console` for the dashboard view, and pass
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
            :class:`~nvalchemi.hooks.TensorBoardReporter`. Verdicts appear as
            ``<student>/accepted`` with value ``1.0`` or ``0.0``, and a
            top-level measurement such as ``num_parameters`` as
            ``<student>/num_parameters`` — the size axis of the trade-off a
            sweep plots the other two against.
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
        """Render the verdict, Pareto, and drafter tables as one renderable."""
        tables = [_verdict_table(self.verdicts), _pareto_table(self)]
        drafter = _drafter_table(self.evaluations)
        if drafter is not None:
            tables.append(drafter)
        return Group(*tables)


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

    *detail* labels what a measured value was measured over, for the bars whose
    number does not say it. A missing measurement reports *missing* instead,
    which is what lets a bar separate a measurement nobody took from one taken
    without the argument the bar's own number needs.

    A non-finite measurement fails on its own detail rather than on either of
    those, since it is neither missing nor a number: a NaN fails every
    comparison and would read as an ordinary miss, and an infinity passes every
    ``max_*`` bar it is put to.
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

    The gate compares every accuracy metric both students share and keeps the
    worst ratio, so a student that wins on energy and loses on forces fails. A
    ratio is only meaningful between two passes over the same holdout, so a
    baseline that scored a different number of graphs or atoms fails the check
    rather than being divided into. The failure is the student's own, unlike
    the family-wide throughput invariant :func:`build_acceptance_report`
    raises on: one stale baseline should not cost the other students their
    report.

    A baseline that is exactly zero on a metric is unbeatable rather than
    absent — the student matching it clears the gate at ``1.0`` and any error
    at all fails at infinity — and a non-finite error on either side is no
    ratio at all.
    """
    if not thresholds.require_from_scratch_baseline:
        return None
    margin = thresholds.from_scratch_margin
    baseline = evaluation.baseline_accuracy
    if baseline is None:
        return AcceptanceCheck(
            name="from_scratch_ratio",
            value=None,
            limit=margin,
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
            limit=margin,
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
            limit=margin,
            comparison="<=",
            passed=False,
            detail="baseline shares no comparable accuracy metric",
        )
    unusable = sorted(field for field, ratio in ratios.items() if math.isnan(ratio))
    if unusable:
        return AcceptanceCheck(
            name="from_scratch_ratio",
            value=math.nan,
            limit=margin,
            comparison="<=",
            passed=False,
            detail=f"no finite ratio for {unusable!r}",
        )
    worst = max(ratios.values())
    return AcceptanceCheck(
        name="from_scratch_ratio",
        value=worst,
        limit=margin,
        comparison="<=",
        passed=worst <= margin,
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

    :data:`BAR_FAMILIES` is what locates the metrics object each bar reads, so
    a bar added to :class:`AcceptanceThresholds` without an entry in the table
    is neither applied here nor reported by :func:`measured_bars`. A bar whose
    family was measured but whose own number was not says which of the two
    happened, since a quantity the accuracy pass skipped and a rate no timestep
    could form are omissions a caller fixes differently.

    The drafter bar is scoped to the students that carry drafter metrics: it
    reads a capability rather than a measurement, so a plain student is left
    unchecked instead of failed. Every other bar keeps the fail-on-missing
    policy, and :func:`build_acceptance_report` is what stops a drafter bar
    from being scoped away to nothing.
    """
    candidates = []
    for bar, spec in _BARS.items():
        if not spec.check:
            continue
        family = spec.families[0]
        metrics = getattr(evaluation, family)
        if metrics is None and spec.scoped:
            continue
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

    Ranking needs two finite numbers, so a student carrying a non-finite error
    or rate is left off the front for the same reason one that was never timed
    is. Placing it would be the alternative rather than a neutral one: every
    comparison against a NaN is false, so such a point is dominated by nobody
    and would head a front it cannot even be compared to. A family in which no
    student carries both numbers therefore has an empty front.
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

    Nine columns do not fit rich's default 80-column console, and rich pays for
    the overflow by cropping the widest cells. The verdict column is pinned
    unwrappable so that a narrow console abbreviates a header rather than
    truncating the ``ACCEPT``/``REJECT`` a reader came for.
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


def _drafter_table(evaluations: Sequence[StudentEvaluation]) -> Table | None:
    """Build the speculative-MD table, or ``None`` when no student is a drafter."""
    drafters = [
        evaluation for evaluation in evaluations if evaluation.drafter is not None
    ]
    if not drafters:
        return None
    table = Table(title="Speculative MD", box=box.SIMPLE_HEAD, expand=True)
    for column in ("Student", "Acceptance rate", "Speculative speedup", "Draft steps"):
        table.add_column(column)
    for evaluation in drafters:
        drafter = evaluation.drafter
        table.add_row(
            evaluation.name,
            _format(drafter.acceptance_rate),
            _format(drafter.speculative_speedup),
            _MISSING if drafter.draft_steps is None else str(drafter.draft_steps),
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
        were not all scored on the same holdout, if the students that carry a
        throughput measurement were not all measured on the same batch, or if
        ``min_drafter_acceptance_rate`` is set on a family in which no student
        carries drafter metrics.

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
    if resolved.min_drafter_acceptance_rate is not None and all(
        evaluation.drafter is None for evaluation in evaluations
    ):
        raise ValueError(
            "min_drafter_acceptance_rate was set but no student of the family "
            f"carries drafter metrics; got {names!r}. The bar is checked against "
            "the drafters of a mixed family and skipped for the plain students, "
            "so a family with no drafter in it would leave the bar unchecked. "
            "Attach DrafterMetrics to the drafter, or drop the bar."
        )
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
