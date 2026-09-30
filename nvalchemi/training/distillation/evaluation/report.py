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

A caller collects one :class:`StudentEvaluation` per candidate student and sets
the acceptance bars on an :class:`AcceptanceThresholds`.
:func:`build_acceptance_report` then returns an :class:`AcceptanceReport`, which
renders as Rich tables and exports as a plain dictionary. :func:`measured_bars`
reports which bars a partial set of measurements can decide. Every measurement
rebuilds from its own export, so a sweep can evaluate each student in a
separate job and assemble the report in a final job.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping, Sequence
from typing import Annotated, Any, Literal, TypeAlias, get_args

from pydantic import BaseModel, ConfigDict, Field, model_validator
from rich import box
from rich.console import Group
from rich.table import Table

from nvalchemi.training.distillation.evaluation._export import MeasurementRecord
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
    "AcceptanceBar",
    "AcceptanceCheck",
    "AcceptanceReport",
    "AcceptanceThresholds",
    "BAR_FAMILIES",
    "DEFAULT_BARS",
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

_EXTRA_FAMILY_PREFIX = "extra:"
"""Prefix naming a family a bar reads out of ``StudentEvaluation.extra``."""

_COMPARISONS = ("<=", ">=")
"""Directions an acceptance bar can pass in."""

MetricFamily: TypeAlias = Literal[
    "accuracy",
    "stability",
    "throughput",
    "extensivity",
    "rdf",
    "baseline_accuracy",
]
"""Measurement slot of a :class:`StudentEvaluation` an acceptance bar reads."""


_STUDENT_SECTIONS: dict[MetricFamily, type[MeasurementRecord]] = {
    "accuracy": AccuracyMetrics,
    "stability": StabilityMetrics,
    "throughput": ThroughputMetrics,
    "extensivity": ExtensivityMetrics,
    "rdf": RDFComparison,
    "baseline_accuracy": AccuracyMetrics,
}
"""Measurement class behind each nested slot of a student evaluation."""


class StudentEvaluation(MeasurementRecord):
    """Everything measured about one candidate student.

    Only *name* and *accuracy* are required. Each remaining slot holds a
    measurement that a caller may or may not have run. A bar aimed at an empty
    slot fails the student rather than passing it silently. *weights* is not a
    slot: it notes where the numbers came from, and no bar reads it.

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
        scratch. The from-scratch bar compares the student against it. A
        baseline whose graph and atom counts differ from those of *accuracy*
        fails the bar rather than being used for a ratio.
    num_parameters : int | None
        Parameter count, reported alongside the speed/accuracy trade-off.
    weights : Literal["ema", "raw"] | None
        Which of the student's weights the numbers were measured on: its
        EMA-averaged weights (``"ema"``) or its live ones (``"raw"``). Only the
        caller knows whether it passed
        :func:`~nvalchemi.training.distillation.evaluation.evaluate_accuracy` a
        ``strategy.inference_model``, so record it here. ``None`` records
        nothing, which is not the same as ``"raw"``.
    extra : Mapping[str, Mapping[str, float]]
        Measurements outside the typed slots, as one flat map of numbers per
        family. A custom :class:`AcceptanceBar` reads a family through
        ``"extra:<family>"``. Default empty.

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
    extra: dict[str, dict[str, float]] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _check_slots(cls, data: Any) -> Any:
        """Reject a mistyped measurement slot or an unrecognized weights marker.

        The slots are read attribute by attribute much later, when the report
        is built. Without this check, an object of the wrong kind would surface
        as an ``AttributeError`` inside :func:`build_acceptance_report` rather
        than at the line that filled the slot. A mapping is let through for
        field validation to rebuild.
        """
        if not isinstance(data, Mapping):
            return data
        for slot, metric in _STUDENT_SECTIONS.items():
            value = data.get(slot)
            if value is not None and not isinstance(value, (metric, Mapping)):
                raise TypeError(
                    f"StudentEvaluation.{slot} must be a {metric.__name__} or "
                    f"None; got {value!r}. An accessor left uncalled, such as "
                    "StabilityMonitor.metrics rather than the metrics it "
                    "returns, is the usual cause."
                )
        weights = data.get("weights")
        if weights is not None and weights not in _WEIGHT_SOURCES:
            raise ValueError(
                f"StudentEvaluation.weights must be one of {list(_WEIGHT_SOURCES)!r} "
                f"or None; got {weights!r}."
            )
        return data

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
            "extra": self.extra or None,
        }
        return {
            key: value.to_dict() if hasattr(value, "to_dict") else value
            for key, value in measured.items()
            if value is not None
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> StudentEvaluation:
        """Rebuild an evaluation, and its measurements, from a :meth:`to_dict` export.

        An entry taken straight out of :meth:`AcceptanceReport.to_dict` is also
        accepted. Its ``verdict`` is dropped, because verdicts are formed from
        the thresholds of the report being built, not carried between jobs.

        Raises
        ------
        ValueError
            If the export, or one of its nested measurements, carries a key the
            record does not declare or omits a required one.
        """
        rebuilt = {key: value for key, value in data.items() if key != "verdict"}
        for key, metric in _STUDENT_SECTIONS.items():
            if rebuilt.get(key) is not None:
                rebuilt[key] = metric.from_dict(rebuilt[key])
        return super().from_dict(rebuilt)


class AcceptanceThresholds(BaseModel):
    """Acceptance bars a student has to clear to be accepted.

    Every bar defaults to ``None``, which means "do not test this". A bar that
    is set but has no matching measurement fails the student; it is never
    skipped silently. A bar outside the built-in table is set through
    ``extra``, keyed by the name of its :class:`AcceptanceBar`.
    :func:`build_acceptance_report` refuses an ``extra`` key that its table of
    bars does not carry.

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
                "the student's and the target force fields. It is read off "
                "force_cosine_aggregate rather than the per-atom mean, which the "
                "holdout's near-zero forces dominate."
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
        Field(
            default=None,
            gt=0,
            description=(
                "Smallest accepted throughput, in atoms per second: propagator "
                "steps per second times the atom count of the timed batch."
            ),
        ),
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
                "equal-size from-scratch student's error on the same holdout. The "
                "ratio is taken on energy_per_atom_mae, forces_mae, and stress_mae, "
                "whichever both carry, and the worst one is kept. A limit of 1.0 "
                "demands a match and a limit below 1.0 demands a margin."
            ),
        ),
    ] = None
    extra: Annotated[
        dict[str, float],
        Field(
            default_factory=dict,
            description=(
                "Limits of custom acceptance bars, keyed by the AcceptanceBar name "
                "they are applied under in the table the report is built with."
            ),
        ),
    ]

    model_config = ConfigDict(extra="forbid")


@dataclasses.dataclass(frozen=True)
class AcceptanceBar:
    """Where one acceptance bar reads the number it gates, and which way it passes.

    An *acceptance bar* is a limit that one number from a student's
    measurements must clear for the student to be accepted. The built-in bars
    are :data:`DEFAULT_BARS`. To register a custom bar, pass a table that
    includes it to :func:`build_acceptance_report` and :func:`measured_bars`.
    Set its limit under its name in ``AcceptanceThresholds.extra``, and file
    its number under its family in ``StudentEvaluation.extra``.

    Parameters
    ----------
    name : str
        Name the bar's limit is set under: a field of
        :class:`AcceptanceThresholds` for a built-in bar, or a key of its
        ``extra`` for a custom one.
    families : tuple[str, ...]
        Measurement slots the bar reads. A slot of :class:`StudentEvaluation`
        names a typed measurement. ``"extra:<family>"`` names a map of numbers
        under ``StudentEvaluation.extra``.
    check : str, optional
        Row the bar reports under. Unless *attribute* overrides it, this is
        also the field or key the bar reads from its first family. Empty only
        for the from-scratch ratio, which spans two families and is computed
        by the report itself. Default ``""``.
    attribute : str, optional
        Field or key read when it differs from *check*. Default ``""``.
    comparison : {"<=", ">="}, optional
        Direction the check passes in. Default ``"<="``.
    quantities : tuple[str, ...], optional
        Accuracy quantities, any one of which is enough to decide the bar.
        :func:`measured_bars` uses them to narrow its answer. Default ``()``.
    missing : str, optional
        Detail reported when the family was supplied but the number it reads
        was not. Default ``""``.

    Raises
    ------
    ValueError
        If *name* or *families* is empty, if a family is neither a measurement
        slot nor an ``"extra:"`` family, or if *comparison* is not ``"<="`` or
        ``">="``.

    Examples
    --------
    >>> from nvalchemi.training.distillation.evaluation import (
    ...     DEFAULT_BARS,
    ...     AcceptanceBar,
    ... )
    >>> dipole_bar = AcceptanceBar(
    ...     "max_dipole_mae", ("extra:dipole",), "dipole_mae"
    ... )
    >>> bars = (*DEFAULT_BARS, dipole_bar)
    """

    name: str
    families: tuple[str, ...]
    check: str = ""
    attribute: str = ""
    comparison: Literal["<=", ">="] = "<="
    quantities: tuple[str, ...] = ()
    missing: str = ""

    def __post_init__(self) -> None:
        """Reject a bar that names nothing, reads nowhere, or passes in no direction."""
        if not self.name or not self.families:
            raise ValueError(
                "An acceptance bar needs a name and at least one family; got "
                f"name={self.name!r}, families={self.families!r}."
            )
        unknown = [family for family in self.families if not _is_family(family)]
        if unknown:
            raise ValueError(
                f"Acceptance bar {self.name!r} reads unknown families {unknown!r}; "
                f"expected slots from {sorted(_STUDENT_SECTIONS)!r} or "
                f"{_EXTRA_FAMILY_PREFIX!r} followed by a family name."
            )
        if self.comparison not in _COMPARISONS:
            raise ValueError(
                f"Acceptance bar {self.name!r} must compare with one of "
                f"{list(_COMPARISONS)!r}; got {self.comparison!r}."
            )


def _is_family(family: str) -> bool:
    """Return whether *family* names a typed slot or a non-empty extra family."""
    return family in _STUDENT_SECTIONS or (
        family.startswith(_EXTRA_FAMILY_PREFIX)
        and len(family) > len(_EXTRA_FAMILY_PREFIX)
    )


DEFAULT_BARS: tuple[AcceptanceBar, ...] = (
    AcceptanceBar(
        "max_energy_per_atom_mae",
        ("accuracy",),
        "energy_per_atom_mae",
        quantities=("energy",),
        missing="the accuracy pass did not compare energy",
    ),
    AcceptanceBar(
        "max_forces_mae",
        ("accuracy",),
        "forces_mae",
        quantities=("forces",),
        missing="the accuracy pass did not compare forces",
    ),
    AcceptanceBar(
        "max_stress_mae",
        ("accuracy",),
        "stress_mae",
        quantities=("stress",),
        missing="the accuracy pass did not compare stress",
    ),
    AcceptanceBar(
        "min_force_cosine",
        ("accuracy",),
        "force_cosine_aggregate",
        comparison=">=",
        quantities=("forces",),
        missing="the accuracy pass did not compare forces",
    ),
    AcceptanceBar(
        "max_energy_drift_per_atom_per_ns",
        ("stability",),
        "energy_drift_per_atom_per_ns",
        missing="the trajectory was recorded without a timestep, so no rate was fitted",
    ),
    AcceptanceBar(
        "max_energy_drift_per_atom_per_step",
        ("stability",),
        "energy_drift_per_atom_per_step",
    ),
    AcceptanceBar("max_momentum_drift", ("stability",), "max_momentum_drift"),
    AcceptanceBar(
        "max_extensivity_error_per_atom",
        ("extensivity",),
        "extensivity_error_per_atom",
        "max_error_per_atom",
    ),
    AcceptanceBar(
        "max_rdf_jensen_shannon", ("rdf",), "rdf_jensen_shannon", "jensen_shannon"
    ),
    AcceptanceBar(
        "min_atoms_per_second", ("throughput",), "atoms_per_second", comparison=">="
    ),
    AcceptanceBar(
        "min_ns_per_day",
        ("throughput",),
        "ns_per_day",
        comparison=">=",
        missing="the propagator was timed without a timestep, so no rate was formed",
    ),
    AcceptanceBar(
        "max_from_scratch_ratio",
        ("accuracy", "baseline_accuracy"),
        quantities=("energy", "forces", "stress"),
    ),
)
"""Built-in bars of :class:`AcceptanceThresholds`, in the order they are applied."""

BAR_FAMILIES: Mapping[str, frozenset[str]] = {
    bar.name: frozenset(bar.families) for bar in DEFAULT_BARS
}
"""Measurement families each built-in bar reads, keyed by threshold field."""

_BUILTIN_BAR_FIELDS = frozenset(BAR_FAMILIES)
"""Threshold fields the built-in bars are set through, as opposed to ``extra``."""


def measured_bars(
    *families: str,
    accuracy_quantities: Sequence[AccuracyQuantity] | None = None,
    bars: Sequence[AcceptanceBar] = DEFAULT_BARS,
) -> frozenset[str]:
    """Return the acceptance bars *families* hold enough measurements to decide.

    A bar counts as measured only when every family it reads was supplied,
    because :func:`build_acceptance_report` fails a student on a bar whose
    measurement is missing rather than skipping it. ``max_from_scratch_ratio``
    therefore needs both ``"accuracy"`` and ``"baseline_accuracy"``.
    *accuracy_quantities* narrows the answer further. An accuracy pass fills
    only the quantities it compared, so a held-out set scored on energy alone
    leaves ``max_forces_mae`` as unfillable as no pass at all. Two bars have a
    precondition that no argument here can express, and they are reported on
    the strength of their family alone. ``max_energy_drift_per_atom_per_ns``
    needs a :class:`~nvalchemi.training.distillation.evaluation.StabilityMonitor`
    built with ``timestep_fs``, and ``min_ns_per_day`` needs
    :func:`~nvalchemi.training.distillation.evaluation.measure_throughput`
    called with ``timestep_fs``. A check that fails for either reason says so.

    Parameters
    ----------
    *families : str
        Slots of a :class:`StudentEvaluation` the caller fills, or
        ``"extra:<family>"`` for a number map under its ``extra``. Naming none
        returns an empty set.
    accuracy_quantities : Sequence[AccuracyQuantity] | None, optional
        Quantities the accuracy pass compared. Default ``None`` (every
        quantity).
    bars : Sequence[AcceptanceBar], optional
        Table of bars to answer for. Default :data:`DEFAULT_BARS`.

    Returns
    -------
    frozenset[str]
        Names of the bars that may be set: fields of
        :class:`AcceptanceThresholds`, or keys of its ``extra``.

    Raises
    ------
    ValueError
        If a name in *families* is not a measurement family the table reads,
        or a name in *accuracy_quantities* is not an accuracy quantity.

    Examples
    --------
    >>> from nvalchemi.training.distillation.evaluation import measured_bars
    >>> sorted(measured_bars("accuracy"))
    ['max_energy_per_atom_mae', 'max_forces_mae', 'max_stress_mae', 'min_force_cosine']
    >>> sorted(measured_bars("accuracy", accuracy_quantities=["energy"]))
    ['max_energy_per_atom_mae']
    """
    supplied = frozenset(families)
    readable = set(_STUDENT_SECTIONS).union(*(bar.families for bar in bars))
    unknown = sorted(supplied - readable)
    if unknown:
        raise ValueError(
            f"Unknown measurement families {unknown!r}; expected names from "
            f"{sorted(readable)!r}."
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
        bar.name
        for bar in bars
        if frozenset(bar.families) <= supplied
        and (not bar.quantities or compared & frozenset(bar.quantities))
    )


@dataclasses.dataclass(frozen=True)
class AcceptanceCheck:
    """One threshold applied to one measurement.

    Attributes
    ----------
    name : str
        Row label of the check: the bar's ``check`` name, or
        ``"from_scratch_ratio"`` for the baseline bar. It can differ from the
        field read, as ``extensivity_error_per_atom`` reads
        ``max_error_per_atom``.
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

    The report is a final artifact rather than a stream, so it does not
    implement the :class:`~nvalchemi.hooks.Reporter` protocol. Print it to a
    :class:`rich.console.Console` for the dashboard view. Pass :meth:`scalars`
    to a :class:`~nvalchemi.hooks.TensorBoardReporter`, or to any scalar sink,
    for a durable record.

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
            ``<student>/accepted`` with value ``1.0`` or ``0.0``. A top-level
            number such as ``num_parameters`` appears as
            ``<student>/num_parameters``.
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
                    if isinstance(value, dict):
                        for inner, number in value.items():
                            flat[f"{evaluation.name}/{group}/{key}/{inner}"] = float(
                                number
                            )
                    elif isinstance(value, (int, float)) and not isinstance(
                        value, bool
                    ):
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
    ordinary miss, ``-inf`` would pass every ``max_*`` bar, and ``+inf`` every
    ``min_*`` bar.
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
    evaluation: StudentEvaluation, limit: float | None
) -> AcceptanceCheck | None:
    """Return the from-scratch check: the student must match or beat its baseline.

    The ratio is taken on ``energy_per_atom_mae``, ``forces_mae``, and
    ``stress_mae``, whichever both carry, and the worst one is kept. A
    baseline scored on a different number of graphs or atoms fails this
    student's own check, not the report for the whole family. A baseline error
    of exactly zero is unbeatable: a matching student ties at ``1.0``, and any
    error fails at infinity. A non-finite error on either side yields no ratio
    at all.
    """
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


def _limit(thresholds: AcceptanceThresholds, bar: AcceptanceBar) -> float | None:
    """Return the limit set for *bar*: its own field, else its ``extra`` entry."""
    if bar.name in _BUILTIN_BAR_FIELDS:
        return getattr(thresholds, bar.name)
    return thresholds.extra.get(bar.name)


def _family_metrics(evaluation: StudentEvaluation, family: str) -> Any:
    """Return the measurement *family* names on *evaluation*, or ``None``."""
    if family.startswith(_EXTRA_FAMILY_PREFIX):
        return evaluation.extra.get(family[len(_EXTRA_FAMILY_PREFIX) :])
    return getattr(evaluation, family)


def _read(metrics: Any, attribute: str) -> float | None:
    """Return *attribute* off a typed measurement or out of a number map."""
    if isinstance(metrics, Mapping):
        return metrics.get(attribute)
    return getattr(metrics, attribute)


def _student_checks(
    evaluation: StudentEvaluation,
    thresholds: AcceptanceThresholds,
    bars: Sequence[AcceptanceBar],
) -> tuple[AcceptanceCheck, ...]:
    """Apply every bar of *bars* that *thresholds* sets to one student.

    Each bar reads its measurement through its first family. A bar absent from
    *bars* is neither applied nor reported. A bar whose family was measured but
    whose own number was not reports which quantity or timestep was missing.
    The one bar with no ``check`` is the from-scratch ratio, which
    :func:`_baseline_check` forms across two families.
    """
    candidates = []
    for bar in bars:
        limit = _limit(thresholds, bar)
        if not bar.check:
            candidates.append(_baseline_check(evaluation, limit))
            continue
        family = bar.families[0]
        metrics = _family_metrics(evaluation, family)
        candidates.append(
            _check(
                bar.check,
                None if metrics is None else _read(metrics, bar.attribute or bar.check),
                limit,
                bar.comparison,
                _rdf_detail(evaluation.rdf) if family == "rdf" else "",
                "not measured" if metrics is None else bar.missing or "not measured",
            )
        )
    return tuple(check for check in candidates if check is not None)


def _pareto_front(evaluations: Sequence[StudentEvaluation]) -> tuple[str, ...]:
    """Return the students no other student beats on both accuracy and speed.

    Ranking needs two finite numbers, so a student with a non-finite error or
    rate is left off the front, like a student that was never timed. Every
    comparison against a NaN is false, so nothing could dominate such a
    student and it would always sit on the front.
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

    The verdict column cannot wrap, so a narrow console abbreviates a header
    rather than truncating the ``ACCEPT`` or ``REJECT`` the reader is looking
    for.
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
    *,
    bars: Sequence[AcceptanceBar] = DEFAULT_BARS,
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
    bars : Sequence[AcceptanceBar], optional
        Table of bars to apply. Extend :data:`DEFAULT_BARS` to gate a
        measurement filed under ``StudentEvaluation.extra``. Default
        :data:`DEFAULT_BARS`.

    Returns
    -------
    AcceptanceReport
        Verdicts aligned with *evaluations*, plus the Pareto front over force
        MAE and atoms per second.

    Raises
    ------
    ValueError
        If *evaluations* is empty, or two students share a name. If two
        entries of *bars* share a name. If ``thresholds.extra`` sets a limit
        for a bar that the table does not carry, or for a built-in bar that is
        set through its own field. If the students were not all scored on the
        same held-out set, or if the students that carry a throughput
        measurement were not all measured on the same batch.

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
    bar_names = [bar.name for bar in bars]
    if len(set(bar_names)) != len(bar_names):
        raise ValueError(f"Acceptance bar names must be unique; got {bar_names!r}.")
    custom = set(bar_names) - _BUILTIN_BAR_FIELDS
    unplaced = sorted(set(resolved.extra) - custom)
    if unplaced:
        raise ValueError(
            f"AcceptanceThresholds.extra sets {unplaced!r}, but the table applies "
            f"custom bars {sorted(custom)!r} only; a built-in bar is set through "
            "its own field, and a custom one needs its AcceptanceBar in bars."
        )
    holdouts = {
        (evaluation.accuracy.num_graphs, evaluation.accuracy.num_atoms)
        for evaluation in evaluations
    }
    if len(holdouts) > 1:
        raise ValueError(
            f"Students were scored on different holdouts {sorted(holdouts)!r} as "
            "(num_graphs, num_atoms). Errors are comparable, and the Pareto front "
            "ranks them, only across students scored on one holdout. Re-run "
            "evaluate_accuracy for every student over the same held-out data."
        )
    workloads = {
        (evaluation.throughput.num_atoms, evaluation.throughput.num_graphs)
        for evaluation in evaluations
        if evaluation.throughput is not None
    }
    if len(workloads) > 1:
        raise ValueError(
            f"Students were timed on different batches {sorted(workloads)!r} as "
            "(num_atoms, num_graphs). Throughput scales with the batch it was "
            "measured on, so the rates are comparable only when every student was "
            "timed on the same batch. Re-measure every student with "
            "measure_throughput on the same batch."
        )
    verdicts = []
    for evaluation in evaluations:
        checks = _student_checks(evaluation, resolved, bars)
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
