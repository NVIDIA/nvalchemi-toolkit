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
"""Accuracy, teacher-consistency, and non-conservative diagnostics for students.

Two evaluations live here. :func:`evaluate_accuracy` runs a student over a
held-out set and reports energy, force, and stress errors against either a
reference dataset's own labels or a teacher's, together with the
teacher-consistency diagnostics that only make sense against a teacher.
:func:`nonconservative_residual` measures the part of a teacher's force field
no conservative student can fit, which is the floor the first evaluation is
read against.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Literal, TypeAlias

import torch

from nvalchemi.data import Batch
from nvalchemi.training._validation import (
    ValidationConfig,
    ValidationLoop,
    _ensure_reiterable_validation_data,
)
from nvalchemi.training.distillation._labels import _attach_teacher_labels
from nvalchemi.training.distillation.evaluation._export import _rebuild
from nvalchemi.training.distillation.scoring import (
    InProcessTeacherScorer,
    TeacherScorer,
    scorer_fields,
    signal_fields,
)
from nvalchemi.training.distillation.strategy import (
    _student_label_dtype,
    _to_device,
)
from nvalchemi.training.distributed import all_reduce, is_distributed_initialized
from nvalchemi.training.losses.composition import ComposedLossFunction
from nvalchemi.training.losses.terms import (
    EnergyMSELoss,
    ForceMSELoss,
    StressMSELoss,
)
from nvalchemi.training.strategy import default_training_fn

if TYPE_CHECKING:
    from collections.abc import Callable

    from nvalchemi.models.base import BaseModelMixin

__all__ = [
    "AccuracyMetrics",
    "AccuracyQuantity",
    "NonConservativeResidual",
    "evaluate_accuracy",
    "nonconservative_residual",
]

AccuracyQuantity: TypeAlias = Literal["energy", "forces", "stress", "atomic_energies"]
"""Quantity an accuracy evaluation compares between a student and a target."""

_QUANTITY_SIGNALS: dict[str, str] = {
    "energy": "energy",
    "forces": "forces",
    "stress": "stress",
    "atomic_energies": "node_energies",
}
"""Teacher signal behind each quantity, which the signal surface cannot invert."""

_REFERENCE_TARGET_KEYS: dict[str, str] = {
    "energy": "energy",
    "forces": "forces",
    "stress": "stress",
    "atomic_energies": "atomic_energies",
}
"""Batch field a reference dataset carries, keyed by quantity."""

_PREDICTION_KEYS: dict[str, str] = {
    "energy": "predicted_energy",
    "forces": "predicted_forces",
    "stress": "predicted_stress",
    "atomic_energies": "predicted_atomic_energies",
}
"""Prediction key :func:`default_training_fn` publishes, keyed by quantity."""

_SUPERVISED_QUANTITIES = ("energy", "forces", "stress")
"""Quantities a built-in loss term supervises, in report order."""

_DEFAULT_QUANTITIES: tuple[AccuracyQuantity, ...] = ("energy", "forces")
"""Quantities evaluated when a caller names none."""

_EPS = 1e-12
"""Denominator guard for direction normalization and force-scale ratios."""

_RESIDUAL_SUFFIXES = ("abs", "sq", "count")
"""Sums accumulated per quantity, in the order :func:`_mae_rmse` reads them."""

_FORCE_ALIGNMENT_KEYS = (
    "force_cosine_sum",
    "force_cosine_count",
    "force_dot",
    "force_predicted_sq",
    "force_target_sq",
    "force_nonfinite_atoms",
)
"""Extra sums the force quantity contributes on top of its residuals."""


@dataclasses.dataclass(frozen=True)
class AccuracyMetrics:
    """Errors of one student against one set of targets over a held-out set.

    Every metric is an exact global reduction over the evaluated set — the sum
    of residuals divided by the total count, not a mean of per-batch means — so
    the value does not depend on how the data was batched. Errors are reported
    in the units the batch carries them in: eV for energies, eV/A for forces,
    and the stress units of the dataset.

    Fields are ``None`` for quantities the pass could not measure, either
    because they were not requested or because a batch carried no such
    prediction or target. A quantity that was measured and came out non-finite
    reports ``nan`` rather than ``None``, since the two are different
    statements — nobody looked, against an answer that is garbage — and an
    acceptance bar fails the second instead of reporting it missing.

    Attributes
    ----------
    name : str
        Label carried into reports.
    num_graphs : int
        Number of graphs evaluated.
    num_atoms : int
        Number of atoms evaluated.
    energy_mae, energy_rmse : float | None
        Total-energy error per graph.
    energy_per_atom_mae, energy_per_atom_rmse : float | None
        Total-energy error divided by each graph's atom count.
    forces_mae, forces_rmse : float | None
        Force error per Cartesian component, averaged over every component of
        every atom.
    stress_mae, stress_rmse : float | None
        Stress error per component, averaged over all nine components.
    force_cosine_mean : float | None
        Mean over atoms of the cosine similarity between the predicted and
        target force vectors, weighting every atom equally however small its
        force is. An atom sitting near a symmetric site carries a force at or
        below the student's own error and scores an essentially random angle,
        so this number describes the holdout's low-force tail as much as it
        describes the student: adding relaxed frames to a set of thermal ones
        moves it far while leaving ``forces_mae`` where it was. Atoms whose
        force vanishes exactly on either side are not counted, since the angle
        between them is undefined; a set in which no atom carries a force on
        both sides reports ``None``.
    force_cosine_aggregate : float | None
        Cosine similarity of the two force fields taken as single vectors over
        the whole evaluated set, which weights atoms by force magnitude instead
        of equally and is the alignment an acceptance bar is read against.
        Magnitude weighting is what makes a single non-finite atom carry the
        whole set, so this reports ``nan`` where ``force_cosine_mean`` still
        reports the angle of the atoms that stayed finite.
    atomic_energy_mae, atomic_energy_rmse : float | None
        Per-atom energy residual, populated only when both sides publish an
        atomic energy decomposition.
    force_nonfinite_atoms : int
        Atoms whose predicted or target force carried a non-finite component.
        Those atoms are dropped from ``force_cosine_mean``, whose per-atom
        average has no meaning at an undefined angle, so this count is what
        says the mean was formed over fewer atoms than ``num_atoms``.
    """

    name: str
    num_graphs: int
    num_atoms: int
    energy_mae: float | None = None
    energy_rmse: float | None = None
    energy_per_atom_mae: float | None = None
    energy_per_atom_rmse: float | None = None
    forces_mae: float | None = None
    forces_rmse: float | None = None
    stress_mae: float | None = None
    stress_rmse: float | None = None
    force_cosine_mean: float | None = None
    force_cosine_aggregate: float | None = None
    atomic_energy_mae: float | None = None
    atomic_energy_rmse: float | None = None
    force_nonfinite_atoms: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Return the populated fields as a plain dictionary."""
        return {
            field.name: getattr(self, field.name)
            for field in dataclasses.fields(self)
            if getattr(self, field.name) is not None
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> AccuracyMetrics:
        """Rebuild the metrics from a :meth:`to_dict` export.

        The quantities the export dropped because they were not measured come
        back as ``None``, so the rebuilt metrics report exactly what the
        original did.
        """
        return _rebuild(cls, data)


@dataclasses.dataclass(frozen=True)
class NonConservativeResidual:
    r"""Non-conservative component of a teacher's force field.

    A force field decomposes into a conservative part and a remainder,
    :math:`F = -\nabla E + F_{\perp}`, and a student that predicts forces as
    the gradient of an energy can represent only the first term. The work
    integral of the first term around any closed path vanishes, so a closed-path
    integral of the teacher's field measures :math:`F_{\perp}` alone.

    Every reported force is a root-mean-square per-atom magnitude in eV/A, so
    ``force_floor`` sits in the same norm as ``force_rms`` and can be read
    against the ``forces_rmse`` of an :class:`AccuracyMetrics` — which is a
    per-Cartesian-component figure, smaller by ``sqrt(3)`` for an isotropic
    error.

    Attributes
    ----------
    num_probes : int
        Number of closed loops integrated, counting each graph of each batch
        separately.
    amplitude : float
        Loop side length in A, as a per-atom displacement: every atom moves
        this far in root-mean-square along each side.
    segments : int
        Midpoint-rule samples per side.
    loop_work_mean_abs, loop_work_max_abs : float
        Mean and maximum over probes of the absolute work accumulated around
        one closed loop, in eV.
    force_floor, force_floor_max : float
        Mean and maximum over probes of the lower bound the loop work places on
        the root-mean-square per-atom force error of a conservative student, in
        eV/A.
    force_rms : float
        Root-mean-square teacher force magnitude at the loop centers, taken
        over every atom of every graph, in eV/A.
    relative_floor, relative_floor_max : float
        Mean and maximum over probes of a probe's own floor divided by the
        root-mean-square teacher force of the graph that probe visited. The
        ratio is formed inside one graph, so a batch mixing force scales reports
        a figure that lies between its graphs' own ratios; the quotient of
        ``force_floor`` by ``force_rms`` weights its numerator by graph and its
        denominator by atom, and need not. A graph at equilibrium has no force scale to
        be relative to, so its ratio divides by a clamp rather than by zero.
    """

    num_probes: int
    amplitude: float
    segments: int
    loop_work_mean_abs: float
    loop_work_max_abs: float
    force_floor: float
    force_floor_max: float
    force_rms: float
    relative_floor: float
    relative_floor_max: float

    def to_dict(self) -> dict[str, Any]:
        """Return every field as a plain dictionary."""
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> NonConservativeResidual:
        """Rebuild the residual from a :meth:`to_dict` export."""
        return _rebuild(cls, data)


class _PlacedBatches:
    """Re-iterable view that moves each batch to *device*, labeling it if scored.

    The placement is unconditional because the evaluation's own move is not
    safe on every destination: :meth:`ValidationLoop.execute` copies with
    ``non_blocking=True``, which on a host destination returns before the
    transfer lands and lets the loop read index tensors out of a half-written
    buffer. Every batch therefore already sits on the run device by the time
    the loop receives it, and the loop's own move degenerates to a same-device
    copy. A ``batch.device == device`` short-circuit is deliberately not taken
    either: :func:`_to_device` clones an already-placed batch for a fraction of
    a percent of a pass, and that clone is what keeps the ``teacher_*`` fields
    a scorer attaches off the caller's own batches.

    A scorer labels after the device move rather than inside the loop, so a
    teacher evaluating a host-resident dataset on GPU runs where the student
    does.
    """

    def __init__(
        self,
        source: Iterable[Batch],
        device: torch.device,
        scorer: TeacherScorer | None = None,
    ) -> None:
        self.source = source
        self.device = device
        self.scorer = scorer

    def __iter__(self) -> Iterator[Batch]:
        """Yield each source batch on the run device, labeled if a scorer was given."""
        for batch in self.source:
            placed = _to_device(batch, self.device)
            if self.scorer is not None:
                _attach_teacher_labels(placed, self.scorer.label(placed))
            yield placed


class _MetricAccumulator:
    """Exact residual sums over a validation pass, as a per-batch callback.

    Implements :class:`~nvalchemi.training.BatchValidationCallback`, so the
    accumulator rides along on a :class:`ValidationLoop` pass and sees the
    predictions the loop already computed. Sums are kept as float64 device
    tensors and reduced once at the end, so no metric forces a
    host synchronization per batch.

    Every sum the requested quantities can produce is seeded at zero up front
    rather than created on first use, so the packed all-reduce tensor has the
    same shape and key order on every rank even when one rank's shard happened
    to carry no target for some quantity.
    """

    def __init__(
        self,
        device: torch.device,
        quantities: Sequence[str],
        target_keys: Mapping[str, str],
    ) -> None:
        self.device = device
        self.quantities = tuple(quantities)
        self.target_keys = dict(target_keys)
        self._sums: dict[str, torch.Tensor] = {
            key: torch.zeros((), device=device, dtype=torch.float64)
            for key in _metric_keys(self.quantities)
        }

    def __call__(
        self,
        *,
        batch: Batch,
        predictions: Mapping[str, torch.Tensor],
        loss: Any,  # noqa: ARG002
        batch_count: int,  # noqa: ARG002
        step_count: int,  # noqa: ARG002
        epoch: int,  # noqa: ARG002
    ) -> None:
        """Accumulate one validation batch's residual sums."""
        self._add("num_graphs", batch.num_graphs)
        self._add("num_atoms", batch.num_nodes)
        for quantity in self.quantities:
            prediction = predictions.get(_PREDICTION_KEYS[quantity])
            target = getattr(batch, self.target_keys[quantity], None)
            if prediction is None or target is None:
                continue
            if quantity == "atomic_energies":
                self._accumulate(
                    quantity, prediction.reshape(-1), target.reshape(-1), target.numel()
                )
                continue
            self._accumulate(quantity, prediction, target, target.numel())
            if quantity == "energy":
                counts = batch.num_nodes_per_graph.reshape(
                    (-1,) + (1,) * (target.ndim - 1)
                )
                self._accumulate(
                    "energy_per_atom",
                    prediction / counts,
                    target / counts,
                    target.shape[0],
                )
            elif quantity == "forces":
                self._accumulate_cosine(prediction, target)

    def _add(self, key: str, value: torch.Tensor | float) -> None:
        """Add one contribution to the running float64 sum *key*."""
        tensor = value.detach() if isinstance(value, torch.Tensor) else value
        scalar = torch.as_tensor(tensor, device=self.device, dtype=torch.float64)
        previous = self._sums.get(key)
        self._sums[key] = scalar if previous is None else previous + scalar

    def _accumulate(
        self, name: str, prediction: torch.Tensor, target: torch.Tensor, count: int
    ) -> None:
        """Add the absolute and squared residual sums of one quantity.

        Shapes must match exactly; broadcasting a ``(B,)`` target against a
        ``(B, 1)`` prediction would silently measure every pairing.
        """
        if prediction.shape != target.shape:
            raise ValueError(
                f"Prediction and target of {name!r} must have the same shape; got "
                f"{tuple(prediction.shape)!r} and {tuple(target.shape)!r}."
            )
        residual = prediction.detach().to(torch.float64) - target.detach().to(
            torch.float64
        )
        self._add(f"{name}_abs", residual.abs().sum())
        self._add(f"{name}_sq", residual.pow(2).sum())
        self._add(f"{name}_count", float(count))

    def _accumulate_cosine(
        self, prediction: torch.Tensor, target: torch.Tensor
    ) -> None:
        """Add the per-atom and aggregate force-alignment sums.

        The ``> 0.0`` test is the guard against dividing zero by zero and
        nothing else: an atom whose force is merely small is still counted at
        full weight in the per-atom mean, which is what makes that mean a
        property of the holdout's low-force tail. The aggregate sums need no
        guard at all, being magnitude-weighted.

        An atom carrying a non-finite force has no angle either, so it leaves
        the per-atom mean by the same door and is counted on the way out. It
        stays in the aggregate sums deliberately: that number is one alignment
        over the whole set, and a set holding an unmeasurable atom is better
        reported as unmeasurable than averaged over what is left of it.
        """
        predicted = prediction.detach().to(torch.float64)
        reference = target.detach().to(torch.float64)
        dot = (predicted * reference).sum(dim=-1)
        predicted_norm = predicted.norm(dim=-1)
        reference_norm = reference.norm(dim=-1)
        finite = torch.isfinite(predicted).all(dim=-1) & torch.isfinite(reference).all(
            dim=-1
        )
        aligned = finite & (predicted_norm > 0.0) & (reference_norm > 0.0)
        norms = predicted_norm * reference_norm
        self._add("force_cosine_sum", (dot[aligned] / norms[aligned]).sum())
        self._add("force_cosine_count", float(aligned.sum()))
        self._add("force_nonfinite_atoms", float((~finite).sum()))
        self._add("force_dot", dot.sum())
        self._add("force_predicted_sq", predicted.pow(2).sum())
        self._add("force_target_sq", reference.pow(2).sum())

    def metrics(self, *, name: str, distributed_manager: Any | None) -> AccuracyMetrics:
        """Return the reduced metrics, all-reducing sums under distributed runs.

        Raises
        ------
        ValueError
            If no quantity was measured, which means every batch was missing
            either the prediction or the target of every requested quantity.
        """
        keys = tuple(sorted(self._sums))
        packed = torch.stack([self._sums[key] for key in keys])
        if is_distributed_initialized(distributed_manager):
            all_reduce(packed, distributed_manager)
        totals = {key: float(packed[index]) for index, key in enumerate(keys)}
        if not any(
            value > 0.0 for key, value in totals.items() if key.endswith("_count")
        ):
            raise ValueError(
                "No accuracy metric could be measured; every batch was missing "
                f"the prediction or the target of every requested quantity "
                f"{list(self.quantities)!r} (targets {self.target_keys!r})."
            )
        energy_mae, energy_rmse = _mae_rmse(totals, "energy")
        per_atom_mae, per_atom_rmse = _mae_rmse(totals, "energy_per_atom")
        forces_mae, forces_rmse = _mae_rmse(totals, "forces")
        stress_mae, stress_rmse = _mae_rmse(totals, "stress")
        atomic_mae, atomic_rmse = _mae_rmse(totals, "atomic_energies")
        return AccuracyMetrics(
            name=name,
            num_graphs=int(totals.get("num_graphs", 0.0)),
            num_atoms=int(totals.get("num_atoms", 0.0)),
            energy_mae=energy_mae,
            energy_rmse=energy_rmse,
            energy_per_atom_mae=per_atom_mae,
            energy_per_atom_rmse=per_atom_rmse,
            forces_mae=forces_mae,
            forces_rmse=forces_rmse,
            stress_mae=stress_mae,
            stress_rmse=stress_rmse,
            force_cosine_mean=_ratio(
                totals.get("force_cosine_sum"), totals.get("force_cosine_count")
            ),
            force_cosine_aggregate=_aggregate_cosine(totals),
            atomic_energy_mae=atomic_mae,
            atomic_energy_rmse=atomic_rmse,
            force_nonfinite_atoms=int(totals.get("force_nonfinite_atoms", 0.0)),
        )


def _metric_keys(quantities: Sequence[str]) -> tuple[str, ...]:
    """Return every sum the requested *quantities* can contribute to."""
    keys = ["num_graphs", "num_atoms"]
    for quantity in quantities:
        prefixes = ["energy_per_atom", quantity] if quantity == "energy" else [quantity]
        keys.extend(
            f"{prefix}_{suffix}" for prefix in prefixes for suffix in _RESIDUAL_SUFFIXES
        )
        if quantity == "forces":
            keys.extend(_FORCE_ALIGNMENT_KEYS)
    return tuple(keys)


def _mae_rmse(
    totals: Mapping[str, float], prefix: str
) -> tuple[float | None, float | None]:
    """Return the MAE and RMSE of one quantity, or two ``None`` when unmeasured."""
    count = totals.get(f"{prefix}_count", 0.0)
    if count <= 0.0:
        return None, None
    return totals[f"{prefix}_abs"] / count, math.sqrt(totals[f"{prefix}_sq"] / count)


def _ratio(numerator: float | None, denominator: float | None) -> float | None:
    """Return ``numerator / denominator``, or ``None`` when either is missing."""
    if numerator is None or not denominator:
        return None
    return numerator / denominator


def _aggregate_cosine(totals: Mapping[str, float]) -> float | None:
    """Return the cosine similarity of the two force fields taken as one vector.

    The two ways this cannot answer are told apart rather than folded together.
    A non-finite sum is an alignment that was measured and came out garbage,
    and reports ``nan`` so an acceptance bar fails it; a norm of exactly zero
    is a set whose forces all vanish, which carries no direction to compare and
    reports ``None`` like any quantity nobody measured.
    """
    dot = totals.get("force_dot")
    if dot is None:
        return None
    squares = totals["force_predicted_sq"] * totals["force_target_sq"]
    if not math.isfinite(dot) or not math.isfinite(squares):
        return math.nan
    norm = math.sqrt(squares)
    return dot / norm if norm > 0.0 else None


def _teacher_target_keys() -> dict[str, str]:
    """Return the teacher field each quantity is compared against.

    Resolved through the scoring signal surface rather than restated here, so a
    signal that moves the field it writes carries the evaluation with it.

    Returns
    -------
    dict[str, str]
        Batch field a teacher scorer writes, keyed by quantity.

    Raises
    ------
    RuntimeError
        If the signal behind a quantity publishes more than one field, which
        leaves no single field a prediction can be compared against.
    """
    resolved: dict[str, str] = {}
    for quantity, signal in _QUANTITY_SIGNALS.items():
        fields = signal_fields([signal])
        if len(fields) != 1:
            raise RuntimeError(
                f"Quantity {quantity!r} is compared against a single teacher "
                f"field, but its signal {signal!r} publishes {list(fields)!r}."
            )
        resolved[quantity] = fields[0]
    return resolved


def _resolve_device(model: Any, device: torch.device | str | None) -> torch.device:
    """Return the requested device, else the device the model's parameters sit on."""
    if device is not None:
        return torch.device(device)
    parameters = getattr(model, "parameters", None)
    if callable(parameters):
        for parameter in parameters():
            return parameter.device
    return torch.device("cpu")


def _differentiated_outputs(model: Any) -> set[str]:
    """Return the active outputs a model produces by differentiating its forward.

    This is the declaration
    :meth:`~nvalchemi.models.base.BaseModelMixin.adapt_input` itself reads to
    decide which inputs to mark ``requires_grad``, so it is non-empty exactly
    when the model's own forward pass needs autograd enabled around it.
    """
    config = getattr(model, "model_config", None)
    autograd = getattr(config, "autograd_outputs", None) or frozenset()
    active = getattr(config, "active_outputs", None) or frozenset()
    return set(autograd) & set(active)


def _as_scorer(
    model: TeacherScorer | BaseModelMixin,
    signals: Sequence[str],
    cast_to: torch.dtype | None = None,
) -> Any:
    """Return *model* as a scorer, wrapping a bare model in an in-process one.

    A supplied scorer is checked against the batch fields the requested signals
    are read from rather than against the signal names it declares, since a
    scorer is free to publish its labels under fields of its own and one that
    does would otherwise pass the check and fail deep inside the evaluation. A
    scorer whose fields cannot be known is let through, because an undeclared
    custom scorer is not the same as one declaring nothing. *cast_to* applies
    only to a wrapped bare model, so a probe that has to read a teacher in its
    native precision leaves it unset.
    """
    if isinstance(model, TeacherScorer):
        fields = scorer_fields(model)
        required = signal_fields(signals)
        missing = None if fields is None else sorted(set(required) - set(fields))
        if missing:
            raise ValueError(
                f"Scorer must publish the fields {list(required)!r} this evaluation "
                f"reads; got {list(fields)!r}, missing {missing!r}."
            )
        return model
    return InProcessTeacherScorer(model, signals, cast_to=cast_to)


def _metric_loss(
    quantities: Sequence[str], target_keys: Mapping[str, str]
) -> ComposedLossFunction:
    """Build the composed loss whose gradient requirement drives the pass."""
    terms = []
    if "energy" in quantities:
        terms.append(EnergyMSELoss(target_key=target_keys["energy"], per_atom=True))
    if "forces" in quantities:
        terms.append(ForceMSELoss(target_key=target_keys["forces"]))
    if "stress" in quantities:
        terms.append(StressMSELoss(target_key=target_keys["stress"]))
    if not terms:
        raise ValueError(
            "At least one of 'energy', 'forces', or 'stress' must be evaluated so "
            f"the validation pass has a loss to run; got {list(quantities)!r}."
        )
    return ComposedLossFunction(terms, dtype_policy="prediction_to_target")


def evaluate_accuracy(
    model: BaseModelMixin,
    data: Iterable[Batch],
    *,
    targets: Literal["reference", "teacher"] = "reference",
    quantities: Sequence[AccuracyQuantity] | None = None,
    scorer: TeacherScorer | BaseModelMixin | None = None,
    target_keys: Mapping[str, str] | None = None,
    loss_fn: ComposedLossFunction | None = None,
    validation_fn: Callable[..., Any] = default_training_fn,
    grad_mode: Literal["auto", "enabled", "disabled"] = "auto",
    device: torch.device | str | None = None,
    distributed_manager: Any | None = None,
    name: str = "accuracy",
) -> AccuracyMetrics:
    """Measure a student's error over a held-out set.

    The pass itself runs through :class:`~nvalchemi.training.ValidationLoop`, so
    eval mode, autocast, and device placement behave exactly as they do during
    training validation; the autograd policy is settled here first, because a
    student that differentiates inside its own forward needs gradients even
    when nothing derivative is being scored.
    The metrics are accumulated separately, as exact global residual sums, and
    the loop's own loss value is discarded: a loss is a training objective with
    its own graph balancing, while an evaluation wants the plain per-atom and
    per-component errors a paper reports.

    Point *targets* at ``"reference"`` to compare against the dataset's own
    labels — the DFT holdout — and at ``"teacher"`` to compare against the
    ``teacher_*`` fields, whether they were written offline by
    :func:`~nvalchemi.training.distillation.label_dataset` or on the fly by a
    *scorer* passed here. Against a teacher the force-alignment and per-atom
    energy diagnostics fill in as well, since both sides then describe the same
    decomposition. Passing a *scorer* while comparing against the dataset's own
    labels is a contradiction — the teacher pass would be paid for and thrown
    away, and the diagnostics that only mean something against a teacher would
    be read off the dataset — so that combination raises instead of running.

    Parameters
    ----------
    model : BaseModelMixin
        Student to evaluate. Left in the training mode it arrived in, and
        scored on exactly the weights handed over: a student trained under
        an ``EMAHook`` needs ``strategy.inference_model`` here to be scored
        on the averaged ones. Nothing downstream can tell which of the two
        arrived, so record the choice on
        :class:`~nvalchemi.training.distillation.evaluation.StudentEvaluation`.
    data : Iterable[Batch]
        Re-iterable holdout set. One-shot iterators are rejected.
    targets : {"reference", "teacher"}, optional
        Which family of batch fields to compare against. Default
        ``"reference"``.
    quantities : Sequence[AccuracyQuantity] | None, optional
        Quantities to evaluate. ``"atomic_energies"`` is a diagnostic only and
        never enters the pass's loss. Default ``None`` (energy and forces).
    scorer : TeacherScorer | BaseModelMixin | None, optional
        Teacher used to label each batch before it is evaluated. A bare model
        is wrapped in an
        :class:`~nvalchemi.training.distillation.InProcessTeacherScorer` for
        the requested quantities. Only meaningful when something is compared
        against a teacher field, so it is rejected rather than silently paid
        for when nothing is. Default ``None`` (the batches are used as they
        arrive).
    target_keys : Mapping[str, str] | None, optional
        Per-quantity overrides of the batch field to compare against, applied
        on top of the map *targets* selects. Default ``None``.
    loss_fn : ComposedLossFunction | None, optional
        Loss driving the pass. Default ``None`` (mean-squared terms over the
        requested supervised quantities).
    validation_fn : Callable, optional
        Forward callable invoked as ``validation_fn(model, batch)``. Default
        :func:`~nvalchemi.training.default_training_fn`.
    grad_mode : {"auto", "enabled", "disabled"}, optional
        Autograd policy. ``"auto"`` enables gradients whenever the student's
        own forward needs them or the loss does, which is what lets an
        autograd-force student be evaluated at all; ``"disabled"`` is rejected
        for such a student rather than failing inside it. Default ``"auto"``.
    device : torch.device | str | None, optional
        Device the pass runs on. Default ``None`` (the model's own device).
    distributed_manager : Any | None, optional
        Manager used to all-reduce the metric sums. Default ``None``.
    name : str, optional
        Label stored on the result. Default ``"accuracy"``.

    Returns
    -------
    AccuracyMetrics
        Errors and consistency diagnostics over the whole set.

    Raises
    ------
    ValueError
        If *quantities* names an unknown quantity, if no supervised quantity is
        requested, if a *scorer* is given but no requested quantity is compared
        against a teacher field or the scorer does not publish the fields the
        evaluation reads, if gradients are disabled for a student that
        differentiates inside its own forward, if a prediction and its target
        disagree on shape, or if no metric could be measured at all.

    Examples
    --------
    >>> from nvalchemi.training.distillation.evaluation import evaluate_accuracy
    >>> metrics = evaluate_accuracy(student, holdout)  # doctest: +SKIP
    >>> metrics.forces_mae  # doctest: +SKIP
    0.031

    Against the teacher, labeling on the fly:

    >>> metrics = evaluate_accuracy(  # doctest: +SKIP
    ...     student,
    ...     holdout,
    ...     targets="teacher",
    ...     scorer=teacher,
    ...     quantities=("energy", "forces", "atomic_energies"),
    ... )

    Notes
    -----
    The student is called through *validation_fn* exactly as a training loop
    would call it, so a student that reads a neighbor list needs batches that
    carry one — built with
    :func:`~nvalchemi.neighbors.compute_neighbors`, produced by the loader, or
    assembled by a composed model pipeline. A *scorer* has no such requirement:
    it builds and rolls back the teacher's own list per batch.

    Which quantities are scored does not decide the autograd policy on its own.
    A student whose active outputs include one of its ``autograd_outputs``
    builds a graph inside its own forward — the same declaration
    :meth:`~nvalchemi.models.base.BaseModelMixin.adapt_input` reads to mark the
    inputs it differentiates — so ``"auto"`` enables gradients for it even when
    only energies are being compared, rather than leaving it to fail in its own
    ``torch.autograd.grad``. A student declaring no autograd output keeps the
    loss-driven fast path and is scored under ``torch.no_grad()``; narrowing
    ``active_outputs`` puts an autograd-force student back on that path too.

    Under a distributed run the reduce follows the initialized process group, so
    every rank must call this with the same *quantities* and a non-empty shard.
    The metric sums are packed into one tensor in a shared key order before the
    all-reduce, so ranks asked for different quantities would pack differently
    shaped tensors and deadlock, and a rank whose shard is empty raises out of
    the validation loop before that reduce and strands the others. Shard sizes,
    and the targets a shard happens to carry, may differ: every sum a requested
    quantity can produce starts at zero whether or not a rank's own batches
    carried a target for it.

    A *scorer*'s labels are cast to the dtype the student's own labels are
    stored at — the float32-floored parameter dtype the strategy and
    :func:`~nvalchemi.training.distillation.label_dataset` already agree on — so
    a float64 teacher scores a float32 student exactly as the store would, and
    the cast changes no metric, every residual being accumulated in float64
    either way. A :class:`~nvalchemi.training.distillation.TeacherScorer` handed
    in already labels batches its own way and is left uncast.
    """
    requested = tuple(quantities) if quantities is not None else _DEFAULT_QUANTITIES
    unknown = sorted(set(requested) - set(_PREDICTION_KEYS))
    if unknown:
        raise ValueError(
            f"Accuracy quantities must be names from {sorted(_PREDICTION_KEYS)!r}; "
            f"got unsupported {unknown!r}."
        )
    teacher_keys = _teacher_target_keys()
    base = teacher_keys if targets == "teacher" else _REFERENCE_TARGET_KEYS
    resolved_keys = dict(base) | dict(target_keys or {})
    compared = {resolved_keys[quantity] for quantity in requested}
    if scorer is not None and not (compared & set(teacher_keys.values())):
        raise ValueError(
            "A scorer labels every batch with the teacher's fields, but "
            f"targets={targets!r} compares against {sorted(compared)!r}, so the "
            "teacher pass would be paid and thrown away and the errors reported "
            "would be against the dataset's own labels rather than the teacher's. "
            "Pass targets='teacher', drop the scorer, or name the teacher fields "
            "to compare against in target_keys."
        )
    supervised = [
        quantity for quantity in _SUPERVISED_QUANTITIES if quantity in requested
    ]
    differentiated = _differentiated_outputs(model)
    if differentiated and grad_mode == "disabled":
        raise ValueError(
            f"Student {type(model).__name__!r} computes {sorted(differentiated)!r} "
            "by differentiating inside its own forward, which "
            "grad_mode='disabled' makes impossible whatever is being scored. "
            "Pass grad_mode='auto', or narrow the student's "
            "model_config.active_outputs so it stops differentiating."
        )
    resolved_device = _resolve_device(model, device)

    signals = [_QUANTITY_SIGNALS[quantity] for quantity in requested]
    evaluation_data: Iterable[Batch] = _PlacedBatches(
        _ensure_reiterable_validation_data(data),
        resolved_device,
        scorer=(
            _as_scorer(scorer, signals, _student_label_dtype(model))
            if scorer is not None
            else None
        ),
    )

    accumulator = _MetricAccumulator(resolved_device, requested, resolved_keys)
    config = ValidationConfig(
        validation_data=evaluation_data,
        loss_fn=loss_fn or _metric_loss(supervised, resolved_keys),
        grad_mode="enabled" if differentiated else grad_mode,
        batch_callback=accumulator,
        name=name,
    )
    loop = ValidationLoop(
        validation_data=evaluation_data,
        config=config,
        device=resolved_device,
        model=model,
        validation_fn=validation_fn,
        distributed_manager=distributed_manager,
    )
    with loop:
        loop.execute()
    return accumulator.metrics(name=name, distributed_manager=distributed_manager)


@contextmanager
def _displaced(batch: Batch, positions: torch.Tensor) -> Iterator[None]:
    """Swap *positions* onto *batch* for the block, restoring the originals after."""
    original = batch.positions
    batch.positions = positions
    try:
        yield
    finally:
        batch.positions = original


def _per_graph_sum(values: torch.Tensor, batch: Batch) -> torch.Tensor:
    """Sum a per-atom scalar or vector into one entry per graph."""
    totals = values.new_zeros((batch.num_graphs, *values.shape[1:]))
    return totals.index_add_(0, batch.batch_idx, values)


def _probe_directions(
    batch: Batch, generator: torch.Generator | None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return two per-graph orthogonal displacement directions for *batch*.

    Each direction has unit root-mean-square per-atom norm within each graph,
    so displacing by ``amplitude`` times a direction moves every atom of every
    graph by ``amplitude`` in root-mean-square, whatever the graph's size. A
    graph-Frobenius normalization would instead shrink the per-atom step as
    ``1 / sqrt(N)`` and probe a 300-atom cell at a tenth of the displacement it
    probes a 3-atom one with.
    """
    positions = batch.positions
    device = positions.device if generator is None else generator.device
    shape = tuple(positions.shape)
    first = torch.randn(shape, generator=generator, device=device).to(positions)
    second = torch.randn(shape, generator=generator, device=device).to(positions)
    index = batch.batch_idx
    sizes = batch.num_nodes_per_graph.to(positions)
    overlap = _per_graph_sum((first * second).sum(dim=-1), batch)
    norm = _per_graph_sum(first.pow(2).sum(dim=-1), batch)
    second = second - (overlap / norm.clamp_min(_EPS))[index].unsqueeze(-1) * first
    first = first / (norm / sizes).sqrt().clamp_min(_EPS)[index].unsqueeze(-1)
    second_norm = _per_graph_sum(second.pow(2).sum(dim=-1), batch) / sizes
    second = second / second_norm.sqrt().clamp_min(_EPS)[index].unsqueeze(-1)
    return first, second


def nonconservative_residual(
    teacher: TeacherScorer | BaseModelMixin,
    data: Iterable[Batch] | Batch,
    *,
    num_loops: int = 4,
    amplitude: float = 0.05,
    segments: int = 4,
    generator: torch.Generator | None = None,
) -> NonConservativeResidual:
    r"""Estimate the part of a teacher's force field no student can fit.

    A student that differentiates an energy produces a curl-free field, so it
    can only ever fit the conservative part of a teacher trained to emit forces
    directly. This probe measures the rest, without assuming the teacher is
    differentiable and without training anything.

    **The estimator.** Around each graph, two per-graph orthogonal
    displacement directions :math:`u` and :math:`v`, each of unit
    root-mean-square per-atom norm, span a rectangular loop of side *amplitude*
    :math:`\varepsilon` through configuration space, from :math:`R` to
    :math:`R + \varepsilon u` to :math:`R + \varepsilon u + \varepsilon v` to
    :math:`R + \varepsilon v` and back — a loop every atom of which travels
    :math:`\varepsilon` per side in root-mean-square. The teacher's work around
    that closed loop, :math:`W = \oint F \cdot \mathrm{d}R`, is integrated with
    the midpoint rule using *segments* samples per side. For a conservative
    field the integrand is :math:`-\nabla E \cdot \mathrm{d}R` and :math:`W`
    vanishes identically, so what the probe reports is the non-conservative
    component alone.

    **The floor.** A conservative student makes force error
    :math:`\Delta F = F - F_{\text{student}}` with
    :math:`\oint \Delta F \cdot \mathrm{d}R = W`, and Cauchy-Schwarz over the
    graph's :math:`3N` configuration coordinates then gives
    :math:`|W| \le \max_t \lVert \Delta F \rVert_F \cdot L` along that loop,
    where :math:`L = 4 \varepsilon \sqrt{N}` is its configuration-space path
    length. Writing the Frobenius norm as :math:`\sqrt{N}` times the
    root-mean-square per-atom force error turns that into a per-atom statement,
    :math:`\max_t \Delta F_{\mathrm{rms}} \ge |W| / (4 \varepsilon N)`, and it
    is that per-atom bound that is reported as ``force_floor``.

    **What it does not measure.** The floor is a lower bound on the *largest*
    force error along a probed loop at a probed displacement scale, not a bound
    on the error averaged over a dataset, and it shrinks linearly with
    *amplitude* — a loop of zero size proves nothing. Choose *amplitude* to
    match the displacements the student will see, of the order of a thermal
    vibration. Nor is the bound tight for a large cell: one randomly oriented
    loop only sees the component of the field's curl that its own plane spans,
    which is a :math:`1 / \sqrt{3N}` fraction of it, so the floor of a
    size-extensive non-conservative field falls off as :math:`1 / \sqrt{N}` and
    floors are only comparable between probes of similar system size. A
    conservative teacher does not report exactly zero either; it reports the
    quadrature error of the midpoint rule, which falls as *segments* rises, and
    below that the round-off of the batch's own dtype: the displaced positions
    and the teacher's forces stay in the precision they arrived in, so a float32
    batch cannot resolve a loop closing to better than the resolution of its
    coordinates. However small *amplitude* is then made, the floor plateaus at
    a value of order the teacher's force constant times the float32 resolution
    of the centered coordinates — a model-dependent number, near ``1e-9`` eV/A
    for an argon-like Lennard-Jones solid and near ``1e-7`` eV/A for a lattice a
    hundred times stiffer. A floor below that needs a float64 batch and a
    float64 teacher. Either way it is what a comparison against a direct-force
    teacher should be read against.

    Parameters
    ----------
    teacher : TeacherScorer | BaseModelMixin
        Teacher whose field is probed. A bare model is wrapped in an
        :class:`~nvalchemi.training.distillation.InProcessTeacherScorer`, which
        also handles building and rolling back whatever neighbor list the
        teacher needs at each probe point.
    data : Iterable[Batch] | Batch
        Held-out structures to probe. Positions are displaced in place and
        restored before returning.
    num_loops : int, optional
        Loops integrated per graph. Default ``4``.
    amplitude : float, optional
        Loop side length in A, applied as a per-atom displacement. Default
        ``0.05``.
    segments : int, optional
        Midpoint-rule samples per side; each costs one teacher force
        evaluation, so one loop costs ``4 * segments``. Default ``4``.
    generator : torch.Generator | None, optional
        Generator drawing the loop directions. Default ``None`` (the global
        RNG).

    Returns
    -------
    NonConservativeResidual
        Loop work, the force floor it implies, and its size relative to the
        teacher's own force scale.

    Raises
    ------
    ValueError
        If *amplitude*, *num_loops*, or *segments* is not positive, if the
        scorer does not publish the teacher force field, or if *data* holds no
        graphs.

    Examples
    --------
    >>> from nvalchemi.training.distillation.evaluation import (
    ...     nonconservative_residual,
    ... )
    >>> residual = nonconservative_residual(teacher, holdout)  # doctest: +SKIP
    >>> residual.relative_floor  # doctest: +SKIP
    0.02
    """
    if amplitude <= 0.0 or num_loops <= 0 or segments <= 0:
        raise ValueError(
            "amplitude, num_loops, and segments must all be positive; got "
            f"amplitude={amplitude!r}, num_loops={num_loops!r}, "
            f"segments={segments!r}."
        )
    scorer = _as_scorer(teacher, ["forces"])
    works: list[torch.Tensor] = []
    sizes: list[torch.Tensor] = []
    graph_scales: list[torch.Tensor] = []
    force_squares: list[torch.Tensor] = []
    for batch in [data] if isinstance(data, Batch) else data:
        squares = (
            scorer.label(batch)["teacher_forces"][0]
            .pow(2)
            .sum(dim=-1)
            .flatten()
            .to(torch.float64)
        )
        force_squares.append(squares)
        base = batch.positions
        counts = batch.num_nodes_per_graph.to(torch.float64)
        scale = (_per_graph_sum(squares, batch) / counts).sqrt()
        for _ in range(num_loops):
            first, second = _probe_directions(batch, generator)
            works.append(
                _loop_work(scorer, batch, base, first, second, amplitude, segments)
            )
            sizes.append(counts)
            graph_scales.append(scale)
    if not works:
        raise ValueError("data must hold at least one graph to probe.")
    work = torch.cat(works).abs().to(torch.float64)
    floor = work / (4.0 * amplitude * torch.cat(sizes))
    relative = floor / torch.cat(graph_scales).clamp_min(_EPS)
    magnitudes = torch.cat(force_squares)
    return NonConservativeResidual(
        num_probes=int(work.numel()),
        amplitude=amplitude,
        segments=segments,
        loop_work_mean_abs=float(work.mean()),
        loop_work_max_abs=float(work.max()),
        force_floor=float(floor.mean()),
        force_floor_max=float(floor.max()),
        force_rms=float(magnitudes.mean().sqrt()),
        relative_floor=float(relative.mean()),
        relative_floor_max=float(relative.max()),
    )


def _loop_work(
    scorer: Any,
    batch: Batch,
    base: torch.Tensor,
    first: torch.Tensor,
    second: torch.Tensor,
    amplitude: float,
    segments: int,
) -> torch.Tensor:
    """Integrate the teacher's work around one closed rectangular loop.

    The samples very nearly cancel, so they are accumulated in float64: over a
    conservative teacher the residue is meant to report the midpoint rule's
    quadrature error rather than the roundoff of a float32 sum. The probe points
    themselves are laid out around each graph's own centroid rather than around
    wherever in space the frame was handed over, so the resolution a displaced
    position is representable at — and with it the floor a conservative teacher
    reports — does not depend on how far from the origin any graph sits.
    Centering on the batch's centroid instead would leave every graph offset by
    its distance to that centroid, and a float32 batch holding one frame far
    from the others would read a floor inflated by the coarser resolution there.
    """
    counts = batch.num_nodes_per_graph.to(base).unsqueeze(-1)
    centered = base - (_per_graph_sum(base, batch) / counts)[batch.batch_idx]
    corners = (
        torch.zeros_like(first),
        amplitude * first,
        amplitude * (first + second),
        amplitude * second,
    )
    work = base.new_zeros(batch.num_graphs, dtype=torch.float64)
    for index in range(4):
        start = corners[index]
        step = (corners[(index + 1) % 4] - start) / segments
        for sample in range(segments):
            with _displaced(batch, centered + (start + step * (sample + 0.5))):
                forces = scorer.label(batch)["teacher_forces"][0]
            contribution = (forces.to(torch.float64) * step.to(torch.float64)).sum(
                dim=-1
            )
            work = work + _per_graph_sum(contribution, batch)
    return work
