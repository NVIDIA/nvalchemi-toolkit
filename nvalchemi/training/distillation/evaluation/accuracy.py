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

:func:`evaluate_accuracy` runs a student over a held-out set and reports energy,
force, and stress errors against a reference dataset's own labels or a
teacher's; :func:`non_conservative_residual` measures the part of a teacher's
force field no conservative student can fit, the floor the first evaluation is
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
from nvalchemi.training.distillation.evaluation._export import _rebuild
from nvalchemi.training.distillation.hooks import _score_and_attach
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
    "non_conservative_residual",
]

AccuracyQuantity: TypeAlias = Literal["energy", "forces", "stress", "atomic_energies"]
"""Quantity an accuracy evaluation compares between a student and a target."""

_QUANTITY_SIGNALS: dict[str, str] = {
    "energy": "energy",
    "forces": "forces",
    "stress": "stress",
    "atomic_energies": "atomic_energies",
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
    of residuals divided by the total count, not a mean of per-batch means — in
    the units the batch carries: eV for energies, eV/A for forces, and the
    dataset's stress units. A quantity the pass could not measure, because it
    was not requested or no batch carried its prediction or target, is
    ``None``; one that was measured and came out non-finite is ``nan``, which
    an acceptance bar fails instead of reporting missing.

    Attributes
    ----------
    name : str
        Label carried into reports.
    num_graphs, num_atoms : int
        Graphs and atoms evaluated.
    energy_mae, energy_rmse : float | None
        Total-energy error per graph.
    energy_per_atom_mae, energy_per_atom_rmse : float | None
        Total-energy error divided by each graph's atom count.
    forces_mae, forces_rmse : float | None
        Force error per Cartesian component over every atom.
    stress_mae, stress_rmse : float | None
        Stress error per component over all nine components.
    force_cosine_mean : float | None
        Mean over atoms of the cosine between the predicted and target force,
        weighting every atom equally. An atom whose force sits at or below the
        student's own error scores an essentially random angle, so this number
        describes the holdout's low-force tail as much as the student. Atoms
        whose force vanishes on either side are not counted; ``None`` when no
        atom carries a force on both sides.
    force_cosine_aggregate : float | None
        Cosine between the two force fields taken as single vectors over the
        whole set, weighting atoms by force magnitude; the alignment an
        acceptance bar reads. A single non-finite atom makes it ``nan`` where
        ``force_cosine_mean`` still reports the atoms that stayed finite.
    atomic_energies_mae, atomic_energies_rmse : float | None
        Per-atom energy residual, when both sides publish a decomposition.
    force_nonfinite_atoms : int
        Atoms dropped from ``force_cosine_mean`` for a non-finite force.
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
    atomic_energies_mae: float | None = None
    atomic_energies_rmse: float | None = None
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

    A force field decomposes as :math:`F = -\nabla E + F_{\perp}`, and a student
    predicting forces as the gradient of an energy represents only the first
    term, whose work around any closed path vanishes; a closed-path integral of
    the teacher's field therefore measures :math:`F_{\perp}` alone. Every force
    here is a root-mean-square per-atom magnitude in eV/A, so ``force_floor``
    reads against an :class:`AccuracyMetrics` ``forces_rmse`` — a
    per-component figure, smaller by ``sqrt(3)`` for an isotropic error.

    Attributes
    ----------
    num_probes : int
        Closed loops integrated, one per graph per loop.
    amplitude : float
        Loop side length in A, as a root-mean-square per-atom displacement.
    segments : int
        Midpoint-rule samples per side.
    loop_work_mean_abs, loop_work_max_abs : float
        Mean and maximum absolute work around one loop, in eV.
    force_floor, force_floor_max : float
        Mean and maximum lower bound the loop work places on a conservative
        student's root-mean-square per-atom force error, in eV/A.
    force_rms : float
        Root-mean-square teacher force at the loop centers over every atom.
    relative_floor, relative_floor_max : float
        Mean and maximum of each probe's floor divided by the root-mean-square
        teacher force of its own graph, so a batch mixing force scales reports
        a figure between its graphs' ratios; a graph at equilibrium divides by
        a clamp rather than by zero.
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

    Placement is unconditional: :meth:`ValidationLoop.execute` copies with
    ``non_blocking=True``, which on a host destination can hand the loop a
    half-written batch, so every batch already sits on the run device when the
    loop receives it. The move clones, which keeps a scorer's ``teacher_*``
    fields off the caller's batches, and the scorer labels after the move,
    under the guards every labeling route shares: autocast disabled and a label
    outside ``teacher_*`` refused.
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
                _score_and_attach(self.scorer, placed)
            yield placed


class _MetricAccumulator:
    """Exact residual sums over a validation pass, as a per-batch callback.

    Implements :class:`~nvalchemi.training.BatchValidationCallback`, riding on a
    :class:`ValidationLoop` pass. Sums are float64 device tensors reduced once
    at the end, and every sum the requested quantities can produce is seeded at
    zero up front, so the packed all-reduce tensor has one shape and key order
    on every rank even when a shard carried no target for some quantity.
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

        The ``> 0.0`` test guards zero divided by zero and nothing else: a small
        force still counts at full weight in the per-atom mean. A non-finite atom
        leaves the per-atom mean the same way and is counted on the way out, but
        stays in the aggregate sums so the whole-set alignment reports as
        unmeasurable rather than averaged over what is left.
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
            atomic_energies_mae=atomic_mae,
            atomic_energies_rmse=atomic_rmse,
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

    A non-finite sum was measured and came out garbage, so it reports ``nan``
    for a bar to fail; a norm of exactly zero is a set whose forces all vanish
    and reports ``None`` like any quantity nobody measured.
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

    Resolved through the scoring signal surface, so a signal that moves the
    field it writes carries the evaluation with it.

    Raises
    ------
    RuntimeError
        If the signal behind a quantity publishes more than one field.
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
    dtype: torch.dtype | None = None,
) -> Any:
    """Return *model* as a scorer, wrapping a bare model in an in-process one.

    A supplied scorer is checked against the batch fields the requested signals
    are read from rather than the signal names it declares, since it may
    publish under fields of its own; one whose fields cannot be known is let
    through. *dtype* applies only to a wrapped bare model.
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
    return InProcessTeacherScorer(model, signals, dtype=dtype)


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

    The pass runs through :class:`~nvalchemi.training.ValidationLoop`, so eval
    mode and device placement behave as in training validation; the autograd
    policy is settled here first, since a student that differentiates inside
    its own forward needs gradients whatever is scored. No autocast is applied:
    the loop is built with no strategy and no
    :class:`~nvalchemi.training.hooks.mixed_precision.MixedPrecisionHook` to
    reuse a context from, so the student predicts in its own dtype. The
    metrics are exact global residual sums accumulated in float64, and the
    loop's graph-balanced loss value is discarded.

    ``targets="reference"`` compares against the dataset's own labels and
    ``"teacher"`` against the ``teacher_*`` fields, written offline by
    :func:`~nvalchemi.training.distillation.label_dataset` or on the fly by a
    *scorer*; against a teacher the force-alignment and per-atom energy
    diagnostics fill in as well. A *scorer* beside reference targets is
    refused: its pass would be paid for and thrown away.

    Parameters
    ----------
    model : BaseModelMixin
        Student to evaluate, left in the training mode it arrived in and scored
        on exactly the weights handed over: a student trained under an
        ``EMAHook`` needs ``strategy.inference_model`` here, and nothing
        downstream can tell which arrived, so record the choice on
        :class:`~nvalchemi.training.distillation.evaluation.StudentEvaluation`.
    data : Iterable[Batch]
        Re-iterable holdout set. One-shot iterators are rejected.
    targets : {"reference", "teacher"}, optional
        Family of batch fields to compare against. Default ``"reference"``.
    quantities : Sequence[AccuracyQuantity] | None, optional
        Quantities to evaluate; ``"atomic_energies"`` is a diagnostic that
        never enters the pass's loss. Default ``None`` (energy and forces).
    scorer : TeacherScorer | BaseModelMixin | None, optional
        Teacher labeling each batch before it is evaluated. A bare model is
        wrapped in an
        :class:`~nvalchemi.training.distillation.InProcessTeacherScorer` for
        the requested quantities, with its labels cast to the dtype the
        student's own labels are stored at; a supplied scorer is left uncast.
        Default ``None``.
    target_keys : Mapping[str, str] | None, optional
        Per-quantity overrides of the batch field to compare against, applied
        over the map *targets* selects. Default ``None``.
    loss_fn : ComposedLossFunction | None, optional
        Loss driving the pass. Default ``None`` (mean-squared terms over the
        requested supervised quantities).
    validation_fn : Callable, optional
        Forward callable invoked as ``validation_fn(model, batch)``. Default
        :func:`~nvalchemi.training.default_training_fn`.
    grad_mode : {"auto", "enabled", "disabled"}, optional
        Autograd policy. ``"auto"`` enables gradients whenever the student's
        forward needs them or the loss does; ``"disabled"`` is refused for a
        student whose forward differentiates. Default ``"auto"``.
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
        If *quantities* names an unknown quantity or no supervised one, if a
        *scorer* is given but no requested quantity is compared against a
        teacher field, does not publish the fields the evaluation reads, or
        returns a label outside ``teacher_*``, if gradients are disabled for a
        student that differentiates inside its forward, if a prediction and its
        target disagree on shape, or if no metric could be measured at all.

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
    carry one; a *scorer* builds and rolls back the teacher's own list per
    batch. ``"auto"`` reads the student's ``autograd_outputs`` — the declaration
    :meth:`~nvalchemi.models.base.BaseModelMixin.adapt_input` reads — so an
    autograd-force student is scored with gradients even when only energies are
    compared, and narrowing ``active_outputs`` puts it back on the
    ``torch.no_grad()`` path. Under a distributed run every rank must call this
    with the same *quantities* and a non-empty shard: the sums are packed in
    one key order before the all-reduce, so differently shaped packs would
    deadlock, and an empty shard raises out of the loop before the reduce and
    strands the others; shard sizes and the targets a shard carries may differ.
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

    Each has unit root-mean-square per-atom norm within each graph, so
    ``amplitude`` times a direction moves every atom by ``amplitude`` in
    root-mean-square whatever the graph's size, where a graph-Frobenius
    normalization would shrink the step as ``1 / sqrt(N)``.
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


def non_conservative_residual(
    teacher: TeacherScorer | BaseModelMixin,
    data: Iterable[Batch] | Batch,
    *,
    num_loops: int = 4,
    amplitude: float = 0.05,
    segments: int = 4,
    generator: torch.Generator | None = None,
) -> NonConservativeResidual:
    r"""Estimate the part of a teacher's force field no conservative student can fit.

    Around each graph, two per-graph orthogonal directions :math:`u` and
    :math:`v` of unit root-mean-square per-atom norm span a square loop of side
    *amplitude* :math:`\varepsilon` through configuration space, and the
    teacher's work :math:`W = \oint F \cdot \mathrm{d}R` around it is
    integrated with the midpoint rule at *segments* samples per side. A
    conservative field integrates to zero, so :math:`W` measures the
    non-conservative component alone, and Cauchy-Schwarz over the loop's path
    length :math:`4 \varepsilon \sqrt{N}` turns it into a lower bound on the
    largest root-mean-square per-atom force error a conservative student makes
    on that loop, :math:`|W| / (4 \varepsilon N)`, reported as ``force_floor``.

    The floor is a bound at the probed displacement scale, not a dataset-wide
    error bar: it shrinks linearly with *amplitude*, so choose one of the order
    of a thermal vibration, and one randomly oriented loop sees a
    :math:`1 / \sqrt{3N}` fraction of the field's curl, so floors are only
    comparable between probes of similar system size. A conservative teacher
    reports the midpoint rule's quadrature error and, below that, the round-off
    of the batch's own dtype; a floor below the float32 plateau needs a float64
    batch and teacher. See :ref:`distillation-evaluation` for the magnitudes.

    Parameters
    ----------
    teacher : TeacherScorer | BaseModelMixin
        Teacher whose field is probed. A bare model is wrapped in an
        :class:`~nvalchemi.training.distillation.InProcessTeacherScorer`, which
        builds and rolls back the teacher's neighbor list at each probe point.
    data : Iterable[Batch] | Batch
        Held-out structures to probe. Positions are displaced in place and
        restored before returning.
    num_loops : int, optional
        Loops integrated per graph. Default ``4``.
    amplitude : float, optional
        Loop side length in A, as a per-atom displacement. Default ``0.05``.
    segments : int, optional
        Midpoint-rule samples per side; one loop costs ``4 * segments`` teacher
        force evaluations. Default ``4``.
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
    ...     non_conservative_residual,
    ... )
    >>> residual = non_conservative_residual(teacher, holdout)  # doctest: +SKIP
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

    The samples nearly cancel, so they are accumulated in float64, and the
    probe points are laid out around each graph's own centroid so the
    resolution a displaced position is representable at — and the floor a
    conservative teacher reports — does not depend on where in space a graph
    sits or how far it lies from the others in the batch.
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
