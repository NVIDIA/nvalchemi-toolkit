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

:func:`evaluate_accuracy` runs a student over a held-out set. It reports energy,
force, and stress errors against either the reference dataset's own labels or a
teacher's labels. :func:`non_conservative_residual` measures the part of a
teacher's force field that no conservative student can fit. That residual is
the floor against which the accuracy evaluation's force error is read.
"""

from __future__ import annotations

import dataclasses
import functools
import math
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Literal, TypeAlias

import torch
from pydantic import Field

from nvalchemi.data import Batch, resolve_device
from nvalchemi.training import (
    ValidationConfig,
    ValidationLoop,
    default_training_fn,
    ensure_reiterable_validation_data,
    unwrap_model,
)
from nvalchemi.training.distillation.evaluation._export import MeasurementRecord
from nvalchemi.training.distillation.hooks import _score_and_attach
from nvalchemi.training.distillation.scoring import (
    TeacherScorer,
    _as_scorer,
    signal_fields,
)
from nvalchemi.training.distillation.strategy import (
    _student_label_dtype,
    _to_device,
)
from nvalchemi.training.distributed import (
    all_reduce,
    all_reduce_flags,
    is_distributed_initialized,
)
from nvalchemi.training.losses.composition import (
    BaseLossFunction,
    ComposedLossFunction,
)
from nvalchemi.training.losses.reductions import per_graph_sum
from nvalchemi.training.losses.terms import (
    EnergyMSELoss,
    ForceMSELoss,
    StressMSELoss,
)

if TYPE_CHECKING:
    from nvalchemi.models.base import BaseModelMixin

__all__ = [
    "BUILTIN_ACCURACY_QUANTITIES",
    "AccuracyMetrics",
    "AccuracyQuantity",
    "AccuracyQuantitySpec",
    "NonConservativeResidual",
    "evaluate_accuracy",
    "non_conservative_residual",
]

AccuracyQuantity: TypeAlias = Literal["energy", "forces", "stress", "atomic_energies"]
"""Built-in quantity an accuracy evaluation compares between a student and a target."""


@dataclasses.dataclass(frozen=True)
class AccuracyQuantitySpec:
    """Where one accuracy quantity's prediction and target are read from.

    The built-in quantities are :data:`BUILTIN_ACCURACY_QUANTITIES`. A custom
    quantity is passed to :func:`evaluate_accuracy` as an instance, and its
    errors are reported in ``AccuracyMetrics.errors`` under its name. It can be
    a head that the training function publishes under its own key, or a
    built-in quantity read from another field.

    Parameters
    ----------
    name : str
        Name the quantity is requested and reported under.
    prediction_key : str
        Key of the validation function's output holding the prediction.
    reference_key : str
        Batch field holding the dataset's own label.
    signal : str | None, optional
        Teacher signal the quantity is compared against when
        ``targets="teacher"``. The signal is resolved to its batch field
        through :func:`~nvalchemi.training.distillation.signal_fields`.
        Default ``None`` (no teacher target; name a teacher field in
        ``target_keys`` to compare against a teacher).
    supervised_loss : Callable[..., BaseLossFunction] | None, optional
        Factory, called as ``supervised_loss(target_key=...)``, that builds the
        loss term driving the validation pass. Default ``None`` (the quantity
        is a diagnostic that never enters the pass's loss).

    Examples
    --------
    >>> from nvalchemi.training.distillation.evaluation import (
    ...     AccuracyQuantitySpec,
    ... )
    >>> charges = AccuracyQuantitySpec(
    ...     "charges", prediction_key="predicted_charges", reference_key="charges"
    ... )
    >>> charges.supervised_loss is None
    True
    """

    name: str
    prediction_key: str
    reference_key: str
    signal: str | None = None
    supervised_loss: Callable[..., BaseLossFunction] | None = None


BUILTIN_ACCURACY_QUANTITIES: Mapping[str, AccuracyQuantitySpec] = {
    "energy": AccuracyQuantitySpec(
        "energy",
        "predicted_energy",
        "energy",
        signal="energy",
        supervised_loss=functools.partial(EnergyMSELoss, per_atom=True),
    ),
    "forces": AccuracyQuantitySpec(
        "forces",
        "predicted_forces",
        "forces",
        signal="forces",
        supervised_loss=ForceMSELoss,
    ),
    "stress": AccuracyQuantitySpec(
        "stress",
        "predicted_stress",
        "stress",
        signal="stress",
        supervised_loss=StressMSELoss,
    ),
    "atomic_energies": AccuracyQuantitySpec(
        "atomic_energies",
        "predicted_atomic_energies",
        "atomic_energies",
        signal="atomic_energies",
    ),
}
"""Specs of the built-in quantities, keyed by the name a caller requests them under."""

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


class AccuracyMetrics(MeasurementRecord):
    """Errors of one student against one set of targets over a held-out set.

    Every metric is an exact global reduction over the evaluated set: the sum
    of residuals divided by the total count, not a mean of per-batch means.
    Metrics are in the units the batch carries: eV for energies, eV/A for
    forces, and the dataset's stress units. A quantity the pass could not
    measure is ``None``. That happens when the quantity was not requested, or
    when no batch carried its prediction or target. A quantity that was
    measured but came out non-finite is ``nan``. An acceptance bar fails a
    ``nan`` value instead of reporting it as missing.

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
        weighting every atom equally. An atom whose force is at or below the
        student's own error has an essentially random angle, so this number
        describes the held-out set's low-force tail as much as the student.
        Atoms whose force vanishes on either side are not counted. ``None``
        when no atom carries a force on both sides.
    force_cosine_aggregate : float | None
        Cosine between the two force fields, each taken as a single vector over
        the whole set, so atoms are weighted by force magnitude. This is the
        alignment an acceptance bar reads. A single non-finite atom makes it
        ``nan``, whereas ``force_cosine_mean`` still reports the atoms that
        stayed finite.
    atomic_energies_mae, atomic_energies_rmse : float | None
        Per-atom energy error, measured when both the prediction and the target
        include a per-atom energy decomposition.
    force_nonfinite_atoms : int
        Atoms dropped from ``force_cosine_mean`` for a non-finite force.
    errors : Mapping[str, Mapping[str, float]]
        ``{"mae": ..., "rmse": ...}`` per custom :class:`AccuracyQuantitySpec`
        that was measured; the built-in quantities report through their own
        fields.
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
    errors: dict[str, dict[str, float]] = Field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return the populated fields as a plain dictionary.

        A quantity that was not measured is left out rather than exported as
        ``None``, and :meth:`from_dict` restores it as ``None``. An empty
        ``errors`` map is left out the same way.
        """
        exported = self.model_dump(exclude_none=True)
        if not exported["errors"]:
            del exported["errors"]
        return exported


class NonConservativeResidual(MeasurementRecord):
    r"""Non-conservative component of a teacher's force field.

    The *non-conservative residual* is the part of a teacher's force field that
    no conservative student can fit. A force field decomposes as
    :math:`F = -\nabla E + F_{\perp}`. A student that predicts forces as the
    gradient of an energy represents only the first term, whose work around any
    closed path vanishes. A closed-path integral of the teacher's field
    therefore measures :math:`F_{\perp}` alone. Every force here is a
    root-mean-square per-atom magnitude in eV/A. Compare ``force_floor`` with
    the ``forces_rmse`` of :class:`AccuracyMetrics` accordingly: ``forces_rmse``
    is a per-component figure, smaller by ``sqrt(3)`` for an isotropic error.

    Attributes
    ----------
    num_probes : int
        Closed loops integrated in total, ``num_loops`` per graph.
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
        teacher force of its own graph. A batch that mixes force scales
        therefore reports a figure between its graphs' ratios. A graph at
        equilibrium is divided by a small clamp rather than by zero.
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


class _PlacedBatches:
    """Re-iterable view that places each batch on *device* and labels it if scored.

    Every batch is placed before the loop sees it, whatever device it starts
    on. The placement is a clone: a copy into device memory is asynchronous, a
    copy into host memory blocks so the moved batch's pointers can be read at
    once, and a batch already on the device is cloned without a transfer.
    Because it is a clone, a scorer's ``teacher_*`` fields stay off the
    caller's batches. The scorer labels after the placement, at the precision
    its own ``autocast`` setting selects, and a label outside ``teacher_*`` is
    refused.
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
    """Per-batch callback accumulating exact residual sums over a validation pass.

    Implements :class:`~nvalchemi.training.BatchValidationCallback` and runs
    inside a :class:`ValidationLoop` pass. Sums are float64 device tensors,
    reduced once at the end. Every sum the requested quantities can produce is
    seeded at zero up front. The packed all-reduce tensor therefore has the
    same shape and key order on every rank, even when a shard carried no target
    for some quantity.
    """

    def __init__(
        self,
        device: torch.device,
        quantities: Sequence[AccuracyQuantitySpec],
        target_keys: Mapping[str, str],
    ) -> None:
        self.device = device
        self.specs = tuple(quantities)
        self.quantities = tuple(spec.name for spec in self.specs)
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
        for spec in self.specs:
            quantity = spec.name
            prediction = predictions.get(spec.prediction_key)
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

        Shapes must match exactly. Broadcasting a ``(B,)`` target against a
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

        The ``> 0.0`` test only guards against zero divided by zero, so a small
        force still counts at full weight in the per-atom mean. A non-finite
        atom is also dropped from the per-atom mean and is counted in
        ``force_nonfinite_atoms``. It stays in the aggregate sums, so the
        whole-set alignment reports as unmeasurable rather than as an average
        over the atoms that are left.
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

        Whether this rank measured anything is decided locally and exchanged
        with :func:`~nvalchemi.training.distributed.all_reduce_flags` before
        the sums are reduced, so every rank raises the same refusal, naming
        the ranks that came up empty, and none is left waiting in the
        all-reduce for a rank that raised.

        Raises
        ------
        ValueError
            If a rank measured no quantity, meaning every batch it saw was
            missing either the prediction or the target of every requested
            quantity.
        """
        keys = tuple(sorted(self._sums))
        packed = torch.stack([self._sums[key] for key in keys])
        measured = any(
            float(self._sums[key]) > 0.0 for key in keys if key.endswith("_count")
        )
        empty = all_reduce_flags(not measured, distributed_manager)
        if bool(empty.any()):
            raise ValueError(
                "No accuracy metric could be measured on rank(s) "
                f"{empty.nonzero().flatten().tolist()!r}; every batch there was "
                "missing the prediction or the target of every requested quantity "
                f"{list(self.quantities)!r} (targets {self.target_keys!r})."
            )
        if is_distributed_initialized(distributed_manager):
            all_reduce(packed, distributed_manager)
        totals = {key: float(packed[index]) for index, key in enumerate(keys)}
        energy_mae, energy_rmse = _mae_rmse(totals, "energy")
        per_atom_mae, per_atom_rmse = _mae_rmse(totals, "energy_per_atom")
        forces_mae, forces_rmse = _mae_rmse(totals, "forces")
        stress_mae, stress_rmse = _mae_rmse(totals, "stress")
        atomic_mae, atomic_rmse = _mae_rmse(totals, "atomic_energies")
        errors = {}
        for quantity in self.quantities:
            if quantity in BUILTIN_ACCURACY_QUANTITIES:
                continue
            mae, rmse = _mae_rmse(totals, quantity)
            if mae is not None:
                errors[quantity] = {"mae": mae, "rmse": rmse}
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
            errors=errors,
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
    """Return ``numerator / denominator``, or ``None`` when either is missing.

    A denominator of zero also yields ``None``.
    """
    if numerator is None or not denominator:
        return None
    return numerator / denominator


def _aggregate_cosine(totals: Mapping[str, float]) -> float | None:
    """Return the cosine similarity of the two force fields taken as one vector.

    A non-finite sum means the alignment was measured but is unusable, so it
    reports ``nan`` for a bar to fail. A norm of exactly zero means every force
    in the set vanishes, so it reports ``None``, like any quantity that was not
    measured.
    """
    dot = totals.get("force_dot")
    if dot is None:
        return None
    squares = totals["force_predicted_sq"] * totals["force_target_sq"]
    if not math.isfinite(dot) or not math.isfinite(squares):
        return math.nan
    norm = math.sqrt(squares)
    return dot / norm if norm > 0.0 else None


def _teacher_target_keys(specs: Sequence[AccuracyQuantitySpec]) -> dict[str, str]:
    """Return the teacher field each quantity with a signal is compared against.

    Each signal is resolved to its field through the scoring module, so if a
    signal changes the field it writes, the evaluation follows it. A spec
    without a signal has no teacher field and is left out.

    Raises
    ------
    RuntimeError
        If the signal behind a quantity publishes more than one field.
    """
    resolved: dict[str, str] = {}
    for spec in specs:
        if spec.signal is None:
            continue
        fields = signal_fields([spec.signal])
        if len(fields) != 1:
            raise RuntimeError(
                f"Quantity {spec.name!r} is compared against a single teacher "
                f"field, but its signal {spec.signal!r} publishes {list(fields)!r}."
            )
        resolved[spec.name] = fields[0]
    return resolved


def _resolve_quantities(
    quantities: Sequence[str | AccuracyQuantitySpec] | None,
) -> tuple[AccuracyQuantitySpec, ...]:
    """Return the specs *quantities* request, built-ins looked up by name.

    Raises
    ------
    ValueError
        If a name is not a built-in quantity, or two specs share a name.
    """
    if quantities is None:
        return tuple(BUILTIN_ACCURACY_QUANTITIES[name] for name in _DEFAULT_QUANTITIES)
    unknown = sorted(
        quantity
        for quantity in quantities
        if isinstance(quantity, str) and quantity not in BUILTIN_ACCURACY_QUANTITIES
    )
    if unknown:
        raise ValueError(
            "Accuracy quantities must be names from "
            f"{sorted(BUILTIN_ACCURACY_QUANTITIES)!r} or AccuracyQuantitySpec "
            f"instances; got unsupported {unknown!r}."
        )
    specs = tuple(
        BUILTIN_ACCURACY_QUANTITIES[quantity] if isinstance(quantity, str) else quantity
        for quantity in quantities
    )
    names = [spec.name for spec in specs]
    if len(set(names)) != len(names):
        raise ValueError(f"Accuracy quantity names must be unique; got {names!r}.")
    return specs


def _resolve_device(model: Any, device: torch.device | str | None) -> torch.device:
    """Return the requested device, else the device the model's parameters sit on.

    An explicit request goes through :func:`~nvalchemi.data.resolve_device`,
    so an index-less ``"cuda"`` names the current CUDA device and compares
    equal to the device the placed batches report.
    """
    if device is not None:
        return resolve_device(device)
    parameters = getattr(model, "parameters", None)
    if callable(parameters):
        for parameter in parameters():
            return parameter.device
    return torch.device("cpu")


def _requires_autograd(model: Any) -> bool:
    """Return whether a model's own forward pass needs autograd enabled around it.

    It reads :attr:`~nvalchemi.models.base.BaseModelMixin.requires_autograd`
    through :func:`~nvalchemi.training.unwrap_model`, so a parallelism
    wrapper around the student does not hide the declaration. A model without
    the attribute is taken to publish direct outputs only.
    """
    return bool(getattr(unwrap_model(model), "requires_autograd", False))


def _metric_loss(
    specs: Sequence[AccuracyQuantitySpec], target_keys: Mapping[str, str]
) -> ComposedLossFunction:
    """Build the composed loss whose gradient requirement drives the pass."""
    terms = [
        spec.supervised_loss(target_key=target_keys[spec.name])
        for spec in specs
        if spec.supervised_loss is not None
    ]
    if not terms:
        raise ValueError(
            "At least one quantity carrying a supervised loss ('energy', 'forces', "
            "or 'stress' among the built-ins) must be evaluated so the validation "
            f"pass has a loss to run; got {[spec.name for spec in specs]!r}."
        )
    return ComposedLossFunction(terms, dtype_policy="prediction_to_target")


def evaluate_accuracy(
    model: BaseModelMixin,
    data: Iterable[Batch],
    *,
    targets: Literal["reference", "teacher"] = "reference",
    quantities: Sequence[str | AccuracyQuantitySpec] | None = None,
    scorer: TeacherScorer | BaseModelMixin | None = None,
    target_keys: Mapping[str, str] | None = None,
    loss_fn: ComposedLossFunction | None = None,
    validation_fn: Callable[..., Any] = default_training_fn,
    grad_mode: Literal["auto", "enabled", "disabled"] = "auto",
    label_dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
    distributed_manager: Any | None = None,
    name: str = "accuracy",
) -> AccuracyMetrics:
    """Measure a student's error over a held-out set.

    The pass runs through :class:`~nvalchemi.training.ValidationLoop`, so eval
    mode and device placement behave as in training validation. The autograd
    policy is settled before the loop runs, because a student that
    differentiates inside its own forward needs gradients whatever is scored.
    The loop is built without an autocast context, so the student predicts in
    its own dtype. A bare *scorer* model labels with autocast disabled, the
    in-process scorer's default; a supplied scorer decides its own precision.
    The metrics are exact global residual sums accumulated in float64. The
    loop's graph-balanced loss value is discarded.

    ``targets="reference"`` compares against the dataset's own labels.
    ``targets="teacher"`` compares against the ``teacher_*`` fields, which
    :func:`~nvalchemi.training.distillation.label_dataset` writes offline or a
    *scorer* writes on the fly. Against either family, the force-alignment
    numbers fill in whenever forces are compared, and the per-atom energy
    diagnostic whenever ``"atomic_energies"`` is requested and both sides carry
    it. A *scorer* passed with reference targets is refused, because its
    teacher pass would be paid for and thrown away.

    Parameters
    ----------
    model : BaseModelMixin
        Student to evaluate. The call leaves it in the training mode it arrived
        in, and scores exactly the weights passed. For a student trained under
        an ``EMAHook``, pass ``strategy.inference_model``. Nothing downstream
        can tell which weights were scored, so record the choice on
        :class:`~nvalchemi.training.distillation.evaluation.StudentEvaluation`.
    data : Iterable[Batch]
        Re-iterable held-out set. One-shot iterators are rejected.
    targets : {"reference", "teacher"}, optional
        Family of batch fields to compare against. Default ``"reference"``.
    quantities : Sequence[str | AccuracyQuantitySpec] | None, optional
        Quantities to evaluate. Each entry is a built-in name from
        :data:`BUILTIN_ACCURACY_QUANTITIES`, or an
        :class:`AccuracyQuantitySpec` instance for a custom quantity, which is
        reported under ``errors``. The built-in ``"atomic_energies"`` is a
        diagnostic that never enters the pass's loss. Default ``None`` (energy
        and forces).
    scorer : TeacherScorer | BaseModelMixin | None, optional
        Teacher that labels each batch before it is evaluated. A bare model is
        wrapped in an
        :class:`~nvalchemi.training.distillation.InProcessTeacherScorer` for
        the requested quantities, and its labels are cast to *label_dtype*. A
        supplied scorer's labels are not cast. Default ``None``.
    target_keys : Mapping[str, str] | None, optional
        Per-quantity overrides of the batch field to compare against. They take
        precedence over the fields *targets* selects. Default ``None``.
    loss_fn : ComposedLossFunction | None, optional
        Loss driving the pass. Default ``None`` (mean-squared terms over the
        requested supervised quantities).
    validation_fn : Callable, optional
        Forward callable invoked as ``validation_fn(model, batch)``. Default
        :func:`~nvalchemi.training.default_training_fn`.
    grad_mode : {"auto", "enabled", "disabled"}, optional
        Autograd policy. ``"auto"`` enables gradients whenever the student's
        forward needs them or the loss does. ``"disabled"`` is refused for a
        student whose forward differentiates. Default ``"auto"``.
    label_dtype : torch.dtype | None, optional
        Dtype that a bare *scorer* model's labels are cast to. Default ``None``
        (the dtype the distillation strategy infers for the student's own
        labels: the dtype of its first floating-point parameter, floored at
        float32).
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
        If *quantities* names an unknown quantity, repeats a name, or includes
        no supervised quantity. If a quantity without a teacher signal is
        compared against the teacher and has no ``target_keys`` entry. If a
        *scorer* is given but no requested quantity is compared against a
        teacher field, or the scorer does not publish the fields the
        evaluation reads, or it returns a label outside ``teacher_*``. If
        gradients are disabled for a student that differentiates inside its
        forward. If a prediction and its target disagree on shape, or if no
        metric could be measured at all.

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
    would call it. A student that reads a neighbor list therefore needs batches
    that carry one. A *scorer* builds the teacher's own list for each batch and
    rolls it back afterwards.

    ``grad_mode="auto"`` reads the student's ``autograd_outputs``, the same
    declaration that :meth:`~nvalchemi.models.base.BaseModelMixin.adapt_input`
    reads, and looks through any ``DistributedDataParallel`` or similar wrapper
    to find it. An autograd-force student is therefore scored with gradients
    even when only energies are compared. Narrowing its ``active_outputs`` puts
    it back on the ``torch.no_grad()`` path.

    Under a distributed run, every rank must call this function with the same
    *quantities* and a non-empty shard. The sums are packed in one key order
    before the all-reduce, so packs of different shapes would deadlock. An
    empty shard raises out of the loop before the reduce and strands the other
    ranks. Shard sizes, and the targets each shard carries, may differ.
    """
    specs = _resolve_quantities(quantities)
    requested = tuple(spec.name for spec in specs)
    teacher_keys = _teacher_target_keys(specs)
    base = (
        teacher_keys
        if targets == "teacher"
        else {spec.name: spec.reference_key for spec in specs}
    )
    resolved_keys = dict(base) | dict(target_keys or {})
    unresolved = sorted(set(requested) - set(resolved_keys))
    if unresolved:
        raise ValueError(
            f"Quantities {unresolved!r} declare no teacher signal, so "
            f"targets={targets!r} has no field to compare them against; name "
            "the teacher field in target_keys or give the spec a signal."
        )
    compared = {resolved_keys[quantity] for quantity in requested}
    if scorer is not None and not (compared & set(teacher_keys.values())):
        raise ValueError(
            f"A scorer was given, but targets={targets!r} compares against "
            f"{sorted(compared)!r}, none of which is a teacher field. The teacher "
            "pass would be paid and thrown away, and the reported errors would be "
            "against the dataset's own labels rather than the teacher's. Pass "
            "targets='teacher', drop the scorer, or name the teacher fields to "
            "compare against in target_keys."
        )
    differentiated = _requires_autograd(model)
    if differentiated and grad_mode == "disabled":
        raise ValueError(
            f"Student {type(model).__name__!r} computes an active output by "
            "differentiating inside its own forward, so it needs autograd "
            "whatever is being scored; got grad_mode='disabled'. Pass "
            "grad_mode='auto', or narrow the student's model_config.active_outputs "
            "so it stops differentiating."
        )
    resolved_device = _resolve_device(model, device)

    signals = [spec.signal for spec in specs if spec.signal is not None]
    evaluation_data: Iterable[Batch] = _PlacedBatches(
        ensure_reiterable_validation_data(data),
        resolved_device,
        scorer=(
            _as_scorer(
                scorer,
                signals,
                _student_label_dtype(model) if label_dtype is None else label_dtype,
            )
            if scorer is not None
            else None
        ),
    )

    accumulator = _MetricAccumulator(resolved_device, specs, resolved_keys)
    config = ValidationConfig(
        validation_data=evaluation_data,
        loss_fn=loss_fn or _metric_loss(specs, resolved_keys),
        grad_mode=grad_mode,
        batch_callback=accumulator,
        name=name,
    )
    loop = ValidationLoop(
        validation_data=evaluation_data,
        config=config,
        device=resolved_device,
        model=model,
        validation_fn=validation_fn,
        grad_enabled=True if differentiated else None,
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


def _probe_directions(
    batch: Batch, generator: torch.Generator | None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return two per-graph orthogonal displacement directions for *batch*.

    Each direction has unit root-mean-square per-atom norm within each graph.
    Scaling a direction by ``amplitude`` therefore moves the atoms by
    ``amplitude`` in root-mean-square, whatever the graph's size. Normalizing
    by the graph's Frobenius norm instead would shrink the step as
    ``1 / sqrt(N)``.
    """
    positions = batch.positions
    device = positions.device if generator is None else generator.device
    shape = tuple(positions.shape)
    first = torch.randn(shape, generator=generator, device=device).to(positions)
    second = torch.randn(shape, generator=generator, device=device).to(positions)
    index = batch.batch_idx
    sizes = batch.num_nodes_per_graph.to(positions)
    overlap = per_graph_sum(
        (first * second).sum(dim=-1), batch.batch_idx, num_graphs=batch.num_graphs
    )
    norm = per_graph_sum(
        first.pow(2).sum(dim=-1), batch.batch_idx, num_graphs=batch.num_graphs
    )
    second = second - (overlap / norm.clamp_min(_EPS))[index].unsqueeze(-1) * first
    first = first / (norm / sizes).sqrt().clamp_min(_EPS)[index].unsqueeze(-1)
    second_norm = (
        per_graph_sum(
            second.pow(2).sum(dim=-1), batch.batch_idx, num_graphs=batch.num_graphs
        )
        / sizes
    )
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

    For each graph, two orthogonal directions :math:`u` and :math:`v`, each
    with unit root-mean-square per-atom norm, span a square loop of side
    *amplitude* :math:`\varepsilon` through configuration space. The teacher's
    work :math:`W = \oint F \cdot \mathrm{d}R` around the loop is integrated
    with the midpoint rule at *segments* samples per side. A conservative field
    integrates to zero, so :math:`W` measures the non-conservative component
    alone. The loop's path length is :math:`4 \varepsilon \sqrt{N}`, so by
    Cauchy-Schwarz :math:`|W| / (4 \varepsilon N)` is a lower bound on the
    largest root-mean-square per-atom force error a conservative student makes
    on that loop. This bound is reported as ``force_floor``.

    The floor is a bound at the probed displacement scale, not a dataset-wide
    error bar. It shrinks linearly with *amplitude*, so choose an amplitude on
    the order of a thermal vibration. One randomly oriented loop sees a
    :math:`1 / \sqrt{3N}` fraction of the field's curl, so floors are
    comparable only between probes of similar system size. A conservative
    teacher reports the midpoint rule's quadrature error and, below that, the
    round-off of the batch's own dtype. A floor below the float32 round-off
    plateau needs a float64 batch and teacher. See
    :ref:`distillation-evaluation` for the magnitudes.

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
        scale = (
            per_graph_sum(squares, batch.batch_idx, num_graphs=batch.num_graphs)
            / counts
        ).sqrt()
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

    The samples nearly cancel, so they are accumulated in float64. The probe
    points are laid out around each graph's own centroid. The precision with
    which a displaced position is represented, and so the floor a conservative
    teacher reports, then does not depend on where in space a graph sits or
    how far it lies from the other graphs in the batch.
    """
    counts = batch.num_nodes_per_graph.to(base).unsqueeze(-1)
    centered = (
        base
        - (per_graph_sum(base, batch.batch_idx, num_graphs=batch.num_graphs) / counts)[
            batch.batch_idx
        ]
    )
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
            work = work + per_graph_sum(
                contribution, batch.batch_idx, num_graphs=batch.num_graphs
            )
    return work
