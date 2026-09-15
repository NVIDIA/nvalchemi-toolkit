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
"""Tests for :mod:`nvalchemi.training.distillation.evaluation.accuracy`."""

from __future__ import annotations

import itertools
import math
from typing import Any
from unittest.mock import patch

import pytest
import torch

from nvalchemi.data import AtomicData, Batch
from nvalchemi.models.lj import LennardJonesModelWrapper
from nvalchemi.neighbors import compute_neighbors
from nvalchemi.training import ValidationLoop
from nvalchemi.training.distillation.evaluation import accuracy as accuracy_module
from nvalchemi.training.distillation.evaluation import (
    evaluate_accuracy,
    extensivity_error,
    nonconservative_residual,
)
from nvalchemi.training.distillation.scoring import InProcessTeacherScorer
from nvalchemi.training.distillation.strategy import _student_label_dtype
from nvalchemi.training.losses.terms import EnergyMSELoss
from nvalchemi.training.strategy import default_training_fn
from test.training.conftest import _build_atomic_data, _build_batch, _build_demo_model
from test.training.distillation.conftest import (
    _build_direct_force_teacher,
    _build_lattice_batch,
    _build_lattice_data,
    _build_lj_teacher,
)

_BATCH_CENTERED_FLOOR = 0.0003601186815915754
"""Float64 floor a 64-atom curl lattice reported while centering was batch-wide."""

_RAGGED_SIZES = (3, 7, 2, 11, 5)
"""Atom counts of the placement holdout's graphs, distinct so its pointers are too."""

_PLACEMENT_TRIALS = 32
"""Evaluations per placement test, enough for an unsynchronized copy to surface."""


def _make_holdout(sizes: tuple[int, ...] = (2, 5)) -> list[Batch]:
    """Return one batch per entry of *sizes*, so per-batch means differ."""
    return [
        _build_batch(n_systems=1, n_atoms_each=size, seed=40 + index)
        for index, size in enumerate(sizes)
    ]


def _make_ragged_holdout() -> list[Batch]:
    """Return a one-batch holdout whose graphs hold distinct atom counts."""
    return [
        Batch.from_data_list(
            [
                _build_atomic_data(size, seed=90 + index)
                for index, size in enumerate(_RAGGED_SIZES)
            ]
        )
    ]


def _make_lj_batch(n_atoms: int = 6, cell_length: float = 20.0) -> Batch:
    """Return one periodic argon batch with a Lennard-Jones neighbor list built."""
    generator = torch.Generator().manual_seed(11)
    data = AtomicData(
        positions=torch.rand(n_atoms, 3, generator=generator) * 8.0,
        atomic_numbers=torch.full((n_atoms,), 18, dtype=torch.long),
        atomic_masses=torch.full((n_atoms,), 39.948),
        cell=torch.eye(3).unsqueeze(0) * cell_length,
        pbc=torch.ones(1, 3, dtype=torch.bool),
    )
    batch = Batch.from_data_list([data])
    compute_neighbors(batch, cutoff=5.0)
    return batch


def _make_lattice_holdout() -> list[Batch]:
    """Return a jittered argon frame and the equilibrium frame, forces built.

    Every atom of the equilibrium frame sits at a symmetric site, so its
    Lennard-Jones force cancels to numerical zero — the ordinary case of a
    holdout carrying relaxed structures.
    """
    batches = [_build_lattice_batch(jitter=0.25), _build_lattice_batch()]
    for batch in batches:
        compute_neighbors(batch, cutoff=5.0)
    return batches


def _make_curl_lattice(images: int) -> Batch:
    """Return a cubic cell holding a whole period of :class:`_CurlScorer`'s field.

    The spacing is a quarter period, so replicating the cell along ``x`` leaves
    every atom's force exactly as it was and the two cells differ only in size.
    """
    spacing = math.pi / 2
    cells = (4 * images, 4, 4)
    positions = torch.tensor(
        [
            [i * spacing, j * spacing, k * spacing]
            for i, j, k in itertools.product(*(range(count) for count in cells))
        ],
        dtype=torch.float64,
    )
    lengths = torch.tensor([count * spacing for count in cells], dtype=torch.float64)
    data = AtomicData(
        positions=positions,
        atomic_numbers=torch.full((positions.shape[0],), 18, dtype=torch.long),
        atomic_masses=torch.full((positions.shape[0],), 39.948, dtype=torch.float64),
        cell=torch.diag(lengths).unsqueeze(0),
        pbc=torch.ones(1, 3, dtype=torch.bool),
    )
    return Batch.from_data_list([data])


def _make_grid_data(cells: int, spacing: float = 2.8) -> AtomicData:
    """Return one float64 cubic grid of ``cells ** 3`` argon atoms."""
    positions = torch.tensor(
        [
            [i * spacing, j * spacing, k * spacing]
            for i, j, k in itertools.product(range(cells), repeat=3)
        ],
        dtype=torch.float64,
    )
    n_atoms = positions.shape[0]
    return AtomicData(
        positions=positions,
        atomic_numbers=torch.full((n_atoms,), 18, dtype=torch.long),
        atomic_masses=torch.full((n_atoms,), 39.948, dtype=torch.float64),
        cell=(torch.eye(3, dtype=torch.float64) * (cells * spacing)).unsqueeze(0),
        pbc=torch.ones(1, 3, dtype=torch.bool),
    )


def _make_translated_pair(offset: float) -> Batch:
    """Return two identical float32 argon lattices, the second *offset* A away."""
    near = _build_lattice_data(cells=3, jitter=0.2)
    far = _build_lattice_data(cells=3, jitter=0.2)
    far.positions = far.positions + offset
    return Batch.from_data_list([near, far])


def _per_graph_relative_floor(
    scorer: _ScaledCurlScorer, batch: Batch, amplitude: float, seed: int
) -> torch.Tensor:
    """Return each graph's own floor over its own force scale, computed by hand.

    :class:`_ScaledCurlScorer`'s field is linear in the positions, so the
    midpoint rule is exact and one loop of side *amplitude* accumulates
    ``2 * amplitude ** 2 * scale`` times the graph's summed ``u x v`` cross
    term, whatever *segments* the probe used.
    """
    generator = torch.Generator().manual_seed(seed)
    first, second = accuracy_module._probe_directions(batch, generator)
    cross = first[:, 0] * second[:, 1] - first[:, 1] * second[:, 0]
    positions = batch.positions
    scales = torch.tensor(scorer.scales, dtype=positions.dtype)
    counts = batch.num_nodes_per_graph.to(positions)
    work = 2.0 * amplitude**2 * scales * accuracy_module._per_graph_sum(cross, batch)
    floor = work.abs() / (4.0 * amplitude * counts)
    squares = scorer.label(batch)["teacher_forces"][0].pow(2).sum(dim=-1)
    scale = (accuracy_module._per_graph_sum(squares, batch) / counts).sqrt()
    return floor / scale


def _center_per_graph(batch: Batch) -> torch.Tensor:
    """Return *batch*'s positions with each graph's own centroid removed."""
    positions = batch.positions
    totals = positions.new_zeros(batch.num_graphs, 3).index_add_(
        0, batch.batch_idx, positions
    )
    counts = batch.num_nodes_per_graph.to(positions).unsqueeze(-1)
    return positions - (totals / counts)[batch.batch_idx]


def _probe_displacement(batch: Batch, amplitude: float = 0.05) -> torch.Tensor:
    """Return the per-atom squared displacement of every point a probe visits.

    The probe lays its loops out around each graph's own centroid, so the
    geometry it visits is measured against the per-graph centered positions
    rather than against the ones the batch arrived carrying.
    """
    scorer = _RecordingScorer()
    nonconservative_residual(
        scorer,
        batch,
        num_loops=2,
        amplitude=amplitude,
        generator=torch.Generator().manual_seed(0),
    )
    base = _center_per_graph(batch)
    visited = torch.stack(scorer.positions[1:])
    return (visited - base).pow(2).sum(dim=-1)


def _reference_force_error(model: Any, batches: list[Batch]) -> tuple[float, float]:
    """Return the hand-computed global force MAE and RMSE over *batches*.

    Residuals are formed in float64, the way the accumulator forms them, so a
    student predicting in some other precision is compared the same way.
    """
    absolute = 0.0
    squared = 0.0
    count = 0
    for batch in batches:
        predicted = default_training_fn(model, batch)["predicted_forces"]
        residual = predicted.detach().to(torch.float64) - batch.forces.to(torch.float64)
        absolute += float(residual.abs().sum())
        squared += float(residual.pow(2).sum())
        count += residual.numel()
    return absolute / count, (squared / count) ** 0.5


def _reference_energy_error(model: Any, batches: list[Batch]) -> tuple[float, float]:
    """Return the hand-computed per-atom energy MAE and RMSE over *batches*."""
    absolute = 0.0
    squared = 0.0
    count = 0
    for batch in batches:
        predicted = default_training_fn(model, batch)["predicted_energy"]
        counts = batch.num_nodes_per_graph.reshape(-1, 1)
        residual = ((predicted - batch.energy) / counts).detach()
        absolute += float(residual.abs().sum())
        squared += float(residual.pow(2).sum())
        count += residual.numel()
    return absolute / count, (squared / count) ** 0.5


def _turn_forces(forces: torch.Tensor, angle: float) -> torch.Tensor:
    """Return every force turned through *angle*, its magnitude left alone.

    Each force turns in the plane it spans with the axis it is least aligned
    to, so the plane is always well defined however the field is oriented.
    """
    reference = torch.zeros_like(forces)
    reference.scatter_(1, forces.abs().argmin(dim=-1, keepdim=True), 1.0)
    perpendicular = torch.linalg.cross(torch.linalg.cross(forces, reference), forces)
    perpendicular = perpendicular / perpendicular.norm(dim=-1, keepdim=True)
    return math.cos(angle) * forces + math.sin(angle) * perpendicular * forces.norm(
        dim=-1, keepdim=True
    )


def _noisy_force_fn(model: Any, batch: Batch) -> dict[str, Any]:
    """Predict with *model* and add a fixed 5 meV/A of noise to every force."""
    predictions = default_training_fn(model, batch)
    generator = torch.Generator().manual_seed(17)
    noise = torch.randn(predictions["predicted_forces"].shape, generator=generator)
    predictions["predicted_forces"] = predictions["predicted_forces"] + 0.005 * noise
    return predictions


def _with_first_force(predictions: dict[str, Any], value: float) -> dict[str, Any]:
    """Return *predictions* with the first atom's predicted force set to *value*."""
    forces = predictions["predicted_forces"].clone()
    forces[0] = value
    predictions["predicted_forces"] = forces
    return predictions


def _nan_force_fn(model: Any, batch: Batch) -> dict[str, Any]:
    """Predict with *model* and hand back the first atom's force as ``nan``."""
    return _with_first_force(default_training_fn(model, batch), math.nan)


def _infinite_force_fn(model: Any, batch: Batch) -> dict[str, Any]:
    """Predict with *model* and hand back the first atom's force as ``inf``."""
    return _with_first_force(default_training_fn(model, batch), math.inf)


def _double_in_place(packed: torch.Tensor, manager: Any) -> None:  # noqa: ARG001
    """Stand in for an all-reduce over two ranks that measured the same shard."""
    packed.mul_(2.0)


class _SignallessScorer:
    """Scorer that declares a signal set with no forces in it."""

    signals = frozenset({"energy"})

    def label(self, batch: Batch) -> dict[str, Any]:  # noqa: ARG002
        """Return no labels; the evaluation never gets this far."""
        return {}


class _PlacementScorer:
    """Scorer recording which device each batch sat on when it was labeled."""

    def __init__(self, model: Any) -> None:
        self.inner = InProcessTeacherScorer(model, ["energy", "forces"])
        self.signals = self.inner.signals
        self.label_fields = self.inner.label_fields
        self.devices: list[torch.device] = []

    def label(self, batch: Batch) -> dict[str, Any]:
        """Record where the batch is resident and delegate to the wrapped scorer."""
        self.devices.append(batch.positions.device)
        return self.inner.label(batch)


class _CustomSignalScorer:
    """Scorer declaring a signal of its own, so its fields cannot be resolved."""

    signals = frozenset({"my_forces"})

    def label(self, batch: Batch) -> dict[str, Any]:
        """Return the forces under the framework's own field name anyway."""
        return {"teacher_forces": (torch.zeros_like(batch.positions), "node")}


class _MislabelingScorer:
    """Scorer declaring the built-in signals but publishing fields of its own."""

    signals = frozenset({"energy", "forces"})
    label_fields = ("teacher_E", "teacher_F")

    def label(self, batch: Batch) -> dict[str, Any]:
        """Return the labels under the names this scorer declared."""
        return {
            "teacher_E": (torch.zeros(batch.num_graphs, 1), "system"),
            "teacher_F": (torch.zeros_like(batch.positions), "node"),
        }


class _RecordingScorer:
    """Force-free scorer keeping every position it was asked to score."""

    signals = frozenset({"forces"})

    def __init__(self) -> None:
        self.positions: list[torch.Tensor] = []

    def label(self, batch: Batch) -> dict[str, Any]:
        """Record the probed geometry and return a vanishing force field."""
        self.positions.append(batch.positions.detach().clone())
        return {"teacher_forces": (torch.zeros_like(batch.positions),)}


class _GradRecordingFn:
    """Validation function recording whether autograd was live at each forward."""

    def __init__(self) -> None:
        self.grad_enabled: list[bool] = []

    def __call__(self, model: Any, batch: Batch) -> dict[str, Any]:
        """Record the ambient autograd state and predict as the default would."""
        self.grad_enabled.append(torch.is_grad_enabled())
        return default_training_fn(model, batch)


class _CurlScorer:
    """Analytic non-conservative field ``F = (-sin y, sin x, 0)``.

    Purely local and periodic in ``2 pi``, so a supercell of a commensurate
    cell carries exactly the same per-atom forces as the cell it replicates.
    """

    signals = frozenset({"forces"})

    def label(self, batch: Batch) -> dict[str, Any]:
        """Return the field's forces at the batch's current positions."""
        positions = batch.positions
        forces = torch.stack(
            [
                -torch.sin(positions[:, 1]),
                torch.sin(positions[:, 0]),
                torch.zeros_like(positions[:, 0]),
            ],
            dim=-1,
        )
        return {"teacher_forces": (forces,)}


class _ScaledCurlScorer:
    """Analytic non-conservative field ``F = scale * (-y, x, 0)``, per graph.

    Linear in the positions, so the midpoint rule integrates it exactly, and
    each graph carries its own force scale so a batch can mix them.
    """

    signals = frozenset({"forces"})

    def __init__(self, scales: tuple[float, ...]) -> None:
        self.scales = scales

    def label(self, batch: Batch) -> dict[str, Any]:
        """Return the field's forces, scaled by the graph each atom belongs to."""
        positions = batch.positions
        scales = torch.tensor(self.scales, dtype=positions.dtype)
        forces = torch.stack(
            [-positions[:, 1], positions[:, 0], torch.zeros_like(positions[:, 0])],
            dim=-1,
        )
        return {"teacher_forces": (scales[batch.batch_idx].unsqueeze(-1) * forces,)}


class _MoveRecordingBatch:
    """Batch stand-in recording the ``non_blocking`` flag a placement asked for."""

    def __init__(self) -> None:
        """Start with no placement recorded."""
        self.non_blocking: bool | None = None

    def to(
        self,
        device: torch.device,  # noqa: ARG002
        non_blocking: bool = False,
    ) -> _MoveRecordingBatch:
        """Record the flag and stand in for the moved copy."""
        self.non_blocking = non_blocking
        return self


class _LoopRecorder:
    """Validation-loop factory recording the data each evaluation handed it."""

    def __init__(self) -> None:
        """Start with no loop built."""
        self.validation_data: list[Any] = []

    def __call__(self, **kwargs: Any) -> ValidationLoop:
        """Record the validation data, then build the loop the evaluation wanted."""
        self.validation_data.append(kwargs["validation_data"])
        return ValidationLoop(**kwargs)


class TestEvaluateAccuracy:
    """Accuracy metrics of a student against reference and teacher targets."""

    def test_student_scored_against_itself_reports_zero_error(self) -> None:
        """A teacher that is the student gives zero error and perfect alignment."""
        student = _build_demo_model()
        metrics = evaluate_accuracy(
            student, _make_holdout(), targets="teacher", scorer=student
        )
        assert metrics.energy_mae == 0.0
        assert metrics.forces_rmse == 0.0
        assert metrics.force_cosine_mean == pytest.approx(1.0)
        assert metrics.force_cosine_aggregate == pytest.approx(1.0)

    def test_force_errors_match_hand_computed_global_sums(self) -> None:
        """Force MAE and RMSE equal the residual sums divided by component count."""
        student = _build_demo_model()
        holdout = _make_holdout()
        expected_mae, expected_rmse = _reference_force_error(student, holdout)
        metrics = evaluate_accuracy(student, holdout)
        assert metrics.forces_mae == pytest.approx(expected_mae, rel=1e-6)
        assert metrics.forces_rmse == pytest.approx(expected_rmse, rel=1e-6)

    def test_per_atom_energy_error_divides_each_graph_by_its_own_size(self) -> None:
        """Per-atom energy MAE and RMSE divide every graph by its own size."""
        student = _build_demo_model()
        holdout = _make_holdout(sizes=(2, 5))
        expected_mae, expected_rmse = _reference_energy_error(student, holdout)
        metrics = evaluate_accuracy(student, holdout)
        assert metrics.energy_per_atom_mae == pytest.approx(expected_mae, rel=1e-6)
        assert metrics.energy_per_atom_rmse == pytest.approx(expected_rmse, rel=1e-6)

    def test_a_uniformly_turned_reference_field_scores_the_cosine_of_that_angle(
        self,
    ) -> None:
        """Targets turned 60 degrees off the student score a cosine of one half."""
        student = _build_demo_model()
        holdout = _make_holdout()
        angle = math.radians(60.0)
        for batch in holdout:
            predicted = default_training_fn(student, batch)["predicted_forces"]
            batch.forces = _turn_forces(predicted.detach(), angle)
        metrics = evaluate_accuracy(student, holdout)
        assert metrics.force_cosine_mean == pytest.approx(math.cos(angle), rel=1e-6)
        assert metrics.force_cosine_aggregate == pytest.approx(
            math.cos(angle), rel=1e-6
        )

    def test_metrics_do_not_depend_on_how_the_holdout_is_batched(self) -> None:
        """The same systems split into one or two batches give the same metrics."""
        student = _build_demo_model()
        split = _make_holdout()
        joined = [
            Batch.from_data_list([*split[0].to_data_list(), *split[1].to_data_list()])
        ]
        assert evaluate_accuracy(student, split).to_dict() == pytest.approx(
            evaluate_accuracy(student, joined).to_dict(), rel=1e-6
        )

    def test_counts_cover_every_graph_and_atom(self) -> None:
        """Reported graph and atom counts are the totals over the whole holdout."""
        metrics = evaluate_accuracy(_build_demo_model(), _make_holdout(sizes=(2, 5)))
        assert metrics.num_graphs == 2
        assert metrics.num_atoms == 7

    def test_scaled_teacher_is_perfectly_aligned_but_wrong_in_magnitude(self) -> None:
        """Doubling the teacher's epsilon leaves cosine at one and MAE at |F|."""
        student = _build_lj_teacher()
        student.set_config("active_outputs", {"energy", "forces", "stress"})
        teacher = LennardJonesModelWrapper(epsilon=0.02, sigma=3.4, cutoff=5.0)
        batch = _make_lj_batch()
        predictions = default_training_fn(student, batch)
        metrics = evaluate_accuracy(
            student,
            [batch],
            targets="teacher",
            scorer=teacher,
            quantities=("energy", "forces", "stress"),
        )
        assert metrics.force_cosine_mean == pytest.approx(1.0)
        assert metrics.forces_mae == pytest.approx(
            float(predictions["predicted_forces"].abs().mean()), rel=1e-5
        )
        assert metrics.stress_mae == pytest.approx(
            float(predictions["predicted_stress"].abs().mean()), rel=1e-5
        )

    def test_atomic_energy_residuals_fill_in_when_both_sides_publish_them(self) -> None:
        """A student and teacher with energy heads report per-atom energy error."""
        student = _build_direct_force_teacher(seed=1)
        teacher = _build_direct_force_teacher(seed=2)
        metrics = evaluate_accuracy(
            student,
            _make_holdout(),
            targets="teacher",
            scorer=teacher,
            quantities=("energy", "forces", "atomic_energies"),
        )
        assert metrics.atomic_energy_mae is not None
        assert metrics.atomic_energy_rmse >= metrics.atomic_energy_mae

    def test_unknown_quantity_is_rejected(self) -> None:
        """A misspelled quantity raises before any forward pass."""
        with pytest.raises(ValueError, match="unsupported"):
            evaluate_accuracy(
                _build_demo_model(), _make_holdout(), quantities=("dipole",)
            )

    def test_evaluation_without_a_supervised_quantity_is_rejected(self) -> None:
        """Diagnostic-only quantities leave the pass with no loss to run."""
        with pytest.raises(ValueError, match="must be evaluated"):
            evaluate_accuracy(
                _build_demo_model(), _make_holdout(), quantities=("atomic_energies",)
            )

    def test_measuring_nothing_at_all_raises(self) -> None:
        """A quantity neither side carries leaves no metric to report."""
        with pytest.raises(ValueError, match="No accuracy metric"):
            evaluate_accuracy(
                _build_demo_model(),
                _make_holdout(),
                quantities=("atomic_energies",),
                loss_fn=EnergyMSELoss(),
                grad_mode="enabled",
            )

    def test_a_target_of_the_wrong_shape_is_rejected(self) -> None:
        """A mismatched target would broadcast into meaningless residuals."""
        with pytest.raises(ValueError, match="same shape"):
            evaluate_accuracy(
                _build_direct_force_teacher(),
                _make_holdout(),
                quantities=("atomic_energies",),
                target_keys={"atomic_energies": "energy"},
                loss_fn=EnergyMSELoss(),
                grad_mode="enabled",
            )

    def test_scorer_missing_a_requested_signal_is_rejected(self) -> None:
        """A scorer that cannot produce forces cannot back a force evaluation."""
        with pytest.raises(ValueError, match="missing"):
            evaluate_accuracy(
                _build_demo_model(),
                _make_holdout(),
                targets="teacher",
                scorer=_SignallessScorer(),
            )

    def test_a_perfect_student_scores_one_on_a_holdout_holding_relaxed_frames(
        self,
    ) -> None:
        """A student reproducing the teacher exactly is aligned everywhere."""
        student = _build_lj_teacher()
        metrics = evaluate_accuracy(
            student, _make_lattice_holdout(), targets="teacher", scorer=student
        )
        assert metrics.forces_mae == 0.0
        assert metrics.force_cosine_mean == pytest.approx(1.0)

    def test_a_relaxed_frame_does_not_move_the_magnitude_weighted_cosine(self) -> None:
        """A relaxed frame collapses the per-atom mean and leaves the aggregate."""
        teacher = _build_lj_teacher()
        metrics = evaluate_accuracy(
            _build_lj_teacher(),
            _make_lattice_holdout(),
            targets="teacher",
            scorer=teacher,
            validation_fn=_noisy_force_fn,
        )
        assert metrics.forces_mae < 0.01
        assert metrics.force_cosine_mean < 0.7
        assert metrics.force_cosine_aggregate > 0.9

    def test_forces_far_below_the_scale_of_a_clamp_are_scored_by_their_angle(
        self,
    ) -> None:
        """Two aligned 1e-8 eV/A force fields are aligned, not orthogonal."""
        student = _build_lj_teacher()
        metrics = evaluate_accuracy(
            student, _make_lattice_holdout()[1:], targets="teacher", scorer=student
        )
        assert metrics.forces_mae == 0.0
        assert metrics.force_cosine_mean == pytest.approx(1.0)

    def test_a_holdout_whose_forces_all_vanish_reports_no_cosine(self) -> None:
        """With no angle defined anywhere the mean is unmeasured rather than zero."""
        metrics = evaluate_accuracy(_build_lj_teacher(), _make_lattice_holdout())
        assert metrics.forces_mae > 0.0
        assert metrics.force_cosine_mean is None

    def test_ranks_pack_the_same_sums_whatever_their_shard_carried(self) -> None:
        """A shard missing a target still all-reduces an identically shaped tensor."""
        student = _build_demo_model()
        holdout = _make_holdout()
        packed = []
        for target_keys in ({}, {"forces": "absent_forces"}):
            with (
                patch.object(
                    accuracy_module, "is_distributed_initialized", return_value=True
                ),
                patch.object(accuracy_module, "all_reduce") as reduction,
            ):
                evaluate_accuracy(
                    student,
                    holdout,
                    target_keys=target_keys,
                    loss_fn=EnergyMSELoss(),
                    grad_mode="enabled",
                )
            packed.append(reduction.call_args.args[0])
        assert packed[0].shape == packed[1].shape

    def test_an_empty_shard_raises_before_the_metric_reduce(self) -> None:
        """A rank with nothing to score leaves the loop before the all-reduce."""
        with (
            patch.object(
                accuracy_module, "is_distributed_initialized", return_value=True
            ),
            patch.object(accuracy_module, "all_reduce") as reduction,
            pytest.raises(ValueError, match="no batches"),
        ):
            evaluate_accuracy(_build_demo_model(), [])
        reduction.assert_not_called()

    def test_a_float64_teacher_scores_a_float32_student_as_the_store_would(
        self,
    ) -> None:
        """Teacher labels reach the loss at the dtype the store would hold them at."""
        student = _build_direct_force_teacher(seed=2)
        teacher = _build_direct_force_teacher(seed=1).to(torch.float64)
        holdout = _make_holdout()
        scored = evaluate_accuracy(student, holdout, targets="teacher", scorer=teacher)
        explicit = evaluate_accuracy(
            student,
            holdout,
            targets="teacher",
            scorer=InProcessTeacherScorer(
                teacher, ["energy", "forces"], cast_to=_student_label_dtype(student)
            ),
        )
        assert scored.to_dict() == explicit.to_dict()

    @pytest.mark.parametrize(
        "dtype", [torch.bfloat16, torch.float64], ids=["bfloat16", "float64"]
    )
    def test_a_student_off_the_datasets_precision_is_still_measured(
        self, dtype: torch.dtype
    ) -> None:
        """A student in its own precision is scored against a float32 holdout."""
        student = _build_direct_force_teacher(seed=2).to(dtype)
        holdout = _make_holdout()
        expected_mae, _ = _reference_force_error(student, holdout)
        metrics = evaluate_accuracy(student, holdout)
        assert metrics.forces_mae == pytest.approx(expected_mae, rel=1e-12)

    def test_a_one_shot_holdout_is_rejected_even_behind_a_scorer(self) -> None:
        """Wrapping the holdout for a scorer does not hide it from the guard."""
        student = _build_demo_model()
        with pytest.raises(ValueError, match="re-iterable"):
            evaluate_accuracy(
                student,
                (batch for batch in _make_holdout()),
                targets="teacher",
                scorer=student,
            )

    def test_a_scorer_nothing_is_compared_against_is_rejected(self) -> None:
        """Teacher labels the reference targets would then ignore are an error."""
        student = _build_demo_model()
        with pytest.raises(ValueError, match="targets='teacher'"):
            evaluate_accuracy(student, _make_holdout(), scorer=student)

    def test_a_scorer_is_honored_when_target_keys_name_its_fields(self) -> None:
        """A teacher field named in target_keys honors the scorer."""
        student = _build_demo_model()
        metrics = evaluate_accuracy(
            student,
            _make_holdout(),
            scorer=student,
            target_keys={"forces": "teacher_forces"},
        )
        assert metrics.forces_mae == 0.0
        assert metrics.energy_mae > 0.0

    def test_explicit_target_keys_override_the_selected_family(self) -> None:
        """A target-key override points one quantity at any batch field."""
        student = _build_demo_model()
        holdout = _make_holdout()
        overridden = evaluate_accuracy(
            student,
            holdout,
            targets="teacher",
            scorer=student,
            target_keys={"energy": "energy"},
        )
        assert overridden.energy_mae > 0.0
        assert overridden.forces_mae == 0.0


class TestNonFiniteForceMetrics:
    """What a force that is not a number does to the alignment metrics."""

    @pytest.mark.parametrize(
        "validation_fn", [_nan_force_fn, _infinite_force_fn], ids=["nan", "inf"]
    )
    def test_one_nonfinite_atom_is_counted_and_carries_the_aggregate(
        self, validation_fn: Any
    ) -> None:
        """The set-wide alignment reads ``nan``; the per-atom mean drops the atom."""
        teacher = _build_lj_teacher()
        holdout = _make_lattice_holdout()[:1]
        metrics = evaluate_accuracy(
            _build_lj_teacher(),
            holdout,
            targets="teacher",
            scorer=teacher,
            validation_fn=validation_fn,
        )
        assert metrics.force_nonfinite_atoms == 1
        assert metrics.num_atoms == holdout[0].num_nodes
        assert math.isnan(metrics.force_cosine_aggregate)
        assert not math.isfinite(metrics.forces_mae)
        assert metrics.force_cosine_mean == pytest.approx(1.0)

    def test_a_holdout_whose_forces_all_vanish_counts_no_nonfinite_atom(self) -> None:
        """A vanishing field is unmeasured, not garbage: ``None`` and a zero count."""
        metrics = evaluate_accuracy(_build_lj_teacher(), _make_lattice_holdout())
        assert metrics.force_cosine_aggregate is None
        assert metrics.force_cosine_mean is None
        assert metrics.force_nonfinite_atoms == 0

    def test_the_nonfinite_count_rides_in_the_packed_all_reduce(self) -> None:
        """Ranks sum their dropped atoms like every other count of the pass."""
        teacher = _build_lj_teacher()
        with (
            patch.object(
                accuracy_module, "is_distributed_initialized", return_value=True
            ),
            patch.object(
                accuracy_module, "all_reduce", side_effect=_double_in_place
            ) as reduction,
        ):
            metrics = evaluate_accuracy(
                _build_lj_teacher(),
                _make_lattice_holdout()[:1],
                targets="teacher",
                scorer=teacher,
                validation_fn=_nan_force_fn,
            )
        reduction.assert_called_once()
        assert metrics.force_nonfinite_atoms == 2


class TestAccuracyGradPolicy:
    """Which students an evaluation runs with autograd live around the forward."""

    def test_an_energy_only_evaluation_of_a_conservative_student_runs(self) -> None:
        """A student differentiating its own energy is scored on energies alone."""
        metrics = evaluate_accuracy(
            _build_demo_model(), _make_holdout(), quantities=("energy",)
        )
        assert metrics.energy_mae is not None
        assert metrics.forces_mae is None

    def test_auto_and_enabled_agree_exactly_for_a_conservative_student(self) -> None:
        """Inferring the policy scores identically to demanding it outright."""
        student = _build_demo_model()
        holdout = _make_holdout()
        inferred = evaluate_accuracy(student, holdout, quantities=("energy",))
        demanded = evaluate_accuracy(
            student, holdout, quantities=("energy",), grad_mode="enabled"
        )
        assert inferred == demanded

    def test_a_conservative_student_is_scored_with_gradients_live(self) -> None:
        """Autograd is enabled around every forward, not only derivative losses."""
        recorder = _GradRecordingFn()
        evaluate_accuracy(
            _build_demo_model(),
            _make_holdout(),
            quantities=("energy",),
            validation_fn=recorder,
        )
        assert recorder.grad_enabled == [True, True]

    def test_a_direct_force_student_keeps_the_gradient_free_fast_path(self) -> None:
        """A student declaring no autograd output is still scored under no_grad."""
        recorder = _GradRecordingFn()
        metrics = evaluate_accuracy(
            _build_direct_force_teacher(),
            _make_holdout(),
            quantities=("energy",),
            validation_fn=recorder,
        )
        assert recorder.grad_enabled == [False, False]
        assert metrics.energy_mae is not None

    def test_narrowing_the_active_outputs_returns_a_student_to_the_fast_path(
        self,
    ) -> None:
        """A conservative student told not to produce forces stops needing grad."""
        student = _build_demo_model()
        student.set_config("active_outputs", {"energy"})
        recorder = _GradRecordingFn()
        evaluate_accuracy(
            student,
            _make_holdout(),
            quantities=("energy",),
            validation_fn=recorder,
        )
        assert recorder.grad_enabled == [False, False]

    def test_disabling_gradients_for_a_conservative_student_is_rejected(self) -> None:
        """The refusal names the student and the outputs it differentiates."""
        with pytest.raises(ValueError, match="differentiating inside its own forward"):
            evaluate_accuracy(
                _build_demo_model(),
                _make_holdout(),
                quantities=("energy",),
                grad_mode="disabled",
            )

    def test_disabling_gradients_for_a_direct_force_student_is_allowed(self) -> None:
        """Only a student that needs autograd is refused the disabled policy."""
        metrics = evaluate_accuracy(
            _build_direct_force_teacher(),
            _make_holdout(),
            quantities=("energy",),
            grad_mode="disabled",
        )
        assert metrics.energy_mae is not None


class TestDevicePlacement:
    """Which device an evaluation runs, and labels, on."""

    def test_an_explicit_device_outranks_the_model_parameters(self) -> None:
        """A requested device is honored rather than read back off the model."""
        resolved = accuracy_module._resolve_device(_build_demo_model(), "meta")
        assert resolved == torch.device("meta")

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_an_explicit_device_outranks_a_parameterless_model(self) -> None:
        """A model exposing no parameters would otherwise fall back to the host."""
        teacher = _build_lj_teacher()
        assert not list(teacher.parameters())
        assert accuracy_module._resolve_device(teacher, "cuda") == torch.device("cuda")

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_a_gpu_teacher_labels_the_batch_after_it_reaches_the_gpu(self) -> None:
        """A host-resident holdout is moved first, so the teacher runs on the GPU."""
        student = _build_demo_model().to("cuda")
        scorer = _PlacementScorer(student)
        metrics = evaluate_accuracy(
            student, _make_holdout(), targets="teacher", scorer=scorer, device="cuda"
        )
        assert scorer.devices
        assert all(device.type == "cuda" for device in scorer.devices)
        assert metrics.forces_rmse == 0.0
        assert metrics.energy_mae == 0.0

    @pytest.mark.parametrize(
        ("destination", "asynchronous"),
        [("cpu", False), ("cuda", True), ("cuda:1", True)],
        ids=["host", "device", "second-device"],
    )
    def test_only_a_device_destination_takes_an_asynchronous_copy(
        self, destination: str, asynchronous: bool
    ) -> None:
        """A placement overlaps a copy into device memory and blocks on one into host."""
        batch = _MoveRecordingBatch()

        list(accuracy_module._PlacedBatches([batch], torch.device(destination)))

        assert batch.non_blocking is asynchronous

    def test_an_unscored_evaluation_places_its_batches_too(self) -> None:
        """Reference targets get the same placement a scorer's targets always got."""
        recorder = _LoopRecorder()
        with patch.object(accuracy_module, "ValidationLoop", recorder):
            evaluate_accuracy(_build_demo_model(), _make_holdout())

        (placed,) = recorder.validation_data
        assert isinstance(placed, accuracy_module._PlacedBatches)
        assert all(batch.positions.device == torch.device("cpu") for batch in placed)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_a_device_resident_holdout_lands_before_its_metrics_are_read(self) -> None:
        """A GPU holdout evaluated on the host reports what a host holdout does.

        Without the placement the loop's own asynchronous move returns before
        the transfer lands and its index tensors are read out of a half-written
        buffer. The pinned-host allocator hides that once a correctly valued
        block of the same shape has been cached, so nothing here copies onto
        the host before the device-resident pass does.
        """
        holdout = _make_ragged_holdout()
        student = _build_demo_model()
        expected = evaluate_accuracy(student, holdout).to_dict()
        device = torch.device("cuda")
        resident = [batch.to(device) for batch in holdout]

        for _ in range(_PLACEMENT_TRIALS):
            pressure = torch.randn(4096, 4096, device=device)
            pressure @ pressure
            assert evaluate_accuracy(student, resident).to_dict() == expected


class TestScorerContract:
    """What a supplied scorer has to publish for any evaluation to read it."""

    def test_a_scorer_publishing_other_fields_is_refused_everywhere(self) -> None:
        """Every entry point checks the fields a scorer declares, not its signals."""
        scorer = _MislabelingScorer()
        batch = _build_lattice_batch()
        with pytest.raises(ValueError, match="teacher_forces"):
            evaluate_accuracy(
                _build_demo_model(), _make_holdout(), targets="teacher", scorer=scorer
            )
        with pytest.raises(ValueError, match="teacher_forces"):
            nonconservative_residual(scorer, batch)
        with pytest.raises(ValueError, match="teacher_energy"):
            extensivity_error(scorer, batch)

    def test_a_scorer_whose_fields_cannot_be_known_is_let_through(self) -> None:
        """A custom signal name leaves the fields unknowable, so nothing is refused."""
        residual = nonconservative_residual(
            _CustomSignalScorer(), _build_batch(n_atoms_each=4, seed=3), num_loops=1
        )
        assert residual.force_floor == 0.0


class TestNonConservativeResidual:
    """The residual floor separating conservative from direct-force teachers."""

    def test_conservative_teacher_reports_a_negligible_floor(self) -> None:
        """An autograd-force teacher's loops close to orders below a direct one's."""
        residual = nonconservative_residual(
            _build_demo_model(), _build_batch(n_atoms_each=4, seed=3), num_loops=2
        )
        assert residual.relative_floor < 1e-5

    def test_direct_force_teacher_reports_a_nonzero_floor(self) -> None:
        """A force head that is not an energy gradient leaves work in every loop."""
        batch = _build_batch(n_atoms_each=4, seed=3)
        conservative = nonconservative_residual(_build_demo_model(), batch, num_loops=2)
        direct = nonconservative_residual(
            _build_direct_force_teacher(), batch, num_loops=2
        )
        assert direct.relative_floor > 1e-5
        assert direct.force_floor > 100.0 * conservative.force_floor

    def test_floor_scales_linearly_with_the_probe_amplitude(self) -> None:
        """Doubling the loop side doubles the reported force floor."""
        teacher = _build_direct_force_teacher()
        batch = _build_batch(n_atoms_each=4, seed=3)
        small = nonconservative_residual(
            teacher,
            batch,
            num_loops=3,
            amplitude=0.02,
            generator=torch.Generator().manual_seed(0),
        )
        large = nonconservative_residual(
            teacher,
            batch,
            num_loops=3,
            amplitude=0.04,
            generator=torch.Generator().manual_seed(0),
        )
        assert large.force_floor / small.force_floor == pytest.approx(2.0, rel=0.05)

    def test_every_atom_moves_by_the_amplitude_whatever_the_system_size(self) -> None:
        """The probe displaces atoms by *amplitude*, not by amplitude over sqrt(N)."""
        small = _probe_displacement(_build_batch(n_systems=1, n_atoms_each=4, seed=3))
        large = _probe_displacement(_build_batch(n_systems=1, n_atoms_each=64, seed=3))
        assert float(small.mean()) == pytest.approx(float(large.mean()), rel=1e-5)
        assert math.sqrt(float(small.mean())) == pytest.approx(0.05, rel=0.15)

    def test_every_graph_of_a_mixed_batch_is_probed_at_the_same_scale(self) -> None:
        """A 3-atom and a 48-atom graph in one batch move by the same amount."""
        batch = Batch.from_data_list(
            [
                _build_atomic_data(n_atoms=3, seed=1),
                _build_atomic_data(n_atoms=48, seed=2),
            ]
        )
        squared = _probe_displacement(batch)
        first = float(squared[:, batch.batch_idx == 0].mean())
        second = float(squared[:, batch.batch_idx == 1].mean())
        assert first == pytest.approx(second, rel=1e-5)

    def test_a_supercell_follows_the_documented_size_law(self) -> None:
        """Doubling the cell of an identical field divides the floor by sqrt(2)."""
        scorer = _CurlScorer()
        cell = nonconservative_residual(
            scorer,
            _make_curl_lattice(1),
            num_loops=60,
            segments=6,
            generator=torch.Generator().manual_seed(0),
        )
        supercell = nonconservative_residual(
            scorer,
            _make_curl_lattice(2),
            num_loops=60,
            segments=6,
            generator=torch.Generator().manual_seed(0),
        )
        assert supercell.force_rms == pytest.approx(cell.force_rms, rel=1e-9)
        assert supercell.force_floor * math.sqrt(2.0) == pytest.approx(
            cell.force_floor, rel=0.25
        )

    def test_the_floor_does_not_depend_on_where_in_space_the_frame_sits(self) -> None:
        """Loops laid out around the centroid read a translated frame the same."""
        here = _build_lattice_batch(jitter=0.2)
        far = _build_lattice_batch(jitter=0.2)
        far.positions = far.positions + 200.0
        origin = nonconservative_residual(
            _build_lj_teacher(),
            here,
            num_loops=2,
            amplitude=0.02,
            generator=torch.Generator().manual_seed(0),
        )
        translated = nonconservative_residual(
            _build_lj_teacher(),
            far,
            num_loops=2,
            amplitude=0.02,
            generator=torch.Generator().manual_seed(0),
        )
        assert translated.force_rms == pytest.approx(origin.force_rms, rel=1e-4)
        assert translated.force_floor == pytest.approx(origin.force_floor, rel=0.5)

    def test_probing_restores_the_positions_it_displaced(self) -> None:
        """The probed batch is left exactly as it arrived."""
        batch = _build_batch(n_atoms_each=4, seed=3)
        original = batch.positions.clone()
        nonconservative_residual(_build_direct_force_teacher(), batch, num_loops=2)
        torch.testing.assert_close(batch.positions, original)

    def test_a_periodic_teacher_needing_neighbors_is_probed_too(self) -> None:
        """A neighbor-list teacher is probed through its scorer's own rebuilds."""
        batch = _build_lattice_batch(jitter=0.2)
        residual = nonconservative_residual(
            _build_lj_teacher(), batch, num_loops=2, amplitude=0.02
        )
        assert residual.relative_floor < 1e-4
        assert "neighbor_matrix" not in batch

    @pytest.mark.parametrize(
        ("amplitude", "num_loops", "segments"),
        [(0.0, 2, 4), (0.05, 0, 4), (0.05, 2, 0)],
        ids=["zero-amplitude", "no-loops", "no-segments"],
    )
    def test_degenerate_probe_geometry_is_rejected(
        self, amplitude: float, num_loops: int, segments: int
    ) -> None:
        """A probe with no extent, no loops, or no samples raises."""
        with pytest.raises(ValueError, match="must all be positive"):
            nonconservative_residual(
                _build_demo_model(),
                _build_batch(),
                amplitude=amplitude,
                num_loops=num_loops,
                segments=segments,
            )

    def test_scorer_without_forces_is_rejected(self) -> None:
        """The probe integrates forces, so a scorer without them cannot serve it."""
        with pytest.raises(ValueError, match="missing"):
            nonconservative_residual(_SignallessScorer(), _build_batch())


class TestNonConservativeFloorConditioning:
    """How the floor and its relative form behave on a batch of unlike graphs."""

    def test_a_graph_far_from_its_batch_reads_the_same_floor(self) -> None:
        """Per-graph centering keeps a float32 batch's floor where the frame sits."""
        together = nonconservative_residual(
            _build_lj_teacher(),
            _make_translated_pair(0.0),
            num_loops=3,
            amplitude=0.02,
            generator=torch.Generator().manual_seed(0),
        )
        apart = nonconservative_residual(
            _build_lj_teacher(),
            _make_translated_pair(1e4),
            num_loops=3,
            amplitude=0.02,
            generator=torch.Generator().manual_seed(0),
        )
        assert apart.force_floor == pytest.approx(together.force_floor, rel=0.1)

    def test_every_graph_is_probed_around_its_own_centroid(self) -> None:
        """A pair 1e4 A apart is displaced by the amplitude, not by the separation."""
        batch = _make_translated_pair(1e4)
        squared = _probe_displacement(batch, amplitude=0.02)
        for graph in (0, 1):
            moved = math.sqrt(float(squared[:, batch.batch_idx == graph].mean()))
            assert moved == pytest.approx(0.02, rel=0.15)

    def test_a_single_graph_float64_floor_is_what_batch_centering_reported(
        self,
    ) -> None:
        """One graph is its own batch, so per-graph centering did not move it."""
        residual = nonconservative_residual(
            _CurlScorer(),
            _make_curl_lattice(1),
            num_loops=3,
            amplitude=0.05,
            segments=6,
            generator=torch.Generator().manual_seed(0),
        )
        assert residual.force_floor == pytest.approx(_BATCH_CENTERED_FLOOR, rel=1e-12)

    def test_relative_floor_stays_between_the_ratios_it_averages(self) -> None:
        """Dividing each probe by its own graph's force scale keeps it in range."""
        scorer = _ScaledCurlScorer((1.0, 100.0))
        batch = Batch.from_data_list([_make_grid_data(4), _make_grid_data(2)])
        residual = nonconservative_residual(
            scorer,
            batch,
            num_loops=1,
            amplitude=0.05,
            segments=4,
            generator=torch.Generator().manual_seed(5),
        )
        ratios = _per_graph_relative_floor(scorer, batch, amplitude=0.05, seed=5)
        assert residual.relative_floor == pytest.approx(float(ratios.mean()), rel=1e-9)
        assert residual.relative_floor_max == pytest.approx(
            float(ratios.max()), rel=1e-9
        )
        assert float(ratios.min()) < residual.relative_floor < float(ratios.max())
        assert residual.force_floor / residual.force_rms > float(ratios.max())

    def test_a_single_graph_relative_floor_is_the_plain_quotient(self) -> None:
        """With one graph to weight there is nothing for the two forms to differ on."""
        residual = nonconservative_residual(
            _ScaledCurlScorer((1.0,)),
            Batch.from_data_list([_make_grid_data(4)]),
            num_loops=3,
            amplitude=0.05,
            generator=torch.Generator().manual_seed(5),
        )
        assert residual.relative_floor == pytest.approx(
            residual.force_floor / residual.force_rms, rel=1e-12
        )
