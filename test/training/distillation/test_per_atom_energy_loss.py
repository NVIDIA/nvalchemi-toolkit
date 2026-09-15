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
"""Tests for :mod:`nvalchemi.training.distillation.losses.per_atom_energy`.

Also covers the built-in loss terms pointed at ``teacher_*`` targets, which is
how energy, force, and stress distillation objectives are expressed.
"""

from __future__ import annotations

import json

import pytest
import torch

from nvalchemi.data import Batch
from nvalchemi.models.lj import LennardJonesModelWrapper
from nvalchemi.training._spec import create_model_spec_from_json
from nvalchemi.training.distillation import (
    InProcessTeacherScorer,
    PerAtomEnergyMatchingLoss,
)
from nvalchemi.training.distillation._labels import _attach_teacher_labels
from nvalchemi.training.losses.composition import (
    BaseLossFunction,
    ComposedLossFunction,
    as_composed_loss,
    compute_supervised_loss,
    loss_component_to_spec,
)
from nvalchemi.training.losses.schedules import LinearWeight
from nvalchemi.training.losses.terms import (
    EnergyMSELoss,
    ForceMSELoss,
    StressMSELoss,
)
from test.training.distillation.conftest import _DirectForceTeacher

_UNEVEN_BATCH_IDX = torch.tensor([0, 0, 0, 1])
"""Graph assignment of four atoms split three-to-one across two graphs."""

_LARGE_GRAPH_ATOMS = 600
"""Atom count of one graph, past the 256 a bfloat16 sum saturates at."""

_OVERFLOW_GRAPH_ATOMS = 2048
"""Atom count whose squared residuals overflow a float16 sum."""

_PREDICTION_KEYS = {
    "teacher_energy": "predicted_energy",
    "teacher_forces": "predicted_forces",
    "teacher_stress": "predicted_stress",
    "teacher_node_energies": "predicted_atomic_energies",
}
"""Student prediction key each teacher field is compared against."""


def _make_labeled_batch(batch: Batch, teacher: object, signals: list[str]) -> Batch:
    """Return *batch* carrying the teacher fields for *signals*."""
    scorer = InProcessTeacherScorer(teacher, signals)
    _attach_teacher_labels(batch, scorer.label(batch))
    return batch


def _make_teacher_predictions(
    batch: Batch, teacher: object, signals: list[str]
) -> dict:
    """Return predictions that reproduce the teacher's own labels exactly."""
    scorer = InProcessTeacherScorer(teacher, signals)
    return {
        _PREDICTION_KEYS[field]: values
        for field, (values, _) in scorer.label(batch).items()
    }


class TestPerAtomEnergyMatchingLossValues:
    """Scalar values the per-atom energy matching loss produces."""

    def test_graph_balanced_value_matches_a_hand_computation(self) -> None:
        """Each graph's mean squared residual is averaged over graphs."""
        loss_fn = PerAtomEnergyMatchingLoss()
        pred = torch.tensor([1.0, 2.0, 3.0, 4.0])
        loss = loss_fn(pred, torch.zeros(4), batch_idx=_UNEVEN_BATCH_IDX, num_graphs=2)
        assert loss.item() == pytest.approx(((1 + 4 + 9) / 3 + 16) / 2)

    def test_global_mean_weights_every_atom_equally(self) -> None:
        """Without graph balancing the loss is one mean over all atoms."""
        loss_fn = PerAtomEnergyMatchingLoss(normalize_by_atom_count=False)
        pred = torch.tensor([1.0, 2.0, 3.0, 4.0])
        loss = loss_fn(pred, torch.zeros(4), batch_idx=_UNEVEN_BATCH_IDX, num_graphs=2)
        assert loss.item() == pytest.approx((1 + 4 + 9 + 16) / 4)

    def test_graph_balancing_ignores_atom_count_imbalance(self) -> None:
        """Two graphs with equal per-atom error give that error, whatever their size."""
        loss_fn = PerAtomEnergyMatchingLoss()
        pred = torch.full((4,), 2.0)
        loss = loss_fn(pred, torch.zeros(4), batch_idx=_UNEVEN_BATCH_IDX, num_graphs=2)
        assert loss.item() == pytest.approx(4.0)

    def test_per_sample_loss_holds_one_value_per_graph(self) -> None:
        """The graph-balanced reduction publishes a detached ``(B,)`` diagnostic."""
        loss_fn = PerAtomEnergyMatchingLoss()
        pred = torch.tensor([1.0, 2.0, 3.0, 4.0])
        loss_fn(pred, torch.zeros(4), batch_idx=_UNEVEN_BATCH_IDX, num_graphs=2)
        torch.testing.assert_close(
            loss_fn.per_sample_loss, torch.tensor([(1 + 4 + 9) / 3, 16.0])
        )

    def test_zero_residual_gives_zero_loss(self) -> None:
        """A student matching the teacher exactly scores zero."""
        loss_fn = PerAtomEnergyMatchingLoss()
        target = torch.randn(4)
        loss = loss_fn(
            target.clone(), target, batch_idx=_UNEVEN_BATCH_IDX, num_graphs=2
        )
        assert loss.item() == pytest.approx(0.0)

    def test_gradients_flow_to_the_prediction(self) -> None:
        """The loss is differentiable with respect to the student's energies."""
        loss_fn = PerAtomEnergyMatchingLoss()
        pred = torch.tensor([1.0, 2.0, 3.0, 4.0], requires_grad=True)
        loss_fn(
            pred, torch.zeros(4), batch_idx=_UNEVEN_BATCH_IDX, num_graphs=2
        ).backward()
        assert pred.grad is not None
        assert torch.count_nonzero(pred.grad) == 4


class TestPerAtomEnergyMatchingLossMasking:
    """Handling of non-finite teacher energies."""

    def test_nonfinite_targets_are_excluded_from_the_loss(self) -> None:
        """A ``NaN`` target drops its atom from both numerator and denominator."""
        loss_fn = PerAtomEnergyMatchingLoss()
        pred = torch.tensor([1.0, 2.0, 3.0, 4.0])
        target = torch.tensor([0.0, 0.0, float("nan"), 0.0])
        loss = loss_fn(pred, target, batch_idx=_UNEVEN_BATCH_IDX, num_graphs=2)
        assert loss.item() == pytest.approx(((1 + 4) / 2 + 16) / 2)

    def test_fully_nonfinite_graph_contributes_zero(self) -> None:
        """A graph with no valid atom scores zero instead of dividing by zero."""
        loss_fn = PerAtomEnergyMatchingLoss()
        pred = torch.tensor([1.0, 2.0, 3.0, 4.0])
        target = torch.tensor([float("nan"), float("nan"), 0.0, 0.0])
        loss = loss_fn(pred, target, batch_idx=torch.tensor([0, 0, 1, 1]), num_graphs=2)
        assert loss.item() == pytest.approx((0.0 + (9 + 16) / 2) / 2)

    def test_global_mean_divides_by_the_valid_atom_count(self) -> None:
        """The global mean drops a masked atom from numerator and denominator alike."""
        loss_fn = PerAtomEnergyMatchingLoss(normalize_by_atom_count=False)
        pred = torch.tensor([1.0, 2.0, 3.0, 4.0])
        target = torch.tensor([0.0, 0.0, float("nan"), 0.0])
        loss = loss_fn(pred, target, batch_idx=_UNEVEN_BATCH_IDX, num_graphs=2)
        assert loss.item() == pytest.approx((1 + 4 + 16) / 3)

    def test_fully_masked_global_mean_contributes_zero(self) -> None:
        """With no valid atom the global mean clamps its divisor to one."""
        loss_fn = PerAtomEnergyMatchingLoss(normalize_by_atom_count=False)
        loss = loss_fn(torch.ones(4), torch.full((4,), float("nan")))
        assert loss.item() == pytest.approx(0.0)

    def test_disabled_masking_propagates_nonfinite_targets(self) -> None:
        """``ignore_nonfinite=False`` lets a ``NaN`` target poison the loss."""
        loss_fn = PerAtomEnergyMatchingLoss(ignore_nonfinite=False)
        target = torch.tensor([0.0, 0.0, float("nan"), 0.0])
        loss = loss_fn(torch.ones(4), target, batch_idx=_UNEVEN_BATCH_IDX, num_graphs=2)
        assert torch.isnan(loss)


class TestPerAtomEnergyMatchingLossPrecision:
    """Reduced-precision residuals and the float32 accumulation that carries them."""

    def test_bfloat16_residual_is_reduced_in_float32(self) -> None:
        """A graph past the bfloat16 saturation point still reduces to its true mean."""
        loss_fn = PerAtomEnergyMatchingLoss()
        pred = torch.zeros(_LARGE_GRAPH_ATOMS, dtype=torch.bfloat16)
        pred[: _LARGE_GRAPH_ATOMS // 2] = 1.0
        loss = loss_fn(
            pred,
            torch.zeros(_LARGE_GRAPH_ATOMS, dtype=torch.bfloat16),
            batch_idx=torch.zeros(_LARGE_GRAPH_ATOMS, dtype=torch.long),
            num_graphs=1,
        )
        assert loss.dtype == torch.float32
        assert loss_fn.per_sample_loss.dtype == torch.float32
        assert loss.item() == pytest.approx(0.5)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_bfloat16_graph_balanced_reduction_is_exact_on_cuda(self) -> None:
        """The device scatter saturates at 256 in bfloat16, which float32 avoids."""
        loss_fn = PerAtomEnergyMatchingLoss()
        pred = torch.zeros(_LARGE_GRAPH_ATOMS, dtype=torch.bfloat16, device="cuda")
        pred[: _LARGE_GRAPH_ATOMS // 2] = 1.0
        loss = loss_fn(
            pred,
            torch.zeros(_LARGE_GRAPH_ATOMS, dtype=torch.bfloat16, device="cuda"),
            batch_idx=torch.zeros(_LARGE_GRAPH_ATOMS, dtype=torch.long, device="cuda"),
            num_graphs=1,
        )
        assert loss.item() == pytest.approx(0.5)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_float16_reduction_stays_finite_on_cuda(self) -> None:
        """Large float16 residuals overflow their own sum but not a float32 one."""
        loss_fn = PerAtomEnergyMatchingLoss()
        loss = loss_fn(
            torch.full(
                (_OVERFLOW_GRAPH_ATOMS,), 10.0, dtype=torch.float16, device="cuda"
            ),
            torch.zeros(_OVERFLOW_GRAPH_ATOMS, dtype=torch.float16, device="cuda"),
            batch_idx=torch.zeros(
                _OVERFLOW_GRAPH_ATOMS, dtype=torch.long, device="cuda"
            ),
            num_graphs=1,
        )
        assert torch.isfinite(loss)
        assert loss.item() == pytest.approx(100.0)

    def test_reduced_precision_gradients_stay_in_the_prediction_dtype(self) -> None:
        """Accumulating in float32 leaves the gradient in the student's own dtype."""
        loss_fn = PerAtomEnergyMatchingLoss()
        pred = torch.zeros(_LARGE_GRAPH_ATOMS, dtype=torch.bfloat16)
        pred[: _LARGE_GRAPH_ATOMS // 2] = 1.0
        pred = pred.requires_grad_(True)
        loss_fn(
            pred,
            torch.zeros(_LARGE_GRAPH_ATOMS, dtype=torch.bfloat16),
            batch_idx=torch.zeros(_LARGE_GRAPH_ATOMS, dtype=torch.long),
            num_graphs=1,
        ).backward()
        assert pred.grad.dtype == torch.bfloat16
        assert torch.isfinite(pred.grad).all()
        assert pred.grad[0].item() == pytest.approx(2 / _LARGE_GRAPH_ATOMS, rel=1e-2)


class TestPerAtomEnergyMatchingLossContract:
    """Shape, metadata, and serialization contract of the loss term."""

    def test_shape_mismatch_is_rejected(self) -> None:
        """Column-vector predictions are refused rather than broadcast."""
        loss_fn = PerAtomEnergyMatchingLoss()
        with pytest.raises(ValueError, match="shape must match exactly"):
            loss_fn(
                torch.zeros(4, 1),
                torch.zeros(4),
                batch_idx=_UNEVEN_BATCH_IDX,
                num_graphs=2,
            )

    def test_missing_graph_metadata_is_rejected(self) -> None:
        """The graph-balanced reduction names the metadata it needs."""
        loss_fn = PerAtomEnergyMatchingLoss()
        with pytest.raises(ValueError, match="batch_idx"):
            loss_fn(torch.zeros(4), torch.zeros(4))

    def test_global_mean_needs_no_graph_metadata(self) -> None:
        """The global mean reduces without ``batch_idx`` or ``num_graphs``."""
        loss_fn = PerAtomEnergyMatchingLoss(normalize_by_atom_count=False)
        assert loss_fn(torch.ones(4), torch.zeros(4)).item() == pytest.approx(1.0)

    def test_loss_declares_it_needs_no_evaluation_gradients(self) -> None:
        """Per-atom energies are a direct output, so validation can run no-grad."""
        assert PerAtomEnergyMatchingLoss().requires_eval_grad is False

    def test_default_keys_match_the_teacher_and_student_fields(self) -> None:
        """Defaults line up the teacher's node energies with the student's head."""
        loss_fn = PerAtomEnergyMatchingLoss()
        assert loss_fn.target_key == "teacher_node_energies"
        assert loss_fn.prediction_key == "predicted_atomic_energies"

    def test_spec_round_trip_rebuilds_an_equivalent_loss(self) -> None:
        """A JSON round-tripped spec rebuilds the loss with its configuration."""
        spec = loss_component_to_spec(
            PerAtomEnergyMatchingLoss(
                target_key="teacher_node_energies",
                normalize_by_atom_count=False,
                ignore_nonfinite=False,
            )
        )
        rebuilt = create_model_spec_from_json(
            json.loads(spec.model_dump_json())
        ).build()
        assert isinstance(rebuilt, PerAtomEnergyMatchingLoss)
        assert rebuilt.normalize_by_atom_count is False
        assert rebuilt.ignore_nonfinite is False
        assert rebuilt.target_key == "teacher_node_energies"


class TestPerAtomEnergyMatchingLossComposition:
    """Behavior of the loss inside a composed distillation objective."""

    def test_composition_sums_weighted_teacher_terms(
        self, small_batch: Batch, direct_force_teacher: _DirectForceTeacher
    ) -> None:
        """A three-term teacher objective reports every component it composed."""
        signals = ["energy", "forces", "node_energies"]
        batch = _make_labeled_batch(small_batch, direct_force_teacher, signals)
        loss_fn = ComposedLossFunction(
            [
                EnergyMSELoss(target_key="teacher_energy"),
                ForceMSELoss(target_key="teacher_forces"),
                PerAtomEnergyMatchingLoss(),
            ],
            weights=[1.0, 1.0, 2.0],
        )
        predictions = {
            "predicted_energy": torch.zeros_like(batch.teacher_energy),
            "predicted_forces": torch.zeros_like(batch.teacher_forces),
            "predicted_atomic_energies": torch.zeros_like(batch.teacher_node_energies),
        }
        out = compute_supervised_loss(loss_fn, predictions, batch, step=0, epoch=0)
        assert set(out["per_component_unweighted"]) == {
            "EnergyMSELoss",
            "ForceMSELoss",
            "PerAtomEnergyMatchingLoss",
        }
        assert out["per_component_weight"][
            "PerAtomEnergyMatchingLoss"
        ] == pytest.approx(0.5)
        assert out["total_loss"].item() > 0.0

    def test_weight_schedule_ramps_the_teacher_term(
        self, small_batch: Batch, direct_force_teacher: _DirectForceTeacher
    ) -> None:
        """A scheduled weight on the per-atom term grows with the step count."""
        batch = _make_labeled_batch(
            small_batch, direct_force_teacher, ["energy", "node_energies"]
        )
        loss_fn = ComposedLossFunction(
            [EnergyMSELoss(target_key="teacher_energy"), PerAtomEnergyMatchingLoss()],
            weights=[1.0, LinearWeight(start=0.0, end=1.0, num_steps=10)],
        )
        predictions = {
            "predicted_energy": torch.zeros_like(batch.teacher_energy),
            "predicted_atomic_energies": torch.zeros_like(batch.teacher_node_energies),
        }
        early = compute_supervised_loss(loss_fn, predictions, batch, step=0, epoch=0)
        late = compute_supervised_loss(loss_fn, predictions, batch, step=10, epoch=0)
        assert (
            late["per_component_weight"]["PerAtomEnergyMatchingLoss"]
            > early["per_component_weight"]["PerAtomEnergyMatchingLoss"]
        )


class TestTeacherTargetLosses:
    """Built-in loss terms reading teacher-labeled batch fields."""

    @pytest.mark.parametrize(
        ("loss_fn", "signals"),
        [
            pytest.param(
                EnergyMSELoss(target_key="teacher_energy"), ["energy"], id="energy"
            ),
            pytest.param(
                ForceMSELoss(target_key="teacher_forces", normalize_by_atom_count=True),
                ["forces"],
                id="forces",
            ),
            pytest.param(
                PerAtomEnergyMatchingLoss(), ["node_energies"], id="node_energies"
            ),
        ],
    )
    def test_teacher_predictions_score_zero_against_teacher_labels(
        self,
        loss_fn: BaseLossFunction,
        signals: list[str],
        small_batch: Batch,
        direct_force_teacher: _DirectForceTeacher,
    ) -> None:
        """Predictions equal to the labels give zero, so target routing is exact."""
        predictions = _make_teacher_predictions(
            small_batch, direct_force_teacher, signals
        )
        batch = _make_labeled_batch(small_batch, direct_force_teacher, signals)
        out = compute_supervised_loss(
            as_composed_loss(loss_fn), predictions, batch, step=0, epoch=0
        )
        assert out["total_loss"].item() == pytest.approx(0.0)

    def test_teacher_stress_drives_a_stress_loss(
        self, periodic_batch: Batch, lj_teacher: LennardJonesModelWrapper
    ) -> None:
        """A stress teacher signal feeds ``StressMSELoss`` through ``teacher_stress``."""
        signals = ["stress"]
        predictions = _make_teacher_predictions(periodic_batch, lj_teacher, signals)
        batch = _make_labeled_batch(periodic_batch, lj_teacher, signals)
        loss_fn = as_composed_loss(StressMSELoss(target_key="teacher_stress"))
        matched = compute_supervised_loss(loss_fn, predictions, batch, step=0, epoch=0)
        mismatched = compute_supervised_loss(
            loss_fn,
            {"predicted_stress": torch.zeros_like(batch.teacher_stress)},
            batch,
            step=0,
            epoch=0,
        )
        assert matched["total_loss"].item() == pytest.approx(0.0)
        assert mismatched["total_loss"].item() > 0.0

    def test_missing_teacher_field_names_the_absent_target(
        self, small_batch: Batch
    ) -> None:
        """An unlabeled batch reports which teacher target the loss wanted."""
        loss_fn = as_composed_loss(EnergyMSELoss(target_key="teacher_energy"))
        with pytest.raises(AttributeError, match="teacher_energy"):
            compute_supervised_loss(
                loss_fn,
                {"predicted_energy": torch.zeros(small_batch.num_graphs, 1)},
                small_batch,
                step=0,
                epoch=0,
            )
