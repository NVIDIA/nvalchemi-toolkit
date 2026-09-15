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
"""Tests for :mod:`nvalchemi.training.distillation.losses.embedding`."""

from __future__ import annotations

import json

import pytest
import torch

from nvalchemi.data import Batch
from nvalchemi.training._spec import create_model_spec_from_json
from nvalchemi.training._spec_utils import _module_spec_from_attrs
from nvalchemi.training.distillation import (
    EmbeddingMatchingLoss,
    EmbeddingProjector,
    InProcessTeacherScorer,
)
from nvalchemi.training.distillation._labels import _attach_teacher_labels
from nvalchemi.training.losses.composition import (
    ComposedLossFunction,
    compute_supervised_loss,
    loss_component_to_spec,
)
from nvalchemi.training.losses.terms import EnergyMSELoss
from test.training.distillation.conftest import (
    _build_direct_force_teacher,
    _DirectForceTeacher,
)

_UNEVEN_BATCH_IDX = torch.tensor([0, 0, 0, 1])
"""Graph assignment of four atoms split three-to-one across two graphs."""


def _make_labeled_batch(batch: Batch, teacher: _DirectForceTeacher) -> Batch:
    """Return *batch* carrying the teacher's node embeddings."""
    scorer = InProcessTeacherScorer(teacher, ["embeddings"])
    _attach_teacher_labels(batch, scorer.label(batch))
    return batch


class TestEmbeddingMatchingLossValues:
    """Scalar values the embedding matching loss produces."""

    def test_graph_balanced_value_matches_a_hand_computation(self) -> None:
        """Each graph's mean squared component error is averaged over graphs."""
        loss_fn = EmbeddingMatchingLoss()
        pred = torch.tensor([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [2.0, 2.0]])
        loss = loss_fn(
            pred, torch.zeros(4, 2), batch_idx=_UNEVEN_BATCH_IDX, num_graphs=2
        )
        assert loss.item() == pytest.approx((1.0 + 4.0) / 2)

    def test_global_mean_weights_every_component_equally(self) -> None:
        """Without graph balancing the loss is one mean over all components."""
        loss_fn = EmbeddingMatchingLoss(normalize_by_atom_count=False)
        pred = torch.tensor([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [2.0, 2.0]])
        loss = loss_fn(
            pred, torch.zeros(4, 2), batch_idx=_UNEVEN_BATCH_IDX, num_graphs=2
        )
        assert loss.item() == pytest.approx((6 * 1.0 + 2 * 4.0) / 8)

    def test_per_sample_loss_holds_one_value_per_graph(self) -> None:
        """The graph-balanced reduction publishes a detached ``(B,)`` diagnostic."""
        loss_fn = EmbeddingMatchingLoss()
        pred = torch.tensor([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [2.0, 2.0]])
        loss_fn(pred, torch.zeros(4, 2), batch_idx=_UNEVEN_BATCH_IDX, num_graphs=2)
        torch.testing.assert_close(loss_fn.per_sample_loss, torch.tensor([1.0, 4.0]))

    def test_zero_residual_gives_zero_loss(self) -> None:
        """A student whose representation matches the teacher's scores zero."""
        loss_fn = EmbeddingMatchingLoss()
        target = torch.randn(4, 3)
        loss = loss_fn(
            target.clone(), target, batch_idx=_UNEVEN_BATCH_IDX, num_graphs=2
        )
        assert loss.item() == pytest.approx(0.0)

    def test_gradients_flow_to_the_prediction(self) -> None:
        """The student's embeddings receive gradient from the term."""
        loss_fn = EmbeddingMatchingLoss()
        pred = torch.ones(4, 2, requires_grad=True)
        loss_fn(
            pred, torch.zeros(4, 2), batch_idx=_UNEVEN_BATCH_IDX, num_graphs=2
        ).backward()
        assert pred.grad is not None
        assert bool((pred.grad != 0).all())

    def test_nonfinite_targets_are_excluded_from_the_loss(self) -> None:
        """A masked component leaves the rest dividing by their own count."""
        loss_fn = EmbeddingMatchingLoss()
        target = torch.zeros(2, 2)
        target[0, 0] = float("nan")
        loss = loss_fn(
            torch.ones(2, 2), target, batch_idx=torch.tensor([0, 1]), num_graphs=2
        )
        assert loss.item() == pytest.approx(1.0)


class TestEmbeddingMatchingLossContract:
    """Shape, metadata, and serialization contract of the loss term."""

    def test_width_mismatch_names_the_projector_remedy(self) -> None:
        """Comparing widths that differ points at the auxiliary projector."""
        loss_fn = EmbeddingMatchingLoss()
        with pytest.raises(ValueError, match="EmbeddingProjector"):
            loss_fn(
                torch.zeros(4, 2),
                torch.zeros(4, 5),
                batch_idx=_UNEVEN_BATCH_IDX,
                num_graphs=2,
            )

    def test_node_count_mismatch_is_rejected(self) -> None:
        """A disagreement that is not a width still fails shape validation."""
        loss_fn = EmbeddingMatchingLoss()
        with pytest.raises(ValueError, match="shape must match exactly"):
            loss_fn(
                torch.zeros(3, 2),
                torch.zeros(4, 2),
                batch_idx=_UNEVEN_BATCH_IDX,
                num_graphs=2,
            )

    def test_missing_graph_metadata_is_rejected(self) -> None:
        """The graph-balanced reduction names the metadata it needs."""
        loss_fn = EmbeddingMatchingLoss()
        with pytest.raises(ValueError, match="batch_idx"):
            loss_fn(torch.zeros(4, 2), torch.zeros(4, 2))

    def test_loss_declares_it_needs_no_evaluation_gradients(self) -> None:
        """Embeddings are computed directly, so validation can run no-grad."""
        assert EmbeddingMatchingLoss().requires_eval_grad is False

    def test_default_keys_match_the_teacher_and_student_fields(self) -> None:
        """Defaults line up the teacher's embedding signal with the student's."""
        loss_fn = EmbeddingMatchingLoss()
        assert loss_fn.target_key == "teacher_node_embeddings"
        assert loss_fn.prediction_key == "predicted_node_embeddings"

    def test_spec_round_trip_rebuilds_an_equivalent_loss(self) -> None:
        """A JSON round-tripped spec rebuilds the loss with its configuration."""
        spec = loss_component_to_spec(
            EmbeddingMatchingLoss(normalize_by_atom_count=False, ignore_nonfinite=False)
        )
        rebuilt = create_model_spec_from_json(
            json.loads(spec.model_dump_json())
        ).build()
        assert isinstance(rebuilt, EmbeddingMatchingLoss)
        assert rebuilt.normalize_by_atom_count is False
        assert rebuilt.ignore_nonfinite is False

    def test_teacher_embeddings_score_zero_against_themselves(
        self, small_batch: Batch, direct_force_teacher: _DirectForceTeacher
    ) -> None:
        """Labels read back as predictions reproduce the teacher exactly."""
        batch = _make_labeled_batch(small_batch, direct_force_teacher)
        predictions = {
            "predicted_node_embeddings": batch.teacher_node_embeddings.clone()
        }
        out = compute_supervised_loss(
            ComposedLossFunction([EmbeddingMatchingLoss()]),
            predictions,
            batch,
            step=0,
            epoch=0,
        )
        assert out["total_loss"].item() == pytest.approx(0.0)

    def test_composition_reports_the_embedding_component(
        self, small_batch: Batch, direct_force_teacher: _DirectForceTeacher
    ) -> None:
        """The term composes with an energy term like any other leaf."""
        scorer = InProcessTeacherScorer(direct_force_teacher, ["energy", "embeddings"])
        _attach_teacher_labels(small_batch, scorer.label(small_batch))
        loss_fn = ComposedLossFunction(
            [EnergyMSELoss(target_key="teacher_energy"), EmbeddingMatchingLoss()],
            weights=[1.0, 0.5],
        )
        predictions = {
            "predicted_energy": torch.zeros_like(small_batch.teacher_energy),
            "predicted_node_embeddings": torch.zeros_like(
                small_batch.teacher_node_embeddings
            ),
        }
        out = compute_supervised_loss(
            loss_fn, predictions, small_batch, step=0, epoch=0
        )
        assert set(out["per_component_unweighted"]) == {
            "EnergyMSELoss",
            "EmbeddingMatchingLoss",
        }
        assert out["total_loss"].item() > 0.0


class TestEmbeddingProjector:
    """The learnable width adapter registered as an auxiliary model."""

    def test_linear_projector_maps_between_the_two_widths(self) -> None:
        """A single linear layer takes the student's width to the teacher's."""
        projector = EmbeddingProjector(4, 6)
        assert projector(torch.zeros(5, 4)).shape == (5, 6)

    def test_hidden_layer_builds_a_two_layer_map(self) -> None:
        """``hidden_features`` inserts one nonlinearity between two linear maps."""
        projector = EmbeddingProjector(4, 6, hidden_features=8)
        assert len(projector.projection) == 3
        assert projector(torch.zeros(5, 4)).shape == (5, 6)

    def test_projection_is_differentiable_into_its_parameters(self) -> None:
        """Gradients reach the projector's own weights, which its optimizer needs."""
        projector = EmbeddingProjector(4, 6)
        projector(torch.ones(5, 4)).pow(2).sum().backward()
        assert projector.projection.weight.grad is not None

    def test_compute_embeddings_projects_a_batch_in_place(
        self, small_batch: Batch
    ) -> None:
        """The batch-shaped entry point replaces the embeddings it is handed."""
        student = _build_direct_force_teacher(hidden_dim=4, seed=1)
        student.compute_embeddings(small_batch)
        EmbeddingProjector(4, 6).compute_embeddings(small_batch)
        assert small_batch["node_embeddings"].shape[-1] == 6

    def test_published_embedding_shape_is_the_output_width(self) -> None:
        """The projector advertises the width it maps to, as a model must."""
        assert EmbeddingProjector(4, 6).embedding_shapes == {"node_embeddings": (6,)}

    def test_nonpositive_width_is_rejected(self) -> None:
        """A width that cannot describe a representation fails at construction."""
        with pytest.raises(ValueError, match="widths must be positive"):
            EmbeddingProjector(4, 0)

    def test_biasless_linear_projector_survives_a_spec_round_trip(self) -> None:
        """``bias`` changes the state-dict keys, so the spec has to carry it."""
        projector = EmbeddingProjector(4, 6, bias=False)

        rebuilt = _module_spec_from_attrs(projector).build()
        rebuilt.load_state_dict(projector.state_dict())

        assert rebuilt.projection.bias is None

    def test_biasless_hidden_projector_survives_a_spec_round_trip(self) -> None:
        """The two-layer form drops two bias tensors rather than one."""
        projector = EmbeddingProjector(4, 6, hidden_features=8, bias=False)

        rebuilt = _module_spec_from_attrs(projector).build()
        rebuilt.load_state_dict(projector.state_dict())

        assert rebuilt.projection[0].bias is None
        assert rebuilt.projection[2].bias is None

    def test_repr_reports_every_constructor_knob(self) -> None:
        """What the spec carries is what the repr shows, widths and bias alike."""
        assert "bias=False" in repr(EmbeddingProjector(4, 6, bias=False))
