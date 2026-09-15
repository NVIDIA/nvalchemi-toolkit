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
"""Strategy-level wiring of the representation, curvature, and ensemble objectives.

Covers what :class:`~nvalchemi.training.distillation.DistillationStrategy` adds
around the loss terms themselves: the training functions that produce their
predictions, the auxiliary projector model, and the validation that refuses a
run those objectives cannot be trained by.
"""

from __future__ import annotations

import json
import math
import warnings
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
import torch
from torch import distributed as dist

from nvalchemi.data import Batch
from nvalchemi.data.datapipes.backends.zarr import AtomicDataZarrReader
from nvalchemi.data.datapipes.in_memory_dataset import InMemoryDataset
from nvalchemi.dynamics.base import BaseDynamics, ConvergenceHook, FusedStage
from nvalchemi.dynamics.integrators.nvt_langevin import NVTLangevin
from nvalchemi.dynamics.optimizers.fire import FIRE
from nvalchemi.hooks import TrainContext
from nvalchemi.models.base import BaseModelMixin
from nvalchemi.neighbors import compute_neighbors
from nvalchemi.training import (
    CheckpointHook,
    EnergyMSELoss,
    ForceMSELoss,
    OptimizerConfig,
    TrainingStage,
    ValidationConfig,
)
from nvalchemi.training.distillation import (
    BoltzmannMatchingLoss,
    DistillationStrategy,
    EmbeddingMatchingLoss,
    EmbeddingProjector,
    HessianMatchingLoss,
    InProcessTeacherScorer,
    OnPolicyConfig,
    SeedSource,
    default_distillation_fn,
    embedding_distillation_fn,
    hessian_distillation_fn,
    hessian_vector_product,
    label_dataset,
)
from nvalchemi.training.distillation import scoring as distillation_scoring
from nvalchemi.training.distillation._labels import _attach_teacher_labels
from nvalchemi.training.distillation.strategy import _supplied_runtime_objects
from test.training.conftest import _build_batch, _build_demo_model
from test.training.distillation.conftest import (
    _build_direct_force_model,
    _build_direct_force_teacher,
    _build_pair_potential_teacher,
    _build_periodic_batch,
    _build_replica_batch,
    _build_replica_dataset,
    _build_small_dataset,
    _DirectForceTeacher,
)

_STUDENT_WIDTH = 4
"""Embedding width of every student built here, narrower than the teacher's."""

_TEACHER_WIDTH = 8
"""Embedding width of every teacher built here."""

_LANGEVIN_KWARGS: dict[str, Any] = {
    "dt": 0.5,
    "temperature": 300.0,
    "friction": 0.01,
    "random_seed": 7,
}
"""Thermostat settings shared by every propagator built here."""


def _make_optimizer_config() -> list[OptimizerConfig]:
    """Return the Adam config every trained model here is given."""
    return [
        OptimizerConfig(optimizer_cls=torch.optim.Adam, optimizer_kwargs={"lr": 1e-2})
    ]


def _make_student(width: int = _STUDENT_WIDTH, seed: int = 1) -> _DirectForceTeacher:
    """Return a student whose embeddings are *width* wide."""
    return _build_direct_force_teacher(hidden_dim=width, seed=seed)


def _make_teacher(width: int = _TEACHER_WIDTH, seed: int = 2) -> _DirectForceTeacher:
    """Return a teacher whose embeddings are *width* wide."""
    return _build_direct_force_teacher(hidden_dim=width, seed=seed)


def _make_embedding_strategy(
    *,
    student: _DirectForceTeacher | None = None,
    teacher: _DirectForceTeacher | None = None,
    projector: EmbeddingProjector | None = None,
    training_fn: Any = embedding_distillation_fn,
    num_steps: int = 3,
    **overrides: Any,
) -> DistillationStrategy:
    """Return a strategy distilling energies and the teacher's representation."""
    models: dict[str, Any] = {
        "student": _make_student() if student is None else student,
        "teacher": _make_teacher() if teacher is None else teacher,
    }
    optimizer_configs = {"student": _make_optimizer_config()}
    if projector is not None:
        models["projector"] = projector
        optimizer_configs["projector"] = _make_optimizer_config()
    kwargs: dict[str, Any] = {
        "models": models,
        "optimizer_configs": optimizer_configs,
        "loss_fn": EnergyMSELoss(target_key="teacher_energy") + EmbeddingMatchingLoss(),
        "training_fn": training_fn,
        "num_steps": num_steps,
    }
    kwargs.update(overrides)
    return DistillationStrategy(**kwargs)


@contextmanager
def _direct_force_warning_filtered() -> Iterator[None]:
    """Silence the curvature validator's warning about a direct-force student.

    Every student built here predicts its forces with a head rather than as its
    energy's gradient, so a curvature objective warns at each construction. The
    warning itself is pinned by
    :meth:`TestHessianObjectiveValidation.test_direct_force_student_is_warned_about`.

    Yields
    ------
    None
    """
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            "Loss component.*differentiate the student's energy twice",
            UserWarning,
        )
        yield


def _make_hessian_strategy(
    *,
    student: BaseModelMixin | None = None,
    training_fn: Any = hessian_distillation_fn,
    num_steps: int = 3,
    **overrides: Any,
) -> DistillationStrategy:
    """Return a strategy distilling energies and the teacher's curvature."""
    kwargs: dict[str, Any] = {
        "models": {
            "student": _make_student() if student is None else student,
            "teacher": _make_teacher(),
        },
        "optimizer_configs": {"student": _make_optimizer_config()},
        "loss_fn": EnergyMSELoss(target_key="teacher_energy") + HessianMatchingLoss(),
        "training_fn": training_fn,
        "num_steps": num_steps,
    }
    kwargs.update(overrides)
    with _direct_force_warning_filtered():
        return DistillationStrategy(**kwargs)


def _make_anchor_dataset(scorer: InProcessTeacherScorer) -> InMemoryDataset:
    """Return a teacher-labeled anchor in the shape a generated frame has."""
    frames = _build_replica_batch(n_systems=8, base_seed=700, predictions=False)
    _attach_teacher_labels(frames, scorer.label(frames))
    return InMemoryDataset(in_memory_batch=frames)


def _make_on_policy_config(
    student: BaseModelMixin,
    teacher: BaseModelMixin,
    *,
    dynamics_fn: Callable[[BaseModelMixin], BaseDynamics] | None = None,
    replay_ratio: float = 1.0,
    replay_capacity: int | None = 20,
    seeds: SeedSource | None = None,
    **config_overrides: Any,
) -> OnPolicyConfig:
    """Return the segment loop every ensemble objective here generates with.

    The seed set is replicas of one 4-atom structure, since the ensemble the
    term is defined on is one system's configurations and a segment propagates
    the whole seed set as a single batch. The buffer is bounded by default
    because the estimator reads a batch as a sample of the *current* policy,
    which an unbounded buffer dilutes segment by segment.
    """
    return OnPolicyConfig(
        dynamics=NVTLangevin(student, **_LANGEVIN_KWARGS)
        if dynamics_fn is None
        else dynamics_fn(student),
        teacher_scorer=InProcessTeacherScorer(teacher, ["energy"]),
        seeds=SeedSource(_build_replica_dataset()) if seeds is None else seeds,
        replay_ratio=replay_ratio,
        replay_capacity=replay_capacity,
        steps_per_segment=2,
        batch_size=4,
        segment_steps=2,
        **config_overrides,
    )


def _make_distribution_strategy(
    *,
    dynamics_fn: Callable[[BaseModelMixin], BaseDynamics] | None = None,
    replay_ratio: float = 1.0,
    replay_capacity: int | None = 20,
    seeds: SeedSource | None = None,
    config_overrides: dict[str, Any] | None = None,
    **overrides: Any,
) -> DistillationStrategy:
    """Return an on-policy strategy whose objective includes a Boltzmann term."""
    student = _make_student()
    teacher = _make_teacher()
    config = _make_on_policy_config(
        student,
        teacher,
        dynamics_fn=dynamics_fn,
        replay_ratio=replay_ratio,
        replay_capacity=replay_capacity,
        seeds=seeds,
        **(config_overrides or {}),
    )
    kwargs: dict[str, Any] = {
        "models": {"student": student, "teacher": teacher},
        "optimizer_configs": {"student": _make_optimizer_config()},
        "loss_fn": EnergyMSELoss(target_key="teacher_energy") + BoltzmannMatchingLoss(),
        "num_steps": 4,
        "on_policy": config,
        "reference_dataset": None
        if replay_ratio == 1.0
        else _make_anchor_dataset(config.teacher_scorer),
    }
    kwargs.update(overrides)
    return DistillationStrategy(**kwargs)


def _make_offline_distribution_strategy() -> DistillationStrategy:
    """Return the ensemble objective configured without the segment loop."""
    return DistillationStrategy(
        models={"student": _make_student(), "teacher": _make_teacher()},
        optimizer_configs={"student": _make_optimizer_config()},
        loss_fn=EnergyMSELoss(target_key="teacher_energy") + BoltzmannMatchingLoss(),
        num_steps=2,
    )


def _make_fused_propagator(student: BaseModelMixin, **kwargs: Any) -> FusedStage:
    """Return two thermostatted sub-stages fused behind one propagator."""
    return FusedStage(
        sub_stages=[
            (0, NVTLangevin(student, **_LANGEVIN_KWARGS)),
            (1, NVTLangevin(student, **_LANGEVIN_KWARGS)),
        ],
        **kwargs,
    )


def _make_registered_convergence_propagator(student: BaseModelMixin) -> NVTLangevin:
    """Return a thermostat whose convergence criterion arrives by ``register_hook``."""
    dynamics = NVTLangevin(student, **_LANGEVIN_KWARGS)
    dynamics.register_hook(
        ConvergenceHook.from_fmax(
            0.05, source_status=0, target_status=dynamics.exit_status
        )
    )
    return dynamics


def _labeled_batch(strategy: DistillationStrategy, seed: int = 0) -> Batch:
    """Return a batch carrying the teacher fields *strategy* reads."""
    batch = _build_batch(seed=seed)
    strategy.attach_teacher_labels(batch)
    return batch


def _labeled_replica_batch(strategy: DistillationStrategy) -> Batch:
    """Return a labeled ensemble of equal-size replicas, as a segment produces."""
    batch = _build_replica_batch(base_seed=800, predictions=False)
    strategy.attach_teacher_labels(batch)
    return batch


@contextmanager
def _single_rank_process_group() -> Iterator[None]:
    """Initialize a one-rank gloo group, so a real DDP replica can be built."""
    dist.init_process_group("gloo", store=dist.HashStore(), rank=0, world_size=1)
    try:
        yield
    finally:
        dist.destroy_process_group()


def _replicated(models: dict[str, Any]) -> dict[str, Any]:
    """Return *models* with every trained model wrapped as a DDP replica."""
    return {
        name: model
        if name == "teacher"
        else torch.nn.parallel.DistributedDataParallel(model)
        for name, model in models.items()
    }


def _round_trip(
    strategy: DistillationStrategy, models: dict[str, Any]
) -> DistillationStrategy:
    """Return *strategy* rebuilt from its own JSON spec, over fresh *models*."""
    spec = json.loads(json.dumps(strategy.to_spec_dict()))
    with _direct_force_warning_filtered():
        return DistillationStrategy.from_spec_dict(spec, models=models)


class _EmbeddinglessStudent(_DirectForceTeacher):
    """Student that computes energies and forces but publishes no representation."""

    @property
    def embedding_shapes(self) -> dict[str, tuple[int, ...]]:
        """Return no embedding shapes."""
        return {}


class _RecordingLossHook:
    """Record the total loss of every completed training batch."""

    frequency = 1
    stage = TrainingStage.AFTER_BATCH

    def __init__(self) -> None:
        """Start with an empty loss trace."""
        self.losses: list[float] = []

    def __call__(self, ctx: TrainContext, stage: TrainingStage) -> None:  # noqa: ARG002
        """Append the loss the strategy just backpropagated."""
        self.losses.append(float(ctx.loss))


class _RecordingProbeHook:
    """Record the probe direction of every batch a forward pass is about to see."""

    frequency = 1
    stage = TrainingStage.BEFORE_FORWARD

    def __init__(self) -> None:
        """Start with an empty probe trace."""
        self.probes: list[torch.Tensor] = []

    def __call__(self, ctx: TrainContext, stage: TrainingStage) -> None:  # noqa: ARG002
        """Append the direction this batch's curvature label was taken along."""
        if ctx.batch is not None:
            self.probes.append(ctx.batch.teacher_hvp_probe.clone())


class TestEmbeddingDistillationFn:
    """The training function that produces the student's representation."""

    def test_embeddings_are_returned_as_a_prediction(self) -> None:
        """The stock prediction set gains the key the embedding term reads."""
        strategy = _make_embedding_strategy(projector=EmbeddingProjector(4, 8))
        predictions = embedding_distillation_fn(strategy.models, _build_batch())
        assert "predicted_node_embeddings" in predictions
        assert "predicted_energy" in predictions

    def test_projector_sets_the_prediction_width(self) -> None:
        """A cross-architecture run reaches the loss at the teacher's width."""
        strategy = _make_embedding_strategy(projector=EmbeddingProjector(4, 8))
        predictions = embedding_distillation_fn(strategy.models, _build_batch())
        assert predictions["predicted_node_embeddings"].shape[-1] == _TEACHER_WIDTH

    def test_matched_widths_need_no_projector(self) -> None:
        """A student as wide as its teacher is compared without an adapter."""
        strategy = _make_embedding_strategy(student=_make_student(_TEACHER_WIDTH))
        predictions = embedding_distillation_fn(strategy.models, _build_batch())
        assert predictions["predicted_node_embeddings"].shape[-1] == _TEACHER_WIDTH

    def test_predictions_stay_attached_to_the_student_graph(self) -> None:
        """The embedding prediction is what gradients reach the student through."""
        strategy = _make_embedding_strategy(projector=EmbeddingProjector(4, 8))
        predictions = embedding_distillation_fn(strategy.models, _build_batch())
        assert predictions["predicted_node_embeddings"].requires_grad

    def test_batch_keeps_no_embedding_field(self) -> None:
        """The batch is left as it was found, embeddings included."""
        strategy = _make_embedding_strategy(projector=EmbeddingProjector(4, 8))
        batch = _build_batch()
        embedding_distillation_fn(strategy.models, batch)
        assert "node_embeddings" not in batch

    @pytest.mark.skipif(not dist.is_gloo_available(), reason="gloo backend required")
    def test_distributed_replicas_are_unwrapped_for_the_embedding_pass(self) -> None:
        """DDP proxies ``__call__`` alone, so compute_embeddings needs the module."""
        strategy = _make_embedding_strategy(projector=EmbeddingProjector(4, 8))
        with _single_rank_process_group():
            predictions = embedding_distillation_fn(
                _replicated(strategy.models), _build_batch()
            )
        assert predictions["predicted_node_embeddings"].shape[-1] == _TEACHER_WIDTH
        assert predictions["predicted_node_embeddings"].requires_grad


class TestEmbeddingObjectiveRun:
    """Training a student and its projector against the teacher's representation."""

    def test_projector_is_trained_by_its_own_optimizer(self) -> None:
        """The auxiliary model's parameters move, which only its optimizer can do."""
        projector = EmbeddingProjector(_STUDENT_WIDTH, _TEACHER_WIDTH)
        strategy = _make_embedding_strategy(projector=projector)
        before = projector.projection.weight.detach().clone()

        strategy.run([_build_batch(seed=index) for index in range(3)])

        assert not torch.equal(before, projector.projection.weight)

    def test_student_is_trained_alongside_the_projector(self) -> None:
        """The student moves too, rather than the projector absorbing the objective.

        The composition is the representation term alone, so the only path to
        the student's trunk is the one the objective is for.
        """
        student = _make_student()
        strategy = _make_embedding_strategy(
            student=student,
            projector=EmbeddingProjector(_STUDENT_WIDTH, _TEACHER_WIDTH),
            loss_fn=EmbeddingMatchingLoss(),
        )
        before = student.model.trunk[0].weight.detach().clone()

        strategy.run([_build_batch(seed=index) for index in range(3)])

        assert not torch.equal(before, student.model.trunk[0].weight)

    @pytest.mark.parametrize(
        "width", [_STUDENT_WIDTH, _TEACHER_WIDTH], ids=["projected", "matched-widths"]
    )
    def test_embedding_objective_trains_the_student_trunk(self, width: int) -> None:
        """The term's gradient reaches the trunk the representation comes from.

        Nothing else in the composition can move it: the term is the whole
        objective, so a detached embedding would leave the trunk with no
        gradient at all and the projector absorbing the objective by itself.
        """
        student = _make_student(width)
        strategy = _make_embedding_strategy(
            student=student,
            projector=None
            if width == _TEACHER_WIDTH
            else EmbeddingProjector(width, _TEACHER_WIDTH),
            loss_fn=EmbeddingMatchingLoss(),
        )

        strategy.train_batch(_labeled_batch(strategy))

        gradient = student.model.trunk[0].weight.grad
        assert gradient is not None
        assert bool(gradient.any())

    def test_repeated_batch_drives_the_objective_down(self) -> None:
        """Training on one batch reduces the loss measured on it."""
        recorder = _RecordingLossHook()
        strategy = _make_embedding_strategy(
            projector=EmbeddingProjector(_STUDENT_WIDTH, _TEACHER_WIDTH),
            num_steps=20,
            hooks=[recorder],
        )

        strategy.run([_build_batch()] * 20)

        assert recorder.losses[-1] < recorder.losses[0]

    def test_embedding_signal_is_derived_from_the_loss(self) -> None:
        """The teacher is asked for embeddings because the loss reads them."""
        strategy = _make_embedding_strategy(projector=EmbeddingProjector(4, 8))
        assert "embeddings" in strategy.teacher_scorer.signals

    def test_training_fn_survives_a_spec_round_trip(self) -> None:
        """The training function is module-level, so the recipe stays serializable."""
        strategy = _make_embedding_strategy(
            projector=EmbeddingProjector(_STUDENT_WIDTH, _TEACHER_WIDTH)
        )
        rebuilt = _round_trip(
            strategy,
            {
                "student": _make_student(),
                "teacher": _make_teacher(),
                "projector": EmbeddingProjector(_STUDENT_WIDTH, _TEACHER_WIDTH),
            },
        )
        assert rebuilt.training_fn is embedding_distillation_fn

    def test_biasless_projector_restores_from_a_checkpoint(
        self, tmp_path: Path
    ) -> None:
        """A resume rebuilds the projector from its spec, non-default knobs included."""
        projector = EmbeddingProjector(_STUDENT_WIDTH, _TEACHER_WIDTH, bias=False)
        strategy = _make_embedding_strategy(projector=projector)
        strategy.run([_build_batch(seed=index) for index in range(3)])
        strategy.save_checkpoint(tmp_path)

        restored = DistillationStrategy.load_checkpoint(tmp_path, map_location="cpu")

        rebuilt = restored.models["projector"]
        assert rebuilt.projection.bias is None
        torch.testing.assert_close(
            rebuilt.projection.weight, projector.projection.weight
        )


class TestEmbeddingObjectiveValidation:
    """What a representation objective is refused for."""

    @pytest.mark.parametrize(
        "training_fn",
        [default_distillation_fn, hessian_distillation_fn],
        ids=["default", "hessian"],
    )
    def test_stock_training_fn_names_the_embedding_training_fn(
        self, training_fn: Any
    ) -> None:
        """No other stock training function produces embeddings, and each says so."""
        with pytest.raises(ValueError, match="embedding_distillation_fn"):
            _make_embedding_strategy(training_fn=training_fn)

    def test_validation_only_embedding_term_is_width_checked_at_construction(
        self,
    ) -> None:
        """A term only the validation loss holds is reconciled before the run too."""
        with pytest.raises(ValueError, match="validation loss component"):
            _make_embedding_strategy(
                training_fn=default_distillation_fn,
                loss_fn=EnergyMSELoss(target_key="teacher_energy"),
                validation_config=ValidationConfig(
                    validation_data=[_build_batch(seed=9)],
                    validation_fn=embedding_distillation_fn,
                    loss_fn=EnergyMSELoss(target_key="teacher_energy")
                    + EmbeddingMatchingLoss(),
                ),
            )

    def test_width_mismatch_without_a_projector_is_rejected(self) -> None:
        """A student narrower than its teacher needs the adapter, at construction."""
        with pytest.raises(ValueError, match="EmbeddingProjector"):
            _make_embedding_strategy()

    def test_projector_input_width_must_match_the_student(self) -> None:
        """A projector reading a width the student does not emit is refused."""
        with pytest.raises(ValueError, match="in_features"):
            _make_embedding_strategy(projector=EmbeddingProjector(6, _TEACHER_WIDTH))

    def test_projector_output_width_must_match_the_teacher(self) -> None:
        """A projector emitting a width the teacher does not have is refused."""
        with pytest.raises(ValueError, match="teacher's width"):
            _make_embedding_strategy(projector=EmbeddingProjector(_STUDENT_WIDTH, 5))

    def test_projector_must_be_optimized(self) -> None:
        """An auxiliary model with no optimizer would never train at all."""
        with pytest.raises(ValueError, match="unconfigured"):
            _make_embedding_strategy(
                projector=EmbeddingProjector(_STUDENT_WIDTH, _TEACHER_WIDTH),
                optimizer_configs={"student": _make_optimizer_config()},
            )

    def test_student_publishing_no_embeddings_is_rejected(self) -> None:
        """A student with no representation to match cannot serve the objective."""
        student = _EmbeddinglessStudent(
            _build_direct_force_model(hidden_dim=_STUDENT_WIDTH, seed=1)
        )
        with pytest.raises(ValueError, match="must publish a 'node_embeddings' shape"):
            _make_embedding_strategy(
                student=student,
                projector=EmbeddingProjector(_STUDENT_WIDTH, _TEACHER_WIDTH),
            )


class TestHessianDistillationFn:
    """The training function that produces the student's Hessian-vector product."""

    def test_product_is_returned_as_a_prediction(self) -> None:
        """The stock prediction set gains the key the curvature term reads."""
        strategy = _make_hessian_strategy()
        predictions = hessian_distillation_fn(strategy.models, _labeled_batch(strategy))
        assert predictions["predicted_hvp"].shape == (6, 3)

    def test_prediction_stays_attached_for_a_second_backward(self) -> None:
        """The product is created with a graph, which the loss backpropagates."""
        strategy = _make_hessian_strategy()
        predictions = hessian_distillation_fn(strategy.models, _labeled_batch(strategy))
        assert predictions["predicted_hvp"].requires_grad

    def test_product_uses_the_probe_the_teacher_was_labeled_with(self) -> None:
        """The student is differentiated along the stored direction, not a fresh one."""
        strategy = _make_hessian_strategy()
        batch = _labeled_batch(strategy)
        predictions = hessian_distillation_fn(strategy.models, batch)
        positions = batch.positions
        positions.requires_grad_(True)
        expected = hessian_vector_product(
            strategy.models["student"](batch)["energy"],
            positions,
            batch.teacher_hvp_probe,
        )
        torch.testing.assert_close(
            predictions["predicted_hvp"].detach(), expected.detach()
        )

    def test_unlabeled_batch_names_the_missing_probe(self) -> None:
        """Without a probe there is no direction to compare products along."""
        strategy = _make_hessian_strategy()
        with pytest.raises(AttributeError, match="teacher_hvp_probe"):
            hessian_distillation_fn(strategy.models, _build_batch())

    def test_conservative_student_gives_one_product_in_either_mode(self) -> None:
        """The narrowed pass derives no forces, so evaluation mode frees no graph."""
        strategy = _make_hessian_strategy(student=_build_demo_model())
        batch = _labeled_batch(strategy)
        student = strategy.models["student"]

        student.train()
        trained = hessian_distillation_fn(strategy.models, batch)["predicted_hvp"]
        student.eval()
        evaluated = hessian_distillation_fn(strategy.models, batch)["predicted_hvp"]

        assert evaluated.requires_grad
        torch.testing.assert_close(evaluated.detach(), trained.detach())

    def test_batch_grad_flags_are_left_as_they_were_found(self) -> None:
        """A batch trained on once stays usable as a propagator state afterwards."""
        strategy = _make_hessian_strategy(student=_build_demo_model())
        batch = _labeled_batch(strategy)
        assert batch.positions.requires_grad is False

        hessian_distillation_fn(strategy.models, batch)

        assert batch.positions.requires_grad is False
        batch.positions.add_(0.01)

    def test_narrowed_pass_reuses_the_student_list(self) -> None:
        """The second pass runs on the list the first one just consumed.

        Both passes are the same model over the same batch, so rebuilding the
        neighbor list for the narrowed one would cost a full build per
        optimizer step and produce the list that is already there.
        """
        student = _build_pair_potential_teacher(seed=1)
        strategy = _make_hessian_strategy(student=student)
        batch = _build_periodic_batch()
        compute_neighbors(batch, config=student.model_config.neighbor_config)
        strategy.attach_teacher_labels(batch)

        with patch.object(
            distillation_scoring,
            "compute_neighbors",
            wraps=distillation_scoring.compute_neighbors,
        ) as build:
            predictions = hessian_distillation_fn(strategy.models, batch)

        assert build.call_count == 0
        assert predictions["predicted_hvp"].shape == batch.positions.shape

    @pytest.mark.skipif(not dist.is_gloo_available(), reason="gloo backend required")
    def test_distributed_replicas_are_unwrapped_for_the_narrowed_pass(self) -> None:
        """The narrowed pass reads ``model_config``, which DDP does not proxy."""
        strategy = _make_hessian_strategy()
        batch = _labeled_batch(strategy)
        with _single_rank_process_group():
            predictions = hessian_distillation_fn(_replicated(strategy.models), batch)
        assert predictions["predicted_hvp"].shape == (6, 3)
        assert predictions["predicted_hvp"].requires_grad


class TestHessianObjectiveRun:
    """Training a student against the teacher's curvature."""

    def test_student_is_trained_through_the_second_derivative(self) -> None:
        """A run whose objective is curvature moves the student's weights."""
        student = _make_student()
        strategy = _make_hessian_strategy(student=student)
        before = student.model.energy_head.weight.detach().clone()

        strategy.run([_build_batch(seed=index) for index in range(3)])

        assert not torch.equal(before, student.model.energy_head.weight)

    def test_repeated_batch_drives_the_objective_down(self) -> None:
        """Training on one batch reduces the loss measured on it.

        The batch is labeled once up front so every step sees the same probe:
        the strategy relabels an unlabeled batch on every pass, which would
        redraw the direction and leave a fresh objective each step.
        """
        recorder = _RecordingLossHook()
        strategy = _make_hessian_strategy(num_steps=20, hooks=[recorder])

        strategy.run([_labeled_batch(strategy)] * 20)

        assert recorder.losses[-1] < recorder.losses[0]

    def test_training_fn_survives_a_spec_round_trip(self) -> None:
        """The training function is module-level, so the recipe stays serializable."""
        rebuilt = _round_trip(
            _make_hessian_strategy(),
            {"student": _make_student(), "teacher": _make_teacher()},
        )
        assert rebuilt.training_fn is hessian_distillation_fn

    def test_hessian_signal_and_probe_field_are_derived_from_the_loss(self) -> None:
        """The teacher is asked for curvature, and the probe travels with it."""
        strategy = _make_hessian_strategy()
        assert "hessian" in strategy.teacher_scorer.signals
        batch = _labeled_batch(strategy)
        assert "teacher_hvp" in batch
        assert "teacher_hvp_probe" in batch

    def test_conservative_student_validates_in_evaluation_mode(self) -> None:
        """Validation runs the student in eval mode with grad on, which the term needs."""
        strategy = _make_hessian_strategy(
            student=_build_demo_model(),
            validation_config=ValidationConfig(validation_data=[_build_batch(seed=5)]),
        )

        summary = strategy.validate()

        assert summary is not None
        assert "HessianMatchingLoss" in summary["per_component_unweighted"]
        assert math.isfinite(float(summary["total_loss"]))

    def test_validation_passes_score_along_one_probe(self) -> None:
        """The metric moves with the student rather than with a redrawn direction."""
        strategy = _make_hessian_strategy(
            student=_build_demo_model(),
            validation_config=ValidationConfig(validation_data=[_build_batch(seed=5)]),
        )

        first = strategy.validate()
        second = strategy.validate()

        assert first is not None
        assert second is not None
        assert first["per_component_unweighted"][
            "HessianMatchingLoss"
        ] == pytest.approx(second["per_component_unweighted"]["HessianMatchingLoss"])

    def test_validation_batches_are_scored_along_distinct_probes(self) -> None:
        """Each batch is keyed to its own direction, and to the same one next pass."""
        recorder = _RecordingProbeHook()
        strategy = _make_hessian_strategy(
            student=_build_demo_model(),
            hooks=[recorder],
            validation_config=ValidationConfig(
                validation_data=[_build_batch(seed=5), _build_batch(seed=6)]
            ),
        )

        strategy.validate()
        strategy.validate()

        assert len(recorder.probes) == 4
        first, second, third, fourth = recorder.probes
        assert not torch.equal(first, second)
        torch.testing.assert_close(third, first)
        torch.testing.assert_close(fourth, second)

    def test_training_labels_keep_redrawing_the_probe(self) -> None:
        """Outside validation, coverage of the Hessian still comes from redrawing."""
        strategy = _make_hessian_strategy()

        first = _labeled_batch(strategy)
        second = _labeled_batch(strategy)

        assert not torch.equal(first["teacher_hvp_probe"], second["teacher_hvp_probe"])

    def test_labeled_store_carries_both_hessian_fields(
        self, tmp_path: Path, small_dataset: InMemoryDataset
    ) -> None:
        """Offline labeling persists the probe alongside the product it belongs to."""
        store = tmp_path / "hessian.zarr"
        scorer = InProcessTeacherScorer(_make_teacher(), ["energy", "hessian"])

        label_dataset(small_dataset, scorer, store, batch_size=2)

        levels = AtomicDataZarrReader(store).field_levels
        assert levels["teacher_hvp"] == "atom"
        assert levels["teacher_hvp_probe"] == "atom"


class TestHessianObjectiveValidation:
    """What a curvature objective is refused for."""

    @pytest.mark.parametrize(
        "training_fn",
        [default_distillation_fn, embedding_distillation_fn],
        ids=["default", "embedding"],
    )
    def test_stock_training_fn_names_the_hessian_training_fn(
        self, training_fn: Any
    ) -> None:
        """No other stock training function differentiates twice, and each says so."""
        with pytest.raises(ValueError, match="hessian_distillation_fn"):
            _make_hessian_strategy(training_fn=training_fn)

    def test_student_computing_no_energy_is_rejected(self) -> None:
        """There is nothing to differentiate twice without an energy."""
        student = _make_student()
        student.set_config("active_outputs", {"forces"})
        with pytest.raises(ValueError, match="must.*compute an energy"):
            _make_hessian_strategy(student=student, loss_fn=HessianMatchingLoss())

    def test_validation_only_hessian_term_is_checked_at_construction(self) -> None:
        """A term only the validation loss holds is checked before the run too."""
        student = _make_student()
        student.set_config("active_outputs", {"forces"})
        with pytest.raises(ValueError, match="validation loss component"):
            _make_hessian_strategy(
                student=student,
                training_fn=default_distillation_fn,
                loss_fn=ForceMSELoss(target_key="teacher_forces"),
                validation_config=ValidationConfig(
                    validation_data=[_build_batch(seed=9)],
                    validation_fn=hessian_distillation_fn,
                    loss_fn=HessianMatchingLoss(),
                ),
            )

    def test_probe_field_is_not_a_loss_target(self) -> None:
        """The stored direction is how the product was taken, not what to match."""
        with pytest.raises(ValueError, match="not a quantity"):
            _make_hessian_strategy(
                loss_fn=HessianMatchingLoss()
                + ForceMSELoss(target_key="teacher_hvp_probe")
            )

    def test_direct_force_student_is_warned_about(self) -> None:
        """A force head gets no curvature signal, whatever the energy head learns."""
        with pytest.warns(UserWarning, match="predicts its forces with a head"):
            DistillationStrategy(
                models={"student": _make_student(), "teacher": _make_teacher()},
                optimizer_configs={"student": _make_optimizer_config()},
                loss_fn=HessianMatchingLoss(),
                training_fn=hessian_distillation_fn,
                num_steps=1,
            )

    def test_conservative_student_is_not_warned_about(
        self, recwarn: pytest.WarningsRecorder
    ) -> None:
        """A student whose forces are its energy's gradient gets the whole signal."""
        DistillationStrategy(
            models={"student": _build_demo_model(), "teacher": _make_teacher()},
            optimizer_configs={"student": _make_optimizer_config()},
            loss_fn=HessianMatchingLoss(),
            training_fn=hessian_distillation_fn,
            num_steps=1,
        )
        assert not [
            warning
            for warning in recwarn.list
            if "predicts its forces with a head" in str(warning.message)
        ]


class TestDistributionObjectiveValidation:
    """What an ensemble objective is refused for, and what it warns about."""

    def test_offline_run_is_rejected(self) -> None:
        """A dataset is not a sample of the student's own ensemble."""
        with pytest.raises(ValueError, match="on_policy=None"):
            _make_offline_distribution_strategy()

    def test_offline_rejection_names_the_reference_dataset_remedy(self) -> None:
        """Reweighting is not offered, so an existing dataset is mixed in instead."""
        with pytest.raises(ValueError, match="reaches the term as reference_dataset"):
            _make_offline_distribution_strategy()

    def test_on_policy_run_is_accepted(self) -> None:
        """The segment loop is what the estimator's uniform weights assume."""
        strategy = _make_distribution_strategy()
        assert strategy.on_policy is not None

    def test_relaxation_propagator_is_rejected(self) -> None:
        """A minimizer produces a path to a minimum rather than an ensemble."""
        with pytest.raises(ValueError, match="relaxation propagator"):
            _make_distribution_strategy(
                dynamics_fn=lambda student: FIRE(student, dt=0.1)
            )

    def test_converging_propagator_is_rejected(self) -> None:
        """A graph frozen at its exit status has stopped being sampled."""
        with pytest.raises(ValueError, match="converges graphs out"):
            _make_distribution_strategy(
                dynamics_fn=lambda student: NVTLangevin(
                    student,
                    convergence_hook=ConvergenceHook.from_fmax(0.05),
                    **_LANGEVIN_KWARGS,
                )
            )

    def test_registered_convergence_hook_is_rejected(self) -> None:
        """A criterion attached with register_hook freezes the same graphs out."""
        with pytest.raises(ValueError, match="converges graphs out"):
            _make_distribution_strategy(
                dynamics_fn=_make_registered_convergence_propagator
            )

    def test_fused_relaxation_sub_stage_is_rejected(self) -> None:
        """Flattening reaches a minimizer a composite propagator drives."""
        with pytest.raises(ValueError, match="relaxation propagator"):
            _make_distribution_strategy(
                dynamics_fn=lambda student: FusedStage(
                    sub_stages=[
                        (0, FIRE(student, dt=0.1)),
                        (1, NVTLangevin(student, **_LANGEVIN_KWARGS)),
                    ]
                )
            )

    def test_fused_sub_stage_convergence_hook_is_rejected(self) -> None:
        """A hook one sub-stage carries stops sampling for the graphs that reach it."""
        with pytest.raises(ValueError, match="converges graphs out"):
            _make_distribution_strategy(
                dynamics_fn=lambda student: FusedStage(
                    sub_stages=[
                        (0, NVTLangevin(student, **_LANGEVIN_KWARGS)),
                        (
                            1,
                            NVTLangevin(
                                student,
                                convergence_hook=ConvergenceHook.from_fmax(0.05),
                                **_LANGEVIN_KWARGS,
                            ),
                        ),
                    ]
                )
            )

    def test_fused_propagator_own_convergence_hook_is_rejected(self) -> None:
        """The composite's own hook is not hidden by sub-stages that carry none."""
        with pytest.raises(ValueError, match="converges graphs out"):
            _make_distribution_strategy(
                dynamics_fn=lambda student: _make_fused_propagator(
                    student, convergence_hook=ConvergenceHook.from_fmax(0.05)
                )
            )

    def test_config_level_convergence_is_rejected(self) -> None:
        """A lifecycle the loop installs at run time is still refused up front."""
        with pytest.raises(ValueError, match="converges graphs out"):
            _make_distribution_strategy(config_overrides={"convergence": 0.05})

    def test_config_level_convergence_hook_is_rejected(self) -> None:
        """The criterion spelled as a live hook converges the same graphs out."""
        with pytest.raises(ValueError, match="converges graphs out"):
            _make_distribution_strategy(
                config_overrides={
                    "convergence_hook": ConvergenceHook.from_fmax(
                        0.05, source_status=0, target_status=1
                    )
                }
            )

    def test_one_propagator_composed_twice_is_named_once(self) -> None:
        """The walk is identity-deduped, so a shared sub-stage is one propagator."""

        def _twice(student: BaseModelMixin) -> BaseDynamics:
            relaxation = FIRE(student, dt=0.1)
            return relaxation + relaxation

        with pytest.raises(ValueError, match=r"driving \['FIRE'\];"):
            _make_distribution_strategy(dynamics_fn=_twice)

    def test_fused_thermostats_are_accepted(self) -> None:
        """Flattening a composite of thermostats leaves nothing to object to."""
        strategy = _make_distribution_strategy(dynamics_fn=_make_fused_propagator)
        assert strategy.on_policy is not None

    def test_mixed_replay_ratio_warns(self) -> None:
        """Anchor frames the student never visited bias the estimator."""
        with pytest.warns(UserWarning, match="sample of the student's own ensemble"):
            _make_distribution_strategy(replay_ratio=0.5)

    def test_mixed_replay_ratio_warning_promises_no_unbiased_estimate(self) -> None:
        """``replay_ratio=1`` drops the anchor rows, not the stale generated ones."""
        with pytest.warns(UserWarning, match="what replay_capacity bounds") as record:
            _make_distribution_strategy(replay_ratio=0.5)
        assert not any("unbiased" in str(warning.message) for warning in record)

    def test_unbounded_replay_buffer_warns(self) -> None:
        """A buffer that retires nothing is a draw over every policy the run has had."""
        with pytest.warns(UserWarning, match="replay_capacity=None"):
            _make_distribution_strategy(replay_capacity=None)

    def test_bounded_replay_buffer_is_not_warned_about(
        self, recwarn: pytest.WarningsRecorder
    ) -> None:
        """Bounding the buffer is what keeps the uniform draw close to the policy."""
        _make_distribution_strategy(replay_capacity=20)
        assert not [
            warning
            for warning in recwarn.list
            if "replay_capacity" in str(warning.message)
        ]

    def test_validation_config_reusing_the_objective_is_rejected(self) -> None:
        """A held-out set is off-policy, so the term cannot be a validation loss."""
        with pytest.raises(ValueError, match="validation_config.loss_fn=None"):
            _make_distribution_strategy(
                validation_config=ValidationConfig(
                    validation_data=[_build_replica_batch(base_seed=900)]
                )
            )

    def test_validation_config_with_its_own_pointwise_loss_is_accepted(self) -> None:
        """An explicit validation loss keeps the ensemble term off held-out data."""
        strategy = _make_distribution_strategy(
            validation_config=ValidationConfig(
                validation_data=[_build_replica_batch(base_seed=900)],
                loss_fn=EnergyMSELoss(target_key="teacher_energy"),
            )
        )

        summary = strategy.validate()

        assert summary is not None
        assert "BoltzmannMatchingLoss" not in summary["per_component_unweighted"]


class TestDistributionObjectiveRun:
    """Generating, labeling, and training against the teacher's ensemble."""

    def test_segment_loop_trains_the_student_on_its_own_ensemble(self) -> None:
        """A seeded on-policy run completes with a finite loss on every batch."""
        recorder = _RecordingLossHook()
        strategy = _make_distribution_strategy(hooks=[recorder])
        student = strategy.models["student"]
        before = student.model.energy_head.weight.detach().clone()

        strategy.run()

        assert strategy.step_count == 4
        assert len(recorder.losses) == 4
        assert all(math.isfinite(loss) for loss in recorder.losses)
        assert not torch.equal(before, student.model.energy_head.weight)

    def test_repeated_ensemble_drives_the_objective_down(self) -> None:
        """Training on one ensemble reduces the relative entropy measured on it."""
        recorder = _RecordingLossHook()
        strategy = _make_distribution_strategy(hooks=[recorder], num_steps=20)
        batch = _labeled_replica_batch(strategy)

        for _ in range(20):
            strategy.train_batch(batch)

        assert recorder.losses[-1] < recorder.losses[0]

    def test_mixed_size_seeds_are_refused_by_the_term(self) -> None:
        """Generated frames of different sizes are not one Boltzmann distribution."""
        strategy = _make_distribution_strategy(seeds=SeedSource(_build_small_dataset()))
        with pytest.raises(ValueError, match="graphs of different sizes"):
            strategy.run()

    def test_ensemble_checkpoint_resumes_with_on_policy_resupplied(
        self, tmp_path: Path
    ) -> None:
        """A checkpointed ensemble run restarts once the segment loop is passed back.

        The spec carries no propagator, so the loop and the models it was built
        around go back in at the call; the checkpoint's weights are restored
        into those very models.
        """
        strategy = _make_distribution_strategy(
            hooks=[CheckpointHook(checkpoint_dir=tmp_path, step_interval=2)]
        )
        strategy.run()
        student = _make_student()
        teacher = _make_teacher()

        restored = DistillationStrategy.load_checkpoint(
            tmp_path,
            map_location="cpu",
            models={"student": student, "teacher": teacher},
            on_policy=_make_on_policy_config(student, teacher),
        )

        assert restored.step_count == strategy.step_count
        assert restored.on_policy is not None
        torch.testing.assert_close(
            student.model.energy_head.weight,
            strategy.models["student"].model.energy_head.weight,
        )

    def test_ensemble_checkpoint_refusal_names_the_way_back(
        self, tmp_path: Path
    ) -> None:
        """Reloading without the loop is refused, and the message says what restores it."""
        strategy = _make_distribution_strategy(
            hooks=[CheckpointHook(checkpoint_dir=tmp_path, step_interval=2)]
        )
        strategy.run()

        with pytest.raises(ValueError, match="restore_checkpoint"):
            DistillationStrategy.load_checkpoint(tmp_path, map_location="cpu")

    def test_bounded_buffer_retires_the_frames_it_overflows_by(self) -> None:
        """Eviction is what keeps a segment's draw close to the current policy."""
        strategy = _make_distribution_strategy(replay_capacity=5)

        strategy.run()

        assert strategy.replay_buffer is not None
        assert len(strategy.replay_buffer) == 5


class TestSuppliedRuntimeObjectPrecedence:
    """Which segment loop a rebuild ends up with when more than one is on offer.

    Runtime objects resolve in a fixed order: an explicit keyword on
    ``from_spec_dict``, then whatever a checkpoint rebuild offered over
    :func:`~nvalchemi.training.distillation.strategy._supplied_runtime_objects`,
    then a rebuild from the recipe the spec carries. The last leg is pinned on
    the branch that owns the recipe half of ``from_spec_dict``; this branch's
    spec carries no ``on_policy``, so the two legs pinned here are the two it
    can build.
    """

    def test_an_explicit_loop_outranks_the_supplied_one(self) -> None:
        """A loop passed at the call wins over one a checkpoint rebuild offered."""
        strategy = _make_distribution_strategy()
        student = _make_student()
        teacher = _make_teacher()
        explicit = _make_on_policy_config(student, teacher)
        offered = _make_on_policy_config(student, teacher)
        spec = json.loads(json.dumps(strategy.to_spec_dict()))

        with _supplied_runtime_objects(on_policy=offered):
            restored = DistillationStrategy.from_spec_dict(
                spec,
                models={"student": student, "teacher": teacher},
                on_policy=explicit,
            )

        assert restored.on_policy is explicit

    def test_a_supplied_loop_is_taken_when_the_call_passes_none(self) -> None:
        """The offer is what a checkpoint rebuild has instead of a keyword."""
        strategy = _make_distribution_strategy()
        student = _make_student()
        teacher = _make_teacher()
        offered = _make_on_policy_config(student, teacher)
        spec = json.loads(json.dumps(strategy.to_spec_dict()))

        with _supplied_runtime_objects(on_policy=offered):
            restored = DistillationStrategy.from_spec_dict(
                spec, models={"student": student, "teacher": teacher}
            )

        assert restored.on_policy is offered


class TestAdvancedObjectivesOnCuda:
    """The two objectives that add a pass of their own, run on a device."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_embedding_objective_trains_on_cuda(self) -> None:
        """Labeling, the second student pass, and the projector all follow the batch."""
        recorder = _RecordingLossHook()
        student = _make_student()
        projector = EmbeddingProjector(_STUDENT_WIDTH, _TEACHER_WIDTH)
        strategy = _make_embedding_strategy(
            student=student,
            projector=projector,
            devices=[torch.device("cuda")],
            hooks=[recorder],
        )
        before = student.model.trunk[0].weight.detach().clone()

        strategy.run([_build_batch(seed=index) for index in range(3)])

        assert projector.projection.weight.device.type == "cuda"
        assert len(recorder.losses) == 3
        assert all(math.isfinite(loss) for loss in recorder.losses)
        assert not torch.equal(before, student.model.trunk[0].weight.cpu())

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_hessian_objective_trains_on_cuda(self) -> None:
        """The probe is drawn on the batch's device and the double backward follows."""
        recorder = _RecordingLossHook()
        student = _make_student()
        strategy = _make_hessian_strategy(
            student=student, devices=[torch.device("cuda")], hooks=[recorder]
        )
        before = student.model.energy_head.weight.detach().clone()

        strategy.run([_build_batch(seed=index) for index in range(3)])

        assert len(recorder.losses) == 3
        assert all(math.isfinite(loss) for loss in recorder.losses)
        assert not torch.equal(before, student.model.energy_head.weight.cpu())
