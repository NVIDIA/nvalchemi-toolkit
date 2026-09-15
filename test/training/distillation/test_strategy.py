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
"""Tests for :mod:`nvalchemi.training.distillation.strategy`."""

from __future__ import annotations

import json
from collections import OrderedDict
from collections.abc import Iterable
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
import torch

from nvalchemi.data import Batch
from nvalchemi.data.datapipes.backends.zarr import AtomicDataZarrReader
from nvalchemi.data.datapipes.dataloader import DataLoader
from nvalchemi.data.datapipes.dataset import Dataset
from nvalchemi.data.datapipes.in_memory_dataset import InMemoryDataset
from nvalchemi.hooks import TrainContext
from nvalchemi.models.base import BaseModelMixin, ModelConfig
from nvalchemi.training import (
    EnergyMSELoss,
    ForceMSELoss,
    OptimizerConfig,
    TrainingStage,
    TrainingStrategy,
    ValidationConfig,
)
from nvalchemi.training.distillation import (
    DistillationStrategy,
    InProcessTeacherScorer,
    OnPolicyConfig,
    PerAtomEnergyMatchingLoss,
    default_distillation_fn,
    label_dataset,
)
from nvalchemi.training.distillation.strategy import _TeacherLabelHook
from nvalchemi.training.hooks.ema import EMAHook
from nvalchemi.training.hooks.mixed_precision import MixedPrecisionHook
from nvalchemi.training.losses.composition import ComposedLossFunction, DTypePolicy
from test.training.conftest import _build_batch, _build_demo_model
from test.training.distillation.conftest import (
    _build_direct_force_teacher,
    _build_lj_teacher,
    _DirectForceTeacher,
)

_TEACHER_FIELDS = ("teacher_energy", "teacher_forces", "teacher_node_energies")
"""Batch fields the three-signal teacher objective below reads."""


def _make_optimizer_config() -> OptimizerConfig:
    """Return the Adam config the distillation tests optimize students with."""
    return OptimizerConfig(
        optimizer_cls=torch.optim.Adam, optimizer_kwargs={"lr": 1e-2}
    )


def _make_teacher_loss(dtype_policy: DTypePolicy = "strict") -> ComposedLossFunction:
    """Return the three-signal teacher objective the execution tests train on."""
    return (
        EnergyMSELoss(target_key="teacher_energy", dtype_policy=dtype_policy)
        + ForceMSELoss(
            target_key="teacher_forces",
            normalize_by_atom_count=True,
            dtype_policy=dtype_policy,
        )
        + PerAtomEnergyMatchingLoss(dtype_policy=dtype_policy)
    )


def _make_models(teacher: BaseModelMixin | None = None) -> dict[str, BaseModelMixin]:
    """Return a student/teacher pair of independently seeded demo models."""
    return {
        "student": _build_direct_force_teacher(seed=1),
        "teacher": teacher
        if teacher is not None
        else _build_direct_force_teacher(seed=2),
    }


def _make_strategy(**overrides: Any) -> DistillationStrategy:
    """Return a distillation strategy over a direct-force student/teacher pair."""
    kwargs: dict[str, Any] = {
        "models": _make_models(),
        "optimizer_configs": {"student": [_make_optimizer_config()]},
        "loss_fn": _make_teacher_loss(),
        "num_steps": 4,
    }
    kwargs.update(overrides)
    return DistillationStrategy(**kwargs)


def _make_reduced_precision_strategy(
    dtype: torch.dtype = torch.bfloat16, dtype_policy: DTypePolicy = "strict"
) -> DistillationStrategy:
    """Return a distillation strategy whose student runs below single precision."""
    return _make_strategy(
        models=_make_models()
        | {"student": _build_direct_force_teacher(seed=1).to(dtype)},
        loss_fn=_make_teacher_loss(dtype_policy),
    )


def _make_loader(n_batches: int = 4) -> list[Batch]:
    """Return a small re-iterable list of unlabeled training batches."""
    return [_build_batch(seed=10 * index) for index in range(n_batches)]


def _make_labeled_loader(
    dataset: InMemoryDataset,
    strategy: DistillationStrategy,
    store: Path,
    signals: Iterable[str] | None = None,
) -> DataLoader:
    """Return a loader over *dataset* labeled offline by *strategy*'s teacher."""
    label_dataset(
        dataset,
        InProcessTeacherScorer(
            strategy.models["teacher"],
            strategy.teacher_scorer.signals if signals is None else signals,
        ),
        store,
        batch_size=2,
    )
    return DataLoader(
        Dataset(reader=AtomicDataZarrReader(store), device="cpu"),
        batch_size=2,
        use_streams=False,
    )


def _labeling_hook_count(strategy: DistillationStrategy) -> int:
    """Return how many internal teacher-labeling hooks *strategy* holds."""
    return sum(isinstance(hook, _TeacherLabelHook) for hook in strategy.hooks)


def _make_on_policy_recipe(seed_store: Path, segment_steps: int) -> dict[str, Any]:
    """Return a segment-loop recipe naming its propagator and seed store."""
    return {
        "dynamics": {
            "cls_path": "nvalchemi.dynamics.integrators.nvt_langevin.NVTLangevin",
            "kwargs": {
                "dt": 0.5,
                "temperature": 300.0,
                "friction": 0.01,
                "random_seed": 7,
            },
        },
        "teacher_scorer": {
            "teacher": "teacher",
            "signals": ["energy", "forces"],
            "cast_to": None,
        },
        "seeds": {
            "dataset": {"path": str(seed_store), "device": "cpu"},
            "max_atoms": None,
            "max_edges": None,
            "max_batch_size": None,
            "recycle": False,
        },
        "replay_ratio": 1.0,
        "steps_per_segment": 2,
        "batch_size": 4,
        "segment_steps": segment_steps,
        "label_frequency": 1,
    }


class _ToyDistillationStrategy(DistillationStrategy):
    """A user-authored subclass a spec can name by dotted path."""


_TOY_STRATEGY_PATH = (
    f"{_ToyDistillationStrategy.__module__}.{_ToyDistillationStrategy.__qualname__}"
)
"""Dotted path of the subclass above, as a spec's ``strategy_cls`` carries it."""


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


class _RecordingLabelHook:
    """Snapshot the teacher fields a batch carries when its forward pass starts."""

    frequency = 1
    stage = TrainingStage.BEFORE_FORWARD

    def __init__(self, fields: tuple[str, ...]) -> None:
        """Start with an empty trace of the given teacher fields."""
        self.fields = fields
        self.seen: list[dict[str, torch.Tensor]] = []

    def __call__(self, ctx: TrainContext, stage: TrainingStage) -> None:  # noqa: ARG002
        """Append a clone of every recorded field present on the batch."""
        batch = ctx.batch
        self.seen.append(
            {field: batch[field].clone() for field in self.fields if field in batch}
        )


class _PartialOutputStudent(torch.nn.Module, BaseModelMixin):
    """Student that declares a stress output it never computes."""

    def __init__(self) -> None:
        """Declare energy and stress, and hold one trainable scale."""
        super().__init__()
        self.scale = torch.nn.Parameter(torch.ones(1))
        self.model_config = ModelConfig(
            outputs=frozenset({"energy", "stress"}),
            autograd_outputs=frozenset(),
            autograd_inputs=frozenset(),
            neighbor_config=None,
        )

    @property
    def embedding_shapes(self) -> dict[str, tuple[int, ...]]:
        """Return no embeddings for this stub student."""
        return {}

    def compute_embeddings(self, data: Batch, **kwargs: Any) -> Batch:  # noqa: ARG002
        """Return *data* unchanged because this stub has no embeddings."""
        return data

    def forward(self, data: Batch, **kwargs: Any) -> OrderedDict:  # noqa: ARG002
        """Return an energy and leave the declared stress unset."""
        return self.adapt_output(
            {"energy": self.scale.expand(data.num_graphs, 1).clone()}, data
        )


def _student_energy_only_fn(
    models: dict[str, BaseModelMixin], batch: Batch
) -> dict[str, torch.Tensor]:
    """Return only the student's energy prediction."""
    return {"predicted_energy": models["student"](batch)["energy"]}


def _student_plus_projector_fn(
    models: dict[str, BaseModelMixin], batch: Batch
) -> dict[str, torch.Tensor]:
    """Return the student's energy corrected by an auxiliary projector's."""
    return {
        "predicted_energy": models["student"](batch)["energy"]
        + models["projector"](batch)["energy"]
    }


class TestDistillationStrategyValidation:
    """Construction-time contract of :class:`DistillationStrategy`."""

    def test_direct_force_teacher_is_accepted(self) -> None:
        """A teacher predicting forces from a head, not a gradient, is first class."""
        strategy = _make_strategy()
        assert strategy.models["teacher"].model_config.autograd_outputs == frozenset()
        assert sorted(strategy.teacher_scorer.signals) == [
            "energy",
            "forces",
            "node_energies",
        ]

    def test_autograd_force_teacher_is_accepted(self) -> None:
        """A conservative teacher works through the same path, unvalidated either way."""
        strategy = _make_strategy(
            models={
                "student": _build_direct_force_teacher(seed=1),
                "teacher": _build_demo_model(),
            },
            loss_fn=EnergyMSELoss(target_key="teacher_energy")
            + ForceMSELoss(target_key="teacher_forces"),
        )
        assert sorted(strategy.teacher_scorer.signals) == ["energy", "forces"]

    def test_single_model_input_is_rejected(self) -> None:
        """A bare model cannot express the student/teacher contract."""
        with pytest.raises(ValueError, match="named-model mapping"):
            _make_strategy(
                models=_build_direct_force_teacher(),
                optimizer_configs=_make_optimizer_config(),
            )

    def test_missing_teacher_model_is_rejected(self) -> None:
        """Named models without a teacher entry are refused."""
        with pytest.raises(ValueError, match="named-model mapping"):
            _make_strategy(
                models={"student": _build_direct_force_teacher(seed=1)},
            )

    def test_optimizer_config_for_the_teacher_is_rejected(self) -> None:
        """Configuring the teacher would train it, so it is refused."""
        with pytest.raises(ValueError, match="frozen by omission"):
            _make_strategy(
                optimizer_configs={
                    "student": [_make_optimizer_config()],
                    "teacher": [_make_optimizer_config()],
                }
            )

    def test_unconfigured_student_is_rejected(self) -> None:
        """A student without an optimizer would never be updated."""
        models = _make_models()
        models["helper"] = _build_direct_force_teacher(seed=3)
        with pytest.raises(ValueError, match="unconfigured"):
            _make_strategy(
                models=models, optimizer_configs={"helper": [_make_optimizer_config()]}
            )

    def test_unconfigured_auxiliary_model_is_rejected(self) -> None:
        """An extra model is trainable or absent; silently freezing it is not offered."""
        models = _make_models()
        models["projector"] = _build_direct_force_teacher(seed=3)
        with pytest.raises(ValueError, match="projector"):
            _make_strategy(models=models)

    def test_configured_auxiliary_model_is_accepted(self) -> None:
        """A third model with its own optimizer config satisfies the contract."""
        models = _make_models()
        models["projector"] = _build_direct_force_teacher(seed=3)
        strategy = _make_strategy(
            models=models,
            optimizer_configs={
                "student": [_make_optimizer_config()],
                "projector": [_make_optimizer_config()],
            },
            training_fn=_student_plus_projector_fn,
            loss_fn=EnergyMSELoss(target_key="teacher_energy"),
        )
        assert sorted(strategy.models) == ["projector", "student", "teacher"]

    def test_student_without_a_declared_output_is_rejected(self) -> None:
        """A loss reading a prediction the student never declares fails up front."""
        with pytest.raises(ValueError, match="Student cannot produce"):
            _make_strategy(
                models=_make_models() | {"student": _PartialOutputStudent()},
                loss_fn=EnergyMSELoss(target_key="teacher_energy")
                + ForceMSELoss(target_key="teacher_forces"),
            )

    def test_student_without_atomic_energies_is_rejected(self) -> None:
        """The per-atom term needs a student head, and its absence names the term."""
        with pytest.raises(ValueError, match="PerAtomEnergyMatchingLoss"):
            _make_strategy(
                models=_make_models() | {"student": _build_demo_model()},
                loss_fn=EnergyMSELoss(target_key="teacher_energy")
                + PerAtomEnergyMatchingLoss(),
            )

    def test_student_with_an_inactive_output_is_rejected(self) -> None:
        """A declared-but-inactive output fails at construction, not on batch one."""
        student = _build_direct_force_teacher(seed=1)
        student.set_config("active_outputs", {"energy", "forces"})
        with pytest.raises(ValueError, match="active_outputs"):
            _make_strategy(models=_make_models() | {"student": student})

    def test_widening_the_active_outputs_accepts_the_student(self) -> None:
        """The error's own remedy — widening the active set — makes the run valid."""
        student = _build_direct_force_teacher(seed=1)
        student.set_config("active_outputs", {"energy", "forces", "atomic_energies"})
        strategy = _make_strategy(models=_make_models() | {"student": student})
        assert strategy.models["student"] is student

    def test_embedding_prediction_key_names_compute_embeddings(self) -> None:
        """An embedding prediction is refused with the route that would serve it."""
        with pytest.raises(ValueError, match="compute_embeddings"):
            _make_strategy(
                loss_fn=PerAtomEnergyMatchingLoss(
                    prediction_key="predicted_node_embeddings"
                )
            )

    def test_custom_training_fn_owns_the_student_output_contract(self) -> None:
        """A caller-supplied ``training_fn`` opts out of the student-output check."""
        strategy = _make_strategy(
            models=_make_models() | {"student": _PartialOutputStudent()},
            loss_fn=EnergyMSELoss(target_key="teacher_energy"),
            training_fn=_student_energy_only_fn,
        )
        assert strategy.training_fn is _student_energy_only_fn

    def test_unmappable_teacher_target_is_rejected(self) -> None:
        """A ``teacher_*`` target with no signal behind it names the field and the fix."""
        with pytest.raises(ValueError, match="teacher_dipole") as excinfo:
            _make_strategy(loss_fn=EnergyMSELoss(target_key="teacher_dipole"))
        assert "named outside it" in str(excinfo.value)

    def test_loss_without_teacher_targets_is_rejected(self) -> None:
        """A strategy that would never consult the teacher is refused."""
        with pytest.raises(ValueError, match="at least one teacher signal"):
            _make_strategy(loss_fn=EnergyMSELoss())

    def test_explicit_signals_must_cover_the_loss_targets(self) -> None:
        """An explicit signal set that starves a loss term is refused."""
        with pytest.raises(ValueError, match="missing"):
            _make_strategy(teacher_signals={"energy"})

    def test_explicit_signals_may_exceed_the_loss_targets(self) -> None:
        """Requesting more signals than the loss reads is allowed."""
        strategy = _make_strategy(
            loss_fn=EnergyMSELoss(target_key="teacher_energy"),
            teacher_signals={"energy", "forces"},
        )
        assert strategy.teacher_scorer.signals == frozenset({"energy", "forces"})

    def test_signal_the_teacher_cannot_produce_is_rejected(self) -> None:
        """Signals are checked against the teacher's declared outputs at construction."""
        with pytest.raises(ValueError, match="Teacher cannot produce"):
            _make_strategy(
                loss_fn=EnergyMSELoss(target_key="teacher_energy"),
                teacher_signals={"energy", "stress"},
            )

    def test_unprefixed_prediction_key_is_refused_at_construction(self) -> None:
        """The stock ``training_fn`` only emits ``predicted_*``, so nothing else fits."""
        with pytest.raises(ValueError, match="predicted_"):
            _make_strategy(
                loss_fn=PerAtomEnergyMatchingLoss(prediction_key="atomic_energies")
            )

    def test_validation_loss_widens_the_resolved_teacher_signals(self) -> None:
        """A validation-only teacher target is part of the derived signal set."""
        strategy = _make_strategy(
            loss_fn=EnergyMSELoss(target_key="teacher_energy"),
            validation_config=ValidationConfig(
                validation_data=[_build_batch(seed=5)],
                loss_fn=ForceMSELoss(target_key="teacher_forces"),
            ),
        )
        assert strategy.teacher_scorer.signals == frozenset({"energy", "forces"})

    def test_explicit_teacher_signals_must_cover_the_validation_loss(self) -> None:
        """An explicit set that starves the validation loss names it and is refused."""
        with pytest.raises(ValueError, match="teacher_signals must cover") as excinfo:
            _make_strategy(
                loss_fn=EnergyMSELoss(target_key="teacher_energy"),
                teacher_signals={"energy"},
                validation_config=ValidationConfig(
                    validation_data=[_build_batch(seed=5)],
                    loss_fn=ForceMSELoss(target_key="teacher_forces"),
                ),
            )
        assert "'validation': ['forces']" in str(excinfo.value)

    def test_validation_loss_rejects_an_unknown_teacher_target(self) -> None:
        """A validation target under the reserved prefix is checked like a training one."""
        with pytest.raises(ValueError, match="supported teacher target"):
            _make_strategy(
                validation_config=ValidationConfig(
                    validation_data=[_build_batch(seed=5)],
                    loss_fn=EnergyMSELoss(target_key="teacher_dipole"),
                )
            )

    def test_validation_loss_prediction_keys_are_checked_at_construction(self) -> None:
        """A validation loss the narrowed student cannot serve fails up front."""
        student = _build_direct_force_teacher(seed=1)
        student.set_config("active_outputs", {"energy"})
        with pytest.raises(ValueError, match="active_outputs") as excinfo:
            _make_strategy(
                models=_make_models() | {"student": student},
                loss_fn=EnergyMSELoss(target_key="teacher_energy"),
                validation_config=ValidationConfig(
                    validation_data=[_build_batch(seed=5)],
                    loss_fn=ForceMSELoss(target_key="teacher_forces"),
                ),
            )
        assert "validation loss component" in str(excinfo.value)

    def test_custom_validation_fn_skips_the_prediction_key_check(self) -> None:
        """A caller-supplied validation function owns its own prediction contract."""
        student = _build_direct_force_teacher(seed=1)
        student.set_config("active_outputs", {"energy"})
        strategy = _make_strategy(
            models=_make_models() | {"student": student},
            loss_fn=EnergyMSELoss(target_key="teacher_energy"),
            validation_config=ValidationConfig(
                validation_data=[_build_batch(seed=5)],
                loss_fn=ForceMSELoss(target_key="teacher_forces"),
                validation_fn=_student_energy_only_fn,
            ),
        )
        assert strategy.teacher_scorer.signals == frozenset({"energy", "forces"})

    def test_validation_config_assigned_after_construction_keeps_the_resolved_signals(
        self,
    ) -> None:
        """Assignment does not re-validate, so a late validation loss widens nothing."""
        strategy = _make_strategy(loss_fn=EnergyMSELoss(target_key="teacher_energy"))
        strategy.validation_config = ValidationConfig(
            validation_data=[_build_batch(seed=5)],
            loss_fn=ForceMSELoss(target_key="teacher_forces"),
        )
        assert strategy.teacher_scorer.signals == frozenset({"energy"})

    def test_default_training_fn_is_the_student_forward(self) -> None:
        """An omitted ``training_fn`` falls back to the stock student forward."""
        assert _make_strategy().training_fn is default_distillation_fn

    def test_explicit_training_fn_is_preserved(self) -> None:
        """A caller-supplied ``training_fn`` is not replaced by the default."""
        strategy = _make_strategy(training_fn=_student_energy_only_fn)
        assert strategy.training_fn is _student_energy_only_fn


class TestDistillationStrategyLabeling:
    """On-the-fly labeling of training batches."""

    def test_unlabeled_batch_is_labeled_by_the_teacher(self) -> None:
        """A batch missing teacher fields triggers exactly one teacher pass."""
        strategy = _make_strategy()
        with patch.object(
            strategy.teacher_scorer,
            "label",
            wraps=strategy.teacher_scorer.label,
        ) as spy:
            strategy.train_batch(_build_batch())
        assert spy.call_count == 1

    def test_prelabeled_batch_bypasses_the_teacher(self) -> None:
        """A batch carrying every teacher field is trained on without the teacher."""
        strategy = _make_strategy()
        batch = _build_batch()
        strategy.attach_teacher_labels(batch)
        with patch.object(
            strategy.teacher_scorer,
            "label",
            wraps=strategy.teacher_scorer.label,
        ) as spy:
            strategy.train_batch(batch)
        assert spy.call_count == 0

    def test_attaching_labels_twice_scores_once(self) -> None:
        """Labeling is idempotent, so a labeled batch is never re-scored."""
        strategy = _make_strategy()
        batch = _build_batch()
        with patch.object(
            strategy.teacher_scorer,
            "label",
            wraps=strategy.teacher_scorer.label,
        ) as spy:
            assert strategy.attach_teacher_labels(batch) is True
            assert strategy.attach_teacher_labels(batch) is False
        assert spy.call_count == 1

    def test_partially_labeled_batch_is_relabeled_in_full(self) -> None:
        """A batch carrying only some teacher fields is re-scored, overwriting them."""
        narrow = _make_strategy(
            models=_make_models(teacher=_build_direct_force_teacher(seed=4)),
            loss_fn=EnergyMSELoss(target_key="teacher_energy"),
        )
        strategy = _make_strategy()
        batch = _build_batch()
        narrow.attach_teacher_labels(batch)
        stale = batch.teacher_energy.clone()
        with patch.object(
            strategy.teacher_scorer,
            "label",
            wraps=strategy.teacher_scorer.label,
        ) as spy:
            assert strategy.attach_teacher_labels(batch) is True
        assert spy.call_count == 1
        assert "teacher_forces" in batch
        assert not torch.equal(batch.teacher_energy, stale)

    def test_labeling_ignores_an_ambient_autocast_region(self) -> None:
        """Labels computed inside an autocast block match the full-precision ones."""
        strategy = _make_strategy()
        reference = _build_batch()
        strategy.attach_teacher_labels(reference)
        probe = _build_batch()
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            assert strategy.attach_teacher_labels(probe) is True
        for field in ("teacher_energy", "teacher_forces", "teacher_node_energies"):
            assert probe[field].dtype == reference[field].dtype
            assert torch.equal(probe[field], reference[field])

    def test_mixed_precision_run_labels_at_the_teacher_precision(self) -> None:
        """An AMP training step attaches the same labels a full-precision one does."""
        recorder = _RecordingLabelHook(("teacher_energy", "teacher_forces"))
        strategy = _make_strategy(
            loss_fn=EnergyMSELoss(
                target_key="teacher_energy", dtype_policy="prediction_to_target"
            )
            + ForceMSELoss(
                target_key="teacher_forces", dtype_policy="prediction_to_target"
            ),
            hooks=[MixedPrecisionHook(precision=torch.bfloat16), recorder],
        )
        reference = _build_batch()
        strategy.attach_teacher_labels(reference)
        strategy.train_batch(_build_batch())
        assert len(recorder.seen) == 1
        for field, values in recorder.seen[0].items():
            assert torch.equal(values, reference[field])

    def test_attached_fields_match_the_resolved_signals(self) -> None:
        """Every resolved signal lands on the batch at the level it declares."""
        strategy = _make_strategy()
        batch = _build_batch()
        strategy.attach_teacher_labels(batch)
        assert batch.teacher_energy.shape == (batch.num_graphs, 1)
        assert batch.teacher_forces.shape == (batch.num_nodes, 3)
        assert batch.teacher_node_energies.shape == (batch.num_nodes,)

    def test_attached_labels_match_a_direct_scorer_call(self) -> None:
        """Attached values equal the scorer's own output for the same batch."""
        strategy = _make_strategy()
        batch = _build_batch()
        expected = InProcessTeacherScorer(
            strategy.models["teacher"], strategy.teacher_scorer.signals
        ).label(batch)
        strategy.attach_teacher_labels(batch)
        for field, (values, _) in expected.items():
            torch.testing.assert_close(batch[field], values)

    def test_stress_signal_is_attached_from_a_periodic_teacher(
        self, periodic_batch: Batch
    ) -> None:
        """A stress-capable teacher attaches ``teacher_stress`` at system level."""
        strategy = _make_strategy(
            models=_make_models(teacher=_build_lj_teacher()),
            loss_fn=EnergyMSELoss(target_key="teacher_energy"),
            teacher_signals={"energy", "stress"},
        )
        assert strategy.attach_teacher_labels(periodic_batch) is True
        assert periodic_batch.teacher_stress.shape == (periodic_batch.num_graphs, 3, 3)

    def test_label_missing_false_leaves_a_batch_unlabeled(self) -> None:
        """Opting out of labeling surfaces the missing target instead of hiding it."""
        strategy = _make_strategy(label_missing=False)
        with pytest.raises(AttributeError, match="teacher_energy"):
            strategy.train_batch(_build_batch())

    def test_label_missing_false_trains_on_prelabeled_batches(self) -> None:
        """Pre-labeled data needs no on-the-fly labeling at all."""
        strategy = _make_strategy(label_missing=False)
        batch = _build_batch()
        strategy.attach_teacher_labels(batch)
        strategy.train_batch(batch)
        assert strategy.step_count == 1

    def test_validation_batches_are_labeled_on_the_fly(self) -> None:
        """The same seam labels validation data, so an unlabeled set evaluates."""
        strategy = _make_strategy(
            validation_config=ValidationConfig(validation_data=[_build_batch(seed=5)])
        )
        with patch.object(
            strategy.teacher_scorer,
            "label",
            wraps=strategy.teacher_scorer.label,
        ) as spy:
            summary = strategy.validate()
        assert spy.call_count == 1
        assert summary is not None
        assert summary["total_loss"] > 0.0

    def test_label_missing_false_leaves_validation_batches_unlabeled(self) -> None:
        """Opting out covers validation too, so the missing target still surfaces."""
        strategy = _make_strategy(
            label_missing=False,
            validation_config=ValidationConfig(validation_data=[_build_batch(seed=5)]),
        )
        with pytest.raises(AttributeError, match="Validation batch is missing"):
            strategy.validate()

    def test_prelabeled_validation_batches_are_evaluated(self) -> None:
        """Validation data labeled up front evaluates through the ordinary loop."""
        strategy = _make_strategy()
        validation_batch = _build_batch(seed=5)
        strategy.attach_teacher_labels(validation_batch)
        strategy.validation_config = ValidationConfig(
            validation_data=[validation_batch], grad_mode="enabled"
        )
        summary = strategy.validate()
        assert summary is not None
        assert summary["total_loss"] > 0.0

    def test_run_validates_an_unlabeled_validation_set(self, device: str) -> None:
        """A mid-run validation pass over unlabeled data completes on either device."""
        strategy = _make_strategy(
            num_steps=2,
            devices=[torch.device(device)],
            validation_config=ValidationConfig(
                validation_data=[_build_batch(seed=5)], every_n_steps=2
            ),
        )
        strategy.run(_make_loader(2))
        assert strategy.step_count == 2
        assert strategy.last_validation is not None
        assert strategy.last_validation["total_loss"] > 0.0

    def test_run_validates_a_teacher_target_the_training_loss_omits(self) -> None:
        """The seam labels a validation-only teacher field because derivation saw it."""
        strategy = _make_strategy(
            num_steps=2,
            loss_fn=EnergyMSELoss(target_key="teacher_energy"),
            validation_config=ValidationConfig(
                validation_data=[_build_batch(seed=5)],
                loss_fn=ForceMSELoss(target_key="teacher_forces"),
                every_n_steps=2,
            ),
        )
        strategy.run(_make_loader(2))
        assert strategy.last_validation is not None
        assert strategy.last_validation["total_loss"] > 0.0

    def test_use_ema_auto_validates_the_student_ema_against_the_live_teacher(
        self,
    ) -> None:
        """``use_ema="auto"`` pairs the averaged student with the unaveraged teacher."""
        strategy = _make_strategy(
            num_steps=2,
            hooks=[EMAHook(model_key="student", decay=0.9)],
            validation_config=ValidationConfig(
                validation_data=[_build_batch(seed=5)],
                every_n_steps=2,
                use_ema="auto",
            ),
        )
        strategy.run(_make_loader(2))
        assert strategy.last_validation is not None
        assert strategy.last_validation["model_source"] == "mixed"
        assert strategy.last_validation["ema_model_keys"] == ["student"]
        assert strategy.teacher_scorer.teacher is strategy.models["teacher"]


class TestDistillationStrategyExecution:
    """Optimization behavior of a distillation run."""

    def test_teacher_parameters_receive_no_gradients(self) -> None:
        """Backward through the loss touches the student only."""
        strategy = _make_strategy()
        strategy.train_batch(_build_batch())
        assert all(
            parameter.grad is None
            for parameter in strategy.models["teacher"].parameters()
        )
        assert any(
            parameter.grad is not None
            for parameter in strategy.models["student"].parameters()
        )

    def test_run_labels_each_batch_and_leaves_the_caller_batches_alone(self) -> None:
        """Every training batch costs one teacher pass, and the caller's stay unlabeled."""
        strategy = _make_strategy(num_steps=6)
        loader = _make_loader(2)
        with patch.object(
            strategy.teacher_scorer,
            "label",
            wraps=strategy.teacher_scorer.label,
        ) as spy:
            strategy.run(loader)
        assert strategy.step_count == 6
        assert spy.call_count == 6
        assert all("teacher_energy" not in batch for batch in loader)

    def test_teacher_weights_are_unchanged_by_training(self) -> None:
        """A frozen teacher is bit-for-bit identical after a run."""
        strategy = _make_strategy(num_steps=8)
        before = [
            parameter.detach().clone()
            for parameter in strategy.models["teacher"].parameters()
        ]
        strategy.run(_make_loader())
        for parameter, snapshot in zip(
            strategy.models["teacher"].parameters(), before, strict=True
        ):
            torch.testing.assert_close(parameter, snapshot)

    def test_run_decreases_the_teacher_loss(self) -> None:
        """Twelve steps against a direct-force teacher lower the composed loss."""
        recorder = _RecordingLossHook()
        strategy = _make_strategy(num_steps=12, hooks=[recorder])
        strategy.run(_make_loader(2))
        assert strategy.step_count == 12
        assert len(recorder.losses) == 12
        assert sum(recorder.losses[-2:]) < sum(recorder.losses[:2])

    def test_run_decreases_the_loss_for_an_autograd_teacher(self) -> None:
        """The same run converges when the teacher's forces come from autograd."""
        recorder = _RecordingLossHook()
        strategy = _make_strategy(
            models={
                "student": _build_direct_force_teacher(seed=1),
                "teacher": _build_demo_model(),
            },
            loss_fn=EnergyMSELoss(target_key="teacher_energy")
            + ForceMSELoss(target_key="teacher_forces", normalize_by_atom_count=True),
            num_steps=12,
            hooks=[recorder],
        )
        strategy.run(_make_loader(2))
        assert sum(recorder.losses[-2:]) < sum(recorder.losses[:2])

    def test_auxiliary_model_is_updated_alongside_the_student(self) -> None:
        """A configured third model receives its own optimizer updates in a run."""
        models = _make_models()
        models["projector"] = _build_direct_force_teacher(seed=3)
        strategy = _make_strategy(
            models=models,
            optimizer_configs={
                "student": [_make_optimizer_config()],
                "projector": [_make_optimizer_config()],
            },
            training_fn=_student_plus_projector_fn,
            loss_fn=EnergyMSELoss(target_key="teacher_energy"),
        )
        before = [
            parameter.detach().clone()
            for parameter in strategy.models["projector"].parameters()
        ]
        strategy.run(_make_loader(2))
        assert strategy.step_count == 4
        assert any(
            not torch.equal(parameter, snapshot)
            for parameter, snapshot in zip(
                strategy.models["projector"].parameters(), before, strict=True
            )
        )
        assert all(
            parameter.grad is None
            for parameter in strategy.models["teacher"].parameters()
        )

    def test_float64_teacher_labels_a_float32_student(self) -> None:
        """Labels are cast to the student's dtype, so a mixed pair trains as usual."""
        strategy = _make_strategy(
            models=_make_models(teacher=_build_direct_force_teacher(seed=2).double())
        )
        batch = _build_batch()
        strategy.attach_teacher_labels(batch)
        assert batch.teacher_energy.dtype == torch.float32
        strategy.train_batch(batch)
        assert strategy.step_count == 1

    def test_bfloat16_student_gets_float32_labels(self) -> None:
        """A reduced-precision student receives labels at single precision."""
        strategy = _make_reduced_precision_strategy()
        assert strategy.teacher_scorer.cast_to == torch.float32
        batch = _build_batch()
        strategy.attach_teacher_labels(batch)
        assert batch.teacher_energy.dtype == torch.float32
        assert batch.teacher_forces.dtype == torch.float32

    def test_float16_student_gets_float32_labels(self) -> None:
        """The floor applies to every dtype narrower than single precision."""
        strategy = _make_reduced_precision_strategy(torch.float16)
        assert strategy.teacher_scorer.cast_to == torch.float32
        batch = _build_batch()
        strategy.attach_teacher_labels(batch)
        assert batch.teacher_node_energies.dtype == torch.float32

    def test_float64_student_keeps_float64_labels(self) -> None:
        """Precision above the floor is preserved, so a float64 student stays exact."""
        strategy = _make_strategy(
            models=_make_models()
            | {"student": _build_direct_force_teacher(seed=1).double()}
        )
        assert strategy.teacher_scorer.cast_to == torch.float64
        batch = _build_batch()
        strategy.attach_teacher_labels(batch)
        assert batch.teacher_node_energies.dtype == torch.float64

    def test_float64_student_raises_over_a_float32_store(
        self, small_dataset: InMemoryDataset, tmp_path: Path
    ) -> None:
        """A float64 student reads float32 labels back from a store and fails strict."""
        strategy = _make_strategy(
            models=_make_models()
            | {"student": _build_direct_force_teacher(seed=1).double()}
        )
        loader = _make_labeled_loader(
            small_dataset, strategy, tmp_path / "labeled.zarr"
        )
        assert strategy.attach_teacher_labels(_build_batch()) is True
        with pytest.raises(ValueError, match="dtype mismatch"):
            strategy.train_batch(next(iter(loader)))

    def test_reduced_precision_student_raises_on_both_labeling_paths(
        self, small_dataset: InMemoryDataset, tmp_path: Path
    ) -> None:
        """Strict dtypes fail a bf16 student the same way offline and on the fly."""
        online = _make_reduced_precision_strategy()
        offline = _make_reduced_precision_strategy()
        loader = _make_labeled_loader(small_dataset, offline, tmp_path / "labeled.zarr")
        with pytest.raises(ValueError, match="dtype mismatch"):
            online.train_batch(
                next(iter(DataLoader(small_dataset, batch_size=2, use_streams=False)))
            )
        with pytest.raises(ValueError, match="dtype mismatch"):
            offline.train_batch(next(iter(loader)))

    def test_stored_and_on_the_fly_labels_are_bit_equal_for_a_bfloat16_student(
        self, small_dataset: InMemoryDataset, tmp_path: Path
    ) -> None:
        """Both label paths land on float32, so a store and the seam agree exactly."""
        strategy = _make_reduced_precision_strategy()
        stored = next(
            iter(
                _make_labeled_loader(small_dataset, strategy, tmp_path / "labeled.zarr")
            )
        )
        probe = next(iter(DataLoader(small_dataset, batch_size=2, use_streams=False)))
        assert strategy.attach_teacher_labels(probe) is True
        for field in _TEACHER_FIELDS:
            assert probe[field].dtype == stored[field].dtype
            assert torch.equal(probe[field], stored[field])

    def test_bfloat16_student_trains_identically_from_a_store_and_on_the_fly(
        self, small_dataset: InMemoryDataset, tmp_path: Path
    ) -> None:
        """The two label paths drive one objective once the loss casts to the target."""
        offline = _make_reduced_precision_strategy(dtype_policy="prediction_to_target")
        online = _make_reduced_precision_strategy(dtype_policy="prediction_to_target")
        stored = next(
            iter(
                _make_labeled_loader(small_dataset, offline, tmp_path / "labeled.zarr")
            )
        )
        offline.train_batch(stored)
        online.train_batch(
            next(iter(DataLoader(small_dataset, batch_size=2, use_streams=False)))
        )
        torch.testing.assert_close(
            offline.loss_fn.components[2].per_sample_loss,
            online.loss_fn.components[2].per_sample_loss,
        )

    def test_mixed_teacher_and_reference_objective_trains(self) -> None:
        """Reference and teacher targets compose, and the reference fields survive."""
        strategy = _make_strategy(
            loss_fn=EnergyMSELoss() + ForceMSELoss(target_key="teacher_forces")
        )
        assert strategy.teacher_scorer.signals == frozenset({"forces"})
        batch = _build_batch()
        reference_energy = batch.energy.clone()
        assert strategy.attach_teacher_labels(batch) is True
        assert "teacher_forces" in batch
        assert "teacher_energy" not in batch
        assert torch.equal(batch.energy, reference_energy)
        strategy.train_batch(batch)
        assert strategy.step_count == 1

    def test_run_over_a_labeled_store_never_calls_the_teacher(
        self, small_dataset: InMemoryDataset, tmp_path: Path
    ) -> None:
        """Offline labels stream through the reader and loader with no teacher pass."""
        strategy = _make_strategy(num_steps=6)
        loader = _make_labeled_loader(
            small_dataset, strategy, tmp_path / "labeled.zarr"
        )
        with patch.object(
            strategy.teacher_scorer,
            "label",
            wraps=strategy.teacher_scorer.label,
        ) as spy:
            strategy.run(loader)
        assert spy.call_count == 0
        assert strategy.step_count == 6

    def test_explicit_signals_missing_from_a_store_are_scored_every_batch(
        self, small_dataset: InMemoryDataset, tmp_path: Path
    ) -> None:
        """A resolved field the store lacks costs one teacher pass per batch."""
        strategy = _make_strategy(
            num_steps=3,
            loss_fn=EnergyMSELoss(target_key="teacher_energy"),
            teacher_signals={"energy", "forces"},
        )
        loader = _make_labeled_loader(
            small_dataset, strategy, tmp_path / "energy_only.zarr", signals=["energy"]
        )
        with patch.object(
            strategy.teacher_scorer,
            "label",
            wraps=strategy.teacher_scorer.label,
        ) as spy:
            strategy.run(loader)
        assert spy.call_count == 3


class TestDistillationStrategySerialization:
    """Spec and checkpoint round-trips of the subclass."""

    def test_spec_dict_carries_the_distillation_fields(self) -> None:
        """Signals serialize as a sorted list alongside the labeling policy."""
        spec = _make_strategy(
            teacher_signals={"forces", "energy", "node_energies"},
            label_missing=False,
        ).to_spec_dict()
        assert spec["teacher_signals"] == ["energy", "forces", "node_energies"]
        assert spec["label_missing"] is False
        assert spec["training_fn"].endswith("default_distillation_fn")

    def test_spec_round_trip_keeps_one_labeling_hook(self) -> None:
        """Specs carry no hooks, so a rebuild re-injects the seam exactly once."""
        strategy = _make_strategy(hooks=[_RecordingLossHook()])
        assert isinstance(strategy.hooks[0], _TeacherLabelHook)
        spec = json.loads(json.dumps(strategy.to_spec_dict()))
        rebuilt = DistillationStrategy.from_spec_dict(spec, models=_make_models())
        assert _labeling_hook_count(rebuilt) == 1

    def test_checkpoint_round_trip_keeps_one_labeling_hook(
        self, tmp_path: Path
    ) -> None:
        """A restored strategy labels through one seam, not a duplicated one."""
        strategy = _make_strategy()
        strategy.save_checkpoint(tmp_path)
        restored = DistillationStrategy.load_checkpoint(tmp_path, map_location="cpu")
        assert _labeling_hook_count(restored) == 1

    def test_derived_signals_serialize_as_null(self) -> None:
        """A derived signal set stays derived across a round-trip."""
        assert _make_strategy().to_spec_dict()["teacher_signals"] is None

    def test_spec_round_trip_rebuilds_the_strategy(self) -> None:
        """A JSON round-trip rebuilds a runnable strategy from re-supplied models."""
        strategy = _make_strategy(teacher_signals={"energy", "forces", "node_energies"})
        spec = json.loads(json.dumps(strategy.to_spec_dict()))
        rebuilt = DistillationStrategy.from_spec_dict(spec, models=_make_models())
        assert isinstance(rebuilt, DistillationStrategy)
        assert rebuilt.teacher_signals == frozenset(
            {"energy", "forces", "node_energies"}
        )
        assert rebuilt.training_fn is default_distillation_fn
        rebuilt.train_batch(_build_batch())
        assert rebuilt.step_count == 1

    def test_checkpoint_round_trip_restores_the_subclass(self, tmp_path: Path) -> None:
        """``strategy_cls`` brings back a distillation strategy with its counters."""
        strategy = _make_strategy(label_missing=False)
        batch = _build_batch()
        strategy.attach_teacher_labels(batch)
        strategy.train_batch(batch)
        assert strategy.save_checkpoint(tmp_path) == 0

        restored = DistillationStrategy.load_checkpoint(tmp_path, map_location="cpu")
        assert isinstance(restored, DistillationStrategy)
        assert restored.step_count == 1
        assert restored.label_missing is False
        assert sorted(restored.models) == ["student", "teacher"]
        for parameter, expected in zip(
            restored.models["teacher"].parameters(),
            strategy.models["teacher"].parameters(),
            strict=True,
        ):
            torch.testing.assert_close(parameter, expected)

    def test_restored_strategy_labels_and_trains(self, tmp_path: Path) -> None:
        """The rebuilt scorer reproduces the original labels and drives a step."""
        strategy = _make_strategy(label_missing=False)
        strategy.save_checkpoint(tmp_path)
        restored = DistillationStrategy.load_checkpoint(tmp_path, map_location="cpu")

        expected = _build_batch(seed=7)
        strategy.attach_teacher_labels(expected)
        probe = _build_batch(seed=7)
        assert restored.attach_teacher_labels(probe) is True
        torch.testing.assert_close(probe.teacher_energy, expected.teacher_energy)
        torch.testing.assert_close(probe.teacher_forces, expected.teacher_forces)
        restored.train_batch(probe)
        assert restored.step_count == 1

    def test_spec_dict_names_the_strategy_class(self) -> None:
        """A bare spec says which strategy rebuilds it, not just the base one."""
        spec = _make_strategy().to_spec_dict()
        assert spec["strategy_cls"].endswith(".DistillationStrategy")

    def test_spec_class_path_survives_the_checkpoint_dict(self) -> None:
        """The checkpoint bundle keeps one class path and still restores the subclass."""
        spec = json.loads(json.dumps(_make_strategy().to_checkpoint_dict()))
        assert spec["strategy_cls"].endswith(".DistillationStrategy")
        restored = DistillationStrategy.from_checkpoint_dict(
            spec, models=_make_models()
        )
        assert isinstance(restored, DistillationStrategy)
        assert _labeling_hook_count(restored) == 1

    def test_from_spec_dict_rejects_a_foreign_strategy_class(self) -> None:
        """A spec written by another strategy is refused rather than reinterpreted."""
        spec = _make_strategy().to_spec_dict()
        spec["strategy_cls"] = "nvalchemi.training.strategy.TrainingStrategy"
        with pytest.raises(
            ValueError, match="must resolve to a DistillationStrategy subclass"
        ):
            DistillationStrategy.from_spec_dict(spec, models=_make_models())

    def test_from_spec_dict_builds_the_subclass_the_spec_names(self) -> None:
        """A spec naming a subclass rebuilds that subclass, not the base one."""
        spec = _make_strategy().to_spec_dict()
        spec["strategy_cls"] = _TOY_STRATEGY_PATH

        rebuilt = DistillationStrategy.from_spec_dict(spec, models=_make_models())

        assert type(rebuilt) is _ToyDistillationStrategy
        assert rebuilt.to_checkpoint_dict()["strategy_cls"] == _TOY_STRATEGY_PATH
        assert rebuilt.to_spec_dict()["strategy_cls"] == _TOY_STRATEGY_PATH

    def test_a_supplied_loop_survives_the_subclass_dispatch(
        self, tmp_path: Path
    ) -> None:
        """Runtime overrides reach the subclass, so the caller's loop still runs."""
        models = _make_models()
        seed_store = tmp_path / "seeds.zarr"
        label_dataset(
            InMemoryDataset(in_memory_batch=_build_batch(seed=3)),
            InProcessTeacherScorer(models["teacher"], ("energy", "forces")),
            seed_store,
            batch_size=2,
        )
        loops = [
            OnPolicyConfig.from_spec_dict(
                _make_on_policy_recipe(seed_store, segment_steps),
                student=models["student"],
                teacher=models["teacher"],
            )
            for segment_steps in (3, 9)
        ]
        spec = _make_strategy(
            models=models,
            loss_fn=EnergyMSELoss(target_key="teacher_energy")
            + ForceMSELoss(target_key="teacher_forces", normalize_by_atom_count=True),
            on_policy=loops[0],
        ).to_spec_dict()
        spec["strategy_cls"] = _TOY_STRATEGY_PATH

        rebuilt = DistillationStrategy.from_spec_dict(
            spec, models=models, on_policy=loops[1]
        )

        assert type(rebuilt) is _ToyDistillationStrategy
        assert rebuilt.on_policy is loops[1]
        assert rebuilt.on_policy.segment_steps == 9

    def test_an_unimportable_strategy_class_is_refused(self) -> None:
        """A class path that does not import is a spec error, not a traceback."""
        spec = _make_strategy().to_spec_dict()
        spec["strategy_cls"] = "no_such_module.NoSuchStrategy"
        with pytest.raises(ValueError, match="could not be imported"):
            DistillationStrategy.from_spec_dict(spec, models=_make_models())

    def test_runtime_overrides_survive_the_subclass_dispatch(self) -> None:
        """Every override reaches the subclass, so none is lost to the recipe."""
        spec = _make_strategy().to_spec_dict()
        spec["strategy_cls"] = _TOY_STRATEGY_PATH
        hook = _RecordingLossHook()

        rebuilt = DistillationStrategy.from_spec_dict(
            spec, models=_make_models(), hooks=[hook]
        )

        assert type(rebuilt) is _ToyDistillationStrategy
        assert hook in rebuilt.hooks
        assert _labeling_hook_count(rebuilt) == 1

    def test_base_from_spec_dict_ignores_the_strategy_class(self) -> None:
        """The base class does not dispatch on ``strategy_cls``, which this pins."""
        spec = _make_strategy().to_spec_dict()
        rebuilt = TrainingStrategy.from_spec_dict(spec, models=_make_models())
        assert type(rebuilt) is TrainingStrategy
        assert _labeling_hook_count(rebuilt) == 0

    def test_rebuilding_with_live_hooks_keeps_one_labeling_hook(self) -> None:
        """Handing a live strategy's hooks to a rebuild does not duplicate the seam."""
        live = _make_strategy(hooks=[_RecordingLossHook()])
        spec = json.loads(json.dumps(live.to_spec_dict()))
        rebuilt = DistillationStrategy.from_spec_dict(
            spec, models=_make_models(), hooks=live.hooks
        )
        assert _labeling_hook_count(rebuilt) == 1

    def test_checkpoint_restore_with_live_hooks_keeps_one_labeling_hook(
        self, tmp_path: Path
    ) -> None:
        """A restart that carries the running strategy's hooks keeps one seam too."""
        strategy = _make_strategy()
        strategy.save_checkpoint(tmp_path)
        restored = DistillationStrategy.load_checkpoint(
            tmp_path, map_location="cpu", hooks=strategy.hooks
        )
        assert _labeling_hook_count(restored) == 1

    def test_the_labeling_seam_stays_ahead_of_carried_hooks(self) -> None:
        """The fresh seam is prepended, so a carried caller hook still sees labels."""
        live = _make_strategy()
        recorder = _RecordingLabelHook(_TEACHER_FIELDS)
        spec = json.loads(json.dumps(live.to_spec_dict()))
        rebuilt = DistillationStrategy.from_spec_dict(
            spec, models=_make_models(), hooks=[recorder, *live.hooks]
        )
        assert isinstance(rebuilt.hooks[0], _TeacherLabelHook)
        rebuilt.train_batch(_build_batch())
        assert set(recorder.seen[0]) == set(_TEACHER_FIELDS)

    def test_repeated_rebuilds_do_not_accumulate_labeling_hooks(self) -> None:
        """Three chained spec rebuilds still leave exactly one labeling hook."""
        strategy = _make_strategy()
        for _ in range(3):
            spec = json.loads(json.dumps(strategy.to_spec_dict()))
            strategy = DistillationStrategy.from_spec_dict(
                spec, models=_make_models(), hooks=strategy.hooks
            )
        assert _labeling_hook_count(strategy) == 1

    def test_checkpoint_restores_student_weights(self, tmp_path: Path) -> None:
        """The student's trained weights survive the checkpoint round-trip."""
        strategy = _make_strategy(num_steps=4)
        strategy.run(_make_loader(2))
        strategy.save_checkpoint(tmp_path)

        restored = DistillationStrategy.load_checkpoint(tmp_path, map_location="cpu")
        for parameter, expected in zip(
            restored.models["student"].parameters(),
            strategy.models["student"].parameters(),
            strict=True,
        ):
            torch.testing.assert_close(parameter, expected)


class TestDefaultDistillationFn:
    """The stock student-forward training function."""

    def test_student_outputs_are_prefixed(
        self, small_batch: Batch, direct_force_teacher: _DirectForceTeacher
    ) -> None:
        """Every non-``None`` student output is exposed as ``predicted_<key>``."""
        predictions = default_distillation_fn(
            {"student": direct_force_teacher}, small_batch
        )
        assert set(predictions) == {
            "predicted_energy",
            "predicted_forces",
            "predicted_atomic_energies",
        }

    def test_predictions_stay_attached_to_the_student_graph(
        self, small_batch: Batch, direct_force_teacher: _DirectForceTeacher
    ) -> None:
        """Gradients can flow from the returned predictions into the student."""
        predictions = default_distillation_fn(
            {"student": direct_force_teacher}, small_batch
        )
        assert predictions["predicted_energy"].requires_grad

    def test_declared_but_uncomputed_outputs_are_omitted(
        self, small_batch: Batch
    ) -> None:
        """A declared output the student left unset never reaches the predictions."""
        predictions = default_distillation_fn(
            {"student": _PartialOutputStudent()}, small_batch
        )
        assert set(predictions) == {"predicted_energy"}
