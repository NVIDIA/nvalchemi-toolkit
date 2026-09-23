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
"""Tests for the on-policy segment loop of :class:`DistillationStrategy`."""

from __future__ import annotations

import itertools
import json
import time
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
import torch

from nvalchemi.data import AtomicData, Batch
from nvalchemi.data.datapipes.backends.zarr import AtomicDataZarrReader
from nvalchemi.data.datapipes.dataset import Dataset
from nvalchemi.data.datapipes.in_memory_dataset import InMemoryDataset
from nvalchemi.data.datapipes.multidataset import MultiDataset
from nvalchemi.dynamics.base import ConvergenceHook, DynamicsStage
from nvalchemi.dynamics.integrators.nvt_langevin import NVTLangevin
from nvalchemi.dynamics.sinks import GPUBuffer, HostMemory
from nvalchemi.hooks import TrainContext
from nvalchemi.models.base import BaseModelMixin
from nvalchemi.models.demo import DemoModel, DemoModelWrapper
from nvalchemi.models.pipeline import PipelineGroup, PipelineModelWrapper
from nvalchemi.training import (
    EnergyMSELoss,
    ForceMSELoss,
    OptimizerConfig,
    TrainingStage,
    ValidationConfig,
)
from nvalchemi.training.distillation import (
    DistillationStrategy,
    InitialStructures,
    InProcessTeacherScorer,
    OnPolicyConfig,
    ResizableSink,
    TeacherLabelHook,
    label_dataset,
)
from nvalchemi.training.distillation._attach import _attach_teacher_labels
from nvalchemi.training.distillation.scoring import TeacherLabels
from nvalchemi.training.distillation.strategy import _to_device
from test.training.conftest import _build_demo_model
from test.training.distillation.conftest import (
    _ATOMS_PER_SYSTEM,
    _INITIAL_ELEMENT,
    _REFERENCE_ELEMENT,
    _build_direct_force_teacher,
    _build_initial_dataset,
    _build_lj_teacher,
    _build_propagator_batch,
    _build_propagator_system,
    _build_reference_dataset,
    _ListSource,
)

_SUPPLIED_FIELD = "teacher_scaled_energy"
"""Teacher field only the propagator's own scorer writes, read as a loss target."""

_DECLARED_FIELDS = ("teacher_energy", "teacher_forces", _SUPPLIED_FIELD)
"""``label_fields`` the custom generation scorer publishes."""

_RAGGED_SIZES = (3, 7, 2, 11, 5)
"""Atom counts of the placement probe's graphs, distinct so its pointers are too."""

_PLACEMENT_TRIALS = 32
"""Placements per host-move test, enough for an unsynchronized copy to surface."""

_LANGEVIN_KWARGS: dict[str, Any] = {
    "dt": 0.5,
    "temperature": 300.0,
    "friction": 0.01,
    "random_seed": 7,
}
"""Thermostat settings shared by every propagator built here."""


def _make_ragged_batch() -> Batch:
    """Return a batch whose graphs hold distinct atom counts."""
    generator = torch.Generator().manual_seed(11)
    return Batch.from_data_list(
        [
            AtomicData(
                positions=torch.randn(size, 3, generator=generator),
                atomic_numbers=torch.full((size,), _INITIAL_ELEMENT, dtype=torch.long),
                atomic_masses=torch.ones(size),
            )
            for size in _RAGGED_SIZES
        ]
    )


def _make_predicted_reference_dataset(
    scorer: InProcessTeacherScorer, n_systems: int = 8, base_seed: int = 700
) -> InMemoryDataset:
    """Return a reference dataset that keeps the reference ``energy`` and ``forces``.

    This is the shape :func:`label_dataset` leaves an existing reference set in,
    and the one a run graduating from offline distillation reaches for.
    """
    frames = _build_propagator_batch(
        _REFERENCE_ELEMENT, n_systems, base_seed, predictions=True
    )
    _attach_teacher_labels(frames, scorer.label(frames))
    return InMemoryDataset(in_memory_batch=frames)


def _make_statused_initial_dataset(
    status: int, n_systems: int = 4, base_seed: int = 500
) -> InMemoryDataset:
    """Return structures carrying the ``status`` a previous run graduated them at."""
    frames = _build_propagator_batch(_INITIAL_ELEMENT, n_systems, base_seed)
    frames.add_key(
        "status",
        [torch.full((1, 1), status, dtype=torch.long) for _ in range(n_systems)],
        level="system",
    )
    return InMemoryDataset(in_memory_batch=frames)


def _make_scorer(teacher: BaseModelMixin) -> InProcessTeacherScorer:
    """Return an energy-and-forces scorer over *teacher*."""
    return InProcessTeacherScorer(teacher, ("energy", "forces"))


def _make_loss() -> Any:
    """Return the energy-plus-forces teacher objective the loop trains on."""
    return EnergyMSELoss(target_key="teacher_energy") + ForceMSELoss(
        target_key="teacher_forces", normalize_by_atom_count=True
    )


def _make_supplied_loss() -> Any:
    """Return that objective widened by one custom teacher target."""
    return _make_loss() + EnergyMSELoss(target_key=_SUPPLIED_FIELD)


def _make_optimizer_configs() -> dict[str, list[OptimizerConfig]]:
    """Return the Adam configuration the student is optimized with."""
    return {
        "student": [
            OptimizerConfig(
                optimizer_cls=torch.optim.Adam, optimizer_kwargs={"lr": 1e-2}
            )
        ]
    }


def _make_on_policy_strategy(
    *,
    student: BaseModelMixin | None = None,
    teacher: BaseModelMixin | None = None,
    num_steps: int = 12,
    training_steps_per_segment: int = 4,
    generation_steps: int = 3,
    label_frequency: int = 1,
    replay_ratio: float = 0.5,
    batch_size: int = 4,
    device: str = "cpu",
    config_overrides: dict[str, Any] | None = None,
    **overrides: Any,
) -> DistillationStrategy:
    """Return a runnable on-policy strategy over independently seeded demo models."""
    student = _build_demo_model() if student is None else student
    teacher = _build_direct_force_teacher(seed=2) if teacher is None else teacher
    scorer = _make_scorer(teacher)
    config_kwargs: dict[str, Any] = {
        "dynamics": NVTLangevin(student, **_LANGEVIN_KWARGS),
        "teacher_scorer": scorer,
        "initial_structures": InitialStructures(_build_initial_dataset()),
        "replay_ratio": replay_ratio,
        "training_steps_per_segment": training_steps_per_segment,
        "batch_size": batch_size,
        "generation_steps": generation_steps,
        "label_frequency": label_frequency,
    }
    config_kwargs.update(config_overrides or {})
    kwargs: dict[str, Any] = {
        "models": {"student": student, "teacher": teacher},
        "optimizer_configs": _make_optimizer_configs(),
        "loss_fn": _make_loss(),
        "num_steps": num_steps,
        "devices": [torch.device(device)],
        "reference_dataset": None
        if replay_ratio == 1.0
        else _build_reference_dataset(scorer),
        "on_policy": OnPolicyConfig(**config_kwargs),
    }
    kwargs.update(overrides)
    return DistillationStrategy(**kwargs)


def _make_recording_student() -> _ModeRecordingStudent:
    """Return a student that traces the mode of every forward pass it runs."""
    torch.manual_seed(0)
    return _ModeRecordingStudent(DemoModel(num_atom_types=20, hidden_dim=8))


def _make_composed_propagator(
    student: BaseModelMixin, correction: BaseModelMixin
) -> _RecordingPipeline:
    """Return a shared-autograd composition of *student* and *correction*.

    The composition differentiates its summed energy itself, so its own mode —
    not the student's — decides whether generation builds a second-order graph.
    """
    return _RecordingPipeline(
        groups=[PipelineGroup(steps=[student, correction], use_autograd=True)]
    )


def _make_labeled_store(store: Path, scorer: InProcessTeacherScorer) -> Dataset:
    """Return the documented reference dataset: a labeled store opened device-less."""
    label_dataset(
        InMemoryDataset(
            in_memory_batch=_build_propagator_batch(
                _REFERENCE_ELEMENT, 8, base_seed=700, predictions=False
            )
        ),
        scorer,
        store,
        batch_size=4,
    )
    return Dataset(reader=AtomicDataZarrReader(store))


def _seeded_reference_draws(seed: int) -> list[list[float]]:
    """Return the reference frames every batch drew, for a run mixing at *seed*."""
    teacher = _build_direct_force_teacher(seed=2)
    recorder = _RecordingBatchHook()
    strategy = _make_on_policy_strategy(
        teacher=teacher,
        num_steps=8,
        hooks=[recorder],
        reference_dataset=_build_reference_dataset(_make_scorer(teacher), 32),
        config_overrides={"seed": seed},
    )

    strategy.run()

    return recorder.reference_draws


def _labeled_steps(strategy: DistillationStrategy) -> list[int]:
    """Return the propagator steps *strategy*'s run actually paid a teacher pass for.

    The hook's private entry point is wrapped rather than the scorer, because
    the scorer alone cannot say which step a pass was made for; the scorer is
    spied on alongside it to tell a pass from a dispatch the hook passed over.
    """
    steps: list[int] = []
    label_frame = TeacherLabelHook._label_frame
    scorer = strategy.on_policy.teacher_scorer

    with patch.object(scorer, "label", wraps=scorer.label) as spy:

        def recording(
            hook: TeacherLabelHook, batch: Batch, step_count: int, **kwargs: Any
        ) -> None:
            """Record *step_count* when the wrapped call reaches the teacher."""
            before = spy.call_count
            label_frame(hook, batch, step_count, **kwargs)
            if spy.call_count > before:
                steps.append(step_count)

        with patch.object(TeacherLabelHook, "_label_frame", recording):
            strategy.run()
    return steps


def _graph_tags(batch: Batch) -> list[int]:
    """Return the atomic number tagging each graph of *batch*."""
    return [
        int(batch.atomic_numbers[batch.batch_idx == index][0])
        for index in range(batch.num_graphs)
    ]


def _reference_draw(batch: Batch) -> list[float]:
    """Return a fingerprint of the reference frames *batch* drew, order-free."""
    return sorted(
        round(float(batch.teacher_energy[index]), 6)
        for index, tag in enumerate(_graph_tags(batch))
        if tag == _REFERENCE_ELEMENT
    )


class _EmptyReferenceDataset(InMemoryDataset):
    """Reference dataset reporting no samples, as a filtered-down store does."""

    def __len__(self) -> int:
        """Report the dataset as empty whatever batch it was built around."""
        return 0

    def load_batches(self, *args: Any, **kwargs: Any) -> list[Batch]:  # noqa: ARG002
        """Fail the way indexing an empty dataset does, which construction pre-empts."""
        raise IndexError("index 0 is out of bounds for dimension 0 with size 0")


class _CustomFieldScorer:
    """Scorer writing one custom ``teacher_*`` field beside the built-in ones."""

    def __init__(
        self,
        teacher: BaseModelMixin,
        label_fields: tuple[str, ...] | None = _DECLARED_FIELDS,
    ) -> None:
        """Score energies and forces under a signal name of its own."""
        self._inner = InProcessTeacherScorer(teacher, ("energy", "forces"))
        self.signals = frozenset({"energy", "forces", "scaled_energy"})
        self.label_fields = label_fields

    def label(self, batch: Batch) -> TeacherLabels:
        """Return the built-in labels plus a rescaled copy of the teacher energy."""
        labels = self._inner.label(batch)
        energy, level = labels["teacher_energy"]
        labels[_SUPPLIED_FIELD] = (energy * 2.0, level)
        return labels


class _RecordingSink(HostMemory):
    """Host-memory sink recording every capacity it is resized to."""

    def __init__(self, capacity: int) -> None:
        super().__init__(capacity)
        self.resizes: list[int] = []

    def resize(self, capacity: int) -> None:
        """Grow to *capacity* and record the request."""
        self.resizes.append(capacity)
        self._capacity = capacity


class _ModeRecordingStudent(DemoModelWrapper):
    """Demo student recording the mode and force graph of every forward pass."""

    def __init__(self, model: DemoModel) -> None:
        """Wrap *model* and start with an empty trace."""
        super().__init__(model)
        self.forwards: list[tuple[bool, bool]] = []

    def forward(self, data: AtomicData | Batch, **kwargs: Any) -> Any:
        """Record ``(training mode, forces carry a graph)`` and return the outputs."""
        outputs = super().forward(data, **kwargs)
        forces = outputs.get("forces")
        self.forwards.append(
            (self.training, forces is not None and forces.grad_fn is not None)
        )
        return outputs


class _RecordingPipeline(PipelineModelWrapper):
    """Composed propagator model recording its placements and every pass's mode."""

    def __init__(self, **kwargs: Any) -> None:
        """Compose the given groups and start with an empty trace."""
        super().__init__(**kwargs)
        self.forwards: list[tuple[bool, bool]] = []
        self.moves: list[torch.device] = []

    def to(self, *args: Any, **kwargs: Any) -> _RecordingPipeline:
        """Record the device a placement names and move the composition."""
        device = kwargs.get("device", args[0] if args else None)
        if device is not None:
            self.moves.append(torch.device(device))
        return super().to(*args, **kwargs)

    def forward(self, data: AtomicData | Batch, **kwargs: Any) -> Any:
        """Record ``(training mode, forces carry a graph)`` and return the outputs."""
        outputs = super().forward(data, **kwargs)
        forces = outputs.get("forces")
        self.forwards.append(
            (self.training, forces is not None and forces.grad_fn is not None)
        )
        return outputs


class _RecordingBatchHook:
    """Record the loss, composition, and fields of every training batch."""

    frequency = 1
    stage = TrainingStage.AFTER_BATCH

    def __init__(self) -> None:
        """Start with an empty trace."""
        self.losses: list[float] = []
        self.tags: list[list[int]] = []
        self.reference_draws: list[list[float]] = []
        self.predictions: list[list[str]] = []
        self.epoch_steps: list[int] = []

    def __call__(self, ctx: TrainContext, stage: TrainingStage) -> None:  # noqa: ARG002
        """Append the loss and composition of the batch just trained on."""
        self.losses.append(float(ctx.loss))
        self.tags.append(_graph_tags(ctx.batch))
        self.reference_draws.append(_reference_draw(ctx.batch))
        self.predictions.append(
            [key for key in ("energy", "forces") if key in ctx.batch]
        )
        self.epoch_steps.append(ctx.epoch_step_count)


class _EpochStartHook:
    """Record the counters each training epoch opens on."""

    frequency = 1
    stage = TrainingStage.BEFORE_EPOCH

    def __init__(self) -> None:
        """Start with an empty trace."""
        self.calls: list[tuple[int, int]] = []

    def __call__(self, ctx: TrainContext, stage: TrainingStage) -> None:  # noqa: ARG002
        """Append the step and epoch this epoch opened on."""
        self.calls.append((ctx.step_count, ctx.epoch))


class _PhaseProbe:
    """Take one reading each time *stage* fires, to separate the loop's phases."""

    def __init__(self, stage: Any, reading: Callable[[], Any]) -> None:
        """Start with an empty trace of *reading* at *stage*."""
        self.stage = stage
        self.frequency = 1
        self.reading = reading
        self.readings: list[Any] = []

    def __call__(self, ctx: Any, stage: Any) -> None:  # noqa: ARG002
        """Append the reading taken at this stage."""
        self.readings.append(self.reading())


class _ExplodingHook:
    """Raise from inside a training segment, on the second batch."""

    frequency = 1
    stage = TrainingStage.AFTER_BATCH

    def __init__(self) -> None:
        """Start counting batches."""
        self.calls = 0

    def __call__(self, ctx: TrainContext, stage: TrainingStage) -> None:  # noqa: ARG002
        """Fail once the segment loop is well underway."""
        self.calls += 1
        if self.calls == 2:
            raise RuntimeError("boom")


class _FixedWorldManager:
    """Distributed manager reporting a fixed world size and rank."""

    def __init__(self, world_size: int) -> None:
        """Report *world_size* ranks, always as rank zero."""
        self.world_size = world_size
        self.rank = 0


class _RecordingValidationHook:
    """Record the step and epoch each validation pass closes on."""

    frequency = 1
    stage = TrainingStage.AFTER_VALIDATION

    def __init__(self) -> None:
        """Start with an empty schedule."""
        self.calls: list[tuple[int, int]] = []

    def __call__(self, ctx: TrainContext, stage: TrainingStage) -> None:  # noqa: ARG002
        """Append the counters the strategy holds at this validation."""
        self.calls.append((ctx.workflow.step_count, ctx.workflow.epoch_count))


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


class TestOnPolicySegmentLoop:
    def test_three_segments_train_on_labeled_generated_frames(
        self, device: str
    ) -> None:
        """A full run reaches its step target on generated, teacher-labeled data."""
        recorder = _RecordingBatchHook()
        strategy = _make_on_policy_strategy(device=device, hooks=[recorder])

        strategy.run()

        assert strategy.step_count == 12
        assert strategy.epoch_count == 3
        assert len(recorder.losses) == 12
        assert strategy.on_policy.dynamics.step_count == 9

    def test_generated_frames_land_in_the_buffer_with_the_label_schema(self) -> None:
        """Every propagated step is labeled and stored under one frozen schema."""
        strategy = _make_on_policy_strategy()

        strategy.run()

        buffer = strategy.replay_buffer
        assert buffer is not None
        assert "node.teacher_forces" in buffer.schema
        assert "system.teacher_energy" in buffer.schema
        assert len(buffer) == 3 * 3 * len(strategy.on_policy.initial_structures.dataset)

    def test_every_batch_holds_the_configured_mixture(self) -> None:
        """A replay ratio of one half puts two generated frames in a batch of four."""
        recorder = _RecordingBatchHook()
        strategy = _make_on_policy_strategy(hooks=[recorder])

        strategy.run()

        for tags in recorder.tags:
            assert len(tags) == 4
            assert tags.count(_INITIAL_ELEMENT) == 2
            assert tags.count(_REFERENCE_ELEMENT) == 2

    def test_loss_falls_across_the_segments(self) -> None:
        """Twelve steps against the teacher's energies and forces lower the loss."""
        recorder = _RecordingBatchHook()
        strategy = _make_on_policy_strategy(hooks=[recorder])

        strategy.run()

        assert sum(recorder.losses[-4:]) < sum(recorder.losses[:4])

    def test_the_student_generates_from_its_updated_weights(self) -> None:
        """The propagator holds the trained module, so generation follows training."""
        strategy = _make_on_policy_strategy()
        before = strategy.models["student"].model.projection.weight.detach().clone()

        strategy.run()

        assert strategy.on_policy.dynamics.model is strategy.models["student"]
        assert not torch.allclose(
            strategy.models["student"].model.projection.weight, before
        )

    def test_replay_only_runs_need_no_reference_dataset(self) -> None:
        """A ratio of one draws every sample from the buffer the run itself filled."""
        recorder = _RecordingBatchHook()
        strategy = _make_on_policy_strategy(
            replay_ratio=1.0, num_steps=6, hooks=[recorder]
        )

        strategy.run()

        assert strategy.step_count == 6
        assert all(set(tags) == {_INITIAL_ELEMENT} for tags in recorder.tags)

    def test_labeling_hook_is_removed_from_the_caller_propagator(self) -> None:
        """The loop leaves the propagator as it found it, so a rerun labels once."""
        strategy = _make_on_policy_strategy(num_steps=4)

        strategy.run()

        assert strategy.on_policy.dynamics.hooks == []

    def test_teacher_weights_survive_the_run(self) -> None:
        """Generation and training both leave the frozen teacher untouched."""
        strategy = _make_on_policy_strategy(num_steps=4)
        before = [
            parameter.detach().clone()
            for parameter in strategy.models["teacher"].parameters()
        ]

        strategy.run()

        for parameter, snapshot in zip(
            strategy.models["teacher"].parameters(), before, strict=True
        ):
            torch.testing.assert_close(parameter, snapshot)


class TestOnPolicyModelModes:
    def test_the_student_generates_in_eval_mode_and_trains_in_training_mode(
        self,
    ) -> None:
        """Generation costs no dropout, no moving statistics, and no second-order graph."""
        student = _make_recording_student()
        generation = _PhaseProbe(DynamicsStage.AFTER_STEP, lambda: student.forwards[-1])
        training = _PhaseProbe(TrainingStage.AFTER_BATCH, lambda: student.forwards[-1])
        strategy = _make_on_policy_strategy(
            student=student, num_steps=8, hooks=[training]
        )
        strategy.on_policy.dynamics.register_hook(generation)

        strategy.run()

        assert len(generation.readings) == 6
        assert all(reading == (False, False) for reading in generation.readings)
        assert len(training.readings) == 8
        assert all(reading == (True, True) for reading in training.readings)

    def test_the_caller_gets_its_student_back_in_the_mode_it_handed_over(self) -> None:
        """Both phases restore what they found, so the run leaves no mode behind."""
        strategy = _make_on_policy_strategy(num_steps=4)
        strategy.models["student"].eval()

        strategy.run()

        assert strategy.models["student"].training is False

    def test_the_teacher_is_frozen_and_in_eval_mode_across_both_phases(self) -> None:
        """The frozen-by-omission contract holds while the student generates too."""
        strategy = _make_on_policy_strategy(num_steps=4)
        teacher = strategy.models["teacher"]

        def reading() -> tuple[bool, bool]:
            """Return the teacher's mode and whether every parameter is frozen."""
            return (
                teacher.training,
                all(not parameter.requires_grad for parameter in teacher.parameters()),
            )

        generation = _PhaseProbe(DynamicsStage.AFTER_STEP, reading)
        training = _PhaseProbe(TrainingStage.AFTER_BATCH, reading)
        strategy.hooks.append(training)
        strategy.on_policy.dynamics.register_hook(generation)

        strategy.run()

        assert all(seen == (False, True) for seen in generation.readings)
        assert all(seen == (False, True) for seen in training.readings)
        assert all(parameter.requires_grad for parameter in teacher.parameters())


class TestOnPolicyComposedPropagator:
    def test_a_composed_propagator_generates_in_eval_mode(self) -> None:
        """A composition holding the student is no entry of ``models``, so the loop evals it."""
        student = _build_demo_model()
        correction = _build_demo_model()
        composed = _make_composed_propagator(student, correction)
        generation = _PhaseProbe(
            DynamicsStage.AFTER_STEP,
            lambda: (composed.training, correction.training, composed.forwards[-1][1]),
        )
        strategy = _make_on_policy_strategy(
            student=student,
            num_steps=4,
            config_overrides={"dynamics": NVTLangevin(composed, **_LANGEVIN_KWARGS)},
        )
        strategy.on_policy.dynamics.register_hook(generation)

        strategy.run()

        assert len(generation.readings) == 3
        assert all(reading == (False, False, False) for reading in generation.readings)

    def test_the_caller_gets_the_composition_back_in_the_mode_it_handed_over(
        self,
    ) -> None:
        """Restoring the composition sets the student's mode too, so it goes back first."""
        student = _build_demo_model()
        correction = _build_demo_model()
        composed = _make_composed_propagator(student, correction)
        strategy = _make_on_policy_strategy(
            student=student,
            num_steps=4,
            config_overrides={"dynamics": NVTLangevin(composed, **_LANGEVIN_KWARGS)},
        )
        strategy.models["student"].eval()

        strategy.run()

        assert composed.training is True
        assert correction.training is True
        assert student.training is False

    def test_the_composition_is_placed_on_the_generation_device(self) -> None:
        """The loop moves the composition itself, not only the student it names."""
        student = _build_demo_model()
        composed = _make_composed_propagator(student, _build_demo_model())
        strategy = _make_on_policy_strategy(
            student=student,
            num_steps=1,
            training_steps_per_segment=1,
            generation_steps=1,
            config_overrides={"dynamics": NVTLangevin(composed, **_LANGEVIN_KWARGS)},
        )

        strategy.run()

        assert torch.device("cpu") in composed.moves

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_a_correction_only_the_composition_holds_follows_the_student(
        self,
    ) -> None:
        """A CPU-built correction lands on the generation device before the first step."""
        student = _build_demo_model()
        correction = _build_demo_model()
        composed = _make_composed_propagator(student, correction)
        strategy = _make_on_policy_strategy(
            student=student,
            num_steps=1,
            training_steps_per_segment=1,
            generation_steps=1,
            replay_ratio=1.0,
            device="cuda:0",
            config_overrides={"dynamics": NVTLangevin(composed, **_LANGEVIN_KWARGS)},
        )

        strategy.run()

        assert strategy.step_count == 1
        assert next(correction.parameters()).device == torch.device("cuda:0")

    def test_a_submodule_the_caller_froze_alone_keeps_its_own_mode(self) -> None:
        """Restoring the composition's single flag would unfreeze a frozen head."""
        student = _build_demo_model()
        correction = _build_demo_model()
        composed = _make_composed_propagator(student, correction)
        correction.eval()
        strategy = _make_on_policy_strategy(
            student=student,
            num_steps=4,
            config_overrides={"dynamics": NVTLangevin(composed, **_LANGEVIN_KWARGS)},
        )

        strategy.run()

        assert correction.training is False
        assert composed.training is True
        assert student.training is True


class TestOnPolicySeeding:
    def test_a_structure_carrying_a_stale_status_still_moves(self) -> None:
        """A status a previous run graduated the structures at must not freeze them."""
        strategy = _make_on_policy_strategy(
            num_steps=4,
            replay_ratio=1.0,
            config_overrides={
                "initial_structures": InitialStructures(
                    _make_statused_initial_dataset(status=1)
                )
            },
        )
        initial = _build_propagator_batch(_INITIAL_ELEMENT, 4, base_seed=500)

        strategy.run()

        stored = strategy.replay_buffer.dataset.in_memory_batch
        assert len(strategy.replay_buffer) > 0
        assert not torch.allclose(
            stored.positions[: initial.num_nodes], initial.positions
        )

    def test_structures_carrying_no_model_outputs_run_a_segment(self) -> None:
        """The propagator primes its forces, so an initial structure need not carry any."""
        frames = _build_propagator_batch(
            _INITIAL_ELEMENT, 4, base_seed=500, predictions=False
        )
        strategy = _make_on_policy_strategy(
            num_steps=4,
            replay_ratio=1.0,
            config_overrides={
                "initial_structures": InitialStructures(
                    InMemoryDataset(in_memory_batch=frames)
                )
            },
        )

        strategy.run()

        assert "forces" not in frames
        assert len(strategy.replay_buffer) == 3 * 4

    def test_a_rerun_reopens_the_structure_cursor(self) -> None:
        """A second run reseeds the trajectory, so the shard rewinds with it."""
        strategy = _make_on_policy_strategy(num_steps=4, training_steps_per_segment=4)
        structures = strategy.on_policy.initial_structures

        strategy.run()
        strategy.num_steps = 8
        strategy.run()

        assert structures.next_system_id == len(structures)


class TestOnPolicyMixtureSeed:
    def test_two_seeds_draw_different_reference_frames(self) -> None:
        """The mixture seed is a setting, so replicate runs can be made independent."""
        assert _seeded_reference_draws(0) != _seeded_reference_draws(17)

    def test_one_seed_reproduces_the_reference_draw(self) -> None:
        """The setting is a seed rather than a fresh source of noise."""
        assert _seeded_reference_draws(17) == _seeded_reference_draws(17)


class TestOnPolicyMixtureSchema:
    def test_generated_frames_carry_no_student_predictions(self) -> None:
        """A replay frame is a training sample, so the student's own outputs are gone."""
        strategy = _make_on_policy_strategy(num_steps=4)

        strategy.run()

        schema = strategy.replay_buffer.schema
        assert "node.teacher_forces" in schema
        assert "system.teacher_energy" in schema
        assert "node.forces" not in schema
        assert "system.energy" not in schema

    def test_no_training_batch_carries_a_reference_target_key(self) -> None:
        """Mixing keeps only shared fields, so a batch offers teacher targets only."""
        recorder = _RecordingBatchHook()
        strategy = _make_on_policy_strategy(num_steps=4, hooks=[recorder])

        strategy.run()

        assert recorder.predictions == [[]] * len(recorder.predictions)

    def test_consecutive_segments_draw_different_reference_frames(self) -> None:
        """Each segment's fresh sampler advances its epoch instead of replaying one draw."""
        teacher = _build_direct_force_teacher(seed=2)
        recorder = _RecordingBatchHook()
        strategy = _make_on_policy_strategy(
            teacher=teacher,
            num_steps=8,
            hooks=[recorder],
            reference_dataset=_build_reference_dataset(_make_scorer(teacher), 32),
        )

        strategy.run()

        assert strategy.epoch_count == 2
        assert recorder.reference_draws[:4] != recorder.reference_draws[4:]


class TestOnPolicyMixtureDevice:
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_a_gpu_loaded_reference_dataset_runs_the_mixed_path(
        self, tmp_path: Path
    ) -> None:
        """A Zarr store resolves to CUDA, and generated frames are staged there too."""
        teacher = _build_direct_force_teacher(seed=2)
        strategy = _make_on_policy_strategy(
            teacher=teacher,
            num_steps=4,
            device="cuda",
            reference_dataset=_make_labeled_store(
                tmp_path / "reference.zarr", _make_scorer(teacher)
            ),
        )

        strategy.run()

        assert strategy.step_count == 4
        assert strategy.replay_buffer.dataset.in_memory_batch.device.type == "cuda"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_a_composed_reference_dataset_runs_the_mixed_path(
        self, tmp_path: Path
    ) -> None:
        """A MultiDataset declares no device, so the buffer follows the one it emits on."""
        teacher = _build_direct_force_teacher(seed=2)
        scorer = _make_scorer(teacher)
        strategy = _make_on_policy_strategy(
            teacher=teacher,
            num_steps=8,
            device="cuda",
            reference_dataset=MultiDataset(
                _make_labeled_store(tmp_path / "first.zarr", scorer),
                _make_labeled_store(tmp_path / "second.zarr", scorer),
            ),
        )

        strategy.run()

        assert strategy.step_count == 8
        assert strategy.epoch_count == 2
        assert strategy.replay_buffer.dataset.in_memory_batch.device.type == "cuda"

    @pytest.mark.multigpu
    def test_a_device_less_store_resolves_to_the_index_it_emits_on(
        self, tmp_path: Path
    ) -> None:
        """An index-less ``cuda`` store is no longer a wildcard a second GPU slips past."""
        teacher = _build_direct_force_teacher(seed=2)
        with pytest.raises(ValueError, match="replay_device=cuda:1"):
            _make_on_policy_strategy(
                teacher=teacher,
                device="cuda",
                reference_dataset=_make_labeled_store(
                    tmp_path / "reference.zarr", _make_scorer(teacher)
                ),
                config_overrides={"replay_device": "cuda:1"},
            )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_a_reference_dataset_on_another_accelerator_is_rejected(self) -> None:
        """The mixture is collated on the reference dataset's device, so the run has to own it."""
        with pytest.raises(ValueError, match="devices\\[0\\]=cpu"):
            _make_on_policy_strategy(
                reference_dataset=InMemoryDataset(
                    in_memory_batch=_build_propagator_batch(
                        _REFERENCE_ELEMENT, 8, base_seed=700, predictions=False
                    ),
                    device="cuda",
                )
            )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_a_cuda_defaulted_reference_dataset_is_rejected_by_a_cpu_run(
        self, tmp_path: Path
    ) -> None:
        """A Zarr reference dataset resolves an unset device to CUDA, which a CPU run names."""
        teacher = _build_direct_force_teacher(seed=2)
        with pytest.raises(ValueError, match="Dataset\\(\\.\\.\\., device='cpu'\\)"):
            _make_on_policy_strategy(
                teacher=teacher,
                reference_dataset=_make_labeled_store(
                    tmp_path / "reference.zarr", _make_scorer(teacher)
                ),
            )

    def test_a_replay_device_off_the_reference_dataset_is_rejected(self) -> None:
        """A mixed batch is collated before training moves it, so both sources agree."""
        with pytest.raises(ValueError, match="replay_device=cuda"):
            _make_on_policy_strategy(config_overrides={"replay_device": "cuda"})

    def test_a_replay_only_run_keeps_its_frames_in_host_memory(self) -> None:
        """With no reference dataset to follow, the sink's device stands."""
        strategy = _make_on_policy_strategy(replay_ratio=1.0, num_steps=4)

        strategy.run()

        assert strategy.replay_buffer.dataset.in_memory_batch.device.type == "cpu"


class TestOnPolicyBatchPlacement:
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

        _to_device(batch, torch.device(destination))

        assert batch.non_blocking is asynchronous

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_a_host_placement_lands_before_its_index_tensors_are_read(self) -> None:
        """Moves onto the host read back the pointers and rows the source batch held."""
        device = torch.device("cuda")
        expected_ptr = [0, *itertools.accumulate(_RAGGED_SIZES)]
        expected_idx = [
            index for index, size in enumerate(_RAGGED_SIZES) for _ in range(size)
        ]
        original = _make_ragged_batch()
        expected_rows = original.positions.index_select(0, torch.tensor(expected_idx))
        source = original.to(device)
        host = torch.device("cpu")

        for _ in range(_PLACEMENT_TRIALS):
            pressure = torch.randn(4096, 4096, device=device)
            pressure @ pressure
            placed = _to_device(source, host)
            assert placed.batch_ptr.tolist() == expected_ptr
            assert placed.batch_idx.tolist() == expected_idx
            torch.testing.assert_close(
                placed.positions.index_select(0, placed.batch_idx.long()),
                expected_rows,
            )


class TestOnPolicySegmentAccounting:
    def test_final_partial_segment_trains_only_the_remainder(self) -> None:
        """A step target that is not a multiple of the segment never overshoots."""
        recorder = _RecordingBatchHook()
        strategy = _make_on_policy_strategy(
            num_steps=7, training_steps_per_segment=3, hooks=[recorder]
        )

        strategy.run()

        assert strategy.step_count == 7
        assert len(recorder.losses) == 7
        assert strategy.epoch_count == 3

    def test_reaching_the_target_returns_without_generating(self) -> None:
        """A run already at its target neither propagates nor trains."""
        strategy = _make_on_policy_strategy(num_steps=4)
        strategy.step_count = 4

        strategy.run()

        assert strategy.on_policy.dynamics.step_count == 0
        assert strategy.replay_buffer is None

    def test_the_last_frame_of_a_segment_is_labeled_off_cadence(self) -> None:
        """A cadence that skips the segment's end still stores its final frame."""
        strategy = _make_on_policy_strategy(
            num_steps=3,
            training_steps_per_segment=1,
            generation_steps=3,
            label_frequency=10,
        )

        strategy.run()

        structures = len(strategy.on_policy.initial_structures.dataset)
        assert strategy.on_policy.dynamics.step_count == 9
        assert len(strategy.replay_buffer) == 4 * structures

    def test_an_early_exiting_propagator_still_produces_a_segment(self) -> None:
        """A chunk that converges out short is read from ``step_count``, not assumed."""
        student = _build_demo_model()
        strategy = _make_on_policy_strategy(
            student=student,
            num_steps=4,
            training_steps_per_segment=2,
            generation_steps=5,
            label_frequency=10,
            config_overrides={
                "dynamics": NVTLangevin(
                    student,
                    convergence_hook=ConvergenceHook.from_fmax(threshold=1e6),
                    **_LANGEVIN_KWARGS,
                )
            },
        )

        strategy.run()

        assert strategy.step_count == 4
        assert strategy.on_policy.dynamics.step_count == 2
        assert len(strategy.replay_buffer) == 2 * len(
            strategy.on_policy.initial_structures.dataset
        )

    def test_a_replay_only_run_segments_like_a_mixed_one(self) -> None:
        """A lone source is oversampled to the segment, not cut short by its length."""
        recorder = _RecordingBatchHook()
        strategy = _make_on_policy_strategy(
            replay_ratio=1.0,
            num_steps=12,
            training_steps_per_segment=4,
            hooks=[recorder],
        )

        strategy.run()

        assert strategy.step_count == 12
        assert strategy.epoch_count == 3
        assert len(recorder.losses) == 12

    def test_a_failure_inside_a_segment_leaves_the_propagator_pristine(self) -> None:
        """The caller's propagator and the student's gradients survive a crash."""
        strategy = _make_on_policy_strategy(num_steps=8, hooks=[_ExplodingHook()])

        with pytest.raises(RuntimeError, match="boom"):
            strategy.run()

        assert strategy.on_policy.dynamics.hooks == []
        student = strategy.models["student"]
        assert all(parameter.requires_grad for parameter in student.parameters())
        assert student.training is True
        assert all(
            parameter.requires_grad
            for parameter in strategy.models["teacher"].parameters()
        )

    def test_capacity_bounds_the_buffer_across_segments(self) -> None:
        """A bounded buffer keeps the newest frames instead of growing forever."""
        strategy = _make_on_policy_strategy(
            num_steps=8, config_overrides={"replay_capacity": 6}
        )

        strategy.run()

        assert len(strategy.replay_buffer) == 6


class TestChunkedPropagatorResume:
    def _run_langevin(self, chunks: tuple[int, ...]) -> tuple[Batch, NVTLangevin]:
        """Return the state and propagator after running *chunks* back to back."""
        state = _build_propagator_batch(_INITIAL_ELEMENT, 3, base_seed=500)
        dynamics = NVTLangevin(_build_demo_model(), **_LANGEVIN_KWARGS)
        for n_steps in chunks:
            state = dynamics.run(state, n_steps=n_steps)
        return state, dynamics

    def test_two_chunks_reproduce_one_run_exactly(self) -> None:
        """Segmenting a trajectory changes nothing: the thermostat RNG is counter-based."""
        one_shot, one_shot_dynamics = self._run_langevin((6,))
        chunked, chunked_dynamics = self._run_langevin((3, 3))

        torch.testing.assert_close(
            chunked.positions, one_shot.positions, rtol=0.0, atol=0.0
        )
        torch.testing.assert_close(
            chunked.velocities, one_shot.velocities, rtol=0.0, atol=0.0
        )
        assert chunked_dynamics.step_count == one_shot_dynamics.step_count == 6

    def test_uneven_chunks_reproduce_one_run_exactly(self) -> None:
        """Resume exactness does not depend on the chunks being the same length."""
        one_shot, _ = self._run_langevin((6,))
        chunked, _ = self._run_langevin((1, 4, 1))

        torch.testing.assert_close(
            chunked.positions, one_shot.positions, rtol=0.0, atol=0.0
        )

    def test_labeling_does_not_perturb_the_trajectory(self) -> None:
        """A teacher pass between steps leaves the propagated state bit-identical."""
        unlabeled, _ = self._run_langevin((3, 3))
        labeled = _build_propagator_batch(_INITIAL_ELEMENT, 3, base_seed=500)
        dynamics = NVTLangevin(_build_demo_model(), **_LANGEVIN_KWARGS)
        dynamics.register_hook(
            TeacherLabelHook(_make_scorer(_build_direct_force_teacher(seed=2)))
        )

        for _ in range(2):
            labeled = dynamics.run(labeled, n_steps=3)

        torch.testing.assert_close(
            labeled.positions, unlabeled.positions, rtol=0.0, atol=0.0
        )


class TestOnPolicyDirectForceTeacher:
    def test_a_direct_force_teacher_trains_a_conservative_student(self) -> None:
        """Forces from a teacher head distill into a student whose forces are a gradient."""
        recorder = _RecordingBatchHook()
        student = _build_demo_model()
        strategy = _make_on_policy_strategy(
            student=student,
            teacher=_build_direct_force_teacher(seed=3),
            hooks=[recorder],
        )

        strategy.run()

        assert strategy.models["teacher"].model_config.autograd_outputs == frozenset()
        assert "forces" in student.model_config.autograd_outputs
        assert strategy.step_count == 12
        assert sum(recorder.losses[-4:]) < sum(recorder.losses[:4])


def _make_validated_strategy(
    recorder: _RecordingValidationHook, **cadence: int
) -> DistillationStrategy:
    """Return an eight-step run validating on unlabeled data at *cadence*."""
    return _make_on_policy_strategy(
        num_steps=8,
        hooks=[recorder],
        validation_config=ValidationConfig(
            validation_data=[
                _build_propagator_batch(_REFERENCE_ELEMENT, 2, base_seed=900)
            ],
            **cadence,
        ),
    )


class TestOnPolicyValidation:
    def test_step_cadence_validation_fires_inside_the_segments(self) -> None:
        """A step cadence lands mid-segment, ahead of the segment boundary."""
        recorder = _RecordingValidationHook()
        strategy = _make_validated_strategy(recorder, every_n_steps=4)

        strategy.run()

        assert strategy.step_count == 8
        assert recorder.calls == [(4, 0), (8, 1)]
        assert strategy.last_validation["total_loss"] > 0.0

    def test_a_terminal_validation_closes_the_run(self) -> None:
        """A cadence that never fires still gets exactly one closing validation."""
        recorder = _RecordingValidationHook()
        strategy = _make_validated_strategy(recorder, every_n_steps=1000)

        strategy.run()

        assert recorder.calls == [(8, 2)]

    def test_the_closing_validation_does_not_repeat_the_cadence(self) -> None:
        """A cadence landing on the final step is not validated a second time."""
        recorder = _RecordingValidationHook()
        strategy = _make_validated_strategy(recorder, every_n_epochs=1)

        strategy.run()

        assert [step for step, _ in recorder.calls].count(strategy.step_count) == 1

    def test_prelabeled_training_batches_are_never_labeled_twice(self) -> None:
        """Generated and reference frames arrive labeled, so the seam skips them all."""
        strategy = _make_on_policy_strategy(num_steps=8)

        with patch.object(
            strategy.teacher_scorer, "label", wraps=strategy.teacher_scorer.label
        ) as spy:
            strategy.run()

        assert strategy.step_count == 8
        assert spy.call_count == 0

    def test_unlabeled_validation_data_is_labeled_by_the_training_seam(self) -> None:
        """The one batch the run cannot pre-label is the validation set."""
        recorder = _RecordingValidationHook()
        strategy = _make_validated_strategy(recorder, every_n_steps=4)

        with patch.object(
            strategy.teacher_scorer, "label", wraps=strategy.teacher_scorer.label
        ) as spy:
            strategy.run()

        assert spy.call_count == len(recorder.calls) == 2

    def test_epoch_cadence_validation_follows_the_segments(self) -> None:
        """One segment is one epoch, so an epoch cadence fires at segment boundaries."""
        recorder = _RecordingValidationHook()
        strategy = _make_validated_strategy(recorder, every_n_epochs=1)

        strategy.run()

        assert strategy.epoch_count == 2
        assert recorder.calls == [(4, 1), (8, 2)]


class TestOnPolicyValidationContract:
    def test_a_propagator_over_another_module_is_rejected(self) -> None:
        """On-policy data requires the propagator to hold the trained student itself."""
        with pytest.raises(ValueError, match="models\\['student'\\]"):
            _make_on_policy_strategy(
                config_overrides={
                    "dynamics": NVTLangevin(_build_demo_model(), **_LANGEVIN_KWARGS)
                }
            )

    def test_epoch_sized_runs_are_rejected(self) -> None:
        """Segments build their own loaders, so there is no epoch to convert."""
        with pytest.raises(ValueError, match="sized in optimizer steps"):
            _make_on_policy_strategy(num_steps=None, num_epochs=2)

    def test_a_partial_ratio_without_a_reference_dataset_is_rejected(self) -> None:
        """Mixing in reference data requires a reference dataset to mix from."""
        with pytest.raises(ValueError, match="reference_dataset is required"):
            _make_on_policy_strategy(reference_dataset=None)

    def test_an_empty_reference_dataset_is_rejected(self) -> None:
        """The mixture's reference share has to come from somewhere."""
        teacher = _build_direct_force_teacher(seed=2)
        reference = _build_reference_dataset(_make_scorer(teacher))

        with pytest.raises(ValueError, match="at least one sample"):
            _make_on_policy_strategy(
                teacher=teacher,
                reference_dataset=_EmptyReferenceDataset(
                    in_memory_batch=reference.in_memory_batch
                ),
            )

    def test_a_full_replay_ratio_alongside_a_reference_dataset_is_rejected(
        self,
    ) -> None:
        """A reference dataset never drawn from is the mirror of a zero ratio."""
        teacher = _build_direct_force_teacher(seed=2)
        with pytest.raises(ValueError, match="Drop reference_dataset"):
            _make_on_policy_strategy(
                teacher=teacher,
                replay_ratio=1.0,
                reference_dataset=_build_reference_dataset(_make_scorer(teacher)),
            )

    def test_a_reference_dataset_carrying_reference_predictions_is_rejected_up_front(
        self,
    ) -> None:
        """A guaranteed mixture failure must not cost a whole generation segment."""
        teacher = _build_direct_force_teacher(seed=2)
        with pytest.raises(ValueError, match="no generated frame can"):
            _make_on_policy_strategy(
                teacher=teacher,
                reference_dataset=_make_predicted_reference_dataset(
                    _make_scorer(teacher)
                ),
            )

    def test_a_multi_rank_launch_without_gradient_sync_is_rejected(self) -> None:
        """An unwrapped student leaves every rank training a policy of its own."""
        strategy = _make_on_policy_strategy(
            num_steps=2, distributed_manager=_FixedWorldManager(world_size=2)
        )

        with pytest.raises(ValueError, match="gradients have to be synchronized"):
            strategy.run()

        assert strategy.step_count == 0

    def test_a_single_rank_launch_runs(self) -> None:
        """The guard reads the world size rather than the presence of a manager."""
        strategy = _make_on_policy_strategy(
            num_steps=2, distributed_manager=_FixedWorldManager(world_size=1)
        )

        strategy.run()

        assert strategy.step_count == 2

    def test_a_reference_dataset_without_on_policy_is_rejected(self) -> None:
        """Offline distillation trains on the dataloader, not on reference_dataset."""
        teacher = _build_direct_force_teacher(seed=2)
        with pytest.raises(ValueError, match="read only by the segment loop"):
            DistillationStrategy(
                models={"student": _build_demo_model(), "teacher": teacher},
                optimizer_configs=_make_optimizer_configs(),
                loss_fn=_make_loss(),
                num_steps=2,
                reference_dataset=_build_reference_dataset(_make_scorer(teacher)),
            )

    def test_a_dataloader_is_rejected_in_on_policy_mode(self) -> None:
        """The segment loop owns its loader, so a caller's would be silently dropped."""
        strategy = _make_on_policy_strategy(num_steps=2)
        with pytest.raises(ValueError, match="builds its own loader"):
            strategy.run([_build_propagator_batch(_INITIAL_ELEMENT, 2, base_seed=800)])

    def test_offline_mode_still_requires_a_dataloader(self) -> None:
        """Without a segment loop there is nothing to train on but the caller's batches."""
        teacher = _build_direct_force_teacher(seed=2)
        strategy = DistillationStrategy(
            models={"student": _build_demo_model(), "teacher": teacher},
            optimizer_configs=_make_optimizer_configs(),
            loss_fn=_make_loss(),
            num_steps=2,
        )
        with pytest.raises(ValueError, match="run\\(dataloader=None\\)"):
            strategy.run()

    def test_a_propagator_over_a_composed_model_is_accepted(self) -> None:
        """A student composed with a correction term is still the module being trained."""
        student = _build_demo_model()
        composed = student + _build_lj_teacher()

        strategy = _make_on_policy_strategy(
            student=student,
            num_steps=2,
            config_overrides={"dynamics": NVTLangevin(composed, **_LANGEVIN_KWARGS)},
        )

        assert strategy.on_policy.dynamics.model is composed

    def test_a_narrower_generation_scorer_warns_on_a_replay_only_run(self) -> None:
        """With no reference dataset the missing signal costs a second teacher pass."""
        teacher = _build_direct_force_teacher(seed=2)
        with pytest.warns(UserWarning, match="scored twice"):
            _make_on_policy_strategy(
                teacher=teacher,
                replay_ratio=1.0,
                config_overrides={
                    "teacher_scorer": InProcessTeacherScorer(teacher, ("energy",))
                },
            )

    def test_a_narrower_generation_scorer_than_the_loss_warns_with_a_reference_dataset(
        self,
    ) -> None:
        """A reference set as narrow as the propagator still relabels every batch."""
        teacher = _build_direct_force_teacher(seed=2)
        narrow = InProcessTeacherScorer(teacher, ("energy",))
        with pytest.warns(UserWarning, match="scored twice"):
            _make_on_policy_strategy(
                teacher=teacher,
                reference_dataset=_build_reference_dataset(narrow),
                config_overrides={"teacher_scorer": narrow},
            )

    def test_a_narrower_generation_scorer_than_the_reference_dataset_is_rejected(
        self,
    ) -> None:
        """Mixing keeps shared fields only, so a narrower scorer is a broken batch."""
        teacher = _build_direct_force_teacher(seed=2)
        with pytest.raises(ValueError, match="same teacher fields"):
            _make_on_policy_strategy(
                teacher=teacher,
                config_overrides={
                    "teacher_scorer": InProcessTeacherScorer(teacher, ("energy",))
                },
            )

    def test_a_wider_generation_scorer_than_the_reference_dataset_is_rejected(
        self,
    ) -> None:
        """The mirror case is a broken batch too, and was silent before the run."""
        teacher = _build_direct_force_teacher(seed=2)
        with pytest.raises(ValueError, match="same teacher fields"):
            _make_on_policy_strategy(
                teacher=teacher,
                reference_dataset=_build_reference_dataset(
                    InProcessTeacherScorer(teacher, ("energy",))
                ),
            )


class TestOnPolicyLabelingCadence:
    def test_a_segment_aligned_cadence_labels_each_segment_once(self) -> None:
        """A cadence landing beside the forced last frame is not paid for twice."""
        strategy = _make_on_policy_strategy(
            num_steps=3,
            training_steps_per_segment=1,
            generation_steps=5,
            label_frequency=5,
        )

        assert _labeled_steps(strategy) == [0, 4, 9, 14]

    def test_a_segment_aligned_cadence_stores_one_frame_per_segment(self) -> None:
        """Each trajectory contributes its segments' last frames, plus the seeded one."""
        strategy = _make_on_policy_strategy(
            num_steps=3,
            training_steps_per_segment=1,
            generation_steps=5,
            label_frequency=5,
        )

        strategy.run()

        assert len(strategy.replay_buffer) == 4 * len(_build_initial_dataset())

    def test_an_unaligned_cadence_keeps_every_labeling_but_the_adjacent_one(
        self,
    ) -> None:
        """Only the cadence step immediately after a forced frame is dropped."""
        strategy = _make_on_policy_strategy(
            num_steps=3,
            training_steps_per_segment=1,
            generation_steps=20,
            label_frequency=10,
        )

        assert _labeled_steps(strategy) == [0, 10, 19, 30, 39, 50, 59]

    def test_labeling_every_step_is_unaffected(self) -> None:
        """``label_frequency=1`` asks for every frame and still gets every frame."""
        strategy = _make_on_policy_strategy(
            num_steps=2,
            training_steps_per_segment=1,
            generation_steps=3,
            label_frequency=1,
        )

        assert _labeled_steps(strategy) == [0, 1, 2, 3, 4, 5]


class TestOnPolicyResume:
    def test_a_resumed_run_opens_a_fresh_segment(self) -> None:
        """A restored mid-segment counter is closed, so BEFORE_EPOCH fires again."""
        opened = _EpochStartHook()
        strategy = _make_on_policy_strategy(
            num_steps=9,
            training_steps_per_segment=4,
            hooks=[opened],
            step_count=5,
            epoch_count=1,
            epoch_step_count=1,
        )

        strategy.run()

        assert opened.calls == [(5, 2)]

    def test_a_resumed_segment_stays_inside_its_step_budget(self) -> None:
        """The interrupted segment's progress is not carried into the fresh one."""
        recorder = _RecordingBatchHook()
        strategy = _make_on_policy_strategy(
            num_steps=9,
            training_steps_per_segment=4,
            hooks=[recorder],
            step_count=5,
            epoch_count=1,
            epoch_step_count=1,
        )

        strategy.run()

        assert recorder.epoch_steps == [1, 2, 3, 4]

    def test_a_resumed_segment_does_not_redraw_the_interrupted_one(self) -> None:
        """The mixture sampler advances past the epoch index the crash consumed."""
        interrupted = _RecordingBatchHook()
        clean = _RecordingBatchHook()
        _make_on_policy_strategy(
            num_steps=9,
            training_steps_per_segment=4,
            hooks=[interrupted],
            step_count=5,
            epoch_count=1,
            epoch_step_count=1,
        ).run()
        _make_on_policy_strategy(
            num_steps=9,
            training_steps_per_segment=4,
            hooks=[clean],
            step_count=5,
            epoch_count=1,
            epoch_step_count=0,
        ).run()

        assert interrupted.reference_draws != clean.reference_draws

    def test_graduating_from_a_partial_offline_epoch_runs(self) -> None:
        """Offline epochs of another size are closed, not reconciled against one."""
        strategy = _make_on_policy_strategy(
            num_steps=8,
            training_steps_per_segment=4,
            step_count=3,
            epoch_count=0,
            epoch_step_count=3,
            batch_count=3,
        )

        strategy.run()

        assert strategy.step_count == 8
        assert strategy.epoch_count == 3


class TestOnPolicyRerun:
    def test_a_second_run_keeps_the_buffer_it_filled(self) -> None:
        """Continuing a finished run trains on everything generated so far."""
        strategy = _make_on_policy_strategy(num_steps=4, training_steps_per_segment=4)

        strategy.run()
        buffer = strategy.replay_buffer
        first_frames = len(buffer)
        first_steps = strategy.on_policy.dynamics.step_count
        strategy.num_steps = 8
        strategy.run()

        assert strategy.replay_buffer is buffer
        assert len(buffer) > first_frames
        assert strategy.on_policy.dynamics.step_count > first_steps


class TestOnPolicyGenerationSuppliedTargets:
    def test_a_declared_custom_target_trains_through_the_segments(self) -> None:
        """A field only the propagator's scorer writes supervises every batch."""
        teacher = _build_direct_force_teacher(seed=2)
        scorer = _CustomFieldScorer(teacher)
        recorder = _RecordingBatchHook()
        strategy = _make_on_policy_strategy(
            teacher=teacher,
            hooks=[recorder],
            loss_fn=_make_supplied_loss(),
            reference_dataset=_build_reference_dataset(scorer),
            config_overrides={"teacher_scorer": scorer},
        )

        strategy.run()

        assert strategy.step_count == 12
        assert strategy.teacher_scorer.signals == frozenset({"energy", "forces"})
        assert any(
            name.endswith(f".{_SUPPLIED_FIELD}")
            for name in strategy.replay_buffer.schema
        )
        assert len(recorder.losses) == 12

    def test_an_undeclared_scorer_still_supplies_a_custom_target(self) -> None:
        """A custom teacher field is an ordinary target whether or not it is declared."""
        teacher = _build_direct_force_teacher(seed=2)
        scorer = _CustomFieldScorer(teacher, label_fields=None)

        with pytest.warns(UserWarning, match="declare label_fields"):
            strategy = _make_on_policy_strategy(
                teacher=teacher,
                num_steps=4,
                loss_fn=_make_supplied_loss(),
                reference_dataset=_build_reference_dataset(scorer),
                config_overrides={"teacher_scorer": scorer},
            )
        strategy.run()

        assert strategy.step_count == 4

    def test_an_all_custom_objective_is_rejected(self) -> None:
        """A supplied target derives no signal, so the strategy's scorer has none."""
        teacher = _build_direct_force_teacher(seed=2)
        scorer = _CustomFieldScorer(teacher)

        with pytest.raises(ValueError, match="at least one teacher signal"):
            _make_on_policy_strategy(
                teacher=teacher,
                loss_fn=EnergyMSELoss(target_key=_SUPPLIED_FIELD),
                reference_dataset=_build_reference_dataset(scorer),
                config_overrides={"teacher_scorer": scorer},
            )

    def test_a_declaration_outside_the_namespace_is_rejected_up_front(self) -> None:
        """The propagator's scorer is policed for the namespace before it runs."""
        teacher = _build_direct_force_teacher(seed=2)
        scorer = _CustomFieldScorer(teacher, label_fields=("teacher_energy", "forces"))

        with pytest.raises(ValueError, match="teacher_\\*"):
            _make_on_policy_strategy(
                teacher=teacher, config_overrides={"teacher_scorer": scorer}
            )


class TestOnPolicyUnknownGenerationFields:
    def test_an_undeclared_scorer_warns_instead_of_rejecting_the_reference_dataset(
        self,
    ) -> None:
        """Unknown fields are not an empty set, so the parity check is deferred."""
        teacher = _build_direct_force_teacher(seed=2)
        scorer = _CustomFieldScorer(teacher, label_fields=None)

        with pytest.warns(UserWarning, match="declare label_fields"):
            strategy = _make_on_policy_strategy(
                teacher=teacher,
                reference_dataset=_build_reference_dataset(scorer),
                config_overrides={"teacher_scorer": scorer},
            )

        assert strategy.on_policy.teacher_scorer is scorer

    def test_a_declared_scorer_covering_the_reference_dataset_is_silent(self) -> None:
        """Custom signal names declaring the reference fields pass both checks."""
        teacher = _build_direct_force_teacher(seed=2)
        scorer = _CustomFieldScorer(teacher)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            strategy = _make_on_policy_strategy(
                teacher=teacher,
                loss_fn=_make_supplied_loss(),
                reference_dataset=_build_reference_dataset(scorer),
                config_overrides={"teacher_scorer": scorer},
            )

        assert strategy.on_policy.teacher_scorer is scorer
        assert not [
            record
            for record in caught
            if "scored twice" in str(record.message)
            or "declare label_fields" in str(record.message)
        ]

    def test_declared_fields_disagreeing_with_the_reference_dataset_are_rejected(
        self,
    ) -> None:
        """A declaration is taken at its word, so parity is checked against it."""
        teacher = _build_direct_force_teacher(seed=2)

        with pytest.raises(ValueError, match="same teacher fields"):
            _make_on_policy_strategy(
                teacher=teacher,
                loss_fn=_make_supplied_loss(),
                reference_dataset=_build_reference_dataset(_make_scorer(teacher)),
                config_overrides={"teacher_scorer": _CustomFieldScorer(teacher)},
            )


class TestOnPolicySerialization:
    def test_serializing_an_on_policy_strategy_warns(self) -> None:
        """A spec cannot describe a live propagator, so the omission is announced."""
        strategy = _make_on_policy_strategy(num_steps=2)

        with pytest.warns(UserWarning, match="omitted from the spec"):
            spec = strategy.to_spec_dict()

        assert "on_policy" not in spec
        assert "reference_dataset" not in spec

    def test_the_round_trip_rebuilds_an_offline_strategy(self) -> None:
        """A rebuilt strategy runs offline until the on-policy pieces are re-supplied."""
        strategy = _make_on_policy_strategy(num_steps=2)
        with pytest.warns(UserWarning, match="omitted from the spec"):
            spec = json.loads(json.dumps(strategy.to_spec_dict()))

        teacher = _build_direct_force_teacher(seed=2)
        rebuilt = DistillationStrategy.from_spec_dict(
            spec, models={"student": _build_demo_model(), "teacher": teacher}
        )

        assert rebuilt.on_policy is None
        assert rebuilt.reference_dataset is None
        rebuilt.run([_build_propagator_batch(_REFERENCE_ELEMENT, 2, base_seed=900)])
        assert rebuilt.step_count == 2

    def test_an_offline_strategy_serializes_without_warning(self) -> None:
        """The warning is about the on-policy fields, not about distillation."""
        teacher = _build_direct_force_teacher(seed=2)
        strategy = DistillationStrategy(
            models={"student": _build_demo_model(), "teacher": teacher},
            optimizer_configs=_make_optimizer_configs(),
            loss_fn=_make_loss(),
            num_steps=2,
        )

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            spec = strategy.to_spec_dict()

        assert spec["label_missing"] is True
        assert not [w for w in caught if "omitted from the spec" in str(w.message)]


@pytest.mark.slow
class TestOnPolicyLabelingOverhead:
    def _time_segment(self, dynamics: NVTLangevin, n_steps: int) -> float:
        """Return the fastest of three warmed-up segments of *n_steps*, in seconds."""
        dynamics.run(
            _build_propagator_batch(_INITIAL_ELEMENT, 4, base_seed=500), n_steps=2
        )
        timings = []
        for _ in range(3):
            state = _build_propagator_batch(_INITIAL_ELEMENT, 4, base_seed=500)
            start = time.perf_counter()
            dynamics.run(state, n_steps=n_steps)
            timings.append(time.perf_counter() - start)
        return min(timings)

    def test_labeling_overhead_stays_within_budget(self) -> None:
        """Labeling every step of a segment costs a bounded multiple of generating it."""
        student = _build_demo_model()
        scorer = _make_scorer(_build_direct_force_teacher(seed=2))
        labeled = NVTLangevin(student, **_LANGEVIN_KWARGS)
        labeled.register_hook(TeacherLabelHook(scorer))
        captured = NVTLangevin(student, **_LANGEVIN_KWARGS)
        captured.register_hook(
            TeacherLabelHook(scorer, sink=HostMemory(capacity=10_000))
        )

        bare_seconds = self._time_segment(NVTLangevin(student, **_LANGEVIN_KWARGS), 40)
        labeled_seconds = self._time_segment(labeled, 40)
        captured_seconds = self._time_segment(captured, 40)

        measured = (
            f"generation {bare_seconds:.3f}s; labeling every step "
            f"{labeled_seconds / bare_seconds:.2f}x; labeling and capturing "
            f"{captured_seconds / bare_seconds:.2f}x"
        )
        assert labeled_seconds / bare_seconds < 3.0, measured
        assert captured_seconds / bare_seconds < 4.0, measured


class TestOnPolicyCustomSource:
    def test_a_custom_source_seeds_a_segment_loop(self) -> None:
        """A minimal ``InitialStructuresSource`` runs the loop end to end."""
        source = _ListSource(
            [
                _build_propagator_system(_INITIAL_ELEMENT, 500 + index)
                for index in range(4)
            ]
        )
        strategy = _make_on_policy_strategy(
            config_overrides={"initial_structures": source}
        )

        strategy.run()

        assert source.shards == [(0, 1)]
        assert source.exhausted
        assert strategy.step_count == 12
        assert len(strategy.replay_buffer) > 0


class TestOnPolicyCaptureSink:
    def test_a_configured_sink_stages_every_frame_the_default_one_would(self) -> None:
        """A user sink captures exactly what a host-memory sink captures, then drains."""
        reference = _make_on_policy_strategy()
        reference.run()
        sink = _RecordingSink(capacity=64)
        strategy = _make_on_policy_strategy(config_overrides={"capture_sink": sink})

        strategy.run()

        assert len(strategy.replay_buffer) == len(reference.replay_buffer)
        assert len(sink) == 0
        assert sink.resizes == []

    def test_a_small_resizable_sink_is_grown_to_the_segment_capacity(self) -> None:
        """The loop asks for (generation_steps + 1) frames per trajectory."""
        sink = _RecordingSink(capacity=1)
        strategy = _make_on_policy_strategy(
            generation_steps=3, config_overrides={"capture_sink": sink}
        )

        strategy.run()

        assert sink.resizes == [(3 + 1) * 4]
        assert len(strategy.replay_buffer) > 0

    def test_a_sink_is_resizable_when_it_offers_capacity_and_resize(self) -> None:
        """The protocol is checked structurally; a stock host-memory sink lacks resize."""
        assert isinstance(_RecordingSink(capacity=1), ResizableSink)
        assert not isinstance(HostMemory(capacity=1), ResizableSink)

    def test_a_small_sink_without_resize_is_refused(self) -> None:
        """The refusal names the capacity the segment needs and both remedies."""
        strategy = _make_on_policy_strategy(
            generation_steps=3,
            config_overrides={"capture_sink": HostMemory(capacity=1)},
        )

        with pytest.raises(ValueError, match="capacity 1 without a resize method"):
            strategy.run()

    def test_a_sink_still_holding_frames_is_refused(self) -> None:
        """Foreign frames would be drained into the buffer as generated ones."""
        sink = HostMemory(capacity=64)
        sink.write(_build_propagator_batch(_INITIAL_ELEMENT, 1, base_seed=900))
        strategy = _make_on_policy_strategy(config_overrides={"capture_sink": sink})

        with pytest.raises(ValueError, match="must be empty when a segment starts"):
            strategy.run()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_a_gpu_buffer_stages_frames_on_the_generation_device(self) -> None:
        """The in-tree device-resident sink completes a run with every frame captured."""
        reference = _make_on_policy_strategy(device="cuda")
        reference.run()
        sink = GPUBuffer(
            capacity=(3 + 1) * 4,
            max_atoms=_ATOMS_PER_SYSTEM,
            max_edges=0,
            device="cuda",
        )
        strategy = _make_on_policy_strategy(
            device="cuda", generation_steps=3, config_overrides={"capture_sink": sink}
        )

        strategy.run()

        assert len(strategy.replay_buffer) == len(reference.replay_buffer)
        assert len(sink) == 0


class _CountingAdmission:
    """Admission predicate admitting every frame and counting the batches it saw."""

    def __init__(self) -> None:
        self.batches: list[int] = []

    def __call__(self, frames: Batch) -> torch.Tensor:
        """Admit every frame."""
        self.batches.append(frames.num_graphs)
        return torch.ones(frames.num_graphs, dtype=torch.bool, device=frames.device)


class _EvictOldestHalfFirst:
    """Eviction policy naming the oldest frames, recording every capacity event."""

    def __init__(self) -> None:
        self.calls = 0

    def select(self, buffer: Batch, incoming: Batch, capacity: int) -> torch.Tensor:  # noqa: ARG002
        """Drop the oldest frames past capacity."""
        self.calls += 1
        return torch.arange(buffer.num_graphs - capacity, device=buffer.device)


class TestOnPolicyReplayPolicies:
    def test_the_loop_wires_both_policies_into_its_buffer(self) -> None:
        """The buffer the loop builds admits and evicts through the configured policies."""
        admission = _CountingAdmission()
        eviction = _EvictOldestHalfFirst()
        strategy = _make_on_policy_strategy(
            config_overrides={
                "replay_admission": admission,
                "replay_eviction": eviction,
                "replay_capacity": 8,
            }
        )

        strategy.run()

        assert strategy.replay_buffer.admission is admission
        assert strategy.replay_buffer.eviction is eviction
        assert len(admission.batches) == 3
        assert eviction.calls >= 1
        assert len(strategy.replay_buffer) == 8
