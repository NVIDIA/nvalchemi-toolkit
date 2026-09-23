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
"""Tests for the relaxation trajectory lifecycle of the on-policy segment loop."""

from __future__ import annotations

import warnings
from typing import Any
from unittest.mock import patch

import pytest
import torch
from pydantic import ValidationError

from nvalchemi.data import AtomicData, Batch
from nvalchemi.data.datapipes.in_memory_dataset import InMemoryDataset
from nvalchemi.dynamics.base import (
    BaseDynamics,
    ConvergenceHook,
    DynamicsStage,
    FusedStage,
)
from nvalchemi.dynamics.integrators.nve import NVE
from nvalchemi.dynamics.optimizers.fire import FIRE
from nvalchemi.dynamics.sampler import SizeAwareSampler
from nvalchemi.dynamics.sinks import HostMemory
from nvalchemi.models.base import BaseModelMixin
from nvalchemi.training import (
    EnergyMSELoss,
    ForceMSELoss,
    OptimizerConfig,
    TrainingStage,
)
from nvalchemi.training.distillation import (
    DistillationStrategy,
    InitialStructures,
    InProcessTeacherScorer,
    OnPolicyConfig,
    ReplayBuffer,
    TeacherLabelHook,
    nonfinite_divergence,
)
from nvalchemi.training.distillation.scoring import TeacherLabels
from nvalchemi.training.distillation.strategy import _relaxation_lifecycle
from test.training.conftest import _build_demo_model
from test.training.distillation.conftest import (
    _INITIAL_ELEMENT,
    _build_direct_force_teacher,
    _build_initial_dataset,
    _build_lj_teacher,
    _build_propagator_batch,
    _build_reference_dataset,
    _ListSource,
)

_SCORE_KEY = "convergence_score"
"""Graph-level key the scripted criterion converges a relaxation on."""


def _shard_then_restore(bundle: dict[str, int]) -> Any:
    """Return the shard install a restart follows with the cursor it recorded.

    The restart bundle itself lands with the recipe work; what the loop already
    owes is the order — the rank shard first, because it reopens the cursor,
    and the recorded position over it.
    """
    real_shard = InitialStructures.shard

    def _install(source: InitialStructures, rank: int, world_size: int) -> None:
        """Install the shard, then resume the cursor the bundle recorded."""
        real_shard(source, rank, world_size)
        source.load_state_dict(bundle)

    return _install


def _make_scripted_criterion() -> ConvergenceHook:
    """Return the migrating hook reading what the scripted source writes."""
    return ConvergenceHook(
        criteria=[{"key": _SCORE_KEY, "threshold": 0.5}],
        source_status=0,
        target_status=1,
    )


def _make_prediction_less_dataset(n_systems: int = 3) -> InMemoryDataset:
    """Return structures a store kept without the model outputs FIRE primes itself."""
    return InMemoryDataset(
        in_memory_batch=_build_propagator_batch(
            _INITIAL_ELEMENT, n_systems, base_seed=500, predictions=False
        )
    )


def _make_velocity_less_dataset(n_systems: int = 3) -> InMemoryDataset:
    """Return structures a store kept without the velocities FIRE updates in place."""
    frames = _build_propagator_batch(_INITIAL_ELEMENT, n_systems, base_seed=500)
    del frames["velocities"]
    return InMemoryDataset(in_memory_batch=frames)


def _make_graduated_dataset(n_systems: int = 3) -> InMemoryDataset:
    """Return structures stored the way a converged relaxation would have left them."""
    frames = _build_propagator_batch(_INITIAL_ELEMENT, n_systems, base_seed=500)
    frames["status"] = torch.ones(n_systems, 1, dtype=torch.long)
    frames["system_id"] = torch.arange(n_systems, dtype=torch.long).unsqueeze(-1)
    return InMemoryDataset(in_memory_batch=frames)


def _diverge_first_system(batch: Batch) -> torch.Tensor:
    """Divergence predicate flagging the structure the source numbered 0."""
    return batch.system_id.view(-1)[: batch.num_graphs] == 0


def _diverge_per_atom(batch: Batch) -> torch.Tensor:
    """Divergence predicate returning one flag per atom instead of per graph."""
    return torch.zeros(batch.num_nodes, dtype=torch.bool, device=batch.device)


def _diverge_as_floats(batch: Batch) -> torch.Tensor:
    """Divergence predicate returning a float mask instead of a boolean one."""
    return torch.zeros(batch.num_graphs, device=batch.device)


def _diverge_as_list(batch: Batch) -> list[bool]:
    """Divergence predicate returning a list instead of a tensor."""
    return [False] * batch.num_graphs


def _make_sized_dataset(sizes: list[int]) -> InMemoryDataset:
    """Return structures of *sizes* atoms each, carrying what FIRE opens on."""
    data_list = []
    for index, size in enumerate(sizes):
        generator = torch.Generator().manual_seed(600 + index)
        data_list.append(
            AtomicData(
                positions=torch.randn(size, 3, generator=generator),
                atomic_numbers=torch.full((size,), _INITIAL_ELEMENT, dtype=torch.long),
                atomic_masses=torch.ones(size),
                energy=torch.zeros(1, 1),
                forces=torch.zeros(size, 3),
            )
        )
    return InMemoryDataset(in_memory_batch=Batch.from_data_list(data_list))


def _make_relaxation_strategy(
    *,
    fmax: float | None = None,
    convergence_hook: ConvergenceHook | None = None,
    student: BaseModelMixin | None = None,
    teacher: BaseModelMixin | None = None,
    structures: InitialStructures | None = None,
    num_steps: int = 6,
    training_steps_per_segment: int = 2,
    generation_steps: int = 4,
    label_frequency: int = 1,
    replay_ratio: float = 1.0,
    device: str = "cpu",
    config_overrides: dict[str, Any] | None = None,
    **overrides: Any,
) -> DistillationStrategy:
    """Return a runnable FIRE relaxation strategy over independent demo models."""
    student = _build_demo_model() if student is None else student
    teacher = _build_direct_force_teacher(seed=2) if teacher is None else teacher
    scorer = InProcessTeacherScorer(teacher, ("energy", "forces"))
    config_kwargs: dict[str, Any] = {
        "dynamics": FIRE(student, dt=0.1),
        "teacher_scorer": scorer,
        "initial_structures": InitialStructures(_build_initial_dataset(n_systems=3))
        if structures is None
        else structures,
        "replay_ratio": replay_ratio,
        "training_steps_per_segment": training_steps_per_segment,
        "batch_size": 4,
        "generation_steps": generation_steps,
        "label_frequency": label_frequency,
        "fmax": fmax,
        "convergence_hook": convergence_hook,
    }
    config_kwargs.update(config_overrides or {})
    kwargs: dict[str, Any] = {
        "models": {"student": student, "teacher": teacher},
        "optimizer_configs": {
            "student": [
                OptimizerConfig(
                    optimizer_cls=torch.optim.Adam, optimizer_kwargs={"lr": 1e-2}
                )
            ]
        },
        "loss_fn": EnergyMSELoss(target_key="teacher_energy")
        + ForceMSELoss(target_key="teacher_forces", normalize_by_atom_count=True),
        "num_steps": num_steps,
        "devices": [torch.device(device)],
        "reference_dataset": None
        if replay_ratio == 1.0
        else _build_reference_dataset(scorer),
        "on_policy": OnPolicyConfig(**config_kwargs),
    }
    kwargs.update(overrides)
    return DistillationStrategy(**kwargs)


def _positions(batch: Batch, index: int) -> tuple[float, ...]:
    """Return the rounded positional fingerprint of one graph of *batch*."""
    return tuple(
        round(float(value), 6)
        for value in batch.positions[batch.batch_idx == index].flatten()
    )


def _frame_fingerprints(strategy: DistillationStrategy) -> list[tuple[float, ...]]:
    """Return one positional fingerprint per frame the run stored."""
    frames = strategy.replay_buffer.dataset.in_memory_batch
    return [_positions(frames, index) for index in range(frames.num_graphs)]


class _ScriptedRelaxation:
    """Write a graph-level score converging each system after its own step count."""

    stage = DynamicsStage.BEFORE_STEP
    frequency = 1

    def __init__(self, schedule: dict[int, int]) -> None:
        """Start every scripted system at zero propagated steps."""
        self.schedule = schedule
        self.propagated: dict[int, int] = {}

    def __call__(self, ctx: Any, stage: Any) -> None:  # noqa: ARG002
        """Score each live system against the steps it has been propagated."""
        batch = ctx.batch
        scores = torch.ones(batch.num_graphs, 1, device=batch.device)
        for row, system in enumerate(batch.system_id.view(-1).tolist()):
            self.propagated[system] = self.propagated.get(system, 0) + 1
            if self.propagated[system] > self.schedule.get(system, 10_000):
                scores[row] = 0.0
        batch[_SCORE_KEY] = scores


class _StateProbe:
    """Record the live composition and the propagator's own state every step."""

    frequency = 1

    def __init__(self, stage: Any = DynamicsStage.BEFORE_STEP) -> None:
        """Start with an empty trace, taken at *stage*."""
        self.stage = stage
        self.systems: list[list[int]] = []
        self.state_rows: list[int] = []
        self.graph_counts: list[int] = []
        self.node_counts: list[int] = []
        self.n_steps_positive: list[list[int]] = []
        self.first_positions: dict[int, tuple[float, ...]] = {}

    def __call__(self, ctx: Any, stage: Any) -> None:  # noqa: ARG002
        """Append the batch composition and the FIRE state rows behind it."""
        state = getattr(ctx.workflow, "_state", None)
        systems = ctx.batch.system_id.view(-1).tolist()
        self.systems.append(systems)
        self.graph_counts.append(ctx.batch.num_graphs)
        self.node_counts.append(int(ctx.batch.num_nodes))
        self.state_rows.append(0 if state is None else state.num_graphs)
        self.n_steps_positive.append(
            [] if state is None else [int(value) for value in state.n_steps_positive]
        )
        for row, system in enumerate(systems):
            self.first_positions.setdefault(system, _positions(ctx.batch, row))


class _StatusProbe:
    """Record the live batch's status column, or its absence, every step."""

    frequency = 1
    stage = DynamicsStage.AFTER_STEP

    def __init__(self) -> None:
        """Start with an empty trace."""
        self.statuses: list[list[int] | None] = []

    def __call__(self, ctx: Any, stage: Any) -> None:  # noqa: ARG002
        """Append the per-graph status, or ``None`` for a frame carrying none."""
        status = getattr(ctx.batch, "status", None)
        self.statuses.append(None if status is None else status.view(-1).tolist())


class _FrameProbe:
    """Record the positional fingerprint each system is left with, every step."""

    frequency = 1
    stage = DynamicsStage.AFTER_STEP

    def __init__(self) -> None:
        """Start with an empty trace."""
        self.frames: list[dict[int, tuple[float, ...]]] = []

    def __call__(self, ctx: Any, stage: Any) -> None:  # noqa: ARG002
        """Append the frame the step just left, one fingerprint per system."""
        systems = ctx.batch.system_id.view(-1).tolist()
        self.frames.append(
            {system: _positions(ctx.batch, row) for row, system in enumerate(systems)}
        )


class _NaNInjector:
    """Overwrite one system's forces with NaN from a given step on, after the model."""

    frequency = 1
    stage = DynamicsStage.AFTER_COMPUTE

    def __init__(self, system: int, at_step: int) -> None:
        """Diverge *system* on propagator step *at_step*."""
        self.system = system
        self.at_step = at_step

    def __call__(self, ctx: Any, stage: Any) -> None:  # noqa: ARG002
        """Poison the forces of the scripted system once its step has come."""
        if ctx.step_count < self.at_step:
            return
        batch = ctx.batch
        graphs = batch.system_id.view(-1)[: batch.num_graphs] == self.system
        batch.forces[graphs[batch.batch_idx.long()]] = float("nan")


class _AutocastProbeScorer:
    """Scorer recording whether autocast was live on each call, then delegating."""

    signals = frozenset({"energy", "forces"})

    def __init__(self, inner: InProcessTeacherScorer) -> None:
        """Wrap *inner* with an empty trace."""
        self.inner = inner
        self.autocast_states: list[bool] = []

    def label(self, batch: Batch) -> TeacherLabels:
        """Record the autocast state for the batch's device and score it."""
        self.autocast_states.append(torch.is_autocast_enabled(batch.device.type))
        return self.inner.label(batch)


class _ForeignFieldScorer:
    """Scorer that tries to write the propagator's own force field."""

    signals = frozenset({"unregistered"})

    def label(self, batch: Batch) -> TeacherLabels:
        """Return a label aimed at ``forces`` instead of ``teacher_forces``."""
        return {"forces": (torch.zeros(batch.num_nodes, 3), "node")}


class _ResizableSink(HostMemory):
    """Host-memory sink recording every capacity the loop resizes it to."""

    def __init__(self, capacity: int) -> None:
        super().__init__(capacity)
        self.resizes: list[int] = []

    def resize(self, capacity: int) -> None:
        """Grow to *capacity* and record the request."""
        self.resizes.append(capacity)
        self._capacity = capacity


class _RecordingBatchHook:
    """Record the loss of every training batch."""

    frequency = 1
    stage = TrainingStage.AFTER_BATCH

    def __init__(self) -> None:
        """Start with an empty trace."""
        self.losses: list[float] = []

    def __call__(self, ctx: Any, stage: Any) -> None:  # noqa: ARG002
        """Append the loss of the batch just trained on."""
        self.losses.append(float(ctx.loss))


class TestRelaxationConfig:
    def test_the_fmax_shorthand_resolves_to_a_status_migrating_hook(self) -> None:
        """A float becomes a force criterion that graduates on the exit status."""
        strategy = _make_relaxation_strategy(fmax=0.05)

        resolved = strategy.on_policy.convergence_criterion

        assert isinstance(resolved, ConvergenceHook)
        assert resolved.source_status == 0
        assert resolved.target_status == strategy.on_policy.dynamics.exit_status
        assert resolved.criteria[0].key == "forces"
        assert resolved.criteria[0].threshold == 0.05

    def test_the_resolved_criterion_is_the_same_object_every_read(self) -> None:
        """The lifecycle registers and removes one hook, so identity has to hold."""
        config = _make_relaxation_strategy(fmax=0.05).on_policy

        assert config.convergence_criterion is config.convergence_criterion

    def test_a_criterion_passed_whole_is_its_own_resolution(self) -> None:
        """Nothing is rebuilt around a hook the caller already wired up."""
        criterion = _make_scripted_criterion()
        config = _make_relaxation_strategy(convergence_hook=criterion).on_policy

        assert config.convergence_criterion is criterion

    def test_the_fmax_field_survives_the_run_as_a_float(self) -> None:
        """The serializable shorthand is not traded away for the live hook."""
        strategy = _make_relaxation_strategy(fmax=0.05, num_steps=2)

        strategy.run()

        assert strategy.on_policy.fmax == 0.05
        assert strategy.on_policy.settings.fmax == 0.05

    def test_both_spellings_of_the_criterion_are_rejected_together(self) -> None:
        """The threshold and the hook name one criterion, so exactly one is taken."""
        with pytest.raises(ValidationError, match="two spellings of one"):
            _make_relaxation_strategy(
                fmax=0.05, convergence_hook=_make_scripted_criterion()
            )

    def test_a_criterion_that_migrates_no_status_is_rejected(self) -> None:
        """A hook that only reports convergence would freeze and graduate nothing."""
        with pytest.raises(ValueError, match="has to migrate status"):
            _make_relaxation_strategy(convergence_hook=ConvergenceHook.from_fmax(0.05))

    def test_a_target_status_below_the_exit_status_is_rejected(self) -> None:
        """Migrating below the exit status leaves the structure in the batch."""
        with pytest.raises(ValueError, match="at least the propagator's exit status"):
            _make_relaxation_strategy(
                convergence_hook=ConvergenceHook.from_fmax(
                    0.05, source_status=0, target_status=0
                )
            )

    def test_a_criterion_that_skips_steps_is_rejected(self) -> None:
        """A gated criterion graduates late, so both routes store the same frame."""
        with pytest.raises(ValueError, match="has to run on every step"):
            _make_relaxation_strategy(
                convergence_hook=ConvergenceHook.from_fmax(
                    0.05, source_status=0, target_status=1, frequency=3
                )
            )

    def test_recycling_without_a_convergence_criterion_is_rejected(self) -> None:
        """Nothing backfills without a lifecycle, so the flag would be a no-op."""
        with pytest.raises(ValidationError, match="InitialStructures.recycle restarts"):
            _make_relaxation_strategy(
                fmax=None,
                structures=InitialStructures(
                    _build_initial_dataset(n_systems=3), recycle=True
                ),
            )


class _InertCriterion(ConvergenceHook):
    """Criterion that converges but never writes the status it promises to migrate."""

    def __call__(self, ctx: Any, stage: Any) -> None:  # noqa: ARG002
        """Migrate nothing."""


class TestRelaxationCriterionProbe:
    """Construction dispatches a copy of the criterion to the probed row."""

    def test_a_threshold_every_row_meets_migrates_the_probe_and_passes(self) -> None:
        """A criterion that fires on the row proves the mechanism and is accepted."""
        strategy = _make_relaxation_strategy(fmax=1e3)

        criterion = strategy.on_policy.convergence_criterion

        assert criterion.target_status == strategy.on_policy.dynamics.exit_status

    def test_the_live_criterion_is_not_the_one_dispatched(self) -> None:
        """The probe fires a copy, so the object the lifecycle registers is untouched."""
        criterion = ConvergenceHook.from_fmax(1e3, source_status=0, target_status=1)
        with patch.object(
            ConvergenceHook,
            "__call__",
            autospec=True,
            side_effect=ConvergenceHook.__call__,
        ) as dispatched:
            _make_relaxation_strategy(convergence_hook=criterion)

        assert dispatched.call_count >= 1
        assert all(call.args[0] is not criterion for call in dispatched.call_args_list)

    def test_a_criterion_whose_firing_leaves_status_unmoved_is_refused(self) -> None:
        """Converging without migrating would freeze and graduate nothing."""
        criterion = _InertCriterion(
            criteria=[
                {
                    "key": "forces",
                    "threshold": 1e3,
                    "reduce_op": "norm",
                    "reduce_dims": -1,
                }
            ],
            source_status=0,
            target_status=1,
        )

        with pytest.raises(ValueError, match="status column did not migrate"):
            _make_relaxation_strategy(convergence_hook=criterion)

    def test_a_criterion_reading_a_key_compute_never_writes_warns_and_passes(
        self,
    ) -> None:
        """A key a step hook writes cannot be checked on one compute(), so it warns."""
        with pytest.warns(UserWarning, match=r"reads \['convergence_score'\]"):
            strategy = _make_relaxation_strategy(
                convergence_hook=_make_scripted_criterion()
            )

        assert strategy.on_policy.convergence_criterion.criteria[0].key == _SCORE_KEY

    def test_probe_false_skips_the_criterion_dispatch_as_well(self) -> None:
        """The flag that skips the student forward skips the criterion probe too."""
        criterion = _InertCriterion(
            criteria=[
                {
                    "key": "forces",
                    "threshold": 1e3,
                    "reduce_op": "norm",
                    "reduce_dims": -1,
                }
            ],
            source_status=0,
            target_status=1,
        )

        strategy = _make_relaxation_strategy(
            convergence_hook=criterion, config_overrides={"probe": False}
        )

        assert strategy.on_policy.convergence_criterion is criterion
        assert strategy.on_policy.probe is False


class TestRelaxationStructureContract:
    def test_structures_without_the_propagated_predictions_relax(self) -> None:
        """FIRE primes the forces it opens on, so a structure need not carry them."""
        strategy = _make_relaxation_strategy(
            fmax=0.05,
            num_steps=2,
            structures=InitialStructures(_make_prediction_less_dataset()),
        )

        strategy.run()

        assert len(strategy.replay_buffer) > 0

    def test_structures_without_velocities_are_rejected(self) -> None:
        """A store that dropped the propagator state names it back at construction."""
        with pytest.raises(ValidationError, match="missing \\['velocities'\\]"):
            _make_relaxation_strategy(
                fmax=0.05, structures=InitialStructures(_make_velocity_less_dataset())
            )

    def test_the_rejection_names_what_the_propagator_declares(self) -> None:
        """The message points at the propagator's own declarations, not at a guess."""
        with pytest.raises(
            ValidationError, match="__provides_keys__=\\['positions', 'velocities'\\]"
        ):
            _make_relaxation_strategy(
                fmax=0.05, structures=InitialStructures(_make_velocity_less_dataset())
            )


class TestRelaxationLifecycle:
    def _run_scripted(
        self, schedule: dict[int, int], **kwargs: Any
    ) -> tuple[DistillationStrategy, _StateProbe]:
        """Run a relaxation whose systems converge on a scripted schedule."""
        strategy = _make_relaxation_strategy(
            convergence_hook=_make_scripted_criterion(), **kwargs
        )
        probe = _StateProbe()
        strategy.on_policy.dynamics.register_hook(_ScriptedRelaxation(schedule))
        strategy.on_policy.dynamics.register_hook(probe)
        strategy.run()
        return strategy, probe

    def test_converged_structures_graduate_and_fresh_ones_backfill(self) -> None:
        """A converged relaxation leaves the batch and a fresh structure takes its slot."""
        strategy, probe = self._run_scripted(
            {0: 2, 1: 5},
            structures=InitialStructures(
                _build_initial_dataset(n_systems=3), recycle=True
            ),
            num_steps=6,
        )

        assert probe.systems[0] == [0, 1, 2]
        assert probe.systems[-1] == [2, 3, 4]
        assert all(count == 3 for count in probe.graph_counts)
        assert strategy.on_policy.dynamics.step_count == 12

    def test_the_backfill_serves_the_row_the_recycled_cursor_reached(self) -> None:
        """A wrapped cursor hands back rows 0 and 1, not an arbitrary pair."""
        _, probe = self._run_scripted(
            {0: 2, 1: 5},
            structures=InitialStructures(
                _build_initial_dataset(n_systems=3), recycle=True
            ),
            num_steps=6,
        )

        structures = _build_propagator_batch(_INITIAL_ELEMENT, 3, base_seed=500)
        assert probe.first_positions[3] == _positions(structures, 0)
        assert probe.first_positions[4] == _positions(structures, 1)

    def test_the_state_rows_follow_the_live_batch_through_a_refill(self) -> None:
        """FIRE keeps one state row per graph across every graduation."""
        _, probe = self._run_scripted(
            {0: 2, 1: 5},
            structures=InitialStructures(
                _build_initial_dataset(n_systems=3), recycle=True
            ),
            num_steps=6,
        )

        assert probe.state_rows[1:] == probe.graph_counts[1:]

    def test_a_surviving_structure_keeps_its_state_while_a_fresh_one_resets(
        self,
    ) -> None:
        """Refill preserves the rows that stayed and defaults the ones that arrived."""
        strategy = _make_relaxation_strategy(
            convergence_hook=_make_scripted_criterion(),
            structures=InitialStructures(
                _build_initial_dataset(n_systems=3), recycle=True
            ),
            num_steps=4,
        )
        closing = _StateProbe(stage=DynamicsStage.AFTER_STEP)
        opening = _StateProbe()
        strategy.on_policy.dynamics.register_hook(_ScriptedRelaxation({0: 2}))
        strategy.on_policy.dynamics.register_hook(closing)
        strategy.on_policy.dynamics.register_hook(opening)

        strategy.run()

        assert closing.systems[3] == [0, 1, 2]
        assert opening.systems[4] == [1, 2, 3]
        assert opening.n_steps_positive[4][:2] == closing.n_steps_positive[3][1:]
        assert opening.n_steps_positive[4][-1] == 0

    def test_a_backfilled_structure_enters_moving_whatever_its_source_stored(
        self,
    ) -> None:
        """A store of graduated minima backfills structures the run still relaxes."""
        strategy = _make_relaxation_strategy(
            convergence_hook=_make_scripted_criterion(),
            structures=InitialStructures(_make_graduated_dataset(), recycle=True),
            num_steps=6,
        )
        status = _StatusProbe()
        frames = _FrameProbe()
        opening = _StateProbe()
        strategy.on_policy.dynamics.register_hook(_ScriptedRelaxation({0: 2, 1: 5}))
        strategy.on_policy.dynamics.register_hook(status)
        strategy.on_policy.dynamics.register_hook(frames)
        strategy.on_policy.dynamics.register_hook(opening)

        strategy.run()

        assert opening.systems[4] == [1, 2, 3]
        assert status.statuses[4] == [0, 0, 0]
        assert frames.frames[7][3] != frames.frames[4][3]

    def test_a_budgeted_backfill_is_restatused_and_keeps_its_system_id(self) -> None:
        """The source numbers the replacement; the run decides whether it moves."""
        strategy = _make_relaxation_strategy(
            convergence_hook=_make_scripted_criterion(),
            num_steps=6,
            structures=InitialStructures(
                _make_graduated_dataset(n_systems=5),
                max_atoms=64,
                max_batch_size=3,
            ),
        )
        status = _StatusProbe()
        frames = _FrameProbe()
        opening = _StateProbe()
        strategy.on_policy.dynamics.register_hook(_ScriptedRelaxation({0: 2, 1: 5}))
        strategy.on_policy.dynamics.register_hook(status)
        strategy.on_policy.dynamics.register_hook(frames)
        strategy.on_policy.dynamics.register_hook(opening)

        strategy.run()

        assert opening.systems[4] == [1, 2, 3]
        assert status.statuses[4] == [0, 0, 0]
        assert frames.frames[7][3] != frames.frames[4][3]

    def test_a_run_that_converges_nothing_generates_like_a_trajectory(self) -> None:
        """Without a graduation the loop is the molecular-dynamics loop underneath."""
        strategy = _make_relaxation_strategy(fmax=1e-6, num_steps=6, generation_steps=3)
        probe = _StateProbe()
        strategy.on_policy.dynamics.register_hook(probe)

        strategy.run()

        assert all(systems == [0, 1, 2] for systems in probe.systems)
        assert len(strategy.replay_buffer) == 9 * 3

    def test_the_propagator_is_left_as_it_was_handed_over(self) -> None:
        """The criterion and the capture hook are temporary, and nothing else is touched."""
        strategy = _make_relaxation_strategy(fmax=0.05, num_steps=2)

        strategy.run()

        dynamics = strategy.on_policy.dynamics
        assert dynamics.hooks == []
        assert dynamics.convergence_hook is None
        assert dynamics.sampler is None
        assert dynamics.done is False

    def test_a_run_that_exhausts_its_structures_still_hands_the_propagator_back(
        self,
    ) -> None:
        """Running dry leaves no sampler and no done flag on the propagator."""
        strategy = _make_relaxation_strategy(
            fmax=1e3,
            num_steps=4,
            training_steps_per_segment=2,
            generation_steps=2,
        )

        with pytest.warns(UserWarning, match="nothing left to start a fresh one"):
            strategy.run()

        assert strategy.on_policy.dynamics.sampler is None
        assert strategy.on_policy.dynamics.done is False

    def test_a_reused_propagator_generates_a_whole_second_run(self) -> None:
        """A second strategy over the same FIRE instance generates every segment."""
        student = _build_demo_model()
        dynamics = FIRE(student, dt=0.1)
        exhausted = _make_relaxation_strategy(
            fmax=1e3,
            student=student,
            num_steps=4,
            training_steps_per_segment=2,
            generation_steps=2,
            config_overrides={"dynamics": dynamics},
        )
        with pytest.warns(UserWarning, match="nothing left to start a fresh one"):
            exhausted.run()

        reused = _make_relaxation_strategy(
            fmax=1e-6,
            student=student,
            num_steps=4,
            training_steps_per_segment=2,
            generation_steps=2,
            config_overrides={"dynamics": dynamics},
        )
        reused.run()

        assert len(reused.replay_buffer) == 2 * 2 * 3


class TestRelaxationStructureExhaustion:
    def test_exhaustion_stops_generation_and_training_still_finishes(self) -> None:
        """The remaining steps train on the frames the run already generated."""
        strategy = _make_relaxation_strategy(
            fmax=1e3,
            num_steps=8,
            training_steps_per_segment=2,
            generation_steps=4,
        )

        with pytest.warns(UserWarning, match="nothing left to start a fresh one"):
            strategy.run()

        assert strategy.step_count == 8
        assert strategy.on_policy.dynamics.step_count == 1
        assert len(strategy.replay_buffer) == 3

    def test_recycled_structures_keep_generation_going(self) -> None:
        """Restarting at the beginning of the dataset never runs the loop dry."""
        strategy = _make_relaxation_strategy(
            fmax=1e3,
            num_steps=8,
            training_steps_per_segment=2,
            generation_steps=4,
            structures=InitialStructures(
                _build_initial_dataset(n_systems=3), recycle=True
            ),
        )

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            strategy.run()

        assert not [
            entry for entry in caught if "nothing left to start" in str(entry.message)
        ]
        assert strategy.step_count == 8
        assert strategy.on_policy.dynamics.step_count == 4
        assert len(strategy.replay_buffer) == 12

    def test_a_shrinking_batch_keeps_generating_until_the_last_trajectory(self) -> None:
        """Without a structure to backfill with, the batch narrows instead of stopping."""
        strategy = _make_relaxation_strategy(
            convergence_hook=_make_scripted_criterion(),
            num_steps=8,
            training_steps_per_segment=2,
            generation_steps=2,
        )
        probe = _StateProbe()
        strategy.on_policy.dynamics.register_hook(
            _ScriptedRelaxation({0: 1, 1: 3, 2: 5})
        )
        strategy.on_policy.dynamics.register_hook(probe)

        with pytest.warns(UserWarning, match="nothing left to start a fresh one"):
            strategy.run()

        assert probe.graph_counts == [3, 3, 2, 2, 1, 1]
        assert probe.state_rows[2:] == probe.graph_counts[2:]
        assert strategy.step_count == 8


class TestRelaxationBackfill:
    def test_the_lifecycle_backfills_from_the_configured_structures(self) -> None:
        """The config's own InitialStructures serves the backfill, not a copy of it."""
        strategy = _make_relaxation_strategy(fmax=1e3)
        state = strategy.on_policy.initial_structures.initial_batch()

        with _relaxation_lifecycle(strategy.on_policy, state) as lifecycle:
            assert lifecycle.structures is strategy.on_policy.initial_structures

    def test_the_backfill_opens_where_the_initial_batch_left_the_cursor(self) -> None:
        """One cursor seeds and backfills, so no row is propagated twice."""
        strategy = _make_relaxation_strategy(
            fmax=1e3,
            structures=InitialStructures(
                _build_initial_dataset(n_systems=3), max_batch_size=1
            ),
        )
        state = strategy.on_policy.initial_structures.initial_batch()

        with _relaxation_lifecycle(strategy.on_policy, state) as lifecycle:
            replacements = lifecycle.structures.draw(limit=3)

            assert state.num_graphs == 1
            assert len(replacements) == 2
            assert lifecycle.structures.exhausted is True

    def test_a_backfill_draws_for_the_room_the_graduates_freed(self) -> None:
        """One four-atom graduation passes over an eight-atom row for the next four."""
        structures = InitialStructures(
            _make_sized_dataset([4, 4, 4, 8, 4]), max_batch_size=3
        )
        strategy = _make_relaxation_strategy(
            convergence_hook=_make_scripted_criterion(),
            num_steps=4,
            generation_steps=2,
            structures=structures,
        )
        probe = _StateProbe()
        strategy.on_policy.dynamics.register_hook(_ScriptedRelaxation({0: 1}))
        strategy.on_policy.dynamics.register_hook(probe)

        strategy.run()

        assert probe.systems[-1] == [1, 2, 3]
        assert probe.node_counts[-1] == 12
        assert (structures.cursor, structures.next_system_id) == (5, 4)


class TestRelaxationRestartExactness:
    def _make_run(self, num_steps: int) -> DistillationStrategy:
        """Return a graduating relaxation over six structures, two trajectories wide."""
        return _make_relaxation_strategy(
            fmax=1e3,
            num_steps=num_steps,
            training_steps_per_segment=2,
            generation_steps=2,
            structures=InitialStructures(
                _build_initial_dataset(n_systems=6), max_batch_size=2
            ),
        )

    def test_a_run_restored_mid_refill_backfills_the_same_rows(self) -> None:
        """The cursor is state, so a restart continues the deal instead of rewinding."""
        unbroken = self._make_run(6)
        whole = _StateProbe()
        unbroken.on_policy.dynamics.register_hook(whole)
        with pytest.warns(UserWarning, match="nothing left to start a fresh one"):
            unbroken.run()

        interrupted = self._make_run(2)
        interrupted.run()
        bundle = interrupted.on_policy.initial_structures.state_dict()

        resumed = self._make_run(4)
        tail = _StateProbe()
        resumed.on_policy.dynamics.register_hook(tail)
        with patch.object(
            InitialStructures,
            "shard",
            autospec=True,
            side_effect=_shard_then_restore(bundle),
        ):
            with pytest.warns(UserWarning, match="nothing left to start a fresh one"):
                resumed.run()

        assert bundle == {
            "cursor": 4,
            "wraps": 0,
            "next_system_id": 4,
            "rank": 0,
            "world_size": 1,
        }
        assert sorted(tail.first_positions) == [4, 5]
        assert tail.first_positions == {
            system: whole.first_positions[system] for system in (4, 5)
        }


class TestRelaxationLifecycleOwnership:
    def test_a_propagator_carrying_its_own_migrator_is_rejected(self) -> None:
        """A second migrator graduates structures neither capture route stores."""
        strategy = _make_relaxation_strategy(fmax=0.05, num_steps=2)
        strategy.on_policy.dynamics.register_hook(
            ConvergenceHook.from_fmax(1e3, source_status=0, target_status=1)
        )

        with pytest.raises(ValueError, match="no other status-migrating"):
            strategy.run()

    def test_a_migrating_detector_on_the_propagator_is_rejected(self) -> None:
        """A criterion the propagator graduates on is not swapped out unsaid."""
        strategy = _make_relaxation_strategy(fmax=0.05, num_steps=2)
        strategy.on_policy.dynamics.convergence_hook = ConvergenceHook.from_fmax(
            1e3, source_status=0, target_status=1
        )

        with pytest.raises(ValueError, match="no other status-migrating"):
            strategy.run()

    def test_a_reporting_detector_is_replaced_and_handed_back(self) -> None:
        """Only migration competes; a plain detector is swapped as documented."""
        detector = ConvergenceHook.from_fmax(1e3)
        strategy = _make_relaxation_strategy(fmax=1e-6, num_steps=2)
        strategy.on_policy.dynamics.convergence_hook = detector

        strategy.run()

        assert strategy.on_policy.dynamics.convergence_hook is detector
        assert len(strategy.replay_buffer) > 0

    def test_a_criterion_migrating_off_an_unseeded_status_is_rejected(self) -> None:
        """A hook aimed elsewhere freezes nothing and graduates nothing."""
        strategy = _make_relaxation_strategy(
            convergence_hook=ConvergenceHook.from_fmax(
                0.05, source_status=1, target_status=2
            ),
            num_steps=2,
        )

        with pytest.raises(ValueError, match="off the status its initial structure"):
            strategy.run()

    def test_a_fused_sub_stage_criterion_is_rejected(self) -> None:
        """FusedStage turns a sub-stage criterion into a migrator of its own."""
        student = _build_demo_model()
        strategy = _make_relaxation_strategy(
            fmax=1e-6,
            student=student,
            num_steps=2,
            config_overrides={
                "dynamics": FusedStage(
                    sub_stages=[
                        (
                            0,
                            FIRE(
                                student,
                                dt=0.1,
                                convergence_hook=ConvergenceHook.from_fmax(1e3),
                            ),
                        )
                    ]
                )
            },
        )

        with pytest.raises(
            ValueError, match="no other status-migrating ConvergenceHook"
        ):
            strategy.run()

    def test_a_multi_sub_stage_fused_propagator_is_rejected(self) -> None:
        """Sub-stage shape is fixed at construction, so it is refused there."""
        student = _build_demo_model()

        with pytest.raises(
            ValidationError, match="no other status-migrating ConvergenceHook"
        ):
            _make_relaxation_strategy(
                fmax=1e-6,
                student=student,
                num_steps=2,
                config_overrides={
                    "dynamics": FusedStage(
                        sub_stages=[
                            (0, FIRE(student, dt=0.1)),
                            (1, NVE(student, dt=0.1)),
                        ]
                    )
                },
            )

    def test_a_fused_level_migrator_is_rejected(self) -> None:
        """A migrator registered on the fused stage itself competes as well."""
        student = _build_demo_model()
        propagator = FusedStage(sub_stages=[(0, FIRE(student, dt=0.1))])
        propagator.register_hook(
            ConvergenceHook.from_fmax(1e3, source_status=0, target_status=1)
        )
        strategy = _make_relaxation_strategy(
            fmax=1e-6,
            student=student,
            num_steps=2,
            config_overrides={"dynamics": propagator},
        )

        with pytest.raises(
            ValueError, match="no other status-migrating ConvergenceHook"
        ):
            strategy.run()

    def test_a_propagator_carrying_its_own_sampler_is_rejected(self) -> None:
        """A mid-run refill compacts the batch under the capture's bookkeeping."""
        student = _build_demo_model()
        strategy = _make_relaxation_strategy(
            fmax=0.05,
            student=student,
            num_steps=2,
            config_overrides={
                "dynamics": FusedStage(
                    sub_stages=[(0, FIRE(student, dt=0.1))],
                    sampler=SizeAwareSampler(
                        _build_initial_dataset(n_systems=3),
                        max_atoms=64,
                        max_batch_size=3,
                    ),
                )
            },
        )

        with pytest.raises(ValueError, match="carry no sampler of its own"):
            strategy.run()

    def test_a_propagator_sampler_is_left_alone_without_a_lifecycle(self) -> None:
        """Nothing is owned where no lifecycle is installed, so nothing is refused."""
        student = _build_demo_model()
        strategy = _make_relaxation_strategy(
            fmax=None,
            student=student,
            num_steps=2,
            generation_steps=4,
            config_overrides={
                "dynamics": FusedStage(
                    sub_stages=[(0, FIRE(student, dt=0.1))],
                    sampler=SizeAwareSampler(
                        _build_initial_dataset(n_systems=6),
                        max_atoms=64,
                        max_batch_size=3,
                    ),
                )
            },
        )

        strategy.run()

        assert len(strategy.replay_buffer) == 4 * 3


class TestUnmanagedGeneration:
    def test_an_unmanaged_run_captures_every_frame(self) -> None:
        """Structures enter on status 0 and nothing migrates it, so nothing is filtered."""
        strategy = _make_relaxation_strategy(fmax=None, num_steps=2, generation_steps=4)
        probe = _StatusProbe()
        strategy.on_policy.dynamics.register_hook(probe)

        strategy.run()

        assert probe.statuses == [[0, 0, 0]] * 4
        assert len(strategy.replay_buffer) == 4 * 3

    def test_a_status_carrying_run_captures_every_moving_frame(self) -> None:
        """A fused stage stamps a status the unmanaged path must read as active."""
        student = _build_demo_model()
        strategy = _make_relaxation_strategy(
            fmax=None,
            student=student,
            num_steps=2,
            generation_steps=4,
            config_overrides={
                "dynamics": FusedStage(sub_stages=[(0, FIRE(student, dt=0.1))])
            },
        )
        probe = _StatusProbe()
        strategy.on_policy.dynamics.register_hook(probe)

        strategy.run()

        assert probe.statuses == [[0, 0, 0]] * 4
        assert len(strategy.replay_buffer) == 4 * 3

    def test_a_budgeted_sub_stage_ends_the_chunk_when_it_graduates(self) -> None:
        """The budget graduates the batch after two steps, both of them stored."""
        student = _build_demo_model()
        strategy = _make_relaxation_strategy(
            fmax=None,
            student=student,
            num_steps=2,
            generation_steps=4,
            config_overrides={
                "dynamics": FusedStage(
                    sub_stages=[(0, FIRE(student, dt=0.1, n_steps=2))]
                )
            },
        )

        strategy.run()

        assert len(strategy.replay_buffer) == 2 * 3

    def test_a_propagator_managing_its_own_convergence_keeps_its_final_frames(
        self,
    ) -> None:
        """Without a lifecycle nothing else stores a graduated graph, so this route does."""
        strategy = _make_relaxation_strategy(fmax=None, num_steps=2, generation_steps=4)
        strategy.on_policy.dynamics.register_hook(
            ConvergenceHook.from_fmax(1e6, source_status=0, target_status=1)
        )
        probe = _StatusProbe()
        strategy.on_policy.dynamics.register_hook(probe)

        strategy.run()

        assert probe.statuses[0] == [1, 1, 1]
        assert strategy.step_count == 2
        assert len(strategy.replay_buffer) == 4 * 3

    def test_a_fused_sub_stage_criterion_keeps_its_final_frames_unmanaged(
        self,
    ) -> None:
        """A sub-stage criterion graduating on the first step still fills the buffer."""
        student = _build_demo_model()
        strategy = _make_relaxation_strategy(
            fmax=None,
            student=student,
            num_steps=2,
            generation_steps=4,
            config_overrides={
                "dynamics": FusedStage(
                    sub_stages=[
                        (
                            0,
                            FIRE(
                                student,
                                dt=0.1,
                                convergence_hook=ConvergenceHook.from_fmax(1e6),
                            ),
                        )
                    ]
                )
            },
        )

        strategy.run()

        assert strategy.step_count == 2
        assert len(strategy.replay_buffer) == 3


class TestRelaxationDivergence:
    def _run_diverging(
        self, *, at_step: int, **kwargs: Any
    ) -> tuple[DistillationStrategy, _StateProbe, _StatusProbe]:
        """Run a managed relaxation whose first system diverges at *at_step*."""
        strategy = _make_relaxation_strategy(
            convergence_hook=_make_scripted_criterion(),
            structures=InitialStructures(
                _build_initial_dataset(n_systems=3), recycle=True
            ),
            num_steps=4,
            generation_steps=4,
            **kwargs,
        )
        opening = _StateProbe()
        status = _StatusProbe()
        strategy.on_policy.dynamics.register_hook(_ScriptedRelaxation({}))
        strategy.on_policy.dynamics.register_hook(_NaNInjector(0, at_step))
        strategy.on_policy.dynamics.register_hook(opening)
        strategy.on_policy.dynamics.register_hook(status)
        with pytest.warns(UserWarning, match="1 of 3 generated trajectories diverged"):
            strategy.run()
        return strategy, opening, status

    def test_a_diverged_trajectory_is_frozen_retired_and_backfilled(self) -> None:
        """NaN forces end the trajectory the way convergence does, minus the capture.

        The probe is registered ahead of the lifecycle's hooks, so it reads the
        frozen status from the step after the divergence on.
        """
        strategy, opening, status = self._run_diverging(at_step=2)

        assert status.statuses[2] == [0, 0, 0]
        assert status.statuses[3] == [1, 0, 0]
        assert opening.systems[4] == [1, 2, 3]
        assert strategy.step_count == 4

    def test_nothing_a_diverged_trajectory_produced_reaches_the_buffer(self) -> None:
        """Both capture routes skip the frozen graph, so every stored frame is finite."""
        strategy, _, _ = self._run_diverging(at_step=2)

        frames = strategy.replay_buffer.dataset.in_memory_batch
        assert len(strategy.replay_buffer) == 3 + 3 + 2 + 2 + 4 * 3
        assert bool(torch.isfinite(frames.positions).all())
        assert bool(torch.isfinite(frames.teacher_forces).all())

    def test_a_budget_graduate_with_a_non_finite_state_is_not_captured(self) -> None:
        """The converged route reads finiteness, not only the status transition.

        The cadence stores the whole finite frame of step 0, and the budget
        graduates the batch on step 1, where only the two finite structures are
        written.
        """
        student = _build_demo_model()
        strategy = _make_relaxation_strategy(
            fmax=1e-9,
            student=student,
            num_steps=2,
            generation_steps=4,
            label_frequency=100,
            config_overrides={
                "dynamics": FusedStage(
                    sub_stages=[(0, FIRE(student, dt=0.1, n_steps=2))]
                )
            },
        )
        strategy.on_policy.dynamics.register_hook(_NaNInjector(0, 1))

        with pytest.warns(UserWarning, match="1 of 3 generated trajectories diverged"):
            strategy.run()

        frames = strategy.replay_buffer.dataset.in_memory_batch
        assert len(strategy.replay_buffer) == 3 + 2
        assert bool(torch.isfinite(frames.positions).all())

    def test_the_lifecycle_defaults_to_the_non_finite_predicate(self) -> None:
        """Without a ``divergence`` setting every route reads the built-in."""
        strategy = _make_relaxation_strategy(fmax=1e3)
        config = strategy.on_policy
        state = config.initial_structures.initial_batch()

        assert config.divergence is None
        with _relaxation_lifecycle(config, state) as lifecycle:
            assert lifecycle.divergence is nonfinite_divergence
            assert lifecycle.capture.divergence is nonfinite_divergence
            assert any(
                getattr(hook, "divergence", None) is nonfinite_divergence
                and not isinstance(hook, type(lifecycle.capture))
                for hook in config.dynamics.hooks
            )

    def test_a_custom_predicate_freezes_the_graphs_it_flags(self) -> None:
        """A finite trajectory the predicate flags ends the way a NaN one does.

        The predicate flags the source's first structure from the first step,
        so it is frozen on step 0, stored by neither route, and retired and
        backfilled at the boundary, with the warning naming the predicate.
        """
        strategy = _make_relaxation_strategy(
            convergence_hook=_make_scripted_criterion(),
            structures=InitialStructures(
                _build_initial_dataset(n_systems=3), recycle=True
            ),
            num_steps=4,
            generation_steps=4,
            config_overrides={"divergence": _diverge_first_system},
        )
        opening = _StateProbe()
        status = _StatusProbe()
        strategy.on_policy.dynamics.register_hook(_ScriptedRelaxation({}))
        strategy.on_policy.dynamics.register_hook(opening)
        strategy.on_policy.dynamics.register_hook(status)

        with pytest.warns(
            UserWarning, match="1 of 3 .*diverged: the divergence predicate .*flagged"
        ):
            strategy.run()

        assert status.statuses[0] == [0, 0, 0]
        assert status.statuses[1] == [1, 0, 0]
        assert opening.systems[4] == [1, 2, 3]
        assert len(strategy.replay_buffer) == 4 * 2 + 4 * 3

    @pytest.mark.parametrize(
        ("predicate", "message"),
        [
            (
                _diverge_per_atom,
                r"got shape=\(\d+,\) of dtype torch.bool, expected \(3,\)",
            ),
            (_diverge_as_floats, r"got shape=\(3,\) of dtype torch.float32"),
        ],
        ids=["per_atom", "float_mask"],
    )
    def test_a_predicate_returning_the_wrong_mask_is_refused(
        self, predicate: Any, message: str
    ) -> None:
        """Anything but one boolean per graph is refused naming what came back."""
        strategy = _make_relaxation_strategy(
            fmax=1e-9, num_steps=2, config_overrides={"divergence": predicate}
        )

        with pytest.raises(ValueError, match=message):
            strategy.run()

    def test_a_predicate_returning_no_tensor_is_refused(self) -> None:
        """A list of flags is not the tensor the lifecycle masks status with."""
        strategy = _make_relaxation_strategy(
            fmax=1e-9, num_steps=2, config_overrides={"divergence": _diverge_as_list}
        )

        with pytest.raises(TypeError, match="got 'list'"):
            strategy.run()


class TestNonfiniteDivergence:
    def test_a_non_finite_position_flags_its_graph_alone(self) -> None:
        """One NaN coordinate marks the graph holding it and no other."""
        batch = _build_propagator_batch(_INITIAL_ELEMENT, 3, base_seed=500)
        second = torch.where(batch.batch_idx.long() == 1)[0][0]
        batch.positions[second, 0] = float("nan")

        assert nonfinite_divergence(batch).tolist() == [False, True, False]

    def test_an_infinite_force_flags_its_graph(self) -> None:
        """Forces count when the frame carries them."""
        batch = _build_propagator_batch(_INITIAL_ELEMENT, 3, base_seed=500)
        last = torch.where(batch.batch_idx.long() == 2)[0][-1]
        batch.forces[last, 2] = float("inf")

        assert nonfinite_divergence(batch).tolist() == [False, False, True]

    def test_a_frame_without_forces_is_judged_on_positions(self) -> None:
        """A store kept without predictions still gets a per-graph verdict."""
        batch = _build_propagator_batch(
            _INITIAL_ELEMENT, 2, base_seed=500, predictions=False
        )

        flags = nonfinite_divergence(batch)

        assert flags.dtype == torch.bool
        assert flags.tolist() == [False, False]


class TestTeacherLabelHookExitStatus:
    def _make_statused_batch(self) -> Batch:
        """Return three structures, the last one frozen at status 1."""
        batch = _build_propagator_batch(_INITIAL_ELEMENT, 3, base_seed=500)
        batch["status"] = torch.tensor([[0], [0], [1]], dtype=torch.long)
        return batch

    def test_a_hook_given_the_exit_status_leaves_graduated_graphs_out(self) -> None:
        """The frozen graph is neither scored nor stored by this route."""
        sink = HostMemory(capacity=3)
        hook = TeacherLabelHook(
            InProcessTeacherScorer(_build_direct_force_teacher(), ("energy",)),
            sink=sink,
            exit_status=1,
        )

        hook._label_frame(self._make_statused_batch(), 0)

        assert sink.drain().num_graphs == 2

    def test_a_hook_without_one_captures_every_graph(self) -> None:
        """A status the hook was told nothing about does not filter the copy."""
        sink = HostMemory(capacity=3)
        hook = TeacherLabelHook(
            InProcessTeacherScorer(_build_direct_force_teacher(), ("energy",)),
            sink=sink,
        )

        hook._label_frame(self._make_statused_batch(), 0)

        assert sink.drain().num_graphs == 3


class TestRelaxationCapture:
    def test_the_converged_frames_are_labeled_when_the_sink_is_drained(self) -> None:
        """Deferred labeling scores a segment's graduates in one teacher pass.

        The two path passes after the graduation cover the two structures still
        relaxing and not the frozen third, which the last pass — the drained
        converged sink — scores once instead.
        """
        strategy = _make_relaxation_strategy(
            convergence_hook=_make_scripted_criterion(), num_steps=2, generation_steps=4
        )
        strategy.on_policy.dynamics.register_hook(_ScriptedRelaxation({0: 2}))
        scorer = strategy.on_policy.teacher_scorer

        with patch.object(scorer, "label", wraps=scorer.label) as spy:
            strategy.run()

        scored = [call.args[0].num_graphs for call in spy.call_args_list]
        assert scored == [3, 3, 2, 2, 1]

    def test_an_all_frozen_segment_tail_costs_no_teacher_pass(self) -> None:
        """A batch that graduates whole pays one path pass and one drain pass.

        The step every trajectory converges on, and the forced dispatch that
        closes the segment behind it, find nothing left moving to score.
        """
        strategy = _make_relaxation_strategy(
            convergence_hook=_make_scripted_criterion(),
            num_steps=2,
            generation_steps=4,
            structures=InitialStructures(
                _build_initial_dataset(n_systems=3), recycle=True
            ),
        )
        strategy.on_policy.dynamics.register_hook(
            _ScriptedRelaxation({0: 1, 1: 1, 2: 1})
        )
        scorer = strategy.on_policy.teacher_scorer

        with patch.object(scorer, "label", wraps=scorer.label) as spy:
            strategy.run()

        scored = [call.args[0].num_graphs for call in spy.call_args_list]
        assert scored == [3, 3]

    def _stored_frames(self, *, fused: bool) -> int:
        """Run one scripted relaxation through a bare or fused FIRE and count frames."""
        student = _build_demo_model()
        propagator: BaseDynamics = FIRE(student, dt=0.1)
        if fused:
            propagator = FusedStage(sub_stages=[(0, propagator)])
        strategy = _make_relaxation_strategy(
            convergence_hook=_make_scripted_criterion(),
            student=student,
            num_steps=2,
            generation_steps=4,
            config_overrides={"dynamics": propagator},
        )
        strategy.on_policy.dynamics.register_hook(_ScriptedRelaxation({0: 2}))

        strategy.run()

        return len(strategy.replay_buffer)

    def test_a_fused_propagator_stores_the_frames_a_bare_one_does(self) -> None:
        """FusedStage fires no ON_CONVERGE of its own, so capture reads the status."""
        assert self._stored_frames(fused=True) == self._stored_frames(fused=False)

    def test_the_stored_frames_carry_the_replay_frame_schema(self) -> None:
        """Both capture routes strip the run and keep the teacher's labels."""
        strategy = _make_relaxation_strategy(
            convergence_hook=_make_scripted_criterion(), num_steps=2, generation_steps=4
        )
        strategy.on_policy.dynamics.register_hook(_ScriptedRelaxation({0: 2}))

        strategy.run()

        schema = strategy.replay_buffer.schema
        assert "node.teacher_forces" in schema
        assert "system.teacher_energy" in schema
        assert "node.forces" not in schema
        assert "system.energy" not in schema
        assert "system.status" not in schema
        assert "system.system_id" not in schema

    def _drain_through_the_converged_route(
        self, strategy: DistillationStrategy
    ) -> ReplayBuffer:
        """Store one raw frame in the converged sink and drain it into a fresh buffer."""
        config = strategy.on_policy
        state = config.initial_structures.initial_batch()
        buffer = ReplayBuffer()
        with _relaxation_lifecycle(config, state) as lifecycle:
            lifecycle.capture.sink.write(state.clone())
            strategy._capture_converged(config, lifecycle, buffer)
        return buffer

    def test_the_converged_route_labels_with_autocast_disabled(self) -> None:
        """A mixed-precision generation phase does not reach the teacher pass."""
        probe = _AutocastProbeScorer(
            InProcessTeacherScorer(_build_direct_force_teacher(), ("energy", "forces"))
        )
        strategy = _make_relaxation_strategy(
            fmax=1e3, config_overrides={"teacher_scorer": probe}
        )

        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            buffer = self._drain_through_the_converged_route(strategy)

        assert probe.autocast_states == [False]
        assert len(buffer) == 3

    def test_the_converged_route_refuses_a_foreign_field(self) -> None:
        """A scorer writing outside teacher_* is stopped here as on the path route."""
        with pytest.warns(UserWarning, match="declare label_fields"):
            strategy = _make_relaxation_strategy(
                fmax=1e3,
                config_overrides={"teacher_scorer": _ForeignFieldScorer()},
            )

        with pytest.raises(ValueError, match="teacher_"):
            self._drain_through_the_converged_route(strategy)

    def test_a_neighbor_list_teacher_labels_the_drained_frames(
        self, device: str
    ) -> None:
        """The deferred route's labels match a fresh scoring of the stored frames."""
        strategy = _make_relaxation_strategy(
            fmax=1e3,
            teacher=_build_lj_teacher(),
            num_steps=2,
            generation_steps=2,
            device=device,
        )

        with pytest.warns(UserWarning, match="nothing left to start a fresh one"):
            strategy.run()

        frames = strategy.replay_buffer.dataset.in_memory_batch
        assert len(strategy.replay_buffer) == 3
        assert "neighbor_matrix" not in frames
        rescored, _ = strategy.on_policy.teacher_scorer.label(
            frames.clone().to(torch.device(device))
        )["teacher_forces"]
        torch.testing.assert_close(frames.teacher_forces, rescored.cpu())
        assert float(frames.teacher_forces.abs().max()) > 0.0

    def test_both_capture_routes_store_on_one_device(self, device: str) -> None:
        """A partly converged segment feeds one buffer from both routes.

        Twelve of the thirteen frames come from the path route and one from the
        converged route, which used to land on the propagation device while the
        path route left its own in host memory.
        """
        strategy = _make_relaxation_strategy(
            fmax=0.5, num_steps=2, generation_steps=6, device=device
        )

        strategy.run()

        frames = strategy.replay_buffer.dataset.in_memory_batch
        assert len(strategy.replay_buffer) == 13
        assert frames.device.type == "cpu"

    def test_a_frozen_structure_is_stored_once_and_not_once_per_step(self) -> None:
        """The two routes partition the frames, so nothing is inserted twice."""
        strategy = _make_relaxation_strategy(
            convergence_hook=_make_scripted_criterion(), num_steps=4, generation_steps=4
        )
        strategy.on_policy.dynamics.register_hook(_ScriptedRelaxation({0: 1, 1: 2}))

        strategy.run()

        fingerprints = _frame_fingerprints(strategy)
        assert len(set(fingerprints)) == len(fingerprints)

    def test_the_path_route_drops_the_graduated_structures(self) -> None:
        """A frozen structure leaves the cadence capture the step it converges on.

        Steps 0 and 1 store three frames each, steps 2 and 3 store the two
        structures still relaxing, and the converged route stores the third.
        """
        strategy = _make_relaxation_strategy(
            convergence_hook=_make_scripted_criterion(), num_steps=2, generation_steps=4
        )
        strategy.on_policy.dynamics.register_hook(_ScriptedRelaxation({0: 2}))

        strategy.run()

        assert len(strategy.replay_buffer) == 3 + 3 + 2 + 2 + 1

    def _run_budgeted(
        self, *, n_steps: int, label_frequency: int
    ) -> tuple[DistillationStrategy, _FrameProbe]:
        """Run one segment of a sub-stage budgeted to graduate before it ends."""
        student = _build_demo_model()
        strategy = _make_relaxation_strategy(
            fmax=1e-9,
            student=student,
            num_steps=2,
            generation_steps=4,
            label_frequency=label_frequency,
            config_overrides={
                "dynamics": FusedStage(
                    sub_stages=[(0, FIRE(student, dt=0.1, n_steps=n_steps))]
                )
            },
        )
        probe = _FrameProbe()
        strategy.on_policy.dynamics.register_hook(probe)

        with pytest.warns(UserWarning, match="nothing left to start a fresh one"):
            strategy.run()

        return strategy, probe

    def test_a_budget_graduating_the_batch_is_captured_when_the_chunk_returns(
        self,
    ) -> None:
        """A budget migrates after the dispatch, so the last frame is stored late.

        The chunk ends on that step too, so no later dispatch of the segment
        reaches the frame and neither capture route used to store it.
        """
        strategy, probe = self._run_budgeted(n_steps=4, label_frequency=100)

        stored = set(_frame_fingerprints(strategy))
        assert len(strategy.replay_buffer) == 6
        assert all(fingerprint in stored for fingerprint in probe.frames[3].values())

    def test_a_cadence_that_stored_the_last_step_is_not_captured_twice(self) -> None:
        """The label hook's marker is what keeps the closing capture idempotent."""
        strategy, _ = self._run_budgeted(n_steps=4, label_frequency=1)

        fingerprints = _frame_fingerprints(strategy)
        assert len(fingerprints) == 12
        assert len(set(fingerprints)) == 12

    def test_a_mid_segment_budget_graduation_is_captured_too(self) -> None:
        """A budget that empties the batch ends the chunk wherever it lands."""
        strategy, probe = self._run_budgeted(n_steps=2, label_frequency=100)

        fingerprints = _frame_fingerprints(strategy)
        stored = set(fingerprints)
        assert all(fingerprint in stored for fingerprint in probe.frames[1].values())
        assert len(fingerprints) == len(stored)


class TestRelaxationCaptureSink:
    def test_a_configured_sink_is_sized_once_for_the_initial_batch(self) -> None:
        """A backfill never grows the batch, so the first segment's capacity holds."""
        sink = _ResizableSink(capacity=1)
        strategy = _make_relaxation_strategy(
            fmax=1e3, generation_steps=4, config_overrides={"capture_sink": sink}
        )

        with pytest.warns(UserWarning, match="generation stopped"):
            strategy.run()

        assert sink.resizes == [(4 + 1) * 3]
        assert len(sink) == 0
        assert len(strategy.replay_buffer) > 0

    def test_a_small_sink_without_resize_is_refused_under_a_lifecycle(self) -> None:
        """The refusal names the capacity the segment needs and both remedies."""
        strategy = _make_relaxation_strategy(
            fmax=1e3,
            generation_steps=4,
            config_overrides={"capture_sink": HostMemory(capacity=2)},
        )

        with pytest.raises(ValueError, match="capacity 2 without a resize method"):
            strategy.run()

    def test_a_custom_source_stamping_bookkeeping_drives_the_lifecycle(self) -> None:
        """A minimal source without budgets graduates and reports exhaustion."""
        dataset = _build_initial_dataset(n_systems=3)
        source = _ListSource([dataset[index][0] for index in range(3)])
        strategy = _make_relaxation_strategy(fmax=1e3, structures=source)

        with pytest.warns(UserWarning, match="generation stopped"):
            strategy.run()

        assert source.exhausted
        assert source.shards == [(0, 1)]
        assert len(strategy.replay_buffer) == 3

    def test_a_source_stamping_no_status_is_refused_naming_the_contract(self) -> None:
        """The lifecycle cannot graduate on a column the initial batch lacks."""
        dataset = _build_initial_dataset(n_systems=3)
        source = _ListSource([dataset[index][0] for index in range(3)])
        source.initial_batch = lambda: Batch.from_data_list(source.structures)
        strategy = _make_relaxation_strategy(fmax=1e3, structures=source)

        with pytest.raises(ValueError, match="stamps status zeros and system_ids"):
            strategy.run()


class TestRelaxationEndToEnd:
    def test_a_relaxation_run_trains_on_the_paths_it_generated(
        self, device: str
    ) -> None:
        """Student-driven relaxations reach the step target and lower the loss."""
        recorder = _RecordingBatchHook()
        strategy = _make_relaxation_strategy(
            fmax=0.05,
            num_steps=12,
            training_steps_per_segment=4,
            generation_steps=3,
            replay_ratio=0.5,
            device=device,
            hooks=[recorder],
        )

        strategy.run()

        assert strategy.step_count == 12
        assert strategy.epoch_count == 3
        assert len(strategy.replay_buffer) == 9 * 3
        assert sum(recorder.losses[-4:]) < sum(recorder.losses[:4])

    def test_converged_frames_mix_with_the_reference_dataset_like_path_frames(
        self,
    ) -> None:
        """The deferred route's frames collate with the reference dataset too."""
        strategy = _make_relaxation_strategy(
            fmax=1e3,
            num_steps=4,
            training_steps_per_segment=2,
            generation_steps=2,
            replay_ratio=0.5,
        )

        with pytest.warns(UserWarning, match="nothing left to start a fresh one"):
            strategy.run()

        assert strategy.step_count == 4
        assert len(strategy.replay_buffer) == 3

    def test_a_budgeted_source_run_backfills_from_its_remainder(self) -> None:
        """A budgeted source serves the backfill from the rows it did not pack."""
        structures = InitialStructures(
            _build_initial_dataset(n_systems=4), max_atoms=64, max_batch_size=2
        )
        strategy = _make_relaxation_strategy(
            fmax=1e3,
            num_steps=4,
            training_steps_per_segment=2,
            generation_steps=2,
            structures=structures,
        )
        probe = _StateProbe()
        strategy.on_policy.dynamics.register_hook(probe)

        with pytest.warns(UserWarning, match="nothing left to start a fresh one"):
            strategy.run()

        assert probe.systems == [[0, 1], [2, 3]]
        assert strategy.step_count == 4
