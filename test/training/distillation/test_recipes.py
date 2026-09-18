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
"""Tests for recipe serialization and restart of the on-policy segment loop."""

from __future__ import annotations

import json
import warnings
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

import pytest
import torch

from nvalchemi.data import AtomicData, Batch
from nvalchemi.data.datapipes.backends.zarr import AtomicDataZarrReader
from nvalchemi.data.datapipes.dataset import Dataset
from nvalchemi.data.datapipes.in_memory_dataset import InMemoryDataset
from nvalchemi.data.datapipes.multidataset import MultiDataset
from nvalchemi.dynamics.base import BaseDynamics, DynamicsStage
from nvalchemi.dynamics.demo import DemoDynamics
from nvalchemi.dynamics.integrators.nvt_langevin import NVTLangevin
from nvalchemi.dynamics.optimizers.fire import FIRE
from nvalchemi.dynamics.sampler import SizeAwareSampler
from nvalchemi.dynamics.sinks import HostMemory
from nvalchemi.hooks import NeighborListHook, TrainContext
from nvalchemi.models.base import BaseModelMixin, NeighborConfig
from nvalchemi.training import (
    EnergyMSELoss,
    ForceMSELoss,
    OptimizerConfig,
    TrainingStage,
)
from nvalchemi.training._checkpoint import _snapshot_hook_states
from nvalchemi.training.distillation import (
    DistillationStrategy,
    InitialStructures,
    InProcessTeacherScorer,
    OnPolicyConfig,
    OnPolicySettings,
    TeacherLabelHook,
    label_dataset,
)
from nvalchemi.training.distillation._restart import (
    _batch_from_state,
    _batch_state,
    _OnPolicyRestartHook,
)
from nvalchemi.training.distillation.config import (
    _dynamics_from_spec_dict,
    _dynamics_spec_dict,
    _on_policy_settings,
)
from nvalchemi.training.distillation.replay import ReplayBuffer
from nvalchemi.training.hooks import CheckpointHook
from test.training.conftest import _build_demo_model
from test.training.distillation.conftest import (
    _build_direct_force_teacher,
    _build_small_dataset,
    _ListSource,
)

if TYPE_CHECKING:
    from nvalchemi.models.base import BaseModelMixin as _TypeCheckedModel

_ATOMS_PER_SYSTEM = 4
"""Atoms in every synthetic system, so the segments stay small."""

_SEED_ELEMENT = 1
"""Atomic number tagging the structures the propagator generates from."""

_REFERENCE_ELEMENT = 6
"""Atomic number tagging the structures the reference dataset supplies."""

_SEGMENT_STEPS = 3
"""Propagator steps one segment generates, as every recipe here sets it."""

_LANGEVIN = {
    "cls_path": "nvalchemi.dynamics.integrators.nvt_langevin.NVTLangevin",
    "kwargs": {
        "dt": 0.5,
        "temperature": 300.0,
        "friction": 0.01,
        "random_seed": 7,
    },
}
"""Propagator reference every recipe here builds its segment loop from."""

_RECIPE_OBJECT_KEYS = frozenset({"dynamics", "teacher_scorer", "initial_structures"})
"""Recipe entries describing a live object rather than a scalar setting."""


def _make_system(
    atomic_number: int,
    seed: int,
    *,
    predictions: bool = False,
    n_atoms: int = _ATOMS_PER_SYSTEM,
) -> AtomicData:
    """Return one system of *n_atoms* atoms tagged by *atomic_number*.

    ``predictions=True`` carries the ``energy`` and ``forces`` a propagator
    reads on its first step and the labeling hook strips again, which is the
    shape an initial structure has and a replay frame — and so the reference
    dataset — has not.
    """
    generator = torch.Generator().manual_seed(seed)
    predicted = (
        {"energy": torch.zeros(1, 1), "forces": torch.zeros(n_atoms, 3)}
        if predictions
        else {}
    )
    return AtomicData(
        positions=torch.randn(n_atoms, 3, generator=generator),
        atomic_numbers=torch.full((n_atoms,), atomic_number, dtype=torch.long),
        atomic_masses=torch.ones(n_atoms),
        **predicted,
    )


def _make_batch(
    atomic_number: int,
    n_systems: int,
    base_seed: int,
    *,
    predictions: bool = False,
    sizes: Sequence[int] | None = None,
) -> Batch:
    """Return a batch of systems all tagged by *atomic_number*.

    ``sizes`` gives the atom count of every system in turn, for a store whose
    rows are not all one width; it stands in for *n_systems*, which sizes a
    batch of uniform ones.
    """
    widths = list(sizes) if sizes is not None else [_ATOMS_PER_SYSTEM] * n_systems
    return Batch.from_data_list(
        [
            _make_system(
                atomic_number,
                base_seed + index,
                predictions=predictions,
                n_atoms=width,
            )
            for index, width in enumerate(widths)
        ]
    )


def _make_scorer(teacher: BaseModelMixin) -> InProcessTeacherScorer:
    """Return an energy-and-forces scorer over *teacher*."""
    return InProcessTeacherScorer(teacher, ("energy", "forces"))


def _make_store(
    store: Path,
    scorer: InProcessTeacherScorer,
    element: int,
    n_systems: int,
    seed: int,
    *,
    predictions: bool = False,
    sizes: Sequence[int] | None = None,
) -> Dataset:
    """Return a teacher-labeled Zarr store a recipe can name by path."""
    label_dataset(
        InMemoryDataset(
            in_memory_batch=_make_batch(
                element, n_systems, seed, predictions=predictions, sizes=sizes
            )
        ),
        scorer,
        store,
        batch_size=4,
    )
    return Dataset(reader=AtomicDataZarrReader(store), device="cpu")


def _initial_structures_spec(seed_store: Path, **budget: Any) -> dict[str, Any]:
    """Return the recipe entry naming *seed_store*, under an optional budget."""
    return {
        "dataset": {"path": str(seed_store), "device": "cpu"},
        "max_atoms": None,
        "max_edges": None,
        "max_batch_size": None,
        "recycle": False,
        **budget,
    }


def _make_recipe(seed_store: Path, **overrides: Any) -> dict[str, Any]:
    """Return an on-policy recipe seeded from *seed_store*."""
    recipe: dict[str, Any] = {
        "dynamics": json.loads(json.dumps(_LANGEVIN)),
        "teacher_scorer": {
            "teacher": "teacher",
            "signals": ["energy", "forces"],
            "dtype": None,
            "probe_seed": None,
        },
        "initial_structures": _initial_structures_spec(seed_store),
        "replay_ratio": 0.5,
        "training_steps_per_segment": 2,
        "batch_size": 4,
        "generation_steps": _SEGMENT_STEPS,
        "label_frequency": 1,
        "replay_capacity": None,
        "replay_eviction": "fifo",
        "replay_device": None,
        "seed": 0,
        "fmax": None,
        "weight_sync_frequency": 1,
    }
    recipe.update(overrides)
    return recipe


def _make_config(
    tmp_path: Path, student: BaseModelMixin, teacher: BaseModelMixin
) -> OnPolicyConfig:
    """Return a segment loop built from a recipe, so it remembers its reference."""
    seed_store = tmp_path / "seeds.zarr"
    _make_store(
        seed_store, _make_scorer(teacher), _SEED_ELEMENT, 4, 500, predictions=True
    )
    return OnPolicyConfig.from_spec_dict(
        _make_recipe(seed_store), student=student, teacher=teacher
    )


def _make_strategy(
    tmp_path: Path,
    *,
    student: BaseModelMixin,
    teacher: BaseModelMixin,
    num_steps: int,
    hooks: list[Any] | None = None,
    distributed_manager: Any = None,
    reference_dataset: Any = None,
    **recipe_overrides: Any,
) -> DistillationStrategy:
    """Return an on-policy strategy whose segment loop came from a recipe.

    ``recipe_overrides`` reach the recipe verbatim, so a caller can vary one
    setting of the shared loop; ``reference_dataset`` replaces the single
    store the mixture draws its reference share from by default.
    """
    scorer = _make_scorer(teacher)
    seed_store = tmp_path / "seeds.zarr"
    reference_store = tmp_path / "reference.zarr"
    if not seed_store.exists():
        _make_store(seed_store, scorer, _SEED_ELEMENT, 4, 500, predictions=True)
        _make_store(reference_store, scorer, _REFERENCE_ELEMENT, 8, 700)
    return DistillationStrategy(
        models={"student": student, "teacher": teacher},
        optimizer_configs={
            "student": [
                OptimizerConfig(
                    optimizer_cls=torch.optim.Adam, optimizer_kwargs={"lr": 1e-2}
                )
            ]
        },
        loss_fn=EnergyMSELoss(target_key="teacher_energy")
        + ForceMSELoss(target_key="teacher_forces", normalize_by_atom_count=True),
        num_steps=num_steps,
        hooks=list(hooks or []),
        distributed_manager=distributed_manager,
        reference_dataset=reference_dataset
        if reference_dataset is not None
        else Dataset(reader=AtomicDataZarrReader(reference_store), device="cpu"),
        on_policy=OnPolicyConfig.from_spec_dict(
            _make_recipe(seed_store, **recipe_overrides),
            student=student,
            teacher=teacher,
        ),
    )


def _make_dynamics_spec(
    propagator_cls: type[BaseDynamics], **kwargs: Any
) -> dict[str, Any]:
    """Return the propagator reference a recipe names *propagator_cls* by."""
    return {
        "cls_path": f"{propagator_cls.__module__}.{propagator_cls.__qualname__}",
        "kwargs": kwargs,
    }


def _make_neighbor_batch(n_systems: int = 3) -> Batch:
    """Return a batch whose graphs each carry a three-edge neighbor list.

    Batching offsets the per-graph indices into batch-global ones, which is the
    shape a restart bundle must not carry.
    """
    return Batch.from_data_list(
        [
            AtomicData(
                positions=torch.randn(3, 3),
                atomic_numbers=torch.full((3,), _SEED_ELEMENT, dtype=torch.long),
                atomic_masses=torch.ones(3),
                neighbor_list=torch.tensor([[0, 1], [1, 2], [2, 0]]),
            )
            for _ in range(n_systems)
        ]
    )


def _make_defragged_batch() -> Batch:
    """Return a batch holding two graphs in storage sized for four.

    :meth:`~nvalchemi.data.Batch.defrag` compacts the kept graphs to the front
    of the buffer it was allocated at, so the segment lengths describe fewer
    rows than the tensors hold.
    """
    batch = _make_batch(_SEED_ELEMENT, 4, 900)
    batch.defrag(copied_mask=torch.tensor([False, False, True, True]))
    return batch


def _restart_hook(strategy: DistillationStrategy) -> _OnPolicyRestartHook:
    """Return the internal restart hook a strategy's validator installed."""
    return next(
        hook for hook in strategy.hooks if isinstance(hook, _OnPolicyRestartHook)
    )


class _FakeWorld:
    """Distributed manager stand-in reporting a fixed world size."""

    def __init__(self, world_size: int) -> None:
        """Report *world_size* ranks, all of them rank zero's process."""
        self.world_size = world_size
        self.rank = 0
        self.global_rank = 0
        self.local_rank = 0
        self.device = torch.device("cpu")

    def is_initialized(self) -> bool:
        """Report the group as initialized once there is more than one rank."""
        return self.world_size > 1


class _EpochBoundaryHook:
    """Count the epoch-open and epoch-close dispatches of a segment loop."""

    frequency = 1

    def __init__(self, stage: TrainingStage) -> None:
        """Count dispatches of *stage*."""
        self.stage = stage
        self.calls = 0

    def __call__(self, ctx: TrainContext, stage: TrainingStage) -> None:  # noqa: ARG002
        """Add one to the tally."""
        self.calls += 1


class _ScalarPropagator(BaseDynamics):
    """Propagator keeping every constructor argument on a same-named attribute."""

    def __init__(
        self,
        model: BaseModelMixin,
        dt: float = 0.25,
        precision: torch.dtype = torch.float32,
        autocast: torch.dtype | None = None,
        staging: torch.device = torch.device("cpu"),
    ) -> None:
        """Record the settings a recipe carries."""
        super().__init__(model=model)
        self.dt = dt
        self.precision = precision
        self.autocast = autocast
        self.staging = staging

    def pre_update(self, batch: Batch) -> Batch:
        """Return *batch* untouched."""
        return batch

    def post_update(self, batch: Batch) -> Batch:
        """Return *batch* untouched."""
        return batch


class _TypeCheckedPropagator(BaseDynamics):
    """Propagator annotating its model the way every shipped one does.

    ``_TypeCheckedModel`` is imported under ``TYPE_CHECKING`` alone, exactly as
    :mod:`nvalchemi.dynamics` imports ``BaseModelMixin``, so resolving this
    constructor's string annotations raises :exc:`NameError` at runtime. Apart
    from that one import this is :class:`_ScalarPropagator`, which is what
    makes the pair pin the annotation and nothing else.
    """

    def __init__(
        self,
        model: _TypeCheckedModel,
        dt: float = 0.25,
        precision: torch.dtype = torch.float32,
        autocast: torch.dtype | None = None,
        staging: torch.device = torch.device("cpu"),
    ) -> None:
        """Record the settings a recipe carries."""
        super().__init__(model=model)
        self.dt = dt
        self.precision = precision
        self.autocast = autocast
        self.staging = staging

    def pre_update(self, batch: Batch) -> Batch:
        """Return *batch* untouched."""
        return batch

    def post_update(self, batch: Batch) -> Batch:
        """Return *batch* untouched."""
        return batch


class _OptionalAnnotationPropagator(BaseDynamics):
    """Propagator spelling its optional torch settings ``Optional[...]``.

    ``_TypeCheckedModel`` is imported under ``TYPE_CHECKING`` alone here too,
    so the signature stays the strings the source wrote and the union spelling
    is what a recipe has to read back through.
    """

    def __init__(
        self,
        model: _TypeCheckedModel,
        dt: float = 0.25,
        precision: torch.dtype = torch.float32,
        autocast: Optional[torch.dtype] = None,
        staging: Optional[torch.device] = torch.device("cpu"),
    ) -> None:
        """Record the settings a recipe carries."""
        super().__init__(model=model)
        self.dt = dt
        self.precision = precision
        self.autocast = autocast
        self.staging = staging

    def pre_update(self, batch: Batch) -> Batch:
        """Return *batch* untouched."""
        return batch

    def post_update(self, batch: Batch) -> Batch:
        """Return *batch* untouched."""
        return batch


class _PrivateKnobPropagator(BaseDynamics):
    """Propagator storing a defaulted constructor argument under a private name."""

    def __init__(
        self,
        model: BaseModelMixin,
        dt: float = 0.25,
        temperature: float = 300.0,
    ) -> None:
        """Keep the timestep exposed and hide the temperature."""
        super().__init__(model=model)
        self.dt = dt
        self._temperature = temperature

    def pre_update(self, batch: Batch) -> Batch:
        """Return *batch* untouched."""
        return batch

    def post_update(self, batch: Batch) -> Batch:
        """Return *batch* untouched."""
        return batch


class _SpecListSource(_ListSource):
    """List source that names itself in a recipe by the count it was built from."""

    def __init__(self, count: int) -> None:
        super().__init__(
            [_make_system(_SEED_ELEMENT, 500 + index) for index in range(count)]
        )

    def to_spec_dict(self) -> dict[str, Any]:
        """Return the count this source rebuilds from."""
        return {"count": len(self.structures)}

    @classmethod
    def from_spec_dict(cls, spec: Mapping[str, Any]) -> _SpecListSource:
        """Rebuild the source :meth:`to_spec_dict` described."""
        return cls(int(spec["count"]))


def _admit_all(frames: Batch) -> torch.Tensor:
    """Admission predicate keeping every frame."""
    return torch.ones(frames.num_graphs, dtype=torch.bool)


class _DropNewest:
    """Eviction policy retiring the frames that arrived last."""

    def select(self, buffer: Batch, incoming: Batch, capacity: int) -> torch.Tensor:  # noqa: ARG002
        """Name the newest frames past capacity."""
        return torch.arange(capacity, buffer.num_graphs)


class TestOnPolicyRecipeRoundTrip:
    def test_a_source_without_spec_methods_is_refused_at_serialization(
        self, tmp_path: Path
    ) -> None:
        """A streaming source has no stable cursor to serialize, so the recipe refuses it."""
        teacher = _build_direct_force_teacher(seed=2)
        config = _make_config(tmp_path, _build_demo_model(), teacher)
        loop = config.model_copy(
            update={
                "initial_structures": _ListSource([_make_system(_SEED_ELEMENT, 500)])
            }
        )

        with pytest.raises(ValueError, match="no stable cursor position to serialize"):
            loop.to_spec_dict(teacher=teacher)

    def test_a_source_with_spec_methods_round_trips_under_its_class_path(
        self, tmp_path: Path
    ) -> None:
        """A source naming itself travels as its own block and is rebuilt by its class."""
        teacher = _build_direct_force_teacher(seed=2)
        student = _build_demo_model()
        config = _make_config(tmp_path, student, teacher)
        loop = config.model_copy(update={"initial_structures": _SpecListSource(3)})

        spec = loop.to_spec_dict(teacher=teacher)
        rebuilt = OnPolicyConfig.from_spec_dict(
            json.loads(json.dumps(spec)), student=student, teacher=teacher
        )

        assert spec["initial_structures"] == {
            "source_cls": f"{__name__}._SpecListSource",
            "count": 3,
        }
        assert isinstance(rebuilt.initial_structures, _SpecListSource)
        assert len(rebuilt.initial_structures.structures) == 3

    def test_live_replay_collaborators_are_omitted_with_a_warning(
        self, tmp_path: Path
    ) -> None:
        """The sink, the admission predicate, and a policy instance never reach the recipe."""
        teacher = _build_direct_force_teacher(seed=2)
        config = _make_config(tmp_path, _build_demo_model(), teacher)
        loop = config.model_copy(
            update={
                "capture_sink": HostMemory(capacity=16),
                "replay_admission": _admit_all,
                "replay_eviction": _DropNewest(),
            }
        )

        with pytest.warns(
            UserWarning, match="capture_sink and replay_admission hold runtime objects"
        ):
            spec = loop.to_spec_dict(teacher=teacher)

        assert spec["replay_eviction"] == "fifo"
        assert "capture_sink" not in spec
        assert "replay_admission" not in spec

    def test_a_recipe_built_config_serializes_back_to_its_recipe(
        self, tmp_path: Path
    ) -> None:
        """Rebuilding from a recipe and re-serializing returns the same recipe."""
        teacher = _build_direct_force_teacher(seed=2)
        seed_store = tmp_path / "seeds.zarr"
        _make_store(
            seed_store, _make_scorer(teacher), _SEED_ELEMENT, 4, 500, predictions=True
        )
        recipe = _make_recipe(seed_store)

        config = OnPolicyConfig.from_spec_dict(
            recipe, student=_build_demo_model(), teacher=teacher
        )

        assert config.to_spec_dict(teacher=teacher) == recipe
        assert isinstance(config.dynamics, NVTLangevin)

    def test_the_scorer_probe_seed_round_trips(self, tmp_path: Path) -> None:
        """A pinned Hessian probe survives the recipe, so a resumed run trains the same objective."""
        teacher = _build_direct_force_teacher(seed=2)
        seed_store = tmp_path / "seeds.zarr"
        _make_store(
            seed_store, _make_scorer(teacher), _SEED_ELEMENT, 4, 500, predictions=True
        )
        recipe = _make_recipe(seed_store)
        recipe["teacher_scorer"]["probe_seed"] = 42

        config = OnPolicyConfig.from_spec_dict(
            recipe, student=_build_demo_model(), teacher=teacher
        )

        assert config.teacher_scorer.probe_seed == 42
        assert (
            config.to_spec_dict(teacher=teacher)["teacher_scorer"]["probe_seed"] == 42
        )

    def test_every_knob_reaches_the_recipe(self, tmp_path: Path) -> None:
        """A setting added to ``OnPolicySettings`` cannot silently drop out of a recipe."""
        teacher = _build_direct_force_teacher(seed=2)
        seed_store = tmp_path / "seeds.zarr"
        _make_store(
            seed_store, _make_scorer(teacher), _SEED_ELEMENT, 4, 500, predictions=True
        )
        config = OnPolicyConfig.from_spec_dict(
            _make_recipe(seed_store), student=_build_demo_model(), teacher=teacher
        )

        spec = config.to_spec_dict(teacher=teacher)

        assert set(spec) - _RECIPE_OBJECT_KEYS == set(OnPolicySettings.model_fields)

    def test_the_rebuilt_propagator_holds_the_supplied_student(
        self, tmp_path: Path
    ) -> None:
        """On-policy data is only on-policy because the propagator holds the student."""
        teacher = _build_direct_force_teacher(seed=2)
        student = _build_demo_model()
        seed_store = tmp_path / "seeds.zarr"
        _make_store(
            seed_store, _make_scorer(teacher), _SEED_ELEMENT, 4, 500, predictions=True
        )

        config = OnPolicyConfig.from_spec_dict(
            _make_recipe(seed_store), student=student, teacher=teacher
        )

        assert config.dynamics.model is student
        assert config.teacher_scorer.teacher is teacher

    def test_a_strategy_spec_carries_the_recipe_and_the_anchor(
        self, tmp_path: Path
    ) -> None:
        """A whole on-policy run survives to_spec_dict as references."""
        teacher = _build_direct_force_teacher(seed=2)
        strategy = _make_strategy(
            tmp_path, student=_build_demo_model(), teacher=teacher, num_steps=2
        )

        spec = json.loads(json.dumps(strategy.to_spec_dict()))

        assert spec["on_policy"]["dynamics"] == _LANGEVIN
        assert spec["reference_dataset"]["path"].endswith("reference.zarr")
        assert spec["teacher_signals"] is None

    def test_the_round_trip_rebuilds_a_running_on_policy_strategy(
        self, tmp_path: Path
    ) -> None:
        """The rebuilt strategy generates rather than falling back to offline."""
        teacher = _build_direct_force_teacher(seed=2)
        strategy = _make_strategy(
            tmp_path, student=_build_demo_model(), teacher=teacher, num_steps=2
        )
        spec = json.loads(json.dumps(strategy.to_spec_dict()))

        rebuilt = DistillationStrategy.from_spec_dict(
            spec,
            models={"student": _build_demo_model(), "teacher": teacher},
        )

        assert rebuilt.on_policy is not None
        assert rebuilt.on_policy.dynamics.model is rebuilt.models["student"]
        assert rebuilt.reference_dataset is not None
        rebuilt.run()
        assert rebuilt.step_count == 2
        assert len(rebuilt.replay_buffer) > 0

    def test_an_in_memory_seed_dataset_names_what_to_do(self, tmp_path: Path) -> None:
        """A dataset a path cannot name is refused with the fix in the message."""
        teacher = _build_direct_force_teacher(seed=2)
        student = _build_demo_model()
        seed_store = tmp_path / "seeds.zarr"
        _make_store(
            seed_store, _make_scorer(teacher), _SEED_ELEMENT, 4, 500, predictions=True
        )
        config = OnPolicyConfig.from_spec_dict(
            _make_recipe(seed_store), student=student, teacher=teacher
        )
        config.initial_structures = InitialStructures(
            InMemoryDataset(in_memory_batch=_make_batch(_SEED_ELEMENT, 2, 500))
        )

        with pytest.raises(ValueError, match="OnPolicyConfig.initial_structures is a"):
            config.to_spec_dict(teacher=teacher)

    def test_a_hand_built_propagator_is_omitted_with_its_reason(
        self, tmp_path: Path
    ) -> None:
        """A propagator that hides its constructor arguments cannot be described."""
        teacher = _build_direct_force_teacher(seed=2)
        student = _build_demo_model()
        seed_store = tmp_path / "seeds.zarr"
        _make_store(
            seed_store, _make_scorer(teacher), _SEED_ELEMENT, 4, 500, predictions=True
        )
        config = OnPolicyConfig.from_spec_dict(
            _make_recipe(seed_store), student=student, teacher=teacher
        )
        config.dynamics = NVTLangevin(
            student, dt=0.5, temperature=300.0, friction=0.01, random_seed=7
        )

        with pytest.raises(ValueError, match="does not expose its"):
            config.to_spec_dict(teacher=teacher)

    def test_a_locally_defined_propagator_is_omitted_with_its_reason(
        self, tmp_path: Path
    ) -> None:
        """A propagator class no import names loses neither a step nor a checkpoint."""

        class _LocalPropagator(DemoDynamics):
            """Propagator class defined where no dotted path reaches it."""

        teacher = _build_direct_force_teacher(seed=2)
        student = _build_demo_model()
        checkpoints = tmp_path / "checkpoints"
        strategy = _make_strategy(
            tmp_path,
            student=student,
            teacher=teacher,
            num_steps=2,
            hooks=[CheckpointHook(checkpoints, step_interval=1)],
        )
        strategy.on_policy.dynamics = _LocalPropagator(
            student, n_steps=_SEGMENT_STEPS, dt=0.5
        )

        with pytest.warns(UserWarning, match="recipe is omitted"):
            strategy.run()

        assert strategy.step_count == 2
        assert checkpoints.is_dir()
        assert "on_policy" not in strategy.to_spec_dict()

    def test_a_scorer_over_another_teacher_is_refused(self, tmp_path: Path) -> None:
        """A recipe names the teacher by role, so a second one cannot be described."""
        teacher = _build_direct_force_teacher(seed=2)
        student = _build_demo_model()
        seed_store = tmp_path / "seeds.zarr"
        _make_store(
            seed_store, _make_scorer(teacher), _SEED_ELEMENT, 4, 500, predictions=True
        )
        config = OnPolicyConfig.from_spec_dict(
            _make_recipe(seed_store), student=student, teacher=teacher
        )

        with pytest.raises(ValueError, match="not the strategy's"):
            config.to_spec_dict(teacher=_build_direct_force_teacher(seed=5))


class TestIntrospectedPropagatorRecipes:
    def test_a_dtype_argument_round_trips_through_json(self) -> None:
        """A torch.dtype kwarg travels as its name and comes back as a dtype."""
        student = _build_demo_model()
        propagator = _ScalarPropagator(student, dt=0.5, precision=torch.float64)

        spec = _dynamics_spec_dict(propagator)
        rebuilt = _dynamics_from_spec_dict(json.loads(json.dumps(spec)), student)

        assert spec["kwargs"]["precision"] == "float64"
        assert rebuilt.precision is torch.float64

    def test_a_device_argument_round_trips_through_json(self) -> None:
        """A torch.device kwarg travels as its name and comes back as a device."""
        student = _build_demo_model()
        propagator = _ScalarPropagator(student, staging=torch.device("cpu"))

        spec = _dynamics_spec_dict(propagator)
        rebuilt = _dynamics_from_spec_dict(json.loads(json.dumps(spec)), student)

        assert spec["kwargs"]["staging"] == "cpu"
        assert rebuilt.staging == torch.device("cpu")

    def test_the_student_is_not_reported_as_a_dropped_collaborator(self) -> None:
        """The one collaborator a rebuild rebinds is not warned about."""
        propagator = _ScalarPropagator(_build_demo_model(), dt=0.5)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _dynamics_spec_dict(propagator)

        assert not [w for w in caught if "runtime objects" in str(w.message)]

    def test_a_privately_stored_defaulted_argument_is_refused(self) -> None:
        """A setting no attribute exposes is named, not rebuilt at the library default."""
        propagator = _PrivateKnobPropagator(
            _build_demo_model(), dt=0.5, temperature=1500.0
        )

        with pytest.raises(ValueError, match="does not expose its"):
            _dynamics_spec_dict(propagator)

    def test_a_type_checking_only_annotation_does_not_block_introspection(self) -> None:
        """A propagator written in the shipped style is introspected, not refused."""
        student = _build_demo_model()
        propagator = _TypeCheckedPropagator(
            student, dt=0.5, precision=torch.float64, staging=torch.device("cpu")
        )

        spec = _dynamics_spec_dict(propagator)
        rebuilt = _dynamics_from_spec_dict(json.loads(json.dumps(spec)), student)

        assert spec["kwargs"]["precision"] == "float64"
        assert rebuilt.precision is torch.float64
        assert rebuilt.staging == torch.device("cpu")

    def test_a_shipped_propagator_round_trips_when_it_exposes_its_knobs(self) -> None:
        """DemoDynamics keeps its constructor arguments, so a recipe describes it."""
        student = _build_demo_model()
        propagator = DemoDynamics(model=student, n_steps=_SEGMENT_STEPS, dt=0.5)

        spec = _dynamics_spec_dict(propagator)
        rebuilt = _dynamics_from_spec_dict(json.loads(json.dumps(spec)), student)

        assert spec["kwargs"] == {"n_steps": _SEGMENT_STEPS, "dt": 0.5}
        assert isinstance(rebuilt, DemoDynamics)
        assert rebuilt.n_steps == _SEGMENT_STEPS

    @pytest.mark.parametrize(
        "propagator_cls",
        [_ScalarPropagator, _TypeCheckedPropagator, _OptionalAnnotationPropagator],
        ids=[
            "resolved-annotation",
            "type-checking-only-annotation",
            "optional-spelling",
        ],
    )
    def test_an_optional_dtype_argument_round_trips_through_json(
        self, propagator_cls: type[BaseDynamics]
    ) -> None:
        """A ``torch.dtype | None`` setting decodes back whichever form its union takes."""
        student = _build_demo_model()
        propagator = propagator_cls(student, autocast=torch.bfloat16)

        spec = _dynamics_spec_dict(propagator)
        rebuilt = _dynamics_from_spec_dict(json.loads(json.dumps(spec)), student)

        assert spec["kwargs"]["autocast"] == "bfloat16"
        assert rebuilt.autocast is torch.bfloat16


class TestRecordedPropagatorRecipes:
    def test_a_recipe_kwarg_json_cannot_carry_is_refused_at_build(self) -> None:
        """A tensor setting is named where it entered, not at the checkpoint."""
        spec = _make_dynamics_spec(_ScalarPropagator, dt=torch.tensor(0.5))

        with pytest.raises(ValueError, match="JSON cannot carry"):
            _dynamics_from_spec_dict(spec, _build_demo_model())

    def test_a_dtype_supplied_as_an_object_is_recorded_by_name(self) -> None:
        """A dtype passed as an object reaches the constructor and travels as a name."""
        student = _build_demo_model()
        spec = _make_dynamics_spec(_ScalarPropagator, dt=0.5, precision=torch.float64)

        propagator = _dynamics_from_spec_dict(spec, student)
        emitted = _dynamics_spec_dict(propagator)

        assert propagator.precision is torch.float64
        assert emitted["kwargs"]["precision"] == "float64"
        assert json.loads(json.dumps(emitted)) == emitted

    def test_mutating_an_emitted_spec_leaves_the_recorded_reference_alone(self) -> None:
        """The emitted spec is a copy of the reference rather than the reference."""
        spec = _make_dynamics_spec(_ScalarPropagator, dt=0.5)
        propagator = _dynamics_from_spec_dict(spec, _build_demo_model())

        emitted = _dynamics_spec_dict(propagator)
        emitted["kwargs"]["dt"] = 999.0

        assert _dynamics_spec_dict(propagator)["kwargs"]["dt"] == 0.5
        assert spec["kwargs"]["dt"] == 0.5


class TestPropagatorCollaboratorWarnings:
    def test_a_hook_registered_after_a_recipe_build_is_reported(
        self, tmp_path: Path
    ) -> None:
        """A hook added after a recipe build is reported, not silently dropped."""
        teacher = _build_direct_force_teacher(seed=2)
        config = _make_config(tmp_path, _build_demo_model(), teacher)
        config.dynamics.register_hook(
            NeighborListHook(
                NeighborConfig(cutoff=5.0), stage=DynamicsStage.BEFORE_COMPUTE
            )
        )

        with pytest.warns(UserWarning, match="runtime objects no recipe"):
            config.to_spec_dict(teacher=teacher)

    def test_a_hook_on_an_introspected_propagator_is_reported(self) -> None:
        """The same report reaches a propagator no recipe built."""
        propagator = _ScalarPropagator(_build_demo_model(), dt=0.5)
        propagator.register_hook(
            NeighborListHook(
                NeighborConfig(cutoff=5.0), stage=DynamicsStage.BEFORE_COMPUTE
            )
        )

        with pytest.warns(UserWarning, match=r"\['hooks'\] hold runtime objects"):
            _dynamics_spec_dict(propagator)

    def test_the_loops_own_label_hook_is_not_reported(self, tmp_path: Path) -> None:
        """A mid-segment checkpoint stays quiet about the hook a rebuild re-adds."""
        teacher = _build_direct_force_teacher(seed=2)
        config = _make_config(tmp_path, _build_demo_model(), teacher)
        config.dynamics.register_hook(TeacherLabelHook(config.teacher_scorer))

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            config.to_spec_dict(teacher=teacher)

        assert not [w for w in caught if "runtime objects" in str(w.message)]


class TestOnPolicyRestart:
    def test_a_restored_run_continues_the_same_trajectory(self, tmp_path: Path) -> None:
        """Resuming from a checkpoint reaches the trajectory an unbroken run does."""
        torch.manual_seed(0)
        teacher = _build_direct_force_teacher(seed=2)
        uninterrupted = _make_strategy(
            tmp_path / "whole",
            student=_build_demo_model(),
            teacher=teacher,
            num_steps=4,
        )
        uninterrupted.run()

        torch.manual_seed(0)
        interrupted = _make_strategy(
            tmp_path / "split",
            student=_build_demo_model(),
            teacher=teacher,
            num_steps=2,
            hooks=[CheckpointHook(tmp_path / "split" / "ckpt", epoch_interval=1)],
        )
        interrupted.run()
        resumed = _make_strategy(
            tmp_path / "split",
            student=_build_demo_model(),
            teacher=teacher,
            num_steps=4,
        )
        resumed.restore_checkpoint(tmp_path / "split" / "ckpt")
        resumed.run()

        assert resumed.step_count == uninterrupted.step_count == 4
        assert (
            resumed.on_policy.dynamics.step_count
            == uninterrupted.on_policy.dynamics.step_count
        )
        torch.testing.assert_close(
            resumed._on_policy_state.positions,
            uninterrupted._on_policy_state.positions,
        )

    def test_a_checkpoint_inside_a_segment_resumes_the_trajectory_it_held(
        self, tmp_path: Path
    ) -> None:
        """A mid-training-phase checkpoint carries the propagator state anyway."""
        torch.manual_seed(0)
        teacher = _build_direct_force_teacher(seed=2)
        interrupted = _make_strategy(
            tmp_path,
            student=_build_demo_model(),
            teacher=teacher,
            num_steps=4,
            hooks=[CheckpointHook(tmp_path / "ckpt", step_interval=1)],
        )
        interrupted.run()

        resumed = _make_strategy(
            tmp_path, student=_build_demo_model(), teacher=teacher, num_steps=4
        )
        resumed.restore_checkpoint(tmp_path / "ckpt", checkpoint_index=0)
        bundle = next(
            hook for hook in resumed.hooks if isinstance(hook, _OnPolicyRestartHook)
        )._restored
        resumed.run()

        assert int(bundle["dynamics_step_count"]) == _SEGMENT_STEPS
        assert resumed.step_count == 4
        assert resumed.on_policy.dynamics.step_count == 3 * _SEGMENT_STEPS
        assert interrupted.on_policy.dynamics.step_count == 2 * _SEGMENT_STEPS

    def test_a_restored_run_keeps_the_frames_it_generated(self, tmp_path: Path) -> None:
        """The replay buffer travels with the checkpoint instead of restarting empty."""
        torch.manual_seed(0)
        teacher = _build_direct_force_teacher(seed=2)
        interrupted = _make_strategy(
            tmp_path,
            student=_build_demo_model(),
            teacher=teacher,
            num_steps=2,
            hooks=[CheckpointHook(tmp_path / "ckpt", epoch_interval=1)],
        )
        interrupted.run()
        stored = len(interrupted.replay_buffer)

        resumed = _make_strategy(
            tmp_path, student=_build_demo_model(), teacher=teacher, num_steps=4
        )
        resumed.restore_checkpoint(tmp_path / "ckpt")
        resumed.run()

        assert stored > 0
        assert len(resumed.replay_buffer) > stored

    @pytest.mark.parametrize("label_frequency", [1, _SEGMENT_STEPS])
    def test_a_restored_run_labels_the_frames_an_unbroken_run_labels(
        self, tmp_path: Path, label_frequency: int
    ) -> None:
        """The forced boundary label survives the restart, so no frame is relabeled."""
        torch.manual_seed(0)
        teacher = _build_direct_force_teacher(seed=2)
        uninterrupted = _make_strategy(
            tmp_path / "whole",
            student=_build_demo_model(),
            teacher=teacher,
            num_steps=4,
            label_frequency=label_frequency,
        )
        uninterrupted.run()

        torch.manual_seed(0)
        interrupted = _make_strategy(
            tmp_path / "split",
            student=_build_demo_model(),
            teacher=teacher,
            num_steps=2,
            hooks=[CheckpointHook(tmp_path / "split" / "ckpt", epoch_interval=1)],
            label_frequency=label_frequency,
        )
        interrupted.run()
        resumed = _make_strategy(
            tmp_path / "split",
            student=_build_demo_model(),
            teacher=teacher,
            num_steps=4,
            label_frequency=label_frequency,
        )
        resumed.restore_checkpoint(tmp_path / "split" / "ckpt")
        resumed.run()

        assert len(resumed.replay_buffer) == len(uninterrupted.replay_buffer)
        torch.testing.assert_close(
            resumed._on_policy_state.positions,
            uninterrupted._on_policy_state.positions,
            rtol=0.0,
            atol=0.0,
        )
        reference = uninterrupted.models["student"].state_dict()
        for name, tensor in resumed.models["student"].state_dict().items():
            torch.testing.assert_close(tensor, reference[name], rtol=0.0, atol=0.0)

    def test_a_run_that_never_generated_restarts_by_seeding(
        self, tmp_path: Path
    ) -> None:
        """A checkpoint taken outside a segment loop carries no trajectory."""
        teacher = _build_direct_force_teacher(seed=2)
        strategy = _make_strategy(
            tmp_path, student=_build_demo_model(), teacher=teacher, num_steps=2
        )

        strategy.save_checkpoint(tmp_path / "ckpt")
        resumed = _make_strategy(
            tmp_path, student=_build_demo_model(), teacher=teacher, num_steps=2
        )
        resumed.restore_checkpoint(tmp_path / "ckpt")
        resumed.run()

        assert resumed.step_count == 2
        assert resumed.on_policy.dynamics.step_count == _SEGMENT_STEPS


def _make_exhausting_strategy(
    tmp_path: Path,
    *,
    teacher: BaseModelMixin,
    num_steps: int,
    hooks: list[Any] | None = None,
) -> DistillationStrategy:
    """Return a FIRE relaxation loop whose four structures all graduate on the first step."""
    return _make_strategy(
        tmp_path,
        student=_build_demo_model(),
        teacher=teacher,
        num_steps=num_steps,
        hooks=hooks,
        dynamics=_make_dynamics_spec(FIRE, dt=0.1),
        fmax=1e3,
        generation_steps=2,
    )


class TestExhaustedGenerationRestart:
    def test_an_exhausted_run_checkpoints_its_frames_and_the_exhaustion(
        self, tmp_path: Path
    ) -> None:
        """Once generation runs dry the bundle carries the buffer, not an empty state."""
        torch.manual_seed(0)
        teacher = _build_direct_force_teacher(seed=2)
        strategy = _make_exhausting_strategy(tmp_path, teacher=teacher, num_steps=4)
        with pytest.warns(UserWarning, match="nothing left to start a fresh one"):
            strategy.run()
        hook = _restart_hook(strategy)
        hook.prepare_strategy(strategy)

        bundle = hook.state_dict()

        assert strategy.on_policy.initial_structures.exhausted
        assert bundle["generation_exhausted"] is True
        assert "trajectory" not in bundle
        assert len(_batch_from_state(bundle["replay_frames"])) == len(
            strategy.replay_buffer.dataset.in_memory_batch
        )
        assert bundle["initial_structures"] == (
            strategy.on_policy.initial_structures.state_dict()
        )

    def test_a_resumed_exhausted_run_trains_on_its_buffer_without_regenerating(
        self, tmp_path: Path
    ) -> None:
        """The restart continues the tail, so no relaxed structure is served again."""
        torch.manual_seed(0)
        teacher = _build_direct_force_teacher(seed=2)
        interrupted = _make_exhausting_strategy(
            tmp_path,
            teacher=teacher,
            num_steps=4,
            hooks=[CheckpointHook(tmp_path / "ckpt", epoch_interval=1)],
        )
        with pytest.warns(UserWarning, match="nothing left to start a fresh one"):
            interrupted.run()
        frames = len(interrupted.replay_buffer)
        propagated = interrupted.on_policy.dynamics.step_count
        resumed = _make_exhausting_strategy(tmp_path, teacher=teacher, num_steps=8)
        resumed.restore_checkpoint(tmp_path / "ckpt")

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            resumed.run()

        assert resumed.step_count == 8
        assert len(resumed.replay_buffer) == frames
        assert resumed.on_policy.dynamics.step_count == propagated
        assert resumed.on_policy.initial_structures.exhausted
        assert not [w for w in caught if "nothing left" in str(w.message)]


class TestRebuildDevicePlacement:
    def test_recipe_datasets_follow_the_spec_devices(self, tmp_path: Path) -> None:
        """A spec restored onto another device opens its stores there, not where recorded."""
        teacher = _build_direct_force_teacher(seed=2)
        strategy = _make_strategy(
            tmp_path, student=_build_demo_model(), teacher=teacher, num_steps=2
        )
        spec = strategy.to_spec_dict()
        spec["reference_dataset"]["device"] = "cuda:0"
        spec["on_policy"]["initial_structures"]["dataset"]["device"] = "cuda:0"
        spec["on_policy"]["replay_device"] = "cuda:0"

        rebuilt = DistillationStrategy.from_spec_dict(spec, models=strategy.models)

        assert str(rebuilt.reference_dataset.target_device) == "cpu"
        assert str(rebuilt.on_policy.initial_structures.dataset.target_device) == "cpu"
        assert rebuilt.on_policy.replay_device == "cpu"

    def test_a_checkpoint_recorded_elsewhere_restores_where_map_location_says(
        self, tmp_path: Path
    ) -> None:
        """``map_location`` moves the run's data with the strategy, not only its weights."""
        torch.manual_seed(0)
        teacher = _build_direct_force_teacher(seed=2)
        strategy = _make_strategy(
            tmp_path,
            student=_build_demo_model(),
            teacher=teacher,
            num_steps=2,
            hooks=[CheckpointHook(tmp_path / "ckpt", epoch_interval=1)],
        )
        strategy.run()
        for path in (tmp_path / "ckpt").rglob("*.json"):
            metadata = json.loads(path.read_text())
            if "reference_dataset" not in metadata:
                continue
            metadata["devices"] = ["cuda:0"]
            metadata["reference_dataset"]["device"] = "cuda:0"
            metadata["on_policy"]["initial_structures"]["dataset"]["device"] = "cuda:0"
            path.write_text(json.dumps(metadata))

        resumed = DistillationStrategy.load_checkpoint(
            tmp_path / "ckpt", map_location="cpu"
        )

        assert resumed.devices == [torch.device("cpu")]
        assert str(resumed.reference_dataset.target_device) == "cpu"
        assert str(resumed.on_policy.initial_structures.dataset.target_device) == "cpu"


class TestRestartBundleIntegrity:
    def test_a_defragged_trajectory_packs_only_the_graphs_it_kept(self) -> None:
        """Storage wider than the kept graphs is truncated, not written whole."""
        batch = _make_defragged_batch()

        state = _batch_state(batch)

        assert state["atom:positions"].shape[0] == sum(batch.num_nodes_list)
        rebuilt = _batch_from_state(state)
        assert rebuilt.num_nodes_list == batch.num_nodes_list
        torch.testing.assert_close(
            rebuilt.positions, batch.positions[: batch.num_nodes]
        )

    def test_counts_that_outrun_the_rows_are_refused_at_write_time(self) -> None:
        """An inconsistent batch fails at the checkpoint, not at the resume."""
        batch = _make_batch(_SEED_ELEMENT, 2, 900)
        batch._atoms_group.segment_lengths = torch.tensor([4, 12])

        with pytest.raises(RuntimeError, match="internally inconsistent"):
            _batch_state(batch)

    def test_a_negative_segment_length_is_refused_at_write_time(self) -> None:
        """A corrupt segment length names the bundle rather than torch.split."""
        batch = _make_batch(_SEED_ELEMENT, 2, 900)
        batch._atoms_group.segment_lengths = torch.tensor([-4, 12])

        with pytest.raises(RuntimeError, match="negative segment length"):
            _batch_state(batch)

    def test_a_negative_segment_length_is_refused_on_the_way_back_in(self) -> None:
        """A bundle an older build wrote is refused rather than half-rebuilt."""
        state = _batch_state(_make_batch(_SEED_ELEMENT, 2, 900))
        state["num_nodes_list"] = torch.tensor([-4, 12])

        with pytest.raises(RuntimeError, match="negative segment length"):
            _batch_from_state(state)

    def test_rows_that_do_not_match_the_counts_are_refused_on_the_way_back_in(
        self,
    ) -> None:
        """The bundle an unfixed build wrote is named for what it is."""
        state = _batch_state(_make_defragged_batch())
        state["atom:positions"] = torch.randn(16, 3)

        with pytest.raises(RuntimeError, match="cannot be resumed"):
            _batch_from_state(state)

    def test_packing_a_batch_global_index_field_is_refused(self) -> None:
        """A neighbor list cannot be represented, so it is never silently written."""
        with pytest.raises(RuntimeError, match="offsets a second time"):
            _batch_state(_make_neighbor_batch())

    def test_the_replay_bundle_leaves_out_the_neighbor_tensors(
        self, tmp_path: Path
    ) -> None:
        """Both halves of the bundle drop them, not the trajectory alone."""
        teacher = _build_direct_force_teacher(seed=2)
        strategy = _make_strategy(
            tmp_path, student=_build_demo_model(), teacher=teacher, num_steps=2
        )
        strategy._on_policy_state = _make_batch(_SEED_ELEMENT, 2, 500)
        buffer = ReplayBuffer()
        buffer.extend(_make_neighbor_batch())
        strategy._replay_buffer = buffer
        hook = _restart_hook(strategy)
        hook.prepare_strategy(strategy)

        frames = hook.state_dict()["replay_frames"]

        assert not [key for key in frames if "neighbor" in key]
        rebuilt = _batch_from_state(frames)
        torch.testing.assert_close(
            rebuilt.positions, buffer.dataset.in_memory_batch.positions
        )


class TestMultiStoreReference:
    def test_a_multi_store_reference_dataset_round_trips_as_its_paths(
        self, tmp_path: Path
    ) -> None:
        """A MultiDataset serializes as the stores it concatenates and rebuilds as one."""
        teacher = _build_direct_force_teacher(seed=2)
        scorer = _make_scorer(teacher)
        stores = [tmp_path / "reference_a.zarr", tmp_path / "reference_b.zarr"]
        composed = MultiDataset(
            *[
                _make_store(store, scorer, _REFERENCE_ELEMENT, 4, 700 + index)
                for index, store in enumerate(stores)
            ]
        )
        strategy = _make_strategy(
            tmp_path,
            student=_build_demo_model(),
            teacher=teacher,
            num_steps=2,
            reference_dataset=composed,
        )

        spec = strategy.to_spec_dict()
        rebuilt = DistillationStrategy.from_spec_dict(spec, models=strategy.models)

        assert spec["reference_dataset"] == {
            "paths": [str(store) for store in stores],
            "device": "cpu",
        }
        assert isinstance(rebuilt.reference_dataset, MultiDataset)
        assert len(rebuilt.reference_dataset) == len(composed)
        assert rebuilt.on_policy is not None

    def test_initial_structures_over_a_composition_round_trip(
        self, tmp_path: Path
    ) -> None:
        """A cursor over several stores names every one of them in the recipe."""
        scorer = _make_scorer(_build_direct_force_teacher(seed=2))
        stores = [tmp_path / "initial_a.zarr", tmp_path / "initial_b.zarr"]
        composed = MultiDataset(
            *[
                _make_store(
                    store, scorer, _SEED_ELEMENT, 2, 500 + index, predictions=True
                )
                for index, store in enumerate(stores)
            ]
        )

        spec = InitialStructures(composed, max_batch_size=3).to_spec_dict()
        rebuilt = InitialStructures.from_spec_dict(spec)

        assert spec["dataset"] == {
            "paths": [str(store) for store in stores],
            "device": "cpu",
        }
        assert isinstance(rebuilt.dataset, MultiDataset)
        assert (len(rebuilt), rebuilt.max_batch_size) == (4, 3)


class TestStructureCursorRestart:
    def test_the_bundle_carries_the_cursor_and_the_settings_it_ran_under(
        self, tmp_path: Path
    ) -> None:
        """A restart bundle records where the cursor stopped and under which settings."""
        torch.manual_seed(0)
        teacher = _build_direct_force_teacher(seed=2)
        strategy = _make_strategy(
            tmp_path, student=_build_demo_model(), teacher=teacher, num_steps=2
        )
        strategy.run()
        hook = _restart_hook(strategy)
        hook.prepare_strategy(strategy)

        bundle = hook.state_dict()

        assert (
            bundle["initial_structures"]
            == strategy.on_policy.initial_structures.state_dict()
        )
        assert bundle["settings"] == strategy.on_policy.settings.model_dump(mode="json")
        assert "initial_structures" not in bundle["settings"]

    def test_a_restored_source_serves_the_rows_an_unbroken_one_would(
        self, tmp_path: Path
    ) -> None:
        """The cursor resumes where the interrupted run left it, not at row zero."""
        torch.manual_seed(0)
        teacher = _build_direct_force_teacher(seed=2)
        seeds = _initial_structures_spec(tmp_path / "seeds.zarr", max_batch_size=2)
        interrupted = _make_strategy(
            tmp_path,
            student=_build_demo_model(),
            teacher=teacher,
            num_steps=2,
            initial_structures=seeds,
        )
        interrupted.run()
        interrupted.on_policy.initial_structures.draw(limit=1)
        hook = _restart_hook(interrupted)
        hook.prepare_strategy(interrupted)
        bundle = hook.state_dict()
        resumed = _make_strategy(
            tmp_path,
            student=_build_demo_model(),
            teacher=teacher,
            num_steps=4,
            initial_structures=seeds,
        )
        _restart_hook(resumed).load_state_dict(bundle)

        resumed._resume_or_seed(resumed.on_policy, ReplayBuffer())

        restored = resumed.on_policy.initial_structures.draw(limit=1)
        unbroken = interrupted.on_policy.initial_structures.draw(limit=1)
        assert len(restored) == 1
        assert int(restored[0].system_id.view(-1)[0]) == int(
            unbroken[0].system_id.view(-1)[0]
        )
        torch.testing.assert_close(restored[0].positions, unbroken[0].positions)

    def test_declared_budgets_are_configuration_rather_than_restart_state(
        self, tmp_path: Path
    ) -> None:
        """The bundle carries the cursor alone; a budget stays where the recipe set it."""
        torch.manual_seed(0)
        teacher = _build_direct_force_teacher(seed=2)
        structures = _initial_structures_spec(tmp_path / "seeds.zarr", max_atoms=32)
        interrupted = _make_strategy(
            tmp_path,
            student=_build_demo_model(),
            teacher=teacher,
            num_steps=2,
            initial_structures=structures,
        )
        interrupted.run()
        hook = _restart_hook(interrupted)
        hook.prepare_strategy(interrupted)
        bundle = hook.state_dict()
        resumed = _make_strategy(
            tmp_path,
            student=_build_demo_model(),
            teacher=teacher,
            num_steps=4,
            initial_structures=structures,
        )
        _restart_hook(resumed).load_state_dict(bundle)
        source = resumed.on_policy.initial_structures

        resumed._resume_or_seed(resumed.on_policy, ReplayBuffer())

        assert set(bundle["initial_structures"]) == {
            "cursor",
            "wraps",
            "next_system_id",
            "rank",
            "world_size",
        }
        assert (source.max_atoms, source.max_batch_size) == (32, None)


class TestRestartSettingsDrift:
    def test_a_setting_the_resumed_loop_changed_is_reported(
        self, tmp_path: Path
    ) -> None:
        """The halves of a run generated under different settings are named, not merged."""
        torch.manual_seed(0)
        teacher = _build_direct_force_teacher(seed=2)
        interrupted = _make_strategy(
            tmp_path, student=_build_demo_model(), teacher=teacher, num_steps=2
        )
        interrupted.run()
        hook = _restart_hook(interrupted)
        hook.prepare_strategy(interrupted)
        bundle = hook.state_dict()
        resumed = _make_strategy(
            tmp_path,
            student=_build_demo_model(),
            teacher=teacher,
            num_steps=4,
            label_frequency=2,
            replay_capacity=16,
        )
        _restart_hook(resumed).load_state_dict(bundle)

        with pytest.warns(UserWarning, match="label_frequency', 'replay_capacity"):
            resumed._resume_or_seed(resumed.on_policy, ReplayBuffer())

    def test_an_unchanged_loop_is_not_reported(self, tmp_path: Path) -> None:
        """Every restart would warn if the comparison were of objects, not settings."""
        torch.manual_seed(0)
        teacher = _build_direct_force_teacher(seed=2)
        interrupted = _make_strategy(
            tmp_path, student=_build_demo_model(), teacher=teacher, num_steps=2
        )
        interrupted.run()
        hook = _restart_hook(interrupted)
        hook.prepare_strategy(interrupted)
        bundle = hook.state_dict()
        resumed = _make_strategy(
            tmp_path, student=_build_demo_model(), teacher=teacher, num_steps=4
        )
        _restart_hook(resumed).load_state_dict(bundle)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            resumed._resume_or_seed(resumed.on_policy, ReplayBuffer())

        assert not [w for w in caught if "differently from the run" in str(w.message)]


class TestRestartAcrossWorldSizes:
    def test_a_rank_above_world_size_one_seeds_instead_of_replaying_rank_zeros_run(
        self, tmp_path: Path
    ) -> None:
        """The bundle rides in a rank-zero-only checkpoint, so no rank replays it."""
        torch.manual_seed(0)
        teacher = _build_direct_force_teacher(seed=2)
        interrupted = _make_strategy(
            tmp_path,
            student=_build_demo_model(),
            teacher=teacher,
            num_steps=2,
            hooks=[CheckpointHook(tmp_path / "ckpt", epoch_interval=1)],
        )
        interrupted.run()
        resumed = _make_strategy(
            tmp_path,
            student=_build_demo_model(),
            teacher=teacher,
            num_steps=4,
            distributed_manager=_FakeWorld(world_size=2),
        )
        resumed.restore_checkpoint(tmp_path / "ckpt")
        config = resumed.on_policy
        buffer = ReplayBuffer()

        with pytest.warns(UserWarning, match="cold replay buffer"):
            state, labeled_step = resumed._resume_or_seed(config, buffer)

        assert len(buffer) == 0
        assert labeled_step is None
        assert config.dynamics.step_count == 0
        torch.testing.assert_close(
            state.positions,
            InitialStructures(config.initial_structures.dataset)
            .initial_batch()
            .positions,
        )

    def test_a_matched_two_rank_restart_is_dropped_like_any_wider_one(
        self, tmp_path: Path
    ) -> None:
        """A world size that agrees at both ends is still one rank's bundle."""
        torch.manual_seed(0)
        teacher = _build_direct_force_teacher(seed=2)
        interrupted = _make_strategy(
            tmp_path,
            student=_build_demo_model(),
            teacher=teacher,
            num_steps=2,
            hooks=[CheckpointHook(tmp_path / "ckpt", epoch_interval=1)],
        )
        interrupted.run()
        resumed = _make_strategy(
            tmp_path,
            student=_build_demo_model(),
            teacher=teacher,
            num_steps=4,
            distributed_manager=_FakeWorld(world_size=2),
        )
        resumed.restore_checkpoint(tmp_path / "ckpt")
        resumed.global_step_count = 2 * resumed.step_count
        buffer = ReplayBuffer()

        with pytest.warns(UserWarning, match="resuming on world_size=2"):
            _, labeled_step = resumed._resume_or_seed(resumed.on_policy, buffer)

        assert len(buffer) == 0
        assert labeled_step is None
        assert resumed.on_policy.dynamics.step_count == 0

    def test_a_bundle_saved_on_more_ranks_is_dropped_when_one_rank_resumes(
        self, tmp_path: Path
    ) -> None:
        """The world the bundle was saved on is the shard its cursor records."""
        torch.manual_seed(0)
        teacher = _build_direct_force_teacher(seed=2)
        interrupted = _make_strategy(
            tmp_path,
            student=_build_demo_model(),
            teacher=teacher,
            num_steps=2,
            hooks=[CheckpointHook(tmp_path / "ckpt", epoch_interval=1)],
        )
        interrupted.run()
        resumed = _make_strategy(
            tmp_path, student=_build_demo_model(), teacher=teacher, num_steps=4
        )
        resumed.restore_checkpoint(tmp_path / "ckpt")
        _restart_hook(resumed)._restored["initial_structures"] |= {
            "rank": 0,
            "world_size": 2,
        }

        with pytest.warns(UserWarning, match="written on world_size=2"):
            resumed.run()

        assert resumed.on_policy.dynamics.step_count == _SEGMENT_STEPS

    def test_a_cursor_from_a_wider_world_drops_the_bundle_whatever_the_counters_say(
        self, tmp_path: Path
    ) -> None:
        """The world a bundle was written on is read off its cursor, not the counters."""
        torch.manual_seed(0)
        teacher = _build_direct_force_teacher(seed=2)
        interrupted = _make_strategy(
            tmp_path,
            student=_build_demo_model(),
            teacher=teacher,
            num_steps=2,
            hooks=[CheckpointHook(tmp_path / "ckpt", epoch_interval=1)],
        )
        interrupted.run()
        resumed = _make_strategy(
            tmp_path, student=_build_demo_model(), teacher=teacher, num_steps=4
        )
        resumed.restore_checkpoint(tmp_path / "ckpt")
        resumed.step_count, resumed.global_step_count = 8, 12
        _restart_hook(resumed)._restored["initial_structures"] |= {
            "rank": 0,
            "world_size": 2,
        }
        buffer = ReplayBuffer()

        with pytest.warns(UserWarning, match="written on world_size=2"):
            _, labeled_step = resumed._resume_or_seed(resumed.on_policy, buffer)

        assert len(buffer) == 0
        assert labeled_step is None

    def test_a_cursor_from_a_foreign_shard_is_dropped_rather_than_raised(
        self, tmp_path: Path
    ) -> None:
        """The source still refuses it; the segment loop degrades to the drop."""
        torch.manual_seed(0)
        teacher = _build_direct_force_teacher(seed=2)
        interrupted = _make_strategy(
            tmp_path,
            student=_build_demo_model(),
            teacher=teacher,
            num_steps=2,
            hooks=[CheckpointHook(tmp_path / "ckpt", epoch_interval=1)],
        )
        interrupted.run()
        resumed = _make_strategy(
            tmp_path, student=_build_demo_model(), teacher=teacher, num_steps=4
        )
        resumed.restore_checkpoint(tmp_path / "ckpt")
        _restart_hook(resumed)._restored["initial_structures"] |= {"rank": 1}
        buffer = ReplayBuffer()

        with pytest.warns(UserWarning, match="written for rank 1 of 1"):
            _, labeled_step = resumed._resume_or_seed(resumed.on_policy, buffer)

        assert len(buffer) == 0
        assert labeled_step is None

    def test_a_single_rank_bundle_is_still_consumed(self, tmp_path: Path) -> None:
        """The guard is a world-size guard, not a new refusal of every restart."""
        torch.manual_seed(0)
        teacher = _build_direct_force_teacher(seed=2)
        interrupted = _make_strategy(
            tmp_path,
            student=_build_demo_model(),
            teacher=teacher,
            num_steps=2,
            hooks=[CheckpointHook(tmp_path / "ckpt", epoch_interval=1)],
        )
        interrupted.run()
        resumed = _make_strategy(
            tmp_path, student=_build_demo_model(), teacher=teacher, num_steps=4
        )
        resumed.restore_checkpoint(tmp_path / "ckpt")
        buffer = ReplayBuffer()

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            state, labeled_step = resumed._resume_or_seed(resumed.on_policy, buffer)

        assert not [w for w in caught if "restart bundle is dropped" in str(w.message)]
        assert len(buffer) == len(interrupted.replay_buffer)
        assert labeled_step == _SEGMENT_STEPS - 1
        assert resumed.on_policy.dynamics.step_count == _SEGMENT_STEPS
        assert not torch.allclose(
            state.positions,
            InitialStructures(resumed.on_policy.initial_structures.dataset)
            .initial_batch()
            .positions,
        )


class TestSegmentLoopRestartOrder:
    def test_a_mid_segment_restore_opens_as_many_epochs_as_it_closes(
        self, tmp_path: Path
    ) -> None:
        """The interrupted segment is closed before the resumed one is opened."""
        torch.manual_seed(0)
        teacher = _build_direct_force_teacher(seed=2)
        interrupted = _make_strategy(
            tmp_path,
            student=_build_demo_model(),
            teacher=teacher,
            num_steps=3,
            hooks=[CheckpointHook(tmp_path / "ckpt", step_interval=1)],
        )
        interrupted.run()
        opened = _EpochBoundaryHook(TrainingStage.BEFORE_EPOCH)
        closed = _EpochBoundaryHook(TrainingStage.AFTER_EPOCH)
        resumed = _make_strategy(
            tmp_path,
            student=_build_demo_model(),
            teacher=teacher,
            num_steps=5,
            hooks=[opened, closed],
        )
        resumed.restore_checkpoint(tmp_path / "ckpt", checkpoint_index=0)

        assert resumed.epoch_step_count == 1
        resumed.run()

        assert opened.calls == closed.calls
        assert opened.calls > 0


class TestInternalHookIdentity:
    def test_rebuilding_from_a_live_strategys_hooks_keeps_one_of_each_seam(
        self, tmp_path: Path
    ) -> None:
        """Round-tripping hooks does not accumulate the labeling and restart seams."""
        teacher = _build_direct_force_teacher(seed=2)
        strategy = _make_strategy(
            tmp_path, student=_build_demo_model(), teacher=teacher, num_steps=2
        )

        for _ in range(2):
            strategy = _make_strategy(
                tmp_path,
                student=_build_demo_model(),
                teacher=teacher,
                num_steps=2,
                hooks=list(strategy.hooks),
            )

        restart = [
            hook for hook in strategy.hooks if isinstance(hook, _OnPolicyRestartHook)
        ]
        assert len(restart) == 1
        assert len(strategy.hooks) == 2

    def test_a_round_tripped_strategy_contributes_one_restart_bundle(
        self, tmp_path: Path
    ) -> None:
        """Two restart hooks would each write the whole bundle into one checkpoint."""
        torch.manual_seed(0)
        teacher = _build_direct_force_teacher(seed=2)
        strategy = _make_strategy(
            tmp_path, student=_build_demo_model(), teacher=teacher, num_steps=2
        )
        strategy.run()
        rebuilt = _make_strategy(
            tmp_path,
            student=_build_demo_model(),
            teacher=teacher,
            num_steps=2,
            hooks=list(strategy.hooks),
        )
        rebuilt._on_policy_state = strategy._on_policy_state
        rebuilt._replay_buffer = strategy.replay_buffer
        rebuilt._prepare_setup_hooks()

        states = _snapshot_hook_states(rebuilt)

        bundles = [key for key in states if key.endswith("_OnPolicyRestartHook:0")]
        assert len(bundles) == 1
        assert not [key for key in states if key.endswith("_OnPolicyRestartHook:1")]
        assert set(states[bundles[0]]) == {
            "dynamics_step_count",
            "settings",
            "trajectory",
            "replay_frames",
            "generation_exhausted",
            "initial_structures",
        }


class TestStrategySpecIdentity:
    def test_the_spec_names_the_strategy_class_that_rebuilds_it(
        self, tmp_path: Path
    ) -> None:
        """The distillation override adds ``strategy_cls`` and agrees with checkpoints."""
        teacher = _build_direct_force_teacher(seed=2)
        strategy = _make_strategy(
            tmp_path, student=_build_demo_model(), teacher=teacher, num_steps=2
        )

        spec = strategy.to_spec_dict()

        expected = f"{DistillationStrategy.__module__}.DistillationStrategy"
        assert spec["strategy_cls"] == expected
        assert strategy.to_checkpoint_dict()["strategy_cls"] == expected

    def test_an_omitted_recipe_still_names_the_strategy_class(
        self, tmp_path: Path
    ) -> None:
        """The warning path drops ``on_policy``, not the key that says what rebuilds it."""
        teacher = _build_direct_force_teacher(seed=2)
        strategy = _make_strategy(
            tmp_path, student=_build_demo_model(), teacher=teacher, num_steps=2
        )
        strategy.on_policy.initial_structures = InitialStructures(
            InMemoryDataset(in_memory_batch=_make_batch(_SEED_ELEMENT, 2, 500))
        )

        with pytest.warns(UserWarning, match="on-policy recipe is omitted"):
            spec = strategy.to_spec_dict()

        assert "on_policy" not in spec
        assert (
            spec["strategy_cls"]
            == f"{DistillationStrategy.__module__}.DistillationStrategy"
        )


def _make_supplied_loop(
    student: BaseModelMixin, teacher: BaseModelMixin
) -> OnPolicyConfig:
    """Return a live loop no recipe describes, seeded from an in-memory dataset."""
    return OnPolicyConfig(
        dynamics=NVTLangevin(
            student, dt=0.25, temperature=17.0, friction=0.02, random_seed=99
        ),
        teacher_scorer=_make_scorer(teacher),
        initial_structures=InitialStructures(
            InMemoryDataset(
                in_memory_batch=_make_batch(_SEED_ELEMENT, 2, 500, predictions=True)
            )
        ),
        replay_ratio=0.5,
        training_steps_per_segment=2,
        batch_size=4,
        generation_steps=_SEGMENT_STEPS,
        label_frequency=1,
    )


class TestSuppliedLoopPrecedence:
    def test_a_supplied_loop_wins_over_the_recipe_the_checkpoint_carries(
        self, tmp_path: Path
    ) -> None:
        """An explicitly supplied loop is the run, whatever recipe the spec holds."""
        teacher = _build_direct_force_teacher(seed=2)
        strategy = _make_strategy(
            tmp_path, student=_build_demo_model(), teacher=teacher, num_steps=2
        )
        spec = json.loads(json.dumps(strategy.to_spec_dict()))
        assert spec["on_policy"]["dynamics"] == _LANGEVIN
        student = _build_demo_model()
        supplied = _make_supplied_loop(student, teacher)

        rebuilt = DistillationStrategy.from_spec_dict(
            spec,
            models={"student": student, "teacher": teacher},
            on_policy=supplied,
        )

        assert rebuilt.on_policy is supplied
        assert rebuilt.on_policy.dynamics.model is student
        assert isinstance(rebuilt.on_policy.initial_structures.dataset, InMemoryDataset)

    def test_a_loop_offered_for_the_restore_wins_over_the_recipe(
        self, tmp_path: Path
    ) -> None:
        """A loop offered over the restore contextvar still outranks the recipe.

        :meth:`DistillationStrategy.from_checkpoint_dict` does not forward its
        *on_policy* to ``from_spec_dict``; it offers it over
        ``_supplied_runtime_objects``, so the offer has to be read before the
        spec's own recipe is rebuilt or a describable recipe swallows the live
        loop the caller handed over.
        """
        teacher = _build_direct_force_teacher(seed=2)
        strategy = _make_strategy(
            tmp_path, student=_build_demo_model(), teacher=teacher, num_steps=2
        )
        checkpoint = json.loads(json.dumps(strategy.to_checkpoint_dict()))
        assert checkpoint["on_policy"]["dynamics"] == _LANGEVIN
        student = _build_demo_model()
        supplied = _make_supplied_loop(student, teacher)

        rebuilt = DistillationStrategy.from_checkpoint_dict(
            checkpoint,
            models={"student": student, "teacher": teacher},
            on_policy=supplied,
        )

        assert rebuilt.on_policy is supplied
        assert isinstance(rebuilt.on_policy.initial_structures.dataset, InMemoryDataset)


class TestPreflightBoundary:
    def test_the_knob_preflight_leaves_the_objects_to_the_loop(
        self, tmp_path: Path
    ) -> None:
        """Pre-flight reads the declarative settings; it never builds the propagator."""
        recipe = _make_recipe(
            tmp_path / "absent.zarr",
            dynamics={"cls_path": "no.such.module.Propagator", "kwargs": {}},
        )

        settings = _on_policy_settings(recipe)

        assert settings.generation_steps == _SEGMENT_STEPS
        assert settings.replay_ratio == 0.5

    def test_a_propagator_carrying_its_own_sampler_is_the_loops_call(
        self, tmp_path: Path
    ) -> None:
        """Whether the objects compose is settled where the loop is installed."""
        teacher = _build_direct_force_teacher(seed=2)
        student = _build_demo_model()
        seed_store = tmp_path / "seeds.zarr"
        _make_store(
            seed_store, _make_scorer(teacher), _SEED_ELEMENT, 4, 500, predictions=True
        )

        config = OnPolicyConfig(
            dynamics=NVTLangevin(
                student,
                dt=0.5,
                temperature=300.0,
                friction=0.01,
                random_seed=7,
                sampler=SizeAwareSampler(
                    _build_small_dataset(), max_atoms=64, max_batch_size=4
                ),
            ),
            teacher_scorer=_make_scorer(teacher),
            initial_structures=InitialStructures(
                Dataset(reader=AtomicDataZarrReader(seed_store), device="cpu")
            ),
            replay_ratio=0.5,
            training_steps_per_segment=2,
            batch_size=4,
            generation_steps=_SEGMENT_STEPS,
        )

        assert config.dynamics.inflight_mode


class TestOnPolicyCheckpointResume:
    def test_a_resumed_segment_loop_continues_instead_of_reseeding(
        self, tmp_path: Path
    ) -> None:
        """Trajectory, propagator counter, and replay frames all survive the restart."""
        torch.manual_seed(0)
        teacher = _build_direct_force_teacher(seed=2)
        interrupted = _make_strategy(
            tmp_path,
            student=_build_demo_model(),
            teacher=teacher,
            num_steps=2,
            hooks=[CheckpointHook(tmp_path / "ckpt", epoch_interval=1)],
        )
        interrupted.run()
        stopped_at = interrupted._on_policy_state.positions.clone()
        generated = len(interrupted.replay_buffer)

        resumed = _make_strategy(
            tmp_path, student=_build_demo_model(), teacher=teacher, num_steps=4
        )
        resumed.restore_checkpoint(tmp_path / "ckpt")
        bundle = _restart_hook(resumed)._restored
        resumed.run()

        torch.testing.assert_close(
            _batch_from_state(bundle["trajectory"]).positions, stopped_at
        )
        assert int(bundle["dynamics_step_count"]) == _SEGMENT_STEPS
        assert resumed.on_policy.dynamics.step_count == 2 * _SEGMENT_STEPS
        assert len(resumed.replay_buffer) > generated

        torch.manual_seed(0)
        reseeded = _make_strategy(
            tmp_path, student=_build_demo_model(), teacher=teacher, num_steps=2
        )
        reseeded.run()

        assert not torch.allclose(
            resumed._on_policy_state.positions, reseeded._on_policy_state.positions
        )

    def test_a_restart_replaces_the_frames_a_live_buffer_holds(
        self, tmp_path: Path
    ) -> None:
        """A restored bundle is the buffer as of the checkpoint, not frames to append."""
        torch.manual_seed(0)
        teacher = _build_direct_force_teacher(seed=2)
        strategy = _make_strategy(
            tmp_path,
            student=_build_demo_model(),
            teacher=teacher,
            num_steps=2,
            hooks=[CheckpointHook(tmp_path / "ckpt", epoch_interval=1)],
        )
        strategy.run()
        stored = len(strategy.replay_buffer)

        strategy.restore_checkpoint(tmp_path / "ckpt")
        strategy._resume_or_seed(strategy.on_policy, strategy.replay_buffer)

        assert stored > 0
        assert len(strategy.replay_buffer) == stored
