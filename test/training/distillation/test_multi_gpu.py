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
"""Tests for multi-rank on-policy distillation."""

from __future__ import annotations

import os
import socket
import time
import warnings
from collections.abc import Callable, Iterator, Sequence
from itertools import combinations
from pathlib import Path
from queue import Empty
from typing import Any, ClassVar
from unittest.mock import patch

import pytest
import torch
from torch import distributed as dist

from nvalchemi.data import AtomicData, Batch
from nvalchemi.data.datapipes.dataloader import DataLoader
from nvalchemi.data.datapipes.in_memory_dataset import InMemoryDataset
from nvalchemi.data.datapipes.multidataset import MultiDataset
from nvalchemi.dynamics.base import (
    BaseDynamics,
    ConvergenceHook,
    DistributedPipeline,
    DynamicsStage,
    FusedStage,
)
from nvalchemi.dynamics.demo import DemoDynamics
from nvalchemi.dynamics.integrators.nvt_langevin import NVTLangevin
from nvalchemi.models.base import BaseModelMixin
from nvalchemi.training import (
    CheckpointHook,
    EnergyMSELoss,
    ForceMSELoss,
    OptimizerConfig,
    TrainingStage,
    ValidationConfig,
)
from nvalchemi.training.distillation import InitialStructures
from nvalchemi.training.distillation import strategy as distillation_strategy
from nvalchemi.training.distillation.replay import _same_device, build_mixed_loader
from nvalchemi.training.distillation.strategy import (
    _RANK_SEED_STRIDE,
    DistillationStrategy,
    _rank_local_propagator_seed,
)
from nvalchemi.training.distributed import (
    destroy_distributed,
    get_rank,
    get_world_size,
    init_distributed,
)
from nvalchemi.training.hooks import DDPHook
from nvalchemi.training.runtime import unwrap_model
from test.training.conftest import _build_demo_model
from test.training.distillation.conftest import (
    _INITIAL_ELEMENT,
    _REFERENCE_ELEMENT,
    _build_direct_force_teacher,
    _build_initial_dataset,
    _build_propagator_batch,
    _build_propagator_system,
    _build_reference_dataset,
    _build_small_dataset,
    _ListSource,
)
from test.training.distillation.test_on_policy import (
    _LANGEVIN_KWARGS,
    _make_composed_propagator,
    _make_on_policy_strategy,
    _make_recording_student,
    _make_scorer,
)
from test.training.distillation.test_relaxation import _make_relaxation_strategy

_WORKER_STEPS = 4
"""Optimizer steps every spawned rank takes, as two segments of two."""

_SEGMENT_KWARGS: dict[str, Any] = {
    "num_steps": _WORKER_STEPS,
    "training_steps_per_segment": 2,
    "generation_steps": 2,
}
"""Segment budget shared by the in-process and the spawned runs."""

_INITIAL_STRUCTURES = 4
"""Structures every initial dataset the spawned runs use holds."""

_FRAMES_PER_TRAJECTORY = (
    _WORKER_STEPS // _SEGMENT_KWARGS["training_steps_per_segment"]
) * _SEGMENT_KWARGS["generation_steps"]
"""Frames one initial structure yields: one per propagated step of every segment."""

_GENERATED_FRAMES = _INITIAL_STRUCTURES * _FRAMES_PER_TRAJECTORY
"""Frames the whole world generates, whatever world size it is sharded across."""

_UNEVEN_STRUCTURES = 3
"""Initial structures a two-rank world cannot deal out in equal shards."""

_RELAXED_KEY = "relaxed_score"
"""Graph-level key the scripted criterion converges every structure on."""

_UNION_STRUCTURES = 8
"""Structures the offline union run trains over, dealt out in equal shards."""

_UNION_SHARD_BATCH = 2
"""Graphs one rank of the union run draws per optimizer step."""

_UNION_EPOCHS = 2
"""Passes each side of the union run takes over its own loader."""

_RELAXATION_WORLDS = [
    pytest.param(4, False, id="even_shards"),
    pytest.param(3, False, id="uneven_shards"),
    pytest.param(2, True, id="recycled_one_row_shards"),
]
"""Structure counts and recycling a two-rank relaxation loop is dealt."""

_RANK_REPORT_TIMEOUT = 600.0
"""Seconds every spawned rank has to report before the world is declared hung."""


def _free_port() -> int:
    """Return an available localhost TCP port for process-group setup."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _make_distributed_strategy(
    *, rank: int = 0, world_size: int = 2, **overrides: Any
) -> DistillationStrategy:
    """Return an on-policy strategy that believes it is *rank* of *world_size*."""
    hooks = list(overrides.pop("hooks", []))
    return _make_on_policy_strategy(
        distributed_manager=_FakeManager(world_size=world_size, rank=rank),
        hooks=[DDPHook(), *hooks],
        **{**_SEGMENT_KWARGS, **overrides},
    )


def _make_restart_strategy(rank: int, **overrides: Any) -> DistillationStrategy:
    """Return a strategy whose ``DDPHook`` really pins *rank* to its own GPU.

    A manager reporting a device is what
    :meth:`~nvalchemi.training.hooks.DDPHook.prepare_strategy` reads before it
    calls :func:`torch.cuda.set_device`, so two of these in one process take
    the placement a launcher would give them without a spawn, a process group,
    or a collective that could leave a rank hanging.
    """
    manager = _FakeManager(world_size=2, rank=rank)
    manager.device = torch.device("cuda", rank)
    hooks = list(overrides.pop("hooks", []))
    return _make_on_policy_strategy(
        distributed_manager=manager,
        hooks=[DDPHook(), *hooks],
        **{**_SEGMENT_KWARGS, **overrides},
    )


def _make_replica_initial_dataset(n_systems: int = 4) -> InMemoryDataset:
    """Return *n_systems* byte-identical copies of one structure.

    Starting a world from replicas is how a run asks for one trajectory per rank
    from a single geometry, and it is the shape where sharding alone separates
    nothing: only the propagator's own RNG stream can make the ranks' frames
    differ.
    """
    return InMemoryDataset(
        in_memory_batch=Batch.from_data_list(
            [_build_propagator_system(_INITIAL_ELEMENT, 500) for _ in range(n_systems)]
        )
    )


def _make_union_strategy(**overrides: Any) -> DistillationStrategy:
    """Return the offline strategy the world and the single process both train.

    Everything the two sides have to hold in common is fixed here. Both loss
    terms reduce as a plain mean over the graphs of a batch, so a rank's
    gradient is the mean over its own shard and the all-reduce averages those
    means again: over equal-sized shards that is exactly the mean over their
    union, which is the gradient one process takes over the whole batch. The
    optimizer is plain SGD rather than Adam because ``m / sqrt(v)`` is
    invariant to the scale of the gradient, and would step identically whether
    or not the reduction averaged twice.
    """
    return DistillationStrategy(
        models={
            "student": _build_direct_force_teacher(seed=1),
            "teacher": _build_direct_force_teacher(seed=2),
        },
        optimizer_configs={
            "student": [
                OptimizerConfig(
                    optimizer_cls=torch.optim.SGD, optimizer_kwargs={"lr": 0.01}
                )
            ]
        },
        loss_fn=EnergyMSELoss(target_key="teacher_energy")
        + ForceMSELoss(target_key="teacher_forces"),
        num_epochs=_UNION_EPOCHS,
        **overrides,
    )


def _make_union_loader(batch_size: int) -> DataLoader:
    """Return an unshuffled loader over the union run's fixed dataset.

    Sharding is the hook's to install: handed this loader on a multi-rank
    launch, :meth:`DDPHook.prepare_dataloader` wraps a
    :class:`~torch.utils.data.DistributedSampler` in a batch sampler, which
    deals the rows out strided, so the batches ``world_size`` ranks draw at
    step ``k`` are together the batch one process draws at step ``k`` from the
    same dataset at ``world_size`` times the batch size.
    """
    return DataLoader(
        _build_small_dataset(_UNION_STRUCTURES),
        batch_size=batch_size,
        use_streams=False,
    )


def _make_composed_reference_dataset() -> MultiDataset:
    """Return a host-memory reference dataset composed of parts declaring no device.

    A :class:`MultiDataset` reports neither a ``target_device`` nor a resident
    batch, so its placement is only knowable by drawing from it.
    """
    scorer = _make_scorer(_build_direct_force_teacher(seed=2))
    return MultiDataset(
        _build_reference_dataset(scorer, 4, base_seed=700),
        _build_reference_dataset(scorer, 4, base_seed=740),
    )


def _structure_geometry(system: AtomicData) -> tuple[float, ...]:
    """Return the positions telling one structure of an initial dataset from another."""
    return tuple(round(float(value), 6) for value in system.positions.flatten())


def _loaded_structure_rows(state: Batch) -> list[int]:
    """Return the initial-dataset rows *state* was loaded from, matched by geometry."""
    rows = {
        _structure_geometry(system): index
        for index, system in enumerate(
            _build_initial_dataset().in_memory_batch.to_data_list()
        )
    }
    return sorted(rows[_structure_geometry(system)] for system in state.to_data_list())


def _make_annealing_propagator(student: BaseModelMixin) -> FusedStage:
    """Return a hot sampling stage composed with a cold one, both stochastic.

    A fused stage exposes no seed of its own — each thermostat holds its own,
    in a sub-stage — so this is the propagator a root-only seed probe misses.
    """
    hot = NVTLangevin(student, n_steps=2, **{**_LANGEVIN_KWARGS, "temperature": 1000.0})
    return hot + NVTLangevin(student, **_LANGEVIN_KWARGS)


def _replay_energies(strategy: DistillationStrategy) -> list[float]:
    """Return the teacher energy of every frame a run generated and stored."""
    frames = strategy.replay_buffer.dataset.in_memory_batch
    return sorted(round(float(value), 6) for value in frames.teacher_energy.flatten())


def _student_state(strategy: DistillationStrategy) -> dict[str, torch.Tensor]:
    """Return the trained student's state dict, detached on the host."""
    student = unwrap_model(strategy.models["student"])
    return {
        key: value.detach().cpu().clone() for key, value in student.state_dict().items()
    }


def _optimizer_state_devices(strategy: DistillationStrategy) -> set[torch.device]:
    """Return every device the strategy's optimizer state tensors sit on."""
    return {
        value.device
        for optimizer in strategy._optimizers
        for state in optimizer.state.values()
        for value in state.values()
        if torch.is_tensor(value)
    }


def _own_structure_rows(backend: str) -> dict[str, Any]:
    """Return the rows this rank claims and the ones its initial batch came from."""
    init_distributed(backend=backend)
    try:
        strategy = _make_on_policy_strategy(**_SEGMENT_KWARGS)
        structures = strategy.on_policy.initial_structures
        structures.shard(get_rank(), get_world_size())
        state = structures.initial_batch()
        return {
            "shard": list(strategy.structure_shard),
            "loaded": _loaded_structure_rows(state),
        }
    finally:
        destroy_distributed()


def _own_backfill_rows(backend: str) -> dict[str, Any]:
    """Return every row a budgeted source served this rank, initially or backfilled.

    The draw is the one the lifecycle's backfill makes, taken by hand so the
    source's own contract is what the world proves: a backfill behind a
    graduation serves this rank's shard alone, which two ranks refilling from
    one dataset would otherwise violate.
    """
    init_distributed(backend=backend)
    try:
        source = InitialStructures(_build_initial_dataset(), max_batch_size=1)
        source.shard(get_rank(), get_world_size())
        served = _loaded_structure_rows(source.initial_batch())
        while not source.exhausted:
            fresh = source.draw(limit=1, on_miss="skip")
            served.extend(_loaded_structure_rows(Batch.from_data_list(fresh)))
        return {"shard": list(source.rows), "served": sorted(served)}
    finally:
        destroy_distributed()


def _own_restart_bundle(backend: str, structures: int) -> dict[str, Any]:
    """Return the shard this rank was dealt and the restart bundle it wrote.

    The bundle records the rank and world its cursor was counted in, which is
    what lets a world whose shards differ in size refuse a peer's cursor.
    """
    init_distributed(backend=backend)
    try:
        source = InitialStructures(_build_initial_dataset(structures))
        source.shard(get_rank(), get_world_size())
        state = source.initial_batch()
        return {
            "shard": list(source.rows),
            "seeded": int(state.num_graphs),
            "bundle": source.state_dict(),
        }
    finally:
        destroy_distributed()


def _own_relaxation_rows(
    backend: str, structures: int, recycle: bool
) -> dict[str, Any]:
    """Run a relaxation segment loop on this rank and report the rows it relaxed.

    Every trajectory converges on its first step, so each segment graduates the
    whole one-structure batch and the lifecycle backfills it from the cursor:
    the route a real rank's backfill takes, rather than a hand-driven draw.
    """
    dataset = _RowRecordingDataset(_build_initial_dataset(structures))
    strategy = _make_relaxation_strategy(
        structures=InitialStructures(dataset, max_batch_size=1, recycle=recycle),
        convergence_hook=ConvergenceHook(
            criteria=[{"key": _RELAXED_KEY, "threshold": 0.5}],
            source_status=0,
            target_status=1,
        ),
        hooks=[DDPHook(backend=backend, find_unused_parameters=True)],
        **_SEGMENT_KWARGS,
    )
    strategy.on_policy.dynamics.register_hook(_ConvergeOnFirstStep())
    dataset.rows_served.clear()
    with warnings.catch_warnings(record=True) as reported:
        warnings.simplefilter("always")
        try:
            strategy.run()
        except ValueError as exc:
            return {"error": str(exc)}
    source = strategy.on_policy.initial_structures
    return {
        "shard": list(source.rows),
        "served": list(dataset.rows_served),
        "wraps": source.wraps,
        "steps": strategy.step_count,
        "warnings": [str(entry.message) for entry in reported],
    }


def _run_worker(
    rank: int,
    world_size: int,
    port: int,
    backend: str,
    local_rank: int,
    composed: bool,
    probe: str | None,
    structures: int,
    recycle: bool,
    result_queue: Any,
) -> None:
    """Run one rank of the segment loop and report what it produced."""
    os.environ.update(
        {
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": str(port),
            "RANK": str(rank),
            "WORLD_SIZE": str(world_size),
            "LOCAL_RANK": str(local_rank),
        }
    )
    torch.manual_seed(0)
    if probe == "structures":
        result_queue.put((rank, _own_structure_rows(backend)))
        return
    if probe == "backfill":
        result_queue.put((rank, _own_backfill_rows(backend)))
        return
    if probe == "bundle":
        result_queue.put((rank, _own_restart_bundle(backend, structures)))
        return
    if probe == "relaxation":
        result_queue.put((rank, _own_relaxation_rows(backend, structures, recycle)))
        return
    recorder = _RecordingValidationHook()
    overrides: dict[str, Any] = {}
    if composed:
        student = _build_demo_model()
        overrides = {
            "student": student,
            "config_overrides": {
                "dynamics": _make_annealing_propagator(student),
                "initial_structures": InitialStructures(
                    _make_replica_initial_dataset()
                ),
            },
        }
    elif structures != _INITIAL_STRUCTURES:
        overrides = {
            "config_overrides": {
                "initial_structures": InitialStructures(
                    _build_initial_dataset(structures)
                )
            }
        }
    strategy = _make_on_policy_strategy(
        device="cuda" if backend == "nccl" else "cpu",
        hooks=[recorder, DDPHook(backend=backend, find_unused_parameters=True)],
        validation_config=ValidationConfig(
            validation_data=[
                _build_propagator_batch(_REFERENCE_ELEMENT, 2, base_seed=900)
            ],
            every_n_steps=2,
        ),
        **_SEGMENT_KWARGS,
        **overrides,
    )
    with warnings.catch_warnings(record=True) as reported:
        warnings.simplefilter("always")
        strategy.run()
    result_queue.put(
        (
            rank,
            {
                "state": {
                    key: value.tolist()
                    for key, value in _student_state(strategy).items()
                },
                "energies": _replay_energies(strategy),
                "frames": len(strategy.replay_buffer),
                "device": str(strategy.devices[0]),
                "steps": strategy.step_count,
                "validations": recorder.calls,
                "shard": list(strategy.structure_shard),
                "warnings": [str(entry.message) for entry in reported],
            },
        )
    )


def _run_union_worker(rank: int, world_size: int, port: int, result_queue: Any) -> None:
    """Train one rank of the offline loop on its shard and report the student."""
    os.environ.update(
        {
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": str(port),
            "RANK": str(rank),
            "WORLD_SIZE": str(world_size),
            "LOCAL_RANK": "0",
        }
    )
    torch.manual_seed(0)
    strategy = _make_union_strategy(hooks=[DDPHook(backend="gloo")])
    strategy.run(_make_union_loader(_UNION_SHARD_BATCH))
    result_queue.put(
        (
            rank,
            {
                "state": {
                    key: value.tolist()
                    for key, value in _student_state(strategy).items()
                },
                "steps": strategy.step_count,
            },
        )
    )


def _spawn_ranks(
    worker: Callable[..., None], rank_args: Sequence[tuple[Any, ...]]
) -> dict[int, dict[str, Any]]:
    """Spawn one process per entry of *rank_args* and collect what each reports.

    Every worker is handed its own arguments followed by the result queue. The
    wait polls the children rather than blocking on the queue for the whole
    timeout, so a rank that dies without reporting — taking its peers down into
    a collective that will never complete — fails the call in seconds.
    """
    ctx = torch.multiprocessing.get_context("spawn")
    result_queue = ctx.Queue()
    procs = [
        ctx.Process(target=worker, args=(*args, result_queue)) for args in rank_args
    ]
    for proc in procs:
        proc.start()
    results: dict[int, dict[str, Any]] = {}
    deadline = time.monotonic() + _RANK_REPORT_TIMEOUT
    try:
        while len(results) < len(procs):
            try:
                rank, payload = result_queue.get(timeout=1)
            except Empty:
                dead = {
                    index: proc.exitcode
                    for index, proc in enumerate(procs)
                    if proc.exitcode not in (None, 0)
                }
                assert not dead, f"ranks exited before reporting: {dead}."
                assert time.monotonic() < deadline, (
                    f"{len(procs) - len(results)} rank(s) never reported."
                )
                continue
            results[rank] = payload
        for proc in procs:
            proc.join(timeout=60)
            assert proc.exitcode == 0
    finally:
        for proc in procs:
            if proc.is_alive():
                proc.terminate()
    return results


def _run_ranks(
    world_size: int,
    *,
    backend: str = "gloo",
    local_ranks: tuple[int, ...] | None = None,
    composed: bool = False,
    probe: str | None = None,
    structures: int = _INITIAL_STRUCTURES,
    recycle: bool = False,
) -> dict[int, dict[str, Any]]:
    """Spawn *world_size* ranks of the segment loop and collect their results.

    ``local_ranks`` names the node-local rank each process reports, which is
    what decides its device: ``None`` places rank ``r`` on device ``r`` (one
    node), while zeros everywhere is the one-rank-per-node placement.
    ``composed`` swaps the bare thermostat for a fused one and the distinct
    initial structures for replicas of a single geometry, and ``structures``
    resizes the dataset the ranks share out. ``probe`` stops each rank after
    its initial batch (``"structures"``), after a hand-driven backfill behind
    it (``"backfill"``), or after the restart bundle the initial batch
    established (``"bundle"``), or runs a one-structure relaxation loop whose
    lifecycle backfills from the shard (``"relaxation"``, honoring ``recycle``).
    """
    ranks = local_ranks or tuple(range(world_size))
    port = _free_port()
    return _spawn_ranks(
        _run_worker,
        [
            (
                rank,
                world_size,
                port,
                backend,
                ranks[rank],
                composed,
                probe,
                structures,
                recycle,
            )
            for rank in range(world_size)
        ],
    )


def _assert_one_student(results: dict[int, dict[str, Any]]) -> None:
    """Assert every rank came out of the run holding the same student weights."""
    reference = results[min(results)]["state"]
    for result in results.values():
        for key, value in reference.items():
            torch.testing.assert_close(
                torch.as_tensor(result["state"][key]), torch.as_tensor(value)
            )


def _assert_disjoint_frames(
    results: dict[int, dict[str, Any]], *, total: int = _GENERATED_FRAMES
) -> None:
    """Assert the world dealt its structures out rather than copying them to every rank.

    Disjoint energies alone would pass a world whose ranks each propagated the
    whole structure set and merely separated their streams; it is the counts
    that say the trajectories were shared out, summing to the single-process
    yield however many ranks generated them. A structure set the world cannot
    deal out evenly still sums to *total*, one shard simply running a trajectory
    longer than the other.
    """
    generated = [frozenset(result["energies"]) for result in results.values()]
    assert all(generated)
    assert not frozenset.intersection(*generated)
    counts = [result["frames"] for result in results.values()]
    assert sum(counts) == total
    assert max(counts) - min(counts) <= _FRAMES_PER_TRAJECTORY


class _FakeManager:
    """Distributed manager reporting a fixed world size, rank, and node-local rank."""

    def __init__(
        self, *, world_size: int = 2, rank: int = 0, local_rank: int | None = None
    ) -> None:
        """Report a world of *world_size* ranks, seen from *rank*."""
        self.world_size = world_size
        self.rank = rank
        self.global_rank = rank
        self.local_rank = rank if local_rank is None else local_rank
        self.device = torch.device("cpu")
        self.broadcast_buffers = False
        self.find_unused_parameters = True

    def is_initialized(self) -> bool:
        """Report communication as established for any multi-rank world."""
        return self.world_size > 1


class _ConcentratedWorld(_FakeManager):
    """Manager whose other ranks report a replay device that is not their own."""

    def all_reduce(
        self,
        tensor: torch.Tensor,
        *,
        op: Any = None,  # noqa: ARG002
    ) -> torch.Tensor:
        """Return the raised flag a MAX reduce brings back from another rank."""
        return tensor.fill_(1)


class _RecordingDDP(torch.nn.Module):
    """Data-parallel stand-in counting the forwards routed through the wrapper."""

    calls: ClassVar[list[dict[str, Any]]] = []
    forwards: ClassVar[int] = 0

    def __init__(self, module: torch.nn.Module, **kwargs: Any) -> None:
        """Wrap *module* and record the data-parallel options it was given."""
        super().__init__()
        self.module = module
        type(self).calls.append(kwargs)

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Count the pass whose gradients a real wrapper would all-reduce."""
        type(self).forwards += 1
        return self.module(*args, **kwargs)


class _RecordingValidationHook:
    """Count the validation passes a run closes."""

    frequency = 1
    stage = TrainingStage.AFTER_VALIDATION

    def __init__(self) -> None:
        """Start with no validation passes seen."""
        self.calls = 0

    def __call__(self, ctx: Any, stage: Any) -> None:  # noqa: ARG002
        """Count one closed validation pass."""
        self.calls += 1


class _ModelProbe:
    """Report the strategy's live models at every forward pass."""

    frequency = 1
    stage = TrainingStage.BEFORE_FORWARD

    def __init__(self) -> None:
        """Start with an empty trace of wrapped model names."""
        self.wrapped: list[list[str]] = []

    def __call__(self, ctx: Any, stage: Any) -> None:  # noqa: ARG002
        """Record which models a data-parallel wrapper is holding."""
        self.wrapped.append(
            sorted(
                key
                for key, model in ctx.workflow.models.items()
                if isinstance(model, _RecordingDDP)
            )
        )


class _StubPropagator:
    """Propagator stand-in exposing no RNG seed at all."""


class _PropertySeededPropagator:
    """Propagator stand-in exposing its seed through a getter-only property."""

    def __init__(self) -> None:
        """Hold the writable private seed the read-only public name forwards."""
        self._random_seed = 7

    @property
    def random_seed(self) -> int:
        """Return the seed under the public name the probe reaches first."""
        return self._random_seed


class _RowRecordingDataset(InMemoryDataset):
    """In-memory dataset remembering every row it was asked to load."""

    def __init__(self, source: InMemoryDataset) -> None:
        """Serve *source*'s batch and start with no rows recorded."""
        super().__init__(in_memory_batch=source.in_memory_batch)
        self.rows_served: list[int] = []

    def __getitem__(self, index: int) -> Any:
        """Record *index* and serve the row."""
        self.rows_served.append(int(index))
        return super().__getitem__(index)

    def load_batches(self, batch_index_lists: Any, *args: Any, **kwargs: Any) -> Any:
        """Record every index of every batch and serve them."""
        self.rows_served.extend(
            int(index) for indices in batch_index_lists for index in indices
        )
        return super().load_batches(batch_index_lists, *args, **kwargs)


class _ConvergeOnFirstStep:
    """Score every live structure as relaxed, so each segment graduates its batch."""

    frequency = 1
    stage = DynamicsStage.BEFORE_STEP

    def __call__(self, ctx: Any, stage: Any) -> None:  # noqa: ARG002
        """Write the converging score onto every graph of the live batch."""
        batch = ctx.batch
        batch[_RELAXED_KEY] = torch.zeros(batch.num_graphs, 1, device=batch.device)


class _GeneratorKick(BaseDynamics):
    """Stochastic stage drawing its noise from a torch.Generator it holds."""

    __needs_keys__: set[str] = set()
    __provides_keys__: set[str] = {"velocities"}

    def __init__(self, model: BaseModelMixin, **kwargs: Any) -> None:
        """Hold a generator no seed-attribute probe can offset."""
        super().__init__(model=model, **kwargs)
        self.generator = torch.Generator().manual_seed(1234)

    def pre_update(self, batch: Batch) -> None:
        """Kick the velocities from a stream the rank offset cannot reach."""
        batch.velocities.add_(
            torch.randn(batch.velocities.shape, generator=self.generator)
        )

    def post_update(self, batch: Batch) -> None:
        """Leave the post-force half of the step alone."""


@pytest.fixture(autouse=True)
def _reset_recording_ddp() -> None:
    """Reset the data-parallel stand-in's counters before every test."""
    _RecordingDDP.calls.clear()
    _RecordingDDP.forwards = 0


@pytest.fixture(autouse=True)
def _restore_current_cuda_device() -> Iterator[None]:
    """Put back the current CUDA device a test pinning this process leaves.

    ``DDPHook`` pins through :func:`torch.cuda.set_device`, and so does a test
    standing in for a launcher; it is process state a later test measuring an
    index-less ``cuda`` would otherwise read as its own.
    """
    if not torch.cuda.is_available():
        yield
        return
    previous = torch.cuda.current_device()
    try:
        yield
    finally:
        torch.cuda.set_device(previous)


class TestStructureSharding:
    def test_ranks_take_disjoint_strided_shards_that_cover_the_structures(self) -> None:
        """Every initial structure is propagated once, by exactly one rank."""
        shards = [
            _make_distributed_strategy(
                rank=rank,
                world_size=3,
                config_overrides={
                    "initial_structures": InitialStructures(_build_initial_dataset(8))
                },
            ).structure_shard
            for rank in range(3)
        ]

        assert shards == [(0, 3, 6), (1, 4, 7), (2, 5)]
        assert all(
            not set(left) & set(right) for left, right in combinations(shards, 2)
        )
        assert sorted(index for shard in shards for index in shard) == list(range(8))

    def test_shards_follow_the_global_rank_rather_than_the_node_local_one(self) -> None:
        """Across nodes the node-local rank repeats, so sharding on it would too."""
        strategy = _make_on_policy_strategy(
            num_steps=2,
            distributed_manager=_FakeManager(world_size=4, rank=3, local_rank=1),
            config_overrides={
                "initial_structures": InitialStructures(_build_initial_dataset(8))
            },
        )

        assert strategy.structure_shard == (3, 7)

    def test_a_single_rank_run_propagates_every_structure(self) -> None:
        """Sharding is a no-op on one process, so single-rank runs are unchanged."""
        strategy = _make_on_policy_strategy(num_steps=2)

        assert strategy.structure_shard == (0, 1, 2, 3)

    def test_a_source_publishing_neither_rows_nor_a_count_deals_its_own(self) -> None:
        """A streaming source shards itself, so the strategy reports no rows and checks no count."""
        source = _ListSource(
            [
                _build_propagator_system(_INITIAL_ELEMENT, 500 + index)
                for index in range(2)
            ]
        )
        strategy = _make_on_policy_strategy(
            num_steps=2,
            distributed_manager=_FakeManager(world_size=4, rank=3, local_rank=1),
            config_overrides={"initial_structures": source},
        )

        assert strategy.structure_shard == ()
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            strategy._validate_structure_shards(strategy.on_policy)
            strategy._warn_unequal_structure_shards(strategy.on_policy)

    def test_the_structure_shard_reads_the_rows_the_run_installed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """After a run the property reports the source's own rows, not a fresh deal."""
        monkeypatch.setattr(torch.nn.parallel, "DistributedDataParallel", _RecordingDDP)
        strategy = _make_distributed_strategy(rank=1)

        strategy.run()

        structures = strategy.on_policy.initial_structures
        assert strategy.structure_shard == structures.rows == (1, 3)
        assert structures.cursor == len(structures.rows)

    def test_the_structure_shard_is_available_before_the_initial_batch_is_drawn(
        self,
    ) -> None:
        """A restart backfills without an initial batch, and still serves only its rows."""
        strategy = _make_distributed_strategy(rank=1)

        assert strategy.structure_shard == (1, 3)

    def test_each_rank_generates_and_labels_its_own_frames(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Frames, and the teacher passes paying for them, never leave their rank."""
        monkeypatch.setattr(torch.nn.parallel, "DistributedDataParallel", _RecordingDDP)
        energies = []
        for rank in range(2):
            strategy = _make_distributed_strategy(rank=rank)
            scorer = strategy.on_policy.teacher_scorer
            with patch.object(scorer, "label", wraps=scorer.label) as labeled:
                strategy.run()
            assert {call.args[0].num_graphs for call in labeled.call_args_list} == {2}
            energies.append(_replay_energies(strategy))

        assert len(energies[0]) == len(energies[1])
        assert not set(energies[0]) & set(energies[1])

    def test_fewer_structures_than_ranks_is_rejected(self) -> None:
        """A rank dealt no structure of its own would have nothing to propagate."""
        strategy = _make_distributed_strategy(
            world_size=8,
            config_overrides={
                "initial_structures": InitialStructures(_build_initial_dataset(4))
            },
        )

        with pytest.raises(ValueError, match="at least one for each"):
            strategy.run()

    def test_structures_that_do_not_divide_across_the_world_are_reported(self) -> None:
        """Unequal shards reweight the frames a shorter one generates."""
        strategy = _make_distributed_strategy(
            world_size=3,
            config_overrides={
                "initial_structures": InitialStructures(_build_initial_dataset(4))
            },
        )

        with pytest.warns(UserWarning, match="do not divide evenly") as reported:
            strategy._warn_unequal_structure_shards(strategy.on_policy)

        assert "2.00x" in str(reported[0].message)

    def test_structures_that_divide_evenly_are_reported_as_nothing(self) -> None:
        """A whole multiple of the world weights every generated frame alike."""
        strategy = _make_distributed_strategy()

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            strategy._warn_unequal_structure_shards(strategy.on_policy)

    def test_a_single_rank_run_reports_no_shard_imbalance(self) -> None:
        """One process takes every structure, so there is nothing to deal out."""
        strategy = _make_on_policy_strategy(
            num_steps=2,
            config_overrides={
                "initial_structures": InitialStructures(_build_initial_dataset(5))
            },
        )

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            strategy._warn_unequal_structure_shards(strategy.on_policy)

    def test_a_budgeted_source_packs_and_backfills_from_its_own_shard(self) -> None:
        """A size budget packs the initial batch out of this rank's rows alone."""
        source = InitialStructures(_build_initial_dataset(), max_batch_size=1)
        source.shard(1, 2)

        seeded = _loaded_structure_rows(source.initial_batch())
        backfilled = [
            _loaded_structure_rows(Batch.from_data_list([data]))[0]
            for data in source.draw(limit=1, on_miss="skip")
        ]

        assert seeded == [1]
        assert backfilled == [3]
        assert source.exhausted


class TestRankSeedStreams:
    def test_the_mixture_seed_is_moved_onto_this_rank_stride(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Ranks sharing a base seed would otherwise draw the same reference samples."""
        monkeypatch.setattr(torch.nn.parallel, "DistributedDataParallel", _RecordingDDP)
        strategy = _make_distributed_strategy(rank=1, config_overrides={"seed": 5})

        with patch.object(
            distillation_strategy, "build_mixed_loader", wraps=build_mixed_loader
        ) as built:
            strategy.run()

        assert {call.kwargs["seed"] for call in built.call_args_list} == {
            5 + _RANK_SEED_STRIDE
        }

    def test_a_single_rank_run_keeps_the_configured_mixture_seed(self) -> None:
        """The offset is zero on rank zero, so single-process draws are unchanged."""
        strategy = _make_on_policy_strategy(num_steps=2, config_overrides={"seed": 5})

        with patch.object(
            distillation_strategy, "build_mixed_loader", wraps=build_mixed_loader
        ) as built:
            strategy.run()

        assert {call.kwargs["seed"] for call in built.call_args_list} == {5}

    def test_seed_streams_follow_the_global_rank_rather_than_the_node_local_one(
        self,
    ) -> None:
        """Every node's rank zero would otherwise draw one stream."""
        strategy = _make_on_policy_strategy(
            num_steps=2,
            distributed_manager=_FakeManager(world_size=4, rank=3, local_rank=1),
        )

        assert strategy._rank_seed_offset() == 3 * _RANK_SEED_STRIDE

    def test_the_propagator_seed_is_moved_onto_this_rank_stride(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Nothing but the loop puts the propagator on this rank's own stream."""
        monkeypatch.setattr(torch.nn.parallel, "DistributedDataParallel", _RecordingDDP)
        strategy = _make_distributed_strategy(rank=1)

        with patch.object(
            distillation_strategy,
            "_rank_local_propagator_seed",
            wraps=distillation_strategy._rank_local_propagator_seed,
        ) as entered:
            strategy.run()

        assert {call.args[1] for call in entered.call_args_list} == {_RANK_SEED_STRIDE}

    def test_a_stochastic_propagator_draws_from_its_own_rank_stream(self) -> None:
        """A counter-based thermostat would otherwise kick every rank identically."""
        dynamics = NVTLangevin(_make_recording_student(), **_LANGEVIN_KWARGS)
        base = dynamics._random_seed

        with _rank_local_propagator_seed(dynamics, _RANK_SEED_STRIDE):
            offset = dynamics._random_seed

        assert offset == base + _RANK_SEED_STRIDE
        assert dynamics._random_seed == base

    def test_a_composed_propagator_moves_every_stochastic_sub_stage(self) -> None:
        """A fused stage holds no seed itself; the thermostat drawing the noise does."""
        fused = _make_annealing_propagator(_make_recording_student())
        bases = [sub._random_seed for _, sub in fused.sub_stages]

        with _rank_local_propagator_seed(fused, _RANK_SEED_STRIDE):
            offsets = [sub._random_seed for _, sub in fused.sub_stages]

        assert not hasattr(fused, "_random_seed")
        assert offsets == [base + _RANK_SEED_STRIDE for base in bases]
        assert [sub._random_seed for _, sub in fused.sub_stages] == bases

    def test_one_integrator_composed_twice_is_strided_once(self) -> None:
        """Two sub-stages can be the same object, which must not take two strides."""
        dynamics = NVTLangevin(_make_recording_student(), **_LANGEVIN_KWARGS)
        base = dynamics._random_seed

        with _rank_local_propagator_seed(dynamics + dynamics, _RANK_SEED_STRIDE):
            offset = dynamics._random_seed

        assert offset == base + _RANK_SEED_STRIDE
        assert dynamics._random_seed == base

    def test_a_rank_keyed_stage_map_is_walked_as_a_pipeline_holds_it(self) -> None:
        """A pipeline keys its stages by rank, so its propagators sit in a mapping."""
        dynamics = NVTLangevin(_make_recording_student(), **_LANGEVIN_KWARGS)
        base = dynamics._random_seed

        with _rank_local_propagator_seed(
            DistributedPipeline(stages={0: dynamics}), _RANK_SEED_STRIDE
        ):
            offset = dynamics._random_seed

        assert offset == base + _RANK_SEED_STRIDE
        assert dynamics._random_seed == base

    def test_a_propagator_hiding_its_seed_is_left_alone(self) -> None:
        """Nothing is written to a propagator naming its seed somewhere else."""
        propagator = _StubPropagator()

        with _rank_local_propagator_seed(propagator, _RANK_SEED_STRIDE):
            pass

        assert not hasattr(propagator, "random_seed")

    def test_a_seed_read_through_a_property_is_moved_under_its_writable_name(
        self,
    ) -> None:
        """A getter-only public name would raise where the offsets are applied."""
        propagator = _PropertySeededPropagator()

        with _rank_local_propagator_seed(propagator, _RANK_SEED_STRIDE):
            offset = propagator.random_seed

        assert offset == 7 + _RANK_SEED_STRIDE
        assert propagator._random_seed == 7

    def test_rank_zero_leaves_the_propagator_seed_untouched(self) -> None:
        """A zero offset must not rewrite the seed a single-process run reads."""
        dynamics = NVTLangevin(_make_recording_student(), **_LANGEVIN_KWARGS)

        with _rank_local_propagator_seed(dynamics, 0):
            assert dynamics._random_seed == _LANGEVIN_KWARGS["random_seed"]


class TestSharedStreamReport:
    def test_a_generator_stage_beside_a_seeded_one_is_named(self) -> None:
        """One seed found in the tree must not pass the stages left beside it."""
        student = _build_demo_model()
        strategy = _make_distributed_strategy(
            student=student,
            config_overrides={
                "dynamics": _GeneratorKick(student, n_steps=1)
                + NVTLangevin(student, **_LANGEVIN_KWARGS)
            },
        )

        with pytest.warns(UserWarning, match="shared random stream") as reported:
            strategy._warn_shared_propagator_streams(strategy.on_policy)

        message = str(reported[0].message)
        assert "'_GeneratorKick'" in message
        assert "NVTLangevin" not in message
        assert "FusedStage" not in message

    def test_a_fully_seeded_composition_is_reported_as_nothing(self) -> None:
        """Every stage of an annealing propagator holds a seed of its own."""
        student = _build_demo_model()
        strategy = _make_distributed_strategy(
            student=student,
            config_overrides={"dynamics": _make_annealing_propagator(student)},
        )

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            strategy._warn_shared_propagator_streams(strategy.on_policy)

    def test_a_deterministic_stage_beside_a_seeded_one_is_not_named(self) -> None:
        """A relax-then-sample propagator has no stream to separate in its first half."""
        student = _build_demo_model()
        strategy = _make_distributed_strategy(
            student=student,
            config_overrides={
                "dynamics": DemoDynamics(student, n_steps=1)
                + NVTLangevin(student, **_LANGEVIN_KWARGS)
            },
        )

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            strategy._warn_shared_propagator_streams(strategy.on_policy)

    def test_rank_zero_reports_a_propagator_it_could_not_separate(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Rank zero applies no offset, and is the rank whose stderr survives."""
        monkeypatch.setattr(torch.nn.parallel, "DistributedDataParallel", _RecordingDDP)
        student = _build_demo_model()
        strategy = _make_distributed_strategy(
            rank=0,
            student=student,
            config_overrides={"dynamics": DemoDynamics(student, n_steps=1)},
        )

        with pytest.warns(UserWarning, match="could not be moved onto per-rank"):
            strategy.run()

        assert strategy.step_count == _WORKER_STEPS

    def test_a_single_rank_run_reports_nothing(self) -> None:
        """One process draws one stream by definition, so the report would be noise."""
        student = _build_demo_model()
        strategy = _make_on_policy_strategy(
            num_steps=2,
            student=student,
            config_overrides={"dynamics": DemoDynamics(student, n_steps=1)},
        )

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            strategy._warn_shared_propagator_streams(strategy.on_policy)


class TestReplayPlacementAcrossRanks:
    def test_a_reference_dataset_pinned_to_one_indexed_device_is_reported(self) -> None:
        """Every rank would stage its buffer and collate its mixture on that device."""
        strategy = _make_distributed_strategy(rank=1)
        strategy.devices = [torch.device("cuda:1")]
        strategy.reference_dataset.target_device = torch.device("cuda:0")

        with pytest.warns(UserWarning, match="not the device every rank trains on"):
            device = strategy._resolve_replay_device(strategy.on_policy)

        assert device == torch.device("cuda:0")

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_an_index_less_reference_device_is_left_alone(self) -> None:
        """'cuda' names whichever device the launcher pinned this rank to."""
        strategy = _make_distributed_strategy(rank=1)
        strategy.devices = [torch.device("cuda", torch.cuda.current_device())]
        strategy.reference_dataset.target_device = torch.device("cuda")

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            device = strategy._resolve_replay_device(strategy.on_policy)

        assert _same_device(device, strategy.devices[0])

    @pytest.mark.multigpu
    def test_an_index_less_replay_device_names_the_device_this_rank_pinned(
        self,
    ) -> None:
        """Left index-less it would stage every rank's frames under one spelling."""
        strategy = _make_distributed_strategy(
            rank=1, replay_ratio=1.0, config_overrides={"replay_device": "cuda"}
        )
        strategy.devices = [torch.device("cuda", 1)]
        torch.cuda.set_device(1)

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            device = strategy._resolve_replay_device(strategy.on_policy)

        assert device == torch.device("cuda", 1)

    def test_a_host_memory_replay_device_is_passed_through(self) -> None:
        """Only an index-less accelerator is resolved; ``cpu`` names its device."""
        strategy = _make_distributed_strategy(
            rank=1, replay_ratio=1.0, config_overrides={"replay_device": "cpu"}
        )
        strategy.devices = [torch.device("cuda:1")]

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            device = strategy._resolve_replay_device(strategy.on_policy)

        assert device == torch.device("cpu")

    def test_a_host_memory_reference_dataset_is_left_alone(self) -> None:
        """A mixture collated on the host concentrates nothing on one accelerator."""
        strategy = _make_distributed_strategy(rank=1)
        strategy.devices = [torch.device("cuda:1")]

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            device = strategy._resolve_replay_device(strategy.on_policy)

        assert device == torch.device("cpu")

    def test_a_single_rank_run_stages_where_the_reference_dataset_is(self) -> None:
        """One process holds one buffer, so an indexed reference concentrates nothing."""
        strategy = _make_on_policy_strategy(num_steps=2)
        strategy.reference_dataset.target_device = torch.device("cuda:0")

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            device = strategy._resolve_replay_device(strategy.on_policy)

        assert device == torch.device("cuda:0")

    def test_a_composed_reference_dataset_is_measured_rather_than_assumed(
        self,
    ) -> None:
        """A composition declares no device, and host memory is a placement too."""
        strategy = _make_distributed_strategy(
            rank=1, reference_dataset=_make_composed_reference_dataset()
        )
        strategy.devices = [torch.device("cuda:1")]

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            device = strategy._resolve_replay_device(strategy.on_policy)

        assert device == torch.device("cpu")

    def test_a_reference_dataset_moved_after_setup_is_staged_where_it_now_is(
        self,
    ) -> None:
        """The documented remedy runs after construction, so nothing may be memoized."""
        strategy = _make_distributed_strategy(rank=1)
        strategy.devices = [torch.device("cuda:1")]
        strategy.reference_dataset.target_device = torch.device("cuda:1")

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            device = strategy._resolve_replay_device(strategy.on_policy)

        assert device == torch.device("cuda:1")

    def test_a_shared_reference_dataset_is_reported_from_the_rank_it_piles_onto(
        self,
    ) -> None:
        """Rank zero owns the device the world piles onto, so only its peers see it."""
        strategy = _make_on_policy_strategy(
            num_steps=2, distributed_manager=_ConcentratedWorld(world_size=2, rank=0)
        )
        strategy.devices = [torch.device("cuda:0")]
        strategy.reference_dataset.target_device = torch.device("cuda:0")

        with pytest.warns(UserWarning, match="not the device every rank trains on"):
            device = strategy._resolve_replay_device(strategy.on_policy)

        assert device == torch.device("cuda:0")

    def test_a_reference_dataset_each_rank_built_for_itself_is_left_alone(self) -> None:
        """Every peer staging on its own device leaves the reduced flag down."""
        strategy = _make_on_policy_strategy(
            num_steps=2, distributed_manager=_FakeManager(world_size=2, rank=0)
        )
        strategy.devices = [torch.device("cuda:0")]
        strategy.reference_dataset.target_device = torch.device("cuda:0")

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            device = strategy._resolve_replay_device(strategy.on_policy)

        assert device == torch.device("cuda:0")

    def test_a_host_memory_rank_still_hears_a_peer_that_concentrates(self) -> None:
        """Every rank reads the world's verdict, whatever its own placement is."""
        strategy = _make_on_policy_strategy(
            num_steps=2, distributed_manager=_ConcentratedWorld(world_size=2, rank=0)
        )
        strategy.devices = [torch.device("cuda:0")]

        with pytest.warns(UserWarning, match="not the device every rank trains on"):
            device = strategy._resolve_replay_device(strategy.on_policy)

        assert device == torch.device("cpu")


class TestGradientSynchronization:
    def test_only_the_student_is_wrapped_for_the_all_reduce(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The frozen teacher is replicated per rank and stays out of the collective."""
        monkeypatch.setattr(torch.nn.parallel, "DistributedDataParallel", _RecordingDDP)
        probe = _ModelProbe()
        strategy = _make_distributed_strategy(hooks=[probe])

        strategy.run()

        assert _RecordingDDP.calls == [
            {
                "find_unused_parameters": True,
                "broadcast_buffers": False,
                "static_graph": False,
            }
        ]
        assert probe.wrapped and all(keys == ["student"] for keys in probe.wrapped)

    def test_generation_bypasses_the_wrapper_that_training_goes_through(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Only gradient steps cross ranks; propagating and labeling stay local."""
        monkeypatch.setattr(torch.nn.parallel, "DistributedDataParallel", _RecordingDDP)
        student = _make_recording_student()
        strategy = _make_distributed_strategy(student=student)

        strategy.run()

        assert _RecordingDDP.forwards == strategy.step_count == _WORKER_STEPS
        assert len(student.forwards) > _RecordingDDP.forwards

    def test_an_unsynchronized_student_is_rejected(self) -> None:
        """Without a wrapper every rank trains, and generates from, its own policy."""
        strategy = _make_on_policy_strategy(
            num_steps=2, distributed_manager=_FakeManager(world_size=2)
        )

        with pytest.raises(ValueError, match="gradients have to be synchronized"):
            strategy.run()

    def test_the_guard_reads_the_student_rather_than_the_propagator(self) -> None:
        """A propagator over another model must not clear the synchronization check."""
        strategy = _make_on_policy_strategy(
            num_steps=2, distributed_manager=_FakeManager(world_size=2)
        )
        strategy.on_policy.dynamics = NVTLangevin(
            _build_demo_model(), **_LANGEVIN_KWARGS
        )

        with pytest.raises(ValueError, match="gradients have to be synchronized"):
            strategy.run()

        assert strategy.step_count == 0

    def test_a_single_rank_run_needs_no_wrapper(self) -> None:
        """The contract is about ranks, so a lone process runs the loop bare."""
        strategy = _make_on_policy_strategy(
            num_steps=2, distributed_manager=_FakeManager(world_size=1)
        )

        strategy.run()

        assert strategy.step_count == 2


class TestRankLocalModelModes:
    def test_the_mode_context_is_given_the_module_the_wrapper_owns(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Handed the wrapper instead, it would take a composition's mode off it."""
        monkeypatch.setattr(torch.nn.parallel, "DistributedDataParallel", _RecordingDDP)
        student = _build_demo_model()
        composed = _make_composed_propagator(student, _build_demo_model())
        strategy = _make_distributed_strategy(
            student=student,
            config_overrides={"dynamics": NVTLangevin(composed, **_LANGEVIN_KWARGS)},
        )

        with patch.object(
            distillation_strategy,
            "_eval_propagator_model",
            wraps=distillation_strategy._eval_propagator_model,
        ) as entered:
            strategy.run()

        assert entered.call_args_list
        assert all(call.args[1] is student for call in entered.call_args_list)

    def test_a_composed_propagator_generates_in_eval_mode_under_a_wrapper(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Training goes through the wrapper, so nothing else takes the composition down."""
        monkeypatch.setattr(torch.nn.parallel, "DistributedDataParallel", _RecordingDDP)
        student = _build_demo_model()
        composed = _make_composed_propagator(student, _build_demo_model())
        strategy = _make_distributed_strategy(
            student=student,
            config_overrides={"dynamics": NVTLangevin(composed, **_LANGEVIN_KWARGS)},
        )

        strategy.run()

        assert composed.forwards
        assert all(reading == (False, False) for reading in composed.forwards)
        assert composed.training is True


class TestRankConsistentBookkeeping:
    def test_validation_runs_on_every_rank_while_only_rank_zero_checkpoints(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Validation all-reduces its metrics, so no rank may skip it."""
        monkeypatch.setattr(torch.nn.parallel, "DistributedDataParallel", _RecordingDDP)
        validated: list[int] = []
        written: list[list[str]] = []
        for rank in range(2):
            recorder = _RecordingValidationHook()
            checkpoints = tmp_path / f"rank{rank}"
            strategy = _make_distributed_strategy(
                rank=rank,
                hooks=[
                    recorder,
                    CheckpointHook(checkpoints, step_interval=2, async_save=False),
                ],
                validation_config=ValidationConfig(
                    validation_data=[
                        _build_propagator_batch(_REFERENCE_ELEMENT, 2, base_seed=900)
                    ],
                    every_n_steps=2,
                ),
            )
            strategy.run()
            validated.append(recorder.calls)
            written.append(sorted(path.name for path in checkpoints.glob("*")))

        assert validated[0] == validated[1] > 0
        assert written[0]
        assert not written[1]

    def test_what_rank_zero_wrote_restores_unwrapped_and_resumes(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """A wrapper owns the student at save time, so a restart must not need one."""
        monkeypatch.setattr(torch.nn.parallel, "DistributedDataParallel", _RecordingDDP)
        checkpoints = tmp_path / "rank0"
        strategy = _make_distributed_strategy(
            hooks=[CheckpointHook(checkpoints, step_interval=2, async_save=False)]
        )
        strategy.run()
        trained = _student_state(strategy)

        resumed = _make_distributed_strategy(num_steps=_WORKER_STEPS + 2)
        resumed.restore_checkpoint(checkpoints)
        restored = _student_state(resumed)
        resumed.run()

        assert set(restored) == set(trained)
        for key, value in trained.items():
            torch.testing.assert_close(restored[key], value)
        assert resumed.step_count == _WORKER_STEPS + 2

    def test_a_restored_rank_reseeds_from_its_own_shard(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """A restart restores one student, not the shard it was written from."""
        monkeypatch.setattr(torch.nn.parallel, "DistributedDataParallel", _RecordingDDP)
        checkpoints = tmp_path / "rank0"
        _make_distributed_strategy(
            hooks=[CheckpointHook(checkpoints, step_interval=2, async_save=False)]
        ).run()

        resumed = _make_distributed_strategy(rank=1, num_steps=_WORKER_STEPS + 2)
        resumed.restore_checkpoint(checkpoints)
        resumed.run()

        structures = resumed.on_policy.initial_structures
        assert resumed.structure_shard == structures.rows == (1, 3)
        assert structures.cursor == len(structures.rows)
        assert resumed.step_count == _WORKER_STEPS + 2


class TestRestartDevicePlacement:
    @pytest.mark.multigpu
    def test_a_rank_naming_its_device_in_both_places_resumes_wholly_on_it(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Naming this rank's device explicitly leaves no optimizer state on rank zero's GPU."""
        monkeypatch.setattr(torch.nn.parallel, "DistributedDataParallel", _RecordingDDP)
        checkpoints = tmp_path / "rank0"
        _make_restart_strategy(
            0,
            device="cuda",
            hooks=[CheckpointHook(checkpoints, step_interval=2, async_save=False)],
        ).run()

        resumed = _make_restart_strategy(
            1, device="cuda:1", num_steps=_WORKER_STEPS + 2
        )
        resumed.restore_checkpoint(checkpoints, map_location="cuda:1")
        resumed.run()

        assert _optimizer_state_devices(resumed) == {torch.device("cuda", 1)}
        assert resumed.step_count == _WORKER_STEPS + 2

    @pytest.mark.multigpu
    def test_a_rank_taking_the_recorded_device_still_lands_wholly_on_its_own(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Naming neither place, a restart still resumes on this rank's own GPU."""
        monkeypatch.setattr(torch.nn.parallel, "DistributedDataParallel", _RecordingDDP)
        checkpoints = tmp_path / "rank0"
        _make_restart_strategy(
            0,
            device="cuda",
            hooks=[CheckpointHook(checkpoints, step_interval=2, async_save=False)],
        ).run()

        resumed = _make_restart_strategy(1, device="cuda", num_steps=_WORKER_STEPS + 2)
        resumed.restore_checkpoint(checkpoints)
        resumed.run()

        assert _optimizer_state_devices(resumed) == {torch.device("cuda", 1)}
        assert resumed.step_count == _WORKER_STEPS + 2


@pytest.mark.skipif(not dist.is_gloo_available(), reason="gloo backend required")
def test_two_cpu_ranks_train_one_student_from_disjoint_trajectories() -> None:
    """Two real ranks self-label their own frames and end on one set of weights."""
    results = _run_ranks(2)

    assert set(results) == {0, 1}
    assert all(result["steps"] == _WORKER_STEPS for result in results.values())
    assert results[0]["validations"] == results[1]["validations"] > 0
    _assert_disjoint_frames(results)
    _assert_one_student(results)


@pytest.mark.skipif(not dist.is_gloo_available(), reason="gloo backend required")
def test_two_cpu_ranks_own_disjoint_structure_rows() -> None:
    """The rows a real rank reports owning are the rows its initial batch came from."""
    results = _run_ranks(2, probe="structures")

    assert set(results) == {0, 1}
    shards = [results[rank]["shard"] for rank in sorted(results)]
    assert [results[rank]["loaded"] for rank in sorted(results)] == shards
    assert not set(shards[0]) & set(shards[1])
    assert sorted(shards[0] + shards[1]) == list(range(len(_build_initial_dataset())))


@pytest.mark.skipif(not dist.is_gloo_available(), reason="gloo backend required")
def test_two_cpu_ranks_backfill_from_disjoint_structure_rows() -> None:
    """A backfill under a real world of two draws on the shard its rank owns.

    Two ranks refilling a graduated trajectory from one dataset would otherwise
    serve the same row twice, propagating a structure on both and billing the
    teacher for it on both.
    """
    results = _run_ranks(2, probe="backfill")

    assert set(results) == {0, 1}
    served = [results[rank]["served"] for rank in sorted(results)]
    assert [results[rank]["shard"] for rank in sorted(results)] == served
    assert not set(served[0]) & set(served[1])
    assert sorted(served[0] + served[1]) == list(range(len(_build_initial_dataset())))


@pytest.mark.skipif(not dist.is_gloo_available(), reason="gloo backend required")
def test_ranks_dealt_shards_of_different_sizes_stay_in_lockstep() -> None:
    """A structure set the world cannot halve still runs one reported, synchronized loop."""
    results = _run_ranks(2, structures=_UNEVEN_STRUCTURES)

    assert set(results) == {0, 1}
    assert sorted(len(result["shard"]) for result in results.values()) == [1, 2]
    assert all(result["steps"] == _WORKER_STEPS for result in results.values())
    assert all(
        any("do not divide evenly" in message for message in result["warnings"])
        for result in results.values()
    )
    _assert_disjoint_frames(results, total=_UNEVEN_STRUCTURES * _FRAMES_PER_TRAJECTORY)
    _assert_one_student(results)


@pytest.mark.skipif(not dist.is_gloo_available(), reason="gloo backend required")
def test_ranks_dealt_shards_of_different_sizes_refuse_each_others_bundle() -> None:
    """Each rank restarts under its own shard, and a peer's cursor is refused by name.

    The lockstep the sibling test asserts says the ranks agree on the step, not
    on the row: unequal shards write bundles counted in different row sets, and
    the refusal of a foreign cursor keeps one rank from resuming under the other's.
    """
    results = _run_ranks(2, probe="bundle", structures=_UNEVEN_STRUCTURES)

    assert sorted(len(result["shard"]) for result in results.values()) == [1, 2]
    for rank, result in results.items():
        assert result["seeded"] == len(result["shard"])
        assert (result["bundle"]["rank"], result["bundle"]["world_size"]) == (rank, 2)
        assert result["bundle"]["cursor"] == len(result["shard"])

    for rank in (0, 1):
        restored = InitialStructures(_build_initial_dataset(_UNEVEN_STRUCTURES))
        restored.shard(rank, 2)
        with pytest.raises(ValueError, match=f"written for rank {1 - rank!r} of 2"):
            restored.load_state_dict(results[1 - rank]["bundle"])
        restored.load_state_dict(results[rank]["bundle"])
        assert restored.exhausted


@pytest.mark.skipif(not dist.is_gloo_available(), reason="gloo backend required")
@pytest.mark.parametrize(("structures", "recycle"), _RELAXATION_WORLDS)
def test_two_cpu_ranks_relax_and_backfill_from_their_own_shards(
    structures: int, recycle: bool
) -> None:
    """A lifecycle-driven backfill under a real world serves this rank's shard alone."""
    results = _run_ranks(2, probe="relaxation", structures=structures, recycle=recycle)

    assert set(results) == {0, 1}
    shards = [results[rank]["shard"] for rank in (0, 1)]
    assert not set(shards[0]) & set(shards[1])
    assert sorted(shards[0] + shards[1]) == list(range(structures))
    assert {result["steps"] for result in results.values()} == {_WORKER_STEPS}
    for result in results.values():
        assert result["served"]
        assert set(result["served"]) <= set(result["shard"])
        exhausted = [
            "nothing left to start a fresh one" in message
            for message in result["warnings"]
        ]
        if recycle:
            assert result["wraps"] >= 1
            assert len(result["served"]) > len(result["shard"])
            assert not any(exhausted)
        else:
            assert sorted(result["served"]) == sorted(result["shard"])
            assert result["wraps"] == 0
            assert any(exhausted)


@pytest.mark.skipif(not dist.is_gloo_available(), reason="gloo backend required")
def test_fewer_structures_than_ranks_is_refused_identically_on_every_rank() -> None:
    """Every rank refuses the same way, before any collective could strand a peer."""
    results = _run_ranks(2, probe="relaxation", structures=1)

    assert set(results) == {0, 1}
    errors = {result["error"] for result in results.values()}
    assert len(errors) == 1
    assert "at least one for each" in errors.pop()


@pytest.mark.skipif(not dist.is_gloo_available(), reason="gloo backend required")
def test_a_composed_propagator_seeded_from_replicas_still_diverges_per_rank() -> None:
    """Sharding replicas separates nothing, so the sub-stage thermostats have to."""
    results = _run_ranks(2, composed=True)

    assert set(results) == {0, 1}
    _assert_disjoint_frames(results)
    _assert_one_student(results)


@pytest.mark.skipif(not dist.is_gloo_available(), reason="gloo backend required")
def test_two_ranks_take_the_step_one_process_takes_over_the_union() -> None:
    """Two ranks land on the student a single process trains on both shards.

    Weights agreeing across ranks says only that the all-reduce ran; it is
    agreeing with the union that says the reduction averaged, once, over the
    gradients of both shards.
    """
    world_size = 2
    port = _free_port()
    global_batch = world_size * _UNION_SHARD_BATCH

    results = _spawn_ranks(
        _run_union_worker, [(rank, world_size, port) for rank in range(world_size)]
    )
    reference = _make_union_strategy()
    reference.run(_make_union_loader(global_batch))

    assert set(results) == set(range(world_size))
    assert reference.step_count == _UNION_EPOCHS * _UNION_STRUCTURES // global_batch
    assert {result["steps"] for result in results.values()} == {reference.step_count}
    trained = _student_state(reference)
    for result in results.values():
        for key, value in trained.items():
            torch.testing.assert_close(torch.as_tensor(result["state"][key]), value)


@pytest.mark.slow
@pytest.mark.skipif(not dist.is_gloo_available(), reason="gloo backend required")
def test_ranks_placed_one_per_node_train_one_student() -> None:
    """The multi-node shape: node-local rank zero everywhere, one global world."""
    results = _run_ranks(4, local_ranks=(0, 0, 0, 0))

    assert set(results) == {0, 1, 2, 3}
    _assert_disjoint_frames(results)
    _assert_one_student(results)


@pytest.mark.multigpu
def test_two_gpu_ranks_train_one_student_from_disjoint_trajectories() -> None:
    """The single-node placement: a teacher replica and a student rank per GPU."""
    results = _run_ranks(2, backend="nccl")

    assert set(results) == {0, 1}
    assert {result["device"] for result in results.values()} == {"cuda:0", "cuda:1"}
    _assert_disjoint_frames(results)
    _assert_one_student(results)


@pytest.mark.multigpu
@pytest.mark.slow
def test_four_gpu_ranks_train_one_student_from_disjoint_trajectories() -> None:
    """Scaling the node out changes the world size and nothing else."""
    results = _run_ranks(4, backend="nccl")

    assert set(results) == {0, 1, 2, 3}
    _assert_disjoint_frames(results)
    _assert_one_student(results)
