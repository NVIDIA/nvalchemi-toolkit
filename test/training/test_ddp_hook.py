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
"""Tests for DDPHook and distributed manager integration."""

from __future__ import annotations

import os
import queue
import socket
from enum import Enum
from typing import Any, Sequence

import pytest
import torch
from torch import distributed as dist
from torch.utils.data import (
    BatchSampler,
    DataLoader,
    DistributedSampler,
    Sampler,
    SequentialSampler,
)

from nvalchemi.data.atomic_data import AtomicData
from nvalchemi.hooks._context import HookContext, TrainContext
from nvalchemi.training import TrainingStage
from nvalchemi.training.hooks import DDPHook
from nvalchemi.training.strategy import TrainingStrategy
from test.training.conftest import (
    _build_baseline_strategy_kwargs,
    _build_batch,
    _build_dataset,
)


class _FakeManager:
    """Structural distributed manager used by hook tests."""

    def __init__(self, *, world_size: int = 2, rank: int = 0) -> None:
        self.world_size = world_size
        self.rank = rank
        self.global_rank = rank
        self.local_rank = rank
        self.initialized = world_size > 1
        self.device = torch.device("cpu")
        self.broadcast_buffers = False
        self.find_unused_parameters = False

    def is_initialized(self) -> bool:
        return self.initialized


class _FakeDDP(torch.nn.Module):
    """Small DDP stand-in that records constructor kwargs."""

    calls: list[dict[str, Any]] = []

    def __init__(self, module: torch.nn.Module, **kwargs: Any) -> None:
        super().__init__()
        self.module = module
        self.kwargs = kwargs
        type(self).calls.append(kwargs)

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self.module(*args, **kwargs)


class _CustomDistributedSampler(Sampler[int]):
    """Sampler with non-DistributedSampler constructor argument names."""

    def __init__(
        self,
        data_source: Any,
        *,
        shards: int,
        position: int,
        token: object,
    ) -> None:
        self.data_source = data_source
        self.shards = shards
        self.position = position
        self.token = token

    def __iter__(self) -> Any:
        return iter(range(self.position, len(self.data_source), self.shards))

    def __len__(self) -> int:
        return len(range(self.position, len(self.data_source), self.shards))


class _TorchKeywordDistributedSampler(Sampler[int]):
    """Sampler that follows PyTorch DistributedSampler constructor keywords."""

    def __init__(
        self,
        data_source: Any,
        *,
        num_replicas: int,
        rank: int,
        shuffle: bool = False,
        seed: int = 0,
        drop_last: bool = False,
    ) -> None:
        self.data_source = data_source
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last
        self.epoch = 0

    def __iter__(self) -> Any:
        return iter(range(self.rank, len(self.data_source), self.num_replicas))

    def __len__(self) -> int:
        return len(range(self.rank, len(self.data_source), self.num_replicas))

    def set_epoch(self, epoch: int) -> None:
        """Record the sampler epoch."""
        self.epoch = epoch


class _MutableSamplerDataloader:
    """Minimal dataloader-like object with a mutable sampler attribute."""

    def __init__(
        self,
        dataset: Any,
        *,
        sampler: Any | None = None,
        drop_last: bool = False,
    ) -> None:
        self.dataset = dataset
        self.sampler = sampler
        self.drop_last = drop_last


class _ContextCaptureHook:
    """Capture contexts observed at a given stage."""

    frequency = 1

    def __init__(self, stage: TrainingStage) -> None:
        self.stage = stage
        self.contexts: list[TrainContext] = []

    def __call__(self, ctx: HookContext, stage: Enum) -> None:
        assert isinstance(ctx, TrainContext)
        self.contexts.append(ctx)


class _OptimizerParamHook:
    """Assert optimizers are constructed after DDP wrapping."""

    frequency = 1
    stage = TrainingStage.BEFORE_TRAINING

    def __init__(self) -> None:
        self.saw_wrapped_model = False

    def __call__(self, ctx: HookContext, stage: Enum) -> None:
        assert isinstance(ctx, TrainContext)
        assert ctx.models is not None
        model = ctx.models["main"]
        self.saw_wrapped_model = isinstance(model, _FakeDDP)
        model_param_ids = {id(param) for param in model.parameters()}
        optimizer_param_ids = {
            id(param)
            for optimizer in ctx.optimizers
            for group in optimizer.param_groups
            for param in group["params"]
        }
        assert optimizer_param_ids <= model_param_ids


class _Reader:
    """Minimal datapipe reader for sampler mutation tests."""

    def __len__(self) -> int:
        return 4

    def _load_sample(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "positions": torch.zeros(1, 3),
            "atomic_numbers": torch.ones(1, dtype=torch.long),
            "atomic_masses": torch.ones(1),
        }

    @property
    def field_names(self) -> list[str]:
        """Return fields exposed by this reader."""
        return list(self._load_sample(0))

    def _get_sample_metadata(self, index: int) -> dict[str, Any]:
        return {}

    def read_many(
        self, indices: Sequence[int]
    ) -> list[tuple[dict[str, torch.Tensor], dict[str, Any]]]:
        """Load multiple samples and metadata records."""
        return [
            (self._load_sample(index), self._get_sample_metadata(index))
            for index in indices
        ]

    def close(self) -> None:
        pass


def _make_strategy(**overrides: Any) -> TrainingStrategy:
    """Build a baseline TrainingStrategy with local overrides."""
    kwargs = _build_baseline_strategy_kwargs()
    if "num_steps" in overrides and "num_epochs" not in overrides:
        kwargs["num_epochs"] = None
    kwargs.update(overrides)
    return TrainingStrategy(**kwargs)


def _free_port() -> int:
    """Return an available localhost TCP port for process-group setup."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _state_dict_cpu(strategy: TrainingStrategy) -> dict[str, torch.Tensor]:
    """Return the main model state dict detached on CPU."""
    return {
        key: value.detach().cpu().clone()
        for key, value in strategy.models["main"].state_dict().items()
    }


def _run_ddp_worker(
    rank: int,
    world_size: int,
    port: int,
    result_queue: Any,
) -> None:
    """Run one CPU DDP training step and send final parameters to the parent."""
    os.environ.update(
        {
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": str(port),
            "RANK": str(rank),
            "WORLD_SIZE": str(world_size),
            "LOCAL_RANK": str(rank),
        }
    )
    strategy = _make_strategy(
        hooks=[DDPHook(backend="gloo", find_unused_parameters=True)],
        num_steps=1,
    )
    strategy.run([_build_batch(n_systems=1, seed=5)])
    result_queue.put(
        (
            rank,
            {key: value.tolist() for key, value in _state_dict_cpu(strategy).items()},
        )
    )


@pytest.fixture(autouse=True)
def _reset_fake_ddp() -> None:
    """Reset fake DDP call history before every test."""
    _FakeDDP.calls.clear()


class TestDistributedManagerField:
    def test_nvalchemi_distributed_reexports_physicsnemo_manager(self) -> None:
        from physicsnemo.distributed import DistributedManager as PhysicsNeMoManager

        from nvalchemi.distributed import DistributedManager

        assert DistributedManager is PhysicsNeMoManager

    def test_resolves_rank_and_world_size_from_environment(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from nvalchemi import distributed

        class _UninitializedManager:
            @classmethod
            def is_initialized(cls) -> bool:
                return False

        monkeypatch.setattr(distributed, "DistributedManager", _UninitializedManager)
        monkeypatch.setenv("RANK", "3")
        monkeypatch.setenv("WORLD_SIZE", "8")

        assert distributed.resolve_global_rank() == 3
        assert distributed.resolve_world_size() == 8

    def test_explicit_rank_overrides_runtime_state(self) -> None:
        from nvalchemi.distributed import resolve_global_rank

        assert resolve_global_rank(5) == 5

    def test_manager_is_runtime_only_and_visible_to_context(self) -> None:
        manager = _FakeManager(world_size=1)
        capture = _ContextCaptureHook(TrainingStage.BEFORE_BATCH)
        strategy = _make_strategy(
            distributed_manager=manager,
            hooks=[capture],
            num_steps=1,
        )

        assert "distributed_manager" not in strategy.to_spec_dict()
        strategy.run([_build_batch()])

        assert capture.contexts
        assert capture.contexts[0].workflow.distributed_manager is manager
        assert capture.contexts[0].global_rank == manager.rank


class TestDDPHookWrapping:
    def test_wraps_before_optimizer_construction_and_restores(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(torch.nn.parallel, "DistributedDataParallel", _FakeDDP)
        ddp = DDPHook(find_unused_parameters=True, broadcast_buffers=False)
        recorder = _OptimizerParamHook()
        strategy = _make_strategy(
            distributed_manager=_FakeManager(),
            hooks=[ddp, recorder],
            num_steps=1,
        )
        original = strategy.models["main"]

        strategy.run([_build_batch()])

        assert recorder.saw_wrapped_model
        assert strategy.models["main"] is original
        assert _FakeDDP.calls == [
            {
                "find_unused_parameters": True,
                "broadcast_buffers": False,
                "static_graph": False,
            }
        ]

    def test_defaults_to_manager_ddp_flags(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(torch.nn.parallel, "DistributedDataParallel", _FakeDDP)
        manager = _FakeManager()
        manager.find_unused_parameters = True
        manager.broadcast_buffers = True
        strategy = _make_strategy(
            distributed_manager=manager,
            hooks=[DDPHook()],
            num_steps=1,
        )

        strategy.run([_build_batch()])

        assert _FakeDDP.calls == [
            {
                "find_unused_parameters": True,
                "broadcast_buffers": True,
                "static_graph": False,
            }
        ]

    def test_unknown_model_key_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(torch.nn.parallel, "DistributedDataParallel", _FakeDDP)
        strategy = _make_strategy(
            distributed_manager=_FakeManager(),
            hooks=[DDPHook(model_keys=("missing",))],
            num_steps=1,
        )

        with pytest.raises(KeyError, match="unknown model"):
            strategy.run([_build_batch()])


class TestDDPHookDataloaderMutation:
    def test_sets_sampler_on_generic_dataloader_with_sampler_attribute(self) -> None:
        hook = DDPHook()
        hook._manager = _FakeManager(rank=1)
        loader = _MutableSamplerDataloader(list(range(8)), drop_last=True)

        prepared = hook.prepare_dataloader(loader)

        assert prepared is loader
        assert isinstance(loader.sampler, DistributedSampler)
        assert loader.sampler.rank == 1
        assert loader.sampler.num_replicas == 2
        assert loader.sampler.drop_last is True

    def test_strategy_setup_uses_workflow_dataloader(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(torch.nn.parallel, "DistributedDataParallel", _FakeDDP)
        manager = _FakeManager(rank=1)
        loader = DataLoader(
            _build_dataset(n_batches=4),
            batch_size=1,
            shuffle=True,
            collate_fn=lambda x: x[0],
        )
        strategy = _make_strategy(
            distributed_manager=manager,
            hooks=[DDPHook()],
            num_steps=1,
        )

        strategy.run(loader)

        assert strategy.active_dataloader is not loader
        assert isinstance(strategy.active_dataloader.sampler, DistributedSampler)
        assert strategy.active_dataloader.sampler.rank == manager.rank

    def test_replaces_torch_dataloader_sampler(self) -> None:
        hook = DDPHook()
        hook._manager = _FakeManager(rank=1)
        dataset = list(range(8))
        loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)

        prepared = hook.prepare_dataloader(loader)

        assert prepared is not loader
        assert isinstance(prepared, DataLoader)
        assert isinstance(prepared.sampler, DistributedSampler)
        assert prepared.sampler.rank == 1
        assert prepared.sampler.num_replicas == 2
        assert prepared.sampler.shuffle is False

    def test_sampler_kwargs_override_default_sampler_args(self) -> None:
        hook = DDPHook(
            sampler_kwargs={
                "shuffle": True,
                "seed": 17,
                "drop_last": True,
            }
        )
        hook._manager = _FakeManager(rank=1)
        dataset = list(range(8))
        loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)

        prepared = hook.prepare_dataloader(loader)

        assert isinstance(prepared, DataLoader)
        assert isinstance(prepared.sampler, DistributedSampler)
        assert prepared.sampler.shuffle is True
        assert prepared.sampler.seed == 17
        assert prepared.sampler.drop_last is True

    def test_uses_custom_sampler_cls_and_kwargs(self) -> None:
        token = object()
        hook = DDPHook(
            sampler_cls=_CustomDistributedSampler,
            sampler_kwargs={
                "shards": 4,
                "position": 2,
                "token": token,
            },
        )
        hook._manager = _FakeManager(rank=1)
        dataset = list(range(8))
        loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)

        prepared = hook.prepare_dataloader(loader)

        assert isinstance(prepared, DataLoader)
        assert isinstance(prepared.sampler, _CustomDistributedSampler)
        assert prepared.sampler.shards == 4
        assert prepared.sampler.position == 2
        assert prepared.sampler.token is token

    def test_keeps_existing_custom_sampler(self) -> None:
        token = object()
        hook = DDPHook(
            sampler_cls=_CustomDistributedSampler,
            sampler_kwargs={
                "shards": 2,
                "position": 1,
                "token": token,
            },
        )
        hook._manager = _FakeManager()
        dataset = list(range(8))
        sampler = _CustomDistributedSampler(
            dataset,
            shards=2,
            position=1,
            token=token,
        )
        loader = DataLoader(dataset, batch_size=2, sampler=sampler)

        prepared = hook.prepare_dataloader(loader)

        assert prepared is loader
        assert prepared.sampler is sampler

    def test_keeps_existing_protocol_distributed_sampler(self) -> None:
        hook = DDPHook()
        hook._manager = _FakeManager(rank=1)
        dataset = list(range(8))
        sampler = _TorchKeywordDistributedSampler(
            dataset,
            num_replicas=2,
            rank=1,
        )
        loader = DataLoader(dataset, batch_size=2, sampler=sampler)

        prepared = hook.prepare_dataloader(loader)

        assert prepared is loader
        assert prepared.sampler is sampler

    def test_injects_defaults_into_protocol_compatible_sampler_cls(self) -> None:
        hook = DDPHook(
            sampler_cls=_TorchKeywordDistributedSampler,
            sampler_kwargs={"seed": 23},
        )
        hook._manager = _FakeManager(rank=1)
        dataset = list(range(8))
        loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)

        prepared = hook.prepare_dataloader(loader)

        assert isinstance(prepared.sampler, _TorchKeywordDistributedSampler)
        assert prepared.sampler.num_replicas == 2
        assert prepared.sampler.rank == 1
        assert prepared.sampler.shuffle is False
        assert prepared.sampler.seed == 23
        assert prepared.sampler.drop_last is False

    def test_keeps_existing_distributed_sampler(self) -> None:
        hook = DDPHook()
        hook._manager = _FakeManager()
        dataset = list(range(8))
        sampler = DistributedSampler(dataset, num_replicas=2, rank=0)
        loader = DataLoader(dataset, batch_size=2, sampler=sampler)

        prepared = hook.prepare_dataloader(loader)

        assert prepared is loader
        assert prepared.sampler is sampler

    def test_rejects_custom_batch_sampler(self) -> None:
        hook = DDPHook()
        hook._manager = _FakeManager()
        dataset = list(range(8))
        batch_sampler = BatchSampler(
            SequentialSampler(dataset),
            batch_size=2,
            drop_last=False,
        )
        loader = DataLoader(dataset, batch_sampler=batch_sampler)

        with pytest.raises(ValueError, match="batch_sampler"):
            hook.prepare_dataloader(loader)

    def test_mutates_nvalchemi_datapipe_batch_sampler(self) -> None:
        from nvalchemi.data.datapipes.dataloader import DataLoader as NVCDataLoader
        from nvalchemi.data.datapipes.dataset import Dataset

        hook = DDPHook()
        hook._manager = _FakeManager(rank=1)
        dataset = Dataset(_Reader(), device="cpu")
        loader = NVCDataLoader(dataset, batch_size=2, use_streams=False)

        prepared = hook.prepare_dataloader(loader)

        assert prepared is loader
        assert loader.sampler is None
        assert isinstance(loader.batch_sampler, BatchSampler)
        assert isinstance(loader.batch_sampler.sampler, DistributedSampler)
        assert loader.batch_sampler.sampler.rank == 1
        assert loader.batch_sampler.sampler.num_replicas == 2

    def test_nvalchemi_datapipe_set_epoch_updates_nested_sampler(self) -> None:
        from nvalchemi.data.datapipes.dataloader import DataLoader as NVCDataLoader
        from nvalchemi.data.datapipes.dataset import Dataset

        hook = DDPHook()
        hook._manager = _FakeManager(rank=1)
        dataset = Dataset(_Reader(), device="cpu")
        loader = NVCDataLoader(dataset, batch_size=2, use_streams=False)

        hook.prepare_dataloader(loader)
        loader.set_epoch(7)

        assert isinstance(loader.batch_sampler, BatchSampler)
        assert loader.batch_sampler.sampler.epoch == 7

    def test_mutates_nvalchemi_multidataset_batch_sampler(self) -> None:
        from nvalchemi.data.datapipes.dataloader import DataLoader as NVCDataLoader
        from nvalchemi.data.datapipes.dataset import Dataset
        from nvalchemi.data.datapipes.multidataset import MultiDataset
        from nvalchemi.data.datapipes.samplers import MultiDatasetBatchSampler

        hook = DDPHook()
        hook._manager = _FakeManager(rank=1)
        dataset = MultiDataset(
            Dataset(_Reader(), device="cpu"),
            Dataset(_Reader(), device="cpu"),
        )
        loader = NVCDataLoader(dataset, batch_size=2, use_streams=False)

        prepared = hook.prepare_dataloader(loader)

        assert prepared is loader
        assert loader.sampler is None
        assert isinstance(loader.batch_sampler, MultiDatasetBatchSampler)
        assert loader.batch_sampler.rank == 1
        assert loader.batch_sampler.num_replicas == 2


def test_single_process_ddp_hook_is_noop(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.nn.parallel, "DistributedDataParallel", _FakeDDP)
    strategy = _make_strategy(
        distributed_manager=_FakeManager(world_size=1),
        hooks=[DDPHook()],
        num_steps=1,
    )

    strategy.run([_build_batch()])

    assert _FakeDDP.calls == []


def test_torch_distributed_sampler_epoch_is_preserved() -> None:
    hook = DDPHook()
    hook._manager = _FakeManager()
    dataset = _build_dataset(n_batches=4)
    loader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=lambda x: x[0])
    strategy = _make_strategy(num_steps=1)

    prepared = hook.prepare_dataloader(loader)
    assert isinstance(prepared.sampler, DistributedSampler)
    strategy._set_sampler_epoch(prepared)

    assert prepared.sampler.epoch == 0


def test_batch_sampler_epoch_is_preserved() -> None:
    hook = DDPHook()
    hook._manager = _FakeManager()
    dataset = list(range(8))
    sampler = DistributedSampler(dataset, num_replicas=2, rank=0)
    loader = DataLoader(
        dataset,
        batch_sampler=BatchSampler(sampler, batch_size=2, drop_last=False),
    )
    strategy = _make_strategy(num_steps=1)

    strategy._set_sampler_epoch(loader)

    assert sampler.epoch == 0


def test_reader_protocol_builds_atomic_data() -> None:
    reader = _Reader()
    sample = AtomicData(**reader._load_sample(0))
    assert sample.positions.shape == (1, 3)


@pytest.mark.skipif(not dist.is_gloo_available(), reason="gloo backend required")
def test_two_process_cpu_ddp_matches_single_process_baseline() -> None:
    baseline = _make_strategy(num_steps=1)
    baseline.run([_build_batch(n_systems=1, seed=5)])
    expected = _state_dict_cpu(baseline)

    ctx = torch.multiprocessing.get_context("spawn")
    result_queue = ctx.Queue()
    port = _free_port()
    procs = [
        ctx.Process(
            target=_run_ddp_worker,
            args=(rank, 2, port, result_queue),
        )
        for rank in range(2)
    ]
    for proc in procs:
        proc.start()
    for proc in procs:
        proc.join(timeout=30)
    for proc in procs:
        assert proc.exitcode == 0

    results: dict[int, dict[str, Any]] = {}
    for _ in range(2):
        rank, state = result_queue.get(timeout=5)
        results[rank] = state

    assert set(results) == {0, 1}
    for state in results.values():
        for key, expected_value in expected.items():
            actual = torch.as_tensor(state[key], dtype=expected_value.dtype)
            assert torch.allclose(actual, expected_value, atol=1e-6, rtol=1e-6)

    try:
        result_queue.close()
    except (AttributeError, OSError, queue.Empty):
        pass
