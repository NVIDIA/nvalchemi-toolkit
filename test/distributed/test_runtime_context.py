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

from __future__ import annotations

import os
import sys
from enum import Enum
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _dd_harness import free_port  # noqa: E402
from _gloo_harness import run_gloo  # noqa: E402

pytestmark = pytest.mark.skipif(
    not dist.is_gloo_available(), reason="gloo backend required"
)


def _materialize(value: str) -> str:
    return value


class _CountedMetadata:
    calls = 0

    def __init__(self, value: str) -> None:
        self.value = value

    def __reduce__(self):
        type(self).calls += 1
        return (_materialize, (self.value,))


class _BrokenStringError(Exception):
    def __str__(self) -> str:
        raise ValueError("broken string conversion")


class _NoSerialize:
    def __reduce__(self):
        raise ValueError("cannot serialize metadata")


_DESERIALIZED = 0


def _mark_deserialized() -> str:
    global _DESERIALIZED
    _DESERIALIZED += 1
    return "loaded"


class _DeserializationMarker:
    def __reduce__(self):
        return (_mark_deserialized, ())


class _PickleFailString(str):
    reducer_calls = 0

    def __reduce_ex__(self, protocol: int) -> Any:
        type(self).reducer_calls += 1
        raise ValueError("string subclasses must be normalized before pickling")


class _MessageSubclassError(Exception):
    def __str__(self) -> str:
        return _PickleFailString("message subclass contents")


def _context_queries_worker(rank: int, world_size: int, queue: Any) -> None:
    from nvalchemi.distributed._runtime import ProcessGroupContext

    with (
        patch.object(dist, "init_process_group", side_effect=AssertionError),
        patch.object(dist, "new_group", side_effect=AssertionError),
        patch.object(dist, "all_gather_object", side_effect=AssertionError),
        patch.object(dist, "all_reduce", side_effect=AssertionError),
        patch.object(torch.cuda, "set_device", side_effect=AssertionError),
        patch.object(torch.cuda, "device", side_effect=AssertionError),
    ):
        context = ProcessGroupContext(dist.group.WORLD, execution_device="cuda:999")
        with context.communication_scope():
            pass
        translated = tuple(context.global_rank(i) for i in range(world_size))
        assert context.process_group is dist.group.WORLD
        assert context.rank == rank
        assert context.world_size == world_size
        assert context.backend == "gloo"
        assert context.collective_device == torch.device("cpu")
        with pytest.raises(AttributeError):
            context.rank = 10
        with pytest.raises(TypeError):
            context.global_rank(True)
        with pytest.raises(TypeError):
            context.global_rank(0.0)
        with pytest.raises(ValueError):
            context.global_rank(world_size)
        with pytest.raises(TypeError):
            ProcessGroupContext(None)  # type: ignore[arg-type]
        queue.put((rank, translated))


def _subgroup_worker(rank: int, world_size: int, queue: Any) -> None:
    from nvalchemi.distributed._runtime import ProcessGroupContext

    del world_size
    group = dist.new_group(ranks=[1, 3], backend="gloo")
    if rank in (1, 3):
        context = ProcessGroupContext(group)
        assert context.rank == (rank - 1) // 2
        assert context.world_size == 2
        assert context.global_rank(0) == 1
        assert context.global_rank(1) == 3
        with pytest.raises(ValueError):
            context.global_rank(-1)
        queue.put((rank, context.rank, context.global_rank(context.rank)))
    else:
        with pytest.raises(TypeError, match="actual torch ProcessGroup"):
            ProcessGroupContext(group)
        queue.put((rank, "non-member"))


def _metadata_worker(rank: int, world_size: int, queue: Any) -> None:
    from nvalchemi.distributed._runtime import (
        ProcessGroupContext,
        collective_error_sync,
    )

    context = ProcessGroupContext(dist.group.WORLD)
    counted = _CountedMetadata("rank-zero") if rank == 0 else None
    _CountedMetadata.calls = 0
    metadata = {"rank": rank, "token": counted} if rank == 0 else None
    gather_calls = 0
    original_gather = dist.all_gather_object

    def count_gather(*args: Any, **kwargs: Any) -> None:
        nonlocal gather_calls
        gather_calls += 1
        original_gather(*args, **kwargs)

    with patch.object(dist, "all_gather_object", side_effect=count_gather):
        with collective_error_sync(context, phase="all_none") as none_phase:
            pass
        assert none_phase.records == (None,) * world_size
        assert gather_calls == 1

        with collective_error_sync(context, phase="pack") as phase:
            with pytest.raises(RuntimeError, match="available only after success"):
                _ = phase.records
            phase.phase = "snapshot"
            with pytest.raises(ValueError, match="non-empty"):
                phase.phase = ""
            assert phase.phase == "snapshot"
            phase.metadata = metadata
        assert gather_calls == 2
    if metadata is not None:
        metadata["rank"] = 99
    records = phase.records
    assert len(records) == world_size
    with pytest.raises(AttributeError):
        phase.records = (None,) * world_size
    if rank == 0:
        assert _CountedMetadata.calls == 1
    else:
        assert _CountedMetadata.calls == 0
    queue.put((rank, phase.phase, records, _CountedMetadata.calls))


def _body_error_worker(rank: int, world_size: int, queue: Any) -> None:
    from nvalchemi.distributed._runtime import (
        ProcessGroupContext,
        collective_error_sync,
    )

    del world_size
    context = ProcessGroupContext(dist.group.WORLD)
    phase_state = None
    try:
        with collective_error_sync(
            context, phase="setup", error_type=LookupError
        ) as phase:
            phase_state = phase
            phase.phase = "prepare" if rank == 0 else "finish"
            if rank == 0:
                raise _BrokenStringError("message intentionally unavailable")
            raise ValueError("rank-one local failure")
    except LookupError as exc:
        diagnostic = str(exc)
        cause = exc.__cause__
        with pytest.raises(RuntimeError, match="available only after success"):
            _ = phase_state.records
        queue.put(
            (
                rank,
                diagnostic,
                type(cause).__name__ if cause is not None else None,
                exc.__suppress_context__,
            )
        )


def _serialization_error_worker(rank: int, world_size: int, queue: Any) -> None:
    from nvalchemi.distributed._runtime import (
        ProcessGroupContext,
        collective_error_sync,
    )

    del world_size
    global _DESERIALIZED
    _DESERIALIZED = 0
    context = ProcessGroupContext(dist.group.WORLD)
    phase_state = None
    try:
        with collective_error_sync(context, phase="metadata") as phase:
            phase_state = phase
            phase.metadata = _DeserializationMarker() if rank == 0 else _NoSerialize()
    except RuntimeError as exc:
        with pytest.raises(RuntimeError, match="available only after success"):
            _ = phase_state.records
        queue.put(
            (
                rank,
                str(exc),
                type(exc.__cause__).__name__ if exc.__cause__ is not None else None,
                _DESERIALIZED,
            )
        )


def _phase_label_worker(rank: int, world_size: int, queue: Any) -> None:
    from nvalchemi.distributed._runtime import (
        ProcessGroupContext,
        collective_error_sync,
    )

    class LocalPhase(str, Enum):
        START = "function_local_start"
        FINISH = "function_local_finish"

    context = ProcessGroupContext(dist.group.WORLD)
    _PickleFailString.reducer_calls = 0
    gather_calls = 0
    original_gather = dist.all_gather_object

    def count_gather(*args: Any, **kwargs: Any) -> None:
        nonlocal gather_calls
        gather_calls += 1
        original_gather(*args, **kwargs)

    results: list[tuple[str, tuple[Any, ...]]] = []
    with patch.object(dist, "all_gather_object", side_effect=count_gather):
        with collective_error_sync(context, phase=LocalPhase.START) as first:
            assert type(first.phase) is str
            first.metadata = {"round": 0, "rank": rank}
        results.append((first.phase, first.records))

        with collective_error_sync(context, phase="setter_start") as second:
            second.phase = LocalPhase.FINISH
            assert type(second.phase) is str
            second.metadata = {"round": 1, "rank": rank}
        results.append((second.phase, second.records))

        with collective_error_sync(
            context, phase=_PickleFailString("guarded_start")
        ) as third:
            assert type(third.phase) is str
            assert third.phase == "guarded_start"
            third.phase = _PickleFailString("guarded_finish")
            assert type(third.phase) is str
            third.metadata = {"round": 2, "rank": rank}
        results.append((third.phase, third.records))

    assert gather_calls == 3
    assert _PickleFailString.reducer_calls == 0
    queue.put((rank, tuple(results), gather_calls, _PickleFailString.reducer_calls))


def _subclass_exception_message_worker(rank: int, world_size: int, queue: Any) -> None:
    from nvalchemi.distributed._runtime import (
        ProcessGroupContext,
        collective_error_sync,
    )

    del world_size
    context = ProcessGroupContext(dist.group.WORLD)
    _PickleFailString.reducer_calls = 0
    gather_calls = 0
    original_gather = dist.all_gather_object

    def count_gather(*args: Any, **kwargs: Any) -> None:
        nonlocal gather_calls
        gather_calls += 1
        original_gather(*args, **kwargs)

    try:
        with patch.object(dist, "all_gather_object", side_effect=count_gather):
            with collective_error_sync(
                context, phase="message_normalization", error_type=RuntimeError
            ):
                if rank == 0:
                    raise _MessageSubclassError("ignored constructor message")
    except RuntimeError as exc:
        assert (
            str(exc) == "distributed message_normalization failed at rank=0 "
            "phase=message_normalization: "
            "_MessageSubclassError: message subclass contents"
        )
        assert (exc.__cause__ is not None) == (rank == 0)
        assert exc.__suppress_context__
        assert gather_calls == 1
        assert _PickleFailString.reducer_calls == 0
        queue.put(
            (
                rank,
                str(exc),
                type(exc.__cause__).__name__ if exc.__cause__ is not None else None,
                gather_calls,
                _PickleFailString.reducer_calls,
            )
        )
    else:
        raise AssertionError("expected shared RuntimeError")


def _validation_worker(rank: int, world_size: int, queue: Any) -> None:
    from nvalchemi.distributed._runtime import (
        ProcessGroupContext,
        collective_error_sync,
    )

    del world_size
    context = ProcessGroupContext(dist.group.WORLD)
    with patch.object(dist, "all_gather_object", side_effect=AssertionError):
        with patch.object(dist, "is_initialized", return_value=False):
            with pytest.raises(RuntimeError, match="must be initialized"):
                ProcessGroupContext(dist.group.WORLD)
        with patch.object(dist, "get_backend", return_value="mpi"):
            with pytest.raises(NotImplementedError, match="not supported"):
                ProcessGroupContext(dist.group.WORLD)
        with pytest.raises(TypeError, match="ProcessGroupContext"):
            with collective_error_sync(object(), phase="valid"):
                pass
        with pytest.raises(ValueError, match="non-empty"):
            with collective_error_sync(context, phase=""):
                pass
        with pytest.raises(TypeError, match="Exception subclass"):
            with collective_error_sync(
                context, phase="valid", error_type=BaseException
            ):
                pass

        class NeedsTwoArguments(Exception):
            def __init__(self, first: str, second: str) -> None:
                super().__init__(first, second)

        with pytest.raises(TypeError, match="accept a message"):
            with collective_error_sync(
                context, phase="valid", error_type=NeedsTwoArguments
            ):
                pass
    with pytest.raises(TypeError, match="created by collective_error_sync"):
        from nvalchemi.distributed._runtime import CollectivePhase

        CollectivePhase()
    queue.put((rank, "validated"))


def _cuda_selection_worker(rank: int, world_size: int, queue: Any) -> None:
    from nvalchemi.distributed import _runtime
    from nvalchemi.distributed._runtime import ProcessGroupContext

    del world_size

    class FakeManager:
        initialized = True
        device = torch.device("cuda:2")

        @classmethod
        def is_initialized(cls) -> bool:
            return cls.initialized

    backend = patch.object(_runtime.dist, "get_backend", return_value="nccl")
    available = patch.object(torch.cuda, "is_available", return_value=True)
    device_count = patch.object(torch.cuda, "device_count", return_value=3)
    current_device = patch.object(torch.cuda, "current_device", return_value=1)
    manager = patch.object(_runtime, "DistributedManager", FakeManager)
    with backend, available, device_count, current_device, manager:
        assert ProcessGroupContext(
            dist.group.WORLD, execution_device="cuda:0"
        ).collective_device == torch.device("cuda:0")
        assert ProcessGroupContext(
            dist.group.WORLD, execution_device="cuda"
        ).collective_device == torch.device("cuda:1")
        assert ProcessGroupContext(
            dist.group.WORLD, execution_device="cpu"
        ).collective_device == torch.device("cuda:2")
        assert ProcessGroupContext(dist.group.WORLD).collective_device == torch.device(
            "cuda:2"
        )

        FakeManager.device = torch.device("cpu")
        assert ProcessGroupContext(dist.group.WORLD).collective_device == torch.device(
            "cuda:1"
        )
        FakeManager.initialized = False
        assert ProcessGroupContext(dist.group.WORLD).collective_device == torch.device(
            "cuda:1"
        )
        FakeManager.initialized = True
        FakeManager.device = torch.device("cuda:2")

        with pytest.raises(ValueError, match="outside"):
            ProcessGroupContext(dist.group.WORLD, execution_device="cuda:3")
        with patch.object(torch.cuda, "is_available", return_value=False):
            with pytest.raises(RuntimeError, match="explicit CUDA"):
                ProcessGroupContext(dist.group.WORLD, execution_device="cuda:0")

        device_scope = MagicMock()
        with patch.object(torch.cuda, "device", return_value=device_scope):
            context = ProcessGroupContext(dist.group.WORLD, execution_device="cuda:0")
            with pytest.raises(ValueError, match="scope body"):
                with context.communication_scope():
                    raise ValueError("scope body")
        device_scope.__enter__.assert_called_once_with()
        assert device_scope.__exit__.call_args.args[0] is ValueError
        assert device_scope.__exit__.call_args.args[1].args == ("scope body",)
    queue.put((rank, "cuda-priority-checked"))


def _manager_nccl_worker(
    rank: int, world_size: int, manager_port: str, external_port: str
) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = manager_port
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(rank)

    from physicsnemo.distributed import DistributedManager

    from nvalchemi.distributed._runtime import (
        ProcessGroupContext,
        collective_error_sync,
    )

    DistributedManager.initialize()
    try:
        manager = DistributedManager()
        torch.cuda.set_device(rank)
        DistributedManager.create_process_subgroup("runtime-context", size=world_size)
        group = manager.group("runtime-context")
        assert manager.group() is None
        torch.cuda.set_device(1 - rank)
        explicit_context = ProcessGroupContext(group, execution_device=f"cuda:{rank}")
        assert explicit_context.collective_device == torch.device("cuda", rank)
        with explicit_context.communication_scope():
            assert torch.cuda.current_device() == rank
        assert torch.cuda.current_device() == 1 - rank
        with collective_error_sync(explicit_context, phase="explicit_records") as phase:
            phase.metadata = {"rank": rank}
        assert phase.records == tuple({"rank": i} for i in range(world_size))
        assert torch.cuda.current_device() == 1 - rank

        # A CPU hint follows the initialized Manager's CUDA device for NCCL.
        context = ProcessGroupContext(group, execution_device="cpu")
        assert context.rank == rank
        assert context.world_size == world_size
        assert context.global_rank(rank) == rank
        assert context.collective_device == torch.device("cuda", rank)
        with context.communication_scope():
            assert torch.cuda.current_device() == rank
        assert torch.cuda.current_device() == 1 - rank

        with collective_error_sync(context, phase="records") as phase:
            phase.metadata = {"rank": rank}
        assert phase.records == tuple({"rank": i} for i in range(world_size))
        assert torch.cuda.current_device() == 1 - rank

        try:
            with collective_error_sync(context, phase="nccl_failure"):
                if rank == 1:
                    raise OSError("manager subgroup failure")
        except RuntimeError as exc:
            assert (
                str(exc) == "distributed nccl_failure failed at rank=1 "
                "phase=nccl_failure: OSError: manager subgroup failure"
            )
            assert (exc.__cause__ is not None) == (rank == 1)
        else:
            raise AssertionError("expected distributed error")
        assert torch.cuda.current_device() == 1 - rank
    finally:
        DistributedManager.cleanup()

    # With no Manager, NCCL falls back to this external group's current device.
    os.environ["MASTER_PORT"] = external_port
    torch.cuda.set_device(1 - rank)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    try:
        context = ProcessGroupContext(dist.group.WORLD)
        assert not DistributedManager.is_initialized()
        assert context.collective_device == torch.device("cuda", 1 - rank)
        with context.communication_scope():
            assert torch.cuda.current_device() == 1 - rank
        with collective_error_sync(context, phase="external_records") as phase:
            phase.metadata = {"rank": rank}
        assert phase.records == tuple({"rank": i} for i in range(world_size))
        assert torch.cuda.current_device() == 1 - rank
    finally:
        dist.destroy_process_group()


def test_external_gloo_context_is_read_only_and_group_local() -> None:
    assert sorted(run_gloo(world_size=2, fn=_context_queries_worker)) == [
        (0, (0, 1)),
        (1, (0, 1)),
    ]


def test_gloo_subgroup_translates_nontrivial_global_ranks() -> None:
    results = run_gloo(world_size=4, fn=_subgroup_worker)
    assert sorted(results) == [
        (0, "non-member"),
        (1, 0, 1),
        (2, "non-member"),
        (3, 1, 3),
    ]


def test_collective_metadata_is_rank_ordered_and_snapshotted_once() -> None:
    results = sorted(run_gloo(world_size=2, fn=_metadata_worker))
    assert results[0] == (
        0,
        "snapshot",
        ({"rank": 0, "token": "rank-zero"}, None),
        1,
    )
    assert results[1] == (1, "snapshot", ({"rank": 0, "token": "rank-zero"}, None), 0)


def test_collective_error_uses_lowest_rank_and_only_origin_cause() -> None:
    results = sorted(run_gloo(world_size=2, fn=_body_error_worker))
    diagnostic = (
        "distributed prepare failed at rank=0 phase=prepare: "
        "_BrokenStringError: exception message unavailable"
    )
    assert results == [
        (0, diagnostic, "_BrokenStringError", True),
        (1, diagnostic, None, True),
    ]


def test_metadata_serialization_failure_is_shared_before_deserialization() -> None:
    results = sorted(run_gloo(world_size=2, fn=_serialization_error_worker))
    diagnostic = (
        "distributed metadata failed at rank=1 phase=metadata: "
        "ValueError: cannot serialize metadata"
    )
    assert results == [
        (0, diagnostic, None, 0),
        (1, diagnostic, "ValueError", 0),
    ]


def test_function_local_phase_labels_are_canonicalized_before_each_gather() -> None:
    results = sorted(run_gloo(world_size=2, fn=_phase_label_worker))
    expected_records = (
        ({"round": 0, "rank": 0}, {"round": 0, "rank": 1}),
        ({"round": 1, "rank": 0}, {"round": 1, "rank": 1}),
        ({"round": 2, "rank": 0}, {"round": 2, "rank": 1}),
    )
    for rank, row in enumerate(results):
        assert row == (
            rank,
            (
                ("function_local_start", expected_records[0]),
                ("function_local_finish", expected_records[1]),
                ("guarded_finish", expected_records[2]),
            ),
            3,
            0,
        )


def test_exception_message_subclass_is_canonicalized_before_gather() -> None:
    results = sorted(run_gloo(world_size=2, fn=_subclass_exception_message_worker))
    diagnostic = (
        "distributed message_normalization failed at rank=0 "
        "phase=message_normalization: "
        "_MessageSubclassError: message subclass contents"
    )
    assert results == [
        (0, diagnostic, "_MessageSubclassError", 1, 0),
        (1, diagnostic, None, 1, 0),
    ]


def test_collective_configuration_is_validated_before_any_gather() -> None:
    results = sorted(run_gloo(world_size=2, fn=_validation_worker))
    assert results == [(0, "validated"), (1, "validated")]


def test_mocked_nccl_device_priority_and_explicit_failures() -> None:
    results = sorted(run_gloo(world_size=2, fn=_cuda_selection_worker))
    assert results == [(0, "cuda-priority-checked"), (1, "cuda-priority-checked")]


@pytest.mark.multigpu
def test_manager_named_nccl_group_scopes_device_and_agrees_on_errors() -> None:
    ctx = mp.get_context("spawn")
    manager_port = free_port()
    external_port = free_port()
    processes = [
        ctx.Process(
            target=_manager_nccl_worker,
            args=(rank, 2, manager_port, external_port),
        )
        for rank in range(2)
    ]
    for process in processes:
        process.start()
    try:
        for process in processes:
            process.join(timeout=180)
            if process.is_alive():
                process.terminate()
                raise TimeoutError("Manager/NCCL worker did not finish")
            assert process.exitcode == 0
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
