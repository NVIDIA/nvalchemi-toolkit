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
"""Evaluation sink interfaces and Zarr-backed storage."""

from __future__ import annotations

import contextlib
from collections.abc import Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import numpy as np
import torch
import zarr
from tensordict import TensorDict
from zarr.abc.store import Store
from zarr.errors import ContainsGroupError
from zarr.storage import LocalStore, MemoryStore, StorePath

from nvalchemi.data import Batch
from nvalchemi.data.datapipes.backends.zarr import (
    AtomicDataZarrWriter,
    StoreLike,
    ZarrWriteConfig,
)
from nvalchemi.data.level_storage import (
    MultiLevelStorage,
    SegmentedLevelStorage,
    UniformLevelStorage,
)
from nvalchemi.training.distributed import (
    barrier as distributed_barrier,
)
from nvalchemi.training.distributed import (
    get_rank as distributed_get_rank,
)
from nvalchemi.training.distributed import (
    get_world_size as distributed_get_world_size,
)
from nvalchemi.training.distributed import (
    is_distributed_initialized,
)

if TYPE_CHECKING:
    from nvalchemi.distributed import DistributedManager

__all__ = ["EvaluationSink", "EvaluationZarrSink"]


@runtime_checkable
class EvaluationSink(Protocol):
    """Protocol for sinks that consume granular evaluation output batches."""

    def begin_evaluation(self, *, step_count: int, epoch: int, name: str) -> None:
        """Start one validation/evaluation run."""
        ...

    def write_samples(
        self,
        batch: Batch,
        *,
        step_count: int,
        epoch: int,
        batch_count: int,
    ) -> None:
        """Write an augmented per-sample validation batch."""
        ...

    def write_batch_summary(
        self,
        batch: Batch,
        *,
        step_count: int,
        epoch: int,
        batch_count: int,
    ) -> None:
        """Write a summary batch for one validation batch."""
        ...

    def write_epoch_summary(
        self,
        batch: Batch,
        *,
        step_count: int,
        epoch: int,
        local_summary: Mapping[str, torch.Tensor],
        global_summary: Mapping[str, torch.Tensor] | None,
    ) -> None:
        """Write validation-epoch summary statistics."""
        ...

    def end_evaluation(self, *, step_count: int, epoch: int, name: str) -> None:
        """Finish one validation/evaluation run."""
        ...


class EvaluationZarrSink:
    """Asynchronously write evaluation outputs into a single Zarr store.

    Parameters
    ----------
    store : StoreLike
        Root Zarr store for all evaluation outputs.
    config : ZarrWriteConfig | Mapping[str, Any] | None, optional
        Configuration forwarded to :class:`AtomicDataZarrWriter` when writing
        augmented sample batches.
    distributed_manager : DistributedManager | None, optional
        Structural distributed manager used for rank/world metadata and
        barriers. ``TrainingStrategy.validate()`` wires the strategy
        manager into this sink automatically when omitted.
    """

    def __init__(
        self,
        store: StoreLike,
        config: ZarrWriteConfig | Mapping[str, Any] | None = None,
        distributed_manager: DistributedManager | None = None,
    ) -> None:
        self.store = store
        if isinstance(config, Mapping):
            config = ZarrWriteConfig.model_validate(config)
        self.config = config if config is not None else ZarrWriteConfig()
        self.distributed_manager = distributed_manager
        self._store_path = _as_store_path(store)
        self._streams: dict[torch.device, torch.cuda.Stream] = {}
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._futures: list[Future[None]] = []

    def __enter__(self) -> EvaluationZarrSink:
        """Return this sink as a context manager."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        """Flush pending writes and release executor resources."""
        self.close()

    def set_distributed_manager(self, manager: DistributedManager | None) -> None:
        """Attach a workflow distributed manager when none was configured."""
        if self.distributed_manager is None:
            self.distributed_manager = manager

    def begin_evaluation(self, *, step_count: int, epoch: int, name: str) -> None:
        """Ensure the root store exists for one evaluation run."""
        del epoch, name
        if _distributed_rank(self.distributed_manager) == 0:
            self._mark_step(step_count)
        if _distributed_active(self.distributed_manager):
            distributed_barrier(self.distributed_manager)

    def write_samples(
        self,
        batch: Batch,
        *,
        step_count: int,
        epoch: int,
        batch_count: int,
    ) -> None:
        """Write one augmented validation batch under ``<step>/<rank>/<batch>``."""
        del epoch
        path = self._batch_path(step_count=step_count, batch_count=batch_count)
        snapshot, stream = self._snapshot_batch(batch)
        self._submit(stream, self._write_batch, path, snapshot)

    def write_batch_summary(
        self,
        batch: Batch,
        *,
        step_count: int,
        epoch: int,
        batch_count: int,
    ) -> None:
        """Write one compact validation-batch summary."""
        del epoch
        path = (
            self._store_path
            / str(step_count)
            / str(_distributed_rank(self.distributed_manager))
            / "batch_summaries"
            / str(batch_count)
        )
        snapshot, stream = self._snapshot_batch(batch)
        self._submit(stream, self._write_batch, path, snapshot)

    def write_epoch_summary(
        self,
        batch: Batch,
        *,
        step_count: int,
        epoch: int,
        local_summary: Mapping[str, torch.Tensor],
        global_summary: Mapping[str, torch.Tensor] | None,
    ) -> None:
        """Write per-rank and rank-zero validation-epoch summaries."""
        del epoch
        snapshot, stream = self._snapshot_batch(batch)
        self.flush()
        self._ensure_epoch_summary_arrays(
            step_count=step_count,
            local_summary=local_summary,
            global_summary=global_summary,
        )
        self._submit(
            stream,
            self._write_epoch_summary,
            step_count,
            snapshot,
            _summary_to_numpy(local_summary),
            None if global_summary is None else _summary_to_numpy(global_summary),
        )

    def end_evaluation(self, *, step_count: int, epoch: int, name: str) -> None:
        """Flush all writes queued for one evaluation run."""
        del step_count, epoch, name
        self.flush()

    def close(self) -> None:
        """Flush pending writes and reset the background worker."""
        self.flush()
        self._executor.shutdown(wait=True)
        self._executor = ThreadPoolExecutor(max_workers=1)
        for stream in self._streams.values():
            stream.synchronize()
        self._streams.clear()

    def flush(self) -> None:
        """Wait for all queued asynchronous writes to finish."""
        futures, self._futures = self._futures, []
        for future in futures:
            future.result()

    def _snapshot_batch(self, batch: Batch) -> tuple[Batch, torch.cuda.Stream | None]:
        """Detach and stage ``batch`` on CPU without blocking CUDA compute."""
        device = batch.device
        if device.type != "cuda":
            return _snapshot_batch_to_cpu(batch, stream=None), None

        stream = self._stream_for_device(device)
        main_stream = torch.cuda.current_stream(device)
        with torch.cuda.device(device), torch.cuda.stream(stream):
            stream.wait_stream(main_stream)
            snapshot = _snapshot_batch_to_cpu(batch, stream=stream)
        return snapshot, stream

    def _stream_for_device(self, device: torch.device) -> torch.cuda.Stream:
        """Return the CUDA copy stream for ``device``."""
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        stream = self._streams.get(device)
        if stream is None:
            with torch.cuda.device(device):
                stream = torch.cuda.Stream(device=device)
            self._streams[device] = stream
        return stream

    def _submit(
        self,
        stream: torch.cuda.Stream | None,
        callback: Any,
        *args: Any,
    ) -> None:
        """Submit a callback that waits on ``stream`` before touching data."""
        self._futures.append(
            self._executor.submit(_run_after_stream, stream, callback, *args)
        )

    def _submit_no_stream(self, callback: Any, *args: Any) -> None:
        """Submit a callback that does not depend on a CUDA transfer stream."""
        self._futures.append(self._executor.submit(callback, *args))

    def _batch_path(self, *, step_count: int, batch_count: int) -> StorePath:
        """Return the Zarr path for one per-rank validation batch."""
        return (
            self._store_path
            / str(step_count)
            / str(_distributed_rank(self.distributed_manager))
            / str(batch_count)
        )

    def _mark_step(self, step_count: int) -> None:
        """Create the step group and basic metadata."""
        root = zarr.open(self._store_path, mode="a")
        step_group = _require_group(root, str(step_count))
        step_group.attrs["format"] = "nvalchemi-evaluation-v1"

    def _write_batch(self, path: StorePath, batch: Batch) -> None:
        """Write one augmented batch to a leaf Zarr store."""
        AtomicDataZarrWriter(path, config=self.config).write(batch)

    def _ensure_epoch_summary_arrays(
        self,
        *,
        step_count: int,
        local_summary: Mapping[str, torch.Tensor],
        global_summary: Mapping[str, torch.Tensor] | None,
    ) -> None:
        """Create rank-mean and summary arrays before distributed writes."""
        rank = _distributed_rank(self.distributed_manager)
        world_size = _distributed_world_size(self.distributed_manager)
        if rank == 0:
            self._create_epoch_summary_arrays(
                step_count=step_count,
                world_size=world_size,
                local_summary=local_summary,
                global_summary=global_summary,
            )
        if _distributed_active(self.distributed_manager):
            distributed_barrier(self.distributed_manager)
        if rank != 0:
            self._ensure_rank_mean_arrays_exist(step_count, local_summary)

    def _create_epoch_summary_arrays(
        self,
        *,
        step_count: int,
        world_size: int,
        local_summary: Mapping[str, torch.Tensor],
        global_summary: Mapping[str, torch.Tensor] | None,
    ) -> None:
        """Create summary arrays if they are absent."""
        root = zarr.open(self._store_path, mode="a")
        step_group = _require_group(root, str(step_count))
        rank_group = _require_group(step_group, "rank_means")
        for name in local_summary:
            if name not in rank_group:
                rank_group.create_array(
                    name,
                    data=np.full(world_size, np.nan, dtype=np.float64),
                    chunks=(1,),
                )
        if global_summary is None:
            return
        summary_group = _require_group(step_group, "summary")
        for name, value in _summary_to_numpy(global_summary).items():
            if name not in summary_group:
                summary_group.create_array(name, data=value)

    def _write_epoch_summary(
        self,
        step_count: int,
        batch: Batch,
        local_summary: Mapping[str, np.ndarray],
        global_summary: Mapping[str, np.ndarray] | None,
    ) -> None:
        """Write epoch summary batch and scalar arrays."""
        root = zarr.open(self._store_path, mode="a")
        step_group = _require_group(root, str(step_count))
        rank = _distributed_rank(self.distributed_manager)
        rank_group = _require_group(step_group, "rank_means")
        for name, value in local_summary.items():
            rank_group[name][rank] = value

        if rank == 0 and global_summary is not None:
            summary_group = _require_group(step_group, "summary")
            for name, value in global_summary.items():
                if name in summary_group:
                    summary_group[name][...] = value
                else:
                    summary_group.create_array(name, data=value)

        if rank == 0:
            summary_path = self._store_path / str(step_count) / "summary_batch"
            with contextlib.suppress(FileExistsError):
                AtomicDataZarrWriter(summary_path, config=self.config).write(batch)

    def _ensure_rank_mean_arrays_exist(
        self,
        step_count: int,
        local_summary: Mapping[str, torch.Tensor],
    ) -> None:
        """Verify rank-zero created all rank-mean arrays."""
        root = zarr.open(self._store_path, mode="a")
        step_group = _require_group(root, str(step_count))
        rank_group = _require_group(step_group, "rank_means")
        missing = sorted(name for name in local_summary if name not in rank_group)
        if missing:
            raise RuntimeError(
                "EvaluationZarrSink rank-zero summary setup did not create "
                f"rank-mean array(s): {missing}."
            )


def _as_store_path(store: StoreLike) -> StorePath:
    """Return ``store`` as a Zarr :class:`StorePath`."""
    if isinstance(store, StorePath):
        return store
    if isinstance(store, (str, Path)):
        return StorePath(LocalStore(store))
    if isinstance(store, dict):
        return StorePath(MemoryStore(store))
    if isinstance(store, Store):
        return StorePath(store)
    return StorePath(store)


def _snapshot_batch_to_cpu(
    batch: Batch,
    *,
    stream: torch.cuda.Stream | None,
) -> Batch:
    """Return a detached CPU batch snapshot with one tensor copy per field."""
    groups: dict[str, UniformLevelStorage | SegmentedLevelStorage] = {}
    for name, group in batch._storage.groups.items():
        data = {
            key: _snapshot_tensor_to_cpu(tensor, stream=stream)
            for key, tensor in group.items()
        }
        attr_map = group.attr_map.clone()
        if isinstance(group, SegmentedLevelStorage):
            groups[name] = _snapshot_segmented_group(group, data, attr_map, stream)
        else:
            groups[name] = _snapshot_uniform_group(group, data, attr_map)
    storage = MultiLevelStorage(
        groups=groups,
        validate=False,
        attr_map=batch._storage.attr_map.clone(),
        device=torch.device("cpu"),
    )
    return Batch._construct(
        device=torch.device("cpu"),
        keys={key: value.copy() for key, value in batch.keys.items()}
        if batch.keys is not None
        else None,
        storage=storage,
        data_class=batch._data_class,
    )


def _snapshot_uniform_group(
    group: UniformLevelStorage,
    data: dict[str, torch.Tensor],
    attr_map: Any,
) -> UniformLevelStorage:
    """Return a CPU snapshot of a uniform storage group."""
    out = UniformLevelStorage(
        data=_tensor_dict(data, batch_size=group._data.batch_size),
        device=torch.device("cpu"),
        attr_map=attr_map,
        validate=False,
    )
    if getattr(group, "_num_kept", None) is not None:
        object.__setattr__(out, "_num_kept", group._num_kept)
    return out


def _snapshot_segmented_group(
    group: SegmentedLevelStorage,
    data: dict[str, torch.Tensor],
    attr_map: Any,
    stream: torch.cuda.Stream | None,
) -> SegmentedLevelStorage:
    """Return a CPU snapshot of a segmented storage group."""
    out = SegmentedLevelStorage(
        data=_tensor_dict(data, batch_size=group._data.batch_size),
        device=torch.device("cpu"),
        attr_map=attr_map,
        segment_lengths=_snapshot_tensor_to_cpu(group.segment_lengths, stream=stream),
        batch_idx=None
        if group._batch_idx is None
        else _snapshot_tensor_to_cpu(group._batch_idx, stream=stream),
        batch_ptr=None
        if group._batch_ptr is None
        else _snapshot_tensor_to_cpu(group._batch_ptr, stream=stream),
        validate=False,
    )
    if getattr(group, "_num_segments", None) is not None:
        object.__setattr__(out, "_num_segments", group._num_segments)
        object.__setattr__(out, "_num_elements_kept", group._num_elements_kept)
    return out


def _tensor_dict(
    data: dict[str, torch.Tensor],
    *,
    batch_size: torch.Size,
) -> TensorDict:
    """Return a CPU TensorDict with batch size inferred from ``data``."""
    if not data:
        return TensorDict({}, batch_size=batch_size, device=torch.device("cpu"))
    first_dim = next(iter(data.values())).shape[0]
    return TensorDict(data, batch_size=[first_dim], device=torch.device("cpu"))


def _snapshot_tensor_to_cpu(
    tensor: torch.Tensor,
    *,
    stream: torch.cuda.Stream | None,
) -> torch.Tensor:
    """Detach ``tensor`` and copy it to CPU for background serialization."""
    source = tensor.detach()
    if source.device.type != "cuda":
        return source.to("cpu", copy=True)
    if stream is not None:
        source.record_stream(stream)
    target = torch.empty(
        source.shape,
        dtype=source.dtype,
        device=torch.device("cpu"),
        pin_memory=True,
    )
    target.copy_(source, non_blocking=True)
    return target


def _run_after_stream(
    stream: torch.cuda.Stream | None,
    callback: Any,
    *args: Any,
) -> None:
    """Synchronize ``stream`` before running ``callback``."""
    if stream is not None:
        stream.synchronize()
    callback(*args)


def _summary_to_numpy(
    summary: Mapping[str, torch.Tensor],
) -> dict[str, np.ndarray]:
    """Convert scalar tensor summaries to NumPy arrays."""
    return {
        key: value.detach().cpu().to(torch.float64).reshape(()).numpy()
        for key, value in summary.items()
    }


def _distributed_active(manager: DistributedManager | None = None) -> bool:
    """Return whether distributed communication is initialized."""
    return is_distributed_initialized(manager)


def _distributed_rank(manager: DistributedManager | None = None) -> int:
    """Return the current distributed rank, defaulting to zero."""
    return distributed_get_rank(manager)


def _distributed_world_size(manager: DistributedManager | None = None) -> int:
    """Return the distributed world size, defaulting to one."""
    return distributed_get_world_size(manager)


def _require_group(parent: zarr.Group, name: str) -> zarr.Group:
    """Return an existing child group or create it."""
    if name in parent:
        return parent[name]
    try:
        return parent.create_group(name)
    except (ContainsGroupError, FileExistsError):
        return parent[name]
