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
"""
In-memory dataset with fully loaded :class:`Batch` support.

:class:`InMemoryDataset` keeps the entire dataset resident in memory as one
:class:`~nvalchemi.data.batch.Batch`. Provide the batch directly when data is
already loaded, or pass a :class:`~nvalchemi.data.datapipes.dataset.ReaderProtocol`
to materialize the full dataset during initialization.

Once loaded, iteration and :meth:`load_batches` select graphs from the
in-memory batch instead of reading from storage on each access.
``InMemoryDataset`` follows
:class:`~nvalchemi.data.datapipes.dataset.BatchDatasetProtocol`, exposing the
same DataLoader-facing batch methods as ``Dataset``: ``load_batches(...)``,
``prefetch_fused_batches(...)``, ``get_fused_batches()``,
``has_pending_fused_batches()``, and ``cancel_prefetch(...)``.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor

from nvalchemi.data.atomic_data import AtomicData
from nvalchemi.data.batch import Batch
from nvalchemi.data.transforms import Compose

if TYPE_CHECKING:
    from nvalchemi._typing import BatchTransform
    from nvalchemi.data.datapipes.dataset import ReaderProtocol


@dataclass(slots=True)
class _PendingInMemoryBatches:
    """Prepared in-memory batches with an optional CUDA transfer event.

    Attributes
    ----------
    batches : list[Batch]
        Selected batches ready for consumption.
    event : torch.cuda.Event | None
        CUDA event for stream synchronization, or None.
    """

    batches: list[Batch]
    event: torch.cuda.Event | None = None


class InMemoryDataset:
    """Dataset that keeps the full dataset in memory as one :class:`Batch`.

    The entire dataset is stored in ``in_memory_batch``. Provide it directly if
    you have already loaded the data, or pass a ``reader`` to materialize the
    full dataset during initialization.

    Once loaded, iteration and :meth:`load_batches` select graphs from that
    in-memory batch instead of reading from storage on each access.
    ``InMemoryDataset`` follows
    :class:`~nvalchemi.data.datapipes.dataset.BatchDatasetProtocol`, exposing
    the same DataLoader-facing batch methods as ``Dataset``: :meth:`__len__`,
    :meth:`load_batches`, :meth:`prefetch_fused_batches`,
    :meth:`get_fused_batches`, :meth:`has_pending_fused_batches`, and
    :meth:`cancel_prefetch`. These methods use in-memory implementations rather
    than reader-backed implementations. The :meth:`__getitem__` and
    :meth:`__iter__` methods are provided for
    dataset-style access, but normal
    :class:`~nvalchemi.data.datapipes.DataLoader` iteration loads whole
    :class:`Batch` objects through the aforementioned methods instead of calling
    :meth:`__getitem__` for each sample.

    Parameters
    ----------
    in_memory_batch : Batch | None, default=None
        Fully loaded batch containing all graphs in the dataset. Pass this when
        the batch has already been built.
    reader : ReaderProtocol | None, default=None
        Reader to materialize the full dataset into ``in_memory_batch``. Pass
        either ``in_memory_batch`` or ``reader``, not both.
    chunk_size : int, default=4096
        Number of reader samples to materialize per intermediate batch.
    device : str | torch.device | None, default=None
        Target device for emitted samples and batches. ``None`` leaves emitted
        batches on the resident cache device. ``"auto"`` selects CUDA when
        available, otherwise CPU.
    skip_validation : bool, default=False
        If ``True``, bypass ``AtomicData`` construction and Pydantic
        validation while materializing from a reader, building batches
        directly from raw tensor dicts via
        :meth:`~nvalchemi.data.batch.Batch.from_raw_dicts`. Enable this for
        trusted stores that are already validated.
    batch_transforms : Sequence[BatchTransform] | None, default=None
        Optional per-batch transforms applied while building the resident
        batch. For reader-backed construction they run on each materialized
        chunk; for a pre-built ``in_memory_batch`` they run once on the full
        batch. This mirrors the ``DataLoader(batch_transforms=...)`` API.

    Raises
    ------
    ValueError
        If neither or both of ``in_memory_batch`` and ``reader`` are provided.
    TypeError
        If ``batch_transforms`` is not a
        :class:`~collections.abc.Sequence` (e.g. a single callable or a
        generator was passed).

    Attributes
    ----------
    in_memory_batch : Batch
        Fully loaded batch containing all graphs in the dataset.
    target_device : torch.device | None
        Resolved target device for emitted samples and batches.

    Examples
    --------
    >>> from nvalchemi.data.datapipes.in_memory_dataset import InMemoryDataset
    >>> # Assuming a concrete Reader implementation exists:
    >>> # reader = MyReader("dataset.zarr")  # doctest: +SKIP
    >>> # ds = InMemoryDataset(reader=reader, device="cpu")  # doctest: +SKIP
    >>> # batch = ds.load_batches([[0, 1, 2]])[0]              # doctest: +SKIP
    >>> # trusted = InMemoryDataset(reader=reader, device="cuda", skip_validation=True)  # doctest: +SKIP

    With a pre-built in-memory batch:

    >>> # batch = Batch.from_raw_dicts(raw_dicts)              # doctest: +SKIP
    >>> # ds = InMemoryDataset(in_memory_batch=batch)          # doctest: +SKIP
    """

    def __init__(
        self,
        in_memory_batch: Batch | None = None,
        *,
        reader: "ReaderProtocol | None" = None,
        chunk_size: int = 4096,
        device: str | torch.device | None = None,
        skip_validation: bool = False,
        batch_transforms: "Sequence[BatchTransform] | None" = None,
    ) -> None:
        """Initialize the in-memory dataset."""
        if (in_memory_batch is None) == (reader is None):
            raise ValueError("Pass exactly one of in_memory_batch or reader.")
        self.target_device = self._resolve_target_device(device)
        if reader is not None:
            in_memory_batch = self._materialize_reader(
                reader,
                chunk_size=chunk_size,
                skip_validation=skip_validation,
                batch_transforms=batch_transforms,
            )
        else:
            transform = self._build_batch_transform(batch_transforms)
            if transform is not None:
                in_memory_batch = transform(in_memory_batch)
        self.in_memory_batch = in_memory_batch
        self._pin_memory = False
        self._fused_batch_prefetch_queue: deque[_PendingInMemoryBatches] = deque()

    @staticmethod
    def _resolve_target_device(
        device: str | torch.device | None,
    ) -> torch.device | None:
        """Resolve the optional emitted-batch target device.

        Parameters
        ----------
        device : str | torch.device | None
            Requested device. ``None`` leaves emitted batches on the resident
            cache device. ``"auto"`` selects CUDA when available, otherwise CPU.

        Returns
        -------
        torch.device | None
            Resolved target device, or ``None`` when no transfer is requested.

        Raises
        ------
        TypeError
            If *device* is not a string, ``torch.device``, or ``None``.
        """
        if device is None:
            return None
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        elif not isinstance(device, (str, torch.device)):
            raise TypeError(
                f"Device expected to be a string, torch.device, or None. Got {device}."
            )
        return torch.device(device)

    @staticmethod
    def _build_batch_transform(
        batch_transforms: "Sequence[BatchTransform] | None",
    ) -> Compose | None:
        """Validate and compose optional per-batch transforms.

        Parameters
        ----------
        batch_transforms : Sequence[BatchTransform] | None
            Optional per-batch transforms.

        Returns
        -------
        Compose | None
            Composed transform pipeline, or ``None`` when no transforms are set.

        Raises
        ------
        TypeError
            If ``batch_transforms`` is not a
            :class:`~collections.abc.Sequence`.
        """
        if batch_transforms is not None and not isinstance(batch_transforms, Sequence):
            raise TypeError(
                "batch_transforms must be a Sequence of callables, not a "
                "single callable or generator. Pass [fn] instead of fn."
            )
        return Compose(batch_transforms) if batch_transforms else None

    @staticmethod
    def _materialize_reader(
        reader: "ReaderProtocol",
        *,
        chunk_size: int,
        skip_validation: bool,
        batch_transforms: "Sequence[BatchTransform] | None",
    ) -> Batch:
        """Read an entire reader into one in-memory batch.

        Parameters
        ----------
        reader : ReaderProtocol
            Reader providing raw tensor dicts from a data source.
        chunk_size : int
            Number of reader samples to materialize per intermediate batch.
        skip_validation : bool
            If ``True``, build chunks directly from raw tensor dictionaries.
            If ``False``, validate raw samples as :class:`AtomicData` before
            batching.
        batch_transforms : Sequence[BatchTransform] | None
            Optional per-batch transforms applied to each materialized chunk.

        Returns
        -------
        Batch
            Fully loaded batch containing all graphs from ``reader``.

        Raises
        ------
        ValueError
            If ``chunk_size`` is not positive or ``reader`` is empty.
        TypeError
            If ``batch_transforms`` is not a
            :class:`~collections.abc.Sequence`.
        RuntimeError
            If no samples were materialized from ``reader``.
        """
        in_memory_batch: Batch | None = None
        try:
            if chunk_size <= 0:
                raise ValueError(f"chunk_size must be positive, got {chunk_size}")

            reader_field_levels = getattr(reader, "field_levels", None)
            transform = InMemoryDataset._build_batch_transform(batch_transforms)
            reader_len = len(reader)
            if reader_len <= 0:
                raise ValueError("Cannot materialize an empty reader.")
            for start in range(0, reader_len, chunk_size):
                end = min(start + chunk_size, reader_len)
                raw_samples = reader.read_many(range(start, end))
                raw_dicts = [tensor_dict for tensor_dict, _metadata in raw_samples]
                if skip_validation:
                    chunk = Batch.from_raw_dicts(
                        raw_dicts,
                        device="cpu",
                        field_levels=reader_field_levels,
                    )
                else:
                    chunk = Batch.from_data_list(
                        [AtomicData.model_validate(data) for data in raw_dicts],
                        device="cpu",
                        field_levels=reader_field_levels,
                    )
                if transform is not None:
                    chunk = transform(chunk)
                if in_memory_batch is None:
                    in_memory_batch = chunk
                else:
                    in_memory_batch.append(chunk)
        finally:
            reader.close()

        if in_memory_batch is None:
            raise RuntimeError("No samples were materialized from reader.")
        return in_memory_batch

    def __len__(self) -> int:
        """Return the number of graphs in the in-memory batch.

        Returns
        -------
        int
            Number of graphs stored in ``in_memory_batch``.
        """
        return self.in_memory_batch.num_graphs

    @property
    def field_names(self) -> list[str]:
        """Return field names available in in-memory samples.

        Returns
        -------
        list[str]
            Field names exposed by the resident batch.
        """
        if len(self) > 0:
            return [
                key
                for key in self.in_memory_batch[0].to_dict()
                if not key.startswith("__")
            ]
        if self.in_memory_batch.keys is None:
            return []
        return [
            *sorted(self.in_memory_batch.keys.get("node", set())),
            *sorted(self.in_memory_batch.keys.get("edge", set())),
            *sorted(self.in_memory_batch.keys.get("system", set())),
        ]

    @property
    def pin_memory(self) -> bool:
        """Whether the materialized CPU batch is pinned in page-locked memory."""
        return self._pin_memory

    @pin_memory.setter
    def pin_memory(self, enabled: bool) -> None:
        """Pin the materialized CPU batch when enabled by :class:`DataLoader`.

        Parameters
        ----------
        enabled : bool
            Whether the resident CPU batch should be page-locked.
        """
        enabled = bool(enabled)
        if (
            enabled
            and not self._pin_memory
            and self.in_memory_batch.device.type == "cpu"
        ):
            self.in_memory_batch.pin_memory()
        self._pin_memory = enabled

    def _normalize_indices(self, indices: Any) -> Tensor:
        """Normalize DataLoader sampler output to CPU int64 indices.

        Parameters
        ----------
        indices : Any
            Index batch from a sampler or batch-index list.

        Returns
        -------
        Tensor
            1-D CPU tensor of graph indices with dtype ``torch.long``.

        Raises
        ------
        TypeError
            If *indices* is not a supported index container type.
        """
        if isinstance(indices, Tensor):
            return indices.to(dtype=torch.long, device="cpu")
        if isinstance(indices, list) and indices and isinstance(indices[0], Tensor):
            return torch.stack(indices).to(dtype=torch.long, device="cpu")
        if isinstance(indices, Sequence):
            return torch.as_tensor(indices, dtype=torch.long, device="cpu")
        raise TypeError(f"Unexpected index batch type: {type(indices).__name__}")

    def _move_batch_to_target(self, batch: Batch) -> Batch:
        """Move a selected batch to the configured target device when needed."""
        if self.target_device is None or batch.device == self.target_device:
            return batch
        return batch.to(self.target_device, non_blocking=True)

    def _prepare_batches(
        self,
        batch_index_lists: Sequence[Sequence[int]] | Sequence[Tensor],
        stream: torch.cuda.Stream | None = None,
    ) -> _PendingInMemoryBatches:
        """Select batches and optionally enqueue target-device transfers.

        Parameters
        ----------
        batch_index_lists : Sequence[Sequence[int]] | Sequence[Tensor]
            Per-batch graph indices.
        stream : torch.cuda.Stream | None, default=None
            Optional CUDA stream for target-device transfers.

        Returns
        -------
        _PendingInMemoryBatches
            Selected batches with an optional CUDA synchronization event.
        """
        batches = [
            self.in_memory_batch.index_select(self._normalize_indices(indices))
            for indices in batch_index_lists
        ]
        event: torch.cuda.Event | None = None
        if (
            stream is not None
            and self.target_device is not None
            and self.target_device.type == "cuda"
        ):
            with torch.cuda.stream(stream):
                batches = [self._move_batch_to_target(batch) for batch in batches]
            event = torch.cuda.Event()
            event.record(stream)
        else:
            batches = [self._move_batch_to_target(batch) for batch in batches]
        return _PendingInMemoryBatches(batches=batches, event=event)

    def load_batches(
        self,
        batch_index_lists: Sequence[Sequence[int]] | Sequence[Tensor],
        stream: torch.cuda.Stream | None = None,
    ) -> list[Batch]:
        """Load several batches immediately.

        This is the synchronous counterpart to
        :meth:`prefetch_fused_batches`/:meth:`get_fused_batches`. Each
        requested index list selects graphs from ``in_memory_batch`` and
        returns one :class:`~nvalchemi.data.batch.Batch` per input list.

        Parameters
        ----------
        batch_index_lists : Sequence[Sequence[int]] | Sequence[Tensor]
            Per-batch graph indices.
        stream : torch.cuda.Stream | None, default=None
            CUDA stream for target-device transfers when supported.

        Returns
        -------
        list[Batch]
            One :class:`Batch` per input batch-index list.
        """
        pending = self._prepare_batches(batch_index_lists, stream=stream)
        if pending.event is not None:
            pending.event.synchronize()
        return pending.batches

    def prefetch_fused_batches(
        self,
        batch_index_lists: Sequence[Sequence[int]] | Sequence[Tensor],
        stream: torch.cuda.Stream | None = None,
    ) -> None:
        """Submit multiple batches for fused prefetch.

        Selected batches are prepared immediately and queued for later
        consumption via :meth:`get_fused_batches`.

        Parameters
        ----------
        batch_index_lists : Sequence[Sequence[int]] | Sequence[Tensor]
            Per-batch graph indices.
        stream : torch.cuda.Stream | None, default=None
            CUDA stream for target-device transfers when supported.

        Raises
        ------
        RuntimeError
            If the fused batch prefetch queue is already full.
        """
        if len(self._fused_batch_prefetch_queue) >= 2:
            raise RuntimeError(
                "Fused batch prefetch queue is full; consume a pending chunk first."
            )
        self._fused_batch_prefetch_queue.append(
            self._prepare_batches(batch_index_lists, stream=stream)
        )

    def prefetch(self, index: int, stream: torch.cuda.Stream | None = None) -> None:
        """Satisfy the sample-prefetch API without queuing work.

        ``InMemoryDataset`` already holds every graph in ``in_memory_batch``, so
        there is no reader I/O to overlap for an individual sample.

        Parameters
        ----------
        index : int
            Graph index within ``in_memory_batch``.
        stream : torch.cuda.Stream | None, default=None
            Unused CUDA stream argument retained for compatibility.
        """
        return None

    def has_pending_fused_batches(self) -> bool:
        """Return whether a fused prefetch chunk is waiting to be consumed."""
        return bool(self._fused_batch_prefetch_queue)

    def get_fused_batches(self) -> Iterator[Batch]:
        """Consume the pending fused prefetch and yield per-batch results.

        Blocks until any queued CUDA transfers complete, then yields one
        :class:`~nvalchemi.data.batch.Batch` per sub-batch from the prepared
        request.

        Yields
        ------
        Batch
            One batch per sub-batch from the fused prefetch request.

        Raises
        ------
        RuntimeError
            If no fused prefetch is pending.
        """
        if not self._fused_batch_prefetch_queue:
            raise RuntimeError(
                "No fused batch prefetch pending; call prefetch_fused_batches() "
                "before get_fused_batches()."
            )
        pending = self._fused_batch_prefetch_queue.popleft()
        if pending.event is not None:
            pending.event.synchronize()
        yield from pending.batches

    def cancel_prefetch(self, index: int | None = None) -> None:
        """Cancel pending prefetch operations.

        Parameters
        ----------
        index : int | None, default=None
            Unused; retained for API compatibility with
            :class:`~nvalchemi.data.datapipes.dataset.Dataset`. Clears all
            pending fused batch requests.
        """
        del index
        self._fused_batch_prefetch_queue.clear()

    @property
    def prefetch_count(self) -> int:
        """Return the number of queued fused-batch prefetch requests."""
        return len(self._fused_batch_prefetch_queue)

    def get_metadata(self, index: int) -> tuple[int, int]:
        """Return lightweight graph-size metadata for an in-memory sample.

        Parameters
        ----------
        index : int
            Graph index within ``in_memory_batch``.

        Returns
        -------
        tuple[int, int]
            ``(num_atoms, num_edges)`` for the sample.
        """
        data = self.in_memory_batch[index]
        return data.num_nodes, data.num_edges

    def __getitem__(self, index: int) -> tuple["AtomicData", dict[str, Any]]:
        """Get an AtomicData sample by graph index.

        Parameters
        ----------
        index : int
            Graph index within ``in_memory_batch``.

        Returns
        -------
        tuple[AtomicData, dict[str, Any]]
            Tuple of (:class:`~nvalchemi.data.atomic_data.AtomicData`, empty
            metadata dict).

        Raises
        ------
        IndexError
            If *index* is out of range.
        """
        data = self.in_memory_batch[index]
        if self.target_device is not None:
            data = data.to(self.target_device, non_blocking=True)
        return data, {}

    def __iter__(self) -> Iterator[tuple["AtomicData", dict[str, Any]]]:
        """Iterate over all samples in the dataset.

        Yields
        ------
        tuple[AtomicData, dict[str, Any]]
            ``(AtomicData, metadata)`` for each graph in
            ``in_memory_batch``.
        """
        for index in range(len(self)):
            yield self[index]

    def close(self) -> None:
        """Release resources held by the dataset.

        Clears any pending fused batch prefetch requests.
        """
        self.cancel_prefetch()

    def __enter__(self) -> InMemoryDataset:
        """Enter context manager.

        Returns
        -------
        InMemoryDataset
            This dataset instance.
        """
        return self

    def __exit__(
        self, exc_type: type | None, exc_val: BaseException | None, exc_tb: Any
    ) -> None:
        """Exit context manager, calling :meth:`close`.

        Parameters
        ----------
        exc_type : type | None
            Exception type, if any.
        exc_val : BaseException | None
            Exception value, if any.
        exc_tb : Any
            Exception traceback, if any.
        """
        self.close()
