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
AtomicData-native dataset with CUDA-stream prefetching.

The main ``Dataset`` class is intended to be a drop-in replacement
for ``torch.data.Dataset``, and specializes for atomistic systems
beyond graphs. ``Dataset``s are constructed by passing in something
that implements the ``ReaderProtocol``, or users can subclass the
:class:`nvalchemi.data.datapipes.backends.base.Reader` class as well
to implement their own file format support.

In addition to treating atomistic systems as a first-class citizen,
the class also provides mechanisms data prefetching and use of
CUDA streams, which allow for highly performant data loading and
pre-processing workflows.
"""

from __future__ import annotations

import logging
from collections import deque
from collections.abc import Iterator, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import torch

from nvalchemi.data.atomic_data import AtomicData
from nvalchemi.data.batch import Batch
from nvalchemi.data.datapipes.backends.base import Reader
from nvalchemi.data.transforms import Compose

if TYPE_CHECKING:
    from nvalchemi._typing import SampleTransform

logger = logging.getLogger(__name__)


@runtime_checkable
class ReaderProtocol(Protocol):
    """Protocol for reader objects compatible with Dataset.

    This protocol enables duck-typed Reader implementations to be used
    with :class:`Dataset` without inheriting from the
    :class:`~nvalchemi.data.datapipes.backends.base.Reader` ABC.
    """

    def read_many(
        self, indices: Sequence[int]
    ) -> list[tuple[dict[str, torch.Tensor], dict[str, Any]]]:
        """Load raw tensor data and metadata for multiple samples."""
        ...

    def __len__(self) -> int:
        """Return the total number of available samples."""
        ...

    def close(self) -> None:
        """Release resources held by the reader."""
        ...


@dataclass
class _PrefetchResult:
    """Container for async prefetch results.

    Attributes
    ----------
    index : int
        Sample index that was loaded.
    data : AtomicData | None
        Loaded data, or None if not yet available or error occurred.
    metadata : dict[str, Any] | None
        Sample metadata, or None.
    error : Exception | None
        Exception if loading failed, or None.
    event : torch.cuda.Event | None
        CUDA event for stream synchronization, or None.
    """

    index: int
    data: AtomicData | None = None
    metadata: dict[str, Any] | None = None
    error: Exception | None = None
    event: torch.cuda.Event | None = None


@dataclass
class _FusedBatchPrefetchResult:
    """Container for fused multi-batch prefetch results.

    Used for both validated (AtomicData) and raw (dict) fused-prefetch
    paths.  When ``raw`` is ``True``, ``data`` holds raw tensor dicts
    and ``metadata`` is ``None``.

    Attributes
    ----------
    batch_splits : list[int]
        Number of samples in each sub-batch, used to split
        the flat result list back into per-batch groups.
    raw : bool
        Whether the data contains raw tensor dicts (True) or
        AtomicData objects (False).
    data : list[Any] | None
        Loaded samples in request order, or None on error.
    metadata : list[dict[str, Any]] | None
        Per-sample metadata (validated path only), or None.
    error : Exception | None
        Exception if loading failed, or None.
    event : torch.cuda.Event | None
        CUDA event for stream synchronization, or None.
    """

    batch_splits: list[int]
    raw: bool = False
    data: list[Any] | None = None
    metadata: list[dict[str, Any]] | None = None
    error: Exception | None = None
    event: torch.cuda.Event | None = None


@dataclass
class _PendingFusedBatch:
    """Queued fused batch request and its submitted future."""

    batch_index_lists: tuple[tuple[int, ...], ...]
    future: Future[_FusedBatchPrefetchResult]


class Dataset:
    """AtomicData-native dataset that bypasses TensorDict conversion.

    Wraps a :class:`~nvalchemi.data.datapipes.backends.base.Reader` and returns
    :class:`~nvalchemi.data.atomic_data.AtomicData` objects directly,
    with CUDA-stream prefetching support.

    Parameters
    ----------
    reader : Reader | ReaderProtocol
        Reader providing raw tensor dicts from a data source.
    device : str | torch.device | None, default=None
        Target device. ``"auto"`` picks CUDA if available, otherwise CPU.
    num_workers : int, default=2
        Thread pool size for async prefetch.
    transforms : Sequence[SampleTransform] | None, default=None
        Optional per-sample transforms applied after device transfer.
        See :meth:`__init__` for details.

    Attributes
    ----------
    reader : Reader | ReaderProtocol
        The underlying data reader.
    target_device : torch.device | None
        Resolved target device for data transfer.
    num_workers : int
        Number of worker threads for prefetching.

    Examples
    --------
    >>> from nvalchemi.data.datapipes.dataset import Dataset
    >>> from nvalchemi.data.datapipes.backends.base import Reader
    >>> # Assuming a concrete Reader implementation exists:
    >>> # reader = MyReader("dataset.zarr")  # doctest: +SKIP
    >>> # ds = Dataset(reader, device="cpu")  # doctest: +SKIP
    >>> # atomic_data, meta = ds[0]           # doctest: +SKIP

    With a user-supplied per-sample transform:

    >>> def shift(data, metadata):                              # doctest: +SKIP
    ...     return data.replace(positions=data.positions + 1.0), metadata
    >>> ds = Dataset(reader, device="cpu", transforms=[shift])  # doctest: +SKIP
    >>> atomic_data, meta = ds[0]                               # doctest: +SKIP
    """

    def __init__(
        self,
        reader: Reader | ReaderProtocol,
        *,
        device: str | torch.device | None = None,
        num_workers: int = 2,
        skip_validation: bool = False,
        transforms: Sequence[SampleTransform] | None = None,
    ) -> None:
        """Initialize the AtomicData-native dataset.

        Parameters
        ----------
        reader : Reader | ReaderProtocol
            Reader providing raw data from a data source.
        device : str | torch.device | None, default=None
            Target device. ``"auto"`` picks CUDA if available, otherwise CPU.
        num_workers : int, default=2
            Thread pool size for async prefetch.
        skip_validation : bool, default=False
            If ``True``, bypass ``AtomicData`` construction and Pydantic
            validation in the fused batch prefetch path, building batches
            directly from raw tensor dicts via
            :meth:`~nvalchemi.data.batch.Batch.from_raw_dicts`. This
            is safe when the backing store is already validated (e.g.
            data written by :class:`AtomicDataZarrWriter`).
        transforms : Sequence[SampleTransform] | None, default=None
            Optional per-sample transforms applied after device transfer.
            ``None`` or an empty sequence disables transform application
            (zero runtime overhead on the hot path). Non-empty sequences
            are composed via :class:`~nvalchemi.data.transforms.Compose`;
            see :data:`~nvalchemi.data.transforms._types.SampleTransform`
            for the expected signature.

        Raises
        ------
        TypeError
            If ``reader`` does not implement the required interface, or
            if ``transforms`` is not a :class:`~collections.abc.Sequence`
            (e.g. a single callable or a generator was passed).
        RuntimeError
            Raised from :meth:`__getitem__` when any transform fails;
            the original exception is attached via ``__cause__``.

        Notes
        -----
        Transforms execute on the prefetch CUDA stream when prefetching
        is active. They must use stream-aware ops only; avoid ``.item()``,
        ``.cpu()``, ``.numpy()``, :func:`torch.cuda.synchronize`, or
        overriding ``stream=`` inside transforms, as these would
        serialize the prefetch worker with the main stream.
        """
        has_batch_reader = hasattr(reader, "read_many")
        has_sample_reader = hasattr(reader, "_load_sample") and hasattr(
            reader, "_get_sample_metadata"
        )
        if not isinstance(reader, Reader) and not (
            has_batch_reader or has_sample_reader
        ):
            raise TypeError(
                f"reader must implement Reader interface, got {type(reader).__name__}"
            )

        # Validate transforms is a Sequence (catches single-callable / generator)
        if transforms is not None and not isinstance(transforms, Sequence):
            raise TypeError(
                "transforms must be a Sequence of callables, not a single "
                "callable or generator. Pass [fn] instead of fn."
            )

        target_device = self._resolve_target_device(device)
        self.reader = reader
        self.num_workers = num_workers
        self.target_device = target_device

        self.skip_validation = skip_validation
        self._field_levels: dict[str, str] = getattr(reader, "field_levels", {}) or {}

        # Prefetch state
        self._prefetch_futures: dict[int, Future[_PrefetchResult]] = {}
        self._fused_batch_prefetch_queue: deque[_PendingFusedBatch] = deque()
        self._executor: ThreadPoolExecutor | None = None

        # Per-sample transform pipeline (None when no transforms configured so
        # the hot path short-circuits with a single is-None check).
        self._sample_transform: Compose | None = (
            Compose(transforms) if transforms else None
        )

    @staticmethod
    def _resolve_target_device(
        device: str | torch.device | None,
    ) -> torch.device:
        """Resolve the target device while preserving nvalchemi defaults.

        Parameters
        ----------
        device : str | torch.device | None
            Requested device. ``None`` and ``"auto"`` select CUDA when
            available, otherwise CPU.

        Returns
        -------
        torch.device
            Resolved target device.

        Raises
        ------
        TypeError
            If *device* is not a string, ``torch.device``, or ``None``.
        """
        if device is None or device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        elif not isinstance(device, (str, torch.device)):
            raise TypeError(
                "Device expected to be a string or instance of `torch.device`."
                f" Got {device}."
            )
        return torch.device(device)

    def _ensure_executor(self) -> ThreadPoolExecutor:
        """Lazily create the thread pool executor.

        Returns
        -------
        ThreadPoolExecutor
            The executor for async prefetching.
        """
        if self._executor is None:
            self._executor = ThreadPoolExecutor(
                max_workers=self.num_workers,
                thread_name_prefix="datapipe_prefetch",
            )
        return self._executor

    def _read_raw_samples(
        self, indices: Sequence[int]
    ) -> list[tuple[dict[str, torch.Tensor], dict[str, Any]]]:
        """Read raw samples from the underlying reader."""
        if hasattr(self.reader, "read_many"):
            return self.reader.read_many(indices)  # type: ignore[attr-defined]
        return [
            (
                self.reader._load_sample(index),  # type: ignore[attr-defined]
                self.reader._get_sample_metadata(index),  # type: ignore[attr-defined]
            )
            for index in indices
        ]

    def _to_atomic_samples(
        self,
        raw_samples: Sequence[tuple[dict[str, torch.Tensor], dict[str, Any]]],
        stream: torch.cuda.Stream | None = None,
    ) -> tuple[list[tuple[AtomicData, dict[str, Any]]], torch.cuda.Event | None]:
        """Validate raw samples and transfer them to the target device."""
        samples = [
            (AtomicData.model_validate(data_dict), metadata)
            for data_dict, metadata in raw_samples
        ]

        event: torch.cuda.Event | None = None
        if stream is not None:
            with torch.cuda.stream(stream):
                if self.target_device is not None:
                    samples = [
                        (data.to(self.target_device, non_blocking=True), metadata)
                        for data, metadata in samples
                    ]
                if self._sample_transform is not None:
                    samples = [
                        self._sample_transform(data, metadata)
                        for data, metadata in samples
                    ]
            event = torch.cuda.Event()
            event.record(stream)
        else:
            samples = [
                self._finalize_on_device(data, metadata) for data, metadata in samples
            ]

        return samples, event

    def _load_and_transform(
        self,
        index: int,
        stream: torch.cuda.Stream | None = None,
    ) -> _PrefetchResult:
        """Load a sample and construct AtomicData.

        Called by worker threads during prefetch operations.

        Parameters
        ----------
        index : int
            Sample index.
        stream : torch.cuda.Stream | None, default=None
            Optional CUDA stream for GPU operations.

        Returns
        -------
        _PrefetchResult
            PrefetchResult with AtomicData, metadata, or error.
        """
        result = _PrefetchResult(index=index)

        try:
            samples, event = self._to_atomic_samples(
                self._read_raw_samples([index]), stream
            )
            result.data = samples[0][0]
            result.metadata = samples[0][1]
            result.event = event

        except Exception as e:
            result.error = e

        return result

    def prefetch(self, index: int, stream: torch.cuda.Stream | None = None) -> None:
        """Submit a sample for async prefetching.

        If the sample is already being prefetched, this is a no-op.

        Parameters
        ----------
        index : int
            Sample index.
        stream : torch.cuda.Stream | None, default=None
            CUDA stream for GPU operations.
        """
        if index in self._prefetch_futures:
            return
        executor = self._ensure_executor()
        self._prefetch_futures[index] = executor.submit(
            self._load_and_transform, index, stream
        )

    def prefetch_batch(
        self, indices: Sequence[int], streams: Sequence[torch.cuda.Stream] | None = None
    ) -> None:
        """Prefetch multiple samples asynchronously.

        Parameters
        ----------
        indices : Sequence[int]
            Sample indices to prefetch.
        streams : Sequence[torch.cuda.Stream] | None, default=None
            CUDA streams to distribute across. Streams are assigned
            round-robin to the indices.
        """
        for i, idx in enumerate(indices):
            stream = streams[i % len(streams)] if streams else None
            self.prefetch(idx, stream=stream)

    def prefetch_many(
        self, indices: Sequence[int], stream: torch.cuda.Stream | None = None
    ) -> None:
        """Submit one batch of sample indices as a fused async prefetch.

        Parameters
        ----------
        indices : Sequence[int]
            Sample indices to prefetch as one batch.
        stream : torch.cuda.Stream | None, default=None
            CUDA stream for GPU operations.
        """
        self.prefetch_fused_batches([indices], stream=stream)

    def _load_fused_batches(
        self,
        batch_index_lists: Sequence[Sequence[int]],
        stream: torch.cuda.Stream | None = None,
    ) -> _FusedBatchPrefetchResult:
        """Load multiple batches in one fused read_many call.

        When ``self.skip_validation`` is ``True``, returns raw tensor
        dicts (no ``AtomicData`` construction).  Otherwise validates
        each sample through ``AtomicData.model_validate``.

        Parameters
        ----------
        batch_index_lists : Sequence[Sequence[int]]
            Per-batch index lists to concatenate and read together.
        stream : torch.cuda.Stream | None, default=None
            Optional CUDA stream for GPU operations.

        Returns
        -------
        _FusedBatchPrefetchResult
            Combined result with batch split metadata.
        """
        batch_splits = [len(b) for b in batch_index_lists]
        raw = self.skip_validation
        result = _FusedBatchPrefetchResult(batch_splits=batch_splits, raw=raw)

        try:
            all_indices: list[int] = []
            for batch_indices in batch_index_lists:
                all_indices.extend(batch_indices)

            raw_samples = self._read_raw_samples(all_indices)

            if raw:
                raw_dicts = [tensor_dict for tensor_dict, _ in raw_samples]
                result.data = raw_dicts
                result.event = None
            else:
                samples, event = self._to_atomic_samples(raw_samples, stream)
                result.data = [atomic_data for atomic_data, _ in samples]
                result.metadata = [metadata for _, metadata in samples]
                result.event = event
        except Exception as e:
            result.error = e

        return result

    def prefetch_fused_batches(
        self,
        batch_index_lists: Sequence[Sequence[int]],
        stream: torch.cuda.Stream | None = None,
    ) -> None:
        """Submit multiple batches as one fused async read.

        All indices across the provided batch lists are concatenated
        into a single ``read_many`` call, amortizing Zarr I/O overhead.
        Use :meth:`get_fused_batches` to consume the results.

        Parameters
        ----------
        batch_index_lists : Sequence[Sequence[int]]
            Per-batch index lists.
        stream : torch.cuda.Stream | None, default=None
            CUDA stream for GPU operations.
        """
        if len(self._fused_batch_prefetch_queue) >= 2:
            raise RuntimeError(
                "Fused batch prefetch queue is full; consume a pending chunk first."
            )
        frozen_batch_index_lists = tuple(
            tuple(indices) for indices in batch_index_lists
        )
        executor = self._ensure_executor()
        self._fused_batch_prefetch_queue.append(
            _PendingFusedBatch(
                batch_index_lists=frozen_batch_index_lists,
                future=executor.submit(
                    self._load_fused_batches, frozen_batch_index_lists, stream
                ),
            )
        )

    def _fused_result_to_batches(
        self, result: _FusedBatchPrefetchResult
    ) -> list[Batch]:
        """Convert a fused prefetch result into per-batch objects."""
        if result.error is not None:
            raise result.error
        if result.event is not None:
            result.event.synchronize()
        if result.data is None:
            raise RuntimeError("Fused batch prefetch returned None data without error")

        batches: list[Batch] = []
        offset = 0
        for size in result.batch_splits:
            batch_slice = result.data[offset : offset + size]
            offset += size
            if result.raw:
                batches.append(
                    Batch.from_raw_dicts(
                        batch_slice,
                        device=self.target_device,
                        field_levels=self._field_levels,
                    )
                )
            else:
                batches.append(
                    Batch.from_data_list(
                        batch_slice,
                        skip_validation=True,
                        field_levels=self._field_levels,
                    )
                )
        return batches

    def load_batches(
        self,
        batch_index_lists: Sequence[Sequence[int]],
        stream: torch.cuda.Stream | None = None,
    ) -> list[Batch]:
        """Load several batches immediately.

        This is the synchronous counterpart to
        :meth:`prefetch_fused_batches`/:meth:`get_fused_batches`. The provided
        batch index lists are read through one fused reader request so backends
        can coalesce I/O while returning one :class:`Batch` per input list.

        Parameters
        ----------
        batch_index_lists : Sequence[Sequence[int]]
            Per-batch sample indices.
        stream : torch.cuda.Stream | None, default=None
            CUDA stream for device transfer when supported.

        Returns
        -------
        list[Batch]
            One :class:`Batch` per input batch-index list.
        """
        return self._fused_result_to_batches(
            self._load_fused_batches(batch_index_lists, stream)
        )

    def has_pending_fused_batches(self) -> bool:
        """Return whether a fused prefetch chunk is waiting to be consumed."""
        return bool(self._fused_batch_prefetch_queue)

    def get_fused_batches(self) -> Iterator[Batch]:
        """Consume the pending fused prefetch and yield per-batch results.

        Blocks until the fused read completes, then splits the flat
        result list according to the original batch sizes and yields
        one :class:`~nvalchemi.data.batch.Batch` per sub-batch.

        Yields
        ------
        Batch
            One batch per sub-batch from the fused read.

        Raises
        ------
        RuntimeError
            If no fused prefetch is pending.
        Exception
            If the background read failed, re-raises the original error.
        """
        if not self._fused_batch_prefetch_queue:
            raise RuntimeError(
                "No fused batch prefetch pending; call prefetch_fused_batches() "
                "before get_fused_batches()."
            )
        pending = self._fused_batch_prefetch_queue.popleft()

        yield from self._fused_result_to_batches(pending.future.result())

    def cancel_prefetch(self, index: int | None = None) -> None:
        """Cancel pending prefetch operations.

        Parameters
        ----------
        index : int | None, default=None
            Specific index to cancel, or None to cancel all.
        """
        if index is None:
            self._prefetch_futures.clear()
            self._fused_batch_prefetch_queue.clear()
        else:
            self._prefetch_futures.pop(index, None)

    def __getitem__(self, index: int) -> tuple[AtomicData, dict[str, Any]]:
        """Get an AtomicData sample by index.

        If the index was prefetched, returns the prefetched result
        (waiting for completion if necessary). Otherwise loads synchronously.

        Parameters
        ----------
        index : int
            Sample index.

        Returns
        -------
        tuple[AtomicData, dict[str, Any]]
            Tuple of (AtomicData with loaded data, metadata dict).

        Raises
        ------
        IndexError
            If index is out of range.
        RuntimeError
            Raised when a configured transform fails; the original
            exception is chained via ``__cause__``. See
            :class:`~nvalchemi.data.transforms.Compose`.
        Exception
            If prefetch failed, re-raises the original error.
        """
        # Check if prefetched
        future = self._prefetch_futures.pop(index, None)

        if future is not None:
            # Wait for prefetch to complete
            result = future.result()

            if result.error is not None:
                raise result.error

            # Sync stream if needed
            if result.event is not None:
                result.event.synchronize()

            # Data and metadata are guaranteed to be set when error is None
            if result.data is None or result.metadata is None:
                raise RuntimeError(
                    f"Prefetch for index {index} returned None data/metadata without error"
                )
            return result.data, result.metadata

        # Not prefetched, load synchronously through the reader batch path.
        raw_samples = self._read_raw_samples([index])
        samples, _ = self._to_atomic_samples(raw_samples)
        return samples[0]

    def read_many(
        self, indices: Sequence[int]
    ) -> list[tuple[AtomicData, dict[str, Any]]]:
        """Read and validate multiple samples in one dataset request.

        Parameters
        ----------
        indices : Sequence[int]
            Sample indices to load in order.

        Returns
        -------
        list[tuple[AtomicData, dict[str, Any]]]
            Ordered ``(AtomicData, metadata)`` pairs.
        """
        raw_samples = self._read_raw_samples(indices)
        samples, _ = self._to_atomic_samples(raw_samples)
        return samples

    def get_batch(self, indices: Sequence[int]) -> Batch:
        """Read sample indices and return a validated :class:`Batch`.

        Parameters
        ----------
        indices : Sequence[int]
            Sample indices to batch in order.

        Returns
        -------
        Batch
            Batched AtomicData as a disjoint graph.
        """
        key = (tuple(indices),)
        if (
            self._fused_batch_prefetch_queue
            and self._fused_batch_prefetch_queue[0].batch_index_lists == key
        ):
            pending = self._fused_batch_prefetch_queue.popleft()
            batches = self._fused_result_to_batches(pending.future.result())
            if len(batches) != 1:
                raise RuntimeError(
                    f"Prefetch for indices {key[0]} returned {len(batches)} batches"
                )
            return batches[0]

        return self.load_batches([indices])[0]

    def _finalize_on_device(
        self, data: AtomicData, metadata: dict[str, Any]
    ) -> tuple[AtomicData, dict[str, Any]]:
        """Move ``data`` to ``target_device`` and apply the transform pipeline.

        Shared by the prefetch worker path (both stream and non-stream
        branches) and the synchronous ``__getitem__`` fallback. When
        ``self._sample_transform`` is ``None`` the transform step is
        skipped, making the no-transforms hot path a single
        ``is None`` check past the device transfer.

        Parameters
        ----------
        data : AtomicData
            Freshly constructed sample on the reader's (CPU) device.
        metadata : dict[str, Any]
            Per-sample metadata dict.

        Returns
        -------
        tuple[AtomicData, dict[str, Any]]
            The (possibly transformed) pair, ready to return to the caller.
        """
        if self.target_device is not None:
            data = data.to(self.target_device, non_blocking=True)
        if self._sample_transform is not None:
            data, metadata = self._sample_transform(data, metadata)
        return data, metadata

    def __len__(self) -> int:
        """Return the number of samples in the dataset.

        Returns
        -------
        int
            Number of samples, delegated to the reader.
        """
        return len(self.reader)

    @property
    def pin_memory(self) -> bool:
        """Whether the underlying reader should return pinned CPU tensors."""
        return bool(getattr(self.reader, "pin_memory", False))

    @pin_memory.setter
    def pin_memory(self, enabled: bool) -> None:
        """Request pinned-memory reads from the underlying reader.

        Parameters
        ----------
        enabled : bool
            Whether reader outputs should be page-locked.
        """
        if hasattr(self.reader, "pin_memory"):
            self.reader.pin_memory = enabled

    @property
    def prefetch_count(self) -> int:
        """Return the number of pending prefetch requests.

        Returns
        -------
        int
            Count of queued single-sample and fused-batch prefetches.
        """
        return len(self._prefetch_futures) + len(self._fused_batch_prefetch_queue)

    @property
    def field_names(self) -> list[str]:
        """Return field names available in reader samples.

        Returns
        -------
        list[str]
            Field names exposed by the backing reader.
        """
        field_names = getattr(self.reader, "field_names", None)
        if field_names is not None:
            return list(field_names)

        if len(self) == 0:
            return []
        raw_samples = self._read_raw_samples([0])
        if not raw_samples:
            return []
        data_dict, _metadata = raw_samples[0]
        return list(data_dict)

    def get_metadata(self, index: int) -> tuple[int, int]:
        """Return lightweight metadata for a sample without full construction.

        Delegates to the reader when it provides lightweight metadata;
        otherwise loads the raw tensor dictionary and extracts shape
        information for atom and edge counts, avoiding the overhead of full
        ``AtomicData`` construction and validation.

        Parameters
        ----------
        index : int
            Sample index.

        Returns
        -------
        tuple[int, int]
            ``(num_atoms, num_edges)`` for the sample.

        Raises
        ------
        IndexError
            If index is out of range.
        KeyError
            If the sample dict does not contain ``"atomic_numbers"``.
        """
        if hasattr(self.reader, "get_metadata"):
            return self.reader.get_metadata(index)  # type: ignore[attr-defined]

        data_dict, _metadata = self._read_raw_samples([index])[0]
        num_atoms = len(data_dict["atomic_numbers"])
        num_edges = 0
        if "neighbor_list" in data_dict and data_dict["neighbor_list"] is not None:
            num_edges = data_dict["neighbor_list"].shape[0]
        return num_atoms, num_edges

    def __iter__(self) -> Iterator[tuple[AtomicData, dict[str, Any]]]:
        """Iterate over all samples in the dataset.

        Yields
        ------
        tuple[AtomicData, dict[str, Any]]
            ``(AtomicData, metadata)`` for each sample.
        """
        for i in range(len(self)):
            yield self[i]

    def close(self) -> None:
        """Release resources held by the dataset.

        Drains pending prefetch futures, shuts down the thread pool
        executor, and closes the underlying reader.
        """
        # Drain pending futures
        futures_to_drain: list[Future] = [
            *self._prefetch_futures.values(),
            *[pending.future for pending in self._fused_batch_prefetch_queue],
        ]
        for future in futures_to_drain:
            try:
                future.result(timeout=1.0)
            except Exception:
                logger.debug("Ignoring error during prefetch future cleanup")
        self._prefetch_futures.clear()
        self._fused_batch_prefetch_queue.clear()

        # Shutdown executor
        if self._executor is not None:
            self._executor.shutdown(wait=False)
            self._executor = None

        # Close reader
        self.reader.close()

    def __enter__(self) -> Dataset:
        """Enter context manager.

        Returns
        -------
        Dataset
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

    def __repr__(self) -> str:
        """Return a string representation of the dataset.

        Returns
        -------
        str
            Human-readable summary including length and device.
        """
        return (
            f"{self.__class__.__name__}("
            f"len={len(self)}, "
            f"device={self.target_device}, "
            f"num_workers={self.num_workers})"
        )
