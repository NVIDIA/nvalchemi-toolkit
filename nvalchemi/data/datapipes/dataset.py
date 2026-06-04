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
from collections.abc import Iterator, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import torch

from nvalchemi.data.atomic_data import AtomicData
from nvalchemi.data.batch import Batch
from nvalchemi.data.datapipes.backends.base import Reader

logger = logging.getLogger(__name__)

# TODO: refactor to subclass PNM when stable


@runtime_checkable
class ReaderProtocol(Protocol):
    """Protocol for reader objects compatible with Dataset.

    This protocol enables duck-typed Reader implementations to be used
    with :class:`Dataset` without inheriting from the
    :class:`~nvalchemi.data.datapipes.backends.base.Reader` ABC.
    """

    def _load_sample(self, index: int) -> dict[str, torch.Tensor]:
        """Load raw tensor data for a single sample."""
        ...

    def _get_sample_metadata(self, index: int) -> dict[str, Any]:
        """Return additional metadata for a sample."""
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
class _PrefetchBatchResult:
    """Container for async batch prefetch results.

    Attributes
    ----------
    indices : tuple[int, ...]
        Sample indices that were loaded.
    data : list[AtomicData] | None
        Loaded data in requested order, or None if an error occurred.
    metadata : list[dict[str, Any]] | None
        Per-sample metadata in requested order, or None.
    error : Exception | None
        Exception if loading failed, or None.
    event : torch.cuda.Event | None
        CUDA event for stream synchronization, or None.
    """

    indices: tuple[int, ...]
    data: list[AtomicData] | None = None
    metadata: list[dict[str, Any]] | None = None
    error: Exception | None = None
    event: torch.cuda.Event | None = None


@dataclass
class _MegaPrefetchResult:
    """Container for amortized multi-batch prefetch results.

    Attributes
    ----------
    batch_splits : list[int]
        Number of samples in each sub-batch, used to split
        the flat result list back into per-batch groups.
    data : list[AtomicData] | None
        All loaded samples in request order, or None on error.
    metadata : list[dict[str, Any]] | None
        Per-sample metadata in request order, or None.
    error : Exception | None
        Exception if loading failed, or None.
    event : torch.cuda.Event | None
        CUDA event for stream synchronization, or None.
    """

    batch_splits: list[int]
    data: list[AtomicData] | None = None
    metadata: list[dict[str, Any]] | None = None
    error: Exception | None = None
    event: torch.cuda.Event | None = None


@dataclass
class _MegaPrefetchRawResult:
    """Container for raw (no-validation) multi-batch prefetch results.

    Attributes
    ----------
    batch_splits : list[int]
        Number of samples in each sub-batch.
    data : list[dict[str, torch.Tensor]] | None
        Raw tensor dicts in request order, or None on error.
    error : Exception | None
        Exception if loading failed, or None.
    event : torch.cuda.Event | None
        CUDA event for stream synchronization, or None.
    """

    batch_splits: list[int]
    data: list[dict[str, torch.Tensor]] | None = None
    error: Exception | None = None
    event: torch.cuda.Event | None = None


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
    """

    def __init__(
        self,
        reader: Reader | ReaderProtocol,
        *,
        device: str | torch.device | None = None,
        num_workers: int = 2,
        skip_validation: bool = False,
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
            validation in the mega-prefetch path, building batches
            directly from raw tensor dicts via
            :meth:`~nvalchemi.data.batch.Batch.from_raw_dicts`.  This
            is safe when the backing store is already validated (e.g.
            data written by :class:`AtomicDataZarrWriter`).

        Raises
        ------
        TypeError
            If reader does not implement the required interface.
        """
        # Validate reader implements the required protocol
        if not isinstance(reader, (Reader, ReaderProtocol)):
            raise TypeError(
                f"reader must implement Reader interface, got {type(reader).__name__}"
            )

        self.reader = reader
        self.num_workers = num_workers
        self.skip_validation = skip_validation

        # Resolve device
        if device is not None:
            if isinstance(device, str):
                device = torch.device(device)
            if not isinstance(device, torch.device):
                raise TypeError(
                    "Device expected to be a string or instance of `torch.device`."
                    f" Got {device}."
                )
            self.target_device = device
        else:
            # fallback
            if torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
            self.target_device = torch.device(device)

        # Prefetch state
        self._prefetch_futures: dict[int, Future[_PrefetchResult]] = {}
        self._batch_prefetch_futures: dict[
            tuple[int, ...], Future[_PrefetchBatchResult]
        ] = {}
        self._mega_prefetch_future: Future[_MegaPrefetchResult] | None = None
        self._executor: ThreadPoolExecutor | None = None

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

        samples: list[tuple[dict[str, torch.Tensor], dict[str, Any]]] = []
        for index in indices:
            data_dict = self.reader._load_sample(index)
            metadata = self.reader._get_sample_metadata(index)
            samples.append((data_dict, metadata))
        return samples

    def _to_atomic_samples(
        self,
        raw_samples: Sequence[tuple[dict[str, torch.Tensor], dict[str, Any]]],
        stream: torch.cuda.Stream | None = None,
    ) -> tuple[list[tuple[AtomicData, dict[str, Any]]], torch.cuda.Event | None]:
        """Validate raw samples and transfer them to the target device."""
        samples: list[tuple[AtomicData, dict[str, Any]]] = []

        for data_dict, metadata in raw_samples:
            samples.append((AtomicData.model_validate(data_dict), metadata))

        event: torch.cuda.Event | None = None
        if self.target_device is not None:
            if stream is not None:
                with torch.cuda.stream(stream):
                    samples = [
                        (data.to(self.target_device, non_blocking=True), metadata)
                        for data, metadata in samples
                    ]
                event = torch.cuda.Event()
                event.record(stream)
            else:
                samples = [
                    (data.to(self.target_device, non_blocking=True), metadata)
                    for data, metadata in samples
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
            data_dict = self.reader._load_sample(index)
            metadata = self.reader._get_sample_metadata(index)
            samples, event = self._to_atomic_samples([(data_dict, metadata)], stream)
            result.data = samples[0][0]
            result.metadata = samples[0][1]
            result.event = event

        except Exception as e:
            result.error = e

        return result

    def _load_many_and_transform(
        self,
        indices: Sequence[int],
        stream: torch.cuda.Stream | None = None,
    ) -> _PrefetchBatchResult:
        """Load multiple samples and construct AtomicData instances.

        Called by worker threads during batch prefetch operations.

        Parameters
        ----------
        indices : Sequence[int]
            Sample indices.
        stream : torch.cuda.Stream | None, default=None
            Optional CUDA stream for GPU operations.

        Returns
        -------
        _PrefetchBatchResult
            Prefetch result with ordered AtomicData, metadata, or error.
        """
        result = _PrefetchBatchResult(indices=tuple(indices))

        try:
            raw_samples = self._read_raw_samples(indices)
            samples, event = self._to_atomic_samples(raw_samples, stream)
            result.data = [atomic_data for atomic_data, _ in samples]
            result.metadata = [metadata for _, metadata in samples]
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
        """Submit multiple samples as one async batch prefetch.

        Parameters
        ----------
        indices : Sequence[int]
            Sample indices to prefetch as a single reader request.
        stream : torch.cuda.Stream | None, default=None
            CUDA stream for GPU operations.
        """
        key = tuple(indices)
        if key in self._batch_prefetch_futures:
            return
        executor = self._ensure_executor()
        self._batch_prefetch_futures[key] = executor.submit(
            self._load_many_and_transform, key, stream
        )

    def _load_mega_and_transform(
        self,
        batch_index_lists: Sequence[Sequence[int]],
        stream: torch.cuda.Stream | None = None,
    ) -> _MegaPrefetchResult:
        """Load multiple batches in one fused read_many call.

        Parameters
        ----------
        batch_index_lists : Sequence[Sequence[int]]
            Per-batch index lists to concatenate and read together.
        stream : torch.cuda.Stream | None, default=None
            Optional CUDA stream for GPU operations.

        Returns
        -------
        _MegaPrefetchResult
            Combined result with batch split metadata.
        """
        batch_splits = [len(b) for b in batch_index_lists]
        result = _MegaPrefetchResult(batch_splits=batch_splits)

        try:
            all_indices: list[int] = []
            for batch_indices in batch_index_lists:
                all_indices.extend(batch_indices)

            raw_samples = self._read_raw_samples(all_indices)
            samples, event = self._to_atomic_samples(raw_samples, stream)
            result.data = [atomic_data for atomic_data, _ in samples]
            result.metadata = [metadata for _, metadata in samples]
            result.event = event
        except Exception as e:
            result.error = e

        return result

    def _load_mega_raw(
        self,
        batch_index_lists: Sequence[Sequence[int]],
        stream: torch.cuda.Stream | None = None,
    ) -> _MegaPrefetchRawResult:
        """Load multiple batches as raw dicts without AtomicData validation.

        Parameters
        ----------
        batch_index_lists : Sequence[Sequence[int]]
            Per-batch index lists to concatenate and read together.
        stream : torch.cuda.Stream | None, default=None
            Optional CUDA stream for GPU operations.

        Returns
        -------
        _MegaPrefetchRawResult
            Combined result with batch split metadata.
        """
        batch_splits = [len(b) for b in batch_index_lists]
        result = _MegaPrefetchRawResult(batch_splits=batch_splits)

        try:
            all_indices: list[int] = []
            for batch_indices in batch_index_lists:
                all_indices.extend(batch_indices)

            raw_samples = self._read_raw_samples(all_indices)
            # raw_samples is list[(dict[str, Tensor], metadata_dict)]
            # Extract just the tensor dicts, skip metadata.
            raw_dicts = [tensor_dict for tensor_dict, _ in raw_samples]

            event: torch.cuda.Event | None = None
            if self.target_device is not None and stream is not None:
                with torch.cuda.stream(stream):
                    raw_dicts = [
                        {k: v.to(self.target_device, non_blocking=True) for k, v in d.items()}
                        for d in raw_dicts
                    ]
                event = torch.cuda.Event()
                event.record(stream)
            elif self.target_device is not None:
                raw_dicts = [
                    {k: v.to(self.target_device, non_blocking=True) for k, v in d.items()}
                    for d in raw_dicts
                ]

            result.data = raw_dicts
            result.event = event
        except Exception as e:
            result.error = e

        return result

    def prefetch_mega(
        self,
        batch_index_lists: Sequence[Sequence[int]],
        stream: torch.cuda.Stream | None = None,
    ) -> None:
        """Submit multiple batches as one fused async read.

        All indices across the provided batch lists are concatenated
        into a single ``read_many`` call, amortizing Zarr I/O overhead.
        Use :meth:`get_mega_batches` to consume the results.

        Parameters
        ----------
        batch_index_lists : Sequence[Sequence[int]]
            Per-batch index lists.
        stream : torch.cuda.Stream | None, default=None
            CUDA stream for GPU operations.
        """
        if self._mega_prefetch_future is not None:
            return
        executor = self._ensure_executor()
        load_fn = (
            self._load_mega_raw
            if self.skip_validation
            else self._load_mega_and_transform
        )
        self._mega_prefetch_future = executor.submit(
            load_fn, batch_index_lists, stream
        )

    def get_mega_batches(self) -> Iterator[Batch]:
        """Consume the pending mega-prefetch and yield per-batch results.

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
            If no mega-prefetch is pending.
        Exception
            If the background read failed, re-raises the original error.
        """
        future = self._mega_prefetch_future
        if future is None:
            raise RuntimeError("No mega-prefetch pending.")
        self._mega_prefetch_future = None

        result = future.result()
        if result.error is not None:
            raise result.error
        if result.event is not None:
            result.event.synchronize()

        if isinstance(result, _MegaPrefetchRawResult):
            if result.data is None:
                raise RuntimeError(
                    "Mega-prefetch returned None data without error"
                )
            offset = 0
            for size in result.batch_splits:
                batch_dicts = result.data[offset : offset + size]
                offset += size
                yield Batch.from_raw_dicts(batch_dicts)
        else:
            if result.data is None or result.metadata is None:
                raise RuntimeError(
                    "Mega-prefetch returned None data/metadata without error"
                )
            offset = 0
            for size in result.batch_splits:
                batch_data = result.data[offset : offset + size]
                offset += size
                yield Batch.from_data_list(batch_data, skip_validation=True)

    def cancel_prefetch(self, index: int | None = None) -> None:
        """Cancel pending prefetch operations.

        Parameters
        ----------
        index : int | None, default=None
            Specific index to cancel, or None to cancel all.
        """
        if index is None:
            self._prefetch_futures.clear()
            self._batch_prefetch_futures.clear()
            self._mega_prefetch_future = None
        else:
            self._prefetch_futures.pop(index, None)
            for key in list(self._batch_prefetch_futures):
                if index in key:
                    self._batch_prefetch_futures.pop(key, None)

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

        # Not prefetched, load synchronously through the batch-read path.
        return self.read_many([index])[0]

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

        Raises
        ------
        Exception
            If a queued batch prefetch failed, re-raises the original error.
        """
        key = tuple(indices)
        future = self._batch_prefetch_futures.pop(key, None)

        if future is not None:
            result = future.result()
            if result.error is not None:
                raise result.error
            if result.event is not None:
                result.event.synchronize()
            if result.data is None or result.metadata is None:
                raise RuntimeError(
                    f"Prefetch for indices {key} returned None data/metadata without error"
                )
            return Batch.from_data_list(result.data, skip_validation=True)

        if self.skip_validation:
            raw_samples = self._read_raw_samples(indices)
            raw_dicts = [tensor_dict for tensor_dict, _ in raw_samples]
            if self.target_device is not None:
                raw_dicts = [
                    {k: v.to(self.target_device) for k, v in d.items()}
                    for d in raw_dicts
                ]
            return Batch.from_raw_dicts(raw_dicts)

        samples = self.read_many(indices)
        data_list = [atomic_data for atomic_data, _ in samples]
        return Batch.from_data_list(data_list, skip_validation=True)

    def __len__(self) -> int:
        """Return the number of samples in the dataset.

        Returns
        -------
        int
            Number of samples, delegated to the reader.
        """
        return len(self.reader)

    def get_metadata(self, index: int) -> tuple[int, int]:
        """Return lightweight metadata for a sample without full construction.

        Loads the raw tensor dictionary from the reader and extracts shape
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
        data_dict = self.reader._load_sample(index)
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
            *self._batch_prefetch_futures.values(),
        ]
        if self._mega_prefetch_future is not None:
            futures_to_drain.append(self._mega_prefetch_future)
        for future in futures_to_drain:
            try:
                future.result(timeout=1.0)
            except Exception:
                logger.debug("Ignoring error during prefetch future cleanup")
        self._prefetch_futures.clear()
        self._batch_prefetch_futures.clear()
        self._mega_prefetch_future = None

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
