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
            # Load raw dict from reader (CPU, potentially slow IO)
            data_dict = self.reader._load_sample(index)
            metadata = self.reader._get_sample_metadata(index)

            # Construct AtomicData directly from dict
            data = AtomicData.model_validate(data_dict)

            # Auto-transfer to target device if specified
            if self.target_device is not None:
                if stream is not None:
                    with torch.cuda.stream(stream):
                        data = data.to(self.target_device, non_blocking=True)
                    # Record event for synchronization
                    result.event = torch.cuda.Event()
                    result.event.record(stream)
                else:
                    data = data.to(self.target_device, non_blocking=True)

            result.data = data
            result.metadata = metadata

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

    def cancel_prefetch(self, index: int | None = None) -> None:
        """Cancel pending prefetch operations.

        Parameters
        ----------
        index : int | None, default=None
            Specific index to cancel, or None to cancel all.
        """
        if index is None:
            self._prefetch_futures.clear()
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

        # Not prefetched, load synchronously
        data_dict = self.reader._load_sample(index)
        metadata = self.reader._get_sample_metadata(index)

        # Construct AtomicData directly from dict
        data = AtomicData.model_validate(data_dict)

        # Auto-transfer to target device if specified
        if self.target_device is not None:
            data = data.to(self.target_device, non_blocking=True)

        return data, metadata

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
            If the sample dict does not contain ``"numbers"``.
        """
        data_dict = self.reader._load_sample(index)
        num_atoms = len(data_dict["numbers"])
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
        for future in self._prefetch_futures.values():
            try:
                future.result(timeout=1.0)
            except Exception:
                logger.debug("Ignoring error during prefetch future cleanup")
        self._prefetch_futures.clear()

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
