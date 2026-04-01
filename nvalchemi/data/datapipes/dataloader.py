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
"""AtomicData-native DataLoader with CUDA-stream prefetching.

The ``DataLoader`` class is designed to be a drop-in replacement
for ``torch.data.DataLoader``, specializing for ``nvalchemi``
and atomistic systems by emitting ``Batch`` data.

Additionally, the ``DataLoader`` provides two mechanisms for
performant data loading: an asynchronous prefetching mechanism,
as well as the use of CUDA streams; both of which can be used
to developer highly performance data loading and preprocessing
workflows.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterator

import torch
from torch.utils.data import RandomSampler, Sampler, SequentialSampler

from nvalchemi.data.batch import Batch
from nvalchemi.data.datapipes.dataset import Dataset


class DataLoader:
    """Batch-iterating data loader that yields :class:`~nvalchemi.data.batch.Batch`.

    Wraps a :class:`Dataset` and yields ``Batch`` objects
    built via :meth:`Batch.from_data_list`.  CUDA-stream prefetching is
    supported for overlapping I/O with computation.

    Parameters
    ----------
    dataset : Dataset
        AtomicData-native dataset to load from.
    batch_size : int, default=1
        Number of samples per batch.
    shuffle : bool, default=False
        Randomize sample order each epoch.
    drop_last : bool, default=False
        Drop the last incomplete batch.
    sampler : torch.utils.data.Sampler | None, default=None
        Custom sampler (overrides ``shuffle``).
    prefetch_factor : int, default=2
        How many batches to prefetch ahead.
    num_streams : int, default=4
        Number of CUDA streams for prefetching.
    use_streams : bool, default=True
        Enable CUDA-stream prefetching.

    Examples
    --------
    >>> from nvalchemi.data.datapipes import AtomicDataZarrReader, Dataset, DataLoader
    >>> reader = AtomicDataZarrReader("dataset.zarr")  # doctest: +SKIP
    >>> ds = Dataset(reader, device="cpu")              # doctest: +SKIP
    >>> loader = DataLoader(ds, batch_size=4)           # doctest: +SKIP
    >>> for batch in loader:                            # doctest: +SKIP
    ...     print(batch.positions.shape)
    """

    def __init__(
        self,
        dataset: Dataset,
        *,
        batch_size: int = 1,
        shuffle: bool = False,
        drop_last: bool = False,
        sampler: Sampler | None = None,
        prefetch_factor: int = 2,
        num_streams: int = 4,
        use_streams: bool = True,
    ) -> None:
        """Initialize the AtomicData-native DataLoader.

        Parameters
        ----------
        dataset : Dataset
            AtomicData-native dataset to load from.
        batch_size : int, default=1
            Number of samples per batch.
        shuffle : bool, default=False
            Randomize sample order each epoch.
        drop_last : bool, default=False
            Drop the last incomplete batch.
        sampler : torch.utils.data.Sampler | None, default=None
            Custom sampler (overrides ``shuffle``).
        prefetch_factor : int, default=2
            How many batches to prefetch ahead.
        num_streams : int, default=4
            Number of CUDA streams for prefetching.
        use_streams : bool, default=True
            Enable CUDA-stream prefetching.

        Raises
        ------
        ValueError
            If batch_size < 1.
        """
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")

        # Set up attributes directly (standalone class)
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.prefetch_factor = prefetch_factor
        self.num_streams = num_streams
        self.use_streams = use_streams and torch.cuda.is_available()

        # Handle sampler
        if sampler is not None:
            self.sampler = sampler
        elif shuffle:
            self.sampler = RandomSampler(dataset)
        else:
            self.sampler = SequentialSampler(dataset)

        # Create CUDA streams for prefetching
        self._streams: list[torch.cuda.Stream] = []
        if self.use_streams:
            for _ in range(num_streams):
                self._streams.append(torch.cuda.Stream())

    def __len__(self) -> int:
        """Return the number of batches.

        Returns
        -------
        int
            Number of batches in the dataloader.
        """
        n_samples = len(self.dataset)
        if self.drop_last:
            return n_samples // self.batch_size
        return (n_samples + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> Iterator[Batch]:
        """Iterate over batches.

        Uses stream-based prefetching when enabled to overlap IO,
        GPU transfers, and computation.

        Yields
        ------
        Batch
            Batched AtomicData as a disjoint graph.
        """
        if self.prefetch_factor > 0 and self.use_streams:
            yield from self._iter_prefetch()
        else:
            yield from self._iter_simple()

    def _generate_batches(self) -> Iterator[list[int]]:
        """Generate batches of indices.

        Yields
        ------
        list[int]
            List of sample indices for each batch.
        """
        batch: list[int] = []
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []

        if batch and not self.drop_last:
            yield batch

    def _iter_simple(self) -> Iterator[Batch]:
        """Simple synchronous iteration without prefetching.

        Yields
        ------
        Batch
            Collated batch of AtomicData.
        """
        for batch_indices in self._generate_batches():
            samples = [self.dataset[idx] for idx in batch_indices]
            # Extract AtomicData from (AtomicData, metadata) tuples
            data_list = [atomic_data for atomic_data, _ in samples]
            batch = Batch.from_data_list(data_list, skip_validation=True)
            yield batch

    def _iter_prefetch(self) -> Iterator[Batch]:
        """Iteration with stream-based prefetching.

        Uses a lazy sliding window of size ``prefetch_factor`` over the
        batch-index generator so that the full epoch plan is never
        materialised in memory.

        Strategy:

        1. Fill a window of up to ``prefetch_factor`` batches, submitting
           each for async prefetch.
        2. Pop the front batch, yield it, then pull one more batch from
           the generator and prefetch it (keeping the window full).
        3. Cleanup runs in a ``finally`` block so that
           ``cancel_prefetch()`` fires on normal exhaustion, early break,
           and exceptions.

        Yields
        ------
        Batch
            Collated batch of AtomicData.
        """
        stream_idx = 0

        def _prefetch_batch(batch_indices: list[int]) -> None:
            nonlocal stream_idx
            for sample_idx in batch_indices:
                stream = self._streams[stream_idx % self.num_streams]
                self.dataset.prefetch(sample_idx, stream=stream)
                stream_idx += 1

        batch_iter = self._generate_batches()
        window: deque[list[int]] = deque()

        try:
            for _ in range(self.prefetch_factor):
                batch_indices = next(batch_iter, None)
                if batch_indices is None:
                    break
                window.append(batch_indices)
                _prefetch_batch(batch_indices)

            while window:
                batch_indices = window.popleft()
                samples = [self.dataset[idx] for idx in batch_indices]
                data_list = [atomic_data for atomic_data, _ in samples]
                yield Batch.from_data_list(data_list, skip_validation=True)

                next_batch = next(batch_iter, None)
                if next_batch is not None:
                    window.append(next_batch)
                    _prefetch_batch(next_batch)
        finally:
            self.dataset.cancel_prefetch()

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch for the sampler (used in distributed training).

        Parameters
        ----------
        epoch : int
            Current epoch number.
        """
        if hasattr(self.sampler, "set_epoch"):
            self.sampler.set_epoch(epoch)
