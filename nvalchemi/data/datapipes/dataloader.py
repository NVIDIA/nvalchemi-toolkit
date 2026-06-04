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

from collections.abc import Iterator, Sequence
from math import ceil

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
    batch_sampler : torch.utils.data.Sampler | None, default=None
        Custom sampler that yields batches of sample indices.
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
        batch_sampler: Sampler[Sequence[int]] | None = None,
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
        batch_sampler : torch.utils.data.Sampler | None, default=None
            Custom sampler that yields batches of sample indices.
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
        if batch_sampler is not None and (sampler is not None or shuffle):
            raise ValueError(
                "batch_sampler is mutually exclusive with sampler and shuffle"
            )

        # Set up attributes directly (standalone class)
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.prefetch_factor = prefetch_factor
        self.num_streams = num_streams
        self.use_streams = use_streams and torch.cuda.is_available()
        self.batch_sampler = batch_sampler

        # Handle sampler
        if self.batch_sampler is None:
            if sampler is not None:
                self.sampler = sampler
            elif shuffle:
                self.sampler = RandomSampler(dataset)
            else:
                self.sampler = SequentialSampler(dataset)
        else:
            self.sampler = None

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
        if self.batch_sampler is not None:
            return len(self.batch_sampler)  # type: ignore[arg-type]

        n_samples = len(self.sampler) if self.sampler is not None else len(self.dataset)
        if self.drop_last:
            return n_samples // self.batch_size
        return ceil(n_samples / self.batch_size)

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
        if self.batch_sampler is not None:
            for batch_indices in self.batch_sampler:
                yield list(batch_indices)
            return

        batch: list[int] = []
        if self.sampler is None:
            return
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
            yield self.dataset.get_batch(batch_indices)

    def _iter_prefetch(self) -> Iterator[Batch]:
        """Iteration with amortized stream-based prefetching.

        Fuses ``prefetch_factor`` consecutive batches into a single
        ``read_many`` call so that the Zarr gap-merge optimisation
        can coalesce scattered indices into fewer large reads.

        Strategy (double-buffered):

        1. Collect up to ``prefetch_factor`` batch-index lists from the
           sampler into a *chunk*.
        2. Submit the chunk as one fused ``prefetch_mega`` on a CUDA
           stream.
        3. Immediately collect the *next* chunk and submit it so the
           background thread can work while batches are yielded.
        4. Yield batches from the completed chunk, then rotate.
        5. Cleanup runs in a ``finally`` block so that
           ``cancel_prefetch()`` fires on normal exhaustion, early break,
           and exceptions.

        Yields
        ------
        Batch
            Collated batch of AtomicData.
        """
        stream_idx = 0
        batch_iter = self._generate_batches()

        def _collect_chunk() -> list[list[int]]:
            """Collect up to prefetch_factor batch-index lists."""
            chunk: list[list[int]] = []
            for _ in range(self.prefetch_factor):
                batch_indices = next(batch_iter, None)
                if batch_indices is None:
                    break
                chunk.append(batch_indices)
            return chunk

        def _submit_chunk(chunk: list[list[int]]) -> None:
            nonlocal stream_idx
            stream = self._streams[stream_idx % self.num_streams] if self._streams else None
            self.dataset.prefetch_mega(chunk, stream=stream)
            stream_idx += 1

        try:
            # Prime: collect and submit first chunk
            pending_chunk = _collect_chunk()
            if not pending_chunk:
                return
            _submit_chunk(pending_chunk)

            while True:
                # Eagerly collect the next chunk indices (CPU-cheap)
                # while the background thread reads the current one.
                next_chunk = _collect_chunk()

                # Block until current mega-read completes, yield batches
                yield from self.dataset.get_mega_batches()

                if not next_chunk:
                    break

                # Submit next chunk now that the slot is free
                _submit_chunk(next_chunk)
                pending_chunk = next_chunk
        finally:
            self.dataset.cancel_prefetch()

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch for the sampler (used in distributed training).

        Parameters
        ----------
        epoch : int
            Current epoch number.
        """
        sampler = self.batch_sampler if self.batch_sampler is not None else self.sampler
        if hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)
