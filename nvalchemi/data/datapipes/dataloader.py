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
to develop highly performant data loading and preprocessing
workflows. An optional ``batch_transforms`` hook applies
user-supplied callables to each collated :class:`Batch` on the
consumer thread.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterator, Sequence

import torch
from torch.utils.data import RandomSampler, Sampler, SequentialSampler

from nvalchemi._typing import BatchTransform
from nvalchemi.data.batch import Batch
from nvalchemi.data.datapipes.dataset import Dataset
from nvalchemi.data.transforms import Compose


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
    batch_transforms : Sequence[BatchTransform] | None, default=None
        Optional per-batch transforms applied to each yielded
        :class:`~nvalchemi.data.batch.Batch` after collation. ``None``
        or an empty sequence disables the hook (zero runtime overhead
        on the hot path). See the Notes section for thread placement
        and CUDA-stream semantics. For per-sample transforms applied
        before collation, see :class:`Dataset` (``transforms`` parameter).

    Attributes
    ----------
    dataset : Dataset
        The underlying dataset.
    batch_size : int
        Number of samples per batch.
    sampler : torch.utils.data.Sampler
        Resolved sampler (``RandomSampler`` if ``shuffle=True``, else
        :class:`~torch.utils.data.SequentialSampler`; user-supplied
        ``sampler`` overrides both).
    drop_last : bool
        Whether the trailing partial batch is dropped.
    prefetch_factor : int
        Configured prefetch depth (see :meth:`__iter__`).
    num_streams : int
        Configured CUDA-stream pool size for prefetching.
    use_streams : bool
        Whether stream-based prefetching is actually enabled. Stored as
        ``use_streams and torch.cuda.is_available()``; reflects runtime
        availability, not the raw argument.

    Raises
    ------
    ValueError
        Raised at construction if ``batch_size < 1``.
    TypeError
        Raised at construction if ``batch_transforms`` is not a
        :class:`~collections.abc.Sequence` (e.g. a single callable or a
        generator was passed).
    RuntimeError
        Raised during iteration (not construction) when any batch
        transform fails; the original exception is chained via
        ``__cause__``.

    Notes
    -----
    Batch transforms run on the consumer (main) thread after
    collation, not on the prefetch workers — the fully-assembled
    ``Batch`` does not exist until the main thread constructs it.
    Transforms are applied in order via
    :class:`~nvalchemi.data.transforms.Compose` and execute on the
    current CUDA stream at yield time; wrap iteration in your own
    ``torch.cuda.stream(...)`` context to control placement.

    Examples
    --------
    >>> from nvalchemi.data.datapipes import AtomicDataZarrReader, Dataset, DataLoader
    >>> reader = AtomicDataZarrReader("dataset.zarr")  # doctest: +SKIP
    >>> ds = Dataset(reader, device="cpu")              # doctest: +SKIP
    >>> def center_positions(batch):                    # doctest: +SKIP
    ...     batch.positions = batch.positions - batch.positions.mean(0)
    ...     return batch
    >>> loader = DataLoader(ds, batch_size=4, batch_transforms=[center_positions])  # doctest: +SKIP
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
        batch_transforms: Sequence[BatchTransform] | None = None,
    ) -> None:
        """Initialize the AtomicData-native DataLoader."""
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")

        if batch_transforms is not None and not isinstance(batch_transforms, Sequence):
            raise TypeError(
                "batch_transforms must be a Sequence of callables, not a "
                "single callable or generator. Pass [fn] instead of fn."
            )

        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.prefetch_factor = prefetch_factor
        self.num_streams = num_streams
        self.use_streams = use_streams and torch.cuda.is_available()
        self._batch_transform: Compose | None = (
            Compose(batch_transforms) if batch_transforms else None
        )

        # Handle sampler
        if sampler is not None:
            self.sampler = sampler
        elif shuffle:
            self.sampler = RandomSampler(dataset)
        else:
            self.sampler = SequentialSampler(dataset)

        self._streams: list[torch.cuda.Stream] = (
            [torch.cuda.Stream() for _ in range(num_streams)]
            if self.use_streams
            else []
        )

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
        transform = self._batch_transform
        for batch_indices in self._generate_batches():
            samples = [self.dataset[idx] for idx in batch_indices]
            # Extract AtomicData from (AtomicData, metadata) tuples
            data_list = [atomic_data for atomic_data, _ in samples]
            batch = Batch.from_data_list(data_list, skip_validation=True)
            if transform is not None:
                batch = transform(batch)
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
        transform = self._batch_transform

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
                batch = Batch.from_data_list(data_list, skip_validation=True)
                if transform is not None:
                    batch = transform(batch)
                yield batch

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
