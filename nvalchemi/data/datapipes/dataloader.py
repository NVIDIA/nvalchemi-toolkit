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
"""AtomicData-native DataLoader with amortized prefetching.

The ``DataLoader`` class is designed to be a drop-in replacement
for ``torch.data.DataLoader``, specializing for ``nvalchemi``
and atomistic systems by emitting ``Batch`` data.

Additionally, the ``DataLoader`` can fuse several emitted batches into one
backend read. ``prefetch_factor`` controls that read window, while optional
CUDA streams can overlap device transfers when available. An optional
``batch_transforms`` hook applies user-supplied callables to each collated
:class:`Batch` on the consumer thread.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from math import ceil

import torch
from torch.utils.data import RandomSampler, Sampler, SequentialSampler

from nvalchemi._typing import BatchTransform
from nvalchemi.data.batch import Batch
from nvalchemi.data.datapipes.dataset import Dataset
from nvalchemi.data.transforms import Compose


class DataLoader:
    """Batch-iterating data loader that yields :class:`~nvalchemi.data.batch.Batch`.

    Wraps a :class:`Dataset` and yields ``Batch`` objects
    built via :meth:`Batch.from_data_list`. Fused prefetching is used by
    default to amortize I/O across multiple emitted batches; CUDA streams are
    supported for overlapping device transfers when available.

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
        Number of emitted batches to fuse into each backend read. The effective
        read window is ``batch_size * prefetch_factor``. Set to 0 to disable
        fused prefetching and read one emitted batch at a time.
    num_streams : int, default=4
        Number of CUDA streams for prefetching.
    use_streams : bool, default=True
        Enable CUDA-stream prefetching.
    pin_memory : bool, default=False
        If True, request page-locked CPU tensors from readers that support
        pinned-memory reads.
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
    pin_memory : bool
        Whether page-locked CPU tensors are requested from compatible readers.

    Raises
    ------
    ValueError
        Raised at construction if ``batch_size < 1`` or
        ``prefetch_factor < 0``.
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
    collation, not on the prefetch workers; the fully assembled
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
        batch_sampler: Sampler[Sequence[int]] | None = None,
        prefetch_factor: int = 2,
        num_streams: int = 4,
        use_streams: bool = True,
        pin_memory: bool = False,
        batch_transforms: Sequence[BatchTransform] | None = None,
    ) -> None:
        """Initialize the AtomicData-native DataLoader."""
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        if prefetch_factor < 0:
            raise ValueError(f"prefetch_factor must be >= 0, got {prefetch_factor}")
        if batch_sampler is not None and (sampler is not None or shuffle):
            raise ValueError(
                "batch_sampler is mutually exclusive with sampler and shuffle"
            )

        if batch_transforms is not None and not isinstance(batch_transforms, Sequence):
            raise TypeError(
                "batch_transforms must be a Sequence of callables, not a "
                "single callable or generator. Pass [fn] instead of fn."
            )

        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.prefetch_factor = prefetch_factor
        self.num_streams = num_streams
        self.use_streams = use_streams and torch.cuda.is_available()
        self.batch_sampler = batch_sampler
        self.pin_memory = pin_memory
        self._epoch_step_start = 0

        if pin_memory:
            self._set_pin_memory(self.dataset, True)

        self._batch_transform: Compose | None = (
            Compose(batch_transforms) if batch_transforms else None
        )

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

        self._streams: list[torch.cuda.Stream] = (
            [torch.cuda.Stream() for _ in range(num_streams)]
            if self.use_streams
            else []
        )

    @staticmethod
    def _set_pin_memory(dataset: object, enabled: bool) -> None:
        """Request pinned-memory reads from a single dataset when supported."""
        if hasattr(dataset, "pin_memory"):
            setattr(dataset, "pin_memory", enabled)

    @property
    def effective_read_window(self) -> int:
        """Return the maximum sample count in one fused backend read."""
        return self.batch_size * max(self.prefetch_factor, 1)

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

        Uses fused prefetching when ``prefetch_factor`` is positive, with
        CUDA streams added when enabled and available.

        Yields
        ------
        Batch
            Batched AtomicData as a disjoint graph.
        """
        if self.prefetch_factor > 0:
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
        start_batch = self._consume_epoch_step_start()
        emitted = 0
        if self.batch_sampler is not None:
            for batch_indices in self.batch_sampler:
                if emitted < start_batch:
                    emitted += 1
                    continue
                emitted += 1
                yield list(batch_indices)
            return

        batch: list[int] = []
        if self.sampler is None:
            return
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                if emitted >= start_batch:
                    yield batch
                emitted += 1
                batch = []

        if batch and not self.drop_last and emitted >= start_batch:
            yield batch

    def _consume_epoch_step_start(self) -> int:
        """Return and clear the pending intra-epoch batch start offset."""
        start = self._epoch_step_start
        self._epoch_step_start = 0
        return start

    def set_epoch_step(self, step: int) -> None:
        """Seek the next iterator to an intra-epoch batch offset.

        Parameters
        ----------
        step : int
            Number of complete batches to skip in sampler order before the next
            iterator starts yielding. The skip advances only the sampler/index
            stream; it does not load or collate skipped batches.

        Raises
        ------
        ValueError
            If ``step`` is negative.
        """
        if step < 0:
            raise ValueError(f"step must be >= 0, got {step}")
        self._epoch_step_start = step

    def _iter_simple(self) -> Iterator[Batch]:
        """Simple synchronous iteration without prefetching.

        Yields
        ------
        Batch
            Collated batch of AtomicData.
        """
        transform = self._batch_transform
        for batch_indices in self._generate_batches():
            batch = self.dataset.load_batches([batch_indices])[0]
            if transform is not None:
                batch = transform(batch)
            yield batch

    def _iter_prefetch(self) -> Iterator[Batch]:
        """Iteration with fused prefetching.

        Fuses ``prefetch_factor`` consecutive batches into a single
        ``read_many`` call so that Zarr reader optimisations can coalesce
        scattered indices into fewer large reads.

        Strategy (true double-buffered):

        1. Collect and submit two chunks upfront so that one Zarr
           read is always in flight while the other is being consumed.
        2. Consume the oldest completed chunk, submit a fresh chunk
           into the now-free queue slot, then yield the consumed
           batches.  The next Zarr read runs in the background while
           the caller processes each yielded batch.
        3. Drain the remaining queued chunk after the sampler is
           exhausted.
        4. Cleanup runs in a ``finally`` block so that
           ``cancel_prefetch()`` fires on normal exhaustion, early
           break, and exceptions.

        Yields
        ------
        Batch
            Collated batch of AtomicData.
        """
        stream_idx = 0
        batch_iter = self._generate_batches()
        transform = self._batch_transform

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
            stream = (
                self._streams[stream_idx % self.num_streams] if self._streams else None
            )
            self.dataset.prefetch_fused_batches(chunk, stream=stream)
            stream_idx += 1

        try:
            # Prime: fill both queue slots so one read is always in
            # flight while the other is consumed.
            chunk_a = _collect_chunk()
            if not chunk_a:
                return
            _submit_chunk(chunk_a)

            chunk_b = _collect_chunk()
            if chunk_b:
                _submit_chunk(chunk_b)

            while True:
                # Consume oldest completed read.
                completed_batches = list(self.dataset.get_fused_batches())

                # Refill: collect and submit next chunk into the freed
                # queue slot so the background thread starts reading
                # immediately -- *before* we yield any batches.
                next_chunk = _collect_chunk()
                if next_chunk:
                    _submit_chunk(next_chunk)

                for batch in completed_batches:
                    if transform is not None:
                        batch = transform(batch)
                    yield batch

                # Stop when both the sampler is exhausted and the
                # queue has been drained.
                if not next_chunk and not self.dataset.has_pending_fused_batches():
                    break
        finally:
            self.dataset.cancel_prefetch()

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch for the sampler (used in distributed training).

        Parameters
        ----------
        epoch : int
            Current epoch number.
        """
        candidates = (
            self.batch_sampler,
            getattr(self.batch_sampler, "sampler", None),
            self.sampler,
        )
        seen: set[int] = set()
        for sampler in candidates:
            if sampler is None or id(sampler) in seen:
                continue
            seen.add(id(sampler))
            if hasattr(sampler, "set_epoch"):
                sampler.set_epoch(epoch)
                return
