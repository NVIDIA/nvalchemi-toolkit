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
"""Compose multiple AtomicData-native datasets behind one index space."""

from __future__ import annotations

import logging
from bisect import bisect_right
from collections import deque
from collections.abc import Iterator, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

import torch

from nvalchemi.data.atomic_data import AtomicData
from nvalchemi.data.batch import Batch
from nvalchemi.data.datapipes.dataset import Dataset

logger = logging.getLogger(__name__)

DATASET_INDEX_METADATA_KEY = "dataset_index"


@dataclass
class _FusedBatchResult:
    """Container for async multidataset fused-batch results."""

    batches: list[Batch] | None = None
    error: Exception | None = None


@dataclass
class _DelegatedFusedBatch:
    """Marker for fused reads delegated to one child dataset."""

    dataset_index: int


@dataclass
class _ChildFusedBatchRequest:
    """Per-child route for one mixed multidataset fused read."""

    output_batch_indices: list[int]
    local_batch_lists: list[list[int]]
    output_positions: list[list[int]]


@dataclass
class _BatchRoute:
    """Route for one child dataset within a global batch request."""

    dataset_index: int
    local_indices: list[int]
    positions: list[int]


@dataclass
class _BatchRoutePlan:
    """Child-dataset routes for one global sample request."""

    routes: list[_BatchRoute]
    size: int

    @property
    def single_route(self) -> _BatchRoute | None:
        """Return the only route when all samples belong to one child."""
        return self.routes[0] if len(self.routes) == 1 else None


PendingFusedBatch = Future[_FusedBatchResult] | _DelegatedFusedBatch


class MultiDataset:
    """Compose multiple :class:`Dataset` instances behind one index space.

    The class provides concatenated indexing and nvalchemi-specific batch APIs
    used by :class:`~nvalchemi.data.datapipes.dataloader.DataLoader`.

    Parameters
    ----------
    *datasets : Dataset
        One or more nvalchemi datasets. Order defines the global index mapping.
    output_strict : bool, default=True
        If True, require all datasets to expose identical field names.
    num_workers : int, default=2
        Thread pool size for mixed-dataset fused prefetches.
    """

    def __init__(
        self,
        *datasets: Dataset,
        output_strict: bool = True,
        num_workers: int = 2,
    ) -> None:
        """Initialize the multidataset wrapper.

        Parameters
        ----------
        *datasets : Dataset
            Datasets to concatenate.
        output_strict : bool, default=True
            Require matching field names across datasets.
        num_workers : int, default=2
            Worker count for mixed-dataset fused prefetches.

        Raises
        ------
        TypeError
            If any child is not a nvalchemi Dataset.
        ValueError
            If no datasets are provided or strict field names differ.
        """
        if len(datasets) < 1:
            raise ValueError(
                f"MultiDataset requires at least one dataset, got {len(datasets)}"
            )
        for i, dataset in enumerate(datasets):
            if not isinstance(dataset, Dataset):
                raise TypeError(
                    f"datasets[{i}] must be a Dataset instance, got {type(dataset).__name__}"
                )

        self._datasets = list(datasets)
        self._output_strict = output_strict
        self.num_workers = num_workers

        cumulative_lengths = [0]
        for dataset in self._datasets:
            cumulative_lengths.append(cumulative_lengths[-1] + len(dataset))
        self._cumul = cumulative_lengths

        self._field_names = self.validate_field_names(output_strict)
        self._fused_batch_prefetch_queue: deque[PendingFusedBatch] = deque()
        self._executor: ThreadPoolExecutor | None = None

    def validate_field_names(self, output_strict: bool | None = None) -> list[str]:
        """Validate and return the field names exposed by this wrapper.

        Parameters
        ----------
        output_strict : bool | None, default=None
            Strictness mode to use for validation. ``None`` uses the mode passed
            to :class:`MultiDataset` at construction time.

        Returns
        -------
        list[str]
            Field names this multidataset exposes.

        Raises
        ------
        ValueError
            If ``output_strict=True`` and non-empty child datasets expose
            different field names.

        Notes
        -----
        With ``output_strict=True``, all non-empty child datasets must expose
        identical field names. Empty children are skipped, matching
        the standalone ``MultiDataset`` strict-output behavior.

        With ``output_strict=False``, no cross-dataset validation is performed
        and the first child dataset's field names are returned. Use this mode
        for heterogeneous datasets where a custom training loop or collator
        handles source-specific fields.
        """
        if output_strict is None:
            output_strict = self._output_strict
        if not output_strict:
            return list(self._datasets[0].field_names)

        reference: list[str] | None = None
        reference_index: int | None = None
        for i, dataset in enumerate(self._datasets):
            if len(dataset) == 0:
                continue

            current = list(dataset.field_names)
            if reference is None:
                reference = current
                reference_index = i
                continue

            reference_set = set(reference)
            field_names = set(dataset.field_names)
            if field_names != reference_set:
                raise ValueError(
                    "output_strict=True requires identical field names across "
                    f"datasets: dataset {reference_index} has {sorted(reference_set)}, "
                    f"dataset {i} has {sorted(field_names)}"
                )
        return (
            reference if reference is not None else list(self._datasets[0].field_names)
        )

    def _ensure_executor(self) -> ThreadPoolExecutor:
        """Lazily create the thread pool executor."""
        if self._executor is None:
            self._executor = ThreadPoolExecutor(
                max_workers=self.num_workers,
                thread_name_prefix="multidataset_prefetch",
            )
        return self._executor

    def _index_to_dataset_and_local(self, index: int) -> tuple[int, int]:
        """Map a global index to ``(dataset_index, local_index)``."""
        length = len(self)
        original_index = index
        if index < 0:
            index += length
        if index < 0 or index >= length:
            raise IndexError(
                f"Index {original_index} out of range for MultiDataset with {length} samples"
            )

        dataset_index = bisect_right(self._cumul, index) - 1
        return dataset_index, index - self._cumul[dataset_index]

    def _index_to_dataset_and_local_optional(
        self, index: int
    ) -> tuple[int, int] | None:
        """Map a global index, returning None when it is out of range."""
        try:
            return self._index_to_dataset_and_local(index)
        except IndexError:
            return None

    @staticmethod
    def _with_dataset_metadata(
        metadata: dict[str, Any], dataset_index: int
    ) -> dict[str, Any]:
        """Return metadata annotated with its source dataset index."""
        enriched = dict(metadata)
        enriched[DATASET_INDEX_METADATA_KEY] = dataset_index
        return enriched

    def _route_indices(self, indices: Sequence[int]) -> _BatchRoutePlan:
        """Plan child-dataset reads for a global sample request."""
        grouped_indices: dict[int, list[int]] = {}
        grouped_positions: dict[int, list[int]] = {}
        for position, index in enumerate(indices):
            dataset_index, local_index = self._index_to_dataset_and_local(index)
            grouped_indices.setdefault(dataset_index, []).append(local_index)
            grouped_positions.setdefault(dataset_index, []).append(position)

        return _BatchRoutePlan(
            routes=[
                _BatchRoute(
                    dataset_index=dataset_index,
                    local_indices=local_indices,
                    positions=grouped_positions[dataset_index],
                )
                for dataset_index, local_indices in grouped_indices.items()
            ],
            size=len(indices),
        )

    @staticmethod
    def _combine_child_batches(parts: list[tuple[list[int], Batch]]) -> Batch:
        """Append child batch parts and restore the original sample order."""
        if not parts:
            raise ValueError("MultiDataset.load_batches() requires non-empty batches")

        combined_positions = list(parts[0][0])
        combined = parts[0][1]
        if combined.num_graphs != len(combined_positions):
            raise RuntimeError(
                "Child dataset returned a batch with "
                f"{combined.num_graphs} graphs for {len(combined_positions)} indices"
            )

        if len(parts) > 1:
            combined = combined.clone()
            for positions, child_batch in parts[1:]:
                if child_batch.num_graphs != len(positions):
                    raise RuntimeError(
                        "Child dataset returned a batch with "
                        f"{child_batch.num_graphs} graphs for {len(positions)} indices"
                    )
                combined.append(child_batch)
                combined_positions.extend(positions)

        restore_order = [
            combined_index
            for combined_index, _position in sorted(
                enumerate(combined_positions), key=lambda item: item[1]
            )
        ]
        if restore_order == list(range(len(restore_order))):
            return combined
        return combined.index_select(restore_order)

    def __len__(self) -> int:
        """Return the total number of samples."""
        return self._cumul[-1]

    @property
    def datasets(self) -> tuple[Dataset, ...]:
        """Child datasets in global index order."""
        return tuple(self._datasets)

    @property
    def offsets(self) -> tuple[int, ...]:
        """Cumulative global index offsets for child datasets."""
        return tuple(self._cumul)

    def to_global_index(self, dataset_index: int, local_index: int) -> int:
        """Map a child dataset index and local index to one global index."""
        if dataset_index < 0:
            dataset_index += len(self._datasets)
        if dataset_index < 0 or dataset_index >= len(self._datasets):
            raise IndexError(
                f"dataset_index {dataset_index} out of range for "
                f"{len(self._datasets)} child datasets"
            )

        child_length = len(self._datasets[dataset_index])
        original_local_index = local_index
        if local_index < 0:
            local_index += child_length
        if local_index < 0 or local_index >= child_length:
            raise IndexError(
                f"local_index {original_local_index} out of range for "
                f"dataset {dataset_index} with {child_length} samples"
            )
        return self._cumul[dataset_index] + local_index

    def to_local_index(self, index: int) -> tuple[int, int]:
        """Map one global index to ``(dataset_index, local_index)``."""
        return self._index_to_dataset_and_local(index)

    def __getitem__(self, index: int) -> tuple[AtomicData, dict[str, Any]]:
        """Return one sample by global index."""
        dataset_index, local_index = self._index_to_dataset_and_local(index)
        data, metadata = self._datasets[dataset_index][local_index]
        return data, self._with_dataset_metadata(metadata, dataset_index)

    def prefetch(self, index: int, stream: torch.cuda.Stream | None = None) -> None:
        """Start prefetching one sample by global index."""
        dataset_index, local_index = self._index_to_dataset_and_local(index)
        self._datasets[dataset_index].prefetch(local_index, stream=stream)

    def prefetch_batch(
        self,
        indices: Sequence[int],
        streams: Sequence[torch.cuda.Stream] | None = None,
    ) -> None:
        """Start prefetching multiple samples by global index."""
        for i, index in enumerate(indices):
            stream = streams[i % len(streams)] if streams else None
            self.prefetch(index, stream=stream)

    def _local_batch_lists_if_single_dataset(
        self, batch_index_lists: Sequence[Sequence[int]]
    ) -> tuple[int, list[list[int]]] | None:
        """Return local batch lists when a fused chunk belongs to one child."""
        dataset_index: int | None = None
        local_batch_lists: list[list[int]] = []
        for batch_indices in batch_index_lists:
            local_batch: list[int] = []
            for index in batch_indices:
                current_dataset_index, local_index = self._index_to_dataset_and_local(
                    index
                )
                if dataset_index is None:
                    dataset_index = current_dataset_index
                elif current_dataset_index != dataset_index:
                    return None
                local_batch.append(local_index)
            local_batch_lists.append(local_batch)

        if dataset_index is None:
            return None
        return dataset_index, local_batch_lists

    def _child_fused_batch_requests(
        self, batch_index_lists: Sequence[Sequence[int]]
    ) -> dict[int, _ChildFusedBatchRequest]:
        """Build per-child fused-batch routes for a mixed global chunk."""
        requests: dict[int, _ChildFusedBatchRequest] = {}
        for output_batch_index, batch_indices in enumerate(batch_index_lists):
            if not batch_indices:
                raise ValueError("Fused batch prefetch does not support empty batches")

            route_plan = self._route_indices(batch_indices)
            for route in route_plan.routes:
                request = requests.setdefault(
                    route.dataset_index,
                    _ChildFusedBatchRequest(
                        output_batch_indices=[],
                        local_batch_lists=[],
                        output_positions=[],
                    ),
                )
                request.output_batch_indices.append(output_batch_index)
                request.local_batch_lists.append(route.local_indices)
                request.output_positions.append(route.positions)
        return requests

    def _load_fused_batches(
        self,
        batch_index_lists: Sequence[Sequence[int]],
        stream: torch.cuda.Stream | None = None,
    ) -> _FusedBatchResult:
        """Load multiple global batches by grouping reads per child dataset."""
        try:
            routed_requests = self._child_fused_batch_requests(batch_index_lists)
            batch_parts: list[list[tuple[list[int], Batch]]] = [
                [] for _ in batch_index_lists
            ]

            for dataset_index, request in routed_requests.items():
                child_batches = self._datasets[dataset_index].load_batches(
                    request.local_batch_lists, stream=stream
                )
                if len(child_batches) != len(request.local_batch_lists):
                    raise RuntimeError(
                        f"Dataset {dataset_index} returned {len(child_batches)} "
                        f"batches for {len(request.local_batch_lists)} fused requests"
                    )
                for output_batch_index, positions, child_batch in zip(
                    request.output_batch_indices,
                    request.output_positions,
                    child_batches,
                    strict=True,
                ):
                    batch_parts[output_batch_index].append((positions, child_batch))

            batches = [self._combine_child_batches(parts) for parts in batch_parts]
            return _FusedBatchResult(batches=batches)
        except Exception as e:
            return _FusedBatchResult(error=e)

    def prefetch_fused_batches(
        self,
        batch_index_lists: Sequence[Sequence[int]],
        stream: torch.cuda.Stream | None = None,
    ) -> None:
        """Submit multiple global batches as one fused async read."""
        if len(self._fused_batch_prefetch_queue) >= 2:
            raise RuntimeError(
                "Fused batch prefetch queue is full; consume a pending chunk first."
            )

        local = self._local_batch_lists_if_single_dataset(batch_index_lists)
        if local is not None:
            dataset_index, local_batch_lists = local
            self._datasets[dataset_index].prefetch_fused_batches(
                local_batch_lists, stream=stream
            )
            self._fused_batch_prefetch_queue.append(
                _DelegatedFusedBatch(dataset_index=dataset_index)
            )
            return

        executor = self._ensure_executor()
        self._fused_batch_prefetch_queue.append(
            executor.submit(self._load_fused_batches, batch_index_lists, stream)
        )

    def load_batches(
        self,
        batch_index_lists: Sequence[Sequence[int]],
        stream: torch.cuda.Stream | None = None,
    ) -> list[Batch]:
        """Load several global batches immediately.

        This is the synchronous counterpart to
        :meth:`prefetch_fused_batches`/:meth:`get_fused_batches`. Same-child
        chunks are delegated directly to the owning child dataset, while mixed
        chunks are routed per child and recombined in the requested batch order.

        Parameters
        ----------
        batch_index_lists : Sequence[Sequence[int]]
            Per-batch global sample indices.
        stream : torch.cuda.Stream | None, default=None
            CUDA stream for child dataset transfers when supported.

        Returns
        -------
        list[Batch]
            One :class:`Batch` per input batch-index list.
        """
        local = self._local_batch_lists_if_single_dataset(batch_index_lists)
        if local is not None:
            dataset_index, local_batch_lists = local
            return self._datasets[dataset_index].load_batches(
                local_batch_lists, stream=stream
            )

        result = self._load_fused_batches(batch_index_lists, stream=stream)
        if result.error is not None:
            raise result.error
        if result.batches is None:
            raise RuntimeError(
                "MultiDataset fused batch load returned None batches without error"
            )
        return result.batches

    def has_pending_fused_batches(self) -> bool:
        """Return whether a fused prefetch chunk is waiting to be consumed."""
        return bool(self._fused_batch_prefetch_queue)

    def get_fused_batches(self) -> Iterator[Batch]:
        """Consume one pending fused prefetch chunk."""
        if not self._fused_batch_prefetch_queue:
            raise RuntimeError(
                "No fused batch prefetch pending; call prefetch_fused_batches() "
                "before get_fused_batches()."
            )

        pending = self._fused_batch_prefetch_queue.popleft()
        if isinstance(pending, _DelegatedFusedBatch):
            yield from self._datasets[pending.dataset_index].get_fused_batches()
            return

        result = pending.result()
        if result.error is not None:
            raise result.error
        if result.batches is None:
            raise RuntimeError(
                "MultiDataset fused batch prefetch returned None batches without error"
            )
        yield from result.batches

    def cancel_prefetch(self, index: int | None = None) -> None:
        """Cancel prefetch for one global index or all child datasets."""
        if index is None:
            self._fused_batch_prefetch_queue.clear()
            for dataset in self._datasets:
                dataset.cancel_prefetch()
            return

        mapped = self._index_to_dataset_and_local_optional(index)
        if mapped is None:
            return

        dataset_index, local_index = mapped
        self._datasets[dataset_index].cancel_prefetch(local_index)

    @property
    def prefetch_count(self) -> int:
        """Return queued prefetch count across this wrapper and children."""
        return len(self._fused_batch_prefetch_queue) + sum(
            dataset.prefetch_count for dataset in self._datasets
        )

    @property
    def field_names(self) -> list[str]:
        """Return field names exposed by child datasets."""
        return list(self._field_names)

    def get_metadata(self, index: int) -> tuple[int, int]:
        """Return lightweight metadata for a sample by global index."""
        dataset_index, local_index = self._index_to_dataset_and_local(index)
        return self._datasets[dataset_index].get_metadata(local_index)

    def __iter__(self) -> Iterator[tuple[AtomicData, dict[str, Any]]]:
        """Iterate over all samples in global index order."""
        for index in range(len(self)):
            yield self[index]

    def close(self) -> None:
        """Close all child datasets and release wrapper resources."""
        futures_to_drain: list[Future] = [
            *[
                pending
                for pending in self._fused_batch_prefetch_queue
                if not isinstance(pending, _DelegatedFusedBatch)
            ],
        ]
        for future in futures_to_drain:
            try:
                future.result(timeout=1.0)
            except Exception:
                logger.debug("Ignoring error during multidataset prefetch cleanup")

        self._fused_batch_prefetch_queue.clear()

        if self._executor is not None:
            self._executor.shutdown(wait=False)
            self._executor = None

        for dataset in self._datasets:
            dataset.close()

    def __enter__(self) -> MultiDataset:
        """Enter context manager."""
        return self

    def __exit__(
        self, exc_type: type | None, exc_val: BaseException | None, exc_tb: Any
    ) -> None:
        """Exit context manager."""
        self.close()

    def __repr__(self) -> str:
        """Return a human-readable representation."""
        parts = [f"  ({i}): {dataset}" for i, dataset in enumerate(self._datasets)]
        return (
            f"{self.__class__.__name__}(\n"
            f"  output_strict={self._output_strict},\n"
            f"  datasets=[\n" + ",\n".join(parts) + "\n  ]\n)"
        )
