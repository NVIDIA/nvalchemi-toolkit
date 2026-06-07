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
"""Samplers for datasets composed with :class:`MultiDataset`."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from math import ceil

import torch
from torch.utils.data import Sampler

from nvalchemi.data.datapipes.multidataset import MultiDataset


def _generator_kwargs(generator: torch.Generator | None) -> dict[str, torch.Generator]:
    """Return keyword arguments for torch random APIs."""
    return {"generator": generator} if generator is not None else {}


def _normalise_weights(
    weights: Sequence[float] | None, lengths: Sequence[int]
) -> torch.Tensor:
    """Return positive finite weights for each child dataset."""
    if weights is None:
        weights = lengths
    if len(weights) != len(lengths):
        raise ValueError(f"Expected {len(lengths)} dataset weights, got {len(weights)}")

    tensor = torch.as_tensor(list(weights), dtype=torch.float64)
    if not torch.isfinite(tensor).all():
        raise ValueError("Dataset weights must be finite")
    if (tensor < 0).any():
        raise ValueError("Dataset weights must be non-negative")
    if tensor.sum().item() <= 0:
        raise ValueError("At least one dataset weight must be positive")

    for i, (weight, length) in enumerate(zip(tensor.tolist(), lengths, strict=True)):
        if weight > 0 and length == 0:
            raise ValueError(f"Dataset {i} has positive weight but no samples")
    return tensor / tensor.sum()


def _counts_from_weights(weights: torch.Tensor, total: int) -> list[int]:
    """Allocate an integer total according to fractional weights."""
    if total < 1:
        raise ValueError(f"total must be >= 1, got {total}")

    raw_counts = weights * total
    counts = torch.floor(raw_counts).to(torch.int64)
    remaining = total - int(counts.sum().item())
    if remaining > 0:
        fractions = raw_counts - counts
        for index in torch.argsort(fractions, descending=True)[:remaining].tolist():
            counts[index] += 1
    return counts.tolist()


def _local_order(
    length: int, *, shuffle: bool, generator: torch.Generator | None
) -> list[int]:
    """Return one local index order for a child dataset."""
    if shuffle:
        return torch.randperm(length, **_generator_kwargs(generator)).tolist()
    return list(range(length))


def _shuffle_indices(
    indices: list[int], generator: torch.Generator | None
) -> list[int]:
    """Return a shuffled copy of indices."""
    if len(indices) <= 1:
        return indices
    order = torch.randperm(len(indices), **_generator_kwargs(generator)).tolist()
    return [indices[i] for i in order]


class MultiDatasetSampler(Sampler[int]):
    """Sample global indices from a :class:`MultiDataset` at dataset-level rates.

    Parameters
    ----------
    dataset : MultiDataset
        Dataset wrapper that defines child dataset offsets.
    weights : Sequence[float] | None, default=None
        Per-child dataset sampling rates. ``None`` uses child lengths, matching
        proportional sampling from the concatenated global index space.
    num_samples : int | None, default=None
        Number of global indices emitted per epoch. ``None`` emits
        ``len(dataset)`` samples.
    replacement : bool, default=True
        Whether local samples may repeat within an epoch.
    shuffle : bool, default=True
        Randomize dataset choices and local sample order.
    generator : torch.Generator | None, default=None
        Optional random generator for reproducible sampling.
    """

    def __init__(
        self,
        dataset: MultiDataset,
        *,
        weights: Sequence[float] | None = None,
        num_samples: int | None = None,
        replacement: bool = True,
        shuffle: bool = True,
        generator: torch.Generator | None = None,
    ) -> None:
        """Initialize the sampler."""
        self.dataset = dataset
        self.lengths = [len(child) for child in dataset.datasets]
        self.weights = _normalise_weights(weights, self.lengths)
        self.num_samples = len(dataset) if num_samples is None else num_samples
        if self.num_samples < 1:
            raise ValueError(f"num_samples must be >= 1, got {self.num_samples}")
        self.replacement = replacement
        self.shuffle = shuffle
        self.generator = generator

        if not replacement:
            counts = _counts_from_weights(self.weights, self.num_samples)
            for dataset_index, (count, length) in enumerate(
                zip(counts, self.lengths, strict=True)
            ):
                if count > length:
                    raise ValueError(
                        "replacement=False cannot draw "
                        f"{count} samples from dataset {dataset_index} "
                        f"with only {length} samples"
                    )

    def __iter__(self) -> Iterator[int]:
        """Yield global sample indices."""
        if self.replacement and self.shuffle:
            dataset_choices = torch.multinomial(
                self.weights,
                self.num_samples,
                replacement=True,
                **_generator_kwargs(self.generator),
            ).tolist()
            for dataset_index in dataset_choices:
                local_index = int(
                    torch.randint(
                        self.lengths[dataset_index],
                        (1,),
                        **_generator_kwargs(self.generator),
                    ).item()
                )
                yield self.dataset.to_global_index(dataset_index, local_index)
            return

        counts = _counts_from_weights(self.weights, self.num_samples)
        dataset_choices = [
            dataset_index
            for dataset_index, count in enumerate(counts)
            for _ in range(count)
        ]
        if self.shuffle:
            dataset_choices = _shuffle_indices(dataset_choices, self.generator)

        local_orders = [
            _local_order(length, shuffle=self.shuffle, generator=self.generator)
            for length in self.lengths
        ]
        cursors = [0] * len(self.lengths)
        for dataset_index in dataset_choices:
            cursor = cursors[dataset_index]
            if self.replacement:
                local_index = local_orders[dataset_index][
                    cursor % self.lengths[dataset_index]
                ]
            else:
                local_index = local_orders[dataset_index][cursor]
            cursors[dataset_index] += 1
            yield self.dataset.to_global_index(dataset_index, local_index)

    def __len__(self) -> int:
        """Return the number of emitted global indices."""
        return self.num_samples


class MultiDatasetBatchSampler(Sampler[list[int]]):
    """Sample full global-index batches from a :class:`MultiDataset`.

    Parameters
    ----------
    dataset : MultiDataset
        Dataset wrapper that defines child dataset offsets.
    batch_size : int
        Number of samples in each emitted batch.
    weights : Sequence[float] | None, default=None
        Per-child rates used to allocate ``batch_size`` slots. ``None`` uses
        child lengths, matching proportional sampling from the global index
        space.
    samples_per_dataset : Sequence[int] | None, default=None
        Exact per-child sample counts per batch. Mutually exclusive with
        ``weights``.
    num_batches : int | None, default=None
        Number of batches per epoch. For replacement sampling, the default is
        ``ceil(len(dataset) / batch_size)``. Without replacement, the default is
        the number of complete batches supported by the smallest requested child
        allocation.
    replacement : bool, default=True
        Whether local samples may repeat within an epoch.
    shuffle : bool, default=True
        Randomize local sample order and sample order within each batch.
    generator : torch.Generator | None, default=None
        Optional random generator for reproducible sampling.
    """

    def __init__(
        self,
        dataset: MultiDataset,
        *,
        batch_size: int,
        weights: Sequence[float] | None = None,
        samples_per_dataset: Sequence[int] | None = None,
        num_batches: int | None = None,
        replacement: bool = True,
        shuffle: bool = True,
        generator: torch.Generator | None = None,
    ) -> None:
        """Initialize the batch sampler."""
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        if weights is not None and samples_per_dataset is not None:
            raise ValueError("weights and samples_per_dataset are mutually exclusive")

        self.dataset = dataset
        self.batch_size = batch_size
        self.lengths = [len(child) for child in dataset.datasets]
        self.replacement = replacement
        self.shuffle = shuffle
        self.generator = generator

        if samples_per_dataset is None:
            normalised_weights = _normalise_weights(weights, self.lengths)
            self.samples_per_dataset = _counts_from_weights(
                normalised_weights, batch_size
            )
        else:
            if len(samples_per_dataset) != len(self.lengths):
                raise ValueError(
                    f"Expected {len(self.lengths)} per-dataset counts, "
                    f"got {len(samples_per_dataset)}"
                )
            self.samples_per_dataset = [int(count) for count in samples_per_dataset]

        if any(count < 0 for count in self.samples_per_dataset):
            raise ValueError("samples_per_dataset counts must be non-negative")
        if sum(self.samples_per_dataset) != batch_size:
            raise ValueError(
                "samples_per_dataset counts must sum to batch_size: "
                f"{sum(self.samples_per_dataset)} != {batch_size}"
            )
        if all(count == 0 for count in self.samples_per_dataset):
            raise ValueError("At least one dataset must contribute samples per batch")

        for dataset_index, (count, length) in enumerate(
            zip(self.samples_per_dataset, self.lengths, strict=True)
        ):
            if count > 0 and length == 0:
                raise ValueError(
                    f"Dataset {dataset_index} contributes {count} samples per "
                    "batch but has no samples"
                )

        if replacement:
            self.num_batches = (
                ceil(len(dataset) / batch_size) if num_batches is None else num_batches
            )
        else:
            max_complete_batches = min(
                length // count
                for length, count in zip(
                    self.lengths, self.samples_per_dataset, strict=True
                )
                if count > 0
            )
            self.num_batches = (
                max_complete_batches if num_batches is None else num_batches
            )
            if self.num_batches > max_complete_batches:
                raise ValueError(
                    "replacement=False supports at most "
                    f"{max_complete_batches} complete batches for the requested "
                    "per-dataset counts"
                )
        if self.num_batches < 1:
            raise ValueError(f"num_batches must be >= 1, got {self.num_batches}")

    def __iter__(self) -> Iterator[list[int]]:
        """Yield batches of global sample indices."""
        if self.replacement:
            cursors = [0] * len(self.lengths)
            for _ in range(self.num_batches):
                batch: list[int] = []
                for dataset_index, count in enumerate(self.samples_per_dataset):
                    if count == 0:
                        continue
                    if self.shuffle:
                        local_indices = torch.randint(
                            self.lengths[dataset_index],
                            (count,),
                            **_generator_kwargs(self.generator),
                        ).tolist()
                    else:
                        cursor = cursors[dataset_index]
                        local_indices = [
                            (cursor + i) % self.lengths[dataset_index]
                            for i in range(count)
                        ]
                        cursors[dataset_index] += count
                    batch.extend(
                        self.dataset.to_global_index(dataset_index, local_index)
                        for local_index in local_indices
                    )
                yield _shuffle_indices(batch, self.generator) if self.shuffle else batch
            return

        local_orders = [
            _local_order(length, shuffle=self.shuffle, generator=self.generator)
            for length in self.lengths
        ]
        cursors = [0] * len(self.lengths)
        for _ in range(self.num_batches):
            batch = []
            for dataset_index, count in enumerate(self.samples_per_dataset):
                if count == 0:
                    continue
                cursor = cursors[dataset_index]
                local_indices = local_orders[dataset_index][cursor : cursor + count]
                cursors[dataset_index] += count
                batch.extend(
                    self.dataset.to_global_index(dataset_index, local_index)
                    for local_index in local_indices
                )
            yield _shuffle_indices(batch, self.generator) if self.shuffle else batch

    def __len__(self) -> int:
        """Return the number of emitted batches."""
        return self.num_batches


class BalancedMultiDatasetBatchSampler(MultiDatasetBatchSampler):
    """Batch sampler that allocates batch slots evenly across child datasets."""

    def __init__(
        self,
        dataset: MultiDataset,
        *,
        batch_size: int,
        num_batches: int | None = None,
        replacement: bool = True,
        shuffle: bool = True,
        generator: torch.Generator | None = None,
    ) -> None:
        """Initialize an evenly balanced multidataset batch sampler."""
        super().__init__(
            dataset,
            batch_size=batch_size,
            weights=[1.0] * len(dataset.datasets),
            num_batches=num_batches,
            replacement=replacement,
            shuffle=shuffle,
            generator=generator,
        )
