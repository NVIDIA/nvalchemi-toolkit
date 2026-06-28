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
from numbers import Integral, Real
from typing import TYPE_CHECKING, Literal, Protocol, Self, TypeAlias, runtime_checkable

import torch
from torch.utils.data import Sampler

from nvalchemi.data.datapipes.multidataset import MultiDataset

if TYPE_CHECKING:
    from nvalchemi.distributed import DistributedManager

EpochPolicy: TypeAlias = Literal["dataset_size", "min_size", "max_size"]


@runtime_checkable
class DistributedSamplerProtocol(Protocol):
    """Protocol for samplers that partition work across distributed ranks.

    This intentionally matches the public surface provided by
    :class:`torch.utils.data.DistributedSampler` so native PyTorch samplers
    satisfy the protocol structurally.

    Attributes
    ----------
    num_replicas : int
        Number of distributed workers participating in sampling.
    rank : int
        Rank local to the sampler's process group.
    """

    num_replicas: int
    rank: int

    def set_epoch(self, epoch: int) -> None:
        """Set the current epoch for deterministic per-epoch shuffling."""


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


def _num_sharded_items(length: int, num_replicas: int, drop_last: bool) -> int:
    """Return number of items emitted by one distributed rank."""
    if num_replicas == 1:
        return length
    if drop_last and length % num_replicas != 0:
        return ceil((length - num_replicas) / num_replicas)
    return ceil(length / num_replicas)


def _distributed_shard(
    indices: list,
    *,
    num_replicas: int,
    rank: int,
    drop_last: bool,
) -> list:
    """Return the subset of epoch items assigned to one distributed rank.

    Parameters
    ----------
    indices : list
        Sample indices in the order they would be retrieved for this epoch
        before splitting the work across data-parallel ranks. In a
        single-process run, this would be the sampler order.
    num_replicas : int
        Number of distributed ranks sharing the epoch.
    rank : int
        Rank whose local shard should be returned.
    drop_last : bool
        Whether to truncate the full epoch instead of padding it when the epoch
        length is not evenly divisible by ``num_replicas``.

    Returns
    -------
    list
        Rank-local shard of ``indices``.

    Notes
    -----
    To make strided sharding produce the same number of items on each rank, the
    full epoch order is first resized to ``total_size``. ``num_samples`` is the
    number of items one rank should emit, computed as
    ``ceil(len(indices) / num_replicas)`` unless ``drop_last=True`` requires
    truncating an uneven tail. ``total_size`` is the all-rank item count,
    ``num_samples * num_replicas``.

    With ``drop_last=True``, the full list is truncated to ``total_size``.
    Otherwise, items from the beginning of the epoch are repeated until the list
    is evenly divisible across ranks, matching PyTorch
    :class:`~torch.utils.data.DistributedSampler` behavior. After resizing, rank
    ``r`` receives every ``num_replicas``-th item starting at offset ``r``:
    ``indices[r:total_size:num_replicas]``.
    """
    if num_replicas == 1:
        return indices

    num_samples = _num_sharded_items(len(indices), num_replicas, drop_last)
    total_size = num_samples * num_replicas
    if drop_last:
        indices = indices[:total_size]
    elif len(indices) < total_size:
        padding_size = total_size - len(indices)
        if padding_size <= len(indices):
            indices += indices[:padding_size]
        else:
            indices += (indices * ceil(padding_size / len(indices)))[:padding_size]
    return indices[rank:total_size:num_replicas]


def _contains_float(values: Sequence[int | float]) -> bool:
    """Return whether any value should switch counts to ratio semantics."""
    return any(
        isinstance(value, Real) and not isinstance(value, Integral) for value in values
    )


def _num_batches_from_policy(
    *,
    epoch_policy: EpochPolicy,
    lengths: Sequence[int],
    samples_per_dataset: Sequence[int],
    batch_size: int,
    total_length: int,
    replacement: bool,
) -> int:
    """Compute default epoch length from per-dataset batch allocations."""
    contributing = [
        (length, count)
        for length, count in zip(lengths, samples_per_dataset, strict=True)
        if count > 0
    ]
    if not contributing:
        raise ValueError("At least one dataset must contribute samples per batch")

    if replacement:
        min_batches = min(ceil(length / count) for length, count in contributing)
        max_batches = max(ceil(length / count) for length, count in contributing)
    else:
        min_batches = min(length // count for length, count in contributing)
        max_batches = max(length // count for length, count in contributing)

    if epoch_policy == "dataset_size":
        return ceil(total_length / batch_size) if replacement else min_batches
    if epoch_policy == "min_size":
        return min_batches
    if epoch_policy == "max_size":
        if not replacement and max_batches > min_batches:
            raise ValueError(
                "epoch_policy='max_size' requires replacement=True when smaller "
                "datasets would need oversampling"
            )
        return max_batches
    raise ValueError(
        "epoch_policy must be one of 'dataset_size', 'min_size', or 'max_size'"
    )


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
    num_replicas : int | None, default=None
        Number of distributed ranks. ``None`` uses initialized
        ``distributed_manager.world_size`` or defaults to ``1``.
    rank : int | None, default=None
        Rank for this sampler. ``None`` uses initialized
        ``distributed_manager.rank`` or defaults to ``0``.
    distributed_manager : DistributedManager | None, default=None
        Optional distributed manager used to infer rank and world size.
    seed : int, default=0
        Base seed used for deterministic shuffling across epochs when
        ``generator`` is ``None``.
    drop_last : bool, default=False
        Drop tail samples to make the epoch evenly divisible across ranks.
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
        num_replicas: int | None = None,
        rank: int | None = None,
        distributed_manager: DistributedManager | None = None,
        seed: int = 0,
        drop_last: bool = False,
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
        if distributed_manager is not None and distributed_manager.is_initialized():
            num_replicas = distributed_manager.world_size
            rank = distributed_manager.rank
        if num_replicas is None:
            num_replicas = 1
        if rank is None:
            rank = 0
        if num_replicas < 1:
            raise ValueError(f"num_replicas must be >= 1, got {num_replicas}")
        if rank < 0 or rank >= num_replicas:
            raise ValueError(
                f"rank must be in the range [0, {num_replicas}), got {rank}"
            )
        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = seed
        self.drop_last = drop_last
        self.epoch = 0

        # if not sampling without replacement, we go through the datasets
        # and make sure there are sufficient samples to meet the weights
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

    def _epoch_generator(self) -> torch.Generator | None:
        """Return the generator used for this epoch."""
        if self.generator is not None:
            return self.generator
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        return generator

    def _global_indices(self) -> list[int]:
        """Return the full unsharded epoch of global sample indices."""
        generator = self._epoch_generator()
        if self.replacement and self.shuffle:
            # Draw dataset choices once, then draw per-dataset local indices in
            # vectors to avoid one scalar RNG call per emitted sample.
            dataset_choices = torch.multinomial(
                self.weights,
                self.num_samples,
                replacement=True,
                **_generator_kwargs(generator),
            )
            counts = torch.bincount(
                dataset_choices, minlength=len(self.lengths)
            ).tolist()
            local_orders = [
                torch.randint(
                    length,
                    (count,),
                    **_generator_kwargs(generator),
                ).tolist()
                for length, count in zip(self.lengths, counts, strict=True)
            ]
            cursors = [0] * len(self.lengths)
            indices = []
            for dataset_index in dataset_choices.tolist():
                cursor = cursors[dataset_index]
                local_index = local_orders[dataset_index][cursor]
                cursors[dataset_index] += 1
                indices.append(self.dataset.to_global_index(dataset_index, local_index))
            return indices

        # case where we may be shuffling or replacing samples
        counts = _counts_from_weights(self.weights, self.num_samples)
        dataset_choices = [
            dataset_index
            for dataset_index, count in enumerate(counts)
            for _ in range(count)
        ]
        if self.shuffle:
            dataset_choices = _shuffle_indices(dataset_choices, generator)

        # Without replacement, build one local order per child dataset and
        # advance cursors as batches consume those fixed orders.
        local_orders = [
            _local_order(length, shuffle=self.shuffle, generator=generator)
            for length in self.lengths
        ]
        cursors = [0] * len(self.lengths)
        indices = []
        for dataset_index in dataset_choices:
            cursor = cursors[dataset_index]
            if self.replacement:
                local_index = local_orders[dataset_index][
                    cursor % self.lengths[dataset_index]
                ]
            else:
                local_index = local_orders[dataset_index][cursor]
            cursors[dataset_index] += 1
            indices.append(self.dataset.to_global_index(dataset_index, local_index))
        return indices

    def __iter__(self) -> Iterator[int]:
        """Yield rank-local global sample indices."""
        yield from _distributed_shard(
            self._global_indices(),
            num_replicas=self.num_replicas,
            rank=self.rank,
            drop_last=self.drop_last,
        )

    def __len__(self) -> int:
        """Return the number of rank-local emitted global indices."""
        return _num_sharded_items(self.num_samples, self.num_replicas, self.drop_last)

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch used for deterministic distributed shuffling.

        Parameters
        ----------
        epoch : int
            Epoch number added to ``seed`` when this sampler owns its generator.
        """
        self.epoch = epoch


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
    samples_per_dataset : Sequence[int | float] | None, default=None
        Per-child batch allocation. Integer entries are exact sample counts
        per batch. If any entry is a float, the full sequence is interpreted
        as relative per-dataset rates and allocated across ``batch_size``.
        Mutually exclusive with ``weights``.
    num_batches : int | None, default=None
        Number of batches per epoch. For replacement sampling, the default is
        ``ceil(len(dataset) / batch_size)``. Without replacement, the default is
        the number of complete batches supported by the smallest requested child
        allocation.
    epoch_policy : {"dataset_size", "min_size", "max_size"}, default="dataset_size"
        Policy used to compute ``num_batches`` when it is not provided.
        ``"dataset_size"`` simply returns the combined dataset length divided
        by the batch size when ``replacement=True``, otherwise ``min_size``. ``"min_size"``
        stops when the smallest contributing dataset would be exhausted.
        ``"max_size"`` runs until the largest contributing dataset would be
        exhausted, oversampling smaller datasets when ``replacement=True``.
    replacement : bool, default=True
        Whether local samples may repeat within an epoch.
    shuffle : bool, default=True
        Randomize local sample order and sample order within each batch.
    generator : torch.Generator | None, default=None
        Optional random generator for reproducible sampling.
    num_replicas : int | None, default=None
        Number of distributed ranks. ``None`` uses initialized
        ``distributed_manager.world_size`` or defaults to ``1``.
    rank : int | None, default=None
        Rank for this sampler. ``None`` uses initialized
        ``distributed_manager.rank`` or defaults to ``0``.
    distributed_manager : DistributedManager | None, default=None
        Optional distributed manager used to infer rank and world size.
    seed : int, default=0
        Base seed used for deterministic shuffling across epochs when
        ``generator`` is ``None``.
    drop_last : bool, default=False
        Drop tail batches to make the epoch evenly divisible across ranks.
    """

    def __init__(
        self,
        dataset: MultiDataset,
        *,
        batch_size: int,
        weights: Sequence[float] | None = None,
        samples_per_dataset: Sequence[int | float] | None = None,
        num_batches: int | None = None,
        epoch_policy: EpochPolicy = "dataset_size",
        replacement: bool = True,
        shuffle: bool = True,
        generator: torch.Generator | None = None,
        num_replicas: int | None = None,
        rank: int | None = None,
        distributed_manager: DistributedManager | None = None,
        seed: int = 0,
        drop_last: bool = False,
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
        self.epoch_policy = epoch_policy
        if distributed_manager is not None and distributed_manager.is_initialized():
            num_replicas = distributed_manager.world_size
            rank = distributed_manager.rank
        if num_replicas is None:
            num_replicas = 1
        if rank is None:
            rank = 0
        if num_replicas < 1:
            raise ValueError(f"num_replicas must be >= 1, got {num_replicas}")
        if rank < 0 or rank >= num_replicas:
            raise ValueError(
                f"rank must be in the range [0, {num_replicas}), got {rank}"
            )
        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = seed
        self.drop_last = drop_last
        self.epoch = 0

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
            # if floats are provided, we treat them as ratios
            if _contains_float(samples_per_dataset):
                normalised_weights = _normalise_weights(
                    samples_per_dataset, self.lengths
                )
                self.samples_per_dataset = _counts_from_weights(
                    normalised_weights, batch_size
                )
            else:
                exact_counts: list[int] = []
                for count in samples_per_dataset:
                    if isinstance(count, bool) or not isinstance(count, Integral):
                        raise TypeError(
                            "Integer samples_per_dataset entries must be "
                            f"integral counts, got {count!r}"
                        )
                    exact_counts.append(int(count))
                self.samples_per_dataset = exact_counts

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
                _num_batches_from_policy(
                    epoch_policy=epoch_policy,
                    lengths=self.lengths,
                    samples_per_dataset=self.samples_per_dataset,
                    batch_size=batch_size,
                    total_length=len(dataset),
                    replacement=True,
                )
                if num_batches is None
                else num_batches
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
                _num_batches_from_policy(
                    epoch_policy=epoch_policy,
                    lengths=self.lengths,
                    samples_per_dataset=self.samples_per_dataset,
                    batch_size=batch_size,
                    total_length=len(dataset),
                    replacement=False,
                )
                if num_batches is None
                else num_batches
            )
            if self.num_batches > max_complete_batches:
                raise ValueError(
                    "replacement=False supports at most "
                    f"{max_complete_batches} complete batches for the requested "
                    "per-dataset counts"
                )
        if self.num_batches < 1:
            raise ValueError(f"num_batches must be >= 1, got {self.num_batches}")

    @classmethod
    def balanced(
        cls,
        dataset: MultiDataset,
        *,
        batch_size: int,
        num_batches: int | None = None,
        epoch_policy: EpochPolicy = "dataset_size",
        replacement: bool = True,
        shuffle: bool = True,
        generator: torch.Generator | None = None,
        num_replicas: int | None = None,
        rank: int | None = None,
        distributed_manager: DistributedManager | None = None,
        seed: int = 0,
        drop_last: bool = False,
    ) -> Self:
        """Create a batch sampler with equal dataset-level sampling rates.

        Parameters
        ----------
        dataset : MultiDataset
            Dataset wrapper that defines child dataset offsets.
        batch_size : int
            Number of samples in each emitted batch.
        num_batches : int | None, default=None
            Number of batches per epoch.
        epoch_policy : {"dataset_size", "min_size", "max_size"}, default="dataset_size"
            Policy used to compute ``num_batches`` when it is not provided.
        replacement : bool, default=True
            Whether local samples may repeat within an epoch.
        shuffle : bool, default=True
            Randomize local sample order and sample order within each batch.
        generator : torch.Generator | None, default=None
            Optional random generator for reproducible sampling.
        num_replicas : int | None, default=None
            Number of distributed ranks.
        rank : int | None, default=None
            Rank for this sampler.
        distributed_manager : DistributedManager | None, default=None
            Optional distributed manager used to infer rank and world size.
        seed : int, default=0
            Base seed used for deterministic shuffling across epochs.
        drop_last : bool, default=False
            Drop tail batches to make the epoch evenly divisible across ranks.

        Returns
        -------
        Self
            Batch sampler with one equal relative weight per child dataset.
        """
        return cls(
            dataset,
            batch_size=batch_size,
            weights=[1.0] * len(dataset.datasets),
            num_batches=num_batches,
            epoch_policy=epoch_policy,
            replacement=replacement,
            shuffle=shuffle,
            generator=generator,
            num_replicas=num_replicas,
            rank=rank,
            distributed_manager=distributed_manager,
            seed=seed,
            drop_last=drop_last,
        )

    def _epoch_generator(self) -> torch.Generator | None:
        """Return the generator used for this epoch."""
        if self.generator is not None:
            return self.generator
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        return generator

    def _iter_global_batches(self) -> Iterator[list[int]]:
        """Yield the unsharded epoch of global-index batches."""
        generator = self._epoch_generator()
        if self.replacement:
            # Replacement batches can be generated independently, so stream
            # each batch without materializing the all-rank epoch.
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
                            **_generator_kwargs(generator),
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
                yield _shuffle_indices(batch, generator) if self.shuffle else batch
            return

        # Without replacement, build fixed local orders and consume them
        # cursor-style so each child dataset is exhausted predictably.
        local_orders = [
            _local_order(length, shuffle=self.shuffle, generator=generator)
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
            yield _shuffle_indices(batch, generator) if self.shuffle else batch

    def _global_batches(self) -> list[list[int]]:
        """Return the full unsharded epoch of global-index batches."""
        return list(self._iter_global_batches())

    def __iter__(self) -> Iterator[list[int]]:
        """Yield rank-local batches of global sample indices."""
        if self.num_replicas == 1:
            # Single-process runs can stream directly; no padding or rank
            # filtering is needed.
            yield from self._iter_global_batches()
            return

        num_samples = _num_sharded_items(
            self.num_batches, self.num_replicas, self.drop_last
        )
        total_size = num_samples * self.num_replicas
        padding_size = 0 if self.drop_last else total_size - self.num_batches
        # Cache only the prefix needed for DistributedSampler-style padding,
        # then rank-filter the streamed global batch order.
        prefix: list[list[int]] = []
        for ordinal, batch in enumerate(self._iter_global_batches()):
            if ordinal >= total_size:
                break
            if len(prefix) < padding_size:
                prefix.append(batch)
            if ordinal % self.num_replicas == self.rank:
                yield batch
        if not self.drop_last and prefix:
            for offset in range(padding_size):
                batch = prefix[offset % len(prefix)]
                ordinal = self.num_batches + offset
                if ordinal % self.num_replicas == self.rank:
                    yield batch

    def __len__(self) -> int:
        """Return the number of rank-local emitted batches."""
        return _num_sharded_items(self.num_batches, self.num_replicas, self.drop_last)

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch used for deterministic distributed shuffling.

        Parameters
        ----------
        epoch : int
            Epoch number added to ``seed`` when this sampler owns its generator.
        """
        self.epoch = epoch
