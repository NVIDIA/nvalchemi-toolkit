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
"""Tests for multidataset samplers."""

from __future__ import annotations

from collections.abc import Sequence

import pytest
import torch
from torch.utils.data import DistributedSampler

from nvalchemi.data.atomic_data import AtomicData
from nvalchemi.data.datapipes import (
    DataLoader,
    Dataset,
    DistributedSamplerProtocol,
    MultiDataset,
    MultiDatasetBatchSampler,
    MultiDatasetSampler,
)


def _make_ordered_atomic_data(label: int) -> AtomicData:
    """Create one-atom AtomicData with an order-identifying atomic number."""
    return AtomicData(
        atomic_numbers=torch.tensor([label], dtype=torch.long),
        positions=torch.tensor([[float(label), 0.0, 0.0]]),
        cell=torch.eye(3).unsqueeze(0),
        pbc=torch.tensor([[True, True, True]]),
    )


class _OrderedReadManyReader:
    """Minimal reader that records read_many calls for DataLoader tests."""

    def __init__(self, n: int = 5) -> None:
        self._n = n
        self.pin_memory = False

    def _load_sample(self, index: int) -> dict[str, torch.Tensor]:
        return _make_ordered_atomic_data(index + 1).to_dict()

    @property
    def field_names(self) -> list[str]:
        return list(self._load_sample(0)) if self._n > 0 else []

    def read_many(
        self, indices: Sequence[int]
    ) -> list[tuple[dict[str, torch.Tensor], dict[str, int]]]:
        return [(self._load_sample(index), {"src_index": index}) for index in indices]

    def __len__(self) -> int:
        return self._n

    def close(self) -> None:
        """Release reader resources."""


class _FakeDistributedManager:
    """Structural distributed manager for sampler tests."""

    def __init__(self, *, world_size: int, rank: int) -> None:
        self.world_size = world_size
        self.rank = rank
        self.initialized = True

    def is_initialized(self) -> bool:
        """Return whether the manager is initialized."""
        return self.initialized


def test_torch_distributed_sampler_satisfies_protocol() -> None:
    """Verify native PyTorch distributed samplers satisfy the shared protocol."""
    sampler = DistributedSampler(range(4), num_replicas=2, rank=0)

    assert isinstance(sampler, DistributedSamplerProtocol)


def test_multidataset_sampler_shards_across_distributed_ranks() -> None:
    """Verify regular multi-dataset sampling emits a rank-local shard."""
    dataset = MultiDataset(
        Dataset(_OrderedReadManyReader(n=3), device="cpu"),
        Dataset(_OrderedReadManyReader(n=3), device="cpu"),
    )
    rank0 = MultiDatasetSampler(
        dataset,
        num_replicas=2,
        rank=0,
        replacement=False,
        shuffle=False,
    )
    rank1 = MultiDatasetSampler(
        dataset,
        num_replicas=2,
        rank=1,
        replacement=False,
        shuffle=False,
    )

    assert isinstance(rank0, DistributedSamplerProtocol)
    assert len(rank0) == 3
    assert len(rank1) == 3
    assert list(rank0) == [0, 2, 4]
    assert list(rank1) == [1, 3, 5]


def test_multidataset_sampler_infers_rank_from_distributed_manager() -> None:
    """Verify distributed manager metadata configures sampler sharding."""
    dataset = MultiDataset(
        Dataset(_OrderedReadManyReader(n=3), device="cpu"),
        Dataset(_OrderedReadManyReader(n=3), device="cpu"),
    )
    manager = _FakeDistributedManager(world_size=2, rank=1)
    sampler = MultiDatasetSampler(
        dataset,
        distributed_manager=manager,
        replacement=False,
        shuffle=False,
    )

    assert sampler.num_replicas == 2
    assert sampler.rank == 1
    assert list(sampler) == [1, 3, 5]


def test_multidataset_sampler_set_epoch_changes_owned_shuffle() -> None:
    """Verify set_epoch changes deterministic shuffling when no generator is passed."""
    dataset = MultiDataset(
        Dataset(_OrderedReadManyReader(n=8), device="cpu"),
        Dataset(_OrderedReadManyReader(n=8), device="cpu"),
    )
    sampler = MultiDatasetSampler(
        dataset,
        num_samples=12,
        replacement=False,
        shuffle=True,
        seed=17,
    )

    epoch0 = list(sampler)
    assert epoch0 == list(sampler)

    sampler.set_epoch(1)

    assert list(sampler) != epoch0


def test_multidataset_batch_sampler_pads_streamed_batches() -> None:
    """Streaming batch sharding preserves padded DistributedSampler semantics."""
    dataset = MultiDataset(
        Dataset(_OrderedReadManyReader(n=2), device="cpu"),
        Dataset(_OrderedReadManyReader(n=2), device="cpu"),
    )
    rank_batches = []
    for rank in range(4):
        sampler = MultiDatasetBatchSampler(
            dataset,
            batch_size=2,
            samples_per_dataset=[1, 1],
            num_batches=1,
            num_replicas=4,
            rank=rank,
            replacement=False,
            shuffle=False,
        )
        rank_batches.append(list(sampler))

    assert rank_batches == [[[0, 2]], [[0, 2]], [[0, 2]], [[0, 2]]]


def test_multidataset_batch_sampler_shards_batches_across_distributed_ranks() -> None:
    """Verify multi-dataset batch sampling shards whole batches by rank."""
    dataset = MultiDataset(
        Dataset(_OrderedReadManyReader(n=6), device="cpu"),
        Dataset(_OrderedReadManyReader(n=6), device="cpu"),
    )
    rank0 = MultiDatasetBatchSampler(
        dataset,
        batch_size=4,
        samples_per_dataset=[2, 2],
        num_batches=3,
        num_replicas=2,
        rank=0,
        replacement=False,
        shuffle=False,
    )
    rank1 = MultiDatasetBatchSampler(
        dataset,
        batch_size=4,
        samples_per_dataset=[2, 2],
        num_batches=3,
        num_replicas=2,
        rank=1,
        replacement=False,
        shuffle=False,
    )

    assert isinstance(rank0, DistributedSamplerProtocol)
    assert len(rank0) == 2
    assert len(rank1) == 2
    assert list(rank0) == [[0, 1, 6, 7], [4, 5, 10, 11]]
    assert list(rank1) == [[2, 3, 8, 9], [0, 1, 6, 7]]


def test_multidataset_sampler_uses_custom_rates_without_replacement() -> None:
    """Verify regular MultiDataset sampling emits global indices at given rates."""
    dataset = MultiDataset(
        Dataset(_OrderedReadManyReader(n=3), device="cpu"),
        Dataset(_OrderedReadManyReader(n=8), device="cpu"),
    )
    sampler = MultiDatasetSampler(
        dataset,
        weights=[1.0, 3.0],
        num_samples=8,
        replacement=False,
        shuffle=False,
    )

    indices = list(sampler)

    assert indices == [0, 1, 3, 4, 5, 6, 7, 8]
    assert [dataset.to_local_index(index)[0] for index in indices] == [
        0,
        0,
        1,
        1,
        1,
        1,
        1,
        1,
    ]


def test_balanced_multidataset_batch_sampler_forms_balanced_batches() -> None:
    """Verify balanced batches include equal samples from each child dataset."""
    dataset = MultiDataset(
        Dataset(_OrderedReadManyReader(n=4), device="cpu"),
        Dataset(_OrderedReadManyReader(n=6), device="cpu"),
    )
    sampler = MultiDatasetBatchSampler.balanced(
        dataset,
        batch_size=4,
        num_batches=2,
        replacement=False,
        shuffle=False,
    )

    assert list(sampler) == [[0, 1, 4, 5], [2, 3, 6, 7]]

    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        prefetch_factor=0,
        use_streams=False,
    )
    batches = list(loader)

    assert [batch.atomic_numbers.tolist() for batch in batches] == [
        [1, 2, 1, 2],
        [3, 4, 3, 4],
    ]


def test_weighted_multidataset_batch_sampler_uses_dataset_rates() -> None:
    """Verify weighted batch sampling allocates batch slots by dataset rate."""
    dataset = MultiDataset(
        Dataset(_OrderedReadManyReader(n=8), device="cpu"),
        Dataset(_OrderedReadManyReader(n=8), device="cpu"),
    )
    sampler = MultiDatasetBatchSampler(
        dataset,
        batch_size=5,
        weights=[4.0, 1.0],
        num_batches=2,
        replacement=False,
        shuffle=False,
    )

    assert sampler.samples_per_dataset == [4, 1]
    assert list(sampler) == [[0, 1, 2, 3, 8], [4, 5, 6, 7, 9]]


def test_samples_per_dataset_floats_are_relative_rates() -> None:
    """Verify float samples_per_dataset entries allocate by relative ratio."""
    dataset = MultiDataset(
        Dataset(_OrderedReadManyReader(n=8), device="cpu"),
        Dataset(_OrderedReadManyReader(n=8), device="cpu"),
    )
    sampler = MultiDatasetBatchSampler(
        dataset,
        batch_size=8,
        samples_per_dataset=[1.0, 3.0],
        num_batches=1,
        replacement=False,
        shuffle=False,
    )

    assert sampler.samples_per_dataset == [2, 6]
    assert list(sampler) == [[0, 1, 8, 9, 10, 11, 12, 13]]


def test_batch_sampler_min_size_epoch_policy_stops_at_smallest_dataset() -> None:
    """Verify min_size avoids oversampling smaller contributing datasets."""
    dataset = MultiDataset(
        Dataset(_OrderedReadManyReader(n=2), device="cpu"),
        Dataset(_OrderedReadManyReader(n=6), device="cpu"),
    )
    sampler = MultiDatasetBatchSampler.balanced(
        dataset,
        batch_size=4,
        epoch_policy="min_size",
        replacement=True,
        shuffle=False,
    )

    assert len(sampler) == 1
    assert list(sampler) == [[0, 1, 2, 3]]


def test_batch_sampler_max_size_epoch_policy_oversamples_smaller_dataset() -> None:
    """Verify max_size can balance batches across the largest dataset span."""
    dataset = MultiDataset(
        Dataset(_OrderedReadManyReader(n=2), device="cpu"),
        Dataset(_OrderedReadManyReader(n=6), device="cpu"),
    )
    sampler = MultiDatasetBatchSampler.balanced(
        dataset,
        batch_size=4,
        epoch_policy="max_size",
        replacement=True,
        shuffle=False,
    )

    assert len(sampler) == 3
    assert list(sampler) == [
        [0, 1, 2, 3],
        [0, 1, 4, 5],
        [0, 1, 6, 7],
    ]


def test_batch_sampler_max_size_epoch_policy_requires_replacement() -> None:
    """Verify max_size fails without replacement when oversampling is required."""
    dataset = MultiDataset(
        Dataset(_OrderedReadManyReader(n=2), device="cpu"),
        Dataset(_OrderedReadManyReader(n=6), device="cpu"),
    )

    with pytest.raises(ValueError, match="replacement=True"):
        MultiDatasetBatchSampler.balanced(
            dataset,
            batch_size=4,
            epoch_policy="max_size",
            replacement=False,
            shuffle=False,
        )
