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

from nvalchemi.data.atomic_data import AtomicData
from nvalchemi.data.datapipes import (
    BalancedMultiDatasetBatchSampler,
    DataLoader,
    Dataset,
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
    sampler = BalancedMultiDatasetBatchSampler(
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
    sampler = BalancedMultiDatasetBatchSampler(
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
    sampler = BalancedMultiDatasetBatchSampler(
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
        BalancedMultiDatasetBatchSampler(
            dataset,
            batch_size=4,
            epoch_policy="max_size",
            replacement=False,
            shuffle=False,
        )
