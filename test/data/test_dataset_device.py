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
"""Tests for the device helpers in :mod:`nvalchemi.data.datapipes.dataset`."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from nvalchemi.data.atomic_data import AtomicData
from nvalchemi.data.batch import Batch
from nvalchemi.data.datapipes import (
    AtomicDataZarrReader,
    AtomicDataZarrWriter,
    Dataset,
    InMemoryDataset,
    MultiDataset,
    dataset_device,
    same_device,
)


def _make_batch(num_systems: int = 2, device: str = "cpu") -> Batch:
    """Return a small batch of two-atom systems on *device*."""
    generator = torch.Generator().manual_seed(0)
    items = [
        AtomicData(
            positions=torch.randn(2, 3, generator=generator),
            atomic_numbers=torch.ones(2, dtype=torch.long),
        )
        for _ in range(num_systems)
    ]
    return Batch.from_data_list(items).to(device)


def _make_zarr_dataset(path: Path, device: str) -> Dataset:
    """Write a two-system store under *path* and open it as a dataset on *device*."""
    AtomicDataZarrWriter(path / "store.zarr").write(_make_batch())
    return Dataset(AtomicDataZarrReader(path / "store.zarr"), device=device)


class TestDatasetDevice:
    """Resolving the device a dataset emits its batches on."""

    def test_an_in_memory_dataset_reports_its_resident_batch_device(self) -> None:
        """An in-memory dataset without a target device is measured by its batch."""
        dataset = InMemoryDataset(in_memory_batch=_make_batch())

        assert dataset_device(dataset) == torch.device("cpu")

    def test_a_declared_device_is_taken_without_drawing_a_batch(
        self, tmp_path: Path
    ) -> None:
        """A Zarr dataset opened on an indexed device declares it; no probe is drawn."""
        dataset = _make_zarr_dataset(tmp_path, device="cpu")

        with patch.object(dataset, "load_batches") as load_batches:
            assert dataset_device(dataset) == torch.device("cpu")
        load_batches.assert_not_called()

    def test_a_composed_dataset_is_measured_by_a_probe(self) -> None:
        """A MultiDataset declares no device, so the batch it emits is read instead."""
        multi = MultiDataset(
            InMemoryDataset(in_memory_batch=_make_batch()),
            InMemoryDataset(in_memory_batch=_make_batch(1)),
        )

        assert dataset_device(multi) == torch.device("cpu")

    def test_a_supplied_probe_replaces_the_drawn_batch(self) -> None:
        """A probe already in hand is read instead of loading another batch."""
        multi = MultiDataset(InMemoryDataset(in_memory_batch=_make_batch()))

        with patch.object(multi, "load_batches") as load_batches:
            assert dataset_device(multi, probe=_make_batch()) == torch.device("cpu")
        load_batches.assert_not_called()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_an_index_less_cuda_declaration_is_resolved_by_a_probe(
        self, tmp_path: Path
    ) -> None:
        """``cuda`` names whichever device is current, so the emitted index is read."""
        dataset = _make_zarr_dataset(tmp_path, device="cuda")

        resolved = dataset_device(dataset)

        assert dataset.target_device == torch.device("cuda")
        assert resolved == torch.device("cuda", torch.cuda.current_device())


class TestSameDevice:
    """Comparing two emission devices for collation."""

    def test_two_indexed_devices_of_one_type_must_match(self) -> None:
        """cuda:0 and cuda:1 concatenate no better than a host and a device tensor."""
        assert not same_device(torch.device("cuda:0"), torch.device("cuda:1"))
        assert same_device(torch.device("cuda:1"), torch.device("cuda:1"))

    def test_an_index_less_device_is_compared_by_type(self) -> None:
        """``cuda`` names whichever device is current, so it matches an indexed one."""
        assert same_device(torch.device("cuda"), torch.device("cuda:0"))
        assert not same_device(torch.device("cuda"), torch.device("cpu"))

    def test_a_missing_device_is_no_constraint(self) -> None:
        """``None`` on either side matches anything."""
        assert same_device(None, torch.device("cuda:1"))
        assert same_device(torch.device("cpu"), None)
