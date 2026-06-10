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
"""Tests for per-batch ``batch_transforms`` in :class:`DataLoader`.

These tests mirror the in-memory stub-reader pattern from
``test_dataset_transforms.py`` to exercise the batch-level transform
hook added in step 3 of the dataset-transforms feature.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pytest
import torch

from nvalchemi._typing import BatchTransform
from nvalchemi.data.atomic_data import AtomicData
from nvalchemi.data.batch import Batch
from nvalchemi.data.datapipes.dataloader import DataLoader
from nvalchemi.data.datapipes.dataset import Dataset


class _StubReader:
    """Minimal in-memory reader satisfying :class:`ReaderProtocol`.

    Parameters
    ----------
    samples : list[dict[str, torch.Tensor]]
        Raw tensor dicts, each one compatible with
        :meth:`AtomicData.model_validate`.
    """

    def __init__(self, samples: list[dict[str, torch.Tensor]]) -> None:
        self._samples = samples

    def __len__(self) -> int:
        """Return the number of held samples."""
        return len(self._samples)

    def _load_sample(self, index: int) -> dict[str, torch.Tensor]:
        """Return a shallow copy of the sample dict at ``index``."""
        return dict(self._samples[index])

    def _get_sample_metadata(self, index: int) -> dict[str, Any]:
        """Return a fresh metadata dict tagged with the index."""
        return {"index": index}

    def close(self) -> None:
        """No-op; stub owns no resources."""


def _make_samples(n_samples: int = 4) -> list[dict[str, torch.Tensor]]:
    """Build a deterministic list of small AtomicData-compatible dicts."""
    torch.manual_seed(0)
    sizes = [3, 5, 4, 6, 2, 7, 3, 5]
    return [
        {
            "positions": torch.randn(sizes[i % len(sizes)], 3),
            "atomic_numbers": torch.ones(sizes[i % len(sizes)], dtype=torch.long),
        }
        for i in range(n_samples)
    ]


# -- Named module-level transforms.


def _tag_batch(batch: Batch) -> Batch:
    """Mark ``batch`` by setting ``batch.transformed = True``."""
    batch.transformed = True
    return batch


def _scale_positions_batch(batch: Batch) -> Batch:
    """Scale the batch's positions tensor by 2.0 in place and return it."""
    batch.positions = batch.positions * 2.0
    return batch


def _make_batch_appender(tag: str) -> BatchTransform:
    """Return a transform that appends ``tag`` to ``batch.order``.

    Parameters
    ----------
    tag : str
        Marker to append.

    Returns
    -------
    BatchTransform
        A transform suitable for :class:`~nvalchemi.data.transforms.Compose`.
    """

    def _append(batch: Batch) -> Batch:
        if not hasattr(batch, "order"):
            batch.order = []
        batch.order.append(tag)
        return batch

    _append.__name__ = f"_batch_append_{tag}"
    return _append


class _BoomError(RuntimeError):
    """Sentinel error raised by :func:`_raise_on_second_batch`."""


class _CounterTransform:
    """Batch transform that raises on its Nth invocation.

    Parameters
    ----------
    raise_on : int
        One-indexed call count at which :class:`_BoomError` is raised.
    """

    def __init__(self, raise_on: int) -> None:
        self.raise_on = raise_on
        self.calls = 0

    def __call__(self, batch: Batch) -> Batch:
        """Increment the call counter, raising on the configured call."""
        self.calls += 1
        if self.calls == self.raise_on:
            raise _BoomError(f"boom on call {self.calls}")
        return batch


def _tag_metadata_sample(
    data: AtomicData, metadata: dict[str, Any]
) -> tuple[AtomicData, dict[str, Any]]:
    """Per-sample transform adding ``metadata['sample_tag'] = True``."""
    metadata["sample_tag"] = True
    return data, metadata


def _tag_batch_after_sample(batch: Batch) -> Batch:
    """Per-batch transform adding ``batch.batch_tag = True``."""
    batch.batch_tag = True
    return batch


class TestDataLoaderBatchTransforms:
    """Exercise per-batch transforms in :class:`DataLoader`."""

    def setup_method(self) -> None:
        """Create a fresh stub reader and dataset for each test."""
        self.samples = _make_samples(n_samples=4)
        self.reader = _StubReader(self.samples)
        self.dataset = Dataset(self.reader, device="cpu")

    def _fresh_dataset(self) -> Dataset:
        """Return an independent CPU ``Dataset`` over fresh stub samples."""
        return Dataset(_StubReader(_make_samples(n_samples=4)), device="cpu")

    @pytest.mark.parametrize("batch_transforms", [None, []])
    def test_batch_transforms_disabled(
        self, batch_transforms: Sequence[BatchTransform] | None
    ) -> None:
        """``None`` and ``[]`` both short-circuit ``_batch_transform`` to ``None``."""
        loader = DataLoader(
            self.dataset,
            batch_size=2,
            use_streams=False,
            batch_transforms=batch_transforms,
        )
        baseline = DataLoader(
            self._fresh_dataset(),
            batch_size=2,
            use_streams=False,
        )

        assert loader._batch_transform is None
        assert baseline._batch_transform is None

        for b_loader, b_baseline in zip(loader, baseline, strict=True):
            assert torch.equal(b_loader.positions, b_baseline.positions)
            assert torch.equal(b_loader.atomic_numbers, b_baseline.atomic_numbers)

    def test_single_batch_transform_simple_path(self) -> None:
        """A single batch transform is applied to each yielded batch."""
        loader = DataLoader(
            self.dataset,
            batch_size=2,
            use_streams=False,
            batch_transforms=[_tag_batch],
        )

        batches = list(loader)
        assert len(batches) == 2
        for batch in batches:
            assert getattr(batch, "transformed", False) is True

    def test_single_batch_transform_modifies_tensor(self) -> None:
        """A transform mutating positions produces the expected values."""
        baseline = DataLoader(
            self._fresh_dataset(),
            batch_size=2,
            use_streams=False,
        )
        transformed = DataLoader(
            self._fresh_dataset(),
            batch_size=2,
            use_streams=False,
            batch_transforms=[_scale_positions_batch],
        )
        for b_raw, b_scaled in zip(baseline, transformed, strict=True):
            assert torch.equal(b_scaled.positions, b_raw.positions * 2.0)

    def test_multiple_batch_transforms_compose_order(self) -> None:
        """Three transforms run left-to-right on each yielded batch."""
        loader = DataLoader(
            self.dataset,
            batch_size=2,
            use_streams=False,
            batch_transforms=[
                _make_batch_appender("a"),
                _make_batch_appender("b"),
                _make_batch_appender("c"),
            ],
        )
        for batch in loader:
            assert batch.order == ["a", "b", "c"]

    def test_batch_transform_raises_propagates(self) -> None:
        """Transform errors surface as ``RuntimeError`` with ``__cause__``."""
        transform = _CounterTransform(raise_on=2)
        loader = DataLoader(
            self.dataset,
            batch_size=2,
            use_streams=False,
            batch_transforms=[transform],
        )

        iterator = iter(loader)
        # First batch should succeed.
        next(iterator)
        # Second batch raises — Compose wraps in RuntimeError; original is __cause__.
        with pytest.raises(RuntimeError) as excinfo:
            next(iterator)
        assert isinstance(excinfo.value.__cause__, _BoomError)
        # Compose prefixes errors with "transform[<i>]" — see Compose.__call__.
        assert "transform[0]" in str(excinfo.value)

    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="CUDA required to exercise the prefetch iteration path",
    )
    def test_batch_transform_prefetch_path(self) -> None:
        """The prefetch path also applies the batch transform."""
        dataset_cuda = Dataset(self.reader, device="cuda")
        loader = DataLoader(
            dataset_cuda,
            batch_size=2,
            use_streams=True,
            prefetch_factor=2,
            batch_transforms=[_tag_batch],
        )

        batches = list(loader)
        assert len(batches) == 2
        for batch in batches:
            assert getattr(batch, "transformed", False) is True

    def test_combined_sample_and_batch_transforms(self) -> None:
        """Per-sample and per-batch transforms both run, in their own paths."""
        dataset_with_sample_tx = Dataset(
            self.reader,
            device="cpu",
            transforms=[_tag_metadata_sample],
        )
        loader = DataLoader(
            dataset_with_sample_tx,
            batch_size=2,
            use_streams=False,
            batch_transforms=[_tag_batch_after_sample],
        )

        batches = list(loader)
        assert len(batches) == 2
        for batch in batches:
            assert getattr(batch, "batch_tag", False) is True

        # Sanity: fetching a sample directly exposes the per-sample metadata tag.
        _, metadata = dataset_with_sample_tx[0]
        assert metadata.get("sample_tag") is True
