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
"""Tests for per-sample transforms in :class:`nvalchemi.data.Dataset`.

These tests use an in-memory stub reader (not zarr) to exercise the
transform hook points added in step 2 of the dataset-transforms feature.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch

from nvalchemi._typing import SampleTransform
from nvalchemi.data.atomic_data import AtomicData
from nvalchemi.data.datapipes.dataset import Dataset
from nvalchemi.data.transforms import Compose


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
        return None


def _make_samples() -> list[dict[str, torch.Tensor]]:
    """Build a deterministic pair of small AtomicData-compatible dicts."""
    torch.manual_seed(0)
    return [
        {
            "positions": torch.randn(n, 3),
            "atomic_numbers": torch.ones(n, dtype=torch.long),
        }
        for n in (3, 5)
    ]


# -- Named module-level transforms (avoid lambdas so pickle/thread paths stay
#    well-behaved even though current Compose does not require picklability).


def _shift_positions(
    data: AtomicData, metadata: dict[str, Any]
) -> tuple[AtomicData, dict[str, Any]]:
    """Return ``data`` with ``positions`` shifted by +1.0."""
    new_positions = data.positions + 1.0
    return (
        AtomicData(positions=new_positions, atomic_numbers=data.atomic_numbers),
        metadata,
    )


def _scale_positions(
    data: AtomicData, metadata: dict[str, Any]
) -> tuple[AtomicData, dict[str, Any]]:
    """Return ``data`` with ``positions`` scaled by 2.0."""
    new_positions = data.positions * 2.0
    return (
        AtomicData(positions=new_positions, atomic_numbers=data.atomic_numbers),
        metadata,
    )


def _make_appender(tag: str) -> SampleTransform:
    """Return a transform that appends ``tag`` to ``metadata['order']``.

    Parameters
    ----------
    tag : str
        Marker to append.

    Returns
    -------
    SampleTransform
        A transform suitable for :class:`~nvalchemi.data.transforms.Compose`.
    """

    def _append(
        data: AtomicData, metadata: dict[str, Any]
    ) -> tuple[AtomicData, dict[str, Any]]:
        metadata.setdefault("order", []).append(tag)
        return data, metadata

    _append.__name__ = f"_append_{tag}"
    return _append


def _tag_metadata(
    data: AtomicData, metadata: dict[str, Any]
) -> tuple[AtomicData, dict[str, Any]]:
    """Add a ``'tagged'`` flag to ``metadata`` and return it."""
    metadata["tagged"] = True
    return data, metadata


class _BoomError(RuntimeError):
    """Sentinel error raised by :func:`_raise_boom`."""


def _raise_boom(
    data: AtomicData, metadata: dict[str, Any]
) -> tuple[AtomicData, dict[str, Any]]:
    """Always raise :class:`_BoomError` to test error propagation."""
    raise _BoomError("boom")


class TestDatasetTransforms:
    """Per-sample transforms in :class:`Dataset`."""

    def setup_method(self) -> None:
        """Create a fresh stub reader for each test."""
        self.samples = _make_samples()
        self.reader = _StubReader(self.samples)

    @pytest.mark.parametrize("transforms", [None, []])
    def test_none_or_empty_transforms_is_passthrough(
        self, transforms: list | None
    ) -> None:
        """``None`` and ``[]`` both short-circuit ``_sample_transform`` to ``None``."""
        ds = Dataset(self.reader, device="cpu", transforms=transforms)
        ds_default = Dataset(_StubReader(_make_samples()), device="cpu")

        assert ds._sample_transform is None
        assert ds_default._sample_transform is None

        data_a, meta_a = ds[0]
        data_b, meta_b = ds_default[0]
        assert torch.equal(data_a.positions, data_b.positions)
        assert torch.equal(data_a.atomic_numbers, data_b.atomic_numbers)
        assert set(meta_a.keys()) == set(meta_b.keys())

    def test_single_sample_transform_applied_sync_path(self) -> None:
        """A single shift transform is applied on the sync fallback path."""
        ds = Dataset(self.reader, device="cpu", transforms=[_shift_positions])

        assert isinstance(ds._sample_transform, Compose)
        expected = self.samples[0]["positions"] + 1.0
        data, _ = ds[0]
        assert torch.equal(data.positions, expected)

    def test_three_transforms_composed_left_to_right_sync_path(self) -> None:
        """Three transforms run in the order they were supplied."""
        ds = Dataset(
            self.reader,
            device="cpu",
            transforms=[_make_appender("a"), _make_appender("b"), _make_appender("c")],
        )
        _, metadata = ds[0]
        assert metadata["order"] == ["a", "b", "c"]

    def test_metadata_mutation_propagates(self) -> None:
        """Metadata mutations from a transform are visible to the caller."""
        ds = Dataset(self.reader, device="cpu", transforms=[_tag_metadata])
        _, metadata = ds[0]
        assert metadata.get("tagged") is True
        assert metadata["index"] == 0

    @pytest.mark.parametrize("prefetch", [False, True], ids=["sync", "prefetch"])
    def test_transform_exception_propagates(self, prefetch: bool) -> None:
        """Transform errors surface as ``RuntimeError`` (with ``__cause__``).

        Covers both the synchronous ``__getitem__`` path and the worker-thread
        prefetch path; ``Compose`` wraps the original exception in a
        ``RuntimeError`` and ``_load_and_transform`` re-raises it unchanged.
        """
        ds = Dataset(self.reader, device="cpu", transforms=[_raise_boom])
        if prefetch:
            ds.prefetch(0)
        with pytest.raises(RuntimeError) as excinfo:
            ds[0]
        assert isinstance(excinfo.value.__cause__, _BoomError)
        ds.close()

    def test_prefetch_path_applies_transform_cpu(self) -> None:
        """CPU prefetch path also runs the transform (no CUDA stream)."""
        ds = Dataset(self.reader, device="cpu", transforms=[_shift_positions])

        ds.prefetch(0)
        data, _ = ds[0]

        expected = self.samples[0]["positions"] + 1.0
        assert torch.equal(data.positions, expected)

        ds.close()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
    def test_cuda_prefetch_applies_transform(self) -> None:
        """Transform runs inside the prefetch CUDA stream; data lands on GPU.

        No external ``torch.cuda.synchronize()`` — the dataset's own
        ``event.synchronize()`` must be sufficient; any regression in the
        ``event.record(stream)`` ordering will surface as a data race here.
        """
        ds = Dataset(self.reader, device="cuda", transforms=[_scale_positions])

        stream = torch.cuda.Stream()
        ds.prefetch(0, stream=stream)
        data, _ = ds[0]

        assert data.positions.device.type == "cuda"
        expected = (self.samples[0]["positions"] * 2.0).to(data.positions.device)
        assert torch.equal(data.positions, expected)

        ds.close()

    @pytest.mark.parametrize(
        "bad_transforms",
        [_shift_positions, (x for x in [_shift_positions])],
        ids=["single_callable", "generator"],
    )
    def test_non_sequence_transforms_raises_type_error(
        self, bad_transforms: object
    ) -> None:
        """A single callable or a generator is rejected with ``TypeError``."""
        with pytest.raises(TypeError, match="Sequence"):
            Dataset(self.reader, device="cpu", transforms=bad_transforms)  # type: ignore[arg-type]
