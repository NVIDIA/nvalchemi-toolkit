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
"""Tests for segmented reduction wrappers (GPU-only).

These tests compare the custom-op implementations against naive
PyTorch scatter/loop equivalents on CUDA tensors.
"""

from __future__ import annotations

import pytest
import torch

_CUDA = torch.cuda.is_available()
pytestmark = pytest.mark.skipif(not _CUDA, reason="CUDA required")


def _naive_segmented_sum(
    values: torch.Tensor, idx: torch.Tensor, num_segments: int
) -> torch.Tensor:
    out = torch.zeros(num_segments, device=values.device, dtype=values.dtype)
    out.scatter_add_(0, idx.long(), values)
    return out


def _naive_segmented_mean(
    values: torch.Tensor, idx: torch.Tensor, num_segments: int
) -> torch.Tensor:
    sums = _naive_segmented_sum(values, idx, num_segments)
    counts = _naive_segmented_sum(torch.ones_like(values), idx, num_segments)
    return sums / counts.clamp(min=1)


class TestSegmentedSum:
    """Tests for ``_segmented_sum`` custom op."""

    def test_basic(self) -> None:
        from nvalchemi.training.losses._reductions import _segmented_sum

        values = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], device="cuda")
        idx = torch.tensor([0, 0, 0, 1, 1], device="cuda", dtype=torch.long)
        result = _segmented_sum(values, idx, 2)
        expected = _naive_segmented_sum(values, idx, 2)
        torch.testing.assert_close(result, expected)

    def test_single_segment(self) -> None:
        from nvalchemi.training.losses._reductions import _segmented_sum

        values = torch.tensor([1.0, 2.0, 3.0], device="cuda")
        idx = torch.zeros(3, device="cuda", dtype=torch.long)
        result = _segmented_sum(values, idx, 1)
        assert result.shape == (1,)
        torch.testing.assert_close(result, torch.tensor([6.0], device="cuda"))

    def test_unequal_segments(self) -> None:
        from nvalchemi.training.losses._reductions import _segmented_sum

        values = torch.arange(10.0, device="cuda")
        idx = torch.tensor(
            [0, 0, 0, 0, 0, 1, 1, 2, 2, 2], device="cuda", dtype=torch.long
        )
        result = _segmented_sum(values, idx, 3)
        expected = _naive_segmented_sum(values, idx, 3)
        torch.testing.assert_close(result, expected)


class TestSegmentedMean:
    """Tests for ``_segmented_mean`` custom op."""

    def test_basic(self) -> None:
        from nvalchemi.training.losses._reductions import _segmented_mean

        values = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], device="cuda")
        idx = torch.tensor([0, 0, 0, 1, 1], device="cuda", dtype=torch.long)
        result = _segmented_mean(values, idx, 2)
        expected = _naive_segmented_mean(values, idx, 2)
        torch.testing.assert_close(result, expected)

    def test_single_element_segments(self) -> None:
        from nvalchemi.training.losses._reductions import _segmented_mean

        values = torch.tensor([10.0, 20.0, 30.0], device="cuda")
        idx = torch.tensor([0, 1, 2], device="cuda", dtype=torch.long)
        result = _segmented_mean(values, idx, 3)
        torch.testing.assert_close(result, values)


class TestSegmentedMSE:
    """Tests for ``segmented_mse``."""

    def test_sum_reduction(self) -> None:
        from nvalchemi.training.losses._reductions import segmented_mse

        pred = torch.tensor([1.0, 2.0, 3.0, 4.0], device="cuda")
        target = torch.tensor([0.0, 0.0, 0.0, 0.0], device="cuda")
        idx = torch.tensor([0, 0, 1, 1], device="cuda", dtype=torch.long)

        result = segmented_mse(pred, target, idx, 2, reduction="sum")
        # seg 0: 1^2 + 2^2 = 5, seg 1: 3^2 + 4^2 = 25
        expected = torch.tensor([5.0, 25.0], device="cuda")
        torch.testing.assert_close(result, expected)

    def test_mean_reduction(self) -> None:
        from nvalchemi.training.losses._reductions import segmented_mse

        pred = torch.tensor([1.0, 2.0, 3.0, 4.0], device="cuda")
        target = torch.zeros(4, device="cuda")
        idx = torch.tensor([0, 0, 1, 1], device="cuda", dtype=torch.long)

        result = segmented_mse(pred, target, idx, 2, reduction="mean")
        # seg 0: (1+4)/2=2.5, seg 1: (9+16)/2=12.5
        expected = torch.tensor([2.5, 12.5], device="cuda")
        torch.testing.assert_close(result, expected)


class TestSegmentedMAE:
    """Tests for ``segmented_mae``."""

    def test_sum_reduction(self) -> None:
        from nvalchemi.training.losses._reductions import segmented_mae

        pred = torch.tensor([1.0, -2.0, 3.0], device="cuda")
        target = torch.zeros(3, device="cuda")
        idx = torch.tensor([0, 0, 1], device="cuda", dtype=torch.long)

        result = segmented_mae(pred, target, idx, 2, reduction="sum")
        # seg 0: |1|+|-2| = 3, seg 1: |3| = 3
        expected = torch.tensor([3.0, 3.0], device="cuda")
        torch.testing.assert_close(result, expected)

    def test_mean_reduction(self) -> None:
        from nvalchemi.training.losses._reductions import segmented_mae

        pred = torch.tensor([1.0, -2.0, 3.0], device="cuda")
        target = torch.zeros(3, device="cuda")
        idx = torch.tensor([0, 0, 1], device="cuda", dtype=torch.long)

        result = segmented_mae(pred, target, idx, 2, reduction="mean")
        # seg 0: 3/2=1.5, seg 1: 3/1=3
        expected = torch.tensor([1.5, 3.0], device="cuda")
        torch.testing.assert_close(result, expected)
