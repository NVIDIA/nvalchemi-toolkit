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
"""Tests for :class:`StressLoss`."""

from __future__ import annotations

from collections import OrderedDict
from unittest.mock import patch

import torch
from torch import Tensor

from nvalchemi.training.losses.stress import StressLoss
from test.training.conftest import _make_batch


def _cpu_segmented_sum(values: Tensor, idx: Tensor, num_segments: int) -> Tensor:
    out = torch.zeros(num_segments, device=values.device, dtype=values.dtype)
    out.scatter_add_(0, idx.long(), values)
    return out


_PATCH = "nvalchemi.training.losses._base._segmented_sum"


class TestStressLoss:
    """Known-value tests for StressLoss."""

    def test_defaults(self) -> None:
        """Default attributes are correct."""
        loss = StressLoss()
        assert loss.level == "system"
        assert loss.pred_key == "stresses"
        assert loss.target_key == "stresses"
        assert loss.normalize_by_atoms is False

    def test_known_values(self, two_graph_batch) -> None:
        """Hand-computed stress loss on the fixture batch."""
        batch = two_graph_batch
        outputs = batch._outputs

        loss = StressLoss()
        with patch(_PATCH, side_effect=_cpu_segmented_sum):
            result = loss(batch, outputs)

        # stress_pred = 0.1 * ones(2,3,3), target = zeros(2,3,3)
        # diff^2 = 0.01 for each of 9 elements -> per graph = 0.09
        # mean -> 0.09
        assert result is not None
        torch.testing.assert_close(result, torch.tensor(0.09))

    def test_normalize_by_atoms(self, two_graph_batch) -> None:
        """Stress loss with atom-count normalization."""
        batch = two_graph_batch
        outputs = batch._outputs

        loss = StressLoss(normalize_by_atoms=True)
        with patch(_PATCH, side_effect=_cpu_segmented_sum):
            result = loss(batch, outputs)

        # per_graph = [0.09, 0.09], atoms = [3, 2]
        # normalized = [0.09/3, 0.09/2] = [0.03, 0.045]
        # mean -> (0.03 + 0.045) / 2 = 0.0375
        assert result is not None
        torch.testing.assert_close(result, torch.tensor(0.0375))

    def test_sum_reduction(self, two_graph_batch) -> None:
        """Stress loss with sum reduction."""
        batch = two_graph_batch
        outputs = batch._outputs

        loss = StressLoss(reduction="sum")
        with patch(_PATCH, side_effect=_cpu_segmented_sum):
            result = loss(batch, outputs)

        # per_graph = [0.09, 0.09], sum -> 0.18
        assert result is not None
        torch.testing.assert_close(result, torch.tensor(0.18))

    def test_varying_stress_values(self, make_batch) -> None:
        """Non-uniform stress tensors with different graphs."""
        B = 3
        stress_target = torch.zeros(B, 3, 3)
        stress_pred = torch.zeros(B, 3, 3)
        # Graph 0: diff of 1.0 in (0,0) only -> frobenius sq = 1.0
        stress_pred[0, 0, 0] = 1.0
        # Graph 1: diff of 0.5 everywhere -> frobenius sq = 9 * 0.25 = 2.25
        stress_pred[1] = 0.5
        # Graph 2: no diff -> 0
        data = {"stresses": stress_target}
        batch = make_batch([2, 3, 1], data=data)
        outputs = OrderedDict(stresses=stress_pred)

        loss = StressLoss()
        with patch(_PATCH, side_effect=_cpu_segmented_sum):
            result = loss(batch, outputs)

        # per_graph = [1.0, 2.25, 0.0], mean -> 3.25/3
        assert result is not None
        torch.testing.assert_close(result, torch.tensor(3.25 / 3.0))

    def test_weight(self, two_graph_batch) -> None:
        """Weight is applied to the result."""
        batch = two_graph_batch
        outputs = batch._outputs

        loss = StressLoss(weight=100.0)
        with patch(_PATCH, side_effect=_cpu_segmented_sum):
            result = loss(batch, outputs)

        assert result is not None
        torch.testing.assert_close(result, torch.tensor(0.09 * 100.0))

    def test_known_values_on_device(self, device) -> None:
        """StressLoss produces correct results on the parametrized device."""
        B = 2
        stress_target = torch.zeros(B, 3, 3, device=device)
        stress_pred = torch.ones(B, 3, 3, device=device) * 0.1
        batch = _make_batch([3, 2], device=device, data={"stresses": stress_target})
        outputs = OrderedDict(stresses=stress_pred)

        loss = StressLoss()
        with patch(_PATCH, side_effect=_cpu_segmented_sum):
            result = loss(batch, outputs)

        assert result is not None
        assert str(result.device).startswith(device)
        torch.testing.assert_close(result, torch.tensor(0.09, device=device))
