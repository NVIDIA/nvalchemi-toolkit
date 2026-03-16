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
"""Tests for :class:`ForceLoss`."""

from __future__ import annotations

from collections import OrderedDict
from unittest.mock import patch

import torch
from torch import Tensor

from nvalchemi.training.losses.forces import ForceLoss
from test.training.conftest import _make_batch


def _cpu_segmented_sum(values: Tensor, idx: Tensor, num_segments: int) -> Tensor:
    out = torch.zeros(num_segments, device=values.device, dtype=values.dtype)
    out.scatter_add_(0, idx.long(), values)
    return out


_PATCH = "nvalchemi.training.losses._base._segmented_sum"


class TestForceLoss:
    """Known-value tests for ForceLoss."""

    def test_defaults(self) -> None:
        """Default attributes are correct."""
        loss = ForceLoss()
        assert loss.level == "node"
        assert loss.pred_key == "forces"
        assert loss.target_key == "forces"
        assert loss.normalize_by_atoms is False

    def test_known_values(self, two_graph_batch) -> None:
        """Hand-computed force loss on the fixture batch."""
        batch = two_graph_batch
        outputs = batch._outputs

        loss = ForceLoss()
        with patch(_PATCH, side_effect=_cpu_segmented_sum):
            result = loss(batch, outputs)

        # Forces diff (pred - target):
        #   node 0: [0.5, 0, 0]   -> sq=[0.25, 0, 0]   sum=0.25
        #   node 1: [0, 0.5, 0]   -> sq=[0, 0.25, 0]   sum=0.25
        #   node 2: [0, 0, 0.5]   -> sq=[0, 0, 0.25]   sum=0.25
        #   node 3: [0, -0.5, 0]  -> sq=[0, 0.25, 0]   sum=0.25
        #   node 4: [0, 0, -0.5]  -> sq=[0, 0, 0.25]   sum=0.25
        # segmented_sum: graph 0 = 0.75, graph 1 = 0.50
        # mean -> 0.625
        assert result is not None
        torch.testing.assert_close(result, torch.tensor(0.625))

    def test_normalize_by_atoms(self, two_graph_batch) -> None:
        """Force loss with normalize_by_atoms=True."""
        batch = two_graph_batch
        outputs = batch._outputs

        loss = ForceLoss(normalize_by_atoms=True)
        with patch(_PATCH, side_effect=_cpu_segmented_sum):
            result = loss(batch, outputs)

        # per_graph [0.75, 0.50], atoms [3, 2]
        # normalized [0.75/3, 0.50/2] = [0.25, 0.25]
        # mean -> 0.25
        assert result is not None
        torch.testing.assert_close(result, torch.tensor(0.25))

    def test_sum_reduction(self, two_graph_batch) -> None:
        """Force loss with sum reduction."""
        batch = two_graph_batch
        outputs = batch._outputs

        loss = ForceLoss(reduction="sum")
        with patch(_PATCH, side_effect=_cpu_segmented_sum):
            result = loss(batch, outputs)

        # per_graph [0.75, 0.50], sum -> 1.25
        assert result is not None
        torch.testing.assert_close(result, torch.tensor(1.25))

    def test_varying_sizes(self, make_batch) -> None:
        """Correct segmented reduction with varying graph sizes."""
        # 2 graphs: 1 atom and 3 atoms
        forces_target = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        forces_pred = torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        data = {"forces": forces_target}
        batch = make_batch([1, 3], data=data)
        outputs = OrderedDict(forces=forces_pred)

        loss = ForceLoss()
        with patch(_PATCH, side_effect=_cpu_segmented_sum):
            result = loss(batch, outputs)

        # node 0: (1-0)^2 = 1 per atom
        # nodes 1-3: all zero diff -> 0
        # per_graph: [1.0, 0.0], mean -> 0.5
        assert result is not None
        torch.testing.assert_close(result, torch.tensor(0.5))

    def test_weight(self, two_graph_batch) -> None:
        """Weight is applied to the result."""
        batch = two_graph_batch
        outputs = batch._outputs

        loss = ForceLoss(weight=5.0)
        with patch(_PATCH, side_effect=_cpu_segmented_sum):
            result = loss(batch, outputs)

        assert result is not None
        torch.testing.assert_close(result, torch.tensor(0.625 * 5.0))

    def test_known_values_on_device(self, device) -> None:
        """ForceLoss produces correct results on the parametrized device."""
        forces_target = torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 1.0],
            ],
            device=device,
        )
        forces_pred = torch.tensor(
            [
                [1.5, 0.0, 0.0],
                [0.0, 1.5, 0.0],
                [0.0, 0.0, 1.5],
                [1.0, 0.5, 0.0],
                [0.0, 1.0, 0.5],
            ],
            device=device,
        )
        batch = _make_batch([3, 2], device=device, data={"forces": forces_target})
        outputs = OrderedDict(forces=forces_pred)

        loss = ForceLoss()
        with patch(_PATCH, side_effect=_cpu_segmented_sum):
            result = loss(batch, outputs)

        assert result is not None
        assert str(result.device).startswith(device)
        torch.testing.assert_close(result, torch.tensor(0.625, device=device))
