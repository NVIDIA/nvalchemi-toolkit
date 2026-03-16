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
"""Tests for :class:`EnergyLoss`."""

from __future__ import annotations

from collections import OrderedDict
from unittest.mock import patch

import torch
from torch import Tensor

from nvalchemi.training.losses.energy import EnergyLoss
from test.training.conftest import _make_batch


# Pure-torch fallback for CPU testing
def _cpu_segmented_sum(values: Tensor, idx: Tensor, num_segments: int) -> Tensor:
    out = torch.zeros(num_segments, device=values.device, dtype=values.dtype)
    out.scatter_add_(0, idx.long(), values)
    return out


_PATCH = "nvalchemi.training.losses._base._segmented_sum"


class TestEnergyLoss:
    """Known-value tests for EnergyLoss."""

    def test_defaults(self) -> None:
        """Default attributes are correct."""
        loss = EnergyLoss()
        assert loss.level == "system"
        assert loss.pred_key == "energies"
        assert loss.target_key == "energies"
        assert loss.normalize_by_atoms is True
        assert loss.error_fn == "mse"

    def test_mse_known_values(self, two_graph_batch) -> None:
        """Hand-computed MSE energy loss with normalize_by_atoms=True."""
        batch = two_graph_batch
        outputs = batch._outputs

        loss = EnergyLoss(normalize_by_atoms=True)
        with patch(_PATCH, side_effect=_cpu_segmented_sum):
            result = loss(batch, outputs)

        # pred=[11,19], target=[10,20], diff^2 = [1, 1]  (B,1)
        # reshape(B,-1).sum(-1) -> [1, 1]
        # normalize: [1/3, 1/2] (atoms_per_graph = [3, 2])
        # mean -> (1/3 + 1/2) / 2 = 5/12
        assert result is not None
        torch.testing.assert_close(result, torch.tensor(5.0 / 12.0))

    def test_mse_no_normalize(self, two_graph_batch) -> None:
        """MSE without atom-count normalization."""
        batch = two_graph_batch
        outputs = batch._outputs

        loss = EnergyLoss(normalize_by_atoms=False)
        with patch(_PATCH, side_effect=_cpu_segmented_sum):
            result = loss(batch, outputs)

        # per_graph = [1, 1], mean -> 1.0
        assert result is not None
        torch.testing.assert_close(result, torch.tensor(1.0))

    def test_mae(self, two_graph_batch) -> None:
        """MAE energy loss."""
        batch = two_graph_batch
        outputs = batch._outputs

        loss = EnergyLoss(error_fn="mae", normalize_by_atoms=False)
        with patch(_PATCH, side_effect=_cpu_segmented_sum):
            result = loss(batch, outputs)

        # |pred - target| = [1, 1], mean -> 1.0
        assert result is not None
        torch.testing.assert_close(result, torch.tensor(1.0))

    def test_huber_below_delta(self) -> None:
        """Huber behaves like MSE/2 when |error| < delta."""
        pred = torch.tensor([[0.3], [0.2]])
        target = torch.tensor([[0.0], [0.0]])
        data = {"energies": target}

        from test.training.conftest import _make_batch

        batch = _make_batch([1, 1], data=data)
        outputs = OrderedDict(energies=pred)

        loss = EnergyLoss(error_fn="huber", huber_delta=1.0, normalize_by_atoms=False)
        with patch(_PATCH, side_effect=_cpu_segmented_sum):
            result = loss(batch, outputs)

        # below delta: 0.5 * diff^2 = [0.045, 0.02]
        # mean -> (0.045 + 0.02) / 2 = 0.0325
        assert result is not None
        expected = 0.5 * (0.3**2 + 0.2**2) / 2
        torch.testing.assert_close(result, torch.tensor(expected))

    def test_huber_above_delta(self) -> None:
        """Huber linear regime when |error| > delta."""
        delta = 0.5
        pred = torch.tensor([[2.0]])
        target = torch.tensor([[0.0]])
        data = {"energies": target}

        from test.training.conftest import _make_batch

        batch = _make_batch([1], data=data)
        outputs = OrderedDict(energies=pred)

        loss = EnergyLoss(error_fn="huber", huber_delta=delta, normalize_by_atoms=False)
        with patch(_PATCH, side_effect=_cpu_segmented_sum):
            result = loss(batch, outputs)

        # |diff| = 2 > delta=0.5: delta * (|diff| - 0.5*delta) = 0.5*(2-0.25) = 0.875
        assert result is not None
        expected = delta * (2.0 - 0.5 * delta)
        torch.testing.assert_close(result, torch.tensor(expected))

    def test_varying_batch_sizes(self, make_batch) -> None:
        """Correct results across batches with different graph sizes."""
        energy_target = torch.tensor([[1.0], [2.0], [3.0]])
        energy_pred = torch.tensor([[1.5], [2.5], [3.5]])
        data = {"energies": energy_target}
        batch = make_batch([4, 1, 7], data=data)
        outputs = OrderedDict(energies=energy_pred)

        loss = EnergyLoss(normalize_by_atoms=True)
        with patch(_PATCH, side_effect=_cpu_segmented_sum):
            result = loss(batch, outputs)

        # diff^2 = [0.25, 0.25, 0.25], atoms = [4, 1, 7]
        # normalized = [0.25/4, 0.25/1, 0.25/7]
        # mean = (0.0625 + 0.25 + 0.25/7) / 3
        assert result is not None
        per_graph = torch.tensor([0.25 / 4, 0.25 / 1, 0.25 / 7])
        torch.testing.assert_close(result, per_graph.mean())

    def test_mse_on_device(self, device) -> None:
        """EnergyLoss produces correct results on the parametrized device."""
        energy_target = torch.tensor([[10.0], [20.0]], device=device)
        energy_pred = torch.tensor([[11.0], [19.0]], device=device)
        batch = _make_batch([3, 2], device=device, data={"energies": energy_target})
        outputs = OrderedDict(energies=energy_pred)

        loss = EnergyLoss(normalize_by_atoms=True)
        with patch(_PATCH, side_effect=_cpu_segmented_sum):
            result = loss(batch, outputs)

        assert result is not None
        assert str(result.device).startswith(device)
        torch.testing.assert_close(result, torch.tensor(5.0 / 12.0, device=device))

    def test_mae_on_device(self, device) -> None:
        """MAE EnergyLoss produces correct results on the parametrized device."""
        energy_target = torch.tensor([[10.0], [20.0]], device=device)
        energy_pred = torch.tensor([[11.0], [19.0]], device=device)
        batch = _make_batch([3, 2], device=device, data={"energies": energy_target})
        outputs = OrderedDict(energies=energy_pred)

        loss = EnergyLoss(error_fn="mae", normalize_by_atoms=False)
        with patch(_PATCH, side_effect=_cpu_segmented_sum):
            result = loss(batch, outputs)

        assert result is not None
        assert str(result.device).startswith(device)
        torch.testing.assert_close(result, torch.tensor(1.0, device=device))
