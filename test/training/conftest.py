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
"""Shared fixtures for training tests."""

from __future__ import annotations

from collections import OrderedDict
from typing import TYPE_CHECKING

import pytest
import torch

if TYPE_CHECKING:
    from nvalchemi._typing import ModelOutputs


class MockBatch:
    """Lightweight stand-in for :class:`nvalchemi.data.Batch`.

    Provides the interface consumed by the loss system without requiring
    the full ``MultiLevelStorage`` machinery.

    Parameters
    ----------
    data : dict[str, torch.Tensor]
        Attribute tensors keyed by name.
    batch_idx : torch.Tensor
        Node-to-graph assignment, shape ``(V,)``.
    num_nodes_per_graph : torch.Tensor
        Per-graph node counts, shape ``(B,)``.
    device : torch.device
        Device to place tensors on.
    """

    def __init__(
        self,
        data: dict[str, torch.Tensor],
        batch_idx: torch.Tensor,
        num_nodes_per_graph: torch.Tensor,
        device: torch.device | None = None,
    ) -> None:
        self._data = data
        self._batch_idx = batch_idx
        self._num_nodes_per_graph = num_nodes_per_graph
        self.device = device or batch_idx.device

    @property
    def batch(self) -> torch.Tensor:
        return self._batch_idx

    @property
    def num_graphs(self) -> int:
        return int(self._num_nodes_per_graph.shape[0])

    @property
    def num_nodes_per_graph(self) -> torch.Tensor:
        return self._num_nodes_per_graph

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def __getitem__(self, key: str) -> torch.Tensor:
        return self._data[key]


def _make_batch(
    num_nodes_per_graph: list[int],
    *,
    device: torch.device | str = "cpu",
    data: dict[str, torch.Tensor] | None = None,
) -> MockBatch:
    """Build a :class:`MockBatch` from per-graph node counts.

    Parameters
    ----------
    num_nodes_per_graph : list[int]
        Node counts per graph.
    device : torch.device or str
        Target device.
    data : dict[str, torch.Tensor] or None
        Extra attribute tensors.

    Returns
    -------
    MockBatch
    """
    dev = torch.device(device)
    counts = torch.tensor(num_nodes_per_graph, dtype=torch.long, device=dev)
    batch_idx = torch.repeat_interleave(
        torch.arange(len(num_nodes_per_graph), device=dev), counts
    )
    return MockBatch(
        data=data or {},
        batch_idx=batch_idx,
        num_nodes_per_graph=counts,
        device=dev,
    )


@pytest.fixture()
def two_graph_batch() -> MockBatch:
    """Batch with 2 graphs: 3 and 2 nodes (5 total).

    Includes energy, forces, and stress targets/predictions for use
    across multiple loss test modules.
    """
    nodes = [3, 2]
    B = len(nodes)

    # Forces: V x 3
    forces_target = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 1.0],
        ],
    )
    forces_pred = torch.tensor(
        [
            [1.5, 0.0, 0.0],
            [0.0, 1.5, 0.0],
            [0.0, 0.0, 1.5],
            [1.0, 0.5, 0.0],
            [0.0, 1.0, 0.5],
        ],
    )

    # Energies: B x 1
    energy_target = torch.tensor([[10.0], [20.0]])
    energy_pred = torch.tensor([[11.0], [19.0]])

    # Stresses: B x 3 x 3
    stress_target = torch.zeros(B, 3, 3)
    stress_pred = torch.ones(B, 3, 3) * 0.1

    data = {
        "forces": forces_target,
        "energies": energy_target,
        "stresses": stress_target,
    }
    batch = _make_batch(nodes, data=data)

    outputs: ModelOutputs = OrderedDict(
        energies=energy_pred,
        forces=forces_pred,
        stresses=stress_pred,
    )
    # Stash predictions on the fixture so tests can retrieve them
    batch._outputs = outputs  # type: ignore[attr-defined]
    batch._forces_pred = forces_pred  # type: ignore[attr-defined]
    batch._energy_pred = energy_pred  # type: ignore[attr-defined]
    batch._stress_pred = stress_pred  # type: ignore[attr-defined]
    return batch


@pytest.fixture()
def make_batch():
    """Factory fixture returning :func:`_make_batch`."""
    return _make_batch
