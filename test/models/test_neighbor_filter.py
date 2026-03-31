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
from __future__ import annotations

import pytest
import torch

from nvalchemi.data import AtomicData, Batch
from nvalchemi.models import (
    NeighborListBuilder,
    NeighborListBuilderConfig,
    neighbor_result_key,
)
from nvalchemi.models.results import CalculatorResults


def _make_batch() -> Batch:
    """Return a small non-periodic batch."""

    data = [
        AtomicData(
            positions=torch.tensor(
                [[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]], dtype=torch.float32
            ),
            atomic_numbers=torch.tensor([8, 1], dtype=torch.long),
        )
    ]
    return Batch.from_data_list(data)


def _matrix_neighbor_results(
    batch: Batch, *, name: str = "default"
) -> CalculatorResults:
    """Return a minimal matrix-neighbor result set."""

    num_nodes = batch.num_nodes
    neighbor_matrix = torch.full((num_nodes, 2), -1, dtype=torch.int32)
    if num_nodes >= 2:
        neighbor_matrix[0, 0] = 1
        neighbor_matrix[1, 0] = 0
    num_neighbors = torch.tensor([1] * num_nodes, dtype=torch.int32)
    neighbor_shifts = torch.zeros((num_nodes, 2, 3), dtype=torch.int32)
    return CalculatorResults(
        {
            neighbor_result_key(name, "neighbor_matrix"): neighbor_matrix,
            neighbor_result_key(name, "num_neighbors"): num_neighbors,
            neighbor_result_key(name, "neighbor_shifts"): neighbor_shifts,
        }
    )


def test_neighbor_builder_reuses_existing_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Neighbor builders should reuse matching outputs when configured."""

    batch = _make_batch()
    builder = NeighborListBuilder(
        NeighborListBuilderConfig(
            cutoff=4.0,
            format="matrix",
            reuse_if_available=True,
        )
    )
    existing = _matrix_neighbor_results(batch)

    def _should_not_build(**kwargs):  # type: ignore[no-untyped-def]
        raise AssertionError("neighbor_list backend should not run when reuse succeeds")

    monkeypatch.setattr("nvalchemi.models.neighbors.neighbor_list", _should_not_build)
    outputs = builder(batch, results=existing, outputs=builder.profile.result_keys)

    assert torch.equal(
        outputs[neighbor_result_key("default", "neighbor_matrix")],
        existing[neighbor_result_key("default", "neighbor_matrix")],
    )
