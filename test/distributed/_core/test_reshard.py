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

"""Tests for reshard_by_destination and migrate_sharded_batch.

Single-GPU tests validate the no-op path (without torch.distributed).
Multi-GPU tests are in test_multigpu.py.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import torch

from nvalchemi.distributed._core.reshard import reshard_by_destination


class TestReshardSingleProcess:
    """Without torch.distributed initialized, reshard returns tensor unchanged."""

    def test_returns_tensor_unchanged(self):
        tensor = torch.randn(10, 3)
        destinations = torch.zeros(10, dtype=torch.long)
        mesh = MagicMock()
        result = reshard_by_destination(tensor, destinations, mesh)
        assert torch.equal(result, tensor)

    def test_1d_tensor(self):
        tensor = torch.arange(5, dtype=torch.float)
        destinations = torch.zeros(5, dtype=torch.long)
        mesh = MagicMock()
        result = reshard_by_destination(tensor, destinations, mesh)
        assert torch.equal(result, tensor)

    def test_preserves_dtype_int64(self):
        tensor = torch.arange(5, dtype=torch.int64)
        destinations = torch.zeros(5, dtype=torch.long)
        result = reshard_by_destination(tensor, destinations, MagicMock())
        assert result.dtype == torch.int64

    def test_preserves_dtype_float32(self):
        tensor = torch.randn(5, 3, dtype=torch.float32)
        destinations = torch.zeros(5, dtype=torch.long)
        result = reshard_by_destination(tensor, destinations, MagicMock())
        assert result.dtype == torch.float32

    def test_empty_tensor(self):
        tensor = torch.zeros(0, 3)
        destinations = torch.zeros(0, dtype=torch.long)
        result = reshard_by_destination(tensor, destinations, MagicMock())
        assert result.shape == (0, 3)

    def test_single_element(self):
        tensor = torch.tensor([[1.0, 2.0, 3.0]])
        destinations = torch.tensor([0])
        result = reshard_by_destination(tensor, destinations, MagicMock())
        assert torch.equal(result, tensor)


# Multi-GPU reshard tests are in test_multigpu.py
