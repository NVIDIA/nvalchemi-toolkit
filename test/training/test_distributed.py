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
"""Tests for distributed training wrappers."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import torch
from torch import nn

from nvalchemi.training._distributed import reduce_loss_across_ranks, wrap_model


class _SimpleModel(nn.Module):
    """Minimal model for wrapping tests."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def _make_dist_manager(
    *,
    world_size: int = 2,
    local_rank: int = 0,
    rank: int = 0,
    distributed: bool = True,
    broadcast_buffers: bool = True,
    find_unused_parameters: bool = False,
) -> MagicMock:
    """Build a mock DistributedManager."""
    mgr = MagicMock()
    mgr.world_size = world_size
    mgr.local_rank = local_rank
    mgr.rank = rank
    mgr.distributed = distributed
    mgr.broadcast_buffers = broadcast_buffers
    mgr.find_unused_parameters = find_unused_parameters
    return mgr


class TestWrapModel:
    """Tests for :func:`wrap_model`."""

    def test_no_manager_returns_original(self) -> None:
        model = _SimpleModel()
        wrapped = wrap_model(model, strategy="ddp", dist_manager=None)
        assert wrapped is model

    def test_world_size_one_returns_original(self) -> None:
        model = _SimpleModel()
        mgr = _make_dist_manager(world_size=1)
        wrapped = wrap_model(model, strategy="ddp", dist_manager=mgr)
        assert wrapped is model

    @patch("nvalchemi.training._distributed.DistributedDataParallel")
    def test_ddp_wrapping(self, mock_ddp: MagicMock) -> None:
        model = _SimpleModel()
        mgr = _make_dist_manager(world_size=4, local_rank=2)
        mock_ddp.return_value = MagicMock(spec=nn.Module)

        result = wrap_model(model, strategy="ddp", dist_manager=mgr)

        mock_ddp.assert_called_once_with(
            model,
            device_ids=[2],
            broadcast_buffers=True,
            find_unused_parameters=False,
        )
        assert result is mock_ddp.return_value

    @patch("nvalchemi.training._distributed.DistributedDataParallel")
    def test_ddp_forwards_dist_manager_attrs(self, mock_ddp: MagicMock) -> None:
        model = _SimpleModel()
        mgr = _make_dist_manager(
            world_size=4,
            local_rank=1,
            broadcast_buffers=False,
            find_unused_parameters=True,
        )
        mock_ddp.return_value = MagicMock(spec=nn.Module)

        wrap_model(model, strategy="ddp", dist_manager=mgr)

        mock_ddp.assert_called_once_with(
            model,
            device_ids=[1],
            broadcast_buffers=False,
            find_unused_parameters=True,
        )

    @patch("torch.distributed.fsdp.FullyShardedDataParallel")
    def test_fsdp_wrapping(self, mock_fsdp: MagicMock) -> None:
        model = _SimpleModel()
        mgr = _make_dist_manager(world_size=4, local_rank=0)
        mock_fsdp.return_value = MagicMock(spec=nn.Module)

        result = wrap_model(model, strategy="fsdp", dist_manager=mgr)

        mock_fsdp.assert_called_once_with(model)
        assert result is mock_fsdp.return_value

    def test_invalid_strategy_raises(self) -> None:
        model = _SimpleModel()
        mgr = _make_dist_manager(world_size=2)
        with pytest.raises(ValueError, match="Unknown distributed strategy"):
            wrap_model(model, strategy="bad", dist_manager=mgr)  # type: ignore[arg-type]


class TestReduceLossAcrossRanks:
    """Tests for :func:`reduce_loss_across_ranks`."""

    def test_no_manager_identity(self) -> None:
        loss = torch.tensor(3.14)
        result = reduce_loss_across_ranks(loss, dist_manager=None)
        assert result is loss

    def test_not_distributed_identity(self) -> None:
        loss = torch.tensor(2.71)
        mgr = _make_dist_manager(distributed=False)
        result = reduce_loss_across_ranks(loss, dist_manager=mgr)
        assert result is loss

    @patch("nvalchemi.training._distributed._reduce_loss", return_value=5.0)
    def test_distributed_reduces(self, mock_reduce: MagicMock) -> None:
        loss = torch.tensor(4.0)
        mgr = _make_dist_manager(distributed=True)

        result = reduce_loss_across_ranks(loss, dist_manager=mgr)

        mock_reduce.assert_called_once_with(4.0, dst_rank=0, mean=True)
        assert result.item() == pytest.approx(5.0)

    @patch("nvalchemi.training._distributed._reduce_loss", return_value=None)
    def test_distributed_non_dst_rank_returns_original(
        self, mock_reduce: MagicMock
    ) -> None:
        loss = torch.tensor(4.0)
        mgr = _make_dist_manager(distributed=True)

        result = reduce_loss_across_ranks(loss, dist_manager=mgr)

        assert result is loss
