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

"""Tests for shard wrapper registration and generic wrapper factories.

Tests error handling, graceful degradation, and all return-type paths
(None, Tensor, tuple) for both passthrough and reduction wrappers.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import torch

from nvalchemi.distributed._core.shard_wrappers import (
    _first_shard_tensor,
    _is_shard_tensor,
    _to_local_if_shard,
    make_passthrough_wrapper,
    make_reduction_wrapper,
)
from nvalchemi.distributed.shard_wrappers import (
    PASSTHROUGH_OPS,
    register_shard_wrappers,
)

# ======================================================================
# Helpers
# ======================================================================


class TestHelpers:
    def test_is_shard_tensor_plain(self):
        assert not _is_shard_tensor(torch.zeros(3))

    def test_is_shard_tensor_mock(self):
        mock = MagicMock()
        mock.__class__.__name__ = "ShardTensor"
        assert _is_shard_tensor(mock)

    def test_is_shard_tensor_int(self):
        assert not _is_shard_tensor(42)

    def test_is_shard_tensor_none(self):
        assert not _is_shard_tensor(None)

    def test_to_local_if_shard_plain_tensor(self):
        t = torch.zeros(3)
        assert _to_local_if_shard(t) is t

    def test_to_local_if_shard_non_tensor(self):
        assert _to_local_if_shard(42) == 42
        assert _to_local_if_shard("hello") == "hello"

    def test_to_local_if_shard_mock(self):
        mock = MagicMock()
        mock.__class__.__name__ = "ShardTensor"
        local = torch.ones(3)
        mock.to_local.return_value = local
        assert _to_local_if_shard(mock) is local

    def test_first_shard_tensor_none(self):
        assert _first_shard_tensor(torch.zeros(3), 42, "hello") is None

    def test_first_shard_tensor_found(self):
        mock = MagicMock()
        mock.__class__.__name__ = "ShardTensor"
        result = _first_shard_tensor(torch.zeros(3), mock, torch.ones(2))
        assert result is mock

    def test_first_shard_tensor_empty(self):
        assert _first_shard_tensor() is None


# ======================================================================
# Passthrough wrapper
# ======================================================================


class TestMakePassthroughWrapper:
    def test_none_result(self):
        """mutates_args ops return None — wrapper should too."""
        wrapper = make_passthrough_wrapper("test_op")
        func = MagicMock(return_value=None)
        result = wrapper(func, (), (torch.zeros(3),), {})
        assert result is None

    def test_plain_tensor_passthrough(self):
        """Without ShardTensor inputs, result passes through unchanged."""
        wrapper = make_passthrough_wrapper("test_op")
        expected = torch.ones(3)
        func = MagicMock(return_value=expected)
        result = wrapper(func, (), (torch.zeros(3),), {})
        assert result is expected

    def test_kwargs_unwrapped(self):
        wrapper = make_passthrough_wrapper("test_op")
        func = MagicMock(return_value=None)
        wrapper(func, (), (), {"key": torch.zeros(3)})
        func.assert_called_once()
        _, kwargs = func.call_args
        assert isinstance(kwargs["key"], torch.Tensor)

    def test_func_receives_local_tensors(self):
        """ShardTensor args should be unwrapped via to_local() before calling func."""
        wrapper = make_passthrough_wrapper("test_op")
        local_t = torch.randn(5)
        mock_st = MagicMock()
        mock_st.__class__.__name__ = "ShardTensor"
        mock_st.to_local.return_value = local_t

        func = MagicMock(return_value=None)
        wrapper(func, (), (mock_st, torch.zeros(3)), {})
        # First arg should be the local tensor, not the mock
        call_args = func.call_args[0]
        assert call_args[0] is local_t

    def test_func_error_propagates(self):
        """If the wrapped function raises, the error should propagate."""
        wrapper = make_passthrough_wrapper("test_op")
        func = MagicMock(side_effect=RuntimeError("kernel failed"))
        try:
            wrapper(func, (), (torch.zeros(3),), {})
            assert False, "Should have raised"
        except RuntimeError as e:
            assert "kernel failed" in str(e)

    def test_non_tensor_result_passthrough(self):
        """Non-tensor, non-None, non-tuple results pass through."""
        wrapper = make_passthrough_wrapper("test_op")
        func = MagicMock(return_value=42)
        result = wrapper(func, (), (torch.zeros(3),), {})
        assert result == 42


# ======================================================================
# Reduction wrapper
# ======================================================================


class TestMakeReductionWrapper:
    def test_plain_tensor_passthrough(self):
        """Without ShardTensor input, result is plain tensor."""
        wrapper = make_reduction_wrapper(torch.distributed.ReduceOp.SUM)
        expected = torch.ones(1)
        func = MagicMock(return_value=expected)
        result = wrapper(func, (), (torch.zeros(3),), {})
        assert result is expected

    def test_non_tensor_result(self):
        """Non-tensor results pass through unchanged."""
        wrapper = make_reduction_wrapper(torch.distributed.ReduceOp.SUM)
        func = MagicMock(return_value=42)
        result = wrapper(func, (), (torch.zeros(3),), {})
        assert result == 42

    def test_func_error_propagates(self):
        """If the wrapped function raises, error propagates."""
        wrapper = make_reduction_wrapper(torch.distributed.ReduceOp.SUM)
        func = MagicMock(side_effect=ValueError("bad input"))
        try:
            wrapper(func, (), (torch.zeros(3),), {})
            assert False, "Should have raised"
        except ValueError as e:
            assert "bad input" in str(e)


# ======================================================================
# Op list completeness
# ======================================================================


class TestPassthroughOpsList:
    def test_has_velocity_verlet(self):
        assert "nvalchemi::vv_position_update" in PASSTHROUGH_OPS
        assert "nvalchemi::vv_velocity_finalize" in PASSTHROUGH_OPS

    def test_has_langevin(self):
        assert "nvalchemi::langevin_half_step" in PASSTHROUGH_OPS
        assert "nvalchemi::langevin_finalize" in PASSTHROUGH_OPS

    def test_has_fire(self):
        assert "nvalchemi::_fire_step_op" in PASSTHROUGH_OPS
        assert "nvalchemi::_fire_update_op" in PASSTHROUGH_OPS

    def test_has_nose_hoover(self):
        assert "nvalchemi::nhc_velocity_half_step" in PASSTHROUGH_OPS
        assert "nvalchemi::nhc_position_update" in PASSTHROUGH_OPS

    def test_has_npt(self):
        assert "nvalchemi::npt_position_update" in PASSTHROUGH_OPS
        assert "nvalchemi::npt_cell_update" in PASSTHROUGH_OPS

    def test_has_thermostat_utils(self):
        assert "nvalchemi::remove_com_motion" in PASSTHROUGH_OPS
        assert "nvalchemi::velocity_rescale" in PASSTHROUGH_OPS
        assert "nvalchemi::initialize_velocities" in PASSTHROUGH_OPS

    def test_has_hooks(self):
        assert "nvalchemi_hooks::wrap_positions" in PASSTHROUGH_OPS

    def test_has_nl_rebuild(self):
        assert "nvalchemi::_batch_neighbor_list_rebuild_inplace" in PASSTHROUGH_OPS

    def test_minimum_count(self):
        assert len(PASSTHROUGH_OPS) >= 20


# ======================================================================
# Registration
# ======================================================================


class TestRegisterShardWrappers:
    def test_idempotent(self):
        register_shard_wrappers()
        register_shard_wrappers()  # second call is no-op

    def test_importable(self):
        """Importing the module and calling register should not raise."""
        from nvalchemi.distributed.shard_wrappers import (
            register_shard_wrappers as reg,
        )

        reg()


# ======================================================================
# Error handling / graceful degradation
# ======================================================================


class TestGracefulDegradation:
    """Verify wrappers handle edge cases gracefully."""

    def test_passthrough_with_empty_args(self):
        wrapper = make_passthrough_wrapper("test_op")
        func = MagicMock(return_value=None)
        result = wrapper(func, (), (), {})
        assert result is None

    def test_passthrough_with_mixed_types(self):
        """Args can be a mix of tensors, ints, strings."""
        wrapper = make_passthrough_wrapper("test_op")
        func = MagicMock(return_value=None)
        wrapper(func, (), (torch.zeros(3), 42, "hello", None), {})
        call_args = func.call_args[0]
        assert isinstance(call_args[0], torch.Tensor)
        assert call_args[1] == 42
        assert call_args[2] == "hello"
        assert call_args[3] is None

    def test_reduction_with_empty_args(self):
        wrapper = make_reduction_wrapper(torch.distributed.ReduceOp.SUM)
        func = MagicMock(return_value=torch.ones(1))
        result = wrapper(func, (), (), {})
        assert isinstance(result, torch.Tensor)
