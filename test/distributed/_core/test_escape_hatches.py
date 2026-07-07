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

"""wrap_custom_op tests.

Synthetic Python "custom op" that swallows a tensor and returns a
modified version — proving the wrapper installs correctly and passes
through when no halo context is active.
"""

from __future__ import annotations

import pytest
import torch

from nvalchemi.distributed._core.escape_hatches import wrap_custom_op
from nvalchemi.distributed._core.shard_tensor import (
    ShardTensor,
    clear_handlers,
    list_handlers,
)

# ShardTensor.wrap requires a DeviceMesh; the session fixture provides a
# 1-rank gloo mesh.
pytestmark = pytest.mark.usefixtures("_session_gloo_pg")


# ======================================================================
# wrap_custom_op — no halo context = pass-through
# ======================================================================


def test_wrap_custom_op_passthrough_without_halo_meta() -> None:
    """Without halo metadata on the wrapped tensor, the wrapper should
    transparently pass through to the underlying op."""

    call_count = [0]

    def my_op(x: torch.Tensor, y: float) -> torch.Tensor:
        call_count[0] += 1
        return x * y

    wrap_custom_op(my_op)
    try:
        t = ShardTensor.wrap(torch.ones(5, dtype=torch.float64))
        result = my_op(t, 3.0)
        assert call_count[0] == 1
        assert isinstance(result, ShardTensor)
        torch.testing.assert_close(
            result.unwrap(), torch.full((5,), 3.0, dtype=torch.float64)
        )
    finally:
        clear_handlers(my_op)


def test_wrap_custom_op_with_plain_tensor_input_still_passes_through() -> None:
    """If the op is called on a plain Tensor (no ShardTensor), our
    registered handler shouldn't even fire — ``__torch_function__`` only
    dispatches on tensor subclasses."""
    call_count = [0]

    def my_op(x: torch.Tensor) -> torch.Tensor:
        call_count[0] += 1
        return x + 1

    wrap_custom_op(my_op)
    try:
        result = my_op(torch.ones(3, dtype=torch.float64))
        assert call_count[0] == 1
        torch.testing.assert_close(result, torch.full((3,), 2.0, dtype=torch.float64))
    finally:
        clear_handlers(my_op)


# ======================================================================
# wrap_custom_op — with halo context (forward halo flow tested; comm
# primitives mocked so we can test the wrapping logic in-process)
# ======================================================================


def test_wrap_custom_op_registers_handler_in_registry() -> None:
    def my_op(x: torch.Tensor) -> torch.Tensor:
        return x

    wrap_custom_op(my_op, scatter_outputs=[0])
    try:
        names = [n for _, n in list_handlers()]
        assert any("wrap_custom_op" in n for n in names)
    finally:
        clear_handlers(my_op)


def test_wrap_custom_op_deep_unwraps_list_tensor_arg() -> None:
    """Wrapped ops that take a ``List[Tensor]`` must see plain tensors
    inside the list, not ShardTensor subclasses.

    Without deep-unwrap, nested subclasses leak through → the op's
    internal dispatch fires ``__torch_function__`` → the handler
    re-enters itself → ``RecursionError``. This bites cueq's
    ``torch.ops.cuequivariance.uniform_1d(name, ..., tensors)`` where
    ``tensors`` is a list of per-operand tensors.

    Uses ``torch.library.custom_op`` so the call actually routes through
    ``__torch_function__`` — plain Python functions wouldn't.
    """

    seen_types: list[type] = []

    @torch.library.custom_op("nvalchemi_test::list_tensor_op", mutates_args=())
    def _list_op(tensors: list[torch.Tensor]) -> torch.Tensor:
        seen_types.extend(type(t) for t in tensors)
        return tensors[0].clone()

    op = torch.ops.nvalchemi_test.list_tensor_op.default
    wrap_custom_op(op)
    try:
        a = ShardTensor.wrap(torch.ones(3, dtype=torch.float64))
        b = ShardTensor.wrap(torch.full((3,), 2.0, dtype=torch.float64))
        # Completes in bounded recursion depth AND the kernel saw plain
        # tensors.
        torch.ops.nvalchemi_test.list_tensor_op([a, b])
        assert seen_types == [torch.Tensor, torch.Tensor], (
            f"leaked subclasses into list arg: {seen_types}"
        )
    finally:
        clear_handlers(op)
        clear_handlers(torch.ops.nvalchemi_test.list_tensor_op)


def test_wrap_custom_op_recursion_guard() -> None:
    """The deep-unwrap contract prevents unbounded re-entry of the
    dispatcher when the op takes a ``List[Tensor]`` (the cueq
    ``RecursionError`` signature).
    """

    @torch.library.custom_op("nvalchemi_test::recurse_guard_op", mutates_args=())
    def _op(tensors: list[torch.Tensor]) -> torch.Tensor:
        return tensors[0].sum().view(1)

    op = torch.ops.nvalchemi_test.recurse_guard_op.default
    wrap_custom_op(op)
    try:
        a = ShardTensor.wrap(torch.ones(4, dtype=torch.float64))
        b = ShardTensor.wrap(torch.ones(4, dtype=torch.float64))
        # Should complete without hitting Python's recursion limit.
        out = torch.ops.nvalchemi_test.recurse_guard_op([a, b])
        # Value is correct — sum of 4 ones.
        val = out.unwrap() if isinstance(out, ShardTensor) else out
        torch.testing.assert_close(val, torch.tensor([4.0], dtype=torch.float64))
    finally:
        clear_handlers(op)
        clear_handlers(torch.ops.nvalchemi_test.recurse_guard_op)
