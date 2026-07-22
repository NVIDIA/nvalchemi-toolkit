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

"""Compile smoke — does ``torch.compile`` trace through our
:class:`ShardTensor` without graph breaks?

The smoke is single-rank, in-process, no distribution. It deliberately
does *not* exercise halo-correction or per-system-reduce dispatch — see
``test_compile_smoke_distributed`` for that. The only assertion here is
"Dynamo can trace a model that receives a ShardTensor at function entry
and produces a result, end-to-end, with no graph breaks."

A small ``nn.Module`` (not a lambda) is the right unit: Dynamo routes
through :meth:`nn.Module._call_impl` with its own tracing quirks
(parameter handling, hook traversal, module-state guards) that a lambda
wouldn't exercise.
"""

from __future__ import annotations

import pytest
import torch
import torch._dynamo
import torch._dynamo.utils
from torch import nn

import nvalchemi.distributed  # noqa: F401 — registers ShardTensor with Dynamo
from nvalchemi.distributed._core.shard_tensor import ShardTensor

# ShardTensor.wrap requires a DeviceMesh; the session fixture provides a
# 1-rank gloo mesh.
pytestmark = pytest.mark.usefixtures("_session_gloo_pg")


# ----------------------------------------------------------------------
# Tiny MPNN-flavoured nn.Module: Linear → ReLU → Linear → per-graph scatter
# ----------------------------------------------------------------------


class _TinyMPNN(nn.Module):
    """Linear projection over per-atom positions, scatter to per-graph energy.

    Op surface chosen to exercise the dispatch sites real wrappers hit:
    ``aten::linear`` (parameters flowing through ShardTensor),
    ``aten::scatter_add`` (per-graph reduction), an autograd backward
    through the whole chain.
    """

    def __init__(self, hidden: int = 8) -> None:
        super().__init__()
        self.lin = nn.Linear(3, hidden)
        self.head = nn.Linear(hidden, 1)

    def forward(
        self, positions: torch.Tensor, batch_idx: torch.Tensor, num_graphs: int
    ) -> torch.Tensor:
        x = self.lin(positions).relu()
        per_atom_e = self.head(x).squeeze(-1)
        e_total = torch.zeros(num_graphs, device=x.device, dtype=x.dtype)
        return e_total.scatter_add(0, batch_idx, per_atom_e)


def _build_inputs(
    n_atoms: int = 8,
    num_graphs: int = 2,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    positions = torch.randn(n_atoms, 3, device=device, dtype=dtype, requires_grad=True)
    half = n_atoms // 2
    batch_idx = torch.cat(
        [
            torch.zeros(half, device=device, dtype=torch.long),
            torch.ones(n_atoms - half, device=device, dtype=torch.long),
        ]
    )
    return positions, batch_idx


# ----------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "device",
    [
        torch.device("cpu"),
        pytest.param(
            torch.device("cuda:0"),
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="requires CUDA"
            ),
        ),
    ],
    ids=["cpu", "cuda"],
)
def test_compile_smoke_world1_forward(device: torch.device) -> None:
    """Compile a tiny nn.Module that receives a ShardTensor input.

    fullgraph=True is the implicit graph-break gate — any graph break
    raises rather than silently degrading.
    """
    torch.manual_seed(0)
    model = _TinyMPNN().to(device)
    positions, batch_idx = _build_inputs(device=device)
    num_graphs = 2

    # Eager reference
    eager_out = model(ShardTensor.wrap(positions), batch_idx, num_graphs).clone()

    # Compile and run. ``backend="eager"`` skips Inductor codegen — the
    # gate here is "Dynamo can trace through our subclass cleanly," not
    # "Inductor can produce optimized kernels." Inductor's codegen path is
    # exercised by the real-wrapper DD tests.
    torch._dynamo.reset()
    torch._dynamo.utils.counters.clear()
    compiled = torch.compile(model, fullgraph=True, dynamic=False, backend="eager")
    compiled_out = compiled(ShardTensor.wrap(positions), batch_idx, num_graphs)

    # Numerical match
    torch.testing.assert_close(eager_out, compiled_out, atol=1e-5, rtol=1e-5)

    # Belt-and-suspenders: no graph breaks (fullgraph=True should already error
    # on any break, but explicit check surfaces edge cases where Dynamo logged
    # a break but didn't raise).
    graph_breaks = dict(torch._dynamo.utils.counters.get("graph_break", {}))
    assert not graph_breaks, f"unexpected graph breaks: {graph_breaks}"


@pytest.mark.parametrize(
    "device",
    [
        torch.device("cpu"),
        pytest.param(
            torch.device("cuda:0"),
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="requires CUDA"
            ),
        ),
    ],
    ids=["cpu", "cuda"],
)
def test_compile_smoke_world1_backward(device: torch.device) -> None:
    torch.manual_seed(0)
    model = _TinyMPNN().to(device)
    positions_eager, batch_idx = _build_inputs(device=device)
    num_graphs = 2

    eager_out = model(ShardTensor.wrap(positions_eager), batch_idx, num_graphs)
    eager_grad = torch.autograd.grad(eager_out.sum(), positions_eager)[0]

    positions_compiled, _ = _build_inputs(device=device)
    torch._dynamo.reset()
    torch._dynamo.utils.counters.clear()
    compiled = torch.compile(model, fullgraph=True, dynamic=False, backend="aot_eager")
    compiled_out = compiled(ShardTensor.wrap(positions_compiled), batch_idx, num_graphs)
    compiled_grad = torch.autograd.grad(compiled_out.sum(), positions_compiled)[0]

    torch.testing.assert_close(eager_grad, compiled_grad, atol=1e-5, rtol=1e-5)
    graph_breaks = dict(torch._dynamo.utils.counters.get("graph_break", {}))
    assert not graph_breaks, f"unexpected graph breaks: {graph_breaks}"
