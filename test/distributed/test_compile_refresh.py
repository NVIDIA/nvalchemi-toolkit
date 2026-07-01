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

"""The compile-refresh graph pass — pass mechanics.

CPU, no distribution: proves the FX pass + backend do the right structural thing
with autograd intact, using a stand-in correction op (scale-by-3) so the assertions
are unambiguous. The real ``halo_scatter_correct_static`` op + routing semantics
are validated per model on GPU (the model halo-compile equivalence gates).

Three properties:
1. The pass inserts the correction on a node-scatter whose index traces to the
   tagged ``edge_index`` input — and *only* there: a per-graph scatter indexed by
   ``batch_idx`` is left alone.
2. Autograd flows through the inserted op (the gradient reflects its backward).
3. The backend fails safe: routing threaded + zero sites found -> it raises.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from nvalchemi.distributed.compile_refresh import (
    insert_halo_refresh,
    make_dd_halo_backend,
)

# Stand-in for halo_scatter_correct_static: scale-by-3 so its presence (fwd) and
# its backward (grad) are unmistakable. Ignores the routing args (shape-matched to
# the real op: node_feats + 4 routing tensors).
_NS = "probe_refresh"


@torch.library.custom_op(f"{_NS}::mark3", mutates_args=())
def mark3(
    x: torch.Tensor,
    si: torch.Tensor,
    rd: torch.Tensor,
    rr: torch.Tensor,
    no: torch.Tensor,
    ws: int,
) -> torch.Tensor:
    return x * 3.0


@mark3.register_fake
def _(x, si, rd, rr, no, ws):
    return torch.empty_like(x)


def _mark3_bwd(ctx, grad):
    return grad * 3.0, None, None, None, None, None


mark3.register_autograd(_mark3_bwd, setup_context=lambda ctx, inputs, output: None)

_MARK3 = torch.ops.probe_refresh.mark3.default
_ROUTING = ("_halo_si", "_halo_rd", "_halo_rr", "_halo_no")


class _TwoScatterMP(nn.Module):
    """A message-passing node-scatter (by ``edge_index``) AND a per-graph scatter
    (by ``batch_idx``). Only the former is a halo-refresh site."""

    def __init__(self, feat: int = 4) -> None:
        super().__init__()
        self.lin = nn.Linear(feat, feat)

    def forward(self, x, edge_index, batch_idx, _halo_si, _halo_rd, _halo_rr, _halo_no):
        # message passing: gather senders, message, scatter into receivers.
        msg = self.lin(x[edge_index[0]])
        node_out = torch.zeros_like(x)
        recv = edge_index[1].unsqueeze(-1).expand_as(msg)
        node_out = node_out.scatter_add(0, recv, msg)
        # per-graph reduction: indexed by batch_idx, NOT edge_index -> NOT a site.
        # Fixed graph count (test data has 2) keeps the graph break-free + traceable.
        graph_out = torch.zeros(2, x.shape[1], dtype=x.dtype, device=x.device)
        bcast = batch_idx.unsqueeze(-1).expand_as(x)
        graph_out = graph_out.scatter_add(0, bcast, x)
        # Keep the routing inputs LIVE in the traced graph (Dynamo prunes unused
        # args). They are graph inputs the pass wires the inserted op to.
        # Returned as an (ignored) output here.
        routing_live = torch.stack(
            [_halo_si.sum(), _halo_rd.sum(), _halo_rr.sum(), _halo_no.sum()]
        )
        return node_out, graph_out, routing_live


def _inputs(n=6, feat=4):
    torch.manual_seed(0)
    x = torch.randn(n, feat, requires_grad=True)
    edge_index = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 5]])
    batch_idx = torch.tensor([0, 0, 0, 1, 1, 1])
    # routing placeholders (unused by the stub). DISTINCT tensor objects —
    # Dynamo dedupes identical inputs to one placeholder, which would hide the
    # others from the name-keyed routing match.
    rt = [torch.zeros(1), torch.zeros(1), torch.zeros(1), torch.zeros(1)]
    return x, edge_index, batch_idx, *rt


def _backend():
    return make_dd_halo_backend(
        2, "aot_eager", correction_op=_MARK3, routing_names=_ROUTING
    )


def test_pass_corrects_edge_scatter_only_and_autograd_flows():
    m = _TwoScatterMP()
    x, ei, bi, *rt = _inputs()

    node_e, graph_e, _ = m(x, ei, bi, *rt)
    (gx_e,) = torch.autograd.grad(node_e.sum(), x, retain_graph=True)

    cm = torch.compile(m, backend=_backend(), fullgraph=False)
    xc = x.detach().clone().requires_grad_(True)
    node_c, graph_c, _ = cm(xc, ei, bi, *rt)
    (gx_c,) = torch.autograd.grad(node_c.sum(), xc)

    # (1) message-passing scatter corrected -> 3x; (2) backward flows -> 3x grad.
    torch.testing.assert_close(node_c, node_e * 3.0, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(gx_c, gx_e * 3.0, rtol=1e-5, atol=1e-5)
    # per-graph scatter (batch_idx) left untouched.
    torch.testing.assert_close(graph_c, graph_e, rtol=1e-5, atol=1e-5)


def test_insert_halo_refresh_reports_one_site():
    # Trace the module to an FX graph and run the pass directly (no compile),
    # asserting it finds exactly the edge_index site and wires the routing.
    import torch.fx

    m = _TwoScatterMP()
    gm = torch.fx.symbolic_trace(m)
    report = insert_halo_refresh(
        gm, correction_op=_MARK3, world_size=2, routing_names=_ROUTING,
        require_routing=True,
    )
    assert report.n_sites == 1, (report.n_sites, report.site_names)
    assert report.routing_present
    # the inserted op is now in the graph
    assert any(
        n.op == "call_function" and n.target is _MARK3 for n in gm.graph.nodes
    )


def test_backend_fails_safe_when_routing_present_but_no_site():
    # A graph with routing inputs but NO edge_index-keyed scatter must raise
    # rather than silently run with stale ghosts.
    class _NoScatter(nn.Module):
        def forward(self, x, edge_index, _halo_si, _halo_rd, _halo_rr, _halo_no):
            # routing threaded + kept live (returned), edge_index consumed, but NO
            # scatter keyed on it -> the backend must fail safe.
            live = torch.stack(
                [_halo_si.sum(), _halo_rd.sum(), _halo_rr.sum(),
                 _halo_no.sum(), edge_index.sum().float()]
            )
            return x * 2.0, live

    m = _NoScatter()
    x = torch.randn(4, 3, requires_grad=True)
    ei = torch.tensor([[0, 1], [1, 2]])
    rt = [torch.zeros(1), torch.zeros(1), torch.zeros(1), torch.zeros(1)]
    cm = torch.compile(m, backend=_backend(), fullgraph=False)
    with pytest.raises(RuntimeError, match="no message-passing node-scatter"):
        cm(x, ei, *rt)
