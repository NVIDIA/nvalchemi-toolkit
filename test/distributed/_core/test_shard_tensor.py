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

"""ShardTensor subclass-propagation and semantics tests.

Single-process tests. No distribution, no MACE, no halo context. The goal is
to verify ShardTensor behaves as a drop-in torch.Tensor for every op the
downstream models (ToyGNN, MACE) actually execute, so that the distributed
variants can be built on a trusted foundation.

Specifically we verify:
  - Subclass propagation through arithmetic, gather, zeros_like / new_zeros,
    F.linear, F.embedding, reshape/squeeze/unsqueeze, reductions, concat.
  - ``.shape`` returns local shape (standard Tensor semantics).
  - ``scatter_add_`` without an active halo_context behaves like plain
    scatter_add (no correction, same values).
  - Autograd flows through ShardTensor wrap → model forward → grad.
  - ToyGNN forward on ShardTensor inputs produces a ShardTensor output with
    values equal to a plain-tensor run.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from _toy_gnn import ToyGNN, brute_force_edges, build_fcc_argon

from nvalchemi.distributed._core.shard_tensor import (
    ShardTensor,
    _make_handler_output,
    _unwrap,
    _unwrap_grad_aware,
    clear_handlers,
    list_handlers,
    register_handler,
)
from nvalchemi.distributed.spec import SPEC_MPNN_HALO

pytestmark = pytest.mark.usefixtures("_session_gloo_pg")


def _assert_halo(x: object) -> None:
    assert isinstance(x, ShardTensor), f"expected ShardTensor, got {type(x).__name__}"


def test_unwrap_grad_aware_preserves_graph_through_handler() -> None:
    """Regression for the MACE distributed-autograd graph break.

    The wrapper-subclass tracks autograd on the WRAPPER; an op-result's
    ``_local_tensor`` is autograd-detached. A dispatch handler that computes on
    the plain local (via ``_unwrap``) and re-wraps would sever the graph —
    energy reaches ``autograd.grad`` with no ``grad_fn`` (the MACE symptom).
    ``_unwrap_grad_aware`` keeps the handler's computation connected so
    ``autograd.grad(energy, leaf)`` flows back, matching a plain reference.
    """
    leaf = torch.randn(5, 3, dtype=torch.float64, requires_grad=True)
    st = ShardTensor.wrap(leaf)
    feat = st * 2.0  # generic op: wrapper carries grad, local is detached
    assert not _unwrap(feat).requires_grad  # documents the detach

    # Simulate a handler: grad-aware unwrap → compute on plain → re-wrap.
    local = _unwrap_grad_aware(feat)
    assert local.requires_grad
    result = local.sum(dim=0, keepdim=True).expand(5, 3).contiguous() * 3.0
    out = _make_handler_output(result, feat)
    energy = out.sum()
    (g,) = torch.autograd.grad(energy, leaf)

    leaf_ref = leaf.detach().clone().requires_grad_(True)
    r_ref = (leaf_ref * 2.0).sum(dim=0, keepdim=True).expand(5, 3).contiguous() * 3.0
    (g_ref,) = torch.autograd.grad(r_ref.sum(), leaf_ref)
    torch.testing.assert_close(g, g_ref)


def test_arithmetic_propagates() -> None:
    a = ShardTensor.wrap(torch.randn(5, 3))
    _assert_halo(a + 1.0)
    _assert_halo(a - 2.0)
    _assert_halo(a * 0.5)
    _assert_halo(a / 2.0)
    _assert_halo(-a)
    _assert_halo(a + a)
    # ShardTensor + plain Tensor should still yield ShardTensor
    _assert_halo(a + torch.zeros(5, 3))


def test_indexing_propagates() -> None:
    a = ShardTensor.wrap(torch.arange(20, dtype=torch.float64).reshape(5, 4))
    _assert_halo(a[torch.tensor([0, 2, 4])])
    _assert_halo(a[:3])
    _assert_halo(a[1:4, :2])
    _assert_halo(a[..., 0])


def test_creation_methods_propagate() -> None:
    a = ShardTensor.wrap(torch.randn(4, 3))
    _assert_halo(torch.zeros_like(a))
    _assert_halo(torch.ones_like(a))
    _assert_halo(a.new_zeros((7, 3)))
    _assert_halo(a.new_empty((2, 3)))
    _assert_halo(a.new_full((3, 3), 2.5))


def test_linear_propagates() -> None:
    a = ShardTensor.wrap(torch.randn(4, 8, dtype=torch.float64))
    w = torch.randn(16, 8, dtype=torch.float64)
    b = torch.randn(16, dtype=torch.float64)
    out = F.linear(a, w, b)
    _assert_halo(out)
    assert out.shape == (4, 16)


def test_embedding_propagates() -> None:
    # nn.Embedding calls F.embedding(input, weight); input here is indices.
    # With ShardTensor indices, DTensor-style "mixed tensor" rejection would
    # bite — ShardTensor avoids this because it's a plain Tensor subclass.
    indices = ShardTensor.wrap(torch.tensor([0, 2, 1, 3], dtype=torch.long))
    weight = torch.randn(5, 8, dtype=torch.float64)
    out = F.embedding(indices, weight)
    _assert_halo(out)
    assert out.shape == (4, 8)


def test_reshape_propagates() -> None:
    a = ShardTensor.wrap(torch.arange(24, dtype=torch.float64))
    _assert_halo(a.reshape(6, 4))
    _assert_halo(a.view(4, 6))
    _assert_halo(a.unsqueeze(0))
    b = a.reshape(4, 6, 1)
    _assert_halo(b.squeeze(-1))
    _assert_halo(a.unflatten(0, (4, 6)))


def test_reductions_propagate() -> None:
    a = ShardTensor.wrap(torch.randn(4, 3, dtype=torch.float64))
    _assert_halo(a.sum())
    _assert_halo(a.sum(dim=0))
    _assert_halo(a.mean(dim=-1))
    _assert_halo(a.max(dim=1).values)


def test_activation_propagates() -> None:
    a = ShardTensor.wrap(torch.randn(4, 3, dtype=torch.float64))
    _assert_halo(F.silu(a))
    _assert_halo(F.relu(a))
    _assert_halo(a.sigmoid())


def test_matmul_propagates() -> None:
    a = ShardTensor.wrap(torch.randn(4, 8, dtype=torch.float64))
    w = torch.randn(8, 5, dtype=torch.float64)
    _assert_halo(a @ w)
    _assert_halo(torch.matmul(a, w))


def test_cat_propagates() -> None:
    a = ShardTensor.wrap(torch.randn(4, 3, dtype=torch.float64))
    b = torch.randn(2, 3, dtype=torch.float64)
    _assert_halo(torch.cat([a, b], dim=0))


def test_shape_reports_local_size() -> None:
    # ``.shape`` returns the local size (standard Tensor semantics).
    a = ShardTensor.wrap(torch.zeros(7, 3, dtype=torch.float64))
    assert a.shape == (7, 3)
    assert a.shape[0] == 7
    assert a.size(0) == 7
    assert a.numel() == 21
    assert a.ndim == 2


def test_scatter_add_without_halo_context_matches_plain() -> None:
    torch.manual_seed(0)
    src = torch.randn(10, 4, dtype=torch.float64)
    index_rows = torch.randint(0, 5, (10,))
    index = index_rows.unsqueeze(-1).expand(-1, 4)

    zeros_plain = torch.zeros(5, 4, dtype=torch.float64)
    expected = zeros_plain.clone().scatter_add_(0, index, src)

    # ShardTensor path
    ht_zeros = ShardTensor.wrap(torch.zeros(5, 4, dtype=torch.float64))
    got = ht_zeros.scatter_add_(0, index, src)

    _assert_halo(got)
    torch.testing.assert_close(got.unwrap(), expected, rtol=1e-12, atol=1e-14)


def test_scatter_add_return_type_is_halotensor() -> None:
    # chained usage: torch.zeros_like(x).scatter_add_(...) — the common pattern
    # in both ToyGNN and MACE.
    x = ShardTensor.wrap(torch.randn(6, 3, dtype=torch.float64))
    index = torch.tensor([[0], [1], [2], [0], [1], [2]]).expand(-1, 3)
    src = torch.randn(6, 3, dtype=torch.float64)
    out = torch.zeros_like(x).scatter_add_(0, index, src)
    _assert_halo(out)
    assert out.shape == (6, 3)


def test_autograd_through_halo_tensor() -> None:
    leaf = torch.randn(3, 4, dtype=torch.float64, requires_grad=True)
    h = ShardTensor.wrap(leaf)
    loss = (h * 2.0).sum()
    (grad,) = torch.autograd.grad(loss, leaf)
    expected = torch.full_like(leaf, 2.0)
    torch.testing.assert_close(grad, expected)


def test_autograd_through_scatter_add_without_context() -> None:
    # With no halo_context, our handler just does the functional scatter —
    # autograd must still work end-to-end for leaf → scatter output → loss.
    leaf = torch.randn(6, 3, dtype=torch.float64, requires_grad=True)
    index = torch.tensor([[0], [1], [2], [0], [1], [2]]).expand(-1, 3)

    h_leaf = ShardTensor.wrap(leaf)
    out = torch.zeros_like(h_leaf).scatter_add_(0, index, h_leaf)
    loss = out.sum()
    (grad,) = torch.autograd.grad(loss, leaf)
    # Each entry of leaf contributes to the scatter once, so grad = ones.
    torch.testing.assert_close(grad, torch.ones_like(leaf))


def test_toygnn_forward_halo_tensor_matches_plain() -> None:
    torch.manual_seed(7)
    dtype = torch.float64
    cutoff = 5.0
    positions, cell, atomic_numbers = build_fcc_argon(
        n_per_side=2, lattice_const=5.26, dtype=dtype
    )
    positions = positions + 0.05 * torch.randn_like(positions)
    pbc = torch.ones(3, dtype=torch.bool)

    torch.manual_seed(1234)
    model = ToyGNN(num_species=20, hidden=8, num_layers=2, r_cut=cutoff).to(dtype=dtype)
    model.eval()

    edge_index, edge_vec = brute_force_edges(positions, cutoff, cell, pbc)

    with torch.no_grad():
        out_plain = model(positions, atomic_numbers, edge_index, edge_vec)
        out_halo = model(
            positions,
            ShardTensor.wrap(atomic_numbers),
            edge_index,
            edge_vec,
        )

    _assert_halo(out_halo)
    torch.testing.assert_close(out_halo.unwrap(), out_plain, rtol=1e-12, atol=1e-14)


def test_toygnn_backward_halo_tensor_matches_plain() -> None:
    torch.manual_seed(11)
    dtype = torch.float64
    cutoff = 5.0
    positions, cell, atomic_numbers = build_fcc_argon(
        n_per_side=2, lattice_const=5.26, dtype=dtype
    )
    positions = positions + 0.05 * torch.randn_like(positions)
    pbc = torch.ones(3, dtype=torch.bool)

    torch.manual_seed(1234)
    model = ToyGNN(num_species=20, hidden=8, num_layers=2, r_cut=cutoff).to(dtype=dtype)
    model.eval()

    def forces(positions_req, wrap_z: bool) -> torch.Tensor:
        positions_req = positions_req.clone().requires_grad_(True)
        edge_index, edge_vec = brute_force_edges(positions_req, cutoff, cell, pbc)
        z_in = ShardTensor.wrap(atomic_numbers) if wrap_z else atomic_numbers
        e = model(positions_req, z_in, edge_index, edge_vec).sum()
        (grad,) = torch.autograd.grad(e, positions_req)
        return -grad.detach()

    f_plain = forces(positions, wrap_z=False)
    f_halo = forces(positions, wrap_z=True)
    torch.testing.assert_close(f_halo, f_plain, rtol=1e-12, atol=1e-14)


def test_list_handlers_includes_default_intercepts() -> None:
    handlers = list_handlers()
    op_names = [op for op, _ in handlers]
    assert any("scatter_add_" in n for n in op_names)
    assert any("index_add_" in n for n in op_names)
    assert any("index_copy_" in n for n in op_names)


def test_user_registered_handler_fires() -> None:
    fired = []

    def my_handler(*args, **kwargs):
        fired.append(True)
        return args[0].unwrap() * 42.0

    # Register a handler on torch.sigmoid specifically for our tests.
    # Note: predicate=None always matches — take care to clean up.
    register_handler(torch.sigmoid, handler=my_handler, name="test_sigmoid")
    try:
        t = ShardTensor.wrap(torch.ones(3))
        result = torch.sigmoid(t)
        assert fired, "custom handler was not invoked"
        torch.testing.assert_close(result, torch.full((3,), 42.0))
    finally:
        clear_handlers(torch.sigmoid)


def test_handler_branches_internally() -> None:
    """A handler registered for an op is the sole handler for that op; if its
    behavior is conditional it branches internally (the registry is
    one-handler-per-op, not a predicate race)."""
    calls = []

    def handler(*args, **kwargs):
        x = args[0]
        if x.shape[0] < 5:
            calls.append("small")
            return x.unwrap() + 1.0
        calls.append("large")
        return x.unwrap() - 1.0

    register_handler(torch.tanh, handler=handler, name="test_tanh")
    try:
        _ = torch.tanh(ShardTensor.wrap(torch.zeros(3)))
        _ = torch.tanh(ShardTensor.wrap(torch.zeros(10)))
        assert calls == ["small", "large"]
    finally:
        clear_handlers(torch.tanh)


def test_index_add_without_halo_meta_matches_plain() -> None:
    torch.manual_seed(0)
    src = torch.randn(10, 4, dtype=torch.float64)
    indices = torch.randint(0, 5, (10,))

    expected = torch.zeros(5, 4, dtype=torch.float64).index_add_(0, indices, src)

    # No meta on the wrapped tensor → predicate returns False → default
    # Tensor.__torch_function__ handles the op with plain semantics.
    ht_zeros = ShardTensor.wrap(torch.zeros(5, 4, dtype=torch.float64))
    got = ht_zeros.index_add_(0, indices, src)

    assert isinstance(got, ShardTensor)
    torch.testing.assert_close(got.unwrap(), expected, rtol=1e-12, atol=1e-14)


def test_index_copy_without_halo_meta_matches_plain() -> None:
    torch.manual_seed(0)
    src = torch.randn(3, 4, dtype=torch.float64)
    indices = torch.tensor([0, 2, 4])

    expected = torch.zeros(5, 4, dtype=torch.float64).index_copy_(0, indices, src)

    ht_zeros = ShardTensor.wrap(torch.zeros(5, 4, dtype=torch.float64))
    got = ht_zeros.index_copy_(0, indices, src)

    assert isinstance(got, ShardTensor)
    torch.testing.assert_close(got.unwrap(), expected, rtol=1e-12, atol=1e-14)


def test_metadata_propagates_through_elementwise_ops() -> None:
    """After wrap + elementwise op, the output ShardTensor should
    carry the same metadata as the input (spec-driven dispatch relies on
    metadata being on the tensor)."""
    from unittest.mock import MagicMock

    meta = MagicMock(n_padded=10, n_owned=8)
    cfg = MagicMock()

    x = ShardTensor.wrap(
        torch.zeros(10, 3, dtype=torch.float64),
        meta=meta,
        config=cfg,
        spec=SPEC_MPNN_HALO,
    )
    y = x + 1.0
    assert isinstance(y, ShardTensor)
    assert y.meta is meta
    assert y.config is cfg

    z = torch.relu(x)
    assert isinstance(z, ShardTensor)
    assert z.meta is meta
    assert z.config is cfg
