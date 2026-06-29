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

"""Context-aware helpers for domain-decomposed model wrappers.

Functions a model author calls inside a wrapper to express a distributed
operation by intent — "refresh my neighbor rows", "sum per system" — without
naming the mechanism. Each reads the live distributed context the framework
sets up for the forward and does the right thing under the halo policy, in
single-process, and under ``torch.compile``.

This is the shared home of logic that would otherwise be copy-pasted across
model wrappers.
"""

from __future__ import annotations

import functools
from typing import Any

import torch

from nvalchemi.distributed._core.compile_routing import (
    compile_routing_active,
    get_compile_routing,
)
from nvalchemi.distributed._core.context import current_dd_context
from nvalchemi.distributed._core.enums import Scope
from nvalchemi.distributed._core.particle_halo import (
    halo_forward_static_op,
    halo_scatter_correct_static_op,
)
from nvalchemi.distributed._core.per_system import per_system_reduce

__all__ = [
    "Scope",
    "distributed_method",
    "neighbor_refresh_adapters",
    "localize",
    "refresh_neighbors",
    "scatter_to_owners",
    "system_sum",
    "to_local",
]


def to_local(x: Any) -> Any:
    """Return the plain local tensor backing a ShardTensor, else ``x`` unchanged.

    Call this before handing a tensor to a kernel that must not see a
    ShardTensor. On the halo policy the result is the rank's owned+ghost block.

    Parameters
    ----------
    x : Any
        A tensor, ShardTensor, or any non-tensor value.

    Returns
    -------
    Any
        ``x.to_local()`` for a ShardTensor; ``x`` unchanged otherwise (a plain
        tensor or non-tensor passes straight through, so the call is always safe).
    """
    if x is None or not hasattr(x, "to_local"):
        return x
    return x.to_local()


def localize(data: dict[str, Any]) -> dict[str, Any]:
    """Run :func:`to_local` over every value in a model-input dict.

    Localize a whole input dict in one call so a kernel that consumes it never
    sees a ShardTensor.

    Parameters
    ----------
    data : dict[str, Any]
        A model-input dict whose values may include ShardTensors.

    Returns
    -------
    dict[str, Any]
        A shallow copy with each value run through :func:`to_local`; non-tensor
        entries (configs, ints) pass through unchanged.
    """
    return {k: to_local(v) for k, v in data.items()}


def distributed_method(body: Any) -> Any:
    """Decorate a ``MethodAdapter`` body that only diverges under the halo policy.

    Removes the boilerplate guard repeated across halo method adapters so the
    body holds only the distributed behavior. The wrapped replacement runs the
    original method verbatim whenever the live context is not on the halo path
    (single-process, or any call outside a distributed forward), and otherwise
    invokes ``body`` with the live context already in hand.

    Parameters
    ----------
    body : callable
        The halo behavior, called as
        ``body(ctx, original, instance, *args, **kwargs)`` where ``ctx`` is the
        live :class:`~nvalchemi.distributed._core.context.DistributedContext` and
        ``original`` is the unpatched method.

    Returns
    -------
    callable
        An ``(original, instance, *args, **kwargs)`` replacement suitable for
        :class:`~nvalchemi.distributed._core.adapter.MethodAdapter`.

    Examples
    --------
    >>> @distributed_method
    ... def _refresh_block(ctx, original, block, x, *args, **kwargs):
    ...     return original(block, refresh_neighbors(x), *args, **kwargs)
    """

    @functools.wraps(body)
    def wrapped(original: Any, instance: Any, *args: Any, **kwargs: Any) -> Any:
        ctx = current_dd_context()
        if not ctx.is_halo:
            return original(instance, *args, **kwargs)
        return body(ctx, original, instance, *args, **kwargs)

    return wrapped


def neighbor_refresh_adapters(
    modules: Any, *, output: int = 0, always: bool = False
) -> tuple:
    """Build adapters that recombine each module's per-node ``forward`` output
    across ranks.

    For message-passing blocks whose internal scatter the framework cannot reach:
    pass the live sub-modules (``model.interactions``); this finds the concrete
    classes that define ``forward`` and returns one :class:`MethodAdapter` per
    class. Each adapter runs the block, then applies :func:`scatter_to_owners` to
    forward output ``output`` (an ``int`` index when the block returns a tuple,
    else the whole output).

    By default this fires only inside a compiled DD region (the halo path needs
    it only under compile; eager halo corrects via dispatch). ``always=True``
    fires in eager too — the node-replicate strategy, where the block's
    per-node output is each rank's partial message sum and the recombine is the
    all-reduce that must run every forward. In single-process ``scatter_to_owners``
    is the identity, so the adapter is a no-op there regardless.

    Correcting the block's per-node output equals correcting its internal scatter
    when the downstream ops are linear in the message (true at MACE's interaction
    boundary: the nonlinear product basis is a separate downstream block, so it
    sees the recombined message).
    """
    from nvalchemi.distributed._core.adapter import MethodAdapter  # noqa: PLC0415

    def _refresh(original: Any, *args: Any, **kwargs: Any) -> Any:
        out = original(*args, **kwargs)
        if not always and not compile_routing_active():
            return out
        if isinstance(out, tuple):
            fixed = list(out)
            fixed[output] = scatter_to_owners(fixed[output])
            return type(out)(fixed)
        return scatter_to_owners(out)

    seen: dict[tuple, Any] = {}
    for m in modules:
        cls = type(m)
        seen.setdefault((cls.__module__, cls.__qualname__), cls)
    return tuple(MethodAdapter(cls, "forward", _refresh) for cls in seen.values())


def refresh_neighbors(x: torch.Tensor) -> torch.Tensor:
    """Populate this rank's neighbor (ghost) rows of a per-node tensor.

    Call this at the start of a message-passing block that reads a node's
    neighbors: it refreshes the ghost rows of ``x`` so each rank sees current
    neighbor features. Autograd-aware — gradients on the refreshed rows
    accumulate back to the owning ranks.

    On the halo policy ``x`` is ``[owned | ghost (| dead padding)]``; owned rows
    are exchanged into the ghost region and any trailing padding rows are
    preserved. In single-process this is the identity.

    Parameters
    ----------
    x : torch.Tensor
        ``(n_rows, *F)`` per-node features with this rank's owned rows
        first.

    Returns
    -------
    torch.Tensor
        Same shape as ``x`` with the neighbor rows populated.
    """
    # Under compile: use the fixed-shape static op wired to the step's routing
    # tensors (not current_dd_context, which would bake stale values). ``x`` is
    # already capped, so the op runs over the whole padded tensor.
    routing = get_compile_routing()
    if routing is not None:
        si, rd, rr, no, ws = routing
        return halo_forward_static_op(x, si, rd, rr, no, ws)
    ctx = current_dd_context()
    if not ctx.is_distributed:
        return x
    return ctx.policy.replicate(x, ctx)


def scatter_to_owners(out: torch.Tensor) -> torch.Tensor:
    """Fold per-edge contributions written into ghost rows back to owners.

    After a message-passing block scatters per-edge messages into nodes — leaving
    each rank's partial sums in its ghost rows — this accumulates those partials
    into the owning ranks and re-broadcasts, so every rank's owned and ghost rows
    hold the correct totals for the next block. Autograd-aware. Identity in
    single-process.

    Parameters
    ----------
    out : torch.Tensor
        ``(n_rows, *F)`` per-node tensor with this rank's partial sums in
        the ghost rows.

    Returns
    -------
    torch.Tensor
        Same shape, with owners and ghosts carrying the cross-rank totals.
    """
    # Under compile: the fixed-shape static op wired to the step's routing
    # tensors — the in-graph form of the eager reverse+forward below.
    routing = get_compile_routing()
    if routing is not None:
        si, rd, rr, no, ws = routing
        return halo_scatter_correct_static_op(out, si, rd, rr, no, ws)
    ctx = current_dd_context()
    if not ctx.is_distributed:
        return out
    return ctx.policy.fold(out, ctx)


def system_sum(
    vals: torch.Tensor,
    idx: torch.Tensor,
    n: int,
    scope: Scope = Scope.OWNED,
) -> torch.Tensor:
    """Sum per-node values into per-system totals, without double-counting.

    Each rank holds neighbor copies of atoms it does not own, so a plain
    ``scatter_add`` over all rows would over-count. This sums only this
    rank's owned rows and (for :attr:`Scope.OWNED`) all-reduces across the
    mesh to the true global per-system total. In single-process it is a
    plain ``scatter_add`` over all rows.

    Under compile it masks the ghost / dead rows by the routing's n_owned tensor
    (a tensor mask, not a dynamic slice, so the partition can drift without
    forcing a recompile) and reduces over all rows. A wrapper calls it the same
    way in both modes.

    Parameters
    ----------
    vals : torch.Tensor
        ``(n_rows, *F)`` per-node values with owned rows first.
    idx : torch.Tensor
        ``(n_rows,)`` integer system index for each row, in ``[0, n)``.
    n : int
        Number of systems in the (global) batch.
    scope : Scope, default ``Scope.OWNED``
        ``OWNED`` → owned-only sum + cross-rank all-reduce (global total on
        every rank). ``LOCAL`` → this rank's owned-only partial with no
        all-reduce (the framework's output consolidation finishes it).

    Returns
    -------
    torch.Tensor
        ``(n, *F)`` per-system totals (replicated on every rank for
        ``OWNED``; a per-rank partial for ``LOCAL``).
    """
    idx_long = idx.to(torch.long)
    # Under compile: mask the ghost / dead rows by the n_owned tensor (not a
    # dynamic ``[:n_owned]`` slice, which would recompile as the partition
    # drifts), then reduce over all rows (masked rows contribute 0).
    routing = get_compile_routing()
    if routing is not None:
        ctx = current_dd_context()
        _, _, _, n_owned_t, _ = routing
        rowidx = torch.arange(vals.shape[0], device=vals.device)
        owned = (
            (rowidx < n_owned_t)
            .reshape((-1,) + (1,) * (vals.ndim - 1))
            .to(vals.dtype)
        )
        masked = vals * owned
        if scope is Scope.OWNED:
            return per_system_reduce(masked, idx_long, n, ctx.halo_config)
        out = vals.new_zeros((n, *vals.shape[1:]))
        return out.index_add_(0, idx_long, masked)
    ctx = current_dd_context()
    if not ctx.is_distributed:
        out = vals.new_zeros((n, *vals.shape[1:]))
        return out.index_add_(0, idx_long, vals)
    n_owned = ctx.n_owned
    vals_owned = vals[:n_owned].contiguous()
    idx_owned = idx_long[:n_owned].contiguous()
    if scope is Scope.OWNED:
        # ``per_system_reduce`` needs only the mesh to all-reduce over. The halo
        # policy carries it on ``halo_config``; a halo-free policy (graph
        # parallel) supplies it straight off the context.
        cfg = ctx.halo_config
        if cfg is None:
            from types import SimpleNamespace  # noqa: PLC0415

            cfg = SimpleNamespace(mesh=ctx.mesh)
        return per_system_reduce(vals_owned, idx_owned, n, cfg)
    # LOCAL: per-rank partial, no all-reduce; consolidation finishes the sum.
    out = vals.new_zeros((n, *vals.shape[1:]))
    return out.index_add_(0, idx_owned, vals_owned)
