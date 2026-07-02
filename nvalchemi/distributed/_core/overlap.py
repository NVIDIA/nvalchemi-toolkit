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

"""Strategy-agnostic core for async comm/compute overlap of the per-layer DD
collective.

A message layer computes, for this rank's **owned receivers**, an aggregation
over edges whose *sender* feature is either already resident on this rank or must
be fetched by the strategy's per-layer exchange (an all-gather for
graph-parallel; an all-to-all ghost borrow for halo). Splitting the owned-receiver
edges by **sender residency** exposes compute that is independent of the exchange:

    handle = exchange.start(owned_x)              # issue the collective (async-capable)
    g_local  = message_fn(owned_x,  local_edges)  # sender resident — runs during the collective
    x_exch   = exchange.wait(handle)              # exchanged features (full for GP; [owned|ghost] halo)
    g_remote = message_fn(x_exch,   remote_edges) # sender needed the exchange
    out      = g_local + g_remote[:n_receivers]   # exact: aggregation is linear over a disjoint split

Only the two seams — :class:`AsyncExchange` and :class:`LocalityPartition` —
vary by strategy, and both are functions of the strategy's *layout* (owner_rank
for GP, n_owned for halo), so the strategy owns them. The model contributes only
its own message-module forward (``message_fn``), invoked once per edge bucket by
a spec-declared adapter — the model body is never edited.

Overlap is **OFF** here (synchronous two-pass): this isolates the correctness of
the split from the scheduling. Enabling the actual concurrency — either via the
inductor compute/comm reorder pass or a manual async handle — is a later phase;
it does not change this contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Protocol, runtime_checkable

import torch


@runtime_checkable
class AsyncExchange(Protocol):
    """A strategy's per-layer node-feature exchange, factored into issue/complete
    so the collective can be launched before the compute that depends on it.

    ``start`` issues the collective (asynchronously when the implementation
    supports it) and returns an opaque handle; ``wait`` blocks on that handle and
    returns the exchanged features — the **full gathered** node set for
    graph-parallel, or the **``[owned | ghost]`` padded** set for halo. In both
    the owned block is rows ``[0:n_owned]``. The backward (reduce-scatter for the
    gather; scatter-correct for the halo borrow) is carried by the underlying
    autograd primitive the implementation calls, so the driver stays
    autograd-transparent.
    """

    def start(self, owned: torch.Tensor) -> Any:
        """Issue the exchange of ``owned`` features; return a wait handle."""
        ...

    def wait(self, handle: Any) -> torch.Tensor:
        """Block on ``handle``; return the exchanged features (owned block first)."""
        ...


@dataclass(frozen=True)
class LocalityPartition:
    """Static (per-NL-rebuild) split of a message layer's owned-receiver edges by
    whether the **sender** feature is already resident on this rank.

    Both edge tensors are ``(2, E)`` ``[sender; receiver]``. Receivers are always
    owned (rows ``0:n_receivers``). Sender ids are pre-remapped into the feature
    space each bucket is evaluated in:

    * ``local_edges`` — sender resident; senders indexed into the **owned**
      feature tensor (``0:n_owned``). Computable with no exchange.
    * ``remote_edges`` — sender needs the exchange; senders indexed into the
      **exchanged** feature tensor (full for GP, padded for halo).

    ``n_receivers`` is this rank's owned node count. ``owned_offset`` is where the
    owned receiver rows sit in the **exchanged** feature space that the remote
    pass scatters into: ``0`` for halo (owned block is first in ``[owned|ghost]``),
    but the rank's block offset for graph-parallel (the all-gather concatenates
    owned blocks by rank). The local pass always scatters into ``[0:n_receivers]``
    of the owned tensor; the two buckets are summed over the owned rows.
    """

    local_edges: torch.Tensor
    remote_edges: torch.Tensor
    n_receivers: int
    owned_offset: int = 0


# A model's message-module forward, invoked on an edge subset: it takes the node
# features it should index as senders and an ``(2, E)`` edge tensor, and returns a
# per-node aggregation whose rows ``0:n_receivers`` are the owned receivers (the
# standard scatter-add message-passing signature). The framework never writes
# one — it is the wrapped module's own ``forward``, bound by the adapter.
MessageFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def run_overlapped(
    message_fn: MessageFn,
    owned_features: torch.Tensor,
    exchange: AsyncExchange,
    partition: LocalityPartition,
) -> torch.Tensor:
    """Two-pass overlapped message aggregation (overlap OFF: synchronous).

    Issues the exchange, computes the resident-sender messages from owned
    features, waits, computes the remote-sender messages from the exchanged
    features, and sums the two — exact because a disjoint edge partition of a
    linear (scatter-add) aggregation sums to the whole.
    """
    handle = exchange.start(owned_features)
    g_local = message_fn(owned_features, partition.local_edges)
    exchanged = exchange.wait(handle)
    g_remote = message_fn(exchanged, partition.remote_edges)
    n = partition.n_receivers
    off = partition.owned_offset
    return g_local[:n] + g_remote[off : off + n]


def split_by_sender_residency(
    edge_index: torch.Tensor,
    sender_is_resident: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Partition ``edge_index`` ``(2, E)`` by a boolean per-edge mask.

    ``sender_is_resident[e]`` is True when edge ``e``'s sender is owned by this
    rank (no exchange needed). Returns ``(local_edges, remote_edges)``. The
    strategy computes the mask from its layout — ``owner_rank[sender] == rank``
    for graph-parallel, ``sender < n_owned`` for halo — and remaps the sender ids
    into the appropriate feature space before building the
    :class:`LocalityPartition`.
    """
    return edge_index[:, sender_is_resident], edge_index[:, ~sender_is_resident]


def overlapped_message(
    original_forward: MessageFn,
    features: torch.Tensor,
    edge_index: torch.Tensor,
) -> torch.Tensor:
    """Overlap-aware invocation of a message module's own ``forward``.

    Reads the active DD context: with a strategy present it obtains that
    strategy's :class:`AsyncExchange` + :class:`LocalityPartition` (from the
    per-step ``gather_meta`` / ``halo_meta``) and drives :func:`run_overlapped`,
    calling ``original_forward`` once per edge bucket. Outside a distributed scope
    (no strategy) it is a pass-through — the single-process model is untouched.

    This is the whole adapter body: a spec-declared ``ModuleForwardAdapter`` binds
    it to a model's message module, so overlap is added by naming that module in
    the ``distribution_spec`` — the model ``forward`` is never edited.
    """
    from nvalchemi.distributed._core.context import (  # noqa: PLC0415
        current_dd_context,
    )

    ctx = current_dd_context()
    strategy = getattr(ctx, "strategy", None)
    if strategy is None:
        return original_forward(features, edge_index)
    meta = ctx.gather_meta if ctx.gather_meta is not None else ctx.halo_meta
    exchange = strategy.async_exchange(meta, getattr(ctx, "halo_config", None))
    partition = strategy.locality_partition(edge_index, meta)
    return run_overlapped(original_forward, features, exchange, partition)


class BufferPool:
    """Reuse fixed-shape collective buffers across layers and MD steps instead of
    allocating per layer (proposal §3.C). Keyed by ``(shape, dtype, device)``;
    the caller writes in place. Under fixed-shape (capped) compilation the shapes
    are graph constants, so the pool holds a small stable set.
    """

    def __init__(self) -> None:
        self._buffers: dict[tuple[Any, ...], torch.Tensor] = {}

    def get(
        self, shape: tuple[int, ...], dtype: torch.dtype, device: torch.device
    ) -> torch.Tensor:
        key = (tuple(shape), dtype, device)
        buf = self._buffers.get(key)
        if buf is None:
            buf = torch.empty(shape, dtype=dtype, device=device)
            self._buffers[key] = buf
        return buf

    def clear(self) -> None:
        self._buffers.clear()
