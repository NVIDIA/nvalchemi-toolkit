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

"""Exactness of the async-overlap two-pass split (proposal §3.0), single-process.

Proves that running an opaque scatter-add message module on the sender-resident
edge bucket (owned features) plus the remote bucket (exchanged features), then
summing, reproduces the full-message reference — through the actual
:func:`run_overlapped` driver + :class:`AsyncExchange`/:class:`LocalityPartition`
protocols. Covers BOTH index-space regimes the two production strategies present:
graph-parallel (exchanged = the full gathered node set) and halo (exchanged = the
``[owned | ghost]`` padded set). No distribution / GPU — this isolates the split
math from the collectives.
"""

from __future__ import annotations

import torch

from nvalchemi.distributed._core.overlap import (
    LocalityPartition,
    run_overlapped,
    split_by_sender_residency,
)


def _message_fn(features: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    """Opaque scatter-add message layer: a per-edge, sender-dependent feature
    aggregated into receivers. Additive over edges (the overlap precondition),
    output allocated to ``features.shape[0]`` (the standard MPNN signature)."""
    send, recv = edge_index[0], edge_index[1]
    edge_feat = torch.tanh(features[send]) * 0.7 + features[recv] * 0.3
    out = torch.zeros(
        features.shape[0], features.shape[1], dtype=features.dtype, device=features.device
    )
    return out.scatter_add_(0, recv.unsqueeze(-1).expand_as(edge_feat), edge_feat)


class _StashExchange:
    """In-process :class:`AsyncExchange`: ``start`` records the owned features,
    ``wait`` returns the pre-known exchanged set (what the real collective would
    produce). Stands in for the strategy's all-gather / halo borrow so the split
    math can be checked without any process group."""

    def __init__(self, exchanged: torch.Tensor) -> None:
        self._exchanged = exchanged

    def start(self, owned: torch.Tensor) -> object:
        return owned

    def wait(self, handle: object) -> torch.Tensor:
        return self._exchanged


def _build_partition(
    edge_index: torch.Tensor, n_owned: int
) -> LocalityPartition:
    """Split owned-receiver edges by sender residency (sender in the owned block
    ``[0:n_owned]``). Sender ids already index the correct feature space in each
    bucket: local into owned (``< n_owned``), remote into the exchanged set."""
    resident = edge_index[0] < n_owned
    local_edges, remote_edges = split_by_sender_residency(edge_index, resident)
    return LocalityPartition(
        local_edges=local_edges, remote_edges=remote_edges, n_receivers=n_owned
    )


def _case(n_exchanged: int, n_owned: int, feat: int, n_edges: int, seed: int):
    """Build a full exchanged feature set with the owned block first, and an
    edge set whose receivers are all owned. Returns (exchanged, owned, edges)."""
    g = torch.Generator().manual_seed(seed)
    exchanged = torch.randn(n_exchanged, feat, dtype=torch.float64, generator=g)
    senders = torch.randint(0, n_exchanged, (n_edges,), generator=g)
    receivers = torch.randint(0, n_owned, (n_edges,), generator=g)
    edges = torch.stack([senders, receivers])
    return exchanged, exchanged[:n_owned].clone(), edges


def _assert_split_exact(n_exchanged: int, n_owned: int) -> None:
    exchanged, owned, edges = _case(n_exchanged, n_owned, feat=6, n_edges=300, seed=0)
    # Reference: the full message over the exchanged set, owned receiver rows.
    reference = _message_fn(exchanged, edges)[:n_owned]
    # Overlapped: two-pass split through the driver + protocols.
    exchange = _StashExchange(exchanged)
    partition = _build_partition(edges, n_owned)
    got = run_overlapped(_message_fn, owned, exchange, partition)
    assert got.shape == reference.shape
    torch.testing.assert_close(got, reference, rtol=1e-10, atol=1e-10)
    # Sanity: the split actually exercised both buckets.
    assert partition.local_edges.shape[1] > 0
    assert partition.remote_edges.shape[1] > 0


def test_overlap_split_exact_graph_parallel() -> None:
    """GP regime: exchanged = full gathered node set (owned block + other ranks')."""
    _assert_split_exact(n_exchanged=64, n_owned=20)


def test_overlap_split_exact_halo() -> None:
    """Halo regime: exchanged = [owned | ghost] padded set (fewer 'remote' rows)."""
    _assert_split_exact(n_exchanged=28, n_owned=20)


def _mk_config():
    from nvalchemi.distributed.config import DomainConfig

    return DomainConfig(cutoff=5.0)


def test_graph_partition_strategy_locality_partition() -> None:
    """GraphPartitionStrategy.locality_partition: split by owner_rank[sender]==rank,
    remap resident senders to owned space via local_index; routed through the
    driver it must reproduce the full-message reference."""
    from nvalchemi.distributed._core.placement import ShardRouting
    from nvalchemi.distributed._core.storage_policy import GraphParallelPolicy
    from nvalchemi.distributed.strategy import GraphPartitionStrategy

    n_owned, n_global, feat = 15, 30, 6
    # Owned block first (rank 0 owns rows [0:15]); rank 1 owns [15:30].
    owner_rank = torch.cat([torch.zeros(15, dtype=torch.long), torch.ones(15, dtype=torch.long)])
    local_index = torch.cat([torch.arange(15), torch.arange(15)])
    meta = ShardRouting(
        n_owned=n_owned, n_global=n_global, owner_rank=owner_rank, local_index=local_index
    )
    g = torch.Generator().manual_seed(1)
    full = torch.randn(n_global, feat, dtype=torch.float64, generator=g)
    edges = torch.stack(
        [torch.randint(0, n_global, (300,), generator=g),   # global senders
         torch.randint(0, n_owned, (300,), generator=g)]    # owned receivers
    )
    strat = GraphPartitionStrategy(GraphParallelPolicy(), _mk_config(), rank=0)
    part = strat.locality_partition(edges, meta)
    assert part.remote_edges.shape[1] > 0 and part.local_edges.shape[1] > 0
    got = run_overlapped(_message_fn, full[:n_owned].clone(), _StashExchange(full), part)
    ref = _message_fn(full, edges)[:n_owned]
    torch.testing.assert_close(got, ref, rtol=1e-10, atol=1e-10)


def test_halo_strategy_locality_partition() -> None:
    """HaloStrategy.locality_partition: split by sender < n_owned (ghost senders
    index the padded set); routed through the driver reproduces the reference."""
    from nvalchemi.distributed._core.halo_types import ParticleHaloMetadata
    from nvalchemi.distributed._core.storage_policy import HaloStoragePolicy
    from nvalchemi.distributed.strategy import HaloStrategy

    n_owned, n_padded, feat = 15, 25, 6
    meta = ParticleHaloMetadata(
        n_owned=n_owned, n_padded=n_padded, send_indices=[], send_sizes=[], recv_sizes=[]
    )
    g = torch.Generator().manual_seed(2)
    padded = torch.randn(n_padded, feat, dtype=torch.float64, generator=g)
    edges = torch.stack(
        [torch.randint(0, n_padded, (300,), generator=g),   # owned or ghost senders
         torch.randint(0, n_owned, (300,), generator=g)]     # owned receivers
    )
    strat = HaloStrategy(HaloStoragePolicy(), _mk_config(), rank=0)
    part = strat.locality_partition(edges, meta)
    assert part.remote_edges.shape[1] > 0 and part.local_edges.shape[1] > 0
    got = run_overlapped(_message_fn, padded[:n_owned].clone(), _StashExchange(padded), part)
    ref = _message_fn(padded, edges)[:n_owned]
    torch.testing.assert_close(got, ref, rtol=1e-10, atol=1e-10)


def test_overlap_all_local_is_noop_split() -> None:
    """When every sender is resident, remote bucket is empty and the result is
    just the local message (no exchange contribution)."""
    n_owned = 16
    g = torch.Generator().manual_seed(3)
    owned = torch.randn(n_owned, 5, dtype=torch.float64, generator=g)
    edges = torch.stack(
        [
            torch.randint(0, n_owned, (120,), generator=g),
            torch.randint(0, n_owned, (120,), generator=g),
        ]
    )
    reference = _message_fn(owned, edges)
    exchange = _StashExchange(owned)
    partition = _build_partition(edges, n_owned)
    assert partition.remote_edges.shape[1] == 0
    got = run_overlapped(_message_fn, owned, exchange, partition)
    torch.testing.assert_close(got, reference, rtol=1e-10, atol=1e-10)
