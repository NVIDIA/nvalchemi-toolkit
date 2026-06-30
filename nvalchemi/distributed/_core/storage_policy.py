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

"""Storage policies: how a field's *local storage* relates to its *placement*.

A :class:`StoragePolicy` is the per-field declaration that a distributed
collection consumes to scatter, view, and gather one field. The placement
stays honest (``Shard(0)`` / ``Replicate()``); the policy says how this rank's
local storage relates to it and how to move the data on/off the mesh.

Two policies ship here:

* :class:`PlainShard` — ``Shard(0)``: each rank stores only its owned rows as a
  ``ShardTensor``.
* :class:`HaloStoragePolicy` — ``Shard(0)`` + a borrowed-row overlay that
  intercepts ``scatter_add`` for per-layer halo correction. Note the overlay is
  built downstream (per message-passing layer), not at collection-construction
  time — so for *collection* construction this behaves like a plain shard.

The policies are domain-agnostic (rows, ranks, shards — no chemistry). They
own both halves of the protocol: the construction/transport methods
(``place_from_*`` / ``to_local`` / ``full_tensor``) and the op-dispatch methods
(``scatter`` / ``gather`` by global index, with halo correction) that the
ShardTensor dispatch handlers route through.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import torch
import torch.distributed as dist

__all__ = [
    "StoragePolicy",
    "PlainShard",
    "GraphParallelPolicy",
    "GraphReplicatePolicy",
    "HaloStoragePolicy",
    "RefreshOnlyHaloPolicy",
    "row_offsets",
    "policy_to_dict",
    "policy_from_dict",
    "register_policy_kind",
]


def row_offsets(sizes: list[int]) -> list[int]:
    """Prefix-sum row offsets for per-rank ``sizes`` (length ``W + 1``)."""
    offsets = [0]
    for s in sizes:
        offsets.append(offsets[-1] + s)
    return offsets


def _make_shard_tensor(local_t: torch.Tensor, mesh: Any, sizes: list[int]) -> Any:
    """Wrap this rank's owned rows as a ``Shard(0)`` ShardTensor whose
    per-rank ``sharding_shapes`` are taken from ``sizes`` (uneven across
    ranks). Lazy backend import keeps the policy importable without the
    physicsnemo backend on the path."""
    from torch.distributed.tensor import Shard

    from nvalchemi.distributed._core._st_backend import ShardTensor

    trailing = tuple(local_t.shape[1:])
    return ShardTensor.from_local(
        local_t.contiguous(),
        mesh,
        (Shard(0),),
        sharding_shapes={0: tuple(torch.Size((s,) + trailing) for s in sizes)},
    )


def _gloo_safe_gather(stored: Any, mesh: Any, dst: int) -> torch.Tensor:
    """Gloo-safe gather of a ``Shard(0)`` ShardTensor's rows onto rank *dst*.

    ``ShardTensor.full_tensor()`` routes through an ``all_gather`` that requires
    same-sized local tensors; gloo rejects uneven allgathers (e.g. when one rank
    owns 0 rows), and this path is exercised by the distributed end-to-end
    tests. Instead, use explicit send/recv driven by the spec's per-rank sizes.

    All ranks must call this (the send/recv is collective). Non-*dst* ranks
    return an empty placeholder with the matching trailing shape.
    """
    local_rank = mesh.get_local_rank()
    world_size = mesh.size()
    group = mesh.get_group()

    shapes = stored._spec.sharding_shapes()[0]
    local = stored.to_local().contiguous()

    if world_size == 1:
        return local.clone()

    if local_rank == dst:
        parts: list[torch.Tensor] = []
        for r in range(world_size):
            if r == dst:
                parts.append(local.clone())
            else:
                buf = torch.empty(shapes[r], dtype=local.dtype, device=local.device)
                if buf.numel() > 0:
                    dist.recv(buf, src=r, group=group)
                parts.append(buf)
        return torch.cat(parts, dim=0)

    if local.numel() > 0:
        dist.send(local, dst=dst, group=group)
    return torch.empty(
        (0,) + tuple(local.shape[1:]), dtype=local.dtype, device=local.device
    )


@runtime_checkable
class StoragePolicy(Protocol):
    """Per-field declaration of how local storage relates to its placement.

    A collection consumes one policy per field to (a) build this rank's stored
    field from a distributed source, (b) produce the local-rank view, and (c)
    gather the full tensor back. The ``placement`` is the honest DTensor
    placement the field would carry. In addition, the policy owns the cross-rank
    op behavior — :meth:`scatter` / :meth:`gather` by global index — that the
    ShardTensor dispatch handlers route through.
    """

    @property
    def placement(self) -> Any:
        """Honest placement (``Shard(0)`` / ``Replicate()``)."""
        ...

    def place_from_full(
        self, full_t: torch.Tensor, *, mesh: Any, sizes: list[int], local_rank: int
    ) -> Any:
        """Build this rank's stored field from the full ``(n_global, *F)`` tensor
        (already broadcast to every rank). ``sizes`` are per-rank owned-row
        counts (sum == ``n_global``)."""
        ...

    def place_from_local(
        self, local_t: torch.Tensor, *, mesh: Any, sizes: list[int]
    ) -> Any:
        """Build this rank's stored field from its already-local rows."""
        ...

    def to_local(self, stored: Any) -> torch.Tensor:
        """This rank's local-storage view of the field (no communication)."""
        ...

    def full_tensor(
        self, stored: Any, *, mesh: Any, dst: int | None = None
    ) -> torch.Tensor:
        """Reconstruct the semantic global tensor (collective).

        ``dst=int`` materializes onto that rank (others may get a placeholder);
        ``dst=None`` materializes on every rank. (A policy whose gather is
        inherently symmetric — e.g. :class:`HaloStoragePolicy` — materializes on
        all ranks regardless of ``dst``.)
        """
        ...

    def scatter(self, stored: Any, dim: int, index: Any, src: Any) -> Any:
        """Cross-rank ``scatter_add`` by global index (overlay/route-aware)."""
        ...

    def gather(self, stored: Any, dim: int, index: Any) -> Any:
        """Cross-rank gather by global index (overlay/route-aware)."""
        ...

    def replicate(self, x: Any, ctx: Any) -> Any:
        """Per-message-passing-layer input sync of a per-node tensor.

        Backs :func:`~nvalchemi.distributed.helpers.refresh_neighbors`: make this
        rank see the neighbor node features its edges read (refresh ghost rows,
        all-gather to a replicated tensor, ...). Default identity."""
        ...

    def fold(self, out: Any, ctx: Any) -> Any:
        """Per-layer output fold of per-edge contributions back to owners.

        Backs :func:`~nvalchemi.distributed.helpers.scatter_to_owners`. Default
        identity (a strategy whose scatter is already complete locally)."""
        ...

    @property
    def partition_mode(self) -> str:
        """How :meth:`ShardedBatch.from_batch` assigns atoms to ranks for this
        strategy: ``"spatial"`` (locality-preserving, halo) or
        ``"contiguous_block"`` (balanced index ranges, graph parallel)."""
        ...

    def build_topology(self, config: Any, sharded: Any) -> Any:
        """Return ``(partitioner, halo_config | None)`` for this strategy.

        Owns how atoms are assigned to ranks and any ghost geometry — the
        framework calls it once to initialize, so a strategy plugs in without a
        framework type-switch."""
        ...

    def run_forward(
        self, dist_model: Any, sharded: Any, wired_fields: Any
    ) -> "dict[str, Any]":
        """Run this strategy's distributed forward and return the consolidated
        outputs. The framework dispatches the forward through here."""
        ...

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly ``{"kind": ..., ...}`` record; inverse is the ``kind``
        registered via :func:`register_policy_kind`."""
        ...


@dataclass(frozen=True)
class PlainShard:
    """``Shard(0)`` storage substrate: each rank stores only its owned rows.

    The base row-sharded storage policy used internally by
    :class:`~nvalchemi.distributed.sharded_batch.ShardedBatch` to hold every
    model's per-atom fields (the halo path re-promotes positions to the halo
    spec on top of this). It is storage-only — it carries the ``Shard(0)``
    placement and ``to_local`` / ``full_tensor`` materialization, but no
    cross-rank gather/scatter compute; :class:`HaloStoragePolicy` is the only
    cross-rank storage policy. Not part of the public API and not selectable as
    a model's ``distribution_spec`` policy."""

    @property
    def placement(self) -> Any:
        from torch.distributed.tensor import Shard

        return Shard(0)

    @property
    def partition_mode(self) -> str:
        return "contiguous_block"

    def place_from_full(
        self, full_t: torch.Tensor, *, mesh: Any, sizes: list[int], local_rank: int
    ) -> Any:
        offsets = row_offsets(sizes)
        local_t = full_t[offsets[local_rank] : offsets[local_rank + 1]]
        return _make_shard_tensor(local_t, mesh, sizes)

    def place_from_local(
        self, local_t: torch.Tensor, *, mesh: Any, sizes: list[int]
    ) -> Any:
        return _make_shard_tensor(local_t, mesh, sizes)

    def to_local(self, stored: Any) -> torch.Tensor:
        return stored.to_local()

    def full_tensor(
        self, stored: Any, *, mesh: Any, dst: int | None = None
    ) -> torch.Tensor:
        if dst is not None:
            # Gloo-safe gather onto the single rank *dst*.
            return _gloo_safe_gather(stored, mesh, dst)
        # dst=None → materialize on every rank via an uneven all-gather.
        from nvalchemi.distributed._core.gather_primitives import (
            _all_gather_v_rows,
            mesh_group,
        )

        local = stored.to_local().contiguous()
        if not dist.is_initialized() or mesh.size() == 1:
            return local.clone()
        sizes = [int(s[0]) for s in stored._spec.sharding_shapes()[0]]
        return _all_gather_v_rows(local, sizes, mesh_group(mesh))

    def scatter(self, stored: Any, dim: int, index: Any, src: Any) -> Any:
        raise NotImplementedError(
            "PlainShard is a storage-only policy with no cross-rank scatter. "
            "Use HaloStoragePolicy for cross-rank scatter."
        )

    def gather(self, stored: Any, dim: int, index: Any) -> Any:
        raise NotImplementedError(
            "PlainShard is a storage-only policy with no cross-rank gather. "
            "Use HaloStoragePolicy for cross-rank gather."
        )

    def replicate(self, x: Any, ctx: Any) -> Any:
        return x

    def fold(self, out: Any, ctx: Any) -> Any:
        return out

    def build_topology(self, config: Any, sharded: Any) -> Any:
        raise NotImplementedError(
            "PlainShard is storage-only and not a selectable forward strategy."
        )

    def run_forward(
        self, dist_model: Any, sharded: Any, wired_fields: Any
    ) -> "dict[str, Any]":
        raise NotImplementedError(
            "PlainShard is storage-only and not a selectable forward strategy."
        )

    def to_dict(self) -> dict[str, Any]:
        return {"kind": "shard"}


@dataclass(frozen=True)
class HaloStoragePolicy:
    """Halo storage: a ``Shard(0)`` of the OWNED rows + a borrowed-row overlay.

    This is the honest semantic layer for halo-padded fields. Its
    ``placement`` of ``Shard(0)`` is the conceptual truth — the owned rows are a
    row-shard of the global tensor; the halo rows are a *declared overlay*, not
    part of the global tensor. The local storage (``stored._local_tensor``) is
    the padded ``[n_owned + n_halo, *F]`` view the model operates on; the
    overlay-aware operations live here.

    Note: the underlying ``ShardTensorSpec``'s placement is left as a documented
    placeholder (an honest ``Shard(0)`` spec would need every rank's padded row
    count — an all-gather on every construction). This policy, not the base
    spec, carries the honest semantics; cross-rank routing flows through the
    registered dispatch handlers via :meth:`scatter` / :meth:`gather`.

    The ``scatter_mode`` / ``gather_mode`` capture the cross-rank behavior a
    field needs (e.g. UMA's wrapper sets ``scatter_mode="local"`` to skip halo
    correction). The dispatch classifier reads them to decide whether the halo
    branch fires. The topology (``n_owned`` / routing) and process group live on
    the ShardTensor (``_meta`` / ``_config``); the policy reads them per call.

    Parameters
    ----------
    scatter_mode
        ``"halo_correction"`` — local scatter + halo reverse/forward exchange;
        ``"local"`` — purely local scatter (no halo synchronization).
    gather_mode
        ``"halo_read"`` — refresh borrowed halo rows before gathering;
        ``"local"`` — indices are local, nothing cross-rank.
    """

    scatter_mode: str = "halo_correction"
    gather_mode: str = "halo_read"

    @property
    def placement(self) -> Any:
        from torch.distributed.tensor import Shard

        return Shard(0)

    @property
    def partition_mode(self) -> str:
        return "spatial"

    def to_local(self, stored: Any) -> torch.Tensor:
        # The padded view (owned + halo) — what the model operates on.
        return stored._local_tensor

    def full_tensor(
        self, stored: Any, *, mesh: Any, dst: int | None = None
    ) -> torch.Tensor:
        """Honest global tensor: gather only the OWNED rows (drop the overlay).

        Reconstructed on **every** rank regardless of ``dst`` (the owned-row
        gather is symmetric, so a single-rank variant would save nothing); the
        ``dst`` argument is accepted for protocol parity. A collective is
        acceptable here — ``full_tensor`` is an explicit gather, not a per-op
        hot path.
        """
        from nvalchemi.distributed._core.gather_primitives import (
            _all_gather_v_rows,
            mesh_group,
        )

        n_owned = stored._meta.n_owned
        owned = stored._local_tensor[:n_owned].contiguous()
        if not dist.is_initialized() or mesh.size() == 1:
            return owned.clone()
        group = mesh_group(mesh)
        world_size = dist.get_world_size(group=group)
        sizes_t = torch.empty(world_size, dtype=torch.int64, device=owned.device)
        dist.all_gather_into_tensor(
            sizes_t,
            torch.tensor([n_owned], dtype=torch.int64, device=owned.device),
            group=group,
        )
        return _all_gather_v_rows(owned, [int(s) for s in sizes_t.tolist()], group)

    def scatter(self, stored: Any, dim: int, index: Any, src: Any) -> Any:
        """Per-atom scatter with halo correction (reverse + forward exchange)."""
        from nvalchemi.distributed._core.shard_tensor import _halo_scatter_correction

        return _halo_scatter_correction(stored, dim, index, src)

    def gather(self, stored: Any, dim: int, index: Any) -> Any:
        """Gather by index after refreshing the borrowed halo rows."""
        from nvalchemi.distributed._core.shard_tensor import (
            _halo_forward_sync_before_index_select,
        )

        return _halo_forward_sync_before_index_select(stored, dim, index)

    def replicate(self, x: Any, ctx: Any) -> Any:
        # Refresh the borrowed ghost rows from their owners; preserve any
        # trailing padding rows.
        from nvalchemi.distributed._core.particle_halo import (  # noqa: PLC0415
            halo_forward_exchange,
        )

        meta, cfg = ctx.halo_meta, ctx.halo_config
        n_owned, n_padded = int(meta.n_owned), int(meta.n_padded)
        refreshed = halo_forward_exchange(x[:n_owned].contiguous(), meta, cfg)
        if x.shape[0] > n_padded:
            import torch  # noqa: PLC0415

            return torch.cat([refreshed, x[n_padded:]], dim=0)
        return refreshed

    def fold(self, out: Any, ctx: Any) -> Any:
        # Accumulate ghost-row partial sums back to owners, then re-broadcast so
        # owned + ghost rows carry the cross-rank totals for the next block.
        from nvalchemi.distributed._core.particle_halo import (  # noqa: PLC0415
            halo_forward_exchange,
            halo_reverse_exchange,
        )

        meta, cfg = ctx.halo_meta, ctx.halo_config
        n_padded = int(meta.n_padded)
        owned = halo_reverse_exchange(out[:n_padded].contiguous(), meta, cfg)
        refreshed = halo_forward_exchange(owned, meta, cfg)
        if out.shape[0] > n_padded:
            import torch  # noqa: PLC0415

            return torch.cat([refreshed, out[n_padded:]], dim=0)
        return refreshed

    def build_topology(self, config: Any, sharded: Any) -> Any:
        from nvalchemi.distributed._core.particle_halo import (  # noqa: PLC0415
            ParticleHaloConfig,
        )
        from nvalchemi.distributed.partitioner import (  # noqa: PLC0415
            SpatialPartitioner,
        )

        partitioner = sharded.partitioner or SpatialPartitioner(
            config=config, cell_matrix=sharded.cell, pbc=sharded.pbc
        )
        halo_config = ParticleHaloConfig(
            ghost_width=config.effective_ghost_width(),
            partitioner=partitioner,
            mesh=config.mesh,
        )
        return partitioner, halo_config

    def run_forward(
        self, dist_model: Any, sharded: Any, wired_fields: Any
    ) -> "dict[str, Any]":
        return dist_model._call_halo_storage(sharded, wired_fields=wired_fields)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "halo",
            "scatter_mode": self.scatter_mode,
            "gather_mode": self.gather_mode,
        }


@dataclass(frozen=True)
class RefreshOnlyHaloPolicy(HaloStoragePolicy):
    """Halo storage for *owned-complete* aggregation: ``replicate`` ghost-refresh
    + an IDENTITY ``fold``.

    When the ghost shell gives each rank every edge into its owned atoms, a
    message-passing layer needs only a ghost-row refresh of its inputs (the
    inherited :meth:`replicate`); the owned outputs are already complete, so there
    is no per-layer fold to do. This is the distinction from the scatter-heavy
    :class:`HaloStoragePolicy`, whose ``fold`` reverse-exchanges ghost partials to
    owners. UMA's per-block eSCN aggregation is owned-complete (refresh-only);
    MACE's edge ``scatter_sum`` is not. Making ``fold`` the identity here lets a
    wrapper express its message-passing layer as the single policy-agnostic
    sandwich ``scatter_to_owners(block(refresh_neighbors(x)))`` and have it be
    correct under both this policy (refresh real, fold no-op) and
    :class:`GraphReplicatePolicy` (refresh no-op, fold all-reduce)."""

    def fold(self, out: Any, ctx: Any) -> Any:
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "refresh_halo",
            "scatter_mode": self.scatter_mode,
            "gather_mode": self.gather_mode,
        }


@dataclass(frozen=True)
class GraphParallelPolicy(PlainShard):
    """Owned-row storage for the graph-parallel strategy.

    Storage is identical to :class:`PlainShard` (each rank holds its owned rows
    as a ``Shard(0)``); the distinct type selects the graph-parallel execution
    path, whose cross-rank communication is a per-layer all-gather to a
    replicated node tensor (with a reduce-scatter adjoint) injected at
    message-passing boundaries, rather than a borrowed-row halo overlay. Unlike
    :class:`PlainShard` this is a selectable ``distribution_spec`` policy.
    """

    def replicate(self, x: Any, ctx: Any) -> Any:
        # All-gather this rank's owned node rows to the full replicated tensor so
        # its (globally-indexed) edges can read their source nodes. ``fold`` is
        # the inherited identity: each rank owns every edge into its nodes, so
        # the block's local scatter already holds the complete owned sums.
        from nvalchemi.distributed._core.gather_primitives import (  # noqa: PLC0415
            gather_to_replicate,
            mesh_group,
        )

        meta = ctx.gather_meta
        counts = [int((meta.owner_rank == r).sum()) for r in range(ctx.world_size)]
        return gather_to_replicate(
            x[: int(meta.n_owned)].contiguous(), counts, mesh_group(ctx.mesh)
        )

    def build_topology(self, config: Any, sharded: Any) -> Any:
        # Balanced index partition; no ghost shell, so no halo config.
        from nvalchemi.distributed.partitioner import (  # noqa: PLC0415
            IndexPartitioner,
        )

        return IndexPartitioner(config=config), None

    def run_forward(
        self, dist_model: Any, sharded: Any, wired_fields: Any
    ) -> "dict[str, Any]":
        return dist_model._call_graph_parallel(sharded, wired_fields=wired_fields)

    def to_dict(self) -> dict[str, Any]:
        return {"kind": "graph_parallel"}


@dataclass(frozen=True)
class GraphReplicatePolicy(PlainShard):
    """Node-replicate graph-parallel: every rank holds the full node set, the
    *edges* are sharded across ranks, and each message-passing layer's partial
    per-receiver message sum is recombined by a cross-rank all-reduce.

    For *opaque* models whose message passing cannot call the intent verbs
    (upstream MACE): the model's forward runs unchanged on the full node tensor
    with its sharded edge slice, and the framework injects the recombine at the
    conv via a declared adapter (``scatter_to_owners`` → :meth:`fold`). The
    complement of :class:`GraphParallelPolicy` (which partitions nodes and is
    driven by the intent verbs a BYO-authored forward calls directly): this one
    replicates node-wise work but needs no forward changes, only adapters."""

    def replicate(self, x: Any, ctx: Any) -> Any:
        # Nodes are already full on every rank — nothing to gather.
        return x

    def fold(self, out: Any, ctx: Any) -> Any:
        # Recombine each rank's partial per-receiver message sum into the full
        # sum (standard tensor-parallel SUM all-reduce, fwd + bwd). Correct
        # because the framework reads the energy off each rank's *owned* node
        # slice, so the gradients summed here are distinct per rank — not the
        # replicated energy gradient that would over-count.
        from types import SimpleNamespace  # noqa: PLC0415

        from nvalchemi.distributed._core.gather_primitives import (  # noqa: PLC0415
            distributed_all_reduce,
        )

        return distributed_all_reduce(out, SimpleNamespace(mesh=ctx.mesh))

    def build_topology(self, config: Any, sharded: Any) -> Any:
        # Atoms are replicated in the forward and edges sharded there; the index
        # partitioner is returned only as the (non-None) initialized marker.
        from nvalchemi.distributed.partitioner import (  # noqa: PLC0415
            IndexPartitioner,
        )

        return IndexPartitioner(config=config), None

    def run_forward(
        self, dist_model: Any, sharded: Any, wired_fields: Any
    ) -> "dict[str, Any]":
        return dist_model._call_graph_replicate(sharded, wired_fields=wired_fields)

    def to_dict(self) -> dict[str, Any]:
        return {"kind": "graph_replicate"}


# ----------------------------------------------------------------------
# Policy (de)serialization. A storage policy is JSON-encoded as a small
# ``{"kind": ...}`` record. ``None`` denotes the local (no cross-rank) policy.
# ----------------------------------------------------------------------


# ``kind`` -> builder, populated by :func:`register_policy_kind`. A BYO policy
# registers its kind instead of editing a framework type-switch.
_POLICY_FROM_DICT: dict[str, Any] = {
    "local": lambda d: None,
    "shard": lambda d: PlainShard(),
    "graph_parallel": lambda d: GraphParallelPolicy(),
    "graph_replicate": lambda d: GraphReplicatePolicy(),
    "halo": lambda d: HaloStoragePolicy(
        scatter_mode=d.get("scatter_mode", "halo_correction"),
        gather_mode=d.get("gather_mode", "halo_read"),
    ),
    "refresh_halo": lambda d: RefreshOnlyHaloPolicy(
        scatter_mode=d.get("scatter_mode", "local"),
        gather_mode=d.get("gather_mode", "halo_read"),
    ),
}


def register_policy_kind(kind: str, from_dict: Any) -> None:
    """Register a ``kind`` -> ``from_dict(record)`` builder so a custom policy
    round-trips through :func:`policy_from_dict` without a framework change."""
    _POLICY_FROM_DICT[kind] = from_dict


def policy_to_dict(policy: Any) -> dict[str, Any]:
    """JSON-friendly record for a storage policy (``None`` -> local)."""
    if policy is None:
        return {"kind": "local"}
    return policy.to_dict()


def policy_from_dict(d: dict[str, Any]) -> Any:
    """Inverse of :func:`policy_to_dict`, via the ``kind`` registry."""
    kind = d["kind"]
    builder = _POLICY_FROM_DICT.get(kind)
    if builder is None:
        raise ValueError(f"policy_from_dict: unknown kind {kind!r}")
    return builder(d)


