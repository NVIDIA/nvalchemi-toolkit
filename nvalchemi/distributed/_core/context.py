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

"""Per-scope distributed runtime metadata.

:class:`DistributedContext` is the single object carrying the
runtime-only metadata that a model wrapper needs to read on every
forward pass under a :class:`~nvalchemi.distributed.DistributedModel`
scope.

Lifecycle::

    DistributedModel.__enter__
        ctx = DistributedContext(...)               # built once
        wrapper.distributed_setup(ctx)              # wrapper stashes ref
    DistributedModel.__call__(sharded_batch)
        ctx.halo_meta   = sharded.halo_meta         # per-step write
        ctx.gather_meta = ...                       # per-step write
        wrapper(padded)                              # wrapper reads ctx
    DistributedModel.__exit__
        wrapper.distributed_teardown()              # wrapper drops ref

The ctx is *mutable* by design: per-step values like
:attr:`halo_meta` / :attr:`gather_meta` are updated by the framework
on every forward pass, with the wrapper holding a single reference
that always observes the current state.

Part of the upstream-candidate ``_core/`` surface; must not import
from ``nvalchemi.models`` / ``nvalchemi.data`` / ``nvalchemi.dynamics`` /
``nvalchemi.distributed._chemistry``.
"""

from __future__ import annotations

import contextvars
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from collections.abc import Iterator

__all__ = [
    "NOT_DISTRIBUTED",
    "DistributedContext",
    "activate_dd_context",
    "current_dd_context",
]


@dataclass
class DistributedContext:
    """Runtime metadata for one ``DistributedModel`` scope.

    Mutable by design — see module docstring for the lifecycle.

    Attributes
    ----------
    mesh
        The :class:`torch.distributed.device_mesh.DeviceMesh` that the
        scope's collective ops dispatch over. ``None`` for single-rank
        runs.
    halo_config
        :class:`~nvalchemi.distributed._core.particle_halo.ParticleHaloConfig`
        carrying the partitioner + ghost-width + process group needed
        for halo exchanges. Set once in
        :meth:`DistributedModel._ensure_initialized`; stays constant
        across calls.
    n_systems_global
        Number of graphs in the global batch (i.e. before sharding).
        Used by wrappers that need to size per-system tensors against
        the un-partitioned count rather than the per-rank slice.
    n_atoms_total
        Number of atoms in the global batch. Used by Ewald / PME for
        cache sizing of reciprocal-space resources keyed on the global
        atom count.
    halo_meta
        Per-step :class:`~nvalchemi.distributed._core.particle_halo.ParticleHaloMetadata`
        produced by the latest halo exchange. ``None`` outside of the
        halo-storage call path or before the first forward pass.
    gather_meta
        Per-step :class:`~nvalchemi.distributed._core.gather_primitives.ShardRouting`
        produced by the latest sharded-storage call. ``None`` outside of
        the sharded-storage path or before the first forward pass.
    """

    mesh: Any = None
    # The active field StoragePolicy; the per-layer intent verbs
    # (refresh_neighbors / scatter_to_owners) delegate their cross-rank behavior
    # to it so a new strategy plugs in without framework branches.
    policy: Any = None
    halo_config: Any = None
    n_systems_global: int | None = None
    n_atoms_total: int | None = None
    halo_meta: Any = None
    gather_meta: Any = None
    # First row of this rank's owned slice within the per-rank node tensor. 0
    # when owned rows come first (halo padded view; node-partition shard), so
    # owned-only reductions slice ``[:n_owned]``. Under the node-replicate
    # strategy every rank holds the full node set, so its owned rows are an
    # interior slice ``[owned_offset : owned_offset + n_owned]`` instead.
    owned_offset: int = 0
    # Fixed-shape-padding cap state (grow-on-overflow / stride buckets), owned by
    # the framework. ``DistributedModel`` points this at its persistent per-model
    # cap dict each forward, so a wrapper that pads inside its own forward
    # (AIMNet2's dense nbmat, UMA's fairchem graph) grows the SAME caps the
    # framework persists across MD steps — via ``current_dd_context().cap_state``
    # + the shared ``resolve_cap`` — instead of a private holder.
    cap_state: dict[str, int] = field(default_factory=dict)
    # The fixed-shape-caps GraphPadder for a model that builds + internally
    # compiles its own graph (UMA): set by ``DistributedModel`` for this forward
    # so the wrapper can pad its adapted graph in one call via
    # :meth:`maybe_pad_graph`. ``None`` when no caps apply (single-process, or a
    # model the framework pads at the Batch level instead).
    graph_padder: Any = None
    # Free-form scratch space for wrapper-private state that should
    # share the ctx's lifetime. Kept untyped on purpose — the spec
    # layer is generic and shouldn't know about per-wrapper conventions.
    extras: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Derived read-only view (the ``current_dd_context()`` vocabulary).
    #
    # These properties expose the runtime facts an adapter body / refresh
    # hook needs, derived from the per-step metadata the framework writes
    # above. Compile-safety: ``policy`` / ``world_size`` / ``rank`` /
    # ``compiling`` are constant for a forward and safe anywhere;
    # ``n_owned`` / ``n_padded`` *vary* per step and must only be read in
    # eager or ``@torch._dynamo.disable``d code (the value would otherwise
    # bake into a compiled graph as a stale constant).
    # ------------------------------------------------------------------

    @property
    def is_halo(self) -> bool:
        """True when this forward runs on the halo-storage path."""
        return self.halo_meta is not None

    @property
    def is_sharded(self) -> bool:
        """True when this forward runs on the sharded-storage path."""
        return self.gather_meta is not None

    @property
    def is_distributed(self) -> bool:
        """True inside a real domain-decomposed forward.

        False for the :data:`NOT_DISTRIBUTED` sentinel and for any
        single-process call, so distribution-agnostic helper bodies can
        early-return to plain local behavior.
        """
        if self.is_halo or self.is_sharded:
            return True
        return self.mesh is not None and self.world_size > 1

    @property
    def rank(self) -> int:
        """This process's rank within the scope's mesh (0 if unknown)."""
        cfg = self.halo_config
        if cfg is not None and getattr(cfg, "rank", None) is not None:
            return int(cfg.rank)
        if self.mesh is not None:
            try:
                return int(self.mesh.get_local_rank())
            except Exception:  # pragma: no cover — defensive
                return 0
        return 0

    @property
    def world_size(self) -> int:
        """Number of ranks in the scope (1 if single-process)."""
        if self.halo_meta is not None:
            return len(self.halo_meta.send_sizes)
        if self.mesh is not None:
            try:
                return int(self.mesh.size())
            except Exception:  # pragma: no cover — defensive
                return 1
        return 1

    @property
    def n_owned(self) -> int | None:
        """Rows this rank owns this step (``None`` if not yet known).

        Varies per step — read only in eager / dynamo-disabled code.
        """
        if self.halo_meta is not None:
            return int(self.halo_meta.n_owned)
        if self.gather_meta is not None:
            return int(self.gather_meta.n_owned)
        return None

    @property
    def n_padded(self) -> int | None:
        """Count of real rows in this rank's node tensor (``None`` if unknown).

        On the halo path this is owned + halo (``n_padded``). Under the
        node-replicate strategy every rank holds the full node set, so it is the
        global node count (``gather_meta.n_global``) — every real row is present.
        Varies per step — read only in eager / dynamo-disabled code.
        """
        if self.halo_meta is not None:
            return int(self.halo_meta.n_padded)
        if self.gather_meta is not None:
            return int(self.gather_meta.n_global)
        return None

    def maybe_pad_graph(self, data: Any) -> Any:
        """Pad a model's adapted graph to fixed per-rank shapes, if caps apply.

        The one-call seam a model that builds + internally compiles its own graph
        (e.g. UMA's fairchem graph) uses inside its forward: when the framework has
        set :attr:`graph_padder` for this forward it pads ``data`` to the
        persistent :attr:`cap_state` capacities; otherwise (single-process, or a
        model padded at the Batch level) it returns ``data`` unchanged. The
        framework owns the matching ``unpad`` / ``restore`` after the forward, so
        the wrapper carries no other caps logic.

        Parameters
        ----------
        data
            The model's adapted graph/input to pad in place.

        Returns
        -------
        Any
            ``data`` padded to the fixed caps, or unchanged when no padder is set.
        """
        if self.graph_padder is None:
            return data
        return self.graph_padder.pad(data, self.cap_state)

    @property
    def compiling(self) -> bool:
        """True while tracing under ``torch.compile``.

        A helper that reads varying state (``n_owned`` …) must consult
        this and route varying values through threaded graph inputs
        rather than baking the Python value.
        """
        return bool(torch.compiler.is_compiling())


# ----------------------------------------------------------------------
# Ambient accessor — the public ``current_dd_context()`` surface.
#
# The framework activates the live :class:`DistributedContext` for the
# duration of the wrapper's forward (see ``DistributedModel`` /
# ``DomainParallel``); adapter bodies and refresh hooks read it through
# :func:`current_dd_context`, the way ``torch.no_grad()`` is read.
# ----------------------------------------------------------------------

#: Returned by :func:`current_dd_context` outside any DD forward. Inert:
#: ``is_distributed`` is False, so single-process code that happens to
#: call a context-aware helper falls through to plain local behavior.
NOT_DISTRIBUTED = DistributedContext()

_ACTIVE_DD_CONTEXT: contextvars.ContextVar[DistributedContext | None] = (
    contextvars.ContextVar("nvalchemi_active_dd_context", default=None)
)


def current_dd_context() -> DistributedContext:
    """Return the live DD context for the current forward.

    Inside a :class:`~nvalchemi.distributed.DomainParallel` /
    ``DistributedModel`` forward this is the framework's per-step context
    (policy, halo metadata, counts). Outside one — single-process code,
    or before the first forward — it is the inert :data:`NOT_DISTRIBUTED`
    sentinel.

    Read it in eager or ``@torch._dynamo.disable``d code only. Inside a
    compiled region the varying fields would bake as stale constants;
    code there receives what it needs as threaded graph inputs instead.
    """
    return _ACTIVE_DD_CONTEXT.get() or NOT_DISTRIBUTED


@contextmanager
def activate_dd_context(ctx: DistributedContext) -> Iterator[DistributedContext]:
    """Make ``ctx`` the active context for the duration of the block.

    The framework wraps each wrapper forward in this scope so
    :func:`current_dd_context` resolves to the live, per-step context.
    Restores the previous context on exit (re-entrant via
    :class:`contextvars.ContextVar`).
    """
    token = _ACTIVE_DD_CONTEXT.set(ctx)
    try:
        yield ctx
    finally:
        _ACTIVE_DD_CONTEXT.reset(token)
