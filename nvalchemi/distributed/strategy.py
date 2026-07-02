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

"""Parallelization strategy: the single owner of strategy-dependent behavior.

A :class:`ParallelizationStrategy` owns the whole vertical slice of behavior
that varies along the parallelization axis — how the batch is scattered, how the
cell/PBC is tracked, how atoms migrate, how per-system quantities reduce, and how
the forward is prepared and consolidated. Models, integrators, and drivers stay
strategy-agnostic and express *intent*; the strategy provides *mechanism*.

Two strategies ship here, one per existing layout:

* :class:`HaloStrategy` — spatial domain decomposition (owned + ghost halo). The
  cell is load-bearing (fractional coords + ghost widths), the partition evolves
  as atoms cross domains, and per-system reductions sum owned shards.
* :class:`GraphPartitionStrategy` — node-partition graph parallel. Atoms split by
  index; features all-gathered per layer, gradients reduce-scattered. The cell is
  an ordinary model input; no migration.

Each strategy wraps the per-field :class:`StoragePolicy` that carries its
tensor-level transport (scatter/gather/refresh/fold); the strategy adds the
orchestration verbs that a driver sequences. The protocol methods are derived
from the responsibility table in ``proposal-distributed-strategy-refactor.md``
§2 — every method corresponds to a column that genuinely differs across the
strategies, with no speculative hooks.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import torch
import torch.distributed as dist

from nvalchemi.distributed._core.gather_primitives import mesh_group

if TYPE_CHECKING:
    from nvalchemi.data.batch import Batch
    from nvalchemi.distributed.config import DomainConfig

__all__ = [
    "Reduce",
    "ShardState",
    "MigrationPlan",
    "ParallelizationStrategy",
    "HaloStrategy",
    "GraphPartitionStrategy",
    "strategy_for_policy",
]


class Reduce(Enum):
    """Reduction op for :meth:`ParallelizationStrategy.reduce_system`."""

    SUM = "sum"
    MAX = "max"
    MIN = "min"

    def to_op(self) -> Any:
        return {
            Reduce.SUM: dist.ReduceOp.SUM,
            Reduce.MAX: dist.ReduceOp.MAX,
            Reduce.MIN: dist.ReduceOp.MIN,
        }[self]


@runtime_checkable
class ShardState(Protocol):
    """The per-rank physical layout a strategy produces from a global batch.

    A structural protocol covering the **generic** surface every layout shares
    (the base :class:`~nvalchemi.distributed.sharded_batch.ShardedBatch`
    conforms). Strategy-specific state lives on concretions: the spatial-halo
    :class:`~nvalchemi.distributed.sharded_batch.HaloShardState` adds
    ``partitioner`` / ``padded_batch`` / ``halo_meta`` / ``invalidate_padded_view``
    (read only by :class:`HaloStrategy`), which the graph-parallel layouts never
    carry. The model and integrator never touch a ``ShardState`` directly — they
    see :meth:`ParallelizationStrategy.local_view`, a plain ``Batch``.
    """

    @property
    def n_owned(self) -> int: ...

    @property
    def local_batch(self) -> Batch: ...

    cell: Any
    pbc: Any

    def update_from_batch(self, batch: Batch) -> None: ...
    def full_batch(self, dst: int = 0) -> Batch | None: ...
    def to_global_batch(self) -> Batch: ...


@dataclass
class MigrationPlan:
    """A strategy's deferred migration decision.

    :class:`HaloStrategy` issues an async consensus ``all_reduce`` at end-of-step
    and consumes it at the start of the next step, hiding the latency; the plan
    carries that in-flight handle. Strategies that never migrate return
    :meth:`none`.
    """

    work: Any = None
    flag: torch.Tensor | None = None

    @classmethod
    def none(cls) -> MigrationPlan:
        return cls(work=None, flag=None)

    @property
    def is_pending(self) -> bool:
        return self.work is not None


class ParallelizationStrategy(ABC):
    """Single owner of strategy-dependent behavior for one autograd group.

    Constructed with the per-field :class:`StoragePolicy`, the
    :class:`DomainConfig`, and this rank's index within the mesh. The strategy is
    otherwise stateless: its methods act on a :class:`ShardState` (which holds the
    per-run partitioner + views) passed in per call.
    """

    def __init__(self, policy: Any, config: DomainConfig, rank: int) -> None:
        self._policy = policy
        self._config = config
        self._rank = rank

    # ---- identity -------------------------------------------------------

    @property
    def policy(self) -> Any:
        """The per-field :class:`StoragePolicy` this strategy transports with."""
        return self._policy

    # ---- capabilities (so drivers assert, not branch) -------------------

    @property
    @abstractmethod
    def evolves_partition(self) -> bool:
        """True if atoms migrate across ranks during dynamics (halo only)."""

    @property
    @abstractmethod
    def uses_cell_for_partition(self) -> bool:
        """True if the cell is load-bearing for the partition (halo only)."""

    @property
    def caps_atoms(self) -> bool:
        """Whether the per-rank atom count is dynamic under this layout, so the
        compiled-MD graph padder must cap the atom dim (not just edges).

        Halo's per-rank set (owned + ghost) fluctuates as atoms move near domain
        boundaries → ``True``. A graph-parallel node partition holds a fixed atom
        set — only the edge count drifts as atoms move — so it caps edges only
        (``False``); padding atoms there would also break the node all-gather,
        whose routing is keyed on the unpadded atom count. Edges are always
        capped (every message-passing model's edge count drifts under MD).
        """
        return True

    # ---- data layout ----------------------------------------------------

    def scatter(
        self, global_batch: Batch | None, mesh: Any, config: DomainConfig, src: int = 0
    ) -> ShardState:
        """Scatter the global batch into this rank's :class:`ShardState`."""
        from nvalchemi.distributed.sharded_batch import ShardedBatch

        return ShardedBatch.from_batch(
            batch=global_batch,
            mesh=mesh,
            config=config,
            src=src,
            partition_mode=self._policy.partition_mode,
        )

    def local_view(self, state: ShardState) -> Batch:
        """The plain ``Batch`` the model / integrator operate on."""
        return state.local_batch

    def gather(self, state: ShardState, dst: int | None = 0) -> Batch | None:
        """Reconstruct the global batch on *dst* (``None`` → every rank)."""
        if dst is None:
            return state.to_global_batch()
        return state.full_batch(dst=dst)

    # ---- forward -------------------------------------------------------

    def build_topology(self, config: DomainConfig, state: ShardState) -> Any:
        """Return ``(partitioner, halo_config | None)`` for this strategy."""
        return self._policy.build_topology(config, state)

    @abstractmethod
    def run_forward(
        self, dist_model: Any, state: ShardState, wired_fields: Any = None
    ) -> dict[str, Any]:
        """Run this strategy's distributed forward, returning consolidated
        outputs. Each strategy owns its forward mechanism; ``dist_model`` is the
        shared forward toolkit (wrapper, adapters, consolidation, compile
        machinery) it drives."""

    # ---- dynamics / evolving geometry -----------------------------------

    @abstractmethod
    def on_cell_change(self, state: ShardState, cell: torch.Tensor | None) -> None:
        """React to a moving cell (barostat). Halo re-tracks; GP no-ops."""

    @abstractmethod
    def plan_migration(
        self, state: ShardState, batch: Batch
    ) -> MigrationPlan:
        """Decide (async) whether any atoms crossed a boundary this step."""

    @abstractmethod
    def apply_migration(
        self, state: ShardState, batch: Batch, plan: MigrationPlan
    ) -> Batch:
        """Consume a prior :meth:`plan_migration` and reshard if needed."""

    # ---- reductions (intent → mechanism) --------------------------------

    def _group(self) -> Any:
        """Process group this strategy's collectives run on.

        Confined to the domain sub-mesh: when the mesh declares named dims and
        ``config.mesh_dim`` is one of them, resolve that named sub-mesh's group
        (the correct form for a multi-dim mesh, e.g. DD × data-parallel);
        otherwise fall back to the whole mesh (the 1-D case all current scopes
        build, where the two are equivalent).
        """
        mesh = self._config.mesh
        dim = self._config.mesh_dim
        names = getattr(mesh, "mesh_dim_names", None)
        # Only take the named-sub-mesh path for a real dim-name sequence — a
        # ducktyped test mock exposes a truthy attribute that isn't a real list.
        if mesh is not None and isinstance(names, (list, tuple)) and dim in names:
            return mesh_group(mesh[dim])
        return mesh_group(mesh)

    @property
    def process_group(self) -> Any:
        """The mesh process group for collectives that aren't reductions (e.g. a
        replicated-state broadcast). Same group :meth:`reduce_system` uses, so a
        caller never reaches for the default/global group directly."""
        return self._group()

    @abstractmethod
    def reduce_system(self, per_system: torch.Tensor, op: Reduce) -> torch.Tensor:
        """Reduce a per-system quantity to its mesh-global value in place."""

    @abstractmethod
    def global_atom_count(self, n_owned: int, device: torch.device) -> torch.Tensor:
        """Mesh-global atom count (for DOF), as a scalar tensor."""


# ----------------------------------------------------------------------
# Async-overlap exchanges (proposal §3.0) — the per-strategy AsyncExchange impls
# the framework OverlapAdapter drives. Overlap OFF: the collective runs
# synchronously in ``start`` and ``wait`` returns it; making ``start`` truly
# async is a later phase that does not change this interface.
# ----------------------------------------------------------------------


class _GatherExchange:
    """Graph-parallel exchange: all-gather owned node features to the full
    replicated set (reduce-scatter adjoint, carried by ``gather_to_replicate``)."""

    def __init__(self, rank_sizes: list[int], group: Any) -> None:
        self._rank_sizes = rank_sizes
        self._group = group

    def start(self, owned: torch.Tensor) -> torch.Tensor:
        from nvalchemi.distributed._core.gather_primitives import (  # noqa: PLC0415
            gather_to_replicate,
        )

        return gather_to_replicate(owned, self._rank_sizes, self._group)

    def wait(self, handle: torch.Tensor) -> torch.Tensor:
        return handle


class _HaloExchange:
    """Halo exchange: borrow ghost rows → ``[owned | ghost]`` padded set
    (scatter-correct adjoint, carried by ``halo_forward_exchange``)."""

    def __init__(self, meta: Any, config: Any) -> None:
        self._meta = meta
        self._config = config

    def start(self, owned: torch.Tensor) -> torch.Tensor:
        from nvalchemi.distributed._core.particle_halo import (  # noqa: PLC0415
            halo_forward_exchange,
        )

        return halo_forward_exchange(owned, self._meta, self._config)

    def wait(self, handle: torch.Tensor) -> torch.Tensor:
        return handle


# ----------------------------------------------------------------------
# Halo (spatial domain decomposition)
# ----------------------------------------------------------------------


class HaloStrategy(ParallelizationStrategy):
    """Spatial domain decomposition: owned atoms + a ghost halo per rank.

    Owns the load-bearing cell (fractional coords + ghost widths tracked as the
    box deforms), the evolving partition (atoms migrate as they cross domains),
    and owned-shard reductions.
    """

    @property
    def evolves_partition(self) -> bool:
        return True

    @property
    def uses_cell_for_partition(self) -> bool:
        return True

    def run_forward(
        self, dist_model: Any, state: ShardState, wired_fields: Any = None
    ) -> dict[str, Any]:
        dist_model._dist_ctx.cap_atoms = self.caps_atoms
        dist_model._dist_ctx.strategy = self
        return _halo_run_forward(dist_model, state, wired_fields)

    def on_cell_change(self, state: ShardState, cell: torch.Tensor | None) -> None:
        # Track a barostat-deformed cell so rank membership is judged against the
        # current box, not the partition-time one. Single entry point for the
        # cell — the partitioner is the source of truth.
        part = state.partitioner
        if part is not None and cell is not None:
            part.update_cell(cell)

    def plan_migration(self, state: ShardState, batch: Batch) -> MigrationPlan:
        """Issue the consensus all_reduce that decides whether ANY rank's atoms
        crossed a boundary this step; the result is consumed by
        :meth:`apply_migration` at the start of the next step.

        We deliberately discard the per-atom destination here — it is recomputed
        fresh in :meth:`apply_migration` if migration fires. Recomputation is
        cheap (one cell-list pass) and avoids holding a stale rank assignment
        across hook calls that could mutate positions (barostats, freezers, ...).
        """
        part = state.partitioner
        if part is None or not dist.is_initialized():
            return MigrationPlan.none()
        # Judge membership against the current (barostat-deformed) box.
        self.on_cell_change(state, getattr(batch, "cell", None))
        # Hysteresis-aware: flag migration only when an atom has LEFT this rank's
        # domain expanded by the hysteresis margin (not merely crossed the bare
        # boundary) — stops thrashing of atoms vibrating across the plane.
        h = self._config.effective_migration_hysteresis()
        leaving = ~part.keeps_owner(batch.positions, self._rank, h)
        flag = leaving.any().to(torch.int32).view(1)
        work = dist.all_reduce(
            flag, op=dist.ReduceOp.MAX, group=self._group(), async_op=True
        )
        return MigrationPlan(work=work, flag=flag)

    def apply_migration(
        self, state: ShardState, batch: Batch, plan: MigrationPlan
    ) -> Batch:
        """Wait on a prior :meth:`plan_migration` consensus and reshard atoms if
        any crossed a boundary. Returns the (possibly rebuilt) owned batch."""
        if not plan.is_pending:
            return batch
        plan.work.wait()
        needs = bool(plan.flag.item())
        if not needs:
            return batch

        from nvalchemi.distributed._core.reshard import reshard_by_destination

        part = state.partitioner
        device = batch.positions.device
        # Recompute destinations from the latest positions — AFTER_STEP hooks
        # could have nudged positions between plan and apply. Hysteresis-aware:
        # atoms still within this rank's expanded domain KEEP this rank (else the
        # reshard would move band atoms anyway, defeating hysteresis); only atoms
        # that have left get their natural spatial rank. Assign against the
        # current (barostat-deformed) cell, not the stale partition-time one.
        h = self._config.effective_migration_hysteresis()
        self.on_cell_change(state, getattr(batch, "cell", None))
        keep = part.keeps_owner(batch.positions, self._rank, h)
        natural = part.assign_atoms_to_ranks(batch.positions)
        new_rank = torch.where(
            keep, torch.full_like(natural, self._rank), natural
        ).to(torch.int64)
        mesh = self._config.mesh

        # Reshard EVERY per-atom field independently (preserves dtypes). The
        # atoms group holds exactly the per-atom (node-level) tensors, so each
        # can be resharded by the per-atom destination. Enumerating the group
        # (rather than a fixed list) keeps custom fields like atomic charges from
        # vanishing when an atom crosses ranks.
        fields: dict[str, torch.Tensor] = {"positions": batch.positions}
        atoms_group = getattr(batch, "_atoms_group", None)
        if atoms_group is not None:
            for name in atoms_group.keys():
                if name != "positions":
                    fields[name] = atoms_group[name]
        else:  # fallback: attribute access
            for name in ("atomic_numbers", "atomic_masses", "velocities", "forces"):
                val = getattr(batch, name, None)
                if val is not None:
                    fields[name] = val

        new_fields = {
            name: reshard_by_destination(tensor, new_rank, mesh)
            for name, tensor in fields.items()
        }

        new_batch = _build_batch_from_fields(new_fields, device)
        if getattr(batch, "cell", None) is not None:
            new_batch.cell = batch.cell.clone()
        if getattr(batch, "pbc", None) is not None:
            new_batch.pbc = batch.pbc.clone()
        if getattr(batch, "energy", None) is not None:
            new_batch.energy = batch.energy.clone()
        # Per-graph extras the integrator reads back (e.g. NPT/NPH read
        # ``batch.stress`` in pre_update before the next compute fills it).
        # Migration changes atom ownership, not system count, so carry these
        # per-system tensors across verbatim.
        stress = getattr(batch, "stress", None)
        if stress is not None:
            new_batch["stress"] = stress.clone()

        # Refresh the persistent state to match the new layout and invalidate the
        # padded view — migration changes rank ownership, so the halo routing and
        # any cached NL are stale.
        state.update_from_batch(new_batch)
        state.invalidate_padded_view()

        return new_batch

    def reduce_system(self, per_system: torch.Tensor, op: Reduce) -> torch.Tensor:
        # Each rank holds a per-system partial over its OWNED atoms; sum (or
        # max/min) across the mesh to the global value.
        if dist.is_initialized() and self._config.mesh is not None:
            dist.all_reduce(per_system, op=op.to_op(), group=self._group())
        return per_system

    def global_atom_count(self, n_owned: int, device: torch.device) -> torch.Tensor:
        count = torch.tensor([n_owned], dtype=torch.int64, device=device)
        if dist.is_initialized() and self._config.mesh is not None:
            dist.all_reduce(count, op=dist.ReduceOp.SUM, group=self._group())
        return count

    # ---- async overlap (proposal §3.0) ----------------------------------

    def locality_partition(self, edge_index: torch.Tensor, meta: Any) -> Any:
        """Split owned-receiver edges by sender residency.

        ``edge_index`` is ``[sender ; owned-receiver]`` over the ``[owned | ghost]``
        padded set. A sender is resident when it is owned (``sender < n_owned``);
        its id already indexes the owned features. Ghost senders (``>= n_owned``)
        index the exchanged padded set, unchanged.
        """
        from nvalchemi.distributed._core.overlap import (
            LocalityPartition,  # noqa: PLC0415
        )

        resident = edge_index[0] < meta.n_owned
        return LocalityPartition(
            local_edges=edge_index[:, resident],
            remote_edges=edge_index[:, ~resident],
            n_receivers=meta.n_owned,
        )

    def async_exchange(self, meta: Any, halo_config: Any) -> Any:
        """The halo ghost-borrow as an :class:`AsyncExchange`."""
        return _HaloExchange(meta, halo_config)


# ----------------------------------------------------------------------
# Graph parallel — node partition
# ----------------------------------------------------------------------


class GraphPartitionStrategy(ParallelizationStrategy):
    """Node-partition graph parallel: atoms split by index, features
    all-gathered per layer with a reduce-scatter adjoint. The cell is an
    ordinary model input (partition is geometry-free), so there is no cell
    tracking and no migration; per-system quantities sum owned shards like
    halo (each rank holds a distinct owned node slice)."""

    @property
    def evolves_partition(self) -> bool:
        return False

    @property
    def uses_cell_for_partition(self) -> bool:
        return False

    @property
    def caps_atoms(self) -> bool:
        # Node partition holds a fixed atom set (no migration, no ghosts); only
        # the edge count drifts under MD. Cap edges only — padding atoms would
        # also desync the node all-gather (routing keyed on the unpadded count).
        return False

    def run_forward(
        self, dist_model: Any, state: ShardState, wired_fields: Any = None
    ) -> dict[str, Any]:
        dist_model._dist_ctx.cap_atoms = self.caps_atoms
        dist_model._dist_ctx.strategy = self
        return _graph_partition_run_forward(dist_model, state, wired_fields)

    def on_cell_change(self, state: ShardState, cell: torch.Tensor | None) -> None:
        return None

    def plan_migration(self, state: ShardState, batch: Batch) -> MigrationPlan:
        return MigrationPlan.none()

    def apply_migration(
        self, state: ShardState, batch: Batch, plan: MigrationPlan
    ) -> Batch:
        return batch

    def reduce_system(self, per_system: torch.Tensor, op: Reduce) -> torch.Tensor:
        if dist.is_initialized() and self._config.mesh is not None:
            dist.all_reduce(per_system, op=op.to_op(), group=self._group())
        return per_system

    def global_atom_count(self, n_owned: int, device: torch.device) -> torch.Tensor:
        count = torch.tensor([n_owned], dtype=torch.int64, device=device)
        if dist.is_initialized() and self._config.mesh is not None:
            dist.all_reduce(count, op=dist.ReduceOp.SUM, group=self._group())
        return count

    # ---- async overlap (proposal §3.0) ----------------------------------

    def locality_partition(self, edge_index: torch.Tensor, meta: Any) -> Any:
        """Split owned-receiver edges by sender residency.

        ``edge_index`` is ``[global-sender ; owned-local-receiver]``. A sender is
        resident when this rank owns it (``meta.owner_rank[sender] == rank``).

        * local bucket runs on the **owned** features: sender remapped to owned
          space via ``meta.local_index``, receiver stays owned-local.
        * remote bucket runs on the **gathered full** features: sender keeps its
          global id, and the owned receiver is shifted to its global row
          (``owned_offset`` = this rank's block offset in the concatenated-by-rank
          gather), so the remote scatter lands in the owned block.
        """
        from nvalchemi.distributed._core.overlap import (
            LocalityPartition,  # noqa: PLC0415
        )

        owner = meta.owner_rank.to(edge_index.device)
        local_index = meta.local_index.to(edge_index.device)
        sender, recv_local = edge_index[0], edge_index[1]
        resident = owner[sender] == self._rank
        nlo = int((owner < self._rank).sum())
        local = torch.stack([local_index[sender[resident]], recv_local[resident]])
        remote = torch.stack([sender[~resident], recv_local[~resident] + nlo])
        return LocalityPartition(
            local_edges=local,
            remote_edges=remote,
            n_receivers=meta.n_owned,
            owned_offset=nlo,
        )

    def async_exchange(self, meta: Any, halo_config: Any = None) -> Any:
        """The per-layer node-feature all-gather as an :class:`AsyncExchange`."""
        group = self._group()
        world = dist.get_world_size(group) if dist.is_initialized() else 1
        rank_sizes = torch.bincount(
            meta.owner_rank, minlength=world
        ).tolist()
        return _GatherExchange(rank_sizes, group)


# ----------------------------------------------------------------------
# Batch reconstruction (shared by migration)
# ----------------------------------------------------------------------


def _build_batch_from_fields(
    fields: dict[str, torch.Tensor], device: torch.device
) -> Batch:
    from nvalchemi.data.atomic_data import AtomicData
    from nvalchemi.data.batch import Batch as BatchCls

    known = set(AtomicData.model_fields)
    data = AtomicData(
        positions=fields["positions"],
        atomic_numbers=fields.get(
            "atomic_numbers", torch.zeros(0, dtype=torch.long, device=device)
        ),
    )
    # Reattach every migrated field generically: typed AtomicData fields by
    # attribute, custom per-atom fields via add_node_property.
    for name, tensor in fields.items():
        if name in ("positions", "atomic_numbers"):
            continue
        if name in known:
            setattr(data, name, tensor)
        else:
            data.add_node_property(name, tensor)
    return BatchCls.from_data_list([data], device=device)


# ----------------------------------------------------------------------
# Relocated per-strategy distributed forwards (S2). These own the forward
# *mechanism*; DistributedModel is the shared forward toolkit they drive
# via ``dist_model``. A new strategy adds its forward here, not on the driver.
# ----------------------------------------------------------------------


def _graph_partition_run_forward(
    dist_model,
    sharded: "ShardedBatch",
    wired_fields: "dict[str, Any] | None" = None,
) -> dict[str, Any]:
    from nvalchemi.distributed._core.context import activate_dd_context
    """Graph-parallel forward.

    Each rank owns a balanced index slice of atoms plus the edges into them.
    The node features are all-gathered to a replicated tensor per
    message-passing layer (``refresh_neighbors`` → the policy's replicate) so
    every edge sees its source, and the per-graph node-energy sum drops to
    owners and all-reduces. Forces come from autograd over the owned
    positions: the all-gather's reduce-scatter adjoint routes each owned
    atom's cross-rank gradient back, so they're globally-correct on their
    owning rank with no halo reverse.
    """
    if wired_fields:
        raise NotImplementedError(
            "wired_fields (cross-model field injection) is not supported on "
            "the graph-parallel path."
        )
    _cp = dist_model._spec.compile
    if _cp is None or not _cp.forces_via_autograd:
        # The model computes its own forces internally (e.g. UMA's autograd
        # force head, which consumes + frees the energy graph), so it cannot
        # hand the framework a differentiable energy to grad over the owned
        # leaf. Take the node-partition internal path: full geometry, the
        # model's own forces, cross-rank SUM consolidation.
        return dist_model._graph_parallel_internal(sharded)
    if dist_model._spec.gp_replicate_geometry:
        # Dense-nbmat model whose kernel indexes the position array (PME): run on
        # the replicated full geometry with the neighbour matrix masked to owned
        # receivers; the framework owns the owned-energy autograd.
        return dist_model._graph_parallel_dense_full_autograd(sharded)
    import torch.distributed as dist  # noqa: PLC0415

    from nvalchemi.distributed._core.placement import (  # noqa: PLC0415
        ShardRouting,
    )
    from nvalchemi.distributed.output_consolidation import (  # noqa: PLC0415
        consolidate_sharded_outputs,
    )

    mesh = dist_model._config.mesh
    rank = mesh.get_local_rank() if mesh is not None else 0
    world = dist_model._world_size or 1

    # Global<->owned index map for the balanced partition.
    assignment = sharded.rank_assignment
    meta = ShardRouting.from_assignment(assignment, rank, world)
    meta.n_systems_global = sharded.num_graphs

    # Prepare this rank's owned-receiver neighbours in the model's native
    # format: COO ``neighbor_list`` (senders global, receivers owned-local) for
    # edge-based MPNNs, or a dense ``neighbor_matrix`` (owned receiver rows,
    # global sender columns) for dense-nbmat models (PME real-space, AIMNet2).
    # Both keep senders as global ids into the all-gathered node set the wrapper
    # rebuilds via ``refresh_neighbors``.
    from nvalchemi.models.base import NeighborListFormat  # noqa: PLC0415

    nb_format = dist_model._wrapper.model_config.neighbor_config.format
    if nb_format == NeighborListFormat.MATRIX:
        node_props = dist_model._graph_parallel_owned_nbmat(sharded, meta, rank)
        owned = sharded.local_batch_with_edges(node_properties=node_props)
    else:
        nl = dist_model._graph_parallel_owned_edges(sharded, meta, rank)
        owned = sharded.local_batch_with_edges({"neighbor_list": nl})
    # Positions become a fresh autograd leaf for the energy-force grad.
    atoms = owned._atoms_group
    pos = atoms["positions"]
    pos = (pos.to_local() if hasattr(pos, "to_local") else pos).detach()
    pos.requires_grad_(True)
    atoms["positions"] = pos

    # Publish the per-step routing + policy so the wrapper's intent verbs
    # (refresh_neighbors / system_sum) resolve to the GP collectives.
    dist_model._dist_ctx.policy = dist_model._spec.distribution.policy
    dist_model._dist_ctx.gather_meta = meta
    dist_model._dist_ctx.halo_meta = None

    # Framework owns the force autograd here (grad of the owned energy over the
    # owned-position leaf). The wrapper must therefore run *energy-only*: a model
    # that also computes forces internally (e.g. MACE, when ``forces`` is active)
    # would consume/free the energy graph inside its own forward and the grad
    # below would raise "backward through the graph a second time". Capture the
    # force intent first, then narrow ``active_outputs`` to energy for the forward
    # and restore it after. Energy-only wrappers (the toy) are unaffected.
    _want_forces = dist_model._needs_forces()
    _mc = dist_model._wrapper.model_config
    _saved_active = _mc.active_outputs
    _mc.active_outputs = {"energy"}
    try:
        with activate_dd_context(dist_model._dist_ctx):
            output = dist_model._wrapper(owned)
            # The wrapper returns this rank's owned per-graph energy partial.
            # Forces differentiate that partial: the per-layer node-gather's
            # reduce-scatter adjoint already routes each owned atom's cross-rank
            # gradient back, so the owned forces come out globally-correct.
            energy_partial = output["energy"]
            if _want_forces:
                (grad,) = torch.autograd.grad(
                    [energy_partial.sum()],
                    [pos],
                    create_graph=False,
                    retain_graph=False,
                    allow_unused=True,
                )
                output["forces"] = torch.zeros_like(pos) if grad is None else -grad
            # Global energy for reporting: a plain SUM across ranks of the owned
            # partials (every atom is owned once, so no double count). Detached —
            # the force path is already complete, and an autograd-aware reduce
            # would inflate a re-differentiated energy by the world size.
            energy_global = energy_partial.detach().clone()
            if dist.is_initialized() and world > 1:
                from nvalchemi.distributed._core.gather_primitives import (  # noqa: PLC0415
                    mesh_group,
                )

                dist.all_reduce(
                    energy_global, op=dist.ReduceOp.SUM, group=mesh_group(mesh)
                )
            output["energy"] = energy_global
    finally:
        _mc.active_outputs = _saved_active

    return consolidate_sharded_outputs(
        output,
        model_config=dist_model._wrapper.model_config,
        world_size=dist_model._world_size,
        owned_only_outputs=dist_model._spec.owned_only_outputs,
        all_reduce_outputs=dist_model._spec.all_reduce_outputs,
        halo_config=dist_model._halo_config,
    )


def _halo_run_forward(
    dist_model,
    sharded: "ShardedBatch",
    wired_fields: "dict[str, Any] | None" = None,
) -> dict[str, Any]:
    from nvalchemi.distributed._core.context import activate_dd_context
    from nvalchemi.distributed.distributed_model import (
        _mark_halo_receiver_edges_as_padding,
        _promote_positions_to_shardtensor,
    )
    from nvalchemi.distributed.output_consolidation import consolidate_padded_outputs
    from nvalchemi.neighbors import compute_neighbors
    """Halo-storage forward.

    Preconditions (typically set up by :class:`DomainParallel` via
    ``HaloExchangeHook`` + ``NeighborListHook`` before each call, or
    manually in benchmark / test harnesses):

    - ``sharded.padded_batch`` is populated
      (see :func:`nvalchemi.distributed.particle_halo.halo_exchange`).
    - The padded batch has a neighbor list
      (e.g. ``compute_neighbors(sharded.padded_batch, cfg)``).

    If either is missing, the adapter falls back to doing both here
    — convenient for one-shot calls but avoids the per-step NL cost
    that makes skin-amortized NL worthwhile.
    """
    from nvalchemi.distributed.particle_halo import halo_exchange

    compute_forces = dist_model._needs_forces()

    # Fallback: populate the padded view if the caller didn't.
    if sharded.padded_batch is None:
        halo_exchange(sharded, dist_model._halo_config, compute_forces=compute_forces)

    padded_batch = sharded.padded_batch
    meta = sharded.halo_meta

    # Flag a degenerate halo partition once, up front.
    dist_model._check_partition_health(meta, padded_batch.positions.device)

    # Fallback: compute NL on the padded block if it isn't already there.
    if (
        getattr(padded_batch, "neighbor_matrix", None) is None
        and getattr(padded_batch, "neighbor_list", None) is None
    ):
        compute_neighbors(
            padded_batch, config=dist_model._wrapper.model_config.neighbor_config
        )
    # Mark halo-receiver edges so the wrapper's ``(edge_index < n_atoms)``
    # filter drops them; see the helper's docstring for the rationale.
    _mark_halo_receiver_edges_as_padding(padded_batch, meta.n_owned)

    # Update per-step ctx state so the wrapper's ``adapt_input`` reads it.
    dist_model._dist_ctx.policy = dist_model._spec.distribution.policy
    dist_model._dist_ctx.halo_meta = meta
    dist_model._dist_ctx.halo_config = dist_model._halo_config
    # Expose the persistent cap dict so a wrapper that pads inside its own
    # forward grows the same caps via current_dd_context().cap_state.
    dist_model._dist_ctx.cap_state = dist_model._cap_state

    # Fixed-shape padding (compile-only): pad to per-rank caps so the
    # compiled energy graph sees static atom/edge counts. Active only when
    # the model uses the energy-autograd force strategy and compile was
    # requested; eager instances skip padding entirely.
    _cp = dist_model._spec.compile
    _dd_compile = bool(
        _cp is not None
        and _cp.forces_via_autograd
        and dist_model._dd_compile_requested
    )
    if wired_fields and _dd_compile:
        raise NotImplementedError(
            "wired_fields (cross-model field injection) is only supported "
            "on the eager distributed path, not compiled."
        )
    _pad_active = _dd_compile
    _orig_atoms = _orig_edges = None
    if _pad_active:
        from nvalchemi.distributed.graph_padder import resolve_cap  # noqa: PLC0415

        # max_send required this step. ``meta.send_sizes`` is identical on
        # every rank, so ranks grow this cap in lockstep and the halo
        # all_to_all sizes stay matched. It's a send-buffer cap, not a
        # graph-shape cap (the graph padder owns those), so it lives here.
        _ms_req = max((max(r) for r in meta.send_sizes), default=0)
        resolve_cap(
            dist_model._cap_state, "max_send", _ms_req,
            initial_factor=1.20, grow_factor=1.30, stride=16,
        )
        # The padded view is transient — only the compiled forward needs
        # fixed shapes. Stash the real-sized storage groups to restore after
        # the forward, since ``halo_exchange`` reuses ``padded_batch`` in
        # place and a cap-sized buffer would mismatch next step.
        _groups = padded_batch._storage.groups
        _orig_atoms = _groups.get("atoms")
        _orig_edges = _groups.get("edges")
        # Pad to the atom/edge caps from ``dist_model._cap_state`` (grow-only).
        dist_model._graph_padder.pad(padded_batch, dist_model._cap_state)

    # Make the live per-step context ambient for the wrapper's forward, so
    # context-aware helpers and adapter bodies read it through
    # ``current_dd_context()``.
    if _dd_compile:
        # Compiled energy-autograd forward, framework-owned: the wrapper runs
        # energy-only on plain tensors with the halo routing threaded as
        # graph inputs; the framework consolidates per-node energy and takes
        # the force autograd.
        with activate_dd_context(dist_model._dist_ctx):
            output = dist_model._compiled_energy_autograd_forward(
                padded_batch, meta, sharded.num_graphs
            )
    else:
        # Cross-model wired fields: overwrite named per-atom inputs with an
        # upstream model's owned values, gathered into this model's ghost
        # layout via the autograd-aware halo exchange. Runs before promotion
        # so the gathered (grad-carrying) tensor is what gets wrapped; its
        # backward scatter-adds ghost grads to the producing rank's owner.
        if wired_fields:
            from nvalchemi.distributed._core.particle_halo import (  # noqa: PLC0415
                halo_forward_exchange,
            )

            _atoms = padded_batch._atoms_group
            for _name, _owned in wired_fields.items():
                _atoms[_name] = halo_forward_exchange(
                    _owned, meta, dist_model._halo_config
                )
        # Eager: promote ``positions`` (and other primary per-atom inputs)
        # to ShardTensors so custom ops see a ShardTensor input and the
        # per-layer halo correction fires.
        _promote_positions_to_shardtensor(
            padded_batch, dist_model._spec, meta, dist_model._halo_config,
            sharded.num_graphs, None,
        )
        # A model that builds + compiles its own graph declares a
        # ``graph_padder`` without ``forces_via_autograd``: the framework
        # can't pad the Batch (the graph only exists once ``adapt_input``
        # runs), so it publishes the padder on the context for the wrapper to
        # apply, then unpads after the forward.
        _eager_padder = (
            _cp.graph_padder
            if (
                _cp is not None
                and _cp.graph_padder is not None
                and not _cp.forces_via_autograd
            )
            else None
        )
        dist_model._dist_ctx.graph_padder = _eager_padder
        # A wrapper that delegates its per-system energy reduction to the
        # framework (``spec.node_energy_key``) emits raw per-node energies
        # under that key; widen active outputs so the forward produces them,
        # then reduce owned-aware below. Restored in ``finally``.
        _nek = dist_model._spec.node_energy_key
        _mc = dist_model._wrapper.model_config
        _saved_active = None
        if _nek is not None and _nek not in _mc.active_outputs:
            _saved_active = _mc.active_outputs
            _mc.active_outputs = set(_saved_active) | {_nek}
        with activate_dd_context(dist_model._dist_ctx):
            try:
                output = dist_model._wrapper(padded_batch)
                if _eager_padder is not None:
                    output = _eager_padder.unpad(output)
                if _nek is not None and _nek in output:
                    output = dist_model._reduce_node_energy(
                        output, _nek, padded_batch, sharded.num_graphs
                    )
            finally:
                if _eager_padder is not None:
                    _eager_padder.restore()
                if _saved_active is not None:
                    _mc.active_outputs = _saved_active
    # Under compile, forces/stress come from autograd over the global
    # energy, so they need the halo-reverse consolidation rather than the
    # eager owned-only slice — drop them from owned_only. Eager keeps the
    # declared slice.
    owned_only = dist_model._spec.owned_only_outputs
    if _dd_compile:
        owned_only = owned_only - dist_model._wrapper.model_config.autograd_outputs
    result = consolidate_padded_outputs(
        output,
        model_config=dist_model._wrapper.model_config,
        meta=meta,
        halo_config=dist_model._halo_config,
        world_size=dist_model._world_size,
        owned_only_outputs=owned_only,
        all_reduce_outputs=dist_model._spec.all_reduce_outputs,
        output_kinds=dist_model._spec.output_kinds,
    )
    if _pad_active:
        _groups = padded_batch._storage.groups
        if _orig_atoms is not None:
            _groups["atoms"] = _orig_atoms
        if _orig_edges is not None:
            _groups["edges"] = _orig_edges
    return result



# ----------------------------------------------------------------------
# Factory: policy -> strategy
# ----------------------------------------------------------------------


# policy class -> strategy class. Built-ins register lazily (import cycle); a
# user-defined policy calls ``register_strategy`` to bind its own strategy without
# editing the factory — the same open-registry discipline as OpAdapter kinds.
_STRATEGY_REGISTRY: "dict[type, type]" = {}


def register_strategy(policy_cls: type, strategy_cls: type) -> None:
    """Bind a storage-policy class to its :class:`ParallelizationStrategy`.

    Lets a user-defined policy register its strategy so
    :func:`strategy_for_policy` resolves it without editing that function.
    """
    _STRATEGY_REGISTRY[policy_cls] = strategy_cls


def _ensure_builtin_strategies() -> None:
    """Register the two shipped policy→strategy bindings on first use (deferred to
    dodge the ``storage_policy`` import cycle at module load)."""
    if _STRATEGY_REGISTRY:
        return
    from nvalchemi.distributed._core.storage_policy import (
        GraphParallelPolicy,
        HaloStoragePolicy,
    )

    register_strategy(GraphParallelPolicy, GraphPartitionStrategy)
    register_strategy(HaloStoragePolicy, HaloStrategy)


def strategy_for_policy(
    policy: Any, config: DomainConfig, rank: int
) -> ParallelizationStrategy:
    """Build the :class:`ParallelizationStrategy` for a storage *policy*.

    Resolution is registry-driven (:func:`register_strategy`); a new strategy
    registers its policy binding rather than editing a driver type-switch.
    """
    if policy is None:
        raise ValueError(
            "strategy_for_policy: no storage policy (local / single-process path "
            "has no parallelization strategy)."
        )
    _ensure_builtin_strategies()
    # Walk the MRO so a subclass resolves to its nearest registered base
    # (GraphParallelPolicy subclasses PlainShard; RefreshOnlyHaloPolicy subclasses
    # HaloStoragePolicy) — most-derived match wins, so registration order is
    # irrelevant.
    for cls in type(policy).__mro__:
        strategy_cls = _STRATEGY_REGISTRY.get(cls)
        if strategy_cls is not None:
            return strategy_cls(policy, config, rank)
    raise ValueError(
        f"strategy_for_policy: no strategy registered for policy {type(policy).__name__}"
    )
