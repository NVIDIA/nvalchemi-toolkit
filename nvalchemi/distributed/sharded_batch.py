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

"""ShardedBatch: user-facing distributed counterpart to ``Batch``.

Per-atom fields (positions, velocities, forces, atomic_numbers, atomic_masses)
are stored as ``ShardTensor`` with ``Shard(0)`` placement (uneven across ranks);
per-system fields (cell, pbc) are replicated.

``ShardedBatch`` is what the user hands to ``DistributedModel``; the adapter
pulls ``.local_batch`` out per call to drive the halo-padded forward.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import torch
import torch.distributed as dist

from nvalchemi.distributed._core.collection import ShardedCollection
from nvalchemi.distributed._core.gather_primitives import mesh_group
from nvalchemi.distributed._core.storage_policy import (
    PlainShard,
    StoragePolicy,
)
from nvalchemi.distributed.config import DomainConfig
from nvalchemi.distributed.partitioner import SpatialPartitioner

if TYPE_CHECKING:
    from torch.distributed import DeviceMesh

    from nvalchemi.data.batch import Batch
    from nvalchemi.distributed._core.particle_halo import ParticleHaloMetadata

logger = logging.getLogger(__name__)

# Per-atom fields that must exist on the source batch. The rest of the
# atoms-group is discovered dynamically (see ``_discover_atom_schema``).
_REQUIRED_ATOM_FIELDS: tuple[str, ...] = (
    "positions",
    "atomic_numbers",
    "atomic_masses",
)

# Atoms-group fields that are neighbor-list artifacts — rebuilt per-rank
# on the halo-padded block by ``compute_neighbors``, never scattered.
_NL_ATOM_FIELDS: frozenset[str] = frozenset(
    {"neighbor_matrix", "neighbor_matrix_shifts", "num_neighbors"}
)

# Fields whose dtype we pin regardless of what the source batch holds
# (keeps downstream consumers — integer sentinels, index kernels —
# honest across checkpoint formats).
_ATOM_FIELD_DTYPE_OVERRIDES: dict[str, torch.dtype] = {
    "atomic_numbers": torch.int64,
}

# Float dtypes we can broadcast across the mesh. Positions (and every
# field that inherits their precision) must pick one of these. Single source
# of truth for the int code <-> dtype mapping used by the broadcast.
_FLOAT_DTYPE_CODES: tuple[torch.dtype, ...] = (
    torch.float32,
    torch.float64,
    torch.float16,
    torch.bfloat16,
)
_FLOAT_DTYPE_TO_CODE: dict[torch.dtype, int] = {
    dt: i for i, dt in enumerate(_FLOAT_DTYPE_CODES)
}


def _discover_atom_schema(batch: Batch) -> list[dict[str, Any]]:
    """Enumerate scatter-eligible per-atom fields on *batch*.

    Reads the atoms group directly so every per-atom field the producer
    attached (charges, momenta, node_attrs, custom user fields via
    ``add_node_property``) is carried through — no hand-maintained
    allowlist to drift out of sync with model wrapper requirements.
    Neighbor-list artifacts are excluded because they're rebuilt
    per-rank on the halo-padded block.
    """
    atoms = batch._atoms_group
    if atoms is None:
        raise ValueError(
            "ShardedBatch.from_batch requires a Batch with an atoms group."
        )
    schema: list[dict[str, Any]] = []
    for name, tensor in atoms.items():
        if name in _NL_ATOM_FIELDS:
            continue
        schema.append(
            {
                "name": name,
                "dtype": _ATOM_FIELD_DTYPE_OVERRIDES.get(name, tensor.dtype),
                "trailing_shape": tuple(tensor.shape[1:]),
            }
        )
    return schema


def _has_field(batch: Batch, name: str) -> bool:
    """Check if a batch has a non-None field."""
    return hasattr(batch, name) and getattr(batch, name) is not None


def _broadcast_float_dtype(
    src_dtype: torch.dtype | None, device: torch.device, src: int
) -> torch.dtype:
    """Broadcast a float dtype from ``src`` to every rank as an int code
    and decode back to a ``torch.dtype``. Non-src ranks pass ``None``.
    """
    if src_dtype is not None and src_dtype not in _FLOAT_DTYPE_TO_CODE:
        raise ValueError(
            f"ShardedBatch positions dtype {src_dtype} not supported; "
            f"must be one of {_FLOAT_DTYPE_CODES}."
        )
    code = _FLOAT_DTYPE_TO_CODE.get(src_dtype, 0) if src_dtype else 0
    code_t = torch.tensor([code], dtype=torch.int32, device=device)
    if dist.is_initialized():
        dist.broadcast(code_t, src=src)
    return _FLOAT_DTYPE_CODES[int(code_t.item())]


class ShardedBatch(ShardedCollection):
    """A ``Batch`` distributed across a 1-D ``DeviceMesh``.

    The chemistry-specific subclass of
    :class:`~nvalchemi.distributed._core.collection.ShardedCollection`: it
    supplies the atomic-data field->policy map (per-atom fields ->
    :class:`PlainShard`; ``cell`` / ``pbc`` are replicated side metadata) and
    the ``Batch``-packing logic. The generic scatter / local / gather machinery
    lives on the base.

    Per-atom fields are ``ShardTensor(Shard(0))`` of global shape
    ``(n_global, ...)`` with each rank physically holding ``n_owned``
    rows. Per-system fields (``cell``, ``pbc``) are replicated.

    Obtained via :meth:`from_batch` (scatter from the source rank) and
    consumed by :class:`~nvalchemi.distributed.distributed_model.DistributedModel`
    via :attr:`local_batch`. :meth:`full_batch` / :meth:`to_global_batch`
    gather back when the user wants a whole-system view.
    """

    def __init__(
        self,
        mesh: DeviceMesh,
        atom_fields: dict[str, Any],
        cell: torch.Tensor,
        pbc: torch.Tensor,
        n_global: int,
        partitioner: SpatialPartitioner | None = None,
        partition_mode: str = "spatial",
    ) -> None:
        super().__init__(
            mesh,
            atom_fields,
            self._policies_for(list(atom_fields.keys())),
        )
        self.cell = cell
        self.pbc = pbc
        self._n_global = n_global
        # Spatial decomposition built from the broadcast geometry during
        # :meth:`from_batch`. Cached here so downstream consumers
        # (``DomainParallel`` for atom migration, ``DistributedModel``
        # for halo config) don't re-build it from the same inputs.
        self._partitioner = partitioner
        # Storage flavour. ``"spatial"`` / ``"contiguous_block"`` both use
        # ShardTensor with ``Shard(0)`` placement (per-rank ``n_owned`` rows);
        # they differ only in how the rank assignment is computed.
        self._partition_mode = partition_mode
        # Per-rank local padded view (owned + halo rows). Populated by
        # :func:`nvalchemi.distributed.particle_halo.halo_exchange`. Holds
        # plain tensors (not ShardTensors) packed into a standard ``Batch``
        # — models consume it via ``DistributedModel`` exactly like a
        # single-system Batch, and ``compute_neighbors`` / ``NeighborListHook``
        # operate on it unchanged. ``None`` until ``halo_exchange`` runs.
        self.padded_batch: Batch | None = None
        self.halo_meta: ParticleHaloMetadata | None = None

    @staticmethod
    def _policies_for(field_names: list[str]) -> dict[str, StoragePolicy]:
        """Map per-atom fields to a storage policy.

        Both partition modes (``"spatial"`` / ``"contiguous_block"``) store
        per-atom fields as :class:`PlainShard` (each rank holds its ``n_owned``
        rows as a ``Shard(0)`` ShardTensor); they differ only in *how* the rank
        assignment is computed upstream, not in the storage policy.
        """
        return {name: PlainShard() for name in field_names}

    # ------------------------------------------------------------------
    # Padded view management (halo-storage path)
    # ------------------------------------------------------------------

    def invalidate_padded_view(self) -> None:
        """Drop the cached padded view and halo metadata. Called after
        atom migration or any operation that changes which atoms are
        owned by which rank. Next ``halo_exchange`` will repopulate."""
        self.padded_batch = None
        self.halo_meta = None

    def pad_padded_view_to_caps(self, n_pad_max: int, e_max: int) -> None:
        """Pad the halo-padded view to fixed shapes for ``torch.compile``.

        Pads per-atom fields to ``n_pad_max`` atoms and per-edge fields to
        ``e_max`` edges on the generic Batch storage, so the compiled DD graph
        sees static atom/edge counts across steps (otherwise per-step migration
        / NL-rebuild vary the counts and trigger per-rank recompiles).

        Parameters
        ----------
        n_pad_max : int
            Fixed per-atom row count to pad to.
        e_max : int
            Fixed per-edge row count to pad to.

        Notes
        -----
        Pad rows (``[n_padded, n_pad_max)``) carry zero values and join the
        last graph; their node outputs are dropped by the owned-only
        consolidation. Invalid edges and edge fill are routed to an isolated
        dead node (row ``n_pad_max - 1``) as self-loops, with
        ``neighbor_list_shifts`` set to the ``[1, 0, 0]`` image so the edge
        vector is non-degenerate (a zero vector NaNs through MLIP spherical
        harmonics). Per-rank caps are safe: the cross-rank halo collectives key
        on the consensus ``max_send``, not on ``n_pad_max``. Raises on cap
        overflow (grow-on-overflow is layered on by the caller). A thin
        delegator to :class:`~nvalchemi.distributed.graph_padder.COOPadder` for
        callers that hold a ``ShardedBatch`` and have resolved explicit caps.
        """
        from nvalchemi.distributed.graph_padder import (  # noqa: PLC0415
            _pad_coo_to_caps,
        )

        if self.padded_batch is None:
            return
        _pad_coo_to_caps(self.padded_batch, n_pad_max, e_max)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def positions(self) -> Any:
        return self.fields["positions"]

    @property
    def velocities(self) -> Any | None:
        return self.fields.get("velocities")

    @property
    def forces(self) -> Any | None:
        return self.fields.get("forces")

    @property
    def charges(self) -> Any | None:
        return self.fields.get("charges")

    @property
    def atomic_numbers(self) -> Any:
        return self.fields["atomic_numbers"]

    @property
    def atomic_masses(self) -> Any:
        return self.fields["atomic_masses"]

    @property
    def n_owned(self) -> int:
        """Number of atoms owned by this rank — the local shard size,
        ``len(positions.to_local())`` (both ``spatial`` and
        ``contiguous_block`` modes store ``Shard(0)`` per-rank rows)."""
        return self.fields["positions"].to_local().shape[0]

    @property
    def n_global(self) -> int:
        """Total number of atoms across the mesh."""
        return self._n_global

    @property
    def partition_mode(self) -> str:
        """``"spatial"`` / ``"contiguous_block"``.

        Set at :meth:`from_batch` time. Both shard per-atom fields ``Shard(0)``
        (each rank holds ``n_owned`` rows); they differ only in how the rank
        assignment is computed (spatial decomposition vs contiguous blocks).
        """
        return self._partition_mode

    @property
    def num_graphs(self) -> int:
        """Number of graphs (systems) — replicated across ranks. Currently
        inferred as 1 for the single-system domain-decomposition case."""
        return 1

    @property
    def partitioner(self) -> SpatialPartitioner | None:
        """Spatial decomposition built from the broadcast geometry during
        :meth:`from_batch`. Shared by downstream consumers that would
        otherwise re-build it from the same ``config`` / ``cell`` / ``pbc``.

        Returns ``None`` when the ShardedBatch was constructed outside
        :meth:`from_batch` (e.g. gloo-harness test helpers). Consumers should
        fall back to building a partitioner from ``config`` + ``self.cell`` /
        ``pbc`` in that case.
        """
        return self._partitioner

    @property
    def rank_assignment(self) -> torch.Tensor:
        """``(n_global,)`` int64 tensor: ``rank_assignment[g]`` is the rank
        that owns global sharded-atom ``g``.

        Atoms are in rank-contiguous order after :meth:`from_batch`'s
        scatter-sort, so this is a block tensor with each rank's block
        sized by that rank's ``n_owned``. Built by all-gathering per-rank
        sizes in a single shot.
        """
        device = self.fields["positions"].to_local().device
        world_size = self.mesh.size() if dist.is_initialized() else 1
        if world_size == 1:
            return torch.zeros(self._n_global, dtype=torch.int64, device=device)

        # Single all_gather into a flat (world_size,) tensor, one sync.
        n_owned_t = torch.tensor([self.n_owned], dtype=torch.int64, device=device)
        sizes_t = torch.empty(world_size, dtype=torch.int64, device=device)
        dist.all_gather_into_tensor(sizes_t, n_owned_t)

        # Build the block-constant assignment via repeat_interleave — no
        # per-rank Python loop or slicing.
        ranks = torch.arange(world_size, dtype=torch.int64, device=device)
        return ranks.repeat_interleave(sizes_t)

    def atom_fields(self) -> dict[str, Any]:
        """Return a shallow copy of the atom-field ShardTensor dict."""
        return dict(self.fields)

    # ------------------------------------------------------------------
    # Construction: scatter from src
    # ------------------------------------------------------------------

    @staticmethod
    def from_batch(
        batch: Batch | None,
        mesh: DeviceMesh,
        config: DomainConfig,
        src: int = 0,
        partition_mode: str = "spatial",
    ) -> ShardedBatch:
        """Scatter a full ``Batch`` from *src* rank across *mesh*.

        Parameters
        ----------
        batch
            Full-system batch on *src*; ``None`` elsewhere.
        mesh
            1-D device mesh for domain parallelism.
        config
            Domain-decomposition config. Its ``mesh`` / ``cutoff`` /
            ``grid_dims`` drive the :class:`SpatialPartitioner` built
            internally.
        src
            The global rank that holds the full batch (default 0).
        partition_mode
            How to assign atoms to ranks.

            * ``"spatial"`` (default) — :class:`SpatialPartitioner`,
              required by halo exchange so a rank's owned atoms' neighbors
              live in adjacent ranks.
            * ``"contiguous_block"`` — atoms ``0..N/W-1`` to rank 0,
              ``N/W..2N/W-1`` to rank 1, and so on. Avoids degenerate
              partitions on geometries spatial would choke on (1D chains,
              perfectly cubic lattices on partition boundaries, clusters in
              oversized cells).

        Returns
        -------
        ShardedBatch

        Notes
        -----
        Atoms are scattered honoring the chosen partitioner's rank assignment
        verbatim (a per-rank point-to-point scatter), not by an even
        ``Shard(0)`` split of ``batch.positions``. A balanced split would
        silently override the partitioner whenever the assignment isn't already
        balanced (e.g. a cluster not centered in the box), placing atoms on
        ranks that don't own their spatial domain so halo exchange can't reach
        their real neighbors.
        """
        if partition_mode not in ("spatial", "contiguous_block"):
            raise ValueError(
                f"partition_mode must be 'spatial' or 'contiguous_block'; "
                f"got {partition_mode!r}"
            )
        local_rank = mesh.get_local_rank()

        # --- Resolve device ---
        if batch is not None:
            device = batch.positions.device
        elif torch.cuda.is_available():
            device = torch.device(f"cuda:{torch.cuda.current_device()}")
        else:
            device = torch.device("cpu")

        # --- Broadcast positions dtype first so cell can match ---
        pos_dtype = _broadcast_float_dtype(
            batch.positions.dtype if batch is not None else None,
            device=device,
            src=src,
        )

        # --- Broadcast cell + pbc + n_global from src ---
        # Cell matches the positions dtype so ``sharded.cell.dtype ==
        # sharded.positions.dtype`` for any downstream op that mixes them.
        if batch is not None:
            cell = batch.cell.clone().to(device=device, dtype=pos_dtype)
            pbc = (
                batch.pbc.clone().to(device=device)
                if _has_field(batch, "pbc")
                else torch.ones(1, 3, dtype=torch.bool, device=device)
            )
            n_global_t = torch.tensor(
                [batch.positions.shape[0]], dtype=torch.int64, device=device
            )
        else:
            cell = torch.zeros(1, 3, 3, dtype=pos_dtype, device=device)
            pbc = torch.ones(1, 3, dtype=torch.bool, device=device)
            n_global_t = torch.zeros(1, dtype=torch.int64, device=device)

        if dist.is_initialized():
            dist.broadcast(cell, src=src)
            dist.broadcast(pbc, src=src)
            dist.broadcast(n_global_t, src=src)
        n_global = int(n_global_t.item())

        # --- Build partitioner from broadcast geometry + config ---
        # Only used in ``spatial`` mode, but retained on the returned
        # ShardedBatch unconditionally so downstream code that consults
        # ``sharded.partitioner`` (skin tracking, ghost padding) still
        # has something to read in ``contiguous_block`` mode too.
        partitioner = SpatialPartitioner(config=config, cell_matrix=cell, pbc=pbc)

        # --- Chemistry prep on src: assign atoms to ranks, order the
        # per-atom fields to match, declare each field's storage policy.
        # The generic broadcast/slice/wrap mechanics are delegated to
        # ``ShardedCollection.scatter`` below — this is the only chemistry-
        # aware part of constructing the distributed collection. ---
        world_size = mesh.size(0) if hasattr(mesh, "size") else dist.get_world_size()
        sizes_list: list[int] | None = None
        source: dict[str, torch.Tensor] | None = None

        if local_rank == src:
            if batch is None:
                raise ValueError("batch must be provided on src rank")
            for name in _REQUIRED_ATOM_FIELDS:
                if not _has_field(batch, name):
                    raise ValueError(
                        f"ShardedBatch.from_batch requires '{name}' on the "
                        "source batch."
                    )

            n_atoms_src = batch.positions.shape[0]
            if partition_mode == "spatial":
                rank_assignment = partitioner.assign_atoms_to_ranks(
                    batch.positions
                ).to(torch.int64)
                # Stable sort so atoms within a rank keep their original order.
                sorted_idx = torch.argsort(rank_assignment, stable=True)
            else:  # contiguous_block
                # ``arange(N) // (N // W)`` with clamp so the last rank absorbs
                # any remainder.
                per_rank = max(n_atoms_src // world_size, 1)
                rank_assignment = (
                    torch.arange(n_atoms_src, dtype=torch.int64) // per_rank
                ).clamp(max=world_size - 1)
                sorted_idx = torch.argsort(rank_assignment, stable=True)

            sorted_assignment = rank_assignment[sorted_idx]
            sizes_list = [
                int((sorted_assignment == r).sum().item())
                for r in range(world_size)
            ]

            # Discover every per-atom field (positions, atomic_numbers, masses,
            # forces, velocities, charges, momenta, custom user fields via
            # ``add_node_property``, …), order it by the rank assignment, and
            # apply any dtype override. One pass; no hand-maintained allowlist.
            schema = _discover_atom_schema(batch)
            source = {
                entry["name"]: getattr(batch, entry["name"])[sorted_idx]
                .to(dtype=entry["dtype"])
                .contiguous()
                for entry in schema
            }

        # --- Broadcast the field order so every rank can build the policy
        # map keyed by field name, then delegate the scatter. ---
        names_holder: list[Any] = [
            list(source.keys()) if source is not None else None
        ]
        if dist.is_initialized():
            dist.broadcast_object_list(names_holder, src=src)
        field_names = names_holder[0]
        assert field_names is not None  # noqa: S101

        policies = ShardedBatch._policies_for(field_names)
        coll = ShardedCollection.scatter(
            source,
            mesh=mesh,
            policies=policies,
            sizes=sizes_list,
            device=device,
            src=src,
        )

        return ShardedBatch(
            mesh=mesh,
            atom_fields=coll.fields,
            cell=cell,
            pbc=pbc,
            n_global=n_global,
            partitioner=partitioner,
            partition_mode=partition_mode,
        )

    # ------------------------------------------------------------------
    # Local view: per-rank owned rows as a plain Batch
    # ------------------------------------------------------------------

    @property
    def local_batch(self) -> Batch:
        """This rank's owned atoms as a plain ``Batch``.

        Calls ``.to_local()`` on each ShardTensor field (no communication,
        no copy — the returned tensors share storage with the shards).
        In-place mutations on the returned batch's tensors propagate back
        to the ``ShardTensor`` automatically; for non-in-place
        replacements, call :meth:`update_from_batch` to sync.
        """
        from nvalchemi.data.atomic_data import AtomicData
        from nvalchemi.data.batch import Batch as BatchCls

        # Per-field local view via each field's policy (PlainShard ->
        # ``to_local()`` owned rows).
        locals_ = self.local()
        device = locals_["positions"].device

        # Hot-path construction: bypass pydantic validation via
        # ``model_construct``. AtomicData's ``atom_categories`` Enum-coercion
        # path calls ``repr`` on each tensor, which on CUDA syncs per element
        # — hundreds of host syncs per forward. Skipping validation is safe
        # here because ``self.fields`` holds tensors already validated at
        # scatter time and the per-forward Batch is internal.
        ctor_known: set[str] = set(AtomicData.model_fields)
        ctor_kwargs: dict[str, Any] = {
            "cell": self.cell if self.cell.ndim == 3 else self.cell.unsqueeze(0),
            "pbc": self.pbc if self.pbc.ndim == 2 else self.pbc.unsqueeze(0),
        }
        extras: dict[str, torch.Tensor] = {}
        for name, tensor in locals_.items():
            if name in ctor_known:
                ctor_kwargs[name] = tensor
            else:
                extras[name] = tensor
        data = AtomicData.model_construct(**ctor_kwargs)
        # Custom fields (not on the model) still need add_node_property
        # for the level_storage bookkeeping. Those don't carry the
        # Enum-coercion bug because the slow path is in the model's
        # own field-validator chain, which extras bypass entirely.
        for name, tensor in extras.items():
            data.add_node_property(name, tensor)

        return BatchCls.from_data_list([data], device=device)

    # ------------------------------------------------------------------
    # Gathering: local shards → full Batch
    # ------------------------------------------------------------------

    def full_batch(self, dst: int = 0) -> Batch | None:
        """Gather all shards into a full ``Batch`` on rank *dst*.

        All ranks must call this — the underlying send/recv is collective.
        Returns ``None`` on ranks other than *dst*.
        """
        gathered = self.gather(dst=dst)
        if gathered is None:
            return None
        return self._build_batch_from_tensors(gathered)

    def to_global_batch(self) -> Batch:
        """Gather all shards into a full ``Batch`` on **every** rank."""
        gathered = self.gather(dst=None)
        assert gathered is not None  # dst=None populates every rank  # noqa: S101
        return self._build_batch_from_tensors(gathered)

    def _build_batch_from_tensors(self, tensors: dict[str, torch.Tensor]) -> Batch:
        from nvalchemi.data.atomic_data import AtomicData
        from nvalchemi.data.batch import Batch as BatchCls

        tensors = dict(tensors)
        device = tensors["positions"].device
        data = AtomicData(
            positions=tensors.pop("positions"),
            atomic_numbers=tensors.pop("atomic_numbers"),
            atomic_masses=tensors.pop("atomic_masses"),
            cell=self.cell.clone(),
            pbc=self.pbc.clone(),
        )
        for name, tensor in tensors.items():
            data.add_node_property(name, tensor)

        return BatchCls.from_data_list([data], device=device)

    # ------------------------------------------------------------------
    # Syncing back: replaced tensors → ShardTensor storage
    # ------------------------------------------------------------------

    def update_from_batch(self, batch: Batch) -> None:
        """Sync non-in-place tensor replacements from *batch* back into
        the ``ShardTensor`` backing storage.

        In-place mutations are already reflected automatically because
        ``to_local()`` returns the backing storage. This method rewraps
        any per-atom field whose identity has changed on the plain batch.
        """
        from torch.distributed.tensor import Shard

        from nvalchemi.distributed._core._st_backend import ShardTensor

        # Atom migration can change this rank's local row count, invalidating
        # the old ``sharding_shapes``. Whether n_owned drifted is per-rank
        # state, so gating the all_gather on a local check would fire
        # asymmetrically across ranks and diverge collective order. Always
        # all_gather (one int per rank) to keep every rank in lockstep.
        sizes_dim0_cached: list[int] | None = None
        for name in self.fields:
            if not _has_field(batch, name):
                continue
            batch_tensor = getattr(batch, name)
            if batch_tensor is None:
                continue
            sizes_dim0_cached = self._all_gather_n_owned(int(batch_tensor.shape[0]))
            break

        for name in list(self.fields.keys()):
            if not _has_field(batch, name):
                continue
            batch_tensor = getattr(batch, name)
            st = self.fields[name]
            if batch_tensor is not st.to_local():
                if sizes_dim0_cached is not None:
                    sizes_dim0 = sizes_dim0_cached
                else:
                    old_shapes_by_dim = st._spec.sharding_shapes()
                    sizes_dim0 = [int(s[0]) for s in old_shapes_by_dim[0]]
                new_trailing = tuple(batch_tensor.shape[1:])
                new_shapes = {
                    0: tuple(torch.Size((s,) + new_trailing) for s in sizes_dim0)
                }
                self.fields[name] = ShardTensor.from_local(
                    batch_tensor,
                    self.mesh,
                    (Shard(0),),
                    sharding_shapes=new_shapes,
                )

    def _all_gather_n_owned(self, my_n: int) -> list[int]:
        """All-gather per-rank ``n_owned`` so every rank knows the full
        post-migration layout. Cheap (one int per rank); skipped when
        single-process.
        """
        if not dist.is_initialized():
            return [my_n]
        group = mesh_group(self.mesh)
        world_size = dist.get_world_size(group=group)
        device = (
            torch.device("cuda", torch.cuda.current_device())
            if torch.cuda.is_available()
            and dist.get_backend(group) == dist.Backend.NCCL
            else torch.device("cpu")
        )
        my_t = torch.tensor([my_n], dtype=torch.long, device=device)
        out = torch.empty(world_size, dtype=torch.long, device=device)
        dist.all_gather_into_tensor(out, my_t, group=group)
        return [int(x) for x in out.tolist()]
