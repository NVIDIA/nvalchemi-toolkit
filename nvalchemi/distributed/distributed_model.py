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

"""Adapter that turns a single-process model wrapper into a distributed callable.

Pattern::

    wrapper    = MACEWrapper.from_checkpoint("small")
    sharded    = ShardedBatch.from_batch(full_batch, mesh=m, config=cfg)
    dist_model = DistributedModel(wrapper, cfg)
    out        = dist_model(sharded)        # dict[str, Tensor]
    e          = out["energy"]              # globally reduced
    f          = out["forces"]              # per-rank owned rows

The adapter owns framework concerns (halo padding, output consolidation) so
inner wrappers stay single-process-focused. Per-model distributed knowledge
lives in the wrapper's ``distribution_spec``.

Composite wrappers (``PipelineModelWrapper``) are rejected here — use
``DistributedPipelineModel`` for distributed composition.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

import torch

from nvalchemi.data import Batch
from nvalchemi.distributed._core.context import (
    DistributedContext,
    activate_dd_context,
)
from nvalchemi.distributed._core.particle_halo import ParticleHaloConfig
from nvalchemi.distributed.config import DomainConfig
from nvalchemi.distributed.output_consolidation import (
    consolidate_padded_outputs,
)
from nvalchemi.distributed.partitioner import SpatialPartitioner
from nvalchemi.neighbors import compute_neighbors

if TYPE_CHECKING:
    from nvalchemi.distributed.sharded_batch import ShardedBatch
    from nvalchemi.models.base import BaseModelMixin


__all__ = ["DistributedModel", "DistributionError"]


def _prepare_dd_compile(spec: "Any", compile_kwargs: "dict | None") -> dict:
    """Validate the spec supports a compiled distributed forward, tune Dynamo,
    and return the resolved ``torch.compile`` kwargs.

    Distributed compile is fixed-shape (graphs are padded to per-rank caps), so
    ``dynamic`` defaults to ``False``. Cache limits are raised because the path
    fans out many per-layer shape variants that would otherwise fall back to
    eager. Raises if the spec declares no ``CompilePolicy``.
    """
    cp = getattr(spec, "compile", None)
    if cp is None:
        raise DistributionError(
            "compile=True requires the model's distribution_spec to declare a "
            "CompilePolicy (force_strategy). This model does not support a "
            "compiled distributed forward."
        )
    import os  # noqa: PLC0415

    import torch._dynamo as _td  # noqa: PLC0415

    _td.config.cache_size_limit = max(_td.config.cache_size_limit, 64)
    _td.config.accumulated_cache_size_limit = max(
        _td.config.accumulated_cache_size_limit, 512
    )
    _td.config.force_parameter_static_shapes = False
    # Optional activation-memory budget (env-gated): backward can't recompute
    # across opaque custom ops, so it saves their outputs, which dominates peak
    # memory. <1.0 recomputes the rest instead. Unset -> default 1.0.
    _actb = os.environ.get("NVALCHEMI_ACT_BUDGET")
    if _actb:
        import torch._functorch.config as _fcfg  # noqa: PLC0415

        _fcfg.activation_memory_budget = float(_actb)
    ck = dict(compile_kwargs or {})
    ck.setdefault("dynamic", False)
    return ck


def _partition_health_verdict(
    n_owned: int, n_padded: int, group: Any, device: Any
) -> tuple[bool, bool, int]:
    """Collective verdict on halo-partition health, identical on every rank.

    Returns ``(any_empty, any_trivial, n_global)``:

    * ``any_empty`` — some rank has 0 owned atoms (more ranks than the geometry
      can fill); the caller raises. Genuinely broken.
    * ``any_trivial`` — some rank's halo already covers every atom
      (0 remote atoms); the caller warns. Correct but no parallelism is gained.

    A SUM gives the global atom count; a MAX over the two flags shares the
    verdict so no rank raises while others proceed (avoids collective desync).
    """
    import torch.distributed as dist  # noqa: PLC0415

    gsum = torch.tensor([float(n_owned)], device=device)
    dist.all_reduce(gsum, op=dist.ReduceOp.SUM, group=group)
    n_global = int(round(gsum.item()))
    flags = torch.tensor(
        [1 if n_owned == 0 else 0, 1 if n_padded >= n_global else 0],
        device=device,
        dtype=torch.int32,
    )
    dist.all_reduce(flags, op=dist.ReduceOp.MAX, group=group)
    return bool(flags[0].item()), bool(flags[1].item()), n_global


def _mark_halo_receiver_edges_as_padding(padded_batch: "Batch", n_owned: int) -> None:
    """Rewrite ``neighbor_list`` so halo-receiver edges look like the
    padding-sentinel rows ``compute_neighbors`` already emits.

    Each global edge is replicated on every rank holding both endpoints, so a
    per-receiver scatter must count each edge on exactly one rank — the one
    owning its receiver — else halo-receiver edges double-count. Wrappers
    already drop genuine padding rows (indices == ``num_nodes``) via a
    ``(edge_index < n_atoms)`` filter; marking halo-receiver rows with the same
    sentinel routes them through that drop with no per-rank logic in the wrapper.

    Sync-free and idempotent: one compare plus one in-place ``masked_fill_``.
    No-ops when the ``edges`` group is missing, the NL is empty, or
    ``n_owned == n_padded`` (single-process).
    """
    edges = padded_batch._edges_group
    if edges is None:
        return
    nl = edges._data.get("neighbor_list")
    if nl is None or nl.shape[0] == 0:
        return
    sentinel = padded_batch.num_nodes  # matches compute_neighbors padding
    halo_recv = nl[:, 1] >= n_owned
    nl[:, 1].masked_fill_(halo_recv, sentinel)


def _build_halo_meta_packed(
    meta: "Any", config: "Any", device: "Any", n_pad: int, max_send_cap: "int | None" = None
) -> "Any":
    """Build the fixed-shape halo routing tensor from the per-step ``meta``, or
    ``None`` when there is no cross-rank halo.

    Carried as a graph input under compile, this lets the compile-path halo
    handlers route through the static halo ops with the routing as a runtime
    tensor rather than baked-in constants.

    ``max_send`` is the max over ``meta.send_sizes`` — the all-gathered
    send-count matrix, identical on every rank — so the cap is consistent
    across ranks.
    """
    if not meta.send_sizes or meta.n_padded <= meta.n_owned:
        return None
    max_send = max((max(row) for row in meta.send_sizes), default=0)
    if max_send <= 0:
        return None
    # Use the fixed per-rank cap when compiling so the routing tensor keeps a
    # constant shape across steps -> no recompile as send counts drift.
    eff_max_send = int(max_send_cap) if max_send_cap is not None else int(max_send)
    from nvalchemi.distributed._core.particle_halo import (  # noqa: PLC0415
        build_halo_meta_tensors,
        pack_halo_meta,
    )

    si, rd, rr, no = build_halo_meta_tensors(
        meta, config.rank, eff_max_send, n_pad, device
    )
    return pack_halo_meta(si, rd, rr, no)


def _promote_positions_to_shardtensor(
    padded_batch: "Batch",
    spec: "Any",
    meta: "Any",
    config: "ParticleHaloConfig",
    n_systems: int,
    max_send_cap: "int | None" = None,
) -> None:
    """Wrap the padded batch's per-atom fields in-place as ShardTensors.

    Mutates the ``_atoms_group`` slots named by ``spec.distribution.shard_fields``
    so each primary op input (e.g. ``positions``, ``charges``,
    ``atomic_numbers``) is a ShardTensor. Custom ops consuming them fire
    ShardTensor dispatch, which routes their outputs through the registered
    per-system / halo-correction handlers.

    A field is promoted whenever an op needs a ShardTensor arg for its handler
    to fire (e.g. ``charges`` for the PME total-charge op, ``atomic_numbers`` so
    one-hot encoding carries ShardTensor-ness into ``node_attrs``). The set is
    spec-driven, so each model promotes exactly the fields it needs.
    """
    from nvalchemi.distributed._core.shard_tensor import ShardTensor

    atoms = padded_batch._atoms_group
    if atoms is None:
        return
    # Build the fixed-shape halo routing once (same for every per-atom field).
    # Under compile the handlers route through the static halo ops with this as
    # a runtime graph input; eager ignores it.
    _pos = atoms.get("positions")
    _device = _pos.device if _pos is not None else None
    halo_meta_packed = (
        _build_halo_meta_packed(meta, config, _device, int(_pos.shape[0]), max_send_cap)
        if _device is not None
        else None
    )
    # Spec-driven, always a concrete tuple, so ``()`` (promote nothing) is valid.
    for key in spec.distribution.shard_fields:
        if key not in atoms:
            continue
        t = atoms[key]
        if isinstance(t, ShardTensor):
            continue
        atoms[key] = ShardTensor.wrap(
            t,
            spec=spec,
            meta=meta,
            config=config,
            n_systems=n_systems,
            halo_meta_packed=halo_meta_packed,
        )


class DistributionError(ValueError):
    """Raised when a wrapper cannot be adapted by :class:`DistributedModel`.

    Typical causes: the wrapper is composite (``PipelineModelWrapper`` — use
    ``DistributedPipelineModel``); or its ``distribution_spec`` is ``None``.
    """


class DistributedModel:
    """Wrap an atomic single-process model wrapper for domain-decomposed
    inference.

    Parameters
    ----------
    wrapper
        Atomic :class:`~nvalchemi.models.base.BaseModelMixin`. Its
        ``distribution_spec`` must be non-None. Composite wrappers
        (``PipelineModelWrapper``) are rejected — use
        :class:`DistributedPipelineModel` for composition.
    domain_config
        Shared simulation config carrying the cutoff, skin, mesh, and
        optional grid_dims. The partitioner and halo config are built
        lazily from the first :class:`ShardedBatch`'s geometry.

    Notes
    -----
    Construction is side-effect-free. The first call to ``__call__``
    initializes the partitioner / halo config / world size from the
    supplied ``ShardedBatch`` and invokes
    ``wrapper.distributed_setup``.

    ``close()`` — or ``__exit__`` / ``__del__`` — calls
    ``wrapper.distributed_teardown`` to restore any module-level state.
    Use as a context manager for scoped lifecycle::

        with DistributedModel(wrapper, config) as dist_model:
            out = dist_model(sharded)
    """

    def __init__(
        self,
        wrapper: "BaseModelMixin",
        domain_config: DomainConfig,
        *,
        spec: "MLIPSpec | None" = None,
        compile: bool = False,
        compile_kwargs: dict | None = None,
    ) -> None:
        # Reject composite wrappers. Delayed import avoids a circular import.
        from nvalchemi.models.pipeline import PipelineModelWrapper

        if isinstance(wrapper, PipelineModelWrapper):
            raise DistributionError(
                "DistributedModel wraps atomic BaseModelMixin instances only. "
                "For composite wrappers, compose their adapters via "
                "DistributedPipelineModel([...])."
            )

        # Explicit ``spec=`` wins, else fall back to ``wrapper.distribution_spec``.
        if spec is None:
            spec = getattr(wrapper, "distribution_spec", None)
        if spec is None:
            raise DistributionError(
                f"{type(wrapper).__name__} has distribution_spec=None and no "
                "explicit spec= was passed. Atomic wrappers must either "
                "declare a MLIPSpec property or be constructed via "
                "`DistributedModel(wrapper, cfg, spec=...)`."
            )

        from nvalchemi.distributed._core.adapter import (  # noqa: PLC0415
            AdapterRegistry,
        )

        self._wrapper = wrapper
        # Fixed-shape padding caps (compile-only, per-rank), keyed
        # "atoms"/"edges"/"max_send". Grown on overflow; empty until first
        # compiled forward.
        self._cap_state: dict[str, int] = {}
        self._config = domain_config
        self._spec = spec
        # ``compile=True`` makes the forward compile the energy-autograd path.
        # The spec carries only the compile contract; the switch lives here.
        self._dd_compile_requested: bool = bool(compile)
        self._dd_compile_kwargs: dict | None = (
            _prepare_dd_compile(self._spec, compile_kwargs) if compile else None
        )
        # Fixed-shape graph padder for the compiled halo path. A model may
        # declare a custom padder via its CompilePolicy; the default is the
        # generic COO ``edge_index`` padder, so a standard MPNN declares nothing.
        from nvalchemi.distributed.graph_padder import COOPadder  # noqa: PLC0415

        _compile_policy = getattr(self._spec, "compile", None)
        self._graph_padder = (
            getattr(_compile_policy, "graph_padder", None) or COOPadder()
        )
        self._setup_called = False
        # Installs/restores the spec's custom_ops + third_party_helpers.
        # Populated on first forward; restored in ``close()``.
        self.adapter_registry: AdapterRegistry = AdapterRegistry()

        # Lazy-built from the first batch's geometry (cell / pbc).
        self._partitioner: SpatialPartitioner | None = None
        self._halo_config: ParticleHaloConfig | None = None

        # Per-scope runtime context, built in ``_ensure_initialized`` and shared
        # by reference with the wrapper so per-step mutations are visible.
        self._dist_ctx: DistributedContext | None = None

        # World size, read from the mesh on first call (or 1).
        self._world_size: int | None = None

        # Partition-health check runs once (first halo forward).
        self._partition_health_checked: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def wrapper(self) -> "BaseModelMixin":
        """The underlying single-process model wrapper."""
        return self._wrapper

    @property
    def config(self) -> DomainConfig:
        """The :class:`DomainConfig` held by this adapter."""
        return self._config

    def __call__(
        self,
        sharded: "ShardedBatch",
        *,
        wired_fields: "dict[str, Any] | None" = None,
    ) -> dict[str, Any]:
        """Run a distributed forward on a :class:`ShardedBatch`.

        Parameters
        ----------
        sharded : ShardedBatch
            The sharded system to run the forward on.
        wired_fields : dict[str, Any] | None
            Optional ``{field_name: owned_value}`` overrides for per-atom inputs
            produced by an upstream model (cross-model composition). Each owned
            tensor is gathered into *this* model's ghost layout via the
            autograd-aware :func:`halo_forward_exchange` and written onto the
            padded batch before the forward, so the consumer sees the producer's
            value on its ghosts and the pathway stays differentiable (backward
            scatter-adds ghost grads to owners). Eager-only; raises under
            compiled distribution.

        Returns
        -------
        dict[str, Any]
            Output dict with owned-shape (per-atom) and replicated
            (per-system) tensors.

        Notes
        -----
        Halo exchange and neighbor-list management are the caller's
        responsibility (typically via :func:`halo_exchange` +
        ``NeighborListHook`` inside ``DomainParallel``). The adapter handles
        spec-driven input adaptation, the wrapper forward, and output
        consolidation.
        """
        from nvalchemi.distributed._core.gather_primitives import (  # noqa: PLC0415
            _clear_exchange_counts_cache,
        )
        # The exchange-counts cache key is per-rank, so ranks with different
        # send histories could diverge and deadlock. Resetting symmetrically
        # each forward avoids this at the cost of one redundant collective
        # (per-layer reuse within a forward is preserved).
        _clear_exchange_counts_cache()

        self._ensure_initialized(sharded)

        # Each storage policy owns its distributed forward, so a new strategy
        # plugs in without a framework type-switch.
        return self._spec.distribution.policy.run_forward(
            self, sharded, wired_fields
        )

    def from_batch(self, batch: "Batch | None", *, src: int = 0) -> dict[str, Any]:
        """One-call distributed inference from a full ``Batch``.

        The convenience entry for one-off inference: shards ``batch`` across the
        scope's mesh (via :meth:`ShardedBatch.from_batch`, using the held
        :class:`DomainConfig`) and runs the distributed forward — so a caller
        never constructs a :class:`ShardedBatch` by hand. Collective: every rank
        calls it, with the full system on rank ``src`` and ``None`` elsewhere;
        every rank gets the consolidated output dict back.

        Parameters
        ----------
        batch
            The full-system :class:`~nvalchemi.data.Batch` on rank ``src``;
            ``None`` on the other ranks.
        src
            The rank holding the full batch (default 0).

        Returns
        -------
        dict[str, Any]
            The consolidated outputs (owned-shape per-atom + replicated
            per-system), identical to calling :meth:`__call__` on a hand-built
            :class:`ShardedBatch`.
        """
        from nvalchemi.distributed.sharded_batch import ShardedBatch  # noqa: PLC0415

        # The policy chooses how atoms map to ranks: spatial (halo) or balanced
        # index ranges (graph parallel).
        partition_mode = getattr(
            self._spec.distribution.policy, "partition_mode", "spatial"
        )
        sharded = ShardedBatch.from_batch(
            batch,
            mesh=self._config.mesh,
            config=self._config,
            src=src,
            partition_mode=partition_mode,
        )
        return self(sharded)

    def close(self) -> None:
        """Release resources and restore any state setup mutated. Safe
        to call multiple times.

        Restores all adapters installed by
        :attr:`adapter_registry` (custom_ops + third_party_helpers),
        then defers to the wrapper's optional ``distributed_teardown``
        hook for any wrapper-side runtime state.
        """
        if self._setup_called:
            self.adapter_registry.restore()
            from nvalchemi.distributed._core.adapter import (  # noqa: PLC0415
                restore_auto_marshalled,
            )

            restore_auto_marshalled(getattr(self, "_auto_marshal_mementos", []))

            if hasattr(self._wrapper, "distributed_teardown"):
                self._wrapper.distributed_teardown()
            self._setup_called = False

    def __enter__(self) -> "DistributedModel":
        # Drop the process-global exchange-counts cache so the first forward in
        # this context starts cold; a stale entry (recv_counts depend on all
        # ranks' send_counts) could deadlock if some ranks hit it and others
        # recompute the all_gather.
        from nvalchemi.distributed._core.gather_primitives import (  # noqa: PLC0415
            _clear_exchange_counts_cache,
        )

        _clear_exchange_counts_cache()
        return self

    def __exit__(self, *_exc: Any) -> None:
        from nvalchemi.distributed._core.gather_primitives import (  # noqa: PLC0415
            _clear_exchange_counts_cache,
        )

        _clear_exchange_counts_cache()
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:  # noqa: S110
            pass

    # ------------------------------------------------------------------
    # Initialization: build partitioner + halo config + world size
    # ------------------------------------------------------------------

    def _ensure_initialized(self, sharded: "ShardedBatch") -> None:
        """Build the halo config from the sharded batch's geometry the
        first time we see one. When available, reuse the partitioner
        cached on :attr:`ShardedBatch.partitioner` (built there from the
        same config + broadcast cell/pbc) to avoid duplicate work and
        potential drift. Fall back to constructing one from the sharded
        batch's geometry when not available (e.g. gloo-harness batches
        built outside :meth:`ShardedBatch.from_batch`).
        """
        if self._partitioner is not None:
            return

        # The policy owns its topology — spatial partitioner + halo config, or a
        # balanced index partition with no ghost shell — so this stays generic.
        self._partitioner, self._halo_config = (
            self._spec.distribution.policy.build_topology(self._config, sharded)
        )

        # World size from the configured mesh; default to 1.
        if self._config.mesh is not None:
            try:
                self._world_size = self._config.mesh.size()
            except Exception:
                self._world_size = 1
        else:
            self._world_size = 1

        # Per-scope runtime context; per-step fields are mutated by the call
        # paths below.
        self._dist_ctx = DistributedContext(
            mesh=self._config.mesh,
            halo_config=self._halo_config,
            n_systems_global=sharded.num_graphs,
            n_atoms_total=sharded.n_global,
        )

        # Spec-driven adapter installation: install every custom_op and
        # third_party_helper in declaration order; restored in close().
        from nvalchemi.distributed._core.adapter import (  # noqa: PLC0415
            JitAdapter,
            auto_marshal_scripted_submodules,
        )

        # Scripted-op marshalling mode: env override > config > "auto".
        marshal_mode = os.environ.get("NVALCHEMI_SCRIPTED_MARSHAL") or getattr(
            self._config, "scripted_marshal", "auto"
        )
        if marshal_mode not in ("auto", "declared", "off"):
            marshal_mode = "auto"

        adapters = list(self._spec.distribution.custom_ops) + list(
            self._spec.distribution.third_party_helpers
        )
        if marshal_mode == "off":
            # Drop marshal-mode JitAdapters; leave eager JitAdapters /
            # PythonAdapters / OpAdapters in place.
            adapters = [
                a
                for a in adapters
                if not (
                    isinstance(a, JitAdapter) and getattr(a, "mode", "eager") == "marshal"
                )
            ]
        self.adapter_registry.install(adapters)

        # Auto-discovery ("auto" mode only): wrap scripted submodules' forward
        # with the marshaller, deduped against declared adapters and the config
        # exclude-list. Restored in close().
        self._auto_marshal_mementos: list[Any] = []
        if marshal_mode == "auto":
            declared_targets = tuple(
                a.attr_name
                for a in self._spec.distribution.third_party_helpers
                if isinstance(a, JitAdapter)
            )
            self._auto_marshal_mementos = auto_marshal_scripted_submodules(
                self._wrapper,
                exclude=tuple(getattr(self._config, "scripted_marshal_exclude", ())),
                declared_targets=declared_targets,
            )

        # Always invoke the wrapper's setup hook last, so wrappers that
        # build closures over ``ctx.gather_meta`` see the spec handlers
        # already in place.
        if hasattr(self._wrapper, "distributed_setup"):
            self._wrapper.distributed_setup(self._dist_ctx)
        self._setup_called = True

    def _needs_forces(self) -> bool:
        return bool(
            self._wrapper.model_config.autograd_outputs
            & self._wrapper.model_config.active_outputs
        )

    # ------------------------------------------------------------------
    # Halo-storage path
    # ------------------------------------------------------------------

    def _check_partition_health(self, meta: Any, device: Any) -> None:
        """Flag a degenerate halo partition once (first halo forward).

        An empty shard (a rank with 0 owned atoms — more ranks than the
        geometry can fill) is broken: raise on every rank. A trivial partition
        (every rank's halo covers the whole system, 0 remote atoms) is correct
        but gains no parallelism — warn once. The verdict is taken collectively
        so every rank acts identically (avoids desync from one rank raising)."""
        if self._partition_health_checked:
            return
        self._partition_health_checked = True
        if not self._world_size or self._world_size <= 1:
            return  # single process — not domain-decomposed

        from nvalchemi.distributed._core.gather_primitives import (  # noqa: PLC0415
            mesh_group,
        )

        group = mesh_group(self._halo_config.mesh)
        any_empty, any_trivial, n_global = _partition_health_verdict(
            int(meta.n_owned), int(meta.n_padded), group, device
        )
        if any_empty:
            raise RuntimeError(
                "Degenerate domain decomposition: a rank was assigned 0 owned "
                f"atoms (world_size={self._world_size}, total atoms={n_global}). "
                "There are more ranks than this geometry can partition — use "
                "fewer ranks or a larger system."
            )
        if any_trivial:
            rank = (
                self._config.mesh.get_local_rank()
                if self._config.mesh is not None
                else 0
            )
            if rank == 0:
                from loguru import logger  # noqa: PLC0415

                logger.warning(
                    "Degenerate (trivial) domain decomposition: every rank's "
                    f"halo already covers all {n_global} atoms (0 remote atoms), "
                    "so domain parallelism gains nothing here — results are still "
                    "correct, but each rank does the full system's work. This "
                    "happens when box/ranks <= ~2*ghost_width (ghost_width = "
                    "cutoff + skin); use fewer ranks or a larger system to "
                    "actually decompose."
                )

    def _graph_parallel_owned_edges(
        self, sharded: "ShardedBatch", meta: Any, rank: int
    ) -> torch.Tensor:
        """This rank's ``(E, 2)`` owned-target neighbor list for the GP path.

        Materializes the full graph once from the replicated geometry, keeps the
        edges whose receiver this rank owns, and remaps that receiver to its
        owned-local row; senders stay global ids into the per-layer replicated
        node tensor. The edge index is non-differentiable routing — the
        differentiable geometry flows through ``refresh_neighbors`` in the
        wrapper — so the gather + neighbor build run under ``no_grad``.
        """
        with torch.no_grad():
            global_batch = sharded.to_global_batch()
            compute_neighbors(
                global_batch, config=self._wrapper.model_config.neighbor_config
            )
        nl = global_batch.neighbor_list.to(torch.long)
        src_g, dst_g = nl[:, 0], nl[:, 1]
        owner = meta.owner_rank.to(dst_g.device)
        local = meta.local_index.to(dst_g.device)
        keep = owner[dst_g] == rank
        return torch.stack([src_g[keep], local[dst_g[keep]]], dim=1)

    def _call_graph_parallel(
        self,
        sharded: "ShardedBatch",
        wired_fields: "dict[str, Any] | None" = None,
    ) -> dict[str, Any]:
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
        import torch.distributed as dist  # noqa: PLC0415

        from nvalchemi.distributed._core.placement import (  # noqa: PLC0415
            ShardRouting,
        )
        from nvalchemi.distributed.output_consolidation import (  # noqa: PLC0415
            consolidate_sharded_outputs,
        )

        mesh = self._config.mesh
        rank = mesh.get_local_rank() if mesh is not None else 0
        world = self._world_size or 1

        # Global<->owned index map for the balanced partition.
        assignment = sharded.rank_assignment
        meta = ShardRouting.from_assignment(assignment, rank, world)
        meta.n_systems_global = sharded.num_graphs

        nl = self._graph_parallel_owned_edges(sharded, meta, rank)

        # Owned rows as a plain batch carrying the prepared edges; positions
        # become a fresh autograd leaf for the energy-force grad.
        owned = sharded.local_batch_with_edges({"neighbor_list": nl})
        atoms = owned._atoms_group
        pos = atoms["positions"]
        pos = (pos.to_local() if hasattr(pos, "to_local") else pos).detach()
        pos.requires_grad_(True)
        atoms["positions"] = pos

        # Publish the per-step routing + policy so the wrapper's intent verbs
        # (refresh_neighbors / system_sum) resolve to the GP collectives.
        self._dist_ctx.policy = self._spec.distribution.policy
        self._dist_ctx.gather_meta = meta
        self._dist_ctx.halo_meta = None

        with activate_dd_context(self._dist_ctx):
            output = self._wrapper(owned)
            # The wrapper returns this rank's owned per-graph energy partial.
            # Forces differentiate that partial: the per-layer node-gather's
            # reduce-scatter adjoint already routes each owned atom's cross-rank
            # gradient back, so the owned forces come out globally-correct.
            energy_partial = output["energy"]
            if self._needs_forces():
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

        return consolidate_sharded_outputs(
            output,
            model_config=self._wrapper.model_config,
            world_size=self._world_size,
            owned_only_outputs=self._spec.owned_only_outputs,
            all_reduce_outputs=self._spec.all_reduce_outputs,
            halo_config=self._halo_config,
        )

    def _call_graph_replicate(
        self,
        sharded: "ShardedBatch",
        wired_fields: "dict[str, Any] | None" = None,
    ) -> dict[str, Any]:
        """Node-replicate graph-parallel forward for *opaque* models.

        Every rank holds the full node set; the global edge list is sharded
        across ranks; the model's forward runs unchanged on its edge slice and a
        declared conv adapter all-reduces each layer's partial message
        (``scatter_to_owners`` → :meth:`GraphReplicatePolicy.fold`) so the
        nonlinear node-wise ops downstream see the complete node features. After
        all layers the node features (hence per-node energy) are identical on
        every rank, so the energy is taken as-is; forces are the rank-local
        ``-dE/dx`` summed across ranks (each edge contributes on exactly one
        rank — no double count, no division).
        """
        if wired_fields:
            raise NotImplementedError(
                "wired_fields (cross-model field injection) is not supported on "
                "the graph-parallel path."
            )
        import torch.distributed as dist  # noqa: PLC0415

        from nvalchemi.data.atomic_data import AtomicData  # noqa: PLC0415
        from nvalchemi.data.batch import Batch as BatchCls  # noqa: PLC0415
        from nvalchemi.distributed._core.gather_primitives import (  # noqa: PLC0415
            mesh_group,
        )
        from nvalchemi.distributed.output_consolidation import (  # noqa: PLC0415
            consolidate_sharded_outputs,
        )

        mesh = self._config.mesh
        rank = mesh.get_local_rank() if mesh is not None else 0
        world = self._world_size or 1

        # Full node set on every rank; build the global graph; shard the edges.
        full = sharded.to_global_batch()
        with torch.no_grad():
            compute_neighbors(
                full, config=self._wrapper.model_config.neighbor_config
            )
        nl = full.neighbor_list
        shifts = getattr(full, "neighbor_list_shifts", None)
        n_edges = nl.shape[0]
        lo = (n_edges * rank) // world
        hi = (n_edges * (rank + 1)) // world

        # Reassemble the per-rank batch: full atoms + this rank's edge slice.
        # Positions become a fresh leaf for the force autograd.
        atoms = full._atoms_group
        pos = atoms["positions"].detach().requires_grad_(True)
        ctor = {"positions": pos}
        known = set(AtomicData.model_fields)
        extras: dict[str, Any] = {}
        for name, t in atoms.items():
            if name == "positions":
                continue
            (ctor if name in known else extras)[name] = t
        data = AtomicData(
            **ctor,
            cell=full.cell if full.cell.ndim == 3 else full.cell.unsqueeze(0),
            pbc=full.pbc if full.pbc.ndim == 2 else full.pbc.unsqueeze(0),
        )
        for name, t in extras.items():
            data.add_node_property(name, t)
        data.add_edge_property("neighbor_list", nl[lo:hi].contiguous())
        if shifts is not None:
            data.add_edge_property(
                "neighbor_list_shifts", shifts[lo:hi].contiguous()
            )
        batch_r = BatchCls.from_data_list([data], device=pos.device)

        # Publish the policy so the conv adapter's scatter_to_owners resolves to
        # the all-reduce recombine.
        self._dist_ctx.policy = self._spec.distribution.policy
        self._dist_ctx.gather_meta = None
        self._dist_ctx.halo_meta = None

        # Run energy-only: the wrapper emits per-node energies under
        # ``node_energy_key`` and the framework takes the force autograd. Widen
        # active outputs to that key for the forward; restored in ``finally``.
        nek = self._spec.node_energy_key or "atomic_energies"
        mc = self._wrapper.model_config
        saved_active = None
        if nek in mc.outputs and mc.active_outputs != {nek}:
            saved_active = mc.active_outputs
            mc.active_outputs = {nek}

        # Compile the energy-only forward when requested. The per-layer message
        # recombine (a mesh-static funcol all-reduce) traces inside the graph;
        # there is no per-step routing to thread (unlike halo), so the region is
        # just the wrapper forward. Forces autograd run outside, over the leaf.
        _compile = bool(
            self._dd_compile_requested
            and self._spec.compile is not None
            and self._spec.compile.forces_via_autograd
        )
        try:
            with activate_dd_context(self._dist_ctx):
                output = (
                    self._gp_replicate_compiled_region()(batch_r)
                    if _compile
                    else self._wrapper(batch_r)
                )
        finally:
            if saved_active is not None:
                mc.active_outputs = saved_active

        # Read the energy off this rank's OWNED node slice. Node features are
        # identical on every rank after the per-layer recombine, so any contiguous
        # owned partition is correct — and it makes each rank's energy gradient
        # DISTINCT, so the conv recombine's all-reduce adjoint sums distinct
        # partials (no replicated-energy over-count). Forces are the rank-local
        # ``-dE/dx`` summed across ranks (edges partitioned → no double count);
        # the reported energy is the owned partials summed.
        n_atoms = pos.shape[0]
        nlo = (n_atoms * rank) // world
        nhi = (n_atoms * (rank + 1)) // world
        atomic_e = output[nek]
        batch_idx = batch_r.batch_idx.long()
        owned_e = atomic_e[nlo:nhi]
        energy_local = owned_e.new_zeros(int(batch_r.num_graphs)).index_add(
            0, batch_idx[nlo:nhi], owned_e
        )
        if self._needs_forces():
            (grad,) = torch.autograd.grad(
                [energy_local.sum()],
                [pos],
                create_graph=False,
                retain_graph=False,
                allow_unused=True,
            )
            forces = torch.zeros_like(pos) if grad is None else -grad
            if dist.is_initialized() and world > 1:
                dist.all_reduce(forces, op=dist.ReduceOp.SUM, group=mesh_group(mesh))
            output["forces"] = forces
        energy_global = energy_local.detach().clone()
        if dist.is_initialized() and world > 1:
            dist.all_reduce(energy_global, op=dist.ReduceOp.SUM, group=mesh_group(mesh))
        output["energy"] = energy_global

        return consolidate_sharded_outputs(
            output,
            model_config=self._wrapper.model_config,
            world_size=self._world_size,
            owned_only_outputs=self._spec.owned_only_outputs,
            all_reduce_outputs=self._spec.all_reduce_outputs,
            halo_config=self._halo_config,
        )

    def _call_halo_storage(
        self,
        sharded: "ShardedBatch",
        wired_fields: "dict[str, Any] | None" = None,
    ) -> dict[str, Any]:
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

        compute_forces = self._needs_forces()

        # Fallback: populate the padded view if the caller didn't.
        if sharded.padded_batch is None:
            halo_exchange(sharded, self._halo_config, compute_forces=compute_forces)

        padded_batch = sharded.padded_batch
        meta = sharded.halo_meta

        # Flag a degenerate halo partition once, up front.
        self._check_partition_health(meta, padded_batch.positions.device)

        # Fallback: compute NL on the padded block if it isn't already there.
        if (
            getattr(padded_batch, "neighbor_matrix", None) is None
            and getattr(padded_batch, "neighbor_list", None) is None
        ):
            compute_neighbors(
                padded_batch, config=self._wrapper.model_config.neighbor_config
            )
        # Mark halo-receiver edges so the wrapper's ``(edge_index < n_atoms)``
        # filter drops them; see the helper's docstring for the rationale.
        _mark_halo_receiver_edges_as_padding(padded_batch, meta.n_owned)

        # Update per-step ctx state so the wrapper's ``adapt_input`` reads it.
        self._dist_ctx.policy = self._spec.distribution.policy
        self._dist_ctx.halo_meta = meta
        self._dist_ctx.halo_config = self._halo_config
        # Expose the persistent cap dict so a wrapper that pads inside its own
        # forward grows the same caps via current_dd_context().cap_state.
        self._dist_ctx.cap_state = self._cap_state

        # Fixed-shape padding (compile-only): pad to per-rank caps so the
        # compiled energy graph sees static atom/edge counts. Active only when
        # the model uses the energy-autograd force strategy and compile was
        # requested; eager instances skip padding entirely.
        _cp = self._spec.compile
        _dd_compile = bool(
            _cp is not None
            and _cp.forces_via_autograd
            and self._dd_compile_requested
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
                self._cap_state, "max_send", _ms_req,
                initial_factor=1.20, grow_factor=1.30, stride=16,
            )
            # The padded view is transient — only the compiled forward needs
            # fixed shapes. Stash the real-sized storage groups to restore after
            # the forward, since ``halo_exchange`` reuses ``padded_batch`` in
            # place and a cap-sized buffer would mismatch next step.
            _groups = padded_batch._storage.groups
            _orig_atoms = _groups.get("atoms")
            _orig_edges = _groups.get("edges")
            # Pad to the atom/edge caps from ``self._cap_state`` (grow-only).
            self._graph_padder.pad(padded_batch, self._cap_state)

        # Make the live per-step context ambient for the wrapper's forward, so
        # context-aware helpers and adapter bodies read it through
        # ``current_dd_context()``.
        if _dd_compile:
            # Compiled energy-autograd forward, framework-owned: the wrapper runs
            # energy-only on plain tensors with the halo routing threaded as
            # graph inputs; the framework consolidates per-node energy and takes
            # the force autograd.
            with activate_dd_context(self._dist_ctx):
                output = self._compiled_energy_autograd_forward(
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
                        _owned, meta, self._halo_config
                    )
            # Eager: promote ``positions`` (and other primary per-atom inputs)
            # to ShardTensors so custom ops see a ShardTensor input and the
            # per-layer halo correction fires.
            _promote_positions_to_shardtensor(
                padded_batch, self._spec, meta, self._halo_config,
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
            self._dist_ctx.graph_padder = _eager_padder
            # A wrapper that delegates its per-system energy reduction to the
            # framework (``spec.node_energy_key``) emits raw per-node energies
            # under that key; widen active outputs so the forward produces them,
            # then reduce owned-aware below. Restored in ``finally``.
            _nek = self._spec.node_energy_key
            _mc = self._wrapper.model_config
            _saved_active = None
            if _nek is not None and _nek not in _mc.active_outputs:
                _saved_active = _mc.active_outputs
                _mc.active_outputs = set(_saved_active) | {_nek}
            with activate_dd_context(self._dist_ctx):
                try:
                    output = self._wrapper(padded_batch)
                    if _eager_padder is not None:
                        output = _eager_padder.unpad(output)
                    if _nek is not None and _nek in output:
                        output = self._reduce_node_energy(
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
        owned_only = self._spec.owned_only_outputs
        if _dd_compile:
            owned_only = owned_only - self._wrapper.model_config.autograd_outputs
        result = consolidate_padded_outputs(
            output,
            model_config=self._wrapper.model_config,
            meta=meta,
            halo_config=self._halo_config,
            world_size=self._world_size,
            owned_only_outputs=owned_only,
            all_reduce_outputs=self._spec.all_reduce_outputs,
            output_kinds=self._spec.output_kinds,
        )
        if _pad_active:
            _groups = padded_batch._storage.groups
            if _orig_atoms is not None:
                _groups["atoms"] = _orig_atoms
            if _orig_edges is not None:
                _groups["edges"] = _orig_edges
        return result

    def _reduce_node_energy(
        self,
        output: dict[str, Any],
        node_energy_key: str,
        padded_batch: "Batch",
        num_graphs: int,
    ) -> dict[str, Any]:
        """Reduce a wrapper's per-node energy into the per-system ``"energy"``.

        Owned-slice + per-graph scatter + cross-rank all-reduce (autograd-aware,
        fp64-accumulated) via :func:`~nvalchemi.distributed.helpers.system_sum`.
        Pops ``node_energy_key`` and overrides ``"energy"`` so downstream
        consolidation sees the owned-aware total rather than the wrapper's plain
        sum (which double-counts ghosts). Must run inside an active DD context.
        """
        from nvalchemi.distributed._core.enums import Scope  # noqa: PLC0415
        from nvalchemi.distributed.helpers import system_sum, to_local  # noqa: PLC0415

        node_e = to_local(output.pop(node_energy_key))
        reduced = system_sum(
            node_e,
            to_local(padded_batch.batch_idx).to(torch.long),
            int(num_graphs),
            scope=Scope.OWNED,
        )
        ref = output.get("energy")
        if ref is not None:
            reduced = reduced.to(ref.dtype).reshape(ref.shape)
        output["energy"] = reduced
        return output

    # ------------------------------------------------------------------
    # Compiled energy-autograd path (framework-owned)
    # ------------------------------------------------------------------

    def _gp_replicate_compiled_region(self) -> Any:
        """Build (once, cached) the compiled energy-only region for the
        node-replicate path.

        Just the wrapper forward: the conv recombine is a mesh-static funcol
        all-reduce that traces inside the graph, and there is no per-step routing
        to thread (the node tensor is full and the all-reduce group is constant),
        so unlike the halo path no graph-input tensors are needed. Force autograd
        runs outside, over the owned-position leaf.
        """
        region = getattr(self, "_gp_region", None)
        if region is not None:
            return region
        wrapper = self._wrapper
        ck = dict(self._dd_compile_kwargs or {})
        backend = ck.pop("backend", "inductor")

        def _region(batch: Any) -> Any:
            return wrapper.forward(batch)

        self._gp_region = torch.compile(_region, backend=backend, **ck)
        return self._gp_region

    def _dd_compiled_region(self) -> Any:
        """Build (once, cached) the compiled energy-only region.

        The region publishes the halo routing — carried as tensor attributes on
        the batch — so the wrapper's per-layer halo-refresh adapters fire inside
        the traced graph, then runs the energy-only wrapper forward. The routing
        is read from the batch so Dynamo lifts it to graph inputs (it drifts per
        step and can't be baked); ``world_size`` is static and bakes in.
        """
        region = getattr(self, "_dd_region", None)
        if region is not None:
            return region

        from nvalchemi.distributed._core.compile_routing import (  # noqa: PLC0415
            clear_compile_routing,
            set_compile_routing,
        )

        wrapper = self._wrapper
        ck = dict(self._dd_compile_kwargs or {})
        backend = ck.pop("backend", "inductor")

        def _region(batch: Any) -> Any:
            si = getattr(batch, "_halo_si", None)
            if si is not None:
                set_compile_routing(
                    si,
                    batch._halo_rd,
                    batch._halo_rr,
                    batch._halo_no,
                    int(getattr(batch, "_halo_ws", 1)),
                )
            return wrapper.forward(batch)

        compiled = torch.compile(_region, backend=backend, **ck)

        def runner(batch: Any) -> Any:
            # Clear the holder after each call so a later eager refresh never
            # reads trace-time (fake / stale) routing.
            try:
                return compiled(batch)
            finally:
                clear_compile_routing()

        self._dd_region = runner
        return runner

    def _compiled_energy_autograd_forward(
        self, padded_batch: "Batch", meta: Any, n_graphs: int
    ) -> dict[str, Any]:
        """Compiled energy + autograd-force forward.

        For a model using ``forces_via_autograd``, the framework owns the whole
        compile path so the wrapper carries none of it: make ``positions`` a
        fresh leaf (autograd boundary is outside compile); thread the halo
        routing as graph-input batch attributes; run the wrapper energy-only
        through the cached compiled region; consolidate per-node energy (owned
        per-graph sum + cross-rank all-reduce); take
        ``forces = -d(energy)/d(positions)``. The returned ``{energy, forces}``
        feeds the shared ``consolidate_padded_outputs`` like the eager output.
        """
        from nvalchemi.distributed._core.particle_halo import (  # noqa: PLC0415
            build_halo_meta_tensors,
        )
        from nvalchemi.distributed.compile_bridge import (  # noqa: PLC0415
            _consolidate_node_energy,
        )

        atoms = padded_batch._atoms_group
        pos = atoms["positions"]
        pos_plain = pos.to_local() if hasattr(pos, "to_local") else pos
        # Fresh leaf so autograd.grad (outside compile) differentiates the
        # compiled output w.r.t. it.
        pos_plain = pos_plain.detach().requires_grad_(True)
        atoms["positions"] = pos_plain

        # Fixed-shape halo routing as graph inputs, attached to the batch so
        # Dynamo lifts them (they drift per step). ``max_send`` is the persistent
        # per-rank cap (grown in lockstep across ranks above).
        n_padded = int(pos_plain.shape[0])
        max_send = self._cap_state.get("max_send") or max(
            (max(r) for r in meta.send_sizes), default=0
        )
        ws = len(meta.send_sizes)
        si, rd, rr, no = build_halo_meta_tensors(
            meta, self._halo_config.rank, max_send, n_padded, pos_plain.device
        )
        for key, val in (
            ("_halo_si", si),
            ("_halo_rd", rd),
            ("_halo_rr", rr),
            ("_halo_no", no),
            ("_halo_ws", ws),
        ):
            object.__setattr__(padded_batch, key, val)

        # The model declares how its energy-only forward yields a global energy:
        # per-node ``atomic_energies`` (framework consolidates) or an already
        # self-consolidated global ``energy``.
        _cp = self._spec.compile
        energy_key = _cp.energy_output
        consolidate = _cp.consolidate_node_energy

        # Run energy-only through the compiled region, restoring the wrapper's
        # active_outputs afterward.
        mc = self._wrapper.model_config
        saved_active = mc.active_outputs
        mc.active_outputs = {energy_key}
        try:
            out = self._dd_compiled_region()(padded_batch)
        finally:
            mc.active_outputs = saved_active
        e = out[energy_key]

        if consolidate:
            # Per-node energy: owned-only per-graph sum + cross-rank all-reduce.
            energy = _consolidate_node_energy(
                e, padded_batch.batch_idx.long(), int(n_graphs)
            )
        else:
            # Model self-consolidated the global per-system energy already.
            energy = e
        (grad,) = torch.autograd.grad(
            [energy],
            [pos_plain],
            grad_outputs=[torch.ones_like(energy)],
            create_graph=False,
            retain_graph=False,
            allow_unused=True,
        )
        forces = torch.zeros_like(pos_plain) if grad is None else -grad
        return self._wrapper.adapt_output(
            {"energy": energy, "forces": forces}, padded_batch
        )
