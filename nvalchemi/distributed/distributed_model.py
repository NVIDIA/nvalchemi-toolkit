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
from nvalchemi.distributed.partitioner import SpatialPartitioner
from nvalchemi.neighbors import compute_neighbors

if TYPE_CHECKING:
    from nvalchemi.distributed.sharded_batch import ShardedBatch
    from nvalchemi.models.base import BaseModelMixin


__all__ = ["DistributedModel", "DistributionError"]


def isolate_compile_cache_per_rank() -> None:
    """Give each rank its own ``torch.compile`` on-disk cache dir (multi-rank DD).

    The inductor FxGraphCache + the AOTAutograd cache default to one shared dir;
    under multi-rank DD a rank can deserialize another rank's guarded entry and
    raise a ``KeyError`` mid-forward → a skipped collective → NCCL deadlock.
    Pointing each rank at its own dir removes the collision while keeping the
    caches ON (disabling them instead re-lowers the AOT graph every step).

    Idempotent and launcher-friendly: only sets a var that is currently unset (so a
    launcher/user setting wins), keys off ``LOCAL_RANK`` (torchrun) / ``RANK``, and
    is a no-op single-process. These vars are read when inductor/triton actually
    lower a graph — i.e. at the *first forward*, not when ``torch.compile`` merely
    wraps the model — so calling this at ``DistributedModel`` construction (via
    :func:`_configure_dd_dynamo`) reliably lands before the first DD forward. That
    covers the common ``from_checkpoint(compile_model=True)`` -> ``DistributedModel``
    order: the loader only wraps the model (lazily) and runs no forward, so nothing
    is lowered until the DD forward, by which point the dirs are set. The one gap is
    a forward triggered *before* construction (e.g. a manual sanity-check call); for
    that, a launcher exporting these from ``LOCAL_RANK`` at process start is the
    bulletproof path.
    """
    import os  # noqa: PLC0415

    rank = os.environ.get("LOCAL_RANK") or os.environ.get("RANK")
    if rank is None:
        return
    import tempfile  # noqa: PLC0415

    root = os.path.join(tempfile.gettempdir(), "nvalchemi_dd_compile_cache")
    for _var, _sub in (
        ("TORCHINDUCTOR_CACHE_DIR", "inductor"),
        ("TRITON_CACHE_DIR", "triton"),
    ):
        os.environ.setdefault(_var, os.path.join(root, f"{_sub}_rank{rank}"))


def _configure_dd_dynamo() -> None:
    """Tune Dynamo/inductor for a distributed forward.

    Safe (and correct) to call whenever a model runs under a multi-rank DD scope —
    compiled by the framework OR pre-compiled by its own loader (e.g. MACE
    ``loader.compile``), since the hazards below are triggered by the *wrapped*
    model recompiling under DD, independent of who invoked ``torch.compile``.

    1. **Recompile ceiling.** The fixed-shape caps grow during warmup
       (``max_send`` / ``n_cap`` / ``e_cap`` each bump shapes a few times before
       settling), so more than the default 8 recompiles are expected. If a rank
       hits the limit and stops recompiling mid-warmup it diverges from its peer
       and the halo all-to-all deadlocks (NCCL watchdog timeout). torch>=2.6
       renamed ``cache_size_limit`` -> ``recompile_limit`` (plus the accumulated
       twin); set whichever names exist so the ceiling actually takes effect.
    2. **Per-rank on-disk compile caches.** The inductor FxGraphCache + the
       (separate) AOTAutograd cache key on the *local* rank's graph but default to
       ONE shared dir, so under multi-rank DD a rank can deserialize another rank's
       guarded entry and hit a ``KeyError`` (e.g. a cueq segment ``lengths`` dim) →
       skip a collective → NCCL deadlock. The fix is to point each rank at its OWN
       cache dir (keeping the caches ON — disabling them re-lowers the AOT graph
       every step). This calls :func:`isolate_compile_cache_per_rank`, a no-op if the
       dirs are already set. Since those dirs are read at first-forward lowering (not
       when ``torch.compile`` wraps the model) and this runs at ``DistributedModel``
       construction, it lands before the first DD forward — including the common
       ``from_checkpoint(compile_model=True)`` -> ``DistributedModel`` order, where
       the loader only wraps (lazily) and runs no forward. A launcher exporting the
       dirs from ``LOCAL_RANK`` at process start still covers the edge case of a
       forward triggered before construction.
    """
    import torch._dynamo as _td  # noqa: PLC0415

    for _attr, _val in (
        ("recompile_limit", 64),
        ("cache_size_limit", 64),
        ("accumulated_recompile_limit", 512),
        ("accumulated_cache_size_limit", 512),
    ):
        if hasattr(_td.config, _attr):
            setattr(_td.config, _attr, max(getattr(_td.config, _attr), _val))
    _td.config.force_parameter_static_shapes = False

    isolate_compile_cache_per_rank()


def _prepare_dd_compile(spec: "Any", compile_kwargs: "dict | None") -> dict:
    """Validate the spec supports a compiled distributed forward and return the
    resolved ``torch.compile`` kwargs.

    Distributed compile is fixed-shape (graphs are padded to per-rank caps), so
    ``dynamic`` defaults to ``False``. Dynamo tuning is applied separately in
    :func:`_configure_dd_dynamo` (unconditionally at scope setup). Raises if the
    spec declares no ``CompilePolicy``.
    """
    cp = getattr(spec, "compile", None)
    if cp is None:
        raise DistributionError(
            "compile=True requires the model's distribution_spec to declare a "
            "CompilePolicy (force_strategy). This model does not support a "
            "compiled distributed forward."
        )
    import os  # noqa: PLC0415

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


def _wrapper_is_precompiled(wrapper: "Any") -> bool:
    """True if the wrapper already holds a ``torch.compile``-d module.

    ``torch.compile`` returns an ``OptimizedModule`` (carrying ``_orig_mod``); a
    wrapper that compiled itself in its loader (e.g. MACE
    ``from_checkpoint(compile_model=True)``) holds one in its module tree.
    Walking the tree keeps the check model-agnostic. This matters because running
    the *eager* distributed path against such a model is silently wrong: the halo
    correction (eager per-op handlers, or the compile-refresh adapters) never sees
    the message-passing ops sealed inside the loader's compiled graph, so every
    rank returns an uncorrected owned-only forward.
    """
    modules = getattr(wrapper, "modules", None)
    if not callable(modules):
        return False
    try:
        from torch._dynamo.eval_frame import OptimizedModule  # noqa: PLC0415
    except Exception:  # pragma: no cover - torch internal layout changed
        OptimizedModule = None
    for m in modules():
        if OptimizedModule is not None and isinstance(m, OptimizedModule):
            return True
        if type(m).__name__ == "OptimizedModule":
            return True
    return False


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


def _resolve_partition_health(
    any_empty: bool,
    any_trivial: bool,
    n_global: int,
    *,
    world_size: int,
    require_nondegenerate: bool,
    rank: int,
) -> None:
    """Act on a collective partition-health verdict, identically on every rank.

    * ``any_empty`` (a rank owns 0 atoms) is always fatal — the geometry can't
      fill this many ranks.
    * ``any_trivial`` (some rank's halo covers all atoms, 0 remote) means DD
      isn't exercised: fatal when ``require_nondegenerate`` (a force-equivalence
      check there proves nothing), otherwise a rank-0 warning.

    Pure (no collectives) so the empty/trivial branches are unit-testable on
    CPU; the verdict flags are already reduced across the mesh by the caller, so
    every rank passes the same values and raises identically (no desync)."""
    if any_empty:
        raise RuntimeError(
            "Degenerate domain decomposition: a rank was assigned 0 owned "
            f"atoms (world_size={world_size}, total atoms={n_global}). There "
            "are more ranks than this geometry can partition — use fewer ranks "
            "or a larger system."
        )
    if not any_trivial:
        return
    msg = (
        "Degenerate (trivial) domain decomposition: every rank's halo already "
        f"covers all {n_global} atoms (0 remote atoms), so domain parallelism "
        "gains nothing here — each rank does the full system's work. This "
        "happens when box/ranks <= ~2*ghost_width (ghost_width = cutoff + "
        "skin); use fewer ranks or a larger system to actually decompose."
    )
    # Opt-in strict mode (tests, guaranteed-decomposed runs): a trivial
    # partition can't validate the halo path, so fail loud.
    if require_nondegenerate:
        raise RuntimeError(
            msg + " (require_nondegenerate=True — refusing to run a partition "
            "that doesn't exercise the halo boundary.)"
        )
    if rank == 0:
        from loguru import logger  # noqa: PLC0415

        logger.warning(msg + " (results are still correct.)")


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
    meta: "Any",
    config: "Any",
    device: "Any",
    n_pad: int,
    max_send_cap: "int | None" = None,
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


def _reduce_scatter_owned(
    full: torch.Tensor,
    counts: "list[int]",
    rank: int,
    nlo: int,
    nhi: int,
    grp: "Any",
) -> torch.Tensor:
    """Sum a full ``[N, *]`` tensor across ranks and return this rank's owned,
    rank-contiguous block ``[counts[rank], *]``.

    Node-partition GP replicates the full node set, so each rank produces a full
    ``[N, *]`` partial that must be summed then sliced to owned. An even
    reduce-scatter — each rank-block padded to ``max(counts)`` so the chunks are
    uniform — lands only this rank's owned slice and moves ~half the cross-rank
    volume of ``all_reduce([N, *])`` + slice. Falls back to a local slice with no
    process group (single rank).
    """
    if grp is None:
        return full[nlo:nhi].contiguous()

    import torch.distributed as dist  # noqa: PLC0415

    world = len(counts)
    mc = max(counts)
    tail = tuple(full.shape[1:])
    buf = full.new_zeros((world, mc, *tail))
    off = 0
    for r in range(world):
        c = counts[r]
        if c:
            buf[r, :c] = full[off : off + c]
        off += c
    buf = buf.reshape(world * mc, *tail)
    owned = full.new_empty((mc, *tail))
    dist.reduce_scatter_tensor(owned, buf, op=dist.ReduceOp.SUM, group=grp)
    return owned[: counts[rank]].contiguous()


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

        # Explicit ``spec=`` wins, else ask the wrapper for the spec matching the
        # config-selected strategy (the spec is a joint model×strategy product).
        if spec is None:
            _ds = getattr(wrapper, "distribution_spec", None)
            spec = (
                _ds(getattr(domain_config, "strategy", None)) if callable(_ds) else _ds
            )
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
        # A model may arrive already torch.compiled by its own loader (e.g.
        # ``MACEWrapper.from_checkpoint(compile_model=True)``) in front of a
        # *plain* DistributedModel. The eager DD path is silently WRONG for such
        # a model — the halo correction never sees the message-passing ops sealed
        # inside the compiled graph, so each rank returns an uncorrected owned-only
        # forward. When the model uses a framework-owned energy-autograd force
        # strategy, engage the compiled DD path (it consumes a pre-compiled inner
        # model correctly); if the spec declares no compiled forward at all, raise
        # rather than return garbage. Models that keep force autograd inside the
        # model (``MODEL_INTERNAL``, e.g. UMA's fairchem-owned internal compile)
        # run the eager path correctly and are left untouched.
        if not self._dd_compile_requested and _wrapper_is_precompiled(wrapper):
            _cp_pre = getattr(self._spec, "compile", None)
            if _cp_pre is not None and _cp_pre.forces_via_autograd:
                self._dd_compile_requested = True
                self._dd_compile_kwargs = _prepare_dd_compile(
                    self._spec, compile_kwargs
                )
            elif _cp_pre is None:
                raise DistributionError(
                    f"{type(wrapper).__name__}'s model is already torch.compiled "
                    "(e.g. from_checkpoint(compile_model=True)), but its "
                    "distribution_spec declares no CompilePolicy, so a correct "
                    "compiled distributed forward cannot be built. Either build "
                    "the wrapper WITHOUT compiling it and pass "
                    "DistributedModel(..., compile=True), or add a CompilePolicy "
                    "to its distribution_spec."
                )
        # Tune Dynamo/inductor for DD unconditionally: the wrapped model may be
        # compiled by its own loader (e.g. MACE ``loader.compile``) rather than the
        # framework, in which case ``compile`` above is False yet the model still
        # recompiles under DD and needs the raised ceiling + cross-rank-safe caches
        # (see :func:`_configure_dd_dynamo`). A no-op when nothing compiles.
        _configure_dd_dynamo()
        # Fixed-shape graph padder for the compiled halo path. A model may
        # declare a custom padder via its CompilePolicy; the default is the
        # generic COO ``edge_index`` padder, so a standard MPNN declares nothing.
        from nvalchemi.distributed.graph_padder import COOPadder  # noqa: PLC0415

        _compile_policy = getattr(self._spec, "compile", None)
        self._graph_padder = (
            getattr(_compile_policy, "graph_padder", None) or COOPadder()
        )
        self._setup_called = False
        # DistributedModel is single-lifecycle: once ``close()`` has torn down the
        # process-wide adapter state, re-entering ``with model:`` won't re-install
        # it, so a second use is rejected (construct a fresh instance instead).
        self._closed = False
        # The parallelization strategy owning this model's distributed forward;
        # built lazily from the resolved storage policy (see ``_strategy``).
        self._strategy_obj: Any = None
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

        # The parallelization strategy owns its distributed forward; this model
        # is the shared forward toolkit it drives. A new strategy plugs in as a
        # new class, without a framework type-switch here.
        return self._strategy().run_forward(self, sharded, wired_fields)

    def _strategy(self) -> Any:
        """The :class:`ParallelizationStrategy` for this model's storage policy
        (built once, cached)."""
        if self._strategy_obj is None:
            from nvalchemi.distributed.strategy import (  # noqa: PLC0415
                strategy_for_policy,
            )

            mesh = self._config.mesh
            rank = mesh.get_local_rank() if mesh is not None else 0
            self._strategy_obj = strategy_for_policy(
                self._spec.distribution.policy, self._config, rank
            )
        return self._strategy_obj

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
        self._closed = True

    def __enter__(self) -> "DistributedModel":
        # Single-lifecycle: setup installs process-wide adapter state that
        # ``close()`` restores, and re-entry would not re-install it — fail loudly
        # rather than run half-set-up.
        if self._closed:
            raise RuntimeError(
                "DistributedModel is single-use; construct a new one after close()"
            )
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
                    isinstance(a, JitAdapter)
                    and getattr(a, "mode", "eager") == "marshal"
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
        rank = (
            self._config.mesh.get_local_rank() if self._config.mesh is not None else 0
        )
        _resolve_partition_health(
            any_empty,
            any_trivial,
            n_global,
            world_size=self._world_size,
            require_nondegenerate=getattr(self._config, "require_nondegenerate", False),
            rank=rank,
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

    def _graph_parallel_owned_nbmat(
        self, sharded: "ShardedBatch", meta: Any, rank: int
    ) -> "dict[str, torch.Tensor]":
        """This rank's owned-receiver dense neighbour matrix for the GP path.

        The dense analogue of :meth:`_graph_parallel_owned_edges`. Materializes
        the full dense ``neighbor_matrix`` once from the replicated geometry, then
        keeps only the rows whose receiver atom this rank owns. Sender columns stay
        global ids into the all-gathered node set (``refresh_neighbors(positions)``
        in the wrapper); receiver rows are this rank's owned atoms in owned-local
        order. Non-differentiable routing built under ``no_grad`` — geometry
        differentiates through ``refresh_neighbors``.

        Returns a ``node_properties`` dict (``neighbor_matrix`` / ``num_neighbors``
        / optionally ``neighbor_matrix_shifts``) to hand to
        :meth:`ShardedBatch.local_batch_with_edges`.
        """
        with torch.no_grad():
            global_batch = sharded.to_global_batch()
            compute_neighbors(
                global_batch, config=self._wrapper.model_config.neighbor_config
            )
        # Owned receiver rows, in owned-local order. Post-scatter the sharded
        # atoms are in rank-contiguous order, so the boolean mask selects this
        # rank's block in local_index order (row i = owned-local atom i).
        owned = meta.owner_rank.to(global_batch.neighbor_matrix.device) == rank
        props: dict[str, torch.Tensor] = {
            "neighbor_matrix": global_batch.neighbor_matrix[owned].to(torch.long),
            "num_neighbors": global_batch.num_neighbors[owned].to(torch.long),
        }
        shifts = getattr(global_batch, "neighbor_matrix_shifts", None)
        if shifts is not None:
            props["neighbor_matrix_shifts"] = shifts[owned]
        return props

    def _graph_parallel_dense_full_autograd(
        self, sharded: "ShardedBatch"
    ) -> dict[str, Any]:
        """Node-partition GP for dense-``neighbor_matrix`` models whose kernel
        indexes the position array (``gp_replicate_geometry``; e.g. PME's fused
        real-space+reciprocal kernel).

        The full geometry is replicated on every rank so the kernel can index
        global senders and spread the full charge set (correct reciprocal). The
        dense ``neighbor_matrix`` is masked to this rank's owned receivers
        (``num_neighbors[non-owned] = 0``), so the **real-space** work partitions
        while the **reciprocal** reads all charges (replicated — correct, not yet
        compute-partitioned). Energy is the framework's owned-aware sum of the
        per-node ``node_energy_key`` output; forces come from autograd of that
        owned energy over the full-position leaf, cross-rank ``SUM``, sliced to
        owned — the same adjoint as :meth:`_graph_parallel_internal`, but the
        framework (not the model) owns the force autograd.
        """
        from types import SimpleNamespace  # noqa: PLC0415

        import torch.distributed as dist  # noqa: PLC0415

        from nvalchemi.distributed._core.context import (
            activate_dd_context,  # noqa: PLC0415
        )
        from nvalchemi.distributed._core.gather_primitives import (  # noqa: PLC0415
            mesh_group,
        )
        from nvalchemi.distributed._core.placement import ShardRouting  # noqa: PLC0415
        from nvalchemi.distributed.output_consolidation import (  # noqa: PLC0415
            consolidate_sharded_outputs,
        )

        mesh = self._config.mesh
        rank = mesh.get_local_rank() if mesh is not None else 0
        world = self._world_size or 1

        # Full node set on every rank; positions are a fresh autograd leaf.
        full = sharded.to_global_batch()
        atoms = full._atoms_group
        pos = atoms["positions"].detach().requires_grad_(True)
        atoms["positions"] = pos

        assignment = sharded.rank_assignment.to(pos.device)
        counts_t = torch.bincount(assignment, minlength=world)
        counts = [int(c) for c in counts_t.tolist()]
        nlo = int(counts_t[:rank].sum().item())
        nhi = nlo + counts[rank]
        owned_mask = assignment == rank
        meta = ShardRouting.from_assignment(assignment, rank, world)
        meta.n_systems_global = sharded.num_graphs

        # Dense neighbours over the full geometry, masked to owned receivers so the
        # kernel's real-space loop does no work for non-owned rows (their energy is
        # dropped by the owned-aware sum anyway). The reciprocal reads full charges.
        from nvalchemi.neighbors import compute_neighbors  # noqa: PLC0415

        compute_neighbors(full, config=self._wrapper.model_config.neighbor_config)
        num = full._atoms_group.get("num_neighbors")
        if num is not None:
            num = num.clone()
            num[~owned_mask] = 0
            full._atoms_group["num_neighbors"] = num

        self._dist_ctx.policy = self._spec.distribution.policy
        self._dist_ctx.gather_meta = meta
        self._dist_ctx.halo_meta = None

        # Run energy-only: the framework owns the force autograd, so the wrapper
        # must not consume/free the energy graph with its own force head. Widen
        # active outputs to include the per-node energy key.
        nek = self._spec.node_energy_key
        _mc = self._wrapper.model_config
        _saved_active = _mc.active_outputs
        _mc.active_outputs = {"energy"} | ({nek} if nek else set())
        try:
            with activate_dd_context(self._dist_ctx):
                output = self._wrapper(full)
        finally:
            _mc.active_outputs = _saved_active

        # Owned-aware per-system energy from the per-node key (each atom counted
        # once by its owner), then a plain cross-rank SUM for the global energy.
        node_e = output[nek]
        batch_idx = full.batch_idx.long()
        e_partial = torch.zeros(
            sharded.num_graphs, dtype=node_e.dtype, device=node_e.device
        ).index_add(0, batch_idx[owned_mask], node_e[owned_mask])

        grp = (
            mesh_group(mesh)
            if (dist.is_initialized() and world > 1 and mesh is not None)
            else None
        )
        out: dict[str, Any] = {}
        # Energy is a tiny ``[n_systems]`` reduction (latency-bound); launch it
        # async so it overlaps the force autograd + reduce-scatter below.
        e_global = e_partial.detach().clone()
        e_handle = (
            dist.all_reduce(e_global, op=dist.ReduceOp.SUM, group=grp, async_op=True)
            if grp is not None
            else None
        )
        if self._needs_forces():
            (grad,) = torch.autograd.grad(
                [e_partial.sum()], [pos], create_graph=False, allow_unused=True
            )
            f = torch.zeros_like(pos) if grad is None else -grad
            out["forces"] = _reduce_scatter_owned(f, counts, rank, nlo, nhi, grp)
        if e_handle is not None:
            e_handle.wait()
        out["energy"] = e_global

        self._dist_ctx.gather_meta = None
        return consolidate_sharded_outputs(
            output=out,
            model_config=self._wrapper.model_config,
            world_size=self._world_size,
            owned_only_outputs=frozenset({"energy", "forces"}),
            all_reduce_outputs=frozenset(),
            halo_config=SimpleNamespace(mesh=mesh),
        )

    def _graph_parallel_internal(self, sharded: "ShardedBatch") -> dict[str, Any]:
        """Node-partition graph-parallel for models that compute forces internally.

        Each rank owns a balanced index slice of the atoms. The full geometry is
        replicated so the model's internal (otf) graph build can index global
        senders, but a declared adapter (the wrapper's ``_generate_graph``)
        restricts the node-wise work to this rank's owned slice and the per-layer
        node-feature all-gather (``refresh_neighbors`` → the policy's replicate;
        reduce-scatter on the backward) feeds the convolution its global sources.

        The model computes its own per-system energy (an owned partial, via its
        declared ``LOCAL``-scope reduction) and its own forces
        (``-dE_owned/d pos`` over the *full* positions). Because the feature
        all-gather's reduce-scatter backward routes each node's feature gradient
        to its owner exactly once, a plain cross-rank ``SUM`` of the per-rank
        force — with **no** ``/world_size`` — recovers the global force; it is
        then sliced to this rank's owned atoms. The energy partials likewise sum
        to the global energy. The complement of :meth:`_call_graph_parallel`'s
        framework-autograd path, for opaque force heads (e.g. UMA).
        """
        from types import SimpleNamespace  # noqa: PLC0415

        import torch.distributed as dist  # noqa: PLC0415

        from nvalchemi.distributed._core.gather_primitives import (  # noqa: PLC0415
            mesh_group,
        )
        from nvalchemi.distributed._core.placement import (  # noqa: PLC0415
            ShardRouting,
        )
        from nvalchemi.distributed.output_consolidation import (  # noqa: PLC0415
            consolidate_sharded_outputs,
        )

        mesh = self._config.mesh
        rank = mesh.get_local_rank() if mesh is not None else 0
        world = self._world_size or 1

        import os as _os  # noqa: PLC0415
        import time as _time  # noqa: PLC0415

        _prof = _os.environ.get("NVALCHEMI_DD_PROFILE") and rank == 0
        _marks: list = []

        def _mark(label: str) -> None:
            if _prof:
                torch.cuda.synchronize()
                _marks.append((label, _time.perf_counter()))

        _mark("start")
        # Full node set on every rank; positions become a fresh autograd leaf for
        # the model's internal force autograd. Mutate the gathered batch in place
        # rather than reconstructing AtomicData/Batch — the rebuild (pydantic
        # validation + collation) was the dominant per-forward DD overhead and is
        # redundant: ``to_global_batch`` already returns a complete batch.
        full = sharded.to_global_batch()
        _mark("to_global_batch")
        atoms = full._atoms_group
        pos = atoms["positions"].detach().requires_grad_(True)
        atoms["positions"] = pos
        batch_r = full
        _mark("rebuild_batch")

        # Owned partition = the ShardState's own split (the strategy is the single
        # owner of the layout). ``to_global_batch`` ordered the full node set by
        # rank, so this rank's owned atoms are exactly the contiguous block where
        # ``rank_assignment == rank``. Deriving the split here from a freshly
        # recomputed *balanced* formula instead would disagree with the scattered
        # owned batch whenever the atom count doesn't divide evenly across ranks,
        # mis-slicing the per-atom outputs (owned rows) relative to the integrator
        # batch. Reading it from the ShardState keeps forward-output, local_view,
        # and dynamics batch on one split by construction.
        n_atoms = pos.shape[0]
        assignment = sharded.rank_assignment.to(pos.device)
        counts_t = torch.bincount(assignment, minlength=world)
        counts = [int(c) for c in counts_t.tolist()]
        nlo = int(counts_t[:rank].sum().item())
        nhi = nlo + counts[rank]
        meta = ShardRouting.from_assignment(assignment, rank, world)
        meta.n_systems_global = sharded.num_graphs

        self._dist_ctx.policy = self._spec.distribution.policy
        self._dist_ctx.gather_meta = meta
        self._dist_ctx.halo_meta = None
        self._dist_ctx.owned_offset = 0
        # Publish the fixed-shape graph padder (declared on ``CompilePolicy``, the
        # same one halo uses) so the wrapper's ``maybe_pad_graph`` precomputes an
        # edge-capped, ``otf_graph=False`` graph — the strategy's ``cap_atoms=False``
        # (set in ``run_forward``) makes it edge-only, restricting the edges to this
        # rank's owned receivers. Static per-rank edge shapes ⇒ no recompile churn.
        self._dist_ctx.cap_state = self._cap_state
        _cp = self._spec.compile
        _padder = (
            _cp.graph_padder
            if (_cp is not None and _cp.graph_padder is not None)
            else None
        )
        self._dist_ctx.graph_padder = _padder
        _mark("meta_setup")

        # Publish the static node-partition all-gather routing so the per-layer
        # ``refresh_neighbors`` inside the model's compiled forward uses the
        # fullgraph-traceable fixed gather (fetch every node from its owner). The
        # routing is index-based and constant across MD steps, so it is read as
        # trace-time constants without recompiling. Eager forwards ignore it
        # (``refresh_neighbors`` gates the fixed gather on ``is_compiling``).
        from nvalchemi.distributed._core.compile_routing import (  # noqa: PLC0415
            clear_gp_compile_routing,
            set_gp_compile_routing,
        )

        gi = torch.arange(n_atoms, device=pos.device)
        set_gp_compile_routing(
            gi, meta.owner_rank, meta.local_index, max(counts), world, mesh
        )
        try:
            with activate_dd_context(self._dist_ctx):
                output = self._wrapper(batch_r)
            # No dead-atom rows under node partition (edge-only caps), so unpad is
            # a no-op on the per-atom outputs; kept for symmetry with the halo path.
            if _padder is not None:
                output = _padder.unpad(output)
        finally:
            clear_gp_compile_routing()
            # Restore the backbone's ``otf_graph`` flag the padder flipped, even on
            # a forward error, so the next step isn't stuck on the fixed-shape path.
            if _padder is not None:
                _padder.restore()
        _mark("wrapper_forward")

        grp = (
            mesh_group(mesh)
            if (dist.is_initialized() and world > 1 and mesh is not None)
            else None
        )
        # Energy: each rank holds its owned per-system partial → global SUM. It is
        # a tiny ``[n_systems]`` reduction (latency-bound); launch it async so it
        # overlaps the (larger) force reduce-scatter below.
        e_handle = None
        if "energy" in output and isinstance(output["energy"], torch.Tensor):
            e = output["energy"]
            if grp is not None:
                e = e.clone()
                e_handle = dist.all_reduce(
                    e, op=dist.ReduceOp.SUM, group=grp, async_op=True
                )
            output["energy"] = e
        # Forces: the model returns ``-dE_owned/d pos`` over the full positions.
        # The feature all-gather's reduce-scatter backward already routed each
        # node's gradient to its owner once, so a plain SUM (no ``/world``) is
        # the global force. Reduce-scatter over the rank-contiguous owned blocks
        # lands only this rank's owned slice — half the cross-rank volume of
        # all-reduce + slice (consolidation gathers it back to global order).
        if "forces" in output and isinstance(output["forces"], torch.Tensor):
            output["forces"] = _reduce_scatter_owned(
                output["forces"], counts, rank, nlo, nhi, grp
            )
        if e_handle is not None:
            e_handle.wait()

        self._dist_ctx.gather_meta = None
        self._dist_ctx.owned_offset = 0
        _mark("reduce_outputs")

        out = consolidate_sharded_outputs(
            output,
            model_config=self._wrapper.model_config,
            world_size=self._world_size,
            owned_only_outputs=frozenset({"energy", "forces"}),
            all_reduce_outputs=frozenset(),
            halo_config=SimpleNamespace(mesh=mesh),
        )
        _mark("consolidate")
        if _prof:
            segs = ", ".join(
                f"{_marks[i][0]}={1000 * (_marks[i][1] - _marks[i - 1][1]):.1f}"
                for i in range(1, len(_marks))
            )
            total = 1000 * (_marks[-1][1] - _marks[0][1])
            print(f"[dd-prof] total={total:.1f}ms | {segs}", flush=True)
        return out

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

    def _reduce_node_virial(
        self,
        output: dict[str, Any],
        node_virial_key: str,
        padded_batch: "Batch",
        num_graphs: int,
    ) -> dict[str, Any]:
        """Reduce a wrapper's per-node virial into the per-system ``"stress"``.

        Analytic-kernel-virial wrappers (LJ, DFTD3) return a per-system virial
        summed over each rank's all-local (owned + ghost) atoms, which is wrong
        under decomposition and can't be owned-masked once collapsed. They
        instead emit the per-atom virial ``(n_nodes, 3, 3)`` (energy units) under
        ``node_virial_key``; this owned-slices + all-reduces it (each pair counted
        once by its owning atom, mirroring ``atomic_energies``), converts to the
        tensile-positive Cauchy stress ``-W/V`` using the cell volume, and
        overrides the wrapper's all-local ``"stress"``. Must run inside an active
        DD context.
        """
        from nvalchemi.distributed._core.enums import Scope  # noqa: PLC0415
        from nvalchemi.distributed.helpers import system_sum, to_local  # noqa: PLC0415

        node_v = to_local(output.pop(node_virial_key))
        virial = system_sum(
            node_v,
            to_local(padded_batch.batch_idx).to(torch.long),
            int(num_graphs),
            scope=Scope.OWNED,
        )  # (n_systems, 3, 3), replicated
        cell = to_local(padded_batch.cell)
        volume = torch.det(cell).abs().view(-1, 1, 1)
        stress = -virial / volume
        ref = output.get("stress")
        if ref is not None:
            stress = stress.to(ref.dtype).reshape(ref.shape)
        output["stress"] = stress
        return output

    # ------------------------------------------------------------------
    # Compiled energy-autograd path (framework-owned)
    # ------------------------------------------------------------------

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

        # Stress via the strain trick: perturb positions AND cell by a symmetric
        # per-system strain leaf, then virial = d(energy)/d(strain). Because we
        # differentiate the framework's already-consolidated GLOBAL energy, this is
        # correct for every force strategy (real + reciprocal spaces alike), filling
        # the compiled-DD stress the energy-autograd path otherwise omits. The
        # per-rank virial is summed across ranks by consolidation (stress declared
        # ALL_REDUCE), exactly like the autograd forces. Gated on stress being
        # requested. Strain application is OUTSIDE the compiled region (like the
        # positions leaf), so it adds no graph ops.
        want_stress = "stress" in self._wrapper.model_config.active_outputs and bool(
            getattr(self._spec.compile, "stress_via_strain", False)
        )
        strain = None
        cell_orig = cell_local = None
        if want_stress:
            _bidx = padded_batch.batch_idx
            _bidx = (_bidx.to_local() if hasattr(_bidx, "to_local") else _bidx).long()
            strain = torch.zeros(
                int(n_graphs), 3, 3, dtype=pos_plain.dtype, device=pos_plain.device
            ).requires_grad_(True)
            strain_sym = 0.5 * (strain + strain.transpose(-1, -2))
            strain_atom = strain_sym.index_select(0, _bidx)  # [N, 3, 3]
            pos_use = pos_plain + torch.einsum("nij,nj->ni", strain_atom, pos_plain)
            cell_orig = getattr(padded_batch, "cell", None)
            if cell_orig is not None:
                cell_local = (
                    cell_orig.to_local()
                    if hasattr(cell_orig, "to_local")
                    else cell_orig
                )
                cell_use = cell_local + torch.einsum(
                    "bij,bjk->bik", cell_local, strain_sym
                )
                object.__setattr__(padded_batch, "cell", cell_use)
            atoms["positions"] = pos_use
        else:
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
        grad_inputs = [pos_plain] if not want_stress else [pos_plain, strain]
        grads = torch.autograd.grad(
            [energy],
            grad_inputs,
            grad_outputs=[torch.ones_like(energy)],
            create_graph=False,
            retain_graph=False,
            allow_unused=True,
        )
        grad = grads[0]
        forces = torch.zeros_like(pos_plain) if grad is None else -grad
        result: dict[str, Any] = {"energy": energy, "forces": forces}
        if want_stress:
            virial = grads[1]  # d(energy)/d(strain): this rank's partial virial
            if virial is None or cell_local is None:
                result["stress"] = torch.zeros(
                    int(n_graphs), 3, 3, dtype=pos_plain.dtype, device=pos_plain.device
                )
            else:
                # sigma = (1/V) dE/d(strain) with the strain applied as
                # r->r(I+eps), cell->cell(I+eps) (matches the analytic wrappers'
                # tensile-positive Cauchy stress). Consolidation sums the per-rank
                # virial across ranks (stress declared ALL_REDUCE).
                vol = torch.det(cell_local).abs().reshape(-1, 1, 1)
                result["stress"] = virial / vol
            if cell_orig is not None:
                object.__setattr__(padded_batch, "cell", cell_orig)
        return self._wrapper.adapt_output(result, padded_batch)
