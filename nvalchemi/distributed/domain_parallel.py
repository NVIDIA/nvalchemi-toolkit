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

"""Domain-parallel dynamics wrapper.

Holds a :class:`ShardedBatch` across the step loop and delegates the
per-step model call to a :class:`DistributedModel` — the adapter owns
halo exchange, neighbor-list rebuild, and output consolidation. This
class contributes the orchestration: partition, pre/post-update, atom
migration, and trajectory gather.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import torch
import torch.distributed as dist

from nvalchemi.distributed._core.gather_primitives import mesh_group
from nvalchemi.distributed._dynamics_coordinator import (
    DynamicsDistributionCoordinator,
)
from nvalchemi.distributed.config import DomainConfig, HookScope
from nvalchemi.distributed.strategy import (
    MigrationPlan,
    ParallelizationStrategy,
    strategy_for_policy,
)
from nvalchemi.dynamics.base import BaseDynamics, DynamicsStage
from nvalchemi.hooks._context import HookContext

if TYPE_CHECKING:
    from nvalchemi.data.batch import Batch
    from nvalchemi.distributed.sharded_batch import ShardedBatch

logger = logging.getLogger(__name__)


class DomainParallel(BaseDynamics):
    """Wraps any :class:`BaseDynamics` subclass with spatial domain
    decomposition.

    Flow per step:

    1. Outer BEFORE_STEP hooks on owned batch.
    2. Inner dynamics ``pre_update`` (velocity-Verlet half-kick) on owned batch.
    3. Wrap positions into the periodic box.
    4. Sync the updated positions back into the persistent
       :class:`ShardedBatch` (``update_from_batch``).
    5. ``DistributedModel(sharded)`` — the adapter rebuilds the halo block,
       rebuilds NL, runs the wrapper, consolidates owned-shape outputs.
    6. Write the consolidated outputs back to the owned batch in-place.
    7. Inner dynamics ``post_update`` (velocity-Verlet finalize) on owned batch.
    8. Atom migration (``reshard_by_destination``) for atoms that crossed
       domain boundaries.
    9. Outer AFTER_STEP hooks on owned batch.

    Parameters
    ----------
    dynamics
        The underlying single-GPU dynamics integrator or optimizer.
    config
        Domain decomposition configuration.
    **kwargs
        Forwarded to ``BaseDynamics.__init__`` (``hooks``, ``n_steps``,
        ``device_type``, ...).
    """

    def __init__(
        self,
        dynamics: BaseDynamics,
        config: DomainConfig,
        **kwargs: Any,
    ) -> None:
        super().__init__(model=dynamics.model, **kwargs)
        self._dynamics: BaseDynamics = dynamics
        self._config: DomainConfig = config

        # Globalizes thermodynamic state for NHC/NPT/NPH; inert for NVE/Langevin
        # and the single-process / world-size-1 paths. Built in partition() once
        # the strategy exists (it owns the reductions).
        self._thermo: DynamicsDistributionCoordinator | None = None

        # Lazy-initialized in partition().
        self._strategy: ParallelizationStrategy | None = None
        self._sharded_batch: ShardedBatch | None = None
        # DistributedModel for an atomic model, or DistributedPipelineModel for a
        # composite (PipelineModelWrapper); both share the ShardedBatch->output +
        # close() contract the per-step machinery relies on.
        self._dist_model: Any = None
        self._composite: bool = False

        # Runtime state.
        self._n_owned: int = 0
        self._forces_primed: bool = False

        # Pipeline-stage state (2D pipeline x domain). A DomainParallel used as a
        # DistributedPipeline stage spans a domain sub-mesh; the group lead does
        # the cross-stage isend/irecv while the group scatters/gathers the full
        # system to/from its sub-mesh (see the comm-override section below). Inert
        # unless the pipeline sets prior_rank/next_rank.
        self._pending_input: "Batch | None" = None  # first stage's seed system
        self._first_stage_seeded: bool = False
        self._system_step: int = 0
        self._sentinel_sent: bool = False
        # Process group for cross-stage hand-offs (the pipeline-dim / leads' group),
        # set by DistributedPipeline in grouped mode. ``None`` = default group (the
        # single-stage / test paths). Never the world group in a 2-D pipeline —
        # NCCL requires consistent op ordering per communicator (see base.py).
        self._pipeline_group: Any = None

        # Deferred-migration state. The strategy issues an async consensus
        # all_reduce at the END of step N and consumes it at the START of step
        # N+1, hiding its latency under the intervening hooks + pre_update.
        # Migration ordering is unchanged in physical time: atoms that crossed at
        # end-of-N still migrate before any compute in N+1.
        self._pending_plan: MigrationPlan | None = None

        # Rank resolution — prefer mesh, fall back to global dist rank, else 0.
        if config.mesh is not None:
            try:
                self._domain_rank: int = config.mesh.get_local_rank()
            except Exception:
                self._domain_rank = 0
        elif dist.is_initialized():
            self._domain_rank = dist.get_rank()
        else:
            self._domain_rank = 0

        # Register shard wrappers for nvalchemiops kernels.
        from nvalchemi.distributed.shard_wrappers import register_shard_wrappers

        register_shard_wrappers()

    # ------------------------------------------------------------------
    # Properties delegated to inner dynamics
    # ------------------------------------------------------------------

    @property
    def __needs_keys__(self) -> set[str]:  # type: ignore[override]
        return self._dynamics.__needs_keys__

    @property
    def __provides_keys__(self) -> set[str]:  # type: ignore[override]
        return self._dynamics.__provides_keys__

    # ------------------------------------------------------------------
    # Partition
    # ------------------------------------------------------------------

    def partition(self, batch: Batch | None) -> Batch:
        """Scatter the full-system batch across ranks and build the
        per-step machinery (:class:`ShardedBatch` + :class:`DistributedModel`).

        Must be called once before ``run()`` / ``step()``.

        Parameters
        ----------
        batch
            Full-system batch on rank 0; ``None`` elsewhere. In the
            single-process fallback (no distributed init), passes through.

        Returns
        -------
        Batch
            This rank's owned local batch (per-atom fields are
            ``.to_local()`` views of the ShardedBatch's ShardTensors).
        """
        from nvalchemi.distributed.distributed_model import DistributedModel

        # Single-process fallback — no distribution, just pass through. Gate on
        # the default group's world size too (not just ``is_initialized``) so a
        # leaked 1-rank process group (e.g. a session-scoped gloo PG under
        # pytest) still takes this path rather than the distributed one.
        if not dist.is_initialized() or dist.get_world_size() == 1:
            if batch is None:
                raise ValueError("batch must be provided in single-process mode")
            self._n_owned = batch.positions.shape[0]
            return batch

        mesh = self._config.mesh

        # Adapter around the inner dynamics' model. Owns halo exchange, NL rebuild,
        # and output consolidation. An atomic model rides ``DistributedModel``; a
        # composite (``PipelineModelWrapper``, e.g. MACE+DFTD3 / AIMNet2+PME) can't
        # (``DistributedModel`` wraps atomic models only), so it rides
        # ``DistributedPipelineModel`` — the same ``ShardedBatch``->output contract,
        # doing per-sub-model halo/NL/consolidation internally. Both are halo-based,
        # so the strategy is selected from a halo policy either way.
        from nvalchemi.models.pipeline import PipelineModelWrapper  # noqa: PLC0415

        if isinstance(self._dynamics.model, PipelineModelWrapper):
            from nvalchemi.distributed._core.storage_policy import (  # noqa: PLC0415
                HaloStoragePolicy,
            )
            from nvalchemi.distributed.distributed_pipeline import (  # noqa: PLC0415
                DistributedPipelineModel,
            )

            self._composite = True
            self._dist_model = DistributedPipelineModel(
                self._dynamics.model, self._config, compile=self._config.compile
            )
            policy: Any = HaloStoragePolicy()
        else:
            self._composite = False
            self._dist_model = DistributedModel(
                self._dynamics.model,
                self._config,
                compile=self._config.compile,
            )
            policy = self._dist_model._spec.distribution.policy

        # The parallelization strategy owns data layout, cell tracking,
        # migration, and reductions for this run — selected from the (halo)
        # storage policy so a new strategy plugs in without a driver type-switch.
        self._strategy = strategy_for_policy(policy, self._config, self._domain_rank)

        # Coordinator globalizes NHC/NPT/NPH thermodynamic state via the
        # strategy's reductions (integrator declares intent; inert otherwise).
        self._thermo = DynamicsDistributionCoordinator(self._dynamics, self._strategy)

        # Scatter the full batch across the mesh. The strategy chooses the
        # partition layout (spatial for halo, contiguous-block for graph
        # parallel); ``from_batch`` broadcasts cell/pbc from src and builds the
        # partitioner. The persistent ShardedBatch is shared with the halo config
        # so migration and halo exchange can't disagree on domain boundaries.
        self._sharded_batch = self._strategy.scatter(batch, mesh, self._config, src=0)
        self._n_owned = self._sharded_batch.n_owned

        return self._sharded_batch.local_batch

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def step(self, batch: Batch) -> tuple[Batch, torch.Tensor | None]:
        """Execute one domain-decomposed dynamics step."""
        # Single-process fallback — no distribution set up, delegate to
        # the inner dynamics' own step (which fires its own hook chain).
        if self._dist_model is None:
            return self._dynamics.step(batch)

        # Resolve the previous step's deferred migrate-or-not decision.
        # The async all_reduce was issued at end-of-previous-step; by
        # now it has likely completed in the background. Migrating here
        # (start of step N) is physically equivalent to migrating at
        # end of step N-1 — atoms that crossed at end-of-N-1 still get
        # to their owners before any compute in step N.
        batch = self._resolve_pending_migrate(batch)

        if not self._forces_primed:
            self._prime_forces(batch)
            self._forces_primed = True

        # 1. Outer BEFORE_STEP hooks.
        self._call_hooks(DynamicsStage.BEFORE_STEP, batch)

        dyn = self._dynamics
        dyn._ensure_state_initialized(batch)
        # Globalize per-shard DOF + derived controller masses once the inner
        # state exists (no-op for NVE/Langevin and the single-process path).
        self._thermo.globalize_dof(batch)

        # 2. Pre-update on owned batch (velocity-Verlet half-kick). The reduce
        # scope makes NHC/NPT/NPH couple to mesh-global kinetic state.
        dyn._call_hooks(DynamicsStage.BEFORE_PRE_UPDATE, batch)
        with self._thermo.reduce_scope():
            dyn.pre_update(batch)
        dyn._call_hooks(DynamicsStage.AFTER_PRE_UPDATE, batch)

        # 3. Wrap positions into the periodic box — but ONLY on axes that are
        # not spatially partitioned. Wrapping a *partitioned* axis teleports an
        # owned atom that has drifted just past the periodic boundary a full box
        # length onto the far side, out of its owner's ghost region, so this
        # step computes its force with the wrong (missing) neighbors and injects
        # energy (migration only corrects ownership next step). On the
        # partitioned axis, migration bounds positions instead; the halo design
        # tolerates small unwrapped boundary drift (``keeps_owner``) and the
        # neighbor build uses minimum-image PBC. Non-partitioned axes have no
        # migration to bound them, so they DO need wrapping (safe there — every
        # rank spans the full extent of a non-partitioned axis).
        self._wrap_owned_positions(batch)

        # 4-6. Compute via DistributedModel. ``_distributed_compute``
        # fires the inner BEFORE_COMPUTE / AFTER_COMPUTE hooks on the
        # correct view (padded for halo-storage, owned for sharded).
        self._distributed_compute(batch)

        # 7. Post-update (velocity-Verlet finalize).
        dyn._call_hooks(DynamicsStage.BEFORE_POST_UPDATE, batch)
        with self._thermo.reduce_scope():
            dyn.post_update(batch)
        dyn._call_hooks(DynamicsStage.AFTER_POST_UPDATE, batch)
        # Keep the replicated controller + cell state byte-identical across ranks.
        self._thermo.broadcast_state(batch)

        # 8. Atom migration — DEFERRED. We dispatch the consensus
        # all_reduce here (async); the result is consumed at the start
        # of the NEXT step in ``_resolve_pending_migrate``. This hides
        # the all_reduce latency under the AFTER_STEP hooks + next
        # step's pre_update + halo_exchange instead of forcing a
        # CPU↔GPU sync at end-of-step.
        self._dispatch_async_migrate_check(batch)

        # 9. Outer AFTER_STEP hooks.
        self._call_hooks(DynamicsStage.AFTER_STEP, batch)

        self.step_count += 1
        dyn.step_count += 1

        converged = dyn._check_convergence(batch)
        # Convergence must be a mesh-wide decision: each rank only sees its own
        # atoms, so ranks can disagree and take divergent control flow (one stops
        # while others continue → collective desync / hang). Reduce to the AND
        # across the domain — converged only when EVERY rank is converged.
        if (
            converged is not None
            and dist.is_initialized()
            and self._config.mesh is not None
        ):
            flag = torch.tensor(
                [1 if bool(converged) else 0],
                device=batch.positions.device,
                dtype=torch.int64,
            )
            dist.all_reduce(
                flag, op=dist.ReduceOp.MIN, group=mesh_group(self._config.mesh)
            )
            converged = bool(flag.item())
        dyn._last_converged = converged
        if converged is not None:
            dyn._call_hooks(DynamicsStage.ON_CONVERGE, batch)
        return batch, converged

    # ------------------------------------------------------------------
    # Force priming (initial compute before the first integrator step)
    # ------------------------------------------------------------------

    def _prime_forces(self, batch: Batch) -> None:
        """Run one compute pass to initialize ``batch.forces`` /
        ``batch.energy`` before the first integrator step.

        Velocity-Verlet's first half-kick needs ``batch.forces``; if the
        caller didn't supply them, this pass populates them. ``_distributed_compute``
        handles halo exchange + hook firing internally.
        """
        logger.info("[rank %d] priming forces (initial compute)", self._domain_rank)
        self._distributed_compute(batch)
        logger.info("[rank %d] force priming complete", self._domain_rank)

    # ------------------------------------------------------------------
    # Distributed compute: delegate to DistributedModel
    # ------------------------------------------------------------------

    def _distributed_compute(self, batch: Batch) -> None:
        """Run the model via :class:`DistributedModel` and write the
        owned-shape outputs back into *batch* in-place.

        Flow:

        1. ``update_from_batch`` — sync non-in-place pre_update changes
           back into the persistent ``ShardedBatch``.
        2. ``halo_exchange`` — populate ``sharded.padded_batch`` with the
           refreshed owned + halo atoms.
        3. Fire inner ``BEFORE_COMPUTE`` hooks on ``sharded.padded_batch``
           — ``NeighborListHook`` et al. see the padded view and write
           neighbor data onto it.
        4. ``dist_model(sharded)`` — reads the prepared padded batch + NL,
           runs the wrapper, consolidates to owned-shape outputs.
        5. Fire inner ``AFTER_COMPUTE`` hooks (NaN detectors, etc.).
        6. Write outputs back into the owned ``batch`` in-place.

        Single-process fallback: delegate to ``dyn.compute(batch)`` with
        the owned batch — the inner dynamics' own NL hook fires normally.
        """
        dyn = self._dynamics

        # Single-process fallback.
        if self._sharded_batch is None or self._dist_model is None:
            dyn._call_hooks(DynamicsStage.BEFORE_COMPUTE, batch)
            dyn.compute(batch)
            dyn._call_hooks(DynamicsStage.AFTER_COMPUTE, batch)
            return

        # 1. Sync owned state back into the persistent ShardedBatch.
        self._sharded_batch.update_from_batch(batch)

        if self._composite:
            # Composite adapter (DistributedPipelineModel) owns per-sub-model halo
            # exchange, neighbor rebuild, and owned-shape consolidation internally,
            # so it runs directly on the ShardedBatch — no external halo_exchange /
            # NL hook. Fire the compute hooks on the owned batch (parity with the
            # single-process path; the composite builds its own padded views).
            dyn._call_hooks(DynamicsStage.BEFORE_COMPUTE, batch)
            outputs = self._dist_model(self._sharded_batch)
            dyn._call_hooks(DynamicsStage.AFTER_COMPUTE, batch)
        else:
            # 2. Populate sharded.padded_batch. ``halo_exchange`` needs the halo
            # config which ``DistributedModel`` builds lazily on first call, so
            # prime it here before the external exchange.
            from nvalchemi.distributed._core.storage_policy import HaloStoragePolicy

            if isinstance(
                self._dist_model._spec.distribution.policy, HaloStoragePolicy
            ):
                from nvalchemi.distributed.particle_halo import halo_exchange

                self._dist_model._ensure_initialized(self._sharded_batch)
                halo_exchange(
                    self._sharded_batch,
                    self._dist_model._halo_config,
                    compute_forces=self._dist_model._needs_forces(),
                )
                compute_batch = self._sharded_batch.padded_batch
            else:
                compute_batch = batch

            # 3. BEFORE_COMPUTE hooks — fire on the view the model will see.
            dyn._call_hooks(DynamicsStage.BEFORE_COMPUTE, compute_batch)

            # 4. Model forward via the adapter.
            outputs = self._dist_model(self._sharded_batch)

            # 5. AFTER_COMPUTE hooks.
            dyn._call_hooks(DynamicsStage.AFTER_COMPUTE, compute_batch)

        # 6. Detach all output tensors before writing to the batch and stashing
        # on ``dyn._last_outputs`` (mirrors ``BaseDynamics.compute``). Outputs
        # may carry a live ``grad_fn`` from the energy backward; without
        # detaching, ``_last_outputs`` would pin the whole forward graph until
        # the next step, causing multi-x memory bloat per step. Detach + del
        # breaks every reference so the next forward starts clean.
        from collections import OrderedDict as _OrderedDict  # noqa: PLC0415

        detached: dict[str, Any] = _OrderedDict()
        for key, value in outputs.items():
            if isinstance(value, torch.Tensor):
                detached[key] = value.detach()
            else:
                detached[key] = value
        del outputs

        # Write owned-shape outputs back to the owned batch in-place.
        for out_key, batch_attr in dyn._OUTPUT_KEY_TO_BATCH_ATTR.items():
            value = detached.get(out_key)
            if value is None or not isinstance(value, torch.Tensor):
                continue
            target = getattr(batch, batch_attr, None)
            if target is None:
                setattr(batch, batch_attr, value.clone())
            else:
                target.copy_(value.view(target.shape))

        # Clear ``requires_grad`` on batch tensors that the model
        # enabled for autograd (a conservative-force model flips
        # ``positions.requires_grad_(True)`` per forward); without
        # clearing here, the flag stays on across steps and downstream
        # in-place ops (velocity-Verlet half-kick on positions) raise.
        # Same fix BaseDynamics.compute applies for the single-rank path.
        cfg = dyn.model_config
        grad_keys: set[str] = {"positions"}
        grad_keys |= cfg.gradient_keys
        if cfg.autograd_outputs & cfg.active_outputs:
            grad_keys |= cfg.autograd_inputs
        for key in grad_keys:
            value = getattr(batch, key, None)
            if isinstance(value, torch.Tensor) and value.requires_grad:
                value.requires_grad_(False)

        dyn._last_outputs = detached

    # ------------------------------------------------------------------
    # Atom migration
    # ------------------------------------------------------------------

    def _dispatch_async_migrate_check(self, batch: Batch) -> None:
        """Ask the strategy to decide (async) whether atoms crossed a boundary
        this step. The result is consumed at the START of the next step in
        :meth:`_resolve_pending_migrate`. No-op for strategies that don't
        migrate (graph parallel)."""
        if self._strategy is None or self._sharded_batch is None:
            return
        self._pending_plan = self._strategy.plan_migration(self._sharded_batch, batch)

    def _resolve_pending_migrate(self, batch: Batch) -> Batch:
        """Consume the previous step's deferred migrate-or-not decision and let
        the strategy reshard atoms that crossed a boundary. Called at the START
        of every step (after the first); a no-op until the first
        ``_dispatch_async_migrate_check`` has run and for non-migrating
        strategies.

        The async dispatch was issued at end-of-previous-step, so by the time we
        get here the consensus has typically completed in the background while
        the CPU ran AFTER_STEP + next-step pre_update hooks — a near-instant
        memory fetch, not a forced GPU sync.
        """
        plan = self._pending_plan
        if plan is None or not plan.is_pending or self._sharded_batch is None:
            return batch
        self._pending_plan = None
        new_batch = self._strategy.apply_migration(self._sharded_batch, batch, plan)
        if new_batch is not batch:
            self._n_owned = self._sharded_batch.n_owned
            logger.info(
                "[rank %d] step %d: migrated atoms (deferred consensus)",
                self._domain_rank,
                self.step_count,
            )
        return new_batch

    # ------------------------------------------------------------------
    # Position wrapping
    # ------------------------------------------------------------------

    def _wrap_owned_positions(self, batch: Batch) -> None:
        """Wrap owned positions into the box on the NON-partitioned PBC axes.

        Reuses the shared warp-kernel wrapper
        :func:`nvalchemi.hooks.periodic.wrap_positions_into_cell` (respects
        per-dimension periodicity); the partitioned-axis skip is expressed by
        zeroing that axis in the PBC mask, so the partitioned axis is left to
        migration (wrapping it teleports owned boundary atoms out of ghost
        coverage — see the rationale at the call site in :meth:`step`).
        """
        cell = getattr(batch, "cell", None)
        pbc = getattr(batch, "pbc", None)
        if cell is None or pbc is None or not bool(pbc.any()):
            return

        wrap_pbc = pbc.clone()
        rank_grid = getattr(
            getattr(self._sharded_batch, "partitioner", None), "rank_grid", None
        )
        if rank_grid is not None:  # spatial partition: don't wrap the split axes
            for i, p in enumerate(rank_grid):
                if int(p) > 1:
                    wrap_pbc[..., i] = False
        if not bool(wrap_pbc.any()):
            return

        from nvalchemi.hooks.periodic import wrap_positions_into_cell  # noqa: PLC0415

        batch_idx = getattr(batch, "batch_idx", None)
        if batch_idx is None:
            batch_idx = torch.zeros(
                batch.positions.shape[0],
                dtype=torch.long,
                device=batch.positions.device,
            )
        wrap_positions_into_cell(batch.positions, cell, wrap_pbc, batch_idx)

    # ------------------------------------------------------------------
    # Gather (trajectory output)
    # ------------------------------------------------------------------

    def gather(self, local_batch: Batch, dst: int = 0) -> Batch | None:
        """Gather the distributed system back into a full :class:`Batch`
        on rank *dst*. Returns ``None`` on other ranks.

        Single-process fallback: returns ``local_batch`` unchanged.
        """
        if not dist.is_initialized() or self._sharded_batch is None:
            return local_batch

        # Sync the latest local state into the ShardedBatch before gathering.
        self._sharded_batch.update_from_batch(local_batch)
        return self._sharded_batch.full_batch(dst=dst)

    def _gather_all(self, local_batch: Batch) -> Batch:
        """Gather the full system onto **every** rank (for GLOBAL-scope hooks).

        Single-process fallback returns ``local_batch`` unchanged. The per-system
        ``energy`` is already globally reduced + replicated by the forward, so it
        is carried through as-is (never re-reduced).
        """
        if not dist.is_initialized() or self._sharded_batch is None:
            return local_batch
        self._sharded_batch.update_from_batch(local_batch)
        full = self._sharded_batch.to_global_batch()
        if getattr(local_batch, "energy", None) is not None:
            full.energy = local_batch.energy.clone()
        return full

    # ------------------------------------------------------------------
    # Hook overrides
    # ------------------------------------------------------------------

    def _build_context(self, batch: Batch) -> HookContext:
        ctx = super()._build_context(batch)
        ctx.n_owned = self._n_owned
        ctx.domain_mesh = self._config.mesh
        ctx.is_domain_parallel = True
        ctx.global_cell = (
            self._sharded_batch.cell.clone()
            if self._sharded_batch is not None
            else None
        )
        return ctx

    def _call_hooks(self, stage: DynamicsStage, batch: Batch) -> None:
        """Invoke hooks respecting their ``HookScope``.

        - LOCAL: hook sees the per-rank owned batch (no communication).
        - GLOBAL: per-system ``energy`` all-reduced before the hook fires.
        - RANK_ZERO: system gathered to rank 0; hook runs only there.
        """
        ctx = self._build_context(batch)

        for hook in self.hooks:
            runs_on_stage = getattr(hook, "_runs_on_stage", None)
            if runs_on_stage is not None:
                if not runs_on_stage(stage):
                    continue
            elif stage != hook.stage:
                continue

            if self.step_count % hook.frequency != 0:
                continue

            scope = getattr(hook, "scope", HookScope.LOCAL)

            if scope == HookScope.GLOBAL:
                # GLOBAL means the hook sees the COMPLETE system. Gather the full
                # batch onto every rank (not the local shard) and run the hook on
                # the gathered batch. Do NOT re-reduce ``energy``: the forward's
                # consolidation already all-reduced it to the global value and
                # replicated it per rank, so summing again would multiply it by
                # the rank count.
                full = self._gather_all(batch)
                ctx_full = self._build_context(full) if full is not None else ctx
                hook(ctx_full, stage)

            elif scope == HookScope.RANK_ZERO:
                full_batch = self.gather(batch, dst=0)
                if self._domain_rank == 0 and full_batch is not None:
                    ctx_full = self._build_context(full_batch)
                    hook(ctx_full, stage)

            else:
                hook(ctx, stage)

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def run(self, batch: Batch, n_steps: int | None = None) -> Batch:
        """Run the domain-decomposed simulation for *n_steps* steps."""
        # Single-process fallback — delegate to the inner dynamics' run.
        if self._dist_model is None:
            return self._dynamics.run(batch, n_steps=n_steps)

        resolved = n_steps if n_steps is not None else self.n_steps
        if resolved is None:
            raise ValueError(
                "No step count provided. Either pass `n_steps` to run() "
                "or set it at construction time."
            )
        self._open_hooks()
        try:
            if not self._forces_primed:
                self._prime_forces(batch)
                self._forces_primed = True

            for _ in range(resolved):
                batch, _converged = self.step(batch)
        finally:
            self._close_hooks()
        return batch

    # ------------------------------------------------------------------
    # Pipeline-stage communication (group-aware _CommunicationMixin override)
    # ------------------------------------------------------------------
    # A DomainParallel used as a DistributedPipeline stage spans a whole domain
    # sub-mesh. The pipeline drives every stage through the identical
    # _CommunicationMixin API (_ensure_buffers -> _prestep_sync_buffers -> step ->
    # _poststep_sync_buffers, plus done/is_first_stage/is_last_stage); these
    # overrides make that API group-aware. The group LEAD (domain-rank 0) performs
    # the cross-stage isend/irecv to adjacent stage-groups' leads; the group then
    # scatters/gathers the full system to/from its sub-mesh. Non-lead ranks do no
    # cross-stage I/O. Granularity: one DD step per pipeline iteration; a system
    # graduates on the iteration it finishes (converged, or its step budget spent),
    # so a group re-partitions only when a system arrives — never every step.

    @property
    def _is_group_lead(self) -> bool:
        """Whether this rank is its stage-group's lead (domain-rank 0), the only
        rank that transmits full systems across stages."""
        return self._domain_rank == 0

    def _bcast_group_flag(self, flag: bool) -> bool:
        """Broadcast a bool from the group lead to the whole domain sub-mesh so all
        ranks take the same partition/idle control flow (single-process: identity)."""
        group = mesh_group(self._config.mesh)
        if not dist.is_initialized() or group is None:
            return flag
        t = torch.tensor([1 if flag else 0], dtype=torch.int32, device=self.device)
        dist.broadcast(t, src=dist.get_global_rank(group, 0), group=group)
        return bool(t.item())

    def _system_finished(self, converged: Any) -> bool:
        """A resident system leaves this stage when it converges (FIRE) or spends
        its per-system step budget (``n_steps``, e.g. an NVT leg)."""
        if converged:
            return True
        return self.n_steps is not None and self._system_step >= self.n_steps

    def _dd_event(self, msg: str) -> None:
        """Emit a stage-transition line — a system arriving, finishing, or being
        handed to another GPU/stage — when ``debug_mode`` is on. This is the "where
        is each system and when does it change GPUs/stages" trace for 2-D pipelines.
        """
        if not getattr(self, "debug_mode", False):
            return
        from loguru import logger as _logger

        rank = dist.get_rank() if dist.is_initialized() else 0
        _logger.info("[DD rank {}] {}", rank, msg)

    def _ensure_buffers(self, template: "Batch") -> None:
        """No-op: a DD stage hands off whole systems via ``Batch.send``/``irecv``
        (template-driven), not the streaming fixed-capacity send/recv buffers."""
        return

    def _prestep_sync_buffers(self) -> None:
        """Pull the next system into this stage when idle (whole-system-in-flight).

        First stage: seed once from the injected initial batch. Downstream stage:
        the lead ``irecv``s the next full system from the prior stage's lead (a
        0-graph sentinel means the upstream is exhausted), then the group
        ``partition``s it across the domain sub-mesh.
        """
        if self.active_batch is not None and self.active_batch.num_graphs > 0:
            return  # still working the current system

        if self.prior_rank is None:
            # First stage: seed from _pending_input exactly once, then it's spent.
            if not self._first_stage_seeded:
                self._first_stage_seeded = True
                self._system_step = 0
                seed = self._pending_input if self._is_group_lead else None
                self._pending_input = None
                self.active_batch = self.partition(seed)
                self._dd_event(
                    f"seeded initial system → scattered across the domain group "
                    f"(n_owned={self._n_owned})"
                )
            else:
                self.done = True
                self._send_sentinel()  # tell the next stage no more systems are coming
                self._dd_event("first stage exhausted → done (drain signal sent)")
            return

        # Downstream stage: lead receives the next full system from the prior lead.
        from nvalchemi.data.batch import Batch

        received = None
        got_system = True
        if self._is_group_lead:
            received = Batch.irecv(
                src=self.prior_rank,
                device=self.device,
                template=self._recv_template,
                group=self._pipeline_group,
            ).wait()
            got_system = received.num_graphs > 0  # 0-graph sentinel = upstream done
        got_system = self._bcast_group_flag(got_system)
        if not got_system:
            self.active_batch = None
            self.done = True
            self._send_sentinel()  # forward the drain signal down the chain
            self._dd_event(f"upstream (rank {self.prior_rank}) drained → stage done")
            return
        self._system_step = 0
        self.active_batch = self.partition(received if self._is_group_lead else None)
        self._dd_event(
            f"received a system from rank {self.prior_rank} → scattered across the "
            f"domain group (n_owned={self._n_owned})"
        )

    def _send_sentinel(self) -> None:
        """Lead sends a one-shot 0-graph batch to the next stage's lead — the drain
        signal that unblocks its ``irecv`` and propagates ``done`` down the chain
        (``Batch.isend`` of an empty batch ships only the meta header)."""
        if self.next_rank is None or not self._is_group_lead or self._sentinel_sent:
            return
        from nvalchemi.data.batch import Batch

        self._sentinel_sent = True
        Batch(device=self.device).isend(
            dst=self.next_rank, group=self._pipeline_group
        ).wait()

    def _complete_pending_recv(self) -> None:
        """No-op: the lead completes its ``irecv`` inline in
        :meth:`_prestep_sync_buffers` (nothing is deferred)."""
        return

    def _poststep_sync_buffers(self, converged: Any = None) -> None:
        """Graduate the resident system when it finishes this stage: gather it to
        the group lead, which ``send``s it to the next stage's lead. The stage then
        goes idle so the next system can enter. The last stage (no ``next_rank``)
        just retires the finished system — its trajectory is already captured by
        hooks/sinks."""
        if self.active_batch is None:
            return
        self._system_step += 1
        if not self._system_finished(converged):
            return
        reason = "converged" if converged else f"reached its {self.n_steps}-step budget"
        if self.next_rank is not None:
            full = self.gather(self.active_batch, dst=0)
            if self._is_group_lead and full is not None:
                full.send(dst=self.next_rank, group=self._pipeline_group)
            self._dd_event(
                f"system {reason} after {self._system_step} steps → gathered + "
                f"handed off to the next stage's lead (rank {self.next_rank})"
            )
        else:
            self._dd_event(
                f"system {reason} after {self._system_step} steps → retired "
                "(final stage)"
            )
        self.active_batch = None
        self._system_step = 0

    # ------------------------------------------------------------------
    # Teardown
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Release resources held by the adapter (restores any state its
        ``distributed_setup`` mutated on the inner wrapper). Safe to call
        multiple times."""
        # Drain any pending deferred migrate-or-not all_reduce so the
        # NCCL work handle doesn't outlive the process group.
        if self._pending_plan is not None and self._pending_plan.is_pending:
            try:
                self._pending_plan.work.wait()
            except Exception:  # noqa: S110 — teardown best-effort
                pass
        self._pending_plan = None
        if self._dist_model is not None:
            self._dist_model.close()

    def __enter__(self) -> "DomainParallel":
        """Enter a scope that releases the adapter's setup on exit.

        Lets a caller write ``with DomainParallel(...) as dyn: dyn.partition(...);
        dyn.run(...)`` so teardown (``close()``) is exception-safe. The process
        group / ``DistributedManager`` lifecycle stays at launcher scope."""
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:  # noqa: S110
            pass
