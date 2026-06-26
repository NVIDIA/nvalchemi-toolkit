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
from nvalchemi.distributed.config import DomainConfig, HookScope
from nvalchemi.dynamics.base import BaseDynamics, DynamicsStage
from nvalchemi.hooks._context import HookContext

if TYPE_CHECKING:
    from nvalchemi.data.batch import Batch
    from nvalchemi.distributed.distributed_model import DistributedModel
    from nvalchemi.distributed.partitioner import SpatialPartitioner
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

        # Lazy-initialized in partition().
        self._partitioner: SpatialPartitioner | None = None
        self._sharded_batch: ShardedBatch | None = None
        self._dist_model: DistributedModel | None = None

        # Runtime state.
        self._n_owned: int = 0
        self._forces_primed: bool = False

        # Deferred-migration state. The consensus all_reduce that decides
        # whether to reshard is issued ``async_op=True`` at the END of step N
        # and read at the START of step N+1, hiding its latency under the
        # intervening hooks + pre_update. Migration ordering is unchanged in
        # physical time: atoms that crossed at end-of-N still migrate before
        # any compute in N+1.
        self._pending_migrate_work: Any = None
        self._pending_migrate_flag: torch.Tensor | None = None

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
        from nvalchemi.distributed.sharded_batch import ShardedBatch

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

        # Scatter the full batch. ``ShardedBatch.from_batch`` broadcasts
        # cell/pbc from src and builds the partitioner from the config. Halo is
        # the only storage policy, so the partition is always spatial.
        partition_mode = "spatial"

        self._sharded_batch = ShardedBatch.from_batch(
            batch=batch,
            mesh=mesh,
            config=self._config,
            src=0,
            partition_mode=partition_mode,
        )
        self._n_owned = self._sharded_batch.n_owned

        # Reuse the partitioner built by ``from_batch`` (same config +
        # broadcast cell/pbc as we'd feed any re-construction). Shared
        # with ``DistributedModel``'s halo config so migration and halo
        # exchange can't disagree on domain boundaries.
        from nvalchemi.distributed.partitioner import SpatialPartitioner

        self._partitioner = self._sharded_batch.partitioner or SpatialPartitioner(
            config=self._config,
            cell_matrix=self._sharded_batch.cell,
            pbc=self._sharded_batch.pbc,
        )

        # Adapter around the inner dynamics' model. Owns halo exchange,
        # NL rebuild, and output consolidation.
        self._dist_model = DistributedModel(self._dynamics.model, self._config)

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

        # 2. Pre-update on owned batch (velocity-Verlet half-kick).
        dyn._call_hooks(DynamicsStage.BEFORE_PRE_UPDATE, batch)
        dyn.pre_update(batch)
        dyn._call_hooks(DynamicsStage.AFTER_PRE_UPDATE, batch)

        # 3. Wrap positions into the periodic box.
        self._wrap_positions(batch)

        # 4-6. Compute via DistributedModel. ``_distributed_compute``
        # fires the inner BEFORE_COMPUTE / AFTER_COMPUTE hooks on the
        # correct view (padded for halo-storage, owned for sharded).
        self._distributed_compute(batch)

        # 7. Post-update (velocity-Verlet finalize).
        dyn._call_hooks(DynamicsStage.BEFORE_POST_UPDATE, batch)
        dyn.post_update(batch)
        dyn._call_hooks(DynamicsStage.AFTER_POST_UPDATE, batch)

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

        # 2. Populate sharded.padded_batch. ``halo_exchange`` needs the halo
        # config which ``DistributedModel`` builds lazily on first call, so
        # prime it here before the external exchange.
        from nvalchemi.distributed._core.storage_policy import HaloStoragePolicy

        if isinstance(self._dist_model._spec.distribution.policy, HaloStoragePolicy):
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
        """Issue the consensus all_reduce that decides whether ANY rank's
        atoms crossed a boundary this step. Result is consumed at the
        START of the next step in :meth:`_resolve_pending_migrate`.

        We deliberately discard ``new_rank`` here — it gets recomputed
        fresh in the resolver if migration fires. Recomputation is cheap
        (one cell-list ``floor(positions @ inv_cell)`` pass; the
        ``inv_cell`` is cached on the partitioner) and avoids holding a
        stale rank assignment across hook calls that could mutate
        positions (barostats, freezers, etc.).

        """
        if self._partitioner is None or not dist.is_initialized():
            return
        # Hysteresis-aware: flag migration only when an atom has LEFT this
        # rank's domain expanded by the hysteresis margin (not merely crossed the
        # bare boundary) — stops thrashing of atoms vibrating across the plane.
        h = self._config.effective_migration_hysteresis()
        leaving = ~self._partitioner.keeps_owner(
            batch.positions, self._domain_rank, h
        )
        flag = leaving.any().to(torch.int32).view(1)
        group = mesh_group(self._config.mesh)
        work = dist.all_reduce(flag, op=dist.ReduceOp.MAX, group=group, async_op=True)
        self._pending_migrate_flag = flag
        self._pending_migrate_work = work

    def _resolve_pending_migrate(self, batch: Batch) -> Batch:
        """Wait on the previous step's deferred migrate-or-not consensus
        and migrate atoms if any crossed a boundary. Called at the START
        of every step (after the first); a no-op until the first
        ``_dispatch_async_migrate_check`` has run.

        The async dispatch was issued at end-of-previous-step, so by the
        time we get here the all_reduce has typically completed in the
        background while the CPU was running AFTER_STEP hooks and the
        next step's pre_update hooks. The ``.item()`` here is therefore
        a near-instant memory fetch, not a forced GPU sync.
        """
        if self._pending_migrate_work is None:
            return batch
        self._pending_migrate_work.wait()
        needs = bool(self._pending_migrate_flag.item())
        self._pending_migrate_work = None
        self._pending_migrate_flag = None
        if not needs:
            return batch

        logger.info(
            "[rank %d] step %d: migrating atoms (deferred consensus)",
            self._domain_rank,
            self.step_count,
        )

        from nvalchemi.distributed._core.reshard import reshard_by_destination

        device = batch.positions.device
        # Recompute the rank assignment from the latest positions —
        # AFTER_STEP hooks could have nudged positions between dispatch
        # and resolve, and we want the migration to be based on what's
        # in batch right now.
        # Hysteresis-aware destinations: atoms still within this rank's
        # expanded domain KEEP this rank (else the reshard would move band atoms
        # anyway, defeating hysteresis); only atoms that have left get their
        # natural spatial rank.
        h = self._config.effective_migration_hysteresis()
        keep = self._partitioner.keeps_owner(batch.positions, self._domain_rank, h)
        natural = self._partitioner.assign_atoms_to_ranks(batch.positions)
        new_rank = torch.where(
            keep,
            torch.full_like(natural, self._domain_rank),
            natural,
        ).to(torch.int64)
        mesh = self._config.mesh

        # Reshard each per-atom field independently (preserves dtypes).
        fields: dict[str, torch.Tensor] = {"positions": batch.positions}
        for name in ("atomic_numbers", "atomic_masses", "velocities", "forces"):
            val = getattr(batch, name, None)
            if val is not None:
                fields[name] = val

        new_fields: dict[str, torch.Tensor] = {
            name: reshard_by_destination(tensor, new_rank, mesh)
            for name, tensor in fields.items()
        }

        new_batch = self._build_batch_from_fields(new_fields, device)
        if getattr(batch, "cell", None) is not None:
            new_batch.cell = batch.cell.clone()
        if getattr(batch, "pbc", None) is not None:
            new_batch.pbc = batch.pbc.clone()
        if getattr(batch, "energy", None) is not None:
            new_batch.energy = batch.energy.clone()

        self._n_owned = new_fields["positions"].shape[0]

        # Refresh the persistent ShardedBatch to match the new layout and
        # invalidate the padded view — migration changes rank ownership,
        # so the halo routing and any cached NL are stale.
        if self._sharded_batch is not None:
            self._sharded_batch.update_from_batch(new_batch)
            self._sharded_batch.invalidate_padded_view()

        return new_batch

    @staticmethod
    def _build_batch_from_fields(
        fields: dict[str, torch.Tensor], device: torch.device
    ) -> Batch:
        from nvalchemi.data.atomic_data import AtomicData
        from nvalchemi.data.batch import Batch as BatchCls

        data = AtomicData(
            positions=fields["positions"],
            atomic_numbers=fields.get(
                "atomic_numbers", torch.zeros(0, dtype=torch.long, device=device)
            ),
        )
        if "atomic_masses" in fields:
            data.atomic_masses = fields["atomic_masses"]
        if "velocities" in fields:
            data.add_node_property("velocities", fields["velocities"])
        if "forces" in fields:
            data.add_node_property("forces", fields["forces"])
        return BatchCls.from_data_list([data], device=device)

    # ------------------------------------------------------------------
    # Position wrapping
    # ------------------------------------------------------------------

    @staticmethod
    def _wrap_positions(batch: Batch) -> None:
        """Wrap atom positions back into the periodic box (fractional
        coordinates modulo 1 along any PBC dim)."""
        cell = getattr(batch, "cell", None)
        pbc = getattr(batch, "pbc", None)
        if cell is None or pbc is None or not pbc.any():
            return

        cell_3x3 = cell.squeeze(0)
        inv_cell = torch.linalg.inv(cell_3x3)
        frac = batch.positions @ inv_cell.T
        pbc_mask = pbc.squeeze(0)
        frac[:, pbc_mask] = frac[:, pbc_mask] % 1.0
        batch.positions = frac @ cell_3x3

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
                if dist.is_initialized() and getattr(batch, "energy", None) is not None:
                    dist.all_reduce(batch.energy, op=dist.ReduceOp.SUM)
                hook(ctx, stage)

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
    # Teardown
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Release resources held by the adapter (restores any state its
        ``distributed_setup`` mutated on the inner wrapper). Safe to call
        multiple times."""
        # Drain any pending deferred migrate-or-not all_reduce so the
        # NCCL work handle doesn't outlive the process group.
        if self._pending_migrate_work is not None:
            try:
                self._pending_migrate_work.wait()
            except Exception:  # noqa: S110 — teardown best-effort
                pass
            self._pending_migrate_work = None
            self._pending_migrate_flag = None
        if self._dist_model is not None:
            self._dist_model.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:  # noqa: S110
            pass
