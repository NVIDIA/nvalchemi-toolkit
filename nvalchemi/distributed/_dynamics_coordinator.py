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

"""Dynamics distribution coordinator: globalize NHC / NPT / NPH / FIRE under any strategy.

``DomainParallel`` integrates each rank's owned atom shard independently. That is
exact for ensembles that depend only on local atom state (NVE, Langevin).
Nosé-Hoover (NVT), NPT, and NPH couple their controllers to **global**
thermodynamic quantities — the system kinetic energy, the kinetic pressure
tensor, and the degrees of freedom — which the single-process integrators compute
from whatever atoms they are handed (a rank's shard under DD). Left alone, every
rank would thermostat/barostat against a per-shard quantity and the trajectory
would be wrong. FIRE / FIRE2 geometry optimizers are the same story: their
velocity mixing and timestep adaptation are gated by **global** per-system
power/norm scalars (``v·f``, ``v·v``, ``f·f``) reduced over all atoms, so each
rank must mix against the same global values or the replicated relaxation
desyncs.

This module makes those ensembles correct without touching the integrators. The
integrator **declares intent** as class metadata (``__dd_thermo_kind__``,
``__dd_replicated__``) — it never contains DD verbs in its body — and the
coordinator provides the mechanism, globalizing via the **strategy's**
``reduce_system`` / ``global_atom_count``. Routing reductions through the strategy
is what keeps this correct across layouts: a naive ``all_reduce`` over-counts by
the world size under a node-replicate strategy (every rank already holds all
nodes), where the strategy's reduction is the identity.

* **DOF** — overwrite the integrator's per-shard degree-of-freedom state (and the
  controller masses derived from it) with the mesh-global value, once, after
  ``partition()``.
* **Kinetic energy / tensor** — intercept the kinetic helpers the integrator
  calls so their per-shard result is reduced across the mesh (the kinetic
  pressure tensor is fed back through the ops ``compute_kinetic=False`` seam; the
  virial term is already global because the forward consolidates stress).
* **Controller lockstep** — broadcast the replicated controller + cell state
  (the integrator's declared ``__dd_replicated__``) from rank 0 each step so
  floating-point divergence can't accumulate.

The reductions reduce to two primitives (kinetic energy and DOF); see
``proposal-dd-global-thermo.md`` and ``proposal-distributed-strategy-refactor.md``.
"""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, Any

import torch
import torch.distributed as dist

from nvalchemi.distributed.strategy import ParallelizationStrategy, Reduce

if TYPE_CHECKING:
    from nvalchemi.data.batch import Batch
    from nvalchemi.dynamics.base import BaseDynamics

__all__ = ["DynamicsDistributionCoordinator"]


class DynamicsDistributionCoordinator:
    """Globalizes the thermodynamic state of an NHC/NPT/NPH integrator running
    under :class:`~nvalchemi.distributed.domain_parallel.DomainParallel`, for any
    :class:`~nvalchemi.distributed.strategy.ParallelizationStrategy`.

    Inert (``active`` is False) for local-only ensembles (no ``__dd_thermo_kind__``
    declared) or when no real decomposition is present (no distributed init /
    world size 1), so the single-process and NVE/Langevin paths are completely
    unaffected.

    Parameters
    ----------
    dynamics
        The inner single-process integrator. Its ``__dd_thermo_kind__`` /
        ``__dd_replicated__`` class attributes declare its DD intent.
    strategy
        The parallelization strategy owning the reductions + collective group.
    """

    def __init__(
        self, dynamics: BaseDynamics, strategy: ParallelizationStrategy
    ) -> None:
        self._dyn = dynamics
        self._strategy = strategy
        self._kind: str | None = getattr(dynamics, "__dd_thermo_kind__", None)
        self._patches: list[tuple[Any, str, Any]] = []
        self._dof_done = False

    # ------------------------------------------------------------------

    @property
    def active(self) -> bool:
        """True only when this is a global-thermo ensemble *and* a real
        multi-rank decomposition is in effect."""
        return (
            self._kind is not None
            and dist.is_initialized()
            and dist.get_world_size() > 1
        )

    def _reduce_sum(self, t: torch.Tensor) -> None:
        """Mesh-reduce a per-system quantity to its global value, in place, via
        the strategy (identity for node-replicate, all_reduce SUM otherwise)."""
        self._strategy.reduce_system(t, Reduce.SUM)

    # ------------------------------------------------------------------
    # Degrees of freedom (run once, after partition)
    # ------------------------------------------------------------------

    def globalize_dof(self, local_batch: Batch) -> None:
        """Replace the integrator's per-shard DOF (and the controller masses
        derived from it) with the mesh-global value.

        DOF is fixed for the run (modulo constraints), so this runs once. The
        shard's atom count leaks into: the thermostat/barostat ``ndof``, the
        first NHC chain mass ``Q_0 = ndof·kT·τ²``, and (NPT/NPH) the barostat
        inertia ``W``. All are overwritten here from the global atom count.
        """
        if not self.active or self._dof_done:
            return
        if self._kind == "fire":
            # FIRE carries no DOF-derived controller state; its only global
            # coupling is the per-step v·f / v·v / f·f reduction (see
            # ``reduce_scope``). Nothing to globalize once, so this is inert.
            self._dof_done = True
            return
        state = getattr(self._dyn, "_state", None)
        if state is None:  # state not yet initialized — caller retries post-init
            return

        M = state.dt.shape[0]
        dev = local_batch.positions.device
        dtype = local_batch.positions.dtype
        global_counts = torch.bincount(local_batch.batch_idx, minlength=M).to(
            dtype=torch.int64, device=dev
        )
        # Per-system owned-atom count → mesh-global via the strategy (identity for
        # node-replicate, where the shard already holds every atom).
        self._reduce_sum(global_counts)
        global_ndof = (global_counts * 3).to(dtype=dtype)

        if self._kind == "nhc":
            state.nhc_ndof.copy_(global_ndof)
            # Q_0 = ndof·kT·τ²; higher links are ndof-independent.
            tau = state.thermostat_time
            state.nhc_Q[:, 0] = global_ndof * state.temperature * tau * tau
        else:  # npt / nph
            state.num_atoms_per_system.copy_(global_counts.to(torch.int32))
            self._recompute_barostat_mass(state, global_counts, dtype, dev)
            if self._kind == "npt":
                # NPT also carries a particle NHC chain whose Q_0 depends on ndof.
                tau_t = state.thermostat_time
                state.nhc_Q[:, 0] = global_ndof * state.temperature * tau_t * tau_t

        self._dof_done = True

    @staticmethod
    def _recompute_barostat_mass(
        state: Batch, global_counts: torch.Tensor, dtype: torch.dtype, dev: Any
    ) -> None:
        """Recompute the barostat inertia ``W`` from the global atom count,
        mirroring the integrator's init (``compute_barostat_mass`` then ``/3``)."""
        from nvalchemi.dynamics._ops.npt_nph import (  # noqa: PLC0415
            compute_barostat_mass,
        )

        # NPH does not store a target temperature (its W uses a fixed kT
        # estimate at init); reconstruct that same estimate so the global-N
        # rescale is consistent with how the shard's W was built.
        kT = getattr(state, "temperature", None)
        if kT is None:
            kT = torch.full(
                (global_counts.shape[0],),
                300.0 * _kb_ev(),
                dtype=dtype,
                device=dev,
            )
        W = torch.zeros(global_counts.shape[0], dtype=dtype, device=dev)
        compute_barostat_mass(kT, state.barostat_time, global_counts.to(torch.int32), W)
        state.W.copy_(W / 3)

    # ------------------------------------------------------------------
    # Kinetic-quantity interception (active during pre/post_update)
    # ------------------------------------------------------------------

    @contextlib.contextmanager
    def reduce_scope(self):
        """Context manager that patches the integrator module's kinetic helpers
        to mesh-reduce (via the strategy), for the duration of a pre/post_update
        call."""
        if not self.active:
            yield
            return
        self._install()
        try:
            yield
        finally:
            self._restore()

    def _install(self) -> None:
        strat = self._strategy
        if self._kind == "fire":
            # FIRE mixing is driven by global v·f / v·v / f·f. Patch the ops
            # entry points the optimizer imported so each per-shard reduction is
            # summed across the mesh and fed back via ``compute_reductions=False``.
            from nvalchemi.dynamics.optimizers import fire as mod  # noqa: PLC0415

            self._swap(mod, "fire_step", _make_global_fire_step(strat))
            self._swap(mod, "fire_update", _make_global_fire_update(strat))
        elif self._kind == "nhc":
            from nvalchemi.dynamics.integrators import (
                nvt_nose_hoover as mod,  # noqa: PLC0415
            )

            self._swap(mod, "nhc_chain_update", _make_global_nhc_chain_update(strat))
        else:  # npt / nph
            mod_name = "npt" if self._kind == "npt" else "nph"
            mod = __import__(
                f"nvalchemi.dynamics.integrators.{mod_name}",
                fromlist=["compute_kinetic_energy", "compute_pressure_tensor"],
            )
            self._swap(
                mod,
                "compute_kinetic_energy",
                _make_global_kinetic_energy(mod.compute_kinetic_energy, strat),
            )
            self._swap(
                mod,
                "compute_pressure_tensor",
                _make_global_pressure_tensor(mod.compute_pressure_tensor, strat),
            )

    def _swap(self, module: Any, name: str, fn: Any) -> None:
        self._patches.append((module, name, getattr(module, name)))
        setattr(module, name, fn)

    def _restore(self) -> None:
        while self._patches:
            module, name, original = self._patches.pop()
            setattr(module, name, original)

    # ------------------------------------------------------------------
    # Controller / cell lockstep (anti-drift, after each step)
    # ------------------------------------------------------------------

    def broadcast_state(self, batch: Batch) -> None:
        """Broadcast the replicated controller + cell state (the integrator's
        declared ``__dd_replicated__``) from rank 0 so floating-point divergence
        cannot accumulate over a long run.

        With global KE/DOF and identical config the controller evolves
        identically on every rank, so this is anti-drift insurance, not a
        correctness requirement; the cost is a handful of small tensors.
        """
        if not self.active:
            return
        group = self._strategy.process_group
        src = 0
        state = getattr(self._dyn, "_state", None)
        fields = getattr(self._dyn, "__dd_replicated__", ())
        for name in fields:
            t = getattr(state, name, None)
            if isinstance(t, torch.Tensor):
                dist.broadcast(t, src=src, group=group)
        # The barostat mutates the cell; keep it byte-identical across ranks.
        if self._kind in ("npt", "nph") and isinstance(
            getattr(batch, "cell", None), torch.Tensor
        ):
            dist.broadcast(batch.cell, src=src, group=group)


def _kb_ev() -> float:
    from nvalchemi.dynamics.hooks._utils import KB_EV  # noqa: PLC0415

    return KB_EV


# ----------------------------------------------------------------------
# Reduced-helper factories. Each wraps the real op so the per-shard kinetic
# quantity is reduced across the mesh (via the strategy) and fed back through the
# ops flag that skips the in-kernel recompute.
# ----------------------------------------------------------------------


def _fire_local_reductions(
    velocities: torch.Tensor,
    forces: torch.Tensor,
    batch_idx: torch.Tensor,
    vf: torch.Tensor,
    vv: torch.Tensor,
    ff: torch.Tensor,
    strategy: ParallelizationStrategy,
) -> None:
    """Fill ``vf/vv/ff`` with the mesh-global per-system power/norm sums.

    Each rank accumulates its owned-atom partials — ``vf=Σ Fᵢ·vᵢ``,
    ``vv=Σ vᵢ·vᵢ``, ``ff=Σ Fᵢ·Fᵢ`` — then reduces them SUM across the mesh via
    the strategy (identity for node-replicate, all_reduce otherwise). The result
    is fed straight into the ops kernel with ``compute_reductions=False`` so
    every rank mixes velocities against the same global scalars.
    """
    bidx = batch_idx.long()
    vf.zero_()
    vv.zero_()
    ff.zero_()
    vf.index_add_(0, bidx, (forces * velocities).sum(-1))
    vv.index_add_(0, bidx, (velocities * velocities).sum(-1))
    ff.index_add_(0, bidx, (forces * forces).sum(-1))
    strategy.reduce_system(vf, Reduce.SUM)
    strategy.reduce_system(vv, Reduce.SUM)
    strategy.reduce_system(ff, Reduce.SUM)


def _make_global_fire_step(strategy: ParallelizationStrategy):
    """Wrap the FIRE optimizer's ``fire_step`` so its velocity mixing uses the
    mesh-global power/norm reductions rather than the owned shard's.

    The MD integration, per-atom ``maxstep`` clamp, and energy/uphill handling
    stay local/global exactly as in the single-process op (per-atom quantities
    are correct under the shard; energy is already consolidated by the forward).
    Only ``vf/vv/ff`` are reduced. For vanilla FIRE (``uphill=False``) the revert
    is a no-op, so the pre-call reductions equal the post-revert ones the op
    would compute.
    """
    from nvalchemi.dynamics._ops.fire import fire_step as _real  # noqa: PLC0415

    def _global_fire_step(
        positions,
        velocities,
        forces,
        masses,
        alpha,
        dt,
        n_steps_positive,
        alpha_start,
        f_alpha,
        dt_min,
        dt_max,
        maxstep,
        n_min,
        f_dec,
        f_inc,
        uphill_flag,
        *,
        vf=None,
        vv=None,
        ff=None,
        batch_idx=None,
    ):
        M = alpha.shape[0]
        dev, dtype = velocities.device, velocities.dtype
        if vf is None:
            vf = torch.zeros(M, dtype=dtype, device=dev)
        if vv is None:
            vv = torch.zeros(M, dtype=dtype, device=dev)
        if ff is None:
            ff = torch.zeros(M, dtype=dtype, device=dev)
        if batch_idx is None:
            batch_idx = torch.zeros(velocities.shape[0], dtype=torch.int32, device=dev)
        _fire_local_reductions(velocities, forces, batch_idx, vf, vv, ff, strategy)
        _real(
            positions,
            velocities,
            forces,
            masses,
            alpha,
            dt,
            n_steps_positive,
            alpha_start,
            f_alpha,
            dt_min,
            dt_max,
            maxstep,
            n_min,
            f_dec,
            f_inc,
            uphill_flag,
            vf=vf,
            vv=vv,
            ff=ff,
            batch_idx=batch_idx,
            compute_reductions=False,
        )

    return _global_fire_step


def _make_global_fire_update(strategy: ParallelizationStrategy):
    """Wrap the FIRE optimizer's ``fire_update`` (variable-cell mixing path) so
    its velocity mixing uses the mesh-global ``vf/vv/ff``. The MD position/cell
    step is applied separately (per-atom + replicated cell), so only the mixing
    reduction needs globalizing here."""
    from nvalchemi.dynamics._ops.fire import fire_update as _real  # noqa: PLC0415

    def _global_fire_update(
        velocities,
        forces,
        alpha,
        dt,
        n_steps_positive,
        alpha_start,
        f_alpha,
        dt_min,
        dt_max,
        n_min,
        f_dec,
        f_inc,
        *,
        vf=None,
        vv=None,
        ff=None,
        batch_idx=None,
    ):
        M = alpha.shape[0]
        dev, dtype = velocities.device, velocities.dtype
        if vf is None:
            vf = torch.zeros(M, dtype=dtype, device=dev)
        if vv is None:
            vv = torch.zeros(M, dtype=dtype, device=dev)
        if ff is None:
            ff = torch.zeros(M, dtype=dtype, device=dev)
        if batch_idx is None:
            batch_idx = torch.zeros(velocities.shape[0], dtype=torch.int32, device=dev)
        _fire_local_reductions(velocities, forces, batch_idx, vf, vv, ff, strategy)
        _real(
            velocities,
            forces,
            alpha,
            dt,
            n_steps_positive,
            alpha_start,
            f_alpha,
            dt_min,
            dt_max,
            n_min,
            f_dec,
            f_inc,
            vf=vf,
            vv=vv,
            ff=ff,
            batch_idx=batch_idx,
            compute_reductions=False,
        )

    return _global_fire_update


def _make_global_nhc_chain_update(strategy: ParallelizationStrategy):
    """Wrap the toolkit ``nhc_chain_update`` so it thermostats against the
    mesh-global 2·KE rather than the owned shard's."""
    from nvalchemi.dynamics._ops.nose_hoover import (
        nhc_chain_update as _real,  # noqa: PLC0415
    )

    def _global_nhc_chain_update(
        velocities,
        masses,
        eta,
        eta_dot,
        Q,
        temperature,
        dt,
        ndof,
        ke2,
        total_scale,
        step_scale,
        dt_chain,
        batch_idx,
        compute_ke=True,
    ):
        M = temperature.shape[0]
        local_2ke = torch.zeros(M, dtype=velocities.dtype, device=velocities.device)
        local_2ke.index_add_(
            0, batch_idx.long(), masses * (velocities * velocities).sum(-1)
        )
        strategy.reduce_system(local_2ke, Reduce.SUM)
        ke2.copy_(local_2ke)
        _real(
            velocities,
            masses,
            eta,
            eta_dot,
            Q,
            temperature,
            dt,
            ndof,
            ke2,
            total_scale,
            step_scale,
            dt_chain,
            batch_idx,
            compute_ke=False,
        )

    return _global_nhc_chain_update


def _make_global_kinetic_energy(real_fn: Any, strategy: ParallelizationStrategy):
    """Wrap ``compute_kinetic_energy`` to mesh-reduce the per-shard kinetic energy."""

    def _global_kinetic_energy(velocities, masses, batch_idx, num_systems):
        ke = real_fn(velocities, masses, batch_idx, num_systems)
        strategy.reduce_system(ke, Reduce.SUM)
        return ke

    return _global_kinetic_energy


def _make_global_pressure_tensor(real_fn: Any, strategy: ParallelizationStrategy):
    """Wrap ``compute_pressure_tensor`` so the kinetic term is the mesh-global
    ``Σ m (v ⊗ v)``. The virial term is already global (the forward consolidates
    stress), so only the kinetic tensor is reduced; it is fed back through the
    ops ``compute_kinetic=False`` seam."""

    def _global_pressure_tensor(
        velocities,
        masses,
        virial,
        cell,
        kinetic_tensors,
        pressure_tensors,
        volumes,
        batch_idx,
        compute_kinetic=True,
    ):
        M = virial.shape[0]
        # Local kinetic tensor K[s] = Σ_{i∈s} m_i (v_i ⊗ v_i), vec9 row-major,
        # mesh-reduced then handed to the kernel finalize. On CUDA use the same
        # ops kernel the single-process reference computes K with, so the only
        # DD-vs-bare difference is the reduction order (not torch-vs-kernel
        # summation) — this tightens the barostat trajectory match. On CPU the
        # tiled kernel is unavailable, so fall back to a torch reduction.
        K = torch.zeros(M, 9, dtype=velocities.dtype, device=velocities.device)
        if velocities.is_cuda:
            from nvalchemi.dynamics._ops.npt_nph import (  # noqa: PLC0415
                compute_kinetic_tensor,
            )

            compute_kinetic_tensor(velocities, masses, K, batch_idx)
        else:
            outer = (
                masses.view(-1, 1, 1)
                * velocities.unsqueeze(-1)
                * velocities.unsqueeze(-2)
            ).reshape(velocities.shape[0], 9)
            K.index_add_(0, batch_idx.long(), outer)
        strategy.reduce_system(K, Reduce.SUM)
        kinetic_tensors.copy_(K)
        return real_fn(
            velocities,
            masses,
            virial,
            cell,
            kinetic_tensors,
            pressure_tensors,
            volumes,
            batch_idx,
            compute_kinetic=False,
        )

    return _global_pressure_tensor
