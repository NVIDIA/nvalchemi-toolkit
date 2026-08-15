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
"""FIRE geometry optimizer under :class:`DomainParallel` — CPU/gloo gates.

FIRE's velocity mixing and timestep adaptation are gated by *global* per-system
power/norm scalars (``v·f``, ``v·v``, ``f·f``) summed over **all** atoms. Under
domain decomposition each rank owns a spatial slice, so left alone every rank
mixes against a per-shard power and the replicated relaxation desyncs. The
dynamics coordinator globalizes those reductions (SUM over the mesh) and feeds
them back via ``fire_step(..., compute_reductions=False)``.

Three levels of gate, all on CPU with gloo:

* **reduction math (exact)** — the coordinator's ``_make_global_fire_step`` /
  ``_make_global_fire_update`` wrappers, applied to two real shards, feed the
  ops kernel the *whole-system* ``vf/vv/ff`` (the local partials summed across
  the mesh). We assert the globalized scalars equal the single-process sum
  bit-for-bit — this is the direct DD correctness claim the coordinator owns.
* **end-to-end descent** — default (adapting-alpha) ``DomainParallel(FIRE(LJ))``
  runs to completion and relaxes the cluster (global energy descends well below
  the start).
* **same-geometry force correctness** — save every gathered frame from a
  distributed FIRE trajectory, then replay those exact positions through
  single-process LJ. Distributed energy and forces must match the replay,
  including frames where ownership migration is deferred.

The tests do not compare independently evolved trajectories: small
floating-point differences can accumulate even when each force evaluation is
correct.

.. note:: ``_bare_fire_relax`` remains available for the 2-D pipeline tests,
   which configure ``f_alpha=1`` and ``f_inc=1`` when comparing CPU trajectories.
   This avoids the Warp CPU kernel's atom-order-dependent in-place updates of
   per-system ``alpha`` and ``dt``. The same-geometry gate here does not depend
   on trajectory identity.
"""

from __future__ import annotations

import os
from typing import Any

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed import DeviceMesh

from nvalchemi.data import AtomicData, Batch

# ======================================================================
# gloo harness (mirrors test_domain_parallel.py)
# ======================================================================


def _init_gloo(rank: int, world_size: int, port: str) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = port
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
    # physicsnemo's gloo all-to-all shim (reused from the distributed tests).
    import physicsnemo.distributed.utils as pn_utils

    def _impl(tensor, indices, sizes, dim=0, group=None):
        cs = dist.get_world_size(group=group)
        r = dist.get_rank(group=group)
        x_send = [tensor[idx].contiguous() for idx in indices]
        x_recv = []
        shape = list(tensor.shape)
        for i in range(cs):
            shape[dim] = sizes[i][r]
            x_recv.append(torch.empty(shape, dtype=tensor.dtype, device=tensor.device))
        ops = []
        for i in range(cs):
            if i == r:
                x_recv[i].copy_(x_send[i])
            else:
                if x_send[i].numel() > 0:
                    ops.append(dist.isend(x_send[i], dst=i, group=group))
                if x_recv[i].numel() > 0:
                    ops.append(dist.irecv(x_recv[i], src=i, group=group))
        for op in ops:
            op.wait()
        return torch.cat(x_recv, dim=dim)

    pn_utils.indexed_all_to_all_v_wrapper = _impl


def _worker(rank: int, world_size: int, port: str, fn_name: str, *args: Any) -> None:
    _init_gloo(rank, world_size, port)
    try:
        globals()[fn_name](rank, world_size, *args)
    finally:
        dist.destroy_process_group()


def _spawn(world_size: int, port: str, fn_name: str, *args: Any) -> None:
    mp.spawn(_worker, args=(world_size, port, fn_name, *args), nprocs=world_size)


def _halo_strategy(mesh, rank):
    """Real :class:`HaloStrategy` for the reduction wrappers; a 2-rank CPU mesh
    routes ``reduce_system`` through an all_reduce SUM over its gloo group."""
    from nvalchemi.distributed._core.storage_policy import HaloStoragePolicy
    from nvalchemi.distributed.config import DomainConfig as _DC
    from nvalchemi.distributed.strategy import HaloStrategy

    return HaloStrategy(HaloStoragePolicy(), _DC(cutoff=5.0, mesh=mesh), rank)


# ======================================================================
# Level 1 — reduction math: the coordinator globalizes vf/vv/ff exactly.
#
# The coordinator wrapper is the only place the DD reduction lives; we capture
# the vf/vv/ff it hands the ops kernel (via compute_reductions=False) and assert
# it equals the whole-system single-process sum, bit-for-bit.
# ======================================================================


def _fire_state(M, dtype):
    z = lambda v: torch.full((M,), v, dtype=dtype)  # noqa: E731
    return dict(
        alpha=z(0.1),
        dt=z(1.0),
        n_steps_positive=torch.full((M,), 6, dtype=torch.int32),
        alpha_start=z(0.1),
        f_alpha=z(0.99),
        dt_min=z(0.02),
        dt_max=z(10.0),
        maxstep=z(0.2),
        n_min=torch.full((M,), 5, dtype=torch.int32),
        f_dec=z(0.5),
        f_inc=z(1.1),
        uphill_flag=torch.zeros(M, dtype=torch.int32),
    )


class _CaptureFireStep:
    """Stand-in ops ``fire_step`` that records the vf/vv/ff the coordinator
    wrapper passes with ``compute_reductions=False`` (no actual FIRE update)."""

    captured: dict[str, torch.Tensor] = {}

    def __call__(self, *args, vf=None, vv=None, ff=None, batch_idx=None, **kw):
        assert kw.get("compute_reductions") is False
        _CaptureFireStep.captured = {
            "vf": vf.clone(),
            "vv": vv.clone(),
            "ff": ff.clone(),
        }


def _fire_reduction_worker(rank: int, world_size: int) -> None:
    import nvalchemi.distributed._dynamics_coordinator as dcm

    dtype = torch.float64
    torch.manual_seed(11)
    N, M = 12, 1
    vel0 = torch.randn(N, 3, dtype=dtype)
    frc = torch.randn(N, 3, dtype=dtype)

    # Whole-system reference sums.
    vf_ref = (frc * vel0).sum()
    vv_ref = (vel0 * vel0).sum()
    ff_ref = (frc * frc).sum()

    # This rank owns half the atoms; the wrapper reduces its local partials
    # (SUM) across the mesh and feeds the global result to the (captured) kernel.
    lo, hi = (0, N // 2) if rank == 0 else (N // 2, N)
    v_sh = vel0[lo:hi].clone().contiguous()
    f_sh = frc[lo:hi].clone().contiguous()
    m_sh = torch.ones(hi - lo, dtype=dtype)
    st = _fire_state(M, dtype)

    mesh = DeviceMesh("cpu", list(range(world_size)), mesh_dim_names=("domain",))
    strat = _halo_strategy(mesh, rank)

    # The factory closes over ``fire_step`` imported from the ops-binding module;
    # patch that symbol with a capturing stub so we can inspect the global
    # vf/vv/ff the wrapper feeds it (no actual FIRE update is performed).
    import nvalchemi.dynamics._ops.fire as fire_ops

    saved = fire_ops.fire_step
    fire_ops.fire_step = _CaptureFireStep()
    try:
        wrapped = dcm._make_global_fire_step(strat)
        wrapped(
            vel0[lo:hi].clone(),
            v_sh,
            f_sh,
            m_sh,
            st["alpha"],
            st["dt"],
            st["n_steps_positive"],
            st["alpha_start"],
            st["f_alpha"],
            st["dt_min"],
            st["dt_max"],
            st["maxstep"],
            st["n_min"],
            st["f_dec"],
            st["f_inc"],
            st["uphill_flag"],
            batch_idx=torch.zeros(hi - lo, dtype=torch.int32),
        )
    finally:
        fire_ops.fire_step = saved

    cap = _CaptureFireStep.captured
    torch.testing.assert_close(cap["vf"][0], vf_ref, rtol=0, atol=1e-12)
    torch.testing.assert_close(cap["vv"][0], vv_ref, rtol=0, atol=1e-12)
    torch.testing.assert_close(cap["ff"][0], ff_ref, rtol=0, atol=1e-12)


def test_fire_coordinator_globalizes_reductions_2ranks() -> None:
    """The coordinator's ``fire_step`` wrapper hands the ops kernel the
    whole-system vf/vv/ff (each rank's owned partial summed across the mesh)."""
    _spawn(2, "29740", "_fire_reduction_worker")


# ======================================================================
# Level 2 — end-to-end: DomainParallel(FIRE(LJ)) relaxes to the bare minimum.
# ======================================================================


def _build_lj_cluster(n_per_side: int = 6, dtype: torch.dtype = torch.float64):
    """Open-cell argon cluster, perturbed off the minimum so FIRE has a
    non-trivial relaxation. Box sized to the atoms so the spatial bisection puts
    real owned atoms on every rank; ``n_per_side=6`` (216 atoms) with the 5 Å
    LJ cutoff below is wide enough that each rank has genuinely-remote atoms (not
    a degenerate full-halo partition)."""
    spacing = 2 ** (1.0 / 6.0) * 3.40 * 1.05
    coords = torch.arange(n_per_side, dtype=dtype) * spacing
    gx, gy, gz = torch.meshgrid(coords, coords, coords, indexing="ij")
    positions = torch.stack([gx.flatten(), gy.flatten(), gz.flatten()], dim=-1)
    positions = positions + 0.05 * torch.randn(
        positions.shape, dtype=dtype, generator=torch.Generator().manual_seed(3)
    )
    n = positions.shape[0]
    atomic_numbers = torch.full((n,), 18, dtype=torch.long)
    masses = torch.full((n,), 39.948, dtype=dtype)
    box = (n_per_side - 1) * spacing + spacing
    cell = torch.eye(3, dtype=dtype) * box
    pbc = torch.zeros(3, dtype=torch.bool)
    return positions, atomic_numbers, masses, cell, pbc


def _make_fire_lj(mesh_or_none, cutoff=5.0, f_alpha=0.99, f_inc=1.1):
    from nvalchemi.distributed.config import DomainConfig as _DC
    from nvalchemi.dynamics.base import DynamicsStage
    from nvalchemi.dynamics.optimizers.fire import FIRE
    from nvalchemi.hooks.neighbor_list import NeighborListHook
    from nvalchemi.models.lj import LennardJonesModelWrapper

    model = LennardJonesModelWrapper(epsilon=0.0104, sigma=3.40, cutoff=cutoff)
    fire = FIRE(
        model=model,
        dt=2.0,
        maxstep=0.2,
        f_alpha=f_alpha,
        f_inc=f_inc,
        hooks=[
            NeighborListHook(
                config=model.model_config.neighbor_config,
                skin=0.0,
                stage=DynamicsStage.BEFORE_COMPUTE,
            )
        ],
    )
    cfg = None
    if mesh_or_none is not None:
        cfg = _DC(cutoff=float(model.cutoff), skin=0.0, mesh=mesh_or_none)
    return model, fire, cfg


def _bare_fire_relax(n_steps: int, f_alpha=0.99, f_inc=1.1):
    """Single-process bare-FIRE reference relaxation → (E0, final_batch)."""
    from nvalchemi.neighbors import compute_neighbors

    dtype = torch.float64
    positions, atomic_numbers, masses, cell, pbc = _build_lj_cluster()
    n = positions.shape[0]
    model, fire, _ = _make_fire_lj(None, f_alpha=f_alpha, f_inc=f_inc)
    data = AtomicData(
        atomic_numbers=atomic_numbers,
        positions=positions.clone(),
        atomic_masses=masses,
        forces=torch.zeros(n, 3, dtype=dtype),
        energy=torch.zeros(1, 1, dtype=dtype),
        cell=cell.unsqueeze(0),
        pbc=pbc.unsqueeze(0),
    )
    data.add_node_property("velocities", torch.zeros(n, 3, dtype=dtype))
    batch = Batch.from_data_list([data])
    fire._ensure_state_initialized(batch)
    compute_neighbors(batch, config=model.model_config.neighbor_config)
    batch.forces = model(batch)["forces"].detach()
    e0 = float(model(batch)["energy"].sum().item())
    for _ in range(n_steps):
        batch, _ = fire.step(batch)
    return e0, batch


def _fire_e2e_worker(rank: int, world_size: int, n_steps: int) -> None:
    from nvalchemi.distributed.domain_parallel import DomainParallel

    dtype = torch.float64
    positions, atomic_numbers, masses, cell, pbc = _build_lj_cluster()
    n = positions.shape[0]

    mesh = DeviceMesh("cpu", list(range(world_size)), mesh_dim_names=("domain",))
    _, dist_fire, cfg = _make_fire_lj(mesh)
    dp = DomainParallel(dynamics=dist_fire, config=cfg)

    if rank == 0:
        full_data = AtomicData(
            atomic_numbers=atomic_numbers,
            positions=positions.clone(),
            atomic_masses=masses,
            forces=torch.zeros(n, 3, dtype=dtype),
            energy=torch.zeros(1, 1, dtype=dtype),
            cell=cell.unsqueeze(0),
            pbc=pbc.unsqueeze(0),
        )
        full_data.add_node_property("velocities", torch.zeros(n, 3, dtype=dtype))
        full_batch = Batch.from_data_list([full_data])
    else:
        full_batch = None
    local_batch = dp.partition(full_batch)

    # The DistributedModel consolidates energy during ``step``; the first step's
    # energy is the (near-)initial value, the last step's the relaxed value.
    energies = []
    for _ in range(n_steps):
        local_batch, _ = dp.step(local_batch)
        energies.append(float(local_batch.energy.sum().item()))

    if rank != 0:
        return

    # Default (adapting-alpha) FIRE under DD must run to completion and relax:
    # the (global, forward-consolidated) energy descends well below the start.
    e_first, e_final = energies[0], energies[-1]
    assert e_final < e_first - 0.3, (
        f"DomainParallel(FIRE) did not relax: E_first={e_first:.6f} "
        f"E_final={e_final:.6f}"
    )
    # Relaxation should be broadly downhill (allow small FIRE overshoots).
    assert e_final <= min(energies[: max(1, n_steps // 4)]), (
        "DomainParallel(FIRE) energy did not decrease over the run"
    )


def test_fire_lj_2ranks_end_to_end() -> None:
    """``DomainParallel(FIRE(LJ))`` runs to completion on a genuinely-decomposed
    argon cluster and relaxes it (global energy descends well below the start)."""
    _spawn(2, "29742", "_fire_e2e_worker", 200)


# ======================================================================
# Level 3 — same-geometry correctness through a FIRE trajectory.
#
# The trajectory deliberately places atoms near the rank split and reaches
# deferred-migration states. Save every gathered DD frame, then replay those
# fixed positions through a single-process LJ model. This checks distributed
# energy and forces without comparing two separately evolved trajectories.
# ======================================================================


def _fire_same_geometry_worker(rank: int, world_size: int, n_steps: int) -> None:
    from nvalchemi.distributed.domain_parallel import DomainParallel
    from nvalchemi.neighbors import compute_neighbors

    dtype = torch.float64
    positions, atomic_numbers, masses, cell, pbc = _build_lj_cluster()
    n = positions.shape[0]

    mesh = DeviceMesh("cpu", list(range(world_size)), mesh_dim_names=("domain",))
    _, dist_fire, cfg = _make_fire_lj(mesh, f_alpha=1.0, f_inc=1.0)
    dp = DomainParallel(dynamics=dist_fire, config=cfg)

    if rank == 0:
        full_data = AtomicData(
            atomic_numbers=atomic_numbers,
            positions=positions.clone(),
            atomic_masses=masses,
            forces=torch.zeros(n, 3, dtype=dtype),
            energy=torch.zeros(1, 1, dtype=dtype),
            cell=cell.unsqueeze(0),
            pbc=pbc.unsqueeze(0),
        )
        full_data.add_node_property("velocities", torch.zeros(n, 3, dtype=dtype))
        full_batch = Batch.from_data_list([full_data])
    else:
        full_batch = None
    local_batch = dp.partition(full_batch)
    assert dp._sharded_batch is not None
    assert dp._sharded_batch.partitioner is not None

    saw_deferred_owner = torch.zeros(1, dtype=torch.int32)
    trajectory: list[tuple[Batch, torch.Tensor, torch.Tensor]] = []

    for _ in range(n_steps):
        local_batch, _ = dp.step(local_batch)
        natural_owner = dp._sharded_batch.partitioner.assign_atoms_to_ranks(
            local_batch.positions
        )
        saw_deferred_owner |= (natural_owner != rank).any().to(torch.int32).view(1)

        distributed_energy = local_batch.energy.sum().detach().clone()
        full_step = dp.gather(local_batch, dst=0)

        if rank != 0:
            continue

        assert full_step is not None
        assert full_step.num_nodes == n
        trajectory.append(
            (
                full_step,
                distributed_energy,
                full_step.forces.detach().clone(),
            )
        )

    # Finish every step/gather collective before replaying frames on rank 0.
    dist.all_reduce(saw_deferred_owner, op=dist.ReduceOp.MAX)
    if rank != 0:
        return

    assert bool(saw_deferred_owner.item()), (
        "the FIRE fixture never entered a deferred-ownership state"
    )
    assert len(trajectory) == n_steps

    # Replay the saved DD trajectory without advancing another FIRE integrator:
    # each reference calculation sees exactly the positions from that DD frame.
    reference_model, _, _ = _make_fire_lj(None, f_alpha=1.0, f_inc=1.0)
    for step, (
        reference_batch,
        distributed_energy,
        distributed_forces,
    ) in enumerate(trajectory, start=1):
        compute_neighbors(
            reference_batch,
            config=reference_model.model_config.neighbor_config,
        )
        reference = reference_model(reference_batch)

        reference_energy = reference["energy"].sum().detach()
        reference_forces = reference["forces"].detach()
        torch.testing.assert_close(
            distributed_energy,
            reference_energy,
            rtol=1e-10,
            atol=1e-12,
            msg=f"distributed energy differs at the same geometry on step {step}",
        )
        torch.testing.assert_close(
            distributed_forces,
            reference_forces,
            rtol=1e-10,
            atol=1e-12,
            msg=f"distributed forces differ at the same geometry on step {step}",
        )


def test_fire_lj_2ranks_same_geometry() -> None:
    """Replay a saved DD FIRE trajectory through single-process LJ and compare
    energy and forces at every identical-position frame."""
    _spawn(2, "29743", "_fire_same_geometry_worker", 30)
