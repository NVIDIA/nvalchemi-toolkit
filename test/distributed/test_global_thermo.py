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
"""Global-thermo reduction wrappers for domain-parallel NHC / NPT / NPH.

These gate the *reduction math* that makes the global-coupled ensembles correct
under ``DomainParallel`` — entirely on CPU with gloo, no GPU needed:

* 1-rank parity: each wrapper equals the bare op when there is nothing to reduce
  (the all-reduce is the identity), and the pressure wrapper's
  ``compute_kinetic=False`` path is CPU-correct (it bypasses the GPU-only tiled
  kinetic-tensor kernel and runs only the finalize).
* 2-rank equivalence: the NHC wrapper applied to two real shards (local 2·KE
  summed across the mesh) reproduces the whole-system single-process update
  exactly — the actual DD correctness claim.

The full ``DomainParallel`` NHC/NPT/NPH trajectory equivalence (bare vs world=1
vs world=2, with a real model + partition + migration) rides the multi-GPU
dynamics gates; NPT/NPH pressure additionally needs a GPU (the kinetic-tensor
kernel the reference path uses is tiled/GPU-only).
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from nvalchemi.distributed import _dynamics_coordinator as gt


def _halo_strategy(mesh=None, rank=0):
    """Real :class:`HaloStrategy` for the reduction wrappers. ``mesh=None`` →
    ``reduce_system`` is the identity (1-rank parity); a 2-rank CPU mesh → an
    all_reduce SUM over that mesh's gloo group."""
    from nvalchemi.distributed._core.storage_policy import HaloStoragePolicy
    from nvalchemi.distributed.config import DomainConfig
    from nvalchemi.distributed.strategy import HaloStrategy

    return HaloStrategy(HaloStoragePolicy(), DomainConfig(cutoff=5.0, mesh=mesh), rank)


def _nhc_inputs(vel, mass, batch_idx, M, ndof, *, ke2=None):
    """Build the argument bundle for ``nhc_chain_update`` from fixed state."""
    from nvalchemi.dynamics._ops.nose_hoover import nhc_compute_masses

    dtype = vel.dtype
    temp = torch.full((M,), 0.02585, dtype=dtype)  # ~300 K in eV
    tau = torch.full((M,), 20.0, dtype=dtype)
    Q = nhc_compute_masses(temp, tau, mass, batch_idx.int(), 3)
    # Q_0 depends on ndof; force the supplied (global) ndof so shard runs agree.
    Q[:, 0] = ndof * temp * tau * tau
    dt = torch.full((M,), 0.5, dtype=dtype)
    zeros = lambda: torch.zeros(M, dtype=dtype)  # noqa: E731
    ke2 = zeros() if ke2 is None else ke2
    return dict(
        eta=torch.zeros(M, 3, dtype=dtype),
        eta_dot=torch.zeros(M, 3, dtype=dtype),
        Q=Q,
        temperature=temp,
        dt=dt,
        ndof=ndof,
        ke2=ke2,
        total_scale=zeros(),
        step_scale=zeros(),
        dt_chain=zeros(),
    )


# ----------------------------------------------------------------------
# 1-rank parity (session gloo fixture supplies the default group)
# ----------------------------------------------------------------------


def test_nhc_wrapper_matches_bare_single_rank(_session_gloo_pg) -> None:
    """On 1 rank the all-reduce is the identity, so the global wrapper must
    reproduce the bare ``nhc_chain_update`` exactly."""
    from nvalchemi.dynamics._ops.nose_hoover import nhc_chain_update

    torch.manual_seed(1)
    N, M = 8, 1
    vel0 = torch.randn(N, 3, dtype=torch.float64)
    mass = (torch.rand(N, dtype=torch.float64) + 0.5) * 12.0
    bidx = torch.zeros(N, dtype=torch.long)
    ndof = torch.full((M,), 3.0 * N, dtype=torch.float64)

    v = vel0.clone()
    args = _nhc_inputs(v, mass, bidx, M, ndof)
    nhc_chain_update(
        v,
        mass,
        args["eta"],
        args["eta_dot"],
        args["Q"],
        args["temperature"],
        args["dt"],
        args["ndof"],
        args["ke2"],
        args["total_scale"],
        args["step_scale"],
        args["dt_chain"],
        bidx.int(),
        compute_ke=True,
    )
    v_bare = v.clone()

    v = vel0.clone()
    args = _nhc_inputs(v, mass, bidx, M, ndof)
    gt._make_global_nhc_chain_update(_halo_strategy())(
        v,
        mass,
        args["eta"],
        args["eta_dot"],
        args["Q"],
        args["temperature"],
        args["dt"],
        args["ndof"],
        args["ke2"],
        args["total_scale"],
        args["step_scale"],
        args["dt_chain"],
        bidx.int(),
    )
    assert torch.equal(v_bare, v)


def test_kinetic_energy_wrapper_matches_bare_single_rank(_session_gloo_pg) -> None:
    from nvalchemi.dynamics._ops.thermostat_utils import compute_kinetic_energy

    torch.manual_seed(2)
    N, M = 8, 1
    vel = torch.randn(N, 3, dtype=torch.float64)
    mass = (torch.rand(N, dtype=torch.float64) + 0.5) * 12.0
    bidx = torch.zeros(N, dtype=torch.long)
    bare = compute_kinetic_energy(vel, mass, bidx.int(), M)
    wrapped = gt._make_global_kinetic_energy(compute_kinetic_energy, _halo_strategy())(
        vel, mass, bidx.int(), M
    )
    assert torch.equal(bare, wrapped)


def test_pressure_wrapper_compute_kinetic_false_cpu(_session_gloo_pg) -> None:
    """The pressure wrapper builds the kinetic tensor in torch and feeds it via
    ``compute_kinetic=False``, which runs only the (non-tiled) finalize kernel —
    so it is correct on CPU, unlike the default tiled kinetic-tensor path."""
    from nvalchemi.dynamics._ops.npt_nph import compute_pressure_tensor

    torch.manual_seed(3)
    N, M = 8, 1
    vel = torch.randn(N, 3, dtype=torch.float64)
    mass = (torch.rand(N, dtype=torch.float64) + 0.5) * 12.0
    bidx = torch.zeros(N, dtype=torch.long)
    cell = torch.eye(3, dtype=torch.float64).unsqueeze(0) * 10.0
    vol = torch.linalg.det(cell).abs()
    virial = torch.randn(M, 3, 3, dtype=torch.float64)
    virial = (virial + virial.transpose(-1, -2)) / 2
    kt = torch.zeros(M, 9, dtype=torch.float64)
    pt = torch.zeros(M, 9, dtype=torch.float64)

    P = gt._make_global_pressure_tensor(compute_pressure_tensor, _halo_strategy())(
        vel, mass, virial, cell, kt, pt, vol.clone(), bidx.int()
    )
    K = (mass.view(-1, 1, 1) * vel.unsqueeze(-1) * vel.unsqueeze(-2)).sum(0)
    P_ref = ((K + virial[0]) / vol[0]).reshape(9)
    torch.testing.assert_close(P[0], P_ref, rtol=1e-12, atol=1e-12)


# ----------------------------------------------------------------------
# 2-rank equivalence: real cross-shard reduction == whole-system update
# ----------------------------------------------------------------------


def _nhc_2rank_worker(rank: int, world_size: int) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29687"
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
    try:
        from nvalchemi.dynamics._ops.nose_hoover import nhc_chain_update

        # Both ranks build the SAME full system (fixed seed).
        torch.manual_seed(7)
        N, M = 8, 1
        vel0 = torch.randn(N, 3, dtype=torch.float64)
        mass = (torch.rand(N, dtype=torch.float64) + 0.5) * 12.0
        bidx_full = torch.zeros(N, dtype=torch.long)
        ndof = torch.full((M,), 3.0 * N, dtype=torch.float64)  # GLOBAL dof

        # Reference: whole-system bare update.
        v_full = vel0.clone()
        a = _nhc_inputs(v_full, mass, bidx_full, M, ndof)
        nhc_chain_update(
            v_full,
            mass,
            a["eta"],
            a["eta_dot"],
            a["Q"],
            a["temperature"],
            a["dt"],
            a["ndof"],
            a["ke2"],
            a["total_scale"],
            a["step_scale"],
            a["dt_chain"],
            bidx_full.int(),
            compute_ke=True,
        )

        # DD: this rank owns half the atoms; the wrapper sums 2*KE across ranks.
        lo, hi = (0, N // 2) if rank == 0 else (N // 2, N)
        v_sh = vel0[lo:hi].clone().contiguous()
        m_sh = mass[lo:hi].contiguous()
        bidx_sh = torch.zeros(hi - lo, dtype=torch.long)
        a = _nhc_inputs(v_sh, m_sh, bidx_sh, M, ndof)
        # Q must be the GLOBAL-N thermostat mass, identical on both ranks; rebuild
        # Q_0 from global ndof (already done in _nhc_inputs via ndof) but recompute
        # the per-shard masses entry isn't ndof-dependent, so it matches.
        # A real HaloStrategy over the 2-rank CPU mesh does the cross-shard 2·KE
        # all_reduce SUM (routing the reduction through the strategy verb).
        from torch.distributed.device_mesh import DeviceMesh

        mesh = DeviceMesh("cpu", list(range(world_size)), mesh_dim_names=("domain",))
        gt._make_global_nhc_chain_update(_halo_strategy(mesh=mesh, rank=rank))(
            v_sh,
            m_sh,
            a["eta"],
            a["eta_dot"],
            a["Q"],
            a["temperature"],
            a["dt"],
            a["ndof"],
            a["ke2"],
            a["total_scale"],
            a["step_scale"],
            a["dt_chain"],
            bidx_sh.int(),
        )
        torch.testing.assert_close(v_sh, v_full[lo:hi], rtol=1e-12, atol=1e-12)
    finally:
        dist.destroy_process_group()


def test_nhc_global_reduction_2ranks() -> None:
    """Across two real gloo ranks, the global-2*KE wrapper applied to each shard
    reproduces the whole-system single-process velocity update exactly."""
    mp.spawn(_nhc_2rank_worker, args=(2,), nprocs=2)
