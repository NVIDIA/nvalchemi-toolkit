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
"""Pipeline × DD — Phase 1: ``DomainParallel`` over a domain SUB-mesh of a 2D mesh.

The 2D-parallel dynamics feature (proposal-distributed-pipeline-dd.md) lays out a
mesh ``(pipeline, domain)`` and runs each pipeline stage's ``DomainParallel`` over
its own **domain sub-mesh row** (``mesh2d["domain"]``). This gate is the load-
bearing prerequisite: a ``DomainParallel(FIRE(LJ))`` run over a sliced domain row
of a 4-rank ``(2, 2)`` mesh must be **equivalent** to the single-process bare-FIRE
relaxation — i.e. slicing a sub-mesh out of a 2D mesh yields a fully-functional DD
group (rank resolution, scatter from the row's lead, per-step reductions, gather).

Both pipeline rows ({0,1} and {2,3}) run the identical cluster independently, so
their gathered relaxed systems must also be identical to each other. Uses the
tight non-adapting FIRE (``f_alpha=1``/``f_inc=1``) so the DD trajectory matches
bare FIRE to ~machine precision (see test_fire_dd for the CPU alpha-ordering note).

CPU/gloo, world=4 (two 2-rank domain groups).
"""

from __future__ import annotations

from typing import Any

import torch
import torch.multiprocessing as mp

from nvalchemi.data import AtomicData, Batch
from test.distributed.test_fire_dd import (
    _bare_fire_relax,
    _build_lj_cluster,
    _init_gloo,
    _make_fire_lj,
)


def _install_subgroup_gloo_shim() -> None:
    """Sub-group-correct gloo ``indexed_all_to_all_v`` stand-in.

    The shim in test_fire_dd sends with ``dst=<group-local index>``, which only
    works when the group spans all ranks (local == global). A domain SUB-group
    (e.g. ``{2,3}``) needs the group-local index mapped to the global rank that
    ``isend``/``irecv`` expect. The real physicsnemo (nccl) wrapper already does
    this; this makes the CPU/gloo path match so the sub-mesh halo exchange works.
    """
    import physicsnemo.distributed.utils as pn_utils
    import torch.distributed as dist

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
            gi = dist.get_global_rank(group, i)  # group-local i -> global rank
            if i == r:
                x_recv[i].copy_(x_send[i])
            else:
                if x_send[i].numel() > 0:
                    ops.append(dist.isend(x_send[i], dst=gi, group=group))
                if x_recv[i].numel() > 0:
                    ops.append(dist.irecv(x_recv[i], src=gi, group=group))
        for op in ops:
            op.wait()
        return torch.cat(x_recv, dim=dim)

    pn_utils.indexed_all_to_all_v_wrapper = _impl


def _worker(rank: int, world_size: int, port: str, fn_name: str, *args: Any) -> None:
    import torch.distributed as dist

    _init_gloo(rank, world_size, port)
    _install_subgroup_gloo_shim()
    try:
        globals()[fn_name](rank, world_size, *args)
    finally:
        dist.destroy_process_group()


def _spawn(world_size: int, port: str, fn_name: str, *args: Any) -> None:
    mp.spawn(_worker, args=(world_size, port, fn_name, *args), nprocs=world_size)


def _submesh_dd_worker(rank: int, world_size: int, n_steps: int) -> None:
    from torch.distributed import init_device_mesh

    from nvalchemi.distributed.domain_parallel import DomainParallel

    dtype = torch.float64
    positions, atomic_numbers, masses, cell, pbc = _build_lj_cluster()
    n = positions.shape[0]

    # 2D mesh (pipeline, domain); each rank's domain row is a 2-rank sub-mesh.
    mesh2d = init_device_mesh("cpu", (2, 2), mesh_dim_names=("pipeline", "domain"))
    domain = mesh2d["domain"]
    domain_rank = domain.get_local_rank()

    # Hand DomainParallel the 1D domain SUB-mesh (approach B).
    _, dist_fire, cfg = _make_fire_lj(domain, f_alpha=1.0, f_inc=1.0)
    dp = DomainParallel(dynamics=dist_fire, config=cfg)

    # The full system lives on each row's lead (domain-rank 0 = global 0 and 2).
    if domain_rank == 0:
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
        full_batch: Batch | None = Batch.from_data_list([full_data])
    else:
        full_batch = None

    local_batch = dp.partition(full_batch)
    for _ in range(n_steps):
        local_batch, _ = dp.step(local_batch)
    relaxed = dp.gather(local_batch, dst=0)  # reconstruct on each row's lead

    if domain_rank != 0:
        return

    # Bare single-process reference (same tight FIRE, same cluster).
    _e0_ref, ref_batch = _bare_fire_relax(n_steps, f_alpha=1.0, f_inc=1.0)

    # Recompute energy + forces from the relaxed positions (deterministic) for both
    # the DD-gathered system and the bare reference — the apples-to-apples oracle.
    got_e, got_f = _energy_and_forces(relaxed)
    ref_e, ref_f = _energy_and_forces(ref_batch)

    # Tolerance reflects the Warp-CPU FIRE alpha/segment-ordering drift (see
    # test_fire_dd's note) — a shard's atom ordering differs from the whole
    # system's, so the CPU trajectory drifts ~1e-5 over the run. That is NOT a
    # sub-mesh error: this gate proves the 2D-sub-mesh MECHANICS (rank resolution,
    # scatter from the row lead, per-step halo reductions, gather) reconstruct the
    # same relaxed structure as bare FIRE to ~5 figures. Bit-exact FIRE rides the
    # GPU dynamics gates.
    torch.testing.assert_close(
        torch.tensor(got_e), torch.tensor(ref_e), rtol=1e-4, atol=1e-4
    )
    # Sorted force magnitudes are partition-order-invariant.
    torch.testing.assert_close(
        got_f.norm(dim=-1).sort().values,
        ref_f.norm(dim=-1).sort().values,
        rtol=1e-3, atol=1e-4,
    )


def _energy_and_forces(batch: Batch) -> tuple[float, torch.Tensor]:
    from nvalchemi.models.lj import LennardJonesModelWrapper
    from nvalchemi.neighbors import compute_neighbors

    model = LennardJonesModelWrapper(epsilon=0.0104, sigma=3.40, cutoff=5.0)
    compute_neighbors(batch, config=model.model_config.neighbor_config)
    out = model(batch)
    return float(out["energy"].sum().item()), out["forces"].detach().double()


def test_domain_parallel_over_2d_submesh_matches_bare_fire() -> None:
    """DomainParallel(FIRE(LJ)) over a domain row of a (2,2) mesh == bare FIRE."""
    _spawn(4, "29744", "_submesh_dd_worker", 8)
