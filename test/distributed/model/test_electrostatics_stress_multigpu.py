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
"""Multi-GPU stress equivalence for Ewald / PME under halo decomposition.

Stress is taken by strain-autograd of the consolidated global energy, so each
rank holds a partial that the declared reduction sums. These tests pin the
result against a single-GPU reference on a partition asserted non-degenerate —
a degenerate split gives every rank the whole system and hides any reduction
error entirely.
"""

from __future__ import annotations

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from _dd_harness import free_port
from _dd_harness import nccl_worker as _worker
from _electrostatics import build_nacl

from nvalchemi.data import AtomicData, Batch
from nvalchemi.distributed.config import DomainConfig

WORLD_SIZE = 2
N_SIDE = 10
BOX = 28.0


def _build_wrapper(kind: str, cutoff: float):
    """Wrapper of the requested flavour on the supported (non-analytic) path."""
    if kind == "pme":
        from nvalchemi.models.pme import PMEModelWrapper

        return PMEModelWrapper(cutoff=cutoff, hybrid_forces=False)
    from nvalchemi.models.ewald import EwaldModelWrapper

    return EwaldModelWrapper(cutoff=cutoff, hybrid_forces=False)


def _make_batch(positions, atomic_numbers, masses, charges, cell, pbc, device, dtype):
    """Single-system Batch on *device*."""
    n = positions.shape[0]
    data = AtomicData(
        atomic_numbers=atomic_numbers.to(device),
        positions=positions.to(device=device, dtype=dtype).clone(),
        atomic_masses=masses.to(device=device, dtype=dtype),
        charges=charges.to(device=device, dtype=dtype),
        cell=cell.to(device=device, dtype=dtype).unsqueeze(0),
        pbc=pbc.to(device).unsqueeze(0),
        forces=torch.zeros(n, 3, device=device, dtype=dtype),
        energy=torch.zeros(1, 1, device=device, dtype=dtype),
    )
    return Batch.from_data_list([data])


def _stress_equivalence_worker(rank: int, world_size: int, kind: str) -> None:
    """Distributed analytic stress must equal the single-GPU stress."""
    from torch.distributed import DeviceMesh

    from nvalchemi.distributed.distributed_model import DistributedModel
    from nvalchemi.distributed.sharded_batch import ShardedBatch
    from nvalchemi.neighbors import compute_neighbors

    dtype = torch.float64
    device = torch.device(f"cuda:{rank}")
    positions, atomic_numbers, masses, charges, cell, pbc = build_nacl(
        N_SIDE, BOX, dtype=dtype
    )
    n_global = positions.shape[0]
    cutoff = min(5.0, 0.45 * cell[0, 0].item())
    args = (positions, atomic_numbers, masses, charges, cell, pbc)

    # ---- Single-process reference on rank 0, broadcast to all ----
    s_ref = torch.zeros(1, 3, 3, device=device, dtype=dtype)
    e_ref = torch.zeros(1, device=device, dtype=dtype)
    if rank == 0:
        ref_wrapper = _build_wrapper(kind, cutoff).to(device)
        ref_wrapper.model_config.active_outputs = {"energy", "forces", "stress"}
        ref_batch = _make_batch(*args, device, dtype)
        compute_neighbors(ref_batch, config=ref_wrapper.model_config.neighbor_config)
        ref_out = ref_wrapper(ref_batch)
        s_ref = ref_out["stress"].detach().reshape(1, 3, 3).clone()
        e_ref = ref_out["energy"].sum().detach().reshape(1).clone()
        del ref_wrapper, ref_batch, ref_out
    dist.broadcast(s_ref, src=0)
    dist.broadcast(e_ref, src=0)

    # ---- Distributed forward ----
    mesh = DeviceMesh("cuda", list(range(world_size)), mesh_dim_names=("domain",))
    domain_config = DomainConfig(
        cutoff=cutoff, skin=0.0, mesh=mesh, require_nondegenerate=True
    )
    wrapper = _build_wrapper(kind, cutoff).to(device)
    wrapper.model_config.active_outputs = {"energy", "forces", "stress"}

    full_batch = _make_batch(*args, device, dtype) if rank == 0 else None
    sharded = ShardedBatch.from_batch(
        batch=full_batch, mesh=mesh, config=domain_config, src=0
    )
    # A degenerate partition would hand every rank the whole system and pass
    # regardless of how the virial is reduced.
    assert 0 < sharded.n_owned < n_global, (
        f"rank {rank}: degenerate partition ({sharded.n_owned} of {n_global} owned); "
        "the test would not exercise the cross-rank reduction"
    )

    with DistributedModel(wrapper, domain_config) as dist_model:
        out = dist_model(sharded)

    s_dist = out["stress"].detach().reshape(1, 3, 3)
    e_dist = out["energy"].sum().detach().reshape(1)
    print(
        f"[{kind}-stress rank {rank}] n_owned={sharded.n_owned}/{n_global} "
        f"dE={(e_dist - e_ref).abs().max().item():.3e} "
        f"dS={(s_dist - s_ref).abs().max().item():.3e}\n"
        f"  dist diag={torch.diagonal(s_dist[0]).tolist()}\n"
        f"   ref diag={torch.diagonal(s_ref[0]).tolist()}",
        flush=True,
    )

    torch.testing.assert_close(e_dist, e_ref, rtol=1e-8, atol=1e-8)
    torch.testing.assert_close(
        s_dist,
        s_ref,
        rtol=1e-7,
        atol=1e-9,
        msg=(
            f"rank {rank}: {kind} stress disagrees with the single-GPU "
            f"reference — max |dS|={(s_dist - s_ref).abs().max().item():.3e}"
        ),
    )


@pytest.mark.multigpu
@pytest.mark.parametrize("kind", ["ewald", "pme"])
def test_stress_equivalence_2ranks(kind):
    """Halo-DD stress matches single-GPU for Ewald and PME."""
    pytest.importorskip("nvalchemiops", reason="nvalchemiops not installed")

    mp.spawn(
        _worker,
        args=(WORLD_SIZE, free_port(), _stress_equivalence_worker, kind),
        nprocs=WORLD_SIZE,
    )
