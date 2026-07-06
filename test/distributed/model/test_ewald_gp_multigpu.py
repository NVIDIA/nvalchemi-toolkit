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

"""Multi-GPU regression: Ewald under node-partition graph-parallel.

The graph-parallel sibling of ``test_ewald_multigpu`` (halo). A 2-rank
``DistributedModel(EwaldModelWrapper)`` on the ``GRAPH_PARTITION`` strategy must
match the single-GPU reference on total energy and per-atom forces.

Exercises the Ewald GP path: ``EwaldModelWrapper._distribution_spec_gp``
(``gp_replicate_geometry`` + ``node_energy_key`` + ``FRAMEWORK_FROM_NODE_ENERGY``)
routed through ``DistributedModel._graph_parallel_dense_full_autograd`` — full
geometry replicated per rank (correct reciprocal structure factor from the full
charge set), the dense ``neighbor_matrix`` masked to owned receivers (partitioned
real-space), the owned-aware ``node_energy_key`` sum, and framework autograd of the
owned energy over the full-position leaf. Requires ``hybrid_forces=False``.

Run (box, 2 GPUs, ``NCCL_P2P_DISABLE=1`` on trx40-03)::

    NVALCHEMI_EWALD_N_SIDE=3 NVALCHEMI_EWALD_BOX=8.46 \\
        pytest test/distributed/model/test_ewald_gp_multigpu.py -v -s
"""

from __future__ import annotations

import os
from typing import Any

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from nvalchemi.data import AtomicData, Batch
from nvalchemi.distributed.config import DomainConfig, StrategyKind

WORLD_SIZE = 2

_skip = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < WORLD_SIZE,
    reason=f"Need {WORLD_SIZE}+ CUDA GPUs",
)


def _init_pg(rank: int, world_size: int, port: str) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = port
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(rank)
    torch.cuda.set_device(rank)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)


def _worker(rank: int, world_size: int, port: str, fn: Any, *args: Any) -> None:
    _init_pg(rank, world_size, port)
    try:
        fn(rank, world_size, *args)
    finally:
        dist.destroy_process_group()


def _build_nacl(dtype: torch.dtype = torch.float32, seed: int = 0):
    n_side = int(os.environ.get("NVALCHEMI_EWALD_N_SIDE", 3))
    box = float(os.environ.get("NVALCHEMI_EWALD_BOX", 8.46))

    coords = torch.arange(n_side, dtype=dtype) * (box / n_side)
    gx, gy, gz = torch.meshgrid(coords, coords, coords, indexing="ij")
    positions = torch.stack([gx.flatten(), gy.flatten(), gz.flatten()], dim=-1)
    n = positions.shape[0]

    g = torch.Generator().manual_seed(seed)
    positions = positions + 0.05 * torch.randn(
        positions.shape, dtype=dtype, generator=g
    )
    positions = positions % box

    signs = torch.ones(n, dtype=dtype)
    signs[1::2] = -1.0
    charges = signs
    atomic_numbers = torch.where(
        signs > 0,
        torch.full((n,), 11, dtype=torch.long),
        torch.full((n,), 17, dtype=torch.long),
    )
    masses = torch.where(
        signs > 0,
        torch.full((n,), 22.99, dtype=dtype),
        torch.full((n,), 35.45, dtype=dtype),
    )
    cell = torch.eye(3, dtype=dtype) * box
    pbc = torch.ones(3, dtype=torch.bool)
    return positions, atomic_numbers, masses, charges, cell, pbc


def _owned_counts(n: int, world: int) -> list[int]:
    per_rank = max(n // world, 1)
    counts = [per_rank] * world
    counts[-1] = n - per_rank * (world - 1)
    return counts


def _ewald_gp_equivalence_worker(rank: int, world_size: int) -> None:
    from torch.distributed import DeviceMesh

    from nvalchemi.distributed.distributed_model import DistributedModel
    from nvalchemi.distributed.sharded_batch import ShardedBatch
    from nvalchemi.models.ewald import EwaldModelWrapper

    dtype = torch.float32
    device = torch.device(f"cuda:{rank}")

    positions, atomic_numbers, masses, charges, cell, pbc = _build_nacl(dtype=dtype)
    n_global = positions.shape[0]

    # ---- Single-process reference on rank 0 ----
    # Apples-to-apples with the GP path: the framework's
    # ``_graph_parallel_dense_full_autograd`` runs the forward ENERGY-ONLY
    # (``compute_forces=False`` -> the differentiable monolithic reciprocal) and
    # derives forces by autograd over the energy. The single-GPU reference must
    # take the same branch: the deprecated ``compute_forces=True`` warp reciprocal
    # drifts from the energy-only reciprocal by a constant (fp32 backend drift, not
    # DD), so comparing against it would flag a spurious energy offset.
    e_ref_host = torch.zeros(1, dtype=dtype)
    f_ref_host = torch.zeros(n_global, 3, dtype=dtype)
    if rank == 0:
        ref_wrapper = EwaldModelWrapper(
            cutoff=min(5.0, 0.45 * cell[0, 0].item()), hybrid_forces=False
        )
        ref_pos = (
            positions.to(device=device, dtype=dtype).clone().requires_grad_(True)
        )
        ref_data = AtomicData(
            atomic_numbers=atomic_numbers.to(device),
            positions=ref_pos,
            atomic_masses=masses.to(device=device, dtype=dtype),
            charges=charges.to(device=device, dtype=dtype),
            cell=cell.to(device=device, dtype=dtype).unsqueeze(0),
            pbc=pbc.to(device).unsqueeze(0),
            forces=torch.zeros(n_global, 3, device=device, dtype=dtype),
            energy=torch.zeros(1, 1, device=device, dtype=dtype),
        )
        ref_batch = Batch.from_data_list([ref_data])
        from nvalchemi.neighbors import compute_neighbors

        compute_neighbors(ref_batch, config=ref_wrapper.model_config.neighbor_config)
        ref_wrapper.model_config.active_outputs = {"energy"}
        e_sum = ref_wrapper(ref_batch)["energy"].sum()
        (ref_grad,) = torch.autograd.grad(e_sum, ref_pos)
        e_ref_host = e_sum.detach().cpu().view(1)
        f_ref_host = (-ref_grad).detach().cpu()
        del ref_wrapper, ref_batch, e_sum, ref_grad

    e_ref = e_ref_host.to(device=device, dtype=dtype)
    f_ref = f_ref_host.to(device=device, dtype=dtype)
    dist.broadcast(e_ref, src=0)
    dist.broadcast(f_ref, src=0)

    # ---- Distributed GP forward ----
    dist_wrapper = EwaldModelWrapper(
        cutoff=min(5.0, 0.45 * cell[0, 0].item()), hybrid_forces=False
    )
    gp_spec = dist_wrapper.distribution_spec(StrategyKind.GRAPH_PARTITION)
    mesh = DeviceMesh("cuda", list(range(world_size)), mesh_dim_names=("domain",))
    cutoff = float(dist_wrapper.cutoff)
    domain_config = DomainConfig(
        cutoff=cutoff, skin=0.0, mesh=mesh, strategy=StrategyKind.GRAPH_PARTITION
    )

    if rank == 0:
        full_batch = Batch.from_data_list(
            [
                AtomicData(
                    atomic_numbers=atomic_numbers.to(device),
                    positions=positions.to(device=device, dtype=dtype).clone(),
                    atomic_masses=masses.to(device=device, dtype=dtype),
                    charges=charges.to(device=device, dtype=dtype),
                    cell=cell.to(device=device, dtype=dtype).unsqueeze(0),
                    pbc=pbc.to(device).unsqueeze(0),
                    forces=torch.zeros(n_global, 3, device=device, dtype=dtype),
                    energy=torch.zeros(1, 1, device=device, dtype=dtype),
                )
            ]
        )
    else:
        full_batch = None

    sharded = ShardedBatch.from_batch(
        batch=full_batch,
        mesh=mesh,
        config=domain_config,
        src=0,
        partition_mode="contiguous_block",
    )
    dist_model = DistributedModel(dist_wrapper, domain_config, spec=gp_spec)
    out = dist_model(sharded)

    e_local = out["energy"].sum().detach()
    f_owned = out["forces"].detach()

    # ---- Owned slice: contiguous index block ----
    counts = _owned_counts(n_global, world_size)
    offset = sum(counts[:rank])
    n_owned = counts[rank]
    f_ref_owned = f_ref[offset : offset + n_owned]

    e_delta = e_local.item() - e_ref.item()
    diff = (f_owned - f_ref_owned).detach()
    abs_diff = diff.abs()
    print(
        f"[ewald-gp rank {rank}] n_owned={n_owned} "
        f"dist_e={e_local.item():+.6f} ref_e={e_ref.item():+.6f} Δ={e_delta:+.3e}  "
        f"|ΔF| max={abs_diff.max().item():.3e} rms="
        f"{abs_diff.pow(2).mean().sqrt().item():.3e}",
        flush=True,
    )

    assert f_owned.shape[0] == n_owned, (
        f"rank {rank}: force shape {f_owned.shape}, expected ({n_owned}, 3)"
    )
    # Normalize dtype + shape so the comparison is purely on values (the GP path
    # may return a different fp width / rank than the single-GPU reference).
    torch.testing.assert_close(
        e_local.reshape(-1).double(), e_ref.reshape(-1).double(),
        rtol=5e-4, atol=5e-4,
        msg=f"rank {rank}: energy mismatch Δ={e_delta:+.3e}",
    )
    torch.testing.assert_close(
        f_owned.reshape(-1).double(), f_ref_owned.reshape(-1).double(),
        rtol=1e-3, atol=5e-4,
        msg=f"rank {rank}: force mismatch max|ΔF|={abs_diff.max().item():.3e}",
    )


@_skip
def test_ewald_gp_dist_model_equivalence_2ranks():
    """``DistributedModel(EwaldModelWrapper, GRAPH_PARTITION)`` matches single-GPU
    Ewald on total energy and per-atom forces (correctness-first GP path)."""
    pytest.importorskip("nvalchemiops", reason="nvalchemiops not installed")
    mp.spawn(
        _worker,
        args=(WORLD_SIZE, "29577", _ewald_gp_equivalence_worker),
        nprocs=WORLD_SIZE,
    )


if __name__ == "__main__":
    mp.spawn(
        _worker,
        args=(WORLD_SIZE, "29577", _ewald_gp_equivalence_worker),
        nprocs=WORLD_SIZE,
    )
