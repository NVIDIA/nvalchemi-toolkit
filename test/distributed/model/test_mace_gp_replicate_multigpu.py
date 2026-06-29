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
"""Real MACE on the node-replicate graph-parallel strategy (2-GPU equivalence).

``NVALCHEMI_MACE_GP=1`` selects ``GraphReplicatePolicy``: every rank holds the
full node set + a sharded edge slice, the conv message is recombined by an
all-reduce on each interaction's output, and the energy is read off each rank's
owned node slice. With a contiguous partition the gathered order matches the
source order, so the full ``[N]`` energy + forces are compared to the
single-process reference directly.
"""

from __future__ import annotations

import os
import warnings

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from nvalchemi.data.atomic_data import AtomicData
from nvalchemi.data.batch import Batch
from nvalchemi.distributed.config import DomainConfig


def _build_pbc_argon(reps=(8, 3, 3), dtype=torch.float64, seed=0):
    spacing = 2 ** (1.0 / 6.0) * 3.40 * 1.05
    cx = torch.arange(reps[0], dtype=dtype) * spacing
    cy = torch.arange(reps[1], dtype=dtype) * spacing
    cz = torch.arange(reps[2], dtype=dtype) * spacing
    gx, gy, gz = torch.meshgrid(cx, cy, cz, indexing="ij")
    positions = torch.stack([gx.flatten(), gy.flatten(), gz.flatten()], dim=-1)
    gen = torch.Generator().manual_seed(seed)
    positions = positions + 0.05 * torch.randn(
        positions.shape, dtype=dtype, generator=gen
    )
    n = positions.shape[0]
    box = torch.tensor([r * spacing for r in reps], dtype=dtype)
    positions = positions - torch.floor(positions / box) * box
    atomic_numbers = torch.full((n,), 18, dtype=torch.long)
    masses = torch.full((n,), 39.948, dtype=dtype)
    cell = torch.diag(box)
    pbc = torch.ones(3, dtype=torch.bool)
    return positions, atomic_numbers, masses, cell, pbc


def _worker(rank: int, world_size: int) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29906"
    os.environ["NVALCHEMI_MACE_GP"] = "1"
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from nvalchemi.models.mace import MACEWrapper

        from torch.distributed import DeviceMesh

        from nvalchemi.distributed.distributed_model import DistributedModel
        from nvalchemi.distributed.sharded_batch import ShardedBatch
        from nvalchemi.neighbors import compute_neighbors

        cueq = os.environ.get("MACE_GP_CUEQ") == "1"
        dtype = torch.float32 if cueq else torch.float64
        device = torch.device(f"cuda:{rank}")
        positions, atomic_numbers, masses, cell, pbc = _build_pbc_argon((8, 3, 3))
        n_global = positions.shape[0]

        def _mk_batch():
            return Batch.from_data_list(
                [
                    AtomicData(
                        atomic_numbers=atomic_numbers.to(device),
                        positions=positions.to(device=device, dtype=dtype).clone(),
                        atomic_masses=masses.to(device=device, dtype=dtype),
                        cell=cell.to(device=device, dtype=dtype).unsqueeze(0),
                        pbc=pbc.to(device).unsqueeze(0),
                    )
                ]
            )

        # Single-process reference on rank 0, broadcast.
        e_ref = torch.zeros(1, dtype=dtype, device=device)
        f_ref = torch.zeros(n_global, 3, dtype=dtype, device=device)
        if rank == 0:
            ref = MACEWrapper.from_checkpoint(
                "small", device=device, dtype=dtype, enable_cueq=False
            )
            rb = _mk_batch()
            compute_neighbors(rb, config=ref.model_config.neighbor_config)
            ro = ref(rb)
            e_ref = ro["energy"].sum().detach().view(1)
            f_ref = ro["forces"].detach()
            del ref, rb, ro
        dist.broadcast(e_ref, src=0)
        dist.broadcast(f_ref, src=0)

        # Node-replicate graph-parallel forward.
        wrapper = MACEWrapper.from_checkpoint(
            "small", device=device, dtype=dtype, enable_cueq=False
        )
        mesh = DeviceMesh("cuda", list(range(world_size)), mesh_dim_names=("domain",))
        cfg = DomainConfig(cutoff=float(wrapper.cutoff), skin=0.0, mesh=mesh)
        full = _mk_batch() if rank == 0 else None
        sharded = ShardedBatch.from_batch(
            full, mesh=mesh, config=cfg, src=0, partition_mode="contiguous_block"
        )
        compile_dd = os.environ.get("MACE_GP_COMPILE") == "1"
        with DistributedModel(wrapper, cfg, compile=compile_dd) as dist_model:
            out = dist_model(sharded)

        e_gp = out["energy"].sum().detach().view(1)
        f_gp = out["forces"].detach()
        assert f_gp.shape[0] == n_global, (
            f"rank {rank}: forces are not full [N]: {tuple(f_gp.shape)}"
        )
        e_tol = 1e-4 if cueq else 1e-7
        f_tol = 1e-4 if cueq else 1e-6
        # Compare values at fp64 (the GP path and the reference may carry
        # different float precisions through cueq's fused kernels).
        torch.testing.assert_close(e_gp.double(), e_ref.double(), rtol=e_tol, atol=e_tol)
        torch.testing.assert_close(f_gp.double(), f_ref.double(), rtol=f_tol, atol=f_tol)
        if rank == 0:
            print(f"[mace-gp-replicate w={world_size}] energy + forces match ref")
        dist.barrier()
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    reason="Need 2+ CUDA GPUs",
)
def test_mace_gp_replicate_2ranks() -> None:
    mp.spawn(_worker, args=(2,), nprocs=2)


if __name__ == "__main__":
    mp.spawn(_worker, args=(2,), nprocs=2)
