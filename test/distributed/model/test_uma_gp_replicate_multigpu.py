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
"""UMA on the node-replicate graph-parallel strategy (2-GPU equivalence).

``NVALCHEMI_UMA_GP=1`` selects ``GraphReplicatePolicy`` for UMA: every rank holds
the full node set + a sharded edge slice; UMA's per-block eSCN aggregation is
recombined by the all-reduce its (policy-agnostic) block adapter now performs;
energy is read off each rank's owned node slice; forces are UMA's internal
autograd over the edge slice, summed across ranks (``/world_size`` + all-reduce
in consolidation). With a contiguous partition the gathered order matches the
source order, so the full ``[N]`` energy + forces are compared to the
single-process reference. Stress (omat) is computed but not asserted here — the
virial recombine is a follow-on.
"""

from __future__ import annotations

import datetime
import os

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from ase.build import bulk

from nvalchemi.data import AtomicData, Batch
from nvalchemi.distributed.config import DomainConfig, StrategyKind

_CKPT = os.environ.get("NVALCHEMI_UMA_CKPT", "uma-s-1p1")
_TASK = os.environ.get("NVALCHEMI_UMA_TASK", "omat")
_PG_TIMEOUT = datetime.timedelta(minutes=30)


def _build_bcc_fe(dtype=torch.float32):
    atoms = bulk("Fe", "bcc", a=2.87, cubic=True) * (9, 9, 9)
    # Rattle so per-node energies + forces vary — makes forces a sensitive probe
    # (the perfect crystal has ~zero forces, masking feature errors).
    g = torch.Generator().manual_seed(1234)
    pos = torch.tensor(atoms.get_positions(), dtype=dtype)
    pos = pos + 0.1 * torch.randn(pos.shape, generator=g, dtype=dtype)
    return (
        pos,
        torch.tensor(atoms.get_atomic_numbers(), dtype=torch.long),
        torch.tensor(atoms.get_masses(), dtype=dtype),
        torch.tensor(atoms.get_cell().array, dtype=dtype),
        torch.ones(3, dtype=torch.bool),
    )


def _worker(rank: int, world_size: int) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29908"
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl", rank=rank, world_size=world_size, timeout=_PG_TIMEOUT
    )
    try:
        from torch.distributed import DeviceMesh

        from nvalchemi.distributed.distributed_model import DistributedModel
        from nvalchemi.distributed.sharded_batch import ShardedBatch
        from nvalchemi.models.uma import UMAWrapper

        dtype = torch.float32
        device = torch.device(f"cuda:{rank}")
        pos, z, m, cell, pbc = _build_bcc_fe(dtype)
        n_global = pos.shape[0]

        def _mk():
            return Batch.from_data_list(
                [
                    AtomicData(
                        atomic_numbers=z.to(device),
                        positions=pos.to(device=device, dtype=dtype),
                        atomic_masses=m.to(device=device, dtype=dtype),
                        cell=cell.to(device=device, dtype=dtype).unsqueeze(0),
                        pbc=pbc.to(device).unsqueeze(0),
                    )
                ]
            )

        _inf = "default"
        if os.environ.get("UMA_GP_COMPILE"):
            from fairchem.core.units.mlip_unit.api.inference import (
                InferenceSettings,
            )

            # Compile needs merge_mole=True: the per-forward MoLE state
            # (mole_sizes / expert_mixing_coefficients) is mutated inside the
            # compiled graph and lost, so the unmerged MoLE forward reads empty
            # state. Merging folds the experts into the weights once (compile-
            # safe, algebraically exact). tf32 off keeps the compiled-GP-vs-
            # eager-reference numerics tight.
            _inf = InferenceSettings(
                compile=True, merge_mole=True, tf32=False,
                activation_checkpointing=False,
            )

        # Single-GPU reference uses its OWN eager wrapper (rank 0). Sharing the
        # GP wrapper would let fairchem compile the model on the reference call —
        # before DistributedModel installs the DD adapters — so torch.compile
        # would capture the unpatched methods and the GP forward would silently
        # run un-sharded. The compiled GP wrapper must compile on its first DD
        # forward, with adapters already installed.
        e_ref = torch.zeros(1, dtype=dtype, device=device)
        f_ref = torch.zeros(n_global, 3, dtype=dtype, device=device)
        if rank == 0:
            ref_wrapper = UMAWrapper.from_checkpoint(
                _CKPT, task_name=_TASK, device=device, inference_settings="default"
            )
            ro = ref_wrapper(_mk())
            e_ref = ro["energy"].sum().detach().view(1)
            f_ref = ro["forces"].detach()
            del ro, ref_wrapper
        dist.broadcast(e_ref, src=0)
        dist.broadcast(f_ref, src=0)

        wrapper = UMAWrapper.from_checkpoint(
            _CKPT, task_name=_TASK, device=device, inference_settings=_inf
        )

        mesh = DeviceMesh("cuda", list(range(world_size)), mesh_dim_names=("domain",))
        cfg = DomainConfig(cutoff=float(wrapper.cutoff), skin=0.0, mesh=mesh, strategy=StrategyKind.GRAPH_REPLICATE)
        full = _mk() if rank == 0 else None
        sharded = ShardedBatch.from_batch(
            full, mesh=mesh, config=cfg, src=0, partition_mode="contiguous_block"
        )
        with DistributedModel(wrapper, cfg) as dm:
            out = dm(sharded)

        e_gp = out["energy"].sum().detach().view(1)
        f_gp = out["forces"].detach()
        assert f_gp.shape[0] == n_global, (
            f"rank {rank}: forces not full [N]: {tuple(f_gp.shape)}"
        )
        dist.barrier()
        if rank == 0:
            e_abs = (e_gp - e_ref).abs().item()
            e_rel = e_abs / (e_ref.abs().item() + 1e-9)
            fd = (f_gp.double() - f_ref.double()).abs()
            print(
                f"[uma-gp w={world_size}] dE_abs={e_abs:.4f} dE_rel={e_rel:.4e} "
                f"e_ref={e_ref.item():.2f} | f_max_abs={fd.max().item():.4e} "
                f"f_ref_max={f_ref.abs().max().item():.4e} f_mean_abs={fd.mean().item():.4e}",
                flush=True,
            )
        torch.testing.assert_close(f_gp.double(), f_ref.double(), rtol=1e-3, atol=1e-3)
        torch.testing.assert_close(e_gp.double(), e_ref.double(), rtol=1e-3, atol=1e-3)
        dist.barrier()
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    reason="Need 2+ CUDA GPUs",
)
def test_uma_gp_replicate_2ranks() -> None:
    w = int(os.environ.get("UMA_GP_WORLD", "2"))
    mp.spawn(_worker, args=(w,), nprocs=w)


if __name__ == "__main__":
    w = int(os.environ.get("UMA_GP_WORLD", "2"))
    mp.spawn(_worker, args=(w,), nprocs=w)
