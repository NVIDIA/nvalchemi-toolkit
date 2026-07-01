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
"""UMA on the node-partition graph-parallel strategy (2-GPU equivalence).

``NVALCHEMI_UMA_GP=partition`` selects ``GraphParallelPolicy`` for UMA with the
node-partition adapter set: each rank runs the backbone on its owned atom block
(node-wise work + owned-receiver edges), the per-layer node features are
all-gathered for the convolution (reduce-scatter adjoint), and energy/refs are
LOCAL owned partials. The framework's node-partition internal path SUM-reduces
the per-rank energy and forces across ranks (no ``/world``: the feature
all-gather's reduce-scatter backward distributed each node's gradient to its
owner once) and returns this rank's owned forces. We reassemble the global
forces from the disjoint owned blocks and compare both to the single-process
reference. Stress (omat) is not asserted — node-partition stress is a follow-on.
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
    os.environ["MASTER_PORT"] = "29909"
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

        # Single-GPU reference (rank 0) on its own eager wrapper.
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

        # Node-partition runs the backbone on owned atoms; the unmerged MoLE
        # head masks per-dataset over the FULL atom set (mismatching the owned
        # embedding) and ``set_MOLE_sizes`` counts edges over the full batch.
        # ``merge_mole=True`` folds the experts into the weights once
        # (algebraically exact) and returns merged head outputs with no per-atom
        # masking, sidestepping both — and is the production inference path.
        from fairchem.core.units.mlip_unit.api.inference import InferenceSettings

        _inf = InferenceSettings(
            compile=bool(os.environ.get("UMA_GP_COMPILE")),
            merge_mole=True,
            tf32=False,
            activation_checkpointing=False,
        )
        wrapper = UMAWrapper.from_checkpoint(
            _CKPT, task_name=_TASK, device=device, inference_settings=_inf
        )

        mesh = DeviceMesh("cuda", list(range(world_size)), mesh_dim_names=("domain",))
        cfg = DomainConfig(cutoff=float(wrapper.cutoff), skin=0.0, mesh=mesh, strategy=StrategyKind.GRAPH_PARTITION)
        full = _mk() if rank == 0 else None
        sharded = ShardedBatch.from_batch(
            full, mesh=mesh, config=cfg, src=0, partition_mode="contiguous_block"
        )
        with DistributedModel(wrapper, cfg) as dm:
            out = dm(sharded)

        # Energy is global (all-reduced) on every rank.
        e_gp = out["energy"].sum().detach().view(1)
        # Forces come back as this rank's OWNED block; reassemble the global
        # [N, 3] from the disjoint contiguous blocks (a SUM all-reduce of each
        # rank's zero-padded slice).
        f_owned = out["forces"].detach()
        nlo = (n_global * rank) // world_size
        nhi = (n_global * (rank + 1)) // world_size
        assert f_owned.shape[0] == nhi - nlo, (
            f"rank {rank}: owned forces {tuple(f_owned.shape)} != block {nhi - nlo}"
        )
        f_gp = torch.zeros(n_global, 3, dtype=torch.float64, device=device)
        f_gp[nlo:nhi] = f_owned.double()
        dist.all_reduce(f_gp, op=dist.ReduceOp.SUM)
        dist.barrier()
        if rank == 0:
            e_abs = (e_gp - e_ref).abs().item()
            e_rel = e_abs / (e_ref.abs().item() + 1e-9)
            fd = (f_gp - f_ref.double()).abs()
            print(
                f"[uma-gp-part w={world_size}] dE_abs={e_abs:.4f} dE_rel={e_rel:.4e} "
                f"e_ref={e_ref.item():.2f} | f_max_abs={fd.max().item():.4e} "
                f"f_ref_max={f_ref.abs().max().item():.4e} "
                f"f_mean_abs={fd.mean().item():.4e}",
                flush=True,
            )
        torch.testing.assert_close(f_gp, f_ref.double(), rtol=1e-3, atol=1e-3)
        torch.testing.assert_close(e_gp.double(), e_ref.double(), rtol=1e-3, atol=1e-3)
        dist.barrier()
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    reason="Need 2+ CUDA GPUs",
)
def test_uma_gp_partition_2ranks() -> None:
    w = int(os.environ.get("UMA_GP_WORLD", "2"))
    mp.spawn(_worker, args=(w,), nprocs=w)


if __name__ == "__main__":
    w = int(os.environ.get("UMA_GP_WORLD", "2"))
    mp.spawn(_worker, args=(w,), nprocs=w)
