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
"""Multi-GPU regression: UMA under halo-storage domain decomposition.

Mirrors ``test_mace_cueq_multigpu.py`` but exercises the UMA-specific
Triton ops (``torch.ops.fairchem._kernel_*``) registered via
``UMAWrapper.distribution_spec.custom_ops`` + installed by
:meth:`UMAWrapper.distributed_setup`.

The fused node→edge Wigner-permute kernel needs
``gather_inputs=(0,)`` so its per-node ``x`` argument is
halo-materialised before the Triton kernel indexes into it; the other
four kernels (inverse edge→node and three backward kernels) operate on
per-edge tensors and only need pass-through subclass handling.

Requires:
* 2+ CUDA GPUs.
* ``fairchem-core`` installed (``nvalchemi-toolkit[uma]``).
* HF access to a UMA checkpoint (default ``uma-s-1p1``).

Run with::

    pytest test/distributed/test_uma_multigpu.py -v

Override checkpoint / task via env:
    NVALCHEMI_UMA_CKPT=uma-s-1p2 NVALCHEMI_UMA_TASK=omat pytest ...
"""

from __future__ import annotations

import datetime
import os
from typing import Any

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from ase.build import bulk

from nvalchemi.data import AtomicData, Batch
from nvalchemi.distributed.config import DomainConfig

WORLD_SIZE = 2
_CKPT = os.environ.get("NVALCHEMI_UMA_CKPT", "uma-s-1p1")
_TASK = os.environ.get("NVALCHEMI_UMA_TASK", "omat")
# fairchem inference preset: "default" (eager) or "turbo" (compile + tf32 +
# merge_mole). Override to exercise the compiled DD path.
_INFERENCE = os.environ.get("NVALCHEMI_UMA_INFERENCE", "default")
# First-time UMA checkpoint download from HuggingFace can run multiple
# minutes; the default 10-minute PG init timeout is enough for a warm
# cache but not always for a cold one. Bumping to 30min is cheap insurance.
_PG_TIMEOUT = datetime.timedelta(minutes=30)

_skip = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < WORLD_SIZE,
    reason=f"Need {WORLD_SIZE}+ CUDA GPUs",
)


# ======================================================================
# Harness
# ======================================================================


def _init_pg(rank: int, world_size: int, port: str) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = port
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(rank)
    torch.cuda.set_device(rank)
    dist.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size,
        timeout=_PG_TIMEOUT,
    )


def _worker(rank: int, world_size: int, port: str, fn: Any, *args: Any) -> None:
    _init_pg(rank, world_size, port)
    try:
        fn(rank, world_size, *args)
    finally:
        dist.destroy_process_group()


# ======================================================================
# System — bcc Fe 2x2x2 (16 atoms, OMat task)
# ======================================================================


def _build_bcc_fe(dtype: torch.dtype = torch.float32):
    # 9x9x9 cubic bcc cell -> box 25.83 Ang, 1458 atoms. This size is
    # deliberate, not arbitrary: a 2-rank split partitions along a single
    # axis, and that axis only develops *remote* atoms (ones a rank neither
    # owns nor ghosts) once its per-rank domain exceeds two ghost widths,
    # i.e. box / 2 > 2 * ghost_width. With UMA's ~6 Ang cutoff that needs
    # box > 24 Ang. Smaller cells (e.g. 2x2x2 / box 5.74, or even 8x8x8 /
    # box 22.96) are DEGENERATE: every rank ghosts its neighbour's entire
    # domain (remote == 0), so a passing equivalence check would prove
    # nothing about the halo's remote-atom handling (at the non-degenerate
    # size: owned=495 / halo / remote=190, with 0 missing / 0 extra neighbour
    # coverage).
    atoms = bulk("Fe", "bcc", a=2.87, cubic=True) * (9, 9, 9)
    positions = torch.as_tensor(atoms.positions, dtype=dtype)
    atomic_numbers = torch.as_tensor(atoms.get_atomic_numbers(), dtype=torch.long)
    masses = torch.full((len(atoms),), 55.845, dtype=dtype)
    cell = torch.as_tensor(atoms.cell.array, dtype=dtype)
    pbc = torch.ones(3, dtype=torch.bool)
    return positions, atomic_numbers, masses, cell, pbc


# ======================================================================
# Worker
# ======================================================================


def _uma_equivalence_worker(rank: int, world_size: int) -> None:
    """Single-GPU UMA reference on rank 0 → broadcast → each rank asserts
    its owned slice of forces matches and the total energy matches.
    """
    from torch.distributed import DeviceMesh

    from nvalchemi.distributed.distributed_model import DistributedModel
    from nvalchemi.distributed.partitioner import SpatialPartitioner
    from nvalchemi.distributed.sharded_batch import ShardedBatch
    from nvalchemi.models.uma import UMAWrapper

    dtype = torch.float32
    device = torch.device(f"cuda:{rank}")

    positions, atomic_numbers, masses, cell, pbc = _build_bcc_fe(dtype=dtype)
    n_global = positions.shape[0]

    # Load UMA on every rank in parallel — same checkpoint hits the HF
    # cache on the second rank, and crucially keeps every rank reaching the
    # first collective at roughly the same time. A load-on-rank-0-only pattern
    # would starve rank 1's lazy ncclUniqueId exchange (TCPStore timeout →
    # "Failed to recv, got 0 bytes") whenever rank 0's load took longer than
    # the PG init timeout.
    # Pass the ``torch.device`` object directly — ``UMAWrapper.from_checkpoint``
    # reduces it to ``device.type`` ("cuda"), which is what fairchem's
    # ``_setup_device`` asserts on. A ``"cuda:N"`` string bypasses that
    # coercion and trips the assert; relying on the per-process
    # ``torch.cuda.set_device(rank)`` call above is the correct idiom.
    # "compile" = turbo MINUS tf32 (compile + merge_mole, no tf32), to isolate
    # the compiled-DD halo path from turbo's tf32 precision loss. merge_mole is
    # kept on because fairchem's MoLE layer asserts under compile without it.
    # "turbo"/"default" are the stock fairchem presets.
    if _INFERENCE == "compile":
        from fairchem.core.units.mlip_unit.api.inference import InferenceSettings

        _inf: Any = InferenceSettings(
            compile=True, merge_mole=True, activation_checkpointing=False
        )
    else:
        _inf = _INFERENCE
    wrapper = UMAWrapper.from_checkpoint(
        _CKPT, task_name=_TASK, device=device, inference_settings=_inf
    )

    # ---- Single-process reference on rank 0 only ----
    e_ref_host = torch.zeros(1, dtype=dtype)
    f_ref_host = torch.zeros(n_global, 3, dtype=dtype)
    # Under compile, computing the reference on rank 0 only makes rank 0 compile
    # an extra (reference-shape) graph while rank 1 idles — divergent Dynamo
    # caches desync the in-graph halo collectives. Run the reference on every
    # rank so compilation is symmetric (the production/MD case); rank 0's values
    # stay authoritative via the broadcast below.
    _ref_here = (
        rank == 0
        or _INFERENCE != "default"
        or bool(os.environ.get("NVALCHEMI_UMA_REF_ALL_RANKS"))
    )
    if _ref_here:
        ref_data = AtomicData(
            atomic_numbers=atomic_numbers.to(device),
            positions=positions.to(device=device, dtype=dtype).clone(),
            atomic_masses=masses.to(device=device, dtype=dtype),
            cell=cell.to(device=device, dtype=dtype).unsqueeze(0),
            pbc=pbc.to(device).unsqueeze(0),
        )
        ref_batch = Batch.from_data_list([ref_data])
        ref_out = wrapper(ref_batch)
        e_ref_host = ref_out["energy"].sum().detach().cpu().view(1)
        f_ref_host = ref_out["forces"].detach().cpu()
        del ref_batch, ref_out

    e_ref = e_ref_host.to(device=device, dtype=dtype)
    f_ref = f_ref_host.to(device=device, dtype=dtype)
    dist.broadcast(e_ref, src=0)
    dist.broadcast(f_ref, src=0)

    # ---- Distributed forward ----
    mesh = DeviceMesh("cuda", list(range(world_size)), mesh_dim_names=("domain",))

    cutoff = float(wrapper.cutoff)
    domain_config = DomainConfig(cutoff=cutoff, skin=0.0, mesh=mesh)

    if rank == 0:
        full_batch = Batch.from_data_list(
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
    else:
        full_batch = None

    sharded = ShardedBatch.from_batch(
        batch=full_batch, mesh=mesh, config=domain_config, src=0
    )
    local_n = sharded.n_owned

    with DistributedModel(wrapper, domain_config) as dist_model:
        out = dist_model(sharded)

    e_local = out["energy"].sum().detach()
    f_owned = out["forces"].detach()

    # ---- Recover this rank's owned slice of reference forces ----
    partitioner = SpatialPartitioner(
        config=domain_config,
        cell_matrix=cell.to(device=device, dtype=dtype).unsqueeze(0),
        pbc=pbc.to(device).unsqueeze(0),
    )
    rank_assignment = partitioner.assign_atoms_to_ranks(
        positions.to(device=device, dtype=dtype)
    )
    local_mask = rank_assignment == rank
    f_ref_owned = f_ref[local_mask]

    # ---- Assertions ----
    torch.testing.assert_close(
        e_local.view(1),
        e_ref,
        rtol=1e-4,
        atol=1e-4,
        msg=(
            f"rank {rank}: [uma-halo] dist_e={e_local.item():.4f}  "
            f"ref_e={e_ref.item():.4f}  delta={(e_local.item() - e_ref.item()):+.3e}"
        ),
    )
    assert f_owned.shape[0] == local_n, (
        f"rank {rank}: force shape mismatch — got {f_owned.shape}, "
        f"expected ({local_n}, 3)"
    )
    assert f_ref_owned.shape[0] == local_n, (
        f"rank {rank}: partitioner / ShardedBatch disagreement — "
        f"partitioner says {local_mask.sum().item()} atoms, "
        f"ShardedBatch says {local_n}"
    )
    _fd = (f_owned - f_ref_owned).abs()
    print(
        f"[uma-fdiff rank {rank}] inference={_INFERENCE} "
        f"max|Δf|={_fd.max().item():.3e} mean|Δf|={_fd.mean().item():.3e} "
        f"max|f_ref|={f_ref_owned.abs().max().item():.3e}",
        flush=True,
    )
    # "turbo" enables tf32, whose ~1e-3 relative round-off is uncorrelated
    # between the single-process reference graph and the per-rank distributed
    # graphs -- on this near-equilibrium system (forces ~1e-4 eV/A) that noise
    # is the whole signal, so the strict force equivalence is only meaningful
    # without tf32. The "compile" preset (compile + merge_mole, no tf32) is the
    # tight compiled-DD correctness gate (forces match to ~1e-6); turbo gets a
    # tf32-aware tolerance and serves as a runs-clean + energy-exact smoke test.
    f_rtol, f_atol = (3e-3, 1e-3) if _INFERENCE == "turbo" else (1e-3, 1e-4)
    torch.testing.assert_close(
        f_owned,
        f_ref_owned,
        rtol=f_rtol,
        atol=f_atol,
        msg=(
            f"rank {rank}: per-atom forces disagree with single-process UMA reference"
        ),
    )


@_skip
def test_uma_dist_model_equivalence_2ranks():
    """Regression: ``DistributedModel(UMAWrapper)`` matches a single-GPU
    UMA reference on force + total energy under halo storage.

    Gates the five Triton ops registered via
    ``UMAWrapper.distribution_spec`` — specifically, that
    ``_kernel_node_to_edge_wigner_permute`` halo-materialises its
    ``x`` input before the Triton kernel indexes into it, and that
    the subsequent edge→node ``index_add_`` fires the halo-correction
    dispatch so halo rows are owner-consistent on return.
    """
    pytest.importorskip("fairchem.core", reason="fairchem-core not installed")

    mp.spawn(
        _worker,
        args=(WORLD_SIZE, "29572", _uma_equivalence_worker),
        nprocs=WORLD_SIZE,
    )
