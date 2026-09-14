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

    pytest test/distributed/model/test_uma_multigpu.py -v

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
# First-time UMA checkpoint download from HuggingFace can run multiple
# minutes; the default 10-minute PG init timeout is enough for a warm
# cache but not always for a cold one. Bumping to 30min is cheap insurance.
_PG_TIMEOUT = datetime.timedelta(minutes=30)


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


# Reference values generated with fairchem-core==2.21.0 in eager/general mode.
# They are kept here so later Fairchem upgrades cannot silently change predictions.
_REFERENCE_FORCE_ROWS = (0, 1, 431, 432, 433, 863)
_REFERENCE_ENERGY = -7139.705078125
_REFERENCE_FORCE_L2_NORM = 2.308806805476699
_REFERENCE_FORCE_MAX_ABS = 1.2710708379745483
_REFERENCE_SELECTED_FORCES = (
    (-1.2710708379745483, 0.6833414435386658, -0.48717576265335083),
    (0.11172015219926834, 0.09676739573478699, 0.10676532983779907),
    (-0.030572697520256042, -0.006757380440831184, -0.020882971584796906),
    (0.8775665163993835, -1.072834849357605, 0.3883421719074249),
    (-0.022328056395053864, 0.0036295573227107525, -0.017870236188173294),
    (0.09056300669908524, 0.06264057755470276, 0.07942171394824982),
)


# ======================================================================
# System — elongated bcc Fe (864 atoms, OMat task)
# ======================================================================


def _build_bcc_fe(dtype: torch.dtype = torch.float32):
    # The 12-cell x axis leaves remote atoms after a deterministic two-rank
    # x split. The shorter y/z axes reduce the cost without changing the halo
    # geometry under test.
    atoms = bulk("Fe", "bcc", a=2.87, cubic=True) * (12, 6, 6)
    positions = torch.as_tensor(atoms.positions, dtype=dtype)
    positions[0] += torch.tensor([0.13, -0.07, 0.05], dtype=dtype)
    positions[len(atoms) // 2] += torch.tensor([-0.09, 0.11, -0.04], dtype=dtype)
    atomic_numbers = torch.as_tensor(atoms.get_atomic_numbers(), dtype=torch.long)
    masses = torch.as_tensor(atoms.get_masses(), dtype=dtype)
    cell = torch.as_tensor(atoms.cell.array, dtype=dtype)
    positions = positions.remainder(torch.diag(cell))
    pbc = torch.ones(3, dtype=torch.bool)
    return positions, atomic_numbers, masses, cell, pbc


def _assert_reference(energy: torch.Tensor, forces: torch.Tensor) -> None:
    """Check the fixed Fairchem 2.21 eager/general UMA reference."""
    expected_rows = torch.tensor(
        _REFERENCE_SELECTED_FORCES, dtype=forces.dtype, device=forces.device
    )
    actual_rows = forces[list(_REFERENCE_FORCE_ROWS)]
    torch.testing.assert_close(
        energy,
        energy.new_tensor(_REFERENCE_ENERGY),
        rtol=1e-4,
        atol=1e-4,
    )
    torch.testing.assert_close(
        torch.linalg.vector_norm(forces.double()),
        forces.new_tensor(_REFERENCE_FORCE_L2_NORM, dtype=torch.float64),
        rtol=1e-4,
        atol=1e-5,
    )
    torch.testing.assert_close(
        forces.abs().max(),
        forces.new_tensor(_REFERENCE_FORCE_MAX_ABS),
        rtol=1e-4,
        atol=1e-5,
    )
    torch.testing.assert_close(actual_rows, expected_rows, rtol=1e-4, atol=1e-5)


def _inference_settings(mode: str) -> Any:
    """Return explicit, reproducible Fairchem inference settings."""
    from fairchem.core.units.mlip_unit.api.inference import InferenceSettings

    if mode == "eager":
        return InferenceSettings(
            compile=False,
            merge_mole=False,
            tf32=False,
            activation_checkpointing=False,
            execution_mode="general",
        )
    if mode == "compiled":
        return InferenceSettings(
            compile=True,
            merge_mole=True,
            tf32=False,
            activation_checkpointing=False,
            execution_mode=None,
        )
    raise ValueError(f"unknown UMA inference mode: {mode}")


# ======================================================================
# Worker
# ======================================================================


def _uma_equivalence_worker(rank: int, world_size: int, mode: str) -> None:
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
    # Do not rely on Fairchem preset names here: their meaning changed in 2.22.
    # The compiled mode keeps tf32 off so numerical drift does not hide a DD bug.
    _inf = _inference_settings(mode)
    wrapper = UMAWrapper.from_checkpoint(
        _CKPT, task_name=_TASK, device=device, inference_settings=_inf
    )

    # merge_mole (compile/turbo) lazily merges the MoLE experts on the first
    # forward and rebinds fairchem's consistency hooks onto the merged backbone
    # at that moment. Running the single-process reference on the DD wrapper first
    # would capture those hooks OUTSIDE the DD scope (stock, caps-unaware), so the
    # later DD forward's check trips on the caps dead-atoms. Give the reference its
    # own wrapper; the DD wrapper then merges cleanly inside the DD scope.
    if mode == "compiled":
        ref_wrapper = UMAWrapper.from_checkpoint(
            _CKPT, task_name=_TASK, device=device, inference_settings=_inf
        )
    else:
        ref_wrapper = wrapper

    # ---- Single-process reference on rank 0 only ----
    e_ref_host = torch.zeros(1, dtype=dtype)
    f_ref_host = torch.zeros(n_global, 3, dtype=dtype)
    # Under compile, computing the reference on rank 0 only makes rank 0 compile
    # an extra (reference-shape) graph while rank 1 idles — divergent Dynamo
    # caches desync the in-graph halo collectives. Run the reference on every
    # rank so compilation is symmetric (the production/MD case); rank 0's values
    # stay authoritative via the broadcast below.
    _ref_here = rank == 0 or mode == "compiled"
    if _ref_here:
        ref_data = AtomicData(
            atomic_numbers=atomic_numbers.to(device),
            positions=positions.to(device=device, dtype=dtype).clone(),
            atomic_masses=masses.to(device=device, dtype=dtype),
            cell=cell.to(device=device, dtype=dtype).unsqueeze(0),
            pbc=pbc.to(device).unsqueeze(0),
        )
        ref_batch = Batch.from_data_list([ref_data])
        ref_out = ref_wrapper(ref_batch)
        e_ref_host = ref_out["energy"].sum().detach().cpu().view(1)
        f_ref_host = ref_out["forces"].detach().cpu()
        if rank == 0 and mode == "eager":
            _assert_reference(e_ref_host.squeeze(0), f_ref_host)
        del ref_batch, ref_out

    e_ref = e_ref_host.to(device=device, dtype=dtype)
    f_ref = f_ref_host.to(device=device, dtype=dtype)
    dist.broadcast(e_ref, src=0)
    dist.broadcast(f_ref, src=0)

    # ---- Distributed forward ----
    mesh = DeviceMesh("cuda", list(range(world_size)), mesh_dim_names=("domain",))

    cutoff = float(wrapper.cutoff)
    domain_config = DomainConfig(
        cutoff=cutoff,
        skin=0.0,
        mesh=mesh,
        grid_dims=(2, 1, 1),
        require_nondegenerate=True,
    )

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

    halo_meta = sharded.halo_meta
    assert halo_meta is not None, f"rank {rank}: halo metadata was not populated"
    n_halo = int(halo_meta.n_padded) - int(halo_meta.n_owned)
    n_remote = n_global - int(halo_meta.n_padded)
    assert n_halo > 0, f"rank {rank}: partition has no halo atoms"
    assert n_remote > 0, f"rank {rank}: partition has no remote atoms"
    print(
        f"[uma-halo rank {rank}] mode={mode} owned={halo_meta.n_owned} "
        f"halo={n_halo} remote={n_remote}",
        flush=True,
    )

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
        f"[uma-fdiff rank {rank}] mode={mode} "
        f"max|Δf|={_fd.max().item():.3e} mean|Δf|={_fd.mean().item():.3e} "
        f"max|f_ref|={f_ref_owned.abs().max().item():.3e}",
        flush=True,
    )
    torch.testing.assert_close(
        f_owned,
        f_ref_owned,
        rtol=1e-3,
        atol=1e-4,
        msg=(
            f"rank {rank}: per-atom forces disagree with single-process UMA reference"
        ),
    )


@pytest.mark.parametrize(("mode", "port"), (("eager", "29704"), ("compiled", "29705")))
@pytest.mark.multigpu
def test_uma_dist_model_equivalence_2ranks(mode: str, port: str) -> None:
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
        args=(WORLD_SIZE, port, _uma_equivalence_worker, mode),
        nprocs=WORLD_SIZE,
    )
