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
"""Multi-GPU regression (EAGER DD): MACE with cuequivariance conv-fusion on a
non-degenerate partition.

On CUDA ``convert_e3nn_cueq`` fuses the InteractionBlock message pass
(sender-gather + channel-wise TP + receiver-scatter) into one opaque kernel
with the edge indices internal, hiding both the gather and the scatter from
ShardTensor dispatch — so under DD it computes a purely local message and is
wrong on a non-degenerate partition.

Correctness is restored declaratively: the MACE spec unfuses the conv for the
DD scope (``nvalchemi/models/mace.py::_cueq_conv_unfuse_adapters`` →
``ModuleForwardAdapter``), reverting to the external ``node_feats[sender]``
gather + ``scatter_sum`` plain MACE uses, where ``_halo_scatter_correction``
and the halo read-refresh fire. The remaining node/edge-local cueq kernels
(``uniform_1d``, ``indexed_linear_B/C``, ``segmented_transpose``) keep
pass-through OpAdapters so ShardTensor identity survives the opaque kernel.
The single-process path is untouched (the adapter installs only inside the DD
scope) and keeps the fused kernel.

Requires:
* 2+ CUDA GPUs (cueq kernels are CUDA-only).
* ``cuequivariance`` / ``cuequivariance_torch`` installed.
* ``mace-torch`` installed.

Run with::

    pytest test/distributed/model/test_mace_cueq_multigpu.py -v
"""

from __future__ import annotations

import os

# cueq JIT-compiles its C++ kernels at first call. Under multi-GPU
# (one rank per GPU) the parallel-compile path races across ranks and
# yields corrupted kernel handles — symptom is ``CUDA_ERROR_INVALID_HANDLE``
# at the first cueq forward on rank > 0. Upstream fix: serialize the
# compile step by setting this env var before any torch import.
# See https://github.com/NVIDIA/cuEquivariance/issues/253.
os.environ.setdefault("CUEQUIVARIANCE_OPS_PARALLEL_COMPILE", "0")

import warnings

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from _dd_harness import nccl_worker as _worker

from nvalchemi.data import AtomicData, Batch
from nvalchemi.distributed.config import DomainConfig

WORLD_SIZE = 2

_skip = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < WORLD_SIZE,
    reason=f"Need {WORLD_SIZE}+ CUDA GPUs",
)


# ======================================================================
# Process-group / test-harness setup
# ======================================================================


# ======================================================================
# System builder — small orthorhombic Ar supercell, PBC
# ======================================================================


def _build_pbc_argon(
    reps: tuple[int, int, int] = (8, 3, 3),
    dtype: torch.dtype = torch.float64,
    seed: int = 0,
):
    """Orthorhombic Ar supercell, PBC. Default ``reps`` is **elongated**
    (8×3×3 ≈ 32×12×12 Å) so a 2-rank split along x gives a *non-degenerate*
    partition (per-rank extent ≫ 2×ghost_width): the halo correction does real
    cross-rank work, so a conv that skips it (the cueq fused kernel) fails
    visibly. A near-cubic cell would make the halo a near no-op and mask the
    bug — the trap that hid the fused-conv DD error behind a loose tolerance.
    """
    spacing = 2 ** (1.0 / 6.0) * 3.40 * 1.05  # ~4.007 Å
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


# ======================================================================
# Main worker — single-process cueq reference vs. 2-rank cueq distributed
# ======================================================================


def _mace_cueq_equivalence_worker(rank: int, world_size: int) -> None:
    """Run the distributed cueq forward on every rank; compute a single
    cueq reference on rank 0 and broadcast it; assert each rank's owned
    slice matches.

    Per-rank reference re-computation is wasteful (same forward, same
    answer) and extra cueq contexts on every GPU are extra failure
    surface for stream/context bugs. Compute once on rank 0, broadcast
    to all ranks, and have every rank assert its owned slice.

    Uses float32 — the dtype cueq targets for GPU speedup; force/energy
    tolerances reflect fp32 arithmetic in the fused kernels.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from nvalchemi.models.mace import MACEWrapper

    from torch.distributed import DeviceMesh

    from nvalchemi.distributed.distributed_model import DistributedModel
    from nvalchemi.distributed.partitioner import SpatialPartitioner
    from nvalchemi.distributed.sharded_batch import ShardedBatch
    from nvalchemi.neighbors import compute_neighbors

    dtype = torch.float32
    device = torch.device(f"cuda:{rank}")

    positions, atomic_numbers, masses, cell, pbc = _build_pbc_argon(reps=(8, 3, 3))
    n_global = positions.shape[0]

    # ---- Single-process reference on rank 0 only ----
    e_ref_host = torch.zeros(1, dtype=dtype)
    f_ref_host = torch.zeros(n_global, 3, dtype=dtype)
    if rank == 0:
        ref_wrapper = MACEWrapper.from_checkpoint(
            "small", device=device, dtype=dtype, enable_cueq=True
        )
        ref_data = AtomicData(
            atomic_numbers=atomic_numbers.to(device),
            positions=positions.to(device=device, dtype=dtype).clone(),
            atomic_masses=masses.to(device=device, dtype=dtype),
            cell=cell.to(device=device, dtype=dtype).unsqueeze(0),
            pbc=pbc.to(device).unsqueeze(0),
        )
        ref_batch = Batch.from_data_list([ref_data])
        compute_neighbors(ref_batch, config=ref_wrapper.model_config.neighbor_config)
        ref_out = ref_wrapper(ref_batch)
        e_ref_host = ref_out["energy"].sum().detach().cpu().view(1)
        f_ref_host = ref_out["forces"].detach().cpu()
        # Free ref_wrapper so we don't hold two cueq models on rank 0's GPU.
        del ref_wrapper, ref_batch, ref_out

    # Broadcast reference tensors. Stage on device so NCCL works.
    e_ref = e_ref_host.to(device=device, dtype=dtype)
    f_ref = f_ref_host.to(device=device, dtype=dtype)
    dist.broadcast(e_ref, src=0)
    dist.broadcast(f_ref, src=0)

    # ---- Distributed forward: cueq MACE across 2 GPUs ----
    dist_wrapper = MACEWrapper.from_checkpoint(
        "small", device=device, dtype=dtype, enable_cueq=True
    )
    mesh = DeviceMesh("cuda", list(range(world_size)), mesh_dim_names=("domain",))

    cutoff = float(dist_wrapper.cutoff)
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

    with DistributedModel(dist_wrapper, domain_config) as dist_model:
        out = dist_model(sharded)

    e_local = out["energy"].sum().detach()
    f_owned = out["forces"].detach()

    # ---- Recover this rank's owned slice of the reference forces ----
    # ShardedBatch.from_batch sorts source atoms by
    # ``partitioner.assign_atoms_to_ranks`` and scatters in that order.
    # Reconstruct the same permutation so we know which rows of the full
    # reference forces belong to this rank.
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
    # fp32 + fused cueq kernels: total-energy tolerance ~1e-5 is fine for
    # ~64 atoms; forces ~1e-4 abs in eV/Å (checkpoint-dependent).
    torch.testing.assert_close(
        e_local.view(1),
        e_ref,
        rtol=1e-4,
        atol=1e-4,
        msg=(
            f"rank {rank}: [mace+cueq] dist_e={e_local.item():.4f}  "
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
    # ---- Force-error fingerprint (diagnostic; prints on BOTH ranks) ----
    # Energy matches but forces may not → the bug is in the backward/halo
    # path. The error *shape* tells us which: a clean per-atom factor at
    # boundary atoms ⇒ double/missing halo correction on fused_bwd; noise
    # everywhere ⇒ a broken adjoint. positions[local_mask] aligns with
    # f_owned (both original-order restricted to this rank, via stable sort).
    err = (f_owned - f_ref_owned).abs()
    per_atom = err.norm(dim=1)
    # Fractional coord along each axis (boundary detection: atoms near the
    # split plane are where halo correction acts).
    pos_owned = positions.to(device=device, dtype=dtype)[local_mask]
    box = torch.diagonal(cell.to(device=device, dtype=dtype))
    frac = (pos_owned / box) % 1.0
    order = torch.argsort(per_atom, descending=True)
    topk = order[: min(8, order.numel())]
    lines = [
        f"rank {rank}: FORCE DIAG  n_owned={local_n}  "
        f"max|Δ|={err.max().item():.3e}  mean|Δ|={err.mean().item():.3e}  "
        f"median|Δf|={per_atom.median().item():.3e}  "
        f"n_atoms|Δf|>1e-3={(per_atom > 1e-3).sum().item()}",
    ]
    for i in topk.tolist():
        fo, fr = f_owned[i], f_ref_owned[i]
        ratio = fo / fr.where(fr.abs() > 1e-6, torch.ones_like(fr))
        lines.append(
            f"  atom{i:>3} |Δf|={per_atom[i].item():.3e}  frac={frac[i].tolist()}  "
            f"f_dist={fo.tolist()}  f_ref={fr.tolist()}  ratio={ratio.tolist()}"
        )
    print("\n".join(lines), flush=True)

    torch.testing.assert_close(
        f_owned,
        f_ref_owned,
        rtol=1e-3,
        atol=1e-4,
        msg=f"rank {rank}: per-atom forces disagree with single-process cueq reference",
    )


@_skip
def test_mace_cueq_dist_model_equivalence_2ranks():
    """Regression: ``DistributedModel(MACEWrapper(enable_cueq=True))`` matches a
    single-GPU cueq reference on force + total energy on a **non-degenerate**
    partition (8×3×3 Ar, split along the long axis).

    Exercises the conv-fusion path: ``convert_e3nn_cueq`` fuses the
    InteractionBlock message pass (sender-gather + channel-wise TP +
    receiver-scatter) into one opaque kernel with the edge indices internal,
    hiding both the gather and the scatter from ShardTensor dispatch. The
    MACE spec unfuses that conv for the DD scope
    (``_cueq_conv_unfuse_adapters`` → ``ModuleForwardAdapter``), reverting to the
    external ``node_feats[sender]`` gather + ``scatter_sum`` plain MACE uses, so
    the halo read-refresh + scatter-correction fire as normal.

    History: this was an ``xfail`` — the conv fusion computed a purely local
    message under DD and was wrong by ~N·1.8e-4 relative; a near-cubic 64-atom
    cell + loose tolerance hid it. The non-degenerate cell + tight tolerance now
    gates the fix.
    """
    pytest.importorskip("mace", reason="mace-torch not installed")
    pytest.importorskip("cuequivariance", reason="cuequivariance not installed")

    mp.spawn(
        _worker,
        args=(WORLD_SIZE, "29571", _mace_cueq_equivalence_worker),
        nprocs=WORLD_SIZE,
    )
