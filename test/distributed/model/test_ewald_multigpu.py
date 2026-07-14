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

"""Multi-GPU regressions for distributed Ewald electrostatics.

Three scenarios, each a distinct DD code path, all gating a 2-rank
``DistributedModel(EwaldModelWrapper)`` against a single-GPU reference on total
energy and per-atom forces over a charge-neutral NaCl lattice:

* ``test_ewald_dist_model_equivalence_2ranks`` — eager halo storage
  (staged-bindings / ``wrap_custom_op`` owned-slice + all-reduce).
* ``test_ewald_compile_dd_2ranks`` — ``hybrid_forces=False`` on the compiled
  energy-autograd DD path; also asserts no steady-state recompiles. The
  single-GPU reference is itself compiled so the gate measures DD correctness,
  not compile-vs-eager fp32 drift.
* ``test_ewald_gp_dist_model_equivalence_2ranks`` — node-partition
  graph-parallel (``GRAPH_PARTITION`` / ``_distribution_spec_gp``).

Requires 2+ CUDA GPUs and ``nvalchemiops``. Systems are non-degenerate by
default; override via ``NVALCHEMI_EWALD_N_SIDE`` / ``NVALCHEMI_EWALD_BOX``
(keep box > 4*(cutoff+skin))."""

from __future__ import annotations

import os

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from _dd_harness import nccl_worker as _worker
from _electrostatics import build_nacl

from nvalchemi.data import AtomicData, Batch
from nvalchemi.distributed.config import DomainConfig, StrategyKind

WORLD_SIZE = 2
WARMUP_STEPS = 4
STEADY_STEPS = 4
JITTER = 0.05
_EWALD_CUT = 6.0

_skip = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < WORLD_SIZE,
    reason=f"Need {WORLD_SIZE}+ CUDA GPUs",
)


def _ewald_equivalence_worker(rank: int, world_size: int) -> None:
    """Single-GPU Ewald reference on rank 0 → broadcast → each rank
    runs the distributed forward and asserts its owned slice of forces
    + the total energy match the reference.

    Uses ``hybrid_forces=False`` and no ``stress`` because those modes
    route through the staged bindings under halo (see
    ``EwaldModelWrapper.forward``). Hybrid + stress have their own
    dispatch paths and are single-GPU only in this MVP; they'd
    land in a follow-up test once multi-GPU charge-grad /
    virial-staged wiring is verified.
    """
    from torch.distributed import DeviceMesh

    from nvalchemi.distributed.distributed_model import DistributedModel
    from nvalchemi.distributed.partitioner import SpatialPartitioner
    from nvalchemi.distributed.sharded_batch import ShardedBatch
    from nvalchemi.models.ewald import EwaldModelWrapper

    dtype = torch.float32
    device = torch.device(f"cuda:{rank}")

    n_side = int(os.environ.get("NVALCHEMI_EWALD_N_SIDE", 10))
    box = float(os.environ.get("NVALCHEMI_EWALD_BOX", 28.0))
    positions, atomic_numbers, masses, charges, cell, pbc = build_nacl(n_side, box)
    n_global = positions.shape[0]

    # ---- Single-process reference on rank 0 only ----
    e_ref_host = torch.zeros(1, dtype=dtype)
    f_ref_host = torch.zeros(n_global, 3, dtype=dtype)
    if rank == 0:
        ref_wrapper = EwaldModelWrapper(
            cutoff=min(5.0, 0.45 * cell[0, 0].item()), hybrid_forces=False
        )
        ref_data = AtomicData(
            atomic_numbers=atomic_numbers.to(device),
            positions=positions.to(device=device, dtype=dtype).clone(),
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
        ref_out = ref_wrapper(ref_batch)
        e_ref_host = ref_out["energy"].sum().detach().cpu().view(1)
        f_ref_host = ref_out["forces"].detach().cpu()
        del ref_wrapper, ref_batch, ref_out

    e_ref = e_ref_host.to(device=device, dtype=dtype)
    f_ref = f_ref_host.to(device=device, dtype=dtype)
    dist.broadcast(e_ref, src=0)
    dist.broadcast(f_ref, src=0)

    # ---- Distributed forward ----
    dist_wrapper = EwaldModelWrapper(
        cutoff=min(5.0, 0.45 * cell[0, 0].item()), hybrid_forces=False
    )
    mesh = DeviceMesh("cuda", list(range(world_size)), mesh_dim_names=("domain",))

    cutoff = float(dist_wrapper.cutoff)
    domain_config = DomainConfig(
        cutoff=cutoff, skin=0.0, mesh=mesh, require_nondegenerate=True
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
        batch=full_batch, mesh=mesh, config=domain_config, src=0
    )
    local_n = sharded.n_owned

    with DistributedModel(dist_wrapper, domain_config) as dist_model:
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

    # ---- Diagnostics: energy delta + force error stats on BOTH ranks
    # print before the assert so we see the full picture even on failure.
    e_delta = e_local.item() - e_ref.item()
    print(
        f"[ewald-halo rank {rank}] "
        f"dist_e={e_local.item():+.6f}  ref_e={e_ref.item():+.6f}  "
        f"Δ={e_delta:+.3e}",
        flush=True,
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

    diff = (f_owned - f_ref_owned).detach()
    abs_diff = diff.abs()
    ref_norm = f_ref_owned.norm(dim=1).clamp_min(1e-12)
    rel_per_atom = diff.norm(dim=1) / ref_norm
    worst = int(abs_diff.norm(dim=1).argmax().item())
    local_global_idx = torch.nonzero(local_mask, as_tuple=False).flatten()[worst].item()
    print(
        f"[ewald-halo rank {rank}] "
        f"|ΔF| max={abs_diff.max().item():.3e}  mean={abs_diff.mean().item():.3e}  "
        f"rms={(abs_diff.pow(2).mean().sqrt()).item():.3e}  "
        f"|ΔF|/|F_ref| max={rel_per_atom.max().item():.3e}  "
        f"median={rel_per_atom.median().item():.3e}  "
        f"|F_ref| max={f_ref_owned.norm(dim=1).max().item():.3e}  "
        f"min={f_ref_owned.norm(dim=1).min().item():.3e}\n"
        f"[ewald-halo rank {rank}] worst owned atom local_idx={worst} "
        f"global_idx={local_global_idx}  "
        f"dist_F={f_owned[worst].tolist()}  ref_F={f_ref_owned[worst].tolist()}",
        flush=True,
    )

    # ---- Assertions ----
    # fp32 + long-range kernels with an FFT-free direct k-sum: total
    # energy holds to ~1e-4 absolute; per-atom forces to ~1e-3 relative
    # (same tolerance the cueq/MACE multi-GPU test uses).
    torch.testing.assert_close(
        e_local.view(1),
        e_ref,
        rtol=1e-4,
        atol=1e-4,
        msg=(
            f"rank {rank}: energy mismatch Δ={e_delta:+.3e} "
            f"(dist={e_local.item():.6f}, ref={e_ref.item():.6f})"
        ),
    )
    torch.testing.assert_close(
        f_owned,
        f_ref_owned,
        rtol=1e-3,
        atol=1e-4,
        msg=(
            f"rank {rank}: per-atom forces disagree with single-process Ewald "
            f"reference — max |ΔF|={abs_diff.max().item():.3e}, "
            f"max |ΔF|/|F|={rel_per_atom.max().item():.3e}"
        ),
    )


@_skip
def test_ewald_dist_model_equivalence_2ranks():
    """Regression: ``DistributedModel(EwaldModelWrapper)`` under halo
    matches single-GPU Ewald on total energy and per-atom forces.

    Gates the staged-bindings + wrap_custom_op owned_slice + all_reduce
    path end-to-end: the per-rank partial structure factors sum
    correctly across the mesh, and the per-atom reciprocal energy
    drops halo rows via per_system_reduce at the final scatter.
    """
    pytest.importorskip("nvalchemiops", reason="nvalchemiops not installed")

    mp.spawn(
        _worker,
        args=(WORLD_SIZE, "29573", _ewald_equivalence_worker),
        nprocs=WORLD_SIZE,
    )


def _make_data(an, positions, masses, charges, cell, pbc, device, dtype):
    n = positions.shape[0]
    return AtomicData(
        atomic_numbers=an.to(device),
        positions=positions.to(device=device, dtype=dtype).clone(),
        atomic_masses=masses.to(device=device, dtype=dtype),
        charges=charges.to(device=device, dtype=dtype),
        cell=cell.to(device=device, dtype=dtype).unsqueeze(0),
        pbc=pbc.to(device).unsqueeze(0),
        forces=torch.zeros(n, 3, device=device, dtype=dtype),
        energy=torch.zeros(1, 1, device=device, dtype=dtype),
    )


def _compile_worker(rank: int, world_size: int) -> None:
    from torch._dynamo.utils import counters
    from torch.distributed import DeviceMesh

    from nvalchemi.distributed.distributed_model import DistributedModel
    from nvalchemi.distributed.partitioner import SpatialPartitioner
    from nvalchemi.distributed.sharded_batch import ShardedBatch
    from nvalchemi.models.ewald import EwaldModelWrapper
    from nvalchemi.neighbors import compute_neighbors

    dtype = torch.float32
    device = torch.device(f"cuda:{rank}")
    n_side = int(os.environ.get("NVALCHEMI_EWALD_N_SIDE", 12))
    box = float(os.environ.get("NVALCHEMI_EWALD_BOX", 40.0))
    positions0, an, masses, charges, cell, pbc = build_nacl(n_side, box, jitter=0.15)
    n_global = positions0.shape[0]

    # COMPILED single-GPU reference (energy-only + autograd) — isolates DD
    # correctness from compile-vs-eager fp32 drift (see module docstring / PME).
    e_ref = torch.zeros(1, dtype=dtype, device=device)
    f_ref = torch.zeros(n_global, 3, dtype=dtype, device=device)
    if rank == 0:
        ref = EwaldModelWrapper(cutoff=_EWALD_CUT, hybrid_forces=False)
        batch = Batch.from_data_list(
            [_make_data(an, positions0, masses, charges, cell, pbc, device, dtype)]
        )
        compute_neighbors(batch, config=ref.model_config.neighbor_config)

        def _ref_energy(b):
            return ref(b)["energy"]

        compiled_ref = torch.compile(_ref_energy, dynamic=False)
        ref.model_config.active_outputs = {"energy"}
        pos_leaf = batch.positions.detach().requires_grad_(True)
        batch._atoms_group["positions"] = pos_leaf
        e = compiled_ref(batch)
        (grad,) = torch.autograd.grad([e.sum()], [pos_leaf])
        e_ref.copy_(e.sum().detach().view(1))
        f_ref.copy_((-grad).detach())
        del ref
    dist.broadcast(e_ref, src=0)
    dist.broadcast(f_ref, src=0)

    wrapper = EwaldModelWrapper(cutoff=_EWALD_CUT, hybrid_forces=False)
    cp = wrapper.distribution_spec().compile
    assert cp is not None and cp.forces_via_autograd, (
        "Ewald(hybrid_forces=False) must declare a forces_via_autograd CompilePolicy"
    )
    mesh = DeviceMesh("cuda", list(range(world_size)), mesh_dim_names=("domain",))
    cfg = DomainConfig(
        cutoff=_EWALD_CUT, skin=2.0, mesh=mesh, require_nondegenerate=True
    )
    partitioner = SpatialPartitioner(
        config=cfg,
        cell_matrix=cell.to(device=device, dtype=dtype).unsqueeze(0),
        pbc=pbc.to(device).unsqueeze(0),
    )

    def _sharded(pos):
        full = (
            Batch.from_data_list(
                [_make_data(an, pos, masses, charges, cell, pbc, device, dtype)]
            )
            if rank == 0
            else None
        )
        return ShardedBatch.from_batch(batch=full, mesh=mesh, config=cfg, src=0)

    gen = torch.Generator(device="cpu").manual_seed(11)
    graphs_after_warmup = [0]

    with DistributedModel(wrapper, cfg, compile=True) as dm:
        for step in range(WARMUP_STEPS + STEADY_STEPS):
            if step == 0:
                pos = positions0
            else:
                disp = JITTER * torch.randn(
                    positions0.shape, dtype=dtype, generator=gen
                ).to(device)
                pos = (positions0.to(device) + disp) % box
            out = dm(_sharded(pos))
            if step == 0:
                f_owned = out["forces"].detach()
                e_local = out["energy"].sum().detach()
                local_mask = (
                    partitioner.assign_atoms_to_ranks(
                        positions0.to(device=device, dtype=dtype)
                    )
                    == rank
                )
                f_ref_owned = f_ref[local_mask]
                de = (e_local - e_ref).abs().item()
                df = (f_owned - f_ref_owned).abs().max().item()
                print(
                    f"[ewald-compile rank {rank}] step0 ΔE={de:.3e} |ΔF|max={df:.3e} "
                    f"n_owned={f_owned.shape[0]}",
                    flush=True,
                )
                torch.testing.assert_close(
                    e_local.view(1).to(e_ref.dtype),
                    e_ref,
                    rtol=1e-4,
                    atol=1e-2,
                    msg=f"rank {rank}: compiled Ewald energy mismatch ΔE={de:.3e}",
                )
                torch.testing.assert_close(
                    f_owned,
                    f_ref_owned,
                    rtol=1e-2,
                    atol=2e-3,
                    msg=f"rank {rank}: compiled Ewald forces mismatch |ΔF|max={df:.3e}",
                )
            if step == WARMUP_STEPS - 1:
                graphs_after_warmup[0] = counters["stats"]["unique_graphs"]

    final_graphs = counters["stats"]["unique_graphs"]
    print(
        f"[ewald-compile rank {rank}] unique_graphs warmup={graphs_after_warmup[0]} final={final_graphs}",
        flush=True,
    )
    assert final_graphs == graphs_after_warmup[0], (
        f"rank {rank}: compiled Ewald recompiled in steady state "
        f"({graphs_after_warmup[0]} -> {final_graphs})"
    )


@_skip
def test_ewald_compile_dd_2ranks():
    """Compiled ``DistributedModel(Ewald, hybrid_forces=False)`` == single-GPU; no steady recompiles."""
    pytest.importorskip("nvalchemiops", reason="nvalchemiops not installed")
    mp.spawn(_worker, args=(WORLD_SIZE, "29591", _compile_worker), nprocs=WORLD_SIZE)


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

    n_side = int(os.environ.get("NVALCHEMI_EWALD_N_SIDE", 3))
    box = float(os.environ.get("NVALCHEMI_EWALD_BOX", 8.46))
    positions, atomic_numbers, masses, charges, cell, pbc = build_nacl(n_side, box)
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
        ref_pos = positions.to(device=device, dtype=dtype).clone().requires_grad_(True)
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
        e_local.reshape(-1).double(),
        e_ref.reshape(-1).double(),
        rtol=5e-4,
        atol=5e-4,
        msg=f"rank {rank}: energy mismatch Δ={e_delta:+.3e}",
    )
    torch.testing.assert_close(
        f_owned.reshape(-1).double(),
        f_ref_owned.reshape(-1).double(),
        rtol=1e-3,
        atol=5e-4,
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
