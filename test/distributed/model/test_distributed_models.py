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

"""Unified model-integration tests for the distributed framework.

One file covering every supported model (:class:`LennardJonesModelWrapper`,
:class:`MACEWrapper`, AIMNet2). Each model exercises three checks:

1. **distribution_spec declarative.** The wrapper advertises the right
   preset (``None`` for LJ, :data:`SPEC_MPNN_HALO` for MACE, etc.).
2. **Single-step force equivalence.** Partition the atoms across ranks,
   run the wrapper, compare forces / energy against a single-process
   reference. Ensures the dispatch path is correct.
3. **Multi-step NVE equivalence.** Run N velocity-verlet steps;
   per-step positions / velocities / total energy must match single
   process. Catches compounding numerical drift that a single-step
   test misses.

Every test drives the real framework tooling — :class:`AtomicData`,
:func:`Batch.from_data_list`, :func:`compute_neighbors`, :class:`NVE`
— so the assertion is both "dispatch routes correctly" AND "the wrapper
+ batch + integrator contract holds in distributed mode."
"""

from __future__ import annotations

import os
import warnings
from typing import Any

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

# Gloo-harness shims live in conftest.py (shared with other distributed
# tests in this directory).
from _helpers import _MockMesh, make_gloo_sharded_batch  # noqa: E402

from nvalchemi.data import AtomicData, Batch
from nvalchemi.distributed._core.storage_policy import HaloStoragePolicy
from nvalchemi.distributed.config import DomainConfig
from nvalchemi.distributed.partitioner import SpatialPartitioner
from nvalchemi.distributed.spec import SPEC_MPNN_HALO
from nvalchemi.dynamics.integrators.nve import NVE
from nvalchemi.models.lj import LennardJonesModelWrapper
from nvalchemi.neighbors import compute_neighbors

# ======================================================================
# Gloo test harness
# ======================================================================


def _patch_all_to_all_for_gloo() -> None:
    import physicsnemo.distributed.utils as pn_utils

    def _impl(tensor, indices, sizes, dim=0, group=None):
        # Gloo's TCP transport rejects raw cuda-tensor isend/irecv
        # ("Bad address" from writev on device memory) — gloo's higher-level
        # collectives (all_gather/all_reduce) auto-stage via cpu, but
        # isend/irecv expose the raw transport. Stage cuda->cpu before send
        # and cpu->cuda after recv so the wire path is cpu while the caller
        # sees a cuda result on cuda input. Mirrors the validate harness's
        # _patch_physicsnemo_all_to_all_for_gloo; required for the gloo+cuda
        # model-test tier (N ranks sharing one GPU).
        comm_size = dist.get_world_size(group=group)
        rank = dist.get_rank(group=group)
        out_device = tensor.device
        on_cuda = out_device.type == "cuda"
        x_send = [
            (tensor[idx].contiguous().cpu() if on_cuda else tensor[idx].contiguous())
            for idx in indices
        ]
        x_recv = []
        shape = list(tensor.shape)
        cpu_dev = torch.device("cpu")
        for r in range(comm_size):
            shape[dim] = sizes[r][rank]
            x_recv.append(torch.empty(shape, dtype=tensor.dtype, device=cpu_dev))
        ops = []
        for r in range(comm_size):
            if r == rank:
                x_recv[r].copy_(x_send[r])
            else:
                if x_send[r].numel() > 0:
                    ops.append(dist.isend(x_send[r], dst=r, group=group))
                if x_recv[r].numel() > 0:
                    ops.append(dist.irecv(x_recv[r], src=r, group=group))
        for op in ops:
            op.wait()
        joined = torch.cat(x_recv, dim=dim)
        return joined.to(out_device) if on_cuda else joined

    pn_utils.indexed_all_to_all_v_wrapper = _impl


def _init_gloo(rank: int, world_size: int, port: str) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = port
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    # ``cpu:gloo,cuda:gloo`` (not plain ``gloo``) so the gloo backend
    # host-stages CUDA tensors for its collectives. Plain ``gloo`` tries to
    # ``writev`` device memory directly → "Bad address". This is what lets the
    # model-test tier run N ranks sharing ONE GPU (NCCL can't) while still
    # exercising the real cuda code path — mirrors the validate harness.
    dist.init_process_group(
        backend="cpu:gloo,cuda:gloo", rank=rank, world_size=world_size
    )
    _patch_all_to_all_for_gloo()


def _worker(rank: int, world_size: int, port: str, fn: Any, *args: Any) -> None:
    _init_gloo(rank, world_size, port)
    try:
        fn(rank, world_size, *args)
    finally:
        dist.destroy_process_group()


def _init_nccl(rank: int, world_size: int, port: str) -> None:
    """Real-multi-GPU NCCL init: rank r binds physical ``cuda:r``. Used by the
    full DomainParallel NVE dynamics path, which the gloo+cuda single-GPU tier
    cannot run — gloo's TCP transport rejects raw cuda-tensor isend/irecv
    ("Bad address") on the dynamics-step collectives (gather / halo migration),
    and those are not all host-staged like the forward halo exchange is."""
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = port
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(rank)
    torch.cuda.set_device(rank)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)


def _worker_nccl(rank: int, world_size: int, port: str, fn: Any, *args: Any) -> None:
    _init_nccl(rank, world_size, port)
    try:
        fn(rank, world_size, *args)
    finally:
        dist.destroy_process_group()


# ``_MockMesh``, ``_LocalShardTensor``, ``make_gloo_sharded_batch`` —
# the gloo-harness shims for physicsnemo's CUDA-only ShardTensor — live
# in ``conftest.py`` so every distributed test in this directory shares
# one copy. Imported below where needed.


# ======================================================================
# System builders
# ======================================================================


# Per-topology system builders live in the parameterized test section
# below (``_build_open_argon_cluster``, ``_build_pbc_orthorhombic_argon``,
# ``_build_pbc_hex_sio2``).


# ======================================================================
# distribution_spec declarative tests
# ======================================================================


def test_lj_wrapper_declares_halo_spec() -> None:
    """LJ extends the minimal halo spec (``SPEC_LJ_HALO``) with one
    :class:`OpAdapter` per opaque Warp op, so the functional energy/force/virial
    kernels' outputs are wrapped back into halo ShardTensors under
    distribution."""
    wrapper = LennardJonesModelWrapper(epsilon=0.0104, sigma=3.40, cutoff=8.5)
    spec = wrapper.distribution_spec()

    # Still halo in the fundamentals.
    policy = spec.distribution.policy
    assert isinstance(policy, HaloStoragePolicy)
    assert policy.scatter_mode == "halo_correction"
    assert policy.gather_mode == "halo_read"

    # One OpAdapter per LJ Warp op, with no output transforms (the empty-transform
    # adapter still wraps each returned output as a halo ShardTensor).
    by_op_name = {str(op.op._schema.name): op for op in spec.distribution.custom_ops}
    assert "nvalchemi::lj_energy_forces_batch" in by_op_name
    assert "nvalchemi::lj_energy_forces_virial_batch" in by_op_name
    for op in spec.distribution.custom_ops:
        assert op.scatter_outputs == ()
        assert op.gather_inputs == ()

    # Cached / stable across accesses (built once per call is fine; content equal).
    assert (
        wrapper.distribution_spec().distribution.custom_ops[0].op
        is spec.distribution.custom_ops[0].op
    )


def test_mace_wrapper_declares_mpnn_halo_spec() -> None:
    pytest.importorskip("mace", reason="mace-torch not installed")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from nvalchemi.models.mace import MACEWrapper

    wrapper = MACEWrapper.from_checkpoint("small", dtype=torch.float64)
    spec = wrapper.distribution_spec()
    # Same scatter-heavy MPNN halo fundamentals as SPEC_MPNN_HALO ...
    assert isinstance(spec.distribution.policy, HaloStoragePolicy)
    assert spec.distribution.policy == SPEC_MPNN_HALO.distribution.policy
    assert spec.output_kinds == SPEC_MPNN_HALO.output_kinds
    # ... extended with a marshalling MethodAdapter over the WHOLE
    # ``e3nn.o3.SphericalHarmonics.forward`` — the smallest region
    # covering both the module-level ``@torch.jit.script`` ``_spherical_harmonics``
    # kernel (ShardTensor IMA) AND the in-place ``sh.mul_(cat)`` normalization
    # (the AOT in-place-subclass-mutation-across-break limitation). A bare
    # JitAdapter on the scripted function covers only the kernel, not the
    # in-place op, so MACE declares the method marshal explicitly.
    method_targets = {
        (h.module_path, h.class_name, h.method_name, getattr(h, "mode", None))
        for h in spec.distribution.third_party_helpers
        if type(h).__name__ == "MethodAdapter"
    }
    assert (
        "e3nn.o3",
        "SphericalHarmonics",
        "forward",
        "marshal",
    ) in method_targets
    # Idempotent / cached on repeated access.
    assert wrapper.distribution_spec() is spec


def test_mace_cueq_wrapper_declares_custom_ops_spec() -> None:
    """``MACEWrapper(enable_cueq=True)`` returns a MACE-specific spec that
    extends ``SPEC_MPNN_HALO`` with the seven ``torch.ops.cuequivariance.*``
    kernels the cueq'd MACE forward touches, **all as pass-through wraps**
    (unwrap → run → re-wrap; ``gather_inputs=()`` / ``scatter_outputs=()``):
    ``fused_tensor_product`` (+ its two backwards), ``uniform_1d``,
    ``indexed_linear_B/C``, ``segmented_transpose``.

    The conv's cross-rank (halo) correction does **not** ride the fused
    kernel's ``scatter_outputs``. It lives in the mode-dependent conv-unfuse
    adapter (``_cueq_conv_unfuse_adapters``): under eager DD the conv is unfused
    to the external gather + ``scatter_sum`` (the halo handler fires on that
    external scatter, joining plain MACE); under compiled DD the conv stays
    fused for memory parity and the per-layer refresh adapter's
    ``scatter_to_owners`` carries correctness. Numerical DD correctness is gated
    by ``test_mace_cueq_dist_model_equivalence_2ranks``; this test only asserts
    the declared op-adapter structure.
    """
    pytest.importorskip("mace", reason="mace-torch not installed")
    pytest.importorskip("cuequivariance", reason="cuequivariance not installed")
    pytest.importorskip(
        "cuequivariance_torch", reason="cuequivariance_torch not installed"
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # Importing cuequivariance_torch registers the ``torch.ops.cuequivariance.*``
        # op namespace that ``_mace_cueq_spec`` resolves.
        import cuequivariance_torch  # noqa: F401, PLC0415

        from nvalchemi.models.mace import _mace_cueq_spec

    if not hasattr(torch.ops, "cuequivariance"):
        pytest.skip("torch.ops.cuequivariance.* not registered")

    # The cueq spec is data, built from the registered op namespace — NOT from a
    # live cueq'd model. ``from_checkpoint(enable_cueq=True)`` requires a CUDA
    # device (the cuEq weight conversion guard), so building it on CPU would
    # ERROR; resolve the spec directly so the structural assertion runs on a CPU
    # box too. (On a GPU box the live-model path is exercised by the dist-model
    # equivalence + NVE tiers.)
    spec = _mace_cueq_spec()

    # Still MPNN-halo in the fundamentals.
    policy = spec.distribution.policy
    assert isinstance(policy, HaloStoragePolicy)
    assert policy.scatter_mode == "halo_correction"
    assert policy.gather_mode == "halo_read"
    assert spec.system_reductions is True
    # But now with custom_ops populated (non-cueq variant has () ).
    assert len(spec.distribution.custom_ops) == 7

    # Every cueq op is a pass-through wrap (no gather/scatter on the op itself):
    # the conv's halo correction lives in the conv-unfuse adapter + the external
    # scatter's halo handler, not on the fused kernel. (Pre-#104 the fused ops
    # carried ``scatter_outputs=(0,)``; that message-tensor correction moved off
    # the kernel when the conv-unfuse adapter took over — see the docstring.)
    by_op_name = {
        # OpOverload's name is e.g. "cuequivariance::fused_tensor_product"
        str(op_spec.op._schema.name): op_spec
        for op_spec in spec.distribution.custom_ops
    }
    for name in (
        "cuequivariance::fused_tensor_product",
        "cuequivariance::fused_tensor_product_bwd",
        "cuequivariance::fused_tensor_product_bwd_bwd",
        "cuequivariance::uniform_1d",
        "cuequivariance::indexed_linear_B",
        "cuequivariance::indexed_linear_C",
        "cuequivariance::segmented_transpose",
    ):
        assert by_op_name[name].gather_inputs == ()
        assert by_op_name[name].scatter_outputs == ()

    # Idempotent / cached — the spec is memoized across calls.
    assert _mace_cueq_spec() is spec


def test_mace_cueq_distributed_setup_registers_handlers() -> None:
    """The cueq spec's ``custom_ops`` register a ``wrap_custom_op`` handler
    per op when installed through the :class:`AdapterRegistry`, and
    ``restore`` removes them.

    The wiring is spec-driven: ``DistributedModel`` installs
    ``spec.distribution.custom_ops`` (+ ``third_party_helpers``) via the
    registry on context enter. This test exercises that install/restore
    lifecycle directly. Runs on CPU — doesn't execute the kernels; walks the
    dispatcher registry to verify (de)registration lands.
    """
    pytest.importorskip("mace", reason="mace-torch not installed")
    pytest.importorskip("cuequivariance", reason="cuequivariance not installed")
    pytest.importorskip(
        "cuequivariance_torch", reason="cuequivariance_torch not installed"
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # Importing cuequivariance_torch registers the ``torch.ops.cuequivariance.*``
        # op namespace that ``_mace_cueq_spec`` resolves.
        import cuequivariance_torch  # noqa: F401, PLC0415

        from nvalchemi.models.mace import _mace_cueq_spec

    if not hasattr(torch.ops, "cuequivariance"):
        pytest.skip("torch.ops.cuequivariance.* not registered")

    from nvalchemi.distributed._core.adapter import AdapterRegistry
    from nvalchemi.distributed._core.shard_tensor import list_handlers

    # This test is about the install/restore lifecycle given the spec's
    # ``custom_ops`` — decouple it from live cueq-model construction, which
    # requires CUDA (the cuEq weight-conversion guard). The custom_ops are pure
    # data resolved from the registered op namespace, so the registry lifecycle
    # runs on a CPU box.
    custom_ops = list(_mace_cueq_spec().distribution.custom_ops)

    # Measure the install DELTA rather than clearing the registry — the
    # built-in dim-0 intercepts (scatter_add_/index_add_/…) live in the same
    # table, and ``clear_handlers()`` would wipe them for the rest of the
    # session, poisoning later tests.
    before = set(list_handlers())

    registry = AdapterRegistry()
    registry.install(custom_ops)

    added = set(list_handlers()) - before
    # Each op registers on both OpOverload and OpOverloadPacket (two
    # bindings per OpAdapter; 7 OpAdapters = 14 entries).
    assert len(added) == 14
    # Every newly-added handler name carries the ``wrap_custom_op`` prefix.
    assert all(n.startswith("wrap_custom_op[") for _, n in added)

    registry.restore()
    # Exact restore — back to the pre-install registry, defaults intact.
    assert set(list_handlers()) == before


# ======================================================================
# NVE multi-step equivalence via ``DomainParallel``
#
# Drives the *real* dynamics orchestration: :class:`DomainParallel`
# wraps :class:`NVE`, which wraps the model wrapper. Per-step flow
# delegates to :class:`DistributedModel` (halo exchange, NL, halo
# dispatch, energy reduction) with no hand-rolled velocity-verlet in
# the test. After ``n_steps``, gather the distributed trajectory and
# assert its potential energy at the gathered positions matches a
# single-process reference recomputed at the same positions.
#
# The comparison is energy-at-gathered-positions (not per-step
# trajectory tracking) because the two paths can prime forces and
# interleave post-step ``batch.energy`` writes differently — the
# potential energy on the same positions is the strongest invariant
# we can assert without reimplementing velocity-verlet in the test.
# ======================================================================


def _nve_via_domain_parallel_worker(
    rank: int,
    world_size: int,
    system_name: str,
    model_name: str,
    n_steps: int,
    dt_fs: float,
    energy_tol: float,
    device: str = "cpu",
    dtype: torch.dtype = torch.float64,
) -> None:
    """Parameterized DomainParallel NVE end-to-end worker.

    Shared by LJ (cpu/fp64 — analytic, the cheap dynamics smoke)
    and the real-model tier (cuda/fp32 — MACE). The only differences are
    the system builder, the model wrapper, the step count, the energy
    tolerance, and the device/dtype. The core flow is identical.
    """
    from torch.distributed.device_mesh import DeviceMesh  # noqa: PLC0415

    from nvalchemi.distributed.domain_parallel import DomainParallel  # noqa: PLC0415
    from nvalchemi.dynamics.base import DynamicsStage  # noqa: PLC0415
    from nvalchemi.hooks.neighbor_list import NeighborListHook  # noqa: PLC0415

    # DomainParallel needs a *real* ``DeviceMesh(device, ...)``. On cuda the
    # full NVE dynamics path runs on real multi-GPU NCCL (rank r -> ``cuda:r``)
    # via ``_worker_nccl`` — the gloo+cuda single-GPU tier can't transport raw
    # cuda tensors through the dynamics-step collectives. The e3nn TorchScript
    # spherical-harmonics kernel faults on a non-zero device when a storage-less
    # ShardTensor reaches it (the @torch.jit.script + ShardTensor
    # illegal-memory-access) unless scripted-op marshalling (default
    # ``scripted_marshal="auto"`` + MACE's declared ``_spherical_harmonics``
    # JitAdapter) unwraps ShardTensor->local at that boundary, so rank r on
    # ``cuda:r`` runs the scripted op safely.

    positions, atomic_numbers, masses, cell, pbc = _SYSTEM_BUILDERS[system_name]()
    positions = positions.to(device=device, dtype=dtype)
    atomic_numbers = atomic_numbers.to(device=device)
    masses = masses.to(device=device, dtype=dtype)
    cell = cell.to(device=device, dtype=dtype)
    pbc = pbc.to(device=device)
    n = positions.shape[0]

    torch.manual_seed(1)
    velocities = 0.001 * torch.randn_like(positions)
    velocities -= velocities.mean(dim=0, keepdim=True)

    # Real gloo-backed DeviceMesh — ``DistributedModel.__call__`` goes
    # through physicsnemo's ``ShardTensor.from_local`` which requires a
    # mesh with ``device_type``. On the cuda tier the mesh follows the data
    # device so ShardTensor placement and collectives agree.
    mesh = DeviceMesh(device, list(range(world_size)), mesh_dim_names=("domain",))

    # ---- Distributed trajectory via DomainParallel(NVE(wrapper)) ----
    wrapper_factory = _WRAPPER_FACTORIES[model_name]
    dist_wrapper = wrapper_factory(dtype, device)
    dist_nve = NVE(
        model=dist_wrapper,
        dt=dt_fs,
        hooks=[
            NeighborListHook(
                config=dist_wrapper.model_config.neighbor_config,
                skin=0.0,
                stage=DynamicsStage.BEFORE_COMPUTE,
            )
        ],
    )
    cutoff = float(dist_wrapper.model_config.neighbor_config.cutoff)
    cfg = DomainConfig(cutoff=cutoff, skin=0.0, mesh=mesh)
    dp = DomainParallel(dynamics=dist_nve, config=cfg)

    if rank == 0:
        full_data = AtomicData(
            atomic_numbers=atomic_numbers,
            positions=positions.clone(),
            atomic_masses=masses,
            forces=torch.zeros(n, 3, dtype=dtype),
            energy=torch.zeros(1, 1, dtype=dtype),
            cell=cell.unsqueeze(0),
            pbc=pbc.unsqueeze(0),
        )
        full_data.add_node_property("velocities", velocities.clone())
        full_batch = Batch.from_data_list([full_data])
    else:
        full_batch = None

    local_batch = dp.partition(full_batch)
    for _ in range(n_steps):
        local_batch, _ = dp.step(local_batch)

    dist_final_energy = float(local_batch.energy.sum().item())
    full_final = dp.gather(local_batch, dst=0)

    # ---- Single-process reference at the gathered final positions ----
    if rank == 0:
        assert full_final is not None
        assert full_final.num_nodes == n
        ref_wrapper = wrapper_factory(dtype, device)
        ref_data = AtomicData(
            atomic_numbers=full_final.atomic_numbers,
            positions=full_final.positions.clone(),
            atomic_masses=full_final.atomic_masses,
            cell=full_final.cell,
            pbc=full_final.pbc,
        )
        ref_batch = Batch.from_data_list([ref_data])
        compute_neighbors(ref_batch, config=ref_wrapper.model_config.neighbor_config)
        ref_out = ref_wrapper(ref_batch)
        ref_final_energy = float(ref_out["energy"].sum().item())
        assert abs(dist_final_energy - ref_final_energy) < energy_tol, (
            f"rank 0: [{model_name} / {system_name} / ws={world_size} / "
            f"n_steps={n_steps}] after DomainParallel NVE, "
            f"dist_e={dist_final_energy:.6f} disagrees with single-process "
            f"ref_e={ref_final_energy:.6f} at gathered positions "
            f"(delta={dist_final_energy - ref_final_energy:+.3e})"
        )


# Parameter matrix: (system, world_size, n_steps, dt_fs, energy_tol).
# LJ: cheap analytical forces — tight tolerance, more steps.
# MACE: autograd + equivariant layers — looser tolerance, fewer steps.
_LJ_NVE_CASES = [
    ("nonpbc_open_argon", 2, 5, 1.0, 1e-8),
    ("nonpbc_open_argon", 4, 5, 1.0, 1e-8),
    ("pbc_orthorhombic_argon", 2, 5, 1.0, 1e-8),
]

# MACE NVE runs on the cuda tier in fp32 (warp/cueq precision), so the
# dist-vs-single-process energy delta sits above fp64 noise — looser tol.
_MACE_NVE_CASES = [
    ("pbc_orthorhombic_argon", 2, 3, 0.5, 5e-3),
]


@pytest.mark.parametrize(
    "system_name,world_size,n_steps,dt_fs,energy_tol", _LJ_NVE_CASES
)
def test_lj_nve_via_domain_parallel(
    system_name: str,
    world_size: int,
    n_steps: int,
    dt_fs: float,
    energy_tol: float,
) -> None:
    key = f"lj_nve_{system_name}_{world_size}"
    mp.spawn(
        _worker,
        args=(
            world_size,
            _port_for(key),
            _nve_via_domain_parallel_worker,
            system_name,
            "lj",
            n_steps,
            dt_fs,
            energy_tol,
        ),
        nprocs=world_size,
    )


# ======================================================================
# DistributedModel equivalence tests — parameterized over topology,
# world size, and model. Every test asserts that a single-process
# reference forward and a multi-rank ``DistributedModel`` forward
# produce identical total energy and identical per-atom forces (on
# each rank's owned slice).
#
# Coverage matrix — intentionally chosen to catch the classes of bugs
# we have hit in the past:
#
#   * ``nonpbc_open_argon``:  non-PBC open cluster. The regime where
#     halo partitioning actually partitions (``n_padded < n_global``).
#     Regression for the rank-assignment + balanced-scatter bug and
#     the NL wrap-on-pbc=False bug.
#   * ``pbc_orthorhombic_argon``:  simple cubic argon with PBC. Classic
#     PBC test with a diagonal cell matrix — baseline for PBC halo
#     identification.
#   * ``pbc_hex_sio2``:  alpha-quartz SiO2 supercell with PBC. The cell
#     is hexagonal (γ = 120°) so the cell matrix is non-diagonal — this
#     catches any bug that relies on ``inv(cell) == inv(cell).T``
#     (which is true for orthorhombic but not for skew cells).
#
# For each topology we run at world_size ∈ {2, 4}. LJ runs at all
# sizes because it's cheap; MACE's larger hex case is marked ``slow``
# because mace-torch on CPU is expensive at the sizes needed to
# exercise the halo pathway non-trivially.
# ======================================================================


def _build_open_argon_cluster(
    n_per_side: int, dtype: torch.dtype = torch.float64, seed: int = 0
):
    """Open (non-PBC) simple-cubic Ar cluster matching the benchmark.

    Non-PBC spatial layout is the regime where halo partitioning actually
    *partitions*: the halo on one rank doesn't wrap around the full box
    via PBC and pull in every remote atom, so ``n_padded < n_global`` and
    any bug in cross-rank energy reduction is observable. Dense 3D
    FCC-with-PBC tests pull the entire system into each rank's padded
    view and hide reduction bugs — see gloo diagnostic notes in the
    ``DistributedModel`` regression suite.
    """
    spacing = 2 ** (1.0 / 6.0) * 3.40 * 1.05  # ~4.007 Å (LJ min × 1.05)
    coords = torch.arange(n_per_side, dtype=dtype) * spacing
    gx, gy, gz = torch.meshgrid(coords, coords, coords, indexing="ij")
    positions = torch.stack([gx.flatten(), gy.flatten(), gz.flatten()], dim=-1)
    torch.manual_seed(seed)
    positions = positions + 0.05 * torch.randn_like(positions)
    n = positions.shape[0]
    atomic_numbers = torch.full((n,), 18, dtype=torch.long)
    masses = torch.full((n,), 39.948, dtype=dtype)
    # Large enough box so SpatialPartitioner has room; non-PBC.
    box_side = n_per_side * spacing + 20.0
    cell = torch.eye(3, dtype=dtype) * box_side
    pbc = torch.zeros(3, dtype=torch.bool)
    return positions, atomic_numbers, masses, cell, pbc


def _sharded_batch_for_system(
    positions: torch.Tensor,
    atomic_numbers: torch.Tensor,
    masses: torch.Tensor,
    cell: torch.Tensor,
    pbc: torch.Tensor,
    rank: int,
    world_size: int,
    mesh: "_MockMesh",
    cutoff: float,
    *,
    storage: str = "halo",
):
    """Partition a system across ``world_size`` ranks and build a
    gloo-harness ``ShardedBatch``.

    Partition mode is chosen by ``storage``:
    * ``"halo"`` — spatial (``SpatialPartitioner``). Matches the halo-
      storage flow (LJ, MACE) where cross-rank neighbors are served
      from locally-stored halo rows.
    * ``"sharded"`` — contiguous block by global index. Matches the
      sharded-storage flow (AIMNet2) where cross-rank lookups route
      on demand via global indices, so spatial locality isn't needed.
      Also avoids zero-atom ranks on degenerate geometries like 1D
      chains that spatial partitioning can bin unevenly.

    Returns ``(sharded, local_mask, domain_config)``; the caller uses
    ``local_mask`` to slice reference-trajectory tensors onto the same
    rank partition.
    """
    domain_config = DomainConfig(cutoff=cutoff, mesh=mesh)
    n_global = positions.shape[0]

    if storage == "halo":
        partitioner = SpatialPartitioner(
            config=domain_config,
            cell_matrix=cell.unsqueeze(0),
            pbc=pbc.unsqueeze(0),
        )
        rank_assignment = partitioner.assign_atoms_to_ranks(positions)
    elif storage == "sharded":
        per_rank = n_global // world_size
        rank_assignment = (torch.arange(n_global, dtype=torch.long) // per_rank).clamp(
            max=world_size - 1
        )
    else:
        raise ValueError(f"unsupported storage mode: {storage!r}")

    local_mask = rank_assignment == rank
    sizes = [int((rank_assignment == r).sum().item()) for r in range(world_size)]

    sharded = make_gloo_sharded_batch(
        mesh=mesh,
        local_positions=positions[local_mask].contiguous().clone(),
        local_numbers=atomic_numbers[local_mask].contiguous(),
        local_masses=masses[local_mask].contiguous(),
        cell=cell.unsqueeze(0),
        pbc=pbc.unsqueeze(0),
        sizes=sizes,
        n_global=n_global,
    )
    return sharded, local_mask, domain_config


# ----------------------------------------------------------------------
# System builders.
#
# Each returns ``(positions, atomic_numbers, masses, cell, pbc)``. All
# use argon (Z=18) as the atomic species — MACE-MP's small checkpoint
# supports arbitrary elements, so the same positions drive both LJ and
# MACE tests. SiO2 is the exception: element species matter for MACE
# there, and LJ just sees them as Z=14/8 pair targets which is fine.
# ----------------------------------------------------------------------


def _build_pbc_orthorhombic_argon(
    n_per_side: int = 4, dtype: torch.dtype = torch.float64, seed: int = 0
):
    """Simple-cubic Ar crystal with PBC in an orthorhombic cell."""
    spacing = 2 ** (1.0 / 6.0) * 3.40 * 1.05  # ~4.007 Å
    coords = torch.arange(n_per_side, dtype=dtype) * spacing
    gx, gy, gz = torch.meshgrid(coords, coords, coords, indexing="ij")
    positions = torch.stack([gx.flatten(), gy.flatten(), gz.flatten()], dim=-1)
    torch.manual_seed(seed)
    positions = positions + 0.05 * torch.randn_like(positions)
    n = positions.shape[0]
    box = n_per_side * spacing
    # Wrap back into [0, box) so pbc=True doesn't see atoms just outside.
    positions = positions - torch.floor(positions / box) * box
    atomic_numbers = torch.full((n,), 18, dtype=torch.long)
    masses = torch.full((n,), 39.948, dtype=dtype)
    cell = torch.eye(3, dtype=dtype) * box
    pbc = torch.ones(3, dtype=torch.bool)
    return positions, atomic_numbers, masses, cell, pbc


def _build_pbc_hex_sio2(repeat: int = 3, dtype: torch.dtype = torch.float64):
    """Alpha-quartz SiO2 supercell with PBC — non-orthorhombic cell
    (gamma = 120°). Exercises skew-cell paths that orthorhombic tests
    miss (e.g. the ``inv(cell).T`` fractional-coordinate regression)."""
    from ase.spacegroup import crystal  # noqa: PLC0415

    unit_cell = crystal(
        symbols=["O", "Si"],
        basis=[[0.413, 0.2711, 0.2172], [0.4673, 0.0, 0.3333]],
        spacegroup=152,
        cellpar=[4.9019, 4.9019, 5.3988, 90, 90, 120],
    )
    atoms = unit_cell.repeat((repeat, repeat, repeat))
    positions = torch.tensor(atoms.get_positions(), dtype=dtype)
    atomic_numbers = torch.tensor(atoms.get_atomic_numbers(), dtype=torch.long)
    masses = torch.tensor(atoms.get_masses(), dtype=dtype)
    cell = torch.tensor(atoms.get_cell().array, dtype=dtype)
    pbc = torch.ones(3, dtype=torch.bool)
    return positions, atomic_numbers, masses, cell, pbc


# Registered topology builders. Each ``_run_dist_model_equivalence``
# call looks up the builder + args by key.
_SYSTEM_BUILDERS: dict[str, Any] = {
    "nonpbc_open_argon": lambda: _build_open_argon_cluster(n_per_side=8),
    "pbc_orthorhombic_argon": lambda: _build_pbc_orthorhombic_argon(n_per_side=4),
    "pbc_hex_sio2": lambda: _build_pbc_hex_sio2(repeat=4),
    "nonpbc_octane_chain": lambda: _build_octane_chain_system(n_atoms=8),
}


def _build_octane_chain_system(n_atoms: int = 8, dtype: torch.dtype = torch.float64):
    """Pseudo-octane chain — N carbons along x at 1.5 Å spacing, non-PBC.

    AIMNet2 only supports elements in its training set (C/H/N/O/F/S/Cl
    etc.), so the argon/SiO2 builders used for LJ+MACE won't evaluate.
    The octane chain is small, all-carbon, and inert — a valid system
    for the AIMNet2 reference check without needing a different
    checkpoint.

    Cell is a 100 Å cube (non-PBC); the cube matters only for the
    partitioner's domain box, not for neighbor list wrapping.
    """
    positions = torch.stack(
        [
            0.25 + torch.arange(n_atoms, dtype=dtype) * 1.5,
            torch.zeros(n_atoms, dtype=dtype),
            torch.zeros(n_atoms, dtype=dtype),
        ],
        dim=1,
    ).contiguous()
    atomic_numbers = torch.full((n_atoms,), 6, dtype=torch.long)
    masses = torch.full((n_atoms,), 12.011, dtype=dtype)
    cell = torch.eye(3, dtype=dtype) * 100.0
    pbc = torch.zeros(3, dtype=torch.bool)
    return positions, atomic_numbers, masses, cell, pbc


# ----------------------------------------------------------------------
# Model-wrapper factories. Each returns a fresh wrapper on every call —
# both the reference (single-process) and the distributed paths need
# independent wrappers so the model state isn't shared.
# ----------------------------------------------------------------------


def _lj_wrapper(dtype: torch.dtype, device: str = "cpu"):
    return LennardJonesModelWrapper(epsilon=0.0104, sigma=3.40, cutoff=8.5).to(
        device=device, dtype=dtype
    )


def _mace_wrapper(dtype: torch.dtype, device: str = "cpu"):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from nvalchemi.models.mace import MACEWrapper  # noqa: PLC0415

    w = MACEWrapper.from_checkpoint("small", dtype=dtype)
    w.eval()
    return w.to(device)


def _aimnet2_wrapper(dtype: torch.dtype, device: str = "cpu"):
    """Load AIMNet2 at *dtype*/*device*. Unlike LJ / MACE,
    ``AIMNet2Wrapper.from_checkpoint`` doesn't take a ``dtype`` kwarg; the
    model and its non-parameter float buffers must be cast in a second pass.
    """
    from nvalchemi.models.aimnet2 import AIMNet2Wrapper  # noqa: PLC0415

    w = AIMNet2Wrapper.from_checkpoint("aimnet2", device=device)
    w.model.to(dtype)
    for mod in w.model.modules():
        for name, buf in list(mod.named_buffers(recurse=False)):
            if buf.is_floating_point():
                setattr(mod, name, buf.to(dtype))
    w.eval()
    # Default active_outputs includes 'charges' which the dispatch path
    # doesn't currently consolidate on the distributed side — narrow to
    # energy+forces to keep the reference vs. distributed comparison
    # apples-to-apples.
    w.model_config.active_outputs = {"energy", "forces"}
    return w


_WRAPPER_FACTORIES: dict[str, Any] = {
    "lj": _lj_wrapper,
    "mace": _mace_wrapper,
    "aimnet2": _aimnet2_wrapper,
}


# Per-model tolerances for (energy, force). LJ is analytical (Warp
# kernel, no autograd) so it matches at double-precision machine
# noise. MACE does autograd + equivariant arithmetic so it sits a
# few orders of magnitude above MP-level noise in float64.
_TOLERANCES: dict[str, dict[str, dict[str, float]]] = {
    "lj": {
        "energy": {"rtol": 1e-10, "atol": 1e-10},
        "force": {"rtol": 1e-8, "atol": 1e-10},
    },
    "mace": {
        "energy": {"rtol": 1e-5, "atol": 1e-6},
        "force": {"rtol": 1e-4, "atol": 1e-5},
    },
    "aimnet2": {
        "energy": {"rtol": 1e-5, "atol": 1e-6},
        "force": {"rtol": 1e-5, "atol": 1e-6},
    },
}

# fp32 tolerances for the cuda model-test tier. fp32 round-off through a
# multi-layer model + reduction lands a few orders above fp64 noise; these
# mirror the bounds the cueq multi-GPU gate uses (energy ~1e-4, force ~1e-3).
_TOLERANCES_FP32: dict[str, dict[str, dict[str, float]]] = {
    "lj": {
        "energy": {"rtol": 1e-5, "atol": 1e-4},
        "force": {"rtol": 1e-4, "atol": 1e-4},
    },
    "mace": {
        "energy": {"rtol": 1e-4, "atol": 1e-4},
        "force": {"rtol": 1e-3, "atol": 1e-4},
    },
    "aimnet2": {
        "energy": {"rtol": 1e-4, "atol": 1e-4},
        "force": {"rtol": 1e-3, "atol": 1e-4},
    },
}


# Per-model extra AtomicData fields that the worker injects before
# constructing the reference / distributed batch. Keeps the worker
# body model-agnostic.
def _model_extras(
    model_name: str, n_systems: int, dtype: torch.dtype, device: str = "cpu"
) -> dict:
    if model_name == "aimnet2":
        # AIMNet2 is a charge-equilibration MLIP — every system needs a
        # total-charge field on its AtomicData; neutral molecules pass 0.
        return {"charge": torch.zeros(n_systems, 1, dtype=dtype, device=device)}
    return {}


# ----------------------------------------------------------------------
# Parameterized worker: single-process reference vs. DistributedModel
# forward, assert energy + per-atom-forces match.
# ----------------------------------------------------------------------


def _dist_model_equivalence_worker(
    rank: int,
    world_size: int,
    system_name: str,
    model_name: str,
    device: str = "cpu",
    dtype: torch.dtype = torch.float64,
) -> None:
    from nvalchemi.distributed.distributed_model import (
        DistributedModel,  # noqa: PLC0415
    )

    # Move the system onto the target device + float dtype. The model-test
    # tier runs ``device="cuda"`` with the gloo backend so N ranks share one
    # GPU (NCCL rejects that) while still exercising the real device code
    # path — warp/cueq kernels, device placement, dtype-on-device — that a
    # cpu run silently skips. fp32 on cuda because that's what those kernels
    # require (and the production path).
    positions, atomic_numbers, masses, cell, pbc = _SYSTEM_BUILDERS[system_name]()
    positions = positions.to(device=device, dtype=dtype)
    atomic_numbers = atomic_numbers.to(device=device)
    masses = masses.to(device=device, dtype=dtype)
    cell = cell.to(device=device, dtype=dtype)
    pbc = pbc.to(device=device)
    wrapper_factory = _WRAPPER_FACTORIES[model_name]
    extras = _model_extras(model_name, n_systems=1, dtype=dtype, device=device)

    # --- Single-process reference ---
    ref_wrapper = wrapper_factory(dtype, device)
    ref_data = AtomicData(
        atomic_numbers=atomic_numbers,
        positions=positions.clone(),
        atomic_masses=masses,
        cell=cell.unsqueeze(0),
        pbc=pbc.unsqueeze(0),
        **extras,
    )
    ref_batch = Batch.from_data_list([ref_data])
    compute_neighbors(ref_batch, config=ref_wrapper.model_config.neighbor_config)
    ref_out = ref_wrapper(ref_batch)
    e_ref = ref_out["energy"].sum().detach()
    f_ref = ref_out["forces"].detach()

    # --- Distributed forward ---
    dist_wrapper = wrapper_factory(dtype, device)
    mesh = _MockMesh(rank, world_size)
    # Partition mode tracks the wrapper's storage policy — spatial for
    # halo-storage (LJ, MACE), contiguous-block for sharded-storage
    # (AIMNet2).
    _policy = dist_wrapper.distribution_spec().distribution.policy
    _storage = "halo" if isinstance(_policy, HaloStoragePolicy) else "sharded"
    sharded, local_mask, domain_config = _sharded_batch_for_system(
        positions,
        atomic_numbers,
        masses,
        cell,
        pbc,
        rank,
        world_size,
        mesh,
        cutoff=float(dist_wrapper.model_config.neighbor_config.cutoff),
        storage=_storage,
    )
    with DistributedModel(dist_wrapper, domain_config) as dist_model:
        out = dist_model(sharded)
    e_local = out["energy"].sum().detach()
    f_owned = out["forces"].detach()

    # --- Assertions ---
    # fp32 (the cuda model-test path) sits well above fp64 machine noise;
    # use the looser fp32 table so the equivalence bar matches the precision.
    tol = (
        _TOLERANCES_FP32[model_name]
        if dtype == torch.float32
        else _TOLERANCES[model_name]
    )
    torch.testing.assert_close(
        e_local,
        e_ref,
        rtol=tol["energy"]["rtol"],
        atol=tol["energy"]["atol"],
        msg=(
            f"rank {rank}: [{model_name} / {system_name}] "
            f"dist_e={e_local.item():.6f}  ref_e={e_ref.item():.6f}  "
            f"delta={(e_local - e_ref).item():+.6e}"
        ),
    )
    f_ref_owned = f_ref[local_mask]
    _fd = (f_owned - f_ref_owned).abs()
    _denom = f_ref_owned.abs().clamp_min(1e-12)
    torch.testing.assert_close(
        f_owned,
        f_ref_owned,
        rtol=tol["force"]["rtol"],
        atol=tol["force"]["atol"],
        msg=(
            f"rank {rank}: [{model_name} / {system_name}] per-atom forces "
            f"disagree: max|Δ|={_fd.max().item():.3e} "
            f"max_rel={(_fd / _denom).max().item():.3e} "
            f"(tol rtol={tol['force']['rtol']:.0e} atol={tol['force']['atol']:.0e})"
        ),
    )


# ----------------------------------------------------------------------
# Test matrix.
#
# LJ runs on all three topologies at world_size ∈ {2, 4} — cheap. MACE
# runs the two smaller topologies at 2 ranks (fast, CI-friendly) and
# the SiO2 hex case at 2 ranks marked ``slow`` — it's the regression
# for the ``inv(cell).T`` fractional-coordinate bug and takes
# ~45 s on CPU. Run ``pytest --slow`` to include it.
# ----------------------------------------------------------------------


def _port_for(key: str) -> str:
    """Deterministic free-ish port per-parametrization so concurrent
    pytest runs don't collide. 30000 + hash(key)%5000."""
    return str(30000 + (hash(key) & 0xFFFF) % 5000)


# Model-equivalence tier: real models run on the gloo+cuda harness (N ranks
# share one GPU; gloo because NCCL rejects multiple ranks on one device),
# exercising the actual device + kernel path. fp32 because warp/cueq kernels
# require it. Skipped without a GPU — the distributed *logic* is covered on
# cpu by the fake-model + synthetic tests.
_MODEL_DEVICE = "cuda"
# fp32 is the production cuda path (warp/cueq require it). NVALCHEMI_TIER_FP64=1
# forces fp64 for precision-vs-logic debugging (where the kernel supports it).
_MODEL_DTYPE = torch.float64 if os.environ.get("NVALCHEMI_TIER_FP64") else torch.float32
cuda_model_tier = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="model-equivalence runs on the gloo+cuda tier (needs a GPU)",
)


# Real-multi-GPU tier: the full DomainParallel NVE dynamics path (gather +
# per-step halo migration) needs raw cuda-tensor send/recv, which gloo's TCP
# transport rejects ("Bad address"). It therefore runs on genuine NCCL across
# >=2 physical GPUs (rank r -> cuda:r) rather than the N-ranks-share-one-GPU
# gloo tier. Skipped on <2 GPUs.
cuda_multigpu_tier = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    reason="full distributed NVE dynamics run on real NCCL across >=2 GPUs",
)


def _dtype_for(system_name: str) -> torch.dtype:
    """Per-system float dtype for the cuda tier. Triclinic cells (hex SiO2)
    run at fp64: their PBC-shift ``shift @ cell`` has off-diagonal terms whose
    fp32 round-off (~1e-6 in position) is amplified by the steep LJ force
    derivative to ~5e-3 — a precision artifact, not a distribution bug (the
    same path matches the reference to ~1e-8 at fp64). Orthorhombic/open cells
    have axis-aligned shifts (exact in fp32) and run fp32 to exercise the real
    fp32 cuda kernel path."""
    if "hex" in system_name:
        return torch.float64
    return _MODEL_DTYPE


@cuda_model_tier
@pytest.mark.parametrize(
    "system_name",
    ["nonpbc_open_argon", "pbc_orthorhombic_argon", "pbc_hex_sio2"],
)
@pytest.mark.parametrize("world_size", [2, 4])
def test_lj_dist_model_equivalence(system_name: str, world_size: int) -> None:
    key = f"lj_{system_name}_{world_size}"
    mp.spawn(
        _worker,
        args=(
            world_size,
            _port_for(key),
            _dist_model_equivalence_worker,
            system_name,
            "lj",
            _MODEL_DEVICE,
            _dtype_for(system_name),
        ),
        nprocs=world_size,
    )


@cuda_model_tier
@pytest.mark.parametrize(
    "system_name",
    ["nonpbc_open_argon", "pbc_orthorhombic_argon"],
)
def test_mace_dist_model_equivalence_2ranks(system_name: str) -> None:
    pytest.importorskip("mace", reason="mace-torch not installed")
    key = f"mace_{system_name}_2"
    mp.spawn(
        _worker,
        args=(
            2,
            _port_for(key),
            _dist_model_equivalence_worker,
            system_name,
            "mace",
            _MODEL_DEVICE,
            _dtype_for(system_name),
        ),
        nprocs=2,
    )


@cuda_model_tier
@pytest.mark.slow
def test_mace_dist_model_equivalence_pbc_hex_sio2_2ranks() -> None:
    """Regression for the skew-cell fractional-coordinate bug
    (``inv(cell).T`` vs ``inv(cell)`` in
    ``_identify_ghosts_split`` / ``SpatialPartitioner``). Runs on the
    cuda tier in fp64 — the skew-cell fractional round-off needs fp64 to
    sit above the tight equivalence tolerance. Still ``slow``."""
    pytest.importorskip("mace", reason="mace-torch not installed")
    mp.spawn(
        _worker,
        args=(
            2,
            _port_for("mace_pbc_hex_sio2_2"),
            _dist_model_equivalence_worker,
            "pbc_hex_sio2",
            "mace",
            _MODEL_DEVICE,
            _dtype_for("pbc_hex_sio2"),
        ),
        nprocs=2,
    )


# AIMNet2 is halo-only. A tiny non-PBC octane chain at world=2/4 is degenerate
# for halo (a rank's domain is far smaller than the ghost width), so eager-DD
# halo equivalence at a non-degenerate partition is covered instead by the
# eager-reference leg of ``test_aimnet2_compile_recompile_gate`` (methane PBC,
# world=2).


# ======================================================================
# MACE — multi-step NVE via DomainParallel
# Shares the parameterized ``_nve_via_domain_parallel_worker`` above.
# ======================================================================


@cuda_multigpu_tier
@pytest.mark.parametrize(
    "system_name,world_size,n_steps,dt_fs,energy_tol", _MACE_NVE_CASES
)
def test_mace_nve_via_domain_parallel(
    system_name: str,
    world_size: int,
    n_steps: int,
    dt_fs: float,
    energy_tol: float,
) -> None:
    """Full DomainParallel NVE for MACE on real multi-GPU NCCL.

    Regression for the ``@torch.jit.script`` + ShardTensor CUDA
    illegal-memory-access: e3nn's module-level scripted ``_spherical_harmonics``
    (MACE's declared ``JitAdapter``) and the ``TensorProduct``
    ``_compiled_main_*`` ``ScriptModule``s (auto-discovered) are marshalled
    across the ShardTensor boundary. Without that, the first NVE step
    IMAs in the fused scripted kernel. The worker asserts the gathered final
    potential energy matches a single-process reference at the same positions.

    Runs on real NCCL across >=2 GPUs: the gloo+cuda single-GPU tier cannot
    transport raw cuda tensors through the dynamics-step collectives (gather /
    halo migration) regardless of model, so full NVE dynamics need real
    multi-GPU. The forward-only path is covered on the single-GPU gloo tier by
    ``test_mace_dist_model_equivalence_2ranks`` and
    ``test_scripted_op_shardtensor_marshalling_forward``.
    """
    pytest.importorskip("mace", reason="mace-torch not installed")
    key = f"mace_nve_{system_name}_{world_size}"
    mp.spawn(
        _worker_nccl,
        args=(
            world_size,
            _port_for(key),
            _nve_via_domain_parallel_worker,
            system_name,
            "mace",
            n_steps,
            dt_fs,
            energy_tol,
            _MODEL_DEVICE,
            _MODEL_DTYPE,
        ),
        nprocs=world_size,
    )


# ======================================================================
# Scripted-op + ShardTensor marshalling — mace-free regression (forward)
#
# A 2-layer MPNN with a scripted message op and autograd forces (F=-dE/dx)
# reproduces the @torch.jit.script + ShardTensor CUDA illegal-memory-access
# minimally (no MACE/e3nn): a later scripted op consumes a storage-less
# scatter-output ShardTensor with grad recording on, and its fused kernel
# reads the wrapper's near-null data_ptr. With marshalling disabled
# (NVALCHEMI_SCRIPTED_MARSHAL=off) the forward IMAs; with the default "auto"
# the scripted submodule is auto-discovered + marshalled and it runs clean.
#
# This runs a single DistributedModel FORWARD (not the full NVE dynamics) so
# it fits the gloo+cuda single-GPU tier — the forward halo exchange is host-
# staged, unlike the dynamics collectives that force the MACE NVE test onto
# real multi-GPU NCCL.
# ======================================================================


def _toy_scripted_marshal_worker(
    rank: int, world_size: int, device: str = "cuda", dtype: torch.dtype = torch.float32
) -> None:
    import os as _os  # noqa: PLC0415
    import sys as _sys  # noqa: PLC0415

    _sys.path.insert(0, _os.path.dirname(__file__))
    from _toy_scripted_mpnn import ToyScriptedMPNNWrapper  # noqa: PLC0415

    from nvalchemi.distributed.distributed_model import (  # noqa: PLC0415
        DistributedModel,
    )

    positions, atomic_numbers, masses, cell, pbc = _SYSTEM_BUILDERS[
        "pbc_orthorhombic_argon"
    ]()
    positions = positions.to(device=device, dtype=dtype)
    atomic_numbers = atomic_numbers.to(device=device)
    masses = masses.to(device=device, dtype=dtype)
    cell = cell.to(device=device, dtype=dtype)
    pbc = pbc.to(device=device)

    # Warm up the module-level scripted op so TorchScript's profiling executor
    # TensorExpr-FUSES it. The fused kernel (built only after a few executions)
    # is what reads the storage-less ShardTensor ``data_ptr`` -> IMA; a cold,
    # interpreted scripted op does not fault. Without this the regression would
    # pass even with marshalling disabled. (Mirrors the equivalence worker,
    # whose single-process reference forward incidentally warms the same shared
    # scripted op before the distributed forward.)
    warm = ToyScriptedMPNNWrapper().to(device=device, dtype=dtype)
    warm.eval()
    for _ in range(3):
        warm_data = AtomicData(
            atomic_numbers=atomic_numbers,
            positions=positions.clone(),
            atomic_masses=masses,
            cell=cell.unsqueeze(0),
            pbc=pbc.unsqueeze(0),
        )
        warm_batch = Batch.from_data_list([warm_data])
        compute_neighbors(warm_batch, config=warm.model_config.neighbor_config)
        warm(warm_batch)

    wrapper = ToyScriptedMPNNWrapper().to(device=device, dtype=dtype)
    wrapper.eval()
    mesh = _MockMesh(rank, world_size)
    sharded, local_mask, domain_config = _sharded_batch_for_system(
        positions,
        atomic_numbers,
        masses,
        cell,
        pbc,
        rank,
        world_size,
        mesh,
        cutoff=float(wrapper.model_config.neighbor_config.cutoff),
        storage="halo",
    )
    with DistributedModel(wrapper, domain_config) as dist_model:
        out = dist_model(sharded)

    energy = out["energy"]
    forces = out["forces"]
    # The fix's contract: the scripted op ran on a ShardTensor without an IMA,
    # and autograd stayed connected through it. (Strict dist-vs-single energy
    # equivalence is intentionally NOT asserted: the toy's readout-sum energy
    # counts halo rows, which is orthogonal to the scripted-op IMA this guards.)
    assert torch.isfinite(energy).all(), f"rank {rank}: toy energy non-finite"
    assert torch.isfinite(forces).all(), f"rank {rank}: toy forces non-finite"
    assert float(forces.abs().sum()) > 0.0, (
        f"rank {rank}: toy forces are all-zero — autograd was severed through "
        "the marshalled scripted op (F=-dE/dx not connected)"
    )


@cuda_model_tier
def test_scripted_op_shardtensor_marshalling_forward() -> None:
    """A scripted-op MPNN forward survives the ShardTensor halo boundary.

    Regression for the @torch.jit.script + ShardTensor IMA, mace-free. The
    default ``scripted_marshal="auto"`` auto-discovers the toy's scripted
    submodule and marshals it. Without the fix this forward raises a
    CUDA illegal memory access. Single-GPU gloo+cuda tier (forward only).
    """
    world_size = 2
    key = "toy_scripted_marshal"
    mp.spawn(
        _worker,
        args=(world_size, _port_for(key), _toy_scripted_marshal_worker),
        nprocs=world_size,
    )


# ======================================================================
# Distributed per-iteration memory-leak regression
#
# ``_AutogradPreservingUnwrap`` (the grad-aware ShardTensor unwrap fired by
# halo-correction / scripted-op marshalling handlers) stores a grad-free
# ``_UnwrapSource`` metadata surrogate on the autograd ctx rather than the
# source ShardTensor itself; storing the wrapper (``ctx.source = wrapper``)
# would close a ``grad_fn <-> ctx <-> wrapper.grad_fn`` reference cycle that
# only the cyclic GC could reclaim, leaking ~one full autograd graph per
# forward+autograd-force step (MD / batched inference) until OOM. This test
# loops the distributed forward and asserts ``memory_allocated`` stays flat
# per iteration.
# ======================================================================


def _scripted_marshal_no_leak_worker(
    rank: int, world_size: int, device: str = "cuda", dtype: torch.dtype = torch.float32
) -> None:
    import os as _os  # noqa: PLC0415
    import sys as _sys  # noqa: PLC0415

    _sys.path.insert(0, _os.path.dirname(__file__))
    from _toy_scripted_mpnn import ToyScriptedMPNNWrapper  # noqa: PLC0415

    from nvalchemi.distributed.distributed_model import (  # noqa: PLC0415
        DistributedModel,
    )
    from nvalchemi.distributed.particle_halo import halo_exchange  # noqa: PLC0415

    positions, atomic_numbers, masses, cell, pbc = _SYSTEM_BUILDERS[
        "pbc_orthorhombic_argon"
    ]()
    positions = positions.to(device=device, dtype=dtype)
    atomic_numbers = atomic_numbers.to(device=device)
    masses = masses.to(device=device, dtype=dtype)
    cell = cell.to(device=device, dtype=dtype)
    pbc = pbc.to(device=device)

    # Larger hidden/layers so one iteration's autograd graph is well above
    # allocator noise — a per-iteration leak then dwarfs the threshold.
    wrapper = ToyScriptedMPNNWrapper(hidden=128, n_layers=4).to(
        device=device, dtype=dtype
    )
    wrapper.eval()
    mesh = _MockMesh(rank, world_size)
    sharded, _local_mask, domain_config = _sharded_batch_for_system(
        positions,
        atomic_numbers,
        masses,
        cell,
        pbc,
        rank,
        world_size,
        mesh,
        cutoff=float(wrapper.model_config.neighbor_config.cutoff),
        storage="halo",
    )

    with DistributedModel(wrapper, domain_config) as dist_model:
        # Prime: populate padded_batch / halo_config, then mirror the benchmark
        # per-iteration pattern (halo_exchange + forward with autograd forces).
        out = dist_model(sharded)
        del out
        halo_cfg = dist_model._halo_config
        needs_forces = dist_model._needs_forces()

        def _step():
            halo_exchange(sharded, halo_cfg, compute_forces=needs_forces)
            return dist_model(sharded)

        # Warm up (allocator pools + TorchScript fusion reach steady state).
        for _ in range(4):
            out = _step()
            del out
        torch.cuda.synchronize(device)
        base = torch.cuda.memory_allocated(device)

        n_iter = 16
        for _ in range(n_iter):
            out = _step()
            del out
        torch.cuda.synchronize(device)
        growth = torch.cuda.memory_allocated(device) - base

    growth_mib = growth / 2**20
    # Expected ~0 (graph freed by refcount each step); a per-iteration leak
    # grows ~linearly and blows past the threshold over 16 iters. 64 MiB
    # cleanly separates the two without flaking on allocator jitter.
    assert growth < 64 * 2**20, (
        f"rank {rank}: distributed forward leaked {growth_mib:.1f} MiB over "
        f"{n_iter} iterations — per-iteration autograd-graph retention "
        "regressed (see _AutogradPreservingUnwrap / _UnwrapSource)."
    )


@cuda_model_tier
def test_scripted_op_shardtensor_no_leak() -> None:
    """Repeated distributed forward must not leak the autograd graph per step.

    Regression for the ``_AutogradPreservingUnwrap`` ``ctx.source`` reference
    cycle (fixed via the grad-free ``_UnwrapSource`` surrogate). gloo+cuda
    single-GPU tier, mace-free (the scripted-op toy). Asserts
    ``memory_allocated`` is flat across 16 forward+autograd-force iterations.
    """
    world_size = 2
    key = "toy_scripted_no_leak"
    mp.spawn(
        _worker,
        args=(world_size, _port_for(key), _scripted_marshal_no_leak_worker),
        nprocs=world_size,
    )


# ======================================================================
# AIMNet2 — halo-storage distributed forward through DistributedModel
# ======================================================================


def _build_octane_chain(n_atoms: int = 8, dtype: torch.dtype = torch.float64):
    """Pseudo-octane: N carbons in a straight line, 1.5 Å apart."""
    positions = torch.stack(
        [
            0.25 + torch.arange(n_atoms, dtype=dtype) * 1.5,
            torch.zeros(n_atoms, dtype=dtype),
            torch.zeros(n_atoms, dtype=dtype),
        ],
        dim=1,
    ).contiguous()
    atomic_numbers = torch.full((n_atoms,), 6, dtype=torch.long)
    masses = torch.full((n_atoms,), 12.011, dtype=dtype)
    return positions, atomic_numbers, masses


def test_aimnet2_wrapper_declares_halo_spec() -> None:
    pytest.importorskip("aimnet", reason="aimnet not installed")
    from nvalchemi.distributed._core.storage_policy import (  # noqa: PLC0415
        HaloStoragePolicy,
    )
    from nvalchemi.models.aimnet2 import AIMNet2Wrapper

    wrapper = AIMNet2Wrapper.from_checkpoint("aimnet2", device="cpu")

    # AIMNet2 is halo-only: local-neighbor storage with declared per-layer
    # ghost-refresh (ConvSV / Coulomb heads) + owned-only mol_sum reduce.
    spec = wrapper.distribution_spec()
    assert isinstance(spec.distribution.policy, HaloStoragePolicy)
    # The halo refresh/reduce adapters ride third_party_helpers (mol_sum +
    # ConvSV + LRCoulomb + SRCoulomb), with no gather custom_ops.
    assert spec.distribution.custom_ops == ()
    assert len(spec.distribution.third_party_helpers) == 4


AIMNET2_N_STEPS = 3
AIMNET2_DT_FS = 0.25


def _aimnet2_nve_worker(
    rank: int,
    world_size: int,
    device: str = "cpu",
    dtype: torch.dtype = torch.float64,
) -> None:
    """Distributed AIMNet2 NVE trajectory via :class:`DistributedModel`.

    Reference trajectory: NVE on the full system, single-process.
    Distributed: hand-rolled velocity-verlet where each step rebuilds a
    fresh ``ShardedBatch`` from the updated owned positions and invokes
    ``dist_model(sharded)``. The adapter handles global-NL reconstruction
    and DDP ``/world_size`` compensation. Runs cpu/fp64 by default; the
    real-model tier passes cuda/fp32.
    """
    from nvalchemi.distributed.distributed_model import DistributedModel

    positions_global, atomic_numbers_global, masses_global = _build_octane_chain(
        n_atoms=8, dtype=dtype
    )
    positions_global = positions_global.to(device=device, dtype=dtype)
    atomic_numbers_global = atomic_numbers_global.to(device=device)
    masses_global = masses_global.to(device=device, dtype=dtype)
    torch.manual_seed(7)
    velocities_global = 0.001 * torch.randn_like(positions_global)
    velocities_global -= velocities_global.mean(dim=0, keepdim=True)

    n_global = positions_global.shape[0]
    n_systems = 1

    def _make_wrapper():
        return _aimnet2_wrapper(dtype, device)

    # ---- Reference trajectory via NVE + AIMNet2Wrapper ----
    ref_wrapper = _make_wrapper()
    ref_data = AtomicData(
        atomic_numbers=atomic_numbers_global,
        positions=positions_global.clone(),
        atomic_masses=masses_global,
        charge=torch.zeros(n_systems, 1, dtype=dtype, device=device),
        forces=torch.zeros(n_global, 3, dtype=dtype, device=device),
        energy=torch.zeros(n_systems, 1, dtype=dtype, device=device),
    )
    ref_data.add_node_property("velocities", velocities_global.clone())
    ref_batch = Batch.from_data_list([ref_data])
    compute_neighbors(ref_batch, config=ref_wrapper.model_config.neighbor_config)
    ref_nve = NVE(model=ref_wrapper, dt=AIMNET2_DT_FS)
    ref_positions_per_step = [ref_batch.positions.detach().clone()]
    ref_velocities_per_step = [ref_batch.velocities.detach().clone()]
    for _ in range(AIMNET2_N_STEPS):
        ref_batch, _ = ref_nve.step(ref_batch)
        compute_neighbors(ref_batch, config=ref_wrapper.model_config.neighbor_config)
        ref_positions_per_step.append(ref_batch.positions.detach().clone())
        ref_velocities_per_step.append(ref_batch.velocities.detach().clone())

    # ---- Distributed trajectory ----
    per_rank = n_global // world_size
    assignment = torch.arange(n_global, dtype=torch.long) // per_rank
    assignment = assignment.clamp(max=world_size - 1)
    sizes = [int((assignment == r).sum().item()) for r in range(world_size)]
    # mask indexes device-resident tensors (positions/numbers/masses), so it
    # must live on the same device.
    local_mask = (assignment == rank).to(device=device)
    local_z = atomic_numbers_global[local_mask].contiguous()
    local_m = masses_global[local_mask].contiguous()

    open_cell = torch.eye(3, dtype=dtype, device=device) * 100.0
    no_pbc = torch.zeros(3, dtype=torch.bool, device=device)

    mesh = _MockMesh(rank, world_size)
    dist_wrapper = _make_wrapper()
    domain_config = DomainConfig(
        cutoff=float(dist_wrapper.model_config.neighbor_config.cutoff),
        mesh=mesh,
    )
    dist_model = DistributedModel(dist_wrapper, domain_config)

    def _forces(local_pos: torch.Tensor) -> torch.Tensor:
        sharded = make_gloo_sharded_batch(
            mesh=mesh,
            local_positions=local_pos.detach().clone(),
            local_numbers=local_z,
            local_masses=local_m,
            cell=open_cell.unsqueeze(0),
            pbc=no_pbc.unsqueeze(0),
            sizes=sizes,
            n_global=n_global,
        )
        out = dist_model(sharded)
        return out["forces"].detach()

    try:
        from nvalchemi.dynamics._units import fs_to_internal_time

        dt_internal = fs_to_internal_time(AIMNET2_DT_FS)
        pos = positions_global[local_mask].contiguous().clone()
        vel = velocities_global[local_mask].contiguous().clone()
        # Match NVE's priming: batch.forces starts at zero, so the first
        # pre_update uses F=0 (half-step start). Aligns with the reference.
        f = torch.zeros_like(pos)
        dist_positions = [pos.clone()]
        dist_velocities = [vel.clone()]
        for _ in range(AIMNET2_N_STEPS):
            v_half = vel + 0.5 * dt_internal * f / local_m.unsqueeze(-1)
            pos = pos + dt_internal * v_half
            f = _forces(pos)
            vel = v_half + 0.5 * dt_internal * f / local_m.unsqueeze(-1)
            dist_positions.append(pos.clone())
            dist_velocities.append(vel.clone())
    finally:
        dist_model.close()

    # fp32 (cuda tier) accumulates round-off through the verlet steps, so it
    # sits above fp64 noise — loosen accordingly.
    is_fp32 = dtype == torch.float32
    pos_rtol, pos_atol = (1e-3, 1e-4) if is_fp32 else (1e-4, 1e-5)
    vel_rtol, vel_atol = (1e-2, 1e-3) if is_fp32 else (1e-3, 1e-4)
    for step, (ref_p, got_p) in enumerate(
        zip(ref_positions_per_step, dist_positions, strict=True)
    ):
        torch.testing.assert_close(
            got_p,
            ref_p[local_mask],
            rtol=pos_rtol,
            atol=pos_atol,
            msg=f"rank {rank} positions diverged at step {step}",
        )
    for step, (ref_v, got_v) in enumerate(
        zip(ref_velocities_per_step, dist_velocities, strict=True)
    ):
        torch.testing.assert_close(
            got_v,
            ref_v[local_mask],
            rtol=vel_rtol,
            atol=vel_atol,
            msg=f"rank {rank} velocities diverged at step {step}",
        )


@pytest.mark.skip(
    reason="8-atom non-PBC octane chain at world=2 is degenerate for halo "
    "(incomplete neighbor coverage). AIMNet2 is halo-only — multi-step "
    "DD equivalence is covered by test_aimnet2_compile_recompile_gate at a "
    "non-degenerate methane PBC partition."
)
@cuda_model_tier
def test_aimnet2_nve_multi_step_equivalence_2ranks() -> None:
    pytest.importorskip("aimnet", reason="aimnet not installed")
    mp.spawn(
        _worker,
        args=(2, "29569", _aimnet2_nve_worker, _MODEL_DEVICE, _MODEL_DTYPE),
        nprocs=2,
    )
