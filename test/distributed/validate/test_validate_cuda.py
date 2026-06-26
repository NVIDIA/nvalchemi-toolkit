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

"""End-to-end ``trace_and_validate`` against real wrappers on a real GPU.

CUDA-gated. The single-GPU multi-process spawn under NCCL is the
validator's entry-level behaviour — anything CPU-or-Gloo-based
risks GPU/CPU drift (per design discussion) and is out of scope for
this test.

Each test
---------
- builds a small ``sample_batch`` on cuda:0,
- calls ``trace_and_validate(model_factory, sample_batch, world_size=2)``,
- asserts ``report.ok`` and that the inferred / fixed spec serializes.

Run with::

    pytest test/distributed/test_validate_cuda.py -v
"""

from __future__ import annotations

# IMPORTANT: set WARP_CACHE_PATH BEFORE any nvalchemi/nvalchemiops
# import. The default ``~/.cache/warp/`` is read-only in some sandboxed
# dev environments; warp's cache_dir is fixed at first ``warp.init()``
# call, so once a wrapper accidentally triggers warp before this is
# set, no later override takes effect (PME's ``spline_spread`` then
# fails with ``OSError: Read-only file system``). Setting it here at
# import time protects every test in this file uniformly.
import os as _os  # noqa: E402
import tempfile as _tempfile  # noqa: E402

_os.environ.setdefault(
    "WARP_CACHE_PATH",
    _os.path.join(_tempfile.gettempdir(), "nvalchemi-validate-warp-cache"),
)

# fairchem's UMA loader does an HF Hub etag check on every from_checkpoint
# call. With the model already cached locally, that's pure latency and
# fails outright in offline / sandboxed environments. Force the cache-only
# path so the test doesn't depend on network reachability.
_os.environ["HF_HUB_OFFLINE"] = "1"
# httpx auto-picks up ftp_proxy=socks5h://… and instantiates a SOCKS
# transport even with HF_HUB_OFFLINE — fails in environments that lack
# the optional socksio dep. Strip proxy env vars so the cache-only path
# never tries to construct an httpx client at all.
for _v in (
    "ftp_proxy",
    "FTP_PROXY",
    "grpc_proxy",
    "GRPC_PROXY",
    "all_proxy",
    "ALL_PROXY",
    "https_proxy",
    "HTTPS_PROXY",
    "http_proxy",
    "HTTP_PROXY",
):
    _os.environ.pop(_v, None)

import pytest  # noqa: E402
import torch  # noqa: E402

cuda_required = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA required: trace_and_validate uses single-GPU multi-process spawn",
)


def _make_lj_wrapper():
    from nvalchemi.models.lj import LennardJonesModelWrapper

    return LennardJonesModelWrapper(
        epsilon=0.0103, sigma=3.4, cutoff=10.0, half_list=False
    )


def _make_argon_batch(n_per_side: int = 5, lattice: float = 3.4):
    """Cubic argon lattice on cuda:0. ``n_per_side=5`` → 125 atoms,
    enough that a 2-rank split has non-trivial halo overlap and the
    validator exercises real cross-rank communication."""
    from nvalchemi.data import AtomicData, Batch

    coords = torch.arange(n_per_side, dtype=torch.float32) * lattice
    gx, gy, gz = torch.meshgrid(coords, coords, coords, indexing="ij")
    positions = torch.stack([gx.flatten(), gy.flatten(), gz.flatten()], dim=-1).cuda()
    n = positions.shape[0]
    box = n_per_side * lattice
    cell = torch.eye(3, device="cuda") * box
    data = AtomicData(
        positions=positions,
        atomic_numbers=torch.full((n,), 18, dtype=torch.long, device="cuda"),
        cell=cell.unsqueeze(0),
        pbc=torch.tensor([[True, True, True]], device="cuda"),
    )
    return Batch.from_data_list([data], device="cuda")


def _make_argon_batch_for_mpnn(
    n_per_side: int = 4, dtype: torch.dtype = torch.float64, seed: int = 0
):
    """Argon batch sized for MPNN-style cutoffs.

    Mirrors ``test_distributed_models.py::_build_pbc_orthorhombic_argon``
    — LJ-equilibrium spacing (~4.007 Å) plus 0.05 Å random jitter, so
    the box (~4×spacing ≈ 16 Å) sits comfortably above MACE's 6 Å cutoff
    and atoms aren't on partition boundaries. The plain ``_make_argon_batch``
    above (lattice=3.4 Å, no jitter) is fine for LJ/Ewald/PME custom-op
    paths, but MACE's halo-aware message passing exposes a real
    correctness gap when cutoff approaches box/2 on a perfect lattice
    (multi-rank energy drifts ~0.6%).
    """
    from nvalchemi.data import AtomicData, Batch

    spacing = 2 ** (1.0 / 6.0) * 3.40 * 1.05  # ≈ 4.007
    coords = torch.arange(n_per_side, dtype=dtype) * spacing
    gx, gy, gz = torch.meshgrid(coords, coords, coords, indexing="ij")
    positions = torch.stack([gx.flatten(), gy.flatten(), gz.flatten()], dim=-1)
    g = torch.Generator().manual_seed(seed)
    positions = positions + 0.05 * torch.randn(
        positions.shape, generator=g, dtype=dtype
    )
    n = positions.shape[0]
    box = n_per_side * spacing
    positions = positions - torch.floor(positions / box) * box
    cell = torch.eye(3, dtype=dtype) * box
    data = AtomicData(
        positions=positions.cuda(),
        atomic_numbers=torch.full((n,), 18, dtype=torch.long, device="cuda"),
        cell=cell.unsqueeze(0).cuda(),
        pbc=torch.tensor([[True, True, True]], device="cuda"),
    )
    return Batch.from_data_list([data], device="cuda")


@cuda_required
def test_lj_passes_initial_inference():
    """LJ has a clean ``halo_correction`` spec; the validator should
    infer it from the wrapper's existing ``distribution_spec``,
    spawn 2 ranks, run, and pass without auto-fix engaging."""
    from nvalchemi.distributed.validate import trace_and_validate

    report = trace_and_validate(
        _make_lj_wrapper,
        _make_argon_batch(n_per_side=5),
        world_size=2,
        device="cuda:0",
        atol=1e-4,
        rtol=0.0,
        auto_fix=True,
    )
    assert report.ok, report.next_action
    assert len(report.attempts) == 1, (
        "LJ should pass on first attempt (no auto-fix needed); "
        f"got {len(report.attempts)} attempts. Last rationale: "
        f"{report.attempts[-1].rationale}"
    )
    assert report.fix_applied is None


@cuda_required
def test_spec_round_trips_through_disk(tmp_path):
    """End-to-end: validate, save the spec, reload it, build a
    ``DistributedModel`` from the reloaded spec — same wrapper class,
    new spec instance, same behaviour."""
    from nvalchemi.distributed.spec import MLIPSpec
    from nvalchemi.distributed.validate import trace_and_validate

    report = trace_and_validate(
        _make_lj_wrapper,
        _make_argon_batch(n_per_side=4),  # 64 atoms — quick
        world_size=2,
        device="cuda:0",
    )
    assert report.ok, report.next_action

    path = tmp_path / "lj_spec.json"
    report.spec.save(path)

    # Reload + assert the on-disk form equals the in-memory spec.
    reloaded = MLIPSpec.load(path)
    assert reloaded == report.spec


@cuda_required
def test_auto_fix_kicks_in_when_initial_spec_is_wrong():
    """Validate the auto-fix path on LJ with a deliberately wrong
    ``all_reduce_outputs={"energy"}``. LJ's energy already routes
    through ``per_system_reduce`` so an extra ``all_reduce`` would
    double the result. The hypothesis engine's
    ``drop-extra-all_reduce`` rule should fire and converge to LJ's
    real spec (no ``all_reduce_outputs``).
    """
    from nvalchemi.distributed.validate import trace_and_validate

    report = trace_and_validate(
        _make_lj_with_wrong_spec,  # module-level so mp.spawn can pickle it
        _make_argon_batch(n_per_side=4),
        world_size=2,
        device="cuda:0",
        auto_fix=True,
        max_fix_attempts=4,
    )
    assert report.ok, report.next_action
    # The first attempt should fail; the fix should drop the bad key.
    assert len(report.attempts) >= 2, (
        f"expected auto-fix to engage; got {len(report.attempts)} attempts"
    )
    assert "all_reduce_outputs" in report.fix_applied
    # Final spec should have no all_reduce_outputs.
    assert report.spec.all_reduce_outputs == frozenset()


def _make_lj_with_wrong_spec():
    """Module-level factory for the auto-fix test — picklable across
    ``mp.spawn``. Wraps ``LennardJonesModelWrapper`` with a deliberately
    wrong ``all_reduce_outputs={"energy"}`` to exercise the drop-extra
    rule.
    """
    return _LJWithWrongSpec()


class _LJWithWrongSpec:
    """LJ wrapper with a deliberately-wrong distribution_spec.

    Wraps :class:`LennardJonesModelWrapper` and overrides the spec to
    include ``all_reduce_outputs={"energy"}`` — the validator should
    catch this and the auto-fix engine should drop the bad key.

    Module-level (not a closure) so it pickles across mp.spawn.
    """

    def __init__(self) -> None:
        self._inner = _make_lj_wrapper()

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def __call__(self, *args, **kwargs):
        return self._inner(*args, **kwargs)

    def to(self, device):
        self._inner = self._inner.to(device)
        return self

    def parameters(self):
        return self._inner.parameters()

    @property
    def model_config(self):
        return self._inner.model_config

    @property
    def distribution_spec(self):
        from dataclasses import replace

        base = self._inner.distribution_spec
        # Add a deliberately-wrong all_reduce_outputs.
        return replace(base, all_reduce_outputs=frozenset({"energy"}))


# ----------------------------------------------------------------------
# Wrappers that ship their own distribution_spec — should pass on
# initial inference (no auto-fix).
# ----------------------------------------------------------------------


def _make_ewald_wrapper():
    from nvalchemi.models.ewald import EwaldModelWrapper

    return EwaldModelWrapper(cutoff=10.0)


def _make_charged_argon_batch(n_per_side: int = 5, lattice: float = 3.4):
    """Argon-with-fake-charges batch for Ewald/PME — alternating ±1 e."""
    import torch

    from nvalchemi.data import AtomicData, Batch

    coords = torch.arange(n_per_side, dtype=torch.float32) * lattice
    gx, gy, gz = torch.meshgrid(coords, coords, coords, indexing="ij")
    positions = torch.stack([gx.flatten(), gy.flatten(), gz.flatten()], dim=-1).cuda()
    n = positions.shape[0]
    box = n_per_side * lattice
    cell = torch.eye(3, device="cuda") * box
    charges = torch.tensor(
        [1.0 if i % 2 == 0 else -1.0 for i in range(n)],
        dtype=torch.float32,
        device="cuda",
    )
    data = AtomicData(
        positions=positions,
        atomic_numbers=torch.full((n,), 18, dtype=torch.long, device="cuda"),
        cell=cell.unsqueeze(0),
        pbc=torch.tensor([[True, True, True]], device="cuda"),
        charges=charges,
    )
    return Batch.from_data_list([data], device="cuda")


@cuda_required
def test_ewald_passes_initial_inference():
    """Ewald has the same halo+halo_correction+system_reductions
    pattern as LJ, plus 2 staged-binding custom_ops. Should pass on
    first attempt."""
    from nvalchemi.distributed.validate import trace_and_validate

    report = trace_and_validate(
        _make_ewald_wrapper,
        _make_charged_argon_batch(n_per_side=4),
        world_size=2,
        device="cuda:0",
        auto_fix=True,
    )
    assert report.ok, report.next_action
    assert len(report.attempts) == 1, (
        f"Ewald should pass on first attempt; got {len(report.attempts)}. "
        f"Last rationale: {report.attempts[-1].rationale}"
    )


def _make_pme_wrapper():
    from nvalchemi.models.pme import PMEModelWrapper

    return PMEModelWrapper(cutoff=10.0)


@cuda_required
def test_pme_passes_initial_inference():
    """PME has 4 custom_ops (spline_spread x2 + total_charge x2). Should
    pass on first attempt with the staged-bindings spec."""
    from nvalchemi.distributed.validate import trace_and_validate

    report = trace_and_validate(
        _make_pme_wrapper,
        _make_charged_argon_batch(n_per_side=4),
        world_size=2,
        device="cuda:0",
        auto_fix=True,
    )
    assert report.ok, report.next_action
    assert len(report.attempts) == 1, (
        f"PME should pass on first attempt; got {len(report.attempts)}. "
        f"Last rationale: {report.attempts[-1].rationale}"
    )


# ----------------------------------------------------------------------
# MACE — halo storage + halo_correction. The "small" checkpoint has no
# custom_ops; the cueq fused-kernel path is exercised separately when
# enable_cueq=True (not tested here — small box).
# ----------------------------------------------------------------------


def _make_mace_wrapper():
    from nvalchemi.models.mace import MACEWrapper

    # Load on cuda — both reference run (single-process) and the
    # spawned workers expect the wrapper's parameters to be on the
    # same device as the input batch. The factory is invoked once on
    # the launcher side and once per rank inside spawn, so each gets
    # a pristine cuda-resident model.
    #
    # fp64 because MACE's equivariant tensor-product layers accumulate
    # enough round-off in fp32 that a 2-rank partition (different
    # reduction order vs single-process) drifts ~0.6% on energy /
    # ~3e-2 on forces — pure noise, not a spec correctness signal.
    # The production multi-GPU MACE test
    # (``test_distributed_models.py::test_mace_*``) standardises on
    # fp64 for the same reason and clears its 1e-4 NVE tolerance.
    return MACEWrapper.from_checkpoint("small", dtype=torch.float64).to("cuda")


@cuda_required
def test_mace_passes_initial_inference():
    """MACE is the canonical halo-aware MPNN. Should pass on first
    inference; if auto-fix engages here, that's a real bug worth
    investigating."""
    pytest.importorskip("mace")
    from nvalchemi.distributed.validate import trace_and_validate

    report = trace_and_validate(
        _make_mace_wrapper,
        _make_argon_batch_for_mpnn(n_per_side=8),
        world_size=2,
        device="cuda:0",
        auto_fix=True,
    )
    print(report.spec)
    assert report.ok, report.next_action
    assert len(report.attempts) == 1, (
        f"MACE should pass on first attempt; got {len(report.attempts)}. "
        f"Last rationale: {report.attempts[-1].rationale}"
    )


# ----------------------------------------------------------------------
# AIMNet2 — halo storage + PythonAdapter helpers. The runtime helpers
# (``mol_sum``, ``calc_masks``) are declared as ``PythonAdapter`` entries on
# the spec's ``third_party_helpers`` — the framework's ``AdapterRegistry``
# installs + restores them. Validates that ``third_party_helpers``
# round-trips through the spawn boundary (``spec.to_dict()`` →
# ``from_dict`` resolves the replacement functions by qualname) and that
# the framework installs them on the worker side.
# ----------------------------------------------------------------------


def _make_aimnet2_wrapper():
    """fp32 AIMNet2 on cuda. AIMNet2's internal AEV uses
    ``aimnet.kernels.conv_sv_2d_sp_wp`` — a Warp kernel registered for
    ``vec4f`` only — so the cuda forward path is fp32-only. The
    existing ``test_distributed_models.py::_aimnet2_wrapper(dtype=fp64)``
    only works because that test runs on CPU, where ``conv_sv_2d`` has
    a non-Warp path. We keep ``model.to(fp32)`` (the checkpoint's
    native precision) and narrow ``active_outputs`` to
    ``{energy, forces}`` so the dispatch path's output-consolidation
    contract matches single-process (charges aren't currently
    consolidated on the distributed side).
    """
    from nvalchemi.models.aimnet2 import AIMNet2Wrapper

    w = AIMNet2Wrapper.from_checkpoint("aimnet2", device="cuda")
    w.eval()
    w.model_config.active_outputs = {"energy", "forces"}
    return w


def _make_octane_chain_for_aimnet2(n_atoms: int = 8, bond: float = 1.5):
    """Pseudo-octane carbon chain for AIMNet2 — 1D along x, ``n_atoms``
    carbons at C-C bond length. Mirrors the production test
    (``test_distributed_models.py::_build_octane_chain``).

    fp32 to match AIMNet2's cuda Warp-kernel constraint (see
    ``_make_aimnet2_wrapper``).
    """
    from nvalchemi.data import AtomicData, Batch

    dtype = torch.float32
    positions = torch.stack(
        [
            0.25 + torch.arange(n_atoms, dtype=dtype) * bond,
            torch.zeros(n_atoms, dtype=dtype),
            torch.zeros(n_atoms, dtype=dtype),
        ],
        dim=1,
    ).contiguous()
    atomic_numbers = torch.full((n_atoms,), 6, dtype=torch.long)
    # The partitioner's domain box is the cell, and it splits along the axis with
    # the most cells (floor(dim / cutoff) per axis). Size x to the chain extent
    # (n_atoms * bond can far exceed a fixed 100 Å box — 1024 atoms => ~1536 Å —
    # which would spill outside the box and degenerate to one rank owning 0
    # atoms) and keep the off-axis dims below the cutoff (one cell each) so the
    # 2-rank split is forced onto x and every rank owns atoms.
    x_extent = 0.5 + n_atoms * bond
    cell = torch.diag(torch.tensor([x_extent, 3.0, 3.0], dtype=dtype))
    pbc = torch.zeros(3, dtype=torch.bool)
    data = AtomicData(
        positions=positions.cuda(),
        atomic_numbers=atomic_numbers.cuda(),
        cell=cell.unsqueeze(0).cuda(),
        pbc=pbc.unsqueeze(0).cuda(),
    )
    return Batch.from_data_list([data], device="cuda")


def _make_methane_packing_for_aimnet2(n_per_side: int = 4, spacing: float = 4.4):
    """3-D methane packing with full PBC for AIMNet2 — exercises the
    PBC + ``calc_masks`` / ``mol_sum`` code path that the 1-D carbon chain
    (no PBC) doesn't.

    n_per_side**3 methane molecules on a cubic lattice (5 atoms each:
    1 C + 4 H), full periodic boundary conditions. Default 4×4×4 = 64
    molecules = 320 atoms; small enough to keep the validator quick
    yet realistic enough to drive ``calc_masks`` / ``mol_sum`` under
    PBC.
    """
    from nvalchemi.data import AtomicData, Batch

    dtype = torch.float32
    n_per_side = int(n_per_side)
    box = float(n_per_side) * spacing

    # Methane geometry: C at origin, four H at the canonical tetrahedral
    # vertices scaled to the C–H bond length (~1.087 Å for CH4).
    bond = 1.087
    s = bond / (3.0**0.5)
    methane_offsets = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [s, s, s],
            [-s, -s, s],
            [-s, s, -s],
            [s, -s, -s],
        ],
        dtype=dtype,
    )

    grid = torch.arange(n_per_side, dtype=dtype)
    centres = (
        torch.stack(torch.meshgrid(grid, grid, grid, indexing="ij"), dim=-1).reshape(
            -1, 3
        )
        * spacing
    )  # (M, 3)

    positions = (centres.unsqueeze(1) + methane_offsets.unsqueeze(0)).reshape(-1, 3)
    n_atoms = positions.shape[0]
    atomic_numbers = torch.tensor([6, 1, 1, 1, 1] * (n_atoms // 5), dtype=torch.long)

    cell = torch.eye(3, dtype=dtype) * box
    pbc = torch.ones(3, dtype=torch.bool)
    data = AtomicData(
        positions=positions.cuda(),
        atomic_numbers=atomic_numbers.cuda(),
        cell=cell.unsqueeze(0).cuda(),
        pbc=pbc.unsqueeze(0).cuda(),
    )
    return Batch.from_data_list([data], device="cuda")


@cuda_required
def test_aimnet2_passes_initial_inference():
    """AIMNet2 uses halo storage. The wrapper's ``distributed_setup``
    installs runtime ``mol_sum`` / ``calc_masks`` helpers on
    ``aimnet.nbops``; the validator should pass on first inference
    without auto-fix engaging."""
    pytest.importorskip("aimnet")
    from nvalchemi.distributed.validate import trace_and_validate

    # 1024 atoms — large enough that the partition genuinely exercises
    # cross-rank index_select traffic at scale rather than the trivial
    # 4-atoms-per-rank case. Energy is extensive (~10⁶ eV at this size)
    # so absolute energy diff scales linearly with N while relative
    # stays at fp32 round-off; the validator's pass criterion is
    # ``abs <= atol OR rel <= rtol`` (mirroring
    # :func:`torch.testing.assert_close`) so this test stays meaningful
    # across system sizes.
    report = trace_and_validate(
        _make_aimnet2_wrapper,
        _make_octane_chain_for_aimnet2(n_atoms=1024),
        world_size=2,
        device="cuda:0",
        atol=1e-5,
        rtol=1e-4,  # fp32-on-cuda baseline; AIMNet2 measured at ~1e-5 here.
        auto_fix=True,
    )
    assert report.ok, report.next_action
    assert len(report.attempts) == 1, (
        f"AIMNet2 should pass on first attempt; got {len(report.attempts)}. "
        f"Last rationale: {report.attempts[-1].rationale}"
    )


@cuda_required
def test_aimnet2_methane_pbc_passes():
    """3-D methane packing with full PBC. Exercises the multi-rank halo
    force path under PBC that the (non-PBC) carbon-chain sample doesn't —
    owned forces must match the single-process reference."""
    pytest.importorskip("aimnet")
    from nvalchemi.distributed.validate import trace_and_validate

    report = trace_and_validate(
        _make_aimnet2_wrapper,
        _make_methane_packing_for_aimnet2(n_per_side=4),  # 320 atoms
        world_size=2,
        device="cuda:0",
        atol=1e-5,
        rtol=1e-4,
        auto_fix=True,
    )
    assert report.ok, report.next_action


# ----------------------------------------------------------------------
# UMA — halo storage with 5 Triton custom_ops.
# ----------------------------------------------------------------------


_UMA_CKPT = "uma-s-1p1"
_UMA_TASK = "omat"


def _make_uma_wrapper():
    from nvalchemi.models.uma import UMAWrapper

    # ``device=torch.device("cuda")`` (not the string "cuda:0") because
    # fairchem's ``_setup_device`` asserts on ``device.type``.
    return UMAWrapper.from_checkpoint(
        _UMA_CKPT, task_name=_UMA_TASK, device=torch.device("cuda")
    )


def _make_bcc_fe_batch(n_per_side: int = 4, jitter: float = 0.05, seed: int = 0):
    """bcc Fe ``n_per_side``³×2 supercell with thermal jitter.

    Default: 4×4×4 → **128 atoms**, ~0.05 Å random displacement.
    A 16-atom (n_per_side=2) cell is force-zero by symmetry (perfect
    crystal at lattice minimum), which makes the validator's diff metric
    meaningless on forces *and* gives every rank full halo coverage of a
    globally-symmetric edge graph (so partition geometry doesn't matter).
    128 atoms with jitter:
      - real per-atom forces (~1e-1 to 1e0 magnitude) → diff metric
        is substantive,
      - partial halo coverage (some edges genuinely cross-rank, not
        every-rank-sees-everything) → exercises the real halo path,
      - large enough that fp32 reduction-order drift across ranks is
        observable → catches collective-precision bugs.
    """
    from ase.build import bulk

    from nvalchemi.data import AtomicData, Batch

    dtype = torch.float32
    atoms = bulk("Fe", "bcc", a=2.87, cubic=True) * (n_per_side, n_per_side, n_per_side)
    positions = torch.as_tensor(atoms.positions, dtype=dtype)
    g = torch.Generator().manual_seed(seed)
    positions = positions + jitter * torch.randn(
        positions.shape, generator=g, dtype=dtype
    )
    atomic_numbers = torch.as_tensor(atoms.get_atomic_numbers(), dtype=torch.long)
    cell = torch.as_tensor(atoms.cell.array, dtype=dtype)
    pbc = torch.ones(3, dtype=torch.bool)
    data = AtomicData(
        positions=positions.cuda(),
        atomic_numbers=atomic_numbers.cuda(),
        cell=cell.unsqueeze(0).cuda(),
        pbc=pbc.unsqueeze(0).cuda(),
    )
    return Batch.from_data_list([data], device="cuda")


@cuda_required
def test_uma_passes_initial_inference():
    """UMA halo-storage correctness on a 128-atom bcc Fe cell: owned
    forces and total energy match the single-process reference."""
    pytest.importorskip("fairchem")
    from nvalchemi.distributed.validate import trace_and_validate

    report = trace_and_validate(
        _make_uma_wrapper,
        _make_bcc_fe_batch(),
        world_size=2,
        device="cuda:0",
        atol=1e-5,
        rtol=1e-4,
        auto_fix=False,
        watched_helper_packages=(
            "aimnet.nbops",
            "fairchem.core.models.uma.outputs",
            "fairchem.core.models.uma.escn_md",
            "fairchem.core.models.uma.common",
        ),
    )
    print(report)
    assert report.ok, report.next_action


@cuda_required
def test_uma_passes_at_n_1024():
    """UMA correctness at n=1024 (BCC 8³ Fe): energy and owned forces
    match the single-process reference at fp32 noise.

    Diagnostics are gated off here because attaching 218 layer hooks
    at n=1024 plus helper-tracing every fairchem function call across
    5 forward/backward runs blows past any reasonable timeout. The
    plain forward+consolidation finishes in normal time and is the
    correctness contract; diagnostic-augmented runs are an n<=128
    feature.
    """
    pytest.importorskip("fairchem")
    from nvalchemi.distributed.validate import trace_and_validate

    report = trace_and_validate(
        _make_uma_wrapper,
        _make_bcc_fe_batch(n_per_side=8),
        world_size=2,
        device="cuda:0",
        atol=1e-5,
        rtol=1e-4,
        auto_fix=False,
        layer_diagnostic=False,
        watched_helper_packages=(),
        timeout_sec=1800.0,
    )
    print(report)
    assert report.ok, report.next_action
    assert report.ok, report.next_action
