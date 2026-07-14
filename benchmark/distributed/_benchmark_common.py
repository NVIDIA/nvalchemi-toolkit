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

"""Shared utilities for the distributed benchmarks.

The two runners — ``benchmark_dd_model_forward.py`` (per-forward timing
+ single-vs-multi force equivalence) and ``benchmark_dd_nvt.py``
(end-to-end NVT timing) — select a model with a ``--config <model>.yaml``
file and drive it through the shared harness here. Keeping the shared
code in one module means:

* Timing semantics are identical across models (same warmup /
  synchronize / averaging rules).
* Force / energy equivalence checks use the same gather-and-compare
  path — no per-model divergence in what "equivalent" means.
* The CLI flag surface stays consistent (``--sizes``, ``--iters``,
  ``--single-only``, ``--profile``, ``--tolerance``, ...).

Each model's geometry, loader, and distribution knobs live in a YAML
config (see ``configs/``); :func:`load_config` parses one into a
:class:`BenchConfig`, :func:`build_system` realises its test system, and
:func:`build_loader` constructs the model wrapper.

What is timed: the model forward + autograd + halo communication; the
neighbour list is pre-built outside the timed window.
"""

from __future__ import annotations

import argparse
import gc
import importlib
import math
import os
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import torch
import torch.distributed as dist
import yaml

# Physical constants re-used by system builders.
KB_EV = 8.617333e-5
AR_LJ_SIGMA = 3.40
AR_LJ_EPSILON = 0.0104
AR_MASS = 39.948
R_MIN_AR = 2 ** (1.0 / 6.0) * AR_LJ_SIGMA  # ~3.816 Å


# ======================================================================
# Timing
# ======================================================================


@dataclass
class Timing:
    """Per-iteration mean wall time, a sanity energy, and peak GPU memory.

    ``peak_mem_mb`` is the CUDA peak allocated memory observed during
    the timed window (warmup + timed iterations), measured on the
    ``device`` passed to :func:`time_step`. It is ``float('nan')`` on
    CPU runs where the peak stat isn't meaningful.

    The ``min_ms`` / ``p50_ms`` / ``p99_ms`` / ``max_ms`` fields are
    NaN by default and only populated by :func:`time_run`, which uses
    per-step CUDA events and can therefore report a distribution. The
    single-forward path :func:`time_step` doesn't bother — its
    iteration count is small and its purpose is amortized perf.
    """

    step_ms: float = 0.0
    final_energy_eV: float = float("nan")
    peak_mem_mb: float = float("nan")
    min_ms: float = float("nan")
    p50_ms: float = float("nan")
    p99_ms: float = float("nan")
    max_ms: float = float("nan")


# ======================================================================
# Memory tracking + OOM handling
# ======================================================================


def _reset_peak_mem(device: torch.device) -> None:
    """Reset CUDA peak-allocated stat before a timed run."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)


def _peak_mem_mb(device: torch.device) -> float:
    """Read peak-allocated in MiB on ``device``; NaN on CPU."""
    if device.type != "cuda":
        return float("nan")
    torch.cuda.synchronize(device)
    return float(torch.cuda.max_memory_allocated(device)) / (1024.0 * 1024.0)


def _is_oom(e: BaseException) -> bool:
    """Recognize a CUDA OOM from any of the wording variants we've seen
    in practice:

    * ``torch.cuda.OutOfMemoryError`` (typed — modern PyTorch).
    * ``RuntimeError: CUDA out of memory. ...`` (older PyTorch).
    * ``RuntimeError: CUDA error: out of memory``.
    * ``RuntimeError: Failed to allocate X bytes on device 'cuda:N'``
      (Warp's allocator — nvalchemiops kernels raise this when the
      device-side allocation fails, e.g. the cos/sin scratch in the
      batched Ewald stage-2 kernel).
    """
    oom_cls = getattr(torch.cuda, "OutOfMemoryError", None)
    if oom_cls is not None and isinstance(e, oom_cls):
        return True
    if isinstance(e, RuntimeError):
        msg = str(e).lower()
        oom_markers = (
            "out of memory",
            "cuda error: out of memory",
            "failed to allocate",  # warp allocator
            "cudaerrormemoryallocation",
        )
        if any(marker in msg for marker in oom_markers):
            return True
    return False


def _recover_from_oom(device: torch.device) -> None:
    """Free as much GPU memory as possible after an OOM and reset peak stats."""
    gc.collect()
    if device.type == "cuda":
        try:
            torch.cuda.synchronize(device)
        except Exception:  # noqa: S110 — post-OOM sync may itself fail
            pass
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)


def _broadcast_failure(failed: bool, device: torch.device, group: Any = None) -> bool:
    """All-reduce (MAX) a boolean failure flag across ranks so every rank
    stays in sync and either all skip or all proceed for a given size.

    Single-rank mode is a no-op: the passed ``failed`` is returned.
    """
    if not dist.is_initialized():
        return failed
    flag = torch.tensor([1 if failed else 0], device=device, dtype=torch.int32)
    dist.all_reduce(flag, op=dist.ReduceOp.MAX, group=group)
    return bool(flag.item())


@dataclass
class EquivalenceReport:
    """Energy + force comparison single-rank reference vs. multi-rank run.

    ``energy_abs_diff`` — ``|E_multi - E_single|`` (eV).
    ``force_max_abs_diff`` / ``force_mean_abs_diff`` — max and mean of
    ``|F_multi - F_single|`` (eV/Å) over all atoms. ``force_rms_diff``
    is the RMS of the per-atom difference norm — a single summary
    number convenient for pass/fail gates. All four default to NaN
    when not computed (single-only runs).
    """

    energy_abs_diff: float = float("nan")
    force_max_abs_diff: float = float("nan")
    force_mean_abs_diff: float = float("nan")
    force_rms_diff: float = float("nan")
    stress_max_abs_diff: float = float("nan")
    stress_rms_diff: float = float("nan")
    n_atoms: int = 0
    tolerance: float = 1e-4
    passed: bool = True

    def fmt_row(self) -> str:
        """Format a row of the equivalence report."""
        status = "OK  " if self.passed else "FAIL"
        line = (
            f"  [{status}]  n={self.n_atoms:>6}  "
            f"ΔE={self.energy_abs_diff:+.3e} eV  "
            f"|ΔF|_max={self.force_max_abs_diff:.3e} "
            f"|ΔF|_mean={self.force_mean_abs_diff:.3e} "
            f"|ΔF|_rms={self.force_rms_diff:.3e} eV/Å  "
            f"(tol={self.tolerance:.0e})"
        )
        # Stress is optional — only some models compute it (UMA does;
        # LJ / pure-pair don't). NaN means "not measured", omit silently.
        import math  # noqa: PLC0415

        if not math.isnan(self.stress_max_abs_diff):
            line += (
                f"  |Δσ|_max={self.stress_max_abs_diff:.3e} "
                f"|Δσ|_rms={self.stress_rms_diff:.3e}"
            )
        return line


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def time_step(
    fn: Callable[[], tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
    n_iters: int = 20,
    n_warmup: int = 3,
) -> Timing:
    """Time ``fn`` (returns ``(energy_scalar, forces)``) — mean wall ms.

    Also records CUDA peak allocated memory (in MiB) observed over the **timed**
    iterations only. Peak is reset at entry AND again after warmup, so the one-
    time ``torch.compile`` / inductor-autotuning transient (which can be far
    larger than steady state — it clones mutated kernel args while benchmarking)
    is excluded. The reported peak is the per-step memory the model actually uses
    in a loop (MD), not the compile spike.
    """
    _reset_peak_mem(device)

    for _ in range(n_warmup):
        out = fn()
        del out
    _sync(device)
    # Drop the compile/autotuning transient from the peak — measure steady state.
    _reset_peak_mem(device)

    total = 0.0
    last_energy = float("nan")
    for _ in range(n_iters):
        _sync(device)
        t0 = time.perf_counter()
        out = fn()
        _sync(device)
        t1 = time.perf_counter()
        total += t1 - t0
        # Step fns return (energy, forces) or (energy, forces, stress);
        # we only need the scalar energy here.
        last_energy = float(out[0].detach().item())
        del out

    return Timing(
        step_ms=(total / n_iters) * 1e3,
        final_energy_eV=last_energy,
        peak_mem_mb=_peak_mem_mb(device),
    )


def profile_step(
    fn: Callable[[], tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
    trace_path: Path,
    n_steps: int,
    n_warmup: int = 2,
) -> None:
    """Warmup + torch.profiler trace export for ``fn``."""
    from torch.profiler import ProfilerActivity
    from torch.profiler import profile as torch_profile

    for _ in range(n_warmup):
        out = fn()
        del out
    _sync(device)

    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)

    trace_path.parent.mkdir(parents=True, exist_ok=True)
    with torch_profile(
        activities=activities, with_stack=True, record_shapes=True
    ) as prof:
        for _ in range(n_steps):
            out = fn()
            del out
        _sync(device)
    prof.export_chrome_trace(str(trace_path))


def profile_trace_path(
    profile_dir: Path, model: str, n_atoms: int, rank: int, world_size: int
) -> Path:
    """Consistent per-(model, size, rank) Chrome-trace path."""
    tag = "single" if world_size == 1 else f"rank{rank}of{world_size}"
    return profile_dir / f"{model}-n{n_atoms}-{tag}.json"


# ======================================================================
# Distributed init
# ======================================================================


def init_distributed(device_name: str) -> tuple[int, int, Any]:
    """``DistributedManager`` + ``DeviceMesh`` from torchrun env."""
    from physicsnemo.distributed import DistributedManager
    from torch.distributed import DeviceMesh

    DistributedManager.initialize()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    mesh = DeviceMesh(device_name, list(range(world_size)), mesh_dim_names=("domain",))
    return rank, world_size, mesh


def launched_by_torchrun() -> bool:
    """Check if the script is launched by torchrun."""
    return "RANK" in os.environ and "WORLD_SIZE" in os.environ


# ======================================================================
# System builders
# ======================================================================


def build_argon_cluster(
    n_per_side: int, dtype: torch.dtype, seed: int = 0
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Argon cluster on a cubic lattice at LJ-equilibrium spacing.

    Returns ``(positions, atomic_numbers, masses, cell, velocities)``.
    Non-PBC — cell is just a containment box. Deterministic via
    ``torch.manual_seed(seed)`` on the positional jitter + velocity
    Maxwell-Boltzmann sample.
    """
    n = n_per_side**3
    spacing = R_MIN_AR * 1.05
    coords = torch.arange(n_per_side, dtype=dtype) * spacing
    gx, gy, gz = torch.meshgrid(coords, coords, coords, indexing="ij")
    positions = torch.stack([gx.flatten(), gy.flatten(), gz.flatten()], dim=-1)
    torch.manual_seed(seed)
    positions = positions + 0.05 * torch.randn_like(positions)

    atomic_numbers = torch.full((n,), 18, dtype=torch.long)
    masses = torch.full((n,), AR_MASS, dtype=dtype)

    box_side = n_per_side * spacing + 20.0
    cell = torch.eye(3, dtype=dtype) * box_side

    v_std = math.sqrt(KB_EV * 300.0 / AR_MASS)
    velocities = v_std * torch.randn(n, 3, dtype=dtype)
    velocities = velocities - velocities.mean(dim=0, keepdim=True)
    return positions, atomic_numbers, masses, cell, velocities


def build_carbon_chain(
    n_atoms: int, dtype: torch.dtype, seed: int = 0
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pseudo-polymer: N carbons in a straight line, non-PBC. Used by
    AIMNet2 because its element set is molecular."""
    positions = torch.stack(
        [
            0.25 + torch.arange(n_atoms, dtype=dtype) * 1.5,
            torch.zeros(n_atoms, dtype=dtype),
            torch.zeros(n_atoms, dtype=dtype),
        ],
        dim=1,
    ).contiguous()
    torch.manual_seed(seed)
    positions = positions + 0.01 * torch.randn_like(positions)
    atomic_numbers = torch.full((n_atoms,), 6, dtype=torch.long)
    masses = torch.full((n_atoms,), 12.011, dtype=dtype)
    cell = torch.eye(3, dtype=dtype) * 100.0
    velocities = torch.zeros_like(positions)
    return positions, atomic_numbers, masses, cell, velocities


def build_methane_packing(
    n_atoms: int, dtype: torch.dtype, seed: int = 0
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """3-D packing of methane (CH4) molecules, full PBC. Used by AIMNet2.

    Each CH4 unit contributes 5 atoms (1 C + 4 H) — actual atom count is
    rounded down to ``5 * n_per_side**3`` where ``n_per_side`` is chosen
    so the total is closest to (but not exceeding) ``n_atoms``. C-C
    spacing is 4.4 Å (typical liquid-methane density); within each
    molecule the C-H bond is 1.09 Å in a tetrahedral geometry.

    Returns ``(positions, atomic_numbers, masses, cell, velocities)``.
    """
    n_molecules_target = max(1, n_atoms // 5)
    n_per_side = max(1, round(n_molecules_target ** (1.0 / 3.0)))
    n_molecules = n_per_side**3
    n_total = n_molecules * 5

    spacing = 4.4
    box_side = n_per_side * spacing

    # Tetrahedral H positions around a central C, normalised to bond length 1.09 Å.
    h_dirs = (
        torch.tensor(
            [
                [1.0, 1.0, 1.0],
                [1.0, -1.0, -1.0],
                [-1.0, 1.0, -1.0],
                [-1.0, -1.0, 1.0],
            ],
            dtype=dtype,
        )
        / math.sqrt(3.0)
        * 1.09
    )

    coords = torch.arange(n_per_side, dtype=dtype) * spacing + spacing / 2.0
    cx, cy, cz = torch.meshgrid(coords, coords, coords, indexing="ij")
    c_positions = torch.stack([cx.flatten(), cy.flatten(), cz.flatten()], dim=-1)

    torch.manual_seed(seed)
    # Per-molecule random rotation (tiny — keeps tetrahedral shape, just
    # avoids a perfectly aligned lattice).
    jitter = 0.05 * torch.randn_like(c_positions)
    c_positions = (c_positions + jitter) % box_side

    positions_list = []
    atomic_numbers_list = []
    masses_list = []
    for i in range(n_molecules):
        c_pos = c_positions[i]
        positions_list.append(c_pos.unsqueeze(0))
        atomic_numbers_list.append(torch.tensor([6], dtype=torch.long))
        masses_list.append(torch.tensor([12.011], dtype=dtype))
        for h_dir in h_dirs:
            positions_list.append((c_pos + h_dir).unsqueeze(0))
            atomic_numbers_list.append(torch.tensor([1], dtype=torch.long))
            masses_list.append(torch.tensor([1.008], dtype=dtype))

    positions = torch.cat(positions_list, dim=0) % box_side
    atomic_numbers = torch.cat(atomic_numbers_list, dim=0)
    masses = torch.cat(masses_list, dim=0)
    cell = torch.eye(3, dtype=dtype) * box_side
    kT = KB_EV * 300.0
    v_std = torch.sqrt(kT / masses).unsqueeze(-1)
    velocities = v_std * torch.randn(n_total, 3, dtype=dtype)
    velocities = velocities - velocities.mean(dim=0, keepdim=True)
    return positions, atomic_numbers, masses, cell, velocities


def methane_n_per_side_for_size(n_atoms: int) -> int:
    """Pick ``n_per_side`` so 5 × n_per_side³ ≤ n_atoms (rounding to the
    nearest plausible integer); used by the AIMNet2 NVT harness."""
    n_molecules_target = max(1, n_atoms // 5)
    return max(1, round(n_molecules_target ** (1.0 / 3.0)))


def build_sio2_supercell(
    repeats: tuple[int, int, int], dtype: torch.dtype, seed: int = 0
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Alpha-quartz SiO2 supercell (9 atoms/cell × product(repeats)).

    Full-PBC, uses ASE's ``crystal`` with spacegroup 152. Used by the
    MACE benchmark.
    """
    from ase.spacegroup import crystal

    unit_cell = crystal(
        symbols=["O", "Si"],
        basis=[[0.413, 0.2711, 0.2172], [0.4673, 0.0, 0.3333]],
        spacegroup=152,
        cellpar=[4.9019, 4.9019, 5.3988, 90, 90, 120],
    )
    atoms = unit_cell.repeat(repeats)
    positions = torch.tensor(atoms.get_positions(), dtype=dtype)
    atomic_numbers = torch.tensor(atoms.get_atomic_numbers(), dtype=torch.long)
    masses = torch.tensor(atoms.get_masses(), dtype=dtype)
    cell = torch.tensor(atoms.get_cell().array, dtype=dtype)

    torch.manual_seed(seed)
    kT = KB_EV * 300.0
    v_std = torch.sqrt(kT / masses).unsqueeze(-1)
    velocities = v_std * torch.randn(len(atoms), 3, dtype=dtype)
    velocities = velocities - velocities.mean(dim=0, keepdim=True)
    return positions, atomic_numbers, masses, cell, velocities


def build_nacl(
    n_per_side: int, dtype: torch.dtype, seed: int = 0
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    """Simple-cubic NaCl-like lattice: alternating ±1 charges on Na/Cl.

    Fully periodic; used by the Ewald and PME benchmarks. Returns the
    usual 5-tuple PLUS a ``charges`` tensor since electrostatics
    models require it.
    """
    box = n_per_side * 2.82  # typical Na-Cl nearest-neighbour distance
    coords = torch.arange(n_per_side, dtype=dtype) * (box / n_per_side)
    gx, gy, gz = torch.meshgrid(coords, coords, coords, indexing="ij")
    positions = torch.stack([gx.flatten(), gy.flatten(), gz.flatten()], dim=-1)
    n = positions.shape[0]

    torch.manual_seed(seed)
    positions = positions + 0.05 * torch.randn_like(positions)
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
    v_std = torch.sqrt(KB_EV * 300.0 / masses).unsqueeze(-1)
    velocities = v_std * torch.randn(n, 3, dtype=dtype)
    velocities = velocities - velocities.mean(dim=0, keepdim=True)
    return positions, atomic_numbers, masses, charges, cell, velocities


def build_bcc_fe(
    n_cells_per_side: int, dtype: torch.dtype, seed: int = 0
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """bcc iron supercell (2 atoms/cell × n³ cells). Used by UMA."""
    from ase.build import bulk

    atoms = bulk("Fe", "bcc", a=2.87, cubic=True) * (
        n_cells_per_side,
        n_cells_per_side,
        n_cells_per_side,
    )
    positions = torch.tensor(atoms.get_positions(), dtype=dtype)
    atomic_numbers = torch.tensor(atoms.get_atomic_numbers(), dtype=torch.long)
    masses = torch.tensor(atoms.get_masses(), dtype=dtype)
    cell = torch.tensor(atoms.get_cell().array, dtype=dtype)

    # Rattle off the perfect lattice so forces are non-trivial: a symmetric
    # crystal has ~zero forces by symmetry, which would make the single- vs
    # multi-rank force-equivalence check pass vacuously (it cannot see a DD
    # error that respects the lattice symmetry). Seeded for reproducibility.
    torch.manual_seed(seed + 12345)
    positions = positions + 0.05 * torch.randn_like(positions)

    torch.manual_seed(seed)
    kT = KB_EV * 300.0
    v_std = torch.sqrt(kT / masses).unsqueeze(-1)
    velocities = v_std * torch.randn(len(atoms), 3, dtype=dtype)
    velocities = velocities - velocities.mean(dim=0, keepdim=True)
    return positions, atomic_numbers, masses, cell, velocities


def build_bcc_fe_elongated(
    nz_cells: int, dtype: torch.dtype, seed: int = 0
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Elongated bcc-Fe supercell: fixed 6x6 cross-section x ``nz_cells`` along z
    (72 atoms per z-cell), ``a=2.866``. The long z axis keeps a 1-D domain split
    along z non-degenerate as the atom count grows (used for the LAMMPS<->DD
    scaling comparison)."""
    from ase.build import bulk

    nxy = 6
    atoms = bulk("Fe", "bcc", a=2.866, cubic=True) * (nxy, nxy, nz_cells)
    positions = torch.tensor(atoms.get_positions(), dtype=dtype)
    atomic_numbers = torch.tensor(atoms.get_atomic_numbers(), dtype=torch.long)
    masses = torch.tensor(atoms.get_masses(), dtype=dtype)
    cell = torch.tensor(atoms.get_cell().array, dtype=dtype)

    torch.manual_seed(seed + 12345)
    positions = positions + 0.05 * torch.randn_like(positions)
    torch.manual_seed(seed)
    kT = KB_EV * 300.0
    v_std = torch.sqrt(kT / masses).unsqueeze(-1)
    velocities = v_std * torch.randn(len(atoms), 3, dtype=dtype)
    velocities = velocities - velocities.mean(dim=0, keepdim=True)
    return positions, atomic_numbers, masses, cell, velocities


def bcc_fe_elong_cells_for_size(n_atoms: int) -> int:
    """z-cell count for a 6x6xNz elongated bcc-Fe cell (72 atoms per z-cell)."""
    return max(2, round(n_atoms / 72.0))


def n_per_side_for_size(n_atoms: int) -> int:
    """Calculate the number of atoms per side for a given number of atoms."""
    return max(2, round(n_atoms ** (1.0 / 3.0)))


def sio2_repeats_for_size(n_atoms: int) -> tuple[int, int, int]:
    """Calculate the number of repeats for a given number of atoms."""
    r = max(1, round((n_atoms / 9.0) ** (1.0 / 3.0)))
    return (r, r, r)


def bcc_fe_cells_for_size(n_atoms: int) -> int:
    """Calculate the number of cells per side for a given number of atoms."""
    return max(1, round((n_atoms / 2.0) ** (1.0 / 3.0)))


# ======================================================================
# Force equivalence — gather per-rank owned forces to rank 0 and compare
# ======================================================================


def gather_owned_forces_to_rank0(
    forces_owned: torch.Tensor,
    rank_assignment: torch.Tensor,
    rank: int,
    world_size: int,
    group: Any = None,
) -> torch.Tensor | None:
    """Reconstruct a global ``(n_global, 3)`` force tensor on rank 0.

    Each rank holds ``forces_owned`` of shape ``(n_owned_rank, 3)`` —
    the forces on the atoms it owns. ``rank_assignment`` is the global
    ``(n_global,)`` int tensor mapping each global atom index to its
    owning rank; every rank holds the same copy. We:

      1. All-gather each rank's owned forces into a flat list.
      2. Scatter them back into their original positions using
         ``rank_assignment`` (stable-sort preserves within-rank order).

    Returns the reconstructed ``(n_global, 3)`` tensor on rank 0, and
    ``None`` on other ranks (they don't need it for the comparison).
    """
    device = forces_owned.device
    dtype = forces_owned.dtype
    n_global = rank_assignment.shape[0]

    # 1. All-gather the per-rank owned counts so we can size receive buffers.
    n_owned_this = torch.tensor(
        [forces_owned.shape[0]], device=device, dtype=torch.int64
    )
    owned_counts = [
        torch.zeros(1, device=device, dtype=torch.int64) for _ in range(world_size)
    ]
    dist.all_gather(owned_counts, n_owned_this, group=group)
    owned_counts = [int(c.item()) for c in owned_counts]

    # 2. Pad local forces to the max per-rank count so all_gather can
    # use a uniform receive shape, then trim on rank 0.
    max_owned = max(owned_counts)
    padded = torch.zeros(max_owned, 3, device=device, dtype=dtype)
    padded[: forces_owned.shape[0]] = forces_owned
    all_padded = [
        torch.zeros(max_owned, 3, device=device, dtype=dtype) for _ in range(world_size)
    ]
    dist.all_gather(all_padded, padded, group=group)

    if rank != 0:
        return None

    # 3. Reassemble by scanning rank_assignment in atom order — ranks
    # receive their atoms in global-index order (ShardedBatch uses a
    # stable sort), so within-rank positions line up with the order
    # atoms appear in rank_assignment.
    within_rank_pos = torch.zeros(world_size, dtype=torch.int64)
    full = torch.zeros(n_global, 3, device=device, dtype=dtype)
    for g in range(n_global):
        r = int(rank_assignment[g].item())
        local_i = int(within_rank_pos[r].item())
        if local_i < owned_counts[r]:
            full[g] = all_padded[r][local_i]
        within_rank_pos[r] += 1
    return full


def compare_forces(
    forces_single: torch.Tensor,
    forces_multi_full: torch.Tensor,
    energy_single: float,
    energy_multi: float,
    tolerance: float = 1e-4,
    stress_single: torch.Tensor | None = None,
    stress_multi: torch.Tensor | None = None,
) -> EquivalenceReport:
    """Build an :class:`EquivalenceReport` comparing two force fields.

    Force inputs must be ``(n_global, 3)`` on the same device/dtype.
    Reports per-component max / mean / RMS for forces; pass-fail is
    ``|ΔF|_max < tolerance``.

    Optional stress inputs report ``|Δσ|_max`` and ``|Δσ|_rms`` (in the
    stress tensor's native units, typically eV/Å³). Stress diff does
    not gate pass-fail — it's diagnostic-only.
    """
    import math  # noqa: PLC0415

    assert forces_single.shape == forces_multi_full.shape, (
        f"shape mismatch: single {forces_single.shape} vs "
        f"multi {forces_multi_full.shape}"
    )
    diff = (forces_multi_full - forces_single).detach()
    max_abs = float(diff.abs().max().item())
    mean_abs = float(diff.abs().mean().item())
    rms = float((diff.pow(2).sum(dim=-1).sqrt().pow(2).mean().sqrt()).item())
    energy_abs_diff = abs(energy_multi - energy_single)
    # Energy is EXTENSIVE (scales with atom count), so the per-atom force
    # tolerance is the wrong yardstick: an fp32 total summed in a different
    # order across ranks drifts ~1e-5 relative (~0.08 eV on an ~8879 eV
    # system) while forces stay exact to ~1e-5 eV/Å. Gate energy RELATIVELY
    # (assert_close-style: atol=tolerance, rtol=1e-4) so fp32 routes (cueq) on
    # large systems aren't failed by benign summation-order noise; plain fp64
    # still agrees to ~1e-12, and a genuine halo energy-accounting bug
    # (typically a sizeable fraction of the total) is still caught.
    energy_rtol = 1e-4
    energy_passed = energy_abs_diff <= tolerance + energy_rtol * abs(energy_single)
    passed = max_abs < tolerance and energy_passed

    stress_max = float("nan")
    stress_rms = float("nan")
    if stress_single is not None and stress_multi is not None:
        s_diff = (stress_multi - stress_single).detach().to(torch.float64)
        stress_max = float(s_diff.abs().max().item())
        stress_rms = float(s_diff.pow(2).mean().sqrt().item())
        if math.isfinite(stress_max):
            passed = passed and stress_max < tolerance

    return EquivalenceReport(
        energy_abs_diff=energy_abs_diff,
        force_max_abs_diff=max_abs,
        force_mean_abs_diff=mean_abs,
        force_rms_diff=rms,
        stress_max_abs_diff=stress_max,
        stress_rms_diff=stress_rms,
        n_atoms=forces_single.shape[0],
        tolerance=tolerance,
        passed=passed,
    )


# ======================================================================
# Pretty-printing
# ======================================================================


@dataclass
class SweepResult:
    """One row of the sweep: 1-rank + multi-rank timings + equivalence.

    ``single_failed`` / ``multi_failed`` / ``failure_reason`` record an
    OOM (or any other caught exception) during the respective path.
    When ``single_failed`` is ``True``, ``multi`` is not attempted for
    this size. Failure rows still print, so the user sees the OOM size
    explicitly rather than silently missing data.
    """

    model: str
    n_atoms: int
    single: Timing
    multi: Timing | None = None
    equivalence: EquivalenceReport | None = None
    single_failed: bool = False
    multi_failed: bool = False
    failure_reason: str = ""


def _fmt_mem(mb: float) -> str:
    """MiB formatter: ``'—'`` on NaN (CPU), ``'{:.1f}'`` otherwise."""
    return "—" if math.isnan(mb) else f"{mb:.1f}"


def print_scaling_table(results: list[SweepResult], world_size: int) -> None:
    """Print one row per (model, size) with energies, timings, and peak GPU
    memory. Rows that OOM'd print ``OOM`` in place of the timing/energy."""
    header_multi = f"{world_size}-rank step" if world_size > 1 else "—"
    header_multi_mem = f"{world_size}r peak MB" if world_size > 1 else "—"
    print()
    print(
        f"{'model':<15}{'n_atoms':>8}"
        f"{'1-rank step':>14}{header_multi:>14}{'speedup':>9}"
        f"{'1r peak MB':>13}{header_multi_mem:>13}"
        f"{'1r energy (eV)':>18}{'multi energy (eV)':>20}"
    )
    print("-" * 124)
    for r in results:
        t1 = r.single
        t_multi = r.multi
        # Single-rank columns — independent of multi-rank status.
        if r.single_failed:
            t1_str = "OOM"
            e1_str = "—"
            mem1_str = "—"
        else:
            t1_str = f"{t1.step_ms:>11.3f} ms"
            e1_str = f"{t1.final_energy_eV:>18.6f}"
            mem1_str = _fmt_mem(t1.peak_mem_mb)
        # Multi-rank columns — independent of single-rank status so the
        # sweep still surfaces timings/memory for sizes that only the
        # multi-rank setup can fit.
        if r.multi_failed:
            t_multi_str = "OOM"
            speedup_str = "—"
            e_multi_str = "—"
            mem_multi_str = "—"
        elif t_multi is None or t_multi.step_ms == 0:
            t_multi_str = "—"
            speedup_str = "—"
            e_multi_str = "—"
            mem_multi_str = "—"
        else:
            t_multi_str = f"{t_multi.step_ms:>11.3f} ms"
            speedup_str = (
                f"{t1.step_ms / t_multi.step_ms:>7.2f}×" if not r.single_failed else "—"
            )
            e_multi_str = f"{t_multi.final_energy_eV:>20.6f}"
            mem_multi_str = _fmt_mem(t_multi.peak_mem_mb)
        print(
            f"{r.model:<15}{r.n_atoms:>8}"
            f"{t1_str:>14}{t_multi_str:>14}{speedup_str:>9}"
            f"{mem1_str:>13}{mem_multi_str:>13}"
            f"{e1_str:>18}{e_multi_str:>20}"
        )
        if (r.single_failed or r.multi_failed) and r.failure_reason:
            print(f"{'':<15}  └─ {r.failure_reason}")


def print_equivalence_table(results: list[SweepResult]) -> None:
    """Print one row per (model, size) with energy+force equivalence.

    Skipped silently when no row has an equivalence report (e.g. pure
    single-rank runs).
    """
    rows = [r for r in results if r.equivalence is not None]
    if not rows:
        return
    print()
    print("=== Single-rank vs multi-rank equivalence ===")
    any_fail = False
    for r in rows:
        print(f"  {r.model:<15}  {r.equivalence.fmt_row()}")
        if not r.equivalence.passed:
            any_fail = True
    if any_fail:
        print()
        print(
            "  WARNING: at least one (model, size) exceeded the force "
            "tolerance. Treat the timing numbers skeptically until the "
            "distribution math is reconciled."
        )


# ======================================================================
# CLI scaffolding
# ======================================================================


def add_common_args(parser: argparse.ArgumentParser) -> None:
    """Attach the shared CLI flags used by every ``benchmark_<model>.py``."""
    parser.add_argument(
        "--sizes",
        nargs="+",
        type=int,
        default=None,
        help="Atom counts to sweep (default: the config's ``default_sizes``).",
    )
    parser.add_argument("--iters", type=int, default=20, help="Timed iterations.")
    parser.add_argument("--warmup", type=int, default=3, help="Warmup iterations.")
    parser.add_argument(
        "--device",
        default=None,
        help="'cpu' or 'cuda' (defaults to 'cuda' when available).",
    )
    parser.add_argument(
        "--single-only",
        action="store_true",
        help="Skip the multi-rank run (single-rank baseline only).",
    )
    parser.add_argument(
        "--run-reference",
        action="store_true",
        help=(
            "In multi-rank runs, also compute the single-rank full-system "
            "reference + equivalence check. OFF by default: the multi-rank leg "
            "runs DD-only, so no rank builds the whole system — required for a "
            "scaling sweep (otherwise every rank holds the full system, which "
            "pollutes the multi-rank peak memory and caps max-N at the "
            "single-GPU OOM ceiling). Single-rank (``world_size==1``) runs are "
            "unaffected — they are the reference."
        ),
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-4,
        help="Max allowable |ΔF| (eV/Å) between single- and multi-rank forces.",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help=(
            "After timing each (model, size), run an additional profiled "
            "pass with ``torch.profiler`` and export a Chrome trace."
        ),
    )
    parser.add_argument(
        "--profile-steps",
        type=int,
        default=10,
        help="Number of iterations to profile per (model, size).",
    )
    parser.add_argument(
        "--profile-dir",
        type=str,
        default=None,
        help=(
            "Directory to write Chrome traces into. Defaults to "
            "``benchmark_profiles/<model>``."
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Per-rank debug: n_owned / n_halo / n_padded + raw energies.",
    )


def resolve_device(args: argparse.Namespace) -> torch.device:
    """Resolve the device to use for the benchmark."""
    if args.device is not None:
        return torch.device(args.device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ======================================================================
# Sweep driver — timing + force equivalence for one (model, n_atoms)
# ======================================================================


# Harness builder contract. Each per-model benchmark implements one of
# these and hands it to :func:`run_sweep_one_size`; the signature is
# normalised so the sweep driver stays model-agnostic.
#
# Returns ``(step_fn, n_actual, positions_global, cell_3x3, pbc_3, domain_config)``:
#
# - ``step_fn``: the timed forward; returns ``(energy_scalar, forces)``.
# - ``n_actual``: the realised atom count (may differ slightly from the
#   requested ``n_atoms`` because builders round to full lattices).
# - ``positions_global``: the CPU-side positions used to seed the
#   partitioner on rank 0; also used to recover rank assignment for
#   force gathering.
# - ``cell_3x3``: global cell (3, 3) — feeds the partitioner.
# - ``pbc_3``: global pbc (3,) bool — feeds the partitioner.
# - ``domain_config``: the ``DomainConfig`` used by the distributed
#   path, or ``None`` in single-rank mode.


def run_sweep_one_size(
    model_name: str,
    build_harness: Callable[
        ..., tuple[Any, int, torch.Tensor, torch.Tensor, torch.Tensor, Any]
    ],
    wrapper: Any,
    n_atoms: int,
    device: torch.device,
    dtype: torch.dtype,
    args: argparse.Namespace,
    *,
    rank: int,
    world_size: int,
    mesh: Any,
    profile_dir: Path,
) -> SweepResult:
    """Run the single-rank baseline + (when launched under torchrun) the
    multi-rank run; capture forces from an extra untimed step and
    compute equivalence against the single-rank reference.

    An OOM (or any other exception) in either path is caught — memory is
    reclaimed via ``empty_cache``, the failure is broadcast to all ranks
    so they stay in sync, and a :class:`SweepResult` marked
    ``single_failed`` / ``multi_failed`` is returned so the sweep can
    continue to larger sizes without crashing. Only OOM exceptions are
    recovered; other exceptions are re-raised once ranks have synced.
    """
    # ------------------------------------------------------------------
    # Single-rank baseline.
    # ------------------------------------------------------------------
    from nvalchemi.distributed._core.gather_primitives import mesh_group

    group = mesh_group(mesh) if mesh is not None else None
    placeholder_timing = Timing()
    n_actual = n_atoms

    positions_global: torch.Tensor | None = None
    cell: torch.Tensor | None = None
    pbc: torch.Tensor | None = None
    e_single_val = float("nan")
    f_single_full: torch.Tensor | None = None
    s_single_full: torch.Tensor | None = None
    single_timing = placeholder_timing
    single_failed = False
    single_reason = ""
    multi_reason = ""

    # The single-rank full-system reference runs only when it IS the measurement
    # (``world_size == 1``) or when explicitly requested (``--run-reference``).
    # Otherwise the multi-rank leg is DD-only: no rank builds the whole system,
    # so the multi-rank peak reflects the per-GPU shard and max-N is bounded by
    # the shard, not the single-GPU OOM ceiling. ``run_ref`` is identical on
    # every rank (same args + world_size), so the collectives below stay
    # symmetric.
    run_ref = world_size == 1 or getattr(args, "run_reference", False)
    if run_ref:
        try:
            step_single, n_actual, positions_global, cell, pbc, _ = build_harness(
                wrapper,
                n_atoms,
                device,
                dtype,
                distributed=False,
            )
            trace_single = (
                profile_trace_path(
                    profile_dir, model_name, n_actual, rank=0, world_size=1
                )
                if args.profile and rank == 0
                else None
            )
            single_timing = time_step(step_single, device, args.iters, args.warmup)
            single_out = step_single()
            e_single = single_out[0]
            f_single = single_out[1]
            s_single = single_out[2] if len(single_out) >= 3 else None
            e_single_val = float(e_single.detach().item())
            f_single_full = f_single.detach().clone()
            s_single_full = s_single.detach().clone() if s_single is not None else None
            del e_single, f_single, s_single, single_out
            if trace_single is not None:
                profile_step(step_single, device, trace_single, args.profile_steps)
        except Exception as exc:  # noqa: BLE001 — broad catch is the point
            if not _is_oom(exc):
                # Non-OOM failure: surface it BEFORE the broadcast (which can
                # deadlock if only some ranks failed, against the survivors'
                # in-flight model collectives), then broadcast so ranks sync on
                # the skip, then re-raise so the user sees the full trace.
                traceback.print_exc()
                _broadcast_failure(True, device, group=group)
                raise
            single_failed = True
            single_reason = f"single-rank OOM: {exc}"
            traceback.print_exc()
            _recover_from_oom(device)

        # Cross-rank sync on single-rank status — every rank must agree
        # before we attempt the multi-rank branch, or collectives will hang.
        # We still run the multi-rank leg when the single-rank reference
        # OOM'd: the whole point of domain-parallel inference is that it
        # fits systems the single-GPU path can't, so those are precisely
        # the sizes the user cares most about. Equivalence is skipped
        # downstream (no reference), but timings + peak memory still print.
        single_failed = _broadcast_failure(single_failed, device, group=group)

    if args.single_only or world_size == 1:
        return SweepResult(
            model=model_name,
            n_atoms=n_actual,
            single=single_timing,
            single_failed=single_failed,
            failure_reason=single_reason,
        )

    # ------------------------------------------------------------------
    # Multi-rank — runs even when the single-rank reference OOM'd. The
    # whole point of domain-parallel inference is that it can fit sizes
    # the single-GPU path cannot; those are the rows the user most
    # wants in the table. Equivalence is skipped for that case (no
    # reference), but timings and peak memory still print.
    # ------------------------------------------------------------------
    multi_timing = placeholder_timing
    multi_failed = False
    report: EquivalenceReport | None = None
    f_owned: torch.Tensor | None = None
    s_owned: torch.Tensor | None = None
    domain_config: Any = None
    # We may not have the ``cell`` / ``pbc`` / ``positions_global`` from
    # the single-rank build_harness (if it OOM'd) — the multi-rank
    # build_harness call below returns fresh copies, which we capture
    # here and use for the gather-and-compare step.
    dist_cell = cell
    dist_pbc = pbc
    dist_positions = positions_global

    try:
        step_dist, n_actual_dist, dist_positions, dist_cell, dist_pbc, domain_config = (
            build_harness(
                wrapper,
                n_atoms,
                device,
                dtype,
                distributed=True,
                rank=rank,
                world_size=world_size,
                mesh=mesh,
            )
        )
        # Keep the realised count in sync even when the single-rank path
        # skipped and n_actual is still the requested value.
        n_actual = n_actual_dist
        trace_multi = (
            profile_trace_path(profile_dir, model_name, n_actual, rank, world_size)
            if args.profile
            else None
        )
        multi_timing = time_step(step_dist, device, args.iters, args.warmup)
        multi_out = step_dist()
        f_owned = multi_out[1].detach()
        s_owned = multi_out[2].detach() if len(multi_out) >= 3 else None
        del multi_out
        if trace_multi is not None:
            profile_step(step_dist, device, trace_multi, args.profile_steps)
    except Exception as exc:  # noqa: BLE001
        if not _is_oom(exc):
            # Surface the non-OOM error BEFORE the failure broadcast: if only
            # some ranks failed, the broadcast collides with the survivors'
            # in-flight model collectives and deadlocks, which would otherwise
            # hide the real exception. Printing first keeps it diagnosable.
            traceback.print_exc()
            _broadcast_failure(True, device, group=group)
            raise
        multi_failed = True
        multi_reason = f"multi-rank OOM: {exc}"
        traceback.print_exc()
        _recover_from_oom(device)

    multi_failed = _broadcast_failure(multi_failed, device, group=group)

    # ------------------------------------------------------------------
    # Gather + equivalence. Only meaningful when (a) multi-rank produced
    # forces AND (b) single-rank produced a reference to compare against.
    # Gather itself can OOM on huge systems, so it's guarded too.
    # ------------------------------------------------------------------
    can_gather = (
        not multi_failed
        and f_owned is not None
        and domain_config is not None
        and dist_cell is not None
        and dist_pbc is not None
        and dist_positions is not None
    )
    if run_ref and can_gather and not single_failed:
        try:
            from nvalchemi.distributed.partitioner import SpatialPartitioner

            partitioner = SpatialPartitioner(
                config=domain_config,
                cell_matrix=dist_cell.to(device=device, dtype=dtype).unsqueeze(0),
                pbc=dist_pbc.to(device=device).reshape(1, 3),
            )
            rank_assignment = partitioner.assign_atoms_to_ranks(
                dist_positions.to(device=device, dtype=dtype)
            )
            full_multi = gather_owned_forces_to_rank0(
                f_owned,
                rank_assignment,
                rank,
                world_size,
                group=group,
            )
            if rank == 0 and full_multi is not None and f_single_full is not None:
                # Stress (when both halves produced it) is replicated
                # global on every rank under the Replicated policy, so
                # rank 0's local copy is the reference. No
                # gather-and-reassemble needed for stress like forces.
                report = compare_forces(
                    f_single_full.to(device=device, dtype=dtype),
                    full_multi,
                    energy_single=e_single_val,
                    energy_multi=multi_timing.final_energy_eV,
                    tolerance=args.tolerance,
                    stress_single=(
                        s_single_full.to(device=device, dtype=dtype)
                        if s_single_full is not None
                        else None
                    ),
                    stress_multi=s_owned if s_owned is not None else None,
                )
        except Exception as exc:  # noqa: BLE001
            if not _is_oom(exc):
                # Surface the non-OOM error before the failure broadcast so a
                # partial-rank failure is never hidden by the ensuing deadlock.
                traceback.print_exc()
                _broadcast_failure(True, device, group=group)
                raise
            multi_failed = True
            multi_reason = f"force-gather OOM: {exc}"
            traceback.print_exc()
            _recover_from_oom(device)

    multi_failed = _broadcast_failure(multi_failed, device, group=group)

    # Compose the user-facing failure reason: single and multi legs are
    # independent, so we report both when both OOM'd.
    if single_reason and multi_reason:
        combined_reason = f"{single_reason}; {multi_reason}"
    else:
        combined_reason = single_reason or multi_reason

    return SweepResult(
        model=model_name,
        n_atoms=n_actual,
        single=single_timing,
        multi=multi_timing,
        equivalence=report,
        single_failed=single_failed,
        multi_failed=multi_failed,
        failure_reason=combined_reason,
    )


def run_main(
    model_name: str,
    load_wrapper: Callable[[torch.device, torch.dtype], Any],
    build_harness: Callable[
        ..., tuple[Any, int, torch.Tensor, torch.Tensor, torch.Tensor, Any]
    ],
    args: argparse.Namespace,
    dtype: torch.dtype = torch.float64,
) -> None:
    """Drive the full sweep (single-rank-only or multi-rank) for one model.

    Centralises the init + sweep + print flow so each
    ``benchmark_<model>.py`` is a thin wrapper that provides
    ``load_wrapper`` and ``build_harness``.
    """
    device = resolve_device(args)
    profile_dir = Path(
        args.profile_dir
        if args.profile_dir is not None
        else f"benchmark_profiles/{model_name}"
    )

    if args.single_only or not launched_by_torchrun():
        if not args.single_only and not launched_by_torchrun():
            print(
                "Not launched under torchrun — running single-rank only. "
                "For multi-rank timings, launch with "
                "``torchrun --nproc_per_node=N``."
            )
        print(f"=== {model_name}: single-rank scaling ===")
        wrapper = load_wrapper(device, dtype)
        results: list[SweepResult] = []
        for n in args.sizes:
            r = run_sweep_one_size(
                model_name,
                build_harness,
                wrapper,
                n,
                device,
                dtype,
                args,
                rank=0,
                world_size=1,
                mesh=None,
                profile_dir=profile_dir,
            )
            if r.single_failed:
                print(
                    f"  n_req={n:>6} n_actual={r.n_atoms:>6}  [OOM] {r.failure_reason}"
                )
            else:
                print(
                    f"  n_req={n:>6} n_actual={r.n_atoms:>6}  "
                    f"E={r.single.final_energy_eV:+.6f} eV  "
                    f"step={r.single.step_ms:.3f} ms  "
                    f"peak={_fmt_mem(r.single.peak_mem_mb)} MiB"
                )
            results.append(r)
        print_scaling_table(results, world_size=1)
        return

    rank, world_size, mesh = init_distributed(device.type)
    if device.type == "cuda":
        # Pin to the node-local GPU: multi-node has global rank >= gpus/node,
        # so cuda:{global_rank} is an invalid ordinal on nodes past the first.
        device = torch.device(f"cuda:{int(os.environ.get('LOCAL_RANK', rank))}")

    wrapper = load_wrapper(device, dtype)
    results = []
    for n in args.sizes:
        r = run_sweep_one_size(
            model_name,
            build_harness,
            wrapper,
            n,
            device,
            dtype,
            args,
            rank=rank,
            world_size=world_size,
            mesh=mesh,
            profile_dir=profile_dir,
        )
        if rank == 0:
            # Single-rank and multi-rank legs fail/succeed independently;
            # report both. The most interesting row is "single-OOM but
            # multi-OK" — that's the size the user couldn't fit on one GPU.
            one_e = f"{r.single.final_energy_eV:+.6f}" if not r.single_failed else "—"
            one_peak = _fmt_mem(r.single.peak_mem_mb) if not r.single_failed else "OOM"
            if r.multi_failed or r.multi is None:
                multi_e = "—"
                multi_peak = "OOM" if r.multi_failed else "—"
                speedup = ""
            else:
                multi_e = f"{r.multi.final_energy_eV:+.6f}"
                multi_peak = _fmt_mem(r.multi.peak_mem_mb)
                speedup = (
                    f"  Δ={r.single.step_ms / r.multi.step_ms:.2f}×"
                    if not r.single_failed and r.multi.step_ms > 0
                    else ""
                )
            print(
                f"  n_req={n:>6} n_actual={r.n_atoms:>6}  "
                f"1r_E={one_e} eV  multi_E={multi_e} eV{speedup}  "
                f"peak(1r/multi)={one_peak}/{multi_peak} MiB"
            )
            if (r.single_failed or r.multi_failed) and r.failure_reason:
                print(f"    └─ {r.failure_reason}")
        results.append(r)

    if rank == 0:
        print()
        print(f"=== {model_name}: {world_size}-rank scaling ===")
        print_scaling_table(results, world_size=world_size)
        print_equivalence_table(results)


# ======================================================================
# NVT end-to-end benchmarking — drives ``dynamics.run(batch, n_steps)``
# with hook-based per-step timing so we can report a distribution
# (min/p50/p99/max) instead of just a mean.
# ======================================================================


class _StepTimerHook:
    """Records a CUDA event at one stage of the dynamics step.

    A pair of these (BEFORE_STEP + AFTER_STEP) bracket each step's GPU
    work; ``elapsed_time`` between them gives per-step ms. We register
    the pair on **both** the inner integrator *and* the outer
    DomainParallel adapter so exactly one set fires regardless of which
    step path actually runs:

    * world == 0 (raw integrator): inner hooks fire.
    * world == 1 (DomainParallel falls through to ``inner.run``): inner
      hooks fire; the DD wrapper's BEFORE/AFTER_STEP never trigger.
    * world >= 2 (DD owns the step loop): DD hooks fire; the inner
      integrator's BEFORE_STEP/AFTER_STEP are not called by ``dd.step``.

    Attaching both means the events list grows by exactly one per step
    in every mode without needing to know in advance which path is
    taken.
    """

    def __init__(self, stage: Any, events: list[Any]) -> None:
        self.stage = stage
        self.frequency = 1
        self._events = events

    def __call__(self, ctx: Any, stage: Any) -> None:
        e = torch.cuda.Event(enable_timing=True)
        e.record()
        self._events.append(e)


def _attach_step_timer(
    dyn: Any, dd: Any | None, starts: list[Any], ends: list[Any]
) -> Callable[[], None]:
    """Register BEFORE_STEP + AFTER_STEP timer hooks on whatever
    dynamics objects might fire them. Returns a ``detach()`` callable
    that pops them out of ``hooks`` after the timed window.
    """
    from nvalchemi.dynamics.base import DynamicsStage

    targets = [dyn]
    if dd is not None and dd is not dyn:
        targets.append(dd)
    detach_ops = []
    for t in targets:
        s_hook = _StepTimerHook(DynamicsStage.BEFORE_STEP, starts)
        e_hook = _StepTimerHook(DynamicsStage.AFTER_STEP, ends)
        t.register_hook(s_hook)
        t.register_hook(e_hook)

        def _detach(t=t, s_hook=s_hook, e_hook=e_hook) -> None:
            try:
                t.hooks.remove(s_hook)
            except ValueError:
                pass
            try:
                t.hooks.remove(e_hook)
            except ValueError:
                pass

        detach_ops.append(_detach)

    def detach_all() -> None:
        for op in detach_ops:
            op()

    return detach_all


def time_run(
    runner: Callable[[int], None],
    inner: Any,
    outer: Any | None,
    device: torch.device,
    n_iters: int,
    n_warmup: int,
) -> Timing:
    """Time ``runner(n_steps)`` (which should call ``dyn.run(batch, n_steps)``)
    with hook-based per-step CUDA event timing.

    ``inner`` is the integrator (e.g. NVTLangevin). ``outer`` is the
    DomainParallel adapter when wrapping, or ``None`` for raw integrator.
    Returns mean ms/step + min/p50/p99/max distribution + peak memory.
    """
    _reset_peak_mem(device)

    runner(n_warmup)
    _sync(device)

    starts: list[Any] = []
    ends: list[Any] = []
    detach = _attach_step_timer(inner, outer, starts, ends)
    try:
        _sync(device)
        t0 = time.perf_counter()
        runner(n_iters)
        _sync(device)
        wall_ms = (time.perf_counter() - t0) * 1e3
    finally:
        detach()

    if not starts or not ends:
        # No hook fired — possible if both inner and outer paths declined
        # to dispatch BEFORE_STEP/AFTER_STEP. Fall back to amortized only.
        return Timing(
            step_ms=wall_ms / max(1, n_iters),
            peak_mem_mb=_peak_mem_mb(device),
        )

    n = min(len(starts), len(ends))
    per_step_ms = sorted(starts[i].elapsed_time(ends[i]) for i in range(n))
    return Timing(
        step_ms=wall_ms / n_iters,
        peak_mem_mb=_peak_mem_mb(device),
        min_ms=per_step_ms[0],
        p50_ms=per_step_ms[n // 2],
        p99_ms=per_step_ms[max(0, int(0.99 * (n - 1)))],
        max_ms=per_step_ms[-1],
    )


# Harness builder contract for NVT end-to-end. Each ``benchmark_<model>_nvt.py``
# implements one of these and hands it to :func:`run_nvt_sweep_one_size`. Returns
# ``(runner, inner, outer, n_actual)``:
#
# - ``runner``: ``Callable[[int], None]`` — when called with ``n_steps``,
#   invokes ``dyn.run(batch, n_steps=...)`` end-to-end. The harness can
#   close over batch + dynamics and just dispatch.
# - ``inner``: the ``BaseDynamics`` integrator used; needed by the timer
#   so it can attach BEFORE_STEP/AFTER_STEP hooks.
# - ``outer``: the ``DomainParallel`` wrapper, or ``None`` if not used.
# - ``n_actual``: realised atom count.


@dataclass
class NVTSweepResult:
    """One row of the NVT sweep: world_size + Timing + failure state.

    ``world_label`` distinguishes the three modes we care about:
      * ``"world=0"`` — raw integrator (no DomainParallel wrapper).
      * ``"world=1"`` — DomainParallel single-rank fallback.
      * ``"world=N"`` — full distributed.
    """

    model: str
    n_atoms: int
    world_label: str
    timing: Timing | None = None
    failed: bool = False
    failure_reason: str = ""


def run_nvt_sweep_one_size(
    model_name: str,
    build_nvt_harness: Callable[
        ..., tuple[Callable[[int], None], Any, Any | None, int]
    ],
    wrapper: Any,
    n_atoms: int,
    device: torch.device,
    dtype: torch.dtype,
    args: argparse.Namespace,
    *,
    rank: int,
    world_size: int,
    mesh: Any,
    use_dd: bool,
) -> NVTSweepResult:
    """Time one (model, n_atoms, world) end-to-end. ``use_dd`` selects
    the DomainParallel wrapper path; ``world_size == 1`` + ``use_dd``
    exercises the single-rank fallback. OOM is caught + logged so the
    sweep proceeds to the next size.
    """
    from nvalchemi.distributed._core.gather_primitives import mesh_group

    group = mesh_group(mesh) if mesh is not None and use_dd else None
    if not use_dd:
        world_label = "world=0"
    elif world_size == 1:
        world_label = "world=1"
    else:
        world_label = f"world={world_size}"

    timing: Timing | None = None
    failed = False
    reason = ""
    try:
        runner, inner, outer, n_actual = build_nvt_harness(
            wrapper,
            n_atoms,
            device,
            dtype,
            distributed=use_dd,
            rank=rank,
            world_size=world_size,
            mesh=mesh,
        )
        timing = time_run(
            runner,
            inner=inner,
            outer=outer,
            device=device,
            n_iters=args.iters,
            n_warmup=args.warmup,
        )
    except Exception as exc:  # noqa: BLE001
        if not _is_oom(exc):
            # Surface the non-OOM error BEFORE the failure broadcast: if only
            # some ranks failed, the broadcast collides with the survivors'
            # in-flight model collectives and deadlocks, which would otherwise
            # hide the real exception. Printing first keeps it diagnosable.
            traceback.print_exc()
            _broadcast_failure(True, device, group=group)
            raise
        failed = True
        reason = f"{world_label} OOM: {exc}"
        traceback.print_exc()
        _recover_from_oom(device)
        n_actual = n_atoms

    if use_dd:
        failed = _broadcast_failure(failed, device, group=group)

    return NVTSweepResult(
        model=model_name,
        n_atoms=n_actual,
        world_label=world_label,
        timing=timing,
        failed=failed,
        failure_reason=reason,
    )


def print_nvt_table(results: list[NVTSweepResult]) -> None:
    """Pretty table: rows = (n_atoms, world_label), cols = ms-stats + peak MB."""
    print()
    print(
        f"{'model':<15}{'n_atoms':>8}  {'world':<8}"
        f"{'mean ms':>10}{'min':>9}{'p50':>9}{'p99':>9}{'max':>9}{'peak MB':>11}"
    )
    print("-" * 100)
    for r in results:
        if r.failed or r.timing is None:
            print(
                f"{r.model:<15}{r.n_atoms:>8}  {r.world_label:<8}"
                f"{'OOM':>10}{'—':>9}{'—':>9}{'—':>9}{'—':>9}{'—':>11}"
            )
            if r.failure_reason:
                print(f"{'':<15}  └─ {r.failure_reason}")
            continue
        t = r.timing
        print(
            f"{r.model:<15}{r.n_atoms:>8}  {r.world_label:<8}"
            f"{t.step_ms:>10.3f}{t.min_ms:>9.3f}{t.p50_ms:>9.3f}"
            f"{t.p99_ms:>9.3f}{t.max_ms:>9.3f}{_fmt_mem(t.peak_mem_mb):>11}"
        )


def run_nvt_main(
    model_name: str,
    load_wrapper: Callable[[torch.device, torch.dtype], Any],
    build_nvt_harness: Callable[
        ..., tuple[Callable[[int], None], Any, Any | None, int]
    ],
    args: argparse.Namespace,
    dtype: torch.dtype = torch.float32,
) -> None:
    """Drive the NVT sweep for one model.

    * Without torchrun: world=0 only (raw integrator, no DomainParallel).
    * With ``torchrun --nproc_per_node=1``: world=1 only (DD fallback).
    * With ``torchrun --nproc_per_node=N`` (N>=2): world=N only (full DD).

    Each leg is a separate sbatch invocation — keeps allocator pools
    clean between modes (no cross-contamination from cueq compile cache
    or warp module cache between backends).
    """
    device = resolve_device(args)

    if not launched_by_torchrun():
        # world == 0 — raw integrator, no DD wrapper.
        print(f"=== {model_name}: NVT world=0 (raw integrator) ===")
        wrapper = load_wrapper(device, dtype)
        results: list[NVTSweepResult] = []
        for n in args.sizes:
            r = run_nvt_sweep_one_size(
                model_name,
                build_nvt_harness,
                wrapper,
                n,
                device,
                dtype,
                args,
                rank=0,
                world_size=1,
                mesh=None,
                use_dd=False,
            )
            results.append(r)
            if r.failed:
                print(f"  n={r.n_atoms}  [OOM] {r.failure_reason}")
            else:
                t = r.timing
                print(
                    f"  n={r.n_atoms:>6}  mean={t.step_ms:.3f} ms  "
                    f"p50={t.p50_ms:.3f}  p99={t.p99_ms:.3f}  "
                    f"peak={_fmt_mem(t.peak_mem_mb)} MiB"
                )
        print_nvt_table(results)
        return

    rank, world_size, mesh = init_distributed(device.type)
    if device.type == "cuda":
        # Pin to the node-local GPU: multi-node has global rank >= gpus/node,
        # so cuda:{global_rank} is an invalid ordinal on nodes past the first.
        device = torch.device(f"cuda:{int(os.environ.get('LOCAL_RANK', rank))}")

    wrapper = load_wrapper(device, dtype)
    results = []
    for n in args.sizes:
        r = run_nvt_sweep_one_size(
            model_name,
            build_nvt_harness,
            wrapper,
            n,
            device,
            dtype,
            args,
            rank=rank,
            world_size=world_size,
            mesh=mesh,
            use_dd=True,
        )
        results.append(r)
        if rank == 0:
            if r.failed:
                print(f"  n={r.n_atoms}  [OOM] {r.failure_reason}")
            else:
                t = r.timing
                print(
                    f"  n={r.n_atoms:>6}  mean={t.step_ms:.3f} ms  "
                    f"p50={t.p50_ms:.3f}  p99={t.p99_ms:.3f}  "
                    f"peak={_fmt_mem(t.peak_mem_mb)} MiB"
                )

    if rank == 0:
        print_nvt_table(results)


# ======================================================================
# Config-driven model selection
#
# The runners take ``--config <model>.yaml`` (see ``configs/``). A config
# names a system builder, a loader, and the per-mode distribution knobs;
# everything model-specific lives in data here rather than in a separate
# script per model.
# ======================================================================

_DTYPES = {
    "fp32": torch.float32,
    "fp64": torch.float64,
    "float32": torch.float32,
    "float64": torch.float64,
}


@dataclass
class SystemConfig:
    """Geometry knobs — which builder makes the test system and how it
    is fed into :class:`~nvalchemi.data.AtomicData`."""

    builder: str
    sizing: str | None = None
    pbc: bool | list[bool] = True
    charges: bool = False
    compute_neighbors: bool = True
    partition_mode: str | None = None
    # Parallelization strategy for the DD run: "halo" | "graph_partition".
    # Config-driven (the framework reads DomainConfig.strategy);
    # ``partition_mode`` is derived from it when unset (halo→spatial, GP→
    # contiguous_block). Override per run with ``--set system.strategy=...``.
    strategy: str = "halo"


@dataclass
class BenchConfig:
    """Parsed benchmark config (one per model)."""

    model: str
    loader: dict[str, Any]
    system: SystemConfig
    dtype: str = "fp64"
    default_sizes: list[int] = field(default_factory=lambda: [1000, 4000])
    forward: dict[str, Any] = field(default_factory=dict)
    nvt: dict[str, Any] | None = None


def _coerce(value: str) -> Any:
    """Coerce a ``--set`` string value to bool / int / float / None / str."""
    low = value.lower()
    if low in ("true", "false"):
        return low == "true"
    if low in ("null", "none"):
        return None
    for cast in (int, float):
        try:
            return cast(value)
        except ValueError:
            pass
    return value


def _set_dotted(raw: dict[str, Any], dotted: str, value: Any) -> None:
    """Apply a dotted-key override (e.g. ``loader.enable_cueq``) in place."""
    parts = dotted.split(".")
    node = raw
    for part in parts[:-1]:
        node = node.setdefault(part, {})
    node[parts[-1]] = value


def load_config(path: str, overrides: list[str] | None = None) -> BenchConfig:
    """Load a model config from YAML.

    Parameters
    ----------
    path : str
        Path to the ``<model>.yaml`` config.
    overrides : list[str], optional
        ``KEY=VALUE`` strings (dotted keys, e.g. ``loader.enable_cueq=true``)
        applied on top of the file, for one-off sweeps without new files.

    Returns
    -------
    BenchConfig
        The parsed config.
    """
    with open(path) as fh:
        raw = yaml.safe_load(fh)
    for override in overrides or []:
        key, _, value = override.partition("=")
        _set_dotted(raw, key.strip(), _coerce(value.strip()))
    return BenchConfig(
        model=raw["model"],
        loader=raw.get("loader", {}),
        system=SystemConfig(**raw.get("system", {})),
        dtype=raw.get("dtype", "fp64"),
        default_sizes=raw.get("default_sizes", [1000, 4000]),
        forward=raw.get("forward", {}),
        nvt=raw.get("nvt"),
    )


def resolve_dtype(cfg: BenchConfig) -> torch.dtype:
    """Resolve the forward dtype — cueq forces fp32, else the config dtype."""
    if cfg.loader.get("enable_cueq"):
        return torch.float32
    return _DTYPES[cfg.dtype]


def resolve_attr(obj: Any, dotted: str) -> Any:
    """Read a dotted attribute path (e.g. ``model_config.neighbor_config.cutoff``)."""
    for part in dotted.split("."):
        obj = getattr(obj, part)
    return obj


def pbc_tensor(spec: bool | list[bool]) -> torch.Tensor:
    """Build a ``(3,)`` bool PBC tensor from a scalar or per-axis spec."""
    if isinstance(spec, bool):
        return torch.full((3,), spec, dtype=torch.bool)
    return torch.tensor(spec, dtype=torch.bool)


@dataclass
class System:
    """A realised test system (CPU tensors)."""

    positions: torch.Tensor
    atomic_numbers: torch.Tensor
    masses: torch.Tensor
    cell: torch.Tensor
    velocities: torch.Tensor
    pbc: torch.Tensor
    charges: torch.Tensor | None = None


def build_system(cfg: BenchConfig, n_atoms: int, dtype: torch.dtype) -> System:
    """Realise the config's test system at (approximately) ``n_atoms`` atoms.

    Dispatches ``cfg.system.builder`` / ``cfg.system.sizing`` from this
    module by name. Builders that return charges (e.g. ``build_nacl``)
    are flagged by ``cfg.system.charges``.

    Parameters
    ----------
    cfg : BenchConfig
        The parsed config.
    n_atoms : int
        Requested atom count; the realised count may differ slightly
        because builders round to whole lattices.
    dtype : torch.dtype
        Floating dtype for the geometry tensors.

    Returns
    -------
    System
        The realised geometry as CPU tensors.
    """
    builder = globals()[cfg.system.builder]
    sized = globals()[cfg.system.sizing](n_atoms) if cfg.system.sizing else n_atoms
    out = builder(sized, dtype)
    if cfg.system.charges:
        positions, atomic_numbers, masses, charges, cell, velocities = out
    else:
        positions, atomic_numbers, masses, cell, velocities = out
        charges = None
    return System(
        positions=positions,
        atomic_numbers=atomic_numbers,
        masses=masses,
        cell=cell,
        velocities=velocities,
        pbc=pbc_tensor(cfg.system.pbc),
        charges=charges,
    )


def resolve_strategy(name: str) -> tuple[Any, str]:
    """Map a ``--set system.strategy`` choice to ``(StrategyKind, partition_mode)``.

    ``halo`` → spatial domain decomposition (owned + ghost);
    ``graph_partition`` → graph parallel over a balanced contiguous-block
    partition. The framework selects the model's per-strategy spec from
    ``DomainConfig.strategy``; the returned ``partition_mode`` keeps the
    ``ShardedBatch`` layout consistent with that choice.
    """
    from nvalchemi.distributed.config import StrategyKind

    mapping = {
        "halo": (StrategyKind.HALO, "spatial"),
        "graph_partition": (StrategyKind.GRAPH_PARTITION, "contiguous_block"),
    }
    if name not in mapping:
        raise ValueError(
            f"system.strategy must be one of {sorted(mapping)}; got {name!r}"
        )
    return mapping[name]


def resolve_inference(name: str) -> Any:
    """Map a ``--set loader.inference`` choice to a fairchem inference setting.

    ``default`` -> eager; ``turbo`` -> stock turbo preset (compile + tf32 +
    merge_mole); ``compile`` -> compile + merge_mole WITHOUT tf32 (tight
    compiled-DD numerics; merge_mole is required — fairchem's MoLE asserts
    under compile without it).
    """
    if name in ("default", "turbo"):
        return name
    if name == "compile":
        from fairchem.core.units.mlip_unit.api.inference import (  # noqa: PLC0415
            InferenceSettings,
        )

        return InferenceSettings(
            compile=True, merge_mole=True, activation_checkpointing=False
        )
    raise ValueError(f"unknown inference setting {name!r}")


def _load_lj(device: torch.device, dtype: torch.dtype, lc: dict, **_: Any) -> Any:
    from nvalchemi.models.lj import LennardJonesModelWrapper

    wrapper = LennardJonesModelWrapper(
        epsilon=lc["epsilon"], sigma=lc["sigma"], cutoff=lc["cutoff"]
    )
    return wrapper.eval().to(device=device)


def _load_electrostatic(
    device: torch.device, dtype: torch.dtype, lc: dict, **_: Any
) -> Any:
    module = importlib.import_module(lc["module"])
    wrapper_cls = getattr(module, lc["class"])
    # hybrid_forces=False routes through the staged bindings + the
    # owned_slice/all_reduce handler under halo — the distributed path.
    wrapper = wrapper_cls(
        cutoff=lc.get("cutoff", 6.0), hybrid_forces=lc.get("hybrid_forces", False)
    )
    wrapper.eval().to(device=device)
    return wrapper


def _load_mace(
    device: torch.device, dtype: torch.dtype, lc: dict, *, compile_model: bool = False
) -> Any:
    import warnings  # noqa: PLC0415

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from nvalchemi.models.mace import MACEWrapper
    wrapper = MACEWrapper.from_checkpoint(
        lc["checkpoint"],
        dtype=dtype,
        device=device,
        enable_cueq=lc.get("enable_cueq", False),
        compile_model=compile_model,
    )
    return wrapper.eval()


def _load_aimnet2(
    device: torch.device, dtype: torch.dtype, lc: dict, *, compile_model: bool = False
) -> Any:
    from nvalchemi.models.aimnet2 import AIMNet2Wrapper

    wrapper = AIMNet2Wrapper.from_checkpoint(
        lc["checkpoint"], device=device, compile_model=compile_model
    )
    # AIMNet2's warp kernels are float32-only; cast the model + its buffers.
    wrapper.model.to(dtype)
    for mod in wrapper.model.modules():
        for name, buf in list(mod.named_buffers(recurse=False)):
            if buf.is_floating_point():
                setattr(mod, name, buf.to(dtype))
    wrapper.eval()
    # The benchmark only consumes energy + forces; trim charges.
    wrapper.model_config.active_outputs = {"energy", "forces"}
    return wrapper


def _load_uma(device: torch.device, dtype: torch.dtype, lc: dict, **_: Any) -> Any:
    from nvalchemi.models.uma import UMAWrapper

    return UMAWrapper.from_checkpoint(
        lc["checkpoint"],
        task_name=lc.get("task", "omat"),
        device=device,
        inference_settings=resolve_inference(lc.get("inference", "default")),
    )


LOADERS: dict[str, Callable[..., Any]] = {
    "lj": _load_lj,
    "electrostatic": _load_electrostatic,
    "mace": _load_mace,
    "aimnet2": _load_aimnet2,
    "uma": _load_uma,
}


def build_loader(
    cfg: BenchConfig, *, compile_model: bool = False
) -> Callable[[torch.device, torch.dtype], Any]:
    """Build the ``load_wrapper(device, dtype)`` callback for a config.

    Parameters
    ----------
    cfg : BenchConfig
        The parsed config; ``cfg.loader["kind"]`` selects the loader.
    compile_model : bool, optional
        Whether the inner model should be ``torch.compile``-d at load
        (only the compile-capable loaders honour it).

    Returns
    -------
    Callable[[torch.device, torch.dtype], Any]
        A loader matching the ``run_main`` / ``run_nvt_main`` contract.
    """
    fn = LOADERS[cfg.loader["kind"]]

    def load_wrapper(device: torch.device, dtype: torch.dtype) -> Any:
        return fn(device, dtype, cfg.loader, compile_model=compile_model)

    return load_wrapper
