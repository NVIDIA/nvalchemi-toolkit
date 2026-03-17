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
"""MACE / LJ performance benchmark.

Three benchmarks, each timed over ``--n-repeats`` iterations after
``--n-warmup`` warmup iterations:

1. **prebuilt_nl** — neighbor list built once; ``n-repeats`` model evaluations.
2. **nl_plus_eval** — neighbor list rebuilt + model evaluation each iteration.
3. **nvt_dynamics** — ``n-repeats`` NVT-Langevin steps (NL rebuild via hook).

Both MACE (COO neighbor list) and Lennard-Jones (neighbor matrix) are supported
via ``--model mace`` / ``--model lj``.

NVTX ranges are emitted for profiling with Nsight Systems::

    nsys profile --trace=cuda,nvtx \\
        python benchmarks/mace_benchmark.py \\
            --model mace --checkpoint medium-mpa-0 \\
            --n-atoms 512 --batch-size 4

Usage
-----
::

    python benchmarks/mace_benchmark.py --model lj --n-atoms 256 --batch-size 8
    python benchmarks/mace_benchmark.py --model mace --checkpoint medium-mpa-0 --n-atoms 128
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass, field

import torch

try:
    import nvtx as _nvtx
except ImportError:
    _nvtx = None


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------


@dataclass
class BenchResult:
    name: str
    times_ms: list[float] = field(default_factory=list)
    n_atoms: int = 0

    @property
    def mean_ms(self) -> float:
        return sum(self.times_ms) / len(self.times_ms) if self.times_ms else 0.0

    @property
    def std_ms(self) -> float:
        if len(self.times_ms) < 2:
            return 0.0
        mu = self.mean_ms
        return math.sqrt(
            sum((t - mu) ** 2 for t in self.times_ms) / (len(self.times_ms) - 1)
        )

    @property
    def min_ms(self) -> float:
        return min(self.times_ms) if self.times_ms else 0.0

    @property
    def max_ms(self) -> float:
        return max(self.times_ms) if self.times_ms else 0.0

    @property
    def throughput_atoms_per_s(self) -> float:
        if self.mean_ms == 0.0:
            return 0.0
        return self.n_atoms / (self.mean_ms * 1e-3)

    @property
    def us_per_atom_per_step(self) -> float:
        if self.n_atoms == 0:
            return 0.0
        return self.mean_ms * 1e3 / self.n_atoms

    def report(self) -> None:
        """Report benchmark results."""
        print(
            f"  {self.name:30s}  "
            f"mean={self.mean_ms:8.2f} ms  "
            f"std={self.std_ms:7.2f} ms  "
            f"min={self.min_ms:8.2f} ms  "
            f"max={self.max_ms:8.2f} ms  "
            f"throughput={self.throughput_atoms_per_s:.3e} atoms/s  "
            f"us/atom/step={self.us_per_atom_per_step:.4f}"
        )


class BenchTimer:
    """Measures wall time (CPU) or CUDA event time (GPU) for a code block."""

    def __init__(self, device: torch.device) -> None:
        self._device = device
        self._use_cuda = device.type == "cuda"
        self._start_event: torch.cuda.Event | None = None
        self._end_event: torch.cuda.Event | None = None
        self._start_cpu: float = 0.0
        self._end_cpu: float = 0.0

    def __enter__(self) -> "BenchTimer":
        if self._use_cuda:
            self._start_event = torch.cuda.Event(enable_timing=True)
            self._end_event = torch.cuda.Event(enable_timing=True)
            self._start_event.record()
        else:
            self._start_cpu = time.perf_counter()
        return self

    def __exit__(self, *_: object) -> None:
        if self._use_cuda:
            assert self._end_event is not None  # noqa: S101
            self._end_event.record()
            torch.cuda.synchronize(self._device)
        else:
            self._end_cpu = time.perf_counter()

    @property
    def elapsed_ms(self) -> float:
        if self._use_cuda:
            assert self._start_event is not None and self._end_event is not None  # noqa: S101
            return self._start_event.elapsed_time(self._end_event)
        return (self._end_cpu - self._start_cpu) * 1e3


# ---------------------------------------------------------------------------
# System builder
# ---------------------------------------------------------------------------


def _make_argon_crystal(
    n_atoms_target: int,
    batch_size: int,
    device: torch.device,
) -> tuple["Batch", int]:
    """Build a batch of ``batch_size`` periodic argon simple-cubic crystals.

    Each system has ~``n_atoms_target`` atoms (scaled to the nearest cube:
    ``n_per_side = round(n_atoms_target ** (1/3))``, actual N = n_per_side³).
    Atoms are placed on a simple cubic lattice at the LJ equilibrium spacing
    (≈ 3.82 Å).  All systems are identical copies.

    Returns
    -------
    batch : Batch
    n_atoms_total : int
        Total atom count across all systems in the batch.
    """
    from nvalchemi.data import AtomicData, Batch

    n_per_side = max(1, round(n_atoms_target ** (1 / 3)))
    spacing = 2 ** (1 / 6) * 3.40  # Å — argon LJ r_min

    coords = torch.arange(n_per_side, dtype=torch.float32) * spacing
    gx, gy, gz = torch.meshgrid(coords, coords, coords, indexing="ij")
    positions = torch.stack([gx.flatten(), gy.flatten(), gz.flatten()], dim=-1)
    n = positions.shape[0]

    box_len = n_per_side * spacing
    cell = torch.eye(3, dtype=torch.float32) * box_len  # [3, 3]
    pbc = torch.ones(3, dtype=torch.bool)

    data = AtomicData(
        positions=positions,
        atomic_numbers=torch.full((n,), 18, dtype=torch.long),  # Ar Z=18
        forces=torch.zeros(n, 3),
        energies=torch.zeros(1, 1),
        velocities=torch.zeros(n, 3),
        cell=cell.unsqueeze(0),  # [1, 3, 3]
        pbc=pbc.unsqueeze(0),  # [1, 3]
    )
    batch = Batch.from_data_list([data] * batch_size, device=device)
    return batch, n * batch_size


# ---------------------------------------------------------------------------
# Model loaders
# ---------------------------------------------------------------------------


def _load_mace(checkpoint: str, device: torch.device, dtype: torch.dtype):
    from nvalchemi.models.mace import MACEWrapper

    wrapper = MACEWrapper.from_checkpoint(
        checkpoint_path=checkpoint, device=device, dtype=dtype
    )
    wrapper.eval()
    return wrapper


def _load_lj(device: torch.device, dtype: torch.dtype):
    from nvalchemi.models.lj import LennardJonesModelWrapper

    wrapper = LennardJonesModelWrapper(
        epsilon=0.0104,  # eV — argon ε
        sigma=3.40,  # Å  — argon σ
        cutoff=8.5,  # Å
        max_neighbors=64,
    )
    wrapper.eval()
    return wrapper.to(device)


# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------


def run_bench_prebuilt_nl(
    wrapper,
    batch: "Batch",
    n_atoms: int,
    n_repeats: int,
    n_warmup: int,
    device: torch.device,
    skin: float,
) -> BenchResult:
    """Build neighbor list once; time ``n_repeats`` model forward passes."""
    from nvalchemi.dynamics.hooks import NeighborListHook

    nl_hook = NeighborListHook(wrapper.model_card.neighbor_config, skin=skin)
    nl_hook(batch, None)  # build NL once

    wrapper.model_config.compute_forces = True
    timer = BenchTimer(device)
    result = BenchResult(name="prebuilt_nl", n_atoms=n_atoms)

    for i in range(n_warmup + n_repeats):
        if _nvtx is not None and i == n_warmup:
            _nvtx.push_range("mace_bench/prebuilt_nl")
        with timer:
            out = wrapper.forward(batch)
            _ = out["energies"].sum()
        if i >= n_warmup:
            result.times_ms.append(timer.elapsed_ms)

    if _nvtx is not None:
        _nvtx.pop_range()

    return result


def run_bench_nl_plus_eval(
    wrapper,
    batch: "Batch",
    n_atoms: int,
    n_repeats: int,
    n_warmup: int,
    device: torch.device,
    skin: float,
) -> BenchResult:
    """Time neighbor-list rebuild + model forward pass each iteration."""
    from nvalchemi.dynamics.hooks import NeighborListHook

    nl_hook = NeighborListHook(wrapper.model_card.neighbor_config, skin=skin)
    wrapper.model_config.compute_forces = True
    timer = BenchTimer(device)
    result = BenchResult(name="nl_plus_eval", n_atoms=n_atoms)

    for i in range(n_warmup + n_repeats):
        if _nvtx is not None and i == n_warmup:
            _nvtx.push_range("mace_bench/nl_plus_eval")
        with timer:
            nl_hook(batch, None)
            out = wrapper.forward(batch)
            _ = out["energies"].sum()
        if i >= n_warmup:
            result.times_ms.append(timer.elapsed_ms)

    if _nvtx is not None:
        _nvtx.pop_range()

    return result


def run_bench_nvt(
    wrapper,
    batch: "Batch",
    n_atoms: int,
    n_repeats: int,
    n_warmup: int,
    device: torch.device,
    dt: float,
    skin: float,
) -> BenchResult:
    """Time ``n_repeats`` NVT-Langevin steps (includes NL rebuild via hook)."""
    from nvalchemi.dynamics import NVTLangevin
    from nvalchemi.dynamics.hooks import NeighborListHook

    def _make_nvt(n_steps: int, seed: int) -> NVTLangevin:
        nvt = NVTLangevin(
            model=wrapper,
            dt=dt,
            temperature=300.0,
            friction=1.0,
            n_steps=n_steps,
            random_seed=seed,
        )
        if wrapper.model_card.neighbor_config is not None:
            nvt.register_hook(
                NeighborListHook(wrapper.model_card.neighbor_config, skin=skin)
            )
        return nvt

    timer = BenchTimer(device)
    result = BenchResult(name="nvt_dynamics", n_atoms=n_atoms)

    if n_warmup > 0:
        _make_nvt(n_warmup, seed=1).run(batch)
    torch.cuda.cudart().cudaProfilerStart()
    nvt = _make_nvt(n_repeats, seed=42)

    with timer:
        nvt.run(batch)
    torch.cuda.cudart().cudaProfilerStop()
    # Distribute total time evenly across steps for per-step stats.
    per_step_ms = timer.elapsed_ms / n_repeats
    result.times_ms = [per_step_ms] * n_repeats

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="MACE / LJ performance benchmark",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--model",
        default="mace",
        choices=["mace", "lj"],
        help="Model to benchmark.  'mace' uses COO neighbor list; 'lj' uses neighbor matrix.",
    )
    p.add_argument(
        "--checkpoint",
        default="medium-mpa-0",
        help="MACE checkpoint path or named model (e.g. 'medium-mpa-0').  Ignored for --model lj.",
    )
    p.add_argument("--n-atoms", type=int, default=64, help="Target atoms per system")
    p.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Number of systems in the batch.  Total atoms = n-atoms × batch-size (approx).",
    )
    p.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device string",
    )
    p.add_argument(
        "--dtype",
        default="float32",
        choices=["float32", "float64"],
        help="Model dtype",
    )
    p.add_argument("--n-repeats", type=int, default=100, help="Timed iterations")
    p.add_argument("--n-warmup", type=int, default=10, help="Warmup iterations")
    p.add_argument(
        "--skin",
        type=float,
        default=2.0,
        help="Verlet skin distance in Å.  0 = rebuild every step.",
    )
    p.add_argument(
        "--dt",
        type=float,
        default=0.01,
        help="NVT timestep in nvalchemi time units (~10 fs/unit)",
    )
    p.add_argument(
        "--skip-nvt",
        action="store_true",
        help="Skip NVT benchmark",
    )
    p.add_argument(
        "--enable-cueq",
        action="store_true",
        help="Enable cuEquivariance for MACE (ignored for LJ)",
    )
    return p.parse_args()


def main() -> None:
    """Main entry point."""
    args = _parse_args()

    device = torch.device(args.device)
    dtype = torch.float32 if args.dtype == "float32" else torch.float64

    # ---- Load model ---------------------------------------------------------
    if args.model == "mace":
        print(f"Loading MACE checkpoint: {args.checkpoint}")
        wrapper = _load_mace(args.checkpoint, device, dtype)
    else:
        print("Using Lennard-Jones model (argon parameters)")
        wrapper = _load_lj(device, dtype)

    nc = wrapper.model_card.neighbor_config
    print(
        f"Neighbor config: cutoff={nc.cutoff:.2f} Å, "
        f"format={nc.format.name}"
        + (f", max_neighbors={nc.max_neighbors}" if nc.max_neighbors else "")
    )

    # ---- Build system -------------------------------------------------------
    batch, n_atoms_total = _make_argon_crystal(args.n_atoms, args.batch_size, device)
    n_per_system = n_atoms_total // args.batch_size
    print(
        f"System: {args.batch_size} × {n_per_system} atoms "
        f"= {n_atoms_total} total atoms on {device}"
        + (f"  skin={args.skin:.2f} Å" if args.skin > 0 else "")
    )
    print(f"Repeats: {args.n_repeats} timed + {args.n_warmup} warmup\n")

    results: list[BenchResult] = []
    common = dict(
        n_atoms=n_atoms_total,
        n_repeats=args.n_repeats,
        n_warmup=args.n_warmup,
        device=device,
        skin=args.skin,
    )

    # ---- Benchmark 1: prebuilt neighbor list --------------------------------
    print("Benchmark 1: prebuilt neighbor list …")
    results.append(run_bench_prebuilt_nl(wrapper, batch, **common))

    # ---- Benchmark 2: NL rebuild + eval -------------------------------------
    print("Benchmark 2: NL rebuild + model eval …")
    results.append(run_bench_nl_plus_eval(wrapper, batch, **common))

    # ---- Benchmark 3: NVT dynamics ------------------------------------------
    if not args.skip_nvt:
        print("Benchmark 3: NVT dynamics …")
        results.append(run_bench_nvt(wrapper, batch, **common, dt=args.dt))

    # ---- Report ----------------------------------------------------------------
    print("\n--- Results ---")
    for r in results:
        r.report()


if __name__ == "__main__":
    main()
