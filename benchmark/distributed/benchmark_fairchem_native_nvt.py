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
"""fairchem-native distributed NVT MD benchmark — a baseline for §7a.

This times fairchem's OWN graph-parallel MD throughput + per-rank peak
memory so it can sit next to the ``UMA graph_partition`` NVT numbers in
``DD_SCALING_SWEEP_2026-07-01.md`` §7a (p50 step_ms + per-rank peak MiB,
worlds 1/2/4/8, atom ladder ~2k–31k). It is the head-to-head reference:
"what does fairchem give you natively for the same model, same systems,
same metric" — no toolkit DD wrapper in the loop.

Path (verified against installed fairchem 2.21)
-----------------------------------------------
* **Distributed GP inference is Ray-backed, NOT torchrun.**
  ``pretrained_mlip.get_predict_unit(..., workers=N)`` with ``N > 1``
  returns a ``ParallelMLIPPredictUnit`` that spawns one Ray GPU-actor
  per rank. A single-system predict splits the nodes across ranks via
  ``tensor_split(arange(n_atoms), N)`` — the same graph_partition our
  §7a UMA leg uses. Launch with ``WORKERS=N python <script>`` (Ray
  places the actors); do NOT wrap it in ``torchrun``.
* ``inference_settings="turbo"`` = compile + merge_mole + tf32, matching
  §7a's UMA turbo env.
* **MD driver is ASE Langevin** through ``FAIRChemCalculator``. One
  ``get_forces`` fans out across all GP ranks; only rank-0 (the driver,
  an in-process ``MLIPWorkerLocal``) is visible to the MD loop.

Metrics (match §7a)
-------------------
* **p50 step_ms** = median of per-step ``time.perf_counter`` over
  ``--timed`` timed ``dyn.run(1)`` steps after ``--warmup`` warmup steps
  (the first compiled step is absorbed by warmup).
* **per-rank peak MiB**. The model runs in Ray worker processes, so the
  driver's ``torch.cuda.max_memory_allocated`` only sees rank-0. Two
  measurement methods, selected with ``--mem-method``:

  1. ``patch`` (default): apply ``fairchem_worker_peakmem.patch`` to the
     fairchem checkout first (adds ``get_peak_mem`` / ``reset_peak_mem``
     remote methods to ``MLIPWorkerLocal``). The script resets after
     warmup and ``ray.get``s every worker's peak at the end — the SAME
     ``max_memory_allocated`` statistic as §7a, on every rank.
  2. ``smi``: no patch needed — samples ``nvidia-smi`` ``memory.used``
     during steady state. NOTE: ``memory.used`` includes the CUDA
     context (~hundreds of MiB) and is not directly comparable to §7a's
     ``max_memory_allocated``; it is a fallback, footnoted as such.

Peak note: rank-0 runs in the driver process, so under ``--mem-method
patch`` its peak is read locally (``torch.cuda.max_memory_allocated``);
only ranks 1..N-1 need the remote call (they are the entries in
``predictor.workers``).

Output
------
A CSV with the same columns as §7a's NVT CSV —
``model,env_torch,strategy,world,n_atoms,step_ms_p50,peak_MiB_per_rank,status``
— labeled ``fairchem_native_gp``, plus a human table on stdout. Per-size
OOM is caught, recorded as ``status=OOM``, and the sweep continues.

Launch
------
Single node only; Ray spawns the actors (no torchrun)::

    # A6000 box (48 GB): worlds 1/2 only. UMA turbo ~13.7 MiB/atom → a
    # ~3k-atom/GPU ceiling, so keep the ladder small at w1/w2.
    WORKERS=1 .../.tlkit-uma/bin/python benchmark_fairchem_native_nvt.py \
        --sizes 2000 4394 --nsteps 20
    WORKERS=2 .../.tlkit-uma/bin/python benchmark_fairchem_native_nvt.py \
        --sizes 2000 4394 8192 --nsteps 20

    # DFW H100 (80 GB): worlds 4/8 for the full ladder.
    WORKERS=4 .../.tlkit-uma/bin/python benchmark_fairchem_native_nvt.py \
        --sizes 2000 4394 8192 16000 --nsteps 20
    WORKERS=8 .../.tlkit-uma/bin/python benchmark_fairchem_native_nvt.py \
        --sizes 2000 4394 8192 16000 24334 31250 --nsteps 20

See ``README_fairchem_native.md`` for the patch-apply step and how to
fold the CSV back into §7a as a comparison column.
"""

from __future__ import annotations

import argparse
import os
import statistics
import subprocess
import time

# Reference atom ladder from §7a (UMA graph_partition NVT). bcc-Fe cubic is
# 2 atoms/cell, so N = 2·r³ where r is the supercell repeat. Every §7a target
# lands on an exact 2·r³: r ∈ {10, 13, 16, 20, 23, 25} → {2000, 4394, 8192,
# 16000, 24334, 31250}. The builder snaps any requested size to the nearest r.
DEFAULT_SIZES = [2000, 4394, 8192, 16000, 24334, 31250]


def is_oom(exc: BaseException) -> bool:
    """True if ``exc`` looks like a CUDA out-of-memory error.

    ``torch`` is imported lazily here so the module lints / ``--help``s
    without torch installed.
    """
    import torch

    oom_cls = getattr(torch.cuda, "OutOfMemoryError", None)
    if oom_cls is not None and isinstance(exc, oom_cls):
        return True
    if isinstance(exc, RuntimeError):
        msg = str(exc).lower()
        return any(
            m in msg
            for m in ("out of memory", "failed to allocate", "cudaerrormemoryallocation")
        )
    return False


def bcc_reps_for_size(n_atoms: int) -> int:
    """Supercell repeat ``r`` whose ``2·r³`` is closest to ``n_atoms``."""
    return max(1, round((n_atoms / 2.0) ** (1.0 / 3.0)))


def build_atoms(n_atoms: int, temperature_K: float):
    """bcc-Fe supercell mirroring ``fairchem_bcc_bench.py`` exactly.

    ``bulk("Fe", "bcc", a=2.87, cubic=True) * (r, r, r)`` (2 atoms/cell),
    a small seed-0 rattle, and a Maxwell-Boltzmann velocity draw so the
    integrator starts from a physical state.
    """
    import numpy as np
    from ase.build import bulk
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

    reps = bcc_reps_for_size(n_atoms)
    atoms = bulk("Fe", "bcc", a=2.87, cubic=True) * (reps, reps, reps)
    rng = np.random.default_rng(0)
    atoms.positions[:] = atoms.positions + 0.05 * rng.standard_normal(
        atoms.positions.shape
    )
    MaxwellBoltzmannDistribution(atoms, temperature_K=temperature_K)
    return atoms


def build_predictor(model: str, workers: int, sett: str):
    """Construct the (parallel when ``workers > 1``) predict unit + calc.

    ``inference_settings="turbo"`` == compile + merge_mole + tf32 (§7a's
    UMA env). ``workers > 1`` routes to ``ParallelMLIPPredictUnit`` (one
    Ray GPU-actor per rank, single-system node split).
    """
    from fairchem.core import FAIRChemCalculator, pretrained_mlip

    predictor = pretrained_mlip.get_predict_unit(
        model, inference_settings=sett, device="cuda", workers=workers
    )
    calc = FAIRChemCalculator(predictor, task_name="omat")
    # Forces MUST be an implemented property for the MD loop to fan out. UMA's
    # omat task ships trained forces, so this holds for turbo; assert loudly so a
    # future checkpoint/task change surfaces here (fix: predict_untrained_forces
    # ={"omat"} on a custom InferenceSettings) rather than as a silent MD stall.
    assert "forces" in calc.implemented_properties, (
        "'forces' not in calc.implemented_properties for task 'omat' — set "
        "predict_untrained_forces={'omat'} on a custom InferenceSettings."
    )
    return predictor, calc


def _visible_gpu_ids(workers: int) -> str:
    """Comma-joined device ids for ``nvidia-smi -i`` under ``--mem-method smi``.

    Honours ``CUDA_VISIBLE_DEVICES`` when set; otherwise assumes the first
    ``workers`` physical GPUs (Ray packs one actor per GPU, driver on 0).
    """
    env = os.environ.get("CUDA_VISIBLE_DEVICES")
    if env:
        ids = [x for x in env.split(",") if x.strip() != ""]
        return ",".join(ids[:workers]) if ids else ",".join(str(i) for i in range(workers))
    return ",".join(str(i) for i in range(workers))


def _smi_used_mib(gpu_ids: str) -> list[float]:
    """Sample ``nvidia-smi`` ``memory.used`` (MiB) for ``gpu_ids``."""
    # Fixed argv (nvidia-smi on PATH in the CUDA/UMA env); gpu_ids from our own env.
    argv = ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits", "-i", gpu_ids]  # noqa: S607
    out = subprocess.run(argv, check=True, capture_output=True, text=True)  # noqa: S603
    return [float(line) for line in out.stdout.splitlines() if line.strip()]


def _reset_peak_all(predictor, workers: int) -> None:
    """Reset peak-mem stats on every rank (patch path).

    Rank-0 lives in the driver process; ranks 1..N-1 are the Ray actors
    in ``predictor.workers``. Reset all of them so the post-warmup window
    is what the reported peak reflects.
    """
    import ray
    import torch

    torch.cuda.reset_peak_memory_stats()
    if workers > 1:
        ray.get([w.reset_peak_mem.remote() for w in predictor.workers])


def _peak_mib_all(predictor, workers: int) -> list[float]:
    """Per-rank peak allocated MiB across all ranks (patch path)."""
    import ray
    import torch

    peaks = [float(torch.cuda.max_memory_allocated()) / (1024.0 * 1024.0)]
    if workers > 1:
        remote = ray.get([w.get_peak_mem.remote() for w in predictor.workers])
        peaks.extend(float(b) / (1024.0 * 1024.0) for b in remote)
    return peaks


def _one_step(dyn) -> None:
    """Advance the MD by exactly one step (one fanned-out force eval)."""
    dyn.run(1)


def bench_size(
    n_atoms: int,
    args: argparse.Namespace,
    predictor,
    calc,
) -> dict:
    """Time one system size; return a CSV-ready row dict.

    Returns ``step_ms_p50`` (median of ``--timed`` per-step wall times),
    ``peak_MiB_per_rank`` (max over ranks — the per-GPU ceiling), and a
    ``status`` of ``ok`` / ``OOM`` / ``error``. On OOM the caller keeps
    sweeping smaller-safe sizes.
    """
    import torch
    from ase import units
    from ase.md.langevin import Langevin

    atoms = build_atoms(n_atoms, args.temperature)
    n_actual = len(atoms)
    atoms.calc = calc
    dyn = Langevin(
        atoms,
        timestep=1.0 * units.fs,
        temperature_K=args.temperature,
        friction=0.01 / units.fs,
    )

    row = {
        "model": args.model,
        "env_torch": torch.__version__.split("+")[0],
        "strategy": "fairchem_native_gp",
        "world": args.workers,
        "n_atoms": n_actual,
        "step_ms_p50": "",
        "peak_MiB_per_rank": "",
        "status": "ok",
    }
    gpu_ids = _visible_gpu_ids(args.workers)

    try:
        for _ in range(args.warmup):
            _one_step(dyn)
        torch.cuda.synchronize()

        # Peak-mem window opens after warmup so the one-time compile/autotune
        # transient (larger than steady state) is excluded — same as §7a.
        if args.mem_method == "patch":
            _reset_peak_all(predictor, args.workers)
        smi_samples: list[list[float]] = []

        per_step_ms: list[float] = []
        for _ in range(args.timed):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            _one_step(dyn)
            torch.cuda.synchronize()
            per_step_ms.append((time.perf_counter() - t0) * 1e3)
            if args.mem_method == "smi":
                smi_samples.append(_smi_used_mib(gpu_ids))

        row["step_ms_p50"] = f"{statistics.median(per_step_ms):.3f}"

        if args.mem_method == "patch":
            peaks = _peak_mib_all(predictor, args.workers)
        else:
            # Per-rank steady-state max of memory.used across the timed samples.
            peaks = [max(col) for col in zip(*smi_samples)] if smi_samples else []
        row["peak_MiB_per_rank"] = f"{max(peaks):.1f}" if peaks else ""
    except Exception as exc:  # noqa: BLE001 — per-size failure must not kill the sweep
        row["status"] = "OOM" if is_oom(exc) else "error"
        row["step_ms_p50"] = ""
        row["peak_MiB_per_rank"] = ""
        print(f"  n={n_actual}: {row['status']} — {type(exc).__name__}: {str(exc)[:100]}")
        if os.environ.get("NVALCHEMI_FC_TRACEBACK"):
            import traceback  # noqa: PLC0415

            traceback.print_exc()
        torch.cuda.empty_cache()

    return row


def _run_isolated(args: Any) -> None:
    """Re-exec one fresh subprocess per size, then merge the per-size CSVs.

    Each child runs a single ``--sizes N --no-isolate`` and inherits the env
    (WORKERS / NCCL_P2P_DISABLE / PYTHONPATH), so every N gets its own Ray group +
    fresh compile. Avoids fairchem's cross-size ``AtomicData.validate`` assertion.
    """
    import csv as _csv  # noqa: PLC0415
    import subprocess  # noqa: PLC0415
    import sys as _sys  # noqa: PLC0415

    import torch  # noqa: PLC0415 — for the crashed-child fallback row's env_torch

    merged: list[dict] = []
    for n in args.sizes:
        tmp = f"{args.csv}.size{n}.tmp"
        cmd = [
            _sys.executable, os.path.abspath(__file__),
            "--sizes", str(n),
            "--nsteps", str(args.nsteps), "--warmup", str(args.warmup),
            "--timed", str(args.timed), "--model", args.model, "--sett", args.sett,
            "--task", args.task, "--temperature", str(args.temperature),
            "--mem-method", args.mem_method, "--csv", tmp, "--no-isolate",
        ]
        print(f"[isolate] size {n} → fresh subprocess", flush=True)
        rc = subprocess.run(  # noqa: S603 — cmd is our own argv, not untrusted
            cmd, env=os.environ.copy(), check=False
        ).returncode
        if os.path.exists(tmp):
            with open(tmp, newline="") as f:
                merged.extend(list(_csv.DictReader(f)))
            os.remove(tmp)
        else:
            # Child crashed before writing its CSV (e.g. OOM-killed process).
            merged.append({
                "model": args.model, "env_torch": torch.__version__.split("+")[0],
                "strategy": "fairchem_native_gp", "world": args.workers,
                "n_atoms": n, "step_ms_p50": "", "peak_MiB_per_rank": "",
                "status": f"crashed(rc={rc})",
            })
    _write_csv(args.csv, merged)
    _print_table(merged, args)


def main() -> None:
    """Entry point — sweep sizes, print a table, write the CSV."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--sizes",
        nargs="+",
        type=int,
        default=DEFAULT_SIZES,
        help="Target atom counts (snapped to nearest bcc 2·r³). Default: §7a ladder.",
    )
    parser.add_argument(
        "--nsteps",
        type=int,
        default=20,
        help="Total NVT steps to plan for (warmup + timed must not exceed this).",
    )
    parser.add_argument("--warmup", type=int, default=5, help="Warmup steps (absorb compile).")
    parser.add_argument("--timed", type=int, default=10, help="Timed steps for the p50.")
    parser.add_argument("--model", default="uma-s-1p1", help="Pretrained model name.")
    parser.add_argument(
        "--sett",
        default="turbo",
        help="fairchem inference setting (turbo = compile+merge_mole+tf32, matches §7a).",
    )
    parser.add_argument("--task", default="omat", help="UMA task name (matches §7a's omat).")
    parser.add_argument("--temperature", type=float, default=300.0, help="Langevin T (K).")
    parser.add_argument(
        "--mem-method",
        choices=["patch", "smi"],
        default="patch",
        help=(
            "Peak-mem source: 'patch' (fairchem_worker_peakmem.patch — "
            "max_memory_allocated, matches §7a) or 'smi' (nvidia-smi memory.used, "
            "no patch; includes CUDA context)."
        ),
    )
    parser.add_argument(
        "--csv",
        default="fairchem_native_nvt.csv",
        help="Output CSV path.",
    )
    parser.add_argument(
        "--no-isolate",
        action="store_true",
        help=(
            "Run all sizes in THIS process. Default (off) re-execs one fresh "
            "subprocess per size: fairchem carries compiled-graph / atomic-data "
            "state across sizes in a shared process, so the 2nd+ size trips "
            "AtomicData.validate — a fresh process per N avoids it."
        ),
    )
    args = parser.parse_args()

    args.workers = int(os.environ.get("WORKERS", "1"))
    if args.warmup + args.timed > args.nsteps:
        args.nsteps = args.warmup + args.timed

    # Self-isolation: one fresh subprocess per size (see --no-isolate). A single
    # size, or an explicit --no-isolate, runs inline.
    if len(args.sizes) > 1 and not args.no_isolate:
        _run_isolated(args)
        return

    predictor, calc = build_predictor(args.model, args.workers, args.sett)

    print(
        f"=== fairchem-native GP NVT — model={args.model} sett={args.sett} "
        f"world={args.workers} mem={args.mem_method} ==="
    )
    rows: list[dict] = []
    for n in args.sizes:
        row = bench_size(n, args, predictor, calc)
        rows.append(row)
        if row["status"] == "ok":
            print(
                f"  n={row['n_atoms']:>6}  step_ms(p50)={row['step_ms_p50']:>9}  "
                f"peak_MiB/rank={row['peak_MiB_per_rank']:>10}"
            )

    _write_csv(args.csv, rows)
    _print_table(rows, args)


def _write_csv(path: str, rows: list[dict]) -> None:
    """Write the §7a-schema CSV."""
    import csv

    cols = [
        "model",
        "env_torch",
        "strategy",
        "world",
        "n_atoms",
        "step_ms_p50",
        "peak_MiB_per_rank",
        "status",
    ]
    _dir = os.path.dirname(path)
    if _dir:
        os.makedirs(_dir, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {len(rows)} rows -> {path}")


def _print_table(rows: list[dict], args: argparse.Namespace) -> None:
    """Human-readable summary table on stdout."""
    print()
    print(f"{'world':>6}{'n_atoms':>9}{'step_ms(p50)':>14}{'peak_MiB/rank':>16}{'status':>8}")
    print("-" * 53)
    for r in rows:
        print(
            f"{r['world']:>6}{r['n_atoms']:>9}"
            f"{(r['step_ms_p50'] or '—'):>14}{(r['peak_MiB_per_rank'] or '—'):>16}"
            f"{r['status']:>8}"
        )
    if args.mem_method == "smi":
        print(
            "\nNOTE: --mem-method smi reports nvidia-smi memory.used (includes the "
            "CUDA context), NOT max_memory_allocated. Use --mem-method patch for a "
            "peak directly comparable to §7a."
        )


if __name__ == "__main__":
    main()
