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
"""Compare serial and batched atomistic NEB workflows."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from nvalchemi._typing import AtomCategory
from nvalchemi.data import AtomicData, Batch
from nvalchemi.dynamics.hooks import FreezeAtomsHook
from nvalchemi.dynamics.paths.interpolate import interpolate_paths
from nvalchemi.dynamics.paths.neb.neb import NEB, ClimbingImageConfig

CASE_ROOT = Path(__file__).parent / "cases"
FIXTURES = ("cu-vacancy", "al100-au")
MACE_MODEL = "small-0b"
MACE_SHA = "7e3a0abcaf41e03a80e69f778e1b11b29de1cca704783dc25917a736392f8cf0"
AFTER_REGULAR_EXIT_STATUS = 2


def _cuda_sync(device: torch.device) -> None:
    """Synchronize CUDA work before reading timing or result tensors."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _load_case(
    path: Path, device: torch.device, dtype: torch.dtype
) -> tuple[AtomicData, AtomicData]:
    """Load and validate one paired NEB endpoint case."""
    case = json.loads(path.read_text())
    if case.get("name") != path.stem:
        raise ValueError(f"Case name in {path} must match its filename")
    if case.get("units") != {"cell": "angstrom", "positions": "angstrom"}:
        raise ValueError(f"Case {path} must use angstrom units")

    atomic_numbers = torch.tensor(case["atomic_numbers"], dtype=torch.long)
    initial_positions = torch.tensor(case["initial_positions"], dtype=dtype)
    final_positions = torch.tensor(case["final_positions"], dtype=dtype)
    cell = torch.tensor(case["cell"], dtype=dtype)
    pbc = torch.tensor(case["pbc"], dtype=torch.bool)
    num_atoms = atomic_numbers.numel()

    expected_positions_shape = (num_atoms, 3)
    if initial_positions.shape != expected_positions_shape:
        raise ValueError(
            f"Initial positions in {path} have shape {tuple(initial_positions.shape)}; "
            f"expected {expected_positions_shape}"
        )
    if final_positions.shape != expected_positions_shape:
        raise ValueError(
            f"Final positions in {path} have shape {tuple(final_positions.shape)}; "
            f"expected {expected_positions_shape}"
        )
    if cell.shape != (3, 3):
        raise ValueError(f"Cell in {path} must have shape (3, 3)")
    if pbc.shape != (3,):
        raise ValueError(f"PBC in {path} must have shape (3,)")
    if not torch.isfinite(initial_positions).all():
        raise ValueError(f"Initial positions in {path} must be finite")
    if not torch.isfinite(final_positions).all():
        raise ValueError(f"Final positions in {path} must be finite")
    if not torch.isfinite(cell).all():
        raise ValueError(f"Cell in {path} must be finite")

    fixed_atom_indices = case.get("fixed_atom_indices", [])
    if any(
        isinstance(index, bool) or not isinstance(index, int)
        for index in fixed_atom_indices
    ):
        raise ValueError(f"Fixed atom indices in {path} must be integers")
    if len(fixed_atom_indices) != len(set(fixed_atom_indices)):
        raise ValueError(f"Fixed atom indices in {path} must be unique")
    if any(index < 0 or index >= num_atoms for index in fixed_atom_indices):
        raise ValueError(f"Fixed atom index out of range in {path}")

    atom_categories = torch.full((num_atoms,), AtomCategory.GAS.value, dtype=torch.long)
    atom_categories[fixed_atom_indices] = AtomCategory.SPECIAL.value

    def endpoint(positions: torch.Tensor) -> AtomicData:
        return AtomicData(
            atomic_numbers=atomic_numbers.to(device),
            positions=positions.to(device),
            atom_categories=atom_categories.to(device),
            cell=cell.to(device).reshape(1, 3, 3),
            pbc=pbc.to(device).reshape(1, 3),
        )

    return endpoint(initial_positions), endpoint(final_positions)


def _build_paths(
    fixtures: tuple[str, ...],
    images: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Batch:
    """Build linearly interpolated paths for the requested fixtures."""
    cases = [
        _load_case(CASE_ROOT / f"{fixture}.json", device, dtype) for fixture in fixtures
    ]
    initial = [case[0] for case in cases]
    final = [case[1] for case in cases]
    bands = interpolate_paths(
        Batch.from_data_list(initial, device=device),
        Batch.from_data_list(final, device=device),
        images,
    )
    bands["velocities"] = torch.zeros_like(bands.positions)
    return bands


def _load_model(device: torch.device, dtype: torch.dtype, compile_model: bool) -> Any:
    """Load and verify the pinned potential used by both reaction paths."""
    from mace.calculators.foundations_models import download_mace_mp_checkpoint

    from nvalchemi.models.mace import MACEWrapper

    checkpoint = Path(download_mace_mp_checkpoint(MACE_MODEL))
    actual = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if actual != MACE_SHA:
        raise RuntimeError(f"MACE checkpoint hash mismatch: {actual} != {MACE_SHA}")
    return MACEWrapper.from_checkpoint(
        str(checkpoint),
        dtype=dtype,
        device=device,
        compile_model=compile_model,
    ).eval()


def _path_graph_bounds(batch: Batch, path_index: int) -> tuple[int, int]:
    """Return the half-open graph range for one grouped path."""
    group_ptr = batch.group_layout.group_ptr
    return int(group_ptr[path_index]), int(group_ptr[path_index + 1])


def _path_positions(batch: Batch, path_index: int) -> torch.Tensor:
    """Return all image positions for one path as a dense tensor."""
    graph_start, graph_stop = _path_graph_bounds(batch, path_index)
    node_start = int(batch.batch_ptr[graph_start])
    node_stop = int(batch.batch_ptr[graph_stop])
    return batch.positions[node_start:node_stop].reshape(
        graph_stop - graph_start, -1, 3
    )


def _path_results(
    batch: Batch,
    fixtures: tuple[str, ...],
    effective_forces: torch.Tensor,
    fmax: float,
) -> list[dict[str, Any]]:
    """Summarize completion, convergence, energy, and force for each path."""
    results = []
    status = batch.status.reshape(-1)[: batch.num_graphs]
    for path_index, fixture in enumerate(fixtures):
        start, stop = _path_graph_bounds(batch, path_index)
        energies = batch.energy[start:stop, 0].detach().cpu().double().numpy()
        highest = int(np.argmax(energies))
        node_start = int(batch.batch_ptr[start])
        node_stop = int(batch.batch_ptr[stop])
        force_norms = torch.linalg.vector_norm(
            effective_forces[node_start:node_stop], dim=-1
        )
        movable = (
            batch.atom_categories[node_start:node_stop] != AtomCategory.SPECIAL.value
        )
        force_norms = force_norms[movable]
        max_effective_force = float(force_norms.max())
        results.append(
            {
                "fixture": fixture,
                "completed": bool(
                    torch.all(status[start:stop] == AFTER_REGULAR_EXIT_STATUS)
                ),
                "converged": max_effective_force <= fmax,
                "image_energies": energies.tolist(),
                "highest_energy_image_index": highest,
                "forward_barrier": float(energies[highest] - energies[0]),
                "reverse_barrier": float(energies[highest] - energies[-1]),
                "max_effective_force": max_effective_force,
                "fmax_threshold": fmax,
            }
        )
    return results


def _run_workflow(
    fixtures: tuple[str, ...],
    model: Any,
    args: argparse.Namespace,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[dict[str, Any], Batch]:
    """Run one complete staged NEB workflow and return its summary and bands."""
    bands = _build_paths(fixtures, args.images, device, dtype)
    calls = [0]
    handle = model.register_forward_hook(
        lambda _module, _inputs, _output: calls.__setitem__(0, calls[0] + 1)
    )
    neb = NEB(
        model=model,
        spring=args.spring,
        fmax=args.fmax,
        climbing=ClimbingImageConfig(
            max_regular_steps=args.max_regular_steps,
            max_climbing_steps=args.max_climbing_steps,
        ),
        n_steps=(args.max_regular_steps + args.max_climbing_steps + 1),
        # Accounting for one reprime-only iteration when entering climbing-image NEB
        optimizer_kwargs={"dt": args.dt},
        extra_hooks=[FreezeAtomsHook()],
        compile=args.compile_workflow,
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    _cuda_sync(device)
    started = time.perf_counter()
    try:
        bands = neb.run(bands)
        _cuda_sync(device)
    finally:
        handle.remove()
    elapsed = time.perf_counter() - started
    # NEB primes forces once, then makes one model call per coordinate update.
    steps = max(calls[0] - 1, 0)
    paths = _path_results(bands, fixtures, bands.forces, args.fmax)
    summary = {
        "fixtures": list(fixtures),
        "completed": all(path["completed"] for path in paths),
        "converged": all(path["converged"] for path in paths),
        "steps": steps,
        "model_calls": calls[0],
        "image_evaluations": calls[0] * bands.num_graphs,
        "elapsed_seconds": elapsed,
        "seconds_per_step": elapsed / max(steps, 1),
        "image_evaluations_per_second": (
            calls[0] * bands.num_graphs / max(elapsed, 1.0e-12)
        ),
        "peak_cuda_memory_bytes": (
            torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
        ),
        "paths": paths,
    }
    return summary, bands


def _compare_results(
    standalone: dict[str, tuple[dict[str, Any], Batch]],
    grouped_summary: dict[str, Any],
    grouped_bands: Batch,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    """Compare each grouped path with the corresponding standalone run."""
    comparisons = []
    for path_index, fixture in enumerate(FIXTURES):
        standalone_summary, standalone_bands = standalone[fixture]
        expected = standalone_summary["paths"][0]
        actual = grouped_summary["paths"][path_index]
        energy_difference = float(
            np.max(
                np.abs(
                    np.asarray(actual["image_energies"])
                    - np.asarray(expected["image_energies"])
                )
            )
        )
        position_difference = float(
            torch.max(
                torch.abs(
                    _path_positions(grouped_bands, path_index)
                    - _path_positions(standalone_bands, 0)
                )
            )
        )
        comparisons.append(
            {
                "fixture": fixture,
                "max_abs_energy_difference": energy_difference,
                "max_abs_position_difference": position_difference,
                "forward_barrier_difference": float(
                    actual["forward_barrier"] - expected["forward_barrier"]
                ),
                "equivalent": (
                    energy_difference <= args.energy_atol
                    and position_difference <= args.position_atol
                ),
            }
        )
    return comparisons


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Run serial and batched NEB workflows for two paths and compare their results.

    Parameters
    ----------
    args
        Parsed benchmark arguments.

    Returns
    -------
    dict[str, Any]
        JSON-serializable benchmark configuration and results.

    Raises
    ------
    FileNotFoundError
        If an input case or model checkpoint is unavailable.
    RuntimeError
        If CUDA is requested but unavailable or model validation fails.
    ValueError
        If an input case is invalid.
    """
    device, dtype = torch.device(args.device), getattr(torch, args.dtype)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    model = _load_model(device, dtype, args.compile_model)
    if args.warmup:
        warmup_args = argparse.Namespace(**vars(args))
        warmup_args.fmax = float("inf")
        warmup_args.max_regular_steps = 1
        warmup_args.max_climbing_steps = 1
        for fixtures in ((FIXTURES[0],), (FIXTURES[1],), FIXTURES):
            _run_workflow(fixtures, model, warmup_args, device, dtype)

    # Run 2 paths in serial as well as in group, then compare results
    standalone: dict[str, tuple[dict[str, Any], Batch]] = {}
    for fixture in FIXTURES:
        standalone[fixture] = _run_workflow((fixture,), model, args, device, dtype)
    grouped_summary, grouped_bands = _run_workflow(FIXTURES, model, args, device, dtype)
    comparisons = _compare_results(standalone, grouped_summary, grouped_bands, args)
    standalone_summaries = {
        fixture: summary for fixture, (summary, _bands) in standalone.items()
    }
    standalone_time = sum(
        summary["elapsed_seconds"] for summary in standalone_summaries.values()
    )
    standalone_calls = sum(
        summary["model_calls"] for summary in standalone_summaries.values()
    )

    return {
        "success": (
            all(
                summary["completed"] and summary["converged"]
                for summary in standalone_summaries.values()
            )
            and grouped_summary["completed"]
            and grouped_summary["converged"]
            and all(comparison["equivalent"] for comparison in comparisons)
        ),
        "configuration": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": str(device),
        },
        "model": {"id": MACE_MODEL, "sha256": MACE_SHA},
        "standalone": standalone_summaries,
        "grouped": grouped_summary,
        "equivalence": comparisons,
        "performance": {
            "standalone_elapsed_seconds": standalone_time,
            "grouped_elapsed_seconds": grouped_summary["elapsed_seconds"],
            "grouped_speedup": standalone_time
            / max(grouped_summary["elapsed_seconds"], 1.0e-12),
            "standalone_model_calls": standalone_calls,
            "grouped_model_calls": grouped_summary["model_calls"],
        },
    }


def parser() -> argparse.ArgumentParser:
    """Build the command-line parser.

    Returns
    -------
    argparse.ArgumentParser
        Parser configured with the benchmark options.
    """
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--images", type=int, choices=(5, 9), default=5)
    result.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    result.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    result.add_argument("--dt", type=float, default=0.01)
    result.add_argument("--fmax", type=float, default=0.002)
    result.add_argument("--spring", type=float, default=0.1)
    result.add_argument("--max-regular-steps", type=int, default=5000)
    result.add_argument("--max-climbing-steps", type=int, default=5000)
    result.add_argument("--energy-atol", type=float, default=5.0e-5)
    result.add_argument("--position-atol", type=float, default=1.0e-5)
    result.add_argument(
        "--compile-model", action=argparse.BooleanOptionalAction, default=False
    )
    result.add_argument(
        "--compile-workflow", action=argparse.BooleanOptionalAction, default=False
    )
    result.add_argument("--warmup", action=argparse.BooleanOptionalAction, default=True)
    result.add_argument("--output", type=Path)
    return result


def main(argv: list[str] | None = None) -> None:
    """Run the benchmark CLI.

    Parameters
    ----------
    argv
        Command-line arguments, or ``None`` to read from :data:`sys.argv`.
    """
    args = parser().parse_args(argv)
    summary = run(args)
    encoded = json.dumps(summary, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n")
        print(f"wrote {args.output}")
    else:
        print(encoded)
    if not summary["success"]:
        print(
            "[FAIL] Batched NEB benchmark did not satisfy all completion, "
            "convergence, and equivalence checks.",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
