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
"""Validate NEB against the canonical Muller--Brown transition state."""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

import torch

from benchmark.neb._mueller_brown_model import (
    MULLER_BROWN_FINAL,
    MULLER_BROWN_INITIAL,
    MULLER_BROWN_TRANSITION_STATE,
    MULLER_BROWN_TRANSITION_STATE_ENERGY,
    MullerBrownModel,
)
from nvalchemi.data import AtomicData, Batch
from nvalchemi.dynamics.paths.interpolate import interpolate_paths
from nvalchemi.dynamics.paths.neb.neb import NEB, ClimbingImageConfig

AFTER_REGULAR_EXIT_STATUS = 2


def _cuda_sync(device: torch.device) -> None:
    """Synchronize CUDA work before reading timing or result tensors."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Check the converged climbing image against the canonical saddle."""
    device, dtype = torch.device(args.device), getattr(torch, args.dtype)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    model = MullerBrownModel(device, dtype).eval()

    # Load endpoints and perform linear interpolation
    endpoints = [
        AtomicData(
            atomic_numbers=torch.tensor([2], dtype=torch.long, device=device),
            positions=torch.tensor([[*xy, 0.0]], dtype=dtype, device=device),
        )
        for xy in (MULLER_BROWN_INITIAL, MULLER_BROWN_FINAL)
    ]
    bands = interpolate_paths(
        Batch.from_data_list([endpoints[0]], device=device),
        Batch.from_data_list([endpoints[1]], device=device),
        7,
    )
    # Initialize velocities required by the optimizer
    bands["velocities"] = torch.zeros_like(bands.positions)

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
        compile=args.compile_workflow,
    )
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

    energies = bands.energy[:, 0]
    highest = int(torch.argmax(energies))
    saddle = bands.positions[highest, :2]
    expected_position = torch.tensor(
        MULLER_BROWN_TRANSITION_STATE, device=device, dtype=dtype
    )
    position_error = float(torch.linalg.vector_norm(saddle - expected_position))
    energy_error = abs(float(energies[highest]) - MULLER_BROWN_TRANSITION_STATE_ENERGY)
    status = bands.status.reshape(-1)[: bands.num_graphs]
    completed = bool(torch.all(status == AFTER_REGULAR_EXIT_STATUS))
    max_effective_force = float(torch.linalg.vector_norm(bands.forces, dim=-1).max())
    converged = max_effective_force <= args.fmax
    accurate = position_error <= args.position_atol and energy_error <= args.energy_atol
    return {
        "success": completed and converged and accurate,
        "configuration": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": str(device),
        },
        "completed": completed,
        "converged": converged,
        "accurate": accurate,
        "images": bands.num_graphs,
        "steps": steps,
        "model_calls": calls[0],
        "elapsed_seconds": elapsed,
        "highest_energy_image_index": highest,
        "saddle_position": saddle.detach().cpu().double().tolist(),
        "expected_saddle_position": list(MULLER_BROWN_TRANSITION_STATE),
        "saddle_position_error": position_error,
        "saddle_energy": float(energies[highest]),
        "expected_saddle_energy": MULLER_BROWN_TRANSITION_STATE_ENERGY,
        "saddle_energy_error": energy_error,
        "max_effective_force": max_effective_force,
        "fmax_threshold": args.fmax,
    }


def parser() -> argparse.ArgumentParser:
    """Build the command-line parser.

    Returns
    -------
    argparse.ArgumentParser
        Parser configured with the benchmark options.
    """
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    result.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    result.add_argument("--spring", type=float, default=0.1)
    result.add_argument("--position-atol", type=float, default=1.0e-3)
    result.add_argument("--energy-atol", type=float, default=1.0e-3)
    result.add_argument("--fmax", type=float, default=0.002)
    result.add_argument("--dt", type=float, default=0.1)
    result.add_argument("--max-regular-steps", type=int, default=5000)
    result.add_argument("--max-climbing-steps", type=int, default=5000)
    result.add_argument(
        "--compile-workflow", action=argparse.BooleanOptionalAction, default=False
    )
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
            "[FAIL] Muller--Brown NEB benchmark did not satisfy all "
            "completion, convergence, and accuracy checks.",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
