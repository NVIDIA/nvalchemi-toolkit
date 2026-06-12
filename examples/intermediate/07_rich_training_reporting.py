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
"""
Rich Training Reporting
=======================

This example drives the Rich reporting dashboard with synthetic training
metrics. The scalar values are deterministic and intentionally lightweight; the
goal is to demonstrate the live terminal UI without requiring a real model,
dataset, or training strategy.

Run it directly from the repository root to watch the dashboard refresh:

.. code-block:: bash

   uv run python examples/intermediate/07_rich_training_reporting.py --steps 80 --delay 0.05
"""

from __future__ import annotations

import argparse
import math
import time
from collections.abc import Sequence
from enum import Enum, auto
from types import SimpleNamespace

import torch

from nvalchemi.hooks import ReportingOrchestrator, RichReporter, TrainContext


class SyntheticTrainingStage(Enum):
    """Minimal training-like hook stage enum for this reporting demo."""

    AFTER_OPTIMIZER_STEP = auto()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the Rich reporting demo."""
    parser = argparse.ArgumentParser(
        description="Preview the Rich training reporter with synthetic metrics.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=24,
        help="Number of synthetic reporting steps to emit.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of synthetic epochs represented in the progress panel.",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.03,
        help="Seconds to sleep between dashboard refreshes.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1.0e-3,
        help="Initial optimizer learning rate shown in the dashboard.",
    )
    parser.add_argument(
        "--refresh-per-second",
        type=float,
        default=8.0,
        help="Rich Live refresh rate.",
    )
    parser.add_argument(
        "--final-delay",
        type=float,
        default=0.0,
        help="Seconds to keep the final dashboard visible before exit.",
    )
    return parser.parse_args(argv)


def synthetic_losses(step: int, total_steps: int) -> dict[str, float]:
    """Return deterministic loss values for one synthetic training step."""
    progress = step / max(total_steps, 1)
    energy = 0.70 * math.exp(-3.0 * progress) + 0.04
    forces = 1.10 * math.exp(-2.1 * progress) + 0.08
    ripple = 0.015 * math.sin(step / 2.5)
    validation = 0.55 * math.exp(-2.4 * progress) + 0.06 + abs(ripple)
    total = 0.25 * energy + 0.75 * forces + ripple
    return {
        "total": max(total, 0.0),
        "energy": max(energy, 0.0),
        "forces": max(forces, 0.0),
        "validation": max(validation, 0.0),
    }


def build_context(
    *,
    step: int,
    total_steps: int,
    epochs: int,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    workflow: SimpleNamespace,
) -> TrainContext:
    """Build a training hook context populated with synthetic metrics."""
    losses = synthetic_losses(step, total_steps)
    steps_per_epoch = max(math.ceil(total_steps / max(epochs, 1)), 1)
    epoch = min((step - 1) // steps_per_epoch, max(epochs - 1, 0))
    epoch_step = step - epoch * steps_per_epoch

    return TrainContext(
        batch=None,
        global_rank=0,
        workflow=workflow,
        step_count=step,
        batch_count=step,
        epoch_step_count=epoch_step,
        epoch=epoch,
        loss=torch.tensor(losses["total"]),
        losses={
            "total_loss": torch.tensor(losses["total"]),
            "validation": torch.tensor(losses["validation"]),
            "per_component_unweighted": {
                "energy": torch.tensor(losses["energy"]),
                "forces": torch.tensor(losses["forces"]),
            },
            "per_component_weight": {
                "energy": torch.tensor(0.25),
                "forces": torch.tensor(0.75),
            },
            "per_component_raw_weight": {
                "energy": torch.tensor(1.0),
                "forces": torch.tensor(3.0),
            },
        },
        optimizers=[optimizer],
        lr_schedulers=[scheduler],
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run the synthetic Rich reporting demo."""
    args = parse_args(argv)
    if args.steps < 1:
        raise ValueError("--steps must be at least 1.")
    if args.epochs < 1:
        raise ValueError("--epochs must be at least 1.")
    if args.delay < 0:
        raise ValueError("--delay must be non-negative.")
    if args.final_delay < 0:
        raise ValueError("--final-delay must be non-negative.")

    parameter = torch.nn.Parameter(torch.tensor(0.0))
    optimizer = torch.optim.AdamW([parameter], lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(args.steps, 1),
        eta_min=args.lr * 0.08,
    )
    workflow = SimpleNamespace(num_steps=args.steps, num_epochs=args.epochs)
    stage = SyntheticTrainingStage.AFTER_OPTIMIZER_STEP
    reporter = RichReporter(
        title="nvalchemi synthetic training",
        layout="training",
        max_scalars=12,
        max_plots=4,
        plot_height=6,
        plot_keys=(
            "loss/total",
            "loss/validation",
            "loss/energy/unweighted",
            "loss/forces/unweighted",
            "scheduler/lr",
        ),
        refresh_per_second=args.refresh_per_second,
        transient=False,
    )
    reporting = ReportingOrchestrator([reporter], stages={stage}, rank_zero_only=True)

    with reporting:
        for step in range(1, args.steps + 1):
            losses = synthetic_losses(step, args.steps)
            parameter.grad = torch.tensor(losses["total"])
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

            ctx = build_context(
                step=step,
                total_steps=args.steps,
                epochs=args.epochs,
                optimizer=optimizer,
                scheduler=scheduler,
                workflow=workflow,
            )
            if step == 1:
                reporting.state.add_message(
                    "info",
                    "synthetic warmup finished",
                    ctx=ctx,
                    stage=stage,
                )
            elif step == math.ceil(args.steps * 0.55):
                reporting.state.add_message(
                    "info",
                    "validation curve refreshed",
                    ctx=ctx,
                    stage=stage,
                )
            reporting(ctx, stage)
            time.sleep(args.delay)

        if args.final_delay:
            time.sleep(args.final_delay)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
