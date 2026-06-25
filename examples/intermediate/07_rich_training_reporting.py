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

This example shows how the hook-native reporting API turns training state into
Rich terminal dashboards. It uses synthetic losses instead of a real model so
that the moving parts are visible: a workflow emits a
:class:`~nvalchemi.hooks.TrainContext`, a
:class:`~nvalchemi.hooks.ReportingOrchestrator` decides when reporting runs, and
:class:`~nvalchemi.hooks.RichReporter` renders the scalar snapshot.

The same pattern applies inside :class:`~nvalchemi.training.TrainingStrategy`:
register the reporting orchestrator as a hook, choose the stages to observe,
and configure the reporter with the scalar keys and plots that matter for your
workflow.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import torch
from rich.console import Console

from nvalchemi.hooks import ReportingOrchestrator, RichReporter, TrainContext
from nvalchemi.training import TrainingStage

# %%
# Configure a small gallery run
# -----------------------------
# Sphinx-gallery examples should execute quickly and deterministically. These
# constants replace command-line arguments so reviewers can read the complete
# configuration in one place. For an interactive terminal demo, increase
# ``NUM_STEPS`` and set ``TRANSIENT_DASHBOARD = False``.

NUM_STEPS = 12
NUM_EPOCHS = 3
INITIAL_LR = 1.0e-3
TRANSIENT_DASHBOARD = True


# %%
# Build synthetic metrics
# -----------------------
# Reporters consume the same ``TrainContext`` object that normal training hooks
# receive. The helper below fabricates the pieces that scalar extraction knows
# how to read: ``ctx.loss`` for the headline loss, ``ctx.losses`` for named loss
# components, and optimizer/scheduler objects for learning-rate reporting.


def synthetic_losses(step: int, total_steps: int) -> dict[str, float]:
    """Return deterministic training and validation losses for one step.

    Parameters
    ----------
    step : int
        One-indexed optimizer step.
    total_steps : int
        Total number of synthetic optimizer steps in this example.

    Returns
    -------
    dict[str, float]
        Loss components that look like a small training run converging.
    """
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
    """Build the hook context that a training workflow would normally provide.

    Parameters
    ----------
    step : int
        One-indexed optimizer step.
    total_steps : int
        Total number of synthetic optimizer steps.
    epochs : int
        Number of synthetic epochs represented in progress metadata.
    optimizer : torch.optim.Optimizer
        Optimizer whose learning rate will be reported.
    scheduler : torch.optim.lr_scheduler.LRScheduler
        Scheduler included so reporter output can show LR evolution.
    workflow : SimpleNamespace
        Minimal workflow-like object with ``num_steps`` and ``num_epochs``.

    Returns
    -------
    TrainContext
        Hook context populated with training-style scalar state.
    """
    losses = synthetic_losses(step, total_steps)
    steps_per_epoch = max(math.ceil(total_steps / max(epochs, 1)), 1)
    epoch = min((step - 1) // steps_per_epoch, max(epochs - 1, 0))
    epoch_step = step - epoch * steps_per_epoch

    # ``losses`` mirrors the structure emitted by composed nvalchemi losses.
    # Reporter scalar extraction turns these nested tensors into keys like
    # ``loss/energy/unweighted`` and ``loss/forces/weight``.
    loss_components = {
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
    }

    return TrainContext(
        batch=None,
        global_rank=0,
        workflow=workflow,
        step_count=step,
        batch_count=step,
        epoch_step_count=epoch_step,
        epoch=epoch,
        loss=torch.tensor(losses["total"]),
        losses=loss_components,
        optimizers=[optimizer],
        lr_schedulers=[scheduler],
    )


# %%
# Create the reporter stack
# -------------------------
# ``ReportingOrchestrator`` is the hook registered with a workflow. It owns one
# or more reporters and decides whether they should run on a given hook stage.
# Here we observe ``AFTER_OPTIMIZER_STEP`` because losses and learning rates have
# just been updated, which is the common choice for training dashboards.

stage = TrainingStage.AFTER_OPTIMIZER_STEP
workflow = SimpleNamespace(num_steps=NUM_STEPS, num_epochs=NUM_EPOCHS)

# A real training loop would already have an optimizer and scheduler. They are
# included here only so the reporter can demonstrate LR scalar extraction.
parameter = torch.nn.Parameter(torch.tensor(0.0))
optimizer = torch.optim.AdamW([parameter], lr=INITIAL_LR)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=NUM_STEPS,
    eta_min=INITIAL_LR * 0.08,
)

# The console is recorded so Sphinx-gallery can capture text output without
# depending on a user's terminal dimensions or alternate-screen behavior.
console = Console(width=100, record=True)
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
    console=console,
    refresh_per_second=8.0,
    transient=TRANSIENT_DASHBOARD,
)

# ``rank_zero_only=True`` is suitable for local printing. Reporters that perform
# cross-rank reductions, such as ``RichReporter(rank_reduction="mean")``, mark
# themselves as requiring all ranks so the orchestrator does not skip collectives.
reporting = ReportingOrchestrator(
    [reporter],
    stages={stage},
    rank_zero_only=True,
)

# %%
# Attach it to a real strategy
# ----------------------------
# In a real training script, the object above is passed directly as a hook:
#
# .. code-block:: python
#
#    strategy = TrainingStrategy(
#        ...,
#        hooks=[
#            ReportingOrchestrator(
#                [RichReporter(layout="training", rank_reduction="mean")],
#                stages={TrainingStage.AFTER_OPTIMIZER_STEP},
#                rank_zero_only=True,
#            )
#        ],
#    )
#
# ``rank_reduction="mean"`` asks the reporter abstraction to reduce scalars
# across ranks before rendering, so user code does not need raw
# ``torch.distributed`` reduction calls for reporting.


# %%
# Emit reporting events
# ---------------------
# In normal use, ``TrainingStrategy`` calls hooks for you. This short loop does
# the same thing by hand: update optimizer state, build a context snapshot, add
# optional messages, then invoke the reporting hook with ``reporting(ctx, stage)``.

with reporting:
    for step in range(1, NUM_STEPS + 1):
        losses = synthetic_losses(step, NUM_STEPS)

        # The optimizer step is intentionally simple; it just gives the reporter
        # real optimizer state from which to read the current learning rate.
        parameter.grad = torch.tensor(losses["total"])
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()

        ctx = build_context(
            step=step,
            total_steps=NUM_STEPS,
            epochs=NUM_EPOCHS,
            optimizer=optimizer,
            scheduler=scheduler,
            workflow=workflow,
        )

        # Messages are optional annotations attached to the shared reporting
        # state. They are useful for events such as warmup completion,
        # validation refreshes, checkpoint saves, or early-stopping decisions.
        if step == 1:
            reporting.state.add_message(
                "info",
                "synthetic warmup finished",
                ctx=ctx,
                stage=stage,
            )
        elif step == math.ceil(NUM_STEPS * 0.55):
            reporting.state.add_message(
                "info",
                "validation curve refreshed",
                ctx=ctx,
                stage=stage,
            )

        reporting(ctx, stage)


# %%
# Inspect what the reporter retained
# ----------------------------------
# The Rich dashboard is meant for a terminal, but its retained history is still
# easy to inspect in docs or tests. These are the scalar series available for
# plotting after the synthetic run.

print("reported scalar series:")
for key in sorted(reporter.history):
    print(f"- {key}: {len(reporter.history[key])} points")
