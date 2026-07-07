---
name: nvalchemi-reporting
description: >-
  How to add observability to nvalchemi dynamics and training workflows using
  ReportingOrchestrator, RichReporter, TensorBoardReporter, scalar extraction,
  custom reporter callbacks, and dynamics LoggingHook. Use when showing live
  progress, writing TensorBoard summaries, preserving dynamics CSV rows,
  adding rank-safe distributed reporting, previewing Rich dashboards, or
  deciding between logging and reporting for training or molecular dynamics
  runs.
---

# nvalchemi Reporting

## Overview

Use reporting for curated workflow summaries and dashboards. Use logging for
direct event records such as per-system dynamics rows. Link agents to
`docs/userguide/reporting.md`, `docs/userguide/training.md`,
`docs/userguide/dynamics.md`, and `docs/userguide/hooks.md` for full details.

```python
from nvalchemi.hooks import (
    ReportingOrchestrator,
    RichReporter,
    TensorBoardReporter,
)
from nvalchemi.dynamics.hooks import LoggingHook
```

---

## Choose The Layer

Use `ReportingOrchestrator` when the user wants progress summaries, live Rich
dashboards, TensorBoard scalar snapshots, rank reductions, or one observability
hook that works across training, dynamics, and custom hook-enabled workflows.
`ReportingOrchestrator` is the hook to register with `hooks=[...]` or
`register_hook(...)`; `RichReporter`, `TensorBoardReporter`, and custom
`Reporter` objects are sinks owned by that hook and are not registered directly.
Workflow engines enter and close hook context managers automatically during
`run()`, so user code should not wrap reporting hooks manually in normal cases.

Use `nvalchemi.dynamics.hooks.LoggingHook` when the user wants a durable
per-graph dynamics event stream. It computes dynamics observables such as
energy, `fmax`, temperature, status, and graph index, then writes one row per
system to CSV, TensorBoard, or a custom writer.

Do not reuse the dynamics `LoggingHook` as a training logger. For training,
prefer reporters unless the task explicitly requires a raw training-event log;
then implement a training-specific hook with the same hook protocol.

---

## Training Pattern

Attach the `ReportingOrchestrator` as a normal training hook. Pick stages by
enum name when the code already serializes hook specs or when avoiding imports
in config files. Use `AFTER_OPTIMIZER_STEP` for high-frequency loss and
learning-rate progress, and validation stages when summaries should align with
validation output.

```python
from nvalchemi.hooks import ReportingOrchestrator, RichReporter, TensorBoardReporter
from nvalchemi.training import CheckpointHook, TrainingStrategy

reporting = ReportingOrchestrator(
    [
        TensorBoardReporter("runs/example/tensorboard"),
        RichReporter(layout="training", refresh_per_second=2.0),
    ],
    stages={"AFTER_OPTIMIZER_STEP"},
    frequency=10,
)

strategy = TrainingStrategy(
    models=model,
    optimizer_configs=optimizer_config,
    loss_fn=loss_fn,
    hooks=[
        reporting,
        CheckpointHook("runs/example/checkpoints", epoch_interval=1),
    ],
    num_epochs=20,
)

strategy.run(train_loader)
```

Guidelines:

- Register reporting through `hooks=[reporting]`; `strategy.run(...)` enters and
  closes the reporting hook automatically.
- Keep checkpoints and reporting separate; reporters observe, checkpoint hooks
  preserve restart state.
- Use `RichReporter(layout="training")` for terminal monitoring during local or
  interactive runs.
- Use `TensorBoardReporter(...)` for durable scalar summaries and later review.
- Set `frequency` high enough to avoid excessive terminal refresh or file I/O on
  large distributed runs.
- Add custom scalar callbacks when a metric is not already exposed through loss,
  optimizer, scheduler, validation, or workflow context.

---

## Dynamics Pattern

For live summaries or TensorBoard dashboards, use `ReportingOrchestrator` with a
dynamics layout. The dynamics Rich layout asks the reporter to collect default
dynamics scalars such as energy, `fmax`, temperature, convergence fraction,
active count, graduated count, and status counts when available.

```python
from nvalchemi.dynamics import DynamicsStage, NVE
from nvalchemi.hooks import ReportingOrchestrator, RichReporter

reporting = ReportingOrchestrator(
    [RichReporter(layout="dynamics", refresh_per_second=2.0)],
    stages={DynamicsStage.AFTER_STEP},
    frequency=25,
)

dynamics = NVE(
    model=model,
    dt=0.5,
    n_steps=10_000,
    hooks=[reporting],
)

final_batch = dynamics.run(batch)
```

For a durable per-system trajectory-adjacent log, add `LoggingHook` instead of
or in addition to reporting:

```python
from nvalchemi.dynamics import DynamicsStage, NVTLangevin
from nvalchemi.dynamics.hooks import LoggingHook

logger = LoggingHook(
    backend="csv",
    log_path="runs/md/scalars.csv",
    frequency=100,
    stage=DynamicsStage.AFTER_STEP,
)

dynamics = NVTLangevin(
    model=model,
    dt=0.5,
    temperature=300.0,
    n_steps=50_000,
    hooks=[logger],
)

final_batch = dynamics.run(batch)
```

Guidelines:

- Register reporting or logging through `hooks=[...]`; dynamics `run(...)` enters
  and closes hook context managers automatically.
- Use reporting for dashboards and scalar summaries; use `SnapshotHook` or data
  sinks for full batch states and trajectories.
- Use `LoggingHook(backend="custom", writer_fn=...)` when rows should go to
  `loguru`, a database, or a user-owned writer.
- Give each distributed rank a unique `log_path` when using dynamics
  `LoggingHook`; it writes directly and does not coordinate file access.
- Attach observation hooks at `DynamicsStage.AFTER_STEP` unless the metric needs
  a more specific stage.

---

## Distributed Reporting

Reporters can be rank-gated. Defaults are conservative for terminal and file
outputs: rank zero writes or renders unless a reporter requires all ranks for a
collective reduction.

```python
reporting = ReportingOrchestrator(
    [
        RichReporter(
            layout="training",
            rank_reduction="mean",
            rank_zero_only=True,
        ),
        TensorBoardReporter("runs/ddp/tensorboard", rank_zero_only=True),
    ],
    stages={"AFTER_OPTIMIZER_STEP"},
    frequency=20,
)
```

Guidelines:

- Use `rank_reduction="mean"` for loss curves in DDP when every rank reports
  matching scalar keys.
- Ensure every rank reaches reporters that perform reductions; do not add local
  control flow that skips nonzero ranks before the collective.
- Use rank-specific paths such as `"runs/ddp/rank-{rank}"` only when every rank
  intentionally writes its own artifact.
- Keep Rich dashboards rank-zero-only for normal terminal runs.

---

## Custom Scalars

Pass `custom_scalars` to reporters when metrics are present in the context or
batch but are not part of the default extraction path. Return plain numbers or
scalar tensors; keep callbacks cheap because they run at reporting frequency.

```python
def grad_norm(ctx):
    total = 0.0
    for parameter in ctx.model.parameters():
        if parameter.grad is not None:
            total += float(parameter.grad.detach().norm().item())
    return total

reporting = ReportingOrchestrator(
    [
        RichReporter(custom_scalars={"optimizer/grad_norm": grad_norm}),
        TensorBoardReporter(
            "runs/example/tensorboard",
            custom_scalars={"optimizer/grad_norm": grad_norm},
        ),
    ],
    stages={"AFTER_OPTIMIZER_STEP"},
    frequency=25,
)
```

For dynamics `LoggingHook`, callbacks receive `DynamicsContext` and must return
either a per-graph tensor with shape `(B,)` or a scalar that can be broadcast to
all graphs:

```python
logger = LoggingHook(
    backend="csv",
    log_path="runs/md/scalars.csv",
    custom_scalars={
        "max_velocity": lambda ctx: ctx.batch.velocities.norm(dim=-1).max(),
    },
    frequency=100,
)
```

---

## Rich Dashboard Workflows

Use `RichReporter.preview(...)` when choosing or checking a dashboard surface
without running a real workflow:

```python
from nvalchemi.hooks import RichReporter

RichReporter.preview(layout="training", title="training preview")
RichReporter.preview(layout="dynamics", title="dynamics preview")
```

Use `BaseRichLayout` for custom table-plus-plot dashboards. Implement the
`RichLayout` protocol directly only when the output needs a custom Rich
renderable. Do not create a nested `rich.live.Live`; `RichReporter` owns the
console, lifecycle, refresh rate, history, and rank filtering.

---

## Agent Checklist

- Add observability by default for long-running examples, CLI scaffolds,
  fine-tuning scripts, DDP jobs, and dynamics simulations.
- Prefer `ReportingOrchestrator([RichReporter(...), TensorBoardReporter(...)])`
  for training progress.
- Add `LoggingHook(backend="csv", ...)` for dynamics runs that need analyzable
  per-system rows.
- Register `ReportingOrchestrator` as the hook; let `TrainingStrategy.run()`
  and dynamics `run()` enter and close hook context managers automatically.
- Do not ask users to wrap reporters or dynamics `LoggingHook` manually unless
  they are invoking hook calls outside a workflow engine.
- Avoid expensive scalar callbacks and `.item()` calls inside hot training code
  unless they run at a controlled reporting frequency.
- Mention generated output paths in examples and tests so users know where to
  inspect CSV, TensorBoard, checkpoints, and dashboards.
