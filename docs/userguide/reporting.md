<!-- markdownlint-disable MD014 -->

(reporting_guide)=

# Reporting

Reporting is the higher-level observability layer for hook-enabled workflows.
It collects scalar summaries from hook contexts, tracks reporting metadata,
optionally reduces values across ranks, and sends the resulting snapshots to
reporting sinks such as JSONL files, TensorBoard, or live Rich dashboards.

## Reporting vs. logging

Logging and reporting have different intent:

| Use this | When you want |
|----------|----------------|
| Logging | Workflow event records: rows, files, or backend writes that preserve a direct stream of events. |
| Reporting | Curated workflow summaries: scalar snapshots, rank-safe reductions, previews, dashboards, and analysis-facing output. |

Logging is not inherently dynamics-specific. A training workflow can also have a
logger when it needs a direct record of training events, optimizer steps,
gradient statistics, or validation passes. The current built-in
{py:class}`~nvalchemi.dynamics.hooks.LoggingHook` is dynamics-focused because it
computes per-graph observables such as energy, `fmax`, and temperature and writes
one row per system. A future training logger should be a separate
training-specific implementation rather than overloading the dynamics hook with a
different event model.

Reporters sit one level up. They receive the current hook context and shared
reporting state, collect scalar metrics, decide whether to reduce across ranks,
and then render or serialize a summary. A reporter may intentionally discard
low-level detail if the output is meant to be a compact dashboard or analysis
record.

Backends do not define the layer. CSV, JSONL, TensorBoard, W&B, and MLflow can be
used for logging or reporting depending on what is being written. In this
package, {py:class}`~nvalchemi.hooks.JSONLReporter` and
{py:class}`~nvalchemi.hooks.TensorBoardReporter` are reporters because they write
{py:class}`~nvalchemi.hooks.ScalarSnapshot` payloads collected by
{py:class}`~nvalchemi.hooks.ReportingOrchestrator`. By contrast, the dynamics
`LoggingHook` TensorBoard backend is logging because it writes the hook's raw
per-graph dynamics rows directly.

## Basic usage

{py:class}`~nvalchemi.hooks.ReportingOrchestrator` is the hook that fans events
out to reporters:

```python
from nvalchemi.hooks import JSONLReporter, ReportingOrchestrator, RichReporter

reporting = ReportingOrchestrator(
    [
        JSONLReporter("metrics.jsonl"),
        RichReporter(),
    ],
    stages={"AFTER_OPTIMIZER_STEP"},
    frequency=10,
)
```

`RichReporter()` defaults to automatic layout selection. It chooses the first
built-in layout that matches the first reported context and keeps that choice
for the workflow run. Pin a layout when you want a specific dashboard surface:

```python
from nvalchemi.hooks import ReportingOrchestrator, RichReporter

reporting = ReportingOrchestrator(
    [RichReporter(layout="dynamics", refresh_per_second=2.0)],
    stages={"AFTER_STEP"},
)
```

You can preview a Rich layout without running a workflow:

```python
from nvalchemi.hooks import RichReporter

RichReporter.preview(layout="dynamics", title="dynamics preview")
```

## What happens under the hood

The reporting path has two boundaries: workflow engines emit hook contexts, and
reporters decide how to turn those contexts into an output artifact.

```{graphviz}
digraph reporting_orchestrator {
  graph [rankdir=LR, bgcolor="transparent"];
  node [
    shape=box,
    style="rounded,filled",
    fillcolor="#F8F9FA",
    color="#5C677D",
    fontname="Helvetica"
  ];
  edge [color="#5C677D", fontname="Helvetica"];

  workflow [label="Training, dynamics,\nor custom workflow"];
  context [label="HookContext\n+ stage enum"];
  orchestrator [label="ReportingOrchestrator"];
  state [label="ReportingState\n event metadata"];
  reporter [label="Reporter\n(JSONL, TensorBoard, Rich, ...)"];
  output [label="Output\nfile, run log, dashboard"];

  workflow -> context [label="engine hook call"];
  context -> orchestrator [label="stage and frequency match"];
  orchestrator -> state [label="mark_event"];
  orchestrator -> reporter [label="report(ctx, stage, state)"];
  reporter -> output [label="write or render"];
}
```

At each matching hook event, `ReportingOrchestrator`:

1. Updates a shared {py:class}`~nvalchemi.hooks.ReportingState`.
2. Skips rank-zero-only reporters on nonzero ranks.
3. Calls each reporter with `(ctx, stage, state)`.
4. Applies the configured error policy if a reporter raises.

Scalar reporters then call {py:func}`~nvalchemi.hooks.collect_scalars`. The
collector builds a {py:class}`~nvalchemi.hooks.ScalarSnapshot` containing:

- `stage`, timestamp, elapsed time, event count, step count, rank, optional
  training metadata, and recent reporter messages.
- A flat dictionary of scalar values, using slash-separated keys such as
  `loss/total`, `optimizer/lr`, `scheduler/lr`, `converged_fraction`, or
  `dynamics/graduated_count`.

Reporters can also request rank reductions. When enabled, every rank must call
the reporter with the same scalar keys, and only rank zero writes or renders the
reduced result.

```{graphviz}
digraph reporting_reduction {
  graph [rankdir=LR, bgcolor="transparent"];
  node [
    shape=box,
    style="rounded,filled",
    fillcolor="#F8F9FA",
    color="#5C677D",
    fontname="Helvetica"
  ];
  edge [color="#5C677D", fontname="Helvetica"];

  rank0 [label="rank 0\ncollect_scalars"];
  rank1 [label="rank 1\ncollect_scalars"];
  rankn [label="rank n\ncollect_scalars"];
  reduce [label="reduce_scalar_snapshot\nmean, sum, min, or max"];
  write [label="rank 0\nwrites or renders"];
  skip [label="nonzero ranks\nreturn after reduction"];

  rank0 -> reduce;
  rank1 -> reduce;
  rankn -> reduce;
  reduce -> write;
  reduce -> skip;
}
```

## Rich dashboards

{py:class}`~nvalchemi.hooks.RichReporter` owns the terminal dashboard mechanics:

- scalar collection and optional rank reduction,
- retained per-metric history,
- Rich `Live` lifecycle,
- automatic layout selection,
- static preview seeding,
- rank-zero-only rendering.

The selected layout owns the visual policy. Built-in layouts live under
`nvalchemi.hooks.reporting.layouts`:

```python
from nvalchemi.hooks.reporting.layouts.train import TrainingRichLayout
from nvalchemi.hooks.reporting.layouts.dynamics import DynamicsRichLayout
```

`layout="auto"` and `layout=None` defer layout selection until the first report.
`layout="training"` prioritizes loss curves, optimizer and scheduler learning
rates, step progress, throughput, ETA, and recent reporter messages.
`layout="dynamics"` prioritizes energy, `fmax`, temperature, convergence,
active/graduated counts, status counts, dynamics progress, throughput, ETA, and
recent reporter messages. The dynamics layout also requests default dynamics
scalar collection when it is selected.

Progress and ETA scalars are collected for Rich dashboards only. Durable
reporters keep their scalar snapshots stable unless you add the same values with
custom scalar callbacks.

## Custom Rich layouts

Rich layouts are plain Python objects. `RichReporter` passes the layout:

- the latest `ScalarSnapshot`, or `None` before the first report,
- retained scalar history as `dict[str, Sequence[tuple[int, float]]]`,
- display options such as title, precision, max rows, plot keys, and plot size.

The layout returns a Rich renderable, usually a {py:class}`rich.layout.Layout`.
It does not collect scalars, perform rank reduction, or manage `Live`.
Use `snapshot.scalars` for current values, `history` for curves, and
`snapshot.messages` for recent reporter messages or warnings. RichReporter also
adds workflow progress scalars when the context exposes enough metadata, such as
`training/progress_fraction`, `training/eta_s`, `dynamics/progress_fraction`,
and `dynamics/eta_s`.

### Subclass BaseRichLayout

For most dashboards, subclass {py:class}`~nvalchemi.hooks.BaseRichLayout`. This
keeps the standard header, latest-metric table, and plot panel. You only choose
metric priority, panel titles, and preview curves:

```python
from collections.abc import Mapping, Sequence

from nvalchemi.hooks import BaseRichLayout, RichReporter


class ValidationRichLayout(BaseRichLayout):
    def __init__(self) -> None:
        super().__init__(
            name="validation",
            preferred_plot_keys=("validation/loss", "validation/mae"),
            latest_title="Validation",
            history_title="Curves",
        )

    def default_preview_history(self) -> Mapping[str, Sequence[float]]:
        return {
            "validation/loss": (0.8, 0.62, 0.51, 0.44),
            "validation/mae": (0.31, 0.24, 0.19, 0.16),
        }


reporter = RichReporter(layout=ValidationRichLayout())
```

`BaseRichLayout` also provides preview metadata hooks. Override them when the
default training metadata is wrong for your workflow:

```python
class ValidationRichLayout(BaseRichLayout):
    ...

    def default_preview_stage(self) -> str:
        return "AFTER_VALIDATION"

    def default_preview_epoch(self) -> int | None:
        return None

    def default_preview_batch_count(self) -> int | None:
        return None
```

### Implement render directly

For a fully custom surface, implement {py:class}`~nvalchemi.hooks.RichLayout`
directly. This is useful when the dashboard is not a table plus plots.
Custom layouts compose normal Rich renderables, but they do so inside the
`RichReporter` lifecycle: the reporter owns the console, `Live`, rank filtering,
scalar collection, history retention, and refresh cadence. The layout should
remain a pure rendering policy that turns `snapshot`, `history`, and display
options into a renderable.

Useful Rich components inside `render(...)` include:

| Component | Use in a `RichReporter` layout | API |
|-----------|--------------------------------|-----|
| `Layout` | Split the terminal into named regions that can hold independent panels. | [Layout](https://rich.readthedocs.io/en/stable/layout.html) |
| `Panel` | Frame one region, table, plot, or status summary with a title. | [Panel](https://rich.readthedocs.io/en/stable/panel.html) |
| `Table` | Show latest scalar values, rank summaries, or status counts. | [Table](https://rich.readthedocs.io/en/stable/tables.html) |
| `Text` | Build styled labels, headers, and compact status lines. | [Text](https://rich.readthedocs.io/en/stable/text.html) |
| `Group` | Stack several renderables inside one layout region. | [Renderables](https://rich.readthedocs.io/en/stable/group.html) |
| `Columns` | Arrange small repeated panels, such as per-rank or per-status summaries. | [Columns](https://rich.readthedocs.io/en/stable/columns.html) |
| `Align` and `Padding` | Position or pad a renderable without creating another `Layout` region. | [Padding](https://rich.readthedocs.io/en/stable/padding.html) |

`Live` is intentionally absent from this list because `RichReporter` manages it.
Do not create or enter a nested `Live` display inside `render(...)`. If you want
standard line plots from retained metric history, subclass `BaseRichLayout`; it
already converts `history` into plotext-backed Rich renderables.

```python
from collections.abc import Mapping, Sequence

from rich import box
from rich.console import Group
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from nvalchemi.hooks import RichLayout, RichReporter, ScalarSnapshot
from nvalchemi.hooks.reporting.layouts import RichMetricHistory, RichPreviewHistory


class CompactRichLayout:
    include_dynamics_scalars = False

    def default_preview_history(self) -> RichPreviewHistory:
        return {"metric": (1.0, 0.8, 0.6)}

    def default_preview_stage(self) -> str:
        return "AFTER_STEP"

    def default_preview_epoch(self) -> None:
        return None

    def default_preview_batch_count(self) -> None:
        return None

    def render(
        self,
        snapshot: ScalarSnapshot | None,
        history: RichMetricHistory,
        *,
        title: str,
        precision: int,
        max_scalars: int | None,
        plot_keys: Sequence[str] | None,
        max_plots: int,
        plot_height: int,
    ) -> Layout:
        layout = Layout(name="root")
        layout.split_column(Layout(name="header", size=3), Layout(name="body"))
        subtitle = Text("waiting for metrics" if snapshot is None else snapshot.stage)
        layout["header"].update(Panel(Group(Text(title), subtitle), box=box.SIMPLE))

        table = Table(box=box.SIMPLE_HEAD, expand=True)
        table.add_column("Metric")
        table.add_column("Latest", justify="right")
        if snapshot is None:
            table.add_row("(waiting)", "")
        else:
            for key, value in sorted(snapshot.scalars.items()):
                table.add_row(key, f"{value:.{precision}g}")

        layout["body"].update(Panel(table, title="Summary"))
        return layout

layout: RichLayout = CompactRichLayout()
reporter = RichReporter(layout=layout)
```

The `render(...)` parameters are intentionally the same values that
`RichReporter` already manages:

- `snapshot` is the latest scalar payload.
- `history` contains retained `(step, value)` points for each metric.
- `plot_keys`, `max_plots`, and `plot_height` are user display preferences.
- `max_scalars` is the row limit for latest-value tables.

If your layout wants default dynamics observables, set
`include_dynamics_scalars = True`. The reporter will then include available
dynamics metrics such as energy, `fmax`, temperature, convergence fraction, and
active fraction before calling `render(...)`.
