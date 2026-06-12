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
"""Rich live dashboard reporting sink."""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping, Sequence
from enum import Enum
from types import TracebackType

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from torch import distributed as dist

from nvalchemi.hooks._context import DynamicsContext, HookContext, TrainContext
from nvalchemi.hooks.reporting._distributed import (
    normalize_rank_reduction,
    reduce_scalar_snapshot,
)
from nvalchemi.hooks.reporting._scalars import (
    ScalarCallback,
    ScalarSnapshot,
    collect_scalars,
)
from nvalchemi.hooks.reporting._state import ReportingState
from nvalchemi.hooks.reporting.layouts import (
    DynamicsRichLayout,
    RichLayout,
    TrainingRichLayout,
    resolve_rich_layout,
)

_PREVIEW_DEFAULT = object()


class RichReporter:
    """Render scalar reporting snapshots as a live Rich dashboard.

    Parameters
    ----------
    custom_scalars : Mapping[str, ScalarCallback] | None, optional
        Additional scalar callbacks passed to :func:`collect_scalars`.
    include_losses : bool, default True
        When ``True``, include loss scalars from the hook context.
    include_optimizer_lrs : bool, default True
        When ``True``, include optimizer learning rates from the hook context.
    include_dynamics_scalars : bool | None, optional
        When ``True``, include default dynamics observables from the hook
        context. ``None`` lets the selected layout choose; the built-in
        dynamics layout enables them.
    rank_reduction : torch.distributed.ReduceOp | {"none", "mean", "sum", "min", "max"} | None, default None
        Optional distributed reduction applied to scalars before rendering.
        String values are normalized to :class:`torch.distributed.ReduceOp`.
        Reduction requires every rank to call this reporter; only rank zero
        renders the reduced dashboard.
    title : str, default "nvalchemi report"
        Dashboard title.
    precision : int, default 6
        Significant digits used when formatting scalar values.
    max_scalars : int | None, optional
        Maximum number of scalar rows to show. When omitted, all scalars are
        shown.
    history_size : int, default 200
        Maximum history points retained per scalar.
    layout : RichLayout | {"training", "dynamics"} | None, optional
        Dashboard layout policy. ``None`` and ``"auto"`` select the first
        built-in layout that supports the first reported context.
    plot_keys : Sequence[str] | None, optional
        Scalar keys to plot. When omitted, the selected layout chooses common
        metrics for that workflow before falling back to alphabetical order.
    max_plots : int, default 3
        Maximum number of history plots shown in the dashboard.
    plot_height : int, default 8
        Height in terminal rows for each plotext plot.
    refresh_per_second : float, default 2.0
        Rich ``Live`` refresh rate used while the reporter is entered.
    console : Console | None, optional
        Rich console used for output. When omitted, a stderr console is created.
    screen : bool, default False
        Whether Rich ``Live`` should use the terminal alternate screen.
    transient : bool, default False
        Whether Rich ``Live`` should clear the dashboard on exit.
    rank_zero_only : bool, default True
        Request rank-zero-only dispatch from :class:`ReportingOrchestrator`.
    strict_layout : bool, default False
        When ``True``, raise if automatic layout selection cannot match the
        incoming context. When ``False``, unmatched contexts are ignored.
    """

    def __init__(
        self,
        *,
        custom_scalars: Mapping[str, ScalarCallback] | None = None,
        include_losses: bool = True,
        include_optimizer_lrs: bool = True,
        include_dynamics_scalars: bool | None = None,
        rank_reduction: dist.ReduceOp | str | None = None,
        title: str = "nvalchemi report",
        precision: int = 6,
        max_scalars: int | None = None,
        history_size: int = 200,
        layout: RichLayout | str | None = None,
        plot_keys: Sequence[str] | None = None,
        max_plots: int = 3,
        plot_height: int = 8,
        refresh_per_second: float = 2.0,
        console: Console | None = None,
        screen: bool = False,
        transient: bool = False,
        rank_zero_only: bool = True,
        strict_layout: bool = False,
    ) -> None:
        if precision < 0:
            raise ValueError("RichReporter precision must be non-negative.")
        if max_scalars is not None and max_scalars < 1:
            raise ValueError("RichReporter max_scalars must be positive.")
        if history_size < 1:
            raise ValueError("RichReporter history_size must be positive.")
        if max_plots < 0:
            raise ValueError("RichReporter max_plots must be non-negative.")
        if plot_height < 4:
            raise ValueError("RichReporter plot_height must be at least 4.")
        if refresh_per_second <= 0:
            raise ValueError("RichReporter refresh_per_second must be positive.")
        self.custom_scalars = custom_scalars
        self.include_losses = include_losses
        self.include_optimizer_lrs = include_optimizer_lrs
        self.rank_reduction = rank_reduction
        self._rank_reduction_op, _ = normalize_rank_reduction(rank_reduction)
        self.title = title
        self.precision = precision
        self.max_scalars = max_scalars
        self.history_size = history_size
        self._auto_layout = layout is None or layout == "auto"
        self._layout_selected = not self._auto_layout
        self.layout = (
            TrainingRichLayout() if self._auto_layout else resolve_rich_layout(layout)
        )
        self._include_dynamics_scalars_override = include_dynamics_scalars
        self.include_dynamics_scalars = (
            bool(getattr(self.layout, "include_dynamics_scalars", False))
            if include_dynamics_scalars is None
            else include_dynamics_scalars
        )
        self.plot_keys = tuple(plot_keys) if plot_keys is not None else None
        self.max_plots = max_plots
        self.plot_height = plot_height
        self.refresh_per_second = refresh_per_second
        self.console = console if console is not None else Console(stderr=True)
        self.screen = screen
        self.transient = transient
        self.strict_layout = strict_layout
        self._write_rank_zero_only = (
            rank_zero_only or self._rank_reduction_op is not None
        )
        self.rank_zero_only = rank_zero_only and self._rank_reduction_op is None
        self.requires_all_ranks = self._rank_reduction_op is not None
        self._history: dict[str, deque[tuple[int, float]]] = {}
        self._latest_snapshot: ScalarSnapshot | None = None
        self._live: Live | None = None
        self._entered = False

    @classmethod
    def preview(
        cls,
        *,
        history: Mapping[str, Sequence[float]] | None = None,
        layout: RichLayout | str | None = None,
        steps: Sequence[int] | None = None,
        console: Console | None = None,
        stage: str | None = None,
        step_count: int | None = None,
        epoch: int | None | object = _PREVIEW_DEFAULT,
        batch_count: int | None | object = _PREVIEW_DEFAULT,
        **reporter_kwargs: object,
    ) -> None:
        """Render a synthetic dashboard preview.

        Parameters
        ----------
        history : Mapping[str, Sequence[float]] | None, optional
            Metric history used to populate plots and latest values. Defaults to
            representative curves from the selected layout.
        layout : RichLayout | {"training", "dynamics"} | None, optional
            Dashboard layout policy. ``None`` selects the training layout.
        steps : Sequence[int] | None, optional
            Step values aligned with each history sequence. Defaults to
            ``range(len(series))``.
        console : Console | None, optional
            Rich console used for preview output.
        stage : str | None, optional
            Stage label shown in the dashboard header. When omitted, the
            selected layout supplies a workflow-appropriate default.
        step_count : int | None, optional
            Step shown in the dashboard header. Defaults to the final step.
        epoch : int | None, optional
            Epoch shown in dashboard metadata. When omitted, the selected
            layout supplies a workflow-appropriate default.
        batch_count : int | None, optional
            Batch count shown in dashboard metadata. When omitted, the
            selected layout supplies a workflow-appropriate default.
        **reporter_kwargs : object
            Additional keyword arguments forwarded to :class:`RichReporter`.
        """
        reporter = cls(
            console=console,
            layout=layout,
            rank_zero_only=False,
            **reporter_kwargs,
        )
        if reporter._auto_layout:
            reporter._set_layout(TrainingRichLayout())
        reporter.seed_history(
            reporter.layout.default_preview_history() if history is None else history,
            steps=steps,
            stage=stage
            if stage is not None
            else reporter.layout.default_preview_stage(),
            step_count=step_count,
            epoch=reporter.layout.default_preview_epoch()
            if epoch is _PREVIEW_DEFAULT
            else epoch,
            batch_count=reporter.layout.default_preview_batch_count()
            if batch_count is _PREVIEW_DEFAULT
            else batch_count,
        )
        reporter.console.print(reporter.renderable())

    @property
    def history(self) -> dict[str, tuple[tuple[int, float], ...]]:
        """Return retained scalar history.

        Returns
        -------
        dict[str, tuple[tuple[int, float], ...]]
            Mapping from scalar key to ``(step, value)`` history tuples.
        """
        return {key: tuple(values) for key, values in self._history.items()}

    def __enter__(self) -> RichReporter:
        """Start the live dashboard."""
        if self._entered:
            return self
        self._entered = True
        if self._rank_reduction_op is None and not (
            self._auto_layout and not self._layout_selected
        ):
            self._start_live()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Stop the live dashboard."""
        self.close()

    def close(self) -> None:
        """Stop the live dashboard if it is active."""
        if self._live is None:
            self._entered = False
            return
        self._live.stop()
        self._live = None
        self._entered = False

    def report(self, ctx: HookContext, stage: Enum, state: ReportingState) -> None:
        """Update the dashboard from one scalar snapshot.

        Parameters
        ----------
        ctx : HookContext
            Workflow hook context.
        stage : Enum
            Hook stage being reported.
        state : ReportingState
            Shared reporting state from the orchestrator.
        """
        if not self._ensure_layout(ctx, stage):
            return
        snapshot = collect_scalars(
            ctx,
            stage,
            state,
            custom_scalars=self.custom_scalars,
            include_losses=self.include_losses,
            include_optimizer_lrs=self.include_optimizer_lrs,
            include_dynamics=self.include_dynamics_scalars,
            include_progress=True,
        )
        if self._rank_reduction_op is not None:
            snapshot = reduce_scalar_snapshot(
                snapshot,
                self.rank_reduction,
                reporter_name=type(self).__name__,
            )
            if not self._is_rank_zero(ctx):
                return
        elif self._write_rank_zero_only and not self._is_rank_zero(ctx):
            return
        self._record_snapshot(snapshot)
        renderable = self.renderable()
        if self._live is not None:
            self._live.update(renderable, refresh=False)
        elif self._entered:
            self._start_live(renderable)
        else:
            self.console.print(renderable)

    def seed_history(
        self,
        history: Mapping[str, Sequence[float]],
        *,
        steps: Sequence[int] | None = None,
        stage: str = "AFTER_OPTIMIZER_STEP",
        step_count: int | None = None,
        epoch: int | None = None,
        batch_count: int | None = None,
        global_rank: int = 0,
    ) -> ScalarSnapshot:
        """Seed dashboard history without running a workflow.

        Parameters
        ----------
        history : Mapping[str, Sequence[float]]
            Metric history used to populate plots and latest scalar values.
        steps : Sequence[int] | None, optional
            Step values aligned with each metric series.
        stage : str, default "AFTER_OPTIMIZER_STEP"
            Stage label for the synthetic snapshot.
        step_count : int | None, optional
            Step count for the synthetic snapshot. Defaults to the final step.
        epoch : int | None, optional
            Epoch metadata for the synthetic snapshot.
        batch_count : int | None, optional
            Batch metadata for the synthetic snapshot.
        global_rank : int, default 0
            Rank metadata for the synthetic snapshot.

        Returns
        -------
        ScalarSnapshot
            Synthetic latest snapshot produced from ``history``.
        """
        if not history:
            raise ValueError("RichReporter preview history cannot be empty.")
        first_values = next(iter(history.values()))
        if not first_values:
            raise ValueError(
                "RichReporter preview history cannot contain empty series."
            )
        if steps is None:
            resolved_steps = tuple(range(len(first_values)))
        else:
            resolved_steps = tuple(steps)
        if len(resolved_steps) != len(first_values):
            raise ValueError("RichReporter preview steps must match series length.")
        self._history = {}
        latest_scalars: dict[str, float] = {}
        for key, values in history.items():
            if len(values) != len(resolved_steps):
                raise ValueError("RichReporter preview series lengths must match.")
            numeric_values = tuple(float(value) for value in values)
            self._history[key] = deque(
                zip(resolved_steps, numeric_values, strict=True),
                maxlen=self.history_size,
            )
            latest_scalars[key] = numeric_values[-1]
        resolved_step_count = (
            step_count if step_count is not None else resolved_steps[-1]
        )
        snapshot = ScalarSnapshot(
            stage=stage,
            scalars=latest_scalars,
            step_count=resolved_step_count,
            batch_count=batch_count,
            epoch=epoch,
            global_rank=global_rank,
        )
        self._latest_snapshot = snapshot
        return snapshot

    def renderable(self) -> Layout:
        """Build the current dashboard renderable.

        Returns
        -------
        Layout
            Rich layout containing the header, latest scalar table, and plots.
        """
        return self.layout.render(
            self._latest_snapshot,
            self.history,
            title=self.title,
            precision=self.precision,
            max_scalars=self.max_scalars,
            plot_keys=self.plot_keys,
            max_plots=self.max_plots,
            plot_height=self.plot_height,
        )

    def _ensure_layout(self, ctx: HookContext, stage: Enum) -> bool:
        if not self._auto_layout:
            return True
        if self._layout_selected:
            return True
        if isinstance(ctx, DynamicsContext) or stage.name == "AFTER_STEP":
            self._set_layout(DynamicsRichLayout())
            return True
        if isinstance(ctx, TrainContext) or _looks_like_training_context(ctx, stage):
            self._set_layout(TrainingRichLayout())
            return True
        if self.strict_layout:
            raise ValueError(
                "RichReporter could not select a layout for "
                f"context {type(ctx).__name__} at stage {stage.name!r}."
            )
        return False

    def _set_layout(self, layout: RichLayout) -> None:
        self.layout = layout
        self._layout_selected = True
        if self._include_dynamics_scalars_override is None:
            self.include_dynamics_scalars = bool(
                getattr(self.layout, "include_dynamics_scalars", False)
            )

    def _record_snapshot(self, snapshot: ScalarSnapshot) -> None:
        self._latest_snapshot = snapshot
        step = self._history_step(snapshot)
        for key, value in snapshot.scalars.items():
            if key not in self._history:
                self._history[key] = deque(maxlen=self.history_size)
            self._history[key].append((step, value))

    def _history_step(self, snapshot: ScalarSnapshot) -> int:
        if snapshot.step_count is not None:
            return snapshot.step_count
        if snapshot.event_count is not None:
            return snapshot.event_count
        lengths = [len(values) for values in self._history.values()]
        return max(lengths, default=0)

    def _is_rank_zero(self, ctx: HookContext) -> bool:
        return ctx.global_rank == 0

    def _start_live(self, renderable: Layout | None = None) -> None:
        if self._live is not None:
            return
        self._live = Live(
            renderable if renderable is not None else self.renderable(),
            console=self.console,
            refresh_per_second=self.refresh_per_second,
            screen=self.screen,
            transient=self.transient,
        )
        self._live.start()


def _looks_like_training_context(ctx: HookContext, stage: Enum) -> bool:
    if stage.name == "AFTER_OPTIMIZER_STEP":
        return True
    return any(
        hasattr(ctx, name)
        for name in (
            "loss",
            "losses",
            "optimizers",
            "lr_schedulers",
            "batch_count",
            "epoch_step_count",
        )
    )
