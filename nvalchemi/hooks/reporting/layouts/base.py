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
"""Base classes and protocols for Rich reporting layouts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Literal, Protocol, TypeAlias

import plotext as plt
from rich import box
from rich.ansi import AnsiDecoder
from rich.console import Console, ConsoleOptions, Group, RenderResult
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from nvalchemi.hooks.reporting._scalars import ScalarSnapshot

RichMetricHistory: TypeAlias = Mapping[str, Sequence[tuple[int, float]]]
RichPreviewHistory: TypeAlias = Mapping[str, Sequence[float]]
RichLayoutName: TypeAlias = Literal["auto", "training", "dynamics"]


class RichLayout(Protocol):
    """Layout policy used by :class:`~nvalchemi.hooks.reporting.RichReporter`."""

    def default_preview_history(self) -> RichPreviewHistory:
        """Return synthetic metric curves for static dashboard previews."""
        ...

    def default_preview_stage(self) -> str:
        """Return the hook stage label used by static dashboard previews."""
        ...

    def default_preview_epoch(self) -> int | None:
        """Return the epoch metadata used by static dashboard previews."""
        ...

    def default_preview_batch_count(self) -> int | None:
        """Return the batch metadata used by static dashboard previews."""
        ...

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
        """Build the Rich layout for one reporter snapshot."""
        ...


class BaseRichLayout:
    """Reusable Rich dashboard layout for scalar tables and plot panels.

    Attributes
    ----------
    name : str
        Short layout name displayed in the dashboard header.
    include_dynamics_scalars : bool
        Whether :class:`~nvalchemi.hooks.reporting.RichReporter` should collect
        default dynamics observables when this layout is selected.
    """

    def __init__(
        self,
        *,
        name: str,
        preferred_plot_keys: Sequence[str],
        latest_title: str,
        history_title: str,
        include_dynamics_scalars: bool = False,
    ) -> None:
        self.name = name
        self._preferred_plot_keys = tuple(preferred_plot_keys)
        self._latest_title = latest_title
        self._history_title = history_title
        self.include_dynamics_scalars = include_dynamics_scalars

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
        """Build the Rich layout for one reporter snapshot.

        Parameters
        ----------
        snapshot : ScalarSnapshot | None
            Latest scalar snapshot, or ``None`` before the first report.
        history : RichMetricHistory
            Retained scalar history keyed by metric name.
        title : str
            Dashboard title.
        precision : int
            Significant digits used for scalar values.
        max_scalars : int | None
            Maximum number of latest scalar rows.
        plot_keys : Sequence[str] | None
            Explicit plot key ordering override.
        max_plots : int
            Maximum number of plot panels.
        plot_height : int
            Plot height in terminal rows.

        Returns
        -------
        Layout
            Renderable Rich layout.
        """
        layout = Layout(name="root")
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="body"),
        )
        layout["body"].split_row(
            Layout(name="latest", ratio=2),
            Layout(name="plots", ratio=3),
        )
        layout["header"].update(self._build_header(snapshot, title))
        layout["latest"].update(
            Panel(
                self._build_table(snapshot, precision, max_scalars),
                title=self._latest_title,
            )
        )
        layout["plots"].update(
            Panel(
                self._build_plots(
                    history,
                    precision=precision,
                    plot_keys=plot_keys,
                    max_plots=max_plots,
                    plot_height=plot_height,
                ),
                title=self._history_title,
            )
        )
        return layout

    def default_preview_history(self) -> RichPreviewHistory:
        """Return synthetic metric curves for static dashboard previews."""
        raise NotImplementedError

    def default_preview_stage(self) -> str:
        """Return the hook stage label used by static dashboard previews."""
        return "AFTER_OPTIMIZER_STEP"

    def default_preview_epoch(self) -> int | None:
        """Return the epoch metadata used by static dashboard previews."""
        return 3

    def default_preview_batch_count(self) -> int | None:
        """Return the batch metadata used by static dashboard previews."""
        return 128

    def _build_header(
        self,
        snapshot: ScalarSnapshot | None,
        title: str,
    ) -> Panel:
        if snapshot is None:
            body = f"{title} | {self.name} | waiting for metrics"
        else:
            body = f"{title} | {self.name} | {snapshot.stage}"
            if snapshot.step_count is not None:
                body = f"{body} | step {snapshot.step_count}"
        return Panel(Text(body, overflow="fold"), box=box.SIMPLE)

    def _build_table(
        self,
        snapshot: ScalarSnapshot | None,
        precision: int,
        max_scalars: int | None,
    ) -> Table:
        table = Table(box=box.SIMPLE_HEAD, show_lines=False, expand=True)
        table.add_column("Metric", overflow="fold")
        table.add_column("Latest", justify="right", no_wrap=True)
        if snapshot is None or not snapshot.scalars:
            table.add_row("(no scalars)", "")
            return table
        items = self._scalar_table_items(snapshot)
        visible_items = items[:max_scalars] if max_scalars is not None else items
        for key, value in visible_items:
            table.add_row(key, self._format_value(value, precision))
        if len(visible_items) < len(items):
            table.add_row("...", f"{len(items) - len(visible_items)} omitted")
        table.caption = self._caption(snapshot)
        return table

    def _scalar_table_items(self, snapshot: ScalarSnapshot) -> list[tuple[str, float]]:
        preferred = [
            (key, snapshot.scalars[key])
            for key in self._preferred_plot_keys
            if key in snapshot.scalars
        ]
        seen = {key for key, _ in preferred}
        preferred.extend(
            (key, value)
            for key, value in sorted(snapshot.scalars.items())
            if key not in seen
        )
        return preferred

    def _build_plots(
        self,
        history: RichMetricHistory,
        *,
        precision: int,
        plot_keys: Sequence[str] | None,
        max_plots: int,
        plot_height: int,
    ) -> Group | Text:
        keys = self._selected_plot_keys(
            history,
            plot_keys=plot_keys,
            max_plots=max_plots,
        )
        if not keys:
            return Text("No scalar history yet.")
        panels = [
            Panel(
                _PlotextSeries(
                    key=key,
                    series=tuple(history[key]),
                    precision=precision,
                    height=plot_height,
                ),
                title=key,
                box=box.SIMPLE,
            )
            for key in keys
        ]
        return Group(*panels)

    def _selected_plot_keys(
        self,
        history: RichMetricHistory,
        *,
        plot_keys: Sequence[str] | None,
        max_plots: int,
    ) -> tuple[str, ...]:
        if max_plots == 0:
            return ()
        available = [key for key, values in history.items() if values]
        if plot_keys is not None:
            keys = [key for key in plot_keys if key in available]
        else:
            keys = [key for key in self._preferred_plot_keys if key in available]
            keys.extend(sorted(key for key in available if key not in keys))
        return tuple(keys[:max_plots])

    def _format_value(self, value: float, precision: int) -> str:
        return f"{value:.{precision}g}"

    def _caption(self, snapshot: ScalarSnapshot) -> str:
        parts = [f"rank={snapshot.global_rank}"]
        if snapshot.event_count is not None:
            parts.append(f"event={snapshot.event_count}")
        if snapshot.epoch is not None:
            parts.append(f"epoch={snapshot.epoch}")
        if snapshot.batch_count is not None:
            parts.append(f"batch={snapshot.batch_count}")
        return " | ".join(parts)

    def _build_messages(self, snapshot: ScalarSnapshot | None) -> Table:
        table = Table.grid(expand=True)
        table.add_column("Level", no_wrap=True)
        table.add_column("Message", overflow="fold")
        if snapshot is None or not snapshot.messages:
            table.add_row("info", "No reporter messages.")
            return table
        for message in snapshot.messages[-3:]:
            prefix = message.level
            if message.reporter is not None:
                prefix = f"{prefix}/{message.reporter}"
            table.add_row(prefix, message.message)
        return table

    def _format_duration(self, seconds: float) -> str:
        if seconds < 60:
            return f"{seconds:.1f}s"
        minutes, remaining_seconds = divmod(int(seconds), 60)
        if minutes < 60:
            return f"{minutes}m {remaining_seconds}s"
        hours, remaining_minutes = divmod(minutes, 60)
        return f"{hours}h {remaining_minutes}m"

    def _add_scalar_row(
        self,
        table: Table,
        snapshot: ScalarSnapshot,
        key: str,
        label: str,
        precision: int,
        *,
        suffix: str = "",
        scale: float = 1.0,
    ) -> None:
        if key not in snapshot.scalars:
            return
        value = snapshot.scalars[key] * scale
        table.add_row(label, f"{self._format_value(value, precision)}{suffix}")


class _PlotextSeries:
    def __init__(
        self,
        *,
        key: str,
        series: Sequence[tuple[int, float]],
        precision: int,
        height: int,
    ) -> None:
        self.key = key
        self.series = series
        self.precision = precision
        self.height = height
        self.decoder = AnsiDecoder()

    def __rich_console__(
        self,
        console: Console,
        options: ConsoleOptions,
    ) -> RenderResult:
        width = max(20, options.max_width or console.width)
        canvas = self._build_canvas(width)
        yield Group(*self.decoder.decode(canvas))

    def _build_canvas(self, width: int) -> str:
        plt.clf()
        steps = [step for step, _ in self.series]
        values = [value for _, value in self.series]
        plt.plotsize(width, self.height)
        plt.theme("dark")
        plt.title(self.key)
        plt.xlabel("step")
        if len(values) == 1:
            plt.scatter(steps, values)
        else:
            plt.plot(steps, values)
        return plt.build()
