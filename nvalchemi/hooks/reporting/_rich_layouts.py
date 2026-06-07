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
"""Rich dashboard layout policies."""

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
RichLayoutName: TypeAlias = Literal["training", "dynamics"]


class RichLayout(Protocol):
    """Layout policy used by :class:`~nvalchemi.hooks.reporting.RichReporter`."""

    def default_preview_history(self) -> RichPreviewHistory:
        """Return synthetic metric curves for static dashboard previews."""
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


class _BaseRichLayout:
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
        """Build the Rich layout for one reporter snapshot."""
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
        items = sorted(snapshot.scalars.items())
        visible_items = items[:max_scalars] if max_scalars is not None else items
        for key, value in visible_items:
            table.add_row(key, self._format_value(value, precision))
        if len(visible_items) < len(items):
            table.add_row("...", f"{len(items) - len(visible_items)} omitted")
        table.caption = self._caption(snapshot)
        return table

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


class TrainingRichLayout(_BaseRichLayout):
    """Rich dashboard layout for training workflows."""

    def __init__(self) -> None:
        super().__init__(
            name="training",
            preferred_plot_keys=(
                "loss/total",
                "loss/energy/total",
                "loss/forces/total",
                "optimizer/lr",
            ),
            latest_title="Latest",
            history_title="History",
        )

    def default_preview_history(self) -> RichPreviewHistory:
        """Return representative training metrics for preview rendering."""
        return {
            "loss/total": (1.2, 0.86, 0.61, 0.43, 0.31, 0.24),
            "loss/energy/total": (0.54, 0.39, 0.27, 0.19, 0.14, 0.11),
            "loss/forces/total": (0.66, 0.47, 0.34, 0.24, 0.17, 0.13),
            "optimizer/lr": (1e-3, 1e-3, 8e-4, 5e-4, 2e-4, 1e-4),
        }


class DynamicsRichLayout(_BaseRichLayout):
    """Rich dashboard layout for dynamics workflows."""

    def __init__(self) -> None:
        super().__init__(
            name="dynamics",
            preferred_plot_keys=(
                "energy",
                "fmax",
                "temperature",
                "energy_drift",
                "converged_fraction",
                "active_fraction",
            ),
            latest_title="State",
            history_title="Traces",
            include_dynamics_scalars=True,
        )

    def default_preview_history(self) -> RichPreviewHistory:
        """Return representative dynamics metrics for preview rendering."""
        return {
            "energy": (-15.2, -15.18, -15.21, -15.19, -15.2, -15.18),
            "fmax": (0.42, 0.31, 0.22, 0.18, 0.12, 0.08),
            "temperature": (297.0, 301.0, 299.0, 300.0, 302.0, 300.0),
            "energy_drift": (0.0, 0.02, -0.01, 0.01, 0.0, 0.02),
            "converged_fraction": (0.05, 0.12, 0.25, 0.41, 0.68, 0.92),
            "active_fraction": (1.0, 1.0, 0.95, 0.9, 0.72, 0.5),
        }


def resolve_rich_layout(layout: RichLayout | RichLayoutName | str | None) -> RichLayout:
    """Resolve a Rich layout name or instance to a layout object.

    Parameters
    ----------
    layout : RichLayout | {"training", "dynamics"} | str | None
        Layout instance or built-in layout name. ``None`` selects the training
        layout for backward compatibility.

    Returns
    -------
    RichLayout
        Resolved layout policy.

    Raises
    ------
    ValueError
        If a string layout name is not recognized.
    TypeError
        If an object does not implement the layout protocol.
    """
    if layout is None or layout == "training":
        return TrainingRichLayout()
    if layout == "dynamics":
        return DynamicsRichLayout()
    if isinstance(layout, str):
        raise ValueError(
            "RichReporter layout must be 'training', 'dynamics', or a layout object."
        )
    if not callable(getattr(layout, "default_preview_history", None)) or not callable(
        getattr(layout, "render", None)
    ):
        raise TypeError(
            "RichReporter layout objects must define default_preview_history() "
            "and render()."
        )
    return layout


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
