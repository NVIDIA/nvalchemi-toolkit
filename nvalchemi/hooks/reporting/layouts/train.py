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
"""Training Rich reporting layout."""

from __future__ import annotations

from collections.abc import Sequence

from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table

from nvalchemi.hooks.reporting._scalars import ScalarSnapshot
from nvalchemi.hooks.reporting.layouts.base import (
    BaseRichLayout,
    RichMetricHistory,
    RichPreviewHistory,
)


class TrainingRichLayout(BaseRichLayout):
    """Rich dashboard layout for training workflows."""

    def __init__(self) -> None:
        super().__init__(
            name="training",
            preferred_plot_keys=(
                "loss/total",
                "optimizer/lr",
                "scheduler/lr",
                "loss/energy/unweighted",
                "loss/forces/unweighted",
            ),
            latest_title="Latest",
            history_title="History",
        )

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
        """Build a training-specific Rich dashboard.

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
            Renderable Rich layout with training metrics and progress.
        """
        layout = Layout(name="root")
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="body"),
            Layout(name="messages", size=5),
        )
        layout["body"].split_row(
            Layout(name="left", ratio=2),
            Layout(name="plots", ratio=3),
        )
        layout["left"].split_column(
            Layout(name="latest", ratio=3),
            Layout(name="progress", size=9),
        )
        layout["header"].update(self._build_header(snapshot, title))
        layout["latest"].update(
            Panel(
                self._build_table(snapshot, precision, max_scalars),
                title="Latest Metrics",
            )
        )
        layout["progress"].update(
            Panel(
                self._build_progress(snapshot, precision),
                title="Progress",
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
                title="Training Curves",
            )
        )
        layout["messages"].update(
            Panel(self._build_messages(snapshot), title="Messages")
        )
        return layout

    def default_preview_history(self) -> RichPreviewHistory:
        """Return representative training metrics for preview rendering."""
        return {
            "loss/total": (1.2, 0.86, 0.61, 0.43, 0.31, 0.24),
            "loss/energy/unweighted": (0.54, 0.39, 0.27, 0.19, 0.14, 0.11),
            "loss/forces/unweighted": (0.66, 0.47, 0.34, 0.24, 0.17, 0.13),
            "optimizer/lr": (1e-3, 1e-3, 8e-4, 5e-4, 2e-4, 1e-4),
            "scheduler/lr": (1e-3, 1e-3, 8e-4, 5e-4, 2e-4, 1e-4),
        }

    def _build_progress(
        self,
        snapshot: ScalarSnapshot | None,
        precision: int,
    ) -> Table:
        table = Table.grid(expand=True)
        table.add_column("Field", overflow="fold")
        table.add_column("Value", justify="right", no_wrap=True)
        if snapshot is None:
            table.add_row("state", "waiting")
            return table
        table.add_row("rank", str(snapshot.global_rank))
        if snapshot.event_count is not None:
            table.add_row("event", str(snapshot.event_count))
        if snapshot.step_count is not None:
            table.add_row("step", str(snapshot.step_count))
        if snapshot.batch_count is not None:
            table.add_row("batch", str(snapshot.batch_count))
        if snapshot.epoch is not None:
            table.add_row("epoch", str(snapshot.epoch))
        if snapshot.epoch_step_count is not None:
            table.add_row("epoch batch", str(snapshot.epoch_step_count))
        self._add_scalar_row(
            table,
            snapshot,
            "training/progress_fraction",
            "progress",
            precision,
            suffix="%",
            scale=100.0,
        )
        self._add_scalar_row(
            table,
            snapshot,
            "training/steps_per_s",
            "steps/s",
            precision,
        )
        self._add_scalar_row(
            table,
            snapshot,
            "training/batches_per_s",
            "batches/s",
            precision,
        )
        if "training/eta_s" in snapshot.scalars:
            table.add_row(
                "eta", self._format_duration(snapshot.scalars["training/eta_s"])
            )
        if "scheduler/lr" in snapshot.scalars:
            table.add_row(
                "scheduler lr",
                self._format_value(snapshot.scalars["scheduler/lr"], precision),
            )
        elif "optimizer/lr" in snapshot.scalars:
            table.add_row(
                "optimizer lr",
                self._format_value(snapshot.scalars["optimizer/lr"], precision),
            )
        return table
