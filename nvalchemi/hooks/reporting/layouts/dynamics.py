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
"""Dynamics Rich reporting layout."""

from __future__ import annotations

from collections.abc import Sequence

from rich import box
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from nvalchemi.hooks.reporting._scalars import ScalarSnapshot
from nvalchemi.hooks.reporting.layouts.base import (
    BaseRichLayout,
    RichMetricHistory,
    RichPreviewHistory,
)


class DynamicsRichLayout(BaseRichLayout):
    """Rich dashboard layout for dynamics workflows."""

    _observable_keys = ("energy", "fmax", "temperature", "energy_drift")
    _status_keys = ("active_fraction", "converged_fraction")

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
        """Build a dynamics-specific Rich dashboard.

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
            Maximum number of observable rows.
        plot_keys : Sequence[str] | None
            Explicit plot key ordering override.
        max_plots : int
            Maximum number of plot panels.
        plot_height : int
            Plot height in terminal rows.

        Returns
        -------
        Layout
            Renderable Rich layout with dynamics observables, status, and traces.
        """
        layout = Layout(name="root")
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="body"),
        )
        layout["body"].split_row(
            Layout(name="state", ratio=2),
            Layout(name="traces", ratio=3),
        )
        layout["state"].split_column(
            Layout(name="observables", ratio=2),
            Layout(name="pipeline", ratio=2),
            Layout(name="messages", size=4),
        )
        layout["header"].update(self._build_header(snapshot, title))
        layout["observables"].update(
            Panel(
                self._build_observables(snapshot, precision, max_scalars),
                title="Observables",
            )
        )
        layout["pipeline"].update(
            Panel(
                self._build_pipeline(snapshot, precision),
                title="Convergence / Pipeline",
            )
        )
        layout["messages"].update(
            Panel(self._build_messages(snapshot), title="Messages")
        )
        layout["traces"].update(
            Panel(
                self._build_plots(
                    history,
                    precision=precision,
                    plot_keys=plot_keys,
                    max_plots=max_plots,
                    plot_height=plot_height,
                ),
                title="Dynamics Traces",
            )
        )
        return layout

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

    def default_preview_stage(self) -> str:
        """Return the dynamics hook stage label used by static previews."""
        return "AFTER_STEP"

    def default_preview_epoch(self) -> None:
        """Return no epoch metadata for dynamics previews."""
        return None

    def default_preview_batch_count(self) -> None:
        """Return no batch metadata for dynamics previews."""
        return None

    def _build_observables(
        self,
        snapshot: ScalarSnapshot | None,
        precision: int,
        max_scalars: int | None,
    ) -> Table:
        table = Table(box=box.SIMPLE_HEAD, show_lines=False, expand=True)
        table.add_column("Observable", overflow="fold")
        table.add_column("Latest", justify="right", no_wrap=True)
        if snapshot is None or not snapshot.scalars:
            table.add_row("(waiting)", "")
            return table
        keys = [key for key in self._observable_keys if key in snapshot.scalars]
        keys.extend(
            sorted(
                key
                for key in snapshot.scalars
                if key not in keys and key not in self._status_keys
            )
        )
        visible_keys = keys[:max_scalars] if max_scalars is not None else keys
        for key in visible_keys:
            table.add_row(key, self._format_value(snapshot.scalars[key], precision))
        if len(visible_keys) < len(keys):
            table.add_row("...", f"{len(keys) - len(visible_keys)} omitted")
        return table

    def _build_status(self, snapshot: ScalarSnapshot | None, precision: int) -> Table:
        table = Table.grid(expand=True)
        table.add_column("Field", overflow="fold")
        table.add_column("Value", justify="right", no_wrap=True)
        if snapshot is None:
            table.add_row("state", Text("waiting"))
            return table
        for key in self._status_keys:
            if key in snapshot.scalars:
                table.add_row(key, self._format_value(snapshot.scalars[key], precision))
        table.add_row("rank", str(snapshot.global_rank))
        if snapshot.event_count is not None:
            table.add_row("event", str(snapshot.event_count))
        if snapshot.step_count is not None:
            table.add_row("step", str(snapshot.step_count))
        return table

    def _build_pipeline(
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
        for key in self._status_keys:
            if key in snapshot.scalars:
                table.add_row(key, self._format_value(snapshot.scalars[key], precision))
        if "dynamics/num_graphs" in snapshot.scalars:
            table.add_row(
                "systems",
                self._format_value(snapshot.scalars["dynamics/num_graphs"], precision),
            )
        if "dynamics/active_count" in snapshot.scalars:
            table.add_row(
                "active",
                self._format_value(
                    snapshot.scalars["dynamics/active_count"], precision
                ),
            )
        if "dynamics/graduated_count" in snapshot.scalars:
            table.add_row(
                "graduated",
                self._format_value(
                    snapshot.scalars["dynamics/graduated_count"],
                    precision,
                ),
            )
        if "dynamics/converged_count" in snapshot.scalars:
            table.add_row(
                "converged",
                self._format_value(
                    snapshot.scalars["dynamics/converged_count"],
                    precision,
                ),
            )
        for key, value in sorted(snapshot.scalars.items()):
            prefix = "dynamics/status/"
            suffix = "/count"
            if key.startswith(prefix) and key.endswith(suffix):
                status = key[len(prefix) : -len(suffix)]
                table.add_row(f"status {status}", self._format_value(value, precision))
        self._add_scalar_row(
            table,
            snapshot,
            "dynamics/progress_fraction",
            "progress",
            precision,
            suffix="%",
            scale=100.0,
        )
        self._add_scalar_row(
            table,
            snapshot,
            "dynamics/steps_per_s",
            "steps/s",
            precision,
        )
        if "dynamics/eta_s" in snapshot.scalars:
            table.add_row(
                "eta", self._format_duration(snapshot.scalars["dynamics/eta_s"])
            )
        table.add_row("rank", str(snapshot.global_rank))
        if snapshot.event_count is not None:
            table.add_row("event", str(snapshot.event_count))
        if snapshot.step_count is not None:
            table.add_row("step", str(snapshot.step_count))
        return table
