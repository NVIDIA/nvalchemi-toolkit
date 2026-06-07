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

from nvalchemi.hooks.reporting.layouts.base import BaseRichLayout, RichPreviewHistory


class DynamicsRichLayout(BaseRichLayout):
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
