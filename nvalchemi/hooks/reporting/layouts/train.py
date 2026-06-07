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

from nvalchemi.hooks.reporting.layouts.base import BaseRichLayout, RichPreviewHistory


class TrainingRichLayout(BaseRichLayout):
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
