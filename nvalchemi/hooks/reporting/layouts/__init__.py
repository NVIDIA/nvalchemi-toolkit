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
"""Rich reporting layouts."""

from __future__ import annotations

from nvalchemi.hooks.reporting.layouts.base import (
    BaseRichLayout,
    RichLayout,
    RichLayoutName,
    RichMetricHistory,
    RichPreviewHistory,
)
from nvalchemi.hooks.reporting.layouts.dynamics import DynamicsRichLayout
from nvalchemi.hooks.reporting.layouts.train import TrainingRichLayout

__all__ = [
    "BaseRichLayout",
    "DynamicsRichLayout",
    "RichLayout",
    "RichLayoutName",
    "RichMetricHistory",
    "RichPreviewHistory",
    "TrainingRichLayout",
    "resolve_rich_layout",
]

_REQUIRED_RICH_LAYOUT_METHODS = (
    "default_preview_history",
    "default_preview_stage",
    "default_preview_epoch",
    "default_preview_batch_count",
    "render",
)


def resolve_rich_layout(layout: RichLayout | RichLayoutName | str | None) -> RichLayout:
    """Resolve a Rich layout name or instance to a layout object.

    Parameters
    ----------
    layout : RichLayout | {"training", "dynamics"} | str | None
        Layout instance or concrete built-in layout name. ``"auto"`` and
        ``None`` are handled by :class:`~nvalchemi.hooks.RichReporter` before
        this resolver is called.

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
            "RichReporter layout must be 'auto', 'training', 'dynamics', "
            "or a layout object."
        )
    missing = [
        method
        for method in _REQUIRED_RICH_LAYOUT_METHODS
        if not callable(getattr(layout, method, None))
    ]
    if missing:
        raise TypeError(
            "RichReporter layout objects must define "
            f"{', '.join(f'{method}()' for method in _REQUIRED_RICH_LAYOUT_METHODS)}. "
            f"Missing: {', '.join(f'{method}()' for method in missing)}."
        )
    return layout
