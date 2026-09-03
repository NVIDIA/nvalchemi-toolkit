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

"""Climbing-image selection for nudged elastic band dynamics."""

from __future__ import annotations

from enum import Enum
from typing import Literal

import torch

from nvalchemi.dynamics.base import DynamicsStage
from nvalchemi.dynamics.paths.hooks.path_energy_stats import PathEnergyStatsHook
from nvalchemi.dynamics.paths.neb._ops.modes import CLIMBING_NEB, REGULAR_NEB
from nvalchemi.hooks import DynamicsContext


class ClimbingImageSelectionHook:
    """Select the highest-energy interior image in each NEB path.

    Register this hook after its shared :class:`PathEnergyStatsHook` and before
    :class:`NEBForceHook`. Fixed selection chooses a climber when each path first
    becomes active and retains it, while dynamic selection refreshes climbing
    images after every model evaluation.

    Parameters
    ----------
    energy_stats_hook : PathEnergyStatsHook
        Shared path-energy statistics hook registered before this hook.
    selection : {"fixed", "dynamic"}, optional
        Whether to retain the initial selection or refresh it after every model
        evaluation. Default is ``"fixed"``.
    frequency : int, optional
        Apply the hook every ``frequency`` dynamics steps. Default is ``1``.

    Attributes
    ----------
    stage : DynamicsStage
        Hook stage, :attr:`DynamicsStage.AFTER_COMPUTE`.

    Notes
    -----
    This hook requires dynamics configured with ``by_group=True``. Group-aware
    status transitions keep ``active_graph_mask`` uniform within each NEB path,
    so the graph-level mask can be applied directly.
    """

    stage = DynamicsStage.AFTER_COMPUTE

    def __init__(
        self,
        *,
        energy_stats_hook: PathEnergyStatsHook,
        selection: Literal["fixed", "dynamic"] = "fixed",
        frequency: int = 1,
    ) -> None:
        """Initialize the climbing-image selection hook.

        Parameters
        ----------
        energy_stats_hook : PathEnergyStatsHook
            Shared path-energy statistics hook registered before this hook.
        selection : {"fixed", "dynamic"}, optional
            Whether to retain the initial selection or refresh it after every
            model evaluation.
        frequency : int, optional
            Apply the hook every ``frequency`` dynamics steps.

        Raises
        ------
        TypeError
            If the statistics hook or frequency has an unsupported type.
        ValueError
            If the selection policy is unknown or frequency is not positive.
        """
        if not isinstance(energy_stats_hook, PathEnergyStatsHook):
            raise TypeError("energy_stats_hook must be a PathEnergyStatsHook")
        if selection not in {"fixed", "dynamic"}:
            raise ValueError("selection must be 'fixed' or 'dynamic'")
        if isinstance(frequency, bool) or not isinstance(frequency, int):
            raise TypeError(
                f"frequency must be an integer; got {type(frequency).__name__}"
            )
        if frequency < 1:
            raise ValueError("frequency must be positive")

        self.energy_stats_hook = energy_stats_hook
        self.selection = selection
        self.frequency = frequency
        self._initialized_graphs: torch.Tensor | None = None
        self._fixed_selection_complete = False

    def _runs_on_stage(self, stage: Enum) -> bool:
        """Return whether to initialize state or select climbing images."""
        return stage in {
            DynamicsStage.ON_ADMISSION,
            DynamicsStage.AFTER_COMPUTE,
        }

    def on_register(self, workflow: object) -> None:
        """Require group-aware dynamics for climbing-image selection.

        Parameters
        ----------
        workflow : object
            Workflow on which this hook is being registered.

        Raises
        ------
        ValueError
            If the workflow does not enable group-aware dynamics.
        """
        if not getattr(workflow, "by_group", False):
            raise ValueError(
                "ClimbingImageSelectionHook requires dynamics configured "
                "with by_group=True"
            )

    def __call__(self, ctx: DynamicsContext, stage: DynamicsStage) -> None:
        """Update image force modes from the current path-energy statistics."""
        if stage == DynamicsStage.ON_ADMISSION:
            if self.selection == "fixed":
                # Without an active mask, all paths initialize together. This flag lets
                # subsequent fixed-selection calls return immediately.
                self._fixed_selection_complete = False
                # In :class:`~nvalchemi.dynamics.FusedStage`, paths may become active on
                # different iterations. Track initialization per graph, while grouped dynamics
                # keeps this mask uniform within each path, avoiding a reduction and rebroadcast.
                self._initialized_graphs = torch.zeros(
                    ctx.batch.num_graphs,
                    dtype=torch.bool,
                    device=ctx.batch.device,
                )
            return
        if stage != DynamicsStage.AFTER_COMPUTE:
            return
        if (
            self.selection == "fixed"
            and ctx.active_graph_mask is None
            and self._fixed_selection_complete
        ):
            return

        batch = ctx.batch
        if "force_mode" not in batch:
            raise RuntimeError(
                "ClimbingImageSelectionHook requires NEB force-mode preparation. "
                "Dispatch NEBForceHook or a similar hook at ON_ADMISSION first"
            )
        force_mode = batch.force_mode
        active_mask = (
            torch.ones_like(force_mode, dtype=torch.bool)
            if ctx.active_graph_mask is None
            else ctx.active_graph_mask
        )

        if self.selection == "fixed":
            if self._initialized_graphs is None:
                raise RuntimeError(
                    "ClimbingImageSelectionHook must run at ON_ADMISSION before "
                    "fixed climbing-image selection"
                )
            # Select only active graphs whose fixed climber has not been initialized.
            active_mask = active_mask & ~self._initialized_graphs

        # Clear stale selections before this hook selects a climber.
        climbing_mask = force_mode == CLIMBING_NEB
        climbing_mask.logical_and_(active_mask)
        force_mode.masked_fill_(climbing_mask, REGULAR_NEB)

        # Mark the selected image for each path
        stats = self.energy_stats_hook.get_stats()
        selected_images = stats.highest_interior_image_idx.long()
        selected_graph_mask = torch.zeros_like(force_mode, dtype=torch.bool)
        selected_graph_mask.scatter_(
            0, selected_images, torch.ones_like(selected_images, dtype=torch.bool)
        )
        selected_graph_mask.logical_and_(active_mask)
        force_mode.masked_fill_(selected_graph_mask, CLIMBING_NEB)

        if self.selection == "fixed":
            self._initialized_graphs.logical_or_(active_mask)
            if ctx.active_graph_mask is None:
                self._fixed_selection_complete = True
