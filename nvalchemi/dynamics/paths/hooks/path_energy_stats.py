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

"""Per-path energy statistics shared by reaction-path dynamics methods."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import torch
from torch import Tensor

from nvalchemi.dynamics.base import DynamicsStage
from nvalchemi.dynamics.paths._ops.torch_paths import path_energy_stats
from nvalchemi.dynamics.paths.validate import validate_paths
from nvalchemi.hooks import DynamicsContext


@dataclass(frozen=True)
class PathEnergyStats:
    """Preallocated energy statistics for a grouped path batch.

    Attributes
    ----------
    endpoint_reference_energy : Tensor
        Smaller endpoint energy for each path, shape ``(num_paths,)``.
    highest_interior_energy : Tensor
        Maximum interior-image energy for each path, shape ``(num_paths,)``.
    highest_interior_image_idx : Tensor
        Packed image index attaining ``highest_interior_energy`` for each path,
        shape ``(num_paths,)``. The tensor has dtype ``torch.int32``. Ties
        select the first interior image.
    """

    endpoint_reference_energy: Tensor
    highest_interior_energy: Tensor
    highest_interior_image_idx: Tensor


class PathEnergyStatsHook:
    """Compute reusable energy statistics for each path after model evaluation.

    Register this hook before other hooks such as climbing-image selection,
    path-force construction, and diagnostics. It prepares persistent buffers at
    :attr:`DynamicsStage.ON_ADMISSION` and refreshes them from ``batch.energy``
    at :attr:`DynamicsStage.AFTER_COMPUTE`.

    Downstream hooks should call :meth:`get_stats` after this hook has run for the
    current evaluation. Correct values depend on hook registration order.

    The hook prepares its own int32 path offsets during ``ON_ADMISSION`` so it
    can be used independently of path-force hooks.
    """

    # Multi-stage dispatch is defined by `_runs_on_stage()`.
    stage = None
    frequency = 1

    def __init__(self) -> None:
        """Initialize an unprepared path-energy statistics hook."""
        self._stats: PathEnergyStats | None = None
        self._path_ptr: Tensor

    def _runs_on_stage(self, stage: Enum) -> bool:
        """Return whether to prepare or refresh path-energy statistics."""
        return stage in {
            DynamicsStage.ON_ADMISSION,
            DynamicsStage.AFTER_COMPUTE,
        }

    @torch.compiler.disable
    def _prepare_batch(self, ctx: DynamicsContext) -> None:
        """Validate paths and allocate per-path output buffers.

        Compilation is disabled because validation, shape-dependent tensor
        allocation, and mutation of hook-owned Python state introduce graph
        breaks. This path runs only during admission.
        """
        batch = ctx.batch
        validate_paths(batch)
        num_paths = batch.group_layout.num_groups
        dtype = batch.positions.dtype
        device = batch.device
        self._stats = PathEnergyStats(
            endpoint_reference_energy=torch.empty(
                num_paths, dtype=dtype, device=device
            ),
            highest_interior_energy=torch.empty(num_paths, dtype=dtype, device=device),
            highest_interior_image_idx=torch.empty(
                num_paths, dtype=torch.int32, device=device
            ),
        )
        self._path_ptr = batch.group_layout.group_ptr.to(torch.int32).contiguous()

    def get_stats(self) -> PathEnergyStats:
        """Return statistics computed for the current energy evaluation.

        Returns
        -------
        PathEnergyStats
            Current per-path energy statistics.

        Raises
        ------
        RuntimeError
            If the hook has not prepared its buffers at ``ON_ADMISSION``.
        """
        if self._stats is None:
            raise RuntimeError(
                "PathEnergyStatsHook must run at ON_ADMISSION before get_stats"
            )
        return self._stats

    def __call__(self, ctx: DynamicsContext, stage: DynamicsStage) -> None:
        """Prepare buffers or refresh them based on the current image energies.

        This hook intentionally ignores ``ctx.active_graph_mask`` because it only
        computes statistics on energies. Downstream consumers can separately enforce
        active-path ownership when mutating dynamics state.
        """
        batch = ctx.batch
        if stage == DynamicsStage.ON_ADMISSION:
            self._prepare_batch(ctx)
            return
        if stage != DynamicsStage.AFTER_COMPUTE:
            return

        stats = self._stats
        if stats is None:
            raise RuntimeError(
                "PathEnergyStatsHook must run at ON_ADMISSION before AFTER_COMPUTE"
            )
        path_ptr = self._path_ptr
        path_energy_stats(
            batch.energy.squeeze(-1).contiguous(),
            path_ptr,
            stats.endpoint_reference_energy,
            stats.highest_interior_energy,
            stats.highest_interior_image_idx,
        )
