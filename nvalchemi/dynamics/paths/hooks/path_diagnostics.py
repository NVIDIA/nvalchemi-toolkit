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

"""Per-path diagnostics for reaction-path dynamics methods."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import torch
from torch import Tensor

from nvalchemi.dynamics.base import DynamicsStage
from nvalchemi.dynamics.paths.hooks.path_energy_stats import PathEnergyStatsHook
from nvalchemi.dynamics.paths.validate import validate_paths
from nvalchemi.hooks import DynamicsContext


@dataclass(frozen=True)
class PathDiagnostics:
    """Preallocated diagnostics for a grouped reaction-path batch.

    Attributes
    ----------
    fmax : Tensor
        Maximum effective-force norm over all atoms in all images of each path,
        shape (num_paths,).
    energy_barrier : Tensor
        Highest interior energy minus the lower endpoint energy for each path,
        shape (num_paths,).
    highest_interior_image_idx : Tensor
        Zero-based path-local index of the highest-energy interior image for
        each path, shape (num_paths,) and dtype torch.int32.
    path_length : Tensor
        Sum of minimum-image forward-link lengths for each path, shape
        (num_paths,).
    """

    fmax: Tensor
    energy_barrier: Tensor
    highest_interior_image_idx: Tensor
    path_length: Tensor


class PathDiagnosticsHook:
    """Compute and cache one diagnostics record per reaction path.

    A reaction path is represented by one contiguous group of graphs in
    ``batch.group_layout``. Each graph is one path image, ordered from the
    initial endpoint to the final endpoint. Custom reaction-path methods can
    use this hook without subclassing it by satisfying the input and
    registration contracts below.

    Parameters
    ----------
    energy_stats_hook : PathEnergyStatsHook
        Shared path-energy statistics hook registered before this hook.

    Required batch fields
    ---------------------
    ``batch.group_idx``
        One group index per image, shape ``(num_images,)``. Call
        ``batch.set_group_layout(...)`` before running the workflow. Images
        belonging to one path must be contiguous and ordered along that path.

    ``batch.energy``
        Potential energy for every image. This must be current before the
        shared :class:`PathEnergyStatsHook` runs.

    ``batch.forces``
        Effective optimization force for every atom, shape ``(num_atoms, 3)``.
        This is the force whose norm contributes to the reported path
        ``fmax``; it need not be the model's unmodified physical force.

    ``batch.forward_link_length``
        Distance from each image to the next image in the same path, shape
        ``(num_images,)``. Values must use the same floating-point dtype and
        device as ``batch.positions``. The final image of every path has no
        successor and must therefore have a link length of zero.

        A method-specific force hook must create this field before this hook
        runs at ``ON_ADMISSION`` and refresh it before this hook runs at
        ``AFTER_COMPUTE``.

    Hook ordering
    -------------
    At ``AFTER_COMPUTE``, hooks must be registered in this order:

    1. :class:`PathEnergyStatsHook`
    2. The method-specific effective-force and link-length hook
    3. :class:`PathDiagnosticsHook`

    This hook reduces image- and atom-cardinality inputs to one value per path.
    Downstream consumers should be registered after it and retrieve the current
    values with :meth:`get_diagnostics`.

    Active-path behavior
    --------------------
    Activity must be consistent across every image in a path. Diagnostics for
    inactive paths retain their last valid values. Newly admitted paths contain
    NaN floating-point diagnostics and a highest-interior-image index of ``-1`` until
    their first active evaluation.
    """

    stage = None
    frequency = 1

    def __init__(self, *, energy_stats_hook: PathEnergyStatsHook) -> None:
        """Initialize an unprepared path diagnostics hook."""
        if not isinstance(energy_stats_hook, PathEnergyStatsHook):
            raise TypeError("energy_stats_hook must be a PathEnergyStatsHook")
        self.energy_stats_hook = energy_stats_hook
        self._diagnostics: PathDiagnostics | None = None

    def on_register(self, workflow: object) -> None:
        """Require group-aware dynamics for per-path reductions."""
        if not getattr(workflow, "by_group", False):
            raise ValueError(
                f"{type(self).__name__} requires dynamics configured with by_group=True"
            )

    def _runs_on_stage(self, stage: Enum) -> bool:
        """Return whether to prepare or refresh path diagnostics."""
        return stage in {
            DynamicsStage.ON_ADMISSION,
            DynamicsStage.AFTER_COMPUTE,
        }

    @torch.compiler.disable
    def _prepare_batch(self, ctx: DynamicsContext) -> None:
        """Validate inputs and allocate persistent per-path reductions."""
        batch = ctx.batch
        validate_paths(batch)
        link_lengths = getattr(batch, "forward_link_length", None)
        if not isinstance(link_lengths, Tensor):
            raise RuntimeError(
                "PathDiagnosticsHook requires batch.forward_link_length to be "
                "published before it runs at ON_ADMISSION"
            )
        if link_lengths.shape != (batch.num_graphs,):
            raise ValueError(
                f"batch.forward_link_length must have shape ({batch.num_graphs},)"
            )
        if (
            link_lengths.dtype != batch.positions.dtype
            or link_lengths.device != batch.positions.device
        ):
            raise ValueError(
                "batch.forward_link_length must match positions dtype and device"
            )

        layout = batch.group_layout
        num_paths = layout.num_groups
        dtype = batch.positions.dtype
        device = batch.positions.device

        self._diagnostics = PathDiagnostics(
            fmax=torch.full((num_paths,), torch.nan, dtype=dtype, device=device),
            energy_barrier=torch.full(
                (num_paths,), torch.nan, dtype=dtype, device=device
            ),
            highest_interior_image_idx=torch.full(
                (num_paths,), -1, dtype=torch.int32, device=device
            ),
            path_length=torch.full((num_paths,), torch.nan, dtype=dtype, device=device),
        )
        self._node_to_path = layout.node_to_group.long().contiguous()
        self._image_to_path = layout.group_idx.long().contiguous()
        self._path_start_index = layout.group_ptr[:-1].long().contiguous()
        self._path_start = self._path_start_index.to(torch.int32)
        self._active_paths = torch.empty(num_paths, dtype=torch.bool, device=device)
        self._force_norms = torch.empty(batch.num_nodes, dtype=dtype, device=device)
        self._candidate_fmax = torch.empty(num_paths, dtype=dtype, device=device)
        self._candidate_energy_barrier = torch.empty(
            num_paths, dtype=dtype, device=device
        )
        self._candidate_path_length = torch.empty(num_paths, dtype=dtype, device=device)
        self._candidate_highest_interior_image_idx = torch.empty(
            num_paths, dtype=torch.int32, device=device
        )

    def get_diagnostics(self) -> PathDiagnostics:
        """Return diagnostics refreshed for the current path evaluation.

        Raises
        ------
        RuntimeError
            If the hook has not prepared its buffers at ``ON_ADMISSION``.
        """
        if self._diagnostics is None:
            raise RuntimeError(
                "PathDiagnosticsHook must run at ON_ADMISSION before get_diagnostics"
            )
        return self._diagnostics

    def __call__(self, ctx: DynamicsContext, stage: DynamicsStage) -> None:
        """Prepare buffers or refresh diagnostics from current path state."""
        if stage == DynamicsStage.ON_ADMISSION:
            self._prepare_batch(ctx)
            return
        if stage != DynamicsStage.AFTER_COMPUTE:
            return
        if self._diagnostics is None:
            raise RuntimeError(
                "PathDiagnosticsHook must run at ON_ADMISSION before AFTER_COMPUTE"
            )

        batch = ctx.batch
        stats = self.energy_stats_hook.get_stats()
        link_lengths = getattr(batch, "forward_link_length", None)
        if not isinstance(link_lengths, Tensor):
            raise RuntimeError(
                "PathDiagnosticsHook requires batch.forward_link_length; "
                "run a compatible path-force hook before this hook"
            )

        torch.linalg.vector_norm(batch.forces, dim=-1, out=self._force_norms)
        self._candidate_fmax.zero_()
        self._candidate_fmax.scatter_reduce_(
            0,
            self._node_to_path,
            self._force_norms,
            reduce="amax",
            include_self=True,
        )
        self._candidate_path_length.zero_()
        self._candidate_path_length.scatter_add_(
            0,
            self._image_to_path,
            link_lengths,
        )
        torch.sub(
            stats.highest_interior_energy,
            stats.endpoint_reference_energy,
            out=self._candidate_energy_barrier,
        )
        torch.sub(
            stats.highest_interior_image_idx,
            self._path_start,
            out=self._candidate_highest_interior_image_idx,
        )

        # Group-aware dynamics keeps `ctx.active_graph_mask` uniform within each path,
        # so the first image's mask represents the whole path.
        active_paths = self._active_paths
        if ctx.active_graph_mask is None:
            active_paths.fill_(True)
        else:
            torch.index_select(
                ctx.active_graph_mask,
                0,
                self._path_start_index,
                out=active_paths,
            )
        diagnostics = self._diagnostics
        torch.where(
            active_paths,
            self._candidate_fmax,
            diagnostics.fmax,
            out=diagnostics.fmax,
        )
        torch.where(
            active_paths,
            self._candidate_energy_barrier,
            diagnostics.energy_barrier,
            out=diagnostics.energy_barrier,
        )
        torch.where(
            active_paths,
            self._candidate_path_length,
            diagnostics.path_length,
            out=diagnostics.path_length,
        )
        torch.where(
            active_paths,
            self._candidate_highest_interior_image_idx,
            diagnostics.highest_interior_image_idx,
            out=diagnostics.highest_interior_image_idx,
        )
