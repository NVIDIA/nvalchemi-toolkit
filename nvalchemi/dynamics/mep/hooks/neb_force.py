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

"""NEB force hook configuration."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Literal, Sequence

import torch
from torch import Tensor

from nvalchemi.dynamics.base import DynamicsStage
from nvalchemi.dynamics.paths.hooks.path_energy_stats import PathEnergyStatsHook
from nvalchemi.dynamics.paths.neb._ops.modes import ENDPOINT, REGULAR_NEB
from nvalchemi.dynamics.paths.neb._ops.registry import get_neb_method
from nvalchemi.dynamics.paths.neb._ops.torch_ops import neb_forces
from nvalchemi.dynamics.paths.neb.configs import (
    ConstantSpringConfig,
    NEBMethod,
    SpringConfig,
    SpringContext,
)
from nvalchemi.dynamics.paths.validate import validate_paths
from nvalchemi.hooks import DynamicsContext


@dataclass
class _NEBWorkspace:
    """Persistent buffers resolved for a NEB batch containing multiple
    paths.

    Attributes
    ----------
    image_ptr : Tensor
        Int32 atom offsets delimiting packed images, shape ``(num_images + 1,)``.
        Copied from the batch so the kernel receives its required index dtype.
    path_ptr : Tensor
        Int32 image offsets delimiting packed paths, shape ``(num_paths + 1,)``.
        Copied from the group layout for the kernel's required index dtype.
    image_path_idx : Tensor
        Int32 path index for each packed image, shape ``(num_images,)``.
        Copied from the group layout for the kernel's required index dtype and
        stable workspace ownership.
    link_source_image_idx : Tensor
        Image index owning each forward link, shape
        ``(num_images - num_paths,)``. The final image in every path is omitted.
    spring_constants : Tensor
        Per-link spring constants, shape ``(num_images - num_paths,)``.
    fixed_node_mask : Tensor
        Boolean mask of atoms that must not be updated, shape
        ``(num_atoms,)``.
    cell : Tensor
        Representative cell per path, shape ``(num_paths, 3, 3)``.
    inv_cell : Tensor
        Inverse representative cell per path, shape ``(num_paths, 3, 3)``.
    pbc : Tensor
        Periodic-boundary flags per path, shape ``(num_paths, 3)``.
    effective_forces : Tensor
        Per-atom output force buffer, shape ``(num_atoms, 3)``.
    link_lengths : Tensor
        Per-link distance buffer, shape ``(num_images - num_paths,)``.
    tangent_buffer : Tensor
        Per-atom tangent scratch buffer, shape ``(num_atoms, 3)``.
    """

    image_ptr: Tensor
    path_ptr: Tensor
    image_path_idx: Tensor
    link_source_image_idx: Tensor

    spring_constants: Tensor
    fixed_node_mask: Tensor
    cell: Tensor
    inv_cell: Tensor
    pbc: Tensor
    effective_forces: Tensor
    link_lengths: Tensor
    tangent_buffer: Tensor


def _resolve_neb_method(method: str | NEBMethod) -> NEBMethod:
    """Resolve a registered method name or validate a method configuration.

    Parameters
    ----------
    method : str or NEBMethod
        Registered built-in name or an explicit equation configuration.

    Returns
    -------
    NEBMethod
        The normalized method configuration.

    Raises
    ------
    TypeError
        If ``method`` is neither a string nor an :class:`NEBMethod`.
    ValueError
        If a string does not name a registered method.
    """
    if isinstance(method, NEBMethod):
        return method
    if not isinstance(method, str):
        raise TypeError(
            "method must be a registered method name or an NEBMethod; "
            f"got {type(method).__name__}"
        )
    get_neb_method(method)
    return NEBMethod(name=method)


class NEBForceHook:
    """Configure optimizer-facing forces for nudged elastic band dynamics.

    The hook prepares structural state at :attr:`DynamicsStage.ON_ADMISSION`,
    freezes constrained nodes in the currently active images before both
    optimizer update phases, then updates spring constants before replacing
    physical forces with the selected NEB force formulation at
    :attr:`DynamicsStage.AFTER_COMPUTE`. Energy extrema and image force modes
    are derived from hooks registered before this hook.

    Parameters
    ----------
    energy_stats_hook : PathEnergyStatsHook
        Shared path-energy statistics hook registered before this hook.
    spring : float or SpringConfig, optional
        Positive constant spring value or a policy that resolves one
        spring constant for every adjacent image pair. Default is ``0.1``.
    method : str or NEBMethod, optional
        Registered method name or a custom set of Warp equation functions.
        Default is ``"improved_tangent"``.
    endpoint_mode : {"fixed", "relaxed"}, optional
        Whether path endpoint images remain fixed or use their physical
        forces. Default is ``"fixed"``.
    fixed_atom_indices : sequence of int, optional
        Atom indices local to every image that must remain fixed. Atoms in
        fixed endpoint images are added automatically.

    Attributes
    ----------
    stage : DynamicsStage
        Primary hook stage, :attr:`DynamicsStage.AFTER_COMPUTE`. The hook also
        runs at :attr:`DynamicsStage.ON_ADMISSION` for batch preparation.
    frequency : int
        Always ``1``. NEB forces must be refreshed after every model-force
        evaluation.

    Examples
    --------
    Use the registered improved-tangent method with a constant spring::

        energy_stats = PathEnergyStatsHook()
        hook = NEBForceHook(
            energy_stats_hook=energy_stats,
            spring=0.1,
            method="improved_tangent",
        )

    Pass a custom equation configuration directly::

        method = NEBMethod(
            tangent_weights_fn=my_tangent_weights,
            effective_force_fn=my_effective_force,
            climbing_force_fn=my_climbing_force,
        )
        hook = NEBForceHook(energy_stats_hook=energy_stats, method=method)
    """

    stage = DynamicsStage.AFTER_COMPUTE
    frequency = 1

    def __init__(
        self,
        *,
        energy_stats_hook: PathEnergyStatsHook,
        spring: float | SpringConfig = 0.1,
        method: str | NEBMethod = "improved_tangent",
        endpoint_mode: Literal["fixed", "relaxed"] = "fixed",
        fixed_atom_indices: Sequence[int] | None = None,
    ) -> None:
        """Initialize the NEB force hook.

        Parameters
        ----------
        energy_stats_hook : PathEnergyStatsHook
            Shared path-energy statistics hook registered before this hook.
        spring : float or SpringConfig, optional
            Positive constant spring value or a policy that resolves one
            spring constant for every adjacent image pair.
        method : str or NEBMethod, optional
            Registered method name or a custom set of Warp equation functions.
        endpoint_mode : {"fixed", "relaxed"}, optional
            Whether path endpoint images are fixed or physically relaxed.
        fixed_atom_indices : sequence of int, optional
            Atom indices local to every image that must remain fixed. Default
            is ``None``.
        Raises
        ------
        TypeError
            If any argument has an unsupported type.
        ValueError
            If the spring is negative or the method name or endpoint mode is
            unknown.
        """
        if not isinstance(energy_stats_hook, PathEnergyStatsHook):
            raise TypeError("energy_stats_hook must be a PathEnergyStatsHook")

        # Normalize numeric springs into the common policy interface and reject
        # refresh stages that this multi-stage hook cannot handle.
        if isinstance(spring, (int, float)):
            spring = ConstantSpringConfig(spring)
        elif not isinstance(spring, SpringConfig):
            raise TypeError(
                "spring must be a number or implement SpringConfig.resolve; "
                f"got {type(spring).__name__}"
            )
        if not isinstance(spring.refresh, DynamicsStage):
            raise TypeError("SpringConfig.refresh must be a DynamicsStage")
        if spring.refresh not in {
            DynamicsStage.ON_ADMISSION,
            DynamicsStage.AFTER_COMPUTE,
        }:
            raise ValueError(
                "SpringConfig.refresh must be DynamicsStage.ON_ADMISSION or "
                "DynamicsStage.AFTER_COMPUTE"
            )

        # Resolve registered names during setup so invalid methods fail early.
        resolved_method = _resolve_neb_method(method)

        # Validate endpoint behavior and image-local atom constraints together.
        if endpoint_mode not in {"fixed", "relaxed"}:
            raise ValueError("endpoint_mode must be 'fixed' or 'relaxed'")
        if fixed_atom_indices is not None:
            if isinstance(fixed_atom_indices, (str, bytes)):
                raise TypeError("fixed_atom_indices must be a sequence of integers")
            fixed_atom_indices = tuple(fixed_atom_indices)
            if any(
                isinstance(index, bool) or not isinstance(index, int)
                for index in fixed_atom_indices
            ):
                raise TypeError("fixed_atom_indices must contain only integers")

        self.energy_stats_hook = energy_stats_hook
        self.spring = spring
        self.method = resolved_method
        self.endpoint_mode = endpoint_mode
        self.fixed_atom_indices = fixed_atom_indices
        self._workspace: _NEBWorkspace | None = None

    def on_register(self, workflow: object) -> None:
        """Require group-aware dynamics for NEB force construction.

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
                f"{type(self).__name__} requires dynamics configured with by_group=True"
            )

    def _runs_on_stage(self, stage: Enum) -> bool:
        """Return whether the hook participates in a dynamics stage.

        Parameters
        ----------
        stage : Enum
            Stage currently being dispatched by the hook registry.

        Returns
        -------
        bool
            Whether to run structural preparation or force construction.
        """
        return stage in {
            DynamicsStage.ON_ADMISSION,
            DynamicsStage.BEFORE_PRE_UPDATE,
            DynamicsStage.AFTER_COMPUTE,
            DynamicsStage.BEFORE_POST_UPDATE,
        }

    @torch.compiler.disable
    def _prepare_batch(self, ctx: DynamicsContext) -> None:
        """Validate and initialize private state for a new batch structure.

        This helper is reached only through the ``ON_ADMISSION`` branch of
        :meth:`__call__` and is typically called once for a batch.

        Parameters
        ----------
        ctx : DynamicsContext
            Context containing the newly admitted grouped path batch.

        Raises
        ------
        ValueError
            If the batch has no valid group layout, or if the paths, fixed atom
            indices, cells, or PBC flags are incompatible with NEB evaluation.
        """
        batch = ctx.batch
        validate_paths(batch)
        layout = batch.group_layout
        dtype = batch.positions.dtype
        device = batch.device

        # Each path's first and last images use endpoint force handling.
        last_rank: Tensor = layout.num_graphs_per_group[layout.group_idx] - 1
        is_terminal = layout.graph_rank == last_rank
        is_endpoint = (layout.graph_rank == 0) | is_terminal
        force_mode = torch.full_like(
            layout.group_idx,
            REGULAR_NEB,
            dtype=torch.int32,
        )
        force_mode[is_endpoint] = ENDPOINT

        # Identify fixed atoms including those in fixed endpoints.
        fixed_atoms = torch.zeros(
            batch.num_nodes,
            dtype=torch.bool,
            device=device,
        )
        if self.fixed_atom_indices:
            num_nodes_per_image = batch.num_nodes_per_graph
            for atom_index in self.fixed_atom_indices:
                if atom_index < 0 or torch.any(num_nodes_per_image <= atom_index):
                    raise ValueError(
                        "fixed_atom_indices must be valid for every image; "
                        f"got {atom_index}"
                    )
                fixed_atoms[batch.batch_ptr[:-1].long() + atom_index] = True
        if self.endpoint_mode == "fixed":
            fixed_atoms[is_endpoint[batch.batch_idx.long()]] = True

        # `validate_paths` guarantees cell and PBC consistency within each path,
        # so the kernels only need the first image's geometry. Missing cells use
        # dense identity matrices rather than zero-stride expanded views.
        cells = batch.cell if "cell" in batch else None
        pbcs = batch.pbc if "pbc" in batch else None
        # `path_start` is a list of graph indices that correspond to the first image of each graph
        path_start = layout.group_ptr[:-1]
        if cells is None:
            path_cell = (
                torch.eye(3, dtype=dtype, device=device)
                .unsqueeze(0)
                .repeat(layout.num_groups, 1, 1)
            )
        else:
            path_cell = cells[path_start].to(dtype=dtype).contiguous()
        if pbcs is None:
            path_pbc = torch.zeros(
                layout.num_groups, 3, dtype=torch.bool, device=device
            )
        else:
            path_pbc = pbcs[path_start].bool().contiguous()

        n_links = batch.num_graphs - layout.num_groups
        spring_constants = torch.empty(n_links, dtype=dtype, device=device)

        # Normalize kernel index buffers to int32.
        int32_max = torch.iinfo(torch.int32).max
        if batch.num_graphs > int32_max or batch.num_nodes > int32_max:
            raise ValueError("Path kernel layout exceeds int32 index capacity")
        image_ptr = batch.batch_ptr.to(torch.int32).contiguous()
        path_ptr = layout.group_ptr.to(torch.int32).contiguous()
        image_path_idx = layout.group_idx.to(torch.int32).contiguous()
        link_source_image_idx = torch.where(~is_terminal)[0].contiguous()

        # Include fields shared with other hooks on `Batch`.
        def set_batch_field(key: str, value: Tensor, level: str) -> None:
            if key in batch:
                target = getattr(batch, key)
                if (
                    target.shape == value.shape
                    and target.dtype == value.dtype
                    and target.device == value.device
                ):
                    target.copy_(value)
                    return
            values = (
                list(value.unbind())
                if level == "system"
                else list(value.split(batch.num_nodes_list))
            )
            batch.add_key(key, values, level=level, overwrite=key in batch)

        set_batch_field("force_mode", force_mode, "system")
        set_batch_field(
            "physical_forces",
            torch.zeros_like(batch.positions),
            "node",
        )
        set_batch_field(
            "forward_link_length",
            torch.zeros(batch.num_graphs, dtype=dtype, device=device),
            "system",
        )

        # Keep kernel-only masks and buffers private on workspace.
        self._workspace = _NEBWorkspace(
            spring_constants=spring_constants,
            fixed_node_mask=fixed_atoms,
            cell=path_cell,
            inv_cell=torch.linalg.inv(path_cell).contiguous(),
            pbc=path_pbc,
            effective_forces=torch.empty_like(batch.positions),
            link_lengths=torch.empty(n_links, dtype=dtype, device=device),
            tangent_buffer=torch.empty_like(batch.positions),
            image_ptr=image_ptr,
            path_ptr=path_ptr,
            image_path_idx=image_path_idx,
            link_source_image_idx=link_source_image_idx,
        )

        self._refresh_spring_constants(ctx, DynamicsStage.ON_ADMISSION)

    def __call__(self, ctx: DynamicsContext, stage: DynamicsStage) -> None:
        """Prepare NEB state, constrain nodes, or construct NEB forces.

        Parameters
        ----------
        ctx : DynamicsContext
            Context containing the current grouped path batch.
        stage : DynamicsStage
            Current dynamics stage. Preparation occurs at ``ON_ADMISSION``,
            fixed nodes are frozen at ``BEFORE_PRE_UPDATE`` and
            ``BEFORE_POST_UPDATE``, and force construction occurs at
            ``AFTER_COMPUTE``.

        Raises
        ------
        RuntimeError
            If the force phase runs before the batch has been prepared.
        """
        if stage == DynamicsStage.ON_ADMISSION:
            self._prepare_batch(ctx)
            return

        if stage in {
            DynamicsStage.BEFORE_PRE_UPDATE,
            DynamicsStage.BEFORE_POST_UPDATE,
        }:
            # Handle fixed atoms/nodes by zeroing out velocities and forces at
            # BEFORE_PRE_UPDATE and BEFORE_POST_UPDATE stages.
            batch = ctx.batch
            workspace = self._workspace
            active_fixed_nodes = workspace.fixed_node_mask
            if ctx.active_graph_mask is not None:
                active_nodes = ctx.active_graph_mask[batch.batch_idx.long()]
                active_fixed_nodes = active_fixed_nodes & active_nodes
            with torch.no_grad():
                batch.forces.masked_fill_(active_fixed_nodes.unsqueeze(-1), 0)
                velocities = getattr(batch, "velocities", None)
                if velocities is not None:
                    velocities.masked_fill_(active_fixed_nodes.unsqueeze(-1), 0)
            return

        if stage != DynamicsStage.AFTER_COMPUTE:
            return

        batch = ctx.batch
        workspace = self._workspace
        method_name = self.method.name
        active_nodes = (
            torch.ones_like(workspace.fixed_node_mask)
            if ctx.active_graph_mask is None
            else ctx.active_graph_mask[batch.batch_idx.long()]
        )
        active_fixed_nodes = active_nodes & workspace.fixed_node_mask
        stats = self.energy_stats_hook.get_stats()
        # Copy to `batch.physical_forces` before computation
        torch.where(
            active_nodes.unsqueeze(-1),
            batch.forces,
            batch.physical_forces,
            out=batch.physical_forces,
        )
        batch.physical_forces.masked_fill_(active_fixed_nodes.unsqueeze(-1), 0)
        self._refresh_spring_constants(ctx, DynamicsStage.AFTER_COMPUTE)
        neb_forces(
            positions=batch.positions.contiguous(),
            physical_forces=batch.physical_forces.contiguous(),
            image_energies=batch.energy.reshape(-1).contiguous(),
            image_ptr=workspace.image_ptr,
            path_ptr=workspace.path_ptr,
            image_path_idx=workspace.image_path_idx,
            spring_constants=workspace.spring_constants,
            image_force_mode=batch.force_mode,
            path_energy_ref=stats.endpoint_reference_energy.contiguous(),
            path_energy_max=stats.highest_interior_energy.contiguous(),
            cell=workspace.cell,
            inv_cell=workspace.inv_cell,
            pbc=workspace.pbc,
            method=method_name,
            tangent_buffer=workspace.tangent_buffer,
            effective_forces=workspace.effective_forces,
            link_lengths=workspace.link_lengths,
        )
        # Expand packed link lengths into the per-image field. Terminal entries
        # are initialized to zero on admission and are not modified here.
        batch.forward_link_length.index_copy_(
            0,
            workspace.link_source_image_idx,
            workspace.link_lengths,
        )
        # Populate `batch.forces` with newly computed NEB forces
        torch.where(
            active_nodes.unsqueeze(-1),
            workspace.effective_forces,
            batch.forces,
            out=batch.forces,
        )
        batch.forces.masked_fill_(active_fixed_nodes.unsqueeze(-1), 0)

        # Newly entered paths skip both optimizer updates while these NEB forces
        # are primed. Reset their carried velocity before the first real update.
        velocities = getattr(batch, "velocities", None)
        reprime_pending = getattr(batch, "reprime_pending", None)
        if velocities is not None and reprime_pending is not None:
            pending_graphs = reprime_pending.view(-1)[: batch.num_graphs]
            pending_nodes = active_nodes & pending_graphs[batch.batch_idx.long()]
            velocities.masked_fill_(pending_nodes.unsqueeze(-1), 0)

    def _spring_context(self, ctx: DynamicsContext) -> SpringContext:
        """Build the read-only spring-policy view of the current NEB stat
        instead of providing the full workspace that may contain irrelevant fields."""
        workspace = self._workspace
        batch = ctx.batch
        return SpringContext(
            energies=batch.energy.squeeze(-1),
            positions=batch.positions,
            physical_forces=batch.physical_forces,
            image_ptr=workspace.image_ptr,
            layout=batch.group_layout,
            cell=workspace.cell,
            inv_cell=workspace.inv_cell,
            pbc=workspace.pbc,
            step_count=ctx.step_count,
        )

    def _refresh_spring_constants(
        self,
        ctx: DynamicsContext,
        stage: DynamicsStage,
    ) -> None:
        """Resolve spring constants when requested by the configured policy."""
        workspace = self._workspace
        if self.spring.refresh is not stage:
            return

        resolved = self.spring.resolve(self._spring_context(ctx))
        if not isinstance(resolved, Tensor):
            raise TypeError("SpringConfig.resolve must return a Tensor")
        if resolved.shape != workspace.spring_constants.shape:
            raise ValueError(
                "SpringConfig.resolve must return shape "
                f"{tuple(workspace.spring_constants.shape)}"
            )
        workspace.spring_constants.copy_(resolved)
