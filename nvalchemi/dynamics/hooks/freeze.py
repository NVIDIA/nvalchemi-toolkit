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
"""
Freeze atoms hook for constraining selected atoms during dynamics.

Provides :class:`FreezeAtomsHook`, which freezes atoms by category,
restoring their positions and zeroing velocities each step.
"""

from __future__ import annotations

from enum import Enum

import torch

from nvalchemi._typing import AtomCategory
from nvalchemi.data import Batch
from nvalchemi.dynamics.base import DynamicsStage
from nvalchemi.hooks._context import DynamicsContext

__all__ = ["FreezeAtomsHook"]


class FreezeAtomsHook:
    """Freeze selected atoms during molecular dynamics simulation.

    During dynamics, certain atoms may need to remain fixed in place,
    such as substrate atoms in surface simulations, boundary atoms in
    slab models, or anchor atoms in constrained optimization. This
    hook identifies atoms by their ``atom_categories`` field and
    constrains them by:

    1. **Snapshotting and clearing state** — At ``BEFORE_PRE_UPDATE``, the hook
       snapshots all atomic positions and clears frozen atoms' velocities and
       forces before the integrator uses them.
    2. **Restoring before compute** — At ``AFTER_PRE_UPDATE``, the hook
       restores frozen atoms' positions and clears their velocities so
       compute-preparation hooks and the model see the constrained geometry.
    3. **Clearing new forces** — At ``BEFORE_POST_UPDATE``, the hook clears
       frozen atoms' forces newly computed by the model before the integrator's
       second update.
    4. **Restoring positions** — At ``AFTER_POST_UPDATE``, the hook
       restores positions of frozen atoms using ``torch.where``,
       effectively undoing any displacement applied by the integrator.
    5. **Zeroing velocities** — Velocities of frozen atoms are set to
       zero to prevent momentum accumulation.
    6. **Optionally exposing forces** — When ``zero_forces=False``, raw
       model forces remain available at ``AFTER_COMPUTE`` for reporting.
       However, they are still cleared at ``BEFORE_POST_UPDATE`` so they
       cannot propagate through the integrator.

    The hook fires at five stages: ``BEFORE_PRE_UPDATE`` (to snapshot
    positions and clear prior state), ``AFTER_PRE_UPDATE`` (to restore the
    constrained geometry before compute preparation), ``AFTER_COMPUTE``
    (optionally clear model forces), ``BEFORE_POST_UPDATE`` (to clear newly
    computed forces), and ``AFTER_POST_UPDATE`` (to restore frozen positions
    and clear velocities). This multi-stage design enables
    ``torch.compile(fullgraph=True)`` compatibility by avoiding
    data-dependent branching.

    Parameters
    ----------
    frequency : int, optional
        Apply constraints every ``frequency`` steps. Default ``1``
        (every step). Setting this higher than 1 is not recommended
        as frozen atoms will drift between constraint applications.
    freeze_category : int, optional
        The ``atom_categories`` value that identifies frozen atoms.
        Default is ``AtomCategory.SPECIAL.value`` (-1). Atoms with
        ``batch.atom_categories == freeze_category`` will be frozen.
    zero_forces : bool, optional
        Whether to zero forces immediately at ``AFTER_COMPUTE``.
        Default ``True``. Set to ``False`` to inspect raw frozen-atom
        forces in an ``AFTER_COMPUTE`` hook. Note that forces are always
        cleared before ``post_update``.

    Attributes
    ----------
    frequency : int
        Constraint application frequency in steps.
    freeze_category : int
        Category value identifying frozen atoms.
    zero_forces : bool
        Whether forces are zeroed immediately after compute.
    stage : DynamicsStage
        Primary stage, set to ``BEFORE_PRE_UPDATE`` for protocol compliance.

    Examples
    --------
    Freeze atoms marked as SPECIAL (default):

    >>> from nvalchemi.dynamics.hooks import FreezeAtomsHook
    >>> hook = FreezeAtomsHook()
    >>> dynamics = DemoDynamics(model=model, n_steps=1000, hooks=[hook])
    >>> dynamics.run(batch)

    Freeze bulk atoms instead:

    >>> from nvalchemi._typing import AtomCategory
    >>> hook = FreezeAtomsHook(freeze_category=AtomCategory.BULK.value)

    Keep forces for analysis:

    >>> hook = FreezeAtomsHook(zero_forces=False)

    Notes
    -----
    * Fires at ``BEFORE_PRE_UPDATE`` (snapshot positions and clear prior
      forces/velocities), ``AFTER_PRE_UPDATE`` (restore positions before
      compute preparation), ``AFTER_COMPUTE`` (optionally expose new forces),
      ``BEFORE_POST_UPDATE`` (clear new forces), and ``AFTER_POST_UPDATE``
      (restore positions and clear velocities).
    * Uses ``torch.where`` for branchless GPU-vectorized restore, enabling
      ``torch.compile(fullgraph=True)`` compatibility.
    * All positions are snapshotted each step (not just frozen ones) to
      avoid shape-dependent logic.
    * When using with :class:`WrapPeriodicHook`, both hooks fire at
      ``AFTER_POST_UPDATE``. Registration order determines execution order;
      register this hook **before** the periodic wrapping hook to ensure
      frozen positions are restored before wrapping is applied.
    """

    def __init__(
        self,
        frequency: int = 1,
        freeze_category: int = AtomCategory.SPECIAL.value,
        zero_forces: bool = True,
        stage: Enum = DynamicsStage.BEFORE_PRE_UPDATE,
    ) -> None:
        self.frequency = frequency
        self.freeze_category = freeze_category
        self.zero_forces = zero_forces
        self.stage = stage
        # Multi-stage hooks need both a primary stage and a list of active stages
        self._active_stages = frozenset(
            {
                DynamicsStage.BEFORE_PRE_UPDATE,
                DynamicsStage.AFTER_PRE_UPDATE,
                DynamicsStage.AFTER_COMPUTE,
                DynamicsStage.BEFORE_POST_UPDATE,
                DynamicsStage.AFTER_POST_UPDATE,
            }
        )
        self._saved_positions: torch.Tensor | None = None

    def _runs_on_stage(self, stage: Enum) -> bool:
        """Check if this hook should run on the given stage.

        Parameters
        ----------
        stage : Enum
            The stage to check.

        Returns
        -------
        bool
            True if this hook runs on the given stage.
        """
        return stage in self._active_stages

    def _mask(
        self, batch: Batch, active_graph_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Return the active frozen-atom mask with a coordinate dimension."""
        # mask shape: [V] -> [V, 1] for broadcasting with [V, 3]
        mask = batch.atom_categories == self.freeze_category
        if active_graph_mask is not None:
            mask = mask & active_graph_mask[batch.batch_idx]
        return mask.unsqueeze(-1)

    def _clear_forces(
        self, batch: Batch, active_graph_mask: torch.Tensor | None = None
    ) -> None:
        """Clear forces on active frozen atoms before an integrator update."""
        with torch.no_grad():
            mask = self._mask(batch, active_graph_mask)
            batch.forces.copy_(torch.where(mask, 0, batch.forces))

    def _restore(
        self, batch: Batch, active_graph_mask: torch.Tensor | None = None
    ) -> None:
        """Restore frozen positions and clear frozen velocities."""
        with torch.no_grad():
            mask = self._mask(batch, active_graph_mask)
            batch.positions.copy_(
                torch.where(mask, self._saved_positions, batch.positions)
            )
            batch.velocities.copy_(torch.where(mask, 0, batch.velocities))

    def __call__(self, ctx: DynamicsContext, stage: Enum) -> None:
        """Apply the stage-specific frozen-atom constraint."""
        if stage == DynamicsStage.BEFORE_PRE_UPDATE:
            self._saved_positions = ctx.batch.positions.clone()
            self._clear_forces(ctx.batch, ctx.active_graph_mask)
            self._restore(ctx.batch, ctx.active_graph_mask)
        elif stage == DynamicsStage.AFTER_PRE_UPDATE:
            self._restore(ctx.batch, ctx.active_graph_mask)
        elif stage == DynamicsStage.AFTER_COMPUTE:
            if self.zero_forces:
                self._clear_forces(ctx.batch, ctx.active_graph_mask)
        elif stage == DynamicsStage.BEFORE_POST_UPDATE:
            self._clear_forces(ctx.batch, ctx.active_graph_mask)
        else:
            self._restore(ctx.batch, ctx.active_graph_mask)
