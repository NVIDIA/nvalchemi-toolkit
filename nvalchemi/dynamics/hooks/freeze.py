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
from nvalchemi.hooks._context import HookContext

__all__ = ["FreezeAtomsHook"]


class FreezeAtomsHook:
    """Freeze selected atoms during molecular dynamics simulation.

    During dynamics, certain atoms may need to remain fixed in place,
    such as substrate atoms in surface simulations, boundary atoms in
    slab models, or anchor atoms in constrained optimization. This
    hook identifies atoms by their ``atom_categories`` field and
    constrains them by:

    1. **Snapshotting positions** — At ``BEFORE_PRE_UPDATE``, the hook
       snapshots all atomic positions.
    2. **Restoring positions** — At ``AFTER_POST_UPDATE``, the hook
       restores positions of frozen atoms using ``torch.where``,
       effectively undoing any displacement applied by the integrator.
    3. **Zeroing velocities** — Velocities of frozen atoms are set to
       zero to prevent momentum accumulation.
    4. **Optionally zeroing forces** — By default, forces on frozen
       atoms are also zeroed. This prevents force contributions from
       propagating through the integrator and ensures clean energy
       conservation diagnostics.

    The hook fires at two stages: ``BEFORE_PRE_UPDATE`` (to snapshot
    positions) and ``AFTER_POST_UPDATE`` (to restore frozen positions
    and zero velocities/forces). This two-stage design enables
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
        Whether to zero forces on frozen atoms. Default ``True``.
        Set to ``False`` if you need to measure forces on frozen
        atoms for analysis purposes.

    Attributes
    ----------
    frequency : int
        Constraint application frequency in steps.
    freeze_category : int
        Category value identifying frozen atoms.
    zero_forces : bool
        Whether forces are zeroed on frozen atoms.
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
    * Fires at two stages: ``BEFORE_PRE_UPDATE`` (snapshot all positions)
      and ``AFTER_POST_UPDATE`` (restore frozen positions via ``torch.where``).
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
            {DynamicsStage.BEFORE_PRE_UPDATE, DynamicsStage.AFTER_POST_UPDATE}
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

    def _restore(self, batch: Batch) -> None:
        """Restore frozen atom positions and zero velocities/forces.

        The restore logic runs under :func:`torch.no_grad` because
        ``positions`` may carry ``requires_grad=True`` from the model's
        conservative-force computation.

        Parameters
        ----------
        batch : Batch
            The current batch of atomic data. ``batch.positions``,
            ``batch.velocities``, and optionally ``batch.forces`` are
            modified in-place.
        """
        with torch.no_grad():
            # mask shape: [V] -> [V, 1] for broadcasting with [V, 3]
            mask = (batch.atom_categories == self.freeze_category).unsqueeze(-1)
            zeros = torch.zeros_like(batch.positions)

            batch.positions.copy_(
                torch.where(mask, self._saved_positions, batch.positions)
            )
            batch.velocities.copy_(torch.where(mask, zeros, batch.velocities))
            if self.zero_forces:
                batch.forces.copy_(torch.where(mask, zeros, batch.forces))

    def __call__(self, ctx: HookContext, stage: Enum) -> None:
        """Snapshot or restore frozen atom positions."""
        if stage == DynamicsStage.BEFORE_PRE_UPDATE:
            self._saved_positions = ctx.batch.positions.clone()
        else:
            self._restore(ctx.batch)
