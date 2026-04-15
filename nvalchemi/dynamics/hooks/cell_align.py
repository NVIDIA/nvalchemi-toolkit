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
Cell alignment hook for variable-cell optimization.

Provides :class:`AlignCellHook`, which aligns periodic simulation cells to
upper-triangular (right-handed) form before the first optimizer step, and
:func:`_align_atomic_data_cell`, a standalone utility for aligning a single
:class:`~nvalchemi.data.AtomicData` instance.
"""

from __future__ import annotations

from enum import Enum

import torch

from nvalchemi.dynamics._ops.cell_align import align_cell
from nvalchemi.dynamics.base import DynamicsStage
from nvalchemi.hooks._context import HookContext

__all__ = ["AlignCellHook"]


class AlignCellHook:
    """Align periodic cells before the first variable-cell FIRE2 step.

    Transforms each periodic system's cell matrix to the standard
    upper-triangular (right-handed) form::

        H = [[a,         0,    0],
             [b*cos(γ),  b*sin(γ), 0],
             [c1,        c2,   c3]]

    and rotates positions to preserve fractional coordinates.  This
    representation reduces rotational ambiguity (improving optimizer
    stability) and has 6 independent parameters instead of 9.

    The hook fires at :attr:`~DynamicsStage.BEFORE_STEP` and skips
    non-periodic systems.

    Parameters
    ----------
    frequency : int, optional
        Run every ``frequency`` steps.  Default ``1``.

    Attributes
    ----------
    frequency : int
        Hook execution frequency.
    stage : DynamicsStage
        Always :attr:`DynamicsStage.BEFORE_STEP`.

    Examples
    --------
    >>> from nvalchemi.dynamics.hooks import AlignCellHook
    >>> hook = AlignCellHook()
    >>> optimizer = FIRE2VariableCell(model=model, hooks=[hook])
    """

    def __init__(self, frequency: int = 1) -> None:
        self.stage = DynamicsStage.BEFORE_STEP
        self.frequency = frequency

    def __call__(self, ctx: HookContext, stage: Enum) -> None:
        """Align the current batch when any periodic cell is not triangular."""
        del stage
        batch = ctx.batch

        if not hasattr(batch, "cell") or batch.cell is None:
            return
        if not hasattr(batch, "pbc") or batch.pbc is None:
            return

        # Determine which systems are periodic
        pbc = batch.pbc
        if pbc.dim() == 1:
            if pbc.shape[0] == batch.num_graphs:
                periodic_mask = pbc.to(dtype=torch.bool)
            else:
                periodic_mask = pbc.unsqueeze(0).any(dim=-1)
        else:
            periodic_mask = pbc.any(dim=-1)
        if not periodic_mask.any():
            return

        positions = batch.positions.contiguous()
        cell = batch.cell.contiguous()
        if positions.dtype not in (torch.float32, torch.float64):
            raise TypeError(
                "Cell alignment only supports float32/float64 positions, got "
                f"{positions.dtype}."
            )
        if cell.dtype != positions.dtype:
            cell = cell.to(dtype=positions.dtype)

        batch_idx = batch.batch_idx.to(dtype=torch.int32).contiguous()
        align_cell(positions, cell, batch_idx)

        batch.positions = positions
        batch.cell = cell
