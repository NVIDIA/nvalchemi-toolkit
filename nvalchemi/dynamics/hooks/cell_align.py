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
lower-triangular (right-handed) form before the first optimizer step, and
:func:`_aligned_periodic`, the standalone alignment implementation it shares
with :meth:`~nvalchemi.dynamics.optimizers.lbfgs.LBFGSVariableCell._reference_cells`.
"""

from __future__ import annotations

from enum import Enum

import torch

from nvalchemi.dynamics._ops.cell_align import align_cell, cell_alignment_offenders
from nvalchemi.dynamics.base import DynamicsStage
from nvalchemi.hooks._context import DynamicsContext

__all__ = ["AlignCellHook"]


class AlignCellHook:
    r"""Align periodic cells before the first variable-cell FIRE2 or L-BFGS step.

    Transforms each periodic system's cell matrix to the standard
    lower-triangular (right-handed) form:

    .. math::

       H = \begin{bmatrix}
       a & 0 & 0 \\
       b\cos\gamma & b\sin\gamma & 0 \\
       c_1 & c_2 & c_3
       \end{bmatrix}

    and rotates positions to preserve fractional coordinates.  This
    representation reduces rotational ambiguity (improving optimizer
    stability) and has 6 independent parameters instead of 9.

    ``forces`` and ``stress``, if present on the batch, are rotated by the
    same transform.  This matters because :class:`~nvalchemi.dynamics.base.BaseDynamics`
    primes forces/stress (one model call) before ``BEFORE_STEP`` hooks run,
    so on the first step of a new admission this hook would otherwise rotate
    positions/cell into a new frame while leaving already-computed
    forces/stress in the old one.

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

    def __call__(self, ctx: DynamicsContext, stage: Enum) -> None:
        """Align the current batch when any periodic cell is not triangular."""
        del stage
        aligned = _aligned_periodic(ctx.batch, ctx.active_graph_mask)
        if aligned is not None:
            positions, cell, forces, stress = aligned
            with torch.no_grad():
                ctx.batch.positions.copy_(positions)
                ctx.batch.cell.copy_(cell)
                # BaseDynamics primes forces/stress via a model call before
                # BEFORE_STEP hooks run (base.py), i.e. before this hook
                # rotates positions/cell on the first step of an admission.
                # Rotate them the same way so pre_update doesn't mix a
                # pre-rotation force/stress with post-rotation positions.
                if forces is not None:
                    ctx.batch.forces.copy_(forces)
                if stress is not None:
                    ctx.batch.stress.copy_(stress)


def _aligned_periodic(
    batch, active_graph_mask: torch.Tensor | None = None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None] | None:
    """Return ``(positions, cell, forces, stress)`` with periodic systems aligned.

    ``forces``/``stress`` are rotated by the same transform as ``positions``/
    ``cell`` (``None`` when *batch* doesn't carry that field) so a caller that
    already has forces/stress computed in the old frame — e.g. because forces
    were primed before this hook ran — can keep them consistent with the
    newly-aligned frame instead of silently mixing the two.  Reads *batch*
    without writing it; other systems keep their values.  Returns ``None``
    when there is nothing to align.

    Shared by :class:`AlignCellHook` (which copies the result back into the
    live batch) and
    :meth:`~nvalchemi.dynamics.optimizers.lbfgs.LBFGSVariableCell._reference_cells`
    (which only needs the aligned cell to validate/seed its reference chart,
    without touching *batch*) — kept standalone rather than inlined into the
    hook so both call sites share one alignment implementation.
    """
    if getattr(batch, "cell", None) is None or getattr(batch, "pbc", None) is None:
        return None
    cell = batch.cell.detach()
    if cell.shape[0] == 0:
        return None

    # Determine which systems are periodic
    pbc = batch.pbc
    if pbc.dim() == 1:
        if pbc.shape[0] == batch.num_graphs:
            periodic_mask = pbc.to(dtype=torch.bool)
        else:
            periodic_mask = pbc.unsqueeze(0).any(dim=-1)
    else:
        periodic_mask = pbc.any(dim=-1)
    if active_graph_mask is not None:
        periodic_mask = periodic_mask & active_graph_mask

    # Cheap per-system check on the *unmodified* cell, before cloning
    # anything or launching the Warp kernel.  ``AlignCellHook`` must run
    # every step for ``LBFGSVariableCell`` (frequency=1), and by the second
    # step every active periodic cell is typically already aligned — the
    # common case is "nothing to do", not "some systems are periodic".
    # ``cell_alignment_offenders`` is the same criterion
    # ``LBFGSVariableCell._reference_cells`` validates admitted cells
    # against, so a cell this considers already-aligned can never then fail
    # that check, and vice versa.  Skipping the clone + kernel launch for
    # systems that don't need it also avoids nudging already-aligned
    # positions by ulp-level amounts every step, which would otherwise
    # violate L-BFGS's "don't edit positions between steps" contract.
    needs_align = periodic_mask & cell_alignment_offenders(cell)
    # Eager early exit; compiled graphs run branchless (all-False is a no-op).
    if not torch.compiler.is_compiling() and not needs_align.any():
        return None

    positions_dtype = batch.positions.dtype
    if positions_dtype not in (torch.float32, torch.float64):
        raise TypeError(
            "Cell alignment only supports float32/float64 positions, got "
            f"{positions_dtype}."
        )
    positions = batch.positions.detach().contiguous().clone()
    cell = cell.contiguous().clone()
    if cell.dtype != positions_dtype:
        cell = cell.to(dtype=positions_dtype)

    batch_idx = batch.batch_idx.to(dtype=torch.int32).contiguous()
    transform = align_cell(positions, cell, batch_idx)

    # Blend by *needs_align*, not just *periodic_mask*: align_cell ran on
    # every periodic+active system's cell above (the kernel has no
    # per-system skip of its own), so a system that was already aligned got
    # recomputed too and can differ from the input by a few ULP of rounding.
    # Systems that didn't need realignment must come back bit-identical to
    # the input, not that rounded recomputation.
    aligned_atoms = needs_align[batch.batch_idx].unsqueeze(-1)
    aligned_cells = needs_align[:, None, None]

    forces = None
    batch_forces = getattr(batch, "forces", None)
    if batch_forces is not None:
        rotated = torch.einsum(
            "nij,nj->ni", transform[batch.batch_idx.long()], batch_forces.detach()
        )
        forces = torch.where(aligned_atoms, rotated, batch_forces.detach())

    stress = None
    batch_stress = getattr(batch, "stress", None)
    if batch_stress is not None:
        # Cauchy stress is a rank-2 Cartesian tensor: sigma' = R sigma R^T.
        rotated = torch.einsum(
            "mij,mjk,mlk->mil", transform, batch_stress.detach(), transform
        )
        stress = torch.where(aligned_cells, rotated, batch_stress.detach())

    return (
        torch.where(aligned_atoms, positions, batch.positions.detach()),
        torch.where(aligned_cells, cell, batch.cell.detach()),
        forces,
        stress,
    )
