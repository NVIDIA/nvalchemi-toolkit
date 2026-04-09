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
Periodic boundary condition hook for coordinate wrapping.

Provides :class:`WrapPeriodicHook`, which wraps atomic positions back
into the unit cell under periodic boundary conditions.
"""

from __future__ import annotations

from enum import Enum

import torch
import warp as wp
from jaxtyping import Float
from nvalchemiops.dynamics.utils import compute_cell_inverse, wrap_positions_to_cell

from nvalchemi.data import Batch
from nvalchemi.hooks._context import HookContext

__all__ = ["WrapPeriodicHook"]


# ---------------------------------------------------------------------------
# Custom op + helper for position wrapping (moved from dynamics/hooks/_utils)
# ---------------------------------------------------------------------------


@torch.library.custom_op("nvalchemi_hooks::wrap_positions", mutates_args=())
def _wrap_positions(
    positions: torch.Tensor,
    cell: torch.Tensor,
    batch_idx: torch.Tensor,
) -> torch.Tensor:
    vec_dtype = wp.vec3d if positions.dtype == torch.float64 else wp.vec3f
    mat_dtype = wp.mat33d if positions.dtype == torch.float64 else wp.mat33f

    # Transpose cell from row-convention (nvalchemi) to column-convention (nvalchemiops)
    cell_T = cell.transpose(-2, -1).contiguous()

    # Convert to warp arrays
    num_systems = cell_T.shape[0]
    wp_pos = wp.from_torch(positions.clone().contiguous(), dtype=vec_dtype)
    wp_cell = wp.from_torch(cell_T, dtype=mat_dtype)
    wp_cell_inv = wp.zeros(num_systems, dtype=mat_dtype, device=wp_pos.device)
    wp_batch_idx = wp.from_torch(batch_idx.to(torch.int32))

    compute_cell_inverse(wp_cell, wp_cell_inv)
    wrap_positions_to_cell(wp_pos, wp_cell, wp_cell_inv, wp_batch_idx)

    return wp.to_torch(wp_pos)


@_wrap_positions.register_fake
def _(
    positions: torch.Tensor,
    cell: torch.Tensor,
    batch_idx: torch.Tensor,
) -> torch.Tensor:
    return torch.empty_like(positions)


def wrap_positions_into_cell(
    positions: Float[torch.Tensor, "V 3"],
    cell: Float[torch.Tensor, "B 3 3"],
    pbc: torch.Tensor,
    batch_idx: torch.Tensor,
) -> Float[torch.Tensor, "V 3"]:
    """Wrap positions into the unit cell using fractional coordinates.

    Respects per-dimension periodicity: only periodic dimensions are
    wrapped.  Non-periodic dimensions are left unchanged.

    This function modifies ``positions`` **in-place** and returns the
    same tensor.  Delegates to ``nvalchemiops.dynamics.utils.wrap_positions_to_cell``
    for GPU-optimized wrapping, then applies per-dimension PBC masking
    in pure PyTorch.

    Parameters
    ----------
    positions : Float[Tensor, "V 3"]
        Per-atom Cartesian positions. Modified in-place.
    cell : Float[Tensor, "B 3 3"]
        Lattice vectors as rows, one ``(3, 3)`` matrix per graph.
    pbc : Tensor
        Per-dimension periodicity flags, shape ``(B, 3)``, boolean.
    batch_idx : Tensor
        Per-atom graph membership indices of shape ``(V,)``.

    Returns
    -------
    Float[Tensor, "V 3"]
        The same ``positions`` tensor (modified in-place).
    """
    original = positions.clone()
    wrapped = _wrap_positions(positions, cell, batch_idx)

    # Restore non-periodic dimensions
    per_atom_pbc = pbc[batch_idx]  # (V, 3)
    positions.copy_(torch.where(per_atom_pbc, wrapped, original))
    return positions


class WrapPeriodicHook:
    """Wrap atomic positions back into the simulation cell under PBC.

    During long molecular dynamics trajectories, atomic positions
    drift away from the unit cell as the integrator applies unbounded
    displacements.  While physically valid (forces are invariant under
    lattice translations), large coordinates can cause problems:

    * **Neighbor list overflow** — distance calculations may exceed
      the numerical range of the cell-shift representation, leading
      to missed interactions or incorrect forces.
    * **Precision loss** — large coordinate magnitudes reduce the
      effective floating-point precision available for inter-atomic
      distances.
    * **Visualization artifacts** — trajectories with unwrapped
      coordinates are difficult to analyze and visualize.

    This hook wraps positions back into the unit cell by computing
    fractional coordinates, taking their modulo, and converting back
    to Cartesian::

        frac = positions @ inv(cell)
        frac = frac % 1.0
        positions = frac @ cell

    The wrapping is applied in-place to ``batch.positions`` and
    respects per-system periodicity flags in ``batch.pbc``:

    * If ``batch.pbc`` is ``[True, True, True]``, all three
      dimensions are wrapped.
    * If ``batch.pbc`` is ``[True, True, False]`` (e.g. a slab),
      only the *x* and *y* coordinates are wrapped; the *z* coordinate
      is left unwrapped to allow vacuum gaps.
    * If ``batch.pbc`` is ``[False, False, False]`` (non-periodic),
      the hook is a no-op for that system.

    The hook fires at :attr:`~DynamicsStage.AFTER_POST_UPDATE`, after
    velocities have been updated but before the next step begins.
    This ensures that the neighbor list built at the start of the
    next step sees wrapped coordinates.

    Parameters
    ----------
    frequency : int, optional
        Wrap positions every ``frequency`` steps. Default ``1``
        (every step). For simulations with moderate drift, wrapping
        every 10--100 steps is sufficient and reduces overhead.
    stage : Enum | None, optional
        The workflow stage at which this hook runs.  Defaults to
        ``None`` (stage-agnostic until registered with a specific engine).

    Attributes
    ----------
    frequency : int
        Wrapping frequency in steps.
    stage : Enum | None
        The stage at which this hook fires.

    Examples
    --------
    >>> from nvalchemi.hooks import WrapPeriodicHook
    >>> hook = WrapPeriodicHook(frequency=10)
    >>> dynamics = DemoDynamics(model=model, n_steps=10_000, dt=0.5, hooks=[hook])
    >>> dynamics.run(batch)

    Notes
    -----
    * Wrapping does **not** modify velocities, momenta, or forces —
      only positions.  This is correct because forces depend on
      relative distances (invariant under translation) and velocities
      are already in Cartesian space.
    * For triclinic (non-orthorhombic) cells, the fractional-coordinate
      approach naturally handles skewed lattice vectors.
    * This hook assumes ``batch.cell`` has shape ``(B, 3, 3)`` with
      lattice vectors as **rows** (consistent with ASE convention).
    * In batched simulations, wrapping is applied per-graph using
      ``batch.batch_idx`` to associate each atom with its cell.
    """

    def __init__(
        self,
        frequency: int = 1,
        stage: Enum | None = None,
    ) -> None:
        self.frequency = frequency
        self.stage = stage

    def _wrap_positions(self, batch: Batch) -> None:
        """Wrap positions into the unit cell in-place."""
        cell = batch.cell
        pbc = batch.pbc
        # System-level tensors may have a leading singleton dim: (B, 1, 3, 3) -> (B, 3, 3)
        if cell.dim() == 4:
            cell = cell.squeeze(1)
        if pbc.dim() == 3:
            pbc = pbc.squeeze(1)
        wrap_positions_into_cell(batch.positions, cell, pbc, batch.batch_idx)

    def __call__(self, ctx: HookContext, stage: Enum) -> None:
        """Wrap positions into the unit cell in-place."""
        self._wrap_positions(ctx.batch)
