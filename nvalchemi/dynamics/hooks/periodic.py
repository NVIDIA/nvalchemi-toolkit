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

from typing import TYPE_CHECKING

from nvalchemi.dynamics.base import HookStageEnum

if TYPE_CHECKING:
    from nvalchemi.data import Batch
    from nvalchemi.dynamics.base import BaseDynamics

__all__ = ["WrapPeriodicHook"]


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

    The hook fires at :attr:`~HookStageEnum.AFTER_POST_UPDATE`, after
    velocities have been updated but before the next step begins.
    This ensures that the neighbor list built at the start of the
    next step sees wrapped coordinates.

    Parameters
    ----------
    frequency : int, optional
        Wrap positions every ``frequency`` steps. Default ``1``
        (every step). For simulations with moderate drift, wrapping
        every 10--100 steps is sufficient and reduces overhead.

    Attributes
    ----------
    frequency : int
        Wrapping frequency in steps.
    stage : HookStageEnum
        Fixed to ``AFTER_POST_UPDATE``.

    Examples
    --------
    >>> from nvalchemi.dynamics.hooks import WrapPeriodicHook
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
      ``batch.batch`` to associate each atom with its cell.
    """

    stage: HookStageEnum = HookStageEnum.AFTER_POST_UPDATE

    def __init__(self, frequency: int = 1) -> None:
        self.frequency = frequency

    def __call__(self, batch: Batch, dynamics: BaseDynamics) -> None:
        """Wrap positions into the unit cell in-place.

        Parameters
        ----------
        batch : Batch
            The current batch of atomic data. ``batch.positions`` is
            modified in-place.
        dynamics : BaseDynamics
            The dynamics engine instance.

        Raises
        ------
        NotImplementedError
            This hook is not yet implemented.
        """
        raise NotImplementedError("WrapPeriodicHook is not yet implemented.")
