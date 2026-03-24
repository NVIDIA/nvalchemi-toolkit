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
Biased potential hooks for enhanced sampling workflows.

Provides :class:`BiasedPotentialHook`, which adds external bias
potentials to the forces and energies computed by the ML model.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from nvalchemi.data import Batch
from nvalchemi.dynamics.base import DynamicsStage
from nvalchemi.hooks._context import HookContext

if TYPE_CHECKING:
    from collections.abc import Callable

    from nvalchemi._typing import Energy, Forces

__all__ = ["BiasedPotentialHook"]


class BiasedPotentialHook:
    """Add an external bias potential to forces and energies after the forward pass.

    This hook enables enhanced sampling techniques by composing an
    arbitrary bias potential on top of the ML potential **without**
    modifying the model itself.  The bias is applied in-place to
    ``batch.forces`` and ``batch.energies`` at
    :attr:`~HookStageEnum.AFTER_COMPUTE`, keeping the model output
    pure and the bias fully composable.

    The bias is specified via a callable ``bias_fn`` with signature::

        bias_fn(batch: Batch) -> tuple[Tensor, Tensor]
            Returns (bias_energy, bias_forces) where:
            - bias_energy: Float[Tensor, "B 1"]  — per-graph energy bias
            - bias_forces: Float[Tensor, "V 3"]  — per-atom force bias

    The hook adds the bias terms to the existing batch values::

        batch.energies += bias_energy
        batch.forces   += bias_forces

    This design supports a wide range of enhanced sampling methods:

    * **Harmonic restraints** — bias that penalizes deviation from a
      reference geometry (e.g. collective variable restraints).
    * **Umbrella sampling** — harmonic bias along a reaction
      coordinate, with the umbrella window center parameterizing
      ``bias_fn``.
    * **Metadynamics** — time-dependent Gaussian bias deposited along
      collective variables.  ``bias_fn`` would maintain internal state
      (deposited Gaussians) and compute the accumulated bias.
    * **Steered MD** — time-dependent bias that pulls the system along
      a path.  ``bias_fn`` can read ``dynamics.step_count`` to vary
      the bias over time.
    * **Wall potentials** — repulsive bias that confines atoms to a
      region of space (e.g. preventing evaporation from a surface).

    Parameters
    ----------
    bias_fn : Callable[[Batch], tuple[Tensor, Tensor]]
        A callable that computes the bias energy and forces given the
        current batch.  Must return a tuple of
        ``(bias_energy, bias_forces)`` with shapes ``(B, 1)`` and
        ``(V, 3)`` respectively, on the same device as the batch.
    stage : Enum, optional
        The stage at which to run this hook. Default is
        ``DynamicsStage.AFTER_COMPUTE``.
    frequency : int, optional
        Apply the bias every ``frequency`` steps. Default ``1``
        (every step).
    inplace : bool, optional
        If True, will modify energies and forces in the batch in-place.
        Otherwise, replaces the existing tensors.

    Attributes
    ----------
    bias_fn : Callable
        The bias potential function.
    frequency : int
        Bias application frequency in steps.
    stage : Enum
        The stage at which this hook fires (default ``AFTER_COMPUTE``).

    Examples
    --------
    Harmonic restraint on center of mass:

    >>> import torch
    >>> from nvalchemi.dynamics.hooks import BiasedPotentialHook
    >>> def harmonic_restraint(batch):
    ...     # Restrain center of mass to origin with k=10 eV/A^2
    ...     k = 10.0
    ...     com = batch.positions.mean(dim=0, keepdim=True)
    ...     bias_energy = 0.5 * k * (com ** 2).sum().unsqueeze(0).unsqueeze(0)
    ...     bias_forces = -k * com.expand_as(batch.positions) / batch.num_nodes
    ...     return bias_energy, bias_forces
    >>> hook = BiasedPotentialHook(bias_fn=harmonic_restraint)

    Notes
    -----
    * The ``bias_fn`` is called **after** the model forward pass, so
      it has access to the model-computed forces and energies via the
      batch if needed (e.g. for force-matching penalties).
    * Because the bias modifies forces in-place, it interacts correctly
      with :class:`MaxForceClampHook` — register the clamp hook
      **after** the bias hook to clamp the total (model + bias) forces.
    * For metadynamics, the ``bias_fn`` closure should hold a reference
      to a mutable state object (e.g. a list of deposited Gaussians)
      that is updated externally or within the callable.
    * The bias does **not** contribute to the autograd graph.  If the
      model uses conservative forces (``forces_via_autograd=True``),
      the bias forces are added after ``torch.autograd.grad`` has
      already computed the model forces.
    """

    def __init__(
        self,
        bias_fn: Callable[[Batch], tuple[Energy, Forces]],
        stage: Enum = DynamicsStage.AFTER_COMPUTE,
        frequency: int = 1,
        inplace: bool = True,
    ) -> None:
        self.bias_fn = bias_fn
        self.stage = stage
        self.frequency = frequency
        self.inplace = inplace

    def __call__(self, ctx: HookContext, stage: DynamicsStage) -> None:
        """Compute and add the bias potential."""
        self._apply_bias(ctx.batch)

    def _apply_bias(self, batch: Batch) -> None:
        """Apply bias potential to the batch."""
        bias_energy, bias_forces = self.bias_fn(batch)

        if not torch.compiler.is_compiling():
            if bias_energy.shape != batch.energies.shape:
                raise RuntimeError(
                    f"bias_energy shape {bias_energy.shape} does not match "
                    f"batch.energies shape {batch.energies.shape}"
                )
            if bias_forces.shape != batch.forces.shape:
                raise RuntimeError(
                    f"bias_forces shape {bias_forces.shape} does not match "
                    f"batch.forces shape {batch.forces.shape}"
                )

        if self.inplace:
            batch.energies.add_(bias_energy)
            batch.forces.add_(bias_forces)
        else:
            batch["energies"] = batch.energies + bias_energy
            batch["forces"] = batch.forces + bias_forces
