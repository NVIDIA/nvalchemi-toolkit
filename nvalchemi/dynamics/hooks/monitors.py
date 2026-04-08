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
Diagnostic monitor hooks for long-running simulations.

Provides :class:`EnergyDriftMonitorHook`, which tracks cumulative
energy drift over time and can warn or halt the simulation if the
drift exceeds a configurable threshold.
"""

from __future__ import annotations

from enum import Enum
from typing import Literal

import torch
from loguru import logger

from nvalchemi.data import Batch
from nvalchemi.dynamics.base import DynamicsStage
from nvalchemi.dynamics.hooks._utils import kinetic_energy_per_graph
from nvalchemi.hooks._context import HookContext

__all__ = ["EnergyDriftMonitorHook"]


class EnergyDriftMonitorHook:
    """Track energy drift and warn or stop if it exceeds a threshold.

    In a well-behaved NVE (microcanonical) simulation with a symplectic
    integrator, the total energy should be conserved to within numerical
    precision.  Significant energy drift indicates problems with:

    * The integration timestep (too large for the force magnitudes).
    * The ML potential (non-smooth or discontinuous energy surface).
    * Numerical precision (single vs. double precision accumulation).
    * Force clamping or other hook-induced modifications breaking
      energy conservation.

    This hook monitors the **total energy** (potential + kinetic) over
    the simulation and computes drift metrics.  It supports two modes:

    **Absolute drift mode** (``metric="absolute"``)
        Tracks ``|E(t) - E(0)|``, the absolute deviation from the
        initial total energy.  Suitable for NVE validation runs.

    **Per-atom-per-step drift mode** (``metric="per_atom_per_step"``)
        Tracks ``|E(t) - E(0)| / (N_atoms * step_count)``, a
        normalized metric that allows comparison across systems of
        different size and simulation length.  This is the standard
        metric reported in ML potential benchmarks.

    When the drift exceeds ``threshold``, the hook either emits a
    warning or raises a :class:`RuntimeError`, controlled by the
    ``action`` parameter.

    The hook records the reference energy on the first firing and
    computes drift on all subsequent firings.  For NVT or NPT
    simulations, energy drift is expected (the thermostat/barostat
    injects or removes energy), so use this hook primarily for NVE
    validation.

    Parameters
    ----------
    threshold : float
        Maximum acceptable drift before triggering the ``action``.
        Units depend on ``metric``: eV for ``"absolute"``,
        eV/atom/step for ``"per_atom_per_step"``.
    metric : {"absolute", "per_atom_per_step"}, optional
        Drift metric to use. Default ``"per_atom_per_step"``.
    action : {"warn", "raise"}, optional
        What to do when the threshold is exceeded. ``"warn"`` emits a
        :mod:`loguru` warning; ``"raise"`` raises a
        :class:`RuntimeError`. Default ``"warn"``.
    frequency : int, optional
        Evaluate drift every ``frequency`` steps. Default ``1``.
    include_kinetic : bool, optional
        Whether to include kinetic energy in the total energy
        calculation. Set to ``False`` if only monitoring potential
        energy drift (e.g. for optimizers). Default ``True``.

    Attributes
    ----------
    threshold : float
        Drift threshold.
    metric : str
        Drift metric mode.
    action : str
        Threshold violation behavior.
    include_kinetic : bool
        Whether kinetic energy is included.
    frequency : int
        Evaluation frequency in steps.
    stage : DynamicsStage
        Fixed to ``AFTER_STEP``.

    Examples
    --------
    NVE validation with strict drift tolerance:

    >>> from nvalchemi.dynamics.hooks import EnergyDriftMonitorHook
    >>> hook = EnergyDriftMonitorHook(
    ...     threshold=1e-5,
    ...     metric="per_atom_per_step",
    ...     action="raise",
    ...     frequency=100,
    ... )
    >>> dynamics = DemoDynamics(model=model, n_steps=10_000, dt=0.5, hooks=[hook])
    >>> dynamics.run(batch)

    Soft monitoring during production:

    >>> hook = EnergyDriftMonitorHook(
    ...     threshold=1e-3,
    ...     action="warn",
    ...     frequency=1000,
    ... )

    Notes
    -----
    * The reference energy is captured on the **first** hook firing
      (step 0 by default), not at construction time.  This allows the
      hook to be registered before the batch is available.
    * For batched simulations, drift is computed **per graph** and the
      maximum drift across all graphs is compared to the threshold.
    """

    def __init__(
        self,
        threshold: float,
        metric: Literal["absolute", "per_atom_per_step"] = "per_atom_per_step",
        action: Literal["warn", "raise"] = "warn",
        frequency: int = 1,
        include_kinetic: bool = True,
        stage: Enum = DynamicsStage.AFTER_STEP,
    ) -> None:
        self.frequency = frequency
        self.stage = stage
        self.threshold = threshold
        self.metric = metric
        self.action = action
        self.include_kinetic = include_kinetic
        self._reference_total_energy: torch.Tensor | None = None

    @torch.compiler.disable
    def _check_drift(self, batch: Batch, step_count: int, global_rank: int) -> None:
        """Compute energy drift and compare against the threshold.

        On the first firing, this method captures the reference total
        energy and returns immediately.  On all subsequent firings, it
        computes drift relative to that reference and compares against
        the configured threshold.

        Parameters
        ----------
        batch : Batch
            The current batch of atomic data.  Must have ``energy``
            (and ``velocities`` if ``include_kinetic=True``).
        step_count : int
            The current step number.
        global_rank : int
            The distributed rank of this process.

        Raises
        ------
        RuntimeError
            If ``action="raise"`` and drift exceeds the threshold.
        """
        energy = batch.energy.squeeze(-1)  # (B,)

        if self.include_kinetic and getattr(batch, "velocities", None) is not None:
            ke = kinetic_energy_per_graph(
                batch.velocities,
                batch.masses,
                batch.batch_idx,
                batch.num_graphs,
            ).squeeze(-1)  # (B,)
            total = energy + ke
        else:
            total = energy

        # Capture reference on first firing
        if self._reference_total_energy is None:
            self._reference_total_energy = total.clone()
            return

        drift = (total - self._reference_total_energy).abs()  # (B,)

        if self.metric == "per_atom_per_step":
            effective_step = max(step_count, 1)
            drift = drift / (batch.num_nodes_per_graph * effective_step)

        max_drift = drift.max().item()

        if max_drift > self.threshold:
            msg = (
                f"Energy drift {max_drift:.2e} exceeds threshold "
                f"{self.threshold:.2e} at step {step_count}"
                f" on rank {global_rank}."
            )
            if self.action == "raise":
                raise RuntimeError(msg)
            else:
                # TODO: use a distributed aware logger
                logger.warning(msg)

    def __call__(self, ctx: HookContext, stage: Enum) -> None:
        """Check energy drift against the configured threshold."""
        self._check_drift(ctx.batch, ctx.step_count, ctx.global_rank or 0)
