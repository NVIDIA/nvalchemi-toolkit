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
Numerical safety hooks for dynamics simulations.

Provides two post-compute hooks:

* :class:`NaNDetectorHook` — halts the simulation immediately when
  NaN or Inf values are detected in forces or energy.
* :class:`MaxForceClampHook` — clamps force magnitudes to a safe
  maximum, preventing numerical explosions from extrapolation.

Both hooks fire at :attr:`~DynamicsStage.AFTER_COMPUTE`, immediately
after the model forward pass writes forces and energy to the batch.
"""

from __future__ import annotations

from enum import Enum

import torch

from nvalchemi.data import Batch
from nvalchemi.dynamics.base import DynamicsStage
from nvalchemi.hooks._context import HookContext

__all__ = ["MaxForceClampHook", "NaNDetectorHook"]


class NaNDetectorHook:
    """Detect NaN or Inf values in model outputs and raise immediately.

    After each model forward pass, this hook inspects ``batch.forces``
    and ``batch.energy`` for non-finite values (``NaN`` or ``Inf``).
    If any are found, it raises a :class:`RuntimeError` with diagnostic
    information including:

    * Which field(s) contain non-finite values (forces, energy, or
      both).
    * The graph indices of affected samples (via ``batch.batch_idx``).
    * The current ``dynamics.step_count``.
    * The number of non-finite elements.

    This early detection prevents corrupted state from propagating
    through the integrator, which would produce meaningless trajectories
    and waste compute.  It is especially useful when running ML
    potentials on geometries outside their training distribution, where
    force predictions can diverge without warning.

    The hook can optionally check additional tensor keys beyond forces
    and energy by specifying ``extra_keys``.

    Parameters
    ----------
    frequency : int, optional
        Check every ``frequency`` steps. Default ``1`` (every step).
        Setting this higher reduces overhead at the cost of delayed
        detection.
    extra_keys : list[str] | None, optional
        Additional batch attribute names to check for non-finite
        values (e.g. ``["stress", "velocities"]``). Each key must
        be a tensor attribute on :class:`~nvalchemi.data.Batch`.
        Default ``None`` (check only forces and energy).
    stage : Enum, optional
        The stage at which to run this hook. Default is
        ``DynamicsStage.AFTER_COMPUTE``.

    Attributes
    ----------
    extra_keys : list[str]
        Additional keys to check beyond forces and energy.
    frequency : int
        Check frequency in steps.
    stage : Enum
        The stage at which this hook fires (default ``AFTER_COMPUTE``).

    Examples
    --------
    >>> from nvalchemi.dynamics.hooks import NaNDetectorHook
    >>> hook = NaNDetectorHook()  # check every step
    >>> dynamics = DemoDynamics(model=model, n_steps=1000, dt=0.5, hooks=[hook])
    >>> dynamics.run(batch)

    Check additional fields:

    >>> hook = NaNDetectorHook(extra_keys=["stress", "velocities"])

    Notes
    -----
    * The check uses ``torch.isfinite`` and operates on the full
      concatenated tensors, so the overhead scales with total atom
      count rather than batch size.
    * For production runs where overhead is a concern, set
      ``frequency=10`` or ``frequency=100`` to amortize the cost.
    * Consider pairing with :class:`MaxForceClampHook` as a first
      line of defense — clamping prevents many NaN-producing
      integration failures.
    """

    def __init__(
        self,
        frequency: int = 1,
        extra_keys: list[str] | None = None,
        stage: Enum = DynamicsStage.AFTER_COMPUTE,
    ) -> None:
        self.frequency = frequency
        self.stage = stage
        self.extra_keys: list[str] = extra_keys if extra_keys is not None else []

    def _check_finite(
        self,
        batch: Batch,
        step_count: int,
    ) -> None:
        """Check forces, energy, and extra keys for NaN/Inf values.

        This is the hot-path check shared by both dispatch overloads.

        Parameters
        ----------
        batch : Batch
            The current batch of atomic data.
        step_count : int
            The current step number.

        Raises
        ------
        RuntimeError
            If any checked tensor contains NaN or Inf values.
        """
        # --- Fast detection (hot path) ---
        # Collect tensors; skip None values
        keys_to_check = ["forces", "energy"] + self.extra_keys
        tensors: list[torch.Tensor] = []
        present_keys: list[str] = []
        for key in keys_to_check:
            tensor = getattr(batch, key, None)
            if tensor is not None:
                tensors.append(tensor)
                present_keys.append(key)

        if not tensors:
            return

        # Single-pass finiteness check — one bool per tensor
        # torch.isfinite(t).all() returns a scalar bool tensor
        all_finite = torch.stack([torch.isfinite(t).all() for t in tensors])

        # Early exit if everything is finite (hot path — no CPU sync)
        if all_finite.all():
            return

        # --- Cold diagnostic path (only on failure) ---
        self._raise_with_diagnostics(
            batch, step_count, present_keys, tensors, all_finite
        )

    def __call__(self, ctx: HookContext, stage: Enum) -> None:
        """Check forces, energy, and extra keys for NaN/Inf values."""
        self._check_finite(ctx.batch, ctx.step_count)

    @torch.compiler.disable
    def _raise_with_diagnostics(
        self,
        batch: Batch,
        step_count: int,
        keys: list[str],
        tensors: list[torch.Tensor],
        all_finite: torch.Tensor,
    ) -> None:
        """Build diagnostic message and raise RuntimeError.

        This method is only called when non-finite values are detected.
        GPU-CPU synchronization here is acceptable since we are about to
        halt the simulation. We batch tensor operations and do a single
        conversion to Python at the end to minimize D2H sync points.
        """
        bad_fields: list[str] = []
        # Collect tensor results for batch conversion
        counts: list[torch.Tensor] = []
        graph_lists: list[torch.Tensor] = []

        for i, (key, tensor) in enumerate(zip(keys, tensors)):
            if all_finite[i]:
                continue

            bad_fields.append(key)
            non_finite_mask = ~torch.isfinite(tensor)
            counts.append(non_finite_mask.sum())

            # Map back to graph indices
            if tensor.shape[0] == batch.num_nodes:
                # Node-level tensor: find which atoms have non-finite values
                affected_nodes = non_finite_mask.any(dim=-1)  # (V,)
                affected_graphs = batch.batch_idx[affected_nodes].unique()
            else:
                # Graph-level tensor
                affected_graphs = non_finite_mask.any(dim=-1).nonzero().squeeze(-1)
                # Ensure 1-D even for scalar case
                if affected_graphs.dim() == 0:
                    affected_graphs = affected_graphs.unsqueeze(0)

            graph_lists.append(affected_graphs)

        # --- Single batch conversion to CPU ---
        # Stack counts into one tensor for a single D2H transfer
        count_values = torch.stack(counts).tolist()

        diagnostics: list[str] = []
        for field, n_bad, graphs in zip(bad_fields, count_values, graph_lists):
            diagnostics.append(
                f"  {field}: {int(n_bad)} non-finite element(s) in graph(s) "
                f"{graphs.tolist()}"
            )

        msg = (
            f"Non-finite values detected at step {step_count} "
            f"in field(s): {bad_fields}\n" + "\n".join(diagnostics)
        )
        raise RuntimeError(msg)


class MaxForceClampHook:
    """Clamp per-atom force vectors to a maximum magnitude.

    After the model forward pass, this hook checks whether any atom
    has a force vector whose L2 norm exceeds ``max_force``.  If so,
    the offending force vectors are rescaled in-place to have norm
    exactly equal to ``max_force``, preserving their direction.

    This is a lightweight safety mechanism that prevents numerical
    explosions caused by:

    * ML potential extrapolation on out-of-distribution geometries.
    * Bad initial configurations with overlapping atoms.
    * Sudden large gradients from discontinuities in the potential
      energy surface.

    The clamping is applied **before** the velocity update
    (``post_update``), so the integrator sees bounded accelerations.
    This can prevent irreversible simulation blowups while allowing
    the system to recover.

    Parameters
    ----------
    max_force : float
        Maximum allowed force magnitude (L2 norm) per atom, in the
        same units as the model's force output (typically eV/A).
    stage : Enum, optional
        The stage at which to run this hook. Default is
        ``DynamicsStage.AFTER_COMPUTE``.
    frequency : int, optional
        Apply clamping every ``frequency`` steps. Default ``1``
        (every step).

    Attributes
    ----------
    max_force : float
        Maximum allowed force norm.
    frequency : int
        Clamping frequency in steps.
    stage : Enum
        The stage at which this hook fires (default ``AFTER_COMPUTE``).

    Examples
    --------
    >>> from nvalchemi.dynamics.hooks import MaxForceClampHook
    >>> hook = MaxForceClampHook(max_force=50.0)
    >>> dynamics = DemoDynamics(model=model, n_steps=1000, dt=0.5, hooks=[hook])
    >>> dynamics.run(batch)

    Notes
    -----
    * Clamping is a band-aid, not a fix.  Frequent clamping indicates
      that the model is being evaluated outside its domain of
      applicability or that the timestep is too large.
    * The implementation uses ``torch.linalg.vector_norm`` and
      ``torch.where`` for efficient, in-place operation on the full
      ``(V, 3)`` force tensor.
    * When used with :class:`NaNDetectorHook`, register
      ``MaxForceClampHook`` **first** so that forces are clamped
      before the NaN check (both fire at ``AFTER_COMPUTE`` in
      registration order).
    """

    def __init__(
        self,
        max_force: float,
        stage: Enum = DynamicsStage.AFTER_COMPUTE,
        frequency: int = 1,
    ) -> None:
        self.max_force = max_force
        self.stage = stage
        self.frequency = frequency

    def __call__(self, ctx: HookContext, stage: Enum) -> None:
        """Clamp force vectors exceeding ``max_force`` in-place."""
        self._clamp_forces(ctx.batch)

    def _clamp_forces(self, batch: Batch) -> None:
        """Clamp force vectors exceeding ``max_force`` in-place."""
        norms = torch.linalg.vector_norm(batch.forces, dim=-1, keepdim=True)  # (V, 1)
        needs_clamp = norms > self.max_force  # (V, 1) bool

        # Always compute and apply scale unconditionally (torch.compile-friendly).
        # torch.where is a no-op when nothing needs clamping.
        scale = torch.where(needs_clamp, self.max_force / norms, torch.ones_like(norms))
        batch.forces.mul_(scale)  # in-place, preserves direction
