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
  NaN or Inf values are detected in forces or energies.
* :class:`MaxForceClampHook` — clamps force magnitudes to a safe
  maximum, preventing numerical explosions from extrapolation.

Both hooks fire at :attr:`~HookStageEnum.AFTER_COMPUTE`, immediately
after the model forward pass writes forces and energies to the batch.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from nvalchemi.dynamics.hooks._base import _PostComputeHook

if TYPE_CHECKING:
    from nvalchemi.data import Batch
    from nvalchemi.dynamics.base import BaseDynamics

__all__ = ["MaxForceClampHook", "NaNDetectorHook"]


class NaNDetectorHook(_PostComputeHook):
    """Detect NaN or Inf values in model outputs and raise immediately.

    After each model forward pass, this hook inspects ``batch.forces``
    and ``batch.energies`` for non-finite values (``NaN`` or ``Inf``).
    If any are found, it raises a :class:`RuntimeError` with diagnostic
    information including:

    * Which field(s) contain non-finite values (forces, energies, or
      both).
    * The graph indices of affected samples (via ``batch.batch``).
    * The current ``dynamics.step_count``.
    * The number of non-finite elements.

    This early detection prevents corrupted state from propagating
    through the integrator, which would produce meaningless trajectories
    and waste compute.  It is especially useful when running ML
    potentials on geometries outside their training distribution, where
    force predictions can diverge without warning.

    The hook can optionally check additional tensor keys beyond forces
    and energies by specifying ``extra_keys``.

    Parameters
    ----------
    frequency : int, optional
        Check every ``frequency`` steps. Default ``1`` (every step).
        Setting this higher reduces overhead at the cost of delayed
        detection.
    extra_keys : list[str] | None, optional
        Additional batch attribute names to check for non-finite
        values (e.g. ``["stresses", "velocities"]``). Each key must
        be a tensor attribute on :class:`~nvalchemi.data.Batch`.
        Default ``None`` (check only forces and energies).

    Attributes
    ----------
    extra_keys : list[str]
        Additional keys to check beyond forces and energies.
    frequency : int
        Check frequency in steps.
    stage : HookStageEnum
        Fixed to ``AFTER_COMPUTE``.

    Examples
    --------
    >>> from nvalchemi.dynamics.hooks import NaNDetectorHook
    >>> hook = NaNDetectorHook()  # check every step
    >>> dynamics = DemoDynamics(model=model, n_steps=1000, dt=0.5, hooks=[hook])
    >>> dynamics.run(batch)

    Check additional fields:

    >>> hook = NaNDetectorHook(extra_keys=["stresses", "velocities"])

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
    ) -> None:
        super().__init__(frequency=frequency)
        self.extra_keys: list[str] = extra_keys if extra_keys is not None else []

    def __call__(self, batch: Batch, dynamics: BaseDynamics) -> None:
        """Check forces, energies, and extra keys for NaN/Inf values.

        Parameters
        ----------
        batch : Batch
            The current batch of atomic data.
        dynamics : BaseDynamics
            The dynamics engine instance.

        Raises
        ------
        NotImplementedError
            This hook is not yet implemented.
        """
        raise NotImplementedError("NaNDetectorHook is not yet implemented.")


class MaxForceClampHook(_PostComputeHook):
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
    frequency : int, optional
        Apply clamping every ``frequency`` steps. Default ``1``
        (every step).
    log_clamps : bool, optional
        If ``True``, emit a :mod:`loguru` warning each time forces
        are clamped, including the number of affected atoms and the
        original maximum magnitude. Default ``False``.

    Attributes
    ----------
    max_force : float
        Maximum allowed force norm.
    log_clamps : bool
        Whether to log clamping events.
    frequency : int
        Clamping frequency in steps.
    stage : HookStageEnum
        Fixed to ``AFTER_COMPUTE``.

    Examples
    --------
    >>> from nvalchemi.dynamics.hooks import MaxForceClampHook
    >>> hook = MaxForceClampHook(max_force=50.0, log_clamps=True)
    >>> dynamics = DemoDynamics(model=model, n_steps=1000, dt=0.5, hooks=[hook])
    >>> dynamics.run(batch)

    Notes
    -----
    * Clamping is a band-aid, not a fix.  Frequent clamping indicates
      that the model is being evaluated outside its domain of
      applicability or that the timestep is too large.
    * The implementation should use ``torch.linalg.vector_norm`` and
      ``torch.clamp`` for efficient, in-place operation on the full
      ``(V, 3)`` force tensor.
    * When used with :class:`NaNDetectorHook`, register
      ``MaxForceClampHook`` **first** so that forces are clamped
      before the NaN check (both fire at ``AFTER_COMPUTE`` in
      registration order).
    """

    def __init__(
        self,
        max_force: float,
        frequency: int = 1,
        log_clamps: bool = False,
    ) -> None:
        super().__init__(frequency=frequency)
        self.max_force = max_force
        self.log_clamps = log_clamps

    def __call__(self, batch: Batch, dynamics: BaseDynamics) -> None:
        """Clamp force vectors exceeding ``max_force`` in-place.

        Parameters
        ----------
        batch : Batch
            The current batch of atomic data. ``batch.forces`` is
            modified in-place.
        dynamics : BaseDynamics
            The dynamics engine instance.

        Raises
        ------
        NotImplementedError
            This hook is not yet implemented.
        """
        raise NotImplementedError("MaxForceClampHook is not yet implemented.")
