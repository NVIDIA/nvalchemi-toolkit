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
"""One-sided walls and flat-bottom restraints on a collective variable.

All three classes here share one energy shape — a power-law penalty on how
far the CV has strayed past a threshold, and exactly zero inside the allowed
region.  They differ only in which side is penalised.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor

from nvalchemi.enhanced_sampling._bias import ConservativeBias

if TYPE_CHECKING:
    from collections.abc import Callable

    from nvalchemi.data import Batch

__all__ = ["FlatBottomRestraint", "LowerWall", "UpperWall"]


class _WallBase(ConservativeBias):
    r"""Shared machinery for one- and two-sided CV penalties.

    The energy is built from ``clamp(excess, min=0) ** exponent``, which is
    what keeps a wall usable as a bias at all:

    * **The graph stays connected inside the wall.** A wall implemented as
      ``if inside: return zeros(B, 1)`` returns a tensor with no ``grad_fn``,
      and autograd rejects such an output outright. ``clamp`` returns a
      genuine zero *that is still attached*, so forces come back as zeros
      rather than as an error.
    * **The force is continuous at the boundary** for ``exponent >= 2``.
      With ``exponent = 1`` the force jumps from ``0`` to ``k`` at the wall,
      which a finite time step turns into an impulse; hence the default of 2
      and the warning below.

    Parameters
    ----------
    cv:
        Differentiable ``cv(batch) -> Tensor[B, D]``.
    threshold:
        Wall position, broadcastable to the CV shape ``[D]``.
    stiffness:
        Penalty prefactor ``k``, broadcastable to ``[D]``.
    name:
        Unique bias identifier.
    exponent:
        Power of the penalty.  Must be ``>= 1``.
    compute_stress:
        Passed through to :class:`ConservativeBias`.

    Raises
    ------
    ValueError
        If ``exponent < 1`` or ``stiffness`` is negative.
    """

    def __init__(
        self,
        cv: Callable[[Batch], Tensor],
        threshold: Tensor | float,
        stiffness: Tensor | float,
        *,
        name: str,
        exponent: float = 2.0,
        compute_stress: bool = True,
    ) -> None:
        super().__init__(name=name, compute_stress=compute_stress)
        if exponent < 1:
            raise ValueError(
                f"{type(self).__name__}: exponent must be >= 1, got {exponent}. "
                "A sub-linear wall has unbounded force at the boundary."
            )
        stiffness_t = torch.as_tensor(
            stiffness, dtype=torch.get_default_dtype()
        ).reshape(-1)
        if bool((stiffness_t < 0).any()):
            raise ValueError(
                f"{type(self).__name__}: stiffness must be non-negative, got "
                f"{stiffness_t.tolist()}. A negative wall pushes the system out "
                "of the allowed region instead of back into it."
            )
        self.cv = cv
        self.exponent = float(exponent)
        self.register_buffer(
            "threshold",
            torch.as_tensor(threshold, dtype=torch.get_default_dtype()).reshape(-1),
        )
        self.register_buffer("stiffness", stiffness_t)

    def _excess(self, values: Tensor) -> Tensor:
        """Return the signed distance past the wall, before clamping.

        Parameters
        ----------
        values:
            CV values ``[B, D]``.

        Returns
        -------
        Tensor
            Positive where the wall is violated, shape ``[B, D]``.

        Raises
        ------
        NotImplementedError
            If the subclass does not override.
        """
        raise NotImplementedError

    def energy(self, current: Batch) -> Tensor:
        """Return the wall energy ``[B, 1]`` in eV.

        Parameters
        ----------
        current:
            Batch with strained positions supplied by
            :meth:`ConservativeBias.evaluate`.

        Returns
        -------
        Tensor
            Shape ``[B, 1]``; exactly zero for configurations inside the wall.
        """
        values = self.cv(current)  # [B, D]
        excess = torch.clamp(self._excess(values), min=0.0)
        penalty = self.stiffness * excess**self.exponent
        return penalty.sum(dim=-1, keepdim=True) / self.exponent  # [B, 1]


class UpperWall(_WallBase):
    r"""Penalise the CV for rising above a threshold.

    .. math:: E_b = \frac{k}{p}\,\max(s - s_0,\,0)^p

    Parameters
    ----------
    cv:
        Differentiable ``cv(batch) -> Tensor[B, D]``.
    threshold:
        Upper bound ``s_0``.
    stiffness:
        Penalty prefactor ``k``.  Default ``10.0``.
    name:
        Unique bias identifier.  Default ``"upper_wall"``.
    exponent:
        Power ``p``.  Default ``2.0``.
    compute_stress:
        Passed through to :class:`ConservativeBias`.

    Examples
    --------
    >>> import torch
    >>> from nvalchemi.enhanced_sampling import UpperWall, pair_distance
    >>> idx = torch.tensor([0, 1])
    >>> wall = UpperWall(cv=lambda b: pair_distance(b, idx), threshold=5.0)
    >>> wall.name
    'upper_wall'
    """

    def __init__(
        self,
        cv: Callable[[Batch], Tensor],
        threshold: Tensor | float,
        stiffness: Tensor | float = 10.0,
        *,
        name: str = "upper_wall",
        exponent: float = 2.0,
        compute_stress: bool = True,
    ) -> None:
        super().__init__(
            cv,
            threshold,
            stiffness,
            name=name,
            exponent=exponent,
            compute_stress=compute_stress,
        )

    def _excess(self, values: Tensor) -> Tensor:
        """Return ``values - threshold``.

        Parameters
        ----------
        values:
            CV values ``[B, D]``.

        Returns
        -------
        Tensor
            Shape ``[B, D]``.
        """
        return values - self.threshold


class LowerWall(_WallBase):
    r"""Penalise the CV for falling below a threshold.

    .. math:: E_b = \frac{k}{p}\,\max(s_0 - s,\,0)^p

    Parameters
    ----------
    cv:
        Differentiable ``cv(batch) -> Tensor[B, D]``.
    threshold:
        Lower bound ``s_0``.
    stiffness:
        Penalty prefactor ``k``.  Default ``10.0``.
    name:
        Unique bias identifier.  Default ``"lower_wall"``.
    exponent:
        Power ``p``.  Default ``2.0``.
    compute_stress:
        Passed through to :class:`ConservativeBias`.
    """

    def __init__(
        self,
        cv: Callable[[Batch], Tensor],
        threshold: Tensor | float,
        stiffness: Tensor | float = 10.0,
        *,
        name: str = "lower_wall",
        exponent: float = 2.0,
        compute_stress: bool = True,
    ) -> None:
        super().__init__(
            cv,
            threshold,
            stiffness,
            name=name,
            exponent=exponent,
            compute_stress=compute_stress,
        )

    def _excess(self, values: Tensor) -> Tensor:
        """Return ``threshold - values``.

        Parameters
        ----------
        values:
            CV values ``[B, D]``.

        Returns
        -------
        Tensor
            Shape ``[B, D]``.
        """
        return self.threshold - values


class FlatBottomRestraint(_WallBase):
    r"""Confine the CV to an interval, with no force inside it.

    .. math::

        E_b = \frac{k}{p}\left[
            \max(s - s_\mathrm{hi},\,0)^p + \max(s_\mathrm{lo} - s,\,0)^p
        \right]

    Equivalent to registering a :class:`LowerWall` and an :class:`UpperWall`
    with the same stiffness, but as one bias — which matters because the
    proposal's rule is that intentionally coupled terms belong in a single
    bias object rather than being summed by the runner.

    Parameters
    ----------
    cv:
        Differentiable ``cv(batch) -> Tensor[B, D]``.
    lower:
        Lower bound ``s_lo``.
    upper:
        Upper bound ``s_hi``.
    stiffness:
        Penalty prefactor ``k``.  Default ``10.0``.
    name:
        Unique bias identifier.  Default ``"flat_bottom"``.
    exponent:
        Power ``p``.  Default ``2.0``.
    compute_stress:
        Passed through to :class:`ConservativeBias`.

    Raises
    ------
    ValueError
        If any ``lower`` bound is not strictly below its ``upper`` bound.
    """

    def __init__(
        self,
        cv: Callable[[Batch], Tensor],
        lower: Tensor | float,
        upper: Tensor | float,
        stiffness: Tensor | float = 10.0,
        *,
        name: str = "flat_bottom",
        exponent: float = 2.0,
        compute_stress: bool = True,
    ) -> None:
        lower_t = torch.as_tensor(lower, dtype=torch.get_default_dtype()).reshape(-1)
        upper_t = torch.as_tensor(upper, dtype=torch.get_default_dtype()).reshape(-1)
        if bool((lower_t >= upper_t).any()):
            raise ValueError(
                f"FlatBottomRestraint: every lower bound must be strictly below "
                f"its upper bound, got lower={lower_t.tolist()} and "
                f"upper={upper_t.tolist()}."
            )
        super().__init__(
            cv,
            upper_t,
            stiffness,
            name=name,
            exponent=exponent,
            compute_stress=compute_stress,
        )
        self.register_buffer("lower", lower_t)

    def energy(self, current: Batch) -> Tensor:
        """Return the two-sided confinement energy ``[B, 1]`` in eV.

        Parameters
        ----------
        current:
            Batch with strained positions supplied by
            :meth:`ConservativeBias.evaluate`.

        Returns
        -------
        Tensor
            Shape ``[B, 1]``; exactly zero inside ``[lower, upper]``.
        """
        values = self.cv(current)  # [B, D]
        above = torch.clamp(values - self.threshold, min=0.0)
        below = torch.clamp(self.lower - values, min=0.0)
        penalty = self.stiffness * (above**self.exponent + below**self.exponent)
        return penalty.sum(dim=-1, keepdim=True) / self.exponent

    def _excess(self, values: Tensor) -> Tensor:
        """Return the distance above the upper bound.

        Unused — :meth:`energy` is overridden to handle both sides — but
        defined so the class is not abstract in spirit.

        Parameters
        ----------
        values:
            CV values ``[B, D]``.

        Returns
        -------
        Tensor
            Shape ``[B, D]``.
        """
        return values - self.threshold
