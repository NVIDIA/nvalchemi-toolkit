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
"""Harmonic umbrella bias for umbrella sampling."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor

from nvalchemi.enhanced_sampling._bias import BiasResult, ConservativeBias
from nvalchemi.enhanced_sampling.cv._periodic import periodic_difference

if TYPE_CHECKING:
    from collections.abc import Callable

    from nvalchemi.data import Batch

__all__ = ["HarmonicUmbrellaBias"]


class HarmonicUmbrellaBias(ConservativeBias):
    r"""Multi-dimensional harmonic restraint on one or more collective variables.

    .. math::

        E_b = \tfrac{1}{2}\,\Delta^\top K \,\Delta,
        \qquad \Delta = s(x) - s_0

    with ``s(x)`` the CV value, ``s_0`` the window center, and ``K`` the
    stiffness matrix.  Forces and stress come from
    :class:`~nvalchemi.enhanced_sampling.ConservativeBias`, so they are
    guaranteed consistent with this energy.

    Per-state parameters
    --------------------
    ``centers`` and ``stiffness`` may carry a leading state dimension ``S``.
    Each graph then selects its row by ``batch.thermodynamic_state_id``,
    which is what makes a batch of umbrella windows a single batched run
    rather than ``S`` separate simulations.  Without that field every graph
    uses state ``0``.

    Parameters
    ----------
    cv:
        Any differentiable ``cv(batch) -> Tensor[B, D]``.  A plain callable;
        no base class, no registration.
    centers:
        Window centers.  Shape ``[D]`` (shared) or ``[S, D]`` (per state).
    stiffness:
        Force constants, in energy per CV-unit squared.  Accepted as:

        * scalar — isotropic, ``k·I``
        * ``[D]`` — diagonal
        * ``[D, D]`` — full matrix, shared across states
        * ``[S, D, D]`` — full matrix per state
    name:
        Unique bias identifier.
    periods:
        Period per CV component, shape ``[D]``; ``0`` marks a non-periodic
        component.  See
        :func:`~nvalchemi.enhanced_sampling.cv._periodic.periodic_difference`.
    compute_stress:
        Passed through to :class:`ConservativeBias`.

    Raises
    ------
    ValueError
        If shapes are inconsistent, or if a stiffness matrix is not
        symmetric positive-semidefinite.

    Examples
    --------
    >>> import torch
    >>> from nvalchemi.enhanced_sampling import HarmonicUmbrellaBias, pair_distance
    >>> idx = torch.tensor([0, 5])
    >>> bias = HarmonicUmbrellaBias(
    ...     cv=lambda b: pair_distance(b, idx),
    ...     centers=torch.tensor([[2.0], [2.5], [3.0]]),   # 3 windows
    ...     stiffness=10.0,                                 # eV/A^2
    ... )
    >>> bias.centers.shape
    torch.Size([3, 1])
    """

    def __init__(
        self,
        cv: Callable[[Batch], Tensor],
        centers: Tensor | float,
        stiffness: Tensor | float,
        *,
        name: str = "umbrella",
        periods: Tensor | None = None,
        compute_stress: bool = True,
    ) -> None:
        super().__init__(name=name, compute_stress=compute_stress)
        self.cv = cv

        centers_t = torch.as_tensor(centers, dtype=torch.get_default_dtype())
        if centers_t.ndim == 0:
            centers_t = centers_t.reshape(1, 1)
        elif centers_t.ndim == 1:
            centers_t = centers_t.unsqueeze(0)  # [D] -> [1, D]
        elif centers_t.ndim != 2:
            raise ValueError(
                f"HarmonicUmbrellaBias: centers must be [D] or [S, D], got "
                f"shape {tuple(centers_t.shape)}."
            )
        n_states, dim = centers_t.shape

        stiffness_t = self._expand_stiffness(stiffness, n_states, dim)
        self._validate_stiffness(stiffness_t)

        # Buffers, not plain attributes: nn.Module then moves them with .to()
        # and round-trips them through state_dict.
        self.register_buffer("centers", centers_t)
        self.register_buffer("stiffness", stiffness_t)
        if periods is None:
            self.periods: Tensor | None = None
        else:
            periods_t = torch.as_tensor(
                periods, dtype=torch.get_default_dtype()
            ).reshape(-1)
            if periods_t.numel() != dim:
                raise ValueError(
                    f"HarmonicUmbrellaBias: periods must have {dim} entries to "
                    f"match the CV dimension, got {periods_t.numel()}."
                )
            self.register_buffer("periods", periods_t)

    @staticmethod
    def _expand_stiffness(stiffness: Tensor | float, n_states: int, dim: int) -> Tensor:
        """Broadcast any accepted stiffness form to ``[S, D, D]``.

        Parameters
        ----------
        stiffness:
            Scalar, ``[D]``, ``[D, D]``, or ``[S, D, D]``.
        n_states:
            Number of thermodynamic states ``S``.
        dim:
            CV dimension ``D``.

        Returns
        -------
        Tensor
            Shape ``[S, D, D]``.

        Raises
        ------
        ValueError
            If the shape is none of the accepted forms.
        """
        k = torch.as_tensor(stiffness, dtype=torch.get_default_dtype())
        eye = torch.eye(dim, dtype=k.dtype)

        if k.ndim == 0:
            full = k * eye
        elif k.ndim == 1:
            if k.numel() != dim:
                raise ValueError(
                    f"HarmonicUmbrellaBias: diagonal stiffness must have {dim} "
                    f"entries to match the CV dimension, got {k.numel()}."
                )
            full = torch.diag(k)
        elif k.ndim == 2:
            if k.shape != (dim, dim):
                raise ValueError(
                    f"HarmonicUmbrellaBias: matrix stiffness must be "
                    f"[{dim}, {dim}], got {tuple(k.shape)}."
                )
            full = k
        elif k.ndim == 3:
            if k.shape != (n_states, dim, dim):
                raise ValueError(
                    f"HarmonicUmbrellaBias: per-state stiffness must be "
                    f"[{n_states}, {dim}, {dim}] to match centers, got "
                    f"{tuple(k.shape)}."
                )
            return k.clone()
        else:
            raise ValueError(
                f"HarmonicUmbrellaBias: stiffness must be scalar, [D], [D, D], "
                f"or [S, D, D]; got shape {tuple(k.shape)}."
            )
        return full.unsqueeze(0).expand(n_states, dim, dim).clone()

    @staticmethod
    def _validate_stiffness(stiffness: Tensor) -> None:
        """Reject a stiffness that is not symmetric positive-semidefinite.

        An asymmetric ``K`` makes the quadratic form ambiguous, and a
        negative eigenvalue turns the restraint into a repulsion that drives
        the CV away without bound — a runaway that is far cheaper to catch
        here than to diagnose from a diverging trajectory.

        Parameters
        ----------
        stiffness:
            Shape ``[S, D, D]``.

        Raises
        ------
        ValueError
            If any state's matrix is asymmetric or has a negative eigenvalue.
        """
        if not torch.allclose(stiffness, stiffness.mT, atol=1e-8):
            raise ValueError(
                "HarmonicUmbrellaBias: stiffness must be symmetric; got a "
                "matrix that differs from its transpose."
            )
        eigenvalues = torch.linalg.eigvalsh(stiffness.double())
        if bool((eigenvalues < -1e-8).any()):
            worst = float(eigenvalues.min())
            raise ValueError(
                f"HarmonicUmbrellaBias: stiffness must be positive-semidefinite; "
                f"smallest eigenvalue is {worst:.6g}. A negative eigenvalue makes "
                "the restraint repulsive along that direction."
            )

    def _validate_state_ids(self, current: Batch) -> None:
        """Raise if any ``thermodynamic_state_id`` is out of range.

        Called from :meth:`evaluate`, never from :meth:`energy`.  That
        placement is deliberate: ``energy()`` is the path
        ``EnhancedSampling(compile_biases=True)`` hands to ``torch.compile``,
        and ``bool(tensor.any())`` there is a data-dependent Python branch
        that breaks ``fullgraph=True`` outright.  ``evaluate()`` is eager by
        construction, so hoisting the check keeps it running in **every**
        mode, rather than skipping it under compile the way the eager-only
        guards in ``pair_distance`` must.

        Parameters
        ----------
        current:
            The batch; read for ``thermodynamic_state_id`` if present.

        Raises
        ------
        IndexError
            If a state id is negative or beyond the configured windows.
        """
        state_ids = getattr(current, "thermodynamic_state_id", None)
        if state_ids is None:
            return
        index = state_ids.reshape(-1).to(torch.long)
        n_states = self.centers.shape[0]
        out_of_range = (index < 0) | (index >= n_states)
        if bool(out_of_range.any()):
            bad = out_of_range.nonzero(as_tuple=False).squeeze(-1).tolist()
            raise IndexError(
                f"HarmonicUmbrellaBias: thermodynamic_state_id out of range for "
                f"{n_states} configured window(s). Graph(s) {bad} have "
                f"{index[out_of_range].tolist()}; valid ids are 0..{n_states - 1}."
            )

    def evaluate(self, current: Batch) -> BiasResult:
        """Validate the state ids, then derive energy, forces, and stress.

        Parameters
        ----------
        current:
            The live batch.

        Returns
        -------
        BiasResult
            As :meth:`ConservativeBias.evaluate`.

        Raises
        ------
        IndexError
            If a ``thermodynamic_state_id`` is out of range.
        """
        self._validate_state_ids(current)
        return super().evaluate(current)

    def _select_per_graph(
        self, current: Batch, values: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Return ``(centers, stiffness)`` broadcast to this batch.

        Contains no data-dependent Python branch, so it compiles with
        ``fullgraph=True``.  Bounds checking lives in
        :meth:`_validate_state_ids`, which :meth:`evaluate` runs first.

        Parameters
        ----------
        current:
            The batch; read for ``thermodynamic_state_id`` if present.
        values:
            CV values ``[B, D]``, used for shape and device.

        Returns
        -------
        tuple[Tensor, Tensor]
            ``centers`` ``[B, D]`` and ``stiffness`` ``[B, D, D]``.
        """
        state_ids = getattr(current, "thermodynamic_state_id", None)
        if state_ids is None:
            index = torch.zeros(values.shape[0], dtype=torch.long, device=values.device)
        else:
            index = state_ids.reshape(-1).to(torch.long)
        return self.centers[index], self.stiffness[index]

    def energy(self, current: Batch) -> Tensor:
        """Return the harmonic restraint energy ``[B, 1]`` in eV.

        Parameters
        ----------
        current:
            Batch with strained positions supplied by
            :meth:`ConservativeBias.evaluate`.

        Returns
        -------
        Tensor
            Shape ``[B, 1]``.
        """
        values = self.cv(current)  # [B, D]
        centers, stiffness = self._select_per_graph(current, values)
        delta = periodic_difference(values, centers, self.periods)  # [B, D]
        quadratic = torch.einsum("bi,bij,bj->b", delta, stiffness, delta)
        return 0.5 * quadratic.unsqueeze(-1)  # [B, 1]
