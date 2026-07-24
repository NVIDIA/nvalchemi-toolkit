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
r"""Concrete loss terms for energy, forces, and stress.

All three accept prediction and target tensors directly. The configurable
``target_key`` / ``prediction_key`` names are used by
:class:`~nvalchemi.training.losses.composition.ComposedLossFunction`
when routing keyed prediction/target mappings into these tensor-first
loss terms.

Notation
--------
Every loss in this module uses one shared set of symbols:

* :math:`B` is the number of graphs (structures) in the minibatch, and
  :math:`i \in \{1, \dots, B\}` indexes a graph / sample.
* :math:`N_i` is the number of atoms in graph :math:`i`, with
  :math:`a \in \{1, \dots, N_i\}` indexing an atom; :math:`V = \sum_i N_i`
  is the total atom count in the batch.
* :math:`\alpha \in \{1, 2, 3\}` indexes a Cartesian force component, and
  :math:`p, q \in \{1, 2, 3\}` index the row/column of the :math:`3 \times 3`
  stress tensor.
* A hat marks a prediction and a bare symbol the target, so a residual is
  :math:`r = \hat{y} - y`. The per-graph energy is :math:`\hat{E}_i, E_i`; the
  force on atom :math:`a` of graph :math:`i` is
  :math:`\hat{F}_{ia\alpha}, F_{ia\alpha}`; the stress is
  :math:`\hat{\sigma}_{ipq}, \sigma_{ipq}`.
* Sums run only over *valid* entries. With ``ignore_nonfinite`` a non-finite
  target is dropped, and padded (non-atom) entries are always dropped, so the
  normalizing counts (:math:`B`, :math:`N_i`, :math:`3 N_i`, :math:`V`, ...)
  count valid entries only.
* :math:`H_\delta` is the Huber function, quadratic near zero and linear in the
  tails:

  .. math::

     H_\delta(r) =
     \begin{cases}
       \tfrac{1}{2}\, r^2 & |r| < \delta, \\
       \delta \left(|r| - \tfrac{1}{2}\delta\right) & |r| \ge \delta.
     \end{cases}
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeAlias

import torch
from jaxtyping import Bool, Float, Integer
from plum import dispatch, overload

from nvalchemi._typing import BatchIndices, Energy, Forces
from nvalchemi.training.losses.composition import (
    BaseLossFunction,
    DTypePolicy,
    ReductionContext,
)
from nvalchemi.training.losses.reductions import per_graph_sum

if TYPE_CHECKING:
    from nvalchemi.data.batch import Batch

_NodeCounts: TypeAlias = Integer[torch.Tensor, "B"]
_PaddedNodeMask: TypeAlias = Bool[torch.Tensor, "B V_max"]
_PaddedForces: TypeAlias = Float[torch.Tensor, "B V_max 3"]
_ForceTensor: TypeAlias = Forces | _PaddedForces
_DenseForceMask: TypeAlias = Bool[torch.Tensor, "V 3"]
_PaddedForceMask: TypeAlias = Bool[torch.Tensor, "B V_max 3"]
_PerGraphValues: TypeAlias = Float[torch.Tensor, "B"]


def _require_metadata(value: Any, name: str, *, loss_name: str) -> Any:
    """Return required loss metadata or raise a focused error."""
    if value is None:
        raise ValueError(f"{loss_name} requires {name}=... metadata.")
    return value


def _node_counts(
    num_nodes_per_graph: _NodeCounts | _PaddedNodeMask | None,
    ref: Energy,
) -> Float[torch.Tensor, "B"]:
    """Return per-graph node counts from counts or a padded node mask."""
    nodes = _require_metadata(
        num_nodes_per_graph,
        "num_nodes_per_graph",
        loss_name="per-atom energy loss",
    ).to(ref)
    if nodes.ndim not in (1, 2):
        raise ValueError(
            "num_nodes_per_graph must be a 1-D count tensor or a 2-D padded node mask."
        )
    if nodes.shape[0] != ref.shape[0]:
        raise ValueError(
            "num_nodes_per_graph leading dimension "
            f"({nodes.shape[0]}) must match energy batch size ({ref.shape[0]})."
        )
    if nodes.ndim == 1:
        return nodes.clamp_min(1)
    return nodes.sum(dim=-1).clamp_min(1)


def _padded_node_mask(
    num_nodes_per_graph: _NodeCounts | _PaddedNodeMask | None,
    ref: _PaddedForces,
    max_nodes: int,
) -> _PaddedNodeMask:
    """Return a padded node-validity mask for padded force tensors."""
    nodes = _require_metadata(
        num_nodes_per_graph, "num_nodes_per_graph", loss_name="padded force loss"
    )
    if nodes.ndim == 2:
        mask = nodes.to(device=ref.device, dtype=torch.bool)
        if mask.shape[0] != ref.shape[0]:
            raise ValueError(
                f"padded node mask batch dimension ({mask.shape[0]}) "
                f"must match force batch size ({ref.shape[0]})."
            )
        if mask.shape[1] != max_nodes:
            raise ValueError(
                f"padded node mask width ({mask.shape[1]}) must match "
                f"force max nodes ({max_nodes}) for padded force tensors."
            )
        return mask
    if nodes.ndim != 1:
        raise ValueError(
            "num_nodes_per_graph must be a 1-D count tensor or a 2-D padded node mask."
        )
    if nodes.shape[0] != ref.shape[0]:
        raise ValueError(
            f"num_nodes_per_graph length ({nodes.shape[0]}) "
            f"must match force batch size ({ref.shape[0]})."
        )
    counts = nodes.to(device=ref.device)
    return torch.arange(max_nodes, device=ref.device).unsqueeze(0) < counts.unsqueeze(
        -1
    )


def _huber_loss(residual: torch.Tensor, delta: float) -> torch.Tensor:
    """Return elementwise Huber loss for a residual tensor.

    Parameters
    ----------
    residual : torch.Tensor
        Prediction-minus-target residual.
    delta : float
        Positive transition point between quadratic and linear regimes.

    Returns
    -------
    torch.Tensor
        Elementwise Huber loss with the same shape as ``residual``.
    """
    abs_residual = residual.abs()
    return torch.where(
        abs_residual < delta,
        0.5 * abs_residual.pow(2),
        delta * (abs_residual - 0.5 * delta),
    )


class EnergyMSELoss(BaseLossFunction):
    r"""Mean-squared-error loss on per-graph total energy.

    Energies enter this loss as one total-energy value per graph, with
    canonical shape ``(B, 1)``. With ``per_atom=False`` the loss is the
    graph-balanced MSE of total-energy residuals, so every graph has equal
    weight regardless of size:

    .. math::

        L = \frac{1}{B} \sum_{i=1}^{B} \left(\hat{E}_i - E_i\right)^2.

    With ``per_atom=True`` the prediction and target are first divided by each
    graph's atom count :math:`N_i`, so the residual is measured in
    energy-per-atom units, and the reduction is weighted by :math:`N_i` so that
    larger graphs contribute in proportion to their size:

    .. math::

        L = \frac{\sum_{i=1}^{B} N_i
        \left(\dfrac{\hat{E}_i - E_i}{N_i}\right)^2}{\sum_{i=1}^{B} N_i}
        = \frac{\sum_{i=1}^{B} (\hat{E}_i - E_i)^2 / N_i}{\sum_{i=1}^{B} N_i}.

    A hat denotes the prediction and :math:`N_i` the atom count of graph
    :math:`i` (see the module docstring for the shared notation). Counts
    :math:`N_i` may be supplied directly as ``(B,)`` or recovered from a padded
    node mask of shape ``(B, V_max)``.

    Tensor Contract
    ---------------
    pred, target : Energy
        Per-graph energy tensors of shape ``(B, 1)``. Shape validation
        requires exact equality; ``(B, 1)`` and ``(B,)`` are rejected
        even though they are broadcast-compatible.
    num_nodes_per_graph : Integer[torch.Tensor, "B"] | Bool[torch.Tensor, "B V_max"], optional
        Required only when ``per_atom=True``. May be explicit per-graph
        counts or a padded node-validity mask.

    Parameters
    ----------
    target_key : str, default "energy"
        Target container key for the target tensor.
    prediction_key : str, default "predicted_energy"
        Prediction container key for the model output.
    per_atom : bool, default False
        Measure residuals in energy-per-atom units and reduce them with
        atom-count weights: larger graphs contribute in proportion to
        their atom counts.
    ignore_nonfinite : bool, default False
        When ``True``, target entries that are ``NaN`` or infinite are
        excluded from both loss value and gradient using
        :func:`torch.isfinite`. Intended for inputs where some samples
        lack an energy label. Implemented with branch-free tensor ops
        for ``torch.compile`` compatibility. When ``per_atom=True``,
        atom-count weights for invalid targets are also excluded from
        the denominator. When every target entry is non-finite the loss
        is ``0.0``.
    dtype_policy : {"strict", "prediction_to_target", "target_to_prediction"}, default "strict"
        How to handle prediction/target dtype mismatches before validation.
        ``"strict"`` raises; the other policies cast one tensor to match the
        other.
    """

    requires_eval_grad: bool = False

    def __init__(
        self,
        *,
        target_key: str = "energy",
        prediction_key: str = "predicted_energy",
        per_atom: bool = False,
        ignore_nonfinite: bool = False,
        dtype_policy: DTypePolicy = "strict",
    ) -> None:
        """Configure attribute keys and energy reduction semantics."""
        super().__init__(dtype_policy=dtype_policy)
        self.target_key = target_key
        self.prediction_key = prediction_key
        self.per_atom = per_atom
        self.ignore_nonfinite = ignore_nonfinite

    def normalize(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor, ReductionContext]:
        """Divide by atom counts when ``per_atom=True``."""
        ctx = ReductionContext()
        if not self.per_atom:
            return pred, target, ctx
        batch: Batch | None = kwargs.get("batch")
        num_nodes_per_graph = kwargs.get("num_nodes_per_graph")
        if batch is not None and num_nodes_per_graph is None:
            num_nodes_per_graph = getattr(batch, "num_nodes_per_graph", None)
        counts = _node_counts(num_nodes_per_graph, pred).unsqueeze(-1)
        ctx["weights"] = counts
        return pred / counts, target / counts, ctx

    def mask(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        ctx: ReductionContext,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Exclude non-finite target entries when ``ignore_nonfinite=True``."""
        if self.ignore_nonfinite:
            return torch.isfinite(target)
        return torch.ones_like(target, dtype=torch.bool)

    def compute_residual(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        """Return squared residuals, zeroing invalid entries."""
        residual = torch.where(valid, pred - target, torch.zeros_like(pred))
        return residual.pow(2)

    def extra_repr(self) -> str:
        """Human-readable hyperparameter summary for :class:`nn.Module`'s repr."""
        return (
            f"target_key={self.target_key!r}, "
            f"prediction_key={self.prediction_key!r}, "
            f"per_atom={self.per_atom!r}, "
            f"ignore_nonfinite={self.ignore_nonfinite!r}, "
            f"dtype_policy={self.dtype_policy!r}"
        )


class EnergyMAELoss(BaseLossFunction):
    r"""Mean-absolute-error loss for per-graph energy targets.

    This loss operates on per-graph total energies with identical
    prediction and target shapes, commonly ``(B, 1)`` or ``(B,)``. With
    ``per_atom=True`` (default), prediction and target energies are first
    divided by each graph's atom count :math:`N_i`, then absolute residuals are
    reduced with atom-count weights so that larger graphs contribute
    in proportion to their size:

    .. math::

        L = \frac{\sum_{i=1}^{B} N_i
        \left|\dfrac{\hat{E}_i - E_i}{N_i}\right|}{\sum_{i=1}^{B} N_i}
        = \frac{\sum_{i=1}^{B} |\hat{E}_i - E_i|}{\sum_{i=1}^{B} N_i}.

    With ``per_atom=False`` the loss is the graph-balanced mean absolute error
    of total-energy residuals, :math:`L = \tfrac{1}{B} \sum_{i=1}^{B}
    |\hat{E}_i - E_i|`. A hat denotes the prediction and :math:`N_i` the atom
    count of graph :math:`i` (see the module docstring for the shared notation).

    Parameters
    ----------
    target_key : str, default "energy"
        Target container key for the target tensor.
    prediction_key : str, default "predicted_energy"
        Prediction container key for the model output.
    per_atom : bool, default True
        Divide prediction and target by ``num_nodes_per_graph`` before
        computing absolute residuals.
    ignore_nonfinite : bool, default True
        When ``True``, target entries that are ``NaN`` or infinite are
        excluded from both loss value and gradient using
        :func:`torch.isfinite`.
    dtype_policy : {"strict", "prediction_to_target", "target_to_prediction"}, default "strict"
        How to handle prediction/target dtype mismatches before validation.
        ``"strict"`` raises; the other policies cast one tensor to match the
        other.
    """

    def __init__(
        self,
        *,
        target_key: str = "energy",
        prediction_key: str = "predicted_energy",
        per_atom: bool = True,
        ignore_nonfinite: bool = True,
        dtype_policy: DTypePolicy = "strict",
    ) -> None:
        """Configure attribute keys and energy MAE semantics."""
        super().__init__(dtype_policy=dtype_policy)
        self.target_key = target_key
        self.prediction_key = prediction_key
        self.per_atom = per_atom
        self.ignore_nonfinite = ignore_nonfinite

    def normalize(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor, ReductionContext]:
        """Divide by atom counts when ``per_atom=True``."""
        ctx = ReductionContext()
        if not self.per_atom:
            return pred, target, ctx
        batch: Batch | None = kwargs.get("batch")
        num_nodes_per_graph = kwargs.get("num_nodes_per_graph")
        if batch is not None and num_nodes_per_graph is None:
            num_nodes_per_graph = getattr(batch, "num_nodes_per_graph", None)
        counts = _node_counts(num_nodes_per_graph, pred).reshape(
            (-1,) + (1,) * (pred.ndim - 1)
        )
        ctx["weights"] = counts
        return pred / counts, target / counts, ctx

    def mask(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        ctx: ReductionContext,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Exclude non-finite target entries when ``ignore_nonfinite=True``."""
        if self.ignore_nonfinite:
            return torch.isfinite(target)
        return torch.ones_like(target, dtype=torch.bool)

    def compute_residual(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        """Return absolute residuals, zeroing invalid entries."""
        return torch.where(valid, pred - target, torch.zeros_like(pred)).abs()

    def extra_repr(self) -> str:
        """Human-readable hyperparameter summary for :class:`nn.Module`'s repr."""
        return (
            f"target_key={self.target_key!r}, "
            f"prediction_key={self.prediction_key!r}, "
            f"per_atom={self.per_atom!r}, "
            f"ignore_nonfinite={self.ignore_nonfinite!r}, "
            f"dtype_policy={self.dtype_policy!r}"
        )


class EnergyHuberLoss(BaseLossFunction):
    r"""Huber loss on total energy or energy per atom.

    The elementwise Huber function :math:`H_\delta` (quadratic within
    :math:`\delta` of zero, linear beyond it; see the module docstring) is
    applied to each per-graph energy residual, then averaged over the :math:`B`
    labeled graphs:

    .. math::

        L = \frac{1}{B} \sum_{i=1}^{B} H_\delta\!\left(\hat{E}_i - E_i\right).

    With ``per_atom=True`` (default) the prediction and target are first divided
    by each graph's atom count :math:`N_i`, so the Huber function acts on
    energy-per-atom residuals :math:`(\hat{E}_i - E_i) / N_i`. Unlike the
    per-atom MSE/MAE terms, the final reduction is an unweighted mean over
    labeled structures rather than an atom-count-weighted mean.

    Parameters
    ----------
    target_key : str, default "energy"
        Target container key for the target tensor.
    prediction_key : str, default "predicted_energy"
        Prediction container key for the model output.
    per_atom : bool, default True
        Divide prediction and target by ``num_nodes_per_graph`` before
        computing Huber residuals.
    delta : float, default 0.01
        Positive transition point between quadratic and linear Huber regimes.
    ignore_nonfinite : bool, default True
        When ``True``, target entries that are ``NaN`` or infinite are
        excluded from both loss value and gradient using
        :func:`torch.isfinite`.
    dtype_policy : {"strict", "prediction_to_target", "target_to_prediction"}, default "strict"
        How to handle prediction/target dtype mismatches before validation.
        ``"strict"`` raises; the other policies cast one tensor to match the
        other.
    """

    requires_eval_grad: bool = False

    def __init__(
        self,
        *,
        target_key: str = "energy",
        prediction_key: str = "predicted_energy",
        per_atom: bool = True,
        delta: float = 0.01,
        ignore_nonfinite: bool = True,
        dtype_policy: DTypePolicy = "strict",
    ) -> None:
        """Configure energy Huber loss keys and threshold."""
        super().__init__(dtype_policy=dtype_policy)
        self.target_key = target_key
        self.prediction_key = prediction_key
        self.per_atom = per_atom
        self.ignore_nonfinite = ignore_nonfinite
        self.delta = float(delta)

    def normalize(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor, ReductionContext]:
        """Divide by atom counts when ``per_atom=True``."""
        ctx = ReductionContext()
        if not self.per_atom:
            return pred, target, ctx
        batch: Batch | None = kwargs.get("batch")
        num_nodes_per_graph = kwargs.get("num_nodes_per_graph")
        if batch is not None and num_nodes_per_graph is None:
            num_nodes_per_graph = getattr(batch, "num_nodes_per_graph", None)
        counts = _node_counts(num_nodes_per_graph, pred).reshape(
            (-1,) + (1,) * (pred.ndim - 1)
        )
        return pred / counts, target / counts, ctx

    def mask(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        ctx: ReductionContext,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Exclude non-finite target entries when ``ignore_nonfinite=True``."""
        if self.ignore_nonfinite:
            return torch.isfinite(target)
        return torch.ones_like(target, dtype=torch.bool)

    def compute_residual(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        """Return elementwise Huber losses, zeroing invalid entries."""
        residual = torch.where(valid, pred - target, torch.zeros_like(pred))
        return _huber_loss(residual, self.delta)

    def extra_repr(self) -> str:
        """Human-readable hyperparameter summary for :class:`nn.Module`'s repr."""
        return (
            f"target_key={self.target_key!r}, "
            f"prediction_key={self.prediction_key!r}, "
            f"per_atom={self.per_atom!r}, "
            f"ignore_nonfinite={self.ignore_nonfinite!r}, "
            f"delta={self.delta!r}, "
            f"dtype_policy={self.dtype_policy!r}"
        )


class ForceMSELoss(BaseLossFunction):
    r"""Mean-squared-error loss on per-atom forces.

    Forces enter this loss as per-atom vector quantities, unlike energy
    totals. Writing :math:`r_{ia\alpha} = \hat{F}_{ia\alpha} - F_{ia\alpha}`
    for the residual of Cartesian component :math:`\alpha` of atom :math:`a` in
    graph :math:`i`, the ``normalize_by_atom_count`` flag selects how the
    squared component residuals are reduced across a mixed-size batch:

    - ``normalize_by_atom_count=True`` (default): each graph's mean squared
      component error is averaged over graphs, so every graph has equal weight
      (a graph-balanced reduction):

      .. math::

          L = \frac{1}{B} \sum_{i=1}^{B}
          \frac{1}{3 N_i} \sum_{a=1}^{N_i} \sum_{\alpha=1}^{3} r_{ia\alpha}^2.

    - ``normalize_by_atom_count=False``: one global mean over every valid force
      component, so a graph's weight is proportional to its atom count:

      .. math::

          L = \frac{1}{3V} \sum_{i=1}^{B} \sum_{a=1}^{N_i} \sum_{\alpha=1}^{3}
          r_{ia\alpha}^2, \qquad 3V = \sum_{i=1}^{B} 3 N_i.

    Dense force tensors use shape ``(V, 3)``. Padded force tensors use
    shape ``(B, V_max, 3)`` and ignore padding entries according to
    ``num_nodes_per_graph`` supplied either as ``(B,)`` counts or
    a ``(B, V_max)`` node mask. A hat denotes the prediction; see the module
    docstring for the shared notation.

    Tensor Contract
    ---------------
    pred, target : Forces | Float[torch.Tensor, "B V_max 3"]
        Dense per-node forces of shape ``(V, 3)`` or padded per-graph
        forces of shape ``(B, V_max, 3)``. Shape validation requires
        exact equality.
    batch_idx : BatchIndices, optional
        Required for dense ``(V, 3)`` forces when
        ``normalize_by_atom_count=True``. Ignored for padded forces.
    num_nodes_per_graph : Integer[torch.Tensor, "B"] | Bool[torch.Tensor, "B V_max"], optional
        Required for padded ``(B, V_max, 3)`` forces. May be explicit
        per-graph counts or a padded node-validity mask.

    Parameters
    ----------
    target_key : str, default "forces"
        Target container key for the target tensor.
    prediction_key : str, default "predicted_forces"
        Prediction container key for the model output.
    normalize_by_atom_count : bool, default True
        Control the batch reduction for already-per-atom force
        residuals. ``True`` computes a graph-balanced mean by dividing
        each graph's force-error sum by its valid component count before
        averaging over graphs. ``False`` computes one global elementwise
        mean over all valid force components.
    ignore_nonfinite : bool, default False
        When ``True``, target force components that are ``NaN`` or
        infinite are excluded from both loss value and gradient using
        :func:`torch.isfinite`. Intended for batches where some
        atoms/graphs lack force labels. Implemented with branch-free
        tensor ops for ``torch.compile`` compatibility. A graph whose
        entire force tensor is non-finite contributes ``0.0`` to the
        loss.
    dtype_policy : {"strict", "prediction_to_target", "target_to_prediction"}, default "strict"
        How to handle prediction/target dtype mismatches before validation.
        ``"strict"`` raises; the other policies cast one tensor to match the
        other.
    """

    requires_eval_grad: bool = True

    def __init__(
        self,
        *,
        target_key: str = "forces",
        prediction_key: str = "predicted_forces",
        normalize_by_atom_count: bool = True,
        ignore_nonfinite: bool = False,
        dtype_policy: DTypePolicy = "strict",
    ) -> None:
        """Configure attribute keys and per-graph normalization."""
        super().__init__(dtype_policy=dtype_policy)
        self.target_key = target_key
        self.prediction_key = prediction_key
        self.normalize_by_atom_count = normalize_by_atom_count
        self.ignore_nonfinite = ignore_nonfinite

    def mask(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        ctx: ReductionContext,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Return component-level validity mask for dense or padded forces."""
        num_nodes_per_graph = kwargs.get("num_nodes_per_graph")
        batch: Batch | None = kwargs.get("batch")
        if batch is not None and pred.ndim == 3 and num_nodes_per_graph is None:
            num_nodes_per_graph = getattr(batch, "num_nodes_per_graph", None)
        return self._valid_force_components(pred, target, num_nodes_per_graph)

    def compute_residual(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        """Return squared force-component residuals, zeroing invalid entries."""
        residual = torch.where(valid, pred - target, torch.zeros_like(pred))
        return residual.pow(2)

    def reduce(
        self,
        residual: torch.Tensor,
        valid: torch.Tensor,
        ctx: ReductionContext,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Reduce force-component residuals to a scalar loss."""
        valid_components = valid.to(dtype=residual.dtype)
        batch: Batch | None = kwargs.get("batch")
        batch_idx: BatchIndices | None = kwargs.get("batch_idx")
        num_graphs: int | None = kwargs.get("num_graphs")
        if batch is not None and self.normalize_by_atom_count and residual.ndim == 2:
            if batch_idx is None:
                batch_idx = getattr(batch, "batch_idx", None)
            if num_graphs is None:
                num_graphs = getattr(batch, "num_graphs", None)
        if not self.normalize_by_atom_count:
            if residual.ndim == 3:
                per_graph_num = residual.sum(dim=(-2, -1))
                per_graph_den = valid_components.sum(dim=(-2, -1))
                self.per_sample_loss = (
                    per_graph_num / per_graph_den.clamp_min(1.0)
                ).detach()
                return per_graph_num.sum() / per_graph_den.sum().clamp_min(1.0)
            return residual.sum() / valid_components.sum().clamp_min(1.0)
        per_graph_num, per_graph_den = self._per_graph_force_terms(
            residual, valid_components, batch_idx, num_graphs
        )
        per_sample = per_graph_num / per_graph_den.clamp_min(1.0)
        self.per_sample_loss = per_sample.detach()
        return per_sample.mean()

    @overload
    def _valid_force_components(  # noqa: F811
        self,
        pred: Forces,  # noqa: ARG002
        target: Forces,
        num_nodes_per_graph: object,  # noqa: ARG002
    ) -> _DenseForceMask:
        """Return a valid-component mask for dense forces."""
        valid = torch.ones_like(target, dtype=torch.bool)
        if self.ignore_nonfinite:
            valid = valid & torch.isfinite(target)
        return valid

    @overload
    def _valid_force_components(  # noqa: F811
        self,
        pred: _PaddedForces,
        target: _PaddedForces,
        num_nodes_per_graph: _NodeCounts | _PaddedNodeMask | None,
    ) -> _PaddedForceMask:
        """Return a valid-component mask for padded forces."""
        node_mask = _padded_node_mask(num_nodes_per_graph, pred, pred.shape[1])
        valid = node_mask.unsqueeze(-1).expand_as(pred)
        if self.ignore_nonfinite:
            valid = valid & torch.isfinite(target)
        return valid

    @dispatch
    def _valid_force_components(  # noqa: F811
        self, pred: object, target: object, num_nodes_per_graph: object
    ) -> _DenseForceMask | _PaddedForceMask:
        pass

    @overload
    def _per_graph_force_terms(  # noqa: F811
        self,
        squared_error: Forces,
        valid_components: Forces,
        batch_idx: BatchIndices | None,
        num_graphs: int | None,
    ) -> tuple[_PerGraphValues, _PerGraphValues]:
        """Return dense-force per-graph numerators and denominators."""
        batch_idx = _require_metadata(batch_idx, "batch_idx", loss_name="ForceMSELoss")
        num_graphs = _require_metadata(
            num_graphs, "num_graphs", loss_name="ForceMSELoss"
        )
        per_atom_se = squared_error.sum(dim=-1)
        per_atom_valid = valid_components.sum(dim=-1)
        per_graph_se_sum = per_graph_sum(per_atom_se, batch_idx, num_graphs=num_graphs)
        per_graph_valid = per_graph_sum(
            per_atom_valid, batch_idx, num_graphs=num_graphs
        )
        return per_graph_se_sum, per_graph_valid

    @overload
    def _per_graph_force_terms(  # noqa: F811
        self,
        squared_error: _PaddedForces,
        valid_components: _PaddedForces,
        batch_idx: object,  # noqa: ARG002
        num_graphs: object,  # noqa: ARG002
    ) -> tuple[_PerGraphValues, _PerGraphValues]:
        """Return padded-force per-graph numerators and denominators."""
        return squared_error.sum(dim=(-2, -1)), valid_components.sum(dim=(-2, -1))

    @dispatch
    def _per_graph_force_terms(  # noqa: F811
        self,
        squared_error: object,
        valid_components: object,
        batch_idx: object,
        num_graphs: object,
    ) -> tuple[_PerGraphValues, _PerGraphValues]:
        pass

    def extra_repr(self) -> str:
        """Human-readable hyperparameter summary for :class:`nn.Module`'s repr."""
        return (
            f"target_key={self.target_key!r}, "
            f"prediction_key={self.prediction_key!r}, "
            f"normalize_by_atom_count={self.normalize_by_atom_count!r}, "
            f"ignore_nonfinite={self.ignore_nonfinite!r}, "
            f"dtype_policy={self.dtype_policy!r}"
        )


class ForceHuberLoss(ForceMSELoss):
    r"""Huber loss on per-component force residuals.

    Applies the Huber function :math:`H_\delta` to each force-component
    residual :math:`r_{ia\alpha} = \hat{F}_{ia\alpha} - F_{ia\alpha}`, reusing
    the masking and batch reduction of :class:`ForceMSELoss`. With
    ``normalize_by_atom_count=False`` (default) this is a global mean over every
    valid component:

    .. math::

        L = \frac{1}{3V} \sum_{i=1}^{B} \sum_{a=1}^{N_i} \sum_{\alpha=1}^{3}
        H_\delta\!\left(r_{ia\alpha}\right).

    With ``normalize_by_atom_count=True`` the per-graph mean is averaged over
    graphs instead, :math:`L = \tfrac{1}{B} \sum_{i} \tfrac{1}{3 N_i}
    \sum_{a, \alpha} H_\delta(r_{ia\alpha})`.

    Parameters
    ----------
    target_key : str, default "forces"
        Target container key for the target tensor.
    prediction_key : str, default "predicted_forces"
        Prediction container key for the model output.
    normalize_by_atom_count : bool, default False
        Control the batch reduction for already-per-atom force
        residuals. ``True`` computes a graph-balanced mean by dividing
        each graph's force-error sum by its valid component count before
        averaging over graphs. ``False`` computes one global elementwise
        mean over all valid force components.
    delta : float, default 0.01
        Positive transition point between quadratic and linear Huber regimes.
    ignore_nonfinite : bool, default True
        When ``True``, target force components that are ``NaN`` or
        infinite are excluded from both loss value and gradient using
        :func:`torch.isfinite`.
    dtype_policy : {"strict", "prediction_to_target", "target_to_prediction"}, default "strict"
        How to handle prediction/target dtype mismatches before validation.
        ``"strict"`` raises; the other policies cast one tensor to match the
        other.
    """

    def __init__(
        self,
        *,
        target_key: str = "forces",
        prediction_key: str = "predicted_forces",
        normalize_by_atom_count: bool = False,
        delta: float = 0.01,
        ignore_nonfinite: bool = True,
        dtype_policy: DTypePolicy = "strict",
    ) -> None:
        """Configure force Huber loss keys, threshold, and reduction."""
        super().__init__(
            target_key=target_key,
            prediction_key=prediction_key,
            normalize_by_atom_count=normalize_by_atom_count,
            ignore_nonfinite=ignore_nonfinite,
            dtype_policy=dtype_policy,
        )
        self.delta = float(delta)

    def compute_residual(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        """Return componentwise Huber force losses, zeroing invalid entries."""
        residual = torch.where(valid, pred - target, torch.zeros_like(pred))
        return _huber_loss(residual, self.delta)

    def extra_repr(self) -> str:
        """Human-readable hyperparameter summary for :class:`nn.Module`'s repr."""
        return f"{super().extra_repr()}, delta={self.delta!r}"


class ForceL2NormLoss(BaseLossFunction):
    r"""Mean per-atom force-vector L2 loss.

    Unlike the component-wise force terms, the per-atom residual here is the
    Euclidean norm of the force-vector error,

    .. math::

        \rho_{ia} = \bigl\|\hat{\mathbf{F}}_{ia} - \mathbf{F}_{ia}\bigr\|_2
        = \sqrt{\sum_{\alpha=1}^{3}
        \left(\hat{F}_{ia\alpha} - F_{ia\alpha}\right)^2},

    reduced over atoms according to ``normalize_by_atom_count``:

    - ``normalize_by_atom_count=True`` (default): the mean atom norm per graph
      is averaged over graphs (graph-balanced):

      .. math::

          L = \frac{1}{B} \sum_{i=1}^{B}
          \frac{1}{N_i} \sum_{a=1}^{N_i} \rho_{ia}.

    - ``normalize_by_atom_count=False``: one global mean over all valid atoms,
      :math:`L = \tfrac{1}{V} \sum_{i=1}^{B} \sum_{a=1}^{N_i} \rho_{ia}`.

    Dense ``(V, 3)`` inputs can be graph-balanced with ``batch_idx`` and
    ``num_graphs``. Padded ``(B, V_max, 3)`` inputs require
    ``num_nodes_per_graph`` counts or a node mask so padding can be
    excluded from the atom-level reduction. The reduction is over atoms (one
    norm per atom), not over individual components.

    Parameters
    ----------
    target_key : str, default "forces"
        Target container key for the target tensor.
    prediction_key : str, default "predicted_forces"
        Prediction container key for the model output.
    normalize_by_atom_count : bool, default True
        When ``True``, compute a mean atom L2 norm per graph, then mean
        over graphs. When ``False``, compute one global mean over valid
        atom L2 norms.
    ignore_nonfinite : bool, default True
        When ``True``, atoms whose target vector contains ``NaN`` or
        infinity are excluded from both loss value and gradient using
        :func:`torch.isfinite`.
    dtype_policy : {"strict", "prediction_to_target", "target_to_prediction"}, default "strict"
        How to handle prediction/target dtype mismatches before validation.
        ``"strict"`` raises; the other policies cast one tensor to match the
        other.
    """

    def __init__(
        self,
        *,
        target_key: str = "forces",
        prediction_key: str = "predicted_forces",
        normalize_by_atom_count: bool = True,
        ignore_nonfinite: bool = True,
        dtype_policy: DTypePolicy = "strict",
    ) -> None:
        """Configure attribute keys and force L2 semantics."""
        super().__init__(dtype_policy=dtype_policy)
        self.target_key = target_key
        self.prediction_key = prediction_key
        self.normalize_by_atom_count = normalize_by_atom_count
        self.ignore_nonfinite = ignore_nonfinite

    def mask(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        ctx: ReductionContext,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Return atom-level validity mask (not component-level) for forces.

        The mask has shape ``(V,)`` for dense or ``(B, V_max)`` for
        padded forces — one validity flag per atom, not per component.
        """
        num_nodes_per_graph = kwargs.get("num_nodes_per_graph")
        batch: Batch | None = kwargs.get("batch")
        if batch is not None and pred.ndim == 3 and num_nodes_per_graph is None:
            num_nodes_per_graph = getattr(batch, "num_nodes_per_graph", None)
        return self._valid_force_atoms(pred, target, num_nodes_per_graph)

    def compute_residual(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        """Return per-atom L2 norm of force residuals, zeroing invalid atoms."""
        valid_vectors = valid.unsqueeze(-1)
        residual = torch.where(valid_vectors, pred - target, torch.zeros_like(pred))
        return torch.linalg.vector_norm(residual, ord=2, dim=-1)

    def reduce(
        self,
        residual: torch.Tensor,
        valid: torch.Tensor,
        ctx: ReductionContext,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Reduce per-atom L2 norms to a scalar loss."""
        atom_weights = valid.to(dtype=residual.dtype)
        batch: Batch | None = kwargs.get("batch")
        batch_idx: BatchIndices | None = kwargs.get("batch_idx")
        num_graphs: int | None = kwargs.get("num_graphs")
        if batch is not None and self.normalize_by_atom_count and residual.ndim == 1:
            if batch_idx is None:
                batch_idx = getattr(batch, "batch_idx", None)
            if num_graphs is None:
                num_graphs = getattr(batch, "num_graphs", None)
        if not self.normalize_by_atom_count:
            if residual.ndim == 2:
                per_graph_counts = atom_weights.sum(dim=-1).clamp_min(1.0)
                self.per_sample_loss = (
                    residual.sum(dim=-1) / per_graph_counts
                ).detach()
            return residual.sum() / atom_weights.sum().clamp_min(1.0)
        per_graph_sum_l2, per_graph_counts = self._per_graph_atom_terms(
            residual, atom_weights, batch_idx, num_graphs
        )
        per_sample = per_graph_sum_l2 / per_graph_counts.clamp_min(1.0)
        self.per_sample_loss = per_sample.detach()
        return per_sample.mean()

    def _valid_force_atoms(
        self,
        pred: _ForceTensor,
        target: _ForceTensor,
        num_nodes_per_graph: _NodeCounts | _PaddedNodeMask | None,
    ) -> Bool[torch.Tensor, "V"] | _PaddedNodeMask:
        """Return atom-validity mask for dense or padded forces."""
        if pred.ndim == 2:
            if self.ignore_nonfinite:
                return torch.isfinite(target).all(dim=-1)
            return torch.ones_like(target[..., 0], dtype=torch.bool)
        node_mask = _padded_node_mask(num_nodes_per_graph, pred, pred.shape[1])
        if self.ignore_nonfinite:
            return node_mask & torch.isfinite(target).all(dim=-1)
        return node_mask

    def _per_graph_atom_terms(
        self,
        per_atom_values: Float[torch.Tensor, "..."],
        atom_weights: Float[torch.Tensor, "..."],
        batch_idx: BatchIndices | None,
        num_graphs: int | None,
    ) -> tuple[_PerGraphValues, _PerGraphValues]:
        """Return per-graph atom-value sums and valid atom counts."""
        if per_atom_values.ndim == 1:
            batch_idx = _require_metadata(
                batch_idx, "batch_idx", loss_name="ForceL2NormLoss"
            )
            num_graphs = _require_metadata(
                num_graphs, "num_graphs", loss_name="ForceL2NormLoss"
            )
            return (
                per_graph_sum(per_atom_values, batch_idx, num_graphs=num_graphs),
                per_graph_sum(atom_weights, batch_idx, num_graphs=num_graphs),
            )
        return per_atom_values.sum(dim=-1), atom_weights.sum(dim=-1)

    def extra_repr(self) -> str:
        """Human-readable hyperparameter summary for :class:`nn.Module`'s repr."""
        return (
            f"target_key={self.target_key!r}, "
            f"prediction_key={self.prediction_key!r}, "
            f"normalize_by_atom_count={self.normalize_by_atom_count!r}, "
            f"ignore_nonfinite={self.ignore_nonfinite!r}, "
            f"dtype_policy={self.dtype_policy!r}"
        )


class StressMSELoss(BaseLossFunction):
    r"""Mean-squared-error loss on the per-graph stress tensor.

    Both prediction and target are :math:`3 \times 3` tensors of shape
    ``(B, 3, 3)``. Each graph contributes the mean of its nine squared
    component residuals -- equivalently its squared Frobenius norm divided by
    the number of valid components -- and these are averaged over graphs:

    .. math::

        L = \frac{1}{B} \sum_{i=1}^{B} \frac{1}{9}
        \sum_{p=1}^{3} \sum_{q=1}^{3}
        \left(\hat{\sigma}_{ipq} - \sigma_{ipq}\right)^2
        = \frac{1}{B} \sum_{i=1}^{B}
        \frac{\bigl\|\hat{\sigma}_i - \sigma_i\bigr\|_F^2}{9},

    where the fraction shown assumes all nine components are valid. Concretely,
    :meth:`reduce` sums the squared component residuals of each graph and
    divides by the number of valid components for that graph (clamped to at
    least 1), giving a per-graph component-mean; these per-graph values are then
    averaged over graphs. When ``ignore_nonfinite`` drops components the
    per-graph denominator is the number of remaining valid components rather
    than 9, and a graph whose entire stress tensor is non-finite contributes
    ``0.0``. A hat denotes the prediction; see the module docstring for the
    shared notation.

    Tensor Contract
    ---------------
    pred, target : Stress
        Per-graph stress tensors of shape ``(B, 3, 3)``. Shape
        validation requires exact equality.

    Parameters
    ----------
    target_key : str, default "stress"
        Target container key for the target tensor.
    prediction_key : str, default "predicted_stress"
        Prediction container key for the model output.
    ignore_nonfinite : bool, default False
        When ``True``, target stress components that are ``NaN`` or
        infinite are excluded from both loss value and gradient using
        :func:`torch.isfinite`. Intended for inputs that mix samples
        with and without stress labels. Implemented with branch-free
        tensor ops for ``torch.compile`` compatibility. A graph whose
        entire stress tensor is non-finite contributes ``0.0`` to the
        loss.
    dtype_policy : {"strict", "prediction_to_target", "target_to_prediction"}, default "strict"
        How to handle prediction/target dtype mismatches before validation.
        ``"strict"`` raises; the other policies cast one tensor to match the
        other.
    """

    requires_eval_grad: bool = True

    def __init__(
        self,
        *,
        target_key: str = "stress",
        prediction_key: str = "predicted_stress",
        ignore_nonfinite: bool = False,
        dtype_policy: DTypePolicy = "strict",
    ) -> None:
        """Configure attribute keys for target and prediction."""
        super().__init__(dtype_policy=dtype_policy)
        self.target_key = target_key
        self.prediction_key = prediction_key
        self.ignore_nonfinite = ignore_nonfinite

    def mask(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        ctx: ReductionContext,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Exclude non-finite stress components when ``ignore_nonfinite=True``."""
        if self.ignore_nonfinite:
            return torch.isfinite(target)
        return torch.ones_like(target, dtype=torch.bool)

    def compute_residual(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        """Return squared stress residuals, zeroing invalid entries."""
        residual = torch.where(valid, pred - target, torch.zeros_like(pred))
        return residual.pow(2)

    def reduce(
        self,
        residual: torch.Tensor,
        valid: torch.Tensor,
        ctx: ReductionContext,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Reduce per-component stress residuals to a per-graph mean scalar."""
        per_graph_num = residual.sum(dim=(-2, -1))
        per_graph_den = valid.to(dtype=residual.dtype).sum(dim=(-2, -1)).clamp_min(1.0)
        per_sample = per_graph_num / per_graph_den
        self.per_sample_loss = per_sample.detach()
        return per_sample.mean()

    def extra_repr(self) -> str:
        """Human-readable hyperparameter summary for :class:`nn.Module`'s repr."""
        return (
            f"target_key={self.target_key!r}, "
            f"prediction_key={self.prediction_key!r}, "
            f"ignore_nonfinite={self.ignore_nonfinite!r}, "
            f"dtype_policy={self.dtype_policy!r}"
        )


class StressHuberLoss(StressMSELoss):
    r"""Huber loss on per-graph stress tensors.

    Applies the Huber function :math:`H_\delta` to each stress-component
    residual, then reuses the per-graph averaging of :class:`StressMSELoss`:

    .. math::

        L = \frac{1}{B} \sum_{i=1}^{B} \frac{1}{9}
        \sum_{p=1}^{3} \sum_{q=1}^{3}
        H_\delta\!\left(\hat{\sigma}_{ipq} - \sigma_{ipq}\right).

    Parameters
    ----------
    target_key : str, default "stress"
        Target container key for the target tensor.
    prediction_key : str, default "predicted_stress"
        Prediction container key for the model output.
    delta : float, default 0.01
        Positive transition point between quadratic and linear Huber regimes.
    ignore_nonfinite : bool, default True
        When ``True``, target stress components that are ``NaN`` or
        infinite are excluded from both loss value and gradient using
        :func:`torch.isfinite`.
    dtype_policy : {"strict", "prediction_to_target", "target_to_prediction"}, default "strict"
        How to handle prediction/target dtype mismatches before validation.
        ``"strict"`` raises; the other policies cast one tensor to match the
        other.
    """

    def __init__(
        self,
        *,
        target_key: str = "stress",
        prediction_key: str = "predicted_stress",
        delta: float = 0.01,
        ignore_nonfinite: bool = True,
        dtype_policy: DTypePolicy = "strict",
    ) -> None:
        """Configure stress Huber loss keys and threshold."""
        super().__init__(
            target_key=target_key,
            prediction_key=prediction_key,
            ignore_nonfinite=ignore_nonfinite,
            dtype_policy=dtype_policy,
        )
        self.delta = float(delta)

    def compute_residual(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        """Return componentwise Huber stress losses, zeroing invalid entries."""
        residual = torch.where(valid, pred - target, torch.zeros_like(pred))
        return _huber_loss(residual, self.delta)

    def extra_repr(self) -> str:
        """Human-readable hyperparameter summary for :class:`nn.Module`'s repr."""
        return f"{super().extra_repr()}, delta={self.delta!r}"
