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
"""Concrete loss terms: :class:`EnergyLoss`, :class:`ForceLoss`, :class:`StressLoss`.

All three accept prediction and target tensors directly. The configurable
``target_key`` / ``prediction_key`` names are used by
:class:`~nvalchemi.training.losses.composition.ComposedLossFunction`
when routing keyed prediction/target mappings into these tensor-first
loss terms.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeAlias

import torch
from jaxtyping import Bool, Float, Integer
from plum import dispatch, overload

from nvalchemi._typing import BatchIndices, Energy, Forces, Scalar, Stress
from nvalchemi.training.losses.composition import BaseLossFunction, assert_same_shape
from nvalchemi.training.losses.reductions import frobenius_mse, per_graph_sum

if TYPE_CHECKING:
    from nvalchemi.data.batch import Batch

_AnyFloatTensor: TypeAlias = Float[torch.Tensor, "..."]
_NodeCounts: TypeAlias = Integer[torch.Tensor, "B"]
_PaddedNodeMask: TypeAlias = Bool[torch.Tensor, "B V_max"]
_PaddedForces: TypeAlias = Float[torch.Tensor, "B V_max 3"]
_ForceTensor: TypeAlias = Forces | _PaddedForces
_DenseForceMask: TypeAlias = Bool[torch.Tensor, "V 3"]
_PaddedForceMask: TypeAlias = Bool[torch.Tensor, "B V_max 3"]
_PerGraphValues: TypeAlias = Float[torch.Tensor, "B"]


def _masked_mse(pred: _AnyFloatTensor, target: _AnyFloatTensor) -> Scalar:
    """Return mean-squared-error over finite target entries only.

    Uses branch-free tensor ops so the loss is safe under ``torch.compile``:
    no Python ``if`` on tensor values, no boolean indexing, no ``.item()``.
    Target positions with ``NaN`` contribute zero to both numerator and
    denominator; predictions at those positions receive zero gradient.
    When every target entry is ``NaN`` the denominator is clamped to ``1``
    so the loss is ``0.0`` rather than ``NaN``.

    Parameters
    ----------
    pred : Float[torch.Tensor, "..."]
        Prediction tensor of any floating shape. Expected to be fully finite.
    target : Float[torch.Tensor, "..."]
        Target tensor with the same shape as ``pred``; may contain ``NaN``
        at positions representing missing labels.

    Returns
    -------
    Scalar
        Scalar mean of squared residuals over valid target entries.
    """
    valid = ~target.isnan()
    residual = torch.where(valid, pred - target, torch.zeros_like(pred))
    denom = valid.to(dtype=pred.dtype).sum().clamp_min(1.0)
    return residual.pow(2).sum() / denom


def _require_metadata(value: Any, name: str, *, loss_name: str) -> Any:
    """Return required loss metadata or raise a focused error."""
    if value is None:
        raise ValueError(f"{loss_name} requires {name}=... metadata.")
    return value


def _node_counts(
    num_nodes_per_graph: _NodeCounts | _PaddedNodeMask | None, ref: Energy
) -> Float[torch.Tensor, "B"]:
    """Return per-graph node counts from explicit counts or a padded mask.

    Parameters
    ----------
    num_nodes_per_graph : Integer[torch.Tensor, "B"] | Bool[torch.Tensor, "B V_max"] | None
        Either one integer count per graph or a padded node-validity mask.
        ``None`` raises because per-atom energy residuals require graph
        sizes.
    ref : Energy
        Energy tensor of shape ``(B, 1)`` whose device and dtype are used
        for the returned counts.

    Returns
    -------
    Float[torch.Tensor, "B"]
        Per-graph node counts, clamped to at least one.

    Raises
    ------
    ValueError
        If ``num_nodes_per_graph`` is ``None``.
    """
    nodes = _require_metadata(
        num_nodes_per_graph,
        "num_nodes_per_graph",
        loss_name="EnergyLoss(per_atom=True)",
    ).to(ref)
    if nodes.ndim == 1:
        return nodes.clamp_min(1)
    return nodes.sum(dim=-1).clamp_min(1)


def _padded_node_mask(
    num_nodes_per_graph: _NodeCounts | _PaddedNodeMask | None,
    ref: _PaddedForces,
    max_nodes: int,
) -> _PaddedNodeMask:
    """Return a padded node-validity mask for padded force layouts.

    Parameters
    ----------
    num_nodes_per_graph : Integer[torch.Tensor, "B"] | Bool[torch.Tensor, "B V_max"] | None
        Either one integer count per graph or an existing padded
        node-validity mask. ``None`` raises because padded forces require
        padding metadata.
    ref : Float[torch.Tensor, "B V_max 3"]
        Padded force tensor whose leading dimensions define the expected
        mask shape and whose device is used for generated masks.
    max_nodes : int
        Expected padded node dimension, equal to ``ref.shape[1]``.

    Returns
    -------
    Bool[torch.Tensor, "B V_max"]
        Boolean mask indicating valid, non-padding nodes.

    Raises
    ------
    ValueError
        If ``num_nodes_per_graph`` is ``None`` or if a supplied mask has
        width different from ``max_nodes``.
    """
    nodes = _require_metadata(
        num_nodes_per_graph, "num_nodes_per_graph", loss_name="ForceLoss"
    )
    if nodes.ndim == 2:
        mask = nodes.to(device=ref.device, dtype=torch.bool)
        if mask.shape[1] != max_nodes:
            raise ValueError(
                f"padded node mask width ({mask.shape[1]}) must match "
                f"force max nodes ({max_nodes})"
            )
        return mask
    counts = nodes.to(device=ref.device)
    return torch.arange(max_nodes, device=ref.device).unsqueeze(0) < counts.unsqueeze(
        -1
    )


class EnergyLoss(BaseLossFunction):
    r"""Mean-squared-error loss on per-graph total energy.

    Energies enter this loss as one total-energy value per graph, with
    canonical shape ``(B, 1)``. With ``per_atom=False`` the scalar is the
    graph-balanced MSE of total-energy residuals, so every graph has equal
    weight regardless of size.

    With ``per_atom=True``, prediction and target are first divided by
    each graph's atom count. The squared residual is therefore measured in
    energy-per-atom units, then reduced with atom-count weights:

    .. math::

        L = \frac{\sum_i N_i
        \left(\frac{E_i^\mathrm{pred} - E_i^\mathrm{target}}{N_i}\right)^2}
        {\sum_i N_i}.

    This makes contributions proportional to graph size while
    keeping the error quantity in per-atom energy units. Counts may be
    supplied directly as ``(B,)`` or recovered from a padded node mask of
    shape ``(B, V_max)``.

    Tensor Contract
    ---------------
    pred, target : Energy
        Per-graph energy tensors of shape ``(B, 1)``. Shape validation
        uses :func:`torch.broadcast_shapes`, so broadcast-compatible
        pairs pass the check, but callers should provide both tensors
        at the canonical ``(B, 1)`` layout. In particular, pairing a
        ``(B, 1)`` prediction with a ``(B,)`` target broadcasts to
        ``(B, B)`` and silently computes pairwise residuals across the
        batch rather than per-graph residuals.
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
    ignore_nan : bool, default False
        When ``True``, target entries equal to ``NaN`` are excluded from
        both loss value and gradient (a "nanmean"-style reduction).
        Intended for inputs where some samples lack an energy label.
        Implemented with branch-free tensor ops for ``torch.compile``
        compatibility. When ``per_atom=True``, atom-count weights for
        invalid targets are also excluded from the denominator. When
        every target entry is ``NaN`` the loss is ``0.0``.
    """

    def __init__(
        self,
        *,
        target_key: str = "energy",
        prediction_key: str = "predicted_energy",
        per_atom: bool = False,
        ignore_nan: bool = False,
    ) -> None:
        """Configure attribute keys and energy reduction semantics."""
        super().__init__()
        self.target_key = target_key
        self.prediction_key = prediction_key
        self.per_atom = per_atom
        self.ignore_nan = ignore_nan

    def forward(
        self,
        pred: Energy,
        target: Energy,
        *,
        batch: Batch | None = None,
        num_nodes_per_graph: _NodeCounts | _PaddedNodeMask | None = None,
        **kwargs: Any,  # noqa: ARG002
    ) -> Scalar:
        """Return the energy MSE using the configured reduction semantics.

        Parameters
        ----------
        pred : Energy
            Predicted per-graph energies of shape ``(B, 1)``.
        target : Energy
            Target per-graph energies of shape ``(B, 1)``.
        batch : Batch | None, optional
            Source for missing graph metadata. Explicit metadata kwargs
            override batch-derived values when both are provided.
        num_nodes_per_graph : Integer[torch.Tensor, "B"] | Bool[torch.Tensor, "B V_max"] | None, optional
            Per-graph node counts or padded node-validity mask. Required
            when ``per_atom=True``.
        **kwargs : Any
            Ignored keyword arguments accepted for the common loss-call
            interface.

        Returns
        -------
        Scalar
            Scalar energy loss.
        """
        self.per_sample_loss = None
        assert_same_shape(
            pred,
            target,
            name=type(self).__name__,
            prediction_key=self.prediction_key,
            target_key=self.target_key,
            strict=True,
        )
        if batch is not None and self.per_atom and num_nodes_per_graph is None:
            num_nodes_per_graph = getattr(batch, "num_nodes_per_graph", None)
        weights = None
        if self.per_atom:
            counts = _node_counts(num_nodes_per_graph, pred).unsqueeze(-1)
            pred = pred / counts
            target = target / counts
            weights = counts
        if self.ignore_nan:
            valid = ~target.isnan()
            residual = torch.where(valid, pred - target, torch.zeros_like(pred))
            residual_sq = residual.pow(2)
            valid_weights = valid.to(dtype=pred.dtype)
            if weights is not None:
                valid_weights = valid_weights * weights.expand_as(residual_sq)
            scalar = residual_sq.mul(
                valid_weights
            ).sum() / valid_weights.sum().clamp_min(1.0)
        else:
            residual_sq = (pred - target).pow(2)
            if weights is None:
                scalar = residual_sq.mean()
            else:
                sample_weights = weights.expand_as(residual_sq)
                scalar = residual_sq.mul(sample_weights).sum() / sample_weights.sum()
        # Only populate when the residual has a recognizable per-graph
        # shape; broadcast-trap shapes leave ``per_sample_loss`` cleared.
        # For ``per_atom=True`` this remains the per-graph squared
        # per-atom residual; the scalar applies the atom-count weights.
        if residual_sq.ndim == 1:
            self.per_sample_loss = residual_sq.detach()
        elif residual_sq.ndim == 2 and residual_sq.shape[-1] == 1:
            self.per_sample_loss = residual_sq.squeeze(-1).detach()
        return scalar

    def extra_repr(self) -> str:
        """Human-readable hyperparameter summary for :class:`nn.Module`'s repr."""
        return (
            f"target_key={self.target_key!r}, "
            f"prediction_key={self.prediction_key!r}, "
            f"per_atom={self.per_atom!r}, "
            f"ignore_nan={self.ignore_nan!r}"
        )


class ForceLoss(BaseLossFunction):
    """Mean-squared-error loss on per-atom forces.

    Forces enter this loss as per-atom vector quantities, unlike energy
    totals. The ``normalize_by_atom_count`` flag therefore does not
    convert total quantities into per-atom units; it controls how
    per-atom force residuals are reduced across a mixed-size batch.

    Dense force tensors use shape ``(V, 3)``. Padded force tensors use
    shape ``(B, V_max, 3)`` and ignore padding entries according to
    ``num_nodes_per_graph`` supplied either as ``(B,)`` counts or
    a ``(B, V_max)`` node mask.

    - ``normalize_by_atom_count=True`` (default): per-graph mean of
      squared-component error, then mean over graphs. This is a
      graph-balanced reduction: small and large graph contributions
      are somewhat normalized.
    - ``normalize_by_atom_count=False``: elementwise mean over all valid
      force components. This is an atom/component-weighted reduction:
      graph contributions are proportional to their number of valid
      force components.

    Tensor Contract
    ---------------
    pred, target : Forces | Float[torch.Tensor, "B V_max 3"]
        Dense per-node forces of shape ``(V, 3)`` or padded per-graph
        forces of shape ``(B, V_max, 3)``. Shape validation accepts any
        broadcast-compatible ``pred`` / ``target`` pair, but the
        canonical contract remains the dense or padded layout above.
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
    ignore_nan : bool, default False
        When ``True``, target force components equal to ``NaN`` are
        excluded from both loss value and gradient. Intended for batches
        where some atoms/graphs lack force labels. Implemented with
        branch-free tensor ops for ``torch.compile`` compatibility. A
        graph whose entire force tensor is ``NaN`` contributes ``0.0``
        to the loss.
    """

    def __init__(
        self,
        *,
        target_key: str = "forces",
        prediction_key: str = "predicted_forces",
        normalize_by_atom_count: bool = True,
        ignore_nan: bool = False,
    ) -> None:
        """Configure attribute keys and per-graph normalization."""
        super().__init__()
        self.target_key = target_key
        self.prediction_key = prediction_key
        self.normalize_by_atom_count = normalize_by_atom_count
        self.ignore_nan = ignore_nan

    def forward(
        self,
        pred: _ForceTensor,
        target: _ForceTensor,
        *,
        batch: Batch | None = None,
        batch_idx: BatchIndices | None = None,
        num_graphs: int | None = None,
        num_nodes_per_graph: _NodeCounts | _PaddedNodeMask | None = None,
        **kwargs: Any,  # noqa: ARG002
    ) -> Scalar:
        """Return the force-component MSE, optionally graph-balanced.

        Parameters
        ----------
        pred : Forces | Float[torch.Tensor, "B V_max 3"]
            Predicted forces. Dense layout is ``(V, 3)``; padded layout
            is ``(B, V_max, 3)``.
        target : Forces | Float[torch.Tensor, "B V_max 3"]
            Target forces with the same shape as ``pred``.
        batch : Batch | None, optional
            Source for missing graph metadata. Explicit metadata kwargs
            override batch-derived values when both are provided.
        batch_idx : BatchIndices | None, optional
            Dense-layout graph index for each node, shape ``(V,)``.
            Required for dense graph-balanced reduction and ignored for
            padded inputs.
        num_graphs : int | None, optional
            Number of graphs represented by ``batch_idx``. Required for
            dense graph-balanced reduction.
        num_nodes_per_graph : Integer[torch.Tensor, "B"] | Bool[torch.Tensor, "B V_max"] | None, optional
            Per-graph node counts or padded node-validity mask. Required
            for padded inputs.
        **kwargs : Any
            Ignored keyword arguments accepted for the common loss-call
            interface.

        Returns
        -------
        Scalar
            Scalar force loss.
        """
        self.per_sample_loss = None
        assert_same_shape(
            pred,
            target,
            name=type(self).__name__,
            prediction_key=self.prediction_key,
            target_key=self.target_key,
            strict=True,
        )
        if batch is not None:
            if self.normalize_by_atom_count and pred.ndim == 2:
                if batch_idx is None:
                    batch_idx = getattr(batch, "batch_idx", None)
                if num_graphs is None:
                    num_graphs = getattr(batch, "num_graphs", None)
            if pred.ndim == 3 and num_nodes_per_graph is None:
                num_nodes_per_graph = getattr(batch, "num_nodes_per_graph", None)
        valid = self._valid_force_components(pred, target, num_nodes_per_graph)
        residual = torch.where(valid, pred - target, torch.zeros_like(pred))
        squared_error = residual.pow(2)
        valid_components = valid.to(dtype=pred.dtype)
        if not self.normalize_by_atom_count:
            # Padded inputs still admit a per-graph view; dense ``(V, 3)``
            # has no batch dim on the scalar path, leaving
            # ``per_sample_loss`` cleared as a documented gap.
            if pred.ndim == 3:
                per_graph_num = squared_error.sum(dim=(-2, -1))
                per_graph_den = valid_components.sum(dim=(-2, -1))
                self.per_sample_loss = (
                    per_graph_num / per_graph_den.clamp_min(1.0)
                ).detach()
                return per_graph_num.sum() / per_graph_den.sum().clamp_min(1.0)
            return squared_error.sum() / valid_components.sum().clamp_min(1.0)
        per_graph_num, per_graph_den = self._per_graph_force_terms(
            squared_error, valid_components, batch_idx, num_graphs
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
        """Return a valid-component mask for dense forces.

        Parameters
        ----------
        pred : Forces
            Predicted dense force tensor of shape ``(V, 3)``. Unused;
            included for dispatch symmetry.
        target : Forces
            Target dense force tensor of shape ``(V, 3)``.
        num_nodes_per_graph : object
            Ignored for dense force tensors.

        Returns
        -------
        Bool[torch.Tensor, "V 3"]
            Valid force-component mask. All entries are valid unless
            ``ignore_nan=True``, in which case ``NaN`` target entries are
            invalid.
        """
        valid = torch.ones_like(target, dtype=torch.bool)
        if self.ignore_nan:
            valid = valid & ~target.isnan()
        return valid

    @overload
    def _valid_force_components(  # noqa: F811
        self,
        pred: _PaddedForces,
        target: _PaddedForces,
        num_nodes_per_graph: _NodeCounts | _PaddedNodeMask | None,
    ) -> _PaddedForceMask:
        """Return a valid-component mask for padded forces.

        Parameters
        ----------
        pred : Float[torch.Tensor, "B V_max 3"]
            Predicted padded force tensor.
        target : Float[torch.Tensor, "B V_max 3"]
            Target padded force tensor with the same shape as ``pred``.
        num_nodes_per_graph : Integer[torch.Tensor, "B"] | Bool[torch.Tensor, "B V_max"] | None
            Per-graph node counts or padded node-validity mask.

        Returns
        -------
        Bool[torch.Tensor, "B V_max 3"]
            Valid force-component mask. Padding entries are invalid; if
            ``ignore_nan=True``, ``NaN`` target entries are also invalid.
        """
        node_mask = _padded_node_mask(num_nodes_per_graph, pred, pred.shape[1])
        valid = node_mask.unsqueeze(-1).expand_as(pred)
        if self.ignore_nan:
            valid = valid & ~target.isnan()
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
        """Return dense-force per-graph numerators and denominators.

        Parameters
        ----------
        squared_error : Forces
            Squared force residuals of shape ``(V, 3)``.
        valid_components : Forces
            Component-validity weights of shape ``(V, 3)``.
        batch_idx : BatchIndices | None
            Graph index for each node, shape ``(V,)``. Required.
        num_graphs : int | None
            Number of graphs represented by ``batch_idx``. Required.

        Returns
        -------
        tuple[Float[torch.Tensor, "B"], Float[torch.Tensor, "B"]]
            Per-graph summed squared error and per-graph valid-component
            counts.
        """
        batch_idx = _require_metadata(batch_idx, "batch_idx", loss_name="ForceLoss")
        num_graphs = _require_metadata(num_graphs, "num_graphs", loss_name="ForceLoss")
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
        """Return padded-force per-graph numerators and denominators.

        Parameters
        ----------
        squared_error : Float[torch.Tensor, "B V_max 3"]
            Squared force residuals in padded layout.
        valid_components : Float[torch.Tensor, "B V_max 3"]
            Component-validity weights in padded layout.
        batch_idx : object
            Ignored for padded force tensors.
        num_graphs : object
            Ignored for padded force tensors.

        Returns
        -------
        tuple[Float[torch.Tensor, "B"], Float[torch.Tensor, "B"]]
            Per-graph summed squared error and per-graph valid-component
            counts.
        """
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
            f"ignore_nan={self.ignore_nan!r}"
        )


class StressLoss(BaseLossFunction):
    """Mean-squared-error loss on the per-graph stress tensor.

    Both pred and target are shape ``(B, 3, 3)``. The loss is the mean
    of the per-graph squared-Frobenius residual, computed via
    :func:`~nvalchemi.training.losses.reductions.frobenius_mse`.

    Tensor Contract
    ---------------
    pred, target : Stress
        Per-graph stress tensors of shape ``(B, 3, 3)``. Shape
        validation accepts any broadcast-compatible ``pred`` / ``target``
        pair, but the canonical contract remains ``(B, 3, 3)``.

    Parameters
    ----------
    target_key : str, default "stress"
        Target container key for the target tensor.
    prediction_key : str, default "predicted_stress"
        Prediction container key for the model output.
    ignore_nan : bool, default False
        When ``True``, target stress components equal to ``NaN`` are
        excluded from both loss value and gradient. Intended for inputs
        that mix samples with and without stress labels. Implemented
        with branch-free tensor ops for ``torch.compile`` compatibility.
        A graph whose entire stress tensor is ``NaN`` contributes
        ``0.0`` to the loss.
    """

    def __init__(
        self,
        *,
        target_key: str = "stress",
        prediction_key: str = "predicted_stress",
        ignore_nan: bool = False,
    ) -> None:
        """Configure attribute keys for target and prediction."""
        super().__init__()
        self.target_key = target_key
        self.prediction_key = prediction_key
        self.ignore_nan = ignore_nan

    def forward(
        self,
        pred: Stress,
        target: Stress,
        **kwargs: Any,  # noqa: ARG002
    ) -> Scalar:
        """Return the mean per-graph Frobenius MSE of the stress tensor.

        Parameters
        ----------
        pred : Stress
            Predicted per-graph stress tensors of shape ``(B, 3, 3)``.
        target : Stress
            Target per-graph stress tensors of shape ``(B, 3, 3)``.
        **kwargs : Any
            Ignored keyword arguments accepted for the common loss-call
            interface.

        Returns
        -------
        Scalar
            Scalar stress loss.
        """
        self.per_sample_loss = None
        assert_same_shape(
            pred,
            target,
            name=type(self).__name__,
            prediction_key=self.prediction_key,
            target_key=self.target_key,
            strict=True,
        )
        if self.ignore_nan:
            # Per-component masking over ``(B, 3, 3)``; all-NaN graph has
            # numerator 0 and clamped denominator 1, contributing zero.
            valid = ~target.isnan()
            residual = torch.where(valid, pred - target, torch.zeros_like(pred))
            per_graph_num = residual.pow(2).sum(dim=(-2, -1))
            per_graph_den = valid.to(dtype=pred.dtype).sum(dim=(-2, -1)).clamp_min(1.0)
            per_sample = per_graph_num / per_graph_den
            self.per_sample_loss = per_sample.detach()
            return per_sample.mean()
        per_sample = frobenius_mse(pred, target)
        self.per_sample_loss = per_sample.detach()
        return per_sample.mean()

    def extra_repr(self) -> str:
        """Human-readable hyperparameter summary for :class:`nn.Module`'s repr."""
        return (
            f"target_key={self.target_key!r}, "
            f"prediction_key={self.prediction_key!r}, "
            f"ignore_nan={self.ignore_nan!r}"
        )
