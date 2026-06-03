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
"""Concrete loss terms for energy, forces, and stress.

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


class EnergyMSELoss(BaseLossFunction):
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
    """

    def __init__(
        self,
        *,
        target_key: str = "energy",
        prediction_key: str = "predicted_energy",
        per_atom: bool = False,
        ignore_nonfinite: bool = False,
    ) -> None:
        """Configure attribute keys and energy reduction semantics."""
        super().__init__()
        self.target_key = target_key
        self.prediction_key = prediction_key
        self.per_atom = per_atom
        self.ignore_nonfinite = ignore_nonfinite

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
        if self.ignore_nonfinite:
            valid = torch.isfinite(target)
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
            f"ignore_nonfinite={self.ignore_nonfinite!r}"
        )


class EnergyMAELoss(BaseLossFunction):
    r"""Mean-absolute-error loss for per-graph energy targets.

    This loss operates on per-graph total energies with identical
    prediction and target shapes, commonly ``(B, 1)`` or ``(B,)``. With
    ``per_atom=True`` (default), prediction and target energies are first
    divided by each graph's atom count, then absolute residuals are
    reduced with atom-count weights so that larger graphs contribute
    in proportion to their size:

    .. math::

        L = \frac{\sum_i N_i
        \left|\frac{E_i^\mathrm{pred} - E_i^\mathrm{target}}{N_i}\right|}
        {\sum_i N_i}.

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
    """

    def __init__(
        self,
        *,
        target_key: str = "energy",
        prediction_key: str = "predicted_energy",
        per_atom: bool = True,
        ignore_nonfinite: bool = True,
    ) -> None:
        """Configure attribute keys and energy MAE semantics."""
        super().__init__()
        self.target_key = target_key
        self.prediction_key = prediction_key
        self.per_atom = per_atom
        self.ignore_nonfinite = ignore_nonfinite

    def forward(
        self,
        pred: Energy,
        target: Energy,
        *,
        batch: Batch | None = None,
        num_nodes_per_graph: _NodeCounts | _PaddedNodeMask | None = None,
        **kwargs: Any,  # noqa: ARG002
    ) -> Scalar:
        """Return the energy MAE scalar.

        Parameters
        ----------
        pred : Energy
            Predicted per-graph energies of shape ``(B, 1)`` or ``(B,)``.
        target : Energy
            Target per-graph energies with the exact same shape as
            ``pred``.
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
            Scalar energy MAE over valid graph entries.
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
        valid = torch.ones_like(target, dtype=torch.bool)
        if self.ignore_nonfinite:
            valid = torch.isfinite(target)
        if batch is not None and self.per_atom and num_nodes_per_graph is None:
            num_nodes_per_graph = getattr(batch, "num_nodes_per_graph", None)
        weights = None
        if self.per_atom:
            counts = _node_counts(num_nodes_per_graph, pred).reshape(
                (-1,) + (1,) * (pred.ndim - 1)
            )
            pred = pred / counts
            target = target / counts
            weights = counts
        residual = torch.where(valid, pred - target, torch.zeros_like(pred)).abs()
        valid_weights = valid.to(dtype=pred.dtype)
        if weights is not None:
            valid_weights = valid_weights * weights.expand_as(residual)
        scalar = residual.mul(valid_weights).sum() / valid_weights.sum().clamp_min(1.0)
        if residual.ndim == 1:
            self.per_sample_loss = residual.detach()
        elif residual.ndim == 2 and residual.shape[-1] == 1:
            self.per_sample_loss = residual.squeeze(-1).detach()
        return scalar

    def extra_repr(self) -> str:
        """Human-readable hyperparameter summary for :class:`nn.Module`'s repr."""
        return (
            f"target_key={self.target_key!r}, "
            f"prediction_key={self.prediction_key!r}, "
            f"per_atom={self.per_atom!r}, "
            f"ignore_nonfinite={self.ignore_nonfinite!r}"
        )


class ForceMSELoss(BaseLossFunction):
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
    """

    def __init__(
        self,
        *,
        target_key: str = "forces",
        prediction_key: str = "predicted_forces",
        normalize_by_atom_count: bool = True,
        ignore_nonfinite: bool = False,
    ) -> None:
        """Configure attribute keys and per-graph normalization."""
        super().__init__()
        self.target_key = target_key
        self.prediction_key = prediction_key
        self.normalize_by_atom_count = normalize_by_atom_count
        self.ignore_nonfinite = ignore_nonfinite

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
            ``ignore_nonfinite=True``, in which case non-finite target
            entries are invalid.
        """
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
            ``ignore_nonfinite=True``, non-finite target entries are also invalid.
        """
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
            f"ignore_nonfinite={self.ignore_nonfinite!r}"
        )


class ForceL2NormLoss(BaseLossFunction):
    """Mean per-atom force-vector L2 loss.

    The per-atom residual is the vector norm
    ``torch.linalg.vector_norm(pred - target, ord=2, dim=-1)``. Dense
    ``(V, 3)`` inputs can be graph-balanced with ``batch_idx`` and
    ``num_graphs``. Padded ``(B, V_max, 3)`` inputs require
    ``num_nodes_per_graph`` counts or a node mask so padding can be
    excluded from the atom-level reduction.

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
    """

    def __init__(
        self,
        *,
        target_key: str = "forces",
        prediction_key: str = "predicted_forces",
        normalize_by_atom_count: bool = True,
        ignore_nonfinite: bool = True,
    ) -> None:
        """Configure attribute keys and force L2 semantics."""
        super().__init__()
        self.target_key = target_key
        self.prediction_key = prediction_key
        self.normalize_by_atom_count = normalize_by_atom_count
        self.ignore_nonfinite = ignore_nonfinite

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
        """Return the force-vector L2 scalar.

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
            Required for dense graph-balanced reduction.
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
            Scalar force L2 norm loss.
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

        valid_atoms = self._valid_force_atoms(pred, target, num_nodes_per_graph)
        valid_vectors = valid_atoms.unsqueeze(-1)
        residual = torch.where(valid_vectors, pred - target, torch.zeros_like(pred))
        per_atom_l2 = torch.linalg.vector_norm(residual, ord=2, dim=-1)
        atom_weights = valid_atoms.to(dtype=pred.dtype)
        if not self.normalize_by_atom_count:
            if pred.ndim == 3:
                per_graph_counts = atom_weights.sum(dim=-1).clamp_min(1.0)
                self.per_sample_loss = (
                    per_atom_l2.sum(dim=-1) / per_graph_counts
                ).detach()
            return per_atom_l2.sum() / atom_weights.sum().clamp_min(1.0)
        per_graph_sum_l2, per_graph_counts = self._per_graph_atom_terms(
            per_atom_l2, atom_weights, batch_idx, num_graphs
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
            f"ignore_nonfinite={self.ignore_nonfinite!r}"
        )


class StressMSELoss(BaseLossFunction):
    """Mean-squared-error loss on the per-graph stress tensor.

    Both pred and target are shape ``(B, 3, 3)``. The loss is the mean
    of the per-graph squared-Frobenius residual, computed via
    :func:`~nvalchemi.training.losses.reductions.frobenius_mse`.

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
    """

    def __init__(
        self,
        *,
        target_key: str = "stress",
        prediction_key: str = "predicted_stress",
        ignore_nonfinite: bool = False,
    ) -> None:
        """Configure attribute keys for target and prediction."""
        super().__init__()
        self.target_key = target_key
        self.prediction_key = prediction_key
        self.ignore_nonfinite = ignore_nonfinite

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
        if self.ignore_nonfinite:
            # Per-component masking over ``(B, 3, 3)``; all-non-finite
            # graph has numerator 0 and clamped denominator 1,
            # contributing zero.
            valid = torch.isfinite(target)
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
            f"ignore_nonfinite={self.ignore_nonfinite!r}"
        )
