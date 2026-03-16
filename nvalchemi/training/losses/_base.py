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
"""LossComponent ABC and CompositeLoss.

The loss system follows a two-level design:

* :class:`LossComponent` is the abstract base for individual loss terms
  (energy MSE, force MAE, etc.).  Each subclass implements
  :meth:`elementwise_error` to define the per-element error metric.
* :class:`CompositeLoss` aggregates multiple weighted ``LossComponent``
  instances.  Arithmetic operators (``+``, ``*``) on ``LossComponent``
  return a ``CompositeLoss`` for ergonomic multi-task composition.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Literal

import torch
from loguru import logger
from torch import Tensor, nn

from nvalchemi.training.losses._reductions import _segmented_sum

if TYPE_CHECKING:
    from nvalchemi._typing import ModelOutputs
    from nvalchemi.data import Batch


class LossComponent(nn.Module):
    """Abstract base class for a single loss term.

    Subclasses must implement :meth:`elementwise_error` to define the
    per-element error computation (e.g. squared error, absolute error).
    The :meth:`forward` method handles key extraction, segmented
    reduction, atom-count normalisation, and batch-level reduction.

    Attributes
    ----------
    name : str
        Human-readable name for this loss term (e.g. ``"energy_mse"``).
    pred_key : str
        Key to look up predictions in :class:`ModelOutputs`.
    target_key : str
        Key to look up targets in :class:`Batch`.
    weight : float
        Per-component weight applied *inside* :meth:`forward`.
    level : {"node", "system"}
        Whether predictions are node-level (requiring segmented
        reduction) or system-level.
    reduction : {"mean", "sum"}
        How to reduce per-graph losses to a scalar.
    normalize_by_atoms : bool
        If ``True``, divide each per-graph loss by its atom count
        before batch reduction.
    """

    def __init__(
        self,
        *,
        name: str,
        pred_key: str,
        target_key: str,
        weight: float = 1.0,
        level: Literal["node", "system"] = "system",
        reduction: Literal["mean", "sum"] = "mean",
        normalize_by_atoms: bool = False,
    ) -> None:
        super().__init__()
        self.name = name
        self.pred_key = pred_key
        self.target_key = target_key
        self.weight = weight
        self.level = level
        self.reduction = reduction
        self.normalize_by_atoms = normalize_by_atoms

    def elementwise_error(self, pred: Tensor, target: Tensor) -> Tensor:
        """Compute per-element error.

        Parameters
        ----------
        pred : Tensor
            Predictions, same shape as *target*.
        target : Tensor
            Ground-truth values.

        Returns
        -------
        Tensor
            Per-element error, same leading dimension as *pred*.

        Raises
        ------
        NotImplementedError
            Always — subclasses must override this method.
        """
        raise NotImplementedError

    def forward(self, batch: Batch, outputs: ModelOutputs) -> Tensor | None:
        """Compute the scalar loss for this component.

        Parameters
        ----------
        batch : Batch
            The current mini-batch.
        outputs : ModelOutputs
            Model predictions.

        Returns
        -------
        Tensor or None
            Scalar loss, or ``None`` if the required key is missing
            from either *batch* or *outputs*.
        """
        # 1. Check keys
        pred = outputs.get(self.pred_key)
        if pred is None:
            logger.debug(
                "LossComponent '{}': pred_key '{}' missing from outputs, skipping",
                self.name,
                self.pred_key,
            )
            return None

        if self.target_key not in batch:
            logger.debug(
                "LossComponent '{}': target_key '{}' missing from batch, skipping",
                self.name,
                self.target_key,
            )
            return None

        target = batch[self.target_key]

        # 2. Elementwise error
        error = self.elementwise_error(pred, target)

        # 3. Reduce to per-graph scalars
        if self.level == "node":
            # error shape: (V, ...) -> flatten trailing dims -> (V,)
            error_flat = error.reshape(error.shape[0], -1).sum(dim=-1)
            per_graph = self._segmented_reduce(
                error_flat, batch.batch, batch.num_graphs
            )
        else:
            # system-level: error shape (B, ...) -> reduce trailing dims -> (B,)
            per_graph = error.reshape(error.shape[0], -1).sum(dim=-1)

        # 4. Normalize by atom count
        if self.normalize_by_atoms:
            atoms_per_graph = batch.num_nodes_per_graph.to(per_graph.dtype)
            per_graph = per_graph / atoms_per_graph

        # 5. Batch reduction
        if self.reduction == "mean":
            loss = per_graph.mean()
        else:
            loss = per_graph.sum()

        return loss * self.weight

    def _segmented_reduce(
        self, values: Tensor, batch_idx: Tensor, num_segments: int
    ) -> Tensor:
        """Reduce node-level values to per-graph using segmented ops.

        Parameters
        ----------
        values : Tensor
            1-D tensor of shape ``(V,)``.
        batch_idx : Tensor
            Node-to-graph assignment, shape ``(V,)``.
        num_segments : int
            Number of graphs.

        Returns
        -------
        Tensor
            Shape ``(num_segments,)``.
        """
        return _segmented_sum(values, batch_idx, num_segments)

    # ------------------------------------------------------------------
    # Arithmetic operators for composing losses
    # ------------------------------------------------------------------

    def __add__(self, other: LossComponent | CompositeLoss) -> CompositeLoss:
        """Combine two loss components into a :class:`CompositeLoss`.

        Parameters
        ----------
        other : LossComponent or CompositeLoss
            The other loss term.

        Returns
        -------
        CompositeLoss
        """
        if isinstance(other, CompositeLoss):
            return CompositeLoss(terms=[(self, 1.0)] + other.terms)
        if isinstance(other, LossComponent):
            return CompositeLoss(terms=[(self, 1.0), (other, 1.0)])
        return NotImplemented

    def __radd__(self, other: int | LossComponent | CompositeLoss) -> CompositeLoss:
        """Support ``sum([a, b, c])`` which starts with ``0 + a``.

        Parameters
        ----------
        other : int or LossComponent or CompositeLoss
            Left operand; ``0`` for ``sum()`` bootstrapping.

        Returns
        -------
        CompositeLoss
        """
        if other == 0:
            return CompositeLoss(terms=[(self, 1.0)])
        if isinstance(other, (LossComponent, CompositeLoss)):
            return other.__add__(self)
        return NotImplemented

    def __mul__(self, coeff: float) -> CompositeLoss:
        """Scale a loss component by a coefficient.

        Parameters
        ----------
        coeff : float
            Scaling coefficient.

        Returns
        -------
        CompositeLoss
        """
        if not isinstance(coeff, (int, float)):
            return NotImplemented
        return CompositeLoss(terms=[(self, float(coeff))])

    def __rmul__(self, coeff: float) -> CompositeLoss:
        """Support ``0.5 * loss``.

        Parameters
        ----------
        coeff : float
            Scaling coefficient.

        Returns
        -------
        CompositeLoss
        """
        return self.__mul__(coeff)


class CompositeLoss(nn.Module):
    """Weighted sum of multiple :class:`LossComponent` instances.

    Created via arithmetic operators on ``LossComponent`` or by direct
    construction.

    Attributes
    ----------
    terms : list[tuple[LossComponent, float]]
        Pairs of ``(component, coefficient)``.  The coefficient scales
        the component's output *in addition* to the component's own
        ``weight`` attribute.

    Examples
    --------
    >>> loss = 1.0 * EnergyLoss() + 10.0 * ForceLoss()
    >>> total, per_term = loss(batch, outputs)
    """

    def __init__(self, *, terms: list[tuple[LossComponent, float]]) -> None:
        super().__init__()
        self.terms = terms
        # Register sub-modules for parameter tracking
        self._components = nn.ModuleList([t[0] for t in terms])

    def forward(
        self, batch: Batch, outputs: ModelOutputs
    ) -> tuple[Tensor, dict[str, Tensor | None]]:
        """Compute the composite loss.

        Parameters
        ----------
        batch : Batch
            The current mini-batch.
        outputs : ModelOutputs
            Model predictions.

        Returns
        -------
        tuple[Tensor, dict[str, Tensor | None]]
            ``(total_loss, per_term_dict)`` where *per_term_dict* maps
            each component's ``name`` to its individual loss (or ``None``
            if the key was missing).
        """
        per_term: dict[str, Tensor | None] = {}
        total = torch.tensor(0.0, device=batch.device)
        any_computed = False

        for component, coeff in self.terms:
            value = component(batch, outputs)
            per_term[component.name] = value
            if value is not None:
                total = total + coeff * value
                any_computed = True

        if not any_computed:
            warnings.warn(
                "CompositeLoss: all terms returned None (missing keys); "
                "returning zero loss",
                UserWarning,
                stacklevel=2,
            )

        return total, per_term

    # ------------------------------------------------------------------
    # Arithmetic for further composition
    # ------------------------------------------------------------------

    def __add__(self, other: LossComponent | CompositeLoss) -> CompositeLoss:
        """Combine with another loss term or composite.

        Parameters
        ----------
        other : LossComponent or CompositeLoss
            The other loss.

        Returns
        -------
        CompositeLoss
        """
        if isinstance(other, CompositeLoss):
            return CompositeLoss(terms=self.terms + other.terms)
        if isinstance(other, LossComponent):
            return CompositeLoss(terms=self.terms + [(other, 1.0)])
        return NotImplemented

    def __radd__(self, other: int | LossComponent | CompositeLoss) -> CompositeLoss:
        """Support ``sum([composite, ...])``.

        Parameters
        ----------
        other : int or LossComponent or CompositeLoss
            Left operand.

        Returns
        -------
        CompositeLoss
        """
        if other == 0:
            return self
        if isinstance(other, (LossComponent, CompositeLoss)):
            return other.__add__(self)
        return NotImplemented
