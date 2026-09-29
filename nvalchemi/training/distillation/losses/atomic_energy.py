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
"""Atomic energy matching loss for knowledge distillation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeAlias

import torch
from jaxtyping import Bool, Float

from nvalchemi._typing import BatchIndices
from nvalchemi.training.losses.composition import (
    BaseLossFunction,
    DTypePolicy,
    ReductionContext,
)
from nvalchemi.training.losses.reductions import graph_balanced_mean

if TYPE_CHECKING:
    from nvalchemi.data.batch import Batch

__all__ = ["AtomicEnergyMatchingLoss"]

_NodeEnergies: TypeAlias = Float[torch.Tensor, "V"]
_NodeMask: TypeAlias = Bool[torch.Tensor, "V"]


class AtomicEnergyMatchingLoss(BaseLossFunction):
    r"""Mean-squared-error loss on per-atom energies.

    Distillation targets that a total energy cannot express — how a teacher
    distributes energy across the atoms of a structure — enter training through
    this term. Prediction and target are node-level tensors of shape ``(V,)``:
    the student's per-atom energy head against the teacher's
    ``atomic_energies`` signal, which
    :class:`~nvalchemi.training.distillation.InProcessTeacherScorer` writes to
    ``teacher_atomic_energies``. The per-atom residual is

    .. math::

        \rho_{ia} = \left(\hat{\varepsilon}_{ia} - \varepsilon_{ia}\right)^2,

    where :math:`\hat{\varepsilon}_{ia}` is the student's energy for atom
    :math:`a` of graph :math:`i` and :math:`\varepsilon_{ia}` the teacher's.
    Only valid atoms enter either sum or either denominator: writing
    :math:`\mathcal{V}_i` for the atoms of graph :math:`i` that ``mask``
    accepts, :math:`M_i = |\mathcal{V}_i|`, and :math:`M = \sum_i M_i`, the
    residuals are reduced according to ``normalize_by_atom_count``:

    - ``normalize_by_atom_count=True`` (default): the mean per-atom residual of
      each graph is averaged over graphs, so every structure contributes
      equally regardless of size,

      .. math::

          L = \frac{1}{B} \sum_{i=1}^{B}
          \frac{1}{\max(M_i, 1)} \sum_{a \in \mathcal{V}_i} \rho_{ia}.

    - ``normalize_by_atom_count=False``: one global mean over all valid atoms,
      :math:`L = \tfrac{1}{\max(M, 1)} \sum_{i=1}^{B}
      \sum_{a \in \mathcal{V}_i} \rho_{ia}`, so a large structure dominates a
      small one.

    With the default ``ignore_nonfinite=True`` those counts drop every atom
    whose target energy is not finite, so a graph with one masked atom is
    divided by its finite-atom count rather than by its size, and a graph with
    none contributes ``0.0``.

    The graph-balanced reduction needs ``batch_idx`` and ``num_graphs``
    metadata, which :func:`~nvalchemi.training.losses.composition.compute_supervised_loss`
    threads from the batch automatically.

    Parameters
    ----------
    target_key : str, default "teacher_atomic_energies"
        Target container key for the teacher's per-atom energies.
    prediction_key : str, default "predicted_atomic_energies"
        Prediction container key for the student's per-atom energy head.
    normalize_by_atom_count : bool, default True
        When ``True``, compute a mean per-atom residual per graph, then mean
        over graphs. When ``False``, compute one global mean over valid atoms.
    ignore_nonfinite : bool, default True
        When ``True``, atoms whose target energy is ``NaN`` or infinite are
        excluded from both loss value and gradient using :func:`torch.isfinite`.
        A graph whose atoms are all non-finite contributes ``0.0``.
    dtype_policy : {"strict", "prediction_to_target", "target_to_prediction"}, default "strict"
        How to handle prediction/target dtype mismatches before validation.
        ``"strict"`` raises; the other policies cast one tensor to match the
        other.

    Raises
    ------
    ValueError
        If the graph-balanced reduction is requested without ``batch_idx`` and
        ``num_graphs`` metadata.

    Examples
    --------
    >>> import torch
    >>> from nvalchemi.training.distillation import AtomicEnergyMatchingLoss
    >>> loss_fn = AtomicEnergyMatchingLoss()
    >>> pred = torch.tensor([0.0, 2.0, 0.0])
    >>> target = torch.zeros(3)
    >>> batch_idx = torch.tensor([0, 0, 1])
    >>> loss_fn(pred, target, batch_idx=batch_idx, num_graphs=2)
    tensor(1.)

    Notes
    -----
    :class:`~nvalchemi.training.distillation.DistillationStrategy` derives the
    ``atomic_energies`` signal from this term's ``target_key`` and, under the
    stock ``training_fn``, refuses a student that does not compute
    ``atomic_energies``; a custom ``training_fn`` owns that contract, and a
    prediction it never produces surfaces as a missing-prediction
    :class:`KeyError` on the first batch.

    A residual below single precision is reduced in float32, so the returned
    scalar and :attr:`per_sample_loss` come back as float32 for a ``bfloat16``
    or ``float16`` prediction while gradients reach it in its own dtype.

    Per-atom energies are not observable on their own, so this term is a
    regularizer on the student's decomposition: pair it with a total-energy
    term that keeps the extensive quantity anchored.
    """

    requires_eval_grad: bool = False

    def __init__(
        self,
        *,
        target_key: str = "teacher_atomic_energies",
        prediction_key: str = "predicted_atomic_energies",
        normalize_by_atom_count: bool = True,
        ignore_nonfinite: bool = True,
        dtype_policy: DTypePolicy = "strict",
    ) -> None:
        """Configure attribute keys and per-atom energy reduction semantics."""
        super().__init__(dtype_policy=dtype_policy)
        self.target_key = target_key
        self.prediction_key = prediction_key
        self.normalize_by_atom_count = normalize_by_atom_count
        self.ignore_nonfinite = ignore_nonfinite

    def mask(
        self,
        pred: _NodeEnergies,
        target: _NodeEnergies,
        ctx: ReductionContext,
        **kwargs: Any,
    ) -> _NodeMask:
        """Return one validity flag per atom for the target energies."""
        if self.ignore_nonfinite:
            return torch.isfinite(target)
        return torch.ones_like(target, dtype=torch.bool)

    def compute_residual(
        self,
        pred: _NodeEnergies,
        target: _NodeEnergies,
        valid: _NodeMask,
    ) -> _NodeEnergies:
        """Return squared per-atom energy residuals, zeroing invalid atoms."""
        residual = torch.where(valid, pred - target, torch.zeros_like(pred))
        return residual.pow(2)

    def reduce(
        self,
        residual: _NodeEnergies,
        valid: _NodeMask,
        ctx: ReductionContext,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Reduce per-atom squared residuals to a scalar loss.

        Both branches reduce in at least float32: the graph-balanced one gets
        that from :func:`~nvalchemi.training.losses.reductions.graph_balanced_mean`,
        the global mean needs the cast itself.
        """
        if not self.normalize_by_atom_count:
            acc_dtype = torch.promote_types(residual.dtype, torch.float32)
            return residual.to(acc_dtype).sum() / valid.sum(dtype=acc_dtype).clamp_min(
                1.0
            )
        batch: Batch | None = kwargs.get("batch")
        batch_idx: BatchIndices | None = kwargs.get("batch_idx")
        num_graphs: int | None = kwargs.get("num_graphs")
        if batch is not None:
            if batch_idx is None:
                batch_idx = getattr(batch, "batch_idx", None)
            if num_graphs is None:
                num_graphs = getattr(batch, "num_graphs", None)
        loss, per_sample = graph_balanced_mean(
            residual, valid, batch_idx, num_graphs, loss_name=type(self).__name__
        )
        self.per_sample_loss = per_sample.detach()
        return loss

    def extra_repr(self) -> str:
        """Human-readable hyperparameter summary for :class:`nn.Module`'s repr."""
        return (
            f"target_key={self.target_key!r}, "
            f"prediction_key={self.prediction_key!r}, "
            f"normalize_by_atom_count={self.normalize_by_atom_count!r}, "
            f"ignore_nonfinite={self.ignore_nonfinite!r}, "
            f"dtype_policy={self.dtype_policy!r}"
        )
