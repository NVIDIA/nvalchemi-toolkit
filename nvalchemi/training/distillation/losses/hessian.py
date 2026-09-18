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
"""Hessian-vector-product matching loss for knowledge distillation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeAlias

import torch
from jaxtyping import Bool, Float

from nvalchemi._typing import BatchIndices, Forces
from nvalchemi.training.losses.composition import (
    BaseLossFunction,
    DTypePolicy,
    ReductionContext,
)
from nvalchemi.training.losses.reductions import per_graph_sum

if TYPE_CHECKING:
    from nvalchemi.data import Batch

__all__ = ["HessianMatchingLoss"]

_ForceMask: TypeAlias = Bool[torch.Tensor, "V 3"]
_PerGraphValues: TypeAlias = Float[torch.Tensor, "B"]


class HessianMatchingLoss(BaseLossFunction):
    r"""Mean-squared-error loss on Hessian-vector products.

    Energies and forces pin down the value and slope of the student's
    potential-energy surface; its curvature decides vibrational spectra, the
    stiffness of a minimum, and whether an integrator stays stable at a given
    timestep. This term supervises that curvature without forming a Hessian:
    teacher and student are compared through their products with one random
    probe direction :math:`\mathbf{v}`,

    .. math::

        \rho_{ia\alpha} = \left(
        (\hat{\mathbf{H}}\mathbf{v})_{ia\alpha} -
        (\mathbf{H}\mathbf{v})_{ia\alpha} \right)^2,

    two backward passes per model rather than :math:`3V`. The residuals are
    reduced as force residuals are, according to ``normalize_by_atom_count``.
    The ``hessian`` teacher signal writes the product to ``teacher_hvp`` and
    the probe to ``teacher_hvp_probe``; the student's product comes from
    :func:`~nvalchemi.training.distillation.hessian_distillation_fn` along
    that same probe.

    Parameters
    ----------
    target_key : str, default "teacher_hvp"
        Target container key for the teacher's Hessian-vector product.
    prediction_key : str, default "predicted_hvp"
        Prediction container key for the student's Hessian-vector product.
    normalize_by_atom_count : bool, default True
        When ``True``, compute a mean squared residual per graph, then mean
        over graphs. When ``False``, compute one global mean over valid
        components.
    ignore_nonfinite : bool, default True
        When ``True``, components whose target is ``NaN`` or infinite are
        excluded from both loss value and gradient. Second derivatives are
        where a marginally stable model overflows first, so this defaults on.
    dtype_policy : {"strict", "prediction_to_target", "target_to_prediction"}, default "strict"
        How to handle prediction/target dtype mismatches before validation.

    Raises
    ------
    ValueError
        If the graph-balanced reduction is requested without ``batch_idx`` and
        ``num_graphs`` metadata.

    Examples
    --------
    >>> import torch
    >>> from nvalchemi.training.distillation import HessianMatchingLoss
    >>> loss_fn = HessianMatchingLoss()
    >>> pred = torch.tensor([[2.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    >>> target = torch.zeros(2, 3)
    >>> batch_idx = torch.tensor([0, 1])
    >>> loss_fn(pred, target, batch_idx=batch_idx, num_graphs=2)
    tensor(0.6667)

    See Also
    --------
    nvalchemi.training.distillation.hessian_vector_product : The shared estimator.

    Notes
    -----
    ``requires_eval_grad`` is ``True``: the prediction is a second derivative,
    so validation runs with gradients enabled and costs the same two student
    passes a training step does. One probe constrains one direction, so
    coverage comes from redrawing: an on-policy run draws a fresh probe every
    time it labels a frame, while a store labeled once freezes one direction
    per structure.

    The probe is standard normal per component, which makes the graph-balanced
    value a Hutchinson estimate of
    :math:`\frac{1}{B}\sum_g \lVert \Delta\mathbf{H}_g \rVert_F^2 / 3V_g` in
    (eV/A^2)^2 — one to two orders of magnitude above a force mean-squared
    error for a near-converged student, and a one-sample estimate whose
    relative spread is of order one. Start the term a hundred to ten thousand
    times lighter than the force term, and read a single batch's value as
    noise.

    The curvature matched is the *energy's*. A direct-force model, teacher or
    student, has a well-defined energy Hessian that is not the derivative of
    the forces distilled beside it; for such a student the term supervises the
    energy head alone.
    """

    requires_eval_grad: bool = True

    def __init__(
        self,
        *,
        target_key: str = "teacher_hvp",
        prediction_key: str = "predicted_hvp",
        normalize_by_atom_count: bool = True,
        ignore_nonfinite: bool = True,
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
        pred: Forces,
        target: Forces,
        ctx: ReductionContext,
        **kwargs: Any,
    ) -> _ForceMask:
        """Return one validity flag per Cartesian component."""
        if self.ignore_nonfinite:
            return torch.isfinite(target)
        return torch.ones_like(target, dtype=torch.bool)

    def compute_residual(
        self,
        pred: Forces,
        target: Forces,
        valid: _ForceMask,
    ) -> Forces:
        """Return squared component residuals, zeroing invalid components."""
        residual = torch.where(valid, pred - target, torch.zeros_like(pred))
        return residual.pow(2)

    def reduce(
        self,
        residual: Forces,
        valid: _ForceMask,
        ctx: ReductionContext,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Reduce squared component residuals to a scalar loss."""
        valid_components = valid.to(dtype=residual.dtype)
        if not self.normalize_by_atom_count:
            return residual.sum() / valid_components.sum().clamp_min(1.0)
        batch: Batch | None = kwargs.get("batch")
        batch_idx: BatchIndices | None = kwargs.get("batch_idx")
        num_graphs: int | None = kwargs.get("num_graphs")
        if batch is not None:
            if batch_idx is None:
                batch_idx = getattr(batch, "batch_idx", None)
            if num_graphs is None:
                num_graphs = getattr(batch, "num_graphs", None)
        per_graph_residual, per_graph_counts = self._per_graph_terms(
            residual, valid_components, batch_idx, num_graphs
        )
        per_sample = per_graph_residual / per_graph_counts.clamp_min(1.0)
        self.per_sample_loss = per_sample.detach()
        return per_sample.mean()

    def _per_graph_terms(
        self,
        residual: Forces,
        valid_components: Forces,
        batch_idx: BatchIndices | None,
        num_graphs: int | None,
    ) -> tuple[_PerGraphValues, _PerGraphValues]:
        """Return per-graph residual sums and valid component counts."""
        if batch_idx is None or num_graphs is None:
            raise ValueError(
                "HessianMatchingLoss needs batch_idx and num_graphs metadata for "
                f"its graph-balanced reduction; got batch_idx={batch_idx!r}, "
                f"num_graphs={num_graphs!r}."
            )
        return (
            per_graph_sum(residual.sum(dim=-1), batch_idx, num_graphs=num_graphs),
            per_graph_sum(
                valid_components.sum(dim=-1), batch_idx, num_graphs=num_graphs
            ),
        )

    def extra_repr(self) -> str:
        """Human-readable hyperparameter summary for :class:`nn.Module`'s repr."""
        return (
            f"target_key={self.target_key!r}, "
            f"prediction_key={self.prediction_key!r}, "
            f"normalize_by_atom_count={self.normalize_by_atom_count!r}, "
            f"ignore_nonfinite={self.ignore_nonfinite!r}, "
            f"dtype_policy={self.dtype_policy!r}"
        )
