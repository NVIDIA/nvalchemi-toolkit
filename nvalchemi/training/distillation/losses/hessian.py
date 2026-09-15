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

    Energies and forces pin down the value and the slope of the student's
    potential-energy surface; its curvature is what decides vibrational spectra,
    the stiffness of a minimum, and whether an integrator stays stable at a
    given timestep. This term supervises that curvature without ever forming a
    Hessian: teacher and student are compared through their products with one
    random probe direction :math:`\mathbf{v}`,

    .. math::

        \rho_{ia\alpha} = \left(
        (\hat{\mathbf{H}}\mathbf{v})_{ia\alpha} -
        (\mathbf{H}\mathbf{v})_{ia\alpha} \right)^2,

    which costs two backward passes per model rather than the :math:`3V` a full
    Hessian would. The residuals are reduced exactly as force residuals are,
    according to ``normalize_by_atom_count``: graph-balanced by default, so
    every structure contributes equally regardless of size, and as one global
    mean over valid components when it is ``False``.

    Both sides of the comparison are materialized tensors on the batch. The
    ``hessian`` teacher signal writes the product to ``teacher_hvp`` and the
    probe it drew to ``teacher_hvp_probe``, either offline through
    :func:`~nvalchemi.training.distillation.label_dataset` or on the fly
    through the strategy's labeling seam; the student's product comes from
    :func:`~nvalchemi.training.distillation.hessian_distillation_fn`, which
    differentiates the student's energy twice along that same probe.

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
    ``requires_eval_grad`` is ``True``: the student's prediction is a second
    derivative, so validation has to run with gradients enabled like any
    force-based term — and unlike a force term, it needs a forward pass whose
    first derivative is still attached, which
    :func:`~nvalchemi.training.distillation.hessian_distillation_fn` arranges by
    taking the product on a pass of its own narrowed to the student's energy.
    Validation therefore costs the same two passes a training step does.

    One probe is one direction of a :math:`3V \times 3V` operator, so a single
    labeled batch constrains the curvature only along it. Coverage comes from
    redrawing: an on-policy run draws a fresh probe every time it labels a
    frame, so the objective sweeps directions over a run. A store labeled once
    offline freezes one direction per structure — relabel it, or mix in
    on-policy frames, when the term saturates while forces are still improving.

    The probe is drawn per component from the standard normal, which makes the
    graph-balanced value a Hutchinson estimate of a Frobenius norm,

    .. math::

        \mathbb{E}_{\mathbf{v}}[L] = \frac{1}{B} \sum_g
        \frac{\lVert \Delta\mathbf{H}_g \rVert_F^2}{3 V_g},

    with :math:`\Delta\mathbf{H}_g` the student's curvature error on graph
    :math:`g` in eV/A^2, so the term carries units of (eV/A^2)^2 and grows with
    the square of the stiffness rather than with a force. One probe is one
    sample of that estimate and its relative spread is of order one — enough for
    a gradient, misleading read as a metric. For a near-converged student the
    value sits one to two orders of magnitude above a force mean-squared error on
    the same batch, so start this term a hundred to ten thousand times lighter
    than the force term; the ratio follows the system's stiffness and is a
    starting point rather than a rule.

    The curvature being matched is that of the *energy*. A teacher whose forces
    come from a head rather than from its energy gradient still has a
    well-defined energy Hessian, but it is not the derivative of the forces
    being distilled alongside it; the two supervise the student with fields
    that need not agree, and the weight on this term is the statement of how
    much that matters. The same split runs on the student's side: a student
    whose own forces are a head output rather than an energy gradient has this
    term supervising its energy head alone, while the force head its force loss
    trains receives no curvature signal at all.
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
