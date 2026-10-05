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

from typing import Any, TypeAlias

import torch
from jaxtyping import Bool

from nvalchemi._typing import Forces
from nvalchemi.training.losses.composition import (
    BaseLossFunction,
    DTypePolicy,
    ReductionContext,
)
from nvalchemi.training.losses.reductions import masked_mean

__all__ = ["HessianMatchingLoss"]

_ForceMask: TypeAlias = Bool[torch.Tensor, "V 3"]


class HessianMatchingLoss(BaseLossFunction):
    r"""Mean-squared-error loss on Hessian-vector products.

    Energies and forces pin down the value and slope of the student's
    potential-energy surface. Its curvature decides vibrational spectra, the
    stiffness of a minimum, and whether an integrator stays stable at a given
    timestep. This term supervises that curvature without forming a Hessian.
    It compares the teacher's and student's Hessian-vector products, the
    product of each model's energy Hessian with one random probe direction
    :math:`\mathbf{v}`. The per-component residual is

    .. math::

        \rho_{ia\alpha} = \left(
        (\hat{\mathbf{H}}\mathbf{v})_{ia\alpha} -
        (\mathbf{H}\mathbf{v})_{ia\alpha} \right)^2,

    and each product costs a few backward passes per model rather than
    :math:`3V`. The residuals are reduced the way force residuals are,
    according to ``normalize_by_atom_count``. The ``hessian`` teacher signal
    writes the teacher's product to ``teacher_hvp`` and the probe to
    ``teacher_hvp_probe``.
    :func:`~nvalchemi.training.distillation.hessian_distillation_fn` computes
    the student's product along that same probe.

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
        excluded from both loss value and gradient. It defaults on because a
        marginally stable model overflows first in its second derivatives.
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
    nvalchemi.models.HessianOperator : The shared estimator.

    Notes
    -----
    ``requires_eval_grad`` is ``True`` because the prediction is a second
    derivative. Validation therefore runs with gradients enabled and costs the
    same two student passes as a training step. One probe constrains one
    direction, so coverage of the curvature comes from redrawing the probe. An
    on-policy run draws a fresh probe every time it labels a frame, whereas a
    store labeled once freezes one direction per structure.

    Each probe component is standard normal, which makes the graph-balanced
    value a Hutchinson estimate of
    :math:`\frac{1}{B}\sum_g \lVert \Delta\mathbf{H}_g \rVert_F^2 / 3V_g` in
    (eV/A^2)^2. For a near-converged student, this value is one to two orders
    of magnitude above the force mean-squared error. It is also a one-sample
    estimate whose relative spread is of order one. Start the term a hundred to
    ten thousand times lighter than the force term, and treat a single batch's
    value as noise.

    The term matches the curvature of the *energy*. A direct-force model,
    teacher or student, has a well-defined energy Hessian, but that Hessian is
    not the derivative of the forces distilled beside it. For a direct-force
    student, the term therefore supervises the energy head alone.
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
        """Reduce squared component residuals to a scalar loss.

        :func:`~nvalchemi.training.losses.reductions.masked_mean` takes the
        global mean over valid entries, or the graph-balanced one when
        ``normalize_by_atom_count`` is set, in at least float32 either way.
        """
        loss, per_sample = masked_mean(
            residual,
            valid,
            graph_balanced=self.normalize_by_atom_count,
            batch_idx=kwargs.get("batch_idx"),
            num_graphs=kwargs.get("num_graphs"),
            batch=kwargs.get("batch"),
            loss_name=type(self).__name__,
        )
        if per_sample is not None:
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
