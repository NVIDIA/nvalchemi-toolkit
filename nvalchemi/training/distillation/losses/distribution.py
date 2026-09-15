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
"""Boltzmann-distribution matching loss for on-policy knowledge distillation."""

from __future__ import annotations

from typing import Any, TypeAlias

import torch
from jaxtyping import Bool

from nvalchemi._typing import Energy
from nvalchemi.dynamics.hooks._utils import KB_EV
from nvalchemi.training.losses.composition import (
    BaseLossFunction,
    DTypePolicy,
    ReductionContext,
)

__all__ = ["BoltzmannMatchingLoss"]

_EnergyMask: TypeAlias = Bool[torch.Tensor, "B 1"]

_ENSEMBLE_REMEDY = (
    "A Boltzmann distribution is defined over the configurations of one "
    "system, so every graph in a batch has to be a configuration of the same "
    "one. Seed the on-policy run with replicas of a single structure — one "
    "walker per graph — and set replay_ratio=1 so no anchor rows are mixed in."
)
"""What to do about a batch that is not one system's ensemble."""


class BoltzmannMatchingLoss(BaseLossFunction):
    r"""Relative entropy between the teacher's and student's Boltzmann ensembles.

    Energy and force matching are pointwise: they ask the student to reproduce
    the teacher's numbers configuration by configuration. This term asks for
    something coarser and more directly useful — that the student's
    *distribution* match the teacher's, which is what decides whether a
    simulation driven by the student visits the same states with the same
    frequencies. It is therefore blind to a constant energy offset, and to any
    error that does not change relative populations.

    Both distributions are the canonical ensemble at ``temperature``
    :math:`T`. Writing :math:`u(x) = U(x) / k_\mathrm{B}T` for a reduced
    energy, the teacher's :math:`p \propto e^{-u_T}` and the student's
    :math:`q \propto e^{-u_S}` are compared on the batch's own configurations
    :math:`\{x_i\}_{i=1}^{B}`, which the on-policy loop drew from the student's
    own trajectory — that is, from :math:`q` itself. The empirical
    distribution of the batch therefore *is* the student's, :math:`\hat q_i =
    1/B`, and the teacher's is what reweighting those same samples gives:

    .. math::

        \Delta_i = \frac{U_T(x_i) - U_S(x_i)}{k_\mathrm{B}T}, \qquad
        \hat p_i = \frac{e^{-\Delta_i}}{\sum_j e^{-\Delta_j}}.

    Both relative entropies then follow in closed form from
    :math:`\ell_i = \log(B \hat p_i)`,

    .. math::

        D_{\mathrm{KL}}(\hat p \Vert \hat q) = \sum_i \hat p_i \ell_i,
        \qquad
        D_{\mathrm{KL}}(\hat q \Vert \hat p) = -\frac{1}{B} \sum_i \ell_i,

    and ``beta`` interpolates between them, sweeping the generalized
    Jensen-Shannon family's two endpoints:

    .. math::

        L(\beta) = (1 - \beta)\, D_{\mathrm{KL}}(\hat p \Vert \hat q)
        + \beta\, D_{\mathrm{KL}}(\hat q \Vert \hat p).

    ``beta=0`` is the forward, mass-covering direction: it is dominated by the
    configurations the *teacher* considers likely, and punishes a student that
    assigns them too little weight. ``beta=1`` is the reverse, mode-seeking
    direction: it is dominated by the configurations the student actually
    visits, and punishes a student that visits states the teacher considers
    unlikely — the failure that makes a small student's trajectory drift off
    the teacher's manifold. Both terms are non-negative and vanish together
    exactly when :math:`U_T - U_S` is constant across the batch, which is the
    invariance an ensemble objective should have.

    Parameters
    ----------
    target_key : str, default "teacher_energy"
        Target container key for the teacher's total energies, shape ``(B, 1)``.
    prediction_key : str, default "predicted_energy"
        Prediction container key for the student's total energies.
    beta : float, default 0.5
        Interpolation between the forward (``0``) and reverse (``1``) relative
        entropy. Must lie in ``[0, 1]``.
    temperature : float, default 300.0
        Ensemble temperature in Kelvin. Should match the temperature the
        on-policy propagator samples at; see the Notes.
    ignore_nonfinite : bool, default True
        When ``True``, graphs whose target energy is ``NaN`` or infinite are
        dropped from the ensemble entirely rather than poisoning every weight.
    dtype_policy : {"strict", "prediction_to_target", "target_to_prediction"}, default "strict"
        How to handle prediction/target dtype mismatches before validation.

    Raises
    ------
    ValueError
        If ``beta`` falls outside ``[0, 1]``, if ``temperature`` is not
        positive, or if the batch's graphs do not all hold the same number of
        atoms, which no single Boltzmann distribution can describe — the last
        only when the batch's ``num_nodes_per_graph`` metadata reaches the
        term; see the Notes.

    Examples
    --------
    >>> import torch
    >>> from nvalchemi.training.distillation import BoltzmannMatchingLoss
    >>> loss_fn = BoltzmannMatchingLoss(beta=1.0)
    >>> pred = torch.tensor([[0.0], [0.0]])
    >>> target = torch.tensor([[0.0], [0.0]])
    >>> loss_fn(pred, target)
    tensor(0.)

    Notes
    -----
    The estimator assumes the batch was drawn from the student's own ensemble,
    which is why
    :class:`~nvalchemi.training.distillation.DistillationStrategy` refuses this
    term without ``on_policy``, refuses to reuse the objective as a validation
    loss, and warns when ``replay_ratio`` mixes anchor frames into the batch or
    when the replay buffer is unbounded: an off-policy sample carries importance
    weights this form has folded away as uniform. It further assumes the
    propagator samples the canonical ensemble at ``temperature`` — a relaxation
    propagator samples nothing, and the strategy rejects that pairing, but a
    thermostat set to a different temperature than this term is a mismatch
    nothing can detect from the batch. Set both from the same number.

    Validation data is the off-policy case that looks legitimate. A held-out set
    is a fixed sample of whatever produced it rather than of the student's
    ensemble, its graphs need not be one system's configurations, and reducing
    energies by :math:`k_\mathrm{B}T` makes this term large next to a pointwise
    one, so it would dominate the composite metric that checkpoint selection and
    the metric schedulers read. Give
    :class:`~nvalchemi.training.ValidationConfig` a pointwise loss of its own
    rather than letting it reuse an objective this term is part of.

    The one-ensemble precondition is checked rather than enforced. The check
    reads the ``num_nodes_per_graph`` metadata
    :func:`~nvalchemi.training.losses.composition.compute_supervised_loss`
    forwards, so a direct call or a custom ``loss_target_assembler`` that does
    not supply it passes silently, and equal atom counts are necessary rather
    than sufficient: two different species of the same size clear the check
    while their total energies differ by tens of eV, which puts the whole
    softmax on one of them. Composition itself is not checkable here, because a
    loss term is handed the batch's graph metadata rather than its atomic
    numbers. Seed the run with replicas of one structure and the precondition
    holds by construction.

    Gradients flow through the energies of a fixed set of configurations; the
    dependence of the sampling distribution itself on the student's parameters
    is not differentiated. That is the usual on-policy approximation, and it is
    the second reason segments have to keep regenerating: the samples are only
    the student's for as long as the weights that produced them are current.
    Regenerating keeps current frames arriving but does not by itself make a
    batch current, because the batch is a uniform draw over the whole replay
    buffer rather than over the segment that just ran. An unbounded buffer
    retires nothing, so after ``N`` segments only about one ``N``-th of a batch
    came from the current student; what bounds the staleness of the sample is
    :attr:`~nvalchemi.training.distillation.OnPolicyConfig.replay_capacity`.

    Batch size is the estimator's resolution. A batch is one Monte Carlo sample
    of the two distributions, so a single-graph batch reports exactly ``0.0``,
    a handful of graphs gives a high-variance signal, and the self-normalized
    weights are biased at any finite size — pair it with a pointwise energy or
    force term rather than running on it alone.

    It also caps what the forward direction can say. Self-normalized weights
    make :math:`D_{\mathrm{KL}}(\hat p \Vert \hat q)` a divergence against the
    uniform distribution on the batch's own :math:`B` points, so it cannot
    exceed :math:`\log B` — the whole term is bounded by :math:`(1 - \beta)\log
    B + \beta D_{\mathrm{KL}}(\hat q \Vert \hat p)` — and it reaches that
    ceiling as soon as the softmax saturates, which a student whose error
    spreads over more than roughly four :math:`k_\mathrm{B}T` already does. Its
    gradient vanishes there: once the teacher's weight sits entirely on one
    configuration the batch says only that, however wrong the rest are. So
    ``beta=0`` can report a flat value near :math:`\log B` and move nothing,
    which is indistinguishable from convergence, and at the default half the
    term is dead weight until the student is close. The reverse direction has no
    such ceiling and keeps pulling, so hold ``beta`` at ``0.5`` or ``1.0`` until
    the student is within a couple of :math:`k_\mathrm{B}T`, and lower it only
    then, if the mass-covering direction is what you are after.

    Reducing energies by :math:`k_\mathrm{B}T` sets the term's scale too. The
    reverse direction's gradient with respect to one configuration's energy is
    exactly :math:`(\hat p_i - 1/B)/k_\mathrm{B}T`, and the forward one,
    :math:`\hat p_i(\ell_i - D_{\mathrm{KL}}(\hat p \Vert \hat
    q))/k_\mathrm{B}T`, is bounded by the same :math:`1/k_\mathrm{B}T` — about
    39 eV^-1 at 300 K, one to two orders above what a pointwise energy term
    produces on the same residuals. Weight it accordingly rather than composing
    it at parity.
    """

    requires_eval_grad: bool = False

    def __init__(
        self,
        *,
        target_key: str = "teacher_energy",
        prediction_key: str = "predicted_energy",
        beta: float = 0.5,
        temperature: float = 300.0,
        ignore_nonfinite: bool = True,
        dtype_policy: DTypePolicy = "strict",
    ) -> None:
        """Configure attribute keys, the KL direction, and the ensemble temperature."""
        super().__init__(dtype_policy=dtype_policy)
        if not 0.0 <= beta <= 1.0:
            raise ValueError(
                "beta interpolates between the forward and reverse relative "
                f"entropy, so it must lie in [0, 1]; got beta={beta!r}."
            )
        if temperature <= 0.0:
            raise ValueError(
                "temperature sets the ensemble the energies are compared in and "
                f"must be positive Kelvin; got temperature={temperature!r}."
            )
        self.target_key = target_key
        self.prediction_key = prediction_key
        self.beta = beta
        self.temperature = temperature
        self.ignore_nonfinite = ignore_nonfinite

    @property
    def reduced_energy_scale(self) -> float:
        """Thermal energy ``k_B T`` in eV, the unit energies are reduced by."""
        return KB_EV * self.temperature

    def normalize(
        self,
        pred: Energy,
        target: Energy,
        **kwargs: Any,
    ) -> tuple[Energy, Energy, ReductionContext]:
        """Check the batch is one system's ensemble, then pass the energies through."""
        counts = kwargs.get("num_nodes_per_graph")
        if (
            counts is not None
            and counts.numel() > 1
            and not bool((counts == counts[0]).all())
        ):
            raise ValueError(
                "BoltzmannMatchingLoss compares the energies of one ensemble, "
                "but the batch holds graphs of different sizes, whose energies "
                "are not comparable at all: got atom counts "
                f"{sorted(set(counts.tolist()))!r}. {_ENSEMBLE_REMEDY}"
            )
        return pred, target, ReductionContext()

    def mask(
        self,
        pred: Energy,
        target: Energy,
        ctx: ReductionContext,
        **kwargs: Any,
    ) -> _EnergyMask:
        """Return one validity flag per graph of the ensemble."""
        if self.ignore_nonfinite:
            return torch.isfinite(target)
        return torch.ones_like(target, dtype=torch.bool)

    def compute_residual(
        self,
        pred: Energy,
        target: Energy,
        valid: _EnergyMask,
    ) -> Energy:
        """Return the log teacher weight of each graph, relative to a uniform one."""
        count = valid.sum()
        if count == 0:
            return torch.zeros_like(target)
        delta = (target - pred) / self.reduced_energy_scale
        logits = torch.where(valid, -delta, torch.full_like(delta, -torch.inf))
        log_weights = torch.log_softmax(logits, dim=0)
        offset = torch.log(count.to(dtype=target.dtype))
        return torch.where(valid, log_weights + offset, torch.zeros_like(target))

    def reduce(
        self,
        residual: Energy,
        valid: _EnergyMask,
        ctx: ReductionContext,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Combine the log weights into the beta-interpolated relative entropy."""
        valid_graphs = valid.to(dtype=residual.dtype)
        count = valid_graphs.sum().clamp_min(1.0)
        weights = torch.where(valid, residual.exp() / count, torch.zeros_like(residual))
        forward = (weights * residual).sum()
        reverse = -(residual * valid_graphs).sum() / count
        num_graphs = residual.shape[0]
        per_sample = (
            num_graphs * (1.0 - self.beta) * weights * residual
            - (self.beta * num_graphs / count) * residual * valid_graphs
        )
        self.per_sample_loss = per_sample.reshape(num_graphs).detach()
        return (1.0 - self.beta) * forward + self.beta * reverse

    def extra_repr(self) -> str:
        """Human-readable hyperparameter summary for :class:`nn.Module`'s repr."""
        return (
            f"target_key={self.target_key!r}, "
            f"prediction_key={self.prediction_key!r}, "
            f"beta={self.beta!r}, "
            f"temperature={self.temperature!r}, "
            f"ignore_nonfinite={self.ignore_nonfinite!r}, "
            f"dtype_policy={self.dtype_policy!r}"
        )
