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
from torch import distributed as dist
from torch.distributed.nn.functional import all_gather as _differentiable_all_gather

from nvalchemi._typing import Energy
from nvalchemi.dynamics.hooks._utils import KB_EV
from nvalchemi.training.losses.composition import (
    BaseLossFunction,
    DTypePolicy,
    ReductionContext,
)

__all__ = ["BoltzmannMatchingLoss"]

_EnergyMask: TypeAlias = Bool[torch.Tensor, "B 1"]

_ONE_SYSTEM_REMEDY = (
    "A Boltzmann distribution is defined over the configurations of one "
    "system, so every graph in a batch has to be a configuration of the same "
    "one. Seed the on-policy run with replicas of a single structure — one "
    "walker per graph — and set replay_ratio=1 so no reference rows are mixed in."
)
"""What to do about a batch that is not one system's configurations."""


def _world_batch(gaps: Energy, valid: _EnergyMask) -> tuple[Energy, _EnergyMask, slice]:
    """Return every rank's reduced energy gaps and validity, and this rank's rows in them.

    Under data parallelism each rank holds a shard of one world batch, and a
    softmax over the shard alone would weight a rank's configurations against
    each other rather than against the whole sample. The gaps travel through an
    autograd-aware all-gather, so every rank computes the same world loss and
    the gradient reaching a shard's energies sums every rank's copy of it,
    which the data-parallel mean over ranks turns back into the world loss's
    own gradient. Shards of unequal size are padded to the largest and trimmed
    again. Without an initialized process group, or with one rank, the batch is
    its own world.
    """
    if (
        not (dist.is_available() and dist.is_initialized())
        or dist.get_world_size() == 1
    ):
        return gaps, valid, slice(None)
    world_size = dist.get_world_size()
    count = torch.tensor([gaps.shape[0]], device=gaps.device)
    counts = [torch.zeros_like(count) for _ in range(world_size)]
    dist.all_gather(counts, count)
    sizes = [int(size) for size in counts]
    padding = (0, 0, 0, max(sizes) - gaps.shape[0])
    padded_valid = torch.nn.functional.pad(valid.to(gaps.dtype), padding)
    gathered_valid = [torch.zeros_like(padded_valid) for _ in range(world_size)]
    dist.all_gather(gathered_valid, padded_valid)
    gathered_gaps = _differentiable_all_gather(torch.nn.functional.pad(gaps, padding))
    world_gaps = torch.cat(
        [shard[:size] for shard, size in zip(gathered_gaps, sizes, strict=True)]
    )
    world_valid = (
        torch.cat(
            [shard[:size] for shard, size in zip(gathered_valid, sizes, strict=True)]
        )
        > 0.5
    )
    start = sum(sizes[: dist.get_rank()])
    return world_gaps, world_valid, slice(start, start + gaps.shape[0])


class BoltzmannMatchingLoss(BaseLossFunction):
    r"""Relative entropy between the teacher's and student's Boltzmann distributions.

    Energy and force matching are pointwise; this term asks that the student's
    *distribution* over configurations match the teacher's, which is what
    decides whether a simulation driven by the student visits the same states
    with the same frequencies. It is blind to a constant energy offset and to
    any error that does not change relative populations.

    Both distributions are the canonical ensemble at ``temperature`` :math:`T`.
    With reduced energies :math:`u = U / k_\mathrm{B}T`, the batch's
    configurations :math:`\{x_i\}_{i=1}^{B}` are read as a sample of the
    *student's* distribution, so its empirical weights are uniform,
    :math:`\hat q_i = 1/B`, and the teacher's follow by reweighting:

    .. math::

        \Delta_i = \frac{U_T(x_i) - U_S(x_i)}{k_\mathrm{B}T}, \qquad
        \hat p_i = \frac{e^{-\Delta_i}}{\sum_j e^{-\Delta_j}}, \qquad
        \ell_i = \log(B \hat p_i).

    ``beta`` interpolates the forward, mass-covering direction
    :math:`D_{\mathrm{KL}}(\hat p \Vert \hat q) = \sum_i \hat p_i \ell_i`
    (``0``) and the reverse, mode-seeking one
    :math:`D_{\mathrm{KL}}(\hat q \Vert \hat p) = -\frac{1}{B}\sum_i \ell_i`
    (``1``). Both vanish exactly when :math:`U_T - U_S` is constant across the
    batch.

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
        Ensemble temperature in Kelvin; set it from the same number as the
        on-policy thermostat, which nothing here can check.
    ignore_nonfinite : bool, default True
        When ``True``, graphs whose target energy is ``NaN`` or infinite are
        dropped from the distribution rather than poisoning every weight.
    dtype_policy : {"strict", "prediction_to_target", "target_to_prediction"}, default "strict"
        How to handle prediction/target dtype mismatches before validation.

    Raises
    ------
    ValueError
        If ``beta`` falls outside ``[0, 1]``, if ``temperature`` is not
        positive, or if the batch's graphs do not all hold the same number of
        atoms — the last only when ``num_nodes_per_graph`` metadata reaches the
        term, which a direct call does not supply.

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
    The uniform student weights hold only for a batch the student itself
    generated, which is why
    :class:`~nvalchemi.training.distillation.DistillationStrategy` requires
    ``on_policy``, refuses the term on the validation side, and warns when
    ``replay_ratio`` mixes reference frames in or the replay buffer is
    unbounded — a uniform draw over a buffer nothing retires from is a draw
    over every policy the run has had, so
    :attr:`~nvalchemi.training.distillation.OnPolicyConfig.replay_capacity`
    bounds the staleness. The dependence of the sampling distribution on the
    student's parameters is not differentiated, the usual on-policy
    approximation. Equal atom counts are checked but are necessary rather than
    sufficient; seed the run with replicas of one structure.

    Under data parallelism every rank holds a shard of one world batch, so the
    reduced energies are gathered across ranks with an autograd-aware
    all-gather and the softmax is normalized over the world batch: every rank
    reports the world loss, and the gradient the data-parallel mean produces is
    the world loss's own. The gather is a collective, so every rank has to reach
    the term on every step; without a process group, or with one rank, the
    batch is its own world.

    A batch is one Monte Carlo sample of the two distributions: a single graph
    reports ``0.0``, a handful gives a high-variance signal, and the
    self-normalized weights are biased at any finite size, so pair the term
    with a pointwise one. The forward direction is bounded by :math:`\log B`
    and its gradient vanishes once the softmax saturates — a student whose
    error spreads over more than a few :math:`k_\mathrm{B}T` — so ``beta=0``
    can read as converged while the student is far off; hold ``beta`` at
    ``0.5`` or above until the student is within a couple of
    :math:`k_\mathrm{B}T`. Either direction's gradient per configuration is
    bounded by :math:`1/k_\mathrm{B}T`, about 39 eV^-1 at 300 K, one to two
    orders above a pointwise energy term's, so weight it accordingly.
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
    def thermal_energy(self) -> float:
        """Thermal energy ``k_B T`` in eV, the unit energies are reduced by."""
        return KB_EV * self.temperature

    def normalize(
        self,
        pred: Energy,
        target: Energy,
        **kwargs: Any,
    ) -> tuple[Energy, Energy, ReductionContext]:
        """Check the batch is one system's configurations, then pass the energies through."""
        counts = kwargs.get("num_nodes_per_graph")
        if (
            counts is not None
            and counts.numel() > 1
            and not bool((counts == counts[0]).all())
        ):
            raise ValueError(
                "BoltzmannMatchingLoss compares the energies of one system's "
                "configurations, but the batch holds graphs of different sizes, whose energies "
                "are not comparable at all: got atom counts "
                f"{sorted(set(counts.tolist()))!r}. {_ONE_SYSTEM_REMEDY}"
            )
        return pred, target, ReductionContext()

    def mask(
        self,
        pred: Energy,
        target: Energy,
        ctx: ReductionContext,
        **kwargs: Any,
    ) -> _EnergyMask:
        """Return one validity flag per graph of the batch."""
        if self.ignore_nonfinite:
            return torch.isfinite(target)
        return torch.ones_like(target, dtype=torch.bool)

    def compute_residual(
        self,
        pred: Energy,
        target: Energy,
        valid: _EnergyMask,
    ) -> Energy:
        """Return each graph's reduced energy gap, zero where its target is invalid.

        An invalid graph's zero is still attached to *pred*, so a batch with no
        valid graph backpropagates a zero update.
        """
        gap = (target - pred) / self.thermal_energy
        return torch.where(valid, gap, pred * 0.0)

    def reduce(
        self,
        residual: Energy,
        valid: _EnergyMask,
        ctx: ReductionContext,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Combine the world batch's gaps into the beta-interpolated relative entropy.

        ``per_sample_loss`` holds this rank's graphs, scaled so that its mean
        over the world batch is the scalar loss.
        """
        gaps, world_valid, rows = _world_batch(residual, valid)
        count = world_valid.sum()
        if count == 0:
            self.per_sample_loss = torch.zeros_like(residual).reshape(-1)
            return residual.sum() * 0.0
        logits = torch.where(world_valid, -gaps, torch.full_like(gaps, -torch.inf))
        log_ratio = torch.log_softmax(logits, dim=0) + torch.log(count.to(gaps.dtype))
        log_ratio = torch.where(world_valid, log_ratio, torch.zeros_like(gaps))
        weights = torch.where(
            world_valid, log_ratio.exp() / count, torch.zeros_like(gaps)
        )
        per_graph = (1.0 - self.beta) * weights * log_ratio - (
            self.beta / count
        ) * log_ratio
        self.per_sample_loss = (gaps.shape[0] * per_graph[rows]).reshape(-1).detach()
        return per_graph.sum()

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
