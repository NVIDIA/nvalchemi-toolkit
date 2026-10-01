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
from nvalchemi.dynamics.hooks import KB_EV
from nvalchemi.training.distributed import (
    all_gather_objects,
    all_gather_rows,
    get_world_size,
    is_distributed_initialized,
)
from nvalchemi.training.losses.composition import (
    BaseLossFunction,
    DTypePolicy,
    ReductionContext,
)

__all__ = ["BoltzmannMatchingLoss"]

_EnergyMask: TypeAlias = Bool[torch.Tensor, "B 1"]

_ONE_SYSTEM_REMEDY = (
    "A Boltzmann distribution is defined over the configurations of one "
    "system, so the graphs in a batch have to be configurations of one system "
    "(same atoms, same count). Seed the on-policy run with replicas of a single "
    "structure, one walker per graph, and set replay_ratio=1 so no reference "
    "rows are mixed in."
)
"""Remedy for a batch whose graphs are not configurations of one system."""


def _gathers_world(world_batch: bool | None) -> bool:
    """Return whether the term reduces over every rank's shard of the batch.

    ``None`` gathers when a process group with more than one rank is
    initialized. ``False`` never gathers. ``True`` always gathers, so a run that
    means to train on the world batch fails loudly without a process group
    instead of silently reducing over its own shard.
    """
    initialized = is_distributed_initialized()
    if world_batch is None:
        return initialized and get_world_size() > 1
    if world_batch and not initialized:
        raise RuntimeError(
            "BoltzmannMatchingLoss(world_batch=True) gathers every rank's energy "
            "gaps before the softmax, but no process group is initialized; "
            "initialize one, or leave world_batch=None to gather only under one."
        )
    return world_batch


class BoltzmannMatchingLoss(BaseLossFunction):
    r"""Relative entropy between the teacher's and student's Boltzmann distributions.

    Energy and force matching are pointwise: they compare the two models one
    configuration at a time. Boltzmann matching instead asks that the
    student's Boltzmann *distribution* over configurations match the
    teacher's. That distribution decides whether a simulation driven by the
    student visits the same states with the same frequencies as one driven by
    the teacher. The term is blind to a constant energy offset and to any
    error that does not change relative populations.

    Both distributions are the canonical ensemble at ``temperature`` :math:`T`,
    with reduced energies :math:`u = U / k_\mathrm{B}T`. The batch's
    configurations :math:`\{x_i\}_{i=1}^{B}` are read as a sample of the
    *student's* distribution, so the student's empirical weights are uniform,
    :math:`\hat q_i = 1/B`. The teacher's weights follow by reweighting:

    .. math::

        \Delta_i = \frac{U_T(x_i) - U_S(x_i)}{k_\mathrm{B}T}, \qquad
        \hat p_i = \frac{e^{-\Delta_i}}{\sum_j e^{-\Delta_j}}, \qquad
        \ell_i = \log(B \hat p_i).

    ``beta`` interpolates between the forward, mass-covering direction
    :math:`D_{\mathrm{KL}}(\hat p \Vert \hat q) = \sum_i \hat p_i \ell_i`
    (``beta=0``) and the reverse, mode-seeking direction
    :math:`D_{\mathrm{KL}}(\hat q \Vert \hat p) = -\frac{1}{B}\sum_i \ell_i`
    (``beta=1``). Both vanish exactly when :math:`U_T - U_S` is constant across
    the batch.

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
        Ensemble temperature in Kelvin. Set it to the on-policy thermostat's
        temperature; nothing here can check that the two agree.
    ignore_nonfinite : bool, default True
        When ``True``, graphs whose target or predicted energy is ``NaN`` or
        infinite are dropped from the distribution. Otherwise one non-finite
        student energy would poison every weight, and through the cross-rank
        gather (see ``world_batch``) it would reach every rank's softmax.
        ``False`` skips the check, for a run trusted to generate only finite
        energies.
    dtype_policy : {"strict", "prediction_to_target", "target_to_prediction"}, default "strict"
        How to handle prediction/target dtype mismatches before validation.
    world_batch : bool | None, default None
        Whether the softmax is normalized over the world batch, the union of
        every rank's shard of the batch. ``None`` gathers whenever a process
        group with more than one rank is initialized. ``True`` always gathers
        and refuses to run without a process group. ``False`` reduces over the
        local batch alone, for a run whose ranks hold independent ensembles.
    check_one_system : bool, default True
        Whether the term refuses a batch whose graphs hold different numbers
        of atoms. Equal atom counts are necessary but not sufficient for the
        graphs to be configurations of one system. Under data parallelism the
        check costs one ``all_gather_object`` collective per step. Disable it
        for an ensemble whose composition legitimately varies, such as a
        grand-canonical run or a student compared across compositions on
        purpose. The caller then takes responsibility for the energies being
        comparable.

    Raises
    ------
    ValueError
        If ``beta`` falls outside ``[0, 1]`` or ``temperature`` is not
        positive. Also if the batch's graphs (every rank's shard of it, under
        data parallelism) do not all hold the same number of atoms. That check
        runs only when ``check_one_system`` is ``True`` and
        ``num_nodes_per_graph`` metadata reaches the term, which a direct call
        does not supply.
    RuntimeError
        If ``world_batch=True`` and no process group is initialized when the
        term is evaluated.

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
    generated. :class:`~nvalchemi.training.distillation.DistillationStrategy`
    therefore requires ``on_policy`` and refuses the term on the validation
    side. It also warns when ``replay_ratio`` mixes reference frames in or when
    the replay buffer is unbounded. A uniform draw over a buffer that never
    retires frames is a draw over every policy the run has had, and
    :attr:`~nvalchemi.training.distillation.OnPolicyConfig.replay_capacity`
    bounds that staleness. As in the usual on-policy approximation, the
    dependence of the sampling distribution on the student's parameters is not
    differentiated. Equal atom counts are checked (on the gathered world batch,
    under data parallelism), but they are necessary rather than sufficient.
    Seed the run with replicas of one structure, and turn ``check_one_system``
    off only where the composition is meant to vary.

    Under data parallelism every rank holds a shard of one world batch. The
    reduced energies are gathered across ranks with an autograd-aware
    all-gather, and the softmax is normalized over the world batch. Every rank
    therefore reports the world loss, and the gradient the data-parallel mean
    produces is the world loss's own. The gather is a collective, so every rank
    has to reach the term on every step. Without a process group, or with one
    rank, the local batch is the world batch. ``world_batch`` makes the gather
    decision explicit.

    A batch is one Monte Carlo sample of the two distributions. A single graph
    reports ``0.0``, a handful of graphs gives a high-variance signal, and the
    self-normalized weights are biased at any finite batch size, so pair this
    term with a pointwise one. The forward direction is bounded by
    :math:`\log B`, and its gradient vanishes once the softmax saturates, as it
    does for a student whose error spreads over more than a few
    :math:`k_\mathrm{B}T`. ``beta=0`` can therefore read as converged while the
    student is far off. Hold ``beta`` at ``0.5`` or above until the student is
    within a couple of :math:`k_\mathrm{B}T`. Either direction's gradient per
    configuration is bounded by :math:`1/k_\mathrm{B}T`, about 39 eV^-1 at
    300 K. That bound is one to two orders of magnitude above a pointwise
    energy term's, so weight the term accordingly.
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
        world_batch: bool | None = None,
        check_one_system: bool = True,
    ) -> None:
        """Configure attribute keys, the KL direction, the ensemble temperature, and the gather."""
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
        self.world_batch = world_batch
        self.check_one_system = check_one_system

    @property
    def thermal_energy(self) -> float:
        """Thermal energy ``k_B T`` in eV, the unit by which energies are reduced."""
        return KB_EV * self.temperature

    def normalize(
        self,
        pred: Energy,
        target: Energy,
        **kwargs: Any,
    ) -> tuple[Energy, Energy, ReductionContext]:
        """Check the world batch is one system's configurations, then pass the energies through.

        The guard checks the set the softmax compares, which under data
        parallelism is the gathered world batch: with a strided deal one rank
        can hold replicas of one system and another rank a second system of a
        different size, so no shard alone fails the guard while the world
        batch does. Atom counts alone are compared, so a second system of the
        same size passes, on one rank or across ranks. Every rank reaches the
        collective whenever it reaches the term, and the union is the same on
        every rank, so the ranks either all refuse or all proceed.
        """
        counts = kwargs.get("num_nodes_per_graph")
        if self.check_one_system and counts is not None:
            distinct = sorted(set(counts.tolist()))
            gathered = _gathers_world(self.world_batch)
            if gathered:
                distinct = sorted(set().union(*all_gather_objects(distinct)))
            if len(distinct) > 1:
                scope = "world batch" if gathered else "batch"
                raise ValueError(
                    "BoltzmannMatchingLoss compares the energies of one system's "
                    f"configurations, but the {scope} holds graphs of different "
                    "sizes, whose energies are not comparable at all: got atom "
                    f"counts {distinct!r}. {_ONE_SYSTEM_REMEDY}"
                )
        return pred, target, ReductionContext()

    def mask(
        self,
        pred: Energy,
        target: Energy,
        ctx: ReductionContext,
        **kwargs: Any,
    ) -> _EnergyMask:
        """Return one validity flag per graph, finite on both sides when checked."""
        if self.ignore_nonfinite:
            return torch.isfinite(target) & torch.isfinite(pred)
        return torch.ones_like(target, dtype=torch.bool)

    def compute_residual(
        self,
        pred: Energy,
        target: Energy,
        valid: _EnergyMask,
    ) -> Energy:
        """Return each graph's reduced energy gap, zero where the graph is invalid.

        The zero is selected rather than computed from *pred*, so a non-finite
        student energy cannot leak through it. The selection still keeps the
        result attached to *pred*'s autograd graph, so a batch with no valid
        graph backpropagates a zero update.
        """
        gap = (target - pred) / self.thermal_energy
        return torch.where(valid, gap, torch.zeros_like(gap))

    def reduce(
        self,
        residual: Energy,
        valid: _EnergyMask,
        ctx: ReductionContext,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Combine the world batch's gaps into the beta-interpolated relative entropy.

        Under data parallelism each rank holds a shard of one world batch, and
        a softmax over the shard alone would weight a rank's configurations
        against each other rather than against the whole sample. The gaps
        therefore travel through the differentiable gather of
        :func:`~nvalchemi.training.distributed.all_gather_rows`, so every rank
        computes the same world loss. The gradient that reaches a shard's
        energies is the sum of every rank's copy of it, and the data-parallel
        mean over ranks turns that sum back into the world loss's own
        gradient. ``per_sample_loss`` holds one entry per graph on this rank,
        scaled so that the mean of the entries over the world batch is the
        scalar loss.
        """
        if _gathers_world(self.world_batch):
            gaps, rows = all_gather_rows(residual)
            world_valid, _ = all_gather_rows(valid.to(gaps.dtype), differentiable=False)
            world_valid = world_valid > 0.5
        else:
            gaps, world_valid, rows = residual, valid, slice(None)
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
            f"dtype_policy={self.dtype_policy!r}, "
            f"world_batch={self.world_batch!r}, "
            f"check_one_system={self.check_one_system!r}"
        )
