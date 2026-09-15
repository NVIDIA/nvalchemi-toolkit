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
"""Tests for :mod:`nvalchemi.training.distillation.losses.distribution`."""

from __future__ import annotations

import json
import math

import pytest
import torch

from nvalchemi.dynamics.hooks._utils import KB_EV
from nvalchemi.training._spec import create_model_spec_from_json
from nvalchemi.training.distillation import BoltzmannMatchingLoss
from nvalchemi.training.losses.composition import loss_component_to_spec

_TEMPERATURE = 300.0
"""Ensemble temperature every hand computation here is done at."""

_KT = KB_EV * _TEMPERATURE
"""Thermal energy in eV at :data:`_TEMPERATURE`."""

# Teacher-minus-student energies of two configurations, chosen so the teacher's
# self-normalized weights come out as the exact fractions 3/4 and 1/4.
_TWO_STATE_GAP = _KT * math.log(3.0)
"""Energy difference giving teacher weights of ``3/4`` and ``1/4``."""

_FORWARD_KL = 0.75 * math.log(1.5) + 0.25 * math.log(0.5)
"""``D_KL(p||q)`` for weights ``(3/4, 1/4)`` against a uniform pair."""

_REVERSE_KL = -0.5 * (math.log(1.5) + math.log(0.5))
"""``D_KL(q||p)`` for the same pair."""

# Three unequal weights, because a two-state ensemble cannot see which way round
# the reduced gap is formed: its flipped weights are the correct ones, swapped.
_THREE_STATE_WEIGHTS = (12.0, 4.0, 3.0)
"""Unnormalized teacher weights of an asymmetric three-configuration ensemble."""

_THREE_STATE_SHIFT = 0.75
"""Constant added to every reduced gap, which no relative entropy can see."""

_THREE_STATE_PROBABILITIES = [
    weight / sum(_THREE_STATE_WEIGHTS) for weight in _THREE_STATE_WEIGHTS
]
"""Normalized teacher weights ``12/19``, ``4/19`` and ``3/19``."""

_THREE_STATE_LOG_RATIOS = [
    math.log(len(_THREE_STATE_WEIGHTS) * probability)
    for probability in _THREE_STATE_PROBABILITIES
]
"""``log(B p_i)``: each configuration's log weight against a uniform ensemble."""

_THREE_STATE_FORWARD_KL = sum(
    probability * log_ratio
    for probability, log_ratio in zip(
        _THREE_STATE_PROBABILITIES, _THREE_STATE_LOG_RATIOS, strict=True
    )
)
"""``D_KL(p||q)`` of the asymmetric ensemble against a uniform triple."""

_THREE_STATE_REVERSE_KL = -sum(_THREE_STATE_LOG_RATIOS) / len(_THREE_STATE_LOG_RATIOS)
"""``D_KL(q||p)`` for the same ensemble."""

_SATURATION_BATCH = 8
"""Walker count of the ensemble the forward-direction sweep runs on."""

_SATURATION_SPREADS = (1.0, 4.0, 16.0, 64.0)
"""Student error spreads, in units of ``k_B T``, the forward sweep walks through."""


def _two_state_energies() -> tuple[torch.Tensor, torch.Tensor]:
    """Return student and teacher energies whose weights are ``(3/4, 1/4)``."""
    return torch.zeros(2, 1), torch.tensor([[0.0], [_TWO_STATE_GAP]])


def _three_state_energies(
    student: tuple[float, float, float],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return energies of *student* whose teacher weights are ``(12, 4, 3)/19``.

    The teacher's energies are the student's plus the reduced gaps ``-log w_i``
    the weights imply, so the ensemble is the same whatever *student* is — and
    shifting every gap by :data:`_THREE_STATE_SHIFT` leaves it the same again.
    """
    pred = torch.tensor(student).reshape(-1, 1)
    gaps = torch.tensor([-math.log(weight) for weight in _THREE_STATE_WEIGHTS])
    return pred, pred + _KT * (gaps.reshape(-1, 1) + _THREE_STATE_SHIFT)


def _three_state_gradient(beta: float) -> torch.Tensor:
    """Return the closed-form gradient of the asymmetric ensemble's loss.

    The forward direction contributes ``p_i (log(B p_i) - D_KL(p||q))`` and the
    reverse one ``p_i - 1/B``, both divided by ``k_B T`` because the reduced gap
    carries the student's energy with a minus sign.
    """
    count = len(_THREE_STATE_WEIGHTS)
    return torch.tensor(
        [
            (
                (1.0 - beta) * probability * (log_ratio - _THREE_STATE_FORWARD_KL)
                + beta * (probability - 1.0 / count)
            )
            / _KT
            for probability, log_ratio in zip(
                _THREE_STATE_PROBABILITIES, _THREE_STATE_LOG_RATIOS, strict=True
            )
        ]
    ).reshape(-1, 1)


def _loss_and_gradient_norm(beta: float, target: torch.Tensor) -> tuple[float, float]:
    """Return the loss at uniform student energies and its gradient norm there."""
    pred = torch.zeros_like(target, requires_grad=True)
    loss = BoltzmannMatchingLoss(beta=beta, temperature=_TEMPERATURE)(pred, target)
    loss.backward()
    return loss.item(), float(pred.grad.norm())


class TestBoltzmannMatchingLossValues:
    """Scalar values the beta-interpolated relative entropy produces."""

    @pytest.mark.parametrize(
        ("beta", "expected"),
        [
            pytest.param(0.0, _FORWARD_KL, id="forward"),
            pytest.param(1.0, _REVERSE_KL, id="reverse"),
            pytest.param(0.5, 0.5 * (_FORWARD_KL + _REVERSE_KL), id="symmetric"),
        ],
    )
    def test_value_matches_a_hand_computed_relative_entropy(
        self, beta: float, expected: float
    ) -> None:
        """Both endpoints and their midpoint reproduce the closed-form divergence."""
        pred, target = _two_state_energies()
        loss_fn = BoltzmannMatchingLoss(beta=beta, temperature=_TEMPERATURE)
        assert loss_fn(pred, target).item() == pytest.approx(expected, rel=1e-5)

    @pytest.mark.parametrize(
        ("beta", "expected"),
        [
            pytest.param(0.0, _THREE_STATE_FORWARD_KL, id="forward"),
            pytest.param(1.0, _THREE_STATE_REVERSE_KL, id="reverse"),
            pytest.param(
                0.5,
                0.5 * (_THREE_STATE_FORWARD_KL + _THREE_STATE_REVERSE_KL),
                id="symmetric",
            ),
        ],
    )
    def test_asymmetric_ensemble_matches_a_hand_computed_relative_entropy(
        self, beta: float, expected: float
    ) -> None:
        """Three unequal weights fix the divergence a symmetric pair leaves free.

        The student's energies are unequal too, and every reduced gap carries a
        constant offset, so the value depends on the teacher-minus-student
        difference and on nothing else.
        """
        pred, target = _three_state_energies((0.4, -1.3, 2.1))
        loss_fn = BoltzmannMatchingLoss(beta=beta, temperature=_TEMPERATURE)
        assert loss_fn(pred, target).item() == pytest.approx(expected, rel=1e-4)

    @pytest.mark.parametrize("beta", [0.0, 0.5, 1.0])
    def test_gradient_lowers_the_teachers_most_likely_configuration(
        self, beta: float
    ) -> None:
        """The gradient is largest, and positive, on the teacher's lowest energy.

        At uniform student energies the batch weights are the teacher's own, so
        the gradient reduces to the closed form :func:`_three_state_gradient`
        returns. Its sign on the teacher's most likely configuration is positive
        because descent has to *lower* the student's energy there to give the
        configuration more weight, and negative on the two the student
        over-populates; forming the reduced gap the other way round reverses
        which configurations those are.
        """
        pred, target = _three_state_energies((0.0, 0.0, 0.0))
        pred = pred.requires_grad_(True)

        BoltzmannMatchingLoss(beta=beta, temperature=_TEMPERATURE)(
            pred, target
        ).backward()

        assert int(target.argmin()) == 0
        assert float(pred.grad[0]) > 0.0
        assert bool((pred.grad[1:] < 0.0).all())
        torch.testing.assert_close(
            pred.grad, _three_state_gradient(beta), rtol=1e-4, atol=0.0
        )

    def test_forward_direction_saturates_at_log_batch_size(self) -> None:
        """Past a few kT the forward half stops growing and stops pulling.

        Self-normalized weights make the forward direction a divergence against
        a uniform distribution on ``B`` points, so it cannot exceed ``log B``,
        and once the softmax has collapsed onto one configuration its gradient
        goes to zero while the reverse direction keeps growing — bounded in
        gradient by ``1/kT`` but not in value. This is why the docstring holds
        ``beta`` at ``0.5`` or above until the student is close.
        """
        generator = torch.Generator().manual_seed(0)
        direction = torch.randn(_SATURATION_BATCH, 1, generator=generator)
        forward_values, forward_gradients, reverse_values = [], [], []
        for spread in _SATURATION_SPREADS:
            target = direction * spread * _KT
            forward, forward_gradient = _loss_and_gradient_norm(0.0, target)
            reverse, reverse_gradient = _loss_and_gradient_norm(1.0, target)
            assert forward <= math.log(_SATURATION_BATCH) + 1e-5
            assert reverse_gradient <= 1.0 / _KT
            forward_values.append(forward)
            forward_gradients.append(forward_gradient)
            reverse_values.append(reverse)

        assert forward_values[-1] == pytest.approx(
            math.log(_SATURATION_BATCH), rel=1e-3
        )
        assert forward_gradients[-1] < 1e-3 * forward_gradients[0]
        assert reverse_values == sorted(reverse_values)

    def test_matching_energies_give_zero(self) -> None:
        """A student reproducing the teacher's energies scores zero."""
        loss_fn = BoltzmannMatchingLoss(temperature=_TEMPERATURE)
        energies = torch.tensor([[0.0], [0.1], [-0.2]])
        assert loss_fn(energies.clone(), energies).item() == pytest.approx(0.0)

    def test_constant_energy_offset_is_invisible(self) -> None:
        """Only relative populations matter, so a shifted student still scores zero."""
        loss_fn = BoltzmannMatchingLoss(temperature=_TEMPERATURE)
        target = torch.tensor([[0.0], [0.1], [-0.2]])
        assert loss_fn(target + 5.0, target).item() == pytest.approx(0.0)

    def test_loss_is_non_negative_for_random_energies(self) -> None:
        """Both directions are relative entropies, so neither can go negative."""
        loss_fn = BoltzmannMatchingLoss(beta=0.5, temperature=_TEMPERATURE)
        for seed in range(5):
            generator = torch.Generator().manual_seed(seed)
            pred = torch.randn(6, 1, generator=generator) * _KT
            target = torch.randn(6, 1, generator=generator) * _KT
            assert loss_fn(pred, target).item() >= 0.0

    def test_a_single_configuration_carries_no_signal(self) -> None:
        """One sample is one distribution of one state, which always matches."""
        loss_fn = BoltzmannMatchingLoss(temperature=_TEMPERATURE)
        assert loss_fn(torch.zeros(1, 1), torch.ones(1, 1)).item() == pytest.approx(0.0)

    def test_lower_temperature_sharpens_the_divergence(self) -> None:
        """The same energy gap is a larger population difference when it is colder."""
        pred, target = _two_state_energies()
        cold = BoltzmannMatchingLoss(temperature=_TEMPERATURE / 3.0)
        warm = BoltzmannMatchingLoss(temperature=_TEMPERATURE * 3.0)
        assert cold(pred, target).item() > warm(pred, target).item()

    def test_per_sample_loss_averages_to_the_scalar_loss(self) -> None:
        """The per-graph diagnostic decomposes the value it is published beside."""
        pred, target = _two_state_energies()
        loss_fn = BoltzmannMatchingLoss(beta=0.5, temperature=_TEMPERATURE)
        loss = loss_fn(pred, target)
        assert loss_fn.per_sample_loss.shape == (2,)
        assert loss_fn.per_sample_loss.mean().item() == pytest.approx(loss.item())

    def test_gradients_flow_to_the_student_energies(self) -> None:
        """The student's energies receive gradient, and only their spread matters."""
        target = torch.tensor([[0.0], [_TWO_STATE_GAP], [-_TWO_STATE_GAP]])
        pred = torch.zeros(3, 1, requires_grad=True)
        BoltzmannMatchingLoss(temperature=_TEMPERATURE)(pred, target).backward()
        assert pred.grad is not None
        assert float(pred.grad.sum()) == pytest.approx(0.0, abs=1e-4)


class TestBoltzmannMatchingLossMasking:
    """Which configurations enter the ensemble."""

    def test_nonfinite_target_drops_its_configuration(self) -> None:
        """A masked graph leaves the others normalized among themselves."""
        pred, target = _two_state_energies()
        pred = torch.cat([pred, torch.zeros(1, 1)])
        target = torch.cat([target, torch.full((1, 1), float("nan"))])
        loss_fn = BoltzmannMatchingLoss(beta=0.0, temperature=_TEMPERATURE)
        assert loss_fn(pred, target).item() == pytest.approx(_FORWARD_KL, rel=1e-5)

    def test_fully_masked_ensemble_contributes_zero(self) -> None:
        """No valid configuration is no distribution, which scores zero."""
        loss_fn = BoltzmannMatchingLoss(temperature=_TEMPERATURE)
        target = torch.full((2, 1), float("nan"))
        assert loss_fn(torch.zeros(2, 1), target).item() == pytest.approx(0.0)


class TestBoltzmannMatchingLossContract:
    """Configuration, ensemble, and serialization contract of the loss term."""

    @pytest.mark.parametrize("beta", [-0.1, 1.5])
    def test_beta_outside_the_unit_interval_is_rejected(self, beta: float) -> None:
        """Beta interpolates two divergences and cannot extrapolate past them."""
        with pytest.raises(ValueError, match=r"must lie in \[0, 1\]"):
            BoltzmannMatchingLoss(beta=beta)

    def test_nonpositive_temperature_is_rejected(self) -> None:
        """A temperature that defines no ensemble fails at construction."""
        with pytest.raises(ValueError, match="must be positive Kelvin"):
            BoltzmannMatchingLoss(temperature=0.0)

    def test_mixed_system_sizes_are_rejected(self) -> None:
        """Energies of different systems are not one Boltzmann distribution."""
        loss_fn = BoltzmannMatchingLoss(temperature=_TEMPERATURE)
        with pytest.raises(ValueError, match="graphs of different sizes"):
            loss_fn(
                torch.zeros(2, 1),
                torch.zeros(2, 1),
                num_nodes_per_graph=torch.tensor([3, 4]),
            )

    def test_uniform_system_sizes_are_accepted(self) -> None:
        """One system's replicas are exactly the ensemble the term is defined on."""
        loss_fn = BoltzmannMatchingLoss(temperature=_TEMPERATURE)
        loss = loss_fn(
            torch.zeros(2, 1),
            torch.zeros(2, 1),
            num_nodes_per_graph=torch.tensor([4, 4]),
        )
        assert loss.item() == pytest.approx(0.0)

    def test_absent_size_metadata_skips_the_ensemble_check(self) -> None:
        """The guard reads metadata a direct call never supplies, so it cannot fire."""
        loss_fn = BoltzmannMatchingLoss(temperature=_TEMPERATURE)

        loss = loss_fn(torch.zeros(3, 1), torch.tensor([[0.0], [1.0], [2.0]]))

        assert math.isfinite(loss.item())

    def test_equal_atom_counts_do_not_prove_one_system(self) -> None:
        """Sizes are necessary but not sufficient, which is the guard's known reach."""
        loss_fn = BoltzmannMatchingLoss(temperature=_TEMPERATURE)

        loss = loss_fn(
            torch.zeros(2, 1),
            torch.tensor([[0.0], [30.0]]),
            num_nodes_per_graph=torch.tensor([3, 3]),
        )

        assert math.isfinite(loss.item())

    def test_loss_declares_it_needs_no_evaluation_gradients(self) -> None:
        """Total energies are a direct output, so validation can run no-grad."""
        assert BoltzmannMatchingLoss().requires_eval_grad is False

    def test_default_keys_read_the_teacher_and_student_energies(self) -> None:
        """The term needs no target of its own beyond the teacher's energies."""
        loss_fn = BoltzmannMatchingLoss()
        assert loss_fn.target_key == "teacher_energy"
        assert loss_fn.prediction_key == "predicted_energy"

    def test_reduced_energy_scale_is_the_thermal_energy(self) -> None:
        """Energies are reduced by ``k_B T`` in the toolkit's eV convention."""
        loss_fn = BoltzmannMatchingLoss(temperature=_TEMPERATURE)
        assert loss_fn.reduced_energy_scale == pytest.approx(_KT)

    def test_spec_round_trip_rebuilds_an_equivalent_loss(self) -> None:
        """A JSON round-tripped spec rebuilds the loss with its configuration."""
        spec = loss_component_to_spec(
            BoltzmannMatchingLoss(beta=0.25, temperature=500.0)
        )
        rebuilt = create_model_spec_from_json(
            json.loads(spec.model_dump_json())
        ).build()
        assert isinstance(rebuilt, BoltzmannMatchingLoss)
        assert rebuilt.beta == pytest.approx(0.25)
        assert rebuilt.temperature == pytest.approx(500.0)
