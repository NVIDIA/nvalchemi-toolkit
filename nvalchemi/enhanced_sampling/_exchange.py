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
"""Synchronous replica exchange over a batch of walkers.

Exchange swaps **thermodynamic-state labels, not atomic coordinates**.  A
walker keeps its execution slot, its history, and its integrator arrays; what
changes is the temperature or bias window assigned to it.  That keeps the
move local — no coordinate traffic, no reallocation — which is what makes it
viable inside one batched GPU step.

Every walker holds exactly one state and every state exactly one walker, so
an exchange is a permutation of :attr:`Batch.thermodynamic_state_id`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import torch
from pydantic import BaseModel, ConfigDict, Field

from nvalchemi.dynamics.hooks._utils import KB_EV

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

__all__ = ["ReplicaExchange", "ThermodynamicState"]


class ThermodynamicState(BaseModel):
    """One set of conditions a walker can be assigned to.

    Attributes
    ----------
    state_id:
        Index into the ladder.  Must be dense and start at zero across the
        set of states, because pairing walks neighbouring indices.
    temperature:
        Temperature in Kelvin.  Equal across all states means the ladder
        varies by bias window instead, which selects umbrella acceptance.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    state_id: int = Field(ge=0)
    temperature: float = Field(gt=0.0)


class ReplicaExchange:
    r"""Synchronous replica exchange with an even/odd pair schedule.

    Parameters
    ----------
    states:
        The ladder.  ``state_id`` values must be exactly ``0..S-1``.
    initial_state_ids:
        Assignment of states to walkers, shape ``[B]``.  Must be a
        permutation of ``0..S-1``: replica exchange presumes a bijection
        between walkers and states, and a duplicate would let two walkers
        claim the same rung.
    mode:
        ``"synchronous"`` only.  Asynchronous exchange is not implemented.
    attempt_interval:
        Dynamics steps per exchange segment.
    random_seed:
        Base seed for acceptance draws.  Randomness is derived per attempt as
        ``random_seed + exchange_id`` rather than from a long-lived generator,
        so a checkpoint needs two integers instead of an opaque RNG blob —
        the same counter-based scheme ``NVTLangevin`` uses for its noise.

    Raises
    ------
    ValueError
        If the mode is unsupported, the ladder is not dense, the assignment
        is not a permutation, or the acceptance rule cannot be inferred.

    Notes
    -----
    Acceptance
        Which rule applies is inferred from the ladder and validated, rather
        than being a free parameter that can silently disagree with it.

        *Temperature exchange* — temperatures differ:

        .. math::

            \log a = \min\bigl(0,\ (\beta_i - \beta_j)(U_i - U_j)\bigr)

        A cold replica holding anomalously high energy therefore moves up the
        ladder with probability one, which is the point of the method.

        *Umbrella exchange* — temperatures are equal and the states differ by
        bias window:

        .. math::

            \log a = \min\bigl(0,\ u_i(x_i) + u_j(x_j)
                                 - u_i(x_j) - u_j(x_i)\bigr)

        with :math:`u_k` the reduced bias potential of state *k*.  The two
        cross terms need the bias re-evaluated under swapped labels, which
        costs one extra bias evaluation per attempt.

    Not supported
        A ladder that varies temperature *and* bias window at once needs a
        combined acceptance rule that is not implemented.  It is also not
        detectable here — a :class:`ThermodynamicState` carries only a
        temperature, and which windows a bias exposes is the bias's business
        — so this is a documented constraint on the caller, not a guard.
        Vary one or the other.
    """

    def __init__(
        self,
        states: Sequence[ThermodynamicState],
        initial_state_ids: torch.Tensor,
        *,
        mode: Literal["synchronous"] = "synchronous",
        attempt_interval: int = 100,
        random_seed: int = 1234,
    ) -> None:
        if mode != "synchronous":
            raise ValueError(
                f"ReplicaExchange: mode={mode!r} is not supported. Only "
                "'synchronous' is implemented; asynchronous exchange "
                "(pair-local rendezvous, non-blocking workers) is future work."
            )
        if len(states) < 2:
            raise ValueError(
                f"ReplicaExchange: need at least 2 states to exchange, got "
                f"{len(states)}."
            )
        ladder = sorted(states, key=lambda s: s.state_id)
        if [s.state_id for s in ladder] != list(range(len(ladder))):
            raise ValueError(
                f"ReplicaExchange: state_id values must be exactly 0..{len(ladder) - 1}, "
                f"got {sorted(s.state_id for s in states)}. Pairing walks "
                "neighbouring indices, so a sparse ladder has no defined "
                "neighbours."
            )
        self.states = tuple(ladder)
        self.mode = mode
        self.attempt_interval = int(attempt_interval)
        self.random_seed = int(random_seed)

        ids = initial_state_ids.reshape(-1).to(torch.long)
        if sorted(ids.tolist()) != list(range(len(self.states))):
            raise ValueError(
                f"ReplicaExchange: initial_state_ids must be a permutation of "
                f"0..{len(self.states) - 1}, got {ids.tolist()}. Replica "
                "exchange presumes one walker per state; a duplicate would let "
                "two walkers claim the same rung of the ladder."
            )
        self.initial_state_ids = ids

        self._acceptance = self._infer_acceptance()
        self.exchange_id = 0
        self.attempts = 0
        self.accepted = 0
        # Per neighbouring-state-pair tallies, for the acceptance-rate
        # diagnostics a REMD run is tuned on.
        self.pair_attempts = [0] * (len(self.states) - 1)
        self.pair_accepted = [0] * (len(self.states) - 1)

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def _infer_acceptance(self) -> Literal["temperature", "umbrella"]:
        """Return which acceptance rule this ladder implies.

        A :class:`ThermodynamicState` carries only a temperature, so the
        rule is read from the ladder: varying temperatures mean temperature
        exchange, equal ones mean the states can only differ by which bias
        window they select.

        Inferring rather than accepting a parameter is deliberate — a
        mismatch between a declared rule and the ladder it runs on would be
        silent, and wrong acceptance breaks detailed balance without any
        symptom a run would show.

        Returns
        -------
        Literal["temperature", "umbrella"]
            The applicable rule.
        """
        temperatures = [s.temperature for s in self.states]
        varies_temperature = max(temperatures) - min(temperatures) > 1e-12
        return "temperature" if varies_temperature else "umbrella"

    @property
    def acceptance(self) -> str:
        """Return the inferred acceptance rule name."""
        return self._acceptance

    @property
    def temperatures(self) -> torch.Tensor:
        """Return the ladder temperatures in Kelvin, shape ``[S]``."""
        return torch.tensor([s.temperature for s in self.states])

    def validate_for(self, biases: Mapping[str, Any]) -> None:
        """Reject bias/ladder combinations whose acceptance is undefined.

        Parameters
        ----------
        biases:
            The runner's bias mapping.

        Raises
        ------
        ValueError
            If umbrella exchange is configured with no bias to exchange over,
            or if any bias cannot supply the energy the acceptance rule needs.
        """
        if self._acceptance == "umbrella" and not biases:
            raise ValueError(
                "ReplicaExchange: every state has the same temperature, so the "
                "ladder can only differ by bias window — but no biases were "
                "registered. Either vary the temperatures for temperature "
                "exchange, or register the bias whose windows the states select."
            )
        for name, bias in biases.items():
            energy_less = getattr(bias, "supplies_exchange_energy", None)
            if energy_less is False:
                raise ValueError(
                    f"ReplicaExchange: bias {name!r} declares that it supplies "
                    "no exchange energy (a force-only bias such as adaptive "
                    "biasing force). The acceptance rule needs a cross-state "
                    "bias energy, so such a bias cannot participate; run it "
                    "without replica exchange."
                )

    # ------------------------------------------------------------------
    # Scheduling
    # ------------------------------------------------------------------

    def pair_schedule(self, segment: int) -> list[tuple[int, int]]:
        """Return the neighbouring state pairs attempted in *segment*.

        Alternating even/odd offsets means every rung of the ladder is
        exchangeable with both neighbours over two segments, while no state
        appears in two pairs of the same segment — which is what lets all
        pairs be decided simultaneously.

        Parameters
        ----------
        segment:
            Exchange segment index.

        Returns
        -------
        list[tuple[int, int]]
            Neighbouring ``(state_id, state_id + 1)`` pairs.
        """
        offset = segment % 2
        return [(index, index + 1) for index in range(offset, len(self.states) - 1, 2)]

    def _uniforms(self, count: int, device: torch.device) -> torch.Tensor:
        """Draw acceptance uniforms for one attempt, reproducibly.

        Parameters
        ----------
        count:
            Number of draws.
        device:
            Device for the result.

        Returns
        -------
        torch.Tensor
            Shape ``[count]`` in ``[0, 1)``.

        Notes
        -----
        Drawn on the CPU from a generator seeded with
        ``random_seed + exchange_id``, then moved.  Seeding per attempt makes
        the sequence a pure function of two checkpointed integers, and drawing
        on the CPU keeps it independent of the device the run happens to use —
        so a restored run reproduces the same accept/reject decisions.
        """
        generator = torch.Generator()
        generator.manual_seed(self.random_seed + self.exchange_id)
        return torch.rand(count, generator=generator).to(device)

    # ------------------------------------------------------------------
    # Acceptance
    # ------------------------------------------------------------------

    def _log_acceptance_temperature(
        self,
        pairs: list[tuple[int, int]],
        walker_of_state: dict[int, int],
        energies: torch.Tensor,
    ) -> torch.Tensor:
        """Return ``log a`` per pair for temperature exchange.

        Parameters
        ----------
        pairs:
            Neighbouring state pairs.
        walker_of_state:
            State id to the walker row currently holding it.
        energies:
            Per-walker potential energy ``U``, shape ``[B]``.

        Returns
        -------
        torch.Tensor
            ``log a`` per pair, shape ``[len(pairs)]``, capped at zero.
        """
        temperatures = self.temperatures.to(energies.device, energies.dtype)
        beta = 1.0 / (KB_EV * temperatures)  # [S]

        values = []
        for state_i, state_j in pairs:
            walker_i = walker_of_state[state_i]
            walker_j = walker_of_state[state_j]
            delta = (beta[state_i] - beta[state_j]) * (
                energies[walker_i] - energies[walker_j]
            )
            values.append(delta)
        return torch.clamp(torch.stack(values), max=0.0)

    def _log_acceptance_umbrella(
        self,
        pairs: list[tuple[int, int]],
        walker_of_state: dict[int, int],
        bias_current: torch.Tensor,
        bias_swapped: torch.Tensor,
    ) -> torch.Tensor:
        """Return ``log a`` per pair for umbrella exchange.

        Parameters
        ----------
        pairs:
            Neighbouring state pairs.
        walker_of_state:
            State id to the walker row currently holding it.
        bias_current:
            Reduced bias potential per walker under its current state,
            shape ``[B]``.
        bias_swapped:
            Reduced bias potential per walker under the proposed state,
            shape ``[B]``.

        Returns
        -------
        torch.Tensor
            ``log a`` per pair, shape ``[len(pairs)]``, capped at zero.
        """
        values = []
        for state_i, state_j in pairs:
            walker_i = walker_of_state[state_i]
            walker_j = walker_of_state[state_j]
            before = bias_current[walker_i] + bias_current[walker_j]
            after = bias_swapped[walker_i] + bias_swapped[walker_j]
            values.append(before - after)
        return torch.clamp(torch.stack(values), max=0.0)

    def decide(
        self,
        segment: int,
        state_ids: torch.Tensor,
        energies: torch.Tensor,
        bias_current: torch.Tensor | None = None,
        bias_swapped: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, list[tuple[int, int]], torch.Tensor]:
        """Decide this segment's swaps and return the new assignment.

        Pure with respect to the batch: it takes energies and returns a
        permutation, so the caller owns every side effect (rebinding the
        integrator, rescaling velocities, re-priming forces).  That keeps the
        acceptance rule testable against hand-computed numbers.

        Parameters
        ----------
        segment:
            Exchange segment index, which selects the even/odd pairing.
        state_ids:
            Current assignment, shape ``[B]``.
        energies:
            Per-walker potential energy ``U`` in eV, shape ``[B]``.  Used by
            temperature acceptance.
        bias_current:
            Reduced bias potential per walker under its current state, shape
            ``[B]``.  Required for umbrella acceptance.
        bias_swapped:
            Reduced bias potential per walker under its proposed state, shape
            ``[B]``.  Required for umbrella acceptance.

        Returns
        -------
        tuple[torch.Tensor, list[tuple[int, int]], torch.Tensor]
            The new state assignment ``[B]``, the pairs attempted, and the
            boolean accept mask over those pairs.

        Raises
        ------
        ValueError
            If umbrella acceptance is in force but the bias energies were not
            supplied.
        """
        pairs = self.pair_schedule(segment)
        ids = state_ids.reshape(-1).to(torch.long)
        walker_of_state = {int(state): row for row, state in enumerate(ids.tolist())}

        if not pairs:
            empty = torch.zeros(0, dtype=torch.bool, device=ids.device)
            return ids.clone(), pairs, empty

        if self._acceptance == "temperature":
            log_alpha = self._log_acceptance_temperature(
                pairs, walker_of_state, energies.reshape(-1)
            )
        else:
            if bias_current is None or bias_swapped is None:
                raise ValueError(
                    "ReplicaExchange: umbrella acceptance needs the bias "
                    "energy under both the current and the proposed state "
                    "assignment, but one was not supplied."
                )
            log_alpha = self._log_acceptance_umbrella(
                pairs,
                walker_of_state,
                bias_current.reshape(-1),
                bias_swapped.reshape(-1),
            )

        uniforms = self._uniforms(len(pairs), log_alpha.device)
        accepted = log_acceptance_is_accepted(log_alpha, uniforms)

        new_ids = ids.clone()
        for index, ((state_i, state_j), take) in enumerate(
            zip(pairs, accepted.tolist(), strict=True)
        ):
            self.attempts += 1
            self.pair_attempts[state_i] += 1
            if take:
                walker_i = walker_of_state[state_i]
                walker_j = walker_of_state[state_j]
                new_ids[walker_i] = state_j
                new_ids[walker_j] = state_i
                self.accepted += 1
                self.pair_accepted[state_i] += 1
            del index
        self.exchange_id += 1
        return new_ids, pairs, accepted

    def proposed_assignment(
        self, segment: int, state_ids: torch.Tensor
    ) -> torch.Tensor:
        """Return the assignment that would result if every pair swapped.

        Umbrella acceptance needs the bias evaluated under the proposed
        labels *before* the decision is made, so the caller needs the
        proposal separately from the outcome.

        Parameters
        ----------
        segment:
            Exchange segment index.
        state_ids:
            Current assignment, shape ``[B]``.

        Returns
        -------
        torch.Tensor
            The all-swaps-accepted assignment, shape ``[B]``.
        """
        ids = state_ids.reshape(-1).to(torch.long)
        walker_of_state = {int(state): row for row, state in enumerate(ids.tolist())}
        proposed = ids.clone()
        for state_i, state_j in self.pair_schedule(segment):
            walker_i = walker_of_state[state_i]
            walker_j = walker_of_state[state_j]
            proposed[walker_i] = state_j
            proposed[walker_j] = state_i
        return proposed

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    def state_dict(self) -> dict[str, Any]:
        """Return exchange state for checkpointing.

        Returns
        -------
        dict[str, Any]
            Counters and the acceptance-RNG position.  The position is two
            integers rather than a generator blob; see :meth:`_uniforms`.
        """
        return {
            "exchange_id": int(self.exchange_id),
            "attempts": int(self.attempts),
            "accepted": int(self.accepted),
            "random_seed": int(self.random_seed),
            "attempt_interval": int(self.attempt_interval),
            "pair_attempts": list(self.pair_attempts),
            "pair_accepted": list(self.pair_accepted),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore exchange state produced by :meth:`state_dict`.

        Parameters
        ----------
        state:
            The mapping previously returned by :meth:`state_dict`.
        """
        self.exchange_id = int(state.get("exchange_id", 0))
        self.attempts = int(state.get("attempts", 0))
        self.accepted = int(state.get("accepted", 0))
        self.random_seed = int(state.get("random_seed", self.random_seed))
        self.attempt_interval = int(
            state.get("attempt_interval", self.attempt_interval)
        )
        pair_attempts = state.get("pair_attempts")
        if pair_attempts is not None:
            self.pair_attempts = [int(v) for v in pair_attempts]
        pair_accepted = state.get("pair_accepted")
        if pair_accepted is not None:
            self.pair_accepted = [int(v) for v in pair_accepted]

    @property
    def acceptance_rate(self) -> float:
        """Return the fraction of attempted pair swaps that were accepted."""
        return self.accepted / self.attempts if self.attempts else 0.0

    def pair_acceptance_rates(self) -> list[float]:
        """Return the per-neighbouring-pair acceptance rates.

        A REMD ladder is tuned on these: a pair far below the others is a gap
        the walkers cannot cross, and the ladder needs another rung there.

        Returns
        -------
        list[float]
            One rate per neighbouring pair, ``0.0`` where never attempted.
        """
        return [
            (accepted / attempts if attempts else 0.0)
            for accepted, attempts in zip(
                self.pair_accepted, self.pair_attempts, strict=True
            )
        ]

    def __repr__(self) -> str:
        """Return a concise description of the exchange."""
        return (
            f"{type(self).__name__}(states={len(self.states)}, "
            f"acceptance={self._acceptance!r}, "
            f"attempt_interval={self.attempt_interval}, "
            f"accepted={self.accepted}/{self.attempts})"
        )


def log_acceptance_is_accepted(
    log_alpha: torch.Tensor, uniforms: torch.Tensor
) -> torch.Tensor:
    """Return the accept mask for the given log-acceptance values.

    ``log_alpha`` is capped at zero, so ``log_alpha == 0`` means accept with
    probability one.  Comparing ``log(u) < log_alpha`` rather than
    ``u < exp(log_alpha)`` keeps a very negative ``log_alpha`` from
    underflowing to exactly zero and turning a rare-but-possible swap into an
    impossible one.

    Parameters
    ----------
    log_alpha : torch.Tensor
        Log acceptance probability per pair, ``<= 0``.
    uniforms : torch.Tensor
        Draws in ``[0, 1)``, same shape.

    Returns
    -------
    torch.Tensor
        Boolean accept mask.
    """
    safe = torch.clamp(uniforms, min=torch.finfo(uniforms.dtype).tiny)
    return torch.log(safe) < log_alpha
