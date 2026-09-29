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

import math
from typing import TYPE_CHECKING, Any, Literal

import torch
from pydantic import BaseModel, ConfigDict, Field

from nvalchemi.dynamics.hooks._utils import KB_EV
from nvalchemi.dynamics.hooks.swap import (
    PairSwapHook,
    apply_pair_swaps,
    even_odd_pairs,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from nvalchemi.data import Batch

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
        If the mode is unsupported, ``attempt_interval`` is below 1, the
        ladder is not dense, or the assignment is not a permutation.

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
        combined acceptance rule that is not implemented.  The temperature
        rule alone omits the cross-state bias terms, so running it anyway
        would break detailed balance with no symptom — it is therefore
        **rejected**, twice over: a bias that declares
        ``state_dependent_for_exchange`` is refused at construction, and the
        runner additionally probes empirically at prime time by evaluating
        every bias under a permuted assignment.  Vary one or the other.
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
        if int(attempt_interval) < 1:
            raise ValueError(
                f"ReplicaExchange: attempt_interval must be at least 1, got "
                f"{attempt_interval}. A non-positive interval has no meaning "
                "as a segment length — it would be clamped to every-step "
                "exchange while the checkpoint recorded the value you passed, "
                "so the run and its metadata would disagree."
            )
        self.states = tuple(ladder)
        self.mode = mode
        self.attempt_interval = int(attempt_interval)
        self.random_seed = int(random_seed)

        self._acceptance = self._infer_acceptance()
        self.initial_state_ids = self.validate_assignment(
            initial_state_ids, source="initial_state_ids"
        )
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

    def validate_assignment(
        self,
        state_ids: torch.Tensor,
        num_graphs: int | None = None,
        *,
        source: str = "assignment",
    ) -> torch.Tensor:
        """Return *state_ids* as a validated ``[B]`` long tensor.

        The bijection between walkers and states is the assumption every
        other piece rests on: pairing looks up "which walker holds state k",
        and a duplicate or a wrong-length assignment makes that lookup
        meaningless.  Without this the failures surface far from their cause
        — a length mismatch as ``ValueError: Length mismatch: 4 vs 2`` from
        inside the batch storage, a duplicate as ``KeyError: 3`` from the
        pair lookup — neither of which names the ladder or the batch.

        Used for both the constructor argument and whatever assignment the
        batch actually carries, so the rule lives in one place.

        Parameters
        ----------
        state_ids : torch.Tensor
            Candidate assignment, any shape reshapeable to ``[B]``.
        num_graphs : int | None
            Walker count to check the ladder against, when known.
        source : str
            Name of the thing being validated, used in the error.

        Returns
        -------
        torch.Tensor
            The assignment as a 1-D long tensor.

        Raises
        ------
        ValueError
            If the walker count disagrees with the ladder, or the assignment
            is not a permutation of ``0..S-1``.
        """
        n_states = len(self.states)
        if num_graphs is not None and num_graphs != n_states:
            raise ValueError(
                f"ReplicaExchange: the ladder has {n_states} state(s) but the "
                f"batch has {num_graphs} walker(s). Replica exchange presumes "
                "one walker per state — pairing looks up which walker holds "
                "each rung, which has no answer when the counts differ. Build "
                "the batch with one graph per ThermodynamicState."
            )

        ids = state_ids.reshape(-1).to(torch.long)
        if ids.numel() != n_states:
            raise ValueError(
                f"ReplicaExchange: {source} has {ids.numel()} entr(ies) but the "
                f"ladder has {n_states} state(s); they must agree."
            )
        if sorted(ids.tolist()) != list(range(n_states)):
            raise ValueError(
                f"ReplicaExchange: {source} must be a permutation of "
                f"0..{n_states - 1}, got {ids.tolist()}. Replica exchange "
                "presumes one walker per state; a duplicate would let two "
                "walkers claim the same rung of the ladder, and leave another "
                "rung held by none."
            )
        return ids

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
            if any bias cannot supply the energy the acceptance rule needs, or
            if a state-dependent bias is combined with a temperature ladder.
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
            if self._acceptance == "temperature" and (
                getattr(bias, "state_dependent_for_exchange", False) is True
            ):
                raise ValueError(
                    f"ReplicaExchange: the ladder varies temperature, which "
                    f"selects temperature acceptance, but bias {name!r} has "
                    "per-state parameters. The combined temperature-plus-window "
                    "acceptance rule is not implemented, and the temperature "
                    "rule alone omits the cross-state bias terms — so detailed "
                    "balance would be wrong with nothing to show for it. Use a "
                    "single-window bias with a temperature ladder, or equal "
                    "temperatures with a multi-window bias."
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
        return even_odd_pairs(segment, len(self.states))

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
            If *state_ids* is not a permutation of the ladder, or if umbrella
            acceptance is in force but the bias energies were not supplied.
        """
        # Validate before touching any counter. Pairing looks up "which walker
        # holds state k", which a duplicate or short assignment answers with a
        # bare KeyError — and by then attempts/pair_attempts have already been
        # incremented, leaving the tallies corrupted by a call that failed.
        ids = self.validate_assignment(state_ids, source="state_ids")
        pairs = self.pair_schedule(segment)
        walker_of_state = {int(state): row for row, state in enumerate(ids.tolist())}

        if not pairs:
            empty = torch.zeros(0, dtype=torch.bool, device=ids.device)
            return ids.clone(), pairs, empty

        rows_i = torch.tensor(
            [walker_of_state[i] for i, _ in pairs], device=ids.device, dtype=torch.long
        )
        rows_j = torch.tensor(
            [walker_of_state[j] for _, j in pairs], device=ids.device, dtype=torch.long
        )
        accepted = self._decide_rows(
            ids,
            rows_i,
            rows_j,
            energies=energies,
            bias_current=bias_current,
            bias_swapped=bias_swapped,
        )
        new_ids = apply_pair_swaps(ids, pairs, accepted, walker_of_state)
        return new_ids, pairs, accepted

    def _decide_rows(
        self,
        slots: torch.Tensor,
        rows_i: torch.Tensor,
        rows_j: torch.Tensor,
        *,
        energies: torch.Tensor | None = None,
        bias_current: torch.Tensor | None = None,
        bias_swapped: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply the acceptance rule to one round of proposed pairs.

        The single implementation of the rule.  Both callers reach it — the
        :class:`~nvalchemi.dynamics.hooks.PairSwapHook` that runs a live
        ladder, and :meth:`decide`, which exposes the same decision without a
        batch so the formula can be checked against hand-computed numbers.
        Two implementations would drift, and a drifted acceptance rule breaks
        detailed balance with nothing to show for it.

        Parameters
        ----------
        slots:
            Current assignment, shape ``[B]``.
        rows_i, rows_j:
            Graph rows holding each proposed pair's two slots, shape ``[P]``.
        energies:
            Per-walker potential energy ``U``.  Temperature acceptance only.
        bias_current, bias_swapped:
            Reduced bias potential under the current and proposed assignment.
            Umbrella acceptance only.

        Returns
        -------
        torch.Tensor
            One boolean per pair.

        Raises
        ------
        ValueError
            If the inputs the rule in force needs were not supplied.
        """
        if self._acceptance == "temperature":
            if energies is None:
                raise ValueError(
                    "ReplicaExchange: temperature acceptance needs the "
                    "per-walker potential energy, but none was supplied."
                )
            log_alpha = self._log_acceptance_temperature_rows(
                slots, rows_i, rows_j, energies.reshape(-1)
            )
        else:
            if bias_current is None or bias_swapped is None:
                raise ValueError(
                    "ReplicaExchange: umbrella acceptance needs the bias "
                    "energy under both the current and the proposed state "
                    "assignment, but one was not supplied."
                )
            log_alpha = self._log_acceptance_umbrella_rows(
                rows_i, rows_j, bias_current.reshape(-1), bias_swapped.reshape(-1)
            )

        uniforms = self._uniforms(rows_i.numel(), log_alpha.device)
        accepted = log_acceptance_is_accepted(log_alpha, uniforms)
        self._tally(
            [
                (int(a), int(b))
                for a, b in zip(
                    slots[rows_i].tolist(), slots[rows_j].tolist(), strict=True
                )
            ],
            accepted,
        )
        return accepted

    def _tally(self, pairs: list[tuple[int, int]], accepted: torch.Tensor) -> None:
        """Record one round of attempts, and advance the acceptance RNG.

        Parameters
        ----------
        pairs:
            The pairs that were decided.
        accepted:
            One boolean per pair.
        """
        for (state_i, _state_j), take in zip(pairs, accepted.tolist(), strict=True):
            self.attempts += 1
            self.pair_attempts[state_i] += 1
            if take:
                self.accepted += 1
                self.pair_accepted[state_i] += 1
        self.exchange_id += 1

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

        Raises
        ------
        ValueError
            If *state_ids* is not a permutation of the ladder.
        """
        ids = self.validate_assignment(state_ids, source="state_ids")
        walker_of_state = {int(state): row for row, state in enumerate(ids.tolist())}
        pairs = self.pair_schedule(segment)
        every = torch.ones(len(pairs), dtype=torch.bool, device=ids.device)
        return apply_pair_swaps(ids, pairs, every, walker_of_state)

    # ------------------------------------------------------------------
    # The PairSwapHook surface
    # ------------------------------------------------------------------

    def swap_hook(
        self,
        *,
        bias_energy_fn: Callable[[Batch, torch.Tensor], torch.Tensor] | None = None,
        on_swap: Callable[[Batch], None] | None = None,
        slot_field: str = "thermodynamic_state_id",
    ) -> PairSwapHook:
        """Return the generic swap hook that runs this ladder.

        Everything mechanical — the pair schedule, the permutation, the
        cadence, the parameter rebinding — belongs to
        :class:`~nvalchemi.dynamics.hooks.PairSwapHook`.  This supplies the
        two pieces that are enhanced-sampling physics: the Sugita-Okamoto
        acceptance rule and the temperature table it rebinds from.

        Parameters
        ----------
        bias_energy_fn:
            ``(batch, state_ids) -> Tensor[B]`` giving the reduced bias
            potential under an assignment.  Required for umbrella acceptance,
            which evaluates the bias under both the current and the proposed
            labels; unused by temperature acceptance.
        on_swap:
            Called with the batch after an accepted swap.  Forces computed
            under the previous labels are what this is for.
        slot_field:
            Per-graph batch field holding the assignment.

        Returns
        -------
        PairSwapHook
            Configured for this ladder's interval and rung count.
        """
        return PairSwapHook(
            self._make_accept_fn(bias_energy_fn, slot_field),
            self.per_system_params,
            slot_field=slot_field,
            n_slots=len(self.states),
            frequency=self.attempt_interval,
            on_swap=on_swap,
        )

    def per_system_params(self, state_ids: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return the integrator parameters implied by an assignment.

        The ``params_fn`` half of the swap: a state id is an index into the
        ladder, and what the integrator needs is the temperature it points at.

        Parameters
        ----------
        state_ids:
            Assignment, shape ``[B]``.

        Returns
        -------
        dict[str, torch.Tensor]
            ``{"temperature": T}`` in Kelvin per graph, shape ``[B]``.
        """
        table = self.temperatures.to(state_ids.device)
        return {"temperature": table[state_ids.reshape(-1).to(torch.long)]}

    def _make_accept_fn(
        self,
        bias_energy_fn: Callable[[Batch, torch.Tensor], torch.Tensor] | None,
        slot_field: str,
    ) -> Callable[[Batch, torch.Tensor, torch.Tensor], torch.Tensor]:
        """Return the ``accept_fn`` a :class:`PairSwapHook` calls.

        The rule is read off the pairs rather than off a segment index, which
        is what lets it match the generic ``accept_fn(batch, i, j)``
        signature: the proposal is "swap the slots these rows hold", and both
        acceptance formulas need only that.

        Parameters
        ----------
        bias_energy_fn:
            Reduced bias potential under a given assignment, or ``None``.
        slot_field:
            Per-graph batch field holding the assignment.

        Returns
        -------
        Callable
            ``(batch, rows_i, rows_j) -> Bool[Tensor, "P"]``.
        """

        def accept(
            batch: Batch, rows_i: torch.Tensor, rows_j: torch.Tensor
        ) -> torch.Tensor:
            slots = batch[slot_field].reshape(-1).to(torch.long)
            if self._acceptance == "temperature":
                energies = getattr(batch, "energy", None)
                if energies is None:
                    energies = torch.zeros(
                        batch.num_graphs, device=batch.positions.device
                    )
                return self._decide_rows(slots, rows_i, rows_j, energies=energies)

            if bias_energy_fn is None:
                raise ValueError(
                    "ReplicaExchange: umbrella acceptance needs the bias "
                    "energy under both the current and the proposed state "
                    "assignment, but no bias_energy_fn was supplied."
                )
            proposed = slots.clone()
            proposed[rows_i] = slots[rows_j]
            proposed[rows_j] = slots[rows_i]
            return self._decide_rows(
                slots,
                rows_i,
                rows_j,
                bias_current=self._reduced_bias_energy(batch, slots, bias_energy_fn),
                bias_swapped=self._reduced_bias_energy(batch, proposed, bias_energy_fn),
            )

        return accept

    def _reduced_bias_energy(
        self,
        batch: Batch,
        state_ids: torch.Tensor,
        bias_energy_fn: Callable[[Batch, torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        """Return ``beta * E_bias`` per walker under *state_ids*.

        The reduction is the ladder's, not the bias's: ``beta`` comes from the
        temperature a state id points at, and the bias knows nothing about
        temperatures.

        Parameters
        ----------
        batch:
            The live batch.
        state_ids:
            Assignment to evaluate under, shape ``[B]``.
        bias_energy_fn:
            Total bias energy per walker under a given assignment.

        Returns
        -------
        torch.Tensor
            ``beta * E_bias`` per walker, shape ``[B]``.
        """
        total = bias_energy_fn(batch, state_ids).reshape(-1)
        temperatures = self.temperatures.to(total.device, total.dtype)
        beta = 1.0 / (KB_EV * temperatures[state_ids.reshape(-1).to(torch.long)])
        return beta * total

    def _log_acceptance_temperature_rows(
        self,
        slots: torch.Tensor,
        rows_i: torch.Tensor,
        rows_j: torch.Tensor,
        energies: torch.Tensor,
    ) -> torch.Tensor:
        """Return ``log a`` per pair for temperature exchange, by row.

        Parameters
        ----------
        slots:
            Current assignment, shape ``[B]``.
        rows_i, rows_j:
            Graph rows holding each pair's two slots, shape ``[P]``.
        energies:
            Per-walker potential energy ``U``, shape ``[B]``.

        Returns
        -------
        torch.Tensor
            ``log a`` per pair, capped at zero.
        """
        beta = 1.0 / (KB_EV * self.temperatures.to(energies.device, energies.dtype))
        delta = (beta[slots[rows_i]] - beta[slots[rows_j]]) * (
            energies[rows_i] - energies[rows_j]
        )
        return torch.clamp(delta, max=0.0)

    @staticmethod
    def _log_acceptance_umbrella_rows(
        rows_i: torch.Tensor,
        rows_j: torch.Tensor,
        bias_current: torch.Tensor,
        bias_swapped: torch.Tensor,
    ) -> torch.Tensor:
        """Return ``log a`` per pair for umbrella exchange, by row.

        Parameters
        ----------
        rows_i, rows_j:
            Graph rows holding each pair's two slots, shape ``[P]``.
        bias_current:
            Reduced bias potential under the current assignment, shape ``[B]``.
        bias_swapped:
            Reduced bias potential under the proposed one, shape ``[B]``.

        Returns
        -------
        torch.Tensor
            ``log a`` per pair, capped at zero.
        """
        before = bias_current[rows_i] + bias_current[rows_j]
        after = bias_swapped[rows_i] + bias_swapped[rows_j]
        return torch.clamp(before - after, max=0.0)

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    def config_fingerprint(self) -> dict[str, Any]:
        """Return the configuration a restored run must match.

        Exchange semantics live in the ladder, not in the counters: the
        temperatures set the acceptance exponent, the rule follows from them,
        and the interval sets the segment cadence.  Restoring a run into a
        different ladder would keep the counters and the walker assignment
        while silently changing what a swap *means* — the walkers would carry
        on labelled 0..S-1 against temperatures they were never sampled at.

        ``initial_state_ids`` is deliberately excluded.  It seeds the
        assignment only when the batch does not already carry one, and a
        restored batch always does, so it has no effect after step zero and
        would be a false mismatch.

        Returns
        -------
        dict[str, Any]
            JSON-representable configuration.
        """
        return {
            "mode": self.mode,
            "acceptance": self._acceptance,
            "attempt_interval": int(self.attempt_interval),
            "temperatures": [float(state.temperature) for state in self.states],
        }

    @staticmethod
    def describe_config_mismatch(
        saved: Mapping[str, Any] | None, actual: Mapping[str, Any] | None
    ) -> list[str]:
        """Return human-readable differences between two fingerprints.

        Parameters
        ----------
        saved:
            The checkpoint's fingerprint, or ``None`` when it had no exchange.
        actual:
            The live runner's fingerprint, or ``None`` when it has none.

        Returns
        -------
        list[str]
            One line per difference; empty when they agree.
        """
        if saved is None and actual is None:
            return []
        if saved is None:
            return [
                "  exchange: the checkpoint was written without replica "
                "exchange, but this runner has one configured"
            ]
        if actual is None:
            return [
                "  exchange: the checkpoint was written with replica exchange "
                f"({saved.get('acceptance')}, "
                f"{len(saved.get('temperatures', []))} states), but this "
                "runner has replica_exchange=None"
            ]

        problems: list[str] = [
            f"  exchange {key}: checkpoint has {saved.get(key)!r}, "
            f"this runner has {actual.get(key)!r}"
            for key in ("mode", "acceptance", "attempt_interval")
            if saved.get(key) != actual.get(key)
        ]
        saved_temps = [float(t) for t in saved.get("temperatures", [])]
        actual_temps = [float(t) for t in actual.get("temperatures", [])]
        if len(saved_temps) != len(actual_temps):
            problems.append(
                f"  exchange ladder: checkpoint has {len(saved_temps)} state(s), "
                f"this runner has {len(actual_temps)}"
            )
        elif any(
            not math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-9)
            for a, b in zip(saved_temps, actual_temps, strict=True)
        ):
            problems.append(
                f"  exchange temperatures: checkpoint has {saved_temps}, "
                f"this runner has {actual_temps}"
            )
        return problems

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
            "pair_attempts": list(self.pair_attempts),
            "pair_accepted": list(self.pair_accepted),
            # Carried so the component is self-describing: loading it into a
            # different ladder is refused rather than silently accepted.
            "config": self.config_fingerprint(),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore exchange state produced by :meth:`state_dict`.

        Parameters
        ----------
        state:
            The mapping previously returned by :meth:`state_dict`.

        Raises
        ------
        ValueError
            If the saved configuration disagrees with this instance's.
        """
        saved_config = state.get("config")
        if saved_config is not None:
            problems = self.describe_config_mismatch(
                saved_config, self.config_fingerprint()
            )
            if problems:
                detail = "\n".join(problems)
                raise ValueError(
                    "ReplicaExchange.load_state_dict: the saved exchange was "
                    f"configured differently:\n{detail}\n"
                    "Counters and the acceptance-RNG position only mean "
                    "anything against the ladder they were produced on."
                )

        self.exchange_id = int(state.get("exchange_id", 0))
        self.attempts = int(state.get("attempts", 0))
        self.accepted = int(state.get("accepted", 0))
        # random_seed is RNG *position* and is restored; attempt_interval is
        # configuration and is validated above, never silently overwritten.
        self.random_seed = int(state.get("random_seed", self.random_seed))
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
