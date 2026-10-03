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
"""Pairwise swaps of per-system state, without the physics that decides them.

Strip the thermodynamics out of replica exchange and the mechanism is:
propose pairs, evaluate an acceptance rule, and permute per-system parameters
for the pairs that pass, rebinding any integrator state that travels with
them.  That mechanism is also basin hopping with swaps, population or
evolutionary structure search, and any annealing ladder — so it lives here,
and the physics is supplied as ``accept_fn``.

The half that was already reusable is on the integrator:
:meth:`~nvalchemi.dynamics.BaseDynamics.apply_per_system_params` rebinds the
parameters and transforms whatever private state depends on them — thermostat
chain masses, velocity scaling — as one indivisible change.

What a caller supplies
----------------------
``accept_fn(batch, i, j) -> Bool[Tensor, "P"]``
    The rule.  *i* and *j* are the graph rows holding the two slots of each
    proposed pair, so the rule reads whatever it needs off the batch — energy,
    a bias, an order parameter — and returns one decision per pair.
``pairing(segment, n_slots) -> list[tuple[int, int]]``
    Optional.  Which rungs to propose.  Its pairs must be disjoint — they are
    decided simultaneously, so an overlap has no defined meaning and its
    writes would collide — and that is checked rather than assumed.
``params_fn(slots) -> Mapping[str, Tensor]``
    Optional.  Maps the post-swap slot assignment to per-graph parameters for
    ``apply_per_system_params``.  Omit it when a method permutes labels only.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import torch

from nvalchemi.dynamics.base import DynamicsStage

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping
    from enum import Enum

    from nvalchemi.data import Batch
    from nvalchemi.hooks import HookContext

__all__ = ["PairSwapHook", "apply_pair_swaps", "even_odd_pairs"]


def even_odd_pairs(segment: int, n_slots: int) -> list[tuple[int, int]]:
    """Return the neighbouring slot pairs attempted in *segment*.

    Alternating even and odd offsets means every slot is exchangeable with
    both neighbours over two segments, while no slot appears in two pairs of
    the same segment — which is what lets every pair be decided
    simultaneously rather than in sequence.

    Parameters
    ----------
    segment:
        Segment index.  Even segments pair ``(0,1), (2,3), ...``; odd ones
        pair ``(1,2), (3,4), ...``.
    n_slots:
        Number of slots on the ladder.

    Returns
    -------
    list[tuple[int, int]]
        Neighbouring ``(slot, slot + 1)`` pairs, possibly empty.
    """
    offset = segment % 2
    return [(index, index + 1) for index in range(offset, n_slots - 1, 2)]


_PAIRINGS: dict[str, Callable[[int, int], list[tuple[int, int]]]] = {
    "even_odd": even_odd_pairs,
}


def _require_disjoint(values: torch.Tensor, what: str, detail: str) -> None:
    """Raise if *values* names anything twice.

    A swap writes each member of a pair into the other's place, so two pairs
    sharing a member write over each other: one label is duplicated and the
    other lost, and what comes out is no longer a permutation.

    Parameters
    ----------
    values:
        Every member of every pair, flattened.
    what:
        What the values are, for the error.
    detail:
        What the caller should do about it.

    Raises
    ------
    ValueError
        If any value appears more than once.
    """
    flat = values.reshape(-1).tolist()
    seen: set[int] = set()
    repeated = sorted({int(v) for v in flat if int(v) in seen or seen.add(int(v))})
    if repeated:
        raise ValueError(
            f"{what} appears in more than one pair: {repeated}. Pairs within "
            "a segment have to be disjoint — they are decided simultaneously, "
            f"and overlapping writes duplicate one label and lose another. "
            f"{detail}"
        )


def apply_pair_swaps(
    slots: torch.Tensor,
    rows_i: torch.Tensor,
    rows_j: torch.Tensor,
    accepted: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return *slots* with the accepted row pairs exchanged.

    Labels move, rows do not: a walker keeps its place in the batch and its
    slot assignment changes.  Nothing is copied between rows, which is what
    makes the move viable inside a batched GPU step.

    Keyed by row rather than by slot because rows are what every caller
    already holds — proposing a pair means naming the two rows that hold it —
    and re-deriving them from a slot lookup is the step that can disagree
    with the lookup the acceptance rule used.

    Parameters
    ----------
    slots:
        Current per-graph slot assignment, shape ``[B]``.
    rows_i, rows_j:
        Graph rows holding each pair's two slots, shape ``[P]``.
    accepted:
        One boolean per pair, or ``None`` to swap every pair — which is what
        a rule needs when it has to evaluate the *proposal* before deciding
        on it.

    Returns
    -------
    torch.Tensor
        A new assignment; *slots* is not modified.

    Raises
    ------
    ValueError
        If two accepted pairs share a row, which would make the result
        something other than a permutation.
    """
    if accepted is None:
        accepted = torch.ones(rows_i.numel(), dtype=torch.bool)
    take = accepted.to(dtype=torch.bool, device=slots.device)
    # Only the accepted pairs are written, so only those have to be disjoint:
    # a schedule may propose an overlap that acceptance happens to resolve.
    _require_disjoint(
        torch.cat([rows_i[take], rows_j[take]]),
        "apply_pair_swaps: a graph row",
        "Propose each row at most once per round.",
    )
    new_slots = slots.clone()
    new_slots[rows_i[take]] = slots[rows_j[take]]
    new_slots[rows_j[take]] = slots[rows_i[take]]
    return new_slots


class PairSwapHook:
    """Propose and accept pairwise swaps of per-system state.

    The cadence is :attr:`frequency`, so the hook registry gates the dispatch:
    fired at step *kN*, the hook acts on segment ``step // N - 1``, the one
    that has just completed.  Attempts stay idempotent by segment index,
    because a caller may also drain a completed segment from elsewhere — a
    checkpoint, say — and neither pass may decide the same segment twice.

    Parameters
    ----------
    accept_fn:
        ``accept_fn(batch, i, j) -> Bool[Tensor, "P"]``.  The physics: given
        the graph rows holding each proposed pair, return one decision per
        pair.  Called once per attempted segment, and not at all when the
        segment proposes no pairs.
    params_fn:
        ``params_fn(slots) -> Mapping[str, Tensor]``, mapping the post-swap
        assignment to per-graph parameters handed to
        ``apply_per_system_params``.  ``None`` for a method that permutes
        labels only and rebinds nothing.
    slot_field:
        Per-graph batch field holding the assignment.  Must be a permutation
        of ``0..n_slots-1``.  Required rather than defaulted: the name is the
        caller's, and a default borrowed from one method would quietly make
        this hook that method's.
    n_slots:
        Number of slots on the ladder.
    pairing:
        Pair schedule.  ``"even_odd"`` is the only built-in; pass a callable
        ``(segment, n_slots) -> list[tuple[int, int]]`` for another.
    frequency:
        Dynamics steps per segment.
    on_swap:
        Called with the batch after an accepted swap has been applied, for
        whatever the method has to repair — forces computed under the old
        parameters, most obviously.

    Notes
    -----
    Applying is the indivisible half: the slot labels, the rebound
    parameters, and anything derived from them move together.  Leaving any of
    them behind would sample a state the assignment says the walker is no
    longer in — which is why the parameter rebinding goes through
    ``apply_per_system_params`` rather than being open-coded per method.

    Indivisibility is enforced by ordering rather than by rollback: every
    step that can refuse — building the parameters, the engine checking it
    can rebind them — runs before the labels are written, and the label write
    itself cannot fail.  ``on_swap`` runs last because it repairs quantities
    derived from a swap that has, by then, definitely happened.
    """

    def __init__(
        self,
        accept_fn: Callable[[Batch, torch.Tensor, torch.Tensor], torch.Tensor],
        params_fn: Callable[[torch.Tensor], Mapping[str, torch.Tensor]] | None = None,
        *,
        slot_field: str,
        n_slots: int,
        pairing: Literal["even_odd"]
        | Callable[[int, int], list[tuple[int, int]]] = "even_odd",
        frequency: int = 100,
        on_swap: Callable[[Batch], None] | None = None,
    ) -> None:
        if isinstance(pairing, str) and pairing not in _PAIRINGS:
            raise ValueError(
                f"PairSwapHook: unknown pairing {pairing!r}. Built-in "
                f"schedules are {sorted(_PAIRINGS)}; pass a callable "
                "(segment, n_slots) -> list[tuple[int, int]] for another."
            )
        if int(frequency) < 1:
            raise ValueError(
                f"PairSwapHook: frequency must be at least 1, got {frequency}. "
                "It is the segment length, and the registry uses it to gate "
                "the dispatch."
            )
        self.stage: Enum | None = DynamicsStage.BEFORE_STEP
        self.frequency = int(frequency)
        self.accept_fn = accept_fn
        self.params_fn = params_fn
        self.slot_field = slot_field
        self.n_slots = int(n_slots)
        self.pairing = _PAIRINGS[pairing] if isinstance(pairing, str) else pairing
        self.on_swap = on_swap
        self.attempted_segment = -1
        self.dynamics: Any = None
        self._attempting = False

    def on_register(self, workflow: Any) -> None:
        """Remember the engine whose per-system parameters a swap rebinds.

        Parameters
        ----------
        workflow:
            The ``BaseDynamics`` doing the registering.
        """
        self.dynamics = workflow

    def __call__(self, ctx: HookContext, stage: Enum) -> None:
        """Attempt the segment that has just completed.

        Parameters
        ----------
        ctx:
            The dynamics hook context.
        stage:
            The stage being dispatched.
        """
        step = getattr(ctx, "step_count", 0)
        self.attempt_segment(ctx.batch, step // self.frequency - 1)

    def state_dict(self) -> Mapping[str, Any]:
        """Return the segment cursor that must survive a restart.

        Returns
        -------
        Mapping[str, Any]
            The last segment attempted.  A resumed run that lost it would
            re-decide a segment that was already decided.
        """
        return {"attempted_segment": int(self.attempted_segment)}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore the segment cursor.

        Parameters
        ----------
        state:
            A mapping produced by :meth:`state_dict`.
        """
        self.attempted_segment = int(state.get("attempted_segment", -1))

    def attempt_segment(self, batch: Batch, segment: int) -> None:
        """Attempt *segment*'s pairs, at most once.

        Parameters
        ----------
        batch:
            The live batch.
        segment:
            The completed segment.  Negative, or already attempted, is a
            no-op.

        Notes
        -----
        A segment that raises **before the swap commits** — a rejected
        rebinding, a malformed pairing — stays unattempted, so a caller that
        recovers can try it again.  The acceptance rule will have consumed a
        draw by then, so a retry is a fresh decision rather than a replay of
        the one that failed, which is the right trade against leaving the
        labels and the integrator disagreeing.

        Once the swap commits the cursor advances immediately, even if the
        post-swap repair then raises.  Past that line the batch has already
        moved, so a retry would not re-attempt the segment: it would decide
        afresh and swap an already-swapped batch a second time, giving two
        exchanges where the rule granted one.  A failed repair leaves derived
        quantities — forces evaluated under the old parameters — stale, which
        the next force evaluation corrects; a double swap is silent and
        corrects nothing.
        """
        if segment < 0 or segment <= self.attempted_segment or self._attempting:
            return
        self._attempting = True
        try:
            self._attempt(batch, segment)
        finally:
            self._attempting = False
        # Reached when nothing was committed and nothing raised — no pairs,
        # or none accepted. A segment that did commit advanced the cursor
        # itself, inside _attempt. The in-progress flag keeps a hook driven
        # re-entrantly from attempting the same segment twice.
        self.attempted_segment = max(self.attempted_segment, segment)

    def _validated_pairs(self, segment: int) -> list[tuple[int, int]]:
        """Return this segment's pairs, or say why they cannot be used.

        A pairing is an extension point, so what it returns is input.  Two
        pairs sharing a slot is the failure worth catching here: the writes
        overlap, one label is duplicated and another lost, and the assignment
        stops being a permutation — which the *next* segment reports, after a
        round has already been run at parameters nobody asked for.

        Parameters
        ----------
        segment:
            Segment index.

        Returns
        -------
        list[tuple[int, int]]
            The proposed pairs, possibly empty.

        Raises
        ------
        ValueError
            If a slot is out of range or appears in more than one pair.
        """
        pairs = self.pairing(segment, self.n_slots)
        if not pairs:
            return pairs

        members = [slot for pair in pairs for slot in pair]
        out_of_range = sorted(
            {slot for slot in members if not 0 <= slot < self.n_slots}
        )
        if out_of_range:
            raise ValueError(
                f"PairSwapHook: the pairing proposed slot(s) {out_of_range} "
                f"for a ladder of {self.n_slots}. A slot is an index into the "
                "ladder, so it has to name a rung that exists."
            )
        _require_disjoint(
            torch.tensor(members, dtype=torch.long),
            "PairSwapHook: a slot",
            "Return a schedule whose pairs do not touch — the built-in "
            "'even_odd' alternates offsets for exactly this reason.",
        )
        return pairs

    def _validated_slots(self, batch: Batch) -> torch.Tensor:
        """Return the assignment, or say why it cannot be swapped on.

        Pairing looks up "which row holds slot *k*", so the assignment has to
        be a bijection.  Checked here rather than left to fail later because
        the later failure is a bare ``KeyError`` from a dict lookup, naming
        neither the field nor the ladder.

        Parameters
        ----------
        batch:
            The live batch.

        Returns
        -------
        torch.Tensor
            The assignment as a ``[B]`` long tensor.

        Raises
        ------
        ValueError
            If the field is absent, the wrong length, or not a permutation of
            ``0..n_slots-1``.
        """
        values = getattr(batch, self.slot_field, None)
        if values is None:
            raise ValueError(
                f"PairSwapHook: the batch has no {self.slot_field!r} field, "
                "so there is no assignment to swap. Stamp it before the first "
                "step, or name the field the batch actually carries."
            )
        slots = values.reshape(-1).to(torch.long)
        if slots.numel() != self.n_slots:
            raise ValueError(
                f"PairSwapHook: the ladder has {self.n_slots} slot(s) but "
                f"{self.slot_field!r} holds {slots.numel()} entr(ies). A swap "
                "pairs slots with the rows holding them, which needs one row "
                "per slot."
            )
        if sorted(slots.tolist()) != list(range(self.n_slots)):
            raise ValueError(
                f"PairSwapHook: {self.slot_field!r} must be a permutation of "
                f"0..{self.n_slots - 1}, got {slots.tolist()}. A duplicate "
                "would let two rows claim the same slot, and the pair lookup "
                "would silently decide one of them twice."
            )
        return slots

    def _attempt(self, batch: Batch, segment: int) -> None:
        """Decide one round of swaps and apply the accepted ones.

        Parameters
        ----------
        batch:
            The live batch.
        segment:
            Segment index, which selects the pairing.
        """
        pairs = self._validated_pairs(segment)
        if not pairs:
            return

        slots = self._validated_slots(batch)
        row_of_slot = {int(slot): row for row, slot in enumerate(slots.tolist())}
        rows_i = torch.tensor(
            [row_of_slot[i] for i, _ in pairs], device=slots.device, dtype=torch.long
        )
        rows_j = torch.tensor(
            [row_of_slot[j] for _, j in pairs], device=slots.device, dtype=torch.long
        )

        accepted = self.accept_fn(batch, rows_i, rows_j)
        if not bool(accepted.any()):
            return

        new_slots = apply_pair_swaps(slots, rows_i, rows_j, accepted)

        # Everything that can refuse the swap runs before anything is written.
        # The labels are the cheapest half to commit and the most damaging to
        # commit alone: a batch saying a walker moved rung while the
        # integrator still targets the old one samples a state the assignment
        # says it has left, which is exactly the indivisibility this hook
        # exists to provide.
        params = None
        if self.params_fn is not None:
            if self.dynamics is None:
                raise RuntimeError(
                    "PairSwapHook: no engine to rebind on. The hook takes it "
                    "from on_register, so register it on the dynamics rather "
                    "than driving attempt_segment() by hand."
                )
            params = self.params_fn(new_slots)

        if params is not None:
            # Implementations validate the parameters they were handed before
            # touching any state, so a rebinding they refuse leaves both the
            # integrator and the batch as they were.
            self.dynamics.apply_per_system_params(params, batch)

        batch[self.slot_field] = new_slots

        # The swap is now committed: the integrator and the labels both moved.
        # The cursor advances here rather than after on_swap, because from
        # this line on a retry would not re-attempt the segment, it would
        # swap an already-swapped batch a second time — two exchanges where
        # the acceptance rule granted one.
        self.attempted_segment = segment

        # Last, and deliberately after the labels: this is repair work on
        # quantities derived from the swap — forces computed under the old
        # parameters — so it needs the swap to have happened.
        if self.on_swap is not None:
            self.on_swap(batch)
