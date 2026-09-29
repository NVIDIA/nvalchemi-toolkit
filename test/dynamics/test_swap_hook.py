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
"""Unit tests for ``PairSwapHook`` and its schedule.

The mechanism with the physics taken out: propose pairs, ask an acceptance
rule, permute per-system parameters for the pairs that pass.  Replica exchange
is one caller; basin hopping with swaps and population search are the others
this exists for, so everything here is asserted with a hand-written
``accept_fn`` rather than through a ladder.
"""

from __future__ import annotations

from collections.abc import Mapping

import pytest
import torch

from nvalchemi.data import AtomicData, Batch
from nvalchemi.dynamics.base import DynamicsStage
from nvalchemi.dynamics.hooks import PairSwapHook, apply_pair_swaps, even_odd_pairs


def _make_batch(n_graphs: int = 4) -> Batch:
    """Return a batch carrying a slot assignment."""
    torch.manual_seed(0)
    data_list = [
        AtomicData(
            positions=torch.randn(3, 3),
            atomic_numbers=torch.full((3,), 6, dtype=torch.long),
            atomic_masses=torch.ones(3),
            forces=torch.zeros(3, 3),
            energy=torch.zeros(1, 1),
        )
        for _ in range(n_graphs)
    ]
    batch = Batch.from_data_list(data_list)
    batch["slot"] = torch.arange(n_graphs, dtype=torch.long)
    return batch


class _RecordingEngine:
    """Stand-in for the engine, recording what a swap rebound."""

    def __init__(self) -> None:
        self.calls: list[dict[str, list[float]]] = []

    def apply_per_system_params(
        self, params: Mapping[str, torch.Tensor], batch: Batch
    ) -> None:
        """Record the rebinding."""
        self.calls.append({k: v.reshape(-1).tolist() for k, v in params.items()})


class _Ctx:
    """Minimal stand-in for the dynamics hook context."""

    def __init__(self, batch: Batch | None, step: int) -> None:
        self.batch = batch
        self.step_count = step


def _accept_all(batch: Batch, i: torch.Tensor, j: torch.Tensor) -> torch.Tensor:
    """Accept every proposed pair."""
    return torch.ones(i.numel(), dtype=torch.bool)


def _accept_none(batch: Batch, i: torch.Tensor, j: torch.Tensor) -> torch.Tensor:
    """Reject every proposed pair."""
    return torch.zeros(i.numel(), dtype=torch.bool)


# ===========================================================================
# 1. The pair schedule
# ===========================================================================


class TestEvenOddPairs:
    """No slot may appear twice in a segment, or the pairs are not disjoint."""

    @pytest.mark.parametrize(
        ("segment", "n_slots", "expected"),
        [
            (0, 4, [(0, 1), (2, 3)]),
            (1, 4, [(1, 2)]),
            (2, 4, [(0, 1), (2, 3)]),
            (0, 2, [(0, 1)]),
            (1, 2, []),
            (0, 1, []),
        ],
    )
    def test_schedule(
        self, segment: int, n_slots: int, expected: list[tuple[int, int]]
    ) -> None:
        assert even_odd_pairs(segment, n_slots) == expected

    @pytest.mark.parametrize("segment", range(4))
    def test_pairs_within_a_segment_are_disjoint(self, segment: int) -> None:
        """Disjointness is what lets every pair be decided simultaneously."""
        seen = [slot for pair in even_odd_pairs(segment, 6) for slot in pair]
        assert len(seen) == len(set(seen))

    def test_two_segments_cover_every_neighbour(self) -> None:
        both = set(even_odd_pairs(0, 5)) | set(even_odd_pairs(1, 5))
        assert both == {(0, 1), (1, 2), (2, 3), (3, 4)}


# ===========================================================================
# 2. The permutation
# ===========================================================================


class TestApplyPairSwaps:
    """Labels move; rows do not."""

    def test_accepted_pairs_exchange_labels(self) -> None:
        slots = torch.tensor([0, 1, 2, 3])
        rows = {0: 0, 1: 1, 2: 2, 3: 3}
        new = apply_pair_swaps(
            slots, [(0, 1), (2, 3)], torch.tensor([True, False]), rows
        )
        assert new.tolist() == [1, 0, 2, 3]

    def test_the_input_is_not_modified(self) -> None:
        slots = torch.tensor([0, 1])
        apply_pair_swaps(slots, [(0, 1)], torch.tensor([True]), {0: 0, 1: 1})
        assert slots.tolist() == [0, 1]

    def test_the_result_is_still_a_permutation(self) -> None:
        slots = torch.tensor([2, 0, 3, 1])
        rows = {int(s): r for r, s in enumerate(slots.tolist())}
        new = apply_pair_swaps(
            slots, [(0, 1), (2, 3)], torch.tensor([True, True]), rows
        )
        assert sorted(new.tolist()) == [0, 1, 2, 3]


# ===========================================================================
# 3. The hook
# ===========================================================================


class TestPairSwapHook:
    """Cadence, idempotence, and what an accepted swap is obliged to move."""

    def _hook(self, **kwargs: object) -> PairSwapHook:
        """Return a hook over a four-slot ladder."""
        defaults = {
            "slot_field": "slot",
            "n_slots": 4,
            "frequency": 2,
        }
        return PairSwapHook(
            kwargs.pop("accept_fn", _accept_all), **{**defaults, **kwargs}
        )

    def test_stage_and_frequency(self) -> None:
        hook = self._hook()
        assert hook.stage is DynamicsStage.BEFORE_STEP
        assert hook.frequency == 2

    def test_a_non_positive_frequency_is_refused(self) -> None:
        with pytest.raises(ValueError, match="at least 1"):
            PairSwapHook(_accept_all, slot_field="slot", n_slots=4, frequency=0)

    def test_dispatch_acts_on_the_completed_segment(self) -> None:
        """At step kN the segment that just finished is k-1, not k."""
        seen: list[int] = []
        hook = self._hook()
        hook._attempt = lambda batch, segment: seen.append(segment)  # type: ignore[method-assign]
        for k in range(3):
            hook(_Ctx(_make_batch(), k * 2), DynamicsStage.BEFORE_STEP)
        assert seen == [0, 1]

    def test_a_segment_is_attempted_at_most_once(self) -> None:
        seen: list[int] = []
        hook = self._hook()
        hook._attempt = lambda batch, segment: seen.append(segment)  # type: ignore[method-assign]
        hook.attempt_segment(None, -1)
        hook.attempt_segment(None, 0)
        hook.attempt_segment(None, 0)
        hook.attempt_segment(None, 1)
        assert seen == [0, 1]
        assert hook.attempted_segment == 1

    def test_accepted_swap_writes_the_new_assignment(self) -> None:
        batch = _make_batch()
        hook = self._hook()
        hook.on_register(_RecordingEngine())
        hook.attempt_segment(batch, 0)
        assert batch.slot.reshape(-1).tolist() == [1, 0, 3, 2]

    def test_rejected_swap_changes_nothing(self) -> None:
        batch = _make_batch()
        hook = self._hook(accept_fn=_accept_none)
        hook.on_register(_RecordingEngine())
        hook.attempt_segment(batch, 0)
        assert batch.slot.reshape(-1).tolist() == [0, 1, 2, 3]

    def test_params_are_rebound_from_the_post_swap_assignment(self) -> None:
        """The whole point: labels and integrator parameters move together."""
        table = torch.tensor([300.0, 350.0, 400.0, 450.0])
        engine = _RecordingEngine()
        hook = self._hook(params_fn=lambda slots: {"temperature": table[slots]})
        hook.on_register(engine)
        hook.attempt_segment(_make_batch(), 0)
        assert engine.calls == [{"temperature": [350.0, 300.0, 450.0, 400.0]}]

    def test_nothing_is_rebound_when_no_pair_is_accepted(self) -> None:
        engine = _RecordingEngine()
        hook = self._hook(
            accept_fn=_accept_none, params_fn=lambda slots: {"temperature": slots}
        )
        hook.on_register(engine)
        hook.attempt_segment(_make_batch(), 0)
        assert engine.calls == []

    def test_an_empty_segment_never_reaches_the_rule(self) -> None:
        """A two-slot ladder has no odd-segment pair to decide."""
        asked: list[int] = []

        def _spy(batch, i, j):
            asked.append(i.numel())
            return torch.ones(i.numel(), dtype=torch.bool)

        hook = PairSwapHook(_spy, slot_field="slot", n_slots=2, frequency=2)
        hook.on_register(_RecordingEngine())
        hook.attempt_segment(_make_batch(n_graphs=2), 1)
        assert asked == []

    def test_on_swap_runs_after_an_accepted_swap(self) -> None:
        """Forces computed under the old parameters are what this repairs."""
        repaired: list[int] = []
        hook = self._hook(on_swap=lambda batch: repaired.append(batch.num_graphs))
        hook.on_register(_RecordingEngine())
        hook.attempt_segment(_make_batch(), 0)
        assert repaired == [4]

    def test_on_swap_does_not_run_when_nothing_was_accepted(self) -> None:
        repaired: list[int] = []
        hook = self._hook(
            accept_fn=_accept_none, on_swap=lambda batch: repaired.append(1)
        )
        hook.on_register(_RecordingEngine())
        hook.attempt_segment(_make_batch(), 0)
        assert repaired == []

    def test_the_rule_is_given_the_rows_holding_each_pair(self) -> None:
        """Rows, not slots: a walker keeps its row while its label moves."""
        batch = _make_batch()
        batch["slot"] = torch.tensor([2, 0, 3, 1])
        seen: list[tuple[list[int], list[int]]] = []

        def _spy(b, i, j):
            seen.append((i.tolist(), j.tolist()))
            return torch.zeros(i.numel(), dtype=torch.bool)

        hook = PairSwapHook(_spy, slot_field="slot", n_slots=4, frequency=2)
        hook.on_register(_RecordingEngine())
        hook.attempt_segment(batch, 0)
        # slot 0 is on row 1, slot 1 on row 3, slot 2 on row 0, slot 3 on row 2.
        assert seen == [([1, 0], [3, 2])]

    def test_a_custom_pairing_is_honoured(self) -> None:
        hook = PairSwapHook(
            _accept_all,
            slot_field="slot",
            n_slots=4,
            frequency=2,
            pairing=lambda segment, n: [(0, 3)],
        )
        hook.on_register(_RecordingEngine())
        batch = _make_batch()
        hook.attempt_segment(batch, 0)
        assert batch.slot.reshape(-1).tolist() == [3, 1, 2, 0]

    def test_a_non_permutation_assignment_is_named(self) -> None:
        """Pairing is a bijection lookup; a duplicate must not reach it.

        Left unchecked the failure is a bare ``KeyError`` from a dict lookup
        inside the hook, naming neither the field nor the ladder.
        """
        batch = _make_batch()
        batch["slot"] = torch.tensor([0, 0, 2, 3])
        hook = self._hook()
        hook.on_register(_RecordingEngine())
        with pytest.raises(ValueError, match="must be a permutation"):
            hook.attempt_segment(batch, 0)

    def test_a_wrong_length_assignment_is_named(self) -> None:
        batch = _make_batch(n_graphs=2)
        hook = self._hook()
        hook.on_register(_RecordingEngine())
        with pytest.raises(ValueError, match="slot\\(s\\) but"):
            hook.attempt_segment(batch, 0)

    def test_a_missing_assignment_field_is_named(self) -> None:
        hook = PairSwapHook(
            _accept_all, slot_field="not_stamped", n_slots=4, frequency=2
        )
        hook.on_register(_RecordingEngine())
        with pytest.raises(ValueError, match="no 'not_stamped' field"):
            hook.attempt_segment(_make_batch(), 0)

    def test_rebinding_without_an_engine_is_named(self) -> None:
        """params_fn needs the engine on_register supplies."""
        hook = self._hook(params_fn=lambda slots: {"temperature": slots.float()})
        with pytest.raises(RuntimeError, match="no engine to rebind on"):
            hook.attempt_segment(_make_batch(), 0)

    def test_an_unknown_pairing_name_is_named(self) -> None:
        with pytest.raises(ValueError, match="unknown pairing"):
            PairSwapHook(
                _accept_all, slot_field="slot", n_slots=4, pairing="round_robin"
            )

    def test_the_segment_cursor_round_trips(self) -> None:
        hook = self._hook()
        hook.attempted_segment = 7
        other = self._hook()
        other.load_state_dict(hook.state_dict())
        assert other.attempted_segment == 7
