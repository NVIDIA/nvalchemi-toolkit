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
"""Tests for dependency-aware simulation campaigns."""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from nvalchemi.data import AtomicData
from nvalchemi.scheduling import (
    CampaignScheduler,
    CampaignSpec,
    FinalStateStore,
    RunSpec,
)


def _reference(run_id: str, delta_mu: float) -> RunSpec:
    """Construct one high-temperature Au-Pt reference state point."""
    return RunSpec(
        run_id=run_id,
        temperature_k=3000.0,
        chemical_potentials_ev={79: 0.0, 78: delta_mu},
        species=(79, 78),
        metadata={"particle_size": 1000},
    )


def _state(value: float) -> AtomicData:
    """Create a minimal state with a distinguishable final coordinate."""
    return AtomicData(
        atomic_numbers=torch.tensor([79], dtype=torch.long),
        positions=torch.tensor([[value, 0.0, 0.0]]),
        velocities=torch.zeros(1, 3),
        cell=torch.eye(3).unsqueeze(0),
        pbc=torch.tensor([[True, True, True]]),
    )


def test_cooling_campaign_unlocks_one_temperature_at_a_time(tmp_path) -> None:
    """Each cooling child becomes ready only after its parent is checkpointed."""
    campaign = CampaignSpec.cooling_from_reference(
        [_reference("mu_minus", -0.1), _reference("mu_plus", 0.1)],
        [3000.0, 2800.0, 2600.0],
    )
    scheduler = CampaignScheduler(campaign, FinalStateStore(tmp_path))

    assert [run.run_id for run in scheduler.ready()] == ["mu_minus", "mu_plus"]
    assert [len(batch) for batch in scheduler.ready_batches(max_batch_size=2)] == [2]
    waves = scheduler.ready_batch_waves(max_batch_size=2, gpu_ids=[3, 5])
    assert [
        (assignment.gpu_id, assignment.wave, len(batch)) for assignment, batch in waves
    ] == [(3, 0, 2)]

    scheduler.complete("mu_minus", _state(1.0))
    assert [run.run_id for run in scheduler.ready()] == [
        "mu_plus",
        "mu_minus.cool.T2800",
    ]
    parent = scheduler.parent_state(campaign.by_id["mu_minus.cool.T2800"])
    assert parent is not None
    assert parent.positions.tolist() == [[1.0, 0.0, 0.0]]


def test_cooling_barrier_waits_for_both_high_temperature_scan_directions(
    tmp_path,
) -> None:
    """Cooling can wait until a complete bidirectional reference scan is ready."""
    references = [
        _reference("up_start", -0.1),
        RunSpec(
            run_id="up_end",
            temperature_k=3000.0,
            chemical_potentials_ev={79: 0.0, 78: 0.1},
            species=(79, 78),
            parent_id="up_start",
        ),
        _reference("down_start", 0.1),
        RunSpec(
            run_id="down_end",
            temperature_k=3000.0,
            chemical_potentials_ev={79: 0.0, 78: -0.1},
            species=(79, 78),
            parent_id="down_start",
        ),
    ]
    campaign = CampaignSpec.cooling_from_reference(
        references,
        [3000.0, 2800.0],
        start_after=["up_end", "down_end"],
    )
    scheduler = CampaignScheduler(campaign, FinalStateStore(tmp_path))

    scheduler.complete("up_start", _state(1.0))
    scheduler.complete("up_end", _state(2.0))
    scheduler.complete("down_start", _state(3.0))
    assert [run.run_id for run in scheduler.ready()] == ["down_end"]

    scheduler.complete("down_end", _state(4.0))
    assert {run.run_id for run in scheduler.ready()} == {
        "up_start.cool.T2800",
        "up_end.cool.T2800",
        "down_start.cool.T2800",
        "down_end.cool.T2800",
    }


def test_ready_batches_separate_incompatible_methods(tmp_path) -> None:
    """Method or species changes cannot accidentally share one runner."""
    campaign = CampaignSpec(
        runs=(
            _reference("sgc", 0.0),
            RunSpec(
                run_id="canonical",
                temperature_k=3000.0,
                chemical_potentials_ev={79: 0.0, 78: 0.0},
                species=(79, 78),
                method_key="hybrid_kawasaki_npt",
            ),
        )
    )
    scheduler = CampaignScheduler(campaign, FinalStateStore(tmp_path))

    batches = scheduler.ready_batches(max_batch_size=8)

    assert [[run.run_id for run in batch] for batch in batches] == [
        ["sgc"],
        ["canonical"],
    ]


def test_campaign_rejects_continuation_cycle() -> None:
    """Continuation dependency edges must form a directed acyclic graph."""
    first = _reference("first", 0.0)
    second = _reference("second", 0.0)
    first = replace(first, parent_id="second")
    second = replace(second, parent_id="first")

    with pytest.raises(ValueError, match="cycle"):
        CampaignSpec(runs=(first, second))


def test_run_spec_is_hashable_and_consistent_with_equality() -> None:
    """Runs work in sets and as dict keys, including unhashable metadata values."""
    run = _reference("ref", 0.1)
    same = _reference("ref", 0.1)
    listed = RunSpec(
        run_id="listed",
        temperature_k=3000.0,
        chemical_potentials_ev={79: 0.0, 78: 0.1},
        metadata={"tags": ["a", "b"]},
    )

    assert run == same and hash(run) == hash(same)
    assert {run, same, _reference("other", 0.1), listed} == {
        run,
        _reference("other", 0.1),
        listed,
    }
    assert {run: 1}[same] == 1


def test_delta_mu_scan_branches_march_from_their_own_endpoints(tmp_path) -> None:
    """Each branch chains on its own previous step and never on the other branch."""
    campaign = CampaignSpec.delta_mu_scan_from_endpoints(
        [_reference("lo", -0.2), _reference("hi", 0.2)],
        [[-0.2, 0.0, 0.2], [0.2, 0.0, -0.2]],
        species=78,
    )
    by_id = campaign.by_id
    for seed in ("lo", "hi"):
        chain = [run for run in campaign.runs if run.run_id.startswith(f"{seed}.dmu")]
        assert [run.parent_id for run in chain] == [seed, chain[0].run_id]
        assert all(by_id[run.parent_id].run_id.startswith(seed) for run in chain)
    assert [run.chemical_potentials_ev[78] for run in campaign.runs] == [
        -0.2,
        0.2,
        0.0,
        0.2,
        0.0,
        -0.2,
    ]

    scheduler = CampaignScheduler(campaign, FinalStateStore(tmp_path))
    assert [run.run_id for run in scheduler.ready()] == ["lo", "hi"]
    scheduler.complete("lo", _state(0.0))
    assert [run.run_id for run in scheduler.ready()] == ["hi", "lo.dmu1.mu0"]


def test_delta_mu_scan_ladder_must_start_at_its_seed() -> None:
    with pytest.raises(ValueError, match="must start at its seed"):
        CampaignSpec.delta_mu_scan_from_endpoints(
            [_reference("lo", -0.2), _reference("hi", 0.2)],
            [[0.0, 0.2], [0.2, 0.0]],
            species=78,
        )
