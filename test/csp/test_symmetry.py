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

import pytest
import torch
from pydantic import ValidationError

from nvalchemi.csp import SpaceGroupPolicy
from nvalchemi.csp.symmetry import (
    CrystalSystem,
    csd_space_group_probabilities,
    get_crystal_system,
    get_space_group_candidates,
    get_space_group_operation_count,
    get_space_group_operations,
    is_sohncke_space_group,
    sample_space_groups,
)


@pytest.mark.parametrize("space_group,count", [(1, 1), (2, 2), (14, 4), (225, 192)])
def test_operations_and_counts(space_group: int, count: int) -> None:
    operations = get_space_group_operations(space_group)
    assert operations.shape == (count, 3, 4)
    assert operations.dtype == torch.float32
    assert get_space_group_operation_count(space_group) == count
    assert torch.equal(operations[0, :, :3], torch.eye(3))


def test_crystal_system_sohncke_and_candidates() -> None:
    assert get_crystal_system(1) is CrystalSystem.TRICLINIC
    assert get_crystal_system(225) is CrystalSystem.CUBIC
    assert is_sohncke_space_group(1)
    assert not is_sohncke_space_group(2)
    candidates = get_space_group_candidates(2, sohncke_only=True)
    assert candidates.dtype == torch.int32
    assert 3 in candidates.tolist()
    assert 2 not in candidates.tolist()


def test_candidate_filter_rejects_invalid_public_inputs() -> None:
    with pytest.raises(TypeError, match="num_operations must be an integer"):
        get_space_group_candidates(True)
    with pytest.raises(ValueError, match="num_operations must be positive"):
        get_space_group_candidates(0)
    with pytest.raises(TypeError, match="crystal_system"):
        get_space_group_candidates(2, crystal_system=0)
    with pytest.raises(TypeError, match="sohncke_only"):
        get_space_group_candidates(2, sohncke_only=1)


def test_csd_prior_is_full_and_fresh_and_custom_prior_replaces_it() -> None:
    first = csd_space_group_probabilities()
    second = csd_space_group_probabilities()
    assert first == second
    assert set(first) == set(range(1, 231))
    assert first[1] == pytest.approx(0.010246754283840977)
    assert first[14] == pytest.approx(0.3380245087119827)
    first.clear()
    assert second
    draws = sample_space_groups(20, 2, probabilities={2: 1.0}, seed=11)
    assert torch.equal(draws, torch.full((20,), 2, dtype=torch.int32))


def test_sampling_filters_prior_and_seed_is_repeatable() -> None:
    prior = {3: 0.25, 4: 0.75, 1: 1.0}
    first = sample_space_groups(
        50,
        2,
        probabilities=prior,
        crystal_system=CrystalSystem.MONOCLINIC,
        sohncke_only=True,
        seed=27,
    )
    second = sample_space_groups(
        50,
        2,
        probabilities=prior,
        crystal_system=CrystalSystem.MONOCLINIC,
        sohncke_only=True,
        seed=27,
    )
    assert torch.equal(first, second)
    assert set(first.tolist()) <= {3, 4}


def test_sampling_requires_surviving_positive_mass() -> None:
    with pytest.raises(ValueError, match="positive"):
        sample_space_groups(1, 2, probabilities={3: 0.0})
    with pytest.raises(ValueError, match="positive"):
        sample_space_groups(1, 1, probabilities={2: 1.0})


def test_fixed_policy_draw_repeats_group_without_advancing_random_state() -> None:
    policy = SpaceGroupPolicy.fixed(2)
    torch.manual_seed(71)
    before = torch.random.get_rng_state()
    draws = policy.draw(5, num_operations=2, seed=13)
    after = torch.random.get_rng_state()
    assert torch.equal(draws, torch.full((5,), 2, dtype=torch.int32))
    assert draws.device.type == "cpu"
    assert torch.equal(before, after)
    with pytest.raises(ValueError, match="2 operations; 1 required"):
        policy.draw(1, num_operations=1)


def test_sampled_policy_copies_and_reuses_full_prior_with_stable_weights() -> None:
    prior = {3: 1.0, 14: 3.0, 19: 1.0}
    policy = SpaceGroupPolicy.sampled(probabilities=prior)
    prior[14] = 0.0
    assert policy.probabilities == {3: 1.0, 14: 3.0, 19: 1.0}
    with pytest.raises(TypeError):
        policy.probabilities[14] = 0.0  # type: ignore[index]

    draws = policy.draw(4000, num_operations=4, seed=531)
    assert draws.dtype == torch.int32
    assert set(draws.tolist()) == {14, 19}
    assert (draws == 14).float().mean().item() == pytest.approx(0.75, abs=0.04)

    two_operation_draws = policy.draw(10, num_operations=2, seed=2)
    assert torch.equal(two_operation_draws, torch.full((10,), 3, dtype=torch.int32))
    huge_weight_draws = SpaceGroupPolicy.sampled(
        probabilities={14: 1.0e308, 19: 1.0e308}
    ).draw(20, num_operations=4, seed=7)
    assert set(huge_weight_draws.tolist()) == {14, 19}


def test_policy_constructors_reject_contradictory_and_invalid_values() -> None:
    with pytest.raises(ValueError, match="Sohncke"):
        SpaceGroupPolicy.fixed(2, sohncke_only=True)
    with pytest.raises(TypeError, match="integer"):
        SpaceGroupPolicy.fixed(True)
    with pytest.raises(TypeError, match="crystal_system"):
        SpaceGroupPolicy.sampled(crystal_system=0)  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        SpaceGroupPolicy(mode="fixed", group=1, probabilities={1: 1.0})
    with pytest.raises(ValidationError):
        SpaceGroupPolicy(mode="sampled", group=1)
    with pytest.raises(ValidationError):
        SpaceGroupPolicy(mode="sampled", unrecognized=True)


def test_sample_helper_forwards_to_policy_draw() -> None:
    kwargs = {
        "probabilities": {14: 2.0, 19: 1.0},
        "crystal_system": CrystalSystem.ORTHORHOMBIC,
        "sohncke_only": True,
    }
    forwarded = sample_space_groups(50, 4, seed=18, **kwargs)
    direct = SpaceGroupPolicy.sampled(**kwargs).draw(50, num_operations=4, seed=18)
    assert torch.equal(forwarded, direct)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_policy_draw_places_output_on_requested_device() -> None:
    result = SpaceGroupPolicy.sampled(probabilities={14: 1.0}).draw(
        8, num_operations=4, seed=8, device="cuda:0"
    )
    assert result.dtype == torch.int32
    assert result.device == torch.device("cuda:0")
    assert set(result.cpu().tolist()) == {14}


@pytest.mark.parametrize(
    "probabilities",
    [{0: 1.0}, {231: 1.0}, {1: -1.0}, {1: float("nan")}, {1: float("inf")}],
)
def test_invalid_probability_mapping_rejected(probabilities: dict[int, float]) -> None:
    with pytest.raises((TypeError, ValueError)):
        sample_space_groups(1, 1, probabilities=probabilities)


@pytest.mark.parametrize("space_group", [0, 231])
def test_invalid_space_group_rejected(space_group: int) -> None:
    for lookup in (
        get_space_group_operations,
        get_space_group_operation_count,
        get_crystal_system,
        is_sohncke_space_group,
    ):
        with pytest.raises(ValueError, match=r"\[1, 230\]"):
            lookup(space_group)


@pytest.mark.parametrize("space_group", [True, 3.5, "3"])
def test_non_integer_space_group_rejected(space_group: object) -> None:
    for lookup in (
        get_space_group_operations,
        get_space_group_operation_count,
        get_crystal_system,
        is_sohncke_space_group,
    ):
        with pytest.raises(TypeError, match="integer"):
            lookup(space_group)


def test_operation_dtype_must_be_floating() -> None:
    with pytest.raises(TypeError, match="floating"):
        get_space_group_operations(3, dtype=torch.int32)
