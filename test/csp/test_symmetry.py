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
