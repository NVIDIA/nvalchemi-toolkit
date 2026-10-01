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
"""Public configuration and result contracts for the CSP packer."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest
import torch
from pydantic import ValidationError

from nvalchemi.csp.data import MolecularPackingInput, RigidMoleculeASUBatch
from nvalchemi.csp.packer import (
    CandidateBudgetPacker,
    CrystalPacker,
    OverlapReliefConfig,
    OverlapReliefPacker,
    OverlapReliefProgress,
    PackingContext,
    PackingReport,
    PackingResult,
    PackingStopReason,
)
from nvalchemi.csp.symmetry import CrystalSystem, SpaceGroupPolicy
from nvalchemi.data.batch import Batch


def make_config(**overrides: object) -> OverlapReliefConfig:
    values: dict[str, object] = {
        "z": 1,
        "z_prime": 1,
        "batch_size": 4,
        "cell_volume_scale_range": (0.8, 1.2),
    }
    values.update(overrides)
    if "cell_volume_range" in overrides:
        values.pop("cell_volume_scale_range", None)
    if "cell_volume_scale_range" in overrides:
        values.pop("cell_volume_range", None)
    return OverlapReliefConfig(**values)


def _round_trip_config(config: OverlapReliefConfig, route: str) -> OverlapReliefConfig:
    if route == "json_text":
        return OverlapReliefConfig.model_validate_json(config.model_dump_json())
    if route == "json_dict":
        return OverlapReliefConfig.model_validate(config.model_dump(mode="json"))
    return OverlapReliefConfig.model_validate(config.model_dump(mode="python"))


def make_structures() -> RigidMoleculeASUBatch:
    packing_input = MolecularPackingInput(
        conformer_positions=torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32),
        conformer_ptr=torch.tensor([0, 1], dtype=torch.int32),
        molecule_conformer_ptr=torch.tensor([0, 1], dtype=torch.int32),
        molecule_atom_ptr=torch.tensor([0, 1], dtype=torch.int32),
        atomic_numbers=torch.tensor([6], dtype=torch.int64),
        contact_distances=torch.tensor([[1.0]], dtype=torch.float32),
        component_index=torch.tensor([0], dtype=torch.int32),
        component_charge=torch.zeros(1, dtype=torch.int32),
        formula_unit_volume=10.0,
    )
    return RigidMoleculeASUBatch(
        packing_input=packing_input,
        structure_molecule_ptr=torch.tensor([0, 1, 2], dtype=torch.int32),
        conformer_indices=torch.tensor([0, 0], dtype=torch.int32),
        rotations=torch.eye(3, dtype=torch.float32).repeat(2, 1, 1),
        fractional_centers=torch.zeros((2, 3), dtype=torch.float32),
        cells=torch.eye(3, dtype=torch.float32).repeat(2, 1, 1),
        space_groups=torch.ones(2, dtype=torch.int32),
        z=torch.ones(2, dtype=torch.int32),
        z_prime=torch.ones(2, dtype=torch.int32),
    )


def test_config_requires_exactly_one_cell_volume_mode() -> None:
    with pytest.raises(ValidationError, match="exactly one"):
        OverlapReliefConfig(z=1, z_prime=1, batch_size=1)
    with pytest.raises(ValidationError, match="exactly one"):
        OverlapReliefConfig(
            z=1,
            z_prime=1,
            batch_size=4,
            cell_volume_range=(10.0, 20.0),
            cell_volume_scale_range=(0.8, 1.2),
        )
    assert make_config(cell_volume_range=(10.0, 20.0)).cell_volume_scale_range is None


def test_default_space_group_policy_is_sampled_with_csd_prior() -> None:
    policy = make_config().space_groups
    assert isinstance(policy, SpaceGroupPolicy)
    assert policy.mode == "sampled"
    assert policy.probabilities is None


@pytest.mark.parametrize(
    "updates, message",
    [
        ({"z": 0}, "positive integer"),
        ({"z": 3, "z_prime": 2}, "divide z"),
        ({"batch_size": 0}, "positive integer"),
        ({"max_candidates": 0}, "positive integer"),
        ({"max_step": float("inf")}, "finite and positive"),
        ({"max_step": 0.0}, "finite and positive"),
        ({"overlap_tolerance": -0.1}, "finite and nonnegative"),
        ({"cell_volume_scale_range": (2.0, 1.0)}, "ascending order"),
        ({"max_axis_ratio": 0.9}, "at least 1"),
        ({"cell_oversample_factor": 0.9}, "at least 1"),
        ({"space_groups": SpaceGroupPolicy.fixed(2)}, "operations"),
    ],
)
def test_config_rejects_invalid_values(
    updates: dict[str, object], message: str
) -> None:
    with pytest.raises(ValidationError, match=message):
        make_config(**updates)


def test_automatic_candidate_budget_mode_is_preserved_and_overridable() -> None:
    config = make_config()
    assert config.max_candidates == "auto"
    assert config.effective(max_steps_per_candidate=20).max_candidates == "auto"
    assert config.effective(max_candidates=None).max_candidates is None
    assert config.effective(max_candidates=7).max_candidates == 7
    assert (
        make_config(max_candidates=7).effective(max_candidates="auto").max_candidates
        == "auto"
    )

    for budget in ("auto", None, 7):
        serialized = make_config(max_candidates=budget).model_dump()
        assert OverlapReliefConfig.model_validate(serialized).max_candidates == budget


@pytest.mark.parametrize("budget", [True, 1.5, "unlimited"])
def test_config_rejects_invalid_candidate_budget_types(budget: object) -> None:
    with pytest.raises(ValidationError):
        make_config(max_candidates=budget)


def test_config_is_frozen_forbids_extra_keys_and_revalidates_effective() -> None:
    config = make_config(max_candidates=2)
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        make_config(num_samples=5)
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        config.effective(unrecognized=3)
    with pytest.raises(ValidationError):
        config.effective(z_prime=2)
    updated = config.effective(max_steps_per_candidate=20)
    assert config.max_steps_per_candidate == 500
    assert updated.max_steps_per_candidate == 20
    assert updated.max_candidates == 2
    with pytest.raises(ValidationError):
        config.batch_size = 2  # type: ignore[misc]


def test_probability_mapping_is_copied_read_only_and_filtered() -> None:
    prior = {1: 1.0}
    policy = SpaceGroupPolicy.sampled(probabilities=prior)
    config = make_config(space_groups=policy)
    prior[1] = 0.0
    assert config.space_groups.probabilities == {1: 1.0}
    with pytest.raises(TypeError):
        config.space_groups.probabilities[1] = 0.0  # type: ignore[index]
    with pytest.raises(ValidationError, match="positive sampling weight"):
        make_config(
            z=2,
            space_groups=SpaceGroupPolicy.sampled(probabilities={1: 1.0}),
        )


@pytest.mark.parametrize("route", ["json_text", "json_dict", "python_dict"])
def test_weighted_nested_policy_round_trip_preserves_full_prior_and_draws(
    route: str,
) -> None:
    policy = SpaceGroupPolicy.sampled(
        probabilities={1: 5.0, 2: 9.0, 3: 0.25, 4: 0.75, 14: 4.0},
        crystal_system=CrystalSystem.MONOCLINIC,
        sohncke_only=True,
    )
    config = make_config(z=2, space_groups=policy)

    restored = _round_trip_config(config, route)
    restored_policy = restored.space_groups

    assert restored_policy.probabilities == policy.probabilities
    assert restored_policy.crystal_system is CrystalSystem.MONOCLINIC
    assert restored_policy.sohncke_only is True
    assert set(restored_policy.draw(128, num_operations=2, seed=432).tolist()) == {
        3,
        4,
    }
    assert torch.equal(
        restored_policy.draw(128, num_operations=2, seed=432),
        policy.draw(128, num_operations=2, seed=432),
    )
    with pytest.raises(TypeError):
        restored_policy.probabilities[3] = 0.0  # type: ignore[index]


def test_fixed_and_default_policies_round_trip_in_nested_config() -> None:
    default = make_config()
    default_restored = _round_trip_config(default, "python_dict")
    assert default_restored.space_groups.mode == "sampled"
    assert default_restored.space_groups.probabilities is None

    fixed = make_config(z=2, space_groups=SpaceGroupPolicy.fixed(2))
    fixed_restored = _round_trip_config(fixed, "json_text")
    assert fixed_restored.space_groups.mode == "fixed"
    assert fixed_restored.space_groups.group == 2


@pytest.mark.parametrize(
    "field",
    [
        "step_scale",
        "cell_step_scale",
        "max_cell_strain",
        "volume_compression_scale",
    ],
)
def test_algorithmic_controls_allow_zero_and_reject_negative(field: str) -> None:
    assert getattr(make_config(**{field: 0.0}), field) == 0.0
    with pytest.raises(ValidationError, match="finite and nonnegative"):
        make_config(**{field: -0.1})


def test_policy_requires_positive_compatible_mass() -> None:
    with pytest.raises(ValidationError, match="positive sampling weight"):
        make_config(
            space_groups=SpaceGroupPolicy.sampled(
                crystal_system=CrystalSystem.CUBIC,
            )
        )


def test_effective_revalidates_probabilities_previously_filtered_by_crystal_system() -> (
    None
):
    policy = SpaceGroupPolicy.sampled(
        probabilities={2: 1.0, 3: 1.0},
        crystal_system=CrystalSystem.TRICLINIC,
    )
    config = make_config(
        z=2,
        space_groups=policy,
    )
    monoclinic_policy = SpaceGroupPolicy.sampled(
        probabilities=config.space_groups.probabilities,
        crystal_system=CrystalSystem.MONOCLINIC,
    )
    monoclinic = config.effective(space_groups=monoclinic_policy)
    assert monoclinic.space_groups.probabilities == {2: 1.0, 3: 1.0}
    assert monoclinic.space_groups.crystal_system is CrystalSystem.MONOCLINIC
    assert config.space_groups is policy
    assert config.space_groups.crystal_system is CrystalSystem.TRICLINIC
    assert config.space_groups.probabilities == {2: 1.0, 3: 1.0}
    fixed = config.effective(space_groups=SpaceGroupPolicy.fixed(2))
    assert fixed.space_groups.group == 2
    assert config.space_groups is policy


def test_one_policy_can_validate_against_multiple_operation_counts() -> None:
    policy = SpaceGroupPolicy.sampled(probabilities={2: 1.0, 3: 1.0, 14: 3.0, 19: 1.0})
    two_operations = make_config(z=2, z_prime=1, space_groups=policy)
    four_operations = make_config(z=4, z_prime=1, space_groups=policy)
    assert two_operations.space_groups is policy
    assert four_operations.space_groups is policy
    assert two_operations.space_groups.probabilities == {
        2: 1.0,
        3: 1.0,
        14: 3.0,
        19: 1.0,
    }
    assert (
        four_operations.space_groups.probabilities
        == two_operations.space_groups.probabilities
    )


def test_fixed_policy_checks_operation_count_with_config_context() -> None:
    valid = make_config(z=2, space_groups=SpaceGroupPolicy.fixed(2))
    assert valid.space_groups.group == 2
    with pytest.raises(
        ValidationError, match=r"group 2 has 2 operations.*z=1, z_prime=1"
    ):
        make_config(space_groups=SpaceGroupPolicy.fixed(2))


def test_space_groups_rejects_explicit_none() -> None:
    with pytest.raises(ValidationError):
        make_config(space_groups=None)


def test_result_len_is_accepted_structure_count_and_records_are_frozen() -> None:
    structures = make_structures()
    report = PackingReport(
        rank=0,
        requested_count=3,
        accepted_count=2,
        generated_count=4,
        stop_reason=PackingStopReason.CANDIDATE_BUDGET_EXHAUSTED.value,
    )
    result = PackingResult(
        structures=structures,
        run_id=0,
        reports=(report,),
    )
    assert len(result) == 2
    assert result.accepted_count == 2
    assert result.requested_count == 3
    assert not result.complete
    assert result.generated_count == 4
    assert result.stop_reason is PackingStopReason.CANDIDATE_BUDGET_EXHAUSTED
    with pytest.raises(FrozenInstanceError):
        report.rank = 1  # type: ignore[misc]

    progress = OverlapReliefProgress(4, 2, 2, 1, 0, 1, 10, 0.25, 0.1)
    assert (progress.rank, progress.world_size) == (0, 1)


@pytest.mark.parametrize(
    "field, value",
    [
        ("rank", True),
        ("requested_count", -1),
        ("accepted_count", 2),
        ("generated_count", False),
        ("stop_reason", ""),
    ],
)
def test_packing_report_rejects_invalid_counters_and_reason(
    field: str, value: object
) -> None:
    values: dict[str, object] = {
        "rank": 0,
        "requested_count": 1,
        "accepted_count": 1,
        "stop_reason": PackingStopReason.TARGET_REACHED.value,
        "generated_count": 1,
    }
    values[field] = value
    with pytest.raises((TypeError, ValueError)):
        PackingReport(**values)  # type: ignore[arg-type]


def test_gathered_result_derives_counts_and_shortfall_from_reports() -> None:
    result = PackingResult(
        structures=make_structures(),
        run_id=7,
        reports=(
            PackingReport(0, 1, 1, "target_reached", 1),
            PackingReport(1, 2, 1, "acceptance_limit", None),
        ),
        scope="gathered",
    )
    assert result.accepted_count == 2
    assert result.requested_count == 3
    assert not result.complete
    assert result.generated_count is None
    assert result.stop_reason is PackingStopReason.SHORTFALL

    budget_limited = PackingResult(
        structures=make_structures(),
        run_id=7,
        reports=(
            PackingReport(0, 2, 1, "candidate_budget_exhausted", 2),
            PackingReport(1, 2, 1, "candidate_budget_exhausted", 2),
        ),
        scope="gathered",
    )
    assert budget_limited.generated_count == 4
    assert budget_limited.stop_reason is PackingStopReason.CANDIDATE_BUDGET_EXHAUSTED


def test_result_accepts_toolkit_batch_payload_and_uses_occupied_graph_count() -> None:
    structures = Batch.empty(num_systems=0, num_nodes=0, num_edges=0)
    result = PackingResult(
        structures=structures,
        run_id=4,
        reports=(PackingReport(0, 0, 0, "target_reached", 0),),
    )
    assert len(result) == result.accepted_count == 0
    assert result.complete


def test_context_builds_rank_strided_ids_and_checks_int64_overflow() -> None:
    context = PackingContext(run_id=17, rank=2, world_size=3)
    assert context.structure_ids(3, device="cpu").tolist() == [
        [17, 2],
        [17, 5],
        [17, 8],
    ]
    assert context.structure_ids(0, device="cpu").shape == (0, 2)
    with pytest.raises(OverflowError, match="int64"):
        PackingContext(run_id=1, rank=0, world_size=2**63).structure_ids(
            2, device="cpu"
        )
    with pytest.raises((TypeError, ValueError)):
        PackingContext(run_id=True)


def test_local_packer_implements_structural_protocols() -> None:
    packer = OverlapReliefPacker(make_config(), device="cpu")
    assert isinstance(packer, CrystalPacker)
    assert isinstance(packer, CandidateBudgetPacker)
