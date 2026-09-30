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
    PackingConfig,
    PackingProgress,
    PackingResult,
    PackingStopReason,
)
from nvalchemi.csp.symmetry import CrystalSystem


def make_config(**overrides: object) -> PackingConfig:
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
    return PackingConfig(**values)


def make_structures() -> RigidMoleculeASUBatch:
    packing_input = MolecularPackingInput(
        conformer_positions=torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32),
        conformer_ptr=torch.tensor([0, 1], dtype=torch.int32),
        molecule_conformer_ptr=torch.tensor([0, 1], dtype=torch.int32),
        molecule_atom_ptr=torch.tensor([0, 1], dtype=torch.int32),
        atomic_numbers=torch.tensor([6], dtype=torch.int64),
        contact_distances=torch.tensor([[1.0]], dtype=torch.float32),
        component_index=torch.tensor([0], dtype=torch.int32),
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
        PackingConfig(z=1, z_prime=1, batch_size=1)
    with pytest.raises(ValidationError, match="exactly one"):
        PackingConfig(
            z=1,
            z_prime=1,
            batch_size=4,
            cell_volume_range=(10.0, 20.0),
            cell_volume_scale_range=(0.8, 1.2),
        )
    assert make_config(cell_volume_range=(10.0, 20.0)).cell_volume_scale_range is None


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
        ({"fixed_space_group": 2}, "operation count"),
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
        assert PackingConfig.model_validate(serialized).max_candidates == budget


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
    config = make_config(space_group_probabilities=prior)
    prior[1] = 0.0
    assert config.space_group_probabilities == {1: 1.0}
    with pytest.raises(TypeError):
        config.space_group_probabilities[1] = 0.0  # type: ignore[index]
    with pytest.raises(ValidationError, match="positive weight"):
        make_config(
            z=2,
            space_group_probabilities={1: 1.0},
            crystal_system=CrystalSystem.MONOCLINIC,
        )


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


def test_default_prior_requires_positive_surviving_mass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "nvalchemi.csp.packer.config.csd_space_group_probabilities",
        lambda: {230: 1.0},
    )
    with pytest.raises(ValidationError, match="bundled CSD.*no positive weight"):
        make_config()


def test_effective_revalidates_probabilities_previously_filtered_by_crystal_system() -> (
    None
):
    config = make_config(
        z=2,
        space_group_probabilities={2: 1.0, 3: 1.0},
        crystal_system=CrystalSystem.TRICLINIC,
    )
    monoclinic = config.effective(crystal_system=CrystalSystem.MONOCLINIC)
    assert monoclinic.space_group_probabilities == {2: 1.0, 3: 1.0}
    assert monoclinic.crystal_system is CrystalSystem.MONOCLINIC


def test_fixed_group_checks_operation_count_and_sample_filters_are_exclusive() -> None:
    valid = make_config(z=2, fixed_space_group=2)
    assert valid.fixed_space_group == 2
    with pytest.raises(ValidationError, match="cannot be combined"):
        make_config(fixed_space_group=1, crystal_system=CrystalSystem.CUBIC)


def test_result_len_is_accepted_structure_count_and_records_are_frozen() -> None:
    structures = make_structures()
    result = PackingResult(
        structures=structures,
        generated_count=4,
        stop_reason=PackingStopReason.CANDIDATE_BUDGET_EXHAUSTED,
        run_id=0,
    )
    assert len(result) == 2
    assert result.generated_count == 4
    assert result.stop_reason is PackingStopReason.CANDIDATE_BUDGET_EXHAUSTED
    with pytest.raises(FrozenInstanceError):
        result.generated_count = 1  # type: ignore[misc]

    progress = PackingProgress(4, 2, 2, 1, 0, 1, 10, 0.25, 0.1)
    assert (progress.rank, progress.world_size) == (0, 1)
