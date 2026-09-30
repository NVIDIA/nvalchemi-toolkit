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

"""Public packing behavior and pinned CPU numerical fixtures."""

from __future__ import annotations

from math import sqrt

import pytest
import torch
import warp as wp

from nvalchemi.csp._space_group_tables import SG_OPS_IDX, SG_OPS_PTR, SYMM_OPS
from nvalchemi.csp.data import MolecularPackingInput
from nvalchemi.csp.packer import CrystalPacker, PackingConfig, PackingStopReason
from nvalchemi.csp.packer._contacts import (
    ContactEvaluation,
    ContactWorkspace,
    contact_forces,
    make_contact_maps,
)
from nvalchemi.csp.packer._state import WorkingState, relax_step
from nvalchemi.csp.packer.result import PackingResult
from nvalchemi.csp.symmetry import get_space_group_operations
from test.csp.fixtures.contact_oracle import contact_observables
from test.csp.fixtures.packer_stage2_cpu import SOURCE_REFERENCE


def _one_atom_input(contact_distance: float = 1.0) -> MolecularPackingInput:
    return MolecularPackingInput(
        conformer_positions=torch.zeros((1, 3), dtype=torch.float32),
        conformer_ptr=torch.tensor([0, 1], dtype=torch.int32),
        molecule_conformer_ptr=torch.tensor([0, 1], dtype=torch.int32),
        molecule_atom_ptr=torch.tensor([0, 1], dtype=torch.int32),
        atomic_numbers=torch.tensor([6], dtype=torch.int64),
        contact_distances=torch.tensor([[contact_distance]], dtype=torch.float32),
        component_index=torch.tensor([0], dtype=torch.int32),
        formula_unit_volume=1000.0,
    )


def _fixed_multimolecule_state() -> tuple[dict[str, torch.Tensor], int]:
    conformer_positions = torch.tensor(
        [[-0.45, 0.0, 0.0], [0.45, 0.0, 0.0], [0.0, 0.0, 0.0]],
        dtype=torch.float32,
    )
    conformer_ptr = torch.tensor([0, 2, 3], dtype=torch.int32)
    conformer_ids = torch.tensor([[0, 1]], dtype=torch.int32)
    molecule_atom_ptr = torch.tensor([0, 2, 3], dtype=torch.int32)
    centers = torch.tensor(
        [[[0.11, 0.09, 0.12], [0.87, 0.10, 0.11]]], dtype=torch.float32
    )
    angle = torch.tensor(0.37)
    rotation_first = torch.stack(
        (
            torch.stack((torch.cos(angle), -torch.sin(angle), torch.tensor(0.0))),
            torch.stack((torch.sin(angle), torch.cos(angle), torch.tensor(0.0))),
            torch.tensor([0.0, 0.0, 1.0]),
        )
    )
    rotations = (
        torch.stack((rotation_first, torch.eye(3))).reshape(1, 2, 3, 3).contiguous()
    )
    cells = torch.tensor(
        [[[4.4, 0.0, 0.0], [1.0, 4.2, 0.0], [0.5, 0.4, 4.6]]],
        dtype=torch.float32,
    )
    inverse_cells = torch.linalg.inv(cells).contiguous()
    symmetry_table = torch.tensor(
        [
            [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
            [-1, 0, 0, 0, -1, 0, 0, 0, -1, 0.5, 0.5, 0.5],
        ],
        dtype=torch.float32,
    )
    selected_ops = torch.tensor([[0, 1]], dtype=torch.int32)
    contact_distances = torch.tensor(
        [[1.8, 2.2, 2.4], [2.2, 1.8, 2.4], [2.4, 2.4, 2.0]],
        dtype=torch.float32,
    )
    return (
        {
            "conformer_positions": conformer_positions,
            "conformer_ptr": conformer_ptr,
            "conformer_ids": conformer_ids,
            "molecule_atom_ptr": molecule_atom_ptr,
            "centers": centers,
            "rotations": rotations,
            "cells": cells,
            "inverse_cells": inverse_cells,
            "symmetry_table": symmetry_table,
            "selected_ops": selected_ops,
            "contact_distances": contact_distances,
        },
        2,
    )


def _evaluate_fixed_contacts(state: dict[str, torch.Tensor], operation_count: int):
    maps = make_contact_maps(state["molecule_atom_ptr"], operation_count)
    expanded_atom_count = int(state["molecule_atom_ptr"][-1]) * operation_count
    workspace = ContactWorkspace.allocate(
        batch_size=1,
        num_molecules=state["conformer_ids"].shape[1],
        expanded_atom_count=expanded_atom_count,
        device=state["cells"].device,
    )
    return contact_forces(
        conformer_positions=state["conformer_positions"],
        conformer_ptr=state["conformer_ptr"],
        conformer_ids=state["conformer_ids"],
        centers=state["centers"],
        rotations=state["rotations"],
        cells=state["cells"],
        inverse_cells=state["inverse_cells"],
        symmetry_table=state["symmetry_table"],
        selected_symmetry_ops=state["selected_ops"],
        contact_distances=state["contact_distances"],
        max_contact_distance=float(state["contact_distances"].max().item()),
        maps=maps,
        workspace=workspace,
    )


def _expected_tensor(value: object) -> torch.Tensor:
    return torch.tensor(value, dtype=torch.float32)


def _to_device(state: dict[str, torch.Tensor], device: str) -> dict[str, torch.Tensor]:
    return {name: value.to(device=device) for name, value in state.items()}


def _p3_periodic_contact_oracle(
    local_atoms: torch.Tensor,
    centers: torch.Tensor,
    cell: torch.Tensor,
    cutoff: float,
) -> tuple[torch.Tensor, torch.Tensor, float, float, int]:
    """Differentiate the six designed P3 overlap energies in float64."""
    operations = get_space_group_operations(143).to(dtype=torch.float64)
    cell64 = cell.to(dtype=torch.float64)
    inverse_transpose = torch.linalg.inv(cell64).T
    centers64 = centers[0].to(dtype=torch.float64).detach().clone()
    centers64.requires_grad_()
    local64 = local_atoms.to(dtype=torch.float64).detach().clone()
    local64.requires_grad_()
    energy = torch.zeros((), dtype=torch.float64)
    overlaps: list[float] = []
    periodic_contacts = 0
    shifts = ((1, 0, 0), (0, 0, 0), (0, -1, 0))
    for symmetry_index, operation in enumerate(operations):
        operation_rotation = operation[:, :3]
        operation_translation = operation[:, 3]
        center_a = operation_rotation @ centers64[0] + operation_translation
        center_b = operation_rotation @ centers64[1] + operation_translation
        center_a = center_a - torch.floor(center_a)
        center_b = center_b - torch.floor(center_b)
        atom_b = cell64.T @ center_b
        for local in local64:
            fractional_a = center_a + operation_rotation @ (inverse_transpose @ local)
            fractional_a = fractional_a - torch.floor(fractional_a)
            atom_a = cell64.T @ fractional_a
            shift_values = shifts[symmetry_index]
            shift = torch.tensor(shift_values, dtype=torch.float64)
            vector = atom_b - atom_a + cell64.T @ shift
            distance = torch.linalg.vector_norm(vector)
            overlap = cutoff - distance
            assert overlap > 0.0
            energy = energy + 0.5 * overlap.square()
            overlaps.append(float(overlap.detach()))
            periodic_contacts += int(any(shift_values))

    center_gradient, local_gradient = torch.autograd.grad(energy, (centers64, local64))
    # The center variables are fractional columns, with Cartesian centers
    # x = cell.T @ f. Thus grad_x = cell^-1 @ grad_f for this row-cell layout.
    expected_forces = -torch.linalg.solve(cell64, center_gradient.T).T
    local_forces = -local_gradient
    expected_torques = torch.zeros((2, 3), dtype=torch.float64)
    expected_torques[0] = torch.linalg.cross(local64, local_forces, dim=-1).sum(dim=0)
    return (
        expected_forces,
        expected_torques,
        sum(overlaps),
        max(overlaps),
        periodic_contacts,
    )


def test_cpu_fixed_state_contacts_match_pinned_source_fixtures() -> None:
    atol = SOURCE_REFERENCE["tolerances"]["contact_atol"]
    rtol = SOURCE_REFERENCE["tolerances"]["contact_rtol"]
    state, operation_count = _fixed_multimolecule_state()
    multiop = _evaluate_fixed_contacts(state, operation_count)
    for name in ("forces", "torques", "virial", "total_overlap", "max_overlap"):
        torch.testing.assert_close(
            getattr(multiop, name),
            _expected_tensor(SOURCE_REFERENCE["multiop_skew"]["contacts"][name]),
            atol=atol,
            rtol=rtol,
        )
    p1_state = dict(state)
    p1_state["selected_ops"] = torch.tensor([[0]], dtype=torch.int32)
    p1 = _evaluate_fixed_contacts(p1_state, 1)
    for name in ("forces", "torques", "virial", "total_overlap", "max_overlap"):
        torch.testing.assert_close(
            getattr(p1, name),
            _expected_tensor(SOURCE_REFERENCE["p1_skew"]["contacts"][name]),
            atol=atol,
            rtol=rtol,
        )


def test_cpu_skew_cell_contacts_match_complete_periodic_oracle() -> None:
    cell = torch.tensor(
        [[3.0, 0.0, 0.0], [1.5, 2.8, 0.0], [0.3, 0.4, 3.2]],
        dtype=torch.float32,
    )
    fractional_centers = torch.tensor(
        [[[0.13, 0.21, 0.31], [0.61, 0.68, 0.77]]], dtype=torch.float32
    )
    contact_distances = torch.full((2, 2), 3.25, dtype=torch.float32)
    state = {
        "conformer_positions": torch.zeros((2, 3), dtype=torch.float32),
        "conformer_ptr": torch.tensor([0, 1, 2], dtype=torch.int32),
        "conformer_ids": torch.tensor([[0, 1]], dtype=torch.int32),
        "molecule_atom_ptr": torch.tensor([0, 1, 2], dtype=torch.int32),
        "centers": fractional_centers,
        "rotations": torch.eye(3).expand(1, 2, 3, 3).clone(),
        "cells": cell.unsqueeze(0),
        "inverse_cells": torch.linalg.inv(cell).unsqueeze(0).contiguous(),
        "symmetry_table": torch.tensor(
            [[1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0]], dtype=torch.float32
        ),
        "selected_ops": torch.tensor([[0]], dtype=torch.int32),
        "contact_distances": contact_distances,
    }
    actual = _evaluate_fixed_contacts(state, operation_count=1)
    positions = fractional_centers[0] @ cell
    count, expected_forces, expected_virial, expected_total, expected_max = (
        contact_observables(
            positions,
            cell,
            torch.tensor([0, 1]),
            contact_distances,
            expanded_atom_count=2,
        )
    )

    assert count == 15
    torch.testing.assert_close(expected_virial, expected_virial.T)
    torch.testing.assert_close(
        actual.forces[0].double(), expected_forces, atol=5e-7, rtol=0
    )
    torch.testing.assert_close(
        actual.virial[0].double(), expected_virial, atol=5e-7, rtol=0
    )
    assert actual.total_overlap.item() == pytest.approx(expected_total, abs=1e-6)
    assert actual.max_overlap.item() == pytest.approx(expected_max, abs=5e-7)
    torch.testing.assert_close(actual.forces.sum(dim=1), torch.zeros((1, 3)))


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda:0",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="requires cuda:0"
            ),
        ),
    ],
)
def test_p3_periodic_force_and_torque_match_hexagonal_oracle(device: str) -> None:
    positions = torch.tensor(
        [[0.0, 0.0, -0.4], [0.0, 0.0, 0.4], [0.0, 0.0, 0.0]],
        dtype=torch.float32,
    )
    conformer_ptr = torch.tensor([0, 2, 3], dtype=torch.int32)
    conformer_ids = torch.tensor([[0, 1]], dtype=torch.int32)
    molecule_atom_ptr = torch.tensor([0, 2, 3], dtype=torch.int32)
    centers = torch.tensor([[[0.98, 0.3, 0.5], [0.02, 0.3, 0.58]]])
    rotations = torch.eye(3).repeat(1, 2, 1, 1)
    cell = torch.tensor(
        [[8.0, 0.0, 0.0], [-4.0, 4.0 * sqrt(3.0), 0.0], [0.0, 0.0, 12.0]],
        dtype=torch.float32,
    )
    cells = cell.unsqueeze(0).to(device)
    inverse_cells = torch.linalg.inv(cells).contiguous()
    operation_start = int(SG_OPS_PTR[142])
    operation_stop = int(SG_OPS_PTR[143])
    selected_ops = torch.tensor(
        [SG_OPS_IDX[operation_start:operation_stop].tolist()], dtype=torch.int32
    ).to(device)
    contact_distances = torch.full((3, 3), 2.0, dtype=torch.float32, device=device)
    maps = make_contact_maps(molecule_atom_ptr.to(device), 3)
    workspace = ContactWorkspace.allocate(
        batch_size=1,
        num_molecules=2,
        expanded_atom_count=9,
        device=torch.device(device),
    )
    actual = contact_forces(
        conformer_positions=positions.to(device),
        conformer_ptr=conformer_ptr.to(device),
        conformer_ids=conformer_ids.to(device),
        centers=centers.to(device),
        rotations=rotations.to(device),
        cells=cells,
        inverse_cells=inverse_cells,
        symmetry_table=torch.tensor(
            SYMM_OPS.copy(), dtype=torch.float32, device=device
        ),
        selected_symmetry_ops=selected_ops,
        contact_distances=contact_distances,
        max_contact_distance=2.0,
        maps=maps,
        workspace=workspace,
    )
    expected_forces, expected_torques, total, maximum, periodic_count = (
        _p3_periodic_contact_oracle(positions[:2], centers, cell, 2.0)
    )

    assert periodic_count > 0
    assert expected_forces.abs().max() > 0.1
    assert expected_torques.abs().max() > 0.01
    torch.testing.assert_close(
        actual.forces[0].cpu(),
        expected_forces.to(dtype=torch.float32),
        atol=2e-5,
        rtol=2e-5,
    )
    torch.testing.assert_close(
        actual.torques[0].cpu(),
        expected_torques.to(dtype=torch.float32),
        atol=2e-5,
        rtol=2e-5,
    )
    torch.testing.assert_close(actual.total_overlap.cpu(), torch.tensor([total]))
    torch.testing.assert_close(actual.max_overlap.cpu(), torch.tensor([maximum]))


def test_cpu_one_rigid_and_cell_step_matches_pinned_source_fixture() -> None:
    state_values, operation_count = _fixed_multimolecule_state()
    contacts = _evaluate_fixed_contacts(state_values, operation_count)
    state = WorkingState.allocate(
        batch_size=1,
        molecule_count=2,
        symmetry_operation_count=2,
        device=torch.device("cpu"),
    )
    state.conformer_ids.copy_(state_values["conformer_ids"])
    state.centers.copy_(state_values["centers"])
    state.rotations.copy_(state_values["rotations"])
    state.cells.copy_(state_values["cells"])
    state.inverse_cells.copy_(state_values["inverse_cells"])
    state.reference_volumes.fill_(torch.linalg.det(state_values["cells"]).item())
    state.space_groups.fill_(2)
    state.active.fill_(True)
    relax_step(
        state=state,
        conformer_positions=state_values["conformer_positions"],
        conformer_ptr=state_values["conformer_ptr"],
        molecule_atom_ptr=state_values["molecule_atom_ptr"],
        forces=contacts.forces,
        torques=contacts.torques,
        virial=contacts.virial,
        max_overlap=contacts.max_overlap,
        expanded_atom_count=6,
        step_scale=0.12,
        max_step=0.3,
        cell_step_scale=0.01,
        max_cell_strain=0.003,
        volume_compression_scale=1.0e-5,
    )
    expected = SOURCE_REFERENCE["multiop_skew"]["one_step"]
    atol = SOURCE_REFERENCE["tolerances"]["step_atol"]
    rtol = SOURCE_REFERENCE["tolerances"]["step_rtol"]
    for name, actual in (
        ("cells", state.cells),
        ("inverse_cells", state.inverse_cells),
        ("centers", state.centers),
        ("rotations", state.rotations),
    ):
        torch.testing.assert_close(
            actual, _expected_tensor(expected[name]), atol=atol, rtol=rtol
        )
    torch.testing.assert_close(
        state.steps, torch.tensor(expected["steps"], dtype=torch.int32)
    )


@pytest.mark.parametrize("device", ["cuda:0", "cuda:1"])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_cuda_fixed_state_contact_and_one_step_match_source_fixture(
    device: str,
) -> None:
    values, operation_count = _fixed_multimolecule_state()
    state_values = _to_device(values, device)
    contacts = _evaluate_fixed_contacts(state_values, operation_count)
    expected_contacts = SOURCE_REFERENCE["multiop_skew"]["contacts"]
    contact_atol = SOURCE_REFERENCE["tolerances"]["contact_atol"]
    contact_rtol = SOURCE_REFERENCE["tolerances"]["contact_rtol"]
    for name in ("forces", "torques", "virial", "total_overlap", "max_overlap"):
        torch.testing.assert_close(
            getattr(contacts, name),
            _expected_tensor(expected_contacts[name]).to(device),
            atol=contact_atol,
            rtol=contact_rtol,
        )

    state = WorkingState.allocate(
        batch_size=1,
        molecule_count=2,
        symmetry_operation_count=2,
        device=torch.device(device),
    )
    state.conformer_ids.copy_(state_values["conformer_ids"])
    state.centers.copy_(state_values["centers"])
    state.rotations.copy_(state_values["rotations"])
    state.cells.copy_(state_values["cells"])
    state.inverse_cells.copy_(state_values["inverse_cells"])
    state.reference_volumes.fill_(torch.linalg.det(state_values["cells"]).item())
    state.space_groups.fill_(2)
    state.active.fill_(True)
    relax_step(
        state=state,
        conformer_positions=state_values["conformer_positions"],
        conformer_ptr=state_values["conformer_ptr"],
        molecule_atom_ptr=state_values["molecule_atom_ptr"],
        forces=contacts.forces,
        torques=contacts.torques,
        virial=contacts.virial,
        max_overlap=contacts.max_overlap,
        expanded_atom_count=6,
        step_scale=0.12,
        max_step=0.3,
        cell_step_scale=0.01,
        max_cell_strain=0.003,
        volume_compression_scale=1.0e-5,
    )
    expected_step = SOURCE_REFERENCE["multiop_skew"]["one_step"]
    step_atol = SOURCE_REFERENCE["tolerances"]["step_atol"]
    step_rtol = SOURCE_REFERENCE["tolerances"]["step_rtol"]
    for name, actual in (
        ("cells", state.cells),
        ("inverse_cells", state.inverse_cells),
        ("centers", state.centers),
        ("rotations", state.rotations),
    ):
        torch.testing.assert_close(
            actual,
            _expected_tensor(expected_step[name]).to(device),
            atol=step_atol,
            rtol=step_rtol,
        )
    torch.testing.assert_close(
        state.steps,
        torch.tensor(expected_step["steps"], dtype=torch.int32, device=device),
    )


def test_public_cpu_packing_has_compact_z_prime_and_one_seed_draw() -> None:
    inputs = _one_atom_input()
    config = PackingConfig(
        z=2,
        z_prime=2,
        batch_size=4,
        max_candidates=4,
        cell_volume_range=(10_000.0, 10_000.0),
        fixed_space_group=1,
    )
    generator = torch.Generator(device="cpu").manual_seed(147)
    expected_generator = torch.Generator(device="cpu").manual_seed(147)
    torch.randint(0, 2**31 - 1, (), dtype=torch.int64, generator=expected_generator)
    events = []
    result = CrystalPacker(config, device="cpu")(
        inputs,
        num_samples=4,
        rng=generator,
        progress_callback=events.append,
    )
    assert len(result) == 4
    assert result.generated_count == 4
    assert result.stop_reason is PackingStopReason.TARGET_REACHED
    assert 0 <= result.run_id < 2**63
    torch.testing.assert_close(
        result.structures.structure_ids,
        torch.stack(
            (
                torch.full((4,), result.run_id, dtype=torch.int64),
                torch.arange(4, dtype=torch.int64),
            ),
            dim=1,
        ),
    )
    assert result.structures.packing_input is inputs
    assert result.structures.conformer_indices.shape == (8,)
    assert result.structures.rotations.shape == (8, 3, 3)
    assert result.structures.fractional_centers.shape == (8, 3)
    assert result.structures.structure_molecule_ptr.tolist() == [0, 2, 4, 6, 8]
    assert torch.equal(generator.get_state(), expected_generator.get_state())
    assert len(events) == 1
    assert events[0].generated_count == 4
    assert events[0].accepted_count == 4
    assert events[0].active_count == 4
    assert events[0].converged_count == 4
    assert events[0].iteration == 0
    assert events[0].rank == 0 and events[0].world_size == 1
    assert torch.all(result.structures.properties["steps"] == 0)
    assert torch.all(result.structures.properties["max_overlap"] == 0)
    assert not torch.equal(
        result.structures.fractional_centers[0],
        result.structures.fractional_centers[1],
    )


def test_public_cpu_packing_preserves_explicit_run_id() -> None:
    result = CrystalPacker(
        PackingConfig(
            z=1,
            z_prime=1,
            batch_size=2,
            max_candidates=2,
            cell_volume_range=(10_000.0, 10_000.0),
            fixed_space_group=1,
        ),
        device="cpu",
    ).pack(
        _one_atom_input(),
        num_samples=2,
        rng=torch.Generator().manual_seed(8),
        run_id=17,
    )

    assert result.run_id == 17
    assert result.structures.structure_ids.tolist() == [[17, 0], [17, 1]]


@pytest.mark.parametrize(
    ("run_id", "error"),
    [(True, TypeError), (1.5, TypeError), (-1, ValueError), (2**63, ValueError)],
)
def test_packer_rejects_invalid_run_id(run_id: object, error: type[Exception]) -> None:
    packer = CrystalPacker(
        PackingConfig(
            z=1,
            z_prime=1,
            batch_size=1,
            cell_volume_range=(10_000.0, 10_000.0),
        ),
        device="cpu",
    )
    with pytest.raises(error, match="run_id"):
        packer.pack(_one_atom_input(), run_id=run_id)  # type: ignore[arg-type]


def test_public_cpu_partial_candidate_budget_returns_compact_shortfall() -> None:
    inputs = _one_atom_input()
    config = PackingConfig(
        z=1,
        z_prime=1,
        batch_size=4,
        max_candidates=2,
        cell_volume_range=(10_000.0, 10_000.0),
        fixed_space_group=1,
    )
    result = CrystalPacker(config, device="cpu")(
        inputs,
        num_samples=4,
        rng=torch.Generator().manual_seed(1),
    )
    assert len(result) == 2
    assert result.generated_count == 2
    assert result.stop_reason is PackingStopReason.CANDIDATE_BUDGET_EXHAUSTED
    assert result.structures.cells.device.type == "cpu"


def test_public_cpu_expiration_partially_refills_and_returns_empty_shortfall() -> None:
    inputs = _one_atom_input(contact_distance=5.0)
    config = PackingConfig(
        z=1,
        z_prime=1,
        batch_size=2,
        max_candidates=3,
        max_steps_per_candidate=1,
        convergence_check_interval=1,
        overlap_tolerance=0.0,
        step_scale=0.0,
        cell_step_scale=0.0,
        cell_volume_range=(90.0, 90.0),
        fixed_space_group=1,
    )
    events = []
    result = CrystalPacker(config, device="cpu")(
        inputs,
        num_samples=1,
        rng=torch.Generator().manual_seed(87),
        progress_callback=events.append,
    )
    assert len(result) == 0
    assert result.generated_count == 3
    assert result.stop_reason is PackingStopReason.CANDIDATE_BUDGET_EXHAUSTED
    assert result.structures.cells.device.type == "cpu"
    assert result.structures.structure_molecule_ptr.tolist() == [0]
    assert len(events) == 4
    assert [event.iteration for event in events] == [0, 1, 2, 3]
    assert events[1].generated_count == 3
    assert events[1].expired_count == 2
    assert events[1].replaced_count == 2
    assert events[2].active_count == 1
    assert events[2].expired_count == 0
    assert events[2].generated_count == 3
    assert events[3].expired_count == 1
    assert events[3].generated_count == 3


@pytest.mark.parametrize("num_samples", [1, 2])
def test_public_cpu_default_auto_budget_bounds_unreachable_acceptance(
    num_samples: int,
) -> None:
    config = PackingConfig(
        z=1,
        z_prime=1,
        batch_size=64,
        max_steps_per_candidate=1,
        convergence_check_interval=1,
        overlap_tolerance=0.0,
        step_scale=0.0,
        cell_step_scale=0.0,
        volume_compression_scale=0.0,
        cell_volume_range=(125.0, 125.0),
        fixed_space_group=1,
    )
    result = CrystalPacker(config, device="cpu")(
        _one_atom_input(contact_distance=7.0),
        num_samples=num_samples,
        rng=torch.Generator().manual_seed(87),
    )

    assert config.max_candidates == "auto"
    assert len(result) == 0
    assert result.generated_count == 1000 * num_samples
    assert result.stop_reason is PackingStopReason.CANDIDATE_BUDGET_EXHAUSTED


def test_refilled_candidate_skips_relaxation_with_stale_nonzero_force(
    monkeypatch,
) -> None:
    inputs = _one_atom_input()
    config = PackingConfig(
        z=1,
        z_prime=1,
        batch_size=1,
        max_candidates=2,
        max_steps_per_candidate=5,
        convergence_check_interval=1,
        overlap_tolerance=0.05,
        cell_step_scale=0.0,
        cell_volume_range=(10_000.0, 10_000.0),
        fixed_space_group=1,
    )
    calls = 0

    def synthetic_contacts(**kwargs):
        nonlocal calls
        calls += 1
        ids = kwargs["conformer_ids"]
        device = ids.device
        forces = torch.zeros((*ids.shape, 3), dtype=torch.float32, device=device)
        forces[0, 0, 0] = 1.0
        overlap = 0.01 if calls == 1 else 0.0
        return ContactEvaluation(
            forces=forces,
            torques=torch.zeros_like(forces),
            total_overlap=torch.full((ids.shape[0],), overlap, device=device),
            max_overlap=torch.full((ids.shape[0],), overlap, device=device),
            virial=torch.zeros(
                (ids.shape[0], 3, 3), dtype=torch.float32, device=device
            ),
        )

    monkeypatch.setattr(
        "nvalchemi.csp.packer.engine.contact_forces", synthetic_contacts
    )
    result = CrystalPacker(config, device="cpu")(
        inputs, num_samples=2, rng=torch.Generator().manual_seed(101)
    )

    assert calls == 2
    assert len(result) == 2
    assert result.structures.properties["steps"].tolist() == [0, 0]


@pytest.mark.parametrize(
    ("check_interval", "step_limit", "expected_iterations"),
    [(10, 1, [0, 1]), (2, 3, [0, 2, 3])],
)
def test_progress_callback_reports_exact_expiration_checks(
    check_interval: int, step_limit: int, expected_iterations: list[int]
) -> None:
    inputs = _one_atom_input(contact_distance=5.0)
    config = PackingConfig(
        z=1,
        z_prime=1,
        batch_size=2,
        max_candidates=2,
        max_steps_per_candidate=step_limit,
        convergence_check_interval=check_interval,
        overlap_tolerance=0.0,
        step_scale=0.0,
        cell_step_scale=0.0,
        cell_volume_range=(90.0, 90.0),
        fixed_space_group=1,
    )
    events = []
    result = CrystalPacker(config, device="cpu")(
        inputs,
        num_samples=1,
        rng=torch.Generator().manual_seed(88),
        progress_callback=events.append,
    )
    assert len(result) == 0
    assert result.stop_reason is PackingStopReason.CANDIDATE_BUDGET_EXHAUSTED
    assert [event.iteration for event in events] == expected_iterations
    assert events[0].active_count == 2
    assert events[0].expired_count == 0
    terminal_event = events[-1]
    assert terminal_event.active_count == 2
    assert terminal_event.expired_count == 2
    assert terminal_event.replaced_count == 2
    assert terminal_event.generated_count == 2


def test_budget_exhaustion_recomputes_expiration_for_surviving_row() -> None:
    inputs = MolecularPackingInput(
        conformer_positions=torch.zeros((2, 3), dtype=torch.float32),
        conformer_ptr=torch.tensor([0, 1, 2], dtype=torch.int32),
        molecule_conformer_ptr=torch.tensor([0, 1, 2], dtype=torch.int32),
        molecule_atom_ptr=torch.tensor([0, 1, 2], dtype=torch.int32),
        atomic_numbers=torch.tensor([6, 6], dtype=torch.int64),
        contact_distances=torch.full((2, 2), 2.0, dtype=torch.float32),
        component_index=torch.tensor([0, 1], dtype=torch.int32),
        formula_unit_volume=100.0,
    )
    config = PackingConfig(
        z=2,
        z_prime=2,
        batch_size=2,
        max_candidates=3,
        max_steps_per_candidate=10,
        convergence_check_interval=3,
        overlap_tolerance=0.05,
        step_scale=0.12,
        cell_step_scale=0.0,
        volume_compression_scale=0.0,
        cell_volume_range=(150.0, 150.0),
        fixed_space_group=1,
    )
    events = []
    result = CrystalPacker(config, device="cpu")(
        inputs,
        num_samples=4,
        rng=torch.Generator().manual_seed(2),
        progress_callback=events.append,
    )

    assert result.generated_count == 3
    assert len(result) == 2
    assert result.stop_reason is PackingStopReason.CANDIDATE_BUDGET_EXHAUSTED
    assert [event.iteration for event in events] == [0, 3, 6, 9, 10, 12, 15, 17]
    assert events[2].converged_count == 1
    assert events[4].expired_count == 1
    assert events[4].replaced_count == 1
    assert events[4].generated_count == 3
    assert events[-1].expired_count == 1


def test_public_cpu_same_seed_repeats_and_different_seed_changes_geometry() -> None:
    inputs = _one_atom_input()
    config = PackingConfig(
        z=1,
        z_prime=1,
        batch_size=4,
        max_candidates=4,
        cell_volume_range=(10_000.0, 10_000.0),
        fixed_space_group=1,
    )
    packer = CrystalPacker(config, device="cpu")
    results = [
        packer(inputs, num_samples=4, rng=torch.Generator().manual_seed(seed))
        for seed in (22, 22, 23)
    ]
    for name in (
        "conformer_indices",
        "rotations",
        "fractional_centers",
        "cells",
        "space_groups",
    ):
        first = getattr(results[0].structures, name)
        torch.testing.assert_close(
            first, getattr(results[1].structures, name), atol=0, rtol=0
        )
    assert not torch.equal(
        results[0].structures.fractional_centers,
        results[2].structures.fractional_centers,
    )


def test_public_space_group_weights_are_scale_invariant_and_exclude_zeros() -> None:
    inputs = _one_atom_input()

    def sample_groups(weights: dict[int, float]) -> torch.Tensor:
        config = PackingConfig(
            z=3,
            z_prime=1,
            batch_size=24,
            max_candidates=24,
            cell_volume_range=(1_000_000.0, 1_000_000.0),
            space_group_probabilities=weights,
        )
        result = CrystalPacker(config, device="cpu")(
            inputs,
            num_samples=24,
            rng=torch.Generator().manual_seed(925),
        )
        assert len(result) == 24
        return result.structures.space_groups

    large_weights = sample_groups({143: 1.0e300, 144: 0.0, 145: 3.0e300})
    small_weights = sample_groups({143: 1.0e-300, 144: 0.0, 145: 3.0e-300})
    overflowing_sum_weights = sample_groups({143: 5.0e307, 144: 0.0, 145: 1.5e308})
    assert torch.equal(large_weights, small_weights)
    assert torch.equal(large_weights, overflowing_sum_weights)
    assert torch.all((large_weights == 143) | (large_weights == 145))


def test_public_cpu_conformer_pools_and_geometry_invariants() -> None:
    inputs = MolecularPackingInput(
        conformer_positions=torch.zeros((4, 3), dtype=torch.float32),
        conformer_ptr=torch.tensor([0, 1, 2, 3, 4], dtype=torch.int32),
        molecule_conformer_ptr=torch.tensor([0, 2, 4], dtype=torch.int32),
        molecule_atom_ptr=torch.tensor([0, 1, 2], dtype=torch.int32),
        atomic_numbers=torch.tensor([6, 8], dtype=torch.int64),
        contact_distances=torch.full((2, 2), 1.0, dtype=torch.float32),
        component_index=torch.tensor([0, 1], dtype=torch.int32),
        formula_unit_volume=1000.0,
    )
    config = PackingConfig(
        z=1,
        z_prime=1,
        batch_size=8,
        max_candidates=8,
        cell_volume_range=(10_000.0, 10_000.0),
        fixed_space_group=1,
    )
    result = CrystalPacker(config, device="cpu")(
        inputs,
        num_samples=8,
        rng=torch.Generator().manual_seed(91),
    )
    conformers = result.structures.conformer_indices.reshape(8, 2)
    assert bool(((0 <= conformers[:, 0]) & (conformers[:, 0] < 2)).all())
    assert bool(((2 <= conformers[:, 1]) & (conformers[:, 1] < 4)).all())
    centers = result.structures.fractional_centers
    assert bool(((centers >= 0.0) & (centers < 1.0)).all())
    rotations = result.structures.rotations
    torch.testing.assert_close(
        rotations @ rotations.transpose(-1, -2),
        torch.eye(3).expand_as(rotations),
        atol=2.0e-5,
        rtol=2.0e-5,
    )
    assert bool((torch.linalg.det(rotations) > 0.9999).all())
    cells = result.structures.cells
    assert bool((cells[:, 0, 1:] == 0).all())
    assert bool((cells[:, 1, 2] == 0).all())
    assert bool((cells[:, 0, 0] > 0).all())
    assert bool((cells[:, 1, 1] > 0).all())
    assert bool((cells[:, 2, 2] > 0).all())
    torch.testing.assert_close(
        torch.linalg.det(cells),
        torch.full((8,), 10_000.0),
        atol=0.1,
        rtol=1.0e-5,
    )


def test_public_cpu_callback_exception_propagates() -> None:
    inputs = _one_atom_input()
    config = PackingConfig(
        z=1,
        z_prime=1,
        batch_size=1,
        cell_volume_range=(10_000.0, 10_000.0),
        fixed_space_group=1,
    )

    def fail(_progress) -> None:
        raise RuntimeError("callback failed")

    with pytest.raises(RuntimeError, match="callback failed"):
        CrystalPacker(config, device="cpu")(
            inputs,
            rng=torch.Generator().manual_seed(4),
            progress_callback=fail,
        )


def test_raw_atomistic_generator_returns_packing_result() -> None:
    from nvalchemi.gen.generator import AtomisticGenerator

    inputs = _one_atom_input()
    config = PackingConfig(
        z=1,
        z_prime=1,
        batch_size=1,
        cell_volume_range=(10_000.0, 10_000.0),
        fixed_space_group=1,
    )
    packer = CrystalPacker(config, device="cpu")
    generator = AtomisticGenerator(
        generator_func=packer,
        device="cpu",
        dedicated_stream=False,
    )
    result = generator.sample(inputs, rng=torch.Generator().manual_seed(31))
    assert isinstance(result, PackingResult)
    assert result.structures.packing_input is inputs


@pytest.mark.parametrize("dedicated", [False, True])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_cuda_packer_launches_on_active_torch_stream(
    monkeypatch, dedicated: bool
) -> None:
    from nvalchemi.gen.generator import AtomisticGenerator

    inputs = _one_atom_input()
    config = PackingConfig(
        z=1,
        z_prime=1,
        batch_size=1,
        cell_volume_range=(10_000.0, 10_000.0),
        fixed_space_group=1,
    )
    packer = CrystalPacker(config, device="cuda:0")
    observed_streams = []
    original_launch = wp.launch

    def check_launch_stream(*args, **kwargs):
        device = kwargs.get("device")
        if device is not None and torch.device(device).type == "cuda":
            current_torch_stream = torch.cuda.current_stream(torch.device(device))
            current_warp_stream = wp.get_stream(str(device))
            observed_streams.append(
                (current_torch_stream.cuda_stream, current_warp_stream.cuda_stream)
            )
        return original_launch(*args, **kwargs)

    monkeypatch.setattr(wp, "launch", check_launch_stream)
    rng = torch.Generator(device="cuda:0").manual_seed(19)
    if dedicated:
        generator = AtomisticGenerator(
            generator_func=packer,
            device="cuda:0",
            dedicated_stream=True,
        )
        with generator:
            stream = generator._stream
            result = generator.sample(inputs, rng=rng)
    else:
        stream = torch.cuda.Stream(device="cuda:0")
        with torch.cuda.stream(stream):
            result = packer(inputs, rng=rng)
    stream.synchronize()

    assert isinstance(result, PackingResult)
    assert len(result) == 1
    assert observed_streams
    assert all(
        torch_stream == warp_stream for torch_stream, warp_stream in observed_streams
    )
    assert all(
        torch_stream == stream.cuda_stream for torch_stream, _ in observed_streams
    )


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two CUDA devices")
def test_cuda_input_on_other_device_is_rejected_before_copy() -> None:
    inputs = _one_atom_input().to("cuda:1")
    config = PackingConfig(
        z=1,
        z_prime=1,
        batch_size=1,
        cell_volume_range=(10_000.0, 10_000.0),
        fixed_space_group=1,
    )
    with pytest.raises(ValueError, match=r"inputs\.conformer_positions is on cuda:1"):
        CrystalPacker(config, device="cuda:0")(
            inputs, rng=torch.Generator(device="cuda:0")
        )


@pytest.mark.parametrize("device", ["cuda:0", "cuda:1"])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_cuda_public_packer_preserves_selected_device(device: str) -> None:
    inputs = _one_atom_input()
    config = PackingConfig(
        z=1,
        z_prime=1,
        batch_size=2,
        max_candidates=2,
        cell_volume_range=(10_000.0, 10_000.0),
        fixed_space_group=1,
    )
    result = CrystalPacker(config, device=device)(
        inputs,
        num_samples=2,
        rng=torch.Generator(device=device).manual_seed(12),
    )
    assert len(result) == 2
    assert result.structures.cells.device == torch.device(device)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_cuda_packer_identity_survives_p1_expansion() -> None:
    config = PackingConfig(
        z=1,
        z_prime=1,
        batch_size=2,
        max_candidates=2,
        cell_volume_range=(10_000.0, 10_000.0),
        fixed_space_group=1,
    )
    result = CrystalPacker(config, device="cuda:0")(
        _one_atom_input(),
        num_samples=2,
        rng=torch.Generator(device="cuda:0").manual_seed(26),
        run_id=73,
    )

    assert result.run_id == 73
    assert result.structures.structure_ids.device == torch.device("cuda:0")
    assert result.structures.structure_ids.tolist() == [[73, 0], [73, 1]]
    batch = result.structures.to_batch()
    assert batch.device == torch.device("cuda:0")
    assert batch.csp_source_structure_id.tolist() == [[73, 0], [73, 1]]


@pytest.mark.parametrize("device", ["cuda:0", "cuda:1"])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_cuda_zero_accepted_shortfall_keeps_empty_outputs_on_device(
    device: str,
) -> None:
    inputs = _one_atom_input(contact_distance=5.0)
    config = PackingConfig(
        z=1,
        z_prime=1,
        batch_size=1,
        max_candidates=1,
        max_steps_per_candidate=1,
        convergence_check_interval=1,
        overlap_tolerance=0.0,
        step_scale=0.0,
        cell_step_scale=0.0,
        cell_volume_range=(90.0, 90.0),
        fixed_space_group=1,
    )
    result = CrystalPacker(config, device=device)(
        inputs,
        num_samples=1,
        rng=torch.Generator(device=device).manual_seed(88),
    )

    assert len(result) == 0
    assert result.generated_count == 1
    assert result.stop_reason is PackingStopReason.CANDIDATE_BUDGET_EXHAUSTED
    structures = result.structures
    assert structures.cells.shape == (0, 3, 3)
    assert structures.cells.device == torch.device(device)
    assert structures.structure_molecule_ptr.device == torch.device(device)
    assert structures.conformer_indices.device == torch.device(device)
    assert structures.properties["steps"].device == torch.device(device)
    assert structures.properties["total_overlap"].device == torch.device(device)
    assert structures.properties["max_overlap"].device == torch.device(device)
