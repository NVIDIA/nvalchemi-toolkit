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
"""Public contract tests for CSP formula and compact tensor data."""

from __future__ import annotations

import warnings

import pytest
import torch

from nvalchemi.csp.data import MolecularPackingInput, RigidMoleculeASUBatch


def make_packing_input(**overrides: object) -> MolecularPackingInput:
    """Create a two-component formula unit with three conformers."""
    values: dict[str, object] = {
        "conformer_positions": torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0],
                [2.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 3.0, 2.0],
            ],
            dtype=torch.float32,
        ),
        "conformer_ptr": torch.tensor([0, 2, 4, 5], dtype=torch.int32),
        "molecule_conformer_ptr": torch.tensor([0, 2, 3], dtype=torch.int32),
        "molecule_atom_ptr": torch.tensor([0, 2, 3], dtype=torch.int32),
        "atomic_numbers": torch.tensor([6, 1, 8], dtype=torch.int64),
        "contact_distances": torch.tensor(
            [[1.0, 1.2, 1.3], [1.2, 1.0, 1.4], [1.3, 1.4, 1.0]],
            dtype=torch.float32,
        ),
        "component_index": torch.tensor([0, 1], dtype=torch.int32),
        "component_charge": torch.zeros(2, dtype=torch.int32),
        "formula_unit_volume": 42.5,
        "metadata": {"source": {"name": "seed.xyz", "atom_order": [0, 1, 2]}},
    }
    values.update(overrides)
    return MolecularPackingInput(**values)


def make_digest_input(**overrides: object) -> MolecularPackingInput:
    """Create a tiny formula input with a pinned, independently computed digest."""
    values: dict[str, object] = {
        "conformer_positions": torch.zeros((1, 3), dtype=torch.float32),
        "conformer_ptr": torch.tensor([0, 1], dtype=torch.int32),
        "molecule_conformer_ptr": torch.tensor([0, 1], dtype=torch.int32),
        "molecule_atom_ptr": torch.tensor([0, 1], dtype=torch.int32),
        "atomic_numbers": torch.tensor([6], dtype=torch.int64),
        "contact_distances": torch.tensor([[1.25]], dtype=torch.float32),
        "component_index": torch.tensor([0], dtype=torch.int32),
        "component_charge": torch.zeros(1, dtype=torch.int32),
        "formula_unit_volume": 2.5,
        "metadata": {"source": "fixture", "values": [1, 2]},
    }
    values.update(overrides)
    return MolecularPackingInput(**values)


def make_compact(
    packing_input: MolecularPackingInput | None = None,
    *,
    properties: dict[str, torch.Tensor] | None = None,
) -> RigidMoleculeASUBatch:
    """Create three compact structures with two ASU molecules apiece."""
    packing_input = packing_input or make_packing_input()
    pool_starts = packing_input.molecule_conformer_ptr[:-1].tolist()
    pool_stops = packing_input.molecule_conformer_ptr[1:].tolist()
    conformer_indices = torch.tensor(
        [
            pool_starts[molecule] if row == 0 else pool_stops[molecule] - 1
            for row in range(3)
            for molecule in range(packing_input.num_molecules)
        ],
        dtype=torch.int32,
    )
    return RigidMoleculeASUBatch(
        packing_input=packing_input,
        structure_molecule_ptr=torch.tensor([0, 2, 4, 6], dtype=torch.int32),
        conformer_indices=conformer_indices,
        rotations=torch.eye(3, dtype=torch.float32).expand(6, 3, 3).clone(),
        fractional_centers=torch.zeros((6, 3), dtype=torch.float32),
        cells=torch.eye(3, dtype=torch.float32).expand(3, 3, 3).clone(),
        space_groups=torch.ones(3, dtype=torch.int32),
        z=torch.ones(3, dtype=torch.int32),
        z_prime=torch.ones(3, dtype=torch.int32),
        properties=properties or {"score": torch.tensor([10.0, 20.0, 30.0])},
    )


def make_mixed_compact() -> RigidMoleculeASUBatch:
    """Create valid mixed Z/Z-prime structures for integrity checks."""
    return RigidMoleculeASUBatch(
        packing_input=make_packing_input(),
        structure_molecule_ptr=torch.tensor([0, 4, 6], dtype=torch.int32),
        conformer_indices=torch.tensor([0, 2, 0, 2, 0, 2], dtype=torch.int32),
        rotations=torch.eye(3, dtype=torch.float32).expand(6, 3, 3).clone(),
        fractional_centers=torch.zeros((6, 3), dtype=torch.float32),
        cells=torch.eye(3, dtype=torch.float32).expand(2, 3, 3).clone(),
        space_groups=torch.tensor([2, 2], dtype=torch.int32),
        z=torch.tensor([4, 2], dtype=torch.int32),
        z_prime=torch.tensor([2, 1], dtype=torch.int32),
        structure_ids=torch.tensor([[11, 0], [11, 1]], dtype=torch.int64),
        properties={"score": torch.tensor([10.0, 20.0])},
    )


def _compact_with_updates(
    compact: RigidMoleculeASUBatch, **updates: object
) -> RigidMoleculeASUBatch:
    """Construct through the public schema with selected semantic mutations."""
    values: dict[str, object] = {
        name: getattr(compact, name)
        for name in (
            "packing_input",
            "structure_molecule_ptr",
            "conformer_indices",
            "rotations",
            "fractional_centers",
            "cells",
            "space_groups",
            "z",
            "z_prime",
            "structure_ids",
            "properties",
        )
    }
    values.update(updates)
    return RigidMoleculeASUBatch(**values)


class TestMolecularPackingInput:
    def test_sha256_matches_frozen_encoding_with_component_charge(self) -> None:
        packing_input = make_digest_input()

        # Independently computed from the ordered tensor bytes, including the
        # explicit neutral component charge, sorted metadata JSON, and volume.
        assert packing_input.sha256 == (
            "de4fc5de75aecc4a0341c188e4d08c3e31e5e8c8ac1df786c21ba3dc5d122f68"
        )
        assert len(packing_input.sha256) == 64
        assert packing_input.sha256 == packing_input.sha256.lower()
        assert "sha256" not in packing_input.state_dict()
        assert "sha256" not in MolecularPackingInput.model_fields

    def test_sha256_tracks_exact_input_state_and_does_not_consume_rng(self) -> None:
        packing_input = make_digest_input()
        expected = packing_input.sha256
        restored = MolecularPackingInput.from_state_dict(packing_input.state_dict())
        reordered_metadata = make_digest_input(
            metadata={"values": [1, 2], "source": "fixture"}
        )
        changed_contact = make_digest_input(
            contact_distances=torch.tensor([[1.5]], dtype=torch.float32)
        )
        changed_metadata = make_digest_input(
            metadata={"source": "fixture", "values": [1, 3]}
        )
        changed_volume = make_digest_input(formula_unit_volume=2.75)
        with pytest.warns(UserWarning, match="Qformula=1"):
            changed_charge = make_digest_input(
                component_charge=torch.tensor([1], dtype=torch.int32)
            )
        rng_state = torch.random.get_rng_state().clone()

        assert restored.sha256 == expected
        assert packing_input.to("cpu").sha256 == expected
        assert reordered_metadata.sha256 == expected
        assert changed_contact.sha256 != expected
        assert changed_metadata.sha256 != expected
        assert changed_volume.sha256 != expected
        assert changed_charge.sha256 != expected
        assert torch.equal(torch.random.get_rng_state(), rng_state)

    def test_component_charge_maps_molecule_stoichiometry_and_warns_once(self) -> None:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            balanced = MolecularPackingInput(
                conformer_positions=torch.zeros((3, 3), dtype=torch.float32),
                conformer_ptr=torch.tensor([0, 1, 2, 3], dtype=torch.int32),
                molecule_conformer_ptr=torch.tensor([0, 1, 2, 3], dtype=torch.int32),
                molecule_atom_ptr=torch.tensor([0, 1, 2, 3], dtype=torch.int32),
                atomic_numbers=torch.tensor([11, 11, 8], dtype=torch.int64),
                contact_distances=torch.ones((3, 3), dtype=torch.float32),
                component_index=torch.tensor([0, 0, 1], dtype=torch.int32),
                component_charge=torch.tensor([1, -2], dtype=torch.int32),
                formula_unit_volume=20.0,
            )
        assert not caught
        assert balanced.molecule_charge.tolist() == [1, 1, -2]
        assert balanced.formula_unit_charge == 0

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            nonneutral = make_packing_input(
                component_charge=torch.tensor([1, 0], dtype=torch.int32)
            )
            nonneutral.to("cpu")
            MolecularPackingInput.from_state_dict(nonneutral.state_dict())
        assert len(caught) == 1
        warning = str(caught[0].message)
        assert "Qformula=1" in warning
        assert "Qcell = Z * Qformula" in warning
        assert "counterions and stoichiometry" in warning

    def test_constructor_owns_cpu_tensors_centers_each_conformer_and_copies_metadata(
        self,
    ) -> None:
        values = {
            "conformer_positions": torch.tensor(
                [
                    [0.0, 0.0, 0.0],
                    [2.0, 0.0, 0.0],
                    [1.0, 1.0, 1.0],
                    [3.0, 1.0, 1.0],
                    [4.0, 0.0, 0.0],
                    [6.0, 0.0, 0.0],
                ],
                dtype=torch.float32,
            ),
            "conformer_ptr": torch.tensor([0, 2, 4, 6], dtype=torch.int32),
            "molecule_conformer_ptr": torch.tensor([0, 3], dtype=torch.int32),
            "molecule_atom_ptr": torch.tensor([0, 2], dtype=torch.int32),
            "atomic_numbers": torch.tensor([6, 1], dtype=torch.int64),
            "contact_distances": torch.ones((2, 2), dtype=torch.float32),
            "component_index": torch.tensor([0], dtype=torch.int32),
            "component_charge": torch.zeros(1, dtype=torch.int32),
            "formula_unit_volume": 12,
            "metadata": {"nested": {"values": [1, 2]}},
        }
        original_positions = values["conformer_positions"]
        assert isinstance(original_positions, torch.Tensor)
        packing_input = MolecularPackingInput(**values)

        assert packing_input.conformer_positions.device.type == "cpu"
        assert packing_input.conformer_positions.is_contiguous()
        assert (
            packing_input.conformer_positions.data_ptr()
            != original_positions.data_ptr()
        )
        torch.testing.assert_close(
            packing_input.conformer_positions[:2].mean(dim=0), torch.zeros(3)
        )
        torch.testing.assert_close(
            packing_input.conformer_positions[2:4].mean(dim=0), torch.zeros(3)
        )
        torch.testing.assert_close(
            packing_input.conformer_positions[4:6].mean(dim=0), torch.zeros(3)
        )
        assert packing_input.formula_unit_volume == 12.0
        assert packing_input.num_atoms == 2
        assert packing_input.num_molecules == 1
        assert packing_input.num_conformers == 3
        assert packing_input.num_components == 1

        original_positions.fill_(50.0)
        values["metadata"]["nested"]["values"].append(3)  # type: ignore[index]
        assert not torch.equal(packing_input.conformer_positions, original_positions)
        assert packing_input.metadata["nested"]["values"] == (1, 2)  # type: ignore[index]
        with pytest.raises(TypeError):
            packing_input.metadata["nested"]["new"] = True  # type: ignore[index]

    def test_state_dict_round_trip_and_explicit_transfers_are_independently_owned(
        self,
    ) -> None:
        packing_input = make_packing_input()
        state = packing_input.state_dict()
        restored = MolecularPackingInput.from_state_dict(state)
        assert state["version"] == 1
        assert restored.metadata == packing_input.metadata
        for name in (
            "conformer_positions",
            "conformer_ptr",
            "molecule_conformer_ptr",
            "molecule_atom_ptr",
            "atomic_numbers",
            "contact_distances",
            "component_index",
            "component_charge",
        ):
            source = getattr(packing_input, name)
            state_tensor = state[name]
            restored_tensor = getattr(restored, name)
            assert isinstance(state_tensor, torch.Tensor)
            assert state_tensor.data_ptr() != source.data_ptr()
            assert restored_tensor.data_ptr() != source.data_ptr()
            torch.testing.assert_close(state_tensor, source)
            torch.testing.assert_close(restored_tensor, source)

        state["conformer_positions"].zero_()  # type: ignore[union-attr]
        state["metadata"]["source"]["atom_order"].append(5)  # type: ignore[index]
        assert not torch.equal(
            state["conformer_positions"], packing_input.conformer_positions
        )
        assert packing_input.metadata["source"]["atom_order"] == (0, 1, 2)  # type: ignore[index]

        transferred = packing_input.to("cpu")
        assert transferred is not packing_input
        for name in (
            "conformer_positions",
            "conformer_ptr",
            "molecule_conformer_ptr",
            "molecule_atom_ptr",
            "atomic_numbers",
            "contact_distances",
            "component_index",
            "component_charge",
        ):
            source = getattr(packing_input, name)
            result = getattr(transferred, name)
            assert result.device.type == "cpu"
            assert result.data_ptr() != source.data_ptr()
            torch.testing.assert_close(result, source)

    def test_state_round_trip_preserves_float32_centering_residual_and_concatenates(
        self,
    ) -> None:
        packing_input = MolecularPackingInput(
            conformer_positions=torch.tensor(
                [
                    [100000.0, 0.0, 0.0],
                    [100000.125, 0.0, 0.0],
                    [100000.5, 0.0, 0.0],
                    [-1.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, -1.0, 0.0],
                    [0.0, 1.0, 0.0],
                ],
                dtype=torch.float32,
            ),
            conformer_ptr=torch.tensor([0, 3, 5, 7], dtype=torch.int32),
            molecule_conformer_ptr=torch.tensor([0, 1, 3], dtype=torch.int32),
            molecule_atom_ptr=torch.tensor([0, 3, 5], dtype=torch.int32),
            atomic_numbers=torch.tensor([6, 6, 6, 1, 1], dtype=torch.int64),
            contact_distances=torch.ones((5, 5), dtype=torch.float32),
            component_index=torch.tensor([0, 1], dtype=torch.int32),
            component_charge=torch.zeros(2, dtype=torch.int32),
            formula_unit_volume=15.0,
        )
        residual = packing_input.conformer_positions[:3].mean(dim=0)
        assert residual.abs().max() > 1.0e-4

        restored = MolecularPackingInput.from_state_dict(packing_input.state_dict())
        assert torch.equal(
            restored.conformer_positions, packing_input.conformer_positions
        )
        assert torch.equal(restored.conformer_positions[:3].mean(dim=0), residual)

        original_compact = make_compact(packing_input)
        restored_compact = make_compact(restored)
        joined = RigidMoleculeASUBatch.concatenate(
            [original_compact.select(torch.tensor([1])), restored_compact]
        )
        assert joined.packing_input is packing_input
        assert joined.num_structures == 4

    @pytest.mark.parametrize(
        ("field", "replacement", "error"),
        [
            (
                "molecule_atom_ptr",
                torch.tensor([1, 3], dtype=torch.int32),
                ValueError,
            ),
            (
                "molecule_conformer_ptr",
                torch.tensor([0, 1, 3], dtype=torch.int32),
                ValueError,
            ),
            (
                "contact_distances",
                torch.tensor([[1.0, 1.0, 1.0], [2.0, 1.0, 1.0], [1.0, 1.0, 1.0]]),
                ValueError,
            ),
            ("component_index", torch.tensor([0, 2], dtype=torch.int32), ValueError),
            (
                "component_charge",
                torch.tensor([0], dtype=torch.int32),
                ValueError,
            ),
            (
                "component_charge",
                torch.zeros(2, dtype=torch.int64),
                TypeError,
            ),
            ("atomic_numbers", torch.tensor([6, 0, 8], dtype=torch.int64), ValueError),
            ("formula_unit_volume", float("inf"), ValueError),
            ("formula_unit_volume", True, TypeError),
        ],
    )
    def test_formula_invariants_reject_invalid_values(
        self, field: str, replacement: object, error: type[Exception]
    ) -> None:
        with pytest.raises(error):
            make_packing_input(**{field: replacement})

    def test_atomic_number_boundary_accepts_118_and_rejects_119(self) -> None:
        valid = make_packing_input(
            atomic_numbers=torch.tensor([118, 1, 8], dtype=torch.int64)
        )
        assert valid.atomic_numbers.tolist() == [118, 1, 8]

        with pytest.raises(ValueError, match=r"atomic_numbers.*\[1, 118\]"):
            make_packing_input(
                atomic_numbers=torch.tensor([119, 1, 8], dtype=torch.int64)
            )

    def test_state_dict_rejects_unknown_version_and_nonexact_keys(self) -> None:
        state = make_packing_input().state_dict()
        state["version"] = 2
        with pytest.raises(ValueError, match="version"):
            MolecularPackingInput.from_state_dict(state)
        state = make_packing_input().state_dict()
        state["extra"] = 3
        with pytest.raises(ValueError, match="exactly"):
            MolecularPackingInput.from_state_dict(state)

    def test_component_charge_is_required(self) -> None:
        values = make_digest_input().state_dict()
        del values["component_charge"]
        values.pop("version")
        with pytest.raises(ValueError, match="Missing MolecularPackingInput fields"):
            MolecularPackingInput(**values)
        with pytest.raises(TypeError, match="component_charge must be a torch.Tensor"):
            make_digest_input(component_charge=None)


class TestRigidMoleculeASUBatch:
    def test_check_integrity_accepts_empty_and_mixed_batches_without_side_effects(
        self,
    ) -> None:
        compact = make_mixed_compact()
        empty = compact.select(torch.empty(0, dtype=torch.int64))
        assert empty.check_integrity() is None

        names = (
            "structure_molecule_ptr",
            "conformer_indices",
            "space_groups",
            "z",
            "z_prime",
        )
        snapshots = {name: getattr(compact, name).clone() for name in names}
        formula_ptr = compact.packing_input.molecule_conformer_ptr.clone()
        rng_state = torch.random.get_rng_state().clone()
        before = compact.to_batch()

        assert compact.check_integrity() is None

        after = compact.to_batch()
        for name, snapshot in snapshots.items():
            torch.testing.assert_close(getattr(compact, name), snapshot)
        torch.testing.assert_close(
            compact.packing_input.molecule_conformer_ptr, formula_ptr
        )
        assert torch.equal(torch.random.get_rng_state(), rng_state)
        for name in (
            "positions",
            "atomic_numbers",
            "cell",
            "pbc",
            "csp_source_structure_id",
        ):
            torch.testing.assert_close(getattr(after, name), getattr(before, name))
        assert after.csp_source_structure_id.tolist() == [[11, 0], [11, 1]]

    @pytest.mark.parametrize(
        ("updates", "message"),
        [
            (
                {"structure_molecule_ptr": torch.tensor([1, 4, 6], dtype=torch.int32)},
                "start at zero",
            ),
            (
                {"structure_molecule_ptr": torch.tensor([0, 5, 4], dtype=torch.int32)},
                "nondecreasing",
            ),
            (
                {"structure_molecule_ptr": torch.tensor([0, 4, 5], dtype=torch.int32)},
                "end at the conformer_indices length",
            ),
            (
                {"structure_molecule_ptr": torch.tensor([0, 3, 6], dtype=torch.int32)},
                "span has 3 ASU molecules",
            ),
            ({"z": torch.tensor([0, 2], dtype=torch.int32)}, "z must be positive"),
            ({"z": torch.tensor([-1, 2], dtype=torch.int32)}, "z must be positive"),
            (
                {"z_prime": torch.tensor([0, 1], dtype=torch.int32)},
                "z_prime must be positive",
            ),
            (
                {"z_prime": torch.tensor([-1, 1], dtype=torch.int32)},
                "z_prime must be positive",
            ),
            (
                {
                    "z": torch.tensor([3, 2], dtype=torch.int32),
                    "z_prime": torch.tensor([2, 1], dtype=torch.int32),
                },
                "z_prime must divide z",
            ),
            (
                {"space_groups": torch.tensor([231, 2], dtype=torch.int32)},
                r"space_groups\[0\] must be in \[1, 230\]",
            ),
            (
                {"space_groups": torch.tensor([1, 2], dtype=torch.int32)},
                "has 1 operations; expected 2",
            ),
            (
                {
                    "conformer_indices": torch.tensor(
                        [2, 2, 0, 2, 0, 2], dtype=torch.int32
                    )
                },
                r"ASU row 0 is outside formula molecule 0 pool \[0, 2\)",
            ),
        ],
    )
    def test_check_integrity_rejects_invalid_index_relationships(
        self, updates: dict[str, object], message: str
    ) -> None:
        compact = _compact_with_updates(make_mixed_compact(), **updates)
        with pytest.raises(ValueError, match=message):
            compact.check_integrity()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
    def test_check_integrity_on_cuda_and_digest_transfer_invariance(self) -> None:
        cpu_input = make_packing_input()
        device_input = cpu_input.to("cuda:0")
        assert device_input.sha256 == cpu_input.sha256

        valid = make_mixed_compact().to("cuda:0")
        invalid = _compact_with_updates(
            valid,
            z_prime=torch.tensor([0, 1], dtype=torch.int32, device="cuda:0"),
        )
        snapshots = {
            name: getattr(valid, name).detach().cpu().clone()
            for name in (
                "structure_molecule_ptr",
                "conformer_indices",
                "space_groups",
                "z",
                "z_prime",
            )
        }
        cpu_rng = torch.random.get_rng_state().clone()
        cuda_rng = torch.cuda.get_rng_state("cuda:0").clone()

        assert valid.check_integrity() is None
        with pytest.raises(ValueError, match="z_prime must be positive"):
            invalid.check_integrity()

        for name, snapshot in snapshots.items():
            torch.testing.assert_close(getattr(valid, name).cpu(), snapshot)
        assert torch.equal(torch.random.get_rng_state(), cpu_rng)
        assert torch.equal(torch.cuda.get_rng_state("cuda:0"), cuda_rng)

    def test_schema_properties_selection_order_repetition_and_empty_selection(
        self,
    ) -> None:
        compact = make_compact()
        selected = compact.select(torch.tensor([2, 0, 2], dtype=torch.int32))

        assert selected.packing_input is compact.packing_input
        assert selected.num_structures == 3
        assert selected.structure_molecule_ptr.tolist() == [0, 2, 4, 6]
        assert selected.conformer_indices.tolist() == [1, 2, 0, 2, 1, 2]
        assert selected.properties["score"].tolist() == [30.0, 10.0, 30.0]
        torch.testing.assert_close(
            selected.structure_ids, compact.structure_ids[[2, 0, 2]]
        )
        assert selected.structure_ids.dtype == torch.int64
        assert (selected.structure_ids >= 0).all()
        assert torch.unique(compact.structure_ids[:, 0]).numel() == 1
        assert compact.structure_ids[:, 1].tolist() == [0, 1, 2]

        empty = compact.select(torch.empty(0, dtype=torch.int64))
        assert empty.num_structures == 0
        assert empty.structure_molecule_ptr.tolist() == [0]
        assert empty.conformer_indices.shape == (0,)
        assert empty.rotations.shape == (0, 3, 3)
        assert empty.cells.shape == (0, 3, 3)
        assert empty.structure_ids.shape == (0, 2)
        assert empty.properties["score"].shape == (0,)

    @pytest.mark.parametrize(
        "name",
        [
            "positions",
            "energy",
            "batch_ptr",
            "csp_source_z",
            "csp_source_structure_id",
        ],
    )
    def test_properties_cannot_collide_with_toolkit_or_source_fields(
        self, name: str
    ) -> None:
        with pytest.raises(ValueError, match="collides"):
            make_compact(properties={name: torch.zeros(3)})

    def test_property_leading_dimension_dtype_and_device_are_checked(self) -> None:
        with pytest.raises(ValueError, match="leading dimension"):
            make_compact(properties={"score": torch.zeros(2)})

        compact = make_compact()
        with pytest.raises(TypeError, match="dtype"):
            compact.select(torch.tensor([0.0]))
        with pytest.raises(TypeError, match="one-dimensional"):
            compact.select(torch.tensor([[0]], dtype=torch.int64))

    def test_structure_ids_validate_shape_and_dtype_but_retain_values(self) -> None:
        compact = make_compact()
        values = {
            "packing_input": compact.packing_input,
            "structure_molecule_ptr": compact.structure_molecule_ptr,
            "conformer_indices": compact.conformer_indices,
            "rotations": compact.rotations,
            "fractional_centers": compact.fractional_centers,
            "cells": compact.cells,
            "space_groups": compact.space_groups,
            "z": compact.z,
            "z_prime": compact.z_prime,
            "properties": compact.properties,
        }
        with pytest.raises(TypeError, match="dtype"):
            RigidMoleculeASUBatch(
                **values, structure_ids=torch.zeros((3, 2), dtype=torch.int32)
            )
        with pytest.raises(ValueError, match="shape"):
            RigidMoleculeASUBatch(
                **values, structure_ids=torch.zeros((3, 1), dtype=torch.int64)
            )
        invalid_ids = compact.structure_ids.clone()
        invalid_ids[0, 1] = -1
        retained = RigidMoleculeASUBatch(**values, structure_ids=invalid_ids)
        torch.testing.assert_close(retained.structure_ids, invalid_ids)

    def test_concatenate_checks_schema_and_equal_formula_input(self) -> None:
        compact = make_compact()
        selected_a = compact.select(torch.tensor([2, 0], dtype=torch.int64))
        selected_b = compact.select(torch.tensor([1], dtype=torch.int64))
        joined = RigidMoleculeASUBatch.concatenate([selected_a, selected_b])
        assert joined.packing_input is compact.packing_input
        assert joined.structure_molecule_ptr.tolist() == [0, 2, 4, 6]
        assert joined.properties["score"].tolist() == [30.0, 10.0, 20.0]
        torch.testing.assert_close(
            joined.structure_ids,
            torch.cat([selected_a.structure_ids, selected_b.structure_ids]),
        )

        equal_input_batch = make_compact(make_packing_input()).select(
            torch.tensor([1], dtype=torch.int64)
        )
        equal_joined = RigidMoleculeASUBatch.concatenate(
            [selected_a, equal_input_batch]
        )
        assert equal_joined.num_structures == 3
        assert equal_joined.packing_input is compact.packing_input

        other_schema = make_compact(properties={"weight": torch.ones(3)}).select(
            torch.tensor([0], dtype=torch.int64)
        )
        with pytest.raises(ValueError, match="property keys"):
            RigidMoleculeASUBatch.concatenate([selected_a, other_schema])
        with pytest.raises(ValueError, match="at least one"):
            RigidMoleculeASUBatch.concatenate([])

    def test_same_device_transfer_clones_compact_and_formula_storage(self) -> None:
        compact = make_compact()
        moved = compact.to("cpu")
        assert moved.packing_input is not compact.packing_input
        assert (
            moved.structure_molecule_ptr.data_ptr()
            != compact.structure_molecule_ptr.data_ptr()
        )
        assert (
            moved.properties["score"].data_ptr()
            != compact.properties["score"].data_ptr()
        )
        assert moved.packing_input.conformer_positions.data_ptr() != (
            compact.packing_input.conformer_positions.data_ptr()
        )
        torch.testing.assert_close(moved.cells, compact.cells)
        torch.testing.assert_close(moved.structure_ids, compact.structure_ids)
        assert moved.structure_ids.data_ptr() != compact.structure_ids.data_ptr()
