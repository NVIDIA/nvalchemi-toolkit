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
"""Public persistence contract tests for compact CSP structures."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
import zarr

from nvalchemi.csp.data import MolecularPackingInput, RigidMoleculeASUBatch
from nvalchemi.csp.storage import RigidMoleculeASUZarrReader, RigidMoleculeASUZarrWriter
from nvalchemi.data.datapipes.backends.zarr import ZarrArrayConfig, ZarrWriteConfig


def make_compact(
    ids: list[tuple[int, int]] | None = None,
    *,
    scores: list[float] | None = None,
    metadata: object = {"source": {"name": "test", "labels": ["a", "b"]}},
    multiplicities: list[tuple[int, int]] | None = None,
    property_name: str = "score",
) -> RigidMoleculeASUBatch:
    """Create three valid P1 compact structures with stable caller IDs."""
    count = 3 if ids is None else len(ids)
    packing_input = MolecularPackingInput(
        conformer_positions=torch.tensor(
            [
                [0.1, 0.0, 0.0],
                [0.2, 0.0, 0.0],
                [0.4, 0.0, 0.0],
                [0.0, 0.1, 0.0],
                [0.0, 0.2, 0.0],
                [0.0, 0.4, 0.0],
            ],
            dtype=torch.float32,
        ),
        conformer_ptr=torch.tensor([0, 3, 6], dtype=torch.int32),
        molecule_conformer_ptr=torch.tensor([0, 2], dtype=torch.int32),
        molecule_atom_ptr=torch.tensor([0, 3], dtype=torch.int32),
        atomic_numbers=torch.tensor([6, 6, 6], dtype=torch.int64),
        contact_distances=torch.ones((3, 3), dtype=torch.float32),
        component_index=torch.tensor([0], dtype=torch.int32),
        formula_unit_volume=12.0,
        metadata=metadata,
    )
    if ids is None:
        ids = [(11, i) for i in range(count)]
    if scores is None:
        scores = [float(i) for i in range(count)]
    if multiplicities is None:
        multiplicities = [(1, 1)] * count
    molecules_per_structure = [z_prime for _, z_prime in multiplicities]
    molecule_ptr = torch.tensor(
        [0, *torch.tensor(molecules_per_structure).cumsum(0).tolist()],
        dtype=torch.int32,
    )
    molecule_count = int(molecule_ptr[-1])
    angles = torch.arange(molecule_count, dtype=torch.float32) * 0.2
    rotations = torch.eye(3, dtype=torch.float32).expand(molecule_count, 3, 3).clone()
    rotations[:, 0, 0] = torch.cos(angles)
    rotations[:, 0, 1] = -torch.sin(angles)
    rotations[:, 1, 0] = torch.sin(angles)
    rotations[:, 1, 1] = torch.cos(angles)
    cells = torch.eye(3, dtype=torch.float32).expand(count, 3, 3).clone()
    for index in range(count):
        cells[index] = torch.diag(torch.tensor([1.0 + index, 2.0 + index, 3.0 + index]))
    return RigidMoleculeASUBatch(
        packing_input=packing_input,
        structure_molecule_ptr=molecule_ptr,
        conformer_indices=(torch.arange(molecule_count, dtype=torch.int32) % 2),
        rotations=rotations,
        fractional_centers=(
            torch.arange(molecule_count * 3, dtype=torch.float32).reshape(-1, 3)
            * 0.03125
        ),
        cells=cells,
        space_groups=torch.tensor(
            [1 if z == 1 else 2 for z, _ in multiplicities], dtype=torch.int32
        ),
        z=torch.tensor([z for z, _ in multiplicities], dtype=torch.int32),
        z_prime=torch.tensor(
            [z_prime for _, z_prime in multiplicities], dtype=torch.int32
        ),
        structure_ids=torch.tensor(ids, dtype=torch.int64).reshape(count, 2),
        properties={property_name: torch.tensor(scores, dtype=torch.float32)},
    )


def test_round_trip_order_repeats_metadata_and_p1(tmp_path) -> None:
    store = tmp_path / "compact.zarr"
    source = make_compact(multiplicities=[(1, 1), (2, 1), (4, 2)])
    with RigidMoleculeASUZarrWriter(store) as writer:
        writer.write(source)
    with RigidMoleculeASUZarrReader(store) as reader:
        assert len(reader) == 3
        assert reader.metadata["representation"] == "nvalchemi.csp.rigid_molecule_asu"
        assert reader.get_metadata(
            torch.tensor([2, 0], dtype=torch.int32)
        ).tolist() == [
            [12, 0],
            [3, 0],
        ]
        selected = reader.read(torch.tensor([2, 0, 2], dtype=torch.int64))
        expected = source.select(torch.tensor([2, 0, 2], dtype=torch.int64))
        for name in (
            "conformer_positions",
            "conformer_ptr",
            "molecule_conformer_ptr",
            "molecule_atom_ptr",
            "atomic_numbers",
            "contact_distances",
            "component_index",
        ):
            torch.testing.assert_close(
                getattr(selected.packing_input, name),
                getattr(expected.packing_input, name),
            )
        assert selected.packing_input.formula_unit_volume == (
            expected.packing_input.formula_unit_volume
        )
        assert selected.packing_input.metadata == expected.packing_input.metadata
        for name in (
            "structure_molecule_ptr",
            "structure_ids",
            "conformer_indices",
            "rotations",
            "fractional_centers",
            "cells",
            "space_groups",
            "z",
            "z_prime",
        ):
            torch.testing.assert_close(getattr(selected, name), getattr(expected, name))
        for name, expected_property in expected.properties.items():
            torch.testing.assert_close(selected.properties[name], expected_property)
        assert selected.structure_ids.tolist() == [[11, 2], [11, 0], [11, 2]]
        batch = reader.read_batch(torch.tensor([2, 0], dtype=torch.int64))
        assert batch.num_graphs == 2
        assert batch.num_nodes == 15
        assert batch.csp_source_structure_id.tolist() == [[11, 2], [11, 0]]
        repeated_batch = reader.read_batch(torch.tensor([2, 0, 2], dtype=torch.int64))
        assert repeated_batch.csp_source_structure_id.tolist() == [
            [11, 2],
            [11, 0],
            [11, 2],
        ]
        with pytest.raises(TypeError, match="indices is required"):
            reader.read_batch(None)


def test_external_store_wrong_formula_pool_is_rejected_explicitly(tmp_path) -> None:
    store = tmp_path / "wrong-formula-pool.zarr"
    packing_input = MolecularPackingInput(
        conformer_positions=torch.zeros((2, 3), dtype=torch.float32),
        conformer_ptr=torch.tensor([0, 1, 2], dtype=torch.int32),
        molecule_conformer_ptr=torch.tensor([0, 1, 2], dtype=torch.int32),
        molecule_atom_ptr=torch.tensor([0, 1, 2], dtype=torch.int32),
        atomic_numbers=torch.tensor([6, 8], dtype=torch.int64),
        contact_distances=torch.ones((2, 2), dtype=torch.float32),
        component_index=torch.tensor([0, 1], dtype=torch.int32),
        formula_unit_volume=10.0,
    )
    source = RigidMoleculeASUBatch(
        packing_input=packing_input,
        structure_molecule_ptr=torch.tensor([0, 2], dtype=torch.int32),
        conformer_indices=torch.tensor([0, 1], dtype=torch.int32),
        rotations=torch.eye(3, dtype=torch.float32).expand(2, 3, 3).clone(),
        fractional_centers=torch.zeros((2, 3), dtype=torch.float32),
        cells=torch.eye(3, dtype=torch.float32).unsqueeze(0),
        space_groups=torch.tensor([1], dtype=torch.int32),
        z=torch.tensor([1], dtype=torch.int32),
        z_prime=torch.tensor([1], dtype=torch.int32),
        structure_ids=torch.tensor([[77, 0]], dtype=torch.int64),
    )
    with RigidMoleculeASUZarrWriter(store) as writer:
        writer.write(source)

    root = zarr.open_group(store, mode="r+")
    root["core"]["conformer_indices"][0] = 1

    with RigidMoleculeASUZarrReader(store) as reader:
        loaded = reader.read()
    assert loaded.conformer_indices.tolist() == [1, 1]
    with pytest.raises(
        ValueError, match=r"ASU row 0 is outside formula molecule 0 pool \[0, 1\)"
    ):
        loaded.check_integrity()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_cuda_write_selected_compact_and_p1_round_trip(tmp_path) -> None:
    store = tmp_path / "compact-cuda.zarr"
    source = make_compact(multiplicities=[(1, 1), (2, 1), (4, 2)]).to("cuda:0")
    indices = torch.tensor([2, 0], dtype=torch.int64)
    expected = source.select(indices.to(device="cuda:0"))

    with RigidMoleculeASUZarrWriter(store) as writer:
        writer.write(source)
    with RigidMoleculeASUZarrReader(store) as reader:
        compact = reader.read(indices, device="cuda:0")
        assert compact.cells.device == torch.device("cuda:0")
        assert compact.structure_ids.device == torch.device("cuda:0")
        assert compact.structure_ids.tolist() == expected.structure_ids.tolist()
        torch.testing.assert_close(
            compact.properties["score"], expected.properties["score"]
        )

        batch = reader.read_batch(indices, device="cuda:0")
        assert batch.positions.device == torch.device("cuda:0")
        assert batch.csp_source_structure_id.device == torch.device("cuda:0")
        assert batch.csp_source_structure_id.tolist() == expected.structure_ids.tolist()
        torch.testing.assert_close(batch["score"], expected.properties["score"])


def test_append_is_idempotent_and_conflicts_fail_before_mutation(tmp_path) -> None:
    store = tmp_path / "compact.zarr"
    writer = RigidMoleculeASUZarrWriter(store)
    writer.write(make_compact())
    source = make_compact()
    writer.append(source.select(torch.tensor([0, 0])))
    assert int(zarr.open(store, mode="r").attrs["num_samples"]) == 3
    with pytest.raises(ValueError, match="conflicts"):
        writer.append(make_compact([(11, 1)], scores=[99.0], multiplicities=[(1, 1)]))
    assert int(zarr.open(store, mode="r").attrs["num_samples"]) == 3
    writer.append(make_compact([(12, 0)], scores=[7.0]))
    with RigidMoleculeASUZarrReader(store) as reader:
        assert len(reader) == 4
        assert reader.read(torch.tensor([3])).structure_ids.tolist() == [[12, 0]]


def test_write_deduplicates_identical_rows_and_rejects_conflict_before_creation(
    tmp_path,
) -> None:
    duplicate_store = tmp_path / "deduplicated.zarr"
    duplicate_rows = make_compact([(20, 4)], scores=[5.0]).select(
        torch.tensor([0, 0], dtype=torch.int64)
    )
    RigidMoleculeASUZarrWriter(duplicate_store).write(duplicate_rows)
    with RigidMoleculeASUZarrReader(duplicate_store) as reader:
        assert len(reader) == 1
        assert reader.read().structure_ids.tolist() == [[20, 4]]

    conflict_store = tmp_path / "conflict.zarr"
    with pytest.raises(ValueError, match="conflicts"):
        RigidMoleculeASUZarrWriter(conflict_store).write(
            make_compact(
                [(20, 4), (20, 4)],
                scores=[5.0, 6.0],
                multiplicities=[(1, 1), (1, 1)],
            )
        )
    assert not conflict_store.exists()


def test_write_rejects_existing_empty_group(tmp_path) -> None:
    store = tmp_path / "empty-existing.zarr"
    zarr.open_group(store, mode="w")
    with pytest.raises(FileExistsError):
        RigidMoleculeASUZarrWriter(store).write(make_compact())


def test_writer_rejects_negative_ids_before_mutation(tmp_path) -> None:
    invalid = make_compact([(-1, 0)], scores=[2.0])
    new_store = tmp_path / "negative.zarr"
    with pytest.raises(ValueError, match="structure_ids values must be nonnegative"):
        RigidMoleculeASUZarrWriter(new_store).write(invalid)
    assert not new_store.exists()

    store = tmp_path / "existing.zarr"
    writer = RigidMoleculeASUZarrWriter(store)
    writer.write(make_compact())
    with pytest.raises(ValueError, match="structure_ids values must be nonnegative"):
        writer.append(invalid)
    assert zarr.open_group(store, mode="r").attrs["num_samples"] == 3


def test_reopened_append_empty_append_and_reader_refresh(tmp_path) -> None:
    store = tmp_path / "compact.zarr"
    first = RigidMoleculeASUZarrWriter(store)
    first.write(make_compact())
    reader = RigidMoleculeASUZarrReader(store)
    first.append(make_compact([(12, 0)], scores=[9.0]))
    first.append(make_compact([(13, 0)], scores=[10.0]))
    first.append(make_compact([(13, 0)], scores=[10.0]))
    assert int(zarr.open_group(store, mode="r").attrs["num_samples"]) == 5
    with pytest.raises(ValueError, match="conflicts"):
        first.append(make_compact([(13, 0)], scores=[99.0]))
    reader.refresh()
    assert reader.read().structure_ids.tolist() == [
        [11, 0],
        [11, 1],
        [11, 2],
        [12, 0],
        [13, 0],
    ]
    assert reader.read(torch.tensor([4])).properties["score"].tolist() == [10.0]
    first.close()

    reopened = RigidMoleculeASUZarrWriter(store)
    reopened.append(make_compact([], scores=[]))
    assert int(zarr.open_group(store, mode="r").attrs["num_samples"]) == 5
    reopened.append(make_compact([(14, 0)], scores=[11.0]))
    reopened.append(make_compact([(14, 0)], scores=[11.0]))
    assert int(zarr.open_group(store, mode="r").attrs["num_samples"]) == 6
    with pytest.raises(ValueError, match="conflicts"):
        reopened.append(make_compact([(14, 0)], scores=[99.0]))
    assert int(zarr.open_group(store, mode="r").attrs["num_samples"]) == 6
    assert len(reader) == 5
    reader.refresh()
    assert reader.read().structure_ids.tolist() == [
        [11, 0],
        [11, 1],
        [11, 2],
        [12, 0],
        [13, 0],
        [14, 0],
    ]
    assert reader.read(torch.tensor([3, 4, 5])).properties["score"].tolist() == [
        9.0,
        10.0,
        11.0,
    ]


def test_append_rejects_property_schema_mismatch(tmp_path) -> None:
    store = tmp_path / "compact.zarr"
    writer = RigidMoleculeASUZarrWriter(store)
    writer.write(make_compact())
    with pytest.raises(ValueError, match="property keys"):
        writer.append(make_compact([(12, 0)], property_name="different"))
    assert int(zarr.open_group(store, mode="r").attrs["num_samples"]) == 3


@pytest.mark.parametrize("name", ["a/b", ".", "..", "zarr.json"])
def test_unsupported_property_names_fail_before_write_or_append_mutation(
    tmp_path, name
) -> None:
    invalid = make_compact([(12, 0)], property_name=name)
    new_store = tmp_path / "invalid-name.zarr"
    with pytest.raises(ValueError, match="cannot be stored as a Zarr array key"):
        RigidMoleculeASUZarrWriter(new_store).write(invalid)
    assert not new_store.exists()

    store = tmp_path / "existing.zarr"
    writer = RigidMoleculeASUZarrWriter(store)
    writer.write(make_compact())
    with pytest.raises(ValueError, match="cannot be stored as a Zarr array key"):
        writer.append(invalid)
    root = zarr.open_group(store, mode="r")
    assert root.attrs["num_samples"] == 3
    with RigidMoleculeASUZarrReader(store) as reader:
        assert reader.read().structure_ids.tolist() == [[11, 0], [11, 1], [11, 2]]


def test_none_and_empty_formula_metadata_remain_distinct_on_reopen(tmp_path) -> None:
    store = tmp_path / "compact.zarr"
    writer = RigidMoleculeASUZarrWriter(store)
    writer.write(make_compact(metadata=None))
    with RigidMoleculeASUZarrReader(store) as reader:
        assert reader.read().packing_input.metadata is None
    with pytest.raises(ValueError, match="packing_input"):
        writer.append(make_compact([(12, 0)], metadata={}))


def test_reader_rejects_corrupt_formula_state(tmp_path) -> None:
    store = tmp_path / "corrupt-input.zarr"
    RigidMoleculeASUZarrWriter(store).write(make_compact())
    root = zarr.open_group(store, mode="r+")
    root["packing_input"]["atomic_numbers"][0] = -6
    with pytest.raises(ValueError, match="packing_input version-1 state is invalid"):
        RigidMoleculeASUZarrReader(store)


def test_reader_rejects_corrupt_structure_pointer(tmp_path) -> None:
    store = tmp_path / "corrupt-pointer.zarr"
    RigidMoleculeASUZarrWriter(store).write(make_compact())
    root = zarr.open_group(store, mode="r+")
    root["meta"]["molecules_ptr"][:] = np.asarray([0, 1, 3, 2], dtype=np.int32)
    with pytest.raises(ValueError, match="molecules_ptr"):
        RigidMoleculeASUZarrReader(store)


def test_append_pointer_overflow_is_pre_mutation(tmp_path, monkeypatch) -> None:
    import nvalchemi.csp.storage as csp_storage

    store = tmp_path / "overflow.zarr"
    writer = RigidMoleculeASUZarrWriter(store)
    writer.write(make_compact())
    monkeypatch.setattr(csp_storage, "_MAX_POINTER", 3)
    with pytest.raises(OverflowError, match="molecules_ptr"):
        writer.append(make_compact([(12, 0)], scores=[8.0]))
    unchanged = zarr.open_group(store, mode="r")
    assert unchanged.attrs["num_samples"] == 3
    assert unchanged["meta"]["molecules_ptr"][:].tolist() == [0, 1, 2, 3]
    assert unchanged["meta"]["samples_mask"][:].tolist() == [True, True, True]


def test_delete_reuse_and_defragment_preserve_logical_rows(tmp_path) -> None:
    store = tmp_path / "compact.zarr"
    writer = RigidMoleculeASUZarrWriter(store)
    source = make_compact()
    writer.write(source)
    writer.delete(torch.tensor([0], dtype=torch.int64))
    writer.append(source.select(torch.tensor([0])))
    reader = RigidMoleculeASUZarrReader(store)
    assert reader.read().structure_ids.tolist() == [[11, 1], [11, 2], [11, 0]]
    writer.defragment()
    reader.refresh()
    assert len(reader) == 3
    assert reader.read().structure_ids.tolist() == [[11, 1], [11, 2], [11, 0]]
    root = zarr.open(store, mode="r")
    assert root["meta"]["samples_mask"][:].tolist() == [True, True, True]
    assert root["meta"]["molecules_ptr"][:].tolist() == [0, 1, 2, 3]
    assert np.all(root["meta"]["molecules_mask"][:])
    writer.delete(torch.tensor([0, 1, 2], dtype=torch.int64))
    writer.defragment()
    reader.refresh()
    assert len(reader) == 0
    empty = reader.read()
    assert empty.num_structures == 0
    assert empty.properties["score"].shape == (0,)
    empty_root = zarr.open_group(store, mode="r")
    assert empty_root["packing_input"]["atomic_numbers"][:].tolist() == [6, 6, 6]
    assert empty_root["meta"]["molecules_ptr"][:].tolist() == [0]


def test_delete_uses_active_logical_indices_and_repeated_indices_once(tmp_path) -> None:
    store = tmp_path / "logical-delete.zarr"
    writer = RigidMoleculeASUZarrWriter(store)
    writer.write(make_compact())
    writer.delete(torch.tensor([1], dtype=torch.int64))
    with RigidMoleculeASUZarrReader(store) as reader:
        assert reader.read().structure_ids.tolist() == [[11, 0], [11, 2]]

    with pytest.raises(IndexError, match="active logical"):
        writer.delete(torch.tensor([2], dtype=torch.int64))
    with RigidMoleculeASUZarrReader(store) as reader:
        assert reader.read().structure_ids.tolist() == [[11, 0], [11, 2]]

    writer.delete(torch.tensor([1, 0, 1], dtype=torch.int64))
    with RigidMoleculeASUZarrReader(store) as reader:
        assert reader.read().num_structures == 0


def test_defragment_replacement_failure_restores_original_store(
    tmp_path, monkeypatch
) -> None:
    import nvalchemi.csp.storage as csp_storage

    store = tmp_path / "rollback.zarr"
    writer = RigidMoleculeASUZarrWriter(store)
    writer.write(make_compact())
    writer.delete(torch.tensor([1], dtype=torch.int64))
    with RigidMoleculeASUZarrReader(store) as reader:
        expected = reader.read()

    original_replace = csp_storage.os.replace
    calls = 0

    def fail_staged_install(source, destination):
        nonlocal calls
        source_path = Path(source)
        if source_path.name.startswith(".rollback.zarr.csp-stage-"):
            calls += 1
        if calls == 1 and source_path.name.startswith(".rollback.zarr.csp-stage-"):
            raise OSError("injected staged-store install failure")
        return original_replace(source, destination)

    monkeypatch.setattr(csp_storage.os, "replace", fail_staged_install)
    with pytest.raises(RuntimeError, match="original store was restored"):
        writer.defragment()

    with RigidMoleculeASUZarrReader(store) as reader:
        actual = reader.read()
    assert actual.structure_ids.tolist() == expected.structure_ids.tolist()
    torch.testing.assert_close(actual.cells, expected.cells)


def test_defragment_detects_staged_payload_corruption_before_replacement(
    tmp_path, monkeypatch
) -> None:
    store = tmp_path / "corrupt-stage.zarr"
    writer = RigidMoleculeASUZarrWriter(store)
    writer.write(make_compact())
    writer.delete(torch.tensor([1], dtype=torch.int64))
    original_write_new = RigidMoleculeASUZarrWriter._write_new

    def write_corrupted_stage(stage_writer, batch, *, mode="w-"):
        original_write_new(stage_writer, batch, mode=mode)
        stage_path = Path(stage_writer._store)
        if ".csp-stage-" in stage_path.name:
            root = zarr.open_group(stage_path, mode="r+")
            root["core"]["cells"][0, 0, 0] += 0.25

    monkeypatch.setattr(RigidMoleculeASUZarrWriter, "_write_new", write_corrupted_stage)
    with pytest.raises(ValueError, match="staged CSP Zarr payload differs"):
        writer.defragment()

    original = zarr.open_group(store, mode="r")
    assert original.attrs["num_samples"] == 3
    assert original["meta"]["samples_mask"][:].tolist() == [True, False, True]
    with RigidMoleculeASUZarrReader(store) as reader:
        assert reader.read().structure_ids.tolist() == [[11, 0], [11, 2]]


def test_defragment_rejects_non_filesystem_store(tmp_path) -> None:
    store = zarr.storage.MemoryStore()
    writer = RigidMoleculeASUZarrWriter(store)
    writer.write(make_compact())
    with pytest.raises(TypeError, match="local filesystem"):
        writer.defragment()


def test_defragment_cleanup_failure_keeps_backup_and_installed_store(
    tmp_path, monkeypatch
) -> None:
    import nvalchemi.csp.storage as csp_storage

    store = tmp_path / "cleanup.zarr"
    writer = RigidMoleculeASUZarrWriter(store)
    writer.write(make_compact())
    writer.delete(torch.tensor([1], dtype=torch.int64))
    original_rmtree = csp_storage.shutil.rmtree

    def fail_backup_cleanup(path, *args, **kwargs):
        if Path(path).name.startswith(".cleanup.zarr.csp-backup-"):
            raise OSError("injected backup cleanup failure")
        return original_rmtree(path, *args, **kwargs)

    monkeypatch.setattr(csp_storage.shutil, "rmtree", fail_backup_cleanup)
    with pytest.raises(RuntimeError, match="backup cleanup failed"):
        writer.defragment()

    backups = list(tmp_path.glob(".cleanup.zarr.csp-backup-*"))
    assert len(backups) == 1
    backup_root = zarr.open_group(backups[0], mode="r")
    assert backup_root.attrs["num_samples"] == 3
    assert backup_root["meta"]["samples_mask"][:].tolist() == [True, False, True]
    with RigidMoleculeASUZarrReader(backups[0]) as backup_reader:
        assert backup_reader.read().structure_ids.tolist() == [[11, 0], [11, 2]]
    with RigidMoleculeASUZarrReader(store) as reader:
        assert reader.read().structure_ids.tolist() == [[11, 0], [11, 2]]

    writer.append(make_compact([(12, 0)], scores=[8.0]))
    with RigidMoleculeASUZarrReader(store) as reader:
        assert reader.read().structure_ids.tolist() == [[11, 0], [11, 2], [12, 0]]


def test_defragment_invalid_config_preserves_existing_store(tmp_path) -> None:
    store = tmp_path / "invalid-defragment-config.zarr"
    writer = RigidMoleculeASUZarrWriter(store)
    source = make_compact(multiplicities=[(1, 1), (2, 1), (4, 2)])
    writer.write(source)
    writer.delete(torch.tensor([0], dtype=torch.int64))
    with RigidMoleculeASUZarrReader(store) as reader:
        expected = reader.read()

    invalid = ZarrWriteConfig(core=ZarrArrayConfig(chunk_size=-1))
    with pytest.raises(ValueError):
        writer.defragment(invalid)

    root = zarr.open_group(store, mode="r")
    assert root.attrs["num_samples"] == 3
    assert root["meta"]["samples_mask"][:].tolist() == [False, True, True]
    with RigidMoleculeASUZarrReader(store) as reader:
        actual = reader.read()
    for name in (
        "structure_molecule_ptr",
        "structure_ids",
        "conformer_indices",
        "rotations",
        "fractional_centers",
        "cells",
        "space_groups",
        "z",
        "z_prime",
    ):
        torch.testing.assert_close(getattr(actual, name), getattr(expected, name))
    for name, value in expected.properties.items():
        torch.testing.assert_close(actual.properties[name], value)


def test_empty_store_round_trip_and_empty_selection(tmp_path) -> None:
    store = tmp_path / "empty.zarr"
    empty = make_compact([])
    with RigidMoleculeASUZarrWriter(store) as writer:
        writer.write(empty)
    with RigidMoleculeASUZarrReader(store) as reader:
        assert len(reader) == 0
        assert reader.read().num_structures == 0
        assert reader.read(torch.empty(0, dtype=torch.int64)).num_structures == 0
        assert reader.read_batch(torch.empty(0, dtype=torch.int64)).num_graphs == 0
        assert reader.get_metadata(torch.empty(0, dtype=torch.int64)).shape == (0, 2)


def test_reader_rejects_invalid_field_alignment(tmp_path) -> None:
    store = tmp_path / "invalid.zarr"
    RigidMoleculeASUZarrWriter(store).write(make_compact())
    root = zarr.open_group(store, mode="r+")
    fields = dict(root.attrs["fields"])
    fields["core"] = {**fields["core"], "cells": "molecule"}
    root.attrs["fields"] = fields
    with pytest.raises(ValueError, match="alignment"):
        RigidMoleculeASUZarrReader(store)
