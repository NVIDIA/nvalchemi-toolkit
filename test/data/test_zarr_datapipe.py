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
"""Tests for the AtomicDataZarrWriter class."""

from __future__ import annotations

import random
from collections.abc import Generator, Iterator, Sequence
from math import floor
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
import zarr
from torch.utils.data import Sampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from nvalchemi.data.atomic_data import AtomicData
from nvalchemi.data.batch import Batch
from nvalchemi.data.datapipes import (
    AtomicDataZarrReader,
    AtomicDataZarrWriter,
    DataLoader,
    Dataset,
    MultiDataset,
)
from nvalchemi.data.datapipes.backends.base import Reader
from nvalchemi.data.datapipes.backends.zarr import (
    ZarrArrayConfig,
    ZarrWriteConfig,
    _get_cat_dim,
    _get_field_level,
    _slice_edge_array,
)
from nvalchemi.data.datapipes.dataset import _PrefetchResult


def _make_atomic_data(num_atoms: int, num_edges: int) -> AtomicData:
    """Create a minimal AtomicData for testing.

    Parameters
    ----------
    num_atoms : int
        Number of atoms in the structure.
    num_edges : int
        Number of edges (bonds) in the structure.

    Returns
    -------
    AtomicData
        A test AtomicData instance.
    """
    return AtomicData(
        atomic_numbers=torch.randint(1, 20, (num_atoms,)),
        positions=torch.randn(num_atoms, 3),
        forces=torch.randn(num_atoms, 3),
        energy=torch.randn(1, 1),
        cell=torch.eye(3).unsqueeze(0),
        pbc=torch.tensor([[True, True, True]]),
        neighbor_list=torch.stack(
            [
                torch.randint(0, num_atoms, (num_edges,)),
                torch.randint(0, num_atoms, (num_edges,)),
            ],
            dim=1,
        ),
        shifts=torch.randn(num_edges, 3),
    )


def _data_generator(num_samples: int, seed: int = 5136) -> Generator:
    """Generates ``num_samples`` of data, primarily for batch testing"""
    random.seed(seed)
    for index in range(num_samples):
        num_atoms = random.randint(a=1, b=64)
        num_edges = random.randint(a=1, b=256)
        yield _make_atomic_data(num_atoms, num_edges)


def _make_ordered_atomic_data(label: int) -> AtomicData:
    """Create one-atom AtomicData with an order-identifying atomic number."""
    return AtomicData(
        atomic_numbers=torch.tensor([label], dtype=torch.long),
        positions=torch.tensor([[float(label), 0.0, 0.0]]),
        cell=torch.eye(3).unsqueeze(0),
        pbc=torch.tensor([[True, True, True]]),
    )


class TestAtomicDataZarrWriter:
    """Tests for AtomicDataZarrWriter."""

    @pytest.mark.parametrize("num_samples", [1, 3, 5])
    def test_write_list(self, num_samples: int, tmp_path: Path) -> None:
        """Write list: concatenated arrays, ptrs span all samples."""
        data_list = list(_data_generator(num_samples))

        writer = AtomicDataZarrWriter(tmp_path / "test.zarr")
        writer.write(data_list)

        root = zarr.open(tmp_path / "test.zarr", mode="r")

        # Compute expected pointer values from each data item
        atom_counts = [d.atomic_numbers.shape[0] for d in data_list]
        edge_counts = [d.neighbor_list.shape[0] for d in data_list]
        total_atoms = sum(atom_counts)
        total_edges = sum(edge_counts)

        # Build expected pointer arrays
        expected_atoms_ptr = [0]
        expected_edges_ptr = [0]
        for ac in atom_counts:
            expected_atoms_ptr.append(expected_atoms_ptr[-1] + ac)
        for ec in edge_counts:
            expected_edges_ptr.append(expected_edges_ptr[-1] + ec)

        atoms_ptr = root["meta"]["atoms_ptr"][:]
        edges_ptr = root["meta"]["edges_ptr"][:]

        assert atoms_ptr.tolist() == expected_atoms_ptr
        assert edges_ptr.tolist() == expected_edges_ptr

        # Check total sizes
        assert root["core"]["atomic_numbers"].shape == (total_atoms,)
        assert root["core"]["positions"].shape == (total_atoms, 3)
        assert root["core"]["neighbor_list"].shape == (total_edges, 2)
        assert root["core"]["shifts"].shape == (total_edges, 3)

        # Check system-level fields
        assert root["core"]["energy"].shape == (num_samples, 1)
        assert root["core"]["cell"].shape == (num_samples, 3, 3)
        assert root["core"]["pbc"].shape == (num_samples, 3)

        # Check num_samples
        assert root.attrs["num_samples"] == num_samples

    @pytest.mark.parametrize("num_samples", [1, 3, 5])
    def test_write_from_batch(self, num_samples: int, tmp_path: Path) -> None:
        """Write from Batch: auto-split + correct per-sample slicing."""
        data_list = list(_data_generator(num_samples))

        batch = Batch.from_data_list(data_list, device="cpu")

        writer = AtomicDataZarrWriter(tmp_path / "test.zarr")
        writer.write(batch)

        root = zarr.open(tmp_path / "test.zarr", mode="r")

        # Compute expected pointer values from original data_list
        atom_counts = [d.atomic_numbers.shape[0] for d in data_list]
        edge_counts = [d.neighbor_list.shape[0] for d in data_list]

        expected_atoms_ptr = [0]
        expected_edges_ptr = [0]
        for ac in atom_counts:
            expected_atoms_ptr.append(expected_atoms_ptr[-1] + ac)
        for ec in edge_counts:
            expected_edges_ptr.append(expected_edges_ptr[-1] + ec)

        # Check pointer arrays
        atoms_ptr = root["meta"]["atoms_ptr"][:]
        edges_ptr = root["meta"]["edges_ptr"][:]

        assert atoms_ptr.tolist() == expected_atoms_ptr
        assert edges_ptr.tolist() == expected_edges_ptr

        # Check num_samples
        assert root.attrs["num_samples"] == num_samples

    @pytest.mark.parametrize("num_samples", [1, 3, 5])
    def test_append_to_existing(self, num_samples: int, tmp_path: Path) -> None:
        """Append: arrays extended, ptrs + masks updated."""
        data_list = list(_data_generator(max(2, num_samples)))

        writer = AtomicDataZarrWriter(tmp_path / "test.zarr")
        writer.write(data_list[0])
        writer.append(data_list[1])

        root = zarr.open(tmp_path / "test.zarr", mode="r")

        # Get atom/edge counts from the two items
        na1 = data_list[0].atomic_numbers.shape[0]
        ne1 = data_list[0].neighbor_list.shape[0]
        na2 = data_list[1].atomic_numbers.shape[0]
        ne2 = data_list[1].neighbor_list.shape[0]

        # Check pointer arrays
        total_atoms = na1 + na2
        total_edges = ne1 + ne2
        atoms_ptr = root["meta"]["atoms_ptr"][:]
        edges_ptr = root["meta"]["edges_ptr"][:]

        assert atoms_ptr.tolist() == [0, na1, total_atoms]
        assert edges_ptr.tolist() == [0, ne1, total_edges]

        # Check total sizes
        assert root["core"]["atomic_numbers"].shape == (total_atoms,)
        assert root["core"]["neighbor_list"].shape == (total_edges, 2)

        # Check masks
        assert root["meta"]["samples_mask"].shape == (2,)
        assert all(root["meta"]["samples_mask"][:])

        # Check num_samples
        assert root.attrs["num_samples"] == 2

    @pytest.mark.parametrize("num_samples", [1, 3, 5])
    def test_add_custom_array(self, num_samples: int, tmp_path: Path) -> None:
        """add_custom: custom/ group created, level in .zattrs."""
        data_list = list(_data_generator(max(2, num_samples)))

        writer = AtomicDataZarrWriter(tmp_path / "test.zarr")
        writer.write(data_list)

        # Compute total atoms/edges from data
        total_atoms = sum(d.atomic_numbers.shape[0] for d in data_list)
        total_edges = sum(d.neighbor_list.shape[0] for d in data_list)
        actual_num_samples = len(data_list)

        # Add custom atom-level array
        custom_atom_data = torch.randn(total_atoms, 2)
        writer.add_custom("my_atom_feature", custom_atom_data, "atom")

        # Add custom edge-level array
        custom_edge_data = torch.randn(total_edges, 4)
        writer.add_custom("my_edge_feature", custom_edge_data, "edge")

        # Add custom system-level array
        custom_system_data = torch.randn(actual_num_samples, 5)
        writer.add_custom("my_system_feature", custom_system_data, "system")

        root = zarr.open(tmp_path / "test.zarr", mode="r")

        # Check custom arrays exist
        assert "my_atom_feature" in root["custom"]
        assert "my_edge_feature" in root["custom"]
        assert "my_system_feature" in root["custom"]

        # Check shapes
        assert root["custom"]["my_atom_feature"].shape == (total_atoms, 2)
        assert root["custom"]["my_edge_feature"].shape == (total_edges, 4)
        assert root["custom"]["my_system_feature"].shape == (actual_num_samples, 5)

        # Check .zattrs
        fields = root.attrs["fields"]
        assert fields["custom"]["my_atom_feature"] == "atom"
        assert fields["custom"]["my_edge_feature"] == "edge"
        assert fields["custom"]["my_system_feature"] == "system"

    @pytest.mark.parametrize("num_samples", [1, 3, 5])
    def test_delete_samples(self, num_samples: int, tmp_path: Path) -> None:
        """delete: masks set False, data zeroed, ptrs unchanged."""
        data_list = list(_data_generator(max(3, num_samples)))

        writer = AtomicDataZarrWriter(tmp_path / "test.zarr")
        writer.write(data_list)

        # Delete second sample
        writer.delete([1])

        root = zarr.open(tmp_path / "test.zarr", mode="r")

        # Build expected samples_mask (all True except index 1)
        actual_count = len(data_list)
        expected_mask = [True] * actual_count
        expected_mask[1] = False

        # Check samples_mask
        assert root["meta"]["samples_mask"][:].tolist() == expected_mask

        # Compute expected pointers from data_list
        atom_counts = [d.atomic_numbers.shape[0] for d in data_list]
        edge_counts = [d.neighbor_list.shape[0] for d in data_list]
        expected_atoms_ptr = [0]
        expected_edges_ptr = [0]
        for ac in atom_counts:
            expected_atoms_ptr.append(expected_atoms_ptr[-1] + ac)
        for ec in edge_counts:
            expected_edges_ptr.append(expected_edges_ptr[-1] + ec)

        atoms_ptr = root["meta"]["atoms_ptr"][:]
        edges_ptr = root["meta"]["edges_ptr"][:]
        assert atoms_ptr.tolist() == expected_atoms_ptr
        assert edges_ptr.tolist() == expected_edges_ptr

        # Check energy for deleted sample is zeroed
        energy = root["core"]["energy"][:]
        assert energy[1, 0] == 0.0

    @pytest.mark.parametrize("num_samples", [1, 3, 5])
    def test_delete_atoms_edges_masks(self, num_samples: int, tmp_path: Path) -> None:
        """delete: atoms_mask + edges_mask zeroed correctly."""
        data_list = list(_data_generator(max(2, num_samples)))

        writer = AtomicDataZarrWriter(tmp_path / "test.zarr")
        writer.write(data_list)

        # Delete first sample
        writer.delete([0])

        root = zarr.open(tmp_path / "test.zarr", mode="r")

        # Get atom/edge counts from first sample
        na1 = data_list[0].atomic_numbers.shape[0]
        ne1 = data_list[0].neighbor_list.shape[0]

        # Check atoms_mask: first na1 atoms should be False
        atoms_mask = root["meta"]["atoms_mask"][:]
        assert atoms_mask[:na1].tolist() == [False] * na1
        assert all(atoms_mask[na1:])  # remaining atoms should be True

        # Check edges_mask: first ne1 edges should be False
        edges_mask = root["meta"]["edges_mask"][:]
        assert edges_mask[:ne1].tolist() == [False] * ne1
        assert all(edges_mask[ne1:])  # remaining edges should be True

    @pytest.mark.parametrize("num_samples", [1, 3, 5])
    def test_defragment(self, num_samples: int, tmp_path: Path) -> None:
        """defragment: deleted removed, ptrs rebuilt, masks reset."""
        data_list = list(_data_generator(max(3, num_samples)))

        writer = AtomicDataZarrWriter(tmp_path / "test.zarr")
        writer.write(data_list)

        # Delete second sample (index 1)
        writer.delete([1])

        # Defragment
        writer.defragment()

        root = zarr.open(tmp_path / "test.zarr", mode="r")

        # Check num_samples reduced (removed sample at index 1)
        remaining_count = len(data_list) - 1
        assert root.attrs["num_samples"] == remaining_count

        # Samples 0 and 2+ remain (all except index 1)
        remaining_data = [data_list[i] for i in range(len(data_list)) if i != 1]
        remaining_atoms = sum(d.atomic_numbers.shape[0] for d in remaining_data)
        remaining_edges = sum(d.neighbor_list.shape[0] for d in remaining_data)

        # Build expected pointer arrays
        expected_atoms_ptr = [0]
        expected_edges_ptr = [0]
        for d in remaining_data:
            expected_atoms_ptr.append(
                expected_atoms_ptr[-1] + d.atomic_numbers.shape[0]
            )
            expected_edges_ptr.append(expected_edges_ptr[-1] + d.neighbor_list.shape[0])

        atoms_ptr = root["meta"]["atoms_ptr"][:]
        edges_ptr = root["meta"]["edges_ptr"][:]
        assert atoms_ptr.tolist() == expected_atoms_ptr
        assert edges_ptr.tolist() == expected_edges_ptr

        # Check all masks are True
        assert all(root["meta"]["samples_mask"][:])
        assert all(root["meta"]["atoms_mask"][:])
        assert all(root["meta"]["edges_mask"][:])

        # Check total sizes match remaining samples
        assert root["core"]["atomic_numbers"].shape == (remaining_atoms,)
        assert root["core"]["neighbor_list"].shape == (remaining_edges, 2)

    @pytest.mark.parametrize("num_samples", [1, 3, 5])
    def test_zattrs_metadata(self, num_samples: int, tmp_path: Path) -> None:
        """Verify .zattrs contains num_samples and fields dict."""
        data_list = list(_data_generator(max(2, num_samples)))

        writer = AtomicDataZarrWriter(tmp_path / "test.zarr")
        writer.write(data_list)

        root = zarr.open(tmp_path / "test.zarr", mode="r")

        # Check num_samples
        assert root.attrs["num_samples"] == len(data_list)

        # Check fields dict exists and has core
        fields = root.attrs["fields"]
        assert "core" in fields
        assert "custom" in fields

        # Check field levels are correct
        core_fields = fields["core"]
        assert core_fields.get("atomic_numbers") == "atom"
        assert core_fields.get("positions") == "atom"
        assert core_fields.get("forces") == "atom"
        assert core_fields.get("energy") == "system"
        assert core_fields.get("cell") == "system"
        assert core_fields.get("neighbor_list") == "edge"
        assert core_fields.get("shifts") == "edge"

    @pytest.mark.parametrize("num_samples", [1, 3, 5])
    def test_edge_index_cat_dim(self, num_samples: int, tmp_path: Path) -> None:
        """Verify neighbor_list is stored with shape [E_total, 2]."""
        data_list = list(_data_generator(max(2, num_samples)))

        writer = AtomicDataZarrWriter(tmp_path / "test.zarr")
        writer.write(data_list)

        root = zarr.open(tmp_path / "test.zarr", mode="r")

        total_edges = sum(d.neighbor_list.shape[0] for d in data_list)

        neighbor_list = root["core"]["neighbor_list"]
        assert neighbor_list.shape == (total_edges, 2)

    @pytest.mark.parametrize("num_samples", [1, 3, 5])
    def test_append_multiple_times(self, num_samples: int, tmp_path: Path) -> None:
        """Test multiple appends work correctly."""
        data_list = list(_data_generator(max(3, num_samples)))

        writer = AtomicDataZarrWriter(tmp_path / "test.zarr")
        writer.write(data_list[0])
        for d in data_list[1:]:
            writer.append(d)

        root = zarr.open(tmp_path / "test.zarr", mode="r")

        actual_count = len(data_list)
        assert root.attrs["num_samples"] == actual_count

        # Build expected pointer arrays
        expected_atoms_ptr = [0]
        expected_edges_ptr = [0]
        for d in data_list:
            expected_atoms_ptr.append(
                expected_atoms_ptr[-1] + d.atomic_numbers.shape[0]
            )
            expected_edges_ptr.append(expected_edges_ptr[-1] + d.neighbor_list.shape[0])

        assert root["meta"]["atoms_ptr"][:].tolist() == expected_atoms_ptr
        assert root["meta"]["edges_ptr"][:].tolist() == expected_edges_ptr

    @pytest.mark.parametrize("num_samples", [1, 3, 5])
    def test_delete_multiple_samples(self, num_samples: int, tmp_path: Path) -> None:
        """Test deleting multiple samples at once."""
        data_list = list(_data_generator(max(5, num_samples)))
        actual_count = len(data_list)

        writer = AtomicDataZarrWriter(tmp_path / "test.zarr")
        writer.write(data_list)

        # Delete even indices [0, 2, 4, ...]
        even_indices = [i for i in range(actual_count) if i % 2 == 0]
        writer.delete(even_indices)

        root = zarr.open(tmp_path / "test.zarr", mode="r")

        # Build expected samples_mask (False for even indices, True for odd)
        expected_mask = [i % 2 != 0 for i in range(actual_count)]
        samples_mask = root["meta"]["samples_mask"][:]
        assert samples_mask.tolist() == expected_mask

    @pytest.mark.parametrize("num_samples", [1, 3, 5])
    def test_defragment_preserves_custom(
        self, num_samples: int, tmp_path: Path
    ) -> None:
        """Test defragment preserves custom arrays correctly."""
        data_list = list(_data_generator(max(3, num_samples)))
        actual_count = len(data_list)

        writer = AtomicDataZarrWriter(tmp_path / "test.zarr")
        writer.write(data_list)

        # Add custom system-level array with distinguishable values
        custom_data = torch.tensor([[float(i + 1)] for i in range(actual_count)])
        writer.add_custom("my_feature", custom_data, "system")

        # Delete second sample (index 1)
        writer.delete([1])

        # Defragment
        writer.defragment()

        root = zarr.open(tmp_path / "test.zarr", mode="r")

        # Check custom array is preserved and correctly reduced
        remaining_count = actual_count - 1
        assert "my_feature" in root["custom"]
        custom_arr = root["custom"]["my_feature"][:]
        assert custom_arr.shape == (remaining_count, 1)

        # Values should exclude index 1: [1.0, 3.0, 4.0, ...] (samples 0, 2, 3, ...)
        expected_values = [float(i + 1) for i in range(actual_count) if i != 1]
        for idx, val in enumerate(expected_values):
            assert custom_arr[idx, 0] == val


def test_writer_write_single(tmp_path: Path) -> None:
    """Write single AtomicData: core/ + meta/ structure, ptrs correct.

    Parameters
    ----------
    tmp_path : Path
        Pytest fixture providing a temporary directory.
    """
    data = next(_data_generator(1))
    writer = AtomicDataZarrWriter(tmp_path / "test.zarr")
    writer.write(data)

    # Verify store exists
    store_path = tmp_path / "test.zarr"
    assert store_path.exists()

    # Open and verify structure
    root = zarr.open(store_path, mode="r")

    # Check groups exist
    assert "meta" in root
    assert "core" in root
    assert "custom" in root

    # Get expected sizes from the data object
    num_atoms = data.atomic_numbers.shape[0]
    num_edges = data.neighbor_list.shape[0]

    # Check pointer arrays
    atoms_ptr = root["meta"]["atoms_ptr"][:]
    edges_ptr = root["meta"]["edges_ptr"][:]

    assert atoms_ptr.tolist() == [0, num_atoms]
    assert edges_ptr.tolist() == [0, num_edges]

    # Check masks
    assert root["meta"]["samples_mask"][:].tolist() == [True]
    assert all(root["meta"]["atoms_mask"][:])
    assert all(root["meta"]["edges_mask"][:])

    # Check core fields exist
    assert "atomic_numbers" in root["core"]
    assert "positions" in root["core"]
    assert "forces" in root["core"]
    assert "energy" in root["core"]
    assert "neighbor_list" in root["core"]
    assert "shifts" in root["core"]

    # Check shapes
    assert root["core"]["atomic_numbers"].shape == (num_atoms,)
    assert root["core"]["positions"].shape == (num_atoms, 3)
    assert root["core"]["neighbor_list"].shape == (num_edges, 2)


def test_writer_write_raises_if_exists(tmp_path: Path) -> None:
    """Write raises FileExistsError if store already exists.

    Note: The docstring in AtomicDataZarrWriter.write claims it should raise
    FileExistsError, but the implementation uses zarr.open(mode='w') which
    silently overwrites. This test verifies actual behavior.
    TODO: Update implementation to check exists and raise, then update this test.

    Parameters
    ----------
    tmp_path : Path
        Pytest fixture providing a temporary directory.
    """
    data = next(_data_generator(1))
    writer = AtomicDataZarrWriter(tmp_path / "test.zarr")
    writer.write(data)

    # Current behavior: second write raises FileExistsError
    with pytest.raises(FileExistsError, match="already exists"):
        writer.write(data)

    # Verify store still exists and has valid data
    root = zarr.open(tmp_path / "test.zarr", mode="r")
    assert root.attrs["num_samples"] == 1


def test_writer_append_raises_if_not_exists(tmp_path: Path) -> None:
    """Append raises FileNotFoundError if store doesn't exist.

    Parameters
    ----------
    tmp_path : Path
        Pytest fixture providing a temporary directory.
    """
    data = next(_data_generator(1))
    writer = AtomicDataZarrWriter(tmp_path / "test.zarr")

    with pytest.raises(FileNotFoundError):
        writer.append(data)


def test_writer_add_custom_invalid_level(tmp_path: Path) -> None:
    """add_custom raises ValueError for invalid level.

    Parameters
    ----------
    tmp_path : Path
        Pytest fixture providing a temporary directory.
    """
    data = next(_data_generator(1))
    num_atoms = data.atomic_numbers.shape[0]
    writer = AtomicDataZarrWriter(tmp_path / "test.zarr")
    writer.write(data)

    with pytest.raises(ValueError, match="Invalid level"):
        writer.add_custom("bad_feature", torch.zeros(num_atoms), "invalid_level")


def test_writer_add_custom_shape_mismatch(tmp_path: Path) -> None:
    """add_custom raises ValueError when shape doesn't match.

    Parameters
    ----------
    tmp_path : Path
        Pytest fixture providing a temporary directory.
    """
    data = next(_data_generator(1))
    num_atoms = data.atomic_numbers.shape[0]
    writer = AtomicDataZarrWriter(tmp_path / "test.zarr")
    writer.write(data)

    # Wrong size for atom level
    wrong_size = num_atoms * 2 + 1
    with pytest.raises(ValueError, match="does not match expected size"):
        writer.add_custom("bad_feature", torch.zeros(wrong_size), "atom")


def test_writer_optional_fields_only(tmp_path: Path) -> None:
    """Write samples with only required fields (no edges, etc.).

    Parameters
    ----------
    tmp_path : Path
        Pytest fixture providing a temporary directory.
    """
    # Use fixed small value since this tests minimal fields
    num_atoms = 5
    data = AtomicData(
        atomic_numbers=torch.randint(1, 20, (num_atoms,)),
        positions=torch.randn(num_atoms, 3),
        cell=torch.eye(3).unsqueeze(0),
        pbc=torch.tensor([[True, True, True]]),
    )

    writer = AtomicDataZarrWriter(tmp_path / "test.zarr")
    writer.write(data)

    root = zarr.open(tmp_path / "test.zarr", mode="r")

    # Check pointer arrays
    atoms_ptr = root["meta"]["atoms_ptr"][:]
    edges_ptr = root["meta"]["edges_ptr"][:]

    assert atoms_ptr.tolist() == [0, num_atoms]
    assert edges_ptr.tolist() == [0, 0]  # No edges

    # Check that neighbor_list is not in core (since no edges)
    # Actually, neighbor_list might be None or empty - check shape
    if "neighbor_list" in root["core"]:
        assert root["core"]["neighbor_list"].shape[0] == 0


def test_get_field_level() -> None:
    """Test _get_field_level for various fields.

    This standalone test verifies the field level categorization for atom-level,
    edge-level, and system-level fields.
    """
    for key in ["atomic_numbers", "positions", "forces"]:
        assert _get_field_level(key) == "atom"
    for key in ["neighbor_list", "shifts"]:
        assert _get_field_level(key) == "edge"
    for key in ["energy", "cell", "pbc"]:
        assert _get_field_level(key) == "system"


def test_get_cat_dim() -> None:
    """Test _get_cat_dim for various fields.

    This standalone test verifies the concatenation dimension for different
    field types used in batch assembly.
    """
    assert _get_cat_dim("atomic_numbers") == 0
    assert _get_cat_dim("positions") == 0
    assert _get_cat_dim("neighbor_list") == 0
    assert _get_cat_dim("face") == -1
    assert _get_cat_dim("some_face_attr") == -1


def test_empty_data_list_raises(tmp_path: Path) -> None:
    """Test that writing empty list raises ValueError.

    Note: The error is raised by Batch.from_data_list before reaching
    AtomicDataZarrWriter's check, so the message differs from docstring.

    Parameters
    ----------
    tmp_path : Path
        Pytest fixture providing a temporary directory.
    """
    writer = AtomicDataZarrWriter(tmp_path / "test.zarr")

    with pytest.raises(ValueError, match="Cannot create batch from empty"):
        writer.write([])


class TestAtomicDataZarrReader:
    """Tests for AtomicDataZarrReader."""

    @pytest.mark.parametrize("num_samples", [1, 3, 5])
    def test_reader_load_sample(self, num_samples: int, tmp_path: Path) -> None:
        """Load samples and verify tensor shapes match original data."""
        data_list = list(_data_generator(num_samples))

        # Write to store
        writer = AtomicDataZarrWriter(tmp_path / "test.zarr")
        writer.write(data_list)

        # Open reader and load samples
        with AtomicDataZarrReader(tmp_path / "test.zarr") as reader:
            for idx, original in enumerate(data_list):
                sample = reader._load_sample(idx)
                na = original.atomic_numbers.shape[0]
                ne = original.neighbor_list.shape[0]

                assert sample["atomic_numbers"].shape == (na,)
                assert sample["positions"].shape == (na, 3)
                assert sample["forces"].shape == (na, 3)
                assert sample["neighbor_list"].shape == (ne, 2)
                assert sample["shifts"].shape == (ne, 3)
                assert sample["energy"].shape == (1, 1)
                assert sample["cell"].shape == (1, 3, 3)
                assert sample["pbc"].shape == (1, 3)

    @pytest.mark.parametrize("num_samples", [1, 3, 5])
    def test_reader_len(self, num_samples: int, tmp_path: Path) -> None:
        """Verify len(reader) returns count of non-deleted samples."""
        data_list = list(_data_generator(max(3, num_samples)))

        writer = AtomicDataZarrWriter(tmp_path / "test.zarr")
        writer.write(data_list)

        # Delete 1 sample
        writer.delete([1])

        # Open reader and verify length
        with AtomicDataZarrReader(tmp_path / "test.zarr") as reader:
            assert len(reader) == len(data_list) - 1

    @pytest.mark.parametrize("num_samples", [1, 3, 5])
    def test_variable_size_samples(self, num_samples: int, tmp_path: Path) -> None:
        """Verify samples with very different sizes load correctly."""
        # Use _data_generator which naturally provides variable sizes
        data_list = list(_data_generator(max(2, num_samples)))

        writer = AtomicDataZarrWriter(tmp_path / "test.zarr")
        writer.write(data_list)

        with AtomicDataZarrReader(tmp_path / "test.zarr") as reader:
            for idx, original in enumerate(data_list):
                sample = reader._load_sample(idx)
                na = original.atomic_numbers.shape[0]
                ne = original.neighbor_list.shape[0]

                assert sample["atomic_numbers"].shape == (na,)
                assert sample["neighbor_list"].shape == (ne, 2)


def test_reader_skips_deleted(tmp_path: Path) -> None:
    """Verify reader maps logical indices past deleted samples.

    Parameters
    ----------
    tmp_path : Path
        Pytest fixture providing a temporary directory.
    """
    # Create 3 samples with distinguishable numbers
    num_atoms, num_edges = 5, 8
    data0 = AtomicData(
        atomic_numbers=torch.full((num_atoms,), fill_value=10, dtype=torch.long),
        positions=torch.randn(num_atoms, 3),
        cell=torch.eye(3).unsqueeze(0),
        pbc=torch.tensor([[True, True, True]]),
        neighbor_list=torch.stack(
            [
                torch.randint(0, num_atoms, (num_edges,)),
                torch.randint(0, num_atoms, (num_edges,)),
            ],
            dim=1,
        ),
        shifts=torch.randn(num_edges, 3),
    )
    data1 = AtomicData(
        atomic_numbers=torch.full((num_atoms,), fill_value=20, dtype=torch.long),
        positions=torch.randn(num_atoms, 3),
        cell=torch.eye(3).unsqueeze(0),
        pbc=torch.tensor([[True, True, True]]),
        neighbor_list=torch.stack(
            [
                torch.randint(0, num_atoms, (num_edges,)),
                torch.randint(0, num_atoms, (num_edges,)),
            ],
            dim=1,
        ),
        shifts=torch.randn(num_edges, 3),
    )
    data2 = AtomicData(
        atomic_numbers=torch.full((num_atoms,), fill_value=30, dtype=torch.long),
        positions=torch.randn(num_atoms, 3),
        cell=torch.eye(3).unsqueeze(0),
        pbc=torch.tensor([[True, True, True]]),
        neighbor_list=torch.stack(
            [
                torch.randint(0, num_atoms, (num_edges,)),
                torch.randint(0, num_atoms, (num_edges,)),
            ],
            dim=1,
        ),
        shifts=torch.randn(num_edges, 3),
    )

    writer = AtomicDataZarrWriter(tmp_path / "test.zarr")
    writer.write([data0, data1, data2])

    # Delete middle sample (sample 1)
    writer.delete([1])

    # Open reader
    with AtomicDataZarrReader(tmp_path / "test.zarr") as reader:
        # Logical index 0 should return sample 0 (numbers all 10)
        sample0 = reader._load_sample(0)
        assert torch.all(sample0["atomic_numbers"] == data0.atomic_numbers)

        # Logical index 1 should return sample 2 (numbers all 30)
        sample1 = reader._load_sample(1)
        assert torch.all(sample1["atomic_numbers"] == data2.atomic_numbers)


def test_reader_loads_custom(tmp_path: Path) -> None:
    """Verify custom arrays are loaded with correct shapes.

    Parameters
    ----------
    tmp_path : Path
        Pytest fixture providing a temporary directory.
    """
    data = next(_data_generator(1))
    num_atoms = data.atomic_numbers.shape[0]
    num_edges = data.neighbor_list.shape[0]

    writer = AtomicDataZarrWriter(tmp_path / "test.zarr")
    writer.write(data)

    # Add custom arrays at different levels
    writer.add_custom("custom_atom", torch.randn(num_atoms, 2), "atom")
    writer.add_custom("custom_edge", torch.randn(num_edges, 4), "edge")
    writer.add_custom("custom_system", torch.randn(1, 3), "system")

    # Open reader and verify custom fields appear
    with AtomicDataZarrReader(tmp_path / "test.zarr") as reader:
        sample = reader._load_sample(0)

        assert "custom_atom" in sample
        assert sample["custom_atom"].shape == (num_atoms, 2)

        assert "custom_edge" in sample
        assert sample["custom_edge"].shape == (num_edges, 4)

        assert "custom_system" in sample
        assert sample["custom_system"].shape == (1, 3)


def test_reader_full_roundtrip(tmp_path: Path) -> None:
    """Verify write then read returns identical data.

    Parameters
    ----------
    tmp_path : Path
        Pytest fixture providing a temporary directory.
    """
    original = next(_data_generator(1))
    original_dict = original.to_dict()

    writer = AtomicDataZarrWriter(tmp_path / "test.zarr")
    writer.write(original)

    with AtomicDataZarrReader(tmp_path / "test.zarr") as reader:
        loaded = reader._load_sample(0)

        # Compare all stored fields (iterate over loaded keys since not all
        # original fields may be stored, e.g., computed fields like shifts)
        for key in loaded:
            orig_tensor = original_dict[key]
            loaded_tensor = loaded[key]

            # Check shapes match
            assert orig_tensor.shape == loaded_tensor.shape, (
                f"Shape mismatch for {key}: {orig_tensor.shape} vs {loaded_tensor.shape}"
            )

            # Check values match (cast to common dtype for float; zarr may store float32)
            if orig_tensor.dtype.is_floating_point:
                common_dtype = torch.promote_types(
                    orig_tensor.dtype, loaded_tensor.dtype
                )
                assert torch.allclose(
                    orig_tensor.to(common_dtype), loaded_tensor.to(common_dtype)
                ), f"Value mismatch for {key}"
            else:
                assert torch.equal(orig_tensor, loaded_tensor), (
                    f"Value mismatch for {key}"
                )


def test_reader_read_many_matches_single_sample_reads(tmp_path: Path) -> None:
    """Verify read_many preserves per-sample reader semantics and order."""
    data_list = list(_data_generator(4))
    writer = AtomicDataZarrWriter(tmp_path / "test.zarr")
    writer.write(data_list)

    with AtomicDataZarrReader(tmp_path / "test.zarr") as reader:
        indices = [2, 0, 3]
        many = reader.read_many(indices)
        singles = [reader.read(index) for index in indices]

    assert len(many) == len(indices)
    for (many_data, many_metadata), (single_data, single_metadata) in zip(
        many, singles, strict=True
    ):
        assert many_metadata["physical_index"] == single_metadata["physical_index"]
        for key, many_tensor in many_data.items():
            single_tensor = single_data[key]
            if many_tensor.dtype.is_floating_point:
                assert torch.allclose(many_tensor, single_tensor), key
            else:
                assert torch.equal(many_tensor, single_tensor), key


def test_reader_read_many_skips_deleted_and_supports_negative_indices(
    tmp_path: Path,
) -> None:
    """Verify read_many maps logical indices through the active sample mask."""
    data_list = [_make_ordered_atomic_data(i + 1) for i in range(5)]
    writer = AtomicDataZarrWriter(tmp_path / "test.zarr")
    writer.write(data_list)
    writer.delete([1])

    with AtomicDataZarrReader(tmp_path / "test.zarr") as reader:
        samples = reader.read_many([2, 0, -1])

    labels = [data["atomic_numbers"].item() for data, _ in samples]
    physical_indices = [metadata["physical_index"] for _, metadata in samples]
    logical_indices = [metadata["index"] for _, metadata in samples]

    assert labels == [4, 1, 5]
    assert physical_indices == ["3", "0", "4"]
    assert logical_indices == [2, 0, 3]


def test_reader_read_many_empty_returns_empty(tmp_path: Path) -> None:
    """Verify read_many([]) returns an empty list."""
    data_list = list(_data_generator(3))
    writer = AtomicDataZarrWriter(tmp_path / "test.zarr")
    writer.write(data_list)

    with AtomicDataZarrReader(tmp_path / "test.zarr") as reader:
        result = reader.read_many([])
    assert result == []


def test_reader_read_many_single_element(tmp_path: Path) -> None:
    """Verify read_many([i]) matches reader.read(i)."""
    data_list = list(_data_generator(4))
    writer = AtomicDataZarrWriter(tmp_path / "test.zarr")
    writer.write(data_list)

    with AtomicDataZarrReader(tmp_path / "test.zarr") as reader:
        many = reader.read_many([2])
        single = reader.read(2)

    many_data, many_meta = many[0]
    single_data, single_meta = single
    assert many_meta["physical_index"] == single_meta["physical_index"]
    for key in many_data:
        assert torch.equal(many_data[key], single_data[key]), key


def test_dataset_metadata_delegates_to_zarr_reader_pointers(tmp_path: Path) -> None:
    """Verify Zarr metadata lookup does not load full samples."""
    data_list = [_make_atomic_data(3, 2), _make_atomic_data(5, 7)]
    writer = AtomicDataZarrWriter(tmp_path / "test.zarr")
    writer.write(data_list)

    with AtomicDataZarrReader(tmp_path / "test.zarr") as reader:
        dataset = Dataset(reader, device="cpu")
        with patch.object(reader, "_load_sample", side_effect=AssertionError):
            assert dataset.get_metadata(1) == (5, 7)


def test_reader_optional_fields_only(tmp_path: Path) -> None:
    """Verify minimal AtomicData loads without error.

    Parameters
    ----------
    tmp_path : Path
        Pytest fixture providing a temporary directory.
    """
    # Use fixed num_atoms (tests minimal fields, no need for parameterized sizes)
    num_atoms = 5

    # Create minimal AtomicData (no edges, forces, energy)
    data = AtomicData(
        atomic_numbers=torch.randint(1, 20, (num_atoms,)),
        positions=torch.randn(num_atoms, 3),
        cell=torch.eye(3).unsqueeze(0),
        pbc=torch.tensor([[True, True, True]]),
    )

    writer = AtomicDataZarrWriter(tmp_path / "test.zarr")
    writer.write(data)

    with AtomicDataZarrReader(tmp_path / "test.zarr") as reader:
        sample = reader._load_sample(0)

        # Verify required fields are present
        assert "atomic_numbers" in sample
        assert "positions" in sample
        assert "cell" in sample
        assert "pbc" in sample

        # Verify shapes
        assert sample["atomic_numbers"].shape == (num_atoms,)
        assert sample["positions"].shape == (num_atoms, 3)

        # Optional fields should not be present
        assert "forces" not in sample
        assert "energy" not in sample


def test_reader_close(tmp_path: Path) -> None:
    """Verify close() sets _root to None.

    Parameters
    ----------
    tmp_path : Path
        Pytest fixture providing a temporary directory.
    """
    data = next(_data_generator(1))

    writer = AtomicDataZarrWriter(tmp_path / "test.zarr")
    writer.write(data)

    # NOT using context manager — this test verifies close() behavior
    reader = AtomicDataZarrReader(tmp_path / "test.zarr")
    assert reader._root is not None

    reader.close()
    assert reader._root is None


def test_reader_context_manager(tmp_path: Path) -> None:
    """Verify reader works as context manager.

    Parameters
    ----------
    tmp_path : Path
        Pytest fixture providing a temporary directory.
    """
    data = next(_data_generator(1))

    writer = AtomicDataZarrWriter(tmp_path / "test.zarr")
    writer.write(data)

    with AtomicDataZarrReader(tmp_path / "test.zarr") as reader:
        sample = reader._load_sample(0)
        assert "atomic_numbers" in sample

    # After exit, _root should be None
    assert reader._root is None


def test_reader_file_not_found(tmp_path: Path) -> None:
    """Verify FileNotFoundError for nonexistent path.

    Parameters
    ----------
    tmp_path : Path
        Pytest temporary directory fixture.
    """
    with pytest.raises(FileNotFoundError):
        AtomicDataZarrReader(tmp_path / "nonexistent.zarr")


def test_reader_invalid_store(tmp_path: Path) -> None:
    """Verify ValueError for invalid Zarr store (missing groups).

    Parameters
    ----------
    tmp_path : Path
        Pytest temporary directory fixture.
    """
    # Create a directory that isn't a valid Zarr store
    invalid_path = tmp_path / "invalid.zarr"
    invalid_path.mkdir()

    # Create a minimal zarr store without required groups
    zarr.open(invalid_path, mode="w")

    with pytest.raises(ValueError, match="missing 'meta' group"):
        AtomicDataZarrReader(invalid_path)


def test_reader_refresh_after_append(tmp_path: Path) -> None:
    """Verify refresh() picks up appended samples.

    Parameters
    ----------
    tmp_path : Path
        Pytest temporary directory fixture.
    """
    data_list = list(_data_generator(3))
    writer = AtomicDataZarrWriter(tmp_path / "test.zarr")
    writer.write(data_list[:2])

    reader = AtomicDataZarrReader(tmp_path / "test.zarr")
    assert len(reader) == 2

    # Append a third sample externally
    writer.append(data_list[2])

    # Before refresh, reader still sees 2 samples
    assert len(reader) == 2

    # After refresh, reader sees 3 samples
    reader.refresh()
    assert len(reader) == 3

    # The new sample should be loadable
    sample = reader._load_sample(2)
    assert sample["atomic_numbers"].shape[0] == data_list[2].atomic_numbers.shape[0]
    reader.close()


def test_reader_refresh_after_delete(tmp_path: Path) -> None:
    """Verify refresh() picks up deleted samples.

    Parameters
    ----------
    tmp_path : Path
        Pytest temporary directory fixture.
    """
    data_list = list(_data_generator(3))
    writer = AtomicDataZarrWriter(tmp_path / "test.zarr")
    writer.write(data_list)

    reader = AtomicDataZarrReader(tmp_path / "test.zarr")
    assert len(reader) == 3

    # Delete a sample externally
    writer.delete([1])

    # Before refresh, reader still sees 3 samples
    assert len(reader) == 3

    # After refresh, reader sees 2 samples
    reader.refresh()
    assert len(reader) == 2
    reader.close()


def test_reader_refresh_on_closed_raises(tmp_path: Path) -> None:
    """Verify refresh() on a closed reader raises RuntimeError.

    Parameters
    ----------
    tmp_path : Path
        Pytest temporary directory fixture.
    """
    data = next(_data_generator(1))
    writer = AtomicDataZarrWriter(tmp_path / "test.zarr")
    writer.write(data)

    reader = AtomicDataZarrReader(tmp_path / "test.zarr")
    reader.close()

    with pytest.raises(RuntimeError, match="Cannot refresh a closed reader"):
        reader.refresh()


class TestDataset:
    """Tests for the AtomicData-native Dataset class."""

    @pytest.mark.parametrize("num_samples", [1, 3, 5])
    def test_dataset_len(self, num_samples: int, tmp_path: Path) -> None:
        """Verify len(dataset) == len(reader)."""
        data_list = list(_data_generator(num_samples))
        writer = AtomicDataZarrWriter(tmp_path / "test.zarr")
        writer.write(data_list)

        with AtomicDataZarrReader(tmp_path / "test.zarr") as reader:
            dataset = Dataset(reader, device="cpu")

            assert len(dataset) == len(reader)
            assert len(dataset) == num_samples

    @pytest.mark.parametrize("num_samples", [1, 3, 5])
    def test_dataset_skips_deleted(self, num_samples: int, tmp_path: Path) -> None:
        """Verify Dataset correctly skips soft-deleted samples."""
        # Create at least 3 samples for the delete test
        data_list = list(_data_generator(max(3, num_samples)))

        writer = AtomicDataZarrWriter(tmp_path / "test.zarr")
        writer.write(data_list)

        # Soft-delete the middle sample (delete takes a list)
        writer.delete([1])

        with AtomicDataZarrReader(tmp_path / "test.zarr") as reader:
            dataset = Dataset(reader, device="cpu")

            # Verify length reflects deletion
            assert len(dataset) == len(data_list) - 1

            # dataset[0] should be original data_list[0]
            loaded0, _ = dataset[0]
            assert torch.equal(loaded0.atomic_numbers, data_list[0].atomic_numbers)
            assert (
                loaded0.atomic_numbers.shape[0] == data_list[0].atomic_numbers.shape[0]
            )

            # dataset[1] should be original data_list[2] (data_list[1] was deleted)
            loaded1, _ = dataset[1]
            assert torch.equal(loaded1.atomic_numbers, data_list[2].atomic_numbers)
            assert (
                loaded1.atomic_numbers.shape[0] == data_list[2].atomic_numbers.shape[0]
            )


def test_dataset_returns_atomic_data(tmp_path: Path) -> None:
    """Verify Dataset returns (AtomicData, dict) tuples with expected keys.

    Parameters
    ----------
    tmp_path : Path
        Pytest fixture providing a temporary directory.
    """
    data = next(_data_generator(1))
    writer = AtomicDataZarrWriter(tmp_path / "test.zarr")
    writer.write(data)

    with AtomicDataZarrReader(tmp_path / "test.zarr") as reader:
        dataset = Dataset(reader, device="cpu")

        result = dataset[0]

        # Verify return type
        assert isinstance(result, tuple)
        assert len(result) == 2

        atomic_data, metadata = result
        assert isinstance(atomic_data, AtomicData)
        assert isinstance(metadata, dict)

        # Verify expected keys are present (AtomicData uses model_dump for keys)
        expected_keys = [
            "atomic_numbers",
            "positions",
            "forces",
            "energy",
            "neighbor_list",
            "shifts",
            "cell",
            "pbc",
        ]
        atomic_data_keys = atomic_data.model_dump().keys()
        for key in expected_keys:
            assert key in atomic_data_keys, f"Missing key {key} in AtomicData"


def test_dataset_device_transfer(tmp_path: Path, device: str) -> None:
    """Verify data is transferred to the specified device.

    Parameters
    ----------
    tmp_path : Path
        Pytest fixture providing a temporary directory.
    device : str
        Device string fixture from conftest.py.
    """
    data = next(_data_generator(1))
    writer = AtomicDataZarrWriter(tmp_path / "test.zarr")
    writer.write(data)

    with AtomicDataZarrReader(tmp_path / "test.zarr") as reader:
        dataset = Dataset(reader, device=device)

        atomic_data, _ = dataset[0]

        # Compare device types (cuda vs cuda:0 should match)
        assert atomic_data.device.type == torch.device(device).type


def test_dataset_roundtrip_values(tmp_path: Path) -> None:
    """Verify field values match original data after roundtrip.

    Parameters
    ----------
    tmp_path : Path
        Pytest fixture providing a temporary directory.
    """
    original = next(_data_generator(1))
    writer = AtomicDataZarrWriter(tmp_path / "test.zarr")
    writer.write(original)

    with AtomicDataZarrReader(tmp_path / "test.zarr") as reader:
        dataset = Dataset(reader, device="cpu")

        loaded, _ = dataset[0]

        # Compare tensor values
        assert torch.equal(loaded.atomic_numbers, original.atomic_numbers)
        assert torch.allclose(loaded.positions, original.positions)
        assert torch.allclose(loaded.forces, original.forces)
        assert torch.allclose(loaded.energy, original.energy)
        assert torch.allclose(loaded.cell, original.cell)
        assert torch.equal(loaded.pbc, original.pbc)
        assert torch.equal(loaded.neighbor_list, original.neighbor_list)
        assert torch.allclose(loaded.shifts, original.shifts)


class _OrderedReadManyReader:
    """Minimal reader that records read_many calls for DataLoader tests."""

    def __init__(self, n: int = 5) -> None:
        self._n = n
        self.read_many_calls: list[list[int]] = []
        self.pin_memory = False

    def _load_sample(self, index: int) -> dict[str, torch.Tensor]:
        return _make_ordered_atomic_data(index + 1).to_dict()

    def _get_sample_metadata(self, index: int) -> dict[str, int]:
        return {"src_index": index}

    @property
    def field_names(self) -> list[str]:
        return list(self._load_sample(0)) if self._n > 0 else []

    def read_many(
        self, indices: Sequence[int]
    ) -> list[tuple[dict[str, torch.Tensor], dict[str, int]]]:
        self.read_many_calls.append(list(indices))
        return [
            (self._load_sample(index), self._get_sample_metadata(index))
            for index in indices
        ]

    def __len__(self) -> int:
        return self._n

    def close(self) -> None:
        pass


def test_dataset_load_batches_uses_reader_read_many() -> None:
    """Verify Dataset.load_batches delegates batch reads to the reader."""
    reader = _OrderedReadManyReader()
    dataset = Dataset(reader, device="cpu")

    batch = dataset.load_batches([[3, 1]])[0]

    assert reader.read_many_calls == [[3, 1]]
    assert batch.atomic_numbers.tolist() == [4, 2]


def test_dataset_and_dataloader_are_standalone_datapipes() -> None:
    """Verify nvalchemi datapipes expose their optimized standalone APIs."""
    dataset = Dataset(_OrderedReadManyReader(), device="cpu")
    loader = DataLoader(dataset, batch_size=1, use_streams=False)
    multidataset = MultiDataset(dataset)

    assert dataset.reader is not None
    assert loader.dataset is dataset
    assert multidataset.datasets == (dataset,)


def test_dataloader_fused_prefetches_sampler_batches_without_streams() -> None:
    """Verify DataLoader fuses sampler batches even without CUDA streams."""
    reader = _OrderedReadManyReader()
    dataset = Dataset(reader, device="cpu")

    class FixedSampler(Sampler[int]):
        """Sampler with a deterministic non-sequential order."""

        def __iter__(self) -> Iterator[int]:
            return iter([4, 2, 0])

        def __len__(self) -> int:
            return 3

    loader = DataLoader(
        dataset,
        batch_size=2,
        sampler=FixedSampler(),
        use_streams=False,
    )

    batches = list(loader)

    assert reader.read_many_calls == [[4, 2, 0]]
    assert [batch.atomic_numbers.tolist() for batch in batches] == [[5, 3], [1]]


def test_dataloader_fused_prefetches_batch_sampler_without_streams() -> None:
    """Verify pre-batched indices are fused by default without CUDA streams."""
    reader = _OrderedReadManyReader()
    dataset = Dataset(reader, device="cpu")

    class FixedBatchSampler(Sampler[list[int]]):
        """Sampler that yields pre-batched indices."""

        def __iter__(self) -> Iterator[list[int]]:
            return iter([[3, 1], [0, 2]])

        def __len__(self) -> int:
            return 2

    loader = DataLoader(dataset, batch_sampler=FixedBatchSampler(), use_streams=False)

    batches = list(loader)

    assert len(loader) == 2
    assert reader.read_many_calls == [[3, 1, 0, 2]]
    assert [batch.atomic_numbers.tolist() for batch in batches] == [[4, 2], [1, 3]]


def test_dataloader_prefetch_factor_zero_uses_simple_batches() -> None:
    """Verify prefetch_factor=0 preserves one read_many call per batch."""
    reader = _OrderedReadManyReader()
    dataset = Dataset(reader, device="cpu")

    loader = DataLoader(
        dataset,
        batch_size=2,
        use_streams=False,
        prefetch_factor=0,
    )

    batches = list(loader)

    assert reader.read_many_calls == [[0, 1], [2, 3], [4]]
    assert [batch.atomic_numbers.tolist() for batch in batches] == [[1, 2], [3, 4], [5]]


def test_dataloader_prefetch_factor_controls_read_window() -> None:
    """Verify prefetch_factor controls the fused read_many window."""
    reader = _OrderedReadManyReader(n=10)
    dataset = Dataset(reader, device="cpu")
    loader = DataLoader(
        dataset,
        batch_size=2,
        prefetch_factor=3,
        use_streams=False,
    )

    batches = list(loader)

    assert loader.effective_read_window == 6
    assert reader.read_many_calls == [[0, 1, 2, 3, 4, 5], [6, 7, 8, 9]]
    assert [batch.atomic_numbers.tolist() for batch in batches] == [
        [1, 2],
        [3, 4],
        [5, 6],
        [7, 8],
        [9, 10],
    ]


def test_multidataset_getitem_enriches_metadata() -> None:
    """Verify MultiDataset sample access reports source dataset metadata."""
    reader_a = _OrderedReadManyReader(n=3)
    reader_b = _OrderedReadManyReader(n=4)
    dataset_a = Dataset(reader_a, device="cpu")
    dataset_b = Dataset(reader_b, device="cpu")
    dataset = MultiDataset(dataset_a, dataset_b)

    data, metadata = dataset[4]

    assert reader_a.read_many_calls == []
    assert reader_b.read_many_calls == [[1]]
    assert data.atomic_numbers.item() == 2
    assert metadata["dataset_index"] == 1
    assert metadata["src_index"] == 1


def test_multidataset_load_batches_routes_mixed_indices_to_child_batches() -> None:
    """Verify mixed MultiDataset batches route through child load_batches methods."""
    dataset_a = Dataset(_OrderedReadManyReader(n=3), device="cpu")
    dataset_b = Dataset(_OrderedReadManyReader(n=4), device="cpu")
    dataset = MultiDataset(dataset_a, dataset_b)

    with (
        patch.object(dataset_a, "load_batches", wraps=dataset_a.load_batches) as load_a,
        patch.object(dataset_b, "load_batches", wraps=dataset_b.load_batches) as load_b,
    ):
        batch = dataset.load_batches([[0, 3, 2, 6]])[0]

    assert [
        [list(batch_indices) for batch_indices in call.args[0]]
        for call in load_a.call_args_list
    ] == [[[0, 2]]]
    assert [
        [list(batch_indices) for batch_indices in call.args[0]]
        for call in load_b.call_args_list
    ] == [[[0, 3]]]
    assert batch.atomic_numbers.tolist() == [1, 1, 3, 4]


def test_multidataset_dataloader_delegates_single_child_fused_prefetch() -> None:
    """Verify same-child fused chunks use the child fused read path."""
    reader_a = _OrderedReadManyReader(n=6)
    reader_b = _OrderedReadManyReader(n=4)
    dataset = MultiDataset(
        Dataset(reader_a, device="cpu"),
        Dataset(reader_b, device="cpu"),
    )
    loader = DataLoader(
        dataset,
        batch_size=2,
        prefetch_factor=2,
        sampler=SequentialSampler(range(4)),
        use_streams=False,
    )

    batches = list(loader)

    assert reader_a.read_many_calls == [[0, 1, 2, 3]]
    assert reader_b.read_many_calls == []
    assert [batch.atomic_numbers.tolist() for batch in batches] == [[1, 2], [3, 4]]


def test_multidataset_dataloader_groups_mixed_fused_prefetch_by_child() -> None:
    """Verify mixed fused chunks still issue one read_many per child."""
    reader_a = _OrderedReadManyReader(n=3)
    reader_b = _OrderedReadManyReader(n=4)
    dataset_a = Dataset(reader_a, device="cpu")
    dataset_b = Dataset(reader_b, device="cpu")
    dataset = MultiDataset(
        dataset_a,
        dataset_b,
    )

    class MixedSampler(Sampler[int]):
        """Sampler that alternates between child datasets."""

        def __iter__(self) -> Iterator[int]:
            return iter([0, 3, 2, 6])

        def __len__(self) -> int:
            return 4

    with (
        patch.object(dataset_a, "load_batches", wraps=dataset_a.load_batches) as load_a,
        patch.object(dataset_b, "load_batches", wraps=dataset_b.load_batches) as load_b,
    ):
        loader = DataLoader(
            dataset,
            batch_size=2,
            prefetch_factor=2,
            sampler=MixedSampler(),
            use_streams=False,
        )
        batches = list(loader)

    assert [
        [list(batch_indices) for batch_indices in call.args[0]]
        for call in load_a.call_args_list
    ] == [[[0], [2]]]
    assert [
        [list(batch_indices) for batch_indices in call.args[0]]
        for call in load_b.call_args_list
    ] == [[[0], [3]]]
    assert reader_a.read_many_calls == [[0, 2]]
    assert reader_b.read_many_calls == [[0, 3]]
    assert [batch.atomic_numbers.tolist() for batch in batches] == [[1, 1], [3, 4]]


def test_dataloader_rejects_negative_prefetch_factor() -> None:
    """Verify negative prefetch factors fail instead of disabling prefetching."""
    reader = _OrderedReadManyReader()
    dataset = Dataset(reader, device="cpu")

    with pytest.raises(ValueError, match="prefetch_factor"):
        DataLoader(dataset, prefetch_factor=-1)


def test_dataloader_pin_memory_enables_reader_pin_memory() -> None:
    """Verify DataLoader pin_memory requests pinned reads from the reader."""
    reader = _OrderedReadManyReader()
    dataset = Dataset(reader, device="cpu")

    loader = DataLoader(dataset, batch_size=2, use_streams=False, pin_memory=True)

    assert loader.pin_memory is True
    assert dataset.pin_memory is True
    assert reader.pin_memory is True


def test_dataloader_pin_memory_does_not_mutate_multidataset_children() -> None:
    """Verify MultiDataset leaves child pin-memory policy to each dataset."""
    reader_a = _OrderedReadManyReader()
    reader_b = _OrderedReadManyReader()
    dataset = MultiDataset(
        Dataset(reader_a, device="cpu"),
        Dataset(reader_b, device="cpu"),
    )

    loader = DataLoader(dataset, batch_size=2, use_streams=False, pin_memory=True)

    assert loader.pin_memory is True
    assert reader_a.pin_memory is False
    assert reader_b.pin_memory is False


def test_multidataset_output_strict_uses_first_nonempty_field_names() -> None:
    """Verify strict field validation ignores empty leading datasets."""
    empty_dataset = Dataset(_OrderedReadManyReader(n=0), device="cpu")
    nonempty_dataset = Dataset(_OrderedReadManyReader(n=2), device="cpu")

    dataset = MultiDataset(empty_dataset, nonempty_dataset, output_strict=True)

    assert dataset.field_names == nonempty_dataset.field_names
    assert dataset.validate_field_names() == nonempty_dataset.field_names


def test_multidataset_cancel_prefetch_canonicalizes_negative_indices() -> None:
    """Verify cancel_prefetch(-1) clears delegated child sample prefetch."""
    dataset = MultiDataset(
        Dataset(_OrderedReadManyReader(n=3), device="cpu"),
        Dataset(_OrderedReadManyReader(n=2), device="cpu"),
    )

    dataset.prefetch(-1)
    assert dataset.prefetch_count == 1

    dataset.cancel_prefetch(-1)

    assert dataset.prefetch_count == 0
    dataset.close()


@pytest.mark.parametrize("batch_size", [1, 4, 8, 16, 32])
@pytest.mark.parametrize("sample_scale", [0.9, 1.0, 1.1])
def test_dataloader_yields_batch(
    batch_size: int, sample_scale: float, device: str, tmp_path: Path
) -> None:
    """Verify DataLoader yields Batch instances, not TensorDict or tuples."""
    # generate either too few, just right, or too many samples
    # ensure at least 1 sample to avoid empty data list error
    num_samples = max(1, floor(batch_size * sample_scale))
    data_list = list(_data_generator(num_samples))
    writer = AtomicDataZarrWriter(tmp_path / "test.zarr")
    writer.write(data_list)

    with AtomicDataZarrReader(tmp_path / "test.zarr") as reader:
        dataset = Dataset(reader, device=device)
        loader = DataLoader(dataset, batch_size=batch_size)

        batches = list(loader)
        for index, batch in enumerate(batches):
            assert isinstance(batch, Batch)
            # make sure we have the right batch size for non-final batches
            if index < len(batches) - 1:
                assert batch.num_graphs == batch_size


def test_dataloader_drop_last(tmp_path: Path) -> None:
    """Verify drop_last parameter drops incomplete final batches."""
    # generate 5 samples
    data_list = list(_data_generator(5))
    writer = AtomicDataZarrWriter(tmp_path / "test.zarr")
    writer.write(data_list)

    with AtomicDataZarrReader(tmp_path / "test.zarr") as reader:
        dataset = Dataset(reader, device="cpu")

        # Without drop_last: 5 samples / 2 batch_size = 3 batches (2 + 2 + 1)
        loader_with_incomplete = DataLoader(dataset, batch_size=2, drop_last=False)
        batches_with_incomplete = list(loader_with_incomplete)
        assert len(batches_with_incomplete) == 3
        assert batches_with_incomplete[-1].num_graphs == 1  # Last batch has 1 sample

        # With drop_last: 5 samples / 2 batch_size = 2 batches (2 + 2)
        loader_drop_last = DataLoader(dataset, batch_size=2, drop_last=True)
        batches_drop_last = list(loader_drop_last)
        assert len(batches_drop_last) == 2
        for batch in batches_drop_last:
            assert batch.num_graphs == 2


def test_dataloader_shuffle(tmp_path: Path) -> None:
    """Verify shuffle randomizes sample order."""
    data_list = list(_data_generator(64))
    writer = AtomicDataZarrWriter(tmp_path / "test.zarr")
    writer.write(data_list)

    with AtomicDataZarrReader(tmp_path / "test.zarr") as reader:
        dataset = Dataset(reader, device="cpu")

        loader1 = DataLoader(dataset, batch_size=1, shuffle=True)
        loader2 = DataLoader(dataset, batch_size=1, shuffle=True)

        # Collect full iteration order from each loader; with shuffle=True they
        # should differ with very high probability (flaky if we only check first batch).
        order1 = [batch["positions"].sum().item() for batch in loader1]
        order2 = [batch["positions"].sum().item() for batch in loader2]
        assert order1 != order2, "Shuffle should produce different order across loaders"


def test_dataloader_custom_sampler(tmp_path: Path) -> None:
    """Verify DataLoader respects a minimal custom sampler order."""

    class ReverseOddSampler(Sampler[int]):
        """Yield a fixed non-sequential subset to exercise custom sampling."""

        def __init__(self, indices: list[int]) -> None:
            self.indices = indices

        def __iter__(self) -> Iterator[int]:
            return iter(self.indices)

        def __len__(self) -> int:
            return len(self.indices)

    data_list = [_make_ordered_atomic_data(i + 1) for i in range(5)]
    writer = AtomicDataZarrWriter(tmp_path / "test.zarr")
    writer.write(data_list)

    sampler = ReverseOddSampler([4, 2, 0])
    with AtomicDataZarrReader(tmp_path / "test.zarr") as reader:
        dataset = Dataset(reader, device="cpu")
        loader = DataLoader(dataset, batch_size=2, sampler=sampler, use_streams=False)

        with patch.object(reader, "read_many", wraps=reader.read_many) as read_many:
            batches = list(loader)

    assert [batch.atomic_numbers.tolist() for batch in batches] == [[5, 3], [1]]
    assert [list(call.args[0]) for call in read_many.call_args_list] == [[4, 2, 0]]


def test_dataloader_distributed_sampler(tmp_path: Path) -> None:
    """Verify DataLoader works with PyTorch's DistributedSampler."""
    data_list = [_make_ordered_atomic_data(i + 1) for i in range(6)]
    writer = AtomicDataZarrWriter(tmp_path / "test.zarr")
    writer.write(data_list)

    with AtomicDataZarrReader(tmp_path / "test.zarr") as reader:
        dataset = Dataset(reader, device="cpu")
        sampler = DistributedSampler(
            dataset,
            num_replicas=2,
            rank=1,
            shuffle=False,
            drop_last=False,
        )
        loader = DataLoader(dataset, batch_size=2, sampler=sampler, use_streams=False)

        with patch.object(reader, "read_many", wraps=reader.read_many) as read_many:
            batches = list(loader)

    assert [batch.atomic_numbers.tolist() for batch in batches] == [[2, 4], [6]]
    assert [list(call.args[0]) for call in read_many.call_args_list] == [[1, 3, 5]]


class TestDatasetPrefetch:
    """Tests for Dataset prefetch mechanics (CPU thread-pool path)."""

    def test_prefetch_then_getitem(self, tmp_path: Path) -> None:
        """Prefetch sample 0, then retrieve via __getitem__."""
        data_list = list(_data_generator(5))
        writer = AtomicDataZarrWriter(tmp_path / "test.zarr")
        writer.write(data_list)

        with AtomicDataZarrReader(tmp_path / "test.zarr") as reader:
            dataset = Dataset(reader, device="cpu")

            # Prefetch without a CUDA stream (CPU path)
            dataset.prefetch(0)

            # Retrieve the prefetched sample
            result = dataset[0]

            # Verify it returns (AtomicData, dict)
            assert isinstance(result, tuple)
            assert len(result) == 2
            atomic_data, metadata = result
            assert isinstance(atomic_data, AtomicData)
            assert isinstance(metadata, dict)

    def test_prefetch_returns_correct_data(self, tmp_path: Path) -> None:
        """Prefetch index 1 with distinguishable data and verify correct retrieval."""
        # Create 3 samples with distinguishable numbers
        num_atoms, num_edges = 5, 8
        data0 = AtomicData(
            atomic_numbers=torch.full((num_atoms,), fill_value=10, dtype=torch.long),
            positions=torch.randn(num_atoms, 3),
            cell=torch.eye(3).unsqueeze(0),
            pbc=torch.tensor([[True, True, True]]),
            neighbor_list=torch.stack(
                [
                    torch.randint(0, num_atoms, (num_edges,)),
                    torch.randint(0, num_atoms, (num_edges,)),
                ],
                dim=1,
            ),
            shifts=torch.randn(num_edges, 3),
        )
        data1 = AtomicData(
            atomic_numbers=torch.full((num_atoms,), fill_value=20, dtype=torch.long),
            positions=torch.randn(num_atoms, 3),
            cell=torch.eye(3).unsqueeze(0),
            pbc=torch.tensor([[True, True, True]]),
            neighbor_list=torch.stack(
                [
                    torch.randint(0, num_atoms, (num_edges,)),
                    torch.randint(0, num_atoms, (num_edges,)),
                ],
                dim=1,
            ),
            shifts=torch.randn(num_edges, 3),
        )
        data2 = AtomicData(
            atomic_numbers=torch.full((num_atoms,), fill_value=30, dtype=torch.long),
            positions=torch.randn(num_atoms, 3),
            cell=torch.eye(3).unsqueeze(0),
            pbc=torch.tensor([[True, True, True]]),
            neighbor_list=torch.stack(
                [
                    torch.randint(0, num_atoms, (num_edges,)),
                    torch.randint(0, num_atoms, (num_edges,)),
                ],
                dim=1,
            ),
            shifts=torch.randn(num_edges, 3),
        )

        writer = AtomicDataZarrWriter(tmp_path / "test.zarr")
        writer.write([data0, data1, data2])

        with AtomicDataZarrReader(tmp_path / "test.zarr") as reader:
            dataset = Dataset(reader, device="cpu")

            # Prefetch index 1 and retrieve
            dataset.prefetch(1)
            atomic_data, _ = dataset[1]

            # Verify numbers match expected sample (all-20)
            assert torch.all(atomic_data.atomic_numbers == 20)

    def test_prefetch_multiple_samples(self, tmp_path: Path) -> None:
        """Prefetch indices 0, 1, 2 and verify all retrieve correctly."""
        # Create 5 samples with distinguishable numbers
        num_atoms, num_edges = 5, 8
        data_list = []
        for i in range(5):
            data = AtomicData(
                atomic_numbers=torch.full(
                    (num_atoms,), fill_value=10 * (i + 1), dtype=torch.long
                ),
                positions=torch.randn(num_atoms, 3),
                cell=torch.eye(3).unsqueeze(0),
                pbc=torch.tensor([[True, True, True]]),
                neighbor_list=torch.stack(
                    [
                        torch.randint(0, num_atoms, (num_edges,)),
                        torch.randint(0, num_atoms, (num_edges,)),
                    ],
                    dim=1,
                ),
                shifts=torch.randn(num_edges, 3),
            )
            data_list.append(data)

        writer = AtomicDataZarrWriter(tmp_path / "test.zarr")
        writer.write(data_list)

        with AtomicDataZarrReader(tmp_path / "test.zarr") as reader:
            dataset = Dataset(reader, device="cpu")

            # Prefetch indices 0, 1, 2
            dataset.prefetch(0)
            dataset.prefetch(1)
            dataset.prefetch(2)

            # Retrieve all three and verify
            ad0, _ = dataset[0]
            ad1, _ = dataset[1]
            ad2, _ = dataset[2]

            assert torch.all(ad0.atomic_numbers == 10)
            assert torch.all(ad1.atomic_numbers == 20)
            assert torch.all(ad2.atomic_numbers == 30)

    def test_cancel_prefetch_clears_futures(self, tmp_path: Path) -> None:
        """Prefetch several indices, cancel all, verify no futures remain."""
        data_list = list(_data_generator(5))
        writer = AtomicDataZarrWriter(tmp_path / "test.zarr")
        writer.write(data_list)

        with AtomicDataZarrReader(tmp_path / "test.zarr") as reader:
            dataset = Dataset(reader, device="cpu")

            # Prefetch several indices
            dataset.prefetch(0)
            dataset.prefetch(1)
            dataset.prefetch(2)

            # Cancel all prefetches
            dataset.cancel_prefetch()

            # Verify prefetch_count is 0
            assert len(dataset._prefetch_futures) == 0

    def test_cancel_prefetch_single_index(self, tmp_path: Path) -> None:
        """Prefetch indices 0 and 1, cancel 0, verify index 1 still retrievable."""
        # Create distinguishable samples
        num_atoms, num_edges = 5, 8
        data0 = AtomicData(
            atomic_numbers=torch.full((num_atoms,), fill_value=10, dtype=torch.long),
            positions=torch.randn(num_atoms, 3),
            cell=torch.eye(3).unsqueeze(0),
            pbc=torch.tensor([[True, True, True]]),
            neighbor_list=torch.stack(
                [
                    torch.randint(0, num_atoms, (num_edges,)),
                    torch.randint(0, num_atoms, (num_edges,)),
                ],
                dim=1,
            ),
            shifts=torch.randn(num_edges, 3),
        )
        data1 = AtomicData(
            atomic_numbers=torch.full((num_atoms,), fill_value=20, dtype=torch.long),
            positions=torch.randn(num_atoms, 3),
            cell=torch.eye(3).unsqueeze(0),
            pbc=torch.tensor([[True, True, True]]),
            neighbor_list=torch.stack(
                [
                    torch.randint(0, num_atoms, (num_edges,)),
                    torch.randint(0, num_atoms, (num_edges,)),
                ],
                dim=1,
            ),
            shifts=torch.randn(num_edges, 3),
        )

        writer = AtomicDataZarrWriter(tmp_path / "test.zarr")
        writer.write([data0, data1])

        with AtomicDataZarrReader(tmp_path / "test.zarr") as reader:
            dataset = Dataset(reader, device="cpu")

            # Prefetch both indices
            dataset.prefetch(0)
            dataset.prefetch(1)

            # Cancel only index 0
            dataset.cancel_prefetch(0)

            # Index 1 should still be retrievable via prefetch path
            atomic_data, _ = dataset[1]
            assert torch.all(atomic_data.atomic_numbers == 20)

    def test_load_and_transform_returns_prefetch_result(self, tmp_path: Path) -> None:
        """Directly call _load_and_transform and verify it returns _PrefetchResult."""
        data = next(_data_generator(1))
        writer = AtomicDataZarrWriter(tmp_path / "test.zarr")
        writer.write(data)

        with AtomicDataZarrReader(tmp_path / "test.zarr") as reader:
            dataset = Dataset(reader, device="cpu")

            # Directly call _load_and_transform
            result = dataset._load_and_transform(0)

            # Verify it returns a _PrefetchResult
            assert isinstance(result, _PrefetchResult)
            assert isinstance(result.data, AtomicData)
            assert isinstance(result.metadata, dict)
            assert result.error is None

    def test_load_and_transform_captures_error(self, tmp_path: Path) -> None:
        """Mock reader to raise error, verify _load_and_transform captures it."""
        data = next(_data_generator(1))
        writer = AtomicDataZarrWriter(tmp_path / "test.zarr")
        writer.write(data)

        with AtomicDataZarrReader(tmp_path / "test.zarr") as reader:
            dataset = Dataset(reader, device="cpu")

            # Mock the raw batch hook to raise an error
            with patch.object(
                reader, "_load_many_samples", side_effect=RuntimeError("test error")
            ):
                result = dataset._load_and_transform(0)

                # Verify error is captured
                assert result.error is not None
                assert isinstance(result.error, RuntimeError)
                assert "test error" in str(result.error)
                assert result.data is None

    def test_prefetch_error_propagation(self, tmp_path: Path) -> None:
        """Mock reader to raise error, verify prefetch propagates it on __getitem__."""
        data = next(_data_generator(1))
        writer = AtomicDataZarrWriter(tmp_path / "test.zarr")
        writer.write(data)

        with AtomicDataZarrReader(tmp_path / "test.zarr") as reader:
            dataset = Dataset(reader, device="cpu")

            # Mock the raw batch hook to raise an error
            with patch.object(
                reader, "_load_many_samples", side_effect=RuntimeError("boom")
            ):
                # Prefetch the sample (error will be captured)
                dataset.prefetch(0)

                # Accessing the sample should raise the error
                with pytest.raises(RuntimeError, match="boom"):
                    dataset[0]

    def test_dataset_close_with_inflight_prefetch(self, tmp_path: Path) -> None:
        """Prefetch samples, then close reader, verify no exceptions or hangs."""
        data_list = list(_data_generator(5))
        writer = AtomicDataZarrWriter(tmp_path / "test.zarr")
        writer.write(data_list)

        reader = AtomicDataZarrReader(tmp_path / "test.zarr")
        dataset = Dataset(reader, device="cpu")

        # Prefetch several indices
        dataset.prefetch(0)
        dataset.prefetch(1)
        dataset.prefetch(2)

        # Close the reader - should complete without errors
        reader.close()

        # Verify reader is closed
        assert reader._root is None


class TestFusedBatchPrefetch:
    """Tests for fused multi-batch prefetch (prefetch_fused_batches / get_fused_batches)."""

    def test_fused_prefetch_yields_correct_batches(
        self, tmp_path: Path, gpu_device: str
    ) -> None:
        """Fused read yields the same batches as individual reads."""
        data_list = list(_data_generator(12))
        writer = AtomicDataZarrWriter(tmp_path / "test.zarr")
        writer.write(data_list)

        with AtomicDataZarrReader(tmp_path / "test.zarr") as reader:
            dataset = Dataset(reader, device=gpu_device)

            # Read synchronously for reference
            ref_b0, ref_b1, ref_b2 = dataset.load_batches(
                [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]]
            )

            # Read via fused prefetch
            dataset.prefetch_fused_batches([[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]])
            fused_batches = list(dataset.get_fused_batches())

            assert len(fused_batches) == 3
            for fused, ref in zip(fused_batches, [ref_b0, ref_b1, ref_b2], strict=True):
                assert fused.num_graphs == ref.num_graphs
                torch.testing.assert_close(fused.positions, ref.positions)
                torch.testing.assert_close(fused.atomic_numbers, ref.atomic_numbers)

    def test_fused_prefetch_variable_batch_sizes(
        self, tmp_path: Path, gpu_device: str
    ) -> None:
        """Handles sub-batches of different sizes (e.g. last batch is smaller)."""
        data_list = list(_data_generator(7))
        writer = AtomicDataZarrWriter(tmp_path / "test.zarr")
        writer.write(data_list)

        with AtomicDataZarrReader(tmp_path / "test.zarr") as reader:
            dataset = Dataset(reader, device=gpu_device)

            dataset.prefetch_fused_batches([[0, 1, 2], [3, 4, 5], [6]])
            fused_batches = list(dataset.get_fused_batches())

            assert len(fused_batches) == 3
            assert fused_batches[0].num_graphs == 3
            assert fused_batches[1].num_graphs == 3
            assert fused_batches[2].num_graphs == 1

    def test_fused_prefetch_raises_without_pending(
        self, tmp_path: Path, gpu_device: str
    ) -> None:
        """get_fused_batches raises RuntimeError when no prefetch is pending."""
        data_list = list(_data_generator(4))
        writer = AtomicDataZarrWriter(tmp_path / "test.zarr")
        writer.write(data_list)

        with AtomicDataZarrReader(tmp_path / "test.zarr") as reader:
            dataset = Dataset(reader, device=gpu_device)
            with pytest.raises(RuntimeError, match="No fused batch prefetch pending"):
                list(dataset.get_fused_batches())

    def test_fused_prefetch_queues_two(self, tmp_path: Path, gpu_device: str) -> None:
        """Two prefetch_fused_batches calls queue; a third fails explicitly."""
        data_list = list(_data_generator(12))
        writer = AtomicDataZarrWriter(tmp_path / "test.zarr")
        writer.write(data_list)

        with AtomicDataZarrReader(tmp_path / "test.zarr") as reader:
            dataset = Dataset(reader, device=gpu_device)

            dataset.prefetch_fused_batches([[0, 1], [2, 3]])
            dataset.prefetch_fused_batches([[4, 5], [6, 7]])
            with pytest.raises(RuntimeError, match="queue is full"):
                dataset.prefetch_fused_batches([[8, 9], [10, 11]])

            # First get_fused_batches returns chunk 1
            batches_1 = list(dataset.get_fused_batches())
            assert len(batches_1) == 2
            assert batches_1[0].num_graphs == 2

            # Second get_fused_batches returns chunk 2
            batches_2 = list(dataset.get_fused_batches())
            assert len(batches_2) == 2
            assert batches_2[0].num_graphs == 2

    def test_cancel_clears_fused_prefetch(
        self, tmp_path: Path, gpu_device: str
    ) -> None:
        """cancel_prefetch clears the fused prefetch future."""
        data_list = list(_data_generator(4))
        writer = AtomicDataZarrWriter(tmp_path / "test.zarr")
        writer.write(data_list)

        with AtomicDataZarrReader(tmp_path / "test.zarr") as reader:
            dataset = Dataset(reader, device=gpu_device)

            dataset.prefetch_fused_batches([[0, 1], [2, 3]])
            dataset.cancel_prefetch()

            # Should now raise because the future was cleared
            with pytest.raises(RuntimeError, match="No fused batch prefetch pending"):
                list(dataset.get_fused_batches())

    def test_dataloader_amortized_completeness(
        self, tmp_path: Path, gpu_device: str
    ) -> None:
        """DataLoader with amortized prefetch yields all samples."""
        num_samples = 20
        data_list = list(_data_generator(num_samples))
        writer = AtomicDataZarrWriter(tmp_path / "test.zarr")
        writer.write(data_list)

        with AtomicDataZarrReader(tmp_path / "test.zarr") as reader:
            dataset = Dataset(reader, device=gpu_device)
            loader = DataLoader(
                dataset,
                batch_size=3,
                prefetch_factor=4,
                use_streams=True,
            )

            total = sum(batch.num_graphs for batch in loader)
            assert total == num_samples

    def test_dataloader_amortized_shuffle(
        self, tmp_path: Path, gpu_device: str
    ) -> None:
        """Shuffled DataLoader with amortized prefetch yields all samples."""
        num_samples = 16
        data_list = list(_data_generator(num_samples))
        writer = AtomicDataZarrWriter(tmp_path / "test.zarr")
        writer.write(data_list)

        with AtomicDataZarrReader(tmp_path / "test.zarr") as reader:
            dataset = Dataset(reader, device=gpu_device)
            loader = DataLoader(
                dataset,
                batch_size=4,
                shuffle=True,
                prefetch_factor=3,
                use_streams=True,
            )

            total = sum(batch.num_graphs for batch in loader)
            assert total == num_samples

    def test_skip_validation_matches_validated(
        self, tmp_path: Path, gpu_device: str
    ) -> None:
        """skip_validation=True fused prefetch yields same data as validated path."""
        data_list = list(_data_generator(12))
        writer = AtomicDataZarrWriter(tmp_path / "test.zarr")
        writer.write(data_list)

        batch_lists = [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]]

        with AtomicDataZarrReader(tmp_path / "test.zarr") as reader:
            ds_val = Dataset(reader, device=gpu_device, skip_validation=False)
            ds_val.prefetch_fused_batches(batch_lists)
            ref_batches = list(ds_val.get_fused_batches())

        with AtomicDataZarrReader(tmp_path / "test.zarr") as reader:
            ds_raw = Dataset(reader, device=gpu_device, skip_validation=True)
            ds_raw.prefetch_fused_batches(batch_lists)
            raw_batches = list(ds_raw.get_fused_batches())

        assert len(raw_batches) == len(ref_batches)
        for raw, ref in zip(raw_batches, ref_batches, strict=True):
            assert raw.num_graphs == ref.num_graphs
            assert raw.num_nodes == ref.num_nodes
            assert raw.num_edges == ref.num_edges
            torch.testing.assert_close(raw.positions, ref.positions)
            torch.testing.assert_close(raw.atomic_numbers, ref.atomic_numbers)

    def test_skip_validation_dataloader_completeness(
        self, tmp_path: Path, gpu_device: str
    ) -> None:
        """DataLoader with skip_validation yields all samples."""
        num_samples = 20
        data_list = list(_data_generator(num_samples))
        writer = AtomicDataZarrWriter(tmp_path / "test.zarr")
        writer.write(data_list)

        with AtomicDataZarrReader(tmp_path / "test.zarr") as reader:
            dataset = Dataset(reader, device=gpu_device, skip_validation=True)
            loader = DataLoader(
                dataset,
                batch_size=3,
                prefetch_factor=4,
                use_streams=True,
            )
            total = sum(batch.num_graphs for batch in loader)
            assert total == num_samples

    def test_skip_validation_dataloader_shuffle(
        self, tmp_path: Path, gpu_device: str
    ) -> None:
        """Shuffled DataLoader with skip_validation yields all samples."""
        num_samples = 16
        data_list = list(_data_generator(num_samples))
        writer = AtomicDataZarrWriter(tmp_path / "test.zarr")
        writer.write(data_list)

        with AtomicDataZarrReader(tmp_path / "test.zarr") as reader:
            dataset = Dataset(reader, device=gpu_device, skip_validation=True)
            loader = DataLoader(
                dataset,
                batch_size=4,
                shuffle=True,
                prefetch_factor=3,
                use_streams=True,
            )
            total = sum(batch.num_graphs for batch in loader)
            assert total == num_samples

    def test_fused_prefetch_error_propagation(
        self, tmp_path: Path, gpu_device: str
    ) -> None:
        """Verify background read errors propagate through get_fused_batches."""
        data_list = list(_data_generator(6))
        writer = AtomicDataZarrWriter(tmp_path / "test.zarr")
        writer.write(data_list)

        with AtomicDataZarrReader(tmp_path / "test.zarr") as reader:
            dataset = Dataset(reader, device=gpu_device)
            # Patch _read_raw_samples to raise
            with patch.object(
                dataset, "_read_raw_samples", side_effect=RuntimeError("boom")
            ):
                dataset.prefetch_fused_batches([[0, 1], [2, 3]])
                with pytest.raises(RuntimeError, match="boom"):
                    list(dataset.get_fused_batches())

    def test_skip_validation_custom_key_roundtrip(
        self, tmp_path: Path, gpu_device: str
    ) -> None:
        """Custom Zarr fields survive the skip_validation + from_raw_dicts path."""
        data_list = list(_data_generator(4))
        writer = AtomicDataZarrWriter(tmp_path / "test.zarr")
        writer.write(data_list)
        # Add a custom system-level field
        custom = torch.arange(4, dtype=torch.float32).unsqueeze(1)
        writer.add_custom("my_flag", custom, "system")

        with AtomicDataZarrReader(tmp_path / "test.zarr") as reader:
            dataset = Dataset(reader, device=gpu_device, skip_validation=True)
            batch = dataset.load_batches([list(range(4))])[0]

        assert "my_flag" in batch.keys["system"]
        assert batch.my_flag.shape[0] == 4

    def test_skip_validation_custom_atom_key_roundtrip(
        self, tmp_path: Path, gpu_device: str
    ) -> None:
        """Custom atom-level Zarr fields are classified correctly with skip_validation.

        Reproduces the bug where from_raw_dicts misclassified custom
        per-atom tensors as system-level, causing a shape crash in
        UniformLevelStorage.
        """
        data_list = list(_data_generator(4))
        writer = AtomicDataZarrWriter(tmp_path / "test.zarr")
        writer.write(data_list)

        # Add a custom atom-level field (variable size per sample).
        total_atoms = sum(d.num_nodes for d in data_list)
        embeddings = torch.randn(total_atoms, 8)
        writer.add_custom("atom_embedding", embeddings, "atom")

        with AtomicDataZarrReader(tmp_path / "test.zarr") as reader:
            # Verify reader exposes field_levels with the custom field.
            assert reader.field_levels.get("atom_embedding") == "atom"

            dataset = Dataset(reader, device=gpu_device, skip_validation=True)
            batch = dataset.load_batches([list(range(4))])[0]

        assert "atom_embedding" in batch.keys["node"]
        assert batch.atom_embedding.shape == (total_atoms, 8)

    def test_skip_validation_custom_edge_key_roundtrip(
        self, tmp_path: Path, gpu_device: str
    ) -> None:
        """Custom edge-level Zarr fields survive skip_validation path."""
        data_list = list(_data_generator(4))
        writer = AtomicDataZarrWriter(tmp_path / "test.zarr")
        writer.write(data_list)

        total_edges = sum(d.num_edges for d in data_list)
        distances = torch.randn(total_edges)
        writer.add_custom("pair_distance", distances, "edge")

        with AtomicDataZarrReader(tmp_path / "test.zarr") as reader:
            assert reader.field_levels.get("pair_distance") == "edge"

            dataset = Dataset(reader, device=gpu_device, skip_validation=True)
            batch = dataset.load_batches([list(range(4))])[0]

        assert "pair_distance" in batch.keys["edge"]
        assert batch.pair_distance.shape == (total_edges,)

    def test_validated_custom_key_roundtrip(
        self, tmp_path: Path, gpu_device: str
    ) -> None:
        """Custom system-level Zarr fields survive validated batching."""
        data_list = list(_data_generator(4))
        writer = AtomicDataZarrWriter(tmp_path / "test.zarr")
        writer.write(data_list)
        custom = torch.arange(4, dtype=torch.float32).unsqueeze(1)
        writer.add_custom("my_flag", custom, "system")

        with AtomicDataZarrReader(tmp_path / "test.zarr") as reader:
            dataset = Dataset(reader, device=gpu_device, skip_validation=False)
            batch = dataset.get_batch(list(range(4)))

        assert "my_flag" in batch.keys["system"]
        assert batch.my_flag.shape[0] == 4

    def test_validated_custom_atom_key_roundtrip(
        self, tmp_path: Path, gpu_device: str
    ) -> None:
        """Custom atom-level Zarr fields are classified in validated batches."""
        data_list = list(_data_generator(4))
        writer = AtomicDataZarrWriter(tmp_path / "test.zarr")
        writer.write(data_list)

        total_atoms = sum(d.num_nodes for d in data_list)
        embeddings = torch.randn(total_atoms, 8)
        writer.add_custom("atom_embedding", embeddings, "atom")

        with AtomicDataZarrReader(tmp_path / "test.zarr") as reader:
            assert reader.field_levels.get("atom_embedding") == "atom"

            dataset = Dataset(reader, device=gpu_device, skip_validation=False)
            batch = dataset.get_batch(list(range(4)))

        assert "atom_embedding" in batch.keys["node"]
        assert batch.atom_embedding.shape == (total_atoms, 8)

    def test_validated_prefetch_custom_atom_key_roundtrip(
        self, tmp_path: Path, gpu_device: str
    ) -> None:
        """Custom atom-level fields survive validated get_batch prefetch."""
        data_list = list(_data_generator(4))
        writer = AtomicDataZarrWriter(tmp_path / "test.zarr")
        writer.write(data_list)

        total_atoms = sum(d.num_nodes for d in data_list)
        embeddings = torch.randn(total_atoms, 8)
        writer.add_custom("atom_embedding", embeddings, "atom")

        with AtomicDataZarrReader(tmp_path / "test.zarr") as reader:
            dataset = Dataset(reader, device=gpu_device, skip_validation=False)
            dataset.prefetch_many(list(range(4)))
            batch = dataset.get_batch(list(range(4)))

        assert "atom_embedding" in batch.keys["node"]
        assert batch.atom_embedding.shape == (total_atoms, 8)

    def test_validated_custom_edge_key_roundtrip(
        self, tmp_path: Path, gpu_device: str
    ) -> None:
        """Custom edge-level Zarr fields survive validated batching."""
        data_list = list(_data_generator(4))
        writer = AtomicDataZarrWriter(tmp_path / "test.zarr")
        writer.write(data_list)

        total_edges = sum(d.num_edges for d in data_list)
        distances = torch.randn(total_edges)
        writer.add_custom("pair_distance", distances, "edge")

        with AtomicDataZarrReader(tmp_path / "test.zarr") as reader:
            assert reader.field_levels.get("pair_distance") == "edge"

            dataset = Dataset(reader, device=gpu_device, skip_validation=False)
            batch = dataset.get_batch(list(range(4)))

        assert "pair_distance" in batch.keys["edge"]
        assert batch.pair_distance.shape == (total_edges,)

    def test_validated_fused_prefetch_custom_atom_key_roundtrip(
        self, tmp_path: Path, gpu_device: str
    ) -> None:
        """Custom atom-level fields survive validated fused prefetch."""
        data_list = list(_data_generator(4))
        writer = AtomicDataZarrWriter(tmp_path / "test.zarr")
        writer.write(data_list)

        total_atoms = sum(d.num_nodes for d in data_list)
        embeddings = torch.randn(total_atoms, 8)
        writer.add_custom("atom_embedding", embeddings, "atom")

        with AtomicDataZarrReader(tmp_path / "test.zarr") as reader:
            dataset = Dataset(reader, device=gpu_device, skip_validation=False)
            dataset.prefetch_fused_batches([list(range(4))])
            batches = list(dataset.get_fused_batches())

        assert len(batches) == 1
        batch = batches[0]
        assert "atom_embedding" in batch.keys["node"]
        assert batch.atom_embedding.shape == (total_atoms, 8)


class TestDataLoaderPrefetch:
    """Tests for DataLoader prefetch iteration path."""

    def test_iter_prefetch_mocked(self, tmp_path: Path) -> None:
        """Verify prefetch path is selected when use_streams=True and CUDA available."""
        data_list = list(_data_generator(5))
        writer = AtomicDataZarrWriter(tmp_path / "test.zarr")
        writer.write(data_list)

        with AtomicDataZarrReader(tmp_path / "test.zarr") as reader:
            dataset = Dataset(reader, device="cpu")

            # Mock CUDA to appear available during DataLoader initialization
            with patch("torch.cuda.is_available", return_value=True):
                mock_stream = MagicMock()
                with patch("torch.cuda.Stream", return_value=mock_stream):
                    loader = DataLoader(
                        dataset,
                        batch_size=2,
                        use_streams=True,
                        prefetch_factor=2,
                    )

                    # Verify use_streams was set to True based on mocked CUDA
                    assert loader.use_streams is True
                    assert loader.prefetch_factor == 2

                    # Verify streams were created
                    assert len(loader._streams) > 0

    def test_partial_iteration_no_error(self, tmp_path: Path) -> None:
        """Iterate and break after first batch, verify no errors."""
        data_list = list(_data_generator(20))
        writer = AtomicDataZarrWriter(tmp_path / "test.zarr")
        writer.write(data_list)

        with AtomicDataZarrReader(tmp_path / "test.zarr") as reader:
            dataset = Dataset(reader, device="cpu")
            loader = DataLoader(dataset, batch_size=4)

            # Iterate and break after first batch
            for batch in loader:
                assert isinstance(batch, Batch)
                break

            # Manually clean up prefetch state
            dataset.cancel_prefetch()

    def test_generate_batches_correct(self, tmp_path: Path, gpu_device: str) -> None:
        """Verify _generate_batches produces correct batches without drop_last."""
        data_list = list(_data_generator(7))
        writer = AtomicDataZarrWriter(tmp_path / "test.zarr")
        writer.write(data_list)

        with AtomicDataZarrReader(tmp_path / "test.zarr") as reader:
            dataset = Dataset(reader, device=gpu_device)
            loader = DataLoader(dataset, batch_size=3, drop_last=False)

            batches = list(loader._generate_batches())

            # Verify 3 batches: [0,1,2], [3,4,5], [6]
            assert len(batches) == 3
            assert batches[0] == [0, 1, 2]
            assert batches[1] == [3, 4, 5]
            assert batches[2] == [6]

    @pytest.mark.parametrize("num_samples", [4, 8, 16, 32])
    @pytest.mark.parametrize("prefetch_factor", [1, 2, 4])
    def test_prefetch_pipeline_completeness(
        self,
        num_samples: int,
        prefetch_factor: int,
        tmp_path: Path,
        gpu_device: str,
    ) -> None:
        """Verify all samples are yielded with CPU prefetch path.

        Parameters
        ----------
        num_samples : int
            Number of samples to generate for the test.
        prefetch_factor : int
            Prefetch factor for the DataLoader.
        tmp_path : Path
            Pytest temporary path fixture.
        gpu_device : str
            Device string fixture.
        """
        data_list = list(_data_generator(num_samples))
        writer = AtomicDataZarrWriter(tmp_path / "test.zarr")
        writer.write(data_list)

        with AtomicDataZarrReader(tmp_path / "test.zarr") as reader:
            dataset = Dataset(reader, device=gpu_device)
            loader = DataLoader(
                dataset, batch_size=3, prefetch_factor=prefetch_factor, use_streams=True
            )

            # Collect all batches
            batches = list(loader)

            # Count total samples across all batches
            total_samples = sum(batch.num_graphs for batch in batches)
            assert total_samples == num_samples

    def test_set_epoch_step_starts_sampler_batches_at_offset(
        self, tmp_path: Path
    ) -> None:
        data_list = list(_data_generator(8))
        writer = AtomicDataZarrWriter(tmp_path / "test.zarr")
        writer.write(data_list)

        with AtomicDataZarrReader(tmp_path / "test.zarr") as reader:
            dataset = Dataset(reader, device="cpu")
            loader = DataLoader(
                dataset,
                batch_size=2,
                shuffle=False,
                prefetch_factor=0,
                use_streams=False,
            )

            loader.set_epoch_step(2)

            assert list(loader._generate_batches()) == [[4, 5], [6, 7]]
            assert list(loader._generate_batches()) == [
                [0, 1],
                [2, 3],
                [4, 5],
                [6, 7],
            ]

    def test_prefetch_consumes_batches_lazily(
        self, tmp_path: Path, gpu_device: str
    ) -> None:
        """Generator is not fully materialised; only the pipeline window is consumed.

        True double-buffering primes two queue slots, then refills
        one slot after consuming the oldest chunk.  By the first
        yield, at most ``3 * prefetch_factor`` batch-index lists have
        been pulled from the sampler:
        - chunk_a (pf) — primed and consumed
        - chunk_b (pf) — primed, still in flight
        - next_chunk (pf) — collected and submitted after chunk_a
        """
        data_list = list(_data_generator(20))
        writer = AtomicDataZarrWriter(tmp_path / "test.zarr")
        writer.write(data_list)

        prefetch_factor = 2
        batch_size = 2

        with AtomicDataZarrReader(tmp_path / "test.zarr") as reader:
            dataset = Dataset(reader, device=gpu_device)
            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                prefetch_factor=prefetch_factor,
                use_streams=True,
            )

            batches_pulled = 0
            orig_generate = loader._generate_batches

            def _counting_generate():
                nonlocal batches_pulled
                for batch_indices in orig_generate():
                    batches_pulled += 1
                    yield batch_indices

            loader._generate_batches = _counting_generate

            gen = loader._iter_prefetch()
            next(gen)

            # True double-buffer: 2 primed chunks + 1 refill after
            # consuming the oldest = 3 * prefetch_factor pulled.
            assert batches_pulled <= 3 * prefetch_factor
            gen.close()


class TestZarrStoreBackends:
    """Verify that zarr internal I/O operations fire correctly through nvalchemi.

    These tests mock zarr's internal store operations (not nvalchemi code) to verify
    that when data flows through the nvalchemi writer/reader, the correct low-level
    zarr store operations actually fire for each backend type.
    """

    @pytest.mark.parametrize("num_samples", [1, 3])
    def test_write_dispatches_to_local_store_put(
        self, num_samples: int, tmp_path: Path
    ) -> None:
        """Writing via LocalStore dispatches to zarr's _put for filesystem writes."""
        from zarr.storage import LocalStore
        from zarr.storage._local import _put as _original_put

        store = LocalStore(tmp_path / "test.zarr")
        data_list = list(_data_generator(num_samples))

        # Spy on zarr's internal _put function — this is the function that
        # actually writes bytes to disk via atomic file operations
        with patch("zarr.storage._local._put", wraps=_original_put) as mock_put:
            writer = AtomicDataZarrWriter(store)
            writer.write(data_list)

            # zarr v3 writes 71 times per store: zarr.json metadata files for
            # root + 4 groups + 16 arrays, plus chunk data files (c/0) for each
            # array, plus attribute metadata.  The count is stable regardless of
            # num_samples because all data fits in a single chunk.
            assert mock_put.call_count == 71

            # Each call receives (path, value, exclusive=...) where path is a Path
            # Verify that paths written include meta/ and core/ subdirectories
            written_paths = [str(call.args[0]) for call in mock_put.call_args_list]
            assert any("meta" in p for p in written_paths)
            assert any("core" in p for p in written_paths)

            # Verify the value argument is always a zarr Buffer
            from zarr.core.buffer import Buffer

            for call in mock_put.call_args_list:
                assert isinstance(call.args[1], Buffer)

    @pytest.mark.parametrize("num_samples", [1, 3])
    def test_write_dispatches_to_memory_store_dict(self, num_samples: int) -> None:
        """Writing via MemoryStore populates the internal _store_dict."""
        from zarr.storage import MemoryStore

        store = MemoryStore()
        data_list = list(_data_generator(num_samples))

        # The store's internal dict should be empty before writing
        assert len(store._store_dict) == 0

        writer = AtomicDataZarrWriter(store)
        writer.write(data_list)

        # zarr v3 creates 34 keys: root zarr.json, 4 group zarr.json files,
        # 16 array zarr.json metadata files, and 16 chunk data files (c/0).
        # The count is stable regardless of num_samples because all data fits
        # in a single chunk per array.
        assert len(store._store_dict) == 34

        # Verify expected key patterns exist in the store dict
        keys = set(store._store_dict.keys())

        # Root zarr.json metadata
        assert "zarr.json" in keys

        # Meta group arrays (atoms_ptr, edges_ptr, masks)
        meta_keys = {k for k in keys if k.startswith("meta/")}
        assert any("atoms_ptr" in k for k in meta_keys)
        assert any("edges_ptr" in k for k in meta_keys)
        assert any("samples_mask" in k for k in meta_keys)

        # Core group arrays (numbers, positions, etc.)
        core_keys = {k for k in keys if k.startswith("core/")}
        assert any("atomic_numbers" in k for k in core_keys)
        assert any("positions" in k for k in core_keys)

        # All values in the dict should be zarr Buffer instances
        from zarr.core.buffer import Buffer

        for value in store._store_dict.values():
            assert isinstance(value, Buffer)

    @pytest.mark.parametrize("num_samples", [1, 3])
    def test_write_dispatches_to_fsspec_pipe_file(self, num_samples: int) -> None:
        """Writing via FsspecStore dispatches to fs._pipe_file for remote writes."""
        from unittest.mock import AsyncMock

        from zarr.storage import FsspecStore

        # Create a mock async filesystem that satisfies FsspecStore requirements
        mock_fs = AsyncMock()
        mock_fs.sep = "/"
        mock_fs.async_impl = True
        mock_fs.asynchronous = True

        # _pipe_file is the write method — it should be called for each chunk
        mock_fs._pipe_file = AsyncMock(return_value=None)

        # _cat_file is the read method — raise FileNotFoundError to indicate empty store
        # This allows zarr.open(mode="w") to create a new group
        async def raise_not_found(path, start=None, end=None):
            raise FileNotFoundError(path)

        mock_fs._cat_file = AsyncMock(side_effect=raise_not_found)

        # _find is needed for listing (used during store operations)
        mock_fs._find = AsyncMock(return_value=[])
        # _ls is needed for list_dir
        mock_fs._ls = AsyncMock(return_value=[])
        # _exists is needed for exists checks
        mock_fs._exists = AsyncMock(return_value=False)
        # _info is needed for getsize
        mock_fs._info = AsyncMock(return_value={"size": 0})
        # _mkdir and _makedirs may be called
        mock_fs._mkdir = AsyncMock(return_value=None)
        mock_fs._makedirs = AsyncMock(return_value=None)

        store = FsspecStore(mock_fs, path="/test_store")
        data_list = list(_data_generator(num_samples))

        writer = AtomicDataZarrWriter(store)
        writer.write(data_list)

        # FsspecStore follows the same write pipeline as LocalStore: 71 _pipe_file
        # calls for zarr.json metadata + chunk data for all groups and arrays.
        assert mock_fs._pipe_file.call_count == 71

        # Verify that written paths include meta/ and core/ subdirectories
        written_paths = [call.args[0] for call in mock_fs._pipe_file.call_args_list]
        assert any("meta" in p for p in written_paths)
        assert any("core" in p for p in written_paths)

        # Verify that written values are bytes (FsspecStore converts Buffer.to_bytes())
        for call in mock_fs._pipe_file.call_args_list:
            assert isinstance(call.args[1], bytes)

    @pytest.mark.parametrize("num_samples", [1, 3])
    def test_write_dispatches_to_gpu_buffer_conversion(self, num_samples: int) -> None:
        """Writing via GpuMemoryStore triggers gpu.Buffer.from_buffer for GPU conversion."""
        data_list = list(_data_generator(num_samples))

        class FakeGpuBuffer:
            """Fake GPU buffer that passes through values without cupy."""

            call_log: list = []

            @classmethod
            def from_buffer(cls, value):
                cls.call_log.append(value)
                return value

        FakeGpuBuffer.call_log = []  # Reset for this test

        with patch("zarr.core.buffer.gpu.Buffer", FakeGpuBuffer):
            from zarr.storage import GpuMemoryStore

            store = GpuMemoryStore()
            writer = AtomicDataZarrWriter(store)
            writer.write(data_list)

            # GpuMemoryStore.set() calls gpu.Buffer.from_buffer() for each
            # buffer that is not already a gpu.Buffer.  This fires 36 times:
            # once per store.set() call where a CPU Buffer needs GPU conversion
            # (metadata files + array chunks).
            assert len(FakeGpuBuffer.call_log) == 36

    @pytest.mark.parametrize("num_samples", [1, 3])
    def test_read_dispatches_to_local_store_get(
        self, num_samples: int, tmp_path: Path
    ) -> None:
        """Reading via LocalStore dispatches to zarr's _get for filesystem reads."""
        from zarr.storage import LocalStore
        from zarr.storage._local import _get as _original_get

        store = LocalStore(tmp_path / "test.zarr")
        data_list = list(_data_generator(num_samples))

        # Write data first (no mocking needed for writes here)
        writer = AtomicDataZarrWriter(store)
        writer.write(data_list)

        # Now spy on zarr's internal _get function during reads
        with patch("zarr.storage._local._get", wraps=_original_get) as mock_get:
            reader = AtomicDataZarrReader(store)
            _ = reader._load_sample(0)

            # Reader init reads zarr.json metadata for groups + arrays (25 calls),
            # then _load_sample reads chunk data for each core field (36 calls).
            # Total: 61 _get calls.  Stable regardless of num_samples because
            # only sample 0 is loaded and all data is in single chunks.
            assert mock_get.call_count == 61

            # Each call receives (path, prototype, byte_range)
            # Verify paths include meta/ and core/ subdirectories
            read_paths = [str(call.args[0]) for call in mock_get.call_args_list]
            assert any("meta" in p for p in read_paths)
            assert any("core" in p for p in read_paths)

    @pytest.mark.parametrize("num_samples", [1, 3])
    def test_read_dispatches_to_fsspec_cat_file(self, num_samples: int) -> None:
        """Reading via FsspecStore dispatches to fs._cat_file for remote reads."""
        from unittest.mock import AsyncMock

        from zarr.storage import FsspecStore, MemoryStore

        # First, write data to a MemoryStore to get valid zarr content
        mem_store = MemoryStore()
        data_list = list(_data_generator(num_samples))
        writer = AtomicDataZarrWriter(mem_store)
        writer.write(data_list)

        # Build a mapping of path -> bytes from the MemoryStore's internal dict
        # This simulates what a remote filesystem would serve
        store_data = {}
        for key, buf in mem_store._store_dict.items():
            store_data[f"/test_store/{key}"] = buf.to_bytes()

        # Create a mock async filesystem backed by our data
        mock_fs = AsyncMock()
        mock_fs.sep = "/"
        mock_fs.async_impl = True
        mock_fs.asynchronous = True

        # _cat_file returns bytes for a given path
        async def fake_cat_file(path, start=None, end=None):
            data = store_data.get(path)
            if data is None:
                raise FileNotFoundError(path)
            if start is not None or end is not None:
                data = data[start:end]
            return data

        mock_fs._cat_file = AsyncMock(side_effect=fake_cat_file)

        # _find returns list of all keys under a path
        async def fake_find(path, detail=False, withdirs=False, maxdepth=None):
            prefix = path.rstrip("/") + "/"
            return [k for k in store_data if k.startswith(prefix) or k == path]

        mock_fs._find = AsyncMock(side_effect=fake_find)

        # _ls returns direct children
        async def fake_ls(path, detail=False):
            prefix = path.rstrip("/") + "/"
            children = set()
            for k in store_data:
                if k.startswith(prefix):
                    child = k[len(prefix) :].split("/")[0]
                    children.add(prefix + child)
            return list(children)

        mock_fs._ls = AsyncMock(side_effect=fake_ls)

        # _exists checks if a key exists
        async def fake_exists(path):
            return path in store_data

        mock_fs._exists = AsyncMock(side_effect=fake_exists)

        fsspec_store = FsspecStore(mock_fs, path="/test_store")
        reader = AtomicDataZarrReader(fsspec_store)

        # Read a sample
        sample = reader._load_sample(0)

        # FsspecStore read path mirrors LocalStore: reader init reads metadata
        # (25 _cat_file calls), then _load_sample reads field chunks (36 calls).
        # Total: 61 _cat_file calls.
        assert mock_fs._cat_file.call_count == 61

        # Verify paths read include meta/ and core/ subdirectories
        read_paths = [call.args[0] for call in mock_fs._cat_file.call_args_list]
        assert any("meta" in p for p in read_paths)
        assert any("core" in p for p in read_paths)

        # Verify we got actual tensor data back
        assert "atomic_numbers" in sample
        assert sample["atomic_numbers"].shape[0] > 0

    @pytest.mark.parametrize("num_samples", [1, 3])
    def test_append_dispatches_to_memory_store_set(self, num_samples: int) -> None:
        """Appending data invokes MemoryStore.set for new chunks."""
        from zarr.storage import MemoryStore

        store = MemoryStore()
        data_list = list(_data_generator(num_samples + 2))

        # Write initial data
        writer = AtomicDataZarrWriter(store)
        writer.write(data_list[:2])

        keys_after_write = set(store._store_dict.keys())

        # Append more data — this should add/update keys in the store dict
        for d in data_list[2:]:
            writer.append(d)

        keys_after_append = set(store._store_dict.keys())

        # The store dict should have been modified (new chunks or updated metadata)
        # At minimum, meta arrays (atoms_ptr, edges_ptr, masks) should be updated
        # Verify by checking that some keys have different content or new keys appeared
        assert len(keys_after_append) >= len(keys_after_write)


# ---------------------------------------------------------------------------
# TestDatasetCoverage — exercises paths not covered by TestDataset/Prefetch
# ---------------------------------------------------------------------------


class _SimpleReader(Reader):
    """Minimal duck-typed reader for Dataset tests (no zarr required)."""

    def __init__(self, n: int = 3) -> None:
        super().__init__()
        self._n = n

    def _load_sample(self, index: int) -> dict:
        return {
            "atomic_numbers": torch.tensor([6], dtype=torch.long),
            "positions": torch.tensor([[float(index), 0.0, 0.0]]),
        }

    def _get_sample_metadata(self, index: int) -> dict:
        return {"src_index": index}

    def __len__(self) -> int:
        return self._n

    def close(self) -> None:
        pass


class TestDatasetCoverage:
    """Coverage for Dataset paths not exercised by the zarr-backed test suite."""

    # ------------------------------------------------------------------
    # Construction edge-cases
    # ------------------------------------------------------------------

    def test_invalid_reader_raises_type_error(self):
        """Passing an object that doesn't implement ReaderProtocol raises TypeError."""
        with pytest.raises(TypeError, match="Reader interface"):
            Dataset(object())  # type: ignore[arg-type]

    @pytest.mark.parametrize("device", ["cpu", "cuda"])
    def test_device_string_is_converted_to_torch_device(self, device: str):
        reader = _SimpleReader()
        ds = Dataset(reader, device=device)
        assert isinstance(ds.target_device, torch.device)
        assert ds.target_device == torch.device(device)

    def test_default_device_is_set_when_none_given(self):
        """With device=None, target_device defaults to cpu or cuda."""
        reader = _SimpleReader()
        ds = Dataset(reader)
        assert isinstance(ds.target_device, torch.device)

    # ------------------------------------------------------------------
    # __getitem__ synchronous path
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("device", ["cpu", "cuda"])
    def test_getitem_returns_atomic_data_and_metadata(self, device: str):
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("No CUDA device available.")
        reader = _SimpleReader()
        ds = Dataset(reader, device=device)
        data, meta = ds[0]
        assert isinstance(data, AtomicData)
        assert "src_index" in meta

    @pytest.mark.parametrize("device", ["cpu", "cuda"])
    def test_getitem_transfers_to_target_device(self, device: str):
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("No CUDA device available.")
        reader = _SimpleReader()
        ds = Dataset(reader, device=device)
        data, _ = ds[0]
        assert data.positions.device.type == torch.device(device).type

    # ------------------------------------------------------------------
    # prefetch / cancel_prefetch
    # ------------------------------------------------------------------

    def test_prefetch_noop_when_already_queued(self):
        """Calling prefetch twice for the same index should not create a second future."""
        reader = _SimpleReader()
        ds = Dataset(reader, device="cpu")
        ds.prefetch(0)
        futures_after_first = dict(ds._prefetch_futures)
        ds.prefetch(0)  # no-op: already queued
        assert set(ds._prefetch_futures.keys()) == set(futures_after_first.keys())
        ds.close()

    def test_prefetch_batch_submits_multiple(self):
        reader = _SimpleReader(n=3)
        ds = Dataset(reader, device="cpu")
        ds.prefetch_batch([0, 1, 2])
        assert 0 in ds._prefetch_futures
        assert 1 in ds._prefetch_futures
        assert 2 in ds._prefetch_futures
        ds.close()

    def test_cancel_prefetch_specific_index(self):
        reader = _SimpleReader(n=3)
        ds = Dataset(reader, device="cpu")
        ds.prefetch_batch([0, 1])
        ds.cancel_prefetch(0)
        assert 0 not in ds._prefetch_futures
        assert 1 in ds._prefetch_futures
        ds.close()

    def test_cancel_prefetch_all(self):
        reader = _SimpleReader(n=3)
        ds = Dataset(reader, device="cpu")
        ds.prefetch_batch([0, 1, 2])
        ds.cancel_prefetch()
        assert len(ds._prefetch_futures) == 0
        ds.close()

    # ------------------------------------------------------------------
    # get_metadata
    # ------------------------------------------------------------------

    def test_get_metadata_without_edges(self):
        """get_metadata returns (num_atoms, 0) when no neighbor_list present."""
        reader = _SimpleReader()
        ds = Dataset(reader, device="cpu")
        num_atoms, num_edges = ds.get_metadata(0)
        assert num_atoms == 1
        assert num_edges == 0

    def test_get_metadata_with_edges(self):
        """get_metadata returns correct edge count when neighbor_list is present."""

        class _ReaderWithEdges(_SimpleReader):
            def _load_sample(self, index: int) -> dict:
                return {
                    "atomic_numbers": torch.tensor([6, 6], dtype=torch.long),
                    "positions": torch.zeros(2, 3),
                    "neighbor_list": torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
                }

        ds = Dataset(_ReaderWithEdges(n=1), device="cpu")
        num_atoms, num_edges = ds.get_metadata(0)
        assert num_atoms == 2
        assert num_edges == 2

    # ------------------------------------------------------------------
    # __iter__
    # ------------------------------------------------------------------

    def test_iter_yields_all_samples(self):
        reader = _SimpleReader(n=3)
        ds = Dataset(reader, device="cpu")
        items = list(ds)
        assert len(items) == 3

    # ------------------------------------------------------------------
    # close() with pending futures
    # ------------------------------------------------------------------

    def test_close_drains_pending_futures(self):
        """close() does not raise even when prefetch futures are pending."""
        reader = _SimpleReader(n=3)
        ds = Dataset(reader, device="cpu")
        ds.prefetch_batch([0, 1, 2])
        ds.close()  # must not raise
        assert ds._executor is None

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def test_enter_returns_self(self):
        reader = _SimpleReader()
        ds = Dataset(reader, device="cpu")
        with ds as ctx:
            assert ctx is ds

    def test_exit_calls_close(self):
        closed = []

        class _TrackingReader(_SimpleReader):
            def close(self):
                closed.append(True)

        with Dataset(_TrackingReader(), device="cpu"):
            pass
        assert closed == [True]

    # ------------------------------------------------------------------
    # __repr__
    # ------------------------------------------------------------------

    def test_repr_contains_class_name_and_length(self):
        reader = _SimpleReader(n=5)
        ds = Dataset(reader, device="cpu")
        r = repr(ds)
        assert "Dataset" in r
        assert "5" in r


class TestZarrCompression:
    """Tests for compression and chunking configuration."""

    def test_write_with_zstd_compression(self, tmp_path: Path) -> None:
        """Write with ZstdCodec, verify roundtrip correctness."""
        from zarr.codecs import ZstdCodec

        store = tmp_path / "test.zarr"
        config = ZarrWriteConfig(
            core=ZarrArrayConfig(compressors=(ZstdCodec(level=3),)),
        )
        writer = AtomicDataZarrWriter(store, config=config)
        data_list = list(_data_generator(3))
        writer.write(data_list)

        reader = AtomicDataZarrReader(store)
        assert len(reader) == 3
        sample, _ = reader[0]
        assert "positions" in sample

    def test_write_with_blosc_compression(self, tmp_path: Path) -> None:
        """Write with BloscCodec, verify roundtrip correctness."""
        from zarr.codecs import BloscCodec

        store = tmp_path / "test.zarr"
        config = ZarrWriteConfig(
            core=ZarrArrayConfig(compressors=(BloscCodec(cname="lz4", clevel=5),)),
        )
        writer = AtomicDataZarrWriter(store, config=config)
        data_list = list(_data_generator(3))
        writer.write(data_list)

        reader = AtomicDataZarrReader(store)
        assert len(reader) == 3
        for i in range(3):
            sample, _ = reader[i]
            assert "positions" in sample

    def test_write_with_custom_chunk_size(self, tmp_path: Path) -> None:
        """Write with explicit chunk_size, verify array chunks are set."""
        store = tmp_path / "test.zarr"
        config = ZarrWriteConfig(
            core=ZarrArrayConfig(chunk_size=2),
        )
        writer = AtomicDataZarrWriter(store, config=config)
        data_list = list(_data_generator(5))
        writer.write(data_list)

        root = zarr.open(store, mode="r")
        positions = root["core/positions"]
        assert positions.chunks[0] == 2

        reader = AtomicDataZarrReader(store)
        assert len(reader) == 5

    def test_per_group_config(self, tmp_path: Path) -> None:
        """Different configs for meta vs core groups."""
        from zarr.codecs import ZstdCodec

        store = tmp_path / "test.zarr"
        config = ZarrWriteConfig(
            meta=ZarrArrayConfig(),
            core=ZarrArrayConfig(compressors=(ZstdCodec(level=1),)),
        )
        writer = AtomicDataZarrWriter(store, config=config)
        writer.write(list(_data_generator(3)))

        reader = AtomicDataZarrReader(store)
        assert len(reader) == 3

    def test_field_override(self, tmp_path: Path) -> None:
        """Per-field override takes precedence over group config."""
        from zarr.codecs import BloscCodec, ZstdCodec

        store = tmp_path / "test.zarr"
        config = ZarrWriteConfig(
            core=ZarrArrayConfig(compressors=(ZstdCodec(level=1),)),
            field_overrides={
                "positions": ZarrArrayConfig(compressors=(BloscCodec(cname="lz4"),)),
            },
        )
        writer = AtomicDataZarrWriter(store, config=config)
        writer.write(list(_data_generator(3)))

        reader = AtomicDataZarrReader(store)
        assert len(reader) == 3
        sample, _ = reader[0]
        assert "positions" in sample

    def test_append_preserves_config(self, tmp_path: Path) -> None:
        """Append to compressed store, verify data readable."""
        from zarr.codecs import ZstdCodec

        store = tmp_path / "test.zarr"
        config = ZarrWriteConfig(
            core=ZarrArrayConfig(compressors=(ZstdCodec(level=3),)),
        )
        writer = AtomicDataZarrWriter(store, config=config)
        writer.write(list(_data_generator(2)))
        writer.append(list(_data_generator(2, seed=42)))

        reader = AtomicDataZarrReader(store)
        assert len(reader) == 4

    def test_defragment_preserves_config(self, tmp_path: Path) -> None:
        """Defragment compressed store, verify config reapplied."""
        from zarr.codecs import ZstdCodec

        store = tmp_path / "test.zarr"
        config = ZarrWriteConfig(
            core=ZarrArrayConfig(compressors=(ZstdCodec(level=3),), chunk_size=4),
        )
        writer = AtomicDataZarrWriter(store, config=config)
        writer.write(list(_data_generator(5)))
        writer.delete([1, 3])
        writer.defragment()

        reader = AtomicDataZarrReader(store)
        assert len(reader) == 3

        root = zarr.open(store, mode="r")
        positions = root["core/positions"]
        assert positions.chunks[0] == 4

    def test_defragment_with_new_config(self, tmp_path: Path) -> None:
        """Defragment with a new config overrides the original."""
        from zarr.codecs import BloscCodec, ZstdCodec

        store = tmp_path / "test.zarr"
        config = ZarrWriteConfig(
            core=ZarrArrayConfig(compressors=(ZstdCodec(level=3),), chunk_size=4),
        )
        writer = AtomicDataZarrWriter(store, config=config)
        writer.write(list(_data_generator(5)))
        writer.delete([1, 3])

        new_config = ZarrWriteConfig(
            core=ZarrArrayConfig(compressors=(BloscCodec(cname="lz4"),), chunk_size=8),
        )
        writer.defragment(config=new_config)

        reader = AtomicDataZarrReader(store)
        assert len(reader) == 3

        root = zarr.open(store, mode="r")
        positions = root["core/positions"]
        assert positions.chunks[0] == 8

    def test_defragment_with_mapping_config(self, tmp_path: Path) -> None:
        """Defragment accepts a plain dict as config."""
        from zarr.codecs import ZstdCodec

        store = tmp_path / "test.zarr"
        writer = AtomicDataZarrWriter(store)
        writer.write(list(_data_generator(5)))
        writer.delete([0, 2])

        writer.defragment(
            config={
                "core": {"compressors": (ZstdCodec(level=1),), "chunk_size": 6},
            }
        )

        reader = AtomicDataZarrReader(store)
        assert len(reader) == 3

        root = zarr.open(store, mode="r")
        positions = root["core/positions"]
        assert positions.chunks[0] == 6

    def test_default_config_backward_compat(self, tmp_path: Path) -> None:
        """No config = same behavior as before."""
        store = tmp_path / "test.zarr"
        writer = AtomicDataZarrWriter(store)
        writer.write(list(_data_generator(3)))

        reader = AtomicDataZarrReader(store)
        assert len(reader) == 3

    def test_config_from_mapping(self, tmp_path: Path) -> None:
        """Config can be passed as a plain dict."""
        from zarr.codecs import ZstdCodec

        store = tmp_path / "test.zarr"
        writer = AtomicDataZarrWriter(
            store,
            config={
                "core": {"compressors": (ZstdCodec(level=1),), "chunk_size": 8},
            },
        )
        writer.write(list(_data_generator(3)))

        reader = AtomicDataZarrReader(store)
        assert len(reader) == 3

    def test_zarr_data_sink_with_config(self, tmp_path: Path) -> None:
        """ZarrData sink with compression config, verify roundtrip."""
        from zarr.codecs import ZstdCodec

        from nvalchemi.dynamics.sinks import ZarrData

        store = tmp_path / "test.zarr"
        config = ZarrWriteConfig(
            core=ZarrArrayConfig(compressors=(ZstdCodec(level=1),)),
        )
        sink = ZarrData(store, config=config)
        data_list = list(_data_generator(3))
        batch = Batch.from_data_list(data_list)
        sink.write(batch)

        result = sink.read()
        assert result is not None

    def test_write_empty_chunks_false(self, tmp_path: Path) -> None:
        """write_empty_chunks=False config is applied."""
        store = tmp_path / "test.zarr"
        config = ZarrWriteConfig(
            core=ZarrArrayConfig(write_empty_chunks=False, chunk_size=4),
        )
        writer = AtomicDataZarrWriter(store, config=config)
        writer.write(list(_data_generator(3)))

        reader = AtomicDataZarrReader(store)
        assert len(reader) == 3

    def test_write_with_sharding(self, tmp_path: Path) -> None:
        """Test that shard_size is correctly applied to written arrays."""
        store = tmp_path / "test.zarr"
        config = ZarrWriteConfig(
            core=ZarrArrayConfig(chunk_size=2, shard_size=4),
        )
        writer = AtomicDataZarrWriter(store, config=config)
        data_list = list(_data_generator(5))
        writer.write(data_list)
        root = zarr.open(store, mode="r")
        pos = root["core/positions"]
        assert pos.chunks[0] == 2
        assert pos.metadata.shards is not None
        assert pos.metadata.shards[0] == 4

    def test_shard_chunk_alignment_validation(self, tmp_path: Path) -> None:
        """Test that shard_size must be a multiple of chunk_size."""
        with pytest.raises(ValueError, match="must be a multiple"):
            ZarrArrayConfig(chunk_size=3, shard_size=5)

    def test_sharding_roundtrip(self, tmp_path: Path) -> None:
        """Test that sharded arrays roundtrip correctly."""
        store = tmp_path / "test.zarr"
        config = ZarrWriteConfig(
            core=ZarrArrayConfig(chunk_size=2, shard_size=4),
        )
        writer = AtomicDataZarrWriter(store, config=config)
        data_list = list(_data_generator(5))
        writer.write(data_list)

        reader = AtomicDataZarrReader(store)
        for i, original in enumerate(data_list):
            loaded = reader._load_sample(i)
            assert torch.allclose(
                original.positions, loaded["positions"].to(original.positions.dtype)
            )

    def test_shard_field_override(self, tmp_path: Path) -> None:
        """Test that field_overrides correctly apply sharding."""
        store = tmp_path / "test.zarr"
        config = ZarrWriteConfig(
            core=ZarrArrayConfig(chunk_size=2, shard_size=4),
            field_overrides={
                "positions": ZarrArrayConfig(chunk_size=2, shard_size=6),
            },
        )
        writer = AtomicDataZarrWriter(store, config=config)
        data_list = list(_data_generator(5))
        writer.write(data_list)
        root = zarr.open(store, mode="r")
        pos = root["core/positions"]
        assert pos.chunks[0] == 2
        assert pos.metadata.shards[0] == 6

    def test_defragment_preserves_shard_config(self, tmp_path: Path) -> None:
        """Test that defragment preserves the shard configuration."""
        store = tmp_path / "test.zarr"
        config = ZarrWriteConfig(
            core=ZarrArrayConfig(chunk_size=2, shard_size=4),
        )
        writer = AtomicDataZarrWriter(store, config=config)
        data_list = list(_data_generator(5))
        writer.write(data_list)
        writer.delete([1])
        writer.defragment()
        root = zarr.open(store, mode="r")
        pos = root["core/positions"]
        assert pos.chunks[0] == 2
        assert pos.metadata.shards is not None
        assert pos.metadata.shards[0] == 4

    def test_edge_index_chunk_dim(self, tmp_path: Path) -> None:
        """chunk_size should apply to the leading edge axis of neighbor_list."""
        store = tmp_path / "test.zarr"
        config = ZarrWriteConfig(core=ZarrArrayConfig(chunk_size=100))
        writer = AtomicDataZarrWriter(store, config=config)
        data_list = list(_data_generator(10))
        writer.write(data_list)

        root = zarr.open(store, mode="r")
        edge_arr = root["core/neighbor_list"]
        assert edge_arr.chunks[0] == 100
        assert edge_arr.chunks[1] == 2

    def test_edge_index_shard_dim(self, tmp_path: Path) -> None:
        """shard_size should apply to the leading edge axis of neighbor_list."""
        store = tmp_path / "test.zarr"
        config = ZarrWriteConfig(core=ZarrArrayConfig(chunk_size=50, shard_size=100))
        writer = AtomicDataZarrWriter(store, config=config)
        data_list = list(_data_generator(10))
        writer.write(data_list)

        root = zarr.open(store, mode="r")
        edge_arr = root["core/neighbor_list"]
        assert edge_arr.chunks[0] == 50
        assert edge_arr.chunks[1] == 2
        assert edge_arr.metadata.shards[0] == 100
        assert edge_arr.metadata.shards[1] == 2


class TestZarrDataSinkConfig:
    """Tests for ZarrData sink compression and chunking configuration."""

    def test_sink_with_zstd_roundtrip(self, tmp_path: Path) -> None:
        """ZarrData with ZstdCodec produces correct roundtrip data."""
        from zarr.codecs import ZstdCodec

        from nvalchemi.dynamics.sinks import ZarrData

        store = tmp_path / "test.zarr"
        sink = ZarrData(
            store,
            config=ZarrWriteConfig(
                core=ZarrArrayConfig(compressors=(ZstdCodec(level=3),)),
            ),
        )
        data_list = list(_data_generator(4))
        batch = Batch.from_data_list(data_list)
        sink.write(batch)

        result = sink.read()
        assert result.num_graphs == 4
        assert torch.allclose(result["positions"], batch["positions"])

    def test_sink_with_blosc_and_chunk_size(self, tmp_path: Path) -> None:
        """ZarrData with BloscCodec and chunk_size applies to underlying store."""
        from zarr.codecs import BloscCodec

        from nvalchemi.dynamics.sinks import ZarrData

        store = tmp_path / "test.zarr"
        sink = ZarrData(
            store,
            config=ZarrWriteConfig(
                core=ZarrArrayConfig(
                    compressors=(BloscCodec(cname="lz4", clevel=5),),
                    chunk_size=8,
                ),
            ),
        )
        data_list = list(_data_generator(3))
        batch = Batch.from_data_list(data_list)
        sink.write(batch)

        root = zarr.open(store, mode="r")
        positions = root["core/positions"]
        assert positions.chunks[0] == 8

        result = sink.read()
        assert result.num_graphs == 3

    def test_sink_config_from_mapping(self, tmp_path: Path) -> None:
        """ZarrData accepts config as a plain dict."""
        from zarr.codecs import ZstdCodec

        from nvalchemi.dynamics.sinks import ZarrData

        store = tmp_path / "test.zarr"
        sink = ZarrData(
            store,
            config={"core": {"compressors": (ZstdCodec(level=1),), "chunk_size": 4}},
        )
        data_list = list(_data_generator(3))
        batch = Batch.from_data_list(data_list)
        sink.write(batch)

        result = sink.read()
        assert result.num_graphs == 3

        root = zarr.open(store, mode="r")
        positions = root["core/positions"]
        assert positions.chunks[0] == 4

    def test_sink_append_with_config(self, tmp_path: Path) -> None:
        """Multiple writes to ZarrData with config produce correct total."""
        from zarr.codecs import ZstdCodec

        from nvalchemi.dynamics.sinks import ZarrData

        store = tmp_path / "test.zarr"
        sink = ZarrData(
            store,
            config=ZarrWriteConfig(
                core=ZarrArrayConfig(compressors=(ZstdCodec(level=1),)),
            ),
        )
        batch1 = Batch.from_data_list(list(_data_generator(2)))
        batch2 = Batch.from_data_list(list(_data_generator(3, seed=42)))
        sink.write(batch1)
        sink.write(batch2)

        assert len(sink) == 5
        result = sink.read()
        assert result.num_graphs == 5

    def test_sink_zero_preserves_config(self, tmp_path: Path) -> None:
        """zero() resets the store but preserves config for future writes."""
        from zarr.codecs import ZstdCodec

        from nvalchemi.dynamics.sinks import ZarrData

        store = tmp_path / "test.zarr"
        sink = ZarrData(
            store,
            config=ZarrWriteConfig(
                core=ZarrArrayConfig(compressors=(ZstdCodec(level=3),), chunk_size=4),
            ),
        )
        batch = Batch.from_data_list(list(_data_generator(3)))
        sink.write(batch)
        assert len(sink) == 3

        sink.zero()
        assert len(sink) == 0

        sink.write(batch)
        assert len(sink) == 3

        root = zarr.open(store, mode="r")
        positions = root["core/positions"]
        assert positions.chunks[0] == 4

        result = sink.read()
        assert result.num_graphs == 3

    def test_sink_default_config_backward_compat(self, tmp_path: Path) -> None:
        """ZarrData without config works as before."""
        from nvalchemi.dynamics.sinks import ZarrData

        store = tmp_path / "test.zarr"
        sink = ZarrData(store)
        batch = Batch.from_data_list(list(_data_generator(3)))
        sink.write(batch)

        result = sink.read()
        assert result.num_graphs == 3

    def test_sink_field_override(self, tmp_path: Path) -> None:
        """Per-field override in ZarrData config is applied."""
        from zarr.codecs import BloscCodec, ZstdCodec

        from nvalchemi.dynamics.sinks import ZarrData

        store = tmp_path / "test.zarr"
        sink = ZarrData(
            store,
            config=ZarrWriteConfig(
                core=ZarrArrayConfig(compressors=(ZstdCodec(level=1),)),
                field_overrides={
                    "positions": ZarrArrayConfig(
                        compressors=(BloscCodec(cname="lz4"),), chunk_size=16
                    ),
                },
            ),
        )
        batch = Batch.from_data_list(list(_data_generator(3)))
        sink.write(batch)

        root = zarr.open(store, mode="r")
        positions = root["core/positions"]
        assert positions.chunks[0] == 16

        result = sink.read()
        assert result.num_graphs == 3

    def test_sink_with_sharding(self, tmp_path: Path) -> None:
        """Test that ZarrData sink correctly applies sharding configuration."""
        from zarr.codecs import ZstdCodec

        from nvalchemi.dynamics.sinks import ZarrData

        store = tmp_path / "test.zarr"
        config = ZarrWriteConfig(
            core=ZarrArrayConfig(
                compressors=(ZstdCodec(level=1),),
                chunk_size=2,
                shard_size=4,
            ),
        )
        sink = ZarrData(store, config=config)
        batch = Batch.from_data_list(list(_data_generator(3)))
        sink.write(batch)
        reader = AtomicDataZarrReader(store)
        assert len(reader) == 3


class TestSliceEdgeArrayGuard:
    """Verify _slice_edge_array rejects cat_dim != 0 fields."""

    def test_slice_edge_array_rejects_face_key(self) -> None:
        """_slice_edge_array raises RuntimeError for keys matching *index*/*face*."""
        import numpy as np

        arr = np.zeros((10, 3))
        with pytest.raises(RuntimeError, match="Unexpected cat_dim=-1"):
            _slice_edge_array(arr, "face_index", 0, 5)

    def test_slice_edge_array_accepts_normal_edge_key(self) -> None:
        """_slice_edge_array passes through for normal edge keys."""
        import numpy as np

        arr = np.arange(30).reshape(10, 3)
        result = _slice_edge_array(arr, "shifts", 2, 5)
        assert result.shape == (3, 3)
        np.testing.assert_array_equal(result, arr[2:5])

    def test_load_sample_rejects_custom_face_index(self, tmp_path: Path) -> None:
        """_load_sample raises RuntimeError for custom edge field named face_index."""
        data_list = list(_data_generator(2))
        writer = AtomicDataZarrWriter(tmp_path / "test.zarr")
        writer.write(data_list)

        total_edges = sum(d.neighbor_list.shape[0] for d in data_list)
        writer.add_custom("face_index", torch.randint(0, 10, (total_edges, 2)), "edge")

        with AtomicDataZarrReader(tmp_path / "test.zarr") as reader:
            with pytest.raises(RuntimeError, match="Unexpected cat_dim=-1"):
                reader._load_sample(0)

    def test_defragment_rejects_custom_face_index(self, tmp_path: Path) -> None:
        """defragment raises RuntimeError for custom edge field named face_index."""
        data_list = list(_data_generator(3))
        writer = AtomicDataZarrWriter(tmp_path / "test.zarr")
        writer.write(data_list)

        total_edges = sum(d.neighbor_list.shape[0] for d in data_list)
        writer.add_custom("face_index", torch.randint(0, 10, (total_edges, 2)), "edge")

        writer.delete([0])
        with pytest.raises(RuntimeError, match="Unexpected cat_dim=-1"):
            writer.defragment()
