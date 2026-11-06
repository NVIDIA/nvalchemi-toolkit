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
"""Tests for :mod:`nvalchemi.hooks.neighbor_list`.

Covers all improvements made to NeighborListHook:

* Staging-buffer pre-allocation and copy semantics
* Algorithm-specific kwarg pre-allocation (``_alloc_nl_kwargs``)
* Shape-change invalidation
* Cache reset at ``ON_ADMISSION`` when new systems keep the batch shape
* Verlet skin-check integration
* Correct MATRIX and COO output written to the batch
* ``@torch.compiler.disable`` isolation of allocation code
"""

from __future__ import annotations

import itertools

import pytest
import torch

from nvalchemi.data import AtomicData, Batch
from nvalchemi.dynamics import FIRE2, FusedStage, SizeAwareSampler
from nvalchemi.dynamics.base import DynamicsStage
from nvalchemi.dynamics.demo import DemoDynamics
from nvalchemi.hooks import NeighborListHook, WrapPeriodicHook
from nvalchemi.hooks._context import HookContext
from nvalchemi.hooks._protocol import Hook
from nvalchemi.models.base import NeighborConfig, NeighborListFormat
from nvalchemi.models.demo import DemoModel, DemoModelWrapper
from nvalchemi.models.lj import LennardJonesModelWrapper
from nvalchemi.training import EnergyMSELoss, TrainingStage
from nvalchemi.training.optimizers import OptimizerConfig
from nvalchemi.training.strategy import TrainingStrategy, default_training_fn

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_CUTOFF = 2.5
_STAGE = DynamicsStage.BEFORE_COMPUTE
_GRID_SITES = torch.tensor(list(itertools.product(range(3), repeat=3))).float()


def _ctx(batch: Batch) -> HookContext:
    return HookContext(batch=batch)


def _cfg(
    fmt: NeighborListFormat = NeighborListFormat.MATRIX,
    cutoff: float = _CUTOFF,
    half_list: bool = False,
    skin: float = 0.0,
) -> NeighborConfig:
    return NeighborConfig(
        cutoff=cutoff,
        format=fmt,
        half_list=half_list,
        skin=skin,
    )


def _line_batch(
    device: str,
    *,
    pbc: bool = False,
    n_graphs: int = 1,
    cell_size: float = 20.0,
    int_dtype: torch.dtype = torch.long,
) -> Batch:
    """Three atoms per graph in a line: [0,0,0], [1.5,0,0], [5,0,0].

    Atoms 0 and 1 are within *_CUTOFF* = 2.5 Å of each other.
    Atom 2 is isolated (distance to atom 1 is 3.5 Å > cutoff).
    """
    data_list = []
    for _ in range(n_graphs):
        pos = torch.tensor([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [5.0, 0.0, 0.0]])
        kwargs: dict = dict(
            positions=pos,
            atomic_numbers=torch.tensor([1, 1, 1], dtype=int_dtype),
        )
        if pbc:
            kwargs["cell"] = torch.eye(3).unsqueeze(0) * cell_size
            kwargs["pbc"] = torch.tensor([[True, True, True]])
        data_list.append(AtomicData(**kwargs))
    return Batch.from_data_list(data_list).to(device)


def _pbc_wrap_batch(device: str) -> Batch:
    """Two atoms that are close only through the periodic image.

    Atom 0 at [0.1, 0, 0], atom 1 at [2.9, 0, 0] in a 3 Å cell.
    Direct distance = 2.8 Å (> cutoff 2.5), wrapped distance = 0.2 Å (< cutoff).
    """
    pos = torch.tensor([[0.1, 0.0, 0.0], [2.9, 0.0, 0.0]])
    data = AtomicData(
        positions=pos,
        atomic_numbers=torch.tensor([1, 1], dtype=torch.long),
        cell=torch.eye(3).unsqueeze(0) * 3.0,
        pbc=torch.tensor([[True, True, True]]),
    )
    return Batch.from_data_list([data]).to(device)


def _is_neighbor(
    neighbor_matrix: torch.Tensor,
    num_neighbors: torch.Tensor,
    i: int,
    j: int,
) -> bool:
    """Return True if atom j appears in atom i's neighbor list."""
    n = int(num_neighbors[i].item())
    return j in neighbor_matrix[i, :n].tolist()


def _sorted_pairs(batch: Batch) -> torch.Tensor:
    """Return each atom's ``(neighbor, shift)`` pairs as sorted integer keys.

    Padding slots get the largest key, so rows compare equal regardless of
    the order the neighbors were written in.
    """
    neighbors = batch.neighbor_matrix.long()
    # Image indices are small, so each fits in 4 bits after the offset.
    sx, sy, sz = (batch.neighbor_matrix_shifts.long() + 8).unbind(-1)
    keys = ((neighbors * 16 + sx) * 16 + sy) * 16 + sz
    keys = keys.masked_fill(neighbors >= batch.num_nodes, torch.iinfo(torch.long).max)
    return keys.sort(dim=1).values


def _cluster_batch(
    sizes: list[int],
    device: str,
    *,
    pbc: bool = False,
    positions: torch.Tensor | None = None,
) -> Batch:
    """One compact cluster per graph, with the given atom counts.

    Every atom lies within ``_CUTOFF`` of every other atom of its cluster, so
    an ``n``-atom cluster needs ``n - 1`` neighbor slots per atom.
    *positions*, when given, replaces the positions of all atoms.
    """
    data_list = []
    for n in sizes:
        kwargs: dict = dict(
            positions=0.5 * _GRID_SITES[:n],
            atomic_numbers=torch.ones(n, dtype=torch.long),
        )
        if pbc:
            kwargs["cell"] = torch.eye(3).unsqueeze(0) * 20.0
            kwargs["pbc"] = torch.tensor([[True, True, True]])
        data_list.append(AtomicData(**kwargs))
    batch = Batch.from_data_list(data_list).to(device)
    if positions is not None:
        batch.positions.copy_(positions)
    return batch


def _neighbor_sets(batch: Batch) -> list[set[int]]:
    """Return each atom's neighbors as listed in the batch's neighbor matrix."""
    counts = batch.num_neighbors.tolist()
    return [set(row[:n]) for row, n in zip(batch.neighbor_matrix.tolist(), counts)]


def _edge_set(batch: Batch) -> set[tuple[int, int]]:
    """Return the batch's COO neighbor list as a set of (src, dst) pairs."""
    return {tuple(edge) for edge in batch.neighbor_list.tolist()}


def _fresh_build(
    batch: Batch, config: NeighborConfig
) -> tuple[NeighborListHook, Batch]:
    """Return a new hook and a copy of *batch* holding the hook's first build."""
    hook = NeighborListHook(config, stage=_STAGE)
    fresh = batch.clone()
    hook(_ctx(fresh), _STAGE)
    return hook, fresh


class _ClusterDataset:
    """Lennard-Jones clusters near equilibrium, sampled in dataset order."""

    def __init__(self, sizes: list[int]) -> None:
        self._sizes = sizes

    def __len__(self) -> int:
        return len(self._sizes)

    def __getitem__(self, idx: int) -> tuple[AtomicData, dict]:
        n = self._sizes[idx]
        data = AtomicData(
            positions=1.1 * _GRID_SITES[:n],
            atomic_numbers=torch.full((n,), 18, dtype=torch.long),
            forces=torch.zeros(n, 3),
            energy=torch.zeros(1, 1),
        )
        data.add_node_property("velocities", torch.zeros(n, 3))
        return data, {}

    def get_metadata(self, idx: int) -> tuple[int, int]:
        return self._sizes[idx], 0


class _FreshBuildCheck:
    """Compare every neighbor list built for the batch with a fresh hook's."""

    stage = DynamicsStage.BEFORE_COMPUTE
    frequency = 1

    def __init__(self, config: NeighborConfig) -> None:
        self.config = config
        self.checks: list[tuple[list[int], bool]] = []

    def __call__(self, ctx: HookContext, stage: DynamicsStage) -> None:
        _, fresh = _fresh_build(ctx.batch, self.config)
        sizes = ctx.batch.num_nodes_per_graph.tolist()
        self.checks.append((sizes, _neighbor_sets(ctx.batch) == _neighbor_sets(fresh)))


# ===========================================================================
# TestNeighborListHookProtocol
# ===========================================================================


class TestNeighborListHookProtocol:
    def test_stage(self):
        hook = NeighborListHook(_cfg(), stage=DynamicsStage.BEFORE_COMPUTE)
        assert hook.stage == DynamicsStage.BEFORE_COMPUTE

    def test_frequency(self):
        assert (
            NeighborListHook(_cfg(), stage=DynamicsStage.BEFORE_COMPUTE).frequency == 1
        )

    def test_hook_protocol(self):
        assert isinstance(
            NeighborListHook(_cfg(), stage=DynamicsStage.BEFORE_COMPUTE), Hook
        )

    def test_explicit_method_override(self):
        hook = NeighborListHook(
            _cfg(), method="batch_naive_tile", stage=DynamicsStage.BEFORE_COMPUTE
        )
        batch = _line_batch("cpu", pbc=False)

        method = hook._select_neighbor_list_method(
            batch.num_nodes,
            batch.num_graphs,
            batch.batch_ptr,
            None,
            None,
            batch.positions.dtype,
        )

        assert method == "batch_naive_tile"


# ===========================================================================
# TestStagingBufferAllocation
# ===========================================================================


class TestStagingBufferAllocation:
    """Verify staging buffers are allocated with correct shapes on first call."""

    def test_buffers_none_before_first_call(self):
        hook = NeighborListHook(_cfg(), stage=DynamicsStage.BEFORE_COMPUTE)
        assert hook._buf_positions is None
        assert hook._buf_batch_ptr is None
        assert hook._buf_batch_idx is None

    def test_buffers_allocated_after_first_call(self, device: str):
        hook = NeighborListHook(_cfg(), stage=DynamicsStage.BEFORE_COMPUTE)
        batch = _line_batch(device)
        hook(_ctx(batch), _STAGE)

        assert hook._buf_positions is not None
        assert hook._buf_batch_ptr is not None
        assert hook._buf_batch_idx is not None

    def test_buffer_shapes(self, device: str):
        hook = NeighborListHook(_cfg(), stage=DynamicsStage.BEFORE_COMPUTE)
        batch = _line_batch(device)
        N, B = batch.num_nodes, batch.num_graphs
        hook(_ctx(batch), _STAGE)

        assert hook._buf_positions.shape == (N, 3)
        assert hook._buf_batch_ptr.shape == (B + 1,)
        assert hook._buf_batch_idx.shape == (N,)

    def test_pbc_buffers_allocated_when_pbc_present(self, device: str):
        hook = NeighborListHook(_cfg(), stage=DynamicsStage.BEFORE_COMPUTE)
        batch = _line_batch(device, pbc=True)
        hook(_ctx(batch), _STAGE)

        assert hook._buf_cell is not None
        assert hook._buf_pbc is not None
        B = batch.num_graphs
        assert hook._buf_cell.shape == (B, 3, 3)
        assert hook._buf_pbc.shape == (B, 3)

    def test_pbc_buffers_none_without_pbc(self, device: str):
        hook = NeighborListHook(_cfg(), stage=DynamicsStage.BEFORE_COMPUTE)
        batch = _line_batch(device, pbc=False)
        hook(_ctx(batch), _STAGE)

        assert hook._buf_cell is None
        assert hook._buf_pbc is None

    def test_rebuild_flags_all_true_after_alloc(self, device: str):
        """Initial rebuild_flags must be all-True to force first full build."""
        hook = NeighborListHook(_cfg(), stage=DynamicsStage.BEFORE_COMPUTE)
        # Access the flags AFTER staging allocation (first call triggers it)
        batch = _line_batch(device)
        # Peek at the flags before the NL call by reaching into _alloc_staging_buffers
        # indirectly: after the hook runs _alloc_staging_buffers, flags are set.
        # We can't inspect them mid-call, so verify the alloc behaviour directly:
        N, B = batch.num_nodes, batch.num_graphs
        hook._alloc_staging_buffers(
            N, B, batch.positions.dtype, batch.device, None, None
        )

        assert hook._rebuild_flags is not None
        assert hook._rebuild_flags.all(), "rebuild_flags must be all-True after alloc"

    def test_alloc_N_B_set_after_first_call(self, device: str):
        hook = NeighborListHook(_cfg(), stage=DynamicsStage.BEFORE_COMPUTE)
        batch = _line_batch(device)
        hook(_ctx(batch), _STAGE)

        assert hook._alloc_N == batch.num_nodes
        assert hook._alloc_B == batch.num_graphs

    def test_output_tensors_allocated_after_first_call(self, device: str):
        hook = NeighborListHook(_cfg(), stage=DynamicsStage.BEFORE_COMPUTE)
        batch = _line_batch(device)
        N = batch.num_nodes
        hook(_ctx(batch), _STAGE)

        assert hook._neighbor_matrix is not None
        assert hook._num_neighbors is not None
        assert hook._neighbor_matrix.shape == (N, hook._max_neighbors)
        assert hook._num_neighbors.shape == (N,)
        assert hook._neighbor_matrix.dtype == torch.int32
        assert hook._num_neighbors.dtype == torch.int32

    def test_neighbor_matrix_shifts_allocated_with_pbc(self, device: str):
        hook = NeighborListHook(_cfg(), stage=DynamicsStage.BEFORE_COMPUTE)
        batch = _line_batch(device, pbc=True)
        N = batch.num_nodes
        hook(_ctx(batch), _STAGE)

        assert hook._neighbor_matrix_shifts is not None
        assert hook._neighbor_matrix_shifts.shape == (N, hook._max_neighbors, 3)

    def test_neighbor_matrix_shifts_none_without_pbc(self, device: str):
        hook = NeighborListHook(_cfg(), stage=DynamicsStage.BEFORE_COMPUTE)
        batch = _line_batch(device, pbc=False)
        hook(_ctx(batch), _STAGE)

        assert hook._neighbor_matrix_shifts is None


# ===========================================================================
# TestCopyToStagingBuffers
# ===========================================================================


class TestCopyToStagingBuffers:
    """Verify staging buffers reflect the current batch after each call."""

    def test_positions_copied(self, device: str):
        hook = NeighborListHook(_cfg(), stage=DynamicsStage.BEFORE_COMPUTE)
        batch = _line_batch(device)
        hook(_ctx(batch), _STAGE)  # allocates and copies

        assert torch.allclose(
            hook._buf_positions, batch.positions.to(hook._buf_positions.dtype)
        )

    def test_batch_ptr_copied(self, device: str):
        hook = NeighborListHook(_cfg(), stage=DynamicsStage.BEFORE_COMPUTE)
        batch = _line_batch(device)
        hook(_ctx(batch), _STAGE)

        expected_ptr = batch.batch_ptr.to(torch.int32)
        assert torch.equal(hook._buf_batch_ptr, expected_ptr)

    def test_cell_copied_when_pbc(self, device: str):
        hook = NeighborListHook(_cfg(), stage=DynamicsStage.BEFORE_COMPUTE)
        batch = _line_batch(device, pbc=True)
        hook(_ctx(batch), _STAGE)

        expected_cell = batch.cell.to(hook._buf_cell.dtype).contiguous()
        assert torch.allclose(hook._buf_cell, expected_cell)

    def test_updated_positions_reflected_on_next_call(self, device: str):
        hook = NeighborListHook(_cfg(), stage=DynamicsStage.BEFORE_COMPUTE)
        batch = _line_batch(device)
        hook(_ctx(batch), _STAGE)

        # Move all atoms (in-place, matching real integrator behaviour)
        batch.positions.add_(10.0)
        hook(_ctx(batch), _STAGE)

        assert torch.allclose(
            hook._buf_positions, batch.positions.to(hook._buf_positions.dtype)
        )

    def test_gradient_enabled_positions_do_not_attach_staging_buffer(self, device: str):
        hook = NeighborListHook(_cfg(), stage=DynamicsStage.BEFORE_COMPUTE)
        batch = _line_batch(device)
        batch.positions.requires_grad_(True)

        hook(_ctx(batch), _STAGE)

        assert not hook._buf_positions.requires_grad
        assert hook._buf_positions.grad_fn is None


# ===========================================================================
# TestAllocNlKwargs
# ===========================================================================


class TestAllocNlKwargs:
    """Verify algorithm-specific kwargs are pre-computed correctly."""

    def test_naive_no_pbc_empty_kwargs(self, device: str):
        """No-PBC naive path requires no extra kwargs."""
        hook = NeighborListHook(_cfg(), stage=DynamicsStage.BEFORE_COMPUTE)
        batch = _line_batch(device, pbc=False)
        hook(_ctx(batch), _STAGE)

        assert hook._buf_nl_kwargs == {}

    def test_naive_pbc_has_shift_kwargs(self, device: str):
        """PBC naive path must pre-compute shift-range tensors."""
        hook = NeighborListHook(_cfg(), stage=DynamicsStage.BEFORE_COMPUTE)
        batch = _line_batch(device, pbc=True)
        hook(_ctx(batch), _STAGE)

        expected_keys = {
            "shift_range_per_dimension",
            "num_shifts_per_system",
            "max_shifts_per_system",
            "max_atoms_per_system",
        }
        assert expected_keys.issubset(hook._buf_nl_kwargs.keys()), (
            f"Missing keys: {expected_keys - set(hook._buf_nl_kwargs.keys())}"
        )

    def test_cell_list_has_scratch_tensors(self, device: str):
        """Cell-list path (avg_atoms >= 2000) must pre-allocate seven tensors."""
        N_per = 2000
        # Build a large batch to trigger cell-list selection
        positions = torch.rand(N_per, 3) * 10.0
        data = AtomicData(
            positions=positions,
            atomic_numbers=torch.ones(N_per, dtype=torch.long),
        )
        batch = Batch.from_data_list([data]).to(device)

        hook = NeighborListHook(
            _cfg(), max_neighbors=64, stage=DynamicsStage.BEFORE_COMPUTE
        )
        hook(_ctx(batch), _STAGE)

        expected_keys = {
            "cells_per_dimension",
            "neighbor_search_radius",
            "atom_periodic_shifts",
            "atom_to_cell_mapping",
            "atoms_per_cell_count",
            "cell_atom_start_indices",
            "cell_atom_list",
        }
        assert expected_keys.issubset(hook._buf_nl_kwargs.keys()), (
            f"Missing keys: {expected_keys - set(hook._buf_nl_kwargs.keys())}"
        )

    @pytest.mark.parametrize("n_graphs", [1, 2])
    def test_cell_list_cell_cap_matches_selector(self, device: str, n_graphs: int):
        """Cell lists use the selector's per-system cell cap.

        The cost model assumes 524288 cells for a single system and 8192 per
        system for a batch. A 40 A box with a 1 A cutoff needs about 40^3 =
        64000 cells, so a single system must not be capped at 8192.
        """
        from nvalchemiops.neighbors.base_dispatch import (
            DEFAULT_BATCH_MAX_NBINS,
            DEFAULT_SINGLE_MAX_NBINS,
        )

        gen = torch.Generator().manual_seed(0)
        box, n_atoms = 40.0, 2000
        data_list = [
            AtomicData(
                positions=torch.rand(n_atoms, 3, generator=gen) * box,
                atomic_numbers=torch.ones(n_atoms, dtype=torch.long),
                cell=torch.eye(3).unsqueeze(0) * box,
                pbc=torch.tensor([[True, True, True]]),
            )
            for _ in range(n_graphs)
        ]
        batch = Batch.from_data_list(data_list).to(device)

        counts = {}
        for method in ("cell_list", "naive"):
            hook = NeighborListHook(
                _cfg(cutoff=1.0), method=method, stage=DynamicsStage.BEFORE_COMPUTE
            )
            hook(_ctx(batch), _STAGE)
            counts[method] = batch.num_neighbors.clone()
            if method == "cell_list":
                n_cells = hook._buf_nl_kwargs["cell_atom_start_indices"].numel()

        cap = DEFAULT_SINGLE_MAX_NBINS if n_graphs == 1 else DEFAULT_BATCH_MAX_NBINS
        assert n_cells <= n_graphs * cap
        if n_graphs == 1:
            assert n_cells > DEFAULT_BATCH_MAX_NBINS
        torch.testing.assert_close(counts["cell_list"], counts["naive"])

    def test_kwargs_are_tensors(self, device: str):
        """Tensor-valued pre-allocated kwargs must be torch.Tensor objects.

        Some kwargs (e.g. ``max_shifts_per_system``, ``max_atoms_per_system``)
        are plain Python ``int`` scalars as required by the nvalchemiops API.
        """
        hook = NeighborListHook(_cfg(), stage=DynamicsStage.BEFORE_COMPUTE)
        batch = _line_batch(device, pbc=True)
        hook(_ctx(batch), _STAGE)

        for key, val in hook._buf_nl_kwargs.items():
            if isinstance(val, int):
                continue  # int scalars are valid (e.g. max_shifts_per_system)
            assert isinstance(val, torch.Tensor), f"{key} must be a Tensor"

    def test_kwargs_on_correct_device(self, device: str):
        """Tensor-valued pre-allocated kwargs must live on the same device as the batch."""
        hook = NeighborListHook(_cfg(), stage=DynamicsStage.BEFORE_COMPUTE)
        batch = _line_batch(device, pbc=True)
        hook(_ctx(batch), _STAGE)

        for key, val in hook._buf_nl_kwargs.items():
            if not isinstance(val, torch.Tensor):
                continue  # int scalars have no .device attribute
            assert str(val.device).startswith(device.split(":")[0]), (
                f"{key} is on {val.device}, expected {device}"
            )


# ===========================================================================
# TestShapeInvalidation
# ===========================================================================


class TestShapeInvalidation:
    """Staging buffers must be reallocated when N or B changes."""

    def test_realloc_on_N_change(self, device: str):
        hook = NeighborListHook(_cfg(), stage=DynamicsStage.BEFORE_COMPUTE)
        batch_small = _line_batch(device)  # 3 atoms
        hook(_ctx(batch_small), _STAGE)

        old_N = hook._alloc_N
        assert old_N == 3

        # Build a batch with more atoms
        data = AtomicData(
            positions=torch.rand(5, 3),
            atomic_numbers=torch.ones(5, dtype=torch.long),
        )
        batch_large = Batch.from_data_list([data]).to(device)
        hook(_ctx(batch_large), _STAGE)

        assert hook._alloc_N == 5
        assert hook._buf_positions.shape == (5, 3)

    def test_realloc_on_B_change(self, device: str):
        hook = NeighborListHook(_cfg(), stage=DynamicsStage.BEFORE_COMPUTE)
        batch_1g = _line_batch(device, n_graphs=1)
        hook(_ctx(batch_1g), _STAGE)
        assert hook._alloc_B == 1

        batch_2g = _line_batch(device, n_graphs=2)
        hook(_ctx(batch_2g), _STAGE)
        assert hook._alloc_B == 2
        assert hook._buf_batch_ptr.shape == (3,)  # B+1 = 3

    def test_ref_positions_reset_on_N_change(self, device: str):
        """_ref_positions must be reset when atom count changes."""
        hook = NeighborListHook(_cfg(), skin=0.5, stage=DynamicsStage.BEFORE_COMPUTE)
        batch = _line_batch(device)
        hook(_ctx(batch), _STAGE)
        assert hook._ref_positions is not None

        # New batch with different N forces reallocation
        data = AtomicData(
            positions=torch.rand(5, 3),
            atomic_numbers=torch.ones(5, dtype=torch.long),
        )
        batch_new = Batch.from_data_list([data]).to(device)
        hook(_ctx(batch_new), _STAGE)

        # After reallocation, _ref_positions is reset then re-initialised
        assert hook._ref_positions is not None
        assert hook._ref_positions.shape == (5, 3)

    def test_no_realloc_when_shape_unchanged(self, device: str):
        """Buffer objects must be the same Python objects on repeated calls."""
        hook = NeighborListHook(_cfg(), stage=DynamicsStage.BEFORE_COMPUTE)
        batch = _line_batch(device)
        hook(_ctx(batch), _STAGE)

        buf_pos_id = id(hook._buf_positions)
        hook(_ctx(batch), _STAGE)

        assert id(hook._buf_positions) == buf_pos_id, (
            "_buf_positions was reallocated despite same shape"
        )

    def test_graph_count_change_rederives_k(self, device: str):
        """A new graph count re-derives K although the atom count is unchanged."""
        hook = NeighborListHook(_cfg(), stage=DynamicsStage.BEFORE_COMPUTE)
        hook(_ctx(_cluster_batch([12, 12], device)), _STAGE)

        batch = _cluster_batch([2, 2, 20], device)
        hook(_ctx(batch), _STAGE)

        fresh_hook, fresh = _fresh_build(batch, _cfg())
        assert hook._max_neighbors == fresh_hook._max_neighbors
        assert _neighbor_sets(batch) == _neighbor_sets(fresh)

    def test_graph_count_change_rebuilds_unmoved_atoms(self, device: str):
        """With a skin, a new graph count rebuilds lists for atoms that did not move."""
        hook = NeighborListHook(_cfg(), skin=0.5, stage=DynamicsStage.BEFORE_COMPUTE)
        before = _cluster_batch([6, 6], device)
        hook(_ctx(before), _STAGE)

        # The same atoms, regrouped into three graphs.
        batch = _cluster_batch([3, 3, 6], device, positions=before.positions)
        hook(_ctx(batch), _STAGE)

        _, fresh = _fresh_build(batch, _cfg())
        assert _neighbor_sets(batch) == _neighbor_sets(fresh)


# ===========================================================================
# TestMembershipChange
# ===========================================================================


class TestMembershipChange:
    """New systems in a batch of unchanged shape get their own neighbor list.

    Dynamics engines dispatch ``ON_ADMISSION`` after every change of batch
    membership, for example an inflight refill that swaps two 12-atom
    systems for a 4-atom and a 20-atom one.
    """

    @staticmethod
    def _swap(hook: NeighborListHook, device: str, *, pbc: bool = False) -> Batch:
        """Build for [12, 12], admit [4, 20] (same N and B), build again."""
        hook(_ctx(_cluster_batch([12, 12], device, pbc=pbc)), _STAGE)
        batch = _cluster_batch([4, 20], device, pbc=pbc)
        hook(_ctx(batch), DynamicsStage.ON_ADMISSION)
        hook(_ctx(batch), _STAGE)
        return batch

    def test_runs_at_its_stage_and_on_admission(self):
        hook = NeighborListHook(_cfg(), stage=DynamicsStage.BEFORE_COMPUTE)

        runs_on = {stage for stage in DynamicsStage if hook._runs_on_stage(stage)}

        assert runs_on == {DynamicsStage.BEFORE_COMPUTE, DynamicsStage.ON_ADMISSION}
        assert not any(hook._runs_on_stage(stage) for stage in TrainingStage)

    def test_admission_builds_nothing(self, device: str):
        hook = NeighborListHook(_cfg(), stage=DynamicsStage.BEFORE_COMPUTE)
        batch = _cluster_batch([4, 20], device)

        hook(_ctx(batch), DynamicsStage.ON_ADMISSION)

        assert "neighbor_matrix" not in batch

    def test_hook_registered_at_admission_still_builds(self, device: str):
        hook = NeighborListHook(_cfg(), stage=DynamicsStage.ON_ADMISSION)
        batch = _cluster_batch([4, 20], device)

        hook(_ctx(batch), DynamicsStage.ON_ADMISSION)

        _, fresh = _fresh_build(batch, _cfg())
        assert _neighbor_sets(batch) == _neighbor_sets(fresh)

    def test_skin_list_after_admission_matches_fresh_hook(self, device: str):
        config = _cfg(skin=0.5)
        hook = NeighborListHook(config, stage=DynamicsStage.BEFORE_COMPUTE)
        before = _cluster_batch([12, 12], device)
        hook(_ctx(before), _STAGE)
        # Same atoms at the same places in new systems: no atom moved, so
        # only the admission reset can trigger the rebuild.
        batch = _cluster_batch([4, 20], device, positions=before.positions.clone())
        hook(_ctx(batch), DynamicsStage.ON_ADMISSION)
        hook(_ctx(batch), _STAGE)

        fresh_hook, fresh = _fresh_build(batch, config)
        assert hook._max_neighbors == fresh_hook._max_neighbors
        assert _neighbor_sets(batch) == _neighbor_sets(fresh)

    def test_matrix_after_admission_matches_fresh_hook(self, device: str):
        hook = NeighborListHook(_cfg(), stage=DynamicsStage.BEFORE_COMPUTE)
        batch = self._swap(hook, device)

        fresh_hook, fresh = _fresh_build(batch, _cfg())
        assert hook._max_neighbors == fresh_hook._max_neighbors
        assert _neighbor_sets(batch) == _neighbor_sets(fresh)

    def test_coo_after_admission_matches_fresh_hook(self, device: str):
        config = _cfg(fmt=NeighborListFormat.COO)
        hook = NeighborListHook(config, stage=DynamicsStage.BEFORE_COMPUTE)
        batch = self._swap(hook, device)

        _, fresh = _fresh_build(batch, config)
        assert _edge_set(batch) == _edge_set(fresh)

    def test_naive_pbc_after_admission_matches_fresh_hook(self, device: str):
        hook = NeighborListHook(
            _cfg(), method="batch_naive_scalar", stage=DynamicsStage.BEFORE_COMPUTE
        )
        batch = self._swap(hook, device, pbc=True)

        _, fresh = _fresh_build(batch, _cfg())
        assert hook._buf_nl_kwargs["max_atoms_per_system"] == 20
        assert _neighbor_sets(batch) == _neighbor_sets(fresh)

    def test_registration_without_stage_raises(self):
        dynamics = DemoDynamics(model=DemoModelWrapper(DemoModel()), n_steps=1, dt=0.5)

        with pytest.raises(TypeError, match="has no stage assigned"):
            dynamics.register_hook(NeighborListHook(_cfg()))

    def test_registration_at_training_stage_on_dynamics_raises(self):
        dynamics = DemoDynamics(model=DemoModelWrapper(DemoModel()), n_steps=1, dt=0.5)
        hook = NeighborListHook(_cfg(), stage=TrainingStage.BEFORE_FORWARD)

        with pytest.raises(TypeError, match="only accepts DynamicsStage stages"):
            dynamics.register_hook(hook)

    def test_training_registration_at_before_forward(self):
        strategy = TrainingStrategy(
            models=DemoModelWrapper(DemoModel()),
            optimizer_configs=OptimizerConfig(optimizer_cls=torch.optim.Adam),
            num_epochs=1,
            training_fn=default_training_fn,
            loss_fn=EnergyMSELoss(),
        )

        strategy.register_hook(
            NeighborListHook(_cfg()), stage=TrainingStage.BEFORE_FORWARD
        )

        fired = [
            stage for stage in TrainingStage if strategy._has_hooks_for_stage(stage)
        ]
        assert fired == [TrainingStage.BEFORE_FORWARD]

    def test_fused_stage_refills_match_fresh_hook(self):
        """Every build after an inflight refill matches a fresh hook's."""
        model = LennardJonesModelWrapper(epsilon=0.01, sigma=1.0, cutoff=5.0)
        config = model.model_config.neighbor_config
        check = _FreshBuildCheck(config)
        optimizer = FIRE2(
            model=model,
            dt=0.05,
            n_steps=2,
            hooks=[NeighborListHook(config, stage=_STAGE), check],
            device_type="cpu",
        )
        # One size bin, so each refill admits the next two clusters in
        # dataset order: [12, 12] -> [20, 4] -> [12, 12], always N=24, B=2.
        sampler = SizeAwareSampler(
            _ClusterDataset([12, 12, 20, 4, 12, 12]),
            max_atoms=24,
            max_batch_size=2,
            bin_width=100,
        )
        fused = FusedStage(
            sub_stages=[(0, optimizer)], sampler=sampler, device_type="cpu"
        )

        fused.run()

        assert [20, 4] in [sizes for sizes, _ in check.checks]
        assert all(matches for _, matches in check.checks)


# ===========================================================================
# TestNeighborListHookMatrix
# ===========================================================================


class TestNeighborListHookMatrix:
    """Integration tests for MATRIX output format."""

    def test_neighbor_matrix_written_to_batch(self, device: str):
        hook = NeighborListHook(
            _cfg(fmt=NeighborListFormat.MATRIX), stage=DynamicsStage.BEFORE_COMPUTE
        )
        batch = _line_batch(device)
        hook(_ctx(batch), _STAGE)

        assert hasattr(batch, "neighbor_matrix")
        assert hasattr(batch, "num_neighbors")

    def test_cutoff_stamped_on_batch(self, device: str):
        hook = NeighborListHook(_cfg(), stage=DynamicsStage.BEFORE_COMPUTE)
        batch = _line_batch(device)
        hook(_ctx(batch), _STAGE)

        assert batch._neighbor_list_cutoff == pytest.approx(_CUTOFF)

    def test_nearby_atoms_are_neighbors(self, device: str):
        """Atoms 0 and 1 (dist=1.5) must appear in each other's neighbor list."""
        hook = NeighborListHook(_cfg(), stage=DynamicsStage.BEFORE_COMPUTE)
        batch = _line_batch(device)
        hook(_ctx(batch), _STAGE)

        nm = batch.neighbor_matrix.cpu()
        nn = batch.num_neighbors.cpu()

        assert _is_neighbor(nm, nn, 0, 1), "atom 0 should list atom 1 as neighbor"
        assert _is_neighbor(nm, nn, 1, 0), "atom 1 should list atom 0 as neighbor"

    def test_far_atom_has_no_neighbors(self, device: str):
        """Atom 2 (dist>3.5 from both others) should have zero neighbors."""
        hook = NeighborListHook(_cfg(), stage=DynamicsStage.BEFORE_COMPUTE)
        batch = _line_batch(device)
        hook(_ctx(batch), _STAGE)

        nn = batch.num_neighbors.cpu()
        assert int(nn[2].item()) == 0, "isolated atom should have no neighbors"

    def test_no_self_neighbors(self, device: str):
        hook = NeighborListHook(_cfg(), stage=DynamicsStage.BEFORE_COMPUTE)
        batch = _line_batch(device)
        hook(_ctx(batch), _STAGE)

        nm = batch.neighbor_matrix.cpu()
        nn = batch.num_neighbors.cpu()
        N = batch.num_nodes

        for i in range(N):
            n = int(nn[i].item())
            assert i not in nm[i, :n].tolist(), f"atom {i} lists itself as neighbor"

    def test_multi_graph_isolation(self, device: str):
        """Atoms in different graphs must not appear in each other's neighbor lists."""
        hook = NeighborListHook(_cfg(), stage=DynamicsStage.BEFORE_COMPUTE)
        batch = _line_batch(
            device, n_graphs=2
        )  # 6 atoms: graph 0 = [0,1,2], graph 1 = [3,4,5]
        hook(_ctx(batch), _STAGE)

        nm = batch.neighbor_matrix.cpu()
        nn = batch.num_neighbors.cpu()

        # Atoms 0-2 are graph 0; atoms 3-5 are graph 1.
        # No cross-graph neighbors allowed.
        for i in range(3):
            n = int(nn[i].item())
            for nb in nm[i, :n].tolist():
                assert nb < 3, f"atom {i} (graph 0) has neighbor {nb} from graph 1"

    def test_idempotent_second_call(self, device: str):
        """Calling the hook twice should give the same neighbor counts."""
        hook = NeighborListHook(_cfg(), stage=DynamicsStage.BEFORE_COMPUTE)
        batch = _line_batch(device)
        hook(_ctx(batch), _STAGE)
        nn_first = batch.num_neighbors.clone()

        hook(_ctx(batch), _STAGE)
        nn_second = batch.num_neighbors.clone()

        assert torch.equal(nn_first, nn_second)

    def test_pbc_neighbor_found_across_boundary(self, device: str):
        """Atoms close only through PBC image must be listed as neighbors."""
        hook = NeighborListHook(_cfg(cutoff=2.5), stage=DynamicsStage.BEFORE_COMPUTE)
        batch = _pbc_wrap_batch(device)
        hook(_ctx(batch), _STAGE)

        nm = batch.neighbor_matrix.cpu()
        nn = batch.num_neighbors.cpu()

        assert _is_neighbor(nm, nn, 0, 1), "atoms should be neighbors via PBC"
        assert _is_neighbor(nm, nn, 1, 0)

    def test_pbc_neighbor_not_found_without_pbc(self, device: str):
        """Same geometry without PBC: atoms 2.8 Å apart should NOT be neighbors."""
        pos = torch.tensor([[0.1, 0.0, 0.0], [2.9, 0.0, 0.0]])
        data = AtomicData(
            positions=pos,
            atomic_numbers=torch.tensor([1, 1], dtype=torch.long),
        )
        batch = Batch.from_data_list([data]).to(device)

        hook = NeighborListHook(_cfg(cutoff=2.5), stage=DynamicsStage.BEFORE_COMPUTE)
        hook(_ctx(batch), _STAGE)

        nn = batch.num_neighbors.cpu()
        assert int(nn[0].item()) == 0, "without PBC, far atoms should not be neighbors"

    @pytest.mark.parametrize("int_dtype", [torch.int32, torch.int64])
    def test_matrix_with_int_dtypes(self, device: str, int_dtype: torch.dtype):
        """Neighbor list MATRIX format works with both int32 and int64 indices."""
        hook = NeighborListHook(_cfg(), stage=DynamicsStage.BEFORE_COMPUTE)
        batch = _line_batch(device, int_dtype=int_dtype)
        hook(_ctx(batch), _STAGE)

        nm = batch.neighbor_matrix.cpu()
        nn = batch.num_neighbors.cpu()
        assert _is_neighbor(nm, nn, 0, 1)
        assert _is_neighbor(nm, nn, 1, 0)
        assert int(nn[2].item()) == 0


# ===========================================================================
# TestNeighborListHookCOO
# ===========================================================================


class TestNeighborListHookCOO:
    """Integration tests for COO output format."""

    def test_edge_index_written_to_batch(self, device: str):
        hook = NeighborListHook(
            _cfg(fmt=NeighborListFormat.COO),
            stage=DynamicsStage.BEFORE_COMPUTE,
        )
        batch = _line_batch(device)
        hook(_ctx(batch), _STAGE)

        assert hasattr(batch, "neighbor_list")

    def test_edge_index_shape(self, device: str):
        hook = NeighborListHook(
            _cfg(fmt=NeighborListFormat.COO),
            stage=DynamicsStage.BEFORE_COMPUTE,
        )
        batch = _line_batch(device)
        hook(_ctx(batch), _STAGE)

        ei = batch.neighbor_list
        assert ei.ndim == 2
        assert ei.shape[1] == 2, (
            "neighbor_list should be (E, 2) in nvalchemi convention"
        )

    def test_nearby_atoms_have_edges(self, device: str):
        """Atoms 0 and 1 (dist=1.5 < cutoff) must appear as an edge pair."""
        hook = NeighborListHook(
            _cfg(fmt=NeighborListFormat.COO),
            stage=DynamicsStage.BEFORE_COMPUTE,
        )
        batch = _line_batch(device)
        hook(_ctx(batch), _STAGE)

        ei = batch.neighbor_list.cpu().tolist()
        pairs = {tuple(row) for row in ei}
        assert (0, 1) in pairs or (1, 0) in pairs, "edge between atoms 0 and 1 expected"

    def test_neighbor_list_shifts_present_with_pbc(self, device: str):
        hook = NeighborListHook(
            _cfg(fmt=NeighborListFormat.COO),
            stage=DynamicsStage.BEFORE_COMPUTE,
        )
        batch = _pbc_wrap_batch(device)
        hook(_ctx(batch), _STAGE)

        assert hasattr(batch, "neighbor_list_shifts")
        assert batch.neighbor_list_shifts.shape[1] == 3

    def test_no_edges_for_isolated_atom(self, device: str):
        """Atom 2 (isolated, dist > cutoff to all others) should appear in no edges."""
        hook = NeighborListHook(
            _cfg(fmt=NeighborListFormat.COO),
            stage=DynamicsStage.BEFORE_COMPUTE,
        )
        batch = _line_batch(device)
        hook(_ctx(batch), _STAGE)

        ei = batch.neighbor_list.cpu().tolist()
        atom_indices = {idx for row in ei for idx in row}
        assert 2 not in atom_indices, "isolated atom 2 should have no edges"

    @pytest.mark.parametrize("int_dtype", [torch.int32, torch.int64])
    def test_coo_with_int_dtypes(self, device: str, int_dtype: torch.dtype):
        """Neighbor list COO format works with both int32 and int64 indices."""
        hook = NeighborListHook(
            _cfg(fmt=NeighborListFormat.COO),
            stage=DynamicsStage.BEFORE_COMPUTE,
        )
        batch = _line_batch(device, int_dtype=int_dtype)
        hook(_ctx(batch), _STAGE)

        ei = batch.neighbor_list.cpu().tolist()
        pairs = {tuple(row) for row in ei}
        assert (0, 1) in pairs or (1, 0) in pairs


# ===========================================================================
# TestSkinCheck
# ===========================================================================


class TestSkinCheck:
    """Verify Verlet skin-check behaviour."""

    def test_ref_positions_none_before_first_call_with_skin(self):
        hook = NeighborListHook(_cfg(), skin=0.5, stage=DynamicsStage.BEFORE_COMPUTE)
        assert hook._ref_positions is None

    def test_ref_positions_set_after_first_call(self, device: str):
        hook = NeighborListHook(_cfg(), skin=0.5, stage=DynamicsStage.BEFORE_COMPUTE)
        batch = _line_batch(device)
        hook(_ctx(batch), _STAGE)

        assert hook._ref_positions is not None
        assert hook._ref_positions.shape == (batch.num_nodes, 3)

    def test_ref_positions_not_set_when_skin_zero(self, device: str):
        hook = NeighborListHook(_cfg(), skin=0.0, stage=DynamicsStage.BEFORE_COMPUTE)
        batch = _line_batch(device)
        hook(_ctx(batch), _STAGE)

        assert hook._ref_positions is None

    def test_neighbor_counts_stable_when_positions_unchanged(self, device: str):
        """Without displacement, the same neighbor counts should come back each step."""
        hook = NeighborListHook(_cfg(), skin=1.0, stage=DynamicsStage.BEFORE_COMPUTE)
        batch = _line_batch(device)
        hook(_ctx(batch), _STAGE)
        nn_first = batch.num_neighbors.clone()

        # Call again without moving atoms
        hook(_ctx(batch), _STAGE)
        nn_second = batch.num_neighbors.clone()

        assert torch.equal(nn_first, nn_second)

    def test_rebuild_flags_true_on_first_call(self, device: str):
        """On the first call, rebuild_flags must be all-True (full rebuild)."""
        hook = NeighborListHook(_cfg(), skin=1.0, stage=DynamicsStage.BEFORE_COMPUTE)
        batch = _line_batch(device)

        # After alloc but before the NL call, flags are all-True.
        # We verify this indirectly: allocate directly and check.
        N, B = batch.num_nodes, batch.num_graphs
        hook._alloc_staging_buffers(
            N,
            B,
            batch.positions.dtype,
            batch.device,
            getattr(batch, "cell", None),
            getattr(batch, "pbc", None),
        )
        assert hook._rebuild_flags.all()

    def test_large_displacement_triggers_rebuild(self, device: str):
        """Moving atoms far enough must change the neighbor list."""
        hook = NeighborListHook(_cfg(), skin=0.5, stage=DynamicsStage.BEFORE_COMPUTE)
        batch = _line_batch(device)
        hook(_ctx(batch), _STAGE)

        # Move atom 2 very close to atom 1 (well within cutoff, in-place)
        batch.positions[2, 0] = 2.0  # now only 0.5 Å from atom 1
        hook(_ctx(batch), _STAGE)

        nm = batch.neighbor_matrix.cpu()
        nn = batch.num_neighbors.cpu()
        assert _is_neighbor(nm, nn, 1, 2), (
            "moved atom 2 should now be a neighbor of atom 1"
        )

    def test_periodic_fold_rebuilds_cached_shifts(self, device: str):
        """Wrapping across a cell boundary must refresh periodic image shifts."""
        hook = NeighborListHook(
            _cfg(),
            skin=0.5,
            stage=DynamicsStage.BEFORE_COMPUTE,
            method="naive",
        )
        data = AtomicData(
            positions=torch.tensor([[0.1, 0.0, 0.0], [19.9, 0.0, 0.0]]),
            atomic_numbers=torch.tensor([1, 1], dtype=torch.long),
            cell=torch.eye(3).unsqueeze(0) * 20.0,
            pbc=torch.tensor([[True, True, True]]),
        )
        batch = Batch.from_data_list([data]).to(device)
        ctx = _ctx(batch)
        hook(ctx, _STAGE)

        # Cross the x boundary by less than skin / 2, then fold back into the cell.
        # The physical MIC displacement is small, but the cached shifts refer to
        # the pre-fold coordinate representation and must be rebuilt.
        batch.positions[1, 0] += 0.15
        WrapPeriodicHook()(ctx, DynamicsStage.AFTER_POST_UPDATE)
        hook(ctx, _STAGE)

        assert hook._rebuild_flags.item()

        n = int(batch.num_neighbors[0].item())
        slot = (batch.neighbor_matrix[0, :n] == 1).nonzero().item()
        shift = batch.neighbor_matrix_shifts[0, slot]
        cell = batch.cell.squeeze(0)
        edge = batch.positions[1] - batch.positions[0] + shift.to(cell.dtype) @ cell
        assert torch.linalg.vector_norm(edge) < hook.config.cutoff

    @pytest.mark.parametrize("method", ["naive", "cell_list", "batch_cluster_tile"])
    def test_partial_rebuild_matches_full_build(self, device: str, method: str):
        """A rebuild of only some systems gives the same list as a full build.

        System 0 is displaced by more than ``skin / 2`` and system 1 is not,
        so the second call rebuilds system 0 only and keeps system 1.
        """
        if method == "batch_cluster_tile" and device == "cpu":
            pytest.skip("batch_cluster_tile is CUDA-only")
        gen = torch.Generator().manual_seed(0)
        box, n_atoms, n_graphs = 12.0, 128, 3
        data_list = [
            AtomicData(
                positions=torch.rand(n_atoms, 3, generator=gen) * box,
                atomic_numbers=torch.ones(n_atoms, dtype=torch.long),
                cell=torch.eye(3).unsqueeze(0) * box,
                pbc=torch.tensor([[True, True, True]]),
            )
            for _ in range(n_graphs)
        ]
        batch = Batch.from_data_list(data_list).to(device)
        hook = NeighborListHook(
            _cfg(), skin=0.5, method=method, stage=DynamicsStage.BEFORE_COMPUTE
        )
        hook(_ctx(batch), _STAGE)

        batch.positions[:n_atoms] += 0.3 * (
            torch.rand(n_atoms, 3, generator=gen).to(device) - 0.5
        )
        batch.positions[0] += 0.4  # > skin / 2: system 0 must be rebuilt
        hook(_ctx(batch), _STAGE)
        assert hook._rebuild_flags.tolist() == [True] + [False] * (n_graphs - 1)
        partial = _sorted_pairs(batch), batch.num_neighbors.clone()

        fresh = NeighborListHook(
            _cfg(), skin=0.5, method=method, stage=DynamicsStage.BEFORE_COMPUTE
        )
        fresh(_ctx(batch), _STAGE)
        full = _sorted_pairs(batch), batch.num_neighbors

        torch.testing.assert_close(partial[1], full[1])
        torch.testing.assert_close(partial[0], full[0])


# ===========================================================================
# TestCompilerDisable
# ===========================================================================


class TestCompilerDisable:
    """Verify the @torch.compiler.disable markers are present on allocation methods.

    torch.compiler.disable sets ``_torchdynamo_disable = True`` on the wrapped
    function.  We check this attribute to confirm the decoration was applied
    rather than relying on runtime compilation behaviour.
    """

    def _check_disabled(self, method) -> None:
        # torch.compiler.disable sets this attribute on the underlying function
        fn = getattr(method, "__func__", method)
        assert getattr(fn, "_torchdynamo_disable", False), (
            f"{method} should be decorated with @torch.compiler.disable"
        )

    def test_alloc_output_tensors_disabled(self):
        hook = NeighborListHook(_cfg(), stage=DynamicsStage.BEFORE_COMPUTE)
        self._check_disabled(hook._alloc_output_tensors)

    def test_alloc_staging_buffers_disabled(self):
        hook = NeighborListHook(_cfg(), stage=DynamicsStage.BEFORE_COMPUTE)
        self._check_disabled(hook._alloc_staging_buffers)

    def test_init_ref_positions_disabled(self):
        hook = NeighborListHook(_cfg(), stage=DynamicsStage.BEFORE_COMPUTE)
        self._check_disabled(hook._init_ref_positions)

    def test_alloc_methods_actually_run(self, device: str):
        """Smoke test: disabled methods must still execute correctly."""
        hook = NeighborListHook(_cfg(), stage=DynamicsStage.BEFORE_COMPUTE)
        batch = _line_batch(device)
        N, B = batch.num_nodes, batch.num_graphs

        hook._alloc_output_tensors(N, batch, None)
        assert hook._neighbor_matrix is not None

        hook._alloc_staging_buffers(
            N, B, batch.positions.dtype, batch.device, None, None
        )
        assert hook._buf_positions is not None


# ===========================================================================
# TestAdaptiveK — adaptive neighbor matrix K-dimension tests
# ===========================================================================


class TestAdaptiveK:
    """Tests for adaptive K-dimension sizing in the neighbor matrix."""

    def test_non_pbc_cap(self, device: str):
        """Non-PBC: K should be capped at max_system_size - 1."""
        # 3 atoms, no PBC. estimate_max_neighbors might return ~16+ for
        # cutoff 2.5, but actual cap should be 2 (3 atoms - 1).
        data = AtomicData(
            positions=torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
            atomic_numbers=torch.tensor([1, 1, 1], dtype=torch.long),
        )
        batch = Batch.from_data_list([data]).to(device)
        hook = NeighborListHook(_cfg(cutoff=100.0), stage=DynamicsStage.BEFORE_COMPUTE)
        hook(_ctx(batch), _STAGE)

        # With cutoff=100 and no PBC, cap is ceil_16(max_num_nodes) = 16.
        assert hook._max_neighbors <= 16

    def test_pbc_no_cap(self, device: str):
        """PBC: K should NOT be capped at max_system_size - 1 (images exist)."""
        data = AtomicData(
            positions=torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
            atomic_numbers=torch.tensor([1, 1], dtype=torch.long),
            cell=torch.eye(3).unsqueeze(0) * 3.0,
            pbc=torch.tensor([[True, True, True]]),
        )
        batch = Batch.from_data_list([data]).to(device)
        hook = NeighborListHook(_cfg(cutoff=100.0), stage=DynamicsStage.BEFORE_COMPUTE)
        hook(_ctx(batch), _STAGE)

        # With PBC, max_neighbors should be the full estimate, not capped.
        assert hook._max_neighbors > 1

    def test_first_build_shrinks_overestimate(self, device: str):
        """First build should trim K when estimate is way too large."""
        # 4 atoms, no PBC, cutoff covers all pairs → 3 neighbors each.
        # Force large initial K via max_neighbors override.
        data = AtomicData(
            positions=torch.tensor(
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]]
            ),
            atomic_numbers=torch.tensor([1, 1, 1, 1], dtype=torch.long),
        )
        batch = Batch.from_data_list([data]).to(device)

        # Override to 10000 — way more than the 3 actual neighbors.
        # Non-PBC cap will reduce to 3, but let's test with smaller override
        # that's still > 4x actual to trigger shrink.
        hook = NeighborListHook(
            _cfg(cutoff=5.0), max_neighbors=100, stage=DynamicsStage.BEFORE_COMPUTE
        )
        hook(_ctx(batch), _STAGE)

        # After first build, K should have been trimmed (actual max is 3,
        # 100/3 > 4x, so shrink to 3*2=6).
        assert hook._max_neighbors < 100
        assert hook._neighbor_matrix.shape[1] == hook._max_neighbors

        # Results must still be correct.
        nn = hook._num_neighbors
        assert int(nn.max()) <= 3

    def test_compute_neighbors_non_pbc_cap(self, device: str):
        """compute_neighbors should cap K for non-PBC systems."""
        from nvalchemi.neighbors import compute_neighbors

        data = AtomicData(
            positions=torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
            atomic_numbers=torch.tensor([1, 1, 1], dtype=torch.long),
            forces=torch.zeros(3, 3),
            energy=torch.zeros(1, 1),
        )
        batch = Batch.from_data_list([data]).to(device)
        compute_neighbors(batch, cutoff=100.0)

        # Non-PBC cap: K = ceil_16(max_num_nodes) = 16.
        nm = batch.neighbor_matrix
        assert nm.shape[1] <= 16


# ===========================================================================
# TestStaleCOOEntries — regression test for stale neighbor matrix entries
# ===========================================================================


class TestStaleCOOEntries:
    """Regression tests: shrinking neighbor counts must not leave stale COO edges.

    When a Verlet skin rebuild finds fewer neighbors for an atom than the
    previous build, slots beyond ``num_neighbors[i]`` in the neighbor matrix
    retain stale indices.  The fix in masks these
    before the COO conversion.
    """

    @staticmethod
    def _edges_as_set(batch: Batch) -> set[tuple[int, int]]:
        """Return the COO neighbor_list as a set of (src, dst) pairs."""
        return {tuple(row) for row in batch.neighbor_list.cpu().tolist()}

    @staticmethod
    def _edge_counts(batch: Batch) -> torch.Tensor:
        """Per-atom edge count derived from the COO neighbor_list."""
        ei = batch.neighbor_list.cpu()
        src = ei[:, 0]
        return torch.bincount(src, minlength=batch.num_nodes)

    def _run_shrink_scenario(self, device: str) -> None:
        """Build hook+batch, run initial rebuild, move atom out of range, rebuild."""
        self.hook = NeighborListHook(_cfg(fmt=NeighborListFormat.COO), skin=1.0)
        self.batch = _line_batch(device)
        # Step 1: full rebuild — atoms 0 and 1 are neighbors (dist 1.5)
        self.hook(_ctx(self.batch), _STAGE)
        self.pairs_before = self._edges_as_set(self.batch)

        # Move atom 1 far away from atom 0 (dist 4.0 > cutoff+skin=3.5),
        # close to atom 2 (dist 1.0 < cutoff).
        # Displacement of 2.5 A > skin/2 = 0.5 triggers a rebuild.
        self.batch.positions[1, 0] = 4.0
        self.hook(_ctx(self.batch), _STAGE)
        self.pairs_after = self._edges_as_set(self.batch)

    def test_shrinking_neighbors_no_stale_edges(self, device: str):
        """Moving an atom out of cutoff+skin must remove its stale edges."""
        self._run_shrink_scenario(device)
        assert (0, 1) in self.pairs_before or (1, 0) in self.pairs_before
        # Atom 0 should have NO neighbors now (nearest is 4.0 A away)
        assert (0, 1) not in self.pairs_after, "stale edge 0->1 not removed"
        assert (1, 0) not in self.pairs_after, "stale edge 1->0 not removed"
        # Atoms 1 and 2 should be neighbors (dist 1.0)
        assert (1, 2) in self.pairs_after or (2, 1) in self.pairs_after

    def test_shrinking_neighbors_edge_count_matches_num_neighbors(self, device: str):
        """After rebuild, per-atom COO edge counts must equal num_neighbors."""
        self._run_shrink_scenario(device)
        coo_counts = self._edge_counts(self.batch)
        # In COO mode num_neighbors lives on the hook's internal buffer,
        # not on the batch object.
        nn = self.hook._num_neighbors.cpu()
        assert torch.equal(coo_counts, nn.to(coo_counts.dtype)), (
            f"COO edge counts {coo_counts.tolist()} != num_neighbors {nn.tolist()}"
        )
