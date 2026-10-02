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
"""Tests for the L-BFGS optimizers.

Numerics belong to ``nvalchemiops``; these cover the Toolkit integration:
two-level state, inflight batching, FusedStage masking, cell alignment and
``cell_force_scale``.
"""

from __future__ import annotations

import dataclasses
import warnings
from unittest.mock import patch

import pytest
import torch
from nvalchemiops.torch.lbfgs import LBFGSCellState, LBFGSState

from nvalchemi.data import AtomicData, Batch
from nvalchemi.data.level_storage import SegmentedLevelStorage
from nvalchemi.dynamics import ConvergenceHook, DynamicsStage
from nvalchemi.dynamics._ops.cell_align import align_cell
from nvalchemi.dynamics.base import _level_mask
from nvalchemi.dynamics.hooks import AlignCellHook, FreezeAtomsHook
from nvalchemi.dynamics.hooks.cell_align import _aligned_periodic
from nvalchemi.dynamics.optimizers import (
    FIRE2,
    LBFGS,
    FIRE2VariableCell,
    LBFGSVariableCell,
)
from nvalchemi.dynamics.optimizers.lbfgs import (
    _CELL_PER_DOF,
    _CELL_PER_SYSTEM,
    _DOF_LEVEL,
    _PER_DOF,
    _PER_SYSTEM,
    _anti_transpose,
    _ops_state,
)
from nvalchemi.hooks.periodic import WrapPeriodicHook

from .conftest import _make_atomic_data, _make_batch, _make_model, _MockSampler

_SKEW = torch.tensor([[5.0, 1.0, 0.0], [0.0, 5.0, 0.0], [0.3, 0.2, 5.0]])


def _already_aligned_nontrivial_cell() -> torch.Tensor:
    """A genuinely lower-triangular, non-diagonal cell.

    Unlike a trivial ``k*I`` cell (which align_cell reproduces exactly even
    on re-alignment, since every intermediate angle/length is exact), this
    one picks up a few ULP of rounding if run back through align_cell —
    exactly what a needs_align mask bug would expose for an "already
    aligned" system sharing a batch with one that needs realigning.
    """
    cell = _SKEW.clone().unsqueeze(0)
    align_cell(torch.zeros(1, 3, dtype=cell.dtype), cell)
    return cell[0]


def _forces(dynamics, batch):
    out = dynamics.model(batch)
    batch.forces = out["forces"] if isinstance(out, dict) else out


def _relax(dynamics, batch, steps):
    """Take ``steps`` steps; return the final max force."""
    for _ in range(steps):
        _forces(dynamics, batch)
        dynamics.pre_update(batch)
    _forces(dynamics, batch)
    return batch.forces.norm(dim=1).max().item()


def _random_steps(dynamics, batch, steps, seed=0):
    """Step on seeded random forces until every system holds a curvature pair."""
    g = torch.Generator().manual_seed(seed)
    for _ in range(steps):
        batch.forces = -0.5 * batch.positions.detach() + 0.01 * torch.randn(
            batch.positions.shape, generator=g, dtype=batch.positions.dtype
        )
        dynamics.pre_update(batch)


def _cell_data(n_atoms, seed, cell=None, pbc=True):
    data = _make_atomic_data(n_atoms, seed, with_cell=True)
    data.cell = (5.0 * torch.eye(3) if cell is None else cell).unsqueeze(0)
    data.pbc = torch.tensor([[pbc] * 3])
    return data


def _cell_batch(cells, pbc=True, n_atoms=4):
    return Batch.from_data_list(
        [_cell_data(n_atoms, 10 + i, c, pbc) for i, c in enumerate(cells)]
    )


def _aligned(cell):
    """Whether *cell*'s strict upper triangle is (numerically) zero.

    Scaled by dtype epsilon and cell magnitude rather than a fixed atol:
    float32 rounding in the alignment kernel is ~1e-7 relative, so a bare
    ``1e-10`` is unreachable for float32 cells and silently over-tight.
    """
    atol = torch.finfo(cell.dtype).eps * cell.abs().amax().clamp(min=1.0) * 10
    return torch.triu(cell, 1).abs().max().item() <= atol


class _Record:
    """Hook recording ``batch.cell`` at one stage."""

    frequency = 1

    def __init__(self, stage):
        self.stage = stage
        self.cells = []

    def __call__(self, ctx, stage):
        self.cells.append(ctx.batch.cell.detach().clone())


class _CountingHook:
    """Hook that just counts how many times it fired at one stage."""

    frequency = 1

    def __init__(self, stage):
        self.stage = stage
        self.count = 0

    def __call__(self, ctx, stage):
        del ctx, stage
        self.count += 1


# ---------------------------------------------------------------------------
# Relaxation
# ---------------------------------------------------------------------------


class TestLBFGSRelaxation:
    def test_reduces_the_force(self):
        # DemoModel is random; the seed pins its surface.
        torch.manual_seed(0)
        batch = _make_batch(3, n_atoms_each=5, seed=1)
        dynamics = LBFGS(model=_make_model(), maxstep=0.2)
        dynamics._ensure_state_initialized(batch)
        _forces(dynamics, batch)
        before = batch.forces.norm(dim=1).max().item()
        assert _relax(dynamics, batch, 30) < 0.25 * before

    def test_history_accumulates(self):
        batch = _make_batch(2, n_atoms_each=4, seed=3)
        dynamics = LBFGS(model=_make_model(), history_size=4)
        dynamics._ensure_state_initialized(batch)
        _relax(dynamics, batch, 10)
        assert int(dynamics._state.history_count.min()) > 0

    def test_run_with_convergence_hook(self):
        # A reachable threshold, so run() actually takes the early-exit
        # branch (base.py: `if ... _converged.numel() == batch.num_graphs:
        # break`) instead of just exhausting n_steps every time.
        torch.manual_seed(0)
        batch = _make_batch(2, n_atoms_each=5, seed=1)
        dynamics = LBFGS(
            model=_make_model(),
            n_steps=200,
            convergence_hook=ConvergenceHook.from_fmax(0.05),
        )
        _forces(dynamics, batch)
        before = batch.forces.norm(dim=1).max().item()
        out = dynamics.run(batch)
        assert out.forces.norm(dim=1).max().item() < before
        assert out.forces.norm(dim=1).max().item() < 0.05
        assert dynamics.step_count < 200

    def test_freeze_atoms_hook(self):
        from nvalchemi._typing import AtomCategory

        batch = _make_batch(2, n_atoms_each=5, seed=4)
        frozen = torch.zeros(batch.num_nodes, dtype=torch.bool)
        frozen[[0, 7]] = True
        batch["atom_categories"] = torch.where(frozen, AtomCategory.SPECIAL.value, 0)
        before = batch.positions.detach().clone()
        dynamics = LBFGS(model=_make_model(), n_steps=5, hooks=[FreezeAtomsHook()])
        dynamics.run(batch)
        assert torch.equal(batch.positions[frozen], before[frozen])
        assert not torch.allclose(batch.positions[~frozen], before[~frozen])


# ---------------------------------------------------------------------------
# State levels
# ---------------------------------------------------------------------------


class TestLBFGSState:
    def test_field_split_matches_ops(self):
        fields = {f.name for f in dataclasses.fields(LBFGSState)}
        assert set(_PER_DOF) | set(_PER_SYSTEM) == fields
        cell_fields = {f.name for f in dataclasses.fields(LBFGSCellState)}
        derived = {"ext_batch_idx", "ext_atom_ptr"}
        assert set(_CELL_PER_DOF) | set(_CELL_PER_SYSTEM) | derived == cell_fields

    def test_two_levels(self):
        batch = _make_batch(3, n_atoms_each=5, seed=1)
        dynamics = LBFGS(model=_make_model())
        dynamics._ensure_state_initialized(batch)
        groups = dynamics._state._storage.groups
        assert set(groups) == {"system", _DOF_LEVEL}
        assert isinstance(groups[_DOF_LEVEL], SegmentedLevelStorage)
        assert int(groups[_DOF_LEVEL].segment_lengths.sum()) == batch.num_nodes

    def test_views_alias_state(self):
        batch = _make_batch(2, n_atoms_each=4, seed=2)
        dynamics = LBFGS(model=_make_model())
        dynamics._ensure_state_initialized(batch)
        view = _ops_state(dynamics._state)
        for field in dataclasses.fields(view):
            assert getattr(view, field.name) is dynamics._state[field.name]

    def test_variable_cell_packed_topology(self):
        batch = _cell_batch([None] * 3)
        dynamics = LBFGSVariableCell(model=_make_model(needs_stress=True))
        dynamics._ensure_state_initialized(batch)
        level = dynamics._state._storage.groups[_DOF_LEVEL]
        assert level.segment_lengths.tolist() == [6, 6, 6]
        assert "ext_batch_idx" not in {key for key, _ in dynamics._state}

    @pytest.mark.parametrize("name", ["history_size", "cell_force_scale"])
    def test_allocation_fixed_parameters_are_read_only(self, name):
        dynamics = LBFGSVariableCell(model=_make_model(needs_stress=True))
        with pytest.raises(AttributeError, match="fixed when optimizer state"):
            setattr(dynamics, name, 2)

    @pytest.mark.parametrize("cls", [LBFGS, LBFGSVariableCell])
    @pytest.mark.parametrize("history_size", [0, -1, 2.5])
    def test_history_size_rejected_at_construction(self, cls, history_size):
        kwargs = {"needs_stress": True} if cls is LBFGSVariableCell else {}
        with pytest.raises(ValueError, match="history_size must be a positive int"):
            cls(model=_make_model(**kwargs), history_size=history_size)


# ---------------------------------------------------------------------------
# Grouped fixed-cell optimization
# ---------------------------------------------------------------------------


class TestGroupedLBFGS:
    """Groups share scalar state and curvature history over their combined atoms."""

    @staticmethod
    def _batch(device="cpu", dtype=torch.float32):
        batch = Batch.from_data_list(
            [_make_atomic_data(n, i) for i, n in enumerate((2, 3, 1, 4, 2))]
        ).to(device=device, dtype=dtype)
        batch.set_group_layout(torch.tensor([0, 0, 1, 1, 1], device=device))
        return batch

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    @pytest.mark.parametrize(
        "groups", [[0, 0, 1, 1, 1], [0, 1, 2, 3, 4], [0, 0, 0, 0, 0]]
    )
    def test_matches_concatenated_graphs(self, device, dtype, groups):
        batch = self._batch(device, dtype)
        batch.set_group_layout(torch.tensor(groups, device=device))
        graph_idx = batch.batch_idx.clone()
        node_ptr = batch.batch_ptr[batch.group_layout.group_ptr].tolist()
        reference = Batch.from_data_list(
            [
                AtomicData(
                    positions=batch.positions[lo:hi].detach().clone(),
                    atomic_numbers=batch.atomic_numbers[lo:hi].clone(),
                    forces=torch.zeros_like(batch.positions[lo:hi]),
                )
                for lo, hi in zip(node_ptr[:-1], node_ptr[1:])
            ]
        )
        grouped = LBFGS(model=_make_model(), by_group=True, history_size=3)
        ordinary = LBFGS(model=_make_model(), history_size=3)
        grouped._ensure_state_initialized(batch)
        ordinary._ensure_state_initialized(reference)
        stiffness = batch.positions.new_tensor([0.3, 0.7, 1.1])
        for _ in range(8):
            for dynamics, data in ((grouped, batch), (ordinary, reference)):
                data.forces = -(data.positions * stiffness + 0.05 * data.positions**3)
                dynamics.pre_update(data)
            torch.testing.assert_close(batch.positions, reference.positions)
            for key, value in grouped._state:
                torch.testing.assert_close(value, ordinary._state[key])
        assert torch.all(grouped._state.history_count > 0)
        assert torch.equal(batch.batch_idx, graph_idx)

    @pytest.mark.parametrize("compiled", [False, True])
    @pytest.mark.parametrize("active", [False, True])
    def test_masking_preserves_inactive_group(self, device, compiled, active):
        batch = self._batch(device)
        dynamics = LBFGS(model=_make_model(), by_group=True, history_size=3)
        dynamics._ensure_state_initialized(batch)
        for _ in range(4):
            batch.forces = -(0.5 * batch.positions + 0.05 * batch.positions**3)
            dynamics.pre_update(batch)
        dynamics._warm_state_levels()
        positions = batch.positions.detach().clone()
        state = {key: value.clone() for key, value in dynamics._state}
        untouched = _rows(dynamics, 1)
        mask = torch.tensor([active, active, False, False, False], device=device)
        pre_update = (
            _compile(dynamics._masked_pre_update)
            if compiled
            else dynamics._masked_pre_update
        )
        pre_update(batch, mask)
        assert torch.equal(batch.positions[5:], positions[5:])
        for key, value in _rows(dynamics, 1).items():
            assert torch.equal(value, untouched[key]), key
        if active:
            assert int(dynamics._state.iteration[0]) == int(state["iteration"][0]) + 1
            assert not torch.equal(batch.positions[:5], positions[:5])
        else:
            assert torch.equal(batch.positions, positions)
            for key, value in dynamics._state:
                assert torch.equal(value, state[key]), key


# ---------------------------------------------------------------------------
# Inflight batching
# ---------------------------------------------------------------------------


class TestLBFGSInflight:
    @staticmethod
    def _ragged(counts=(4, 5, 3), seed=7):
        batch = Batch.from_data_list(
            [_make_atomic_data(c, seed + i) for i, c in enumerate(counts)]
        )
        dynamics = LBFGS(model=_make_model(), history_size=4)
        dynamics._ensure_state_initialized(batch)
        return dynamics, batch

    @staticmethod
    def _refill(dynamics, batch, keep, new_atoms=6):
        trimmed = batch.index_select(keep)
        trimmed.append(Batch.from_data_list([_make_atomic_data(new_atoms, 99)]))
        dynamics._sync_state_to_batch(keep, 1, trimmed)
        return trimmed

    def test_survivors_keep_history_replacements_start_fresh(self):
        dynamics, batch = self._ragged()
        _relax(dynamics, batch, 8)
        keep = torch.tensor([0, 2])
        iteration = dynamics._state.iteration[keep].clone()
        history = dynamics._state.history_count[keep].clone()
        assert int(history.min()) > 0
        self._refill(dynamics, batch, keep)
        torch.testing.assert_close(dynamics._state.iteration[:2], iteration)
        torch.testing.assert_close(dynamics._state.history_count[:2], history)
        assert int(dynamics._state.iteration[2]) == -1
        assert int(dynamics._state.history_count[2]) == 0

    def test_ragged_topology_is_rebuilt(self):
        dynamics, batch = self._ragged()
        trimmed = self._refill(dynamics, batch, torch.tensor([0, 2]))
        level = dynamics._state._storage.groups[_DOF_LEVEL]
        assert level.segment_lengths.tolist() == [4, 3, 6]
        assert dynamics._state.level_ptr(_DOF_LEVEL).tolist() == [0, 4, 7, 13]
        assert int(level.segment_lengths.sum()) == trimmed.num_nodes

    def test_per_dof_state_follows_its_system(self):
        dynamics, batch = self._ragged()
        dynamics._state.x_base[:] = torch.arange(batch.num_nodes).unsqueeze(1)
        keep = torch.tensor([0, 2])
        dynamics._sync_state_to_batch(keep, 0, batch.index_select(keep))
        assert dynamics._state.x_base[:, 0].tolist() == [0, 1, 2, 3, 9, 10, 11]

    def test_views_follow_refill(self):
        dynamics, batch = self._ragged()
        self._refill(dynamics, batch, torch.tensor([0, 2]))
        view = _ops_state(dynamics._state)
        assert view.x_base is dynamics._state["x_base"]
        assert view.num_dofs == 13

    def test_relaxation_continues_after_refill(self):
        dynamics, batch = self._ragged()
        _relax(dynamics, batch, 6)
        trimmed = self._refill(dynamics, batch, torch.tensor([0, 2]))
        before = _relax(dynamics, trimmed, 1)
        assert _relax(dynamics, trimmed, 15) < before

    def test_refill_check_with_sampler(self):
        replacement = _make_atomic_data(6, seed=77)
        dynamics = LBFGS(model=_make_model(), sampler=_MockSampler([replacement]))
        batch = _make_batch(3, 4)
        batch["status"] = torch.zeros(3, 1, dtype=torch.long)
        for _ in range(5):
            dynamics.step(batch)
        survivors = dynamics._state.iteration[1:].clone()
        batch.status[0] = 1
        result = dynamics.refill_check(batch, exit_status=1)
        assert result.num_graphs == 3
        torch.testing.assert_close(dynamics._state.iteration[:2], survivors)
        assert int(dynamics._state.iteration[2]) == -1
        dynamics.step(result)


# ---------------------------------------------------------------------------
# Variable cell: alignment
# ---------------------------------------------------------------------------


class TestLBFGSVariableCellAlignment:
    @staticmethod
    def _dynamics(**kwargs):
        return LBFGSVariableCell(model=_make_model(needs_stress=True), **kwargs)

    def test_init_never_writes_batch(self):
        batch = _cell_batch([None, _SKEW])
        positions = batch.positions.detach().clone()
        cell = batch.cell.clone()
        dynamics = self._dynamics(hooks=[AlignCellHook()])
        dynamics._init_state(batch)
        assert torch.equal(batch.positions, positions)
        assert torch.equal(batch.cell, cell)

    def test_align_hook_is_sufficient(self):
        batch = _cell_batch([None, _SKEW])
        dynamics = self._dynamics(hooks=[AlignCellHook()], n_steps=3)
        dynamics.run(batch)
        assert _aligned(batch.cell)

    def test_chart_matches_what_the_hook_writes(self):
        batch = _cell_batch([None, _SKEW])
        record = _Record(DynamicsStage.BEFORE_PRE_UPDATE)
        dynamics = self._dynamics(hooks=[AlignCellHook(), record], n_steps=1)
        dynamics.run(batch)
        # dynamics._state.ref_cell is in the backend's anti-transposed
        # convention (see _anti_transpose); the hook wrote the live
        # batch.cell in nvalchemi's row-vector convention.
        torch.testing.assert_close(
            dynamics._state.ref_cell, _anti_transpose(record.cells[0])
        )

    def test_aligned_cells_need_no_hook(self):
        batch = _cell_batch([None, None])
        self._dynamics(n_steps=2).run(batch)

    def test_skew_without_hook_raises(self):
        batch = _cell_batch([None, _SKEW])
        with pytest.raises(ValueError, match=r"system\(s\) \[1\].*AlignCellHook\(\)"):
            self._dynamics()._init_state(batch)

    def test_near_triangular_float32_cell_is_aligned_not_rejected(self):
        # A float32 cell with a small upper-triangle entry: well above
        # ALIGN_ATOL (1e-10) so AlignCellHook must still fix it, but far
        # below float32 rounding noise so a dtype-scaled "close enough"
        # tolerance would wrongly skip it — leaving it to fail the
        # reference-cell admission check below, which uses the same
        # ALIGN_ATOL.  The hook and the check must never disagree.
        near_triangular = 5.0 * torch.eye(3)
        near_triangular[0, 1] = 1e-6
        batch = _cell_batch([near_triangular])
        assert batch.cell.dtype == torch.float32
        dynamics = self._dynamics(hooks=[AlignCellHook()])
        dynamics._init_state(batch)  # must not raise
        assert _aligned(dynamics._state.ref_cell)

    def test_refill_onto_near_triangular_float32_cell(self):
        near_triangular = 5.0 * torch.eye(3)
        near_triangular[0, 1] = 1e-6
        record = _Record(DynamicsStage.BEFORE_PRE_UPDATE)
        dynamics = self._dynamics(
            hooks=[AlignCellHook(), record],
            sampler=_MockSampler([_cell_data(4, 51, near_triangular)]),
        )
        batch = _cell_batch([None, None])
        batch["status"] = torch.zeros(2, 1, dtype=torch.long)
        dynamics.step(batch)
        batch.status[0] = 1
        result = dynamics.refill_check(batch, exit_status=1)  # must not raise
        dynamics.step(result)
        assert _aligned(result.cell)

    def test_skew_without_pbc_raises_despite_hook(self):
        batch = _cell_batch([_SKEW, _SKEW], pbc=False)
        with pytest.raises(ValueError, match="aligns only periodic"):
            self._dynamics(hooks=[AlignCellHook()])._init_state(batch)

    def test_only_non_periodic_system_is_reported(self):
        batch = Batch.from_data_list(
            [_cell_data(4, 1, _SKEW, pbc=True), _cell_data(4, 2, _SKEW, pbc=False)]
        )
        with pytest.raises(ValueError, match=r"system\(s\) \[1\] "):
            self._dynamics(hooks=[AlignCellHook()])._init_state(batch)
        batch.pbc[:] = True
        self._dynamics(hooks=[AlignCellHook()])._init_state(batch)

    def test_hook_frequency_must_be_one(self):
        batch = _cell_batch([None])
        with pytest.raises(ValueError, match="frequency=1"):
            self._dynamics(hooks=[AlignCellHook(frequency=2)])._init_state(batch)
        dynamics = self._dynamics()
        dynamics.register_hook(AlignCellHook(frequency=2))
        with pytest.raises(ValueError, match="frequency=1"):
            dynamics._init_state(batch)

    def test_refill_onto_skew_cell(self):
        record = _Record(DynamicsStage.BEFORE_PRE_UPDATE)
        dynamics = self._dynamics(
            hooks=[AlignCellHook(), record],
            sampler=_MockSampler([_cell_data(4, 50, _SKEW)]),
        )
        batch = _cell_batch([None, None])
        batch["status"] = torch.zeros(2, 1, dtype=torch.long)
        dynamics.step(batch)
        batch.status[0] = 1
        result = dynamics.refill_check(batch, exit_status=1)
        ref_cell = dynamics._state.ref_cell[-1].clone()
        dynamics.step(result)
        # The chart is the aligned pre-step cell; the live cell moves but stays
        # aligned.  ref_cell is anti-transposed (see _anti_transpose), the
        # recorded live batch.cell is row-convention.
        torch.testing.assert_close(ref_cell, _anti_transpose(record.cells[-1][-1]))
        assert _aligned(result.cell)

    def test_stress_sign_matches_fire2(self):
        stress = 0.01 * torch.eye(3)

        def volume_change(dynamics):
            batch = _cell_batch([None])
            batch.forces = torch.zeros_like(batch.positions)
            batch.stress = stress.unsqueeze(0).clone()
            dynamics._init_state(batch)
            before = torch.linalg.det(batch.cell).item()
            dynamics.pre_update(batch)
            return torch.linalg.det(batch.cell).item() - before

        model = _make_model(needs_stress=True)
        fire2 = volume_change(FIRE2VariableCell(model=model, dt=0.05))
        lbfgs = volume_change(LBFGSVariableCell(model=model))
        assert fire2 * lbfgs > 0


# ---------------------------------------------------------------------------
# cell_force_scale (both variable-cell classes)
# ---------------------------------------------------------------------------


def _one_cell_step(cls, **kwargs):
    batch = _cell_batch([None])
    batch.forces = torch.zeros_like(batch.positions)
    batch.stress = 0.01 * torch.eye(3).unsqueeze(0)
    kwargs = ({"dt": 0.05} if cls is FIRE2VariableCell else {}) | kwargs
    dynamics = cls(model=_make_model(needs_stress=True), **kwargs)
    dynamics._init_state(batch)
    before = batch.cell.clone()
    dynamics.pre_update(batch)
    return (batch.cell - before).abs().max().item()


class TestCellForceScale:
    @pytest.mark.parametrize("cls", [FIRE2VariableCell, LBFGSVariableCell])
    def test_larger_scale_moves_cell_less(self, cls):
        assert _one_cell_step(cls, cell_force_scale=10.0) < _one_cell_step(cls)

    @pytest.mark.parametrize("cls", [FIRE2VariableCell, LBFGSVariableCell])
    @pytest.mark.parametrize(
        "scale", [0.0, -1.0, float("nan"), float("inf"), float("-inf")]
    )
    def test_non_positive_or_non_finite_rejected(self, cls, scale):
        kwargs = {"dt": 0.05} if cls is FIRE2VariableCell else {}
        with pytest.raises(ValueError, match="finite and positive"):
            cls(model=_make_model(), cell_force_scale=scale, **kwargs)

    def test_positional_arguments_unchanged(self):
        model = _make_model()
        fire2 = FIRE2VariableCell(model, 0.05, 60, 1.05, 0.75, 0.985, 0.09, 0.08, 0.005, 0.1, 7)  # fmt: skip
        assert fire2.n_steps == 7 and fire2.cell_force_scale == 1.0
        with pytest.raises(TypeError):
            FIRE2VariableCell(model, 0.05, 60, 1.05, 0.75, 0.985, 0.09, 0.08, 0.005, 0.1, 7, None, None, 2.0)  # fmt: skip
        lbfgs = LBFGSVariableCell(model, 4, None, 0.1, 7)
        assert lbfgs.n_steps == 7 and lbfgs.cell_force_scale == 1.0

    def test_fire2_default_matches_ops_default(self):
        from nvalchemiops.torch.fire2 import fire2_step_coord_cell

        batch = _cell_batch([None])
        batch.forces = torch.randn_like(batch.positions)
        batch.stress = 0.01 * torch.eye(3).unsqueeze(0)
        dynamics = FIRE2VariableCell(model=_make_model(), dt=0.05)
        dynamics._init_state(batch)
        pos, cell = batch.positions.detach().clone(), batch.cell.clone()
        state = {k: v.clone() for k, v in dynamics._state}
        dynamics.pre_update(batch)
        from nvalchemi.dynamics._ops.npt_nph import stress_to_cell_force

        cell_force = stress_to_cell_force(
            batch.stress, cell, torch.linalg.det(cell).abs()
        )
        fire2_step_coord_cell(
            pos,
            torch.zeros_like(pos),
            batch.forces,
            cell,
            state["cell_velocities"],
            cell_force,
            batch.batch_idx.int(),
            state["alpha"],
            state["dt"],
            state["nsteps_inc"],
        )
        assert torch.equal(batch.positions, pos)
        assert torch.equal(batch.cell, cell)

    def test_fire2_forwards_and_reads_every_step(self):
        dynamics = FIRE2VariableCell(model=_make_model(), dt=0.05)
        batch = _cell_batch([None])
        batch.forces = torch.zeros_like(batch.positions)
        dynamics._init_state(batch)
        dynamics.cell_force_scale = 3.0
        target = "nvalchemi.dynamics._ops.fire._fire2_coord_cell"
        with patch(target) as ops:
            dynamics.pre_update(batch)
        assert ops.call_args.kwargs["cell_force_scale"] == 3.0

    def test_lbfgs_kappa_uses_scale_including_refill(self):
        dynamics = LBFGSVariableCell(
            model=_make_model(needs_stress=True), cell_force_scale=2.5
        )
        batch = _cell_batch([None, None], n_atoms=4)
        dynamics._init_state(batch)
        torch.testing.assert_close(
            dynamics._state.kappa, torch.full((2,), 10.0, dtype=torch.float32)
        )
        keep = torch.tensor([1])
        trimmed = batch.index_select(keep)
        trimmed.append(Batch.from_data_list([_cell_data(6, 3)]))
        dynamics._sync_state_to_batch(keep, 1, trimmed)
        assert dynamics._state.kappa.tolist() == [10.0, 15.0]


# ---------------------------------------------------------------------------
# maxstep / curvature_eps forwarding
# ---------------------------------------------------------------------------


class TestLBFGSStepKwargsForwarding:
    """``maxstep``/``curvature_eps`` reach the ops call with non-default
    values, so a regression that drops either kwarg can't hide behind the
    ops-level default happening to equal the class-level default.
    """

    def test_lbfgs_forwards_maxstep_and_curvature_eps(self):
        dynamics = LBFGS(model=_make_model(), maxstep=0.37, curvature_eps=1e-5)
        batch = _make_batch(1, n_atoms_each=4, seed=8)
        batch.forces = torch.zeros_like(batch.positions)
        dynamics._init_state(batch)
        target = "nvalchemi.dynamics._ops.lbfgs._lbfgs_coord"
        with patch(target) as ops:
            dynamics.pre_update(batch)
        assert ops.call_args.kwargs["maxstep"] == 0.37
        assert ops.call_args.kwargs["curvature_eps"] == 1e-5

    def test_lbfgs_variable_cell_forwards_maxstep_and_curvature_eps(self):
        dynamics = LBFGSVariableCell(
            model=_make_model(needs_stress=True), maxstep=0.37, curvature_eps=1e-5
        )
        batch = _cell_batch([None])
        batch.forces = torch.zeros_like(batch.positions)
        batch.stress = torch.zeros(1, 3, 3)
        dynamics._init_state(batch)
        target = "nvalchemi.dynamics._ops.lbfgs._lbfgs_coord_cell"
        with patch(target) as ops:
            dynamics.pre_update(batch)
        assert ops.call_args.kwargs["maxstep"] == 0.37
        assert ops.call_args.kwargs["curvature_eps"] == 1e-5


# ---------------------------------------------------------------------------
# FusedStage: level-aware masked state
# ---------------------------------------------------------------------------


def _warm_lbfgs(counts=(3, 4), seed=0):
    batch = Batch.from_data_list(
        [_make_atomic_data(c, seed + i) for i, c in enumerate(counts)]
    )
    dynamics = LBFGS(model=_make_model(), history_size=3)
    dynamics._ensure_state_initialized(batch)
    for extra in range(20):
        _random_steps(dynamics, batch, 1, seed=extra)
        if int(dynamics._state.history_count.min()) > 0:
            break
    assert int(dynamics._state.history_count.min()) > 0
    batch.forces = torch.randn_like(batch.positions)
    return dynamics, batch


def _rows(dynamics, system):
    """Snapshot one system's state rows at both levels."""
    ptr = dynamics._state.level_ptr(_DOF_LEVEL)
    lo, hi = int(ptr[system]), int(ptr[system + 1])
    rows = {}
    for level, keys in dynamics._state.level_keys.items():
        for key in keys:
            value = dynamics._state[key]
            rows[key] = (value[lo:hi] if level == _DOF_LEVEL else value[system]).clone()
    return rows


def _status_batch(batch: Batch, statuses: list[int]) -> Batch:
    """Set per-graph ``status`` (FusedStage routing) on *batch*, plus ``fmax``.

    Every FusedStage test below needs ``status`` to route systems to their
    sub-stage.  ``fmax`` is set alongside it even though none of these tests'
    auto-registered ``ConvergenceHook`` instances read it (their default
    criterion checks ``forces``, not ``fmax``): it mirrors the field a real
    relaxation loop (``LoggingHook``, an explicit fmax-based
    ``convergence_hook``) would populate, so a batch built here stays valid
    if a test is later extended to use one.
    """
    batch["status"] = torch.tensor([[s] for s in statuses])
    batch["fmax"] = torch.full((len(statuses), 1), float("inf"))
    return batch


class TestFusedStageMasking:
    @pytest.mark.parametrize("counts", [(3, 4), (1, 1)])
    def test_unmasked_system_state_is_bit_identical(self, counts):
        # (1, 1): num_packed == num_systems, where shape dispatch would mis-blend.
        dynamics, batch = _warm_lbfgs(counts)
        iteration = dynamics._state.iteration.clone()
        untouched = _rows(dynamics, 1)
        dynamics._masked_pre_update(batch, torch.tensor([True, False]))
        assert int(dynamics._state.iteration[0]) == int(iteration[0]) + 1
        for key, value in _rows(dynamics, 1).items():
            assert torch.equal(value, untouched[key]), key

    def test_positions_and_x_base_stay_coupled(self):
        dynamics, batch = _warm_lbfgs()
        node = batch.batch_idx == 1
        gap = (batch.positions.detach() - dynamics._state.x_base)[node].clone()
        dynamics._masked_pre_update(batch, torch.tensor([True, False]))
        assert torch.equal(
            (batch.positions.detach() - dynamics._state.x_base)[node], gap
        )

    def test_all_false_mask_changes_nothing(self):
        dynamics, batch = _warm_lbfgs()
        state = {k: v.clone() for k, v in dynamics._state}
        positions = batch.positions.detach().clone()
        dynamics._masked_pre_update(batch, torch.tensor([False, False]))
        assert torch.equal(batch.positions, positions)
        for key, value in dynamics._state:
            assert torch.equal(value, state[key]), key

    def test_variable_cell_unmasked_system_is_bit_identical(self):
        batch = _cell_batch([None, None])
        dynamics = LBFGSVariableCell(model=_make_model(needs_stress=True))
        dynamics._ensure_state_initialized(batch)
        batch.stress = torch.zeros(2, 3, 3)
        _random_steps(dynamics, batch, 4)
        untouched = _rows(dynamics, 1)
        cell = batch.cell[1].clone()
        dynamics._masked_pre_update(batch, torch.tensor([True, False]))
        assert torch.equal(batch.cell[1], cell)
        for key, value in _rows(dynamics, 1).items():
            assert torch.equal(value, untouched[key]), key


# ---------------------------------------------------------------------------
# _post_update_is_noop: masked post_update skips save/restore entirely
# ---------------------------------------------------------------------------


class TestMaskedPostUpdateSkip:
    """``post_update`` is an unconditional no-op for both LBFGS classes, so
    ``_masked_post_update`` should skip the save/blend-back dance that
    ``_masked_pre_update`` still needs.
    """

    def test_flag_is_set_on_both_classes(self):
        assert LBFGS._post_update_is_noop is True
        assert LBFGSVariableCell._post_update_is_noop is True

    def test_save_restore_helpers_are_not_called(self):
        dynamics, batch = _warm_lbfgs()
        mask = torch.tensor([True, False])
        with (
            patch.object(dynamics, "_save_mutable_fields") as save_fields,
            patch.object(dynamics, "_save_state_fields") as save_state,
            patch.object(dynamics, "_restore_unmasked_fields") as restore_fields,
            patch.object(dynamics, "_restore_unmasked_state") as restore_state,
            patch.object(dynamics, "post_update", wraps=dynamics.post_update) as post,
        ):
            dynamics._masked_post_update(batch, mask)
        save_fields.assert_not_called()
        save_state.assert_not_called()
        restore_fields.assert_not_called()
        restore_state.assert_not_called()
        post.assert_called_once_with(batch)

    def test_masked_post_update_changes_nothing(self):
        # post_update is a real no-op, so skipping save/restore must be
        # observationally identical to running it: nothing should change.
        dynamics, batch = _warm_lbfgs()
        state = {k: v.clone() for k, v in dynamics._state}
        positions = batch.positions.detach().clone()
        dynamics._masked_post_update(batch, torch.tensor([True, False]))
        assert torch.equal(batch.positions, positions)
        for key, value in dynamics._state:
            assert torch.equal(value, state[key]), key

    def test_pre_update_masking_is_unaffected(self):
        # The flag only short-circuits _masked_post_update; _masked_pre_update
        # must still fully save and restore unmasked rows.
        dynamics, batch = _warm_lbfgs()
        iteration = dynamics._state.iteration.clone()
        untouched = _rows(dynamics, 1)
        dynamics._masked_pre_update(batch, torch.tensor([True, False]))
        assert int(dynamics._state.iteration[0]) == int(iteration[0]) + 1
        assert int(dynamics._state.iteration[1]) == int(iteration[1])
        for key, value in _rows(dynamics, 1).items():
            assert torch.equal(value, untouched[key]), key

    def test_fused_stage_still_dispatches_post_update_hooks(self):
        # The skip lives inside _masked_post_update; BEFORE/AFTER_POST_UPDATE
        # hooks are dispatched by FusedStage.step() around that call and must
        # still fire once per step regardless of the fast path.
        lbfgs = LBFGS(model=_make_model())
        before_hook = _CountingHook(DynamicsStage.BEFORE_POST_UPDATE)
        after_hook = _CountingHook(DynamicsStage.AFTER_POST_UPDATE)
        lbfgs.register_hook(before_hook)
        lbfgs.register_hook(after_hook)
        fused = lbfgs + FIRE2(model=_make_model(), dt=0.05)
        batch = _make_batch(2)
        _status_batch(batch, [0, 1])
        fused.step(batch)
        assert before_hook.count == 1
        assert after_hook.count == 1


class TestLevelMask:
    def test_uniform_level_returns_mask(self):
        dynamics = FIRE2(model=_make_model(), dt=0.05)
        dynamics._init_state(_make_batch(3))
        mask = torch.tensor([True, False, True])
        assert _level_mask(dynamics._state, "system", mask) is mask

    def test_segmented_level_expands_ragged(self):
        dynamics, _ = TestLBFGSInflight._ragged((2, 3, 1))
        mask = torch.tensor([True, False, True])
        expanded = _level_mask(dynamics._state, _DOF_LEVEL, mask)
        assert expanded.tolist() == [True, True, False, False, False, True]

    def test_variable_cell_level_is_not_the_atom_count(self):
        dynamics = LBFGSVariableCell(model=_make_model(needs_stress=True))
        dynamics._init_state(_cell_batch([None, None], n_atoms=2))
        expanded = _level_mask(dynamics._state, _DOF_LEVEL, torch.tensor([False, True]))
        assert expanded.tolist() == [False] * 4 + [True] * 4

    def test_unknown_level_raises(self):
        dynamics, _ = _warm_lbfgs()
        with pytest.raises(KeyError):
            _level_mask(dynamics._state, "nope", torch.tensor([True, False]))


class TestFusedStage:
    def test_warm_runs_only_in_outer_loop(self):
        lbfgs = LBFGS(model=_make_model())
        fire2 = FIRE2(model=_make_model(), dt=0.05)
        fused = lbfgs + fire2
        batch = _make_batch(2)
        _status_batch(batch, [0, 1])
        with patch.object(
            lbfgs, "_warm_state_levels", wraps=lbfgs._warm_state_levels
        ) as warm:
            fused.step(batch)
            assert warm.call_count == 1
            lbfgs._masked_pre_update(batch, torch.tensor([True, False]))
            lbfgs._masked_post_update(batch, torch.tensor([True, False]))
            assert warm.call_count == 1

    def test_fused_step_with_lbfgs_substage(self):
        lbfgs = LBFGS(model=_make_model())
        fused = lbfgs + FIRE2(model=_make_model(), dt=0.05)
        batch = _make_batch(2)
        _status_batch(batch, [0, 1])
        for _ in range(3):
            fused.step(batch)
        assert lbfgs._state.iteration.tolist()[0] >= 1
        assert lbfgs._state.iteration.tolist()[1] == -1

    def test_skew_cell_in_other_stage_needs_align_hook(self):
        # Known limitation: without the hook, init validates every system.
        model = _make_model(needs_stress=True)
        batch = _cell_batch([None, _SKEW])
        _status_batch(batch, [0, 1])
        fused = LBFGSVariableCell(model=model) + FIRE2VariableCell(model=model, dt=0.05)
        with pytest.raises(ValueError, match=r"\[1\].*AlignCellHook"):
            fused.step(batch)
        fused = LBFGSVariableCell(
            model=model, hooks=[AlignCellHook()]
        ) + FIRE2VariableCell(model=model, dt=0.05)
        fused.step(batch)

    @staticmethod
    def _skew_fused(hook):
        model = _make_model(needs_stress=True)
        batch = _cell_batch([_SKEW, None])
        _status_batch(batch, [0, 1])
        lbfgs = LBFGSVariableCell(model=model)
        fused = lbfgs + FIRE2VariableCell(model=model, dt=0.05)
        fused.register_hook(hook)  # on the FusedStage, after construction
        return fused, lbfgs, batch

    def test_fused_level_align_hook_is_recognized(self):
        fused, lbfgs, batch = self._skew_fused(AlignCellHook())
        expected = _aligned_periodic(batch)[1]
        fused.step(batch)
        # ref_cell is anti-transposed (see _anti_transpose);
        # _aligned_periodic's cell is row-convention like batch.cell.
        torch.testing.assert_close(lbfgs._state.ref_cell, _anti_transpose(expected))
        assert _aligned(batch.cell)

    def test_fused_level_align_hook_frequency_must_be_one(self):
        fused, _, batch = self._skew_fused(AlignCellHook(frequency=2))
        with pytest.raises(ValueError, match="frequency=1"):
            fused.step(batch)


# ---------------------------------------------------------------------------
# WrapPeriodicHook corrupts the curvature history: must warn
# ---------------------------------------------------------------------------


class TestLBFGSWrapPeriodicWarning:
    def test_own_hook_warns_on_fixed_cell(self):
        batch = _make_batch(2, n_atoms_each=4, seed=6)
        dynamics = LBFGS(
            model=_make_model(),
            hooks=[WrapPeriodicHook(stage=DynamicsStage.AFTER_POST_UPDATE)],
        )
        with pytest.warns(UserWarning, match="WrapPeriodicHook"):
            dynamics._ensure_state_initialized(batch)

    def test_own_hook_warns_on_variable_cell(self):
        batch = _cell_batch([None, None])
        dynamics = LBFGSVariableCell(
            model=_make_model(needs_stress=True),
            hooks=[WrapPeriodicHook(stage=DynamicsStage.AFTER_POST_UPDATE)],
        )
        with pytest.warns(UserWarning, match="WrapPeriodicHook"):
            dynamics._ensure_state_initialized(batch)

    def test_no_hook_does_not_warn(self):
        batch = _make_batch(2, n_atoms_each=4, seed=7)
        dynamics = LBFGS(model=_make_model())
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            dynamics._ensure_state_initialized(batch)

    def test_enclosing_fused_stage_hook_warns(self):
        # Registered on the FusedStage, not lbfgs itself; only checking
        # detection here, so a non-periodic batch is enough — the hook
        # never actually has to run.
        lbfgs = LBFGS(model=_make_model())
        fused = lbfgs + FIRE2(model=_make_model(), dt=0.05)
        fused.register_hook(WrapPeriodicHook(), stage=DynamicsStage.AFTER_POST_UPDATE)
        batch = _make_batch(2)
        with pytest.warns(UserWarning, match="WrapPeriodicHook"):
            lbfgs._ensure_state_initialized(batch)

    def test_late_direct_registration_warns(self):
        # Registered only *after* the optimizer's first check (e.g. after
        # its first step): must still be caught, not just at admission.
        dynamics = LBFGS(model=_make_model())
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            dynamics._check_hook_compatibility()  # no hook yet: silent
        dynamics.register_hook(WrapPeriodicHook(stage=DynamicsStage.AFTER_POST_UPDATE))
        with pytest.warns(UserWarning, match="WrapPeriodicHook"):
            dynamics._check_hook_compatibility()

    def test_late_enclosing_registration_warns(self):
        # Same, but the hook is registered on the enclosing FusedStage
        # after composition, not on the LBFGS sub-stage itself.
        lbfgs = LBFGS(model=_make_model())
        fused = lbfgs + FIRE2(model=_make_model(), dt=0.05)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            lbfgs._check_hook_compatibility()  # no hook on fused yet: silent
        fused.register_hook(WrapPeriodicHook(), stage=DynamicsStage.AFTER_POST_UPDATE)
        with pytest.warns(UserWarning, match="WrapPeriodicHook"):
            lbfgs._check_hook_compatibility()

    def test_step_rechecks_every_call(self):
        # End-to-end: step() itself re-runs the check every call (not just
        # via _ensure_state_initialized at first admission), so a hook
        # registered between two step() calls is still caught.
        batch = _make_batch(2, n_atoms_each=4, seed=9, with_cell=True)
        batch.pbc = torch.ones(2, 3, dtype=torch.bool)
        dynamics = LBFGS(model=_make_model(), n_steps=2)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            dynamics.step(batch)
        dynamics.register_hook(WrapPeriodicHook(stage=DynamicsStage.AFTER_POST_UPDATE))
        with pytest.warns(UserWarning, match="WrapPeriodicHook"):
            dynamics.step(batch)


# ---------------------------------------------------------------------------
# torch.compile (fullgraph)
# ---------------------------------------------------------------------------


def _compile(fn):
    torch.compiler.reset()
    return torch.compile(fn, backend="eager", fullgraph=True)


class TestCompile:
    @staticmethod
    def _assert_warm(dynamics):
        level = dynamics._state._storage.groups[_DOF_LEVEL]
        assert level._batch_idx is not None and level._batch_ptr is not None

    @pytest.mark.parametrize("variable_cell", [False, True])
    def test_masked_pre_update_fullgraph_cold_and_after_refill(self, variable_cell):
        if variable_cell:
            batch = _cell_batch([None, None, None])
            batch.stress = torch.zeros(3, 3, 3)
            dynamics = LBFGSVariableCell(model=_make_model(needs_stress=True))
        else:
            batch = _make_batch(3)
            dynamics = LBFGS(model=_make_model())
        batch.forces = torch.randn_like(batch.positions)
        mask = torch.tensor([True, False, True])

        dynamics._ensure_state_initialized(batch)
        dynamics._warm_state_levels()
        self._assert_warm(dynamics)
        _ = batch.batch_idx  # the data batch's own lazy index, as for FIRE2
        _compile(dynamics._masked_pre_update)(batch, mask)

        keep = torch.tensor([0, 2])
        trimmed = batch.index_select(keep)
        new = _cell_data(5, 9) if variable_cell else _make_atomic_data(5, 9)
        trimmed.append(Batch.from_data_list([new]))
        dynamics._sync_state_to_batch(keep, 1, trimmed)
        dynamics._warm_state_levels()
        self._assert_warm(dynamics)
        trimmed.forces = torch.randn_like(trimmed.positions)
        if variable_cell:
            trimmed.stress = torch.zeros(3, 3, 3)
        _ = trimmed.batch_idx
        _compile(dynamics._masked_pre_update)(trimmed, mask)

    @pytest.mark.parametrize("variable_cell", [False, True])
    def test_masked_post_update_fullgraph(self, variable_cell):
        # _post_update_is_noop's branch is on a plain Python bool class
        # attribute, not a tensor, so it must not break fullgraph tracing.
        if variable_cell:
            batch = _cell_batch([None, None, None])
            batch.stress = torch.zeros(3, 3, 3)
            dynamics = LBFGSVariableCell(model=_make_model(needs_stress=True))
        else:
            batch = _make_batch(3)
            dynamics = LBFGS(model=_make_model())
        batch.forces = torch.randn_like(batch.positions)
        mask = torch.tensor([True, False, True])

        dynamics._ensure_state_initialized(batch)
        dynamics._warm_state_levels()
        self._assert_warm(dynamics)
        _ = batch.batch_idx
        state_before = {k: v.clone() for k, v in dynamics._state}
        positions_before = batch.positions.detach().clone()

        _compile(dynamics._masked_post_update)(batch, mask)

        assert torch.equal(batch.positions, positions_before)
        for key, value in dynamics._state:
            assert torch.equal(value, state_before[key]), key

    @pytest.mark.parametrize("cell", [None, _SKEW])
    def test_align_cell_hook_fullgraph(self, cell):
        batch = _cell_batch([None, cell])
        eager = _aligned_periodic(batch)
        compiled = _compile(_aligned_periodic)(batch)
        if eager is None:  # already aligned: compiled path is a no-op blend
            torch.testing.assert_close(compiled[1], batch.cell)
        else:
            torch.testing.assert_close(compiled[1], eager[1])

    def test_align_cell_hook_fullgraph_all_aligned_is_bit_identical(self):
        # The compiled path is branchless (no eager early return), so it
        # must still preserve an already-aligned cell exactly via the
        # per-system needs_align mask, not just approximately — the whole
        # point is no ULP-level nudging on every step.  A non-trivial
        # (non-diagonal) aligned cell, since align_cell reproduces a
        # trivial k*I cell exactly anyway.
        aligned = _already_aligned_nontrivial_cell()
        batch = _cell_batch([aligned, aligned])
        positions_before = batch.positions.detach().clone()
        cell_before = batch.cell.clone()
        compiled = _compile(_aligned_periodic)(batch)
        assert torch.equal(compiled[0], positions_before)
        assert torch.equal(compiled[1], cell_before)

    def test_align_cell_hook_fullgraph_mixed_batch_preserves_aligned_system(self):
        batch = _cell_batch([_already_aligned_nontrivial_cell(), _SKEW])
        positions_before = batch.positions.detach().clone()
        cell_before = batch.cell.clone()
        compiled = _compile(_aligned_periodic)(batch)
        sys0_atoms = batch.batch_idx == 0
        assert torch.equal(compiled[0][sys0_atoms], positions_before[sys0_atoms])
        assert torch.equal(compiled[1][0], cell_before[0])
        assert not torch.equal(compiled[1][1], cell_before[1])

    def test_align_cell_hook_fullgraph_left_handed(self):
        cell = torch.diag(torch.tensor([-5.0, 5.0, 5.0]))
        batch = _cell_batch([cell])
        compiled = _compile(_aligned_periodic)(batch)
        assert torch.linalg.det(compiled[1]).item() > 0


# ---------------------------------------------------------------------------
# Stale reference chart (FusedStage stage entry)
# ---------------------------------------------------------------------------


def _argon(cell, seed=0):
    base = torch.tensor([[0, 0, 0], [0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]])
    shifts = torch.tensor([[i, j, k] for i in (0, 1) for j in (0, 1) for k in (0, 1)])
    frac = ((base[None] + shifts[:, None]) / 2).reshape(-1, 3).double()
    positions = frac @ cell.T
    positions += 0.05 * torch.randn(
        positions.shape,
        generator=torch.Generator().manual_seed(seed),
        dtype=torch.float64,
    )
    data = AtomicData(
        positions=positions,
        atomic_numbers=torch.full((32,), 18),
        cell=cell.unsqueeze(0),
        pbc=torch.tensor([[True] * 3]),
        forces=torch.zeros_like(positions),
        energy=torch.zeros(1, 1, dtype=torch.float64),
        stress=torch.zeros(1, 3, 3, dtype=torch.float64),
    )
    return Batch.from_data_list([data])


def _relax_argon(reference=None, steps=80):
    from nvalchemi.hooks import NeighborListHook
    from nvalchemi.models.lj import LennardJonesModelWrapper

    model = LennardJonesModelWrapper(epsilon=0.0104, sigma=3.40, cutoff=8.5)
    model.set_config("active_outputs", {"energy", "forces", "stress"})
    neighbors = NeighborListHook(
        model.model_config.neighbor_config, stage=DynamicsStage.BEFORE_COMPUTE
    )
    dynamics = LBFGSVariableCell(model=model, hooks=[AlignCellHook(), neighbors])
    batch = _argon(11.4 * torch.eye(3, dtype=torch.float64))
    if reference is not None:
        # As when a system enters the L-BFGS stage after another stage moved
        # its cell: the chart was captured from a different (aligned) cell.
        dynamics._init_state(_argon(reference.double()))
    for n in range(1, steps + 1):
        dynamics.step(batch)
        if batch.forces.norm(dim=1).max() < 1e-4 and batch.stress.abs().max() < 1e-6:
            return n, torch.linalg.det(batch.cell[0]).item() ** (1 / 3)
    raise AssertionError(f"not converged in {steps} steps")


class TestStaleReferenceCell:
    def test_stale_reference_cell_still_converges(self):
        fresh_steps, fresh_edge = _relax_argon()
        for reference in (
            torch.diag(torch.tensor([10.0, 12.0, 13.5])),
            torch.tensor([[11.4, 0.0, 0.0], [1.5, 11.0, 0.0], [0.8, -0.6, 12.0]]),
        ):
            steps, edge = _relax_argon(reference)
            assert edge == pytest.approx(fresh_edge, abs=1e-3)
            assert steps <= 2 * fresh_steps


# ---------------------------------------------------------------------------
# Row/column cell convention regression (finite-difference ground truth)
# ---------------------------------------------------------------------------


class TestLBFGSVariableCellPhysicalCorrectness:
    """Finite-difference regression for the row/column cell convention.

    ``LBFGSVariableCell`` must hand nvalchemiops' backend the cell, stress,
    positions and forces in the convention its packed six-coordinate chart
    (``cell_dof_a/b``, ``cell_force_a/b``) actually assumes -- see
    ``_anti_transpose`` for the derivation.  Getting this wrong doesn't
    raise: it silently produces a cell force that isn't
    ``-dE/d(packed dof)``, which either converges to the wrong state or (if
    the wrong convention also loses information structurally, as a plain
    transpose does) never corrects a genuine shear mismatch at all --
    see ``TestStaleReferenceCell``, which exercises that failure mode
    end-to-end.  This test instead isolates the gradient itself: it
    compares ``cell_force_a/b`` directly against a finite-difference
    estimate of a real periodic Lennard-Jones energy, varying each of the
    six packed degrees of freedom independently, for a reference cell with
    real (non-diagonal) shear relative to the current one -- the case a
    plain transpose structurally cannot represent.
    """

    def test_cell_force_matches_finite_difference(self):
        from nvalchemi.dynamics.optimizers.lbfgs import _anti_transpose
        from nvalchemi.hooks import NeighborListHook
        from nvalchemi.models.lj import LennardJonesModelWrapper

        dtype = torch.float64
        # Both lower-triangular (aligned) and genuinely different, so the
        # deformation gradient between them has real shear content.
        current = torch.tensor(
            [[11.0, 0.0, 0.0], [0.6, 10.8, 0.0], [-0.3, 0.4, 11.3]], dtype=dtype
        )
        reference = torch.tensor(
            [[11.4, 0.0, 0.0], [1.5, 11.0, 0.0], [0.8, -0.6, 12.0]], dtype=dtype
        )
        base = torch.tensor([[0, 0, 0], [0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]])
        shifts = torch.tensor(
            [[i, j, k] for i in (0, 1) for j in (0, 1) for k in (0, 1)]
        )
        frac = ((base[None] + shifts[:, None]) / 2).reshape(-1, 3).double()
        positions = frac @ current.T
        positions += 0.05 * torch.randn(
            positions.shape, generator=torch.Generator().manual_seed(1), dtype=dtype
        )

        def make_batch(cell, pos):
            data = AtomicData(
                positions=pos.clone(),
                atomic_numbers=torch.full((pos.shape[0],), 18),
                cell=cell.unsqueeze(0).clone(),
                pbc=torch.tensor([[True] * 3]),
                forces=torch.zeros_like(pos),
                energy=torch.zeros(1, 1, dtype=dtype),
                stress=torch.zeros(1, 3, 3, dtype=dtype),
            )
            return Batch.from_data_list([data])

        model = LennardJonesModelWrapper(epsilon=0.0104, sigma=3.40, cutoff=8.5)
        model.set_config("active_outputs", {"energy", "forces", "stress"})
        neighbors = NeighborListHook(
            model.model_config.neighbor_config, stage=DynamicsStage.BEFORE_COMPUTE
        )
        optimizer = LBFGSVariableCell(model=model, maxstep=1e-8)
        optimizer.register_hook(neighbors, stage=DynamicsStage.BEFORE_COMPUTE)

        def energy_of(cell, pos):
            b = make_batch(cell, pos)
            optimizer._call_hooks(DynamicsStage.BEFORE_COMPUTE, b, None)
            optimizer.compute(b)
            return b.energy.item()

        batch = make_batch(current, positions)
        optimizer._call_hooks(DynamicsStage.BEFORE_COMPUTE, batch, None)
        optimizer.compute(batch)
        optimizer._init_state(make_batch(reference, positions))
        optimizer.pre_update(batch)

        kappa = optimizer._state.kappa[0].item()
        predicted = torch.cat(
            [
                -kappa * optimizer._state.cell_force_a[0],
                -kappa * optimizer._state.cell_force_b[0],
            ]
        )

        # Ground truth: perturb each packed dof slot directly -- an entry of
        # Phi = anti_transpose(current) @ anti_transpose(reference)^-1 -- and
        # finite-difference the *real* model's energy with the reference and
        # the fractional coordinates of `positions` (relative to `current`)
        # held fixed, exactly what "holding the chart fixed" means.  This
        # does not reimplement the backend's math: it only uses
        # ``_anti_transpose`` to translate a dof slot into a physical cell.
        ref_fed = _anti_transpose(reference)
        phi = _anti_transpose(current) @ torch.linalg.inv(ref_fed)
        frac_fixed = positions @ torch.linalg.inv(current)
        slots = [(0, 0), (1, 0), (2, 0), (1, 1), (2, 1), (2, 2)]

        expected = []
        eps = 1e-6
        for i, j in slots:
            plus, minus = phi.clone(), phi.clone()
            plus[i, j] += eps
            minus[i, j] -= eps
            cell_plus = _anti_transpose(plus @ ref_fed)
            cell_minus = _anti_transpose(minus @ ref_fed)
            e_plus = energy_of(cell_plus, frac_fixed @ cell_plus)
            e_minus = energy_of(cell_minus, frac_fixed @ cell_minus)
            expected.append((e_plus - e_minus) / (2 * eps))

        torch.testing.assert_close(
            predicted, torch.tensor(expected, dtype=dtype), atol=1e-4, rtol=1e-4
        )
