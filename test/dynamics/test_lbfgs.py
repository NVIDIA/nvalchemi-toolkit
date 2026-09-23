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
from unittest.mock import patch

import pytest
import torch
from nvalchemiops.torch.lbfgs import LBFGSCellState, LBFGSState

from nvalchemi.data import AtomicData, Batch
from nvalchemi.data.level_storage import SegmentedLevelStorage
from nvalchemi.dynamics import ConvergenceHook, DynamicsStage
from nvalchemi.dynamics._ops._bridge import _state_level
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
    _ops_state,
)

from .test_state_management import (
    _make_atomic_data,
    _make_batch,
    _make_model,
    _MockSampler,
)

_SKEW = torch.tensor([[5.0, 1.0, 0.0], [0.0, 5.0, 0.0], [0.3, 0.2, 5.0]])


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
    return torch.triu(cell, 1).abs().max().item() <= 1e-10


class _Record:
    """Hook recording ``batch.cell`` at one stage."""

    frequency = 1

    def __init__(self, stage):
        self.stage = stage
        self.cells = []

    def __call__(self, ctx, stage):
        self.cells.append(ctx.batch.cell.detach().clone())


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
        torch.manual_seed(0)
        batch = _make_batch(2, n_atoms_each=5, seed=1)
        dynamics = LBFGS(
            model=_make_model(),
            n_steps=20,
            convergence_hook=ConvergenceHook.from_fmax(1e-8),
        )
        _forces(dynamics, batch)
        before = batch.forces.norm(dim=1).max().item()
        dynamics.run(batch)
        assert batch.forces.norm(dim=1).max().item() < before
        assert dynamics.step_count == 20

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
        level = _state_level(dynamics._state, _DOF_LEVEL)
        assert level.segment_lengths.tolist() == [6, 6, 6]
        assert "ext_batch_idx" not in {key for key, _ in dynamics._state}

    @pytest.mark.parametrize("name", ["history_size", "cell_force_scale"])
    def test_allocation_fixed_parameters_are_read_only(self, name):
        dynamics = LBFGSVariableCell(model=_make_model(needs_stress=True))
        with pytest.raises(AttributeError, match="fixed when optimizer state"):
            setattr(dynamics, name, 2)


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
        level = _state_level(dynamics._state, _DOF_LEVEL)
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
        torch.testing.assert_close(dynamics._state.ref_cell, record.cells[0])

    def test_aligned_cells_need_no_hook(self):
        batch = _cell_batch([None, None])
        self._dynamics(n_steps=2).run(batch)

    def test_skew_without_hook_raises(self):
        batch = _cell_batch([None, _SKEW])
        with pytest.raises(ValueError, match=r"system\(s\) \[1\].*AlignCellHook\(\)"):
            self._dynamics()._init_state(batch)

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
        # The chart is the aligned pre-step cell; the live cell moves but stays aligned.
        torch.testing.assert_close(ref_cell, record.cells[-1][-1])
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
    @pytest.mark.parametrize("scale", [0.0, -1.0])
    def test_non_positive_rejected(self, cls, scale):
        kwargs = {"dt": 0.05} if cls is FIRE2VariableCell else {}
        with pytest.raises(ValueError, match="positive"):
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
        batch["status"] = torch.tensor([[0], [1]])
        batch["fmax"] = torch.full((2, 1), float("inf"))
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
        batch["status"] = torch.tensor([[0], [1]])
        batch["fmax"] = torch.full((2, 1), float("inf"))
        for _ in range(3):
            fused.step(batch)
        assert lbfgs._state.iteration.tolist()[0] >= 1
        assert lbfgs._state.iteration.tolist()[1] == -1

    def test_skew_cell_in_other_stage_needs_align_hook(self):
        # Known limitation: without the hook, init validates every system.
        model = _make_model(needs_stress=True)
        batch = _cell_batch([None, _SKEW])
        batch["status"] = torch.tensor([[0], [1]])
        batch["fmax"] = torch.full((2, 1), float("inf"))
        fused = LBFGSVariableCell(model=model) + FIRE2VariableCell(model=model, dt=0.05)
        with pytest.raises(ValueError, match=r"\[1\].*AlignCellHook"):
            fused.step(batch)
        fused = LBFGSVariableCell(
            model=model, hooks=[AlignCellHook()]
        ) + FIRE2VariableCell(model=model, dt=0.05)
        fused.step(batch)


# ---------------------------------------------------------------------------
# torch.compile (fullgraph)
# ---------------------------------------------------------------------------


def _compile(fn):
    torch.compiler.reset()
    return torch.compile(fn, backend="eager", fullgraph=True)


class TestCompile:
    @staticmethod
    def _assert_warm(dynamics):
        level = _state_level(dynamics._state, _DOF_LEVEL)
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

    @pytest.mark.parametrize("cell", [None, _SKEW])
    def test_align_cell_hook_fullgraph(self, cell):
        batch = _cell_batch([None, cell])
        eager = _aligned_periodic(batch)
        compiled = _compile(_aligned_periodic)(batch)
        if eager is None:  # already aligned: compiled path is a no-op blend
            torch.testing.assert_close(compiled[1], batch.cell)
        else:
            torch.testing.assert_close(compiled[1], eager[1])


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


def test_stale_reference_cell_still_converges():
    fresh_steps, fresh_edge = _relax_argon()
    for reference in (
        torch.diag(torch.tensor([10.0, 12.0, 13.5])),
        torch.tensor([[11.4, 0.0, 0.0], [1.5, 11.0, 0.0], [0.8, -0.6, 12.0]]),
    ):
        steps, edge = _relax_argon(reference)
        assert edge == pytest.approx(fresh_edge, abs=1e-3)
        assert steps <= 2 * fresh_steps
