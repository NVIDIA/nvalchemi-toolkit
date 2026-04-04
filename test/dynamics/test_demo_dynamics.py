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
"""
Unit tests for DemoDynamics paired with DemoModelWrapper.

This module tests the ``DemoDynamics`` Velocity Verlet integrator end-to-end
with ``DemoModelWrapper``.  Tests cover first-step behaviour, multi-graph
independence, interface contracts, full-pipeline behaviour, and
FusedStage / DistributedPipeline composition.
"""

from __future__ import annotations

from collections import OrderedDict

import pytest
import torch

from nvalchemi._typing import ModelOutputs
from nvalchemi.data import AtomicData, Batch
from nvalchemi.dynamics.base import (
    BaseDynamics,
    ConvergenceHook,
    DistributedPipeline,
    DynamicsStage,
    FusedStage,
)
from nvalchemi.dynamics.demo import DemoDynamics
from nvalchemi.models.demo import DemoModelWrapper

from .conftest import RecordingHook

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_batch(n_atoms: int = 1) -> Batch:
    """
    Create a single-graph batch with carbon atoms at the origin.

    Parameters
    ----------
    n_atoms : int
        Number of atoms in the graph.

    Returns
    -------
    Batch
        Batch with pre-allocated ``forces`` and ``energies``.
    """
    data = AtomicData(
        atomic_numbers=torch.tensor([6] * n_atoms, dtype=torch.long),
        positions=torch.zeros(n_atoms, 3),
    )
    batch = Batch.from_data_list([data])
    batch.forces = torch.zeros(n_atoms, 3)
    batch.energies = torch.zeros(1, 1)
    return batch


def _make_multi_batch(
    n_graphs: int = 2,
    n_atoms_per_graph: int = 3,
) -> Batch:
    """
    Create a multi-graph batch with random positions.

    Parameters
    ----------
    n_graphs : int
        Number of graphs.
    n_atoms_per_graph : int
        Atoms per graph.

    Returns
    -------
    Batch
        Batch with pre-allocated ``forces`` and ``energies``.
    """
    data_list = [
        AtomicData(
            atomic_numbers=torch.tensor(
                [6] * n_atoms_per_graph,
                dtype=torch.long,
            ),
            positions=torch.randn(n_atoms_per_graph, 3),
        )
        for _ in range(n_graphs)
    ]
    batch = Batch.from_data_list(data_list)
    batch.forces = torch.zeros(batch.num_nodes, 3)
    batch.energies = torch.zeros(batch.num_graphs, 1)
    return batch


# ---------------------------------------------------------------------------
# First-step behaviour
# ---------------------------------------------------------------------------


class TestFirstStep:
    """
    Test first-step behaviour when ``_prev_accelerations`` is None.

    ``DemoDynamics`` should fall back to Euler integration for
    velocities on the very first step.
    """

    def test_first_step_no_error(self) -> None:
        """Verify that the first step executes without error."""
        model = DemoModelWrapper()
        dynamics = DemoDynamics(model=model, n_steps=1, dt=1.0)

        batch = _make_batch()
        dynamics.step(batch)

        assert dynamics.step_count == 1

    @pytest.mark.parametrize("int_dtype", [torch.int32, torch.int64])
    def test_first_step_with_int_dtypes(self, device, int_dtype: torch.dtype) -> None:
        """Dynamics step works with both int32 and int64 atomic_numbers."""
        model = DemoModelWrapper().to(device)
        dynamics = DemoDynamics(model=model, n_steps=1, dt=1.0)

        data = AtomicData(
            atomic_numbers=torch.tensor([6, 6, 6], dtype=int_dtype),
            positions=torch.zeros(3, 3),
        )
        batch = Batch.from_data_list([data]).to(device)
        batch.forces = torch.zeros(3, 3, device=device)
        batch.energies = torch.zeros(1, 1, device=device)
        dynamics.step(batch)

        assert dynamics.step_count == 1

    def test_first_step_updates_velocities(self) -> None:
        """
        Verify that the first step produces non-zero velocities.

        ``DemoModelWrapper`` produces non-trivial forces, so after one
        Euler-fallback velocity update the velocities must differ from
        the initial zeros.
        """
        model = DemoModelWrapper()
        dynamics = DemoDynamics(model=model, n_steps=1, dt=1.0)

        batch = _make_batch()
        assert dynamics._prev_accelerations is None

        dynamics.step(batch)

        assert not torch.allclose(
            batch.velocities,
            torch.zeros_like(batch.velocities),
        )


# ---------------------------------------------------------------------------
# Multi-graph independence
# ---------------------------------------------------------------------------


class TestMultiGraph:
    """
    Test that graphs in a batch evolve independently.

    Each molecule should only be affected by its own forces.
    """

    def test_independent_evolution(self) -> None:
        """
        Verify two molecules with different initial velocities both move.

        Molecule 1 starts at the origin with ``v = [+1, 0, 0]``.
        Molecule 2 starts at ``[10, 0, 0]`` with ``v = [-1, 0, 0]``.
        After several steps both positions must have changed.
        """
        model = DemoModelWrapper()
        dynamics = DemoDynamics(model=model, n_steps=5, dt=1.0)

        data1 = AtomicData(
            atomic_numbers=torch.tensor([6], dtype=torch.long),
            positions=torch.tensor([[0.0, 0.0, 0.0]]),
        )
        data2 = AtomicData(
            atomic_numbers=torch.tensor([6], dtype=torch.long),
            positions=torch.tensor([[10.0, 0.0, 0.0]]),
        )
        batch = Batch.from_data_list([data1, data2])
        batch.forces = torch.zeros(batch.num_nodes, 3)
        batch.energies = torch.zeros(batch.num_graphs, 1)

        batch.velocities[0] = torch.tensor([1.0, 0.0, 0.0])
        batch.velocities[1] = torch.tensor([-1.0, 0.0, 0.0])

        x0_mol1 = batch.positions[0].clone()
        x0_mol2 = batch.positions[1].clone()

        dynamics.run(batch, 5)

        assert not torch.allclose(batch.positions[0], x0_mol1)
        assert not torch.allclose(batch.positions[1], x0_mol2)


# ---------------------------------------------------------------------------
# Interface contract
# ---------------------------------------------------------------------------


class TestInterfaceContract:
    """
    Verify the public interface contract of ``DemoDynamics``.

    Checks return types, step counter, hook firing order, and that
    ``pre_update`` / ``post_update`` modify the expected fields.
    """

    def test_step_returns_same_batch(self) -> None:
        """``step()`` must return the *same* batch object (in-place)."""
        model = DemoModelWrapper()
        dynamics = DemoDynamics(model=model, n_steps=1, dt=1.0)
        batch = _make_batch()

        result, _ = dynamics.step(batch)

        assert result is batch

    def test_run_increments_step_count(self) -> None:
        """After ``run(batch, N)``, ``step_count`` must equal *N*."""
        model = DemoModelWrapper()
        dynamics = DemoDynamics(model=model, n_steps=7, dt=1.0)
        batch = _make_batch()

        n_steps = 7
        dynamics.run(batch, n_steps)

        assert dynamics.step_count == n_steps

    def test_hooks_fire_in_order(self) -> None:
        """
        A single step must fire hooks in the canonical order.

        Expected: BEFORE_STEP -> BEFORE_PRE_UPDATE -> AFTER_PRE_UPDATE ->
        BEFORE_COMPUTE -> AFTER_COMPUTE -> BEFORE_POST_UPDATE ->
        AFTER_POST_UPDATE -> AFTER_STEP.
        """
        model = DemoModelWrapper()
        dynamics = DemoDynamics(model=model, n_steps=1, dt=1.0)

        record_list: list[str] = []
        stages = [
            DynamicsStage.BEFORE_STEP,
            DynamicsStage.BEFORE_PRE_UPDATE,
            DynamicsStage.AFTER_PRE_UPDATE,
            DynamicsStage.BEFORE_COMPUTE,
            DynamicsStage.AFTER_COMPUTE,
            DynamicsStage.BEFORE_POST_UPDATE,
            DynamicsStage.AFTER_POST_UPDATE,
            DynamicsStage.AFTER_STEP,
        ]
        for stage in stages:
            dynamics.register_hook(
                RecordingHook(stage, record_list, name=stage.name),
            )

        batch = _make_batch()
        dynamics.step(batch)

        assert record_list == [s.name for s in stages]

    def test_pre_update_changes_positions(self) -> None:
        """``pre_update`` must change positions when velocity is non-zero."""
        model = DemoModelWrapper()
        dynamics = DemoDynamics(model=model, n_steps=1, dt=1.0)

        batch = _make_batch()
        batch.velocities = torch.tensor([[1.0, 0.0, 0.0]])
        original = batch.positions.clone()

        dynamics.pre_update(batch)

        assert not torch.allclose(batch.positions, original)

    def test_post_update_changes_velocities(self) -> None:
        """``post_update`` must change velocities when forces are non-zero."""
        model = DemoModelWrapper()
        dynamics = DemoDynamics(model=model, n_steps=1, dt=1.0)

        batch = _make_batch()
        batch.forces = torch.ones(1, 3)
        original = batch.velocities.clone()

        dynamics.post_update(batch)

        assert not torch.allclose(batch.velocities, original)


# ---------------------------------------------------------------------------
# End-to-end dynamics
# ---------------------------------------------------------------------------


class TestEndToEnd:
    """
    End-to-end tests running ``DemoDynamics`` with ``DemoModelWrapper``.

    Verify that positions, velocities, energies, and forces are all
    populated after a short simulation.
    """

    def setup_method(self) -> None:
        """Create a fresh model and dynamics for each test."""
        self.model = DemoModelWrapper()
        self.dynamics = DemoDynamics(model=self.model, n_steps=10, dt=1.0)

    def test_positions_change(self) -> None:
        """Positions must differ from their initial values after 10 steps."""
        batch = _make_multi_batch()
        original = batch.positions.clone()

        self.dynamics.run(batch, 10)

        assert not torch.allclose(batch.positions, original)

    def test_velocities_change(self) -> None:
        """Velocities must differ from the initial zeros after 10 steps."""
        batch = _make_multi_batch()
        original = batch.velocities.clone()

        self.dynamics.run(batch, 10)

        assert not torch.allclose(batch.velocities, original)

    def test_energies_populated(self) -> None:
        """``batch.energies`` must not be ``None`` after a run."""
        batch = _make_multi_batch()

        self.dynamics.run(batch, 10)

        assert batch.energies is not None

    def test_forces_populated(self) -> None:
        """``batch.forces`` must be non-zero after a run."""
        batch = _make_multi_batch()

        self.dynamics.run(batch, 10)

        assert batch.forces is not None
        assert not torch.allclose(batch.forces, torch.zeros_like(batch.forces))


# ---------------------------------------------------------------------------
# FusedStage / DistributedPipeline composition
# ---------------------------------------------------------------------------


class TestFusedStageIntegration:
    """
    Test ``DemoDynamics`` in FusedStage and DistributedPipeline compositions.

    ``DemoDynamics`` inherits ``CommunicationMixin``, so it supports
    ``+`` (FusedStage) and ``|`` (DistributedPipeline) operators.
    """

    def test_fused_step_changes_positions(self) -> None:
        """Two fused ``DemoDynamics`` must update all sample positions."""
        model = DemoModelWrapper()

        dyn0 = DemoDynamics(model=model, n_steps=1, dt=1.0)
        dyn1 = DemoDynamics(model=model, n_steps=1, dt=1.0)

        fused = FusedStage(sub_stages=[(0, dyn0), (1, dyn1)])

        data_list = [
            AtomicData(
                atomic_numbers=torch.tensor([6], dtype=torch.long),
                positions=torch.tensor([[float(i), 0.0, 0.0]]),
            )
            for i in range(4)
        ]
        batch = Batch.from_data_list(data_list)
        batch.forces = torch.zeros(batch.num_nodes, 3)
        batch.energies = torch.zeros(batch.num_graphs, 1)
        batch.status = torch.tensor([0, 0, 1, 1], dtype=torch.long)
        batch.fmax = torch.tensor([[1.0]] * 4)
        # Give non-zero velocities so pre_update (which runs before compute in
        # FusedStage) moves positions via the Euler fallback x += v*dt.
        batch.velocities = torch.ones(batch.num_nodes, 3)

        original = batch.positions.clone()

        fused.step(batch)

        assert not torch.allclose(batch.positions, original)

    def test_plus_creates_fused_stage(self) -> None:
        """``dyn1 + dyn2`` must return a ``FusedStage``."""
        model = DemoModelWrapper()
        dyn1 = DemoDynamics(model=model, n_steps=1, dt=1.0)
        dyn2 = DemoDynamics(model=model, n_steps=1, dt=1.0)

        result = dyn1 + dyn2

        assert isinstance(result, FusedStage)
        assert len(result.sub_stages) == 2

    def test_multiple_fused(self) -> None:
        """``dyn1 + dyn2 + dyn3`` must return a ``FusedStage`` that converges.

        Three sub-stages create exit_status=3. Auto-registered hooks create
        0→1 and 1→2 transitions. We manually add 2→3 (exit) hook on dyn3.
        """
        model = DemoModelWrapper()
        dyn1 = DemoDynamics(model=model, n_steps=1, dt=1.0)
        dyn2 = DemoDynamics(model=model, n_steps=1, dt=1.0)
        dyn3 = DemoDynamics(model=model, n_steps=1, dt=1.0)

        # Manually register hook for last sub-stage to migrate to exit_status
        hook_to_exit = ConvergenceHook.from_fmax(0.05, source_status=2, target_status=3)
        dyn3.register_hook(hook_to_exit)

        result = dyn1 + dyn2 + dyn3

        assert isinstance(result, FusedStage)
        assert len(result.sub_stages) == 3

        data_list = [
            AtomicData(
                atomic_numbers=torch.tensor([6], dtype=torch.long),
                positions=torch.tensor([[float(i), 0.0, 0.0]]),
            )
            for i in range(4)
        ]
        batch = Batch.from_data_list(data_list)
        batch.forces = torch.zeros(batch.num_nodes, 3)
        batch.energies = torch.zeros(batch.num_graphs, 1)
        batch.status = torch.tensor([0, 0, 1, 1], dtype=torch.long)
        batch.fmax = torch.tensor([[0.01]] * 4)

        # run the dynamics - should terminate when all reach exit_status=3
        result.run(batch)

    def test_pipe_creates_pipeline(self) -> None:
        """``dyn1 | dyn2`` must return a ``DistributedPipeline``."""
        model = DemoModelWrapper()
        dyn1 = DemoDynamics(model=model, n_steps=1, dt=1.0)
        dyn2 = DemoDynamics(model=model, n_steps=1, dt=1.0)

        result = dyn1 | dyn2

        assert isinstance(result, DistributedPipeline)
        assert 0 in result.stages
        assert 1 in result.stages

    def test_fused_stage_aggregates_needs_keys(self) -> None:
        """FusedStage.__needs_keys__ is the union of sub-stage needs."""

        class StressNeedingDynamics(BaseDynamics):
            __needs_keys__: set[str] = {"forces", "stresses"}

        model = DemoModelWrapper()
        dyn0 = DemoDynamics(model=model, n_steps=1, dt=1.0)  # needs {"forces"}
        dyn1 = StressNeedingDynamics(model=model)  # needs {"forces", "stresses"}

        fused = FusedStage(sub_stages=[(0, dyn0), (1, dyn1)])

        assert fused.__needs_keys__ == {"forces", "stresses"}

    def test_fused_stage_aggregates_provides_keys(self) -> None:
        """FusedStage.__provides_keys__ is the union of sub-stage provides."""

        class StressProvidingDynamics(BaseDynamics):
            __provides_keys__: set[str] = {"stresses"}

        model = DemoModelWrapper()
        dyn0 = DemoDynamics(
            model=model, n_steps=1, dt=1.0
        )  # provides {"velocities", "positions"}
        dyn1 = StressProvidingDynamics(model=model)  # provides {"stresses"}

        fused = FusedStage(sub_stages=[(0, dyn0), (1, dyn1)])

        assert fused.__provides_keys__ == {"velocities", "positions", "stresses"}


# ---------------------------------------------------------------------------
# ConvergenceHook parameter
# ---------------------------------------------------------------------------


class TestConvergenceHookParam:
    """
    Test the ``convergence_hook`` parameter of ``DemoDynamics.__init__``.

    Verifies that ``DemoDynamics`` accepts ``ConvergenceHook | dict | None``
    and correctly converts dicts to ``ConvergenceHook`` instances.
    """

    def test_none_convergence_hook(self) -> None:
        """Verify that ``None`` convergence_hook is accepted (no convergence checking)."""
        model = DemoModelWrapper()
        dynamics = DemoDynamics(model=model, n_steps=1, dt=1.0, convergence_hook=None)

        assert dynamics.convergence_hook is None

    def test_convergence_hook_object_directly(self) -> None:
        """Verify that a ``ConvergenceHook`` object is passed through correctly."""
        model = DemoModelWrapper()
        hook = ConvergenceHook.from_fmax(0.05)
        dynamics = DemoDynamics(model=model, n_steps=1, dt=1.0, convergence_hook=hook)

        assert dynamics.convergence_hook is hook
        assert len(dynamics.convergence_hook.criteria) == 1
        assert dynamics.convergence_hook.criteria[0].key == "forces"
        assert dynamics.convergence_hook.criteria[0].threshold == 0.05

    def test_convergence_hook_from_dict(self) -> None:
        """Verify that a dict is converted to a ``ConvergenceHook``."""
        model = DemoModelWrapper()
        hook_dict = {"criteria": [{"key": "fmax", "threshold": 0.03}]}
        dynamics = DemoDynamics(
            model=model, n_steps=1, dt=1.0, convergence_hook=hook_dict
        )

        assert dynamics.convergence_hook is not None
        assert isinstance(dynamics.convergence_hook, ConvergenceHook)
        assert len(dynamics.convergence_hook.criteria) == 1
        assert dynamics.convergence_hook.criteria[0].key == "fmax"
        assert dynamics.convergence_hook.criteria[0].threshold == 0.03

    def test_convergence_hook_dict_with_multiple_criteria(self) -> None:
        """Verify that a dict with multiple criteria is converted correctly."""
        model = DemoModelWrapper()
        hook_dict = {
            "criteria": [
                {"key": "fmax", "threshold": 0.05},
                {"key": "energies", "threshold": 0.001, "reduce_op": "max"},
            ],
        }
        dynamics = DemoDynamics(
            model=model, n_steps=1, dt=1.0, convergence_hook=hook_dict
        )

        assert dynamics.convergence_hook is not None
        assert len(dynamics.convergence_hook.criteria) == 2
        assert dynamics.convergence_hook.criteria[0].key == "fmax"
        assert dynamics.convergence_hook.criteria[1].key == "energies"

    def test_convergence_hook_is_used(self) -> None:
        """Verify that the convergence_hook is actually used during simulation.

        Creates a dynamics with a very high fmax threshold (all samples
        should converge immediately) and verifies that the convergence
        hook identifies converged samples.
        """
        model = DemoModelWrapper()
        # Use a very high threshold so samples converge immediately
        hook = ConvergenceHook.from_fmax(threshold=1e6)
        dynamics = DemoDynamics(model=model, n_steps=1, dt=1.0, convergence_hook=hook)

        batch = _make_multi_batch(n_graphs=2, n_atoms_per_graph=3)
        # Set fmax to values below the threshold (convergence hook expects fmax)
        batch.fmax = torch.tensor([[0.01], [0.02]])

        # Run a step - this exercises the convergence hook internally
        dynamics.step(batch)

        # The convergence hook should identify all samples as converged
        # because the threshold is very high (1e6) and fmax values are low
        converged = dynamics.convergence_hook.evaluate(batch)
        assert converged is not None
        assert len(converged) == 2  # Both samples should be converged


# ---------------------------------------------------------------------------
# Model output and key validation
# ---------------------------------------------------------------------------


class TestDemoDynamicsValidation:
    """Test suite for DemoDynamics __needs_keys__ / __provides_keys__ validation."""

    def test_needs_keys_contains_forces(self) -> None:
        """Verify DemoDynamics declares forces in __needs_keys__."""
        assert "forces" in DemoDynamics.__needs_keys__

    def test_provides_keys(self) -> None:
        """Verify DemoDynamics declares velocities and positions in __provides_keys__."""
        assert DemoDynamics.__provides_keys__ == {"velocities", "positions"}

    def test_validate_outputs_missing_forces(self) -> None:
        """Verify _validate_model_outputs raises when forces are missing."""
        model = DemoModelWrapper()
        dynamics = DemoDynamics(model, n_steps=1, dt=0.5)
        outputs: ModelOutputs = OrderedDict()
        outputs["energies"] = torch.ones(1, 1)
        outputs["forces"] = None
        with pytest.raises(RuntimeError, match="requires 'forces'"):
            dynamics._validate_model_outputs(outputs)

    def test_validate_outputs_satisfied(self) -> None:
        """Verify DemoDynamics validation passes when forces are provided."""
        model = DemoModelWrapper()
        dynamics = DemoDynamics(model, n_steps=1, dt=0.5)
        outputs: ModelOutputs = OrderedDict()
        outputs["energies"] = torch.ones(1, 1)
        outputs["forces"] = torch.ones(5, 3)
        # Should not raise
        dynamics._validate_model_outputs(outputs)

    def test_validate_batch_keys_passes(self) -> None:
        """Verify _validate_batch_keys passes when provides keys are present."""
        model = DemoModelWrapper()
        dynamics = DemoDynamics(model, n_steps=1, dt=0.5)
        batch = _make_batch()
        batch.velocities = torch.zeros(batch.num_nodes, 3)
        # positions and velocities are present — should not raise
        dynamics._validate_batch_keys(batch)


class TestDemoDynamicsNSteps:
    """Test suite for n_steps attribute on DemoDynamics."""

    def test_n_steps_at_construction(self) -> None:
        """Verify DemoDynamics stores n_steps from __init__."""
        model = DemoModelWrapper()
        dynamics = DemoDynamics(model=model, n_steps=10, dt=1.0)
        assert dynamics.n_steps == 10

    def test_n_steps_is_required(self) -> None:
        """Verify DemoDynamics raises TypeError when n_steps is not provided."""
        model = DemoModelWrapper()
        with pytest.raises(TypeError):
            DemoDynamics(model=model, dt=1.0)

    def test_run_without_argument_uses_init_n_steps(self) -> None:
        """Verify run(batch) uses n_steps from __init__."""
        model = DemoModelWrapper()
        dynamics = DemoDynamics(model=model, n_steps=5, dt=1.0)
        batch = _make_batch()
        dynamics.run(batch)
        assert dynamics.step_count == 5

    def test_run_argument_overrides_init_n_steps(self) -> None:
        """Verify run(batch, 3) overrides n_steps=10 from __init__."""
        model = DemoModelWrapper()
        dynamics = DemoDynamics(model=model, n_steps=10, dt=1.0)
        batch = _make_batch()
        dynamics.run(batch, 3)
        assert dynamics.step_count == 3
