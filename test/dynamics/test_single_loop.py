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
Tests for fused stage execution strategy components.

Tests cover ConvergenceHook and FusedStage for the single-GPU execution
model where multiple dynamics engines share a batch.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import torch

from nvalchemi.data import AtomicData, Batch
from nvalchemi.dynamics.base import (
    BaseDynamics,
    BufferConfig,
    ConvergenceHook,
    DistributedPipeline,
    DynamicsStage,
    FusedStage,
    Hook,
    _CommunicationMixin,
)
from nvalchemi.dynamics.hooks import ConvergedSnapshotHook
from nvalchemi.dynamics.sinks import HostMemory
from nvalchemi.hooks._context import HookContext
from nvalchemi.models.base import BaseModelMixin, ModelCard
from nvalchemi.models.demo import DemoModelWrapper

# -----------------------------------------------------------------------------
# DemoModelWrapper Subclasses for Testing
# -----------------------------------------------------------------------------


class CountingDemoModel(DemoModelWrapper):
    """DemoModelWrapper that tracks forward pass calls."""

    # TODO: refactor this just to use register hook

    def __init__(self) -> None:
        super().__init__()
        self.forward_count = 0

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Increment counter and delegate to parent."""
        self.forward_count += 1
        return super().forward(*args, **kwargs)


class NonConservativeDemoModel(DemoModelWrapper):
    """DemoModelWrapper with forces_via_autograd=False."""

    @property
    def model_card(self) -> ModelCard:
        """Return a non-conservative model card."""
        return ModelCard(
            forces_via_autograd=False,
            supports_energies=True,
            supports_forces=True,
            supports_stresses=False,
            supports_hessians=False,
            supports_dipoles=False,
            supports_non_batch=True,
            neighbor_config=None,
            needs_pbc=False,
        )


class CountingNonConservativeDemoModel(NonConservativeDemoModel):
    """Non-conservative DemoModelWrapper that tracks forward calls."""

    def __init__(self) -> None:
        super().__init__()
        self.forward_count = 0

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Increment counter and delegate to parent."""
        self.forward_count += 1
        return super().forward(*args, **kwargs)


# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------


def create_batch_with_status(n_graphs: int = 3, device: str = "cpu") -> Batch:
    """Create a batch with N single-atom molecules for testing."""
    data_list = [
        AtomicData(
            atomic_numbers=torch.tensor([6], dtype=torch.long),
            positions=torch.randn(1, 3),
        )
        for _ in range(n_graphs)
    ]
    batch = Batch.from_data_list(data_list, device=device)
    batch.forces = torch.zeros(batch.num_nodes, 3)
    batch.energy = torch.zeros(batch.num_graphs, 1)
    return batch


# -----------------------------------------------------------------------------
# TrackingDynamics for FusedStage tests
# -----------------------------------------------------------------------------


class TrackingDynamics(BaseDynamics):
    """Dynamics subclass that tracks which samples were updated."""

    def __init__(self, model: BaseModelMixin) -> None:
        super().__init__(model=model)
        self.updated_masks: list[torch.Tensor] = []

    def _masked_pre_update(
        self,
        batch: Batch,
        mask: torch.Tensor,
    ) -> None:
        """Record the mask instead of performing actual updates."""
        self.updated_masks.append(mask.clone())


# -----------------------------------------------------------------------------
# TestConvergenceHook
# -----------------------------------------------------------------------------


class TestConvergenceHook:
    """Tests for the ConvergenceHook class."""

    def setup_method(self) -> None:
        """Set up test fixtures before each test method."""
        self.model = DemoModelWrapper()

    def test_satisfies_hook_protocol(self) -> None:
        """ConvergenceHook should satisfy the Hook protocol."""
        hook = ConvergenceHook()
        assert isinstance(hook, Hook)

    def test_default_attributes(self) -> None:
        """ConvergenceHook should have correct default attributes."""
        hook = ConvergenceHook()
        assert hook.frequency == 1
        assert hook.stage == DynamicsStage.AFTER_STEP
        # Default criteria uses forces with norm reduction and threshold 0.05
        assert len(hook.criteria) == 1
        assert hook.criteria[0].key == "forces"
        assert hook.criteria[0].threshold == 0.05
        assert hook.criteria[0].reduce_op == "norm"
        # Default status migration is disabled
        assert hook.source_status is None
        assert hook.target_status is None

    def test_migrates_converged_samples(self) -> None:
        """Hook should migrate all converged samples with matching status."""
        batch = create_batch_with_status(n_graphs=3)
        # All converged (force norms below 0.05), all with source_status=0
        batch.forces = torch.tensor([[0.01, 0, 0], [0.02, 0, 0], [0.03, 0, 0]])
        batch.status = torch.tensor([0, 0, 0])

        hook = ConvergenceHook.from_fmax(0.05, source_status=0, target_status=1)
        ctx = HookContext(batch=batch, step_count=0)
        hook(ctx, DynamicsStage.AFTER_STEP)

        # All should be migrated to status=1
        assert (batch.status == 1).all()

    def test_ignores_wrong_status(self) -> None:
        """Hook should ignore converged samples with wrong source_status."""
        batch = create_batch_with_status(n_graphs=3)
        # All converged (force norms below 0.05), but source_status=2 doesn't match
        batch.forces = torch.tensor([[0.01, 0, 0], [0.02, 0, 0], [0.03, 0, 0]])
        batch.status = torch.tensor([2, 2, 2])

        hook = ConvergenceHook.from_fmax(0.05, source_status=0, target_status=1)
        ctx = HookContext(batch=batch, step_count=0)
        hook(ctx, DynamicsStage.AFTER_STEP)

        # All should remain at status=2
        assert (batch.status == 2).all()

    def test_ignores_unconverged(self) -> None:
        """Hook should ignore samples that haven't converged."""
        batch = create_batch_with_status(n_graphs=3)
        # All have correct status, but force norms too high
        batch.forces = torch.tensor([[0.1, 0, 0], [0.2, 0, 0], [0.3, 0, 0]])
        batch.status = torch.tensor([0, 0, 0])

        hook = ConvergenceHook.from_fmax(0.05, source_status=0, target_status=1)
        ctx = HookContext(batch=batch, step_count=0)
        hook(ctx, DynamicsStage.AFTER_STEP)

        # All should remain at status=0
        assert (batch.status == 0).all()

    def test_mixed_batch(self) -> None:
        """Hook should correctly handle mixed convergence and status."""
        batch = create_batch_with_status(n_graphs=4)
        # Sample 0: converged AND correct status -> migrate
        # Sample 1: converged but wrong status -> no migrate
        # Sample 2: not converged but correct status -> no migrate
        # Sample 3: not converged and wrong status -> no migrate
        batch.forces = torch.tensor(
            [[0.01, 0, 0], [0.01, 0, 0], [0.1, 0, 0], [0.1, 0, 0]]
        )
        batch.status = torch.tensor([0, 2, 0, 2])

        hook = ConvergenceHook.from_fmax(0.05, source_status=0, target_status=1)
        ctx = HookContext(batch=batch, step_count=0)
        hook(ctx, DynamicsStage.AFTER_STEP)

        expected = torch.tensor([1, 2, 0, 2])
        assert torch.equal(batch.status, expected)

    def test_custom_threshold(self) -> None:
        """Hook should use custom threshold correctly."""
        batch = create_batch_with_status(n_graphs=3)
        batch.forces = torch.tensor([[0.01, 0, 0], [0.05, 0, 0], [0.1, 0, 0]])
        batch.status = torch.tensor([0, 0, 0])

        # Use higher threshold - all should converge
        hook = ConvergenceHook.from_fmax(0.2, source_status=0, target_status=1)
        ctx = HookContext(batch=batch, step_count=0)
        hook(ctx, DynamicsStage.AFTER_STEP)

        assert (batch.status == 1).all()

    def test_no_forces_raises_key_error(self) -> None:
        """Hook should raise KeyError if batch has no forces attribute."""
        batch = create_batch_with_status(n_graphs=3)
        batch.status = torch.tensor([0, 0, 0])
        batch.forces = None  # Clear forces

        hook = ConvergenceHook()
        ctx = HookContext(batch=batch, step_count=0)

        with pytest.raises(KeyError, match="forces"):
            hook(ctx, DynamicsStage.AFTER_STEP)

    def test_no_status_is_noop(self) -> None:
        """Hook should be a no-op if batch has no status attribute."""
        batch = create_batch_with_status(n_graphs=3)
        batch.forces = torch.tensor([[0.01, 0, 0], [0.02, 0, 0], [0.03, 0, 0]])
        # No status attribute - delete it if it exists
        if hasattr(batch, "status"):
            delattr(batch, "status")

        hook = ConvergenceHook()
        ctx = HookContext(batch=batch, step_count=0)

        # Should not raise, just no-op
        hook(ctx, DynamicsStage.AFTER_STEP)

    def test_1d_status_tensor(self) -> None:
        """Hook should handle 1D status tensor (shape (B,))."""
        batch = create_batch_with_status(n_graphs=3)
        batch.forces = torch.tensor([[0.01, 0, 0], [0.02, 0, 0], [0.03, 0, 0]])
        batch.status = torch.tensor([0, 0, 0])  # 1D tensor

        hook = ConvergenceHook.from_fmax(0.05, source_status=0, target_status=1)
        ctx = HookContext(batch=batch, step_count=0)
        hook(ctx, DynamicsStage.AFTER_STEP)

        assert (batch.status == 1).all()

    def test_2d_status_tensor(self) -> None:
        """Hook should handle 2D status tensor (shape (B, 1))."""
        batch = create_batch_with_status(n_graphs=3)
        batch.forces = torch.tensor([[0.01, 0, 0], [0.02, 0, 0], [0.03, 0, 0]])
        batch.status = torch.tensor([[0], [0], [0]])  # (B, 1)

        hook = ConvergenceHook.from_fmax(0.05, source_status=0, target_status=1)
        ctx = HookContext(batch=batch, step_count=0)
        hook(ctx, DynamicsStage.AFTER_STEP)

        # Should have updated all to 1 (in-place on the original tensor)
        assert (batch.status.view(-1) == 1).all()


# -----------------------------------------------------------------------------
# TestFusedStage
# -----------------------------------------------------------------------------


class TestFusedStage:
    """Tests for the FusedStage class."""

    def setup_method(self) -> None:
        """Set up test fixtures before each test method.

        Uses NonConservativeDemoModel because tests that call compute() followed
        by masked_update() fail with conservative models (compute enables grads
        on positions, which then conflicts with masked_update's in-place ops).
        """
        self.model = NonConservativeDemoModel()

    def test_plus_operator_creates_fused_stage(self) -> None:
        """BaseDynamics + BaseDynamics should create FusedStage."""
        dyn1 = BaseDynamics(model=self.model)
        dyn2 = BaseDynamics(model=self.model)

        fused = dyn1 + dyn2

        assert isinstance(fused, FusedStage)
        assert len(fused.sub_stages) == 2

    def test_auto_assigned_status_codes(self) -> None:
        """Sub-stages should have auto-assigned status codes 0, 1, ..."""
        dyn1 = BaseDynamics(model=self.model)
        dyn2 = BaseDynamics(model=self.model)

        fused = dyn1 + dyn2

        assert fused.sub_stages[0][0] == 0
        assert fused.sub_stages[1][0] == 1

    def test_exit_status_auto_set(self) -> None:
        """exit_status should auto-set to len(sub_stages)."""
        dynamics1 = BaseDynamics(model=self.model)
        dynamics2 = BaseDynamics(model=self.model)

        fused = FusedStage(sub_stages=[(0, dynamics1), (1, dynamics2)])

        assert fused.exit_status == 2

    def test_single_forward_pass(self) -> None:
        """FusedStage should only call model forward once per step."""
        model = CountingNonConservativeDemoModel()
        dynamics1 = TrackingDynamics(model=model)
        dynamics2 = TrackingDynamics(model=model)

        fused = FusedStage(sub_stages=[(0, dynamics1), (1, dynamics2)])

        batch = create_batch_with_status(n_graphs=4)
        batch.status = torch.tensor([0, 0, 1, 1])
        batch.fmax = torch.tensor([0.1, 0.1, 0.1, 0.1])

        fused.step(batch)

        assert model.forward_count == 1

    def test_masked_updates_correct_indices(self) -> None:
        """FusedStage should dispatch updates to correct dynamics engines."""
        dynamics0 = TrackingDynamics(model=self.model)
        dynamics1 = TrackingDynamics(model=self.model)

        fused = FusedStage(sub_stages=[(0, dynamics0), (1, dynamics1)])

        batch = create_batch_with_status(n_graphs=4)
        batch.status = torch.tensor([0, 1, 0, 1])
        batch.fmax = torch.tensor([0.1, 0.1, 0.1, 0.1])

        fused.step(batch)

        # dynamics0 should get mask for indices 0, 2
        assert len(dynamics0.updated_masks) == 1
        expected_mask0 = torch.tensor([True, False, True, False])
        assert torch.equal(dynamics0.updated_masks[0], expected_mask0)

        # dynamics1 should get mask for indices 1, 3
        assert len(dynamics1.updated_masks) == 1
        expected_mask1 = torch.tensor([False, True, False, True])
        assert torch.equal(dynamics1.updated_masks[0], expected_mask1)

    def test_convergence_migration_auto_registered(self) -> None:
        """ConvergenceHook should be auto-registered between adjacent sub-stages."""
        # Use a high threshold so DemoModel forces always trigger convergence
        dynamics0 = BaseDynamics(
            model=self.model,
            convergence_hook=ConvergenceHook.from_fmax(1e6),
        )
        dynamics1 = BaseDynamics(model=self.model)

        fused = FusedStage(sub_stages=[(0, dynamics0), (1, dynamics1)])

        batch = create_batch_with_status(n_graphs=3)
        batch.status = torch.tensor([0, 0, 0])

        fused.step(batch)

        # All should have migrated to status=1 via auto-registered ConvergenceHook
        assert (batch.status == 1).all()

    def test_single_stage_auto_registers_migration_hook(self) -> None:
        """Single-stage FusedStage with convergence_hook auto-registers 0 -> exit_status."""
        dynamics0 = BaseDynamics(
            model=self.model,
            convergence_hook=ConvergenceHook.from_fmax(1e6),
        )

        fused = FusedStage(sub_stages=[(0, dynamics0)])

        batch = create_batch_with_status(n_graphs=3)
        batch.status = torch.tensor([0, 0, 0])

        fused.step(batch)

        assert (batch.status == fused.exit_status).all()

    def test_single_stage_no_convergence_hook_skips_auto_registration(self) -> None:
        """Single-stage FusedStage without convergence_hook skips migration hook."""
        dynamics0 = BaseDynamics(model=self.model)

        conv_hooks_before = [
            h
            for h in dynamics0.hooks
            if isinstance(h, ConvergenceHook) and h.source_status is not None
        ]

        FusedStage(sub_stages=[(0, dynamics0)])

        conv_hooks_after = [
            h
            for h in dynamics0.hooks
            if isinstance(h, ConvergenceHook) and h.source_status is not None
        ]
        assert len(conv_hooks_after) == len(conv_hooks_before)

    def test_single_stage_converged_snapshot_no_overflow(self) -> None:
        """ConvergedSnapshotHook must not overflow when converged samples graduate."""
        n_graphs = 3
        sink = HostMemory(capacity=n_graphs)
        dynamics0 = BaseDynamics(
            model=self.model,
            convergence_hook=ConvergenceHook.from_fmax(1e6),
            hooks=[ConvergedSnapshotHook(sink=sink)],
        )

        fused = FusedStage(sub_stages=[(0, dynamics0)])

        batch = create_batch_with_status(n_graphs=n_graphs)
        batch.status = torch.tensor([0, 0, 0])

        for _ in range(3):
            fused.step(batch)

        assert len(sink) == n_graphs

    def test_all_complete_true(self) -> None:
        """all_complete should return True when all samples at exit status."""
        batch = create_batch_with_status(n_graphs=3)
        batch.status = torch.tensor([2, 2, 2])

        result = FusedStage.all_complete(batch, exit_status=2)
        assert result is True

    def test_all_complete_false(self) -> None:
        """all_complete should return False when not all samples at exit."""
        batch = create_batch_with_status(n_graphs=3)
        batch.status = torch.tensor([2, 1, 2])

        result = FusedStage.all_complete(batch, exit_status=2)
        assert result is False

    def test_run_stops_on_convergence(self) -> None:
        """run() should stop when all samples reach exit_status."""
        model = CountingNonConservativeDemoModel()
        dynamics = BaseDynamics(model=model)
        # Register convergence hook: migrate 0 -> 1 with high threshold
        # so DemoModel forces always trigger convergence
        hook = ConvergenceHook.from_fmax(1e6, source_status=0, target_status=1)
        dynamics.register_hook(hook)

        fused = FusedStage(sub_stages=[(0, dynamics)])

        batch = create_batch_with_status(n_graphs=3)
        batch.status = torch.tensor([0, 0, 0])

        fused.run(batch)

        assert fused.step_count == 1
        # FusedStage.run() does one initial force-priming forward before the loop,
        # plus one during the actual step → total 2 forwards.
        assert model.forward_count == 2

    def test_run_stops_early_on_exit_status(self) -> None:
        """run() should stop early when all reach exit status."""
        dynamics0 = BaseDynamics(model=self.model)

        # Create single-stage fused, auto-registered hook migrates 0 -> 1
        fused = FusedStage(sub_stages=[(0, dynamics0)])
        # Manually register hook with high threshold so forces always converge
        hook = ConvergenceHook.from_fmax(1e6, source_status=0, target_status=1)
        dynamics0.register_hook(hook)

        batch = create_batch_with_status(n_graphs=3)
        batch.status = torch.tensor([0, 0, 0])

        # exit_status is 1 (len(sub_stages) == 1)
        fused.run(batch)

        # Should stop after 1 step since all migrate to exit_status=1
        assert fused.step_count == 1

    def test_step_increments_count(self) -> None:
        """step() should increment step_count."""
        dynamics = BaseDynamics(model=self.model)

        fused = FusedStage(sub_stages=[(0, dynamics)])

        batch = create_batch_with_status(n_graphs=2)
        batch.status = torch.tensor([0, 0])
        batch.fmax = torch.tensor([0.5, 0.5])

        assert fused.step_count == 0

        fused.step(batch)
        assert fused.step_count == 1

        fused.step(batch)
        assert fused.step_count == 2

    def test_compile_step_creates_compiled_callable(self) -> None:
        """compile_step=True should replace step with compiled callable."""
        dynamics = BaseDynamics(model=self.model)

        fused = FusedStage(sub_stages=[(0, dynamics)], compile_step=True)

        # Verify that _compiled_step is set (compiled function)
        assert fused._compiled_step is not None
        assert callable(fused._compiled_step)

    def test_fused_stage_or_produces_pipeline(self) -> None:
        """FusedStage | BaseDynamics should produce DistributedPipeline."""
        dynamics1 = BaseDynamics(model=self.model)
        dynamics2 = BaseDynamics(model=self.model)

        fused = FusedStage(sub_stages=[(0, dynamics1)])

        pipeline = fused | dynamics2

        assert isinstance(pipeline, DistributedPipeline)
        assert len(pipeline.stages) == 2

    def test_fused_stage_plus_appends_sub_stage(self) -> None:
        """FusedStage + BaseDynamics should append sub-stage."""
        dynamics1 = BaseDynamics(model=self.model)
        dynamics2 = BaseDynamics(model=self.model)

        fused = FusedStage(sub_stages=[(0, dynamics1)])

        fused2 = fused + dynamics2

        assert isinstance(fused2, FusedStage)
        assert len(fused2.sub_stages) == 2
        assert fused2.sub_stages[0][0] == 0
        assert fused2.sub_stages[1][0] == 1

    def test_add_raises_when_self_not_basedynamics(self) -> None:
        """__add__ raises TypeError when self is not BaseDynamics.

        Uses a standalone _CommunicationMixin instance to verify the type
        guard on __add__. The mixin is NOT a BaseDynamics instance.
        """
        mixin = _CommunicationMixin()  # Not a BaseDynamics instance
        dynamics = BaseDynamics(model=self.model)

        with pytest.raises(
            TypeError, match="Both operands of \\+ must be BaseDynamics"
        ):
            mixin + dynamics

    def test_add_raises_when_other_not_basedynamics(self) -> None:
        """__add__ raises TypeError when other is not BaseDynamics.

        Uses a standalone _CommunicationMixin instance to verify the type
        guard on __add__. The mixin is NOT a BaseDynamics instance.
        """
        dynamics = BaseDynamics(model=self.model)
        mixin = _CommunicationMixin()  # Not a BaseDynamics instance

        with pytest.raises(
            TypeError, match="Both operands of \\+ must be BaseDynamics"
        ):
            dynamics + mixin

    def test_fused_add_raises_when_other_not_basedynamics(self) -> None:
        """FusedStage.__add__ raises TypeError when other is not BaseDynamics.

        Uses a standalone _CommunicationMixin instance to verify the type
        guard on FusedStage.__add__. The mixin is NOT a BaseDynamics instance.
        """
        dynamics1 = BaseDynamics(model=self.model)
        fused = FusedStage(sub_stages=[(0, dynamics1)])
        mixin = _CommunicationMixin()  # Not a BaseDynamics instance

        with pytest.raises(TypeError, match="other must be a BaseDynamics instance"):
            fused + mixin

    def test_empty_status_mask_skips_update(self) -> None:
        """Dynamics should not be called if no samples match its status."""
        dynamics0 = TrackingDynamics(model=self.model)
        dynamics1 = TrackingDynamics(model=self.model)

        fused = FusedStage(sub_stages=[(0, dynamics0), (1, dynamics1)])

        batch = create_batch_with_status(n_graphs=3)
        # All samples have status=1, none have status=0
        batch.status = torch.tensor([1, 1, 1])
        batch.fmax = torch.tensor([0.1, 0.1, 0.1])

        fused.step(batch)

        # dynamics0 should not have been called (no samples with status=0)
        assert len(dynamics0.updated_masks) == 0

        # dynamics1 should have been called
        assert len(dynamics1.updated_masks) == 1

    def test_three_stage_fusion(self) -> None:
        """FusedStage should support three or more sub-stages."""
        dyn0 = TrackingDynamics(model=self.model)
        dyn1 = TrackingDynamics(model=self.model)
        dyn2 = TrackingDynamics(model=self.model)

        fused = dyn0 + dyn1 + dyn2

        assert isinstance(fused, FusedStage)
        assert len(fused.sub_stages) == 3
        assert fused.sub_stages[0][0] == 0
        assert fused.sub_stages[1][0] == 1
        assert fused.sub_stages[2][0] == 2
        assert fused.exit_status == 3

    def test_convergence_hooks_chain_through_stages(self) -> None:
        """Samples should migrate through stages via auto-registered hooks.

        When all samples are converged, they migrate through all stages
        within a single step since hooks fire sequentially in AFTER_STEP.
        With 3 sub-stages (0, 1, 2), hooks 0->1 and 1->2 are auto-registered.
        """
        # Use high-threshold convergence hooks so DemoModel forces always converge
        dynamics0 = BaseDynamics(
            model=self.model,
            convergence_hook=ConvergenceHook.from_fmax(1e6),
        )
        dynamics1 = BaseDynamics(
            model=self.model,
            convergence_hook=ConvergenceHook.from_fmax(1e6),
        )
        dynamics2 = BaseDynamics(model=self.model)

        # Create 3-stage fused: auto-hooks 0->1 and 1->2 are registered
        fused = FusedStage(sub_stages=[(0, dynamics0), (1, dynamics1), (2, dynamics2)])

        batch = create_batch_with_status(n_graphs=3)
        batch.status = torch.tensor([0, 0, 0])

        # Hooks fire sequentially in AFTER_STEP, so converged samples
        # cascade through all stages within a single step: 0 -> 1 -> 2
        # Note: No hook for 2->3 (exit_status) since 3 is not a sub-stage
        fused.step(batch)
        assert (batch.status == 2).all()

    def test_compile_method_creates_compiled_callable(self) -> None:
        """Calling .compile() on an uncompiled FusedStage sets _compiled_step."""
        dynamics = BaseDynamics(model=self.model)
        fused = FusedStage(sub_stages=[(0, dynamics)])

        assert fused._compiled_step is None

        result = fused.compile()

        assert fused._compiled_step is not None
        assert callable(fused._compiled_step)
        assert result is fused  # fluent API

    def test_compile_method_merges_kwargs(self) -> None:
        """compile() merges kwargs with init compile_kwargs."""
        dynamics = BaseDynamics(model=self.model)
        fused = FusedStage(
            sub_stages=[(0, dynamics)],
            compile_kwargs={"fullgraph": False, "dynamic": True},
        )

        with patch("torch.compile", return_value=lambda b: b) as mock_compile:
            fused.compile(fullgraph=True)

        # fullgraph=True from compile() should override fullgraph=False from init
        mock_compile.assert_called_once_with(
            fused._step_impl, fullgraph=True, dynamic=True
        )
        assert fused.compile_kwargs == {"fullgraph": True, "dynamic": True}

    def test_compile_method_sets_compile_step_flag(self) -> None:
        """Calling .compile() sets compile_step = True."""
        dynamics = BaseDynamics(model=self.model)
        fused = FusedStage(sub_stages=[(0, dynamics)])

        assert fused.compile_step is False

        fused.compile()

        assert fused.compile_step is True

    def test_add_defers_compilation(self) -> None:
        """FusedStage + dyn should defer compilation, preserving intent."""
        dynamics1 = BaseDynamics(model=self.model)
        dynamics2 = BaseDynamics(model=self.model)
        dynamics3 = BaseDynamics(model=self.model)

        fused = FusedStage(sub_stages=[(0, dynamics1)], compile_step=True)
        assert fused._compiled_step is not None  # compiled in __init__

        fused2 = fused + dynamics2
        # Compilation deferred: intent preserved but not executed
        assert fused2.compile_step is True
        assert fused2._compiled_step is None

        fused3 = fused2 + dynamics3
        # Still deferred through chaining
        assert fused3.compile_step is True
        assert fused3._compiled_step is None

    def test_enter_triggers_lazy_compile(self) -> None:
        """Entering context on a deferred-compile stage triggers compilation."""
        dynamics1 = BaseDynamics(model=self.model)
        dynamics2 = BaseDynamics(model=self.model)

        fused = FusedStage(sub_stages=[(0, dynamics1)], compile_step=True)
        fused2 = fused + dynamics2
        assert fused2._compiled_step is None

        # Use CPU so we don't need actual CUDA
        fused2_cpu = FusedStage(
            sub_stages=[
                (0, BaseDynamics(model=self.model, device_type="cpu")),
                (1, BaseDynamics(model=self.model, device_type="cpu")),
            ],
            device_type="cpu",
            compile_step=False,
        )
        fused2_cpu.compile_step = True  # simulate deferred intent

        with fused2_cpu:
            assert fused2_cpu._compiled_step is not None
            assert callable(fused2_cpu._compiled_step)

    def test_enter_no_double_compile(self) -> None:
        """Entering context when already compiled does not recompile."""
        dynamics = BaseDynamics(model=self.model, device_type="cpu")
        fused = FusedStage(
            sub_stages=[(0, dynamics)],
            compile_step=True,
            device_type="cpu",
        )
        first_compiled = fused._compiled_step
        assert first_compiled is not None

        with fused:
            # Should be the same object, not recompiled
            assert fused._compiled_step is first_compiled

    def test_add_chain_defers_all_intermediates(self) -> None:
        """dyn0 + dyn1 + dyn2 with compile intent defers to final stage."""
        dyn0 = BaseDynamics(model=self.model)
        dyn1 = BaseDynamics(model=self.model)
        dyn2 = BaseDynamics(model=self.model)

        # BaseDynamics.__add__ doesn't set compile_step, so start with FusedStage
        fused_compiled = FusedStage(sub_stages=[(0, dyn0)], compile_step=True)
        fused2 = fused_compiled + dyn1
        fused3 = fused2 + dyn2

        # All intermediates should have deferred compilation
        assert fused3.compile_step is True
        assert fused3._compiled_step is None
        assert len(fused3.sub_stages) == 3

        # Explicit compile triggers it
        fused3.compile()
        assert fused3._compiled_step is not None

    def test_fused_stage_inherits_n_steps(self) -> None:
        """Verify FusedStage stores n_steps inherited from BaseDynamics."""
        dynamics = BaseDynamics(model=self.model)
        fused = FusedStage(sub_stages=[(0, dynamics)], n_steps=100)
        assert fused.n_steps == 100

    def test_fused_stage_run_ignores_n_steps(self) -> None:
        """Verify FusedStage.run() terminates by convergence, not n_steps."""
        model = CountingNonConservativeDemoModel()
        dynamics = BaseDynamics(model=model)
        # High threshold so DemoModel forces always trigger convergence
        hook = ConvergenceHook.from_fmax(1e6, source_status=0, target_status=1)
        dynamics.register_hook(hook)

        # Set a large n_steps — should be ignored, convergence stops it early
        fused = FusedStage(sub_stages=[(0, dynamics)], n_steps=10000)

        batch = create_batch_with_status(n_graphs=3)
        batch.status = torch.tensor([0, 0, 0])

        fused.run(batch)

        # Should stop after 1 step (convergence), NOT 10000
        assert fused.step_count == 1


# -----------------------------------------------------------------------------
# TestFusedStageDeviceValidation
# -----------------------------------------------------------------------------


class TestFusedStageDeviceValidation:
    """Tests for FusedStage device_type consistency validation.

    A FusedStage runs all sub-stages on a single device with a shared
    batch and forward pass. Mixing device types is invalid and must
    raise ``ValueError`` at construction time.
    """

    def setup_method(self) -> None:
        """Set up test fixtures before each test method."""
        self.model = NonConservativeDemoModel()

    def test_same_device_type_allowed(self) -> None:
        """FusedStage should accept sub-stages with the same device_type."""
        dyn0 = BaseDynamics(model=self.model, device_type="cpu")
        dyn1 = BaseDynamics(model=self.model, device_type="cpu")

        fused = FusedStage(sub_stages=[(0, dyn0), (1, dyn1)])
        assert fused.sub_stages[0][1].device_type == "cpu"
        assert fused.sub_stages[1][1].device_type == "cpu"

    def test_mixed_device_types_raises(self) -> None:
        """FusedStage should reject sub-stages with different device_types."""
        dyn_cpu = BaseDynamics(model=self.model, device_type="cpu")
        dyn_cuda = BaseDynamics(model=self.model, device_type="cuda")

        with pytest.raises(ValueError, match="same device_type"):
            FusedStage(sub_stages=[(0, dyn_cpu), (1, dyn_cuda)])

    def test_mixed_device_types_error_message_includes_details(self) -> None:
        """Error message should include per-stage device_type mapping."""
        dyn_cpu = BaseDynamics(model=self.model, device_type="cpu")
        dyn_cuda = BaseDynamics(model=self.model, device_type="cuda")

        with pytest.raises(ValueError, match=r"\{0: 'cpu', 1: 'cuda'\}"):
            FusedStage(sub_stages=[(0, dyn_cpu), (1, dyn_cuda)])

    def test_three_stages_same_device_type(self) -> None:
        """FusedStage with three sub-stages of same device_type should work."""
        dyn0 = BaseDynamics(model=self.model, device_type="cuda")
        dyn1 = BaseDynamics(model=self.model, device_type="cuda")
        dyn2 = BaseDynamics(model=self.model, device_type="cuda")

        fused = FusedStage(sub_stages=[(0, dyn0), (1, dyn1), (2, dyn2)])
        assert len(fused.sub_stages) == 3

    def test_three_stages_one_different_raises(self) -> None:
        """FusedStage should reject if any one sub-stage has a different device_type."""
        dyn0 = BaseDynamics(model=self.model, device_type="cuda")
        dyn1 = BaseDynamics(model=self.model, device_type="cpu")
        dyn2 = BaseDynamics(model=self.model, device_type="cuda")

        with pytest.raises(ValueError, match="same device_type"):
            FusedStage(sub_stages=[(0, dyn0), (1, dyn1), (2, dyn2)])

    def test_plus_operator_mixed_device_types_raises(self) -> None:
        """dyn_a + dyn_b should raise if device_types differ."""
        dyn_cpu = BaseDynamics(model=self.model, device_type="cpu")
        dyn_cuda = BaseDynamics(model=self.model, device_type="cuda")

        with pytest.raises(ValueError, match="same device_type"):
            dyn_cpu + dyn_cuda

    def test_fused_add_mixed_device_types_raises(self) -> None:
        """fused + dyn should raise if the new dyn has a different device_type."""
        dyn0 = BaseDynamics(model=self.model, device_type="cpu")
        dyn1 = BaseDynamics(model=self.model, device_type="cpu")
        fused = FusedStage(sub_stages=[(0, dyn0), (1, dyn1)])

        dyn_cuda = BaseDynamics(model=self.model, device_type="cuda")
        with pytest.raises(ValueError, match="same device_type"):
            fused + dyn_cuda

    def test_default_device_type_is_cuda(self) -> None:
        """BaseDynamics default device_type should be 'cuda'."""
        dyn = BaseDynamics(model=self.model)
        assert dyn.device_type == "cuda"


# -----------------------------------------------------------------------------
# TestCommunicationMixinStreamContext
# -----------------------------------------------------------------------------


class TestCommunicationMixinStreamContext:
    """Tests for _CommunicationMixin.__enter__ / __exit__ stream context."""

    def setup_method(self) -> None:
        """Set up test fixtures before each test method."""
        self.model = DemoModelWrapper()

    def test_enter_creates_stream_on_cuda(self) -> None:
        """__enter__ creates a CUDA stream and enters a StreamContext on CUDA devices."""
        mock_stream = MagicMock(spec=torch.cuda.Stream)
        mock_stream_ctx = MagicMock()

        with patch("torch.cuda.is_available", return_value=True):
            with patch(
                "torch.cuda.Stream", return_value=mock_stream
            ) as mock_stream_cls:
                with patch(
                    "torch.cuda.stream", return_value=mock_stream_ctx
                ) as mock_stream_fn:
                    # Create dynamics with device_type="cuda"
                    dyn = BaseDynamics(model=self.model, device_type="cuda")

                    # Mock device property to avoid actual CUDA device creation
                    with patch.object(
                        type(dyn), "device", property(lambda s: torch.device("cuda:0"))
                    ):
                        result = dyn.__enter__()

                        # Assert Stream was created with the device
                        mock_stream_cls.assert_called_once_with(
                            device=torch.device("cuda:0")
                        )

                        # Assert torch.cuda.stream() was called with the stream
                        mock_stream_fn.assert_called_once_with(mock_stream)

                        # Assert the context was entered
                        mock_stream_ctx.__enter__.assert_called_once()

                        # Assert _stream is the mock stream
                        assert dyn._stream is mock_stream

                        # Assert _stream_ctx is the mock context
                        assert dyn._stream_ctx is mock_stream_ctx

                        # Assert returns self
                        assert result is dyn

    def test_enter_noop_on_cpu(self) -> None:
        """__enter__ is a no-op on CPU devices — _stream remains None."""
        dyn = BaseDynamics(model=self.model, device_type="cpu")

        # Verify initial state
        assert dyn._stream is None
        assert dyn._stream_ctx is None

        result = dyn.__enter__()

        # Stream should remain None on CPU
        assert dyn._stream is None
        assert dyn._stream_ctx is None

        # Should return self
        assert result is dyn

    def test_exit_clears_stream(self) -> None:
        """__exit__ exits the StreamContext and clears stream references."""
        mock_stream = MagicMock(spec=torch.cuda.Stream)
        mock_stream_ctx = MagicMock()

        with patch("torch.cuda.is_available", return_value=True):
            with patch("torch.cuda.Stream", return_value=mock_stream):
                with patch("torch.cuda.stream", return_value=mock_stream_ctx):
                    dyn = BaseDynamics(model=self.model, device_type="cuda")

                    with patch.object(
                        type(dyn), "device", property(lambda s: torch.device("cuda:0"))
                    ):
                        # Enter the context
                        dyn.__enter__()

                        # Verify stream is set
                        assert dyn._stream is mock_stream
                        assert dyn._stream_ctx is mock_stream_ctx

                        # Exit the context
                        dyn.__exit__(None, None, None)

                        # Assert __exit__ was called on the stream context
                        mock_stream_ctx.__exit__.assert_called_once_with(
                            None, None, None
                        )

                        # Assert stream and context are cleared
                        assert dyn._stream is None
                        assert dyn._stream_ctx is None

    def test_stream_property_returns_active_stream(self) -> None:
        """The stream property returns the active stream inside context, None outside."""
        mock_stream = MagicMock(spec=torch.cuda.Stream)
        mock_stream_ctx = MagicMock()

        with patch("torch.cuda.is_available", return_value=True):
            with patch("torch.cuda.Stream", return_value=mock_stream):
                with patch("torch.cuda.stream", return_value=mock_stream_ctx):
                    dyn = BaseDynamics(model=self.model, device_type="cuda")

                    with patch.object(
                        type(dyn), "device", property(lambda s: torch.device("cuda:0"))
                    ):
                        # Before __enter__, stream is None
                        assert dyn.stream is None

                        # After __enter__, stream is the mock stream
                        dyn.__enter__()
                        assert dyn.stream is mock_stream

                        # After __exit__, stream is None again
                        dyn.__exit__(None, None, None)
                        assert dyn.stream is None

    def test_context_manager_protocol(self) -> None:
        """'with dynamics_instance:' works end-to-end."""
        mock_stream = MagicMock(spec=torch.cuda.Stream)
        mock_stream_ctx = MagicMock()

        with patch("torch.cuda.is_available", return_value=True):
            with patch("torch.cuda.Stream", return_value=mock_stream):
                with patch("torch.cuda.stream", return_value=mock_stream_ctx):
                    dyn = BaseDynamics(model=self.model, device_type="cuda")

                    with patch.object(
                        type(dyn), "device", property(lambda s: torch.device("cuda:0"))
                    ):
                        # Use the context manager protocol
                        with dyn:
                            # Inside the with block, stream should be active
                            assert dyn.stream is mock_stream
                            assert dyn._stream_ctx is mock_stream_ctx

                        # After exiting, stream should be cleared
                        assert dyn.stream is None
                        assert dyn._stream_ctx is None

                        # Verify __enter__ and __exit__ were called on the stream context
                        mock_stream_ctx.__enter__.assert_called_once()
                        mock_stream_ctx.__exit__.assert_called_once()


class TestFusedStageStreamContext:
    """Tests for FusedStage.__enter__ / __exit__ stream propagation to sub-stages."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.model = NonConservativeDemoModel()

    def test_fused_enter_propagates_stream_to_substages(self) -> None:
        """__enter__ propagates the CUDA stream to all sub-stage dynamics."""
        # Create two dynamics and a FusedStage
        dyn0 = BaseDynamics(model=self.model, device_type="cuda")
        dyn1 = BaseDynamics(model=self.model, device_type="cuda")
        fused = FusedStage(sub_stages=[(0, dyn0), (1, dyn1)])

        mock_stream = MagicMock(spec=torch.cuda.Stream)
        mock_stream_ctx = MagicMock()

        with patch("torch.cuda.is_available", return_value=True):
            with patch("torch.cuda.Stream", return_value=mock_stream):
                with patch("torch.cuda.stream", return_value=mock_stream_ctx):
                    with patch.object(
                        type(fused),
                        "device",
                        property(lambda s: torch.device("cuda:0")),
                    ):
                        fused.__enter__()

                        # All sub-stages should have the same stream reference
                        assert dyn0._stream is mock_stream
                        assert dyn1._stream is mock_stream
                        # FusedStage itself should also have the stream
                        assert fused._stream is mock_stream

    def test_fused_exit_clears_substage_streams(self) -> None:
        """__exit__ clears _stream on all sub-stages."""
        dyn0 = BaseDynamics(model=self.model, device_type="cuda")
        dyn1 = BaseDynamics(model=self.model, device_type="cuda")
        fused = FusedStage(sub_stages=[(0, dyn0), (1, dyn1)])

        mock_stream = MagicMock(spec=torch.cuda.Stream)
        mock_stream_ctx = MagicMock()

        with patch("torch.cuda.is_available", return_value=True):
            with patch("torch.cuda.Stream", return_value=mock_stream):
                with patch("torch.cuda.stream", return_value=mock_stream_ctx):
                    with patch.object(
                        type(fused),
                        "device",
                        property(lambda s: torch.device("cuda:0")),
                    ):
                        fused.__enter__()
                        fused.__exit__(None, None, None)

                        # All sub-stage streams should be cleared
                        assert dyn0._stream is None
                        assert dyn1._stream is None
                        # FusedStage's own stream should be cleared too
                        assert fused._stream is None
                        assert fused._stream_ctx is None

    def test_fused_context_manager_end_to_end(self) -> None:
        """'with fused:' propagates and cleans up streams end-to-end."""
        dyn0 = BaseDynamics(model=self.model, device_type="cuda")
        dyn1 = BaseDynamics(model=self.model, device_type="cuda")
        fused = FusedStage(sub_stages=[(0, dyn0), (1, dyn1)])

        mock_stream = MagicMock(spec=torch.cuda.Stream)
        mock_stream_ctx = MagicMock()

        with patch("torch.cuda.is_available", return_value=True):
            with patch("torch.cuda.Stream", return_value=mock_stream):
                with patch("torch.cuda.stream", return_value=mock_stream_ctx):
                    with patch.object(
                        type(fused),
                        "device",
                        property(lambda s: torch.device("cuda:0")),
                    ):
                        with fused:
                            # Inside: all sub-stages share the stream
                            assert fused.stream is mock_stream
                            assert dyn0._stream is mock_stream
                            assert dyn1._stream is mock_stream

                        # Outside: everything is cleaned up
                        assert fused.stream is None
                        assert dyn0._stream is None
                        assert dyn1._stream is None

    def test_fused_cpu_noop(self) -> None:
        """On CPU devices, __enter__/__exit__ are no-ops for FusedStage and sub-stages."""
        dyn0 = BaseDynamics(model=self.model, device_type="cpu")
        dyn1 = BaseDynamics(model=self.model, device_type="cpu")
        fused = FusedStage(sub_stages=[(0, dyn0), (1, dyn1)], device_type="cpu")

        with fused:
            assert fused.stream is None
            assert dyn0._stream is None
            assert dyn1._stream is None

        assert fused.stream is None

    def test_fused_three_substages(self) -> None:
        """Stream propagation works with three sub-stages (via + operator)."""
        dyn0 = BaseDynamics(model=self.model, device_type="cuda")
        dyn1 = BaseDynamics(model=self.model, device_type="cuda")
        dyn2 = BaseDynamics(model=self.model, device_type="cuda")
        fused = dyn0 + dyn1 + dyn2

        mock_stream = MagicMock(spec=torch.cuda.Stream)
        mock_stream_ctx = MagicMock()

        with patch("torch.cuda.is_available", return_value=True):
            with patch("torch.cuda.Stream", return_value=mock_stream):
                with patch("torch.cuda.stream", return_value=mock_stream_ctx):
                    with patch.object(
                        type(fused),
                        "device",
                        property(lambda s: torch.device("cuda:0")),
                    ):
                        with fused:
                            # All three sub-stage dynamics should share the stream
                            for _, dyn in fused.sub_stages:
                                assert dyn._stream is mock_stream

                        # All cleaned up
                        for _, dyn in fused.sub_stages:
                            assert dyn._stream is None


# -----------------------------------------------------------------------------
# TestDistributedPipelineComposition
# -----------------------------------------------------------------------------


class TestDistributedPipelineComposition:
    """Tests for multi-stage pipeline composition via the | operator."""

    def setup_method(self) -> None:
        """Set up test fixtures before each test method."""
        self.model = NonConservativeDemoModel()

    def test_three_stage_pipe_operator(self) -> None:
        """dyn1 | dyn2 | dyn3 produces a DistributedPipeline with 3 stages."""
        dyn1 = BaseDynamics(model=self.model)
        dyn2 = BaseDynamics(model=self.model)
        dyn3 = BaseDynamics(model=self.model)
        pipeline = dyn1 | dyn2 | dyn3
        assert isinstance(pipeline, DistributedPipeline)
        assert len(pipeline.stages) == 3
        assert set(pipeline.stages.keys()) == {0, 1, 2}

    def test_four_stage_pipe_operator(self) -> None:
        """dyn1 | dyn2 | dyn3 | dyn4 produces 4 stages at ranks 0, 1, 2, 3."""
        dyn1 = BaseDynamics(model=self.model)
        dyn2 = BaseDynamics(model=self.model)
        dyn3 = BaseDynamics(model=self.model)
        dyn4 = BaseDynamics(model=self.model)
        pipeline = dyn1 | dyn2 | dyn3 | dyn4
        assert isinstance(pipeline, DistributedPipeline)
        assert len(pipeline.stages) == 4
        assert set(pipeline.stages.keys()) == {0, 1, 2, 3}

    def test_pipe_preserves_stage_identity(self) -> None:
        """Each stages[rank] is the exact dynamics object passed in."""
        dyn1 = BaseDynamics(model=self.model)
        dyn2 = BaseDynamics(model=self.model)
        dyn3 = BaseDynamics(model=self.model)
        pipeline = dyn1 | dyn2 | dyn3
        assert pipeline.stages[0] is dyn1
        assert pipeline.stages[1] is dyn2
        assert pipeline.stages[2] is dyn3

    def test_pipe_setup_wires_prior_next_ranks(self) -> None:
        """After setup(), prior_rank/next_rank chain is correctly wired."""
        cfg = BufferConfig(num_systems=10, num_nodes=100, num_edges=200)
        dyn1 = BaseDynamics(model=self.model, buffer_config=cfg)
        dyn2 = BaseDynamics(model=self.model, buffer_config=cfg)
        dyn3 = BaseDynamics(model=self.model, buffer_config=cfg)
        pipeline = dyn1 | dyn2 | dyn3
        pipeline.setup()
        assert pipeline.stages[0].prior_rank is None
        assert pipeline.stages[0].next_rank == 1
        assert pipeline.stages[1].prior_rank == 0
        assert pipeline.stages[1].next_rank == 2
        assert pipeline.stages[2].prior_rank == 1
        assert pipeline.stages[2].next_rank is None

    def test_fused_in_multi_stage_pipeline(self) -> None:
        """(dyn1 + dyn2) | dyn3 | dyn4 works with FusedStage at rank 0."""
        dyn1 = BaseDynamics(model=self.model)
        dyn2 = BaseDynamics(model=self.model)
        dyn3 = BaseDynamics(model=self.model)
        dyn4 = BaseDynamics(model=self.model)
        pipeline = (dyn1 + dyn2) | dyn3 | dyn4
        assert isinstance(pipeline, DistributedPipeline)
        assert len(pipeline.stages) == 3
        assert isinstance(pipeline.stages[0], FusedStage)
        assert pipeline.stages[1] is dyn3
        assert pipeline.stages[2] is dyn4

    def test_pipe_or_rejects_non_dynamics(self) -> None:
        """pipeline | 'not_a_dynamics' raises TypeError."""
        dyn1 = BaseDynamics(model=self.model)
        dyn2 = BaseDynamics(model=self.model)
        pipeline = dyn1 | dyn2
        with pytest.raises(TypeError, match="Right operand of \\|"):
            pipeline | "not_a_dynamics"

    def test_pipe_operator_does_not_mutate_original(self) -> None:
        """pipeline | dyn3 returns a new pipeline; the original is unchanged."""
        dyn1 = BaseDynamics(model=self.model)
        dyn2 = BaseDynamics(model=self.model)
        dyn3 = BaseDynamics(model=self.model)
        original = dyn1 | dyn2
        extended = original | dyn3
        assert len(original.stages) == 2
        assert len(extended.stages) == 3

    def test_two_stage_pipe_unchanged(self) -> None:
        """Basic backward compatibility: dyn1 | dyn2 still produces {0: dyn1, 1: dyn2}."""
        dyn1 = BaseDynamics(model=self.model)
        dyn2 = BaseDynamics(model=self.model)
        pipeline = dyn1 | dyn2
        assert isinstance(pipeline, DistributedPipeline)
        assert len(pipeline.stages) == 2
        assert pipeline.stages[0] is dyn1
        assert pipeline.stages[1] is dyn2

    def test_pipeline_merge_two_pipelines(self) -> None:
        """(dyn1 | dyn2) | (dyn3 | dyn4) merges into a single 4-stage pipeline."""
        dyn1 = BaseDynamics(model=self.model)
        dyn2 = BaseDynamics(model=self.model)
        dyn3 = BaseDynamics(model=self.model)
        dyn4 = BaseDynamics(model=self.model)
        left = dyn1 | dyn2
        right = dyn3 | dyn4
        merged = left | right
        assert isinstance(merged, DistributedPipeline)
        assert len(merged.stages) == 4
        assert set(merged.stages.keys()) == {0, 1, 2, 3}
        assert merged.stages[0] is dyn1
        assert merged.stages[1] is dyn2
        assert merged.stages[2] is dyn3
        assert merged.stages[3] is dyn4

    def test_pipeline_merge_setup_wires_correctly(self) -> None:
        """Merged pipelines have correct prior_rank/next_rank chain after setup()."""
        cfg = BufferConfig(num_systems=10, num_nodes=100, num_edges=200)
        dyn1 = BaseDynamics(model=self.model, buffer_config=cfg)
        dyn2 = BaseDynamics(model=self.model, buffer_config=cfg)
        dyn3 = BaseDynamics(model=self.model, buffer_config=cfg)
        dyn4 = BaseDynamics(model=self.model, buffer_config=cfg)
        left = dyn1 | dyn2
        right = dyn3 | dyn4
        merged = left | right
        merged.setup()
        # Verify linear chain: 0 -> 1 -> 2 -> 3
        assert merged.stages[0].prior_rank is None
        assert merged.stages[0].next_rank == 1
        assert merged.stages[1].prior_rank == 0
        assert merged.stages[1].next_rank == 2
        assert merged.stages[2].prior_rank == 1
        assert merged.stages[2].next_rank == 3
        assert merged.stages[3].prior_rank == 2
        assert merged.stages[3].next_rank is None

    def test_pipe_rejects_composition_after_dist_init(self) -> None:
        """pipeline | dyn raises RuntimeError when dist is already initialized."""
        dyn1 = BaseDynamics(model=self.model)
        dyn2 = BaseDynamics(model=self.model)
        dyn3 = BaseDynamics(model=self.model)
        pipeline = dyn1 | dyn2
        with patch("nvalchemi.dynamics.base.dist.is_initialized", return_value=True):
            with pytest.raises(RuntimeError, match="Cannot compose pipelines"):
                pipeline | dyn3

    def test_pipe_merge_rejects_after_dist_init(self) -> None:
        """pipeline1 | pipeline2 raises RuntimeError when dist is initialized."""
        dyn1 = BaseDynamics(model=self.model)
        dyn2 = BaseDynamics(model=self.model)
        dyn3 = BaseDynamics(model=self.model)
        dyn4 = BaseDynamics(model=self.model)
        pipeline1 = dyn1 | dyn2
        pipeline2 = dyn3 | dyn4
        with patch("nvalchemi.dynamics.base.dist.is_initialized", return_value=True):
            with pytest.raises(RuntimeError, match="Cannot compose pipelines"):
                pipeline1 | pipeline2


# -----------------------------------------------------------------------------
# Helper class for FusedStage substage hook tests
# -----------------------------------------------------------------------------


class _TrackingHook:
    """Minimal hook that records every call for testing."""

    def __init__(self, stage: DynamicsStage, frequency: int = 1) -> None:
        self.stage = stage
        self.frequency = frequency
        self.call_count = 0
        self.call_step_counts: list[int] = []

    def __call__(self, ctx: HookContext, stage: DynamicsStage) -> None:
        """Record the call and current step count."""
        self.call_count += 1
        self.call_step_counts.append(ctx.step_count)


# -----------------------------------------------------------------------------
# TestFusedStageSubstageHooks
# -----------------------------------------------------------------------------


class TestFusedStageSubstageHooks:
    """Tests for FusedStage substage hook firing.

    Verifies that hooks registered on substages fire correctly during
    FusedStage.step(), including:
    - BEFORE_STEP, AFTER_COMPUTE, BEFORE_PRE_UPDATE, AFTER_POST_UPDATE, AFTER_STEP
    - ON_CONVERGE (when convergence is detected)
    - Hook frequency is respected
    - Correct firing order
    - Step counts are incremented on substages

    Stages that are NOT fired on substages:
    - BEFORE_COMPUTE (compute is shared, not per-substage)
    - AFTER_PRE_UPDATE (masked_update is atomic, no intermediate hooks)
    - BEFORE_POST_UPDATE (same reason)
    """

    def setup_method(self) -> None:
        """Set up test fixtures before each test method.

        Uses NonConservativeDemoModel because tests that call compute() followed
        by masked_update() fail with conservative models (compute enables grads
        on positions, which then conflicts with masked_update's in-place ops).
        """
        self.model = NonConservativeDemoModel()

    def test_substage_after_step_hooks_fire(self) -> None:
        """AFTER_STEP hooks on each substage should fire once per step.

        Creates two substages with AFTER_STEP hooks, steps once, and
        verifies both hooks fired exactly once.
        """
        dynamics0 = BaseDynamics(model=self.model)
        dynamics1 = BaseDynamics(model=self.model)

        hook0 = _TrackingHook(DynamicsStage.AFTER_STEP)
        hook1 = _TrackingHook(DynamicsStage.AFTER_STEP)
        dynamics0.register_hook(hook0)
        dynamics1.register_hook(hook1)

        fused = FusedStage(sub_stages=[(0, dynamics0), (1, dynamics1)])

        batch = create_batch_with_status(n_graphs=3)
        batch.status = torch.tensor([0, 0, 1])
        batch.fmax = torch.tensor([0.1, 0.1, 0.1])

        fused.step(batch)

        assert hook0.call_count == 1
        assert hook1.call_count == 1

    def test_substage_before_step_hooks_fire(self) -> None:
        """BEFORE_STEP hooks on substages should fire at the start of each step.

        Registers a BEFORE_STEP hook on a substage, steps once, and
        verifies the hook fired once.
        """
        dynamics0 = BaseDynamics(model=self.model)

        hook = _TrackingHook(DynamicsStage.BEFORE_STEP)
        dynamics0.register_hook(hook)

        fused = FusedStage(sub_stages=[(0, dynamics0)])

        batch = create_batch_with_status(n_graphs=3)
        batch.status = torch.tensor([0, 0, 0])
        batch.fmax = torch.tensor([0.1, 0.1, 0.1])

        fused.step(batch)

        assert hook.call_count == 1

    def test_substage_after_compute_hooks_fire(self) -> None:
        """AFTER_COMPUTE hooks on substages should fire after the shared forward pass.

        Registers an AFTER_COMPUTE hook on a substage, steps once, and
        verifies the hook fired once.
        """
        dynamics0 = BaseDynamics(model=self.model)

        hook = _TrackingHook(DynamicsStage.AFTER_COMPUTE)
        dynamics0.register_hook(hook)

        fused = FusedStage(sub_stages=[(0, dynamics0)])

        batch = create_batch_with_status(n_graphs=3)
        batch.status = torch.tensor([0, 0, 0])
        batch.fmax = torch.tensor([0.1, 0.1, 0.1])

        fused.step(batch)

        assert hook.call_count == 1

    def test_substage_before_pre_update_and_after_post_update_hooks_fire(self) -> None:
        """BEFORE_PRE_UPDATE and AFTER_POST_UPDATE hooks bracket each substage's masked_update.

        Registers both hooks on a substage, steps once, and verifies
        both hooks fired once.
        """
        dynamics0 = BaseDynamics(model=self.model)

        hook_before = _TrackingHook(DynamicsStage.BEFORE_PRE_UPDATE)
        hook_after = _TrackingHook(DynamicsStage.AFTER_POST_UPDATE)
        dynamics0.register_hook(hook_before)
        dynamics0.register_hook(hook_after)

        fused = FusedStage(sub_stages=[(0, dynamics0)])

        batch = create_batch_with_status(n_graphs=3)
        batch.status = torch.tensor([0, 0, 0])
        batch.fmax = torch.tensor([0.1, 0.1, 0.1])

        fused.step(batch)

        assert hook_before.call_count == 1
        assert hook_after.call_count == 1

    def test_substage_on_converge_hooks_fire(self) -> None:
        """ON_CONVERGE hooks on substages should fire when convergence is detected.

        Creates a substage with a high convergence threshold so DemoModel
        forces always trigger convergence, registers an ON_CONVERGE hook,
        and verifies it fired.
        """
        dynamics0 = BaseDynamics(
            model=self.model,
            convergence_hook=ConvergenceHook.from_fmax(1e6),
        )

        hook = _TrackingHook(DynamicsStage.ON_CONVERGE)
        dynamics0.register_hook(hook)

        fused = FusedStage(sub_stages=[(0, dynamics0)])

        batch = create_batch_with_status(n_graphs=3)
        batch.status = torch.tensor([0, 0, 0])

        fused.step(batch)

        assert hook.call_count == 1

    def test_substage_on_converge_does_not_fire_when_not_converged(self) -> None:
        """ON_CONVERGE hooks should NOT fire when samples are not converged.

        Uses threshold of 0 so DemoModel forces never satisfy convergence,
        and verifies the ON_CONVERGE hook did not fire.
        """
        dynamics0 = BaseDynamics(
            model=self.model,
            convergence_hook=ConvergenceHook.from_fmax(0.0),
        )

        hook = _TrackingHook(DynamicsStage.ON_CONVERGE)
        dynamics0.register_hook(hook)

        fused = FusedStage(sub_stages=[(0, dynamics0)])

        batch = create_batch_with_status(n_graphs=3)
        batch.status = torch.tensor([0, 0, 0])

        fused.step(batch)

        assert hook.call_count == 0

    def test_on_converge_only_fires_for_active_samples(self) -> None:
        """ON_CONVERGE converged_mask should only include samples active in that stage.

        Two-stage FusedStage where both stages use high-threshold convergence.
        Samples [0,1] are at status 0, sample [2] at status 1.
        Stage-0's ON_CONVERGE mask must be [True, True, False],
        stage-1's ON_CONVERGE mask must be [False, False, True].
        """

        class _MaskCapture:
            stage = DynamicsStage.ON_CONVERGE
            frequency = 1

            def __init__(self) -> None:
                self.masks: list[torch.Tensor] = []

            def __call__(self, ctx: HookContext, stage: DynamicsStage) -> None:
                self.masks.append(ctx.converged_mask.clone())

        dynamics0 = BaseDynamics(
            model=self.model,
            convergence_hook=ConvergenceHook.from_fmax(1e6),
        )
        dynamics1 = BaseDynamics(
            model=self.model,
            convergence_hook=ConvergenceHook.from_fmax(1e6),
        )

        cap0 = _MaskCapture()
        cap1 = _MaskCapture()
        dynamics0.register_hook(cap0)
        dynamics1.register_hook(cap1)

        fused = FusedStage(sub_stages=[(0, dynamics0), (1, dynamics1)])

        batch = create_batch_with_status(n_graphs=3)
        batch.status = torch.tensor([0, 0, 1])

        fused.step(batch)

        assert len(cap0.masks) == 1
        assert cap0.masks[0].tolist() == [True, True, False]
        assert len(cap1.masks) == 1
        assert cap1.masks[0].tolist() == [False, False, True]

    def test_non_applicable_stages_not_fired(self) -> None:
        """BEFORE_COMPUTE, AFTER_PRE_UPDATE, and BEFORE_POST_UPDATE should NOT fire on substages.

        These stages are intentionally not fired on substages because:
        - BEFORE_COMPUTE: compute is shared, not per-substage
        - AFTER_PRE_UPDATE: masked_update is atomic, no intermediate hooks
        - BEFORE_POST_UPDATE: same reason

        Registers hooks for all three stages, steps once, and verifies
        all hooks have call_count == 0.
        """
        dynamics0 = BaseDynamics(model=self.model)

        hook_before_compute = _TrackingHook(DynamicsStage.BEFORE_COMPUTE)
        hook_after_pre = _TrackingHook(DynamicsStage.AFTER_PRE_UPDATE)
        hook_before_post = _TrackingHook(DynamicsStage.BEFORE_POST_UPDATE)
        dynamics0.register_hook(hook_before_compute)
        dynamics0.register_hook(hook_after_pre)
        dynamics0.register_hook(hook_before_post)

        fused = FusedStage(sub_stages=[(0, dynamics0)])

        batch = create_batch_with_status(n_graphs=3)
        batch.status = torch.tensor([0, 0, 0])
        batch.fmax = torch.tensor([0.1, 0.1, 0.1])

        fused.step(batch)

        assert hook_before_compute.call_count == 0
        assert hook_after_pre.call_count == 0
        assert hook_before_post.call_count == 0

    def test_substage_step_count_incremented(self) -> None:
        """Substages and FusedStage should have step_count incremented together.

        Creates a FusedStage with two substages, steps 3 times, and
        verifies each substage's step_count == 3 and FusedStage's step_count == 3.
        """
        dynamics0 = BaseDynamics(model=self.model)
        dynamics1 = BaseDynamics(model=self.model)

        fused = FusedStage(sub_stages=[(0, dynamics0), (1, dynamics1)])

        batch = create_batch_with_status(n_graphs=3)
        batch.status = torch.tensor([0, 0, 1])
        batch.fmax = torch.tensor([0.1, 0.1, 0.1])

        for _ in range(3):
            fused.step(batch)

        assert dynamics0.step_count == 3
        assert dynamics1.step_count == 3
        assert fused.step_count == 3

    def test_substage_hook_frequency_respected(self) -> None:
        """Substage hooks should respect their frequency setting.

        Registers an AFTER_STEP hook with frequency=3 on a substage,
        steps the FusedStage 6 times, and verifies the hook fired exactly
        2 times (at step_count 0 and 3).
        """
        dynamics0 = BaseDynamics(model=self.model)

        hook = _TrackingHook(DynamicsStage.AFTER_STEP, frequency=3)
        dynamics0.register_hook(hook)

        fused = FusedStage(sub_stages=[(0, dynamics0)])

        batch = create_batch_with_status(n_graphs=3)
        batch.status = torch.tensor([0, 0, 0])
        batch.fmax = torch.tensor([0.1, 0.1, 0.1])

        for _ in range(6):
            fused.step(batch)

        # Hooks fire when step_count % frequency == 0
        # step_count increments AFTER hooks fire, so:
        # - Step 0: step_count=0 before increment, 0 % 3 == 0 -> fires
        # - Step 1: step_count=1 before increment, 1 % 3 != 0 -> no fire
        # - Step 2: step_count=2 before increment, 2 % 3 != 0 -> no fire
        # - Step 3: step_count=3 before increment, 3 % 3 == 0 -> fires
        # - Step 4: step_count=4 before increment, 4 % 3 != 0 -> no fire
        # - Step 5: step_count=5 before increment, 5 % 3 != 0 -> no fire
        assert hook.call_count == 2
        assert hook.call_step_counts == [0, 3]

    def test_hooks_fire_on_substage_with_empty_mask(self) -> None:
        """Hooks should fire even when a substage's mask is empty.

        Creates a FusedStage with 2 substages, sets all batch.status = 1
        (so substage 0 has empty mask), registers BEFORE_PRE_UPDATE and
        AFTER_POST_UPDATE hooks on substage 0, and verifies they still fire.
        """
        dynamics0 = BaseDynamics(model=self.model)
        dynamics1 = BaseDynamics(model=self.model)

        hook_before = _TrackingHook(DynamicsStage.BEFORE_PRE_UPDATE)
        hook_after = _TrackingHook(DynamicsStage.AFTER_POST_UPDATE)
        dynamics0.register_hook(hook_before)
        dynamics0.register_hook(hook_after)

        fused = FusedStage(sub_stages=[(0, dynamics0), (1, dynamics1)])

        batch = create_batch_with_status(n_graphs=3)
        batch.status = torch.tensor([1, 1, 1])  # No samples for substage 0
        batch.fmax = torch.tensor([0.1, 0.1, 0.1])

        fused.step(batch)

        # Hooks should still fire, even though no samples have status=0
        assert hook_before.call_count == 1
        assert hook_after.call_count == 1
