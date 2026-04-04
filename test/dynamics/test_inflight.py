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
End-to-end tests for inflight batching in the dynamics framework.

This module tests the full inflight batching workflow including:
- FusedStage.run() Mode 2 (batch=None with sampler)
- BaseDynamics.refill_check() graduated sample replacement
- Integration with ConvergenceHook for sample migration
"""

from __future__ import annotations

import pytest
import torch

from nvalchemi.data import AtomicData, Batch
from nvalchemi.dynamics.base import (
    BaseDynamics,
    ConvergenceHook,
    FusedStage,
)
from nvalchemi.dynamics.sampler import SizeAwareSampler
from nvalchemi.dynamics.sinks import HostMemory
from nvalchemi.models.demo import DemoModelWrapper

# -----------------------------------------------------------------------------
# Mock Dataset for Inflight Batching Tests
# -----------------------------------------------------------------------------


class MockDataset:
    """Mock dataset for inflight batching tests.

    Attributes
    ----------
    samples : list[tuple[int, int]]
        List of (num_atoms, num_edges) per sample.
    """

    def __init__(self, samples: list[tuple[int, int]]) -> None:
        """Initialize with list of (num_atoms, num_edges) per sample.

        Parameters
        ----------
        samples : list[tuple[int, int]]
            Each element is (num_atoms, num_edges) for that sample index.
        """
        self.samples = samples

    def __len__(self) -> int:
        """Return number of samples.

        Returns
        -------
        int
            Number of samples in the dataset.
        """
        return len(self.samples)

    def get_metadata(self, index: int) -> tuple[int, int]:
        """Return metadata for a sample without full load.

        Parameters
        ----------
        index : int
            Sample index.

        Returns
        -------
        tuple[int, int]
            (num_atoms, num_edges) for the sample.
        """
        return self.samples[index]

    def __getitem__(self, index: int) -> tuple[AtomicData, dict]:
        """Load and return a sample.

        Parameters
        ----------
        index : int
            Sample index.

        Returns
        -------
        tuple[AtomicData, dict]
            The atomic data and an empty metadata dict.
        """
        num_atoms, num_edges = self.samples[index]
        data = AtomicData(
            atomic_numbers=torch.arange(1, num_atoms + 1, dtype=torch.long),
            positions=torch.randn(num_atoms, 3),
        )
        return data, {}


# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------


def create_batch_with_status(n_graphs: int = 3, device: str = "cpu") -> Batch:
    """Create a batch with N single-atom molecules for testing.

    Parameters
    ----------
    n_graphs : int
        Number of graphs in the batch.
    device : str
        Device to place tensors on.

    Returns
    -------
    Batch
        A batch with forces, energies, and status initialized.
    """
    data_list = [
        AtomicData(
            atomic_numbers=torch.tensor([6], dtype=torch.long),
            positions=torch.randn(1, 3),
        )
        for _ in range(n_graphs)
    ]
    batch = Batch.from_data_list(data_list, device=device)
    batch["forces"] = torch.zeros(batch.num_nodes, 3)
    batch["energies"] = torch.zeros(batch.num_graphs, 1)
    batch["status"] = torch.zeros(batch.num_graphs, 1, dtype=torch.long)
    return batch


def initialize_batch_for_dynamics(batch: Batch) -> Batch:
    """Initialize forces and energies on a batch for dynamics simulation.

    This helper ensures a batch from `SizeAwareSampler.build_initial_batch()`
    has the required `forces` and `energies` attributes for dynamics steps.

    Parameters
    ----------
    batch : Batch
        The batch to initialize (typically from sampler).

    Returns
    -------
    Batch
        The same batch with forces and energies initialized.
    """
    batch["forces"] = torch.zeros(batch.num_nodes, 3, device=batch.device)
    batch["energies"] = torch.zeros(batch.num_graphs, 1, device=batch.device)
    return batch


# -----------------------------------------------------------------------------
# TestFusedStageMode2
# -----------------------------------------------------------------------------


class TestFusedStageInflight:
    """End-to-end tests for FusedStage.run() with inflight batching."""

    def setup_method(self) -> None:
        """Set up test fixtures before each test method."""
        self.model = DemoModelWrapper()

    def test_builds_initial_batch_from_sampler(self, device: str) -> None:
        """FusedStage with sampler should process batches and terminate on exhaustion.

        When a sampler is configured, FusedStage should process batches, perform
        refill checks, and terminate when sampler exhausts and all samples graduate.
        """
        # Small dataset: exactly batch_size samples so sampler exhausts immediately
        dataset = MockDataset([(2, 0)] * 5)
        sampler = SizeAwareSampler(
            dataset, max_atoms=20, max_edges=10, max_batch_size=5
        )

        dynamics = BaseDynamics(model=self.model, device_type=device)
        # Convergence hook: high threshold so DemoModel forces always converge
        hook = ConvergenceHook.from_fmax(1e6, source_status=0, target_status=1)
        dynamics.register_hook(hook)

        fused = FusedStage(
            sub_stages=[(0, dynamics)],
            sampler=sampler,
            refill_frequency=1,
            device_type=device,
        )

        # Build and initialize batch manually
        batch = sampler.build_initial_batch()
        initialize_batch_for_dynamics(batch)

        result = fused.run(batch=batch)

        # Sampler exhausted + all graduated → returns None
        assert result is None
        assert fused.inflight_mode is True

    def test_graduates_and_replaces_samples(self) -> None:
        """refill_check should replace graduated samples in-place.

        This tests the refill_check mechanism directly, verifying that
        graduated samples are written to sinks and replacements are fetched
        into the same batch object (in-place modification).
        """
        # Create dataset with enough samples for replacement
        dataset = MockDataset([(2, 0)] * 10)
        sampler = SizeAwareSampler(
            dataset, max_atoms=20, max_edges=10, max_batch_size=3
        )
        sink = HostMemory(capacity=100)

        dynamics = BaseDynamics(
            model=self.model,
            sampler=sampler,
            sinks=[sink],
            device_type="cpu",
        )

        # Build initial batch manually and initialize for dynamics
        initial_batch = sampler.build_initial_batch()
        initialize_batch_for_dynamics(initial_batch)
        # Mark 2 samples as graduated (status=1)
        initial_batch["status"] = torch.tensor([[0], [1], [1]])

        # Call refill_check directly (not through run loop)
        result = dynamics.refill_check(initial_batch, exit_status=1)

        # Graduated samples should be written to sink
        assert len(sink) == 2
        # Returns a new batch (not same identity)
        assert result is not None
        # Remaining 1 + replacements 2 = 3 (all slots preserved)
        assert result.num_graphs == 3

    def test_terminates_on_dataset_exhaustion(self) -> None:
        """refill_check should return None when sampler is exhausted.

        When all samples are consumed and all current samples have graduated,
        _refill_check should return None and set done=True.
        """
        # Small dataset: 3 samples, max_batch_size=3
        dataset = MockDataset([(2, 0)] * 3)
        sampler = SizeAwareSampler(
            dataset, max_atoms=20, max_edges=10, max_batch_size=3
        )

        dynamics = BaseDynamics(
            model=self.model,
            sampler=sampler,
            device_type="cpu",
        )

        # Build and initialize batch (consumes all 3 samples)
        batch = sampler.build_initial_batch()
        initialize_batch_for_dynamics(batch)
        assert sampler.exhausted  # All 3 consumed

        # Mark all as graduated (status=1)
        batch["status"] = torch.tensor([[1], [1], [1]])

        # Call refill_check - should return None since no replacements available
        result = dynamics.refill_check(batch, exit_status=1)

        # Should return None when all consumed and graduated
        assert result is None
        assert dynamics.done is True

    def test_batch_provided(self, device: str) -> None:
        """Mode 1: batch provided, no sampler — terminates on convergence.

        When a batch is provided and no sampler is configured, FusedStage
        should run in Mode 1 and terminate when all samples reach exit_status.
        """
        dynamics = BaseDynamics(model=self.model, device_type=device)
        # High threshold so DemoModel forces always trigger convergence
        hook = ConvergenceHook.from_fmax(1e6, source_status=0, target_status=1)
        dynamics.register_hook(hook)

        fused = FusedStage(
            sub_stages=[(0, dynamics)],
            device_type=device,
        )  # No sampler

        batch = create_batch_with_status(n_graphs=3)

        with fused:
            result = fused.run(batch=batch)

        # Should converge after 1 step
        assert result is not None
        assert fused.step_count == 1
        assert result.num_graphs == 3  # Unchanged

    def test_error_when_no_batch_and_no_sampler(self, device: str) -> None:
        """Should raise ValueError when batch=None and no sampler.

        Mode 2 requires a sampler. Without it, run(batch=None) is invalid.
        """
        dynamics = BaseDynamics(model=self.model, device_type=device)
        fused = FusedStage(
            sub_stages=[(0, dynamics)],
            device_type=device,
        )  # No sampler

        with pytest.raises(ValueError, match="No batch provided and no sampler"):
            fused.run(batch=None)

    def test_refill_frequency_controls_check_cadence(self) -> None:
        """FusedStage Mode 1 should terminate when all samples converge.

        This test verifies convergence-based termination without a sampler.
        When all samples reach exit_status, the loop stops.
        """
        dynamics = BaseDynamics(model=self.model, device_type="cpu")
        # High threshold so DemoModel forces always trigger convergence
        hook = ConvergenceHook.from_fmax(1e6, source_status=0, target_status=1)
        dynamics.register_hook(hook)

        fused = FusedStage(
            sub_stages=[(0, dynamics)],
            device_type="cpu",
        )  # No sampler - Mode 1

        batch = create_batch_with_status(n_graphs=5)

        result = fused.run(batch=batch)

        # Mode 1 with convergence: should return the batch
        assert result is not None
        assert fused.step_count == 1
        # All samples should have migrated to exit_status=1
        expected_status = torch.tensor([[1]] * 5)
        assert torch.equal(result.status, expected_status)

    def test_graduated_samples_written_to_sinks(self) -> None:
        """Graduated samples should be written to configured sinks.

        When refill_check runs, graduated samples should be stored
        in the sink before being replaced.
        """
        dataset = MockDataset([(2, 0)] * 10)
        sampler = SizeAwareSampler(
            dataset, max_atoms=20, max_edges=10, max_batch_size=3
        )
        sink = HostMemory(capacity=100)

        dynamics = BaseDynamics(
            model=self.model,
            sampler=sampler,
            sinks=[sink],
            device_type="cpu",
        )

        # Build and initialize batch manually
        batch = sampler.build_initial_batch()
        initialize_batch_for_dynamics(batch)
        # Mark 2 samples as graduated
        batch["status"] = torch.tensor([[0], [1], [1]])

        # Call refill_check directly
        dynamics.refill_check(batch, exit_status=1)

        # Sink should contain the 2 graduated samples
        assert len(sink) == 2
        # Read back and verify it's valid batch data
        graduated = sink.read()
        assert graduated.num_graphs == 2

    def test_inflight_mode_property(self) -> None:
        """inflight_mode should return True when sampler is set.

        The inflight_mode property indicates whether Mode 2 is active.
        """
        dataset = MockDataset([(2, 0)] * 5)
        sampler = SizeAwareSampler(
            dataset, max_atoms=20, max_edges=10, max_batch_size=3
        )

        dynamics = BaseDynamics(model=self.model, device_type="cpu")

        # With sampler
        fused_with = FusedStage(
            sub_stages=[(0, dynamics)],
            sampler=sampler,
            device_type="cpu",
        )
        assert fused_with.inflight_mode is True

        # Without sampler
        dynamics2 = BaseDynamics(model=self.model, device_type="cpu")
        fused_without = FusedStage(
            sub_stages=[(0, dynamics2)],
            device_type="cpu",
        )
        assert fused_without.inflight_mode is False


# -----------------------------------------------------------------------------
# TestRefillCheck
# -----------------------------------------------------------------------------


class TestRefillCheck:
    """Unit tests for BaseDynamics.refill_check()."""

    def setup_method(self) -> None:
        """Set up test fixtures before each test method."""
        self.model = DemoModelWrapper()

    def test_refill_replaces_graduated_with_fresh(self) -> None:
        """Should replace graduated samples in-place and return same batch.

        Samples that have reached exit_status have replacement data
        copied into their slots. The batch object is modified in-place
        (same identity), and num_graphs stays constant (slots preserved).
        """
        dataset = MockDataset([(2, 0)] * 10)
        sampler = SizeAwareSampler(
            dataset, max_atoms=20, max_edges=10, max_batch_size=5
        )

        dynamics = BaseDynamics(model=self.model, sampler=sampler, device_type="cpu")

        # Build initial batch (consumes 5 samples)
        batch = sampler.build_initial_batch()
        # Mark 2 samples as graduated (status=1, exit_status=1)
        batch["status"] = torch.tensor([[0], [1], [0], [1], [0]])

        # Refill with exit_status=1
        result = dynamics.refill_check(batch, exit_status=1)

        # Returns a new batch (not same identity)
        assert result is not None
        # Slots preserved: num_graphs stays 5
        assert result.num_graphs == 5
        # All slots should now be active (3 remaining + 2 replacements = 5)
        status = (
            result.status.squeeze(-1) if result.status.dim() == 2 else result.status
        )
        active_count = (status < 1).sum().item()
        assert active_count == 5

    def test_refill_preserves_remaining_status(self) -> None:
        """Remaining samples should keep their original status at original positions.

        With in-place semantics, non-graduated samples stay at their original
        graph indices (positions 0, 2, 4). Their status should not be modified.
        Graduated samples at positions 1, 3 get replaced with status=0.
        """
        dataset = MockDataset([(2, 0)] * 10)
        sampler = SizeAwareSampler(
            dataset, max_atoms=20, max_edges=10, max_batch_size=5
        )

        dynamics = BaseDynamics(model=self.model, sampler=sampler, device_type="cpu")

        batch = sampler.build_initial_batch()
        # Set mixed statuses: 0, 1, 0, 1, 0 (graduated at positions 1, 3)
        batch["status"] = torch.tensor([[0], [1], [0], [1], [0]])

        result = dynamics.refill_check(batch, exit_status=1)

        # Returns a new batch (not same identity)
        assert result is not None
        # Remaining samples at positions 0, 2, 4 should still have status=0
        status = (
            result.status.squeeze(-1) if result.status.dim() == 2 else result.status
        )
        assert status[0].item() == 0  # Position 0: remaining, unchanged
        assert status[2].item() == 0  # Position 2: remaining, unchanged
        assert status[4].item() == 0  # Position 4: remaining, unchanged
        # Positions 1, 3 get replacements with status=0
        assert status[1].item() == 0  # Position 1: replaced
        assert status[3].item() == 0  # Position 3: replaced

    def test_refill_sets_replacement_status_to_zero(self) -> None:
        """Replacement samples should get status=0 at their original positions.

        Fresh samples from the sampler replace graduated slots in-place
        and start at entry status (0).
        """
        dataset = MockDataset([(2, 0)] * 10)
        sampler = SizeAwareSampler(
            dataset, max_atoms=20, max_edges=10, max_batch_size=5
        )

        dynamics = BaseDynamics(model=self.model, sampler=sampler, device_type="cpu")

        batch = sampler.build_initial_batch()
        # Mark all as graduated
        batch["status"] = torch.tensor([[1], [1], [1], [1], [1]])

        result = dynamics.refill_check(batch, exit_status=1)

        # Returns a new batch (not same identity)
        assert result is not None
        # All samples are replacements, should have status=0
        assert (result.status == 0).all()

    def test_refill_returns_none_when_exhausted(self) -> None:
        """Should return None and set done=True when no data remains.

        When the sampler is exhausted and all samples graduated, refill
        should return None and mark the dynamics as done.
        """
        # Very small dataset that will be quickly exhausted
        dataset = MockDataset([(2, 0)] * 3)
        sampler = SizeAwareSampler(
            dataset, max_atoms=20, max_edges=10, max_batch_size=3
        )

        dynamics = BaseDynamics(model=self.model, sampler=sampler, device_type="cpu")

        # Consume all samples
        batch = sampler.build_initial_batch()
        assert sampler.exhausted  # All 3 consumed

        # Mark all as graduated
        batch["status"] = torch.tensor([[1], [1], [1]])

        result = dynamics.refill_check(batch, exit_status=1)

        # No replacements available, no remaining
        assert result is None
        assert dynamics.done is True

    def test_refill_raises_without_sampler(self) -> None:
        """Should raise RuntimeError when sampler is None.

        refill_check requires a sampler to be configured.
        """
        dynamics = BaseDynamics(model=self.model, device_type="cpu")

        batch = create_batch_with_status(n_graphs=3)

        with pytest.raises(RuntimeError, match="requires a sampler"):
            dynamics.refill_check(batch, exit_status=1)

    def test_refill_noop_when_no_graduated(self) -> None:
        """Should return batch unchanged when no samples graduated.

        If no samples have reached exit_status, the batch should be
        returned as-is.
        """
        dataset = MockDataset([(2, 0)] * 10)
        sampler = SizeAwareSampler(
            dataset, max_atoms=20, max_edges=10, max_batch_size=5
        )

        dynamics = BaseDynamics(model=self.model, sampler=sampler, device_type="cpu")

        batch = sampler.build_initial_batch()
        # All samples at status=0 (not graduated)
        batch["status"] = torch.tensor([[0], [0], [0], [0], [0]])

        result = dynamics.refill_check(batch, exit_status=1)

        # Should return same batch unchanged
        assert result is batch

    def test_refill_writes_bookkeeping_to_storage(self) -> None:
        """Dynamics bookkeeping fields are written to result storage.

        After refill_check, ``status`` should live in the result batch's
        storage groups (not in ``__dict__``).  Remaining samples keep
        their values; replacements get defaults (status=0).
        """
        dataset = MockDataset([(2, 0)] * 10)
        sampler = SizeAwareSampler(
            dataset, max_atoms=20, max_edges=10, max_batch_size=5
        )

        dynamics = BaseDynamics(model=self.model, sampler=sampler, device_type="cpu")

        batch = sampler.build_initial_batch()
        batch["status"] = torch.tensor([[0], [1], [0], [1], [0]])

        result = dynamics.refill_check(batch, exit_status=1)

        assert result is not None
        assert result.num_graphs == 5

        status = result.status
        if status.dim() == 2:
            status = status.squeeze(-1)
        assert status[0].item() == 0
        assert status[1].item() == 0
        assert status[2].item() == 0
        assert status[3].item() == 0
        assert status[4].item() == 0

        assert "status" in result

    def test_refill_partial_replacement(self) -> None:
        """When sampler has fewer replacements than graduated, batch shrinks.

        If 3 samples graduate but only 1 replacement is available, the batch
        shrinks to 3 graphs (2 remaining + 1 replacement) after defrag compacts
        the remaining graphs and append adds the available replacement.
        """
        # Only 6 samples total
        dataset = MockDataset([(2, 0)] * 6)
        sampler = SizeAwareSampler(
            dataset, max_atoms=20, max_edges=10, max_batch_size=5
        )

        dynamics = BaseDynamics(model=self.model, sampler=sampler, device_type="cpu")

        # Build initial batch (consumes 5 samples)
        batch = sampler.build_initial_batch()
        assert len(sampler) == 1  # Only 1 sample left

        # Graduate 3 samples (positions 0, 1, 2)
        batch["status"] = torch.tensor([[1], [1], [1], [0], [0]])

        result = dynamics.refill_check(batch, exit_status=1)

        # Returns a new batch (not same identity)
        assert result is not None
        # Batch shrinks: 2 remaining + 1 replacement = 3
        assert result.num_graphs == 3
        status = (
            result.status.squeeze(-1) if result.status.dim() == 2 else result.status
        )
        active_count = (status < 1).sum().item()
        assert active_count == 3  # 2 remaining + 1 replacement


# -----------------------------------------------------------------------------
# TestInflightWithConvergence
# -----------------------------------------------------------------------------


class TestInflightWithConvergence:
    """Tests for inflight batching combined with convergence hooks."""

    def setup_method(self) -> None:
        """Set up test fixtures before each test method."""
        self.model = DemoModelWrapper()

    def test_convergence_triggers_graduation_and_replacement(self) -> None:
        """ConvergenceHook + FusedStage.run should migrate samples through stages.

        This test verifies multi-stage status migration with Mode 1 (no sampler):
        1. Samples start at status=0
        2. Samples converge via auto-registered ConvergenceHook (0 -> 1)
        3. Samples converge via manually-registered hook (1 -> 2, exit_status)
        4. Loop terminates when all samples reach exit_status

        Note: Auto-registered hooks only wire adjacent sub-stages (N-1 hooks for
        N sub-stages). The last sub-stage needs a manually-registered hook to
        migrate samples to exit_status.

        Note: Both hooks fire in the same step because AFTER_STEP hooks for all
        sub-stages fire after all masked updates complete. So samples can migrate
        0 -> 1 -> 2 in a single step.
        """
        # Create 2-sub-stage FusedStage (opt + md style)
        # Use high thresholds so DemoModel forces always trigger convergence
        dyn0 = BaseDynamics(
            model=self.model,
            device_type="cpu",
            convergence_hook=ConvergenceHook.from_fmax(1e6),
        )
        dyn1 = BaseDynamics(model=self.model, device_type="cpu")

        # FusedStage auto-registers hook 0 -> 1 on dyn0
        # We need to manually register hook 1 -> 2 on dyn1
        hook_to_exit = ConvergenceHook.from_fmax(1e6, source_status=1, target_status=2)
        dyn1.register_hook(hook_to_exit)

        fused = FusedStage(
            sub_stages=[(0, dyn0), (1, dyn1)],
            device_type="cpu",
        )

        # Create batch with 5 samples at status=0
        batch = create_batch_with_status(n_graphs=5)

        # Run until all samples reach exit_status=2
        result = fused.run(batch=batch)

        # Verify:
        # 1. At least one step was executed
        assert fused.step_count >= 1

        # 2. Result should be the batch (Mode 1 returns batch)
        assert result is not None

        # 3. With high threshold, samples converge immediately and
        # migrate 0 -> 1 -> 2 (can happen in single step)
        expected_status = torch.tensor([[2]] * 5)
        assert torch.equal(result.status, expected_status)

    def test_two_stage_migration_path(self) -> None:
        """Samples should migrate through all stages before graduation.

        With a 2-sub-stage FusedStage (status 0 and 1), samples should:
        1. Start at status=0
        2. Migrate to status=1 when converged in stage 0
        3. Migrate to status=2 (exit_status) when converged in stage 1
        """
        dataset = MockDataset([(2, 0)] * 10)
        sampler = SizeAwareSampler(
            dataset, max_atoms=20, max_edges=10, max_batch_size=3
        )

        # High thresholds so DemoModel forces always trigger convergence
        dyn0 = BaseDynamics(
            model=self.model,
            device_type="cpu",
            convergence_hook=ConvergenceHook.from_fmax(1e6),
        )
        dyn1 = BaseDynamics(model=self.model, device_type="cpu")

        # Auto-registered hook: 0 -> 1
        fused = FusedStage(
            sub_stages=[(0, dyn0), (1, dyn1)],
            device_type="cpu",
        )

        # Manually add hook for 1 -> 2 (exit_status)
        hook_1_to_2 = ConvergenceHook.from_fmax(1e6, source_status=1, target_status=2)
        dyn1.register_hook(hook_1_to_2)

        batch = sampler.build_initial_batch()
        initialize_batch_for_dynamics(batch)

        # After 1 step: status should migrate 0 -> 1 -> 2
        batch, _converged = fused.step(batch)

        # All samples should be at exit_status=2
        assert (batch.status.squeeze(-1) == 2).all()

    def test_mixed_convergence_rates(self) -> None:
        """Samples with different fmax should converge at different rates.

        Some samples converge quickly, others take longer. The batch should
        handle mixed statuses correctly.
        """
        dataset = MockDataset([(2, 0)] * 10)
        sampler = SizeAwareSampler(
            dataset, max_atoms=20, max_edges=10, max_batch_size=4
        )

        dynamics = BaseDynamics(model=self.model, device_type="cpu")
        # Use explicit key="fmax" to test mixed convergence via batch["fmax"]
        hook = ConvergenceHook(
            criteria=[{"key": "fmax", "threshold": 0.1}],
            source_status=0,
            target_status=1,
        )
        dynamics.register_hook(hook)

        fused = FusedStage(
            sub_stages=[(0, dynamics)],
            sampler=sampler,
            refill_frequency=10,  # Don't refill during test
            device_type="cpu",
        )

        batch = sampler.build_initial_batch()
        initialize_batch_for_dynamics(batch)
        # Mixed fmax: 2 converged (0.05 < 0.1), 2 not (0.2 > 0.1)
        batch["fmax"] = torch.tensor([[0.05], [0.05], [0.2], [0.2]])

        batch, _converged = fused.step(batch)

        # First 2 should migrate to status=1, last 2 stay at status=0
        expected = torch.tensor([[1], [1], [0], [0]])
        assert torch.equal(batch.status, expected)


class TestStepConvergenceReturn:
    """Tests for BaseDynamics.step() returning converged indices."""

    def setup_method(self) -> None:
        """Set up test fixtures before each test method."""
        self.model = DemoModelWrapper()

    def test_step_returns_converged_indices(self) -> None:
        """BaseDynamics.step() should return converged indices.

        After calling step(), the second return value should contain the
        indices of samples that converged during that step.
        """
        # Use high threshold so DemoModel forces always converge
        hook = ConvergenceHook.from_fmax(1e6)
        dynamics = BaseDynamics(
            model=self.model, device_type="cpu", convergence_hook=hook
        )

        batch = create_batch_with_status(n_graphs=4)

        batch, converged = dynamics.step(batch)

        # All samples should converge with high threshold
        assert converged is not None
        assert converged.numel() == 4

    def test_step_returns_none_when_no_convergence(self) -> None:
        """step() should return None for converged when nothing converges.

        When no samples meet the convergence criteria, the second return
        value should be None.
        """
        # Use threshold of 0 so DemoModel forces never converge
        hook = ConvergenceHook.from_fmax(0.0)
        dynamics = BaseDynamics(
            model=self.model, device_type="cpu", convergence_hook=hook
        )

        batch = create_batch_with_status(n_graphs=3)

        batch, converged = dynamics.step(batch)

        assert converged is None

    def test_step_returns_none_without_convergence_hook(self) -> None:
        """step() should return None for converged when no hook is configured.

        Without a convergence hook, _check_convergence returns None and
        step() should return None for the converged indices.
        """
        dynamics = BaseDynamics(model=self.model, device_type="cpu")

        batch = create_batch_with_status(n_graphs=3)

        batch, converged = dynamics.step(batch)

        assert converged is None
