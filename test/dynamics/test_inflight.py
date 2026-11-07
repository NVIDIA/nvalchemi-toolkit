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

from typing import Any
from unittest.mock import Mock, patch

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
)
from nvalchemi.dynamics.demo import DemoDynamics
from nvalchemi.dynamics.sampler import SizeAwareSampler
from nvalchemi.dynamics.sinks import HostMemory
from nvalchemi.hooks import DynamicsContext
from nvalchemi.models.demo import DemoModel, DemoModelWrapper

from ..distributed._gloo_harness import run_gloo
from .conftest import _MockSampler

# FIRE is deprecated in favour of FIRE2; these tests still exercise it.
pytestmark = pytest.mark.filterwarnings(
    "ignore:FIRE(VariableCell)? is deprecated:DeprecationWarning"
)

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
        A batch with forces, energy, and status initialized.
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
    batch["energy"] = torch.zeros(batch.num_graphs, 1)
    batch["status"] = torch.zeros(batch.num_graphs, 1, dtype=torch.long)
    return batch


def initialize_batch_for_dynamics(batch: Batch) -> Batch:
    """Initialize forces and energy on a batch for dynamics simulation.

    This helper ensures a batch from `SizeAwareSampler.build_initial_batch()`
    has the required `forces` and `energy` attributes for dynamics steps.

    Parameters
    ----------
    batch : Batch
        The batch to initialize (typically from sampler).

    Returns
    -------
    Batch
        The same batch with forces and energy initialized.
    """
    batch["forces"] = torch.zeros(batch.num_nodes, 3, device=batch.device)
    batch["energy"] = torch.zeros(batch.num_graphs, 1, device=batch.device)
    return batch


class _IncrementDynamics(BaseDynamics):
    """Deterministic dynamics used to observe masked pre/post updates."""

    def pre_update(self, batch: Batch) -> None:
        """Increment positions during the pre-update phase."""
        with torch.no_grad():
            batch.positions.add_(1.0)

    def post_update(self, batch: Batch) -> None:
        """Increment positions during the post-update phase."""
        with torch.no_grad():
            batch.positions.add_(1.0)


def _pipeline_entry_worker(
    rank: int, world_size: int, queue: Any, reprime_on_entry: bool
) -> None:
    """Pass upstream forces through real pipeline buffers and Gloo transport."""
    model = DemoModelWrapper(DemoModel())
    config = BufferConfig(num_systems=1, num_nodes=1, num_edges=0)
    stages = {
        index: _IncrementDynamics(
            model=model,
            buffer_config=config,
            device_type="cpu",
            reprime_on_entry=reprime_on_entry,
        )
        for index in range(world_size)
    }
    pipeline = DistributedPipeline(stages=stages, backend="gloo")
    pipeline.setup()
    stage = pipeline.local_stage
    template = create_batch_with_status(n_graphs=1)
    template.forces.fill_(99.0)
    stage._recv_template = Batch.empty_like(template)
    stage._ensure_buffers(template)
    if rank == 0:
        stage.active_batch = template
        stage._populate_send_buffer(torch.tensor([0]))
        stage.send_buffer.send(dst=1)
        return

    seen_batches = []
    stage.register_hook(
        Mock(
            spec=["stage", "frequency"],
            stage=DynamicsStage.BEFORE_PRE_UPDATE,
            frequency=1,
            side_effect=lambda ctx, stage: seen_batches.append(
                (ctx.batch.forces.clone(), ctx.batch.positions.clone())
            ),
        )
    )
    with (
        patch.object(stage, "_prime_forces", wraps=stage._prime_forces) as prime,
        patch.object(model, "forward", wraps=model.forward) as forward,
    ):
        pipeline.step()
    batch = stage.active_batch
    queue.put(
        (
            prime.call_count,
            forward.call_count,
            bool((seen_batches[0][0] == 99.0).all()),
            torch.allclose(batch.positions, seen_batches[0][1] + 2),
        )
    )


class _MaskRecordingHook:
    """Record active masks at the post-compute boundary."""

    stage = DynamicsStage.AFTER_COMPUTE
    frequency = 1

    def __init__(self) -> None:
        self.active_masks: list[torch.Tensor | None] = []

    def __call__(self, ctx: DynamicsContext, stage: DynamicsStage) -> None:
        """Capture a copy of the active graph mask."""
        active = ctx.active_graph_mask
        self.active_masks.append(None if active is None else active.clone())


# -----------------------------------------------------------------------------
# TestFusedStageMode2
# -----------------------------------------------------------------------------


class TestFusedStageInflight:
    """End-to-end tests for FusedStage.run() with inflight batching."""

    def setup_method(self) -> None:
        """Set up test fixtures before each test method."""
        self.model = DemoModelWrapper(DemoModel())

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

    @pytest.mark.parametrize("replacement_atoms", [2, 5])
    def test_refill_allocates_and_primes_replacement_outputs(
        self, replacement_atoms: int
    ) -> None:
        """Full replacement is primed as a fresh batch, even with new atom counts."""
        dataset = MockDataset([(2, 0)] * 3 + [(replacement_atoms, 0)] * 3)
        sampler = SizeAwareSampler(
            dataset,
            max_atoms=3 * replacement_atoms,
            max_edges=0,
            max_batch_size=3,
            bin_width=replacement_atoms + 1,
        )
        dynamics = DemoDynamics(
            model=self.model,
            n_steps=10,
            convergence_hook=ConvergenceHook.from_fmax(
                1e6,
                source_status=0,
                target_status=1,
            ),
        )
        fused = FusedStage(
            sub_stages=[(0, dynamics)],
            sampler=sampler,
            refill_frequency=1,
        )

        admission_hook = Mock(
            spec=["stage", "frequency"],
            stage=DynamicsStage.ON_ADMISSION,
            frequency=1,
        )
        fused.register_hook(admission_hook)
        with (
            patch.object(fused, "_prime_forces", wraps=fused._prime_forces) as prime,
            patch.object(fused, "compute", wraps=fused.compute) as compute,
        ):
            result = fused.run(batch=None)

        assert result is None
        assert admission_hook.call_count == 2
        for call in admission_hook.call_args_list:
            assert call.args[0].previous_graph_indices.tolist() == [-1] * 3
        assert fused._pending_graph_indices is None
        assert prime.call_count == 2  # Initial batch and fully replaced batch.
        assert compute.call_count == 4  # Two primes and two steps.
        replacement_batch = compute.call_args_list[-1].args[0]
        assert replacement_batch.forces.shape == (3 * replacement_atoms, 3)
        assert replacement_batch.energy.shape == (3, 1)

    def test_base_refill_checks_convergence_after_initial_forces(self) -> None:
        """Refilled graphs can converge after their initial forces are computed."""
        dataset = MockDataset([(2, 0)] * 3)
        sampler = SizeAwareSampler(dataset, max_atoms=4, max_edges=0, max_batch_size=2)
        hook = _MaskRecordingHook()
        dynamics = _IncrementDynamics(
            model=self.model,
            sampler=sampler,
            convergence_hook=ConvergenceHook.from_fmax(1e6),
            hooks=[hook],
        )
        batch = initialize_batch_for_dynamics(sampler.build_initial_batch())
        batch["status"] = torch.tensor([[0], [1]])
        dynamics._forces_primed = True

        result = dynamics.refill_check(batch, exit_status=1)

        assert result is not None
        assert dynamics._forces_primed is True
        assert result.reprime_pending.view(-1).tolist() == [False, True]
        assert result.forces is not None
        positions_before = result.positions.clone()

        with patch.object(
            dynamics, "_prime_forces", wraps=dynamics._prime_forces
        ) as prime_forces:
            result, converged = dynamics.step(result)

        assert prime_forces.call_count == 0
        assert converged is not None and converged.tolist() == [0, 1]
        torch.testing.assert_close(result.positions[:2], positions_before[:2] + 2.0)
        torch.testing.assert_close(result.positions[2:], positions_before[2:])
        assert hook.active_masks[-1].tolist() == [True, True]

    @pytest.mark.parametrize("fused_stage", [False, True])
    @pytest.mark.parametrize("n_replacements", [0, 1, 2])
    def test_refill_primes_only_new_graphs(
        self, n_replacements: int, device: str, fused_stage: bool
    ) -> None:
        """Existing graphs update while refilled graphs get their initial forces."""
        dataset = MockDataset([(2, 0)] * 3 + [(3, 0)] * n_replacements)
        sampler = SizeAwareSampler(dataset, max_batch_size=3)
        hook = _MaskRecordingHook()
        update_hook = _MaskRecordingHook()
        update_hook.stage = DynamicsStage.BEFORE_PRE_UPDATE
        dynamics = _IncrementDynamics(
            model=self.model.to(device), n_steps=10, hooks=[hook, update_hook]
        )
        engine = (
            FusedStage(sub_stages=[(0, dynamics)], sampler=sampler)
            if fused_stage
            else dynamics
        )
        engine.sampler = sampler
        batch = sampler.build_initial_batch().to(device)
        engine.step(batch)
        resident_forces = batch.forces[:2].clone()
        resident_energy = batch.energy[:1].clone()
        batch.status[1:] = engine.exit_status

        result = engine.refill_check(batch, engine.exit_status)

        assert result is not None
        assert engine._pending_graph_indices.tolist() == [0] + [-1] * n_replacements
        assert engine._forces_primed
        assert (
            result.reprime_pending.view(-1).tolist()
            == [False] + [True] * n_replacements
        )
        torch.testing.assert_close(result.forces[:2], resident_forces)
        torch.testing.assert_close(result.energy[:1], resident_energy)
        positions_before = result.positions.clone()
        with (
            patch.object(engine, "_prime_forces", wraps=engine._prime_forces) as prime,
            patch.object(self.model, "forward", wraps=self.model.forward) as forward,
        ):
            engine.step(result)
        assert prime.call_count == 0
        assert forward.call_count == 1
        torch.testing.assert_close(result.positions[:2], positions_before[:2] + 2)
        torch.testing.assert_close(result.positions[2:], positions_before[2:])
        if fused_stage:
            assert (
                result.n_steps_counter_0.view(-1).tolist() == [2] + [0] * n_replacements
            )
        assert not result.reprime_pending.any()
        assert hook.active_masks[-1].tolist() == [True] * result.num_graphs
        assert (
            update_hook.active_masks[-1].tolist() == [True] + [False] * n_replacements
        )
        positions_before = result.positions.clone()
        engine.step(result)
        torch.testing.assert_close(result.positions, positions_before + 2)
        if fused_stage:
            assert (
                result.n_steps_counter_0.view(-1).tolist() == [3] + [1] * n_replacements
            )

    def test_refill_frequency_two_keeps_single_forward_per_step(self) -> None:
        """Steps between refills resume updates without additional priming."""
        sampler = SizeAwareSampler(MockDataset([(2, 0)] * 3), max_batch_size=2)
        hook = _MaskRecordingHook()
        hook.stage = DynamicsStage.BEFORE_PRE_UPDATE
        dynamics = _IncrementDynamics(model=self.model, n_steps=100, hooks=[hook])
        fused = FusedStage(
            sub_stages=[(0, dynamics)],
            sampler=sampler,
            refill_frequency=2,
            device_type="cpu",
        )
        batch = sampler.build_initial_batch()
        batch.status[1] = fused.exit_status

        with (
            patch.object(fused, "_prime_forces", wraps=fused._prime_forces) as prime,
            patch.object(fused, "refill_check", wraps=fused.refill_check) as refill,
            patch.object(self.model, "forward", wraps=self.model.forward) as forward,
        ):
            result = fused.run(batch=batch, n_steps=5)

        assert result is not None
        assert refill.call_count == 2
        assert prime.call_count == 1
        assert forward.call_count == fused.step_count + 1 == 6
        assert result.n_steps_counter_0.view(-1).tolist() == [5, 2]
        assert [mask.tolist() for mask in hook.active_masks[-3:]] == [
            [True, False],
            [True, True],
            [True, True],
        ]

    def test_refill_preserves_survivor_edge_field(self) -> None:
        """Missing replacement edge fields are padded without losing survivor data."""
        data_list = [
            AtomicData(
                atomic_numbers=torch.ones(2, dtype=torch.long),
                positions=torch.randn(2, 3),
                neighbor_list=torch.tensor([[0, 1]]),
            )
            for _ in range(3)
        ]
        for data in data_list[:2]:
            data.add_edge_property("edge_attr", torch.ones(1, 2))
        dynamics = _IncrementDynamics(
            model=self.model, sampler=_MockSampler([data_list[2]]), device_type="cpu"
        )
        batch = Batch.from_data_list(data_list[:2])
        batch["status"] = torch.zeros(2, 1, dtype=torch.long)
        dynamics.step(batch)
        batch.status[0] = dynamics.exit_status

        result = dynamics.refill_check(batch, dynamics.exit_status)

        assert result is not None
        assert "edge_attr" in result
        torch.testing.assert_close(result.edge_attr[:1], torch.ones(1, 2))
        torch.testing.assert_close(result.edge_attr[1:], torch.zeros(1, 2))
        assert result.num_edges == 2

    @pytest.mark.parametrize("fused_stage", [False, True])
    def test_refill_replaces_dataset_forces_before_first_update(
        self, fused_stage: bool
    ) -> None:
        """Incoming dataset forces are refreshed before a replacement advances."""
        replacement = AtomicData(
            atomic_numbers=torch.ones(2, dtype=torch.long),
            positions=torch.ones(2, 3),
            forces=torch.full((2, 3), 1e6),
        )
        dynamics = DemoDynamics(
            model=self.model, n_steps=100, dt=0.1, device_type="cpu"
        )
        engine = (
            FusedStage(sub_stages=[(0, dynamics)], device_type="cpu")
            if fused_stage
            else dynamics
        )
        engine.sampler = _MockSampler([replacement])
        batch = Batch.from_data_list(
            [
                AtomicData(
                    atomic_numbers=torch.ones(2, dtype=torch.long),
                    positions=torch.randn(2, 3),
                )
                for _ in range(2)
            ]
        )
        batch["status"] = torch.zeros(2, 1, dtype=torch.long)
        engine.step(batch)
        batch.status[0] = engine.exit_status
        result = engine.refill_check(batch, engine.exit_status)
        assert result is not None
        torch.testing.assert_close(result.forces[2:], replacement.forces)
        positions_before = result.positions[2:].clone()
        velocities_before = result.velocities[2:].clone()

        with patch.object(engine, "_prime_forces", wraps=engine._prime_forces) as prime:
            engine.step(result)

        assert prime.call_count == 0
        torch.testing.assert_close(result.positions[2:], positions_before)
        torch.testing.assert_close(result.velocities[2:], velocities_before)
        reference = result.clone()
        reference.positions.requires_grad_(True)
        expected_forces = self.model(reference)["forces"].detach()
        torch.testing.assert_close(result.forces[2:], expected_forces[2:])
        expected_positions = (
            positions_before
            + velocities_before * dynamics.dt
            + (
                0.5
                * result.forces[2:]
                / result.atomic_masses[2:].unsqueeze(-1)
                * dynamics.dt**2
            )
        )
        engine.step(result)
        torch.testing.assert_close(result.positions[2:], expected_positions)

    @pytest.mark.parametrize("fused_stage", [False, True])
    def test_refill_preserves_pending_residents(self, fused_stage: bool) -> None:
        """Repeated refill preserves pending residents as graph indices change."""
        sampler = SizeAwareSampler(MockDataset([(2, 0)] * 5), max_batch_size=3)
        dynamics = _IncrementDynamics(model=self.model, n_steps=10)
        engine = FusedStage(sub_stages=[(0, dynamics)]) if fused_stage else dynamics
        engine.sampler = sampler
        batch = sampler.build_initial_batch()
        engine.step(batch)
        batch.reprime_pending = torch.tensor([[False], [True], [False]])
        batch.status[0] = engine.exit_status
        batch = engine.refill_check(batch, engine.exit_status)
        assert batch.reprime_pending.view(-1).tolist() == [True, False, True]
        batch.status[1] = engine.exit_status
        batch = engine.refill_check(batch, engine.exit_status)
        assert batch.reprime_pending.all()
        assert engine._pending_graph_indices.tolist() == [1, -1, -1]
        positions_before = batch.positions.clone()
        engine.step(batch)
        torch.testing.assert_close(batch.positions, positions_before)
        assert not batch.reprime_pending.any()

    def test_refill_applies_bookkeeping_defaults_to_arrivals(self) -> None:
        """Arriving graphs get registered bookkeeping defaults, not zero placeholders."""

        class _IdentityDynamics(_IncrementDynamics):
            """Registers an identity key without changing the shared registry."""

        _IdentityDynamics.register_bookkeeping_key(
            "walker_id",
            lambda n, dev: torch.full((n, 1), -1, dtype=torch.long, device=dev),
        )
        sampler = SizeAwareSampler(MockDataset([(2, 0)] * 4), max_batch_size=3)
        dynamics = _IdentityDynamics(model=self.model, n_steps=10, sampler=sampler)
        batch = sampler.build_initial_batch()
        batch["walker_id"] = torch.tensor([[5], [6], [7]])
        dynamics.step(batch)
        batch.status[0] = dynamics.exit_status
        result = dynamics.refill_check(batch, dynamics.exit_status)
        assert result.reprime_pending.view(-1).tolist() == [False, False, True]
        assert result.walker_id.view(-1).tolist() == [6, 7, -1]

    def test_base_refill_masks_only_the_following_step(self) -> None:
        """Only the step after a refill pays for masked updates."""
        sampler = SizeAwareSampler(MockDataset([(2, 0)] * 3), max_batch_size=2)
        dynamics = _IncrementDynamics(model=self.model, n_steps=10, sampler=sampler)
        batch = sampler.build_initial_batch()
        dynamics.step(batch)
        batch.status[0] = dynamics.exit_status
        batch = dynamics.refill_check(batch, dynamics.exit_status)

        with (
            patch.object(
                dynamics, "_masked_pre_update", wraps=dynamics._masked_pre_update
            ) as masked_pre,
            patch.object(
                dynamics, "_masked_post_update", wraps=dynamics._masked_post_update
            ) as masked_post,
        ):
            dynamics.step(batch)
            dynamics.step(batch)

        assert masked_pre.call_count == 1
        assert masked_post.call_count == 1
        assert "reprime_pending" in batch
        assert not dynamics._refill_pending

    def test_fused_pipeline_full_drain_resets_batch_initialization(self) -> None:
        """A rebuilt FusedStage pipeline batch is admitted and fully primed."""
        dataset = MockDataset([(2, 0)] * 4)
        sampler = SizeAwareSampler(dataset, max_atoms=4, max_edges=0, max_batch_size=2)
        dynamics = _IncrementDynamics(model=self.model)
        stage = FusedStage(
            sub_stages=[(0, dynamics)],
            sampler=sampler,
            prior_rank=None,
            next_rank=1,
            buffer_config=BufferConfig(
                num_systems=2,
                num_nodes=4,
                num_edges=0,
            ),
        )
        stage.active_batch = initialize_batch_for_dynamics(
            sampler.build_initial_batch()
        )
        stage.active_batch["status"] = torch.zeros(2, 1, dtype=torch.long)
        stepped_batch = stage.active_batch
        stage._admission_initialized = True
        stage._forces_primed = True
        pipeline = DistributedPipeline(stages={0: stage, 1: stage})

        def drain_active_batch(converged: torch.Tensor | None) -> None:
            assert converged is not None
            stage.active_batch = None

        with (
            patch("nvalchemi.dynamics.base.dist.is_initialized", return_value=True),
            patch("nvalchemi.dynamics.base.dist.get_rank", return_value=0),
            patch.object(
                stage,
                "step",
                return_value=(stepped_batch, torch.tensor([0, 1])),
            ),
            patch.object(
                stage,
                "_poststep_sync_buffers",
                side_effect=drain_active_batch,
            ),
        ):
            pipeline.step()

        assert stage.active_batch is not None
        assert getattr(stage.active_batch, "reprime_pending", None) is None
        assert getattr(stage.active_batch, "forces", None) is None
        assert getattr(stage.active_batch, "energy", None) is None
        assert stage._admission_initialized is False
        assert stage._forces_primed is False

    def test_base_pipeline_full_drain_resets_batch_initialization(self) -> None:
        """A rebuilt BaseDynamics pipeline batch is admitted and fully primed."""
        dataset = MockDataset([(2, 0)] * 4)
        sampler = SizeAwareSampler(dataset, max_atoms=4, max_edges=0, max_batch_size=2)
        stage = _IncrementDynamics(
            model=self.model,
            sampler=sampler,
            prior_rank=None,
            next_rank=1,
            buffer_config=BufferConfig(
                num_systems=2,
                num_nodes=4,
                num_edges=0,
            ),
        )
        stage.active_batch = initialize_batch_for_dynamics(
            sampler.build_initial_batch()
        )
        stepped_batch = stage.active_batch
        stage._admission_initialized = True
        stage._forces_primed = True
        pipeline = DistributedPipeline(stages={0: stage, 1: stage})

        def drain_active_batch(converged: torch.Tensor | None) -> None:
            assert converged is not None
            stage.active_batch = None

        with (
            patch("nvalchemi.dynamics.base.dist.is_initialized", return_value=True),
            patch("nvalchemi.dynamics.base.dist.get_rank", return_value=0),
            patch.object(
                stage,
                "step",
                return_value=(stepped_batch, torch.tensor([0, 1])),
            ),
            patch.object(
                stage,
                "_poststep_sync_buffers",
                side_effect=drain_active_batch,
            ),
        ):
            pipeline.step()

        assert stage.active_batch is not None
        assert getattr(stage.active_batch, "reprime_pending", None) is None
        assert getattr(stage.active_batch, "forces", None) is None
        assert getattr(stage.active_batch, "energy", None) is None
        assert stage._admission_initialized is False
        assert stage._forces_primed is False

    def test_refill_clears_stale_last_converged(self) -> None:
        """A graduation that shrinks the batch must drop stale ``_last_converged``.

        Otherwise the next ``_build_context`` masks the smaller batch with indices
        from the larger one and over-indexes (CUDA index-out-of-bounds; IndexError
        on CPU).
        """
        # Exactly enough samples for the initial batch → no replacements, so the
        # batch shrinks on graduation.
        dataset = MockDataset([(2, 0)] * 4)
        sampler = SizeAwareSampler(
            dataset, max_atoms=20, max_edges=10, max_batch_size=4
        )
        dynamics = BaseDynamics(model=self.model, sampler=sampler, device_type="cpu")

        batch = sampler.build_initial_batch()
        initialize_batch_for_dynamics(batch)
        batch["status"] = torch.tensor([[0], [0], [1], [1]])
        # Stale indices pointing at the graduating (soon-removed) systems.
        dynamics._last_converged = torch.tensor([2, 3])

        result = dynamics.refill_check(batch, exit_status=1)

        assert result is not None and result.num_graphs == 2
        assert dynamics._last_converged is None
        # Must not over-index the shrunk batch.
        ctx = dynamics._build_context(result)
        assert ctx.converged_mask is None

    def test_send_path_realigns_state_and_last_converged(self) -> None:
        """Sending converged graphs trims the active batch, so per-system state
        and stale converged indices must shrink with it — else the next step
        over-indexes them (the batch_size=2 crash in example 02).
        """
        from nvalchemi.dynamics import FIRE
        from nvalchemi.dynamics.base import BufferConfig

        dataset = MockDataset([(3, 0)] * 4)
        sampler = SizeAwareSampler(
            dataset, max_atoms=40, max_edges=20, max_batch_size=4
        )
        fire = FIRE(
            model=self.model,
            dt=1.0,
            n_steps=50,
            sampler=sampler,
            next_rank=1,
            buffer_config=BufferConfig(num_systems=4, num_nodes=12, num_edges=0),
            device_type="cpu",
        )
        batch = sampler.build_initial_batch()
        initialize_batch_for_dynamics(batch)
        batch, _ = fire.step(batch)  # initializes per-system _state (size 4)
        assert fire._state.num_graphs == 4

        fire.active_batch = batch
        fire._ensure_buffers(batch)  # allocate the send buffer
        fire._last_converged = torch.tensor([0, 3])  # stale; references index 3

        fire._populate_send_buffer(torch.tensor([0, 1]))  # send graphs 0, 1

        assert fire.active_batch.num_graphs == 2
        assert fire._state.num_graphs == 2
        assert fire._last_converged is None
        ctx = fire._build_context(fire.active_batch)
        assert ctx.converged_mask is None

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
# TestPipelinePerGraphPriming
# -----------------------------------------------------------------------------


class TestPipelinePerGraphPriming:
    """Pipeline membership changes keep per-graph force priming valid."""

    def setup_method(self) -> None:
        """Set up test fixtures before each test method."""
        self.model = DemoModelWrapper(DemoModel())

    def _make_stage(
        self,
        fused_stage: bool,
        sampler_batch_size: int = 3,
        reprime_on_entry: bool = False,
        **kwargs,
    ) -> tuple[BaseDynamics, SizeAwareSampler, DistributedPipeline]:
        """Build a CPU pipeline stage and a sampler of two-atom graphs."""
        sampler = SizeAwareSampler(
            MockDataset([(2, 0)] * 8), max_batch_size=sampler_batch_size
        )
        config = BufferConfig(num_systems=1, num_nodes=2, num_edges=0)
        if fused_stage:
            dynamics = _IncrementDynamics(
                model=self.model, device_type="cpu", reprime_on_entry=reprime_on_entry
            )
            stage = FusedStage(
                sub_stages=[(0, dynamics)],
                buffer_config=config,
                device_type="cpu",
                **kwargs,
            )
        else:
            stage = _IncrementDynamics(
                model=self.model,
                buffer_config=config,
                device_type="cpu",
                reprime_on_entry=reprime_on_entry,
                **kwargs,
            )
        upstream = BaseDynamics(
            model=self.model, buffer_config=config, device_type="cpu"
        )
        pipeline = DistributedPipeline(stages={0: upstream, 1: stage})
        pipeline.setup()
        return stage, sampler, pipeline

    def _step_counting_calls(
        self, stage: BaseDynamics, batch: Batch
    ) -> tuple[int, int]:
        """Step once and return ``(_prime_forces calls, model forward calls)``."""
        with (
            patch.object(stage, "_prime_forces", wraps=stage._prime_forces) as prime,
            patch.object(self.model, "forward", wraps=self.model.forward) as forward,
        ):
            stage.step(batch)
        return prime.call_count, forward.call_count

    @pytest.mark.parametrize("fused_stage", [False, True])
    @pytest.mark.parametrize("final_stage", [False, True])
    def test_removal_keeps_retained_forces(
        self, fused_stage: bool, final_stage: bool
    ) -> None:
        """Sending or retiring graphs needs no whole-batch reprime."""
        stage, sampler, pipeline = self._make_stage(
            fused_stage, prior_rank=0, next_rank=None if final_stage else 2
        )
        batch = sampler.build_initial_batch()
        stage.active_batch = batch
        stage._ensure_buffers(batch)
        stage.step(batch)
        # Graph 1 is still waiting for its initial forces from a refill.
        batch["reprime_pending"] = torch.tensor([[False], [True], [False]])
        stage._refill_pending = True
        resident_forces = batch.forces[2:].clone()

        if final_stage:
            stage._remove_converged_final_stage(torch.tensor([0]))
        else:
            stage._populate_send_buffer(torch.tensor([0]))

        result = stage.active_batch
        assert stage._forces_primed is True
        assert result.reprime_pending.view(-1).tolist() == [True, False]
        torch.testing.assert_close(result.forces, resident_forces)
        positions_before = result.positions.clone()
        assert self._step_counting_calls(stage, result) == (0, 1)
        torch.testing.assert_close(result.positions[:2], positions_before[:2])
        torch.testing.assert_close(result.positions[2:], positions_before[2:] + 2)

    @pytest.mark.parametrize("fused_stage", [False, True])
    def test_receive_primes_only_arrivals(self, fused_stage: bool) -> None:
        """Received graphs skip one update and get forces from the next compute."""
        stage, sampler, pipeline = self._make_stage(
            fused_stage, sampler_batch_size=2, prior_rank=0, max_batch_size=4
        )
        batch = sampler.build_initial_batch()
        stage.active_batch = batch
        stage.step(batch)
        batch["label"] = torch.tensor([[5], [6]])
        resident_forces = batch.forces.clone()
        incoming = sampler.build_initial_batch()
        # No upstream forces are available, so arrivals need local initialization.
        stage._buffer_to_batch(incoming)

        result = stage.active_batch
        assert result.num_graphs == 4
        assert stage._forces_primed is True
        assert result.reprime_pending.view(-1).tolist() == [False, False, True, True]
        assert result.label.view(-1).tolist() == [5, 6, 0, 0]
        torch.testing.assert_close(result.forces[:4], resident_forces)
        positions_before = result.positions.clone()
        assert self._step_counting_calls(stage, result) == (0, 1)
        torch.testing.assert_close(result.positions[:4], positions_before[:4] + 2)
        torch.testing.assert_close(result.positions[4:], positions_before[4:])
        assert not (result.forces[4:] == 99.0).any()
        assert not result.reprime_pending.any()
        positions_before = result.positions.clone()
        stage.step(result)
        torch.testing.assert_close(result.positions, positions_before + 2)

    @pytest.mark.parametrize("fused_stage", [False, True])
    @pytest.mark.parametrize("reprime_on_entry", [False, True])
    @pytest.mark.parametrize("has_residents", [False, True])
    @pytest.mark.parametrize("comm_mode", ["sync", "async_recv", "fully_async"])
    def test_receive_reprime_policy(
        self,
        fused_stage: bool,
        reprime_on_entry: bool,
        has_residents: bool,
        comm_mode: str,
    ) -> None:
        """Entry reuses upstream forces or primes arrivals without extra compute."""
        stage, sampler, pipeline = self._make_stage(
            fused_stage,
            sampler_batch_size=2,
            reprime_on_entry=reprime_on_entry,
            prior_rank=0,
            max_batch_size=4,
            comm_mode=comm_mode,
        )
        if has_residents:
            batch = sampler.build_initial_batch()
            stage.active_batch = batch
            stage.step(batch)
        incoming = initialize_batch_for_dynamics(
            sampler.build_initial_batch().index_select([0])
        )
        incoming.forces.fill_(99.0)
        stage._ensure_buffers(stage.active_batch if has_residents else incoming)
        seen_forces = []
        stage.register_hook(
            Mock(
                spec=["stage", "frequency"],
                stage=DynamicsStage.BEFORE_PRE_UPDATE,
                frequency=1,
                side_effect=lambda ctx, stage: seen_forces.append(
                    ctx.batch.forces.clone()
                ),
            )
        )
        with patch.object(Batch, "irecv", return_value=Mock(wait=lambda: incoming)):
            stage._prestep_sync_buffers()
            stage._complete_pending_recv()

        result = stage.active_batch
        assert result.num_graphs == (3 if has_residents else 1)
        positions_before = result.positions.clone()
        # Only an empty stage needs a whole-batch prime under the opt-in.
        full_prime = reprime_on_entry and not has_residents
        assert self._step_counting_calls(stage, result) == (
            (1, 2) if full_prime else (0, 1)
        )
        assert bool((seen_forces[0][-2:] == 99.0).all()) is not full_prime
        # With primed residents, the arrival skips one update instead.
        arrival_shift = 0 if reprime_on_entry and has_residents else 2
        torch.testing.assert_close(result.positions[:-2], positions_before[:-2] + 2)
        torch.testing.assert_close(
            result.positions[-2:], positions_before[-2:] + arrival_shift
        )
        if fused_stage:
            assert not result.reprime_pending.any()
        # The opt-in is admission-driven, not an extra compute on every step.
        assert self._step_counting_calls(stage, result) == (0, 1)

    @pytest.mark.parametrize("reprime_on_entry", [False, True])
    def test_reprime_policy_through_gloo_handoff(self, reprime_on_entry: bool) -> None:
        """Real inter-rank handoff preserves forces and applies the entry policy."""
        results = run_gloo(
            world_size=2,
            fn=_pipeline_entry_worker,
            args=(reprime_on_entry,),
        )
        assert results == [
            (
                int(reprime_on_entry),
                2 if reprime_on_entry else 1,
                not reprime_on_entry,
                True,
            )
        ]

    @pytest.mark.parametrize("fused_stage", [False, True])
    @pytest.mark.parametrize("has_residents", [False, True])
    def test_receive_missing_required_output_initializes_locally(
        self, fused_stage: bool, has_residents: bool
    ) -> None:
        """An upstream force tensor alone cannot replace a required missing output."""
        stage, sampler, pipeline = self._make_stage(fused_stage, sampler_batch_size=2)
        stage.__needs_keys__ = {"forces", "energy"}
        if has_residents:
            stage.active_batch = sampler.build_initial_batch()
            stage.step(stage.active_batch)
        incoming = sampler.build_initial_batch()
        incoming["forces"] = torch.full((incoming.num_nodes, 3), 99.0)
        if "energy" in incoming:
            del incoming["energy"]
        stage._buffer_to_batch(incoming)

        result = stage.active_batch
        before = result.positions.clone()
        assert self._step_counting_calls(stage, result) == (
            (0, 1) if has_residents else (1, 2)
        )
        if has_residents:
            torch.testing.assert_close(result.positions[-4:], before[-4:])
        else:
            torch.testing.assert_close(result.positions, before + 2)
        assert not (result.forces[-4:] == 99.0).any()
        assert "energy" in result

    @pytest.mark.parametrize("fused_stage", [False, True])
    @pytest.mark.parametrize("reprime_on_entry", [False, True])
    @pytest.mark.parametrize("final_stage", [False, True])
    def test_removal_does_not_reprime_residents(
        self, fused_stage: bool, reprime_on_entry: bool, final_stage: bool
    ) -> None:
        """Removal keeps residents' forces under either entry policy."""
        stage, sampler, pipeline = self._make_stage(
            fused_stage,
            reprime_on_entry=reprime_on_entry,
            prior_rank=0,
            next_rank=None if final_stage else 2,
        )
        generation = [0]
        seen_forces = []

        def admission(ctx: DynamicsContext, stage: DynamicsStage) -> None:
            """Change the workflow's force configuration on each admission."""
            generation[0] += 1

        def force_hook(ctx: DynamicsContext, stage: DynamicsStage) -> None:
            """Apply the configuration selected by the latest admission."""
            ctx.batch.forces.fill_(float(generation[0]))

        for boundary, callback in (
            (DynamicsStage.ON_ADMISSION, admission),
            (DynamicsStage.AFTER_COMPUTE, force_hook),
            (
                DynamicsStage.BEFORE_PRE_UPDATE,
                lambda ctx, stage: seen_forces.append(ctx.batch.forces.clone()),
            ),
        ):
            stage.register_hook(
                Mock(
                    spec=["stage", "frequency"],
                    stage=boundary,
                    frequency=1,
                    side_effect=callback,
                )
            )
        batch = sampler.build_initial_batch()
        stage.active_batch = batch
        stage.step(batch)
        stage._ensure_buffers(batch)
        if final_stage:
            stage._remove_converged_final_stage(torch.tensor([0]))
        else:
            stage._populate_send_buffer(torch.tensor([0]))

        assert self._step_counting_calls(stage, stage.active_batch) == (0, 1)
        assert generation[0] == 2
        torch.testing.assert_close(seen_forces[-1], torch.ones_like(seen_forces[-1]))

    @pytest.mark.parametrize("fused_stage", [False, True])
    def test_receive_into_unprimed_stage_primes_whole_batch(
        self, fused_stage: bool
    ) -> None:
        """Without primed residents, the next step primes every graph."""
        stage, sampler, pipeline = self._make_stage(
            fused_stage, sampler_batch_size=2, prior_rank=0, max_batch_size=4
        )
        stage.active_batch = initialize_batch_for_dynamics(
            sampler.build_initial_batch()
        )
        stage._buffer_to_batch(
            initialize_batch_for_dynamics(sampler.build_initial_batch())
        )

        assert stage.active_batch.num_graphs == 4
        assert stage._forces_primed is False
        assert self._step_counting_calls(stage, stage.active_batch) == (1, 2)

    @pytest.mark.parametrize("fused_stage", [False, True])
    @pytest.mark.parametrize("reprime_on_entry", [False, True])
    def test_pipeline_refill_keeps_fields_missing_from_arrivals(
        self, fused_stage: bool, reprime_on_entry: bool
    ) -> None:
        """Refill retains resident fields and honors the pipeline admission policy."""
        stage, sampler, pipeline = self._make_stage(
            fused_stage,
            reprime_on_entry=reprime_on_entry,
            next_rank=1,
            sampler=None,
        )
        stage.sampler = sampler
        batch = sampler.build_initial_batch()
        stage.active_batch = batch
        stage._ensure_buffers(batch)
        stage.step(batch)
        batch["label"] = torch.tensor([[5], [6], [7]])
        # Graphs 0 and 1 graduate, but the send buffer only holds one.
        batch.status[:2] = stage.exit_status
        stage._populate_send_buffer(torch.tensor([0]))

        result = stage.refill_check(stage.active_batch, stage.exit_status)

        assert stage._forces_primed is True
        assert result.reprime_pending.view(-1).tolist() == [False, True, True]
        assert result.label.view(-1).tolist() == [7, 0, 0]
        positions_before = result.positions.clone()
        assert self._step_counting_calls(stage, result) == (0, 1)
        torch.testing.assert_close(result.positions[:2], positions_before[:2] + 2)
        torch.testing.assert_close(result.positions[2:], positions_before[2:])


# -----------------------------------------------------------------------------
# TestRefillCheck
# -----------------------------------------------------------------------------


class TestRefillCheck:
    """Unit tests for BaseDynamics.refill_check()."""

    def setup_method(self) -> None:
        """Set up test fixtures before each test method."""
        self.model = DemoModelWrapper(DemoModel())

    @pytest.mark.parametrize("primed", [False, True])
    def test_refill_preserves_flat_bookkeeping(self, primed: bool) -> None:
        """Flat resident bookkeeping is normalized before appending defaults."""
        sampler = SizeAwareSampler(MockDataset([(2, 0)] * 4), max_batch_size=2)
        dynamics = BaseDynamics(model=self.model, sampler=sampler, device_type="cpu")
        batch = sampler.build_initial_batch()
        batch["status"] = torch.tensor([0, 1])
        batch["system_id"] = torch.tensor([17, 18])
        dynamics._forces_primed = primed

        result = dynamics.refill_check(batch, exit_status=1)

        assert result.status.tolist() == [[0], [0]]
        assert result.system_id.tolist() == [[17], [2]]
        assert result.reprime_pending.view(-1).tolist() == [False, primed]

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

        dynamics._ensure_admission_initialized(batch)
        dynamics._record_batch_change(batch, torch.arange(3))
        # Mark all as graduated
        batch["status"] = torch.tensor([[1], [1], [1]])

        result = dynamics.refill_check(batch, exit_status=1)

        # No replacements available, no remaining
        assert result is None
        assert dynamics.done is True
        assert dynamics._pending_graph_indices is None

    def test_refill_raises_when_pending_samples_cannot_fit_empty_batch(
        self,
    ) -> None:
        """An unfillable empty batch raises instead of reporting completion."""
        dataset = MockDataset([(2, 0)] * 3)
        sampler = SizeAwareSampler(
            dataset, max_atoms=20, max_edges=10, max_batch_size=1
        )
        dynamics = BaseDynamics(model=self.model, sampler=sampler, device_type="cpu")
        batch = sampler.build_initial_batch()
        batch["status"] = torch.tensor([[1]])
        with (
            patch.object(sampler, "request_replacements_budget", return_value=[]),
            pytest.raises(ValueError, match="empty active batch"),
        ):
            dynamics.refill_check(batch, exit_status=1)
        assert not dynamics.done

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

        dynamics._ensure_admission_initialized(batch)
        result = dynamics.refill_check(batch, exit_status=1)

        # Should return same batch unchanged
        assert result is batch
        assert dynamics._admission_initialized
        assert dynamics._pending_graph_indices is None

    def test_compute_detaches_live_inputs_before_refill(self) -> None:
        """Autograd inputs must be detached after compute() so refill is clean.

        Autograd-force models enable gradients on live batch inputs (e.g.
        positions) during compute().  Those must be cleared before
        refill_check performs index_select / append, otherwise the rebuilt
        batch tensors become IndexBackward / CatBackward nodes.
        """
        dataset = MockDataset([(2, 0)] * 10)
        sampler = SizeAwareSampler(
            dataset, max_atoms=20, max_edges=10, max_batch_size=5
        )

        dynamics = BaseDynamics(model=self.model, sampler=sampler, device_type="cpu")

        batch = sampler.build_initial_batch()
        dynamics.compute(batch)

        # compute() must clear requires_grad on positions
        assert batch.positions.requires_grad is False
        assert batch.positions.is_leaf is True
        assert batch.positions.grad_fn is None

        # Now run refill — result tensors should also be graph-free
        batch["status"] = torch.tensor([[1], [0], [0], [0], [0]])
        result = dynamics.refill_check(batch, exit_status=1)

        assert result is not None
        assert result.positions.requires_grad is False
        assert result.positions.is_leaf is True
        assert result.positions.grad_fn is None

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

    def test_refill_preserves_replacement_system_id(self) -> None:
        """Replacement systems keep sampler-assigned system IDs after refill."""
        dataset = MockDataset([(1, 0)] * 3)
        sampler = SizeAwareSampler(dataset, max_atoms=2)
        dynamics = BaseDynamics(model=self.model, sampler=sampler, device_type="cpu")

        batch = sampler.build_initial_batch()
        assert batch.system_id.view(-1).tolist() == [0, 1]

        batch["status"] = torch.tensor([[1], [1]])
        result = dynamics.refill_check(batch, exit_status=1)

        assert result is not None
        assert result.system_id.view(-1).tolist() == [2]
        assert result.status.view(-1).tolist() == [0]

    def test_refill_preserves_mixed_system_ids(self) -> None:
        """Remaining and replacement systems both keep their system IDs."""
        dataset = MockDataset([(1, 0)] * 3)
        sampler = SizeAwareSampler(dataset, max_atoms=2)
        dynamics = BaseDynamics(model=self.model, sampler=sampler, device_type="cpu")

        batch = sampler.build_initial_batch()
        assert batch.system_id.view(-1).tolist() == [0, 1]

        batch["status"] = torch.tensor([[1], [0]])
        result = dynamics.refill_check(batch, exit_status=1)

        assert result is not None
        assert result.system_id.view(-1).tolist() == [1, 2]
        assert result.status.view(-1).tolist() == [0, 0]

    def test_refill_preserves_system_ids_with_multistage_statuses(self) -> None:
        """A status-2 system is replaced while lower-status systems remain."""
        dataset = MockDataset([(1, 0)] * 5)
        sampler = SizeAwareSampler(dataset, max_atoms=3)
        dynamics = BaseDynamics(model=self.model, sampler=sampler, device_type="cpu")

        batch = sampler.build_initial_batch()
        assert batch.system_id.view(-1).tolist() == [0, 1, 2]

        batch["status"] = torch.tensor([[0], [1], [2]])
        result = dynamics.refill_check(batch, exit_status=2)

        assert result is not None
        assert result.system_id.view(-1).tolist() == [0, 1, 3]
        assert result.status.view(-1).tolist() == [0, 1, 0]

    def test_refill_preserves_system_ids_with_multiple_replacements(self) -> None:
        """Multiple graduated systems are replaced in a single refill."""
        dataset = MockDataset([(1, 0)] * 5)
        sampler = SizeAwareSampler(dataset, max_atoms=3)
        dynamics = BaseDynamics(model=self.model, sampler=sampler, device_type="cpu")

        batch = sampler.build_initial_batch()
        assert batch.system_id.view(-1).tolist() == [0, 1, 2]

        batch["status"] = torch.tensor([[1], [0], [1]])
        result = dynamics.refill_check(batch, exit_status=1)

        assert result is not None
        assert result.system_id.view(-1).tolist() == [1, 3, 4]
        assert result.status.view(-1).tolist() == [0, 0, 0]

    def test_refill_preserves_replacement_status_from_dataset(self) -> None:
        """Replacement systems keep nonzero dataset-provided entry status."""

        class StatusDataset(MockDataset):
            def __getitem__(self, index: int) -> tuple[AtomicData, dict]:
                data, metadata = super().__getitem__(index)
                data.add_system_property(
                    "status", torch.tensor([[1]], dtype=torch.long)
                )
                return data, metadata

        dataset = StatusDataset([(1, 0)] * 5)
        sampler = SizeAwareSampler(dataset, max_atoms=3)
        dynamics = BaseDynamics(model=self.model, sampler=sampler, device_type="cpu")

        batch = sampler.build_initial_batch()
        assert batch.system_id.view(-1).tolist() == [0, 1, 2]
        # Initial batching resets status to zero, but refill preserves
        # replacement status already carried by the appended batch.
        assert batch.status.view(-1).tolist() == [0, 0, 0]

        batch["status"] = torch.tensor([[0], [2], [1]])
        result = dynamics.refill_check(batch, exit_status=2)

        assert result is not None
        assert result.system_id.view(-1).tolist() == [0, 2, 3]
        assert result.status.view(-1).tolist() == [0, 1, 1]

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
        self.model = DemoModelWrapper(DemoModel())

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

        Because neither target requires reprime, both convergence hooks may
        migrate a graph sequentially after the same shared model computation.
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

        # Neither stage requires reprime, so status should migrate 0 -> 1 -> 2 after 1 step
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
        self.model = DemoModelWrapper(DemoModel())

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
