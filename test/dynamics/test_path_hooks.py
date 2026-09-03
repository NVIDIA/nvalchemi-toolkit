# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for hooks shared by reaction-path dynamics methods."""

from __future__ import annotations

import pytest
import torch

from nvalchemi.data import AtomicData, Batch
from nvalchemi.dynamics import DynamicsStage
from nvalchemi.dynamics.hooks import LoggingHook
from nvalchemi.dynamics.paths.hooks import (
    PathDiagnosticsHook,
    PathEnergyStatsHook,
)
from nvalchemi.hooks import DynamicsContext


@pytest.fixture
def path() -> Batch:
    """Build a four-image, one-atom path."""
    batch = Batch.from_data_list(
        [
            AtomicData(
                atomic_numbers=torch.tensor([1]),
                positions=torch.tensor([[float(i), 0.0, 0.0]]),
                energy=torch.tensor([[energy]]),
                forces=torch.tensor([[float(i + 1), 0.0, 0.0]]),
            )
            for i, energy in enumerate([0.0, 2.0, 1.0, 4.0])
        ]
    )
    batch.set_group_layout(torch.zeros(4, dtype=torch.long))
    return batch


class TestPathEnergyStatsHook:
    """Exercise reusable per-path energy reductions."""

    def test_computes_ragged_path_energy_stats(self) -> None:
        batch = Batch.from_data_list(
            [
                AtomicData(
                    atomic_numbers=torch.tensor([1]),
                    positions=torch.tensor([[float(i), 0.0, 0.0]]),
                    energy=torch.tensor([[energy]]),
                    forces=torch.tensor([[float(i + 1), 0.0, 0.0]]),
                )
                for i, energy in enumerate(
                    [5.0, 3.0, 3.0, 1.0, 2.0, 4.0, 7.0, 6.0, 8.0]
                )
            ]
        )
        batch.set_group_layout(torch.tensor([0] * 4 + [1] * 5))
        hook = PathEnergyStatsHook()
        ctx = DynamicsContext(
            batch=batch,
            active_graph_mask=torch.ones(
                batch.num_graphs, dtype=torch.bool, device=batch.device
            ),
            step_count=0,
        )

        hook(ctx, DynamicsStage.ON_ADMISSION)
        hook(ctx, DynamicsStage.AFTER_COMPUTE)
        stats = hook.get_stats()

        assert torch.equal(
            stats.endpoint_reference_energy,
            torch.tensor([1.0, 2.0]),
        )
        assert torch.equal(
            stats.highest_interior_energy,
            torch.tensor([3.0, 7.0]),
        )
        assert torch.equal(
            stats.highest_interior_image_idx,
            torch.tensor([1, 6], dtype=torch.int32),
        )

    def test_after_compute_compiles_with_cuda_graphs(self, gpu_device: str) -> None:
        batch = Batch.from_data_list(
            [
                AtomicData(
                    atomic_numbers=torch.tensor([1]),
                    positions=torch.tensor([[float(i), 0.0, 0.0]]),
                    energy=torch.tensor([[energy]]),
                    forces=torch.zeros(1, 3),
                )
                for i, energy in enumerate([0.0, 2.0, 1.0, 4.0])
            ]
        ).to(gpu_device)
        batch.set_group_layout(torch.zeros(4, dtype=torch.long, device=gpu_device))
        hook = PathEnergyStatsHook()
        ctx = DynamicsContext(
            batch=batch,
            active_graph_mask=torch.ones(
                batch.num_graphs, dtype=torch.bool, device=gpu_device
            ),
        )
        hook(ctx, DynamicsStage.ON_ADMISSION)
        compiled = torch.compile(hook, backend="cudagraphs", fullgraph=True)

        try:
            batch.energy.copy_(
                torch.tensor([[3.0], [1.0], [5.0], [2.0]], device=gpu_device)
            )
            ctx.step_count = 1
            compiled(ctx, DynamicsStage.AFTER_COMPUTE)
            torch.cuda.synchronize()

            stats = hook.get_stats()
            assert stats.endpoint_reference_energy.tolist() == [2.0]
            assert stats.highest_interior_energy.tolist() == [5.0]
            assert stats.highest_interior_image_idx.tolist() == [2]
        finally:
            torch.compiler.reset()

    def test_refreshes_preallocated_buffers(self, path: Batch) -> None:
        batch = path
        hook = PathEnergyStatsHook()
        ctx = DynamicsContext(
            batch=batch,
            active_graph_mask=torch.ones(
                batch.num_graphs, dtype=torch.bool, device=batch.device
            ),
            step_count=0,
        )
        hook(ctx, DynamicsStage.ON_ADMISSION)
        hook(ctx, DynamicsStage.AFTER_COMPUTE)
        stats = hook.get_stats()
        pointers = (
            stats.endpoint_reference_energy.data_ptr(),
            stats.highest_interior_energy.data_ptr(),
            stats.highest_interior_image_idx.data_ptr(),
        )

        batch.energy.copy_(torch.tensor([[3.0], [1.0], [5.0], [2.0]]))
        ctx.step_count = 1
        hook(ctx, DynamicsStage.AFTER_COMPUTE)
        refreshed = hook.get_stats()

        assert pointers == (
            refreshed.endpoint_reference_energy.data_ptr(),
            refreshed.highest_interior_energy.data_ptr(),
            refreshed.highest_interior_image_idx.data_ptr(),
        )
        assert refreshed.endpoint_reference_energy.tolist() == [2.0]
        assert refreshed.highest_interior_energy.tolist() == [5.0]
        assert refreshed.highest_interior_image_idx.tolist() == [2]

    def test_rejects_access_before_admission(self) -> None:
        hook = PathEnergyStatsHook()

        with pytest.raises(RuntimeError, match="ON_ADMISSION"):
            hook.get_stats()


_DIAGNOSTIC_ENERGIES = [3.0, 8.0, 5.0, 4.0, 2.0, 3.0, 7.0, 6.0, 1.0]
_DIAGNOSTIC_GROUPS = [0] * 4 + [1] * 5
_DIAGNOSTIC_LINKS = [1.0, 2.0, 3.0, 0.0, 4.0, 5.0, 6.0, 7.0, 0.0]


def _diagnostic_path(
    energies: list[float],
    groups: list[int],
    link_lengths: list[float] | None = None,
    *,
    device: str = "cpu",
) -> Batch:
    """Build a grouped path with deterministic energies and forces."""
    batch = Batch.from_data_list(
        [
            AtomicData(
                atomic_numbers=torch.tensor([1], device=device),
                positions=torch.tensor([[float(i), 0.0, 0.0]], device=device),
                energy=torch.tensor([[energy]], device=device),
                forces=torch.tensor([[float(i + 1), 0.0, 0.0]], device=device),
            )
            for i, energy in enumerate(energies)
        ]
    )
    batch.set_group_layout(torch.tensor(groups, device=device))
    if link_lengths is not None:
        batch.forward_link_length = torch.tensor(link_lengths, device=device)
    return batch


def _prepare_diagnostics(
    *, device: str = "cpu", eager_refresh: bool = True
) -> tuple[Batch, DynamicsContext, PathEnergyStatsHook, PathDiagnosticsHook]:
    """Prepare a two-path diagnostics stack with deterministic values."""
    batch = _diagnostic_path(
        _DIAGNOSTIC_ENERGIES,
        _DIAGNOSTIC_GROUPS,
        _DIAGNOSTIC_LINKS,
        device=device,
    )
    stats_hook = PathEnergyStatsHook()
    diagnostics_hook = PathDiagnosticsHook(energy_stats_hook=stats_hook)
    ctx = DynamicsContext(
        batch=batch,
        active_graph_mask=torch.ones(
            batch.num_graphs,
            dtype=torch.bool,
            device=batch.device,
        ),
    )

    stats_hook(ctx, DynamicsStage.ON_ADMISSION)
    diagnostics_hook(ctx, DynamicsStage.ON_ADMISSION)
    if eager_refresh:
        stats_hook(ctx, DynamicsStage.AFTER_COMPUTE)
        diagnostics_hook(ctx, DynamicsStage.AFTER_COMPUTE)
    return batch, ctx, stats_hook, diagnostics_hook


class TestPathDiagnosticsHook:
    """Exercise reusable diagnostics for reaction paths."""

    def test_requires_link_lengths_at_admission(self) -> None:
        batch = _diagnostic_path([0.0, 1.0, 0.0], [0, 0, 0])
        hook = PathDiagnosticsHook(energy_stats_hook=PathEnergyStatsHook())

        with pytest.raises(RuntimeError, match="forward_link_length.*ON_ADMISSION"):
            hook(DynamicsContext(batch=batch), DynamicsStage.ON_ADMISSION)

    def test_computes_core_diagnostics(self) -> None:
        _, ctx, _, hook = _prepare_diagnostics()

        diagnostics = hook.get_diagnostics()

        assert torch.equal(diagnostics.fmax, torch.tensor([4.0, 9.0]))
        assert torch.equal(diagnostics.energy_barrier, torch.tensor([5.0, 6.0]))
        assert torch.equal(diagnostics.path_length, torch.tensor([6.0, 22.0]))
        assert torch.equal(
            diagnostics.highest_interior_image_idx,
            torch.tensor([1, 2], dtype=torch.int32),
        )

    def test_logs_one_row_per_path_with_logging_hook(self) -> None:
        _, ctx, _, diagnostics_hook = _prepare_diagnostics()
        captured: list[tuple[int, list[dict[str, float]]]] = []

        def writer(step: int, rows: list[dict[str, float]]) -> None:
            captured.append((step, rows))

        logging_hook = LoggingHook(
            backend="custom",
            writer_fn=writer,
            stage=DynamicsStage.AFTER_COMPUTE,
            by_group=True,
            custom_scalars={
                "fmax": lambda ctx: diagnostics_hook.get_diagnostics().fmax,
                "energy_barrier": lambda ctx: diagnostics_hook.get_diagnostics().energy_barrier,
                "path_length": (
                    lambda ctx: diagnostics_hook.get_diagnostics().path_length
                ),
                "highest_interior_image_idx": (
                    lambda ctx: diagnostics_hook.get_diagnostics().highest_interior_image_idx
                ),
            },
        )

        with logging_hook:
            logging_hook(ctx, DynamicsStage.AFTER_COMPUTE)

        assert captured == [
            (
                0,
                [
                    {
                        "step": 0.0,
                        "group_idx": 0.0,
                        "status": 0.0,
                        "fmax": 4.0,
                        "energy_barrier": 5.0,
                        "path_length": 6.0,
                        "highest_interior_image_idx": 1.0,
                    },
                    {
                        "step": 0.0,
                        "group_idx": 1.0,
                        "status": 0.0,
                        "fmax": 9.0,
                        "energy_barrier": 6.0,
                        "path_length": 22.0,
                        "highest_interior_image_idx": 2.0,
                    },
                ],
            )
        ]

    def test_retains_inactive_paths(self) -> None:
        batch, ctx, stats_hook, hook = _prepare_diagnostics()
        active_paths_ptr = hook._active_paths.data_ptr()

        batch.energy.copy_(
            torch.tensor(
                [[0.0], [20.0], [1.0], [0.0], [0.0], [1.0], [30.0], [2.0], [0.0]]
            )
        )
        batch.forces.mul_(10)
        batch.forward_link_length.mul_(2)
        ctx.active_graph_mask = torch.tensor([False] * 4 + [True] * 5)
        ctx.step_count = 1
        stats_hook(ctx, DynamicsStage.AFTER_COMPUTE)
        hook(ctx, DynamicsStage.AFTER_COMPUTE)
        refreshed = hook.get_diagnostics()

        assert hook._active_paths.data_ptr() == active_paths_ptr
        assert torch.equal(refreshed.fmax, torch.tensor([4.0, 90.0]))
        assert torch.equal(refreshed.energy_barrier, torch.tensor([5.0, 30.0]))
        assert torch.equal(refreshed.path_length, torch.tensor([6.0, 44.0]))
        assert torch.equal(
            refreshed.highest_interior_image_idx,
            torch.tensor([1, 2], dtype=torch.int32),
        )

    def test_tied_highest_interior_energy_uses_first_path_local_index(self) -> None:
        batch = _diagnostic_path(
            [0.0, 5.0, 5.0, 0.0],
            [0] * 4,
            [1.0, 1.0, 1.0, 0.0],
        )
        stats_hook = PathEnergyStatsHook()
        hook = PathDiagnosticsHook(energy_stats_hook=stats_hook)
        ctx = DynamicsContext(batch=batch)

        stats_hook(ctx, DynamicsStage.ON_ADMISSION)
        hook(ctx, DynamicsStage.ON_ADMISSION)
        stats_hook(ctx, DynamicsStage.AFTER_COMPUTE)
        hook(ctx, DynamicsStage.AFTER_COMPUTE)

        assert hook.get_diagnostics().highest_interior_image_idx.tolist() == [1]

    def test_after_compute_compiles_with_cuda_graphs(self, gpu_device: str) -> None:
        _, ctx, stats_hook, hook = _prepare_diagnostics(
            device=gpu_device, eager_refresh=False
        )

        def refresh() -> None:
            stats_hook(ctx, DynamicsStage.AFTER_COMPUTE)
            hook(ctx, DynamicsStage.AFTER_COMPUTE)

        compiled = torch.compile(refresh, backend="cudagraphs", fullgraph=True)
        try:
            compiled()
            torch.cuda.synchronize()
            diagnostics = hook.get_diagnostics()
            assert torch.isfinite(diagnostics.fmax).all()
            assert diagnostics.path_length.tolist() == [6.0, 22.0]
        finally:
            torch.compiler.reset()
