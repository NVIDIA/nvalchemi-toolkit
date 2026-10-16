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
"""Tests for hooks shared by reaction-path dynamics methods."""

from __future__ import annotations

import pytest
import torch

from nvalchemi.data import AtomicData, Batch
from nvalchemi.dynamics import DynamicsStage
from nvalchemi.dynamics.hooks import LoggingHook
from nvalchemi.dynamics.mep.hooks import (
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
        stats = hook.get_stats(ctx)

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

            stats = hook._stats
            assert stats is not None
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
        stats = hook.get_stats(ctx)
        pointers = (
            stats.endpoint_reference_energy.data_ptr(),
            stats.highest_interior_energy.data_ptr(),
            stats.highest_interior_image_idx.data_ptr(),
        )

        batch.energy.copy_(torch.tensor([[3.0], [1.0], [5.0], [2.0]]))
        ctx.step_count = 1
        hook(ctx, DynamicsStage.AFTER_COMPUTE)
        refreshed = hook.get_stats(ctx)

        assert pointers == (
            refreshed.endpoint_reference_energy.data_ptr(),
            refreshed.highest_interior_energy.data_ptr(),
            refreshed.highest_interior_image_idx.data_ptr(),
        )
        assert refreshed.endpoint_reference_energy.tolist() == [2.0]
        assert refreshed.highest_interior_energy.tolist() == [5.0]
        assert refreshed.highest_interior_image_idx.tolist() == [2]

    def test_rejects_access_before_admission(self, path: Batch) -> None:
        hook = PathEnergyStatsHook()

        with pytest.raises(RuntimeError, match="ON_ADMISSION"):
            hook.get_stats(DynamicsContext(batch=path))

    def test_rejects_access_before_energy_refresh(self, path: Batch) -> None:
        hook = PathEnergyStatsHook()
        ctx = DynamicsContext(batch=path)
        hook(ctx, DynamicsStage.ON_ADMISSION)

        with pytest.raises(RuntimeError, match="stale"):
            hook.get_stats(ctx)

    def test_rejects_access_after_in_place_energy_update(self, path: Batch) -> None:
        hook = PathEnergyStatsHook()
        ctx = DynamicsContext(batch=path, step_count=0)
        hook(ctx, DynamicsStage.ON_ADMISSION)
        hook(ctx, DynamicsStage.AFTER_COMPUTE)

        path.energy.copy_(torch.zeros_like(path.energy))

        with pytest.raises(RuntimeError, match="stale"):
            hook.get_stats(ctx)

    def test_rejects_access_after_step_advances(self, path: Batch) -> None:
        hook = PathEnergyStatsHook()
        ctx = DynamicsContext(batch=path, step_count=0)
        hook(ctx, DynamicsStage.ON_ADMISSION)
        hook(ctx, DynamicsStage.AFTER_COMPUTE)

        ctx.step_count = 1

        with pytest.raises(RuntimeError, match="stale"):
            hook.get_stats(ctx)

    def test_rejects_inference_energy_tensor(self, path: Batch) -> None:
        hook = PathEnergyStatsHook()
        ctx = DynamicsContext(batch=path)
        hook(ctx, DynamicsStage.ON_ADMISSION)
        with torch.inference_mode():
            path.energy = path.energy.clone()

        with pytest.raises(RuntimeError, match="inference tensors"):
            hook(ctx, DynamicsStage.AFTER_COMPUTE)


_DIAGNOSTIC_ENERGIES = [3.0, 8.0, 5.0, 4.0, 2.0, 3.0, 7.0, 6.0, 1.0]
_DIAGNOSTIC_GROUPS = [0] * 4 + [1] * 5
_DIAGNOSTIC_LINKS = [1.0, 2.0, 3.0, 0.0, 4.0, 5.0, 6.0, 7.0, 0.0]


def _diagnostic_path(
    energies: list[float],
    groups: list[int],
    link_lengths: list[float] | None = None,
    *,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> Batch:
    """Build a grouped path with deterministic energies and forces."""
    batch = Batch.from_data_list(
        [
            AtomicData(
                atomic_numbers=torch.tensor([1], device=device),
                positions=torch.tensor(
                    [[float(i), 0.0, 0.0]], device=device, dtype=dtype
                ),
                energy=torch.tensor([[energy]], device=device, dtype=dtype),
                forces=torch.tensor(
                    [[float(i + 1), 0.0, 0.0]], device=device, dtype=dtype
                ),
            )
            for i, energy in enumerate(energies)
        ]
    )
    batch.set_group_layout(torch.tensor(groups, device=device))
    if link_lengths is not None:
        batch.forward_link_length = torch.tensor(
            link_lengths, device=device, dtype=dtype
        )
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

    def test_admission_resets_values_and_reuses_buffers(self, device: str) -> None:
        """Membership changes reset results without reallocating compatible buffers."""
        batch, _, stats_hook, hook = _prepare_diagnostics(device=device)
        diagnostics = hook.get_diagnostics()
        buffers = {
            key: getattr(hook, key)
            for key in (
                "_path_start",
                "_active_paths",
                "_force_norms",
                "_candidate_fmax",
                "_candidate_energy_barrier",
                "_candidate_path_length",
                "_candidate_highest_interior_image_idx",
            )
        }
        current = batch.index_select_groups([1, 0])
        ctx = DynamicsContext(
            batch=current,
            previous_graph_indices=torch.tensor(
                [4, 5, 6, 7, 8, -1, -1, -1, -1], device=device
            ),
            active_graph_mask=torch.tensor([False] * 5 + [True] * 4, device=device),
        )
        stats_hook(ctx, DynamicsStage.ON_ADMISSION)
        hook(ctx, DynamicsStage.ON_ADMISSION)
        assert hook.get_diagnostics() is diagnostics
        for key, value in buffers.items():
            assert getattr(hook, key) is value
        assert hook._path_start.tolist() == [0, 5]
        assert torch.isnan(diagnostics.fmax).all()
        assert torch.isnan(diagnostics.energy_barrier).all()
        assert torch.isnan(diagnostics.path_length).all()
        assert torch.all(diagnostics.highest_interior_image_idx == -1)

        current.forces.mul_(10)
        stats_hook(ctx, DynamicsStage.AFTER_COMPUTE)
        hook(ctx, DynamicsStage.AFTER_COMPUTE)
        assert torch.isnan(diagnostics.fmax[0])
        assert diagnostics.fmax[1] == 40
        assert diagnostics.energy_barrier[1] == 5
        assert diagnostics.path_length[1] == 6
        assert diagnostics.highest_interior_image_idx[1] == 1

        hook(DynamicsContext(batch=current), DynamicsStage.ON_ADMISSION)
        assert torch.isnan(hook.get_diagnostics().fmax).all()
        assert torch.all(hook.get_diagnostics().highest_interior_image_idx == -1)

    @pytest.mark.parametrize("change", ["paths", "nodes", "dtype", "device"])
    def test_admission_resizes_incompatible_buffers(
        self, device: str, change: str
    ) -> None:
        """Changed sizes, precision, or device produce correctly sized reductions."""
        batch, _, stats_hook, hook = _prepare_diagnostics(device=device)
        diagnostics = hook.get_diagnostics()
        force_norms = hook._force_norms
        if change == "paths":
            current = batch.index_select_groups([1])
        elif change == "nodes":
            current = _diagnostic_path(
                [0.0] * 6, [0] * 3 + [1] * 3, [1.0, 1.0, 0.0] * 2, device=device
            )
        elif change == "dtype":
            current = _diagnostic_path(
                _DIAGNOSTIC_ENERGIES,
                _DIAGNOSTIC_GROUPS,
                _DIAGNOSTIC_LINKS,
                device=device,
                dtype=torch.float64,
            )
        else:
            if not torch.cuda.is_available():
                pytest.skip("Device changes require CUDA")
            current = batch.to("cuda" if device == "cpu" else "cpu")
        ctx = DynamicsContext(batch=current)
        stats_hook(ctx, DynamicsStage.ON_ADMISSION)
        hook(ctx, DynamicsStage.ON_ADMISSION)
        refreshed = hook.get_diagnostics()
        assert (refreshed is diagnostics) == (change == "nodes")
        assert hook._force_norms is not force_norms
        assert hook._force_norms.shape == (current.num_nodes,)
        assert hook._force_norms.dtype == current.positions.dtype
        assert hook._force_norms.device == current.device
        assert refreshed.fmax.shape == (current.group_layout.num_groups,)
        assert refreshed.fmax.dtype == current.positions.dtype
        assert refreshed.fmax.device == current.device
        stats_hook(ctx, DynamicsStage.AFTER_COMPUTE)
        hook(ctx, DynamicsStage.AFTER_COMPUTE)
        assert torch.isfinite(refreshed.fmax).all()
        assert torch.isfinite(refreshed.energy_barrier).all()
        assert torch.isfinite(refreshed.path_length).all()
        assert torch.all(refreshed.highest_interior_image_idx >= 0)

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
                "energy_barrier": lambda ctx: (
                    diagnostics_hook.get_diagnostics().energy_barrier
                ),
                "path_length": (
                    lambda ctx: diagnostics_hook.get_diagnostics().path_length
                ),
                "highest_interior_image_idx": (
                    lambda ctx: (
                        diagnostics_hook.get_diagnostics().highest_interior_image_idx
                    )
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
