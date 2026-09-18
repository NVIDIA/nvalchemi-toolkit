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
"""Tests for :mod:`nvalchemi.training.distillation._attach`."""

from __future__ import annotations

import pytest
import torch

from nvalchemi.training.distillation._attach import _attach_teacher_labels
from test.training.conftest import _build_batch


class TestAttachTeacherLabelsContract:
    """The funnel enforces the scorer contract on every labeling route."""

    def test_a_label_carrying_a_graph_is_attached_detached(self) -> None:
        """A tensor that still requires grad lands on the batch without its graph."""
        batch = _build_batch()
        values = (torch.randn(batch.num_graphs, 1) * 2).requires_grad_()
        _attach_teacher_labels(batch, {"teacher_energy": (values, "system")})
        stored = batch["teacher_energy"]
        assert stored.requires_grad is False
        assert stored.grad_fn is None
        torch.testing.assert_close(stored, values.detach())

    def test_a_node_label_carrying_a_graph_is_attached_detached(self) -> None:
        """Node-level labels are detached like system-level ones."""
        batch = _build_batch()
        values = torch.randn(batch.num_nodes, 3, requires_grad=True).exp()
        assert values.grad_fn is not None
        _attach_teacher_labels(batch, {"teacher_forces": (values, "node")})
        assert batch["teacher_forces"].requires_grad is False

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_a_label_on_another_device_lands_on_the_batch_device(self) -> None:
        """A CUDA label attached to a CPU batch is stored on the CPU."""
        batch = _build_batch()
        values = torch.randn(batch.num_graphs, 1, device="cuda")
        _attach_teacher_labels(batch, {"teacher_energy": (values, "system")})
        assert batch["teacher_energy"].device == batch.device
        torch.testing.assert_close(batch["teacher_energy"], values.cpu())

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_a_cpu_label_lands_on_a_cuda_batch(self) -> None:
        """A CPU label attached to a CUDA batch is stored on the batch's device."""
        batch = _build_batch().to("cuda")
        values = torch.randn(batch.num_nodes, 3)
        _attach_teacher_labels(batch, {"teacher_forces": (values, "node")})
        assert batch["teacher_forces"].device == batch.device

    def test_an_unknown_level_is_refused(self) -> None:
        """A level outside ``node``/``system`` names the field and the level."""
        batch = _build_batch()
        with pytest.raises(ValueError, match="'teacher_energy'.*unknown level 'edge'"):
            _attach_teacher_labels(
                batch, {"teacher_energy": (torch.zeros(batch.num_graphs, 1), "edge")}
            )
        assert "teacher_energy" not in batch

    def test_a_foreign_field_is_refused_before_anything_is_attached(self) -> None:
        """One foreign key refuses the whole label set, teacher fields included."""
        batch = _build_batch()
        labels = {
            "teacher_energy": (torch.zeros(batch.num_graphs, 1), "system"),
            "energy": (torch.zeros(batch.num_graphs, 1), "system"),
        }
        with pytest.raises(ValueError, match="Teacher labels must populate"):
            _attach_teacher_labels(batch, labels)
        assert "teacher_energy" not in batch

    def test_relabeling_overwrites_the_previous_values(self) -> None:
        """Attaching the same field twice keeps the second tensor."""
        batch = _build_batch()
        first = torch.ones(batch.num_graphs, 1)
        second = torch.full((batch.num_graphs, 1), 2.0)
        _attach_teacher_labels(batch, {"teacher_energy": (first, "system")})
        _attach_teacher_labels(batch, {"teacher_energy": (second, "system")})
        torch.testing.assert_close(batch["teacher_energy"], second)
