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
"""Tests for training runtime helpers."""

from __future__ import annotations

import pytest
import torch
from torch import nn
from torch.utils.data import SequentialSampler

from nvalchemi.training.runtime import (
    configure_dataloader,
    freeze_unconfigured_models,
    move_to_devices,
    rehome_optimizer_state,
)


def _stepped_adam(model: nn.Module) -> torch.optim.Optimizer:
    """Return an Adam that has taken one step, so its state exists."""
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    model(torch.zeros(1, 4, device=next(model.parameters()).device)).sum().backward()
    optimizer.step()
    return optimizer


def _nested_state_optimizer(model: nn.Module, device: str) -> torch.optim.Optimizer:
    """Return an optimizer whose state nests tensors in a dict, list, and tuple."""
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    optimizer.state[next(model.parameters())] = {
        "nested": {"m": torch.zeros(2, device=device)},
        "hist": [torch.ones(2, device=device), torch.full((2,), 2.0, device=device)],
        "pair": (torch.zeros(1, device=device),),
        "steps_taken": 3,
    }
    return optimizer


class TestRuntimeHelpers:
    @pytest.mark.parametrize("n_models", [1, 2], ids=["single_model", "two_models"])
    def test_move_to_devices_cpu(self, n_models: int) -> None:
        models = {str(i): nn.Linear(4, 2) for i in range(n_models)}
        devices = [torch.device("cpu")]
        out = move_to_devices(models, devices)
        assert len(out) == n_models
        for m in out.values():
            assert next(m.parameters()).device.type == "cpu"

    def test_move_to_devices_moduledict_preserves_input_shape(self) -> None:
        models = nn.ModuleDict({"a": nn.Linear(4, 2), "b": nn.Linear(4, 2)})
        out = move_to_devices(models, [torch.device("cpu")])
        assert out is models
        assert list(out.keys()) == ["a", "b"]
        for model in out.values():
            assert next(model.parameters()).device.type == "cpu"

    def test_configure_dataloader_supports_sampler(self) -> None:
        dataset = [0, 1, 2]
        loader = configure_dataloader(
            dataset,
            batch_size=1,
            sampler=SequentialSampler(dataset),
        )
        assert [int(batch.item()) for batch in loader] == dataset

    def test_configure_dataloader_sampler_shuffle_conflict(self) -> None:
        dataset = [0, 1, 2]
        with pytest.raises(ValueError, match="shuffle=True is incompatible"):
            configure_dataloader(
                dataset,
                batch_size=1,
                shuffle=True,
                sampler=SequentialSampler(dataset),
            )

    def test_freeze_unconfigured_models_restores_state(self) -> None:
        trained = nn.Linear(2, 1)
        omitted = nn.Linear(2, 1)
        omitted.eval()
        params = list(omitted.parameters())
        params[0].requires_grad_(False)
        initial_training = omitted.training
        initial_requires_grad = [param.requires_grad for param in params]
        with freeze_unconfigured_models(
            {"trained": trained, "omitted": omitted}, {"trained": object()}
        ):
            assert omitted.training is False
            assert [param.requires_grad for param in params] == [False] * len(params)
        assert omitted.training is initial_training
        assert [param.requires_grad for param in params] == initial_requires_grad

    def test_freeze_unconfigured_models_accepts_moduledict(self) -> None:
        models = nn.ModuleDict({"trained": nn.Linear(2, 1), "omitted": nn.Linear(2, 1)})
        omitted = models["omitted"]
        params = list(omitted.parameters())
        with freeze_unconfigured_models(models, {"trained": object()}):
            assert omitted.training is False
            assert [param.requires_grad for param in params] == [False] * len(params)
        assert omitted.training is True
        assert [param.requires_grad for param in params] == [True] * len(params)


class TestRehomeOptimizerState:
    """Tests for :func:`rehome_optimizer_state`."""

    def test_state_already_on_the_parameter_device_is_untouched(self) -> None:
        """Rehoming a same-device optimizer leaves every state tensor identical."""
        model = nn.Linear(4, 2)
        optimizer = _stepped_adam(model)
        before = {
            key: value
            for state in optimizer.state.values()
            for key, value in state.items()
        }

        rehome_optimizer_state(optimizer)

        after = {
            key: value
            for state in optimizer.state.values()
            for key, value in state.items()
        }
        assert before.keys() == after.keys()
        assert all(after[key] is before[key] for key in before)

    def test_unstepped_optimizer_is_a_no_op(self) -> None:
        """An optimizer with no state yet survives rehoming untouched."""
        optimizer = torch.optim.Adam(nn.Linear(4, 2).parameters(), lr=1e-3)

        rehome_optimizer_state(optimizer)

        assert not optimizer.state

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_state_follows_parameters_moved_after_load(self) -> None:
        """Moment tensors left behind by a device move follow their parameter."""
        model = nn.Linear(4, 2)
        optimizer = _stepped_adam(model)
        model.to("cuda")
        assert any(
            value.device.type == "cpu"
            for state in optimizer.state.values()
            for value in state.values()
            if isinstance(value, torch.Tensor)
        )

        rehome_optimizer_state(optimizer)

        state = next(iter(optimizer.state.values()))
        assert state["exp_avg"].device.type == "cuda"
        assert state["exp_avg_sq"].device.type == "cuda"
        assert state["step"].device.type == "cpu"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_nested_state_containers_follow_the_parameter(self) -> None:
        """Tensors nested in dicts, lists, and tuples reach the parameter device."""
        model = nn.Linear(4, 2).to("cuda")
        optimizer = _nested_state_optimizer(model, device="cpu")

        rehome_optimizer_state(optimizer)

        state = optimizer.state[next(model.parameters())]
        assert state["nested"]["m"].device.type == "cuda"
        assert all(tensor.device.type == "cuda" for tensor in state["hist"])
        assert state["pair"][0].device.type == "cuda"

    def test_nested_state_already_placed_keeps_its_containers(self) -> None:
        """Rehoming in place preserves container types, leaves, and plain values."""
        model = nn.Linear(4, 2)
        optimizer = _nested_state_optimizer(model, device="cpu")
        state = optimizer.state[next(model.parameters())]
        nested, hist, moment, first, paired = (
            state["nested"],
            state["hist"],
            state["nested"]["m"],
            state["hist"][0],
            state["pair"][0],
        )

        rehome_optimizer_state(optimizer)

        assert isinstance(state["nested"], dict)
        assert isinstance(state["hist"], list)
        assert isinstance(state["pair"], tuple)
        assert state["nested"] is nested
        assert state["hist"] is hist
        assert state["nested"]["m"] is moment
        assert state["hist"][0] is first
        assert state["pair"][0] is paired
        assert state["steps_taken"] == 3

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_non_cpu_step_follows_the_parameter(self) -> None:
        """A ``step`` recorded on another device is moved, not left behind."""
        model = nn.Linear(4, 2).to("cuda")
        optimizer = _stepped_adam(model)
        state = next(iter(optimizer.state.values()))
        state["step"] = state["step"].to("cuda")
        model.to("cpu")

        rehome_optimizer_state(optimizer)

        assert state["step"].device.type == "cpu"
