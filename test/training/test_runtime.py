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
)


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
