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
"""Tests for optimizer configuration and stepping helpers."""

from __future__ import annotations

import json
from typing import Any

import pytest
import torch
from torch import nn

from nvalchemi.training import register_type_serializer
from nvalchemi.training._spec import create_model_spec_from_json
from nvalchemi.training.optimizers import (
    OptimizerConfig,
    setup_optimizers,
    step_lr_schedulers,
    step_optimizers,
    zero_gradients,
)


class _CustomPlateau(torch.optim.lr_scheduler.ReduceLROnPlateau):
    pass


_OPTIMIZER_CONFIG_REJECTION_CASES: list[tuple[str, dict[str, Any]]] = [
    (
        "Invalid optimizer kwargs",
        {
            "optimizer_cls": torch.optim.Adam,
            "optimizer_kwargs": {"bogus_kwarg": 0.1},
        },
    ),
    (
        "scheduler_kwargs",
        {
            "optimizer_cls": torch.optim.Adam,
            "optimizer_kwargs": {"lr": 1e-3},
            "scheduler_cls": None,
            "scheduler_kwargs": {"step_size": 10},
        },
    ),
    (
        "ReduceLROnPlateau",
        {
            "optimizer_cls": torch.optim.Adam,
            "optimizer_kwargs": {"lr": 1e-3},
            "scheduler_cls": torch.optim.lr_scheduler.ReduceLROnPlateau,
        },
    ),
    (
        "ReduceLROnPlateau",
        {
            "optimizer_cls": torch.optim.Adam,
            "optimizer_kwargs": {"lr": 1e-3},
            "scheduler_cls": _CustomPlateau,
        },
    ),
]


class TestOptimizerConfig:
    def test_public_type_serializer_export_available(self) -> None:
        assert callable(register_type_serializer)

    def test_build_adam_no_scheduler(self) -> None:
        layer = nn.Linear(4, 2)
        cfg = OptimizerConfig(
            optimizer_cls=torch.optim.Adam,
            optimizer_kwargs={"lr": 1e-3},
        )
        optimizer, scheduler = cfg.build(layer.parameters())
        assert isinstance(optimizer, torch.optim.Adam)
        assert scheduler is None

    def test_build_with_step_lr(self) -> None:
        layer = nn.Linear(4, 2)
        cfg = OptimizerConfig(
            optimizer_cls=torch.optim.SGD,
            optimizer_kwargs={"lr": 0.1},
            scheduler_cls=torch.optim.lr_scheduler.StepLR,
            scheduler_kwargs={"step_size": 10, "gamma": 0.5},
        )
        optimizer, scheduler = cfg.build(layer.parameters())
        assert isinstance(optimizer, torch.optim.SGD)
        assert isinstance(scheduler, torch.optim.lr_scheduler.StepLR)

    def test_class_fields_accept_dotted_paths(self) -> None:
        cfg = OptimizerConfig(
            optimizer_cls="torch.optim.sgd.SGD",
            scheduler_cls="torch.optim.lr_scheduler.StepLR",
            scheduler_kwargs={"step_size": 2},
        )
        assert cfg.optimizer_cls is torch.optim.SGD
        assert cfg.scheduler_cls is torch.optim.lr_scheduler.StepLR

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"optimizer_cls": "not.a.real.Optimizer"},
            {
                "optimizer_cls": torch.optim.Adam,
                "scheduler_cls": "not.a.real.Scheduler",
            },
        ],
        ids=["bad_optimizer_cls", "bad_scheduler_cls"],
    )
    def test_class_fields_reject_bad_dotted_paths(self, kwargs: dict[str, Any]) -> None:
        with pytest.raises(ValueError, match="must resolve to an importable class"):
            OptimizerConfig(**kwargs)

    @pytest.mark.parametrize(
        ("match", "kwargs"),
        _OPTIMIZER_CONFIG_REJECTION_CASES,
        ids=[
            "invalid_optimizer_kwarg",
            "orphan_scheduler_kwargs",
            "reduce_lr_on_plateau",
            "reduce_lr_on_plateau_subclass",
        ],
    )
    def test_invalid_config_rejected(self, match: str, kwargs: dict[str, Any]) -> None:
        with pytest.raises(ValueError, match=match):
            OptimizerConfig(**kwargs)

    def test_to_spec_from_spec_roundtrip(self) -> None:
        cfg = OptimizerConfig(
            optimizer_cls=torch.optim.Adam,
            optimizer_kwargs={"lr": 1e-3, "betas": (0.9, 0.95)},
            scheduler_cls=torch.optim.lr_scheduler.StepLR,
            scheduler_kwargs={"step_size": 5, "gamma": 0.1},
        )
        spec = cfg.to_spec()
        restored = OptimizerConfig.from_spec(spec)
        assert restored.optimizer_cls is torch.optim.Adam
        assert restored.optimizer_kwargs["lr"] == pytest.approx(1e-3)
        assert restored.scheduler_cls is torch.optim.lr_scheduler.StepLR
        assert restored.scheduler_kwargs == {"step_size": 5, "gamma": 0.1}

    def test_json_roundtrip_via_spec(self) -> None:
        cfg = OptimizerConfig(
            optimizer_cls=torch.optim.SGD,
            optimizer_kwargs={"lr": 0.01, "momentum": 0.9},
        )
        spec = cfg.to_spec()
        spec_json = spec.model_dump_json()
        spec_back = create_model_spec_from_json(json.loads(spec_json))
        restored = OptimizerConfig.from_spec(spec_back)
        assert restored.optimizer_cls is torch.optim.SGD
        assert restored.optimizer_kwargs == {"lr": 0.01, "momentum": 0.9}
        assert restored.scheduler_cls is None


class TestOptimizerHelpers:
    def test_setup_optimizers_returns_opt_sched_pairs(self) -> None:
        model = nn.Linear(4, 2)
        pairs = setup_optimizers(
            model,
            OptimizerConfig(optimizer_cls=torch.optim.Adam),
        )
        assert set(pairs.keys()) == {"main"}
        assert len(pairs["main"]) == 1
        optimizer, scheduler = pairs["main"][0]
        assert isinstance(optimizer, torch.optim.Adam)
        assert scheduler is None

    def test_setup_optimizers_subset_of_models(self) -> None:
        student = nn.Linear(4, 2)
        teacher = nn.Linear(4, 2)
        pairs = setup_optimizers(
            {"student": student, "teacher": teacher},
            {"student": [OptimizerConfig(optimizer_cls=torch.optim.Adam)]},
        )
        assert set(pairs) == {"student"}

    def test_setup_optimizers_invalid_key_raises(self) -> None:
        with pytest.raises(ValueError, match="not present in models"):
            setup_optimizers(
                {"student": nn.Linear(4, 2)},
                {"teacher": [OptimizerConfig(optimizer_cls=torch.optim.Adam)]},
            )

    def test_setup_optimizers_frozen_model_raises(self) -> None:
        model = nn.Linear(4, 2)
        for param in model.parameters():
            param.requires_grad_(False)
        with pytest.raises(ValueError, match="no trainable parameters"):
            setup_optimizers(model, OptimizerConfig(optimizer_cls=torch.optim.Adam))

    def test_zero_gradients_zeroes_all_optimizers(self) -> None:
        layer_a = nn.Linear(2, 2)
        layer_b = nn.Linear(3, 3)
        opt_a = torch.optim.SGD(layer_a.parameters(), lr=0.1)
        opt_b = torch.optim.SGD(layer_b.parameters(), lr=0.1)
        layer_a.weight.grad = torch.ones_like(layer_a.weight)
        layer_b.weight.grad = torch.ones_like(layer_b.weight)
        zero_gradients([opt_a, opt_b])
        assert layer_a.weight.grad is None
        assert layer_b.weight.grad is None

    def test_step_optimizers_advances_params(self) -> None:
        torch.manual_seed(0)
        layer = nn.Linear(2, 1)
        opt = torch.optim.SGD(layer.parameters(), lr=0.1)
        before = layer.weight.detach().clone()
        layer.weight.grad = torch.ones_like(layer.weight)
        step_optimizers([opt])
        assert not torch.equal(before, layer.weight.detach())

    def test_step_lr_schedulers_skips_none(self) -> None:
        layer = nn.Linear(2, 1)
        opt = torch.optim.SGD(layer.parameters(), lr=1.0)
        sched = torch.optim.lr_scheduler.StepLR(opt, step_size=1, gamma=0.5)
        before_lr = sched.get_last_lr()[0]
        step_lr_schedulers([None, sched, None])
        after_lr = sched.get_last_lr()[0]
        assert after_lr == pytest.approx(before_lr * 0.5)
