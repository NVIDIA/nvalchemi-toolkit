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
from unittest.mock import patch

import pytest
import torch
from torch import nn

from nvalchemi.training import register_type_serializer
from nvalchemi.training._spec import create_model_spec_from_json
from nvalchemi.training.optimizers import (
    OptimizerConfig,
    _extract_scheduler_metric,
    setup_optimizers,
    step_lr_schedulers,
    step_metric_schedulers,
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

    def test_setup_optimizers_accepts_moduledict_models(self) -> None:
        models = nn.ModuleDict(
            {
                "student": nn.Linear(4, 2),
                "teacher": nn.Linear(4, 2),
            }
        )
        pairs = setup_optimizers(
            models,
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

    def test_setup_optimizers_filters_allowed_parameter_names(self) -> None:
        model = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 2))
        pairs = setup_optimizers(
            model,
            OptimizerConfig(optimizer_cls=torch.optim.Adam),
            allowed_parameter_names={"main.1.weight", "main.1.bias"},
        )
        optimizer = pairs["main"][0][0]
        optimizer_param_ids = {
            id(param) for group in optimizer.param_groups for param in group["params"]
        }

        assert id(model[1].weight) in optimizer_param_ids
        assert id(model[1].bias) in optimizer_param_ids
        assert id(model[0].weight) not in optimizer_param_ids
        assert id(model[0].bias) not in optimizer_param_ids

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


class TestMetricDrivenSchedulers:
    """Phase D: metric-driven (ReduceLROnPlateau) scheduler support."""

    @staticmethod
    def _make_plateau(
        lr: float = 0.1, patience: int = 1, factor: float = 0.5
    ) -> tuple[
        nn.Module, torch.optim.Optimizer, torch.optim.lr_scheduler.ReduceLROnPlateau
    ]:
        """Return a (layer, optimizer, ReduceLROnPlateau) triple."""
        layer = nn.Linear(2, 1)
        opt = torch.optim.SGD(layer.parameters(), lr=lr)
        plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, patience=patience, factor=factor
        )
        return layer, opt, plateau

    def test_reduce_lr_on_plateau_now_accepted(self) -> None:
        """OptimizerConfig no longer rejects ReduceLROnPlateau."""
        cfg = OptimizerConfig(
            optimizer_cls=torch.optim.Adam,
            optimizer_kwargs={"lr": 1e-3},
            scheduler_cls=torch.optim.lr_scheduler.ReduceLROnPlateau,
            scheduler_kwargs={"patience": 5},
        )
        assert cfg.scheduler_cls is torch.optim.lr_scheduler.ReduceLROnPlateau

    def test_reduce_lr_on_plateau_subclass_accepted(self) -> None:
        """OptimizerConfig also accepts ReduceLROnPlateau subclasses."""
        cfg = OptimizerConfig(
            optimizer_cls=torch.optim.Adam,
            optimizer_kwargs={"lr": 1e-3},
            scheduler_cls=_CustomPlateau,
        )
        assert cfg.scheduler_cls is _CustomPlateau

    def test_scheduler_metric_adapter_requires_scheduler_cls(self) -> None:
        """scheduler_metric_adapter without scheduler_cls raises ValueError."""
        with pytest.raises(ValueError, match="scheduler_metric_adapter provided"):
            OptimizerConfig(
                optimizer_cls=torch.optim.Adam,
                scheduler_metric_adapter="total_loss",
            )

    def test_step_lr_schedulers_skips_metric_driven(self) -> None:
        """step_lr_schedulers does not call step() on ReduceLROnPlateau."""
        _, _, plateau = self._make_plateau()
        layer2 = nn.Linear(2, 1)
        opt2 = torch.optim.SGD(layer2.parameters(), lr=1.0)
        steplr = torch.optim.lr_scheduler.StepLR(opt2, step_size=1, gamma=0.5)

        steplr_epoch_before = steplr.last_epoch
        with patch.object(plateau, "step", wraps=plateau.step) as mock_plateau_step:
            step_lr_schedulers([plateau, steplr])

        mock_plateau_step.assert_not_called()
        assert steplr.last_epoch == steplr_epoch_before + 1

    def test_step_metric_schedulers_str_adapter(self) -> None:
        """step_metric_schedulers with a str adapter passes the right value."""
        _, opt, plateau = self._make_plateau()
        summary = {"my_loss": torch.tensor(0.42), "other": 99}
        with patch.object(plateau, "step", wraps=plateau.step) as mock_step:
            step_metric_schedulers([plateau], ["my_loss"], summary)
        mock_step.assert_called_once()
        arg = mock_step.call_args[0][0]
        assert arg == pytest.approx(0.42)

    def test_step_metric_schedulers_callable_adapter(self) -> None:
        """step_metric_schedulers with a callable adapter."""
        _, opt, plateau = self._make_plateau()
        summary = {"nested": {"val": 1.23}}
        adapter = lambda s: s["nested"]["val"]  # noqa: E731
        with patch.object(plateau, "step", wraps=plateau.step) as mock_step:
            step_metric_schedulers([plateau], [adapter], summary)
        mock_step.assert_called_once()
        assert mock_step.call_args[0][0] == pytest.approx(1.23)

    def test_step_metric_schedulers_default_adapter(self) -> None:
        """step_metric_schedulers with adapter=None uses 'total_loss' key."""
        _, opt, plateau = self._make_plateau()
        summary = {
            "name": "validation",
            "total_loss": torch.tensor(0.55),
            "per_component_unweighted": {},
        }
        with patch.object(plateau, "step", wraps=plateau.step) as mock_step:
            step_metric_schedulers([plateau], [None], summary)
        mock_step.assert_called_once()
        assert mock_step.call_args[0][0] == pytest.approx(0.55)

    def test_step_metric_schedulers_skips_non_metric(self) -> None:
        """step_metric_schedulers skips time-based schedulers and None."""
        layer = nn.Linear(2, 1)
        opt = torch.optim.SGD(layer.parameters(), lr=1.0)
        steplr = torch.optim.lr_scheduler.StepLR(opt, step_size=1, gamma=0.5)
        epoch_before = steplr.last_epoch
        summary = {"total_loss": torch.tensor(0.5)}
        step_metric_schedulers([None, steplr], [None, None], summary)
        # StepLR should NOT have been stepped (it's not metric-driven)
        assert steplr.last_epoch == epoch_before

    def test_extract_scheduler_metric_missing_key_raises(self) -> None:
        """_extract_scheduler_metric raises KeyError for absent str key."""
        summary = {"a": 1.0, "b": 2.0}
        with pytest.raises(KeyError, match="not_here"):
            _extract_scheduler_metric(summary, "not_here")
