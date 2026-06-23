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
"""Tests for the public standalone :class:`ValidationLoop` API."""

from __future__ import annotations

import math

import pytest
import torch

from nvalchemi.training import (
    EnergyMSELoss,
    ForceMSELoss,
    ValidationConfig,
    ValidationLoop,
)
from test.training.conftest import _build_dataset, _build_demo_model
from test.training.test_strategy import demo_training_fn

device = torch.device("cpu")


def _named_validation_fn(models, batch):
    """Standalone validation function for the named-model (dict) path."""
    return demo_training_fn(models["main"], batch)


def _composed_loss():
    """Return a :class:`ComposedLossFunction` (energy MSE + force MSE)."""
    return EnergyMSELoss() + ForceMSELoss(normalize_by_atom_count=True)


class TestValidationLoopConstruction:
    """Constructor validation for the standalone :class:`ValidationLoop`."""

    def test_rejects_both_model_and_models(self) -> None:
        """Passing both ``model`` and ``models`` raises ``ValueError``."""
        data = _build_dataset(n_batches=2)
        config = ValidationConfig(validation_data=data, loss_fn=_composed_loss())
        with pytest.raises(ValueError, match="Exactly one of"):
            ValidationLoop(
                validation_data=data,
                config=config,
                device=device,
                model=_build_demo_model(),
                models={"main": _build_demo_model()},
                validation_fn=demo_training_fn,
                grad_enabled=False,
            )

    def test_rejects_neither_model_nor_models(self) -> None:
        """Passing neither ``model`` nor ``models`` raises ``ValueError``."""
        data = _build_dataset(n_batches=2)
        config = ValidationConfig(validation_data=data, loss_fn=_composed_loss())
        with pytest.raises(ValueError, match="Exactly one of"):
            ValidationLoop(
                validation_data=data,
                config=config,
                device=device,
                validation_fn=demo_training_fn,
                grad_enabled=False,
            )

    def test_requires_validation_fn(self) -> None:
        """Omitting ``validation_fn`` raises ``ValueError``."""
        data = _build_dataset(n_batches=2)
        config = ValidationConfig(validation_data=data, loss_fn=_composed_loss())
        with pytest.raises(ValueError, match="validation_fn is required"):
            ValidationLoop(
                validation_data=data,
                config=config,
                device=device,
                model=_build_demo_model(),
                grad_enabled=False,
            )

    def test_requires_loss_fn_when_config_has_none(self) -> None:
        """No ``loss_fn`` arg and no ``config.loss_fn`` raises ``ValueError``."""
        data = _build_dataset(n_batches=2)
        config = ValidationConfig(validation_data=data)
        with pytest.raises(ValueError, match="loss_fn must be provided"):
            ValidationLoop(
                validation_data=data,
                config=config,
                device=device,
                model=_build_demo_model(),
                validation_fn=demo_training_fn,
                grad_enabled=False,
            )

    def test_loss_fn_from_config_used(self) -> None:
        """A ``config.loss_fn`` resolves the loss when the arg is omitted."""
        data = _build_dataset(n_batches=2)
        config = ValidationConfig(validation_data=data, loss_fn=_composed_loss())
        loop = ValidationLoop(
            validation_data=data,
            config=config,
            device=device,
            model=_build_demo_model(),
            validation_fn=demo_training_fn,
            grad_enabled=False,
        )
        assert isinstance(loop, ValidationLoop)


class TestValidationLoopExecuteSingleModel:
    """Execution of the single-model standalone path."""

    def test_execute_returns_summary_with_expected_keys(self) -> None:
        """``execute()`` returns a summary with the expected keys and labels."""
        data = _build_dataset(n_batches=2)
        config = ValidationConfig(validation_data=data, loss_fn=_composed_loss())
        loop = ValidationLoop(
            validation_data=data,
            config=config,
            device=device,
            model=_build_demo_model(),
            validation_fn=demo_training_fn,
            grad_enabled=True,
        )
        with loop as active_loop:
            summary = active_loop.execute()
        assert summary is not None
        expected = {"name", "total_loss", "model_source", "precision", "num_batches"}
        assert expected <= set(summary)
        assert summary["model_source"] == "live"
        assert summary["precision"] == "float32"
        assert "total_loss" in summary
        assert math.isfinite(float(summary["total_loss"]))

    def test_execute_outside_context_raises(self) -> None:
        """Calling ``execute()`` outside the ``with`` block raises ``RuntimeError``."""
        data = _build_dataset(n_batches=2)
        config = ValidationConfig(validation_data=data, loss_fn=_composed_loss())
        loop = ValidationLoop(
            validation_data=data,
            config=config,
            device=device,
            model=_build_demo_model(),
            validation_fn=demo_training_fn,
            grad_enabled=False,
        )
        with pytest.raises(RuntimeError, match="inside a 'with' block"):
            loop.execute()

    def test_custom_autocast_labels_precision_mixed(self) -> None:
        """A custom ``autocast`` callable labels the precision as ``mixed``."""
        data = _build_dataset(n_batches=2)
        config = ValidationConfig(validation_data=data, loss_fn=_composed_loss())
        # A disabled autocast keeps dtypes at float32 (so the loss dtype check
        # passes) while still being a non-``None`` callable, which is what the
        # loop uses to label the precision as ``mixed``.
        loop = ValidationLoop(
            validation_data=data,
            config=config,
            device=device,
            model=_build_demo_model(),
            validation_fn=demo_training_fn,
            autocast=lambda: torch.autocast(
                device_type="cpu", dtype=torch.bfloat16, enabled=False
            ),
            grad_enabled=True,
        )
        with loop as active_loop:
            summary = active_loop.execute()
        assert summary is not None
        assert summary["precision"] == "mixed"


class TestValidationLoopNamedModels:
    """Execution of the named-model (dict) standalone path."""

    def test_named_models_execute(self) -> None:
        """The named-model path runs and reports a ``live`` source over all batches."""
        data = _build_dataset(n_batches=2)
        config = ValidationConfig(validation_data=data, loss_fn=_composed_loss())
        loop = ValidationLoop(
            validation_data=data,
            config=config,
            device=device,
            models={"main": _build_demo_model()},
            validation_fn=_named_validation_fn,
            grad_enabled=True,
        )
        with loop as active_loop:
            summary = active_loop.execute()
        assert summary is not None
        assert summary["model_source"] == "live"
        assert summary["num_batches"] == 2


class TestValidationLoopStateRestoration:
    """Restoration of training modes and gradients around the loop."""

    def test_training_modes_restored(self) -> None:
        """Training modes are restored after a successful run."""
        data = _build_dataset(n_batches=2)
        config = ValidationConfig(validation_data=data, loss_fn=_composed_loss())
        model = _build_demo_model()
        model.train()
        loop = ValidationLoop(
            validation_data=data,
            config=config,
            device=device,
            model=model,
            validation_fn=demo_training_fn,
            grad_enabled=True,
        )
        with loop as active_loop:
            active_loop.execute()
        assert model.training is True

    def test_training_modes_restored_on_exception(self) -> None:
        """Training modes are restored even when ``execute()`` raises."""
        data = _build_dataset(n_batches=2)
        config = ValidationConfig(validation_data=data, loss_fn=_composed_loss())
        model = _build_demo_model()
        model.train()

        def _raising_validation_fn(model_arg, batch):
            raise RuntimeError("boom")

        loop = ValidationLoop(
            validation_data=data,
            config=config,
            device=device,
            model=model,
            validation_fn=_raising_validation_fn,
            grad_enabled=False,
        )
        with pytest.raises(RuntimeError):
            with loop as active_loop:
                active_loop.execute()
        assert model.training is True

    def test_grads_restored_after_grad_enabled_run(self) -> None:
        """A pre-existing gradient is restored after a grad-enabled run."""
        data = _build_dataset(n_batches=2)
        config = ValidationConfig(validation_data=data, loss_fn=_composed_loss())
        model = _build_demo_model()
        first_param = next(iter(model.parameters()))
        saved_grad = torch.ones_like(first_param)
        first_param.grad = saved_grad.clone()
        loop = ValidationLoop(
            validation_data=data,
            config=config,
            device=device,
            model=model,
            validation_fn=demo_training_fn,
            grad_enabled=True,
        )
        with loop as active_loop:
            active_loop.execute()
        assert first_param.grad is not None
        assert torch.equal(first_param.grad, saved_grad)
