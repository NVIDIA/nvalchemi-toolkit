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
"""Tests for :class:`nvalchemi.training._validation.ValidationConfig`."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from nvalchemi.training import EnergyMSELoss, ForceMSELoss
from nvalchemi.training._validation import ValidationConfig
from nvalchemi.training.losses.composition import ComposedLossFunction


class TestValidationConfigConstruction:
    """Validate construction defaults, normalization, and rejection."""

    def test_defaults(self) -> None:
        """All optional fields receive sensible defaults."""
        cfg = ValidationConfig(validation_data=[])
        assert cfg.validation_fn is None
        assert cfg.loss_fn is None
        assert cfg.every_n_epochs is None
        assert cfg.every_n_steps is None
        assert cfg.grad_mode == "auto"
        assert cfg.set_eval is True
        assert cfg.use_ema == "auto"
        assert cfg.use_mixed_precision == "auto"
        assert cfg.batch_callback is None
        assert cfg.name == "validation"

    def test_schedule_mutual_exclusion_raises(self) -> None:
        """Setting both every_n_epochs and every_n_steps raises."""
        with pytest.raises(ValidationError, match="Only one of"):
            ValidationConfig(
                validation_data=[],
                every_n_epochs=2,
                every_n_steps=5,
            )

    def test_every_n_epochs_only(self) -> None:
        """Setting only every_n_epochs is accepted."""
        cfg = ValidationConfig(validation_data=[], every_n_epochs=3)
        assert cfg.every_n_epochs == 3
        assert cfg.every_n_steps is None

    def test_every_n_steps_only(self) -> None:
        """Setting only every_n_steps is accepted."""
        cfg = ValidationConfig(validation_data=[], every_n_steps=10)
        assert cfg.every_n_steps == 10
        assert cfg.every_n_epochs is None

    def test_loss_fn_normalization_leaf(self) -> None:
        """A leaf loss is normalized to a ComposedLossFunction."""
        cfg = ValidationConfig(validation_data=[], loss_fn=EnergyMSELoss())
        assert isinstance(cfg.loss_fn, ComposedLossFunction)
        assert len(cfg.loss_fn.components) == 1
        assert isinstance(cfg.loss_fn.components[0], EnergyMSELoss)

    def test_loss_fn_normalization_composed(self) -> None:
        """A ComposedLossFunction passes through unchanged."""
        composed = EnergyMSELoss() + ForceMSELoss()
        cfg = ValidationConfig(validation_data=[], loss_fn=composed)
        assert isinstance(cfg.loss_fn, ComposedLossFunction)
        assert len(cfg.loss_fn.components) == 2

    def test_loss_fn_none_stays_none(self) -> None:
        """loss_fn=None (use strategy default) stays None."""
        cfg = ValidationConfig(validation_data=[])
        assert cfg.loss_fn is None

    def test_extra_fields_rejected(self) -> None:
        """Unknown fields are rejected by extra='forbid'."""
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            ValidationConfig(validation_data=[], bogus_field=True)

    def test_every_n_epochs_minimum_one(self) -> None:
        """every_n_epochs must be >= 1."""
        with pytest.raises(ValidationError, match="greater than or equal to 1"):
            ValidationConfig(validation_data=[], every_n_epochs=0)

    def test_every_n_steps_minimum_one(self) -> None:
        """every_n_steps must be >= 1."""
        with pytest.raises(ValidationError, match="greater than or equal to 1"):
            ValidationConfig(validation_data=[], every_n_steps=0)

    def test_name_minimum_length(self) -> None:
        """name must be non-empty."""
        with pytest.raises(ValidationError):
            ValidationConfig(validation_data=[], name="")


class TestValidationDataReiterability:
    """Ensure validation_data rejects one-shot iterators and preserves re-iterables."""

    def test_validation_data_list_is_reiterable(self) -> None:
        """A list of Batch-like objects survives two full iteration passes."""
        sentinel_a, sentinel_b = object(), object()
        cfg = ValidationConfig(validation_data=[sentinel_a, sentinel_b])
        first_pass = list(cfg.validation_data)
        second_pass = list(cfg.validation_data)
        assert first_pass == [sentinel_a, sentinel_b]
        assert second_pass == [sentinel_a, sentinel_b]

    def test_validation_data_generator_rejected(self) -> None:
        """A generator expression is rejected as one-shot."""
        with pytest.raises(ValidationError, match="re-iterable"):
            ValidationConfig(validation_data=(x for x in [1, 2]))

    def test_validation_data_bare_iterator_rejected(self) -> None:
        """A bare list_iterator is rejected as one-shot."""
        with pytest.raises(ValidationError, match="re-iterable"):
            ValidationConfig(validation_data=iter([1, 2]))

    def test_validation_data_non_iterable_rejected(self) -> None:
        """A non-iterable value is rejected with a clear error."""
        with pytest.raises(ValidationError, match="iterable"):
            ValidationConfig(validation_data=42)
