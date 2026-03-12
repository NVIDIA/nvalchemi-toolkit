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
"""Tests for training configuration models."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from nvalchemi.training._configs import (
    GradClipConfig,
    MixedPrecisionConfig,
    TrainingConfig,
)


class TestMixedPrecisionConfig:
    """Tests for :class:`MixedPrecisionConfig`."""

    def test_defaults(self) -> None:
        cfg = MixedPrecisionConfig()
        assert cfg.enabled is False
        assert cfg.amp_type == "bf16"
        # bf16 forces scaler off
        assert cfg.grad_scaler is False

    def test_fp16_keeps_scaler(self) -> None:
        cfg = MixedPrecisionConfig(amp_type="fp16", grad_scaler=True)
        assert cfg.grad_scaler is True

    def test_bf16_forces_scaler_false(self) -> None:
        cfg = MixedPrecisionConfig(amp_type="bf16", grad_scaler=True)
        assert cfg.grad_scaler is False

    def test_fp16_scaler_false_respected(self) -> None:
        cfg = MixedPrecisionConfig(amp_type="fp16", grad_scaler=False)
        assert cfg.grad_scaler is False

    def test_enabled_flag(self) -> None:
        cfg = MixedPrecisionConfig(enabled=True, amp_type="fp16")
        assert cfg.enabled is True

    def test_torch_dtype_fp16(self) -> None:
        cfg = MixedPrecisionConfig(amp_type="fp16")
        assert cfg.torch_dtype is torch.float16

    def test_torch_dtype_bf16(self) -> None:
        cfg = MixedPrecisionConfig(amp_type="bf16")
        assert cfg.torch_dtype is torch.bfloat16


class TestTrainingConfig:
    """Tests for :class:`TrainingConfig`."""

    def test_minimal_construction(self) -> None:
        cfg = TrainingConfig(max_epochs=10)
        assert cfg.max_epochs == 10
        assert cfg.grad_clip is None
        assert cfg.grad_accumulation_steps == 1
        assert cfg.val_every_n_epochs == 1
        assert cfg.checkpoint_every_n_epochs == 1
        assert cfg.checkpoint_dir == Path("checkpoints")
        assert cfg.resume_from is None
        assert isinstance(cfg.mixed_precision, MixedPrecisionConfig)
        assert cfg.strategy == "ddp"
        assert cfg.torch_compile is False
        assert cfg.log_every_n_steps == 10
        assert cfg.seed == 42

    def test_max_epochs_required(self) -> None:
        with pytest.raises(Exception):  # noqa: B017
            TrainingConfig()  # type: ignore[call-arg]

    def test_max_epochs_positive(self) -> None:
        with pytest.raises(Exception):  # noqa: B017
            TrainingConfig(max_epochs=0)

    def test_grad_clip_non_negative(self) -> None:
        with pytest.raises(Exception):  # noqa: B017
            TrainingConfig(
                max_epochs=5,
                grad_clip=GradClipConfig(max_value=-1.0),
            )

    def test_full_construction(self) -> None:
        cfg = TrainingConfig(
            max_epochs=100,
            grad_clip=GradClipConfig(max_value=1.0),
            grad_accumulation_steps=4,
            val_every_n_epochs=2,
            checkpoint_every_n_epochs=5,
            checkpoint_dir=Path("ckpts"),
            resume_from=Path("ckpts/epoch_10"),
            mixed_precision=MixedPrecisionConfig(enabled=True, amp_type="fp16"),
            strategy="fsdp",
            log_every_n_steps=50,
            seed=123,
        )
        assert cfg.max_epochs == 100
        assert cfg.grad_clip is not None
        assert cfg.grad_clip.max_value == 1.0
        assert cfg.grad_clip.method == "norm"
        assert cfg.grad_accumulation_steps == 4
        assert cfg.strategy == "fsdp"
        assert cfg.mixed_precision.enabled is True
        assert cfg.mixed_precision.amp_type == "fp16"
        assert cfg.mixed_precision.grad_scaler is True

    def test_nested_bf16_scaler_constraint(self) -> None:
        cfg = TrainingConfig(
            max_epochs=1,
            mixed_precision=MixedPrecisionConfig(
                enabled=True, amp_type="bf16", grad_scaler=True
            ),
        )
        assert cfg.mixed_precision.grad_scaler is False

    def test_grad_accumulation_steps_positive(self) -> None:
        with pytest.raises(Exception):  # noqa: B017
            TrainingConfig(max_epochs=5, grad_accumulation_steps=0)

    def test_strategy_invalid(self) -> None:
        with pytest.raises(Exception):  # noqa: B017
            TrainingConfig(max_epochs=5, strategy="invalid")  # type: ignore[arg-type]

    def test_torch_compile_enabled(self) -> None:
        cfg = TrainingConfig(max_epochs=5, torch_compile=True)
        assert cfg.torch_compile is True

    def test_compile_kwargs_default_empty(self) -> None:
        cfg = TrainingConfig(max_epochs=5)
        assert cfg.compile_kwargs == {}

    def test_compile_kwargs_forwarded(self) -> None:
        kwargs = {"backend": "inductor", "mode": "reduce-overhead", "fullgraph": True}
        cfg = TrainingConfig(max_epochs=5, torch_compile=True, compile_kwargs=kwargs)
        assert cfg.compile_kwargs == kwargs
