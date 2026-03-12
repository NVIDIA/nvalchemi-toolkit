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
"""Training configuration models."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any, Literal

import torch
from pydantic import BaseModel, ConfigDict, Field, model_validator

_AMP_DTYPE_MAP: dict[str, torch.dtype] = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


class MixedPrecisionConfig(BaseModel):
    """Configuration for automatic mixed-precision training.

    Attributes
    ----------
    enabled : bool
        Whether to enable mixed-precision training.
    amp_type : Literal["fp16", "bf16"]
        Precision type for the autocast context.
    grad_scaler : bool
        Whether to use gradient scaling.  Only meaningful for ``fp16``;
        automatically forced to ``False`` when *amp_type* is ``"bf16"``.
    """

    enabled: Annotated[bool, Field(description="Enable mixed-precision training.")] = (
        False
    )
    amp_type: Annotated[
        Literal["fp16", "bf16"],
        Field(description="Precision type for autocast."),
    ] = "bf16"
    grad_scaler: Annotated[
        bool,
        Field(description="Use gradient scaling (fp16 only)."),
    ] = True

    @model_validator(mode="after")
    def _disable_scaler_for_bf16(self) -> MixedPrecisionConfig:
        """Force ``grad_scaler`` to False when using bf16."""
        if self.amp_type == "bf16":
            self.grad_scaler = False
        return self

    @property
    def torch_dtype(self) -> torch.dtype:
        """Return the :class:`torch.dtype` corresponding to *amp_type*."""
        return _AMP_DTYPE_MAP[self.amp_type]


class GradClipConfig(BaseModel):
    """Configuration for gradient clipping.

    Attributes
    ----------
    method : Literal["norm", "value"]
        Clipping method.  ``"norm"`` uses
        :func:`torch.nn.utils.clip_grad_norm_`;
        ``"value"`` uses :func:`torch.nn.utils.clip_grad_value_`.
    max_value : float
        Threshold passed to the clipping function.
    """

    model_config = ConfigDict(frozen=True)

    method: Annotated[
        Literal["norm", "value"],
        Field(description="Clipping method: 'norm' or 'value'."),
    ] = "norm"
    max_value: Annotated[
        float,
        Field(ge=0, description="Maximum gradient norm or value."),
    ]


class TrainingConfig(BaseModel):
    """Top-level training hyper-parameters.

    Attributes
    ----------
    max_epochs : int
        Maximum number of training epochs.
    grad_clip : GradClipConfig or None
        Gradient clipping configuration; ``None`` disables clipping.
    grad_accumulation_steps : int
        Number of micro-batches to accumulate before an optimizer step.
    val_every_n_epochs : int
        Run validation every *n* epochs.
    checkpoint_every_n_epochs : int
        Save a checkpoint every *n* epochs.
    checkpoint_dir : Path
        Directory for checkpoint files.
    resume_from : Path or None
        Path to a checkpoint to resume from; ``None`` starts fresh.
    mixed_precision : MixedPrecisionConfig
        Mixed-precision settings.
    strategy : Literal["ddp", "fsdp"]
        Distributed strategy to use.
    torch_compile : bool
        Whether to apply :func:`torch.compile` to the model before training.
    compile_kwargs : dict[str, Any]
        Keyword arguments forwarded to :func:`torch.compile` (e.g.
        ``backend``, ``mode``, ``fullgraph``).
    log_every_n_steps : int
        Logging frequency in training steps.
    seed : int
        Random seed for reproducibility.
    """

    max_epochs: Annotated[
        int,
        Field(ge=1, description="Maximum number of training epochs."),
    ]
    grad_clip: Annotated[
        GradClipConfig | None,
        Field(description="Gradient clipping config; None disables."),
    ] = None
    grad_accumulation_steps: Annotated[
        int,
        Field(ge=1, description="Micro-batches to accumulate before optimizer step."),
    ] = 1
    val_every_n_epochs: Annotated[
        int,
        Field(ge=1, description="Run validation every n epochs."),
    ] = 1
    checkpoint_every_n_epochs: Annotated[
        int,
        Field(ge=1, description="Save a checkpoint every n epochs."),
    ] = 1
    checkpoint_dir: Annotated[
        Path,
        Field(description="Directory for checkpoint files."),
    ] = Path("checkpoints")
    resume_from: Annotated[
        Path | None,
        Field(description="Checkpoint path to resume from; None starts fresh."),
    ] = None
    mixed_precision: Annotated[
        MixedPrecisionConfig,
        Field(description="Mixed-precision settings."),
    ] = MixedPrecisionConfig()
    strategy: Annotated[
        Literal["ddp", "fsdp"],
        Field(description="Distributed strategy."),
    ] = "ddp"
    torch_compile: Annotated[
        bool,
        Field(description="Apply torch.compile to the model before training."),
    ] = False
    compile_kwargs: Annotated[
        dict[str, Any],
        Field(
            default_factory=dict,
            description="Keyword arguments forwarded to torch.compile.",
        ),
    ]
    log_every_n_steps: Annotated[
        int,
        Field(ge=1, description="Log every n training steps."),
    ] = 10
    seed: Annotated[
        int,
        Field(description="Random seed for reproducibility."),
    ] = 42
