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
"""Training framework for ALCHEMI — stages, specs, losses, and checkpoint I/O."""

from __future__ import annotations

from nvalchemi.training._checkpoint import (
    CheckpointManifest,
    CheckpointValidator,
    load_checkpoint,
    save_checkpoint,
)
from nvalchemi.training._spec import (
    BaseSpec,
    create_model_spec,
    create_model_spec_from_json,
    register_type_serializer,
)
from nvalchemi.training._stages import TrainingStage
from nvalchemi.training.hooks import CheckpointHook, EMAHook
from nvalchemi.training.losses import (
    BaseLossFunction,
    ComposedLossFunction,
    ComposedLossOutput,
    ConstantWeight,
    CosineWeight,
    EnergyLoss,
    ForceLoss,
    LinearWeight,
    LossWeightSchedule,
    PiecewiseWeight,
    StressLoss,
    loss_component_to_spec,
)
from nvalchemi.training.optimizers import (
    OptimizerConfig,
    setup_optimizers,
    step_lr_schedulers,
    step_optimizers,
    zero_gradients,
)
from nvalchemi.training.runtime import (
    configure_dataloader,
    configure_parallelism,
    freeze_unconfigured_models,
    move_to_devices,
)
from nvalchemi.training.strategy import TrainingStrategy, default_training_fn

__all__ = [
    "BaseLossFunction",
    "BaseSpec",
    "CheckpointManifest",
    "CheckpointHook",
    "CheckpointValidator",
    "ComposedLossFunction",
    "ComposedLossOutput",
    "ConstantWeight",
    "CosineWeight",
    "EMAHook",
    "EnergyLoss",
    "ForceLoss",
    "LinearWeight",
    "LossWeightSchedule",
    "OptimizerConfig",
    "PiecewiseWeight",
    "StressLoss",
    "TrainingStage",
    "TrainingStrategy",
    "configure_dataloader",
    "configure_parallelism",
    "create_model_spec",
    "create_model_spec_from_json",
    "default_training_fn",
    "freeze_unconfigured_models",
    "loss_component_to_spec",
    "load_checkpoint",
    "move_to_devices",
    "register_type_serializer",
    "save_checkpoint",
    "setup_optimizers",
    "step_lr_schedulers",
    "step_optimizers",
    "zero_gradients",
]
