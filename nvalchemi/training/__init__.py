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
"""Training utilities for MLIP models."""

from __future__ import annotations

from physicsnemo.distributed import DistributedManager

from nvalchemi.training._checkpoint import (  # noqa: F401
    load_training_checkpoint,
    save_training_checkpoint,
)
from nvalchemi.training._configs import (  # noqa: F401
    GradClipConfig,
    MixedPrecisionConfig,
    TrainingConfig,
)
from nvalchemi.training._distributed import (  # noqa: F401
    reduce_loss_across_ranks,
    wrap_model,
)
from nvalchemi.training._hooks import TrainingContext, TrainingHook  # noqa: F401
from nvalchemi.training._hooks_library import (  # noqa: F401
    EarlyStoppingHook,
    SWAFinalizeHook,
    SWAHook,
)
from nvalchemi.training._stages import TrainingStageEnum  # noqa: F401
from nvalchemi.training._terminate import (  # noqa: F401
    StopTraining,
    TerminateOnStepsHook,
)
from nvalchemi.training.trainer import Trainer, TrainingResult  # noqa: F401

__all__ = [
    "DistributedManager",
    "GradClipConfig",
    "MixedPrecisionConfig",
    "TrainingConfig",
    "TrainingContext",
    "Trainer",
    "TrainingResult",
]
