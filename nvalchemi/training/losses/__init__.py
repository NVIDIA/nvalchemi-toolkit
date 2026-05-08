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
"""Loss-function abstractions, schedules, terms, and reductions.

Loss terms are :class:`BaseLossFunction` instances that consume
prediction and target tensors directly and return raw (unweighted) loss
tensors. :class:`ComposedLossFunction` owns the per-component weighting
— either plain floats or :class:`LossWeightSchedule` instances — and,
by default, renormalizes the effective weights to sum to ``1.0``.
Operator sugar (``3.0 * EnergyLoss() + 2.0 * ForceLoss()``) builds a
composition in one expression. Schedule instances attached to a
composition's weights are reconstructed by ``TrainingStrategy`` from
their ``(instance, spec)`` pair, mirroring the pattern used for models
and optimizers.
"""

from __future__ import annotations

from nvalchemi.training.losses.base import LossWeightSchedule
from nvalchemi.training.losses.composition import (
    BaseLossFunction,
    ComposedLossFunction,
    ComposedLossOutput,
    assert_same_shape,
)
from nvalchemi.training.losses.reductions import (
    frobenius_mse,
    per_graph_mean,
    per_graph_sum,
)
from nvalchemi.training.losses.schedules import (
    ConstantWeight,
    CosineWeight,
    LinearWeight,
    PiecewiseWeight,
)
from nvalchemi.training.losses.terms import (
    EnergyLoss,
    ForceLoss,
    StressLoss,
)

__all__ = [
    "BaseLossFunction",
    "ComposedLossFunction",
    "ComposedLossOutput",
    "ConstantWeight",
    "CosineWeight",
    "EnergyLoss",
    "ForceLoss",
    "LinearWeight",
    "LossWeightSchedule",
    "PiecewiseWeight",
    "StressLoss",
    "assert_same_shape",
    "frobenius_mse",
    "per_graph_mean",
    "per_graph_sum",
]
