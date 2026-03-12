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

"""Graceful termination primitives for the training loop."""

from __future__ import annotations

from typing import Any

from nvalchemi.models.base import BaseModelMixin
from nvalchemi.training._hooks import TrainingContext
from nvalchemi.training._stages import TrainingStageEnum


class StopTraining(Exception):
    """Sentinel exception raised by hooks to request graceful training shutdown.

    When caught by ``Trainer.fit()``, the trainer performs orderly cleanup
    (final checkpoint, logging) before returning instead of crashing.
    Parameters

    ----------
    message : str, optional
        Human-readable reason for the stop.
    """


class TerminateOnStepsHook:
    """Hook that raises :class:`StopTraining` after a fixed number of stage firings.

    Parameters
    ----------
    max_count : int
        Number of times the monitored stage must fire before training is
        stopped.
    stage : TrainingStageEnum
        The training stage whose cumulative count is compared against
        *max_count*.  Defaults to ``AFTER_STEP``.

    Attributes
    ----------
    frequency : int
        Always ``1`` — the hook checks on every firing of its stage.
    stage : TrainingStageEnum
        The stage this hook is registered on.
    """

    frequency: int = 1

    def __init__(
        self,
        max_count: int,
        stage: TrainingStageEnum = TrainingStageEnum.AFTER_STEP,
    ) -> None:
        self.max_count = max_count
        self.stage = stage

    def __call__(
        self,
        ctx: TrainingContext,
        model: BaseModelMixin,
        trainer: Any,
    ) -> None:
        """Raise :class:`StopTraining` when the stage count reaches *max_count*.

        Parameters
        ----------
        ctx : TrainingContext
            Current training context; ``ctx.stage_counts[self.stage]`` is read.
        model : BaseModelMixin
            The model being trained (unused by this hook).
        trainer : Any
            The trainer instance (unused by this hook).

        Raises
        ------
        StopTraining
            When ``ctx.stage_counts[self.stage] >= self.max_count``.
        """
        if ctx.stage_counts.get(self.stage, 0) >= self.max_count:
            raise StopTraining(f"Reached {self.max_count} firings of {self.stage.name}")
