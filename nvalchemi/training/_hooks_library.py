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

"""Built-in training hooks: early stopping and stochastic weight averaging."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from loguru import logger
from torch.optim.swa_utils import AveragedModel, update_bn

from nvalchemi.training._stages import TrainingStageEnum
from nvalchemi.training._terminate import StopTraining

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

    from nvalchemi.models.base import BaseModelMixin
    from nvalchemi.training._hooks import TrainingContext


class EarlyStoppingHook:
    """Stop training when a monitored validation metric stops improving.

    Registered at :attr:`~TrainingStageEnum.AFTER_VALIDATION`; reads the
    metric from ``ctx.metrics[self.metric]`` after each validation pass.
    Raises :class:`~nvalchemi.training.StopTraining` when no improvement
    is observed for *patience* consecutive validations.

    Parameters
    ----------
    metric : str
        Key in ``ctx.metrics`` to monitor (e.g. ``"val_loss"``).
    patience : int
        Number of validations without improvement before stopping.
    mode : {"min", "max"}
        Whether a lower or higher metric value is better.

    Attributes
    ----------
    frequency : int
        Always ``1`` — checked every time the stage fires.
    stage : TrainingStageEnum
        ``AFTER_VALIDATION``.
    best : float
        Best metric value observed so far.
    counter : int
        Number of consecutive validations without improvement.
    """

    frequency: int = 1
    stage: TrainingStageEnum = TrainingStageEnum.AFTER_VALIDATION

    def __init__(
        self,
        metric: str = "val_loss",
        patience: int = 5,
        mode: Literal["min", "max"] = "min",
    ) -> None:
        self.metric = metric
        self.patience = patience
        self.mode = mode
        self.best: float = float("inf") if mode == "min" else float("-inf")
        self.counter: int = 0

    def __call__(
        self,
        ctx: TrainingContext,
        model: BaseModelMixin,
        trainer: Any,
    ) -> None:
        """Check the monitored metric and raise if patience is exhausted.

        Parameters
        ----------
        ctx : TrainingContext
            Current training context.
        model : BaseModelMixin
            The model being trained (unused).
        trainer : Any
            The trainer instance (unused).

        Raises
        ------
        StopTraining
            When the metric has not improved for *patience* validations.
        """
        val = ctx.metrics.get(self.metric)
        if val is None:
            return

        improved = val < self.best if self.mode == "min" else val > self.best
        if improved:
            self.best = val
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                raise StopTraining(
                    f"No improvement in '{self.metric}' for "
                    f"{self.patience} validations (best={self.best:.6g})"
                )
            logger.debug(
                "EarlyStopping: no improvement ({}/{})", self.counter, self.patience
            )


class SWAHook:
    """Update an averaged model with stochastic weight averaging.

    Registered at :attr:`~TrainingStageEnum.AFTER_OPTIMIZER_STEP`; begins
    accumulating weight averages once ``ctx.global_step`` reaches
    *swa_start_step*.

    Parameters
    ----------
    swa_model : AveragedModel
        The :class:`~torch.optim.swa_utils.AveragedModel` wrapping the
        training model.
    swa_start_step : int
        Global step at which to begin averaging.

    Attributes
    ----------
    frequency : int
        Always ``1`` — checked every optimizer step.
    stage : TrainingStageEnum
        ``AFTER_OPTIMIZER_STEP``.
    """

    frequency: int = 1
    stage: TrainingStageEnum = TrainingStageEnum.AFTER_OPTIMIZER_STEP

    def __init__(
        self,
        swa_model: AveragedModel,
        swa_start_step: int = 1000,
    ) -> None:
        self.swa_model = swa_model
        self.swa_start_step = swa_start_step

    def __call__(
        self,
        ctx: TrainingContext,
        model: BaseModelMixin,
        trainer: Any,
    ) -> None:
        """Update averaged model parameters if past the start step.

        Parameters
        ----------
        ctx : TrainingContext
            Current training context; ``ctx.global_step`` is checked.
        model : BaseModelMixin
            The model whose parameters are averaged into *swa_model*.
        trainer : Any
            The trainer instance (unused).
        """
        if ctx.global_step >= self.swa_start_step:
            self.swa_model.update_parameters(model)


class SWAFinalizeHook:
    """Recompute BatchNorm statistics for the averaged model at training end.

    Registered at :attr:`~TrainingStageEnum.ON_TRAINING_END`; performs a
    single forward pass over the training data to update running BN
    statistics in the averaged model.

    Parameters
    ----------
    swa_model : AveragedModel
        The :class:`~torch.optim.swa_utils.AveragedModel` to finalize.
    train_loader : DataLoader
        Training data loader used for the BN update pass.
    device : str or torch.device or None
        Device to transfer data to before the forward pass.  If ``None``,
        data is used as-is.

    Attributes
    ----------
    frequency : int
        Always ``1``.
    stage : TrainingStageEnum
        ``ON_TRAINING_END``.
    """

    frequency: int = 1
    stage: TrainingStageEnum = TrainingStageEnum.ON_TRAINING_END

    def __init__(
        self,
        swa_model: AveragedModel,
        train_loader: DataLoader,
        device: Any | None = None,
    ) -> None:
        self.swa_model = swa_model
        self.train_loader = train_loader
        self.device = device

    def __call__(
        self,
        ctx: TrainingContext,
        model: BaseModelMixin,
        trainer: Any,
    ) -> None:
        """Run :func:`~torch.optim.swa_utils.update_bn` on the averaged model.

        Parameters
        ----------
        ctx : TrainingContext
            Current training context (unused).
        model : BaseModelMixin
            The training model (unused — BN update uses *swa_model*).
        trainer : Any
            The trainer instance (unused).
        """
        logger.info("Finalizing SWA: updating BatchNorm statistics.")
        update_bn(self.train_loader, self.swa_model, device=self.device)
