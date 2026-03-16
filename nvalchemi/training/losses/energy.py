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
"""Energy loss component."""

from __future__ import annotations

from typing import Literal

from torch import Tensor

from nvalchemi.training.losses._base import LossComponent


class EnergyLoss(LossComponent):
    """Mean squared error on system-level energies.

    Energy is an extensive quantity, so ``normalize_by_atoms`` defaults
    to ``True``; each per-graph loss is divided by the number of atoms
    in that graph before batch reduction.

    Attributes
    ----------
    error_fn : {"mse", "mae", "huber"}
        Element-wise error metric.  Default ``"mse"``.
    huber_delta : float
        Threshold for the Huber loss transition.  Only used when
        ``error_fn="huber"``.
    """

    def __init__(
        self,
        *,
        name: str = "energy",
        pred_key: str = "energies",
        target_key: str = "energies",
        weight: float = 1.0,
        reduction: Literal["mean", "sum"] = "mean",
        normalize_by_atoms: bool = True,
        error_fn: Literal["mse", "mae", "huber"] = "mse",
        huber_delta: float = 1.0,
    ) -> None:
        super().__init__(
            name=name,
            pred_key=pred_key,
            target_key=target_key,
            weight=weight,
            level="system",
            reduction=reduction,
            normalize_by_atoms=normalize_by_atoms,
        )
        self.error_fn = error_fn
        self.huber_delta = huber_delta

    def elementwise_error(self, pred: Tensor, target: Tensor) -> Tensor:
        """Compute per-element energy error.

        Parameters
        ----------
        pred : Tensor
            Predicted energies, shape ``(B, 1)``.
        target : Tensor
            Target energies, shape ``(B, 1)``.

        Returns
        -------
        Tensor
            Per-element error, same shape as *pred*.
        """
        diff = pred - target
        if self.error_fn == "mse":
            return diff.square()
        if self.error_fn == "mae":
            return diff.abs()
        # Huber
        abs_diff = diff.abs()
        quadratic = 0.5 * diff.square()
        linear = self.huber_delta * (abs_diff - 0.5 * self.huber_delta)
        return quadratic.where(abs_diff <= self.huber_delta, linear)
