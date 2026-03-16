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
"""Stress loss component."""

from __future__ import annotations

from typing import Literal

from torch import Tensor

from nvalchemi.training.losses._base import LossComponent


class StressLoss(LossComponent):
    """Frobenius norm squared of the stress tensor difference.

    Stress tensors are system-level with shape ``(B, 3, 3)``.
    The elementwise error computes the squared difference per element;
    the base class sums over trailing dims (the 3x3 matrix) to get a
    per-graph Frobenius norm squared.
    """

    def __init__(
        self,
        *,
        name: str = "stress",
        pred_key: str = "stresses",
        target_key: str = "stresses",
        weight: float = 1.0,
        reduction: Literal["mean", "sum"] = "mean",
        normalize_by_atoms: bool = False,
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

    def elementwise_error(self, pred: Tensor, target: Tensor) -> Tensor:
        """Squared element-wise difference of stress tensors.

        Parameters
        ----------
        pred : Tensor
            Predicted stresses, shape ``(B, 3, 3)``.
        target : Tensor
            Target stresses, shape ``(B, 3, 3)``.

        Returns
        -------
        Tensor
            Per-element squared error, shape ``(B, 3, 3)``.  The base
            class sums over trailing dims to get the Frobenius norm
            squared per graph.
        """
        return (pred - target).square()
