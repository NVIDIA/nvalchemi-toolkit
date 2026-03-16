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
"""Forces loss component."""

from __future__ import annotations

from typing import Literal

from torch import Tensor

from nvalchemi.training.losses._base import LossComponent


class ForceLoss(LossComponent):
    """Per-component squared error on node-level forces.

    Forces are already per-atom, so ``normalize_by_atoms`` defaults to
    ``False``.  The element-wise error computes the squared difference
    per xyz component, then sums over the spatial dimension to yield a
    scalar error per atom.

    The base class :meth:`compute` handles the segmented reduction over
    the batch dimension (sum per graph) and the final batch-level
    reduction (mean/sum).
    """

    def __init__(
        self,
        *,
        name: str = "forces",
        pred_key: str = "forces",
        target_key: str = "forces",
        weight: float = 1.0,
        reduction: Literal["mean", "sum"] = "mean",
        normalize_by_atoms: bool = False,
    ) -> None:
        super().__init__(
            name=name,
            pred_key=pred_key,
            target_key=target_key,
            weight=weight,
            level="node",
            reduction=reduction,
            normalize_by_atoms=normalize_by_atoms,
        )

    def elementwise_error(self, pred: Tensor, target: Tensor) -> Tensor:
        """Per-component squared error on force vectors.

        Parameters
        ----------
        pred : Tensor
            Predicted forces, shape ``(V, 3)``.
        target : Tensor
            Target forces, shape ``(V, 3)``.

        Returns
        -------
        Tensor
            Per-component squared error, shape ``(V, 3)``.  The base
            class reduces trailing dims (sum over xyz) before segmented
            reduction.
        """
        return (pred - target).square()
