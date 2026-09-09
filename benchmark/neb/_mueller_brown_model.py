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
"""Analytic Muller--Brown potential and reference transition state."""

from __future__ import annotations

from typing import Any

import torch

from nvalchemi.data import AtomicData, Batch
from nvalchemi.models.base import BaseModelMixin, ModelConfig

MULLER_BROWN_INITIAL = (-0.558, 1.442)
MULLER_BROWN_FINAL = (0.623, 0.028)
MULLER_BROWN_TRANSITION_STATE = (-0.82200156, 0.62431280)
MULLER_BROWN_TRANSITION_STATE_ENERGY = -40.6648435087


class MullerBrownModel(torch.nn.Module, BaseModelMixin):
    """Evaluate the canonical Muller--Brown analytic potential.

    Parameters
    ----------
    device
        Device on which to allocate the model buffers.
    dtype
        Floating-point dtype for the model buffers.
    """

    def __init__(self, device: torch.device, dtype: torch.dtype) -> None:
        super().__init__()
        self.model_config = ModelConfig(
            outputs=frozenset({"energy", "forces"}),
            autograd_outputs=frozenset(),
            autograd_inputs=frozenset(),
            neighbor_config=None,
            needs_pbc=False,
        )
        parameters = (
            ("coefficient", (-200.0, -100.0, -170.0, 15.0)),
            ("x_squared", (-1.0, -1.0, -6.5, 0.7)),
            ("xy", (0.0, 0.0, 11.0, 0.6)),
            ("y_squared", (-10.0, -10.0, -6.5, 0.7)),
            ("x_center", (1.0, 0.0, -0.5, -1.0)),
            ("y_center", (0.0, 0.5, 1.5, 1.0)),
        )
        for name, values in parameters:
            self.register_buffer(name, torch.tensor(values, device=device, dtype=dtype))

    @property
    def embedding_shapes(self) -> dict[str, tuple[int, ...]]:
        """Describe the embedding outputs.

        Returns
        -------
        dict[str, tuple[int, ...]]
            Empty mapping because the analytic model has no embedding outputs.
        """
        return {}

    def compute_embeddings(
        self, data: AtomicData | Batch, **_kwargs: Any
    ) -> AtomicData | Batch:
        """Return the input unchanged because embeddings are not defined.

        Parameters
        ----------
        data
            Atomic data or batch passed through the model.
        **_kwargs
            Unused embedding options accepted for interface compatibility.

        Returns
        -------
        AtomicData or Batch
            The original input object.
        """
        return data

    def forward(self, batch: Batch) -> dict[str, torch.Tensor]:
        """Return analytic energy and force values for each path image.

        Parameters
        ----------
        batch
            Batch containing the path-image positions.

        Returns
        -------
        dict[str, torch.Tensor]
            Energy and force tensors for every image.
        """
        x = batch.positions[:, 0:1]
        y = batch.positions[:, 1:2]
        dx = x - self.x_center
        dy = y - self.y_center
        exponent = (
            self.x_squared * dx.square()
            + self.xy * dx * dy
            + self.y_squared * dy.square()
        )
        terms = self.coefficient * torch.exp(exponent)
        gradient_x = torch.sum(
            terms * (2.0 * self.x_squared * dx + self.xy * dy), dim=-1
        )
        gradient_y = torch.sum(
            terms * (self.xy * dx + 2.0 * self.y_squared * dy), dim=-1
        )
        forces = torch.stack(
            (-gradient_x, -gradient_y, torch.zeros_like(gradient_x)), dim=-1
        )
        return {"energy": terms.sum(dim=-1, keepdim=True), "forces": forces}
