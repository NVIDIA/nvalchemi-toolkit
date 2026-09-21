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

"""Private graph-preserving derivative execution helpers."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import torch
from torch import Tensor

from nvalchemi.data import Batch

if TYPE_CHECKING:
    from nvalchemi.models.base import BaseModelMixin

_DerivativeOperation = Literal["hvp", "dense_hessian"]
_DerivativeStrategy = Literal["loop", "vmap"]
_DerivativeExecutionKind = Literal["local", "distributed"]
_DerivativeExecutionMode = Literal["eager", "compiled"]


@dataclass(frozen=True, slots=True)
class _DerivativeRequest:
    """Describe one contextual second-order derivative request."""

    operation: _DerivativeOperation
    execution: _DerivativeExecutionKind
    mode: _DerivativeExecutionMode
    strategy: _DerivativeStrategy | None

    def __post_init__(self) -> None:
        """Validate the request before capability checks or model execution."""
        if self.operation not in {"hvp", "dense_hessian"}:
            raise ValueError(
                f"operation must be 'hvp' or 'dense_hessian', got {self.operation!r}"
            )
        if self.execution not in {"local", "distributed"}:
            raise ValueError(
                f"execution must be 'local' or 'distributed', got {self.execution!r}"
            )
        if self.mode not in {"eager", "compiled"}:
            raise ValueError(f"mode must be 'eager' or 'compiled', got {self.mode!r}")
        if self.strategy not in {None, "loop", "vmap"}:
            raise ValueError(
                f"strategy must be None, 'loop', or 'vmap', got {self.strategy!r}"
            )
        if self.operation == "hvp" and self.strategy is not None:
            raise ValueError("HVP requests must not specify a strategy")
        if self.operation == "dense_hessian" and self.strategy is None:
            raise ValueError(
                "Dense-Hessian requests must specify strategy='loop' or 'vmap'"
            )


@dataclass(slots=True)
class _DerivativeGraph:
    """Graph-connected state yielded while derivative preparation is active."""

    data: Batch
    positions: Tensor
    energy: Tensor


def _detach_batch_tensors(batch: Batch) -> None:
    """Detach every materialized tensor in an already independent batch."""
    for key, value in list(batch):
        if isinstance(value, Tensor):
            batch[key] = value.detach()


def _validate_energy(energy: Any, batch: Batch, positions: Tensor) -> Tensor:
    """Validate the graph-connected per-system energy contract."""
    if not isinstance(energy, Tensor):
        raise RuntimeError(
            f"Derivative energy must be a torch.Tensor, got {type(energy).__name__}"
        )
    expected_shape = (batch.num_graphs, 1)
    if tuple(energy.shape) != expected_shape:
        raise RuntimeError(
            "Derivative energy must have shape "
            f"{expected_shape}, got {tuple(energy.shape)}"
        )
    if energy.device != positions.device:
        raise RuntimeError(
            "Derivative energy must be on the positions device "
            f"{positions.device}, got {energy.device}"
        )
    if not energy.is_floating_point():
        raise RuntimeError(
            f"Derivative energy must have a floating-point dtype, got {energy.dtype}"
        )
    if not energy.requires_grad or energy.grad_fn is None:
        raise RuntimeError(
            "Derivative energy must retain a graph connected to the position leaf"
        )
    return energy


@contextmanager
def _prepare_derivative_graph(
    model: BaseModelMixin,
    batch: Batch,
    request: _DerivativeRequest,
) -> Iterator[_DerivativeGraph]:
    """Yield independent data and a connected energy graph for derivatives."""
    if not isinstance(batch, Batch):
        raise TypeError(f"batch must be a Batch, got {type(batch).__name__}")

    model._validate_derivative_request(request)

    graph: _DerivativeGraph | None = None
    working_batch: Batch | None = None
    positions: Tensor | None = None
    energy: Tensor | None = None
    try:
        with torch.inference_mode(False), torch.enable_grad():
            working_batch = batch.clone()
            _detach_batch_tensors(working_batch)

            stored_positions = getattr(working_batch, "positions", None)
            if not isinstance(stored_positions, Tensor):
                raise RuntimeError(
                    "Derivative graph preparation requires tensor positions"
                )
            if not stored_positions.is_floating_point():
                raise TypeError(
                    "Derivative positions must have a floating-point dtype, "
                    f"got {stored_positions.dtype}"
                )
            positions = stored_positions.detach().clone().requires_grad_(True)
            working_batch["positions"] = positions

            config = model.model_config
            saved_active_outputs = config.active_outputs
            saved_gradient_keys = config.gradient_keys
            try:
                config.active_outputs = {"energy"}
                config.gradient_keys = {"positions"}
                candidate_energy = model._derivative_energy(working_batch)
            finally:
                config.active_outputs = saved_active_outputs
                config.gradient_keys = saved_gradient_keys

            energy = _validate_energy(candidate_energy, working_batch, positions)
            graph = _DerivativeGraph(
                data=working_batch,
                positions=positions,
                energy=energy,
            )

        yield graph
    finally:
        graph = None
        energy = None
        positions = None
        working_batch = None
