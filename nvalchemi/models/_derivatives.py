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
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass
from types import TracebackType
from typing import TYPE_CHECKING, Any, Literal, Self

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


def _validate_hessian_vector(vector: Any, positions: Tensor) -> Tensor:
    """Validate one position-space vector against a reference tensor."""
    if not isinstance(vector, Tensor):
        raise TypeError(
            f"Hessian vector must be a torch.Tensor, got {type(vector).__name__}"
        )
    if not vector.is_floating_point():
        raise TypeError(
            f"Hessian vector must have a floating-point dtype, got {vector.dtype}"
        )
    if vector.shape != positions.shape:
        raise ValueError(
            "Hessian vector must have the same shape as positions "
            f"{tuple(positions.shape)}, got {tuple(vector.shape)}"
        )
    if vector.dtype != positions.dtype:
        raise ValueError(
            "Hessian vector must have the same dtype as positions "
            f"{positions.dtype}, got {vector.dtype}"
        )
    if vector.device != positions.device:
        raise ValueError(
            "Hessian vector must be on the same device as positions "
            f"{positions.device}, got {vector.device}"
        )
    return vector


def _position_gradient(graph: _DerivativeGraph) -> Tensor:
    """Construct the retained first position derivative for an HVP graph."""
    with torch.inference_mode(False), torch.enable_grad():
        gradient = torch.autograd.grad(
            graph.energy.sum(),
            graph.positions,
            create_graph=True,
            retain_graph=True,
            allow_unused=True,
        )[0]
    if gradient is None:
        raise RuntimeError("Derivative energy is not connected to the position leaf")
    return gradient


def _hessian_vector_product(
    gradient: Tensor,
    positions: Tensor,
    vector: Tensor,
) -> Tensor:
    """Evaluate one HVP while retaining the graph for subsequent products."""
    with torch.inference_mode(False), torch.enable_grad():
        if not gradient.requires_grad:
            return torch.zeros_like(positions)
        product = torch.autograd.grad(
            gradient,
            positions,
            grad_outputs=vector.detach(),
            create_graph=False,
            retain_graph=True,
            allow_unused=True,
        )[0]
        if product is None:
            return torch.zeros_like(positions)
        return product.detach()


class HessianOperator:
    """Matrix-free position Hessian retained at one fixed model snapshot.

    Instances are created by
    :meth:`~nvalchemi.models.base.BaseModelMixin.prepare_hessian`. The operator
    is immediately active and must be closed explicitly or used as a context
    manager to release its private autograd graph.
    """

    def __init__(
        self,
        context: AbstractContextManager[_DerivativeGraph],
    ) -> None:
        """Enter a prepared derivative context and retain its first gradient."""
        self._context: AbstractContextManager[_DerivativeGraph] | None = context
        self._graph: _DerivativeGraph | None = None
        self._gradient: Tensor | None = None
        self._closed = True

        try:
            graph = context.__enter__()
        except BaseException:
            self._context = None
            raise

        try:
            gradient = _position_gradient(graph)
        except BaseException as exc:
            try:
                context.__exit__(type(exc), exc, exc.__traceback__)
            finally:
                self._context = None
            raise

        self._graph = graph
        self._gradient = gradient
        self._closed = False

    def matvec(self, vector: Tensor) -> Tensor:
        """Multiply the retained position Hessian by one position-space vector.

        Parameters
        ----------
        vector : torch.Tensor
            Floating tensor matching the prepared positions in shape, dtype,
            and device.

        Returns
        -------
        torch.Tensor
            Detached Hessian-vector product aligned with the prepared positions.

        Raises
        ------
        RuntimeError
            If the operator has been closed.
        TypeError
            If the vector is not a floating-point tensor.
        ValueError
            If shape, dtype, or device does not match the prepared positions.
        """
        if self._closed:
            raise RuntimeError("HessianOperator is closed")
        graph = self._graph
        gradient = self._gradient
        if graph is None or gradient is None:
            raise RuntimeError("HessianOperator has no active derivative graph")
        validated = _validate_hessian_vector(vector, graph.positions)
        return _hessian_vector_product(gradient, graph.positions, validated)

    def close(self) -> None:
        """Release the retained derivative graph; repeated calls are harmless."""
        if self._closed:
            return

        context = self._context
        self._closed = True
        self._gradient = None
        self._graph = None
        self._context = None
        if context is not None:
            context.__exit__(None, None, None)

    def __enter__(self) -> Self:
        """Return this active operator without rebuilding its graph."""
        if self._closed:
            raise RuntimeError("HessianOperator is closed")
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Close the operator without suppressing an active exception."""
        self.close()


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
