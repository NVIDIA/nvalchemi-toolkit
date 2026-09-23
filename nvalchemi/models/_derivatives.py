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
from typing import TYPE_CHECKING, Any, Literal, NoReturn, Self

import torch
from torch import Tensor

from nvalchemi.data import Batch
from nvalchemi.data.level_storage import TORCH_DTYPE_MAP

if TYPE_CHECKING:
    from nvalchemi.models.base import BaseModelMixin

_DerivativeOperation = Literal["hvp", "dense_hessian"]
_DerivativeStrategy = Literal["loop", "vmap"]
_DerivativeExecutionKind = Literal["local", "distributed"]


def _reject_derivative_request(
    model: object,
    request: _DerivativeRequest,
    reason: str,
) -> NoReturn:
    """Raise the common contextual error for an unsupported derivative.

    Parameters
    ----------
    model : object
        Wrapper rejecting the request.  Its class name is included in the
        public error to identify the unsupported model boundary.
    request : _DerivativeRequest
        Context and operation being rejected.
    reason : str
        Configuration-specific explanation of the rejection.
    """
    strategy = request.strategy if request.strategy is not None else "none"
    raise NotImplementedError(
        f"{type(model).__name__} does not support derivative operation "
        f"'{request.operation}' for execution='{request.execution}', "
        f"strategy='{strategy}': {reason}"
    )


@dataclass(frozen=True, slots=True)
class _DerivativeRequest:
    """Describe one contextual second-order derivative request."""

    operation: _DerivativeOperation
    execution: _DerivativeExecutionKind
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
            raise RuntimeError(
                "Derivative energy is not connected to the position leaf"
            )

        if graph.positions.numel() == 0:
            return gradient
        if not gradient.requires_grad:
            raise RuntimeError(
                "First position derivative does not retain a differentiable graph"
            )

        try:
            torch.autograd.grad(
                gradient,
                graph.positions,
                grad_outputs=torch.ones_like(gradient),
                retain_graph=True,
                allow_unused=False,
            )
        except RuntimeError as exc:
            message = str(exc).lower()
            if "not have been used in the graph" not in message:
                raise
            raise RuntimeError(
                "First position derivative is not connected to the position leaf"
            ) from exc
    return gradient


def _gradient_vector_product(
    outputs: Tensor,
    positions: Tensor,
    grad_outputs: Tensor,
    *,
    is_grads_batched: bool,
) -> Tensor:
    """Differentiate a gradient view against positions for one or many seeds."""
    with torch.inference_mode(False), torch.enable_grad():
        if positions.numel() == 0:
            shape = (
                (grad_outputs.shape[0], *positions.shape)
                if is_grads_batched
                else positions.shape
            )
            return positions.new_empty(shape)
        if not outputs.requires_grad:
            raise RuntimeError(
                "First position derivative does not retain a differentiable graph"
            )

        if is_grads_batched:

            def _single_product(seed: Tensor) -> Tensor:
                product = torch.autograd.grad(
                    outputs,
                    positions,
                    grad_outputs=seed,
                    create_graph=False,
                    retain_graph=True,
                    allow_unused=False,
                )[0]
                return product

            return torch.vmap(_single_product)(grad_outputs.detach()).detach()

        product = torch.autograd.grad(
            outputs,
            positions,
            grad_outputs=grad_outputs.detach(),
            create_graph=False,
            retain_graph=True,
            allow_unused=False,
        )[0]
        return product.detach()


def _hessian_vector_product(
    gradient: Tensor,
    positions: Tensor,
    vector: Tensor,
) -> Tensor:
    """Evaluate one HVP while retaining the graph for subsequent products."""
    return _gradient_vector_product(
        gradient,
        positions,
        vector,
        is_grads_batched=False,
    )


def _validate_dense_hessian_inputs(
    batch: Batch,
    strategy: Any,
    row_chunk_size: Any,
) -> tuple[Tensor, list[int]]:
    """Validate dense-Hessian arguments before capability or model execution."""
    if not isinstance(batch, Batch):
        raise TypeError(f"batch must be a Batch, got {type(batch).__name__}")
    if strategy not in ("loop", "vmap"):
        raise ValueError(f"strategy must be 'loop' or 'vmap', got {strategy!r}")
    if row_chunk_size is not None:
        if not isinstance(row_chunk_size, int) or isinstance(row_chunk_size, bool):
            raise TypeError("row_chunk_size must be a positive integer or None")
        if row_chunk_size <= 0:
            raise ValueError("row_chunk_size must be positive")

    positions = getattr(batch, "positions", None)
    if not isinstance(positions, Tensor):
        raise RuntimeError("Dense Hessians require tensor positions")
    if not positions.is_floating_point():
        raise TypeError(
            "Dense Hessian positions must have a floating-point dtype, "
            f"got {positions.dtype}"
        )
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError(
            "Dense Hessian positions must have shape [total_atoms, 3], "
            f"got {tuple(positions.shape)}"
        )

    atomic_numbers = getattr(batch, "atomic_numbers", None)
    if not isinstance(atomic_numbers, Tensor):
        raise RuntimeError("Dense Hessians require tensor atomic_numbers")
    if atomic_numbers.ndim != 1:
        raise ValueError(
            "Dense Hessian atomic_numbers must have shape [total_atoms], "
            f"got {tuple(atomic_numbers.shape)}"
        )

    num_nodes = batch.num_nodes_list
    expected_atoms = sum(num_nodes)
    if positions.shape[0] != expected_atoms:
        raise ValueError(
            "Dense Hessian positions length must match active atom segmentation "
            f"{expected_atoms}, got {positions.shape[0]}"
        )
    if atomic_numbers.shape[0] != expected_atoms:
        raise ValueError(
            "Dense Hessian atomic_numbers length must match active atom "
            f"segmentation {expected_atoms}, got {atomic_numbers.shape[0]}"
        )
    return positions, num_nodes


def _validate_dense_hessian_storage(
    batch: Batch,
    *,
    dtype: torch.dtype,
    num_nodes: list[int],
) -> None:
    """Preflight canonical product storage without mutating the batch."""
    schema = batch.get_level_schema()
    schema.add_product_level("atom_atom", left="atoms", right="atoms")

    field_group = schema.attr_to_group.get("hessian")
    if field_group is not None and field_group != "atom_atom":
        raise ValueError(
            f"Field 'hessian' must belong to level 'atom_atom', not '{field_group}'"
        )
    declared_dtype = schema.dtypes.get("hessian")
    if declared_dtype is not None:
        try:
            expected_dtype = TORCH_DTYPE_MAP[declared_dtype]
        except KeyError as exc:
            raise ValueError(
                f"Field 'hessian' has unsupported declared dtype '{declared_dtype}'"
            ) from exc
        if expected_dtype != dtype:
            raise ValueError(
                f"Field 'hessian' has declared dtype {declared_dtype}, expected {dtype}"
            )

    if "hessian" not in batch:
        return
    hessian = batch["hessian"]
    expected_shape = (sum(count * count for count in num_nodes), 3, 3)
    if tuple(hessian.shape) != expected_shape:
        raise ValueError(
            f"Existing 'hessian' field must have shape {expected_shape}, "
            f"got {tuple(hessian.shape)}"
        )
    if hessian.dtype != dtype:
        raise ValueError(
            f"Existing 'hessian' field must have dtype {dtype}, got {hessian.dtype}"
        )


def _dense_hessian_blocks(
    graph: _DerivativeGraph,
    gradient: Tensor,
    num_nodes: list[int],
    *,
    strategy: _DerivativeStrategy,
    row_chunk_size: int | None,
) -> list[Tensor]:
    """Materialize detached within-system Hessian blocks from one graph."""
    blocks: list[Tensor] = []
    atom_start = 0
    with torch.inference_mode(False), torch.enable_grad():
        for atom_count in num_nodes:
            atom_stop = atom_start + atom_count
            row_count = 3 * atom_count
            if row_count == 0:
                blocks.append(graph.positions.new_empty((0, 0, 3, 3)).detach())
                atom_start = atom_stop
                continue

            local_gradient = gradient[atom_start:atom_stop]
            chunk_size = row_count if row_chunk_size is None else row_chunk_size
            row_chunks: list[Tensor] = []
            for row_start in range(0, row_count, chunk_size):
                row_stop = min(row_start + chunk_size, row_count)
                if strategy == "vmap":
                    rows = torch.arange(
                        row_start,
                        row_stop,
                        device=graph.positions.device,
                    )
                    seeds = graph.positions.new_zeros(
                        (row_stop - row_start, atom_count, 3)
                    )
                    seeds[
                        torch.arange(rows.shape[0], device=rows.device),
                        torch.div(rows, 3, rounding_mode="floor"),
                        torch.remainder(rows, 3),
                    ] = 1
                    products = _gradient_vector_product(
                        local_gradient,
                        graph.positions,
                        seeds,
                        is_grads_batched=True,
                    )
                    row_chunks.append(products[:, atom_start:atom_stop].clone())
                else:
                    loop_rows: list[Tensor] = []
                    for row in range(row_start, row_stop):
                        seed = torch.zeros_like(local_gradient)
                        seed.reshape(-1)[row] = 1
                        product = _gradient_vector_product(
                            local_gradient,
                            graph.positions,
                            seed,
                            is_grads_batched=False,
                        )
                        loop_rows.append(product[atom_start:atom_stop].clone())
                    row_chunks.append(torch.stack(loop_rows, dim=0))

            rows = torch.cat(row_chunks, dim=0)
            block = (
                rows.reshape(atom_count, 3, atom_count, 3)
                .permute(0, 2, 1, 3)
                .contiguous()
                .detach()
            )
            blocks.append(block)
            atom_start = atom_stop
    return blocks


def _attach_hessian_blocks(
    batch: Batch,
    blocks: list[Tensor],
    *,
    dtype: torch.dtype,
) -> None:
    """Attach canonical blocks through public Batch schema and field APIs."""
    batch.add_product_level("atom_atom", left="atoms", right="atoms")
    batch.add_key(
        "hessian",
        blocks,
        level="atom_atom",
        overwrite=True,
        dtype=dtype,
        payload_shape=(3, 3),
    )


class HessianOperator:
    """Matrix-free position Hessian for repeated products at one geometry.

    Obtain an operator through
    :meth:`~nvalchemi.models.base.BaseModelMixin.prepare_hessian`. Its
    :meth:`matvec` method computes ``(d^2 E / dR^2) @ v`` using an independent
    batch snapshot and a retained autograd graph. Keep model parameters,
    buffers, execution mode, and pipeline wiring unchanged while the operator
    is active. Call :meth:`close` or use a context manager to release the graph.
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
            model._copy_derivative_runtime_data(batch, working_batch)

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
