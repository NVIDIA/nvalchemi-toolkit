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
from nvalchemi.data.level_storage import effective_dtype

if TYPE_CHECKING:
    from nvalchemi.models.base import BaseModelMixin

_DerivativeOperation = Literal["hvp", "dense_hessian"]
_DerivativeStrategy = Literal["loop", "vmap"]
_DerivativeExecutionKind = Literal["local", "distributed"]


class DerivativeNotSupported(NotImplementedError):
    """A model or execution context does not support a derivative request.

    Attributes
    ----------
    model_name : str
        Name of the model wrapper rejecting the request.
    operation : {"hvp", "dense_hessian"}
        Derivative operation requested by the caller.
    execution : {"local", "distributed"}
        Execution context in which the request was made.
    strategy : {"loop", "vmap"} or None
        Dense-Hessian strategy, or ``None`` for HVP requests.
    reason : str
        Explanation of why the request is unsupported.
    """

    def __init__(
        self,
        *,
        model_name: str,
        operation: _DerivativeOperation,
        execution: _DerivativeExecutionKind,
        strategy: _DerivativeStrategy | None,
        reason: str,
    ) -> None:
        """Initialize a contextual derivative capability error.

        Parameters
        ----------
        model_name : str
            Name of the model wrapper rejecting the request.
        operation : {"hvp", "dense_hessian"}
            Derivative operation requested by the caller.
        execution : {"local", "distributed"}
            Execution context in which the request was made.
        strategy : {"loop", "vmap"} or None
            Dense-Hessian strategy, or ``None`` for HVP requests.
        reason : str
            Explanation of why the request is unsupported.
        """
        self.model_name = model_name
        self.operation = operation
        self.execution = execution
        self.strategy = strategy
        self.reason = reason
        strategy_text = strategy if strategy is not None else "none"
        super().__init__(
            f"{model_name} does not support derivative operation '{operation}' "
            f"for execution='{execution}', strategy='{strategy_text}': {reason}"
        )

    def __reduce__(self) -> tuple[Any, tuple[str, str, str, str | None, str]]:
        """Reconstruct the exception from its contextual fields when unpickled."""
        return (
            _reconstruct_derivative_not_supported,
            (
                self.model_name,
                self.operation,
                self.execution,
                self.strategy,
                self.reason,
            ),
        )


def _reconstruct_derivative_not_supported(
    model_name: str,
    operation: _DerivativeOperation,
    execution: _DerivativeExecutionKind,
    strategy: _DerivativeStrategy | None,
    reason: str,
) -> DerivativeNotSupported:
    """Rebuild the public exception using its keyword-only constructor."""
    return DerivativeNotSupported(
        model_name=model_name,
        operation=operation,
        execution=execution,
        strategy=strategy,
        reason=reason,
    )


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
    raise DerivativeNotSupported(
        model_name=type(model).__name__,
        operation=request.operation,
        execution=request.execution,
        strategy=request.strategy,
        reason=reason,
    )


def _require_local_derivative_request(
    model: object,
    request: _DerivativeRequest,
) -> None:
    """Reject distributed Hessian requests before wrapper-specific checks."""
    if request.execution != "local":
        _reject_derivative_request(
            model,
            request,
            "distributed second-order derivatives are not supported",
        )


_CONFIG_OVERRIDE_UNSET = object()


@contextmanager
def _temporary_model_config(
    model: BaseModelMixin,
    *,
    active_outputs: Any = _CONFIG_OVERRIDE_UNSET,
    gradient_keys: Any = _CONFIG_OVERRIDE_UNSET,
) -> Iterator[None]:
    """Temporarily override selected config fields and restore their objects."""
    config = model.model_config
    saved: dict[str, Any] = {}
    try:
        if active_outputs is not _CONFIG_OVERRIDE_UNSET:
            saved["active_outputs"] = config.active_outputs
            config.active_outputs = active_outputs
        if gradient_keys is not _CONFIG_OVERRIDE_UNSET:
            saved["gradient_keys"] = config.gradient_keys
            config.gradient_keys = gradient_keys
        yield
    finally:
        for key, value in saved.items():
            setattr(config, key, value)


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


def _validate_hessian_vector(vector: Tensor, positions: Tensor) -> Tensor:
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


@dataclass(frozen=True, slots=True)
class _ValidatedDenseHessianRequest:
    """Dense Hessian inputs validated before any model evaluation."""

    positions: Tensor
    num_nodes: list[int]
    strategy: _DerivativeStrategy
    row_chunk_size: int | None

    @classmethod
    def build(
        cls,
        batch: Batch,
        strategy: Any,
        row_chunk_size: Any,
    ) -> _ValidatedDenseHessianRequest:
        """Validate dense-Hessian inputs before any model evaluation."""
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

        schema = batch.get_level_schema()
        schema.add_product_level("atom_atom", left="atoms", right="atoms")
        field_group = schema.attr_to_group.get("hessian")
        if field_group is not None and field_group != "atom_atom":
            raise ValueError(
                f"Field 'hessian' must belong to level 'atom_atom', not '{field_group}'"
            )
        declared_dtype = schema.dtypes.get("hessian")
        if declared_dtype is not None:
            expected_dtype = effective_dtype(declared_dtype)
            if isinstance(expected_dtype, str):
                raise ValueError(
                    f"Field 'hessian' has unsupported declared dtype '{declared_dtype}'"
                )
            if expected_dtype != positions.dtype:
                raise ValueError(
                    "Field 'hessian' has declared dtype "
                    f"{declared_dtype}, expected {positions.dtype}"
                )

        if "hessian" in batch:
            hessian = batch["hessian"]
            expected_shape = (sum(count * count for count in num_nodes), 3, 3)
            if tuple(hessian.shape) != expected_shape:
                raise ValueError(
                    f"Existing 'hessian' field must have shape {expected_shape}, "
                    f"got {tuple(hessian.shape)}"
                )
            if hessian.dtype != positions.dtype:
                raise ValueError(
                    "Existing 'hessian' field must have dtype "
                    f"{positions.dtype}, got {hessian.dtype}"
                )

        return cls(
            positions=positions,
            num_nodes=num_nodes,
            strategy=strategy,
            row_chunk_size=row_chunk_size,
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
        return _gradient_vector_product(
            gradient,
            graph.positions,
            validated,
            is_grads_batched=False,
        )

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


@contextmanager
def _prepare_derivative_graph(
    model: BaseModelMixin,
    batch: Batch,
    request: _DerivativeRequest,
) -> Iterator[_DerivativeGraph]:
    """Yield independent data and a connected energy graph for derivatives."""
    if not isinstance(batch, Batch):
        raise TypeError(f"batch must be a Batch, got {type(batch).__name__}")

    _require_local_derivative_request(model, request)
    model._validate_derivative_request(request)

    graph: _DerivativeGraph | None = None
    working_batch: Batch | None = None
    positions: Tensor | None = None
    energy: Tensor | None = None
    try:
        with torch.inference_mode(False), torch.enable_grad():
            working_batch = batch.clone()
            for key, value in list(working_batch):
                if isinstance(value, Tensor):
                    working_batch[key] = value.detach()

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

            with _temporary_model_config(
                model,
                active_outputs={"energy"},
                gradient_keys={"positions"},
            ):
                candidate_energy = model._derivative_energy(working_batch)

            if not isinstance(candidate_energy, Tensor):
                raise RuntimeError(
                    "Derivative energy must be a torch.Tensor, "
                    f"got {type(candidate_energy).__name__}"
                )
            expected_shape = (working_batch.num_graphs, 1)
            if tuple(candidate_energy.shape) != expected_shape:
                raise RuntimeError(
                    "Derivative energy must have shape "
                    f"{expected_shape}, got {tuple(candidate_energy.shape)}"
                )
            if candidate_energy.device != positions.device:
                raise RuntimeError(
                    "Derivative energy must be on the positions device "
                    f"{positions.device}, got {candidate_energy.device}"
                )
            if not candidate_energy.is_floating_point():
                raise RuntimeError(
                    "Derivative energy must have a floating-point dtype, "
                    f"got {candidate_energy.dtype}"
                )
            if not candidate_energy.requires_grad or candidate_energy.grad_fn is None:
                raise RuntimeError(
                    "Derivative energy must retain a graph connected to the position leaf"
                )
            energy = candidate_energy
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
