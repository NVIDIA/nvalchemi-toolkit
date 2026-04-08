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
"""Explicit derivative requests for composable model execution.

This module exposes :class:`DerivativeStep`, the explicit derivative node
accepted by :class:`~nvalchemi.models.composable.ComposableModelWrapper`.
It can request standard energy derivatives such as forces and stresses, or
custom autograd products such as Jacobians of dipoles with respect to
positions.

Usage
-----
Request standard forces and stresses::

    step = DerivativeStep(forces=True, stresses=True)

Request a custom derivative product::

    step = DerivativeStep(
        specs={"bec_tensors": ("dipoles", "positions", "jacobian")},
    )
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal, NamedTuple, TypeAlias

import torch
from torch import Tensor

from nvalchemi.data import Batch
from nvalchemi.models.base import PipelineContext

__all__ = ["DerivativeSpec", "DerivativeStep"]

DerivativeMode: TypeAlias = Literal["grad", "jacobian", "stress"]


@dataclass(frozen=True, slots=True)
class DerivativeSpec:
    """Normalized derivative request.

    Attributes
    ----------
    source : str
        Context key holding the tensor to differentiate.
    wrt : str
        Autograd target name with respect to which the derivative is taken.
    mode : {"grad", "jacobian", "stress"}
        Derivative mode used to interpret the autograd result.
    negate : bool, default=False
        Whether to negate the resulting derivative tensor.
    accumulate : bool, default=False
        Whether the result should be added to an existing output key.
    """

    source: str
    wrt: str
    mode: DerivativeMode
    negate: bool = False
    accumulate: bool = False

    def __post_init__(self) -> None:
        if self.mode not in {"grad", "jacobian", "stress"}:
            raise ValueError(
                "DerivativeSpec.mode must be one of "
                f"{sorted({'grad', 'jacobian', 'stress'})!r}, got {self.mode!r}"
            )

    @property
    def grad_mode(self) -> Literal["grad", "jacobian"]:
        """Return the grouped autograd mode used internally."""

        return "grad" if self.mode in {"grad", "stress"} else "jacobian"


DerivativeOutputSpec: TypeAlias = tuple[str, str, DerivativeMode] | DerivativeSpec


class _ResolvedDerivative(NamedTuple):
    """Resolved derivative request ready for grouped execution."""

    output_name: str
    spec: DerivativeSpec
    source_value: Tensor
    target_tensor: Tensor


class DerivativeStep:
    """Terminal derivative step for autograd-based derivative products.

    Parameters
    ----------
    forces
        Request conservative forces from ``energies`` with respect to
        ``positions``.
    stresses
        Request stresses from ``energies`` with respect to the internal
        ``cell_scaling`` target.
    specs
        Optional custom derivative products keyed by output name.
    jacobian_chunk_size
        Optional chunk size used when materializing large Jacobians.
    create_graph
        Whether to retain higher-order gradient information.
    """

    _BUILTIN_SPECS: dict[str, DerivativeSpec] = {
        "forces": DerivativeSpec("energies", "positions", "grad", negate=True, accumulate=True),
        "stresses": DerivativeSpec("energies", "cell_scaling", "stress", accumulate=True),
    }

    def __init__(
        self,
        *,
        forces: bool = False,
        stresses: bool = False,
        specs: Mapping[str, DerivativeOutputSpec] | None = None,
        jacobian_chunk_size: int | None = None,
        create_graph: bool = False,
    ) -> None:
        if jacobian_chunk_size is not None and jacobian_chunk_size <= 0:
            raise ValueError(
                "jacobian_chunk_size must be a positive integer or None, "
                f"got {jacobian_chunk_size!r}"
            )
        derivatives: dict[str, DerivativeSpec] = {}
        enabled_builtins: set[str] = set()
        if forces:
            derivatives["forces"] = self._BUILTIN_SPECS["forces"]
            enabled_builtins.add("forces")
        if stresses:
            derivatives["stresses"] = self._BUILTIN_SPECS["stresses"]
            enabled_builtins.add("stresses")
        if specs is not None:
            overlap = enabled_builtins & set(specs)
            if overlap:
                raise ValueError(
                    f"Specs keys overlap with builtin outputs: {sorted(overlap)}"
                )
            for output_name, spec_value in specs.items():
                normalized = self._normalize_output_spec(output_name, spec_value)
                derivatives[output_name] = DerivativeSpec(*normalized)
        if not derivatives:
            raise ValueError(
                "DerivativeStep has no derivative outputs configured. "
                "Set forces=True, stresses=True, or provide specs."
            )
        self._derivatives = derivatives
        self._forces = forces
        self._stresses = stresses
        self.outputs = {name: (spec.source, spec.wrt, spec.mode) for name, spec in derivatives.items()}
        self.jacobian_chunk_size = jacobian_chunk_size
        self.create_graph = create_graph

    @staticmethod
    def _normalize_output_spec(
        output_name: str,
        spec: DerivativeOutputSpec,
    ) -> tuple[str, str, DerivativeMode]:
        """Normalize one derivative output spec to tuple form."""

        if isinstance(spec, DerivativeSpec):
            return spec.source, spec.wrt, spec.mode
        if not isinstance(spec, tuple) or len(spec) != 3:
            raise TypeError(
                "DerivativeStep output spec must be a DerivativeSpec or a "
                f"(source, wrt, mode) tuple, got {type(spec)} for {output_name!r}"
            )
        source, wrt, mode = spec
        normalized_spec = DerivativeSpec(source, wrt, mode)
        return normalized_spec.source, normalized_spec.wrt, normalized_spec.mode

    def __repr__(self) -> str:
        parts: list[str] = []
        if self._forces:
            parts.append("forces=True")
        if self._stresses:
            parts.append("stresses=True")
        custom = {key: value for key, value in self.outputs.items() if key not in self._BUILTIN_SPECS}
        if custom:
            parts.append(f"specs={custom!r}")
        if self.jacobian_chunk_size is not None:
            parts.append(f"jacobian_chunk_size={self.jacobian_chunk_size!r}")
        if self.create_graph:
            parts.append("create_graph=True")
        return f"{type(self).__name__}({', '.join(parts)})"

    def derivative_targets(self) -> frozenset[str]:
        """Return all derivative target names needed for this step.

        Returns
        -------
        frozenset[str]
            Derivative targets that must be activated in the composable
            autograd region before this step executes.
        """

        return frozenset(spec.wrt for spec in self._derivatives.values())

    @staticmethod
    def _resolve_source(ctx: PipelineContext, spec: DerivativeSpec) -> Tensor:
        if spec.source in ctx.autograd_registered_outputs:
            return ctx.autograd_registered_outputs[spec.source]
        source_value = ctx[spec.source]
        if not isinstance(source_value, Tensor):
            raise TypeError(
                f"Derivative source '{spec.source}' must be a torch.Tensor, "
                f"got {type(source_value)}"
            )
        return source_value

    @staticmethod
    def _resolve_target(ctx: PipelineContext, spec: DerivativeSpec) -> Tensor:
        return ctx.get_autograd_target(spec.wrt)

    def _active_derivatives(self, ctx: PipelineContext) -> dict[str, DerivativeSpec]:
        active: dict[str, DerivativeSpec] = {}
        for name, spec in self._derivatives.items():
            if name == "stresses" and spec.wrt not in ctx.autograd_derivative_targets:
                continue
            active[name] = spec
        return active

    def _group_derivatives(
        self,
        active: dict[str, DerivativeSpec],
        ctx: PipelineContext,
    ) -> list[list[_ResolvedDerivative]]:
        groups: list[list[_ResolvedDerivative]] = []
        grouped_by_key: dict[tuple[str, Literal["grad", "jacobian"], int], list[_ResolvedDerivative]] = {}
        for output_name, spec in active.items():
            source_value = self._resolve_source(ctx, spec)
            target_tensor = self._resolve_target(ctx, spec)
            resolved = _ResolvedDerivative(output_name, spec, source_value, target_tensor)
            key = (spec.wrt, spec.grad_mode, id(target_tensor))
            group = grouped_by_key.get(key)
            if group is None:
                group = []
                grouped_by_key[key] = group
                groups.append(group)
            group.append(resolved)
        return groups

    def _compute_grad_group(
        self,
        group: list[_ResolvedDerivative],
        ctx: PipelineContext,
        *,
        retain_graph: bool,
    ) -> None:
        target_tensor = group[0].target_tensor
        scalar_outputs = torch.stack([item.source_value.sum() for item in group])
        num_outputs = scalar_outputs.numel()
        grad_outputs = torch.eye(
            num_outputs,
            dtype=scalar_outputs.dtype,
            device=scalar_outputs.device,
        )
        (batched_grad,) = torch.autograd.grad(
            scalar_outputs,
            target_tensor,
            grad_outputs=grad_outputs,
            create_graph=self.create_graph,
            retain_graph=retain_graph or self.create_graph,
            allow_unused=False,
            is_grads_batched=True,
        )
        for item, grad in zip(group, batched_grad, strict=True):
            value = grad
            if item.spec.mode == "stress":
                value = self._normalize_stress(ctx, value)
            if item.spec.negate:
                value = -value
            if item.spec.accumulate:
                ctx.accumulate(item.output_name, value)
            else:
                ctx[item.output_name] = value

    def _compute_jacobian_group(
        self,
        group: list[_ResolvedDerivative],
        ctx: PipelineContext,
        *,
        retain_graph: bool,
    ) -> None:
        target_tensor = group[0].target_tensor
        flat_sources = [item.source_value.reshape(-1) for item in group]
        joint_source = torch.cat(flat_sources, dim=0)
        joint_jacobian = self._compute_jacobian(
            joint_source,
            target_tensor,
            retain_graph=retain_graph,
        ).reshape(joint_source.shape + target_tensor.shape)

        start = 0
        for item, flat_source in zip(group, flat_sources, strict=True):
            stop = start + flat_source.numel()
            value = joint_jacobian[start:stop].reshape(item.source_value.shape + target_tensor.shape)
            if item.spec.negate:
                value = -value
            if item.spec.accumulate:
                ctx.accumulate(item.output_name, value)
            else:
                ctx[item.output_name] = value
            start = stop

    def compute(
        self,
        batch: Batch,
        ctx: PipelineContext,
    ) -> None:
        """Perform all requested derivative operations.

        Parameters
        ----------
        batch
            Input batch for the current composite execution.  The batch is
            not modified directly; derivative outputs are written into
            ``ctx``.
        ctx
            Runtime context containing registered autograd outputs and
            derivative targets.
        """

        del batch
        active = self._active_derivatives(ctx)
        if not active:
            return
        groups = self._group_derivatives(active, ctx)
        num_groups = len(groups)
        for index, group in enumerate(groups):
            retain_graph = index < num_groups - 1
            grad_mode = group[0].spec.grad_mode
            if grad_mode == "grad":
                self._compute_grad_group(group, ctx, retain_graph=retain_graph)
            elif grad_mode == "jacobian":
                self._compute_jacobian_group(group, ctx, retain_graph=retain_graph)
            else:
                raise ValueError(f"Unknown derivative mode: {grad_mode!r}")

    @staticmethod
    def _normalize_stress(ctx: PipelineContext, grad: Tensor) -> Tensor:
        cell = ctx.autograd_input_overrides.get("cell")
        if cell is None:
            raise ValueError("Stress computation requires cell (via cell_scaling target).")
        volume = torch.abs(torch.linalg.det(cell))
        if volume.ndim == 0:
            return grad / volume
        return grad / volume.unsqueeze(-1).unsqueeze(-1)

    def _compute_jacobian(
        self,
        source_value: Tensor,
        target_tensor: Tensor,
        *,
        retain_graph: bool,
    ) -> Tensor:
        flat_source = source_value.reshape(-1)
        num_outputs = flat_source.numel()
        chunk_size = self.jacobian_chunk_size or num_outputs
        jacobian_blocks: list[Tensor] = []

        for start in range(0, num_outputs, chunk_size):
            stop = min(start + chunk_size, num_outputs)
            grad_outputs = torch.zeros(
                (stop - start, num_outputs),
                dtype=source_value.dtype,
                device=source_value.device,
            )
            row_index = torch.arange(stop - start, device=source_value.device)
            col_index = torch.arange(start, stop, device=source_value.device)
            grad_outputs[row_index, col_index] = 1.0
            (jacobian_block,) = torch.autograd.grad(
                source_value,
                target_tensor,
                grad_outputs=grad_outputs.reshape(stop - start, *source_value.shape),
                create_graph=self.create_graph,
                retain_graph=(retain_graph or stop < num_outputs or self.create_graph),
                allow_unused=False,
                is_grads_batched=True,
            )
            jacobian_blocks.append(jacobian_block)

        return torch.cat(jacobian_blocks, dim=0).reshape(
            source_value.shape + target_tensor.shape
        )
