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

"""Public configuration for NEB."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

import torch
import warp as wp
from torch import Tensor

from nvalchemi.data import GroupLayout
from nvalchemi.dynamics.base import DynamicsStage
from nvalchemi.dynamics.paths.neb.equations import (
    climbing_image_effective_force,
    improved_tangent_weights,
    neb_effective_force_from_gram_stats,
)

__all__ = [
    "ConstantSpringConfig",
    "NEBMethod",
    "SpringConfig",
    "SpringContext",
]


@dataclass(frozen=True, slots=True)
class SpringContext:
    """Read-only inputs available to spring-constant policies.

    Tensor fields share storage with the active NEB state and must not be
    modified in-place by implementations.

    Parameters
    ----------
    energies : Tensor
        Flattened physical image energies, shape ``(num_images,)``.
    positions : Tensor
        Packed atomic positions, shape ``(num_atoms, 3)``.
    physical_forces : Tensor
        Packed physical model forces, shape ``(num_atoms, 3)``.
    image_ptr : Tensor
        Atom offsets delimiting packed images, shape ``(num_images + 1,)``.
    layout : GroupLayout
        Mapping from packed images to NEB paths.
    cell : Tensor
        Representative cell for each path, shape ``(num_paths, 3, 3)``.
    inv_cell : Tensor
        Inverse representative cell for each path, shape ``(num_paths, 3, 3)``.
    pbc : Tensor
        Periodic-boundary flags for each path, shape ``(num_paths, 3)``.
    step_count : int
        Current dynamics step.
    """

    energies: Tensor
    positions: Tensor
    physical_forces: Tensor
    image_ptr: Tensor
    layout: GroupLayout
    cell: Tensor
    inv_cell: Tensor
    pbc: Tensor
    step_count: int

    @property
    def num_links(self) -> int:
        """Return the number of adjacent image pairs across all paths."""
        return self.layout.group_idx.numel() - self.layout.num_groups


@runtime_checkable
class SpringConfig(Protocol):
    """Resolve spring constants for the links in one or more NEB paths."""

    refresh: Literal[
        DynamicsStage.ON_ADMISSION,
        DynamicsStage.AFTER_COMPUTE,
    ]

    def resolve(self, context: SpringContext) -> Tensor:
        """Return one spring constant per adjacent image link.

        Parameters
        ----------
        context : SpringContext
            Read-only current NEB state.

        Returns
        -------
        Tensor
            Spring constants with shape ``(context.num_links,)`` on the same
            device as ``context.positions``.
        """


@dataclass(frozen=True, slots=True)
class ConstantSpringConfig:
    """Use one constant spring value for every adjacent image pair.

    Parameters
    ----------
    value : float
        Positive spring constant.
    """

    value: float
    refresh: Literal[DynamicsStage.ON_ADMISSION] = DynamicsStage.ON_ADMISSION

    def __post_init__(self) -> None:
        """Normalize and validate the constant spring value."""
        if isinstance(self.value, bool) or not isinstance(self.value, (int, float)):
            raise TypeError(
                "ConstantSpringConfig.value must be a number; "
                f"got {type(self.value).__name__}"
            )
        value = float(self.value)
        if not value > 0:
            raise ValueError("ConstantSpringConfig.value must be positive")
        object.__setattr__(self, "value", value)

    def resolve(self, context: SpringContext) -> Tensor:
        """Return constant spring values for the current path layout."""
        return torch.full(
            (context.num_links,),
            self.value,
            dtype=context.positions.dtype,
            device=context.positions.device,
        )


@dataclass(frozen=True, slots=True)
class NEBMethod:
    """Configure the device equations used to construct NEB forces.

    Equation functions must be module-level functions decorated with
    :func:`warp.func`. They are injected into a specialized Warp kernel when
    the method is bound during NEB setup; they are not passed through the Torch
    custom-operator boundary at runtime.

    Parameters
    ----------
    tangent_weights_fn : wp.Function, optional
        Return the forward and backward weights used to construct the path
        tangent from adjacent image displacements.
    effective_force_fn : wp.Function, optional
        Construct the effective force for an active, non-climbing interior
        image from the full Gram-statistics kernel contract.
    climbing_force_fn : wp.Function, optional
        Construct the effective force for a climbing image.
    name : str or None, optional
        Stable registry name. Named methods can participate in serializable
        configurations when the same registration is available on restore;
        unnamed methods are runtime-only.

    Examples
    --------
    Override only the tangent weighting while retaining regular NEB and
    climbing-image force laws::

        import warp as wp

        from nvalchemi.dynamics.paths.neb.methods import NEBMethod

        @wp.func
        def central_tangent_weights(
            energy_prev: float,
            energy_curr: float,
            energy_next: float,
        ):
            return 1.0, 1.0

        method = NEBMethod(tangent_weights_fn=central_tangent_weights)
    """

    tangent_weights_fn: wp.Function = improved_tangent_weights
    effective_force_fn: wp.Function = neb_effective_force_from_gram_stats
    climbing_force_fn: wp.Function = climbing_image_effective_force
    name: str | None = None

    def __post_init__(self) -> None:
        """Validate the setup-time method configuration."""
        for field_name in (
            "tangent_weights_fn",
            "effective_force_fn",
            "climbing_force_fn",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, wp.Function):
                raise TypeError(
                    f"{field_name} must be a function decorated with warp.func; "
                    f"got {type(value).__name__}"
                )

        if self.name is not None:
            if not isinstance(self.name, str):
                raise TypeError(
                    f"name must be a string or None; got {type(self.name).__name__}"
                )
            if not self.name.strip():
                raise ValueError("name must not be empty or contain only whitespace")
