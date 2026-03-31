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
from __future__ import annotations

import warnings
from typing import Annotated, Literal

import torch
from nvalchemiops.torch.interactions.electrostatics.k_vectors import (
    generate_k_vectors_pme,
)
from nvalchemiops.torch.interactions.electrostatics.parameters import (
    estimate_pme_parameters,
)
from nvalchemiops.torch.interactions.electrostatics.pme import particle_mesh_ewald
from pydantic import BaseModel, ConfigDict, Field

from nvalchemi.data import Batch
from nvalchemi.models.base import _UNSET, ForwardContext, Potential, _resolve_config
from nvalchemi.models.contracts import NeighborRequirement, PotentialCard
from nvalchemi.models.metadata import (
    ATOMIC_CHARGES,
    ELECTROSTATICS,
    ModelCard,
    PhysicalTerm,
)
from nvalchemi.models.neighbors import neighbor_result_key
from nvalchemi.models.results import CalculatorResults
from nvalchemi.models.utils import aggregate_per_system_energy, virial_to_stress

__all__ = ["PMEConfig", "PMEPotential"]


class PMEConfig(BaseModel):
    """Configuration for :class:`PMEPotential`.

    Attributes
    ----------
    cutoff : float
        Real-space interaction cutoff radius (assumed Angstrom).
    accuracy : float, default 1e-6
        Relative accuracy target used to auto-tune *alpha* and mesh
        parameters when they are not set explicitly.
    alpha : float or None, default None
        Ewald splitting parameter (assumed Angstrom^-1).  Auto-tuned from
        *accuracy* when ``None``.
    mesh_spacing : float or None, default None
        Target mesh spacing for the reciprocal grid (assumed Angstrom).
    mesh_dimensions : tuple[int, int, int] or None, default None
        Explicit reciprocal-space mesh dimensions.  Overrides
        *mesh_spacing* when set.
    spline_order : int, default 4
        B-spline interpolation order for the charge assignment.
    coulomb_constant : float, default 14.3996
        Coulomb constant in output energy units (assumed eV * Angstrom / e^2).
    neighbor_list_name : str, default "default"
        Logical neighbor-list name used to namespace result keys.
    charges_key : str, default "node_charges"
        Batch attribute name for per-atom partial charges.
    format : {"matrix"}, default "matrix"
        Neighbor-list storage format.
    reuse_if_available : bool, default True
        Reuse cached reciprocal-space tensors when available.
    derivative_mode : {"direct", "autograd"}, default "direct"
        ``"direct"`` uses analytically derived forces and stresses;
        ``"autograd"`` defers differentiation to an
        :class:`EnergyDerivativesStep`.
    """

    cutoff: Annotated[
        float, Field(description="Real-space cutoff radius (assumed Angstrom).")
    ]
    accuracy: Annotated[
        float, Field(description="Relative accuracy target for auto-tuning.")
    ] = 1e-6
    alpha: Annotated[
        float | None,
        Field(description="Ewald splitting parameter (assumed Angstrom^-1)."),
    ] = None
    mesh_spacing: Annotated[
        float | None,
        Field(description="Target reciprocal mesh spacing (assumed Angstrom)."),
    ] = None
    mesh_dimensions: Annotated[
        tuple[int, int, int] | None,
        Field(description="Explicit reciprocal-space mesh dimensions."),
    ] = None
    spline_order: Annotated[
        int, Field(description="B-spline interpolation order for charge assignment.")
    ] = 4
    coulomb_constant: Annotated[
        float, Field(description="Coulomb constant (assumed eV * Angstrom / e^2).")
    ] = 14.3996
    neighbor_list_name: Annotated[
        str, Field(description="Logical neighbor-list name for result-key namespacing.")
    ] = "default"
    charges_key: Annotated[
        str, Field(description="Batch attribute name for per-atom partial charges.")
    ] = "node_charges"
    format: Annotated[
        Literal["matrix"], Field(description="Neighbor-list storage format.")
    ] = "matrix"
    reuse_if_available: Annotated[
        bool, Field(description="Reuse cached reciprocal-space tensors when available.")
    ] = True
    derivative_mode: Annotated[
        Literal["direct", "autograd"],
        Field(
            description="'direct' for analytic derivatives; 'autograd' for torch.autograd."
        ),
    ] = "direct"

    model_config = ConfigDict(extra="forbid")


PMEPotentialCard = PotentialCard(
    required_inputs=frozenset(
        {
            "positions",
            "node_charges",
            "cell",
            "pbc",
            neighbor_result_key("default", "neighbor_matrix"),
            neighbor_result_key("default", "num_neighbors"),
        }
    ),
    optional_inputs=frozenset(
        {
            neighbor_result_key("default", "neighbor_shifts"),
            "k_vectors",
            "k_squared",
        }
    ),
    result_keys=frozenset({"energies", "forces", "stresses", "k_vectors", "k_squared"}),
    default_result_keys=frozenset({"energies"}),
    additive_result_keys=frozenset({"energies", "forces", "stresses"}),
    boundary_modes=frozenset({"pbc"}),
    neighbor_requirement=NeighborRequirement(
        source="external",
        format="matrix",
        name="default",
    ),
    parameterized_by=frozenset({"neighbor_list_name", "charges_key"}),
)


class PMEPotential(Potential):
    """Particle Mesh Ewald electrostatics using matrix neighbors."""

    card = PMEPotentialCard

    def __init__(
        self,
        config: PMEConfig | None = None,
        *,
        cutoff: float = _UNSET,
        accuracy: float = _UNSET,
        alpha: float | None = _UNSET,
        mesh_spacing: float | None = _UNSET,
        mesh_dimensions: tuple[int, int, int] | None = _UNSET,
        spline_order: int = _UNSET,
        coulomb_constant: float = _UNSET,
        neighbor_list_name: str = _UNSET,
        charges_key: str = _UNSET,
        format: Literal["matrix"] = _UNSET,
        reuse_if_available: bool = _UNSET,
        derivative_mode: Literal["direct", "autograd"] = _UNSET,
        name: str | None = None,
    ) -> None:
        """Initialise a Particle Mesh Ewald electrostatics potential.

        Accepts either a :class:`PMEConfig` object, individual keyword
        arguments matching the config fields, or both (keyword arguments
        override corresponding config fields).

        Parameters
        ----------
        config
            Pre-built configuration object.
        cutoff
            Real-space cutoff radius (assumed Angstrom).
        accuracy
            Relative accuracy target for auto-tuning.
        alpha
            Ewald splitting parameter (assumed Angstrom^-1).
        mesh_spacing
            Target reciprocal mesh spacing (assumed Angstrom).
        mesh_dimensions
            Explicit reciprocal-space mesh dimensions.
        spline_order
            B-spline interpolation order.
        coulomb_constant
            Coulomb constant (assumed eV * Angstrom / e^2).
        neighbor_list_name
            Logical neighbor-list name for result-key namespacing.
        charges_key
            Batch attribute name for per-atom partial charges.
        format
            Neighbor-list storage format.
        reuse_if_available
            Reuse cached reciprocal-space tensors when available.
        derivative_mode
            ``"direct"`` for analytic derivatives; ``"autograd"`` for
            ``torch.autograd``.
        name
            Human-readable step name.
        """

        config = _resolve_config(
            PMEConfig,
            config,
            {
                "cutoff": cutoff,
                "accuracy": accuracy,
                "alpha": alpha,
                "mesh_spacing": mesh_spacing,
                "mesh_dimensions": mesh_dimensions,
                "spline_order": spline_order,
                "coulomb_constant": coulomb_constant,
                "neighbor_list_name": neighbor_list_name,
                "charges_key": charges_key,
                "format": format,
                "reuse_if_available": reuse_if_available,
                "derivative_mode": derivative_mode,
            },
        )
        result_keys = {"energies", "k_vectors", "k_squared"}
        additive_result_keys = {"energies"}
        gradient_setup_targets = frozenset()
        if config.derivative_mode == "direct":
            result_keys |= {"forces", "stresses"}
            additive_result_keys |= {"forces", "stresses"}
        else:
            gradient_setup_targets = frozenset({"positions", "cell_scaling"})
        super().__init__(
            name=name,
            required_inputs=frozenset(
                {
                    "positions",
                    config.charges_key,
                    "cell",
                    "pbc",
                    neighbor_result_key(config.neighbor_list_name, "neighbor_matrix"),
                    neighbor_result_key(config.neighbor_list_name, "num_neighbors"),
                }
            ),
            optional_inputs=frozenset(
                {
                    neighbor_result_key(config.neighbor_list_name, "neighbor_shifts"),
                    "k_vectors",
                    "k_squared",
                }
            ),
            neighbor_requirement=NeighborRequirement(
                source="external",
                cutoff=config.cutoff,
                format="matrix",
                name=config.neighbor_list_name,
            ),
            boundary_modes=frozenset({"pbc"}),
            result_keys=frozenset(result_keys),
            additive_result_keys=frozenset(additive_result_keys),
            gradient_setup_targets=gradient_setup_targets,
        )
        self.config = config
        self.model_card = ModelCard(
            model_family="pme",
            model_name=self.step_name,
            provided_terms=(PhysicalTerm(kind=ELECTROSTATICS, variant=ATOMIC_CHARGES),),
        )

    def _resolve_reciprocal_state(
        self,
        batch: Batch,
        ctx: ForwardContext,
        cell: torch.Tensor,
        positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return reusable or newly generated PME reciprocal-space tensors."""

        needs_stress_path = bool(
            (
                ctx.runtime_state is not None
                and "cell_scaling" in ctx.runtime_state.requested_derivative_targets
            )
            or "stresses" in ctx.outputs
        )

        if self.config.reuse_if_available and not needs_stress_path:
            cached_vectors = self.optional_input(batch, "k_vectors", ctx)
            cached_squared = self.optional_input(batch, "k_squared", ctx)
            if cached_vectors is not None and cached_squared is not None:
                return cached_vectors, cached_squared
        elif needs_stress_path:
            cached_vectors = self.optional_input(batch, "k_vectors", ctx)
            cached_squared = self.optional_input(batch, "k_squared", ctx)
            if cached_vectors is not None or cached_squared is not None:
                warnings.warn(
                    "PMEPotential ignores cached reciprocal tensors when stress "
                    "or cell-derivative paths are active because k-vectors depend on cell.",
                    UserWarning,
                    stacklevel=2,
                )

        params = None
        if self.config.alpha is None or self.config.mesh_dimensions is None:
            params = estimate_pme_parameters(
                positions,
                cell,
                batch_idx=batch.batch.to(torch.int32),
                accuracy=self.config.accuracy,
            )

        if self.config.mesh_dimensions is not None:
            dims = self.config.mesh_dimensions
        else:
            dims = params.mesh_dimensions
        return generate_k_vectors_pme(cell, dims)

    def compute(
        self,
        batch: Batch,
        ctx: ForwardContext,
    ) -> CalculatorResults:
        """Run the PME electrostatics kernel for the current batch."""

        positions = self.require_input(batch, "positions", ctx)
        charges = self.require_input(batch, self.config.charges_key, ctx)
        if charges.ndim > 1:
            charges = charges.squeeze(-1)
        if self.config.derivative_mode == "direct" and charges.requires_grad:
            raise ValueError(
                "PMEPotential(direct) only supports fixed charges. "
                "Use derivative_mode='autograd' for charge-coupled pipelines."
            )

        cell, pbc = self.resolve_periodic_inputs(batch, ctx)
        if cell is None or pbc is None or not bool(torch.as_tensor(pbc).all().item()):
            raise ValueError(
                "PME electrostatics requires periodic inputs 'cell' and 'pbc'."
            )
        cell_tensor = cell if cell.ndim == 3 else cell.unsqueeze(0)

        batch_idx = batch.batch.to(torch.int32)
        neighbor_matrix = self.require_input(
            batch,
            neighbor_result_key(self.config.neighbor_list_name, "neighbor_matrix"),
            ctx,
        ).contiguous()
        neighbor_shifts = self.optional_input(
            batch,
            neighbor_result_key(self.config.neighbor_list_name, "neighbor_shifts"),
            ctx,
        )
        if neighbor_shifts is None:
            neighbor_shifts = torch.zeros(
                (*neighbor_matrix.shape, 3),
                dtype=torch.int32,
                device=neighbor_matrix.device,
            )
        else:
            neighbor_shifts = neighbor_shifts.contiguous()

        k_vectors, k_squared = self._resolve_reciprocal_state(
            batch,
            ctx,
            cell_tensor,
            positions,
        )

        params = None
        if self.config.alpha is None or self.config.mesh_dimensions is None:
            params = estimate_pme_parameters(
                positions,
                cell_tensor,
                batch_idx=batch_idx,
                accuracy=self.config.accuracy,
            )

        if self.config.alpha is not None:
            alpha_val = float(self.config.alpha)
        else:
            alpha_val = float(params.alpha.mean().item())

        alpha = torch.full(
            (cell_tensor.shape[0],),
            alpha_val,
            dtype=cell_tensor.dtype,
            device=cell_tensor.device,
        )

        if self.config.alpha is None and batch.num_graphs > 1:
            volumes = torch.linalg.det(cell_tensor).abs()
            if volumes.min() > 0 and (volumes.max() / volumes.min()) > 1.1:
                warnings.warn(
                    "PMEPotential is using a single mean alpha across a batch with "
                    "heterogeneous cell volumes.",
                    UserWarning,
                    stacklevel=2,
                )

        compute_forces = self.config.derivative_mode == "direct" and bool(
            ctx.outputs & {"forces", "stresses"}
        )
        compute_stresses = (
            self.config.derivative_mode == "direct" and "stresses" in ctx.outputs
        )
        mesh_dimensions = (
            self.config.mesh_dimensions
            if self.config.mesh_dimensions is not None
            else params.mesh_dimensions
        )
        kernel_positions = (
            positions.detach() if self.config.derivative_mode == "direct" else positions
        )
        kernel_cell = (
            cell_tensor.detach()
            if self.config.derivative_mode == "direct"
            else cell_tensor
        )
        result = particle_mesh_ewald(
            positions=kernel_positions,
            charges=charges.view(-1),
            cell=kernel_cell,
            alpha=alpha,
            mesh_dimensions=mesh_dimensions,
            spline_order=self.config.spline_order,
            batch_idx=batch_idx,
            k_vectors=k_vectors,
            k_squared=k_squared,
            neighbor_matrix=neighbor_matrix,
            neighbor_matrix_shifts=neighbor_shifts,
            mask_value=-1,
            compute_forces=compute_forces,
            compute_virial=compute_stresses,
            accuracy=self.config.accuracy,
        )

        if isinstance(result, torch.Tensor):
            per_atom_energies = result
            forces = None
            virial = None
        else:
            items = list(result)
            per_atom_energies = items[0]
            idx = 1
            forces = items[idx] if compute_forces and idx < len(items) else None
            if compute_forces and idx < len(items):
                idx += 1
            virial = items[idx] if compute_stresses and idx < len(items) else None

        energies = aggregate_per_system_energy(
            per_atom_energies.to(positions.dtype),
            batch.batch,
            batch.num_graphs,
        )
        energies = energies * self.config.coulomb_constant

        if forces is not None:
            forces = forces * self.config.coulomb_constant
        stresses = None
        if virial is not None:
            stresses = virial_to_stress(
                virial * self.config.coulomb_constant,
                cell_tensor,
            )

        return self.build_results(
            ctx,
            energies=energies,
            forces=forces if "forces" in ctx.outputs else None,
            stresses=stresses,
            k_vectors=k_vectors if "k_vectors" in ctx.outputs else None,
            k_squared=k_squared if "k_squared" in ctx.outputs else None,
        )
