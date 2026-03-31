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
from nvalchemiops.torch.interactions.electrostatics.ewald import (
    ewald_real_space,
    ewald_reciprocal_space,
)
from nvalchemiops.torch.interactions.electrostatics.k_vectors import (
    generate_k_vectors_ewald_summation,
)
from nvalchemiops.torch.interactions.electrostatics.parameters import (
    estimate_ewald_parameters,
)
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

__all__ = ["EwaldCoulombConfig", "EwaldCoulombPotential"]


class EwaldCoulombConfig(BaseModel):
    """Configuration for :class:`EwaldCoulombPotential`.

    Attributes
    ----------
    cutoff : float
        Real-space interaction cutoff radius (assumed Angstrom).
    accuracy : float, default 1e-6
        Relative accuracy target used to auto-tune *alpha* and
        *k_cutoff* when they are not set explicitly.
    alpha : float or None, default None
        Ewald splitting parameter (assumed Angstrom^-1).  Auto-tuned from
        *accuracy* when ``None``.
    k_cutoff : float or None, default None
        Reciprocal-space cutoff (assumed Angstrom^-1).  Auto-tuned from
        *accuracy* when ``None``.
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
    k_cutoff: Annotated[
        float | None,
        Field(description="Reciprocal-space cutoff (assumed Angstrom^-1)."),
    ] = None
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


EwaldCoulombPotentialCard = PotentialCard(
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
        }
    ),
    result_keys=frozenset({"energies", "forces", "stresses", "k_vectors"}),
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


class EwaldCoulombPotential(Potential):
    """Ewald electrostatics potential using matrix neighbors.

    Implements the Ewald summation for long-range Coulomb interactions
    as a composable pipeline step.  Supports both analytic (direct) and
    ``torch.autograd``-based derivative modes.

    Attributes
    ----------
    card : PotentialCard
        Class-level contract card declaring required inputs and result keys.
    config : EwaldCoulombConfig
        Resolved configuration for this instance.
    model_card : ModelCard
        Provenance metadata for this Ewald potential instance.
    """

    card = EwaldCoulombPotentialCard

    def __init__(
        self,
        config: EwaldCoulombConfig | None = None,
        *,
        cutoff: float = _UNSET,
        accuracy: float = _UNSET,
        alpha: float | None = _UNSET,
        k_cutoff: float | None = _UNSET,
        coulomb_constant: float = _UNSET,
        neighbor_list_name: str = _UNSET,
        charges_key: str = _UNSET,
        format: Literal["matrix"] = _UNSET,
        reuse_if_available: bool = _UNSET,
        derivative_mode: Literal["direct", "autograd"] = _UNSET,
        name: str | None = None,
    ) -> None:
        """Initialise an Ewald-summation electrostatics potential.

        Accepts either an :class:`EwaldCoulombConfig` object, individual
        keyword arguments matching the config fields, or both (keyword
        arguments override corresponding config fields).

        Parameters
        ----------
        config
            Pre-built configuration object.  Default ``None``.
        cutoff
            Real-space cutoff radius (assumed Angstrom).
        accuracy
            Relative accuracy target for auto-tuning.
        alpha
            Ewald splitting parameter (assumed Angstrom^-1).
        k_cutoff
            Reciprocal-space cutoff (assumed Angstrom^-1).
        coulomb_constant
            Coulomb constant (assumed eV * Angstrom / e^2).
        neighbor_list_name
            Logical neighbor-list name for result-key namespacing.
        charges_key
            Batch attribute name for per-atom partial charges.
        format
            Neighbor-list storage format.
        reuse_if_available
            Flag to reuse cached reciprocal-space tensors when
            available.
        derivative_mode
            ``"direct"`` for analytic derivatives; ``"autograd"`` for
            ``torch.autograd``.
        name
            Human-readable step name.  Default ``None``.
        """

        config = _resolve_config(
            EwaldCoulombConfig,
            config,
            {
                "cutoff": cutoff,
                "accuracy": accuracy,
                "alpha": alpha,
                "k_cutoff": k_cutoff,
                "coulomb_constant": coulomb_constant,
                "neighbor_list_name": neighbor_list_name,
                "charges_key": charges_key,
                "format": format,
                "reuse_if_available": reuse_if_available,
                "derivative_mode": derivative_mode,
            },
        )
        result_keys = {"energies", "k_vectors"}
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
            model_family="ewald",
            model_name=self.step_name,
            provided_terms=(PhysicalTerm(kind=ELECTROSTATICS, variant=ATOMIC_CHARGES),),
        )

    def _resolve_k_vectors(
        self,
        batch: Batch,
        ctx: ForwardContext,
        cell: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        """Return reusable or newly generated reciprocal-space vectors.

        Parameters
        ----------
        batch
            Input :class:`~nvalchemi.data.Batch`.
        ctx
            Forward context carrying resolved outputs and runtime state.
        cell
            Cell tensor of shape ``(num_graphs, 3, 3)``.
        positions
            Atom positions tensor.

        Returns
        -------
        torch.Tensor
            Reciprocal-space k-vectors for the Ewald summation.
        """

        needs_stress_path = bool(
            (
                ctx.runtime_state is not None
                and "cell_scaling" in ctx.runtime_state.requested_derivative_targets
            )
            or "stresses" in ctx.outputs
        )

        if self.config.reuse_if_available and not needs_stress_path:
            cached = self.optional_input(batch, "k_vectors", ctx)
            if cached is not None:
                return cached
        elif needs_stress_path:
            cached = self.optional_input(batch, "k_vectors", ctx)
            if cached is not None:
                warnings.warn(
                    "EwaldCoulombPotential ignores cached k_vectors when stress "
                    "or cell-derivative paths are active because k-vectors depend on cell.",
                    UserWarning,
                    stacklevel=2,
                )

        if self.config.k_cutoff is not None:
            reciprocal_cutoff = self.config.k_cutoff
        else:
            params = estimate_ewald_parameters(
                positions,
                cell,
                batch_idx=batch.batch.to(torch.int32),
                accuracy=self.config.accuracy,
            )
            reciprocal_cutoff = params.reciprocal_space_cutoff
        return generate_k_vectors_ewald_summation(cell, reciprocal_cutoff)

    @staticmethod
    def _unpack_kernel(
        result: torch.Tensor | tuple[torch.Tensor, ...],
        *,
        has_forces: bool,
        has_virial: bool,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """Unpack one electrostatics kernel result.

        Parameters
        ----------
        result
            Raw output from an Ewald kernel — either a single energy
            tensor or a tuple of ``(energy, forces, virial)``.
        has_forces
            Flag indicating whether the kernel was called with force
            computation enabled.
        has_virial
            Flag indicating whether the kernel was called with virial
            computation enabled.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]
            ``(energy, forces, virial)`` where *forces* and *virial*
            are ``None`` when not computed.
        """

        if isinstance(result, torch.Tensor):
            return result, None, None
        items = list(result)
        energy = items[0]
        idx = 1
        forces = items[idx] if has_forces and idx < len(items) else None
        if has_forces and idx < len(items):
            idx += 1
        virial = items[idx] if has_virial and idx < len(items) else None
        return energy, forces, virial

    def compute(
        self,
        batch: Batch,
        ctx: ForwardContext,
    ) -> CalculatorResults:
        """Run the Ewald electrostatics kernels for the current batch.

        Parameters
        ----------
        batch
            Input :class:`~nvalchemi.data.Batch` with positions, charges,
            cell, pbc, and matrix-format neighbor data.
        ctx
            Forward context carrying resolved outputs and runtime state.

        Returns
        -------
        CalculatorResults
            Mapping with ``"energies"`` and, when requested, ``"forces"``,
            ``"stresses"``, and ``"k_vectors"``.

        Raises
        ------
        ValueError
            If charges require gradients in ``"direct"`` derivative mode,
            or if periodic inputs are missing or incomplete.
        """

        positions = self.require_input(batch, "positions", ctx)
        charges = self.require_input(batch, self.config.charges_key, ctx)
        if charges.ndim > 1:
            charges = charges.squeeze(-1)
        if self.config.derivative_mode == "direct" and charges.requires_grad:
            raise ValueError(
                "EwaldCoulombPotential(direct) only supports fixed charges. "
                "Use derivative_mode='autograd' for charge-coupled pipelines."
            )

        cell, pbc = self.resolve_periodic_inputs(batch, ctx)
        if cell is None or pbc is None or not bool(torch.as_tensor(pbc).all().item()):
            raise ValueError(
                "Ewald electrostatics requires periodic inputs 'cell' and 'pbc'."
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

        alpha = self.config.alpha
        if alpha is None:
            alpha = estimate_ewald_parameters(
                positions,
                cell_tensor,
                batch_idx=batch_idx,
                accuracy=self.config.accuracy,
            ).alpha
        elif not isinstance(alpha, torch.Tensor):
            alpha = torch.full(
                (cell_tensor.shape[0],),
                float(alpha),
                dtype=cell_tensor.dtype,
                device=cell_tensor.device,
            )

        k_vectors = self._resolve_k_vectors(batch, ctx, cell_tensor, positions)
        compute_forces = self.config.derivative_mode == "direct" and bool(
            ctx.outputs & {"forces", "stresses"}
        )
        compute_stresses = (
            self.config.derivative_mode == "direct" and "stresses" in ctx.outputs
        )
        kernel_positions = (
            positions.detach() if self.config.derivative_mode == "direct" else positions
        )
        kernel_cell = (
            cell_tensor.detach()
            if self.config.derivative_mode == "direct"
            else cell_tensor
        )

        real_result = ewald_real_space(
            positions=kernel_positions,
            charges=charges.view(-1),
            cell=kernel_cell,
            alpha=alpha,
            neighbor_matrix=neighbor_matrix,
            neighbor_matrix_shifts=neighbor_shifts,
            mask_value=-1,
            batch_idx=batch_idx,
            compute_forces=compute_forces,
            compute_virial=compute_stresses,
        )
        recip_result = ewald_reciprocal_space(
            positions=kernel_positions,
            charges=charges.view(-1),
            cell=kernel_cell,
            k_vectors=k_vectors,
            alpha=alpha,
            batch_idx=batch_idx,
            compute_forces=compute_forces,
            compute_virial=compute_stresses,
        )

        e_real, f_real, v_real = self._unpack_kernel(
            real_result,
            has_forces=compute_forces,
            has_virial=compute_stresses,
        )
        e_recip, f_recip, v_recip = self._unpack_kernel(
            recip_result,
            has_forces=compute_forces,
            has_virial=compute_stresses,
        )

        per_atom_energies = (e_real + e_recip).to(positions.dtype)
        energies = aggregate_per_system_energy(
            per_atom_energies, batch.batch, batch.num_graphs
        )
        energies = energies * self.config.coulomb_constant

        forces = None
        if compute_forces and f_real is not None and f_recip is not None:
            forces = (f_real + f_recip) * self.config.coulomb_constant

        stresses = None
        if compute_stresses and v_real is not None and v_recip is not None:
            virial = (v_real + v_recip) * self.config.coulomb_constant
            stresses = virial_to_stress(virial, cell_tensor)

        return self.build_results(
            ctx,
            energies=energies,
            forces=forces if "forces" in ctx.outputs else None,
            stresses=stresses,
            k_vectors=k_vectors if "k_vectors" in ctx.outputs else None,
        )
