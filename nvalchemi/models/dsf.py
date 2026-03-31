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

from collections.abc import Iterable
from typing import Annotated, Any, Literal

import torch
from pydantic import BaseModel, ConfigDict, Field

from nvalchemi.data import Batch
from nvalchemi.models.utils import virial_to_stress
from nvalchemi.models.base import ForwardContext, Potential, _UNSET, _resolve_config
from nvalchemi.models.contracts import NeighborRequirement, PotentialCard
from nvalchemi.models.metadata import (
    ATOMIC_CHARGES,
    ELECTROSTATICS,
    ModelCard,
    PhysicalTerm,
)
from nvalchemi.models.neighbors import neighbor_result_key
from nvalchemi.models.results import CalculatorResults
from nvalchemiops.torch.interactions.electrostatics.dsf import dsf_coulomb

__all__ = ["DSFCoulombConfig", "DSFCoulombPotential"]


class DSFCoulombConfig(BaseModel):
    """Configuration for :class:`DSFCoulombPotential`.

    Attributes
    ----------
    cutoff : float
        Interaction cutoff radius (assumed Angstrom).
    alpha : float, default 0.2
        Damping parameter for the DSF kernel (assumed Angstrom^-1).
    neighbor_list_name : str, default "default"
        Logical neighbor-list name used to namespace result keys.
    format : {"coo", "matrix"}, default "coo"
        Neighbor-list storage format.
    charges_key : str, default "node_charges"
        Batch attribute name for per-atom partial charges.
    """

    cutoff: Annotated[float, Field(description="Interaction cutoff radius (assumed Angstrom).")]
    alpha: Annotated[
        float, Field(description="DSF damping parameter (assumed Angstrom^-1).")
    ] = 0.2
    neighbor_list_name: Annotated[
        str, Field(description="Logical neighbor-list name for result-key namespacing.")
    ] = "default"
    format: Annotated[
        Literal["coo", "matrix"], Field(description="Neighbor-list storage format.")
    ] = "coo"
    charges_key: Annotated[
        str, Field(description="Batch attribute name for per-atom partial charges.")
    ] = "node_charges"

    model_config = ConfigDict(extra="forbid")


DSFCoulombPotentialCard = PotentialCard(
    required_inputs=frozenset(
        {
            "positions",
            "node_charges",
            neighbor_result_key("default", "neighbor_list"),
            neighbor_result_key("default", "neighbor_ptr"),
            neighbor_result_key("default", "unit_shifts"),
        }
    ),
    optional_inputs=frozenset({"cell", "pbc"}),
    result_keys=frozenset({"energies", "forces", "stresses"}),
    default_result_keys=frozenset({"energies"}),
    additive_result_keys=frozenset({"energies", "forces", "stresses"}),
    boundary_modes=frozenset({"non_pbc", "pbc"}),
    neighbor_requirement=NeighborRequirement(
        source="external",
        format="coo",
        name="default",
    ),
    parameterized_by=frozenset({"neighbor_list_name", "charges_key", "format"}),
)


class DSFCoulombPotential(Potential):
    """Concrete DSF Coulomb potential with direct forces and virial-based stress."""

    card = DSFCoulombPotentialCard

    def __init__(
        self,
        config: DSFCoulombConfig | None = None,
        *,
        cutoff: float = _UNSET,
        alpha: float = _UNSET,
        neighbor_list_name: str = _UNSET,
        format: Literal["coo", "matrix"] = _UNSET,
        charges_key: str = _UNSET,
        name: str | None = None,
    ) -> None:
        """Initialise a Damped Shifted Force Coulomb potential.

        Accepts either a :class:`DSFCoulombConfig` object, individual
        keyword arguments matching the config fields, or both (keyword
        arguments override corresponding config fields).

        Parameters
        ----------
        config
            Pre-built configuration object.
        cutoff
            Interaction cutoff radius (assumed Angstrom).
        alpha
            DSF damping parameter (assumed Angstrom^-1).
        neighbor_list_name
            Logical neighbor-list name for result-key namespacing.
        format
            Neighbor-list storage format.
        charges_key
            Batch attribute name for per-atom partial charges.
        name
            Human-readable step name.
        """

        config = _resolve_config(
            DSFCoulombConfig,
            config,
            {
                "cutoff": cutoff,
                "alpha": alpha,
                "neighbor_list_name": neighbor_list_name,
                "format": format,
                "charges_key": charges_key,
            },
        )
        super().__init__(
            name=name,
            required_inputs=self._base_required_inputs(config),
            neighbor_requirement=NeighborRequirement(
                source="external",
                cutoff=config.cutoff,
                format=config.format,
                name=config.neighbor_list_name,
            ),
        )
        self.config = config
        self.model_card = ModelCard(
            model_family="dsf",
            model_name=self.step_name,
            provided_terms=(PhysicalTerm(kind=ELECTROSTATICS, variant=ATOMIC_CHARGES),),
        )

    @staticmethod
    def _base_required_inputs(config: DSFCoulombConfig) -> frozenset[str]:
        """Return required named inputs for the configured neighbor format."""

        keys = {"positions", config.charges_key}
        if config.format == "coo":
            keys |= {
                neighbor_result_key(config.neighbor_list_name, "neighbor_list"),
                neighbor_result_key(config.neighbor_list_name, "neighbor_ptr"),
                neighbor_result_key(config.neighbor_list_name, "unit_shifts"),
            }
        else:
            keys |= {
                neighbor_result_key(config.neighbor_list_name, "neighbor_matrix"),
                neighbor_result_key(config.neighbor_list_name, "neighbor_shifts"),
            }
        return frozenset(keys)

    def required_inputs(
        self,
        outputs: Iterable[str] | None = None,
    ) -> frozenset[str]:
        """Return required inputs for one requested output set."""

        active = self.active_outputs(outputs)
        required = set(self.profile.required_inputs)
        if "stresses" in active:
            required |= {"cell", "pbc"}
        return frozenset(required)

    def optional_inputs(
        self,
        outputs: Iterable[str] | None = None,
    ) -> frozenset[str]:
        """Return optional inputs for one requested output set."""

        active = self.active_outputs(outputs)
        optional = set(self.profile.optional_inputs)
        if "stresses" in active:
            optional -= {"cell", "pbc"}
        return frozenset(optional)

    def _resolve_periodic_state(
        self,
        batch: Batch,
        ctx: ForwardContext,
    ) -> tuple[torch.Tensor | None, bool]:
        """Resolve periodic inputs and detect whether PBC is active."""

        cell, pbc = self.resolve_periodic_inputs(batch, ctx)
        if cell is None or pbc is None:
            return None, False
        periodic = bool(torch.as_tensor(pbc).any().item())
        return cell, periodic

    def compute(
        self,
        batch: Batch,
        ctx: ForwardContext,
    ) -> CalculatorResults:
        """Run DSF electrostatics for the configured neighbor format."""

        positions = self.require_input(batch, "positions", ctx)
        charges = self.require_input(batch, self.config.charges_key, ctx)
        if charges.ndim > 1:
            charges = charges.squeeze(-1)

        cell, periodic = self._resolve_periodic_state(batch, ctx)
        if "stresses" in ctx.outputs and not periodic:
            raise ValueError("DSF stresses require periodic inputs 'cell' and 'pbc'.")

        compute_forces = bool(ctx.outputs & {"forces", "stresses"})
        compute_virial = "stresses" in ctx.outputs
        kwargs: dict[str, Any] = {
            "positions": positions,
            "charges": charges,
            "cutoff": self.config.cutoff,
            "alpha": self.config.alpha,
            "batch_idx": batch.batch.to(torch.int32),
            "num_systems": batch.num_graphs,
            "compute_forces": compute_forces,
            "compute_virial": compute_virial,
        }
        if periodic and cell is not None:
            kwargs["cell"] = cell

        if self.config.format == "coo":
            kwargs["neighbor_list"] = self.require_input(
                batch,
                neighbor_result_key(self.config.neighbor_list_name, "neighbor_list"),
                ctx,
            )
            kwargs["neighbor_ptr"] = self.require_input(
                batch,
                neighbor_result_key(self.config.neighbor_list_name, "neighbor_ptr"),
                ctx,
            )
            kwargs["unit_shifts"] = self.require_input(
                batch,
                neighbor_result_key(self.config.neighbor_list_name, "unit_shifts"),
                ctx,
            )
        else:
            kwargs["neighbor_matrix"] = self.require_input(
                batch,
                neighbor_result_key(self.config.neighbor_list_name, "neighbor_matrix"),
                ctx,
            )
            kwargs["neighbor_matrix_shifts"] = self.require_input(
                batch,
                neighbor_result_key(self.config.neighbor_list_name, "neighbor_shifts"),
                ctx,
            )

        raw = dsf_coulomb(**kwargs)
        energies = raw[0]
        if energies.ndim == 1:
            energies = energies.unsqueeze(-1)

        forces = None
        stresses = None
        if compute_forces and "forces" in ctx.outputs:
            forces = raw[1]
        if compute_virial and "stresses" in ctx.outputs:
            virial = raw[2]
            cell_for_stress = cell if cell is not None and cell.ndim == 3 else cell.unsqueeze(0)
            stresses = virial_to_stress(-virial, cell_for_stress)

        return self.build_results(ctx, energies=energies, forces=forces, stresses=stresses)
