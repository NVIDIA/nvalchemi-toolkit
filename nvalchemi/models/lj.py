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
from typing import Annotated, Literal

import torch
from pydantic import BaseModel, ConfigDict, Field

from nvalchemi.data import Batch
from nvalchemi.models._ops.lj import (
    lj_energy_forces_batch_into,
    lj_energy_forces_virial_batch_into,
)
from nvalchemi.models.base import _UNSET, ForwardContext, Potential, _resolve_config
from nvalchemi.models.contracts import NeighborRequirement, PotentialCard
from nvalchemi.models.metadata import (
    REPULSION,
    SHORT_RANGE,
    ModelCard,
    PhysicalTerm,
)
from nvalchemi.models.neighbors import neighbor_result_key
from nvalchemi.models.results import CalculatorResults
from nvalchemi.models.utils import aggregate_per_system_energy, virial_to_stress

__all__ = ["LennardJonesConfig", "LennardJonesPotential"]


class LennardJonesConfig(BaseModel):
    """Configuration for :class:`LennardJonesPotential`.

    Attributes
    ----------
    epsilon : float
        Well depth of the LJ potential (assumed eV).
    sigma : float
        Finite distance at which the potential is zero (assumed Angstrom).
    cutoff : float
        Interaction cutoff radius (assumed Angstrom).
    switch_width : float, default 0.0
        Width of the switching region below *cutoff* (assumed Angstrom).
        A value of ``0.0`` means a hard cutoff.
    neighbor_list_name : str, default "default"
        Logical neighbor-list name used to namespace result keys.
    format : {"matrix"}, default "matrix"
        Neighbor-list storage format.
    half_list : bool, default False
        Whether the neighbor list is a half-list.
    """

    epsilon: Annotated[float, Field(description="LJ well depth (assumed eV).")]
    sigma: Annotated[
        float, Field(description="Zero-potential distance (assumed Angstrom).")
    ]
    cutoff: Annotated[
        float, Field(description="Interaction cutoff radius (assumed Angstrom).")
    ]
    switch_width: Annotated[
        float,
        Field(description="Switching region width below cutoff (assumed Angstrom)."),
    ] = 0.0
    neighbor_list_name: Annotated[
        str, Field(description="Logical neighbor-list name for result-key namespacing.")
    ] = "default"
    format: Annotated[
        Literal["matrix"], Field(description="Neighbor-list storage format.")
    ] = "matrix"
    half_list: Annotated[
        bool, Field(description="Whether the neighbor list is a half-list.")
    ] = False

    model_config = ConfigDict(extra="forbid")


LennardJonesPotentialCard = PotentialCard(
    required_inputs=frozenset(
        {
            "positions",
            neighbor_result_key("default", "neighbor_matrix"),
            neighbor_result_key("default", "num_neighbors"),
        }
    ),
    optional_inputs=frozenset(
        {"cell", "pbc", neighbor_result_key("default", "neighbor_shifts")}
    ),
    result_keys=frozenset({"energies", "forces", "stresses"}),
    default_result_keys=frozenset({"energies"}),
    additive_result_keys=frozenset({"energies", "forces", "stresses"}),
    boundary_modes=frozenset({"non_pbc", "pbc"}),
    neighbor_requirement=NeighborRequirement(
        source="external",
        format="matrix",
        name="default",
    ),
    parameterized_by=frozenset({"neighbor_list_name", "half_list"}),
)


class LennardJonesPotential(Potential):
    """Warp-accelerated Lennard-Jones potential using matrix neighbors."""

    card = LennardJonesPotentialCard

    def __init__(
        self,
        config: LennardJonesConfig | None = None,
        *,
        epsilon: float = _UNSET,
        sigma: float = _UNSET,
        cutoff: float = _UNSET,
        switch_width: float = _UNSET,
        neighbor_list_name: str = _UNSET,
        format: Literal["matrix"] = _UNSET,
        half_list: bool = _UNSET,
        name: str | None = None,
    ) -> None:
        """Initialise a Lennard-Jones potential.

        Accepts either a :class:`LennardJonesConfig` object, individual
        keyword arguments matching the config fields, or both (keyword
        arguments override corresponding config fields).

        Parameters
        ----------
        config
            Pre-built configuration object.  When ``None``, a config
            is constructed from keyword arguments.
        epsilon
            LJ well depth (assumed eV).
        sigma
            Zero-potential distance (assumed Angstrom).
        cutoff
            Interaction cutoff radius (assumed Angstrom).
        switch_width
            Switching region width below *cutoff* (assumed Angstrom).
        neighbor_list_name
            Logical neighbor-list name for result-key namespacing.
        format
            Neighbor-list storage format.
        half_list
            Whether the neighbor list is a half-list.
        name
            Human-readable step name.
        """

        config = _resolve_config(
            LennardJonesConfig,
            config,
            {
                "epsilon": epsilon,
                "sigma": sigma,
                "cutoff": cutoff,
                "switch_width": switch_width,
                "neighbor_list_name": neighbor_list_name,
                "format": format,
                "half_list": half_list,
            },
        )
        super().__init__(
            name=name,
            required_inputs=frozenset(
                {
                    "positions",
                    neighbor_result_key(config.neighbor_list_name, "neighbor_matrix"),
                    neighbor_result_key(config.neighbor_list_name, "num_neighbors"),
                }
            ),
            optional_inputs=frozenset(
                {
                    "cell",
                    "pbc",
                    neighbor_result_key(config.neighbor_list_name, "neighbor_shifts"),
                }
            ),
            neighbor_requirement=NeighborRequirement(
                source="external",
                cutoff=config.cutoff,
                format="matrix",
                half_list=config.half_list,
                name=config.neighbor_list_name,
            ),
        )
        self.config = config
        self.model_card = ModelCard(
            model_family="lennard_jones",
            model_name=self.step_name,
            provided_terms=(PhysicalTerm(kind=SHORT_RANGE, variant=REPULSION),),
        )

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

    def compute(
        self,
        batch: Batch,
        ctx: ForwardContext,
    ) -> CalculatorResults:
        """Run the Lennard-Jones kernel for the current batch."""

        positions = self.require_input(batch, "positions", ctx)
        neighbor_matrix = self.require_input(
            batch,
            neighbor_result_key(self.config.neighbor_list_name, "neighbor_matrix"),
            ctx,
        ).contiguous()
        num_neighbors = self.require_input(
            batch,
            neighbor_result_key(self.config.neighbor_list_name, "num_neighbors"),
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

        cell, pbc = self.resolve_periodic_inputs(batch, ctx)
        periodic = (
            cell is not None
            and pbc is not None
            and bool(torch.as_tensor(pbc).any().item())
        )
        if "stresses" in ctx.outputs and not periodic:
            raise ValueError(
                "Lennard-Jones stresses require periodic inputs 'cell' and 'pbc'."
            )

        num_graphs = batch.num_graphs
        if cell is None:
            cells = (
                torch.eye(3, dtype=positions.dtype, device=positions.device)
                .unsqueeze(0)
                .expand(num_graphs, 3, 3)
                .contiguous()
            )
        else:
            cells = cell if cell.ndim == 3 else cell.unsqueeze(0)
            cells = cells.contiguous()

        atomic_energies = torch.empty(
            positions.shape[0],
            dtype=positions.dtype,
            device=positions.device,
        )
        forces_buf = torch.empty_like(positions)
        stresses = None
        if "stresses" in ctx.outputs:
            virials_buf = torch.empty(
                (num_graphs, 9),
                dtype=positions.dtype,
                device=positions.device,
            )
            lj_energy_forces_virial_batch_into(
                positions=positions,
                cells=cells,
                neighbor_matrix=neighbor_matrix,
                neighbor_shifts=neighbor_shifts,
                num_neighbors=num_neighbors,
                batch_idx=batch.batch.to(torch.int32).contiguous(),
                fill_value=-1,
                epsilon=self.config.epsilon,
                sigma=self.config.sigma,
                cutoff=self.config.cutoff,
                switch_width=self.config.switch_width,
                half_list=self.config.half_list,
                atomic_energies=atomic_energies,
                forces=forces_buf,
                virials=virials_buf,
            )
            stresses = virial_to_stress(-virials_buf.view(num_graphs, 3, 3), cells)
        else:
            lj_energy_forces_batch_into(
                positions=positions,
                cells=cells,
                neighbor_matrix=neighbor_matrix,
                neighbor_shifts=neighbor_shifts,
                num_neighbors=num_neighbors,
                batch_idx=batch.batch.to(torch.int32).contiguous(),
                fill_value=-1,
                epsilon=self.config.epsilon,
                sigma=self.config.sigma,
                cutoff=self.config.cutoff,
                switch_width=self.config.switch_width,
                half_list=self.config.half_list,
                atomic_energies=atomic_energies,
                forces=forces_buf,
            )

        energies = aggregate_per_system_energy(atomic_energies, batch.batch, num_graphs)
        return self.build_results(
            ctx,
            energies=energies,
            forces=forces_buf if "forces" in ctx.outputs else None,
            stresses=stresses,
        )
