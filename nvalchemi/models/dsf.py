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
"""Damped shifted force (DSF) electrostatics model wrapper.

Wraps the ``nvalchemiops`` DSF Coulomb interaction as a
:class:`~nvalchemi.models.base.BaseModelMixin`-compatible model for
composable execution.

Usage
-----
::

    from nvalchemi.models import DSFModelWrapper

    model = DSFModelWrapper(cutoff=12.0, alpha=0.2)

Notes
-----
* DSF supports both periodic and non-periodic inputs.
* The wrapper expects COO-format neighbor data prepared by the composable
  runtime.
* Forces use a **hybrid** strategy: the kernel returns pair forces directly,
  but the charge-dependent contribution requires autograd through the
  composable runtime (``spec.use_autograd`` is ``True``).
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

import torch
from nvalchemiops.torch.interactions.electrostatics.dsf import dsf_coulomb
from pydantic import BaseModel, ConfigDict, Field, NonNegativeFloat, PositiveFloat
from torch import nn

from nvalchemi.models.base import (
    BaseModelMixin,
    ModelConfig,
    NeighborConfig,
    _resolve_config,
)
from nvalchemi.models.utils import (
    _UNSET,
    build_model_repr,
    collect_nondefault_repr_kwargs,
    infer_periodic_mode,
    initialize_model_repr,
    mapping_get,
    normalize_batch_indices,
    normalize_batched_cell,
    normalize_node_charges,
    replace_model_spec,
    virial_to_stress,
)

__all__ = ["DSFCoulombConfig", "DSFModelWrapper"]


class DSFCoulombConfig(BaseModel):
    """Configuration for :class:`DSFModelWrapper`.

    Attributes
    ----------
    cutoff : float
        Interaction cutoff radius in Å.
    alpha : float
        DSF damping parameter in inverse Å.
    """

    cutoff: Annotated[
        PositiveFloat,
        Field(description="Interaction cutoff radius (assumed Angstrom)."),
    ] = 15.0
    alpha: Annotated[
        NonNegativeFloat,
        Field(description="DSF damping parameter (assumed Angstrom^-1)."),
    ] = 0.2

    model_config = ConfigDict(extra="forbid")


_DSFModelConfig = ModelConfig(
    required_inputs=frozenset(
        {"positions", "node_charges", "edge_index", "neighbor_ptr", "unit_shifts"}
    ),
    optional_inputs=frozenset({"cell", "pbc", "batch"}),
    outputs=frozenset({"energies", "forces"}),
    optional_outputs={"stresses": frozenset({"cell", "pbc"})},
    additive_outputs=frozenset({"energies", "forces", "stresses"}),
    use_autograd=True,
    pbc_mode="any",
    neighbor_config=NeighborConfig(
        source="external",
        cutoff=15.0,
        format="coo",
        half_list=False,
    ),
)


class DSFModelWrapper(nn.Module, BaseModelMixin):
    """DSF Coulomb electrostatics model wrapper.

    Parameters
    ----------
    config : DSFCoulombConfig or None, optional
        Optional prebuilt DSF configuration.
    cutoff : float, optional
        Interaction cutoff radius in Å.
    alpha : float, optional
        DSF damping parameter in inverse Å.
    pbc_mode : {"non-pbc", "pbc", "any"}, optional
        Declared periodic-boundary support contract for the wrapper.
    name : str or None, optional
        Optional stable display name used by the composable runtime.

    Attributes
    ----------
    spec : ModelConfig
        Execution contract describing the wrapper inputs, outputs, and
        neighbor-list requirements.
    """

    spec = _DSFModelConfig

    def __init__(
        self,
        config: DSFCoulombConfig | None = None,
        *,
        cutoff: float = _UNSET,
        alpha: float = _UNSET,
        pbc_mode: Literal["non-pbc", "pbc", "any"] = "any",
        name: str | None = None,
    ) -> None:
        super().__init__()
        self._name = name
        base_config = config
        config = _resolve_config(
            DSFCoulombConfig,
            config,
            {"cutoff": cutoff, "alpha": alpha},
        )
        self._config = config
        self.spec = replace_model_spec(
            _DSFModelConfig,
            pbc_mode=pbc_mode,
            neighbor_cutoff=config.cutoff,
        )
        default_config = DSFCoulombConfig()
        explicit_values: dict[str, object] = {}
        if base_config is not None:
            for field_name in base_config.model_fields_set:
                explicit_values[field_name] = getattr(base_config, field_name)
        explicit_values.update(
            {
                key: value
                for key, value in {
                    "cutoff": cutoff,
                    "alpha": alpha,
                    "pbc_mode": pbc_mode,
                    "name": name,
                }.items()
                if value is not _UNSET
            }
        )
        static_kwargs = collect_nondefault_repr_kwargs(
            explicit_values=explicit_values,
            defaults={
                "cutoff": default_config.cutoff,
                "alpha": default_config.alpha,
                "pbc_mode": "any",
                "name": None,
            },
            order=("cutoff", "alpha", "pbc_mode", "name"),
        )
        initialize_model_repr(
            self,
            static_kwargs=static_kwargs,
            kwarg_order=("cutoff", "alpha", "pbc_mode", "name"),
        )

    def __repr__(self) -> str:
        """Return a compact constructor-style representation."""

        return build_model_repr(self)

    def forward(self, data: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Run DSF electrostatics for one prepared input.

        Parameters
        ----------
        data
            Prepared input mapping resolved by the composable runtime.

        Returns
        -------
        dict[str, torch.Tensor]
            Mapping with ``energies`` and ``forces`` and, for periodic inputs,
            ``stresses``.
        """

        positions = data["positions"]
        charges = normalize_node_charges(data["node_charges"])

        cell = mapping_get(data, "cell")
        pbc = mapping_get(data, "pbc")
        periodic = infer_periodic_mode(
            cell=cell,
            pbc=pbc,
            model_name=type(self).__name__,
            allow_non_pbc=True,
        )

        batch_idx, num_systems = normalize_batch_indices(
            positions, mapping_get(data, "batch")
        )
        compute_forces = True
        compute_virial = periodic

        kwargs: dict[str, Any] = {
            "positions": positions,
            "charges": charges,
            "cutoff": self._config.cutoff,
            "alpha": self._config.alpha,
            "batch_idx": batch_idx,
            "num_systems": num_systems,
            "compute_forces": compute_forces,
            "compute_virial": compute_virial,
            "neighbor_list": data["edge_index"],
            "neighbor_ptr": data["neighbor_ptr"],
            "unit_shifts": data["unit_shifts"],
        }
        if periodic and cell is not None:
            kwargs["cell"] = cell

        raw = dsf_coulomb(**kwargs)
        energies = raw[0]
        if energies.ndim == 1:
            energies = energies.unsqueeze(-1)

        outputs: dict[str, torch.Tensor] = {
            "energies": energies,
            "forces": raw[1],
        }
        if compute_virial:
            virial = raw[2]
            cell_for_stress = normalize_batched_cell(cell)
            outputs["stresses"] = virial_to_stress(-virial, cell_for_stress)
        return outputs
