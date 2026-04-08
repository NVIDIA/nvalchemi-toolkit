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
"""Particle Mesh Ewald (PME) electrostatics model wrapper.

Wraps the ``nvalchemiops`` PME interaction as a
:class:`~nvalchemi.models.base.BaseModelMixin`-compatible model for
composable execution.

Usage
-----
::

    from nvalchemi.models.pme import PMEModelWrapper

    model = PMEModelWrapper(cutoff=10.0)

Notes
-----
* Forces are computed via **autograd** (energy backpropagated to positions)
  inside the composable runtime, so ``spec.use_autograd`` is ``True``.
* Periodic boundary conditions are **required**.
* Input charges are read from ``node_charges`` and are expected to have shape
  ``(N,)`` or ``(N, 1)``.
* The Coulomb constant defaults to ``14.3996`` eV·Å/e², which gives energies
  in eV when positions are in Å and charges are in elementary-charge units.
* PME achieves :math:`O(N \\log N)` scaling via FFT-based reciprocal space
  calculations, making it more efficient than Ewald for large systems.
* Mesh k-vectors and PME parameters are auto-tuned from the requested
  accuracy target and cached per unique unit cell.
"""

from __future__ import annotations

import warnings
from typing import Annotated

import torch
from nvalchemiops.torch.interactions.electrostatics.k_vectors import (
    generate_k_vectors_pme,
)
from nvalchemiops.torch.interactions.electrostatics.parameters import (
    estimate_pme_parameters,
)
from nvalchemiops.torch.interactions.electrostatics.pme import particle_mesh_ewald
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeFloat,
    NonNegativeInt,
    PositiveFloat,
    PositiveInt,
)
from torch import nn

from nvalchemi.models.base import (
    BaseModelMixin,
    ModelConfig,
    NeighborConfig,
    _resolve_config,
)
from nvalchemi.models.utils import (
    _UNSET,
    COULOMB_CONSTANT,
    aggregate_per_system_energy,
    build_model_repr,
    collect_nondefault_repr_kwargs,
    infer_periodic_mode,
    initialize_model_repr,
    mapping_get,
    normalize_batch_indices,
    normalize_batched_cell,
    normalize_node_charges,
    replace_model_spec,
    resolve_matrix_neighbor_shifts,
    virial_to_stress,
)

__all__ = ["PMEConfig", "PMEModelWrapper"]

DEFAULT_ELECTROSTATICS_ACCURACY = 1e-4


class PMEConfig(BaseModel):
    """Configuration for :class:`PMEModelWrapper`.

    Attributes
    ----------
    cutoff : float
        Real-space cutoff radius in Å.
    accuracy : float
        Target accuracy for automatic PME parameter estimation.
    alpha : float | None
        Optional Ewald splitting parameter in inverse Å.
    mesh_spacing : float | None
        Target reciprocal mesh spacing used when explicit mesh dimensions
        are not provided.
    mesh_dimensions : tuple[int, int, int] | None
        Optional explicit reciprocal-space mesh dimensions.
    spline_order : int
        B-spline interpolation order used for charge assignment.
    """

    cutoff: Annotated[
        PositiveFloat, Field(description="Real-space cutoff radius (assumed Angstrom).")
    ] = 15.0
    accuracy: Annotated[
        PositiveFloat, Field(description="Relative accuracy target for auto-tuning.")
    ] = DEFAULT_ELECTROSTATICS_ACCURACY
    alpha: Annotated[
        NonNegativeFloat | None,
        Field(description="Ewald splitting parameter (assumed Angstrom^-1)."),
    ] = None
    mesh_spacing: Annotated[
        PositiveFloat | None,
        Field(description="Target reciprocal mesh spacing (assumed Angstrom)."),
    ] = None
    mesh_dimensions: Annotated[
        tuple[NonNegativeInt, NonNegativeInt, NonNegativeInt] | None,
        Field(description="Explicit reciprocal-space mesh dimensions."),
    ] = None
    spline_order: Annotated[
        PositiveInt,
        Field(description="B-spline interpolation order for charge assignment."),
    ] = 4

    model_config = ConfigDict(extra="forbid")


_PMEModelConfig = ModelConfig(
    required_inputs=frozenset(
        {
            "positions",
            "node_charges",
            "cell",
            "pbc",
            "neighbor_matrix",
            "num_neighbors",
        }
    ),
    optional_inputs=frozenset({"neighbor_shifts"}),
    outputs=frozenset({"energies", "forces"}),
    optional_outputs={"stresses": frozenset({"cell", "pbc"})},
    additive_outputs=frozenset({"energies", "forces", "stresses"}),
    use_autograd=True,
    pbc_mode="pbc",
    neighbor_config=NeighborConfig(
        source="external",
        cutoff=15.0,
        format="matrix",
        half_list=False,
    ),
)


class PMEModelWrapper(nn.Module, BaseModelMixin):
    """Particle Mesh Ewald electrostatics model wrapper.

    Computes long-range Coulomb interactions through the PME method, using
    a real-space short-range term together with FFT-based reciprocal-space
    evaluation.

    Parameters
    ----------
    config : PMEConfig or None, optional
        Optional prebuilt PME configuration.
    cutoff : float, optional
        Real-space interaction cutoff in Å.
    accuracy : float, optional
        Target accuracy for automatic parameter estimation.
    alpha : float or None, optional
        Explicit Ewald splitting parameter in inverse Å.
    mesh_spacing : float or None, optional
        Target reciprocal mesh spacing in Å.
    mesh_dimensions : tuple[int, int, int] or None, optional
        Explicit reciprocal-space mesh dimensions.  Overrides automatic
        estimation from ``mesh_spacing`` when provided.
    spline_order : int, optional
        B-spline interpolation order.
    name : str or None, optional
        Optional stable display name used by the composable runtime.

    Attributes
    ----------
    spec : ModelConfig
        Execution contract describing the wrapper inputs, outputs, and
        neighbor-list requirements.
    """

    spec = _PMEModelConfig

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
        name: str | None = None,
    ) -> None:
        super().__init__()
        self._name = name
        base_config = config
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
            },
        )
        self._config = config
        self.spec = replace_model_spec(
            _PMEModelConfig,
            pbc_mode=_PMEModelConfig.pbc_mode,
            neighbor_cutoff=config.cutoff,
        )

        default_config = PMEConfig()
        explicit_values: dict[str, object] = {}
        if base_config is not None:
            for field_name in base_config.model_fields_set:
                explicit_values[field_name] = getattr(base_config, field_name)
        explicit_values.update(
            {
                key: value
                for key, value in {
                    "cutoff": cutoff,
                    "accuracy": accuracy,
                    "alpha": alpha,
                    "mesh_spacing": mesh_spacing,
                    "mesh_dimensions": mesh_dimensions,
                    "spline_order": spline_order,
                    "name": name,
                }.items()
                if value is not _UNSET
            }
        )
        static_kwargs = collect_nondefault_repr_kwargs(
            explicit_values=explicit_values,
            defaults={
                "cutoff": default_config.cutoff,
                "accuracy": default_config.accuracy,
                "alpha": default_config.alpha,
                "mesh_spacing": default_config.mesh_spacing,
                "mesh_dimensions": default_config.mesh_dimensions,
                "spline_order": default_config.spline_order,
                "name": None,
            },
            order=(
                "cutoff",
                "accuracy",
                "alpha",
                "mesh_spacing",
                "mesh_dimensions",
                "spline_order",
                "name",
            ),
        )
        initialize_model_repr(
            self,
            static_kwargs=static_kwargs,
            kwarg_order=(
                "cutoff",
                "accuracy",
                "alpha",
                "mesh_spacing",
                "mesh_dimensions",
                "spline_order",
                "name",
            ),
        )

    def __repr__(self) -> str:
        """Return one compact constructor-style representation."""

        return build_model_repr(self)

    def forward(self, data: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Run the PME electrostatics kernel for one prepared input.

        Parameters
        ----------
        data
            Prepared input mapping resolved by the composable runtime.

        Returns
        -------
        dict[str, torch.Tensor]
            Mapping with ``energies``, ``forces``, and ``stresses``.

        Raises
        ------
        RuntimeError
            If the underlying PME kernel does not return forces or virial
            tensors.
        """

        positions = data["positions"]
        charges = normalize_node_charges(data["node_charges"])

        cell = data["cell"]
        infer_periodic_mode(
            cell=cell,
            pbc=mapping_get(data, "pbc"),
            model_name=type(self).__name__,
            allow_non_pbc=False,
        )
        cell_tensor = normalize_batched_cell(cell)
        neighbor_matrix = data["neighbor_matrix"].contiguous()
        neighbor_shifts = resolve_matrix_neighbor_shifts(
            neighbor_matrix,
            mapping_get(data, "neighbor_shifts"),
            periodic=True,
            model_name=type(self).__name__,
            allow_missing_non_pbc=False,
        )

        batch_idx, num_systems = normalize_batch_indices(
            positions, mapping_get(data, "batch")
        )

        params = None
        if self._config.alpha is None or self._config.mesh_dimensions is None:
            params = estimate_pme_parameters(
                positions,
                cell_tensor,
                batch_idx=batch_idx,
                accuracy=self._config.accuracy,
            )

        if self._config.alpha is not None:
            alpha_val = float(self._config.alpha)
        else:
            alpha_val = float(params.alpha.mean().item())

        alpha = torch.full(
            (cell_tensor.shape[0],),
            alpha_val,
            dtype=cell_tensor.dtype,
            device=cell_tensor.device,
        )

        if self._config.alpha is None and num_systems > 1:
            volumes = torch.linalg.det(cell_tensor).abs()
            if volumes.min() > 0 and (volumes.max() / volumes.min()) > 1.1:
                warnings.warn(
                    "PMEModelWrapper is using a single mean alpha across a batch with heterogeneous cell volumes.",
                    UserWarning,
                    stacklevel=2,
                )

        mesh_dimensions = (
            self._config.mesh_dimensions
            if self._config.mesh_dimensions is not None
            else params.mesh_dimensions
        )
        k_vectors, k_squared = generate_k_vectors_pme(cell_tensor, mesh_dimensions)

        result = particle_mesh_ewald(
            positions=positions,
            charges=charges.view(-1),
            cell=cell_tensor,
            alpha=alpha,
            mesh_dimensions=mesh_dimensions,
            spline_order=self._config.spline_order,
            batch_idx=batch_idx,
            k_vectors=k_vectors,
            k_squared=k_squared,
            neighbor_matrix=neighbor_matrix,
            neighbor_matrix_shifts=neighbor_shifts,
            compute_forces=True,
            compute_virial=True,
            accuracy=self._config.accuracy,
        )

        if isinstance(result, torch.Tensor):
            raise RuntimeError(
                "PMEModelWrapper expected forces and virial from the PME kernel."
            )
        items = list(result)
        per_atom_energies = items[0]
        forces = items[1] if len(items) > 1 else None
        virial = items[2] if len(items) > 2 else None

        energies = aggregate_per_system_energy(
            per_atom_energies.to(positions.dtype), batch_idx, num_systems
        )
        energies = energies * COULOMB_CONSTANT
        if forces is None or virial is None:
            raise RuntimeError(
                "PMEModelWrapper expected both forces and virial outputs."
            )
        return {
            "energies": energies,
            "forces": forces * COULOMB_CONSTANT,
            "stresses": virial_to_stress(virial * COULOMB_CONSTANT, cell_tensor),
        }
