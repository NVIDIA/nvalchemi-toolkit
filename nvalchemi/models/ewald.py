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
"""Ewald summation electrostatics model wrapper.

Wraps the ``nvalchemiops`` Ewald summation interaction (real-space plus
reciprocal-space) as a :class:`~nvalchemi.models.base.BaseModelMixin`
compatible model for composable execution.

Usage
-----
::

    from nvalchemi.models import EwaldModelWrapper

    model = EwaldModelWrapper(cutoff=10.0)

Notes
-----
* Forces are computed via **autograd** (energy backpropagated to positions)
  inside the composable runtime, so ``spec.use_autograd`` is ``True``.
* Periodic boundary conditions are **required**.
* Input charges are read from ``node_charges`` and are expected to have shape
  ``(N,)`` or ``(N, 1)``.
* The Coulomb constant defaults to ``14.3996`` eV·Å/e², which gives energies
  in eV when positions are in Å and charges are in elementary-charge units.
* Ewald parameters (alpha and k-vectors) are auto-tuned from the requested
  accuracy target and cached per unique unit cell.
"""

from __future__ import annotations

from typing import Annotated

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
from pydantic import BaseModel, ConfigDict, Field, NonNegativeFloat, PositiveFloat
from torch import nn

from nvalchemi.models.base import BaseModelMixin, ModelConfig, NeighborConfig
from nvalchemi.models.utils import (
    _UNSET,
    COULOMB_CONSTANT,
    _resolve_config,
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
    unpack_kernel_result,
    virial_to_stress,
)

__all__ = ["EwaldCoulombConfig", "EwaldModelWrapper"]

DEFAULT_ELECTROSTATICS_ACCURACY = 1e-4


class EwaldCoulombConfig(BaseModel):
    """Configuration for :class:`EwaldModelWrapper`.

    Attributes
    ----------
    cutoff : float
        Real-space cutoff radius in Å.
    accuracy : float
        Target accuracy for automatic Ewald parameter estimation.
    alpha : float | None
        Optional explicit Ewald splitting parameter in inverse Å.
    k_cutoff : float | None
        Optional reciprocal-space cutoff in inverse Å.
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
    k_cutoff: Annotated[
        PositiveFloat | None,
        Field(description="Reciprocal-space cutoff (assumed Angstrom^-1)."),
    ] = None

    model_config = ConfigDict(extra="forbid")


_EwaldModelConfig = ModelConfig(
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


class EwaldModelWrapper(nn.Module, BaseModelMixin):
    """Ewald summation electrostatics model wrapper.

    Computes long-range Coulomb interactions through real-space and
    reciprocal-space Ewald terms and combines them into one additive
    electrostatics correction.

    Parameters
    ----------
    config : EwaldCoulombConfig or None, optional
        Optional prebuilt Ewald configuration.
    cutoff : float, optional
        Real-space interaction cutoff in Å.
    accuracy : float, optional
        Target accuracy for automatic Ewald parameter estimation.
    alpha : float or None, optional
        Explicit Ewald splitting parameter in inverse Å.
    k_cutoff : float or None, optional
        Explicit reciprocal-space cutoff in inverse Å.
    name : str or None, optional
        Optional stable display name used by the composable runtime.

    Attributes
    ----------
    spec : ModelConfig
        Execution contract describing the wrapper inputs, outputs, and
        neighbor-list requirements.
    """

    spec = _EwaldModelConfig

    def __init__(
        self,
        config: EwaldCoulombConfig | None = None,
        *,
        cutoff: float = _UNSET,
        accuracy: float = _UNSET,
        alpha: float | None = _UNSET,
        k_cutoff: float | None = _UNSET,
        name: str | None = None,
    ) -> None:
        super().__init__()
        self._name = name
        base_config = config
        config = _resolve_config(
            EwaldCoulombConfig,
            config,
            {
                "cutoff": cutoff,
                "accuracy": accuracy,
                "alpha": alpha,
                "k_cutoff": k_cutoff,
            },
        )
        self._config = config
        self.spec = replace_model_spec(
            _EwaldModelConfig,
            pbc_mode=_EwaldModelConfig.pbc_mode,
            neighbor_cutoff=config.cutoff,
        )

        default_config = EwaldCoulombConfig()
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
                    "k_cutoff": k_cutoff,
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
                "k_cutoff": default_config.k_cutoff,
                "name": None,
            },
            order=("cutoff", "accuracy", "alpha", "k_cutoff", "name"),
        )
        initialize_model_repr(
            self,
            static_kwargs=static_kwargs,
            kwarg_order=("cutoff", "accuracy", "alpha", "k_cutoff", "name"),
        )

    def __repr__(self) -> str:
        """Return one compact constructor-style representation."""

        return build_model_repr(self)

    def forward(self, data: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Run Ewald summation for one prepared input.

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
            If either Ewald kernel omits forces or virial tensors.
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

        alpha = self._config.alpha
        if alpha is None:
            alpha = estimate_ewald_parameters(
                positions,
                cell_tensor,
                batch_idx=batch_idx,
                accuracy=self._config.accuracy,
            ).alpha
        elif not isinstance(alpha, torch.Tensor):
            alpha = torch.full(
                (cell_tensor.shape[0],),
                float(alpha),
                dtype=cell_tensor.dtype,
                device=cell_tensor.device,
            )

        k_vectors = generate_k_vectors_ewald_summation(
            cell_tensor,
            k_cutoff=self._config.k_cutoff,
            accuracy=(self._config.accuracy if self._config.k_cutoff is None else None),
            alpha=alpha,
        )

        real_result = ewald_real_space(
            positions=positions,
            charges=charges.view(-1),
            cell=cell_tensor,
            alpha=alpha,
            neighbor_matrix=neighbor_matrix,
            neighbor_matrix_shifts=neighbor_shifts,
            batch_idx=batch_idx,
            compute_forces=True,
            compute_virial=True,
        )
        reciprocal_result = ewald_reciprocal_space(
            positions=positions,
            charges=charges.view(-1),
            cell=cell_tensor,
            k_vectors=k_vectors,
            alpha=alpha,
            batch_idx=batch_idx,
            compute_forces=True,
            compute_virial=True,
        )

        e_real, f_real, v_real = unpack_kernel_result(
            real_result, has_forces=True, has_virial=True
        )
        e_recip, f_recip, v_recip = unpack_kernel_result(
            reciprocal_result, has_forces=True, has_virial=True
        )

        per_atom_energies = (e_real + e_recip).to(positions.dtype)
        energies = aggregate_per_system_energy(
            per_atom_energies, batch_idx, num_systems
        )
        energies = energies * COULOMB_CONSTANT
        if f_real is None or f_recip is None or v_real is None or v_recip is None:
            raise RuntimeError(
                "EwaldModelWrapper expected forces and virial outputs from both kernels."
            )
        virial = (v_real + v_recip) * COULOMB_CONSTANT
        return {
            "energies": energies,
            "forces": (f_real + f_recip) * COULOMB_CONSTANT,
            "stresses": virial_to_stress(virial, cell_tensor),
        }
