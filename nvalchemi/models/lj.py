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
"""Lennard-Jones model wrapper.

Wraps the Warp-accelerated Lennard-Jones interaction kernel as a
:class:`~nvalchemi.models.base.BaseModelMixin`-compatible model for
composable execution.

Usage
-----
::

    from nvalchemi.models import ComposableModelWrapper, LennardJonesModelWrapper

    model = LennardJonesModelWrapper(
        epsilon=0.0104,
        sigma=3.40,
        cutoff=8.5,
    )
    calc = ComposableModelWrapper(model)

Notes
-----
* Forces are computed **analytically** inside the Warp kernel (not via
  autograd), so ``spec.use_autograd`` is ``False``.
* Only a **single species** is supported in this wrapper.  ``epsilon`` and
  ``sigma`` are scalar parameters shared across all atom pairs.
* Stress/virial computation is available when periodic cell data is present.
  The kernel returns the virial tensor ``-W_LJ``; the wrapper converts to
  the positive convention expected by barostat integrators.
* Matrix-format neighbor lists are required.  The composable runtime can
  prepare them automatically from the published :class:`NeighborConfig`.
"""

from __future__ import annotations

from typing import Annotated

import torch
from pydantic import BaseModel, ConfigDict, Field
from torch import nn

from nvalchemi.models._ops.lj import (
    lj_energy_forces_batch_into,
    lj_energy_forces_virial_batch_into,
)
from nvalchemi.models.base import BaseModelMixin, ModelConfig, NeighborConfig
from nvalchemi.models.utils import (
    _UNSET,
    _resolve_config,
    aggregate_per_system_energy,
    build_model_repr,
    collect_nondefault_repr_kwargs,
    initialize_model_repr,
    mapping_get,
    normalize_batch_indices,
    virial_to_stress,
)

__all__ = ["LennardJonesConfig", "LennardJonesModelWrapper"]


class LennardJonesConfig(BaseModel):
    """Configuration for :class:`LennardJonesModelWrapper`."""

    epsilon: Annotated[float, Field(description="LJ well depth (eV).")]
    sigma: Annotated[float, Field(description="Zero-potential distance (Angstrom).")]
    cutoff: Annotated[float, Field(description="Interaction cutoff radius (Angstrom).")]
    switch_width: Annotated[
        float, Field(description="Switching width below cutoff.")
    ] = 0.0
    half_list: Annotated[
        bool, Field(description="Whether the neighbor list is a half-list.")
    ] = False

    model_config = ConfigDict(extra="forbid")


_LennardJonesModelConfig = ModelConfig(
    required_inputs=frozenset({"positions", "neighbor_matrix", "num_neighbors"}),
    optional_inputs=frozenset({"batch", "cell", "pbc", "neighbor_shifts"}),
    outputs=frozenset({"energies", "forces"}),
    optional_outputs={"stresses": frozenset({"cell", "pbc"})},
    additive_outputs=frozenset({"energies", "forces", "stresses"}),
    use_autograd=False,
    pbc_mode="any",
    neighbor_config=NeighborConfig(
        source="external",
        cutoff=6.0,
        format="matrix",
        half_list=False,
    ),
)


class LennardJonesModelWrapper(nn.Module, BaseModelMixin):
    """Warp-accelerated Lennard-Jones potential as a model wrapper.

    Parameters
    ----------
    config : LennardJonesConfig or None, optional
        Optional prebuilt Lennard-Jones configuration.
    epsilon : float, optional
        LJ well-depth parameter in energy units such as eV.
    sigma : float, optional
        LJ zero-crossing distance in the same length units as ``positions``.
    cutoff : float, optional
        Interaction cutoff radius.
    switch_width : float, optional
        Width of the C2-continuous switching region. ``0.0`` disables
        switching.
    half_list : bool, optional
        Whether each pair appears only once in the matrix neighbor list.
    name : str or None, optional
        Optional stable display name used by the composable runtime.

    Attributes
    ----------
    spec : ModelConfig
        Execution contract describing the wrapper inputs, outputs, and
        neighbor-list requirements.
    """

    spec = _LennardJonesModelConfig

    def __init__(
        self,
        config: LennardJonesConfig | None = None,
        *,
        epsilon: float = _UNSET,
        sigma: float = _UNSET,
        cutoff: float = _UNSET,
        switch_width: float = _UNSET,
        half_list: bool = _UNSET,
        name: str | None = None,
    ) -> None:
        super().__init__()
        self._name = name
        base_config = config
        config = _resolve_config(
            LennardJonesConfig,
            config,
            {
                "epsilon": epsilon,
                "sigma": sigma,
                "cutoff": cutoff,
                "switch_width": switch_width,
                "half_list": half_list,
            },
        )
        self._config = config
        self.spec = ModelConfig(
            required_inputs=_LennardJonesModelConfig.required_inputs,
            optional_inputs=_LennardJonesModelConfig.optional_inputs,
            outputs=_LennardJonesModelConfig.outputs,
            optional_outputs=dict(_LennardJonesModelConfig.optional_outputs),
            additive_outputs=_LennardJonesModelConfig.additive_outputs,
            use_autograd=_LennardJonesModelConfig.use_autograd,
            autograd_inputs=_LennardJonesModelConfig.autograd_inputs,
            autograd_outputs=_LennardJonesModelConfig.autograd_outputs,
            pbc_mode=_LennardJonesModelConfig.pbc_mode,
            neighbor_config=_LennardJonesModelConfig.neighbor_config.model_copy(
                update={"cutoff": config.cutoff, "half_list": config.half_list}
            ),
        )

        default_config = LennardJonesConfig(epsilon=1.0, sigma=1.0, cutoff=6.0)
        explicit_values: dict[str, object] = {}
        if base_config is not None:
            for field_name in base_config.model_fields_set:
                explicit_values[field_name] = getattr(base_config, field_name)
        explicit_values.update(
            {
                key: value
                for key, value in {
                    "epsilon": epsilon,
                    "sigma": sigma,
                    "cutoff": cutoff,
                    "switch_width": switch_width,
                    "half_list": half_list,
                    "name": name,
                }.items()
                if value is not _UNSET
            }
        )
        initialize_model_repr(
            self,
            static_kwargs=collect_nondefault_repr_kwargs(
                explicit_values=explicit_values,
                defaults={
                    "epsilon": default_config.epsilon,
                    "sigma": default_config.sigma,
                    "cutoff": default_config.cutoff,
                    "switch_width": default_config.switch_width,
                    "half_list": default_config.half_list,
                    "name": None,
                },
                order=(
                    "epsilon",
                    "sigma",
                    "cutoff",
                    "switch_width",
                    "half_list",
                    "name",
                ),
            ),
            kwarg_order=(
                "epsilon",
                "sigma",
                "cutoff",
                "switch_width",
                "half_list",
                "name",
            ),
        )

    def __repr__(self) -> str:
        """Return a compact constructor-style representation."""

        return build_model_repr(self)

    def forward(self, data: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Run the Lennard-Jones interaction kernel.

        Parameters
        ----------
        data
            Prepared input mapping resolved by the composable runtime. The
            mapping must provide ``positions``, ``neighbor_matrix``, and
            ``num_neighbors`` and may additionally include ``batch``,
            ``cell``, ``pbc``, and ``neighbor_shifts``.

        Returns
        -------
        dict[str, torch.Tensor]
            Mapping containing ``energies`` and ``forces`` and, for periodic
            inputs, ``stresses``.
        """

        positions = data["positions"]
        neighbor_matrix = data["neighbor_matrix"].contiguous()
        num_neighbors = data["num_neighbors"].contiguous()
        neighbor_shifts = mapping_get(data, "neighbor_shifts")
        if neighbor_shifts is None:
            neighbor_shifts = torch.zeros(
                (*neighbor_matrix.shape, 3),
                dtype=torch.int32,
                device=neighbor_matrix.device,
            )
        else:
            neighbor_shifts = neighbor_shifts.contiguous()

        batch_idx, num_graphs = normalize_batch_indices(
            positions, mapping_get(data, "batch")
        )
        cell = mapping_get(data, "cell")
        if cell is None:
            cells = (
                torch.eye(3, dtype=positions.dtype, device=positions.device)
                .unsqueeze(0)
                .expand(num_graphs, 3, 3)
                .contiguous()
            )
            periodic = False
        else:
            cells = (cell if cell.ndim == 3 else cell.unsqueeze(0)).contiguous()
            pbc = mapping_get(data, "pbc")
            periodic = pbc is not None and bool(torch.as_tensor(pbc).any().item())

        atomic_energies = torch.empty(
            positions.shape[0],
            dtype=positions.dtype,
            device=positions.device,
        )
        forces_buf = torch.empty_like(positions)
        outputs: dict[str, torch.Tensor] = {}
        if periodic:
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
                batch_idx=batch_idx.to(torch.int32).contiguous(),
                fill_value=positions.shape[0],
                epsilon=self._config.epsilon,
                sigma=self._config.sigma,
                cutoff=self._config.cutoff,
                switch_width=self._config.switch_width,
                half_list=self._config.half_list,
                atomic_energies=atomic_energies,
                forces=forces_buf,
                virials=virials_buf,
            )
            outputs["stresses"] = virial_to_stress(
                -virials_buf.view(num_graphs, 3, 3), cells
            )
        else:
            lj_energy_forces_batch_into(
                positions=positions,
                cells=cells,
                neighbor_matrix=neighbor_matrix,
                neighbor_shifts=neighbor_shifts,
                num_neighbors=num_neighbors,
                batch_idx=batch_idx.to(torch.int32).contiguous(),
                fill_value=positions.shape[0],
                epsilon=self._config.epsilon,
                sigma=self._config.sigma,
                cutoff=self._config.cutoff,
                switch_width=self._config.switch_width,
                half_list=self._config.half_list,
                atomic_energies=atomic_energies,
                forces=forces_buf,
            )

        outputs["energies"] = aggregate_per_system_energy(
            atomic_energies, batch_idx, num_graphs
        )
        outputs["forces"] = forces_buf
        return outputs
