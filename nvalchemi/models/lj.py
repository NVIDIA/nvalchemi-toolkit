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
:class:`~nvalchemi.models.base.BaseModelMixin`-compatible model, ready to
drop into any :class:`~nvalchemi.dynamics.base.BaseDynamics` engine.

Usage
-----
::

    from nvalchemi.models.lj import LennardJonesModelWrapper
    from nvalchemi.dynamics.hooks import NeighborListHook

    model = LennardJonesModelWrapper(
        epsilon=0.0104,   # eV (argon)
        sigma=3.40,       # Å
        cutoff=8.5,       # Å
    )

    # Register the neighbor-list hook so the batch gets neighbor_matrix
    # populated before each compute() call.
    nl_hook = NeighborListHook(model.model_card.neighbor_config)
    dynamics.register_hook(nl_hook)
    dynamics.model = model

Notes
-----
* Forces are computed **analytically** inside the Warp kernel (not via
  autograd), so :attr:`~ModelCard.forces_are_conservative` is ``False``.
* Only a **single species** is supported in this wrapper.  Epsilon and sigma
  are scalar parameters shared across all atom pairs.
* Stress/virial computation (needed for NPT/NPH) is available via
  ``model_config.compute_stresses = True``.  When enabled, the wrapper
  returns a ``"stress"`` key containing ``-W_LJ`` (the physical virial
  ``+Σ r_ij ⊗ F_ij``), which is what the NPT/NPH barostat kernels expect.
  After calling ``Batch.from_data_list``, set the placeholder directly:
  ``batch["stress"] = torch.zeros(batch.num_graphs, 3, 3)``.  This is
  required because ``"stress"`` is not a named ``AtomicData`` field and is
  therefore not carried through batching automatically.
"""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any

import torch
from torch import nn

from nvalchemi._typing import ModelOutputs
from nvalchemi.data import AtomicData, Batch
from nvalchemi.models._ops.lj import (
    lj_energy_forces_batch,
    lj_energy_forces_virial_batch,
)
from nvalchemi.models.base import (
    BaseModelMixin,
    ModelCard,
    ModelConfig,
    NeighborConfig,
    NeighborListFormat,
)

__all__ = ["LennardJonesModelWrapper"]


class LennardJonesModelWrapper(nn.Module, BaseModelMixin):
    """Warp-accelerated Lennard-Jones potential as a model wrapper.

    Parameters
    ----------
    epsilon : float
        LJ well-depth parameter (energy units, e.g. eV).
    sigma : float
        LJ zero-crossing distance (length units, e.g. Å).
    cutoff : float
        Interaction cutoff radius (same length units as positions).
    switch_width : float, optional
        Width of the C2-continuous switching region; ``0.0`` disables
        switching (hard cutoff).  Defaults to ``0.0``.
    half_list : bool, optional
        Pass ``True`` (default) if the neighbor matrix contains each pair
        once (half list).  Must match the ``half_fill`` argument given to
        :class:`~nvalchemi.dynamics.hooks.NeighborListHook`.
    max_neighbors : int, optional
        Maximum neighbors per atom used when building the neighbor matrix.
        Passed through to :class:`~nvalchemi.models.base.NeighborConfig`
        and read by :class:`~nvalchemi.dynamics.hooks.NeighborListHook`.
        Defaults to 128.

    Attributes
    ----------
    model_config : ModelConfig
        Mutable configuration controlling which outputs are computed.
        Set ``model.model_config.compute_stresses = True`` to enable
        virial computation for NPT/NPH simulations.
    """

    def __init__(
        self,
        epsilon: float,
        sigma: float,
        cutoff: float,
        switch_width: float = 0.0,
        half_list: bool = True,
        max_neighbors: int = 128,
    ) -> None:
        super().__init__()
        self.epsilon = epsilon
        self.sigma = sigma
        self.cutoff = cutoff
        self.switch_width = switch_width
        self.half_list = half_list
        self.max_neighbors = max_neighbors
        # Instance-level model_config so callers can mutate it.
        self.model_config = ModelConfig()

    # ------------------------------------------------------------------
    # BaseModelMixin required properties
    # ------------------------------------------------------------------

    @property
    def model_card(self) -> ModelCard:
        return ModelCard(
            forces_are_conservative=False,
            supports_energies=True,
            supports_forces=True,
            supports_stresses=True,
            supports_pbc=True,
            needs_pbc=False,
            supports_non_batch=False,
            neighbor_config=NeighborConfig(
                cutoff=self.cutoff,
                format=NeighborListFormat.MATRIX,
                half_list=self.half_list,
                max_neighbors=self.max_neighbors,
            ),
        )

    @property
    def embedding_shapes(self) -> dict[str, tuple[int, ...]]:
        return {}

    def compute_embeddings(
        self, data: AtomicData | Batch, **kwargs: Any
    ) -> AtomicData | Batch:
        """
        Compute embeddings for the LennardJonesModelWrapper.

        This method is not implemented for the LennardJonesModelWrapper, but it is included
        to demonstrate how to override the super() implementation.
        """
        raise NotImplementedError(
            "LennardJonesModelWrapper does not produce embeddings."
        )

    # ------------------------------------------------------------------
    # Input / output adaptation
    # ------------------------------------------------------------------

    def adapt_input(self, data: AtomicData | Batch, **kwargs: Any) -> dict[str, Any]:
        """Collect required inputs from *data* without enabling gradients.

        Unlike the base-class implementation this method deliberately does
        **not** call ``positions.requires_grad_(True)`` because forces are
        computed analytically by the Warp kernel rather than via autograd.
        """
        input_dict: dict[str, Any] = {}
        for key in self.input_data():
            value = getattr(data, key, None)
            if value is None:
                raise KeyError(f"'{key}' required but not found in input data.")
            input_dict[key] = value

        if isinstance(data, Batch):
            input_dict["batch_idx"] = data.batch.to(torch.int32)
            input_dict["ptr"] = data.ptr.to(torch.int32)
            input_dict["num_graphs"] = data.num_graphs
            input_dict["fill_value"] = data.num_nodes

            # Optional PBC inputs — silently absent for non-periodic runs.
            try:
                input_dict["cells"] = data.cell  # (B, 3, 3)
            except AttributeError:
                input_dict["cells"] = None
            try:
                input_dict["neighbor_shifts"] = data.neighbor_shifts  # (N, K, 3) int32
            except AttributeError:
                input_dict["neighbor_shifts"] = None
        else:
            raise TypeError(
                "LennardJonesModelWrapper requires a Batch input; "
                "got AtomicData.  Use Batch.from_data_list([data]) to wrap it."
            )

        return input_dict

    def adapt_output(self, model_output: Any, data: AtomicData | Batch) -> ModelOutputs:
        """
        Adapts the model output to the framework's expected format.

        The super() implementation will provide the initial OrderedDict with keys
        that are expected to be present in the model output. This method will then
        map the model outputs to this OrderedDict.

        Technically, this is not necessary for the LennardJonesModelWrapper, but it is included
        to demonstrate how to override the super() implementation.
        """
        output: ModelOutputs = OrderedDict()
        output["energies"] = model_output["energies"]
        if self.model_config.compute_forces:
            output["forces"] = model_output["forces"]
        if self.model_config.compute_stresses:
            if "virials" in model_output:
                # LJ kernel returns W = -Σ r_ij ⊗ F_ij (negative-convention virial).
                # NPT/NPH compute_pressure_tensor expects the positive convention
                # W_phys = +Σ r_ij ⊗ F_ij, so we negate here.
                # NOTE: variable-cell optimizers (FIRE2VariableCell, FIREVariableCell)
                # require the mechanical stress σ = W_phys / V, not the raw virial.
                # A volume-aware virial→stress conversion is needed for those cases;
                # see TODO in nvalchemi/models/_ops/lj.py.
                output["stress"] = -model_output["virials"]
            elif "stress" in model_output:
                output["stress"] = model_output["stress"]
        return output

    def output_data(self) -> set[str]:
        """
        Return the set of keys that the model produces.
        """
        keys = {"energies"}
        if self.model_config.compute_forces:
            keys.add("forces")
        if self.model_config.compute_stresses:
            keys.add("stress")
        return keys

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, data: AtomicData | Batch, **kwargs: Any) -> ModelOutputs:
        """Run the LJ kernel and return a :class:`ModelOutputs` dict.

        Parameters
        ----------
        data : Batch
            Batch containing ``positions``, ``neighbor_matrix``,
            ``num_neighbors``, and optionally ``cell`` / ``neighbor_shifts``
            (populated by :class:`~nvalchemi.dynamics.hooks.NeighborListHook`).

        Returns
        -------
        ModelOutputs
            OrderedDict with keys ``"energies"`` (shape ``[B, 1]``),
            ``"forces"`` (shape ``[N, 3]``), and optionally
            ``"stress"`` (shape ``[B, 3, 3]``) — the physical virial
            ``-W_LJ`` in units of eV, ready for NPT/NPH barostat use.
        """
        inp = self.adapt_input(data, **kwargs)

        positions = inp["positions"]  # (N, 3)
        neighbor_matrix = inp["neighbor_matrix"]  # (N, K) int32
        num_neighbors = inp["num_neighbors"]  # (N,) int32
        batch_idx = inp["batch_idx"]  # (N,) int32
        fill_value = inp["fill_value"]  # int
        B = inp["num_graphs"]
        N = positions.shape[0]

        # Build placeholder cell (identity) and shifts (zeros) for non-PBC.
        cells = inp.get("cells")
        if cells is None:
            cells = (
                torch.eye(3, dtype=positions.dtype, device=positions.device)
                .unsqueeze(0)
                .expand(B, 3, 3)
                .contiguous()
            )
        else:
            cells = cells.contiguous()

        neighbor_shifts = inp.get("neighbor_shifts")
        if neighbor_shifts is None:
            K = neighbor_matrix.shape[1]
            neighbor_shifts = torch.zeros(
                N, K, 3, dtype=torch.int32, device=positions.device
            )
        else:
            neighbor_shifts = neighbor_shifts.contiguous()

        if self.model_config.compute_stresses:
            atomic_energies, forces, virials_flat = lj_energy_forces_virial_batch(
                positions=positions,
                cells=cells,
                neighbor_matrix=neighbor_matrix.contiguous(),
                neighbor_shifts=neighbor_shifts,
                num_neighbors=num_neighbors.contiguous(),
                batch_idx=batch_idx.contiguous(),
                fill_value=fill_value,
                epsilon=self.epsilon,
                sigma=self.sigma,
                cutoff=self.cutoff,
                switch_width=self.switch_width,
                half_list=self.half_list,
            )
            # virials_flat: (B, 9) → (B, 3, 3)
            virials = virials_flat.view(B, 3, 3)
        else:
            atomic_energies, forces = lj_energy_forces_batch(
                positions=positions,
                cells=cells,
                neighbor_matrix=neighbor_matrix.contiguous(),
                neighbor_shifts=neighbor_shifts,
                num_neighbors=num_neighbors.contiguous(),
                batch_idx=batch_idx.contiguous(),
                fill_value=fill_value,
                epsilon=self.epsilon,
                sigma=self.sigma,
                cutoff=self.cutoff,
                switch_width=self.switch_width,
                half_list=self.half_list,
            )
            virials = None

        # Scatter per-atom energies to per-system totals.
        energies = torch.zeros(B, dtype=positions.dtype, device=positions.device)
        energies.scatter_add_(0, batch_idx.long(), atomic_energies)
        energies = energies.unsqueeze(-1)  # (B, 1)

        model_output: dict[str, Any] = {"energies": energies, "forces": forces}
        if virials is not None:
            model_output["virials"] = virials

        return self.adapt_output(model_output, data)

    def export_model(self, path: Path, as_state_dict: bool = False) -> None:
        raise NotImplementedError
