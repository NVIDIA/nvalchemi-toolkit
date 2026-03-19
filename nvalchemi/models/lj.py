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
  autograd), so :attr:`~ModelCard.forces_via_autograd` is ``False``.
* Only a **single species** is supported in this wrapper.  Epsilon and sigma
  are scalar parameters shared across all atom pairs.
* Stress/virial computation (needed for NPT/NPH) is available via
  ``model_config.compute_stresses = True``.  When enabled, the wrapper
  returns a ``"stresses"`` key containing the **positive raw virial**
  :math:`W = +\\sum_{ij} r_{ij} \\otimes F_{ij}` (shape ``[B, 3, 3]``) in
  the model's energy unit, which is the framework convention for
  ``batch.stress``.  ``compute_pressure_tensor`` divides by :math:`V`
  internally to give pressure in energy / length\\ :sup:`3`.
  After calling ``Batch.from_data_list``, pre-allocate the placeholder:
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
    lj_energy_forces_batch_into,
    lj_energy_forces_virial_batch_into,
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
        LJ well-depth :math:`\\varepsilon` in the chosen energy unit.
        For argon: 0.0104 eV.
    sigma : float
        LJ zero-crossing distance :math:`\\sigma` in the chosen length unit.
        For argon: 3.40 Å.
    cutoff : float
        Interaction cutoff radius in the same length unit as ``sigma``.
    switch_width : float, optional
        Width of the C2-continuous switching region in the same length unit
        as ``sigma``; ``0.0`` disables switching (hard cutoff).
        Defaults to ``0.0``.
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
        half_list: bool = False,
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
        self._model_card: ModelCard = self._build_model_card()
        # Pre-allocated compute output buffers — resized lazily on first forward
        # or when N/B/dtype/device changes.
        self._atomic_energies_buf: torch.Tensor | None = None
        self._forces_buf: torch.Tensor | None = None
        self._virials_buf: torch.Tensor | None = None
        self._buf_N: int = 0
        self._buf_B: int = 0
        self._buf_dtype: torch.dtype | None = None
        self._buf_device: torch.device | None = None
        # Energy accumulation buffer (shape [B]).
        self._energies_buf: torch.Tensor | None = None
        # Cached all-zero neighbor-shifts for non-PBC runs (shape [N, K, 3] int32).
        self._null_shifts: torch.Tensor | None = None
        self._null_shifts_shape: tuple[int, int] = (0, 0)

    # ------------------------------------------------------------------
    # BaseModelMixin required properties
    # ------------------------------------------------------------------

    def _build_model_card(self) -> ModelCard:
        return ModelCard(
            forces_via_autograd=False,
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
    def model_card(self) -> ModelCard:
        return self._model_card

    def _ensure_compute_buffers(
        self, N: int, B: int, dtype: torch.dtype, device: torch.device
    ) -> None:
        """Allocate or resize per-step output buffers."""
        if (
            N != self._buf_N
            or B != self._buf_B
            or dtype != self._buf_dtype
            or device != self._buf_device
        ):
            self._atomic_energies_buf = torch.empty(N, dtype=dtype, device=device)
            self._forces_buf = torch.empty(N, 3, dtype=dtype, device=device)
            self._virials_buf = torch.empty(B, 9, dtype=dtype, device=device)
            self._buf_N = N
            self._buf_B = B
            self._buf_dtype = dtype
            self._buf_device = device
        if (
            self._energies_buf is None
            or self._energies_buf.shape[0] != B
            or self._energies_buf.dtype != dtype
            or self._energies_buf.device != device
        ):
            self._energies_buf = torch.empty(B, dtype=dtype, device=device)

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
            input_dict["cells"] = getattr(data, "cell", None)  # (B, 3, 3)
            input_dict["neighbor_shifts"] = getattr(
                data, "neighbor_shifts", None
            )  # (N, K, 3) int32
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
                # LJ kernel returns W = -sum r_ij x F_ij (negative-convention virial).
                # The framework convention for batch.stress is the positive raw virial
                # W_phys = +sum r_ij x F_ij (in the model's energy unit), so we negate.
                # NPT/NPH compute_pressure_tensor divides by V internally.
                # Variable-cell optimizers (FIRE2VariableCell) divide by V themselves
                # before calling stress_to_cell_force.
                output["stresses"] = -model_output["virials"]
            elif "stresses" in model_output:
                output["stresses"] = model_output["stresses"]
        return output

    def output_data(self) -> set[str]:
        """
        Return the set of keys that the model produces.
        """
        keys = {"energies"}
        if self.model_config.compute_forces:
            keys.add("forces")
        if self.model_config.compute_stresses:
            keys.add("stresses")
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
            OrderedDict with keys:

            * ``"energies"`` — shape ``[B, 1]``, in the model's energy unit.
            * ``"forces"`` — shape ``[N, 3]``, in energy / length units.
            * ``"stresses"`` *(if* ``compute_stresses=True``\ *)* — shape
              ``[B, 3, 3]``, positive raw virial
              :math:`W = +\\sum_{ij} r_{ij} \\otimes F_{ij}` in the model's
              energy unit.  Divide by cell volume to get the Cauchy stress.
        """
        inp = self.adapt_input(data, **kwargs)

        positions = inp["positions"]  # (N, 3)
        neighbor_matrix = inp["neighbor_matrix"]  # (N, K) int32
        num_neighbors = inp["num_neighbors"]  # (N,) int32
        batch_idx = inp["batch_idx"]  # (N,) int32
        fill_value = inp["fill_value"]  # int
        B = inp["num_graphs"]
        N = positions.shape[0]
        K = neighbor_matrix.shape[1]

        self._ensure_compute_buffers(N, B, positions.dtype, positions.device)

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
            if (
                self._null_shifts is None
                or self._null_shifts_shape != (N, K)
                or self._null_shifts.device != positions.device
            ):
                self._null_shifts = torch.zeros(
                    N, K, 3, dtype=torch.int32, device=positions.device
                )
                self._null_shifts_shape = (N, K)
            neighbor_shifts = self._null_shifts
        else:
            neighbor_shifts = neighbor_shifts.contiguous()

        if self.model_config.compute_stresses:
            lj_energy_forces_virial_batch_into(
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
                atomic_energies=self._atomic_energies_buf,
                forces=self._forces_buf,
                virials=self._virials_buf,
            )
            virials = self._virials_buf.view(B, 3, 3).clone()
        else:
            lj_energy_forces_batch_into(
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
                atomic_energies=self._atomic_energies_buf,
                forces=self._forces_buf,
            )
            virials = None

        # Scatter per-atom energies to per-system totals using pre-allocated buffer.
        self._energies_buf.zero_()
        self._energies_buf.scatter_add_(0, batch_idx.long(), self._atomic_energies_buf)

        # Clone outputs from internal buffers so callers receive independent tensors.
        # Without cloning, the next forward pass would overwrite the returned tensors
        # in-place, silently corrupting any stored references.
        model_output: dict[str, Any] = {
            "energies": self._energies_buf.unsqueeze(-1).clone(),  # (B, 1)
            "forces": self._forces_buf.clone(),
        }
        if virials is not None:
            model_output["virials"] = virials  # already cloned above

        return self.adapt_output(model_output, data)

    def export_model(self, path: Path, as_state_dict: bool = False) -> None:
        """
        Export model is not implemented for LennardJonesModelWrapper.
        """
        raise NotImplementedError
