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
    from nvalchemi.hooks import NeighborListHook
    from nvalchemi.dynamics.base import DynamicsStage

    model = LennardJonesModelWrapper(
        epsilon=0.0104,   # eV (argon)
        sigma=3.40,       # Å
        cutoff=8.5,       # Å
    )

    # Register the neighbor-list hook so the batch gets neighbor_matrix
    # populated before each compute() call.
    nl_hook = NeighborListHook(model.model_config.neighbor_config, stage=DynamicsStage.BEFORE_COMPUTE)
    dynamics.register_hook(nl_hook)
    dynamics.model = model

Notes
-----
* Forces are computed **analytically** inside the Warp kernel (not via
  autograd), so ``"forces"`` is NOT in ``autograd_outputs``.
* Only a **single species** is supported in this wrapper.  Epsilon and sigma
  are scalar parameters shared across all atom pairs.
* Stress/virial computation (needed for NPT/NPH) is available via
  ``model_config.active_outputs`` including ``"stress"``.  When enabled, the
  wrapper returns a ``"stress"`` key containing the Cauchy stress
  ``W/V`` in energy units.  After calling ``Batch.from_data_list``, set the
  placeholder directly:
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
        :class:`~nvalchemi.hooks.NeighborListHook`.

    Attributes
    ----------
    model_config : ModelConfig
        Mutable configuration controlling which outputs are computed.
        Include ``"stress"`` in ``model_config.active_outputs`` to enable
        virial computation for NPT/NPH simulations.
    """

    def __init__(
        self,
        epsilon: float,
        sigma: float,
        cutoff: float,
        switch_width: float = 0.0,
        half_list: bool = False,
    ) -> None:
        super().__init__()
        self.epsilon = epsilon
        self.sigma = sigma
        self.cutoff = cutoff
        self.switch_width = switch_width
        self.half_list = half_list
        # Instance-level model_config so callers can mutate it.
        # active_outputs defaults to energy + forces; stress is opt-in
        # via model.set_config("active_outputs", {"energy", "forces", "stress"})
        # for NPT/NPH simulations.
        self.model_config = ModelConfig(
            outputs=frozenset({"energy", "forces", "stress"}),
            active_outputs={"energy", "forces"},
            autograd_outputs=frozenset(),
            autograd_inputs=frozenset({"positions"}),
            required_inputs=frozenset(),
            optional_inputs=frozenset(),
            supports_pbc=True,
            needs_pbc=False,
            neighbor_config=NeighborConfig(
                cutoff=self.cutoff,
                format=NeighborListFormat.MATRIX,
                half_list=self.half_list,
            ),
        )
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
            input_dict["batch_idx"] = data.batch_idx.to(torch.int32)
            input_dict["ptr"] = data.batch_ptr.to(torch.int32)
            input_dict["num_graphs"] = data.num_graphs
            input_dict["fill_value"] = data.num_nodes

            # Optional PBC inputs — silently absent for non-periodic runs.
            input_dict["cells"] = getattr(data, "cell", None)  # (B, 3, 3)
            input_dict["neighbor_matrix_shifts"] = getattr(
                data, "neighbor_matrix_shifts", None
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
        """
        output: ModelOutputs = OrderedDict()
        output["energy"] = model_output["energy"]
        if "forces" in self.model_config.active_outputs:
            output["forces"] = model_output["forces"]
        if "stress" in self.model_config.active_outputs:
            if "virial" in model_output:
                if not hasattr(data, "cell") or data.cell is None:
                    raise ValueError(
                        "stress output requires cell for volume computation"
                    )
                # Cauchy stress sigma = W/V (eV/A^3).
                virial = model_output["virial"]
                volume = torch.det(data.cell).abs().view(-1, 1, 1)
                output["stress"] = virial / volume
            elif "stress" in model_output:
                output["stress"] = model_output["stress"]
            else:
                raise RuntimeError(
                    "'stress' is in active_outputs but missing from model output"
                )
        return output

    def output_data(self) -> set[str]:
        """
        Return the set of keys that the model produces.
        """
        keys = {"energy"}
        if "forces" in self.model_config.active_outputs:
            keys.add("forces")
        if "stress" in self.model_config.active_outputs:
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
            ``num_neighbors``, and optionally ``cell`` / ``neighbor_matrix_shifts``
            (populated by :class:`~nvalchemi.hooks.NeighborListHook`).

        Returns
        -------
        ModelOutputs
            OrderedDict with keys ``"energy"`` (shape ``[B, 1]``),
            ``"forces"`` (shape ``[N, 3]``), and optionally
            ``"stress"`` (shape ``[B, 3, 3]``) — Cauchy stress
            ``W/V`` in energy units.
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

        neighbor_matrix_shifts = inp.get("neighbor_matrix_shifts")
        if neighbor_matrix_shifts is None:
            if (
                self._null_shifts is None
                or self._null_shifts_shape != (N, K)
                or self._null_shifts.device != positions.device
            ):
                self._null_shifts = torch.zeros(
                    N, K, 3, dtype=torch.int32, device=positions.device
                )
                self._null_shifts_shape = (N, K)
            neighbor_matrix_shifts = self._null_shifts
        else:
            neighbor_matrix_shifts = neighbor_matrix_shifts.contiguous()

        compute_stresses = "stress" in self.model_config.active_outputs

        if compute_stresses:
            lj_energy_forces_virial_batch_into(
                positions=positions,
                cells=cells,
                neighbor_matrix=neighbor_matrix.contiguous(),
                neighbor_matrix_shifts=neighbor_matrix_shifts,
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
                virial=self._virials_buf,
            )
            virials = self._virials_buf.view(B, 3, 3).clone()
        else:
            lj_energy_forces_batch_into(
                positions=positions,
                cells=cells,
                neighbor_matrix=neighbor_matrix.contiguous(),
                neighbor_matrix_shifts=neighbor_matrix_shifts,
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
        self._energies_buf.scatter_add_(0, batch_idx, self._atomic_energies_buf)

        # Clone outputs from internal buffers so callers receive independent tensors.
        # Without cloning, the next forward pass would overwrite the returned tensors
        # in-place, silently corrupting any stored references.
        model_output: dict[str, Any] = {
            "energy": self._energies_buf.unsqueeze(-1).clone(),  # (B, 1)
            "forces": self._forces_buf.clone(),
        }
        if virials is not None:
            model_output["virial"] = virials  # already cloned above

        return self.adapt_output(model_output, data)

    def export_model(self, path: Path, as_state_dict: bool = False) -> None:
        """
        Export model is not implemented for LennardJonesModelWrapper.
        """
        raise NotImplementedError
