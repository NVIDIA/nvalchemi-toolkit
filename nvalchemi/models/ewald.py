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

Wraps the ``nvalchemiops`` Ewald summation interaction (real-space +
reciprocal-space) as a :class:`~nvalchemi.models.base.BaseModelMixin`-compatible
model, ready to drop into any :class:`~nvalchemi.dynamics.base.BaseDynamics`
engine.

Usage
-----
::

    from nvalchemi.models.ewald import EwaldModelWrapper
    from nvalchemi.dynamics.hooks import NeighborListHook

    model = EwaldModelWrapper(cutoff=10.0)

    nl_hook = NeighborListHook(model.model_card.neighbor_config)
    dynamics.register_hook(nl_hook)
    dynamics.model = model

Notes
-----
* Forces are computed **analytically** inside the Warp kernel (not via
  autograd), so :attr:`~ModelCard.forces_via_autograd` is ``False``.
* Periodic boundary conditions are **required** (``needs_pbc=True``).
* Input charges are read from ``data.node_charges`` (shape ``[N]``).
* The Coulomb constant defaults to ``14.3996`` eV·Å/e², which gives energies
  in eV when positions are in Å and charges are in elementary charge units.
* k-vectors and Ewald parameters are cached per unique unit cell.  Call
  :meth:`invalidate_cache` to force recomputation.
"""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any

import torch
from torch import nn

from nvalchemi._typing import ModelOutputs
from nvalchemi.data import AtomicData, Batch
from nvalchemi.models._ops.neighbor_filter import prepare_neighbors_for_model
from nvalchemi.models.base import (
    BaseModelMixin,
    ModelCard,
    ModelConfig,
    NeighborConfig,
    NeighborListFormat,
)

__all__ = ["EwaldModelWrapper"]


class EwaldModelWrapper(nn.Module, BaseModelMixin):
    """Ewald summation electrostatics potential as a model wrapper.

    Computes long-range Coulomb interactions via the Ewald method, splitting
    contributions into real-space (erfc-damped, handled by a neighbor matrix)
    and reciprocal-space (structure factor summation) components.

    Parameters
    ----------
    cutoff : float
        Real-space interaction cutoff in Å.
    accuracy : float, optional
        Target accuracy for automatic Ewald parameter estimation.
        Defaults to ``1e-6``.
    coulomb_constant : float, optional
        Coulomb prefactor :math:`k_e` in eV·Å/e².
        Defaults to ``14.3996`` (standard value for Å/e/eV unit system).
    max_neighbors : int, optional
        Maximum neighbors per atom for the dense neighbor matrix.
        Defaults to 256.


    Attributes
    ----------
    model_config : ModelConfig
        Mutable configuration controlling which outputs are computed.
        Set ``model.model_config.compute_stresses = True`` to enable
        virial computation for NPT/NPH simulations.
    """

    def __init__(
        self,
        cutoff: float,
        accuracy: float = 1e-6,
        coulomb_constant: float = 14.3996,
        max_neighbors: int = 256,
    ) -> None:
        super().__init__()
        self.cutoff = cutoff
        self.accuracy = accuracy
        self.coulomb_constant = coulomb_constant
        self.max_neighbors = max_neighbors
        self.model_config = ModelConfig()
        self._model_card: ModelCard = self._build_model_card()

        # k-vector / parameter cache.
        # Invalidated automatically when cell changes, or manually via invalidate_cache().
        self._cache_valid: bool = False
        self._cached_alpha: torch.Tensor | None = None
        self._cached_k_vectors: torch.Tensor | None = None
        # Cached cell for automatic invalidation detection (e.g. NPT).
        self._cached_cell: torch.Tensor | None = None
        # Pre-allocated energy accumulation buffer (shape [B]).
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
            needs_pbc=True,
            supports_non_batch=False,
            needs_node_charges=True,
            neighbor_config=NeighborConfig(
                cutoff=self.cutoff,
                format=NeighborListFormat.MATRIX,
                half_list=False,
                max_neighbors=self.max_neighbors,
            ),
        )

    @property
    def model_card(self) -> ModelCard:
        return self._model_card

    @property
    def embedding_shapes(self) -> dict[str, tuple[int, ...]]:
        return {}

    def compute_embeddings(
        self, data: AtomicData | Batch, **kwargs: Any
    ) -> AtomicData | Batch:
        """Compute embeddings is not meaningful for Ewald models."""
        raise NotImplementedError("EwaldModelWrapper does not produce embeddings.")

    # ------------------------------------------------------------------
    # Input / output key declarations
    # ------------------------------------------------------------------

    def input_data(self) -> set[str]:
        """Return required input keys (override to drop ``atomic_numbers``)."""
        return {"positions", "node_charges", "neighbor_matrix", "num_neighbors"}

    # ------------------------------------------------------------------
    # Cache management
    # ------------------------------------------------------------------

    def _cache_is_stale(self) -> bool:
        """Return ``True`` when cached Ewald parameters need recomputation.

        The cache is marked stale by :meth:`invalidate_cache`.  Callers that
        modify the unit cell (e.g. NPT integrators) must call
        ``invalidate_cache()`` so that k-vectors and alpha are recomputed on
        the next forward pass.
        """
        return not self._cache_valid

    def invalidate_cache(self) -> None:
        """Force recomputation of Ewald parameters and k-vectors."""
        self._cache_valid = False
        self._cached_alpha = None
        self._cached_k_vectors = None
        # Note: _cached_cell is intentionally NOT cleared here.
        # The cell reference is used for change-detection in forward(); clearing
        # it would cause every call to look like a cell change and invalidate
        # the cache again immediately after recomputation.

    def _update_cache(
        self,
        positions: torch.Tensor,
        cell: torch.Tensor,
        batch_idx: torch.Tensor,
    ) -> None:
        """Recompute Ewald parameters and k-vectors for the given cell."""
        from nvalchemiops.torch.interactions.electrostatics.k_vectors import (  # lazy
            generate_k_vectors_ewald_summation,
        )
        from nvalchemiops.torch.interactions.electrostatics.parameters import (  # lazy
            estimate_ewald_parameters,
        )

        params = estimate_ewald_parameters(
            positions, cell, batch_idx=batch_idx, accuracy=self.accuracy
        )
        k_vectors = generate_k_vectors_ewald_summation(
            cell, params.reciprocal_space_cutoff
        )

        self._cache_valid = True
        self._cached_alpha = params.alpha
        self._cached_k_vectors = k_vectors

    # ------------------------------------------------------------------
    # Input adaptation
    # ------------------------------------------------------------------

    def adapt_input(self, data: AtomicData | Batch, **kwargs: Any) -> dict[str, Any]:
        """Collect required inputs from *data* without enabling gradients."""
        if not isinstance(data, Batch):
            raise TypeError(
                "EwaldModelWrapper requires a Batch input; "
                "got AtomicData.  Use Batch.from_data_list([data]) to wrap it."
            )

        input_dict: dict[str, Any] = {}
        for key in self.input_data():
            value = getattr(data, key, None)
            if value is None:
                raise KeyError(f"'{key}' required but not found in input data.")
            input_dict[key] = value

        # node_charges is stored as (N, 1) in AtomicData to satisfy the Pydantic
        # model shape requirements; the kernel expects shape (N,).
        charges = input_dict["node_charges"]
        if charges.dim() == 2 and charges.shape[-1] == 1:
            input_dict["node_charges"] = charges.squeeze(-1)

        input_dict["batch_idx"] = data.batch.to(torch.int32)
        input_dict["ptr"] = data.ptr.to(torch.int32)
        input_dict["num_graphs"] = data.num_graphs
        input_dict["fill_value"] = data.num_nodes

        # PBC cell (required for Ewald).
        try:
            input_dict["cell"] = data.cell  # (B, 3, 3)
        except AttributeError:
            raise ValueError(
                "EwaldModelWrapper requires periodic boundary conditions "
                "(data.cell must be present)."
            )

        # Collect neighbor tensors.
        neighbor_dict = prepare_neighbors_for_model(
            data, self.cutoff, NeighborListFormat.MATRIX, data.num_nodes
        )
        input_dict["neighbor_matrix"] = neighbor_dict["neighbor_matrix"]
        input_dict["num_neighbors"] = neighbor_dict["num_neighbors"]
        input_dict["neighbor_shifts"] = neighbor_dict.get("neighbor_shifts", None)

        return input_dict

    # ------------------------------------------------------------------
    # Output adaptation
    # ------------------------------------------------------------------

    def adapt_output(self, model_output: Any, data: AtomicData | Batch) -> ModelOutputs:
        """Adapt the model output to the framework output format."""
        output: ModelOutputs = OrderedDict()
        output["energies"] = model_output["energies"]
        if self.model_config.compute_forces:
            output["forces"] = model_output["forces"]
        if self.model_config.compute_stresses:
            if "stresses" in model_output:
                output["stresses"] = model_output["stresses"]
        return output

    def output_data(self) -> set[str]:
        """Return the set of keys that the model produces."""
        keys: set[str] = {"energies"}
        if self.model_config.compute_forces:
            keys.add("forces")
        if self.model_config.compute_stresses:
            keys.add("stresses")
        return keys

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, data: AtomicData | Batch, **kwargs: Any) -> ModelOutputs:
        """Run the Ewald summation and return a :class:`ModelOutputs` dict.

        Parameters
        ----------
        data : Batch
            Batch containing ``positions``, ``node_charges``, ``cell``,
            ``neighbor_matrix``, and ``num_neighbors`` (populated by
            :class:`~nvalchemi.dynamics.hooks.NeighborListHook`).

        Returns
        -------
        ModelOutputs
            OrderedDict with keys ``"energies"`` (shape ``[B, 1]``, eV),
            ``"forces"`` (shape ``[N, 3]``, eV/Å), and optionally
            ``"stress"`` (shape ``[B, 3, 3]``, eV — the raw virial
            :math:`W_{phys}`).
        """
        from nvalchemiops.torch.interactions.electrostatics.ewald import (  # lazy
            ewald_real_space,
            ewald_reciprocal_space,
        )

        inp = self.adapt_input(data, **kwargs)

        positions = inp["positions"]  # (N, 3)
        charges = inp["node_charges"].view(
            -1,
        )  # (N,)
        cell = inp["cell"]  # (B, 3, 3)
        batch_idx = inp["batch_idx"]  # (N,) int32
        fill_value: int = inp["fill_value"]
        B: int = inp["num_graphs"]
        neighbor_matrix = inp["neighbor_matrix"].contiguous()
        neighbor_shifts = inp.get("neighbor_shifts")

        compute_forces = self.model_config.compute_forces
        compute_stresses = self.model_config.compute_stresses

        # Automatically invalidate cache when cell changes (e.g. NPT simulation).
        if self._cached_cell is None or not torch.allclose(
            cell, self._cached_cell, rtol=1e-6, atol=1e-9
        ):
            self._cached_cell = cell.detach().clone()
            self._cache_valid = False

        # Update cached parameters if invalidated.
        if self._cache_is_stale():
            self._update_cache(positions, cell, batch_idx)

        alpha = self._cached_alpha  # (B,)
        k_vectors = self._cached_k_vectors

        # Prepare neighbor_matrix_shifts: reuse cached zero buffer for non-PBC runs.
        if neighbor_shifts is None:
            K = neighbor_matrix.shape[1]
            N = positions.shape[0]
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

        # --- Real-space contribution ---
        real_result = ewald_real_space(
            positions=positions,
            charges=charges,
            cell=cell,
            alpha=alpha,
            neighbor_matrix=neighbor_matrix,
            neighbor_matrix_shifts=neighbor_shifts.contiguous(),
            mask_value=fill_value,
            batch_idx=batch_idx,
            compute_forces=compute_forces,
            compute_virial=compute_stresses,
        )

        # --- Reciprocal-space contribution ---
        recip_result = ewald_reciprocal_space(
            positions=positions,
            charges=charges,
            cell=cell,
            k_vectors=k_vectors,
            alpha=alpha,
            batch_idx=batch_idx,
            compute_forces=compute_forces,
            compute_virial=compute_stresses,
        )

        # Unpack results (energies always first; forces and virial follow
        # in the order they were requested).
        def _unpack(result, compute_f: bool, compute_v: bool):
            """Extract (energies, forces_or_None, virial_or_None) from result."""
            if isinstance(result, torch.Tensor):
                return result, None, None
            result_list = list(result)
            e = result_list[0]
            f: torch.Tensor | None = None
            v: torch.Tensor | None = None
            idx = 1
            if compute_f and idx < len(result_list):
                f = result_list[idx]
                idx += 1
            if compute_v and idx < len(result_list):
                v = result_list[idx]
            return e, f, v

        e_real, f_real, v_real = _unpack(real_result, compute_forces, compute_stresses)
        e_recip, f_recip, v_recip = _unpack(
            recip_result, compute_forces, compute_stresses
        )

        # Sum real + reciprocal contributions.
        per_atom_energies = (e_real + e_recip).to(positions.dtype)  # (N,) float64→dtype

        forces: torch.Tensor | None = None
        if compute_forces and f_real is not None and f_recip is not None:
            forces = f_real + f_recip  # (N, 3)

        virial: torch.Tensor | None = None
        if compute_stresses and v_real is not None and v_recip is not None:
            virial = v_real + v_recip  # (B, 3, 3)

        # Scale by Coulomb constant.
        per_atom_energies = per_atom_energies * self.coulomb_constant
        if forces is not None:
            forces = forces * self.coulomb_constant
        if virial is not None:
            virial = virial * self.coulomb_constant

        # Scatter per-atom energies → per-system totals using pre-allocated buffer.
        if (
            self._energies_buf is None
            or self._energies_buf.shape[0] != B
            or self._energies_buf.dtype != positions.dtype
            or self._energies_buf.device != positions.device
        ):
            self._energies_buf = torch.empty(
                B, dtype=positions.dtype, device=positions.device
            )
        self._energies_buf.zero_()
        self._energies_buf.scatter_add_(0, batch_idx, per_atom_energies)

        # Clone from pre-allocated buffer so the caller receives an independent tensor.
        # Without cloning, the next forward pass would overwrite this tensor in-place.
        model_output: dict[str, Any] = {
            "energies": self._energies_buf.unsqueeze(-1).clone()
        }
        if forces is not None:
            model_output["forces"] = forces
        if virial is not None:
            # The ewald kernels accumulate W = Σ r_ij ⊗ F_ij (positive convention).
            # Store directly as stresses (W_phys) — the barostat divides by V.
            model_output["stresses"] = virial

        return self.adapt_output(model_output, data)

    def export_model(self, path: Path, as_state_dict: bool = False) -> None:
        """Export model is not implemented for Ewald models."""
        raise NotImplementedError
