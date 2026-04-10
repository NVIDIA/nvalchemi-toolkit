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
:class:`~nvalchemi.models.base.BaseModelMixin`-compatible model, ready to
drop into any :class:`~nvalchemi.dynamics.base.BaseDynamics` engine.

Usage
-----
::

    from nvalchemi.models.pme import PMEModelWrapper
    from nvalchemi.hooks import NeighborListHook
    from nvalchemi.dynamics.base import DynamicsStage

    model = PMEModelWrapper(cutoff=10.0)

    nl_hook = NeighborListHook(model.model_card.neighbor_config, stage=DynamicsStage.BEFORE_COMPUTE)
    dynamics.register_hook(nl_hook)
    dynamics.model = model

Notes
-----
* Forces are computed **analytically** inside the Warp kernel (not via
  autograd), so ``"forces"`` is NOT in ``autograd_outputs``.
* Periodic boundary conditions are **required** (``needs_pbc=True``).
* Input charges are read from ``data.charges`` (shape ``[N]``).
* The Coulomb constant defaults to ``14.3996`` eV·Å/e², which gives energies
  in eV when positions are in Å and charges are in elementary charge units.
* PME achieves :math:`O(N \\log N)` scaling via FFT-based reciprocal space
  calculations, making it more efficient than Ewald for large systems.
* Mesh k-vectors and Ewald parameters are cached per unique unit cell.  Call
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
    ModelConfig,
    NeighborConfig,
    NeighborListFormat,
)

__all__ = ["PMEModelWrapper"]


class PMEModelWrapper(nn.Module, BaseModelMixin):
    """Particle Mesh Ewald electrostatics potential as a model wrapper.

    Computes long-range Coulomb interactions via the PME method using
    B-spline charge interpolation and FFT-based reciprocal space evaluation,
    achieving :math:`O(N \\log N)` computational scaling.

    Parameters
    ----------
    cutoff : float
        Real-space interaction cutoff in Å.
    mesh_spacing : float, optional
        Target mesh spacing in Å used to determine mesh dimensions when
        ``mesh_dimensions`` is not provided.  Defaults to ``1.0`` Å.
    mesh_dimensions : tuple[int, int, int] or None, optional
        Explicit mesh dimensions ``(nx, ny, nz)``.  When set, overrides
        automatic estimation from ``mesh_spacing``.  Defaults to ``None``
        (auto-estimated).
    spline_order : int, optional
        B-spline interpolation order.  Higher values give greater accuracy
        at increased cost.  Defaults to ``4``.
    alpha : float or None, optional
        Ewald splitting parameter (inverse Å).  ``None`` causes automatic
        estimation via the Kolafa-Perram formula each time the cell changes.
        Defaults to ``None``.
    accuracy : float, optional
        Target accuracy for automatic parameter estimation.  Defaults to
        ``1e-6``.
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
        Include ``"stress"`` in ``model_config.active_outputs`` to enable
        virial computation for NPT/NPH simulations.
    """

    def __init__(
        self,
        cutoff: float,
        mesh_spacing: float = 1.0,
        mesh_dimensions: tuple[int, int, int] | None = None,
        spline_order: int = 4,
        alpha: float | None = None,
        accuracy: float = 1e-6,
        coulomb_constant: float = 14.3996,
        max_neighbors: int | None = None,
    ) -> None:
        super().__init__()
        self.cutoff = cutoff
        self.mesh_spacing = mesh_spacing
        self.mesh_dimensions = mesh_dimensions
        self.spline_order = spline_order
        self.alpha = alpha
        self.accuracy = accuracy
        self.coulomb_constant = coulomb_constant
        self.max_neighbors = max_neighbors
        self.model_config = ModelConfig(
            outputs=frozenset({"energy", "forces", "stress"}),
            autograd_outputs=frozenset(),
            autograd_inputs=frozenset({"positions"}),
            required_inputs=frozenset({"charges"}),
            optional_inputs=frozenset(),
            supports_pbc=True,
            needs_pbc=True,
            neighbor_config=NeighborConfig(
                cutoff=self.cutoff,
                format=NeighborListFormat.MATRIX,
                half_list=False,
                max_neighbors=self.max_neighbors,
            ),
        )

        # PME k-vector / parameter cache.
        # Automatically invalidated when cell changes, or manually via invalidate_cache().
        self._cache_valid: bool = False
        self._cached_alpha: torch.Tensor | None = None
        self._cached_k_vectors: torch.Tensor | None = None
        self._cached_k_squared: torch.Tensor | None = None
        self._cached_mesh_dims: tuple[int, int, int] | None = None
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

    @property
    def embedding_shapes(self) -> dict[str, tuple[int, ...]]:
        return {}

    def compute_embeddings(
        self, data: AtomicData | Batch, **kwargs: Any
    ) -> AtomicData | Batch:
        raise NotImplementedError("PMEModelWrapper does not produce embeddings.")

    # ------------------------------------------------------------------
    # Input / output key declarations
    # ------------------------------------------------------------------

    def input_data(self) -> set[str]:
        """Return required input keys (override to drop ``atomic_numbers``)."""
        return {"positions", "charges", "neighbor_matrix", "num_neighbors"}

    # ------------------------------------------------------------------
    # Cache management
    # ------------------------------------------------------------------

    def _cache_is_stale(self) -> bool:
        """Return ``True`` when cached PME parameters need recomputation."""
        return not self._cache_valid

    def invalidate_cache(self) -> None:
        """Force recomputation of PME parameters, k-vectors, and mesh."""
        self._cache_valid = False
        self._cached_alpha = None
        self._cached_k_vectors = None
        self._cached_k_squared = None
        self._cached_mesh_dims = None

    def _update_cache(
        self,
        positions: torch.Tensor,
        cell: torch.Tensor,
        batch_idx: torch.Tensor,
    ) -> None:
        """Recompute PME parameters and k-vectors for the given cell."""
        from nvalchemiops.torch.interactions.electrostatics.k_vectors import (  # lazy
            generate_k_vectors_pme,
        )
        from nvalchemiops.torch.interactions.electrostatics.parameters import (  # lazy
            estimate_pme_parameters,
        )

        B = cell.shape[0] if cell.dim() == 3 else 1

        # Determine alpha and mesh_dimensions.
        need_params = (self.alpha is None) or (self.mesh_dimensions is None)
        params = None
        if need_params:
            params = estimate_pme_parameters(
                positions, cell, batch_idx=batch_idx, accuracy=self.accuracy
            )

        if self.alpha is not None:
            alpha_val = float(self.alpha)
        else:
            # Take the mean across batch systems.
            alpha_val = float(params.alpha.mean().item())

        if self.mesh_dimensions is not None:
            dims = self.mesh_dimensions
        else:
            dims = params.mesh_dimensions

        # Build per-system alpha tensor.
        alpha_tensor = torch.full((B,), alpha_val, dtype=cell.dtype, device=cell.device)

        k_vectors, k_squared = generate_k_vectors_pme(cell, dims)

        self._cache_valid = True
        self._cached_alpha = alpha_tensor
        self._cached_k_vectors = k_vectors
        self._cached_k_squared = k_squared
        self._cached_mesh_dims = dims

    # ------------------------------------------------------------------
    # Input adaptation
    # ------------------------------------------------------------------

    def adapt_input(self, data: AtomicData | Batch, **kwargs: Any) -> dict[str, Any]:
        """Collect required inputs from *data* without enabling gradients."""
        if not isinstance(data, Batch):
            raise TypeError(
                "PMEModelWrapper requires a Batch input; "
                "got AtomicData.  Use Batch.from_data_list([data]) to wrap it."
            )

        input_dict: dict[str, Any] = {}
        for key in self.input_data():
            value = getattr(data, key, None)
            if value is None:
                raise KeyError(f"'{key}' required but not found in input data.")
            input_dict[key] = value

        # charges is stored as (N, 1) in AtomicData to satisfy the Pydantic
        # model shape requirements; the kernel expects shape (N,).
        charges = input_dict["charges"]
        if charges.dim() == 2 and charges.shape[-1] == 1:
            input_dict["charges"] = charges.squeeze(-1)

        input_dict["batch_idx"] = data.batch_idx.to(torch.int32)
        input_dict["ptr"] = data.batch_ptr.to(torch.int32)
        input_dict["num_graphs"] = data.num_graphs
        input_dict["fill_value"] = data.num_nodes

        # PBC cell (required for PME).
        try:
            input_dict["cell"] = data.cell  # (B, 3, 3)
        except AttributeError:
            raise ValueError(
                "PMEModelWrapper requires periodic boundary conditions "
                "(data.cell must be present)."
            )

        # Collect neighbor tensors.
        neighbor_dict = prepare_neighbors_for_model(
            data, self.cutoff, NeighborListFormat.MATRIX, data.num_nodes
        )
        input_dict["neighbor_matrix"] = neighbor_dict["neighbor_matrix"]
        input_dict["num_neighbors"] = neighbor_dict["num_neighbors"]
        input_dict["neighbor_matrix_shifts"] = neighbor_dict.get(
            "neighbor_matrix_shifts", None
        )

        return input_dict

    # ------------------------------------------------------------------
    # Output adaptation
    # ------------------------------------------------------------------

    def adapt_output(self, model_output: Any, data: AtomicData | Batch) -> ModelOutputs:
        """Adapt the model output to the framework output format."""
        output: ModelOutputs = OrderedDict()
        output["energy"] = model_output["energy"]
        if "forces" in self.model_config.active_outputs:
            output["forces"] = model_output["forces"]
        if "stress" in self.model_config.active_outputs:
            if "stress" in model_output:
                output["stress"] = model_output["stress"]
        return output

    def output_data(self) -> set[str]:
        """Return the set of keys that the model produces."""
        keys: set[str] = {"energy"}
        if "forces" in self.model_config.active_outputs:
            keys.add("forces")
        if "stress" in self.model_config.active_outputs:
            keys.add("stress")
        return keys

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, data: AtomicData | Batch, **kwargs: Any) -> ModelOutputs:
        """Run the PME kernel and return a :class:`ModelOutputs` dict.

        Parameters
        ----------
        data : Batch
            Batch containing ``positions``, ``charges``, ``cell``,
            ``neighbor_matrix``, and ``num_neighbors`` (populated by
            :class:`~nvalchemi.hooks.NeighborListHook`).

        Returns
        -------
        ModelOutputs
            OrderedDict with keys ``"energy"`` (shape ``[B, 1]``, eV),
            ``"forces"`` (shape ``[N, 3]``, eV/Å), and optionally
            ``"stress"`` (shape ``[B, 3, 3]``, eV — the raw virial
            :math:`W_{phys}`).
        """
        from nvalchemiops.torch.interactions.electrostatics.pme import (  # lazy
            particle_mesh_ewald,
        )

        inp = self.adapt_input(data, **kwargs)

        positions = inp["positions"]  # (N, 3)
        charges = inp["charges"]  # (N,)
        cell = inp["cell"]  # (B, 3, 3)
        batch_idx = inp["batch_idx"]  # (N,) int32
        fill_value: int = inp["fill_value"]
        B: int = inp["num_graphs"]
        neighbor_matrix = inp["neighbor_matrix"].contiguous()
        neighbor_matrix_shifts = inp.get("neighbor_matrix_shifts")

        compute_forces = "forces" in self.model_config.active_outputs
        compute_stresses = "stress" in self.model_config.active_outputs

        # Automatically invalidate cache when cell changes (e.g. NPT simulation).
        if self._cached_cell is None or not torch.allclose(
            cell, self._cached_cell, rtol=1e-6, atol=1e-9
        ):
            self._cached_cell = cell.detach().clone()
            self._cache_valid = False

        # Warn when using a single mean α for heterogeneous batch cell volumes.
        if self.alpha is None and data.num_graphs > 1:
            vols = torch.linalg.det(cell).abs()
            if vols.min() > 0 and (vols.max() / vols.min()) > 1.1:
                import warnings as _warnings

                _warnings.warn(
                    "PMEModelWrapper: using a single mean α for a batch of systems with "
                    "heterogeneous cell volumes (max/min volume ratio > 1.1). "
                    "This may introduce systematic errors in the Ewald real/reciprocal "
                    "balance. For accurate results, use a homogeneous batch.",
                    UserWarning,
                    stacklevel=2,
                )

        # Update cache if invalidated.
        if self._cache_is_stale():
            self._update_cache(positions, cell, batch_idx)

        # Prepare neighbor_matrix_shifts: reuse cached zero buffer for non-PBC runs.
        if neighbor_matrix_shifts is None:
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
            neighbor_matrix_shifts = self._null_shifts

        result = particle_mesh_ewald(
            positions=positions,
            charges=charges.view(
                -1,
            ),
            cell=cell,
            alpha=self._cached_alpha,
            mesh_dimensions=self._cached_mesh_dims,
            spline_order=self.spline_order,
            batch_idx=batch_idx,
            k_vectors=self._cached_k_vectors,
            k_squared=self._cached_k_squared,
            neighbor_matrix=neighbor_matrix,
            neighbor_matrix_shifts=neighbor_matrix_shifts.contiguous(),
            mask_value=fill_value,
            compute_forces=compute_forces,
            compute_virial=compute_stresses,
            accuracy=self.accuracy,
        )

        # Unpack tuple: (energies, [forces], [virial]).
        def _unpack(res, compute_f: bool, compute_v: bool):
            """Extract (per_atom_energies, forces_or_None, virial_or_None)."""
            if isinstance(res, torch.Tensor):
                return res, None, None
            res_list = list(res)
            e = res_list[0]
            f: torch.Tensor | None = None
            v: torch.Tensor | None = None
            idx = 1
            if compute_f and idx < len(res_list):
                f = res_list[idx]
                idx += 1
            if compute_v and idx < len(res_list):
                v = res_list[idx]
            return e, f, v

        per_atom_energies, forces, virial = _unpack(
            result, compute_forces, compute_stresses
        )

        # Scale by Coulomb constant.
        per_atom_energies = (
            per_atom_energies.to(positions.dtype) * self.coulomb_constant
        )
        if forces is not None:
            forces = forces * self.coulomb_constant
        if virial is not None:
            virial = virial * self.coulomb_constant

        # Scatter per-atom energies -> per-system totals using pre-allocated buffer.
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
            "energy": self._energies_buf.unsqueeze(-1).clone()
        }
        if forces is not None:
            model_output["forces"] = forces
        if virial is not None:
            # particle_mesh_ewald accumulates W = Σ r_ij ⊗ F_ij (positive convention).
            # Store directly as stresses (W_phys) — the barostat divides by V.
            model_output["stress"] = virial

        return self.adapt_output(model_output, data)

    def export_model(self, path: Path, as_state_dict: bool = False) -> None:
        """Export model is not implemented for PME models."""
        raise NotImplementedError
