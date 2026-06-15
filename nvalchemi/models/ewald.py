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
    from nvalchemi.hooks import NeighborListHook
    from nvalchemi.dynamics.base import DynamicsStage

    model = EwaldModelWrapper(cutoff=10.0)

    nl_hook = NeighborListHook(model.model_config.neighbor_config, stage=DynamicsStage.BEFORE_COMPUTE)
    dynamics.register_hook(nl_hook)
    dynamics.model = model

Notes
-----
* Forces are computed **analytically** inside the Warp kernel using
  ``hybrid_forces=True``.  Direct kernel forces represent ``dE/dR|_q``
  (derivative at fixed charges).  ``"forces"`` is in ``autograd_outputs``
  so that the pipeline can add the charge chain-rule term
  ``(dE/dq)(dq/dR)`` via autograd on the energy.
* Energy supports ``backward()`` through the charge pathway: when
  ``charges.requires_grad``, the kernel injects analytical ``dE/dq``
  into the energy tensor via ``_InjectChargeGrad``.
* Virial/stress is also computed analytically by the kernel and returned
  detached (no ``grad_fn``), representing ``dE/d(strain)|_q``.  In a
  pipeline with geometry-dependent charges, the total stress is the sum
  of the direct kernel virial and the autograd chain-rule term
  ``(dE/dq)(dq/d(strain))``.
* Periodic boundary conditions are **required** (``needs_pbc=True``).
* Input charges are read from ``data.charges`` (shape ``[N]``).
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
from nvalchemi.models._utils import cell_cache_needs_update
from nvalchemi.models.base import (
    BaseModelMixin,
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
    slab_correction : bool, optional
        Whether to enable the two-dimensional slab correction. Defaults to
        ``False``. When enabled, the input batch must provide ``data.pbc`` as
        a boolean tensor with shape ``(B, 3)``. Rows with exactly one
        ``False`` entry mark slab systems, for example ``[True, True, False]``
        for a non-periodic z axis. Fully periodic rows are no-ops, so mixed
        slab and three-dimensional periodic batches are supported.
    rtol : float, optional
        Relative tolerance for cell change detection.
        See :func:`~nvalchemi.models._utils.cell_cache_needs_update`.
    atol : float or None, optional
        Absolute tolerance for cell change detection.
        See :func:`~nvalchemi.models._utils.cell_cache_needs_update`.

    Attributes
    ----------
    model_config : ModelConfig
        Mutable configuration controlling which outputs are computed.
        ``model_config.autograd_outputs`` includes ``"forces"`` so the
        pipeline accumulates direct kernel forces with charge-path autograd
        forces in hybrid mode. Include ``"stress"`` in
        ``model_config.active_outputs`` to enable virial computation for
        NPT/NPH simulations.
        When ``charges.requires_grad=True``, ``energy.backward()`` propagates
        through the injected :math:`dE/dq` pathway while the wrapper returns
        detached direct kernel forces and detached virial/stress.
    """

    def __init__(
        self,
        cutoff: float,
        accuracy: float = 1e-6,
        coulomb_constant: float = 14.3996,
        hybrid_forces: bool = True,
        slab_correction: bool = False,
        rtol: float = 1e-5,
        atol: float | None = None,
    ) -> None:
        super().__init__()
        self.cutoff = cutoff
        self.accuracy = accuracy
        self.coulomb_constant = coulomb_constant
        self.hybrid_forces = hybrid_forces
        self.slab_correction = slab_correction
        self.rtol = rtol
        self.atol = atol
        self.model_config = ModelConfig(
            outputs=frozenset({"energy", "forces", "stress"}),
            active_outputs={"energy", "forces"},
            autograd_outputs=frozenset({"forces"})
            if hybrid_forces
            else frozenset({"forces", "stress"}),
            autograd_inputs=frozenset({"positions"}),
            required_inputs=frozenset({"charges"}),
            optional_inputs=frozenset(),
            supports_pbc=True,
            needs_pbc=True,
            neighbor_config=NeighborConfig(
                cutoff=self.cutoff,
                format=NeighborListFormat.MATRIX,
                half_list=False,
            ),
        )

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

    @property
    def embedding_shapes(self) -> dict[str, tuple[int, ...]]:
        return {}

    def compute_embeddings(
        self, data: AtomicData | Batch, **kwargs: Any
    ) -> AtomicData | Batch:
        """Compute embeddings is not meaningful for Ewald models."""
        raise NotImplementedError("EwaldModelWrapper does not produce embeddings.")

    def direct_derivative_keys(self) -> set[str]:
        """Analytical force/stress keys when ``hybrid_forces=True``."""
        if not self.hybrid_forces:
            return set()
        keys: set[str] = set()
        if "forces" in self.model_config.outputs:
            keys.add("forces")
        if "stress" in self.model_config.outputs:
            keys.add("stress")
        return keys

    # ------------------------------------------------------------------
    # Input / output key declarations
    # ------------------------------------------------------------------

    def input_data(self) -> set[str]:
        """Return required input keys (override to drop ``atomic_numbers``)."""
        keys = {"positions", "charges", "neighbor_matrix", "num_neighbors"}
        if self.slab_correction:
            keys.add("pbc")
        return keys

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
                if key == "pbc" and self.slab_correction:
                    raise ValueError(
                        "EwaldModelWrapper with slab_correction=True requires "
                        "periodic boundary condition flags "
                        "(data.pbc must be present)."
                    )
                raise KeyError(f"'{key}' required but not found in input data.")
            input_dict[key] = value

        input_dict["batch_idx"] = data.batch_idx.to(torch.int32)
        input_dict["ptr"] = data.batch_ptr.to(torch.int32)
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

        if self.slab_correction:
            pbc = getattr(data, "pbc", None)
            if pbc is None:
                raise ValueError(
                    "EwaldModelWrapper with slab_correction=True requires "
                    "periodic boundary condition flags (data.pbc must be present)."
                )
            input_dict["pbc"] = pbc  # (B, 3)

        # neighbor_matrix and num_neighbors are already collected by the
        # input_data() loop above.  In a pipeline, the pipeline adapts them
        # to this model's cutoff/format before calling forward().
        input_dict["neighbor_matrix_shifts"] = getattr(
            data, "neighbor_matrix_shifts", None
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
            else:
                raise RuntimeError(
                    "'stress' is in active_outputs but missing from model output"
                )
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
        """Run the Ewald summation and return a :class:`ModelOutputs` dict.

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
            ``"stress"`` (shape ``[B, 3, 3]``, eV/Å³ — Cauchy stress
            ``-W/V``).
        """
        from nvalchemiops.torch.interactions.electrostatics.ewald import (  # lazy
            ewald_summation,
        )

        inp = self.adapt_input(data, **kwargs)

        positions = inp["positions"]  # (N, 3)
        charges = inp["charges"].view(
            -1,
        )  # (N,)
        cell = inp["cell"]  # (B, 3, 3)
        batch_idx = inp["batch_idx"]  # (N,) int32
        fill_value: int = inp["fill_value"]
        B: int = inp["num_graphs"]
        neighbor_matrix = inp["neighbor_matrix"].contiguous()
        neighbor_matrix_shifts = inp.get("neighbor_matrix_shifts")
        pbc = inp.get("pbc")

        compute_forces = "forces" in self.model_config.active_outputs
        compute_stresses = "stress" in self.model_config.active_outputs

        # hybrid_forces=True: the kernel detaches positions and cell
        # internally and computes analytical forces/virial without a Warp
        # tape.  Detach here too so that nvalchemiops' backward registration
        # (_register_runtime_state) does not expect a tape when inputs have
        # requires_grad=True (e.g. from prepare_strain in a pipeline).
        if self.hybrid_forces:
            positions = positions.detach()
            cell = cell.detach()

        # Automatically invalidate cache when cell changes (e.g. NPT simulation).
        if cell_cache_needs_update(
            cell, self._cached_cell, rtol=self.rtol, atol=self.atol
        ):
            self._cached_cell = cell.detach().clone()
            self._cache_valid = False

        # Update cached parameters if invalidated.
        if self._cache_is_stale():
            self._update_cache(positions, cell, batch_idx)

        alpha = self._cached_alpha  # (B,)
        k_vectors = self._cached_k_vectors

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

        result = ewald_summation(
            positions=positions,
            charges=charges,
            cell=cell,
            alpha=alpha,
            k_vectors=k_vectors,
            neighbor_matrix=neighbor_matrix,
            neighbor_matrix_shifts=neighbor_matrix_shifts.contiguous(),
            mask_value=fill_value,
            batch_idx=batch_idx,
            compute_forces=compute_forces,
            compute_virial=compute_stresses,
            hybrid_forces=self.hybrid_forces,
            pbc=pbc,
            slab_correction=self.slab_correction,
        )

        # Unpack results (energies always first; forces and virial follow in
        # the order they were requested).
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

        per_atom_energies, forces, virial = _unpack(
            result, compute_forces, compute_stresses
        )
        per_atom_energies = per_atom_energies.to(positions.dtype)

        # Scale by Coulomb constant.
        per_atom_energies = per_atom_energies * self.coulomb_constant
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
        # Clone so callers (e.g. BiasedPotentialHook in-place add_) see
        # storage independent of the persistent buffer; detach so the next
        # zero_() starts a fresh autograd chain (#82).
        model_output: dict[str, Any] = {
            "energy": self._energies_buf.unsqueeze(-1).clone()
        }
        self._energies_buf.detach_()
        if forces is not None:
            model_output["forces"] = forces
        if virial is not None:
            # Tensile-positive Cauchy stress sigma = -W/V (eV/A^3).
            volume = torch.det(data.cell).abs().view(-1, 1, 1)
            model_output["stress"] = -virial / volume
        elif compute_stresses:
            raise RuntimeError(
                "stress was requested but the kernel did not return a virial"
            )

        return self.adapt_output(model_output, data)

    def export_model(self, path: Path, as_state_dict: bool = False) -> None:
        """Export model is not implemented for Ewald models."""
        raise NotImplementedError
