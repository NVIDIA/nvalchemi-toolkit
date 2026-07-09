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
from nvalchemi.models._utils import cell_cache_needs_update
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
        mesh_spacing: float = 1.0,
        mesh_dimensions: tuple[int, int, int] | None = None,
        spline_order: int = 4,
        alpha: float | None = None,
        accuracy: float = 1e-6,
        coulomb_constant: float = 14.3996,
        hybrid_forces: bool = True,
        slab_correction: bool = False,
        rtol: float = 1e-5,
        atol: float | None = None,
    ) -> None:
        super().__init__()
        self.cutoff = cutoff
        self.mesh_spacing = mesh_spacing
        self.mesh_dimensions = mesh_dimensions
        self.spline_order = spline_order
        self.alpha = alpha
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

        # PME k-vector / parameter cache, rebuilt when the cell changes.
        self._cache_valid: bool = False
        self._cached_alpha: torch.Tensor | None = None
        self._cached_k_vectors: torch.Tensor | None = None
        self._cached_k_squared: torch.Tensor | None = None
        self._cached_mesh_dims: tuple[int, int, int] | None = None
        # Last-seen cell, used to detect cell changes (e.g. NPT).
        self._cached_cell: torch.Tensor | None = None
        # Pre-allocated per-system energy accumulator, shape ``[B]``.
        self._energies_buf: torch.Tensor | None = None
        # Reusable all-zero neighbor shifts for non-PBC runs, ``[N, K, 3]``.
        self._null_shifts: torch.Tensor | None = None
        self._null_shifts_shape: tuple[int, int] = (0, 0)

        # Distributed context, set by distributed_setup; None on single-GPU.
        self._dist_ctx: Any = None
        # Global atom count for cache estimation: the per-rank padded count
        # differs across ranks and would give divergent alpha / mesh.
        self._n_global_atoms: int | None = None

    # ------------------------------------------------------------------
    # BaseModelMixin required properties
    # ------------------------------------------------------------------

    def distribution_spec(self, strategy: Any = None) -> Any:
        """Domain-decomposition spec for the PME wrapper.

        Halo-only; the ``strategy`` argument is accepted for the framework
        contract and ignored.

        Four ops get owned-slice + all-reduce handlers so the reciprocal-space
        pathway sees globally-correct quantities: the spline-spread ops
        all-reduce each rank's partial charge mesh into a replicated global
        mesh, and the total-charge ops all-reduce each rank's partial charge
        sum into the true global total charge used by the background
        correction. Every downstream stage (FFT, Green's function, IFFT,
        gather, per-atom corrections) then runs identically on every rank, so
        :meth:`forward` is distribution-agnostic.

        Returns
        -------
        MLIPSpec
            Halo-storage spec carrying the spline-spread and total-charge
            ``custom_ops``, plus output handling: ``energy`` and ``stress``
            per-graph, ``forces`` per-node owned-only, ``atomic_energies``
            per-node.
        """
        from nvalchemi.distributed.config import StrategyKind  # noqa: PLC0415

        if strategy == StrategyKind.GRAPH_PARTITION:
            return self._distribution_spec_gp()

        import torch  # noqa: PLC0415

        # Force op registration before grabbing the handles.
        from nvalchemiops.torch.spline import (  # noqa: F401, PLC0415
            spline_spread,
        )

        from nvalchemi.distributed._core.op_transforms import (  # noqa: PLC0415
            AllReduceSum,
            SliceOwned,
        )
        from nvalchemi.distributed.graph_padder import DenseBatchPadder  # noqa: PLC0415
        from nvalchemi.distributed.spec import (  # noqa: PLC0415
            SPEC_PME_HALO,
            CompilePolicy,
            ForceStrategy,
            MLIPSpec,
            OpAdapter,
            OutputKind,
            OutputSpec,
            Reduce,
        )
        from nvalchemi.models._ops.electrostatics.pme import (  # noqa: F401, PLC0415
            _batch_pme_compute_partial_total_charge,
            _pme_compute_partial_total_charge,
        )
        from nvalchemi.models._ops.electrostatics.slab import (  # noqa: F401, PLC0415
            _batch_slab_compute_partial_moments,
            _slab_compute_partial_moments,
        )

        # Each adapter slices its per-atom inputs to owned and all-reduces the
        # partial output (charge mesh / total charge) to global.
        nvops = torch.ops.nvalchemiops  # type: ignore[attr-defined]
        ops = torch.ops.alchemiops  # type: ignore[attr-defined]
        custom_ops = (
            OpAdapter(
                op=nvops.spline_spread,  # (positions, values, ...)
                arg_transforms={0: SliceOwned(), 1: SliceOwned()},
                output_transforms={0: AllReduceSum()},
            ),
            OpAdapter(
                op=nvops.batch_spline_spread,  # (positions, values, batch_idx, ...)
                arg_transforms={
                    0: SliceOwned(),
                    1: SliceOwned(),
                    2: SliceOwned(),
                },
                output_transforms={0: AllReduceSum()},
            ),
            OpAdapter(
                op=ops._pme_compute_partial_total_charge,  # (charges)
                arg_transforms={0: SliceOwned()},
                output_transforms={0: AllReduceSum()},
            ),
            OpAdapter(
                op=ops._batch_pme_compute_partial_total_charge,  # (charges, batch_idx)
                arg_transforms={0: SliceOwned(), 1: SliceOwned()},
                output_transforms={0: AllReduceSum()},
            ),
            # Slab-correction moments: each rank's partial (M, M2, Q) is summed
            # over its owned atoms then all-reduced into the true global moments.
            OpAdapter(
                op=ops._slab_compute_partial_moments,  # (z, charges)
                arg_transforms={0: SliceOwned(), 1: SliceOwned()},
                output_transforms={
                    0: AllReduceSum(),  # mz
                    1: AllReduceSum(),  # mz2
                    2: AllReduceSum(),  # qtotal
                },
            ),
            OpAdapter(
                op=ops._batch_slab_compute_partial_moments,  # (z, charges, batch_idx)
                arg_transforms={0: SliceOwned(), 1: SliceOwned(), 2: SliceOwned()},
                output_transforms={
                    0: AllReduceSum(),
                    1: AllReduceSum(),
                    2: AllReduceSum(),
                },
            ),
        )
        import dataclasses  # noqa: PLC0415

        # Compiled DD needs differentiable forces, so it is enabled only when
        # hybrid_forces=False: the framework then derives forces via autograd
        # over the global energy.
        compile_policy = (
            None
            if self.hybrid_forces
            else CompilePolicy(
                static_shapes=True,
                force_strategy=ForceStrategy.FRAMEWORK_FROM_NODE_ENERGY,
                graph_padder=DenseBatchPadder(),
                # The framework strains positions + cell and takes the virial by
                # autograd of the consolidated global energy. PME's reciprocal is
                # fully differentiable in the cell (k-vectors + volume + FFT), so
                # this is correct (requires the nvalchemiops convolve-backward
                # grad_k_squared rank fix).
                stress_via_strain=True,
            )
        )
        # Eager kernel forces are complete per owned atom, so slice off the halo
        # duplicates. (The compiled path instead derives forces by autograd over
        # the global energy.)
        forces_spec = OutputSpec(OutputKind.PER_NODE, Reduce.OWNED_ONLY)
        # Emit raw per-atom energies; the framework reduces them owned-aware into
        # the per-system energy, keeping forward distribution-agnostic.
        return MLIPSpec(
            distribution=dataclasses.replace(
                SPEC_PME_HALO.distribution, custom_ops=custom_ops
            ),
            outputs={
                "energy": OutputSpec(OutputKind.PER_GRAPH),
                "forces": forces_spec,
                # Compiled DD (hybrid_forces=False) derives stress via the framework
                # strain-autograd of the consolidated GLOBAL energy: a per-rank
                # virial that ALL_REDUCE sums across ranks (correct for real +
                # reciprocal, since PME's reciprocal is differentiable in the cell).
                # The eager analytic kernel virial (hybrid_forces=True) needs a
                # reciprocal-aware split, so it stays unreduced.
                "stress": OutputSpec(OutputKind.PER_GRAPH, Reduce.ALL_REDUCE)
                if not self.hybrid_forces
                else OutputSpec(OutputKind.PER_GRAPH),
                "atomic_energies": OutputSpec(OutputKind.PER_NODE),
            },
            node_energy_key="atomic_energies",
            compile=compile_policy,
        )

    def _distribution_spec_gp(self) -> Any:
        """PME under node-partition graph-parallel.

        Rides the framework's ``gp_replicate_geometry`` path: the full geometry is
        replicated on every rank so PME's fused real-space+reciprocal kernel indexes
        global senders and spreads the **full charge set** (correct reciprocal),
        while the dense ``neighbor_matrix`` is masked to this rank's owned receivers
        (partitioned real-space). The framework reduces the owned per-node energy
        (``node_energy_key="atomic_energies"``) and derives forces by autograd over
        the full-position leaf.

        Because every rank sees all charges, the reciprocal is globally correct with
        **no** halo reciprocal OpAdapters (the owned-slice + all-reduce spline/total-
        charge handlers are a halo-storage concern). Requires ``hybrid_forces=False``
        so the energy is differentiable for the framework force autograd.
        """
        if self.hybrid_forces:
            raise NotImplementedError(
                "PME graph-parallel requires hybrid_forces=False (a differentiable "
                "energy for the framework force autograd); construct the wrapper "
                "with hybrid_forces=False for the GRAPH_PARTITION strategy."
            )
        import dataclasses  # noqa: PLC0415

        from nvalchemi.distributed.spec import (  # noqa: PLC0415
            SPEC_MPNN_GP,
            CompilePolicy,
            ForceStrategy,
            OutputKind,
            OutputSpec,
            Reduce,
        )

        return dataclasses.replace(
            SPEC_MPNN_GP,
            outputs={
                "energy": OutputSpec(OutputKind.PER_GRAPH),
                "forces": OutputSpec(OutputKind.PER_NODE, Reduce.OWNED_ONLY),
                "atomic_energies": OutputSpec(OutputKind.PER_NODE, Reduce.OWNED_ONLY),
            },
            node_energy_key="atomic_energies",
            gp_replicate_geometry=True,
            compile=CompilePolicy(
                force_strategy=ForceStrategy.FRAMEWORK_FROM_NODE_ENERGY
            ),
        )

    def distributed_setup(self, ctx: Any) -> None:
        """Enter distributed mode for this wrapper.

        Records the distributed context and global atom count, then
        invalidates the cache so ``alpha`` / mesh are re-estimated from the
        global ``N`` rather than a stale per-rank count.

        Parameters
        ----------
        ctx : DistributedContext
            The live distributed context, exposing ``n_atoms_total`` and the
            halo metadata.

        Returns
        -------
        None
        """
        self._dist_ctx = ctx
        self._n_global_atoms = ctx.n_atoms_total
        # Cached alpha / mesh derived from a stale N must be rebuilt.
        self.invalidate_cache()

    def distributed_teardown(self) -> None:
        """Leave distributed mode and return to single-GPU behaviour.

        Clears the distributed context and global atom count and invalidates
        the cache.

        Returns
        -------
        None
        """
        self._dist_ctx = None
        self._n_global_atoms = None
        self.invalidate_cache()

    @property
    def embedding_shapes(self) -> dict[str, tuple[int, ...]]:
        return {}

    def compute_embeddings(
        self, data: AtomicData | Batch, **kwargs: Any
    ) -> AtomicData | Batch:
        """Embeddings are not defined for a PME electrostatics model.

        Parameters
        ----------
        data : AtomicData | Batch
            The input system (unused).
        **kwargs
            Unused; accepted for interface compatibility.

        Returns
        -------
        AtomicData | Batch
            Never returned.

        Raises
        ------
        NotImplementedError
            Always; PME produces no learned embeddings.
        """
        raise NotImplementedError("PMEModelWrapper does not produce embeddings.")

    def direct_derivative_keys(self) -> set[str]:
        """Report which outputs are computed analytically by the kernel.

        Returns
        -------
        set[str]
            ``{"forces", "stress"}`` (intersected with the active outputs)
            when ``hybrid_forces=True``; an empty set otherwise, in which
            case forces/stress come from autograd on the energy.
        """
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
        """List the batch attributes the PME forward reads.

        Returns
        -------
        set[str]
            ``{"positions", "charges", "neighbor_matrix", "num_neighbors"}``,
            plus ``"pbc"`` when ``slab_correction=True``.
            Notably excludes ``atomic_numbers``, which PME does not use.
        """
        keys = {"positions", "charges", "neighbor_matrix", "num_neighbors"}
        if self.slab_correction:
            keys.add("pbc")
        return keys

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
        """Recompute PME parameters and k-vectors for the given cell.

        ``alpha`` and the FFT mesh are estimated from the global atom count.
        When running distributed, a shape-only surrogate of size
        ``self._n_global_atoms`` is fed to ``estimate_pme_parameters`` so
        every rank agrees on ``alpha`` / mesh despite holding a different
        per-rank padded count; otherwise the local ``positions`` are used.
        """
        from nvalchemiops.torch.interactions.electrostatics.k_vectors import (  # lazy
            generate_k_vectors_pme,
        )
        from nvalchemiops.torch.interactions.electrostatics.parameters import (  # lazy
            estimate_pme_parameters,
        )

        B = cell.shape[0] if cell.dim() == 3 else 1

        if self._n_global_atoms is not None:
            est_positions = positions[:1].expand(self._n_global_atoms, -1).contiguous()
            est_batch_idx = batch_idx.new_zeros(self._n_global_atoms)
        else:
            est_positions = positions
            est_batch_idx = batch_idx

        # Determine alpha and mesh_dimensions.
        need_params = (self.alpha is None) or (self.mesh_dimensions is None)
        params = None
        if need_params:
            params = estimate_pme_parameters(
                est_positions, cell, batch_idx=est_batch_idx, accuracy=self.accuracy
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
        """Collect the kernel inputs from *data* without enabling gradients.

        Gathers the required batch attributes, batch indexing tensors, the
        PBC cell, and optional neighbor shifts into a plain dict. Gradients
        are not enabled here: forces and stress are produced analytically by
        the kernel (or, in the charge-dependent pipeline, via autograd on the
        energy).

        Parameters
        ----------
        data : Batch
            Batch with ``positions``, ``charges``, ``cell``,
            ``neighbor_matrix``, and ``num_neighbors``.
        **kwargs
            Unused; accepted for interface compatibility.

        Returns
        -------
        dict[str, Any]
            Kernel inputs including ``positions`` ``[N, 3]``, ``charges``
            ``[N]``, ``cell`` ``[B, 3, 3]``, ``batch_idx`` ``[N]``, ``ptr``,
            ``num_graphs``, ``fill_value``, the neighbor matrix, and
            ``neighbor_matrix_shifts`` (``None`` when non-periodic).

        Raises
        ------
        TypeError
            If *data* is an ``AtomicData`` rather than a ``Batch``.
        KeyError
            If a required input key is missing from *data*.
        ValueError
            If *data* has no ``cell`` (PME requires PBC).
        """
        if not isinstance(data, Batch):
            raise TypeError(
                "PMEModelWrapper requires a Batch input; "
                "got AtomicData.  Use Batch.from_data_list([data]) to wrap it."
            )

        input_dict: dict[str, Any] = {}
        for key in self.input_data():
            value = getattr(data, key, None)
            if value is None:
                if key == "pbc" and self.slab_correction:
                    raise ValueError(
                        "PMEModelWrapper with slab_correction=True requires periodic "
                        "boundary condition flags (data.pbc must be present)."
                    )
                raise KeyError(f"'{key}' required but not found in input data.")
            input_dict[key] = value

        input_dict["batch_idx"] = data.batch_idx.to(torch.int32)
        input_dict["ptr"] = data.batch_ptr.to(torch.int32)
        input_dict["num_graphs"] = data.num_graphs
        input_dict["fill_value"] = data.num_nodes

        # PBC cell (required for PME).
        try:
            input_dict["cell"] = data.cell  # [B, 3, 3]
        except AttributeError:
            raise ValueError(
                "PMEModelWrapper requires periodic boundary conditions "
                "(data.cell must be present)."
            )

        if self.slab_correction:
            pbc = getattr(data, "pbc", None)
            if pbc is None:
                raise ValueError(
                    "PMEModelWrapper with slab_correction=True requires periodic "
                    "boundary condition flags (data.pbc must be present)."
                )
            input_dict["pbc"] = pbc  # (B, 3)

        # Neighbor data is collected by the input_data() loop above; the
        # pipeline adapts it to this model's cutoff/format before forward().
        input_dict["neighbor_matrix_shifts"] = getattr(
            data, "neighbor_matrix_shifts", None
        )

        return input_dict

    # ------------------------------------------------------------------
    # Output adaptation
    # ------------------------------------------------------------------

    def adapt_output(self, model_output: Any, data: AtomicData | Batch) -> ModelOutputs:
        """Select the active outputs into the standard output mapping.

        Always forwards ``energy``; adds ``forces`` and ``stress`` when each
        is in ``model_config.active_outputs``.

        Parameters
        ----------
        model_output : dict[str, Any]
            Raw kernel outputs keyed by ``"energy"``, ``"forces"``, and
            ``"stress"``.
        data : AtomicData | Batch
            The input system the outputs were computed for (unused).

        Returns
        -------
        ModelOutputs
            OrderedDict with ``"energy"`` and any active ``"forces"`` /
            ``"stress"``.

        Raises
        ------
        RuntimeError
            If ``"stress"`` is active but absent from *model_output*.
        """
        output: ModelOutputs = OrderedDict()
        if "energy" in model_output:
            output["energy"] = model_output["energy"]
        if "forces" in self.model_config.active_outputs:
            output["forces"] = model_output["forces"]
        if (
            "atomic_energies" in self.model_config.active_outputs
            and "atomic_energies" in model_output
        ):
            output["atomic_energies"] = model_output["atomic_energies"]
        if "stress" in self.model_config.active_outputs:
            if "stress" in model_output:
                output["stress"] = model_output["stress"]
            else:
                raise RuntimeError(
                    "'stress' is in active_outputs but missing from model output"
                )
        return output

    def output_data(self) -> set[str]:
        """List the output keys the forward currently produces.

        Returns
        -------
        set[str]
            ``{"energy"}`` plus ``"forces"`` and/or ``"stress"`` when each is
            in ``model_config.active_outputs``.
        """
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
            ``"stress"`` (shape ``[B, 3, 3]``, eV/Å³ — Cauchy stress
            ``-W/V``).
        """
        from nvalchemi.models._ops.electrostatics.pme import (  # lazy, PLC0415
            particle_mesh_ewald_from_total_charge,
            pme_compute_partial_total_charge,
        )

        inp = self.adapt_input(data, **kwargs)

        positions = inp["positions"]  # [N, 3]
        charges = inp["charges"]  # [N]
        cell = inp["cell"]  # [B, 3, 3]
        batch_idx = inp["batch_idx"]  # [N] int32
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

        # Warn when one mean alpha spans heterogeneous batch cell volumes.
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

        if self._cache_is_stale():
            self._update_cache(positions, cell, batch_idx)

        # Non-PBC runs have no shifts; reuse a cached zero buffer.
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

        flat_charges = charges.view(-1)
        total_charges = pme_compute_partial_total_charge(
            flat_charges, batch_idx=batch_idx, num_systems=B
        )

        result = particle_mesh_ewald_from_total_charge(
            positions=positions,
            charges=flat_charges,
            cell=cell,
            total_charges=total_charges,
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
            hybrid_forces=self.hybrid_forces,
            pbc=pbc,
            slab_correction=self.slab_correction,
        )

        # Unpack tuple: (energies, [forces], [virial]).
        def _unpack(res, compute_f: bool, compute_v: bool):
            """Split the kernel result into (energies, forces, virial)."""
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

        # Scale by the Coulomb constant.
        per_atom_energies = (
            per_atom_energies.to(positions.dtype) * self.coulomb_constant
        )
        if forces is not None:
            forces = forces * self.coulomb_constant
        if virial is not None:
            virial = virial * self.coulomb_constant

        per_atom_energies = per_atom_energies.to(torch.float64)
        model_output: dict[str, Any] = {}
        if "energy" in self.model_config.active_outputs:
            # Per-atom energies -> per-system totals; accumulate in fp64 so the
            # total is order-independent. This plain inline sum is correct on a
            # single GPU; under decomposition the framework overrides energy with
            # an owned-aware sum of atomic_energies.
            model_output["energy"] = (
                torch.zeros(B, dtype=torch.float64, device=positions.device)
                .scatter_add_(0, batch_idx.to(torch.long), per_atom_energies)
                .to(positions.dtype)
                .unsqueeze(-1)
            )
        if "atomic_energies" in self.model_config.active_outputs:
            model_output["atomic_energies"] = per_atom_energies
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
        """Serialize the model (not supported for the PME wrapper).

        Parameters
        ----------
        path : Path
            Intended output path (unused).
        as_state_dict : bool, optional
            Whether to save only the ``state_dict`` (unused). Defaults to
            ``False``.

        Returns
        -------
        None

        Raises
        ------
        NotImplementedError
            Always; the PME wrapper holds no trainable weights to export.
        """
        raise NotImplementedError
