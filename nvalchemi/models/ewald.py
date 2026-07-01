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
* With ``hybrid_forces=True`` the Warp kernel computes forces analytically as
  ``dE/dR`` at fixed charges. ``"forces"`` is in ``autograd_outputs`` so the
  pipeline can add the charge chain-rule term ``(dE/dq)(dq/dR)`` via autograd.
* When ``charges.requires_grad``, ``energy.backward()`` flows through the charge
  pathway: the kernel injects analytical ``dE/dq`` into the energy.
* Virial/stress is computed analytically and returned detached. In a pipeline
  with geometry-dependent charges, total stress adds the chain-rule term
  ``(dE/dq)(dq/d(strain))``.
* Periodic boundary conditions are required (``needs_pbc=True``).
* Charges are read from ``data.charges`` (shape ``[N]``).
* The Coulomb constant defaults to ``14.3996`` eV·Å/e², giving energies in eV
  for positions in Å and charges in elementary charge units.
* k-vectors and Ewald parameters are cached per unit cell; call
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

        .. note::
           Under domain decomposition the Ewald wrapper runs a split
           real-space / reciprocal-space path; the slab correction is only
           wired through the single-call kernel and is therefore inactive on
           the distributed reciprocal path.
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
        ``autograd_outputs`` includes ``"forces"`` so the pipeline accumulates
        direct kernel forces with charge-path autograd forces in hybrid mode.
        Add ``"stress"`` to ``active_outputs`` to enable virial computation for
        NPT/NPH. When ``charges.requires_grad=True``, ``energy.backward()``
        flows through the injected :math:`dE/dq` pathway while forces and
        virial/stress are returned detached.
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

        # k-vector / parameter cache, invalidated on cell change or manually.
        self._cache_valid: bool = False
        self._cached_alpha: torch.Tensor | None = None
        self._cached_k_vectors: torch.Tensor | None = None
        # Cached cell for automatic invalidation detection (e.g. NPT).
        self._cached_cell: torch.Tensor | None = None
        self._energies_buf: torch.Tensor | None = None
        # Cached all-zero neighbor-shifts for non-PBC runs.
        self._null_shifts: torch.Tensor | None = None
        self._null_shifts_shape: tuple[int, int] = (0, 0)

        # Distributed context and global atom count; both None on single-GPU.
        self._dist_ctx: Any = None
        self._n_global_atoms: int | None = None

    # ------------------------------------------------------------------
    # BaseModelMixin required properties
    # ------------------------------------------------------------------

    def distribution_spec(self, strategy: Any = None) -> Any:
        """MLIPSpec for Ewald electrostatics under domain decomposition.

        Halo-only; the ``strategy`` argument is accepted for the framework
        contract and ignored.

        The real-space pair kernel runs on halo-padded inputs. Reciprocal-space
        is declared via the two structure-factor ops (single-system + batched)
        in ``custom_ops``: each takes owned-slice inputs (every atom contributes
        once globally) and all-reduces its three partial outputs across ranks
        before the energy stage consumes them.

        Returns
        -------
        MLIPSpec
            The Ewald halo spec with structure-factor ``custom_ops`` and
            per-output reduction kinds. ``forces`` is owned-only because
            reciprocal forces come from the all-reduced ``S(k)`` and so are
            exact on every owned row.
        """
        import torch  # noqa: PLC0415

        from nvalchemi.distributed._core.op_transforms import (  # noqa: PLC0415
            AllReduceSum,
            SliceOwned,
        )
        from nvalchemi.distributed.graph_padder import DenseBatchPadder  # noqa: PLC0415
        from nvalchemi.distributed.spec import (  # noqa: PLC0415
            SPEC_EWALD_HALO,
            CompilePolicy,
            ForceStrategy,
            MLIPSpec,
            OpAdapter,
            OutputKind,
            OutputSpec,
            Reduce,
        )

        # Import to register the ops before grabbing the handles below.
        from nvalchemi.models._ops.electrostatics.ewald import (  # noqa: F401, PLC0415
            ewald_compute_partial_structure_factors,
        )
        from nvalchemi.models._ops.electrostatics.slab import (  # noqa: F401, PLC0415
            _batch_slab_compute_partial_moments,
            _slab_compute_partial_moments,
        )

        ops = torch.ops.alchemiops  # type: ignore[attr-defined]
        custom_ops = (
            OpAdapter(
                op=ops._ewald_compute_partial_structure_factors,
                arg_transforms={0: SliceOwned(), 1: SliceOwned()},  # positions, charges
                output_transforms={
                    0: AllReduceSum(),  # real_sf
                    1: AllReduceSum(),  # imag_sf
                    2: AllReduceSum(),  # total_charge
                },
            ),
            OpAdapter(
                op=ops._batch_ewald_compute_partial_structure_factors,
                arg_transforms={
                    0: SliceOwned(),  # positions
                    1: SliceOwned(),  # charges
                    5: SliceOwned(),  # batch_idx
                },
                output_transforms={
                    0: AllReduceSum(),
                    1: AllReduceSum(),
                    2: AllReduceSum(),
                },
            ),
            # Slab-correction moments: owned-slice partial (M, M2, Q) then
            # all-reduce into the global moments (mirrors the PME wrapper).
            OpAdapter(
                op=ops._slab_compute_partial_moments,  # (z, charges)
                arg_transforms={0: SliceOwned(), 1: SliceOwned()},
                output_transforms={
                    0: AllReduceSum(),
                    1: AllReduceSum(),
                    2: AllReduceSum(),
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

        # Compiled DD requires ``hybrid_forces=False``: the framework derives
        # forces via ``autograd.grad(energy, positions)``, which the warp
        # reciprocal kernel cannot provide, so the compiled path runs the
        # autograd-native Torch staged reciprocal (``ewald_recip_torch``)
        # instead. The wrapper emits per-atom ``atomic_energies`` that the
        # framework reduces owned-aware into the per-system energy; the dense
        # [N, K] neighbor matrix is padded to fixed shapes.
        compile_policy = (
            None
            if self.hybrid_forces
            else CompilePolicy(
                static_shapes=True,
                force_strategy=ForceStrategy.FRAMEWORK_FROM_NODE_ENERGY,
                graph_padder=DenseBatchPadder(),
            )
        )
        return MLIPSpec(
            distribution=dataclasses.replace(
                SPEC_EWALD_HALO.distribution, custom_ops=custom_ops
            ),
            # Eager reciprocal forces come from the all-reduced ``S(k)`` and are
            # exact on every owned row, so ``OWNED_ONLY``. Under compile, forces
            # come from autograd over the global energy and ``DistributedModel``
            # routes them through halo-reverse consolidation instead.
            outputs={
                "energy": OutputSpec(OutputKind.PER_GRAPH),
                "forces": OutputSpec(OutputKind.PER_NODE, Reduce.OWNED_ONLY),
                "stress": OutputSpec(OutputKind.PER_GRAPH),
                "atomic_energies": OutputSpec(OutputKind.PER_NODE),
            },
            # The wrapper emits per-atom ``atomic_energies``; the framework
            # reduces them owned-aware into the per-system ``energy``, so
            # ``forward`` carries no DD logic at the energy reduction.
            node_energy_key="atomic_energies",
            compile=compile_policy,
        )

    def distributed_setup(self, ctx: Any) -> None:
        """Enter distributed mode: stash the context and global atom count.

        Caches the global atom count so :meth:`_update_cache` estimates Ewald
        parameters from the global ``N`` rather than the per-rank padded count,
        then invalidates the cache so any params derived from a stale ``N`` are
        rebuilt on the next forward.

        Parameters
        ----------
        ctx : DistributedContext
            The live distributed context for this run.

        Returns
        -------
        None
        """
        self._dist_ctx = ctx
        self._n_global_atoms = ctx.n_atoms_total
        self.invalidate_cache()

    def distributed_teardown(self) -> None:
        """Exit distributed mode: clear the context and global atom count.

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
        """Embeddings are not defined for an electrostatics model.

        Parameters
        ----------
        data : AtomicData | Batch
            The input system (unused).
        **kwargs
            Unused; accepted for interface compatibility.

        Returns
        -------
        AtomicData | Batch
            Never returns.

        Raises
        ------
        NotImplementedError
            Always; Ewald has no learned embeddings.
        """
        raise NotImplementedError("EwaldModelWrapper does not produce embeddings.")

    def direct_derivative_keys(self) -> set[str]:
        """Return the outputs the kernel computes analytically.

        With ``hybrid_forces=True`` the Warp kernel returns analytical
        ``forces`` and ``stress`` directly, so the pipeline must not also derive
        them from energy autograd.

        Returns
        -------
        set[str]
            ``{"forces", "stress"}`` (intersected with the active outputs) when
            ``hybrid_forces`` is set, otherwise an empty set.
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
        """Return the batch keys :meth:`adapt_input` reads.

        Overrides the base set to require ``charges`` and the matrix-format
        neighbor data, and to drop ``atomic_numbers`` (unused by Ewald). When
        ``slab_correction`` is enabled, ``pbc`` is also required.

        Returns
        -------
        set[str]
            ``{"positions", "charges", "neighbor_matrix", "num_neighbors"}``,
            plus ``"pbc"`` when ``slab_correction=True``.
        """
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
        """Force recomputation of Ewald parameters and k-vectors.

        Call this after modifying the unit cell (e.g. an NPT integrator) so the
        next forward rebuilds ``alpha`` and the k-vectors.

        Returns
        -------
        None
        """
        self._cache_valid = False
        self._cached_alpha = None
        self._cached_k_vectors = None
        # Keep _cached_cell so forward()'s change-detection still works; clearing
        # it would re-invalidate the cache on the very next call.

    def _update_cache(
        self,
        positions: torch.Tensor,
        cell: torch.Tensor,
        batch_idx: torch.Tensor,
    ) -> None:
        """Recompute Ewald parameters and k-vectors for the given cell.

        The optimal ``alpha`` depends on the atom count ``N``. Under
        distribution ``positions.shape[0]`` is the per-rank padded count, which
        would diverge across ranks; when ``self._n_global_atoms`` is set, a
        shape-only surrogate of that size is passed to the estimator (which only
        reads ``num_atoms``) so every rank agrees.
        """
        from nvalchemiops.torch.interactions.electrostatics.k_vectors import (  # lazy
            generate_k_vectors_ewald_summation,
        )
        from nvalchemiops.torch.interactions.electrostatics.parameters import (  # lazy
            estimate_ewald_parameters,
        )

        if self._n_global_atoms is not None:
            est_positions = positions[:1].expand(self._n_global_atoms, -1).contiguous()
            est_batch_idx = batch_idx.new_zeros(self._n_global_atoms)
        else:
            est_positions = positions
            est_batch_idx = batch_idx

        params = estimate_ewald_parameters(
            est_positions, cell, batch_idx=est_batch_idx, accuracy=self.accuracy
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
        """Build the input dict for the Ewald kernels.

        Collects the keys named by :meth:`input_data`, casts the topology
        tensors to ``int32``, and attaches ``cell`` and optional
        ``neighbor_matrix_shifts``. Gradients are not enabled here; forces and
        stress come from the kernel analytically (hybrid) or from energy
        autograd downstream.

        Parameters
        ----------
        data : AtomicData | Batch
            The input system; must be a ``Batch`` (Ewald has no single-graph
            path).
        **kwargs
            Unused; accepted for interface compatibility.

        Returns
        -------
        dict[str, Any]
            Kernel inputs: ``positions`` ``[N, 3]``, ``charges`` ``[N]``,
            ``neighbor_matrix``, ``num_neighbors``, ``batch_idx`` ``[N]``,
            ``ptr``, ``num_graphs``, ``fill_value``, ``cell`` ``[B, 3, 3]``, and
            ``neighbor_matrix_shifts`` (or ``None``).

        Raises
        ------
        TypeError
            If *data* is an ``AtomicData`` rather than a ``Batch``.
        KeyError
            If a required input key is missing from *data*.
        ValueError
            If ``data.cell`` is absent (PBC is required).
        """
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
        # Optional PBC shifts; the neighbor matrix itself is collected above.
        input_dict["neighbor_matrix_shifts"] = getattr(
            data, "neighbor_matrix_shifts", None
        )
        return input_dict

    # ------------------------------------------------------------------
    # Output adaptation
    # ------------------------------------------------------------------

    def adapt_output(self, model_output: Any, data: AtomicData | Batch) -> ModelOutputs:
        """Map raw kernel outputs to nvalchemi standard keys.

        Always forwards ``energy``; adds ``forces`` and ``stress`` when they are
        in ``model_config.active_outputs``.

        Parameters
        ----------
        model_output : Any
            The dict produced by :meth:`forward` (``energy`` ``[B, 1]``, and
            ``forces`` / ``stress`` when active).
        data : AtomicData | Batch
            The input system the outputs were computed for.

        Returns
        -------
        ModelOutputs
            Ordered dict with ``energy`` and, when active, ``forces`` ``[N, 3]``
            and ``stress`` ``[B, 3, 3]``.

        Raises
        ------
        RuntimeError
            If ``stress`` is active but missing from *model_output*.
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
        """Return the output keys this model produces for the active config.

        Returns
        -------
        set[str]
            ``{"energy"}`` plus ``"forces"`` and/or ``"stress"`` when those are
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
        """Run the Ewald summation and return a :class:`ModelOutputs` dict.

        Parameters
        ----------
        data : Batch
            Batch containing ``positions``, ``charges``, ``cell``,
            ``neighbor_matrix``, and ``num_neighbors`` (populated by
            :class:`~nvalchemi.hooks.NeighborListHook`).
        **kwargs
            Forwarded to :meth:`adapt_input`.

        Returns
        -------
        ModelOutputs
            OrderedDict with keys ``"energy"`` (shape ``[B, 1]``, eV),
            ``"forces"`` (shape ``[N, 3]``, eV/Å), and optionally
            ``"stress"`` (shape ``[B, 3, 3]``, eV/Å³ — tensile-positive Cauchy
            stress ``-W/V``).
        """
        from nvalchemiops.torch.interactions.electrostatics.ewald import (  # lazy
            ewald_real_space,
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

        # In hybrid mode the kernel computes forces/virial analytically (no
        # autograd tape); detach so backward isn't expected when inputs already
        # carry grad (e.g. a pipeline's strain prep).
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

        # --- Real-space contribution ---
        real_result = ewald_real_space(
            positions=positions,
            charges=charges,
            cell=cell,
            alpha=alpha,
            neighbor_matrix=neighbor_matrix,
            neighbor_matrix_shifts=neighbor_matrix_shifts.contiguous(),
            mask_value=fill_value,
            batch_idx=batch_idx,
            compute_forces=compute_forces,
            compute_virial=compute_stresses,
            hybrid_forces=self.hybrid_forces,
        )

        # Unpack helper (energies first; forces / virial follow as requested).
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

        # --- Reciprocal-space contribution ---
        # The op selects the reciprocal backend (eager kernel forces, or a
        # differentiable energy-only path for autograd forces / DD) internally.
        from nvalchemi.models._ops.electrostatics.ewald import (  # noqa: PLC0415
            ewald_reciprocal_contribution,
        )

        e_recip, f_recip, v_recip = ewald_reciprocal_contribution(
            positions,
            charges,
            cell,
            k_vectors,
            alpha,
            batch_idx,
            B,
            compute_forces=compute_forces,
            compute_virial=compute_stresses,
            hybrid_forces=self.hybrid_forces,
        )

        # --- Slab correction (2D-periodic / Yeh-Berkowitz) ---
        # Added as a separate additive correction consistent with the real +
        # reciprocal split. Its global per-system moments are reduced owned-only
        # and all-reduced across ranks inside the helper, so it is halo-correct
        # under domain decomposition. Per-atom energy / force / virial follow the
        # analytical slab formulas (machine-precision-equal to the warp kernel).
        e_slab = torch.zeros_like(e_real)
        f_slab: torch.Tensor | None = None
        v_slab: torch.Tensor | None = None
        if self.slab_correction:
            from nvalchemi.models._ops.electrostatics.slab import (  # noqa: PLC0415
                compute_slab_correction_from_moments,
            )

            slab = compute_slab_correction_from_moments(
                positions=positions,
                charges=charges,
                cell=cell,
                pbc=pbc,
                batch_idx=batch_idx,
                compute_forces=compute_forces,
                compute_virial=compute_stresses,
            )
            slab_tuple = slab if isinstance(slab, tuple) else (slab,)
            e_slab = slab_tuple[0]
            idx = 1
            if compute_forces:
                f_slab = slab_tuple[idx]
                idx += 1
            if compute_stresses:
                v_slab = slab_tuple[idx]

        # Sum real + reciprocal + slab; scale by the Coulomb constant.
        per_atom_energies = (
            (e_real + e_recip + e_slab).to(positions.dtype) * self.coulomb_constant
        )

        forces: torch.Tensor | None = None
        if compute_forces and f_real is not None and f_recip is not None:
            forces = f_real + f_recip
            if f_slab is not None:
                forces = forces + f_slab
            forces = forces * self.coulomb_constant

        virial: torch.Tensor | None = None
        if compute_stresses and v_real is not None and v_recip is not None:
            virial = v_real + v_recip
            if v_slab is not None:
                virial = virial + v_slab
            virial = virial * self.coulomb_constant

        per_atom_energies = per_atom_energies.to(torch.float64)
        model_output: dict[str, Any] = {}
        if "energy" in self.model_config.active_outputs:
            # Per-atom energies -> per-system totals in fp64. Correct on a single
            # GPU; under decomposition the framework overrides ``energy`` with an
            # owned-aware sum of ``atomic_energies``.
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
        """Serialization is not supported for the Ewald model.

        Parameters
        ----------
        path : Path
            Output path (unused).
        as_state_dict : bool, optional
            Unused. Defaults to ``False``.

        Returns
        -------
        None
            Never returns.

        Raises
        ------
        NotImplementedError
            Always; Ewald has no checkpointable state.
        """
        raise NotImplementedError
