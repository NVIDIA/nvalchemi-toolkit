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

"""Multipole (GTO-Ewald) electrostatics model wrappers.

Wraps the ``nvalchemiops`` multipole electrostatics as
:class:`~nvalchemi.models.base.BaseModelMixin` models: charges, dipoles and
quadrupoles evaluated with either direct-k Ewald
(:class:`MultipoleEwaldModelWrapper`) or particle-mesh Ewald
(:class:`MultipolePMEModelWrapper`).

Usage
-----
::

    from nvalchemi.models.multipole import MultipolePMEModelWrapper

    model = MultipolePMEModelWrapper(cutoff=10.0, sigma=0.6)

Notes
-----
* Moments are read from ``data.multipole_moments`` in the packed e3nn layout
  ``(N, (l_max+1)**2)``: ``[q]``, ``[q, mu_y, mu_z, mu_x]``, or that plus five
  traceless l=2 channels. The trailing width selects ``l_max`` per call; build
  the field from Cartesian channels with
  :func:`nvalchemiops.torch.interactions.electrostatics.pack_multipole_moments`.
  A dipole-only system sends a zero charge column; a quadrupole-only system
  sends zero dipoles.
* The ops return energies in eV for positions in Å and moments in elementary
  charge units; there is no Coulomb-constant knob.
* The Ewald split is **derived from** ``cutoff``, ``sigma`` and ``accuracy`` so
  the real-space sum is consistent with the neighbor list the hook builds, with
  no dependence on system size and no host synchronization. See
  :func:`_derive_split`.
* Forces and stress come from autograd over the per-atom energy. In a
  :class:`~nvalchemi.models.pipeline.PipelineModelWrapper` autograd group the
  chain rule through upstream moment-producing models comes for free.
* Neighbor data is consumed in either the ``MATRIX`` or ``COO`` format; both are
  converted to the CSR the ops expect (see ``neighbor_format``).
"""

from __future__ import annotations

import math
from collections import OrderedDict
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from torch import nn

from nvalchemi.models._derivatives import DerivativeNotSupported, _DerivativeRequest
from nvalchemi.models._utils import (
    autograd_forces_and_stresses,
    cell_cache_needs_update,
    prepare_strain,
)
from nvalchemi.models.base import (
    BaseModelMixin,
    ModelConfig,
    NeighborConfig,
    NeighborListFormat,
)

if TYPE_CHECKING:
    from nvalchemi._typing import ModelOutputs
    from nvalchemi.data import AtomicData, Batch

__all__ = ["MultipoleEwaldModelWrapper", "MultipolePMEModelWrapper"]

_MOMENT_WIDTH_TO_LMAX = {1: 0, 4: 1, 9: 2}

# Higher multipoles carry derivative prefactors the monopole tail estimate
# ignores, so the requested accuracy is tightened per l_max before the split is
# derived. Calibrated on a single geometry; see multipole-wrapper-scoping.md.
_ACCURACY_MARGIN = {0: 1.0, 1: 10.0, 2: 100.0}

# Mesh points per reciprocal-space extent, relative to the Nyquist minimum.
_MESH_OVERSAMPLING = 2.5


def _erfcinv(x: float) -> float:
    """Inverse complementary error function for a small positive ``x``."""
    return torch.erfinv(torch.tensor(1.0 - x, dtype=torch.float64)).item()


def _derive_split(
    cutoff: float,
    sigma: float,
    accuracy: float,
    l_max: int,
    alpha: float | None = None,
) -> tuple[float, float]:
    r"""Derive ``(alpha, k_cutoff)`` from the real-space cutoff.

    The real-space tail decays as :math:`\mathrm{erfc}(r / 2\sigma_c)` and the
    reciprocal envelope as :math:`\exp(-k^2 \sigma_c^2)`, with
    :math:`\sigma_c^2 = \sigma^2 + 1/(4\alpha^2)`. Fixing the truncation error at
    ``cutoff`` to the (``l_max``-tightened) accuracy fixes :math:`\sigma_c`, and
    with it ``alpha`` and ``k_cutoff``.

    Parameters
    ----------
    cutoff : float
        Real-space cutoff in Å.
    sigma : float
        GTO density-basis width in Å.
    accuracy : float
        Target relative accuracy, tightened internally by ``l_max``.
    l_max : int
        Highest multipole order present (0, 1 or 2).
    alpha : float or None, optional
        Explicit splitting parameter. When given, ``sigma_c`` follows from it
        and only ``k_cutoff`` is derived.

    Returns
    -------
    tuple[float, float]
        ``(alpha, k_cutoff)`` in inverse Å.

    Raises
    ------
    ValueError
        If ``cutoff`` is too small for ``sigma`` and ``accuracy`` (no valid
        Ewald split exists), or ``alpha`` is not positive.
    """
    acc = accuracy / _ACCURACY_MARGIN[l_max]
    if alpha is None:
        sigma_c = cutoff / (2.0 * _erfcinv(acc))
        if sigma_c <= sigma:
            min_cutoff = 2.0 * sigma * _erfcinv(acc)
            raise ValueError(
                f"cutoff={cutoff} is too small for sigma={sigma} at "
                f"accuracy={accuracy} (l_max={l_max}); no valid Ewald split "
                f"exists. Use cutoff > {min_cutoff:.3f}, a smaller sigma, or a "
                "looser accuracy."
            )
        alpha = 1.0 / (2.0 * math.sqrt(sigma_c**2 - sigma**2))
    else:
        if alpha <= 0:
            raise ValueError(f"alpha must be positive, got {alpha}.")
        sigma_c = math.sqrt(sigma**2 + 1.0 / (4.0 * alpha**2))
    return alpha, math.sqrt(-math.log(acc)) / sigma_c


def _is_smooth(n: int) -> bool:
    """Return whether ``n`` factors into 2, 3, 5 and 7 only."""
    for p in (2, 3, 5, 7):
        while n % p == 0:
            n //= p
    return n == 1


def _fft_size(n: int) -> int:
    """Smallest multiple of 4 that is ``>= n`` and 7-smooth."""
    n = max(4, -(-n // 4) * 4)
    while not _is_smooth(n):
        n += 4
    return n


class _BaseMultipoleWrapper(nn.Module, BaseModelMixin):
    """Shared machinery for the multipole electrostatics wrappers.

    Owns moment validation, Ewald-split derivation, neighbor-format adaptation
    to CSR, strain/autograd derivatives and output assembly. Subclasses supply
    :meth:`_reciprocal_kwargs` and :meth:`_energy_op`.

    Parameters
    ----------
    cutoff : float
        Real-space cutoff in Å.
    sigma : float
        GTO density-basis width in Å, a property of the moments' source.
    accuracy : float
        Target relative accuracy used to derive the Ewald split.
    alpha : float or None
        Explicit Ewald splitting parameter, or ``None`` to derive it.
    l_max : int or None
        Expected multipole order; ``None`` infers it from the moments width.
    neighbor_format : NeighborListFormat
        ``MATRIX`` (fixed-width rows; static shapes, suits ``torch.compile`` and
        domain decomposition) or ``COO`` (sparse, fewer pair slots).
    rtol, atol : float, float or None
        Cell-change tolerances for cache invalidation.
    """

    def __init__(
        self,
        cutoff: float,
        *,
        sigma: float,
        accuracy: float,
        alpha: float | None,
        l_max: int | None,
        neighbor_format: NeighborListFormat,
        rtol: float,
        atol: float | None,
    ) -> None:
        super().__init__()
        if sigma <= 0:
            raise ValueError(f"sigma must be positive, got {sigma}.")
        if l_max is not None and l_max not in (0, 1, 2):
            raise ValueError(f"l_max must be 0, 1, 2 or None, got {l_max}.")
        if neighbor_format not in (NeighborListFormat.MATRIX, NeighborListFormat.COO):
            raise ValueError(f"Unsupported neighbor_format {neighbor_format!r}.")
        self.cutoff = cutoff
        self.sigma = sigma
        self.accuracy = accuracy
        self.alpha = alpha
        self.l_max = l_max
        self.neighbor_format = NeighborListFormat(neighbor_format)
        self.rtol = rtol
        self.atol = atol
        self.model_config = ModelConfig(
            outputs=frozenset({"energy", "forces", "stress"}),
            active_outputs={"energy", "forces"},
            autograd_outputs=frozenset({"forces", "stress"}),
            autograd_inputs=frozenset({"positions"}),
            required_inputs=frozenset({"multipole_moments"}),
            optional_inputs=frozenset(),
            supports_pbc=True,
            needs_pbc=True,
            neighbor_config=NeighborConfig(
                cutoff=cutoff, format=self.neighbor_format, half_list=False
            ),
        )
        # (alpha, k_cutoff) per l_max; constants of the wrapper, never stale.
        self._split_cache: dict[int, tuple[float, float]] = {}
        # Validate eagerly so a bad (cutoff, sigma, accuracy) fails at construction.
        self._split(l_max if l_max is not None else 0)

    # ------------------------------------------------------------------
    # Ewald split
    # ------------------------------------------------------------------

    def _split(self, l_max: int) -> tuple[float, float]:
        """Return the cached ``(alpha, k_cutoff)`` for ``l_max``."""
        if l_max not in self._split_cache:
            self._split_cache[l_max] = _derive_split(
                self.cutoff, self.sigma, self.accuracy, l_max, self.alpha
            )
        return self._split_cache[l_max]

    def _reciprocal_kwargs(self, l_max: int, cell: torch.Tensor) -> dict[str, Any]:
        """Return the method-specific keyword arguments for the ops call."""
        raise NotImplementedError

    def _energy_op(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Call the ops entry point and return per-atom energies."""
        raise NotImplementedError

    def invalidate_cache(self) -> None:
        """Drop any cell-dependent cached state (a no-op for Ewald)."""

    # ------------------------------------------------------------------
    # BaseModelMixin surface
    # ------------------------------------------------------------------

    @property
    def embedding_shapes(self) -> dict[str, tuple[int, ...]]:
        """No embeddings are produced."""
        return {}

    def compute_embeddings(
        self, data: AtomicData | Batch, **kwargs: Any
    ) -> AtomicData | Batch:
        """Embeddings are not defined for electrostatics models.

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
            Always.
        """
        raise NotImplementedError(f"{type(self).__name__} does not produce embeddings.")

    def _validate_derivative_request(self, request: _DerivativeRequest) -> None:
        """Reject second-order derivatives until each path is verified."""
        raise DerivativeNotSupported(
            model_name=type(self).__name__,
            operation=request.operation,
            execution=request.execution,
            strategy=request.strategy,
            reason="multipole second-order derivatives are not yet verified",
        )

    def input_data(self) -> set[str]:
        """List the batch attributes the forward reads.

        Returns
        -------
        set[str]
            ``positions``, ``multipole_moments`` and the neighbor tensors for
            the configured ``neighbor_format``.
        """
        keys = {"positions", "multipole_moments"}
        if self.neighbor_format == NeighborListFormat.MATRIX:
            keys |= {"neighbor_matrix", "num_neighbors"}
        return keys

    def output_data(self) -> set[str]:
        """List the output keys the forward currently produces.

        Returns
        -------
        set[str]
            ``energy`` plus whichever of ``forces`` / ``stress`` are active.
        """
        keys = {"energy"}
        keys |= self.model_config.active_outputs & {"forces", "stress"}
        return keys

    # ------------------------------------------------------------------
    # Input adaptation
    # ------------------------------------------------------------------

    def _check_moments(self, moments: torch.Tensor, n_atoms: int) -> int:
        """Validate the packed moments and return ``l_max``."""
        if moments.dim() != 2 or moments.shape[0] != n_atoms:
            raise ValueError(
                f"multipole_moments must have shape ({n_atoms}, 1|4|9); "
                f"got {tuple(moments.shape)}."
            )
        width = moments.shape[1]
        if width not in _MOMENT_WIDTH_TO_LMAX:
            raise ValueError(
                f"multipole_moments has width {width}; expected 1 (q), 4 "
                "(q, mu_y, mu_z, mu_x) or 9 (adds five l=2 channels)."
            )
        l_max = _MOMENT_WIDTH_TO_LMAX[width]
        if self.l_max is not None and l_max != self.l_max:
            raise ValueError(
                f"{type(self).__name__} was built with l_max={self.l_max} but "
                f"multipole_moments has width {width} (l_max={l_max})."
            )
        return l_max

    def _csr(
        self, data: Batch, n_atoms: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(idx_j, neighbor_ptr, unit_shifts)`` CSR for the ops.

        ``COO`` edges are sender-sorted, so ``edge_ptr`` is already the CSR
        pointer. ``MATRIX`` rows become a fixed-width CSR with ``K`` slots per
        atom; padding slots point at the atom itself with a zero shift, which the
        kernels skip (zero separation), so every shape is static.
        """
        if self.neighbor_format == NeighborListFormat.COO:
            edges = data.neighbor_list
            idx_j = edges[:, 1].to(torch.int32).contiguous()
            ptr = data.edge_ptr.to(torch.int32)
            shifts = getattr(data, "neighbor_list_shifts", None)
            if shifts is None:
                shifts = torch.zeros(
                    edges.shape[0], 3, dtype=torch.int32, device=device
                )
            return idx_j, ptr, shifts.to(torch.int32).contiguous()

        matrix = data.neighbor_matrix
        K = matrix.shape[1]
        self_idx = torch.arange(n_atoms, device=device, dtype=matrix.dtype)[:, None]
        valid = matrix < n_atoms
        idx_j = torch.where(valid, matrix, self_idx).reshape(-1).to(torch.int32)
        ptr = (torch.arange(n_atoms + 1, device=device) * K).to(torch.int32)
        shifts = getattr(data, "neighbor_matrix_shifts", None)
        if shifts is None:
            shifts = torch.zeros(n_atoms, K, 3, dtype=torch.int32, device=device)
        shifts = torch.where(valid[..., None], shifts, torch.zeros_like(shifts))
        return idx_j, ptr, shifts.reshape(-1, 3).to(torch.int32)

    def adapt_input(self, data: AtomicData | Batch, **kwargs: Any) -> dict[str, Any]:
        """Collect the ops inputs from *data* without enabling gradients.

        Parameters
        ----------
        data : Batch
            Batch with ``positions``, ``multipole_moments``, ``cell`` and
            neighbor data populated by a
            :class:`~nvalchemi.hooks.NeighborListHook`.
        **kwargs
            Unused; accepted for interface compatibility.

        Returns
        -------
        dict[str, Any]
            ``positions``, ``multipole_moments``, ``cell``, ``batch_idx`` (int32),
            ``num_graphs`` and ``l_max``.

        Raises
        ------
        TypeError
            If *data* is an ``AtomicData`` rather than a ``Batch``.
        KeyError
            If a required key is missing.
        ValueError
            If the moments have an invalid shape or width, or there is no cell.
        """
        from nvalchemi.data import Batch  # noqa: PLC0415

        if not isinstance(data, Batch):
            raise TypeError(
                f"{type(self).__name__} requires a Batch input; got AtomicData. "
                "Use Batch.from_data_list([data]) to wrap it."
            )
        for key in self.input_data():
            if getattr(data, key, None) is None:
                raise KeyError(f"'{key}' required but not found in input data.")
        cell = getattr(data, "cell", None)
        if cell is None:
            raise ValueError(
                f"{type(self).__name__} requires periodic boundary conditions "
                "(data.cell must be present)."
            )
        moments = data.multipole_moments
        return {
            "positions": data.positions,
            "multipole_moments": moments,
            "cell": cell,
            "batch_idx": data.batch_idx.to(torch.int32),
            "num_graphs": data.num_graphs,
            "l_max": self._check_moments(moments, data.num_nodes),
        }

    def adapt_output(self, model_output: Any, data: AtomicData | Batch) -> ModelOutputs:
        """Select the active outputs into the standard output mapping.

        Parameters
        ----------
        model_output : dict[str, Any]
            Raw outputs keyed by ``energy``, ``atomic_energies``, ``forces`` and
            ``stress``.
        data : AtomicData | Batch
            The input system (unused).

        Returns
        -------
        ModelOutputs
            Ordered mapping of the active outputs.
        """
        active = self.model_config.active_outputs
        output: ModelOutputs = OrderedDict()
        for key in ("energy", "atomic_energies", "forces", "stress"):
            if key in model_output and (key == "energy" or key in active):
                output[key] = model_output[key]
        return output

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, data: AtomicData | Batch, **kwargs: Any) -> ModelOutputs:
        """Evaluate the multipole electrostatic energy and its derivatives.

        Parameters
        ----------
        data : Batch
            Batch with ``positions``, ``multipole_moments`` (``(N, 1|4|9)``),
            ``cell`` and neighbor data for ``neighbor_format``.

        Returns
        -------
        ModelOutputs
            ``energy`` ``[B, 1]`` (eV), ``atomic_energies`` ``[N]`` (fp64) and,
            when active, ``forces`` ``[N, 3]`` (eV/Å) and ``stress``
            ``[B, 3, 3]`` (eV/Å³, tensile-positive Cauchy).
        """
        inp = self.adapt_input(data, **kwargs)
        positions = inp["positions"]
        cell = inp["cell"]
        batch_idx = inp["batch_idx"]
        B: int = inp["num_graphs"]
        l_max: int = inp["l_max"]

        active = self.model_config.active_outputs
        want_forces = "forces" in active
        want_stress = "stress" in active

        displacement = None
        if want_forces or want_stress:
            if not positions.requires_grad:
                positions = positions.detach().requires_grad_(True)
            if want_stress:
                positions, cell, displacement = prepare_strain(
                    positions, cell, batch_idx.to(torch.long)
                )

        idx_j, ptr, shifts = self._csr(data, positions.shape[0], positions.device)
        per_atom = self._energy_op(
            positions,
            inp["multipole_moments"].to(positions.dtype),
            cell,
            idx_j,
            ptr,
            shifts,
            sigma=self.sigma,
            batch_idx=batch_idx,
            **self._reciprocal_kwargs(l_max, cell),
        ).to(torch.float64)

        total = torch.zeros(B, dtype=torch.float64, device=positions.device)
        total = total.scatter_add(0, batch_idx.to(torch.long), per_atom)

        model_output: dict[str, Any] = {
            "energy": total.to(positions.dtype).unsqueeze(-1),
            "atomic_energies": per_atom,
        }
        if want_stress:
            forces, stress = autograd_forces_and_stresses(
                total.sum(),
                positions,
                displacement,
                data.cell,
                B,
                training=self.training,
                retain_graph=True,
            )
            model_output["forces"] = forces
            model_output["stress"] = stress
        elif want_forces:
            (grad,) = torch.autograd.grad(
                [total.sum()],
                [positions],
                create_graph=self.training,
                retain_graph=True,
                allow_unused=True,
            )
            model_output["forces"] = (
                torch.zeros_like(positions) if grad is None else -grad
            )
        return self.adapt_output(model_output, data)

    def export_model(self, path: Path, as_state_dict: bool = False) -> None:
        """Serialization is not supported (no trainable weights).

        Parameters
        ----------
        path : Path
            Intended output path (unused).
        as_state_dict : bool, optional
            Unused.

        Raises
        ------
        NotImplementedError
            Always.
        """
        raise NotImplementedError


class MultipoleEwaldModelWrapper(_BaseMultipoleWrapper):
    r"""Multipole electrostatics via direct-k GTO-Ewald summation.

    Real-space pair sum on the neighbor list plus a direct reciprocal sum of
    cost :math:`O(N \cdot N_k)`; accurate and simple, best for small and
    medium systems. See :class:`MultipolePMEModelWrapper` for large systems.

    Parameters
    ----------
    cutoff : float
        Real-space cutoff in Å.
    sigma : float
        GTO density-basis width in Å. Required: it belongs to whatever produced
        the moments. Must satisfy ``cutoff > 2 * sigma * erfcinv(accuracy)``.
    accuracy : float, optional
        Target relative accuracy used to derive the Ewald split. Defaults to
        ``1e-6``.
    alpha : float or None, optional
        Ewald splitting parameter (1/Å). ``None`` derives it from ``cutoff``.
    k_cutoff : float or None, optional
        Reciprocal-space cutoff (1/Å). ``None`` derives it from the split.
    l_max : int or None, optional
        Expected multipole order; ``None`` infers it from the moments width.
    neighbor_format : NeighborListFormat, optional
        ``MATRIX`` (default; fixed-width, static shapes) or ``COO``.
    rtol : float, optional
        Relative tolerance for cell-change detection.
    atol : float or None, optional
        Absolute tolerance for cell-change detection.
    """

    def __init__(
        self,
        cutoff: float,
        *,
        sigma: float,
        accuracy: float = 1e-6,
        alpha: float | None = None,
        k_cutoff: float | None = None,
        l_max: int | None = None,
        neighbor_format: NeighborListFormat = NeighborListFormat.MATRIX,
        rtol: float = 1e-5,
        atol: float | None = None,
    ) -> None:
        super().__init__(
            cutoff,
            sigma=sigma,
            accuracy=accuracy,
            alpha=alpha,
            l_max=l_max,
            neighbor_format=neighbor_format,
            rtol=rtol,
            atol=atol,
        )
        self.k_cutoff = k_cutoff

    def _reciprocal_kwargs(self, l_max: int, cell: torch.Tensor) -> dict[str, Any]:
        """Return ``alpha`` and ``k_cutoff`` for the Ewald summation."""
        alpha, k_cut = self._split(l_max)
        return {
            "alpha": alpha,
            "k_cutoff": self.k_cutoff if self.k_cutoff is not None else k_cut,
        }

    def _energy_op(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Call ``multipole_ewald_summation``."""
        from nvalchemiops.torch.interactions.electrostatics import (  # noqa: PLC0415
            multipole_ewald_summation,
        )

        return multipole_ewald_summation(*args, **kwargs)


class MultipolePMEModelWrapper(_BaseMultipoleWrapper):
    r"""Multipole electrostatics via particle-mesh GTO-Ewald.

    The reciprocal sum is evaluated on an FFT mesh with B-spline spreading,
    :math:`O(M \log M + N p^3)`, at the cost of a small spline-truncation
    error that does not shrink with the mesh. Prefer it for large systems.

    Parameters
    ----------
    cutoff : float
        Real-space cutoff in Å.
    sigma : float
        GTO density-basis width in Å. Required: it belongs to whatever produced
        the moments. Must satisfy ``cutoff > 2 * sigma * erfcinv(accuracy)``.
    accuracy : float, optional
        Target relative accuracy used to derive the Ewald split and mesh.
        Defaults to ``1e-6``.
    alpha : float or None, optional
        Ewald splitting parameter (1/Å). ``None`` derives it from ``cutoff``.
    mesh_dimensions : tuple[int, int, int] or None, optional
        FFT mesh ``(nx, ny, nz)``. ``None`` derives it from the reciprocal
        cutoff and the cell; it is recomputed only when the cell changes.
    spline_order : int, optional
        B-spline order. Defaults to ``5``.
    l_max : int or None, optional
        Expected multipole order; ``None`` infers it from the moments width.
    neighbor_format : NeighborListFormat, optional
        ``MATRIX`` (default; fixed-width, static shapes) or ``COO``.
    rtol : float, optional
        Relative tolerance for cell-change detection.
    atol : float or None, optional
        Absolute tolerance for cell-change detection.
    """

    def __init__(
        self,
        cutoff: float,
        *,
        sigma: float,
        accuracy: float = 1e-6,
        alpha: float | None = None,
        mesh_dimensions: tuple[int, int, int] | None = None,
        spline_order: int = 5,
        l_max: int | None = None,
        neighbor_format: NeighborListFormat = NeighborListFormat.MATRIX,
        rtol: float = 1e-5,
        atol: float | None = None,
    ) -> None:
        super().__init__(
            cutoff,
            sigma=sigma,
            accuracy=accuracy,
            alpha=alpha,
            l_max=l_max,
            neighbor_format=neighbor_format,
            rtol=rtol,
            atol=atol,
        )
        self.mesh_dimensions = mesh_dimensions
        self.spline_order = spline_order
        self._cached_cell: torch.Tensor | None = None
        self._cached_mesh: dict[int, tuple[int, int, int]] = {}

    def invalidate_cache(self) -> None:
        """Force the mesh dimensions to be recomputed on the next call."""
        self._cached_cell = None
        self._cached_mesh = {}

    def _mesh(self, l_max: int, cell: torch.Tensor) -> tuple[int, int, int]:
        """Return mesh dimensions, recomputing only when the cell changed."""
        if self.mesh_dimensions is not None:
            return self.mesh_dimensions
        detached = cell.detach()
        if cell_cache_needs_update(
            detached, self._cached_cell, rtol=self.rtol, atol=self.atol
        ):
            self._cached_cell = detached.clone()
            self._cached_mesh = {}
        if l_max not in self._cached_mesh:
            _, k_cut = self._split(l_max)
            cells = detached.reshape(-1, 3, 3)
            volume = torch.det(cells).abs()
            cross = torch.stack(
                [
                    torch.linalg.cross(cells[:, 1], cells[:, 2]),
                    torch.linalg.cross(cells[:, 2], cells[:, 0]),
                    torch.linalg.cross(cells[:, 0], cells[:, 1]),
                ],
                dim=1,
            ).norm(dim=-1)
            heights = (volume[:, None] / cross).amax(dim=0).tolist()
            self._cached_mesh[l_max] = tuple(
                _fft_size(math.ceil(_MESH_OVERSAMPLING * k_cut * h / math.pi))
                for h in heights
            )
        return self._cached_mesh[l_max]

    def _reciprocal_kwargs(self, l_max: int, cell: torch.Tensor) -> dict[str, Any]:
        """Return ``alpha``, mesh and spline order for the PME call."""
        alpha, _ = self._split(l_max)
        return {
            "alpha": alpha,
            "mesh_dimensions": self._mesh(l_max, cell),
            "spline_order": self.spline_order,
        }

    def _energy_op(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Call ``multipole_particle_mesh_ewald``."""
        from nvalchemiops.torch.interactions.electrostatics import (  # noqa: PLC0415
            multipole_particle_mesh_ewald,
        )

        return multipole_particle_mesh_ewald(*args, **kwargs)
