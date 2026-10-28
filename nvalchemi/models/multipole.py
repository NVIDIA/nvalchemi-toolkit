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
* Moments are read from the per-atom batch fields ``charges`` ``[N]``,
  ``dipoles`` ``[N, 3]`` (Cartesian x, y, z) and ``quadrupoles`` ``[N, 3, 3]``
  (symmetric Cartesian). All are optional but at least one is required; the
  highest channel present sets ``l_max`` (``quadrupoles`` -> 2, ``dipoles`` -> 1,
  else 0). A missing lower channel is zero, so a dipole-only system sends just
  ``dipoles``. The l=2 channel is traceless: any isotropic trace of
  ``quadrupoles`` is discarded.
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

import contextlib
import functools
import inspect
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
    from collections.abc import Callable, Iterator

    from nvalchemi._typing import ModelOutputs
    from nvalchemi.data import AtomicData, Batch

__all__ = ["MultipoleEwaldModelWrapper", "MultipolePMEModelWrapper"]

_MOMENT_CHANNELS = ("charges", "dipoles", "quadrupoles")


@functools.cache
def _ops_accept_miller_indices() -> bool:
    """Whether the installed ops take a caller-retained Ewald k-grid topology."""
    from nvalchemiops.torch.interactions.electrostatics import (  # noqa: PLC0415
        multipole_ewald_summation,
    )

    return "miller_indices" in inspect.signature(multipole_ewald_summation).parameters


@functools.cache
def _ops_accept_total_charge() -> bool:
    """Whether the installed ops take an externally reduced total charge."""
    from nvalchemiops.torch.interactions.electrostatics import (  # noqa: PLC0415
        multipole_ewald_summation,
    )

    return "total_charge" in inspect.signature(multipole_ewald_summation).parameters


# Higher multipoles carry derivative prefactors the monopole tail estimate
# ignores, so the requested accuracy is tightened per l_max before the split is
# derived. Calibrated on a single geometry; see multipole-wrapper-scoping.md.
_ACCURACY_MARGIN = {0: 1.0, 1: 10.0, 2: 100.0}

# Mesh points per reciprocal-space extent, relative to the Nyquist minimum.
# l=2 needs a finer mesh (calibrated at spline order 6: worst error 0.65x the
# requested accuracy over three box/sigma combinations).
_MESH_OVERSAMPLING = {0: 2.5, 1: 2.5, 2: 4.0}


def _system_total(
    per_atom: torch.Tensor, batch_idx: torch.Tensor, num_graphs: int
) -> torch.Tensor:
    """Sum per-atom energies into per-system totals (float64)."""
    total = torch.zeros(num_graphs, dtype=torch.float64, device=per_atom.device)
    return total.scatter_add(0, batch_idx.to(torch.long), per_atom)


def _distributed_system_total(
    per_atom: Any, batch_idx: Any, num_graphs: int
) -> torch.Tensor:
    """Owned-aware replacement for :func:`_system_total` under decomposition.

    Sums each rank's owned rows and all-reduces, so the total is global and its
    gradient reaches only owned energies.
    """
    from nvalchemi.distributed._core.enums import Scope  # noqa: PLC0415
    from nvalchemi.distributed.helpers import system_sum, to_local  # noqa: PLC0415

    return system_sum(
        to_local(per_atom), to_local(batch_idx), num_graphs, scope=Scope.OWNED
    )


def _owned_cell_grad_attach() -> type:
    """Cell-gradient attach for l<=1 real space that counts owned receivers only.

    The stock Function re-evaluates the per-system ``dE/dcell`` over every row of
    the neighbor list and scales it by the mean per-atom energy cotangent. Under
    decomposition that would also count pairs whose receiver is a ghost, and the
    mean is diluted by the ghost rows' zero cotangent. Emptying the ghost rows of
    the CSR pointer counts each pair once, on the rank that owns its receiver,
    and the cotangent is taken over owned rows only.
    """
    from nvalchemiops.torch.interactions.electrostatics.multipole_ewald import (  # noqa: PLC0415
        _AttachRealSpaceCellGradLmax1 as stock,
    )

    from nvalchemi.distributed._core.context import current_dd_context  # noqa: PLC0415
    from nvalchemi.distributed.helpers import to_local  # noqa: PLC0415

    class _OwnedCellGradAttach(stock):
        @staticmethod
        def setup_context(ctx: Any, inputs: tuple[Any, ...], output: Any) -> None:
            n_owned = current_dd_context().n_owned
            ptr = inputs[8]
            owned_ptr = torch.minimum(ptr, ptr[n_owned])
            stock.setup_context(ctx, (*inputs[:8], owned_ptr, *inputs[9:]), output)
            ctx.n_owned = n_owned

        @staticmethod
        def backward(ctx: Any, grad_energy: torch.Tensor) -> tuple[Any, ...]:
            if not ctx.has_batch:
                return stock.backward(ctx, grad_energy)
            cell = to_local(ctx.saved_tensors[0])
            batch_idx = to_local(ctx.saved_tensors[-1]).to(torch.long)
            grad = to_local(grad_energy).reshape(-1)
            owned = torch.arange(grad.numel(), device=grad.device) < ctx.n_owned
            zeros = grad.new_zeros(cell.shape[0])
            total = zeros.index_add(0, batch_idx, torch.where(owned, grad, 0.0))
            count = zeros.index_add(0, batch_idx, owned.to(grad.dtype))
            uniform = (total / count.clamp_min(1.0)).index_select(0, batch_idx)
            fixed = torch.where(owned, grad, uniform).reshape(grad_energy.shape)
            return (grad_energy, *stock.backward(ctx, fixed)[1:])

    return _OwnedCellGradAttach


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
        Expected multipole order; ``None`` infers it from the moment channels present.
    neighbor_format : NeighborListFormat
        ``MATRIX`` (fixed-width rows; static shapes, suits ``torch.compile`` and
        domain decomposition) or ``COO`` (sparse, fewer pair slots).
    rtol, atol : float, float or None
        Cell-change tolerances for cache invalidation.
    compile_model : bool
        ``torch.compile`` the energy evaluation; see the concrete classes.
    compile_kwargs : dict[str, Any] or None
        Forwarded to :func:`torch.compile` (``fullgraph=True`` by default).
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
        compile_model: bool,
        compile_kwargs: dict[str, Any] | None,
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
            required_inputs=frozenset(),
            optional_inputs=frozenset(_MOMENT_CHANNELS),
            supports_pbc=True,
            needs_pbc=True,
            neighbor_config=NeighborConfig(
                cutoff=cutoff, format=self.neighbor_format, half_list=False
            ),
        )
        # (alpha, k_cutoff) per l_max; constants of the wrapper, never stale.
        self._split_cache: dict[int, tuple[float, float]] = {}
        self._compiled_energy: Callable[..., torch.Tensor] | None = None
        # Validate eagerly so a bad (cutoff, sigma, accuracy) fails at construction.
        self._split(l_max if l_max is not None else 0)
        if compile_model:
            self._check_compile_supported()
            self._compile_energy(compile_kwargs)

    def _check_compile_supported(self) -> None:
        """Raise if this configuration cannot be compiled (hook for subclasses)."""

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

    def _dd_real_space_ops(self) -> tuple[Any, ...]:
        """Per-atom real-space kernels: unwrap for the kernel, re-wrap per-atom outputs."""
        from nvalchemi.distributed.spec import OpAdapter  # noqa: PLC0415

        nvops = torch.ops.nvalchemiops  # type: ignore[attr-defined]
        return tuple(
            OpAdapter(op=getattr(nvops, name))
            for name in (
                "batch_multipole_real_space_monopole_fused",
                "batch_multipole_real_space_dipole_fused",
                "batch_multipole_real_space_quadrupole",
            )
        )

    def _dd_custom_ops(self) -> tuple[Any, ...]:
        """Return the ``OpAdapter`` set that distributes the reciprocal-space ops."""
        raise NotImplementedError(
            f"{type(self).__name__} does not support domain decomposition; "
            "use MultipolePMEModelWrapper."
        )

    def distribution_spec(self, strategy: Any = None) -> Any:
        """Domain-decomposition spec for the multipole wrappers.

        Halo-only; ``strategy`` is accepted for the framework contract and
        ignored. The reciprocal-space ops get owned-slice and all-reduce
        handlers (see :meth:`_dd_custom_ops`) and the total-charge ops reduce the
        background-correction charge across ranks, so :meth:`forward` is
        distribution-agnostic. Forces and stress come from the wrapper's own autograd.

        Returns
        -------
        MLIPSpec
            Halo-storage spec carrying the op adapters, with per-atom
            ``atomic_energies`` reduced by the framework.

        Raises
        ------
        RuntimeError
            If the installed ``nvalchemiops`` cannot take an external total charge.
        """
        import dataclasses  # noqa: PLC0415

        from nvalchemi.distributed.spec import (  # noqa: PLC0415
            SPEC_PME_HALO,
            MLIPSpec,
            OutputKind,
            OutputSpec,
            PythonAdapter,
            Reduce,
        )

        custom_ops = self._dd_custom_ops()
        if not _ops_accept_total_charge():
            raise RuntimeError(
                "Domain decomposition needs an nvalchemiops whose multipole "
                "entry points accept total_charge."
            )
        return MLIPSpec(
            distribution=dataclasses.replace(
                SPEC_PME_HALO.distribution,
                custom_ops=custom_ops,
                # batch_idx keeps per-system lookups (volume, total charge)
                # sharded alongside the per-atom tensors they multiply.
                shard_fields=(*SPEC_PME_HALO.distribution.shard_fields, "batch_idx"),
                third_party_helpers=(
                    PythonAdapter(
                        module_path=__name__,
                        attr_name="_system_total",
                        replacement=_distributed_system_total,
                    ),
                    PythonAdapter(
                        module_path=(
                            "nvalchemiops.torch.interactions.electrostatics"
                            ".multipole_ewald"
                        ),
                        attr_name="_AttachRealSpaceCellGradLmax1",
                        replacement=_owned_cell_grad_attach(),
                    ),
                ),
            ),
            outputs={
                "energy": OutputSpec(OutputKind.PER_GRAPH),
                "forces": OutputSpec(OutputKind.PER_NODE),
                "stress": OutputSpec(OutputKind.PER_GRAPH, Reduce.ALL_REDUCE),
                "atomic_energies": OutputSpec(OutputKind.PER_NODE),
            },
        )

    def invalidate_cache(self) -> None:
        """Drop any cell-dependent cached state (a no-op for Ewald)."""

    def _per_atom_energy(
        self,
        positions: torch.Tensor,
        moments: torch.Tensor,
        cell: torch.Tensor,
        idx_j: torch.Tensor,
        ptr: torch.Tensor,
        shifts: torch.Tensor,
        batch_idx: torch.Tensor,
        reciprocal: dict[str, Any],
        total_charge: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Per-atom energy as a pure function of tensors and resolved parameters.

        Everything data dependent (``alpha``, mesh, k-grid) arrives in
        ``reciprocal``, so this is the unit ``compile_model`` compiles.
        """
        extra = {} if total_charge is None else {"total_charge": total_charge}
        return self._energy_op(
            positions,
            moments,
            cell,
            idx_j,
            ptr,
            shifts,
            sigma=self.sigma,
            batch_idx=batch_idx,
            **reciprocal,
            **extra,
        ).to(torch.float64)

    def _total_charge(self, data: Batch, batch_idx: torch.Tensor, B: int) -> Any:
        """Per-system total charge for the background term, or ``None`` if unsupported."""
        if not _ops_accept_total_charge():
            return None
        from nvalchemi.models._ops.electrostatics.pme import (  # noqa: F401, PLC0415
            _batch_pme_compute_partial_total_charge,
        )

        charges = getattr(data, "charges", None)
        if charges is None:
            return torch.zeros(B, dtype=torch.float64, device=batch_idx.device)
        return torch.ops.alchemiops._batch_pme_compute_partial_total_charge(  # type: ignore[attr-defined]
            charges, batch_idx, B
        )

    def _compile_energy(self, compile_kwargs: dict[str, Any] | None) -> None:
        """Compile the pure energy function (strain and force/stress autograd stay eager).

        The compiled backward is generated by AOT autograd. Reciprocal
        parameters are resolved eagerly in :meth:`forward` and passed in, so a
        change of mesh or k-grid recompiles rather than fails.
        """
        kwargs = {"fullgraph": True, **(compile_kwargs or {})}
        self._compiled_energy = torch.compile(self._per_atom_energy, **kwargs)

    @contextlib.contextmanager
    def _compile_scope(self) -> Iterator[None]:
        """Context for compiled evaluation and the derivative calls that follow.

        Donated buffers are disabled because ``forward`` differentiates with
        ``retain_graph=True``, which a donated-buffer backward rejects. AOT
        autograd compiles the backward lazily at the first ``autograd.grad``,
        so this must cover the derivatives as well as the energy call. Floats
        are specialized so ``sigma``/``alpha`` stay concrete under
        ``dynamic=True``; symbolic they break Dynamo's guards on the closed-form
        constants derived from them.
        """
        if self._compiled_energy is None:
            yield
            return
        with (
            torch._functorch.config.patch(donated_buffer=False),
            torch._dynamo.config.patch(specialize_float=True),
        ):
            yield

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
            ``positions`` and the neighbor tensors for the configured
            ``neighbor_format``. The moment channels are optional inputs.
        """
        keys = {"positions"}
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

    def _pack_moments(
        self, data: Batch, dtype: torch.dtype
    ) -> tuple[torch.Tensor, int]:
        """Pack the present moment channels into e3nn layout; return ``l_max``.

        Built from the pure-tensor conversion helpers rather than
        ``pack_multipole_moments``, whose trace check forces a host sync.
        """
        from nvalchemiops.torch.interactions.electrostatics._multipole_moments import (  # noqa: PLC0415
            cartesian_quadrupole_to_e3nn,
            dipole_cartesian_to_spherical,
        )

        n_atoms = data.num_nodes
        q, mu, quad = (getattr(data, key, None) for key in _MOMENT_CHANNELS)
        if q is None and mu is None and quad is None:
            raise KeyError(
                "at least one of 'charges', 'dipoles' or 'quadrupoles' is required "
                "but none was found in input data."
            )
        for name, value, shape in (
            ("charges", q, (n_atoms,)),
            ("dipoles", mu, (n_atoms, 3)),
            ("quadrupoles", quad, (n_atoms, 3, 3)),
        ):
            if value is not None and tuple(value.shape) != shape:
                raise ValueError(
                    f"'{name}' must have shape {shape}; got {tuple(value.shape)}."
                )
        l_max = 2 if quad is not None else 1 if mu is not None else 0
        if self.l_max is not None and l_max != self.l_max:
            raise ValueError(
                f"{type(self).__name__} was built with l_max={self.l_max} but the "
                f"batch carries channels for l_max={l_max}."
            )
        ref = next(v for v in (q, mu, quad) if v is not None)
        zeros = ref.new_zeros
        parts = [(q if q is not None else zeros(n_atoms)).unsqueeze(-1)]
        if l_max >= 1:
            parts.append(
                dipole_cartesian_to_spherical(
                    mu if mu is not None else zeros(n_atoms, 3)
                )
            )
        if l_max == 2:
            parts.append(cartesian_quadrupole_to_e3nn(quad))
        return torch.cat([p.to(dtype) for p in parts], dim=-1), l_max

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
            Batch with ``positions``, ``cell``, at least one of ``charges`` /
            ``dipoles`` / ``quadrupoles``, and neighbor data populated by a
            :class:`~nvalchemi.hooks.NeighborListHook`.
        **kwargs
            Unused; accepted for interface compatibility.

        Returns
        -------
        dict[str, Any]
            ``positions``, packed ``multipole_moments`` (e3nn layout, in the
            positions dtype), ``cell``, ``batch_idx`` (int32), ``num_graphs``
            and ``l_max``.

        Raises
        ------
        TypeError
            If *data* is an ``AtomicData`` rather than a ``Batch``.
        KeyError
            If a required key is missing or no moment channel is present.
        ValueError
            If a moment channel has the wrong shape, ``l_max`` disagrees with the
            constructor, or there is no cell.
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
        moments, l_max = self._pack_moments(data, data.positions.dtype)
        return {
            "positions": data.positions,
            "multipole_moments": moments,
            "cell": cell,
            "batch_idx": data.batch_idx.to(torch.int32),
            "num_graphs": data.num_graphs,
            "l_max": l_max,
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

    # The next three methods are eager graph-break points, so a user's plain
    # ``torch.compile(model)`` compiles the energy and runs leaf creation,
    # parameter resolution and the autograd calls eagerly around it.
    # ``torch.autograd.grad`` and ``requires_grad_`` are not traceable by
    # default, and the mesh/k-grid resolution is data dependent.

    @torch.compiler.disable
    def _derivative_inputs(
        self,
        positions: torch.Tensor,
        cell: torch.Tensor,
        batch_idx: torch.Tensor,
        want_stress: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Make the autograd leaves: positions and, for stress, the strain."""
        if not positions.requires_grad:
            positions = positions.detach().requires_grad_(True)
        displacement = None
        if want_stress:
            positions, cell, displacement = prepare_strain(
                positions, cell, batch_idx.to(torch.long)
            )
        return positions, cell, displacement

    @torch.compiler.disable
    def _resolve_reciprocal(self, l_max: int, cell: torch.Tensor) -> dict[str, Any]:
        """Resolve alpha, mesh and k-grid eagerly (see :meth:`_reciprocal_kwargs`)."""
        return self._reciprocal_kwargs(l_max, cell)

    @torch.compiler.disable
    def _derivatives(
        self,
        total: torch.Tensor,
        positions: torch.Tensor,
        displacement: torch.Tensor | None,
        cell: torch.Tensor,
        num_graphs: int,
        want_stress: bool,
        create_graph: bool,
    ) -> dict[str, torch.Tensor]:
        """Forces, and stress when requested, by autograd of the total energy.

        ``create_graph`` keeps the derivative differentiable, which only matters
        when the moments carry a graph from an upstream model. It is off
        otherwise because a compiled backward that uses donated buffers (seen
        on a dynamic-shape recompile) rejects ``create_graph`` and
        ``retain_graph``.
        """
        if want_stress:
            forces, stress = autograd_forces_and_stresses(
                total.sum(),
                positions,
                displacement,
                cell,
                num_graphs,
                training=create_graph,
            )
            return {"forces": forces, "stress": stress}
        (grad,) = torch.autograd.grad(
            [total.sum()],
            [positions],
            create_graph=create_graph,
            retain_graph=create_graph,
            allow_unused=True,
        )
        return {"forces": torch.zeros_like(positions) if grad is None else -grad}

    def forward(self, data: AtomicData | Batch, **kwargs: Any) -> ModelOutputs:
        """Evaluate the multipole electrostatic energy and its derivatives.

        Parameters
        ----------
        data : Batch
            Batch with ``positions``, ``cell``, at least one of ``charges``,
            ``dipoles`` and ``quadrupoles``, and neighbor data for
            ``neighbor_format``.

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
        want_derivatives = want_forces or want_stress

        displacement = None
        if want_derivatives:
            positions, cell, displacement = self._derivative_inputs(
                positions, cell, batch_idx, want_stress
            )
            reciprocal = self._resolve_reciprocal(l_max, cell)
        else:
            reciprocal = self._reciprocal_kwargs(l_max, cell)

        idx_j, ptr, shifts = self._csr(data, positions.shape[0], positions.device)
        total_charge = self._total_charge(data, batch_idx, B)
        with self._compile_scope():
            energy_fn = self._compiled_energy or self._per_atom_energy
            per_atom = energy_fn(
                positions,
                inp["multipole_moments"],
                cell,
                idx_j,
                ptr,
                shifts,
                batch_idx,
                reciprocal,
                total_charge,
            )

            total = _system_total(per_atom, batch_idx, B)

            model_output: dict[str, Any] = {
                "energy": total.to(positions.dtype).unsqueeze(-1),
                "atomic_energies": per_atom,
            }
            if want_derivatives:
                model_output.update(
                    self._derivatives(
                        total,
                        positions,
                        displacement,
                        data.cell,
                        B,
                        want_stress,
                        self.training and inp["multipole_moments"].requires_grad,
                    )
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

    Compilation: ``compile_model=True`` compiles the energy as a full graph;
    forces and stress are taken by autograd around it and match eager. A plain
    ``torch.compile(model)`` also works: it compiles the energy and graph-breaks
    around leaf creation and the autograd calls, so it cannot use
    ``fullgraph=True`` when forces or stress are active (energy-only can). The
    reciprocal k-grid is a static-shape Miller topology built eagerly and
    reused while the cell is unchanged, which needs an ``nvalchemiops`` that
    accepts ``miller_indices``; compiling raises or fails otherwise.

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
        Expected multipole order; ``None`` infers it from the moment channels present.
    neighbor_format : NeighborListFormat, optional
        ``MATRIX`` (default; fixed-width, static shapes) or ``COO``.
    rtol : float, optional
        Relative tolerance for cell-change detection.
    atol : float or None, optional
        Absolute tolerance for cell-change detection.
    compile_model : bool, optional
        ``torch.compile`` the energy evaluation. Strain and the force/stress
        autograd stay eager around it. Defaults to ``False``.
    compile_kwargs : dict[str, Any] or None, optional
        Forwarded to :func:`torch.compile` when ``compile_model=True``
        (``fullgraph=True`` by default).
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
        compile_model: bool = False,
        compile_kwargs: dict[str, Any] | None = None,
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
            compile_model=compile_model,
            compile_kwargs=compile_kwargs,
        )
        self.k_cutoff = k_cutoff
        # Read once: a cached call inside a traced region makes Dynamo warn.
        self._ops_take_miller = _ops_accept_miller_indices()
        self._miller_cell: torch.Tensor | None = None
        self._miller: dict[int, torch.Tensor] = {}

    def invalidate_cache(self) -> None:
        """Force the Miller-index topology to be rebuilt on the next call."""
        self._miller_cell = None
        self._miller = {}

    def _miller_topology(
        self, l_max: int, cell: torch.Tensor, k_cutoff: float
    ) -> torch.Tensor:
        """Return the half-space Miller topology, rebuilt only when the cell changed.

        The topology has a static shape, so the reciprocal sum is traceable by
        ``torch.compile``; the k-vectors built from it stay differentiable in the
        cell (stress). It is retained like the PME mesh: derived eagerly once,
        and reused under compilation.
        """
        if torch.compiler.is_compiling():
            if l_max not in self._miller:
                raise RuntimeError(
                    "The Ewald k-grid must be resolved before torch.compile: run "
                    "one eager forward first."
                )
            return self._miller[l_max]
        from nvalchemiops.torch.interactions.electrostatics.k_vectors import (  # noqa: PLC0415
            generate_ewald_miller_indices,
        )

        detached = cell.detach()
        if cell_cache_needs_update(
            detached, self._miller_cell, rtol=self.rtol, atol=self.atol
        ):
            self._miller_cell = detached.clone()
            self._miller = {}
        if l_max not in self._miller:
            self._miller[l_max] = generate_ewald_miller_indices(
                detached.reshape(-1, 3, 3), k_cutoff
            )
        return self._miller[l_max]

    def _check_compile_supported(self) -> None:
        """Compiling needs ops that accept a retained k-grid topology."""
        if not _ops_accept_miller_indices():
            raise RuntimeError(
                "compile_model=True needs an nvalchemiops whose "
                "multipole_ewald_summation accepts miller_indices."
            )

    def _reciprocal_kwargs(self, l_max: int, cell: torch.Tensor) -> dict[str, Any]:
        """Return ``alpha``, ``k_cutoff`` and, when ops support it, the k-grid topology."""
        alpha, k_cut = self._split(l_max)
        k_cutoff = self.k_cutoff if self.k_cutoff is not None else k_cut
        if self._ops_take_miller:
            # Ops reject k_cutoff alongside a retained topology.
            return {
                "alpha": alpha,
                "miller_indices": self._miller_topology(l_max, cell, k_cutoff),
            }
        return {"alpha": alpha, "k_cutoff": k_cutoff}

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

    Compilation: ``compile_model=True`` compiles the energy as a full graph;
    forces and stress are taken by autograd around it and match eager. A plain
    ``torch.compile(model)`` also works: it compiles the energy and graph-breaks
    around leaf creation, mesh resolution and the autograd calls, so it cannot
    use ``fullgraph=True`` when forces or stress are active. An energy-only,
    fullgraph compile reuses the mesh from an earlier eager call.

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
        B-spline order (ops register 4, 5 and 6). Defaults to ``6``, which
        meets ``accuracy`` for all ``l_max``; ``5`` is cheaper but misses it at
        ``l_max=2``.
    l_max : int or None, optional
        Expected multipole order; ``None`` infers it from the moment channels present.
    neighbor_format : NeighborListFormat, optional
        ``MATRIX`` (default; fixed-width, static shapes) or ``COO``.
    rtol : float, optional
        Relative tolerance for cell-change detection.
    atol : float or None, optional
        Absolute tolerance for cell-change detection.
    compile_model : bool, optional
        ``torch.compile`` the energy evaluation. Strain and the force/stress
        autograd stay eager around it. Defaults to ``False``.
    compile_kwargs : dict[str, Any] or None, optional
        Forwarded to :func:`torch.compile` when ``compile_model=True``
        (``fullgraph=True`` by default).
    """

    def __init__(
        self,
        cutoff: float,
        *,
        sigma: float,
        accuracy: float = 1e-6,
        alpha: float | None = None,
        mesh_dimensions: tuple[int, int, int] | None = None,
        spline_order: int = 6,
        l_max: int | None = None,
        neighbor_format: NeighborListFormat = NeighborListFormat.MATRIX,
        rtol: float = 1e-5,
        atol: float | None = None,
        compile_model: bool = False,
        compile_kwargs: dict[str, Any] | None = None,
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
            compile_model=compile_model,
            compile_kwargs=compile_kwargs,
        )
        self.mesh_dimensions = mesh_dimensions
        self.spline_order = spline_order
        self._cached_cell: torch.Tensor | None = None
        self._cached_mesh: dict[int, tuple[int, int, int]] = {}
        self._mesh_floor: dict[int, tuple[int, int, int]] = {}

    def invalidate_cache(self) -> None:
        """Force the mesh dimensions to be recomputed on the next call."""
        self._cached_cell = None
        self._cached_mesh = {}
        self._mesh_floor = {}

    def _mesh(self, l_max: int, cell: torch.Tensor) -> tuple[int, int, int]:
        """Return mesh dimensions, recomputing only when the cell changed."""
        if self.mesh_dimensions is not None:
            return self.mesh_dimensions
        if torch.compiler.is_compiling():
            # The cell-change check and height reduction are data dependent, so
            # compiled calls reuse the mesh resolved by an earlier eager call.
            if l_max not in self._cached_mesh:
                raise RuntimeError(
                    "mesh_dimensions must be resolved before torch.compile: run "
                    "one eager forward first, or pass mesh_dimensions explicitly."
                )
            return self._cached_mesh[l_max]
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
            mesh = tuple(
                _fft_size(math.ceil(_MESH_OVERSAMPLING[l_max] * k_cut * h / math.pi))
                for h in heights
            )
            if self._compiled_energy is not None:
                # Mesh sizes are compile-time constants, so each distinct one is a
                # recompile. Only grow it: a varying cell or batch then settles on
                # a bounded set of meshes instead of recompiling without limit.
                floor = self._mesh_floor.get(l_max, (0, 0, 0))
                mesh = tuple(max(a, b) for a, b in zip(mesh, floor, strict=True))
                self._mesh_floor[l_max] = mesh
            self._cached_mesh[l_max] = mesh
        return self._cached_mesh[l_max]

    def _reciprocal_kwargs(self, l_max: int, cell: torch.Tensor) -> dict[str, Any]:
        """Return ``alpha``, mesh and spline order for the PME call."""
        alpha, _ = self._split(l_max)
        return {
            "alpha": alpha,
            "mesh_dimensions": self._mesh(l_max, cell),
            "spline_order": self.spline_order,
        }

    def _dd_custom_ops(self) -> tuple[Any, ...]:
        """Owned-slice + all-reduce handlers for the spread and total-charge ops.

        Each rank spreads only its owned atoms and the partial meshes are summed
        into the global mesh, which every downstream stage (FFT, convolution,
        gather) then reads identically on every rank; its incoming gradient is
        already global, hence the replicated adjoint.
        """
        from nvalchemiops.torch.interactions.electrostatics import (  # noqa: F401, PLC0415
            pme_multipole,
        )

        from nvalchemi.distributed._core.op_transforms import (  # noqa: PLC0415
            AllReduceSum,
            SliceOwned,
        )
        from nvalchemi.distributed.spec import OpAdapter  # noqa: PLC0415
        from nvalchemi.models._ops.electrostatics.pme import (  # noqa: F401, PLC0415
            _batch_pme_compute_partial_total_charge,
            _pme_compute_partial_total_charge,
        )

        nvops = torch.ops.nvalchemiops  # type: ignore[attr-defined]
        ops = torch.ops.alchemiops  # type: ignore[attr-defined]
        mesh = {0: AllReduceSum()}
        return self._dd_real_space_ops() + (
            # Fractionalization is per atom: re-wrap its outputs as sharded.
            OpAdapter(op=nvops.multipole_pme_fractionalize),
            # (p_frac, charges, dipoles, quadrupoles, cell_inv_t, ...)
            OpAdapter(
                op=nvops.multipole_pme_spread_unified,
                arg_transforms={i: SliceOwned() for i in range(4)},
                output_transforms=mesh,
            ),
            # (p_frac, charges, dipoles, quadrupoles, batch_idx, cell_inv_t, ...)
            OpAdapter(
                op=nvops.multipole_pme_spread_unified_batch,
                arg_transforms={i: SliceOwned() for i in range(5)},
                output_transforms=mesh,
            ),
            OpAdapter(op=nvops.multipole_pme_gather_via_spread_t_batch),
            OpAdapter(
                op=ops._pme_compute_partial_total_charge,
                arg_transforms={0: SliceOwned()},
                output_transforms={0: AllReduceSum()},
            ),
            OpAdapter(
                op=ops._batch_pme_compute_partial_total_charge,
                arg_transforms={0: SliceOwned(), 1: SliceOwned()},
                output_transforms={0: AllReduceSum()},
            ),
        )

    def _energy_op(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Call ``multipole_particle_mesh_ewald``."""
        from nvalchemiops.torch.interactions.electrostatics import (  # noqa: PLC0415
            multipole_particle_mesh_ewald,
        )

        return multipole_particle_mesh_ewald(*args, **kwargs)
