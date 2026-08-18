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
"""Core bias abstractions: ``BiasPotential`` protocol, ``BiasResult``, and
``ConservativeBias`` autograd helper.

This module is the foundation of the enhanced-sampling subpackage.  Every
downstream built-in bias depends on these three objects.

Design guarantees
-----------------
* ``BiasResult`` is a frozen dataclass; all tensor fields are detached
  (``requires_grad=False``, ``grad_fn is None``).  Validation is enforced
  in eager mode; the check is skipped inside ``torch.compile`` to avoid
  graph breaks on attribute inspection.
* ``BiasPotential`` is a ``@runtime_checkable`` Protocol.  Bias authors
  may satisfy it structurally without inheriting from any base class.
* ``ConservativeBias`` encapsulates the autograd subgraph that derives
  atomic forces and the tensile-positive Cauchy stress from a scalar
  energy function.  The subgraph is isolated from the live ``Batch`` so
  that no ``requires_grad`` leaf ever escapes into model state, batch
  storage, or ``BiasResult``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import torch
from torch import Tensor

from nvalchemi.models._utils import (
    autograd_forces,
    autograd_forces_and_stresses,
    prepare_strain,
)

if TYPE_CHECKING:
    from nvalchemi.data import Batch

__all__ = [
    "BiasResult",
    "BiasPotential",
    "ConservativeBias",
    "aggregate_bias_results",
]

# ---------------------------------------------------------------------------
# BiasResult
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BiasResult:
    """Immutable, fully-detached output of a single bias evaluation.

    All tensor fields must be detached (``requires_grad=False`` and
    ``grad_fn is None``).  Energy, forces, stress, and virial are
    independently optional.  Provide **either** ``stress`` or ``virial``,
    not both; the runner converts stress to virial or vice-versa as needed.

    Parameters
    ----------
    energy:
        Per-graph bias energy, shape ``[B, 1]``, unit eV.
    forces:
        Per-atom bias forces, shape ``[N_atoms, 3]``, unit eV/Å.
    stress:
        Tensile-positive Cauchy stress ``σ = −W/V``, shape ``[B, 3, 3]``,
        unit eV/Å³.  This is the toolkit-wide convention and what
        :class:`ConservativeBias` produces.  Mutually exclusive with
        ``virial``.
    virial:
        Virial ``W = −dE/dε`` with ``ε`` the symmetric infinitesimal
        strain tensor, shape ``[B, 3, 3]``, unit eV.  Provided for biases
        that compute a virial directly.  Mutually exclusive with
        ``stress``.
    state_version:
        Integer version IDs used by ``ReplicaExchange`` to validate that
        accepted state assignments are coherent, shape ``[B]``.
    observables:
        Named diagnostic tensors exposed as ``bias/<name>/<key>`` in the
        runner's output dict.  All tensors must be detached.
    """

    energy: Tensor | None = None
    forces: Tensor | None = None
    stress: Tensor | None = None
    virial: Tensor | None = None
    state_version: Tensor | None = None
    observables: Mapping[str, Tensor] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not torch.compiler.is_compiling():
            _validate_bias_result(self)


def _validate_bias_result(result: BiasResult) -> None:
    """Eager-only validation of a ``BiasResult`` (skipped under compile).

    Checks (in order):

    1. Mutual exclusion of ``stress`` and ``virial``.
    2. All tensor fields are detached (``requires_grad=False``, ``grad_fn is None``).
    3. Shapes match the documented conventions:

       * ``energy``       — ndim=2, shape ``[B, 1]``
       * ``forces``       — ndim=2, shape ``[N, 3]``
       * ``stress``       — ndim=3, shape ``[B, 3, 3]``
       * ``virial``       — ndim=3, shape ``[B, 3, 3]``
       * ``state_version`` — ndim=1, integer dtype

    4. Batch-size consistency: all present system-level fields
       (``energy``, ``stress``, ``virial``, ``state_version``) must agree
       on the leading dimension ``B``.
    5. All floating-point tensors (including ``observables``) are finite
       (no NaN or Inf).
    """
    # 1. stress / virial mutual exclusion
    if result.stress is not None and result.virial is not None:
        raise ValueError("BiasResult: provide either 'stress' or 'virial', not both.")

    # 2. Detachment check for every tensor field
    tensor_fields: dict[str, Tensor | None] = {
        "energy": result.energy,
        "forces": result.forces,
        "stress": result.stress,
        "virial": result.virial,
        "state_version": result.state_version,
    }
    for name, t in tensor_fields.items():
        if t is None:
            continue
        if t.requires_grad:
            raise ValueError(
                f"BiasResult.{name} must be detached "
                f"(requires_grad=False), got requires_grad=True."
            )
        if t.grad_fn is not None:
            raise ValueError(
                f"BiasResult.{name} must be detached "
                f"(grad_fn is None), got grad_fn={t.grad_fn}."
            )
    for key, t in result.observables.items():
        if t.requires_grad:
            raise ValueError(
                f"BiasResult.observables[{key!r}] must be detached "
                f"(requires_grad=False)."
            )
        if t.grad_fn is not None:
            raise ValueError(
                f"BiasResult.observables[{key!r}] must be detached (grad_fn is None)."
            )

    # 3. Shape checks
    if result.energy is not None:
        e = result.energy
        if e.ndim != 2 or e.shape[1] != 1:
            raise ValueError(
                f"BiasResult.energy must have shape [B, 1], got {tuple(e.shape)}."
            )

    if result.forces is not None:
        f = result.forces
        if f.ndim != 2 or f.shape[1] != 3:
            raise ValueError(
                f"BiasResult.forces must have shape [N, 3], got {tuple(f.shape)}."
            )

    for name in ("stress", "virial"):
        t = getattr(result, name)
        if t is not None and (t.ndim != 3 or t.shape[1] != 3 or t.shape[2] != 3):
            raise ValueError(
                f"BiasResult.{name} must have shape [B, 3, 3], got {tuple(t.shape)}."
            )

    if result.state_version is not None:
        sv = result.state_version
        if sv.ndim != 1:
            raise ValueError(
                f"BiasResult.state_version must have shape [B], got {tuple(sv.shape)}."
            )
        if sv.dtype not in (
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
        ):
            raise ValueError(
                f"BiasResult.state_version must be an integer dtype, got {sv.dtype}."
            )

    # 4. Batch-size consistency across system-level fields
    b_sizes: dict[str, int] = {}
    for name in ("energy", "stress", "virial", "state_version"):
        t = getattr(result, name)
        if t is not None:
            b_sizes[name] = t.shape[0]
    if len(set(b_sizes.values())) > 1:
        raise ValueError(
            f"BiasResult: leading batch dimension B is inconsistent across fields: "
            f"{b_sizes}."
        )

    # 5. Finiteness — NaN and Inf are never valid output values
    for name, t in tensor_fields.items():
        if t is None or not t.is_floating_point():
            continue
        if not t.isfinite().all():
            raise ValueError(f"BiasResult.{name} contains NaN or Inf values.")
    for key, t in result.observables.items():
        if t.is_floating_point() and not t.isfinite().all():
            raise ValueError(
                f"BiasResult.observables[{key!r}] contains NaN or Inf values."
            )


# ---------------------------------------------------------------------------
# BiasPotential Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class BiasPotential(Protocol):
    """Structural protocol for all enhanced-sampling bias potentials.

    Every built-in bias satisfies this protocol.  Authors may also satisfy
    it structurally (no inheritance required).

    Attributes
    ----------
    name:
        Unique string identifier used as a dict key in
        ``EnhancedSampling(biases={...})`` and as a Zarr group name in
        checkpoints.

    Methods
    -------
    evaluate(current)
        Read-only evaluation.  Must not mutate bias state, write to
        storage, or communicate.  Called every force evaluation by
        default.

    Adaptive biases additionally implement ``update()``,
    ``commit_epoch()``, ``state_dict()``, and ``load_state_dict()``.
    These are optional extensions that the runner detects via
    ``hasattr``; they are not part of this base protocol.
    """

    name: str

    def evaluate(self, current: Batch) -> BiasResult:
        """Evaluate the bias on the current batch.

        Must be **read-only**: it must not mutate bias internal state,
        deposit hills, write any storage, or communicate across workers.
        It is safe to call ``evaluate`` multiple times on the same batch
        without side effects.

        Parameters
        ----------
        current:
            The live ``Batch`` from the dynamics step.  Treat as
            read-only; do not modify any field.

        Returns
        -------
        BiasResult
            Fully detached outputs.  All tensor fields must satisfy
            ``requires_grad=False`` and ``grad_fn is None``.
        """
        ...


# ---------------------------------------------------------------------------
# ConservativeBias — autograd helper
# ---------------------------------------------------------------------------


class ConservativeBias:
    """Autograd helper that derives atomic forces and stress from energy.

    Subclass ``ConservativeBias`` and override :meth:`energy` to return a
    differentiable per-graph bias energy ``[B, 1]``.  The base class
    provides :meth:`evaluate`, which:

    1. Enters a local ``torch.enable_grad()`` region (safe inside
       ``torch.no_grad()`` outer contexts).
    2. Creates a detached positions leaf ``pos_leaf`` (for forces) and,
       for periodic batches, a per-graph strain leaf via
       :func:`~nvalchemi.models._utils.prepare_strain` (for stress).
    3. Substitutes the strained positions and cell into the batch, calls
       :meth:`energy`, and restores the original tensors unconditionally
       in a ``finally`` block.
    4. Derives forces and stress in one ``autograd.grad`` call through
       :func:`~nvalchemi.models._utils.autograd_forces_and_stresses`.
    5. Constructs a ``BiasResult`` from fully detached output tensors.

    The framework must never place a tensor with ``requires_grad=True`` or
    a non-null ``grad_fn`` into the live ``Batch``, ``BiasResult``,
    retained history, bias state, observables, or a checkpoint.

    Notes
    -----
    Stress rather than virial
        ``evaluate`` populates :attr:`BiasResult.stress`, never
        :attr:`BiasResult.virial`.  Tensile-positive Cauchy stress is the
        toolkit-wide public convention: every model wrapper emits
        ``"stress"``, ``sum_outputs`` treats it as additive, and the
        NPT/NPH integrators read ``batch.stress``.  Emitting stress lets a
        bias contribution be summed directly with model outputs with no
        volume conversion at the boundary.  ``BiasResult.virial`` remains
        available for hand-written biases that produce a virial directly.

    Strain convention
        The strain leaf comes from
        :func:`~nvalchemi.models._utils.prepare_strain`, which applies only
        the symmetric part of the leaf as strain.  The resulting gradient
        is therefore symmetric by construction, matching the project
        definition ``W_ab = −dE/dε_ab`` with ``ε`` the symmetric
        infinitesimal strain tensor (see
        :doc:`/userguide/about/conventions`).  Differentiating with respect
        to the full (unsymmetrised) deformation gradient instead yields an
        asymmetric tensor for any bias that is not a central pair
        interaction.

    Stress computation
        Stress is computed only when the batch carries a cell and at least
        one periodic dimension.  Subclasses that never need stress may set
        ``_supports_stress = False`` to skip the strain leaf entirely.
        There is no ``compute_stress`` constructor parameter.

    torch.compile compatibility
        :meth:`evaluate` runs in eager mode.  It uses
        ``pos_leaf = positions.detach().requires_grad_(True)``, which is
        not supported by ``torch.compile`` (``Unsupported
        Tensor.requires_grad_() call``).  The documented fallback is:
        compile :meth:`energy` independently (the user's hot path);
        keep :meth:`evaluate` as the eager orchestration wrapper.
        ``EnhancedSampling(compile_biases=True)`` applies
        ``torch.compile`` to each bias's ``energy()`` override only.
    """

    # Subclasses may set this to False to skip stress computation even when
    # a periodic cell is present (e.g. force-only biases that extend
    # ConservativeBias but never need a cell response).
    _supports_stress: bool = True

    def energy(self, current: Batch) -> Tensor:
        """Return bias energy ``[B, 1]`` (eV).

        Must be differentiable w.r.t. ``current.positions`` and/or
        ``current.cell``.  Depending on only one of the two is allowed: a
        position-independent term (a volume restraint, say) yields zero
        forces rather than an error.

        Parameters
        ----------
        current:
            A *read-only view* of the live batch whose ``positions`` and
            ``cell`` have been replaced by their strained counterparts from
            :func:`~nvalchemi.models._utils.prepare_strain`.  Do not assign
            to any batch field inside this method.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement energy(self, current: Batch) -> Tensor"
        )

    def evaluate(self, current: Batch) -> BiasResult:
        """Compute energy, forces, and Cauchy stress via autograd.

        This method runs in eager mode.  See the class docstring for the
        compile boundary note and the chosen fallback.

        Stress derivation
        -----------------
        Under a homogeneous strain ``ε`` (ASE row-vector convention), both
        atomic positions and the cell deform together::

            r_n    → r_n    @ (I + ε)
            cell_b → cell_b @ (I + ε)

        :func:`~nvalchemi.models._utils.prepare_strain` applies exactly this
        deformation through a leaf whose symmetric part is used, so a single
        ``autograd.grad`` call yields:

        * ``dE/d(pos_leaf[n])`` at ``ε=0`` → forces (negated).
        * ``dE/dε`` at ``ε=0`` → ``σ = (dE/dε) / V``, tensile-positive
          Cauchy stress, equivalently ``σ = −W/V``.

        Straining positions and cell together is what makes this correct for
        position-dependent biases built on MIC displacements: the position
        term and the cell gradient term are combined automatically.  A
        strain leaf applied to the cell alone misses the position
        contribution and gives the wrong answer for pair restraints that
        span an image boundary.

        Partial dependence
        ------------------
        :meth:`energy` need not depend on both positions and strain.  A term
        that depends only on the cell — a volume restraint, a barostat-style
        term — returns zero forces and a non-zero stress; a term that returns
        a constant on some branch returns zeros for both.  These are ordinary
        outcomes, not errors: the gradient of an energy with respect to
        something it does not use is zero.

        Returns
        -------
        BiasResult
            With ``energy``, ``forces``, and — for periodic batches —
            ``stress``.  ``virial`` is always ``None``; see the class
            docstring for why stress is the chosen field.
        """
        original_positions = current.positions
        original_cell = getattr(current, "cell", None)
        pbc = getattr(current, "pbc", None)
        cell_is_4d = original_cell is not None and original_cell.dim() == 4

        # Stress needs a cell to strain and a non-zero volume to divide by.  A
        # batch may carry a placeholder cell with pbc all-False, whose zero
        # volume would turn the stress into Inf, so gate on periodicity too.
        # bool(pbc.any()) is a data-dependent branch, which is safe here
        # because evaluate() is eager-only.
        strain_cell: Tensor | None = None
        if (
            self._supports_stress
            and original_cell is not None
            and (pbc is None or bool(pbc.any()))
        ):
            # Detach the stored cell so only the strain leaf carries the
            # gradient; a [B, 1, 3, 3] cell is squeezed to [B, 3, 3].
            strain_cell = original_cell.detach()
            if cell_is_4d:
                strain_cell = strain_cell.squeeze(1)

        with torch.enable_grad():
            # requires_grad_() is not supported by torch.compile; evaluate()
            # is intentionally kept eager (see class docstring).
            pos_leaf = original_positions.detach().requires_grad_(True)  # [N, 3]

            displacement: Tensor | None = None
            pos_for_energy: Tensor = pos_leaf
            cell_for_energy: Tensor | None = None

            if strain_cell is not None:
                pos_for_energy, cell_for_energy, displacement = prepare_strain(
                    pos_leaf, strain_cell, current.batch_idx
                )

            try:
                current["positions"] = pos_for_energy
                if cell_for_energy is not None:
                    current["cell"] = (
                        cell_for_energy.unsqueeze(1) if cell_is_4d else cell_for_energy
                    )

                bias_energy: Tensor = self.energy(current)  # [B, 1]

                stress: Tensor | None = None
                if not bias_energy.requires_grad:
                    # An energy with no graph at all — a bias returning a
                    # constant on this branch.  Every gradient is zero, but
                    # autograd rejects such an output outright ("does not
                    # require grad and does not have a grad_fn"), so fill the
                    # zeros directly rather than calling it.
                    forces = torch.zeros_like(pos_leaf)
                    if strain_cell is not None:
                        stress = torch.zeros_like(strain_cell)
                elif strain_cell is not None and displacement is not None:
                    # allow_unused: a bias need not depend on both positions
                    # and strain.  A pure volume restraint has no position
                    # dependence, and its zero force is an answer, not an error.
                    forces, stress = autograd_forces_and_stresses(
                        bias_energy,
                        pos_leaf,
                        displacement,
                        strain_cell,
                        current.num_graphs,
                        allow_unused=True,
                    )
                else:
                    forces = autograd_forces(bias_energy, pos_leaf, allow_unused=True)

            finally:
                current["positions"] = original_positions
                if original_cell is not None:
                    current["cell"] = original_cell

        return BiasResult(
            energy=bias_energy.detach(),
            forces=forces.detach(),
            stress=None if stress is None else stress.detach(),
        )


# ---------------------------------------------------------------------------
# Bias aggregation
# ---------------------------------------------------------------------------


def aggregate_bias_results(results: list[BiasResult]) -> BiasResult:
    """Sum a list of ``BiasResult`` objects into a single combined result.

    All biases are evaluated against the **same unmodified physical
    outputs**; their contributions are summed once here and applied
    together.  A bias cannot accidentally observe the force contribution
    of another bias.

    Rules
    -----
    * ``None`` fields are skipped (treated as zero contribution).
    * All results must agree on which cell-response field they use: every
      result that carries a cell response must use **either** ``stress``
      **or** ``virial`` — never a mix of both across the list.  Mixing
      raises ``ValueError`` at aggregation time (not inside ``BiasResult``)
      with a message identifying which indices contributed each field.
      Converting between the two requires the cell volume and is the
      caller's responsibility before aggregation.
    * ``observables`` dicts are merged; duplicate keys raise ``ValueError``
      so that namespacing (``bias/<name>/<key>``) must be applied before
      calling this function.

    Parameters
    ----------
    results:
        List of ``BiasResult`` objects from individual biases.  May be
        empty, in which case an empty ``BiasResult()`` is returned.

    Returns
    -------
    BiasResult
        Aggregated result with summed contributions.
    """
    if not results:
        return BiasResult()

    energy_total: Tensor | None = None
    forces_total: Tensor | None = None
    stress_total: Tensor | None = None
    virial_total: Tensor | None = None
    observables_total: dict[str, Tensor] = {}

    # Detect stress/virial mixing up-front so the error is raised here with
    # a clear message, not inside BiasResult.__post_init__ with a generic
    # mutual-exclusion message that doesn't identify which results mixed them.
    has_stress = any(r.stress is not None for r in results)
    has_virial = any(r.virial is not None for r in results)
    if has_stress and has_virial:
        stress_indices = [i for i, r in enumerate(results) if r.stress is not None]
        virial_indices = [i for i, r in enumerate(results) if r.virial is not None]
        raise ValueError(
            f"aggregate_bias_results: results[{stress_indices}] provide 'stress' "
            f"and results[{virial_indices}] provide 'virial' — cannot mix both in "
            "the same aggregation.  Make all biases return the same field.  "
            "Converting between stress and virial requires the cell volume and is "
            "the caller's responsibility before aggregation."
        )

    for r in results:
        if r.energy is not None:
            energy_total = r.energy if energy_total is None else energy_total + r.energy
        if r.forces is not None:
            forces_total = r.forces if forces_total is None else forces_total + r.forces
        if r.stress is not None:
            stress_total = r.stress if stress_total is None else stress_total + r.stress
        if r.virial is not None:
            virial_total = r.virial if virial_total is None else virial_total + r.virial
        for key, val in r.observables.items():
            if key in observables_total:
                raise ValueError(
                    f"aggregate_bias_results: duplicate observable key {key!r}. "
                    "Apply 'bias/<name>/<key>' namespacing before aggregation."
                )
            observables_total[key] = val

    return BiasResult(
        energy=energy_total,
        forces=forces_total,
        stress=stress_total,
        virial=virial_total,
        observables=observables_total,
    )
