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
  atomic forces and the canonical cell virial from a scalar energy
  function.  The subgraph is isolated from the live ``Batch`` so that no
  ``requires_grad`` leaf ever escapes into model state, batch storage, or
  ``BiasResult``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import torch
from torch import Tensor

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
        Tensile-positive Cauchy stress, shape ``[B, 3, 3]``.
        Mutually exclusive with ``virial``.
    virial:
        Canonical virial ``W = −dE/dstrain``, shape ``[B, 3, 3]``.
        Mutually exclusive with ``stress``.
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

    Every P0 built-in satisfies this protocol.  Authors may also satisfy
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
    """Autograd helper that derives atomic forces and cell virial from energy.

    Subclass ``ConservativeBias`` and override :meth:`energy` to return a
    differentiable per-graph bias energy ``[B, 1]``.  The base class
    provides :meth:`evaluate`, which:

    1. Enters a local ``torch.enable_grad()`` region (safe inside
       ``torch.no_grad()`` outer contexts).
    2. Creates fresh detached autograd leaves for positions (and cell when
       ``compute_virial=True``).
    3. Evaluates :meth:`energy` on an isolated read-only view of the batch
       that replaces positions (and cell) with the fresh leaves.
    4. Derives forces and (optionally) virial in one ``autograd.grad`` call
       with ``create_graph=False, retain_graph=False``.
    5. Constructs a ``BiasResult`` from fully detached output tensors.
    6. Drops all references to the autograd subgraph before returning, so
       that no ``grad_fn`` ever escapes into the live batch or result.

    The framework must never place a tensor with ``requires_grad=True`` or
    a non-null ``grad_fn`` into the live ``Batch``, ``BiasResult``,
    retained history, bias state, observables, or a checkpoint.

    Parameters
    ----------
    compute_virial:
        When ``True`` (default when a valid cell exists at evaluation
        time), derive the canonical virial
        ``W = −dE/d(strain)`` via ``autograd.grad`` w.r.t. the cell
        leaf.  The virial is shaped ``[B, 3, 3]``.

    Notes
    -----
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

    # Subclasses may set this to False to skip virial computation even when
    # a cell is present (e.g. force-only biases that extend ConservativeBias
    # but never need stress/virial).
    _supports_virial: bool = True

    def energy(self, current: Batch) -> Tensor:
        """Return bias energy ``[B, 1]`` (eV).

        Must be differentiable w.r.t. ``current.positions`` (and
        ``current.cell`` when virial is requested).

        Parameters
        ----------
        current:
            A *read-only view* of the live batch where ``positions`` (and
            optionally ``cell``) have been replaced by fresh autograd
            leaves.  Do not assign to any batch field inside this method.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement energy(self, current: Batch) -> Tensor"
        )

    def evaluate(self, current: Batch) -> BiasResult:
        """Compute energy, forces, and canonical cell virial via autograd.

        This method runs in eager mode.  See class docstring for the
        compile boundary note and the chosen fallback.

        Virial derivation
        -----------------
        The canonical virial is ``W = −dE/dstrain`` evaluated at the
        identity strain.  Under a homogeneous deformation ``F`` (ASE
        row-vector convention), both atomic positions and the cell
        transform together::

            r_n → r_n @ F
            cell_b → cell_b @ F

        A per-graph strain leaf ``F_b`` (initialised to ``I``) is applied
        to both, and a single ``autograd.grad`` call then yields:

        * ``dE/d(pos_leaf[n])`` at ``F=I`` → forces (negated).
        * ``dE/d(F_b)`` at ``F=I`` → canonical virial ``W_b = −dE/dF_b``.

        This is the correct formulation for position-dependent biases that
        use MIC displacements: the position term ``Σ_n r_n ⊗ (−F_n)`` and
        the cell gradient term are automatically combined.  Using an
        independent cell leaf (without straining positions) misses the
        position contribution and returns incorrect virials for pair
        restraints across image boundaries.
        """
        has_cell = (
            self._supports_virial
            and getattr(current, "cell", None) is not None
            and current.cell is not None
        )

        B = current.num_graphs
        original_positions = current.positions
        original_cell = current.cell if has_cell else None

        with torch.enable_grad():
            # --- Positions leaf (for forces) --------------------------------
            # requires_grad_() is not supported by torch.compile; evaluate()
            # is intentionally kept eager (see class docstring).
            pos_leaf = current.positions.detach().requires_grad_(True)  # [N, 3]

            # --- Per-graph strain leaf (for canonical virial) ---------------
            # Initialised to the identity; both positions and cell are
            # right-multiplied by F_b so that dE/dF_b|_{F=I} = −W_b.
            strain_leaf: Tensor | None = None
            pos_for_energy: Tensor = pos_leaf

            if has_cell:
                cell = original_cell
                if cell is not None and cell.dim() == 4:
                    cell = cell.squeeze(1)  # [B, 3, 3]

                strain_leaf = (
                    torch.eye(3, device=pos_leaf.device, dtype=pos_leaf.dtype)
                    .unsqueeze(0)
                    .expand(B, -1, -1)
                    .clone()
                    .requires_grad_(True)
                )  # [B, 3, 3]

                # Apply per-graph strain to each atom's position:
                # pos_n → pos_n @ F_{b(n)}
                strain_per_atom = strain_leaf[current.batch_idx]  # [N, 3, 3]
                pos_for_energy = torch.einsum(
                    "nk,nkj->nj", pos_leaf, strain_per_atom
                )  # [N, 3]

                # Apply per-graph strain to the cell:  cell_b → cell_b @ F_b
                # Detach the stored cell values; only F carries the gradient.
                cell_for_energy = torch.bmm(cell.detach(), strain_leaf)  # [B, 3, 3]

            try:
                current["positions"] = pos_for_energy
                if has_cell and strain_leaf is not None:
                    # Store in the same shape as the original cell tensor.
                    stored = cell_for_energy  # type: ignore[possibly-undefined]
                    if original_cell is not None and original_cell.dim() == 4:
                        stored = stored.unsqueeze(1)
                    current["cell"] = stored

                bias_energy: Tensor = self.energy(current)  # [B, 1]

                inputs: list[Tensor] = [pos_leaf]
                if strain_leaf is not None:
                    inputs.append(strain_leaf)

                grads = torch.autograd.grad(
                    outputs=(bias_energy,),
                    inputs=inputs,
                    grad_outputs=(torch.ones_like(bias_energy),),
                    create_graph=False,
                    retain_graph=False,
                    allow_unused=False,
                )

            finally:
                current["positions"] = original_positions
                if has_cell and original_cell is not None:
                    current["cell"] = original_cell

        # grads[0] = dE/d(pos_leaf) at strain=I → forces = −grad.
        forces = -grads[0].detach()  # [N, 3]

        virial: Tensor | None = None
        if strain_leaf is not None and len(grads) > 1 and grads[1] is not None:
            # grads[1] = dE/dF_b at F=I; canonical virial W_b = −dE/dF_b.
            virial = -grads[1].detach()  # [B, 3, 3]

        return BiasResult(
            energy=bias_energy.detach(),
            forces=forces,
            virial=virial,
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
    * ``stress`` and ``virial`` are accumulated separately.
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
