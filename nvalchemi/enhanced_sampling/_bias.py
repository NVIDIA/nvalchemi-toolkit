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
    """Eager-only validation of a ``BiasResult`` (skipped under compile)."""
    if result.stress is not None and result.virial is not None:
        raise ValueError("BiasResult: provide either 'stress' or 'virial', not both.")
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
        """Compute energy, forces, and (optionally) virial via autograd.

        This method runs in eager mode.  See class docstring for the
        compile boundary note and the chosen fallback.
        """
        has_cell = (
            self._supports_virial
            and getattr(current, "cell", None) is not None
            and current.cell is not None
        )

        with torch.enable_grad():
            # Create isolated autograd leaves.
            # requires_grad_() is not supported by torch.compile — this method
            # is intentionally kept eager (see class docstring).
            pos_leaf = current.positions.detach().requires_grad_(True)

            cell_leaf: Tensor | None = None
            if has_cell:
                cell_leaf = current.cell.detach().requires_grad_(True)

            # Temporarily replace positions (and cell) on the live batch with
            # the fresh leaves so that self.energy() can access other batch
            # fields (atomic_numbers, batch_idx, etc.) normally.
            # Restored unconditionally in the finally block.
            original_positions = current.positions
            original_cell = current.cell if has_cell else None

            try:
                current["positions"] = pos_leaf
                if has_cell and cell_leaf is not None:
                    current["cell"] = cell_leaf

                bias_energy: Tensor = self.energy(current)  # [B, 1]

                grad_outputs = (torch.ones_like(bias_energy),)
                inputs: tuple[Tensor, ...] = (
                    (pos_leaf,) if cell_leaf is None else (pos_leaf, cell_leaf)
                )

                grads = torch.autograd.grad(
                    outputs=(bias_energy,),
                    inputs=inputs,
                    grad_outputs=grad_outputs,
                    create_graph=False,
                    retain_graph=False,
                    allow_unused=False,
                )

            finally:
                current["positions"] = original_positions
                if has_cell and original_cell is not None:
                    current["cell"] = original_cell

        # grads[0]: d(sum(energy))/d(positions) — negate for forces.
        forces = -grads[0].detach()  # [N_atoms, 3]

        virial: Tensor | None = None
        if has_cell and len(grads) > 1 and grads[1] is not None:
            # Canonical virial W = −dE/d(strain); for the row-vector
            # convention (ASE): W = −(dE/d(cell)) @ cell.T
            dcell = grads[1].detach()
            if dcell.dim() == 4:
                dcell = dcell.squeeze(1)
            cell = original_cell
            if cell is not None and cell.dim() == 4:
                cell = cell.squeeze(1)
            if cell is not None:
                virial = -(dcell @ cell.transpose(-1, -2))  # [B, 3, 3]

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
