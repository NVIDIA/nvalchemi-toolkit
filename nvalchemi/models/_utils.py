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
"""Building blocks for model composition, and the additive-contribution contract.

Two groups of functions live here.

The **numerical helpers** — :func:`prepare_strain`, :func:`apply_strain`, the
``autograd_*`` family, :func:`cell_cache_needs_update` — are standalone
building blocks for users who need control beyond what
:class:`~nvalchemi.models.pipeline.PipelineModelWrapper` offers.  They are also
used internally by the pipeline.

The **contribution helpers** define what it means to produce
:data:`~nvalchemi._typing.ModelOutputs` as an *additive contribution* to
someone else's batch, rather than as a model's own product:

* :func:`isolated_energy_derivatives` — differentiate an energy against a
  detached view of a live ``Batch``, restoring every field it touched.
* :func:`validate_contribution` — check a mapping against the ``ModelOutputs``
  conventions before a consumer applies it.
* :func:`sum_outputs` — permissive element-wise sum; non-additive collisions
  resolve last-write-wins.
* :func:`aggregate_contributions` — the strict counterpart, where a collision
  that would drop a producer's value is an error.

They are deliberately not specific to any one producer.  An enhanced-sampling
bias, an NEB spring term, a hand-written wall potential and a dispersion
correction all contribute the same way and need the same guarantees.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import TYPE_CHECKING, Callable

import torch

from nvalchemi._typing import (
    BatchIndices,
    Energy,
    Forces,
    LatticeVectors,
    ModelOutputs,
    NodePositions,
    StrainDisplacement,
    Stress,
)

if TYPE_CHECKING:
    from nvalchemi.data import Batch

__all__ = [
    "APPLIED_OUTPUT_KEYS",
    "DIAGNOSTIC_PREFIX",
    "STATE_VERSION_KEY",
    "aggregate_contributions",
    "apply_strain",
    "autograd_forces",
    "autograd_forces_and_stresses",
    "autograd_stresses",
    "cell_cache_needs_update",
    "isolated_energy_derivatives",
    "prepare_strain",
    "sum_outputs",
    "validate_contribution",
]

#: Key prefix for reported (never summed, never applied) tensors in a
#: :data:`~nvalchemi._typing.ModelOutputs` mapping.
DIAGNOSTIC_PREFIX = "diagnostics/"

#: Key holding integer state-revision IDs, shape ``[B]``, for producers whose
#: internal state evolves during a run.
STATE_VERSION_KEY = "state_version"

#: Physical outputs an additive *contribution* may carry, and the whole of
#: them.  A closed set by design, not by omission: an applied output needs a
#: destination buffer, a per-graph or per-atom reshape rule, a combination
#: rule, and a unit convention, so an open payload would be open only up to
#: the first key a consumer could not apply.  ``ModelOutputs`` itself is
#: wider — a full forward pass may report ``hessian`` or ``dipole`` — but a
#: contribution is narrower than a forward pass precisely because it gets
#: added into a buffer.
APPLIED_OUTPUT_KEYS = ("energy", "forces", "stress", "virial")

_INTEGER_DTYPES = (
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.uint8,
)


def cell_cache_needs_update(
    cell: LatticeVectors,
    cached_cell: LatticeVectors | None,
    rtol: float = 1e-5,
    atol: float | None = None,
) -> bool:
    """Return ``True`` when ``cell`` is incompatible with ``cached_cell``.

    Parameters
    ----------
    cell : torch.Tensor
        Current cell tensor.
    cached_cell : torch.Tensor | None
        Previously cached cell tensor, or ``None`` when no cell has been
        cached yet.
    rtol : float, optional
        Relative tolerance passed to :func:`torch.allclose`.
        Defaults to ``1e-5``.
    atol : float or None, optional
        Absolute tolerance passed to :func:`torch.allclose`.
        When ``None`` (default), uses
        ``max(1e-6, torch.finfo(cell.dtype).eps)``.

    Returns
    -------
    bool
        ``True`` if the cache should be refreshed.
    """
    if atol is None:
        atol = max(1e-6, torch.finfo(cell.dtype).eps)

    if cached_cell is None or cell.shape != cached_cell.shape:
        return True
    # Check device and dtype for compatibility with torch.allclose
    if cell.device != cached_cell.device:
        return True
    if cell.dtype != cached_cell.dtype:
        return True
    if not torch.allclose(cell, cached_cell, rtol=rtol, atol=atol):
        return True
    return False


def autograd_forces(
    energy: Energy,
    positions: NodePositions,
    training: bool = False,
    retain_graph: bool = False,
    allow_unused: bool = False,
) -> Forces:
    """Compute forces as ``-dE/dr`` via autograd.

    Parameters
    ----------
    energy : torch.Tensor
        Total energy tensor (must be part of a computation graph that
        includes *positions*).
    positions : torch.Tensor
        Atomic positions with ``requires_grad=True``.
    training : bool, optional
        If ``True``, ``create_graph=True`` is set so that higher-order
        gradients are available (needed for training).
    retain_graph : bool, optional
        If ``True``, the computation graph is retained after the backward
        pass.  Needed when subsequent autograd calls traverse shared
        graph nodes.
    allow_unused : bool, optional
        If ``True``, an energy that does not depend on *positions* yields
        zero forces instead of raising ``RuntimeError``.  Use this for
        energy terms that are legitimately position-independent, such as a
        pure cell/volume term.  Defaults to ``False``, which surfaces a
        missing dependency as an error.

    Returns
    -------
    torch.Tensor
        Forces tensor with same shape as *positions*.
    """
    effective_retain = retain_graph or training
    return -torch.autograd.grad(
        energy,
        positions,
        grad_outputs=torch.ones_like(energy),
        create_graph=training,
        retain_graph=effective_retain,
        allow_unused=allow_unused,
        materialize_grads=allow_unused,
    )[0]


def prepare_strain(
    positions: NodePositions,
    cell: LatticeVectors,
    batch_idx: BatchIndices,
) -> tuple[NodePositions, LatticeVectors, StrainDisplacement]:
    """Set up the affine strain trick for autograd stress computation.

    Creates a per-system 3x3 displacement tensor with
    ``requires_grad=True``, scales positions and cell through the symmetric
    part of it, and returns all three tensors.  After running the model on
    the scaled positions/cell, compute stresses with standard PyTorch
    autograd::

        scaled_pos, scaled_cell, displacement = prepare_strain(
            positions, cell, batch_idx
        )
        energy = model(scaled_pos, scaled_cell, ...)

        # Forces:
        forces = -torch.autograd.grad(
            energy, scaled_pos, torch.ones_like(energy),
            retain_graph=True,
        )[0]

        # Stresses:
        grad = torch.autograd.grad(
            energy, displacement, torch.ones_like(energy),
        )[0]
        volume = torch.det(cell).abs().view(-1, 1, 1)
        stresses = grad.view(B, 3, 3) / volume

    This function is used internally by :class:`PipelineModelWrapper`
    for autograd groups, and is available for users who want to
    implement autograd stresses in their own model wrappers.

    Parameters
    ----------
    positions : torch.Tensor
        Atomic positions, shape ``[N, 3]``.
    cell : torch.Tensor
        Unit cell, shape ``[B, 3, 3]``.
    batch_idx : torch.Tensor
        Graph index per atom, shape ``[N]``.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        ``(scaled_positions, scaled_cell, displacement)`` where
        ``displacement`` is ``[B, 3, 3]`` with ``requires_grad=True``.
        The returned tensor is an unconstrained autograd leaf; only its
        symmetric part is applied as strain.
    """
    n_systems = cell.shape[0]
    displacement = torch.zeros(
        n_systems,
        3,
        3,
        dtype=positions.dtype,
        device=positions.device,
    )
    displacement.requires_grad_(True)
    scaled_positions, scaled_cell = apply_strain(
        positions, cell, batch_idx, displacement
    )
    return scaled_positions, scaled_cell, displacement


def apply_strain(
    positions: NodePositions,
    cell: LatticeVectors,
    batch_idx: BatchIndices,
    displacement: StrainDisplacement,
    cell_displacement: "StrainDisplacement | None" = None,
) -> tuple[NodePositions, LatticeVectors]:
    """Scale positions and cell through an existing strain leaf.

    The half of :func:`prepare_strain` that does the deformation, split out for
    callers that must strain against a leaf they already hold — a distributed
    forward reapplies the same strain after every halo refresh, and a composed
    pipeline shares one leaf across its models.

    Parameters
    ----------
    positions : torch.Tensor
        Atomic positions, shape ``[N, 3]``.
    cell : torch.Tensor
        Unit cell, shape ``[B, 3, 3]``.
    batch_idx : torch.Tensor
        Graph index per atom, shape ``[N]``.
    displacement : torch.Tensor
        Strain leaf, shape ``[B, 3, 3]``. Only its symmetric part is applied.
    cell_displacement : torch.Tensor, optional
        Separate leaf for the cell, when the position and cell halves of the
        virial are read separately. Defaults to ``displacement``.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        ``(scaled_positions, scaled_cell)``.
    """
    eye = torch.eye(3, dtype=positions.dtype, device=positions.device)
    deformation = eye + 0.5 * (displacement + displacement.mT)
    if cell_displacement is None:
        cell_deformation = deformation
    else:
        cell_deformation = eye + 0.5 * (cell_displacement + cell_displacement.mT)
    scaled_positions = torch.einsum("ni,nij->nj", positions, deformation[batch_idx])
    scaled_cell = torch.einsum("bij,bjk->bik", cell, cell_deformation)
    return scaled_positions, scaled_cell


def autograd_stresses(
    energy: Energy,
    displacement: StrainDisplacement,
    cell: LatticeVectors,
    num_graphs: int,
    training: bool = False,
    retain_graph: bool = False,
    allow_unused: bool = False,
) -> Stress:
    r"""Compute tensile-positive Cauchy stress via autograd.

    Returns ``1/V * dE/d(strain)`` in :math:`\mathrm{eV}/\mathrm{\AA}^3`.

    Parameters
    ----------
    energy : torch.Tensor
        Total energy tensor.
    displacement : torch.Tensor
        Displacement tensor (symmetric strain applied to positions).
    cell : torch.Tensor
        Unit cell tensor of shape ``[B, 3, 3]``.
    num_graphs : int
        Number of graphs (systems) in the batch.
    training : bool, optional
        If ``True``, create the computation graph for higher-order gradients.
    retain_graph : bool, optional
        If ``True``, retain the computation graph.
    allow_unused : bool, optional
        If ``True``, an energy that does not depend on *displacement* yields
        zero stress instead of raising ``RuntimeError``.  Defaults to
        ``False``, which surfaces a missing dependency as an error.

    Returns
    -------
    torch.Tensor
        Cauchy stress tensor of shape ``[B, 3, 3]`` in :math:`\mathrm{eV}/\mathrm{\AA}^3`.
    """
    effective_retain = retain_graph or training
    grad = torch.autograd.grad(
        energy,
        displacement,
        grad_outputs=torch.ones_like(energy),
        create_graph=training,
        retain_graph=effective_retain,
        allow_unused=allow_unused,
        materialize_grads=allow_unused,
    )[0]
    volume = torch.det(cell).abs().view(-1, 1, 1)
    return grad.view(num_graphs, 3, 3) / volume


def autograd_forces_and_stresses(
    energy: Energy,
    positions: NodePositions,
    displacement: StrainDisplacement,
    cell: LatticeVectors,
    num_graphs: int,
    training: bool = False,
    retain_graph: bool = False,
    allow_unused: bool = False,
) -> tuple[Forces, Stress]:
    """Compute forces and tensile-positive Cauchy stress in one autograd call.

    Parameters
    ----------
    energy : torch.Tensor
        Total energy tensor.
    positions : torch.Tensor
        Atomic positions with ``requires_grad=True``.
    displacement : torch.Tensor
        Displacement tensor from :func:`prepare_strain`.
    cell : torch.Tensor
        Original unit cell tensor of shape ``[B, 3, 3]``.
    num_graphs : int
        Number of graphs (systems) in the batch.
    training : bool, optional
        If ``True``, create the computation graph for higher-order gradients.
    retain_graph : bool, optional
        If ``True``, retain the computation graph.
    allow_unused : bool, optional
        If ``True``, whichever of *positions* and *displacement* the energy
        does not depend on yields a zero gradient instead of raising
        ``RuntimeError``.  Use this for energy terms that are legitimately
        independent of one of the two, such as a pure cell/volume term that
        produces stress but no forces.  Defaults to ``False``, which
        surfaces a missing dependency as an error.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        ``(forces, stress)`` with shapes ``[N, 3]`` and ``[B, 3, 3]``.
    """
    effective_retain = retain_graph or training
    position_grad, displacement_grad = torch.autograd.grad(
        energy,
        (positions, displacement),
        grad_outputs=torch.ones_like(energy),
        create_graph=training,
        retain_graph=effective_retain,
        allow_unused=allow_unused,
        materialize_grads=allow_unused,
    )
    forces = -position_grad
    volume = torch.det(cell).abs().view(-1, 1, 1)
    stress = displacement_grad.view(num_graphs, 3, 3) / volume
    return forces, stress


def sum_outputs(
    *outputs: ModelOutputs,
    additive_keys: set[str] | None = None,
) -> ModelOutputs:
    """Element-wise sum of :class:`ModelOutputs` on specified keys.

    Keys in *additive_keys* are summed across all *outputs*.
    Non-additive keys use last-write-wins semantics.

    Parameters
    ----------
    *outputs : ModelOutputs
        One or more model output dicts to combine.
    additive_keys : set[str] | None, optional
        Keys whose values should be summed.  Defaults to
        ``{"energy", "forces", "stress"}``.

    Returns
    -------
    ModelOutputs
        Combined output dict.
    """
    additive = additive_keys or {"energy", "forces", "stress"}
    result: ModelOutputs = OrderedDict()
    for out in outputs:
        for key, val in out.items():
            if val is None:
                continue
            if key in additive and key in result and result[key] is not None:
                result[key] = result[key] + val
            else:
                result[key] = val
    return result


def _unsupported_contribution_keys(outputs: ModelOutputs) -> list[str]:
    """Return the keys in *outputs* that no consumer can act on.

    The single definition of "unsupported", so the producer-side check and
    the aggregation-side one cannot disagree about what they are refusing.

    Parameters
    ----------
    outputs : ModelOutputs
        The mapping to inspect.

    Returns
    -------
    list[str]
        Offending keys, sorted; empty when there are none.
    """
    return sorted(
        key
        for key in outputs
        if key not in APPLIED_OUTPUT_KEYS
        and key != STATE_VERSION_KEY
        and not key.startswith(DIAGNOSTIC_PREFIX)
    )


def _unsupported_key_advice() -> str:
    """Return the shared tail of an unsupported-key error.

    Returns
    -------
    str
        What a contribution may carry, why the set is closed, and the two
        ways out.
    """
    return (
        f"A contribution carries {list(APPLIED_OUTPUT_KEYS)}, "
        f"{STATE_VERSION_KEY!r}, and {DIAGNOSTIC_PREFIX}<key> entries — the "
        "applied set is closed because each member needs a destination "
        "buffer, a reshape rule and a combination rule. Report it instead as "
        f"'{DIAGNOSTIC_PREFIX}<key>', which is carried through and never "
        "summed, or add the output to the framework so a consumer knows "
        "where to put it."
    )


def validate_contribution(
    outputs: ModelOutputs, *, source: str = "ModelOutputs"
) -> None:
    """Check a :data:`~nvalchemi._typing.ModelOutputs` against its conventions.

    Written for *contributions*: outputs a consumer will add into a specific
    buffer, where a malformed entry is applied silently rather than raised.
    Any producer of additive outputs can call it — a bias, a dispersion
    correction, an NEB spring term.

    Checks, in order:

    1. Every key is one a consumer can act on: a member of
       :data:`APPLIED_OUTPUT_KEYS`, :data:`STATE_VERSION_KEY`, or something
       under :data:`DIAGNOSTIC_PREFIX`.  Anything else is refused rather than
       dropped — :func:`aggregate_contributions` keeps only the applied keys,
       so an unrecognised physical output would otherwise vanish between the
       producer and the buffer with the run carrying on as though it had been
       applied.
    2. ``stress`` and ``virial`` are mutually exclusive.  They are the same
       physics in two conventions and converting between them needs the cell
       volume, so carrying both invites two answers that can disagree.
    3. Every tensor is detached (``requires_grad=False`` and
       ``grad_fn is None``).  A live graph reaching a batch buffer, a retained
       history, or a checkpoint keeps the whole forward graph alive.
    4. Shapes match the documented conventions, for the keys that have one:

       * ``energy`` — ndim 2, shape ``[B, 1]``
       * ``forces`` — ndim 2, shape ``[N, 3]``
       * ``stress`` / ``virial`` — ndim 3, shape ``[B, 3, 3]``
       * ``state_version`` — ndim 1, integer dtype

       Keys under :data:`DIAGNOSTIC_PREFIX` have no shape contract and are
       skipped here; they are reported, never applied.
    5. All per-graph fields agree on the leading dimension ``B``.
    6. Every floating-point tensor, diagnostics included, is finite.

    Parameters
    ----------
    outputs : ModelOutputs
        The mapping to check.
    source : str, optional
        Name used in error messages to identify the producer, e.g.
        ``"HarmonicUmbrellaBias 'umbrella'"``.  Defaults to
        ``"ModelOutputs"``.

    Raises
    ------
    ValueError
        On the first violation, naming the key and what is wrong with it.
    """
    unsupported = _unsupported_contribution_keys(outputs)
    if unsupported:
        raise ValueError(
            f"{source}: {unsupported} cannot be applied to a batch. "
            f"{_unsupported_key_advice()}"
        )

    if outputs.get("stress") is not None and outputs.get("virial") is not None:
        raise ValueError(
            f"{source}: provide either 'stress' or 'virial', not both. "
            "They are the same physics in two conventions; converting "
            "between them requires the cell volume."
        )

    for key, value in outputs.items():
        if value is None:
            continue
        if value.requires_grad:
            raise ValueError(
                f"{source}[{key!r}] must be detached (requires_grad=False), "
                f"got requires_grad=True."
            )
        if value.grad_fn is not None:
            raise ValueError(
                f"{source}[{key!r}] must be detached (grad_fn is None), got "
                f"grad_fn={value.grad_fn}."
            )

    energy = outputs.get("energy")
    if energy is not None and (energy.ndim != 2 or energy.shape[1] != 1):
        raise ValueError(
            f"{source}['energy'] must have shape [B, 1], got {tuple(energy.shape)}."
        )

    forces = outputs.get("forces")
    if forces is not None and (forces.ndim != 2 or forces.shape[1] != 3):
        raise ValueError(
            f"{source}['forces'] must have shape [N, 3], got {tuple(forces.shape)}."
        )

    for key in ("stress", "virial"):
        value = outputs.get(key)
        if value is not None and (
            value.ndim != 3 or value.shape[1] != 3 or value.shape[2] != 3
        ):
            raise ValueError(
                f"{source}[{key!r}] must have shape [B, 3, 3], got "
                f"{tuple(value.shape)}."
            )

    version = outputs.get(STATE_VERSION_KEY)
    if version is not None:
        if version.ndim != 1:
            raise ValueError(
                f"{source}[{STATE_VERSION_KEY!r}] must have shape [B], got "
                f"{tuple(version.shape)}."
            )
        if version.dtype not in _INTEGER_DTYPES:
            raise ValueError(
                f"{source}[{STATE_VERSION_KEY!r}] must be an integer dtype "
                f"(it is compared for identity, which a float does not "
                f"support), got {version.dtype}."
            )

    batch_sizes = {
        key: outputs[key].shape[0]
        for key in ("energy", "stress", "virial", STATE_VERSION_KEY)
        if outputs.get(key) is not None
    }
    if len(set(batch_sizes.values())) > 1:
        raise ValueError(
            f"{source}: leading batch dimension B is inconsistent across "
            f"fields: {batch_sizes}."
        )

    for key, value in outputs.items():
        if value is None or not value.is_floating_point():
            continue
        if not value.isfinite().all():
            raise ValueError(f"{source}[{key!r}] contains NaN or Inf values.")


def isolated_energy_derivatives(
    energy_fn: Callable[[Batch], Energy],
    batch: Batch,
    *,
    want_forces: bool = True,
    want_stress: bool = True,
    allow_unused: bool = False,
) -> ModelOutputs:
    r"""Differentiate *energy_fn* against a detached view of *batch*.

    Evaluates ``energy_fn`` on positions (and, for a periodic batch, a cell)
    that carry a private autograd graph, derives forces and tensile-positive
    Cauchy stress from it, and returns everything detached.  Every batch field
    it substitutes is restored in a ``finally`` block, so the caller's
    ``Batch`` never carries a ``grad_fn`` after this returns and no
    ``requires_grad`` leaf escapes into batch storage, retained history, or a
    checkpoint.

    That isolation is the reason this exists as a shared helper rather than as
    a method on any one class: it is what anything computing derivatives
    against a *live* batch needs — an enhanced-sampling bias, an NEB spring
    term, a hand-written wall potential.  The numerical half is
    :func:`autograd_forces_and_stresses`, which model wrappers already call
    directly when they own their own graph.

    Stress derivation
    -----------------
    Under a homogeneous strain :math:`\varepsilon` (ASE row-vector
    convention) atomic positions and the cell deform together::

        r_n    -> r_n    @ (I + eps)
        cell_b -> cell_b @ (I + eps)

    :func:`prepare_strain` applies exactly this through a leaf whose
    *symmetric* part is used, so one ``autograd.grad`` call yields forces from
    the position leaf and :math:`\sigma = (dE/d\varepsilon)/V` from the strain
    leaf.  Straining positions and cell together is what makes this correct
    for energies built on minimum-image displacements: a strain leaf applied
    to the cell alone misses the position contribution and gives the wrong
    answer for a pair term spanning an image boundary.

    Parameters
    ----------
    energy_fn : Callable[[Batch], Energy]
        Returns a differentiable per-graph energy ``[B, 1]`` for the batch it
        is handed.  It receives a *read-only view* of *batch* whose
        ``positions`` and ``cell`` have been replaced by their strained
        counterparts; it must not assign to any batch field.
    batch : Batch
        The live batch.  Restored to its original ``positions`` and ``cell``
        before this returns, including on an exception.
    want_forces : bool, optional
        Compute ``forces = -dE/dr``.  Defaults to ``True``.
    want_stress : bool, optional
        Compute ``stress``.  Defaults to ``True``.  Stress additionally
        requires a cell and at least one periodic dimension: a placeholder
        cell with ``pbc`` all-``False`` has zero volume, which would divide
        the stress to infinity.
    allow_unused : bool, optional
        When ``True``, an energy that does not depend on positions or on
        strain yields a zero gradient for that input instead of raising.  Use
        it for terms that are legitimately independent of one — a pure volume
        restraint produces stress but no forces, and its zero force is an
        answer rather than an error.  Defaults to ``False``.

    Returns
    -------
    ModelOutputs
        ``energy`` always; ``forces`` when *want_forces*; ``stress`` when
        *want_stress* and the batch is periodic.  All entries detached.

    Notes
    -----
    Eager only.  The positions leaf is made with
    ``positions.detach().requires_grad_(True)``, which ``torch.compile``
    rejects (``Unsupported Tensor.requires_grad_() call``), and the
    periodicity test is a data-dependent branch.  Compile *energy_fn* itself —
    the hot path — and leave this as the eager orchestration around it.
    """
    original_positions = batch.positions
    original_cell = getattr(batch, "cell", None)
    pbc = getattr(batch, "pbc", None)
    cell_is_4d = original_cell is not None and original_cell.dim() == 4

    strain_cell: torch.Tensor | None = None
    if want_stress and original_cell is not None and (pbc is None or bool(pbc.any())):
        # Detach the stored cell so only the strain leaf carries the gradient;
        # a [B, 1, 3, 3] cell is squeezed to [B, 3, 3] for prepare_strain.
        strain_cell = original_cell.detach()
        if cell_is_4d:
            strain_cell = strain_cell.squeeze(1)

    with torch.enable_grad():
        pos_leaf = original_positions.detach()
        if want_forces:
            pos_leaf = pos_leaf.requires_grad_(True)  # [N, 3]

        displacement: torch.Tensor | None = None
        pos_for_energy = pos_leaf
        cell_for_energy: torch.Tensor | None = None
        if strain_cell is not None:
            pos_for_energy, cell_for_energy, displacement = prepare_strain(
                pos_leaf, strain_cell, batch.batch_idx
            )

        try:
            batch["positions"] = pos_for_energy
            if cell_for_energy is not None:
                batch["cell"] = (
                    cell_for_energy.unsqueeze(1) if cell_is_4d else cell_for_energy
                )

            energy = energy_fn(batch)  # [B, 1]

            forces: torch.Tensor | None = None
            stress: torch.Tensor | None = None
            if not energy.requires_grad:
                # An energy with no graph at all — a term returning a constant
                # on this branch. Every gradient is zero, but autograd rejects
                # such an output outright ("does not require grad and does not
                # have a grad_fn"), so fill the zeros directly.
                if want_forces:
                    forces = torch.zeros_like(pos_leaf)
                if displacement is not None:
                    stress = torch.zeros_like(strain_cell)
            elif want_forces and displacement is not None:
                forces, stress = autograd_forces_and_stresses(
                    energy,
                    pos_leaf,
                    displacement,
                    strain_cell,
                    batch.num_graphs,
                    allow_unused=allow_unused,
                )
            elif want_forces:
                forces = autograd_forces(energy, pos_leaf, allow_unused=allow_unused)
            elif displacement is not None:
                stress = autograd_stresses(
                    energy,
                    displacement,
                    strain_cell,
                    batch.num_graphs,
                    allow_unused=allow_unused,
                )
        finally:
            batch["positions"] = original_positions
            if original_cell is not None:
                batch["cell"] = original_cell

    outputs: ModelOutputs = OrderedDict()
    outputs["energy"] = energy.detach()
    if forces is not None:
        outputs["forces"] = forces.detach()
    if stress is not None:
        outputs["stress"] = stress.detach()
    return outputs


def aggregate_contributions(contributions: list[ModelOutputs]) -> ModelOutputs:
    """Sum several additive contributions into one, strictly.

    The strict counterpart to :func:`sum_outputs`: same element-wise sum, but
    a key collision that would silently drop a producer's value is an error
    rather than last-write-wins.  Use it where several producers contribute to
    one quantity and each contribution must survive — several biases on one
    trajectory, several correction terms on one energy — and the caller
    evaluates all of them against the *same* unmodified state, so no producer
    observes another's contribution and the total does not depend on ordering.

    Rules
    -----
    * Every key must be one a consumer can act on: a member of
      :data:`APPLIED_OUTPUT_KEYS`, :data:`STATE_VERSION_KEY`, or something
      under :data:`DIAGNOSTIC_PREFIX`.  Anything else raises, naming the
      contribution — the physics sum below keeps only the applied keys, so a
      novel output left to reach it would be dropped in silence, which is the
      one thing this function exists not to do.
    * Missing and ``None`` entries are skipped — a zero contribution.
    * Every contribution that carries a cell response must use the **same**
      field: all ``stress`` or all ``virial``, never a mix.  Mixing raises
      ``ValueError`` identifying which indices supplied each.  Converting
      between the two needs the cell volume and is the caller's job.
    * ``diagnostics/*`` entries are merged, not summed; a duplicate key raises
      rather than silently dropping one producer's value, so namespacing must
      be applied beforehand.
    * ``state_version`` is dropped.  It identifies one producer's state
      revision; an aggregate over several producers has no single revision,
      and last-write-wins would name one of them arbitrarily.  Per-producer
      versions survive in each contribution and should be recorded there.

    Relationship to :func:`sum_outputs`
    -----------------------------------
    The element-wise tensor sum is delegated to ``sum_outputs``.  The rules
    above are a separate function rather than a flag on it, because they are
    stricter than what ``sum_outputs`` can offer its own callers:

    * ``sum_outputs`` resolves a non-additive key collision by
      last-write-wins, which the model pipeline depends on (two composed
      models may both emit ``charges``).  Silently dropping one producer's
      diagnostic is not acceptable, so the collision is an error here.
    * Model wrappers normalise a virial to a tensile-positive stress at the
      adapter boundary, so the outputs that reach ``sum_outputs`` carry
      ``stress`` and never ``virial`` — which is why its default
      ``additive_keys`` omits virial entirely.  A hand-written additive term
      may legitimately produce either, so only this layer can be handed both.

    Parameters
    ----------
    contributions:
        One mapping per producer.  May be empty, in which case an empty
        mapping is returned.

    Returns
    -------
    ModelOutputs
        The summed contribution.

    Raises
    ------
    ValueError
        If ``stress`` and ``virial`` are mixed across contributions, or if two
        contributions supply the same ``diagnostics/*`` key.
    """
    if not contributions:
        return OrderedDict()

    # Refused here and not only in validate_contribution, because this
    # function promises that nothing is dropped silently and it is reachable
    # without that check: it is public, and its callers are whoever composes
    # additive terms next. The filter below keeps only APPLIED_OUTPUT_KEYS,
    # so an unsupported key left to reach it vanishes between the producer
    # and the buffer with the caller none the wiser.
    for index, contribution in enumerate(contributions):
        unsupported = _unsupported_contribution_keys(contribution)
        if unsupported:
            raise ValueError(
                f"aggregate_contributions: contributions[{index}] carries "
                f"{unsupported}, which cannot be applied to a batch. "
                f"{_unsupported_key_advice()}"
            )

    # Detect stress/virial mixing up-front so the error names the offending
    # contributions, rather than surfacing later as a generic mutual-exclusion
    # failure with nothing to point at.
    stress_indices = [
        i for i, c in enumerate(contributions) if c.get("stress") is not None
    ]
    virial_indices = [
        i for i, c in enumerate(contributions) if c.get("virial") is not None
    ]
    if stress_indices and virial_indices:
        raise ValueError(
            f"aggregate_contributions: contributions[{stress_indices}] provide "
            f"'stress' and contributions[{virial_indices}] provide 'virial' — "
            "cannot mix both in the same aggregation.  Make all producers "
            "return the same field.  Converting stress to virial requires "
            "the cell volume and is the caller's responsibility before "
            "aggregation."
        )

    physics = sum_outputs(
        *(
            OrderedDict(
                (key, value)
                for key, value in contribution.items()
                if key in APPLIED_OUTPUT_KEYS
            )
            for contribution in contributions
        ),
        additive_keys=set(APPLIED_OUTPUT_KEYS),
    )

    # Only physical and diagnostic keys are copied forward, which is how
    # state_version is dropped: see the docstring for why an aggregate has no
    # single revision to report.
    merged: ModelOutputs = OrderedDict(physics)
    source: dict[str, int] = {}
    for i, contribution in enumerate(contributions):
        for key, value in contribution.items():
            if not key.startswith(DIAGNOSTIC_PREFIX) or value is None:
                continue
            if key in merged:
                raise ValueError(
                    f"aggregate_contributions: duplicate diagnostic key {key!r} "
                    f"from contributions[{source[key]}] and contributions[{i}]. "
                    f"Apply '{DIAGNOSTIC_PREFIX}bias/<name>/<key>' namespacing "
                    "before aggregation."
                )
            merged[key] = value
            source[key] = i
    return merged
