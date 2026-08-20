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
"""xTB-style Cartesian RMSD metadynamics for conformer exploration."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import torch
from torch import Tensor

from nvalchemi.enhanced_sampling._adaptive import AdaptivePotentialMixin
from nvalchemi.enhanced_sampling._bias import ConservativeBias

if TYPE_CHECKING:
    from collections.abc import Mapping

    from nvalchemi.data import Batch
    from nvalchemi.enhanced_sampling._bias import BiasResult

__all__ = ["RMSDMetaDynamicsBias"]

_HISTORY_MODES = ("shared", "state", "walker")
_STORAGE_POLICIES = ("preallocated", "grow", "fifo")


def _squared_rmsd(coords: Tensor, references: Tensor) -> Tensor:
    r"""Return the optimally aligned squared RMSD, shape ``[B, R]``.

    Alignment is solved by the quaternion characteristic-polynomial route
    (Coutsias, Seok & Dill 2004) rather than an SVD Kabsch: the maximum of
    :math:`\mathrm{tr}(\mathbf{R}\mathbf{H})` over proper rotations is the
    largest eigenvalue of a symmetric ``4 x 4`` key matrix built from the
    covariance ``H``.  Two properties matter here:

    * The proper-rotation constraint is built in.  An SVD needs an explicit
      ``det`` correction, and the sign flip is a non-differentiable branch on
      the compiled path.
    * Only an eigen*value* is needed, never an eigenvector or a rotation
      matrix.  Eigenvector gradients blow up when singular values are nearly
      degenerate, which is exactly what happens for symmetric-top and linear
      molecules; the largest eigenvalue stays simple.

    The result is the *squared* RMSD throughout.  ``sqrt`` has infinite
    derivative at zero, and a reference structure is visited at RMSD zero
    every time one is deposited, so taking the square root would put a force
    singularity at the most frequently sampled point of the run.

    Parameters
    ----------
    coords:
        Current coordinates, shape ``[B, M, 3]``.
    references:
        Reference coordinates, shape ``[R, M, 3]``.  Assumed already
        centered on their centroid.

    Returns
    -------
    Tensor
        Squared RMSD in ``A^2``, shape ``[B, R]``, clamped at zero.
    """
    n_atoms = coords.shape[1]
    centered = coords - coords.mean(dim=1, keepdim=True)  # [B, M, 3]

    # Inner products, both invariant to rotation.
    g_x = (centered**2).sum(dim=(1, 2))  # [B]
    g_y = (references**2).sum(dim=(1, 2))  # [R]

    # Covariance for every (walker, reference) pair.
    cov = torch.einsum("bmi,rmj->brij", centered, references)  # [B, R, 3, 3]
    xx, xy, xz = cov[..., 0, 0], cov[..., 0, 1], cov[..., 0, 2]
    yx, yy, yz = cov[..., 1, 0], cov[..., 1, 1], cov[..., 1, 2]
    zx, zy, zz = cov[..., 2, 0], cov[..., 2, 1], cov[..., 2, 2]

    key = torch.stack(
        [
            torch.stack([xx + yy + zz, yz - zy, zx - xz, xy - yx], dim=-1),
            torch.stack([yz - zy, xx - yy - zz, xy + yx, zx + xz], dim=-1),
            torch.stack([zx - xz, xy + yx, -xx + yy - zz, yz + zy], dim=-1),
            torch.stack([xy - yx, zx + xz, yz + zy, -xx - yy + zz], dim=-1),
        ],
        dim=-2,
    )  # [B, R, 4, 4]

    lambda_max = torch.linalg.eigvalsh(key)[..., -1]  # [B, R]
    msd = (g_x.unsqueeze(1) + g_y.unsqueeze(0) - 2.0 * lambda_max) / n_atoms
    # Exactly-aligned pairs land a rounding step below zero.
    return torch.clamp(msd, min=0.0)


class RMSDMetaDynamicsBias(AdaptivePotentialMixin, ConservativeBias):
    r"""Repulsive Gaussian bias over retained reference structures.

    .. math::

        V(x, t) = \sum_r f_r(t)\, k_\mathrm{push}
                  \exp\!\left(-\alpha\, \mathrm{RMSD}_A(x, x_r)^2\right)

    where :math:`\mathrm{RMSD}_A` is the Cartesian RMSD after optimal
    translation and rotation.  Each deposition appends the current geometry
    to the reference set, so the bias pushes the system away from everywhere
    it has already been.  This is the conformer-exploration scheme used by
    xTB/CREST metadynamics.

    Unlike :class:`~nvalchemi.enhanced_sampling.WellTemperedMetaDynamicsBias`,
    there is no collective variable to choose: the "CV" is the whole
    (selected) geometry, which is what makes it a general-purpose explorer
    for molecules whose interesting degrees of freedom are not known in
    advance.  The price is that it has no free-energy interpretation — it is
    a structure-generation method, not an estimator, and there is
    deliberately no ``free_energy`` here.

    Parameters
    ----------
    k_push:
        Per-reference bias amplitude in eV.  Must be positive; a negative
        amplitude would attract the system to structures it has already
        visited.
    alpha:
        Gaussian width parameter in ``A^-2``.  Larger is narrower.
    name:
        Unique bias identifier.
    atom_indices:
        Atom indices *within each graph* to align and compare, shape ``[M]``.
        ``None`` uses every atom.  The usual choice is heavy atoms only:
        hydrogens rotating on a methyl group produce RMSD the exploration
        does not care about.
    update_frequency:
        Dynamics steps between depositions.
    storage:
        ``"fifo"`` (default), ``"preallocated"``, or ``"grow"``.  FIFO is the
        xTB-compatible default and is not a compromise here the way it is for
        well-tempered metadynamics: with no free energy to reconstruct,
        discarding the oldest references is a deliberate choice to keep
        pushing outward rather than accumulating an ever-stiffer cage.
    max_references:
        Capacity.  Required for ``"preallocated"`` and ``"fifo"``; the
        initial chunk for ``"grow"``.
    history:
        ``"shared"``, ``"state"``, or ``"walker"``, as for well-tempered
        metadynamics.  The latter two require the batch to carry
        ``thermodynamic_state_id`` / ``walker_id`` and raise if it does not;
        :class:`~nvalchemi.enhanced_sampling.EnhancedSampling` stamps both on
        every step, so only direct evaluation has to supply them.
    ramp_depositions:
        Deposition events over which a new reference ramps from zero to full
        amplitude.  A freshly deposited reference is at RMSD zero, so without
        a ramp it switches on at its full value exactly where the system is
        standing, which is the worst possible place for a force
        discontinuity.  Defaults to ``1``.
    references:
        Optional warm start, shape ``[R, M, 3]``, in chronological order.
        The xTB convention of seeding with a displaced copy of the initial
        structure is expressed by passing it here.
    compute_stress:
        Passed through to :class:`ConservativeBias`.  Defaults to ``False``:
        this bias rejects periodic systems, so there is no cell to
        differentiate with respect to.

    Raises
    ------
    ValueError
        For a non-positive ``k_push`` or ``alpha``, an invalid storage
        policy or history mode, a missing capacity, a malformed
        ``atom_indices``, or warm-start references of the wrong shape.

    Notes
    -----
    Periodic systems are rejected
        Cartesian RMSD against a stored reference is not well defined under
        periodic boundary conditions: an atom that diffuses across a cell
        face is physically unmoved but Cartesian-displaced by a lattice
        vector, so the RMSD jumps and the bias delivers a large spurious
        force.  Making this correct needs a minimum-image-aware,
        correspondence-resolving metric.  Rather than return a plausible
        wrong number, :meth:`evaluate` raises.

        Periodicity is read from ``batch.pbc``, not from the presence of a
        cell.  A molecular batch carrying a **bounding box** with ``pbc``
        all-False is accepted, which is the common case for a solvated or
        boxed molecule; a slab (``pbc=[True, True, False]``) is rejected,
        since wrapping along any axis is enough to break the metric.  A
        non-zero cell with no ``pbc`` flags at all is refused as undeclared
        rather than assumed non-periodic.

    Fixed atom correspondence
        Atom ``i`` is always compared against atom ``i`` of the reference.
        No permutation search is performed, so two structures identical up to
        relabelling of equivalent atoms register as distinct.

    Examples
    --------
    >>> import torch
    >>> from nvalchemi.enhanced_sampling import RMSDMetaDynamicsBias
    >>> bias = RMSDMetaDynamicsBias(
    ...     k_push=0.02, alpha=0.5, max_references=50,
    ...     atom_indices=torch.tensor([0, 1, 2]),
    ... )
    >>> bias.reference_count.item()
    0
    """

    def __init__(
        self,
        k_push: float,
        alpha: float,
        *,
        name: str = "rmsd_metadynamics",
        atom_indices: Tensor | None = None,
        update_frequency: int = 500,
        storage: Literal["preallocated", "grow", "fifo"] = "fifo",
        max_references: int | None = None,
        history: Literal["shared", "state", "walker"] = "shared",
        ramp_depositions: int = 1,
        references: Tensor | None = None,
        compute_stress: bool = False,
    ) -> None:
        super().__init__(name=name, compute_stress=compute_stress)

        if storage not in _STORAGE_POLICIES:
            raise ValueError(
                f"RMSDMetaDynamicsBias: storage must be one of "
                f"{list(_STORAGE_POLICIES)}, got {storage!r}."
            )
        if history not in _HISTORY_MODES:
            raise ValueError(
                f"RMSDMetaDynamicsBias: history must be one of "
                f"{list(_HISTORY_MODES)}, got {history!r}."
            )
        if k_push <= 0.0:
            raise ValueError(
                f"RMSDMetaDynamicsBias: k_push must be positive, got {k_push}. "
                "A non-positive amplitude would attract the system back to "
                "structures it has already visited."
            )
        if alpha <= 0.0:
            raise ValueError(
                f"RMSDMetaDynamicsBias: alpha must be positive, got {alpha}."
            )
        if int(update_frequency) < 1:
            raise ValueError(
                f"RMSDMetaDynamicsBias: update_frequency must be at least 1, "
                f"got {update_frequency}."
            )
        if int(ramp_depositions) < 0:
            raise ValueError(
                f"RMSDMetaDynamicsBias: ramp_depositions must be non-negative, "
                f"got {ramp_depositions}."
            )
        if storage in ("preallocated", "fifo") and max_references is None:
            raise ValueError(
                f"RMSDMetaDynamicsBias: storage={storage!r} needs an explicit "
                "max_references — it is the whole point of the policy, either "
                "the ceiling that raises or the ring that overwrites."
            )
        capacity = int(max_references) if max_references is not None else 64
        if capacity < 1:
            raise ValueError(
                f"RMSDMetaDynamicsBias: max_references must be at least 1, got "
                f"{max_references}."
            )

        if atom_indices is None:
            selection: Tensor | None = None
        else:
            selection = torch.as_tensor(atom_indices, dtype=torch.long).reshape(-1)
            if selection.numel() == 0:
                raise ValueError(
                    "RMSDMetaDynamicsBias: atom_indices is empty; pass None to "
                    "use every atom."
                )
            if bool((selection < 0).any()):
                raise ValueError(
                    f"RMSDMetaDynamicsBias: atom_indices must be non-negative "
                    f"per-graph indices, got {selection.tolist()}."
                )
            if selection.unique().numel() != selection.numel():
                raise ValueError(
                    f"RMSDMetaDynamicsBias: atom_indices contains duplicates "
                    f"({selection.tolist()}). A repeated atom is silently "
                    "weighted twice in the RMSD."
                )

        self.k_push = float(k_push)
        self.alpha = float(alpha)
        self.storage = storage
        self.history = history
        self.update_frequency = int(update_frequency)
        self.ramp_depositions = int(ramp_depositions)
        self._capacity = capacity

        if selection is None:
            self.atom_indices: Tensor | None = None
        else:
            self.register_buffer("atom_indices", selection)

        n_sites = 0 if references is None else int(references.shape[1])
        if selection is not None:
            n_sites = selection.numel()
        self._allocate(capacity, max(n_sites, 1))

        if references is not None:
            self._seed(references)

        # The reference set is per-state under "state" history, so an accepted
        # swap changes which references a walker feels.
        self.state_dependent_for_exchange = history == "state"

    # ------------------------------------------------------------------
    # Storage
    # ------------------------------------------------------------------

    def _allocate(self, capacity: int, n_sites: int) -> None:
        """Create or replace the reference buffers.

        Parameters
        ----------
        capacity:
            Number of reference slots.
        n_sites:
            Number of compared atoms per reference.
        """
        dtype = torch.get_default_dtype()
        device = getattr(self, "reference_coords", torch.empty(0)).device
        self.register_buffer(
            "reference_coords",
            torch.zeros(capacity, n_sites, 3, dtype=dtype, device=device),
        )
        self.register_buffer(
            "reference_owner",
            torch.full((capacity,), -1, dtype=torch.long, device=device),
        )
        self.register_buffer(
            "reference_step",
            torch.full((capacity,), -1, dtype=torch.long, device=device),
        )
        # reference_count saturates at capacity; references_written keeps
        # counting, which fixes the FIFO ring position after it has wrapped.
        self.register_buffer(
            "reference_count", torch.zeros((), dtype=torch.long, device=device)
        )
        self.register_buffer(
            "references_written", torch.zeros((), dtype=torch.long, device=device)
        )
        self.register_buffer(
            "deposits", torch.zeros((), dtype=torch.long, device=device)
        )

    def _seed(self, references: Tensor) -> None:
        """Install warm-start references in chronological order.

        Parameters
        ----------
        references:
            Shape ``[R, M, 3]``, oldest first.

        Raises
        ------
        ValueError
            If the shape is not ``[R, M, 3]``, conflicts with
            ``atom_indices``, or exceeds the capacity.
        """
        refs = torch.as_tensor(references, dtype=self.reference_coords.dtype)
        if refs.ndim != 3 or refs.shape[-1] != 3:
            raise ValueError(
                f"RMSDMetaDynamicsBias: references must have shape [R, M, 3], "
                f"got {tuple(refs.shape)}."
            )
        if self.atom_indices is not None and refs.shape[1] != self.atom_indices.numel():
            raise ValueError(
                f"RMSDMetaDynamicsBias: references have {refs.shape[1]} atoms "
                f"but atom_indices selects {self.atom_indices.numel()}."
            )
        if refs.shape[0] > self._capacity:
            raise ValueError(
                f"RMSDMetaDynamicsBias: {refs.shape[0]} warm-start references "
                f"exceed max_references={self._capacity}."
            )
        if refs.shape[1] != self.reference_coords.shape[1]:
            self._allocate(self._capacity, int(refs.shape[1]))

        count = int(refs.shape[0])
        # Stored centered: the metric is translation invariant, so removing the
        # centroid once at deposition keeps it out of every later comparison.
        centered = refs - refs.mean(dim=1, keepdim=True)
        self.reference_coords[:count] = centered.to(self.reference_coords.device)
        self.reference_owner[:count] = -1 if self.history == "shared" else 0
        # Warm-start references are already fully active; a ramp is only for
        # references deposited into a running trajectory.
        self.reference_step[:count] = -self.ramp_depositions - 1
        self.reference_count += count
        self.references_written += count

    def _grow(self) -> None:
        """Extend the reference buffers by one more chunk."""
        extra = self._capacity
        device = self.reference_coords.device

        def _extend(buffer: Tensor, fill: float | int) -> Tensor:
            pad = torch.full(
                (extra, *buffer.shape[1:]), fill, dtype=buffer.dtype, device=device
            )
            return torch.cat([buffer, pad], dim=0)

        self.reference_coords = _extend(self.reference_coords, 0.0)
        self.reference_owner = _extend(self.reference_owner, -1)
        self.reference_step = _extend(self.reference_step, -1)

    @property
    def capacity(self) -> int:
        """Return the current number of reference slots."""
        return int(self.reference_coords.shape[0])

    def _next_slots(self, count: int) -> Tensor:
        """Return the slot indices *count* new references will occupy.

        Parameters
        ----------
        count:
            Number of references about to be deposited.

        Returns
        -------
        Tensor
            Slot indices, shape ``[count]``.

        Raises
        ------
        RuntimeError
            If ``preallocated`` capacity is exhausted.
        """
        written = int(self.references_written)
        device = self.reference_owner.device

        if self.storage == "fifo":
            return (torch.arange(count, device=device) + written) % self.capacity

        if written + count > self.capacity:
            if self.storage == "preallocated":
                raise RuntimeError(
                    f"RMSDMetaDynamicsBias {self.name!r}: reference storage is "
                    f"full ({self.capacity} references) and "
                    "storage='preallocated'. Raise max_references, lengthen "
                    "update_frequency, or switch to storage='fifo', which is "
                    "the xTB-compatible policy and discards the oldest "
                    "reference instead."
                )
            while written + count > self.capacity:
                self._grow()

        return torch.arange(count, device=device) + written

    # ------------------------------------------------------------------
    # Energy
    # ------------------------------------------------------------------

    def _gather_sites(self, current: Batch) -> Tensor:
        """Return the compared coordinates per graph, shape ``[B, M, 3]``.

        Parameters
        ----------
        current:
            The live batch.

        Returns
        -------
        Tensor
            Selected positions, differentiable with respect to
            ``current.positions``.
        """
        positions = current.positions
        batch_ptr = current.batch_ptr.to(positions.device)
        offsets = batch_ptr[:-1]  # [B]

        if self.atom_indices is None:
            # Uniform atom counts are checked in evaluate(), so the -1 here
            # resolves without a data-dependent int() that would break the graph.
            return positions.reshape(offsets.numel(), -1, 3)

        selection = self.atom_indices.to(positions.device)
        flat = offsets.unsqueeze(1) + selection.unsqueeze(0)  # [B, M]
        return positions[flat.reshape(-1)].reshape(offsets.numel(), -1, 3)

    def _owner_key(self, current: Batch, n_graphs: int) -> Tensor:
        """Return the history key per graph, shape ``[B]``.

        Parameters
        ----------
        current:
            The live batch.
        n_graphs:
            Number of graphs.

        Returns
        -------
        Tensor
            Per-graph key matched against ``reference_owner``.
        """
        device = self.reference_owner.device
        if self.history == "state":
            field = "thermodynamic_state_id"
        elif self.history == "walker":
            field = "walker_id"
        else:
            return torch.full((n_graphs,), -1, dtype=torch.long, device=device)

        ids = getattr(current, field, None)
        if ids is None:
            raise ValueError(
                f"RMSDMetaDynamicsBias {self.name!r}: history="
                f"{self.history!r} needs batch.{field}, which this batch "
                f"does not carry. Falling back to a single owner would put "
                f"every reference under one key and silently collapse the "
                f"per-{field} histories into one shared history — the "
                f"opposite of what history={self.history!r} asks for. "
                "EnhancedSampling stamps this field on every step; a bias "
                "driven directly must set it, or use history='shared' if "
                "one history really is intended."
            )
        if ids.numel() != n_graphs:
            raise ValueError(
                f"RMSDMetaDynamicsBias {self.name!r}: batch.{field} has "
                f"{ids.numel()} entries but the batch has {n_graphs} "
                f"graph(s). A shorter tensor broadcasts across graphs, which "
                f"would file every reference under one walker's key without "
                "raising."
            )
        return ids.reshape(-1).to(device=device, dtype=torch.long)

    def _reference_scale(self) -> Tensor:
        """Return each reference's ramp factor, shape ``[capacity]``.

        Returns
        -------
        Tensor
            Factors in ``[0, 1]``.
        """
        if self.ramp_depositions <= 0:
            return torch.ones(
                self.capacity,
                dtype=self.reference_coords.dtype,
                device=self.reference_coords.device,
            )
        age = (self.deposits - self.reference_step).to(self.reference_coords.dtype)
        return torch.clamp(age / float(self.ramp_depositions), min=0.0, max=1.0)

    def repulsion(self, coords: Tensor, owner_key: Tensor) -> Tensor:
        """Return the accumulated repulsion, shape ``[B]``.

        Contains no data-dependent Python branch, so it compiles with
        ``fullgraph=True``.

        Parameters
        ----------
        coords:
            Compared coordinates, shape ``[B, M, 3]``.
        owner_key:
            History key per graph, shape ``[B]``.

        Returns
        -------
        Tensor
            Bias energy per graph in eV, shape ``[B]``.
        """
        if self.reference_coords.shape[1] != coords.shape[1]:
            # A bias built without atom_indices or warm-start references does
            # not learn its site count until the first deposition; until then
            # the reference set is empty and the bias is identically zero.
            # This is a shape guard, not a data-dependent branch.
            return coords.new_zeros(coords.shape[0])

        slots = torch.arange(self.capacity, device=coords.device)
        live = slots < self.reference_count

        msd = _squared_rmsd(coords, self.reference_coords)  # [B, R]
        kernel = torch.exp(-self.alpha * msd)

        weights = self.k_push * self._reference_scale()  # [R]
        mask = live.unsqueeze(0)
        if self.history != "shared":
            mask = mask & (self.reference_owner.unsqueeze(0) == owner_key.unsqueeze(1))
        return (kernel * weights.unsqueeze(0) * mask).sum(dim=-1)

    def evaluate(self, current: Batch) -> BiasResult:
        """Reject periodic batches, then derive energy and forces.

        Parameters
        ----------
        current:
            The live batch.

        Returns
        -------
        BiasResult
            As :meth:`ConservativeBias.evaluate`.

        Raises
        ------
        ValueError
            If the batch is periodic according to ``batch.pbc``, if it
            carries a non-zero cell with no ``pbc`` flags, or if graphs in
            the batch have differing atom counts while ``atom_indices`` is
            ``None``.
        """
        self._reject_periodic(current)
        self._validate_sites(current)
        return super().evaluate(current)

    def _reject_periodic(self, current: Batch) -> None:
        """Raise if the batch is under periodic boundary conditions.

        Periodicity is read from ``batch.pbc`` when it is present, matching
        :func:`~nvalchemi.enhanced_sampling.pair_distance` and the rest of
        the toolkit: a cell is a box, and only the flags say whether atoms
        wrap through its faces.  A molecular batch carrying a bounding box
        with ``pbc`` all-False is therefore accepted — it is exactly the
        non-periodic case this bias is for.

        Without ``pbc`` there is nothing to read, and a non-zero cell is
        refused rather than assumed harmless.  A batch that declares a cell
        but no boundary condition has not said which case it is, and the
        failure this guard exists to prevent is silent.

        Runs in ``evaluate`` rather than ``energy`` so the flag reduction
        stays off the compiled path.

        Parameters
        ----------
        current:
            The live batch.

        Raises
        ------
        ValueError
            If any graph is periodic along any axis, or if a non-zero cell
            is present with no ``pbc`` flags to interpret it.
        """
        cell = getattr(current, "cell", None)
        if cell is None or not bool((cell != 0).any()):
            return

        pbc = getattr(current, "pbc", None)
        if pbc is None:
            raise ValueError(
                f"RMSDMetaDynamicsBias {self.name!r}: the batch carries a "
                "non-zero cell but no pbc flags, so whether atoms wrap "
                "through its faces is undeclared. Set pbc=False on a "
                "molecular system with a bounding box; Cartesian RMSD is "
                "only defined for the non-periodic case."
            )
        if not bool(pbc.any()):
            return

        raise ValueError(
            f"RMSDMetaDynamicsBias {self.name!r}: the batch is periodic "
            f"(pbc={pbc.reshape(-1, pbc.shape[-1])[0].tolist()}), and "
            "Cartesian RMSD against a stored reference is not defined under "
            "periodic boundary conditions — an atom crossing a cell face is "
            "physically unmoved but Cartesian-displaced by a lattice vector, "
            "which would inject a large spurious force. Use this bias on "
            "non-periodic (molecular) systems — a bounding-box cell with "
            "pbc=False is fine — or bias a periodic-aware CV with "
            "WellTemperedMetaDynamicsBias instead."
        )

    def _validate_sites(self, current: Batch) -> None:
        """Check that the compared-atom selection is well defined.

        Parameters
        ----------
        current:
            The live batch.

        Raises
        ------
        ValueError
            If atom counts differ across graphs without an explicit
            selection, or a selected index is out of range for some graph.
        """
        batch_ptr = current.batch_ptr
        counts = batch_ptr[1:] - batch_ptr[:-1]

        if self.atom_indices is None:
            if bool((counts != counts[0]).any()):
                raise ValueError(
                    f"RMSDMetaDynamicsBias {self.name!r}: graphs in the batch "
                    f"have differing atom counts {counts.tolist()}, so there "
                    "is no fixed atom correspondence to compare against a "
                    "shared reference set. Pass atom_indices to select a "
                    "common set of atoms per graph."
                )
            return

        selection = self.atom_indices.to(counts.device)
        smallest = int(counts.min())
        if int(selection.max()) >= smallest:
            raise ValueError(
                f"RMSDMetaDynamicsBias {self.name!r}: atom_indices requests "
                f"local index {int(selection.max())} but the smallest graph in "
                f"the batch has only {smallest} atoms. atom_indices are "
                "per-graph local indices, not global ones."
            )

    def energy(self, current: Batch) -> Tensor:
        """Return the accumulated RMSD bias, shape ``[B, 1]``.

        Parameters
        ----------
        current:
            Batch with strained positions supplied by
            :meth:`ConservativeBias.evaluate`.

        Returns
        -------
        Tensor
            Shape ``[B, 1]`` in eV.
        """
        coords = self._gather_sites(current)
        owner = self._owner_key(current, coords.shape[0])
        return self.repulsion(coords, owner).unsqueeze(-1)

    # ------------------------------------------------------------------
    # Deposition
    # ------------------------------------------------------------------

    def update(self, frames: Batch, result: BiasResult) -> None:
        """Append the current geometry to the reference set, one per walker.

        Parameters
        ----------
        frames:
            Post-step frame captured by the runner.
        result:
            The bias's own result from the preceding force evaluation.
        """
        with torch.no_grad():
            coords = self._gather_sites(frames).detach()  # [B, M, 3]
            centered = coords - coords.mean(dim=1, keepdim=True)
            owner = self._owner_key(frames, coords.shape[0])

            count = coords.shape[0]
            if self.reference_coords.shape[1] != centered.shape[1]:
                # First deposition of a bias built without atom_indices or
                # warm-start references: the site count is only known now.
                self._allocate(self.capacity, int(centered.shape[1]))
            slots = self._next_slots(count)

            self.reference_coords[slots] = centered.to(self.reference_coords.dtype)
            self.reference_owner[slots] = owner
            self.reference_step[slots] = self.deposits

            self.deposits += 1
            self.references_written += count
            self.reference_count = torch.clamp(
                self.references_written, max=self.capacity
            )
        self.bump_state_version()

    def load_state_dict(
        self, state: Mapping[str, Any], *args: Any, **kwargs: Any
    ) -> Any:
        """Restore state, resizing the reference buffers to the checkpoint.

        Both ``storage="grow"`` and a bias whose site count was fixed at its
        first deposition can carry buffer shapes that the constructor did not
        produce, which ``nn.Module.load_state_dict`` would reject.

        Parameters
        ----------
        state:
            Mapping from :meth:`state_dict`.
        *args, **kwargs:
            Forwarded up the MRO.

        Returns
        -------
        Any
            Whatever the next ``load_state_dict`` returns.
        """
        coords = state.get("reference_coords")
        if coords is not None and tuple(coords.shape) != tuple(
            self.reference_coords.shape
        ):
            self._allocate(int(coords.shape[0]), int(coords.shape[1]))
        return super().load_state_dict(state, *args, **kwargs)

    def __repr__(self) -> str:
        """Return a concise description of the bias."""
        sites = "all" if self.atom_indices is None else int(self.atom_indices.numel())
        return (
            f"{type(self).__name__}(name={self.name!r}, "
            f"references={int(self.reference_count)}/{self.capacity}, "
            f"storage={self.storage!r}, history={self.history!r}, "
            f"k_push={self.k_push:g}, alpha={self.alpha:g}, atoms={sites})"
        )
