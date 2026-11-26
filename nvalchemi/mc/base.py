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
"""Base machinery shared by GPU-resident Monte Carlo samplers."""

from __future__ import annotations

import weakref
from abc import abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

from nvalchemi.data import Batch
from nvalchemi.data.atomic_data import _default_mass_table
from nvalchemi.dynamics.base import BaseDynamics, DynamicsStage
from nvalchemi.dynamics.hooks._utils import KB_EV

if TYPE_CHECKING:
    from nvalchemi.dynamics.base import ConvergenceHook
    from nvalchemi.hooks import Hook
    from nvalchemi.models.base import BaseModelMixin

__all__ = ["BaseMonteCarlo", "MonteCarloStats"]

# The helpers below avoid ops whose output size depends on the data (boolean-mask
# indexing, ``nonzero``, ``bincount``): on CUDA those force a device-to-host sync
# every MC step, stalling the queue ahead of the next model call.


def _is_batch(ref: weakref.ref[Batch] | None, batch: Batch) -> bool:
    """Whether *ref* still points at *batch*.

    A weak reference rather than ``id(batch)``: CPython reuses the id of a
    freed batch, so a new batch could otherwise inherit the old one's state.
    """
    return ref is not None and ref() is batch


def _draw_from_mask(
    selectable: torch.Tensor,
    starts: torch.Tensor,
    counts: torch.Tensor,
    draws: torch.Tensor,
) -> torch.Tensor:
    """Return one uniformly drawn selectable row per segment.

    Parameters
    ----------
    selectable
        ``[R]`` 0/1 integer flags over rows ordered by segment.
    starts
        ``[G]`` first row of every segment.
    counts
        ``[G]`` number of selectable rows in every segment.
    draws
        ``[G]`` uniforms in ``[0, 1)``.

    Returns
    -------
    torch.Tensor
        ``[G]`` row indices; a segment without selectable rows returns its start.
    """
    cumulative = torch.cumsum(selectable, dim=0)
    # Rank of the drawn entry among all selectable rows, then the first row whose
    # running total reaches it.
    before = cumulative[starts] - selectable[starts]
    ranks = before + torch.floor(draws * counts.clamp(min=1)).to(torch.long)
    rows = torch.searchsorted(cumulative, ranks + 1)
    return torch.where(counts > 0, rows, starts)


def _swap_pairs(
    numbers: torch.Tensor,
    first: torch.Tensor,
    second: torch.Tensor,
    swap: torch.Tensor,
) -> None:
    """Exchange ``numbers[first]`` and ``numbers[second]`` where *swap* is set.

    One pair per graph, so indices never collide; unswapped pairs are rewritten
    with their own values.
    """
    first_z, second_z = numbers[first], numbers[second]
    numbers[first] = torch.where(swap, second_z, first_z)
    numbers[second] = torch.where(swap, first_z, second_z)


@dataclass(frozen=True)
class MonteCarloStats:
    """Host-side snapshot of attempted and accepted MC moves."""

    attempted: int
    accepted: int

    @property
    def acceptance(self) -> float:
        """Return the accepted fraction, or NaN before the first attempt."""
        return self.accepted / self.attempted if self.attempted else float("nan")


class BaseMonteCarlo(BaseDynamics):
    """Base class for one independent, batched MC proposal per graph per step.

    Subclasses mutate trial atom types in :meth:`_propose` and restore rejected
    proposals in :meth:`_restore_rejected`. The model evaluates the full trial
    batch once; all acceptance arithmetic and state restoration remain on the
    batch device.

    Atomic masses always follow the species. Every accepted move that changes
    an atom's type also sets that atom's ``atomic_masses`` entry to the mass of
    its new species, so any dynamics run afterwards (e.g. alternating MC and
    MD blocks) integrates each atom with the right mass. The per-species
    masses are taken from each graph the first time the sampler sees the batch
    (so custom masses, e.g. a deuterium mass for H, are kept and move with the
    species, and never cross into another graph); a species absent from a
    graph uses the periodic-table mass. Masses are only touched where the
    species changed. Such an atom's velocity is rescaled by
    :math:`\sqrt{m_\text{old}/m_\text{new}}`, keeping its kinetic energy: a
    Maxwell-Boltzmann velocity for the old mass becomes one for the new mass,
    and acceptance on the potential energy alone stays exact.

    :meth:`state_dict` includes the proposal/acceptance generator's state, the
    per-graph mass lookup and the acceptance counters, so a run restored with
    :meth:`load_state_dict` continues exactly as an uninterrupted one: the same
    random sequence, the same masses for species that left a graph before the
    checkpoint, and cumulative :attr:`stats`.
    """

    __needs_keys__: set[str] = {"energy"}
    __provides_keys__: set[str] = {"atomic_numbers", "mc_accepted"}
    _mutable_fields = (*BaseDynamics._mutable_fields, "atomic_numbers", "atomic_masses")

    def __init__(
        self,
        model: BaseModelMixin,
        temperature: float | torch.Tensor,
        random_seed: int = 420,
        n_steps: int | None = None,
        hooks: list[Hook] | None = None,
        convergence_hook: ConvergenceHook | dict | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize an MC sampler.

        Parameters
        ----------
        model
            Model that supplies an energy for each graph in the batch.
        temperature
            Positive scalar temperature in K, or one temperature per graph.
        random_seed
            Seed for the device-resident proposal and acceptance generator.
        n_steps
            Default number of proposals per graph for :meth:`run`.
        hooks
            Hooks dispatched at the standard dynamics stages.
        convergence_hook
            Optional Toolkit convergence hook.
        **kwargs
            Forwarded to :class:`~nvalchemi.dynamics.base.BaseDynamics`.
        """
        super().__init__(
            model=model,
            n_steps=n_steps,
            hooks=hooks,
            convergence_hook=convergence_hook,
            **kwargs,
        )
        initial_temperature = torch.as_tensor(temperature)
        if initial_temperature.ndim > 1 or torch.any(initial_temperature <= 0):
            raise ValueError(
                "temperature must be positive and scalar or one-dimensional"
            )
        self._temperature = temperature
        self._random_seed = random_seed
        # Device copies of constructor parameters (temperature, species, chemical
        # potentials, ...), made once per device and dtype rather than uploaded by
        # a blocking host-to-device copy every step. Treat the attributes they are
        # built from as read-only after construction.
        self._device_parameters: dict[
            tuple[str, torch.device, torch.dtype], torch.Tensor
        ] = {}
        self._generator: torch.Generator | None = None
        self._generator_device: torch.device | None = None
        # Generator state from load_state_dict, applied when the generator is
        # next created (its device is known only once a batch is seen).
        self._pending_generator_state: torch.Tensor | None = None
        # Mass lookup from load_state_dict, adopted when the batch is next seen.
        self._pending_mass_table: torch.Tensor | None = None
        self._energy: torch.Tensor | None = None
        self._energy_batch: weakref.ref[Batch] | None = None
        self._validated_batch: weakref.ref[Batch] | None = None
        self._active: torch.Tensor | None = None
        self._attempted: torch.Tensor | None = None
        self._accepted: torch.Tensor | None = None
        # Per-graph, per-species mass lookup [num_graphs, max Z + 1] and the
        # species each atom's mass currently matches, both for the batch object
        # last seen by _ensure_mass_table.
        self._mass_table: torch.Tensor | None = None
        self._mass_numbers: torch.Tensor | None = None
        self._mass_batch: weakref.ref[Batch] | None = None

    def _on_device(
        self,
        name: str,
        value: Any,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Return the cached copy of parameter *name* on *device* in *dtype*."""
        key = (name, device, dtype)
        cached = self._device_parameters.get(key)
        if cached is None:
            cached = self._device_parameters[key] = torch.as_tensor(
                value, dtype=dtype, device=device
            )
        return cached

    def _temperature_for(self, batch: Batch) -> torch.Tensor:
        """Return one positive temperature in K for every graph."""
        temperature = self._on_device(
            "temperature", self._temperature, batch.device, batch.positions.dtype
        )
        if temperature.ndim == 0:
            return temperature.expand(batch.num_graphs)
        if temperature.shape != (batch.num_graphs,):
            raise ValueError(
                "temperature tensor must have one value per graph "
                f"(expected {(batch.num_graphs,)}, got {tuple(temperature.shape)})"
            )
        return temperature

    def _active_mask(self, batch: Batch) -> torch.Tensor:
        """Return the graphs eligible to receive an MC proposal."""
        if getattr(batch, "status", None) is None:
            return torch.ones(batch.num_graphs, dtype=torch.bool, device=batch.device)
        status = batch.status.squeeze(-1) if batch.status.ndim == 2 else batch.status
        return status[: batch.num_graphs] < self.exit_status

    def _ensure_generator(self, batch: Batch) -> torch.Generator:
        """Create the persistent device generator on first use."""
        if self._generator is None or self._generator_device != batch.device:
            self._generator = torch.Generator(device=batch.device)
            self._generator.manual_seed(self._random_seed)
            self._generator_device = batch.device
            if self._pending_generator_state is not None:
                try:
                    self._generator.set_state(self._pending_generator_state)
                except RuntimeError as error:
                    raise RuntimeError(
                        f"{type(self).__name__}: the restored generator state does not "
                        f"fit a {batch.device.type} generator; restore onto the device "
                        "type the checkpoint was written on"
                    ) from error
                self._pending_generator_state = None
        return self._generator

    def state_dict(self) -> dict[str, Any]:
        """Return the state needed to resume this run exactly.

        Extends :meth:`BaseDynamics.state_dict` with the Monte Carlo run state,
        each entry present once it exists (after the first step):

        * ``generator_state``: the proposal/acceptance generator's byte state,
          so a restored run draws the same random sequence;
        * ``mass_table``: the per-graph species-to-mass lookup, which keeps a
          custom mass for a species that has since left its graph;
        * ``attempted`` / ``accepted``: the cumulative move counters behind
          :attr:`stats`.

        State restored by :meth:`load_state_dict` but not yet applied (no step
        since) is saved as restored, so loading and re-saving loses nothing.
        The acceptance baseline energy is not stored: :meth:`run` re-evaluates
        it on the restored batch.

        Returns
        -------
        dict[str, Any]
            The base state plus the entries above.
        """
        state = super().state_dict()
        # Until the next step adopts it, restored state is held as pending.
        if self._generator is not None:
            state["generator_state"] = self._generator.get_state()
        elif self._pending_generator_state is not None:
            state["generator_state"] = self._pending_generator_state.clone()
        mass_table = (
            self._mass_table
            if self._mass_table is not None
            else self._pending_mass_table
        )
        if mass_table is not None:
            state["mass_table"] = mass_table.detach().clone()
        if self._attempted is not None and self._accepted is not None:
            state["attempted"] = self._attempted.detach().clone()
            state["accepted"] = self._accepted.detach().clone()
        return state

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore state produced by :meth:`state_dict`.

        Everything the sampler accumulated is replaced, so a sampler that
        already ran does not keep its advanced generator, masses or counters,
        and the cached acceptance baseline is dropped: the next step
        re-evaluates it, even on a reused batch object.
        The generator is rebuilt from the restored seed on its next use and set
        to ``generator_state``; the mass lookup is adopted when the sampler
        next sees a batch, which must have the checkpoint's number of graphs.

        Parameters
        ----------
        state
            The mapping previously returned by :meth:`state_dict`.
        """
        super().load_state_dict(state)
        self._energy = None
        self._energy_batch = None
        generator_state = state.get("generator_state")
        self._generator = None
        self._generator_device = None
        self._pending_generator_state = (
            None
            if generator_state is None
            else generator_state.detach().to("cpu", torch.uint8).clone()
        )
        mass_table = state.get("mass_table")
        self._mass_table = None
        self._mass_numbers = None
        self._mass_batch = None
        self._pending_mass_table = (
            None if mass_table is None else mass_table.detach().clone()
        )
        attempted, accepted = state.get("attempted"), state.get("accepted")
        if attempted is None or accepted is None:
            self._attempted = self._accepted = None
        else:
            self._attempted = attempted.detach().clone()
            self._accepted = accepted.detach().clone()

    def _ensure_observables(self, batch: Batch) -> None:
        """Allocate graph-level energy and acceptance storage if absent."""
        # Plain assignment routes an unknown [num_graphs, ...] key into the
        # system level (Batch has no ``add_system_property``).
        energy = getattr(batch, "energy", None)
        if energy is None:
            batch.energy = torch.zeros(
                (batch.num_graphs, 1), dtype=batch.positions.dtype, device=batch.device
            )
        accepted = getattr(batch, "mc_accepted", None)
        if accepted is None:
            batch.mc_accepted = torch.zeros(
                (batch.num_graphs, 1), dtype=torch.bool, device=batch.device
            )
        elif accepted.shape != (batch.num_graphs, 1):
            raise ValueError("batch.mc_accepted must have shape [num_graphs, 1]")

    def _model_energy(
        self, batch: Batch, active_graph_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Evaluate the model and return its one energy value per graph."""
        outputs = self.compute(batch, active_graph_mask)
        energy = outputs["energy"]
        if energy.numel() != batch.num_graphs:
            raise ValueError("Monte Carlo requires exactly one energy per graph")
        return energy.detach().reshape(batch.num_graphs)

    def _initialize_energy(
        self, batch: Batch, active_graph_mask: torch.Tensor | None = None
    ) -> None:
        """Evaluate the accepted starting configuration once."""
        if not _is_batch(self._validated_batch, batch):
            self._validate_batch(batch)
            self._validated_batch = weakref.ref(batch)
        self._call_hooks(DynamicsStage.BEFORE_COMPUTE, batch, active_graph_mask)
        self._energy = self._model_energy(batch, active_graph_mask)
        self._call_hooks(DynamicsStage.AFTER_COMPUTE, batch, active_graph_mask)
        self._energy_batch = weakref.ref(batch)

    def synchronize(self, batch: Batch) -> None:
        """Adopt a trusted current energy after an external state update.

        Call this after an MD block that changes positions or the cell. The
        energy must already describe the current atom types and coordinates.
        """
        self._ensure_observables(batch)
        if not _is_batch(self._validated_batch, batch):
            self._validate_batch(batch)
            self._validated_batch = weakref.ref(batch)
        self._energy = batch.energy.detach().reshape(batch.num_graphs).clone()
        self._energy_batch = weakref.ref(batch)

    def refresh_energy(self, batch: Batch) -> None:
        """Re-evaluate the current configuration's energy and adopt it.

        Use instead of :meth:`synchronize` when the energy already on ``batch``
        came from a differently configured evaluation than the one the trials
        will use -- e.g. an MD force evaluation, when MC runs with
        ``active_outputs={"energy"}`` -- so the acceptance baseline and every
        trial energy come from the same model call. Costs one evaluation.
        """
        self._ensure_observables(batch)
        self._initialize_energy(batch)

    def reset_statistics(self) -> None:
        """Reset cumulative attempted and accepted move counters."""
        self._attempted = None
        self._accepted = None

    @property
    def stats(self) -> MonteCarloStats:
        """Return a host-side statistics snapshot after a synchronization point."""
        attempted = 0 if self._attempted is None else int(self._attempted.item())
        accepted = 0 if self._accepted is None else int(self._accepted.item())
        return MonteCarloStats(attempted=attempted, accepted=accepted)

    @abstractmethod
    def _propose(
        self,
        batch: Batch,
        generator: torch.Generator,
        active: torch.Tensor,
    ) -> None:
        """Apply one trial proposal per active graph in-place.

        *active* is the sampler's own mask for this step: a sampler may narrow
        it in place to drop graphs with no legal move, which then count as
        neither attempted nor accepted.
        """

    @abstractmethod
    def _restore_rejected(self, batch: Batch, rejected: torch.Tensor) -> None:
        """Restore the trial atom types in rejected graphs."""

    def _chemical_delta(self, batch: Batch) -> torch.Tensor:
        """Return the per-graph chemical contribution to the trial potential."""
        return torch.zeros(
            batch.num_graphs, dtype=batch.positions.dtype, device=batch.device
        )

    def _validate_batch(self, batch: Batch) -> None:
        """Validate sampler-specific assumptions for a newly seen batch."""

    def _ensure_mass_table(self, batch: Batch) -> None:
        """Record per-graph, per-species masses from a newly seen batch.

        Each species present in a graph takes its mass from that graph, so
        custom per-species masses survive and never leak into another graph;
        species absent from a graph fall back to the periodic-table mass.
        Rebuilt when the batch object or its size changes; after
        :meth:`load_state_dict` the restored lookup is adopted instead.

        Raises
        ------
        ValueError
            If a restored lookup does not match the batch's number of graphs.
        """
        masses = getattr(batch, "atomic_masses", None)
        if masses is None:
            self._mass_table = None
            return
        numbers = batch.atomic_numbers.reshape(-1).long()
        if (
            self._mass_table is not None
            and _is_batch(self._mass_batch, batch)
            and self._mass_numbers is not None
            and self._mass_numbers.shape == numbers.shape
        ):
            return
        if self._pending_mass_table is not None:
            restored = self._pending_mass_table
            if restored.shape[0] != batch.num_graphs:
                raise ValueError(
                    f"{type(self).__name__}: the restored mass lookup covers "
                    f"{restored.shape[0]} graphs but the batch has {batch.num_graphs}"
                )
            table = restored.to(device=masses.device, dtype=masses.dtype)
            self._pending_mass_table = None
        else:
            default = _default_mass_table().to(device=masses.device, dtype=masses.dtype)
            table = default.expand(batch.num_graphs, -1).clone()
        table[batch.batch_idx.long(), numbers] = masses.detach().reshape(-1)
        self._mass_table = table
        self._mass_numbers = numbers.clone()
        self._mass_batch = weakref.ref(batch)

    def _sync_masses(self, batch: Batch) -> None:
        """Give every atom whose species changed its new mass, at fixed kinetic energy.

        Velocities, when present, are rescaled by ``sqrt(m_old / m_new)``.
        Device-only (no host sync): atoms whose type is unchanged keep their
        mass and velocity bit-for-bit.
        """
        masses = getattr(batch, "atomic_masses", None)
        if masses is None or self._mass_table is None or self._mass_numbers is None:
            return
        numbers = batch.atomic_numbers.reshape(-1).long()
        changed = numbers != self._mass_numbers
        with torch.no_grad():
            flat = masses.reshape(-1)
            new = torch.where(
                changed, self._mass_table[batch.batch_idx.long(), numbers], flat
            )
            velocities = getattr(batch, "velocities", None)
            if velocities is not None:
                # m_old / m_new is exactly 1 for unchanged atoms.
                scale = torch.sqrt(flat / new).to(velocities.dtype)
                velocities.mul_(scale.reshape(-1, *([1] * (velocities.ndim - 1))))
            flat.copy_(new)
        self._mass_numbers.copy_(numbers)

    def pre_update(self, batch: Batch) -> None:
        """Apply one trial proposal to every active graph."""
        self._ensure_mass_table(batch)
        self._active = self._active_mask(batch)
        self._propose(batch, self._ensure_generator(batch), self._active)

    def post_update(self, batch: Batch) -> None:
        """Accept or restore proposals after the trial-energy evaluation."""
        if self._energy is None or self._active is None:
            raise RuntimeError("Monte Carlo proposal state was not initialized")
        trial_energy = batch.energy.detach().reshape(batch.num_graphs)
        delta = trial_energy - self._energy + self._chemical_delta(batch)
        log_probability = (-delta / (KB_EV * self._temperature_for(batch))).clamp(
            max=0.0
        )
        log_uniform = torch.log(
            torch.rand(
                batch.num_graphs,
                device=batch.device,
                generator=self._ensure_generator(batch),
            )
        )
        accepted = self._active & (log_uniform < log_probability)
        rejected = self._active & ~accepted
        with torch.no_grad():
            self._restore_rejected(batch, rejected)
            self._energy = torch.where(accepted, trial_energy, self._energy)
            # Only rejected graphs go back to their baseline: in a FusedStage the
            # other graphs' energies belong to the other sub-stages.
            batch.energy.copy_(
                torch.where(rejected, self._energy, trial_energy).reshape_as(
                    batch.energy
                )
            )
            batch.mc_accepted.copy_(accepted.reshape_as(batch.mc_accepted))
            if self._attempted is None:
                self._attempted = self._active.sum().detach()
                self._accepted = accepted.sum().detach()
            else:
                if self._attempted.device != batch.device:  # restored counters
                    self._attempted = self._attempted.to(batch.device)
                    self._accepted = self._accepted.to(batch.device)
                self._attempted += self._active.sum().detach()
                self._accepted += accepted.sum().detach()
        # Rejected trials are already restored, so only accepted moves differ.
        self._sync_masses(batch)

    def _masked_pre_update(self, batch: Batch, mask: torch.Tensor) -> None:
        """Propose on the graphs in *mask* ahead of a :class:`FusedStage` compute.

        The shared forward pass that follows evaluates the trials. The
        acceptance baseline is the energy already on ``batch``: the previous
        fused compute plus this sampler's own restores leave every graph's
        current energy there, including graphs that just migrated into this
        stage. Only the first fused step on a batch evaluates the model here.
        Atom types are the only state changed, and only in masked graphs, so
        no save/restore of the other graphs is needed.
        """
        self._ensure_state_initialized(batch)
        self._ensure_observables(batch)
        if self._energy is None or not _is_batch(self._energy_batch, batch):
            self._initialize_energy(batch)
        else:
            self._energy = batch.energy.detach().reshape(batch.num_graphs).clone()
        self._ensure_mass_table(batch)
        # A copy: ``_propose`` may narrow it in place (graphs with no legal move),
        # which must not leak into the FusedStage's own mask.
        self._active = mask.reshape(batch.num_graphs).clone()
        with torch.no_grad():
            self._propose(batch, self._ensure_generator(batch), self._active)

    def _stale_on_exit(self, batch: Batch) -> torch.Tensor | None:
        """Return the graphs whose last trial was rejected.

        The shared compute evaluated the trial; a rejection restores the species
        and the energy but not the forces or stress, which still describe the
        undone configuration. An accepted trial is the current state, so its
        outputs carry over. A :class:`~nvalchemi.dynamics.base.FusedStage`
        re-primes the rejected graphs as they leave for another sub-stage.
        """
        accepted = getattr(batch, "mc_accepted", None)
        if accepted is None:
            return None
        return ~accepted.view(-1).bool()

    def _masked_post_update(self, batch: Batch, mask: torch.Tensor) -> None:
        """Accept or restore the :meth:`_masked_pre_update` proposals.

        *mask* was already applied when proposing; unmasked graphs are
        inactive and keep the trial energy the shared compute wrote.
        """
        with torch.no_grad():
            self.post_update(batch)

    def step(self, batch: Batch) -> tuple[Batch, torch.Tensor | None]:
        """Run one complete, hook-aware MC proposal and acceptance step.

        Follows the :meth:`BaseDynamics.step` hook lifecycle (admission, the
        per-step stages with the step's active-graph mask, convergence and
        graduation); the first step on a batch evaluates its accepted energy
        in place of force priming.
        """
        self._ensure_state_initialized(batch)
        self._ensure_admission_initialized(batch)
        self._ensure_observables(batch)
        mask = self.active_graph_mask(batch, self.exit_status)
        self._call_hooks(DynamicsStage.BEFORE_STEP, batch, mask)
        with self._stream_scope(batch.device):
            if self._energy is None or not _is_batch(self._energy_batch, batch):
                self._initialize_energy(batch, mask)
            self._call_hooks(DynamicsStage.BEFORE_PRE_UPDATE, batch, mask)
            self.pre_update(batch)
            self._call_hooks(DynamicsStage.AFTER_PRE_UPDATE, batch, mask)
            self._call_hooks(DynamicsStage.BEFORE_COMPUTE, batch, mask)
            self._model_energy(batch, mask)
            self._call_hooks(DynamicsStage.AFTER_COMPUTE, batch, mask)
            self._call_hooks(DynamicsStage.BEFORE_POST_UPDATE, batch, mask)
            self.post_update(batch)
            self._call_hooks(DynamicsStage.AFTER_POST_UPDATE, batch, mask)
        self._call_hooks(DynamicsStage.AFTER_STEP, batch, mask)
        converged = self._check_convergence(batch)
        self._last_converged = converged
        if converged is not None:
            self._call_hooks(DynamicsStage.ON_CONVERGE, batch, mask)
        # Dispatched whenever a hook listens: gating on the mask would host-sync.
        if mask is not None and self._has_hooks_for_stage(DynamicsStage.ON_GRADUATE):
            still_active = self.active_graph_mask(batch, self.exit_status)
            self._call_hooks(
                DynamicsStage.ON_GRADUATE,
                batch,
                mask,
                ignore_frequency=True,
                graduated_mask=mask & ~still_active,
            )
        self.step_count += 1
        return batch, converged


class _PairSwapMonteCarlo(BaseMonteCarlo):
    """Shared state and rejection for canonical samplers that swap two atoms."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the sampler; see :class:`BaseMonteCarlo`."""
        super().__init__(*args, **kwargs)
        self._proposal_first: torch.Tensor | None = None
        self._proposal_second: torch.Tensor | None = None
        self._proposal_swapped: torch.Tensor | None = None

    def _apply_swap(
        self,
        batch: Batch,
        first: torch.Tensor,
        second: torch.Tensor,
        active: torch.Tensor,
    ) -> None:
        """Swap each active graph's unlike pair in place and record the proposal."""
        numbers = batch.atomic_numbers
        swap = active & (numbers[first] != numbers[second])
        with torch.no_grad():
            _swap_pairs(numbers, first, second, swap)
        self._proposal_first = first
        self._proposal_second = second
        self._proposal_swapped = swap

    def _restore_rejected(self, batch: Batch, rejected: torch.Tensor) -> None:
        """Swap back rejected trial pairs that were actually exchanged."""
        if self._proposal_swapped is None:
            raise RuntimeError(
                f"{type(self).__name__} rejection attempted without a proposal"
            )
        _swap_pairs(
            batch.atomic_numbers,
            self._proposal_first,
            self._proposal_second,
            rejected & self._proposal_swapped,
        )
