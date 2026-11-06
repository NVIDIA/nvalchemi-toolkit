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

"""Local overlap-relief crystal packing."""

from __future__ import annotations

import secrets
from collections.abc import Callable
from dataclasses import dataclass, field
from numbers import Integral
from typing import TYPE_CHECKING, Any, Literal

import torch
from torch import Tensor

from nvalchemi.csp.data import MolecularPackingInput, RigidMoleculeASUBatch
from nvalchemi.csp.packer._cells import sample_valid_cells, sampling_tables
from nvalchemi.csp.packer._contacts import (
    ContactWorkspace,
    contact_forces,
    make_contact_maps,
)
from nvalchemi.csp.packer._state import WorkingState, relax_step
from nvalchemi.csp.packer.config import OverlapReliefConfig
from nvalchemi.csp.packer.protocol import PackingContext
from nvalchemi.csp.packer.result import (
    OverlapReliefProgress,
    PackingReport,
    PackingResult,
    PackingStopReason,
)
from nvalchemi.data import resolve_device

if TYPE_CHECKING:
    from nvalchemi.specs import BaseSpec

__all__ = ["OverlapReliefPacker"]

_DEFAULT_CANDIDATE_MULTIPLIER = 1000

# Fixed arithmetic offsets derive repeatable rank and refill seeds from the call
# seed and packing state. Keep their values stable to preserve seeded outputs.
_WARP_SEED_MODULUS = 2**31 - 1
_RANK_SEED_STRIDE = 2654435761
_REFILL_GENERATED_SEED_STRIDE = 104729
_REFILL_COUNT_SEED_STRIDE = 9176


@dataclass
class _OverlapReliefCore:
    """Mutable local packing state for one packing run."""

    packer: OverlapReliefPacker
    inputs: MolecularPackingInput
    config: OverlapReliefConfig
    device: torch.device
    rank: int
    world_size: int
    local_target: int
    candidate_budget: int | None
    count_cap: int
    base_seed: int
    formula: dict[str, Tensor]
    operation_count: int
    asu_molecule_atom_ptr: Tensor
    asu_conformer_starts: Tensor
    asu_conformer_stops: Tensor
    asu_contact_distances: Tensor
    max_contact_distance: float
    groups: Tensor
    probabilities: Tensor
    symmetry_table: Tensor
    op_indices: Tensor
    op_ptr: Tensor
    state: WorkingState
    contact_maps: Any
    contact_workspace: ContactWorkspace
    fresh_rows: Tensor
    generated: int = 0
    accepted_blocks: list[tuple[Tensor, ...]] = field(default_factory=list)
    accepted_total: int = 0
    iteration: int = 0
    stop_reason: PackingStopReason = PackingStopReason.CANDIDATE_BUDGET_EXHAUSTED
    expiration_iterations: list[int | None] = field(default_factory=list)
    next_expiration_iteration: int | None = None
    active_rows: list[int] = field(default_factory=list)

    def remaining_budget(self) -> int:
        """Return the unspent candidate budget, or row capacity when unlimited."""
        if self.candidate_budget is None:
            return self.count_cap
        return max(0, self.candidate_budget - self.generated)

    def refill(self, rows: Tensor, *, defer_relaxation: bool = False) -> Tensor:
        """Sample and initialize up to the available candidate budget in the selected rows."""
        if rows.numel() == 0:
            return rows
        count = min(int(rows.numel()), self.remaining_budget())
        if count == 0:
            self.next_expiration_iteration = min(
                (value for value in self.expiration_iterations if value is not None),
                default=None,
            )
            return rows[:0]
        selected_rows = rows[:count].to(device=self.device, dtype=torch.int64)
        round_seed = (
            self.base_seed
            + self.generated * _REFILL_GENERATED_SEED_STRIDE
            + count * _REFILL_COUNT_SEED_STRIDE
        ) % _WARP_SEED_MODULUS
        cells, selected_groups = sample_valid_cells(
            count=count,
            config=self.config,
            groups=self.groups,
            probabilities=self.probabilities,
            formula_unit_volume=self.inputs.formula_unit_volume,
            seed=round_seed,
        )
        self.state.initialize(
            rows=selected_rows,
            sampled_cells=cells,
            sampled_groups=selected_groups,
            op_indices=self.op_indices,
            op_ptr=self.op_ptr,
            conformer_starts=self.asu_conformer_starts,
            conformer_stops=self.asu_conformer_stops,
            seed=round_seed,
        )
        self.generated += count
        for row in selected_rows.tolist():
            self.expiration_iterations[row] = (
                self.iteration
                + self.config.max_steps_per_candidate
                + int(defer_relaxation)
            )
        self.next_expiration_iteration = min(
            (value for value in self.expiration_iterations if value is not None),
            default=None,
        )
        return selected_rows

    def run(self, *, progress_callback: Callable[..., Any] | None) -> None:
        """Run local packing until the target is reached or candidates are exhausted."""
        while self.accepted_total < self.local_target and self.active_rows:
            self.fresh_rows.zero_()
            contacts = contact_forces(
                conformer_positions=self.formula["conformer_positions"],
                conformer_ptr=self.formula["conformer_ptr"],
                conformer_ids=self.state.conformer_ids,
                centers=self.state.centers,
                rotations=self.state.rotations,
                cells=self.state.cells,
                inverse_cells=self.state.inverse_cells,
                symmetry_table=self.symmetry_table,
                selected_symmetry_ops=self.state.symmetry_ops,
                contact_distances=self.asu_contact_distances,
                max_contact_distance=self.max_contact_distance,
                maps=self.contact_maps,
                workspace=self.contact_workspace,
                active_mask=self.state.active,
            )
            check = self.iteration % self.config.convergence_check_interval == 0 or (
                self.next_expiration_iteration is not None
                and self.iteration >= self.next_expiration_iteration
            )
            if check:
                converged_mask = self.state.active & (
                    contacts.max_overlap <= float(self.config.overlap_tolerance)
                )
                converged_rows = torch.nonzero(converged_mask, as_tuple=False).flatten()
                converged_count = int(converged_rows.numel())
                active_count_before = len(self.active_rows)
                if progress_callback is not None:
                    active_ids = torch.tensor(
                        self.active_rows, dtype=torch.int64, device=self.device
                    )
                    total_snapshot = contacts.total_overlap.index_select(0, active_ids)
                    max_snapshot = contacts.max_overlap.index_select(0, active_ids)
                else:
                    total_snapshot = max_snapshot = None
                remaining_outputs = self.local_target - self.accepted_total
                keep = converged_rows[:remaining_outputs]
                if keep.numel():
                    self.accepted_total += self.packer._append_accepted(
                        keep, self.state, contacts, self.accepted_blocks
                    )

                if self.accepted_total >= self.local_target:
                    self.stop_reason = PackingStopReason.TARGET_REACHED
                    if progress_callback is not None:
                        progress_callback(
                            self.packer._progress(
                                generated=self.generated,
                                accepted=self.accepted_total,
                                active_count=active_count_before,
                                converged_count=converged_count,
                                expired_count=0,
                                replaced_count=converged_count,
                                iteration=self.iteration,
                                total_snapshot=total_snapshot,
                                max_snapshot=max_snapshot,
                                rank=self.rank,
                                world_size=self.world_size,
                            )
                        )
                    break

                expired_mask = self.state.active & (
                    self.state.steps >= self.config.max_steps_per_candidate
                )
                expired_rows = torch.nonzero(expired_mask, as_tuple=False).flatten()
                expired_count = int(expired_rows.numel())
                replacement = converged_mask | expired_mask
                replaced_rows = torch.nonzero(replacement, as_tuple=False).flatten()
                replaced_count = int(replaced_rows.numel())
                replaced_row_list = replaced_rows.tolist()
                replaced_set = set(replaced_row_list)
                for row in replaced_row_list:
                    self.expiration_iterations[row] = None
                self.state.active[replaced_rows] = False
                self.active_rows = [
                    row for row in self.active_rows if row not in replaced_set
                ]
                refilled_rows = self.refill(replaced_rows, defer_relaxation=True)
                refilled_row_list = refilled_rows.tolist()
                self.active_rows.extend(refilled_row_list)
                self.active_rows.sort()
                self.fresh_rows[refilled_rows] = True
                if progress_callback is not None:
                    progress_callback(
                        self.packer._progress(
                            generated=self.generated,
                            accepted=self.accepted_total,
                            active_count=active_count_before,
                            converged_count=converged_count,
                            expired_count=expired_count,
                            replaced_count=replaced_count,
                            iteration=self.iteration,
                            total_snapshot=total_snapshot,
                            max_snapshot=max_snapshot,
                            rank=self.rank,
                            world_size=self.world_size,
                        )
                    )
                if not self.active_rows:
                    self.stop_reason = PackingStopReason.CANDIDATE_BUDGET_EXHAUSTED
                    break

            relax_step(
                state=self.state,
                conformer_positions=self.formula["conformer_positions"],
                conformer_ptr=self.formula["conformer_ptr"],
                molecule_atom_ptr=self.asu_molecule_atom_ptr,
                forces=contacts.forces,
                torques=contacts.torques,
                virial=contacts.virial,
                max_overlap=contacts.max_overlap,
                expanded_atom_count=int(self.inputs.num_atoms)
                * self.config.z_prime
                * self.operation_count,
                step_scale=float(self.config.step_scale),
                max_step=float(self.config.max_step),
                cell_step_scale=float(self.config.cell_step_scale),
                max_cell_strain=float(self.config.max_cell_strain),
                volume_compression_scale=float(self.config.volume_compression_scale),
                active=self.state.active & ~self.fresh_rows,
            )
            self.iteration += 1
        if self.accepted_total >= self.local_target:
            self.stop_reason = PackingStopReason.TARGET_REACHED


class OverlapReliefPacker:
    """Generate molecular crystal starting structures from supplied conformers.

    The Packer selects conformers, places molecules in periodic cells, and
    translates or rotates whole molecules to reduce intermolecular clashes.
    It can also change the cell. Molecular geometry stays fixed, and accepted
    structures meet the configured overlap tolerance. Energy minimization
    with a physical model is a separate step. Required arrays are transferred
    to the selected device when needed; accepted ASU representations retain
    the original input object.

    Parameters
    ----------
    config : OverlapReliefConfig
        Immutable base configuration containing ``z``, ``z_prime``, sampling
        controls, candidate capacity, and the optional finite candidate
        budget. The requested accepted count belongs to each call.
    device : torch.device or str
        Explicit execution device, either ``"cpu"`` or ``"cuda:N"``.
    """

    def __init__(
        self, config: OverlapReliefConfig, *, device: torch.device | str
    ) -> None:
        """Bind the configuration to a CPU or CUDA device.

        Parameters
        ----------
        config : OverlapReliefConfig
            Validated cell, symmetry, and overlap-relief settings.
        device : torch.device or str
            Explicit local execution device.

        Raises
        ------
        TypeError
            If config is not an OverlapReliefConfig.
        ValueError
            If the selected device is unsupported or unavailable.
        """
        if not isinstance(config, OverlapReliefConfig):
            raise TypeError("config must be an OverlapReliefConfig")
        target = torch.device(device)
        if target.type == "cpu":
            if target.index is not None:
                raise ValueError("CPU device must not have an index")
        elif target.type == "cuda":
            if not torch.cuda.is_available():
                raise ValueError("CUDA is not available")
            target = resolve_device(target)
            index = target.index
            if not 0 <= index < torch.cuda.device_count():
                raise ValueError(f"CUDA device index {index} is unavailable")
        else:
            raise ValueError("device must be 'cpu' or a CUDA device")
        self.config = config
        self.device = target

    def to_spec(self) -> BaseSpec:
        """Capture this packer's configuration and resolved execution device.

        Returns
        -------
        BaseSpec
            A JSON-serializable constructor recipe for a fresh packer.
        """
        from nvalchemi.specs import create_model_spec

        return create_model_spec(
            type(self), config=self.config, device=str(self.device)
        )

    def __call__(
        self,
        inputs: MolecularPackingInput,
        *,
        num_samples: int = 1,
        rng: torch.Generator | None = None,
        context: PackingContext | None = None,
        run_id: int | None = None,
        candidate_budget: int | None | Literal["config"] = "config",
        progress_callback: Callable[[OverlapReliefProgress], None] | None = None,
        **config_overrides: object,
    ) -> PackingResult[RigidMoleculeASUBatch]:
        """Generate up to num_samples local crystal starting structures.

        Parameters
        ----------
        inputs : MolecularPackingInput
            One validated formula-unit tensor input.
        num_samples : int, default=1
            Nonnegative accepted-output target.
        rng : torch.Generator, optional
            Random source for candidate generation.
        context : PackingContext, optional
            Caller-assigned run identity and rank.
        run_id : int, optional
            Alternative identity for a rank-zero local call.
        candidate_budget : int, None, or config, default=config
            Local candidate cap, or the cap from the configuration.
        progress_callback : callable, optional
            Called at performed convergence checks.
        **config_overrides
            Per-call OverlapReliefConfig values.

        Returns
        -------
        PackingResult[RigidMoleculeASUBatch]
            Compact ASU structures and one rank-local report.

        See Also
        --------
        pack : Full option and completion semantics.
        """
        return self.pack(
            inputs,
            num_samples=num_samples,
            rng=rng,
            context=context,
            run_id=run_id,
            candidate_budget=candidate_budget,
            progress_callback=progress_callback,
            **config_overrides,
        )

    def resolve_candidate_budget(
        self, *, num_samples: int, **pack_options: object
    ) -> int | None:
        """Resolve this call's candidate cap without drawing randomness.

        The generic generation driver uses this optional capability to report
        whether a requested output target has a finite candidate budget.
        ``progress_callback`` is a concrete pack option and is validated before
        configuration fields are evaluated.

        Parameters
        ----------
        num_samples : int
            Nonnegative number of accepted structures.
        **pack_options
            Configuration overrides and the optional progress callback.

        Returns
        -------
        int or None
            Resolved nonnegative cap, or None for unlimited candidates.

        Raises
        ------
        TypeError
            If the count or progress callback has an invalid type.
        ValueError
            If the count or effective configuration is invalid.
        """
        target_count = self._validate_num_samples(num_samples)
        options = dict(pack_options)
        callback = options.pop("progress_callback", None)
        if callback is not None and not callable(callback):
            raise TypeError("progress_callback must be callable or None")
        config = self.config.effective(**options)
        if config.max_candidates == "auto":
            return _DEFAULT_CANDIDATE_MULTIPLIER * target_count
        return config.max_candidates

    def pack(
        self,
        inputs: MolecularPackingInput,
        *,
        num_samples: int = 1,
        rng: torch.Generator | None = None,
        context: PackingContext | None = None,
        run_id: int | None = None,
        candidate_budget: int | None | Literal["config"] = "config",
        progress_callback: Callable[[OverlapReliefProgress], None] | None = None,
        **config_overrides: object,
    ) -> PackingResult[RigidMoleculeASUBatch]:
        """Generate and return compact ASU structures on this process only.

        ``num_samples`` is the accepted-output target. ``batch_size`` limits
        the active trial state, while ``candidate_budget`` independently limits
        total initialized candidates. The ``"config"`` default uses
        ``OverlapReliefConfig.max_candidates`` and resolves ``"auto"`` to
        ``1000 * num_samples``. Explicit ``None`` permits unlimited trials;
        zero requests a typed shortfall without sampling or trial setup.

        Parameters
        ----------
        inputs : MolecularPackingInput
            One validated formula-unit tensor input.
        num_samples : int, default=1
            Nonnegative requested number of accepted outputs.
        rng : torch.Generator, optional
            Generator used for one int64 seed draw when candidates are sampled.
            Zero target or zero candidate budget does not consume it.
        context : PackingContext, optional
            Call identity and rank used to assign rank-strided structure IDs.
        run_id : int, optional
            Alternative to ``context`` for a local rank-zero identity.
        candidate_budget : int or None or {"config"}, default="config"
            Local cap on initialized candidates; unlike the positive config cap,
            an explicit zero is allowed.
        progress_callback : callable, optional
            Called at performed convergence checks with cumulative counters and
            current overlap diagnostics.
        **config_overrides
            Per-call :class:`OverlapReliefConfig` values.

        Returns
        -------
        PackingResult[RigidMoleculeASUBatch]
            The accepted compact structures and one rank-local completion
            report. A finite budget can produce a shortfall.

        Raises
        ------
        RuntimeError
            If cell rejection sampling cannot fill candidate rows in 128 rounds.
        """
        if not isinstance(inputs, MolecularPackingInput):
            raise TypeError("inputs must be a MolecularPackingInput")
        target_count = self._validate_num_samples(num_samples)
        if rng is not None and not isinstance(rng, torch.Generator):
            raise TypeError("rng must be a torch.Generator or None")
        if context is not None and run_id is not None:
            raise ValueError("context and run_id cannot both be provided")
        if context is not None and not isinstance(context, PackingContext):
            raise TypeError("context must be a PackingContext or None")
        if run_id is not None and (
            isinstance(run_id, bool) or not isinstance(run_id, Integral)
        ):
            raise TypeError("run_id must be an integer or None")
        if run_id is not None and not 0 <= int(run_id) < 2**63:
            raise ValueError("run_id must satisfy 0 <= run_id < 2**63")
        if progress_callback is not None and not callable(progress_callback):
            raise TypeError("progress_callback must be callable or None")

        config = self.config.effective(**config_overrides)
        molecule_count = inputs.num_molecules * config.z_prime
        if molecule_count * target_count > torch.iinfo(torch.int32).max:
            raise OverflowError("structure_molecule_ptr exceeds int32 capacity")
        if isinstance(candidate_budget, str):
            if candidate_budget != "config":
                raise ValueError(
                    "candidate_budget must be a nonnegative integer, None, or 'config'"
                )
            budget = (
                _DEFAULT_CANDIDATE_MULTIPLIER * target_count
                if config.max_candidates == "auto"
                else config.max_candidates
            )
        elif candidate_budget is None:
            budget = None
        elif isinstance(candidate_budget, bool) or not isinstance(
            candidate_budget, Integral
        ):
            raise TypeError(
                "candidate_budget must be a nonnegative integer, None, or 'config'"
            )
        else:
            budget = int(candidate_budget)
            if budget < 0:
                raise ValueError("candidate_budget must be nonnegative")

        call_context = context or PackingContext(
            run_id=secrets.randbits(63) if run_id is None else int(run_id)
        )
        if target_count and (
            call_context.rank + call_context.world_size * (target_count - 1) >= 2**63
        ):
            raise OverflowError("rank-strided structure IDs exceed int64 capacity")

        device = self.device
        self._validate_input_placement(inputs, device)
        if target_count == 0 or budget == 0:
            stop_reason = (
                PackingStopReason.TARGET_REACHED
                if target_count == 0
                else PackingStopReason.CANDIDATE_BUDGET_EXHAUSTED
            )
            report = PackingReport(
                rank=call_context.rank,
                requested_count=target_count,
                accepted_count=0,
                generated_count=0,
                stop_reason=stop_reason.value,
            )
            return self._make_result(
                inputs=inputs,
                config=config,
                accepted_blocks=[],
                report=report,
                device=device,
                context=call_context,
            )

        base_seed = self._draw_seed(rng, device)
        if call_context.world_size > 1:
            base_seed = (
                base_seed + call_context.rank * _RANK_SEED_STRIDE
            ) % _WARP_SEED_MODULUS
        formula = {
            name: getattr(inputs, name).to(device=device)
            for name in (
                "conformer_positions",
                "conformer_ptr",
                "molecule_conformer_ptr",
                "molecule_atom_ptr",
                "contact_distances",
            )
        }
        operation_count = config.z // config.z_prime
        formula_atom_counts = (
            formula["molecule_atom_ptr"][1:] - formula["molecule_atom_ptr"][:-1]
        )
        asu_atom_counts = formula_atom_counts.repeat(config.z_prime)
        asu_molecule_atom_ptr = torch.cat(
            (
                torch.zeros((1,), dtype=torch.int32, device=device),
                asu_atom_counts.cumsum(dim=0).to(torch.int32),
            )
        )
        asu_conformer_starts = formula["molecule_conformer_ptr"][:-1].repeat(
            config.z_prime
        )
        asu_conformer_stops = formula["molecule_conformer_ptr"][1:].repeat(
            config.z_prime
        )
        asu_contact_distances = formula["contact_distances"].repeat(
            config.z_prime, config.z_prime
        )
        max_contact_distance = float(inputs.contact_distances.max().item())

        groups, probabilities, symmetry_table, op_indices, op_ptr, _ = sampling_tables(
            config=config,
            device=device,
        )
        count_cap = config.batch_size
        if budget is not None:
            count_cap = min(count_cap, budget)
        state = WorkingState.allocate(
            batch_size=count_cap,
            molecule_count=molecule_count,
            symmetry_operation_count=operation_count,
            device=device,
        )
        contact_maps = make_contact_maps(asu_molecule_atom_ptr, operation_count)
        expanded_atom_count = int(asu_molecule_atom_ptr[-1].item()) * operation_count
        contact_workspace = ContactWorkspace.allocate(
            batch_size=count_cap,
            num_molecules=molecule_count,
            expanded_atom_count=expanded_atom_count,
            device=device,
        )
        core = _OverlapReliefCore(
            packer=self,
            inputs=inputs,
            config=config,
            device=device,
            rank=call_context.rank,
            world_size=call_context.world_size,
            local_target=target_count,
            candidate_budget=budget,
            count_cap=count_cap,
            base_seed=base_seed,
            formula=formula,
            operation_count=operation_count,
            asu_molecule_atom_ptr=asu_molecule_atom_ptr,
            asu_conformer_starts=asu_conformer_starts,
            asu_conformer_stops=asu_conformer_stops,
            asu_contact_distances=asu_contact_distances,
            max_contact_distance=max_contact_distance,
            groups=groups,
            probabilities=probabilities,
            symmetry_table=symmetry_table,
            op_indices=op_indices,
            op_ptr=op_ptr,
            state=state,
            contact_maps=contact_maps,
            contact_workspace=contact_workspace,
            fresh_rows=torch.zeros((count_cap,), dtype=torch.bool, device=device),
            expiration_iterations=[None] * count_cap,
        )
        initial_count = min(count_cap, core.remaining_budget())
        initial_rows = torch.arange(initial_count, dtype=torch.int64, device=device)
        core.active_rows = core.refill(initial_rows).tolist()
        core.run(progress_callback=progress_callback)

        report = PackingReport(
            rank=call_context.rank,
            requested_count=target_count,
            accepted_count=core.accepted_total,
            generated_count=core.generated,
            stop_reason=core.stop_reason.value,
        )
        return self._make_result(
            inputs=inputs,
            config=config,
            accepted_blocks=core.accepted_blocks,
            report=report,
            device=device,
            context=call_context,
        )

    @staticmethod
    def _validate_num_samples(num_samples: int) -> int:
        """Return a validated nonnegative output target."""
        if isinstance(num_samples, bool) or not isinstance(num_samples, Integral):
            raise TypeError("num_samples must be a nonnegative integer")
        target = int(num_samples)
        if target < 0:
            raise ValueError("num_samples must be nonnegative")
        return target

    @staticmethod
    def _validate_input_placement(
        inputs: MolecularPackingInput, device: torch.device
    ) -> None:
        """Allow CPU or selected-device input and reject other accelerators."""
        for name in (
            "conformer_positions",
            "conformer_ptr",
            "molecule_conformer_ptr",
            "molecule_atom_ptr",
            "atomic_numbers",
            "contact_distances",
            "component_index",
            "component_charge",
        ):
            input_device = getattr(inputs, name).device
            if input_device.type != "cpu" and input_device != device:
                raise ValueError(
                    f"inputs.{name} is on {input_device}; packer device is {device}. "
                    "Move the input to CPU or the selected packer device first."
                )

    @staticmethod
    def _draw_seed(rng: torch.Generator | None, device: torch.device) -> int:
        """Draw one nonnegative 31-bit seed using the supplied Torch generator."""
        generator_device = device if rng is None else torch.device(rng.device)
        return int(
            torch.randint(
                0,
                _WARP_SEED_MODULUS,
                (),
                dtype=torch.int64,
                device=generator_device,
                generator=rng,
            ).item()
        )

    @staticmethod
    def _append_accepted(
        rows: Tensor,
        state: WorkingState,
        contacts: object,
        accepted_blocks: list[tuple[Tensor, ...]],
    ) -> int:
        """Append snapshots of accepted candidate state and contact metrics."""
        accepted_blocks.append(
            (
                state.conformer_ids.index_select(0, rows),
                state.rotations.index_select(0, rows),
                state.centers.index_select(0, rows),
                state.cells.index_select(0, rows),
                state.space_groups.index_select(0, rows),
                state.steps.index_select(0, rows),
                contacts.total_overlap.index_select(0, rows),
                contacts.max_overlap.index_select(0, rows),
            )
        )
        return int(rows.numel())

    @staticmethod
    def _progress(
        *,
        generated: int,
        accepted: int,
        active_count: int,
        converged_count: int,
        expired_count: int,
        replaced_count: int,
        iteration: int,
        total_snapshot: Tensor | None,
        max_snapshot: Tensor | None,
        rank: int,
        world_size: int,
    ) -> OverlapReliefProgress:
        """Build a progress snapshot with candidate counts and overlap summaries."""
        total = (
            float(total_snapshot.mean().item())
            if total_snapshot is not None and total_snapshot.numel()
            else 0.0
        )
        maximum = (
            float(max_snapshot.max().item())
            if max_snapshot is not None and max_snapshot.numel()
            else 0.0
        )
        return OverlapReliefProgress(
            generated_count=generated,
            accepted_count=accepted,
            active_count=active_count,
            converged_count=converged_count,
            expired_count=expired_count,
            replaced_count=replaced_count,
            iteration=iteration,
            total_overlap=total,
            max_overlap=maximum,
            rank=rank,
            world_size=world_size,
        )

    @staticmethod
    def _make_result(
        *,
        inputs: MolecularPackingInput,
        config: OverlapReliefConfig,
        accepted_blocks: list[tuple[Tensor, ...]],
        report: PackingReport,
        device: torch.device,
        context: PackingContext,
    ) -> PackingResult[RigidMoleculeASUBatch]:
        """Assemble accepted candidate blocks into an ASU packing result."""
        molecule_count = inputs.num_molecules * config.z_prime
        if accepted_blocks:
            fields = [
                torch.cat([block[index] for block in accepted_blocks], dim=0)
                for index in range(8)
            ]
            conformer_ids, rotations, centers, cells, groups, steps, total, maximum = (
                fields
            )
            structure_count = int(cells.shape[0])
        else:
            structure_count = 0
            conformer_ids = torch.empty(
                (0, molecule_count), dtype=torch.int32, device=device
            )
            rotations = torch.empty(
                (0, molecule_count, 3, 3), dtype=torch.float32, device=device
            )
            centers = torch.empty(
                (0, molecule_count, 3), dtype=torch.float32, device=device
            )
            cells = torch.empty((0, 3, 3), dtype=torch.float32, device=device)
            groups = torch.empty((0,), dtype=torch.int32, device=device)
            steps = torch.empty((0,), dtype=torch.int32, device=device)
            total = torch.empty((0,), dtype=torch.float32, device=device)
            maximum = torch.empty((0,), dtype=torch.float32, device=device)
        if structure_count * molecule_count > torch.iinfo(torch.int32).max:
            raise OverflowError("structure_molecule_ptr exceeds int32 capacity")
        structures = RigidMoleculeASUBatch(
            packing_input=inputs,
            structure_molecule_ptr=torch.arange(
                structure_count + 1, dtype=torch.int32, device=device
            )
            * molecule_count,
            conformer_indices=conformer_ids.reshape(-1),
            rotations=rotations.reshape(-1, 3, 3),
            fractional_centers=centers.reshape(-1, 3),
            cells=cells,
            space_groups=groups,
            z=torch.full(
                (structure_count,), config.z, dtype=torch.int32, device=device
            ),
            z_prime=torch.full(
                (structure_count,), config.z_prime, dtype=torch.int32, device=device
            ),
            structure_ids=context.structure_ids(structure_count, device=device),
            properties={"steps": steps, "total_overlap": total, "max_overlap": maximum},
        )
        return PackingResult(
            structures=structures,
            run_id=context.run_id,
            reports=(report,),
            scope="local",
        )
