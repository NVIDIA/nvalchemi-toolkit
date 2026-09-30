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

"""Single-device packing kernels with caller-managed process-group coordination."""

from __future__ import annotations

import hashlib
import json
import math
import secrets
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from numbers import Integral
from typing import Any

import torch
import torch.distributed as dist
from torch import Tensor
from torch.distributed import ProcessGroup

from nvalchemi.csp.data import MolecularPackingInput, RigidMoleculeASUBatch
from nvalchemi.csp.packer._cells import sample_valid_cells, sampling_tables
from nvalchemi.csp.packer._contacts import (
    ContactWorkspace,
    contact_forces,
    make_contact_maps,
)
from nvalchemi.csp.packer._state import WorkingState, relax_step
from nvalchemi.csp.packer.config import PackingConfig
from nvalchemi.csp.packer.result import (
    PackingProgress,
    PackingResult,
    PackingStopReason,
)

__all__ = ["CrystalPacker"]

_DEFAULT_CANDIDATE_MULTIPLIER = 1000


@dataclass
class _PackingCore:
    """Mutable local packing state advanced in bounded, resumable rounds."""

    packer: CrystalPacker
    inputs: MolecularPackingInput
    config: PackingConfig
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
        round_seed = (self.base_seed + self.generated * 104729 + count * 9176) % (
            2**31 - 1
        )
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

    def advance(
        self, *, max_iterations: int, progress_callback: Callable[..., Any] | None
    ) -> bool:
        """Advance local state by at most ``max_iterations`` relaxation steps."""
        for _ in range(max_iterations):
            if self.accepted_total >= self.local_target or not self.active_rows:
                break
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
        return self.accepted_total >= self.local_target or not self.active_rows


class CrystalPacker:
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
    config : PackingConfig
        Immutable base configuration containing ``z``, ``z_prime``, sampling
        controls, candidate capacity, and the optional finite candidate
        budget. The requested accepted count belongs to each call.
    device : torch.device or str
        Explicit execution device, either ``"cpu"`` or ``"cuda:N"``.
    """

    def __init__(self, config: PackingConfig, *, device: torch.device | str) -> None:
        """Bind the packing configuration to an explicitly selected CPU or CUDA device."""
        if not isinstance(config, PackingConfig):
            raise TypeError("config must be a PackingConfig")
        target = torch.device(device)
        if target.type == "cpu":
            if target.index is not None:
                raise ValueError("CPU device must not have an index")
        elif target.type == "cuda":
            if not torch.cuda.is_available():
                raise ValueError("CUDA is not available")
            index = (
                torch.cuda.current_device() if target.index is None else target.index
            )
            if not 0 <= index < torch.cuda.device_count():
                raise ValueError(f"CUDA device index {index} is unavailable")
            target = torch.device("cuda", index)
        else:
            raise ValueError("device must be 'cpu' or a CUDA device")
        self.config = config
        self.device = target

    def __call__(
        self,
        inputs: MolecularPackingInput,
        *,
        num_samples: int = 1,
        rng: torch.Generator | None = None,
        run_id: int | None = None,
        process_group: ProcessGroup | None = None,
        gather_to_rank: int | None = None,
        rank_targets: Sequence[int] | None = None,
        rank_candidate_budgets: Sequence[int] | None = None,
        progress_callback: Callable[[PackingProgress], None] | None = None,
        **config_overrides: object,
    ) -> PackingResult | None:
        """Generate up to ``num_samples`` crystal starting structures in ASU
        representation.

        Calls :meth:`pack`; see that method for trial-budget semantics and
        bounded pre-flight guidance.

        Parameters
        ----------
        inputs : MolecularPackingInput
            One validated formula-unit tensor input.
        num_samples : int, default=1
            Requested number of accepted outputs; a finite candidate budget
            may produce a shorter result.
        rng : torch.Generator, optional
            Generator used for one int64 seed draw on ranks that generate
            candidates. When omitted, the default Torch generator on the
            selected device is used; zero-target or zero-budget ranks return
            without a seed draw.
        run_id : int, optional
            Nonnegative identifier below ``2**63`` for this packing call. If
            omitted, a random ID is generated without consuming ``rng``; output
            rows carry ``[run_id, accepted_row_ordinal]`` identifiers in
            single-rank mode, or ``[run_id, group_rank + group_size *
            local_accept_ordinal]`` identifiers in group mode.
        process_group : ProcessGroup, optional
            Existing Gloo or NCCL group whose ranks participate in the call.
        gather_to_rank : int, optional
            Group-local destination for an ASU-result gather. ``None`` returns
            the local result on each rank.
        rank_targets : sequence of int, optional
            Per-rank accepted-output quotas summing to ``num_samples``. Without
            it, the target is divided by quotient and remainder.
        rank_candidate_budgets : sequence of int, optional
            Per-rank candidate caps summing to finite effective
            ``max_candidates``. Custom targets require these when that global
            candidate cap is finite. Automatic budgets are resolved from the
            global sample target before partitioning.
        progress_callback : callable, optional
            Called at performed convergence checks. A distributed rank with a
            zero target or zero candidate budget may return without a callback.
        **config_overrides
            Per-call :class:`PackingConfig` values. Unknown or invalid fields
            are rejected after full effective-config validation.

        Returns
        -------
        PackingResult or None
            Local ASU results on every rank when ``gather_to_rank`` is
            ``None``; otherwise the gathered result on that group-local rank
            and ``None`` on every other rank.

        Raises
        ------
        RuntimeError
            If cell rejection sampling cannot fill the requested candidate
            rows in 128 rounds.

        Notes
        -----
        On ranks that generate candidates, one Torch integer seed draw
        initializes deterministic Warp cell and candidate rounds.
        """
        return self.pack(
            inputs,
            num_samples=num_samples,
            rng=rng,
            run_id=run_id,
            process_group=process_group,
            gather_to_rank=gather_to_rank,
            rank_targets=rank_targets,
            rank_candidate_budgets=rank_candidate_budgets,
            progress_callback=progress_callback,
            **config_overrides,
        )

    def pack(
        self,
        inputs: MolecularPackingInput,
        *,
        num_samples: int = 1,
        rng: torch.Generator | None = None,
        run_id: int | None = None,
        process_group: ProcessGroup | None = None,
        gather_to_rank: int | None = None,
        rank_targets: Sequence[int] | None = None,
        rank_candidate_budgets: Sequence[int] | None = None,
        progress_callback: Callable[[PackingProgress], None] | None = None,
        **config_overrides: object,
    ) -> PackingResult | None:
        """Generate and return crystal starting structures in ASU representation.

        ``num_samples`` requests accepted structures; ``batch_size`` limits
        active trials. By default, ``max_candidates="auto"`` limits initialized
        trials to ``1000 * num_samples``. A positive integer sets a fixed cap;
        explicit ``None`` allows unlimited trials. Finite budgets can return
        fewer accepted structures than requested.

        An unlimited search assumes reachable acceptance criteria and can keep
        running without useful progress otherwise. Before a large or unlimited
        search, use a short pre-flight with a finite candidate budget and the
        intended starting-volume range, overlap tolerance, and per-candidate
        relaxation limit. Inspect accepted versus generated counts; low or zero
        acceptance can indicate an overly small starting volume. Revisit the
        volume settings and repeat the pre-flight before scaling up.

        Parameters
        ----------
        inputs : MolecularPackingInput
            One validated formula-unit tensor input.
        num_samples : int, default=1
            Requested number of accepted outputs; a finite candidate budget
            may produce a shorter result.
        rng : torch.Generator, optional
            Generator used for one int64 seed draw on ranks that generate
            candidates. When omitted, the default Torch generator on the
            selected device is used; zero-target or zero-budget ranks return
            without a seed draw.
        run_id : int, optional
            Nonnegative identifier below ``2**63`` for this packing call. If
            omitted, group rank zero generates it for a distributed call. IDs
            are ``[run_id, local_ordinal]`` in one-rank mode and
            ``[run_id, group_rank + group_size * local_ordinal]`` in a group.
        process_group : ProcessGroup, optional
            Existing Gloo or NCCL group whose ranks participate in the call.
        gather_to_rank : int, optional
            Group-local destination for an ASU-result gather. ``None`` returns
            the local result on each rank.
        rank_targets : sequence of int, optional
            Per-rank accepted-output quotas summing to ``num_samples``. Without
            it, the target is divided by quotient and remainder.
        rank_candidate_budgets : sequence of int, optional
            Per-rank candidate caps summing to finite effective
            ``max_candidates``. Custom targets require these when that global
            candidate cap is finite. Automatic budgets are resolved from the
            global sample target before partitioning.
        progress_callback : callable, optional
            Called at performed convergence checks. A distributed rank with a
            zero target or zero candidate budget may return without a callback.
        **config_overrides
            Per-call :class:`PackingConfig` values. Unknown or invalid fields
            are rejected after full effective-config validation.

        Returns
        -------
        PackingResult or None
            Local ASU result on every rank without a gather, or the gathered
            ASU result on ``gather_to_rank`` and ``None`` on all
            other ranks.

        Raises
        ------
        RuntimeError
            If cell rejection sampling cannot fill the requested candidate
            rows in 128 rounds.
        """
        grouped = process_group is not None
        validation_error: str | None = None
        try:
            if not isinstance(inputs, MolecularPackingInput):
                raise TypeError("inputs must be a MolecularPackingInput")
            if isinstance(num_samples, bool) or not isinstance(num_samples, Integral):
                raise TypeError("num_samples must be a positive integer")
            target_count = int(num_samples)
            if target_count <= 0:
                raise ValueError("num_samples must be positive")
            if rng is not None and not isinstance(rng, torch.Generator):
                raise TypeError("rng must be a torch.Generator or None")
            if run_id is not None and (
                isinstance(run_id, bool) or not isinstance(run_id, Integral)
            ):
                raise TypeError("run_id must be an integer or None")
            if run_id is not None and not 0 <= int(run_id) < 2**63:
                raise ValueError("run_id must satisfy 0 <= run_id < 2**63")
            if progress_callback is not None and not callable(progress_callback):
                raise TypeError("progress_callback must be callable or None")
            config = self.config.effective(**config_overrides)
            if config.max_candidates == "auto":
                config = config.effective(
                    max_candidates=_DEFAULT_CANDIDATE_MULTIPLIER * target_count
                )
        except Exception as error:
            if not grouped:
                raise
            validation_error = f"{type(error).__name__}: {error}"
        if grouped:
            if not dist.is_available() or not dist.is_initialized():
                raise RuntimeError(
                    "process_group requires an initialized process group"
                )
            validation_errors: list[Any] = [None] * dist.get_world_size(
                group=process_group
            )
            dist.all_gather_object(
                validation_errors, validation_error, group=process_group
            )
            errors = [value for value in validation_errors if value is not None]
            if errors:
                raise ValueError(f"distributed packer validation failed: {errors[0]}")
            identity_run_id = 0 if run_id is None else int(run_id)
        else:
            identity_run_id = secrets.randbits(63) if run_id is None else int(run_id)
        if not grouped and any(
            value is not None
            for value in (gather_to_rank, rank_targets, rank_candidate_budgets)
        ):
            raise ValueError(
                "gather_to_rank, rank_targets, and rank_candidate_budgets require process_group"
            )
        group_rank = 0
        group_size = 1
        local_target = target_count
        group_budget: int | None = config.max_candidates
        destination = gather_to_rank
        if grouped:
            (
                group_rank,
                group_size,
                local_target,
                group_budget,
                identity_run_id,
            ) = self._distributed_preflight(
                inputs=inputs,
                num_samples=target_count,
                config=config,
                run_id=run_id,
                process_group=process_group,
                gather_to_rank=gather_to_rank,
                rank_targets=rank_targets,
                rank_candidate_budgets=rank_candidate_budgets,
            )
            destination = gather_to_rank
        device = self.device
        if grouped and (local_target == 0 or group_budget == 0):
            empty_reason = (
                PackingStopReason.TARGET_REACHED
                if local_target == 0
                else PackingStopReason.CANDIDATE_BUDGET_EXHAUSTED
            )
            self._collective_error(
                process_group=process_group, error=None, device=device
            )
            self._wait_for_group_completion(process_group=process_group, device=device)
            empty_error: Exception | None = None
            try:
                result = self._make_result(
                    inputs=inputs,
                    config=config,
                    accepted_blocks=[],
                    generated=0,
                    stop_reason=empty_reason,
                    device=device,
                    run_id=identity_run_id,
                    rank=group_rank,
                    world_size=group_size,
                )
            except Exception as error:
                empty_error = error
            self._collective_error(
                process_group=process_group, error=empty_error, device=device
            )
            if destination is None:
                return result
            return self._gather_results(
                result,
                process_group=process_group,
                group_rank=group_rank,
                group_size=group_size,
                destination=destination,
                requested=target_count,
            )
        setup_error: Exception | None = None
        try:
            self._validate_input_placement(inputs, device)
            base_seed = self._draw_seed(rng, device)
            if grouped and group_size > 1:
                base_seed = (base_seed + group_rank * 2654435761) % (2**31 - 1)
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
            molecule_count = inputs.num_molecules * config.z_prime
            operation_count = config.z // config.z_prime
            candidate_budget = group_budget if grouped else config.max_candidates
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

            groups, probabilities, symmetry_table, op_indices, op_ptr, _ = (
                sampling_tables(
                    config=config,
                    device=device,
                )
            )
            count_cap = config.batch_size
            if candidate_budget is not None:
                count_cap = min(count_cap, candidate_budget)
            if grouped and local_target == 0:
                count_cap = 0
            state = WorkingState.allocate(
                batch_size=count_cap,
                molecule_count=molecule_count,
                symmetry_operation_count=operation_count,
                device=device,
            )
            contact_maps = make_contact_maps(asu_molecule_atom_ptr, operation_count)
            expanded_atom_count = (
                int(asu_molecule_atom_ptr[-1].item()) * operation_count
            )
            contact_workspace = ContactWorkspace.allocate(
                batch_size=count_cap,
                num_molecules=molecule_count,
                expanded_atom_count=expanded_atom_count,
                device=device,
            )
            core = _PackingCore(
                packer=self,
                inputs=inputs,
                config=config,
                device=device,
                rank=group_rank,
                world_size=group_size,
                local_target=local_target,
                candidate_budget=candidate_budget,
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
        except Exception as error:
            setup_error = error
        if grouped:
            self._collective_error(
                process_group=process_group, error=setup_error, device=device
            )
        elif setup_error is not None:
            raise setup_error
        while True:
            round_error: Exception | None = None
            local_done = False
            try:
                local_done = core.advance(
                    max_iterations=32 if grouped else 2**31,
                    progress_callback=progress_callback,
                )
            except Exception as error:
                round_error = error
            local_done = local_done or round_error is not None
            if grouped:
                control_device = (
                    device
                    if "nccl" in str(dist.get_backend(process_group)).lower()
                    else torch.device("cpu")
                )
                control = torch.tensor(
                    [int(round_error is not None), int(local_done)],
                    dtype=torch.int64,
                    device=control_device,
                )
                dist.all_reduce(control[0:1], op=dist.ReduceOp.MAX, group=process_group)
                dist.all_reduce(control[1:2], op=dist.ReduceOp.MIN, group=process_group)
                if int(control[0].item()):
                    if round_error is not None:
                        raise round_error
                    raise RuntimeError(
                        "a process-group peer failed during crystal packing"
                    )
                if int(control[1].item()):
                    break
            elif local_done:
                if round_error is not None:
                    raise round_error
                break
        result_error: Exception | None = None
        try:
            result = self._make_result(
                inputs=inputs,
                config=config,
                accepted_blocks=core.accepted_blocks,
                generated=core.generated,
                stop_reason=core.stop_reason,
                device=device,
                run_id=identity_run_id,
                rank=group_rank,
                world_size=group_size,
            )
        except Exception as error:
            result_error = error
        if grouped:
            self._collective_error(
                process_group=process_group, error=result_error, device=device
            )
        elif result_error is not None:
            raise result_error
        if grouped:
            if destination is None:
                return result
            return self._gather_results(
                result,
                process_group=process_group,
                group_rank=group_rank,
                group_size=group_size,
                destination=destination,
                requested=target_count,
            )
        return result

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
        ):
            input_device = getattr(inputs, name).device
            if input_device.type != "cpu" and input_device != device:
                raise ValueError(
                    f"inputs.{name} is on {input_device}; packer device is {device}. "
                    "Move the input to CPU or the selected packer device first."
                )

    def _distributed_preflight(
        self,
        *,
        inputs: MolecularPackingInput,
        num_samples: int,
        config: PackingConfig,
        run_id: int | None,
        process_group: Any,
        gather_to_rank: int | None,
        rank_targets: Sequence[int] | None,
        rank_candidate_budgets: Sequence[int] | None,
    ) -> tuple[int, int, int, int | None, int]:
        """Validate group-wide packing contracts and establish shared identity."""
        if not dist.is_available() or not dist.is_initialized():
            raise RuntimeError("process_group requires an initialized process group")
        group_rank = dist.get_rank(group=process_group)
        group_size = dist.get_world_size(group=process_group)
        shared: tuple[Any, ...] | None = None
        local_error: str | None = None
        validated_run_id: int | None = None
        try:
            self._validate_input_placement(inputs, self.device)
            backend = str(dist.get_backend(group=process_group)).lower()
            if "gloo" in backend and self.device.type != "cpu":
                raise ValueError("Gloo process groups require a CPU CrystalPacker")
            if "nccl" in backend and self.device.type != "cuda":
                raise ValueError("NCCL process groups require a CUDA CrystalPacker")
            if "gloo" not in backend and "nccl" not in backend:
                raise ValueError(
                    "CrystalPacker supports only Gloo and NCCL process groups"
                )
            destination = gather_to_rank
            if destination is not None:
                if isinstance(destination, bool) or not isinstance(
                    destination, Integral
                ):
                    raise TypeError(
                        "gather_to_rank must be a group-local integer or None"
                    )
                destination = int(destination)
                if not 0 <= destination < group_size:
                    raise ValueError("gather_to_rank must be a valid group-local rank")
            custom_targets = rank_targets is not None
            if custom_targets:
                targets = self._validate_rank_vector(
                    rank_targets, name="rank_targets", group_size=group_size
                )
                if sum(targets) != num_samples:
                    raise ValueError("rank_targets must sum to num_samples")
            else:
                quotient, remainder = divmod(num_samples, group_size)
                targets = tuple(
                    quotient + (rank < remainder) for rank in range(group_size)
                )
            maximum_id = (
                group_rank + group_size * (targets[group_rank] - 1)
                if targets[group_rank]
                else -1
            )
            if maximum_id >= 2**63:
                raise OverflowError("rank-strided structure IDs exceed int64 capacity")
            local_molecules = (
                inputs.num_molecules * config.z_prime * targets[group_rank]
            )
            if local_molecules > torch.iinfo(torch.int32).max:
                raise OverflowError(
                    "local structure_molecule_ptr exceeds int32 capacity"
                )
            if rank_candidate_budgets is not None:
                budgets = self._validate_rank_vector(
                    rank_candidate_budgets,
                    name="rank_candidate_budgets",
                    group_size=group_size,
                )
                if config.max_candidates is None:
                    raise ValueError(
                        "rank_candidate_budgets require finite max_candidates"
                    )
                if sum(budgets) != config.max_candidates:
                    raise ValueError(
                        "rank_candidate_budgets must sum to max_candidates"
                    )
            elif config.max_candidates is not None:
                if custom_targets:
                    raise ValueError(
                        "custom rank_targets with finite max_candidates require explicit rank_candidate_budgets"
                    )
                quotient, remainder = divmod(config.max_candidates, group_size)
                budgets = tuple(
                    quotient + (rank < remainder) for rank in range(group_size)
                )
            else:
                budgets = tuple(None for _ in range(group_size))
            if run_id is not None:
                if isinstance(run_id, bool) or not isinstance(run_id, Integral):
                    raise TypeError("run_id must be an integer or None")
                validated_run_id = int(run_id)
                if not 0 <= validated_run_id < 2**63:
                    raise ValueError("run_id must satisfy 0 <= run_id < 2**63")
            shared = (
                num_samples,
                config.max_candidates,
                targets,
                budgets,
                destination,
                self._formula_digest(inputs),
                backend,
            )
        except Exception as error:
            local_error = f"{type(error).__name__}: {error}"

        records: list[Any] = [None] * group_size
        dist.all_gather_object(
            records, (local_error, shared, validated_run_id), group=process_group
        )
        errors = [record[0] for record in records if record[0] is not None]
        if errors:
            raise ValueError(f"distributed packer preflight failed: {errors[0]}")
        shared_records = [record[1] for record in records]
        if any(value != shared_records[0] for value in shared_records[1:]):
            raise ValueError(
                "distributed ranks disagree on target, budget, quotas, destination, or formula input"
            )
        supplied_ids = [record[2] for record in records]
        root_id = supplied_ids[0]
        explicit_peer_ids = [value for value in supplied_ids[1:] if value is not None]
        if root_id is not None and any(value != root_id for value in explicit_peer_ids):
            raise ValueError("distributed ranks supplied different run_id values")
        if group_rank == 0 and root_id is None:
            root_id = secrets.randbits(63)
        backend = shared_records[0][6]
        identity_device = self.device if "nccl" in backend else torch.device("cpu")
        identity = torch.tensor(
            [0 if root_id is None else int(root_id)],
            dtype=torch.int64,
            device=identity_device,
        )
        dist.broadcast(
            identity,
            src=dist.get_global_rank(process_group, 0),
            group=process_group,
        )
        actual_run_id = int(identity.item())
        if any(value != actual_run_id for value in explicit_peer_ids):
            raise ValueError("distributed ranks supplied different run_id values")
        target_vector = shared_records[0][2]
        budget_vector = shared_records[0][3]
        return (
            group_rank,
            group_size,
            target_vector[group_rank],
            budget_vector[group_rank],
            int(identity.item()),
        )

    @staticmethod
    def _validate_rank_vector(
        values: Sequence[int], *, name: str, group_size: int
    ) -> tuple[int, ...]:
        """Validate one nonnegative integer value for each process-group rank."""
        if (
            isinstance(values, (str, bytes))
            or not isinstance(values, Sequence)
            or len(values) != group_size
        ):
            raise ValueError(f"{name} must contain one value per group rank")
        result: list[int] = []
        for value in values:
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
                raise ValueError(f"{name} values must be nonnegative integers")
            result.append(int(value))
        return tuple(result)

    @staticmethod
    def _collective_error(
        *, process_group: ProcessGroup, error: Exception | None, device: torch.device
    ) -> None:
        """Propagate one phase's catchable failure to every group rank."""
        backend = str(dist.get_backend(group=process_group)).lower()
        status_device = device if "nccl" in backend else torch.device("cpu")
        failure = torch.tensor(
            [int(error is not None)], dtype=torch.int64, device=status_device
        )
        dist.all_reduce(failure, op=dist.ReduceOp.MAX, group=process_group)
        if int(failure.item()):
            if error is not None:
                raise error
            raise RuntimeError("a process-group peer failed during crystal packing")

    @staticmethod
    def _wait_for_group_completion(
        *, process_group: ProcessGroup, device: torch.device
    ) -> None:
        """Join bounded status rounds without allocating local candidates."""
        backend = str(dist.get_backend(group=process_group)).lower()
        status_device = device if "nccl" in backend else torch.device("cpu")
        while True:
            status = torch.tensor([0, 1], dtype=torch.int64, device=status_device)
            dist.all_reduce(status[0:1], op=dist.ReduceOp.MAX, group=process_group)
            dist.all_reduce(status[1:2], op=dist.ReduceOp.MIN, group=process_group)
            if int(status[0].item()):
                raise RuntimeError("a process-group peer failed during crystal packing")
            if int(status[1].item()):
                return

    @staticmethod
    def _formula_digest(inputs: MolecularPackingInput) -> str:
        """Hash the formula tensors and metadata used for distributed input agreement."""
        digest = hashlib.sha256()
        for name in (
            "conformer_positions",
            "conformer_ptr",
            "molecule_conformer_ptr",
            "molecule_atom_ptr",
            "atomic_numbers",
            "contact_distances",
            "component_index",
        ):
            tensor = getattr(inputs, name).detach().cpu().contiguous()
            digest.update(name.encode())
            digest.update(str(tensor.dtype).encode())
            digest.update(json.dumps(list(tensor.shape)).encode())
            digest.update(tensor.numpy().tobytes())

        def to_jsonable(value: Any) -> Any:
            """Convert mappings and tuples recursively to JSON-compatible containers."""
            if isinstance(value, Mapping):
                return {key: to_jsonable(item) for key, item in value.items()}
            if isinstance(value, tuple):
                return [to_jsonable(item) for item in value]
            return value

        metadata = to_jsonable(inputs.metadata)
        digest.update(
            json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode()
        )
        digest.update(repr(float(inputs.formula_unit_volume)).encode())
        return digest.hexdigest()

    @staticmethod
    def _gather_tensor_rows(
        tensor: Tensor,
        *,
        process_group: ProcessGroup,
        group_rank: int,
        group_size: int,
        destination: int,
        row_counts: Sequence[int],
    ) -> list[Tensor] | None:
        """Gather a variable number of leading rows only to the destination."""
        sizes = [int(value) for value in row_counts]
        maximum = max(sizes, default=0)
        padded_bytes = (
            maximum
            * (math.prod(tensor.shape[1:]) if tensor.ndim > 1 else 1)
            * tensor.element_size()
        )
        size_error: Exception | None = None
        try:
            int32_max = torch.iinfo(torch.int32).max
            if len(sizes) != group_size or sizes[group_rank] != tensor.shape[0]:
                raise ValueError("compact gather row counts do not match local fields")
            if maximum > int32_max or sum(sizes) > int32_max:
                raise OverflowError(
                    "gathered compact rows exceed int32 pointer capacity"
                )
            destination_bytes = padded_bytes * group_size
            if destination_bytes > sys.maxsize:
                raise OverflowError(
                    "padded compact gather size exceeds addressable memory"
                )
        except Exception as error:
            size_error = error
        CrystalPacker._collective_error(
            process_group=process_group,
            error=size_error,
            device=tensor.device,
        )
        if maximum == 0:
            return [tensor[:0] for _ in sizes] if group_rank == destination else None
        allocation_error: Exception | None = None
        padded: Tensor | None = None
        receive: list[Tensor] | None = None
        try:
            padded = torch.zeros(
                (maximum, *tensor.shape[1:]), dtype=tensor.dtype, device=tensor.device
            )
            if tensor.shape[0]:
                padded[: tensor.shape[0]].copy_(tensor)
            receive = (
                [torch.empty_like(padded) for _ in range(group_size)]
                if group_rank == destination
                else None
            )
        except Exception as error:
            allocation_error = error
        CrystalPacker._collective_error(
            process_group=process_group,
            error=allocation_error,
            device=tensor.device,
        )
        if padded is None:
            raise RuntimeError("compact gather buffer allocation failed")
        dist.gather(
            padded,
            gather_list=receive,
            dst=dist.get_global_rank(process_group, destination),
            group=process_group,
        )
        if receive is None:
            return None
        return [value[:size] for value, size in zip(receive, sizes, strict=True)]

    @classmethod
    def _gather_results(
        cls,
        result: PackingResult,
        *,
        process_group: ProcessGroup,
        group_rank: int,
        group_size: int,
        destination: int,
        requested: int,
    ) -> PackingResult | None:
        """Gather ASU fields to one group-local destination and sync errors."""
        structures = result.structures
        counts_error: Exception | None = None
        local_counts: Tensor | None = None
        count_buffers: list[Tensor] | None = None
        try:
            local_counts = torch.tensor(
                [
                    structures.num_structures,
                    structures.conformer_indices.shape[0],
                ],
                dtype=torch.int64,
                device=structures.cells.device,
            )
            count_buffers = [torch.empty_like(local_counts) for _ in range(group_size)]
        except Exception as error:
            counts_error = error
        cls._collective_error(
            process_group=process_group,
            error=counts_error,
            device=structures.cells.device,
        )
        if local_counts is None or count_buffers is None:
            raise RuntimeError("compact gather count allocation failed")
        dist.all_gather(count_buffers, local_counts, group=process_group)
        structure_counts: list[int] | None = None
        molecule_counts: list[int] | None = None
        count_error: Exception | None = None
        try:
            structure_counts = [int(count[0].item()) for count in count_buffers]
            molecule_counts = [int(count[1].item()) for count in count_buffers]
            if sum(structure_counts) > torch.iinfo(torch.int32).max:
                raise OverflowError("gathered structure count exceeds int32 capacity")
            if sum(molecule_counts) > torch.iinfo(torch.int32).max:
                raise OverflowError(
                    "gathered structure_molecule_ptr exceeds int32 capacity"
                )
        except Exception as error:
            count_error = error
        cls._collective_error(
            process_group=process_group,
            error=count_error,
            device=structures.cells.device,
        )
        if structure_counts is None or molecule_counts is None:
            raise RuntimeError("compact gather count decoding failed")
        fields: tuple[Tensor, ...] | None = None
        field_error: Exception | None = None
        try:
            fields = (
                structures.cells,
                structures.space_groups,
                structures.z,
                structures.z_prime,
                structures.structure_ids,
                structures.conformer_indices,
                structures.rotations,
                structures.fractional_centers,
                structures.properties["steps"],
                structures.properties["total_overlap"],
                structures.properties["max_overlap"],
                torch.tensor(
                    [result.generated_count],
                    dtype=torch.int64,
                    device=structures.cells.device,
                ),
            )
        except Exception as error:
            field_error = error
        cls._collective_error(
            process_group=process_group,
            error=field_error,
            device=structures.cells.device,
        )
        if fields is None:
            raise RuntimeError("compact gather field preparation failed")
        assembly_error: Exception | None = None
        field_counts = (
            structure_counts,
            structure_counts,
            structure_counts,
            structure_counts,
            structure_counts,
            molecule_counts,
            molecule_counts,
            molecule_counts,
            structure_counts,
            structure_counts,
            structure_counts,
            [1] * group_size,
        )
        gathered = [
            cls._gather_tensor_rows(
                field,
                process_group=process_group,
                group_rank=group_rank,
                group_size=group_size,
                destination=destination,
                row_counts=counts,
            )
            for field, counts in zip(fields, field_counts, strict=True)
        ]
        gathered_result: PackingResult | None = None
        if group_rank == destination:
            try:
                chunks = [parts for parts in gathered[:-1] if parts is not None]
                if len(chunks) != 11:
                    raise RuntimeError("destination did not receive compact fields")
                rank_batches: list[RigidMoleculeASUBatch] = []
                for rank in range(group_size):
                    cells, groups, z, z_prime, ids = (
                        chunks[index][rank] for index in range(5)
                    )
                    conformers, rotations, centers = (
                        chunks[index][rank] for index in range(5, 8)
                    )
                    steps, total, maximum = (
                        chunks[index][rank] for index in range(8, 11)
                    )
                    rank_molecule_counts = (
                        structures.packing_input.num_molecules * z_prime.to(torch.int64)
                    )
                    if int(rank_molecule_counts.sum().item()) != molecule_counts[rank]:
                        raise ValueError(
                            "gathered ragged molecule count does not match z_prime"
                        )
                    rank_pointer = torch.cat(
                        (
                            torch.zeros((1,), dtype=torch.int32, device=cells.device),
                            rank_molecule_counts.cumsum(0).to(torch.int32),
                        )
                    )
                    rank_batches.append(
                        RigidMoleculeASUBatch(
                            packing_input=structures.packing_input,
                            structure_molecule_ptr=rank_pointer,
                            conformer_indices=conformers,
                            rotations=rotations,
                            fractional_centers=centers,
                            cells=cells,
                            space_groups=groups,
                            z=z,
                            z_prime=z_prime,
                            structure_ids=ids,
                            properties={
                                "steps": steps,
                                "total_overlap": total,
                                "max_overlap": maximum,
                            },
                        )
                    )
                compact = RigidMoleculeASUBatch.concatenate(rank_batches)
                generated_parts = gathered[-1]
                if generated_parts is None:
                    raise RuntimeError(
                        "destination rank did not receive generated counts"
                    )
                generated_count = sum(int(value.item()) for value in generated_parts)
                gathered_result = PackingResult(
                    structures=compact,
                    generated_count=generated_count,
                    stop_reason=(
                        PackingStopReason.TARGET_REACHED
                        if compact.num_structures == requested
                        else PackingStopReason.CANDIDATE_BUDGET_EXHAUSTED
                    ),
                    run_id=result.run_id,
                )
            except Exception as error:
                assembly_error = error
        cls._collective_error(
            process_group=process_group,
            error=assembly_error,
            device=structures.cells.device,
        )
        return gathered_result

    @staticmethod
    def _draw_seed(rng: torch.Generator | None, device: torch.device) -> int:
        """Draw one nonnegative 31-bit seed using the supplied Torch generator."""
        generator_device = device if rng is None else torch.device(rng.device)
        return int(
            torch.randint(
                0,
                2**31 - 1,
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
    ) -> PackingProgress:
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
        return PackingProgress(
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
        config: PackingConfig,
        accepted_blocks: list[tuple[Tensor, ...]],
        generated: int,
        stop_reason: PackingStopReason,
        device: torch.device,
        run_id: int,
        rank: int = 0,
        world_size: int = 1,
    ) -> PackingResult:
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
        if structure_count and rank + world_size * (structure_count - 1) >= 2**63:
            raise OverflowError("rank-strided structure IDs exceed int64 capacity")
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
            structure_ids=torch.stack(
                (
                    torch.full(
                        (structure_count,), run_id, dtype=torch.int64, device=device
                    ),
                    rank
                    + world_size
                    * torch.arange(structure_count, dtype=torch.int64, device=device),
                ),
                dim=1,
            ),
            properties={"steps": steps, "total_overlap": total, "max_overlap": maximum},
        )
        return PackingResult(
            structures=structures,
            generated_count=generated,
            stop_reason=stop_reason,
            run_id=run_id,
        )
