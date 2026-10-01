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
"""Generation driver for local and caller-coordinated CSP packing."""

from __future__ import annotations

import secrets
from collections.abc import Callable, Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from numbers import Integral
from typing import Any

import torch
import torch.distributed as dist
from pydantic import PrivateAttr
from torch.distributed import ProcessGroup

from nvalchemi._device import normalize_device
from nvalchemi.csp._packing_transport import (
    _assemble_result,
    _describe_native,
    _empty_batch,
    _gather_payloads,
)
from nvalchemi.csp.data import (
    MolecularPackingInput,
    RigidMoleculeASUBatch,
    _formula_inputs_equal,
)
from nvalchemi.csp.packer.protocol import CrystalPacker, PackingContext
from nvalchemi.csp.packer.result import PackingReport, PackingResult
from nvalchemi.data import Batch
from nvalchemi.distributed import ProcessGroupContext, collective_error_sync
from nvalchemi.gen.generator import AtomisticGenerator, _PreparedGeneration

__all__ = ["CSPGenerator", "CSP_OUTPUT_FIELDS"]

CSP_OUTPUT_FIELDS = frozenset(
    {"positions", "atomic_numbers", "cell", "pbc", "csp_source_structure_id"}
)

_GENERATOR_OPTIONS = frozenset(
    {
        "condition_func",
        "hooks",
        "seed",
        "num_samples",
        "dedicated_stream",
        "enable_inference_mode",
        "compile_generate",
    }
)
_CALL_ONLY_KEYS = frozenset(
    {
        "process_group",
        "gather_to_rank",
        "expand",
        "on_result",
        "generator_func",
        "device",
        "required_inputs",
        "outputs",
        "condition_func",
        "hooks",
        "seed",
        "dedicated_stream",
        "enable_inference_mode",
        "compile_generate",
        "compile_kwargs",
        "_csp_call_plan",
    }
)


@dataclass(frozen=True)
class _CSPCallPlan:
    """Validated immutable state passed to the packer for one generation call."""

    inputs: MolecularPackingInput
    global_target: int
    local_target: int
    rank: int
    world_size: int
    run_id: int
    context: PackingContext
    rng: torch.Generator | None
    pack_options: dict[str, Any]
    candidate_budget: int | None
    has_budget_capability: bool
    group_context: ProcessGroupContext | None
    gather_to_rank: int | None
    expand: bool


class _CSPGeneratingFunction:
    """Private callable carrying fixed declarations for AtomisticGenerator."""

    def __init__(self, *, device: torch.device, outputs: frozenset[str]) -> None:
        self.device = device
        self.required_inputs = frozenset()
        self.outputs = outputs
        self._driver: CSPGenerator | None = None

    def bind(self, driver: CSPGenerator) -> None:
        """Bind the completed adapter after normal generator initialization."""
        self._driver = driver

    def __call__(
        self,
        inputs: Any = None,
        *,
        num_samples: int = 1,
        rng: torch.Generator | None = None,
        **kwargs: Any,
    ) -> Any:
        """Delegate the call to the CSP driver."""
        if self._driver is None:
            raise RuntimeError("CSP generating function was not bound to its driver")
        return self._driver._pack_call(
            inputs, num_samples=num_samples, rng=rng, **kwargs
        )


class CSPGenerator(AtomisticGenerator):
    """Run one CSP packer call inside the Toolkit generation lifecycle.

    A generator owns the packer and uses its device and compact structures as
    the source of truth. With a process group, startup validates the shared
    request and assigns rank-strided structure identities before any rank
    packs. A configured destination gathers native payloads and expands only
    its result when requested.

    Execution placement comes from ``packer.device``. Scientific
    configurations describe search settings and do not select the device;
    an input Batch or downstream model does not choose it for this generator.

    For distributed calls with custom ``rank_targets``, omitted
    ``rank_candidate_budgets`` divide a finite global candidate cap in
    proportion to the requested rank targets. Integer largest remainders
    determine leftover trials, with ties assigned to the lower group-local
    rank. Explicit budgets retain precedence, and an unlimited cap stays
    unlimited.

    Parameters
    ----------
    packer
        CrystalPacker implementing device and pack(inputs, num_samples, rng,
        context).
    process_group
        Optional initialized process group. The caller owns group lifetime and
        rank launch.
    gather_to_rank
        Optional group-local destination for rank-major gathered output.
        None leaves each rank's result local.
    expand
        Expand rigid ASU structures to a Toolkit Batch on the owning rank.
        Existing Batch payloads pass through.
    on_result
        Optional callable receiving the owning PackingResult after any
        communication and before optional P1 expansion or generation hooks.
    **generator_options
        Supported AtomisticGenerator lifecycle options: conditioning, hooks,
        seed, default num_samples, dedicated stream, inference mode, and
        compile_generate. CSP packing is not torch.compile-compatible.
    """

    _packer: Any = PrivateAttr()
    _process_group: Any = PrivateAttr(default=None)
    _gather_to_rank: int | None = PrivateAttr(default=None)
    _expand: bool = PrivateAttr(default=True)
    _on_result: Any = PrivateAttr(default=None)

    def __init__(
        self,
        packer: CrystalPacker,
        *,
        process_group: ProcessGroup | None = None,
        gather_to_rank: int | None = None,
        expand: bool = True,
        on_result: Callable[[PackingResult], None] | None = None,
        **generator_options: Any,
    ) -> None:
        """Validate adapter options and initialize normal generator state."""
        unknown = set(generator_options) - _GENERATOR_OPTIONS
        if unknown:
            raise TypeError(
                f"unsupported CSPGenerator constructor option(s): {sorted(unknown)}"
            )
        if generator_options.get("compile_generate", False):
            raise NotImplementedError("CSPGenerator does not support compile_generate")
        if "num_samples" in generator_options:
            self._positive_integer(generator_options["num_samples"], name="num_samples")
        if not isinstance(expand, bool):
            raise TypeError("expand must be a bool")
        if on_result is not None and not callable(on_result):
            raise TypeError("on_result must be callable or None")
        if not callable(getattr(packer, "pack", None)):
            raise TypeError("packer must provide a callable pack method")
        try:
            packer_device = packer.device
        except AttributeError as error:
            raise TypeError("packer must provide a device") from error
        if not isinstance(packer_device, (torch.device, str)):
            raise TypeError("packer.device must be a torch.device or device string")
        device = normalize_device(packer_device)
        outputs = CSP_OUTPUT_FIELDS if expand else frozenset()
        generator_func = _CSPGeneratingFunction(device=device, outputs=outputs)
        super().__init__(
            generator_func=generator_func,
            device=device,
            required_inputs=frozenset(),
            outputs=outputs,
            **generator_options,
        )
        self._packer = packer
        self._process_group = process_group
        self._gather_to_rank = gather_to_rank
        self._expand = expand
        self._on_result = on_result
        generator_func.bind(self)

    @staticmethod
    def _positive_integer(value: Any, *, name: str) -> int:
        """Validate a positive integral generation target, excluding bool."""
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise TypeError(f"{name} must be a positive integer")
        result = int(value)
        if result <= 0:
            raise ValueError(f"{name} must be positive")
        return result

    @staticmethod
    def _rank_vector(values: Any, *, name: str, world_size: int) -> tuple[int, ...]:
        """Validate one nonnegative integer per process-group rank."""
        if (
            isinstance(values, (str, bytes))
            or not isinstance(values, Sequence)
            or len(values) != world_size
        ):
            raise ValueError(f"{name} must contain one value per group rank")
        result: list[int] = []
        for value in values:
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise TypeError(f"{name} values must be nonnegative integers")
            value = int(value)
            if value < 0:
                raise ValueError(f"{name} values must be nonnegative integers")
            result.append(value)
        return tuple(result)

    def compile(self, **kwargs: Any) -> CSPGenerator:
        """Reject driver-level compilation for the dynamic packing lifecycle."""
        raise NotImplementedError("CSPGenerator does not support compile()")

    def _startup_plan(
        self,
        prepared: _PreparedGeneration,
        *,
        rank: int,
        world_size: int,
        backend: str | None,
    ) -> dict[str, Any]:
        """Validate local startup state and return setup plus control metadata."""
        requested = prepared.num_samples
        target = self._positive_integer(requested, name="num_samples")
        active_inputs = prepared.ctx.inputs
        if not isinstance(active_inputs, MolecularPackingInput):
            raise TypeError("inputs must be a MolecularPackingInput")
        self._validate_input_device(active_inputs)

        source_kwargs = dict(prepared.kwargs)
        rng = prepared.rng
        if rng is not None and not isinstance(rng, torch.Generator):
            raise TypeError("rng must be a torch.Generator or None")
        self._validate_rng_device(rng)

        injected = set(source_kwargs) & _CALL_ONLY_KEYS
        if injected:
            raise TypeError(
                "CSPGenerator call-only options cannot be overridden per "
                f"call: {sorted(injected)}"
            )
        forbidden = set(source_kwargs) & {"context", "candidate_budget"}
        if forbidden:
            raise TypeError("context and candidate_budget are assigned by CSPGenerator")

        run_id = source_kwargs.pop("run_id", None)
        if run_id is not None and (
            isinstance(run_id, bool) or not isinstance(run_id, Integral)
        ):
            raise TypeError("run_id must be an integer or None")
        if run_id is not None and not 0 <= int(run_id) < 2**63:
            raise ValueError("run_id must satisfy 0 <= run_id < 2**63")
        rank_targets = source_kwargs.pop("rank_targets", None)
        rank_budgets = source_kwargs.pop("rank_candidate_budgets", None)
        if self._process_group is None and any(
            value is not None
            for value in (self._gather_to_rank, rank_targets, rank_budgets)
        ):
            raise ValueError(
                "gather_to_rank, rank_targets, and rank_candidate_budgets "
                "require process_group"
            )
        if backend is not None:
            device = self._packer_device
            if "gloo" in backend and device.type != "cpu":
                raise ValueError("Gloo process groups require a CPU CSP packer")
            if "nccl" in backend and device.type != "cuda":
                raise ValueError("NCCL process groups require a CUDA CSP packer")
            if backend not in {"gloo", "nccl"}:
                raise ValueError(
                    "CSPGenerator supports only Gloo and NCCL process groups"
                )

        if self._gather_to_rank is not None:
            if isinstance(self._gather_to_rank, bool) or not isinstance(
                self._gather_to_rank, Integral
            ):
                raise TypeError("gather_to_rank must be a group-local integer or None")
            destination = int(self._gather_to_rank)
            if not 0 <= destination < world_size:
                raise ValueError("gather_to_rank must be a valid group-local rank")
        else:
            destination = None

        if self._process_group is None:
            targets = (target,)
        elif rank_targets is None:
            quotient, remainder = divmod(target, world_size)
            targets = tuple(
                quotient + (group_rank < remainder) for group_rank in range(world_size)
            )
        else:
            targets = self._rank_vector(
                rank_targets, name="rank_targets", world_size=world_size
            )
            if sum(targets) != target:
                raise ValueError("rank_targets must sum to num_samples")
        local_target = targets[rank]
        maximum_id = rank + world_size * (local_target - 1) if local_target else -1
        if maximum_id >= 2**63:
            raise OverflowError("rank-strided structure IDs exceed int64 capacity")

        resolver = getattr(self._packer, "resolve_candidate_budget", None)
        budget_capability = callable(resolver)
        global_budget = (
            resolver(num_samples=target, **source_kwargs) if budget_capability else None
        )
        if global_budget is not None:
            if isinstance(global_budget, bool) or not isinstance(
                global_budget, Integral
            ):
                raise TypeError(
                    "resolve_candidate_budget must return a nonnegative integer or None"
                )
            global_budget = int(global_budget)
            if global_budget < 0:
                raise ValueError(
                    "resolve_candidate_budget must return a nonnegative integer or None"
                )

        if self._process_group is None:
            budgets: tuple[int | None, ...] = (global_budget,)
        elif rank_budgets is not None:
            explicit_budgets = self._rank_vector(
                rank_budgets,
                name="rank_candidate_budgets",
                world_size=world_size,
            )
            if not budget_capability or global_budget is None:
                raise ValueError(
                    "rank_candidate_budgets require a finite candidate budget capability"
                )
            if sum(explicit_budgets) != global_budget:
                raise ValueError(
                    "rank_candidate_budgets must sum to the global candidate budget"
                )
            budgets = explicit_budgets
        elif global_budget is None:
            budgets = tuple(None for _ in range(world_size))
        elif rank_targets is not None:
            allocations = tuple(
                divmod(global_budget * rank_target, target) for rank_target in targets
            )
            budget_values = [base for base, _ in allocations]
            remaining = global_budget - sum(budget_values)
            remainder_order = sorted(
                range(world_size),
                key=lambda group_rank: (-allocations[group_rank][1], group_rank),
            )
            for group_rank in remainder_order[:remaining]:
                budget_values[group_rank] += 1
            budgets = tuple(budget_values)
        else:
            quotient, remainder = divmod(global_budget, world_size)
            budgets = tuple(
                quotient + (group_rank < remainder) for group_rank in range(world_size)
            )

        rank_allocations = tuple(int(value) for value in targets)
        rank_budgets = tuple(budgets) if self._process_group is not None else None
        digest = active_inputs.sha256 if self._process_group is not None else None
        return {
            "inputs": active_inputs,
            "target": target,
            "local_target": local_target,
            "targets": rank_allocations,
            "budgets": budgets,
            "rank_budgets": rank_budgets,
            "global_budget": global_budget,
            "budget_capability": budget_capability,
            "run_id": None if run_id is None else int(run_id),
            "rng": rng,
            "pack_options": source_kwargs,
            "destination": destination,
            "digest": digest,
            "expand": self._expand,
            "shared_metadata": {
                "digest": digest,
                "global_target": target,
                "destination": destination,
                "expand": self._expand,
                "budget_capability": budget_capability,
                "global_cap": global_budget,
                "rank_allocations": rank_allocations,
                "rank_budgets": rank_budgets,
                "backend": backend,
                "supplied_run_id": None if run_id is None else int(run_id),
            },
        }

    def _finish_preparation(
        self,
        prepared: _PreparedGeneration | None,
        error: Exception | None,
    ) -> _PreparedGeneration:
        """Finish local preparation and agree grouped startup metadata."""
        if self._process_group is None:
            prepared = super()._finish_preparation(prepared, error)
            setup = self._startup_plan(prepared, rank=0, world_size=1, backend=None)
            run_id = (
                secrets.randbits(63) if setup["run_id"] is None else setup["run_id"]
            )
            context = PackingContext(run_id=run_id)
            plan = _CSPCallPlan(
                inputs=setup["inputs"],
                global_target=setup["target"],
                local_target=setup["local_target"],
                rank=0,
                world_size=1,
                run_id=run_id,
                context=context,
                rng=setup["rng"],
                pack_options=setup["pack_options"],
                candidate_budget=setup["global_budget"],
                has_budget_capability=setup["budget_capability"],
                group_context=None,
                gather_to_rank=None,
                expand=self._expand,
            )
            prepared.kwargs["_csp_call_plan"] = plan
            return prepared

        group_context = ProcessGroupContext(
            self._process_group, execution_device=self._packer_device
        )
        group_rank = group_context.rank
        world_size = group_context.world_size
        backend = group_context.backend
        with collective_error_sync(
            group_context,
            phase="generation_preparation",
            error_type=ValueError,
        ) as phase:
            prepared = super()._finish_preparation(prepared, error)
            phase.phase = "startup_validation"
            setup = self._startup_plan(
                prepared,
                rank=group_rank,
                world_size=world_size,
                backend=backend,
            )
            phase.metadata = setup["shared_metadata"]
        records = phase.records
        shared_fields = (
            "digest",
            "global_target",
            "destination",
            "expand",
            "budget_capability",
            "global_cap",
            "rank_allocations",
            "rank_budgets",
            "backend",
        )
        shared = records[0]
        differing = [
            rank
            for rank, record in enumerate(records[1:], start=1)
            if any(record[name] != shared[name] for name in shared_fields)
        ]
        if differing:
            raise ValueError(
                "CSPGenerator startup failed at "
                f"rank={differing[0]} phase=shared_request: ranks disagree on "
                "conditioned input, target, destination, expansion, budget "
                "capability, candidate cap, or rank allocations"
            )

        supplied_ids = [record["supplied_run_id"] for record in records]
        root_id = supplied_ids[0]
        if root_id is not None and any(
            value is not None and value != root_id for value in supplied_ids[1:]
        ):
            peer_rank = next(
                rank
                for rank, value in enumerate(supplied_ids[1:], start=1)
                if value is not None and value != root_id
            )
            raise ValueError(
                "CSPGenerator startup failed at "
                f"rank={peer_rank} phase=run_id: distributed ranks supplied "
                "different run_id values"
            )
        if group_rank == 0 and root_id is None:
            root_id = secrets.randbits(63)
        identity = torch.tensor(
            [0 if root_id is None else int(root_id)],
            dtype=torch.int64,
            device=group_context.collective_device,
        )
        with group_context.communication_scope():
            dist.broadcast(
                identity,
                src=group_context.global_rank(0),
                group=group_context.process_group,
            )
        run_id = int(identity.item())
        mismatched = [
            rank
            for rank, value in enumerate(supplied_ids)
            if value is not None and value != run_id
        ]
        if mismatched:
            raise ValueError(
                "CSPGenerator startup failed at "
                f"rank={mismatched[0]} phase=run_id: distributed ranks supplied "
                "different run_id values"
            )

        plan = _CSPCallPlan(
            inputs=setup["inputs"],
            global_target=setup["target"],
            local_target=setup["local_target"],
            rank=group_rank,
            world_size=world_size,
            run_id=run_id,
            context=PackingContext(
                run_id=run_id, rank=group_rank, world_size=world_size
            ),
            rng=setup["rng"],
            pack_options=setup["pack_options"],
            candidate_budget=setup["budgets"][group_rank],
            has_budget_capability=setup["budget_capability"],
            group_context=group_context,
            gather_to_rank=setup["destination"],
            expand=self._expand,
        )
        prepared.kwargs["_csp_call_plan"] = plan
        return prepared

    @property
    def _packer_device(self) -> torch.device:
        """Return the device resolved and pinned at adapter construction."""
        return self.device

    def _validate_input_device(self, inputs: MolecularPackingInput) -> None:
        """Require formula tensors to be CPU resident or on the packer device."""
        device = self._packer_device
        for name in (
            "conformer_positions",
            "conformer_ptr",
            "molecule_conformer_ptr",
            "molecule_atom_ptr",
            "atomic_numbers",
            "contact_distances",
            "component_index",
        ):
            source = getattr(inputs, name).device
            if source.type != "cpu" and source != device:
                raise ValueError(
                    f"inputs.{name} is on {source}; packer device is {device}. "
                    "Move the input to CPU or the selected packer device first."
                )

    def _validate_rng_device(self, rng: torch.Generator | None) -> None:
        """Permit CPU seed generators and same-device CUDA generators."""
        if rng is None:
            return
        rng_device = torch.device(rng.device)
        packer_device = self._packer_device
        if rng_device.type == "cpu":
            return
        if rng_device != packer_device:
            raise ValueError(
                f"rng is on {rng_device}; it must be CPU or match packer device "
                f"{packer_device}"
            )

    @staticmethod
    def _execution_scope(device: torch.device):
        """Run CUDA packing on its pinned device and restore the caller's device."""
        if device.type == "cuda":
            return torch.cuda.device(device)
        return nullcontext()

    def _pack_once(self, plan: _CSPCallPlan) -> PackingResult:
        """Call the packer once and validate the rank-local result contract."""
        kwargs = dict(plan.pack_options)
        if plan.has_budget_capability:
            kwargs["candidate_budget"] = plan.candidate_budget
        with self._execution_scope(self._packer_device):
            result = self._packer.pack(
                plan.inputs,
                num_samples=plan.local_target,
                rng=plan.rng,
                context=plan.context,
                **kwargs,
            )
        self._validate_result(result, plan)
        return result

    @staticmethod
    def _payload_count(structures: Any) -> int:
        """Return the leading structure count for a supported payload."""
        if isinstance(structures, RigidMoleculeASUBatch):
            return structures.num_structures
        if isinstance(structures, Batch):
            return structures.num_graphs
        raise TypeError(
            "PackingResult.structures must be a RigidMoleculeASUBatch or Batch"
        )

    @staticmethod
    def _payload_device(structures: RigidMoleculeASUBatch | Batch) -> torch.device:
        """Return the primary payload device."""
        if isinstance(structures, RigidMoleculeASUBatch):
            return structures.cells.device
        return structures.device

    def _validate_batch_payload(
        self,
        batch: Batch,
        *,
        accepted: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Validate common native Batch fields, levels, and occupied IDs."""
        levels = batch.level_keys
        node_fields = levels.get("atoms", set())
        system_fields = levels.get("system", set())
        missing_nodes = {"positions", "atomic_numbers"} - node_fields
        missing_system = {"cell", "pbc", "csp_source_structure_id"} - system_fields
        if missing_nodes or missing_system:
            raise ValueError(
                "native Batch payload must place positions and atomic_numbers "
                "at atom level and cell, pbc, and csp_source_structure_id at "
                f"system level; missing atom={sorted(missing_nodes)}, "
                f"system={sorted(missing_system)}"
            )
        for name in (
            "positions",
            "atomic_numbers",
            "cell",
            "pbc",
            "csp_source_structure_id",
        ):
            value = batch[name]
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"native Batch field {name!r} must be a Tensor")
            if value.device != device:
                raise ValueError(
                    f"native Batch field {name!r} is on {value.device}; "
                    f"expected {device}"
                )
        if batch.num_graphs != accepted:
            raise ValueError("native Batch graph count does not match accepted_count")
        if batch["positions"].ndim != 2 or batch["positions"].shape[1] != 3:
            raise ValueError("native Batch positions must have shape [N, 3]")
        if batch["atomic_numbers"].ndim != 1:
            raise ValueError("native Batch atomic_numbers must have shape [N]")
        if batch["cell"].shape != (accepted, 3, 3):
            raise ValueError("native Batch cell must have shape [P, 3, 3]")
        if batch["pbc"].shape != (accepted, 3):
            raise ValueError("native Batch pbc must have shape [P, 3]")
        ids = batch["csp_source_structure_id"]
        if ids.dtype != torch.int64 or ids.shape != (accepted, 2):
            raise ValueError(
                "native Batch csp_source_structure_id must be int64 [P, 2]"
            )
        return ids

    def _validate_result(self, result: Any, plan: _CSPCallPlan) -> None:
        """Check rank report, run identity, payload type, device, and IDs."""
        if not isinstance(result, PackingResult):
            raise TypeError("packer.pack must return a PackingResult")
        if result.scope != "local":
            raise ValueError("packer.pack must return a rank-local PackingResult")
        if result.run_id != plan.run_id:
            raise ValueError("PackingResult.run_id does not match PackingContext")
        if len(result.reports) != 1:
            raise ValueError("rank-local PackingResult must contain exactly one report")
        report = result.reports[0]
        if not isinstance(report, PackingReport):
            raise TypeError("PackingResult.reports must contain PackingReport values")
        if report.rank != plan.rank:
            raise ValueError("PackingReport.rank does not match PackingContext")
        if report.requested_count != plan.local_target:
            raise ValueError("PackingReport.requested_count does not match local quota")
        accepted = self._payload_count(result.structures)
        if report.accepted_count != accepted or result.accepted_count != accepted:
            raise ValueError("PackingResult accepted count does not match payload")
        if plan.candidate_budget == 0 and accepted != 0:
            raise ValueError("a zero candidate budget requires an empty PackingResult")
        if result.generated_count != report.generated_count:
            raise ValueError("PackingResult.generated_count does not match its report")
        if plan.has_budget_capability and report.generated_count is None:
            raise ValueError(
                "candidate-budget-capable packers must report generated_count"
            )
        if plan.candidate_budget is not None:
            if report.generated_count is None:
                raise ValueError(
                    "finite candidate budgets require a known generated_count"
                )
            if report.generated_count > plan.candidate_budget:
                raise ValueError(
                    "PackingReport.generated_count exceeds the local candidate budget"
                )

        device = self._packer_device
        payload_device = self._payload_device(result.structures)
        if payload_device != device:
            raise ValueError(
                f"PackingResult payload is on {payload_device}; expected packer "
                f"device {device}"
            )
        if isinstance(result.structures, RigidMoleculeASUBatch):
            if not _formula_inputs_equal(plan.inputs, result.structures.packing_input):
                raise ValueError(
                    "RigidMoleculeASUBatch.packing_input differs from the "
                    "conditioned formula input"
                )
            ids = result.structures.structure_ids
            if ids.dtype != torch.int64 or ids.shape != (accepted, 2):
                raise ValueError(
                    "RigidMoleculeASUBatch.structure_ids must be int64 [P, 2]"
                )
        else:
            ids = self._validate_batch_payload(
                result.structures,
                accepted=accepted,
                device=device,
            )
        expected_ids = plan.context.structure_ids(accepted, device=device)
        if not torch.equal(ids, expected_ids):
            raise ValueError("payload structure IDs do not match PackingContext")

    def _deliver_result(self, result: PackingResult, plan: _CSPCallPlan) -> Any:
        """Run the owning callback, then pass through or materialize the payload."""
        if self._on_result is not None:
            self._on_result(result)
        if not plan.expand:
            return result
        structures = result.structures
        if isinstance(structures, Batch):
            return structures
        if structures.num_structures == 0:
            return _empty_batch(structures, device=self._packer_device)
        return structures.to_batch(device=self._packer_device)

    def _pack_call(
        self,
        inputs: Any = None,
        *,
        num_samples: int = 1,
        rng: torch.Generator | None = None,
        **kwargs: Any,
    ) -> Any:
        """Execute local packing or the configured caller-owned gather."""
        del inputs, num_samples, rng
        plan = kwargs.pop("_csp_call_plan")
        if not isinstance(plan, _CSPCallPlan):
            raise TypeError("missing validated CSPGenerator call plan")
        if plan.group_context is None or plan.gather_to_rank is None:
            result = self._pack_once(plan)
            return self._deliver_result(result, plan)

        group_context = plan.group_context
        local_result: PackingResult
        metadata: dict[str, Any]
        with collective_error_sync(
            group_context, phase="local_packing", error_type=RuntimeError
        ) as phase:
            local_result = self._pack_once(plan)
            phase.phase = "native_metadata"
            metadata = _describe_native(local_result.structures)
            phase.metadata = {
                "report": local_result.reports[0],
                "metadata": metadata,
            }
            phase.phase = "terminal_record"
        records = phase.records
        native_metadata = [record["metadata"] for record in records]
        reference = native_metadata[0]
        incompatible = [
            rank
            for rank, item in enumerate(native_metadata[1:], start=1)
            if item["kind"] != reference["kind"]
            or item["layout"] != reference["layout"]
        ]
        if incompatible:
            raise RuntimeError(
                "CSPGenerator distributed native metadata failed at "
                f"rank={incompatible[0]} phase=pre_payload: ranks disagree on "
                "native payload kind or schema"
            )

        parts = _gather_payloads(
            local_result,
            metadata=native_metadata,
            inputs=plan.inputs,
            group_context=group_context,
            destination=plan.gather_to_rank,
        )
        gathered_result: PackingResult | None = None
        with collective_error_sync(
            group_context, phase="assembly", error_type=RuntimeError
        ) as phase:
            if group_context.rank == plan.gather_to_rank:
                reports = tuple(record["report"] for record in records)
                gathered_result = _assemble_result(
                    parts, run_id=plan.run_id, reports=reports
                )
                if not isinstance(gathered_result, PackingResult):
                    raise TypeError("_assemble_result must return a PackingResult")
                if gathered_result.scope != "gathered":
                    raise ValueError("_assemble_result must return scope='gathered'")
                if gathered_result.run_id != plan.run_id:
                    raise ValueError("_assemble_result returned the wrong run_id")
            phase.metadata = None

        if gathered_result is not None:
            return self._deliver_result(gathered_result, plan)
        if not plan.expand:
            return None
        return _empty_batch(local_result.structures, device=self._packer_device)
