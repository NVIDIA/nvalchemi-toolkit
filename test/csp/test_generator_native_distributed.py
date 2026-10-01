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
"""Distributed native-payload acceptance tests for CSPGenerator."""

from __future__ import annotations

import socket
from datetime import timedelta
from typing import Any

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from nvalchemi.csp import CSPGenerator
from nvalchemi.csp.data import MolecularPackingInput, RigidMoleculeASUBatch
from nvalchemi.csp.packer import PackingContext, PackingReport, PackingResult
from nvalchemi.data import AtomicData, Batch
from nvalchemi.data.level_storage import LevelSchema


def _formula() -> MolecularPackingInput:
    """Build a one-atom formula input for protocol-only packers."""
    return MolecularPackingInput(
        conformer_positions=torch.zeros((1, 3), dtype=torch.float32),
        conformer_ptr=torch.tensor([0, 1], dtype=torch.int32),
        molecule_conformer_ptr=torch.tensor([0, 1], dtype=torch.int32),
        molecule_atom_ptr=torch.tensor([0, 1], dtype=torch.int32),
        atomic_numbers=torch.tensor([6], dtype=torch.int64),
        contact_distances=torch.ones((1, 1), dtype=torch.float32),
        component_index=torch.tensor([0], dtype=torch.int32),
        component_charge=torch.zeros(1, dtype=torch.int32),
        formula_unit_volume=1000.0,
        metadata={"source": "native-driver-test"},
    )


def _available_tcp_port() -> int:
    """Return an ephemeral localhost port for a spawned process group."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def _run_gloo(worker: Any, *args: Any) -> list[dict[str, Any]]:
    """Run a two-rank Gloo case and collect one compact record per rank."""
    context = mp.get_context("spawn")
    queue = context.Queue()
    mp.spawn(
        worker,
        args=(_available_tcp_port(), queue, *args),
        nprocs=2,
        join=True,
    )
    records = [queue.get(timeout=10) for _ in range(2)]
    queue.close()
    return sorted(records, key=lambda record: record["rank"])


def _rigid_payload(
    inputs: MolecularPackingInput,
    context: PackingContext,
    *,
    rank: int,
    count: int,
    device: torch.device | str = "cpu",
) -> RigidMoleculeASUBatch:
    """Make ragged rigid rows with nonstandard, typed properties."""
    device = torch.device(device)
    z_prime = ([2] if rank == 0 else [1, 2])[:count]
    molecules = sum(z_prime)
    ptr = [0]
    for value in z_prime:
        ptr.append(ptr[-1] + value)
    return RigidMoleculeASUBatch(
        packing_input=inputs,
        structure_molecule_ptr=torch.tensor(ptr, dtype=torch.int32, device=device),
        conformer_indices=torch.zeros(molecules, dtype=torch.int32, device=device),
        rotations=torch.eye(3, device=device).expand(molecules, 3, 3).clone(),
        fractional_centers=torch.zeros((molecules, 3), device=device),
        cells=torch.eye(3, device=device).expand(count, 3, 3).clone() * 10,
        space_groups=torch.ones(count, dtype=torch.int32, device=device),
        z=torch.tensor(
            [2 * item for item in z_prime], dtype=torch.int32, device=device
        ),
        z_prime=torch.tensor(z_prime, dtype=torch.int32, device=device),
        structure_ids=context.structure_ids(count, device=device),
        properties={
            "quality_vector": torch.tensor(
                [[rank * 10 + ordinal, ordinal + 0.25] for ordinal in range(count)],
                dtype=torch.float64,
                device=device,
            ).reshape(count, 2),
            "candidate_code": torch.tensor(
                [[rank, ordinal, 7] for ordinal in range(count)],
                dtype=torch.int32,
                device=device,
            ).reshape(count, 3),
        },
    )


class _RigidProtocolPacker:
    """Structural implementation of the documented packer protocol."""

    device = torch.device("cpu")

    def __init__(self) -> None:
        self.option_keys: list[tuple[str, ...]] = []

    def pack(
        self,
        inputs: MolecularPackingInput,
        *,
        num_samples: int,
        rng: torch.Generator | None,
        context: PackingContext,
        **options: Any,
    ) -> PackingResult:
        del rng
        self.option_keys.append(tuple(sorted(options)))
        accepted = num_samples if context.rank == 0 else min(num_samples, 1)
        structures = _rigid_payload(
            inputs,
            context,
            rank=context.rank,
            count=accepted,
        )
        return PackingResult(
            structures=structures,
            run_id=context.run_id,
            reports=(
                PackingReport(
                    rank=context.rank,
                    requested_count=num_samples,
                    accepted_count=accepted,
                    generated_count=None,
                    stop_reason=(
                        "target_reached"
                        if accepted == num_samples
                        else "custom_shortfall"
                    ),
                ),
            ),
        )


def _rigid_gloo_worker(rank: int, port: int, queue: Any) -> None:
    """Exercise local and gathered compact results from a structural packer."""
    dist.init_process_group(
        backend="gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=30),
    )
    try:
        original_all_gather_object = dist.all_gather_object
        original_broadcast = dist.broadcast
        driver_counts = {"all_gather_object": 0, "broadcast": 0}

        def count_all_gather_object(*args: Any, **kwargs: Any) -> None:
            driver_counts["all_gather_object"] += 1
            original_all_gather_object(*args, **kwargs)

        def count_broadcast(*args: Any, **kwargs: Any) -> None:
            driver_counts["broadcast"] += 1
            original_broadcast(*args, **kwargs)

        dist.all_gather_object = count_all_gather_object
        dist.broadcast = count_broadcast
        packer = _RigidProtocolPacker()
        local_callbacks: list[PackingResult] = []
        local = CSPGenerator(
            packer,
            process_group=dist.group.WORLD,
            expand=False,
            on_result=local_callbacks.append,
            dedicated_stream=False,
        ).sample(
            _formula(),
            num_samples=3,
            rank_targets=(1, 2),
            run_id=701,
            option_marker="kept",
        )
        assert isinstance(local, PackingResult)
        assert local_callbacks == [local]
        assert driver_counts == {"all_gather_object": 1, "broadcast": 1}

        gathered_callbacks: list[PackingResult] = []
        gathered = CSPGenerator(
            packer,
            process_group=dist.group.WORLD,
            gather_to_rank=1,
            expand=False,
            on_result=gathered_callbacks.append,
            dedicated_stream=False,
        ).sample(
            _formula(),
            num_samples=3,
            rank_targets=(1, 2),
            run_id=702,
            option_marker="kept",
        )
        if rank == 0:
            assert gathered is None
            assert gathered_callbacks == []
            gathered_payload = None
        else:
            assert isinstance(gathered, PackingResult)
            assert gathered_callbacks == [gathered]
            assert gathered.scope == "gathered"
            gathered_payload = {
                "ids": gathered.structures.structure_ids.tolist(),
                "z_prime": gathered.structures.z_prime.tolist(),
                "ptr": gathered.structures.structure_molecule_ptr.tolist(),
                "quality": gathered.structures.properties["quality_vector"].tolist(),
                "quality_dtype": str(
                    gathered.structures.properties["quality_vector"].dtype
                ),
                "codes": gathered.structures.properties["candidate_code"].tolist(),
                "code_dtype": str(
                    gathered.structures.properties["candidate_code"].dtype
                ),
                "reports": [
                    (
                        report.rank,
                        report.requested_count,
                        report.accepted_count,
                        report.generated_count,
                        report.stop_reason,
                    )
                    for report in gathered.reports
                ],
            }
        assert driver_counts == {"all_gather_object": 4, "broadcast": 2}
        queue.put(
            {
                "rank": rank,
                "local_ids": local.structures.structure_ids.tolist(),
                "local_generated": local.generated_count,
                "local_reason": local.reports[0].stop_reason,
                "local_options": packer.option_keys,
                "gathered": gathered_payload,
                "driver_counts": dict(driver_counts),
            }
        )
    finally:
        dist.all_gather_object = original_all_gather_object
        dist.broadcast = original_broadcast
        dist.destroy_process_group()


def _native_schema() -> LevelSchema:
    """Describe built-in levels and one level of each custom storage kind."""
    schema = LevelSchema()
    schema.add_level("metadata", segmented=False)
    schema.add_level("molecules", segmented=True)
    schema.add_product_level("atom_molecules", left="atoms", right="molecules")
    schema.set("metadata_values", "metadata", dtype=torch.float32)
    schema.set("molecule_values", "molecules", dtype=torch.int32)
    schema.set("atom_molecule_values", "atom_molecules", dtype=torch.float32)
    return schema


def _native_batch(
    context: PackingContext,
    *,
    count: int,
    device: torch.device | str,
    metadata_width: int = 1,
    include_charge: bool = True,
) -> Batch:
    """Build native Batch payloads with built-in and custom level data."""
    device = torch.device(device)
    schema = _native_schema()
    rows = []
    for ordinal in range(count):
        rank = context.rank
        molecule_count = 1 + (rank + ordinal) % 2
        data = AtomicData(
            positions=torch.tensor(
                [
                    [float(rank), float(ordinal), 0.0],
                    [float(rank), float(ordinal), 1.0],
                ],
                dtype=torch.float32,
                device=device,
            ),
            atomic_numbers=torch.tensor([6, 1], dtype=torch.int64, device=device),
            neighbor_list=torch.tensor([[0, 1]], dtype=torch.int64, device=device),
            cell=torch.eye(3, dtype=torch.float32, device=device).unsqueeze(0) * 10,
            pbc=torch.ones((1, 3), dtype=torch.bool, device=device),
        )
        data.add_system_property(
            "csp_source_structure_id",
            context.structure_ids(ordinal + 1, device=device)[ordinal : ordinal + 1],
        )
        if include_charge:
            data.add_system_property(
                "charge", torch.zeros((1, 1), dtype=torch.float32, device=device)
            )
        marker = rank * 10 + ordinal
        data.metadata_values = torch.full(
            (1, metadata_width),
            marker,
            dtype=torch.float32,
            device=device,
        )
        data.molecule_values = (
            torch.arange(molecule_count, dtype=torch.int32, device=device).reshape(
                molecule_count, 1
            )
            + marker
        )
        data.atom_molecule_values = torch.full(
            (2, molecule_count, 1),
            marker + 0.5,
            dtype=torch.float32,
            device=device,
        )
        rows.append(data)

    if rows:
        return Batch.from_data_list(rows, device=device, attr_map=schema)

    template = _native_batch(
        context,
        count=1,
        device=device,
        metadata_width=metadata_width,
        include_charge=include_charge,
    )
    return Batch.empty(
        num_systems=0,
        num_nodes=0,
        num_edges=0,
        template=template,
        device=device,
        attr_map=schema,
        level_capacities={"molecules": 0, "atom_molecules": 0},
    )


class _BatchProtocolPacker:
    """Structural packer that returns a native Batch without inheritance."""

    def __init__(
        self,
        device: torch.device | str,
        *,
        metadata_width: int = 1,
        include_charge: bool = True,
    ):
        self.device = torch.device(device)
        self.metadata_width = metadata_width
        self.include_charge = include_charge
        self.option_keys: list[tuple[str, ...]] = []

    def pack(
        self,
        inputs: MolecularPackingInput,
        *,
        num_samples: int,
        rng: torch.Generator | None,
        context: PackingContext,
        **options: Any,
    ) -> PackingResult:
        del inputs, rng
        self.option_keys.append(tuple(sorted(options)))
        structures = _native_batch(
            context,
            count=num_samples,
            device=self.device,
            metadata_width=self.metadata_width,
            include_charge=self.include_charge,
        )
        return PackingResult(
            structures=structures,
            run_id=context.run_id,
            reports=(
                PackingReport(
                    rank=context.rank,
                    requested_count=num_samples,
                    accepted_count=num_samples,
                    generated_count=None,
                    stop_reason="target_reached",
                ),
            ),
        )


class _MalformedNativeBatchPacker:
    """Return one malformed native Batch for public boundary validation."""

    device = torch.device("cpu")

    def __init__(self, problem: str) -> None:
        self.problem = problem

    def pack(
        self,
        inputs: MolecularPackingInput,
        *,
        num_samples: int,
        rng: torch.Generator | None,
        context: PackingContext,
        **options: Any,
    ) -> PackingResult:
        del inputs, rng, options
        data = AtomicData(
            positions=torch.zeros((1, 3), dtype=torch.float32),
            atomic_numbers=torch.tensor([6], dtype=torch.int64),
        )
        if self.problem != "missing_fields":
            data.cell = torch.eye(3).unsqueeze(0)
            data.pbc = torch.ones((1, 3), dtype=torch.bool)
            if self.problem == "id_dtype":
                ids = context.structure_ids(1, device="cpu").to(torch.int32)
            elif self.problem == "id_sequence":
                ids = torch.tensor([[context.run_id, 19]], dtype=torch.int64)
            else:
                raise ValueError(f"unknown malformed Batch fixture: {self.problem}")
            data.add_system_property("csp_source_structure_id", ids)
            data.add_system_property("charge", torch.zeros((1, 1), dtype=torch.float32))
        structures = Batch.from_data_list([data], device="cpu")
        return PackingResult(
            structures=structures,
            run_id=context.run_id,
            reports=(
                PackingReport(
                    rank=context.rank,
                    requested_count=num_samples,
                    accepted_count=1,
                    generated_count=None,
                    stop_reason="target_reached",
                ),
            ),
        )


@pytest.mark.parametrize(
    ("problem", "message"),
    [
        ("missing_fields", "native Batch payload must place positions"),
        ("id_dtype", "must be int64"),
        ("id_sequence", "payload structure IDs do not match PackingContext"),
    ],
)
def test_malformed_native_batch_fails_through_public_generator(
    problem: str,
    message: str,
) -> None:
    generator = CSPGenerator(
        _MalformedNativeBatchPacker(problem), dedicated_stream=False
    )
    with pytest.raises(ValueError, match=message):
        generator.sample(_formula(), num_samples=1, run_id=820)


def _batch_summary(batch: Batch) -> dict[str, Any]:
    """Expose public payload values and pointers for queue assertions."""
    summary = {
        "graphs": batch.num_graphs,
        "ids": batch["csp_source_structure_id"].tolist(),
        "metadata": batch["metadata_values"].tolist(),
        "positions": batch["positions"].tolist(),
        "atomic_numbers": batch["atomic_numbers"].tolist(),
        "cell": batch["cell"].tolist(),
        "pbc": batch["pbc"].tolist(),
        "edge_ptr": batch.level_ptr("edges").tolist(),
        "molecule_ptr": batch.level_ptr("molecules").tolist(),
        "product_ptr": batch.level_ptr("atom_molecules").tolist(),
        "molecule_values": batch["molecule_values"].tolist(),
        "product_values": batch["atom_molecule_values"].tolist(),
        "level_keys": {key: sorted(value) for key, value in batch.level_keys.items()},
        "empty_shapes": {
            "metadata": list(batch["metadata_values"].shape),
            "molecules": list(batch["molecule_values"].shape),
            "products": list(batch["atom_molecule_values"].shape),
        },
        "builtin_shapes": {
            "positions": list(batch["positions"].shape),
            "atomic_numbers": list(batch["atomic_numbers"].shape),
            "neighbor_list": list(batch["neighbor_list"].shape),
            "cell": list(batch["cell"].shape),
            "pbc": list(batch["pbc"].shape),
            "ids": list(batch["csp_source_structure_id"].shape),
        },
    }
    if "charge" in batch.level_keys.get("system", set()):
        summary["charge"] = batch["charge"].tolist()
        summary["builtin_shapes"]["charge"] = list(batch["charge"].shape)
    else:
        summary["charge"] = None
    return summary


def _batch_gloo_worker(rank: int, port: int, queue: Any) -> None:
    """Test native Batch pass-through, schema gather, and a zero-quota rank."""
    dist.init_process_group(
        backend="gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=30),
    )
    try:
        packer = _BatchProtocolPacker("cpu")
        local_callbacks: list[PackingResult] = []
        local = CSPGenerator(
            packer,
            process_group=dist.group.WORLD,
            expand=True,
            on_result=local_callbacks.append,
            dedicated_stream=False,
        ).sample(
            _formula(),
            num_samples=2,
            rank_targets=(1, 1),
            run_id=801,
            batch_marker=5,
        )
        assert isinstance(local, Batch)
        assert len(local_callbacks) == 1
        assert local is local_callbacks[0].structures

        gathered_callbacks: list[PackingResult] = []
        gathered = CSPGenerator(
            packer,
            process_group=dist.group.WORLD,
            gather_to_rank=1,
            expand=True,
            on_result=gathered_callbacks.append,
            dedicated_stream=False,
        ).sample(
            _formula(),
            num_samples=2,
            rank_targets=(1, 1),
            run_id=802,
            batch_marker=6,
        )
        if rank == 0:
            assert isinstance(gathered, Batch)
            assert gathered.num_graphs == 0
            assert gathered_callbacks == []
            gathered_info = _batch_summary(gathered)
        else:
            assert isinstance(gathered, Batch)
            assert len(gathered_callbacks) == 1
            assert gathered is gathered_callbacks[0].structures
            assert gathered_callbacks[0].scope == "gathered"
            gathered_info = _batch_summary(gathered)
            gathered_info["reports"] = [
                (report.rank, report.requested_count, report.accepted_count)
                for report in gathered_callbacks[0].reports
            ]

        quota_callbacks: list[PackingResult] = []
        zero_quota = CSPGenerator(
            packer,
            process_group=dist.group.WORLD,
            gather_to_rank=1,
            expand=True,
            on_result=quota_callbacks.append,
            dedicated_stream=False,
        ).sample(
            _formula(),
            num_samples=2,
            rank_targets=(2, 0),
            run_id=803,
            batch_marker=7,
        )
        assert isinstance(zero_quota, Batch)
        if rank == 0:
            assert zero_quota.num_graphs == 0
            assert quota_callbacks == []
        else:
            assert len(quota_callbacks) == 1
            assert zero_quota is quota_callbacks[0].structures
            assert [report.accepted_count for report in quota_callbacks[0].reports] == [
                2,
                0,
            ]

        empty_sender = CSPGenerator(
            packer,
            process_group=dist.group.WORLD,
            gather_to_rank=1,
            expand=True,
            dedicated_stream=False,
        ).sample(
            _formula(),
            num_samples=2,
            rank_targets=(0, 2),
            run_id=804,
            batch_marker=8,
        )
        assert isinstance(empty_sender, Batch)
        if rank == 0:
            assert empty_sender.num_graphs == 0
        else:
            assert empty_sender["csp_source_structure_id"].tolist() == [
                [804, 1],
                [804, 3],
            ]

        charge_free_packer = _BatchProtocolPacker("cpu", include_charge=False)
        charge_free_local = CSPGenerator(
            charge_free_packer,
            process_group=dist.group.WORLD,
            expand=True,
            dedicated_stream=False,
        ).sample(
            _formula(),
            num_samples=2,
            rank_targets=(1, 1),
            run_id=805,
        )
        charge_free_gathered = CSPGenerator(
            charge_free_packer,
            process_group=dist.group.WORLD,
            gather_to_rank=1,
            expand=True,
            dedicated_stream=False,
        ).sample(
            _formula(),
            num_samples=2,
            rank_targets=(1, 1),
            run_id=806,
        )
        charge_free_zero_quota = CSPGenerator(
            charge_free_packer,
            process_group=dist.group.WORLD,
            gather_to_rank=1,
            expand=True,
            dedicated_stream=False,
        ).sample(
            _formula(),
            num_samples=2,
            rank_targets=(2, 0),
            run_id=807,
        )
        queue.put(
            {
                "rank": rank,
                "local": _batch_summary(local),
                "gathered": gathered_info,
                "zero_quota": _batch_summary(zero_quota),
                "empty_sender": _batch_summary(empty_sender),
                "charge_free_local": _batch_summary(charge_free_local),
                "charge_free_gathered": _batch_summary(charge_free_gathered),
                "charge_free_zero_quota": _batch_summary(charge_free_zero_quota),
                "options": packer.option_keys,
            }
        )
    finally:
        dist.destroy_process_group()


class _DisagreeingPacker:
    """Produce rank-varying native kinds or Batch layouts for preflight tests."""

    device = torch.device("cpu")

    def __init__(self, mode: str, rank: int) -> None:
        self.mode = mode
        self.rank = rank
        self.batch = _BatchProtocolPacker(
            "cpu", include_charge=mode != "charge" or rank == 0
        )

    def pack(
        self,
        inputs: MolecularPackingInput,
        *,
        num_samples: int,
        rng: torch.Generator | None,
        context: PackingContext,
        **options: Any,
    ) -> PackingResult:
        if self.mode == "kind" and self.rank == 0:
            structures: RigidMoleculeASUBatch | Batch = _rigid_payload(
                inputs, context, rank=self.rank, count=num_samples
            )
        else:
            result = self.batch.pack(
                inputs,
                num_samples=num_samples,
                rng=rng,
                context=context,
                **options,
            )
            if self.mode == "schema" and self.rank == 1:
                result.structures._storage.attr_map.set(
                    "metadata_values", "metadata", dtype=torch.float64
                )
            return result
        return PackingResult(
            structures=structures,
            run_id=context.run_id,
            reports=(
                PackingReport(
                    rank=context.rank,
                    requested_count=num_samples,
                    accepted_count=num_samples,
                    stop_reason="target_reached",
                    generated_count=None,
                ),
            ),
        )


def _metadata_mismatch_worker(rank: int, port: int, queue: Any, mode: str) -> None:
    """Verify incompatible representations/layouts fail before payload P2P."""
    dist.init_process_group(
        backend="gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=30),
    )
    try:
        generator = CSPGenerator(
            _DisagreeingPacker(mode, rank),
            process_group=dist.group.WORLD,
            gather_to_rank=0,
            expand=False,
            dedicated_stream=False,
        )
        try:
            generator.sample(
                _formula(),
                num_samples=2,
                rank_targets=(1, 1),
                run_id=811,
            )
        except RuntimeError as error:
            message = str(error)
        else:
            message = "missing expected native metadata error"
        queue.put({"rank": rank, "message": message})
    finally:
        dist.destroy_process_group()


class _CudaBatchPacker(_BatchProtocolPacker):
    """CUDA native Batch packer with an optional catchable rank-local failure."""

    def __init__(self, device: torch.device, *, fail_rank: int | None = None):
        super().__init__(device)
        self.fail_rank = fail_rank
        self.pack_current_devices: list[int] = []

    def pack(
        self,
        inputs: MolecularPackingInput,
        *,
        num_samples: int,
        rng: torch.Generator | None,
        context: PackingContext,
        **options: Any,
    ) -> PackingResult:
        self.pack_current_devices.append(torch.cuda.current_device())
        if context.rank == self.fail_rank:
            raise ValueError("designed native driver packing failure")
        return super().pack(
            inputs,
            num_samples=num_samples,
            rng=rng,
            context=context,
            **options,
        )


def _nccl_native_batch_worker(
    rank: int, world_size: int, port: int, queue: Any
) -> None:
    """Check CUDA-device scoping, native Batch gather, and failure propagation."""
    torch.cuda.set_device(rank)
    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=45),
    )
    original_calls = {
        name: getattr(dist, name)
        for name in ("all_gather_object", "broadcast", "isend", "irecv")
    }
    communication_devices: list[int] = []
    for name, original in original_calls.items():

        def record_device(*args: Any, _original: Any = original, **kwargs: Any):
            communication_devices.append(torch.cuda.current_device())
            return _original(*args, **kwargs)

        setattr(dist, name, record_device)

    try:
        packer = _CudaBatchPacker(torch.device("cuda", rank))
        callbacks: list[PackingResult] = []
        destination_generator = CSPGenerator(
            packer,
            process_group=dist.group.WORLD,
            gather_to_rank=1,
            expand=True,
            on_result=callbacks.append,
            dedicated_stream=False,
        )
        caller_device = 1 - rank
        torch.cuda.set_device(caller_device)
        gathered = destination_generator.sample(
            _formula(),
            num_samples=2,
            rank_targets=(1, 1),
            run_id=901,
        )
        first_restored_device = torch.cuda.current_device()
        if rank == 0:
            assert isinstance(gathered, Batch)
            assert gathered.num_graphs == 0
            assert callbacks == []
            gathered_info = _batch_summary(gathered)
        else:
            assert isinstance(gathered, Batch)
            assert len(callbacks) == 1
            assert gathered is callbacks[0].structures
            gathered_info = _batch_summary(gathered)
            gathered_info["reports"] = [
                (report.rank, report.requested_count, report.accepted_count)
                for report in callbacks[0].reports
            ]

        quota_generator = CSPGenerator(
            _CudaBatchPacker(torch.device("cuda", rank)),
            process_group=dist.group.WORLD,
            gather_to_rank=1,
            expand=True,
            dedicated_stream=False,
        )
        zero_quota = quota_generator.sample(
            _formula(),
            num_samples=2,
            rank_targets=(2, 0),
            run_id=902,
        )
        second_restored_device = torch.cuda.current_device()
        assert isinstance(zero_quota, Batch)
        if rank == 1:
            assert zero_quota.num_graphs == 2
            assert zero_quota["csp_source_structure_id"].tolist() == [
                [902, 0],
                [902, 2],
            ]
        else:
            assert zero_quota.num_graphs == 0

        failing = CSPGenerator(
            _CudaBatchPacker(torch.device("cuda", rank), fail_rank=1),
            process_group=dist.group.WORLD,
            gather_to_rank=1,
            expand=False,
            dedicated_stream=False,
        )
        try:
            failing.sample(
                _formula(),
                num_samples=2,
                rank_targets=(1, 1),
                run_id=903,
            )
        except RuntimeError as error:
            failure_message = str(error)
        else:
            failure_message = "missing expected distributed packing failure"
        third_restored_device = torch.cuda.current_device()
        queue.put(
            {
                "rank": rank,
                "caller_device": caller_device,
                "restored_devices": [
                    first_restored_device,
                    second_restored_device,
                    third_restored_device,
                ],
                "pack_devices": packer.pack_current_devices,
                "communication_devices": communication_devices,
                "gathered": gathered_info,
                "failure": failure_message,
            }
        )
    finally:
        for name, original in original_calls.items():
            setattr(dist, name, original)
        dist.destroy_process_group()


def test_gloo_structural_rigid_packer_preserves_ragged_properties_and_reports() -> None:
    records = _run_gloo(_rigid_gloo_worker)
    assert [record["local_ids"] for record in records] == [
        [[701, 0]],
        [[701, 1]],
    ]
    assert all(record["local_generated"] is None for record in records)
    assert [record["local_reason"] for record in records] == [
        "target_reached",
        "custom_shortfall",
    ]
    assert all(
        record["local_options"] == [("option_marker",)] * 2 for record in records
    )
    assert all(
        record["driver_counts"] == {"all_gather_object": 4, "broadcast": 2}
        for record in records
    )

    assert records[0]["gathered"] is None
    gathered = records[1]["gathered"]
    assert gathered["ids"] == [[702, 0], [702, 1]]
    assert gathered["z_prime"] == [2, 1]
    assert gathered["ptr"] == [0, 2, 3]
    assert gathered["quality"] == [[0.0, 0.25], [10.0, 0.25]]
    assert gathered["quality_dtype"] == "torch.float64"
    assert gathered["codes"] == [[0, 0, 7], [1, 0, 7]]
    assert gathered["code_dtype"] == "torch.int32"
    assert gathered["reports"] == [
        (0, 1, 1, None, "target_reached"),
        (1, 2, 1, None, "custom_shortfall"),
    ]


def test_gloo_native_batch_schema_and_zero_quota_rank_round_trip() -> None:
    records = _run_gloo(_batch_gloo_worker)
    assert records[0]["local"]["ids"] == [[801, 0]]
    assert records[1]["local"]["ids"] == [[801, 1]]
    assert all(record["local"]["edge_ptr"] == [0, 1] for record in records)
    assert all(record["local"]["charge"] == [[0.0]] for record in records)
    assert [record["local"]["molecule_ptr"] for record in records] == [
        [0, 1],
        [0, 2],
    ]
    assert [record["local"]["metadata"] for record in records] == [[[0.0]], [[10.0]]]

    empty_nonowner = records[0]["gathered"]
    assert empty_nonowner["graphs"] == 0
    assert empty_nonowner["ids"] == []
    assert empty_nonowner["level_keys"]["metadata"] == ["metadata_values"]
    assert set(empty_nonowner["level_keys"]) == {
        "atoms",
        "edges",
        "system",
        "metadata",
        "molecules",
        "atom_molecules",
    }
    assert empty_nonowner["level_keys"] == records[1]["gathered"]["level_keys"]
    assert empty_nonowner["empty_shapes"] == {
        "metadata": [0, 1],
        "molecules": [0, 1],
        "products": [0, 1],
    }
    assert empty_nonowner["builtin_shapes"] == {
        "positions": [0, 3],
        "atomic_numbers": [0],
        "neighbor_list": [0, 2],
        "cell": [0, 3, 3],
        "pbc": [0, 3],
        "ids": [0, 2],
        "charge": [0, 1],
    }

    gathered = records[1]["gathered"]
    assert gathered["ids"] == [[802, 0], [802, 1]]
    assert gathered["charge"] == [[0.0], [0.0]]
    assert gathered["builtin_shapes"]["charge"] == [2, 1]
    assert gathered["metadata"] == [[0.0], [10.0]]
    assert gathered["atomic_numbers"] == [6, 1, 6, 1]
    assert gathered["cell"] == [
        [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]],
        [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]],
    ]
    assert gathered["pbc"] == [[True, True, True], [True, True, True]]
    assert gathered["edge_ptr"] == [0, 1, 2]
    assert gathered["molecule_ptr"] == [0, 1, 3]
    assert gathered["product_ptr"] == [0, 2, 6]
    assert gathered["reports"] == [(0, 1, 1), (1, 1, 1)]

    assert records[0]["zero_quota"]["graphs"] == 0
    assert records[0]["zero_quota"]["empty_shapes"] == {
        "metadata": [0, 1],
        "molecules": [0, 1],
        "products": [0, 1],
    }
    assert (
        records[0]["zero_quota"]["level_keys"] == records[1]["gathered"]["level_keys"]
    )
    assert records[0]["zero_quota"]["builtin_shapes"] == {
        "positions": [0, 3],
        "atomic_numbers": [0],
        "neighbor_list": [0, 2],
        "cell": [0, 3, 3],
        "pbc": [0, 3],
        "ids": [0, 2],
        "charge": [0, 1],
    }
    zero_quota_gathered = records[1]["zero_quota"]
    assert zero_quota_gathered["ids"] == [[803, 0], [803, 2]]
    assert zero_quota_gathered["metadata"] == [[0.0], [1.0]]
    assert records[0]["empty_sender"]["graphs"] == 0
    assert records[0]["empty_sender"]["empty_shapes"] == {
        "metadata": [0, 1],
        "molecules": [0, 1],
        "products": [0, 1],
    }
    assert (
        records[0]["empty_sender"]["level_keys"]
        == records[1]["empty_sender"]["level_keys"]
    )
    assert records[1]["empty_sender"]["ids"] == [[804, 1], [804, 3]]
    assert records[1]["empty_sender"]["metadata"] == [[10.0], [11.0]]
    assert all(
        options == ("batch_marker",)
        for record in records
        for options in record["options"]
    )

    assert [record["charge_free_local"]["ids"] for record in records] == [
        [[805, 0]],
        [[805, 1]],
    ]
    assert all(record["charge_free_local"]["charge"] is None for record in records)
    assert all(
        "charge" not in record["charge_free_local"]["level_keys"]["system"]
        for record in records
    )

    charge_free_nonowner = records[0]["charge_free_gathered"]
    charge_free_owner = records[1]["charge_free_gathered"]
    assert charge_free_nonowner["graphs"] == 0
    assert charge_free_nonowner["charge"] is None
    assert charge_free_owner["ids"] == [[806, 0], [806, 1]]
    assert charge_free_owner["charge"] is None
    assert "charge" not in charge_free_owner["builtin_shapes"]
    assert charge_free_nonowner["level_keys"] == charge_free_owner["level_keys"]
    assert "charge" not in charge_free_owner["level_keys"]["system"]

    quota_empty = records[0]["charge_free_zero_quota"]
    quota_owner = records[1]["charge_free_zero_quota"]
    assert quota_empty["graphs"] == 0
    assert quota_owner["ids"] == [[807, 0], [807, 2]]
    assert quota_empty["charge"] is None
    assert quota_owner["charge"] is None
    assert quota_empty["level_keys"] == quota_owner["level_keys"]
    assert "charge" not in quota_owner["level_keys"]["system"]


@pytest.mark.parametrize("mode", ["kind", "schema", "charge"])
def test_gloo_native_metadata_disagreement_fails_before_payload_transfer(
    mode: str,
) -> None:
    records = _run_gloo(_metadata_mismatch_worker, mode)
    assert all("phase=pre_payload" in record["message"] for record in records)
    assert all(
        "native payload kind or schema" in record["message"] for record in records
    )


@pytest.mark.multigpu
@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    reason="requires two CUDA devices for NCCL",
)
def test_nccl_native_batch_scopes_packer_device_and_propagates_local_failure() -> None:
    context = mp.get_context("spawn")
    queue = context.Queue()
    mp.spawn(
        _nccl_native_batch_worker,
        args=(2, _available_tcp_port(), queue),
        nprocs=2,
        join=True,
    )
    records = sorted(
        [queue.get(timeout=15) for _ in range(2)], key=lambda record: record["rank"]
    )
    queue.close()

    for record in records:
        rank = record["rank"]
        assert record["caller_device"] == 1 - rank
        assert record["restored_devices"] == [1 - rank] * 3
        assert record["pack_devices"] == [rank]
        assert record["communication_devices"]
        assert set(record["communication_devices"]) == {rank}
        assert "distributed local_packing failed at rank=1" in record["failure"]
        assert "designed native driver packing failure" in record["failure"]

    assert records[0]["gathered"]["graphs"] == 0
    gathered = records[1]["gathered"]
    assert gathered["ids"] == [[901, 0], [901, 1]]
    assert gathered["metadata"] == [[0.0], [10.0]]
    assert gathered["reports"] == [(0, 1, 1), (1, 1, 1)]
