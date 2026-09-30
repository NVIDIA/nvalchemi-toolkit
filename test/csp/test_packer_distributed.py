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
"""Gloo coverage for group-coordinated CrystalPacker calls."""

from __future__ import annotations

import os
from datetime import timedelta
from typing import Any

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from nvalchemi.csp.data import MolecularPackingInput  # noqa: E402
from nvalchemi.csp.packer import (  # noqa: E402
    CrystalPacker,
    PackingConfig,
    PackingStopReason,
)
from test.distributed._dd_harness import free_port  # noqa: E402
from test.distributed._gloo_harness import run_gloo  # noqa: E402


def _packing_input(
    *, marker: str = "same", contact_distance: float = 1.0
) -> MolecularPackingInput:
    return MolecularPackingInput(
        conformer_positions=torch.zeros((1, 3), dtype=torch.float32),
        conformer_ptr=torch.tensor([0, 1], dtype=torch.int32),
        molecule_conformer_ptr=torch.tensor([0, 1], dtype=torch.int32),
        molecule_atom_ptr=torch.tensor([0, 1], dtype=torch.int32),
        atomic_numbers=torch.tensor([6], dtype=torch.int64),
        contact_distances=torch.tensor([[contact_distance]], dtype=torch.float32),
        component_index=torch.tensor([0], dtype=torch.int32),
        formula_unit_volume=1000.0,
        metadata={"marker": marker, "nested": {"value": 1}},
    )


def _config(rank: int, *, max_candidates: int | None = None) -> PackingConfig:
    z = 1 if rank == 0 else 2
    z_prime = 1 if rank == 0 else 2
    return PackingConfig(
        z=z,
        z_prime=z_prime,
        batch_size=2,
        max_candidates=max_candidates,
        cell_volume_range=(1000.0, 1100.0),
        fixed_space_group=1,
        max_steps_per_candidate=4,
    )


def _rank_specific_config(rank: int) -> PackingConfig:
    return PackingConfig(
        z=1 if rank == 0 else 2,
        z_prime=1,
        batch_size=2 if rank == 0 else 1,
        max_candidates=None,
        cell_volume_range=(1000.0, 1050.0) if rank == 0 else (1050.0, 1100.0),
        fixed_space_group=1 if rank == 0 else 2,
        max_steps_per_candidate=4,
    )


def _grouped_pack_worker(rank: int, world_size: int, queue: Any, mode: str) -> None:
    packer = CrystalPacker(_config(rank), device="cpu")
    kwargs: dict[str, Any] = {
        "process_group": dist.group.WORLD,
        "run_id": 515,
        "rng": torch.Generator().manual_seed(8),
    }
    if mode == "mixed":
        local = packer(_packing_input(), num_samples=2, **kwargs)
        gathered_kwargs = kwargs | {"rng": torch.Generator().manual_seed(8)}
        result = packer(
            _packing_input(), num_samples=2, gather_to_rank=1, **gathered_kwargs
        )

        def payload(value: Any) -> Any:
            if value is None:
                return None
            compact = value.structures
            return {
                "generated": value.generated_count,
                "stop_reason": value.stop_reason.value,
                "z": compact.z.tolist(),
                "z_prime": compact.z_prime.tolist(),
                "ptr": compact.structure_molecule_ptr.tolist(),
                "ids": compact.structure_ids.tolist(),
                "metadata": compact.packing_input.metadata["nested"]["value"],
                "conformers": compact.conformer_indices.tolist(),
                "rotations": compact.rotations.tolist(),
                "centers": compact.fractional_centers.tolist(),
                "cells": compact.cells.tolist(),
                "steps": compact.properties["steps"].tolist(),
                "total_overlap": compact.properties["total_overlap"].tolist(),
                "max_overlap": compact.properties["max_overlap"].tolist(),
            }

        queue.put(
            (
                rank,
                payload(local),
                payload(result),
                (
                    int(result.structures.to_batch().atomic_numbers.numel())
                    if result is not None
                    else None
                ),
                (
                    result.structures.to_batch(
                        indices=torch.tensor([1], dtype=torch.int32)
                    ).csp_source_structure_id.tolist()
                    if result is not None
                    else None
                ),
            )
        )
    elif mode == "unequal_gather":
        ranked = CrystalPacker(_rank_specific_config(rank), device="cpu")
        result = ranked(
            _packing_input(),
            num_samples=3,
            rank_targets=(2, 1),
            gather_to_rank=1,
            process_group=dist.group.WORLD,
            run_id=524,
            rng=torch.Generator().manual_seed(23),
        )
        queue.put(
            (
                rank,
                None
                if result is None
                else (
                    result.generated_count,
                    result.stop_reason.value,
                    result.structures.z.tolist(),
                    result.structures.z_prime.tolist(),
                    result.structures.space_groups.tolist(),
                    result.structures.structure_ids.tolist(),
                    result.structures.structure_molecule_ptr.tolist(),
                    result.structures.cells.tolist(),
                ),
            )
        )
    elif mode == "empty_rank":
        result = packer(
            _packing_input(),
            num_samples=1,
            rank_targets=[0, 1],
            **kwargs,
        )
        queue.put(
            (
                rank,
                len(result) if result is not None else None,
                result.generated_count if result is not None else None,
                result.stop_reason.value if result is not None else None,
            )
        )
    elif mode == "local":
        kwargs["run_id"] = None
        result = packer(_packing_input(), num_samples=2, **kwargs)
        queue.put(
            (
                rank,
                len(result) if result is not None else None,
                result.structures.structure_ids.tolist()
                if result is not None
                else None,
                result.structures.z_prime.tolist() if result is not None else None,
                result.run_id if result is not None else None,
            )
        )
    elif mode == "root_id_with_peer_none":
        local_kwargs = kwargs | {"run_id": 515 if rank == 0 else None}
        result = packer(_packing_input(), num_samples=2, **local_kwargs)
        queue.put((rank, result.run_id if result is not None else None))
    elif mode == "callback_failure":

        def fail_progress(progress: Any) -> None:
            if rank == 0:
                raise ValueError("designed local callback failure")

        try:
            packer(
                _packing_input(),
                num_samples=2,
                progress_callback=fail_progress,
                **kwargs,
            )
        except Exception as error:
            queue.put((rank, type(error).__name__, str(error)))
        else:
            queue.put((rank, "no failure", ""))
    elif mode == "budget":
        budgeted = CrystalPacker(_config(rank, max_candidates=2), device="cpu")
        try:
            budgeted(
                _packing_input(),
                num_samples=2,
                rank_targets=[0, 2],
                **kwargs,
            )
        except ValueError as error:
            queue.put((rank, str(error)))
        else:
            queue.put((rank, "missing expected validation error"))
    elif mode == "formula_mismatch":
        try:
            packer(
                _packing_input(marker="rank-zero" if rank == 0 else "rank-one"),
                num_samples=2,
                **kwargs,
            )
        except ValueError as error:
            queue.put((rank, str(error)))
        else:
            queue.put((rank, "missing expected validation error"))
    elif mode == "budgeted":
        budgeted = CrystalPacker(_config(rank, max_candidates=2), device="cpu")
        result = budgeted(
            _packing_input(),
            num_samples=2,
            rank_targets=[0, 2],
            rank_candidate_budgets=[0, 2],
            gather_to_rank=1,
            **kwargs,
        )
        queue.put(
            (
                rank,
                None
                if result is None
                else (len(result), result.generated_count, result.stop_reason.value),
            )
        )
    elif mode == "budget_shortfall":
        budgeted = CrystalPacker(_config(rank, max_candidates=2), device="cpu")
        result = budgeted(
            _packing_input(),
            num_samples=4,
            rank_targets=(2, 2),
            rank_candidate_budgets=(1, 1),
            gather_to_rank=1,
            **kwargs,
        )
        queue.put(
            (
                rank,
                None
                if result is None
                else (
                    len(result),
                    result.generated_count,
                    result.stop_reason.value,
                ),
            )
        )
    elif mode == "default_budget_bounds":
        budgeted = CrystalPacker(_config(rank, max_candidates=3), device="cpu")
        local = budgeted(
            _packing_input(),
            num_samples=4,
            rng=torch.Generator().manual_seed(28),
            run_id=525,
            process_group=dist.group.WORLD,
        )
        gathered = budgeted(
            _packing_input(),
            num_samples=4,
            rng=torch.Generator().manual_seed(28),
            run_id=526,
            gather_to_rank=1,
            process_group=dist.group.WORLD,
        )
        queue.put(
            (
                rank,
                local.generated_count,
                len(local),
                None
                if gathered is None
                else (
                    len(gathered),
                    gathered.generated_count,
                    gathered.stop_reason.value,
                ),
            )
        )
    elif mode == "default_auto_budget_global":
        config = PackingConfig(
            z=2,
            z_prime=2,
            batch_size=64,
            max_steps_per_candidate=1,
            convergence_check_interval=1,
            overlap_tolerance=0.0,
            step_scale=0.0,
            cell_step_scale=0.0,
            volume_compression_scale=0.0,
            cell_volume_range=(125.0, 125.0),
            fixed_space_group=1,
        )
        budgeted = CrystalPacker(config, device="cpu")
        result = budgeted(
            _packing_input(contact_distance=7.0),
            num_samples=2,
            rank_targets=(1, 1),
            rank_candidate_budgets=(1000, 1000),
            gather_to_rank=1,
            process_group=dist.group.WORLD,
            run_id=527,
            rng=torch.Generator().manual_seed(28),
        )
        queue.put(
            (
                rank,
                None
                if result is None
                else (len(result), result.generated_count, result.stop_reason.value),
            )
        )
    elif mode == "preflight_mismatch":
        cases = (
            {"rank_targets": [1, 1] if rank == 0 else [0, 2]},
            {"run_id": 518 + rank},
            {"gather_to_rank": rank},
        )
        for overrides in cases:
            try:
                packer(_packing_input(), num_samples=2, **(kwargs | overrides))
            except ValueError as error:
                queue.put((rank, str(error)))
            else:
                queue.put((rank, "missing expected validation error"))
    elif mode == "validation_failure":
        bad_inputs: Any = _packing_input() if rank == 1 else "invalid formula input"
        try:
            packer(bad_inputs, num_samples=2, **kwargs)
        except ValueError as error:
            queue.put((rank, str(error)))
        else:
            queue.put((rank, "missing expected validation error"))
    elif mode == "single_rank_seed":
        config = PackingConfig(
            z=1,
            z_prime=1,
            batch_size=2,
            cell_volume_range=(1000.0, 1100.0),
            fixed_space_group=1,
        )
        single = CrystalPacker(config, device="cpu")
        inputs = _packing_input()
        direct = single(
            inputs,
            num_samples=1,
            run_id=521,
            rng=torch.Generator().manual_seed(18),
        )
        grouped = single(
            inputs,
            num_samples=1,
            process_group=dist.group.WORLD,
            run_id=521,
            rng=torch.Generator().manual_seed(18),
        )
        queue.put(
            (
                rank,
                torch.equal(direct.structures.cells, grouped.structures.cells),
                torch.equal(
                    direct.structures.fractional_centers,
                    grouped.structures.fractional_centers,
                ),
                torch.equal(
                    direct.structures.structure_ids, grouped.structures.structure_ids
                ),
            )
        )


def _subgroup_worker(rank: int, world_size: int, queue: Any) -> None:
    first = dist.new_group([0, 1], backend="gloo")
    second = dist.new_group([2, 3], backend="gloo")
    group = first if rank < 2 else second
    group_rank = dist.get_rank(group=group)
    packer = CrystalPacker(_config(group_rank), device="cpu")
    result = packer(
        _packing_input(),
        num_samples=2,
        process_group=group,
        gather_to_rank=1,
        run_id=517,
        rng=torch.Generator().manual_seed(12),
    )
    queue.put(
        (rank, None if result is None else result.structures.structure_ids.tolist())
    )


def _failure_worker(rank: int, world_size: int, queue: Any) -> None:
    from nvalchemi.csp.packer import engine

    if rank == 0:
        original = engine.contact_forces

        def fail_once(**kwargs: Any) -> Any:
            raise ValueError("designed local contact failure")

        engine.contact_forces = fail_once
    packer = CrystalPacker(_config(rank), device="cpu")
    try:
        packer(
            _packing_input(),
            num_samples=2,
            process_group=dist.group.WORLD,
            run_id=516,
            rng=torch.Generator().manual_seed(9),
        )
    except Exception as error:
        queue.put((rank, type(error).__name__, str(error)))
    else:
        queue.put((rank, "no failure", ""))
    finally:
        if rank == 0:
            engine.contact_forces = original


def _setup_failure_worker(rank: int, world_size: int, queue: Any) -> None:
    from nvalchemi.csp.packer import engine

    if rank == 0:
        original = engine.sample_valid_cells

        def fail_sampling(**kwargs: Any) -> Any:
            raise ValueError("designed setup failure")

        engine.sample_valid_cells = fail_sampling
    packer = CrystalPacker(_config(rank), device="cpu")
    try:
        packer(
            _packing_input(),
            num_samples=2,
            process_group=dist.group.WORLD,
            run_id=523,
            rng=torch.Generator().manual_seed(19),
        )
    except Exception as error:
        queue.put((rank, type(error).__name__, str(error)))
    else:
        queue.put((rank, "no failure", ""))
    finally:
        if rank == 0:
            engine.sample_valid_cells = original


def _destination_assembly_failure_worker(
    rank: int, world_size: int, queue: Any
) -> None:
    from nvalchemi.csp.packer import engine

    original_result = engine.PackingResult
    if rank == 1:

        def fail_gathered_result(**kwargs: Any) -> Any:
            if kwargs["structures"].num_structures == 2:
                raise ValueError("designed destination assembly failure")
            return original_result(**kwargs)

        engine.PackingResult = fail_gathered_result
    try:
        packer = CrystalPacker(_config(rank), device="cpu")
        packer(
            _packing_input(),
            num_samples=2,
            process_group=dist.group.WORLD,
            gather_to_rank=1,
            run_id=519,
            rng=torch.Generator().manual_seed(14),
        )
    except Exception as error:
        queue.put((rank, type(error).__name__, str(error)))
    else:
        queue.put((rank, "no failure", ""))
    finally:
        if rank == 1:
            engine.PackingResult = original_result


def _nccl_worker(rank: int, world_size: int, port: int, queue: Any) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl",
        rank=rank,
        world_size=world_size,
        init_method=f"tcp://127.0.0.1:{port}",
        timeout=timedelta(seconds=60),
    )
    try:
        packer = CrystalPacker(_config(rank), device=f"cuda:{rank}")
        local = packer(
            _packing_input(),
            num_samples=2,
            process_group=dist.group.WORLD,
            run_id=520,
            rng=torch.Generator(device=f"cuda:{rank}").manual_seed(15),
        )
        gathered = packer(
            _packing_input(),
            num_samples=2,
            process_group=dist.group.WORLD,
            gather_to_rank=1,
            run_id=521,
            rng=torch.Generator(device=f"cuda:{rank}").manual_seed(16),
        )
        assert local.structures.cells.device == torch.device(f"cuda:{rank}")
        if rank == 1:
            assert gathered is not None
            assert gathered.structures.cells.device == torch.device("cuda:1")
        else:
            assert gathered is None
        local_payload = (
            len(local),
            local.structures.structure_ids.cpu().tolist(),
            local.structures.z_prime.cpu().tolist(),
        )
        gathered_payload = (
            None
            if gathered is None
            else (
                len(gathered),
                gathered.structures.structure_ids.cpu().tolist(),
                gathered.structures.z_prime.cpu().tolist(),
            )
        )
        stream_config = PackingConfig(
            z=1,
            z_prime=1,
            batch_size=1,
            cell_volume_range=(1000.0, 1100.0),
            fixed_space_group=1,
            max_steps_per_candidate=4,
        )
        stream_result = CrystalPacker(stream_config, device=f"cuda:{rank}")(
            _packing_input(),
            num_samples=1,
            process_group=dist.group.WORLD,
            run_id=523,
            rng=torch.Generator(device=f"cuda:{rank}").manual_seed(99),
        )
        stream_geometry = (
            stream_result.structures.cells.cpu().tolist(),
            stream_result.structures.fractional_centers.cpu().tolist(),
        )
        queue.put(("payload", rank, local_payload, gathered_payload, stream_geometry))
        from nvalchemi.csp.packer import engine

        if rank == 0:
            original_contact_forces = engine.contact_forces

            def fail_contact(**kwargs: Any) -> Any:
                raise ValueError("designed NCCL local failure")

            engine.contact_forces = fail_contact
        try:
            packer(
                _packing_input(),
                num_samples=2,
                process_group=dist.group.WORLD,
                run_id=522,
                rng=torch.Generator(device=f"cuda:{rank}").manual_seed(17),
            )
        except Exception as error:
            queue.put(("failure", rank, type(error).__name__, str(error)))
        else:
            queue.put(("failure", rank, "no failure", ""))
        finally:
            if rank == 0:
                engine.contact_forces = original_contact_forces
    finally:
        dist.destroy_process_group()


def test_gloo_gather_preserves_rank_major_mixed_z_prime_and_identity() -> None:
    results = run_gloo(world_size=2, fn=_grouped_pack_worker, args=("mixed",))
    by_rank = {
        rank: (local, gathered, atom_count, selected_source_ids)
        for rank, local, gathered, atom_count, selected_source_ids in results
    }
    local_payloads = [by_rank[rank][0] for rank in range(2)]
    gathered = by_rank[1][1]
    assert by_rank[0][1] is None
    assert gathered["generated"] == 4
    assert gathered["stop_reason"] == PackingStopReason.TARGET_REACHED.value
    assert gathered["z"] == [1, 2]
    assert gathered["z_prime"] == [1, 2]
    assert gathered["ptr"] == [0, 1, 3]
    assert gathered["ids"] == [[515, 0], [515, 1]]
    assert gathered["metadata"] == 1
    assert gathered["conformers"] == [0, 0, 0]
    assert gathered["steps"] == [
        row for payload in local_payloads for row in payload["steps"]
    ]
    for field in (
        "z",
        "z_prime",
        "ids",
        "cells",
        "conformers",
        "rotations",
        "centers",
        "steps",
        "total_overlap",
        "max_overlap",
    ):
        assert gathered[field] == [
            row for payload in local_payloads for row in payload[field]
        ]
    assert by_rank[1][2] == 3
    assert by_rank[1][3] == [[515, 1]]


def test_gloo_gather_supports_rank_specific_science_and_unequal_quotas() -> None:
    results = dict(
        run_gloo(world_size=2, fn=_grouped_pack_worker, args=("unequal_gather",))
    )
    assert results[0] is None
    generated, reason, z, z_prime, groups, ids, ptr, cells = results[1]
    assert generated == 3
    assert reason == PackingStopReason.TARGET_REACHED.value
    assert z == [1, 1, 2]
    assert z_prime == [1, 1, 1]
    assert groups == [1, 1, 2]
    assert ids == [[524, 0], [524, 2], [524, 1]]
    assert ptr == [0, 1, 2, 3]
    volumes = [torch.det(torch.tensor(cell)).item() for cell in cells]
    assert all(1000.0 <= value <= 1050.0 for value in volumes[:2])
    assert 1050.0 <= volumes[2] <= 1100.0


def test_gloo_empty_local_quota_and_target_partition() -> None:
    results = {
        rank: (count, generated, reason)
        for rank, count, generated, reason in run_gloo(
            world_size=2, fn=_grouped_pack_worker, args=("empty_rank",)
        )
    }
    assert results == {
        0: (0, 0, PackingStopReason.TARGET_REACHED.value),
        1: (1, 2, PackingStopReason.TARGET_REACHED.value),
    }


def test_gloo_process_group_without_gather_returns_local_results() -> None:
    results = run_gloo(world_size=2, fn=_grouped_pack_worker, args=("local",))
    by_rank = {
        rank: (count, ids, z_prime, run_id)
        for rank, count, ids, z_prime, run_id in results
    }
    assert by_rank[0][0] == by_rank[1][0] == 1
    assert by_rank[0][1] == [[by_rank[0][3], 0]]
    assert by_rank[1][1] == [[by_rank[0][3], 1]]
    assert by_rank[0][2] == [1]
    assert by_rank[1][2] == [2]
    assert by_rank[0][3] == by_rank[1][3]
    assert 0 <= by_rank[0][3] < 2**63


def test_gloo_root_run_id_allows_unspecified_peer_id() -> None:
    assert dict(
        run_gloo(
            world_size=2,
            fn=_grouped_pack_worker,
            args=("root_id_with_peer_none",),
        )
    ) == {0: 515, 1: 515}


def test_gloo_finite_global_budget_requires_explicit_custom_budgets() -> None:
    results = dict(run_gloo(world_size=2, fn=_grouped_pack_worker, args=("budget",)))
    assert all(
        "require explicit rank_candidate_budgets" in value for value in results.values()
    )


def test_gloo_preflight_rejects_formula_metadata_mismatch() -> None:
    results = dict(
        run_gloo(world_size=2, fn=_grouped_pack_worker, args=("formula_mismatch",))
    )
    assert all("formula input" in value for value in results.values())


def test_gloo_custom_targets_and_finite_rank_budgets() -> None:
    results = dict(run_gloo(world_size=2, fn=_grouped_pack_worker, args=("budgeted",)))
    assert results == {0: None, 1: (2, 2, PackingStopReason.TARGET_REACHED.value)}


def test_gloo_gather_reports_aggregate_candidate_budget_shortfall() -> None:
    results = dict(
        run_gloo(world_size=2, fn=_grouped_pack_worker, args=("budget_shortfall",))
    )
    assert results[0] is None
    accepted, generated, reason = results[1]
    assert accepted <= generated == 2
    assert accepted < 4
    assert reason == PackingStopReason.CANDIDATE_BUDGET_EXHAUSTED.value


def test_gloo_default_finite_budget_is_bounded_and_gathered_shortfall_is_aggregate() -> (
    None
):
    results = {
        rank: (generated, accepted, gathered)
        for rank, generated, accepted, gathered in run_gloo(
            world_size=2,
            fn=_grouped_pack_worker,
            args=("default_budget_bounds",),
        )
    }
    assert results[0][0] <= 2
    assert results[1][0] <= 1
    assert results[0][0] + results[1][0] <= 3
    assert results[0][2] is None
    assert results[1][2][1] == 3
    assert results[1][2][0] < 4
    assert results[1][2][2] == PackingStopReason.CANDIDATE_BUDGET_EXHAUSTED.value


def test_gloo_default_auto_budget_uses_global_target_before_partitioning() -> None:
    results = dict(
        run_gloo(
            world_size=2, fn=_grouped_pack_worker, args=("default_auto_budget_global",)
        )
    )
    assert results == {
        0: None,
        1: (0, 2000, PackingStopReason.CANDIDATE_BUDGET_EXHAUSTED.value),
    }


def test_gloo_catchable_callback_failure_reaches_every_rank() -> None:
    results = run_gloo(
        world_size=2, fn=_grouped_pack_worker, args=("callback_failure",)
    )
    assert len(results) == 2
    assert all(item[1] in {"ValueError", "RuntimeError"} for item in results)
    assert any("designed local callback failure" in item[2] for item in results)
    assert any("peer failed" in item[2] for item in results)


def test_gloo_subgroup_destination_is_group_local() -> None:
    results = dict(run_gloo(world_size=4, fn=_subgroup_worker))
    assert results == {
        0: None,
        1: [[517, 0], [517, 1]],
        2: None,
        3: [[517, 0], [517, 1]],
    }


def test_gloo_preflight_rejects_shared_contract_mismatches() -> None:
    results = run_gloo(
        world_size=2, fn=_grouped_pack_worker, args=("preflight_mismatch",)
    )
    assert len(results) == 6
    assert all(
        "disagree" in message or "different run_id" in message for _, message in results
    )


def test_gloo_coordinates_early_validation_failure() -> None:
    results = run_gloo(
        world_size=2, fn=_grouped_pack_worker, args=("validation_failure",)
    )
    assert all(
        "distributed packer validation failed" in message for _, message in results
    )


def test_gloo_single_rank_group_preserves_rng_trajectory() -> None:
    results = run_gloo(
        world_size=1, fn=_grouped_pack_worker, args=("single_rank_seed",)
    )
    assert results == [(0, True, True, True)]


def test_gloo_catchable_local_failure_reaches_every_rank() -> None:
    results = run_gloo(world_size=2, fn=_failure_worker)
    assert len(results) == 2
    assert all(item[1] in {"ValueError", "RuntimeError"} for item in results)
    assert any("designed local contact failure" in item[2] for item in results)
    assert any("peer failed" in item[2] for item in results)


def test_gloo_setup_failure_reaches_every_rank() -> None:
    results = run_gloo(world_size=2, fn=_setup_failure_worker)
    assert len(results) == 2
    assert any("designed setup failure" in item[2] for item in results)
    assert any("peer failed" in item[2] for item in results)


def test_gloo_destination_assembly_failure_reaches_every_rank() -> None:
    results = run_gloo(world_size=2, fn=_destination_assembly_failure_worker)
    assert len(results) == 2
    assert any("designed destination assembly failure" in item[2] for item in results)
    assert any("peer failed" in item[2] for item in results)


@pytest.mark.multigpu
@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    reason="requires two CUDA devices for NCCL",
)
def test_nccl_local_gather_and_failure_coordination() -> None:
    context = mp.get_context("spawn")
    queue = context.Queue()
    world_size = 2
    port = free_port()
    mp.spawn(_nccl_worker, args=(world_size, port, queue), nprocs=world_size, join=True)
    records = [queue.get(timeout=10) for _ in range(4)]
    payloads = [record for record in records if record[0] == "payload"]
    failures = [record for record in records if record[0] == "failure"]
    assert len(payloads) == len(failures) == 2
    payloads_by_rank = {record[1]: record for record in payloads}
    assert all(record[2][0] == 1 for record in payloads_by_rank.values())
    assert payloads_by_rank[0][3] is None
    assert payloads_by_rank[1][3][1] == [[521, 0], [521, 1]]
    assert payloads_by_rank[0][4] != payloads_by_rank[1][4]
    assert all(record[2] in {"ValueError", "RuntimeError"} for record in failures)
    assert any("designed NCCL local failure" in record[3] for record in failures)
