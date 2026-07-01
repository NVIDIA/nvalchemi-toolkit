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

"""Tests for the dispatch-trace mechanism.

Single-process tests — they assert the API contract (record/no-record
based on context, record fields, scope nesting) without needing a
distributed process group. Multi-rank Gloo tests that exercise the
actual handlers live in ``test_dispatch_trace_gloo.py``.
"""

from __future__ import annotations

import torch
import torch.distributed as dist

from nvalchemi.distributed._core.dispatch_trace import (
    dispatch_trace,
    is_tracing,
    record_dispatch,
)
from test.distributed._gloo_harness import run_gloo


def _expected_rank() -> int:
    """The rank ``record_dispatch`` auto-tags: this process's rank when a
    process group is initialised, else ``-1``. Computed from the ambient dist
    state so the assertion is robust to a leaked session-scoped PG from an
    earlier test (the auto-tag contract is what's under test, not the value)."""
    return dist.get_rank() if dist.is_initialized() else -1
class TestTracingScope:
    def test_no_trace_outside_context(self):
        assert not is_tracing()

    def test_trace_inside_context(self):
        with dispatch_trace() as records:
            assert is_tracing()
            assert records == []

    def test_trace_off_after_exit(self):
        with dispatch_trace():
            pass
        assert not is_tracing()

    def test_trace_off_on_exception(self):
        try:
            with dispatch_trace():
                raise RuntimeError("boom")
        except RuntimeError:
            pass
        assert not is_tracing()
class TestRecordDispatch:
    def test_no_record_outside_context(self):
        # Should be a no-op — does not raise, does not allocate.
        record_dispatch("phantom", op="phantom_op")

    def test_record_inside_context(self):
        with dispatch_trace() as records:
            record_dispatch("test_handler", op="aten::foo", branch="path_a")
        assert len(records) == 1
        assert records[0]["handler"] == "test_handler"
        assert records[0]["op"] == "aten::foo"
        assert records[0]["branch"] == "path_a"
        # Auto-tagged with this process's rank (or -1 when no PG is up).
        assert records[0]["rank"] == _expected_rank()

    def test_record_carries_arbitrary_fields(self):
        with dispatch_trace() as records:
            record_dispatch(
                "test_handler",
                shapes={"src": (8, 3), "dst": (1, 3)},
                meta={"n_owned": 8, "n_padded": 16},
            )
        record = records[0]
        assert record["shapes"] == {"src": (8, 3), "dst": (1, 3)}
        assert record["meta"]["n_owned"] == 8

    def test_multiple_records_in_order(self):
        with dispatch_trace() as records:
            record_dispatch("h1", branch="a")
            record_dispatch("h2", branch="b")
            record_dispatch("h1", branch="c")
        assert [r["handler"] for r in records] == ["h1", "h2", "h1"]
        assert [r["branch"] for r in records] == ["a", "b", "c"]

    def test_records_are_independent_across_scopes(self):
        with dispatch_trace() as r1:
            record_dispatch("h1")
        with dispatch_trace() as r2:
            record_dispatch("h2")
        assert len(r1) == 1 and len(r2) == 1
        assert r1[0]["handler"] == "h1"
        assert r2[0]["handler"] == "h2"
class TestRecordingShape:
    """The ``record_dispatch`` records carry rich-enough fields that a
    test can assert against branch / shapes / meta — not just the
    handler name."""

    def test_branch_field_present(self):
        with dispatch_trace() as records:
            record_dispatch("h", branch="halo_reverse+halo_forward")
        assert records[0]["branch"] == "halo_reverse+halo_forward"

    def test_shapes_dict_arbitrary_keys(self):
        with dispatch_trace() as records:
            record_dispatch(
                "h",
                shapes={
                    "self": (1,),
                    "index": (80,),
                    "src": (80, 9),
                },
            )
        shapes = records[0]["shapes"]
        assert shapes["self"] == (1,)
        assert shapes["src"] == (80, 9)

    def test_meta_field_carries_int_metadata(self):
        with dispatch_trace() as records:
            record_dispatch("h", meta={"n_owned": 80, "n_padded": 128})
        assert records[0]["meta"] == {"n_owned": 80, "n_padded": 128}

def _per_system_reduce_worker(
    rank: int,
    world_size: int,
    queue,
    *args,
) -> None:
    """Each rank wraps an owned-shape values tensor as a ShardTensor
    with halo metadata, performs an in-place ``index_add_`` into a
    1-system accumulator, captures the dispatch trace + output, and
    sends the (rank, trace, output) tuple back."""
    from types import SimpleNamespace

    from nvalchemi.distributed._core.dispatch_trace import dispatch_trace
    from nvalchemi.distributed._core.shard_tensor import ShardTensor
    from nvalchemi.distributed.spec import MLIPSpec

    # Per-rank owned slice. Total atoms = 8: rank 0 owns 5 (values
    # 1.0..5.0), rank 1 owns 3 (values 6.0..8.0). Global sum = 36.
    if rank == 0:
        owned = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    else:
        owned = torch.tensor([6.0, 7.0, 8.0])

    n_owned = owned.shape[0]
    halo_meta = SimpleNamespace(
        n_owned=n_owned,
        n_padded=n_owned,  # no halo for this minimal test
        gnn_markers=None,
    )
    halo_config = SimpleNamespace(mesh=None)

    from nvalchemi.distributed._core.spec import DistributionSpec
    from nvalchemi.distributed._core.storage_policy import HaloStoragePolicy

    spec = MLIPSpec(
        distribution=DistributionSpec(
            policy=HaloStoragePolicy(
                scatter_mode="halo_correction",
                gather_mode="halo_read",
            )
        )
    )

    src_st = ShardTensor.wrap(
        owned,
        spec=spec,
        meta=halo_meta,
        config=halo_config,
        n_systems=1,
    )
    accumulator = torch.zeros(1)
    index = torch.zeros(n_owned, dtype=torch.long)

    with dispatch_trace() as records:
        accumulator.index_add_(0, index, src_st)

    # accumulator is mutated in-place; on each rank it should hold the
    # globally-reduced sum (1+2+...+8 = 36) after per_system_reduce
    # all-reduces across the mesh.
    queue.put(
        (
            rank,
            [dict(r) for r in records],  # convert to plain dicts
            float(accumulator.item()),
        )
    )
def test_per_system_reduce_two_rank_gloo():
    """``index_add_`` on a (1,)-accumulator with a ShardTensor src
    fires ``_per_system_reduce_handler``; both ranks see the global
    sum after the in-place all-reduce."""
    results = run_gloo(world_size=2, fn=_per_system_reduce_worker)
    # results: list of (rank, records, accumulator_value)
    by_rank = {r[0]: r for r in results}
    assert set(by_rank) == {0, 1}, f"missing ranks: {set(by_rank)}"

    expected_global_sum = sum(range(1, 9))  # 1+2+..+8 == 36

    for rank in (0, 1):
        _rank, records, acc_value = by_rank[rank]
        # Numerical correctness: per_system_reduce should leave
        # the global sum on every rank's accumulator.
        assert acc_value == expected_global_sum, (
            f"rank {rank}: accumulator={acc_value}, expected={expected_global_sum}"
        )
        # Trace assertions: exactly one per_system_reduce fire on
        # this rank, with branch=owned_slice+all_reduce and the
        # right shape contract.
        per_sys = [r for r in records if r["handler"] == "per_system_reduce"]
        assert len(per_sys) == 1, f"rank {rank}: per_system fires={len(per_sys)}"
        assert per_sys[0]["branch"] == "owned_slice+all_reduce"
        assert per_sys[0]["meta"]["n_systems"] == 1
def _no_trace_outside_scope_worker(
    rank: int,
    world_size: int,
    queue,
    *args,
) -> None:
    """Sanity: when no ``dispatch_trace`` scope is open, handler
    firings produce no records (the ``is_tracing()`` short-circuit
    works under multi-rank too)."""
    from types import SimpleNamespace

    from nvalchemi.distributed._core.dispatch_trace import is_tracing
    from nvalchemi.distributed._core.shard_tensor import ShardTensor
    from nvalchemi.distributed.spec import MLIPSpec

    owned = torch.tensor([float(rank + 1)])
    halo_meta = SimpleNamespace(n_owned=1, n_padded=1, gnn_markers=None)
    halo_config = SimpleNamespace(mesh=None)
    from nvalchemi.distributed._core.spec import DistributionSpec
    from nvalchemi.distributed._core.storage_policy import HaloStoragePolicy

    spec = MLIPSpec(
        distribution=DistributionSpec(
            policy=HaloStoragePolicy(
                scatter_mode="halo_correction",
                gather_mode="halo_read",
            )
        )
    )
    src_st = ShardTensor.wrap(
        owned, spec=spec, meta=halo_meta, config=halo_config, n_systems=1
    )
    accumulator = torch.zeros(1)
    index = torch.zeros(1, dtype=torch.long)
    accumulator.index_add_(0, index, src_st)

    queue.put((rank, is_tracing(), float(accumulator.item())))
def test_dispatch_runs_without_trace_active():
    """Negative test: dispatch handlers still produce correct numerics
    when no trace scope is active. Confirms the trace plumbing is
    purely additive — production runs (no trace) get exactly the
    same dispatch behaviour."""
    results = run_gloo(world_size=2, fn=_no_trace_outside_scope_worker)
    by_rank = {r[0]: r for r in results}
    expected = 1.0 + 2.0  # rank 0 contributes 1, rank 1 contributes 2
    for rank in (0, 1):
        _rank, tracing, acc = by_rank[rank]
        assert tracing is False, f"rank {rank}: tracing leaked outside scope"
        assert acc == expected, f"rank {rank}: acc={acc}, expected={expected}"
