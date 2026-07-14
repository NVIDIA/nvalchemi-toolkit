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

"""Unit tests for ``_mark_halo_receiver_edges_as_padding``.

The helper's contract is: rewrite ``neighbor_list`` so halo-receiver
rows look like NL padding sentinels (``padded_batch.num_nodes``), so
the wrapper's existing distribution-agnostic ``valid`` filter drops
them. The helper has a strict no-sync, no-rebuild performance contract.

These tests are unit-level — they don't spawn workers or run the
validator. End-to-end behaviour is covered by the example in
``examples/distributed/04_byo_pytorch_mpnn.py`` and the BPWrapper
case in ``test_validate_cuda.py``.
"""

from __future__ import annotations

import torch

from nvalchemi.data import AtomicData, Batch
from nvalchemi.distributed.distributed_model import (
    _mark_halo_receiver_edges_as_padding,
)
from nvalchemi.models.base import NeighborConfig, NeighborListFormat
from nvalchemi.neighbors import compute_neighbors


def _make_padded_batch(
    n_atoms: int = 16, cutoff: float = 5.0, device: str = "cpu"
) -> Batch:
    """Build a small periodic batch + COO neighbour list. The batch
    plays the role of a "padded_batch" for the helper — the helper
    cares only about ``num_nodes`` and the edges group's
    ``neighbor_list``, both of which work the same single-process or
    halo-padded.
    """
    torch.manual_seed(0)
    spacing = 2.0
    coords = torch.arange(n_atoms, dtype=torch.float32, device=device)
    positions = torch.stack(
        [coords, torch.zeros_like(coords), torch.zeros_like(coords)], dim=-1
    )
    positions = positions * spacing
    atomic_numbers = torch.full((n_atoms,), 1, dtype=torch.long, device=device)
    cell = torch.eye(3, dtype=torch.float32, device=device) * (n_atoms * spacing)
    pbc = torch.tensor([[True, True, True]], device=device)
    data = AtomicData(
        positions=positions,
        atomic_numbers=atomic_numbers,
        cell=cell.unsqueeze(0),
        pbc=pbc,
    )
    batch = Batch.from_data_list([data], device=device)
    compute_neighbors(
        batch,
        config=NeighborConfig(cutoff=cutoff, format=NeighborListFormat.COO),
    )
    return batch


# ----------------------------------------------------------------------
# (1) Single-process no-op: n_owned == n_padded leaves NL bit-identical.
# ----------------------------------------------------------------------


def test_singleproc_noop_preserves_nl_bit_identically() -> None:
    """When ``n_owned == padded_batch.num_nodes`` (the single-process
    case), every row is an owned receiver, the mask is all-False, and
    the helper must not perturb a single byte of the NL."""
    batch = _make_padded_batch()
    nl_before = batch.neighbor_list.clone()

    _mark_halo_receiver_edges_as_padding(batch, n_owned=batch.num_nodes)

    nl_after = batch.neighbor_list
    assert torch.equal(nl_before, nl_after), (
        "Single-process no-op path mutated the NL; the no-op contract "
        "is what guards single-process callers from paying the helper "
        "as overhead."
    )


# ----------------------------------------------------------------------
# (2) Marks halo-receiver rows to ``padded_batch.num_nodes``.
# ----------------------------------------------------------------------


def test_marks_halo_receivers_with_padding_sentinel() -> None:
    """With ``n_owned < num_nodes``, every row whose receiver was a
    halo atom must have its receiver column rewritten to
    ``padded_batch.num_nodes``. Owned-receiver rows must stay
    untouched in *both* columns."""
    batch = _make_padded_batch(n_atoms=16, cutoff=5.0)
    n_padded = batch.num_nodes
    n_owned = 8  # arbitrary split: first half owned, second half halo
    sentinel = n_padded

    nl_before = batch.neighbor_list.clone()
    halo_recv_rows_before = nl_before[:, 1] >= n_owned
    owned_recv_rows_before = ~halo_recv_rows_before

    _mark_halo_receiver_edges_as_padding(batch, n_owned=n_owned)

    nl_after = batch.neighbor_list
    # Halo-receiver rows: column 1 = sentinel.
    assert torch.all(nl_after[halo_recv_rows_before, 1] == sentinel), (
        "halo-receiver rows were not rewritten to the padding sentinel"
    )
    # Owned-receiver rows: NL bit-identical (both columns).
    assert torch.equal(
        nl_after[owned_recv_rows_before], nl_before[owned_recv_rows_before]
    ), "helper perturbed owned-receiver rows it should have left alone"
    # Sender column for halo-receiver rows: unchanged.
    assert torch.equal(
        nl_after[halo_recv_rows_before, 0], nl_before[halo_recv_rows_before, 0]
    ), (
        "helper modified the sender column of halo-receiver rows; the "
        "documented contract is to mark only the receiver"
    )


# ----------------------------------------------------------------------
# (3) Idempotent: re-running yields the same result. Important for
# skin-buffer reuse across MD steps (the helper runs once at NL
# rebuild but the same NL persists across many forwards).
# ----------------------------------------------------------------------


def test_idempotent_under_re_run() -> None:
    batch = _make_padded_batch(n_atoms=16, cutoff=5.0)
    n_owned = 8

    _mark_halo_receiver_edges_as_padding(batch, n_owned=n_owned)
    nl_first = batch.neighbor_list.clone()
    _mark_halo_receiver_edges_as_padding(batch, n_owned=n_owned)
    nl_second = batch.neighbor_list

    assert torch.equal(nl_first, nl_second), (
        "running the helper twice perturbed the NL — breaks the "
        "skin-buffer reuse contract"
    )


# ----------------------------------------------------------------------
# (4) Preserves NL row count and dtype. The helper rewrites in place;
# allocation, dtype change, or shape change would all imply a storage
# rebuild which the contract forbids.
# ----------------------------------------------------------------------


def test_preserves_row_count_dtype_and_storage_object() -> None:
    batch = _make_padded_batch(n_atoms=16, cutoff=5.0)
    n_owned = 8

    nl_before = batch.neighbor_list
    rows_before = nl_before.shape[0]
    dtype_before = nl_before.dtype
    edges_group_id_before = id(batch._edges_group)
    nl_id_before = id(nl_before)

    _mark_halo_receiver_edges_as_padding(batch, n_owned=n_owned)

    nl_after = batch.neighbor_list
    assert nl_after.shape[0] == rows_before, "row count changed"
    assert nl_after.dtype == dtype_before, "dtype changed"
    assert id(batch._edges_group) == edges_group_id_before, (
        "edges-group object was replaced — should be in-place mutation"
    )
    assert id(nl_after) == nl_id_before, (
        "neighbor_list tensor object was replaced — should be in-place"
    )


# ----------------------------------------------------------------------
# (5) Hot-path performance contract: helper emits no GPU sync. Uses
# CUDA event timing — if the helper inserted a sync, the elapsed time
# between recording the event and reading it would include the
# helper's full GPU work, which is detectable. The actual assertion
# we can make safely is "the call returns before the GPU finishes
# work". On CUDA we can also check via stream queries.
# ----------------------------------------------------------------------


def test_hot_path_no_sync_inducing_calls_in_source() -> None:
    """Static guard: the helper's source must not contain any of the
    well-known sync-inducing patterns. Wall-clock timing tests are
    flaky across host/GPU configurations (memory pressure and CUDA
    launch-queue contention can extend a sync-free helper's CPU time
    well past microseconds), so we enforce the invariant by code
    inspection instead.

    The helper's correctness when no sync-inducing API is used is
    structural: tensor compares, in-place ``masked_fill_``, and
    Python int access from ``shape``/``num_nodes`` are all sync-free
    primitives. If a future edit introduces ``.item()``, ``.nonzero()``,
    boolean-mask-indexing-that-allocates, or ``synchronize``, it
    breaks the helper's no-sync, no-extra-allocation hot-path
    contract — this test is the regression guard.
    """
    import inspect

    from nvalchemi.distributed import distributed_model as dm

    src = inspect.getsource(dm._mark_halo_receiver_edges_as_padding)
    forbidden = [
        ".item(",
        ".nonzero(",
        ".tolist(",
        ".cpu(",
        "synchronize",
        "torch.where",  # not a sync, but allocates a new tensor — prefer masked_fill_
    ]
    for needle in forbidden:
        assert needle not in src, (
            f"_mark_halo_receiver_edges_as_padding source contains "
            f"{needle!r}, which violates the hot-path no-sync / "
            f"no-extra-allocation contract for the owned-receiver filter."
        )


# ----------------------------------------------------------------------
# (6) Defensive no-op cases: empty NL, missing edges group.
# ----------------------------------------------------------------------


def test_no_edges_group_is_noop() -> None:
    """UMA-style wrappers either don't have an edges group at the
    point the helper runs, or have an empty NL the wrapper rebuilds
    inside ``forward``. Either way the helper must early-return
    without raising — this is what keeps it safe to call
    unconditionally for every halo-storage forward."""
    torch.manual_seed(0)
    n_atoms = 8
    positions = torch.arange(n_atoms, dtype=torch.float32).unsqueeze(-1)
    positions = positions.expand(n_atoms, 3) * 2.0
    atomic_numbers = torch.full((n_atoms,), 1, dtype=torch.long)
    cell = torch.eye(3, dtype=torch.float32) * (n_atoms * 2.0)
    pbc = torch.tensor([[True, True, True]])
    data = AtomicData(
        positions=positions.contiguous(),
        atomic_numbers=atomic_numbers,
        cell=cell.unsqueeze(0),
        pbc=pbc,
    )
    batch = Batch.from_data_list([data])
    # No ``compute_neighbors`` was called → no edges group is populated.
    assert batch._edges_group is None
    # Should not raise.
    _mark_halo_receiver_edges_as_padding(batch, n_owned=4)
