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

"""The ``GraphPadder`` protocol + the unified ``resolve_cap`` caps-service.

``resolve_cap`` consolidates the cap resolvers used by the framework, AIMNet2,
and UMA into one function. These tests pin its behavior AND prove it reproduces
the exact capacities the three model implementations compute.
"""

from __future__ import annotations

import torch

from nvalchemi.distributed.graph_padder import (
    COOPadder,
    DenseBatchPadder,
    DensePadder,
    GraphPadder,
    resolve_cap,
)

# ----------------------------------------------------------------------
# resolve_cap — behavior
# ----------------------------------------------------------------------


def test_first_sight_sizes_with_initial_factor_and_stride():
    state: dict[str, int] = {}
    # 100 * 1.15 -> 115, +1 -> 116, ceil to 16 -> 128.
    assert resolve_cap(state, "atoms", 100, initial_factor=1.15, stride=16) == 128
    assert state["atoms"] == 128


def test_grow_only_does_not_shrink_when_real_drops():
    state = {"atoms": 128}
    # real drops to 50: still fits, cap unchanged (no recompile churn).
    assert resolve_cap(state, "atoms", 50, initial_factor=1.15, stride=16) == 128
    assert state["atoms"] == 128


def test_small_fluctuation_stays_in_bucket():
    state: dict[str, int] = {}
    c0 = resolve_cap(state, "atoms", 100, initial_factor=1.15, stride=16)
    # real wiggles within the headroom -> same cap (the whole point: no recompile).
    for r in (101, 110, 120, 127):
        assert resolve_cap(state, "atoms", r, initial_factor=1.15, stride=16) == c0


def test_overflow_regrows_with_grow_factor():
    state = {"atoms": 128}
    # real exceeds cap -> regrow with grow_factor (1.30), ceil to 16.
    # 200 * 1.30 -> 260, +1 -> 261, ceil 16 -> 272.
    assert (
        resolve_cap(
            state, "atoms", 200, initial_factor=1.15, grow_factor=1.30, stride=16
        )
        == 272
    )


def test_strict_gt_false_requires_strictly_larger_cap():
    # atoms use ``>=`` (strict_gt=False): real == cap must regrow, because the
    # dead row lives at index cap-1 (real must be < cap).
    state = {"atoms": 128}
    out = resolve_cap(
        state,
        "atoms",
        128,
        initial_factor=1.15,
        grow_factor=1.30,
        stride=16,
        strict_gt=False,
    )
    assert out > 128
    # with strict_gt=True (edges/send), real == cap is fine (no regrow).
    state2 = {"edges": 128}
    assert resolve_cap(state2, "edges", 128, initial_factor=1.35, stride=16) == 128


# ----------------------------------------------------------------------
# resolve_cap — matches the three model implementations
# ----------------------------------------------------------------------


def _legacy_round_cap(x: int, f: float) -> int:
    # The framework / AIMNet2 ``_round_cap`` (ceil to 16).
    return ((int(x * f) + 1 + 15) // 16) * 16


def test_matches_framework_round_cap_first_sight():
    # Framework atom/edge/send initial factors: 1.15 / 1.35 / 1.20, stride 16.
    for real, factor in [(137, 1.15), (642, 1.35), (33, 1.20), (1, 1.15), (0, 1.15)]:
        state: dict[str, int] = {}
        got = resolve_cap(state, "k", real, initial_factor=factor, stride=16)
        assert got == _legacy_round_cap(real, factor), (real, factor)


def _legacy_uma_resolve(cap_state: dict, key: str, real: int, extra: int, stride: int):
    # UMA's ``_resolve_cap``: growth 1.15, stride bucket, grow-only.
    need = real + extra
    cap = cap_state.get(key, 0)
    if cap < need:
        grown = int(need * 1.15) + 1
        cap = ((grown + stride - 1) // stride) * stride
        cap_state[key] = cap
    return cap


def test_matches_uma_resolve_cap():
    # UMA: n_cap stride 64 (+2 dead-anchor slots), e_cap stride 1024.
    for key, real, extra, stride in [
        ("n_cap", 137, 2, 64),
        ("e_cap", 4096, 0, 1024),
        ("n_cap", 50, 2, 64),
    ]:
        mine: dict[str, int] = {}
        legacy: dict[str, int] = {}
        got = resolve_cap(
            mine,
            key,
            real,
            initial_factor=1.15,
            grow_factor=1.15,
            stride=stride,
            extra=extra,
        )
        exp = _legacy_uma_resolve(legacy, key, real, extra, stride)
        assert got == exp, (key, real, extra, stride, got, exp)


# ----------------------------------------------------------------------
# GraphPadder protocol
# ----------------------------------------------------------------------


def test_graph_padder_is_runtime_checkable():
    class _Conforming:
        def pad(self, data, cap_state):
            return data

        def unpad(self, output):
            return output

    class _Missing:
        def pad(self, data, cap_state):
            return data

    assert isinstance(_Conforming(), GraphPadder)
    assert not isinstance(_Missing(), GraphPadder)  # no unpad


def test_coopadder_conforms_to_protocol_and_unpad_is_identity():
    # The built-in COO padder is the inferred default GraphPadder. Its pad body
    # (storage-shape correctness + dead-node routing) is exercised end-to-end by
    # the GPU recompile gate + the MACE cueq compile equivalence gate, which run
    # it on real COO graphs; here we pin the cheap CPU-checkable invariants.
    padder = COOPadder()
    assert isinstance(padder, GraphPadder)
    # unpad is a no-op: the owned-only output consolidation drops dead rows.
    sentinel = object()
    assert padder.unpad(sentinel) is sentinel
    # pad on an absent padded view is a safe no-op (returns the input). The
    # padder owns cap resolution, so it takes a mutable cap_state dict.
    assert padder.pad(None, {}) is None


def test_densepadder_pads_rows_repoints_sentinel_and_unpads():
    # Dense (N, K) nbmat layout: N input rows where the last is the model's own
    # sentinel/pad atom, so the real atom count is N - 1.
    n_in, K = 6, 3
    sent_old = n_in - 1  # == real atom count
    coord = torch.arange(n_in * 3, dtype=torch.float32).reshape(n_in, 3)
    numbers = torch.arange(n_in, dtype=torch.long)
    mol_idx = torch.zeros(n_in, dtype=torch.long)
    nbmat = torch.full((n_in, K), sent_old, dtype=torch.long)  # all sentinel...
    nbmat[0, 0] = 1  # ...except one genuine neighbor
    data = {
        "coord": coord,
        "numbers": numbers,
        "mol_idx": mol_idx,
        "nbmat": nbmat.clone(),
        "_n_systems_halo": 2,
    }
    padder = DensePadder(
        count_key="coord",
        nbmat_key="nbmat",
        row_pads={"coord": 0, "numbers": 0, "mol_idx": DensePadder.LAST_SYSTEM},
        atom_output_keys=("forces",),
        n_systems_key="_n_systems_halo",
        stride=16,
    )
    assert isinstance(padder, GraphPadder)
    cap_state: dict[str, int] = {}
    out = padder.pad(data, cap_state)
    n_cap = cap_state["atoms"]
    dead = n_cap - 1
    assert n_cap % 16 == 0 and n_cap > n_in  # rounded up, strictly above real
    # All fields padded to the atom cap.
    assert out["coord"].shape[0] == n_cap
    assert out["nbmat"].shape[0] == n_cap
    # The genuine neighbor is preserved; sentinel entries -> dead row.
    assert out["nbmat"][0, 0] == 1
    assert (out["nbmat"][:n_in][nbmat >= sent_old] == dead).all()
    # Pad rows: nbmat = dead self-refs; numbers = 0; mol_idx = last system (1).
    assert (out["nbmat"][n_in:] == dead).all()
    assert (out["numbers"][n_in:] == 0).all()
    assert (out["mol_idx"][n_in:] == 1).all()
    # unpad strips per-atom outputs to the real count (sent_old), leaving
    # non-atom outputs (per-system energy) untouched.
    stripped = padder.unpad({"forces": torch.randn(n_cap, 3), "energy": torch.randn(2)})
    assert stripped["forces"].shape[0] == sent_old
    assert stripped["energy"].shape[0] == 2


def test_densepadder_unpad_honours_explicit_n_real():
    # On eager / sharded paths no pad() ran to stash a count, so the caller
    # passes n_real explicitly; unpad must strip to it (not the stashed value).
    padder = DensePadder(
        count_key="coord",
        nbmat_key="nbmat",
        row_pads={"coord": 0},
        atom_output_keys=("forces", "charges"),
    )
    assert padder._n_real is None  # no pad() ran
    out = padder.unpad(
        {
            "forces": torch.randn(10, 3),
            "charges": torch.randn(10),
            "energy": torch.randn(2),
        },
        n_real=7,
    )
    assert out["forces"].shape[0] == 7
    assert out["charges"].shape[0] == 7
    assert out["energy"].shape[0] == 2  # non-atom output untouched
    # No n_real and no stash -> no-op (returns output unchanged).
    same = padder.unpad({"forces": torch.randn(5, 3)})
    assert same["forces"].shape[0] == 5


# ----------------------------------------------------------------------
# DenseBatchPadder — batch-level dense (N, K) padding (AIMNet2)
# ----------------------------------------------------------------------


def _dense_batch(n_real: int, K: int):
    """A small halo-padded dense-nbmat ``Batch``: ``n_real`` owned+ghost atoms,
    unused neighbor slots == ``n_real`` (what ``compute_neighbors`` fills), plus
    one genuine neighbor."""
    from nvalchemi.data import AtomicData, Batch

    nbmat = torch.full((n_real, K), n_real, dtype=torch.long)  # all sentinel...
    nbmat[0, 0] = 2  # ...except one genuine neighbor
    data = AtomicData(
        positions=torch.arange(n_real * 3, dtype=torch.float32).reshape(n_real, 3),
        atomic_numbers=torch.full((n_real,), 6, dtype=torch.long),
        atomic_masses=torch.ones(n_real, dtype=torch.float32),
        neighbor_matrix=nbmat,
        neighbor_matrix_shifts=torch.zeros(n_real, K, 3, dtype=torch.float32),
    )
    return Batch.from_data_list([data])


def test_densebatchpadder_conforms_and_unpad_identity():
    padder = DenseBatchPadder()
    assert isinstance(padder, GraphPadder)
    sentinel = object()
    assert padder.unpad(sentinel) is sentinel
    assert padder.pad(None, {}) is None


def test_densebatchpadder_pads_batch_and_repoints_sentinel():
    n_real, K = 5, 3
    batch = _dense_batch(n_real, K)
    padder = DenseBatchPadder(stride=16)
    cap_state: dict[str, int] = {}
    out = padder.pad(batch, cap_state)
    n_cap = cap_state["atoms"]
    assert n_cap % 16 == 0 and n_cap > n_real  # rounded up, strictly above real

    # Every atom-level field padded to the cap.
    assert out.num_nodes == n_cap
    assert out.positions.shape[0] == n_cap
    assert out.neighbor_matrix.shape[0] == n_cap
    assert out.neighbor_matrix_shifts.shape[0] == n_cap

    nb = out.neighbor_matrix
    # The genuine neighbor survives; every sentinel (>= n_real) -> n_cap (the
    # index of the pad atom adapt_input will append).
    assert nb[0, 0] == 2
    real_rows = nb[:n_real]
    assert (real_rows[real_rows != 2] == n_cap).all()
    # Dead rows self-reference the pad-atom index too.
    assert (nb[n_real:] == n_cap).all()
    # Dead atoms are inert: Z=0, zero positions, joined to the last graph.
    assert (out.atomic_numbers[n_real:] == 0).all()
    assert (out.positions[n_real:] == 0).all()
    assert (out.batch_idx[n_real:] == out.num_graphs - 1).all()


def test_densebatchpadder_cap_is_grow_only_across_steps():
    padder = DenseBatchPadder(stride=16)
    cap_state: dict[str, int] = {}
    padder.pad(_dense_batch(5, 3), cap_state)
    first = cap_state["atoms"]
    # A smaller step reuses the same cap (no recompile churn).
    padder.pad(_dense_batch(4, 3), cap_state)
    assert cap_state["atoms"] == first
