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
"""Tests for the chemistry-level particle-halo orchestration."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import torch

import nvalchemi.distributed.particle_halo as particle_halo_module
from nvalchemi.distributed._core.halo_types import ParticleHaloMetadata


class _LocalShard:
    """Minimal shard stand-in used by :func:`halo_exchange`."""

    def __init__(self, local: torch.Tensor) -> None:
        self.local = local

    def to_local(self) -> torch.Tensor:
        return self.local


def test_halo_exchange_reuse_refreshes_cell_and_pbc(monkeypatch: Any) -> None:
    """A reused padded Batch must see the current system-level geometry.

    NPT/NPH update the owned batch's cell without necessarily changing the
    owned or halo atom counts.  That takes ``halo_exchange`` through its
    shape-compatible reuse branch, where stale ``cell`` / ``pbc`` values would
    otherwise survive from the first model call.
    """
    positions = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float32)
    fields = {
        "positions": _LocalShard(positions),
        "atomic_numbers": _LocalShard(torch.tensor([6, 8], dtype=torch.int64)),
        "atomic_masses": _LocalShard(torch.tensor([12.0, 16.0])),
    }
    old_cell = torch.eye(3).unsqueeze(0) * 10.0
    old_pbc = torch.tensor([[True, True, True]])
    state = SimpleNamespace(
        positions=fields["positions"],
        cell=old_cell,
        pbc=old_pbc,
        padded_batch=None,
        halo_meta=None,
        atom_fields=lambda: dict(fields),
    )

    # Keep this test focused on Batch reuse rather than communication.  The
    # low-level routing is covered by the multi-rank tests in
    # ``_core/test_halo_primitives_roundtrip.py``.
    def _identity_padding(
        local_positions: torch.Tensor, _config: Any
    ) -> tuple[torch.Tensor, ParticleHaloMetadata]:
        n = local_positions.shape[0]
        return local_positions, ParticleHaloMetadata(
            n_owned=n,
            n_padded=n,
            send_indices=[],
            send_sizes=[],
            recv_sizes=[],
        )

    monkeypatch.setattr(
        particle_halo_module, "particle_halo_padding", _identity_padding
    )
    monkeypatch.setattr(
        particle_halo_module,
        "pad_field",
        lambda shard, _meta, _config: shard.to_local(),
    )

    particle_halo_module.halo_exchange(state, SimpleNamespace())
    padded = state.padded_batch
    assert padded is not None
    torch.testing.assert_close(padded.cell, old_cell)
    torch.testing.assert_close(padded.pbc, old_pbc)
    cell_ptr = padded.cell.data_ptr()
    pbc_ptr = padded.pbc.data_ptr()
    padded.pbc.zero_()

    # Include shear so the assertion checks the whole lattice matrix, not just
    # a scalar box length.
    new_cell = torch.tensor(
        [[[10.5, 0.0, 0.0], [0.0, 11.0, 0.0], [1.25, 0.0, 12.0]]],
    )
    state.cell = new_cell

    particle_halo_module.halo_exchange(state, SimpleNamespace())

    # Prove that the shape-compatible reuse path ran, then check that it did
    # not leave system metadata from the previous step behind.
    assert state.padded_batch is padded
    assert padded.cell.data_ptr() == cell_ptr
    assert padded.pbc.data_ptr() == pbc_ptr
    torch.testing.assert_close(padded.cell, new_cell)
    torch.testing.assert_close(padded.pbc, old_pbc)
