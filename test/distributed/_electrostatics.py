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
"""Shared test system for the distributed electrostatics (Ewald / PME) tests.

The Ewald and PME multi-GPU regressions share one geometry — a charge-neutral
simple-cubic NaCl-like lattice — because PME's real-space path reuses the Ewald
kernel, so the two cover identical geometry with different k-space algorithms.
Lives beside :mod:`_helpers` in ``test/distributed`` (on the test ``pythonpath``)
so the ``model/`` electrostatics tests can ``from _electrostatics import build_nacl``."""

from __future__ import annotations

import torch

__all__ = ["build_nacl"]


def build_nacl(
    n_side: int,
    box: float,
    *,
    jitter: float = 0.05,
    dtype: torch.dtype = torch.float32,
    seed: int = 0,
):
    """Build a periodic simple-cubic lattice of alternating ±1 charges.

    ``n_side`` atoms per axis on a ``box``-Å cubic cell, rattled by Gaussian
    ``jitter`` and wrapped back into the cell. Charges alternate +1/-1 along the
    flattened index (globally neutral for even atom count); species are tagged
    Na (11) / Cl (17), which is immaterial to the electrostatics kernels. Callers
    pick ``n_side``/``box`` to keep their partition non-degenerate.

    Parameters
    ----------
    n_side : int
        Atoms per axis; the system has ``n_side**3`` atoms.
    box : float
        Cubic cell edge length in Å.
    jitter : float, optional
        Standard deviation of the Gaussian position rattle in Å.
    dtype : torch.dtype, optional
        Floating dtype for positions / masses / charges / cell.
    seed : int, optional
        Seed for the rattle generator (CPU, deterministic).

    Returns
    -------
    tuple
        ``(positions, atomic_numbers, masses, charges, cell, pbc)`` — all CPU
        tensors; the cell is ``(3, 3)`` and ``pbc`` is ``(3,)`` all-True.
    """
    coords = torch.arange(n_side, dtype=dtype) * (box / n_side)
    gx, gy, gz = torch.meshgrid(coords, coords, coords, indexing="ij")
    positions = torch.stack([gx.flatten(), gy.flatten(), gz.flatten()], dim=-1)
    n = positions.shape[0]

    g = torch.Generator().manual_seed(seed)
    positions = positions + jitter * torch.randn(
        positions.shape, dtype=dtype, generator=g
    )
    positions = positions % box

    signs = torch.ones(n, dtype=dtype)
    signs[1::2] = -1.0
    charges = signs
    atomic_numbers = torch.where(
        signs > 0,
        torch.full((n,), 11, dtype=torch.long),
        torch.full((n,), 17, dtype=torch.long),
    )
    masses = torch.where(
        signs > 0,
        torch.full((n,), 22.99, dtype=dtype),
        torch.full((n,), 35.45, dtype=dtype),
    )
    cell = torch.eye(3, dtype=dtype) * box
    pbc = torch.ones(3, dtype=torch.bool)
    return positions, atomic_numbers, masses, charges, cell, pbc
