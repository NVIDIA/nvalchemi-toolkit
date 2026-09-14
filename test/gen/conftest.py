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
"""Shared helpers for the generative API test suite.

Mirrors the dynamics/training convention: dummy-data builders and trivial
generating/materialization functions live in a per-suite ``conftest.py`` and
are imported by the test modules (``from test.gen.conftest import
make_batch``) instead of being redefined per module.
"""

from __future__ import annotations

import torch
from tensordict import TensorDict

from nvalchemi.data import AtomicData, Batch


def make_atomic_data(num_atoms: int = 3) -> AtomicData:
    """Build a minimal :class:`AtomicData` for tests.

    Parameters
    ----------
    num_atoms
        Number of atoms in the dummy structure.

    Returns
    -------
    AtomicData
        A small structure with random positions and carbon atomic numbers.
    """
    return AtomicData(
        positions=torch.randn(num_atoms, 3),
        atomic_numbers=torch.full((num_atoms,), 6, dtype=torch.long),
    )


def make_batch(num_graphs: int = 2) -> Batch:
    """Build a small :class:`Batch` for tests.

    Parameters
    ----------
    num_graphs
        Number of graphs to batch.

    Returns
    -------
    Batch
        A batch of dummy structures.
    """
    return Batch.from_data_list([make_atomic_data() for _ in range(num_graphs)])


def trivial_generate(model, *, num_samples=1, rng=None, cond=None, **kwargs):
    """A minimal :class:`~nvalchemi.gen.GeneratingFunction` for tests.

    Parameters
    ----------
    model
        Generative model (ignored).
    num_samples
        Number of draws (used only when ``cond`` is ``None``).
    rng
        Optional generator (ignored).
    cond
        Conditioning batch, if any; the sample's leading size matches it.
    **kwargs
        Family-specific options (ignored).

    Returns
    -------
    TensorDict
        Zeros under the ``"x1"`` key, aligned with ``cond``.
    """
    del model, rng, kwargs
    n = cond.num_graphs if isinstance(cond, Batch) else num_samples
    return TensorDict({"x1": torch.zeros(n, 1, 3)}, batch_size=[n])


def zeros_to_batch(sample: TensorDict, batch) -> Batch:
    """Materialization building a fresh batch sized like the sample.

    Parameters
    ----------
    sample
        Sample TensorDict; its leading size sets the graph count.
    batch
        Conditioning batch (ignored).

    Returns
    -------
    Batch
        ``sample.batch_size[0]`` dummy graphs.
    """
    del batch
    return make_batch(sample.batch_size[0])
