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


def trivial_generate(inputs=None, *, num_samples=1, rng=None, **kwargs):
    """A minimal :class:`~nvalchemi.gen.GeneratingFunction` for tests.

    Parameters
    ----------
    inputs
        Conditioning batch, if any; the sample's leading size matches it.
    num_samples
        Number of draws (used only when ``inputs`` is not a batch).
    rng
        Optional generator (ignored).
    **kwargs
        Family-specific options (ignored).

    Returns
    -------
    TensorDict
        Zeros under the ``"x1"`` key, aligned with ``inputs``.
    """
    del rng, kwargs
    n = inputs.num_graphs if isinstance(inputs, Batch) else num_samples
    return TensorDict({"x1": torch.zeros(n, 1, 3)}, batch_size=[n])


def batch_generate(inputs=None, *, num_samples=1, rng=None, **kwargs):
    """A minimal generating function returning a :class:`Batch` directly.

    Exercises the driver's raw passthrough: no ``batch_mapping`` is needed
    when the function already returns a ``Batch``.

    Parameters
    ----------
    inputs
        Conditioning batch, if any; the batch's graph count matches it.
    num_samples
        Number of draws (used only when ``inputs`` is not a batch).
    rng
        Optional generator (ignored).
    **kwargs
        Family-specific options (ignored).

    Returns
    -------
    Batch
        ``num_samples`` (or ``inputs.num_graphs``) dummy graphs.
    """
    del rng, kwargs
    n = inputs.num_graphs if isinstance(inputs, Batch) else num_samples
    return make_batch(n)


def zeros_to_batch(sample: TensorDict) -> Batch:
    """Materialization building a fresh batch sized like the sample.

    Parameters
    ----------
    sample
        Sample TensorDict; its leading size sets the graph count.

    Returns
    -------
    Batch
        ``sample.batch_size[0]`` dummy graphs.
    """
    return make_batch(sample.batch_size[0])


def passthrough_mapping(sample) -> Batch:
    """Materialization returning the sample unchanged (already a ``Batch``).

    Parameters
    ----------
    sample
        The raw sample, already a :class:`Batch`.

    Returns
    -------
    Batch
        ``sample``, unchanged.
    """
    return sample


def tile_condition(inputs, *, num_samples=None, rng=None):
    """Trivial condition callable: pass ``inputs`` through unchanged.

    Parameters
    ----------
    inputs
        The call's raw inputs.
    num_samples
        Resolved draw count (accepted for the conditioning signature; unused).
    rng
        Resolved RNG (accepted for the conditioning signature; unused).

    Returns
    -------
    Any
        ``inputs``, unchanged.
    """
    del num_samples, rng
    return inputs


class DeviceAwareGenerate:
    """Generating function object carrying ``device`` and materializing there.

    Exercises the driver's device defaults chain, session stream creation,
    and the device-residency check on any host: the object declares a device
    (readable via the chain) and returns batches built on it.
    """

    def __init__(self, device: str | torch.device) -> None:
        self.device = torch.device(device)

    def __call__(self, inputs=None, *, num_samples=1, rng=None, **kwargs) -> Batch:
        """Return a batch of dummy graphs resident on ``self.device``.

        Parameters
        ----------
        inputs
            Conditioning batch, if any; the batch's graph count matches it.
        num_samples
            Number of draws (used only when ``inputs`` is not a batch).
        rng
            Optional generator (ignored).
        **kwargs
            Family-specific options (ignored).

        Returns
        -------
        Batch
            Dummy graphs on ``self.device``.
        """
        del rng, kwargs
        n = inputs.num_graphs if isinstance(inputs, Batch) else num_samples
        return make_batch(n).to(self.device)
