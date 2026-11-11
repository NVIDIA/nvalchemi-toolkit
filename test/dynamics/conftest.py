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
"""
Shared pytest fixtures and helpers for the dynamics test suite.
"""

from __future__ import annotations

import itertools
from enum import Enum
from typing import TYPE_CHECKING

import torch

from nvalchemi.data import AtomicData, Batch
from nvalchemi.dynamics.base import DynamicsStage
from nvalchemi.hooks import DynamicsContext

if TYPE_CHECKING:
    from nvalchemi.dynamics.base import BaseDynamics

LATTICE_SPACING = 3.82
"""Simple-cubic spacing sitting at the energy minimum of the argon Lennard-Jones tests."""

ARGON_MASS = 39.948
"""Mass carried by every atom of the argon lattice builders."""


def make_lattice_data(
    cells: int = 3,
    spacing: float = LATTICE_SPACING,
    speed: float = 0.0,
    jitter: float = 0.0,
) -> AtomicData:
    """Return a simple-cubic argon lattice, optionally jittered and given velocities.

    The structure carries every field an NVE run reads: masses, velocities,
    a periodic cell, and zeroed ``energy`` and ``forces`` output buffers.
    """
    positions = torch.tensor(
        [
            [i * spacing, j * spacing, k * spacing]
            for i, j, k in itertools.product(range(cells), repeat=3)
        ],
        dtype=torch.float32,
    )
    n_atoms = positions.shape[0]
    if jitter != 0.0:
        generator = torch.Generator().manual_seed(5)
        offsets = torch.rand(n_atoms, 3, generator=generator) * 2.0 - 1.0
        positions = positions + jitter * offsets
    data = AtomicData(
        positions=positions,
        atomic_numbers=torch.full((n_atoms,), 18, dtype=torch.long),
        atomic_masses=torch.full((n_atoms,), ARGON_MASS),
        cell=torch.eye(3).unsqueeze(0) * (cells * spacing),
        pbc=torch.ones(1, 3, dtype=torch.bool),
        forces=torch.zeros(n_atoms, 3),
        energy=torch.zeros(1, 1),
    )
    velocities = torch.zeros(n_atoms, 3)
    if speed != 0.0:
        generator = torch.Generator().manual_seed(7)
        velocities = speed * torch.randn(n_atoms, 3, generator=generator)
        velocities -= velocities.mean(dim=0, keepdim=True)
    data.add_node_property("velocities", velocities)
    return data


def make_lattice_batch(
    cells: int = 3,
    spacing: float = LATTICE_SPACING,
    speed: float = 0.0,
    jitter: float = 0.0,
) -> Batch:
    """Return the :func:`make_lattice_data` lattice as a one-graph batch."""
    return Batch.from_data_list([make_lattice_data(cells, spacing, speed, jitter)])


def make_dynamics_context(
    batch: Batch,
    dynamics: BaseDynamics,
    converged: torch.Tensor | None = None,
    *,
    active_graph_mask: torch.Tensor | None = None,
) -> DynamicsContext:
    """Build a DynamicsContext from a batch and dynamics instance.

    Parameters
    ----------
    batch : Batch
        Batch to expose through the context.
    dynamics : BaseDynamics
        Dynamics instance providing step, model, rank, and convergence state,
        and exposed as the context's ``workflow`` as the engine does.
    converged : torch.Tensor | None, optional
        Explicit converged graph indices. When ``None``, use
        ``dynamics._last_converged``.
    active_graph_mask : torch.Tensor | None, optional
        Mask of the graphs active in the dispatch, as a status-filtered
        engine passes it. Default ``None`` (no status filtering).

    Returns
    -------
    DynamicsContext
        Context object for direct hook unit tests.
    """
    raw = converged if converged is not None else dynamics._last_converged
    if raw is not None:
        mask = batch.positions.new_zeros(batch.num_graphs, dtype=torch.bool)
        mask[raw] = True
    else:
        mask = None
    return DynamicsContext(
        batch=batch,
        step_count=dynamics.step_count,
        model=dynamics.model,
        converged_mask=mask,
        active_graph_mask=active_graph_mask,
        global_rank=dynamics.global_rank,
        workflow=dynamics,
    )


class RecordingHook:
    """
    A concrete hook implementation that records when it was called.

    This hook appends its name to a shared list each time it is invoked,
    allowing tests to verify execution order.

    Attributes
    ----------
    frequency : int
        Execute every N steps.
    stage : DynamicsStage
        Stage at which to fire.
    name : str
        Identifier for this hook.
    record_list : list[str]
        Shared list to append name to when called.

    Examples
    --------
    >>> record_list = []
    >>> hook = RecordingHook(DynamicsStage.AFTER_STEP, record_list, name="my_hook")
    >>> hook.stage
    <DynamicsStage.AFTER_STEP: 7>
    """

    def __init__(
        self,
        stage: DynamicsStage,
        record_list: list[str],
        name: str | None = None,
        frequency: int = 1,
    ) -> None:
        """
        Initialize the recording hook.

        Parameters
        ----------
        stage : DynamicsStage
            The stage at which this hook fires.
        record_list : list[str]
            List to append to when called.
        name : str | None, optional
            Name identifier. Defaults to stage name.
        frequency : int, optional
            How often to fire (every N steps). Default 1.
        """
        self.stage = stage
        self.frequency = frequency
        self.name = name if name is not None else stage.name
        self.record_list = record_list

    def __call__(self, ctx: DynamicsContext, stage: Enum) -> None:
        """
        Record that this hook was called.

        Parameters
        ----------
        ctx : DynamicsContext
            The hook context (unused).
        stage : Enum
            The stage being dispatched (unused).
        """
        self.record_list.append(self.name)


# ---------------------------------------------------------------------------
# Shared batch/model builders
# ---------------------------------------------------------------------------


def _make_atomic_data(
    n_atoms: int, seed: int = 0, with_cell: bool = False
) -> AtomicData:
    """Return a minimal AtomicData suitable for dynamics tests."""
    g = torch.Generator()
    g.manual_seed(seed)
    kwargs = dict(
        positions=torch.randn(n_atoms, 3, generator=g),
        atomic_numbers=torch.randint(1, 10, (n_atoms,), dtype=torch.long, generator=g),
        atomic_masses=torch.ones(n_atoms),
        forces=torch.zeros(n_atoms, 3),
        energy=torch.zeros(1, 1),
    )
    if with_cell:
        kwargs["cell"] = torch.eye(3).unsqueeze(0)
        kwargs["stress"] = torch.zeros(1, 3, 3)
    data = AtomicData(**kwargs)
    data.add_node_property("velocities", torch.zeros(n_atoms, 3))
    return data


def _make_batch(
    n_systems: int, n_atoms_each: int = 4, seed: int = 0, with_cell: bool = False
) -> Batch:
    data_list = [
        _make_atomic_data(n_atoms_each, seed + i, with_cell=with_cell)
        for i in range(n_systems)
    ]
    return Batch.from_data_list(data_list)


def _make_stress_model():
    """Return a DemoModelWrapper subclass that also reports zero stress.

    NPT/NPH declare ``"stress"`` in ``__needs_keys__``, so ``step()``
    requires the model to produce stress in its output dict.  This
    factory builds a minimal subclass that appends a (M, 3, 3) zero
    stress tensor so that ``_validate_model_outputs`` passes.  The
    actual stress value used by NPT/NPH kernels is read from
    ``batch.stress``, which is initialised to zeros when the batch is
    built with ``with_cell=True``.
    """
    from collections import OrderedDict

    from nvalchemi.models.base import ModelConfig
    from nvalchemi.models.demo import DemoModel, DemoModelWrapper

    class _Wrapper(DemoModelWrapper):
        def __init__(self):
            super().__init__(DemoModel())
            base = self.model_config
            self.model_config = ModelConfig(
                outputs=frozenset(set(base.outputs) | {"stress"}),
                autograd_outputs=base.autograd_outputs,
                autograd_inputs=base.autograd_inputs,
                required_inputs=base.required_inputs,
                optional_inputs=base.optional_inputs,
                supports_pbc=base.supports_pbc,
                neighbor_config=base.neighbor_config,
                needs_pbc=base.needs_pbc,
            )

        def adapt_output(self, model_output, data):
            M = data.num_graphs if hasattr(data, "num_graphs") else 1
            return OrderedDict(
                [
                    ("energy", model_output["energy"]),
                    ("forces", model_output["forces"]),
                    (
                        "stress",
                        torch.zeros(
                            M,
                            3,
                            3,
                            device=data.positions.device,
                            dtype=data.positions.dtype,
                        ),
                    ),
                ]
            )

    return _Wrapper()


def _make_model(needs_stress: bool = False):
    from nvalchemi.models.demo import DemoModel, DemoModelWrapper

    if needs_stress:
        return _make_stress_model()
    return DemoModelWrapper(DemoModel())


class _MockSampler:
    """Minimal sampler stub for inflight batching tests."""

    def __init__(self, replacements: list):
        self._queue = list(replacements)
        # Eagerly reflect exhausted state so base.py can snapshot it before requesting
        self.exhausted = len(self._queue) == 0

    @property
    def max_atoms(self) -> int | None:
        return None

    @property
    def max_edges(self) -> int | None:
        return None

    @property
    def max_batch_size(self) -> int | None:
        return None

    def request_replacements_budget(
        self,
        atom_budget: int | None = None,
        edge_budget: int | None = None,
        max_count: int | None = None,
    ) -> list:
        if not self._queue:
            self.exhausted = True
            return []
        result = self._queue.pop(0)
        self.exhausted = len(self._queue) == 0
        return [result]
