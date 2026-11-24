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
"""Tests for the alternating MC-MD block scheduler."""

from __future__ import annotations

import torch

from nvalchemi.data import AtomicData, Batch
from nvalchemi.dynamics.demo import DemoDynamics
from nvalchemi.hybrid import HybridMCMD
from nvalchemi.mc import SGC
from nvalchemi.models.demo import DemoModel, DemoModelWrapper


def test_scheduler_alternates_sgc_and_md_on_the_same_batch() -> None:
    """One block advances both stages without a batch conversion."""
    model = DemoModelWrapper(DemoModel())
    data = AtomicData(
        atomic_numbers=torch.tensor([1], dtype=torch.long),
        positions=torch.zeros(1, 3),
    )
    batch = Batch.from_data_list([data])
    batch.energy = torch.zeros(1, 1)
    batch.forces = torch.zeros(1, 3)
    mc = SGC(
        model=model,
        temperature=1000.0,
        species=[1, 2],
        chemical_potentials={1: 0.0, 2: 1.0e6},
        random_seed=4,
    )
    md = DemoDynamics(model=model, n_steps=None, dt=0.01)

    result = HybridMCMD(mc=mc, md=md, mc_steps=1, md_steps=1).run(batch, n_blocks=1)

    assert result is batch
    assert mc.step_count == 1
    assert md.step_count == 1
    assert batch.atomic_numbers.tolist() == [2]


def test_energy_only_mc_blocks_narrow_outputs_and_restore_them_for_md() -> None:
    """MC trials and their re-baseline see energy only; every MD call sees full outputs."""
    model = DemoModelWrapper(DemoModel())
    full = set(model.model_config.active_outputs)
    data = AtomicData(
        atomic_numbers=torch.tensor([1], dtype=torch.long),
        positions=torch.zeros(1, 3),
    )
    batch = Batch.from_data_list([data])
    batch.energy = torch.zeros(1, 1)
    batch.forces = torch.zeros(1, 3)
    mc = SGC(
        model=model,
        temperature=1000.0,
        species=[1, 2],
        chemical_potentials={1: 0.0, 2: 1.0e6},
        random_seed=4,
    )
    md = DemoDynamics(model=model, n_steps=None, dt=0.01)
    seen: list[tuple[str, frozenset[str]]] = []

    def spy(name: str, method):
        def wrapped(*args, **kwargs):
            seen.append((name, frozenset(model.model_config.active_outputs)))
            return method(*args, **kwargs)

        return wrapped

    mc.refresh_energy = spy("mc.refresh", mc.refresh_energy)
    mc.run = spy("mc.run", mc.run)
    md.compute = spy("md.compute", md.compute)
    md.run = spy("md.run", md.run)

    HybridMCMD(mc=mc, md=md, mc_steps=1, md_steps=1, mc_energy_only=True).run(
        batch, n_blocks=2
    )

    energy_only = frozenset({"energy"})
    assert [name for name, _ in seen if name.startswith("mc")] == [
        "mc.refresh",
        "mc.run",
    ] * 2
    assert all(
        outputs == energy_only for name, outputs in seen if name.startswith("mc")
    )
    assert all(
        outputs == frozenset(full) for name, outputs in seen if name.startswith("md")
    )
    assert set(model.model_config.active_outputs) == full
    assert batch.atomic_numbers.tolist() == [2]  # the MC move still happened


def _one_atom_batch() -> Batch:
    data = AtomicData(
        atomic_numbers=torch.tensor([1], dtype=torch.long),
        positions=torch.zeros(1, 3),
    )
    batch = Batch.from_data_list([data])
    batch.energy = torch.zeros(1, 1)
    batch.forces = torch.zeros(1, 3)
    return batch


def test_separate_models_rebaseline_mc_with_its_own_model_every_block() -> None:
    """With two models, MC never adopts an MD energy as its acceptance baseline."""
    mc_model = DemoModelWrapper(DemoModel())
    md_model = DemoModelWrapper(DemoModel())
    batch = _one_atom_batch()
    mc = SGC(
        model=mc_model,
        temperature=1000.0,
        species=[1, 2],
        chemical_potentials={1: 0.0, 2: 1.0e6},
        random_seed=4,
    )
    md = DemoDynamics(model=md_model, n_steps=None, dt=0.01)
    refreshed: list[int] = []
    refresh = mc.refresh_energy

    def spy(b):
        refreshed.append(1)
        return refresh(b)

    mc.refresh_energy = spy
    hybrid = HybridMCMD(mc=mc, md=md, mc_steps=1, md_steps=1)

    assert hybrid.separate_models
    hybrid.run(batch, n_blocks=3)
    assert len(refreshed) == 3
    assert mc.step_count == 3 and md.step_count == 3
    assert batch.atomic_numbers.tolist() == [2]


def test_shared_model_without_energy_only_keeps_adopting_md_energy() -> None:
    model = DemoModelWrapper(DemoModel())
    mc = SGC(
        model=model,
        temperature=1000.0,
        species=[1, 2],
        chemical_potentials={1: 0.0, 2: 1.0e6},
        random_seed=4,
    )
    md = DemoDynamics(model=model, n_steps=None, dt=0.01)
    calls: list[int] = []
    mc.refresh_energy = lambda b: calls.append(1)

    hybrid = HybridMCMD(mc=mc, md=md, mc_steps=1, md_steps=1)
    hybrid.run(_one_atom_batch(), n_blocks=2)

    assert not hybrid.separate_models
    assert calls == []


def _sgc(model: DemoModelWrapper) -> SGC:
    return SGC(
        model=model,
        temperature=1000.0,
        species=[1, 2],
        chemical_potentials={1: 0.0, 2: 1.0e6},
        random_seed=4,
    )


def test_before_md_block_runs_after_mc_and_before_each_md_force_call() -> None:
    """The hook sees the post-MC species and precedes every block's first MD evaluation."""
    md_model = DemoModelWrapper(DemoModel())
    mc = _sgc(DemoModelWrapper(DemoModel()))
    md = DemoDynamics(model=md_model, n_steps=None, dt=0.01)
    events: list[tuple[str, int]] = []
    compute = md.compute

    def spy_compute(b, *args, **kwargs):
        events.append(("md.compute", int(b.atomic_numbers[0])))
        return compute(b, *args, **kwargs)

    md.compute = spy_compute
    HybridMCMD(
        mc=mc,
        md=md,
        mc_steps=1,
        md_steps=1,
        before_md_block=lambda b: events.append(("prepare", int(b.atomic_numbers[0]))),
    ).run(_one_atom_batch(), n_blocks=2)

    names = [name for name, _ in events]
    prepares = [i for i, n in enumerate(names) if n == "prepare"]
    assert len(prepares) == 3  # initial evaluation + one per block
    assert all(names[i + 1] == "md.compute" for i in prepares)
    assert events[prepares[0]][1] == 1  # before any MC move
    assert events[prepares[1]][1] == 2  # after the first MC block transmuted 1 -> 2


def test_md_block_evaluates_the_model_once_per_step_plus_one() -> None:
    """md.run primes forces itself; the scheduler must not add a second evaluation."""
    model = DemoModelWrapper(DemoModel())
    md = DemoDynamics(model=model, n_steps=None, dt=0.01)
    calls: list[int] = []
    compute = md.compute

    def spy(b, *args, **kwargs):
        calls.append(1)
        return compute(b, *args, **kwargs)

    md.compute = spy
    HybridMCMD(mc=_sgc(model), md=md, mc_steps=1, md_steps=3).run(
        _one_atom_batch(), n_blocks=2
    )

    assert len(calls) == 1 + 2 * (3 + 1)


def test_zero_md_steps_runs_mc_only_blocks() -> None:
    """md_steps=0 (rigid-lattice MC) runs, with no MD evaluation between blocks."""
    model = DemoModelWrapper(DemoModel())
    mc = _sgc(model)
    md = DemoDynamics(model=model, n_steps=None, dt=0.01)

    HybridMCMD(mc=mc, md=md, mc_steps=2, md_steps=0).run(_one_atom_batch(), n_blocks=3)

    assert mc.step_count == 6
    assert md.step_count == 0


def test_after_md_block_runs_before_mc_adopts_the_energy() -> None:
    """The hook runs after every MD block (and the initial evaluation), even with md_steps=0."""
    model = DemoModelWrapper(DemoModel())
    mc = _sgc(model)
    md = DemoDynamics(model=model, n_steps=None, dt=0.01)
    events: list[str] = []
    synchronize = mc.synchronize

    def spy_synchronize(b):
        events.append("synchronize")
        return synchronize(b)

    mc.synchronize = spy_synchronize
    HybridMCMD(
        mc=mc,
        md=md,
        mc_steps=1,
        md_steps=0,
        after_md_block=lambda b: events.append("after"),
    ).run(_one_atom_batch(), n_blocks=2)

    assert events == ["after", "synchronize"] * 3


def test_on_block_sees_every_block_and_can_stop_early() -> None:
    """on_block receives the block index; returning True ends the run."""
    model = DemoModelWrapper(DemoModel())
    mc = _sgc(model)
    seen: list[int] = []

    def on_block(batch: Batch, block: int) -> bool:
        seen.append(block)
        return block == 1

    HybridMCMD(
        mc=mc,
        md=DemoDynamics(model=model, n_steps=None, dt=0.01),
        mc_steps=1,
        md_steps=1,
    ).run(_one_atom_batch(), n_blocks=5, on_block=on_block)

    assert seen == [0, 1]
    assert mc.step_count == 2
