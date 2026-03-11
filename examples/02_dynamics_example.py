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
FIRE Optimization and FusedStage (FIRE + Langevin): Dynamics Demo
==================================================================

This example walks through two use cases of the :mod:`nvalchemi.dynamics`
framework, following the same pattern shown in
:class:`~nvalchemi.dynamics.demo.DemoDynamics`:

* Create a dynamics object with a model, hooks, and a convergence criterion.
* Call :meth:`~nvalchemi.dynamics.base.BaseDynamics.run` — everything else
  happens automatically through the hook API.

**Part 1** — :class:`~nvalchemi.dynamics.optimizers.FIRE` geometry optimization.
A :class:`~nvalchemi.dynamics.base.ConvergenceHook` detects convergence
(fmax < 0.05) and fires an ``ON_CONVERGE`` hook; an ``AFTER_STEP`` hook logs
progress every N steps.

**Part 2** — A :class:`~nvalchemi.dynamics.base.FusedStage` that shares one
model forward pass across FIRE (status 0) and NVT Langevin MD (status 1).
The ``+`` operator composes the two sub-stages and auto-registers a
:class:`~nvalchemi.dynamics.base.ConvergenceHook` that migrates relaxed systems
from status 0 → 1.

.. note::

    :class:`~nvalchemi.models.demo.DemoModelWrapper` is used throughout.
    It supports conservative forces (via autograd) but not stresses, so
    variable-cell integrators are not shown here.
"""

from __future__ import annotations

import torch

from nvalchemi.data import AtomicData, Batch
from nvalchemi.dynamics import FIRE, NVTLangevin
from nvalchemi.dynamics.base import HookStageEnum
from nvalchemi.models.demo import DemoModelWrapper

# %%
# Setup — model and system builder
# ---------------------------------
# :class:`~nvalchemi.models.demo.DemoModelWrapper` computes per-atom energies
# and conservative forces via :func:`torch.autograd.grad`.  It requires no
# neighbor list or periodic boundary conditions.

torch.manual_seed(0)
model = DemoModelWrapper()
model.eval()


def _make_system(n_atoms: int, seed: int) -> AtomicData:
    """Build a small AtomicData system with all fields needed by the integrators.

    All integrators require ``positions``, ``atomic_numbers``,
    ``atomic_masses``, and ``velocities`` (node-level).  ``forces`` and
    ``energies`` are pre-allocated as zero placeholders and overwritten
    in-place by :meth:`~nvalchemi.dynamics.base.BaseDynamics.compute` on every
    step.
    """
    g = torch.Generator()
    g.manual_seed(seed)
    data = AtomicData(
        positions=torch.randn(n_atoms, 3, generator=g),
        atomic_numbers=torch.randint(1, 10, (n_atoms,), dtype=torch.long, generator=g),
        atomic_masses=torch.ones(n_atoms),   # unit masses (demo only)
        forces=torch.zeros(n_atoms, 3),      # placeholder; overwritten by compute()
        energies=torch.zeros(1, 1),          # placeholder; overwritten by compute()
    )
    # velocities is not a standard AtomicData field; add it as a node property.
    # FIRE starts from rest; NVTLangevin thermalises the velocities over time.
    data.add_node_property("velocities", torch.zeros(n_atoms, 3))
    return data

# %%
# Part 1: FIRE Geometry Optimization
# ------------------------------------
# Build a batch of three systems and relax with FIRE.
#
# ``FIRE.run(batch, n_steps=N)`` executes the full step loop.  The
# ``ConvergenceHook`` detects convergence each step and fires the
# ``ON_CONVERGE`` hooks when fmax < 0.05.  The ``FmaxLogHook`` prints
# progress at the ``AFTER_STEP`` stage every ``frequency`` steps.

print("=== Part 1: FIRE Geometry Optimization ===")

data_list_opt = [_make_system(n, seed) for n, seed in [(4, 1), (6, 2), (5, 3)]]
batch_opt = Batch.from_data_list(data_list_opt)
print(f"Batch: {batch_opt.num_graphs} systems, {batch_opt.num_nodes} atoms total\n")

from nvalchemi.dynamics.base import ConvergenceHook

fire_opt = FIRE(
    model=model,
    dt=0.1,
    n_steps=200,
    # ConvergenceHook evaluates per-atom force norms, scatter-maxes them to
    # per-system fmax, and marks systems with fmax <= 0.05 as converged.
    convergence_hook=ConvergenceHook(
        criteria=[
            {"key": "forces", "threshold": 0.05, "reduce_op": "norm", "reduce_dims": -1}
        ]
    ),
)

batch_opt = fire_opt.run(batch_opt)
print(f"\nCompleted {fire_opt.step_count} FIRE steps.")

# %%
# Part 2: FusedStage — FIRE Relaxation → NVT Langevin MD
# --------------------------------------------------------
# A :class:`~nvalchemi.dynamics.base.FusedStage` composes sub-stages that
# share **one** model forward pass per step.  Each system carries a ``status``
# field that routes it to the corresponding sub-stage:
#
# * **status = 0** → processed by FIRE (relaxation phase)
# * **status = 1** → processed by NVTLangevin (MD sampling phase)
#
# The ``+`` operator creates the fused stage and auto-registers a
# :class:`~nvalchemi.dynamics.base.ConvergenceHook` on the FIRE sub-stage.
# That hook checks ``batch.fmax`` after each step and migrates systems that
# satisfy fmax < 0.05 from status 0 → 1.
#
# We register a ``ComputeFmaxHook`` at ``AFTER_COMPUTE`` on the FIRE sub-stage
# to keep ``batch.fmax`` current, since the auto-registered hook depends on it.
# A ``StatusLogHook`` at ``AFTER_STEP`` prints the status distribution and
# ``TransitionLogHook`` at ``ON_CONVERGE`` fires when FIRE detects relaxation.

print("\n\n=== Part 2: FusedStage — FIRE + NVTLangevin ===")


# Build a fresh batch and attach status.
data_list_fused = [_make_system(n, seed) for n, seed in [(4, 10), (6, 11), (5, 12)]]
batch_fused = Batch.from_data_list(data_list_fused)

# All systems start in the FIRE stage (status = 0).
batch_fused["status"] = torch.zeros(
    batch_fused.num_graphs, 1, dtype=torch.long
)

# Create FIRE sub-stage.
# ComputeFmaxHook populates batch.fmax so the auto-registered convergence hook
# (which checks batch.fmax by default) can function.
# StatusLogHook and TransitionLogHook handle the human-readable output.
# The convergence_hook here drives ON_CONVERGE hook firing; status migration
# from 0→1 is handled by the ConvergenceHook auto-registered by FusedStage.
fire_stage = FIRE(
    model=model,
    dt=0.1,
    convergence_hook=ConvergenceHook(
        criteria=[
            {"key": "forces", "threshold": 0.05, "reduce_op": "norm", "reduce_dims": -1}
        ]
    ),
    n_steps = 200,
)

langevin_stage = NVTLangevin(
    model=model,
    dt=0.5,
    temperature=300.0,
    friction=0.1,
    random_seed=42,
    n_steps = 200,
)

# ``fire_stage + langevin_stage`` creates:
#   FusedStage(sub_stages=[(0, fire_stage), (1, langevin_stage)], exit_status=2)
# and auto-registers ConvergenceHook(key="fmax", threshold=0.05, 0→1)
# on fire_stage's AFTER_STEP hook list.
fused = fire_stage + langevin_stage
print(f"Created: {fused}\n")

# Run for a bounded number of fused steps.
n_fused_steps = 450
batch_fused = fused.run(batch_fused, n_steps=n_fused_steps)

status_final = batch_fused.status.squeeze(-1).tolist()
print(f"\nFinal status: {status_final}  (0=FIRE, 1=Langevin)")
print(f"FusedStage total steps: {fused.step_count}")
