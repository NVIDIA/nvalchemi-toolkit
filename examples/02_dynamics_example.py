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
# Hook definitions
# -----------------
# Hooks are plain objects with a ``stage`` attribute and a
# ``__call__(batch, dynamics)`` signature.  They are registered at
# construction time via the ``hooks=[...]`` argument.


class FmaxLogHook:
    """Log per-system maximum force norm (fmax) at regular intervals.

    Registered at the ``AFTER_STEP`` stage so that forces are always
    freshly computed before the hook runs.
    """

    def __init__(self, frequency: int = 10) -> None:
        self.frequency = frequency
        self.stage = HookStageEnum.AFTER_STEP

    def __call__(self, batch: Batch, dynamics) -> None:
        f_norm = torch.linalg.vector_norm(batch.forces, dim=-1)  # [N]
        M = batch.num_graphs
        fmax = torch.full((M,), float("-inf"), dtype=f_norm.dtype, device=f_norm.device)
        fmax.scatter_reduce_(0, batch.batch, f_norm, reduce="amax", include_self=False)
        vals = "  ".join(f"sys{i}={fmax[i].item():.4f}" for i in range(M))
        print(f"  step {dynamics.step_count:4d} | fmax: {vals}")


class ConvergeLogHook:
    """Print a notification when the ``ON_CONVERGE`` hook stage fires."""

    frequency: int = 1
    stage: HookStageEnum = HookStageEnum.ON_CONVERGE

    def __call__(self, batch: Batch, dynamics) -> None:
        print(f"  System(s) converged at step {dynamics.step_count}.")


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
    hooks=[FmaxLogHook(frequency=20), ConvergeLogHook()],
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


class ComputeFmaxHook:
    """Populate ``batch.fmax`` (per-system max force norm) after each compute.

    Registered at ``AFTER_COMPUTE`` so that ``batch.fmax`` is always fresh
    before the ``AFTER_STEP`` convergence hooks evaluate it.
    """

    frequency: int = 1
    stage: HookStageEnum = HookStageEnum.AFTER_COMPUTE

    def __call__(self, batch: Batch, dynamics) -> None:
        with torch.no_grad():
            f_norm = torch.linalg.vector_norm(batch.forces, dim=-1)  # [N]
            fmax = torch.full(
                (batch.num_graphs,),
                float("-inf"),
                dtype=f_norm.dtype,
                device=f_norm.device,
            )
            fmax.scatter_reduce_(
                0, batch.batch, f_norm, reduce="amax", include_self=False
            )
        batch.__dict__["fmax"] = fmax


class StatusLogHook:
    """Log the status distribution and per-system fmax every N fused steps."""

    def __init__(self, frequency: int = 30) -> None:
        self.frequency = frequency
        self.stage = HookStageEnum.AFTER_STEP

    def __call__(self, batch: Batch, dynamics) -> None:
        status = batch.status.squeeze(-1)
        n_fire = int((status == 0).sum())
        n_md = int((status == 1).sum())
        fmax = getattr(batch, "fmax", None)
        fmax_str = (
            "  ".join(f"sys{i}={fmax[i].item():.4f}" for i in range(batch.num_graphs))
            if fmax is not None
            else "n/a"
        )
        print(
            f"  step {dynamics.step_count:4d} | FIRE={n_fire}  Langevin={n_md}"
            f"  fmax: {fmax_str}"
        )


class TransitionLogHook:
    """Fire when FIRE's convergence criterion is met (ON_CONVERGE)."""

    frequency: int = 1
    stage: HookStageEnum = HookStageEnum.ON_CONVERGE

    def __call__(self, batch: Batch, dynamics) -> None:
        status = batch.status.squeeze(-1)
        n_md = int((status == 1).sum())
        print(f"  FIRE converged at step {dynamics.step_count}: {n_md} system(s) in Langevin MD")


# Build a fresh batch and attach status.
data_list_fused = [_make_system(n, seed) for n, seed in [(4, 10), (6, 11), (5, 12)]]
batch_fused = Batch.from_data_list(data_list_fused)

# All systems start in the FIRE stage (status = 0).
batch_fused.__dict__["status"] = torch.zeros(
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
    hooks=[ComputeFmaxHook(), StatusLogHook(frequency=30), TransitionLogHook()],
    convergence_hook=ConvergenceHook(
        criteria=[
            {"key": "forces", "threshold": 0.05, "reduce_op": "norm", "reduce_dims": -1}
        ]
    ),
)

langevin_stage = NVTLangevin(
    model=model,
    dt=0.5,
    temperature=300.0,
    friction=0.1,
    random_seed=42,
)

# ``fire_stage + langevin_stage`` creates:
#   FusedStage(sub_stages=[(0, fire_stage), (1, langevin_stage)], exit_status=2)
# and auto-registers ConvergenceHook(key="fmax", threshold=0.05, 0→1)
# on fire_stage's AFTER_STEP hook list.
fused = fire_stage + langevin_stage
print(f"Created: {fused}\n")

# Sub-stage integrator state is lazily initialised in BaseDynamics.step().
# FusedStage drives sub-stages via masked_update() instead, so we initialise
# their state explicitly before the first fused step.
fire_stage._init_state(batch_fused)
object.__setattr__(fire_stage, "_state_initialized", True)
langevin_stage._init_state(batch_fused)
object.__setattr__(langevin_stage, "_state_initialized", True)

# Run for a bounded number of fused steps.
# In production, fused.run(batch) loops until FusedStage.all_complete() returns
# True (all systems at exit_status=2), which requires a convergence criterion
# on the Langevin sub-stage.  Here we use a fixed step count for the demo.
n_fused_steps = 150
for _ in range(n_fused_steps):
    fused.step(batch_fused)

status_final = batch_fused.status.squeeze(-1).tolist()
print(f"\nFinal status: {status_final}  (0=FIRE, 1=Langevin)")
print(f"FusedStage total steps: {fused.step_count}")
