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
UMA Foundation Model: NVE / NVT / NPT Molecular Dynamics
========================================================

UMA (Universal Models for Atoms, from fairchem-core) is a multi-task
foundation model: a single checkpoint ships heads for molecules (OMol),
bulk crystals (OMat), catalysis (OC20), direct air capture (ODAC), and
molecular crystals (OMC).  :class:`~nvalchemi.models.uma.UMAWrapper`
exposes one task head at a time and computes conservative forces (and,
for periodic tasks, stress) via autograd.

This example drives UMA through nvalchemi's velocity-Verlet (NVE), BAOAB
Langevin (NVT), or Martyna-Tobias-Klein (NPT) integrator.  Observation is
delegated to built-in hooks: ``LoggingHook`` logs potential energy,
temperature, and max force, while ``EnergyDriftMonitorHook`` checks NVE
energy conservation.  Energy drift under NVE is the standard check that
the conservative-force path is wired correctly end-to-end; the NPT mode
additionally exercises the **stress** path (the barostat couples the
cell to the model's stress tensor).

Key concepts demonstrated
-------------------------
* Loading a UMA checkpoint and selecting a task head via
  ``UMAWrapper.from_checkpoint(..., task_name=...)``.
* Reaching fairchem's ``torch.compile`` path through the
  ``inference_settings`` argument (``"turbo"``).
* Driving energy/force dynamics (NVE/NVT) and stress-coupled dynamics
  (NPT, periodic ``omat`` head) from the same wrapper.
* Evolving a **batch** of replicas (perturbed starts) in a single shared
  forward pass per step.
* That UMA needs **no** :class:`~nvalchemi.hooks.NeighborListHook` — the
  predict unit builds its own graph internally, so the wrapper plugs
  straight into an integrator.
* Observing a run with built-in
  :class:`~nvalchemi.dynamics.hooks.LoggingHook` and
  :class:`~nvalchemi.dynamics.hooks.EnergyDriftMonitorHook` instead of a
  hand-rolled callback.

Setting up UMA
--------------
UMA checkpoints live in the **gated** ``facebook/UMA`` HuggingFace repo.
To run this example end-to-end:

1. Install the optional dependency (its torch pin conflicts with the
   ``mace`` / ``cuXX`` extras, so use a dedicated environment)::

       uv venv .venv-uma && uv sync --extra uma
       # or, with pip:  pip install 'nvalchemi-toolkit[uma]'

2. Request access at https://huggingface.co/facebook/UMA (one-time
   approval).

3. Create a read token at https://huggingface.co/settings/tokens and
   authenticate, either with::

       huggingface-cli login

   or by exporting it for the shell session::

       export HF_TOKEN=hf_xxx

The first :meth:`~nvalchemi.models.uma.UMAWrapper.from_checkpoint` call
downloads and caches the checkpoint (under ``~/.cache/fairchem``);
subsequent runs reuse the cache.  If the checkpoint cannot be loaded
(no access, missing token, or ``fairchem-core`` not installed) the
example prints guidance and skips the trajectory so it stays safe to run
in a documentation build.
"""

from __future__ import annotations

import logging
import sys
import time
from typing import Any

import numpy as np
import torch
from ase import Atoms
from ase.build import bulk
from ase.data import atomic_masses as ASE_ATOMIC_MASSES

from nvalchemi.data import AtomicData, Batch
from nvalchemi.dynamics.hooks import EnergyDriftMonitorHook, LoggingHook

# KB_EV is an internal helper used here only to set MB velocities.
from nvalchemi.dynamics.hooks._utils import KB_EV
from nvalchemi.dynamics.integrators.npt import NPT
from nvalchemi.dynamics.integrators.nve import NVE
from nvalchemi.dynamics.integrators.nvt_langevin import NVTLangevin
from nvalchemi.models.uma import UMAWrapper

logging.basicConfig(level=logging.INFO)

# 1 GPa in nvalchemi's internal pressure unit (eV/Å³), matching the eV / Å
# energy / length convention used for stress.
_GPA_TO_EV_PER_A3 = 6.241509074e-3

# %%
# Configuration
# -------------
# Edit these constants to explore different checkpoints, tasks, systems,
# and ensembles.
#
# * ``CHECKPOINT`` — ``uma-s-1p1`` / ``uma-s-1p2`` (small) or ``uma-m-1p1``
#   (medium).  See ``fairchem.core.calculate.pretrained_mlip.available_models``.
# * ``TASK`` — which task head to expose: ``omat`` (crystals), ``omol``
#   (molecules), ``oc20``, ``odac``, ``omc``.
# * ``INFERENCE_SETTINGS`` — ``"default"`` or ``"turbo"``.  ``"turbo"``
#   enables fairchem's ``torch.compile`` + TF32 + MoLE merge; it is faster
#   after a one-time compilation but assumes a **fixed atomic composition**
#   for the whole run (true for MD) and shifts numerics slightly.
# * ``N_REPLICAS`` — how many copies of the system to evolve **in one batch**.
#   Replicas start from slightly perturbed positions + independent velocities,
#   so their trajectories diverge while sharing a single batched forward pass.

CHECKPOINT = "uma-s-1p1"
TASK = "omat"  # omat | omol | oc20 | odac | omc
SYSTEM = "bcc-fe"  # bcc-fe | diamond | propane
ENSEMBLE = "nve"  # nve | nvt | npt
INFERENCE_SETTINGS = "turbo"  # "default" | "turbo"
N_REPLICAS = 4
N_STEPS = 300
DT_FS = 0.5
TEMPERATURE_K = 300.0
FRICTION = 0.01  # 1/fs — NVT Langevin only
PRESSURE_GPA = 0.0  # target pressure — NPT only
BAROSTAT_TIME_FS = 1000.0  # τ_P cell-coupling time — NPT only
THERMOSTAT_TIME_FS = 100.0  # τ_T — NVT Nosé-Hoover (inside NPT) only
SEED = 42
LOG_EVERY = 20
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Molecular systems need the OMol head — force it if a periodic task was left set.
if SYSTEM == "propane" and TASK != "omol":
    print(f"[info] forcing TASK='omol' for SYSTEM={SYSTEM!r}")
    TASK = "omol"

# NPT couples the cell to the stress tensor, so it needs a periodic system and
# a task head that produces stress (the periodic tasks, e.g. ``omat``).
if ENSEMBLE == "npt":
    if SYSTEM == "propane":
        print("[info] NPT needs a periodic system — switching SYSTEM='bcc-fe'")
        SYSTEM = "bcc-fe"
    if TASK == "omol":
        print("[info] NPT needs a stress-producing task — switching TASK='omat'")
        TASK = "omat"

# %%
# Build the batch
# ---------------
# Each replica is the same system with Maxwell-Boltzmann velocities at
# ``TEMPERATURE_K`` (zero net momentum). Replica 0 is the unperturbed
# reference; the rest get a small position jitter and an independent velocity
# seed, so the batched trajectories diverge. ``Batch.from_data_list`` packs
# them into one graph batch that shares a single model forward each step.

_PROPANE_POSITIONS = np.array(
    [
        [0.0000, 0.0000, 0.0000],
        [1.5260, 0.0000, 0.0000],
        [2.0330, 1.4360, 0.0000],
        [-0.5093, 1.0222, 0.0000],
        [-0.5093, -0.5111, 0.8853],
        [-0.5093, -0.5111, -0.8853],
        [2.0319, -0.5111, 0.8853],
        [2.0319, -0.5111, -0.8853],
        [3.1193, 1.4360, 0.0000],
        [1.6763, 1.9471, 0.8853],
        [1.6763, 1.9471, -0.8853],
    ]
)
_PROPANE_NUMBERS = [6, 6, 6, 1, 1, 1, 1, 1, 1, 1, 1]


def build_ase_atoms(system: str) -> Atoms:
    """Return an ASE ``Atoms`` object for the requested benchmark system."""
    if system == "bcc-fe":
        return bulk("Fe", "bcc", a=2.87, cubic=True) * (2, 2, 2)
    if system == "diamond":
        return bulk("C", "diamond", a=3.567, cubic=True) * (2, 2, 2)
    if system == "propane":
        atoms = Atoms(numbers=_PROPANE_NUMBERS, positions=_PROPANE_POSITIONS, pbc=False)
        atoms.info["charge"] = 0
        atoms.info["spin"] = 1
        return atoms
    raise ValueError(f"Unknown system {system!r}")


def replica_data(
    atoms: Atoms, temperature_k: float, device: torch.device, seed: int, jitter: float
) -> AtomicData:
    """One replica: positions optionally jittered, fresh MB velocities."""
    n = len(atoms)
    numbers_np = np.asarray(atoms.get_atomic_numbers())
    numbers = torch.as_tensor(numbers_np, dtype=torch.long, device=device)
    masses = torch.as_tensor(
        ASE_ATOMIC_MASSES[numbers_np], dtype=torch.float32, device=device
    )

    g = torch.Generator(device="cpu").manual_seed(seed)
    pos = torch.as_tensor(np.asarray(atoms.positions), dtype=torch.float32)
    if jitter > 0.0:
        pos = pos + jitter * torch.randn(pos.shape, generator=g)
    pos = pos.to(device)

    # Maxwell-Boltzmann velocities (sigma = sqrt(kB T / m)), zero net momentum.
    sigma = torch.sqrt(torch.as_tensor(KB_EV * temperature_k) / masses.cpu())
    vel = (torch.randn(n, 3, generator=g) * sigma.unsqueeze(-1)).to(device)
    vel -= vel.mean(dim=0, keepdim=True)

    kwargs: dict[str, Any] = {
        "positions": pos,
        "atomic_numbers": numbers,
        "atomic_masses": masses,
        "velocities": vel,
        "forces": torch.zeros_like(pos),
        "energy": torch.zeros(1, 1, device=device, dtype=torch.float32),
    }
    if bool(np.any(atoms.pbc)):
        kwargs["cell"] = torch.as_tensor(
            np.asarray(atoms.cell.array), dtype=torch.float32, device=device
        ).unsqueeze(0)
        kwargs["pbc"] = torch.as_tensor(
            np.asarray(atoms.pbc), dtype=torch.bool, device=device
        ).reshape(1, 3)
        # Stress placeholder: the dynamics loop copy_()s model outputs into
        # existing batch fields, so NPT's barostat needs this slot up front.
        kwargs["stress"] = torch.zeros(1, 3, 3, device=device, dtype=torch.float32)

    return AtomicData(**kwargs)


atoms = build_ase_atoms(SYSTEM)
# Replica 0 is unperturbed; replicas 1..N-1 get a small position jitter so the
# batched trajectories diverge from independent starts.
batch = Batch.from_data_list(
    [
        replica_data(
            atoms, TEMPERATURE_K, DEVICE, seed=SEED + i, jitter=0.0 if i == 0 else 0.05
        )
        for i in range(N_REPLICAS)
    ]
)
n_per = batch.num_nodes // batch.num_graphs
print(
    f"UMA {ENSEMBLE.upper()} | system={SYSTEM} | task={TASK} | "
    f"checkpoint={CHECKPOINT} | inference_settings={INFERENCE_SETTINGS} | "
    f"device={DEVICE}"
)
print(
    f"Batch: {batch.num_graphs} replicas × {n_per} atoms = {batch.num_nodes} total, "
    f"pbc={bool(torch.any(batch.pbc)) if batch.pbc is not None else False}"
)

# %%
# Load the UMA model
# ------------------
# ``from_checkpoint`` resolves the registered name, downloads the checkpoint
# via HuggingFace Hub (gated — see the setup notes above), and pins the task
# head.  ``inference_settings="turbo"`` routes through fairchem's
# ``torch.compile`` path.  If the checkpoint can't be loaded we exit early with
# the setup hint, so the rest of the script needn't be guarded.

try:
    load_t0 = time.perf_counter()
    model = UMAWrapper.from_checkpoint(
        CHECKPOINT,
        task_name=TASK,
        device=str(DEVICE),
        inference_settings=INFERENCE_SETTINGS,
    )
except Exception as exc:  # noqa: BLE001
    sys.exit(
        f"Could not load UMA ({exc}). Install 'nvalchemi-toolkit[uma]', request "
        "access to the gated 'facebook/UMA' repo, and authenticate via "
        "'huggingface-cli login' or HF_TOKEN."
    )
print(
    f"Loaded UMAWrapper in {time.perf_counter() - load_t0:.1f}s "
    f"(cutoff={model.cutoff:.2f} Å)"
)

# %%
# Run with built-in observation + drift-monitor hooks
# ---------------------------------------------------
# Rather than hand-rolling observation, register two built-in hooks:
#
# * :class:`~nvalchemi.dynamics.hooks.LoggingHook` — logs per-step potential
#   energy, temperature, and max force; here a custom writer prints them.
# * :class:`~nvalchemi.dynamics.hooks.EnergyDriftMonitorHook` — for NVE, warns
#   when total-energy drift exceeds the per-atom-per-step budget.
#
# No neighbor-list hook is needed (UMA builds its own graph). NPT couples the
# cell to the model's **stress** tensor, exercising the periodic stress path.

if ENSEMBLE == "nve":
    integrator = NVE(model=model, dt=DT_FS)
elif ENSEMBLE == "nvt":
    integrator = NVTLangevin(
        model=model,
        dt=DT_FS,
        temperature=TEMPERATURE_K,
        friction=FRICTION,
        random_seed=SEED,
    )
else:  # npt — needs forces + stress from the model
    integrator = NPT(
        model=model,
        dt=DT_FS,
        temperature=TEMPERATURE_K,
        pressure=PRESSURE_GPA * _GPA_TO_EV_PER_A3,
        barostat_time=BAROSTAT_TIME_FS,
        thermostat_time=THERMOSTAT_TIME_FS,
        pressure_coupling="isotropic",
    )

# LoggingHook emits one row per replica (graph) at each logged step; the
# custom writer prints energy / temperature / fmax with the replica index.
header = f"{'step':>6} {'rep':>4} {'PE(eV)':>16} {'T(K)':>9} {'fmax(eV/Å)':>12}"
log_hook = LoggingHook(
    backend="custom",
    frequency=LOG_EVERY,
    writer_fn=lambda step, rows: print(
        "\n".join(
            f"{step:6d} {int(r['graph_idx']):4d} {r['energy']:16.6f} "
            f"{r['temperature']:9.2f} {r['fmax']:12.4f}"
            for r in rows
        )
    ),
)
integrator.register_hook(log_hook)

# Drift is only meaningful under NVE (NVT/NPT exchange energy by design).
if ENSEMBLE == "nve":
    integrator.register_hook(
        EnergyDriftMonitorHook(
            threshold=1e-4, metric="per_atom_per_step", action="warn"
        )
    )

# NPT evolves the cell; record per-replica volumes up front (always periodic).
vol_initial = (
    torch.det(batch.cell.reshape(-1, 3, 3)).abs() if ENSEMBLE == "npt" else None
)

print(f"\nRunning {N_STEPS} {ENSEMBLE.upper()} steps …")
print(header)
print("-" * len(header))
if DEVICE.type == "cuda":
    torch.cuda.synchronize(DEVICE)
t0 = time.perf_counter()
with log_hook:  # context manager flushes the async logging stream on exit
    batch = integrator.run(batch, n_steps=N_STEPS)
if DEVICE.type == "cuda":
    torch.cuda.synchronize(DEVICE)
wall_s = time.perf_counter() - t0

# %%
# Summary
# -------
print(
    f"\nwall time     : {wall_s:.2f} s ({wall_s * 1e3 / max(1, N_STEPS):.2f} ms/step)"
)
if ENSEMBLE == "nve":
    print(
        "drift         : monitored by EnergyDriftMonitorHook (warns above 1e-4 eV/atom/step)"
    )
elif ENSEMBLE == "nvt":
    print("drift         : not gated for NVT (thermostat absorbs drift)")
else:  # npt — per-replica cell response confirms the stress path ran
    vol_final = torch.det(batch.cell.reshape(-1, 3, 3)).abs()
    pct = (100.0 * (vol_final - vol_initial) / vol_initial).tolist()
    print(
        f"cell volume   : mean {vol_initial.mean():.3f} → {vol_final.mean():.3f} Å³ "
        f"@ {PRESSURE_GPA:.3f} GPa target"
    )
    print("per-replica Δ : " + "  ".join(f"{p:+.2f}%" for p in pct))
