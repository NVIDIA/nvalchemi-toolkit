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
"""Composable model dynamics with real systems and the current models API.

This example mirrors the simplified composition UX, but uses real model stacks
inside molecular dynamics runs.

It shows two practical scenarios:

1. A periodic 24-atom NaCl crystal with ``MACEWrapper + DFTD3ModelWrapper``.
2. A graphene nanoribbon in a padded periodic box with
   ``AIMNet2Wrapper + EwaldModelWrapper``.

Each section:

* builds the structure with ASE,
* constructs the composed model with canonical interfaces and no explicit wire,
* prints ``repr(calc)`` and points out the important steps,
* runs 1 ps of NVT Langevin dynamics,
* reports a compact MD summary including the instantaneous temperature.

The graphene section uses Ewald rather than DSF.  With a finite cutoff,
``DSFModelWrapper(alpha=0)`` does not recover full unscreened long-range
Coulomb; Ewald is the clearer match for the intended electrostatics example.
"""

from __future__ import annotations

import torch
from ase import Atoms
from ase.build import bulk, graphene_nanoribbon

from nvalchemi.data import AtomicData, Batch
from nvalchemi.dynamics.hooks._utils import KB_EV, kinetic_energy_per_graph

TARGET_TEMPERATURE = 300.0
TIMESTEP_FS = 1.0
N_STEPS = 1000
FRICTION = 0.5
RANDOM_SEED = 7


def _print_section(title: str) -> None:
    """Print one section header."""

    print()
    print(title)
    print("=" * len(title))


def _print_snippet(snippet: str) -> None:
    """Print one short construction snippet."""

    print("constructed as:")
    for line in snippet.strip().splitlines():
        print(f"  {line}")


def _print_repr_notes(lines: tuple[str, ...]) -> None:
    """Print concise guidance for reading one composed-model repr."""

    print()
    print("what to look for:")
    for line in lines:
        print(f"  - {line}")


def _velocities_from_masses(
    masses: torch.Tensor,
    *,
    temperature: float,
    seed: int,
) -> torch.Tensor:
    """Sample Maxwell-Boltzmann velocities for one system.

    Parameters
    ----------
    masses
        Per-atom masses in atomic mass units.
    temperature
        Target temperature in Kelvin.
    seed
        Random seed used for reproducible initialization.

    Returns
    -------
    torch.Tensor
        Velocity tensor with shape ``(N, 3)``.
    """

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    std = (KB_EV * temperature / masses).sqrt().unsqueeze(-1)
    return (std * torch.randn((masses.shape[0], 3), generator=generator)).float()


def _batch_from_ase_atoms(
    atoms: Atoms,
    *,
    temperature: float,
    seed: int,
) -> Batch:
    """Convert one ASE ``Atoms`` object into a dynamics-ready batch.

    Parameters
    ----------
    atoms
        ASE structure to convert.
    temperature
        Target temperature used for the initial velocity draw.
    seed
        Random seed for the velocity draw.

    Returns
    -------
    Batch
        Single-graph batch with positions, velocities, cell, and PBC data.
    """

    positions = torch.tensor(atoms.get_positions(), dtype=torch.float32)
    atomic_numbers = torch.tensor(atoms.get_atomic_numbers(), dtype=torch.long)
    masses = torch.tensor(atoms.get_masses(), dtype=torch.float32)
    velocities = _velocities_from_masses(masses, temperature=temperature, seed=seed)
    cell = torch.tensor(atoms.cell.array, dtype=torch.float32).unsqueeze(0)
    pbc = torch.tensor(atoms.get_pbc(), dtype=torch.bool).unsqueeze(0)
    data = AtomicData(
        positions=positions,
        atomic_numbers=atomic_numbers,
        atomic_masses=masses,
        velocities=velocities,
        forces=torch.zeros_like(positions),
        energies=torch.zeros(1, 1, dtype=torch.float32),
        cell=cell,
        pbc=pbc,
    )
    return Batch.from_data_list([data])


def _make_nacl_batch() -> Batch:
    """Build a 24-atom periodic NaCl supercell for the MACE + D3 run."""

    atoms = bulk("NaCl", "rocksalt", a=5.64, cubic=False).repeat((2, 2, 3))
    return _batch_from_ase_atoms(
        atoms,
        temperature=TARGET_TEMPERATURE,
        seed=RANDOM_SEED,
    )


def _make_periodic_gnr_batch() -> Batch:
    """Build a padded fully periodic graphene nanoribbon batch.

    Returns
    -------
    Batch
        Single-graph periodic batch for AIMNet2 + Ewald dynamics.
    """

    atoms = graphene_nanoribbon(
        2,
        6,
        type="zigzag",
        saturated=True,
        C_H=1.1,
        C_C=1.4,
        vacuum=3.0,
        magnetic=False,
    )
    positions = torch.tensor(atoms.get_positions(), dtype=torch.float32)
    mins = positions.min(dim=0).values
    maxs = positions.max(dim=0).values
    extents = maxs - mins
    original_lengths = torch.tensor(atoms.cell.lengths(), dtype=torch.float32)
    padded_lengths = torch.tensor(
        [
            float(extents[0].item() + 15.0),
            float(extents[1].item() + 15.0),
            float(max(original_lengths[2].item(), extents[2].item() + 5.0)),
        ],
        dtype=torch.float32,
    )
    centered_positions = positions - mins + 0.5 * (padded_lengths - extents)
    padded = atoms.copy()
    padded.set_positions(centered_positions.cpu().numpy())
    padded.set_cell(torch.diag(padded_lengths).cpu().numpy())
    padded.set_pbc([True, True, True])
    return _batch_from_ase_atoms(
        padded,
        temperature=TARGET_TEMPERATURE,
        seed=RANDOM_SEED + 1,
    )


def _instantaneous_temperature(batch: Batch) -> torch.Tensor:
    """Compute one instantaneous temperature per graph.

    Parameters
    ----------
    batch
        Batched dynamics state after one run.

    Returns
    -------
    torch.Tensor
        Temperature per graph in Kelvin with shape ``(B,)``.
    """

    kinetic_energy = kinetic_energy_per_graph(
        velocities=batch.velocities,
        masses=batch.atomic_masses,
        batch_idx=batch.batch,
        num_graphs=batch.num_graphs,
    ).squeeze(-1)
    n_atoms = batch.num_nodes_per_graph.float()
    return (2.0 * kinetic_energy) / (3.0 * n_atoms * KB_EV)


def _print_md_summary(batch: Batch) -> None:
    """Print a compact dynamics summary after one completed run."""

    temperature = _instantaneous_temperature(batch).mean().item()
    cell = batch.cell[0] if batch.cell.ndim == 3 else batch.cell
    lengths = torch.linalg.norm(cell, dim=-1)
    print("run summary:")
    print(f"  - positions: shape={tuple(batch.positions.shape)}")
    print(f"  - velocities: shape={tuple(batch.velocities.shape)}")
    print(f"  - forces: shape={tuple(batch.forces.shape)}")
    print(f"  - energies: shape={tuple(batch.energies.shape)}")
    print(f"  - inst. temp.: {temperature:.1f} K (target = {TARGET_TEMPERATURE:.1f} K)")
    print(
        "  - cell lengths: "
        f"{lengths[0].item():.3f}, {lengths[1].item():.3f}, {lengths[2].item():.3f} A"
    )


def _run_dynamics(
    calc,
    batch: Batch,
):
    """Run one NVT Langevin trajectory with the standard example settings."""

    from nvalchemi.dynamics import NVTLangevin
    from nvalchemi.dynamics.hooks import NeighborListHook, WrapPeriodicHook

    dynamics = NVTLangevin(
        model=calc,
        dt=TIMESTEP_FS,
        temperature=TARGET_TEMPERATURE,
        friction=FRICTION,
        n_steps=N_STEPS,
        random_seed=RANDOM_SEED,
    )
    if calc.spec.neighbor_config.source == "external":
        dynamics.register_hook(NeighborListHook(calc.spec.neighbor_config))
    dynamics.register_hook(WrapPeriodicHook())
    result = dynamics.run(batch)
    return dynamics, result


def _run_nacl_example() -> None:
    """Run the NaCl crystal dynamics example."""

    title = "NaCl crystal: MACE + DFT-D3"
    snippet = """
atoms = bulk("NaCl", "rocksalt", a=5.64, cubic=False).repeat((2, 2, 3))
batch = _batch_from_ase_atoms(atoms, temperature=300.0, seed=7)
mace = MACEWrapper(
    "medium-0b2",
    device=device,
    dtype=torch.float32,
    enable_cueq=False,
)
d3 = DFTD3ModelWrapper(functional="pbe")
calc = mace + d3
"""
    _print_section(title)
    _print_snippet(snippet)

    try:
        from nvalchemi.models.dftd3 import DFTD3ModelWrapper
        from nvalchemi.models.mace import MACEWrapper
    except Exception as exc:
        print()
        print(f"status: skipped ({type(exc).__name__}: {exc})")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch = _make_nacl_batch().to(device)
    try:
        calc = MACEWrapper(
            "medium-0b2",
            device=device,
            dtype=torch.float32,
            enable_cueq=True,
            compile_model=True,
        ) + DFTD3ModelWrapper(functional="pbe")
    except ImportError as exc:
        print()
        print(f"status: skipped ({type(exc).__name__}: {exc})")
        return

    calc = calc.to(device)

    print()
    print("repr(calc):")
    print(calc)
    _print_repr_notes(
        (
            "The repr is an effective pipeline summary with zero-based steps.",
            "The external neighbor builders are synthesized from the MACE and D3 cutoffs.",
            "The derivative step appears before the trailing direct D3 correction.",
        )
    )
    print()
    print(f"running {N_STEPS} NVT steps ({TIMESTEP_FS * N_STEPS / 1000.0:.1f} ps) ...")

    dynamics, batch = _run_dynamics(calc, batch)
    print(f"completed steps: {dynamics.step_count}")
    _print_md_summary(batch)


def _run_graphene_example() -> None:
    """Run the graphene nanoribbon dynamics example."""

    title = "Graphene nanoribbon: AIMNet2 + Ewald"
    snippet = """
atoms = graphene_nanoribbon(
    2,
    6,
    type="zigzag",
    saturated=True,
    C_H=1.1,
    C_C=1.4,
    vacuum=3.0,
    magnetic=False,
)
batch = _make_periodic_gnr_batch()
aimnet = AIMNet2Wrapper("aimnet2", device=device, compile_model=False)
ewald = EwaldModelWrapper(cutoff=10.0, accuracy=1e-6)
calc = aimnet + ewald
"""
    _print_section(title)
    _print_snippet(snippet)

    try:
        from nvalchemi.models.aimnet2 import AIMNet2Wrapper
        from nvalchemi.models.ewald import EwaldModelWrapper
    except Exception as exc:
        print()
        print(f"status: skipped ({type(exc).__name__}: {exc})")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch = _make_periodic_gnr_batch().to(device)
    try:
        calc = AIMNet2Wrapper(
            "aimnet2",
            device=device,
            compile_model=True,
        ) + EwaldModelWrapper(
            cutoff=10.0,
            accuracy=1e-6,
        )
    except ImportError as exc:
        print()
        print(f"status: skipped ({type(exc).__name__}: {exc})")
        return

    calc = calc.to(device)

    print()
    print("repr(calc):")
    print(calc)
    _print_repr_notes(
        (
            "AIMNet2 keeps its own internal neighbor handling, so no top-level builder is shown for it.",
            "The Ewald real-space neighbor builder appears explicitly because it is an external requirement.",
            "The derivative step covers the AIMNet2 and Ewald energy sum before forces are exported.",
        )
    )
    print()
    print(f"running {N_STEPS} NVT steps ({TIMESTEP_FS * N_STEPS / 1000.0:.1f} ps) ...")

    dynamics, batch = _run_dynamics(calc, batch)
    print(f"completed steps: {dynamics.step_count}")
    _print_md_summary(batch)


def main() -> None:
    """Run the composed-model dynamics walkthrough."""

    torch.manual_seed(RANDOM_SEED)
    _run_nacl_example()
    _run_graphene_example()


if __name__ == "__main__":
    main()
