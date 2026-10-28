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

"""Multi-GPU regression for distributed multipole electrostatics (PME and Ewald).

A 2-rank halo ``DistributedModel`` of each multipole wrapper is checked against
the single-process wrapper on total energy, per-atom forces and stress for each
multipole order, on a neutral and on a net-charged system (the latter exercises
the globally reduced total charge in the background correction).

Requires 2+ CUDA GPUs and an ``nvalchemiops`` whose multipole entry points
accept ``total_charge``, ``source_weights`` and ``reduce_partial``.
"""

from __future__ import annotations

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from _dd_harness import free_port
from _dd_harness import nccl_worker as _worker

from nvalchemi.data import AtomicData, Batch
from nvalchemi.distributed.config import DomainConfig

WORLD_SIZE = 2
BOX = 36.0
N_ATOMS = 64
CUTOFF = 7.0


def _cell(shear: float, device: torch.device | str = "cpu") -> torch.Tensor:
    """Lattice vectors as rows: cubic for ``shear=0``, else a lower-triangular shear."""
    cell = torch.eye(3, dtype=torch.float64, device=device)
    cell[1, 0] = shear
    cell[2, 0] = shear
    cell[2, 1] = shear
    return cell * BOX


def _system(l_max: int, net_charge: float, shear: float) -> dict[str, torch.Tensor]:
    """A random periodic system with moments up to ``l_max`` (CPU, float64)."""
    g = torch.Generator().manual_seed(11)
    dtype = torch.float64
    q = torch.randn(N_ATOMS, generator=g, dtype=dtype)
    q += (net_charge - q.sum()) / N_ATOMS
    out = {
        "positions": torch.rand(N_ATOMS, 3, generator=g, dtype=dtype) @ _cell(shear),
        "charges": q,
        "cell": _cell(shear),
    }
    if l_max >= 1:
        out["dipoles"] = 0.3 * torch.randn(N_ATOMS, 3, generator=g, dtype=dtype)
    if l_max >= 2:
        quad = 0.1 * torch.randn(N_ATOMS, 3, 3, generator=g, dtype=dtype)
        out["quadrupoles"] = 0.5 * (quad + quad.mT)
    return out


def _batch(system: dict[str, torch.Tensor], device: torch.device) -> Batch:
    data = AtomicData(
        atomic_numbers=torch.ones(N_ATOMS, dtype=torch.long, device=device),
        positions=system["positions"].to(device).clone(),
        atomic_masses=torch.ones(N_ATOMS, dtype=torch.float64, device=device),
        cell=system["cell"].to(device).unsqueeze(0),
        pbc=torch.ones(1, 3, dtype=torch.bool, device=device),
        charges=system["charges"].to(device),
        forces=torch.zeros(N_ATOMS, 3, dtype=torch.float64, device=device),
        energy=torch.zeros(1, 1, dtype=torch.float64, device=device),
    )
    for key in ("dipoles", "quadrupoles"):
        if key in system:
            data.add_node_property(key, system[key].to(device))
    return Batch.from_data_list([data])


def _wrapper(l_max: int, device: torch.device, method: str, neighbor_format: str):
    from nvalchemi.models.base import NeighborListFormat
    from nvalchemi.models.multipole import (
        MultipoleEwaldModelWrapper,
        MultipolePMEModelWrapper,
    )

    kwargs = {
        "cutoff": CUTOFF,
        "sigma": 0.8,
        "accuracy": 1e-5,
        "l_max": l_max,
        "neighbor_format": NeighborListFormat(neighbor_format),
    }
    if method == "pme":
        # Order 4 keeps the first-use Warp kernel build short; DD correctness
        # does not depend on the spline order.
        model = MultipolePMEModelWrapper(spline_order=4, **kwargs)
    else:
        model = MultipoleEwaldModelWrapper(**kwargs)
    model.to(device)
    model.model_config.active_outputs = {"energy", "forces", "stress"}
    return model


def _equivalence_worker(
    rank: int,
    world_size: int,
    l_max: int,
    net_charge: float,
    shear: float,
    method: str,
    neighbor_format: str,
    compile: bool,
) -> None:
    from torch.distributed import DeviceMesh

    from nvalchemi.distributed.distributed_model import DistributedModel
    from nvalchemi.distributed.partitioner import SpatialPartitioner
    from nvalchemi.distributed.sharded_batch import ShardedBatch
    from nvalchemi.neighbors import compute_neighbors

    device = torch.device(f"cuda:{rank}")
    system = _system(l_max, net_charge, shear)

    ref_model = _wrapper(l_max, device, method, neighbor_format)
    ref_batch = _batch(system, device)
    compute_neighbors(ref_batch, config=ref_model.model_config.neighbor_config)
    ref = ref_model(ref_batch)

    mesh = DeviceMesh("cuda", list(range(world_size)), mesh_dim_names=("domain",))
    config = DomainConfig(
        cutoff=CUTOFF, skin=0.0, mesh=mesh, require_nondegenerate=True
    )
    full = _batch(system, device) if rank == 0 else None
    sharded = ShardedBatch.from_batch(batch=full, mesh=mesh, config=config, src=0)
    with DistributedModel(
        _wrapper(l_max, device, method, neighbor_format), config, compile=compile
    ) as model:
        out = model(sharded)
        if compile:
            # A second call on the same geometry must reuse the compiled graphs.
            from torch._dynamo.utils import counters

            graphs = counters["stats"]["unique_graphs"]
            assert graphs > 0, "the compiled path did not compile anything"
            out = model(sharded)
            assert counters["stats"]["unique_graphs"] == graphs, (
                "the second call recompiled"
            )

    partitioner = SpatialPartitioner(
        config=config,
        cell_matrix=system["cell"].to(device).unsqueeze(0),
        pbc=torch.ones(1, 3, dtype=torch.bool, device=device),
    )
    owned = partitioner.assign_atoms_to_ranks(system["positions"].to(device)) == rank

    torch.testing.assert_close(
        out["energy"].sum(), ref["energy"].sum(), rtol=1e-9, atol=1e-9
    )
    torch.testing.assert_close(
        out["forces"], ref["forces"][owned], rtol=1e-8, atol=1e-9
    )
    torch.testing.assert_close(
        out["stress"].reshape(-1, 3, 3),
        ref["stress"].reshape(-1, 3, 3),
        rtol=1e-8,
        atol=1e-12,
    )
    dist.barrier()


@pytest.mark.multigpu
@pytest.mark.parametrize("compile", [False, True], ids=["eager", "compiled"])
@pytest.mark.parametrize(
    ("method", "net_charge", "shear", "neighbor_format"),
    [
        ("pme", 0.0, 0.0, "matrix"),
        ("pme", 1.5, 0.0, "matrix"),
        ("pme", 1.5, 0.3, "matrix"),
        ("pme", 1.5, 0.0, "coo"),
        ("pme", 1.5, 0.3, "coo"),
        ("ewald", 1.5, 0.0, "matrix"),
        ("ewald", 1.5, 0.3, "coo"),
    ],
    ids=[
        "pme-neutral-cubic",
        "pme-charged-cubic",
        "pme-charged-triclinic",
        "pme-charged-cubic-coo",
        "pme-charged-triclinic-coo",
        "ewald-charged-cubic",
        "ewald-charged-triclinic-coo",
    ],
)
@pytest.mark.parametrize("l_max", [0, 1, 2])
def test_multipole_dist_model_equivalence_2ranks(
    l_max, method, net_charge, shear, neighbor_format, compile
):
    """Halo DD multipole electrostatics match single-process energy, forces and stress."""
    pytest.importorskip("nvalchemiops", reason="nvalchemiops not installed")
    from nvalchemi.models.multipole import (
        _ops_accept_partial_sources,
        _ops_accept_total_charge,
    )

    if not (_ops_accept_total_charge() and _ops_accept_partial_sources()):
        pytest.skip(
            "installed nvalchemiops lacks total_charge / source_weights / "
            "reduce_partial on the multipole entry points"
        )

    mp.spawn(
        _worker,
        args=(
            WORLD_SIZE,
            free_port(),
            _equivalence_worker,
            l_max,
            net_charge,
            shear,
            method,
            neighbor_format,
            compile,
        ),
        nprocs=WORLD_SIZE,
    )
