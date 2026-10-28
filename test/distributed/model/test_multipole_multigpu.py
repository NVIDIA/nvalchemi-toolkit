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

"""Multi-GPU regression for distributed multipole PME electrostatics.

A 2-rank halo ``DistributedModel(MultipolePMEModelWrapper)`` is checked against
the single-process wrapper on total energy, per-atom forces and stress for each
multipole order, on a neutral and on a net-charged system (the latter exercises
the globally reduced total charge in the background correction).

Requires 2+ CUDA GPUs and an ``nvalchemiops`` whose multipole entry points
accept ``total_charge``.
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


def _system(l_max: int, net_charge: float) -> dict[str, torch.Tensor]:
    """A random periodic system with moments up to ``l_max`` (CPU, float64)."""
    g = torch.Generator().manual_seed(11)
    dtype = torch.float64
    q = torch.randn(N_ATOMS, generator=g, dtype=dtype)
    q += (net_charge - q.sum()) / N_ATOMS
    out = {
        "positions": torch.rand(N_ATOMS, 3, generator=g, dtype=dtype) * BOX,
        "charges": q,
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
        cell=(torch.eye(3, dtype=torch.float64, device=device) * BOX).unsqueeze(0),
        pbc=torch.ones(1, 3, dtype=torch.bool, device=device),
        charges=system["charges"].to(device),
        forces=torch.zeros(N_ATOMS, 3, dtype=torch.float64, device=device),
        energy=torch.zeros(1, 1, dtype=torch.float64, device=device),
    )
    for key in ("dipoles", "quadrupoles"):
        if key in system:
            data.add_node_property(key, system[key].to(device))
    return Batch.from_data_list([data])


def _wrapper(l_max: int, device: torch.device):
    from nvalchemi.models.multipole import MultipolePMEModelWrapper

    # Order 4 keeps the first-use Warp kernel build short; DD correctness does
    # not depend on the spline order.
    model = MultipolePMEModelWrapper(
        cutoff=CUTOFF, sigma=0.8, accuracy=1e-5, l_max=l_max, spline_order=4
    ).to(device)
    model.model_config.active_outputs = {"energy", "forces", "stress"}
    return model


def _equivalence_worker(
    rank: int, world_size: int, l_max: int, net_charge: float, compile: bool = False
) -> None:
    from torch.distributed import DeviceMesh

    from nvalchemi.distributed.distributed_model import DistributedModel
    from nvalchemi.distributed.partitioner import SpatialPartitioner
    from nvalchemi.distributed.sharded_batch import ShardedBatch
    from nvalchemi.neighbors import compute_neighbors

    device = torch.device(f"cuda:{rank}")
    system = _system(l_max, net_charge)

    ref_model = _wrapper(l_max, device)
    ref_batch = _batch(system, device)
    compute_neighbors(ref_batch, config=ref_model.model_config.neighbor_config)
    ref = ref_model(ref_batch)

    mesh = DeviceMesh("cuda", list(range(world_size)), mesh_dim_names=("domain",))
    config = DomainConfig(
        cutoff=CUTOFF, skin=0.0, mesh=mesh, require_nondegenerate=True
    )
    full = _batch(system, device) if rank == 0 else None
    sharded = ShardedBatch.from_batch(batch=full, mesh=mesh, config=config, src=0)
    with DistributedModel(_wrapper(l_max, device), config, compile=compile) as model:
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
        cell_matrix=(torch.eye(3, dtype=torch.float64, device=device) * BOX).unsqueeze(
            0
        ),
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
@pytest.mark.parametrize("net_charge", [0.0, 1.5])
@pytest.mark.parametrize("l_max", [0, 1, 2])
def test_multipole_pme_dist_model_equivalence_2ranks(l_max, net_charge, compile):
    """Halo DD multipole PME matches single-process energy, forces and stress."""
    pytest.importorskip("nvalchemiops", reason="nvalchemiops not installed")
    from nvalchemi.models.multipole import _ops_accept_total_charge

    if not _ops_accept_total_charge():
        pytest.skip(
            "installed nvalchemiops lacks total_charge on multipole entry points"
        )

    mp.spawn(
        _worker,
        args=(WORLD_SIZE, free_port(), _equivalence_worker, l_max, net_charge, compile),
        nprocs=WORLD_SIZE,
    )


def test_multipole_ewald_distribution_spec_unsupported():
    """The direct-k Ewald wrapper declares no decomposition and says so."""
    from nvalchemi.models.multipole import MultipoleEwaldModelWrapper

    model = MultipoleEwaldModelWrapper(cutoff=CUTOFF, sigma=0.8, accuracy=1e-5)
    with pytest.raises(NotImplementedError, match="MultipolePMEModelWrapper"):
        model.distribution_spec()
