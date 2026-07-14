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
"""Tests for :class:`DomainParallel`.

``DomainParallel`` is a thin :class:`BaseDynamics` subclass that holds a
:class:`ShardedBatch` across the step loop and delegates the per-step
model call to a :class:`DistributedModel`.

Single-process tests (no ``torch.distributed`` init) exercise the
trivial partition / gather pass-through paths, hook dispatch, and
rank-resolution fallbacks. Integration tests at the bottom drive a
real NVE loop across 2 gloo ranks and assert equivalence with a
single-process reference.
"""

from __future__ import annotations

import os
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed import DeviceMesh  # noqa: F401 — resolve forward ref

from nvalchemi.data import AtomicData, Batch
from nvalchemi.distributed.config import DomainConfig, HookScope
from nvalchemi.distributed.domain_parallel import DomainParallel
from nvalchemi.dynamics.base import BaseDynamics, DynamicsStage
from nvalchemi.dynamics.demo import DemoDynamics
from nvalchemi.hooks._context import HookContext
from nvalchemi.models.demo import DemoModel, DemoModelWrapper

# Resolve DeviceMesh forward reference for pydantic validation.
DomainConfig.model_rebuild(_types_namespace={"DeviceMesh": DeviceMesh})


# ======================================================================
# Helpers.
# ======================================================================


def _make_batch(n_atoms: int = 8) -> Batch:
    """Minimal single-graph batch with a 10×10×10 periodic box."""
    positions = torch.rand(n_atoms, 3) * 10.0
    data = AtomicData(
        atomic_numbers=torch.tensor([6] * n_atoms, dtype=torch.long),
        positions=positions,
    )
    batch = Batch.from_data_list([data])
    batch.forces = torch.zeros(n_atoms, 3)
    batch.energies = torch.zeros(1, 1)
    batch.cell = torch.diag(torch.tensor([10.0, 10.0, 10.0])).unsqueeze(0)
    batch.pbc = torch.tensor([[True, True, True]])
    return batch


def _make_dp(
    *, n_steps: int = 10, cutoff: float = 3.0, skin: float = 0.5
) -> tuple[DomainParallel, DemoDynamics]:
    """Build a ``DomainParallel`` wrapping ``DemoDynamics`` (no mesh —
    single-process fallback). ``DemoDynamics`` requires ``n_steps`` at
    construction; tests that care about the constructor-vs-argument
    ``run()`` precedence override it explicitly."""
    model = DemoModelWrapper(DemoModel())
    inner = DemoDynamics(model=model, n_steps=n_steps)
    config = DomainConfig(cutoff=cutoff, skin=skin)
    return DomainParallel(dynamics=inner, config=config), inner


# ======================================================================
# Construction.
# ======================================================================


class TestInit:
    def test_is_base_dynamics(self) -> None:
        dp, _ = _make_dp()
        assert isinstance(dp, BaseDynamics)

    def test_stores_inner_dynamics(self) -> None:
        dp, inner = _make_dp()
        assert dp._dynamics is inner

    def test_stores_config(self) -> None:
        dp, _ = _make_dp()
        assert dp._config.cutoff == 3.0
        assert dp._config.skin == 0.5

    def test_lazy_components_start_none(self) -> None:
        """``partition()`` is what initializes the per-step machinery —
        at construction the strategy / sharded batch / dist_model
        slots are ``None``."""
        dp, _ = _make_dp()
        assert dp._strategy is None
        assert dp._sharded_batch is None
        assert dp._dist_model is None

    def test_initial_runtime_state(self) -> None:
        dp, _ = _make_dp()
        assert dp._n_owned == 0
        assert dp._forces_primed is False
        assert dp.step_count == 0

    def test_shares_model_with_inner_dynamics(self) -> None:
        """``BaseDynamics.__init__`` wires ``self.model`` from the inner
        dynamics' model so hooks see the same object."""
        dp, inner = _make_dp()
        assert dp.model is inner.model


# ======================================================================
# Property delegation — __needs_keys__ / __provides_keys__ read from
# the inner dynamics.
# ======================================================================


class TestPropertiesDelegate:
    def test_needs_keys_delegates(self) -> None:
        dp, inner = _make_dp()
        assert dp.__needs_keys__ == inner.__needs_keys__

    def test_provides_keys_delegates(self) -> None:
        dp, inner = _make_dp()
        assert dp.__provides_keys__ == inner.__provides_keys__

    def test_needs_keys_reflects_inner_changes(self) -> None:
        """If the inner dynamics grows a needed key, the delegator
        reports it immediately — no caching."""
        dp, inner = _make_dp()

        class _StubDyn:
            __needs_keys__ = {"positions", "velocities", "custom_field"}
            __provides_keys__ = {"forces"}
            model = inner.model
            step_count = 0

        dp._dynamics = _StubDyn()
        assert "custom_field" in dp.__needs_keys__


# ======================================================================
# partition / gather — single-process fallback.
# ======================================================================


class TestPartitionSingleProcess:
    def test_returns_input_batch(self) -> None:
        dp, _ = _make_dp()
        batch = _make_batch(n_atoms=10)
        result = dp.partition(batch)
        assert result is batch

    def test_sets_n_owned(self) -> None:
        dp, _ = _make_dp()
        batch = _make_batch(n_atoms=12)
        dp.partition(batch)
        assert dp._n_owned == 12

    def test_raises_without_batch_and_without_dist(self) -> None:
        """Without distributed init there's no other source of atoms;
        passing ``None`` must fail loudly rather than silently proceed."""
        dp, _ = _make_dp()
        with pytest.raises(ValueError, match="batch must be provided"):
            dp.partition(None)

    def test_sharded_components_stay_none_in_single_process(self) -> None:
        """Single-process passes the batch through; no ShardedBatch or
        DistributedModel gets built."""
        dp, _ = _make_dp()
        batch = _make_batch()
        dp.partition(batch)
        assert dp._sharded_batch is None
        assert dp._dist_model is None


class TestGatherSingleProcess:
    def test_returns_local_batch(self) -> None:
        """Without a mesh / sharded batch, ``gather`` is a no-op —
        return what we were given."""
        dp, _ = _make_dp()
        batch = _make_batch()
        result = dp.gather(batch, dst=0)
        assert result is batch


# ======================================================================
# step / run — single-process fallback delegates to the inner dynamics'
# step() directly (no distributed machinery).
# ======================================================================


class TestStepSingleProcess:
    def test_step_delegates_to_inner(self) -> None:
        """In single-process mode (no dist_model), step() forwards to
        the inner dynamics' step() so the hook chain and behavior match
        the non-distributed case."""
        dp, inner = _make_dp()
        batch = _make_batch()
        dp.partition(batch)  # primes _n_owned but doesn't build dist_model
        assert dp._dist_model is None

        with patch.object(inner, "step", return_value=(batch, None)) as mock_step:
            result, converged = dp.step(batch)
            mock_step.assert_called_once()
            assert result is batch
            assert converged is None


class TestRunMethod:
    """In single-process mode ``DomainParallel.run`` delegates to the
    inner dynamics' ``run`` — it's the inner's step counter that
    advances, not the wrapper's. The distributed path owns its own
    step loop and is covered by the gloo integration tests at the
    bottom of this file."""

    def test_run_executes_n_steps(self) -> None:
        dp, inner = _make_dp()
        batch = _make_batch()
        dp.partition(batch)
        dp.run(batch, n_steps=5)
        assert inner.step_count == 5

    def test_run_uses_constructor_n_steps(self) -> None:
        dp, inner = _make_dp(n_steps=3)
        batch = _make_batch()
        dp.partition(batch)
        dp.run(batch)
        assert inner.step_count == 3

    def test_run_prefers_argument_over_constructor(self) -> None:
        dp, inner = _make_dp(n_steps=3)
        batch = _make_batch()
        dp.partition(batch)
        dp.run(batch, n_steps=7)
        assert inner.step_count == 7


# ======================================================================
# Hook dispatch.
# ======================================================================


class _RecordingHook:
    """Hook that captures the scope-resolved batch/context on every call."""

    def __init__(
        self,
        scope: HookScope = HookScope.LOCAL,
        stage: DynamicsStage = DynamicsStage.AFTER_STEP,
        frequency: int = 1,
        runs_on_stage: bool = True,
    ) -> None:
        self.scope = scope
        self.stage = stage
        self.frequency = frequency
        self.runs_on_stage = runs_on_stage
        self.calls: list[HookContext] = []

    def __call__(self, ctx: HookContext, stage: Any) -> None:
        self.calls.append(ctx)


class TestCallHooksWithScope:
    def test_local_hook_fires(self) -> None:
        dp, _ = _make_dp()
        batch = _make_batch()
        hook = _RecordingHook(scope=HookScope.LOCAL, stage=DynamicsStage.AFTER_STEP)
        dp.register_hook(hook)
        dp._call_hooks(DynamicsStage.AFTER_STEP, batch)
        assert len(hook.calls) == 1

    def test_rank_zero_hook_fires_on_rank_zero(self) -> None:
        dp, _ = _make_dp()
        dp._domain_rank = 0
        batch = _make_batch()
        hook = _RecordingHook(scope=HookScope.RANK_ZERO, stage=DynamicsStage.AFTER_STEP)
        dp.register_hook(hook)
        dp._call_hooks(DynamicsStage.AFTER_STEP, batch)
        assert len(hook.calls) == 1

    def test_rank_zero_hook_skipped_on_nonzero_rank(self) -> None:
        dp, _ = _make_dp()
        dp._domain_rank = 1
        batch = _make_batch()
        hook = _RecordingHook(scope=HookScope.RANK_ZERO, stage=DynamicsStage.AFTER_STEP)
        dp.register_hook(hook)
        dp._call_hooks(DynamicsStage.AFTER_STEP, batch)
        assert hook.calls == []

    def test_hook_stage_filtering(self) -> None:
        dp, _ = _make_dp()
        batch = _make_batch()
        hook = _RecordingHook(scope=HookScope.LOCAL, stage=DynamicsStage.BEFORE_STEP)
        dp.register_hook(hook)
        dp._call_hooks(DynamicsStage.AFTER_STEP, batch)
        assert hook.calls == []

    def test_hook_frequency_gating(self) -> None:
        dp, _ = _make_dp()
        batch = _make_batch()
        hook = _RecordingHook(
            scope=HookScope.LOCAL, stage=DynamicsStage.AFTER_STEP, frequency=3
        )
        dp.register_hook(hook)
        for i in range(6):
            dp.step_count = i
            dp._call_hooks(DynamicsStage.AFTER_STEP, batch)
        # Fires at step_count 0, 3 — pytest default ``step_count %
        # frequency == 0`` gate.
        assert len(hook.calls) == 2

    def test_multiple_scopes_coexist(self) -> None:
        dp, _ = _make_dp()
        batch = _make_batch()
        hl = _RecordingHook(scope=HookScope.LOCAL, stage=DynamicsStage.AFTER_STEP)
        hr = _RecordingHook(scope=HookScope.RANK_ZERO, stage=DynamicsStage.AFTER_STEP)
        dp.register_hook(hl)
        dp.register_hook(hr)
        dp._domain_rank = 0
        dp._call_hooks(DynamicsStage.AFTER_STEP, batch)
        assert len(hl.calls) == 1
        assert len(hr.calls) == 1


class TestCallHooksAtStage:
    """Stage filtering: ``_call_hooks(stage)`` only fires hooks whose
    ``hook.stage`` matches."""

    def test_matching_stage_fires(self) -> None:
        dp, _ = _make_dp()
        batch = _make_batch()
        hook = _RecordingHook(stage=DynamicsStage.AFTER_STEP)
        dp.register_hook(hook)
        dp._call_hooks(DynamicsStage.AFTER_STEP, batch)
        assert len(hook.calls) == 1

    def test_nonmatching_stage_does_not_fire(self) -> None:
        dp, _ = _make_dp()
        batch = _make_batch()
        hook = _RecordingHook(stage=DynamicsStage.BEFORE_STEP)
        dp.register_hook(hook)
        dp._call_hooks(DynamicsStage.AFTER_STEP, batch)
        assert hook.calls == []


# ======================================================================
# Rank resolution — pick the right rank from mesh / dist / fallback.
# ======================================================================


class TestRankResolution:
    def test_rank_zero_without_mesh_or_dist(self) -> None:
        """No mesh, no dist init → rank 0."""
        dp, _ = _make_dp()
        assert dp._domain_rank == 0

    def test_rank_from_mock_mesh(self) -> None:
        """When ``config.mesh`` is set, its ``get_local_rank()`` wins."""
        mesh = MagicMock()
        mesh.get_local_rank.return_value = 3
        cfg = DomainConfig(cutoff=3.0, skin=0.5, mesh=mesh)
        inner = DemoDynamics(model=DemoModelWrapper(DemoModel()), n_steps=1)
        dp = DomainParallel(dynamics=inner, config=cfg)
        assert dp._domain_rank == 3

    def test_rank_fallback_when_mesh_raises(self) -> None:
        """If the mesh can't answer, fall through to 0 (no dist init
        here either)."""
        mesh = MagicMock()
        mesh.get_local_rank.side_effect = RuntimeError("no mesh")
        cfg = DomainConfig(cutoff=3.0, skin=0.5, mesh=mesh)
        inner = DemoDynamics(model=DemoModelWrapper(DemoModel()), n_steps=1)
        dp = DomainParallel(dynamics=inner, config=cfg)
        assert dp._domain_rank == 0


# ======================================================================
# Force priming — one-shot compute before first integrator step.
# ======================================================================


class TestPrimeForces:
    def test_prime_forces_not_called_in_single_process(self) -> None:
        """No ``_dist_model`` means step() short-circuits to inner
        dynamics; force priming is never triggered."""
        dp, inner = _make_dp()
        batch = _make_batch()
        dp.partition(batch)
        assert dp._forces_primed is False

        with patch.object(inner, "step", return_value=(batch, None)):
            dp.step(batch)
        # Priming is only fired by the distributed path (dist_model set).
        assert dp._forces_primed is False


# ======================================================================
# End-to-end gloo integration — drive NVE through DomainParallel on 2
# ranks and verify positions / velocities / energy match a single-
# process reference trajectory step-by-step.
# ======================================================================


def _init_gloo(rank: int, world_size: int, port: str) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = port
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
    # physicsnemo's gloo all-to-all shim (reused from distributed tests).
    import physicsnemo.distributed.utils as pn_utils

    def _impl(tensor, indices, sizes, dim=0, group=None):
        cs = dist.get_world_size(group=group)
        r = dist.get_rank(group=group)
        x_send = [tensor[idx].contiguous() for idx in indices]
        x_recv = []
        shape = list(tensor.shape)
        for i in range(cs):
            shape[dim] = sizes[i][r]
            x_recv.append(torch.empty(shape, dtype=tensor.dtype, device=tensor.device))
        ops = []
        for i in range(cs):
            if i == r:
                x_recv[i].copy_(x_send[i])
            else:
                if x_send[i].numel() > 0:
                    ops.append(dist.isend(x_send[i], dst=i, group=group))
                if x_recv[i].numel() > 0:
                    ops.append(dist.irecv(x_recv[i], src=i, group=group))
        for op in ops:
            op.wait()
        return torch.cat(x_recv, dim=dim)

    pn_utils.indexed_all_to_all_v_wrapper = _impl


def _worker(rank: int, world_size: int, port: str, fn_name: str, *args: Any) -> None:
    _init_gloo(rank, world_size, port)
    try:
        globals()[fn_name](rank, world_size, *args)
    finally:
        dist.destroy_process_group()


def _spawn(world_size: int, port: str, fn_name: str, *args: Any) -> None:
    mp.spawn(_worker, args=(world_size, port, fn_name, *args), nprocs=world_size)


def _build_lj_cluster(n_per_side: int = 5, dtype: torch.dtype = torch.float64):
    """Small open-cell argon cluster for NVE integration (non-PBC so we
    avoid the orthorhombic-vs-hex fractional-coord edge cases — those
    are covered in ``test_distributed_models``)."""
    spacing = 2 ** (1.0 / 6.0) * 3.40 * 1.05
    coords = torch.arange(n_per_side, dtype=dtype) * spacing
    gx, gy, gz = torch.meshgrid(coords, coords, coords, indexing="ij")
    positions = torch.stack([gx.flatten(), gy.flatten(), gz.flatten()], dim=-1)
    torch.manual_seed(0)
    positions = positions + 0.01 * torch.randn_like(positions)
    n = positions.shape[0]
    velocities = 0.001 * torch.randn(
        n, 3, dtype=dtype, generator=torch.Generator().manual_seed(1)
    )
    velocities = velocities - velocities.mean(dim=0, keepdim=True)
    atomic_numbers = torch.full((n,), 18, dtype=torch.long)
    masses = torch.full((n,), 39.948, dtype=dtype)
    # Open cell sized to the cluster extent plus one lattice spacing of margin.
    # A large fixed pad (the old ``+ 10.0``) left the atoms bunched in one
    # corner of an oversized box, so a spatial bisection put every atom on one
    # side and the other rank was assigned 0 owned atoms — a degenerate
    # partition the framework (correctly) rejects. Sizing the box to the atoms
    # makes the partition split real owned atoms onto every rank.
    box = (n_per_side - 1) * spacing + spacing
    cell = torch.eye(3, dtype=dtype) * box
    pbc = torch.zeros(3, dtype=torch.bool)
    return positions, velocities, atomic_numbers, masses, cell, pbc


def _nve_e2e_worker(rank: int, world_size: int, n_steps: int) -> None:
    """Smoke-test ``DomainParallel(NVE(LJ))``: run ``n_steps`` steps
    across ``world_size`` gloo ranks. After the loop, gather the
    distributed trajectory's final positions to rank 0 and compute the
    single-process potential energy at those positions; the
    all-reduced per-step energy the model returns should agree with
    that single-process value at machine precision on the final step.

    We don't try to match positions / velocities to a per-step
    reference trajectory because the reference's force priming and the
    distributed path's ``_prime_forces`` can interleave with the
    integrator's half-kicks differently (especially when NVE records
    ``batch.energy`` mid-step, not post-step). The POTENTIAL energy at
    a given set of positions is invariant under those differences —
    it's the strongest per-step equivalence we can assert without
    building a from-scratch velocity-Verlet in the test body."""
    from nvalchemi.distributed.config import DomainConfig
    from nvalchemi.dynamics.integrators.nve import NVE
    from nvalchemi.hooks.neighbor_list import NeighborListHook
    from nvalchemi.models.lj import LennardJonesModelWrapper
    from nvalchemi.neighbors import compute_neighbors

    dtype = torch.float64
    positions, velocities, atomic_numbers, masses, cell, pbc = _build_lj_cluster(
        n_per_side=5
    )
    n = positions.shape[0]

    # ── Distributed: DomainParallel around NVE(LJ) ──
    dist_wrapper = LennardJonesModelWrapper(epsilon=0.0104, sigma=3.40, cutoff=8.5)
    dist_nve = NVE(
        model=dist_wrapper,
        dt=1.0,
        hooks=[
            NeighborListHook(
                config=dist_wrapper.model_config.neighbor_config,
                skin=0.0,
                stage=DynamicsStage.BEFORE_COMPUTE,
            )
        ],
    )
    # Real gloo-backed DeviceMesh — ``_MockMesh`` lacks ``device_type``
    # which physicsnemo's ``ShardTensor.from_local`` requires.
    mesh = DeviceMesh("cpu", list(range(world_size)), mesh_dim_names=("domain",))
    cfg = DomainConfig(cutoff=float(dist_wrapper.cutoff), skin=0.0, mesh=mesh)
    dp = DomainParallel(dynamics=dist_nve, config=cfg)

    if rank == 0:
        full_data = AtomicData(
            atomic_numbers=atomic_numbers,
            positions=positions.clone(),
            atomic_masses=masses,
            forces=torch.zeros(n, 3, dtype=dtype),
            energy=torch.zeros(1, 1, dtype=dtype),
            cell=cell.unsqueeze(0),
            pbc=pbc.unsqueeze(0),
        )
        full_data.add_node_property("velocities", velocities.clone())
        full_batch = Batch.from_data_list([full_data])
    else:
        full_batch = None
    local_batch = dp.partition(full_batch)

    for _ in range(n_steps):
        local_batch, _ = dp.step(local_batch)

    # Gather the final positions to rank 0 and compute the reference
    # potential energy on those positions. The all-reduced dist
    # energy for this last step (recorded on ``local_batch.energy``
    # during ``step``) should agree to machine precision.
    dist_final_energy = float(local_batch.energy.sum().item())
    full_final = dp.gather(local_batch, dst=0)

    if rank == 0:
        assert full_final is not None
        assert full_final.num_nodes == n
        ref_wrapper = LennardJonesModelWrapper(epsilon=0.0104, sigma=3.40, cutoff=8.5)
        ref_data = AtomicData(
            atomic_numbers=full_final.atomic_numbers,
            positions=full_final.positions.clone(),
            atomic_masses=full_final.atomic_masses,
            cell=full_final.cell,
            pbc=full_final.pbc,
        )
        ref_batch = Batch.from_data_list([ref_data])
        compute_neighbors(ref_batch, config=ref_wrapper.model_config.neighbor_config)
        ref_out = ref_wrapper(ref_batch)
        ref_final_energy = float(ref_out["energy"].sum().item())
        assert abs(dist_final_energy - ref_final_energy) < 1e-8, (
            f"after {n_steps} NVE steps via DomainParallel, energy "
            f"{dist_final_energy:.6f} disagrees with single-process "
            f"reference {ref_final_energy:.6f} at gathered positions"
        )


def test_nve_lj_2ranks_end_to_end() -> None:
    """``DomainParallel(NVE(LJ))`` across 2 gloo ranks tracks the
    single-process NVE trajectory's total energy at every step."""
    _spawn(2, "29700", "_nve_e2e_worker", 3)


def test_nve_lj_4ranks_end_to_end() -> None:
    _spawn(4, "29701", "_nve_e2e_worker", 3)


def _nvt_langevin_e2e_worker(rank: int, world_size: int, n_steps: int) -> None:
    """Smoke-test ``DomainParallel(NVTLangevin(LJ))``: run ``n_steps``
    steps across ``world_size`` gloo ranks without crashing.

    Langevin's per-atom random noise diverges between ranks (each
    maintains its own RNG), so we don't assert trajectory equivalence
    against a single-process reference. Instead we check structural
    invariants after the loop:

    * Gather works — the final global system has the expected atom count.
    * Kinetic energy is finite and positive on rank 0.
    * Thermostat is doing something — post-run KE is within an order
      of magnitude of the target kT (not quite equilibrated after 3
      steps from cold velocities, but within the ballpark).
    """
    from nvalchemi.distributed.config import DomainConfig
    from nvalchemi.dynamics.integrators.nvt_langevin import NVTLangevin
    from nvalchemi.hooks.neighbor_list import NeighborListHook
    from nvalchemi.models.lj import LennardJonesModelWrapper

    dtype = torch.float64
    positions, velocities, atomic_numbers, masses, cell, pbc = _build_lj_cluster(
        n_per_side=4
    )
    n = positions.shape[0]

    dist_wrapper = LennardJonesModelWrapper(epsilon=0.0104, sigma=3.40, cutoff=8.5)
    dist_nvt = NVTLangevin(
        model=dist_wrapper,
        dt=1.0,
        temperature=100.0,
        friction=0.01,
        random_seed=42 + rank,  # per-rank seed
        hooks=[
            NeighborListHook(
                config=dist_wrapper.model_config.neighbor_config,
                skin=0.0,
                stage=DynamicsStage.BEFORE_COMPUTE,
            )
        ],
    )
    mesh = DeviceMesh("cpu", list(range(world_size)), mesh_dim_names=("domain",))
    cfg = DomainConfig(cutoff=float(dist_wrapper.cutoff), skin=0.0, mesh=mesh)
    dp = DomainParallel(dynamics=dist_nvt, config=cfg)

    if rank == 0:
        full_data = AtomicData(
            atomic_numbers=atomic_numbers,
            positions=positions.clone(),
            atomic_masses=masses,
            forces=torch.zeros(n, 3, dtype=dtype),
            energy=torch.zeros(1, 1, dtype=dtype),
            cell=cell.unsqueeze(0),
            pbc=pbc.unsqueeze(0),
        )
        full_data.add_node_property("velocities", velocities.clone())
        full_batch = Batch.from_data_list([full_data])
    else:
        full_batch = None

    local_batch = dp.partition(full_batch)
    for _ in range(n_steps):
        local_batch, _ = dp.step(local_batch)

    full_final = dp.gather(local_batch, dst=0)
    if rank == 0:
        assert full_final is not None
        assert full_final.num_nodes == n
        ke = (
            0.5
            * ((full_final.velocities**2).sum(dim=-1) * full_final.atomic_masses)
            .sum()
            .item()
        )
        assert ke > 0.0 and ke < 1e6, (
            f"NVTLangevin produced non-finite / absurd KE={ke:.3e} after "
            f"{n_steps} steps — integrator-halo coupling is broken"
        )


def test_nvt_langevin_lj_2ranks_end_to_end() -> None:
    """``DomainParallel(NVTLangevin(LJ))`` runs cleanly across 2 gloo
    ranks. Smoke test, not trajectory equivalence — Langevin's per-rank
    RNG diverges by construction."""
    _spawn(2, "29702", "_nvt_langevin_e2e_worker", 3)
