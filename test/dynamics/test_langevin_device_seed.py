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
Device-seed Langevin half step and CUDA-graph replay of ``NVTLangevin.step``.

Tests verify:
- ``langevin_half_step_device_seed`` reproduces the Python-int-seed op bit for
  bit, advances its seed by one per call, and follows the int-seed sequence
- ``NVTLangevin`` keeps its noise stream at ``random_seed + step_count``, also
  after ``step_count`` is set from outside
- a ``dyn.step()`` captured on the engine's stream (``with dyn:`` and
  ``torch.cuda.graph(g, stream=dyn.torch_stream)``, the sanctioned path)
  draws fresh noise on every replay, equals the eager run bitwise, and
  samples the Maxwell-Boltzmann kinetic temperature
"""

from __future__ import annotations

from collections import OrderedDict

import pytest
import torch
from torch import nn

from nvalchemi.data import AtomicData, Batch
from nvalchemi.dynamics import NVTLangevin
from nvalchemi.dynamics._ops.langevin import (
    langevin_half_step,
    langevin_half_step_device_seed,
)
from nvalchemi.dynamics._units import KB_EV
from nvalchemi.models.base import BaseModelMixin, ModelConfig

N = 64


class _Free(nn.Module, BaseModelMixin):
    """Toy model: a weak harmonic trap (or none), enough to give ``compute`` something to evaluate."""

    def __init__(self, k: float = 0.0):
        super().__init__()
        self.k = k
        self.model_config = ModelConfig(
            outputs=frozenset({"energy", "forces"}),
            active_outputs={"energy", "forces"},
            autograd_outputs=frozenset(),
            autograd_inputs=frozenset(),
            required_inputs=frozenset(),
            optional_inputs=frozenset(),
            supports_pbc=False,
            needs_pbc=False,
            neighbor_config=None,
        )

    @property
    def embedding_shapes(self):
        return {}

    def compute_embeddings(self, data, **kwargs):
        raise NotImplementedError

    def forward(self, data, **kwargs):
        f = -self.k * data.positions
        return OrderedDict(energy=f.new_zeros(data.num_graphs, 1), forces=f)


def _batch(device: str, n: int = N) -> Batch:
    g = torch.Generator().manual_seed(0)
    data = AtomicData(
        positions=torch.randn(n, 3, generator=g).to(device),
        atomic_numbers=torch.full((n,), 6, dtype=torch.long, device=device),
        cell=(torch.eye(3) * 50.0).unsqueeze(0).to(device),
        pbc=torch.zeros(1, 3, dtype=torch.bool, device=device),
        energy=torch.zeros(1, 1, device=device),
        forces=torch.zeros(n, 3, device=device),
    )
    data.use_default_masses()
    data.use_default_velocities()
    b = Batch.from_data_list([data], device=device)
    b.velocities.copy_(0.1 * torch.randn(n, 3, generator=g).to(device))
    return b


def _langevin(k: float = 0.0, friction: float = 0.01, seed: int = 9) -> NVTLangevin:
    return NVTLangevin(
        _Free(k), dt=1.0, temperature=300.0, friction=friction, random_seed=seed
    )


def _op_system(device: str, dtype: torch.dtype = torch.float32):
    g = torch.Generator().manual_seed(1)
    pos = torch.randn(N, 3, generator=g, dtype=dtype).to(device)
    vel = 0.1 * torch.randn(N, 3, generator=g, dtype=dtype).to(device)
    frc = torch.randn(N, 3, generator=g, dtype=dtype).to(device)
    mass = (1.0 + torch.rand(N, generator=g, dtype=dtype)).to(device)
    dt = torch.full((1,), 0.05, dtype=dtype, device=device)
    kT = torch.full((1,), 0.02, dtype=dtype, device=device)
    gamma = torch.full((1,), 0.5, dtype=dtype, device=device)
    bidx = torch.zeros(N, dtype=torch.int32, device=device)
    return pos, vel, frc, mass, dt, kT, gamma, bidx


class TestDeviceSeedOp:
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_matches_int_seed_op_bitwise(self, device, dtype):
        pos, vel, frc, mass, dt, kT, gamma, bidx = _op_system(device, dtype)
        p1, v1 = pos.clone(), vel.clone()
        langevin_half_step(p1, v1, frc, mass, dt, kT, gamma, 123, bidx)
        p2, v2 = pos.clone(), vel.clone()
        seed = torch.tensor([123], dtype=torch.int32, device=device)
        langevin_half_step_device_seed(p2, v2, frc, mass, dt, kT, gamma, seed, bidx)
        assert torch.equal(p1, p2) and torch.equal(v1, v2)

    def test_seed_advances_by_one_per_call(self, device):
        pos, vel, frc, mass, dt, kT, gamma, bidx = _op_system(device)
        seed = torch.tensor([40], dtype=torch.int32, device=device)
        for expected in (41, 42, 43):
            langevin_half_step_device_seed(
                pos, vel, frc, mass, dt, kT, gamma, seed, bidx
            )
            assert int(seed.item()) == expected

    def test_consecutive_calls_follow_the_int_seed_sequence(self, device):
        pos, vel, frc, mass, dt, kT, gamma, bidx = _op_system(device)
        pe, ve = pos.clone(), vel.clone()
        for s in (7, 8, 9):
            langevin_half_step(pe, ve, frc, mass, dt, kT, gamma, s, bidx)
        pd, vd = pos.clone(), vel.clone()
        seed = torch.tensor([7], dtype=torch.int32, device=device)
        for _ in range(3):
            langevin_half_step_device_seed(pd, vd, frc, mass, dt, kT, gamma, seed, bidx)
        assert torch.equal(pe, pd) and torch.equal(ve, vd)


class TestNVTLangevinSeed:
    def test_noise_stream_is_seed_plus_step_count(self, device):
        dyn, b = _langevin(), _batch(device)
        ref = _batch(device)
        dyn._ensure_state_initialized(b)
        masses = b.atomic_masses.clone()
        for _ in range(4):
            dyn.pre_update(b)
            dyn.post_update(b)
            dyn.step_count += 1
        pr, vr = ref.positions.clone(), ref.velocities.clone()
        st = dyn._state
        for s in range(4):
            langevin_half_step(
                pr,
                vr,
                torch.zeros_like(pr),
                masses,
                st.dt,
                st.temperature,
                st.friction,
                9 + s,
                ref.batch_idx.int(),
            )
            from nvalchemi.dynamics._ops.langevin import langevin_finalize

            langevin_finalize(
                vr, torch.zeros_like(pr), masses, st.dt, ref.batch_idx.int()
            )
        assert torch.allclose(b.positions, pr, atol=1e-6)
        assert torch.allclose(b.velocities, vr, atol=1e-6)

    def test_step_count_set_from_outside_resyncs_the_seed(self, device):
        a, ba = _langevin(), _batch(device)
        c, bc = _langevin(), _batch(device)
        a._ensure_state_initialized(ba)
        c._ensure_state_initialized(bc)
        for _ in range(5):
            a.pre_update(ba)
            a.post_update(ba)
            a.step_count += 1
        # c skips the first 5 steps by setting step_count; its noise must be the one of step 5
        c.step_count = 5
        torch.manual_seed(0)
        pa, va = ba.positions.clone(), ba.velocities.clone()
        bc.positions.copy_(pa)
        bc.velocities.copy_(va)
        a.pre_update(ba)
        c.pre_update(bc)
        assert torch.equal(ba.positions, bc.positions)
        assert torch.equal(ba.velocities, bc.velocities)

    @pytest.mark.parametrize("how", ["setter", "seed_offset"])
    def test_seed_changed_after_steps_takes_effect(self, device, how):
        """A new base seed on an initialised engine must change the noise from the next step on."""
        a, ba = _langevin(seed=9), _batch(device)
        a._ensure_state_initialized(ba)
        for _ in range(3):  # host mirror equals step_count after a normal step
            a.pre_update(ba)
            a.post_update(ba)
            a.step_count += 1
        ref, bref = _langevin(seed=1009), _batch(device)
        ref._ensure_state_initialized(bref)
        ref.step_count = 3
        bref.positions.copy_(ba.positions)
        bref.velocities.copy_(ba.velocities)
        if how == "setter":
            a.random_seed = 1009
        else:
            assert a.seed_offset(1000) == ()
        a.pre_update(ba)
        ref.pre_update(bref)
        assert torch.equal(ba.positions, bref.positions)
        assert torch.equal(ba.velocities, bref.velocities)
        assert int(a._rng_seed.item()) == 1009 + 3 + 1

    def test_sync_rng_is_public_and_eager(self, device):
        dyn, b = _langevin(seed=9), _batch(device)
        with pytest.raises(RuntimeError, match="not initialised"):
            dyn.sync_rng()
        dyn._ensure_state_initialized(b)
        dyn.step_count = 17
        dyn.random_seed = 100
        dyn.sync_rng()
        assert int(dyn._rng_seed.item()) == 117
        # in sync now: a step must not resync (the seed only advances by the op's own +1)
        dyn.pre_update(b)
        assert int(dyn._rng_seed.item()) == 118


@pytest.mark.usefixtures("gpu_device")
class TestCapturedStep:
    def _capture(self, dyn, b):
        """Warm up eagerly, then capture one ``dyn.step`` on the engine's own stream."""
        dyn.step(b)  # warm-up (counts as step 1; loads the Warp modules, primes forces)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=dyn.torch_stream):
            dyn.step(b)
        return graph

    def test_capture_after_a_restore_refuses_a_stale_seed_then_syncs(self, gpu_device):
        """Warm up, restore a checkpoint (new step_count), then capture: never record a resync."""
        dyn, b = _langevin(), _batch(gpu_device)
        dyn._ensure_state_initialized(b)
        with dyn:
            dyn.step(b)  # warm-up
            dyn.step_count = 50  # what load_state_dict does
            graph = torch.cuda.CUDAGraph()
            with pytest.raises(RuntimeError, match="sync_rng"):
                with torch.cuda.graph(graph, stream=dyn.torch_stream):
                    dyn.step(b)
            # the documented remedy: resync eagerly, then capture
            dyn.sync_rng()
            assert int(dyn._rng_seed.item()) == 9 + 50
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=dyn.torch_stream):
                dyn.step(b)
            snap = (b.positions.clone(), b.velocities.clone(), b.forces.clone())
            outs = []
            for _ in range(2):
                b.positions.copy_(snap[0])
                b.velocities.copy_(snap[1])
                b.forces.copy_(snap[2])
                graph.replay()
                torch.cuda.synchronize()
                outs.append(b.velocities.clone())
        assert not torch.equal(
            outs[0], outs[1]
        )  # fresh noise per replay, no recorded reset
        assert int(dyn._rng_seed.item()) == 9 + 50 + 2

    def test_replays_draw_fresh_noise(self, gpu_device):
        dyn, b = _langevin(), _batch(gpu_device)
        dyn._ensure_state_initialized(b)
        with dyn:
            graph = self._capture(dyn, b)
            snap = (b.positions.clone(), b.velocities.clone(), b.forces.clone())
            outs = []
            for _ in range(2):
                b.positions.copy_(snap[0])
                b.velocities.copy_(snap[1])
                b.forces.copy_(snap[2])
                graph.replay()
                torch.cuda.synchronize()
                outs.append(b.velocities.clone())
        # same start state, consecutive replays: only the seed differs
        assert not torch.equal(outs[0], outs[1])

    def test_replays_equal_the_eager_run_bitwise(self, gpu_device):
        n_steps = 12
        ref_dyn, ref = _langevin(k=0.05), _batch(gpu_device)
        ref_dyn._ensure_state_initialized(ref)
        for _ in range(n_steps):
            ref_dyn.step(ref)

        dyn, b = _langevin(k=0.05), _batch(gpu_device)
        dyn._ensure_state_initialized(b)
        with dyn:
            graph = self._capture(dyn, b)  # 1 eager step; the capture does not execute
            for _ in range(n_steps - 1):
                graph.replay()
            torch.cuda.synchronize()
        assert torch.equal(b.positions, ref.positions)
        assert torch.equal(b.velocities, ref.velocities)

    def test_replayed_kinetic_temperature_matches_the_bath(self, gpu_device):
        n = 4096
        dyn, b = _langevin(friction=0.1), _batch(gpu_device, n)
        dyn._ensure_state_initialized(b)
        kT = 300.0 * KB_EV
        with dyn:
            graph = self._capture(dyn, b)
            for _ in range(150):  # ~15 relaxation times
                graph.replay()
            samples = []
            for _ in range(60):
                graph.replay()
                samples.append(
                    (b.atomic_masses.unsqueeze(-1) * b.velocities.pow(2)).mean().item()
                )
            torch.cuda.synchronize()
        measured = sum(samples) / len(samples)
        # m <v^2> per degree of freedom = kT; 4096 atoms x 3 dof x 60 samples is a ~0.3 % estimator
        assert measured == pytest.approx(kT, rel=0.03)
