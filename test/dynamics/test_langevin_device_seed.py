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
Device step offset of the Langevin half step and CUDA-graph replay of ``NVTLangevin.step``.

Tests verify:
- ``langevin_half_step`` with ``step_offset=None`` is the Python-int-seed op bit
  for bit; an int32 ``step_offset`` equals the shifted int seed (also for seeds
  of ``2**31`` and above) and is never written by the op
- ``NVTLangevin`` keeps its noise stream at ``random_seed + step_count``, also
  after ``step_count`` is set from outside, and takes ``random_seed >= 2**31``
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
from nvalchemi.dynamics._ops.langevin import langevin_finalize, langevin_half_step
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


def _offset(value: int, device: str) -> torch.Tensor:
    return torch.tensor([value], dtype=torch.int32, device=device)


class TestStepOffsetOp:
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_none_equals_int_seed_op_bitwise(self, device, dtype):
        pos, vel, frc, mass, dt, kT, gamma, bidx = _op_system(device, dtype)
        p1, v1 = pos.clone(), vel.clone()
        langevin_half_step(p1, v1, frc, mass, dt, kT, gamma, 123, bidx)
        p2, v2 = pos.clone(), vel.clone()
        langevin_half_step(p2, v2, frc, mass, dt, kT, gamma, 123, bidx, None)
        assert torch.equal(p1, p2) and torch.equal(v1, v2)

    @pytest.mark.parametrize("seed", [9, 2**31 + 7])
    def test_offsets_follow_the_shifted_int_seed_and_stay_untouched(self, device, seed):
        """Eager offsets 0..4 with seed s equal the int seeds s .. s + 4, bitwise."""
        pos, vel, frc, mass, dt, kT, gamma, bidx = _op_system(device)
        pe, ve = pos.clone(), vel.clone()
        pd, vd = pos.clone(), vel.clone()
        for k in range(5):
            off = _offset(k, device)
            langevin_half_step(pe, ve, frc, mass, dt, kT, gamma, seed + k, bidx)
            langevin_half_step(pd, vd, frc, mass, dt, kT, gamma, seed, bidx, off)
            assert int(off.item()) == k
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
        assert int(a._step_offset.item()) == 3 + 1

    def test_sync_rng_is_public_and_eager(self, device):
        dyn, b = _langevin(seed=9), _batch(device)
        with pytest.raises(RuntimeError, match="not initialised"):
            dyn.sync_rng()
        dyn._ensure_state_initialized(b)
        dyn.step_count = 17
        dyn.random_seed = 100
        dyn.sync_rng()
        assert int(dyn._step_offset.item()) == 17
        # in sync now: a step must not resync (the counter only advances by pre_update's own +1)
        dyn.pre_update(b)
        assert int(dyn._step_offset.item()) == 18

    def test_seed_of_2_pow_31_and_above_works(self, device):
        """random_seed >= 2**31 is a host launch argument (it overflowed the int32 seed tensor)."""
        seed = 2**31 + 7
        dyn, b = _langevin(seed=seed), _batch(device)
        ref = _batch(device)
        dyn._ensure_state_initialized(b)
        masses = b.atomic_masses.clone()
        for _ in range(3):
            dyn.pre_update(b)
            dyn.post_update(b)
            dyn.step_count += 1
        assert torch.isfinite(b.positions).all() and torch.isfinite(b.velocities).all()
        pr, vr = ref.positions.clone(), ref.velocities.clone()
        st = dyn._state
        zeros = torch.zeros_like(pr)
        for s in range(3):
            langevin_half_step(
                pr, vr, zeros, masses, st.dt, st.temperature, st.friction,
                seed + s, ref.batch_idx.int(),
            )  # fmt: skip
            langevin_finalize(vr, zeros, masses, st.dt, ref.batch_idx.int())
        assert torch.allclose(b.positions, pr, atol=1e-6)
        assert torch.allclose(b.velocities, vr, atol=1e-6)


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
            assert int(dyn._step_offset.item()) == 50
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
        assert int(dyn._step_offset.item()) == 50 + 2

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
