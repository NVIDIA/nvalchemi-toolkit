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
"""Unit tests for transactional Zarr checkpointing of enhanced sampling.

Covers the state encoder, the manifest-gated commit, checksum verification,
``BaseDynamics`` state round-trips, thermodynamic-state rebinding, and exact
trajectory reproduction across a checkpoint/restore boundary.
"""

from __future__ import annotations

import math

import pytest
import torch
import zarr
from torch import Tensor

from nvalchemi.data import AtomicData, Batch
from nvalchemi.dynamics import NVTLangevin, NVTNoseHoover
from nvalchemi.enhanced_sampling import (
    AdaptivePotentialMixin,
    BiasResult,
    ConservativeBias,
    EnhancedSampling,
    HarmonicUmbrellaBias,
    pair_distance,
)
from nvalchemi.enhanced_sampling._checkpoint import (
    CHECKPOINT_FORMAT_VERSION,
    _component_checksum,
    _decode_state,
    _encode_state,
    read_checkpoint,
)
from nvalchemi.models.demo import DemoModel, DemoModelWrapper

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_batch(
    n_graphs: int = 2, atoms_per_graph: int = 4, device: str = "cpu", seed: int = 7
) -> Batch:
    """Return a batch with output buffers and velocities."""
    torch.manual_seed(seed)
    data_list = []
    for _ in range(n_graphs):
        data = AtomicData(
            positions=torch.randn(atoms_per_graph, 3),
            atomic_numbers=torch.full((atoms_per_graph,), 6, dtype=torch.long),
            atomic_masses=torch.ones(atoms_per_graph),
            forces=torch.zeros(atoms_per_graph, 3),
            energy=torch.zeros(1, 1),
        )
        data.add_node_property("velocities", torch.zeros(atoms_per_graph, 3))
        data_list.append(data)
    batch = Batch.from_data_list(data_list).to(device)
    batch["thermodynamic_state_id"] = torch.arange(n_graphs, device=device)
    return batch


def _make_dynamics(device: str = "cpu", seed: int = 0) -> NVTLangevin:
    torch.manual_seed(seed)
    model = DemoModelWrapper(DemoModel()).to(device)
    return NVTLangevin(model=model, dt=0.1, temperature=300.0, friction=0.1)


def _make_runner(
    device: str = "cpu",
    steps_per_epoch: int = 4,
    seed: int = 0,
    n_states: int = 2,
) -> EnhancedSampling:
    idx = torch.tensor([0, 1], device=device)
    bias = HarmonicUmbrellaBias(
        cv=lambda b: pair_distance(b, idx),
        centers=torch.arange(2.0, 2.0 + n_states).reshape(n_states, 1),
        stiffness=4.0,
        name="u",
    )
    return EnhancedSampling(
        _make_dynamics(device, seed), {"u": bias}, steps_per_epoch=steps_per_epoch
    )


class _CountingBias(AdaptivePotentialMixin, ConservativeBias):
    """Adaptive bias with a scalar history worth round-tripping."""

    def __init__(self, name: str = "counter") -> None:
        super().__init__(name=name)
        self.register_buffer("deposits", torch.zeros(1))

    def energy(self, current: Batch) -> Tensor:
        return (
            torch.zeros(current.num_graphs, 1, device=current.positions.device)
            + 0.0 * current.positions.sum()
        )

    def update(self, frames: Batch, result: BiasResult) -> None:
        self.deposits += 1
        self.bump_state_version()


class _SharedHistoryBias(AdaptivePotentialMixin, ConservativeBias):
    """Deposits accumulate as pending; commit_epoch merges them.

    Models the shared-history multi-walker case: the published state only
    becomes correct once the epoch commit has run, so a checkpoint taken
    before it records a bias mid-merge.
    """

    def __init__(self, name: str = "shared") -> None:
        super().__init__(name=name)
        self.register_buffer("pending", torch.zeros(1))
        self.register_buffer("published", torch.zeros(1))
        self.commit_calls = 0

    def energy(self, current: Batch) -> Tensor:
        return (
            torch.zeros(current.num_graphs, 1, device=current.positions.device)
            + 0.0 * current.positions.sum()
        )

    def update(self, frames: Batch, result: BiasResult) -> None:
        self.pending += 1

    def commit_epoch(self) -> None:
        self.commit_calls += 1
        self.published += self.pending
        self.pending.zero_()


# ===========================================================================
# 1. State encoding
# ===========================================================================


class TestStateEncoding:
    """Nested state survives the Zarr round-trip without pickle."""

    def test_tensors_scalars_and_nesting(self, tmp_path) -> None:
        state = {
            "counter": 7,
            "label": "umbrella",
            "ratio": 0.25,
            "flag": True,
            "nothing": None,
            "listy": [1, 2, 3],
            "weights": torch.arange(6, dtype=torch.float64).reshape(2, 3),
            "ids": torch.tensor([4, 5], dtype=torch.int64),
            "nested": {"inner": torch.ones(2), "depth": 2},
        }
        group = zarr.open_group(str(tmp_path / "s.zarr"), mode="w")
        _encode_state(group, state)
        restored = _decode_state(group, "cpu")

        assert restored["counter"] == 7
        assert restored["label"] == "umbrella"
        assert restored["flag"] is True
        assert restored["nothing"] is None
        assert restored["listy"] == [1, 2, 3]
        assert torch.equal(restored["weights"], state["weights"])
        assert restored["weights"].dtype == torch.float64
        assert restored["ids"].dtype == torch.int64
        assert torch.equal(restored["nested"]["inner"], state["nested"]["inner"])
        assert restored["nested"]["depth"] == 2

    def test_empty_tensor_round_trips(self, tmp_path) -> None:
        group = zarr.open_group(str(tmp_path / "s.zarr"), mode="w")
        _encode_state(group, {"empty": torch.zeros(0, 3)})
        restored = _decode_state(group, "cpu")
        assert restored["empty"].shape == (0, 3)

    def test_unsupported_type_raises_rather_than_pickling(self, tmp_path) -> None:
        """Refusing is the point: a pickle payload would make a checkpoint
        executable and unreadable outside Python."""
        group = zarr.open_group(str(tmp_path / "s.zarr"), mode="w")
        with pytest.raises(TypeError, match="no pickle payloads"):
            _encode_state(group, {"bad": object()})

    def test_checksum_is_order_independent(self) -> None:
        a = {"x": torch.ones(3), "y": 2}
        b = {"y": 2, "x": torch.ones(3)}
        assert _component_checksum(a) == _component_checksum(b)

    def test_checksum_detects_value_change(self) -> None:
        base = _component_checksum({"x": torch.ones(3)})
        assert base != _component_checksum({"x": torch.zeros(3)})
        assert base != _component_checksum({"x": torch.ones(3) * 2})

    def test_checksum_detects_dtype_change(self) -> None:
        assert _component_checksum({"x": torch.ones(3)}) != _component_checksum(
            {"x": torch.ones(3, dtype=torch.float64)}
        )


# ===========================================================================
# 2. BaseDynamics state
# ===========================================================================


class TestDynamicsState:
    """state_dict / load_state_dict on the integrator."""

    def test_round_trip_restores_counters_and_state(self, device: str) -> None:
        batch = _make_batch(device=device)
        dynamics = _make_dynamics(device)
        dynamics._ensure_state_initialized(batch)
        dynamics.step_count = 13

        saved = dynamics.state_dict()
        assert saved["step_count"] == 13
        assert saved["random_seed"] == 42
        assert "temperature" in saved["state"]

        other = _make_dynamics(device)
        other._ensure_state_initialized(batch)
        other.load_state_dict(saved)
        assert other.step_count == 13
        assert torch.allclose(other._state.temperature, dynamics._state.temperature)

    def test_load_into_uninitialised_integrator_raises(self, device: str) -> None:
        """Restoring into an uninitialised integrator would silently diverge."""
        batch = _make_batch(device=device)
        source = _make_dynamics(device)
        source._ensure_state_initialized(batch)
        target = _make_dynamics(device)
        with pytest.raises(RuntimeError, match="has not initialised its own"):
            target.load_state_dict(source.state_dict())

    def test_unknown_key_raises(self, device: str) -> None:
        batch = _make_batch(device=device)
        dynamics = _make_dynamics(device)
        dynamics._ensure_state_initialized(batch)
        saved = dynamics.state_dict()
        saved["state"]["not_a_real_key"] = torch.zeros(2)
        with pytest.raises(KeyError, match="no counterpart"):
            dynamics.load_state_dict(saved)

    def test_langevin_noise_is_counter_based(self, device: str) -> None:
        """Exact restart relies on this: no generator state to serialise."""
        dynamics = _make_dynamics(device)
        assert dynamics._random_seed == 42
        state = dynamics.state_dict()
        assert "random_seed" in state and "step_count" in state

    def test_redistribute_state_permutes_rows(self, device: str) -> None:
        batch = _make_batch(n_graphs=3, device=device)
        dynamics = _make_dynamics(device)
        dynamics._ensure_state_initialized(batch)
        with torch.no_grad():
            dynamics._state.temperature.copy_(
                torch.tensor([1.0, 2.0, 3.0], device=device).reshape(
                    dynamics._state.temperature.shape
                )
            )
        dynamics.redistribute_state(torch.tensor([2, 0, 1], device=device))
        assert dynamics._state.temperature.reshape(-1).tolist() == [3.0, 1.0, 2.0]


# ===========================================================================
# 3. Thermodynamic-state rebinding
# ===========================================================================


class TestThermodynamicStateRebinding:
    """The adapters replica exchange will need in PR 5."""

    def test_base_dynamics_refuses(self, device: str) -> None:
        """An integrator that cannot rebind must fail, not accept silently."""
        from nvalchemi.dynamics.base import BaseDynamics

        dynamics = BaseDynamics(DemoModelWrapper(DemoModel()).to(device))
        with pytest.raises(NotImplementedError, match="does not support"):
            dynamics.apply_thermodynamic_state(torch.tensor([0]), torch.tensor([300.0]))

    def test_langevin_rebinds_temperature(self, device: str) -> None:
        batch = _make_batch(device=device)
        dynamics = _make_dynamics(device)
        dynamics._ensure_state_initialized(batch)
        before = dynamics._state.temperature.reshape(-1).clone()

        dynamics.apply_thermodynamic_state(
            torch.tensor([1, 0], device=device),
            torch.tensor([300.0, 600.0], device=device),
        )
        after = dynamics._state.temperature.reshape(-1)
        assert abs(float(after[0] / before[0]) - 2.0) < 1e-5
        assert abs(float(after[1] / before[1]) - 1.0) < 1e-5

    def test_langevin_velocity_scaling_follows_temperature(self, device: str) -> None:
        """The swap is indivisible: target and velocities move together."""
        batch = _make_batch(n_graphs=1, atoms_per_graph=3, device=device)
        batch.velocities.fill_(1.0)
        dynamics = _make_dynamics(device)
        dynamics._ensure_state_initialized(batch)
        dynamics.apply_thermodynamic_state(
            torch.tensor([0], device=device), torch.tensor([1200.0], device=device)
        )
        dynamics.rescale_velocities_for_state(batch)
        # T: 300 -> 1200, so v scales by sqrt(4) = 2.
        assert torch.allclose(
            batch.velocities, torch.full_like(batch.velocities, 2.0), atol=1e-5
        )

    def test_out_of_range_state_id_raises(self, device: str) -> None:
        batch = _make_batch(device=device)
        dynamics = _make_dynamics(device)
        dynamics._ensure_state_initialized(batch)
        with pytest.raises(IndexError, match="out of range"):
            dynamics.apply_thermodynamic_state(
                torch.tensor([0, 5], device=device),
                torch.tensor([300.0], device=device),
            )

    def test_nose_hoover_transforms_chain_state(self, device: str) -> None:
        """Q and eta_dot must move with kT or detailed balance breaks."""
        batch = _make_batch(device=device)
        model = DemoModelWrapper(DemoModel()).to(device)
        dynamics = NVTNoseHoover(
            model=model, dt=0.1, temperature=300.0, thermostat_time=10.0
        )
        dynamics._ensure_state_initialized(batch)
        with torch.no_grad():
            dynamics._state.nhc_eta_dot.fill_(2.0)
        q_before = dynamics._state.nhc_Q.clone()
        eta_dot_before = dynamics._state.nhc_eta_dot.clone()

        dynamics.apply_thermodynamic_state(
            torch.tensor([0, 0], device=device),
            torch.tensor([1200.0], device=device),
        )
        ratio = 4.0  # 300 -> 1200
        assert torch.allclose(dynamics._state.nhc_Q, q_before * ratio, rtol=1e-5), (
            "chain masses must scale with kT"
        )
        assert torch.allclose(
            dynamics._state.nhc_eta_dot,
            eta_dot_before / math.sqrt(ratio),
            rtol=1e-5,
        ), "chain velocities must scale as 1/sqrt(kT)"

    def test_nose_hoover_chain_kinetic_energy_invariant(self, device: str) -> None:
        """Q eta_dot^2 must not change: rebinding injects no thermostat energy."""
        batch = _make_batch(device=device)
        dynamics = NVTNoseHoover(
            model=DemoModelWrapper(DemoModel()).to(device),
            dt=0.1,
            temperature=300.0,
            thermostat_time=10.0,
        )
        dynamics._ensure_state_initialized(batch)
        with torch.no_grad():
            dynamics._state.nhc_eta_dot.fill_(1.5)
        before = (dynamics._state.nhc_Q * dynamics._state.nhc_eta_dot**2).sum()

        dynamics.apply_thermodynamic_state(
            torch.tensor([0, 0], device=device),
            torch.tensor([900.0], device=device),
        )
        after = (dynamics._state.nhc_Q * dynamics._state.nhc_eta_dot**2).sum()
        assert torch.allclose(before, after, rtol=1e-5)


# ===========================================================================
# 4. Transactional checkpoint
# ===========================================================================


class TestCheckpointTransactionality:
    """The manifest is the commit marker; checksums catch later damage."""

    def test_round_trip(self, tmp_path, device: str) -> None:
        batch = _make_batch(device=device)
        runner = _make_runner(device)
        batch = runner.run(batch, n_steps=4)
        path = tmp_path / "ck.zarr"
        runner.checkpoint(path)

        restored_batch, states, manifest = read_checkpoint(path, device)
        assert manifest.format_version == CHECKPOINT_FORMAT_VERSION
        assert manifest.sampling_step == 4
        assert manifest.sampling_epoch == 1
        assert manifest.num_graphs == 2
        assert "dynamics" in states
        assert "biases/u" in states
        assert restored_batch.num_graphs == 2

    def test_store_without_manifest_is_refused(self, tmp_path, device: str) -> None:
        """An interrupted write leaves no manifest; restoring it must fail."""
        from nvalchemi.data.datapipes.backends.zarr import AtomicDataZarrWriter

        path = tmp_path / "torn.zarr"
        AtomicDataZarrWriter(str(path)).write(_make_batch(device=device))
        with pytest.raises(ValueError, match="no committed manifest"):
            read_checkpoint(path, device)

    def test_manifest_is_written_last(self, tmp_path, device: str) -> None:
        """Every declared component must already exist when the manifest lands."""
        batch = _make_batch(device=device)
        runner = _make_runner(device)
        runner.run(batch, n_steps=4)
        path = tmp_path / "ck.zarr"
        runner.checkpoint(path)

        root = zarr.open_group(str(path), mode="r")
        manifest = dict(root["sampling/manifest"].attrs["manifest"])
        for name in manifest["components"]:
            node = root["sampling"]
            for part in name.split("/"):
                assert part in node, f"{name} declared but missing"
                node = node[part]

    def test_corrupted_component_fails_checksum(self, tmp_path, device: str) -> None:
        """Damage after the manifest landed is caught on read."""
        batch = _make_batch(device=device)
        runner = _make_runner(device)
        runner.run(batch, n_steps=4)
        path = tmp_path / "ck.zarr"
        runner.checkpoint(path)

        root = zarr.open_group(str(path), mode="a")
        temperature = root["sampling/dynamics/state/temperature"]
        temperature[...] = temperature[...] * 3.0

        with pytest.raises(ValueError, match="failed its checksum"):
            read_checkpoint(path, device)

    def test_missing_declared_component_is_caught(self, tmp_path, device: str) -> None:
        batch = _make_batch(device=device)
        runner = _make_runner(device)
        runner.run(batch, n_steps=4)
        path = tmp_path / "ck.zarr"
        runner.checkpoint(path)

        root = zarr.open_group(str(path), mode="a")
        del root["sampling/biases"]
        with pytest.raises(ValueError, match="manifest but the group is missing"):
            read_checkpoint(path, device)

    @pytest.mark.parametrize(
        "array_path",
        [
            "core/positions",
            "core/velocities",
            "core/forces",
            "custom/walker_id",
            "custom/thermodynamic_state_id",
        ],
    )
    def test_corrupted_walker_batch_is_rejected(
        self, tmp_path, device: str, array_path: str
    ) -> None:
        """Integrity must cover the batch, not only the sampling/ state.

        These arrays are written by AtomicDataZarrWriter, outside the
        per-component checksum path. Leaving them uncovered would attest to
        the bias and integrator while silently restoring corrupted
        coordinates or a scrambled walker identity — the half of a checkpoint
        a reader is most likely to trust without looking.
        """
        batch = _make_batch(device=device)
        runner = _make_runner(device)
        runner.run(batch, n_steps=4)
        path = tmp_path / "ck.zarr"
        runner.checkpoint(path)

        root = zarr.open_group(str(path), mode="a")
        root[array_path][...] = root[array_path][...] + 1

        with pytest.raises(ValueError, match="walker batch failed its checksum"):
            read_checkpoint(path, device)

    def test_corrupted_pointer_array_is_rejected(self, tmp_path, device: str) -> None:
        """meta/ carries the CSR pointers that define graph boundaries."""
        batch = _make_batch(device=device)
        runner = _make_runner(device)
        runner.run(batch, n_steps=4)
        path = tmp_path / "ck.zarr"
        runner.checkpoint(path)

        root = zarr.open_group(str(path), mode="a")
        root["meta/atoms_ptr"][...] = root["meta/atoms_ptr"][...] + 1

        with pytest.raises(ValueError, match="walker batch failed its checksum"):
            read_checkpoint(path, device)

    def test_manifest_records_a_batch_checksum(self, tmp_path, device: str) -> None:
        batch = _make_batch(device=device)
        runner = _make_runner(device)
        runner.run(batch, n_steps=4)
        path = tmp_path / "ck.zarr"
        runner.checkpoint(path)

        _, _, manifest = read_checkpoint(path, device)
        assert manifest.batch_checksum, "batch is not covered by any checksum"
        assert len(manifest.batch_checksum) == 64

    def test_batch_checksum_is_independent_of_sampling_groups(
        self, tmp_path, device: str
    ) -> None:
        """It must be computed before sampling/ lands, or it would drift."""
        from nvalchemi.enhanced_sampling._checkpoint import _batch_checksum

        batch = _make_batch(device=device)
        runner = _make_runner(device)
        runner.run(batch, n_steps=4)
        path = tmp_path / "ck.zarr"
        runner.checkpoint(path)

        root = zarr.open_group(str(path), mode="r")
        _, _, manifest = read_checkpoint(path, device)
        assert _batch_checksum(root) == manifest.batch_checksum

    def test_intact_checkpoint_still_restores(self, tmp_path, device: str) -> None:
        """The guard must not reject a healthy store."""
        batch = _make_batch(device=device)
        runner = _make_runner(device)
        runner.run(batch, n_steps=4)
        path = tmp_path / "ck.zarr"
        runner.checkpoint(path)
        restored, _, _ = read_checkpoint(path, device)
        assert restored.num_graphs == 2

    def test_no_pickle_in_store(self, tmp_path, device: str) -> None:
        """A checkpoint must not be executable on load."""
        batch = _make_batch(device=device)
        runner = _make_runner(device)
        runner.run(batch, n_steps=4)
        path = tmp_path / "ck.zarr"
        runner.checkpoint(path)
        assert not list(path.rglob("*.pkl"))
        assert not list(path.rglob("*.pt"))


# ===========================================================================
# 5. Runner checkpoint / restore
# ===========================================================================


class TestRunnerCheckpointRestore:
    """The user-facing API and its guard rails."""

    def test_non_boundary_checkpoint_names_next_valid_step(
        self, tmp_path, device: str
    ) -> None:
        batch = _make_batch(device=device)
        runner = _make_runner(device, steps_per_epoch=4)
        runner.run(batch, n_steps=3)
        with pytest.raises(ValueError) as excinfo:
            runner.checkpoint(tmp_path / "ck.zarr")
        message = str(excinfo.value)
        assert "next valid checkpoint step is 4" in message

    def test_checkpoint_without_batch_raises(self, tmp_path, device: str) -> None:
        runner = _make_runner(device)
        with pytest.raises(RuntimeError, match="no batch to save"):
            runner.checkpoint(tmp_path / "ck.zarr")

    def test_walker_identity_survives(self, tmp_path, device: str) -> None:
        """AtomicDataZarrWriter drops unknown fields; identity must not be lost."""
        batch = _make_batch(n_graphs=3, device=device)
        batch["thermodynamic_state_id"] = torch.tensor([2, 0, 1], device=device)
        runner = _make_runner(device, n_states=3)
        batch = runner.run(batch, n_steps=4)
        walker_ids = batch.walker_id.reshape(-1).tolist()

        path = tmp_path / "ck.zarr"
        runner.checkpoint(path)

        runner2 = _make_runner(device, n_states=3)
        restored = runner2.restore(path)
        assert restored.walker_id.reshape(-1).tolist() == walker_ids
        assert restored.thermodynamic_state_id.reshape(-1).tolist() == [2, 0, 1]

    def test_exact_trajectory_reproduction(self, tmp_path, device: str) -> None:
        """The point of exact restart: identical trajectory after resuming."""
        batch = _make_batch(device=device)
        runner = _make_runner(device)
        batch = runner.run(batch, n_steps=4)
        path = tmp_path / "ck.zarr"
        runner.checkpoint(path)

        batch = runner.run(batch, n_steps=4, prime=False)
        reference = batch.positions.clone()

        runner2 = _make_runner(device)
        resumed = runner2.restore(path)
        resumed = runner2.run(resumed, n_steps=4, prime=False)

        assert torch.allclose(reference, resumed.positions, atol=1e-6), (
            f"max deviation {float((reference - resumed.positions).abs().max())}"
        )

    def test_adaptive_bias_history_restored(self, tmp_path, device: str) -> None:
        batch = _make_batch(device=device)
        dynamics = _make_dynamics(device)
        bias = _CountingBias()
        runner = EnhancedSampling(dynamics, {"counter": bias}, steps_per_epoch=4)
        runner.run(batch, n_steps=4)
        assert float(bias.deposits) == 4
        assert bias.state_version == 4

        path = tmp_path / "ck.zarr"
        runner.checkpoint(path)

        bias2 = _CountingBias()
        runner2 = EnhancedSampling(
            _make_dynamics(device), {"counter": bias2}, steps_per_epoch=4
        )
        runner2.restore(path)
        assert float(bias2.deposits) == 4, "bias buffer not restored"
        assert bias2.state_version == 4, "bias history version not restored"

    def test_restore_rejects_different_bias_set(self, tmp_path, device: str) -> None:
        batch = _make_batch(device=device)
        runner = _make_runner(device)
        runner.run(batch, n_steps=4)
        path = tmp_path / "ck.zarr"
        runner.checkpoint(path)

        other = EnhancedSampling(_make_dynamics(device), {}, steps_per_epoch=4)
        with pytest.raises(ValueError, match="different configuration"):
            other.restore(path)

    def test_restore_error_mentions_weights_are_not_restored(
        self, tmp_path, device: str
    ) -> None:
        batch = _make_batch(device=device)
        runner = _make_runner(device)
        runner.run(batch, n_steps=4)
        path = tmp_path / "ck.zarr"
        runner.checkpoint(path)

        other = EnhancedSampling(_make_dynamics(device), {}, steps_per_epoch=4)
        with pytest.raises(ValueError, match="weights"):
            other.restore(path)

    def test_warm_start_after_restore_raises(self, tmp_path, device: str) -> None:
        """The two are mutually exclusive; replaying over a restore corrupts it."""
        batch = _make_batch(device=device)
        runner = _make_runner(device)
        runner.run(batch, n_steps=4)
        path = tmp_path / "ck.zarr"
        runner.checkpoint(path)

        runner2 = _make_runner(device)
        runner2.restore(path)
        with pytest.raises(RuntimeError, match="mutually exclusive"):
            runner2.warm_start(_make_batch(device=device))

    def test_restore_primes_forces(self, tmp_path, device: str) -> None:
        batch = _make_batch(device=device)
        runner = _make_runner(device)
        runner.run(batch, n_steps=4)
        path = tmp_path / "ck.zarr"
        runner.checkpoint(path)

        runner2 = _make_runner(device)
        restored = runner2.restore(path)
        assert torch.count_nonzero(restored.forces) > 0
        assert runner2.last_outputs, "restore did not prime forces"

    def test_checkpoint_drains_the_epoch_commit(self, tmp_path, device: str) -> None:
        """Boundary-aligned is not quiescent unless the commit has run.

        commit_epoch() normally fires lazily, when the *next* step notices
        the epoch advanced. At step N that has not happened, so without an
        explicit drain the checkpoint records a shared-history bias with its
        deposits still pending rather than merged.
        """
        batch = _make_batch(device=device)
        bias = _SharedHistoryBias()
        runner = EnhancedSampling(
            _make_dynamics(device), {"shared": bias}, steps_per_epoch=4
        )
        runner.run(batch, n_steps=4)
        assert float(bias.pending) == 4, "precondition: commit has not run yet"
        assert float(bias.published) == 0

        path = tmp_path / "ck.zarr"
        runner.checkpoint(path)

        _, states, _ = read_checkpoint(path, device)
        recorded = states["biases/shared"]
        assert float(recorded["published"]) == 4, "checkpoint captured pre-commit state"
        assert float(recorded["pending"]) == 0

    def test_commit_is_not_run_twice(self, tmp_path, device: str) -> None:
        """Draining at checkpoint must not double-count against the lazy path.

        A shared-history bias that merged its pending deposits twice would
        silently double them.
        """
        batch = _make_batch(device=device)
        bias = _SharedHistoryBias()
        runner = EnhancedSampling(
            _make_dynamics(device), {"shared": bias}, steps_per_epoch=4
        )
        batch = runner.run(batch, n_steps=4)
        runner.checkpoint(tmp_path / "a.zarr")
        assert bias.commit_calls == 1

        # Continuing crosses into epoch 1. Its first step observes the epoch
        # change and takes the lazy path for epoch 0 — which must be a no-op,
        # since the checkpoint already drained it.
        batch = runner.run(batch, n_steps=4, prime=False)
        assert bias.commit_calls == 1, (
            f"epoch 0 was committed {bias.commit_calls} times"
        )
        assert float(bias.published) == 4

        # Epoch 1 completes and is drained by the next checkpoint.
        runner.checkpoint(tmp_path / "b.zarr")
        assert bias.commit_calls == 2
        assert float(bias.published) == 8

    def test_repeated_checkpoint_commits_once(self, tmp_path, device: str) -> None:
        batch = _make_batch(device=device)
        bias = _SharedHistoryBias()
        runner = EnhancedSampling(
            _make_dynamics(device), {"shared": bias}, steps_per_epoch=4
        )
        runner.run(batch, n_steps=4)
        runner.checkpoint(tmp_path / "a.zarr")
        runner.checkpoint(tmp_path / "b.zarr")
        assert bias.commit_calls == 1
        assert float(bias.published) == 4

    def test_committed_epoch_survives_restore(self, tmp_path, device: str) -> None:
        """A resumed run must not re-commit an epoch the checkpoint drained."""
        batch = _make_batch(device=device)
        bias = _SharedHistoryBias()
        runner = EnhancedSampling(
            _make_dynamics(device), {"shared": bias}, steps_per_epoch=4
        )
        runner.run(batch, n_steps=4)
        path = tmp_path / "ck.zarr"
        runner.checkpoint(path)

        bias2 = _SharedHistoryBias()
        runner2 = EnhancedSampling(
            _make_dynamics(device), {"shared": bias2}, steps_per_epoch=4
        )
        resumed = runner2.restore(path)
        assert bias2.commit_calls == 0, "restore re-ran a committed epoch"
        assert float(bias2.published) == 4

        # Crossing into epoch 1 must not re-commit epoch 0 either.
        resumed = runner2.run(resumed, n_steps=4, prime=False)
        assert bias2.commit_calls == 0
        assert float(bias2.published) == 4

        # Epoch 1 is new, so its drain does fire.
        runner2.checkpoint(tmp_path / "next.zarr")
        assert bias2.commit_calls == 1
        assert float(bias2.published) == 8

    def test_checkpoint_at_step_zero_does_not_commit(
        self, tmp_path, device: str
    ) -> None:
        """No epoch has completed at step 0, so there is nothing to drain."""
        batch = _make_batch(device=device)
        bias = _SharedHistoryBias()
        runner = EnhancedSampling(
            _make_dynamics(device), {"shared": bias}, steps_per_epoch=4
        )
        runner.prime_forces(batch)
        runner.checkpoint(tmp_path / "ck.zarr")
        assert bias.commit_calls == 0

    def test_checkpoint_at_step_zero_is_a_boundary(self, tmp_path, device: str) -> None:
        batch = _make_batch(device=device)
        runner = _make_runner(device)
        runner.prime_forces(batch)
        runner.checkpoint(tmp_path / "ck.zarr")  # step 0 % N == 0
