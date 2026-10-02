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
"""Unit tests for the enhanced-sampling hook family and the strategy surface.

The behaviour these hooks produce in a run is covered in ``test_runner.py``,
``test_exchange.py`` and ``test_checkpoint.py``.  What is asserted here is the
part that only shows up from outside: what ``build_hooks()`` returns and in
what order, that the two boundary hooks carry their cadence as
``Hook.frequency`` rather than re-deriving it, that both stay idempotent by
index, and that ``BiasHook`` presents the ``StatefulHook`` surface it claims.
"""

from __future__ import annotations

import pytest
import torch

from nvalchemi.data import AtomicData, Batch
from nvalchemi.dynamics import NVE, NVTLangevin, SizeAwareSampler
from nvalchemi.dynamics.base import DynamicsStage
from nvalchemi.dynamics.hooks import PairSwapHook
from nvalchemi.enhanced_sampling import (
    BiasHook,
    ConservativeBias,
    EnhancedSampling,
    EpochCommitHook,
    ReplicaExchange,
    ThermodynamicState,
    WalkerIdentityHook,
)
from nvalchemi.hooks import CheckpointableHook, Hook
from nvalchemi.models.demo import DemoModel, DemoModelWrapper

_ENGINE_KWARGS = {"dt": 0.1, "temperature": 300.0, "friction": 0.1}


def _make_batch(n_graphs: int = 2, atoms_per_graph: int = 4) -> Batch:
    """Return a batch with the output buffers dynamics writes back into."""
    torch.manual_seed(0)
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
    return Batch.from_data_list(data_list)


def _make_model() -> DemoModelWrapper:
    """Return the potential the engine calls."""
    return DemoModelWrapper(DemoModel())


class _ConstantForceBias(ConservativeBias):
    """E = c * sum(x) — a constant, known bias force of -c along x."""

    def __init__(self, coefficient: float = 1.0, name: str = "cf") -> None:
        super().__init__(name=name)
        self.coefficient = coefficient

    def energy(self, current: Batch) -> torch.Tensor:
        """Return the per-graph bias energy."""
        ptr = current.batch_ptr
        return torch.stack(
            [
                self.coefficient * current.positions[ptr[b] : ptr[b + 1], 0].sum()
                for b in range(current.num_graphs)
            ]
        ).unsqueeze(-1)


class _CountingCommitTarget:
    """A stand-in stateful hook that only records how often it committed."""

    def __init__(self) -> None:
        self.commits = 0

    def commit(self) -> None:
        """Record one commit."""
        self.commits += 1


class _Ctx:
    """Minimal stand-in for the dynamics hook context."""

    def __init__(self, batch: Batch | None, step: int) -> None:
        self.batch = batch
        self.step_count = step


def _context(step: int, batch: Batch | None = None) -> _Ctx:
    """Return a hook context carrying *step*."""
    return _Ctx(batch, step)


class _RecordingEngine:
    """Stand-in for the engine a swap rebinds parameters on."""

    def apply_per_system_params(self, params: object, batch: Batch) -> None:
        """Accept the rebinding without doing anything."""


def _ladder(n: int = 2) -> list[ThermodynamicState]:
    """Return an *n*-rung temperature ladder."""
    return [
        ThermodynamicState(state_id=i, temperature=300.0 * 1.2**i) for i in range(n)
    ]


# ===========================================================================
# 1. What build_hooks() contributes
# ===========================================================================


class TestBuildHooks:
    """The strategy contributes hooks; that list is its whole contribution."""

    def test_order_is_identity_bias_exchange_commit(self) -> None:
        """Exchange precedes commit: a commit under stale labels publishes wrong."""
        exchange = ReplicaExchange(_ladder(2), torch.arange(2), attempt_interval=3)
        sampling = EnhancedSampling(
            engine=NVTLangevin,
            engine_kwargs=_ENGINE_KWARGS,
            biases={},
            replica_exchange=exchange,
        )
        kinds = [type(hook) for hook in sampling.build_hooks()]
        assert kinds == [
            WalkerIdentityHook,
            BiasHook,
            PairSwapHook,
            EpochCommitHook,
        ]

    def test_exchange_hook_absent_without_a_ladder(self) -> None:
        sampling = EnhancedSampling(engine=NVTLangevin, engine_kwargs=_ENGINE_KWARGS)
        kinds = [type(hook) for hook in sampling.build_hooks()]
        assert PairSwapHook not in kinds
        assert kinds == [WalkerIdentityHook, BiasHook, EpochCommitHook]

    def test_extra_hooks_come_last(self) -> None:
        """A caller's force clamp must act on the total, so the bias goes first."""

        class _NoopHook:
            stage = DynamicsStage.AFTER_COMPUTE
            frequency = 1

            def __call__(self, ctx: object, stage: DynamicsStage) -> None:
                pass

        extra = _NoopHook()
        sampling = EnhancedSampling(
            engine=NVTLangevin, engine_kwargs=_ENGINE_KWARGS, extra_hooks=[extra]
        )
        hooks = sampling.build_hooks()
        assert hooks[-1] is extra
        assert hooks.index(sampling._bias_hook) < hooks.index(extra)

    def test_every_contributed_hook_satisfies_the_protocol(self) -> None:
        exchange = ReplicaExchange(_ladder(2), torch.arange(2), attempt_interval=3)
        sampling = EnhancedSampling(
            engine=NVTLangevin,
            engine_kwargs=_ENGINE_KWARGS,
            biases={},
            replica_exchange=exchange,
        )
        for hook in sampling.build_hooks():
            assert isinstance(hook, Hook), hook

    def test_engine_registers_exactly_the_contributed_hooks(self) -> None:
        sampling = EnhancedSampling(engine=NVTLangevin, engine_kwargs=_ENGINE_KWARGS)
        engine = sampling.dynamics(_make_model())
        assert engine.hooks == sampling.build_hooks()

    def test_dynamics_is_cached(self) -> None:
        """Consecutive run() calls must continue one trajectory, not restart it."""
        model = _make_model()
        sampling = EnhancedSampling(engine=NVTLangevin, engine_kwargs=_ENGINE_KWARGS)
        assert sampling.dynamics(model) is sampling.dynamics(model)


# ===========================================================================
# 2. The cadence is Hook.frequency
# ===========================================================================


class TestCadence:
    """The registry gates the boundary hooks; nothing re-derives the boundary."""

    def test_epoch_hook_frequency_is_steps_per_epoch(self) -> None:
        sampling = EnhancedSampling(
            engine=NVTLangevin, engine_kwargs=_ENGINE_KWARGS, steps_per_epoch=7
        )
        assert sampling._epoch_hook.frequency == 7

    def test_exchange_hook_frequency_is_attempt_interval(self) -> None:
        exchange = ReplicaExchange(_ladder(2), torch.arange(2), attempt_interval=5)
        sampling = EnhancedSampling(
            engine=NVTLangevin,
            engine_kwargs=_ENGINE_KWARGS,
            replica_exchange=exchange,
        )
        assert sampling._exchange_hook.frequency == 5

    @pytest.mark.parametrize("frequency", [1, 2, 5])
    def test_dispatch_at_kN_commits_epoch_k_minus_one(self, frequency: int) -> None:
        """``step // N - 1`` is the boundary that has just completed."""
        target = _CountingCommitTarget()
        hook = EpochCommitHook([target], frequency=frequency)
        for k in range(4):
            hook(_context(step=k * frequency), DynamicsStage.BEFORE_STEP)
        # k = 0 commits epoch -1, which is a no-op; the other three land.
        assert hook.committed_epoch == 2
        assert target.commits == 3

    def test_commit_at_step_zero_is_a_no_op(self) -> None:
        target = _CountingCommitTarget()
        hook = EpochCommitHook([target], frequency=4)
        hook(_context(step=0), DynamicsStage.BEFORE_STEP)
        assert target.commits == 0
        assert hook.committed_epoch == -1


# ===========================================================================
# 3. Idempotence by index
# ===========================================================================


class TestIdempotence:
    """A checkpoint drains the same boundary the cadence does; neither may double."""

    def test_epoch_is_committed_at_most_once(self) -> None:
        target = _CountingCommitTarget()
        hook = EpochCommitHook([target], frequency=4)
        hook.commit_epoch(0)
        hook.commit_epoch(0)
        hook.commit_epoch(-1)
        assert target.commits == 1

    def test_commit_skips_backwards_indices(self) -> None:
        target = _CountingCommitTarget()
        hook = EpochCommitHook([target], frequency=4)
        hook.commit_epoch(3)
        hook.commit_epoch(1)
        assert target.commits == 1
        assert hook.committed_epoch == 3

    def test_segment_is_attempted_at_most_once(self) -> None:
        exchange = ReplicaExchange(_ladder(2), torch.arange(2), attempt_interval=2)
        hook = exchange.swap_hook()
        decided: list[int] = []
        hook._attempt = lambda batch, segment: decided.append(segment)  # type: ignore[method-assign]
        hook.attempt_segment(None, -1)
        hook.attempt_segment(None, 0)
        hook.attempt_segment(None, 0)
        hook.attempt_segment(None, 1)
        assert decided == [0, 1]
        assert hook.attempted_segment == 1


# ===========================================================================
# 4. BiasHook is the StatefulHook the biases dispatch through
# ===========================================================================


class TestBiasHookProtocol:
    """It claims the StatefulHook surface, so assert every member of it."""

    def test_carries_the_stateful_hook_members(self) -> None:
        hook = BiasHook({})
        assert hook.read_only is False
        assert hook.frequency == 1
        assert hook.stage is None  # _runs_on_stage decides instead
        assert isinstance(hook, Hook)
        assert isinstance(hook, CheckpointableHook)

    def test_runs_on_the_two_force_step_stages_only(self) -> None:
        hook = BiasHook({})
        assert hook._runs_on_stage(DynamicsStage.AFTER_COMPUTE)
        assert hook._runs_on_stage(DynamicsStage.AFTER_STEP)
        assert not hook._runs_on_stage(DynamicsStage.BEFORE_STEP)
        assert not hook._runs_on_stage(DynamicsStage.BEFORE_COMPUTE)

    def test_commit_fans_out_to_adaptive_biases_only(self) -> None:
        """A static bias has no commit; asking it for one must not raise."""
        hook = BiasHook({"cf": _ConstantForceBias(name="cf")})
        hook.commit()  # must not raise
        assert hook.adaptive_biases() == {}

    def test_delivery_record_round_trips(self) -> None:
        hook = BiasHook({})
        hook._last_update_step = {"a": 3}
        other = BiasHook({})
        other.load_state_dict(hook.state_dict())
        assert other._last_update_step == {"a": 3}

    def test_an_empty_bias_set_applies_nothing(self) -> None:
        batch = _make_batch()
        before = batch.forces.clone()
        BiasHook({}).evaluate_and_apply(batch)
        assert torch.equal(batch.forces, before)


# ===========================================================================
# 5. Walker identity, without a dynamics behind it
# ===========================================================================


class TestWalkerIdentityHook:
    """``stamp`` is public because priming needs it without a dispatch."""

    def test_walker_ids_are_issued_once_and_advance(self) -> None:
        hook = WalkerIdentityHook(steps_per_epoch=4)
        first, second = _make_batch(n_graphs=2), _make_batch(n_graphs=3)
        hook.stamp(first, 0)
        hook.stamp(second, 0)
        assert first.walker_id.reshape(-1).tolist() == [0, 1]
        assert second.walker_id.reshape(-1).tolist() == [2, 3, 4]
        assert hook.next_walker_id == 5

    def test_counters_are_refreshed_every_stamp(self) -> None:
        hook = WalkerIdentityHook(steps_per_epoch=4)
        batch = _make_batch(n_graphs=1)
        hook.stamp(batch, 9)
        assert batch.sampling_step.reshape(-1).tolist() == [9]
        assert batch.sampling_epoch.reshape(-1).tolist() == [2]
        # No ladder: the segment divisor falls back to the epoch length.
        assert batch.exchange_segment.reshape(-1).tolist() == [2]

    def test_segment_uses_the_attempt_interval_when_there_is_a_ladder(self) -> None:
        exchange = ReplicaExchange(_ladder(2), torch.arange(2), attempt_interval=3)
        hook = WalkerIdentityHook(steps_per_epoch=100, exchange=exchange)
        batch = _make_batch(n_graphs=2)
        hook.stamp(batch, 7)
        assert batch.exchange_segment.reshape(-1).tolist() == [2, 2]

    def test_sentinel_rows_are_given_fresh_identities(self) -> None:
        """A refill writes ``-1`` for rows it has no value for.

        Those rows are new walkers.  Handing them the id of whoever just
        graduated out of the slot would give a replacement that walker's
        per-walker bias history.
        """
        hook = WalkerIdentityHook(steps_per_epoch=4)
        batch = _make_batch(n_graphs=3)
        hook.stamp(batch, 0)
        assert batch.walker_id.reshape(-1).tolist() == [0, 1, 2]

        # Rows 1 and 2 graduated; the registry rebuilt the column with the
        # sentinel for the replacements that took their slots.
        batch["walker_id"] = torch.tensor([0, -1, -1])
        hook.stamp(batch, 1)
        assert batch.walker_id.reshape(-1).tolist() == [0, 3, 4]
        assert hook.next_walker_id == 5

    def test_identities_are_never_reused_across_refills(self) -> None:
        hook = WalkerIdentityHook(steps_per_epoch=4)
        batch = _make_batch(n_graphs=3)
        hook.stamp(batch, 0)
        ever = list(batch.walker_id.reshape(-1).tolist())
        for step in range(1, 4):
            kept = int(batch.walker_id.reshape(-1)[0])
            batch["walker_id"] = torch.tensor([kept, -1, -1])
            hook.stamp(batch, step)
            ever += batch.walker_id.reshape(-1).tolist()[1:]
        assert len(set(ever)) == len(ever), f"an identity was reused: {ever}"

    def test_a_fully_assigned_column_is_left_alone(self) -> None:
        """Immutable identity: nothing is re-stamped once it has one."""
        hook = WalkerIdentityHook(steps_per_epoch=4)
        batch = _make_batch(n_graphs=2)
        batch["walker_id"] = torch.tensor([7, 9])
        hook.stamp(batch, 0)
        assert batch.walker_id.reshape(-1).tolist() == [7, 9]

    def test_the_counter_clears_identities_the_caller_supplied(self) -> None:
        """Otherwise the first refill allocates straight into them."""
        hook = WalkerIdentityHook(steps_per_epoch=4)
        batch = _make_batch(n_graphs=2)
        batch["walker_id"] = torch.tensor([7, 9])
        hook.stamp(batch, 0)
        assert hook.next_walker_id == 10

        batch["walker_id"] = torch.tensor([7, -1])
        hook.stamp(batch, 1)
        assert batch.walker_id.reshape(-1).tolist() == [7, 10]

    def test_a_mix_of_supplied_and_sentinel_clears_the_supplied_ones(self) -> None:
        hook = WalkerIdentityHook(steps_per_epoch=4)
        batch = _make_batch(n_graphs=3)
        batch["walker_id"] = torch.tensor([-1, 42, -1])
        hook.stamp(batch, 0)
        assert batch.walker_id.reshape(-1).tolist() == [43, 42, 44]

    @pytest.mark.parametrize("key", ["walker_id", "thermodynamic_state_id"])
    def test_the_registered_factory_writes_the_sentinel(self, key: str) -> None:
        """The registry and the hook have to agree on what "unset" is.

        Both fields are *allocated*, not defaulted: an identity has to be
        fresh and a rung has to be vacant, and neither is something a static
        per-row factory can know.
        """
        from nvalchemi.dynamics.base import BaseDynamics

        factory = BaseDynamics._bookkeeping_keys[key]
        assert factory(3, torch.device("cpu")).reshape(-1).tolist() == [-1, -1, -1]

    def test_counters_keep_a_plain_default(self) -> None:
        """They are a function of the step and rewritten on every stamp."""
        from nvalchemi.dynamics.base import BaseDynamics

        for key in ("sampling_step", "sampling_epoch", "exchange_segment"):
            factory = BaseDynamics._bookkeeping_keys[key]
            assert factory(2, torch.device("cpu")).reshape(-1).tolist() == [0, 0]

    def test_a_sentinel_state_without_a_ladder_becomes_zero(self) -> None:
        """With no ladder the field is a label, not a rung."""
        hook = WalkerIdentityHook(steps_per_epoch=4)
        batch = _make_batch(n_graphs=3)
        batch["thermodynamic_state_id"] = torch.tensor([0, -1, -1])
        hook.stamp(batch, 0)
        assert batch.thermodynamic_state_id.reshape(-1).tolist() == [0, 0, 0]


class TestStateAssignmentThroughRefill:
    """A graduating walker vacates a rung; its replacement must take it."""

    @staticmethod
    def _hook(n_states: int = 4) -> WalkerIdentityHook:
        """Return an identity hook over an *n_states*-rung ladder."""
        exchange = ReplicaExchange(_ladder(n_states), torch.arange(n_states))
        return WalkerIdentityHook(steps_per_epoch=100, exchange=exchange)

    def test_replacements_take_the_vacated_rungs(self) -> None:
        """The only assignment that keeps the ladder a bijection.

        Defaulting them to state 0 claims a rung another walker already holds
        and leaves the vacated one empty, which is not a ladder replica
        exchange can pair on.
        """
        hook = self._hook()
        batch = _make_batch(n_graphs=4)
        hook.stamp(batch, 0)
        assert batch.thermodynamic_state_id.reshape(-1).tolist() == [0, 1, 2, 3]

        # Swaps moved the labels; then the holders of rungs 0 and 3 graduated.
        batch["thermodynamic_state_id"] = torch.tensor([2, 1, -1, -1])
        hook.stamp(batch, 1)
        assignment = batch.thermodynamic_state_id.reshape(-1).tolist()
        assert assignment == [2, 1, 0, 3]
        assert sorted(assignment) == [0, 1, 2, 3], "the bijection was broken"

    def test_survivors_keep_the_rung_they_held(self) -> None:
        hook = self._hook()
        batch = _make_batch(n_graphs=4)
        batch["thermodynamic_state_id"] = torch.tensor([3, -1, 1, -1])
        hook.stamp(batch, 0)
        assignment = batch.thermodynamic_state_id.reshape(-1).tolist()
        assert assignment[0] == 3 and assignment[2] == 1
        assert sorted(assignment) == [0, 1, 2, 3]

    def test_a_batch_outgrowing_the_ladder_is_named(self) -> None:
        hook = self._hook(n_states=3)
        batch = _make_batch(n_graphs=4)
        batch["thermodynamic_state_id"] = torch.tensor([0, 1, -1, -1])
        with pytest.raises(ValueError, match="outgrown the ladder"):
            hook.stamp(batch, 0)

    def test_the_refilled_assignment_is_revalidated(self) -> None:
        """Membership changing is the one event that can break a valid one."""
        hook = self._hook()
        batch = _make_batch(n_graphs=4)
        hook.stamp(batch, 0)
        # A duplicate among the survivors cannot be repaired by filling.
        batch["thermodynamic_state_id"] = torch.tensor([1, 1, -1, -1])
        with pytest.raises(ValueError, match="must be a permutation"):
            hook.stamp(batch, 1)

    def test_a_swap_still_runs_after_a_refill(self) -> None:
        """The point of restoring the bijection: the ladder stays pairable."""
        exchange = ReplicaExchange(_ladder(4), torch.arange(4), attempt_interval=2)
        hook = WalkerIdentityHook(steps_per_epoch=100, exchange=exchange)
        batch = _make_batch(n_graphs=4)
        hook.stamp(batch, 0)
        batch["thermodynamic_state_id"] = torch.tensor([2, 1, -1, -1])
        hook.stamp(batch, 1)

        swap = exchange.swap_hook()
        swap.on_register(_RecordingEngine())
        swap.attempt_segment(batch, 0)  # must not raise
        assert sorted(batch.thermodynamic_state_id.reshape(-1).tolist()) == [0, 1, 2, 3]

    def test_the_stamped_batch_is_what_checkpoint_would_save(self) -> None:
        hook = WalkerIdentityHook(steps_per_epoch=4)
        batch = _make_batch()
        assert hook.current_batch is None
        hook.stamp(batch, 0)
        assert hook.current_batch is batch


# ===========================================================================
# 6. Identity across a real refill
# ===========================================================================


class TestIdentityThroughRefill:
    """The path the bookkeeping registry exists for, driven end to end."""

    class _Dataset:
        """Minimal dataset the size-aware sampler can refill from."""

        def __init__(self, count: int, atoms: int = 2) -> None:
            self.count = count
            self.atoms = atoms

        def __len__(self) -> int:
            """Return the sample count."""
            return self.count

        def get_metadata(self, index: int) -> tuple[int, int]:
            """Return ``(num_atoms, num_edges)`` without loading."""
            return (self.atoms, 0)

        def __getitem__(self, index: int) -> tuple[AtomicData, dict]:
            """Return one sample and an empty metadata dict."""
            data = AtomicData(
                positions=torch.zeros(self.atoms, 3),
                atomic_numbers=torch.full((self.atoms,), 6, dtype=torch.long),
            )
            return data, {}

    def test_replacements_never_inherit_a_graduated_identity(self) -> None:
        """A replacement taking a slot must not take the identity with it.

        ``walker_id`` keys a per-walker metadynamics history, so an id reused
        across a graduation hands a fresh configuration the hills deposited by
        the walker that just left.
        """
        from nvalchemi.dynamics.base import BaseDynamics
        from nvalchemi.dynamics.sinks import HostMemory

        sampler = SizeAwareSampler(
            self._Dataset(30), max_atoms=20, max_edges=10, max_batch_size=3
        )
        engine = BaseDynamics(
            model=DemoModelWrapper(DemoModel()),
            sampler=sampler,
            sinks=[HostMemory(capacity=200)],
            device_type="cpu",
        )
        batch = sampler.build_initial_batch()
        batch["forces"] = torch.zeros(batch.num_nodes, 3)
        batch["energy"] = torch.zeros(batch.num_graphs, 1)

        hook = WalkerIdentityHook(steps_per_epoch=1000)
        hook.stamp(batch, 0)
        ever = list(batch.walker_id.reshape(-1).tolist())

        for step in range(1, 4):
            survivor = int(batch.walker_id.reshape(-1)[0])
            batch["status"] = torch.tensor([[0], [1], [1]])
            batch = engine.refill_check(batch, exit_status=1)
            assert batch is not None
            hook.stamp(batch, step)
            ids = batch.walker_id.reshape(-1).tolist()
            assert ids[0] == survivor, "a surviving walker lost its identity"
            ever += ids[1:]

        assert len(set(ever)) == len(ever), f"an identity was reused: {ever}"

    def test_the_ladder_stays_a_bijection_across_a_refill(self) -> None:
        """The state half of the same path, driven through ``refill_check``.

        The registry rebuilds ``thermodynamic_state_id`` for the whole batch,
        so a default-valued factory gives every replacement rung 0 — two
        walkers on one rung, and the rungs the graduates vacated held by
        nobody.  Replica exchange cannot pair on that.
        """
        from nvalchemi.dynamics.base import BaseDynamics
        from nvalchemi.dynamics.sinks import HostMemory

        sampler = SizeAwareSampler(
            self._Dataset(30), max_atoms=20, max_edges=10, max_batch_size=3
        )
        engine = BaseDynamics(
            model=DemoModelWrapper(DemoModel()),
            sampler=sampler,
            sinks=[HostMemory(capacity=200)],
            device_type="cpu",
        )
        batch = sampler.build_initial_batch()
        batch["forces"] = torch.zeros(batch.num_nodes, 3)
        batch["energy"] = torch.zeros(batch.num_graphs, 1)

        exchange = ReplicaExchange(_ladder(3), torch.arange(3), attempt_interval=2)
        hook = WalkerIdentityHook(steps_per_epoch=1000, exchange=exchange)
        hook.stamp(batch, 0)
        assert sorted(batch.thermodynamic_state_id.reshape(-1).tolist()) == [0, 1, 2]

        for step in range(1, 4):
            batch["status"] = torch.tensor([[0], [1], [1]])
            batch = engine.refill_check(batch, exit_status=1)
            assert batch is not None
            hook.stamp(batch, step)
            assignment = batch.thermodynamic_state_id.reshape(-1).tolist()
            assert sorted(assignment) == [0, 1, 2], (
                f"refill {step} left the ladder as {assignment}"
            )


# ===========================================================================
# 7. The declarative half of the strategy
# ===========================================================================


class TestStrategySurface:
    """What survives to a spec, and what is refused before an engine exists."""

    def test_spec_carries_the_knobs_and_omits_live_objects(self) -> None:
        exchange = ReplicaExchange(_ladder(2), torch.arange(2), attempt_interval=3)
        spec = EnhancedSampling(
            engine=NVTLangevin,
            engine_kwargs=_ENGINE_KWARGS,
            biases={},
            steps_per_epoch=32,
            compile_biases=False,
            prime_after_update=False,
            replica_exchange=exchange,
            n_steps=17,
        ).to_spec_dict()
        assert spec["engine"].endswith("NVTLangevin")
        assert spec["engine_kwargs"] == _ENGINE_KWARGS
        assert spec["n_steps"] == 17
        assert spec["steps_per_epoch"] == 32
        assert spec["prime_after_update"] is False
        assert "biases" not in spec
        assert "replica_exchange" not in spec
        assert "extra_hooks" not in spec

    def test_exchange_capability_is_checked_against_the_engine_class(self) -> None:
        """A strategy holds a recipe, so there is no instance left to probe."""
        exchange = ReplicaExchange(_ladder(2), torch.arange(2))
        with pytest.raises(TypeError, match="replica exchange needs"):
            EnhancedSampling(
                engine=NVE, engine_kwargs={"dt": 0.1}, replica_exchange=exchange
            )

    def test_state_that_needs_an_engine_says_so(self) -> None:
        sampling = EnhancedSampling(engine=NVTLangevin, engine_kwargs=_ENGINE_KWARGS)
        with pytest.raises(RuntimeError, match="no engine has been built"):
            sampling.checkpoint("unused.zarr")
