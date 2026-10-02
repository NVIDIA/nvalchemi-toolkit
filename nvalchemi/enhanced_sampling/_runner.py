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
"""``EnhancedSampling``: the ``DynamicsStrategy`` that runs biased dynamics.

A biased run differs from a plain one only in *what it configures*: the same
engine, plus hooks for walker identity, bias forces, shared-history commits
and replica-exchange attempts.  That is exactly what a
:class:`~nvalchemi.dynamics.DynamicsStrategy` expresses, so this
module contributes a :meth:`~EnhancedSampling.build_hooks` override and the
restart machinery around it — never a second stepping loop.  The hooks
themselves live in :mod:`nvalchemi.enhanced_sampling.hooks`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from pydantic import Field, PrivateAttr, model_validator

from nvalchemi._checkpoint import (
    CheckpointManifest,
    Stateful,
    _qualified_name,
    load_checkpoint,
    save_checkpoint,
)
from nvalchemi.dynamics.base import BaseDynamics
from nvalchemi.dynamics.hooks.swap import PairSwapHook
from nvalchemi.dynamics.strategy import DynamicsStrategy
from nvalchemi.enhanced_sampling._exchange import ReplicaExchange
from nvalchemi.enhanced_sampling.hooks import (
    BiasHook,
    EpochCommitHook,
    WalkerIdentityHook,
)
from nvalchemi.models.base import BaseModelMixin

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from nvalchemi.data import Batch
    from nvalchemi.hooks import Hook

__all__ = ["EnhancedSampling"]

# Graph-level fields WalkerIdentityHook stamps that AtomicDataZarrWriter does
# not recognise.  Without naming them to save_checkpoint they are dropped
# without complaint, and a "restored" run comes back with fresh ids and default
# state assignments.  The counters are not here: they are a function of the
# step count, which the manifest carries.
_IDENTITY_FIELDS = ("walker_id", "thermodynamic_state_id")


def _register_identity_bookkeeping() -> None:
    """Make the walker identity fields survive refill and graduation.

    ``BaseDynamics._bookkeeping_keys`` is the mechanism ``status`` and
    ``system_id`` already use: ``refill_check`` rebuilds every registered key
    after graduated graphs are dropped and replacements appended, so the field
    keeps its per-graph meaning across a change of batch membership.

    Without registration the identity fields exist only because
    :class:`~nvalchemi.enhanced_sampling.hooks.WalkerIdentityHook` rewrites
    them each step, which is not the same guarantee: a refill that drops row 2
    and appends a replacement leaves the surviving rows' history attached to a
    ``walker_id`` the registry never restored.  Walker identity is documented
    as "immutable identity that follows a physical configuration" (§4), and
    that claim needs the registry behind it.

    ``walker_id`` registers a ``-1`` sentinel rather than a default value, the
    way ``system_id`` does.  The factory runs for the *whole* batch, including
    the replacement rows, and an identity is **allocated** rather than
    defaulted: a factory returning ``arange(n)`` hands each replacement the id
    of whichever walker just graduated out of that slot, so a per-walker
    metadynamics history is inherited by an unrelated configuration — and it
    does so on every refill, not occasionally.  The sentinel says "no identity
    yet" and :meth:`WalkerIdentityHook.stamp` allocates a fresh one.

    Idempotent: registration overwrites by key, so importing this module more
    than once is harmless.
    """
    BaseDynamics.register_bookkeeping_key(
        "walker_id",
        lambda n, dev: torch.full((n, 1), -1, dtype=torch.long, device=dev),
    )
    for key in (
        "thermodynamic_state_id",
        "sampling_step",
        "sampling_epoch",
        "exchange_segment",
    ):
        BaseDynamics.register_bookkeeping_key(
            key, lambda n, dev: torch.zeros(n, 1, dtype=torch.long, device=dev)
        )


_register_identity_bookkeeping()


class EnhancedSampling(DynamicsStrategy):
    """Declarative recipe for biased dynamics on a ``BaseDynamics`` engine.

    A sibling of any other :class:`~nvalchemi.dynamics.DynamicsStrategy`:
    construction validates the configuration, :meth:`build_hooks` contributes
    the enhanced-sampling hook family, and ``DynamicsStrategy.run`` drives the
    engine.  The model, the integrator, the thermostat, and every other hook
    behave exactly as they would unbiased.

    Attributes
    ----------
    biases:
        Mapping of unique name to bias.  A bias is any
        :class:`~nvalchemi.models.base.BaseModelMixin` that maps a ``Batch``
        to :data:`~nvalchemi._typing.ModelOutputs` and carries a ``name`` —
        there is no separate bias protocol.  May be empty, which reduces the
        strategy to identity stamping — what pure temperature replica exchange
        needs, since the ladder alone drives the sampling.
    steps_per_epoch:
        Steps per consistency epoch, the
        :class:`~nvalchemi.hooks.StatefulHook` synchronisation boundary at
        which :meth:`AdaptivePotentialMixin.commit` fires.
    compile_biases:
        When ``True``, ``torch.compile`` each conservative bias's
        ``energy()``.  Not ``forward()`` — that path calls
        ``requires_grad_()``, which ``torch.compile`` cannot trace; see the
        :class:`~nvalchemi.enhanced_sampling.ConservativeBias` docstring.
    prime_after_update:
        When ``True`` (default), re-evaluate biases and rewrite the batch's
        total forces after an ``update()`` bumps a bias's state version, so
        that anything reading ``batch.forces`` between steps sees the current
        bias rather than the previous one.
    replica_exchange:
        Optional ladder.  When given, the strategy also contributes
        :class:`~nvalchemi.enhanced_sampling.hooks.ReplicaExchangeHook`.
        A runtime object rather than a declarative knob, so it is excluded
        from :meth:`to_spec_dict` the way ``extra_hooks`` is.

    Raises
    ------
    TypeError
        If any value in *biases* is not a
        :class:`~nvalchemi.models.base.BaseModelMixin`, or if
        *replica_exchange* is given for an engine that cannot rebind a
        thermodynamic state.
    ValueError
        If ``steps_per_epoch`` is below 1, or a bias's ``name`` disagrees
        with its key in *biases*.

    Examples
    --------
    >>> sampling = EnhancedSampling(                 # doctest: +SKIP
    ...     engine=NVTLangevin,
    ...     engine_kwargs={"dt": 0.5, "temperature": 300.0, "friction": 0.01},
    ...     biases={"umbrella": umbrella, "wall": lower_wall},
    ... )
    >>> batch = sampling.run(batch, model, n_steps=1000)  # doctest: +SKIP

    Notes
    -----
    Hook ordering
        :meth:`build_hooks` returns the strategy's own hooks before
        ``extra_hooks``, so at ``AFTER_COMPUTE`` the bias contribution is
        applied before any caller-supplied hook runs.  A safety hook such as
        ``MaxForceClampHook`` therefore clamps the *total* force, which is
        the physically meaningful quantity, rather than the model force
        alone.  Bias observations that need unbiased physical forces are
        captured inside :class:`~nvalchemi.enhanced_sampling.hooks.BiasHook`
        before the contribution is applied, so they are unaffected.

        Among the strategy's own hooks the order is identity, bias, exchange,
        commit.  Exchange precedes commit because the commit publishes shared
        history, and doing it before the swap would publish under labels that
        are about to change — the same order :meth:`checkpoint` drains in.

    One strategy, one trajectory
        The hook family is built once, at construction, because it holds the
        run's bookkeeping: the walker counter, the committed epoch, the
        attempted segment, and each bias's delivery record.  That is also
        what a checkpoint saves and :meth:`restore` writes back.  Calling
        ``build()`` more than once therefore hands a second engine the first
        one's bookkeeping; construct a second strategy instead.
    """

    biases: dict[str, Any] = Field(
        default_factory=dict,
        description="Unique name to bias; each a BaseModelMixin whose name matches.",
    )
    steps_per_epoch: int = Field(
        default=10_000, description="Steps per consistency epoch."
    )
    compile_biases: bool = Field(
        default=False, description="torch.compile each conservative bias's energy()."
    )
    prime_after_update: bool = Field(
        default=True, description="Re-prime forces when an update() changes a bias."
    )
    replica_exchange: ReplicaExchange | None = Field(
        default=None,
        exclude=True,
        description="Optional ladder; a runtime object, not serialised.",
    )

    _bias_hook: BiasHook = PrivateAttr()
    _identity_hook: WalkerIdentityHook = PrivateAttr()
    _epoch_hook: EpochCommitHook = PrivateAttr()
    _exchange_hook: PairSwapHook | None = PrivateAttr(default=None)
    # One-shot: the empirical state-dependence probe runs at prime time.
    _probed_state_dependence: bool = PrivateAttr(default=False)
    _restored: bool = PrivateAttr(default=False)

    @model_validator(mode="after")
    def _validate_and_build_hooks(self) -> EnhancedSampling:
        """Check the configuration, then construct the hook family.

        Returns
        -------
        EnhancedSampling
            This strategy, with its hooks built.

        Raises
        ------
        TypeError
            If a bias is not a ``BaseModelMixin``, or the engine cannot
            rebind a thermodynamic state under a configured ladder.
        ValueError
            If ``steps_per_epoch`` is below 1, or a bias's ``name`` disagrees
            with its key.
        """
        if self.steps_per_epoch < 1:
            raise ValueError(
                f"EnhancedSampling: steps_per_epoch must be at least 1, got "
                f"{self.steps_per_epoch}. It is a divisor — the epoch index is "
                "step // steps_per_epoch and the checkpoint boundary is "
                "step % steps_per_epoch — so zero raises deep in a run and a "
                "negative value makes both meaningless."
            )
        for key, bias in self.biases.items():
            if not isinstance(bias, BaseModelMixin):
                raise TypeError(
                    f"EnhancedSampling: biases[{key!r}] is a "
                    f"{type(bias).__name__}, which is not a BaseModelMixin. A "
                    "bias is an additive potential like any other: subclass "
                    "ConservativeBias to get forces and stress from an "
                    "energy(), or mix in BaseModelMixin directly and return "
                    "ModelOutputs from forward()."
                )
            if getattr(bias, "name", None) != key:
                raise ValueError(
                    f"EnhancedSampling: biases[{key!r}] has name="
                    f"{getattr(bias, 'name', None)!r}. The key and the bias "
                    "name must agree — both are used as identifiers, in the "
                    "output dict and in checkpoint group names respectively."
                )

        if self.replica_exchange is not None:
            self.replica_exchange.validate_for(self.biases)
            self._validate_exchange_capability()

        self._bias_hook = BiasHook(
            self.biases,
            prime_after_update=self.prime_after_update,
            compile_biases=self.compile_biases,
        )
        self._identity_hook = WalkerIdentityHook(
            steps_per_epoch=self.steps_per_epoch, exchange=self.replica_exchange
        )
        self._epoch_hook = EpochCommitHook(
            [self._bias_hook], frequency=self.steps_per_epoch
        )
        # The swap itself is generic — propose pairs, accept, permute
        # per-system parameters — so it is a PairSwapHook that the ladder
        # configures with the acceptance rule and the temperature table.
        self._exchange_hook = (
            self.replica_exchange.swap_hook(
                bias_energy_fn=self._bias_hook.bias_energy,
                on_swap=self._bias_hook.reprime_from_scratch,
            )
            if self.replica_exchange is not None
            else None
        )
        return self

    def _validate_exchange_capability(self) -> None:
        """Reject an engine that cannot rebind a thermodynamic state.

        An accepted swap must change the target temperature, rescale
        velocities, and transform any thermostat memory as one indivisible
        move.  An integrator that only accepts the new label would keep
        sampling the old temperature, which breaks detailed balance with no
        symptom the run would show — so this fails at construction rather
        than producing a plausible-looking wrong trajectory.

        Checked against the engine *class*, since a strategy holds a recipe
        rather than a live engine.  ``BaseDynamics`` defines
        ``apply_per_system_params`` only to raise, so presence is not enough
        and the subclass must override it.

        Raises
        ------
        TypeError
            If the engine does not implement the rebinding adapter.
        """
        if self.engine.apply_per_system_params is BaseDynamics.apply_per_system_params:
            raise TypeError(
                f"EnhancedSampling: replica exchange needs "
                f"{self.engine.__name__} to implement "
                "apply_per_system_params(), so an accepted swap can rebind "
                "temperature, velocities, and thermostat state together. "
                "NVTLangevin and NVTNoseHoover implement this; other "
                "integrators can run biased dynamics without exchange."
            )

    def build_hooks(self) -> list[Hook]:
        """Return the enhanced-sampling hooks, then ``extra_hooks``.

        Returns
        -------
        list[Hook]
            Identity, bias, exchange (when configured) and epoch-commit
            hooks, followed by whatever the base contributes.
        """
        return [
            self._identity_hook,
            self._bias_hook,
            *([self._exchange_hook] if self._exchange_hook is not None else []),
            self._epoch_hook,
            *super().build_hooks(),
        ]

    @property
    def last_outputs(self) -> dict[str, torch.Tensor]:
        """Diagnostics from the most recent force evaluation.

        ``physical/<field>`` is the model alone, ``bias/<name>/<field>`` one
        bias's contribution, ``bias_total/<field>`` the sum across biases, and
        ``total/<field>`` physical plus bias as read back from the batch.
        """
        return self._bias_hook.last_outputs

    def _adaptive_biases(self) -> dict[str, BaseModelMixin]:
        """Return the biases that implement the stateful-hook ``update``.

        Returns
        -------
        dict[str, BaseModelMixin]
            Name to bias, for adaptive biases only.
        """
        return self._bias_hook.adaptive_biases()

    def _require_engine(self) -> BaseDynamics:
        """Return the built engine, or explain that there is not one yet.

        Returns
        -------
        BaseDynamics
            The engine this strategy has been driving.

        Raises
        ------
        RuntimeError
            If no engine has been built, meaning nothing has been run,
            primed, or restored.
        """
        if self._engine is None:
            raise RuntimeError(
                "EnhancedSampling: no engine has been built yet. run(), "
                "prime_forces() and restore() all take the model the engine "
                "calls; pass it to one of those before asking for state that "
                "only a live engine has."
            )
        return self._engine

    def _probe_state_dependence(self, batch: Batch) -> None:
        """Reject a state-dependent bias under a temperature ladder.

        The declaration checked at construction only covers biases that know
        to declare — ``HarmonicUmbrellaBias`` does, an arbitrary user bias
        does not.  This probes instead of asking: evaluate every bias under
        the current assignment and under a rotated one, at identical
        coordinates.  A bias whose energy is independent of the assignment
        returns the same number twice; one that reads
        ``thermodynamic_state_id`` does not.

        Temperature acceptance uses only ``U``, so a bias whose energy varies
        with the assignment contributes cross-state terms that the rule never
        computes — detailed balance would be wrong with no symptom.  Runs
        once, at prime time, before any exchange has been attempted.

        Parameters
        ----------
        batch:
            The live batch, already carrying a state assignment.

        Raises
        ------
        ValueError
            If any bias's energy changes when the assignment is permuted.
        """
        exchange = self.replica_exchange
        if (
            exchange is None
            or exchange.acceptance != "temperature"
            or not self.biases
            or self._probed_state_dependence
        ):
            return
        self._probed_state_dependence = True

        current = batch.thermodynamic_state_id.reshape(-1).to(torch.long)
        if current.numel() < 2:
            return
        # Rotate by one: every walker sees a different state, and the result
        # is still a permutation, so a per-state lookup stays in range.
        rotated = torch.roll(current, shifts=1)

        original = batch.thermodynamic_state_id
        try:
            offenders: list[str] = []
            for name, bias in self.biases.items():
                batch["thermodynamic_state_id"] = current
                before = bias(batch).get("energy")
                batch["thermodynamic_state_id"] = rotated
                after = bias(batch).get("energy")
                if before is None or after is None:
                    continue
                if not torch.allclose(before, after, rtol=1e-9, atol=1e-12):
                    offenders.append(name)
        finally:
            batch["thermodynamic_state_id"] = original

        if offenders:
            raise ValueError(
                f"EnhancedSampling: bias(es) {sorted(offenders)} produce a "
                "different energy when the thermodynamic-state assignment is "
                "permuted, so they depend on the state — but the ladder varies "
                "temperature, which selects temperature acceptance. That rule "
                "uses only the total energy and omits the cross-state bias "
                "terms, so detailed balance would be wrong with nothing to "
                "show for it. The combined acceptance rule is not implemented: "
                "use a state-independent bias with a temperature ladder, or "
                "equal temperatures with a multi-window bias."
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def prime_forces(self, batch: Batch, model: BaseModelMixin) -> Batch:
        """Run one force evaluation without advancing dynamics.

        Populates ``batch.energy`` / ``forces`` / ``stress`` with the total
        (physical plus bias) values at the current coordinates.  Needed
        before the first step of a warm-started run, where a consumer may
        read forces before any step has happened.

        Parameters
        ----------
        batch:
            The batch to prime.
        model:
            The potential the engine calls.

        Returns
        -------
        Batch
            The same batch, primed in place.

        Raises
        ------
        ValueError
            If the batch has no ``forces`` field to write into.

        Notes
        -----
        Does not advance sampling state.  Identity is stamped directly rather
        than by dispatching ``BEFORE_STEP``, so no exchange is attempted and
        no epoch is committed from here — priming after a restore leaves the
        run exactly where the checkpoint left it.

        ``compute()`` writes its outputs back with ``copy_`` into fields that
        must already exist — a model output whose batch field is absent is
        silently discarded rather than created.  That is the toolkit's
        contract (``AtomicData(..., forces=torch.zeros(n, 3))``), so the
        check below turns a silent no-op into a named error.
        """
        if getattr(batch, "forces", None) is None:
            raise ValueError(
                "EnhancedSampling.prime_forces: batch has no 'forces' field. "
                "Model outputs are written back in place, so the buffer must "
                "exist first — construct AtomicData with "
                "forces=torch.zeros(n_atoms, 3) and energy=torch.zeros(1, 1), "
                "plus stress=torch.zeros(1, 3, 3) whenever a bias produces "
                "stress (any periodic batch, unless the bias was built with "
                "compute_stress=False)."
            )
        engine = self.dynamics(model)
        self._identity_hook.stamp(batch, engine.step_count)
        self._bias_hook.reprime_from_scratch(batch)
        self._probe_state_dependence(batch)
        return batch

    def warm_start(self, frames: Batch, model: BaseModelMixin) -> None:
        """Replay prior frames into every adaptive bias, in order.

        Approximate by construction: it reconstructs bias history but not
        velocities, RNG, or integrator state.  Use
        :meth:`restore` when exact reproducibility matters.

        Parameters
        ----------
        frames:
            Prior frames in chronological order, one graph per frame.
        model:
            The potential the engine calls; a replayed frame's context names
            it the same way a live one does.

        Raises
        ------
        RuntimeError
            If called after :meth:`restore`; the two are mutually exclusive
            and applying a warm start over a restored state would silently
            corrupt it.
        """
        if self._restored:
            raise RuntimeError(
                "EnhancedSampling: warm_start() and restore() are mutually "
                "exclusive. This strategy has already been restored from a "
                "checkpoint; warm-starting over it would replay history the "
                "restored state already contains."
            )
        self.dynamics(model)
        self._bias_hook.replay(frames)

    def run(
        self,
        batch: Batch,
        model: BaseModelMixin,
        n_steps: int | None = None,
        *,
        prime: bool = True,
    ) -> Batch:
        """Run biased dynamics.

        Primes forces first by default.  A velocity-Verlet-style integrator
        reads ``batch.forces`` in its *first* half-step, before any model
        call — so without priming, step 0 integrates against whatever the
        buffer happened to hold (zeros, for a freshly built batch), making it
        the one step in the run that ignores the bias.

        Parameters
        ----------
        batch:
            The initial batch.
        model:
            The potential the engine calls.
        n_steps:
            Number of steps; falls back to :attr:`n_steps`.
        prime:
            Set ``False`` to skip priming when the caller has already
            evaluated forces at these coordinates.

        Returns
        -------
        Batch
            The batch after all steps.
        """
        if prime:
            self.prime_forces(batch, model)
        return super().run(batch, model, n_steps=n_steps)

    def _components(self, engine: BaseDynamics | None = None) -> dict[str, Stateful]:
        """Name every object whose state a checkpoint must carry.

        Each is a :class:`~nvalchemi._checkpoint.Stateful`, so the checkpoint
        layer reads it directly rather than being handed a pre-collected
        schema: the hooks own their own cursors, the biases own their history,
        and the ladder owns its acceptance RNG position.

        Parameters
        ----------
        engine:
            The live engine, included as ``"dynamics"`` when given.  Omitted
            on restore, where the integrator's state has to be applied after
            its per-system arrays are allocated against the restored batch.

        Returns
        -------
        dict[str, Stateful]
            Component name to the object that owns that state.
        """
        components: dict[str, Stateful] = {}
        if engine is not None:
            components["dynamics"] = engine
        components["hooks/identity"] = self._identity_hook
        components["hooks/bias"] = self._bias_hook
        components["hooks/epoch"] = self._epoch_hook
        if self._exchange_hook is not None:
            components["hooks/exchange"] = self._exchange_hook
        for name, bias in self.biases.items():
            if isinstance(bias, Stateful):
                components[f"biases/{name}"] = bias
        if self.replica_exchange is not None:
            components["exchange"] = self.replica_exchange
        return components

    def _compatibility(self, engine: BaseDynamics) -> dict[str, Any]:
        """Describe the configuration a restore has to match.

        Parameters
        ----------
        engine:
            The live engine.

        Returns
        -------
        dict[str, Any]
            Fingerprint stored in the manifest and checked by
            :meth:`_validate_compatibility`.
        """
        return {
            "model_class": _qualified_name(engine.model),
            "dynamics_class": _qualified_name(engine),
            "bias_classes": {
                name: _qualified_name(bias) for name, bias in self.biases.items()
            },
            "exchange_config": (
                self.replica_exchange.config_fingerprint()
                if self.replica_exchange is not None
                else None
            ),
        }

    def checkpoint(self, path: str | Path, batch: Batch | None = None) -> None:
        """Write a transactional checkpoint at a consistency-epoch boundary.

        The boundary is not a convention — it is the only point where there
        are no pending ``update()`` calls and no in-flight epoch commit, so a
        checkpoint taken anywhere else could capture a bias mid-mutation.

        Parameters
        ----------
        path:
            Destination Zarr store.
        batch:
            The batch to save.  Defaults to the one last seen by the identity
            hook.

        Raises
        ------
        RuntimeError
            If no engine has been built or no batch is available, meaning
            nothing has been run or primed.
        ValueError
            If the current step is not an epoch boundary; the message names
            the next valid step.
        """
        engine = self._require_engine()
        target = batch if batch is not None else self._identity_hook.current_batch
        if target is None:
            raise RuntimeError(
                "EnhancedSampling.checkpoint: no batch to save. Run or prime "
                "the sampler first, or pass batch= explicitly."
            )

        step = engine.step_count
        if step % self.steps_per_epoch != 0:
            next_step = ((step // self.steps_per_epoch) + 1) * self.steps_per_epoch
            raise ValueError(
                f"EnhancedSampling.checkpoint: step {step} is not a consistency "
                f"epoch boundary (steps_per_epoch={self.steps_per_epoch}). The "
                f"next valid checkpoint step is {next_step}. Only at a boundary "
                "are there no pending update() calls or in-flight epoch commits "
                "to capture mid-mutation."
            )

        # Boundary-aligned is not the same as quiescent.  Both hooks fire on
        # their own cadence at the *start* of the step that crosses the
        # boundary — so at step N neither has run for the epoch that just
        # ended, and a checkpoint taken here would record pre-exchange labels
        # and a shared-history bias with its deposits still pending rather
        # than merged.  Drain both, in the same order build_hooks() registers
        # them: exchange first, because the commit publishes shared history
        # and doing it before the swap would publish under labels that are
        # about to change.
        if self._exchange_hook is not None:
            interval = self._exchange_hook.frequency
            self._exchange_hook.attempt_segment(target, step // interval - 1)
        self._epoch_hook.commit_epoch(step // self.steps_per_epoch - 1)

        save_checkpoint(
            path,
            self._components(engine),
            batch=target,
            batch_fields=_IDENTITY_FIELDS,
            compatibility=self._compatibility(engine),
            metadata={
                "sampling_step": step,
                "sampling_epoch": step // self.steps_per_epoch,
                "steps_per_epoch": self.steps_per_epoch,
            },
        )

    def restore(
        self,
        path: str | Path,
        model: BaseModelMixin,
        device: torch.device | str | None = None,
    ) -> Batch:
        """Restore a checkpoint exactly, and prime forces before returning.

        The caller must have reconstructed the same model, engine and biases
        first; this validates that they match what was saved.  **Model
        weights are not restored** — load them through the model's own API
        before calling here.  The compatibility metadata proves the
        architecture agrees, not that the weights do.

        Parameters
        ----------
        path:
            Source Zarr store.
        model:
            The potential the engine calls.
        device:
            Device for the restored batch.  Defaults to the model's own.

        Returns
        -------
        Batch
            The restored batch, force-primed and ready to run.

        Raises
        ------
        ValueError
            If the checkpoint is uncommitted, fails a checksum, or was
            written by a different model, engine, or bias set.
        """
        engine = self.dynamics(model)
        target_device = device if device is not None else self._model_device(engine)

        # "dynamics" is deliberately not in the mapping: its per-system arrays
        # have to be allocated against the restored batch before they can be
        # written into, and the batch only exists once the store is read. It is
        # applied by hand below, from the decoded state every component gets
        # back whether or not it was restored automatically.
        contents = load_checkpoint(
            path,
            self._components(),
            device=target_device,
            validate=lambda manifest: self._validate_compatibility(manifest, engine),
        )
        batch = contents.batch

        self.steps_per_epoch = int(contents.manifest.metadata["steps_per_epoch"])
        # The divisor lives on the hooks that use it, so a checkpoint written
        # with a different epoch length re-cadences them rather than leaving
        # the strategy's copy and the hooks' copies disagreeing.
        self._identity_hook.steps_per_epoch = self.steps_per_epoch
        self._epoch_hook.frequency = self.steps_per_epoch

        # Loading wrote each bias's saved state_version straight onto it, which
        # the bias hook never observed as a change; re-baseline so the first
        # post-restore update() does not read it as one.
        self._bias_hook.sync_seen_versions()

        engine._ensure_state_initialized(batch)
        engine.load_state_dict(contents.states["dynamics"])

        self._restored = True
        self.prime_forces(batch, model)
        return batch

    @staticmethod
    def _model_device(engine: BaseDynamics) -> torch.device:
        """Return the device the model's own tensors live on.

        ``BaseDynamics.device`` reports the process's compute device, which
        is CUDA whenever a GPU is visible — even for a model that was never
        moved off the CPU.  Restoring a batch there would put the batch and
        the model on different devices.  The model's own parameters are the
        authority.

        Parameters
        ----------
        engine:
            The live engine, which holds the model.

        Returns
        -------
        torch.device
            The model's device, falling back to the engine's device when the
            model holds no tensors (a pure-physics wrapper such as LJ).
        """
        model = engine.model
        for tensor in list(model.parameters()) + list(model.buffers()):
            return tensor.device
        return engine.device

    def _validate_compatibility(
        self, manifest: CheckpointManifest, engine: BaseDynamics
    ) -> None:
        """Reject a checkpoint written by a different configuration.

        Parameters
        ----------
        manifest:
            The committed manifest.
        engine:
            The live engine, which names the model and dynamics classes.

        Raises
        ------
        ValueError
            If the model class, dynamics class, or bias set disagrees.
        """
        saved = manifest.compatibility
        actual = self._compatibility(engine)

        problems: list[str] = []
        for key, label in (("model_class", "model"), ("dynamics_class", "dynamics")):
            recorded = saved.get(key)
            # An unrecorded class is tolerated; a recorded one that disagrees
            # is not.
            if recorded and recorded != actual[key]:
                problems.append(
                    f"  {label}: checkpoint has {recorded}, "
                    f"this strategy has {actual[key]}"
                )
        # Unconditional, unlike the two above: an empty bias set is a real
        # configuration, so "checkpoint had none, this strategy has three" is a
        # mismatch rather than a missing record.
        if saved.get("bias_classes", {}) != actual["bias_classes"]:
            problems.append(
                f"  biases: checkpoint has {saved.get('bias_classes', {})}, "
                f"this strategy has {actual['bias_classes']}"
            )
        # The ladder decides what a swap means, so a mismatch — including
        # exchange-versus-none in either direction — changes the semantics of
        # every future swap while the counters and assignment carry on looking
        # valid.
        problems.extend(
            ReplicaExchange.describe_config_mismatch(
                saved.get("exchange_config"), actual["exchange_config"]
            )
        )
        if problems:
            detail = "\n".join(problems)
            raise ValueError(
                "EnhancedSampling.restore: the checkpoint was written by a "
                f"different configuration:\n{detail}\n"
                "Reconstruct the same model, dynamics, and biases before "
                "restoring. Note that model *weights* are never restored from "
                "a checkpoint — load them through the model's own API."
            )

    def __repr__(self) -> str:
        """Return a concise description of the strategy."""
        names = ", ".join(self.biases) or "none"
        exchange = (
            f", exchange={self.replica_exchange!r}"
            if self.replica_exchange is not None
            else ""
        )
        return (
            f"{type(self).__name__}(engine={self.engine.__name__}, "
            f"biases=[{names}], steps_per_epoch={self.steps_per_epoch}"
            f"{exchange})"
        )

    def state_dict(self) -> Mapping[str, Any]:
        """Return sampling counters plus each adaptive bias's state.

        Returns
        -------
        Mapping[str, Any]
            Nested mapping; bias state lives under ``biases/<name>``.
        """
        state: dict[str, Any] = {
            "steps_per_epoch": self.steps_per_epoch,
            "next_walker_id": self._identity_hook.next_walker_id,
            "committed_epoch": self._epoch_hook.committed_epoch,
            "biases": {},
        }
        for name, bias in self.biases.items():
            getter = getattr(bias, "state_dict", None)
            if callable(getter):
                state["biases"][name] = getter()
        return state

    def to_spec_dict(self) -> dict[str, Any]:
        """Serialise the declarative knobs to a JSON-ready dict.

        ``biases`` and ``replica_exchange`` are excluded alongside
        ``extra_hooks``: all three are live objects rather than knobs.

        Returns
        -------
        dict[str, Any]
            JSON-ready bundle suitable for :func:`json.dumps`.
        """
        return {
            **super().to_spec_dict(),
            "steps_per_epoch": self.steps_per_epoch,
            "compile_biases": self.compile_biases,
            "prime_after_update": self.prime_after_update,
        }
