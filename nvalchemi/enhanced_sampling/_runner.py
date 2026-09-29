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
"""The ``EnhancedSampling`` runner: orchestration around an existing dynamics.

The runner owns what a bias cannot: walker identity, the ordering of the
force step, exactly-once ``update()`` delivery, and force priming after a
bias changes.  Each of those is a hook — see
:mod:`nvalchemi.enhanced_sampling.hooks` — and integration itself is
delegated entirely to the wrapped ``BaseDynamics``, which owns the stepping
loop.  The runner never touches an integrator.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from nvalchemi.dynamics.base import BaseDynamics
from nvalchemi.enhanced_sampling._checkpoint import (
    CheckpointManifest,
    _qualified_name,
    read_checkpoint,
    write_checkpoint,
)
from nvalchemi.enhanced_sampling._exchange import ReplicaExchange
from nvalchemi.enhanced_sampling.hooks import (
    BiasHook,
    EpochCommitHook,
    ReplicaExchangeHook,
    WalkerIdentityHook,
)
from nvalchemi.models.base import BaseModelMixin

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from nvalchemi.data import Batch
    from nvalchemi.hooks import Hook

__all__ = ["EnhancedSampling"]


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

    Idempotent: registration overwrites by key, so importing this module more
    than once is harmless.
    """
    BaseDynamics.register_bookkeeping_key(
        "walker_id",
        lambda n, dev: torch.arange(n, dtype=torch.long, device=dev).reshape(n, 1),
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


class EnhancedSampling:
    """Run biased dynamics on top of an existing ``BaseDynamics``.

    The runner installs the enhanced-sampling hook family and otherwise stays
    out of the way: the model, the integrator, the thermostat, and every
    other hook behave exactly as they would unbiased.

    Parameters
    ----------
    dynamics:
        Any ``BaseDynamics``.  Not subclassed, not wrapped — the runner
        registers hooks on it and calls its ``run``.
    biases:
        Mapping of unique name to bias.  A bias is any
        :class:`~nvalchemi.models.base.BaseModelMixin` that maps a ``Batch``
        to :data:`~nvalchemi._typing.ModelOutputs` and carries a ``name`` —
        there is no separate bias protocol.  May be empty, which reduces the
        runner to identity stamping — what pure temperature replica exchange
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
        Optional ladder.  When given, the runner also installs
        :class:`~nvalchemi.enhanced_sampling.hooks.ReplicaExchangeHook`.

    Raises
    ------
    TypeError
        If any value in *biases* is not a
        :class:`~nvalchemi.models.base.BaseModelMixin`.
    ValueError
        If ``steps_per_epoch`` is below 1, or a bias's ``name`` disagrees
        with its key in *biases*.

    Examples
    --------
    >>> sampling = EnhancedSampling(              # doctest: +SKIP
    ...     dynamics=md,
    ...     biases={"umbrella": umbrella, "wall": lower_wall},
    ... )
    >>> batch = sampling.run(batch, n_steps=1000)  # doctest: +SKIP

    Notes
    -----
    Hook ordering
        The runner's hooks are moved to the **front** of the dynamics hook
        list, so that at ``AFTER_COMPUTE`` the bias contribution is applied
        before any other hook runs.  A safety hook such as
        ``MaxForceClampHook`` therefore clamps the *total* force, which is
        the physically meaningful quantity, rather than the model force
        alone.  Bias observations that need unbiased physical forces are
        captured inside :class:`~nvalchemi.enhanced_sampling.hooks.BiasHook`
        before the contribution is applied, so they are unaffected by this
        ordering.

        Among the runner's own hooks the order is identity, bias, exchange,
        commit.  Exchange precedes commit because the commit publishes shared
        history, and doing it before the swap would publish under labels that
        are about to change — the same order :meth:`checkpoint` drains in.
    """

    def __init__(
        self,
        dynamics: BaseDynamics,
        biases: Mapping[str, BaseModelMixin] | None = None,
        *,
        steps_per_epoch: int = 10_000,
        compile_biases: bool = False,
        prime_after_update: bool = True,
        replica_exchange: ReplicaExchange | None = None,
    ) -> None:
        if int(steps_per_epoch) < 1:
            raise ValueError(
                f"EnhancedSampling: steps_per_epoch must be at least 1, got "
                f"{steps_per_epoch}. It is a divisor — the epoch index is "
                "step // steps_per_epoch and the checkpoint boundary is "
                "step % steps_per_epoch — so zero raises deep in a run and a "
                "negative value makes both meaningless."
            )
        self.dynamics = dynamics
        self.biases: dict[str, BaseModelMixin] = dict(biases or {})
        self.steps_per_epoch = int(steps_per_epoch)
        self.replica_exchange = replica_exchange

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

        if replica_exchange is not None:
            replica_exchange.validate_for(self.biases)
            self._validate_exchange_capability()

        # One-shot: the empirical state-dependence probe runs at prime time.
        self._probed_state_dependence = False
        self._restored = False

        self._bias_hook = BiasHook(
            self.biases,
            prime_after_update=prime_after_update,
            compile_biases=compile_biases,
        )
        self._identity_hook = WalkerIdentityHook(
            steps_per_epoch=self.steps_per_epoch, exchange=replica_exchange
        )
        self._epoch_hook = EpochCommitHook(
            [self._bias_hook], frequency=self.steps_per_epoch
        )
        self._exchange_hook = (
            ReplicaExchangeHook(replica_exchange, self._bias_hook)
            if replica_exchange is not None
            else None
        )
        self._hooks: list[Hook] = [
            self._identity_hook,
            self._bias_hook,
            *([self._exchange_hook] if self._exchange_hook is not None else []),
            self._epoch_hook,
        ]
        for index, hook in enumerate(self._hooks):
            dynamics.register_hook(hook)
            # Move to the front, preserving relative order: see the "Hook
            # ordering" note in the class docstring.
            dynamics.hooks.remove(hook)
            dynamics.hooks.insert(index, hook)

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

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------

    def _validate_exchange_capability(self) -> None:
        """Reject a dynamics that cannot rebind a thermodynamic state.

        An accepted swap must change the target temperature, rescale
        velocities, and transform any thermostat memory as one indivisible
        move.  An integrator that only accepts the new label would keep
        sampling the old temperature, which breaks detailed balance with no
        symptom the run would show — so this fails at construction rather
        than producing a plausible-looking wrong trajectory.

        Raises
        ------
        TypeError
            If the dynamics does not implement the rebinding adapters.
        """
        for method in ("apply_thermodynamic_state", "rescale_velocities_for_state"):
            if not callable(getattr(self.dynamics, method, None)):
                raise TypeError(
                    f"EnhancedSampling: replica exchange needs "
                    f"{type(self.dynamics).__name__} to implement {method}(), "
                    "so an accepted swap can rebind temperature, velocities, "
                    "and thermostat state together. NVTLangevin and "
                    "NVTNoseHoover implement this; other integrators can run "
                    "biased dynamics without exchange."
                )
        # BaseDynamics defines apply_thermodynamic_state only to raise, so
        # presence is not enough — probe it.
        try:
            self.dynamics.apply_thermodynamic_state(
                torch.zeros(0, dtype=torch.long), torch.zeros(0)
            )
        except NotImplementedError as exc:
            raise TypeError(
                f"EnhancedSampling: {type(self.dynamics).__name__} does not "
                "support thermodynamic-state rebinding, so it cannot take part "
                "in replica exchange."
            ) from exc
        except Exception:  # noqa: S110 - any other failure means it is implemented
            pass

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

    def prime_forces(self, batch: Batch) -> Batch:
        """Run one force evaluation without advancing dynamics.

        Populates ``batch.energy`` / ``forces`` / ``stress`` with the total
        (physical plus bias) values at the current coordinates.  Needed
        before the first step of a warm-started run, where a consumer may
        read forces before any step has happened.

        Parameters
        ----------
        batch:
            The batch to prime.

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
        self._identity_hook.stamp(batch, self.dynamics.step_count)
        self._bias_hook.reprime_from_scratch(batch)
        self._probe_state_dependence(batch)
        return batch

    def warm_start(self, frames: Batch) -> None:
        """Replay prior frames into every adaptive bias, in order.

        Approximate by construction: it reconstructs bias history but not
        velocities, RNG, or integrator state.  Use
        :meth:`restore` when exact reproducibility matters.

        Parameters
        ----------
        frames:
            Prior frames in chronological order, one graph per frame.

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
                "exclusive. This runner has already been restored from a "
                "checkpoint; warm-starting over it would replay history the "
                "restored state already contains."
            )
        self._bias_hook.replay(frames)

    def run(
        self, batch: Batch, n_steps: int | None = None, *, prime: bool = True
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
        n_steps:
            Number of steps; falls back to the dynamics' own ``n_steps``.
        prime:
            Set ``False`` to skip priming when the caller has already
            evaluated forces at these coordinates.

        Returns
        -------
        Batch
            The batch after all steps.
        """
        if prime:
            self.prime_forces(batch)
        return self.dynamics.run(batch, n_steps=n_steps)

    def _components(self) -> dict[str, dict[str, Any]]:
        """Collect every component's state for a checkpoint.

        Returns
        -------
        dict[str, dict[str, Any]]
            Component name to state mapping.
        """
        components: dict[str, dict[str, Any]] = {
            "dynamics": dict(self.dynamics.state_dict()),
            "runner": {
                "steps_per_epoch": self.steps_per_epoch,
                "next_walker_id": self._identity_hook.next_walker_id,
                "committed_epoch": self._epoch_hook.committed_epoch,
                "attempted_segment": (
                    self._exchange_hook.attempted_segment
                    if self._exchange_hook is not None
                    else -1
                ),
                **self._bias_hook.state_dict(),
            },
        }
        for name, bias in self.biases.items():
            getter = getattr(bias, "state_dict", None)
            if callable(getter):
                components[f"biases/{name}"] = dict(getter())
        if self.replica_exchange is not None:
            components["exchange"] = dict(self.replica_exchange.state_dict())
        return components

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
            The batch to save.  Defaults to the one last seen by the runner.

        Raises
        ------
        RuntimeError
            If no batch is available, meaning nothing has been run or primed.
        ValueError
            If the current step is not an epoch boundary; the message names
            the next valid step.
        """
        target = batch if batch is not None else self._identity_hook.current_batch
        if target is None:
            raise RuntimeError(
                "EnhancedSampling.checkpoint: no batch to save. Run or prime "
                "the sampler first, or pass batch= explicitly."
            )

        step = self.dynamics.step_count
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
        # than merged.  Drain both, in the same order the hooks are
        # registered: exchange first, because the commit publishes shared
        # history and doing it before the swap would publish under labels that
        # are about to change.
        if self._exchange_hook is not None:
            interval = self._exchange_hook.frequency
            self._exchange_hook.attempt_segment(target, step // interval - 1)
        self._epoch_hook.commit_epoch(step // self.steps_per_epoch - 1)

        write_checkpoint(
            path,
            target,
            self._components(),
            sampling_step=step,
            sampling_epoch=step // self.steps_per_epoch,
            steps_per_epoch=self.steps_per_epoch,
            model_class=_qualified_name(self.dynamics.model),
            dynamics_class=_qualified_name(self.dynamics),
            bias_classes={
                name: _qualified_name(bias) for name, bias in self.biases.items()
            },
            exchange_config=(
                self.replica_exchange.config_fingerprint()
                if self.replica_exchange is not None
                else None
            ),
        )

    def restore(
        self, path: str | Path, device: torch.device | str | None = None
    ) -> Batch:
        """Restore a checkpoint exactly, and prime forces before returning.

        The caller must have reconstructed the same model, dynamics, and
        biases first; this validates that they match what was saved.  **Model
        weights are not restored** — load them through the model's own API
        before calling here.  The compatibility metadata proves the
        architecture agrees, not that the weights do.

        Parameters
        ----------
        path:
            Source Zarr store.
        device:
            Device for the restored batch.  Defaults to the dynamics' device.

        Returns
        -------
        Batch
            The restored batch, force-primed and ready to run.

        Raises
        ------
        ValueError
            If the checkpoint is uncommitted, fails a checksum, or was
            written by a different model, dynamics, or bias set.
        """
        target_device = device if device is not None else self._model_device()
        batch, states, manifest = read_checkpoint(path, target_device)
        self._validate_compatibility(manifest)

        self.steps_per_epoch = int(manifest.steps_per_epoch)
        # The divisor lives on the hooks that use it, so a checkpoint written
        # with a different epoch length re-cadences them rather than leaving
        # the runner's copy and the hooks' copies disagreeing.
        self._identity_hook.steps_per_epoch = self.steps_per_epoch
        self._epoch_hook.frequency = self.steps_per_epoch

        runner_state = states.get("runner", {})
        self._identity_hook.next_walker_id = int(runner_state.get("next_walker_id", 0))
        self._epoch_hook.committed_epoch = int(runner_state.get("committed_epoch", -1))
        if self._exchange_hook is not None:
            self._exchange_hook.attempted_segment = int(
                runner_state.get("attempted_segment", -1)
            )
        self._bias_hook.load_state_dict(runner_state)

        exchange_state = states.get("exchange")
        if exchange_state is not None and self.replica_exchange is not None:
            self.replica_exchange.load_state_dict(exchange_state)

        for name, bias in self.biases.items():
            state = states.get(f"biases/{name}")
            loader = getattr(bias, "load_state_dict", None)
            if state is not None and callable(loader):
                loader(state)
        # Loading wrote each bias's saved state_version straight onto it, which
        # the bias hook never observed as a change; re-baseline so the first
        # post-restore update() does not read it as one.
        self._bias_hook.sync_seen_versions()

        # The integrator's per-system state must exist before it can be
        # restored into, and its shapes come from the batch — so initialise
        # against the restored batch first, then overwrite.
        self.dynamics._ensure_state_initialized(batch)
        self.dynamics.load_state_dict(states.get("dynamics", {}))

        self._restored = True
        self.prime_forces(batch)
        return batch

    def _model_device(self) -> torch.device:
        """Return the device the model's own tensors live on.

        ``BaseDynamics.device`` reports the process's compute device, which
        is CUDA whenever a GPU is visible — even for a model that was never
        moved off the CPU.  Restoring a batch there would put the batch and
        the model on different devices.  The model's own parameters are the
        authority.

        Returns
        -------
        torch.device
            The model's device, falling back to the dynamics' device when the
            model holds no tensors (a pure-physics wrapper such as LJ).
        """
        model = self.dynamics.model
        for tensor in list(model.parameters()) + list(model.buffers()):
            return tensor.device
        return self.dynamics.device

    def _validate_compatibility(self, manifest: CheckpointManifest) -> None:
        """Reject a checkpoint written by a different configuration.

        Parameters
        ----------
        manifest:
            The committed manifest.

        Raises
        ------
        ValueError
            If the model class, dynamics class, or bias set disagrees.
        """
        problems: list[str] = []
        actual_model = _qualified_name(self.dynamics.model)
        if manifest.model_class and manifest.model_class != actual_model:
            problems.append(
                f"  model: checkpoint has {manifest.model_class}, "
                f"this runner has {actual_model}"
            )
        actual_dynamics = _qualified_name(self.dynamics)
        if manifest.dynamics_class and manifest.dynamics_class != actual_dynamics:
            problems.append(
                f"  dynamics: checkpoint has {manifest.dynamics_class}, "
                f"this runner has {actual_dynamics}"
            )
        actual_biases = {
            name: _qualified_name(bias) for name, bias in self.biases.items()
        }
        if manifest.bias_classes != actual_biases:
            problems.append(
                f"  biases: checkpoint has {manifest.bias_classes}, "
                f"this runner has {actual_biases}"
            )
        # The ladder decides what a swap means, so a mismatch — including
        # exchange-versus-none in either direction — changes the semantics of
        # every future swap while the counters and assignment carry on looking
        # valid.
        problems.extend(
            ReplicaExchange.describe_config_mismatch(
                manifest.exchange_config,
                self.replica_exchange.config_fingerprint()
                if self.replica_exchange is not None
                else None,
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
        """Return a concise description of the runner."""
        names = ", ".join(self.biases) or "none"
        exchange = (
            f", exchange={self.replica_exchange!r}"
            if self.replica_exchange is not None
            else ""
        )
        return (
            f"{type(self).__name__}(dynamics={type(self.dynamics).__name__}, "
            f"biases=[{names}], steps_per_epoch={self.steps_per_epoch}"
            f"{exchange})"
        )

    def state_dict(self) -> Mapping[str, Any]:
        """Return runner counters plus each adaptive bias's state.

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
