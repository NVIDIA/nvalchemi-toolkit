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
"""The hooks an enhanced-sampling run installs on a ``BaseDynamics``.

``BaseDynamics`` owns the stepping loop and ``Hook`` already carries
``frequency`` and ``stage``, so the work an enhanced-sampling run adds is
expressed as hooks on that loop rather than as a second runner around it:

==============================  ======================  =====================
Concern                         Hook                    Cadence
==============================  ======================  =====================
Walker identity and counters    ``WalkerIdentityHook``  every step
Bias forces and updates         ``BiasHook``            every step
Shared-history synchronisation  ``EpochCommitHook``     ``steps_per_epoch``
Replica-exchange attempts       ``PairSwapHook``        ``attempt_interval``
==============================  ======================  =====================

The last of those is not defined here: proposing pairs, accepting them and
permuting per-system parameters is the mechanism behind basin hopping with
swaps and population search too, so it lives in
:class:`~nvalchemi.dynamics.hooks.PairSwapHook` and
:meth:`ReplicaExchange.swap_hook` supplies the physics.

The last two carry their cadence as ``Hook.frequency``, so the registry gates
them and nothing here re-implements "has the boundary been crossed".  A hook
at ``frequency=N`` is dispatched at step *kN*, where ``step // N - 1`` is the
index of the boundary that has just completed — the same index the lazy
"did the epoch change" test used to produce.

Where both are due on the same step, the exchange goes first:
:meth:`EnhancedSampling.build_hooks` registers it ahead of the commit, and
``checkpoint()`` drains them in the same order.  A commit publishes shared
history, and publishing it before the swap would file it under labels that are
about to change.

Why the biases share one hook
-----------------------------
:class:`BiasHook` takes *all* the biases rather than each bias being
registered separately.  Registering them independently would make each apply
its contribution in turn, so the second bias evaluates against a batch that
already carries the first one's forces — measured, on two biases contributing
along ``x``: the second saw ``|forces| = 3.0`` instead of ``0.0``.  That is the
sequential in-place mutation
:class:`~nvalchemi.hooks.BiasedPotentialHook` was deprecated for, and hook
registration order is user-owned, so the total would also depend on it.  One
hook evaluates every bias against the same unmodified model output and sums
once.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import TYPE_CHECKING, Any

import torch

from nvalchemi.dynamics.base import DynamicsStage
from nvalchemi.hooks._context import BiasContext
from nvalchemi.models._utils import (
    DIAGNOSTIC_PREFIX,
    aggregate_contributions,
    validate_contribution,
)

if TYPE_CHECKING:
    from collections.abc import Mapping
    from enum import Enum

    from nvalchemi._typing import ModelOutputs
    from nvalchemi.data import Batch
    from nvalchemi.enhanced_sampling._exchange import ReplicaExchange
    from nvalchemi.hooks import HookContext, StatefulHook
    from nvalchemi.models.base import BaseModelMixin

__all__ = [
    "BiasHook",
    "EpochCommitHook",
    "WalkerIdentityHook",
]

_ALLOCATION_HINT = {
    "energy": "energy=torch.zeros(1, 1)",
    "forces": "forces=torch.zeros(n_atoms, 3)",
    "stress": "stress=torch.zeros(1, 3, 3)",
}


class WalkerIdentityHook:
    """Stamp walker identity and step counters onto the live batch.

    ``walker_id`` and ``thermodynamic_state_id`` are assigned once and then
    preserved; the counters are refreshed every step.  All five fields are
    also registered as ``BaseDynamics`` bookkeeping keys, so they survive
    refill and graduation rather than depending on this hook alone.

    Parameters
    ----------
    steps_per_epoch:
        Divisor for ``sampling_epoch``.
    exchange:
        Ladder, when replica exchange is configured.  Read for the initial
        assignment and the segment divisor only; attempts are
        the swap hook's job.

    Attributes
    ----------
    next_walker_id:
        The identifier the next unstamped graph will receive.  Carried in a
        checkpoint so a resumed run does not reissue identifiers.
    current_batch:
        The batch last stamped, which is what ``checkpoint()`` saves when the
        caller does not name one.
    """

    def __init__(
        self,
        *,
        steps_per_epoch: int,
        exchange: ReplicaExchange | None = None,
    ) -> None:
        self.stage: Enum | None = DynamicsStage.BEFORE_STEP
        self.frequency = 1
        self.steps_per_epoch = int(steps_per_epoch)
        self.exchange = exchange
        self.next_walker_id = 0
        self.current_batch: Batch | None = None
        # One-shot: the walker/state bijection is checked on the first stamp.
        self._validated_assignment = False

    def __call__(self, ctx: HookContext, stage: Enum) -> None:
        """Stamp the live batch.

        Parameters
        ----------
        ctx:
            The dynamics hook context.
        stage:
            The stage being dispatched.
        """
        self.stamp(ctx.batch, getattr(ctx, "step_count", 0))

    def state_dict(self) -> Mapping[str, Any]:
        """Return the identifier allocation that must survive a restart.

        Returns
        -------
        Mapping[str, Any]
            The next identifier to issue.  A resumed run that restarted the
            counter would reissue ids already attached to saved history.
        """
        return {"next_walker_id": int(self.next_walker_id)}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore the identifier allocation.

        Parameters
        ----------
        state:
            A mapping produced by :meth:`state_dict`.
        """
        self.next_walker_id = int(state.get("next_walker_id", 0))

    def stamp(self, batch: Batch, step: int) -> None:
        """Write identity and counter fields for *step* onto *batch*.

        Exposed separately from :meth:`__call__` because force priming needs
        the batch stamped without a hook dispatch behind it.

        Parameters
        ----------
        batch:
            The live batch.
        step:
            The dynamics step the stamp describes.
        """
        n_graphs = batch.num_graphs
        device = batch.positions.device

        if getattr(batch, "walker_id", None) is None:
            batch["walker_id"] = torch.arange(
                self.next_walker_id,
                self.next_walker_id + n_graphs,
                dtype=torch.long,
                device=device,
            )
            self.next_walker_id += n_graphs

        existing = getattr(batch, "thermodynamic_state_id", None)
        if existing is None:
            if self.exchange is not None:
                # Check before attaching: a ladder-sized tensor on a
                # differently-sized batch would otherwise fail as an opaque
                # "Length mismatch" from inside the batch storage.
                self.exchange.validate_assignment(
                    self.exchange.initial_state_ids,
                    n_graphs,
                    source="initial_state_ids",
                )
                batch["thermodynamic_state_id"] = self.exchange.initial_state_ids.to(
                    device
                )
            else:
                batch["thermodynamic_state_id"] = torch.zeros(
                    n_graphs, dtype=torch.long, device=device
                )
        elif self.exchange is not None and not self._validated_assignment:
            # A batch may arrive carrying its own assignment, which never went
            # through the constructor's check. Validate it once — a duplicate
            # would surface later as a KeyError from the pair lookup.
            self.exchange.validate_assignment(
                existing, n_graphs, source="batch.thermodynamic_state_id"
            )
        self._validated_assignment = True

        self.current_batch = batch
        full = torch.full((n_graphs,), step, dtype=torch.long, device=device)
        batch["sampling_step"] = full
        batch["sampling_epoch"] = full // self.steps_per_epoch
        interval = (
            self.exchange.attempt_interval
            if self.exchange is not None
            else self.steps_per_epoch
        )
        batch["exchange_segment"] = torch.full(
            (n_graphs,), step // interval, dtype=torch.long, device=device
        )


class BiasHook:
    """Evaluate every bias against one unmodified model output, then apply.

    The single hook the bias family needs.  It defines ``_runs_on_stage`` so
    the registry lets one object serve both ``AFTER_COMPUTE`` (evaluate and
    apply) and ``AFTER_STEP`` (observe, deliver ``update()``, re-prime),
    which keeps the ordering guarantees in one place rather than spread
    across separately-registered hooks whose relative order would then depend
    on registration sequence.

    It is also the *orchestrator* for the biases, in the sense
    :class:`~nvalchemi.training.hooks.update.TrainingUpdateHook` establishes:
    an adaptive bias carries the
    :class:`~nvalchemi.hooks.StatefulHook` attributes (``frequency``,
    ``stage``, ``read_only``, ``commit``) but dispatches through ``update``
    rather than ``__call__``, because a bias is also an ``nn.Module`` whose
    ``__call__`` is its model forward.  This object owns protocol compliance
    on their behalf: it is the thing the registry sees, and it reads those
    attributes to decide when each bias is due.

    Parameters
    ----------
    biases:
        Mapping of unique name to bias.  May be empty, which reduces this
        hook to a no-op — what pure temperature replica exchange needs.
    prime_after_update:
        When ``True`` (default), re-evaluate the biases and rewrite the
        batch's total forces after an ``update()`` bumps a bias's state
        version, so that anything reading ``batch.forces`` between steps sees
        the current bias rather than the previous one.
    compile_biases:
        When ``True``, ``torch.compile`` each conservative bias's
        ``energy()``.

    Attributes
    ----------
    last_outputs:
        Diagnostics from the most recent force evaluation — ``physical/*``
        (model only), ``bias/<name>/*`` (one bias's contribution),
        ``bias_total/*`` (the sum across biases) and ``total/*`` (physical
        plus bias, read back from the batch).

    Notes
    -----
    Bias *state* is not part of :meth:`state_dict`.  A checkpoint writes each
    bias into its own ``biases/<name>`` group so the pieces can be inspected
    and restored individually; what this hook owns is the per-bias delivery
    bookkeeping that keeps ``update()`` exactly-once across a restart.
    """

    def __init__(
        self,
        biases: Mapping[str, BaseModelMixin] | None = None,
        *,
        prime_after_update: bool = True,
        compile_biases: bool = False,
    ) -> None:
        # stage=None plus _runs_on_stage: one object, two stages.
        self.stage: Enum | None = None
        self.frequency = 1
        self.read_only = False
        self.biases: dict[str, BaseModelMixin] = dict(biases or {})
        self.prime_after_update = bool(prime_after_update)

        self.last_outputs: dict[str, torch.Tensor] = {}

        # Per-bias observation captured at the bias's stage, consumed by the
        # next update() call.
        self._pending: dict[str, BiasContext] = {}

        # Per-bias contributions from the most recent force evaluation. Held
        # because an AFTER_STEP capture happens after that evaluation has
        # returned, and update() is documented to receive the contribution its
        # bias produced during it.
        self._last_results: dict[str, ModelOutputs] = {}
        self._last_update_step: dict[str, int] = {}
        self._last_seen_version: dict[str, int] = {}
        self._physical: dict[str, torch.Tensor] = {}
        self._dynamics: Any = None
        self.sync_seen_versions()

        if compile_biases:
            self._compile_bias_energies()

    # ------------------------------------------------------------------
    # Registry protocol
    # ------------------------------------------------------------------

    def _runs_on_stage(self, stage: Enum) -> bool:
        """Return whether this hook fires at *stage*.

        Parameters
        ----------
        stage:
            The stage being dispatched.

        Returns
        -------
        bool
            ``True`` at ``AFTER_COMPUTE`` and ``AFTER_STEP``.
        """
        return stage in (DynamicsStage.AFTER_COMPUTE, DynamicsStage.AFTER_STEP)

    def on_register(self, workflow: Any) -> None:
        """Remember the engine this hook was registered on.

        The engine is the authority on the current step, supplies the model a
        bias's ``update`` context names, and is what
        :meth:`reprime_from_scratch` drives through a fresh force evaluation.

        Parameters
        ----------
        workflow:
            The ``BaseDynamics`` doing the registering.
        """
        self._dynamics = workflow

    def __call__(self, ctx: HookContext, stage: Enum) -> None:
        """Dispatch to the phase for *stage*.

        Parameters
        ----------
        ctx:
            The dynamics hook context.
        stage:
            The stage being dispatched.
        """
        if stage is DynamicsStage.AFTER_COMPUTE:
            self.evaluate_and_apply(ctx.batch)
        elif stage is DynamicsStage.AFTER_STEP:
            self.observe_and_update(ctx.batch)

    def commit(self) -> None:
        """Publish every adaptive bias's pending state.

        The :class:`~nvalchemi.hooks.StatefulHook` synchronisation point, fanned
        out to the biases.  Called by :class:`EpochCommitHook` at a consistency
        epoch boundary, never on the hot path.
        """
        for bias in self.adaptive_biases().values():
            commit = getattr(bias, "commit", None)
            if callable(commit):
                commit()

    def state_dict(self) -> Mapping[str, Any]:
        """Return the delivery bookkeeping that must survive a restart.

        Returns
        -------
        Mapping[str, Any]
            The step at which each bias last received ``update()``, which is
            what keeps delivery exactly-once across the boundary.
        """
        return {
            "last_update_step": {
                name: int(step) for name, step in self._last_update_step.items()
            }
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore the delivery bookkeeping.

        Does **not** re-baseline the seen-version cache: the biases' own
        states are loaded separately, and the cache is only meaningful once
        they have been.  Call :meth:`sync_seen_versions` after that.

        Parameters
        ----------
        state:
            A mapping produced by :meth:`state_dict`.
        """
        self._last_update_step = {
            name: int(step)
            for name, step in (state.get("last_update_step") or {}).items()
        }

    @property
    def _step(self) -> int:
        """Return the engine's current step, or ``0`` before registration."""
        return int(getattr(self._dynamics, "step_count", 0))

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------

    def _compile_bias_energies(self) -> None:
        """Compile each conservative bias's ``energy()`` in place.

        Assigning the compiled callable as an instance attribute shadows the
        bound method, so ``forward()`` picks it up without an indirection the
        eager path would also pay.
        """
        for bias in self.biases.values():
            if hasattr(bias, "energy"):
                bias.energy = torch.compile(bias.energy)  # type: ignore[method-assign]

    def sync_seen_versions(self) -> None:
        """Re-baseline the cached per-bias state versions from the live biases.

        The cache answers "has this bias changed since the hook last looked",
        which is what decides whether forces need re-priming.  It must be
        re-baselined after anything that changes a bias's version without the
        hook observing it — construction, and a restore, which loads a saved
        version straight onto the bias.

        Called from both, rather than inlined, so the two cannot drift: a
        restore that skipped this would leave the cache at ``0`` against a
        restored version of ``N``, and the first post-restore ``update()``
        would re-prime on a change that never happened.
        """
        self._last_seen_version = {
            name: int(getattr(bias, "state_version", 0))
            for name, bias in self.biases.items()
        }

    def adaptive_biases(self) -> dict[str, BaseModelMixin]:
        """Return the biases that implement the stateful-hook ``update``.

        Detection is structural (``hasattr``), so a bias satisfies the
        adaptive contract without inheriting ``AdaptivePotentialMixin``.

        Returns
        -------
        dict[str, BaseModelMixin]
            Name to bias, for adaptive biases only.
        """
        return {
            name: bias
            for name, bias in self.biases.items()
            if callable(getattr(bias, "update", None))
        }

    # ------------------------------------------------------------------
    # Phase 1 — evaluate and apply (AFTER_COMPUTE)
    # ------------------------------------------------------------------

    def evaluate_and_apply(self, batch: Batch) -> None:
        """Evaluate every bias against the unmodified batch, then apply the sum.

        Implements steps 1-7 of the documented force-step ordering.  The
        ordering is the whole point: every bias sees the same physical
        outputs, so no bias can observe another's contribution and the result
        does not depend on registration order.

        Parameters
        ----------
        batch:
            The live batch, immediately after the model forward pass.
        """
        if not self.biases:
            return

        # 1. Capture the physical outputs before anything is added.
        self._physical = {
            key: value.clone()
            for key in ("energy", "forces", "stress")
            if (value := getattr(batch, key, None)) is not None
        }

        # 2-3. Evaluate each bias against the same unmodified batch.
        results = self._contributions(batch)

        # 4. Capture AFTER_COMPUTE observations while batch.forces still
        #    holds unbiased physical forces.  ABF depends on this: an
        #    estimator fed its own output diverges.
        self._capture(batch, results, DynamicsStage.AFTER_COMPUTE)

        # 5-6. Namespace diagnostics, then sum once.
        total = aggregate_contributions(
            [self._namespace(name, r) for name, r in results.items()]
        )
        self._record_diagnostics(results, total)

        # 7. Apply the total contribution, then record the combined result.
        self._apply(batch, total, results)
        self._record_totals(batch)

    def _contributions(self, batch: Batch) -> dict[str, ModelOutputs]:
        """Evaluate every bias against *batch* and check what each returned.

        Validation lives here rather than in each bias because this is where
        an unchecked contribution does damage: the next thing that happens to
        it is being added into a batch buffer, retained in a capture, or
        written to a checkpoint.  One enforcement point also covers biases
        that never call the checker themselves — a hand-written one, or a
        non-conservative one such as ABF.

        Parameters
        ----------
        batch:
            The live batch, unmodified by any bias.

        Returns
        -------
        dict[str, ModelOutputs]
            Name to contribution.

        Raises
        ------
        ValueError
            If any contribution violates the ``ModelOutputs`` conventions —
            an attached grad graph, a wrong shape, a NaN.  See
            :func:`~nvalchemi.models._utils.validate_contribution`.
        """
        results: dict[str, ModelOutputs] = {}
        for name, bias in self.biases.items():
            outputs = bias(batch)
            # Checked per bias, not on the aggregate: summing absorbs a
            # broadcastable shape ([4, 3] + [1, 3] is [4, 3]), so by the time
            # the total reaches the batch it looks right and the bias that
            # was wrong can no longer be named.
            validate_contribution(
                outputs,
                source=f"{type(bias).__name__} {name!r}",
                num_atoms=batch.num_nodes,
                num_graphs=batch.num_graphs,
            )
            results[name] = outputs
        self._last_results = results
        return results

    def _namespace(self, name: str, result: ModelOutputs) -> ModelOutputs:
        """Return *result* with its diagnostics keyed by contributing bias.

        ``diagnostics/<key>`` becomes ``diagnostics/bias/<name>/<key>``.
        Namespacing has to happen before aggregation, because
        :func:`~nvalchemi.models._utils.aggregate_contributions` rejects
        duplicate diagnostic keys rather than silently dropping one — two
        biases of the same type would otherwise collide on identical names.

        Parameters
        ----------
        name:
            The bias name.
        result:
            The bias's contribution.

        Returns
        -------
        ModelOutputs
            A copy with namespaced diagnostics, or *result* unchanged when it
            has none.
        """
        if not any(key.startswith(DIAGNOSTIC_PREFIX) for key in result):
            return result
        renamed: ModelOutputs = OrderedDict()
        for key, value in result.items():
            if key.startswith(DIAGNOSTIC_PREFIX):
                suffix = key[len(DIAGNOSTIC_PREFIX) :]
                renamed[f"{DIAGNOSTIC_PREFIX}bias/{name}/{suffix}"] = value
            else:
                renamed[key] = value
        return renamed

    def _record_diagnostics(
        self, results: dict[str, ModelOutputs], total: ModelOutputs
    ) -> None:
        """Populate :attr:`last_outputs` with the physical and bias views.

        Writes ``physical/*``, ``bias/<name>/*``, and ``bias_total/*``.
        ``total/*`` is *not* written here: at this point the bias has not
        been applied yet, so there is no combined value to record.
        :meth:`_record_totals` adds it afterwards.

        Parameters
        ----------
        results:
            Per-bias results, un-namespaced.
        total:
            The aggregated bias result — the sum across biases, **not**
            physical plus bias.
        """
        outputs: dict[str, torch.Tensor] = {}
        for key, value in self._physical.items():
            outputs[f"physical/{key}"] = value
        for name, result in results.items():
            for key, value in result.items():
                if value is None:
                    continue
                label = (
                    key[len(DIAGNOSTIC_PREFIX) :]
                    if key.startswith(DIAGNOSTIC_PREFIX)
                    else key
                )
                outputs[f"bias/{name}/{label}"] = value
        for key in ("energy", "forces", "stress", "virial"):
            value = total.get(key)
            if value is not None:
                outputs[f"bias_total/{key}"] = value
        self.last_outputs = outputs

    def _record_totals(self, batch: Batch) -> None:
        """Record ``total/*`` — physical plus bias — from the live batch.

        Must run *after* :meth:`_apply`.  Reading the batch rather than
        adding ``physical/*`` and ``bias_total/*`` back together keeps the
        record faithful to what was actually written, including any reshape
        :meth:`_apply` performed.

        Parameters
        ----------
        batch:
            The live batch, immediately after the bias has been applied.

        Notes
        -----
        This is the state as this hook leaves it, not necessarily the state
        the integrator sees.  The hook is deliberately ahead of caller-supplied
        hooks at ``AFTER_COMPUTE`` (so a force clamp acts on the total rather
        than on the model force alone), which means any later hook at that
        stage can still modify ``batch.forces`` afterwards.  Read the batch
        directly if you need the value the integrator consumed.
        """
        for key in ("energy", "forces", "stress"):
            value = getattr(batch, key, None)
            if value is not None:
                self.last_outputs[f"total/{key}"] = value.detach().clone()

    def _apply(
        self,
        batch: Batch,
        total: ModelOutputs,
        results: dict[str, ModelOutputs] | None = None,
    ) -> None:
        """Add the aggregated bias contribution to the batch, in place.

        Every non-``None`` output must have a destination buffer.  Skipping a
        field whose buffer is absent would discard that contribution in
        silence — for ``stress`` that is precisely the barostat-invisibility
        failure this API exists to remove, arrived at from a different
        direction: the bias is computed correctly, applied nowhere, and the
        cell evolves as though it did not exist.

        Parameters
        ----------
        batch:
            The live batch.
        total:
            The aggregated bias result.
        results:
            Per-bias results, used only to name the contributors in an error.

        Raises
        ------
        ValueError
            If the aggregate carries a virial, or if any non-``None`` output
            has no destination buffer on the batch.
        """
        if total.get("virial") is not None:
            raise ValueError(
                "EnhancedSampling: a bias returned 'virial', but the runner "
                "applies 'stress' to the batch. Convert W -> sigma = -W/V in "
                "the bias before returning it; the cell volume is the bias's "
                "to supply."
            )
        self._check_destinations(batch, total, results or {})
        with torch.no_grad():
            energy = total.get("energy")
            if energy is not None:
                batch.energy.add_(energy.reshape(batch.energy.shape))
            forces = total.get("forces")
            if forces is not None:
                # reshape, like energy and stress above: it raises on a numel
                # mismatch, where a bare add_ would broadcast a single row
                # onto every atom.
                batch.forces.add_(forces.reshape(batch.forces.shape))
            stress = total.get("stress")
            if stress is not None:
                batch.stress.add_(stress.reshape(batch.stress.shape))

    @staticmethod
    def _check_destinations(
        batch: Batch, total: ModelOutputs, results: dict[str, ModelOutputs]
    ) -> None:
        """Raise if any produced output has nowhere to go on the batch.

        Parameters
        ----------
        batch:
            The live batch.
        total:
            The aggregated bias result.
        results:
            Per-bias results, used to name which biases produced each field.

        Raises
        ------
        ValueError
            Listing every missing destination, the biases responsible, and
            both ways to resolve it.
        """
        missing = [
            key
            for key in ("energy", "forces", "stress")
            if total.get(key) is not None and getattr(batch, key, None) is None
        ]
        if not missing:
            return

        lines = []
        for key in missing:
            contributors = sorted(
                name for name, result in results.items() if result.get(key) is not None
            )
            who = f" (from {contributors})" if contributors else ""
            lines.append(f"  '{key}'{who}: add {_ALLOCATION_HINT[key]} to AtomicData")
        detail = "\n".join(lines)
        extra = ""
        if "stress" in missing:
            extra = (
                "\nA bias that produces stress with nowhere to put it is "
                "invisible to an NPT/NPH barostat — the cell would evolve as "
                "if the bias were absent. If this run genuinely has no use for "
                "a cell response (NVE/NVT), pass compute_stress=False to those "
                "biases instead of leaving the output to be discarded."
            )
        raise ValueError(
            f"EnhancedSampling: bias output has no destination buffer on the "
            f"batch, so it would be silently discarded:\n{detail}{extra}"
        )

    # ------------------------------------------------------------------
    # Phase 2 — observe and update (AFTER_STEP)
    # ------------------------------------------------------------------

    def _capture(
        self,
        batch: Batch,
        results: dict[str, ModelOutputs],
        stage: DynamicsStage,
    ) -> None:
        """Snapshot the batch for adaptive biases whose ``stage`` is *stage*.

        The stored :class:`~nvalchemi.hooks.BiasContext` is
        exactly what :meth:`AdaptivePotentialMixin.update` is documented to
        receive: the frame at the bias's ``stage``, and the contribution that
        bias returned during the preceding force evaluation.  An
        ``AFTER_STEP`` capture happens after that evaluation has returned, so
        the contributions are read from the retained results rather than
        recomputed — a metadynamics bias sizing its next hill from the bias
        energy it just applied needs the real value, not an empty
        placeholder.

        Parameters
        ----------
        batch:
            The live batch.
        results:
            Per-bias contributions from the preceding force evaluation.
        stage:
            The stage being captured.
        """
        step = self._step
        for name, bias in self.adaptive_biases().items():
            if getattr(bias, "stage", DynamicsStage.AFTER_STEP) is not stage:
                continue
            if step % max(1, getattr(bias, "frequency", 1)) != 0:
                continue
            self._pending[name] = self._bias_context(
                batch.clone(), results.get(name), step
            )

    def _bias_context(
        self, batch: Batch, contribution: ModelOutputs | None, step: int
    ) -> BiasContext:
        """Build the context handed to one bias's ``update``.

        Parameters
        ----------
        batch:
            The observed frame.
        contribution:
            What the bias returned during the preceding force evaluation, or
            ``None`` when it has not been evaluated yet.
        step:
            The dynamics step the capture belongs to.

        Returns
        -------
        BiasContext
            Populated context; ``workflow`` is this hook, so a bias can reach
            the engine and the full diagnostics dict if it needs to.
        """
        from nvalchemi.training.distributed import get_rank

        return BiasContext(
            batch=batch,
            model=getattr(self._dynamics, "model", None),
            global_rank=get_rank(None),
            workflow=self,
            step_count=step,
            contribution=contribution if contribution is not None else OrderedDict(),
        )

    def observe_and_update(self, batch: Batch) -> None:
        """Capture post-step frames, deliver ``update()``, then re-prime.

        Implements steps 10-12 of the force-step ordering.

        Parameters
        ----------
        batch:
            The live batch, after the integrator has finished.
        """
        adaptive = self.adaptive_biases()
        if not adaptive:
            return

        step = self._step
        self._capture(batch, self._last_results, DynamicsStage.AFTER_STEP)

        changed = False
        for name, bias in adaptive.items():
            if step % max(1, getattr(bias, "frequency", 1)) != 0:
                continue
            # Exactly once per step, even if this hook is dispatched twice.
            if self._last_update_step.get(name) == step:
                continue
            ctx = self._pending.pop(name, None)
            if ctx is None:
                ctx = self._bias_context(batch, self._last_results.get(name), step)
            bias.update(ctx, getattr(bias, "stage", DynamicsStage.AFTER_STEP))  # type: ignore[attr-defined]
            self._last_update_step[name] = step

            version = getattr(bias, "state_version", 0)
            if version != self._last_seen_version.get(name, 0):
                self._last_seen_version[name] = version
                changed = True

        if changed and self.prime_after_update:
            self._reprime(batch)

    def _reprime(self, batch: Batch) -> None:
        """Rewrite total forces from cached physical outputs and current biases.

        A bias that just deposited a hill leaves ``batch.forces`` describing
        the bias as it was *before* the deposition.  Anything reading the
        batch between steps — a reporter, a convergence check — would see
        stale values.  This restores the cached physical outputs and re-adds
        a freshly evaluated bias contribution.

        The physical part is reused rather than recomputed: this is
        *evaluate-only* priming, so it costs one bias evaluation and no model
        forward pass.  The physical forces are therefore the ones from the
        start of the step, not from the post-step coordinates.  That is exact
        only if the model forward were repeated, which is precisely the cost
        this avoids; the next step recomputes them anyway.

        Parameters
        ----------
        batch:
            The live batch.
        """
        if not self._physical:
            return
        with torch.no_grad():
            for key, value in self._physical.items():
                target = getattr(batch, key, None)
                if target is not None:
                    target.copy_(value.reshape(target.shape))
        results = self._contributions(batch)
        total = aggregate_contributions(
            [self._namespace(name, r) for name, r in results.items()]
        )
        self._record_diagnostics(results, total)
        self._apply(batch, total, results)
        self._record_totals(batch)

    def replay(self, frames: Batch) -> None:
        """Deliver each frame in *frames* to every adaptive bias, in order.

        The mechanism behind a warm start: it reconstructs bias history from
        prior frames, one graph at a time, without any dynamics behind it.
        The per-step exactly-once bookkeeping is deliberately bypassed —
        these are replayed frames, not steps of this run — and the frame
        index stands in for the step count.

        Parameters
        ----------
        frames:
            Prior frames in chronological order, one graph per frame.
        """
        adaptive = self.adaptive_biases()
        if not adaptive:
            return
        for index in range(frames.num_graphs):
            frame = frames.index_select(
                torch.tensor([index], device=frames.positions.device)
            )
            for bias in adaptive.values():
                ctx = self._bias_context(frame, None, index)
                bias.update(ctx, getattr(bias, "stage", DynamicsStage.AFTER_STEP))  # type: ignore[attr-defined]

    # ------------------------------------------------------------------
    # Services for the replica-exchange hook
    # ------------------------------------------------------------------

    def bias_energy(self, batch: Batch, state_ids: torch.Tensor) -> torch.Tensor:
        """Return the total bias energy per walker under *state_ids*.

        Umbrella acceptance needs the bias evaluated under both the current
        and the proposed labels, so the assignment is swapped in, the biases
        re-evaluated, and the original restored in a ``finally``.

        Raw energy, not reduced: multiplying by ``beta`` is thermodynamics and
        belongs to whatever ladder is asking, which is also what keeps this
        hook free of any temperature table.

        Parameters
        ----------
        batch:
            The live batch.
        state_ids:
            Assignment to evaluate under, shape ``[B]``.

        Returns
        -------
        torch.Tensor
            ``E_bias`` per walker in eV, shape ``[B]``.
        """
        original = batch.thermodynamic_state_id
        try:
            batch["thermodynamic_state_id"] = state_ids
            total = torch.zeros(
                batch.num_graphs,
                dtype=batch.positions.dtype,
                device=batch.positions.device,
            )
            for bias in self.biases.values():
                energy = bias(batch).get("energy")
                if energy is not None:
                    total = total + energy.reshape(-1)
        finally:
            batch["thermodynamic_state_id"] = original
        return total

    def reprime_from_scratch(self, batch: Batch) -> None:
        """Recompute the model and reapply the bias at fixed coordinates.

        Unlike :meth:`_reprime`, which reuses the cached physical outputs,
        this repeats the model forward pass — the caller has changed
        something the model itself responds to.  An accepted replica-exchange
        swap is that case: the forces in the batch were produced under the
        previous labels, and the integrator reads them in its next half-step
        before any model call.

        Reproduces the prefix of ``BaseDynamics.step`` up to the model call,
        including the ``BEFORE_COMPUTE`` hooks.  Those are not optional: a
        cutoff model reaches ``adapt_input`` expecting a neighbor list that
        ``NeighborListHook`` builds at exactly that stage, so calling
        ``compute()`` bare would fail.

        Parameters
        ----------
        batch:
            The live batch.
        """
        dynamics = self._dynamics
        dynamics._ensure_state_initialized(batch)
        dynamics._call_hooks(DynamicsStage.BEFORE_COMPUTE, batch)
        dynamics.compute(batch)
        self.evaluate_and_apply(batch)


class EpochCommitHook:
    """Fire :meth:`StatefulHook.commit` on a consistency-epoch cadence.

    The cadence is :attr:`frequency`, so the registry gates the dispatch and
    nothing here re-implements "has the boundary been crossed".  Commits stay
    idempotent by epoch index regardless, because the boundary is also reached
    eagerly when a checkpoint drains a completed epoch.

    Parameters
    ----------
    targets:
        The stateful hooks to commit — for enhanced sampling, the single
        :class:`BiasHook`, which fans out to its adaptive biases.
    frequency:
        Steps per consistency epoch.
    """

    def __init__(self, targets: list[StatefulHook], *, frequency: int) -> None:
        self.stage: Enum | None = DynamicsStage.BEFORE_STEP
        self.frequency = int(frequency)
        self.targets = targets
        self.committed_epoch = -1

    def __call__(self, ctx: HookContext, stage: Enum) -> None:
        """Commit the epoch that has just completed.

        Parameters
        ----------
        ctx:
            The dynamics hook context.
        stage:
            The stage being dispatched.
        """
        self.commit_epoch(getattr(ctx, "step_count", 0) // self.frequency - 1)

    def state_dict(self) -> Mapping[str, Any]:
        """Return the boundary cursor that must survive a restart.

        Returns
        -------
        Mapping[str, Any]
            The last epoch committed.  Restarting at ``-1`` would re-commit
            epochs whose shared history was already published, which a bias
            that merges pending deposits would double-count.
        """
        return {"committed_epoch": int(self.committed_epoch)}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore the boundary cursor.

        Parameters
        ----------
        state:
            A mapping produced by :meth:`state_dict`.
        """
        self.committed_epoch = int(state.get("committed_epoch", -1))

    def commit_epoch(self, epoch: int) -> None:
        """Commit *epoch*, at most once.

        Idempotent by design: the boundary is reached from two directions —
        the cadence, and a checkpoint draining a completed epoch — and a
        shared-history bias that merged its pending deposits twice would
        double-count them.

        Parameters
        ----------
        epoch:
            Index of the completed epoch.  Negative, or already committed, is
            a no-op.
        """
        if epoch < 0 or epoch <= self.committed_epoch:
            return
        for target in self.targets:
            commit = getattr(target, "commit", None)
            if callable(commit):
                commit()
        self.committed_epoch = epoch
