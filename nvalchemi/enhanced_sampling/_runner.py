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
bias changes.  Integration itself is delegated entirely to the wrapped
``BaseDynamics`` — the runner never touches an integrator.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any

import torch

from nvalchemi.dynamics.base import DynamicsStage
from nvalchemi.enhanced_sampling._bias import BiasResult, aggregate_bias_results

if TYPE_CHECKING:
    from collections.abc import Mapping
    from enum import Enum

    from nvalchemi.data import Batch
    from nvalchemi.dynamics.base import BaseDynamics
    from nvalchemi.enhanced_sampling._bias import BiasPotential
    from nvalchemi.hooks import HookContext

__all__ = ["EnhancedSampling"]


class _BiasCompositeHook:
    """The single internal hook the runner installs on the dynamics.

    An implementation detail, not a public API.  It defines
    ``_runs_on_stage`` so the registry lets one hook object serve three
    stages, which is what keeps the ordering guarantees in one place instead
    of spread across three separately-registered hooks whose relative order
    would then depend on registration sequence.
    """

    def __init__(self, runner: EnhancedSampling) -> None:
        self._runner = runner
        self.stage: Enum | None = None
        self.frequency = 1

    def _runs_on_stage(self, stage: Enum) -> bool:
        """Return whether this hook fires at *stage*.

        Parameters
        ----------
        stage:
            The stage being dispatched.

        Returns
        -------
        bool
            ``True`` for the three stages the runner needs.
        """
        return stage in (
            DynamicsStage.BEFORE_STEP,
            DynamicsStage.AFTER_COMPUTE,
            DynamicsStage.AFTER_STEP,
        )

    def __call__(self, ctx: HookContext, stage: Enum) -> None:
        """Dispatch to the runner phase for *stage*.

        Parameters
        ----------
        ctx:
            The dynamics hook context.
        stage:
            The stage being dispatched.
        """
        if stage is DynamicsStage.BEFORE_STEP:
            self._runner._stamp_identity(ctx.batch)
        elif stage is DynamicsStage.AFTER_COMPUTE:
            self._runner._evaluate_and_apply(ctx.batch)
        elif stage is DynamicsStage.AFTER_STEP:
            self._runner._observe_and_update(ctx.batch)


class EnhancedSampling:
    """Run biased dynamics on top of an existing ``BaseDynamics``.

    The runner installs one composite hook and otherwise stays out of the
    way: the model, the integrator, the thermostat, and every other hook
    behave exactly as they would unbiased.

    Parameters
    ----------
    dynamics:
        Any ``BaseDynamics``.  Not subclassed, not wrapped — the runner
        registers a hook on it and calls its ``run``.
    biases:
        Mapping of unique name to :class:`BiasPotential`.  May be empty,
        which reduces the runner to identity stamping (useful on its own for
        replica exchange in PR 5).
    steps_per_epoch:
        Steps per consistency epoch, the boundary at which
        :meth:`AdaptivePotentialMixin.commit_epoch` fires.
    compile_biases:
        When ``True``, ``torch.compile`` each conservative bias's
        ``energy()``.  Not ``evaluate()`` — that path calls
        ``requires_grad_()``, which ``torch.compile`` cannot trace; see the
        :class:`~nvalchemi.enhanced_sampling.ConservativeBias` docstring.
    prime_after_update:
        When ``True`` (default), re-evaluate biases and rewrite the batch's
        total forces after an ``update()`` bumps a bias's state version, so
        that anything reading ``batch.forces`` between steps sees the current
        bias rather than the previous one.

    Raises
    ------
    TypeError
        If any value in *biases* does not satisfy :class:`BiasPotential`.
    ValueError
        If a bias's ``name`` disagrees with its key in *biases*.

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
        The composite hook is inserted at the **front** of the dynamics hook
        list, so that at ``AFTER_COMPUTE`` the bias contribution is applied
        before any other hook runs.  A safety hook such as
        ``MaxForceClampHook`` therefore clamps the *total* force, which is
        the physically meaningful quantity, rather than the model force
        alone.  Bias observations that need unbiased physical forces are
        captured inside the runner before the contribution is applied, so
        they are unaffected by this ordering.
    """

    def __init__(
        self,
        dynamics: BaseDynamics,
        biases: Mapping[str, BiasPotential] | None = None,
        *,
        steps_per_epoch: int = 10_000,
        compile_biases: bool = False,
        prime_after_update: bool = True,
    ) -> None:
        from nvalchemi.enhanced_sampling._bias import BiasPotential as _Protocol

        self.dynamics = dynamics
        self.biases: dict[str, BiasPotential] = dict(biases or {})
        self.steps_per_epoch = int(steps_per_epoch)
        self.prime_after_update = bool(prime_after_update)

        for key, bias in self.biases.items():
            if not isinstance(bias, _Protocol):
                raise TypeError(
                    f"EnhancedSampling: biases[{key!r}] is a "
                    f"{type(bias).__name__}, which does not satisfy the "
                    "BiasPotential protocol (needs a 'name' attribute and an "
                    "'evaluate(batch) -> BiasResult' method)."
                )
            if getattr(bias, "name", None) != key:
                raise ValueError(
                    f"EnhancedSampling: biases[{key!r}] has name="
                    f"{getattr(bias, 'name', None)!r}. The key and the bias "
                    "name must agree — both are used as identifiers, in the "
                    "output dict and in checkpoint group names respectively."
                )

        if compile_biases:
            self._compile_bias_energies()

        # Diagnostics from the most recent force evaluation:
        #   physical/<field>        model only, before any bias
        #   bias/<name>/<field>     one bias's contribution
        #   bias_total/<field>      the sum across biases
        #   total/<field>           physical + bias, read back from the batch
        self.last_outputs: dict[str, torch.Tensor] = {}

        # Per-bias observation captured at its observation_stage, consumed by
        # the next update() call.
        self._pending: dict[str, tuple[Batch, BiasResult]] = {}

        # Per-bias results from the most recent force evaluation. Held because
        # an AFTER_STEP capture happens after that evaluation has returned,
        # and update() is documented to receive the result its bias produced
        # during it.
        self._last_results: dict[str, BiasResult] = {}
        self._last_update_step: dict[str, int] = {}
        self._last_seen_version: dict[str, int] = {
            name: getattr(bias, "state_version", 0)
            for name, bias in self.biases.items()
        }
        self._physical: dict[str, torch.Tensor] = {}
        self._next_walker_id = 0
        self._last_epoch = -1

        self._hook = _BiasCompositeHook(self)
        dynamics.register_hook(self._hook, stage=DynamicsStage.AFTER_COMPUTE)
        # Move to the front: see the "Hook ordering" note in the class docstring.
        dynamics.hooks.remove(self._hook)
        dynamics.hooks.insert(0, self._hook)

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------

    def _compile_bias_energies(self) -> None:
        """Compile each conservative bias's ``energy()`` in place.

        Assigning the compiled callable as an instance attribute shadows the
        bound method, so ``evaluate()`` picks it up without an indirection
        the eager path would also pay.
        """
        for bias in self.biases.values():
            if hasattr(bias, "energy"):
                bias.energy = torch.compile(bias.energy)  # type: ignore[method-assign]

    def _adaptive_biases(self) -> dict[str, BiasPotential]:
        """Return the biases that implement ``update``.

        Detection is structural (``hasattr``), so a bias satisfies the
        adaptive contract without inheriting ``AdaptivePotentialMixin``.

        Returns
        -------
        dict[str, BiasPotential]
            Name to bias, for adaptive biases only.
        """
        return {
            name: bias
            for name, bias in self.biases.items()
            if callable(getattr(bias, "update", None))
        }

    # ------------------------------------------------------------------
    # Phase 1 — identity stamping (BEFORE_STEP)
    # ------------------------------------------------------------------

    def _stamp_identity(self, batch: Batch) -> None:
        """Stamp walker identity and counters onto the live batch.

        ``walker_id`` and ``thermodynamic_state_id`` are assigned once and
        then preserved; the counters are refreshed every step.

        Parameters
        ----------
        batch:
            The live batch.
        """
        n_graphs = batch.num_graphs
        device = batch.positions.device
        step = self.dynamics.step_count

        if getattr(batch, "walker_id", None) is None:
            batch["walker_id"] = torch.arange(
                self._next_walker_id,
                self._next_walker_id + n_graphs,
                dtype=torch.long,
                device=device,
            )
            self._next_walker_id += n_graphs

        if getattr(batch, "thermodynamic_state_id", None) is None:
            batch["thermodynamic_state_id"] = torch.zeros(
                n_graphs, dtype=torch.long, device=device
            )

        full = torch.full((n_graphs,), step, dtype=torch.long, device=device)
        batch["sampling_step"] = full
        batch["sampling_epoch"] = full // self.steps_per_epoch
        batch["exchange_segment"] = torch.zeros(
            n_graphs, dtype=torch.long, device=device
        )

        epoch = step // self.steps_per_epoch
        if epoch != self._last_epoch:
            if self._last_epoch >= 0:
                for bias in self._adaptive_biases().values():
                    commit = getattr(bias, "commit_epoch", None)
                    if callable(commit):
                        commit()
            self._last_epoch = epoch

    # ------------------------------------------------------------------
    # Phase 2 — evaluate and apply (AFTER_COMPUTE)
    # ------------------------------------------------------------------

    def _evaluate_and_apply(self, batch: Batch) -> None:
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
        #      BiasResult validates itself on construction.
        results = {name: bias.evaluate(batch) for name, bias in self.biases.items()}
        self._last_results = results

        # 4. Capture AFTER_COMPUTE observations while batch.forces still
        #    holds unbiased physical forces.  ABF depends on this: an
        #    estimator fed its own output diverges.
        self._capture(batch, results, DynamicsStage.AFTER_COMPUTE)

        # 5-6. Namespace observables, then sum once.
        total = aggregate_bias_results(
            [self._namespace(name, r) for name, r in results.items()]
        )
        self._record_diagnostics(results, total)

        # 7. Apply the total contribution, then record the combined result.
        self._apply(batch, total, results)
        self._record_totals(batch)

    def _namespace(self, name: str, result: BiasResult) -> BiasResult:
        """Return *result* with its observables prefixed ``bias/<name>/``.

        Namespacing has to happen before aggregation, because
        ``aggregate_bias_results`` rejects duplicate observable keys rather
        than silently dropping one — two biases of the same type would
        otherwise collide on identical names.

        Parameters
        ----------
        name:
            The bias name.
        result:
            The bias's result.

        Returns
        -------
        BiasResult
            A copy with namespaced observables, or *result* unchanged when it
            has none.
        """
        if not result.observables:
            return result
        return dataclasses.replace(
            result,
            observables={
                f"bias/{name}/{key}": value for key, value in result.observables.items()
            },
        )

    def _record_diagnostics(
        self, results: dict[str, BiasResult], total: BiasResult
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
            for key in ("energy", "forces", "stress", "virial"):
                value = getattr(result, key)
                if value is not None:
                    outputs[f"bias/{name}/{key}"] = value
            for key, value in result.observables.items():
                outputs[f"bias/{name}/{key}"] = value
        for key in ("energy", "forces", "stress", "virial"):
            value = getattr(total, key)
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
        This is the state as the runner leaves it, not necessarily the state
        the integrator sees.  The runner's hook is deliberately first at
        ``AFTER_COMPUTE`` (so a force clamp acts on the total rather than on
        the model force alone), which means any later hook at that stage can
        still modify ``batch.forces`` afterwards.  Read the batch directly if
        you need the value the integrator consumed.
        """
        for key in ("energy", "forces", "stress"):
            value = getattr(batch, key, None)
            if value is not None:
                self.last_outputs[f"total/{key}"] = value.detach().clone()

    def _apply(
        self,
        batch: Batch,
        total: BiasResult,
        results: dict[str, BiasResult] | None = None,
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
        if total.virial is not None:
            raise ValueError(
                "EnhancedSampling: a bias returned 'virial', but the runner "
                "applies 'stress' to the batch. Convert W -> sigma = -W/V in "
                "the bias before returning it; the cell volume is the bias's "
                "to supply."
            )
        self._check_destinations(batch, total, results or {})
        with torch.no_grad():
            if total.energy is not None:
                batch.energy.add_(total.energy.reshape(batch.energy.shape))
            if total.forces is not None:
                batch.forces.add_(total.forces)
            if total.stress is not None:
                batch.stress.add_(total.stress.reshape(batch.stress.shape))

    @staticmethod
    def _check_destinations(
        batch: Batch, total: BiasResult, results: dict[str, BiasResult]
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
        _ALLOCATION_HINT = {
            "energy": "energy=torch.zeros(1, 1)",
            "forces": "forces=torch.zeros(n_atoms, 3)",
            "stress": "stress=torch.zeros(1, 3, 3)",
        }
        missing = [
            key
            for key in ("energy", "forces", "stress")
            if getattr(total, key) is not None and getattr(batch, key, None) is None
        ]
        if not missing:
            return

        lines = []
        for key in missing:
            contributors = sorted(
                name
                for name, result in results.items()
                if getattr(result, key, None) is not None
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
    # Phase 3 — observe and update (AFTER_STEP)
    # ------------------------------------------------------------------

    def _capture(
        self,
        batch: Batch,
        results: dict[str, BiasResult],
        stage: DynamicsStage,
    ) -> None:
        """Snapshot the batch for adaptive biases observing at *stage*.

        The stored pair is exactly what :meth:`AdaptivePotentialMixin.update`
        is documented to receive: the frame at the bias's
        ``observation_stage``, and the :class:`BiasResult` that bias returned
        during the preceding force evaluation.  An ``AFTER_STEP`` capture
        happens after that evaluation has returned, so the results are read
        from :attr:`_last_results` rather than recomputed — a metadynamics
        bias sizing its next hill from the bias energy it just applied needs
        the real value, not an empty placeholder.

        Parameters
        ----------
        batch:
            The live batch.
        results:
            Per-bias results from the preceding force evaluation.
        stage:
            The stage being captured.
        """
        step = self.dynamics.step_count
        for name, bias in self._adaptive_biases().items():
            if (
                getattr(bias, "observation_stage", DynamicsStage.AFTER_STEP)
                is not stage
            ):
                continue
            if step % max(1, getattr(bias, "update_frequency", 1)) != 0:
                continue
            self._pending[name] = (batch.clone(), results.get(name, BiasResult()))

    def _observe_and_update(self, batch: Batch) -> None:
        """Capture post-step frames, deliver ``update()``, then re-prime.

        Implements steps 10-12 of the force-step ordering.

        Parameters
        ----------
        batch:
            The live batch, after the integrator has finished.
        """
        adaptive = self._adaptive_biases()
        if not adaptive:
            return

        step = self.dynamics.step_count
        self._capture(batch, self._last_results, DynamicsStage.AFTER_STEP)

        changed = False
        for name, bias in adaptive.items():
            if step % max(1, getattr(bias, "update_frequency", 1)) != 0:
                continue
            # Exactly once per step, even if this hook is dispatched twice.
            if self._last_update_step.get(name) == step:
                continue
            frames, result = self._pending.pop(
                name, (batch, self._last_results.get(name, BiasResult()))
            )
            bias.update(frames, result)  # type: ignore[attr-defined]
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
        results = {name: bias.evaluate(batch) for name, bias in self.biases.items()}
        self._last_results = results
        total = aggregate_bias_results(
            [self._namespace(name, r) for name, r in results.items()]
        )
        self._record_diagnostics(results, total)
        self._apply(batch, total, results)
        self._record_totals(batch)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def prime_forces(self, batch: Batch) -> Batch:
        """Run one evaluate-only force evaluation without advancing dynamics.

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
        ``compute()`` writes its outputs back with ``copy_`` into fields that
        must already exist — a model output whose batch field is absent is
        silently discarded rather than created.  That is the toolkit's
        contract (``AtomicData(..., forces=torch.zeros(n, 3))``), so the
        check below turns a silent no-op into a named error.

        This reproduces the prefix of ``BaseDynamics.step`` up to the model
        call, including the ``BEFORE_COMPUTE`` hooks.  Those are not
        optional: a cutoff model reaches ``adapt_input`` expecting a neighbor
        list that ``NeighborListHook`` builds at exactly that stage, so
        calling ``compute()`` bare would fail on the first primed evaluation.
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
        self.dynamics._ensure_state_initialized(batch)
        self._stamp_identity(batch)
        self.dynamics._call_hooks(DynamicsStage.BEFORE_COMPUTE, batch)
        self.dynamics.compute(batch)
        self._evaluate_and_apply(batch)
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
        if getattr(self, "_restored", False):
            raise RuntimeError(
                "EnhancedSampling: warm_start() and restore() are mutually "
                "exclusive. This runner has already been restored from a "
                "checkpoint; warm-starting over it would replay history the "
                "restored state already contains."
            )
        adaptive = self._adaptive_biases()
        if not adaptive:
            return
        for index in range(frames.num_graphs):
            frame = frames.index_select(
                torch.tensor([index], device=frames.positions.device)
            )
            for bias in adaptive.values():
                bias.update(frame, BiasResult())  # type: ignore[attr-defined]

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

    def checkpoint(self, path: str) -> None:
        """Write a transactional Zarr checkpoint.

        Parameters
        ----------
        path:
            Destination store.

        Raises
        ------
        NotImplementedError
            Always; exact checkpointing is PR 4 work. ``nn.Module.state_dict``
            on individual biases is available in the meantime.
        """
        raise NotImplementedError(
            "EnhancedSampling.checkpoint() is not implemented yet (PR 4). "
            "Individual biases expose state_dict()/load_state_dict() now."
        )

    def restore(self, path: str) -> Batch:
        """Restore a checkpoint written by :meth:`checkpoint`.

        Parameters
        ----------
        path:
            Source store.

        Raises
        ------
        NotImplementedError
            Always; exact restore is PR 4 work.
        """
        raise NotImplementedError(
            "EnhancedSampling.restore() is not implemented yet (PR 4). "
            "Use warm_start() for approximate continuation."
        )

    def __repr__(self) -> str:
        """Return a concise description of the runner."""
        names = ", ".join(self.biases) or "none"
        return (
            f"{type(self).__name__}(dynamics={type(self.dynamics).__name__}, "
            f"biases=[{names}], steps_per_epoch={self.steps_per_epoch})"
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
            "next_walker_id": self._next_walker_id,
            "last_epoch": self._last_epoch,
            "biases": {},
        }
        for name, bias in self.biases.items():
            getter = getattr(bias, "state_dict", None)
            if callable(getter):
                state["biases"][name] = getter()
        return state
