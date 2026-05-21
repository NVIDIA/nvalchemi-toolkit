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
"""Training-update hook base class and orchestrator."""

from __future__ import annotations

import operator
from collections.abc import Sequence
from functools import reduce
from typing import TYPE_CHECKING, Any, Literal

import plum

from nvalchemi.hooks._context import TrainContext
from nvalchemi.hooks._protocol import Hook
from nvalchemi.training._stages import TrainingStage
from nvalchemi.training.optimizers import (
    step_lr_schedulers,
    step_optimizers,
    zero_gradients,
)

if TYPE_CHECKING:
    import torch


_TRAINING_UPDATE_STAGES: tuple[TrainingStage, ...] = (
    TrainingStage.BEFORE_BATCH,
    TrainingStage.DO_BACKWARD,
    TrainingStage.DO_OPTIMIZER_STEP,
    TrainingStage.AFTER_OPTIMIZER_STEP,
)


_MULTIPLE_ORCHESTRATOR_MSG = (
    "Only one TrainingUpdateOrchestrator is allowed; compose update hooks "
    "with `+` before registration."
)


def _hook_claims_stage(hook: Any, stage: TrainingStage) -> bool:
    """Return True if hook fires on stage (mirrors _registry._call_hooks dispatch)."""
    runs_on_stage = getattr(hook, "_runs_on_stage", None)
    if runs_on_stage is not None:
        return runs_on_stage(stage)
    return getattr(hook, "stage", None) == stage


def _fold_training_update_hooks(
    hooks: Sequence[Hook],
) -> list[Hook]:
    """Fold TrainingUpdateHook/Orchestrator instances into a single orchestrator."""
    others: list[Hook] = []
    update_hooks: list[Hook] = []
    insertion_index: int | None = None
    n_orch = 0
    for h in hooks:
        if isinstance(h, TrainingUpdateOrchestrator):
            if insertion_index is None:
                insertion_index = len(others)
            update_hooks.append(h)
            n_orch += 1
        elif isinstance(h, TrainingUpdateHook):
            if insertion_index is None:
                insertion_index = len(others)
            update_hooks.append(h)
        else:
            others.append(h)
    if not update_hooks:
        return list(hooks)
    if n_orch > 1:
        raise ValueError(_MULTIPLE_ORCHESTRATOR_MSG)
    folded = reduce(operator.add, update_hooks)
    if not isinstance(folded, TrainingUpdateOrchestrator):
        folded = TrainingUpdateOrchestrator(folded)
    insert_at = insertion_index if insertion_index is not None else len(others)
    result: list[Hook] = list(others)
    result.insert(insert_at, folded)
    return result


def _check_veto(decision: object, hook: object, stage: TrainingStage) -> None:
    """Validate that ``__call__`` returned a strict ``bool`` for ``proceed``."""
    if not isinstance(decision, bool):
        raise TypeError(
            f"{type(hook).__name__}.__call__(stage={stage.name}) must return "
            f"(bool, Tensor); proceed got {type(decision).__name__}. "
            "Return True to proceed or False to skip."
        )


class TrainingUpdateHook:
    """Base class for hooks that customize training-update phases.

    Subclasses override :meth:`__call__` and dispatch on ``stage`` to
    handle one or more of the four claimed stages: ``BEFORE_BATCH``,
    ``DO_BACKWARD``, ``DO_OPTIMIZER_STEP``, ``AFTER_OPTIMIZER_STEP``.
    Compose via ``+`` to build a :class:`TrainingUpdateOrchestrator`.

    Attributes
    ----------
    priority : int
        Dispatch order within an orchestrator; lower runs first. Canonical
        buckets: 10 = gradient accumulation, 20 = mixed precision,
        30 = gradient clipping, 40 = spike skipping. Default 50.

    Notes
    -----
    ``TrainingUpdateHook`` is NOT directly compatible with the standard
    :class:`Hook` Protocol -- its ``__call__`` signature includes a
    ``will_skip`` argument and returns ``(bool, torch.Tensor)`` rather
    than the Protocol's ``__call__(ctx, stage) -> None``. This is
    intentional: ``Hook`` is a structural Protocol so domain-specific
    hook families can use signatures suited to their semantics. Bare
    instances must be composed via ``+`` or wrapped by a
    :class:`TrainingUpdateOrchestrator` (the strategy auto-wraps lone
    hooks); the orchestrator owns Protocol compliance.

    ``will_skip`` is a stage-local cumulative veto signal. It is ``True`` when
    an earlier, higher-priority hook has already requested that the current
    stage's gated operation be skipped. The orchestrator still calls later
    hooks after a veto so they can observe the decision, update bookkeeping, or
    emit diagnostics, but those hooks should avoid side effects that assume the
    gated operation will run. A hook may also return ``False`` to veto the
    operation for lower-priority hooks.

    This signal is intended for composable pipeline behavior. For example, a
    gradient-accumulation hook can veto ``DO_OPTIMIZER_STEP`` on non-step
    microbatches; later hooks then receive ``will_skip=True`` and can skip
    work such as gradient clipping, scaler updates, or expensive parameter
    scans. ``will_skip`` is reset for each stage dispatch and should not be
    interpreted as a global training-step status unless the orchestrator also
    records that state on ``ctx``.

    Each ``__call__`` returns ``(proceed, loss)``:

    - ``proceed`` is a strict ``bool`` (``int``/``None`` raise
      ``TypeError``). On ``BEFORE_BATCH`` and ``DO_OPTIMIZER_STEP`` the
      orchestrator applies any-veto-wins composition: if any hook returns
      ``False`` the gated operation (``zero_gradients`` or
      ``optimizer/scheduler.step``) is skipped. On ``DO_BACKWARD`` and
      ``AFTER_OPTIMIZER_STEP`` the value is unused; return ``True``.
    - ``loss`` is the loss tensor the hook would use, transformed or not.
      Default is ``ctx.loss`` unchanged. The orchestrator threads it
      through hooks in priority order during ``DO_BACKWARD`` so each hook
      sees its predecessor's transform; ``backward()`` runs once on the
      final loss.

    Examples
    --------
    >>> import torch
    >>> from nvalchemi.training._stages import TrainingStage
    >>> class ClipGrads(TrainingUpdateHook):
    ...     priority = 30
    ...     def __init__(self, max_norm):
    ...         self.max_norm = max_norm
    ...     def __call__(self, ctx, stage, will_skip):
    ...         match stage:
    ...             case TrainingStage.DO_OPTIMIZER_STEP:
    ...                 if not will_skip:
    ...                     for opt in ctx.optimizers:
    ...                         params = (p for g in opt.param_groups for p in g["params"])
    ...                         torch.nn.utils.clip_grad_norm_(params, self.max_norm)
    ...                 return True, ctx.loss
    ...             case _:
    ...                 return True, ctx.loss
    """

    priority: int = 50

    def _runs_on_stage(self, stage: TrainingStage) -> bool:
        """Return ``True`` for the four stages a training-update hook claims."""
        return stage in _TRAINING_UPDATE_STAGES

    def __call__(
        self,
        ctx: TrainContext,
        stage: TrainingStage,
        will_skip: bool,
    ) -> tuple[bool, torch.Tensor]:
        """Run the hook for an update stage.

        Parameters
        ----------
        ctx : TrainContext
            Mutable training context shared by all hooks during the current
            stage dispatch.
        stage : TrainingStage
            Update stage currently being dispatched.
        will_skip : bool
            ``True`` when an earlier, higher-priority hook has already vetoed
            the gated operation for ``stage``. Hooks should use this to skip
            side effects that only make sense when the operation will run,
            while still performing any bookkeeping that must happen on every
            dispatch.

        Returns
        -------
        tuple[bool, torch.Tensor]
            ``(proceed, loss)``. ``proceed`` controls the skip signal passed
            to subsequent hooks: ``True`` keeps the pipeline proceeding,
            while ``False`` causes later hooks to receive ``will_skip=True``
            and skips the gated operation for ``stage``. ``loss`` is the loss
            tensor to pass to subsequent hooks; return ``ctx.loss`` unchanged
            when the hook does not transform the loss.
        """
        return True, ctx.loss

    def __add__(
        self, other: TrainingUpdateHook | TrainingUpdateOrchestrator
    ) -> TrainingUpdateOrchestrator:
        """Compose this hook with another update hook or orchestrator.

        Parameters
        ----------
        other : TrainingUpdateHook | TrainingUpdateOrchestrator
            Hook or orchestrator to compose with this hook.

        Returns
        -------
        TrainingUpdateOrchestrator
            Orchestrator containing this hook and ``other``. Hook execution
            order is determined by ``priority`` after composition.
        """
        if not isinstance(other, (TrainingUpdateHook, TrainingUpdateOrchestrator)):
            return NotImplemented
        return TrainingUpdateOrchestrator(self, other)


class TrainingUpdateOrchestrator:
    """Composes ``TrainingUpdateHook``s and drives backward/optimizer phases.

    Claims four training-update stages: ``BEFORE_BATCH``, ``DO_BACKWARD``,
    ``DO_OPTIMIZER_STEP``, ``AFTER_OPTIMIZER_STEP``. Per-stage behavior is
    selected via :func:`plum.dispatch` over ``Literal[TrainingStage.X]``
    rather than an ``if``/``match`` ladder.

    Parameters
    ----------
    *hooks : TrainingUpdateHook or TrainingUpdateOrchestrator
        Hooks to compose. Any orchestrator argument is flattened into its
        children. Members are sorted by ``priority`` ascending; ties
        preserve insertion order (Python's stable sort).

    Attributes
    ----------
    frequency : int
        Required by the :class:`Hook` Protocol; always ``1``.
    stage : None
        Set to ``None`` so the registry consults ``_runs_on_stage``.

    Raises
    ------
    TypeError
        If any positional argument is not a ``TrainingUpdateHook`` or
        ``TrainingUpdateOrchestrator``.

    Notes
    -----
    ``TrainingUpdateOrchestrator`` IS compatible with the standard
    :class:`Hook` Protocol -- it is the registry-facing wrapper around
    one or more :class:`TrainingUpdateHook` instances. Concrete training
    update hooks (``MixedPrecisionHook``, ``GradientClipHook``, etc.) are
    NOT directly Protocol-compliant on their own; they must be composed
    into an orchestrator before registration. The training strategy
    auto-wraps a bare :class:`TrainingUpdateHook` for convenience.

    On ``DO_BACKWARD`` each hook returns ``(_, loss)``; the orchestrator
    assigns ``ctx.loss = loss`` between hooks so the next hook sees the
    transformed value. ``backward()`` is called once on the final
    ``ctx.loss``. Example: a ``*0.5`` hook followed by a ``*2.0`` hook
    leaves ``ctx.loss`` equal to the original loss before backward.
    """

    frequency: int = 1
    stage = None

    def __init__(self, *hooks: TrainingUpdateHook | TrainingUpdateOrchestrator) -> None:
        flattened: list[TrainingUpdateHook] = []
        for i, h in enumerate(hooks):
            if isinstance(h, TrainingUpdateOrchestrator):
                flattened.extend(h._hooks)
            elif isinstance(h, TrainingUpdateHook):
                flattened.append(h)
            else:
                raise TypeError(
                    f"argument {i} must be TrainingUpdateHook or "
                    f"TrainingUpdateOrchestrator; got {type(h).__name__}. "
                    "If you have an iterable, call "
                    "TrainingUpdateOrchestrator(*hooks)."
                )
        flattened.sort(key=lambda h: h.priority)
        self._hooks: list[TrainingUpdateHook] = flattened

    def _runs_on_stage(self, stage: TrainingStage) -> bool:
        """Return ``True`` for the four stages this orchestrator claims."""
        return stage in _TRAINING_UPDATE_STAGES

    def _should_run_gated_stage(self, ctx: TrainContext, stage: TrainingStage) -> bool:
        """Run all hooks for a gated stage and return the any-veto-wins decision."""
        should_run = True
        for hook in self._hooks:
            proceed, _ = hook(ctx, stage, not should_run)
            _check_veto(proceed, hook, stage)
            should_run = proceed and should_run
        return should_run

    @plum.dispatch
    def __call__(
        self, ctx: TrainContext, stage: Literal[TrainingStage.BEFORE_BATCH]
    ) -> None:
        # situation where this may skip is gradient accumulation; otherwise
        # the typical workflow would be to actually zero gradients
        if self._should_run_gated_stage(ctx, stage):
            zero_gradients(ctx.optimizers)

    @plum.dispatch
    def __call__(  # noqa: F811
        self, ctx: TrainContext, stage: Literal[TrainingStage.DO_BACKWARD]
    ) -> None:
        for hook in self._hooks:
            _, loss = hook(ctx, stage, False)
            ctx.loss = loss
        ctx.loss.backward()

    @plum.dispatch
    def __call__(  # noqa: F811
        self, ctx: TrainContext, stage: Literal[TrainingStage.DO_OPTIMIZER_STEP]
    ) -> None:
        # situation where this might be skipped is during gradient
        # accumulation, or perhaps spike skipping
        should_run = self._should_run_gated_stage(ctx, stage)
        ctx.did_optimizer_step = False
        ctx.optimizer_step_skipped = not should_run
        if should_run:
            step_optimizers(ctx.optimizers)
            step_lr_schedulers(ctx.lr_schedulers)
            ctx.did_optimizer_step = True

    @plum.dispatch
    def __call__(  # noqa: F811
        self, ctx: TrainContext, stage: Literal[TrainingStage.AFTER_OPTIMIZER_STEP]
    ) -> None:
        for hook in self._hooks:
            hook(ctx, stage, ctx.optimizer_step_skipped)

    @plum.dispatch
    def __call__(self, ctx: TrainContext, stage: TrainingStage) -> None:  # noqa: F811
        # Catch-all for stages outside the four claimed; the registry's
        # _runs_on_stage filter normally prevents this from firing.
        return

    def __add__(
        self, other: TrainingUpdateHook | TrainingUpdateOrchestrator
    ) -> TrainingUpdateOrchestrator:
        """Implements the syntactic sugar to compose multiple update hooks together"""
        if not isinstance(other, (TrainingUpdateHook, TrainingUpdateOrchestrator)):
            return NotImplemented
        return TrainingUpdateOrchestrator(self, other)
