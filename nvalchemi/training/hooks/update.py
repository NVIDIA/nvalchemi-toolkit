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

from collections.abc import Iterator, Sequence
from types import TracebackType
from typing import TYPE_CHECKING, Any, ClassVar

from nvalchemi.hooks._context import TrainContext
from nvalchemi.hooks._protocol import Hook
from nvalchemi.training._stages import TrainingStage
from nvalchemi.training.optimizers import (
    _is_metric_driven,
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
    hooks: Sequence[Hook | TrainingUpdateHook | TrainingUpdateOrchestrator],
) -> list[Hook | TrainingUpdateOrchestrator]:
    """Fold TrainingUpdateHook/Orchestrator instances into a single orchestrator."""
    others: list[Hook] = []
    update_hooks: list[TrainingUpdateHook | TrainingUpdateOrchestrator] = []
    update_insertion_index: int | None = None
    n_orch = 0
    for h in hooks:
        if isinstance(h, TrainingUpdateOrchestrator):
            if update_insertion_index is None:
                update_insertion_index = len(others)
            update_hooks.append(h)
            n_orch += 1
        elif isinstance(h, TrainingUpdateHook):
            if update_insertion_index is None:
                update_insertion_index = len(others)
            update_hooks.append(h)
        else:
            others.append(h)
    if not update_hooks:
        return list(hooks)
    if n_orch > 1:
        raise ValueError(_MULTIPLE_ORCHESTRATOR_MSG)
    if len(update_hooks) == 1 and isinstance(
        update_hooks[0], TrainingUpdateOrchestrator
    ):
        folded = update_hooks[0]
    else:
        folded = TrainingUpdateOrchestrator(*update_hooks)
    insert_at = (
        update_insertion_index if update_insertion_index is not None else len(others)
    )
    result: list[Hook | TrainingUpdateOrchestrator] = list(others)
    result.insert(insert_at, folded)
    return result


def _check_veto(decision: object, hook: object, stage: TrainingStage) -> None:
    """Validate that ``__call__`` returned a strict ``bool`` for ``proceed``."""
    if not isinstance(decision, bool):
        raise TypeError(
            f"{type(hook).__name__}.__call__(stage={stage.name}) must return "
            f"(bool, Tensor | None); proceed got {type(decision).__name__}. "
            "Return True to proceed or False to skip."
        )


def _require_loss(
    loss: torch.Tensor | None, hook: object, stage: TrainingStage
) -> torch.Tensor:
    """Return ``loss`` or raise a stage-specific error for missing losses."""
    if loss is None:
        raise TypeError(
            f"{type(hook).__name__} did not provide a Tensor loss for "
            f"{stage.name}; got None."
        )
    return loss


def _get_scaler_scale(scaler: object) -> float | None:
    """Return the scaler scale as ``float`` when the scaler exposes one."""
    get_scale = getattr(scaler, "get_scale", None)
    if get_scale is None:
        return None
    try:
        return float(get_scale())
    except (TypeError, ValueError):
        return None


def _grad_scaler_step_skipped(
    grad_scaler: Any, opt: torch.optim.Optimizer
) -> bool | None:
    """Return whether ``grad_scaler.step(opt)`` skipped the optimizer step."""
    try:
        found_inf = grad_scaler._found_inf_per_device(opt)
    except Exception:
        return None
    try:
        return any(bool(v.item()) for v in found_inf.values())
    except Exception:
        return None


def _step_optimizers_with_context(ctx: TrainContext) -> bool:
    """Step optimizers/schedulers and return whether optimizer stepping ran."""
    if ctx.grad_scaler is None:
        step_optimizers(ctx.optimizers)
        step_lr_schedulers(ctx.lr_schedulers)
        return True

    if not ctx.lr_schedulers or all(sched is None for sched in ctx.lr_schedulers):
        pre_scale = _get_scaler_scale(ctx.grad_scaler)
        for opt in ctx.optimizers:
            ctx.grad_scaler.step(opt)
        ctx.grad_scaler.update()
        post_scale = _get_scaler_scale(ctx.grad_scaler)
        return (
            True if pre_scale is None or post_scale is None else post_scale >= pre_scale
        )

    skipped_flags: list[bool | None] = []
    for opt in ctx.optimizers:
        ctx.grad_scaler.step(opt)
        skipped_flags.append(_grad_scaler_step_skipped(ctx.grad_scaler, opt))

    need_fallback = any(flag is None for flag in skipped_flags)
    pre_scale = _get_scaler_scale(ctx.grad_scaler) if need_fallback else None
    ctx.grad_scaler.update()
    post_scale = _get_scaler_scale(ctx.grad_scaler) if need_fallback else None
    fallback_skipped = (
        need_fallback
        and pre_scale is not None
        and post_scale is not None
        and post_scale < pre_scale
    )
    schedulers = list(ctx.lr_schedulers)
    if len(schedulers) < len(skipped_flags):
        schedulers.extend([None] * (len(skipped_flags) - len(schedulers)))
    step_skipped_flags = [
        skipped is True or (fallback_skipped and skipped is None)
        for skipped in skipped_flags
    ]
    if not any(step_skipped_flags):
        step_lr_schedulers(ctx.lr_schedulers)
        return True
    for sched, step_skipped in zip(schedulers, step_skipped_flags, strict=True):
        if sched is None:
            continue
        if _is_metric_driven(sched):
            continue
        if step_skipped:
            continue
        sched.step()
    return False


class TrainingUpdateHook:
    """Base class for hooks that customize training-update phases.

    Subclasses override :meth:`__call__` and dispatch on ``stage`` to
    handle one or more claimed stages: ``BEFORE_BATCH``, ``DO_BACKWARD``,
    ``DO_OPTIMIZER_STEP``, ``AFTER_OPTIMIZER_STEP``.
    Compose via ``+`` to build a :class:`TrainingUpdateOrchestrator`.
    See :ref:`training-update-hooks` for the stage contract and restrictions
    each update hook must follow.

    Attributes
    ----------
    priority : int
        Dispatch order within an orchestrator; lower runs first. Canonical
        buckets: 10 = gradient accumulation, 20 = mixed precision,
        30 = gradient clipping, 40 = spike skipping. Default 50.
    _exclusive_update_key : str | None
        Optional key for hook families that must appear at most once inside
        an orchestrator.

    Notes
    -----
    ``TrainingUpdateHook`` is NOT directly compatible with the standard
    :class:`Hook` Protocol -- its ``__call__`` signature includes a
    ``will_skip`` argument and returns ``(bool, torch.Tensor | None)`` rather
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
      final loss. Hooks that run on stages other than ``DO_BACKWARD`` may
      return ``None`` for ``loss`` because the orchestrator ignores it
      there.

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
    _exclusive_update_key: ClassVar[str | None] = None

    def _runs_on_stage(self, stage: TrainingStage) -> bool:
        """Return ``True`` for stages a training-update hook claims."""
        return stage in _TRAINING_UPDATE_STAGES

    def __call__(
        self,
        ctx: TrainContext,
        stage: TrainingStage,
        will_skip: bool,
    ) -> tuple[bool, torch.Tensor | None]:
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
        tuple[bool, torch.Tensor | None]
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
    """Composes :class:`TrainingUpdateHook` instances and drives updates.

    Claims the training-update stages ``BEFORE_BATCH``, ``DO_BACKWARD``,
    ``DO_OPTIMIZER_STEP``, ``AFTER_OPTIMIZER_STEP``. The strategy also calls
    the orchestrator during ``SETUP`` so child hooks can initialize runtime
    state before the first batch. Per-stage behavior is
    selected by direct :class:`TrainingStage` comparisons to avoid per-batch
    multiple-dispatch overhead.
    See :ref:`training-update-hooks` for the stage contract enforced by the
    orchestrator.

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
    update hooks (``EMAHook``, ``GradientClipHook``, etc.) are
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
        exclusive_hooks: dict[str, TrainingUpdateHook] = {}
        for hook in flattened:
            key = hook._exclusive_update_key
            if key is None:
                continue
            if key in exclusive_hooks:
                first = type(exclusive_hooks[key]).__name__
                second = type(hook).__name__
                raise ValueError(
                    f"Only one update hook with exclusive key {key!r} may be "
                    f"registered; got {first} and {second}."
                )
            exclusive_hooks[key] = hook
        self._hooks: list[TrainingUpdateHook] = flattened
        self._optimizer_step_skipped = False

    def _runs_on_stage(self, stage: TrainingStage) -> bool:
        """Return ``True`` for the stages this orchestrator claims."""
        return stage in _TRAINING_UPDATE_STAGES

    def __enter__(self) -> TrainingUpdateOrchestrator:
        """Enter lifecycle contexts owned by child update hooks."""
        for hook in self._hooks:
            enter = getattr(hook, "__enter__", None)
            if enter is not None:
                enter()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Exit or close lifecycle contexts owned by child update hooks."""
        for hook in reversed(self._hooks):
            exit_ = getattr(hook, "__exit__", None)
            if exit_ is not None:
                exit_(exc_type, exc, tb)
            else:
                close = getattr(hook, "close", None)
                if close is not None:
                    close()

    def close(self) -> None:
        """Close child update hooks that expose ``close``."""
        for hook in reversed(self._hooks):
            close = getattr(hook, "close", None)
            if close is not None:
                close()

    @property
    def optimizer_step_skipped(self) -> bool:
        """Whether the most recent optimizer-step stage was vetoed."""
        return self._optimizer_step_skipped

    def iter_hooks(self) -> Iterator[TrainingUpdateHook]:
        """Yield child update hooks in orchestrator dispatch order."""
        return iter(self._hooks)

    def _should_run_gated_stage(self, ctx: TrainContext, stage: TrainingStage) -> bool:
        """Run all hooks for a gated stage and return the any-veto-wins decision."""
        should_run = True
        for hook in self._hooks:
            proceed, _ = hook(ctx, stage, not should_run)
            _check_veto(proceed, hook, stage)
            should_run = proceed and should_run
        return should_run

    def __call__(self, ctx: TrainContext, stage: TrainingStage) -> None:
        """Run orchestrator logic for ``stage`` when it is an update stage."""
        match stage:
            case TrainingStage.SETUP:
                for hook in self._hooks:
                    hook(ctx, stage, False)
            case TrainingStage.BEFORE_BATCH:
                # situation where this may skip is gradient accumulation; otherwise
                # the typical workflow would be to actually zero gradients
                if self._should_run_gated_stage(ctx, stage):
                    zero_gradients(ctx.optimizers)
                    clear_filtered = getattr(
                        ctx.workflow, "_zero_optimizer_filtered_gradients", None
                    )
                    if callable(clear_filtered):
                        clear_filtered(ctx.optimizers)
            case TrainingStage.DO_BACKWARD:
                for hook in self._hooks:
                    _, loss = hook(ctx, stage, False)
                    ctx.loss = _require_loss(loss, hook, stage)
                _require_loss(ctx.loss, self, stage).backward()
            case TrainingStage.DO_OPTIMIZER_STEP:
                # situation where this might be skipped is during gradient
                # accumulation, or perhaps spike skipping
                should_run = self._should_run_gated_stage(ctx, stage)
                if should_run:
                    should_run = _step_optimizers_with_context(ctx)
                self._optimizer_step_skipped = not should_run
            case TrainingStage.AFTER_OPTIMIZER_STEP:
                for hook in self._hooks:
                    hook(ctx, stage, self._optimizer_step_skipped)

    def __add__(
        self, other: TrainingUpdateHook | TrainingUpdateOrchestrator
    ) -> TrainingUpdateOrchestrator:
        """Implements the syntactic sugar to compose multiple update hooks together"""
        if not isinstance(other, (TrainingUpdateHook, TrainingUpdateOrchestrator)):
            return NotImplemented
        return TrainingUpdateOrchestrator(self, other)
