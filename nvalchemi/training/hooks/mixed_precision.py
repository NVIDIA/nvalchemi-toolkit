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
"""Mixed-precision update hook driving ``torch.amp.autocast`` and ``GradScaler``.

See :class:`MixedPrecisionHook` for the user-facing API. The hook composes
through :class:`~nvalchemi.training.hooks.TrainingUpdateOrchestrator` so that
:class:`~nvalchemi.training.strategy.TrainingStrategy` remains free of any
AMP-specific code.
"""

from __future__ import annotations

from contextlib import AbstractContextManager, nullcontext
from types import TracebackType
from typing import Annotated, Any, ClassVar

import torch
from pydantic import AfterValidator, BaseModel, BeforeValidator, ConfigDict, PrivateAttr

from nvalchemi._serialization import _dtype_deserialize, _wrap_custom_type
from nvalchemi.hooks._context import TrainContext
from nvalchemi.training._stages import TrainingStage
from nvalchemi.training.hooks.update import TrainingUpdateHook

__all__ = ["MixedPrecisionHook"]


_SUPPORTED_PRECISIONS: tuple[torch.dtype, ...] = (
    torch.float32,
    torch.bfloat16,
    torch.float16,
)
"""Autocast dtypes this hook understands."""

_PRECISION_ALIASES: dict[str, str] = {
    "fp32": "float32",
    "bf16": "bfloat16",
    "fp16": "float16",
}
"""Common shorthand precision names accepted by :class:`MixedPrecisionHook`."""


def _supported_precision_names() -> str:
    """Return the supported precision names for validation messages."""
    return ", ".join(
        str(dtype).removeprefix("torch.") for dtype in _SUPPORTED_PRECISIONS
    )


def _deserialize_precision(value: Any) -> Any:
    """Deserialize canonical dtype strings plus supported shorthand aliases."""
    if not isinstance(value, str):
        return value
    normalized = value.removeprefix("torch.").lower()
    normalized = _PRECISION_ALIASES.get(normalized, normalized)
    try:
        return _dtype_deserialize(normalized)
    except (TypeError, ValueError) as exc:
        supported = _supported_precision_names()
        raise ValueError(
            f"MixedPrecisionHook.precision must be one of ({supported}); got {value!r}."
        ) from exc


def _restrict_precision(value: torch.dtype) -> torch.dtype:
    """Reject dtypes outside :data:`_SUPPORTED_PRECISIONS`."""
    if value not in _SUPPORTED_PRECISIONS:
        supported = _supported_precision_names()
        raise ValueError(
            f"MixedPrecisionHook.precision must be one of ({supported}); got {value!r}."
        )
    return value


Precision = Annotated[
    _wrap_custom_type(torch.dtype),
    BeforeValidator(_deserialize_precision),
    AfterValidator(_restrict_precision),
]
"""``torch.dtype`` field accepting canonical names, aliases, or dtype objects."""


class MixedPrecisionHook(BaseModel, TrainingUpdateHook):
    """Automatic-mixed-precision hook driving autocast and ``GradScaler``.

    ``MixedPrecisionHook`` is a
    :class:`~nvalchemi.training.hooks.TrainingUpdateHook`. When it is
    registered directly on :class:`~nvalchemi.training.strategy.TrainingStrategy`,
    the strategy auto-wraps it in a
    :class:`~nvalchemi.training.hooks.TrainingUpdateOrchestrator`. The
    orchestrator owns ``backward()`` and optimizer/scheduler stepping;
    this hook supplies a scaled loss, exposes ``ctx.grad_scaler`` for
    scaler-aware stepping, and unscales gradients immediately before an
    optimizer step proceeds so gradient accumulation can keep accumulating
    scaled gradients.

    The first :attr:`TrainingStage.BEFORE_BATCH` lazily constructs the
    autocast region on the workflow's primary device
    (``ctx.workflow.devices[0]``), so the hook need not know the device at
    construction time. For fp16, the same path also lazily constructs the
    :class:`torch.amp.GradScaler`. The autocast region is released inside
    :attr:`TrainingStage.DO_BACKWARD` before the orchestrator calls
    ``backward()``, while the scaler persists across batches. Force and
    stress predictions produced during the model forward, plus the configured
    training losses, are therefore inside the autocast region; backward is
    not.

    Precision modes:

    * :data:`torch.float32` — no autocast context or scaler is created; the
      hook is a functional no-op aside from participating in the orchestrated
      update path.
    * :data:`torch.bfloat16` — autocast casts eligible ops to ``bfloat16``.
      No gradient scaling because bf16's exponent range matches fp32.
    * :data:`torch.float16` — autocast casts eligible ops to ``float16``
      during forward and loss computation. The scaler scales the loss before
      the orchestrator calls ``backward()``, unscales gradients just before
      optimizer stepping,
      and skips optimizer steps that would otherwise consume ``inf``/``nan``
      gradients.

    Parameters
    ----------
    precision : torch.dtype
        Autocast dtype and scaler policy. Accepts either a
        :class:`torch.dtype` (e.g. ``torch.float16``) or the canonical
        string name (``"float32"``, ``"bfloat16"``, ``"float16"``), or a
        shorthand alias (``"fp32"``, ``"bf16"``, ``"fp16"``).

    Attributes
    ----------
    precision : torch.dtype
        Active autocast dtype.
    priority : int
        Training-update priority. Fixed at ``20`` so loss-scaling runs
        after gradient accumulation transforms and before gradient
        clipping / spike-skip hooks.

    Raises
    ------
    pydantic.ValidationError
        If ``precision`` is not one of the supported dtypes.

    Examples
    --------
    >>> import torch
    >>> from nvalchemi.training.hooks import MixedPrecisionHook
    >>> MixedPrecisionHook(precision=torch.bfloat16).precision
    torch.bfloat16
    >>> MixedPrecisionHook(precision="float16").precision
    torch.float16
    >>> MixedPrecisionHook(precision="bf16").precision
    torch.bfloat16

    Notes
    -----
    * When multiple optimizers are configured, every optimizer in
      ``ctx.optimizers`` is unscaled in list order immediately before
      stepping. The orchestrator advances each scheduler in
      ``ctx.lr_schedulers`` only when its paired optimizer step was not
      skipped by the scaler.
    * For gradient accumulation, accumulated gradients remain scaled until
      the effective batch is ready to step. Earlier-priority update hooks
      can veto :attr:`TrainingStage.DO_OPTIMIZER_STEP` to suppress unscale,
      scaler step, and scaler update for intermediate accumulation batches.
    * A strategy may register only one ``MixedPrecisionHook``. Multiple
      instances are rejected to prevent duplicated autocast/scaler operations.
    * Under ``precision=torch.float16`` on CPU, no warning is emitted and
      no exception is raised; the hook still drives ``backward()`` and
      ``step()`` through the same scaler path.
    """

    precision: Precision

    priority: ClassVar[int] = 20
    _exclusive_update_key: ClassVar[str | None] = "MixedPrecisionHook"

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=False,
        extra="forbid",
    )

    _autocast_ctx: torch.amp.autocast | None = PrivateAttr(default=None)
    _scaler: torch.amp.GradScaler | None = PrivateAttr(default=None)
    _active: bool = PrivateAttr(default=False)

    def __enter__(self) -> MixedPrecisionHook:
        """Enter the hook's context; lazy-init is deferred to workflow stages.

        Returns
        -------
        MixedPrecisionHook
            This hook instance, for ``with`` expressions.
        """
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Exit the autocast region and reset internal state for reuse.

        Parameters
        ----------
        exc_type : type[BaseException] | None
            Exception class raised inside the managed block, if any.
        exc : BaseException | None
            Exception instance raised inside the managed block, if any.
        tb : TracebackType | None
            Traceback associated with ``exc``, if any.
        """
        self._exit_autocast(exc_type, exc, tb)
        self._scaler = None

    def inference_autocast(self, device: torch.device) -> AbstractContextManager[None]:
        """Return the inference autocast context matching this precision.

        Parameters
        ----------
        device : torch.device
            Primary workflow device for the validation or inference pass.

        Returns
        -------
        contextlib.AbstractContextManager[None]
            No-op context for ``float32`` precision, otherwise a
            :class:`torch.amp.autocast` context using this hook's configured
            dtype. This helper intentionally does not create or touch a
            :class:`torch.amp.GradScaler`, which is training-update state.
        """
        if self.precision == torch.float32:
            return nullcontext()
        return torch.amp.autocast(
            device_type=device.type,
            dtype=self.precision,
            enabled=True,
        )

    def __call__(
        self,
        ctx: TrainContext,
        stage: TrainingStage,
        will_skip: bool,
    ) -> tuple[bool, torch.Tensor | None]:
        """Handle training-update stages inside ``TrainingUpdateOrchestrator``."""
        match stage:
            case TrainingStage.BEFORE_BATCH:
                self._enter_autocast(ctx)
            case TrainingStage.DO_BACKWARD:
                self._exit_autocast(None, None, None)
                if self.precision == torch.float16:
                    scaler = self._ensure_scaler(ctx)
                    ctx.grad_scaler = scaler
                    return True, scaler.scale(ctx.loss)
            case TrainingStage.DO_OPTIMIZER_STEP:
                if self.precision == torch.float16:
                    scaler = self._ensure_scaler(ctx)
                    ctx.grad_scaler = scaler
                    if not will_skip:
                        self._unscale_gradients(ctx)
            case TrainingStage.AFTER_OPTIMIZER_STEP:
                self._exit_autocast(None, None, None)
            case _:
                pass
        return True, ctx.loss

    def _ensure_scaler(self, ctx: TrainContext) -> torch.amp.GradScaler:
        """Lazily construct the fp16 scaler for this workflow device."""
        if self._scaler is None:
            device_type = ctx.workflow.devices[0].type
            self._scaler = torch.amp.GradScaler(
                device=device_type,
                enabled=True,
            )
        return self._scaler

    def _enter_autocast(self, ctx: TrainContext) -> None:
        """Enter the forward/loss autocast region for this batch."""
        if self.precision == torch.float32:
            return
        if self.precision == torch.float16:
            ctx.grad_scaler = self._ensure_scaler(ctx)
        device_type = ctx.workflow.devices[0].type
        if self._autocast_ctx is None:
            self._autocast_ctx = torch.amp.autocast(
                device_type=device_type,
                dtype=self.precision,
                enabled=True,
            )
            self._autocast_ctx.__enter__()
            self._active = True

    def _unscale_gradients(self, ctx: TrainContext) -> None:
        """Unscale gradients immediately before an optimizer step proceeds."""
        if self.precision != torch.float16:
            return
        if self._scaler is None:
            raise RuntimeError("MixedPrecisionHook: scaler not initialized.")
        for opt in ctx.optimizers:
            self._scaler.unscale_(opt)

    def _exit_autocast(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Exit the active autocast region while preserving scaler state."""
        if self._active and self._autocast_ctx is not None:
            self._autocast_ctx.__exit__(exc_type, exc, tb)
        self._autocast_ctx = None
        self._active = False
