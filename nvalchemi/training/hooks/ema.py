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
"""Exponential-moving-average (EMA) training hook."""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Any, ClassVar

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, StringConstraints
from torch import nn
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

from nvalchemi.training._stages import TrainingStage

if TYPE_CHECKING:
    from nvalchemi.hooks._context import HookContext


__all__ = ["EMAHook"]


def _unwrap_model(m: nn.Module) -> nn.Module:
    """Returns a nested module if it exists, otherwise no-op"""
    return m.module if hasattr(m, "module") else m


class EMAHook(BaseModel):
    """Hook maintaining an exponential moving average of a training model.

    Fires at :attr:`TrainingStage.AFTER_OPTIMIZER_STEP`, lazily builds a
    :class:`~torch.optim.swa_utils.AveragedModel` wrapped around
    ``ctx.models[model_key]`` on the first eligible step, and updates it
    via :func:`~torch.optim.swa_utils.get_ema_multi_avg_fn` — no manual
    parameter arithmetic. The hook is a pure observer: it never calls
    ``backward()``, touches gradients, drives any optimizer / scheduler /
    ``GradScaler``, or mutates ``ctx.models``.

    Access the averaged wrapper via :meth:`get_averaged_model`, which raises
    a :class:`RuntimeError` if no eligible step has yet triggered lazy
    initialization. A ``device`` field is omitted by design;
    ``AveragedModel`` defaults to the source model's device.

    Parameters
    ----------
    model_key : str, optional
        Key identifying the source model inside ``ctx.models``. Default ``"main"``.
    decay : float, optional
        EMA decay factor in ``[0.0, 1.0)``. Default ``0.999``.
    update_every : int, optional
        Positive step stride for averaging updates. Default ``1``.
    start_step : int, optional
        Non-negative minimum completed step before updates begin. Default ``0``.
    use_buffers : bool, optional
        Forwarded to :class:`AveragedModel`; when ``True`` also averages
        module buffers. Default ``True``.

    Raises
    ------
    pydantic.ValidationError
        If any field violates its declared bounds or an unknown kwarg is passed.
    KeyError
        On first eligible call, if ``model_key`` is missing from ``ctx.models``.
    RuntimeError
        From :meth:`get_averaged_model` when called before lazy init.

    See Also
    --------
    torch.optim.swa_utils.AveragedModel : Underlying averaging wrapper.
    torch.optim.swa_utils.get_ema_multi_avg_fn : Factory for the EMA averaging function.

    Notes
    -----
    This hook targets single-node training (Phase 2 scope: DDP and
    unwrapped modules). FSDP and DTensor-sharded models are out of
    scope and not validated here — wrapping one will fail downstream
    inside :meth:`AveragedModel.update_parameters`.
    """

    model_key: Annotated[
        str,
        StringConstraints(strip_whitespace=True, min_length=1),
        Field(description="Key identifying the source model in ctx.models."),
    ] = "main"
    decay: Annotated[
        float, Field(ge=0.0, lt=1.0, description="EMA decay factor in [0.0, 1.0).")
    ] = 0.999
    update_every: Annotated[
        int,
        Field(
            gt=0,
            description="Completed-step interval between EMA updates (global-modulo).",
        ),
    ] = 1
    start_step: Annotated[
        int, Field(ge=0, description="First completed step eligible for EMA updates.")
    ] = 0
    use_buffers: Annotated[
        bool,
        Field(
            description="If True, also average module buffers (e.g. BN running stats)."
        ),
    ] = True

    # Hook Protocol attributes — ClassVar so Pydantic treats them as constants.
    stage: ClassVar[TrainingStage] = TrainingStage.AFTER_OPTIMIZER_STEP
    frequency: ClassVar[int] = 1

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    _averaged_model: AveragedModel | None = PrivateAttr(default=None)
    _num_updates: int = PrivateAttr(default=0)
    # reserved for Step 2: state_dict/load_state_dict
    _pending_averaged_state: dict[str, Any] | None = PrivateAttr(default=None)

    def _ensure_initialized(self, ctx: HookContext) -> None:
        if self._averaged_model is not None:
            return
        try:
            source = ctx.models[self.model_key]
        except KeyError as exc:
            available = sorted(ctx.models.keys())
            raise KeyError(
                f"EMAHook could not resolve model_key={self.model_key!r}; "
                f"available keys in HookContext.models: {available}"
            ) from exc

        inner = _unwrap_model(source)
        self._averaged_model = AveragedModel(
            inner,
            multi_avg_fn=get_ema_multi_avg_fn(self.decay),
            use_buffers=self.use_buffers,
        )

    def __call__(self, ctx: HookContext, stage: TrainingStage) -> None:
        """Update the averaged model when stage and step filter match."""
        if stage is not self.stage:
            return
        completed_step = ctx.step_count + 1
        if completed_step < self.start_step or completed_step % self.update_every:
            return
        self._ensure_initialized(ctx)
        source = ctx.models[self.model_key]
        self.get_averaged_model().update_parameters(_unwrap_model(source))
        self._num_updates += 1

    def get_averaged_model(self) -> AveragedModel:
        """Return the :class:`AveragedModel` wrapper or raise if uninitialized.

        Raises
        ------
        RuntimeError
            If no eligible training step has triggered lazy initialization.
        """
        if self._averaged_model is None:
            raise RuntimeError(
                f"EMAHook has not observed an eligible AFTER_OPTIMIZER_STEP yet "
                f"(start_step={self.start_step}, update_every={self.update_every}). "
                "The hook initializes lazily on the first eligible call."
            )
        return self._averaged_model
