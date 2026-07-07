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

from collections.abc import Mapping
from typing import TYPE_CHECKING, Annotated, Any, ClassVar

import torch
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, StringConstraints
from torch import nn
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

from nvalchemi.training._stages import TrainingStage
from nvalchemi.training.hooks.update import TrainingUpdateHook

if TYPE_CHECKING:
    import torch

    from nvalchemi.hooks._context import TrainContext


__all__ = ["EMAHook"]


def _unwrap_model(m: nn.Module) -> nn.Module:
    """Returns a nested module if it exists, otherwise no-op"""
    return m.module if hasattr(m, "module") else m


def _module_tensors(module: nn.Module) -> dict[str, torch.Tensor]:
    """Return registered parameters and buffers by name."""
    tensors = {
        name: param
        for name, param in module.named_parameters(recurse=True, remove_duplicate=False)
    }
    tensors.update(
        {
            name: buffer
            for name, buffer in module.named_buffers(
                recurse=True, remove_duplicate=False
            )
        }
    )
    return tensors


def _align_tensor_to_source(tensor: torch.Tensor, source: torch.Tensor) -> None:
    """Align a registered tensor to the source tensor's device and dtype."""
    dtype = source.dtype if tensor.is_floating_point() else tensor.dtype
    if tensor.device == source.device and tensor.dtype == dtype:
        return
    with torch.no_grad():
        tensor.data = tensor.data.to(device=source.device, dtype=dtype)
        if tensor.grad is not None:
            tensor.grad.data = tensor.grad.data.to(device=source.device, dtype=dtype)


def _align_to_source_tensors(
    target: nn.Module, source_tensors: Mapping[str, torch.Tensor]
) -> None:
    """Align target parameters and buffers to their corresponding source tensors."""
    for name, param in target.named_parameters(recurse=True, remove_duplicate=False):
        if name in source_tensors:
            _align_tensor_to_source(param, source_tensors[name])
    for name, buffer in target.named_buffers(recurse=True, remove_duplicate=False):
        if name in source_tensors:
            _align_tensor_to_source(buffer, source_tensors[name])


class EMAHook(BaseModel, TrainingUpdateHook):
    """Hook maintaining an exponential moving average of a training model.

    Runs through :class:`~nvalchemi.training.hooks.TrainingUpdateOrchestrator`
    and updates at :attr:`TrainingStage.AFTER_OPTIMIZER_STEP`. It lazily builds a
    :class:`~torch.optim.swa_utils.AveragedModel` wrapped around
    ``ctx.models[model_key]`` on the first eligible step, and updates it
    via :func:`~torch.optim.swa_utils.get_ema_multi_avg_fn` — no manual
    parameter arithmetic. The hook is a pure observer: it never calls
    ``backward()``, touches gradients, drives any optimizer / scheduler /
    ``GradScaler``, or mutates ``ctx.models``. If an earlier update hook
    vetoes :attr:`TrainingStage.DO_OPTIMIZER_STEP`, the orchestrator passes
    ``will_skip=True`` and EMA does not update on that batch.

    Access the averaged wrapper via :meth:`get_averaged_model`, which raises
    a :class:`RuntimeError` if no eligible step has yet triggered lazy
    initialization. A ``device``/``dtype`` field is omitted by design; after
    :class:`~torch.optim.swa_utils.AveragedModel` deep-copies the source,
    EMAHook aligns each averaged parameter and buffer to the corresponding
    source tensor's device and floating-point dtype. This keeps generated or
    monkey-patched modules whose deepcopy/load path materializes registered
    tensors on CPU or in a default dtype usable without model-specific hooks.

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

    Examples
    --------
    Checkpoint recipe for **inference / eval reload** of the EMA-averaged
    weights. Save ``hook.get_averaged_model().module`` alongside the base
    model and rebuild the :class:`~torch.optim.swa_utils.AveragedModel`
    wrapper after loading, because
    :func:`~nvalchemi.training.create_model_spec` only reconstructs plain
    :class:`~torch.nn.Module` objects:

    >>> from torch import nn  # doctest: +SKIP
    >>> from torch.optim.swa_utils import AveragedModel  # doctest: +SKIP
    >>> from nvalchemi.training import (  # doctest: +SKIP
    ...     EMAHook, create_model_spec, load_checkpoint, save_checkpoint,
    ... )
    >>> base = nn.Linear(4, 2)  # doctest: +SKIP
    >>> hook = EMAHook(model_key="main", decay=0.99)  # doctest: +SKIP
    >>> # ... training loop drives `hook` via TrainingStrategy ...
    >>> spec = create_model_spec(nn.Linear, in_features=4, out_features=2)  # doctest: +SKIP
    >>> save_checkpoint(  # doctest: +SKIP
    ...     "ckpt/",
    ...     models={
    ...         "main": (base, spec),
    ...         "main_ema": (hook.get_averaged_model().module, spec),
    ...     },
    ... )
    >>> loaded = load_checkpoint("ckpt/")  # doctest: +SKIP
    >>> reconstructed_ema = AveragedModel(loaded.models["main_ema"][0])  # doctest: +SKIP

    To **resume training with EMA continuing** from a checkpoint, use
    :meth:`state_dict` / :meth:`load_state_dict`, which round-trip
    ``num_updates`` and the averaged weights into a freshly constructed
    hook.

    Notes
    -----
    The default deepcopy-based construction does not support
    ``fully_shard`` (FSDP2) / DTensor models; override
    :meth:`_build_averaged_model` to supply a pre-built sharded copy.
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
    num_updates: Annotated[
        int,
        Field(
            ge=0,
            description="Number of EMA updates performed; restored from checkpoints.",
        ),
    ] = 0

    # Runs after lower-priority update hooks have made step/veto decisions.
    priority: ClassVar[int] = 50

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    _averaged_model: AveragedModel | None = PrivateAttr(default=None)
    _pending_averaged_state: dict[str, Any] | None = PrivateAttr(default=None)

    def _build_averaged_model(self, source: nn.Module) -> AveragedModel:
        """Build the :class:`AveragedModel` wrapping ``source``.

        Override point: a caller that owns model sharding can return a
        pre-built copy instead (the default deepcopy fails on a
        ``fully_shard``-ed source).
        """
        averaged = AveragedModel(
            source,
            multi_avg_fn=get_ema_multi_avg_fn(self.decay),
            use_buffers=self.use_buffers,
        )
        _align_to_source_tensors(averaged.module, _module_tensors(source))
        return averaged

    def _ensure_initialized(self, ctx: TrainContext) -> None:
        if self._averaged_model is not None:
            return
        try:
            source = ctx.models[self.model_key]
        except KeyError as exc:
            available = sorted(ctx.models.keys())
            raise KeyError(
                f"EMAHook could not resolve model_key={self.model_key!r}; "
                f"available keys in TrainContext.models: {available}"
            ) from exc

        self._averaged_model = self._build_averaged_model(_unwrap_model(source))
        if self._pending_averaged_state is not None:
            source_tensors = _module_tensors(_unwrap_model(source))
            self._averaged_model.load_state_dict(self._pending_averaged_state)
            _align_to_source_tensors(self._averaged_model.module, source_tensors)
            self._pending_averaged_state = None

    def _publish_averaged_model(self, ctx: TrainContext) -> None:
        """Publish averaged weights into the strategy inference-model slot."""
        setter = getattr(ctx.workflow, "set_inference_model", None)
        if setter is not None:
            setter(self.get_averaged_model().module, model_key=self.model_key)

    def __call__(
        self,
        ctx: TrainContext,
        stage: TrainingStage,
        will_skip: bool = False,
    ) -> tuple[bool, torch.Tensor | None]:
        """Initialize or update the averaged model at the relevant stages."""
        match stage:
            case TrainingStage.SETUP:
                # Build the EMA copy early so validation can use restored weights.
                self._ensure_initialized(ctx)
                self._publish_averaged_model(ctx)
            case TrainingStage.AFTER_OPTIMIZER_STEP:
                if will_skip:
                    return True, getattr(ctx, "loss", None)
                completed_step = ctx.step_count + 1
                if (
                    completed_step < self.start_step
                    or completed_step % self.update_every
                ):
                    return True, getattr(ctx, "loss", None)
                # Apply the actual EMA update only after an eligible optimizer step.
                self._ensure_initialized(ctx)
                source = ctx.models[self.model_key]
                self.get_averaged_model().update_parameters(_unwrap_model(source))
                self.num_updates += 1
                self._publish_averaged_model(ctx)
            case _:
                # Other training stages do not affect EMA state.
                pass
        return True, getattr(ctx, "loss", None)

    def get_averaged_model(self) -> AveragedModel:
        """Return the :class:`AveragedModel` wrapper or raise if uninitialized.

        Raises
        ------
        RuntimeError
            If neither setup nor an eligible training step has initialized EMA.
        """
        if self._averaged_model is None:
            raise RuntimeError(
                "EMAHook has not initialized an averaged model yet. "
                "The hook initializes during TrainingStage.SETUP or the first "
                f"eligible AFTER_OPTIMIZER_STEP (start_step={self.start_step}, "
                f"update_every={self.update_every})."
            )
        return self._averaged_model

    def state_dict(self) -> dict[str, Any]:
        """Return a serializable snapshot of hook state.

        Returns
        -------
        dict[str, Any]
            Contains the config fields, ``num_updates``, and — if
            available — ``averaged_model_state`` sourced from the live
            :class:`AveragedModel` or, before lazy init, from any
            stashed pending state. No ``device`` key is emitted.
        """
        out: dict[str, Any] = self.model_dump()
        if self._averaged_model is not None:
            out["averaged_model_state"] = self._averaged_model.state_dict()
        elif self._pending_averaged_state is not None:
            out["averaged_model_state"] = self._pending_averaged_state
        return out

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore hook counters and averaged weights from a prior snapshot.

        Parameters
        ----------
        state : Mapping[str, Any]
            Mapping produced by :meth:`state_dict`. Missing config keys
            and ``num_updates`` are ignored. Missing
            ``averaged_model_state`` clears any prior live or pending
            averaged state.
            Any present config key must equal the corresponding
            constructor field.

        Raises
        ------
        ValueError
            If a config field in ``state`` differs from this hook's
            current field.

        Notes
        -----
        Before lazy init, ``averaged_model_state`` is stashed and
        applied during :meth:`_ensure_initialized`. Clearing on absence
        prevents stale averaged state from surviving a config-only
        reload. Checkpoint loaders may still choose a ``map_location``,
        but EMAHook reapplies per-tensor device and floating-point dtype
        placement after loading averaged state so registered tensors remain
        usable for validation.
        """
        for key in type(self).model_fields:
            if key == "num_updates":
                continue
            if key in state and state[key] != (current := getattr(self, key)):
                raise ValueError(
                    f"EMAHook checkpoint conflict: {key}={state[key]!r} vs "
                    f"constructor {key}={current!r}; construct the hook "
                    "with matching config or load into a fresh instance"
                )
        if "num_updates" in state:
            self.num_updates = int(state["num_updates"])
        if "averaged_model_state" in state:
            if self._averaged_model is None:
                self._pending_averaged_state = state["averaged_model_state"]
            else:
                tensors = _module_tensors(self._averaged_model.module)
                self._averaged_model.load_state_dict(state["averaged_model_state"])
                _align_to_source_tensors(self._averaged_model.module, tensors)
                self._pending_averaged_state = None
        else:
            self._averaged_model = None
            self._pending_averaged_state = None
