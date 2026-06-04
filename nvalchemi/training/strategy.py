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
"""Training strategy lifecycle and default forward-pass helper.

``TrainingStrategy`` wires one named model (``"main"``) or a dictionary-like
collection of named models through a user-supplied ``training_fn``.
Single-model strategies call ``training_fn(model, batch)``; named-model
strategies call ``training_fn(models, batch)`` for distillation or multi-model
workflows.
Models omitted from optimizer configs are temporarily set to eval mode and
frozen during ``run``. Named-model training functions that use omitted models as
teacher/auxiliary networks must run those forward passes under
``torch.no_grad()`` or detach returned tensors unless autograd through those
outputs is intentionally required.

Loss hooks see live autograd-connected losses from ``AFTER_LOSS`` through
``BEFORE_BACKWARD``. From ``AFTER_BACKWARD`` onward the hook context carries
detached loss tensors so logging hooks do not accidentally retain graphs.
"""

from __future__ import annotations

import itertools
import math
import warnings
from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import nullcontext
from pathlib import Path
from types import TracebackType
from typing import TYPE_CHECKING, Annotated, Any

import torch
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    SkipValidation,
    field_validator,
    model_validator,
)
from torch.optim.lr_scheduler import LRScheduler

from nvalchemi._serialization import _import_cls
from nvalchemi._typing import ModelOutputs
from nvalchemi.distributed import DistributedManager
from nvalchemi.hooks._context import TrainContext
from nvalchemi.hooks._protocol import Hook
from nvalchemi.hooks._registry import HookRegistryMixin
from nvalchemi.models.base import BaseModelMixin
from nvalchemi.training import _spec_utils as strategy_spec
from nvalchemi.training import _strategy_validation as strategy_validation
from nvalchemi.training._spec import create_model_spec
from nvalchemi.training._stages import TrainingStage
from nvalchemi.training.distributed import get_rank as get_distributed_rank
from nvalchemi.training.hooks import TrainingUpdateHook, TrainingUpdateOrchestrator
from nvalchemi.training.hooks.update import (
    _fold_training_update_hooks,
    _hook_claims_stage,
)
from nvalchemi.training.losses.composition import (
    BaseLossFunction,
    ComposedLossFunction,
    ComposedLossOutput,
    _ProductWeight,
    loss_component_to_spec,
)
from nvalchemi.training.optimizers import (
    OptimizerConfig,
    _normalize_optimizer_configs,
    setup_optimizers,
    step_lr_schedulers,
    step_optimizers,
    zero_gradients,
)
from nvalchemi.training.runtime import freeze_unconfigured_models, move_to_devices

if TYPE_CHECKING:
    from nvalchemi.data.batch import Batch
    from nvalchemi.training._checkpoint import CheckpointValidator

__all__ = ["TrainingStrategy", "default_training_fn"]

_RESTART_COUNTER_FIELDS = (
    "step_count",
    "batch_count",
    "epoch_count",
    "epoch_step_count",
)


def _loss_weight_to_spec(weight: Any) -> Any:
    """Serialize a composed-loss weight schedule while leaving scalars unchanged."""
    if isinstance(weight, _ProductWeight):
        return create_model_spec(
            type(weight),
            left=_loss_weight_to_spec(weight.left),
            right=_loss_weight_to_spec(weight.right),
        )
    if hasattr(weight, "model_dump"):
        return create_model_spec(type(weight), **weight.model_dump())
    return weight


def _validate_single_do_claimants(
    hooks: Sequence[Hook],
    *,
    extra_hook: Hook | None = None,
    extra_stage: TrainingStage | None = None,
) -> None:
    """Raise if more than one hook claims a DO update stage."""
    candidates: list[Hook] = list(hooks)
    if extra_hook is not None and all(h is not extra_hook for h in candidates):
        candidates.append(extra_hook)
    for do_stage in (TrainingStage.DO_BACKWARD, TrainingStage.DO_OPTIMIZER_STEP):
        claimants = [
            h
            for h in candidates
            if _hook_claims_stage(h, do_stage)
            or (h is extra_hook and extra_stage == do_stage)
        ]
        if len(claimants) > 1:
            names = ", ".join(type(h).__name__ for h in claimants)
            migration_hint = (
                " If one claimant is a plain DO-stage hook that should compose "
                "with update policies, implement it as TrainingUpdateHook so it "
                "runs inside the TrainingUpdateOrchestrator."
                if any(isinstance(h, TrainingUpdateOrchestrator) for h in claimants)
                else " Compose claim semantics are reserved for a future feature."
            )
            raise ValueError(
                f"At most one hook may claim {do_stage.name}; got "
                f"{len(claimants)}: {names}.{migration_hint}"
            )


def default_training_fn(model: BaseModelMixin, batch: Batch) -> dict[str, torch.Tensor]:
    """Run a forward pass and prefix output keys with ``predicted_``.

    Parameters
    ----------
    model : BaseModelMixin
        A wrapped MLIP whose ``__call__`` returns model outputs.
    batch : Batch
        Input batch of atomic graphs.

    Returns
    -------
    dict[str, torch.Tensor]
        Predictions keyed by ``predicted_<output_name>`` with ``None`` outputs
        omitted.
    """
    outputs: ModelOutputs = model(batch)
    return {
        f"predicted_{key}": value for key, value in outputs.items() if value is not None
    }


class TrainingStrategy(BaseModel, HookRegistryMixin):
    """Pydantic-driven supervised training loop for MLIP models.

    Attributes
    ----------
    models : dict[str, BaseModelMixin]
        Named models visible to ``training_fn`` and hooks. Single-model inputs
        are stored under ``"main"``; :class:`torch.nn.ModuleDict` inputs are
        accepted and normalized to a plain ``dict``.
    optimizer_configs : dict[str, list[OptimizerConfig]]
        Optimizer/scheduler configs keyed by model name. Keys may target a
        subset of ``models``; omitted models are frozen/eval during ``run``.
    num_epochs : int | None
        Epoch count; mutually exclusive with ``num_steps``. At runtime,
        epochs are converted into a target step count from the dataloader
        length and ``epoch_step_modifier``.
    num_steps : int | None
        Target step count; mutually exclusive with ``num_epochs``.
    epoch_step_modifier : float
        Positive multiplier applied when converting ``num_epochs`` to a
        target step count. Hooks may inspect this value through
        ``ctx.workflow``.
    hooks : list[Hook | TrainingUpdateHook | TrainingUpdateOrchestrator]
        Hooks executed at the stages declared by :class:`TrainingStage`.
        Bare :class:`TrainingUpdateHook` instances are auto-wrapped into a
        single :class:`TrainingUpdateOrchestrator` (see Notes). Duplicate
        hook object instances are rejected, and the list is **not**
        expected to be mutated once the ``TrainingStrategy`` context
        manager has been entered.
    training_fn : Callable[..., Mapping[str, torch.Tensor]]
        Explicit forward-pass callable. Single-model strategies call
        ``(model, batch)``; named-model strategies call ``(models, batch)``.
    loss_fn : ComposedLossFunction
        Composed loss whose components drive target collection. Leaf losses are
        accepted and normalized to one-component composed losses.
    devices : list[torch.device]
        One device shared by all models, or one device per model for helper
        placement. Named-model ``run`` currently supports one device only.
    distributed_manager : DistributedManager | None
        Optional external distributed manager. The strategy passes this through
        hook contexts for distributed-aware hooks.
    step_count : int
        Runtime optimizer-step counter, excluded from specs. Batches whose
        optimizer step is skipped by update hooks do not advance this counter.
    batch_count : int
        Runtime batch counter, excluded from specs. This advances for every
        completed batch, including batches whose optimizer step is skipped.
    epoch_count : int
        Runtime epoch counter, excluded from specs.
    epoch_step_count : int
        Runtime counter for batches consumed within the current epoch,
        excluded from specs.

    Notes
    -----
    Use :meth:`to_spec_dict` / :meth:`from_spec_dict` for JSON-based save/load.
    Optimizer configs, loss specs, devices, importable training functions, and
    best-effort model specs are serialized. Runtime ``models`` and
    ``training_fn`` overrides passed to :meth:`from_spec_dict` take precedence;
    the serialized model call mode is used only when no runtime model override
    is supplied. ``hooks``, ``step_count``, ``batch_count``, ``epoch_count``,
    and ``epoch_step_count`` remain runtime-only.

    Bare :class:`TrainingUpdateHook` instances are auto-wrapped into a single
    :class:`TrainingUpdateOrchestrator` on registration; the orchestrator owns
    the ``zero_gradients`` / ``backward`` / ``optimizer.step`` /
    ``scheduler.step`` calls that the strategy otherwise issues by default.
    Construction-time hook validation errors surface as
    :class:`pydantic.ValidationError`; :meth:`register_hook` raises
    :class:`ValueError` directly.
    """

    models: dict[str, BaseModelMixin]
    optimizer_configs: dict[str, list[OptimizerConfig]] = Field(default_factory=dict)
    num_epochs: int | None = Field(default=None, ge=1)
    num_steps: int | None = Field(default=None, ge=1)
    epoch_step_modifier: float = Field(default=1.0, gt=0, allow_inf_nan=False)
    hooks: list[Hook | TrainingUpdateHook | TrainingUpdateOrchestrator] = Field(
        default_factory=list,
        description=(
            "Hooks to run at training stages. Accepts ``Hook`` Protocol "
            "instances, bare ``TrainingUpdateHook`` instances (auto-wrapped "
            "into a single ``TrainingUpdateOrchestrator``), or an explicit "
            "``TrainingUpdateOrchestrator``. Example: "
            "``hooks=[CheckpointHook(...), MyClipGradHook()]``."
        ),
    )
    training_fn: Callable[..., Mapping[str, torch.Tensor]] | None = None
    loss_fn: ComposedLossFunction
    devices: list[torch.device] = Field(default_factory=lambda: [torch.device("cpu")])
    distributed_manager: Annotated[DistributedManager | None, SkipValidation()] = Field(
        default=None,
        exclude=True,
    )
    step_count: int = Field(default=0, ge=0, exclude=True)
    batch_count: int = Field(default=0, ge=0, exclude=True)
    epoch_count: int = Field(default=0, ge=0, exclude=True)
    epoch_step_count: int = Field(default=0, ge=0, exclude=True)
    single_model_input: bool = Field(default=False, exclude=True)

    _context_depth: int = PrivateAttr(default=0)
    _ctx: TrainContext | None = PrivateAttr(default=None)
    _has_do_backward_claim: bool = PrivateAttr(default=False)
    _has_do_optimizer_step_claim: bool = PrivateAttr(default=False)
    _has_update_orchestrator: bool = PrivateAttr(default=False)
    _resume_optimizer_state: bool = PrivateAttr(default=False)

    _active_dataloader: Any = PrivateAttr(default=None)

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
        # To minimize overhead, validation is only performed at the
        # initial construction
        validate_assignment=False,
        revalidate_instances="never",
    )

    _stage_type = TrainingStage

    @property
    def epoch(self) -> int:
        """Backward-compatible alias for :attr:`epoch_count`."""
        return self.epoch_count

    @epoch.setter
    def epoch(self, value: int) -> None:
        self.epoch_count = value

    @property
    def active_dataloader(self) -> Any:
        """Return the dataloader currently owned by the training workflow."""
        return self._active_dataloader

    @active_dataloader.setter
    def active_dataloader(self, dataloader: Any) -> None:
        """Set the dataloader currently owned by the training workflow."""
        self._active_dataloader = dataloader

    @model_validator(mode="before")
    @classmethod
    def _normalize_inputs(cls, data: Any) -> Any:
        """Normalize model and optimizer input shapes before field validation."""
        if not isinstance(data, dict):
            return data
        normalized = dict(data)
        raw_models = normalized.get("models")
        single_model_input = isinstance(raw_models, BaseModelMixin)
        if "models" in normalized:
            normalized["models"] = strategy_validation._normalize_models(raw_models)
        if "optimizer_configs" in normalized:
            normalized["optimizer_configs"] = _normalize_optimizer_configs(
                normalized["optimizer_configs"], single_model_input=single_model_input
            )
        if "epoch" in normalized and "epoch_count" not in normalized:
            normalized["epoch_count"] = normalized.pop("epoch")
        normalized["single_model_input"] = single_model_input
        return normalized

    @field_validator("loss_fn", mode="before")
    @classmethod
    def _normalize_loss_fn(cls, value: Any) -> Any:
        """Normalize a leaf loss into a one-component composed loss."""
        if isinstance(value, ComposedLossFunction):
            return value
        elif isinstance(value, BaseLossFunction):
            return ComposedLossFunction([value])
        else:
            raise RuntimeError(
                "Only loss functions that inherit `BaseLossFunction` or"
                " a composition of loss functions is accepted."
            )

    @field_validator("training_fn", mode="before")
    @classmethod
    def _resolve_training_fn(cls, value: Any) -> Any:
        """Resolve a dotted-path string to a callable, or accept a callable as-is."""
        if isinstance(value, str):
            value = strategy_spec._resolve_dotted_callable(value)
        if value is None:
            raise ValueError(strategy_validation._TRAINING_FN_REQUIRED_MESSAGE)
        if not callable(value):
            raise ValueError(
                f"training_fn must be callable or a dotted path string, got "
                f"{type(value).__name__}."
            )
        return value

    @field_validator("hooks", mode="before")
    @classmethod
    def _autowrap_update_hooks(cls, value: Any) -> Any:
        """Fold bare ``TrainingUpdateHook`` instances into a single orchestrator."""
        if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
            return value
        return _fold_training_update_hooks(value)

    @model_validator(mode="after")
    def _validate_strategy(self) -> TrainingStrategy:
        """Enforce model, duration, optimizer, and device consistency."""
        have_epochs = self.num_epochs is not None
        have_steps = self.num_steps is not None
        if have_epochs == have_steps:
            raise ValueError(
                "Exactly one of num_epochs or num_steps must be set; "
                f"got num_epochs={self.num_epochs!r}, num_steps={self.num_steps!r}."
            )
        if not self.models:
            raise ValueError("models must contain at least one BaseModelMixin.")
        if not self.optimizer_configs:
            raise ValueError(
                "optimizer_configs must configure at least one model; "
                "got an empty mapping."
            )
        for idx, cfgs in self.optimizer_configs.items():
            if idx not in self.models:
                raise ValueError(
                    f"optimizer_configs key {idx!r} is not present in models; "
                    f"available model keys: {sorted(self.models)}."
                )
            if not cfgs:
                raise ValueError(
                    f"optimizer_configs[{idx!r}] must contain at least one "
                    "OptimizerConfig."
                )
        if not self.devices:
            raise ValueError("devices must contain at least one torch.device.")
        n_devices = len(self.devices)
        if n_devices not in (1, len(self.models)):
            raise ValueError(
                f"devices must have length 1 or len(models)={len(self.models)}; "
                f"got {n_devices}."
            )
        if self.training_fn is None:
            raise ValueError(strategy_validation._TRAINING_FN_REQUIRED_MESSAGE)
        strategy_validation._validate_training_fn_call_shape(
            self.training_fn, single_model_input=self.single_model_input
        )
        hook_ids = [id(hook) for hook in self.hooks]
        if len(hook_ids) != len(set(hook_ids)):
            raise ValueError(
                "hooks must not contain duplicate hook instances; pass distinct "
                "hook objects instead."
            )
        _validate_single_do_claimants(self.hooks)
        return self

    def model_post_init(self, __context: Any) -> None:
        """Initialize hook storage, per-run counters, and cached target keys."""
        self._init_hooks(list(self.hooks))
        self._refresh_hook_claim_flags()
        self._last_batch: Batch | None = None
        self._last_losses: ComposedLossOutput | None = None
        self._last_loss: torch.Tensor | None = None
        self._optimizers: list[torch.optim.Optimizer] = []
        self._lr_schedulers: list[LRScheduler | None] = []
        self._context_depth = 0
        self._ctx = None
        seen_keys: set[str] = set()
        target_keys: list[str] = []
        for component in self.loss_fn.components:
            key = getattr(component, "target_key", None)
            if key is None or key in seen_keys:
                continue
            seen_keys.add(key)
            target_keys.append(key)
        self._target_keys: tuple[str, ...] = tuple(target_keys)

    def _refresh_hook_claim_flags(self) -> None:
        """Recompute cached DO-stage claim and orchestrator-presence flags."""
        self._has_do_backward_claim = (
            sum(
                1
                for hook in self.hooks
                if _hook_claims_stage(hook, TrainingStage.DO_BACKWARD)
            )
            == 1
        )
        self._has_do_optimizer_step_claim = (
            sum(
                1
                for hook in self.hooks
                if _hook_claims_stage(hook, TrainingStage.DO_OPTIMIZER_STEP)
            )
            == 1
        )
        self._has_update_orchestrator = any(
            isinstance(hook, TrainingUpdateOrchestrator) for hook in self.hooks
        )

    def _replace_hooks_with_registry_validation(self, hooks: Sequence[Hook]) -> None:
        """Replace hook storage after validating each hook through the base registry."""
        previous_hooks = self.hooks
        self.hooks = []
        try:
            for hook in hooks:
                HookRegistryMixin.register_hook(self, hook)
        except Exception:
            self.hooks = previous_hooks
            raise

    def register_hook(
        self,
        hook: Hook | TrainingUpdateHook | TrainingUpdateOrchestrator,
        stage: TrainingStage | None = None,
    ) -> None:
        """Register a hook, auto-wrapping bare update hooks when needed."""
        is_update = isinstance(hook, (TrainingUpdateHook, TrainingUpdateOrchestrator))
        if is_update and stage is not None:
            raise ValueError(
                "stage= is not supported for TrainingUpdateHook or "
                "TrainingUpdateOrchestrator registration. Update hooks declare "
                "their stages through _runs_on_stage and are auto-wrapped into "
                "one TrainingUpdateOrchestrator."
            )
        if not is_update:
            _validate_single_do_claimants(
                self.hooks, extra_hook=hook, extra_stage=stage
            )
            super().register_hook(hook, stage=stage)
            self._refresh_hook_claim_flags()
            return
        folded = _fold_training_update_hooks([*self.hooks, hook])
        _validate_single_do_claimants(folded)
        self._replace_hooks_with_registry_validation(folded)
        self._refresh_hook_claim_flags()

    def _build_context(self, batch: Batch | None) -> TrainContext:
        """Build a TrainContext, reusing the per-batch cache when populated."""
        if self._ctx is not None:
            return self._ctx
        global_rank = get_distributed_rank(self.distributed_manager)
        return TrainContext(
            batch=batch,
            model=self.models.get("main"),
            global_rank=global_rank,
            workflow=self,
            step_count=self.step_count,
            batch_count=self.batch_count,
            epoch_step_count=self.epoch_step_count,
            models=self.models,
            epoch=self.epoch_count,
            loss=self._last_loss,
            losses=self._last_losses,
            optimizers=self._optimizers,
            lr_schedulers=self._lr_schedulers,
        )

    def _run_hooks(self, stage: TrainingStage, batch: Batch) -> None:
        """Dispatch hooks for ``stage`` with an early-return fast path."""
        if not self.hooks:
            return
        self._call_hooks(stage, batch)

    def _refresh_hook_counters(self) -> None:
        """Mirror current strategy counters into the cached hook context."""
        if self._ctx is None:
            return
        self._ctx.step_count = self.step_count
        self._ctx.batch_count = self.batch_count
        self._ctx.epoch_step_count = self.epoch_step_count
        self._ctx.epoch = self.epoch_count

    def __enter__(self) -> TrainingStrategy:
        """Enter hook context managers registered on this strategy."""
        if self._context_depth > 0:
            self._context_depth += 1
            return self
        for hook in self.hooks:
            if hasattr(hook, "__enter__"):
                hook.__enter__()
        self._context_depth = 1
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Exit or close hook contexts registered on this strategy."""
        if self._context_depth == 0:
            return
        self._context_depth -= 1
        if self._context_depth > 0:
            return
        for hook in reversed(self.hooks):
            if hasattr(hook, "__exit__"):
                hook.__exit__(exc_type, exc, tb)
            elif hasattr(hook, "close"):
                hook.close()

    def _prepare_setup_hooks(self) -> None:
        """Allow hooks to prepare runtime state before device placement."""
        for hook in self.hooks:
            prepare = getattr(hook, "prepare_strategy", None)
            if callable(prepare):
                prepare(self)

    def _run_setup_hooks(self, dataloader: Any = None) -> Any:
        """Run setup-stage hooks and return the active dataloader."""
        if not self.hooks:
            return dataloader
        self.active_dataloader = dataloader
        ctx = self._build_context(None)
        for hook in self.hooks:
            if not _hook_claims_stage(hook, TrainingStage.SETUP):
                continue
            if self.step_count % hook.frequency != 0:
                continue
            hook(ctx, TrainingStage.SETUP)
        return self.active_dataloader

    def _validate_runtime_devices(self) -> None:
        """Raise for runtime device layouts that cannot be executed."""
        if not self.single_model_input and len(self.devices) > 1:
            raise ValueError(
                "Named-model training with multiple devices is unsupported: "
                "training_fn(models, batch) receives one batch on one device. "
                "Use a single shared device or pass models=model for "
                "single-model behavior."
            )

    def _setup_runtime_optimizers(
        self, *, rebuild: bool = False
    ) -> tuple[list[torch.optim.Optimizer], list[LRScheduler | None]]:
        """Build or reuse flattened runtime optimizer/scheduler lists."""
        if not rebuild and self._optimizers:
            return self._optimizers, self._lr_schedulers

        flat_opts: list[torch.optim.Optimizer] = []
        flat_scheds: list[LRScheduler | None] = []
        for pairs in setup_optimizers(self.models, self.optimizer_configs).values():
            for opt, sched in pairs:
                flat_opts.append(opt)
                flat_scheds.append(sched)
        self._optimizers = flat_opts
        self._lr_schedulers = flat_scheds
        return flat_opts, flat_scheds

    def train_batch(self, batch: Batch) -> None:
        """Train on a single batch using the configured training flow.

        This public one-batch API is intended for interactive workflows and
        tests where the caller already has a batch in hand. It runs the
        per-batch stages from ``BEFORE_BATCH`` through ``AFTER_BATCH``, but it
        does not run the outer ``BEFORE_TRAINING``/``AFTER_TRAINING`` or
        epoch-level hooks and does not enforce ``num_epochs``/``num_steps``.
        It still advances runtime counters: ``batch_count`` and
        ``epoch_step_count`` advance for every completed batch, while
        ``step_count`` advances only when the optimizer step executes.

        Optimizers and schedulers are built from ``optimizer_configs`` on first
        use and then reused by subsequent ``train_batch`` calls. Full
        :meth:`run` calls continue to rebuild optimizer state at the start of
        the run.

        Parameters
        ----------
        batch : Batch
            Batch to train on.
        """
        strategy_context = nullcontext(self) if self._context_depth > 0 else self
        with strategy_context:
            self._prepare_setup_hooks()
            self._validate_runtime_devices()
            self.models = move_to_devices(self.models, self.devices)
            self._run_setup_hooks()
            flat_opts, flat_scheds = self._setup_runtime_optimizers()
            batch = batch.to(self.devices[0], non_blocking=True)
            self._update_hook_snapshot(batch=batch, loss_out=None)

            with freeze_unconfigured_models(self.models, self.optimizer_configs):
                self._train_batch_with_optimizers(batch, flat_opts, flat_scheds)

    def _train_batch_with_optimizers(
        self,
        batch: Batch,
        flat_opts: list[torch.optim.Optimizer],
        flat_scheds: list[LRScheduler | None],
    ) -> None:
        """Forward-backward-optimize a single batch with hook dispatch."""
        self._optimizers = flat_opts
        self._lr_schedulers = flat_scheds
        self._ctx = self._build_context(batch) if self.hooks else None

        try:
            self._run_hooks(TrainingStage.BEFORE_BATCH, batch)
            if not self._has_update_orchestrator:
                zero_gradients(flat_opts)
            self._run_hooks(TrainingStage.BEFORE_FORWARD, batch)
            model_arg = self.models["main"] if self.single_model_input else self.models
            predictions = self.training_fn(model_arg, batch)
            self._run_hooks(TrainingStage.AFTER_FORWARD, batch)

            self._run_hooks(TrainingStage.BEFORE_LOSS, batch)
            loss_out = self._compute_losses(
                predictions,
                batch,
                step=self.step_count,
                epoch=self.epoch_count,
            )
            self._update_hook_snapshot(loss_out=loss_out)
            self._run_hooks(TrainingStage.AFTER_LOSS, batch)

            self._run_hooks(TrainingStage.BEFORE_BACKWARD, batch)
            if self._has_do_backward_claim:
                self._run_hooks(TrainingStage.DO_BACKWARD, batch)
            elif self._ctx is not None and self._ctx.loss is not None:
                self._ctx.loss.backward()
            else:
                loss_out["total_loss"].backward()
            self._run_backward_completion(batch, loss_out)
            optimizer_step_ran = self._run_optimizer_step_phase(
                batch, flat_opts, flat_scheds
            )

            self.batch_count += 1
            self.epoch_step_count += 1
            if optimizer_step_ran:
                self.step_count += 1
            self._refresh_hook_counters()
            self._run_hooks(TrainingStage.AFTER_BATCH, batch)
        finally:
            self._ctx = None

    def _run_backward_completion(
        self, batch: Batch, loss_out: ComposedLossOutput
    ) -> None:
        """Publish detached losses, then fire the gradient-available stage."""
        if self.hooks:
            self._update_hook_snapshot(loss_out=loss_out, detach=True)
        self._run_hooks(TrainingStage.AFTER_BACKWARD, batch)

    def _run_optimizer_step_phase(
        self,
        batch: Batch,
        flat_opts: list[torch.optim.Optimizer],
        flat_scheds: list[LRScheduler | None],
    ) -> bool:
        """Run the last pre-step hook, step owner, and step-aware post hook."""
        self._run_hooks(TrainingStage.BEFORE_OPTIMIZER_STEP, batch)
        if self._has_do_optimizer_step_claim:
            self._run_hooks(TrainingStage.DO_OPTIMIZER_STEP, batch)
            optimizer_step_ran = self._optimizer_step_ran_after_do_stage()
        else:
            step_optimizers(flat_opts)
            step_lr_schedulers(flat_scheds)
            optimizer_step_ran = True
        self._run_hooks(TrainingStage.AFTER_OPTIMIZER_STEP, batch)
        return optimizer_step_ran

    def _optimizer_step_ran_after_do_stage(self) -> bool:
        """Return whether the DO optimizer-step owner reported an executed step."""
        for hook in self.hooks:
            if isinstance(hook, TrainingUpdateOrchestrator):
                return not hook.optimizer_step_skipped
        return True

    def _assemble_targets(self, batch: Batch) -> dict[str, torch.Tensor]:
        """Look up each cached target key on ``batch``."""
        targets: dict[str, torch.Tensor] = {}
        for key in self._target_keys:
            try:
                targets[key] = getattr(batch, key)
            except AttributeError as exc:
                raise AttributeError(
                    f"Batch is missing target attribute {key!r} required by "
                    f"{type(self.loss_fn).__name__}."
                ) from exc
        return targets

    def _compute_losses(
        self,
        predictions: Mapping[str, torch.Tensor],
        batch: Batch,
        *,
        step: int,
        epoch: int,
    ) -> ComposedLossOutput:
        """Run ``loss_fn`` with graph metadata threaded as keyword kwargs."""
        graph_meta: dict[str, Any] = {}
        for attr in ("batch_idx", "num_graphs", "num_nodes_per_graph"):
            value = getattr(batch, attr, None)
            if value is not None:
                graph_meta[attr] = value
        return self.loss_fn(
            predictions,
            self._assemble_targets(batch),
            step=step,
            epoch=epoch,
            **graph_meta,
        )

    def _update_hook_snapshot(
        self,
        *,
        batch: Batch | None = None,
        loss_out: ComposedLossOutput | None = None,
        detach: bool = False,
    ) -> None:
        """Single mutation point for hook-visible transient state."""
        if batch is not None:
            self._last_batch = batch
        if loss_out is None:
            self._last_loss = None
            self._last_losses = None
        elif detach:
            self._last_loss = loss_out["total_loss"].detach()
            self._last_losses = {
                "total_loss": loss_out["total_loss"].detach(),
                "per_component_total": {
                    k: v.detach() for k, v in loss_out["per_component_total"].items()
                },
                "per_component_weight": dict(loss_out["per_component_weight"]),
                "per_component_raw_weight": dict(loss_out["per_component_raw_weight"]),
                "per_component_sample": {
                    k: v.detach() for k, v in loss_out["per_component_sample"].items()
                },
            }
        else:
            self._last_loss = loss_out["total_loss"]
            self._last_losses = loss_out
        if self._ctx is not None:
            if batch is not None:
                self._ctx.batch = batch
            self._ctx.loss = self._last_loss
            self._ctx.losses = self._last_losses
            self._refresh_hook_counters()

    def _dataloader_length(self, dataloader: Iterable[Batch]) -> int | None:
        """Return ``len(dataloader)`` when available without iterating it."""
        try:
            return len(dataloader)  # type: ignore[arg-type]
        except TypeError:
            return None

    def _resolve_target_step_count(self, batches_per_epoch: int | None) -> int:
        """Resolve ``num_steps``/``num_epochs`` to an absolute step target."""
        if self.num_steps is not None:
            return self.num_steps

        if batches_per_epoch is None:
            raise ValueError(
                "num_epochs requires a sized dataloader so epochs can be "
                "converted to a target step count. Use num_steps for unsized "
                "iterables."
            )

        if batches_per_epoch <= 0:
            raise ValueError(
                "dataloader must contain at least one batch when num_epochs "
                "is configured."
            )
        if self.num_epochs is None:
            raise RuntimeError("TrainingStrategy has neither num_epochs nor num_steps.")
        return math.ceil(self.num_epochs * batches_per_epoch * self.epoch_step_modifier)

    def _set_sampler_epoch(self, dataloader: Iterable[Batch]) -> None:
        """Set distributed/data-parallel sampler epoch when supported."""
        candidates = (
            getattr(dataloader, "sampler", None),
            getattr(getattr(dataloader, "batch_sampler", None), "sampler", None),
        )
        seen: set[int] = set()
        for sampler in candidates:
            if sampler is None or id(sampler) in seen:
                continue
            seen.add(id(sampler))
            set_epoch = getattr(sampler, "set_epoch", None)
            if callable(set_epoch):
                set_epoch(self.epoch_count)
                return

    def _prepare_epoch_step_count(self, batches_per_epoch: int | None) -> None:
        """Infer or normalize intra-epoch progress for restartable runs."""
        if batches_per_epoch is None or batches_per_epoch <= 0:
            return
        if self.epoch_step_count >= batches_per_epoch:
            extra_epochs, self.epoch_step_count = divmod(
                self.epoch_step_count, batches_per_epoch
            )
            self.epoch_count += extra_epochs

        completed_epoch_batches = self.epoch_count * batches_per_epoch
        raw_progress = self.batch_count or self.step_count
        if self.epoch_step_count:
            expected_progress = completed_epoch_batches + self.epoch_step_count
            if raw_progress and raw_progress != expected_progress:
                raise ValueError(
                    "restart counters are inconsistent: batch_count or "
                    "step_count does not match epoch_count * len(dataloader) "
                    "+ epoch_step_count."
                )
            self.batch_count = max(self.batch_count, expected_progress)
            return

        if raw_progress < completed_epoch_batches:
            raise ValueError(
                "restart counters are inconsistent: batch_count or step_count "
                "is smaller "
                "than epoch_count * len(dataloader)."
            )
        elapsed_epoch_steps = raw_progress - completed_epoch_batches
        extra_epochs, self.epoch_step_count = divmod(
            elapsed_epoch_steps, batches_per_epoch
        )
        self.epoch_count += extra_epochs
        self.batch_count = max(self.batch_count, raw_progress)

    def run(
        self,
        dataloader: Iterable[Batch],
    ) -> None:
        """Execute the training loop over ``dataloader``.

        Parameters
        ----------
        dataloader : Iterable[Batch]
            Any iterable of batches; need not be a ``DataLoader``.
            The configured duration targets effective optimizer/scheduler
            steps. Batches whose optimizer step is skipped still advance the
            dataloader-position counters.

        Raises
        ------
        ValueError
            If named-model training is configured with multiple devices, or if
            the dataloader produces no batches before the configured target
            step count is reached.
        """
        training_started = False
        strategy_context = nullcontext(self) if self._context_depth > 0 else self
        with strategy_context:
            self._prepare_setup_hooks()
            self._validate_runtime_devices()
            self.models = move_to_devices(self.models, self.devices)
            dataloader = self._run_setup_hooks(dataloader)
            batches_per_epoch = self._dataloader_length(dataloader)
            target_step_count = self._resolve_target_step_count(batches_per_epoch)
            if self.step_count >= target_step_count:
                return
            self._prepare_epoch_step_count(batches_per_epoch)

            primary_device = self.devices[0]
            flat_opts, flat_scheds = self._setup_runtime_optimizers(
                rebuild=not self._resume_optimizer_state
            )

            with freeze_unconfigured_models(self.models, self.optimizer_configs):
                for _epoch_idx in itertools.count():
                    self._set_sampler_epoch(dataloader)
                    processed_epoch_batch = False
                    exhausted_dataloader = True
                    for batch_idx, batch in enumerate(dataloader):
                        if batch_idx < self.epoch_step_count:
                            continue
                        if self.step_count >= target_step_count:
                            exhausted_dataloader = False
                            break
                        batch = batch.to(primary_device, non_blocking=True)
                        self._update_hook_snapshot(batch=batch, loss_out=None)
                        if not training_started:
                            self._run_hooks(TrainingStage.BEFORE_TRAINING, batch)
                            training_started = True
                        if self.epoch_step_count == 0:
                            self._run_hooks(TrainingStage.BEFORE_EPOCH, batch)

                        self._train_batch_with_optimizers(batch, flat_opts, flat_scheds)
                        processed_epoch_batch = True
                        if (
                            batches_per_epoch is not None
                            and self.epoch_step_count >= batches_per_epoch
                        ):
                            exhausted_dataloader = True
                            break
                        if self.step_count >= target_step_count:
                            exhausted_dataloader = False
                            break

                    if (
                        not processed_epoch_batch
                        and self.step_count < target_step_count
                    ):
                        raise ValueError(
                            "dataloader produced no batches before reaching "
                            "the target step count; ensure the dataloader is "
                            "non-empty, re-iterable, and compatible with the "
                            "restored epoch_step_count."
                        )

                    if exhausted_dataloader:
                        self.epoch_count += 1
                        self.epoch_step_count = 0
                        self._refresh_hook_counters()
                        self._run_hooks(TrainingStage.AFTER_EPOCH, self._last_batch)
                    if self.step_count >= target_step_count:
                        break

                if self._last_batch is not None:
                    self._update_hook_snapshot(loss_out=None)
                    self._run_hooks(TrainingStage.AFTER_TRAINING, self._last_batch)

    def to_spec_dict(self) -> dict[str, Any]:
        """Serialize declarative training knobs to a JSON-ready dict.

        Returns
        -------
        dict[str, Any]
            JSON-ready bundle suitable for :func:`json.dumps`.
        """
        component_specs = [
            loss_component_to_spec(comp) for comp in self.loss_fn.components
        ]
        loss_fn_spec = create_model_spec(
            type(self.loss_fn),
            components=component_specs,
            weights=[_loss_weight_to_spec(weight) for weight in self.loss_fn._weights],
            normalize_weights=self.loss_fn.normalize_weights,
        )
        spec = {
            "optimizer_configs": {
                key: [cfg.to_spec().model_dump() for cfg in cfgs]
                for key, cfgs in self.optimizer_configs.items()
            },
            "num_epochs": self.num_epochs,
            "num_steps": self.num_steps,
            "epoch_step_modifier": self.epoch_step_modifier,
            "devices": [str(device) for device in self.devices],
            "loss_fn_spec": loss_fn_spec.model_dump(),
            "model_specs": strategy_spec._model_specs_from_models(self.models),
            "single_model_input": self.single_model_input,
        }
        try:
            spec["training_fn"] = strategy_spec._callable_dotted_path(self.training_fn)
        except ValueError as exc:
            warnings.warn(
                f"Omitting non-importable training_fn from spec: {exc}",
                UserWarning,
                stacklevel=2,
            )
        return spec

    def to_checkpoint_dict(self) -> dict[str, Any]:
        """Serialize strategy recipe and restart counters for checkpoints.

        Returns
        -------
        dict[str, Any]
            JSON-ready checkpoint metadata. Model weights and optimizer state
            remain outside this payload in checkpoint ``state_dict`` files.
        """
        runtime_state = {key: getattr(self, key) for key in _RESTART_COUNTER_FIELDS}
        return {
            **self.to_spec_dict(),
            "strategy_cls": f"{type(self).__module__}.{type(self).__qualname__}",
            "runtime_state": runtime_state,
        }

    def save_checkpoint(
        self,
        root_folder: Path | str,
        *,
        checkpoint_index: int = -1,
    ) -> int:
        """Save this strategy as a restartable checkpoint.

        Parameters
        ----------
        root_folder : Path | str
            Root directory for checkpoint files.
        checkpoint_index : int, optional
            Checkpoint index to write. ``-1`` auto-increments from the latest
            manifest index, or starts at ``0`` when no manifest exists.

        Returns
        -------
        int
            The checkpoint index that was written.
        """
        from nvalchemi.training._checkpoint import save_checkpoint

        return save_checkpoint(
            root_folder,
            checkpoint_index=checkpoint_index,
            strategy=self,
        )

    @classmethod
    def load_checkpoint(
        cls,
        root_folder: Path | str,
        checkpoint_index: int = -1,
        map_location: str | torch.device | None = None,
        *,
        hooks: Sequence[Hook | TrainingUpdateHook | TrainingUpdateOrchestrator]
        | None = None,
        training_fn: Callable[..., Mapping[str, torch.Tensor]] | str | None = None,
        validators: Sequence[CheckpointValidator] | None = None,
    ) -> TrainingStrategy:
        """Load a restartable strategy checkpoint.

        This is the strategy-focused convenience wrapper around
        :func:`nvalchemi.training.load_checkpoint`. Use the module-level
        function when callers need the full manifest, component dictionaries,
        partial component loads, or foreign checkpoint adapters.

        Parameters
        ----------
        root_folder : Path | str
            Root directory containing checkpoint files.
        checkpoint_index : int, optional
            Checkpoint index to load. ``-1`` loads the latest manifest index.
        map_location : str | torch.device | None, optional
            Device override passed through to :func:`torch.load` and the
            restored strategy metadata.
        hooks : Sequence[Hook | TrainingUpdateHook | TrainingUpdateOrchestrator] | None, optional
            Runtime hooks to attach to the restored strategy.
        training_fn : Callable[..., Mapping[str, torch.Tensor]] | str | None, optional
            Runtime training function override. This is required when the saved
            strategy used a local or otherwise non-importable training
            function.
        validators : Sequence[CheckpointValidator] | None, optional
            Optional loaded-checkpoint validators forwarded to the lower-level
            loader.

        Returns
        -------
        TrainingStrategy
            Restored strategy with model, optimizer, scheduler, and runtime
            counters loaded.

        Raises
        ------
        ValueError
            If the checkpoint does not contain restartable strategy metadata.
        TypeError
            If the restored strategy is not an instance of ``cls``.
        """
        from nvalchemi.training._checkpoint import load_checkpoint

        loaded = load_checkpoint(
            root_folder,
            checkpoint_index=checkpoint_index,
            map_location=map_location,
            hooks=hooks,
            training_fn=training_fn,
            validators=validators,
        )
        if not isinstance(loaded, Mapping) or loaded.get("strategy") is None:
            raise ValueError(
                "TrainingStrategy.load_checkpoint requires a checkpoint saved "
                "from a TrainingStrategy. Use nvalchemi.training.load_checkpoint "
                "for component-only checkpoints."
            )
        strategy = loaded["strategy"]
        if not isinstance(strategy, cls):
            raise TypeError(
                f"Loaded strategy has type {type(strategy).__name__}, expected "
                f"{cls.__name__}."
            )
        return strategy

    @classmethod
    def from_spec_dict(
        cls,
        spec: Mapping[str, Any],
        *,
        models: strategy_validation.ModelInput | None = None,
        hooks: Sequence[Hook | TrainingUpdateHook | TrainingUpdateOrchestrator]
        | None = None,
        training_fn: Callable[..., Mapping[str, torch.Tensor]] | str | None = None,
    ) -> TrainingStrategy:
        """Rebuild a :class:`TrainingStrategy` from a :meth:`to_spec_dict` bundle.

        Parameters
        ----------
        spec : Mapping[str, Any]
            A dict produced by :meth:`to_spec_dict`, optionally after a JSON round-trip.
        models : BaseModelMixin | dict[str, BaseModelMixin] | torch.nn.ModuleDict | None, optional
            Runtime model override(s).
        hooks : Sequence[Hook | TrainingUpdateHook | TrainingUpdateOrchestrator] | None, optional
            Runtime hooks; defaults to an empty list. Bare update hooks are
            auto-wrapped into a single orchestrator.
        training_fn : Callable[..., Mapping[str, torch.Tensor]] | str | None, optional
            Runtime callable or dotted-path override.

        Returns
        -------
        TrainingStrategy
            A freshly validated strategy ready to :meth:`run`.
        """
        required = ("optimizer_configs", "devices", "loss_fn_spec")
        missing = [k for k in required if k not in spec]
        if missing:
            raise ValueError(
                f"from_spec_dict: spec is missing required key(s) {missing}. "
                f"Expected keys: {list(required)}."
            )
        model_input = strategy_spec._models_from_spec_and_overrides(
            spec.get("model_specs", {}),
            models,
            single_model_input=strategy_spec._single_model_input_from_spec(
                spec.get("single_model_input")
            ),
        )
        return cls(
            models=model_input,
            optimizer_configs=strategy_spec._optimizer_configs_from_spec(
                spec["optimizer_configs"]
            ),
            num_epochs=spec.get("num_epochs"),
            num_steps=spec.get("num_steps"),
            epoch_step_modifier=spec.get("epoch_step_modifier", 1.0),
            hooks=list(hooks) if hooks is not None else [],
            training_fn=strategy_spec._training_fn_from_spec(spec, training_fn),
            loss_fn=strategy_spec._loss_fn_from_spec(spec["loss_fn_spec"]),
            devices=strategy_spec._devices_from_spec(spec["devices"]),
        )

    @classmethod
    def from_checkpoint_dict(
        cls,
        spec: Mapping[str, Any],
        *,
        models: strategy_validation.ModelInput | None = None,
        hooks: Sequence[Hook | TrainingUpdateHook | TrainingUpdateOrchestrator]
        | None = None,
        training_fn: Callable[..., Mapping[str, torch.Tensor]] | str | None = None,
    ) -> TrainingStrategy:
        """Rebuild a strategy from checkpoint metadata.

        Parameters
        ----------
        spec : Mapping[str, Any]
            A dict produced by :meth:`to_checkpoint_dict`.
        models : BaseModelMixin | dict[str, BaseModelMixin] | torch.nn.ModuleDict | None, optional
            Runtime model override(s), normally the models loaded from the
            checkpoint weight files.
        hooks : Sequence[Hook | TrainingUpdateHook | TrainingUpdateOrchestrator] | None, optional
            Runtime hooks appended by the caller.
        training_fn : Callable[..., Mapping[str, torch.Tensor]] | str | None, optional
            Runtime callable or dotted-path override.

        Returns
        -------
        TrainingStrategy
            A strategy with declarative fields and restart counters restored.
        """
        strategy_cls = cls
        raw_strategy_cls = spec.get("strategy_cls")
        if raw_strategy_cls is not None:
            if not isinstance(raw_strategy_cls, str):
                raise ValueError(
                    "from_checkpoint_dict: 'strategy_cls' must be a dotted "
                    f"class path string; got {type(raw_strategy_cls).__name__}."
                )
            imported = _import_cls(raw_strategy_cls)
            if not issubclass(imported, cls):
                raise ValueError(
                    f"from_checkpoint_dict: {raw_strategy_cls!r} must resolve "
                    f"to a {cls.__name__} subclass."
                )
            strategy_cls = imported

        strategy = strategy_cls.from_spec_dict(
            spec,
            models=models,
            hooks=hooks,
            training_fn=training_fn,
        )
        runtime_state = spec.get("runtime_state", {})
        if runtime_state is None:
            runtime_state = {}
        if not isinstance(runtime_state, Mapping):
            raise ValueError(
                "from_checkpoint_dict: 'runtime_state' must be a mapping when "
                f"present; got {type(runtime_state).__name__}."
            )
        for key in _RESTART_COUNTER_FIELDS:
            if key in runtime_state:
                value = int(runtime_state[key])
                if value < 0:
                    raise ValueError(
                        "from_checkpoint_dict: runtime counter "
                        f"{key!r} must be non-negative; got {value}."
                    )
                setattr(strategy, key, value)
        return strategy
