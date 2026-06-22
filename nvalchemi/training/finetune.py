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
"""Fine-tuning strategy conveniences built on :class:`TrainingStrategy`."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from pydantic import Field, model_validator

from nvalchemi.training import _spec_utils as strategy_spec
from nvalchemi.training import _strategy_validation as strategy_validation
from nvalchemi.training._spec import BaseSpec, create_model_spec_from_json
from nvalchemi.training.hooks.finetune import (
    FreezeMode,
    ModulePatchHook,
    TrainableParameterHook,
)
from nvalchemi.training.strategy import TrainingStrategy

__all__ = ["FineTuningStrategy"]

_DEFAULT_PRETRAINED_CHECKPOINT_LR = 1e-5


def _apply_checkpoint_finetuning_defaults(
    strategy_kwargs: dict[str, Any],
    source_metadata: Mapping[str, Any] | None,
    *,
    use_original_loss: bool,
    use_original_opt_class: bool,
    optimizer_lr: float | None,
) -> None:
    """Fill omitted fine-tuning config from checkpoint strategy metadata.

    This helper intentionally stays outside ``FineTuningStrategy`` because it
    adapts checkpoint metadata into constructor kwargs; it is not a strategy
    behavior users should call directly. Keep source reuse explicit and
    initialization-only: rebuild serializable loss or optimizer config when the
    caller opts in, but never restore optimizer state, scheduler state, hooks,
    counters, or epoch/step limits.

    Future source-reuse options should be added here rather than expanding the
    public strategy surface with helper methods.
    """
    needs_loss = use_original_loss and "loss_fn" not in strategy_kwargs
    needs_optimizer = (
        use_original_opt_class and "optimizer_configs" not in strategy_kwargs
    )
    if not needs_loss and not needs_optimizer:
        return
    if source_metadata is None:
        requested = []
        if needs_loss:
            requested.append("loss_fn")
        if needs_optimizer:
            requested.append("optimizer_configs")
        raise ValueError(
            "Cannot reuse original "
            f"{', '.join(requested)} because checkpoint has no strategy metadata."
        )

    # re-use the original loss target if requested, and it exists
    if needs_loss:
        loss_spec = source_metadata.get("loss_fn_spec")
        if loss_spec is None:
            raise ValueError(
                "Cannot reuse original loss_fn because checkpoint metadata "
                "does not contain loss_fn_spec."
            )
        strategy_kwargs["loss_fn"] = strategy_spec._loss_fn_from_spec(loss_spec)

    # when the user requests to re-use the original optimizer, we
    # reconstruct it but use the fine-tuning LR instead of the base
    if needs_optimizer:
        raw_configs = source_metadata.get("optimizer_configs")
        if raw_configs is None:
            raise ValueError(
                "Cannot reuse original optimizer_configs because checkpoint "
                "metadata does not contain optimizer_configs."
            )
        optimizer_configs = strategy_spec._optimizer_configs_from_spec(raw_configs)
        if optimizer_lr is not None:
            for configs in optimizer_configs.values():
                for config in configs:
                    config.optimizer_kwargs = {
                        **config.optimizer_kwargs,
                        "lr": optimizer_lr,
                    }
        strategy_kwargs["optimizer_configs"] = optimizer_configs


class FineTuningStrategy(TrainingStrategy):
    """Training strategy for patching modules and selecting trainable parameters.

    ``FineTuningStrategy`` is intended for workflows where a pretrained model
    is loaded first and then adapted in-place before optimizer construction.
    The strategy keeps the base :class:`TrainingStrategy` loop, but prepends
    registration-time hooks derived from its convenience fields before any
    explicit ``hooks=`` supplied by the user:

    * ``module_patches`` becomes a :class:`ModulePatchHook`.
    * ``freeze_patterns`` / ``trainable_patterns`` become a
      :class:`TrainableParameterHook`.

    Module patch targets are fully-qualified paths of the form
    ``"<model_key>.<module_path>.<child>"``, for example
    ``"main.model.readouts.1.linear"``. The parent path must already exist.
    The final child is replaced when it is an existing ``torch.nn.Module`` or
    added when missing. Use :func:`nvalchemi.training.create_model_spec` for
    module patches that must round-trip through :meth:`to_spec_dict`; direct
    ``torch.nn.Module`` instances are supported at runtime but are rejected by
    serialization.

    Parameter patterns are matched against fully-qualified names such as
    ``"main.model.readouts.1.linear.weight"``. ``trainable_patterns`` alone is
    an allow-list: only matching parameters remain trainable and enter
    optimizers. When ``freeze_patterns`` is also supplied, matching parameters
    are excluded first, then ``trainable_patterns`` are re-included. With the
    default ``freeze_mode="requires_grad"``, excluded parameters are
    temporarily marked ``requires_grad=False`` during :meth:`run` and restored
    afterward. Use ``freeze_mode="optimizer_only"`` when excluded parameters
    should still receive gradients but must not be updated by optimizers.

    Parameters
    ----------
    module_patches : dict[str, BaseSpec | torch.nn.Module], optional
        Ordered module patches applied before optimizer construction.
    freeze_patterns : tuple[str, ...], optional
        Glob patterns excluded from training. Exclusions can be re-included by
        ``trainable_patterns``.
    trainable_patterns : tuple[str, ...], optional
        Glob patterns included in the trainable parameter allow-list. When no
        ``freeze_patterns`` are supplied, this is the complete allow-list.
    freeze_mode : {"requires_grad", "optimizer_only"}
        Whether excluded parameters are temporarily frozen via
        ``requires_grad=False`` or only excluded from optimizers. Defaults to
        ``"requires_grad"``.
    Attributes
    ----------
    module_patches : dict[str, BaseSpec | torch.nn.Module]
        User-declared module patches.
    freeze_patterns : tuple[str, ...]
        Parameter exclusion patterns.
    trainable_patterns : tuple[str, ...]
        Trainable parameter allow-list patterns.
    freeze_mode : {"requires_grad", "optimizer_only"}
        Parameter-freezing mode.
    Examples
    --------
    Replace a readout head, train only that head, and serialize the workflow
    by declaring the replacement as a :class:`BaseSpec`::

        import torch

        from nvalchemi.training import (
            EnergyMSELoss,
            FineTuningStrategy,
            ForceMSELoss,
            OptimizerConfig,
            create_model_spec,
            default_training_fn,
        )

        strategy = FineTuningStrategy(
            models=pretrained_model,
            module_patches={
                "main.model.readouts.1.linear": create_model_spec(
                    torch.nn.Linear,
                    in_features=128,
                    out_features=1,
                )
            },
            trainable_patterns=("main.model.readouts.1.linear.*",),
            freeze_mode="requires_grad",
            optimizer_configs=OptimizerConfig(
                optimizer_cls=torch.optim.AdamW,
                optimizer_kwargs={"lr": 1e-4},
            ),
            training_fn=default_training_fn,
            loss_fn=EnergyMSELoss() + ForceMSELoss(normalize_by_atom_count=True),
            num_epochs=10,
            devices=[torch.device("cuda")],
        )

        strategy.run(train_loader)

    Use optimizer-only filtering when excluded parameters should still receive
    gradients but must not be updated::

        strategy = FineTuningStrategy(
            models=pretrained_model,
            freeze_patterns=("main.model.*",),
            trainable_patterns=("main.model.readouts.*",),
            freeze_mode="optimizer_only",
            optimizer_configs=optimizer_config,
            training_fn=default_training_fn,
            loss_fn=loss_fn,
            num_steps=1000,
        )
    """

    module_patches: dict[str, BaseSpec | torch.nn.Module] = Field(default_factory=dict)
    freeze_patterns: tuple[str, ...] = ()
    trainable_patterns: tuple[str, ...] = ()
    freeze_mode: FreezeMode = "requires_grad"

    @model_validator(mode="before")
    @classmethod
    def _prepend_finetuning_hooks(cls, data: Any) -> Any:
        """Convert convenience fields into registration-time hooks."""
        if not isinstance(data, dict):
            return data
        normalized = dict(data)
        generated: list[Any] = []
        module_patches = normalized.get("module_patches") or {}
        if module_patches:
            generated.append(ModulePatchHook(patches=module_patches))

        freeze_patterns = tuple(normalized.get("freeze_patterns") or ())
        trainable_patterns = tuple(normalized.get("trainable_patterns") or ())
        if freeze_patterns or trainable_patterns:
            generated.append(
                TrainableParameterHook(
                    freeze_patterns=freeze_patterns,
                    trainable_patterns=trainable_patterns,
                    freeze_mode=normalized.get("freeze_mode", "requires_grad"),
                )
            )

        if generated:
            normalized["hooks"] = [*generated, *list(normalized.get("hooks") or [])]
        return normalized

    @classmethod
    def from_pretrained_checkpoint(
        cls,
        checkpoint_dir: Path | str,
        *,
        checkpoint_index: int = -1,
        map_location: str | torch.device | None = None,
        validators: Sequence[Any] | None = None,
        use_original_loss: bool = False,
        use_original_opt_class: bool = False,
        optimizer_lr: float | None = _DEFAULT_PRETRAINED_CHECKPOINT_LR,
        **strategy_kwargs: Any,
    ) -> FineTuningStrategy:
        """Start a new fine-tuning run from checkpointed model weights.

        This alternate constructor initializes a fresh
        :class:`FineTuningStrategy` from a model stored in a native nvalchemi
        checkpoint. It is intentionally different from
        :meth:`load_checkpoint`, which resumes an interrupted fine-tuning
        strategy by restoring the saved optimizer, scheduler, counters, hooks,
        and strategy configuration.

        ``from_pretrained_checkpoint`` loads the complete checkpoint model set
        as initialization. Single-model checkpoints are passed to the strategy
        as a single model; multi-model checkpoints are passed as a named model
        mapping. Source optimizer state, scheduler state, hooks, epoch/step
        limits, and runtime counters are not inherited. The new fine-tuning
        strategy starts with reset counters and applies any ``module_patches``
        or trainable-parameter filters before optimizer construction.

        By default, callers provide a new ``loss_fn`` and ``optimizer_configs``.
        Set ``use_original_loss=True`` or ``use_original_opt_class=True`` to
        fill either value from the source checkpoint metadata when the caller
        omits it. Reused optimizer configs keep the original optimizer and
        scheduler classes, but their optimizer ``lr`` is overwritten with
        ``optimizer_lr`` unless ``optimizer_lr=None`` is passed.

        Parameters
        ----------
        checkpoint_dir : Path | str
            Root directory containing a checkpoint written by
            :meth:`TrainingStrategy.save_checkpoint` or
            :class:`~nvalchemi.training.hooks.CheckpointHook`.
        checkpoint_index : int, optional
            Checkpoint index to read. ``-1`` loads the latest index recorded in
            the checkpoint manifest.
        map_location : str | torch.device | None, optional
            Device override forwarded to checkpoint loading.
        validators : Sequence[Any] | None, optional
            Optional checkpoint validators forwarded to the lower-level loader.
        use_original_loss : bool, optional
            If ``True`` and ``loss_fn`` is not supplied, rebuild the loss from
            the source strategy checkpoint metadata.
        use_original_opt_class : bool, optional
            If ``True`` and ``optimizer_configs`` is not supplied, rebuild the
            optimizer/scheduler configs from source checkpoint metadata.
        optimizer_lr : float | None, optional
            Learning rate written into reused optimizer configs. Defaults to
            ``1e-5`` for conservative fine-tuning. Pass ``None`` to preserve
            the checkpoint's serialized optimizer learning rates.
        **strategy_kwargs : Any
            Normal :class:`FineTuningStrategy` constructor arguments except
            ``models``. The loaded checkpoint model is supplied as ``models``.

        Returns
        -------
        FineTuningStrategy
            A new fine-tuning strategy initialized from checkpointed model
            weights.

        Raises
        ------
        ValueError
            If ``models`` is supplied, if no checkpoint models are loaded, or
            if requested source loss/optimizer metadata is unavailable.

        Notes
        -----
        Use :meth:`load_checkpoint` instead when the goal is to resume the
        same fine-tuning run with its saved optimizer state, scheduler state,
        hooks, counters, and training limits. Source loss and optimizer config
        reuse here is initialization-only and never restores optimizer state.
        """
        if "models" in strategy_kwargs:
            raise ValueError(
                "FineTuningStrategy.from_pretrained_checkpoint loads models "
                "from checkpoint_dir; pass fine-tuning configuration through "
                "other keyword arguments."
            )

        from nvalchemi.training._checkpoint import CheckpointManifest, load_checkpoint

        # Read the manifest first so we can request every checkpointed model
        # without constructing the saved strategy or inheriting its runtime config.
        manifest = CheckpointManifest.read(Path(checkpoint_dir))
        available_models = sorted(manifest.models)
        if not available_models:
            raise ValueError(f"Checkpoint {checkpoint_dir!s} does not contain models.")

        loaded = load_checkpoint(
            checkpoint_dir,
            checkpoint_index=checkpoint_index,
            map_location=map_location,
            model_names=set(available_models),
            validators=validators,
        )
        # Native component-only checkpoints return a manifest; strategy
        # checkpoints return a dict. Normalize both into a plain model mapping.
        if not isinstance(loaded, dict):
            loaded_models = {
                name: pair[0]
                for name in available_models
                if (pair := loaded.models.get(name)) is not None
            }
        else:
            loaded_models = {
                name: entry["model"] for name, entry in loaded.get("models", {}).items()
            }

        missing_models = set(available_models) - set(loaded_models)
        if missing_models:
            raise ValueError(
                f"Checkpoint did not load model(s) {sorted(missing_models)!r}. "
                f"Available models: {available_models!r}."
            )

        source_metadata = (
            loaded.get("strategy_metadata") if isinstance(loaded, dict) else None
        )
        _apply_checkpoint_finetuning_defaults(
            strategy_kwargs,
            source_metadata,
            use_original_loss=use_original_loss,
            use_original_opt_class=use_original_opt_class,
            optimizer_lr=optimizer_lr,
        )

        # Preserve the familiar single-model constructor UX, but keep named
        # mappings intact when the checkpoint contains multiple models.
        models = (
            next(iter(loaded_models.values()))
            if len(loaded_models) == 1
            else loaded_models
        )
        return cls(models=models, **strategy_kwargs)

    def to_spec_dict(self) -> dict[str, Any]:
        """Serialize declarative fine-tuning knobs to a JSON-ready dict.

        Returns
        -------
        dict[str, Any]
            JSON-ready bundle suitable for :func:`json.dumps`.

        Raises
        ------
        TypeError
            If ``module_patches`` contains direct ``torch.nn.Module`` values.
            Use :func:`nvalchemi.training.create_model_spec` for serializable
            module patches.
        """
        spec = super().to_spec_dict()
        if self.module_patches:
            patch_specs: dict[str, dict[str, Any]] = {}
            for target, value in self.module_patches.items():
                if not isinstance(value, BaseSpec):
                    raise TypeError(
                        "FineTuningStrategy.to_spec_dict only supports "
                        "module_patches declared as BaseSpec values; "
                        f"{target!r} is {type(value).__name__}."
                    )
                patch_specs[target] = value.model_dump()
            spec["module_patches"] = patch_specs
        spec["freeze_patterns"] = list(self.freeze_patterns)
        spec["trainable_patterns"] = list(self.trainable_patterns)
        spec["freeze_mode"] = self.freeze_mode
        return spec

    @classmethod
    def from_spec_dict(
        cls,
        spec: dict[str, Any],
        *,
        models: strategy_validation.ModelInput | None = None,
        hooks: list[Any] | None = None,
        training_fn: Any = None,
    ) -> FineTuningStrategy:
        """Rebuild a :class:`FineTuningStrategy` from ``to_spec_dict`` output.

        Parameters
        ----------
        spec : dict[str, Any]
            A dict produced by :meth:`to_spec_dict`, optionally after a JSON
            round-trip.
        models : BaseModelMixin | dict[str, BaseModelMixin] | None, optional
            Runtime model override(s).
        hooks : list[Any] | None, optional
            Runtime hooks appended after generated fine-tuning hooks.
        training_fn : Any, optional
            Runtime callable or dotted-path override.

        Returns
        -------
        FineTuningStrategy
            A freshly validated fine-tuning strategy ready to :meth:`run`.
        """
        required = ("optimizer_configs", "devices", "loss_fn_spec")
        missing = [key for key in required if key not in spec]
        if missing:
            raise ValueError(
                f"from_spec_dict: spec is missing required key(s) {missing}. "
                f"Expected keys: {list(required)}."
            )
        module_patches = {
            target: create_model_spec_from_json(raw_spec)
            for target, raw_spec in spec.get("module_patches", {}).items()
        }
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
            module_patches=module_patches,
            freeze_patterns=tuple(spec.get("freeze_patterns", ())),
            trainable_patterns=tuple(spec.get("trainable_patterns", ())),
            freeze_mode=spec.get("freeze_mode", "requires_grad"),
        )
