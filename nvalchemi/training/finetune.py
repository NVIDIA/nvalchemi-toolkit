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
    source_checkpoint : Path | str | None, optional
        Reserved for a future checkpoint-loading feature. Passing a value in
        this release raises :class:`NotImplementedError`; load models first
        and pass them through ``models=``.

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
    source_checkpoint : Path | str | None
        Reserved checkpoint source.

    Examples
    --------
    Replace a readout head, train only that head, and serialize the workflow
    by declaring the replacement as a :class:`BaseSpec`::

        import torch

        from nvalchemi.training import (
            EnergyLoss,
            FineTuningStrategy,
            ForceLoss,
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
            loss_fn=EnergyLoss() + ForceLoss(normalize_by_atom_count=True),
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
    source_checkpoint: Path | str | None = None

    @model_validator(mode="before")
    @classmethod
    def _prepend_finetuning_hooks(cls, data: Any) -> Any:
        """Convert convenience fields into registration-time hooks."""
        if not isinstance(data, dict):
            return data
        normalized = dict(data)
        if normalized.get("source_checkpoint") is not None:
            raise NotImplementedError(
                "FineTuningStrategy.source_checkpoint is reserved for the "
                "checkpoint-loading feature. Load pretrained models first "
                "and pass them via models= for now."
            )

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
